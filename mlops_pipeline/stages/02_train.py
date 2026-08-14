"""
PhishGuard AI - Stage 02: Production Model Training & Hyperparameter Evaluation
===============================================================================

Version: 14.0.0
File Path: src/stages/02_train.py

Production-grade model training stage for the PhishGuard AI pipeline.

Responsibilities
----------------
1. Load and validate processed train/validation datasets.
2. Enforce binary classification schema.
3. Detect NaN/Inf and suspicious data conditions.
4. Validate feature consistency between train and validation data.
5. Build supported classifiers through a controlled model factory.
6. Perform optional stratified cross-validation diagnostics.
7. Train the final candidate model.
8. Generate probability predictions.
9. Optimize the decision threshold on validation data.
10. Calculate comprehensive validation metrics.
11. Generate feature importance diagnostics where available.
12. Generate deterministic dataset/model fingerprints.
13. Persist model and metadata atomically.
14. Optionally integrate with MLflow.
15. Verify the saved model can be loaded and predicts correctly.
16. Produce deployment-oriented training metadata.

Security Notes
--------------
- Joblib/pickle artifacts MUST only be loaded from trusted sources.
- Model files are integrity-hashed with SHA-256.
- Input datasets are fingerprinted.
- Silent model fallback is disabled by default.
- Unexpected target leakage is explicitly rejected.
- Non-finite numeric values are rejected.
- Threshold optimization is performed only on validation data.

This module intentionally keeps training deterministic where the underlying
algorithm supports random_state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

try:
    from sklearn.utils.validation import check_is_fitted
except ImportError:  # pragma: no cover
    check_is_fitted = None


# =============================================================================
# OPTIONAL ML LIBRARIES
# =============================================================================

try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except ImportError:
    LGBMClassifier = None
    HAS_LIGHTGBM = False


try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    XGBClassifier = None
    HAS_XGBOOST = False


# =============================================================================
# OPTIONAL MLFLOW
# =============================================================================

try:
    import mlflow
    import mlflow.sklearn

    try:
        from mlflow.models import infer_signature
    except ImportError:
        infer_signature = None

    HAS_MLFLOW = True

except ImportError:
    mlflow = None
    infer_signature = None
    HAS_MLFLOW = False


# =============================================================================
# INTERNAL PROJECT IMPORTS
# =============================================================================

from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger("src.stages.02_train")


# =============================================================================
# CONSTANTS
# =============================================================================

STAGE_VERSION = "14.0.0"

SUPPORTED_MODEL_TYPES = {
    "random_forest",
    "rf",
    "logistic_regression",
    "lr",
    "lightgbm",
    "xgboost",
}

DEFAULT_RANDOM_STATE = 42

DEFAULT_THRESHOLD = 0.50

DEFAULT_THRESHOLD_METRIC = "f1"

DEFAULT_CV_FOLDS = 5

DEFAULT_CV_SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}

TARGET_POSITIVE_VALUES = {1}

TARGET_NEGATIVE_VALUES = {0}


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class DatasetSchema:
    """
    Immutable representation of the model input schema.
    """

    target_column: str
    feature_names: Tuple[str, ...]
    feature_dtypes: Tuple[Tuple[str, str], ...]
    feature_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_column": self.target_column,
            "feature_names": list(self.feature_names),
            "feature_dtypes": dict(self.feature_dtypes),
            "feature_count": self.feature_count,
        }


@dataclass(frozen=True)
class CVSummary:
    """
    Cross-validation stability summary.
    """

    enabled: bool
    folds: int
    scoring: Dict[str, str]
    metrics: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingResult:
    """
    Serializable result contract returned by the training stage.
    """

    model_path: str
    metadata_path: str
    model_sha256: str
    metadata_sha256: str
    model_type: str
    threshold: float
    validation_metrics: Dict[str, float]
    cv_summary: Dict[str, Any]
    feature_count: int
    train_samples: int
    validation_samples: int
    positive_rate_train: float
    positive_rate_validation: float
    mlflow_enabled: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# GENERAL UTILITIES
# =============================================================================


def _utc_timestamp() -> str:
    """
    Return an ISO-8601 UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """
    Convert common NumPy/Pandas values into JSON-compatible objects.
    """
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def _write_json_atomic(
    payload: Mapping[str, Any],
    destination: Union[str, Path],
) -> Path:
    """
    Atomically write JSON metadata.

    The file is first written to a temporary file in the same directory,
    flushed, and then replaced into place.
    """
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Optional[Path] = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)

            json.dump(
                _json_safe(dict(payload)),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )

            handle.flush()
            os.fsync(handle.fileno())

        temp_path.replace(destination)

    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return destination


def _sha256_file(path: Union[str, Path]) -> str:
    """
    Calculate SHA-256 digest for an artifact.
    """
    path = Path(path)

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _hash_dataframe(df: pd.DataFrame) -> str:
    """
    Produce a deterministic fingerprint for a DataFrame.

    Includes:
    - column names
    - dtypes
    - values
    """
    hasher = hashlib.sha256()

    schema_payload = json.dumps(
        {
            "columns": [str(c) for c in df.columns],
            "dtypes": [str(dtype) for dtype in df.dtypes],
        },
        sort_keys=True,
    ).encode("utf-8")

    hasher.update(schema_payload)

    hashed_values = pd.util.hash_pandas_object(
        df,
        index=True,
    ).values

    hasher.update(hashed_values.tobytes())

    return hasher.hexdigest()


def _normalize_model_type(model_type: str) -> str:
    """
    Normalize model identifier.
    """
    if not isinstance(model_type, str):
        raise TypeError("model_type must be a string.")

    key = model_type.strip().lower()

    aliases = {
        "rf": "random_forest",
        "randomforest": "random_forest",
        "random-forest": "random_forest",
        "lr": "logistic_regression",
        "logistic": "logistic_regression",
        "logistic-regression": "logistic_regression",
        "lgbm": "lightgbm",
        "xgb": "xgboost",
    }

    return aliases.get(key, key)


# =============================================================================
# DATA VALIDATION
# =============================================================================


def _validate_probability_vector(
    probabilities: np.ndarray,
    expected_length: int,
    name: str,
) -> None:
    """
    Validate probability predictions.
    """
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional probability vector."
        )

    if len(probabilities) != expected_length:
        raise ValueError(
            f"{name} length mismatch: expected {expected_length}, "
            f"received {len(probabilities)}."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    tolerance = 1e-8

    if np.any(probabilities < -tolerance) or np.any(probabilities > 1.0 + tolerance):
        raise ValueError(
            f"{name} contains probabilities outside [0, 1]."
        )


def _validate_binary_target(
    y: Union[pd.Series, np.ndarray, Sequence[Any]],
    name: str,
) -> np.ndarray:
    """
    Validate and normalize a binary target vector.

    Only 0 and 1 are accepted.
    """
    if isinstance(y, pd.Series):
        values = y.to_numpy()
    else:
        values = np.asarray(y)

    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if len(values) == 0:
        raise ValueError(f"{name} is empty.")

    if pd.isna(values).any():
        raise ValueError(f"{name} contains missing values.")

    try:
        numeric = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain only binary numeric values 0/1."
        ) from exc

    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    if not np.all(np.isin(numeric, [0.0, 1.0])):
        unique_values = np.unique(numeric).tolist()

        raise ValueError(
            f"{name} must contain only 0 and 1. "
            f"Observed values: {unique_values[:20]}"
        )

    return numeric.astype(np.int8)


def _validate_feature_matrix(
    X: pd.DataFrame,
    name: str,
) -> None:
    """
    Validate feature matrix for training.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame."
        )

    if X.empty:
        raise ValueError(f"{name} is empty.")

    if X.columns.duplicated().any():
        duplicates = (
            X.columns[X.columns.duplicated()]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            f"{name} contains duplicate feature names: {duplicates}"
        )

    if any(str(column).strip() == "" for column in X.columns):
        raise ValueError(
            f"{name} contains empty feature names."
        )

    non_numeric = X.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if non_numeric:
        raise TypeError(
            f"{name} contains non-numeric features: "
            f"{non_numeric}"
        )

    numeric_values = X.to_numpy(dtype=float)

    if not np.all(np.isfinite(numeric_values)):
        bad_columns = X.columns[
            ~np.isfinite(numeric_values).all(axis=0)
        ].astype(str).tolist()

        raise ValueError(
            f"{name} contains NaN or infinite values. "
            f"Affected columns: {bad_columns}"
        )


def _validate_train_validation_schema(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> DatasetSchema:
    """
    Enforce exact feature schema equality.
    """
    train_columns = list(X_train.columns)
    val_columns = list(X_val.columns)

    if train_columns != val_columns:
        train_set = set(train_columns)
        val_set = set(val_columns)

        missing_in_val = sorted(
            train_set - val_set,
            key=str,
        )

        extra_in_val = sorted(
            val_set - train_set,
            key=str,
        )

        raise ValueError(
            "Train/validation feature schema mismatch. "
            f"Missing in validation: {missing_in_val}; "
            f"Extra in validation: {extra_in_val}."
        )

    for column in train_columns:
        if str(X_train[column].dtype) != str(X_val[column].dtype):
            raise TypeError(
                f"Feature dtype mismatch for '{column}': "
                f"train={X_train[column].dtype}, "
                f"validation={X_val[column].dtype}"
            )

    return DatasetSchema(
        target_column="",
        feature_names=tuple(str(c) for c in train_columns),
        feature_dtypes=tuple(
            (str(c), str(X_train[c].dtype))
            for c in train_columns
        ),
        feature_count=len(train_columns),
    )


def _validate_no_target_leakage(
    X: pd.DataFrame,
    target_column: str,
) -> None:
    """
    Reject obvious target leakage.
    """
    target_lower = target_column.strip().lower()

    suspicious_names = {
        target_lower,
        "target",
        "label",
        "class",
        "y",
        "is_phishing",
        "phishing_label",
    }

    leakage_columns = [
        str(column)
        for column in X.columns
        if str(column).strip().lower() in suspicious_names
    ]

    if leakage_columns:
        raise ValueError(
            "Potential target leakage detected. "
            f"Target-like columns present in X: {leakage_columns}"
        )


def _validate_dataset_balance(
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """
    Validate that both train and validation sets contain both classes.
    """
    train_classes = np.unique(y_train)
    val_classes = np.unique(y_val)

    if len(train_classes) < 2:
        raise ValueError(
            "Training set contains only one class. "
            "Binary classification requires both classes."
        )

    if len(val_classes) < 2:
        raise ValueError(
            "Validation set contains only one class. "
            "ROC-AUC and several other metrics cannot be reliably "
            "calculated with a single validation class."
        )

    return {
        "train_class_counts": {
            str(int(cls)): int(np.sum(y_train == cls))
            for cls in train_classes
        },
        "validation_class_counts": {
            str(int(cls)): int(np.sum(y_val == cls))
            for cls in val_classes
        },
    }


# =============================================================================
# MODEL FACTORY
# =============================================================================


def create_model_instance(
    model_type: str,
    params: Optional[Mapping[str, Any]] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> BaseEstimator:
    """
    Construct a supported classifier.

    Important
    ---------
    Optional third-party dependencies are NOT silently replaced by another
    algorithm. If a configured algorithm is unavailable, the function fails
    explicitly. Silent fallback can produce an entirely different production
    model than the one requested.
    """
    model_key = _normalize_model_type(model_type)

    if model_key not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model type '{model_type}'. "
            f"Supported models: {sorted(SUPPORTED_MODEL_TYPES)}"
        )

    model_params = dict(params or {})

    if "random_state" not in model_params:
        model_params["random_state"] = random_state

    if model_key == "random_forest":
        return RandomForestClassifier(**model_params)

    if model_key == "logistic_regression":
        return LogisticRegression(**model_params)

    if model_key == "lightgbm":
        if not HAS_LIGHTGBM:
            raise ImportError(
                "LightGBM is configured but not installed. "
                "Install lightgbm or choose another model type."
            )

        return LGBMClassifier(**model_params)

    if model_key == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost is configured but not installed. "
                "Install xgboost or choose another model type."
            )

        return XGBClassifier(**model_params)

    raise RuntimeError(
        f"Unhandled model type: {model_key}"
    )


# =============================================================================
# THRESHOLD OPTIMIZATION
# =============================================================================


def _calculate_threshold_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    metric: str,
    beta: float = 1.0,
) -> float:
    """
    Calculate a threshold-dependent metric.
    """
    predictions = (y_prob >= threshold).astype(np.int8)

    metric = metric.lower().strip()

    if metric == "f1":
        return float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        )

    if metric in {"f0.5", "f0_5"}:
        return float(
            fbeta_score(
                y_true,
                predictions,
                beta=0.5,
                zero_division=0,
            )
        )

    if metric == "precision":
        return float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        )

    if metric == "recall":
        return float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        )

    if metric == "accuracy":
        return float(
            accuracy_score(
                y_true,
                predictions,
            )
        )

    raise ValueError(
        f"Unsupported threshold metric '{metric}'. "
        "Supported: f1, f0.5, precision, recall, accuracy."
    )


def find_optimal_threshold(
    y_true: Union[pd.Series, np.ndarray],
    y_probs: Union[pd.Series, np.ndarray],
    metric: str = DEFAULT_THRESHOLD_METRIC,
    num_steps: int = 199,
    minimum_threshold: float = 0.01,
    maximum_threshold: float = 0.99,
) -> Tuple[float, float]:
    """
    Find the validation threshold maximizing a selected metric.

    Notes
    -----
    Threshold tuning must only be performed on validation/calibration data.
    The untouched test set must not influence threshold selection.
    """
    y_true_arr = _validate_binary_target(
        y_true,
        "y_true",
    )

    y_prob_arr = np.asarray(
        y_probs,
        dtype=float,
    )

    _validate_probability_vector(
        y_prob_arr,
        len(y_true_arr),
        "y_probs",
    )

    if not (
        0.0 < minimum_threshold < maximum_threshold < 1.0
    ):
        raise ValueError(
            "Threshold bounds must satisfy "
            "0 < minimum_threshold < maximum_threshold < 1."
        )

    if num_steps < 2:
        raise ValueError(
            "num_steps must be >= 2."
        )

    thresholds = np.linspace(
        minimum_threshold,
        maximum_threshold,
        num_steps,
    )

    best_threshold = DEFAULT_THRESHOLD
    best_score = -np.inf

    for threshold in thresholds:
        score = _calculate_threshold_score(
            y_true=y_true_arr,
            y_prob=y_prob_arr,
            threshold=float(threshold),
            metric=metric,
        )

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return (
        float(best_threshold),
        float(best_score),
    )


# =============================================================================
# METRICS
# =============================================================================


def calculate_validation_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Calculate production-oriented validation metrics.
    """
    _validate_binary_target(
        y_true,
        "y_true",
    )

    _validate_probability_vector(
        y_prob,
        len(y_true),
        "y_prob",
    )

    predictions = (
        y_prob >= threshold
    ).astype(np.int8)

    metrics: Dict[str, float] = {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f0_5": float(
            fbeta_score(
                y_true,
                predictions,
                beta=0.5,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                y_prob,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                y_prob,
            )
        ),
        "positive_prediction_rate": float(
            np.mean(predictions)
        ),
    }

    # Confusion matrix derived values.
    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))

    metrics.update(
        {
            "true_positive": float(tp),
            "true_negative": float(tn),
            "false_positive": float(fp),
            "false_negative": float(fn),
            "false_positive_rate": float(
                fp / (fp + tn)
            )
            if (fp + tn) > 0
            else 0.0,
            "specificity": float(
                tn / (tn + fp)
            )
            if (tn + fp) > 0
            else 0.0,
            "negative_predictive_value": float(
                tn / (tn + fn)
            )
            if (tn + fn) > 0
            else 0.0,
        }
    )

    return metrics


# =============================================================================
# CROSS-VALIDATION
# =============================================================================


def run_cross_validation(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_folds: int = DEFAULT_CV_FOLDS,
    random_state: int = DEFAULT_RANDOM_STATE,
    scoring: Optional[Mapping[str, str]] = None,
) -> CVSummary:
    """
    Run stratified cross-validation diagnostics.

    Cross-validation is diagnostic here; the final model is still trained on
    the complete training dataset afterward.
    """
    if cv_folds < 2:
        raise ValueError(
            "cv_folds must be >= 2."
        )

    class_counts = np.bincount(
        y_train.astype(int),
        minlength=2,
    )

    smallest_class = int(
        np.min(class_counts)
    )

    if smallest_class < cv_folds:
        raise ValueError(
            f"Cannot perform {cv_folds}-fold stratified CV: "
            f"smallest class contains only {smallest_class} samples."
        )

    scoring_map = dict(
        scoring or DEFAULT_CV_SCORING
    )

    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    logger.info(
        "Running %d-fold stratified cross-validation.",
        cv_folds,
    )

    results = cross_validate(
        clone(model),
        X_train,
        y_train,
        cv=splitter,
        scoring=scoring_map,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )

    metric_summary: Dict[str, Dict[str, float]] = {}

    for metric_name in scoring_map:
        values = np.asarray(
            results[f"test_{metric_name}"],
            dtype=float,
        )

        metric_summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1))
            if len(values) > 1
            else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return CVSummary(
        enabled=True,
        folds=cv_folds,
        scoring=scoring_map,
        metrics=metric_summary,
    )


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================


def extract_feature_importance(
    model: BaseEstimator,
    feature_names: Sequence[str],
) -> Dict[str, float]:
    """
    Extract model-native feature importance when supported.
    """
    names = list(feature_names)

    if hasattr(model, "feature_importances_"):
        values = np.asarray(
            model.feature_importances_,
            dtype=float,
        )

        if len(values) != len(names):
            logger.warning(
                "feature_importances_ length does not match feature count."
            )

            return {}

        return {
            name: float(value)
            for name, value in zip(names, values)
        }

    if hasattr(model, "coef_"):
        coefficients = np.asarray(
            model.coef_,
            dtype=float,
        )

        if coefficients.ndim == 2:
            coefficients = coefficients[0]

        if len(coefficients) != len(names):
            return {}

        # Absolute coefficient magnitude is used as a relative importance proxy.
        values = np.abs(coefficients)

        total = float(values.sum())

        if total > 0:
            values = values / total

        return {
            name: float(value)
            for name, value in zip(names, values)
        }

    return {}


def rank_feature_importance(
    feature_importance: Mapping[str, float],
    top_n: int = 20,
) -> List[Dict[str, float]]:
    """
    Return feature importance sorted descending.
    """
    ranked = sorted(
        feature_importance.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        {
            "feature": str(name),
            "importance": float(value),
        }
        for name, value in ranked[:top_n]
    ]


# =============================================================================
# MODEL SANITY CHECK
# =============================================================================


def verify_fitted_model(
    model: BaseEstimator,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
) -> Dict[str, Any]:
    """
    Verify that the fitted model can generate valid predictions.
    """
    if check_is_fitted is not None:
        check_is_fitted(model)

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"Model '{type(model).__name__}' does not implement predict_proba()."
        )

    probabilities = model.predict_proba(
        X_validation
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "predict_proba() did not return a 2D array."
        )

    if probabilities.shape[1] < 2:
        raise ValueError(
            "Model does not expose a positive-class probability."
        )

    positive_probabilities = probabilities[:, 1]

    _validate_probability_vector(
        positive_probabilities,
        len(y_validation),
        "model validation probabilities",
    )

    predictions = (
        positive_probabilities >= DEFAULT_THRESHOLD
    ).astype(np.int8)

    return {
        "prediction_count": int(len(predictions)),
        "probability_min": float(
            np.min(positive_probabilities)
        ),
        "probability_max": float(
            np.max(positive_probabilities)
        ),
        "probability_mean": float(
            np.mean(positive_probabilities)
        ),
        "positive_prediction_rate": float(
            np.mean(predictions)
        ),
    }


# =============================================================================
# ARTIFACT PERSISTENCE
# =============================================================================


def _atomic_joblib_dump(
    model: BaseEstimator,
    destination: Union[str, Path],
) -> Path:
    """
    Atomically serialize model using joblib.
    """
    destination = Path(destination).resolve()
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path: Optional[Path] = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)

            joblib.dump(
                model,
                handle,
                compress=3,
            )

            handle.flush()
            os.fsync(handle.fileno())

        temp_path.replace(destination)

    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return destination


def verify_serialized_model(
    model_path: Union[str, Path],
    X_validation: pd.DataFrame,
    expected_probability_length: int,
) -> Dict[str, Any]:
    """
    Reload serialized model and verify inference.

    SECURITY
    --------
    Only load joblib artifacts generated by this trusted pipeline or another
    explicitly trusted source. joblib/pickle deserialization is not a safe
    format for untrusted files.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Serialized model does not exist: {model_path}"
        )

    loaded_model = joblib.load(
        model_path
    )

    if not hasattr(loaded_model, "predict_proba"):
        raise ValueError(
            "Serialized model does not expose predict_proba()."
        )

    probabilities = loaded_model.predict_proba(
        X_validation
    )

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "Reloaded model returned invalid probability shape."
        )

    positive_probabilities = probabilities[:, 1]

    _validate_probability_vector(
        positive_probabilities,
        expected_probability_length,
        "reloaded model probabilities",
    )

    return {
        "reload_successful": True,
        "model_class": type(loaded_model).__name__,
        "prediction_count": int(
            len(positive_probabilities)
        ),
    }


# =============================================================================
# MLFLOW
# =============================================================================


class _MLflowRunManager:
    """
    Safe MLflow lifecycle wrapper.

    Prevents accidentally leaving an active run open when training raises.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        enabled: bool,
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.enabled = bool(
            enabled and HAS_MLFLOW
        )
        self.started = False

    def __enter__(self) -> "_MLflowRunManager":
        if not self.enabled:
            return self

        try:
            mlflow.set_experiment(
                self.experiment_name
            )

            mlflow.start_run(
                run_name=self.run_name
            )

            self.started = True

        except Exception as exc:
            logger.warning(
                "MLflow initialization failed. "
                "Continuing without MLflow: %s",
                exc,
            )

            self.enabled = False

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        if self.started:
            try:
                mlflow.end_run()
            except Exception:
                logger.exception(
                    "Failed to close MLflow run."
                )


def _mlflow_log_safe(
    params: Mapping[str, Any],
    metrics: Mapping[str, float],
) -> None:
    """
    Safely log parameters and metrics to MLflow.
    """
    if not HAS_MLFLOW:
        return

    try:
        safe_params = {}

        for key, value in params.items():
            serialized = str(value)

            if len(serialized) > 500:
                serialized = serialized[:500]

            safe_params[str(key)] = serialized

        mlflow.log_params(
            safe_params
        )

        safe_metrics = {}

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if math.isfinite(float(value)):
                    safe_metrics[str(key)] = float(value)

        if safe_metrics:
            mlflow.log_metrics(
                safe_metrics
            )

    except Exception as exc:
        logger.warning(
            "MLflow parameter/metric logging failed: %s",
            exc,
        )


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================


def train_candidate_model(
    train_path: Union[str, Path],
    val_path: Union[str, Path],
    output_model_dir: Union[str, Path],
    target_column: str = "label",
    model_type: str = "random_forest",
    model_params: Optional[Dict[str, Any]] = None,
    experiment_name: str = "phishguard_training",
    random_state: int = DEFAULT_RANDOM_STATE,
    threshold_metric: str = DEFAULT_THRESHOLD_METRIC,
    threshold_steps: int = 199,
    cv_folds: int = DEFAULT_CV_FOLDS,
    enable_cv: bool = True,
    enable_mlflow: bool = True,
    verify_artifact: bool = True,
    top_features: int = 20,
) -> Dict[str, Any]:
    """
    Execute the complete production training workflow.
    """
    train_path = Path(
        train_path
    ).resolve()

    val_path = Path(
        val_path
    ).resolve()

    output_dir = Path(
        output_model_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}"
        )

    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {val_path}"
        )

    if train_path == val_path:
        raise ValueError(
            "Training and validation datasets must be different files."
        )

    logger.info(
        "Starting PhishGuard AI Stage 02 training. "
        "Version=%s, Model=%s",
        STAGE_VERSION,
        model_type,
    )

    # -------------------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------------------

    logger.info(
        "Loading training dataset: %s",
        train_path,
    )

    train_df = pd.read_csv(
        train_path
    )

    logger.info(
        "Loading validation dataset: %s",
        val_path,
    )

    val_df = pd.read_csv(
        val_path
    )

    if target_column not in train_df.columns:
        raise KeyError(
            f"Target column '{target_column}' missing from training dataset."
        )

    if target_column not in val_df.columns:
        raise KeyError(
            f"Target column '{target_column}' missing from validation dataset."
        )

    if len(train_df) == 0:
        raise ValueError(
            "Training dataset is empty."
        )

    if len(val_df) == 0:
        raise ValueError(
            "Validation dataset is empty."
        )

    # -------------------------------------------------------------------------
    # Split X / y
    # -------------------------------------------------------------------------

    X_train = train_df.drop(
        columns=[target_column]
    ).copy()

    y_train = _validate_binary_target(
        train_df[target_column],
        "training target",
    )

    X_val = val_df.drop(
        columns=[target_column]
    ).copy()

    y_val = _validate_binary_target(
        val_df[target_column],
        "validation target",
    )

    # -------------------------------------------------------------------------
    # Feature validation
    # -------------------------------------------------------------------------

    _validate_feature_matrix(
        X_train,
        "training feature matrix",
    )

    _validate_feature_matrix(
        X_val,
        "validation feature matrix",
    )

    _validate_no_target_leakage(
        X_train,
        target_column,
    )

    _validate_no_target_leakage(
        X_val,
        target_column,
    )

    schema = _validate_train_validation_schema(
        X_train,
        X_val,
    )

    schema = DatasetSchema(
        target_column=target_column,
        feature_names=schema.feature_names,
        feature_dtypes=schema.feature_dtypes,
        feature_count=schema.feature_count,
    )

    # -------------------------------------------------------------------------
    # Dataset balance
    # -------------------------------------------------------------------------

    balance_info = _validate_dataset_balance(
        y_train,
        y_val,
    )

    train_positive_rate = float(
        np.mean(y_train)
    )

    val_positive_rate = float(
        np.mean(y_val)
    )

    logger.info(
        "Dataset summary: train=%d, validation=%d, "
        "features=%d, train_positive_rate=%.4f, "
        "validation_positive_rate=%.4f",
        len(X_train),
        len(X_val),
        X_train.shape[1],
        train_positive_rate,
        val_positive_rate,
    )

    # -------------------------------------------------------------------------
    # Dataset fingerprints
    # -------------------------------------------------------------------------

    train_fingerprint = _hash_dataframe(
        train_df
    )

    val_fingerprint = _hash_dataframe(
        val_df
    )

    # -------------------------------------------------------------------------
    # Model construction
    # -------------------------------------------------------------------------

    normalized_model_type = _normalize_model_type(
        model_type
    )

    params = dict(
        model_params or {}
    )

    # Prevent configuration from accidentally losing reproducibility.
    if "random_state" not in params:
        params["random_state"] = random_state

    model = create_model_instance(
        normalized_model_type,
        params,
        random_state=random_state,
    )

    logger.info(
        "Initialized model: %s",
        type(model).__name__,
    )

    # -------------------------------------------------------------------------
    # MLflow lifecycle
    # -------------------------------------------------------------------------

    with _MLflowRunManager(
        experiment_name=experiment_name,
        run_name=f"{normalized_model_type}_candidate",
        enabled=enable_mlflow,
    ) as mlflow_run:

        mlflow_params = {
            "stage_version": STAGE_VERSION,
            "model_type": normalized_model_type,
            "feature_count": X_train.shape[1],
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "random_state": random_state,
            "threshold_metric": threshold_metric,
            "threshold_steps": threshold_steps,
            "cv_folds": cv_folds,
        }

        mlflow_params.update(
            {
                f"model_param_{key}": value
                for key, value in params.items()
            }
        )

        if mlflow_run.enabled:
            _mlflow_log_safe(
                mlflow_params,
                {},
            )

        # ---------------------------------------------------------------------
        # Cross-validation
        # ---------------------------------------------------------------------

        if enable_cv:
            cv_summary = run_cross_validation(
                model=model,
                X_train=X_train,
                y_train=y_train,
                cv_folds=cv_folds,
                random_state=random_state,
            )
        else:
            cv_summary = CVSummary(
                enabled=False,
                folds=0,
                scoring={},
                metrics={},
            )

        # Log CV mean metrics.
        cv_mlflow_metrics: Dict[str, float] = {}

        for metric_name, summary in cv_summary.metrics.items():
            cv_mlflow_metrics[
                f"cv_{metric_name}_mean"
            ] = summary["mean"]

            cv_mlflow_metrics[
                f"cv_{metric_name}_std"
            ] = summary["std"]

        if mlflow_run.enabled:
            _mlflow_log_safe(
                {},
                cv_mlflow_metrics,
            )

        # ---------------------------------------------------------------------
        # Final model fitting
        # ---------------------------------------------------------------------

        logger.info(
            "Training final candidate model on complete training dataset."
        )

        model.fit(
            X_train,
            y_train,
        )

        # ---------------------------------------------------------------------
        # Model sanity verification
        # ---------------------------------------------------------------------

        sanity_info = verify_fitted_model(
            model=model,
            X_validation=X_val,
            y_validation=y_val,
        )

        logger.info(
            "Model sanity check passed. "
            "Probability range=[%.6f, %.6f]",
            sanity_info["probability_min"],
            sanity_info["probability_max"],
        )

        # ---------------------------------------------------------------------
        # Validation probabilities
        # ---------------------------------------------------------------------

        validation_probability_matrix = model.predict_proba(
            X_val
        )

        if validation_probability_matrix.ndim != 2:
            raise ValueError(
                "Model predict_proba() returned invalid dimensionality."
            )

        if validation_probability_matrix.shape[1] < 2:
            raise ValueError(
                "Model does not provide a positive class probability."
            )

        val_probs = np.asarray(
            validation_probability_matrix[:, 1],
            dtype=float,
        )

        _validate_probability_vector(
            val_probs,
            len(y_val),
            "validation probabilities",
        )

        # ---------------------------------------------------------------------
        # Threshold optimization
        # ---------------------------------------------------------------------

        optimal_threshold, threshold_score = find_optimal_threshold(
            y_true=y_val,
            y_probs=val_probs,
            metric=threshold_metric,
            num_steps=threshold_steps,
        )

        logger.info(
            "Optimal threshold=%.6f, %s=%.6f",
            optimal_threshold,
            threshold_metric,
            threshold_score,
        )

        # ---------------------------------------------------------------------
        # Validation metrics
        # ---------------------------------------------------------------------

        validation_metrics = calculate_validation_metrics(
            y_true=y_val,
            y_prob=val_probs,
            threshold=optimal_threshold,
        )

        validation_metrics[
            "threshold_optimization_score"
        ] = float(threshold_score)

        validation_metrics[
            "decision_threshold"
        ] = float(optimal_threshold)

        # ---------------------------------------------------------------------
        # Feature importance
        # ---------------------------------------------------------------------

        feature_importance = extract_feature_importance(
            model=model,
            feature_names=schema.feature_names,
        )

        ranked_features = rank_feature_importance(
            feature_importance,
            top_n=top_features,
        )

        if ranked_features:
            logger.info(
                "Top model features: %s",
                ranked_features[:10],
            )

        # ---------------------------------------------------------------------
        # Model metadata
        # ---------------------------------------------------------------------

        metadata: Dict[str, Any] = {
            "artifact_type": "phishguard_candidate_classifier",
            "stage": "02_train",
            "stage_version": STAGE_VERSION,
            "created_at_utc": _utc_timestamp(),
            "model_type": normalized_model_type,
            "model_class": type(model).__name__,
            "hyperparameters": params,
            "random_state": random_state,
            "threshold": {
                "value": optimal_threshold,
                "metric": threshold_metric,
                "validation_score": threshold_score,
                "search_steps": threshold_steps,
                "search_range": [
                    0.01,
                    0.99,
                ],
            },
            "validation_metrics": validation_metrics,
            "cross_validation": cv_summary.to_dict(),
            "feature_schema": schema.to_dict(),
            "feature_importance": feature_importance,
            "top_features": ranked_features,
            "dataset": {
                "train_path": str(train_path),
                "validation_path": str(val_path),
                "train_samples": len(train_df),
                "validation_samples": len(val_df),
                "train_fingerprint_sha256": train_fingerprint,
                "validation_fingerprint_sha256": val_fingerprint,
                "train_positive_rate": train_positive_rate,
                "validation_positive_rate": val_positive_rate,
                "balance": balance_info,
            },
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "mlflow_available": HAS_MLFLOW,
                "lightgbm_available": HAS_LIGHTGBM,
                "xgboost_available": HAS_XGBOOST,
            },
            "sanity_check": sanity_info,
        }

        # ---------------------------------------------------------------------
        # Persist candidate model
        # ---------------------------------------------------------------------

        model_file = (
            output_dir / "candidate_model.joblib"
        )

        metadata_file = (
            output_dir / "candidate_metadata.json"
        )

        logger.info(
            "Serializing candidate model to: %s",
            model_file,
        )

        _atomic_joblib_dump(
            model=model,
            destination=model_file,
        )

        model_sha256 = _sha256_file(
            model_file
        )

        metadata[
            "artifact_integrity"
        ] = {
            "model_sha256": model_sha256,
            "model_size_bytes": model_file.stat().st_size,
        }

        # ---------------------------------------------------------------------
        # Verify serialized artifact
        # ---------------------------------------------------------------------

        if verify_artifact:
            verification = verify_serialized_model(
                model_path=model_file,
                X_validation=X_val,
                expected_probability_length=len(y_val),
            )

            metadata[
                "artifact_verification"
            ] = verification

            logger.info(
                "Serialized model verification passed."
            )

        # ---------------------------------------------------------------------
        # Persist metadata
        # ---------------------------------------------------------------------

        _write_json_atomic(
            payload=metadata,
            destination=metadata_file,
        )

        metadata_sha256 = _sha256_file(
            metadata_file
        )

        # ---------------------------------------------------------------------
        # MLflow model logging
        # ---------------------------------------------------------------------

        if mlflow_run.enabled:
            try:
                if infer_signature is not None:
                    signature = infer_signature(
                        X_val,
                        model.predict(X_val),
                    )
                else:
                    signature = None

                if signature is not None:
                    mlflow.sklearn.log_model(
                        model,
                        name="candidate_model",
                        signature=signature,
                        input_example=X_val.head(3),
                    )
                else:
                    mlflow.sklearn.log_model(
                        model,
                        name="candidate_model",
                    )

            except Exception as exc:
                logger.warning(
                    "MLflow model logging failed: %s",
                    exc,
                )

            _mlflow_log_safe(
                {
                    "model_sha256": model_sha256,
                    "metadata_sha256": metadata_sha256,
                },
                validation_metrics,
            )

        # ---------------------------------------------------------------------
        # Training completion
        # ---------------------------------------------------------------------

        logger.info(
            "Stage 02 training completed successfully. "
            "F1=%.4f, Recall=%.4f, Precision=%.4f, ROC-AUC=%.4f, "
            "Threshold=%.4f",
            validation_metrics["f1"],
            validation_metrics["recall"],
            validation_metrics["precision"],
            validation_metrics["roc_auc"],
            optimal_threshold,
        )

    # -------------------------------------------------------------------------
    # Final result
    # -------------------------------------------------------------------------

    result = TrainingResult(
        model_path=str(model_file),
        metadata_path=str(metadata_file),
        model_sha256=model_sha256,
        metadata_sha256=metadata_sha256,
        model_type=normalized_model_type,
        threshold=float(optimal_threshold),
        validation_metrics=validation_metrics,
        cv_summary=cv_summary.to_dict(),
        feature_count=int(X_train.shape[1]),
        train_samples=int(len(X_train)),
        validation_samples=int(len(X_val)),
        positive_rate_train=train_positive_rate,
        positive_rate_validation=val_positive_rate,
        mlflow_enabled=bool(
            enable_mlflow and HAS_MLFLOW
        ),
    )

    return result.to_dict()


# =============================================================================
# HIGH-LEVEL MODEL TRAINING ALIAS
# =============================================================================


def train_model(
    train_path: Union[str, Path],
    val_path: Union[str, Path],
    output_model_dir: Union[str, Path],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Compatibility wrapper around train_candidate_model.

    This provides a concise API for external orchestrators.
    """
    return train_candidate_model(
        train_path=train_path,
        val_path=val_path,
        output_model_dir=output_model_dir,
        **kwargs,
    )


# =============================================================================
# CONFIGURATION ENTRYPOINT
# =============================================================================


def main(
    config_path: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Main execution entrypoint for Stage 02.

    Configuration example:

    training:
      model_type: random_forest

      hyperparameters:
        n_estimators: 300
        max_depth: 18
        min_samples_leaf: 2
        n_jobs: -1

      threshold_metric: f1
      threshold_steps: 199

      cv:
        enabled: true
        folds: 5

      mlflow:
        enabled: true
        experiment_name: phishguard_training

      output_model_dir: models/candidate
    """
    config = (
        load_config(config_path)
        if config_path
        else {}
    )

    training_cfg = config.get(
        "training",
        {},
    )

    data_cfg = config.get(
        "data_preparation",
        {},
    )

    main_cfg = config.get(
        "main",
        {},
    )

    random_state = int(
        kwargs.get(
            "random_state",
            main_cfg.get(
                "random_seed",
                DEFAULT_RANDOM_STATE,
            ),
        )
    )

    train_path = kwargs.get(
        "train_path"
    ) or Path(
        data_cfg.get(
            "output_dir",
            "data/processed",
        )
    ) / "train.csv"

    val_path = kwargs.get(
        "val_path"
    ) or Path(
        data_cfg.get(
            "output_dir",
            "data/processed",
        )
    ) / "val.csv"

    output_model_dir = kwargs.get(
        "output_model_dir"
    ) or training_cfg.get(
        "output_model_dir",
        "models/candidate",
    )

    target_column = kwargs.get(
        "target_column"
    ) or data_cfg.get(
        "target_column",
        "label",
    )

    model_type = kwargs.get(
        "model_type"
    ) or training_cfg.get(
        "model_type",
        "random_forest",
    )

    model_params = kwargs.get(
        "model_params"
    )

    if model_params is None:
        model_params = dict(
            training_cfg.get(
                "hyperparameters",
                {},
            )
        )

    threshold_metric = kwargs.get(
        "threshold_metric"
    ) or training_cfg.get(
        "threshold_metric",
        DEFAULT_THRESHOLD_METRIC,
    )

    threshold_steps = int(
        kwargs.get(
            "threshold_steps",
            training_cfg.get(
                "threshold_steps",
                199,
            ),
        )
    )

    cv_cfg = training_cfg.get(
        "cv",
        {},
    )

    enable_cv = bool(
        kwargs.get(
            "enable_cv",
            cv_cfg.get(
                "enabled",
                True,
            ),
        )
    )

    cv_folds = int(
        kwargs.get(
            "cv_folds",
            cv_cfg.get(
                "folds",
                DEFAULT_CV_FOLDS,
            ),
        )
    )

    mlflow_cfg = training_cfg.get(
        "mlflow",
        {},
    )

    if isinstance(
        mlflow_cfg,
        bool,
    ):
        enable_mlflow = mlflow_cfg
        experiment_name = "phishguard_training"
    else:
        enable_mlflow = bool(
            kwargs.get(
                "enable_mlflow",
                mlflow_cfg.get(
                    "enabled",
                    True,
                ),
            )
        )

        experiment_name = str(
            mlflow_cfg.get(
                "experiment_name",
                training_cfg.get(
                    "experiment_name",
                    "phishguard_training",
                ),
            )
        )

    verify_artifact = bool(
        kwargs.get(
            "verify_artifact",
            training_cfg.get(
                "verify_artifact",
                True,
            ),
        )
    )

    top_features = int(
        kwargs.get(
            "top_features",
            training_cfg.get(
                "top_features",
                20,
            ),
        )
    )

    return train_candidate_model(
        train_path=train_path,
        val_path=val_path,
        output_model_dir=output_model_dir,
        target_column=target_column,
        model_type=model_type,
        model_params=model_params,
        experiment_name=experiment_name,
        random_state=random_state,
        threshold_metric=threshold_metric,
        threshold_steps=threshold_steps,
        cv_folds=cv_folds,
        enable_cv=enable_cv,
        enable_mlflow=enable_mlflow,
        verify_artifact=verify_artifact,
        top_features=top_features,
    )


# =============================================================================
# CLI
# =============================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Construct CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "PhishGuard AI Stage 02 - "
            "Production Model Training"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="params.yaml",
        help="Path to pipeline configuration file.",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Override configured model type.",
    )

    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Disable cross-validation diagnostics.",
    )

    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking.",
    )

    parser.add_argument(
        "--no-artifact-verification",
        action="store_true",
        help="Skip serialized model verification.",
    )

    return parser


def cli() -> int:
    """
    CLI execution wrapper.
    """
    parser = build_argument_parser()

    args = parser.parse_args()

    try:
        result = main(
            config_path=args.config,
            model_type=args.model_type,
            enable_cv=not args.no_cv,
            enable_mlflow=not args.no_mlflow,
            verify_artifact=not args.no_artifact_verification,
        )

        logger.info(
            "Training result:\n%s",
            json.dumps(
                _json_safe(result),
                indent=2,
            ),
        )

        return 0

    except Exception:
        logger.exception(
            "Stage 02 execution failed."
        )

        return 1


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    "DatasetSchema",
    "CVSummary",
    "TrainingResult",
    "create_model_instance",
    "find_optimal_threshold",
    "calculate_validation_metrics",
    "run_cross_validation",
    "extract_feature_importance",
    "rank_feature_importance",
    "train_candidate_model",
    "train_model",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(
        cli()
    )
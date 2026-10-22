"""
PhishGuard AI - Stage 03: Global Evaluation, Fairness Audit & Regression Gate
=============================================================================

Version: 14.0.0
File Path: src/stages/03_evaluate.py

Production-grade evaluation/deployment gate for the PhishGuard AI MLOps pipeline.

Responsibilities
----------------
1. Load and validate the immutable test dataset.
2. Load and validate the candidate model.
3. Validate model/data feature-schema compatibility.
4. Evaluate global classification performance.
5. Perform subgroup/slice fairness auditing.
6. Compare challenger against the champion model.
7. Optionally perform calibration and adversarial robustness checks.
8. Apply independent deployment gates.
9. Generate a deterministic, auditable evaluation report.
10. Emit CI/CD-friendly metrics and a manifest with artifact hashes.

Security / reliability principles
----------------------------------
- Fail closed on malformed evaluation inputs.
- Never silently drop required model features.
- Never infer that a missing champion is a successful regression test.
- Validate probabilities before using them.
- Keep the test set untouched by threshold optimization.
- Never mutate the input test dataframe.
- Treat evaluation artifacts as audit evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.evaluation.fairness import evaluate_slice_fairness
from src.evaluation.metrics import calculate_classification_metrics
from src.evaluation.regression import compare_champion_vs_challenger
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("src.stages.03_evaluate")


# ==============================================================================
# CONSTANTS
# ==============================================================================

STAGE_VERSION = "14.0.0"

DEFAULT_TARGET_COLUMN = "label"
DEFAULT_THRESHOLD = 0.50
DEFAULT_MIN_PARITY_RATIO = 0.80

DEFAULT_MIN_TEST_SAMPLES = 100
DEFAULT_MIN_CLASS_SAMPLES = 10

DEFAULT_SLICE_COLUMNS = [
    "tld_extension",
    "organization_id",
]

SUPPORTED_TARGET_VALUES = {0, 1}

GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_SKIP = "SKIP"
GATE_WARN = "WARN"

VERDICT_PROMOTE = "PROMOTE_CHALLENGER"
VERDICT_INITIAL_BASELINE = "PROMOTE_INITIAL_BASELINE"

VERDICT_REJECT_FAIRNESS = "REJECT_FAIRNESS"
VERDICT_REJECT_REGRESSION = "REJECT_REGRESSION"
VERDICT_REJECT_CALIBRATION = "REJECT_CALIBRATION"
VERDICT_REJECT_ADVERSARIAL = "REJECT_ADVERSARIAL"
VERDICT_REJECT_DATA = "REJECT_DATA"
VERDICT_REJECT_SCHEMA = "REJECT_SCHEMA"

VERDICT_NEUTRAL = "NEUTRAL"


# ==============================================================================
# GENERAL UTILITIES
# ==============================================================================

def _utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """
    Convert common NumPy/Pandas/path objects into JSON-safe values.

    This prevents evaluation reporting from failing because of NumPy scalar
    objects or pathlib instances.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value_float = float(value)
        return value_float if math.isfinite(value_float) else None

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return value


def _atomic_write_json(
    path: Union[str, Path],
    payload: Mapping[str, Any],
) -> Path:
    """
    Atomically write JSON to disk.

    A temporary file is written first and then replaced. This prevents partially
    written reports from being consumed by CI/CD jobs.
    """
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
        text=True,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(payload),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")

        os.replace(temporary_name, destination)
        return destination

    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _sha256_file(path: Union[str, Path]) -> str:
    """Calculate SHA-256 digest of a file."""
    file_path = Path(path).resolve()

    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _require_existing_file(
    path: Union[str, Path],
    description: str,
) -> Path:
    """Resolve and validate an existing regular file."""
    resolved = Path(path).resolve()

    if not resolved.exists():
        raise FileNotFoundError(
            f"{description} does not exist: '{resolved}'"
        )

    if not resolved.is_file():
        raise ValueError(
            f"{description} is not a regular file: '{resolved}'"
        )

    return resolved


def _validate_probability_vector(
    probabilities: Any,
    expected_length: int,
    name: str,
) -> np.ndarray:
    """
    Validate model probability output.

    Prevents NaN/Inf/out-of-range probabilities from entering evaluation logic.
    """
    array = np.asarray(probabilities, dtype=float).reshape(-1)

    if len(array) != expected_length:
        raise ValueError(
            f"{name} probability vector length mismatch: "
            f"expected {expected_length}, received {len(array)}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} probabilities contain NaN or infinite values."
        )

    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(
            f"{name} probabilities contain values outside [0, 1]."
        )

    return array


def _predict_positive_probability(
    model: Any,
    X: pd.DataFrame,
    model_name: str,
) -> np.ndarray:
    """
    Safely obtain positive-class probabilities.

    Supports conventional sklearn classifiers exposing predict_proba.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"{model_name} does not expose predict_proba(). "
            "Probability-based evaluation requires a probabilistic classifier."
        )

    probabilities = model.predict_proba(X)

    array = np.asarray(probabilities)

    if array.ndim != 2:
        raise ValueError(
            f"{model_name}.predict_proba() returned shape {array.shape}; "
            "expected a 2D probability matrix."
        )

    if array.shape[0] != len(X):
        raise ValueError(
            f"{model_name}.predict_proba() returned {array.shape[0]} rows "
            f"for {len(X)} input samples."
        )

    if array.shape[1] != 2:
        raise ValueError(
            f"{model_name} is expected to be a binary classifier, but "
            f"predict_proba() returned {array.shape[1]} classes."
        )

    # sklearn normally orders binary classes as [0, 1].
    classes = getattr(model, "classes_", None)

    if classes is not None:
        classes_array = np.asarray(classes)

        if len(classes_array) != 2:
            raise ValueError(
                f"{model_name} exposes {len(classes_array)} classes; "
                "binary PhishGuard evaluation requires exactly 2."
            )

        positive_positions = np.where(classes_array == 1)[0]

        if len(positive_positions) != 1:
            raise ValueError(
                f"{model_name} does not expose a valid positive class '1'. "
                f"Detected classes: {classes_array.tolist()}"
            )

        positive_index = int(positive_positions[0])

    else:
        positive_index = 1

    return _validate_probability_vector(
        array[:, positive_index],
        expected_length=len(X),
        name=model_name,
    )


def _validate_threshold(threshold: float) -> float:
    """Validate classification decision threshold."""
    threshold_float = float(threshold)

    if not math.isfinite(threshold_float):
        raise ValueError("Decision threshold must be finite.")

    if threshold_float <= 0.0 or threshold_float >= 1.0:
        raise ValueError(
            f"Decision threshold must be strictly between 0 and 1; "
            f"received {threshold_float}."
        )

    return threshold_float


# ==============================================================================
# DATA VALIDATION
# ==============================================================================

def _validate_target(
    y: Union[pd.Series, np.ndarray],
    target_column: str,
    min_samples: int,
    min_class_samples: int,
) -> Dict[str, Any]:
    """
    Validate binary evaluation labels.

    The deployment gate fails closed if the evaluation set is too small or
    contains insufficient examples of either class.
    """
    series = pd.Series(y)

    if series.empty:
        raise ValueError("Evaluation target is empty.")

    if series.isna().any():
        raise ValueError(
            f"Target column '{target_column}' contains missing values."
        )

    try:
        numeric = pd.to_numeric(series, errors="raise")
    except Exception as exc:
        raise ValueError(
            f"Target column '{target_column}' must contain binary numeric labels."
        ) from exc

    values = set(numeric.astype(int).unique().tolist())

    if not values.issubset(SUPPORTED_TARGET_VALUES):
        raise ValueError(
            f"Target column '{target_column}' contains unsupported labels "
            f"{sorted(values)}. Expected only 0 and 1."
        )

    if len(values) != 2:
        raise ValueError(
            f"Evaluation dataset must contain both classes 0 and 1. "
            f"Detected: {sorted(values)}."
        )

    sample_count = len(numeric)
    negative_count = int((numeric == 0).sum())
    positive_count = int((numeric == 1).sum())

    if sample_count < min_samples:
        raise ValueError(
            f"Evaluation dataset contains only {sample_count} samples; "
            f"minimum required is {min_samples}."
        )

    if negative_count < min_class_samples:
        raise ValueError(
            f"Only {negative_count} negative samples are available; "
            f"minimum required is {min_class_samples}."
        )

    if positive_count < min_class_samples:
        raise ValueError(
            f"Only {positive_count} positive samples are available; "
            f"minimum required is {min_class_samples}."
        )

    return {
        "sample_count": sample_count,
        "negative_count": negative_count,
        "positive_count": positive_count,
        "positive_ratio": float(positive_count / sample_count),
    }


def _validate_test_dataframe(
    test_df: pd.DataFrame,
    target_column: str,
) -> None:
    """Validate fundamental test dataframe properties."""
    if test_df.empty:
        raise ValueError("Test dataframe is empty.")

    if test_df.columns.duplicated().any():
        duplicated = (
            test_df.columns[test_df.columns.duplicated()]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"Test dataset contains duplicate column names: {duplicated}"
        )

    if target_column not in test_df.columns:
        raise KeyError(
            f"Target column '{target_column}' was not found in test dataset."
        )


# ==============================================================================
# FEATURE SCHEMA VALIDATION
# ==============================================================================

def _extract_expected_features(model: Any) -> Optional[List[str]]:
    """
    Extract feature names recorded by common sklearn estimators/pipelines.

    Returns None when the estimator does not expose feature names.
    """
    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is None:
        return None

    return [str(name) for name in feature_names]


def _validate_feature_schema(
    model: Any,
    X_test: pd.DataFrame,
    model_name: str,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Verify compatibility between the model's training feature schema and X_test.

    This is one of the most important protections in this stage. The previous
    implementation used select_dtypes(), which could silently remove features
    and produce an evaluation that was technically executable but scientifically
    invalid.
    """
    expected = _extract_expected_features(model)

    if expected is None:
        logger.warning(
            "%s does not expose feature_names_in_; schema validation is limited.",
            model_name,
        )

        return {
            "validated": False,
            "mode": "limited",
            "expected_feature_count": None,
            "actual_feature_count": len(X_test.columns),
            "missing_features": [],
            "unexpected_features": [],
        }

    actual = [str(column) for column in X_test.columns]

    expected_set = set(expected)
    actual_set = set(actual)

    missing = [feature for feature in expected if feature not in actual_set]
    unexpected = [feature for feature in actual if feature not in expected_set]

    if missing:
        raise ValueError(
            f"{model_name} feature-schema mismatch. Missing required features: "
            f"{missing}"
        )

    if strict and unexpected:
        raise ValueError(
            f"{model_name} feature-schema mismatch. Unexpected features: "
            f"{unexpected}"
        )

    # Reorder exactly as used during training.
    X_test[:] = X_test.loc[:, expected].values

    return {
        "validated": not bool(unexpected),
        "mode": "strict" if strict else "compatible",
        "expected_feature_count": len(expected),
        "actual_feature_count": len(actual),
        "missing_features": missing,
        "unexpected_features": unexpected,
        "feature_order": expected,
    }


# ==============================================================================
# MODEL ARTIFACT VALIDATION
# ==============================================================================

def _load_model(
    model_path: Union[str, Path],
    model_name: str,
) -> Tuple[Any, Path]:
    """Load and minimally validate a serialized model artifact."""
    path = _require_existing_file(model_path, f"{model_name} model")

    try:
        model = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to deserialize {model_name} model from '{path}'."
        ) from exc

    if model is None:
        raise ValueError(f"{model_name} model artifact resolved to None.")

    if not hasattr(model, "predict"):
        raise TypeError(
            f"{model_name} model does not implement predict()."
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"{model_name} model does not implement predict_proba()."
        )

    return model, path


def _load_model_metadata(
    model_path: Path,
) -> Optional[Dict[str, Any]]:
    """
    Attempt to load adjacent model metadata.

    Metadata is optional for compatibility with older artifacts.
    """
    metadata_path = model_path.with_name(
        f"{model_path.stem.replace('_model', '')}_metadata.json"
    )

    if not metadata_path.exists():
        # Existing Stage 02 naming convention:
        # candidate_model.joblib -> candidate_metadata.json
        if model_path.name == "candidate_model.joblib":
            metadata_path = model_path.with_name("candidate_metadata.json")
        elif model_path.name == "champion_model.joblib":
            metadata_path = model_path.with_name("champion_metadata.json")

    if not metadata_path.exists():
        return None

    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            logger.warning(
                "Model metadata '%s' is not a JSON object.",
                metadata_path,
            )
            return None

        return payload

    except Exception as exc:
        logger.warning(
            "Unable to load model metadata '%s': %s",
            metadata_path,
            exc,
        )
        return None


# ==============================================================================
# FAIRNESS DATA PREPARATION
# ==============================================================================

def _prepare_features_and_slices(
    test_df: pd.DataFrame,
    target_column: str,
    slice_columns: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Prepare X and slice metadata without deleting slice columns from X.

    Slice columns are metadata unless they are actually model features.
    """
    slices: Dict[str, pd.Series] = {}

    for column in slice_columns or []:
        if column not in test_df.columns:
            logger.warning(
                "Configured fairness slice '%s' is absent from test data.",
                column,
            )
            continue

        series = test_df[column]

        if series.isna().all():
            logger.warning(
                "Fairness slice '%s' contains only missing values.",
                column,
            )
            continue

        slices[column] = series.copy()

    X = test_df.drop(columns=[target_column]).copy()

    return X, slices


# ==============================================================================
# ADDITIONAL GLOBAL DIAGNOSTICS
# ==============================================================================

def _calculate_operational_diagnostics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Calculate threshold-level operational diagnostics.

    This supplements the project's central metrics module without duplicating
    the main metric calculation contract.
    """
    predictions = (probabilities >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))

    total = len(y_true)

    return {
        "threshold": float(threshold),
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
        "predicted_positive_count": int(predictions.sum()),
        "predicted_positive_rate": float(predictions.mean()),
        "actual_positive_rate": float(y_true.mean()),
        "prediction_disagreement_baseline": None,
        "probability_statistics": {
            "min": float(np.min(probabilities)),
            "max": float(np.max(probabilities)),
            "mean": float(np.mean(probabilities)),
            "median": float(np.median(probabilities)),
            "std": float(np.std(probabilities)),
        },
        "total_samples": total,
    }


# ==============================================================================
# OPTIONAL CALIBRATION
# ==============================================================================

def _run_calibration_if_available(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    evaluation_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Run calibration evaluation when enabled and available.

    Calibration is kept optional so this stage remains compatible with
    installations where the calibration module is not yet deployed.
    """
    calibration_cfg = evaluation_config.get("calibration", {})

    if calibration_cfg is False:
        return {
            "status": GATE_SKIP,
            "reason": "Calibration evaluation disabled.",
        }

    if isinstance(calibration_cfg, dict):
        enabled = calibration_cfg.get("enabled", False)
    else:
        enabled = False

    if not enabled:
        return {
            "status": GATE_SKIP,
            "reason": "Calibration evaluation disabled.",
        }

    try:
        from src.evaluation.calibration import CalibrationEvaluator

        n_bins = int(calibration_cfg.get("n_bins", 10))
        strategy = str(calibration_cfg.get("strategy", "uniform"))
        threshold = float(
            calibration_cfg.get("ece_threshold", 0.05)
        )

        evaluator = CalibrationEvaluator(
            n_bins=n_bins,
            strategy=strategy,
            ece_threshold=threshold,
        )

        report = evaluator.evaluate(
            y_true=y_true,
            y_prob=probabilities,
        )

        result = report.to_dict()
        result["status"] = (
            GATE_PASS if report.is_well_calibrated else GATE_FAIL
        )

        return result

    except Exception as exc:
        logger.exception("Calibration evaluation failed.")
        return {
            "status": GATE_FAIL,
            "error": str(exc),
        }


# ==============================================================================
# OPTIONAL ADVERSARIAL EVALUATION
# ==============================================================================

def _run_adversarial_if_available(
    candidate_model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    evaluation_config: Mapping[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    """
    Optionally execute configured adversarial robustness evaluation.

    Adversarial feature sets must already be generated by an approved,
    deterministic upstream process.
    """
    adversarial_cfg = evaluation_config.get("adversarial", {})

    if not isinstance(adversarial_cfg, dict):
        return {
            "status": GATE_SKIP,
            "reason": "Adversarial evaluation configuration is disabled.",
        }

    if not adversarial_cfg.get("enabled", False):
        return {
            "status": GATE_SKIP,
            "reason": "Adversarial evaluation disabled.",
        }

    try:
        from src.evaluation.adversarial import (
            evaluate_adversarial_robustness,
        )

        configured_sets = adversarial_cfg.get("sets", {})

        if not configured_sets:
            return {
                "status": GATE_FAIL,
                "error": (
                    "Adversarial evaluation is enabled but no adversarial "
                    "feature sets were configured."
                ),
            }

        adversarial_sets: Dict[str, pd.DataFrame] = {}

        for attack_name, path_value in configured_sets.items():
            attack_path = _require_existing_file(
                path_value,
                f"Adversarial dataset '{attack_name}'",
            )

            attack_df = pd.read_csv(attack_path)

            # The adversarial artifact must have exactly the feature columns
            # expected by the candidate model after schema validation.
            expected_features = _extract_expected_features(candidate_model)

            if expected_features is not None:
                missing = [
                    feature
                    for feature in expected_features
                    if feature not in attack_df.columns
                ]

                if missing:
                    raise ValueError(
                        f"Adversarial dataset '{attack_name}' is missing "
                        f"features: {missing}"
                    )

                attack_df = attack_df.loc[:, expected_features]

            adversarial_sets[str(attack_name)] = attack_df

        result = evaluate_adversarial_robustness(
            model=candidate_model,
            adversarial_sets=adversarial_sets,
            X_clean=X_test,
            y_true=y_test,
            max_recall_decay=float(
                adversarial_cfg.get("max_recall_decay", 0.15)
            ),
            threshold=threshold,
        )

        result["status"] = (
            GATE_PASS
            if result.get("passed_all_gates", False)
            else GATE_FAIL
        )

        return result

    except Exception as exc:
        logger.exception("Adversarial evaluation failed.")
        return {
            "status": GATE_FAIL,
            "error": str(exc),
        }


# ==============================================================================
# MAIN EVALUATION PIPELINE
# ==============================================================================

def run_evaluation_stage(
    candidate_model_path: Union[str, Path],
    test_data_path: Union[str, Path],
    output_eval_dir: Union[str, Path],
    champion_model_path: Optional[Union[str, Path]] = None,
    slice_columns: Optional[List[str]] = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    threshold: float = DEFAULT_THRESHOLD,
    min_parity_ratio: float = DEFAULT_MIN_PARITY_RATIO,
    guardrails: Optional[Dict[str, float]] = None,
    min_test_samples: int = DEFAULT_MIN_TEST_SAMPLES,
    min_class_samples: int = DEFAULT_MIN_CLASS_SAMPLES,
    strict_schema: bool = True,
    evaluation_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute the complete Stage 03 production evaluation.

    The function intentionally fails closed when fundamental evaluation
    integrity checks fail.
    """
    started_at = _utc_now()
    eval_dir = Path(output_eval_dir).resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)

    cfg: Mapping[str, Any] = evaluation_config or {}

    threshold = _validate_threshold(threshold)

    candidate_model, candidate_path = _load_model(
        candidate_model_path,
        "candidate",
    )

    test_path = _require_existing_file(
        test_data_path,
        "test dataset",
    )

    candidate_metadata = _load_model_metadata(candidate_path)

    logger.info(
        "Stage 03 evaluation started. Candidate='%s', Test='%s'",
        candidate_path,
        test_path,
    )

    # --------------------------------------------------------------------------
    # 1. Load immutable test data
    # --------------------------------------------------------------------------

    try:
        test_df = pd.read_csv(test_path)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load test dataset '{test_path}'."
        ) from exc

    _validate_test_dataframe(
        test_df,
        target_column=target_column,
    )

    target_summary = _validate_target(
        test_df[target_column],
        target_column=target_column,
        min_samples=int(min_test_samples),
        min_class_samples=int(min_class_samples),
    )

    y_test = pd.to_numeric(
        test_df[target_column],
        errors="raise",
    ).astype(int).to_numpy()

    # --------------------------------------------------------------------------
    # 2. Prepare feature and slice data
    # --------------------------------------------------------------------------

    X_test, slices_map = _prepare_features_and_slices(
        test_df=test_df,
        target_column=target_column,
        slice_columns=slice_columns,
    )

    # IMPORTANT:
    # Do not blindly select numeric columns here.
    # The model's expected schema is authoritative.
    expected_features = _extract_expected_features(candidate_model)

    if expected_features is not None:
        missing = [
            feature
            for feature in expected_features
            if feature not in X_test.columns
        ]

        if missing:
            raise ValueError(
                "Candidate/test schema mismatch. Missing model features: "
                f"{missing}"
            )

        X_test = X_test.loc[:, expected_features].copy()

    else:
        # Legacy fallback.
        logger.warning(
            "Candidate model does not expose feature_names_in_. "
            "Using numeric columns as a legacy compatibility fallback."
        )

        non_numeric = X_test.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        if non_numeric:
            X_test = X_test.drop(columns=non_numeric)

    if X_test.empty:
        raise ValueError(
            "No usable model features remain after schema preparation."
        )

    if X_test.columns.duplicated().any():
        raise ValueError("Model feature matrix contains duplicate columns.")

    # Validate numerical matrix.
    numeric_columns = X_test.select_dtypes(include=[np.number]).columns

    if len(numeric_columns) != len(X_test.columns):
        non_numeric = [
            column
            for column in X_test.columns
            if column not in numeric_columns
        ]

        raise TypeError(
            "Evaluation data contains non-numeric model features that the "
            f"current training architecture cannot safely evaluate: {non_numeric}"
        )

    if X_test.isna().any().any():
        missing_columns = (
            X_test.columns[X_test.isna().any()]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            f"Model feature matrix contains missing values in: {missing_columns}"
        )

    candidate_schema = _validate_feature_schema(
        model=candidate_model,
        X_test=X_test,
        model_name="candidate",
        strict=strict_schema,
    )

    # --------------------------------------------------------------------------
    # 3. Candidate probabilities
    # --------------------------------------------------------------------------

    candidate_probs = _predict_positive_probability(
        model=candidate_model,
        X=X_test,
        model_name="candidate",
    )

    # --------------------------------------------------------------------------
    # 4. Global metrics
    # --------------------------------------------------------------------------

    logger.info("Calculating candidate global metrics.")

    global_metrics = calculate_classification_metrics(
        model=candidate_model,
        X=X_test,
        y_true=y_test,
        threshold=threshold,
    )

    operational_diagnostics = _calculate_operational_diagnostics(
        y_true=y_test,
        probabilities=candidate_probs,
        threshold=threshold,
    )

    logger.info(
        "Candidate test metrics: F1=%.4f Precision=%.4f Recall=%.4f "
        "ROC-AUC=%.4f FPR=%.4f",
        float(global_metrics.get("f1", 0.0)),
        float(global_metrics.get("precision", 0.0)),
        float(global_metrics.get("recall", 0.0)),
        float(global_metrics.get("roc_auc", 0.0)),
        float(global_metrics.get("fpr", 0.0)),
    )

    # --------------------------------------------------------------------------
    # 5. Fairness audit
    # --------------------------------------------------------------------------

    fairness_results: Dict[str, Any] = {}
    passed_fairness_gate = True

    if slices_map:
        logger.info(
            "Executing fairness audit across %d slice dimensions.",
            len(slices_map),
        )

        fairness_results = evaluate_slice_fairness(
            model=candidate_model,
            X=X_test,
            y_true=y_test,
            slices=slices_map,
            threshold=threshold,
            min_parity_ratio=min_parity_ratio,
        )

        for slice_name, report in fairness_results.items():
            if not report.get("passed_fairness_gate", False):
                passed_fairness_gate = False

                logger.warning(
                    "Fairness gate failed for slice '%s'.",
                    slice_name,
                )

    else:
        fairness_results = {
            "_status": {
                "status": GATE_SKIP,
                "reason": "No configured fairness slices were available.",
            }
        }

    # --------------------------------------------------------------------------
    # 6. Regression comparison
    # --------------------------------------------------------------------------

    regression_results: Optional[Dict[str, Any]] = None
    passed_regression_gate = True
    regression_gate_status = GATE_SKIP

    champion_path: Optional[Path] = None
    champion_metadata: Optional[Dict[str, Any]] = None

    if champion_model_path is not None:
        champion_candidate_path = Path(champion_model_path)

        if champion_candidate_path.exists():
            champion_model, champion_path = _load_model(
                champion_candidate_path,
                "champion",
            )

            champion_metadata = _load_model_metadata(champion_path)

            # Validate champion against exactly the same evaluation features.
            _validate_feature_schema(
                model=champion_model,
                X_test=X_test.copy(),
                model_name="champion",
                strict=strict_schema,
            )

            logger.info(
                "Executing champion vs challenger regression benchmark."
            )

            regression_results = compare_champion_vs_challenger(
                champion=champion_model,
                challenger=candidate_model,
                X_test=X_test,
                y_test=y_test,
                threshold=threshold,
                guardrails=guardrails,
            )

            passed_regression_gate = bool(
                regression_results.get(
                    "passed_all_guardrails",
                    False,
                )
            )

            regression_gate_status = (
                GATE_PASS
                if passed_regression_gate
                else GATE_FAIL
            )

        else:
            logger.warning(
                "Configured champion model does not exist: '%s'. "
                "Regression comparison will be skipped.",
                champion_candidate_path,
            )

            regression_results = {
                "status": GATE_SKIP,
                "reason": "Champion artifact unavailable.",
            }

            # Missing champion is NOT a regression pass.
            passed_regression_gate = True
            regression_gate_status = GATE_SKIP

    else:
        regression_results = {
            "status": GATE_SKIP,
            "reason": "No champion model configured.",
        }

    # --------------------------------------------------------------------------
    # 7. Calibration
    # --------------------------------------------------------------------------

    calibration_results = _run_calibration_if_available(
        y_true=y_test,
        probabilities=candidate_probs,
        evaluation_config=cfg,
    )

    calibration_gate_status = calibration_results.get(
        "status",
        GATE_SKIP,
    )

    calibration_gate_enabled = (
        calibration_gate_status != GATE_SKIP
    )

    passed_calibration_gate = (
        not calibration_gate_enabled
        or calibration_gate_status == GATE_PASS
    )

    # --------------------------------------------------------------------------
    # 8. Adversarial robustness
    # --------------------------------------------------------------------------

    adversarial_results = _run_adversarial_if_available(
        candidate_model=candidate_model,
        X_test=X_test,
        y_test=y_test,
        evaluation_config=cfg,
        threshold=threshold,
    )

    adversarial_gate_status = adversarial_results.get(
        "status",
        GATE_SKIP,
    )

    adversarial_gate_enabled = (
        adversarial_gate_status != GATE_SKIP
    )

    passed_adversarial_gate = (
        not adversarial_gate_enabled
        or adversarial_gate_status == GATE_PASS
    )

    # --------------------------------------------------------------------------
    # 9. Gate decision
    # --------------------------------------------------------------------------

    passed_data_gate = True
    passed_schema_gate = True

    if not passed_fairness_gate:
        deployment_verdict = VERDICT_REJECT_FAIRNESS

    elif not passed_regression_gate:
        deployment_verdict = VERDICT_REJECT_REGRESSION

    elif not passed_calibration_gate:
        deployment_verdict = VERDICT_REJECT_CALIBRATION

    elif not passed_adversarial_gate:
        deployment_verdict = VERDICT_REJECT_ADVERSARIAL

    else:
        if champion_path is None:
            deployment_verdict = VERDICT_INITIAL_BASELINE
        else:
            raw_verdict = str(
                regression_results.get(
                    "verdict",
                    VERDICT_NEUTRAL,
                )
            )

            if raw_verdict == VERDICT_PROMOTE:
                deployment_verdict = VERDICT_PROMOTE
            else:
                deployment_verdict = raw_verdict

    passed_all_gates = bool(
        passed_data_gate
        and passed_schema_gate
        and passed_fairness_gate
        and passed_regression_gate
        and passed_calibration_gate
        and passed_adversarial_gate
    )

    # --------------------------------------------------------------------------
    # 10. Artifact hashes
    # --------------------------------------------------------------------------

    artifact_hashes = {
        "candidate_model_sha256": _sha256_file(candidate_path),
        "test_dataset_sha256": _sha256_file(test_path),
    }

    if champion_path is not None:
        artifact_hashes["champion_model_sha256"] = _sha256_file(
            champion_path
        )

    # --------------------------------------------------------------------------
    # 11. Gate summary
    # --------------------------------------------------------------------------

    gates = {
        "data_integrity": {
            "status": GATE_PASS if passed_data_gate else GATE_FAIL,
            "passed": passed_data_gate,
        },
        "feature_schema": {
            "status": GATE_PASS if passed_schema_gate else GATE_FAIL,
            "passed": passed_schema_gate,
        },
        "fairness": {
            "status": (
                GATE_PASS
                if passed_fairness_gate
                else GATE_FAIL
            ),
            "passed": passed_fairness_gate,
        },
        "regression": {
            "status": regression_gate_status,
            "passed": passed_regression_gate,
            "required": champion_path is not None,
        },
        "calibration": {
            "status": calibration_gate_status,
            "passed": passed_calibration_gate,
            "required": calibration_gate_enabled,
        },
        "adversarial": {
            "status": adversarial_gate_status,
            "passed": passed_adversarial_gate,
            "required": adversarial_gate_enabled,
        },
    }

    # --------------------------------------------------------------------------
    # 12. Final audit report
    # --------------------------------------------------------------------------

    finished_at = _utc_now()

    final_report: Dict[str, Any] = {
        "stage": {
            "name": "03_evaluate",
            "version": STAGE_VERSION,
            "started_at": started_at,
            "finished_at": finished_at,
        },
        "deployment_verdict": deployment_verdict,
        "passed_all_gates": passed_all_gates,
        "gates": gates,
        "candidate_global_metrics": global_metrics,
        "candidate_operational_diagnostics": operational_diagnostics,
        "slice_fairness_audits": fairness_results,
        "regression_benchmark": regression_results,
        "calibration": calibration_results,
        "adversarial": adversarial_results,
        "dataset": {
            "path": str(test_path),
            "sha256": artifact_hashes["test_dataset_sha256"],
            **target_summary,
            "feature_count": len(X_test.columns),
            "feature_names": [str(column) for column in X_test.columns],
        },
        "artifacts": {
            "candidate_model": {
                "path": str(candidate_path),
                "sha256": artifact_hashes[
                    "candidate_model_sha256"
                ],
            },
            "champion_model": (
                {
                    "path": str(champion_path),
                    "sha256": artifact_hashes[
                        "champion_model_sha256"
                    ],
                }
                if champion_path is not None
                else None
            ),
        },
        "model_metadata": {
            "candidate": candidate_metadata,
            "champion": champion_metadata,
        },
        "schema_validation": {
            "candidate": candidate_schema,
        },
        "configuration": {
            "target_column": target_column,
            "decision_threshold": threshold,
            "minimum_test_samples": min_test_samples,
            "minimum_class_samples": min_class_samples,
            "minimum_fairness_parity_ratio": min_parity_ratio,
            "strict_schema": strict_schema,
            "slice_columns": list(slice_columns or []),
            "guardrails": guardrails,
        },
        "runtime": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
    }

    # --------------------------------------------------------------------------
    # 13. Write CI/CD artifacts
    # --------------------------------------------------------------------------

    eval_json_path = eval_dir / "evaluation_report.json"
    metrics_summary_path = eval_dir / "metrics.json"
    gates_path = eval_dir / "quality_gates.json"
    manifest_path = eval_dir / "evaluation_manifest.json"

    _atomic_write_json(
        eval_json_path,
        final_report,
    )

    _atomic_write_json(
        metrics_summary_path,
        {
            "f1": global_metrics.get("f1"),
            "precision": global_metrics.get("precision"),
            "recall": global_metrics.get("recall"),
            "roc_auc": global_metrics.get("roc_auc"),
            "fpr": global_metrics.get("fpr"),
            "threshold": threshold,
            "passed_gates": passed_all_gates,
            "verdict": deployment_verdict,
        },
    )

    _atomic_write_json(
        gates_path,
        gates,
    )

    manifest = {
        "stage": "03_evaluate",
        "version": STAGE_VERSION,
        "generated_at": finished_at,
        "evaluation_report": {
            "path": str(eval_json_path),
            "sha256": _sha256_file(eval_json_path),
        },
        "metrics": {
            "path": str(metrics_summary_path),
            "sha256": _sha256_file(metrics_summary_path),
        },
        "quality_gates": {
            "path": str(gates_path),
            "sha256": _sha256_file(gates_path),
        },
        "input_artifacts": artifact_hashes,
        "deployment_verdict": deployment_verdict,
        "passed_all_gates": passed_all_gates,
    }

    _atomic_write_json(
        manifest_path,
        manifest,
    )

    logger.info(
        "Stage 03 evaluation completed. Verdict='%s', "
        "PassedAllGates=%s, Report='%s'",
        deployment_verdict,
        passed_all_gates,
        eval_json_path,
    )

    return final_report


# ==============================================================================
# STAGE ENTRYPOINT
# ==============================================================================

def main(
    config_path: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Stage 03 orchestration entrypoint.

    Configuration precedence:
        explicit kwargs
            >
        params.yaml evaluation configuration
            >
        safe defaults
    """
    config = load_config(config_path) if config_path else {}

    evaluation_cfg = config.get("evaluation", {})
    data_cfg = config.get("data_preparation", {})
    main_cfg = config.get("main", {})

    data_dir = data_cfg.get(
        "output_dir",
        "data/processed",
    )

    candidate_model_path = (
        kwargs.get("candidate_model_path")
        or evaluation_cfg.get(
            "candidate_model_path",
            "models/candidate/candidate_model.joblib",
        )
    )

    champion_configured = kwargs.get(
        "champion_model_path",
        evaluation_cfg.get(
            "champion_model_path",
            "models/champion/champion_model.joblib",
        ),
    )

    champion_path = (
        champion_configured
        if champion_configured
        and Path(str(champion_configured)).exists()
        else None
    )

    test_data_path = (
        kwargs.get("test_data_path")
        or Path(data_dir) / "test.csv"
    )

    output_eval_dir = (
        kwargs.get("output_eval_dir")
        or evaluation_cfg.get(
            "output_eval_dir",
            "reports/evaluation",
        )
    )

    slice_columns = evaluation_cfg.get(
        "slice_columns",
        DEFAULT_SLICE_COLUMNS,
    )

    target_column = data_cfg.get(
        "target_column",
        DEFAULT_TARGET_COLUMN,
    )

    threshold = evaluation_cfg.get(
        "decision_threshold",
        DEFAULT_THRESHOLD,
    )

    min_parity_ratio = evaluation_cfg.get(
        "min_parity_ratio",
        DEFAULT_MIN_PARITY_RATIO,
    )

    guardrails = evaluation_cfg.get(
        "guardrails",
        None,
    )

    min_test_samples = int(
        evaluation_cfg.get(
            "min_test_samples",
            DEFAULT_MIN_TEST_SAMPLES,
        )
    )

    min_class_samples = int(
        evaluation_cfg.get(
            "min_class_samples",
            DEFAULT_MIN_CLASS_SAMPLES,
        )
    )

    strict_schema = bool(
        evaluation_cfg.get(
            "strict_schema",
            True,
        )
    )

    return run_evaluation_stage(
        candidate_model_path=candidate_model_path,
        champion_model_path=champion_path,
        test_data_path=test_data_path,
        output_eval_dir=output_eval_dir,
        slice_columns=slice_columns,
        target_column=target_column,
        threshold=threshold,
        min_parity_ratio=min_parity_ratio,
        guardrails=guardrails,
        min_test_samples=min_test_samples,
        min_class_samples=min_class_samples,
        strict_schema=strict_schema,
        evaluation_config=evaluation_cfg,
    )


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "PhishGuard AI Stage 03 - Global Evaluation, "
            "Fairness, Calibration, Adversarial & Regression Gates"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="params.yaml",
        help="Path to pipeline configuration YAML.",
    )

    parser.add_argument(
        "--candidate-model",
        type=str,
        default=None,
        help="Override candidate model path.",
    )

    parser.add_argument(
        "--champion-model",
        type=str,
        default=None,
        help="Override champion model path.",
    )

    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Override test dataset path.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override evaluation output directory.",
    )

    args = parser.parse_args()

    cli_kwargs: Dict[str, Any] = {}

    if args.candidate_model:
        cli_kwargs["candidate_model_path"] = args.candidate_model

    if args.champion_model:
        cli_kwargs["champion_model_path"] = args.champion_model

    if args.test_data:
        cli_kwargs["test_data_path"] = args.test_data

    if args.output_dir:
        cli_kwargs["output_eval_dir"] = args.output_dir

    try:
        result = main(
            config_path=args.config,
            **cli_kwargs,
        )

        if result.get("passed_all_gates", False):
            logger.info(
                "Stage 03 completed successfully: all enabled deployment gates passed."
            )
            raise SystemExit(0)

        logger.error(
            "Stage 03 deployment gate FAILED. Verdict='%s'.",
            result.get("deployment_verdict"),
        )
        raise SystemExit(2)

    except KeyboardInterrupt:
        logger.warning("Stage 03 interrupted by user.")
        raise SystemExit(130)

    except Exception as exc:
        logger.exception(
            "Stage 03 evaluation failed closed: %s",
            exc,
        )
        raise SystemExit(1)


__all__ = [
    "STAGE_VERSION",
    "run_evaluation_stage",
    "main",
]

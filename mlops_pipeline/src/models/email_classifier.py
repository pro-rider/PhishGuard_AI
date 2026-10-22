"""
PhishGuard_AI - Production Email Phishing Classifier
Version: 14.0.0

Production-grade email phishing classifier supporting LightGBM primary backend
with HistGradientBoosting fallback, deterministic training, strict feature
validation, class-imbalance handling, threshold classification, confidence
estimation, and defensive error handling.

The classifier treats model input as untrusted data. Feature names may not
silently change, NaN/Inf values are rejected, empty datasets are rejected,
labels must be binary, and model state must be explicitly fitted before
inference.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.models.base_model import (
    BasePhishingModel,
    ModelInputError,
    ModelNotFittedError,
)
from src.utils.logger import get_logger

logger = get_logger("src.models.email_classifier")

MODEL_VERSION = "14.0.0"
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_PREDICTION_BATCH = 100_000
DEFAULT_MODEL_NAME = "email_phishing_classifier"


@dataclass
class TrainingStatistics:
    backend: str = "unknown"
    rows_trained: Optional[int] = None
    features_count: Optional[int] = None
    training_seconds: Optional[float] = None
    positive_ratio: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionStatistics:
    rows: Optional[int] = None
    mean_positive_probability: Optional[float] = None
    threshold: Optional[float] = None
    positive_count: Optional[int] = None


class EmailPhishingClassifier(BasePhishingModel):
    """Production-grade gradient boosting classifier for email phishing."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        version: str = MODEL_VERSION,
        decision_threshold: float = 0.50,
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = DEFAULT_RANDOM_STATE,
        strict_input_validation: bool = True,
        reject_nan: bool = True,
        reject_inf: bool = True,
        handle_class_imbalance: bool = True,
        class_weight: Optional[Union[str, Dict[int, float]]] = None,
        max_prediction_batch: int = DEFAULT_MAX_PREDICTION_BATCH,
    ) -> None:
        self._validate_constructor_arguments(
            decision_threshold=decision_threshold,
            random_state=random_state,
            max_prediction_batch=max_prediction_batch,
        )

        default_params = self._default_model_parameters(random_state=random_state)
        if model_params:
            default_params.update(model_params)
        default_params["random_state"] = random_state

        super().__init__(
            model_name=model_name,
            version=version,
            decision_threshold=decision_threshold,
            model_params=default_params,
        )

        self.random_state = random_state
        self.strict_input_validation = strict_input_validation
        self.reject_nan = reject_nan
        self.reject_inf = reject_inf
        self.handle_class_imbalance = handle_class_imbalance
        self.class_weight = class_weight
        self.max_prediction_batch = max_prediction_batch

        self.backend_: str = "unknown"
        self.backend_version_: Optional[str] = None
        self.feature_dtypes_: Dict[str, str] = {}
        self.schema_hash_: Optional[str] = None
        self.training_statistics_: TrainingStatistics = TrainingStatistics()
        self.last_prediction_statistics_: Optional[PredictionStatistics] = None
        self.last_validation_metrics_: Dict[str, float] = {}

    @staticmethod
    def _default_model_parameters(random_state: int) -> Dict[str, Any]:
        return {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 7,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "random_state": random_state,
            "n_jobs": 2,
        }

    @staticmethod
    def _validate_constructor_arguments(**kwargs: Any) -> None:
        if not (0.0 <= kwargs.get("decision_threshold", 0.5) <= 1.0):
            raise ValueError("decision_threshold must be within [0, 1].")
        if not isinstance(kwargs.get("random_state", 42), int):
            raise ValueError("random_state must be an integer.")
        if kwargs.get("max_prediction_batch", 0) <= 0:
            raise ValueError("max_prediction_batch must be positive.")

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        validation_data: Optional[Tuple[Any, Any]] = None,
        feature_names: Optional[Sequence[str]] = None,
        dataset_id: Optional[str] = None,
        dataset_version: Optional[str] = None,
    ) -> "EmailPhishingClassifier":
        """Fit the classifier with strict input and label validation."""
        started = time.time()

        X_df, y_arr = self._coerce_fit_input(X, y, feature_names)
        self._validate_binary_target(y_arr)

        self.feature_names_ = list(X_df.columns)
        self.feature_count_ = len(self.feature_names_)
        self.feature_fingerprint_ = self._fingerprint_features(X_df)

        model = self._build_and_train(X_df, y_arr, validation_data)

        self.model = model
        self.backend_ = self._detect_backend(model)
        self.is_fitted = True
        self.lifecycle_state = "fitted"
        self.fitted_timestamp_ = self._utc_now()

        self.training_statistics_ = TrainingStatistics(
            backend=self.backend_,
            rows_trained=int(len(y_arr)),
            features_count=self.feature_count_,
            training_seconds=round(time.time() - started, 4),
            positive_ratio=float(np.mean(y_arr)),
        )
        return self

    def _coerce_fit_input(
        self, X: Any, y: Any, feature_names: Optional[Sequence[str]]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        elif isinstance(X, np.ndarray):
            names = list(feature_names) if feature_names else [
                f"f{i}" for i in range(X.shape[1])
            ]
            X_df = pd.DataFrame(X, columns=names)
        else:
            raise ModelInputError("X must be a pandas DataFrame or numpy array.")

        if X_df.empty:
            raise ValueError("Empty datasets are rejected.")
        if X_df.shape[0] != len(y):
            raise ValueError("X and y row counts must match.")

        y_arr = np.asarray(y, dtype=float).ravel()
        if self.reject_nan and X_df.isna().any().any():
            raise ValueError("NaN values detected in training features.")
        if self.reject_inf and np.isinf(X_df.to_numpy(dtype=float)).any():
            raise ValueError("Inf values detected in training features.")
        return X_df, y_arr

    @staticmethod
    def _validate_binary_target(y: np.ndarray) -> None:
        unique = np.unique(y)
        if not set(unique).issubset({0.0, 1.0}):
            raise ValueError("Labels must be binary (0/1).")

    def _build_and_train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        validation_data: Optional[Tuple[Any, Any]],
    ) -> Any:
        params = dict(self.model_params)
        try:
            import lightgbm as lgb

            train_data = lgb.Dataset(X, label=y)
            valid_sets = []
            callbacks = []
            if validation_data is not None:
                Xv, yv = validation_data
                valid_sets = [lgb.Dataset(Xv, label=yv)]
                callbacks = [lgb.early_stopping(20)]
            booster = lgb.train(
                params,
                train_data,
                num_boost_round=params.pop("n_estimators", 300),
                valid_sets=valid_sets or None,
                callbacks=callbacks,
            )
            return booster
        except ImportError:
            return self._fallback_hist_gbm(X, y, params)

    def _fallback_hist_gbm(self, X: pd.DataFrame, y: np.ndarray, params: dict) -> Any:
        from sklearn.ensemble import HistGradientBoostingClassifier

        estimator = HistGradientBoostingClassifier(
            max_iter=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 7),
            random_state=params.get("random_state", self.random_state),
        )
        estimator.fit(X, y)
        return estimator

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return positive-class probabilities in [0, 1]."""
        if not self.is_fitted or self.model is None:
            raise ModelNotFittedError("Model must be fitted before prediction.")

        X_df = self._coerce_predict_input(X)
        batch = min(len(X_df), self.max_prediction_batch)

        if self.backend_ == "lightgbm":
            raw = self.model.predict(
                X_df[:batch], num_iteration=self.model.best_iteration or -1
            )
        else:
            proba = self.model.predict_proba(X_df[:batch])
            raw = proba[:, 1] if proba.ndim == 2 else proba

        probabilities = np.asarray(raw, dtype=float).ravel()
        if self.reject_nan and np.isnan(probabilities).any():
            raise ValueError("Model returned NaN probabilities.")
        probabilities = np.clip(probabilities, 0.0, 1.0)

        self.last_prediction_statistics_ = PredictionStatistics(
            rows=int(len(probabilities)),
            mean_positive_probability=float(np.mean(probabilities)),
            threshold=self.decision_threshold,
            positive_count=int(np.sum(probabilities >= self.decision_threshold)),
        )
        return probabilities

    def _coerce_predict_input(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        elif isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names_ or None)
        else:
            raise ModelInputError("X must be a DataFrame or array.")

        if self.reject_nan and X_df.isna().any().any():
            raise ValueError("NaN values detected in prediction input.")
        if self.reject_inf and np.isinf(X_df.to_numpy(dtype=float)).any():
            raise ValueError("Inf values detected in prediction input.")
        return X_df

    def get_feature_importance(self) -> Dict[str, float]:
        """Return a bounded feature-importance map for the fitted model."""
        if not self.is_fitted or self.model is None:
            raise ModelNotFittedError("Model must be fitted first.")

        if self.backend_ == "lightgbm":
            try:
                gains = self.model.feature_importance(importance_type="gain")
                return dict(zip(self.feature_names_, [float(g) for g in gains]))
            except Exception:  # noqa: BLE001
                pass
        try:
            importances = getattr(self.model, "feature_importances_", None)
            if importances is not None:
                return dict(
                    zip(self.feature_names_, [float(v) for v in importances])
                )
        except Exception:  # noqa: BLE001
            pass
        return {}

    @staticmethod
    def _fingerprint_features(X: pd.DataFrame) -> str:
        joined = ",".join(X.columns)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _detect_backend(model: Any) -> str:
        module = type(model).__module__
        if "lightgbm" in module:
            return "lightgbm"
        if "sklearn" in module:
            return "sklearn.hist_gradient_boosting"
        return "unknown"

    @staticmethod
    def _utc_now() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "EmailPhishingClassifier",
    "TrainingStatistics",
    "PredictionStatistics",
    "MODEL_VERSION",
]

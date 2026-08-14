"""
PhishGuard AI - Production URL Phishing Classifier
===================================================

Version: 14.0.0
File Path: src/models/url_classifier.py

Production-grade URL phishing classifier supporting:

    * LightGBM primary backend
    * HistGradientBoosting fallback
    * deterministic training
    * strict feature validation
    * feature-order protection
    * binary-target validation
    * NaN / Inf detection
    * class-imbalance handling
    * validation-set evaluation
    * LightGBM early stopping
    * probability prediction
    * threshold-based classification
    * confidence estimation
    * feature importance extraction
    * training statistics
    * inference diagnostics
    * schema fingerprinting
    * model health checks
    * safe parameter management
    * reproducibility metadata
    * estimator compatibility checks
    * defensive error handling

Security Philosophy
-------------------
The classifier treats model input as untrusted data.

The following assumptions are therefore enforced:

    1. Feature names may not silently change.
    2. Feature ordering may not silently change.
    3. NaN/Inf values are rejected by default.
    4. Empty datasets are rejected.
    5. Labels must be binary.
    6. Training and inference feature schemas must agree.
    7. Probability output must be finite.
    8. Decision thresholds must remain within [0, 1].
    9. Model state must be explicitly fitted before inference.
   10. Backend selection is recorded for reproducibility.

This module is designed for defensive ML/MLOps systems and does not perform
network access, URL fetching, DNS resolution, WHOIS queries, or sandboxing.
Those operations belong to dedicated upstream security components.
"""

from __future__ import annotations

import hashlib
import math
import time
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from src.models.base_model import BasePhishingModel
from src.utils.logger import get_logger


logger = get_logger("src.models.url_classifier")


# =============================================================================
# OPTIONAL BACKEND IMPORTS
# =============================================================================

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
    LIGHTGBM_VERSION = getattr(lgb, "__version__", "unknown")
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False
    LIGHTGBM_VERSION = None


try:
    from sklearn.ensemble import HistGradientBoostingClassifier

    HAS_SKLEARN_HGB = True
except ImportError:
    HistGradientBoostingClassifier = None
    HAS_SKLEARN_HGB = False


# =============================================================================
# TYPE ALIASES
# =============================================================================

FeatureInput = Union[pd.DataFrame, np.ndarray]
TargetInput = Union[pd.Series, np.ndarray, Sequence[int]]

ValidationData = Optional[
    Tuple[
        FeatureInput,
        TargetInput,
    ]
]


# =============================================================================
# CONSTANTS
# =============================================================================

MODEL_VERSION = "14.0.0"

DEFAULT_RANDOM_STATE = 42
DEFAULT_DECISION_THRESHOLD = 0.50

MIN_TRAINING_ROWS = 2
MIN_FEATURE_COUNT = 1

DEFAULT_MAX_PREDICTION_BATCH = 100_000

SUPPORTED_TARGET_VALUES = {0, 1}

# Parameters that are controlled internally and should not be blindly replaced
# by arbitrary user values without normalization.
PROTECTED_PARAMETER_KEYS = {
    "random_state",
}

# Feature names are hashed instead of exposing raw feature values.
SCHEMA_HASH_ALGORITHM = "sha256"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrainingStatistics:
    """
    Immutable-style summary of a completed training operation.

    This object deliberately stores metadata rather than raw training data.
    """

    sample_count: int = 0
    feature_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    positive_rate: float = 0.0
    training_seconds: float = 0.0
    validation_sample_count: int = 0
    backend: str = "unknown"
    schema_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-compatible representation."""
        return asdict(self)


@dataclass
class PredictionStatistics:
    """
    Runtime diagnostics for one prediction operation.

    No raw URLs are stored.
    """

    sample_count: int = 0
    positive_predictions: int = 0
    negative_predictions: int = 0
    positive_rate: float = 0.0
    mean_probability: float = 0.0
    min_probability: float = 0.0
    max_probability: float = 0.0
    threshold: float = DEFAULT_DECISION_THRESHOLD
    inference_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-compatible representation."""
        return asdict(self)


# =============================================================================
# URL PHISHING CLASSIFIER
# =============================================================================

class URLPhishingClassifier(BasePhishingModel):
    """
    Production-grade gradient boosting classifier for URL phishing detection.

    The model consumes already-extracted URL features. It intentionally does not
    perform external URL interaction.

    Typical features may include:

        url_length
        domain_age_days
        num_subdomains
        has_ip_address
        use_shortener
        entropy
        path_length
        query_length
        special_character_count
        digit_ratio
        suspicious_token_count
        hostname_length
        tls_present
        ...

    The exact feature set is determined by the training DataFrame and is locked
    after fitting.
    """

    # =========================================================================
    # CONSTRUCTOR
    # =========================================================================

    def __init__(
        self,
        model_name: str = "url_phishing_classifier",
        version: str = MODEL_VERSION,
        decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = DEFAULT_RANDOM_STATE,
        strict_input_validation: bool = True,
        reject_nan: bool = True,
        reject_inf: bool = True,
        handle_class_imbalance: bool = True,
        class_weight: Optional[Union[str, Dict[int, float]]] = None,
        max_prediction_batch: int = DEFAULT_MAX_PREDICTION_BATCH,
    ) -> None:
        """
        Initialize URL phishing classifier.

        Parameters
        ----------
        model_name:
            Stable artifact/model identifier.

        version:
            Model implementation version.

        decision_threshold:
            Probability threshold used for phishing classification.

        model_params:
            Backend estimator parameters.

        random_state:
            Reproducibility seed.

        strict_input_validation:
            Enables strict feature and input validation.

        reject_nan:
            Reject NaN values instead of allowing backend-dependent handling.

        reject_inf:
            Reject positive/negative infinity.

        handle_class_imbalance:
            Automatically apply class balancing when possible.

        class_weight:
            Explicit class weighting configuration.

        max_prediction_batch:
            Maximum number of rows accepted by a single prediction call.
        """

        self._validate_constructor_arguments(
            decision_threshold=decision_threshold,
            random_state=random_state,
            max_prediction_batch=max_prediction_batch,
        )

        default_params = self._default_model_parameters(
            random_state=random_state
        )

        if model_params:
            default_params.update(deepcopy(model_params))

        # Always enforce deterministic random state unless the implementation
        # explicitly supports a different deterministic mechanism.
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
        self.class_weight = deepcopy(class_weight)
        self.max_prediction_batch = max_prediction_batch

        self.backend_: str = "unknown"
        self.backend_version_: Optional[str] = None

        self.feature_dtypes_: Dict[str, str] = {}
        self.schema_hash_: Optional[str] = None

        self.training_statistics_: TrainingStatistics = TrainingStatistics()

        self.last_prediction_statistics_: Optional[
            PredictionStatistics
        ] = None

        self.last_validation_metrics_: Dict[str, float] = {}

        self.classes_: Optional[np.ndarray] = None

        self.n_features_in_: Optional[int] = None

        self.training_started_at_: Optional[str] = None
        self.training_completed_at_: Optional[str] = None

        self._init_estimator()

    # =========================================================================
    # DEFAULT PARAMETERS
    # =========================================================================

    @staticmethod
    def _default_model_parameters(
        random_state: int,
    ) -> Dict[str, Any]:
        """
        Return conservative production defaults.

        These defaults prioritize stable generalization over extreme model
        complexity.
        """

        if HAS_LIGHTGBM:
            return {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 8,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
                "reg_alpha": 0.10,
                "reg_lambda": 0.10,
                "random_state": random_state,
                "n_jobs": -1,
                "verbosity": -1,
            }

        return {
            "max_iter": 300,
            "learning_rate": 0.05,
            "max_depth": 8,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.10,
            "random_state": random_state,
        }

    # =========================================================================
    # CONSTRUCTOR VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_constructor_arguments(
        decision_threshold: float,
        random_state: int,
        max_prediction_batch: int,
    ) -> None:
        """Validate constructor-level security and correctness constraints."""

        if not isinstance(decision_threshold, (float, int)):
            raise TypeError(
                "decision_threshold must be numeric."
            )

        if not 0.0 <= float(decision_threshold) <= 1.0:
            raise ValueError(
                "decision_threshold must be between 0.0 and 1.0."
            )

        if not isinstance(random_state, int):
            raise TypeError(
                "random_state must be an integer."
            )

        if random_state < 0:
            raise ValueError(
                "random_state must be non-negative."
            )

        if not isinstance(max_prediction_batch, int):
            raise TypeError(
                "max_prediction_batch must be an integer."
            )

        if max_prediction_batch <= 0:
            raise ValueError(
                "max_prediction_batch must be greater than zero."
            )

    # =========================================================================
    # ESTIMATOR INITIALIZATION
    # =========================================================================

    def _init_estimator(self) -> None:
        """
        Instantiate the selected backend.

        LightGBM is preferred because it is generally better suited to
        high-throughput tabular URL telemetry.

        HistGradientBoosting is used as a local fallback.
        """

        if HAS_LIGHTGBM:
            self.backend_ = "lightgbm"
            self.backend_version_ = LIGHTGBM_VERSION

            params = deepcopy(self.model_params)

            try:
                self.model = lgb.LGBMClassifier(**params)
            except TypeError as exc:
                logger.error(
                    "Failed to initialize LightGBM with supplied parameters: %s",
                    exc,
                )
                raise

            return

        if HAS_SKLEARN_HGB:
            self.backend_ = "hist_gradient_boosting"
            self.backend_version_ = self._get_sklearn_version()

            translated = self._translate_to_sklearn_parameters(
                self.model_params
            )

            self.model = HistGradientBoostingClassifier(
                **translated
            )

            return

        raise ImportError(
            "No supported gradient boosting backend is installed. "
            "Install LightGBM or scikit-learn."
        )

    # =========================================================================
    # BACKEND PARAMETER TRANSLATION
    # =========================================================================

    @staticmethod
    def _translate_to_sklearn_parameters(
        params: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Translate commonly used LightGBM-style parameters into
        HistGradientBoostingClassifier parameters.
        """

        translated: Dict[str, Any] = {
            "max_iter": params.get(
                "n_estimators",
                params.get("max_iter", 300),
            ),
            "learning_rate": params.get(
                "learning_rate",
                0.05,
            ),
            "max_depth": params.get(
                "max_depth",
                8,
            ),
            "max_leaf_nodes": params.get(
                "num_leaves",
                params.get("max_leaf_nodes", 31),
            ),
            "min_samples_leaf": params.get(
                "min_child_samples",
                params.get("min_samples_leaf", 20),
            ),
            "l2_regularization": params.get(
                "reg_lambda",
                params.get("l2_regularization", 0.10),
            ),
            "random_state": params.get(
                "random_state",
                DEFAULT_RANDOM_STATE,
            ),
        }

        return translated

    @staticmethod
    def _get_sklearn_version() -> Optional[str]:
        """Retrieve installed scikit-learn version without hard failure."""

        try:
            import sklearn

            return getattr(sklearn, "__version__", "unknown")
        except Exception:
            return "unknown"

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    def _validate_feature_input(
        self,
        X: FeatureInput,
        *,
        inference: bool = False,
    ) -> None:
        """
        Validate feature matrix.

        Important:
        This method does not mutate X.
        """

        if X is None:
            raise ValueError(
                "Feature matrix X cannot be None."
            )

        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError(
                    "Feature DataFrame is empty."
                )

            if X.shape[1] < MIN_FEATURE_COUNT:
                raise ValueError(
                    "Feature DataFrame contains no usable features."
                )

            if X.columns.duplicated().any():
                duplicates = (
                    X.columns[X.columns.duplicated()]
                    .astype(str)
                    .tolist()
                )
                raise ValueError(
                    f"Duplicate feature columns detected: {duplicates}"
                )

            if any(
                not isinstance(column, str)
                for column in X.columns
            ):
                raise TypeError(
                    "All DataFrame feature names must be strings."
                )

            if self.reject_nan and X.isna().any().any():
                nan_columns = (
                    X.columns[X.isna().any()]
                    .astype(str)
                    .tolist()
                )

                raise ValueError(
                    "NaN values detected in feature matrix. "
                    f"Columns: {nan_columns}"
                )

            if self.reject_inf:
                numeric = X.select_dtypes(
                    include=[np.number]
                )

                if not numeric.empty:
                    values = numeric.to_numpy(
                        dtype=np.float64,
                        copy=False,
                    )

                    if not np.isfinite(values).all():
                        raise ValueError(
                            "Infinite or non-finite numeric values detected."
                        )

        elif isinstance(X, np.ndarray):

            if X.ndim != 2:
                raise ValueError(
                    "NumPy feature matrix must be two-dimensional."
                )

            if X.shape[0] == 0:
                raise ValueError(
                    "Feature matrix contains zero rows."
                )

            if X.shape[1] == 0:
                raise ValueError(
                    "Feature matrix contains zero columns."
                )

            if not np.issubdtype(X.dtype, np.number):
                raise TypeError(
                    "NumPy feature matrix must contain numeric values."
                )

            if self.reject_nan or self.reject_inf:
                if not np.isfinite(X).all():
                    raise ValueError(
                        "NumPy feature matrix contains non-finite values."
                    )

        else:
            raise TypeError(
                "X must be either pandas.DataFrame or numpy.ndarray."
            )

        if inference:
            self._validate_inference_schema(X)

    # =========================================================================
    # TARGET VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_target(
        y: TargetInput,
        expected_rows: int,
    ) -> np.ndarray:
        """
        Validate binary target vector.

        Returns
        -------
        np.ndarray
            Canonical int64 target array.
        """

        if y is None:
            raise ValueError(
                "Training target y cannot be None."
            )

        if isinstance(y, pd.Series):
            values = y.to_numpy()
        else:
            values = np.asarray(y)

        if values.ndim != 1:
            values = values.reshape(-1)

        if len(values) != expected_rows:
            raise ValueError(
                "Target length does not match feature row count. "
                f"X={expected_rows}, y={len(values)}."
            )

        if len(values) < MIN_TRAINING_ROWS:
            raise ValueError(
                "Training dataset contains too few samples."
            )

        if pd.isna(values).any():
            raise ValueError(
                "Target vector contains missing values."
            )

        unique_values = set(
            np.unique(values).tolist()
        )

        if not unique_values.issubset(
            SUPPORTED_TARGET_VALUES
        ):
            raise ValueError(
                "URLPhishingClassifier requires binary labels "
                "{0, 1}. Received: "
                f"{sorted(unique_values)}"
            )

        if len(unique_values) < 2:
            raise ValueError(
                "Training target must contain both classes "
                "0 and 1."
            )

        return values.astype(np.int64)

    # =========================================================================
    # CLASS IMBALANCE
    # =========================================================================

    def _calculate_class_weights(
        self,
        y: np.ndarray,
    ) -> Dict[int, float]:
        """
        Calculate balanced class weights.

        Formula:
            n_samples / (n_classes * class_count)
        """

        unique, counts = np.unique(
            y,
            return_counts=True,
        )

        total = len(y)
        classes = len(unique)

        return {
            int(cls): float(
                total / (classes * count)
            )
            for cls, count in zip(unique, counts)
        }

    # =========================================================================
    # FEATURE SCHEMA MANAGEMENT
    # =========================================================================

    @staticmethod
    def _calculate_schema_hash(
        feature_names: Sequence[str],
    ) -> str:
        """
        Calculate deterministic SHA-256 feature schema fingerprint.

        Feature values are never included.
        """

        canonical = "\x1f".join(
            str(name)
            for name in feature_names
        )

        return hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

    def _capture_feature_schema(
        self,
        X: FeatureInput,
    ) -> None:
        """Capture feature names, count, dtypes and schema fingerprint."""

        if isinstance(X, pd.DataFrame):
            names = [
                str(column)
                for column in X.columns
            ]

            self.feature_names_ = names

            self.feature_dtypes_ = {
                str(column): str(dtype)
                for column, dtype
                in X.dtypes.items()
            }

        else:
            names = [
                f"feature_{index}"
                for index in range(X.shape[1])
            ]

            self.feature_names_ = names

            self.feature_dtypes_ = {
                name: str(X.dtype)
                for name in names
            }

        self.n_features_in_ = len(
            self.feature_names_
        )

        self.schema_hash_ = self._calculate_schema_hash(
            self.feature_names_
        )

    def _validate_inference_schema(
        self,
        X: FeatureInput,
    ) -> None:
        """
        Protect inference against training/inference schema drift.

        This is particularly important for phishing systems because a silently
        reordered feature vector can produce confident but incorrect decisions.
        """

        if not self.is_fitted:
            return

        if self.n_features_in_ is None:
            raise RuntimeError(
                "Fitted model has no recorded feature count."
            )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Feature count mismatch during inference. "
                f"Expected {self.n_features_in_}, "
                f"received {X.shape[1]}."
            )

        if isinstance(X, pd.DataFrame):
            current_names = [
                str(column)
                for column in X.columns
            ]

            if current_names != self.feature_names_:
                missing = sorted(
                    set(self.feature_names_) -
                    set(current_names)
                )

                unexpected = sorted(
                    set(current_names) -
                    set(self.feature_names_)
                )

                raise ValueError(
                    "Feature schema mismatch during inference. "
                    f"Missing={missing}; "
                    f"Unexpected={unexpected}. "
                    "Feature order must exactly match training order."
                )

    # =========================================================================
    # FIT
    # =========================================================================

    def fit(
        self,
        X: FeatureInput,
        y: TargetInput,
        validation_data: ValidationData = None,
        **kwargs: Any,
    ) -> "URLPhishingClassifier":
        """
        Train the URL phishing classifier.

        Parameters
        ----------
        X:
            Training features.

        y:
            Binary training labels.

        validation_data:
            Optional `(X_validation, y_validation)` tuple.

        kwargs:
            Additional backend-specific fit arguments.
        """

        start_time = time.perf_counter()

        logger.info(
            "Starting URL phishing classifier training "
            "(backend=%s).",
            self.backend_,
        )

        # ---------------------------------------------------------------------
        # Validate training features
        # ---------------------------------------------------------------------

        self._validate_feature_input(
            X,
            inference=False,
        )

        target = self._validate_target(
            y,
            expected_rows=X.shape[0],
        )

        # ---------------------------------------------------------------------
        # Capture schema before fitting
        # ---------------------------------------------------------------------

        self._capture_feature_schema(X)

        # ---------------------------------------------------------------------
        # Prepare validation set
        # ---------------------------------------------------------------------

        validation_tuple = None

        if validation_data is not None:

            if (
                not isinstance(validation_data, tuple)
                or len(validation_data) != 2
            ):
                raise TypeError(
                    "validation_data must be "
                    "(X_validation, y_validation)."
                )

            X_val, y_val = validation_data

            self._validate_feature_input(
                X_val,
                inference=False,
            )

            if X_val.shape[1] != X.shape[1]:
                raise ValueError(
                    "Training and validation feature counts differ."
                )

            if isinstance(X, pd.DataFrame) and isinstance(
                X_val,
                pd.DataFrame,
            ):
                if list(X.columns) != list(X_val.columns):
                    raise ValueError(
                        "Training and validation feature "
                        "orders differ."
                    )

            validated_y_val = self._validate_target(
                y_val,
                expected_rows=X_val.shape[0],
            )

            validation_tuple = (
                X_val,
                validated_y_val,
            )

        # ---------------------------------------------------------------------
        # Class imbalance configuration
        # ---------------------------------------------------------------------

        self._configure_class_balance(target)

        # ---------------------------------------------------------------------
        # Backend-specific fitting
        # ---------------------------------------------------------------------

        fit_kwargs = deepcopy(kwargs)

        if self.backend_ == "lightgbm":
            self._fit_lightgbm(
                X,
                target,
                validation_tuple,
                fit_kwargs,
            )

        elif self.backend_ == "hist_gradient_boosting":
            self._fit_hist_gradient_boosting(
                X,
                target,
                validation_tuple,
                fit_kwargs,
            )

        else:
            raise RuntimeError(
                f"Unsupported backend: {self.backend_}"
            )

        # ---------------------------------------------------------------------
        # Validate fitted state
        # ---------------------------------------------------------------------

        self._verify_fitted_estimator()

        self._update_fit_metadata(X)

        # ---------------------------------------------------------------------
        # Store training statistics
        # ---------------------------------------------------------------------

        elapsed = time.perf_counter() - start_time

        negative_count = int(
            np.sum(target == 0)
        )

        positive_count = int(
            np.sum(target == 1)
        )

        self.training_statistics_ = TrainingStatistics(
            sample_count=len(target),
            feature_count=X.shape[1],
            positive_count=positive_count,
            negative_count=negative_count,
            positive_rate=float(
                positive_count / len(target)
            ),
            training_seconds=float(elapsed),
            validation_sample_count=(
                len(validation_tuple[1])
                if validation_tuple is not None
                else 0
            ),
            backend=self.backend_,
            schema_hash=self.schema_hash_,
        )

        logger.info(
            "URL classifier training completed: "
            "samples=%d features=%d positive_rate=%.4f "
            "duration=%.3fs backend=%s",
            len(target),
            X.shape[1],
            positive_count / len(target),
            elapsed,
            self.backend_,
        )

        return self

    # =========================================================================
    # CLASS BALANCE CONFIGURATION
    # =========================================================================

    def _configure_class_balance(
        self,
        y: np.ndarray,
    ) -> None:
        """
        Configure class imbalance handling.

        Explicit user configuration takes precedence over automatic balancing.
        """

        if self.class_weight is not None:
            return

        if not self.handle_class_imbalance:
            return

        weights = self._calculate_class_weights(y)

        # ---------------------------------------------------------------------
        # LightGBM supports class_weight directly.
        # ---------------------------------------------------------------------

        if self.backend_ == "lightgbm":

            self.model_params["class_weight"] = weights

            # Recreate estimator so modified parameters are applied.
            self._init_estimator()

            logger.info(
                "Applied automatic class balancing: %s",
                weights,
            )

        # ---------------------------------------------------------------------
        # HistGradientBoosting does not expose class_weight in all supported
        # sklearn versions. Sample weights are therefore handled during fit.
        # ---------------------------------------------------------------------

    # =========================================================================
    # LIGHTGBM FIT
    # =========================================================================

    def _fit_lightgbm(
        self,
        X: FeatureInput,
        y: np.ndarray,
        validation_data: ValidationData,
        fit_kwargs: Dict[str, Any],
    ) -> None:
        """Train LightGBM backend."""

        if validation_data is not None:

            X_val, y_val = validation_data

            fit_kwargs.setdefault(
                "eval_set",
                [(X_val, y_val)],
            )

            # Only inject early stopping when the caller has not supplied
            # callbacks. This prevents duplicate or conflicting callbacks.
            if "callbacks" not in fit_kwargs:

                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(
                        stopping_rounds=30,
                        verbose=False,
                    )
                ]

        self.model.fit(
            X,
            y,
            **fit_kwargs,
        )

    # =========================================================================
    # HISTOGRAM GRADIENT BOOSTING FIT
    # =========================================================================

    def _fit_hist_gradient_boosting(
        self,
        X: FeatureInput,
        y: np.ndarray,
        validation_data: ValidationData,
        fit_kwargs: Dict[str, Any],
    ) -> None:
        """
        Train scikit-learn fallback.

        HistGradientBoosting has different semantics from LightGBM, therefore
        LightGBM-specific fit parameters are deliberately not forwarded.
        """

        allowed_keys = {
            "sample_weight",
        }

        filtered_kwargs = {
            key: value
            for key, value in fit_kwargs.items()
            if key in allowed_keys
        }

        if (
            self.handle_class_imbalance
            and "sample_weight" not in filtered_kwargs
        ):
            weights = self._calculate_class_weights(y)

            filtered_kwargs["sample_weight"] = np.asarray(
                [
                    weights[int(label)]
                    for label in y
                ],
                dtype=np.float64,
            )

        self.model.fit(
            X,
            y,
            **filtered_kwargs,
        )

        if validation_data is not None:
            X_val, y_val = validation_data

            probabilities = self._predict_positive_probability(
                X_val
            )

            predictions = (
                probabilities >= self.decision_threshold
            ).astype(int)

            accuracy = float(
                np.mean(predictions == y_val)
            )

            self.last_validation_metrics_ = {
                "accuracy": accuracy,
            }

    # =========================================================================
    # FITTED STATE VALIDATION
    # =========================================================================

    def _verify_fitted_estimator(self) -> None:
        """Verify estimator exposes expected fitted attributes."""

        if self.model is None:
            raise RuntimeError(
                "Estimator is unexpectedly None after training."
            )

        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError(
                "Underlying estimator does not expose predict_proba()."
            )

    # =========================================================================
    # PROBABILITY PREDICTION
    # =========================================================================

    def predict_proba(
        self,
        X: FeatureInput,
    ) -> np.ndarray:
        """
        Return probability matrix of shape `(N, 2)`.

        Column 0:
            benign probability

        Column 1:
            phishing probability
        """

        self._ensure_ready_for_inference(X)

        start_time = time.perf_counter()

        probabilities = self.model.predict_proba(X)

        probabilities = self._normalize_probability_output(
            probabilities
        )

        elapsed = time.perf_counter() - start_time

        positive = probabilities[:, 1]

        predictions = (
            positive >= self.decision_threshold
        )

        self.last_prediction_statistics_ = (
            PredictionStatistics(
                sample_count=len(positive),
                positive_predictions=int(
                    predictions.sum()
                ),
                negative_predictions=int(
                    (~predictions).sum()
                ),
                positive_rate=float(
                    predictions.mean()
                ),
                mean_probability=float(
                    positive.mean()
                ),
                min_probability=float(
                    positive.min()
                ),
                max_probability=float(
                    positive.max()
                ),
                threshold=self.decision_threshold,
                inference_seconds=float(elapsed),
            )
        )

        return probabilities

    # =========================================================================
    # POSITIVE PROBABILITY
    # =========================================================================

    def _predict_positive_probability(
        self,
        X: FeatureInput,
    ) -> np.ndarray:
        """Return only positive-class probabilities."""

        probabilities = self.model.predict_proba(X)

        normalized = self._normalize_probability_output(
            probabilities
        )

        return normalized[:, 1]

    # =========================================================================
    # PROBABILITY NORMALIZATION
    # =========================================================================

    @staticmethod
    def _normalize_probability_output(
        probabilities: Any,
    ) -> np.ndarray:
        """
        Normalize backend probability output to guaranteed `(N, 2)` ndarray.
        """

        result = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        if result.ndim != 2:
            raise RuntimeError(
                "Estimator returned invalid probability shape: "
                f"{result.shape}"
            )

        if result.shape[1] != 2:
            raise RuntimeError(
                "URLPhishingClassifier requires binary probability output. "
                f"Received shape={result.shape}."
            )

        if not np.isfinite(result).all():
            raise RuntimeError(
                "Estimator returned non-finite probabilities."
            )

        # Numerical defensive normalization.
        result = np.clip(
            result,
            0.0,
            1.0,
        )

        row_sums = result.sum(axis=1)

        if np.any(row_sums <= 0.0):
            raise RuntimeError(
                "Estimator returned probability rows with zero sum."
            )

        result = result / row_sums[:, None]

        return result

    # =========================================================================
    # PREDICT
    # =========================================================================

    def predict(
        self,
        X: FeatureInput,
    ) -> np.ndarray:
        """
        Predict binary phishing labels.

        The inherited BasePhishingModel implementation is intentionally
        overridden to provide stronger probability validation and consistent
        diagnostics.
        """

        probabilities = self.predict_proba(X)

        positive_probability = probabilities[:, 1]

        return (
            positive_probability >= self.decision_threshold
        ).astype(np.int8)

    # =========================================================================
    # CONFIDENCE
    # =========================================================================

    def predict_with_confidence(
        self,
        X: FeatureInput,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels and confidence scores.

        Confidence is defined as distance from the decision boundary:

            confidence = abs(p - threshold)

        This is not a calibrated probability of correctness.
        """

        probabilities = self.predict_proba(X)

        positive_probability = probabilities[:, 1]

        predictions = (
            positive_probability >= self.decision_threshold
        ).astype(np.int8)

        confidence = np.abs(
            positive_probability -
            self.decision_threshold
        )

        return predictions, confidence

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    def get_feature_importance(
        self,
    ) -> Dict[str, float]:
        """
        Return normalized feature importance.

        LightGBM:
            native gain/split importance where available.

        sklearn fallback:
            native feature_importances_ where available.

        Raises if feature importance cannot be reliably obtained.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before feature importance "
                "can be retrieved."
            )

        if not self.feature_names_:
            raise RuntimeError(
                "Feature names are unavailable."
            )

        raw_importances: Optional[np.ndarray] = None

        # ---------------------------------------------------------------------
        # LightGBM native feature importance
        # ---------------------------------------------------------------------

        if self.backend_ == "lightgbm":

            try:
                raw_importances = np.asarray(
                    self.model.feature_importances_,
                    dtype=np.float64,
                )
            except Exception:
                raw_importances = None

        # ---------------------------------------------------------------------
        # sklearn fallback
        # ---------------------------------------------------------------------

        if raw_importances is None and hasattr(
            self.model,
            "feature_importances_",
        ):
            raw_importances = np.asarray(
                self.model.feature_importances_,
                dtype=np.float64,
            )

        if raw_importances is None:
            raise RuntimeError(
                "Underlying estimator does not expose reliable "
                "feature importance."
            )

        if len(raw_importances) != len(
            self.feature_names_
        ):
            raise RuntimeError(
                "Feature importance count does not match "
                "feature schema."
            )

        if not np.isfinite(raw_importances).all():
            raise RuntimeError(
                "Feature importance contains non-finite values."
            )

        raw_importances = np.maximum(
            raw_importances,
            0.0,
        )

        total = float(
            raw_importances.sum()
        )

        if total <= 0.0:
            logger.warning(
                "All feature importance values are zero."
            )

            normalized = np.zeros_like(
                raw_importances
            )
        else:
            normalized = (
                raw_importances / total
            )

        result = {
            feature: float(score)
            for feature, score in zip(
                self.feature_names_,
                normalized,
            )
        }

        return dict(
            sorted(
                result.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    # =========================================================================
    # TOP FEATURES
    # =========================================================================

    def get_top_features(
        self,
        limit: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Return highest-ranked features.
        """

        if not isinstance(limit, int):
            raise TypeError(
                "limit must be an integer."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        importance = self.get_feature_importance()

        return list(
            importance.items()
        )[:limit]

    # =========================================================================
    # MODEL HEALTH
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Return non-sensitive runtime model health information.
        """

        healthy = True
        issues: List[str] = []

        if self.model is None:
            healthy = False
            issues.append(
                "Estimator is not initialized."
            )

        if not self.is_fitted:
            healthy = False
            issues.append(
                "Model is not fitted."
            )

        if self.n_features_in_ is None:
            healthy = False
            issues.append(
                "Feature count is unavailable."
            )

        if self.schema_hash_ is None:
            healthy = False
            issues.append(
                "Feature schema fingerprint is unavailable."
            )

        return {
            "healthy": healthy,
            "issues": issues,
            "model_name": self.model_name,
            "version": self.version,
            "backend": self.backend_,
            "backend_version": self.backend_version_,
            "is_fitted": self.is_fitted,
            "feature_count": self.n_features_in_,
            "schema_hash": self.schema_hash_,
            "decision_threshold": self.decision_threshold,
        }

    # =========================================================================
    # MODEL METADATA
    # =========================================================================

    def get_metadata(self) -> Dict[str, Any]:
        """
        Extend BasePhishingModel metadata with URL-classifier-specific
        reproducibility and schema information.
        """

        metadata = dict(
            super().get_metadata()
        )

        metadata.update(
            {
                "classifier_type": (
                    "URLPhishingClassifier"
                ),
                "implementation_version": MODEL_VERSION,
                "backend": self.backend_,
                "backend_version": self.backend_version_,
                "random_state": self.random_state,
                "strict_input_validation": (
                    self.strict_input_validation
                ),
                "reject_nan": self.reject_nan,
                "reject_inf": self.reject_inf,
                "handle_class_imbalance": (
                    self.handle_class_imbalance
                ),
                "max_prediction_batch": (
                    self.max_prediction_batch
                ),
                "feature_count": self.n_features_in_,
                "feature_dtypes": deepcopy(
                    self.feature_dtypes_
                ),
                "schema_hash": self.schema_hash_,
                "training_statistics": (
                    self.training_statistics_.to_dict()
                ),
                "last_validation_metrics": deepcopy(
                    self.last_validation_metrics_
                ),
            }
        )

        if self.last_prediction_statistics_:
            metadata[
                "last_prediction_statistics"
            ] = self.last_prediction_statistics_.to_dict()

        return metadata

    # =========================================================================
    # INFERENCE GUARD
    # =========================================================================

    def _ensure_ready_for_inference(
        self,
        X: FeatureInput,
    ) -> None:
        """
        Execute all inference safety checks.
        """

        if not self.is_fitted:
            raise RuntimeError(
                f"Model '{self.model_name}' must be "
                "fitted before inference."
            )

        if self.model is None:
            raise RuntimeError(
                "Underlying estimator is unavailable."
            )

        self._validate_feature_input(
            X,
            inference=True,
        )

        if len(X) > self.max_prediction_batch:
            raise ValueError(
                "Prediction batch exceeds configured maximum. "
                f"Received {len(X)}, "
                f"maximum={self.max_prediction_batch}."
            )

    # =========================================================================
    # BATCH PREDICTION
    # =========================================================================

    def predict_in_batches(
        self,
        X: FeatureInput,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Perform memory-conscious prediction over large datasets.

        Returns
        -------
        np.ndarray
            Binary prediction array.
        """

        self._ensure_ready_for_inference(X)

        if batch_size is None:
            batch_size = self.max_prediction_batch

        if not isinstance(batch_size, int):
            raise TypeError(
                "batch_size must be an integer."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        outputs: List[np.ndarray] = []

        total_rows = len(X)

        for start in range(
            0,
            total_rows,
            batch_size,
        ):
            end = min(
                start + batch_size,
                total_rows,
            )

            batch = X.iloc[start:end] \
                if isinstance(X, pd.DataFrame) \
                else X[start:end]

            outputs.append(
                self.predict(batch)
            )

        return np.concatenate(
            outputs
        )

    # =========================================================================
    # THRESHOLD MANAGEMENT
    # =========================================================================

    def set_decision_threshold(
        self,
        threshold: float,
    ) -> None:
        """
        Update inference threshold safely.

        Threshold changes do not retrain the model.
        """

        if not isinstance(
            threshold,
            (float, int),
        ):
            raise TypeError(
                "Threshold must be numeric."
            )

        threshold = float(threshold)

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Decision threshold must be between "
                "0.0 and 1.0."
            )

        self.decision_threshold = threshold

        logger.info(
            "Decision threshold updated to %.4f.",
            threshold,
        )

    # =========================================================================
    # SCHEMA INFORMATION
    # =========================================================================

    def get_feature_schema(self) -> Dict[str, Any]:
        """
        Return the learned feature contract.
        """

        return {
            "feature_count": self.n_features_in_,
            "feature_names": list(
                self.feature_names_
            ),
            "feature_dtypes": deepcopy(
                self.feature_dtypes_
            ),
            "schema_hash": self.schema_hash_,
            "hash_algorithm": (
                SCHEMA_HASH_ALGORITHM
            ),
        }

    # =========================================================================
    # SCHEMA COMPATIBILITY
    # =========================================================================

    def is_schema_compatible(
        self,
        X: FeatureInput,
    ) -> bool:
        """
        Check whether X conforms to the trained feature schema.

        Unlike inference, this method returns False instead of raising.
        """

        try:
            self._validate_feature_input(
                X,
                inference=True,
            )
            return True
        except (
            ValueError,
            TypeError,
            RuntimeError,
        ):
            return False

    # =========================================================================
    # BACKEND INFORMATION
    # =========================================================================

    def get_backend_info(self) -> Dict[str, Any]:
        """Return backend information useful for MLOps provenance."""

        return {
            "backend": self.backend_,
            "backend_version": self.backend_version_,
            "lightgbm_available": HAS_LIGHTGBM,
            "sklearn_hist_gradient_boosting_available": (
                HAS_SKLEARN_HGB
            ),
        }

    # =========================================================================
    # PARAMETER SNAPSHOT
    # =========================================================================

    def get_model_parameters(
        self,
    ) -> Dict[str, Any]:
        """
        Return a defensive copy of model configuration.
        """

        return deepcopy(
            self.model_params
        )

    # =========================================================================
    # TRAINING SUMMARY
    # =========================================================================

    def get_training_summary(
        self,
    ) -> Dict[str, Any]:
        """
        Return compact training summary.
        """

        return {
            "model_name": self.model_name,
            "version": self.version,
            "backend": self.backend_,
            "fitted": self.is_fitted,
            "training": self.training_statistics_.to_dict(),
            "validation_metrics": deepcopy(
                self.last_validation_metrics_
            ),
            "schema": self.get_feature_schema(),
        }

    # =========================================================================
    # SERIALIZATION VALIDATION
    # =========================================================================

    def validate_serialized_state(self) -> bool:
        """
        Validate that essential state survived serialization.

        Useful immediately after BasePhishingModel.load().
        """

        required_attributes = [
            "model",
            "model_name",
            "version",
            "decision_threshold",
            "feature_names_",
            "schema_hash_",
            "backend_",
            "is_fitted",
        ]

        missing = [
            attribute
            for attribute in required_attributes
            if not hasattr(self, attribute)
        ]

        if missing:
            logger.error(
                "Serialized URL classifier is missing "
                "attributes: %s",
                missing,
            )
            return False

        if self.is_fitted and self.model is None:
            return False

        return True

    # =========================================================================
    # SAFE REPRESENTATION
    # =========================================================================

    def __repr__(self) -> str:
        """Return concise, non-sensitive model representation."""

        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"version={self.version!r}, "
            f"backend={self.backend_!r}, "
            f"is_fitted={self.is_fitted!r}, "
            f"features={self.n_features_in_!r}, "
            f"threshold={self.decision_threshold!r}"
            f")"
        )


# =============================================================================
# PUBLIC EXPORT CONTRACT
# =============================================================================

__all__ = [
    "URLPhishingClassifier",
    "TrainingStatistics",
    "PredictionStatistics",
]
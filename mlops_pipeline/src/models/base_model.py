"""
PhishGuard AI - Production Model Contract & Lifecycle Engine
=============================================================

Version: 14.0.0
File Path: mlops_pipeline/src/models/base_model.py

Purpose
-------
Enterprise-grade abstract model contract used by the PhishGuard AI MLOps
ecosystem.

This module defines the common lifecycle and safety boundary for all phishing
classification models:

    initialize
        |
        v
    validate training input
        |
        v
    subclass.fit(...)
        |
        v
    _update_fit_metadata(...)
        |
        v
    validate inference input
        |
        v
    predict_proba(...)
        |
        v
    predict(...)
        |
        v
    persist + integrity manifest
        |
        v
    verified loading

Security Design
---------------
The following principles are enforced:

1. Never silently accept malformed model inputs.
2. Never silently reorder model features.
3. Never silently accept non-finite numeric values.
4. Validate binary classification outputs.
5. Validate decision thresholds.
6. Track training and inference provenance.
7. Produce deterministic metadata.
8. Generate SHA-256 artifact integrity hashes.
9. Use atomic artifact writes where possible.
10. Refuse unsafe/untrusted artifact loading by default.
11. Detect model/environment version drift.
12. Keep model persistence auditable.
13. Avoid leaking sensitive input data into logs.
14. Preserve compatibility with existing subclasses.
15. Provide extension points for model-specific implementations.

Important Security Note
-----------------------
Joblib uses pickle-compatible serialization. Loading an untrusted joblib file
can execute arbitrary Python code.

Therefore:

    BasePhishingModel.load(...)

requires explicit trust acknowledgement by default.

For cross-trust-boundary model exchange, prefer a safer serialization format
such as skops.io where supported, or export the serving model to ONNX where
appropriate.

This class does NOT attempt to sandbox pickle/joblib because Python pickle
cannot be made safely sandboxed merely by wrapping joblib.load().

Architecture
------------
BasePhishingModel
    |
    +-- URLPhishingClassifier
    +-- EmailPhishingClassifier
    +-- AttachmentPhishingClassifier
    +-- future multimodal classifiers

The implementation intentionally avoids imposing a specific ML framework.
Scikit-learn, LightGBM, XGBoost, CatBoost, PyTorch wrappers, or other
estimators can implement the abstract methods.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
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

import joblib
import numpy as np
import pandas as pd

from src.utils.logger import get_logger


logger = get_logger("src.models.base_model")


# ==============================================================================
# MODULE CONSTANTS
# ==============================================================================

MODEL_CONTRACT_VERSION = "14.0.0"

DEFAULT_MODEL_VERSION = "14.0.0"
DEFAULT_DECISION_THRESHOLD = 0.50

MIN_DECISION_THRESHOLD = 0.0
MAX_DECISION_THRESHOLD = 1.0

POSITIVE_CLASS = 1
NEGATIVE_CLASS = 0

DEFAULT_ARTIFACT_EXTENSION = ".joblib"
DEFAULT_METADATA_SUFFIX = "_metadata.json"
DEFAULT_MANIFEST_SUFFIX = "_manifest.json"

SHA256_HEX_LENGTH = 64

# Maximum amount of information allowed in model names and versions.
# This prevents path traversal and problematic artifact names.
_SAFE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


# ==============================================================================
# TYPE ALIASES
# ==============================================================================

ArrayLike = Union[
    pd.DataFrame,
    pd.Series,
    np.ndarray,
    Sequence[Any],
]

FeatureMatrix = Union[
    pd.DataFrame,
    np.ndarray,
]

TargetVector = Union[
    pd.Series,
    np.ndarray,
    Sequence[Any],
]

ValidationData = Tuple[FeatureMatrix, TargetVector]


# ==============================================================================
# EXCEPTIONS
# ==============================================================================


class ModelError(RuntimeError):
    """Base exception for PhishGuard model lifecycle failures."""


class ModelNotFittedError(ModelError):
    """Raised when an operation requires a fitted model."""


class ModelInputError(ValueError):
    """Raised when model input violates the model contract."""


class ModelArtifactError(ModelError):
    """Raised when a persisted model artifact is invalid or unsafe."""


class ModelIntegrityError(ModelArtifactError):
    """Raised when model artifact integrity verification fails."""


class ModelCompatibilityError(ModelArtifactError):
    """Raised when an artifact is incompatible with the current environment."""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _utc_now_iso() -> str:
    """
    Return the current UTC timestamp in ISO-8601 format.

    The trailing +00:00 representation is intentionally retained rather than
    converting to local time.
    """
    return datetime.now(timezone.utc).isoformat()


def _validate_identifier(value: str, field_name: str) -> str:
    """
    Validate a model identifier used in artifact names.

    Prevents values such as:

        ../../malicious/path
        C:\\Windows\\...
        /tmp/model
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    cleaned = value.strip()

    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")

    if not _SAFE_IDENTIFIER_PATTERN.fullmatch(cleaned):
        raise ValueError(
            f"Invalid {field_name} '{value}'. "
            "Only letters, digits, '.', '_' and '-' are permitted."
        )

    return cleaned


def _safe_json_value(value: Any) -> Any:
    """
    Convert arbitrary Python values into JSON-safe representations.

    This function is intentionally conservative. It is used for metadata,
    not for reconstructing executable objects.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else str(numeric)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _safe_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_safe_json_value(item) for item in value]

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _safe_json_value(value.item())
        return [_safe_json_value(item) for item in value.tolist()]

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _canonical_json(data: Mapping[str, Any]) -> bytes:
    """
    Produce canonical JSON bytes.

    Canonical serialization is useful for metadata fingerprints and provenance.
    """
    payload = json.dumps(
        _safe_json_value(data),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )

    return payload.encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    """Return SHA-256 hex digest for bytes."""
    return hashlib.sha256(data).hexdigest()


def _sha256_file(filepath: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """
    Calculate SHA-256 using streaming reads.

    A 1 MiB buffer provides a reasonable balance between throughput and memory
    consumption for large model artifacts.
    """
    path = Path(filepath)

    if not path.is_file():
        raise FileNotFoundError(f"Artifact file does not exist: '{path}'.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _atomic_write_json(
    filepath: Union[str, Path],
    payload: Mapping[str, Any],
) -> Path:
    """
    Atomically write JSON metadata.

    Writes to a temporary file in the same directory and then replaces the
    destination. This avoids leaving partially written metadata when a process
    crashes during serialization.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = json.dumps(
        _safe_json_value(payload),
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_name, path)

    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass

        raise

    return path


def _atomic_joblib_dump(
    obj: Any,
    filepath: Union[str, Path],
    *,
    compress: Union[int, bool] = 3,
    protocol: int = 5,
) -> Path:
    """
    Atomically persist a joblib artifact.

    Important:
        This provides crash consistency, NOT cryptographic security.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )

    os.close(fd)

    try:
        joblib.dump(
            obj,
            temporary_name,
            compress=compress,
            protocol=protocol,
        )

        # Force replacement of the previous artifact.
        os.replace(temporary_name, path)

    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass

        raise

    return path


def _get_library_versions() -> Dict[str, str]:
    """
    Collect versions of important runtime libraries.

    Missing optional packages are simply omitted.
    """
    versions: Dict[str, str] = {}

    try:
        import sklearn

        versions["scikit-learn"] = sklearn.__version__
    except Exception:
        pass

    try:
        versions["numpy"] = np.__version__
    except Exception:
        pass

    try:
        versions["pandas"] = pd.__version__
    except Exception:
        pass

    try:
        versions["joblib"] = joblib.__version__
    except Exception:
        pass

    return versions


def _fingerprint_feature_names(feature_names: Sequence[str]) -> str:
    """
    Generate a deterministic fingerprint for feature ordering.

    Feature order is part of the model contract.
    """
    normalized = [str(name) for name in feature_names]

    payload = json.dumps(
        normalized,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return _sha256_bytes(payload)


def _fingerprint_target(y: TargetVector) -> str:
    """
    Generate a lightweight deterministic fingerprint for target values.

    This is intended for provenance rather than secret-data protection.

    For very large targets the representation is still materialized into a
    NumPy array, so callers should use this only for normal ML datasets.
    """
    if isinstance(y, pd.Series):
        values = y.to_numpy(copy=True)
    else:
        values = np.asarray(y)

    if values.ndim != 1:
        values = values.reshape(-1)

    try:
        payload = values.tobytes()
    except Exception:
        payload = repr(values.tolist()).encode("utf-8")

    return _sha256_bytes(payload)


# ==============================================================================
# BASE MODEL
# ==============================================================================


class BasePhishingModel(ABC):
    """
    Abstract production model contract for PhishGuard AI.

    The class intentionally contains only framework-neutral functionality.
    Domain-specific model classes remain responsible for implementing:

        fit()
        predict_proba()
        get_feature_importance()

    The base class controls:

        - lifecycle state
        - input validation
        - probability validation
        - threshold handling
        - feature contract tracking
        - provenance
        - model metadata
        - persistence
        - integrity manifests
        - safe loading policy
        - compatibility checks
        - prediction audit information
    """

    # --------------------------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------------------------

    def __init__(
        self,
        model_name: str = "base_phishing_model",
        version: str = DEFAULT_MODEL_VERSION,
        decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
        model_params: Optional[Dict[str, Any]] = None,
        *,
        strict_input_validation: bool = True,
        allow_extra_features: bool = False,
        require_feature_order: bool = True,
        expected_classes: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Initialize the model contract.

        Parameters
        ----------
        model_name:
            Stable model architecture identifier.

        version:
            Model version.

        decision_threshold:
            Positive-class threshold in [0, 1].

        model_params:
            Underlying estimator hyperparameters.

        strict_input_validation:
            If True, reject malformed/non-finite inference input.

        allow_extra_features:
            If False, inference requires exactly the learned feature set.

        require_feature_order:
            If True, DataFrame feature order must match training order.

        expected_classes:
            Expected binary class labels. Defaults to (0, 1).
        """
        self.model_name = _validate_identifier(
            model_name,
            "model_name",
        )

        self.version = _validate_identifier(
            version,
            "version",
        )

        self.decision_threshold = self._validate_threshold(
            decision_threshold
        )

        self.model_params: Dict[str, Any] = dict(
            model_params if model_params is not None else {}
        )

        self.strict_input_validation = bool(strict_input_validation)
        self.allow_extra_features = bool(allow_extra_features)
        self.require_feature_order = bool(require_feature_order)

        classes = expected_classes or (
            NEGATIVE_CLASS,
            POSITIVE_CLASS,
        )

        normalized_classes = tuple(int(value) for value in classes)

        if normalized_classes != (0, 1):
            raise ValueError(
                "PhishGuard binary classifiers must use expected_classes=(0, 1)."
            )

        self.expected_classes_: Tuple[int, ...] = normalized_classes

        # ----------------------------------------------------------------------
        # Lifecycle state
        # ----------------------------------------------------------------------

        self.is_fitted: bool = False
        self.lifecycle_state: str = "initialized"

        # ----------------------------------------------------------------------
        # Underlying estimator
        # ----------------------------------------------------------------------

        self.model: Any = None

        # ----------------------------------------------------------------------
        # Feature contract
        # ----------------------------------------------------------------------

        self.feature_names_: List[str] = []
        self.feature_count_: int = 0
        self.feature_fingerprint_: Optional[str] = None

        # ----------------------------------------------------------------------
        # Training provenance
        # ----------------------------------------------------------------------

        self.fitted_timestamp_: Optional[str] = None
        self.training_started_timestamp_: Optional[str] = None
        self.training_duration_seconds_: Optional[float] = None

        self.training_row_count_: Optional[int] = None
        self.training_target_fingerprint_: Optional[str] = None

        self.training_dataset_id_: Optional[str] = None
        self.training_dataset_version_: Optional[str] = None

        # ----------------------------------------------------------------------
        # Model metadata
        # ----------------------------------------------------------------------

        self.training_metadata_: Dict[str, Any] = {}
        self.evaluation_metadata_: Dict[str, Any] = {}
        self.calibration_metadata_: Dict[str, Any] = {}

        # ----------------------------------------------------------------------
        # Artifact information
        # ----------------------------------------------------------------------

        self.artifact_sha256_: Optional[str] = None
        self.artifact_path_: Optional[str] = None
        self.artifact_created_timestamp_: Optional[str] = None

        # ----------------------------------------------------------------------
        # Runtime environment
        # ----------------------------------------------------------------------

        self.environment_metadata_: Dict[str, Any] = {}

        # ----------------------------------------------------------------------
        # Internal runtime statistics
        # ----------------------------------------------------------------------

        self.prediction_count_: int = 0
        self.last_prediction_timestamp_: Optional[str] = None
        self.last_prediction_duration_seconds_: Optional[float] = None

    # ==========================================================================
    # ABSTRACT CONTRACT
    # ==========================================================================

    @abstractmethod
    def fit(
        self,
        X: FeatureMatrix,
        y: TargetVector,
        validation_data: Optional[ValidationData] = None,
        **kwargs: Any,
    ) -> "BasePhishingModel":
        """
        Train the underlying model.

        Subclasses MUST call:

            self._update_fit_metadata(X, y)

        after successful training.

        Parameters
        ----------
        X:
            Training feature matrix.

        y:
            Binary target vector.

        validation_data:
            Optional validation dataset.

        kwargs:
            Model-specific training options.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(
        self,
        X: FeatureMatrix,
    ) -> np.ndarray:
        """
        Return binary class probabilities.

        Required output shape:

            (n_samples, 2)

        Column 0:
            P(class=0)

        Column 1:
            P(class=1)

        Subclasses may internally return different estimator outputs, but the
        public implementation must conform to this contract.
        """
        raise NotImplementedError

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature importance mapped to feature names.
        """
        raise NotImplementedError

    # ==========================================================================
    # THRESHOLD MANAGEMENT
    # ==========================================================================

    @staticmethod
    def _validate_threshold(value: float) -> float:
        """Validate and normalize a classification threshold."""
        if isinstance(value, bool):
            raise TypeError("decision_threshold must be a numeric value.")

        try:
            threshold = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "decision_threshold must be numeric."
            ) from exc

        if not math.isfinite(threshold):
            raise ValueError(
                "decision_threshold must be finite."
            )

        if not (
            MIN_DECISION_THRESHOLD
            <= threshold
            <= MAX_DECISION_THRESHOLD
        ):
            raise ValueError(
                "decision_threshold must be between 0.0 and 1.0."
            )

        return threshold

    def set_decision_threshold(
        self,
        threshold: float,
    ) -> float:
        """
        Safely update the positive-class decision threshold.

        Returns
        -------
        float
            Validated threshold.
        """
        validated = self._validate_threshold(threshold)

        self.decision_threshold = validated

        logger.info(
            "Decision threshold updated for model '%s' to %.6f.",
            self.model_name,
            validated,
        )

        return validated

    # ==========================================================================
    # INPUT VALIDATION
    # ==========================================================================

    @staticmethod
    def _validate_feature_names(
        feature_names: Sequence[str],
    ) -> List[str]:
        """Validate feature names and prevent duplicate columns."""
        normalized = [str(name) for name in feature_names]

        if not normalized:
            raise ModelInputError(
                "Feature matrix contains zero features."
            )

        duplicates = sorted(
            {
                name
                for name in normalized
                if normalized.count(name) > 1
            }
        )

        if duplicates:
            raise ModelInputError(
                f"Duplicate feature names detected: {duplicates}"
            )

        return normalized

    def _extract_feature_names(
        self,
        X: FeatureMatrix,
    ) -> List[str]:
        """
        Extract feature names from a DataFrame or generate stable names for
        NumPy matrices.
        """
        if isinstance(X, pd.DataFrame):
            return self._validate_feature_names(
                list(X.columns)
            )

        array = np.asarray(X)

        if array.ndim != 2:
            raise ModelInputError(
                "Feature matrix must be two-dimensional."
            )

        return [
            f"feature_{index}"
            for index in range(array.shape[1])
        ]

    def _validate_feature_matrix(
        self,
        X: FeatureMatrix,
        *,
        require_fitted: bool = False,
    ) -> None:
        """
        Validate shape, feature count, names, and finite values.

        This method does not mutate the caller's DataFrame.
        """
        if X is None:
            raise ModelInputError(
                "Feature matrix cannot be None."
            )

        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ModelInputError(
                "Feature matrix must be pandas.DataFrame or numpy.ndarray."
            )

        if isinstance(X, pd.DataFrame):
            if X.ndim != 2:
                raise ModelInputError(
                    "Feature DataFrame must be two-dimensional."
                )

            if X.shape[1] == 0:
                raise ModelInputError(
                    "Feature DataFrame contains zero columns."
                )

            feature_names = self._validate_feature_names(
                list(X.columns)
            )

            if self.strict_input_validation:
                numeric_columns = X.select_dtypes(
                    include=[np.number]
                )

                if not numeric_columns.empty:
                    values = numeric_columns.to_numpy(
                        dtype=float,
                        copy=False,
                    )

                    if not np.isfinite(values).all():
                        raise ModelInputError(
                            "Feature matrix contains NaN or infinite numeric values."
                        )

        else:
            array = np.asarray(X)

            if array.ndim != 2:
                raise ModelInputError(
                    "NumPy feature matrix must be two-dimensional."
                )

            if array.shape[1] == 0:
                raise ModelInputError(
                    "Feature matrix contains zero columns."
                )

            if self.strict_input_validation:
                if not np.issubdtype(
                    array.dtype,
                    np.number,
                ):
                    raise ModelInputError(
                        "NumPy feature matrix must contain numeric values."
                    )

                if not np.isfinite(array).all():
                    raise ModelInputError(
                        "Feature matrix contains NaN or infinite values."
                    )

            feature_names = self._extract_feature_names(X)

        if require_fitted and not self.is_fitted:
            raise ModelNotFittedError(
                f"Model '{self.model_name}' is not fitted."
            )

        if self.is_fitted:
            self._validate_inference_features(
                feature_names
            )

    def _validate_inference_features(
        self,
        feature_names: Sequence[str],
    ) -> None:
        """
        Detect feature count/order/schema drift between training and inference.
        """
        incoming = list(feature_names)
        expected = list(self.feature_names_)

        if not expected:
            return

        if self.allow_extra_features:
            missing = [
                name
                for name in expected
                if name not in incoming
            ]

            if missing:
                raise ModelInputError(
                    f"Required model features are missing: {missing}"
                )

            return

        if len(incoming) != len(expected):
            raise ModelInputError(
                "Feature count mismatch. "
                f"Expected {len(expected)}, received {len(incoming)}."
            )

        if self.require_feature_order:
            if incoming != expected:
                raise ModelInputError(
                    "Feature order mismatch. "
                    "The inference feature order does not match the "
                    "training feature contract."
                )
        else:
            if set(incoming) != set(expected):
                raise ModelInputError(
                    "Feature set mismatch between training and inference."
                )

    @staticmethod
    def _validate_target(
        y: TargetVector,
        expected_length: Optional[int] = None,
    ) -> np.ndarray:
        """
        Validate binary target labels.

        Only labels 0 and 1 are accepted.
        """
        if y is None:
            raise ModelInputError(
                "Target vector cannot be None."
            )

        if isinstance(y, pd.Series):
            values = y.to_numpy()
        else:
            values = np.asarray(y)

        if values.ndim != 1:
            values = values.reshape(-1)

        if expected_length is not None:
            if len(values) != expected_length:
                raise ModelInputError(
                    "Target length does not match feature row count. "
                    f"Expected {expected_length}, got {len(values)}."
                )

        if len(values) == 0:
            raise ModelInputError(
                "Target vector is empty."
            )

        if pd.isna(values).any():
            raise ModelInputError(
                "Target vector contains missing values."
            )

        unique_values = set(
            np.asarray(values).tolist()
        )

        if not unique_values.issubset({0, 1}):
            raise ModelInputError(
                "Target vector must contain only binary labels 0 and 1. "
                f"Observed labels: {sorted(unique_values, key=str)}"
            )

        if len(unique_values) < 2:
            raise ModelInputError(
                "Training target must contain both classes 0 and 1."
            )

        return values.astype(np.int8, copy=False)

    # ==========================================================================
    # FIT METADATA
    # ==========================================================================

    def begin_training(self) -> None:
        """Mark the beginning of a training operation."""
        self.training_started_timestamp_ = _utc_now_iso()
        self.lifecycle_state = "training"

    def _update_fit_metadata(
        self,
        X: FeatureMatrix,
        y: Optional[TargetVector] = None,
        *,
        dataset_id: Optional[str] = None,
        dataset_version: Optional[str] = None,
        training_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Update model metadata after successful training.

        Existing subclasses that only call:

            _update_fit_metadata(X)

        remain supported.
        """
        self._validate_feature_matrix(X)

        if y is not None:
            target = self._validate_target(
                y,
                expected_length=len(X),
            )
            self.training_target_fingerprint_ = _fingerprint_target(
                target
            )

        feature_names = self._extract_feature_names(X)

        self.feature_names_ = feature_names
        self.feature_count_ = len(feature_names)
        self.feature_fingerprint_ = _fingerprint_feature_names(
            feature_names
        )

        self.training_row_count_ = int(len(X))

        self.training_dataset_id_ = dataset_id
        self.training_dataset_version_ = dataset_version

        if training_metadata:
            self.training_metadata_.update(
                _safe_json_value(dict(training_metadata))
            )

        self.is_fitted = True
        self.lifecycle_state = "fitted"

        self.fitted_timestamp_ = _utc_now_iso()

        if self.training_started_timestamp_:
            try:
                started = datetime.fromisoformat(
                    self.training_started_timestamp_
                )
                finished = datetime.fromisoformat(
                    self.fitted_timestamp_
                )

                self.training_duration_seconds_ = max(
                    0.0,
                    (
                        finished - started
                    ).total_seconds(),
                )
            except Exception:
                self.training_duration_seconds_ = None

        self.environment_metadata_ = self._collect_environment_metadata()

        logger.info(
            "Model '%s' successfully marked as fitted "
            "(rows=%d, features=%d).",
            self.model_name,
            self.training_row_count_,
            self.feature_count_,
        )

    def finalize_training(
        self,
        X: FeatureMatrix,
        y: Optional[TargetVector] = None,
        **metadata: Any,
    ) -> "BasePhishingModel":
        """
        Convenience method for subclasses to call after fitting.
        """
        self._update_fit_metadata(
            X,
            y,
            training_metadata=metadata,
        )

        return self

    # ==========================================================================
    # PREDICTION CONTRACT
    # ==========================================================================

    def _validated_predict_proba(
        self,
        X: FeatureMatrix,
    ) -> np.ndarray:
        """
        Internal prediction wrapper that validates subclass output.

        Subclasses should implement predict_proba(). For strongest enforcement,
        subclasses may call this helper from their public implementation, or
        callers can use predict_proba_checked().
        """
        self._validate_feature_matrix(
            X,
            require_fitted=True,
        )

        start = time.perf_counter()

        probabilities = self.predict_proba(X)

        duration = time.perf_counter() - start

        self._validate_probability_output(
            probabilities,
            expected_rows=len(X),
        )

        self._record_prediction(
            duration
        )

        return np.asarray(
            probabilities,
            dtype=np.float64,
        )

    @staticmethod
    def _validate_probability_output(
        probabilities: Any,
        *,
        expected_rows: Optional[int] = None,
    ) -> None:
        """Validate binary probability matrix."""
        if probabilities is None:
            raise ModelError(
                "predict_proba() returned None."
            )

        array = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        if array.ndim != 2:
            raise ModelError(
                "predict_proba() must return a 2D array "
                "with shape (n_samples, 2)."
            )

        if array.shape[1] != 2:
            raise ModelError(
                "Binary predict_proba() must return exactly "
                "two probability columns."
            )

        if expected_rows is not None:
            if array.shape[0] != expected_rows:
                raise ModelError(
                    "Probability row count does not match input row count."
                )

        if not np.isfinite(array).all():
            raise ModelError(
                "predict_proba() returned NaN or infinite probabilities."
            )

        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ModelError(
                "predict_proba() returned probabilities outside [0, 1]."
            )

        row_sums = array.sum(axis=1)

        if not np.allclose(
            row_sums,
            1.0,
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ModelError(
                "Each probability row must sum to approximately 1.0."
            )

    def predict_proba_checked(
        self,
        X: FeatureMatrix,
    ) -> np.ndarray:
        """
        Public hardened probability prediction API.

        This method should be preferred by orchestration/serving layers when
        strict output validation is required.
        """
        return self._validated_predict_proba(X)

    def predict(
        self,
        X: FeatureMatrix,
    ) -> np.ndarray:
        """
        Predict binary phishing labels using the configured threshold.

        Returns
        -------
        np.ndarray
            Array containing only 0 and 1.
        """
        probabilities = self._validated_predict_proba(X)

        positive_probability = probabilities[:, POSITIVE_CLASS]

        predictions = (
            positive_probability
            >= self.decision_threshold
        ).astype(np.int8)

        return predictions

    def predict_with_scores(
        self,
        X: FeatureMatrix,
    ) -> pd.DataFrame:
        """
        Return predictions together with calibrated probability scores.

        Output columns:

            phishing_probability
            predicted_label
            decision_threshold
        """
        probabilities = self._validated_predict_proba(X)

        positive_probability = probabilities[:, POSITIVE_CLASS]

        predictions = (
            positive_probability
            >= self.decision_threshold
        ).astype(np.int8)

        return pd.DataFrame(
            {
                "phishing_probability": positive_probability,
                "predicted_label": predictions,
                "decision_threshold": self.decision_threshold,
            }
        )

    def _record_prediction(
        self,
        duration_seconds: float,
    ) -> None:
        """Record non-sensitive prediction telemetry."""
        self.prediction_count_ += 1
        self.last_prediction_timestamp_ = _utc_now_iso()
        self.last_prediction_duration_seconds_ = float(
            max(0.0, duration_seconds)
        )

    # ==========================================================================
    # FEATURE IMPORTANCE
    # ==========================================================================

    def get_validated_feature_importance(
        self,
    ) -> Dict[str, float]:
        """
        Validate the subclass feature importance contract.
        """
        if not self.is_fitted:
            raise ModelNotFittedError(
                f"Model '{self.model_name}' must be fitted before "
                "feature importance can be requested."
            )

        raw = self.get_feature_importance()

        if raw is None:
            raise ModelError(
                "get_feature_importance() returned None."
            )

        if not isinstance(raw, Mapping):
            raise ModelError(
                "Feature importance must be a mapping."
            )

        result: Dict[str, float] = {}

        for feature, importance in raw.items():
            feature_name = str(feature)

            try:
                numeric_importance = float(importance)
            except (TypeError, ValueError) as exc:
                raise ModelError(
                    f"Invalid importance for feature '{feature_name}'."
                ) from exc

            if not math.isfinite(numeric_importance):
                raise ModelError(
                    f"Non-finite importance for feature '{feature_name}'."
                )

            result[feature_name] = numeric_importance

        return result

    # ==========================================================================
    # CALIBRATION / EVALUATION METADATA
    # ==========================================================================

    def set_calibration_metadata(
        self,
        *,
        method: Optional[str] = None,
        validation_dataset_id: Optional[str] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        calibrated: bool = False,
        **extra: Any,
    ) -> None:
        """
        Record probability calibration provenance.

        This does not perform calibration itself. It records the result of an
        external calibration procedure such as Platt scaling or isotonic
        calibration.
        """
        payload: Dict[str, Any] = {
            "calibrated": bool(calibrated),
            "method": method,
            "validation_dataset_id": validation_dataset_id,
            "metrics": dict(metrics or {}),
        }

        payload.update(extra)

        self.calibration_metadata_ = _safe_json_value(
            payload
        )

    def set_evaluation_metadata(
        self,
        metrics: Mapping[str, Any],
        *,
        dataset_id: Optional[str] = None,
        dataset_version: Optional[str] = None,
    ) -> None:
        """Store model evaluation metrics and evaluation provenance."""
        if not isinstance(metrics, Mapping):
            raise TypeError(
                "metrics must be a mapping."
            )

        self.evaluation_metadata_ = _safe_json_value(
            {
                "dataset_id": dataset_id,
                "dataset_version": dataset_version,
                "metrics": dict(metrics),
                "recorded_at": _utc_now_iso(),
            }
        )

    # ==========================================================================
    # ENVIRONMENT / PROVENANCE
    # ==========================================================================

    @staticmethod
    def _collect_environment_metadata() -> Dict[str, Any]:
        """
        Collect runtime information required for reproducibility audits.
        """
        return {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "architecture": platform.architecture()[0],
            "library_versions": _get_library_versions(),
        }

    def get_environment_metadata(
        self,
    ) -> Dict[str, Any]:
        """Return a fresh runtime environment snapshot."""
        return self._collect_environment_metadata()

    # ==========================================================================
    # METADATA
    # ==========================================================================

    def get_metadata(self) -> Dict[str, Any]:
        """
        Generate comprehensive JSON-safe model metadata.

        No raw training examples, email bodies, URLs, attachment contents,
        credentials, or secrets are included.
        """
        feature_importance: Optional[Dict[str, float]] = None

        if self.is_fitted:
            try:
                feature_importance = (
                    self.get_validated_feature_importance()
                )
            except Exception as exc:
                logger.warning(
                    "Unable to collect feature importance for metadata "
                    "of model '%s': %s",
                    self.model_name,
                    exc,
                )

        metadata: Dict[str, Any] = {
            "contract": {
                "name": "PhishGuard AI Model Contract",
                "version": MODEL_CONTRACT_VERSION,
            },
            "model": {
                "name": self.model_name,
                "version": self.version,
                "class": self.__class__.__name__,
                "module": self.__class__.__module__,
            },
            "lifecycle": {
                "state": self.lifecycle_state,
                "is_fitted": self.is_fitted,
                "fitted_timestamp": self.fitted_timestamp_,
                "training_started_timestamp": (
                    self.training_started_timestamp_
                ),
                "training_duration_seconds": (
                    self.training_duration_seconds_
                ),
            },
            "decision": {
                "threshold": self.decision_threshold,
                "positive_class": POSITIVE_CLASS,
                "negative_class": NEGATIVE_CLASS,
            },
            "features": {
                "count": self.feature_count_,
                "names": self.feature_names_,
                "fingerprint": self.feature_fingerprint_,
                "order_enforced": self.require_feature_order,
                "extra_features_allowed": self.allow_extra_features,
            },
            "training": {
                "row_count": self.training_row_count_,
                "target_fingerprint": self.training_target_fingerprint_,
                "dataset_id": self.training_dataset_id_,
                "dataset_version": self.training_dataset_version_,
                "metadata": self.training_metadata_,
            },
            "evaluation": self.evaluation_metadata_,
            "calibration": self.calibration_metadata_,
            "runtime": self.environment_metadata_
            or self._collect_environment_metadata(),
            "artifact": {
                "path": self.artifact_path_,
                "sha256": self.artifact_sha256_,
                "created_timestamp": self.artifact_created_timestamp_,
            },
            "runtime_statistics": {
                "prediction_count": self.prediction_count_,
                "last_prediction_timestamp": (
                    self.last_prediction_timestamp_
                ),
                "last_prediction_duration_seconds": (
                    self.last_prediction_duration_seconds_
                ),
            },
            "model_parameters": _safe_json_value(
                self.model_params
            ),
            "security": {
                "serialization_format": "joblib/pickle-compatible",
                "trusted_source_required": True,
            },
        }

        if feature_importance is not None:
            metadata["feature_importance"] = feature_importance

        return _safe_json_value(metadata)

    # ==========================================================================
    # ARTIFACT NAMING
    # ==========================================================================

    def _artifact_stem(self) -> str:
        """Return safe artifact basename."""
        return (
            f"{self.model_name}"
            f"_v{self.version}"
        )

    def _artifact_paths(
        self,
        output_dir: Union[str, Path],
    ) -> Tuple[Path, Path, Path]:
        """Return model, metadata, and manifest paths."""
        directory = Path(output_dir).resolve()

        stem = self._artifact_stem()

        return (
            directory / f"{stem}{DEFAULT_ARTIFACT_EXTENSION}",
            directory / f"{stem}{DEFAULT_METADATA_SUFFIX}",
            directory / f"{stem}{DEFAULT_MANIFEST_SUFFIX}",
        )

    # ==========================================================================
    # PERSISTENCE
    # ==========================================================================

    def save(
        self,
        output_dir: Union[str, Path],
        *,
        compress: Union[int, bool] = 3,
        write_manifest: bool = True,
    ) -> Path:
        """
        Persist a fitted model and its metadata.

        Returns
        -------
        Path
            Model artifact path.

        Notes
        -----
        The generated joblib artifact is pickle-based and must be treated as
        executable/trusted content.

        The SHA-256 digest is an integrity mechanism, not an authenticity
        mechanism. An attacker who can replace both artifact and manifest can
        replace the digest as well. Authenticity should additionally be provided
        by a trusted artifact registry, signature, or equivalent control.
        """
        if not self.is_fitted:
            raise ModelNotFittedError(
                f"Cannot save unfitted model '{self.model_name}'."
            )

        directory = Path(output_dir).resolve()
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path, metadata_path, manifest_path = (
            self._artifact_paths(directory)
        )

        logger.info(
            "Persisting model artifact '%s'.",
            self.model_name,
        )

        try:
            # ------------------------------------------------------------------
            # Write model atomically.
            # ------------------------------------------------------------------
            _atomic_joblib_dump(
                self,
                model_path,
                compress=compress,
                protocol=5,
            )

            # ------------------------------------------------------------------
            # Compute artifact checksum after successful persistence.
            # ------------------------------------------------------------------
            artifact_hash = _sha256_file(
                model_path
            )

            self.artifact_sha256_ = artifact_hash
            self.artifact_path_ = str(
                model_path
            )
            self.artifact_created_timestamp_ = (
                _utc_now_iso()
            )

            # ------------------------------------------------------------------
            # Metadata sidecar.
            # ------------------------------------------------------------------
            metadata = self.get_metadata()

            # The object inside the joblib was serialized before these artifact
            # fields were populated. Therefore metadata is intentionally kept
            # as the authoritative sidecar.
            _atomic_write_json(
                metadata_path,
                metadata,
            )

            if write_manifest:
                manifest = self._build_artifact_manifest(
                    model_path=model_path,
                    metadata_path=metadata_path,
                )

                _atomic_write_json(
                    manifest_path,
                    manifest,
                )

        except Exception as exc:
            logger.exception(
                "Failed to persist model '%s'.",
                self.model_name,
            )

            raise ModelArtifactError(
                f"Failed to persist model artifact: {exc}"
            ) from exc

        logger.info(
            "Model '%s' saved successfully. SHA-256=%s",
            self.model_name,
            artifact_hash,
        )

        return model_path

    def _build_artifact_manifest(
        self,
        *,
        model_path: Path,
        metadata_path: Path,
    ) -> Dict[str, Any]:
        """
        Build a manifest describing artifact integrity and provenance.
        """
        return {
            "manifest_version": "1.0",
            "created_at": _utc_now_iso(),
            "model": {
                "name": self.model_name,
                "version": self.version,
                "class": self.__class__.__name__,
                "module": self.__class__.__module__,
            },
            "artifact": {
                "filename": model_path.name,
                "size_bytes": model_path.stat().st_size,
                "sha256": _sha256_file(model_path),
            },
            "metadata": {
                "filename": metadata_path.name,
                "sha256": _sha256_file(metadata_path),
            },
            "contract_version": MODEL_CONTRACT_VERSION,
            "environment": self.environment_metadata_
            or self._collect_environment_metadata(),
        }

    # ==========================================================================
    # ARTIFACT VALIDATION
    # ==========================================================================

    @staticmethod
    def verify_artifact_checksum(
        model_filepath: Union[str, Path],
        expected_sha256: str,
    ) -> bool:
        """
        Verify a model artifact against a trusted SHA-256 value.
        """
        expected = str(
            expected_sha256
        ).strip().lower()

        if not re.fullmatch(
            r"[0-9a-f]{64}",
            expected,
        ):
            raise ValueError(
                "expected_sha256 must be a valid 64-character SHA-256 hex digest."
            )

        actual = _sha256_file(
            model_filepath
        )

        return actual == expected

    @staticmethod
    def inspect_artifact_manifest(
        manifest_filepath: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Read a JSON manifest without loading the executable model artifact.
        """
        path = Path(
            manifest_filepath
        ).resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: '{path}'."
            )

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                payload = json.load(handle)
        except Exception as exc:
            raise ModelArtifactError(
                f"Unable to read artifact manifest: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ModelArtifactError(
                "Artifact manifest must contain a JSON object."
            )

        return payload

    @classmethod
    def verify_artifact_manifest(
        cls,
        model_filepath: Union[str, Path],
        manifest_filepath: Union[str, Path],
    ) -> bool:
        """
        Verify the model SHA-256 against a manifest before deserialization.

        This is an integrity check only. It does not establish authenticity if
        the manifest itself is attacker-controlled.
        """
        manifest = cls.inspect_artifact_manifest(
            manifest_filepath
        )

        artifact = manifest.get(
            "artifact"
        )

        if not isinstance(
            artifact,
            dict,
        ):
            raise ModelArtifactError(
                "Manifest does not contain an artifact section."
            )

        expected = artifact.get(
            "sha256"
        )

        if not isinstance(
            expected,
            str,
        ):
            raise ModelArtifactError(
                "Manifest does not contain a valid artifact SHA-256."
            )

        return cls.verify_artifact_checksum(
            model_filepath,
            expected,
        )

    # ==========================================================================
    # LOADING
    # ==========================================================================

    @classmethod
    def load(
        cls,
        model_filepath: Union[str, Path],
        *,
        trusted: bool = False,
        expected_sha256: Optional[str] = None,
        manifest_filepath: Optional[Union[str, Path]] = None,
        verify_manifest: bool = True,
        mmap_mode: Optional[str] = None,
        allow_version_mismatch: bool = False,
    ) -> "BasePhishingModel":
        """
        Load a previously persisted model.

        Security
        --------
        `trusted` MUST be True for deserialization.

        Joblib/pickle is not a safe interchange format. An attacker-controlled
        file must never reach this method.

        Parameters
        ----------
        model_filepath:
            Joblib artifact path.

        trusted:
            Explicit acknowledgement that the artifact is trusted.

        expected_sha256:
            Optional trusted digest from an external trust source.

        manifest_filepath:
            Optional manifest to verify before loading.

        verify_manifest:
            Verify manifest if supplied.

        mmap_mode:
            Optional joblib memory-map mode.

        allow_version_mismatch:
            Permit contract version differences.
        """
        if not trusted:
            raise ModelArtifactError(
                "Refusing to deserialize model artifact because trusted=False. "
                "Joblib uses pickle-compatible deserialization and can execute "
                "arbitrary code. Verify artifact provenance and call load(..., "
                "trusted=True) only for a trusted artifact."
            )

        path = Path(
            model_filepath
        ).resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: '{path}'."
            )

        # ----------------------------------------------------------------------
        # Verify externally trusted checksum BEFORE loading.
        # ----------------------------------------------------------------------
        if expected_sha256 is not None:
            if not cls.verify_artifact_checksum(
                path,
                expected_sha256,
            ):
                raise ModelIntegrityError(
                    f"SHA-256 integrity verification failed for '{path}'."
                )

        # ----------------------------------------------------------------------
        # Verify manifest BEFORE loading.
        # ----------------------------------------------------------------------
        if manifest_filepath is not None and verify_manifest:
            if not cls.verify_artifact_manifest(
                path,
                manifest_filepath,
            ):
                raise ModelIntegrityError(
                    "Model artifact does not match its manifest."
                )

        logger.info(
            "Loading trusted model artifact '%s'.",
            path,
        )

        try:
            loaded_instance = joblib.load(
                path,
                mmap_mode=mmap_mode,
            )
        except Exception as exc:
            logger.exception(
                "Failed to deserialize model artifact '%s'.",
                path,
            )

            raise ModelArtifactError(
                f"Unable to load model artifact: {exc}"
            ) from exc

        if not isinstance(
            loaded_instance,
            BasePhishingModel,
        ):
            raise ModelArtifactError(
                "Loaded object is not a BasePhishingModel instance. "
                f"Received type: {type(loaded_instance)!r}"
            )

        # ----------------------------------------------------------------------
        # Contract compatibility check.
        # ----------------------------------------------------------------------
        if not allow_version_mismatch:
            loaded_contract = (
                loaded_instance.get_metadata()
                .get("contract", {})
                .get("version")
            )

            if loaded_contract != MODEL_CONTRACT_VERSION:
                raise ModelCompatibilityError(
                    "Model contract version mismatch. "
                    f"Artifact={loaded_contract!r}, "
                    f"Runtime={MODEL_CONTRACT_VERSION!r}."
                )

        # ----------------------------------------------------------------------
        # Verify fitted state.
        # ----------------------------------------------------------------------
        if not loaded_instance.is_fitted:
            raise ModelArtifactError(
                "Loaded model artifact is not marked as fitted."
            )

        loaded_instance.artifact_path_ = str(
            path
        )

        loaded_instance.artifact_sha256_ = (
            _sha256_file(path)
        )

        loaded_instance.lifecycle_state = "loaded"

        logger.info(
            "Trusted model '%s' v%s loaded successfully.",
            loaded_instance.model_name,
            loaded_instance.version,
        )

        return loaded_instance

    # ==========================================================================
    # MODEL HEALTH
    # ==========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Return a non-invasive model health report.

        This method does not perform inference.
        """
        checks: Dict[str, bool] = {
            "fitted": bool(self.is_fitted),
            "has_estimator": self.model is not None,
            "has_feature_contract": bool(
                self.feature_names_
            ),
            "valid_threshold": (
                MIN_DECISION_THRESHOLD
                <= self.decision_threshold
                <= MAX_DECISION_THRESHOLD
            ),
        }

        healthy = all(
            checks.values()
        )

        return {
            "healthy": healthy,
            "model_name": self.model_name,
            "model_version": self.version,
            "lifecycle_state": self.lifecycle_state,
            "checks": checks,
            "feature_count": self.feature_count_,
            "timestamp": _utc_now_iso(),
        }

    # ==========================================================================
    # MODEL SUMMARY
    # ==========================================================================

    def summary(self) -> Dict[str, Any]:
        """
        Return concise operational model information.
        """
        return {
            "model_name": self.model_name,
            "version": self.version,
            "class": self.__class__.__name__,
            "is_fitted": self.is_fitted,
            "lifecycle_state": self.lifecycle_state,
            "feature_count": self.feature_count_,
            "decision_threshold": self.decision_threshold,
            "artifact_sha256": self.artifact_sha256_,
            "training_rows": self.training_row_count_,
            "prediction_count": self.prediction_count_,
        }

    # ==========================================================================
    # SERIALIZATION HELPERS
    # ==========================================================================

    def export_metadata(
        self,
        filepath: Union[str, Path],
    ) -> Path:
        """
        Export current metadata independently of the model artifact.
        """
        path = Path(
            filepath
        ).resolve()

        metadata = self.get_metadata()

        _atomic_write_json(
            path,
            metadata,
        )

        logger.info(
            "Exported metadata for model '%s' to '%s'.",
            self.model_name,
            path,
        )

        return path

    # ==========================================================================
    # LIFECYCLE CONTROL
    # ==========================================================================

    def reset_runtime_statistics(self) -> None:
        """
        Reset non-model prediction telemetry.

        Does not modify fitted estimator parameters.
        """
        self.prediction_count_ = 0
        self.last_prediction_timestamp_ = None
        self.last_prediction_duration_seconds_ = None

    def mark_failed(
        self,
        reason: str,
    ) -> None:
        """Mark model lifecycle as failed without exposing sensitive details."""
        self.lifecycle_state = "failed"

        self.training_metadata_[
            "failure_recorded"
        ] = {
            "timestamp": _utc_now_iso(),
            "reason": str(reason)[:1000],
        }

    # ==========================================================================
    # REPRESENTATION
    # ==========================================================================

    def __repr__(self) -> str:
        """Provide a concise operational representation."""
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"version={self.version!r}, "
            f"is_fitted={self.is_fitted!r}, "
            f"feature_count={self.feature_count_}, "
            f"decision_threshold={self.decision_threshold:.4f}"
            f")"
        )


# ==============================================================================
# PUBLIC EXPORT CONTRACT
# ==============================================================================

__all__ = [
    "BasePhishingModel",
    "ModelError",
    "ModelNotFittedError",
    "ModelInputError",
    "ModelArtifactError",
    "ModelIntegrityError",
    "ModelCompatibilityError",
    "MODEL_CONTRACT_VERSION",
    "DEFAULT_MODEL_VERSION",
    "DEFAULT_DECISION_THRESHOLD",
]
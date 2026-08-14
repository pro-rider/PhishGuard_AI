"""
PhishGuard_AI - Production ML Threat Predictor Engine
Version: 11.0.0

Production-grade asynchronous ML inference service for PhishGuard_AI.

Responsibilities
----------------
- Scan-type aware feature extraction.
- Secure model resolution through ModelRegistry.
- Model/version/artifact isolation.
- Strict feature validation and schema alignment.
- ONNX input/output validation.
- Deadline-aware asynchronous inference.
- Probability/logit normalization.
- Deterministic bounded heuristic fallback.
- Explicit ML/fallback provenance.
- DriftMonitor integration.
- Bounded prediction telemetry.
- Defensive error classification.
- Safe logging without sensitive payload disclosure.
- Fail-closed behavior when model inference is unavailable.
- Protection against malformed model outputs.
- Protection against NaN/Inf propagation.
- Protection against feature-schema mismatch.
- Protection against accidental model-version mixing.

Security & Reliability Principles
---------------------------------
1. Never trust model output blindly.
2. Never allow NaN/Inf values into downstream scoring.
3. Never block the asyncio event loop with synchronous inference.
4. Never execute fallback inference after the orchestrator deadline.
5. Never mix model versions/artifacts in drift telemetry.
6. Never expose raw email, URL, attachment, or document content in logs.
7. Never silently accept malformed feature schemas.
8. Never silently interpret arbitrary ONNX tensors as probabilities.
9. Never allow telemetry failures to break a valid prediction.
10. Always return a bounded probability in [0.0, 1.0].
11. Always identify whether the result came from the model or fallback.
12. Keep prediction metadata bounded.
13. Preserve deterministic fallback behavior.
14. Treat model artifacts as untrusted inputs.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from engine.orchestrator import ScanContext, ScanType, SecurityVerdict
from ml_inference.drift_monitor import DriftMonitor, ModelIdentity
from ml_inference.feature_extractors import (
    AttachmentFeatureExtractor,
    EmailFeatureExtractor,
    UrlFeatureExtractor,
)
from ml_inference.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_CRITICAL_THRESHOLD = 0.90
DEFAULT_HIGH_THRESHOLD = 0.75
DEFAULT_MEDIUM_THRESHOLD = 0.40

DEFAULT_FALLBACK_MINIMUM = 0.05
DEFAULT_FALLBACK_MAXIMUM = 0.99

DEFAULT_MAX_FEATURE_VECTOR_LENGTH = 10_000
DEFAULT_MAX_FEATURE_COUNT = 2_000
DEFAULT_MAX_FEATURE_NAME_LENGTH = 256
DEFAULT_MAX_PREDICTION_METADATA_ITEMS = 64

DEFAULT_MAX_MODEL_ID_LENGTH = 256
DEFAULT_MAX_VERSION_LENGTH = 128
DEFAULT_MAX_ARTIFACT_HASH_LENGTH = 256

DEFAULT_MAX_ONNX_OUTPUT_VALUES = 10_000

DEFAULT_UNKNOWN_MODEL_VERSION = "unknown"
DEFAULT_UNKNOWN_ARTIFACT_HASH = "unknown"

# Used only for diagnostics and safety limits.
DEFAULT_MAX_LOGIT_MAGNITUDE = 1_000.0


# ============================================================================
# Enums
# ============================================================================


class PredictionSource(str, Enum):
    """
    Origin of the final threat probability.
    """

    MODEL = "model"
    HEURISTIC_FALLBACK = "heuristic_fallback"


class PredictionStatus(str, Enum):
    """
    High-level prediction execution status.
    """

    SUCCESS = "success"
    FALLBACK = "fallback"
    REJECTED = "rejected"
    FAILED = "failed"


class PredictionErrorCode(str, Enum):
    """
    Stable machine-readable prediction error categories.

    These values are intentionally independent from Python exception names.
    Downstream systems should depend on these codes instead of exception text.
    """

    NONE = "none"

    INVALID_MODEL_ID = "invalid_model_id"
    INVALID_DEADLINE = "invalid_deadline"

    DEADLINE_EXPIRED = "deadline_expired"

    UNSUPPORTED_SCAN_TYPE = "unsupported_scan_type"
    FEATURE_EXTRACTION_FAILED = "feature_extraction_failed"
    INVALID_FEATURES = "invalid_features"

    MODEL_NOT_FOUND = "model_not_found"
    MODEL_MANIFEST_INVALID = "model_manifest_invalid"
    MODEL_SESSION_UNAVAILABLE = "model_session_unavailable"

    FEATURE_SCHEMA_INVALID = "feature_schema_invalid"
    FEATURE_SCHEMA_MISMATCH = "feature_schema_mismatch"

    ONNX_INPUT_INVALID = "onnx_input_invalid"
    ONNX_EXECUTION_FAILED = "onnx_execution_failed"
    ONNX_OUTPUT_INVALID = "onnx_output_invalid"

    TELEMETRY_FAILED = "telemetry_failed"

    UNKNOWN_FAILURE = "unknown_failure"


# ============================================================================
# Result Models
# ============================================================================


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """
    Immutable prediction result.

    Invariants
    ----------
    probability:
        Always finite and in [0.0, 1.0].

    execution_time_ms:
        Always finite and >= 0.

    feature_count:
        Number of sanitized extracted features.

    features_used:
        Bounded metadata copy only. It must never be interpreted as the
        complete original input.
    """

    model_id: str
    probability: float
    verdict: SecurityVerdict
    execution_time_ms: float

    features_used: Mapping[str, float] = field(default_factory=dict)

    source: PredictionSource = PredictionSource.MODEL
    status: PredictionStatus = PredictionStatus.SUCCESS

    model_version: str = DEFAULT_UNKNOWN_MODEL_VERSION
    artifact_hash: str = DEFAULT_UNKNOWN_ARTIFACT_HASH

    feature_count: int = 0
    fallback_used: bool = False

    error_code: PredictionErrorCode = PredictionErrorCode.NONE

    # Optional operational metadata.
    request_id: Optional[str] = None

    # Number of model features actually supplied to ONNX.
    model_feature_count: int = 0

    # Number of model features absent from extracted features.
    missing_feature_count: int = 0

    # Indicates whether telemetry was successfully recorded.
    telemetry_recorded: bool = False

    @property
    def is_model_prediction(self) -> bool:
        return self.source == PredictionSource.MODEL

    @property
    def is_fallback_prediction(self) -> bool:
        return self.source == PredictionSource.HEURISTIC_FALLBACK


@dataclass(frozen=True, slots=True)
class _ModelRuntime:
    """
    Immutable runtime description of a concrete model artifact.

    The ONNX session itself is treated as an opaque runtime object.
    """

    model_id: str
    version: str
    artifact_hash: str
    session: Any
    feature_names: Tuple[str, ...]

    input_name: str
    input_type: Optional[str] = None
    input_shape: Optional[Tuple[Any, ...]] = None


@dataclass(frozen=True, slots=True)
class _FeatureAlignment:
    """
    Result of aligning extracted features against the model schema.
    """

    vector: Tuple[float, ...]
    missing_features: Tuple[str, ...]
    unknown_features: Tuple[str, ...]


# ============================================================================
# Predictor
# ============================================================================


class MLPredictor:
    """
    Production asynchronous ML inference engine.

    The predictor is intentionally stateless with respect to individual
    requests. Model lifecycle and artifact integrity remain responsibilities
    of ModelRegistry.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        drift_monitor: Optional[DriftMonitor] = None,
        *,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
        high_threshold: float = DEFAULT_HIGH_THRESHOLD,
        medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
        fallback_minimum: float = DEFAULT_FALLBACK_MINIMUM,
        fallback_maximum: float = DEFAULT_FALLBACK_MAXIMUM,
        max_feature_vector_length: int = DEFAULT_MAX_FEATURE_VECTOR_LENGTH,
        max_feature_count: int = DEFAULT_MAX_FEATURE_COUNT,
        max_prediction_metadata_items: int = DEFAULT_MAX_PREDICTION_METADATA_ITEMS,
        max_onnx_output_values: int = DEFAULT_MAX_ONNX_OUTPUT_VALUES,
        strict_feature_schema: bool = False,
        require_model_feature_names: bool = True,
    ) -> None:

        self.registry = registry or ModelRegistry()
        self.drift_monitor = drift_monitor or DriftMonitor()

        self.critical_threshold = self._validate_threshold(
            critical_threshold,
            "critical_threshold",
        )

        self.high_threshold = self._validate_threshold(
            high_threshold,
            "high_threshold",
        )

        self.medium_threshold = self._validate_threshold(
            medium_threshold,
            "medium_threshold",
        )

        if not (
            self.medium_threshold
            < self.high_threshold
            < self.critical_threshold
        ):
            raise ValueError(
                "Thresholds must satisfy: "
                "medium < high < critical"
            )

        if (
            not math.isfinite(fallback_minimum)
            or not math.isfinite(fallback_maximum)
            or not (
                0.0
                <= fallback_minimum
                <= fallback_maximum
                <= 1.0
            )
        ):
            raise ValueError(
                "fallback_minimum and fallback_maximum "
                "must be finite values within [0, 1]"
            )

        if max_feature_vector_length <= 0:
            raise ValueError(
                "max_feature_vector_length must be greater than zero"
            )

        if max_feature_count <= 0:
            raise ValueError(
                "max_feature_count must be greater than zero"
            )

        if max_prediction_metadata_items <= 0:
            raise ValueError(
                "max_prediction_metadata_items must be greater than zero"
            )

        if max_onnx_output_values <= 0:
            raise ValueError(
                "max_onnx_output_values must be greater than zero"
            )

        self.fallback_minimum = float(fallback_minimum)
        self.fallback_maximum = float(fallback_maximum)

        self.max_feature_vector_length = max_feature_vector_length
        self.max_feature_count = max_feature_count
        self.max_prediction_metadata_items = (
            max_prediction_metadata_items
        )
        self.max_onnx_output_values = max_onnx_output_values

        self.strict_feature_schema = bool(strict_feature_schema)
        self.require_model_feature_names = bool(
            require_model_feature_names
        )

        # Feature extractors.
        self.email_extractor = EmailFeatureExtractor()
        self.url_extractor = UrlFeatureExtractor()
        self.attachment_extractor = AttachmentFeatureExtractor()

    # ========================================================================
    # Validation
    # ========================================================================

    @staticmethod
    def _validate_threshold(
        value: float,
        name: str,
    ) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc

        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")

        if not 0.0 <= result <= 1.0:
            raise ValueError(
                f"{name} must be within [0, 1]"
            )

        return result

    @staticmethod
    def _validate_model_id(model_id: str) -> str:
        if not isinstance(model_id, str):
            raise TypeError("model_id must be a string")

        value = model_id.strip()

        if not value:
            raise ValueError("model_id must not be empty")

        if len(value) > DEFAULT_MAX_MODEL_ID_LENGTH:
            raise ValueError(
                f"model_id exceeds maximum length "
                f"of {DEFAULT_MAX_MODEL_ID_LENGTH}"
            )

        return value

    @staticmethod
    def _validate_deadline(
        deadline_monotonic: float,
    ) -> float:
        try:
            deadline = float(deadline_monotonic)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "deadline_monotonic must be numeric"
            ) from exc

        if not math.isfinite(deadline):
            raise ValueError(
                "deadline_monotonic must be finite"
            )

        return deadline

    @staticmethod
    def _validate_probability(
        probability: float,
    ) -> float:
        try:
            value = float(probability)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Probability must be numeric"
            ) from exc

        if not math.isfinite(value):
            raise ValueError(
                "Probability must be finite"
            )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "Probability must be within [0, 1]"
            )

        return value

    # ========================================================================
    # Deadline Management
    # ========================================================================

    @staticmethod
    def _remaining_time(
        deadline_monotonic: float,
    ) -> float:
        return deadline_monotonic - time.monotonic()

    @classmethod
    def _ensure_deadline(
        cls,
        deadline_monotonic: float,
    ) -> None:
        if cls._remaining_time(deadline_monotonic) <= 0.0:
            raise asyncio.TimeoutError(
                "Prediction deadline has expired"
            )

    async def _run_with_deadline(
        self,
        operation: Any,
        deadline_monotonic: float,
    ) -> Any:
        """
        Run blocking work in a worker thread while respecting the deadline.

        Important:
        asyncio.wait_for() cannot forcibly terminate the underlying thread.
        Therefore operations must remain bounded by the caller's overall
        deadline and the registry/runtime must independently avoid permanent
        blocking operations.
        """

        self._ensure_deadline(deadline_monotonic)

        remaining = self._remaining_time(
            deadline_monotonic
        )

        if remaining <= 0.0:
            raise asyncio.TimeoutError(
                "Prediction deadline has expired"
            )

        task = asyncio.create_task(
            asyncio.to_thread(operation)
        )

        try:
            return await asyncio.wait_for(
                task,
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            # Do not allow cancellation of the asyncio task to become a
            # misleading "successful" model operation.
            task.cancel()
            raise

    # ========================================================================
    # Feature Extractor Selection
    # ========================================================================

    def _select_extractor(
        self,
        scan_type: ScanType,
    ) -> Any:

        if scan_type in (
            ScanType.EMAIL,
            ScanType.COMPREHENSIVE,
        ):
            return self.email_extractor

        if scan_type == ScanType.URL:
            return self.url_extractor

        if scan_type == ScanType.ATTACHMENT:
            return self.attachment_extractor

        raise ValueError(
            f"Unsupported scan type for ML prediction: "
            f"{scan_type!r}"
        )

    # ========================================================================
    # Feature Sanitization
    # ========================================================================

    def _sanitize_features(
        self,
        features: Mapping[str, Any],
    ) -> Dict[str, float]:

        if not isinstance(features, Mapping):
            raise TypeError(
                "Feature extractor must return a mapping"
            )

        if len(features) > self.max_feature_count:
            raise ValueError(
                "Feature count exceeds configured maximum"
            )

        sanitized: Dict[str, float] = {}

        for raw_name, raw_value in features.items():

            if not isinstance(raw_name, str):
                raise ValueError(
                    "Feature names must be strings"
                )

            name = raw_name.strip()

            if not name:
                raise ValueError(
                    "Feature name cannot be empty"
                )

            if len(name) > DEFAULT_MAX_FEATURE_NAME_LENGTH:
                raise ValueError(
                    f"Feature name '{name[:64]}' exceeds "
                    "maximum allowed length"
                )

            # bool is technically convertible to float but is usually an
            # accidental feature type.
            if isinstance(raw_value, bool):
                raise ValueError(
                    f"Feature '{name}' must not be boolean"
                )

            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Feature '{name}' is not numeric"
                ) from exc

            if not math.isfinite(value):
                raise ValueError(
                    f"Feature '{name}' contains NaN/Inf"
                )

            sanitized[name] = value

        return sanitized

    # ========================================================================
    # Model Manifest Handling
    # ========================================================================

    @staticmethod
    def _read_manifest_value(
        manifest: Any,
        *names: str,
        default: str = "",
    ) -> str:

        for name in names:
            try:
                value = getattr(
                    manifest,
                    name,
                    None,
                )

                if value is None:
                    continue

                text = str(value).strip()

                if text:
                    return text

            except Exception:
                continue

        return default

    @staticmethod
    def _normalize_feature_names(
        raw_feature_names: Any,
    ) -> Tuple[str, ...]:

        if raw_feature_names is None:
            return ()

        if isinstance(
            raw_feature_names,
            (str, bytes),
        ):
            raise ValueError(
                "Model feature_names must be a sequence, "
                "not a string"
            )

        try:
            names = tuple(raw_feature_names)
        except TypeError as exc:
            raise ValueError(
                "Model feature_names are not iterable"
            ) from exc

        normalized = []

        for raw_name in names:

            if not isinstance(raw_name, str):
                raise ValueError(
                    "Model feature names must be strings"
                )

            name = raw_name.strip()

            if not name:
                raise ValueError(
                    "Model feature names cannot be empty"
                )

            if len(name) > DEFAULT_MAX_FEATURE_NAME_LENGTH:
                raise ValueError(
                    "Model feature name exceeds maximum length"
                )

            normalized.append(name)

        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "Model feature schema contains duplicate names"
            )

        if len(normalized) > DEFAULT_MAX_FEATURE_COUNT:
            raise ValueError(
                "Model feature schema exceeds maximum feature count"
            )

        return tuple(normalized)

    # ========================================================================
    # Model Runtime Loading
    # ========================================================================

    async def _load_model_runtime(
        self,
        model_id: str,
        deadline_monotonic: float,
    ) -> Tuple[Optional[_ModelRuntime], PredictionErrorCode]:

        self._ensure_deadline(deadline_monotonic)

        try:
            manifest = await self._run_with_deadline(
                lambda: self.registry.get_manifest(model_id),
                deadline_monotonic,
            )
        except asyncio.TimeoutError:
            raise
        except Exception:
            logger.warning(
                "Model manifest lookup failed "
                "model_id=%s",
                model_id,
            )
            return (
                None,
                PredictionErrorCode.MODEL_MANIFEST_INVALID,
            )

        if manifest is None:
            return (
                None,
                PredictionErrorCode.MODEL_NOT_FOUND,
            )

        try:
            feature_names = self._normalize_feature_names(
                getattr(
                    manifest,
                    "feature_names",
                    (),
                )
            )
        except Exception:
            logger.error(
                "Invalid feature schema "
                "model_id=%s",
                model_id,
            )
            return (
                None,
                PredictionErrorCode.FEATURE_SCHEMA_INVALID,
            )

        if (
            self.require_model_feature_names
            and not feature_names
        ):
            logger.error(
                "Model has no feature schema "
                "model_id=%s",
                model_id,
            )
            return (
                None,
                PredictionErrorCode.FEATURE_SCHEMA_INVALID,
            )

        try:
            session = await self._run_with_deadline(
                lambda: self.registry.load_session(model_id),
                deadline_monotonic,
            )
        except asyncio.TimeoutError:
            raise
        except Exception:
            logger.error(
                "Model session load failed "
                "model_id=%s",
                model_id,
            )
            return (
                None,
                PredictionErrorCode.MODEL_SESSION_UNAVAILABLE,
            )

        if session is None:
            return (
                None,
                PredictionErrorCode.MODEL_SESSION_UNAVAILABLE,
            )

        try:
            input_name = self._get_input_name(session)
        except Exception:
            logger.error(
                "Invalid ONNX input schema "
                "model_id=%s",
                model_id,
            )
            return (
                None,
                PredictionErrorCode.ONNX_INPUT_INVALID,
            )

        version = self._read_manifest_value(
            manifest,
            "version",
            "model_version",
            default=DEFAULT_UNKNOWN_MODEL_VERSION,
        )

        artifact_hash = self._read_manifest_value(
            manifest,
            "artifact_hash",
            "sha256",
            "model_hash",
            default=DEFAULT_UNKNOWN_ARTIFACT_HASH,
        )

        if len(version) > DEFAULT_MAX_VERSION_LENGTH:
            return (
                None,
                PredictionErrorCode.MODEL_MANIFEST_INVALID,
            )

        if len(artifact_hash) > DEFAULT_MAX_ARTIFACT_HASH_LENGTH:
            return (
                None,
                PredictionErrorCode.MODEL_MANIFEST_INVALID,
            )

        # Capture ONNX input metadata where available.
        input_type: Optional[str] = None
        input_shape: Optional[Tuple[Any, ...]] = None

        try:
            inputs = session.get_inputs()

            if inputs:
                input_metadata = inputs[0]

                raw_type = getattr(
                    input_metadata,
                    "type",
                    None,
                )

                if raw_type:
                    input_type = str(raw_type)

                raw_shape = getattr(
                    input_metadata,
                    "shape",
                    None,
                )

                if raw_shape is not None:
                    input_shape = tuple(raw_shape)

        except Exception:
            # Metadata is useful but should not make an otherwise valid
            # runtime unusable.
            logger.debug(
                "Unable to inspect optional ONNX metadata "
                "model_id=%s",
                model_id,
                exc_info=True,
            )

        return (
            _ModelRuntime(
                model_id=model_id,
                version=version,
                artifact_hash=artifact_hash,
                session=session,
                feature_names=feature_names,
                input_name=input_name,
                input_type=input_type,
                input_shape=input_shape,
            ),
            PredictionErrorCode.NONE,
        )

    # ========================================================================
    # Feature Alignment
    # ========================================================================

    def _align_features(
        self,
        feature_names: Sequence[str],
        features: Mapping[str, float],
    ) -> _FeatureAlignment:

        if not feature_names:
            raise ValueError(
                "Model feature schema is empty"
            )

        if len(feature_names) > self.max_feature_vector_length:
            raise ValueError(
                "Model feature vector exceeds maximum length"
            )

        vector = []
        missing = []

        known = set(feature_names)

        for name in feature_names:
            if name not in features:
                missing.append(name)
                vector.append(0.0)
            else:
                value = features[name]

                if not math.isfinite(value):
                    raise ValueError(
                        f"Feature '{name}' is not finite"
                    )

                vector.append(float(value))

        unknown = tuple(
            name
            for name in features
            if name not in known
        )

        if (
            self.strict_feature_schema
            and unknown
        ):
            raise ValueError(
                "Extracted features do not match model schema"
            )

        return _FeatureAlignment(
            vector=tuple(vector),
            missing_features=tuple(missing),
            unknown_features=unknown,
        )

    # ========================================================================
    # ONNX Input Validation
    # ========================================================================

    @staticmethod
    def _get_input_name(
        session: Any,
    ) -> str:

        try:
            inputs = session.get_inputs()
        except Exception as exc:
            raise RuntimeError(
                "Unable to inspect ONNX model inputs"
            ) from exc

        if not inputs:
            raise RuntimeError(
                "ONNX model exposes no inputs"
            )

        name = getattr(
            inputs[0],
            "name",
            None,
        )

        if not name:
            raise RuntimeError(
                "ONNX model input has no valid name"
            )

        return str(name)

    def _validate_model_input_shape(
        self,
        runtime: _ModelRuntime,
        vector_length: int,
    ) -> None:

        shape = runtime.input_shape

        if not shape:
            return

        # Typical ONNX classification shape:
        # [batch, feature_count]
        if len(shape) != 2:
            return

        expected_features = shape[1]

        if isinstance(
            expected_features,
            int,
        ):
            if expected_features > 0:
                if expected_features != vector_length:
                    raise ValueError(
                        "Feature vector length does not match "
                        "ONNX input schema"
                    )

    # ========================================================================
    # ONNX Output Processing
    # ========================================================================

    def _flatten_numeric_output(
        self,
        output: Any,
    ) -> list[float]:

        if hasattr(output, "tolist"):
            output = output.tolist()

        values: list[float] = []

        def walk(value: Any) -> None:

            if isinstance(
                value,
                (list, tuple),
            ):
                for item in value:
                    if len(values) >= self.max_onnx_output_values:
                        raise ValueError(
                            "ONNX output exceeds configured size limit"
                        )
                    walk(item)
                return

            if isinstance(value, bool):
                raise ValueError(
                    "ONNX output contains boolean value"
                )

            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "ONNX output contains non-numeric value"
                ) from exc

            if not math.isfinite(number):
                raise ValueError(
                    "ONNX output contains NaN/Inf"
                )

            if abs(number) > DEFAULT_MAX_LOGIT_MAGNITUDE:
                raise ValueError(
                    "ONNX output contains an excessively large value"
                )

            values.append(number)

        walk(output)

        if not values:
            raise ValueError(
                "ONNX output contains no numeric values"
            )

        return values

    @staticmethod
    def _sigmoid(value: float) -> float:
        """
        Numerically stable sigmoid.
        """

        if value >= 0.0:
            z = math.exp(-min(value, 60.0))
            return 1.0 / (1.0 + z)

        z = math.exp(
            max(value, -60.0)
        )

        return z / (1.0 + z)

    def _probability_from_values(
        self,
        values: Sequence[float],
    ) -> float:

        if not values:
            raise ValueError(
                "Model output is empty"
            )

        # Binary scalar output.
        if len(values) == 1:
            value = float(values[0])

            if not math.isfinite(value):
                raise ValueError(
                    "Model output is not finite"
                )

            # Already a probability.
            if 0.0 <= value <= 1.0:
                return value

            # Otherwise treat it as a binary logit.
            return self._sigmoid(value)

        # Common binary/multiclass probability output.
        if all(
            0.0 <= value <= 1.0
            for value in values
        ):
            total = sum(values)

            if (
                math.isfinite(total)
                and total > 0.0
            ):
                # For a conventional class-probability tensor,
                # the last class is interpreted as the positive class.
                return float(
                    values[-1] / total
                )

        # Otherwise treat the vector as logits.
        max_logit = max(values)

        exp_values = [
            math.exp(
                max(
                    min(
                        value - max_logit,
                        60.0,
                    ),
                    -60.0,
                )
            )
            for value in values
        ]

        denominator = sum(exp_values)

        if (
            denominator <= 0.0
            or not math.isfinite(denominator)
        ):
            raise ValueError(
                "Unable to normalize ONNX logits"
            )

        probability = (
            exp_values[-1] / denominator
        )

        return probability

    def _parse_model_probability(
        self,
        raw_outputs: Any,
    ) -> float:

        if raw_outputs is None:
            raise ValueError(
                "ONNX runtime returned None"
            )

        if not isinstance(
            raw_outputs,
            (list, tuple),
        ):
            raw_outputs = [raw_outputs]

        if not raw_outputs:
            raise ValueError(
                "ONNX runtime returned no outputs"
            )

        # Current architecture expects one prediction output.
        #
        # We deliberately inspect only the first output rather than
        # arbitrarily combining multiple unrelated ONNX outputs.
        values = self._flatten_numeric_output(
            raw_outputs[0]
        )

        probability = (
            self._probability_from_values(values)
        )

        return self._validate_probability(
            probability
        )

    # ========================================================================
    # Model Execution
    # ========================================================================

    async def _run_model(
        self,
        runtime: _ModelRuntime,
        feature_vector: Sequence[float],
        deadline_monotonic: float,
    ) -> float:

        if not feature_vector:
            raise ValueError(
                "Feature vector cannot be empty"
            )

        if len(feature_vector) > self.max_feature_vector_length:
            raise ValueError(
                "Feature vector exceeds configured maximum"
            )

        self._validate_model_input_shape(
            runtime,
            len(feature_vector),
        )

        for value in feature_vector:
            if not math.isfinite(float(value)):
                raise ValueError(
                    "Feature vector contains NaN/Inf"
                )

        # Keep the structure conventional for an ONNX classifier:
        # batch_size=1, feature_count=N.
        model_input = [
            list(
                map(
                    float,
                    feature_vector,
                )
            )
        ]

        raw_outputs = await self._run_with_deadline(
            lambda: runtime.session.run(
                None,
                {
                    runtime.input_name: model_input,
                },
            ),
            deadline_monotonic,
        )

        return self._parse_model_probability(
            raw_outputs
        )

    # ========================================================================
    # Heuristic Fallback
    # ========================================================================

    def _heuristic_fallback(
        self,
        features: Mapping[str, float],
    ) -> float:
        """
        Deterministic bounded fallback risk estimator.

        This is deliberately transparent and conservative.

        It must never return a value outside the configured range.
        """

        score = 0.0

        # Email authentication failures.
        if features.get(
            "dmarc_fail_score",
            0.0,
        ) > 0.5:
            score += 0.20

        if features.get(
            "spf_fail_score",
            0.0,
        ) > 0.5:
            score += 0.10

        if features.get(
            "dkim_fail_score",
            0.0,
        ) > 0.5:
            score += 0.10

        # Web / credential indicators.
        if features.get(
            "has_external_form_action",
            0.0,
        ) > 0.0:
            score += 0.25

        if features.get(
            "is_typosquat",
            0.0,
        ) > 0.0:
            score += 0.25

        if features.get(
            "url_has_ip_host",
            0.0,
        ) > 0.0:
            score += 0.10

        if features.get(
            "url_has_punycode",
            0.0,
        ) > 0.0:
            score += 0.10

        if features.get(
            "url_shortened",
            0.0,
        ) > 0.0:
            score += 0.08

        # Attachment indicators.
        if features.get(
            "pdf_has_launch_action",
            0.0,
        ) > 0.0:
            score += 0.30

        if features.get(
            "has_macro_attachment",
            0.0,
        ) > 0.0:
            score += 0.30

        if features.get(
            "attachment_contains_script",
            0.0,
        ) > 0.0:
            score += 0.20

        # NLP / social engineering indicators.
        if features.get(
            "suspicious_keyword_score",
            0.0,
        ) > 0.5:
            score += 0.10

        if features.get(
            "credential_request_score",
            0.0,
        ) > 0.5:
            score += 0.15

        bounded = min(
            max(
                score,
                self.fallback_minimum,
            ),
            self.fallback_maximum,
        )

        return float(bounded)

    # ========================================================================
    # Verdict Mapping
    # ========================================================================

    def _probability_to_verdict(
        self,
        probability: float,
    ) -> SecurityVerdict:

        probability = self._validate_probability(
            probability
        )

        if (
            probability
            >= self.critical_threshold
        ):
            return SecurityVerdict.CRITICAL

        if (
            probability
            >= self.high_threshold
        ):
            return SecurityVerdict.HIGH

        if (
            probability
            >= self.medium_threshold
        ):
            return SecurityVerdict.MEDIUM

        return SecurityVerdict.LOW

    # ========================================================================
    # Telemetry
    # ========================================================================

    def _record_drift(
        self,
        identity: ModelIdentity,
        probability: float,
        features: Mapping[str, float],
        timestamp: float,
    ) -> bool:
        """
        Record telemetry without allowing telemetry failure to affect
        the primary prediction result.
        """

        try:
            self.drift_monitor.record_inference(
                identity,
                probability,
                features,
                timestamp=timestamp,
            )
            return True

        except Exception:
            logger.error(
                "Drift telemetry recording failed "
                "model_key=%s",
                identity.key,
                exc_info=True,
            )
            return False

    # ========================================================================
    # Result Helpers
    # ========================================================================

    def _bounded_feature_metadata(
        self,
        features: Mapping[str, float],
    ) -> Dict[str, float]:

        # Sorting gives deterministic metadata ordering and therefore makes
        # tests/auditing easier.
        items = sorted(
            features.items(),
            key=lambda item: item[0],
        )

        return dict(
            items[
                : self.max_prediction_metadata_items
            ]
        )

    @staticmethod
    def _execution_time_ms(
        start_monotonic: float,
    ) -> float:

        duration = (
            time.monotonic()
            - start_monotonic
        ) * 1000.0

        if not math.isfinite(duration):
            return 0.0

        return max(
            0.0,
            float(duration),
        )

    def _build_fallback_result(
        self,
        *,
        model_id: str,
        features: Mapping[str, float],
        start_monotonic: float,
        error_code: PredictionErrorCode,
        runtime: Optional[_ModelRuntime],
        request_id: Optional[str],
        model_feature_count: int = 0,
        missing_feature_count: int = 0,
    ) -> PredictionResult:

        probability = self._heuristic_fallback(
            features
        )

        verdict = self._probability_to_verdict(
            probability
        )

        # IMPORTANT:
        # Do not perform fallback after an expired orchestrator deadline
        # from inside this helper. The caller decides whether fallback is
        # permitted.
        return PredictionResult(
            model_id=model_id,
            probability=probability,
            verdict=verdict,
            execution_time_ms=self._execution_time_ms(
                start_monotonic
            ),
            features_used=self._bounded_feature_metadata(
                features
            ),
            source=PredictionSource.HEURISTIC_FALLBACK,
            status=PredictionStatus.FALLBACK,
            model_version=(
                runtime.version
                if runtime
                else DEFAULT_UNKNOWN_MODEL_VERSION
            ),
            artifact_hash=(
                runtime.artifact_hash
                if runtime
                else DEFAULT_UNKNOWN_ARTIFACT_HASH
            ),
            feature_count=len(features),
            fallback_used=True,
            error_code=error_code,
            request_id=request_id,
            model_feature_count=model_feature_count,
            missing_feature_count=missing_feature_count,
            telemetry_recorded=False,
        )

    # ========================================================================
    # Main Prediction API
    # ========================================================================

    async def predict(
        self,
        context: ScanContext,
        model_id: str,
        deadline_monotonic: float,
        *,
        request_id: Optional[str] = None,
    ) -> PredictionResult:
        """
        Execute asynchronous threat prediction.

        Important deadline behavior
        ----------------------------
        If the deadline is exhausted before a prediction can be completed,
        the method raises asyncio.TimeoutError rather than pretending that
        a late fallback prediction satisfied the deadline.

        This prevents a critical architectural bug where downstream
        orchestrator timeouts could be violated by fallback computation.
        """

        start_monotonic = time.monotonic()

        clean_model_id = self._validate_model_id(
            model_id
        )

        deadline = self._validate_deadline(
            deadline_monotonic
        )

        features: Dict[str, float] = {}
        runtime: Optional[_ModelRuntime] = None

        error_code = PredictionErrorCode.NONE

        try:
            self._ensure_deadline(deadline)

            # --------------------------------------------------------------
            # 1. Feature extraction
            # --------------------------------------------------------------

            try:
                extractor = self._select_extractor(
                    context.scan_type
                )
            except ValueError:
                raise _PredictionPipelineError(
                    PredictionErrorCode.UNSUPPORTED_SCAN_TYPE
                )

            try:
                raw_features = await self._run_with_deadline(
                    lambda: extractor.extract_features(
                        context
                    ),
                    deadline,
                )
            except asyncio.TimeoutError:
                raise
            except Exception as exc:
                logger.error(
                    "Feature extraction failed "
                    "model_id=%s error_type=%s",
                    clean_model_id,
                    type(exc).__name__,
                )

                raise _PredictionPipelineError(
                    PredictionErrorCode.FEATURE_EXTRACTION_FAILED
                ) from exc

            try:
                features = self._sanitize_features(
                    raw_features
                )
            except Exception as exc:
                logger.error(
                    "Feature validation failed "
                    "model_id=%s error_type=%s",
                    clean_model_id,
                    type(exc).__name__,
                )

                raise _PredictionPipelineError(
                    PredictionErrorCode.INVALID_FEATURES
                ) from exc

            # --------------------------------------------------------------
            # 2. Load concrete model runtime
            # --------------------------------------------------------------

            runtime, runtime_error = (
                await self._load_model_runtime(
                    clean_model_id,
                    deadline,
                )
            )

            if runtime is None:

                # Missing model/artifact is a valid fallback condition.
                error_code = runtime_error

                self._ensure_deadline(deadline)

                probability = self._heuristic_fallback(
                    features
                )

                identity = ModelIdentity(
                    model_id=clean_model_id,
                    version=DEFAULT_UNKNOWN_MODEL_VERSION,
                    artifact_hash=DEFAULT_UNKNOWN_ARTIFACT_HASH,
                )

                telemetry_recorded = (
                    self._record_drift(
                        identity,
                        probability,
                        features,
                        time.time(),
                    )
                )

                self._ensure_deadline(deadline)

                return PredictionResult(
                    model_id=clean_model_id,
                    probability=probability,
                    verdict=self._probability_to_verdict(
                        probability
                    ),
                    execution_time_ms=self._execution_time_ms(
                        start_monotonic
                    ),
                    features_used=self._bounded_feature_metadata(
                        features
                    ),
                    source=PredictionSource.HEURISTIC_FALLBACK,
                    status=PredictionStatus.FALLBACK,
                    model_version=(
                        DEFAULT_UNKNOWN_MODEL_VERSION
                    ),
                    artifact_hash=(
                        DEFAULT_UNKNOWN_ARTIFACT_HASH
                    ),
                    feature_count=len(features),
                    fallback_used=True,
                    error_code=error_code,
                    request_id=request_id,
                    telemetry_recorded=(
                        telemetry_recorded
                    ),
                )

            # --------------------------------------------------------------
            # 3. Feature schema alignment
            # --------------------------------------------------------------

            try:
                alignment = self._align_features(
                    runtime.feature_names,
                    features,
                )

                if len(alignment.vector) == 0:
                    raise ValueError(
                        "Aligned model feature vector is empty"
                    )

            except Exception as exc:
                logger.error(
                    "Model feature schema mismatch "
                    "model_id=%s version=%s error_type=%s",
                    clean_model_id,
                    runtime.version,
                    type(exc).__name__,
                )

                raise _PredictionPipelineError(
                    PredictionErrorCode.FEATURE_SCHEMA_MISMATCH
                ) from exc

            # --------------------------------------------------------------
            # 4. Execute ONNX model
            # --------------------------------------------------------------

            try:
                probability = await self._run_model(
                    runtime,
                    alignment.vector,
                    deadline,
                )

            except asyncio.TimeoutError:
                raise

            except ValueError as exc:
                logger.error(
                    "Invalid ONNX model output/input "
                    "model_id=%s version=%s error_type=%s",
                    clean_model_id,
                    runtime.version,
                    type(exc).__name__,
                )

                raise _PredictionPipelineError(
                    PredictionErrorCode.ONNX_OUTPUT_INVALID
                ) from exc

            except Exception as exc:
                logger.error(
                    "ONNX inference failed "
                    "model_id=%s version=%s error_type=%s",
                    clean_model_id,
                    runtime.version,
                    type(exc).__name__,
                )

                raise _PredictionPipelineError(
                    PredictionErrorCode.ONNX_EXECUTION_FAILED
                ) from exc

            # --------------------------------------------------------------
            # 5. Final model probability validation
            # --------------------------------------------------------------

            probability = self._validate_probability(
                probability
            )

            # --------------------------------------------------------------
            # 6. Record model-specific drift telemetry
            # --------------------------------------------------------------

            identity = ModelIdentity(
                model_id=clean_model_id,
                version=runtime.version,
                artifact_hash=runtime.artifact_hash,
            )

            telemetry_recorded = (
                self._record_drift(
                    identity,
                    probability,
                    features,
                    time.time(),
                )
            )

            # --------------------------------------------------------------
            # 7. Ensure deadline before returning
            # --------------------------------------------------------------

            self._ensure_deadline(deadline)

            return PredictionResult(
                model_id=clean_model_id,
                probability=probability,
                verdict=self._probability_to_verdict(
                    probability
                ),
                execution_time_ms=self._execution_time_ms(
                    start_monotonic
                ),
                features_used=self._bounded_feature_metadata(
                    features
                ),
                source=PredictionSource.MODEL,
                status=PredictionStatus.SUCCESS,
                model_version=runtime.version,
                artifact_hash=runtime.artifact_hash,
                feature_count=len(features),
                fallback_used=False,
                error_code=PredictionErrorCode.NONE,
                request_id=request_id,
                model_feature_count=len(
                    alignment.vector
                ),
                missing_feature_count=len(
                    alignment.missing_features
                ),
                telemetry_recorded=(
                    telemetry_recorded
                ),
            )

        # ==================================================================
        # Deadline
        # ==================================================================

        except asyncio.TimeoutError:

            logger.warning(
                "Prediction deadline expired "
                "model_id=%s",
                clean_model_id,
            )

            # Do NOT return a late fallback result.
            #
            # The orchestrator owns deadline semantics. Returning a fallback
            # here after the deadline would violate the caller's contract.
            raise

        # ==================================================================
        # Controlled pipeline failure
        # ==================================================================

        except _PredictionPipelineError as exc:

            error_code = exc.code

            logger.warning(
                "Prediction pipeline fallback "
                "model_id=%s error_code=%s",
                clean_model_id,
                error_code.value,
            )

            # Fallback itself must still respect the deadline.
            self._ensure_deadline(deadline)

            probability = self._heuristic_fallback(
                features
            )

            identity = ModelIdentity(
                model_id=clean_model_id,
                version=(
                    runtime.version
                    if runtime
                    else DEFAULT_UNKNOWN_MODEL_VERSION
                ),
                artifact_hash=(
                    runtime.artifact_hash
                    if runtime
                    else DEFAULT_UNKNOWN_ARTIFACT_HASH
                ),
            )

            telemetry_recorded = (
                self._record_drift(
                    identity,
                    probability,
                    features,
                    time.time(),
                )
            )

            self._ensure_deadline(deadline)

            return PredictionResult(
                model_id=clean_model_id,
                probability=probability,
                verdict=self._probability_to_verdict(
                    probability
                ),
                execution_time_ms=self._execution_time_ms(
                    start_monotonic
                ),
                features_used=self._bounded_feature_metadata(
                    features
                ),
                source=PredictionSource.HEURISTIC_FALLBACK,
                status=PredictionStatus.FALLBACK,
                model_version=identity.version,
                artifact_hash=identity.artifact_hash,
                feature_count=len(features),
                fallback_used=True,
                error_code=error_code,
                request_id=request_id,
                model_feature_count=(
                    len(runtime.feature_names)
                    if runtime
                    else 0
                ),
                missing_feature_count=0,
                telemetry_recorded=(
                    telemetry_recorded
                ),
            )

        # ==================================================================
        # Unexpected failure
        # ==================================================================

        except Exception:
            logger.exception(
                "Unexpected predictor failure "
                "model_id=%s",
                clean_model_id,
            )

            # Do not hide unexpected failures behind a fake successful
            # prediction. We still provide a deterministic fallback when
            # there is enough time to do so.
            self._ensure_deadline(deadline)

            probability = self._heuristic_fallback(
                features
            )

            identity = ModelIdentity(
                model_id=clean_model_id,
                version=(
                    runtime.version
                    if runtime
                    else DEFAULT_UNKNOWN_MODEL_VERSION
                ),
                artifact_hash=(
                    runtime.artifact_hash
                    if runtime
                    else DEFAULT_UNKNOWN_ARTIFACT_HASH
                ),
            )

            telemetry_recorded = (
                self._record_drift(
                    identity,
                    probability,
                    features,
                    time.time(),
                )
            )

            self._ensure_deadline(deadline)

            return PredictionResult(
                model_id=clean_model_id,
                probability=probability,
                verdict=self._probability_to_verdict(
                    probability
                ),
                execution_time_ms=self._execution_time_ms(
                    start_monotonic
                ),
                features_used=self._bounded_feature_metadata(
                    features
                ),
                source=PredictionSource.HEURISTIC_FALLBACK,
                status=PredictionStatus.FAILED,
                model_version=identity.version,
                artifact_hash=identity.artifact_hash,
                feature_count=len(features),
                fallback_used=True,
                error_code=PredictionErrorCode.UNKNOWN_FAILURE,
                request_id=request_id,
                model_feature_count=(
                    len(runtime.feature_names)
                    if runtime
                    else 0
                ),
                missing_feature_count=0,
                telemetry_recorded=(
                    telemetry_recorded
                ),
            )


# ============================================================================
# Internal Pipeline Exception
# ============================================================================


class _PredictionPipelineError(Exception):
    """
    Internal typed exception used to propagate stable predictor error codes
    without exposing implementation-specific exception names.
    """

    def __init__(
        self,
        code: PredictionErrorCode,
    ) -> None:
        super().__init__(code.value)
        self.code = code


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "DEFAULT_CRITICAL_THRESHOLD",
    "DEFAULT_FALLBACK_MAXIMUM",
    "DEFAULT_FALLBACK_MINIMUM",
    "DEFAULT_HIGH_THRESHOLD",
    "DEFAULT_MEDIUM_THRESHOLD",
    "MLPredictor",
    "PredictionErrorCode",
    "PredictionResult",
    "PredictionSource",
    "PredictionStatus",
]
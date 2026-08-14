"""
PhishGuard_AI - ML Drift Monitor & Concept Drift Detector
Version: 10.0.0

Production-grade drift monitoring subsystem for PhishGuard_AI.

This module monitors:

    - Prediction distribution drift
    - Numeric feature distribution drift
    - Feature availability/schema drift
    - Concept/performance drift
    - Probability calibration drift
    - Model/artifact isolation
    - Baseline compatibility
    - Rolling inference windows
    - Drift severity and health state
    - Alert hysteresis / repeated-window confirmation
    - Operational telemetry
    - Bounded memory usage
    - Thread-safe in-process operation

Important architecture note
---------------------------
This component is intentionally an IN-PROCESS drift engine.

In a horizontally scaled deployment, state must eventually be backed by
a shared telemetry/metrics system or persistent state store. Do not assume
that separate Kubernetes replicas share this monitor's memory.

Security goals
--------------
1. Reject NaN/Inf values.
2. Reject invalid probabilities.
3. Reject invalid labels rather than coercing them.
4. Bind baselines to concrete model artifacts.
5. Bind baselines to feature-schema versions.
6. Prevent unbounded model-state creation.
7. Bound all in-memory observation windows.
8. Avoid retaining caller-owned mutable objects.
9. Avoid raw sensitive telemetry where possible.
10. Produce structured drift events suitable for alerting/retraining.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_WINDOW_SIZE = 1_000
DEFAULT_PSI_BUCKETS = 10

DEFAULT_WARNING_PSI = 0.10
DEFAULT_CRITICAL_PSI = 0.25

DEFAULT_MIN_BASELINE_SAMPLES = 100
DEFAULT_MIN_TARGET_SAMPLES = 50

DEFAULT_EPSILON = 1e-6

DEFAULT_WARNING_ACCURACY_DEGRADATION = 0.05
DEFAULT_CRITICAL_ACCURACY_DEGRADATION = 0.10

DEFAULT_WARNING_BRIER_DEGRADATION = 0.05
DEFAULT_CRITICAL_BRIER_DEGRADATION = 0.10

DEFAULT_CONFIRMATION_WINDOWS = 2

DEFAULT_MAX_MODELS = 128
DEFAULT_MAX_FEATURES_PER_INFERENCE = 2_000
DEFAULT_MAX_FEATURES_PER_MODEL = 4_000

DEFAULT_MAX_MODEL_ID_LENGTH = 256
DEFAULT_MAX_VERSION_LENGTH = 128
DEFAULT_MAX_HASH_LENGTH = 256
DEFAULT_MAX_SCHEMA_VERSION_LENGTH = 128

DEFAULT_MAX_TIMESTAMP_SKEW_SECONDS = 300.0


# ============================================================================
# Enums
# ============================================================================


class DriftSeverity(str, Enum):
    """
    Operational severity of detected drift.
    """

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class DriftType(str, Enum):
    """
    Type of detected model/data drift.
    """

    PREDICTION = "prediction"
    FEATURE = "feature"
    SCHEMA = "schema"
    CONCEPT = "concept"
    CALIBRATION = "calibration"


class DriftHealth(str, Enum):
    """
    Overall model health state.

    UNKNOWN
        Not enough evidence.

    HEALTHY
        No actionable drift.

    WARNING
        Drift requires investigation.

    DEGRADED
        Persistent significant drift.

    CRITICAL
        Severe or confirmed model degradation.
    """

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"


# ============================================================================
# Identity / Schema
# ============================================================================


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """
    Immutable identity for one concrete model artifact.
    """

    model_id: str
    version: str = ""
    artifact_hash: str = ""

    def __post_init__(self) -> None:
        model_id = self.model_id.strip()
        version = self.version.strip()
        artifact_hash = self.artifact_hash.strip()

        if not model_id:
            raise ValueError("model_id must not be empty")

        if len(model_id) > DEFAULT_MAX_MODEL_ID_LENGTH:
            raise ValueError("model_id exceeds maximum length")

        if len(version) > DEFAULT_MAX_VERSION_LENGTH:
            raise ValueError("model version exceeds maximum length")

        if len(artifact_hash) > DEFAULT_MAX_HASH_LENGTH:
            raise ValueError("artifact_hash exceeds maximum length")

        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "artifact_hash", artifact_hash)

    @property
    def key(self) -> Tuple[str, str, str]:
        """
        Structured identity key.

        Tuple keys avoid delimiter-based collisions.
        """
        return (
            self.model_id,
            self.version,
            self.artifact_hash,
        )


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """
    Describes the expected feature set for a model artifact.

    feature_names:
        Canonical ordered feature names.

    version:
        Version of the feature extraction schema.

    This is intentionally separate from ModelIdentity because a model
    artifact and its feature schema are related but distinct concepts.
    """

    version: str
    feature_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        version = self.version.strip()

        if not version:
            raise ValueError("feature schema version must not be empty")

        if len(version) > DEFAULT_MAX_SCHEMA_VERSION_LENGTH:
            raise ValueError("feature schema version exceeds maximum length")

        normalized = tuple(
            name.strip()
            for name in self.feature_names
        )

        if any(not name for name in normalized):
            raise ValueError(
                "feature schema contains an empty feature name"
            )

        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "feature schema contains duplicate feature names"
            )

        if len(normalized) > DEFAULT_MAX_FEATURES_PER_MODEL:
            raise ValueError(
                "feature schema exceeds maximum feature count"
            )

        object.__setattr__(self, "version", version)
        object.__setattr__(self, "feature_names", normalized)

    @property
    def key(self) -> Tuple[str, Tuple[str, ...]]:
        return (
            self.version,
            self.feature_names,
        )


@dataclass(frozen=True, slots=True)
class BaselineIdentity:
    """
    Immutable identity for a registered statistical baseline.
    """

    model: ModelIdentity
    schema: Optional[FeatureSchema]
    created_at: float

    @property
    def key(self) -> Tuple[object, object]:
        return (
            self.model.key,
            self.schema.key if self.schema else None,
        )


# ============================================================================
# Observation Models
# ============================================================================


@dataclass(frozen=True, slots=True)
class PredictionMetricPoint:
    """
    Immutable inference telemetry point.

    features_extracted is represented as a tuple to prevent callers from
    mutating stored telemetry after insertion.
    """

    timestamp: float
    model: ModelIdentity
    predicted_probability: float
    features_extracted: Tuple[Tuple[str, float], ...] = ()

    @property
    def features(self) -> Dict[str, float]:
        """
        Materialize feature mapping only when explicitly requested.
        """
        return dict(self.features_extracted)


# ============================================================================
# Result Models
# ============================================================================


@dataclass(frozen=True, slots=True)
class PSIResult:
    psi: float
    baseline_samples: int
    target_samples: int
    buckets: int
    severity: DriftSeverity
    sufficient_data: bool
    excluded_baseline_samples: int = 0
    excluded_target_samples: int = 0

    def __float__(self) -> float:
        return self.psi


@dataclass(frozen=True, slots=True)
class FeatureDriftResult:
    feature_name: str
    psi: float
    baseline_samples: int
    target_samples: int
    severity: DriftSeverity
    sufficient_data: bool
    availability_ratio: float = 1.0


@dataclass(frozen=True, slots=True)
class SchemaDriftResult:
    missing_features: Tuple[str, ...]
    unexpected_features: Tuple[str, ...]
    missing_ratio: float
    unexpected_ratio: float
    severity: DriftSeverity
    sufficient_data: bool


@dataclass(frozen=True, slots=True)
class ConceptDriftResult:
    samples: int
    accuracy: Optional[float]
    error_rate: Optional[float]
    baseline_accuracy: Optional[float]
    degradation: Optional[float]
    severity: DriftSeverity
    sufficient_data: bool


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    samples: int
    brier_score: Optional[float]
    baseline_brier_score: Optional[float]
    degradation: Optional[float]
    severity: DriftSeverity
    sufficient_data: bool


@dataclass(frozen=True, slots=True)
class DriftEvent:
    """
    Structured drift event.

    Suitable for:
        observability.alerting
        audit logging
        retrain_trigger.py
        SIEM integration
    """

    event_id: str
    model: ModelIdentity
    drift_type: DriftType
    severity: DriftSeverity
    created_at: float
    metric_name: str
    metric_value: float
    message: str
    confirmation_count: int


@dataclass(frozen=True, slots=True)
class DriftReport:
    model: ModelIdentity
    generated_at: float

    health: DriftHealth
    overall_severity: DriftSeverity

    prediction_drift: PSIResult
    feature_drifts: Tuple[FeatureDriftResult, ...]
    schema_drift: Optional[SchemaDriftResult]
    concept_drift: Optional[ConceptDriftResult]
    calibration_drift: Optional[CalibrationResult]

    recent_sample_count: int
    monitored_feature_count: int
    last_inference_timestamp: Optional[float]

    events: Tuple[DriftEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class DriftMonitorSnapshot:
    generated_at: float
    registered_models: int
    active_windows: int
    total_observations: int
    rejected_observations: int
    evaluations: int
    drift_events: int
    models: Tuple[str, ...]


# ============================================================================
# Internal State
# ============================================================================


@dataclass(slots=True)
class _ModelState:
    identity: ModelIdentity

    history: Deque[PredictionMetricPoint]

    baseline_probabilities: Tuple[float, ...] = ()

    baseline_features: Dict[str, Tuple[float, ...]] = field(
        default_factory=dict
    )

    baseline_accuracy: Optional[float] = None

    baseline_brier_score: Optional[float] = None

    baseline_identity: Optional[BaselineIdentity] = None

    feature_schema: Optional[FeatureSchema] = None

    labeled_predictions: Deque[Tuple[float, bool]] = field(
        default_factory=deque
    )

    total_observations: int = 0
    rejected_observations: int = 0

    last_inference_timestamp: Optional[float] = None

    health: DriftHealth = DriftHealth.UNKNOWN

    confirmation_counts: Dict[DriftType, int] = field(
        default_factory=dict
    )

    emitted_event_keys: set[Tuple[DriftType, DriftSeverity]] = field(
        default_factory=set
    )


# ============================================================================
# Drift Monitor
# ============================================================================


class DriftMonitor:
    """
    Thread-safe production drift monitor.

    The monitor is intentionally independent from model inference itself.

    predictor.py
        produces predictions/features

    model_registry.py
        establishes trusted model identity

    drift_monitor.py
        evaluates statistical health

    observability/
        exports telemetry

    mlops_pipeline/
        decides whether retraining is appropriate
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        *,
        psi_buckets: int = DEFAULT_PSI_BUCKETS,
        warning_psi: float = DEFAULT_WARNING_PSI,
        critical_psi: float = DEFAULT_CRITICAL_PSI,
        min_baseline_samples: int = DEFAULT_MIN_BASELINE_SAMPLES,
        min_target_samples: int = DEFAULT_MIN_TARGET_SAMPLES,
        epsilon: float = DEFAULT_EPSILON,
        max_labeled_samples: int = 1_000,
        max_models: int = DEFAULT_MAX_MODELS,
        confirmation_windows: int = DEFAULT_CONFIRMATION_WINDOWS,
        warning_accuracy_degradation: float = (
            DEFAULT_WARNING_ACCURACY_DEGRADATION
        ),
        critical_accuracy_degradation: float = (
            DEFAULT_CRITICAL_ACCURACY_DEGRADATION
        ),
        warning_brier_degradation: float = (
            DEFAULT_WARNING_BRIER_DEGRADATION
        ),
        critical_brier_degradation: float = (
            DEFAULT_CRITICAL_BRIER_DEGRADATION
        ),
        max_timestamp_skew_seconds: float = (
            DEFAULT_MAX_TIMESTAMP_SKEW_SECONDS
        ),
    ) -> None:

        self._validate_configuration(
            window_size=window_size,
            psi_buckets=psi_buckets,
            warning_psi=warning_psi,
            critical_psi=critical_psi,
            min_baseline_samples=min_baseline_samples,
            min_target_samples=min_target_samples,
            epsilon=epsilon,
            max_labeled_samples=max_labeled_samples,
            max_models=max_models,
            confirmation_windows=confirmation_windows,
            warning_accuracy_degradation=(
                warning_accuracy_degradation
            ),
            critical_accuracy_degradation=(
                critical_accuracy_degradation
            ),
            warning_brier_degradation=(
                warning_brier_degradation
            ),
            critical_brier_degradation=(
                critical_brier_degradation
            ),
            max_timestamp_skew_seconds=(
                max_timestamp_skew_seconds
            ),
        )

        self.window_size = window_size
        self.psi_buckets = psi_buckets

        self.warning_psi = warning_psi
        self.critical_psi = critical_psi

        self.min_baseline_samples = min_baseline_samples
        self.min_target_samples = min_target_samples

        self.epsilon = epsilon

        self.max_labeled_samples = max_labeled_samples
        self.max_models = max_models

        self.confirmation_windows = confirmation_windows

        self.warning_accuracy_degradation = (
            warning_accuracy_degradation
        )
        self.critical_accuracy_degradation = (
            critical_accuracy_degradation
        )

        self.warning_brier_degradation = (
            warning_brier_degradation
        )
        self.critical_brier_degradation = (
            critical_brier_degradation
        )

        self.max_timestamp_skew_seconds = (
            max_timestamp_skew_seconds
        )

        self._states: Dict[Tuple[str, str, str], _ModelState] = {}

        self._lock = threading.RLock()

        # Operational counters.
        self._rejected_observations = 0
        self._evaluations = 0
        self._drift_events = 0

    # ========================================================================
    # Configuration Validation
    # ========================================================================

    @staticmethod
    def _validate_configuration(
        *,
        window_size: int,
        psi_buckets: int,
        warning_psi: float,
        critical_psi: float,
        min_baseline_samples: int,
        min_target_samples: int,
        epsilon: float,
        max_labeled_samples: int,
        max_models: int,
        confirmation_windows: int,
        warning_accuracy_degradation: float,
        critical_accuracy_degradation: float,
        warning_brier_degradation: float,
        critical_brier_degradation: float,
        max_timestamp_skew_seconds: float,
    ) -> None:

        if window_size <= 0:
            raise ValueError("window_size must be > 0")

        if psi_buckets < 2:
            raise ValueError("psi_buckets must be >= 2")

        if not (
            math.isfinite(warning_psi)
            and math.isfinite(critical_psi)
        ):
            raise ValueError("PSI thresholds must be finite")

        if warning_psi < 0:
            raise ValueError("warning_psi must be >= 0")

        if critical_psi <= warning_psi:
            raise ValueError(
                "critical_psi must be greater than warning_psi"
            )

        if min_baseline_samples <= 0:
            raise ValueError(
                "min_baseline_samples must be > 0"
            )

        if min_target_samples <= 0:
            raise ValueError(
                "min_target_samples must be > 0"
            )

        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError(
                "epsilon must be positive and finite"
            )

        if max_labeled_samples <= 0:
            raise ValueError(
                "max_labeled_samples must be > 0"
            )

        if max_models <= 0:
            raise ValueError(
                "max_models must be > 0"
            )

        if confirmation_windows <= 0:
            raise ValueError(
                "confirmation_windows must be > 0"
            )

        for value, name in (
            (
                warning_accuracy_degradation,
                "warning_accuracy_degradation",
            ),
            (
                critical_accuracy_degradation,
                "critical_accuracy_degradation",
            ),
            (
                warning_brier_degradation,
                "warning_brier_degradation",
            ),
            (
                critical_brier_degradation,
                "critical_brier_degradation",
            ),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} must be finite and >= 0"
                )

        if (
            critical_accuracy_degradation
            <= warning_accuracy_degradation
        ):
            raise ValueError(
                "critical accuracy degradation must exceed warning"
            )

        if (
            critical_brier_degradation
            <= warning_brier_degradation
        ):
            raise ValueError(
                "critical Brier degradation must exceed warning"
            )

        if (
            not math.isfinite(max_timestamp_skew_seconds)
            or max_timestamp_skew_seconds < 0
        ):
            raise ValueError(
                "max_timestamp_skew_seconds must be finite and >= 0"
            )

    # ========================================================================
    # Validation
    # ========================================================================

    @staticmethod
    def _validate_probability(value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "predicted_probability must be numeric"
            ) from exc

        if not math.isfinite(value):
            raise ValueError(
                "predicted_probability must be finite"
            )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "predicted_probability must be in [0, 1]"
            )

        return value

    @staticmethod
    def _validate_feature(
        name: str,
        value: float,
    ) -> Tuple[str, float]:

        if not isinstance(name, str):
            raise TypeError(
                "feature name must be a string"
            )

        name = name.strip()

        if not name:
            raise ValueError(
                "feature name must not be empty"
            )

        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"feature '{name}' must be numeric"
            ) from exc

        if not math.isfinite(value):
            raise ValueError(
                f"feature '{name}' must be finite"
            )

        return name, value

    def _sanitize_features(
        self,
        features: Mapping[str, float],
    ) -> Tuple[Tuple[str, float], ...]:

        if not isinstance(features, Mapping):
            raise TypeError(
                "features_extracted must be a mapping"
            )

        if len(features) > DEFAULT_MAX_FEATURES_PER_INFERENCE:
            raise ValueError(
                "feature count exceeds maximum allowed"
            )

        sanitized = [
            self._validate_feature(name, value)
            for name, value in features.items()
        ]

        sanitized.sort(key=lambda item: item[0])

        return tuple(sanitized)

    @staticmethod
    def _coerce_identity(
        model: str | ModelIdentity,
    ) -> ModelIdentity:

        if isinstance(model, ModelIdentity):
            return model

        return ModelIdentity(
            model_id=str(model)
        )

    # ========================================================================
    # State
    # ========================================================================

    def _get_or_create_state(
        self,
        identity: ModelIdentity,
    ) -> _ModelState:

        state = self._states.get(identity.key)

        if state is not None:
            return state

        if len(self._states) >= self.max_models:
            raise RuntimeError(
                "maximum monitored model count reached"
            )

        state = _ModelState(
            identity=identity,
            history=deque(
                maxlen=self.window_size
            ),
            labeled_predictions=deque(
                maxlen=self.max_labeled_samples
            ),
        )

        self._states[identity.key] = state

        return state

    # ========================================================================
    # Baseline
    # ========================================================================

    def register_baseline(
        self,
        model: ModelIdentity,
        baseline_probs: Sequence[float],
        baseline_features: Optional[
            Mapping[str, Sequence[float]]
        ] = None,
        *,
        feature_schema: Optional[FeatureSchema] = None,
        baseline_accuracy: Optional[float] = None,
        baseline_brier_score: Optional[float] = None,
        replace: bool = False,
    ) -> BaselineIdentity:

        if not isinstance(model, ModelIdentity):
            raise TypeError(
                "model must be a ModelIdentity"
            )

        if len(baseline_probs) < self.min_baseline_samples:
            raise ValueError(
                "insufficient baseline probability samples"
            )

        probabilities = tuple(
            self._validate_probability(value)
            for value in baseline_probs
        )

        sanitized_features: Dict[
            str,
            Tuple[float, ...],
        ] = {}

        if baseline_features:

            if len(baseline_features) > DEFAULT_MAX_FEATURES_PER_MODEL:
                raise ValueError(
                    "baseline feature count exceeds maximum"
                )

            for name, values in baseline_features.items():

                name = name.strip()

                if not name:
                    raise ValueError(
                        "baseline feature name cannot be empty"
                    )

                if len(values) < self.min_baseline_samples:
                    raise ValueError(
                        f"feature '{name}' has insufficient baseline samples"
                    )

                sanitized_features[name] = tuple(
                    self._validate_feature(
                        name,
                        value,
                    )[1]
                    for value in values
                )

        if feature_schema:

            expected = set(
                feature_schema.feature_names
            )

            actual = set(
                sanitized_features
            )

            missing = expected - actual

            if missing:
                raise ValueError(
                    "baseline is missing schema features: "
                    + ", ".join(sorted(missing))
                )

        if baseline_accuracy is not None:

            baseline_accuracy = float(
                baseline_accuracy
            )

            if not (
                math.isfinite(baseline_accuracy)
                and 0.0 <= baseline_accuracy <= 1.0
            ):
                raise ValueError(
                    "baseline_accuracy must be in [0, 1]"
                )

        if baseline_brier_score is not None:

            baseline_brier_score = float(
                baseline_brier_score
            )

            if not (
                math.isfinite(baseline_brier_score)
                and 0.0 <= baseline_brier_score <= 1.0
            ):
                raise ValueError(
                    "baseline_brier_score must be in [0, 1]"
                )

        baseline_identity = BaselineIdentity(
            model=model,
            schema=feature_schema,
            created_at=time.time(),
        )

        with self._lock:

            state = self._get_or_create_state(
                model
            )

            if (
                state.baseline_identity is not None
                and not replace
            ):
                raise RuntimeError(
                    "baseline already exists; "
                    "use replace=True to explicitly replace it"
                )

            state.baseline_probabilities = probabilities

            state.baseline_features = sanitized_features

            state.feature_schema = feature_schema

            state.baseline_accuracy = (
                baseline_accuracy
            )

            state.baseline_brier_score = (
                baseline_brier_score
            )

            state.baseline_identity = (
                baseline_identity
            )

            state.health = DriftHealth.UNKNOWN

            state.confirmation_counts.clear()

            state.emitted_event_keys.clear()

        logger.info(
            "Registered baseline model=%s version=%s hash=%s",
            model.model_id,
            model.version,
            model.artifact_hash,
        )

        return baseline_identity

    # ========================================================================
    # Inference Recording
    # ========================================================================

    def record_inference(
        self,
        model: str | ModelIdentity,
        predicted_probability: float,
        features_extracted: Mapping[str, float],
        *,
        model_version: str = "",
        artifact_hash: str = "",
        timestamp: Optional[float] = None,
    ) -> None:

        if isinstance(model, ModelIdentity):
            identity = model
        else:
            identity = ModelIdentity(
                model_id=str(model),
                version=model_version,
                artifact_hash=artifact_hash,
            )

        try:

            probability = (
                self._validate_probability(
                    predicted_probability
                )
            )

            features = (
                self._sanitize_features(
                    features_extracted
                )
            )

            observation_time = (
                time.time()
                if timestamp is None
                else float(timestamp)
            )

            if not math.isfinite(
                observation_time
            ):
                raise ValueError(
                    "timestamp must be finite"
                )

            if (
                abs(
                    observation_time
                    - time.time()
                )
                > self.max_timestamp_skew_seconds
            ):
                raise ValueError(
                    "timestamp exceeds permitted clock skew"
                )

        except (TypeError, ValueError):

            with self._lock:
                self._rejected_observations += 1

            raise

        point = PredictionMetricPoint(
            timestamp=observation_time,
            model=identity,
            predicted_probability=probability,
            features_extracted=features,
        )

        with self._lock:

            state = self._get_or_create_state(
                identity
            )

            state.history.append(point)

            state.total_observations += 1

            state.last_inference_timestamp = (
                observation_time
            )

    # ========================================================================
    # Label Feedback
    # ========================================================================

    def record_label_feedback(
        self,
        model: str | ModelIdentity,
        predicted_probability: float,
        actual_label: bool,
    ) -> None:

        if not isinstance(actual_label, bool):
            raise TypeError(
                "actual_label must be a boolean"
            )

        identity = self._coerce_identity(
            model
        )

        probability = (
            self._validate_probability(
                predicted_probability
            )
        )

        with self._lock:

            state = self._get_or_create_state(
                identity
            )

            state.labeled_predictions.append(
                (
                    probability,
                    actual_label,
                )
            )

    # ========================================================================
    # Retrieval
    # ========================================================================

    def get_recent_probs(
        self,
        model: str | ModelIdentity,
    ) -> List[float]:

        identity = self._coerce_identity(
            model
        )

        with self._lock:

            state = self._states.get(
                identity.key
            )

            if state is None:
                return []

            return [
                point.predicted_probability
                for point in state.history
            ]

    # ========================================================================
    # PSI
    # ========================================================================

    def calculate_psi(
        self,
        baseline_probs: Sequence[float],
        target_probs: Sequence[float],
        buckets: Optional[int] = None,
    ) -> float:

        result = (
            self._calculate_fixed_range_psi(
                baseline_probs,
                target_probs,
                buckets=(
                    buckets
                    or self.psi_buckets
                ),
                lower=0.0,
                upper=1.0,
            )
        )

        return result.psi

    def _calculate_fixed_range_psi(
        self,
        baseline_values: Sequence[float],
        target_values: Sequence[float],
        *,
        buckets: int,
        lower: float,
        upper: float,
    ) -> PSIResult:

        if buckets < 2:
            raise ValueError(
                "buckets must be >= 2"
            )

        baseline = []
        target = []

        excluded_baseline = 0
        excluded_target = 0

        for value in baseline_values:

            value = float(value)

            if not math.isfinite(value):
                excluded_baseline += 1
                continue

            if value < lower or value > upper:
                excluded_baseline += 1
                continue

            baseline.append(value)

        for value in target_values:

            value = float(value)

            if not math.isfinite(value):
                excluded_target += 1
                continue

            if value < lower or value > upper:
                excluded_target += 1
                continue

            target.append(value)

        if (
            len(baseline)
            < self.min_baseline_samples
            or len(target)
            < self.min_target_samples
        ):

            return PSIResult(
                psi=0.0,
                baseline_samples=len(baseline),
                target_samples=len(target),
                buckets=buckets,
                severity=DriftSeverity.NORMAL,
                sufficient_data=False,
                excluded_baseline_samples=(
                    excluded_baseline
                ),
                excluded_target_samples=(
                    excluded_target
                ),
            )

        width = (
            upper - lower
        ) / buckets

        baseline_counts = [
            0
        ] * buckets

        target_counts = [
            0
        ] * buckets

        for value in baseline:

            index = int(
                (value - lower)
                / width
            )

            index = min(
                index,
                buckets - 1,
            )

            baseline_counts[
                index
            ] += 1

        for value in target:

            index = int(
                (value - lower)
                / width
            )

            index = min(
                index,
                buckets - 1,
            )

            target_counts[
                index
            ] += 1

        psi = self._psi_from_counts(
            baseline_counts,
            target_counts,
        )

        return PSIResult(
            psi=psi,
            baseline_samples=len(baseline),
            target_samples=len(target),
            buckets=buckets,
            severity=self.classify_psi(psi),
            sufficient_data=True,
            excluded_baseline_samples=(
                excluded_baseline
            ),
            excluded_target_samples=(
                excluded_target
            ),
        )

    def _psi_from_counts(
        self,
        baseline_counts: Sequence[int],
        target_counts: Sequence[int],
    ) -> float:

        baseline_total = sum(
            baseline_counts
        )

        target_total = sum(
            target_counts
        )

        if (
            baseline_total <= 0
            or target_total <= 0
        ):
            return 0.0

        psi = 0.0

        for (
            baseline_count,
            target_count,
        ) in zip(
            baseline_counts,
            target_counts,
        ):

            baseline_pct = max(
                baseline_count
                / baseline_total,
                self.epsilon,
            )

            target_pct = max(
                target_count
                / target_total,
                self.epsilon,
            )

            contribution = (
                target_pct
                - baseline_pct
            ) * math.log(
                target_pct
                / baseline_pct
            )

            if math.isfinite(
                contribution
            ):
                psi += contribution

        if not math.isfinite(psi):
            return float("inf")

        return max(
            psi,
            0.0,
        )

    # ========================================================================
    # Numeric Feature PSI
    # ========================================================================

    def calculate_numeric_psi(
        self,
        baseline_values: Sequence[float],
        target_values: Sequence[float],
        *,
        buckets: Optional[int] = None,
    ) -> float:

        buckets = (
            buckets
            or self.psi_buckets
        )

        baseline = sorted(
            float(value)
            for value in baseline_values
            if math.isfinite(
                float(value)
            )
        )

        target = [
            float(value)
            for value in target_values
            if math.isfinite(
                float(value)
            )
        ]

        if (
            len(baseline)
            < self.min_baseline_samples
            or len(target)
            < self.min_target_samples
        ):
            return 0.0

        boundaries = (
            self._quantile_boundaries(
                baseline,
                buckets,
            )
        )

        baseline_counts = [
            0
        ] * (
            len(boundaries) + 1
        )

        target_counts = [
            0
        ] * (
            len(boundaries) + 1
        )

        for value in baseline:

            index = self._find_bucket(
                value,
                boundaries,
            )

            baseline_counts[
                index
            ] += 1

        for value in target:

            index = self._find_bucket(
                value,
                boundaries,
            )

            target_counts[
                index
            ] += 1

        return self._psi_from_counts(
            baseline_counts,
            target_counts,
        )

    @staticmethod
    def _quantile_boundaries(
        values: Sequence[float],
        buckets: int,
    ) -> List[float]:

        if not values:
            return []

        boundaries = []

        n = len(values)

        for index in range(
            1,
            buckets,
        ):

            position = (
                index
                * (n - 1)
                / buckets
            )

            lower = int(
                math.floor(position)
            )

            upper = int(
                math.ceil(position)
            )

            if lower == upper:

                value = values[
                    lower
                ]

            else:

                fraction = (
                    position
                    - lower
                )

                value = (
                    values[lower]
                    + fraction
                    * (
                        values[upper]
                        - values[lower]
                    )
                )

            boundaries.append(
                value
            )

        # Remove duplicate boundaries.
        return sorted(
            set(boundaries)
        )

    @staticmethod
    def _find_bucket(
        value: float,
        boundaries: Sequence[float],
    ) -> int:

        for index, boundary in enumerate(
            boundaries
        ):

            if value <= boundary:
                return index

        return len(boundaries)

    # ========================================================================
    # Feature Schema Drift
    # ========================================================================

    def calculate_schema_drift(
        self,
        state: _ModelState,
    ) -> SchemaDriftResult:

        if state.feature_schema is None:

            return SchemaDriftResult(
                missing_features=(),
                unexpected_features=(),
                missing_ratio=0.0,
                unexpected_ratio=0.0,
                severity=DriftSeverity.NORMAL,
                sufficient_data=False,
            )

        expected = set(
            state.feature_schema.feature_names
        )

        observed = set()

        for point in state.history:

            observed.update(
                name
                for name, _ in
                point.features_extracted
            )

        missing = expected - observed

        unexpected = observed - expected

        expected_count = max(
            len(expected),
            1,
        )

        missing_ratio = (
            len(missing)
            / expected_count
        )

        unexpected_ratio = (
            len(unexpected)
            / max(
                len(observed),
                1,
            )
        )

        if (
            missing_ratio >= 0.25
            or unexpected_ratio >= 0.25
        ):

            severity = (
                DriftSeverity.CRITICAL
            )

        elif (
            missing_ratio > 0
            or unexpected_ratio > 0
        ):

            severity = (
                DriftSeverity.WARNING
            )

        else:

            severity = (
                DriftSeverity.NORMAL
            )

        return SchemaDriftResult(
            missing_features=tuple(
                sorted(missing)
            ),
            unexpected_features=tuple(
                sorted(unexpected)
            ),
            missing_ratio=missing_ratio,
            unexpected_ratio=unexpected_ratio,
            severity=severity,
            sufficient_data=bool(
                state.history
            ),
        )

    # ========================================================================
    # Feature Drift
    # ========================================================================

    def calculate_feature_drift(
        self,
        model: str | ModelIdentity,
    ) -> Tuple[
        FeatureDriftResult,
        ...
    ]:

        identity = self._coerce_identity(
            model
        )

        with self._lock:

            state = self._states.get(
                identity.key
            )

            if (
                state is None
                or not state.baseline_features
            ):
                return ()

            recent: Dict[
                str,
                List[float],
            ] = {}

            for point in state.history:

                for name, value in (
                    point.features_extracted
                ):

                    recent.setdefault(
                        name,
                        [],
                    ).append(value)

            results = []

            for (
                name,
                baseline,
            ) in state.baseline_features.items():

                target = recent.get(
                    name,
                    [],
                )

                psi = (
                    self.calculate_numeric_psi(
                        baseline,
                        target,
                    )
                )

                availability = (
                    len(target)
                    / max(
                        len(state.history),
                        1,
                    )
                )

                severity = (
                    self.classify_psi(
                        psi
                    )
                )

                if (
                    availability
                    < 0.75
                    and severity
                    == DriftSeverity.NORMAL
                ):
                    severity = (
                        DriftSeverity.WARNING
                    )

                results.append(
                    FeatureDriftResult(
                        feature_name=name,
                        psi=psi,
                        baseline_samples=len(
                            baseline
                        ),
                        target_samples=len(
                            target
                        ),
                        severity=severity,
                        sufficient_data=(
                            len(baseline)
                            >= self.min_baseline_samples
                            and len(target)
                            >= self.min_target_samples
                        ),
                        availability_ratio=(
                            availability
                        ),
                    )
                )

            results.sort(
                key=lambda item: item.psi,
                reverse=True,
            )

            return tuple(
                results
            )

    # ========================================================================
    # Prediction Drift
    # ========================================================================

    def calculate_prediction_drift(
        self,
        model: str | ModelIdentity,
    ) -> PSIResult:

        identity = self._coerce_identity(
            model
        )

        with self._lock:

            state = self._states.get(
                identity.key
            )

            if state is None:

                return PSIResult(
                    psi=0.0,
                    baseline_samples=0,
                    target_samples=0,
                    buckets=self.psi_buckets,
                    severity=DriftSeverity.NORMAL,
                    sufficient_data=False,
                )

            target = [
                point.predicted_probability
                for point in state.history
            ]

            return (
                self._calculate_fixed_range_psi(
                    state.baseline_probabilities,
                    target,
                    buckets=self.psi_buckets,
                    lower=0.0,
                    upper=1.0,
                )
            )

    # ========================================================================
    # Concept Drift
    # ========================================================================

    def calculate_concept_drift(
        self,
        model: str | ModelIdentity,
        *,
        classification_threshold: float = 0.5,
    ) -> ConceptDriftResult:

        if not (
            0.0
            < classification_threshold
            < 1.0
        ):
            raise ValueError(
                "classification_threshold must be in (0, 1)"
            )

        identity = self._coerce_identity(
            model
        )

        with self._lock:

            state = self._states.get(
                identity.key
            )

            if state is None:

                return ConceptDriftResult(
                    samples=0,
                    accuracy=None,
                    error_rate=None,
                    baseline_accuracy=None,
                    degradation=None,
                    severity=DriftSeverity.NORMAL,
                    sufficient_data=False,
                )

            samples = list(
                state.labeled_predictions
            )

            if (
                len(samples)
                < self.min_target_samples
            ):

                return ConceptDriftResult(
                    samples=len(samples),
                    accuracy=None,
                    error_rate=None,
                    baseline_accuracy=(
                        state.baseline_accuracy
                    ),
                    degradation=None,
                    severity=DriftSeverity.NORMAL,
                    sufficient_data=False,
                )

            correct = 0

            for (
                probability,
                actual,
            ) in samples:

                predicted = (
                    probability
                    >= classification_threshold
                )

                if predicted == actual:
                    correct += 1

            accuracy = (
                correct
                / len(samples)
            )

            error_rate = (
                1.0 - accuracy
            )

            baseline = (
                state.baseline_accuracy
            )

            if baseline is None:

                return ConceptDriftResult(
                    samples=len(samples),
                    accuracy=accuracy,
                    error_rate=error_rate,
                    baseline_accuracy=None,
                    degradation=None,
                    severity=DriftSeverity.NORMAL,
                    sufficient_data=False,
                )

            degradation = (
                baseline - accuracy
            )

            if (
                degradation
                >= self.critical_accuracy_degradation
            ):

                severity = (
                    DriftSeverity.CRITICAL
                )

            elif (
                degradation
                >= self.warning_accuracy_degradation
            ):

                severity = (
                    DriftSeverity.WARNING
                )

            else:

                severity = (
                    DriftSeverity.NORMAL
                )

            return ConceptDriftResult(
                samples=len(samples),
                accuracy=accuracy,
                error_rate=error_rate,
                baseline_accuracy=baseline,
                degradation=degradation,
                severity=severity,
                sufficient_data=True,
            )

    # ========================================================================
    # Calibration Drift
    # ========================================================================

    def calculate_calibration_drift(
        self,
        model: str | ModelIdentity,
    ) -> CalibrationResult:

        identity = self._coerce_identity(
            model
        )

        with self._lock:

            state = self._states.get(
                identity.key
            )

            if state is None:

                return CalibrationResult(
                    samples=0,
                    brier_score=None,
                    baseline_brier_score=None,
                    degradation=None,
                    severity=DriftSeverity.NORMAL,
                    sufficient_data=False,
                )

            samples = list(
                state.labeled_predictions
            )

            if (
                len(samples)
                < self.min_target_samples
            ):

                return CalibrationResult(
                    samples=len(samples),
                    brier_score=None,
                    baseline_brier_score=(
                        state.baseline_brier_score
                    ),
                    degradation=None,
                    severity=DriftSeverity.NORMAL,
                    sufficient_data=False,
                )

            brier = sum(
                (
                    probability
                    - float(actual)
                ) ** 2
                for probability, actual
                in samples
            ) / len(samples)

            baseline = (
                state.baseline_brier_score
            )

            if baseline is None:

                return CalibrationResult(
                    samples=len(samples),
                    brier_score=brier,
                    baseline_brier_score=None,
                    degradation=None,
                    severity=DriftSeverity.NORMAL,
                    sufficient_data=False,
                )

            degradation = (
                brier - baseline
            )

            if (
                degradation
                >= self.critical_brier_degradation
            ):

                severity = (
                    DriftSeverity.CRITICAL
                )

            elif (
                degradation
                >= self.warning_brier_degradation
            ):

                severity = (
                    DriftSeverity.WARNING
                )

            else:

                severity = (
                    DriftSeverity.NORMAL
                )

            return CalibrationResult(
                samples=len(samples),
                brier_score=brier,
                baseline_brier_score=baseline,
                degradation=degradation,
                severity=severity,
                sufficient_data=True,
            )

    # ========================================================================
    # Severity
    # ========================================================================

    def classify_psi(
        self,
        psi: float,
    ) -> DriftSeverity:

        if (
            not math.isfinite(psi)
            or psi < 0
        ):
            return DriftSeverity.CRITICAL

        if psi >= self.critical_psi:
            return DriftSeverity.CRITICAL

        if psi >= self.warning_psi:
            return DriftSeverity.WARNING

        return DriftSeverity.NORMAL

    @staticmethod
    def _max_severity(
        values: Iterable[
            DriftSeverity
        ],
    ) -> DriftSeverity:

        ranking = {
            DriftSeverity.NORMAL: 0,
            DriftSeverity.WARNING: 1,
            DriftSeverity.CRITICAL: 2,
        }

        highest = DriftSeverity.NORMAL

        for value in values:

            if (
                ranking[value]
                > ranking[highest]
            ):
                highest = value

        return highest

    # ========================================================================
    # Event Generation
    # ========================================================================

    def _create_event(
        self,
        state: _ModelState,
        drift_type: DriftType,
        severity: DriftSeverity,
        metric_name: str,
        metric_value: float,
        message: str,
    ) -> Optional[DriftEvent]:

        if severity == DriftSeverity.NORMAL:
            return None

        current_count = (
            state.confirmation_counts.get(
                drift_type,
                0,
            )
        )

        current_count += 1

        state.confirmation_counts[
            drift_type
        ] = current_count

        if (
            current_count
            < self.confirmation_windows
        ):
            return None

        key = (
            drift_type,
            severity,
        )

        if key in state.emitted_event_keys:
            return None

        event_id = (
            f"{state.identity.model_id}:"
            f"{state.identity.version}:"
            f"{drift_type.value}:"
            f"{int(time.time() * 1000)}"
        )

        event = DriftEvent(
            event_id=event_id,
            model=state.identity,
            drift_type=drift_type,
            severity=severity,
            created_at=time.time(),
            metric_name=metric_name,
            metric_value=float(
                metric_value
            ),
            message=message,
            confirmation_count=current_count,
        )

        state.emitted_event_keys.add(
            key
        )

        self._drift_events += 1

        logger.warning(
            "ML drift event model=%s type=%s severity=%s metric=%s value=%s",
            state.identity.model_id,
            drift_type.value,
            severity.value,
            metric_name,
            metric_value,
        )

        return event

    # ========================================================================
    # Complete Evaluation
    # ========================================================================

    def evaluate(
        self,
        model: str | ModelIdentity,
    ) -> DriftReport:

        identity = self._coerce_identity(
            model
        )

        with self._lock:

            state = self._states.get(
                identity.key
            )

            if state is None:

                empty = PSIResult(
                    psi=0.0,
                    baseline_samples=0,
                    target_samples=0,
                    buckets=self.psi_buckets,
                    severity=DriftSeverity.NORMAL,
                    sufficient_data=False,
                )

                return DriftReport(
                    model=identity,
                    generated_at=time.time(),
                    health=DriftHealth.UNKNOWN,
                    overall_severity=DriftSeverity.NORMAL,
                    prediction_drift=empty,
                    feature_drifts=(),
                    schema_drift=None,
                    concept_drift=None,
                    calibration_drift=None,
                    recent_sample_count=0,
                    monitored_feature_count=0,
                    last_inference_timestamp=None,
                )

            self._evaluations += 1

            prediction = (
                self.calculate_prediction_drift(
                    identity
                )
            )

            features = (
                self.calculate_feature_drift(
                    identity
                )
            )

            schema = (
                self.calculate_schema_drift(
                    state
                )
            )

            concept = (
                self.calculate_concept_drift(
                    identity
                )
            )

            calibration = (
                self.calculate_calibration_drift(
                    identity
                )
            )

            severities = [
                prediction.severity,
            ]

            severities.extend(
                result.severity
                for result in features
                if result.sufficient_data
            )

            if schema.sufficient_data:
                severities.append(
                    schema.severity
                )

            if concept.sufficient_data:
                severities.append(
                    concept.severity
                )

            if calibration.sufficient_data:
                severities.append(
                    calibration.severity
                )

            overall = (
                self._max_severity(
                    severities
                )
            )

            events: List[
                DriftEvent
            ] = []

            event = self._create_event(
                state,
                DriftType.PREDICTION,
                prediction.severity,
                "prediction_psi",
                prediction.psi,
                "Prediction probability distribution drift detected.",
            )

            if event:
                events.append(event)

            if schema.sufficient_data:

                event = self._create_event(
                    state,
                    DriftType.SCHEMA,
                    schema.severity,
                    "missing_feature_ratio",
                    schema.missing_ratio,
                    "Feature schema drift detected.",
                )

                if event:
                    events.append(event)

            if (
                concept.sufficient_data
            ):

                event = self._create_event(
                    state,
                    DriftType.CONCEPT,
                    concept.severity,
                    "accuracy_degradation",
                    concept.degradation or 0.0,
                    "Validated model performance degradation detected.",
                )

                if event:
                    events.append(event)

            if (
                calibration.sufficient_data
            ):

                event = self._create_event(
                    state,
                    DriftType.CALIBRATION,
                    calibration.severity,
                    "brier_score_degradation",
                    calibration.degradation or 0.0,
                    "Prediction calibration degradation detected.",
                )

                if event:
                    events.append(event)

            # Feature event only needs the most severe feature.
            if features:

                worst_feature = max(
                    features,
                    key=lambda item: item.psi,
                )

                event = self._create_event(
                    state,
                    DriftType.FEATURE,
                    worst_feature.severity,
                    worst_feature.feature_name,
                    worst_feature.psi,
                    (
                        "Feature distribution drift detected "
                        f"for '{worst_feature.feature_name}'."
                    ),
                )

                if event:
                    events.append(event)

            if (
                overall
                == DriftSeverity.CRITICAL
            ):
                state.health = (
                    DriftHealth.CRITICAL
                )

            elif (
                overall
                == DriftSeverity.WARNING
            ):
                state.health = (
                    DriftHealth.WARNING
                )

            elif not state.baseline_identity:
                state.health = (
                    DriftHealth.UNKNOWN
                )

            else:
                state.health = (
                    DriftHealth.HEALTHY
                )

            return DriftReport(
                model=identity,
                generated_at=time.time(),
                health=state.health,
                overall_severity=overall,
                prediction_drift=prediction,
                feature_drifts=features,
                schema_drift=schema,
                concept_drift=concept,
                calibration_drift=calibration,
                recent_sample_count=len(
                    state.history
                ),
                monitored_feature_count=len(
                    state.baseline_features
                ),
                last_inference_timestamp=(
                    state.last_inference_timestamp
                ),
                events=tuple(events),
            )

    # ========================================================================
    # Operational Snapshot
    # ========================================================================

    def get_snapshot(
        self,
    ) -> DriftMonitorSnapshot:

        with self._lock:

            return DriftMonitorSnapshot(
                generated_at=time.time(),
                registered_models=len(
                    self._states
                ),
                active_windows=sum(
                    1
                    for state
                    in self._states.values()
                    if state.history
                ),
                total_observations=sum(
                    state.total_observations
                    for state
                    in self._states.values()
                ),
                rejected_observations=(
                    self._rejected_observations
                ),
                evaluations=self._evaluations,
                drift_events=self._drift_events,
                models=tuple(
                    sorted(
                        state.identity.model_id
                        for state
                        in self._states.values()
                    )
                ),
            )

    # Backward-compatible alias.
    snapshot = get_snapshot

    # ========================================================================
    # State Management
    # ========================================================================

    def reset_history(
        self,
        model: str | ModelIdentity,
    ) -> None:

        identity = self._coerce_identity(
            model
        )

        with self._lock:

            state = self._states.get(
                identity.key
            )

            if state is None:
                return

            state.history.clear()
            state.labeled_predictions.clear()

            state.confirmation_counts.clear()
            state.emitted_event_keys.clear()

            state.health = (
                DriftHealth.UNKNOWN
            )

    def remove_model(
        self,
        model: str | ModelIdentity,
    ) -> bool:

        identity = self._coerce_identity(
            model
        )

        with self._lock:

            return (
                self._states.pop(
                    identity.key,
                    None,
                )
                is not None
            )

    def list_models(
        self,
    ) -> Tuple[
        ModelIdentity,
        ...
    ]:

        with self._lock:

            return tuple(
                state.identity
                for state
                in self._states.values()
            )


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "BaselineIdentity",
    "CalibrationResult",
    "ConceptDriftResult",
    "DEFAULT_CRITICAL_PSI",
    "DEFAULT_EPSILON",
    "DEFAULT_MIN_BASELINE_SAMPLES",
    "DEFAULT_MIN_TARGET_SAMPLES",
    "DEFAULT_PSI_BUCKETS",
    "DEFAULT_WARNING_PSI",
    "DEFAULT_WINDOW_SIZE",
    "DriftEvent",
    "DriftHealth",
    "DriftMonitor",
    "DriftMonitorSnapshot",
    "DriftReport",
    "DriftSeverity",
    "DriftType",
    "FeatureDriftResult",
    "FeatureSchema",
    "ModelIdentity",
    "PSIResult",
    "PredictionMetricPoint",
    "SchemaDriftResult",
]
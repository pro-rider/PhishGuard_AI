"""
PhishGuard_AI - Production ML Feature Extractor Foundation (v9.1.0)
Hardened contract for all extractors (email/url/attachment).
"""
from __future__ import annotations
import abc
import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

# Import RuleEngine for operator support (temporary mock for sanitize)
try:
    from engine.rule_engine import RuleEvaluator  # type: ignore
    RULE_ENGINE_AVAILABLE = True
except ImportError:
    RuleEvaluator = None  # type: ignore
    RULE_ENGINE_AVAILABLE = False

from engine.orchestrator import ScanContext, ScanType

logger = logging.getLogger(__name__)

# ============================================================================
# VERSION / GLOBAL LIMITS (UPDATED)
# ============================================================================
BASE_EXTRACTOR_VERSION = "9.1.0"
MAX_FEATURE_COUNT = 4096
MAX_FEATURE_NAME_LENGTH = 256
MAX_FEATURE_ABS_VALUE = 1.0e12
DEFAULT_EXTRACTION_TIMEOUT_MS = 100.0
MAX_BATCH_SIZE = 256

# ============================================================================
# ENUMERATIONS
# ============================================================================
class FeatureValueStatus(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    INVALID = "invalid"
    NON_FINITE = "non_finite"
    OUT_OF_RANGE = "out_of_range"


class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    name: str
    default: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    required: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Feature name cannot be empty.")
        if len(self.name) > MAX_FEATURE_NAME_LENGTH:
            raise ValueError(f"Feature name exceeds {MAX_FEATURE_NAME_LENGTH} characters.")
        if not _is_finite_number(self.default):
            raise ValueError(f"Default value for feature '{self.name}' must be finite.")
        if self.minimum is not None and not _is_finite_number(self.minimum):
            raise ValueError(f"Minimum value for feature '{self.name}' must be finite.")
        if self.maximum is not None and not _is_finite_number(self.maximum):
            raise ValueError(f"Maximum value for feature '{self.name}' must be finite.")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"Invalid bounds for feature '{self.name}': minimum > maximum.")


@dataclass(slots=True)
class FeatureExtractionReport:
    extractor_name: str
    extractor_version: str
    schema_version: str
    status: ExtractionStatus = ExtractionStatus.SUCCESS
    feature_count: int = 0
    valid_count: int = 0
    missing_count: int = 0
    invalid_count: int = 0
    non_finite_count: int = 0
    out_of_range_count: int = 0
    extraction_time_ms: float = 0.0
    schema_fingerprint: str = ""
    errors: List[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return (
            self.status == ExtractionStatus.SUCCESS
            and self.invalid_count == 0
            and self.non_finite_count == 0
            and self.out_of_range_count == 0
        )


@dataclass(slots=True)
class FeatureExtractionResult:
    features: Dict[str, float]
    report: FeatureExtractionReport
    vector: Optional[List[float]] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _schema_fingerprint(feature_names: Sequence[str]) -> str:
    normalized = "\n".join(feature_names).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


# ============================================================================
# BASE FEATURE EXTRACTOR
# ============================================================================
class BaseFeatureExtractor(abc.ABC):
    extractor_name: str = "base_extractor"
    version: str = BASE_EXTRACTOR_VERSION
    schema_version: str = "base-v1"
    supported_scan_types: frozenset[ScanType] = frozenset()
    feature_definitions: Tuple[FeatureDefinition, ...] = ()

    def __init__(self) -> None:
        self._feature_schema = self._build_feature_schema()

    # ---------------------------------------------------------------------
    # Abstract API
    # ---------------------------------------------------------------------
    @abc.abstractmethod
    def extract_features(self, context: ScanContext) -> Dict[str, Any]:
        """
        Core extraction (must be overridden).
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # Feature Schema
    # ---------------------------------------------------------------------
    def _build_feature_schema(self) -> Mapping[str, FeatureDefinition]:
        definitions = self.feature_definitions
        if len(definitions) > MAX_FEATURE_COUNT:
            raise ValueError(f"{self.extractor_name}: feature schema exceeds {MAX_FEATURE_COUNT} features.")
        schema: Dict[str, FeatureDefinition] = {}
        for definition in definitions:
            if definition.name in schema:
                raise ValueError(f"{self.extractor_name}: duplicate feature '{definition.name}'.")
            schema[definition.name] = definition
        return MappingProxyType(schema)

    @property
    def feature_schema(self) -> Mapping[str, FeatureDefinition]:
        return self._feature_schema

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return tuple(self._feature_schema.keys())

    @property
    def schema_fingerprint(self) -> str:
        return _schema_fingerprint(self.feature_names)

    @property
    def feature_count(self) -> int:
        return len(self._feature_schema)

    # ---------------------------------------------------------------------
    # Compatibility
    # ---------------------------------------------------------------------
    def supports_scan_type(self, scan_type: ScanType) -> bool:
        return scan_type in self.supported_scan_types

    def validate_model_schema(self, model_feature_names: Sequence[str]) -> Tuple[bool, str]:
        if not model_feature_names:
            return False, "MODEL_FEATURE_SCHEMA_EMPTY"
        normalized = tuple(str(name) for name in model_feature_names)
        if normalized != self.feature_names:
            return False, "FEATURE_SCHEMA_MISMATCH"
        return True, "OK"

    # ---------------------------------------------------------------------
    # Deadline
    # ---------------------------------------------------------------------
    def _check_deadline(self, deadline_monotonic: Optional[float]) -> None:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError("EXTRACTION_DEADLINE_EXCEEDED")

    # ---------------------------------------------------------------------
    # Value Sanitization (UPDATED with full operators)
    # ---------------------------------------------------------------------
    def sanitize_feature_value(
        self,
        feature_name: str,
        value: Any,
        operator: str = "equals",
    ) -> Tuple[float, FeatureValueStatus]:
        definition = self._feature_schema.get(feature_name)
        if value is None:
            return float(definition.default) if definition else 0.0, FeatureValueStatus.MISSING

        if not _is_finite_number(value):
            return 0.0, FeatureValueStatus.NON_FINITE if isinstance(value, float) and not math.isfinite(value) else FeatureValueStatus.INVALID

        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0, FeatureValueStatus.INVALID

        if abs(numeric_value) > MAX_FEATURE_ABS_VALUE:
            return 0.0, FeatureValueStatus.OUT_OF_RANGE

        if definition:
            if definition.minimum is not None and numeric_value < definition.minimum:
                return 0.0, FeatureValueStatus.OUT_OF_RANGE
            if definition.maximum is not None and numeric_value > definition.maximum:
                return 0.0, FeatureValueStatus.OUT_OF_RANGE

        return numeric_value, FeatureValueStatus.VALID

    # ---------------------------------------------------------------------
    # Feature Mapping Sanitization (UPDATED)
    # ---------------------------------------------------------------------
    def sanitize_features(
        self,
        features: Mapping[str, Any],
        *,
        strict_schema: bool = False,
        deadline_monotonic: Optional[float] = None,
    ) -> Tuple[Dict[str, float], FeatureExtractionReport]:
        start = time.monotonic()
        report = FeatureExtractionReport(
            extractor_name=self.extractor_name,
            extractor_version=self.version,
            schema_version=self.schema_version,
            schema_fingerprint=self.schema_fingerprint,
        )

        sanitized: Dict[str, float] = {}
        if not isinstance(features, Mapping):
            report.status = ExtractionStatus.FAILED
            report.errors.append("EXTRACTOR_RETURNED_NON_MAPPING")
            report.extraction_time_ms = (time.monotonic() - start) * 1000.0
            return sanitized, report

        if len(features) > MAX_FEATURE_COUNT:
            report.status = ExtractionStatus.FAILED
            report.errors.append("FEATURE_COUNT_LIMIT_EXCEEDED")
            report.extraction_time_ms = (time.monotonic() - start) * 1000.0
            return sanitized, report

        # Deadline check
        self._check_deadline(deadline_monotonic)

        unknown_features = [str(name) for name in features if str(name) not in self._feature_schema]
        if unknown_features:
            if strict_schema:
                report.status = ExtractionStatus.FAILED
                report.errors.append("UNKNOWN_FEATURES")
            for feature_name in unknown_features:
                logger.warning(
                    "Extractor '%s' emitted unknown feature '%s'; ignored.",
                    self.extractor_name,
                    feature_name[:MAX_FEATURE_NAME_LENGTH],
                )

        # Process schema
        for feature_name, definition in self._feature_schema.items():
            if feature_name not in features:
                sanitized[feature_name] = float(definition.default)
                report.missing_count += 1
                continue

            value = features[feature_name]
            numeric_value, status = self.sanitize_feature_value(feature_name, value)
            sanitized[feature_name] = numeric_value

            if status == FeatureValueStatus.VALID:
                report.valid_count += 1
            elif status == FeatureValueStatus.MISSING:
                report.missing_count += 1
            elif status == FeatureValueStatus.NON_FINITE:
                report.non_finite_count += 1
                report.errors.append(f"NON_FINITE_FEATURE:{feature_name}")
            elif status == FeatureValueStatus.OUT_OF_RANGE:
                report.out_of_range_count += 1
                report.errors.append(f"OUT_OF_RANGE_FEATURE:{feature_name}")
            else:
                report.invalid_count += 1
                report.errors.append(f"INVALID_FEATURE:{feature_name}")

        report.feature_count = len(sanitized)

        required_missing = [
            name for name, definition in self._feature_schema.items()
            if definition.required and name not in features
        ]
        if required_missing:
            report.status = ExtractionStatus.PARTIAL
            report.errors.append("REQUIRED_FEATURES_MISSING")

        if report.invalid_count > 0 or report.non_finite_count > 0 or report.out_of_range_count > 0:
            report.status = ExtractionStatus.PARTIAL

        report.extraction_time_ms = (time.monotonic() - start) * 1000.0
        return sanitized, report

    # ---------------------------------------------------------------------
    # Safe Extraction (UPDATED with deadline)
    # ---------------------------------------------------------------------
    def extract(
        self,
        context: ScanContext,
        *,
        deadline_monotonic: Optional[float] = None,
        strict_schema: bool = False,
    ) -> FeatureExtractionResult:
        start = time.monotonic()
        if deadline_monotonic is None:
            deadline_monotonic = start + DEFAULT_EXTRACTION_TIMEOUT_MS / 1000.0

        self._check_deadline(deadline_monotonic)

        try:
            raw_features = self.extract_features(context)
        except Exception as exc:
            logger.exception("Feature extraction failed for extractor '%s'.", self.extractor_name)
            report = FeatureExtractionReport(
                extractor_name=self.extractor_name,
                extractor_version=self.version,
                schema_version=self.schema_version,
                status=ExtractionStatus.FAILED,
                schema_fingerprint=self.schema_fingerprint,
                errors=[f"EXTRACTOR_EXCEPTION:{type(exc).__name__}"],
                extraction_time_ms=(time.monotonic() - start) * 1000.0,
            )
            return FeatureExtractionResult(features={}, report=report)

        # Deadline after extraction
        self._check_deadline(deadline_monotonic)

        sanitized, report = self.sanitize_features(
            raw_features,
            strict_schema=strict_schema,
            deadline_monotonic=deadline_monotonic,
        )
        return FeatureExtractionResult(features=sanitized, report=report)

    # ---------------------------------------------------------------------
    # Vector Construction (UPDATED)
    # ---------------------------------------------------------------------
    def extract_vector(
        self,
        context: ScanContext,
        feature_names: Optional[Sequence[str]] = None,
        *,
        deadline_monotonic: Optional[float] = None,
        strict_schema: bool = True,
    ) -> List[float]:
        requested_names = tuple(
            feature_names if feature_names is not None else self.feature_names
        )
        if not requested_names:
            raise ValueError(f"{self.extractor_name}: empty feature schema.")
        if len(requested_names) > MAX_FEATURE_COUNT:
            raise ValueError(f"{self.extractor_name}: feature vector exceeds {MAX_FEATURE_COUNT} values.")

        if len(set(requested_names)) != len(requested_names):
            raise ValueError(f"{self.extractor_name}: duplicate feature names in requested model schema.")

        if strict_schema and tuple(requested_names) != self.feature_names:
            raise ValueError(f"{self.extractor_name}: model feature schema mismatch.")

        self._check_deadline(deadline_monotonic)

        result = self.extract(context, deadline_monotonic=deadline_monotonic, strict_schema=strict_schema)

        vector: List[float] = []
        for feature_name in requested_names:
            definition = self._feature_schema.get(feature_name)
            if definition is None:
                raise ValueError(f"{self.extractor_name}: unknown feature '{feature_name}'.")
            value = result.features.get(feature_name, definition.default)
            if not _is_finite_number(value):
                value = definition.default
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                numeric_value = 0.0
            vector.append(numeric_value)

        result.vector = vector
        return vector

    # ---------------------------------------------------------------------
    # Full Result API (UPDATED)
    # ---------------------------------------------------------------------
    def extract_with_report(
        self,
        context: ScanContext,
        *,
        deadline_monotonic: Optional[float] = None,
        strict_schema: bool = True,
    ) -> FeatureExtractionResult:
        result = self.extract(
            context,
            deadline_monotonic=deadline_monotonic,
            strict_schema=strict_schema,
        )
        result.vector = [
            result.features.get(name, self._feature_schema[name].default)
            for name in self.feature_names
        ]
        return result

    # ---------------------------------------------------------------------
    # Batch Extraction (UPDATED)
    # ---------------------------------------------------------------------
    def extract_batch(
        self,
        contexts: Sequence[ScanContext],
        *,
        deadline_monotonic: Optional[float] = None,
        strict_schema: bool = True,
    ) -> List[FeatureExtractionResult]:
        if len(contexts) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum of {MAX_BATCH_SIZE}.")

        results: List[FeatureExtractionResult] = []
        for context in contexts:
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                break
            results.append(
                self.extract_with_report(
                    context,
                    deadline_monotonic=deadline_monotonic,
                    strict_schema=strict_schema,
                )
            )
        return results

    # ---------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        return {
            "extractor_name": self.extractor_name,
            "version": self.version,
            "schema_version": self.schema_version,
            "schema_fingerprint": self.schema_fingerprint,
            "feature_count": self.feature_count,
            "feature_names": list(self.feature_names),
            "supported_scan_types": [str(scan_type) for scan_type in self.supported_scan_types],
        }


# ============================================================================
# EXPORTS
# ============================================================================
__all__ = [
    "BASE_EXTRACTOR_VERSION",
    "MAX_BATCH_SIZE",
    "MAX_FEATURE_ABS_VALUE",
    "MAX_FEATURE_COUNT",
    "MAX_FEATURE_NAME_LENGTH",
    "DEFAULT_EXTRACTION_TIMEOUT_MS",
    "FeatureDefinition",
    "FeatureExtractionReport",
    "FeatureExtractionResult",
    "FeatureValueStatus",
    "ExtractionStatus",
    "BaseFeatureExtractor",
]
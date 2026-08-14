"""
PhishGuard_AI - ML Feature Extractors Package (v9.1.0)
Central registry and public API for PhishGuard_AI feature extractors.
"""
from __future__ import annotations
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Tuple, Type

from engine.orchestrator import ScanType
from ml_inference.feature_extractors.attachment_feature_extractor import (
    AttachmentFeatureExtractor,
)
from ml_inference.feature_extractors.base_extractor import (
    BaseFeatureExtractor,
)
from ml_inference.feature_extractors.email_feature_extractor import (
    EmailFeatureExtractor,
    EMAIL_FEATURE_NAMES,
)
from ml_inference.feature_extractors.url_feature_extractor import (
    UrlFeatureExtractor,
    FEATURE_NAMES,
)

# ============================================================================
# PACKAGE METADATA
# ============================================================================
FEATURE_EXTRACTORS_VERSION = "9.1.0"
__version__ = FEATURE_EXTRACTORS_VERSION

# ============================================================================
# EXTRACTOR TYPE
# ============================================================================
ExtractorType = Type[BaseFeatureExtractor]

# ============================================================================
# BUILT-IN EXTRACTORS (v9.1.0)
# ============================================================================
_BUILTIN_EXTRACTORS: Tuple[ExtractorType, ...] = (
    EmailFeatureExtractor,
    UrlFeatureExtractor,
    AttachmentFeatureExtractor,
)

# ============================================================================
# REGISTRY VALIDATION (UPDATED)
# ============================================================================
def _validate_extractor_class(extractor_cls: ExtractorType) -> None:
    if not isinstance(extractor_cls, type):
        raise TypeError(f"Extractor must be a class, got {type(extractor_cls).__name__}")
    if not issubclass(extractor_cls, BaseFeatureExtractor):
        raise TypeError(f"{extractor_cls!r} must inherit from BaseFeatureExtractor")
    extractor_name = getattr(extractor_cls, "extractor_name", None)
    if not isinstance(extractor_name, str) or not extractor_name.strip():
        raise ValueError(f"{extractor_cls.__name__} has an invalid extractor_name")
    version = getattr(extractor_cls, "version", None)
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"{extractor_cls.__name__} has an invalid version")
    supported_scan_types = getattr(extractor_cls, "supported_scan_types", None)
    if not isinstance(supported_scan_types, (set, frozenset)):
        raise ValueError(f"{extractor_cls.__name__}.supported_scan_types must be a set or frozenset")
    for scan_type in supported_scan_types:
        if not isinstance(scan_type, ScanType):
            raise ValueError(f"{extractor_cls.__name__}.supported_scan_types contains invalid value: {scan_type!r}")
    schema_version = getattr(extractor_cls, "schema_version", None)
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise ValueError(f"{extractor_cls.__name__} has an invalid schema_version")
    feature_names = getattr(extractor_cls, "feature_names", None)
    if not isinstance(feature_names, (tuple, list)):
        raise ValueError(f"{extractor_cls.__name__} must have a feature_names property returning a tuple/list")

def _build_registry(extractors: Tuple[ExtractorType, ...]) -> Dict[str, ExtractorType]:
    registry: Dict[str, ExtractorType] = {}
    for extractor_cls in extractors:
        _validate_extractor_class(extractor_cls)
        name = extractor_cls.extractor_name.strip()
        if name in registry:
            previous = registry[name]
            raise RuntimeError(
                f"Duplicate feature extractor registration detected: '{name}' is defined by both "
                f"{previous.__name__} and {extractor_cls.__name__}"
            )
        registry[name] = extractor_cls
    return registry

# ============================================================================
# PUBLIC REGISTRY
# ============================================================================
_EXTRACTOR_REGISTRY: Dict[str, ExtractorType] = _build_registry(_BUILTIN_EXTRACTORS)
EXTRACTOR_REGISTRY: Mapping[str, ExtractorType] = MappingProxyType(_EXTRACTOR_REGISTRY)

# ============================================================================
# REGISTRY ACCESS
# ============================================================================
def get_extractor(extractor_name: str) -> Optional[ExtractorType]:
    if not isinstance(extractor_name, str):
        return None
    normalized_name = extractor_name.strip()
    if not normalized_name:
        return None
    return _EXTRACTOR_REGISTRY.get(normalized_name)

def require_extractor(extractor_name: str) -> ExtractorType:
    extractor_cls = get_extractor(extractor_name)
    if extractor_cls is None:
        available = ", ".join(sorted(_EXTRACTOR_REGISTRY))
        raise LookupError(f"Unknown feature extractor '{extractor_name}'. Available extractors: {available}")
    return extractor_cls

# ============================================================================
# SCAN-TYPE DISCOVERY
# ============================================================================
def get_extractors_for_scan_type(scan_type: ScanType) -> Tuple[ExtractorType, ...]:
    if not isinstance(scan_type, ScanType):
        raise TypeError(f"scan_type must be ScanType, got {type(scan_type).__name__}")
    return tuple(
        extractor_cls
        for extractor_cls in _BUILTIN_EXTRACTORS
        if scan_type in extractor_cls.supported_scan_types
    )

def get_extractor_names_for_scan_type(scan_type: ScanType) -> Tuple[str, ...]:
    return tuple(
        extractor_cls.extractor_name
        for extractor_cls in get_extractors_for_scan_type(scan_type)
    )

# ============================================================================
# EXTRACTOR METADATA
# ============================================================================
def get_extractor_metadata() -> Dict[str, Dict[str, object]]:
    metadata: Dict[str, Dict[str, object]] = {}
    for extractor_cls in _BUILTIN_EXTRACTORS:
        metadata[extractor_cls.extractor_name] = {
            "class_name": extractor_cls.__name__,
            "version": extractor_cls.version,
            "schema_version": getattr(extractor_cls, "schema_version", "unknown"),
            "supported_scan_types": tuple(
                sorted(
                    scan_type.value
                    for scan_type in extractor_cls.supported_scan_types
                )
            ),
            "feature_names": getattr(extractor_cls, "feature_names", []),
        }
    return metadata

# ============================================================================
# EXTRACTOR INSTANCE FACTORY
# ============================================================================
def create_extractor(extractor_name: str) -> BaseFeatureExtractor:
    extractor_cls = require_extractor(extractor_name)
    return extractor_cls()

def create_extractors_for_scan_type(scan_type: ScanType) -> Tuple[BaseFeatureExtractor, ...]:
    return tuple(
        extractor_cls()
        for extractor_cls in get_extractors_for_scan_type(scan_type)
    )

# ============================================================================
# REGISTRY HEALTH CHECK
# ============================================================================
def validate_registry() -> None:
    _build_registry(_BUILTIN_EXTRACTORS)

# Eager validation at import time (production safety)
validate_registry()

# ============================================================================
# PUBLIC EXPORTS (UPDATED)
# ============================================================================
__all__ = [
    "FEATURE_EXTRACTORS_VERSION",
    "__version__",
    "BaseFeatureExtractor",
    "EmailFeatureExtractor",
    "UrlFeatureExtractor",
    "AttachmentFeatureExtractor",
    "EXTRACTOR_REGISTRY",
    "get_extractor",
    "require_extractor",
    "get_extractors_for_scan_type",
    "get_extractor_names_for_scan_type",
    "get_extractor_metadata",
    "create_extractor",
    "create_extractors_for_scan_type",
    "validate_registry",
    # NEW: Exposed schemas for model compatibility
    "EMAIL_FEATURE_NAMES",
    "FEATURE_NAMES",
]
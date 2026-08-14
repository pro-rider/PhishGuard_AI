"""
PhishGuard AI - Production Data Engineering Package
====================================================

Version: 13.0.0
File Path: mlops_pipeline/src/data/__init__.py

Purpose
-------
Public package boundary for the PhishGuard AI data-engineering subsystem.

The ``src.data`` package provides the controlled API surface for:

    * Dataset loading
    * Dataset validation
    * Schema enforcement
    * Data-quality assessment
    * Feature preprocessing
    * Feature transformation
    * Leakage-resistant dataset splitting
    * Temporal validation
    * Stratified validation
    * Cross-validation
    * Dataset/schema contracts

Architecture
------------
This module intentionally acts as a PACKAGE BOUNDARY rather than an
implementation module.

The implementation lives in:

    src.data.loaders
    src.data.validators
    src.data.preprocessors
    src.data.splitters
    src.data.schemas

Consumers should preferentially import public data APIs from ``src.data``
rather than depending on internal module paths.

Example
-------
    from src.data import (
        EmailDataLoader,
        SchemaValidator,
        TemporalTrainTestSplitter,
    )

Security Model
--------------
The data layer operates under a zero-trust assumption.

No dataset should be considered trusted merely because it was loaded
successfully.

The data lifecycle should conceptually follow:

    external source
        |
        v
    loader
        |
        v
    structural validation
        |
        v
    schema validation
        |
        v
    quality validation
        |
        v
    preprocessing
        |
        v
    leakage validation
        |
        v
    deterministic split
        |
        v
    training/evaluation

Important
---------
This module MUST NOT:

    * load datasets automatically
    * access credentials
    * initialize model training
    * execute network requests
    * mutate global random state
    * create files/directories
    * configure logging handlers
    * perform expensive computation during import

Only package metadata, public API bindings, and capability diagnostics
should be established here.

Compatibility
-------------
Pipeline Version:
    13.0.0

Data API Version:
    13.0.0

The public API is intentionally explicit through ``__all__``.

Optional Dependencies
---------------------
Some underlying data implementations may depend on optional packages such
as PyArrow, Pandas, scikit-learn, Pydantic, NumPy, or other ecosystem
libraries.

This package should ideally remain importable when an optional dependency
is unavailable, provided the corresponding implementation is not used.

Therefore this initializer uses guarded imports and exposes capability
diagnostics.

Design Goals
------------
1. Stable public API.
2. Fail-safe package initialization.
3. Explicit exports.
4. Optional dependency awareness.
5. Runtime diagnostics.
6. No import-time side effects.
7. Forward compatibility.
8. Strong typing.
9. Circular-import resistance.
10. Production observability.
11. Controlled API deprecation.
12. Clear ownership of implementation modules.

"""

from __future__ import annotations

# ==============================================================================
# STANDARD LIBRARY IMPORTS
# ==============================================================================

import importlib
import platform
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import (
    Any,
    Dict,
    Final,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

# ==============================================================================
# PACKAGE METADATA
# ==============================================================================

__title__: Final[str] = "PhishGuard_AI_Data"
__package_name__: Final[str] = "src.data"

__version__: Final[str] = "13.0.0"
__api_version__: Final[str] = "13.0.0"
__pipeline_version__: Final[str] = "13.0.0"

__author__: Final[str] = "PhishGuard AI SecOps & MLOps Team"

__license__: Final[str] = "Proprietary / Enterprise Closed-Source"

__description__: Final[str] = (
    "Production-grade zero-trust data engineering, validation, preprocessing, "
    "schema enforcement, and leakage-resistant dataset splitting subsystem "
    "for the PhishGuard AI MLOps platform."
)

__status__: Final[str] = "production"

__all_version__: Final[str] = __api_version__


# ==============================================================================
# COMPATIBILITY CONTRACT
# ==============================================================================

SUPPORTED_PYTHON_MIN: Final[Tuple[int, int]] = (3, 10)
SUPPORTED_PYTHON_MAX_EXCLUSIVE: Final[Tuple[int, int]] = (4, 0)

MINIMUM_PIPELINE_VERSION: Final[str] = "13.0.0"

DATA_API_STABILITY: Final[str] = "stable"

PACKAGE_INITIALIZATION_POLICY: Final[str] = "no_side_effects"


# ==============================================================================
# VERSION INFORMATION
# ==============================================================================


class VersionInfo(NamedTuple):
    """
    Structured semantic-version representation.

    Attributes
    ----------
    major:
        Major API version.
    minor:
        Minor API version.
    patch:
        Patch version.
    release_stage:
        Release maturity indicator.
    """

    major: int
    minor: int
    patch: int
    release_stage: str


def _parse_version(version: str) -> VersionInfo:
    """
    Parse a semantic version string safely.

    The package metadata is controlled by the repository, so malformed
    versions indicate a packaging/configuration problem rather than user
    input. We nevertheless fail explicitly instead of silently substituting
    an unrelated version.

    Parameters
    ----------
    version:
        Semantic version in ``MAJOR.MINOR.PATCH`` form.

    Returns
    -------
    VersionInfo
        Parsed version representation.

    Raises
    ------
    ValueError
        If the version is malformed.
    """

    if not isinstance(version, str):
        raise TypeError("Package version must be a string.")

    parts = version.split(".")

    if len(parts) != 3:
        raise ValueError(
            f"Invalid package version '{version}'. "
            "Expected MAJOR.MINOR.PATCH."
        )

    try:
        major, minor, patch = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            f"Invalid package version '{version}'. "
            "Version components must be integers."
        ) from exc

    if min(major, minor, patch) < 0:
        raise ValueError(
            f"Invalid negative package version component: '{version}'."
        )

    return VersionInfo(
        major=major,
        minor=minor,
        patch=patch,
        release_stage="production",
    )


VERSION_INFO: Final[VersionInfo] = _parse_version(__version__)


# ==============================================================================
# PYTHON RUNTIME COMPATIBILITY
# ==============================================================================


def _validate_python_runtime() -> None:
    """
    Validate the currently running Python interpreter.

    This function is intentionally executed only for runtime compatibility
    checking. It does not import project modules or perform external work.
    """

    current = (sys.version_info.major, sys.version_info.minor)

    if current < SUPPORTED_PYTHON_MIN:
        raise RuntimeError(
            "Unsupported Python runtime for PhishGuard AI data subsystem. "
            f"Required >= {SUPPORTED_PYTHON_MIN[0]}."
            f"{SUPPORTED_PYTHON_MIN[1]}, "
            f"detected {sys.version_info.major}.{sys.version_info.minor}."
        )

    if current >= SUPPORTED_PYTHON_MAX_EXCLUSIVE:
        raise RuntimeError(
            "Unsupported Python runtime for the current PhishGuard AI "
            f"data API. Supported range is >= "
            f"{SUPPORTED_PYTHON_MIN[0]}.{SUPPORTED_PYTHON_MIN[1]} and < "
            f"{SUPPORTED_PYTHON_MAX_EXCLUSIVE[0]}."
            f"{SUPPORTED_PYTHON_MAX_EXCLUSIVE[1]}."
        )


_validate_python_runtime()


# ==============================================================================
# OPTIONAL IMPORT MANAGEMENT
# ==============================================================================

@dataclass(frozen=True)
class ImportCapability:
    """
    Describes the availability of one internal data subsystem.

    Attributes
    ----------
    name:
        Public subsystem name.
    module:
        Python module path.
    available:
        Whether the module imported successfully.
    error_type:
        Exception class name if import failed.
    error_message:
        Sanitized diagnostic message if import failed.
    """

    name: str
    module: str
    available: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None


def _safe_import(
    module_name: str,
    capability_name: str,
) -> Tuple[Optional[ModuleType], ImportCapability]:
    """
    Import an internal subsystem while preserving package-level diagnostics.

    Import failures are recorded rather than automatically being converted
    into silent failures.

    This is particularly useful when an implementation has an optional
    dependency such as PyArrow or scikit-learn.

    Parameters
    ----------
    module_name:
        Fully qualified internal module path.
    capability_name:
        Human-readable capability identifier.

    Returns
    -------
    tuple
        Imported module, or ``None``, together with capability information.

    Notes
    -----
    ImportError is deliberately distinguished from arbitrary runtime
    exceptions. A module that raises an unexpected exception during import
    should not be treated as a normal missing optional dependency.
    """

    try:
        module = importlib.import_module(module_name)

        return module, ImportCapability(
            name=capability_name,
            module=module_name,
            available=True,
        )

    except ImportError as exc:
        return None, ImportCapability(
            name=capability_name,
            module=module_name,
            available=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


# ==============================================================================
# DATA LOADER API
# ==============================================================================

_loaders_module, _loaders_capability = _safe_import(
    "src.data.loaders",
    "loaders",
)

if _loaders_module is not None:

    BaseDataLoader = getattr(_loaders_module, "BaseDataLoader", None)
    EmailDataLoader = getattr(_loaders_module, "EmailDataLoader", None)
    URLDataLoader = getattr(_loaders_module, "URLDataLoader", None)
    AttachmentDataLoader = getattr(
        _loaders_module,
        "AttachmentDataLoader",
        None,
    )
    load_raw_dataset = getattr(
        _loaders_module,
        "load_raw_dataset",
        None,
    )

else:

    BaseDataLoader = None
    EmailDataLoader = None
    URLDataLoader = None
    AttachmentDataLoader = None
    load_raw_dataset = None


# ==============================================================================
# DATA VALIDATION API
# ==============================================================================

_validators_module, _validators_capability = _safe_import(
    "src.data.validators",
    "validators",
)

if _validators_module is not None:

    DataQualityValidator = getattr(
        _validators_module,
        "DataQualityValidator",
        None,
    )

    SchemaValidator = getattr(
        _validators_module,
        "SchemaValidator",
        None,
    )

    ValidationResult = getattr(
        _validators_module,
        "ValidationResult",
        None,
    )

    validate_dataset_integrity = getattr(
        _validators_module,
        "validate_dataset_integrity",
        None,
    )

else:

    DataQualityValidator = None
    SchemaValidator = None
    ValidationResult = None
    validate_dataset_integrity = None


# ==============================================================================
# PREPROCESSING API
# ==============================================================================

_preprocessors_module, _preprocessors_capability = _safe_import(
    "src.data.preprocessors",
    "preprocessors",
)

if _preprocessors_module is not None:

    BasePreprocessor = getattr(
        _preprocessors_module,
        "BasePreprocessor",
        None,
    )

    EmailTextPreprocessor = getattr(
        _preprocessors_module,
        "EmailTextPreprocessor",
        None,
    )

    URLFeaturePreprocessor = getattr(
        _preprocessors_module,
        "URLFeaturePreprocessor",
        None,
    )

    AttachmentFeaturePreprocessor = getattr(
        _preprocessors_module,
        "AttachmentFeaturePreprocessor",
        None,
    )

    MultiModalFeatureScaler = getattr(
        _preprocessors_module,
        "MultiModalFeatureScaler",
        None,
    )

else:

    BasePreprocessor = None
    EmailTextPreprocessor = None
    URLFeaturePreprocessor = None
    AttachmentFeaturePreprocessor = None
    MultiModalFeatureScaler = None


# ==============================================================================
# DATA SPLITTING API
# ==============================================================================

_splitters_module, _splitters_capability = _safe_import(
    "src.data.splitters",
    "splitters",
)

if _splitters_module is not None:

    TemporalTrainTestSplitter = getattr(
        _splitters_module,
        "TemporalTrainTestSplitter",
        None,
    )

    AdversarialStratifiedSplitter = getattr(
        _splitters_module,
        "AdversarialStratifiedSplitter",
        None,
    )

    split_dataset = getattr(
        _splitters_module,
        "split_dataset",
        None,
    )

    create_cross_validation_folds = getattr(
        _splitters_module,
        "create_cross_validation_folds",
        None,
    )

else:

    TemporalTrainTestSplitter = None
    AdversarialStratifiedSplitter = None
    split_dataset = None
    create_cross_validation_folds = None


# ==============================================================================
# SCHEMA API
# ==============================================================================

_schemas_module, _schemas_capability = _safe_import(
    "src.data.schemas",
    "schemas",
)

if _schemas_module is not None:

    EmailFeatureSchema = getattr(
        _schemas_module,
        "EmailFeatureSchema",
        None,
    )

    URLFeatureSchema = getattr(
        _schemas_module,
        "URLFeatureSchema",
        None,
    )

    AttachmentFeatureSchema = getattr(
        _schemas_module,
        "AttachmentFeatureSchema",
        None,
    )

    DatasetSchemaManifest = getattr(
        _schemas_module,
        "DatasetSchemaManifest",
        None,
    )

    get_schema_by_model_id = getattr(
        _schemas_module,
        "get_schema_by_model_id",
        None,
    )

else:

    EmailFeatureSchema = None
    URLFeatureSchema = None
    AttachmentFeatureSchema = None
    DatasetSchemaManifest = None
    get_schema_by_model_id = None


# ==============================================================================
# CAPABILITY REGISTRY
# ==============================================================================

_CAPABILITIES: Final[Tuple[ImportCapability, ...]] = (
    _loaders_capability,
    _validators_capability,
    _preprocessors_capability,
    _splitters_capability,
    _schemas_capability,
)


def get_capabilities() -> Mapping[str, ImportCapability]:
    """
    Return the availability state of internal data subsystems.

    Returns
    -------
    Mapping[str, ImportCapability]
        Immutable-style mapping of capability names to diagnostics.

    Example
    -------
    >>> capabilities = get_capabilities()
    >>> capabilities["loaders"].available
    True

    Notes
    -----
    The returned mapping is a new dictionary so callers cannot mutate the
    package's internal capability registry.
    """

    return {
        capability.name: capability
        for capability in _CAPABILITIES
    }


def get_unavailable_capabilities() -> Tuple[ImportCapability, ...]:
    """
    Return all unavailable data capabilities.

    This is useful for startup diagnostics and CI environment validation.
    """

    return tuple(
        capability
        for capability in _CAPABILITIES
        if not capability.available
    )


def all_core_capabilities_available() -> bool:
    """
    Return ``True`` when every declared data subsystem is importable.

    This should be used by production startup checks when the complete
    data stack is required.
    """

    return all(
        capability.available
        for capability in _CAPABILITIES
    )


# ==============================================================================
# PACKAGE HEALTH INFORMATION
# ==============================================================================

def get_package_info() -> Dict[str, Any]:
    """
    Return package-level runtime and compatibility information.

    This method is deliberately side-effect free and suitable for inclusion
    in provenance manifests, diagnostics, health checks, and MLOps reports.

    Returns
    -------
    Dict[str, Any]
        Structured package metadata and capability state.
    """

    capabilities = get_capabilities()

    return {
        "package": {
            "title": __title__,
            "name": __package_name__,
            "version": __version__,
            "api_version": __api_version__,
            "pipeline_version": __pipeline_version__,
            "status": __status__,
            "api_stability": DATA_API_STABILITY,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "architecture": platform.machine(),
        },
        "compatibility": {
            "minimum_pipeline_version": MINIMUM_PIPELINE_VERSION,
            "minimum_python": (
                f"{SUPPORTED_PYTHON_MIN[0]}."
                f"{SUPPORTED_PYTHON_MIN[1]}"
            ),
            "maximum_python_exclusive": (
                f"{SUPPORTED_PYTHON_MAX_EXCLUSIVE[0]}."
                f"{SUPPORTED_PYTHON_MAX_EXCLUSIVE[1]}"
            ),
        },
        "capabilities": {
            name: {
                "available": capability.available,
                "module": capability.module,
                "error_type": capability.error_type,
            }
            for name, capability in capabilities.items()
        },
    }


# ==============================================================================
# PUBLIC API VALIDATION
# ==============================================================================

_EXPECTED_PUBLIC_SYMBOLS: Final[Tuple[str, ...]] = (
    "BaseDataLoader",
    "EmailDataLoader",
    "URLDataLoader",
    "AttachmentDataLoader",
    "load_raw_dataset",
    "SchemaValidator",
    "DataQualityValidator",
    "ValidationResult",
    "validate_dataset_integrity",
    "BasePreprocessor",
    "EmailTextPreprocessor",
    "URLFeaturePreprocessor",
    "AttachmentFeaturePreprocessor",
    "MultiModalFeatureScaler",
    "TemporalTrainTestSplitter",
    "AdversarialStratifiedSplitter",
    "split_dataset",
    "create_cross_validation_folds",
    "EmailFeatureSchema",
    "URLFeatureSchema",
    "AttachmentFeatureSchema",
    "DatasetSchemaManifest",
    "get_schema_by_model_id",
)


def validate_public_api(
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Validate the package's expected public API.

    Parameters
    ----------
    strict:
        If ``True``, raise RuntimeError when one or more public symbols are
        unavailable.

    Returns
    -------
    Dict[str, Any]
        API validation report.

    Raises
    ------
    RuntimeError
        If strict validation is requested and symbols are unavailable.
    """

    available = []
    unavailable = []

    for symbol_name in _EXPECTED_PUBLIC_SYMBOLS:
        value = globals().get(symbol_name)

        if value is None:
            unavailable.append(symbol_name)
        else:
            available.append(symbol_name)

    report = {
        "valid": not unavailable,
        "available_symbols": tuple(available),
        "unavailable_symbols": tuple(unavailable),
        "total_expected": len(_EXPECTED_PUBLIC_SYMBOLS),
        "total_available": len(available),
    }

    if strict and unavailable:
        raise RuntimeError(
            "PhishGuard AI data package public API is incomplete. "
            f"Unavailable symbols: {', '.join(unavailable)}"
        )

    return report


# ==============================================================================
# OPTIONAL DEPENDENCY ERROR HELPER
# ==============================================================================

class DataCapabilityUnavailableError(RuntimeError):
    """
    Raised when a requested data capability is unavailable.

    This exception provides a consistent package-level error contract instead
    of exposing arbitrary import failures to callers.
    """

    def __init__(
        self,
        capability: str,
        message: Optional[str] = None,
    ) -> None:

        self.capability = capability

        default_message = (
            f"Data capability '{capability}' is unavailable. "
            "Check the package installation and optional dependencies."
        )

        super().__init__(message or default_message)


def require_capability(capability: str) -> None:
    """
    Require a named data subsystem to be available.

    Parameters
    ----------
    capability:
        Capability name returned by :func:`get_capabilities`.

    Raises
    ------
    ValueError
        If the capability name is unknown.

    DataCapabilityUnavailableError
        If the capability exists but failed to initialize.
    """

    capabilities = get_capabilities()

    if capability not in capabilities:
        raise ValueError(
            f"Unknown data capability '{capability}'. "
            f"Known capabilities: {', '.join(sorted(capabilities))}"
        )

    state = capabilities[capability]

    if not state.available:
        detail = state.error_message or "unknown import failure"

        raise DataCapabilityUnavailableError(
            capability=capability,
            message=(
                f"Required data capability '{capability}' is unavailable. "
                f"Module: '{state.module}'. "
                f"Import error: {detail}"
            ),
        )


# ==============================================================================
# MODEL / DATA MODALITY REGISTRY
# ==============================================================================

DATA_MODALITIES: Final[Tuple[str, ...]] = (
    "email",
    "url",
    "attachment",
)


MODEL_IDS: Final[Tuple[str, ...]] = (
    "email_classifier",
    "url_classifier",
    "attachment_classifier",
)


def is_supported_modality(modality: str) -> bool:
    """
    Check whether a modality is supported by the data subsystem.
    """

    if not isinstance(modality, str):
        return False

    return modality.strip().lower() in DATA_MODALITIES


def is_supported_model_id(model_id: str) -> bool:
    """
    Check whether a model identifier has a registered data pipeline.
    """

    if not isinstance(model_id, str):
        return False

    return model_id.strip().lower() in MODEL_IDS


def normalize_modality(modality: str) -> str:
    """
    Normalize and validate a modality identifier.

    Raises
    ------
    TypeError
        If the supplied modality is not a string.

    ValueError
        If the modality is not supported.
    """

    if not isinstance(modality, str):
        raise TypeError("Modality must be a string.")

    normalized = modality.strip().lower()

    if normalized not in DATA_MODALITIES:
        raise ValueError(
            f"Unsupported data modality '{modality}'. "
            f"Supported modalities: {', '.join(DATA_MODALITIES)}"
        )

    return normalized


def normalize_model_id(model_id: str) -> str:
    """
    Normalize and validate a registered model identifier.
    """

    if not isinstance(model_id, str):
        raise TypeError("model_id must be a string.")

    normalized = model_id.strip().lower()

    if normalized not in MODEL_IDS:
        raise ValueError(
            f"Unsupported model_id '{model_id}'. "
            f"Supported model IDs: {', '.join(MODEL_IDS)}"
        )

    return normalized


# ==============================================================================
# API DOCUMENTATION HELPERS
# ==============================================================================

PUBLIC_MODULES: Final[Tuple[str, ...]] = (
    "src.data.loaders",
    "src.data.validators",
    "src.data.preprocessors",
    "src.data.splitters",
    "src.data.schemas",
)


def get_public_modules() -> Tuple[str, ...]:
    """
    Return the implementation modules forming the data subsystem boundary.
    """

    return PUBLIC_MODULES


def get_version() -> str:
    """
    Return the data API version.
    """

    return __api_version__


# ==============================================================================
# DEPRECATION SUPPORT
# ==============================================================================

_DEPRECATED_SYMBOLS: Final[Mapping[str, str]] = {}


def get_deprecated_symbols() -> Mapping[str, str]:
    """
    Return the currently registered deprecated API symbols.

    The mapping is intentionally empty for v13.0.0 but provides a stable
    mechanism for future migrations.
    """

    return dict(_DEPRECATED_SYMBOLS)


# ==============================================================================
# PACKAGE-LEVEL EXPORT CONTRACT
# ==============================================================================

__all__ = [

    # --------------------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------------------

    "__title__",
    "__package_name__",
    "__version__",
    "__api_version__",
    "__pipeline_version__",
    "__author__",
    "__license__",
    "__description__",
    "__status__",
    "__all_version__",

    # --------------------------------------------------------------------------
    # Version / compatibility
    # --------------------------------------------------------------------------

    "VersionInfo",
    "VERSION_INFO",
    "SUPPORTED_PYTHON_MIN",
    "SUPPORTED_PYTHON_MAX_EXCLUSIVE",
    "MINIMUM_PIPELINE_VERSION",
    "DATA_API_STABILITY",

    # --------------------------------------------------------------------------
    # Capability management
    # --------------------------------------------------------------------------

    "ImportCapability",
    "DataCapabilityUnavailableError",
    "get_capabilities",
    "get_unavailable_capabilities",
    "all_core_capabilities_available",
    "require_capability",

    # --------------------------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------------------------

    "get_package_info",
    "validate_public_api",
    "get_public_modules",
    "get_version",

    # --------------------------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------------------------

    "BaseDataLoader",
    "EmailDataLoader",
    "URLDataLoader",
    "AttachmentDataLoader",
    "load_raw_dataset",

    # --------------------------------------------------------------------------
    # Validators
    # --------------------------------------------------------------------------

    "SchemaValidator",
    "DataQualityValidator",
    "ValidationResult",
    "validate_dataset_integrity",

    # --------------------------------------------------------------------------
    # Preprocessors
    # --------------------------------------------------------------------------

    "BasePreprocessor",
    "EmailTextPreprocessor",
    "URLFeaturePreprocessor",
    "AttachmentFeaturePreprocessor",
    "MultiModalFeatureScaler",

    # --------------------------------------------------------------------------
    # Splitters
    # --------------------------------------------------------------------------

    "TemporalTrainTestSplitter",
    "AdversarialStratifiedSplitter",
    "split_dataset",
    "create_cross_validation_folds",

    # --------------------------------------------------------------------------
    # Schemas
    # --------------------------------------------------------------------------

    "EmailFeatureSchema",
    "URLFeatureSchema",
    "AttachmentFeatureSchema",
    "DatasetSchemaManifest",
    "get_schema_by_model_id",

    # --------------------------------------------------------------------------
    # Modality / model registry
    # --------------------------------------------------------------------------

    "DATA_MODALITIES",
    "MODEL_IDS",
    "is_supported_modality",
    "is_supported_model_id",
    "normalize_modality",
    "normalize_model_id",

    # --------------------------------------------------------------------------
    # Deprecation
    # --------------------------------------------------------------------------

    "get_deprecated_symbols",
]


# ==============================================================================
# INTERNAL CLEANUP
# ==============================================================================
#
# Keep private implementation module references available for diagnostics.
# They are intentionally NOT included in __all__.
#
# No logging handlers, filesystem operations, network calls, dataset loading,
# random seed changes, or model initialization occur in this module.
#
# This is deliberate: importing ``src.data`` must remain cheap, predictable,
# and side-effect free.
# ==============================================================================

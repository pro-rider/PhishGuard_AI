"""
PhishGuard_AI - Machine Learning Inference Engine
Version: 10.1.0

Production ML inference package for PhishGuard_AI.

This package provides the public API for:

    - asynchronous threat prediction
    - secure ONNX model inference
    - model artifact identity and isolation
    - feature extraction
    - prediction/feature/concept drift monitoring
    - model registry access
    - prediction provenance and status reporting

Architecture
------------

The package intentionally exposes a controlled public API through this
module while keeping implementation details inside their respective
submodules.

Public modules
--------------

predictor
    Asynchronous ML prediction and heuristic fallback engine.

drift_monitor
    Prediction, feature, and concept drift monitoring.

feature_extractors
    Email, URL, and attachment feature extraction.

model_registry
    Secure model artifact registration and runtime management.

Design principles
-----------------

1. Keep package initialization lightweight.
2. Avoid global runtime/service initialization.
3. Avoid broad ImportError suppression.
4. Preserve a stable public API.
5. Prevent accidental exposure of private implementation objects.
6. Keep package metadata deterministic.
7. Use package-relative imports.
8. Fail immediately when a required internal module is broken.
9. Keep compatibility with downstream PhishGuard_AI modules.
10. Avoid logging sensitive model or inference information during import.

This package does not perform inference merely by being imported.
"""

from __future__ import annotations

from importlib import metadata
from typing import Final, TypedDict


# ============================================================================
# Package Metadata
# ============================================================================

__title__: Final[str] = "PhishGuard_AI ML Inference"
__version__: Final[str] = "10.1.0"
__author__: Final[str] = "PhishGuard_AI Engineering Team"
__license__: Final[str] = "Proprietary"
__description__: Final[str] = (
    "Production ML inference, feature extraction, model registry, "
    "and drift monitoring engine for PhishGuard_AI."
)

# Python package/distribution name.
#
# This may differ from the import package name when the project is installed
# from a wheel or source distribution.
__distribution_name__: Final[str] = "PhishGuard_AI"


# ============================================================================
# Feature / Capability Metadata
# ============================================================================

# These values describe capabilities exposed by this package version.
# They are intentionally static and do not initialize runtime components.

CAPABILITY_PREDICTION: Final[str] = "prediction"
CAPABILITY_ONNX: Final[str] = "onnx_inference"
CAPABILITY_DRIFT_MONITORING: Final[str] = "drift_monitoring"
CAPABILITY_FEATURE_EXTRACTION: Final[str] = "feature_extraction"
CAPABILITY_MODEL_REGISTRY: Final[str] = "model_registry"
CAPABILITY_HEURISTIC_FALLBACK: Final[str] = "heuristic_fallback"

SUPPORTED_CAPABILITIES: Final[tuple[str, ...]] = (
    CAPABILITY_PREDICTION,
    CAPABILITY_ONNX,
    CAPABILITY_DRIFT_MONITORING,
    CAPABILITY_FEATURE_EXTRACTION,
    CAPABILITY_MODEL_REGISTRY,
    CAPABILITY_HEURISTIC_FALLBACK,
)


# ============================================================================
# Public Package Information
# ============================================================================


class PackageInfo(TypedDict):
    """
    Serializable package metadata structure.

    This type is intentionally small so it can safely be returned by
    health/version endpoints without exposing implementation details.
    """

    title: str
    version: str
    author: str
    license: str
    description: str
    capabilities: tuple[str, ...]


def get_package_info() -> PackageInfo:
    """
    Return static package metadata.

    This function does not instantiate a model registry, predictor,
    feature extractor, ONNX session, network client, or drift monitor.

    Returns
    -------
    PackageInfo
        Immutable-style metadata dictionary suitable for health,
        diagnostics, and observability endpoints.
    """
    return {
        "title": __title__,
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": __description__,
        "capabilities": SUPPORTED_CAPABILITIES,
    }


# ============================================================================
# Optional Installed Distribution Version
# ============================================================================


def get_installed_distribution_version() -> str | None:
    """
    Return the installed distribution version when available.

    This is intentionally separate from ``__version__``.

    ``__version__`` describes the source package API version, while this
    function attempts to determine the version recorded by the Python
    packaging system.

    Returns
    -------
    str | None
        Installed distribution version, or ``None`` when the package is
        running directly from source or metadata is unavailable.
    """
    try:
        return metadata.version(__distribution_name__)
    except metadata.PackageNotFoundError:
        return None


# ============================================================================
# Predictor API
# ============================================================================

from .predictor import (
    DEFAULT_CRITICAL_THRESHOLD,
    DEFAULT_FALLBACK_MAXIMUM,
    DEFAULT_FALLBACK_MINIMUM,
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_MEDIUM_THRESHOLD,
    MLPredictor,
    PredictionResult,
    PredictionSource,
    PredictionStatus,
)


# ============================================================================
# Drift Monitoring API
# ============================================================================

from .drift_monitor import (
    DEFAULT_CRITICAL_PSI,
    DEFAULT_EPSILON,
    DEFAULT_MIN_BASELINE_SAMPLES,
    DEFAULT_MIN_TARGET_SAMPLES,
    DEFAULT_PSI_BUCKETS,
    DEFAULT_WARNING_PSI,
    DEFAULT_WINDOW_SIZE,
    ConceptDriftResult,
    DriftMonitor,
    DriftMonitorSnapshot,
    DriftReport,
    DriftSeverity,
    DriftType,
    FeatureDriftResult,
    ModelIdentity,
    PSIResult,
    PredictionMetricPoint,
)


# ============================================================================
# Feature Extraction API
# ============================================================================

from .feature_extractors import (
    AttachmentFeatureExtractor,
    EmailFeatureExtractor,
    UrlFeatureExtractor,
)


# ============================================================================
# Model Registry API
# ============================================================================

from .model_registry import ModelRegistry


# ============================================================================
# Public API
# ============================================================================

__all__: Final[tuple[str, ...]] = (
    # ------------------------------------------------------------------------
    # Package metadata
    # ------------------------------------------------------------------------
    "__title__",
    "__version__",
    "__author__",
    "__license__",
    "__description__",
    "__distribution_name__",
    "get_package_info",
    "get_installed_distribution_version",
    "SUPPORTED_CAPABILITIES",
    "CAPABILITY_PREDICTION",
    "CAPABILITY_ONNX",
    "CAPABILITY_DRIFT_MONITORING",
    "CAPABILITY_FEATURE_EXTRACTION",
    "CAPABILITY_MODEL_REGISTRY",
    "CAPABILITY_HEURISTIC_FALLBACK",

    # ------------------------------------------------------------------------
    # Predictor
    # ------------------------------------------------------------------------
    "MLPredictor",
    "PredictionResult",
    "PredictionSource",
    "PredictionStatus",
    "DEFAULT_CRITICAL_THRESHOLD",
    "DEFAULT_HIGH_THRESHOLD",
    "DEFAULT_MEDIUM_THRESHOLD",
    "DEFAULT_FALLBACK_MINIMUM",
    "DEFAULT_FALLBACK_MAXIMUM",

    # ------------------------------------------------------------------------
    # Drift monitoring
    # ------------------------------------------------------------------------
    "DriftMonitor",
    "ModelIdentity",
    "DriftSeverity",
    "DriftType",
    "PSIResult",
    "FeatureDriftResult",
    "ConceptDriftResult",
    "DriftReport",
    "DriftMonitorSnapshot",
    "PredictionMetricPoint",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_PSI_BUCKETS",
    "DEFAULT_WARNING_PSI",
    "DEFAULT_CRITICAL_PSI",
    "DEFAULT_MIN_BASELINE_SAMPLES",
    "DEFAULT_MIN_TARGET_SAMPLES",
    "DEFAULT_EPSILON",

    # ------------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------------
    "EmailFeatureExtractor",
    "UrlFeatureExtractor",
    "AttachmentFeatureExtractor",

    # ------------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------------
    "ModelRegistry",
)


# ============================================================================
# Internal Integrity Checks
# ============================================================================


def _validate_public_api() -> None:
    """
    Validate that every symbol declared in ``__all__`` is actually exported.

    This is intentionally executed during import because a broken package
    public API should fail early during development/deployment rather than
    producing a confusing AttributeError later.

    The function itself remains private and is not part of the public API.
    """
    namespace = globals()

    missing = [
        name
        for name in __all__
        if name not in namespace
    ]

    if missing:
        raise RuntimeError(
            "PhishGuard_AI ml_inference package API is incomplete; "
            f"missing exports: {', '.join(sorted(missing))}"
        )


_validate_public_api()


# ============================================================================
# Explicit Package-Level Documentation
# ============================================================================

__docformat__ = "reStructuredText"
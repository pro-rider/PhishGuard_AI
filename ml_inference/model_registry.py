"""
PhishGuard_AI - Secure Model Registry & Artifact Manager
Version: 10.0.0

Production-grade registry for trusted local ONNX/ORT model artifacts.

Security goals
--------------
* Strict model manifest validation.
* SHA-256 artifact integrity verification.
* Registry-directory confinement.
* Canonical path / traversal protection.
* Defensive handling of symlinks.
* Model lifecycle management.
* Thread-safe per-model lazy loading.
* Duplicate-load prevention.
* Artifact hot-swap detection.
* ONNX Runtime provider enforcement.
* ONNX input/output contract validation.
* Feature-schema integrity validation.
* Model and schema resource limits.
* Explicit model enable/disable/revoke controls.
* Explicit session unload/reload support.
* Operational statistics and health information.
* No dynamic execution of manifest content.

Important
---------
This registry treats model artifacts as untrusted until their integrity,
path, schema, and runtime contract have been validated.

For highest-assurance deployments, artifacts should be distributed through
a trusted build/signing pipeline and the SHA-256 digest should be generated
outside the application.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:  # pragma: no cover
    ort = None  # type: ignore
    ONNX_AVAILABLE = False


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

MODEL_REGISTRY_VERSION = "10.0.0"

DEFAULT_MODELS_DIR = "config/external/model_registry"

DEFAULT_THRESHOLD_SUSPICIOUS = 0.40
DEFAULT_THRESHOLD_PHISHING = 0.75

DEFAULT_MAX_MODEL_SIZE_BYTES = 250 * 1024 * 1024
DEFAULT_MAX_MODELS = 100

DEFAULT_MAX_FEATURES = 10_000
DEFAULT_MAX_FEATURE_NAME_LENGTH = 256
DEFAULT_MAX_TOTAL_SCHEMA_BYTES = 2 * 1024 * 1024

DEFAULT_INTRA_OP_THREADS = 1
DEFAULT_INTER_OP_THREADS = 1

DEFAULT_ALLOWED_PROVIDERS = ("CPUExecutionProvider",)

SHA256_HEX_LENGTH = 64

MODEL_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
)

VERSION_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._+\-]{0,63}$"
)

FEATURE_NAME_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.:\-/]{0,255}$"
)

SUPPORTED_MODEL_TYPES = frozenset(
    {
        "email",
        "url",
        "attachment",
        "web",
        "generic",
    }
)

SUPPORTED_ARTIFACT_SUFFIXES = frozenset(
    {
        ".onnx",
        ".ort",
    }
)

SUPPORTED_INPUT_TYPES = frozenset(
    {
        "tensor(float)",
        "tensor(double)",
        "tensor(float16)",
    }
)

# =============================================================================
# ENUMS
# =============================================================================


class ModelStatus(str, Enum):
    """Lifecycle state of a registered model."""

    REGISTERED = "registered"
    READY = "ready"
    FAILED = "failed"
    DISABLED = "disabled"
    REVOKED = "revoked"


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ModelRegistryError(RuntimeError):
    """Base exception for model registry failures."""


class ManifestValidationError(ModelRegistryError):
    """Raised when a model manifest is invalid."""


class ArtifactValidationError(ModelRegistryError):
    """Raised when a model artifact fails integrity/path validation."""


class ModelLoadError(ModelRegistryError):
    """Raised when ONNX Runtime cannot load or validate a model."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _constant_time_equal(a: str, b: str) -> bool:
    """
    Constant-time comparison for hexadecimal digests.

    This is primarily useful for avoiding accidental timing differences
    during integrity comparisons.
    """
    return hmac.compare_digest(
        str(a).encode("ascii", errors="strict"),
        str(b).encode("ascii", errors="strict"),
    )


def _is_finite_number(value: Any) -> bool:
    """Return True only for finite numeric values."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass(slots=True, frozen=True)
class ModelManifest:
    """
    Immutable description of a deployable ML model artifact.

    feature_names order is part of the model contract and MUST NOT change
    without producing a new model version.
    """

    model_id: str
    model_type: str
    version: str
    artifact_path: str

    feature_names: Tuple[str, ...]

    threshold_phishing: float = DEFAULT_THRESHOLD_PHISHING
    threshold_suspicious: float = DEFAULT_THRESHOLD_SUSPICIOUS

    artifact_sha256: Optional[str] = None
    feature_schema_hash: Optional[str] = None

    expected_input_name: Optional[str] = None
    expected_input_type: Optional[str] = "tensor(float)"

    expected_input_rank: Optional[int] = 2
    expected_feature_count: Optional[int] = None

    expected_output_name: Optional[str] = None
    expected_output_type: Optional[str] = None

    enabled: bool = True
    status: ModelStatus = ModelStatus.REGISTERED

    schema_version: str = "1"

    # ML governance metadata.
    algorithm: Optional[str] = None
    framework: Optional[str] = None
    training_dataset_version: Optional[str] = None
    calibration_method: Optional[str] = None

    # Optional provenance information.
    source: Optional[str] = None
    publisher: Optional[str] = None

    metadata: Mapping[str, Any] = field(default_factory=dict)
    tags: Tuple[str, ...] = field(default_factory=tuple)

    created_at: Optional[str] = None


@dataclass(slots=True, frozen=True)
class ModelRuntimeInfo:
    """Runtime contract information extracted from ONNX Runtime."""

    input_names: Tuple[str, ...]
    input_types: Tuple[str, ...]
    input_shapes: Tuple[str, ...]

    output_names: Tuple[str, ...]
    output_types: Tuple[str, ...]
    output_shapes: Tuple[str, ...]

    providers: Tuple[str, ...]

    runtime_version: Optional[str] = None


@dataclass(slots=True)
class RegistryStats:
    """Operational registry counters."""

    manifest_count: int = 0
    ready_count: int = 0
    registered_count: int = 0
    failed_count: int = 0
    disabled_count: int = 0
    revoked_count: int = 0

    session_count: int = 0

    cache_hits: int = 0
    cache_misses: int = 0

    successful_loads: int = 0
    failed_loads: int = 0

    integrity_failures: int = 0
    validation_failures: int = 0
    contract_failures: int = 0

    unload_count: int = 0
    reload_count: int = 0

    last_load_time_ms: float = 0.0


@dataclass(slots=True)
class _LoadedSession:
    """Internal session-cache entry."""

    session: Any

    artifact_sha256: str
    artifact_size: int

    loaded_at_monotonic: float

    runtime_info: ModelRuntimeInfo


# =============================================================================
# CRYPTOGRAPHIC INTEGRITY
# =============================================================================


class ModelIntegrity:
    """Cryptographic integrity helpers."""

    @staticmethod
    def sha256_file(
        path: Path,
        *,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        """
        Calculate SHA-256 without loading the entire artifact into memory.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        digest = hashlib.sha256()

        try:
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(chunk_size)

                    if not chunk:
                        break

                    digest.update(chunk)

        except OSError as exc:
            raise ArtifactValidationError(
                f"Unable to read model artifact: {path}"
            ) from exc

        return digest.hexdigest()

    @staticmethod
    def feature_schema_hash(
        feature_names: Sequence[str],
    ) -> str:
        """
        Produce deterministic SHA-256 hash for the ordered feature schema.
        """
        canonical = json.dumps(
            list(feature_names),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def normalize_sha256(
        value: Optional[str],
    ) -> Optional[str]:
        """Validate and normalize a SHA-256 digest."""
        if value is None:
            return None

        normalized = str(value).strip().lower()

        if not re.fullmatch(
            rf"[0-9a-f]{{{SHA256_HEX_LENGTH}}}",
            normalized,
        ):
            raise ArtifactValidationError(
                "SHA-256 value must be exactly 64 hexadecimal characters."
            )

        return normalized


# =============================================================================
# MANIFEST VALIDATOR
# =============================================================================


class ModelManifestValidator:
    """Strict validation for model manifests."""

    @classmethod
    def validate(
        cls,
        manifest: ModelManifest,
        *,
        require_artifact_hash: bool = True,
    ) -> None:

        if not isinstance(manifest, ModelManifest):
            raise ManifestValidationError(
                "manifest must be a ModelManifest instance."
            )

        # ---------------------------------------------------------------------
        # Model ID
        # ---------------------------------------------------------------------

        if not MODEL_ID_PATTERN.fullmatch(manifest.model_id):
            raise ManifestValidationError(
                f"Invalid model_id: {manifest.model_id!r}"
            )

        # ---------------------------------------------------------------------
        # Version
        # ---------------------------------------------------------------------

        if not VERSION_PATTERN.fullmatch(manifest.version):
            raise ManifestValidationError(
                f"Invalid model version: {manifest.version!r}"
            )

        # ---------------------------------------------------------------------
        # Model type
        # ---------------------------------------------------------------------

        model_type = manifest.model_type.strip().lower()

        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ManifestValidationError(
                f"Unsupported model_type: {manifest.model_type!r}"
            )

        # ---------------------------------------------------------------------
        # Artifact
        # ---------------------------------------------------------------------

        if not manifest.artifact_path:
            raise ManifestValidationError(
                "artifact_path cannot be empty."
            )

        artifact_suffix = Path(
            manifest.artifact_path
        ).suffix.lower()

        if artifact_suffix not in SUPPORTED_ARTIFACT_SUFFIXES:
            raise ManifestValidationError(
                f"Unsupported artifact suffix: {artifact_suffix!r}"
            )

        # ---------------------------------------------------------------------
        # Features
        # ---------------------------------------------------------------------

        if not manifest.feature_names:
            raise ManifestValidationError(
                "feature_names cannot be empty."
            )

        if len(manifest.feature_names) > DEFAULT_MAX_FEATURES:
            raise ManifestValidationError(
                f"Feature count exceeds {DEFAULT_MAX_FEATURES}."
            )

        total_schema_bytes = 0
        normalized_features: List[str] = []

        for feature in manifest.feature_names:

            if not isinstance(feature, str):
                raise ManifestValidationError(
                    "Every feature name must be a string."
                )

            feature_name = feature.strip()

            if not feature_name:
                raise ManifestValidationError(
                    "Feature names cannot be empty."
                )

            if len(feature_name) > DEFAULT_MAX_FEATURE_NAME_LENGTH:
                raise ManifestValidationError(
                    "Feature name exceeds maximum length."
                )

            if not FEATURE_NAME_PATTERN.fullmatch(feature_name):
                raise ManifestValidationError(
                    f"Invalid feature name: {feature_name!r}"
                )

            normalized_features.append(feature_name)
            total_schema_bytes += len(feature_name.encode("utf-8"))

        if total_schema_bytes > DEFAULT_MAX_TOTAL_SCHEMA_BYTES:
            raise ManifestValidationError(
                "Feature schema exceeds maximum allowed size."
            )

        if len(set(normalized_features)) != len(normalized_features):
            raise ManifestValidationError(
                "feature_names contains duplicate entries."
            )

        # ---------------------------------------------------------------------
        # Thresholds
        # ---------------------------------------------------------------------

        if not _is_finite_number(manifest.threshold_suspicious):
            raise ManifestValidationError(
                "threshold_suspicious must be a finite number."
            )

        if not _is_finite_number(manifest.threshold_phishing):
            raise ManifestValidationError(
                "threshold_phishing must be a finite number."
            )

        suspicious = float(manifest.threshold_suspicious)
        phishing = float(manifest.threshold_phishing)

        if not 0.0 <= suspicious <= 1.0:
            raise ManifestValidationError(
                "threshold_suspicious must be between 0 and 1."
            )

        if not 0.0 <= phishing <= 1.0:
            raise ManifestValidationError(
                "threshold_phishing must be between 0 and 1."
            )

        if suspicious >= phishing:
            raise ManifestValidationError(
                "threshold_suspicious must be lower than threshold_phishing."
            )

        # ---------------------------------------------------------------------
        # Artifact hash
        # ---------------------------------------------------------------------

        normalized_hash = ModelIntegrity.normalize_sha256(
            manifest.artifact_sha256
        )

        if require_artifact_hash and normalized_hash is None:
            raise ManifestValidationError(
                "artifact_sha256 is required for production registration."
            )

        # ---------------------------------------------------------------------
        # Feature schema hash
        # ---------------------------------------------------------------------

        if manifest.feature_schema_hash is not None:

            feature_hash = str(
                manifest.feature_schema_hash
            ).strip().lower()

            if not re.fullmatch(
                rf"[0-9a-f]{{{SHA256_HEX_LENGTH}}}",
                feature_hash,
            ):
                raise ManifestValidationError(
                    "feature_schema_hash must be a valid SHA-256 digest."
                )

        # ---------------------------------------------------------------------
        # ONNX input contract
        # ---------------------------------------------------------------------

        if manifest.expected_input_name is not None:

            if not manifest.expected_input_name.strip():
                raise ManifestValidationError(
                    "expected_input_name cannot be empty."
                )

        if manifest.expected_input_type is not None:

            if manifest.expected_input_type not in SUPPORTED_INPUT_TYPES:
                raise ManifestValidationError(
                    f"Unsupported expected_input_type: "
                    f"{manifest.expected_input_type!r}"
                )

        if manifest.expected_input_rank is not None:

            if not isinstance(manifest.expected_input_rank, int):
                raise ManifestValidationError(
                    "expected_input_rank must be an integer."
                )

            if not 1 <= manifest.expected_input_rank <= 8:
                raise ManifestValidationError(
                    "expected_input_rank must be between 1 and 8."
                )

        if manifest.expected_feature_count is not None:

            if not isinstance(manifest.expected_feature_count, int):
                raise ManifestValidationError(
                    "expected_feature_count must be an integer."
                )

            if manifest.expected_feature_count <= 0:
                raise ManifestValidationError(
                    "expected_feature_count must be positive."
                )

            if (
                manifest.expected_feature_count
                != len(manifest.feature_names)
            ):
                raise ManifestValidationError(
                    "expected_feature_count must equal "
                    "len(feature_names)."
                )

        # ---------------------------------------------------------------------
        # Output contract
        # ---------------------------------------------------------------------

        if manifest.expected_output_name is not None:

            if not manifest.expected_output_name.strip():
                raise ManifestValidationError(
                    "expected_output_name cannot be empty."
                )

        if manifest.expected_output_type is not None:

            if not manifest.expected_output_type.strip():
                raise ManifestValidationError(
                    "expected_output_type cannot be empty."
                )

        # ---------------------------------------------------------------------
        # Schema version
        # ---------------------------------------------------------------------

        if not VERSION_PATTERN.fullmatch(
            str(manifest.schema_version)
        ):
            raise ManifestValidationError(
                "Invalid schema_version."
            )


# =============================================================================
# PATH SECURITY
# =============================================================================


class ModelPathResolver:
    """
    Restricts model artifacts to the configured registry directory.

    Both traversal and canonical symlink escapes are rejected.
    """

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir.resolve()

    def resolve(self, artifact_path: str) -> Path:

        if not isinstance(artifact_path, str):
            raise ArtifactValidationError(
                "artifact_path must be a string."
            )

        if not artifact_path.strip():
            raise ArtifactValidationError(
                "artifact_path cannot be empty."
            )

        candidate = Path(artifact_path)

        if candidate.is_absolute():
            resolved = candidate.resolve(strict=False)
        else:
            resolved = (
                self.models_dir / candidate
            ).resolve(strict=False)

        try:
            resolved.relative_to(self.models_dir)
        except ValueError as exc:
            raise ArtifactValidationError(
                "Model artifact path escapes registry directory."
            ) from exc

        return resolved

    def validate_existing_file(self, path: Path) -> None:
        """
        Validate the resolved artifact as a normal regular file.

        Symlinks are rejected explicitly because the model registry should
        not depend on mutable filesystem links.
        """
        try:
            if path.is_symlink():
                raise ArtifactValidationError(
                    "Symlink model artifacts are not permitted."
                )

            if not path.exists():
                raise ArtifactValidationError(
                    f"Model artifact does not exist: {path}"
                )

            if not path.is_file():
                raise ArtifactValidationError(
                    f"Model artifact is not a regular file: {path}"
                )

        except OSError as exc:
            raise ArtifactValidationError(
                "Unable to inspect model artifact."
            ) from exc


# =============================================================================
# ONNX RUNTIME INSPECTION
# =============================================================================


class ONNXModelInspector:
    """Extracts and validates ONNX Runtime model metadata."""

    @staticmethod
    def inspect(session: Any) -> ModelRuntimeInfo:

        try:
            inputs = session.get_inputs()
            outputs = session.get_outputs()

        except Exception as exc:
            raise ModelLoadError(
                "Unable to inspect ONNX Runtime model contract."
            ) from exc

        input_names = tuple(
            str(item.name)
            for item in inputs
        )

        input_types = tuple(
            str(item.type)
            for item in inputs
        )

        input_shapes = tuple(
            ONNXModelInspector._safe_shape(item.shape)
            for item in inputs
        )

        output_names = tuple(
            str(item.name)
            for item in outputs
        )

        output_types = tuple(
            str(item.type)
            for item in outputs
        )

        output_shapes = tuple(
            ONNXModelInspector._safe_shape(item.shape)
            for item in outputs
        )

        try:
            providers = tuple(
                str(provider)
                for provider in session.get_providers()
            )
        except Exception:
            providers = tuple()

        runtime_version = (
            getattr(ort, "__version__", None)
            if ONNX_AVAILABLE
            else None
        )

        return ModelRuntimeInfo(
            input_names=input_names,
            input_types=input_types,
            input_shapes=input_shapes,
            output_names=output_names,
            output_types=output_types,
            output_shapes=output_shapes,
            providers=providers,
            runtime_version=runtime_version,
        )

    @staticmethod
    def _safe_shape(shape: Any) -> str:

        try:
            return str(tuple(shape))
        except Exception:
            return "<unknown>"

    @staticmethod
    def parse_shape(shape_string: str) -> Tuple[Any, ...]:
        """
        Best-effort parsing of a stringified ONNX shape.

        Dynamic dimensions are represented as strings.
        """
        if not shape_string:
            return tuple()

        cleaned = (
            shape_string
            .strip()
            .lstrip("(")
            .rstrip(")")
        )

        if not cleaned:
            return tuple()

        result: List[Any] = []

        for item in cleaned.split(","):

            value = item.strip()

            if not value:
                continue

            try:
                result.append(int(value))
            except ValueError:
                result.append(value)

        return tuple(result)


# =============================================================================
# MODEL REGISTRY
# =============================================================================


class ModelRegistry:
    """
    Secure, thread-safe production model registry.

    Expensive operations such as hashing and ONNX session creation happen
    outside the global registry lock and are protected by per-model locks.
    """

    def __init__(
        self,
        models_dir: str = DEFAULT_MODELS_DIR,
        *,
        require_artifact_hash: bool = True,
        max_model_size_bytes: int = DEFAULT_MAX_MODEL_SIZE_BYTES,
        max_models: int = DEFAULT_MAX_MODELS,
        allowed_providers: Sequence[str] = DEFAULT_ALLOWED_PROVIDERS,
        intra_op_num_threads: int = DEFAULT_INTRA_OP_THREADS,
        inter_op_num_threads: int = DEFAULT_INTER_OP_THREADS,
        enable_graph_optimization: bool = True,
        reject_symlinks: bool = True,
    ) -> None:

        if max_model_size_bytes <= 0:
            raise ValueError(
                "max_model_size_bytes must be greater than zero."
            )

        if max_models <= 0:
            raise ValueError(
                "max_models must be greater than zero."
            )

        if intra_op_num_threads <= 0:
            raise ValueError(
                "intra_op_num_threads must be greater than zero."
            )

        if inter_op_num_threads <= 0:
            raise ValueError(
                "inter_op_num_threads must be greater than zero."
            )

        provider_tuple = tuple(
            str(provider).strip()
            for provider in allowed_providers
            if str(provider).strip()
        )

        if not provider_tuple:
            raise ValueError(
                "At least one execution provider must be configured."
            )

        self.models_dir = Path(
            models_dir
        ).resolve()

        self.models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._path_resolver = ModelPathResolver(
            self.models_dir
        )

        self.require_artifact_hash = require_artifact_hash
        self.max_model_size_bytes = max_model_size_bytes
        self.max_models = max_models

        self.allowed_providers = provider_tuple

        self.intra_op_num_threads = intra_op_num_threads
        self.inter_op_num_threads = inter_op_num_threads

        self.enable_graph_optimization = (
            enable_graph_optimization
        )

        self.reject_symlinks = reject_symlinks

        self._manifests: Dict[
            str,
            ModelManifest,
        ] = {}

        self._sessions: Dict[
            str,
            _LoadedSession,
        ] = {}

        self._status: Dict[
            str,
            ModelStatus,
        ] = {}

        self._generation: Dict[
            str,
            int,
        ] = {}

        self._model_locks: Dict[
            str,
            threading.RLock,
        ] = {}

        self._lock = threading.RLock()

        self._stats = RegistryStats()

    # =========================================================================
    # INTERNAL MODEL LOCK
    # =========================================================================

    def _get_model_lock(
        self,
        model_id: str,
    ) -> threading.RLock:

        with self._lock:

            lock = self._model_locks.get(model_id)

            if lock is None:

                lock = threading.RLock()

                self._model_locks[model_id] = lock

            return lock

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register_manifest(
        self,
        manifest: ModelManifest,
        *,
        verify_artifact: bool = True,
        replace: bool = False,
    ) -> None:
        """
        Atomically register a model manifest.

        Production mode should keep verify_artifact=True.
        """

        try:
            ModelManifestValidator.validate(
                manifest,
                require_artifact_hash=self.require_artifact_hash,
            )
        except ManifestValidationError:

            with self._lock:
                self._stats.validation_failures += 1

            raise

        with self._lock:

            exists = (
                manifest.model_id
                in self._manifests
            )

            if exists and not replace:
                raise ManifestValidationError(
                    f"Model '{manifest.model_id}' "
                    "is already registered."
                )

            if (
                not exists
                and len(self._manifests)
                >= self.max_models
            ):
                raise ManifestValidationError(
                    "Maximum registered model count exceeded."
                )

        artifact = self._path_resolver.resolve(
            manifest.artifact_path
        )

        if verify_artifact:
            self._verify_artifact(
                manifest,
                artifact,
            )

        with self._lock:

            if replace:

                self._sessions.pop(
                    manifest.model_id,
                    None,
                )

                self._generation[
                    manifest.model_id
                ] = (
                    self._generation.get(
                        manifest.model_id,
                        0,
                    )
                    + 1
                )

            self._manifests[
                manifest.model_id
            ] = manifest

            status = (
                ModelStatus.READY
                if manifest.enabled
                else ModelStatus.DISABLED
            )

            self._status[
                manifest.model_id
            ] = status

            self._model_locks.setdefault(
                manifest.model_id,
                threading.RLock(),
            )

            self._refresh_stats_locked()

        logger.info(
            "Registered model id=%s version=%s type=%s status=%s",
            manifest.model_id,
            manifest.version,
            manifest.model_type,
            status.value,
        )

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    def get_manifest(
        self,
        model_id: str,
    ) -> Optional[ModelManifest]:

        with self._lock:
            return self._manifests.get(model_id)

    def get_status(
        self,
        model_id: str,
    ) -> Optional[ModelStatus]:

        with self._lock:
            return self._status.get(model_id)

    def get_runtime_info(
        self,
        model_id: str,
    ) -> Optional[ModelRuntimeInfo]:

        with self._lock:

            loaded = self._sessions.get(
                model_id
            )

            if loaded is None:
                return None

            return loaded.runtime_info

    def list_manifests(
        self,
    ) -> List[ModelManifest]:

        with self._lock:
            return list(
                self._manifests.values()
            )

    # =========================================================================
    # ENABLE / DISABLE / REVOKE
    # =========================================================================

    def enable(
        self,
        model_id: str,
    ) -> None:

        with self._lock:

            manifest = self._require_manifest_locked(
                model_id
            )

            if (
                self._status.get(model_id)
                == ModelStatus.REVOKED
            ):
                raise ModelRegistryError(
                    f"Model '{model_id}' is revoked."
                )

            self._status[
                model_id
            ] = ModelStatus.READY

            # Respect manifest enabled flag through explicit lifecycle.
            if not manifest.enabled:
                logger.warning(
                    "Enabling model '%s' despite manifest.enabled=False.",
                    model_id,
                )

            self._refresh_stats_locked()

        logger.info(
            "Enabled model '%s'.",
            model_id,
        )

    def disable(
        self,
        model_id: str,
    ) -> None:

        lock = self._get_model_lock(model_id)

        with lock:

            with self._lock:

                self._require_manifest_locked(
                    model_id
                )

                self._status[
                    model_id
                ] = ModelStatus.DISABLED

                self._sessions.pop(
                    model_id,
                    None,
                )

                self._refresh_stats_locked()

        logger.warning(
            "Disabled model '%s'.",
            model_id,
        )

    def revoke(
        self,
        model_id: str,
    ) -> None:

        lock = self._get_model_lock(model_id)

        with lock:

            with self._lock:

                self._require_manifest_locked(
                    model_id
                )

                self._status[
                    model_id
                ] = ModelStatus.REVOKED

                self._sessions.pop(
                    model_id,
                    None,
                )

                self._generation[
                    model_id
                ] = (
                    self._generation.get(
                        model_id,
                        0,
                    )
                    + 1
                )

                self._refresh_stats_locked()

        logger.error(
            "Revoked model '%s'.",
            model_id,
        )

    # =========================================================================
    # UNREGISTER
    # =========================================================================

    def unregister(
        self,
        model_id: str,
    ) -> None:

        lock = self._get_model_lock(model_id)

        with lock:

            with self._lock:

                if model_id not in self._manifests:
                    raise ModelRegistryError(
                        f"Model '{model_id}' is not registered."
                    )

                self._sessions.pop(
                    model_id,
                    None,
                )

                self._manifests.pop(
                    model_id,
                    None,
                )

                self._status.pop(
                    model_id,
                    None,
                )

                self._generation.pop(
                    model_id,
                    None,
                )

                self._refresh_stats_locked()

        logger.info(
            "Unregistered model '%s'.",
            model_id,
        )

    # =========================================================================
    # ARTIFACT VALIDATION
    # =========================================================================

    def validate_artifact(
        self,
        model_id: str,
    ) -> bool:

        manifest = self.get_manifest(
            model_id
        )

        if manifest is None:
            return False

        try:

            artifact = self._path_resolver.resolve(
                manifest.artifact_path
            )

            self._verify_artifact(
                manifest,
                artifact,
            )

            return True

        except Exception:

            logger.exception(
                "Artifact validation failed for model '%s'.",
                model_id,
            )

            return False

    def _verify_artifact(
        self,
        manifest: ModelManifest,
        artifact: Path,
    ) -> str:

        self._path_resolver.validate_existing_file(
            artifact
        )

        if (
            self.reject_symlinks
            and artifact.is_symlink()
        ):
            raise ArtifactValidationError(
                "Symlink artifacts are forbidden."
            )

        try:

            stat = artifact.stat()

            size = int(stat.st_size)

        except OSError as exc:

            raise ArtifactValidationError(
                "Unable to stat model artifact."
            ) from exc

        if size <= 0:
            raise ArtifactValidationError(
                "Model artifact is empty."
            )

        if size > self.max_model_size_bytes:
            raise ArtifactValidationError(
                "Model artifact exceeds configured size limit."
            )

        actual_hash = ModelIntegrity.sha256_file(
            artifact
        )

        expected_hash = (
            ModelIntegrity.normalize_sha256(
                manifest.artifact_sha256
            )
        )

        if (
            expected_hash is not None
            and not _constant_time_equal(
                actual_hash,
                expected_hash,
            )
        ):

            with self._lock:
                self._stats.integrity_failures += 1

            raise ArtifactValidationError(
                f"SHA-256 integrity verification failed "
                f"for model '{manifest.model_id}'."
            )

        return actual_hash

    # =========================================================================
    # LOAD SESSION
    # =========================================================================

    def load_session(
        self,
        model_id: str,
    ) -> Optional[Any]:
        """
        Load and cache an ONNX Runtime session.

        Returns None on controlled registry/model errors to preserve the
        original public API.
        """

        try:

            return self._load_session_strict(
                model_id
            )

        except ModelRegistryError as exc:

            logger.error(
                "Model '%s' could not be loaded: %s",
                model_id,
                exc,
            )

            return None

        except Exception:

            logger.exception(
                "Unexpected model loading failure for '%s'.",
                model_id,
            )

            return None

    def load_session_strict(
        self,
        model_id: str,
    ) -> Any:
        """Strict loading API that raises ModelRegistryError."""
        return self._load_session_strict(
            model_id
        )

    def _load_session_strict(
        self,
        model_id: str,
    ) -> Any:

        if not ONNX_AVAILABLE:
            raise ModelLoadError(
                "onnxruntime is not installed."
            )

        model_lock = self._get_model_lock(
            model_id
        )

        with model_lock:

            with self._lock:

                manifest = (
                    self._manifests.get(
                        model_id
                    )
                )

                if manifest is None:
                    raise ModelLoadError(
                        f"No manifest registered for "
                        f"model '{model_id}'."
                    )

                status = self._status.get(
                    model_id,
                    ModelStatus.REGISTERED,
                )

                generation = (
                    self._generation.get(
                        model_id,
                        0,
                    )
                )

                if (
                    not manifest.enabled
                    or status
                    in {
                        ModelStatus.DISABLED,
                        ModelStatus.REVOKED,
                    }
                ):
                    raise ModelLoadError(
                        f"Model '{model_id}' is disabled "
                        "or revoked."
                    )

                cached = self._sessions.get(
                    model_id
                )

            artifact = self._path_resolver.resolve(
                manifest.artifact_path
            )

            # -----------------------------------------------------------------
            # Cache validation
            # -----------------------------------------------------------------

            if cached is not None:

                try:

                    current_hash = (
                        ModelIntegrity.sha256_file(
                            artifact
                        )
                    )

                except Exception:

                    with self._lock:
                        self._sessions.pop(
                            model_id,
                            None,
                        )

                    raise

                if _constant_time_equal(
                    current_hash,
                    cached.artifact_sha256,
                ):

                    with self._lock:
                        self._stats.cache_hits += 1

                    return cached.session

                logger.warning(
                    "Artifact changed for model '%s'; "
                    "invalidating cached session.",
                    model_id,
                )

                with self._lock:
                    self._sessions.pop(
                        model_id,
                        None,
                    )

                    self._stats.reload_count += 1

            with self._lock:
                self._stats.cache_misses += 1

            # -----------------------------------------------------------------
            # Verify artifact before loading
            # -----------------------------------------------------------------

            started = time.monotonic()

            try:

                verified_hash = (
                    self._verify_artifact(
                        manifest,
                        artifact,
                    )
                )

                session = self._create_session(
                    artifact
                )

                runtime_info = (
                    ONNXModelInspector.inspect(
                        session
                    )
                )

                self._validate_runtime_contract(
                    manifest,
                    runtime_info,
                )

            except ArtifactValidationError:

                with self._lock:

                    self._status[
                        model_id
                    ] = ModelStatus.FAILED

                    self._stats.failed_loads += 1

                raise

            except ModelLoadError:

                with self._lock:

                    self._status[
                        model_id
                    ] = ModelStatus.FAILED

                    self._stats.failed_loads += 1

                raise

            except Exception as exc:

                with self._lock:

                    self._status[
                        model_id
                    ] = ModelStatus.FAILED

                    self._stats.failed_loads += 1

                raise ModelLoadError(
                    f"Unexpected failure loading "
                    f"model '{model_id}'."
                ) from exc

            # -----------------------------------------------------------------
            # Prevent stale concurrent registration from being cached.
            # -----------------------------------------------------------------

            with self._lock:

                current_manifest = (
                    self._manifests.get(
                        model_id
                    )
                )

                current_generation = (
                    self._generation.get(
                        model_id,
                        0,
                    )
                )

                if (
                    current_manifest
                    is not manifest
                    or current_generation
                    != generation
                ):

                    raise ModelLoadError(
                        "Model manifest changed while "
                        "loading; refusing to cache stale session."
                    )

                file_size = artifact.stat().st_size

                self._sessions[
                    model_id
                ] = _LoadedSession(
                    session=session,
                    artifact_sha256=verified_hash,
                    artifact_size=file_size,
                    loaded_at_monotonic=time.monotonic(),
                    runtime_info=runtime_info,
                )

                self._status[
                    model_id
                ] = ModelStatus.READY

                self._stats.successful_loads += 1

                self._stats.last_load_time_ms = (
                    time.monotonic()
                    - started
                ) * 1000.0

                self._refresh_stats_locked()

            logger.info(
                "Loaded model=%s version=%s "
                "providers=%s time=%.2fms",
                model_id,
                manifest.version,
                runtime_info.providers,
                self._stats.last_load_time_ms,
            )

            return session

    # =========================================================================
    # SESSION CREATION
    # =========================================================================

    def _create_session(
        self,
        artifact: Path,
    ) -> Any:

        if not ONNX_AVAILABLE:
            raise ModelLoadError(
                "onnxruntime is unavailable."
            )

        try:

            available = set(
                ort.get_available_providers()
            )

        except Exception as exc:

            raise ModelLoadError(
                "Unable to query ONNX Runtime providers."
            ) from exc

        requested = list(
            self.allowed_providers
        )

        unavailable = [
            provider
            for provider in requested
            if provider not in available
        ]

        if unavailable:
            raise ModelLoadError(
                "Configured ONNX providers are unavailable: "
                + ", ".join(unavailable)
            )

        try:

            options = ort.SessionOptions()

            options.intra_op_num_threads = (
                self.intra_op_num_threads
            )

            options.inter_op_num_threads = (
                self.inter_op_num_threads
            )

            options.execution_mode = (
                ort.ExecutionMode.ORT_SEQUENTIAL
            )

            if self.enable_graph_optimization:

                options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
                )

            else:

                options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                )

            # Disable profiling in production inference unless explicitly
            # enabled elsewhere.
            try:
                options.enable_profiling = False
            except Exception:
                pass

            session = ort.InferenceSession(
                str(artifact),
                sess_options=options,
                providers=requested,
            )

        except Exception as exc:

            raise ModelLoadError(
                "ONNX Runtime failed to initialize the model."
            ) from exc

        # ---------------------------------------------------------------------
        # Verify actual providers after creation.
        # ---------------------------------------------------------------------

        try:

            actual_providers = tuple(
                str(provider)
                for provider
                in session.get_providers()
            )

        except Exception as exc:

            raise ModelLoadError(
                "Unable to inspect active ONNX providers."
            ) from exc

        unauthorized = [
            provider
            for provider in actual_providers
            if provider not in self.allowed_providers
        ]

        if unauthorized:

            try:
                del session
            except Exception:
                pass

            raise ModelLoadError(
                "ONNX Runtime activated unauthorized "
                "execution provider(s): "
                + ", ".join(unauthorized)
            )

        return session

    # =========================================================================
    # CONTRACT VALIDATION
    # =========================================================================

    def _validate_runtime_contract(
        self,
        manifest: ModelManifest,
        runtime_info: ModelRuntimeInfo,
    ) -> None:

        if not runtime_info.input_names:

            with self._lock:
                self._stats.contract_failures += 1

            raise ModelLoadError(
                f"Model '{manifest.model_id}' has no inputs."
            )

        if not runtime_info.output_names:

            with self._lock:
                self._stats.contract_failures += 1

            raise ModelLoadError(
                f"Model '{manifest.model_id}' has no outputs."
            )

        # ---------------------------------------------------------------------
        # Input name
        # ---------------------------------------------------------------------

        if manifest.expected_input_name is not None:

            if (
                manifest.expected_input_name
                not in runtime_info.input_names
            ):

                with self._lock:
                    self._stats.contract_failures += 1

                raise ModelLoadError(
                    f"Expected input "
                    f"'{manifest.expected_input_name}' "
                    "was not found."
                )

        # ---------------------------------------------------------------------
        # Input type
        # ---------------------------------------------------------------------

        if (
            manifest.expected_input_name is not None
            and manifest.expected_input_type is not None
        ):

            index = runtime_info.input_names.index(
                manifest.expected_input_name
            )

            actual_type = (
                runtime_info.input_types[index]
            )

            if (
                actual_type
                != manifest.expected_input_type
            ):

                with self._lock:
                    self._stats.contract_failures += 1

                raise ModelLoadError(
                    "Model input type mismatch: "
                    f"expected={manifest.expected_input_type}, "
                    f"actual={actual_type}"
                )

        # ---------------------------------------------------------------------
        # Input rank
        # ---------------------------------------------------------------------

        if (
            manifest.expected_input_name is not None
            and manifest.expected_input_rank is not None
        ):

            index = runtime_info.input_names.index(
                manifest.expected_input_name
            )

            shape = (
                ONNXModelInspector.parse_shape(
                    runtime_info.input_shapes[index]
                )
            )

            if shape and len(shape) != manifest.expected_input_rank:

                with self._lock:
                    self._stats.contract_failures += 1

                raise ModelLoadError(
                    "Model input rank mismatch: "
                    f"expected={manifest.expected_input_rank}, "
                    f"actual={len(shape)}"
                )

        # ---------------------------------------------------------------------
        # Feature count
        # ---------------------------------------------------------------------

        if (
            manifest.expected_input_name is not None
            and manifest.expected_feature_count is not None
        ):

            index = runtime_info.input_names.index(
                manifest.expected_input_name
            )

            shape = (
                ONNXModelInspector.parse_shape(
                    runtime_info.input_shapes[index]
                )
            )

            if shape:

                last_dimension = shape[-1]

                if isinstance(last_dimension, int):

                    if (
                        last_dimension
                        != manifest.expected_feature_count
                    ):

                        with self._lock:
                            self._stats.contract_failures += 1

                        raise ModelLoadError(
                            "Model feature dimension mismatch: "
                            f"expected="
                            f"{manifest.expected_feature_count}, "
                            f"actual={last_dimension}"
                        )

        # ---------------------------------------------------------------------
        # Output name
        # ---------------------------------------------------------------------

        if manifest.expected_output_name is not None:

            if (
                manifest.expected_output_name
                not in runtime_info.output_names
            ):

                with self._lock:
                    self._stats.contract_failures += 1

                raise ModelLoadError(
                    f"Expected output "
                    f"'{manifest.expected_output_name}' "
                    "was not found."
                )

        # ---------------------------------------------------------------------
        # Output type
        # ---------------------------------------------------------------------

        if (
            manifest.expected_output_name is not None
            and manifest.expected_output_type is not None
        ):

            index = runtime_info.output_names.index(
                manifest.expected_output_name
            )

            actual_type = (
                runtime_info.output_types[index]
            )

            if (
                actual_type
                != manifest.expected_output_type
            ):

                with self._lock:
                    self._stats.contract_failures += 1

                raise ModelLoadError(
                    "Model output type mismatch: "
                    f"expected={manifest.expected_output_type}, "
                    f"actual={actual_type}"
                )

        # ---------------------------------------------------------------------
        # Feature schema hash
        # ---------------------------------------------------------------------

        if manifest.feature_schema_hash is not None:

            calculated_hash = (
                ModelIntegrity.feature_schema_hash(
                    manifest.feature_names
                )
            )

            expected_hash = (
                str(
                    manifest.feature_schema_hash
                ).strip().lower()
            )

            if not _constant_time_equal(
                calculated_hash,
                expected_hash,
            ):

                with self._lock:
                    self._stats.contract_failures += 1

                raise ModelLoadError(
                    f"Feature schema hash mismatch "
                    f"for model '{manifest.model_id}'."
                )

    # =========================================================================
    # UNLOAD / RELOAD
    # =========================================================================

    def unload(
        self,
        model_id: str,
    ) -> bool:

        lock = self._get_model_lock(
            model_id
        )

        with lock:

            with self._lock:

                if model_id not in self._manifests:
                    return False

                existed = (
                    model_id
                    in self._sessions
                )

                self._sessions.pop(
                    model_id,
                    None,
                )

                if existed:
                    self._stats.unload_count += 1

                self._refresh_stats_locked()

            logger.info(
                "Unloaded model session '%s'.",
                model_id,
            )

            return existed

    def reload(
        self,
        model_id: str,
    ) -> Any:

        self.unload(model_id)

        with self._lock:
            self._stats.reload_count += 1

        return self.load_session_strict(
            model_id
        )

    # =========================================================================
    # HEALTH
    # =========================================================================

    def health_check(
        self,
        model_id: str,
    ) -> Dict[str, Any]:

        manifest = self.get_manifest(
            model_id
        )

        if manifest is None:

            return {
                "model_id": model_id,
                "healthy": False,
                "status": "not_registered",
            }

        artifact_ok = self.validate_artifact(
            model_id
        )

        status = self.get_status(
            model_id
        )

        runtime_info = self.get_runtime_info(
            model_id
        )

        return {
            "model_id": model_id,
            "version": manifest.version,
            "model_type": manifest.model_type,
            "healthy": (
                artifact_ok
                and status
                not in {
                    ModelStatus.DISABLED,
                    ModelStatus.REVOKED,
                    ModelStatus.FAILED,
                }
            ),
            "status": (
                status.value
                if status
                else None
            ),
            "artifact_valid": artifact_ok,
            "session_loaded": runtime_info is not None,
            "providers": (
                list(runtime_info.providers)
                if runtime_info
                else []
            ),
            "runtime_version": (
                runtime_info.runtime_version
                if runtime_info
                else None
            ),
        }

    # =========================================================================
    # STATS
    # =========================================================================

    def get_stats(self) -> RegistryStats:

        with self._lock:

            self._refresh_stats_locked()

            return RegistryStats(
                manifest_count=self._stats.manifest_count,
                ready_count=self._stats.ready_count,
                registered_count=self._stats.registered_count,
                failed_count=self._stats.failed_count,
                disabled_count=self._stats.disabled_count,
                revoked_count=self._stats.revoked_count,
                session_count=self._stats.session_count,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
                successful_loads=self._stats.successful_loads,
                failed_loads=self._stats.failed_loads,
                integrity_failures=self._stats.integrity_failures,
                validation_failures=self._stats.validation_failures,
                contract_failures=self._stats.contract_failures,
                unload_count=self._stats.unload_count,
                reload_count=self._stats.reload_count,
                last_load_time_ms=self._stats.last_load_time_ms,
            )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _require_manifest_locked(
        self,
        model_id: str,
    ) -> ModelManifest:

        manifest = self._manifests.get(
            model_id
        )

        if manifest is None:
            raise ModelRegistryError(
                f"Model '{model_id}' is not registered."
            )

        return manifest

    def _refresh_stats_locked(
        self,
    ) -> None:

        self._stats.manifest_count = (
            len(self._manifests)
        )

        self._stats.session_count = (
            len(self._sessions)
        )

        self._stats.ready_count = sum(
            1
            for status in self._status.values()
            if status == ModelStatus.READY
        )

        self._stats.registered_count = sum(
            1
            for status in self._status.values()
            if status == ModelStatus.REGISTERED
        )

        self._stats.failed_count = sum(
            1
            for status in self._status.values()
            if status == ModelStatus.FAILED
        )

        self._stats.disabled_count = sum(
            1
            for status in self._status.values()
            if status == ModelStatus.DISABLED
        )

        self._stats.revoked_count = sum(
            1
            for status in self._status.values()
            if status == ModelStatus.REVOKED
        )


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    "ArtifactValidationError",
    "DEFAULT_ALLOWED_PROVIDERS",
    "DEFAULT_MAX_FEATURES",
    "DEFAULT_MAX_MODEL_SIZE_BYTES",
    "DEFAULT_MAX_MODELS",
    "DEFAULT_MODELS_DIR",
    "DEFAULT_THRESHOLD_PHISHING",
    "DEFAULT_THRESHOLD_SUSPICIOUS",
    "ManifestValidationError",
    "ModelIntegrity",
    "ModelLoadError",
    "ModelManifest",
    "ModelManifestValidator",
    "ModelPathResolver",
    "ModelRegistry",
    "ModelRegistryError",
    "ModelRuntimeInfo",
    "ModelStatus",
    "ONNXModelInspector",
    "RegistryStats",
]
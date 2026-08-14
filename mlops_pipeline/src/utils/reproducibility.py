"""
PhishGuard AI - Deterministic Reproducibility & Experiment Provenance Engine
============================================================================

Version: 13.0.0
File Path: mlops_pipeline/src/utils/reproducibility.py

Purpose
-------
Enterprise-grade reproducibility infrastructure for the PhishGuard AI MLOps
pipeline.

This module provides:

    * Global Python / NumPy / PyTorch seed management
    * Strict and best-effort determinism modes
    * CPU and CUDA RNG state capture/restoration
    * Scoped deterministic execution contexts
    * Distributed/rank-aware seed derivation
    * DataLoader worker seed generation
    * NumPy Generator support
    * PyTorch deterministic algorithm controls
    * cuDNN / cuBLAS configuration
    * Runtime reproducibility diagnostics
    * Environment and dependency provenance
    * Reproducibility fingerprints
    * RNG-state snapshots
    * Deterministic experiment manifests
    * Detection of potentially unsafe runtime configurations
    * Fail-closed behavior for strict production evaluation
    * Safe handling of nested deterministic contexts

Security / Reliability Principles
---------------------------------
1. Determinism must be explicit.
2. Strict mode must fail closed when deterministic execution cannot be
   established.
3. A seed alone MUST NOT be represented as proof of reproducibility.
4. Runtime provenance must accompany every production evaluation.
5. CUDA/PyTorch configuration must be recorded.
6. Environment-dependent sources of nondeterminism must be surfaced.
7. Random state must be restorable for isolated evaluation.
8. Distributed workers must receive independent but reproducible seeds.
9. Secrets and sensitive environment variables must never be included in
   provenance output.
10. Cross-machine bit-exact reproducibility is only claimed when runtime,
    dependency, hardware, and backend conditions are sufficiently aligned.

IMPORTANT
---------
This module does not guarantee mathematical or hardware-independent
reproducibility. GPU drivers, CUDA versions, kernels, BLAS implementations,
hardware architecture, compiler versions, distributed execution order, and
unsupported nondeterministic operators may still produce differences.

Production consumers should persist the reproducibility manifest together with
the model/evaluation artifact.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import random
import sys
import threading
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from typing import Any, Dict, Generator, Iterable, Mapping, Optional, Sequence

import numpy as np

from src.utils.logger import get_logger


logger = get_logger("src.utils.reproducibility")


# ==============================================================================
# OPTIONAL DEPENDENCIES
# ==============================================================================

try:
    import torch

    HAS_TORCH: bool = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False


try:
    import sklearn

    HAS_SKLEARN: bool = True
except ImportError:
    sklearn = None  # type: ignore[assignment]
    HAS_SKLEARN = False


# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_GLOBAL_SEED: int = 42

# Python's Mersenne Twister internally accepts a very broad integer range, but
# keeping the public seed in a bounded unsigned 64-bit range makes provenance
# portable across Python, NumPy, PyTorch, distributed workers, and external
# systems.
MIN_SEED: int = 0
MAX_SEED: int = (2**63) - 1

REPRODUCIBILITY_SCHEMA_VERSION: str = "2.0.0"

# Environment variables which can affect numerical reproducibility.
# Values are intentionally allow-listed instead of dumping the entire
# environment into provenance.
REPRODUCIBILITY_ENV_KEYS: tuple[str, ...] = (
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "MKL_DYNAMIC",
    "CUDA_LAUNCH_BLOCKING",
    "NVIDIA_TF32_OVERRIDE",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
)


# ==============================================================================
# GLOBAL STATE
# ==============================================================================

_STATE_LOCK = threading.RLock()

_ACTIVE_SEED: Optional[int] = None
_ACTIVE_STRICT_MODE: Optional[bool] = None
_CONTEXT_DEPTH: int = 0


# ==============================================================================
# EXCEPTIONS
# ==============================================================================


class ReproducibilityError(RuntimeError):
    """Base exception for reproducibility failures."""


class DeterminismConfigurationError(ReproducibilityError):
    """Raised when strict deterministic execution cannot be configured."""


class InvalidSeedError(ValueError):
    """Raised when a supplied seed is invalid."""


# ==============================================================================
# DATA CLASSES
# ==============================================================================


@dataclass(frozen=True)
class RuntimeFingerprint:
    """
    Immutable runtime identity information.

    This is intentionally descriptive rather than authoritative. A fingerprint
    does not prove two executions are bit-identical; it provides evidence that
    the relevant runtime conditions were recorded.
    """

    python_version: str
    implementation: str
    operating_system: str
    architecture: str
    machine: str
    processor: str
    pytorch_version: Optional[str]
    numpy_version: str
    sklearn_version: Optional[str]
    cuda_available: bool
    cuda_version: Optional[str]
    cudnn_version: Optional[int]
    device_count: int
    device_names: tuple[str, ...]
    deterministic_algorithms_enabled: Optional[bool]
    cudnn_deterministic: Optional[bool]
    cudnn_benchmark: Optional[bool]
    allow_tf32: Optional[bool]
    cublas_workspace_config: Optional[str]


@dataclass(frozen=True)
class SeedConfiguration:
    """
    Seed configuration used for an experiment/process.
    """

    global_seed: int
    rank: int = 0
    world_size: int = 1
    worker_id: int = 0
    derived_seed: Optional[int] = None
    strict_determinism: bool = True


@dataclass
class RNGSnapshot:
    """
    Captured RNG state.

    RNG states are runtime objects and should generally not be serialized
    directly into JSON. Use `snapshot_metadata()` for JSON-safe provenance.
    """

    python_state: Any = None
    numpy_state: Any = None
    torch_cpu_state: Any = None
    torch_cuda_states: Optional[list[Any]] = None
    active_seed: Optional[int] = None


@dataclass
class ReproducibilityManifest:
    """
    JSON-serializable reproducibility manifest.

    Persist this alongside a model artifact, evaluation report, or training
    run metadata when reproducibility is important.
    """

    schema_version: str
    created_at_utc: str
    seed_configuration: Dict[str, Any]
    runtime: Dict[str, Any]
    environment: Dict[str, Optional[str]]
    warnings: list[str] = field(default_factory=list)
    reproducibility_claim: str = "best_effort"
    fingerprint: Optional[str] = None


# ==============================================================================
# SEED VALIDATION
# ==============================================================================


def _validate_seed(seed: int) -> int:
    """
    Validate and normalize a public seed.

    Booleans are explicitly rejected because `bool` subclasses `int` in Python.
    """

    if isinstance(seed, bool):
        raise InvalidSeedError(
            f"Seed must be an integer, not boolean. Received: {seed!r}"
        )

    if not isinstance(seed, int):
        raise InvalidSeedError(
            f"Seed must be an integer. Received {type(seed).__name__}: {seed!r}"
        )

    if not MIN_SEED <= seed <= MAX_SEED:
        raise InvalidSeedError(
            f"Seed must be within [{MIN_SEED}, {MAX_SEED}]. Received: {seed}"
        )

    return seed


# ==============================================================================
# SEED DERIVATION
# ==============================================================================


def derive_seed(
    base_seed: int,
    *,
    rank: int = 0,
    worker_id: int = 0,
    stream: str = "default",
) -> int:
    """
    Deterministically derive an independent seed for a worker/rank/stream.

    SHA-256 is used only as a deterministic derivation mechanism, not as a
    cryptographic secret-generation mechanism.

    Parameters
    ----------
    base_seed:
        Experiment-level seed.
    rank:
        Distributed process rank.
    worker_id:
        DataLoader / worker identifier.
    stream:
        Logical random stream name.

    Returns
    -------
    int
        Stable derived seed.
    """

    base_seed = _validate_seed(base_seed)

    if not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer.")

    if not isinstance(worker_id, int) or worker_id < 0:
        raise ValueError("worker_id must be a non-negative integer.")

    if not isinstance(stream, str) or not stream:
        raise ValueError("stream must be a non-empty string.")

    material = (
        f"phishguard-repro-v2|"
        f"{base_seed}|"
        f"rank={rank}|"
        f"worker={worker_id}|"
        f"stream={stream}"
    ).encode("utf-8")

    digest = hashlib.sha256(material).digest()

    # Stay inside the public signed 64-bit range.
    derived = int.from_bytes(digest[:8], "big") & MAX_SEED

    return derived


# ==============================================================================
# ENVIRONMENT CONFIGURATION
# ==============================================================================


def _configure_cublas_environment(strict: bool) -> None:
    """
    Configure the cuBLAS workspace environment required by certain deterministic
    CUDA operations.

    IMPORTANT:
    This environment variable is ideally configured before CUDA/PyTorch
    initialization. Setting it here is useful for subprocesses or environments
    where initialization has not yet occurred, but cannot retroactively undo
    CUDA initialization that has already happened.
    """

    if not strict:
        return

    expected = ":4096:8"
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    if current is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected
        logger.debug(
            "CUBLAS_WORKSPACE_CONFIG was not set; configured deterministic "
            "workspace value '%s'.",
            expected,
        )
    elif current not in {":4096:8", ":16:8"}:
        message = (
            "CUBLAS_WORKSPACE_CONFIG is set to an unsupported value for the "
            f"strict reproducibility policy: {current!r}."
        )

        if strict:
            raise DeterminismConfigurationError(message)

        logger.warning(message)


def _configure_python_hash_seed(seed: int) -> None:
    """
    Record the requested PYTHONHASHSEED.

    Python's hash randomization is established during interpreter startup.
    Setting this variable after startup does NOT change the current interpreter's
    hash seed. Therefore this function only records the requested value and
    explicitly documents that limitation.
    """

    requested = str(seed)
    current = os.environ.get("PYTHONHASHSEED")

    if current is None:
        os.environ["PYTHONHASHSEED"] = requested

        logger.warning(
            "PYTHONHASHSEED was not set at interpreter startup. The current "
            "process hash seed cannot be changed retroactively. The value '%s' "
            "has been recorded for child-process inheritance only.",
            requested,
        )
        return

    if current != requested:
        logger.warning(
            "PYTHONHASHSEED=%s differs from requested reproducibility seed=%s. "
            "Changing the environment variable now does not change the current "
            "interpreter hash seed.",
            current,
            requested,
        )


def _configure_torch_determinism(strict: bool) -> None:
    """
    Configure PyTorch deterministic execution.

    In strict mode, unsupported nondeterministic operations should fail rather
    than silently produce potentially different results.
    """

    if not HAS_TORCH:
        if strict:
            logger.warning(
                "Strict PyTorch determinism requested but PyTorch is unavailable. "
                "Continuing because the current task may be CPU/NumPy-only."
            )
        return

    assert torch is not None

    try:
        if hasattr(torch, "use_deterministic_algorithms"):
            # Production strict mode deliberately uses warn_only=False.
            torch.use_deterministic_algorithms(strict, warn_only=False)

    except Exception as exc:
        if strict:
            raise DeterminismConfigurationError(
                "Unable to enable strict PyTorch deterministic algorithms."
            ) from exc

        logger.warning(
            "Unable to configure PyTorch deterministic algorithms: %s",
            exc,
        )

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = strict
        torch.backends.cudnn.benchmark = not strict

    # Newer PyTorch versions expose explicit TF32 controls.
    if hasattr(torch.backends, "cuda"):
        matmul = getattr(torch.backends.cuda, "matmul", None)

        if matmul is not None and hasattr(matmul, "allow_tf32"):
            if strict:
                matmul.allow_tf32 = False

    if hasattr(torch.backends, "cudnn"):
        cudnn = torch.backends.cudnn

        if hasattr(cudnn, "allow_tf32"):
            if strict:
                cudnn.allow_tf32 = False


# ==============================================================================
# RNG SEED APPLICATION
# ==============================================================================


def _seed_python(seed: int) -> None:
    """Seed Python's global pseudo-random number generator."""

    random.seed(seed)


def _seed_numpy(seed: int) -> None:
    """
    Seed NumPy's legacy global RandomState.

    NumPy Generator instances are intentionally not modified because they are
    independently owned objects. Call `create_numpy_generator()` when a
    deterministic Generator is required.
    """

    # NumPy legacy seed accepts a uint32-compatible range.
    np.random.seed(seed % (2**32))


def _seed_torch(seed: int) -> None:
    """Seed PyTorch CPU and CUDA generators."""

    if not HAS_TORCH:
        return

    assert torch is not None

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# PUBLIC SEED API
# ==============================================================================


def set_seed(
    seed: int = DEFAULT_GLOBAL_SEED,
    enforce_strict_determinism: bool = True,
) -> int:
    """
    Establish global reproducibility configuration.

    Parameters
    ----------
    seed:
        Global experiment seed.
    enforce_strict_determinism:
        If True, configure PyTorch/CUDA for fail-fast deterministic behavior.

    Returns
    -------
    int
        Validated seed.

    Raises
    ------
    InvalidSeedError
        If the seed is invalid.
    DeterminismConfigurationError
        If strict deterministic configuration cannot be established.

    Notes
    -----
    For maximum reproducibility, set `PYTHONHASHSEED` and
    `CUBLAS_WORKSPACE_CONFIG` before starting Python/PyTorch.
    """

    global _ACTIVE_SEED, _ACTIVE_STRICT_MODE

    seed = _validate_seed(seed)

    with _STATE_LOCK:
        _configure_python_hash_seed(seed)

        if enforce_strict_determinism:
            _configure_cublas_environment(strict=True)

        _seed_python(seed)
        _seed_numpy(seed)
        _seed_torch(seed)

        _configure_torch_determinism(enforce_strict_determinism)

        _ACTIVE_SEED = seed
        _ACTIVE_STRICT_MODE = enforce_strict_determinism

    logger.info(
        "Reproducibility configuration applied. seed=%d strict=%s pytorch=%s",
        seed,
        enforce_strict_determinism,
        HAS_TORCH,
    )

    return seed


# ==============================================================================
# NUMPY GENERATOR SUPPORT
# ==============================================================================


def create_numpy_generator(
    seed: int = DEFAULT_GLOBAL_SEED,
    *,
    rank: int = 0,
    worker_id: int = 0,
    stream: str = "numpy",
) -> np.random.Generator:
    """
    Create an independently reproducible NumPy Generator.

    This is preferred over relying on global NumPy state for modern code.
    """

    derived = derive_seed(
        seed,
        rank=rank,
        worker_id=worker_id,
        stream=stream,
    )

    return np.random.default_rng(derived)


# ==============================================================================
# PYTORCH GENERATOR SUPPORT
# ==============================================================================


def create_torch_generator(
    seed: int = DEFAULT_GLOBAL_SEED,
    *,
    rank: int = 0,
    worker_id: int = 0,
    stream: str = "torch",
    device: Optional[str] = None,
) -> Any:
    """
    Create a reproducible PyTorch Generator.

    Returns
    -------
    torch.Generator

    Raises
    ------
    ReproducibilityError
        If PyTorch is unavailable.
    """

    if not HAS_TORCH:
        raise ReproducibilityError(
            "PyTorch is not installed; create_torch_generator() is unavailable."
        )

    assert torch is not None

    derived = derive_seed(
        seed,
        rank=rank,
        worker_id=worker_id,
        stream=stream,
    )

    generator = torch.Generator(device=device) if device else torch.Generator()
    generator.manual_seed(derived)

    return generator


# ==============================================================================
# DATALOADER WORKER SEEDING
# ==============================================================================


def seed_worker(
    worker_id: int,
    base_seed: Optional[int] = None,
    *,
    rank: int = 0,
) -> None:
    """
    Seed Python and NumPy RNGs for a PyTorch DataLoader worker.

    Intended for use as a DataLoader `worker_init_fn`.
    """

    if base_seed is None:
        base_seed = (
            _ACTIVE_SEED
            if _ACTIVE_SEED is not None
            else DEFAULT_GLOBAL_SEED
        )

    worker_seed = derive_seed(
        base_seed,
        rank=rank,
        worker_id=worker_id,
        stream="dataloader",
    )

    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))

    if HAS_TORCH:
        assert torch is not None
        torch.manual_seed(worker_seed)

    logger.debug(
        "Initialized deterministic DataLoader worker. worker_id=%d rank=%d.",
        worker_id,
        rank,
    )


# ==============================================================================
# RNG SNAPSHOT
# ==============================================================================


def capture_rng_state() -> RNGSnapshot:
    """
    Capture global Python, NumPy, and PyTorch RNG states.
    """

    snapshot = RNGSnapshot(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        active_seed=_ACTIVE_SEED,
    )

    if HAS_TORCH:
        assert torch is not None

        snapshot.torch_cpu_state = torch.get_rng_state()

        if torch.cuda.is_available():
            snapshot.torch_cuda_states = torch.cuda.get_rng_state_all()

    return snapshot


def restore_rng_state(snapshot: RNGSnapshot) -> None:
    """
    Restore a previously captured RNG snapshot.
    """

    if not isinstance(snapshot, RNGSnapshot):
        raise TypeError(
            f"snapshot must be RNGSnapshot, received {type(snapshot).__name__}"
        )

    if snapshot.python_state is not None:
        random.setstate(snapshot.python_state)

    if snapshot.numpy_state is not None:
        np.random.set_state(snapshot.numpy_state)

    if HAS_TORCH:
        assert torch is not None

        if snapshot.torch_cpu_state is not None:
            torch.set_rng_state(snapshot.torch_cpu_state)

        if (
            torch.cuda.is_available()
            and snapshot.torch_cuda_states is not None
        ):
            torch.cuda.set_rng_state_all(snapshot.torch_cuda_states)

    logger.debug("Global RNG state restored from reproducibility snapshot.")


# ==============================================================================
# DETERMINISTIC CONTEXT
# ==============================================================================


class DeterministicContext:
    """
    Scoped deterministic execution context.

    Unlike the original implementation, this context restores:

        * Python RNG state
        * NumPy RNG state
        * PyTorch CPU RNG state
        * PyTorch CUDA RNG state
        * deterministic algorithm setting
        * cuDNN deterministic setting
        * cuDNN benchmark setting
        * TF32-related settings
        * relevant environment variables

    This makes the context substantially safer for evaluation code embedded
    inside larger training processes.

    Example
    -------
    with DeterministicContext(seed=42, strict=True):
        evaluate_model()
    """

    def __init__(
        self,
        seed: int = DEFAULT_GLOBAL_SEED,
        strict: bool = True,
    ) -> None:
        self.seed = _validate_seed(seed)
        self.strict = bool(strict)

        self._snapshot: Optional[RNGSnapshot] = None
        self._environment: Dict[str, Optional[str]] = {}
        self._torch_settings: Dict[str, Any] = {}
        self._previous_seed: Optional[int] = None
        self._previous_strict: Optional[bool] = None
        self._entered = False

    def _capture_torch_settings(self) -> None:
        if not HAS_TORCH:
            return

        assert torch is not None

        if hasattr(torch, "are_deterministic_algorithms_enabled"):
            self._torch_settings[
                "deterministic_algorithms_enabled"
            ] = torch.are_deterministic_algorithms_enabled()

        if hasattr(torch.backends, "cudnn"):
            self._torch_settings["cudnn_deterministic"] = (
                torch.backends.cudnn.deterministic
            )
            self._torch_settings["cudnn_benchmark"] = (
                torch.backends.cudnn.benchmark
            )

            if hasattr(torch.backends.cudnn, "allow_tf32"):
                self._torch_settings["cudnn_allow_tf32"] = (
                    torch.backends.cudnn.allow_tf32
                )

        if hasattr(torch.backends, "cuda"):
            matmul = getattr(torch.backends.cuda, "matmul", None)

            if matmul is not None and hasattr(matmul, "allow_tf32"):
                self._torch_settings["cuda_matmul_allow_tf32"] = (
                    matmul.allow_tf32
                )

    def _restore_torch_settings(self) -> None:
        if not HAS_TORCH:
            return

        assert torch is not None

        deterministic = self._torch_settings.get(
            "deterministic_algorithms_enabled"
        )

        if (
            deterministic is not None
            and hasattr(torch, "use_deterministic_algorithms")
        ):
            try:
                torch.use_deterministic_algorithms(
                    deterministic,
                    warn_only=False,
                )
            except Exception as exc:
                logger.warning(
                    "Could not restore PyTorch deterministic algorithm "
                    "setting: %s",
                    exc,
                )

        if hasattr(torch.backends, "cudnn"):
            if "cudnn_deterministic" in self._torch_settings:
                torch.backends.cudnn.deterministic = self._torch_settings[
                    "cudnn_deterministic"
                ]

            if "cudnn_benchmark" in self._torch_settings:
                torch.backends.cudnn.benchmark = self._torch_settings[
                    "cudnn_benchmark"
                ]

            if hasattr(torch.backends.cudnn, "allow_tf32"):
                if "cudnn_allow_tf32" in self._torch_settings:
                    torch.backends.cudnn.allow_tf32 = self._torch_settings[
                        "cudnn_allow_tf32"
                    ]

        if hasattr(torch.backends, "cuda"):
            matmul = getattr(torch.backends.cuda, "matmul", None)

            if (
                matmul is not None
                and hasattr(matmul, "allow_tf32")
                and "cuda_matmul_allow_tf32" in self._torch_settings
            ):
                matmul.allow_tf32 = self._torch_settings[
                    "cuda_matmul_allow_tf32"
                ]

    def __enter__(self) -> "DeterministicContext":
        global _CONTEXT_DEPTH

        with _STATE_LOCK:
            if self._entered:
                raise RuntimeError(
                    "DeterministicContext instances cannot be entered twice."
                )

            self._snapshot = capture_rng_state()
            self._previous_seed = _ACTIVE_SEED
            self._previous_strict = _ACTIVE_STRICT_MODE

            for key in (
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
            ):
                self._environment[key] = os.environ.get(key)

            self._capture_torch_settings()

            _CONTEXT_DEPTH += 1

            try:
                set_seed(
                    self.seed,
                    enforce_strict_determinism=self.strict,
                )
            except Exception:
                _CONTEXT_DEPTH -= 1
                raise

            self._entered = True

        logger.debug(
            "Entered deterministic context. seed=%d strict=%s depth=%d.",
            self.seed,
            self.strict,
            _CONTEXT_DEPTH,
        )

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        global _ACTIVE_SEED
        global _ACTIVE_STRICT_MODE
        global _CONTEXT_DEPTH

        with _STATE_LOCK:
            if not self._entered:
                return

            try:
                if self._snapshot is not None:
                    restore_rng_state(self._snapshot)

                self._restore_torch_settings()

                for key, previous_value in self._environment.items():
                    if previous_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = previous_value

                _ACTIVE_SEED = self._previous_seed
                _ACTIVE_STRICT_MODE = self._previous_strict

            finally:
                _CONTEXT_DEPTH = max(0, _CONTEXT_DEPTH - 1)
                self._entered = False

        logger.debug(
            "Exited deterministic context; previous RNG and runtime state "
            "restored. depth=%d.",
            _CONTEXT_DEPTH,
        )

        # Returning None/False allows any exception raised inside the context
        # to propagate normally.
        return None


# ==============================================================================
# TORCH DETERMINISM DIAGNOSTICS
# ==============================================================================


def _torch_runtime_info() -> Dict[str, Any]:
    """Return JSON-safe PyTorch runtime information."""

    if not HAS_TORCH:
        return {
            "available": False,
        }

    assert torch is not None

    info: Dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": 0,
    }

    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        info["deterministic_algorithms_enabled"] = (
            torch.are_deterministic_algorithms_enabled()
        )

    if hasattr(torch.backends, "cudnn"):
        info["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        info["cudnn_benchmark"] = torch.backends.cudnn.benchmark

        if hasattr(torch.backends.cudnn, "version"):
            try:
                info["cudnn_version"] = torch.backends.cudnn.version()
            except Exception:
                info["cudnn_version"] = None

        if hasattr(torch.backends.cudnn, "allow_tf32"):
            info["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32

    if hasattr(torch.backends, "cuda"):
        matmul = getattr(torch.backends.cuda, "matmul", None)

        if matmul is not None and hasattr(matmul, "allow_tf32"):
            info["cuda_matmul_allow_tf32"] = matmul.allow_tf32

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        info["device_count"] = count
        info["devices"] = []

        for index in range(count):
            try:
                info["devices"].append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "capability": list(
                            torch.cuda.get_device_capability(index)
                        ),
                    }
                )
            except Exception as exc:
                info["devices"].append(
                    {
                        "index": index,
                        "error": type(exc).__name__,
                    }
                )

    return info


# ==============================================================================
# ENVIRONMENT PROVENANCE
# ==============================================================================


def get_reproducibility_environment() -> Dict[str, Optional[str]]:
    """
    Return an allow-listed set of reproducibility-related environment variables.

    Sensitive environment variables are deliberately excluded.
    """

    return {
        key: os.environ.get(key)
        for key in REPRODUCIBILITY_ENV_KEYS
    }


# ==============================================================================
# RUNTIME FINGERPRINT
# ==============================================================================


def get_runtime_fingerprint() -> RuntimeFingerprint:
    """
    Collect a stable runtime fingerprint suitable for provenance.
    """

    torch_info = _torch_runtime_info()

    devices = torch_info.get("devices", [])
    device_names = tuple(
        str(item.get("name", "unknown"))
        for item in devices
        if isinstance(item, dict)
    )

    return RuntimeFingerprint(
        python_version=platform.python_version(),
        implementation=platform.python_implementation(),
        operating_system=platform.system(),
        architecture=platform.architecture()[0],
        machine=platform.machine(),
        processor=platform.processor(),
        pytorch_version=torch_info.get("version"),
        numpy_version=np.__version__,
        sklearn_version=(
            getattr(sklearn, "__version__", None)
            if HAS_SKLEARN
            else None
        ),
        cuda_available=bool(torch_info.get("cuda_available", False)),
        cuda_version=torch_info.get("cuda_version"),
        cudnn_version=torch_info.get("cudnn_version"),
        device_count=int(torch_info.get("device_count", 0)),
        device_names=device_names,
        deterministic_algorithms_enabled=torch_info.get(
            "deterministic_algorithms_enabled"
        ),
        cudnn_deterministic=torch_info.get("cudnn_deterministic"),
        cudnn_benchmark=torch_info.get("cudnn_benchmark"),
        allow_tf32=(
            torch_info.get("cuda_matmul_allow_tf32")
            if torch_info.get("cuda_matmul_allow_tf32") is not None
            else torch_info.get("cudnn_allow_tf32")
        ),
        cublas_workspace_config=os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
    )


# ==============================================================================
# PACKAGE PROVENANCE
# ==============================================================================


def _package_version(package_name: str) -> Optional[str]:
    """Safely obtain an installed package version."""

    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def get_package_provenance() -> Dict[str, Optional[str]]:
    """
    Return versions of packages that materially affect ML reproducibility.
    """

    packages = (
        "numpy",
        "pandas",
        "pyarrow",
        "PyYAML",
        "scikit-learn",
        "torch",
        "scipy",
        "onnx",
        "onnxruntime",
    )

    return {
        package: _package_version(package)
        for package in packages
    }


# ==============================================================================
# DIAGNOSTIC WARNINGS
# ==============================================================================


def detect_reproducibility_risks(
    *,
    strict: bool = True,
) -> list[str]:
    """
    Inspect the current runtime for known reproducibility risks.

    Returns
    -------
    list[str]
        Human-readable risk descriptions.

    Notes
    -----
    This is diagnostic rather than a proof of determinism.
    """

    risks: list[str] = []

    hash_seed = os.environ.get("PYTHONHASHSEED")

    if hash_seed is None:
        risks.append(
            "PYTHONHASHSEED is not set. For deterministic hash iteration, "
            "configure it before interpreter startup."
        )

    cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    if HAS_TORCH and torch is not None and torch.cuda.is_available():
        if strict and cublas_config not in {":4096:8", ":16:8"}:
            risks.append(
                "CUDA is available but CUBLAS_WORKSPACE_CONFIG is not configured "
                "to a deterministic-compatible value."
            )

        if hasattr(torch.backends, "cudnn"):
            if strict and not torch.backends.cudnn.deterministic:
                risks.append(
                    "cuDNN deterministic mode is disabled."
                )

            if strict and torch.backends.cudnn.benchmark:
                risks.append(
                    "cuDNN benchmark mode is enabled; algorithm selection may "
                    "vary between executions."
                )

        if hasattr(torch.backends, "cuda"):
            matmul = getattr(torch.backends.cuda, "matmul", None)

            if (
                strict
                and matmul is not None
                and hasattr(matmul, "allow_tf32")
                and matmul.allow_tf32
            ):
                risks.append(
                    "CUDA TF32 matmul is enabled; reduced precision can affect "
                    "numerical reproducibility."
                )

    if HAS_TORCH and torch is not None:
        if hasattr(torch, "are_deterministic_algorithms_enabled"):
            if strict and not torch.are_deterministic_algorithms_enabled():
                risks.append(
                    "PyTorch deterministic algorithms are not enabled."
                )

    # Thread-level numerical libraries can introduce differences due to
    # reduction ordering.
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        if variable not in os.environ:
            risks.append(
                f"{variable} is not explicitly configured; numerical thread "
                "scheduling may differ across environments."
            )

    return risks


# ==============================================================================
# JSON-SAFE CONVERSION
# ==============================================================================


def _json_safe(value: Any) -> Any:
    """
    Convert common runtime objects into JSON-compatible structures.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass

    return str(value)


# ==============================================================================
# REPRODUCIBILITY FINGERPRINT
# ==============================================================================


def compute_reproducibility_fingerprint(
    *,
    seed: Optional[int] = None,
    strict: Optional[bool] = None,
) -> str:
    """
    Compute a SHA-256 fingerprint of reproducibility-relevant runtime metadata.

    The fingerprint is not a cryptographic signature of the model or dataset.
    It identifies the recorded execution environment.
    """

    effective_seed = (
        _ACTIVE_SEED
        if seed is None and _ACTIVE_SEED is not None
        else (
            DEFAULT_GLOBAL_SEED
            if seed is None
            else _validate_seed(seed)
        )
    )

    effective_strict = (
        _ACTIVE_STRICT_MODE
        if strict is None and _ACTIVE_STRICT_MODE is not None
        else bool(strict if strict is not None else True)
    )

    payload = {
        "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
        "seed": effective_seed,
        "strict": effective_strict,
        "runtime": asdict(get_runtime_fingerprint()),
        "environment": get_reproducibility_environment(),
        "packages": get_package_provenance(),
    }

    canonical = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ==============================================================================
# MANIFEST GENERATION
# ==============================================================================


def build_reproducibility_manifest(
    *,
    seed: Optional[int] = None,
    strict: Optional[bool] = None,
    include_risk_analysis: bool = True,
) -> Dict[str, Any]:
    """
    Build a complete JSON-safe reproducibility manifest.

    This should be persisted with production model/evaluation artifacts.
    """

    effective_seed = (
        _ACTIVE_SEED
        if seed is None and _ACTIVE_SEED is not None
        else (
            DEFAULT_GLOBAL_SEED
            if seed is None
            else _validate_seed(seed)
        )
    )

    effective_strict = (
        _ACTIVE_STRICT_MODE
        if strict is None and _ACTIVE_STRICT_MODE is not None
        else bool(strict if strict is not None else True)
    )

    risks = (
        detect_reproducibility_risks(strict=effective_strict)
        if include_risk_analysis
        else []
    )

    claim = (
        "strict_determinism_requested"
        if effective_strict and not risks
        else (
            "strict_determinism_with_runtime_warnings"
            if effective_strict
            else "best_effort"
        )
    )

    configuration = SeedConfiguration(
        global_seed=effective_seed,
        strict_determinism=effective_strict,
    )

    manifest = ReproducibilityManifest(
        schema_version=REPRODUCIBILITY_SCHEMA_VERSION,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        seed_configuration=asdict(configuration),
        runtime={
            "fingerprint": asdict(get_runtime_fingerprint()),
            "packages": get_package_provenance(),
        },
        environment=get_reproducibility_environment(),
        warnings=risks,
        reproducibility_claim=claim,
    )

    manifest.fingerprint = compute_reproducibility_fingerprint(
        seed=effective_seed,
        strict=effective_strict,
    )

    return _json_safe(asdict(manifest))


# ==============================================================================
# ENVIRONMENT SEED / DETERMINISM AUDIT
# ==============================================================================


def get_environment_seeds() -> Dict[str, Any]:
    """
    Return comprehensive reproducibility state for evaluation provenance.

    Backward-compatible replacement for the original lightweight function.
    """

    torch_info = _torch_runtime_info()

    result: Dict[str, Any] = {
        "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
        "active_seed": _ACTIVE_SEED,
        "active_strict_mode": _ACTIVE_STRICT_MODE,
        "context_depth": _CONTEXT_DEPTH,
        "python_hash_seed": os.environ.get(
            "PYTHONHASHSEED",
            "Not Set",
        ),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG",
            "Not Set",
        ),
        "has_pytorch": HAS_TORCH,
        "has_numpy": True,
        "numpy_version": np.__version__,
        "runtime_fingerprint": asdict(get_runtime_fingerprint()),
        "torch": torch_info,
        "package_provenance": get_package_provenance(),
        "reproducibility_risks": detect_reproducibility_risks(
            strict=bool(
                _ACTIVE_STRICT_MODE
                if _ACTIVE_STRICT_MODE is not None
                else True
            )
        ),
    }

    return _json_safe(result)


# ==============================================================================
# ASSERTION HELPERS
# ==============================================================================


def assert_reproducible_environment(
    *,
    strict: bool = True,
) -> None:
    """
    Validate that the runtime meets the configured reproducibility policy.

    In strict mode, any detected risk raises DeterminismConfigurationError.

    This function is intended for evaluation/training gates where silent
    degradation of reproducibility is unacceptable.
    """

    risks = detect_reproducibility_risks(strict=strict)

    if not risks:
        return

    message = (
        "Reproducibility environment validation failed:\n"
        + "\n".join(f" - {risk}" for risk in risks)
    )

    if strict:
        raise DeterminismConfigurationError(message)

    logger.warning(message)


# ==============================================================================
# REPRODUCIBILITY SELF-TEST
# ==============================================================================


def self_test(
    *,
    seed: int = DEFAULT_GLOBAL_SEED,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Run a lightweight reproducibility smoke test.

    The test verifies that repeated seeded operations produce identical
    values within the current runtime.

    This is NOT a complete guarantee of deterministic training.
    """

    seed = _validate_seed(seed)

    with DeterministicContext(seed=seed, strict=strict):
        python_values_1 = [random.random() for _ in range(5)]
        numpy_values_1 = np.random.random(5).tolist()

        torch_values_1: Optional[list[float]] = None

        if HAS_TORCH:
            assert torch is not None

            torch_values_1 = (
                torch.rand(5)
                .detach()
                .cpu()
                .tolist()
            )

    with DeterministicContext(seed=seed, strict=strict):
        python_values_2 = [random.random() for _ in range(5)]
        numpy_values_2 = np.random.random(5).tolist()

        torch_values_2: Optional[list[float]] = None

        if HAS_TORCH:
            assert torch is not None

            torch_values_2 = (
                torch.rand(5)
                .detach()
                .cpu()
                .tolist()
            )

    python_match = python_values_1 == python_values_2
    numpy_match = bool(
        np.array_equal(
            np.asarray(numpy_values_1),
            np.asarray(numpy_values_2),
        )
    )

    torch_match = True

    if HAS_TORCH:
        torch_match = torch_values_1 == torch_values_2

    passed = python_match and numpy_match and torch_match

    result = {
        "passed": passed,
        "seed": seed,
        "strict": strict,
        "python_rng_match": python_match,
        "numpy_rng_match": numpy_match,
        "torch_rng_match": torch_match,
        "fingerprint": compute_reproducibility_fingerprint(
            seed=seed,
            strict=strict,
        ),
    }

    if not passed:
        logger.error(
            "Reproducibility self-test FAILED for seed=%d.",
            seed,
        )
    else:
        logger.info(
            "Reproducibility self-test passed for seed=%d.",
            seed,
        )

    return result


# ==============================================================================
# CONTEXTLIB-COMPATIBLE HELPER
# ==============================================================================


@contextlib.contextmanager
def deterministic(
    seed: int = DEFAULT_GLOBAL_SEED,
    *,
    strict: bool = True,
) -> Generator[DeterministicContext, None, None]:
    """
    Convenience wrapper around DeterministicContext.

    Example
    -------
    with deterministic(42):
        run_evaluation()
    """

    context = DeterministicContext(seed=seed, strict=strict)

    with context:
        yield context


# ==============================================================================
# PUBLIC EXPORT CONTRACT
# ==============================================================================

__all__ = [
    # Constants
    "DEFAULT_GLOBAL_SEED",
    "MIN_SEED",
    "MAX_SEED",
    "REPRODUCIBILITY_SCHEMA_VERSION",

    # Exceptions
    "ReproducibilityError",
    "DeterminismConfigurationError",
    "InvalidSeedError",

    # Data structures
    "RuntimeFingerprint",
    "SeedConfiguration",
    "RNGSnapshot",
    "ReproducibilityManifest",

    # Seed management
    "set_seed",
    "derive_seed",
    "create_numpy_generator",
    "create_torch_generator",
    "seed_worker",

    # RNG state
    "capture_rng_state",
    "restore_rng_state",

    # Context management
    "DeterministicContext",
    "deterministic",

    # Diagnostics
    "get_environment_seeds",
    "get_reproducibility_environment",
    "get_runtime_fingerprint",
    "get_package_provenance",
    "detect_reproducibility_risks",
    "assert_reproducible_environment",

    # Provenance
    "compute_reproducibility_fingerprint",
    "build_reproducibility_manifest",

    # Verification
    "self_test",

    # Capability
    "HAS_TORCH",
    "HAS_SKLEARN",
]
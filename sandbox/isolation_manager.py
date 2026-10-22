"""
PhishGuard_AI - Sandbox Isolation Manager

Coordinates isolated analysis environments with validated, bounded lifecycle.

Security / reliability principles:
- Environment names and identifiers are strictly validated to prevent injection.
- Resource limits (CPU, memory, timeout) are bounded by hard caps.
- Environments are created/terminated via a pluggable backend; the default
  backend is a safe no-op that only manages metadata (no container spawn).
- Idle environments are reaped to bound resource usage.
"""

from __future__ import annotations

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Mapping, Optional, Sequence

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

MAX_ENV_NAME_LENGTH = 128
ENV_ID_PREFIX = "sandbox_"
MAX_ENVIRONMENTS = 64
DEFAULT_CPU_LIMIT = 2
DEFAULT_MEMORY_LIMIT_MB = 1024
DEFAULT_TIMEOUT_SECONDS = 300
MAX_TIMEOUT_SECONDS = 3600
MAX_MEMORY_MB = 8192
MAX_CPU = 32
DEFAULT_IDLE_REAP_SECONDS = 600

_ENV_NAME_REGEX = __import__("re").compile(r"^[A-Za-z0-9_\-]{1,128}$")


class SandboxStatus(str, enum.Enum):
    """Lifecycle of an isolation environment."""

    PROVISIONING = "provisioning"
    READY = "ready"
    BUSY = "busy"
    TERMINATED = "terminated"
    FAILED = "failed"


class IsolationError(RuntimeError):
    """Base error for isolation management."""


class IsolationLimitError(IsolationError):
    """An environment limit was exceeded."""


class InvalidEnvironmentNameError(IsolationError):
    """Environment name is invalid."""


@dataclass(frozen=True, slots=True)
class SandboxEnvironment:
    """Metadata describing an isolated analysis environment."""

    env_id: str
    name: str
    status: SandboxStatus
    created_utc: str
    cpu_limit: int
    memory_limit_mb: int
    timeout_seconds: int
    labels: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "env_id": self.env_id,
            "name": self.name,
            "status": self.status.value,
            "created_utc": self.created_utc,
            "cpu_limit": self.cpu_limit,
            "memory_limit_mb": self.memory_limit_mb,
            "timeout_seconds": self.timeout_seconds,
            "labels": dict(self.labels),
        }


# ============================================================================
# MANAGER
# # ============================================================================


class IsolationManager:
    """
    Thread-safe isolation environment manager.

    ``provision_backend`` and ``terminate_backend`` are pluggable callables. By
    default they are no-ops, so the manager is safe in any environment and can
    be wired to a real container backend later.
    """

    def __init__(
        self,
        *,
        provision_backend: Optional[Callable[[SandboxEnvironment], bool]] = None,
        terminate_backend: Optional[Callable[[str], bool]] = None,
        idle_reap_seconds: int = DEFAULT_IDLE_REAP_SECONDS,
    ) -> None:
        self._provision_backend = provision_backend or (lambda env: True)
        self._terminate_backend = terminate_backend or (lambda env_id: True)
        self._idle_reap_seconds = idle_reap_seconds
        self._lock = threading.RLock()
        self._environments: Dict[str, SandboxEnvironment] = {}
        self._last_activity: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def provision(
        self,
        *,
        name: str,
        cpu_limit: int = DEFAULT_CPU_LIMIT,
        memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        labels: Optional[Mapping[str, str]] = None,
    ) -> SandboxEnvironment:
        """Create an isolated analysis environment within configured limits."""
        if not isinstance(name, str) or not _ENV_NAME_REGEX.match(name):
            raise InvalidEnvironmentNameError(f"Invalid env name: {name!r}")

        cpu_limit = self._bound_int(cpu_limit, 1, MAX_CPU, "cpu_limit")
        memory_limit_mb = self._bound_int(
            memory_limit_mb, 64, MAX_MEMORY_MB, "memory_limit_mb"
        )
        timeout_seconds = self._bound_int(
            timeout_seconds, 1, MAX_TIMEOUT_SECONDS, "timeout_seconds"
        )

        with self._lock:
            if len(self._environments) >= MAX_ENVIRONMENTS:
                self._reap_idle()
            if len(self._environments) >= MAX_ENVIRONMENTS:
                raise IsolationLimitError(
                    f"Cannot exceed {MAX_ENVIRONMENTS} active environments."
                )

            env = SandboxEnvironment(
                env_id=ENV_ID_PREFIX + uuid.uuid4().hex[:20],
                name=name,
                status=SandboxStatus.PROVISIONING,
                created_utc=self._now(),
                cpu_limit=cpu_limit,
                memory_limit_mb=memory_limit_mb,
                timeout_seconds=timeout_seconds,
                labels=self._sanitize_labels(labels),
            )

        try:
            ok = self._provision_backend(env)
        except Exception:  # noqa: BLE001
            ok = False

        env = SandboxEnvironment(
            **{
                **env.to_dict(),
                "status": SandboxStatus.READY if ok else SandboxStatus.FAILED,
            }
        )
        with self._lock:
            self._environments[env.env_id] = env
            self._last_activity[env.env_id] = time.time()
        return env

    def mark_busy(self, env_id: str) -> SandboxEnvironment:
        env = self.get(env_id)
        if env.status is not SandboxStatus.READY:
            raise IsolationError(f"Environment is {env.status.value}, not ready.")
        return self._update_status(env, SandboxStatus.BUSY)

    def mark_ready(self, env_id: str) -> SandboxEnvironment:
        env = self.get(env_id)
        if env.status is not SandboxStatus.BUSY:
            raise IsolationError(f"Environment is {env.status.value}, not busy.")
        return self._update_status(env, SandboxStatus.READY)

    def get(self, env_id: str) -> SandboxEnvironment:
        with self._lock:
            env = self._environments.get(env_id)
        if env is None:
            raise IsolationError(f"Unknown environment: {env_id}")
        return env

    def terminate(self, env_id: str) -> bool:
        env = self.get(env_id)
        try:
            ok = self._terminate_backend(env_id)
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            self._update_status(env, SandboxStatus.TERMINATED)
        return ok

    def list_environments(
        self, *, status: Optional[SandboxStatus] = None
    ) -> List[SandboxEnvironment]:
        with self._lock:
            envs = [
                e
                for e in self._environments.values()
                if status is None or e.status == status
            ]
        envs.sort(key=lambda e: e.created_utc)
        return envs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_status(
        self, env: SandboxEnvironment, status: SandboxStatus
    ) -> SandboxEnvironment:
        updated = SandboxEnvironment(
            **{**env.to_dict(), "status": status}
        )
        with self._lock:
            self._environments[env.env_id] = updated
            self._last_activity[env.env_id] = time.time()
        return updated

    def _reap_idle(self) -> None:
        now = time.time()
        idle_ids = [
            env_id
            for env_id, ts in self._last_activity.items()
            if now - ts > self._idle_reap_seconds
            and self._environments.get(env_id)
            and self._environments[env_id].status
            in (SandboxStatus.READY, SandboxStatus.TERMINATED)
        ]
        for env_id in idle_ids:
            self._environments.pop(env_id, None)
            self._last_activity.pop(env_id, None)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _bound_int(value: int, low: int, high: int, name: str) -> int:
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise IsolationError(f"{name} must be an integer.") from None
        return min(high, max(low, value))

    @staticmethod
    def _sanitize_labels(
        labels: Optional[Mapping[str, str]]
    ) -> Mapping[str, str]:
        if not labels:
            return {}
        result: Dict[str, str] = {}
        for key, value in labels.items():
            if len(result) >= 16:
                break
            result[str(key)[:64]] = str(value)[:256]
        return result


# ============================================================================
# SINGLETON
# ============================================================================

isolation_manager = IsolationManager()


__all__ = [
    "SandboxStatus",
    "SandboxEnvironment",
    "IsolationManager",
    "IsolationError",
    "IsolationLimitError",
    "InvalidEnvironmentNameError",
    "isolation_manager",
    "MAX_ENVIRONMENTS",
]

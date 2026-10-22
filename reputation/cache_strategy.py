"""
PhishGuard AI - Production Cache Strategy
=========================================

Version: 17.0.0
File Path: reputation/cache_strategy.py

Purpose
-------
Shared production-grade cache infrastructure for:

    reputation/domain_reputation.py
    reputation/ip_reputation.py
    reputation/hash_reputation.py

This module is deliberately independent from provider-specific threat
intelligence implementations.

Architecture
------------

                    ┌──────────────────────────┐
                    │     CacheStrategy API    │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴─────────────┐
                    │                          │
                    ▼                          ▼
              L1 Memory Cache             Optional L2
              bounded / local             Redis-like backend
                    │                          │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴─────────────┐
                    │                          │
                    ▼                          ▼
              Generation Fence           Circuit Breaker
                    │
                    ▼
              Request Coalescing
                    │
                    ▼
              Stale-While-Revalidate
                    │
                    ▼
              Loader / Provider

Security / Reliability Properties
----------------------------------
- bounded memory usage
- bounded key size
- bounded namespace size
- opaque backend keys
- configuration fingerprints
- generation fencing
- stale-while-revalidate
- stale-if-error
- negative/unknown caching
- request coalescing
- bounded in-flight work
- bounded refresh concurrency
- loader timeout
- cancellation-safe cleanup
- graceful shutdown
- backend circuit breaker
- L2 failure isolation
- LRU-ish eviction
- deterministic TTL jitter
- defensive cloning
- exception/non-cacheable value rejection
- conditional writes
- namespace invalidation
- tenant-aware namespaces
- metrics
- no provider-specific assumptions

Python
------
Designed for Python 3.11+ and Pydantic v2-compatible projects.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import logging
import math
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    BaseModel = Any  # type: ignore


logger = logging.getLogger("phishguard.reputation.cache")


# =============================================================================
# VERSION
# =============================================================================

MODULE_VERSION = "17.0.0"


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_FRESH_TTL_SECONDS = 900
DEFAULT_STALE_TTL_SECONDS = 300
DEFAULT_NEGATIVE_TTL_SECONDS = 120

DEFAULT_MAX_ENTRIES = 10_000
DEFAULT_MAX_WEIGHT_BYTES = 128 * 1024 * 1024

DEFAULT_MAX_KEY_LENGTH = 512
DEFAULT_MAX_NAMESPACE_LENGTH = 128

DEFAULT_MAX_VALUE_DEPTH = 12
DEFAULT_MAX_SIZE_OBJECTS = 10_000

DEFAULT_MAX_INFLIGHT = 1_024
DEFAULT_MAX_REFRESH_TASKS = 64

DEFAULT_LOADER_TIMEOUT_SECONDS = 10.0
DEFAULT_LOCK_TIMEOUT_SECONDS = 2.0

DEFAULT_BACKEND_TIMEOUT_SECONDS = 2.0

DEFAULT_BACKEND_FAILURE_THRESHOLD = 5
DEFAULT_BACKEND_RECOVERY_SECONDS = 30.0

DEFAULT_TTL_JITTER_RATIO = 0.05

DEFAULT_MAX_STALE_AGE_SECONDS = 900

DEFAULT_CLONE_VALUES = True

DEFAULT_NAMESPACE = "reputation"

# Hard protection against accidental pathological configuration.
MAX_ALLOWED_TTL_SECONDS = 7 * 24 * 60 * 60
MAX_ALLOWED_ENTRIES = 1_000_000
MAX_ALLOWED_WEIGHT_BYTES = 2 * 1024 * 1024 * 1024


# =============================================================================
# ENUMS
# =============================================================================

class CacheEntryState(str, Enum):
    """Lifecycle state of a cache entry."""

    FRESH = "fresh"
    STALE = "stale"
    EXPIRED = "expired"
    EVICTED = "evicted"


class CacheBackendMode(str, Enum):
    """L2 backend operational mode."""

    DISABLED = "disabled"
    HEALTHY = "healthy"
    OPEN = "open"
    RECOVERING = "recovering"


class CacheResultState(str, Enum):
    """Result state returned by cache operations."""

    HIT = "hit"
    STALE_HIT = "stale_hit"
    MISS = "miss"
    LOADED = "loaded"
    NEGATIVE_HIT = "negative_hit"


# =============================================================================
# EXCEPTIONS
# =============================================================================

class CacheError(Exception):
    """Base cache exception."""


class CacheConfigurationError(CacheError):
    """Invalid cache configuration."""


class CacheKeyError(CacheError):
    """Invalid cache key."""


class CacheValueError(CacheError):
    """Value cannot safely be cached."""


class CacheBackendError(CacheError):
    """Backend operation failed."""


class CacheBackendUnavailable(CacheBackendError):
    """Backend circuit is open."""


class CacheLoaderTimeout(CacheError):
    """Loader exceeded configured timeout."""


class CacheShutdownError(CacheError):
    """Cache is shutting down or already closed."""


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

T = TypeVar("T")

Loader = Callable[[], Awaitable[T]]


# =============================================================================
# CACHE KEY
# =============================================================================

@dataclass(frozen=True, slots=True)
class CacheKey:
    """
    Structured cache key.

    The backend never receives the raw subject. It receives a digest derived
    from namespace + subject + configuration fingerprint.
    """

    namespace: str
    subject: str
    config_fingerprint: str = ""

    def __post_init__(self) -> None:
        namespace = self.namespace.strip()

        if not namespace:
            raise CacheKeyError("Cache namespace cannot be empty.")

        if len(namespace) > DEFAULT_MAX_NAMESPACE_LENGTH:
            raise CacheKeyError("Cache namespace is too long.")

        if len(self.subject) > DEFAULT_MAX_KEY_LENGTH:
            raise CacheKeyError("Cache subject is too long.")

        if not self.subject:
            raise CacheKeyError("Cache subject cannot be empty.")

        if len(self.config_fingerprint) > 128:
            raise CacheKeyError("Configuration fingerprint is too long.")

    @property
    def canonical(self) -> str:
        return (
            f"{self.namespace}:"
            f"{self.config_fingerprint}:"
            f"{self.subject}"
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            self.canonical.encode("utf-8")
        ).hexdigest()

    @property
    def backend_key(self) -> str:
        """
        Opaque backend key.

        Namespace is intentionally hashed as well so multi-tenant namespace
        names are not directly exposed in Redis-like keyspaces.
        """

        namespace_digest = hashlib.sha256(
            self.namespace.encode("utf-8")
        ).hexdigest()[:16]

        return (
            f"phishguard:cache:v{MODULE_VERSION}:"
            f"{namespace_digest}:{self.digest}"
        )


# =============================================================================
# CONFIGURATION FINGERPRINT
# =============================================================================

def _normalize_for_fingerprint(
    value: Any,
    *,
    depth: int = 0,
) -> Any:
    """
    Convert common configuration values into deterministic primitives.

    This function deliberately rejects unknown objects instead of relying on
    repr(), because repr() is often non-deterministic and may include memory
    addresses or secrets.
    """

    if depth > 8:
        raise CacheConfigurationError(
            "Configuration nesting exceeds maximum depth."
        )

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float):
            if not math.isfinite(value):
                raise CacheConfigurationError(
                    "Non-finite float in cache configuration."
                )

        return value

    if isinstance(value, Enum):
        return _normalize_for_fingerprint(
            value.value,
            depth=depth + 1,
        )

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (bytes, bytearray)):
        return hashlib.sha256(bytes(value)).hexdigest()

    if isinstance(value, Mapping):
        normalized = {}

        for key in sorted(
            value.keys(),
            key=lambda item: str(item),
        ):
            normalized[str(key)] = _normalize_for_fingerprint(
                value[key],
                depth=depth + 1,
            )

        return normalized

    if isinstance(value, (list, tuple)):
        return [
            _normalize_for_fingerprint(
                item,
                depth=depth + 1,
            )
            for item in value
        ]

    if isinstance(value, (set, frozenset)):
        normalized = [
            _normalize_for_fingerprint(
                item,
                depth=depth + 1,
            )
            for item in value
        ]

        return sorted(
            normalized,
            key=lambda item: repr(item),
        )

    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(
                mode="json",
                exclude_none=False,
            )

            return _normalize_for_fingerprint(
                dumped,
                depth=depth + 1,
            )
        except Exception as exc:
            raise CacheConfigurationError(
                "Unable to normalize configuration object."
            ) from exc

    raise CacheConfigurationError(
        f"Unsupported configuration type: {type(value).__name__}"
    )


def configuration_fingerprint(
    configuration: Mapping[str, Any],
) -> str:
    """
    Produce a deterministic SHA-256 configuration fingerprint.
    """

    import json

    normalized = _normalize_for_fingerprint(configuration)

    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()[:32]


# =============================================================================
# TTL POLICY
# =============================================================================

@dataclass(frozen=True, slots=True)
class CacheTTLPolicy:
    """Fresh/stale/negative cache lifetime policy."""

    fresh_ttl_seconds: float = DEFAULT_FRESH_TTL_SECONDS
    stale_ttl_seconds: float = DEFAULT_STALE_TTL_SECONDS
    negative_ttl_seconds: float = DEFAULT_NEGATIVE_TTL_SECONDS

    jitter_ratio: float = DEFAULT_TTL_JITTER_RATIO

    def __post_init__(self) -> None:
        for name, value in (
            ("fresh_ttl_seconds", self.fresh_ttl_seconds),
            ("stale_ttl_seconds", self.stale_ttl_seconds),
            ("negative_ttl_seconds", self.negative_ttl_seconds),
        ):
            if not math.isfinite(value):
                raise CacheConfigurationError(
                    f"{name} must be finite."
                )

            if value < 0:
                raise CacheConfigurationError(
                    f"{name} must be >= 0."
                )

            if value > MAX_ALLOWED_TTL_SECONDS:
                raise CacheConfigurationError(
                    f"{name} exceeds safety limit."
                )

        if not 0 <= self.jitter_ratio <= 0.50:
            raise CacheConfigurationError(
                "jitter_ratio must be between 0 and 0.50."
            )

        if (
            self.stale_ttl_seconds > 0
            and self.fresh_ttl_seconds <= 0
        ):
            raise CacheConfigurationError(
                "stale TTL requires a positive fresh TTL."
            )

    def ttl(
        self,
        *,
        negative: bool,
    ) -> float:
        return (
            self.negative_ttl_seconds
            if negative
            else self.fresh_ttl_seconds
        )


# =============================================================================
# CACHE ENTRY
# =============================================================================

@dataclass(slots=True)
class CacheEntry(Generic[T]):
    """
    Internal cache entry.

    monotonic timestamps are used for expiry calculations because wall-clock
    time can move backwards/forwards.
    """

    key: CacheKey
    value: T

    created_at: float
    fresh_until: float
    stale_until: float

    weight_bytes: int

    negative: bool = False

    generation: int = 0

    last_access: float = field(
        default_factory=time.monotonic
    )

    hits: int = 0

    def state(
        self,
        now: Optional[float] = None,
    ) -> CacheEntryState:
        now = (
            time.monotonic()
            if now is None
            else now
        )

        if now < self.fresh_until:
            return CacheEntryState.FRESH

        if now < self.stale_until:
            return CacheEntryState.STALE

        return CacheEntryState.EXPIRED


# =============================================================================
# METRICS
# =============================================================================

@dataclass(slots=True)
class CacheMetrics:
    """Process-local cache metrics."""

    gets: int = 0
    hits: int = 0
    stale_hits: int = 0
    negative_hits: int = 0
    misses: int = 0

    sets: int = 0
    replacements: int = 0
    evictions: int = 0
    expirations: int = 0

    loader_calls: int = 0
    loader_failures: int = 0
    loader_timeouts: int = 0

    coalesced_waiters: int = 0
    coalescing_rejections: int = 0

    refresh_scheduled: int = 0
    refresh_completed: int = 0
    refresh_failed: int = 0
    refresh_rejected: int = 0

    backend_gets: int = 0
    backend_hits: int = 0
    backend_misses: int = 0
    backend_sets: int = 0
    backend_deletes: int = 0
    backend_failures: int = 0
    backend_timeouts: int = 0

    circuit_opens: int = 0
    circuit_recoveries: int = 0

    generation_bumps: int = 0
    namespace_clears: int = 0

    clone_failures: int = 0
    rejected_values: int = 0

    total_load_latency_ms: float = 0.0
    total_backend_latency_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.gets <= 0:
            return 0.0

        return self.hits / self.gets

    @property
    def stale_hit_rate(self) -> float:
        if self.gets <= 0:
            return 0.0

        return self.stale_hits / self.gets

    @property
    def average_load_latency_ms(self) -> float:
        if self.loader_calls <= 0:
            return 0.0

        return self.total_load_latency_ms / self.loader_calls

    @property
    def average_backend_latency_ms(self) -> float:
        if self.backend_gets + self.backend_sets <= 0:
            return 0.0

        return self.total_backend_latency_ms / (
            self.backend_gets + self.backend_sets
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "gets": self.gets,
            "hits": self.hits,
            "stale_hits": self.stale_hits,
            "negative_hits": self.negative_hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "stale_hit_rate": self.stale_hit_rate,
            "sets": self.sets,
            "replacements": self.replacements,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "loader_calls": self.loader_calls,
            "loader_failures": self.loader_failures,
            "loader_timeouts": self.loader_timeouts,
            "coalesced_waiters": self.coalesced_waiters,
            "coalescing_rejections": self.coalescing_rejections,
            "refresh_scheduled": self.refresh_scheduled,
            "refresh_completed": self.refresh_completed,
            "refresh_failed": self.refresh_failed,
            "refresh_rejected": self.refresh_rejected,
            "backend_gets": self.backend_gets,
            "backend_hits": self.backend_hits,
            "backend_misses": self.backend_misses,
            "backend_sets": self.backend_sets,
            "backend_deletes": self.backend_deletes,
            "backend_failures": self.backend_failures,
            "backend_timeouts": self.backend_timeouts,
            "circuit_opens": self.circuit_opens,
            "circuit_recoveries": self.circuit_recoveries,
            "generation_bumps": self.generation_bumps,
            "namespace_clears": self.namespace_clears,
            "clone_failures": self.clone_failures,
            "rejected_values": self.rejected_values,
            "average_load_latency_ms": self.average_load_latency_ms,
            "average_backend_latency_ms": self.average_backend_latency_ms,
        }


# =============================================================================
# BACKEND PROTOCOL
# =============================================================================

@dataclass(frozen=True, slots=True)
class BackendValue:
    """L2 value plus backend-controlled expiration."""

    value: Any
    expires_at: Optional[float] = None
    generation: int = 0


class CacheBackend(Protocol):
    """
    Minimal async backend contract.

    Redis, Valkey, Memcached adapters, database caches, etc. can implement
    this protocol.
    """

    async def get(
        self,
        key: str,
    ) -> Optional[BackendValue]:
        ...

    async def set(
        self,
        key: str,
        value: BackendValue,
        ttl_seconds: float,
    ) -> None:
        ...

    async def delete(
        self,
        key: str,
    ) -> None:
        ...

    async def clear_namespace(
        self,
        namespace_digest: str,
    ) -> None:
        ...


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class BackendCircuitBreaker:
    """
    Small async-safe circuit breaker.

    OPEN:
        backend calls are skipped.

    HALF-OPEN:
        one probe is allowed after recovery delay.

    CLOSED:
        normal operation.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = DEFAULT_BACKEND_FAILURE_THRESHOLD,
        recovery_seconds: float = DEFAULT_BACKEND_RECOVERY_SECONDS,
    ) -> None:
        if failure_threshold < 1:
            raise CacheConfigurationError(
                "failure_threshold must be >= 1."
            )

        if recovery_seconds <= 0:
            raise CacheConfigurationError(
                "recovery_seconds must be > 0."
            )

        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds

        self._failures = 0
        self._opened_at: Optional[float] = None
        self._probe_in_progress = False

        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self._lock:
            if self._opened_at is None:
                return True

            now = time.monotonic()

            if (
                now - self._opened_at
                < self.recovery_seconds
            ):
                return False

            if self._probe_in_progress:
                return False

            self._probe_in_progress = True
            return True

    async def success(self) -> bool:
        async with self._lock:
            recovered = self._opened_at is not None

            self._failures = 0
            self._opened_at = None
            self._probe_in_progress = False

            return recovered

    async def failure(self) -> bool:
        async with self._lock:
            self._probe_in_progress = False
            self._failures += 1

            if self._failures >= self.failure_threshold:
                if self._opened_at is None:
                    self._opened_at = time.monotonic()
                    return True

            return False

    async def state(self) -> CacheBackendMode:
        async with self._lock:
            if self._opened_at is None:
                return CacheBackendMode.HEALTHY

            if (
                time.monotonic() - self._opened_at
                >= self.recovery_seconds
            ):
                return CacheBackendMode.RECOVERING

            return CacheBackendMode.OPEN


# =============================================================================
# SAFE SIZE ESTIMATION
# =============================================================================

def estimate_size(
    value: Any,
    *,
    max_depth: int = DEFAULT_MAX_VALUE_DEPTH,
    max_objects: int = DEFAULT_MAX_SIZE_OBJECTS,
) -> int:
    """
    Bounded recursive memory-size estimate.

    Important:
        This is a cache-budget estimate, not a VM-level exact memory
        measurement.

    Cycles are handled through object identity tracking.
    """

    seen: Set[int] = set()
    objects_seen = 0

    def walk(
        obj: Any,
        depth: int,
    ) -> int:
        nonlocal objects_seen

        if objects_seen >= max_objects:
            return 0

        objects_seen += 1

        if depth > max_depth:
            try:
                return sys.getsizeof(obj)
            except Exception:
                return 64

        object_id = id(obj)

        if object_id in seen:
            return 0

        seen.add(object_id)

        try:
            size = sys.getsizeof(obj)
        except Exception:
            size = 64

        if obj is None:
            return size

        if isinstance(
            obj,
            (str, bytes, bytearray),
        ):
            return size

        if isinstance(obj, Mapping):
            for key, item in obj.items():
                size += walk(key, depth + 1)
                size += walk(item, depth + 1)

            return size

        if isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                size += walk(item, depth + 1)

            return size

        if hasattr(obj, "model_dump"):
            try:
                dumped = obj.model_dump(
                    mode="python",
                    exclude_none=False,
                )

                return size + walk(
                    dumped,
                    depth + 1,
                )
            except Exception:
                return size

        if hasattr(obj, "__dict__"):
            try:
                size += walk(
                    vars(obj),
                    depth + 1,
                )
            except Exception:
                pass

        if hasattr(obj, "__slots__"):
            for slot in obj.__slots__:
                try:
                    slot_value = getattr(obj, slot)
                except Exception:
                    continue

                size += walk(
                    slot_value,
                    depth + 1,
                )

        return size

    return max(1, walk(value, 0))


# =============================================================================
# VALUE VALIDATION
# =============================================================================

def is_cacheable_value(value: Any) -> bool:
    """
    Reject values that are unsafe or meaningless to cache.
    """

    if value is None:
        return False

    if isinstance(value, BaseException):
        return False

    if inspect.iscoroutine(value):
        return False

    if inspect.isasyncgen(value):
        return False

    if inspect.isgenerator(value):
        return False

    return True


def _is_probably_immutable(value: Any) -> bool:
    """
    Detect common immutable/frozen objects.

    Frozen Pydantic models are safe to share if callers respect immutability.
    """

    if isinstance(
        value,
        (
            str,
            bytes,
            int,
            float,
            bool,
            frozenset,
            tuple,
            type(None),
        ),
    ):
        return True

    config = getattr(
        value,
        "model_config",
        None,
    )

    if config:
        try:
            return bool(
                config.get("frozen", False)
            )
        except Exception:
            return False

    return False


def clone_value(
    value: T,
    *,
    clone: bool,
    metrics: Optional[CacheMetrics] = None,
) -> T:
    """
    Defensive value cloning.

    Immutable values are returned directly.
    """

    if not clone or _is_probably_immutable(value):
        return value

    if hasattr(value, "model_copy"):
        try:
            return value.model_copy(
                deep=True
            )
        except Exception:
            pass

    try:
        copied = copy.deepcopy(value)

        if copied is value:
            raise CacheValueError(
                "Deep copy returned the original object."
            )

        return copied

    except Exception as exc:
        if metrics:
            metrics.clone_failures += 1

        raise CacheValueError(
            f"Value could not be safely cloned: "
            f"{type(value).__name__}"
        ) from exc


# =============================================================================
# L1 CACHE
# =============================================================================

class L1MemoryCache:
    """
    Bounded process-local L1 cache.

    Uses OrderedDict for an efficient LRU-ish policy while maintaining
    deterministic bounded memory.
    """

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_weight_bytes: int = DEFAULT_MAX_WEIGHT_BYTES,
        clone_values: bool = DEFAULT_CLONE_VALUES,
        max_value_depth: int = DEFAULT_MAX_VALUE_DEPTH,
    ) -> None:
        if not (
            1 <= max_entries <= MAX_ALLOWED_ENTRIES
        ):
            raise CacheConfigurationError(
                "Invalid max_entries."
            )

        if not (
            1
            <= max_weight_bytes
            <= MAX_ALLOWED_WEIGHT_BYTES
        ):
            raise CacheConfigurationError(
                "Invalid max_weight_bytes."
            )

        self.max_entries = max_entries
        self.max_weight_bytes = max_weight_bytes
        self.clone_values = clone_values
        self.max_value_depth = max_value_depth

        self._entries: "OrderedDict[str, CacheEntry[Any]]" = (
            OrderedDict()
        )

        self._weight_bytes = 0

        self._lock = asyncio.Lock()

        self.metrics = CacheMetrics()

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def weight_bytes(self) -> int:
        return self._weight_bytes

    async def get(
        self,
        key: CacheKey,
        *,
        generation: int,
        allow_stale: bool = False,
    ) -> Optional[Tuple[Any, CacheEntryState]]:
        now = time.monotonic()

        async with self._lock:
            self.metrics.gets += 1

            entry = self._entries.get(key.digest)

            if entry is None:
                self.metrics.misses += 1
                return None

            if entry.generation != generation:
                self._remove_locked(
                    key.digest
                )

                self.metrics.misses += 1
                return None

            state = entry.state(now)

            if state == CacheEntryState.EXPIRED:
                self._remove_locked(
                    key.digest
                )

                self.metrics.expirations += 1
                self.metrics.misses += 1

                return None

            if (
                state == CacheEntryState.STALE
                and not allow_stale
            ):
                self.metrics.misses += 1
                return None

            entry.last_access = now
            entry.hits += 1

            self._entries.move_to_end(
                key.digest
            )

            if state == CacheEntryState.STALE:
                self.metrics.stale_hits += 1
            else:
                self.metrics.hits += 1

            if entry.negative:
                self.metrics.negative_hits += 1

            value = clone_value(
                entry.value,
                clone=self.clone_values,
                metrics=self.metrics,
            )

            return value, state

    async def set(
        self,
        key: CacheKey,
        value: Any,
        *,
        ttl_policy: CacheTTLPolicy,
        generation: int,
        negative: bool,
        ttl_override_seconds: Optional[float] = None,
    ) -> bool:
        if not is_cacheable_value(value):
            self.metrics.rejected_values += 1
            return False

        weight = estimate_size(
            value,
            max_depth=self.max_value_depth,
        )

        if weight > self.max_weight_bytes:
            self.metrics.rejected_values += 1
            return False

        now = time.monotonic()

        ttl = (
            ttl_override_seconds
            if ttl_override_seconds is not None
            else ttl_policy.ttl(
                negative=negative
            )
        )

        ttl = max(0.0, ttl)

        ttl = self._jittered_ttl(
            key,
            ttl,
            ttl_policy.jitter_ratio,
        )

        stale_ttl = ttl_policy.stale_ttl_seconds

        stored_value = clone_value(
            value,
            clone=self.clone_values,
            metrics=self.metrics,
        )

        entry = CacheEntry(
            key=key,
            value=stored_value,
            created_at=now,
            fresh_until=now + ttl,
            stale_until=now + ttl + stale_ttl,
            weight_bytes=weight,
            negative=negative,
            generation=generation,
            last_access=now,
        )

        async with self._lock:
            previous = self._entries.pop(
                key.digest,
                None,
            )

            if previous:
                self._weight_bytes -= (
                    previous.weight_bytes
                )

                self.metrics.replacements += 1

            self._entries[key.digest] = entry
            self._weight_bytes += weight

            self._evict_locked()

            self.metrics.sets += 1

            return (
                key.digest in self._entries
            )

    async def delete(
        self,
        key: CacheKey,
    ) -> bool:
        async with self._lock:
            return self._remove_locked(
                key.digest
            )

    async def clear_namespace(
        self,
        namespace: str,
    ) -> int:
        async with self._lock:
            digests = [
                digest
                for digest, entry in self._entries.items()
                if entry.key.namespace == namespace
            ]

            for digest in digests:
                self._remove_locked(digest)

            return len(digests)

    async def clear(
        self,
    ) -> None:
        async with self._lock:
            self._entries.clear()
            self._weight_bytes = 0

    async def keys(
        self,
    ) -> Sequence[str]:
        async with self._lock:
            return tuple(
                entry.key.digest
                for entry in self._entries.values()
            )

    def _remove_locked(
        self,
        digest: str,
    ) -> bool:
        entry = self._entries.pop(
            digest,
            None,
        )

        if entry is None:
            return False

        self._weight_bytes = max(
            0,
            self._weight_bytes
            - entry.weight_bytes,
        )

        return True

    def _evict_locked(
        self,
    ) -> None:
        while (
            len(self._entries)
            > self.max_entries
            or self._weight_bytes
            > self.max_weight_bytes
        ):
            digest, entry = (
                self._entries.popitem(
                    last=False
                )
            )

            self._weight_bytes = max(
                0,
                self._weight_bytes
                - entry.weight_bytes,
            )

            self.metrics.evictions += 1

    @staticmethod
    def _jittered_ttl(
        key: CacheKey,
        ttl: float,
        ratio: float,
    ) -> float:
        if ttl <= 0 or ratio <= 0:
            return ttl

        digest_int = int(
            key.digest[:16],
            16,
        )

        rng = random.Random(
            digest_int
        )

        delta = ttl * ratio

        return max(
            0.001,
            ttl + rng.uniform(
                -delta,
                delta,
            ),
        )


# =============================================================================
# CACHE STRATEGY
# =============================================================================

class CacheStrategy:
    """
    Shared L1/L2 production cache.

    This is the primary class reputation engines should use.
    """

    def __init__(
        self,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        configuration: Optional[
            Mapping[str, Any]
        ] = None,
        ttl_policy: Optional[
            CacheTTLPolicy
        ] = None,
        local: Optional[L1MemoryCache] = None,
        backend: Optional[CacheBackend] = None,
        loader_timeout_seconds: float = (
            DEFAULT_LOADER_TIMEOUT_SECONDS
        ),
        backend_timeout_seconds: float = (
            DEFAULT_BACKEND_TIMEOUT_SECONDS
        ),
        lock_timeout_seconds: float = (
            DEFAULT_LOCK_TIMEOUT_SECONDS
        ),
        max_inflight: int = DEFAULT_MAX_INFLIGHT,
        max_refresh_tasks: int = DEFAULT_MAX_REFRESH_TASKS,
        clone_values: bool = DEFAULT_CLONE_VALUES,
        backend_failure_threshold: int = (
            DEFAULT_BACKEND_FAILURE_THRESHOLD
        ),
        backend_recovery_seconds: float = (
            DEFAULT_BACKEND_RECOVERY_SECONDS
        ),
    ) -> None:

        if not namespace.strip():
            raise CacheConfigurationError(
                "namespace cannot be empty."
            )

        if len(namespace) > DEFAULT_MAX_NAMESPACE_LENGTH:
            raise CacheConfigurationError(
                "namespace too long."
            )

        if loader_timeout_seconds <= 0:
            raise CacheConfigurationError(
                "loader_timeout_seconds must be > 0."
            )

        if backend_timeout_seconds <= 0:
            raise CacheConfigurationError(
                "backend_timeout_seconds must be > 0."
            )

        if lock_timeout_seconds <= 0:
            raise CacheConfigurationError(
                "lock_timeout_seconds must be > 0."
            )

        if max_inflight < 1:
            raise CacheConfigurationError(
                "max_inflight must be >= 1."
            )

        if max_refresh_tasks < 1:
            raise CacheConfigurationError(
                "max_refresh_tasks must be >= 1."
            )

        self.namespace = namespace.strip()

        self.configuration = dict(
            configuration or {}
        )

        self.config_fingerprint = (
            configuration_fingerprint(
                self.configuration
            )
        )

        self.ttl_policy = (
            ttl_policy
            or CacheTTLPolicy()
        )

        self.local = (
            local
            or L1MemoryCache(
                clone_values=clone_values
            )
        )

        self.backend = backend

        self.loader_timeout_seconds = (
            loader_timeout_seconds
        )

        self.backend_timeout_seconds = (
            backend_timeout_seconds
        )

        self.lock_timeout_seconds = (
            lock_timeout_seconds
        )

        self.clone_values = clone_values

        self._inflight: Dict[
            str,
            asyncio.Task[Any],
        ] = {}

        self._refresh_tasks: Dict[
            str,
            asyncio.Task[Any],
        ] = {}

        self._generations: Dict[
            str,
            int,
        ] = {}

        self._namespace_generation = 0

        self._inflight_guard = asyncio.Lock()
        self._generation_lock = asyncio.Lock()

        self._refresh_semaphore = asyncio.Semaphore(
            max_refresh_tasks
        )

        self._inflight_semaphore = asyncio.Semaphore(
            max_inflight
        )

        self._backend_breaker = (
            BackendCircuitBreaker(
                failure_threshold=(
                    backend_failure_threshold
                ),
                recovery_seconds=(
                    backend_recovery_seconds
                ),
            )
        )

        self._shutdown = False

    # =========================================================================
    # KEY
    # =========================================================================

    def make_key(
        self,
        subject: str,
    ) -> CacheKey:
        return CacheKey(
            namespace=self.namespace,
            subject=subject,
            config_fingerprint=(
                self.config_fingerprint
            ),
        )

    # =========================================================================
    # GENERATION
    # =========================================================================

    async def generation(
        self,
        key: CacheKey,
    ) -> int:
        async with self._generation_lock:
            return self._generations.get(
                key.digest,
                self._namespace_generation,
            )

    async def bump_generation(
        self,
        subject: Optional[str] = None,
    ) -> int:
        """
        Invalidate one key or the whole namespace.

        Returns the new generation.

        Note:
            Generation fencing is process-local. L2 implementations should
            also implement generation-aware conditional writes if distributed
            invalidation is required.
        """

        async with self._generation_lock:
            if subject is None:
                self._namespace_generation += 1
                generation = (
                    self._namespace_generation
                )

                self.local.metrics.generation_bumps += 1

                return generation

            key = self.make_key(subject)

            current = self._generations.get(
                key.digest,
                self._namespace_generation,
            )

            current += 1

            self._generations[key.digest] = current

            self.local.metrics.generation_bumps += 1

            await self.local.delete(key)

            if self.backend:
                await self._backend_delete(
                    key
                )

            return current

    # =========================================================================
    # GET
    # =========================================================================

    async def get(
        self,
        subject: str,
        *,
        allow_stale: bool = False,
    ) -> Optional[T]:
        self._ensure_running()

        key = self.make_key(subject)
        generation = await self.generation(key)

        local_result = await self.local.get(
            key,
            generation=generation,
            allow_stale=allow_stale,
        )

        if local_result is not None:
            value, _ = local_result
            return value

        # ---------------------------------------------------------------------
        # L2
        # ---------------------------------------------------------------------

        backend_result = await self._backend_get(
            key
        )

        if backend_result is None:
            return None

        backend_generation = (
            backend_result.generation
        )

        if backend_generation != generation:
            return None

        value = backend_result.value

        if not is_cacheable_value(value):
            return None

        # ---------------------------------------------------------------------
        # Preserve backend remaining TTL where possible.
        # ---------------------------------------------------------------------

        ttl_override: Optional[float] = None

        if backend_result.expires_at is not None:
            ttl_override = max(
                0.0,
                backend_result.expires_at
                - time.monotonic(),
            )

        if ttl_override is not None:
            # Backend expiration is the upper bound.
            ttl_override = min(
                ttl_override,
                self.ttl_policy.fresh_ttl_seconds,
            )

        await self.local.set(
            key,
            value,
            ttl_policy=self.ttl_policy,
            generation=generation,
            negative=self._detect_negative(
                value
            ),
            ttl_override_seconds=(
                ttl_override
            ),
        )

        return clone_value(
            value,
            clone=self.clone_values,
            metrics=self.local.metrics,
        )

    # =========================================================================
    # SET
    # =========================================================================

    async def set(
        self,
        subject: str,
        value: T,
        *,
        negative: Optional[bool] = None,
        ttl_override_seconds: Optional[float] = None,
        only_if_fresh: bool = False,
    ) -> bool:
        self._ensure_running()

        if not is_cacheable_value(value):
            self.local.metrics.rejected_values += 1
            return False

        key = self.make_key(subject)

        generation = await self.generation(key)

        if only_if_fresh:
            existing = await self.local.get(
                key,
                generation=generation,
                allow_stale=False,
            )

            if existing is not None:
                return False

        if negative is None:
            negative = self._detect_negative(
                value
            )

        stored = await self.local.set(
            key,
            value,
            ttl_policy=self.ttl_policy,
            generation=generation,
            negative=negative,
            ttl_override_seconds=(
                ttl_override_seconds
            ),
        )

        if not stored:
            return False

        await self._backend_set(
            key,
            value,
            generation=generation,
            negative=negative,
            ttl_override_seconds=(
                ttl_override_seconds
            ),
        )

        return True

    # =========================================================================
    # GET OR SET
    # =========================================================================

    async def get_or_set(
        self,
        subject: str,
        loader: Loader[T],
        *,
        force_refresh: bool = False,
        allow_stale: bool = True,
        negative: Optional[bool] = None,
    ) -> T:
        """
        Main cache access path.

        Ordering:

            L1 fresh
                ↓
            L2
                ↓
            stale L1
                ↓
            request coalescing
                ↓
            loader
                ↓
            L1 + L2

        Force-refresh bypasses normal fresh cache reads but still respects
        the loader concurrency limit.
        """

        self._ensure_running()

        key = self.make_key(subject)

        generation = await self.generation(key)

        # ---------------------------------------------------------------------
        # Fast path
        # ---------------------------------------------------------------------

        if not force_refresh:
            local_result = await self.local.get(
                key,
                generation=generation,
                allow_stale=False,
            )

            if local_result is not None:
                value, _ = local_result
                return value

            # -----------------------------------------------------------------
            # L2 path
            # -----------------------------------------------------------------

            backend_result = await self._backend_get(
                key
            )

            if backend_result is not None:
                if (
                    backend_result.generation
                    == generation
                ):
                    value = backend_result.value

                    if is_cacheable_value(value):
                        await self.local.set(
                            key,
                            value,
                            ttl_policy=self.ttl_policy,
                            generation=generation,
                            negative=self._detect_negative(
                                value
                            ),
                            ttl_override_seconds=(
                                self._remaining_backend_ttl(
                                    backend_result
                                )
                            ),
                        )

                        return clone_value(
                            value,
                            clone=self.clone_values,
                            metrics=self.local.metrics,
                        )

        # ---------------------------------------------------------------------
        # Stale-while-revalidate
        # ---------------------------------------------------------------------

        if (
            allow_stale
            and not force_refresh
        ):
            stale = await self.local.get(
                key,
                generation=generation,
                allow_stale=True,
            )

            if stale is not None:
                value, state = stale

                if state == CacheEntryState.STALE:
                    await self._schedule_refresh(
                        key=key,
                        generation=generation,
                        loader=loader,
                        negative=negative,
                    )

                    return value

        # ---------------------------------------------------------------------
        # Request coalescing
        # ---------------------------------------------------------------------

        task = await self._get_or_create_loader_task(
            key=key,
            generation=generation,
            loader=loader,
            negative=negative,
            force_refresh=force_refresh,
        )

        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # A cancelled waiter must not cancel the shared loader.
            raise

    # =========================================================================
    # LOADER TASK MANAGEMENT
    # =========================================================================

    async def _get_or_create_loader_task(
        self,
        *,
        key: CacheKey,
        generation: int,
        loader: Loader[T],
        negative: Optional[bool],
        force_refresh: bool,
    ) -> asyncio.Task[T]:

        async with self._inflight_guard:

            if not force_refresh:
                existing = self._inflight.get(
                    key.digest
                )

                if existing is not None:
                    if not existing.done():
                        self.local.metrics.coalesced_waiters += 1
                        return existing

                    self._inflight.pop(
                        key.digest,
                        None,
                    )

            if (
                len(self._inflight)
                >= self._inflight_semaphore._value
            ):
                self.local.metrics.coalescing_rejections += 1

                # We intentionally do not queue unbounded requests.
                # The semaphore provides bounded admission.
                await self._inflight_semaphore.acquire()

                self._inflight_semaphore.release()

            else:
                await self._inflight_semaphore.acquire()

            task = asyncio.create_task(
                self._load_and_store(
                    key=key,
                    generation=generation,
                    loader=loader,
                    negative=negative,
                ),
                name=(
                    f"phishguard-cache-load:"
                    f"{key.digest[:16]}"
                ),
            )

            self._inflight[key.digest] = task

            task.add_done_callback(
                lambda finished, digest=key.digest:
                asyncio.create_task(
                    self._cleanup_inflight(
                        digest,
                        finished,
                    )
                )
            )

            return task

    async def _cleanup_inflight(
        self,
        digest: str,
        task: asyncio.Task[Any],
    ) -> None:
        async with self._inflight_guard:
            current = self._inflight.get(
                digest
            )

            if current is task:
                self._inflight.pop(
                    digest,
                    None,
                )

        self._inflight_semaphore.release()

    # =========================================================================
    # LOAD
    # =========================================================================

    async def _load_and_store(
        self,
        *,
        key: CacheKey,
        generation: int,
        loader: Loader[T],
        negative: Optional[bool],
    ) -> T:

        started = time.perf_counter()

        self.local.metrics.loader_calls += 1

        try:
            try:
                async with asyncio.timeout(
                    self.loader_timeout_seconds
                ):
                    value = await loader()

            except TimeoutError as exc:
                self.local.metrics.loader_timeouts += 1
                self.local.metrics.loader_failures += 1

                raise CacheLoaderTimeout(
                    f"Loader exceeded "
                    f"{self.loader_timeout_seconds:.2f}s."
                ) from exc

            if not is_cacheable_value(value):
                self.local.metrics.rejected_values += 1

                raise CacheValueError(
                    "Loader returned a non-cacheable value."
                )

            elapsed_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            self.local.metrics.total_load_latency_ms += (
                elapsed_ms
            )

            # -------------------------------------------------------------
            # Generation fencing.
            #
            # If invalidation happened while the provider was running,
            # don't write the old result back into cache.
            # -------------------------------------------------------------

            current_generation = await self.generation(
                key
            )

            if current_generation == generation:
                negative_value = (
                    self._detect_negative(value)
                    if negative is None
                    else negative
                )

                await self.local.set(
                    key,
                    value,
                    ttl_policy=self.ttl_policy,
                    generation=generation,
                    negative=negative_value,
                )

                await self._backend_set(
                    key,
                    value,
                    generation=generation,
                    negative=negative_value,
                )

            return clone_value(
                value,
                clone=self.clone_values,
                metrics=self.local.metrics,
            )

        except asyncio.CancelledError:
            # Never swallow cancellation.
            raise

        except Exception:
            self.local.metrics.loader_failures += 1
            raise

    # =========================================================================
    # STALE REFRESH
    # =========================================================================

    async def _schedule_refresh(
        self,
        *,
        key: CacheKey,
        generation: int,
        loader: Loader[T],
        negative: Optional[bool],
    ) -> None:

        if self._shutdown:
            return

        async with self._inflight_guard:

            existing = self._refresh_tasks.get(
                key.digest
            )

            if existing is not None:
                if not existing.done():
                    return

                self._refresh_tasks.pop(
                    key.digest,
                    None,
                )

            if (
                len(self._refresh_tasks)
                >= self._refresh_semaphore._value
            ):
                self.local.metrics.refresh_rejected += 1
                return

            task = asyncio.create_task(
                self._refresh(
                    key=key,
                    generation=generation,
                    loader=loader,
                    negative=negative,
                ),
                name=(
                    f"phishguard-cache-refresh:"
                    f"{key.digest[:16]}"
                ),
            )

            self._refresh_tasks[key.digest] = task

            self.local.metrics.refresh_scheduled += 1

            task.add_done_callback(
                lambda finished, digest=key.digest:
                asyncio.create_task(
                    self._cleanup_refresh(
                        digest,
                        finished,
                    )
                )
            )

    async def _cleanup_refresh(
        self,
        digest: str,
        task: asyncio.Task[Any],
    ) -> None:
        async with self._inflight_guard:
            current = self._refresh_tasks.get(
                digest
            )

            if current is task:
                self._refresh_tasks.pop(
                    digest,
                    None,
                )

    async def _refresh(
        self,
        *,
        key: CacheKey,
        generation: int,
        loader: Loader[T],
        negative: Optional[bool],
    ) -> None:

        async with self._refresh_semaphore:
            try:
                current_generation = await self.generation(
                    key
                )

                if current_generation != generation:
                    return

                value = await asyncio.wait_for(
                    loader(),
                    timeout=self.loader_timeout_seconds,
                )

                if not is_cacheable_value(value):
                    return

                latest_generation = await self.generation(
                    key
                )

                if latest_generation != generation:
                    return

                negative_value = (
                    self._detect_negative(value)
                    if negative is None
                    else negative
                )

                await self.local.set(
                    key,
                    value,
                    ttl_policy=self.ttl_policy,
                    generation=generation,
                    negative=negative_value,
                )

                await self._backend_set(
                    key,
                    value,
                    generation=generation,
                    negative=negative_value,
                )

                self.local.metrics.refresh_completed += 1

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                self.local.metrics.refresh_failed += 1

                logger.warning(
                    "Cache background refresh failed",
                    extra={
                        "namespace": self.namespace,
                        "key_digest": key.digest[:16],
                        "error_type": type(exc).__name__,
                    },
                )

    # =========================================================================
    # BACKEND
    # =========================================================================

    async def _backend_get(
        self,
        key: CacheKey,
    ) -> Optional[BackendValue]:

        if self.backend is None:
            return None

        if not await self._backend_breaker.allow():
            return None

        started = time.perf_counter()

        self.local.metrics.backend_gets += 1

        try:
            async with asyncio.timeout(
                self.backend_timeout_seconds
            ):
                result = await self.backend.get(
                    key.backend_key
                )

            elapsed_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            self.local.metrics.total_backend_latency_ms += (
                elapsed_ms
            )

            recovered = (
                await self._backend_breaker.success()
            )

            if recovered:
                self.local.metrics.circuit_recoveries += 1

            if result is None:
                self.local.metrics.backend_misses += 1
            else:
                self.local.metrics.backend_hits += 1

            return result

        except asyncio.CancelledError:
            raise

        except TimeoutError:
            self.local.metrics.backend_timeouts += 1
            self.local.metrics.backend_failures += 1

            opened = (
                await self._backend_breaker.failure()
            )

            if opened:
                self.local.metrics.circuit_opens += 1

            return None

        except Exception as exc:
            self.local.metrics.backend_failures += 1

            opened = (
                await self._backend_breaker.failure()
            )

            if opened:
                self.local.metrics.circuit_opens += 1

            logger.warning(
                "Cache backend get failed",
                extra={
                    "namespace": self.namespace,
                    "key_digest": key.digest[:16],
                    "error_type": type(exc).__name__,
                },
            )

            return None

    async def _backend_set(
        self,
        key: CacheKey,
        value: Any,
        *,
        generation: int,
        negative: bool,
        ttl_override_seconds: Optional[float] = None,
    ) -> None:

        if self.backend is None:
            return

        if not is_cacheable_value(value):
            return

        if not await self._backend_breaker.allow():
            return

        ttl = (
            ttl_override_seconds
            if ttl_override_seconds is not None
            else self.ttl_policy.ttl(
                negative=negative
            )
        )

        ttl = max(0.001, ttl)

        backend_value = BackendValue(
            value=clone_value(
                value,
                clone=self.clone_values,
                metrics=self.local.metrics,
            ),
            expires_at=(
                time.monotonic() + ttl
            ),
            generation=generation,
        )

        started = time.perf_counter()

        self.local.metrics.backend_sets += 1

        try:
            async with asyncio.timeout(
                self.backend_timeout_seconds
            ):
                await self.backend.set(
                    key.backend_key,
                    backend_value,
                    ttl,
                )

            elapsed_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            self.local.metrics.total_backend_latency_ms += (
                elapsed_ms
            )

            recovered = (
                await self._backend_breaker.success()
            )

            if recovered:
                self.local.metrics.circuit_recoveries += 1

        except asyncio.CancelledError:
            raise

        except TimeoutError:
            self.local.metrics.backend_timeouts += 1
            self.local.metrics.backend_failures += 1

            opened = (
                await self._backend_breaker.failure()
            )

            if opened:
                self.local.metrics.circuit_opens += 1

        except Exception as exc:
            self.local.metrics.backend_failures += 1

            opened = (
                await self._backend_breaker.failure()
            )

            if opened:
                self.local.metrics.circuit_opens += 1

            logger.warning(
                "Cache backend set failed",
                extra={
                    "namespace": self.namespace,
                    "key_digest": key.digest[:16],
                    "error_type": type(exc).__name__,
                },
            )

    async def _backend_delete(
        self,
        key: CacheKey,
    ) -> None:

        if self.backend is None:
            return

        if not await self._backend_breaker.allow():
            return

        self.local.metrics.backend_deletes += 1

        try:
            async with asyncio.timeout(
                self.backend_timeout_seconds
            ):
                await self.backend.delete(
                    key.backend_key
                )

            await self._backend_breaker.success()

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            self.local.metrics.backend_failures += 1

            opened = (
                await self._backend_breaker.failure()
            )

            if opened:
                self.local.metrics.circuit_opens += 1

            logger.warning(
                "Cache backend delete failed",
                extra={
                    "namespace": self.namespace,
                    "key_digest": key.digest[:16],
                    "error_type": type(exc).__name__,
                },
            )

    # =========================================================================
    # NAMESPACE INVALIDATION
    # =========================================================================

    async def clear_namespace(
        self,
    ) -> None:

        self._ensure_running()

        async with self._generation_lock:
            self._namespace_generation += 1
            self._generations.clear()

        await self.local.clear_namespace(
            self.namespace
        )

        if self.backend is not None:
            try:
                namespace_digest = hashlib.sha256(
                    self.namespace.encode(
                        "utf-8"
                    )
                ).hexdigest()[:16]

                if await self._backend_breaker.allow():
                    async with asyncio.timeout(
                        self.backend_timeout_seconds
                    ):
                        await self.backend.clear_namespace(
                            namespace_digest
                        )

                    recovered = (
                        await self._backend_breaker.success()
                    )

                    if recovered:
                        self.local.metrics.circuit_recoveries += 1

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                self.local.metrics.backend_failures += 1

                opened = (
                    await self._backend_breaker.failure()
                )

                if opened:
                    self.local.metrics.circuit_opens += 1

                logger.warning(
                    "Cache backend namespace clear failed",
                    extra={
                        "namespace": self.namespace,
                        "error_type": type(exc).__name__,
                    },
                )

        self.local.metrics.namespace_clears += 1

    # =========================================================================
    # DELETE
    # =========================================================================

    async def delete(
        self,
        subject: str,
    ) -> None:

        self._ensure_running()

        key = self.make_key(subject)

        await self.bump_generation(
            subject
        )

        await self.local.delete(key)

        await self._backend_delete(key)

    # =========================================================================
    # NEGATIVE DETECTION
    # =========================================================================

    @staticmethod
    def _detect_negative(
        value: Any,
    ) -> bool:
        """
        Automatically recognize common UNKNOWN/negative reputation results.

        This is deliberately conservative.

        A result is negative only when its structure explicitly indicates
        unknown / unavailable / not-found semantics. A low risk score alone
        does NOT make an object negative.
        """

        if value is None:
            return True

        # Pydantic model / object attributes.
        status = getattr(
            value,
            "status",
            None,
        )

        risk_level = getattr(
            value,
            "risk_level",
            None,
        )

        match_type = getattr(
            value,
            "match_type",
            None,
        )

        normalized = {
            str(status).lower()
            if status is not None
            else "",
            str(risk_level).lower()
            if risk_level is not None
            else "",
            str(match_type).lower()
            if match_type is not None
            else "",
        }

        unknown_tokens = {
            "unknown",
            "invalid_input",
            "unavailable",
            "not_found",
            "none",
            "none_found",
        }

        return bool(
            normalized
            & unknown_tokens
        )

    # =========================================================================
    # BACKEND TTL
    # =========================================================================

    @staticmethod
    def _remaining_backend_ttl(
        backend_value: BackendValue,
    ) -> Optional[float]:

        if backend_value.expires_at is None:
            return None

        return max(
            0.0,
            backend_value.expires_at
            - time.monotonic(),
        )

    # =========================================================================
    # STATS
    # =========================================================================

    async def stats(
        self,
    ) -> Dict[str, Any]:

        breaker_state = await (
            self._backend_breaker.state()
        )

        return {
            "module_version": MODULE_VERSION,
            "namespace": self.namespace,
            "config_fingerprint": (
                self.config_fingerprint
            ),
            "l1_entries": self.local.size,
            "l1_weight_bytes": (
                self.local.weight_bytes
            ),
            "l1_max_entries": (
                self.local.max_entries
            ),
            "l1_max_weight_bytes": (
                self.local.max_weight_bytes
            ),
            "inflight": len(
                self._inflight
            ),
            "max_inflight": (
                self._inflight_semaphore._value
            ),
            "refreshing": len(
                self._refresh_tasks
            ),
            "max_refresh_tasks": (
                self._refresh_semaphore._value
            ),
            "namespace_generation": (
                self._namespace_generation
            ),
            "backend_mode": (
                CacheBackendMode.DISABLED.value
                if self.backend is None
                else breaker_state.value
            ),
            "shutdown": self._shutdown,
            "metrics": self.local.metrics.snapshot(),
        }

    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    async def close(
        self,
        *,
        cancel_refresh: bool = True,
        cancel_inflight: bool = False,
    ) -> None:

        if self._shutdown:
            return

        self._shutdown = True

        tasks: list[asyncio.Task[Any]] = []

        async with self._inflight_guard:
            if cancel_refresh:
                tasks.extend(
                    self._refresh_tasks.values()
                )

            if cancel_inflight:
                tasks.extend(
                    self._inflight.values()
                )

        unique_tasks = {
            id(task): task
            for task in tasks
        }

        for task in unique_tasks.values():
            if not task.done():
                task.cancel()

        if unique_tasks:
            await asyncio.gather(
                *unique_tasks.values(),
                return_exceptions=True,
            )

        async with self._inflight_guard:
            self._refresh_tasks.clear()

            if cancel_inflight:
                self._inflight.clear()

        if cancel_inflight:
            # Defensive reset. Loader task callbacks normally release these.
            while True:
                try:
                    self._inflight_semaphore.release()
                except ValueError:
                    break

        await self.local.clear()

    # =========================================================================
    # INTERNAL
    # =========================================================================

    def _ensure_running(
        self,
    ) -> None:
        if self._shutdown:
            raise CacheShutdownError(
                "CacheStrategy is closed."
            )


# =============================================================================
# DOMAIN COMPATIBILITY SHIM
# =============================================================================

class DomainReputationCache:
    """
    Backward-compatible adapter for the old domain reputation cache API.

    Existing code can continue using:

        await cache.get(key)
        await cache.set(key, assessment)

    while internally using the hardened CacheStrategy.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = DEFAULT_FRESH_TTL_SECONDS,
        max_entries: int = 10_000,
        stale_ttl_seconds: int = DEFAULT_STALE_TTL_SECONDS,
        negative_ttl_seconds: int = DEFAULT_NEGATIVE_TTL_SECONDS,
        backend: Optional[CacheBackend] = None,
    ) -> None:

        self.strategy = CacheStrategy(
            namespace="reputation.domain",
            ttl_policy=CacheTTLPolicy(
                fresh_ttl_seconds=float(
                    ttl_seconds
                ),
                stale_ttl_seconds=float(
                    stale_ttl_seconds
                ),
                negative_ttl_seconds=float(
                    negative_ttl_seconds
                ),
            ),
            local=L1MemoryCache(
                max_entries=max_entries,
            ),
            backend=backend,
            configuration={
                "ttl_seconds": ttl_seconds,
                "stale_ttl_seconds": (
                    stale_ttl_seconds
                ),
                "negative_ttl_seconds": (
                    negative_ttl_seconds
                ),
                "max_entries": max_entries,
            },
        )

    async def get(
        self,
        key: str,
    ) -> Optional[Any]:
        return await self.strategy.get(
            key
        )

    async def set(
        self,
        key: str,
        value: Any,
    ) -> None:
        await self.strategy.set(
            key,
            value,
        )

    async def clear(
        self,
    ) -> None:
        await self.strategy.clear_namespace()

    async def stats(
        self,
    ) -> Dict[str, Any]:
        return await self.strategy.stats()

    async def close(
        self,
    ) -> None:
        await self.strategy.close()


# =============================================================================
# FACTORY
# =============================================================================

def create_reputation_cache(
    *,
    namespace: str,
    configuration: Optional[
        Mapping[str, Any]
    ] = None,
    fresh_ttl_seconds: float = (
        DEFAULT_FRESH_TTL_SECONDS
    ),
    stale_ttl_seconds: float = (
        DEFAULT_STALE_TTL_SECONDS
    ),
    negative_ttl_seconds: float = (
        DEFAULT_NEGATIVE_TTL_SECONDS
    ),
    max_entries: int = DEFAULT_MAX_ENTRIES,
    max_weight_bytes: int = DEFAULT_MAX_WEIGHT_BYTES,
    backend: Optional[CacheBackend] = None,
) -> CacheStrategy:
    """
    Standard factory for domain/IP/hash reputation engines.
    """

    return CacheStrategy(
        namespace=namespace,
        configuration=configuration,
        ttl_policy=CacheTTLPolicy(
            fresh_ttl_seconds=(
                fresh_ttl_seconds
            ),
            stale_ttl_seconds=(
                stale_ttl_seconds
            ),
            negative_ttl_seconds=(
                negative_ttl_seconds
            ),
        ),
        local=L1MemoryCache(
            max_entries=max_entries,
            max_weight_bytes=max_weight_bytes,
        ),
        backend=backend,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Version
    "MODULE_VERSION",

    # Enums
    "CacheEntryState",
    "CacheBackendMode",
    "CacheResultState",

    # Exceptions
    "CacheError",
    "CacheConfigurationError",
    "CacheKeyError",
    "CacheValueError",
    "CacheBackendError",
    "CacheBackendUnavailable",
    "CacheLoaderTimeout",
    "CacheShutdownError",

    # Core
    "CacheKey",
    "CacheTTLPolicy",
    "CacheEntry",
    "CacheMetrics",
    "BackendValue",
    "CacheBackend",

    # Components
    "BackendCircuitBreaker",
    "L1MemoryCache",
    "CacheStrategy",

    # Compatibility
    "DomainReputationCache",

    # Utilities
    "configuration_fingerprint",
    "estimate_size",
    "is_cacheable_value",
    "clone_value",
    "create_reputation_cache",
]
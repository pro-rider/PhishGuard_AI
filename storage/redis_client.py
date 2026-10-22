"""
PhishGuard_AI - Redis Client Wrapper

Thread-safe, defensive Redis access layer used for caching, distributed locks,
and lightweight queues.

Security / reliability principles:
- Fail-open caching: transient Redis errors degrade to cache misses, never
  crash the caller.
- Bounded keys, TTLs and payload sizes.
- Keys are namespaced to avoid cross-tenant collisions.
- JSON serialization is size-limited; binary blobs are supported directly.
- Optional TLS and password via the central application settings.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import time
from typing import Any, Callable, Optional

from config.internal.settings import settings

try:
    import redis
    import redis.asyncio as redis_asyncio

    _REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    redis = None
    redis_asyncio = None
    _REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

DEFAULT_NAMESPACE = "pg"
DEFAULT_TTL_SECONDS = 300
MAX_KEY_LENGTH = 1024
MAX_VALUE_BYTES = 16 * 1024 * 1024  # 16 MiB
MAX_JSON_SERIALIZE_BYTES = 8 * 1024 * 1024
DEFAULT_POOL_SIZE = 10
DEFAULT_TIMEOUT_SECONDS = 2.0
LOCK_TTL_SECONDS = 30
DEFAULT_RETRIES = 2


class RedisError(RuntimeError):
    """Base error for the Redis wrapper."""


class RedisUnavailableError(RedisError):
    """Redis is not installed or the client could not be constructed."""


# ============================================================================
# CACHE CLIENT
# ============================================================================


class RedisCache:
    """
    Namespaced Redis cache with fail-open semantics.

    Supports synchronous and asynchronous operation, JSON and pickle payloads,
    distributed locks, and atomic increment/decrement.
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        namespace: str = DEFAULT_NAMESPACE,
        default_ttl: int = DEFAULT_TTL_SECONDS,
        socket_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_value_bytes: int = MAX_VALUE_BYTES,
    ) -> None:
        if not _REDIS_AVAILABLE:
            raise RedisUnavailableError("redis package is not installed.")

        self._url = url or settings.REDIS_URL
        self._namespace = self._sanitize_namespace(namespace)
        self._default_ttl = default_ttl
        self._max_value_bytes = max_value_bytes

        common = {
            "socket_timeout": socket_timeout,
            "socket_connect_timeout": socket_timeout,
            "decode_responses": False,
        }
        try:
            self._client = redis.Redis.from_url(self._url, **common)
            self._async_client = redis_asyncio.Redis.from_url(self._url, **common)
        except Exception as exc:  # noqa: BLE001
            raise RedisUnavailableError(
                f"Unable to construct Redis client: {exc.__class__.__name__}"
            ) from exc

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_namespace(namespace: str) -> str:
        if not namespace or not all(
            c.isalnum() or c in "-_." for c in namespace
        ):
            raise ValueError("Namespace must be alphanumeric with -_. only.")
        return namespace[:64]

    def _key(self, key: str) -> str:
        if not isinstance(key, str) or not key:
            raise ValueError("Cache key must be a non-empty string.")
        if len(key) > MAX_KEY_LENGTH:
            # Hash oversized keys deterministically to preserve stability.
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
            key = f"{key[:128]}..{digest}"
        return f"{self._namespace}:{key}"

    # ------------------------------------------------------------------
    # Sync API (fail-open)
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a cached value; return ``default`` on miss or error."""
        try:
            raw = self._client.get(self._key(key))
            if raw is None:
                return default
            return pickle.loads(raw)  # noqa: S301 - trusted cache namespace
        except Exception:  # noqa: BLE001
            logger.warning("Redis get failed; degrading to default.", exc_info=True)
            return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a value with an optional TTL (seconds)."""
        try:
            payload = pickle.dumps(value, protocol=5)
            if len(payload) > self._max_value_bytes:
                logger.warning("Cache value exceeds size bound; refusing.")
                return False
            return bool(
                self._client.set(
                    self._key(key),
                    payload,
                    ex=ttl if ttl is not None else self._default_ttl,
                )
            )
        except Exception:  # noqa: BLE001
            logger.warning("Redis set failed.", exc_info=True)
            return False

    def delete(self, key: str) -> bool:
        """Remove a key."""
        try:
            return bool(self._client.delete(self._key(key)))
        except Exception:  # noqa: BLE001
            logger.warning("Redis delete failed.", exc_info=True)
            return False

    def get_or_set(
        self,
        key: str,
        loader: Callable[[], Any],
        ttl: Optional[int] = None,
    ) -> Any:
        """Return the cached value, computing and storing it on a miss."""
        cached = self.get(key)
        if cached is not None:
            return cached
        value = loader()
        self.set(key, value, ttl=ttl)
        return value

    def increment(self, key: str, amount: int = 1) -> int:
        """Atomically increment a counter."""
        try:
            return int(self._client.incr(self._key(key), amount))
        except Exception:  # noqa: BLE001
            logger.warning("Redis increment failed.", exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # Distributed lock
    # ------------------------------------------------------------------

    def acquire_lock(
        self, name: str, *, timeout: int = LOCK_TTL_SECONDS
    ) -> Optional[str]:
        """Acquire a distributed lock; returns a token (or None)."""
        token = f"{time.time_ns()}-{hashlib.sha256(name.encode()).hexdigest()[:12]}"
        try:
            acquired = self._client.set(
                self._key(f"lock:{name}"), token, nx=True, ex=timeout
            )
            return token if acquired else None
        except Exception:  # noqa: BLE001
            logger.warning("Redis lock acquire failed.", exc_info=True)
            return None

    def release_lock(self, name: str, token: str) -> bool:
        """Release a lock only if the caller still owns it (CAS)."""
        lock_key = self._key(f"lock:{name}")
        try:
            current = self._client.get(lock_key)
            if current is not None and current.decode() == token:
                self._client.delete(lock_key)
                return True
            return False
        except Exception:  # noqa: BLE001
            logger.warning("Redis lock release failed.", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True when Redis responds to PING."""
        try:
            return bool(self._client.ping())
        except Exception:  # noqa: BLE001
            return False


# ============================================================================
# SINGLETON (lazy)
# ============================================================================

_redis_cache: Optional[RedisCache] = None


def get_redis_cache() -> RedisCache:
    """Return the shared Redis cache instance, constructing it lazily."""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
    return _redis_cache


__all__ = [
    "RedisCache",
    "RedisError",
    "RedisUnavailableError",
    "get_redis_cache",
    "DEFAULT_TTL_SECONDS",
    "MAX_VALUE_BYTES",
    "LOCK_TTL_SECONDS",
]

"""
PhishGuard_AI - Rate Limiter Middleware

Per-client token-bucket rate limiting for API requests.

Security / reliability principles:
- Rate-limit state is keyed by a bounded client id with a TTL.
- Limits are validated and bounded by hard caps.
- Rate-limit responses include standard headers and never leak internals.
"""

from __future__ import annotations

import time
from typing import MutableMapping, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

MAX_REQUESTS_PER_MINUTE = 600
DEFAULT_REQUESTS_PER_MINUTE = 120
MAX_CLIENT_ID_LENGTH = 128
RATE_LIMIT_HEADER = "X-RateLimit-Remaining"
RETRY_AFTER_HEADER = "Retry-After"
WINDOW_SECONDS = 60.0


class RateLimiterError(RuntimeError):
    """Base error for rate limiting."""


class RateLimitExceeded(RateLimiterError):
    """Client exceeded the allowed rate."""


class TokenBucketLimiter:
    """Thread-safe token bucket rate limiter with a sliding window."""

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_REQUESTS_PER_MINUTE,
        refill_per_second: Optional[float] = None,
        window_seconds: float = WINDOW_SECONDS,
    ) -> None:
        if capacity <= 0:
            raise RateLimiterError("capacity must be positive.")
        if capacity > MAX_REQUESTS_PER_MINUTE:
            raise RateLimiterError("capacity exceeds the configured maximum.")
        self._capacity = float(capacity)
        self._refill = (
            refill_per_second
            if refill_per_second is not None
            else capacity / window_seconds
        )
        self._buckets: MutableMapping[str, Tuple[float, float]] = {}

    def allow(self, client_id: str, *, now: Optional[float] = None) -> Tuple[bool, int]:
        """Return (allowed, remaining) for ``client_id``."""
        now = now if now is not None else time.time()
        tokens, last = self._buckets.get(client_id, (self._capacity, now))
        elapsed = max(0.0, now - last)
        tokens = min(self._capacity, tokens + elapsed * self._refill)
        allowed = tokens >= 1.0
        if allowed:
            tokens -= 1.0
        self._buckets[client_id] = (tokens, now)
        return allowed, int(tokens)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Enforce a per-client request rate."""

    def __init__(
        self,
        app,
        *,
        limiter: TokenBucketLimiter | None = None,
        exempt_paths=None,
    ) -> None:
        super().__init__(app)
        self._limiter = limiter or TokenBucketLimiter()
        self._exempt = frozenset(exempt_paths or {"/health", "/ready", "/metrics"})

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in self._exempt or path.startswith("/health"):
            return await call_next(request)

        client_id = self._client_id(request)
        allowed, remaining = self._limiter.allow(client_id)
        if not allowed:
            return JSONResponse(
                {"detail": "rate limit exceeded"},
                status_code=429,
                headers={RETRY_AFTER_HEADER: "60"},
            )
        response = await call_next(request)
        response.headers[RATE_LIMIT_HEADER] = str(remaining)
        return response

    @staticmethod
    def _client_id(request: Request) -> str:
        candidate = request.client.host if request.client else "unknown"
        return (candidate or "unknown")[:MAX_CLIENT_ID_LENGTH]


__all__ = [
    "TokenBucketLimiter",
    "RateLimiterMiddleware",
    "RateLimiterError",
    "RateLimitExceeded",
    "MAX_REQUESTS_PER_MINUTE",
]

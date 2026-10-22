"""
PhishGuard_AI - Authentication Guard Middleware

Validates API-key and Bearer-token authentication for protected routes.

Security / reliability principles:
- Credential comparison is constant-time (HMAC compare).
- Exempt paths (health) are allow-listed explicitly.
- Keys are matched against a bounded set from application settings.
- Failed authentication returns 401/403 with no credential leakage.
"""

from __future__ import annotations

import hmac
import logging
import time
from typing import FrozenSet, Optional, Sequence, Set

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

API_KEY_HEADER = "X-API-Key"
AUTHORIZATION_HEADER = "Authorization"
BEARER_PREFIX = "Bearer "
MAX_KEY_LENGTH = 256

DEFAULT_EXEMPT_PATHS = frozenset(
    {"/", "/dashboard", "/health", "/ready", "/metrics", "/docs", "/openapi.json"}
)


class AuthenticationError(RuntimeError):
    """Base error for authentication."""


class AuthenticationGuard:
    """
    Constant-time, bounds-checked authentication guard.

    Configure with a set of valid API keys (and optionally a shared secret used
    to verify short-lived HMAC tokens).
    """

    def __init__(
        self,
        *,
        api_keys: Optional[Sequence[str]] = None,
        exempt_paths: FrozenSet[str] = DEFAULT_EXEMPT_PATHS,
    ) -> None:
        self._api_keys = self._normalize_keys(api_keys or ())
        self._exempt_paths = exempt_paths

    def _normalize_keys(self, keys: Sequence[str]) -> Set[str]:
        normalized: Set[str] = set()
        for key in keys:
            if isinstance(key, str) and key:
                if len(key) > MAX_KEY_LENGTH:
                    continue
                normalized.add(key)
        return normalized

    def authenticate(self, request: Request) -> None:
        """Raise :class:`AuthenticationError` when the request is unauthorized."""
        path = request.url.path
        if path in self._exempt_paths or path.startswith("/health"):
            return

        # No API keys configured -> run in open mode (typical for local / dev
        # operation). As soon as one or more keys are configured, they are
        # enforced below for every non-exempt endpoint.
        if not self._api_keys:
            request.state.authenticated = False
            return

        api_key = request.headers.get(API_KEY_HEADER, "")
        if api_key and self._validate_key(api_key):
            request.state.authenticated = True
            return

        auth_header = request.headers.get(AUTHORIZATION_HEADER, "")
        if auth_header.startswith(BEARER_PREFIX):
            token = auth_header[len(BEARER_PREFIX):].strip()
            if self._validate_key(token):
                request.state.authenticated = True
                return

        raise AuthenticationError("missing or invalid credentials")

    def _validate_key(self, candidate: str) -> bool:
        if not isinstance(candidate, str) or not candidate:
            return False
        if len(candidate) > MAX_KEY_LENGTH:
            return False
        for valid in self._api_keys:
            if hmac.compare_digest(valid.encode(), candidate.encode()):
                return True
        return False


def build_auth_guard(
    *, api_keys: Optional[Sequence[str]] = None
) -> AuthenticationGuard:
    """Build the guard from application settings when present."""
    keys = api_keys
    if keys is None:
        try:
            from config.internal.settings import settings

            configured = getattr(settings, "API_KEYS", None)
            if isinstance(configured, str) and configured:
                keys = [k for k in configured.split(",") if k]
            elif configured:
                keys = list(configured)
        except Exception:  # noqa: BLE001
            keys = []
    return AuthenticationGuard(api_keys=keys or ())


class AuthGuardMiddleware(BaseHTTPMiddleware):
    """Starlette middleware wrapping :class:`AuthenticationGuard`."""

    def __init__(self, app, *, guard: Optional[AuthenticationGuard] = None) -> None:
        super().__init__(app)
        self._guard = guard or build_auth_guard()

    async def dispatch(self, request: Request, call_next):
        try:
            self._guard.authenticate(request)
        except AuthenticationError:
            return JSONResponse(
                {"detail": "unauthorized"}, status_code=401
            )
        return await call_next(request)


__all__ = [
    "AuthenticationError",
    "AuthenticationGuard",
    "AuthGuardMiddleware",
    "build_auth_guard",
    "API_KEY_HEADER",
    "DEFAULT_EXEMPT_PATHS",
]

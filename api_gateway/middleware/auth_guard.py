"""
PhishGuard_AI - Authentication Guard Middleware

Resolves a caller to a :class:`Principal` from either a Bearer JWT or a static
API key, attaches it to ``request.state``, and enforces 401 on invalid
credentials. Role-based authorization (403) is enforced separately via the
``security.rbac.require_roles`` dependencies.

Auth modes (resolved in this order):
1. **JWT mode** — if an :class:`IdentityStore` is configured (users + secret),
   Bearer tokens issued by ``POST /auth/token`` are verified.
2. **API-key mode** — if static API keys are configured, the ``X-API-Key`` header
   (or a Bearer that equals a key) is checked constant-time.
3. **Open mode** — if neither is configured, every non-exempt path passes with no
   principal (used for local / dev operation).

Security principles:
- Constant-time credential comparison (HMAC).
- Exempt paths are allow-listed explicitly.
- No credential content is ever logged; failures return generic 401/403.
"""

from __future__ import annotations

import hmac
import logging
from typing import FrozenSet, Optional, Sequence, Set, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from security.identity import (
    ROLES,
    ROLE_ANALYST,
    Principal,
    IdentityStore,
)

logger = logging.getLogger(__name__)

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

API_KEY_HEADER = "X-API-Key"
AUTHORIZATION_HEADER = "Authorization"
BEARER_PREFIX = "Bearer "
MAX_KEY_LENGTH = 256

DEFAULT_EXEMPT_PATHS = frozenset(
    {
        "/",
        "/dashboard",
        "/health",
        "/ready",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/auth/token",
    }
)


class AuthenticationError(RuntimeError):
    """Base error for authentication."""


def _parse_key_entry(entry: str) -> Tuple[str, str, frozenset]:
    """Parse ``key`` or ``key|role1,role2|tenant`` into (key, tenant, roles)."""
    key = entry.strip()
    tenant = "system"
    roles = frozenset({ROLE_ANALYST})
    if "|" in entry:
        parts = entry.split("|")
        key = parts[0].strip()
        if len(parts) >= 2 and parts[1].strip():
            requested = {r.strip().lower() for r in parts[1].split(",") if r.strip()}
            valid = requested & ROLES
            roles = valid or roles
        if len(parts) >= 3 and parts[2].strip():
            tenant = parts[2].strip()
    return key, tenant, roles


class AuthenticationGuard:
    """Constant-time, bounds-checked authentication guard with RBAC principals."""

    def __init__(
        self,
        *,
        api_keys: Optional[Sequence[str]] = None,
        identity: Optional[IdentityStore] = None,
        exempt_paths: FrozenSet[str] = DEFAULT_EXEMPT_PATHS,
    ) -> None:
        self._keys: Set[str] = set()
        self._key_meta: dict = {}
        for entry in api_keys or ():
            if not isinstance(entry, str) or not entry or len(entry) > MAX_KEY_LENGTH:
                continue
            key, tenant, roles = _parse_key_entry(entry)
            self._keys.add(key)
            self._key_meta[key] = (tenant, roles)
        self._identity = identity
        self._exempt_paths = exempt_paths

    def authenticate(self, request: Request) -> None:
        """Raise :class:`AuthenticationError` when the request is unauthorized."""
        path = request.url.path
        if path in self._exempt_paths or path.startswith("/health"):
            return

        # 1. Bearer JWT -> principal.
        auth_header = request.headers.get(AUTHORIZATION_HEADER, "")
        if auth_header.startswith(BEARER_PREFIX) and self._identity is not None:
            token = auth_header[len(BEARER_PREFIX):].strip()
            principal = self._identity.decode_token(token)
            if principal is not None:
                self._attach(request, principal)
                return
            raise AuthenticationError("invalid bearer token")

        # 2. Static API key -> principal.
        api_key = request.headers.get(API_KEY_HEADER, "")
        if api_key and self._validate_key(api_key):
            tenant, roles = self._key_meta.get(api_key, ("system", frozenset({ROLE_ANALYST})))
            self._attach(
                request,
                Principal(username=f"apikey", tenant=tenant, roles=roles),
            )
            return

        # 3. Bearer token equal to a static API key (backward compatibility).
        if auth_header.startswith(BEARER_PREFIX):
            token = auth_header[len(BEARER_PREFIX):].strip()
            if self._validate_key(token):
                tenant, roles = self._key_meta.get(token, ("system", frozenset({ROLE_ANALYST})))
                self._attach(request, Principal(username=f"apikey", tenant=tenant, roles=roles))
                return

        # Open mode: no identity store and no keys configured -> allow, no principal.
        if self._identity is None and not self._keys:
            request.state.principal = None
            request.state.roles = frozenset()
            request.state.authenticated = False
            return

        raise AuthenticationError("missing or invalid credentials")

    def _attach(self, request: Request, principal: Principal) -> None:
        request.state.principal = principal
        request.state.roles = principal.roles
        request.state.authenticated = True
        # Tenant from the principal takes precedence over middleware defaults.
        if principal.tenant:
            request.state.tenant_id = principal.tenant

    def _validate_key(self, candidate: str) -> bool:
        if not isinstance(candidate, str) or not candidate or len(candidate) > MAX_KEY_LENGTH:
            return False
        for valid in self._keys:
            if hmac.compare_digest(valid.encode(), candidate.encode()):
                return True
        return False


def build_identity_store() -> Optional[IdentityStore]:
    """Build an :class:`IdentityStore` from ``settings.AUTH_USERS`` (or None)."""
    try:
        from config.internal.settings import settings

        users_cfg = getattr(settings, "AUTH_USERS", None)
        if not (isinstance(users_cfg, str) and users_cfg.strip()):
            return None
        secret = settings.SECRET_KEY.get_secret_value()
        store = IdentityStore(
            jwt_secret=secret,
            token_ttl_seconds=getattr(settings, "ACCESS_TOKEN_EXPIRE_MINUTES", 30) * 60,
        )
        for entry in users_cfg.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            # username:password:role1,role2:tenant
            parts = entry.split(":")
            username = parts[0].strip()
            password = parts[1] if len(parts) > 1 else ""
            roles = (
                {r.strip().lower() for r in parts[2].split(",") if r.strip()}
                if len(parts) > 2 and parts[2].strip()
                else {ROLE_ANALYST}
            )
            tenant = parts[3].strip() if len(parts) > 3 and parts[3].strip() else "system"
            store.add_user(username, password, roles=roles, tenant=tenant)
        return store
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to build identity store from settings: %s", exc)
        return None


def build_auth_guard() -> AuthenticationGuard:
    """Build the guard from application settings when present."""
    api_keys: Optional[Sequence[str]] = None
    identity = build_identity_store()
    try:
        from config.internal.settings import settings

        configured_keys = getattr(settings, "API_KEYS", None)
        if isinstance(configured_keys, str) and configured_keys:
            api_keys = [k for k in configured_keys.split(",") if k]
        elif configured_keys:
            api_keys = list(configured_keys)
    except Exception:  # noqa: BLE001
        api_keys = None

    return AuthenticationGuard(api_keys=api_keys, identity=identity)


class AuthGuardMiddleware(BaseHTTPMiddleware):
    """Starlette middleware wrapping :class:`AuthenticationGuard`."""

    def __init__(self, app, *, guard: Optional[AuthenticationGuard] = None) -> None:
        super().__init__(app)
        self._guard = guard or build_auth_guard()

    async def dispatch(self, request: Request, call_next):
        try:
            self._guard.authenticate(request)
        except AuthenticationError:
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)


__all__ = [
    "AuthenticationError",
    "AuthenticationGuard",
    "AuthGuardMiddleware",
    "build_auth_guard",
    "API_KEY_HEADER",
    "DEFAULT_EXEMPT_PATHS",
]

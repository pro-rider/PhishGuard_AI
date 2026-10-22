"""
PhishGuard_AI - Identity & Token Service

Principals, password hashing, and JWT issuance/verification for role-based
access control (RBAC). Implemented with only the standard library (PBKDF2-SHA256
for passwords, HS256 JWT via hmac/base64/json) so it works anywhere the project
runs, with no extra dependencies.

Security / reliability principles:
- Passwords are salted with a random 16-byte salt and stretched with PBKDF2-SHA256
  (210,000 iterations); verification uses constant-time comparison.
- JWTs are HS256-signed with the application secret; expiry is enforced; all
  payload values are bounds-checked on decode.
- Roles are validated against a fixed allow-list; a Principal is immutable.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Optional

# ============================================================================
# ROLES
# ============================================================================

ROLE_ADMIN = "admin"
ROLE_ANALYST = "analyst"
ROLE_VIEWER = "viewer"
ROLES: FrozenSet[str] = frozenset({ROLE_ADMIN, ROLE_ANALYST, ROLE_VIEWER})

# ============================================================================
# BOUNDS
# ============================================================================

MAX_USERNAME_LENGTH = 128
MAX_ROLES_PER_PRINCIPAL = 8
JWT_MAX_LIFETIME_SECONDS = 86400  # 24h hard cap
DEFAULT_TTL_SECONDS = 1800

# ============================================================================
# PASSWORD HASHING
# ============================================================================

_PBKDF2_ITERATIONS = 210_000


def hash_password(password: str, *, salt: Optional[str] = None) -> str:
    """Return a ``pbkdf2_sha256$<iters>$<salt>$<hex>`` encoded hash."""
    if not isinstance(password, str) or not password:
        raise ValueError("password must be a non-empty string.")
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time password verification against a stored hash."""
    try:
        algo, iters, salt, expected = stored.split("$", 3)
        if algo != "pbkdf2_sha256" or not iters.isdigit():
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            int(iters),
        )
        return hmac.compare_digest(dk.hex(), expected)
    except Exception:  # noqa: BLE001
        return False


# ============================================================================
# PRINCIPAL
# ============================================================================


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated identity with roles and a tenant scope."""

    username: str
    tenant: str = "system"
    roles: FrozenSet[str] = frozenset({ROLE_VIEWER})

    def __post_init__(self) -> None:
        if not isinstance(self.username, str) or not self.username:
            raise ValueError("username must be a non-empty string.")
        if len(self.username) > MAX_USERNAME_LENGTH:
            raise ValueError("username exceeds maximum length.")
        if not isinstance(self.tenant, str) or not self.tenant:
            raise ValueError("tenant must be a non-empty string.")
        if not self.roles or not self.roles <= ROLES:
            raise ValueError("roles must be non-empty and within the allow-list.")
        if len(self.roles) > MAX_ROLES_PER_PRINCIPAL:
            raise ValueError("too many roles.")

    def has_any(self, *required: str) -> bool:
        return bool(set(required) & set(self.roles))


# ============================================================================
# JWT HELPERS (HS256, stdlib)
# ============================================================================

_JWT_ALG = "HS256"


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    pad = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + pad)


def _sign(secret: str, signing_input: str) -> str:
    sig = hmac.new(secret.encode("utf-8"), signing_input.encode("utf-8"), hashlib.sha256).digest()
    return _b64url_encode(sig)


def issue_jwt(secret: str, payload: Dict[str, Any]) -> str:
    header = {"alg": _JWT_ALG, "typ": "JWT"}
    header_b = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b}.{payload_b}"
    return f"{signing_input}.{_sign(secret, signing_input)}"


def decode_jwt(secret: str, token: str) -> Optional[Dict[str, Any]]:
    """Verify signature + exp and return the payload, or None if invalid."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    header_b, payload_b, sig = parts
    if not hmac.compare_digest(_sign(secret, f"{header_b}.{payload_b}"), sig):
        return None
    try:
        header = json.loads(_b64url_decode(header_b))
        if header.get("alg") != _JWT_ALG:
            return None
        payload = json.loads(_b64url_decode(payload_b))
        exp = payload.get("exp")
        if isinstance(exp, (int, float)) and time.time() > exp:
            return None
        return payload
    except Exception:  # noqa: BLE001
        return None


# ============================================================================
# IDENTITY STORE
# ============================================================================


class IdentityStore:
    """In-process user registry + JWT issuance/verification."""

    def __init__(
        self,
        *,
        jwt_secret: Optional[str] = None,
        token_ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._users: Dict[str, Dict[str, Any]] = {}
        self._jwt_secret = jwt_secret or os.urandom(32).hex()
        self._ttl = max(1, min(int(token_ttl_seconds), JWT_MAX_LIFETIME_SECONDS))

    def add_user(
        self,
        username: str,
        password: str,
        *,
        roles: Iterable[str] = (ROLE_VIEWER,),
        tenant: str = "system",
    ) -> Principal:
        roles_set = frozenset(roles or ())
        if not roles_set <= ROLES:
            raise ValueError("roles must be within the allow-list.")
        principal = Principal(username=username, tenant=tenant, roles=roles_set)
        self._users[username] = {
            "password_hash": hash_password(password),
            "principal": principal,
        }
        return principal

    def authenticate(self, username: str, password: str) -> Optional[Principal]:
        record = self._users.get(username)
        if record is None:
            return None
        if not verify_password(password, record["password_hash"]):
            return None
        return record["principal"]

    def issue_token(self, principal: Principal) -> str:
        payload = {
            "sub": principal.username,
            "tenant": principal.tenant,
            "roles": sorted(principal.roles),
            "iat": int(time.time()),
            "exp": int(time.time()) + self._ttl,
        }
        return issue_jwt(self._jwt_secret, payload)

    def decode_token(self, token: str) -> Optional[Principal]:
        payload = decode_jwt(self._jwt_secret, token)
        if not payload:
            return None
        try:
            roles = frozenset(payload.get("roles") or ())
            if not roles <= ROLES or not roles:
                return None
            return Principal(
                username=str(payload.get("sub") or "jwt"),
                tenant=str(payload.get("tenant") or "system"),
                roles=roles,
            )
        except Exception:  # noqa: BLE001
            return None


__all__ = [
    "ROLES",
    "ROLE_ADMIN",
    "ROLE_ANALYST",
    "ROLE_VIEWER",
    "Principal",
    "IdentityStore",
    "hash_password",
    "verify_password",
    "issue_jwt",
    "decode_jwt",
]

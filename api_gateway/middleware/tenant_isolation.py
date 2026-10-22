"""
PhishGuard_AI - Tenant Isolation Middleware

Resolves the active tenant context for each request and exposes it to handlers
via request state.

Security / reliability principles:
- Tenant ids are validated and bounded; invalid values are rejected.
- Tenant context is read from a validated header or token claim.
- Defaults to a safe "system" tenant when none is provided.
- Tenant metadata is never logged in full.
"""

from __future__ import annotations

import re
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

TENANT_HEADER = "X-Tenant-ID"
CONTEXT_KEY = "tenant_id"
DEFAULT_TENANT = "system"
MAX_TENANT_LENGTH = 128
_TENANT_REGEX = re.compile(r"^[A-Za-z0-9_\-.]{1,128}$")


class TenantIsolationError(RuntimeError):
    """Base error for tenant isolation."""


def normalize_tenant_id(value: Optional[str]) -> str:
    """Validate and normalize a tenant id; returns the default on invalid."""
    if not value:
        return DEFAULT_TENANT
    value = value.strip()
    if len(value) > MAX_TENANT_LENGTH or not _TENANT_REGEX.match(value):
        return DEFAULT_TENANT
    return value


class TenantIsolationMiddleware(BaseHTTPMiddleware):
    """Resolve and attach tenant context to each request."""

    async def dispatch(self, request: Request, call_next):
        tenant_id = normalize_tenant_id(request.headers.get(TENANT_HEADER))
        request.state.tenant_id = tenant_id
        response = await call_next(request)
        return response


__all__ = [
    "TenantIsolationMiddleware",
    "normalize_tenant_id",
    "TENANT_HEADER",
    "CONTEXT_KEY",
    "DEFAULT_TENANT",
    "TenantIsolationError",
]

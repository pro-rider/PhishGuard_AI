"""
PhishGuard_AI - Role-Based Access Control (RBAC) Dependencies

FastAPI dependency helpers that enforce role requirements per route. Combined
with the auth middleware (which attaches the resolved ``Principal`` to
``request.state``), this gives fine-grained Admin / Analyst / Viewer control.

Behavior:
- If a ``Principal`` is present, the request is allowed only when the principal
  holds at least one of the required roles; otherwise a 403 is raised.
- If no principal is attached (open / dev mode), the dependency permits the
  request so local operation without credentials keeps working.
"""

from __future__ import annotations

from fastapi import HTTPException, Request
from typing import Callable

from security.identity import Principal

_WWW = "Bearer realm=\"phishguard\", error=\"insufficient_scope\""


def require_roles(*allowed_roles: str) -> Callable[[Request], None]:
    """Build a dependency that requires the caller to hold >= 1 allowed role."""
    allowed = set(allowed_roles)

    def _dependency(request: Request) -> None:
        principal: Principal = getattr(request.state, "principal", None)
        if principal is None:
            # Open / dev mode: no principal attached -> allow.
            return
        if not (principal.roles & allowed):
            raise HTTPException(
                status_code=403,
                detail="insufficient permissions",
                headers={"WWW-Authenticate": _WWW},
            )

    return _dependency


__all__ = ["require_roles"]

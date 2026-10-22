"""
PhishGuard_AI - Correlation ID Middleware

Ensures every request carries a bounded correlation id, generating one when the
caller did not supply a valid id.

Security / reliability principles:
- Incoming correlation ids are validated and re-bounded to prevent abuse.
- The correlation id is available to handlers via request state.
- Logging of the id never leaks into sensitive fields.
"""

from __future__ import annotations

import re
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

HEADER_NAME = "X-Correlation-ID"
CONTEXT_KEY = "correlation_id"
MAX_LENGTH = 128
_CORRELATION_REGEX = re.compile(r"^[A-Za-z0-9_\-.]{1,128}$")

_DEFAULT_GENERATOR = lambda: f"pg_{uuid.uuid4().hex}"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a validated correlation id to every request."""

    def __init__(self, app, *, generator=_DEFAULT_GENERATOR) -> None:
        super().__init__(app)
        self._generator = generator

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get(HEADER_NAME, "")
        correlation_id = self._normalize(incoming)
        if not correlation_id:
            correlation_id = self._generator()

        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers[HEADER_NAME] = correlation_id
        return response

    @staticmethod
    def _normalize(value: str) -> str:
        if not isinstance(value, str):
            return ""
        value = value.strip()
        if not value or len(value) > MAX_LENGTH:
            return ""
        if not _CORRELATION_REGEX.match(value):
            return ""
        return value


__all__ = [
    "CorrelationIdMiddleware",
    "HEADER_NAME",
    "CONTEXT_KEY",
    "MAX_LENGTH",
]

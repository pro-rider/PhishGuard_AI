"""
PhishGuard_AI - Quarantine Release Manager

Manages release requests and approvals for quarantined items, issuing bounded,
single-use release tokens.

Security / reliability principles:
- Release tokens are random, high-entropy, and single-use.
- Approval is required for non-admin releases; approvals are recorded.
- Requests are rate-limited per requester to prevent token mining.
- All inputs are validated and size-bounded.
"""

from __future__ import annotations

import enum
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

MAX_REQUESTS_PER_HOUR_PER_REQUESTER = 20
TOKEN_HEX_BYTES = 32
RELEASE_REQUEST_PREFIX = "rr_"
MAX_NOTE_LENGTH = 512


class ReleaseStatus(str, enum.Enum):
    """Lifecycle of a release request."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    REVOKED = "revoked"


class ReleaseError(RuntimeError):
    """Base error for release management."""


class ReleaseNotFoundError(ReleaseError):
    """Request does not exist."""


class ReleaseNotPendingError(ReleaseError):
    """Request is not in a pending/approvable state."""


class ReleaseRateLimitedError(ReleaseError):
    """Requester exceeded the allowed request rate."""


@dataclass(frozen=True, slots=True)
class ReleaseRequest:
    """A release approval request."""

    request_id: str
    quarantine_id: str
    requester: str
    status: ReleaseStatus
    created_utc: str
    updated_utc: str
    reason: str = ""
    approver: str = ""
    token: str = ""
    used: bool = False
    metadata: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "request_id": self.request_id,
            "quarantine_id": self.quarantine_id,
            "requester": self.requester,
            "status": self.status.value,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "reason": self.reason,
            "approver": self.approver,
            "token_issued": bool(self.token),
            "used": self.used,
        }


# ============================================================================
# MANAGER
# ============================================================================


class ReleaseManager:
    """
    Thread-safe release request manager with single-use tokens and rate limits.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._requests: Dict[str, ReleaseRequest] = {}
        self._request_counts: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        quarantine_id: str,
        requester: str,
        reason: str = "",
    ) -> ReleaseRequest:
        """Create a pending release request for a quarantined item."""
        requester = self._bounded(requester, 128, "requester")
        quarantine_id = self._bounded(quarantine_id, 128, "quarantine_id")
        reason = self._bounded(reason, MAX_NOTE_LENGTH, "reason")

        self._check_rate_limit(requester)
        now = self._now()
        request = ReleaseRequest(
            request_id=RELEASE_REQUEST_PREFIX + uuid.uuid4().hex[:24],
            quarantine_id=quarantine_id,
            requester=requester,
            status=ReleaseStatus.PENDING,
            created_utc=now,
            updated_utc=now,
            reason=reason,
        )
        with self._lock:
            self._requests[request.request_id] = request
        return request

    def approve(
        self, request_id: str, *, approver: str
    ) -> ReleaseRequest:
        """Approve a pending request and issue a single-use token."""
        request = self.get(request_id)
        if request.status is not ReleaseStatus.PENDING:
            raise ReleaseNotPendingError(
                f"Request is {request.status.value}, not pending."
            )
        token = secrets.token_hex(TOKEN_HEX_BYTES)
        updated = self._derive(
            request,
            status=ReleaseStatus.APPROVED,
            approver=self._bounded(approver, 128, "approver"),
            token=token,
        )
        with self._lock:
            self._requests[request_id] = updated
        return updated

    def deny(self, request_id: str, *, approver: str) -> ReleaseRequest:
        request = self.get(request_id)
        if request.status is not ReleaseStatus.PENDING:
            raise ReleaseNotPendingError(f"Request is {request.status.value}.")
        updated = self._derive(
            request,
            status=ReleaseStatus.DENIED,
            approver=self._bounded(approver, 128, "approver"),
        )
        with self._lock:
            self._requests[request_id] = updated
        return updated

    def redeem(
        self, request_id: str, token: str
    ) -> ReleaseRequest:
        """
        Redeem a single-use release token.

        Only an APPROVED, unused request with a matching token can be redeemed.
        A successful redeem marks the token used (single-use semantics).
        """
        request = self.get(request_id)
        if request.status is not ReleaseStatus.APPROVED:
            raise ReleaseNotPendingError(
                f"Request is {request.status.value}; cannot redeem."
            )
        if request.used:
            raise ReleaseError("Release token has already been used.")
        if not secrets.compare_digest(request.token.encode(), token.encode()):
            raise ReleaseError("Invalid release token.")
        updated = self._derive(request, used=True)
        with self._lock:
            self._requests[request_id] = updated
        return updated

    def revoke(self, request_id: str) -> ReleaseRequest:
        request = self.get(request_id)
        if request.status is not ReleaseStatus.APPROVED:
            raise ReleaseNotPendingError(f"Request is {request.status.value}.")
        updated = self._derive(
            request, status=ReleaseStatus.REVOKED, token="", used=True
        )
        with self._lock:
            self._requests[request_id] = updated
        return updated

    def get(self, request_id: str) -> ReleaseRequest:
        with self._lock:
            request = self._requests.get(request_id)
        if request is None:
            raise ReleaseNotFoundError(request_id)
        return request

    def list_requests(
        self, *, status: Optional[ReleaseStatus] = None
    ) -> List[ReleaseRequest]:
        with self._lock:
            requests = [
                r
                for r in self._requests.values()
                if status is None or r.status == status
            ]
        requests.sort(key=lambda r: r.created_utc, reverse=True)
        return requests

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_rate_limit(self, requester: str) -> None:
        now = time.time()
        with self._lock:
            stamps = self._request_counts.setdefault(requester, [])
            stamps[:] = [s for s in stamps if now - s < 3600.0]
            if len(stamps) >= MAX_REQUESTS_PER_HOUR_PER_REQUESTER:
                raise ReleaseRateLimitedError(
                    f"Requester {requester!r} exceeded the hourly request limit."
                )
            stamps.append(now)

    @staticmethod
    def _derive(request: ReleaseRequest, **changes: object) -> ReleaseRequest:
        data = {
            "request_id": request.request_id,
            "quarantine_id": request.quarantine_id,
            "requester": request.requester,
            "status": request.status,
            "created_utc": request.created_utc,
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reason": request.reason,
            "approver": request.approver,
            "token": request.token,
            "used": request.used,
            "metadata": request.metadata,
        }
        data.update(changes)
        return ReleaseRequest(**data)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _bounded(value: str, limit: int, name: str) -> str:
        if not isinstance(value, str):
            raise ReleaseError(f"{name} must be a string.")
        return value[:limit]


# ============================================================================
# SINGLETON
# ============================================================================

release_manager = ReleaseManager()


__all__ = [
    "ReleaseStatus",
    "ReleaseRequest",
    "ReleaseManager",
    "ReleaseError",
    "ReleaseNotFoundError",
    "ReleaseNotPendingError",
    "ReleaseRateLimitedError",
    "release_manager",
]

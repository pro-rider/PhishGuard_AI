"""
PhishGuard_AI - Incident Case Management

Represents and manages security incident cases with a validated, bounded
metadata model and status lifecycle.

Security / reliability principles:
- Case identifiers and tenant identifiers are validated and size-bounded.
- Status transitions are enforced (see ``case_management.workflow``).
- Case fields are length-bounded to prevent log/DB injection and bloat.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Tuple

from case_management.workflow import CaseStatus, CaseWorkflowManager
from storage.repository import SqlAlchemyStore

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

CASE_ID_PREFIX = "case_"
MAX_TITLE_LENGTH = 256
MAX_DESCRIPTION_LENGTH = 16_384
MAX_TAGS = 32
MAX_TAG_LENGTH = 64
MAX_CUSTOM_FIELDS = 32
MAX_FIELD_VALUE = 2048
MAX_CASES_PER_TENANT = 500_000
TENANT_ID_REGEX = __import__("re").compile(r"^[A-Za-z0-9_\-.]{1,128}$")


class CaseError(RuntimeError):
    """Base error for case management."""


class CaseNotFoundError(CaseError):
    """Requested case does not exist."""


class CaseValidationError(CaseError):
    """Case fields failed validation."""


@dataclass(frozen=True, slots=True)
class SecurityCase:
    """A single security incident case."""

    case_id: str
    tenant_id: str
    title: str
    description: str
    status: CaseStatus
    created_utc: str
    updated_utc: str
    assignee: str = ""
    severity: str = "medium"
    tags: Tuple[str, ...] = ()
    custom_fields: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "case_id": self.case_id,
            "tenant_id": self.tenant_id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "assignee": self.assignee,
            "severity": self.severity,
            "tags": list(self.tags),
            "custom_fields": dict(self.custom_fields),
        }


# ============================================================================
# MANAGER
# ============================================================================


class CaseManager:
    """
    Thread-safe case store with enforced status transitions.

    Uses :class:`CaseWorkflowManager` for lifecycle validation so the status
    machine is defined in exactly one place.
    """

    def __init__(
        self,
        *,
        workflow: Optional[CaseWorkflowManager] = None,
        db_path: Optional[str] = None,
    ) -> None:
        self._workflow = workflow or CaseWorkflowManager()
        self._lock = threading.RLock()
        self._cases: Dict[str, SecurityCase] = {}
        # Optional durable backend: when a db_path/dsn is supplied, records are
        # written through to SQLAlchemy (SQLite by default, PostgreSQL when the
        # SQLALCHEMY_DATABASE_URI env is set) and reloaded on startup so they
        # survive process restarts. In-memory dict stays the fast read index.
        self._store = SqlAlchemyStore(db_path) if db_path else None
        if self._store is not None:
            for _case_id, data in self._store.load_all("cases"):
                try:
                    data["status"] = CaseStatus(data["status"])
                    self._cases[_case_id] = SecurityCase(**data)
                except Exception:  # noqa: BLE001 - skip corrupt row, keep rest
                    continue

    def _persist(self, case: SecurityCase) -> None:
        if self._store is not None:
            self._store.upsert("cases", case.case_id, case.to_dict())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        tenant_id: str,
        title: str,
        description: str = "",
        assignee: str = "",
        severity: str = "medium",
        tags: Optional[List[str]] = None,
        custom_fields: Optional[Mapping[str, str]] = None,
    ) -> SecurityCase:
        """Create a new case in the default (open) status."""
        tenant_id = self._validate_tenant(tenant_id)
        title = self._bound(title, MAX_TITLE_LENGTH, "title", required=True)
        description = self._bound(
            description, MAX_DESCRIPTION_LENGTH, "description"
        )
        assignee = self._bound(assignee, 128, "assignee")
        severity = self._bound(severity, 16, "severity", required=True)
        tags = self._sanitize_tags(tags)
        custom_fields = self._sanitize_fields(custom_fields)

        now = self._now()
        case = SecurityCase(
            case_id=CASE_ID_PREFIX + uuid.uuid4().hex[:24],
            tenant_id=tenant_id,
            title=title,
            description=description,
            status=CaseStatus.OPEN,
            created_utc=now,
            updated_utc=now,
            assignee=assignee,
            severity=severity,
            tags=tags,
            custom_fields=custom_fields,
        )

        with self._lock:
            count = sum(
                1 for c in self._cases.values() if c.tenant_id == tenant_id
            )
            if count >= MAX_CASES_PER_TENANT:
                raise CaseError("Per-tenant case limit reached.")
            self._cases[case.case_id] = case
        self._persist(case)
        return case

    def get(self, case_id: str) -> SecurityCase:
        with self._lock:
            case = self._cases.get(case_id)
        if case is None:
            raise CaseNotFoundError(case_id)
        return case

    def list_for_tenant(
        self,
        tenant_id: str,
        *,
        status: Optional[CaseStatus] = None,
    ) -> List[SecurityCase]:
        with self._lock:
            cases = [
                c
                for c in self._cases.values()
                if c.tenant_id == tenant_id
                and (status is None or c.status == status)
            ]
        cases.sort(key=lambda c: c.created_utc, reverse=True)
        return cases

    def transition(self, case_id: str, new_status: CaseStatus) -> SecurityCase:
        """Move a case to a legal next status (validated by the workflow)."""
        case = self.get(case_id)
        if not self._workflow.can_transition(case.status, new_status):
            raise CaseError(
                f"Illegal transition {case.status.value} -> {new_status.value}"
            )
        updated = SecurityCase(
            **{
                **case.to_dict(),
                "status": new_status,
                "updated_utc": self._now(),
            }
        )
        with self._lock:
            self._cases[case_id] = updated
        self._persist(updated)
        return updated

    def update(
        self,
        case_id: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        assignee: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> SecurityCase:
        """Update editable scalar fields on a case."""
        case = self.get(case_id)
        data = case.to_dict()
        if title is not None:
            data["title"] = self._bound(title, MAX_TITLE_LENGTH, "title", required=True)
        if description is not None:
            data["description"] = self._bound(
                description, MAX_DESCRIPTION_LENGTH, "description"
            )
        if assignee is not None:
            data["assignee"] = self._bound(assignee, 128, "assignee")
        if severity is not None:
            data["severity"] = self._bound(severity, 16, "severity", required=True)
        data["status"] = CaseStatus(data["status"])
        data["updated_utc"] = self._now()
        updated = SecurityCase(**data)
        with self._lock:
            self._cases[case_id] = updated
        self._persist(updated)
        return updated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_tenant(tenant_id: str) -> str:
        if not isinstance(tenant_id, str) or not TENANT_ID_REGEX.match(tenant_id):
            raise CaseValidationError(f"Invalid tenant id: {tenant_id!r}")
        return tenant_id

    @staticmethod
    def _bound(
        value: str, limit: int, name: str, *, required: bool = False
    ) -> str:
        if not isinstance(value, str):
            raise CaseValidationError(f"{name} must be a string.")
        if required and not value.strip():
            raise CaseValidationError(f"{name} must not be empty.")
        return value[:limit]

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _sanitize_tags(tags: Optional[List[str]]) -> Tuple[str, ...]:
        if not tags:
            return ()
        result: List[str] = []
        for tag in tags:
            if len(result) >= MAX_TAGS:
                break
            if isinstance(tag, str) and tag.strip():
                result.append(tag.strip()[:MAX_TAG_LENGTH])
        return tuple(result)

    @staticmethod
    def _sanitize_fields(
        fields: Optional[Mapping[str, str]]
    ) -> Mapping[str, str]:
        if not fields:
            return {}
        result: Dict[str, str] = {}
        for key, value in fields.items():
            if len(result) >= MAX_CUSTOM_FIELDS:
                break
            result[str(key)[:128]] = str(value)[:MAX_FIELD_VALUE]
        return result


# ============================================================================
# SINGLETON
# ============================================================================

case_manager = CaseManager()


__all__ = [
    "SecurityCase",
    "CaseManager",
    "CaseError",
    "CaseNotFoundError",
    "CaseValidationError",
    "case_manager",
]

"""
PhishGuard_AI - Case Evidence

Attaches evidence records (IOCs, findings, attachments) to incident cases.

Security / reliability principles:
- Evidence is referenced by a validated storage key and bounded metadata.
- Evidence count per case is bounded.
- No sensitive content is stored in evidence records; only references.
"""

from __future__ import annotations

import enum
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

EVIDENCE_ID_PREFIX = "ev_"
MAX_EVIDENCE_PER_CASE = 1_000
MAX_SOURCE_LENGTH = 128
MAX_STORAGE_KEY_LENGTH = 1024
MAX_METADATA_KEYS = 32


class EvidenceKind(str, enum.Enum):
    """Kind of evidence attached to a case."""

    IOC = "ioc"
    FINDING = "finding"
    ATTACHMENT = "attachment"
    REPORT = "report"
    LOG = "log"


class EvidenceError(RuntimeError):
    """Base error for evidence management."""


class EvidenceNotFoundError(EvidenceError):
    """Requested evidence record does not exist."""


@dataclass(frozen=True, slots=True)
class CaseEvidence:
    """A single evidence record."""

    evidence_id: str
    case_id: str
    kind: EvidenceKind
    source: str
    storage_key: str
    created_utc: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "case_id": self.case_id,
            "kind": self.kind.value,
            "source": self.source,
            "storage_key": self.storage_key,
            "created_utc": self.created_utc,
            "metadata": dict(self.metadata),
        }


class EvidenceManager:
    """Thread-safe evidence store with an optional durable backend."""

    def __init__(self, *, db_path: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._records: Dict[str, CaseEvidence] = {}
        if db_path:
            from storage.repository import SqlAlchemyStore

            self._store = SqlAlchemyStore(db_path)
            for _eid, data in self._store.load_all("case_evidence"):
                try:
                    data["kind"] = EvidenceKind(data["kind"])
                    self._records[_eid] = CaseEvidence(**data)
                except Exception:  # noqa: BLE001 - skip corrupt row
                    continue
        else:
            self._store = None

    def _persist(self, record: CaseEvidence) -> None:
        if self._store is not None:
            self._store.upsert("case_evidence", record.evidence_id, record.to_dict())

    def attach(
        self,
        *,
        case_id: str,
        kind: EvidenceKind,
        source: str,
        storage_key: str,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> CaseEvidence:
        """Attach an evidence reference to a case."""
        case_id = self._bound(case_id, 128, "case_id", required=True)
        source = self._bound(source, MAX_SOURCE_LENGTH, "source", required=True)
        storage_key = self._bound(
            storage_key, MAX_STORAGE_KEY_LENGTH, "storage_key", required=True
        )

        with self._lock:
            count = sum(1 for r in self._records.values() if r.case_id == case_id)
            if count >= MAX_EVIDENCE_PER_CASE:
                raise EvidenceError("Evidence limit reached for this case.")
            record = CaseEvidence(
                evidence_id=EVIDENCE_ID_PREFIX + uuid.uuid4().hex[:20],
                case_id=case_id,
                kind=kind,
                source=source,
                storage_key=storage_key,
                created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                metadata=self._sanitize_metadata(metadata),
            )
            self._records[record.evidence_id] = record
        self._persist(record)
        return record

    def list_for_case(self, case_id: str) -> List[CaseEvidence]:
        with self._lock:
            records = [
                r for r in self._records.values() if r.case_id == case_id
            ]
        records.sort(key=lambda r: r.created_utc)
        return records

    def get(self, evidence_id: str) -> CaseEvidence:
        with self._lock:
            record = self._records.get(evidence_id)
        if record is None:
            raise EvidenceNotFoundError(evidence_id)
        return record

    def delete(self, evidence_id: str) -> bool:
        with self._lock:
            removed = self._records.pop(evidence_id, None) is not None
        if removed and self._store is not None:
            try:
                self._store.delete_record("case_evidence", evidence_id)
            except Exception:  # noqa: BLE001
                pass
        return removed

    @staticmethod
    def _bound(value: str, limit: int, name: str, *, required: bool) -> str:
        if not isinstance(value, str):
            raise EvidenceError(f"{name} must be a string.")
        if required and not value.strip():
            raise EvidenceError(f"{name} must not be empty.")
        return value[:limit]

    @staticmethod
    def _sanitize_metadata(
        metadata: Optional[Mapping[str, str]]
    ) -> Mapping[str, str]:
        if not metadata:
            return {}
        result: Dict[str, str] = {}
        for key, value in metadata.items():
            if len(result) >= MAX_METADATA_KEYS:
                break
            result[str(key)[:128]] = str(value)[:2048]
        return result


evidence_manager = EvidenceManager()


__all__ = [
    "EvidenceKind",
    "CaseEvidence",
    "EvidenceManager",
    "EvidenceError",
    "EvidenceNotFoundError",
    "evidence_manager",
]

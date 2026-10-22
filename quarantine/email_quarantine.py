"""
PhishGuard_AI - Email Quarantine Manager

Stores quarantined messages with full metadata and drives their lifecycle
(quarantined -> approved/released/deleted) in a state-machine-safe way.

Security / reliability principles:
- Message bodies are stored via the object-storage layer and can be encrypted
  at rest through the encryption manager.
- State transitions are strictly validated; illegal transitions are rejected.
- Tenant and message identifiers are validated and bounded.
- No message content is ever logged; only bounded metadata is.
"""

from __future__ import annotations

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from storage.encryption import EncryptionManager
from storage.object_storage import (
    ObjectStorage,
    ObjectNotFoundError,
    object_storage as _default_storage,
)

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

MAX_MESSAGE_ID_LENGTH = 128
MAX_TENANT_ID_LENGTH = 128
MAX_QUARANTINED_MESSAGES = 250_000
MAX_METADATA_KEYS = 32
MAX_METADATA_VALUE = 2048
QUARANTINE_ID_PREFIX = "qn_"


class QuarantineStatus(str, enum.Enum):
    """Lifecycle states of a quarantined message."""

    QUARANTINED = "quarantined"
    RELEASED = "released"
    DELETED = "deleted"
    PURGED = "purged"


_VALID_TRANSITIONS: Mapping[QuarantineStatus, Sequence[QuarantineStatus]] = {
    QuarantineStatus.QUARANTINED: (
        QuarantineStatus.RELEASED,
        QuarantineStatus.DELETED,
    ),
    QuarantineStatus.RELEASED: (QuarantineStatus.DELETED,),
    QuarantineStatus.DELETED: (QuarantineStatus.PURGED,),
    QuarantineStatus.PURGED: (),
}


class QuarantineError(RuntimeError):
    """Base error for the quarantine manager."""


class InvalidTransitionError(QuarantineError):
    """An illegal state transition was attempted."""


class MessageNotFoundError(QuarantineError):
    """Requested message is not quarantined."""


@dataclass(frozen=True, slots=True)
class QuarantinedMessage:
    """Metadata record for a quarantined message."""

    quarantine_id: str
    message_id: str
    tenant_id: str
    status: QuarantineStatus
    created_utc: str
    updated_utc: str
    risk_score: float = 0.0
    reasons: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    storage_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quarantine_id": self.quarantine_id,
            "message_id": self.message_id,
            "tenant_id": self.tenant_id,
            "status": self.status.value,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "risk_score": self.risk_score,
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
            "storage_key": self.storage_key,
        }


# ============================================================================
# MANAGER
# ============================================================================


class EmailQuarantineManager:
    """
    In-process quarantine store backed by object storage for message bodies.

    Production deployments would persist ``QuarantinedMessage`` records in a
    database; this manager keeps an in-memory index and persists payloads to
    the configured object storage backend.
    """

    def __init__(
        self,
        *,
        storage: Optional[ObjectStorage] = None,
        encryption: Optional[EncryptionManager] = None,
    ) -> None:
        self._storage = storage or _default_storage
        if encryption is None:
            # Auto-provision an active in-process key so quarantine works out
            # of the box (records are stored in-memory too, so the key only
            # needs to live for the process lifetime). Callers that want
            # explicit key management can still pass their own manager.
            _enc = EncryptionManager()
            _default_key = EncryptionManager.generate_key("auto_default")
            _enc.add_key(_default_key)
            _enc.activate(_default_key.key_id)
            encryption = _enc
        self._encryption = encryption
        self._lock = threading.RLock()
        self._messages: Dict[str, QuarantinedMessage] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quarantine(
        self,
        *,
        message_id: str,
        tenant_id: str,
        raw_body: bytes,
        risk_score: float = 0.0,
        reasons: Sequence[str] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> QuarantinedMessage:
        """Store a message body and create a quarantine record."""
        message_id = self._bounded(message_id, MAX_MESSAGE_ID_LENGTH, "message_id")
        tenant_id = self._bounded(tenant_id, MAX_TENANT_ID_LENGTH, "tenant_id")
        if not raw_body:
            raise QuarantineError("raw_body must be non-empty.")

        quarantine_id = QUARANTINE_ID_PREFIX + uuid.uuid4().hex[:24]
        now = self._now()
        storage_key = f"{tenant_id}/quarantine/{quarantine_id}.eml"

        # Encrypt the body at rest, then persist to object storage.
        encrypted = self._encryption.encrypt_bytes(raw_body)
        self._storage.put_bytes(storage_key, encrypted)

        record = QuarantinedMessage(
            quarantine_id=quarantine_id,
            message_id=message_id,
            tenant_id=tenant_id,
            status=QuarantineStatus.QUARANTINED,
            created_utc=now,
            updated_utc=now,
            risk_score=self._clamp(risk_score),
            reasons=tuple(str(r)[:256] for r in reasons),
            metadata=self._sanitize_metadata(metadata),
            storage_key=storage_key,
        )

        with self._lock:
            if len(self._messages) >= MAX_QUARANTINED_MESSAGES:
                raise QuarantineError("Quarantine store is full.")
            self._messages[quarantine_id] = record
        return record

    def get(self, quarantine_id: str) -> QuarantinedMessage:
        with self._lock:
            record = self._messages.get(quarantine_id)
        if record is None:
            raise MessageNotFoundError(quarantine_id)
        return record

    def list_for_tenant(
        self, tenant_id: str, *, status: Optional[QuarantineStatus] = None
    ) -> List[QuarantinedMessage]:
        with self._lock:
            records = [
                r
                for r in self._messages.values()
                if r.tenant_id == tenant_id
                and (status is None or r.status == status)
            ]
        records.sort(key=lambda r: r.created_utc, reverse=True)
        return records

    def read_body(self, quarantine_id: str) -> bytes:
        """Retrieve and decrypt the stored message body."""
        record = self.get(quarantine_id)
        try:
            encrypted = self._storage.get_bytes(record.storage_key)
        except ObjectNotFoundError as exc:
            raise MessageNotFoundError(quarantine_id) from exc
        return self._encryption.decrypt_bytes(encrypted)

    def transition(
        self,
        quarantine_id: str,
        new_status: QuarantineStatus,
    ) -> QuarantinedMessage:
        """Move a record to a legal next state."""
        record = self.get(quarantine_id)
        allowed = _VALID_TRANSITIONS[record.status]
        if new_status not in allowed:
            raise InvalidTransitionError(
                f"Illegal transition {record.status.value} -> {new_status.value}"
            )
        updated = QuarantinedMessage(
            **{
                **record.to_dict(),
                "status": new_status,
                "updated_utc": self._now(),
            }
        )
        with self._lock:
            self._messages[quarantine_id] = updated
        if new_status in (QuarantineStatus.PURGED,):
            self._prune_body(record)
        return updated

    def release(self, quarantine_id: str) -> QuarantinedMessage:
        return self.transition(quarantine_id, QuarantineStatus.RELEASED)

    def delete(self, quarantine_id: str) -> QuarantinedMessage:
        return self.transition(quarantine_id, QuarantineStatus.DELETED)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prune_body(self, record: QuarantinedMessage) -> None:
        try:
            self._storage.delete(record.storage_key)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _bounded(value: str, limit: int, name: str) -> str:
        if not isinstance(value, str) or not value:
            raise QuarantineError(f"{name} must be a non-empty string.")
        return value[:limit]

    @staticmethod
    def _clamp(value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        return min(1.0, max(0.0, value))

    @staticmethod
    def _sanitize_metadata(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if not metadata:
            return result
        for key, value in metadata.items():
            if len(result) >= MAX_METADATA_KEYS:
                break
            if isinstance(value, str) and len(value) > MAX_METADATA_VALUE:
                value = value[:MAX_METADATA_VALUE]
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[str(key)[:128]] = value
        return result


# ============================================================================
# SINGLETON
# ============================================================================

email_quarantine = EmailQuarantineManager()


__all__ = [
    "QuarantineStatus",
    "QuarantinedMessage",
    "EmailQuarantineManager",
    "QuarantineError",
    "InvalidTransitionError",
    "MessageNotFoundError",
    "email_quarantine",
]

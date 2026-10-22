"""
PhishGuard_AI - SQLAlchemy Persistence Store

Drop-in replacement for the stdlib-SQLite adapter (``storage.persistence``),
backed by SQLAlchemy so the same interface works against SQLite (default) or
PostgreSQL (production) without changing the managers.

Exposes the exact methods the managers already use (``upsert``, ``load_all``,
``set_meta``, ``get_meta``, ``close``) plus a generic ``load`` and ``append``
for scan jobs and the audit log. Rows are JSON blobs in ``(id, data)`` tables,
so the in-memory dicts remain the fast read index.

Security:
- Table names are validated against an allow-list; every value is parameterized.
- The DB file / connection is the durable, access-restricted store; production
  should point ``SQLALCHEMY_DATABASE_URI`` at PostgreSQL + Vault/KMS keys.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select

from storage import database
from storage.models import (
    AuditLogRecord,
    CaseCommentRecord,
    CaseEvidenceRecord,
    CaseRecord,
    DATA_TABLES,
    MetaRecord,
    QuarantineRecord,
    ScanJobRecord,
)

logger = logging.getLogger(__name__)

# Map a logical table name -> ORM model.
_MODELS = {
    "cases": CaseRecord,
    "quarantine": QuarantineRecord,
    "scan_jobs": ScanJobRecord,
    "audit_log": AuditLogRecord,
    "case_comments": CaseCommentRecord,
    "case_evidence": CaseEvidenceRecord,
}


class SqlAlchemyStore:
    """Thread-safe durable KV store keyed by record id, JSON-serialized rows."""

    def __init__(self, path_or_dsn: Optional[str] = None) -> None:
        database.init_db(path_or_dsn)
        self._factory = database.get_session_factory(path_or_dsn)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Row persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _check_table(table: str) -> None:
        if table not in DATA_TABLES:
            raise ValueError(f"Unsupported persistence table: {table!r}")

    def upsert(self, table: str, record_id: str, data: Dict[str, Any]) -> None:
        self._check_table(table)
        model = _MODELS[table]
        blob = json.dumps(data, ensure_ascii=False)
        with self._lock:
            with self._factory() as session:
                existing = session.get(model, record_id)
                if existing is not None:
                    existing.data = blob
                else:
                    session.add(model(id=record_id, data=blob))
                session.commit()

    def load_all(self, table: str) -> List[Tuple[str, Dict[str, Any]]]:
        self._check_table(table)
        model = _MODELS[table]
        with self._lock:
            with self._factory() as session:
                rows = session.execute(select(model)).scalars().all()
        return [(r.id, json.loads(r.data)) for r in rows]

    def load(self, table: str, record_id: str) -> Optional[Dict[str, Any]]:
        self._check_table(table)
        model = _MODELS[table]
        with self._lock:
            with self._factory() as session:
                row = session.get(model, record_id)
        return json.loads(row.data) if row is not None else None

    def delete_record(self, table: str, record_id: str) -> None:
        """Remove a single record (used for delete-able child entities)."""
        self._check_table(table)
        model = _MODELS[table]
        with self._lock:
            with self._factory() as session:
                row = session.get(model, record_id)
                if row is not None:
                    session.delete(row)
                    session.commit()

    # ------------------------------------------------------------------
    # Key-value metadata (e.g. persisted encryption key id / material)
    # ------------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            with self._factory() as session:
                existing = session.get(MetaRecord, key)
                if existing is not None:
                    existing.value = value
                else:
                    session.add(MetaRecord(key=key, value=value))
                session.commit()

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            with self._factory() as session:
                row = session.get(MetaRecord, key)
        return row.value if row is not None else None

    def close(self) -> None:
        with self._lock:
            engine = database.get_engine()
            try:
                engine.dispose()
            except Exception as exc:  # noqa: BLE001
                logger.debug("engine dispose: %s", exc)

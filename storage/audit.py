"""
PhishGuard_AI - Audit Log Manager

Immutable, append-only audit trail of security-relevant actions
(user, action, target, timestamp, IP). Records are persisted and never updated
or deleted, so operators have a tamper-evident history of who did what.

The in-memory row is keyed by a generated id; the durable backend stores the
full row as JSON. Only ``record`` (append) and ``list`` (read) are exposed —
there is deliberately no update/delete path.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from storage.repository import SqlAlchemyStore

AUDIT_TABLE = "audit_log"
MAX_DETAIL_LENGTH = 8192
MAX_TARGET_ID_LENGTH = 256


class AuditManager:
    """Append-only, durable audit trail."""

    def __init__(self, store: Optional[SqlAlchemyStore] = None) -> None:
        self._store = store or SqlAlchemyStore()

    def record(
        self,
        *,
        tenant_id: str,
        actor: str,
        action: str,
        target_type: str,
        target_id: str = "",
        ip: Optional[str] = None,
        detail: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Append one immutable audit record and return it."""
        row = {
            "audit_id": f"aud_{uuid.uuid4().hex[:20]}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tenant_id": str(tenant_id)[:128],
            "actor": str(actor)[:128],
            "action": str(action)[:128],
            "target_type": str(target_type)[:64],
            "target_id": str(target_id)[:MAX_TARGET_ID_LENGTH],
            "ip": str(ip or "")[:64],
            "detail": _json_serialize(detail)[:MAX_DETAIL_LENGTH],
        }
        self._store.upsert(AUDIT_TABLE, row["audit_id"], row)
        return row

    def list(self, *, limit: int = 200, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = [data for _, data in self._store.load_all(AUDIT_TABLE)]
        if tenant_id:
            rows = [r for r in rows if r.get("tenant_id") == tenant_id]
        rows.sort(key=lambda r: r.get("timestamp_utc", ""), reverse=True)
        return rows[: max(0, limit)]


def _json_serialize(detail: Any) -> str:
    import json

    if detail is None:
        return ""
    try:
        return json.dumps(detail, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(detail)


__all__ = ["AuditManager", "AUDIT_TABLE"]

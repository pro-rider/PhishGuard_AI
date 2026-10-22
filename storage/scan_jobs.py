"""
PhishGuard_AI - Async Scan Job Manager

Durable lifecycle for long-running scans: ``pending -> running -> completed |
failed``. Jobs (and any incremental progress events) are persisted so operators
can poll ``GET /scan/jobs/{id}`` or subscribe to SSE on
``GET /scan/jobs/{id}/events`` even across restarts.

The result (verdict, score, case_created) is stored on the completed job so a
finished async scan produces the exact same payload as the synchronous ``/scan``.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from storage.repository import SqlAlchemyStore

JOB_TABLE = "scan_jobs"

# ============================================================================
# STATUS CONSTANTS
# ============================================================================

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
_STATUSES = {STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED}


class ScanJobError(RuntimeError):
    """Base error for scan job management."""


class ScanJobNotFoundError(ScanJobError):
    """Requested job does not exist."""


class ScanJobManager:
    """Durable, thread-safe store for async scan jobs."""

    def __init__(self, store: Optional[SqlAlchemyStore] = None) -> None:
        self._store = store or SqlAlchemyStore()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        tenant_id: str,
        actor: str,
        request_summary: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a job in ``pending`` state and return it."""
        now = self._now()
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        job = {
            "job_id": job_id,
            "status": STATUS_PENDING,
            "tenant_id": tenant_id,
            "actor": actor,
            "correlation_id": correlation_id,
            "created_utc": now,
            "started_utc": None,
            "completed_utc": None,
            "request": request_summary,
            "events": [{"ts": now, "type": "created", "message": "job queued"}],
            "result": None,
            "error": None,
        }
        self._upsert(job)
        return job

    def start(self, job_id: str) -> None:
        self._patch(job_id, {"status": STATUS_RUNNING, "started_utc": self._now()})

    def complete(self, job_id: str, result: Dict[str, Any]) -> None:
        self._patch(
            job_id,
            {
                "status": STATUS_COMPLETED,
                "completed_utc": self._now(),
                "result": result,
                "error": None,
            },
        )

    def fail(self, job_id: str, error: str) -> None:
        self._patch(
            job_id,
            {"status": STATUS_FAILED, "completed_utc": self._now(), "error": str(error)[:4000]},
        )

    def append_event(self, job_id: str, type_: str, message: str) -> None:
        with self._lock:
            job = self.get(job_id)
            events = list(job.get("events") or [])
            events.append({"ts": self._now(), "type": type_, "message": message})
            job["events"] = events[-200:]  # bounded ring buffer
            self._upsert(job)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, job_id: str) -> Dict[str, Any]:
        row = self._store.load(JOB_TABLE, job_id)
        if row is None:
            raise ScanJobNotFoundError(job_id)
        return row

    def list(self, tenant_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        jobs = [data for _, data in self._store.load_all(JOB_TABLE)]
        if tenant_id:
            jobs = [j for j in jobs if j.get("tenant_id") == tenant_id]
        jobs.sort(key=lambda j: j.get("created_utc", ""), reverse=True)
        return jobs[: max(0, limit)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _upsert(self, job: Dict[str, Any]) -> None:
        self._store.upsert(JOB_TABLE, job["job_id"], job)

    def _patch(self, job_id: str, patch: Dict[str, Any]) -> None:
        with self._lock:
            job = self.get(job_id)
            job.update(patch)
            self._upsert(job)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "ScanJobManager",
    "ScanJobNotFoundError",
    "ScanJobError",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
]

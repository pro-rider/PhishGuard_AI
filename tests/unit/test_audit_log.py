"""
Unit tests for security.audit_log.AuditLogger.
"""

from __future__ import annotations

import pytest

from security.audit_log import (
    AuditLogger,
    AuditOutcome,
    AuditSeverity,
    AuditWriteStatus,
)


class _NullStorage:
    """In-memory storage that records writes for assertions."""

    def __init__(self) -> None:
        self.writes = []

    def write(self, record) -> object:
        self.writes.append(record)
        return type(
            "WriteResult",
            (),
            {"status": AuditWriteStatus.SUCCESS, "error_message": None},
        )()


@pytest.mark.unit
class TestAuditLogger:
    def test_record_returns_audit_record(self) -> None:
        logger = AuditLogger(service="test-svc", environment="test", storage=None)
        record = logger.record(
            event="scan.completed",
            outcome=AuditOutcome.SUCCESS,
            severity=AuditSeverity.INFO,
            metadata={"rule_id": "R-001"},
        )
        assert record.event_name == "scan.completed"
        payload = record.to_dict()
        assert "event_name" in payload
        assert payload.get("event_name") == "scan.completed"

    def test_record_security_event(self) -> None:
        logger = AuditLogger(service="test-svc", environment="test", storage=None)
        record = logger.record_security_event(
            event="auth.failed",
            actor_id="user-1",
            metadata={"attempt": 3},
        )
        assert record.actor_id == "user-1"

    def test_record_persists_to_storage(self) -> None:
        storage = _NullStorage()
        logger = AuditLogger(service="test-svc", environment="test", storage=storage)
        logger.record(event="scan.started", outcome="success")
        assert len(storage.writes) == 1

"""
PhishGuard_AI - Case Workflow State Machine

Single source of truth for valid case status transitions.

Security / reliability principles:
- Transitions are enumerated explicitly; unknown transitions are rejected.
- The state machine is pure and deterministic (no I/O).
- ``CaseStatus`` is shared with the case model so the lifecycle is defined
  exactly once.
"""

from __future__ import annotations

import enum
from typing import Dict, Optional, Sequence

# ============================================================================
# STATUSES & TRANSITIONS
# ============================================================================


class CaseStatus(str, enum.Enum):
    """Lifecycle status of an incident case."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ARCHIVED = "archived"


_TRANSITIONS: Dict[CaseStatus, Sequence[CaseStatus]] = {
    CaseStatus.OPEN: (CaseStatus.IN_PROGRESS, CaseStatus.ON_HOLD, CaseStatus.CLOSED),
    CaseStatus.IN_PROGRESS: (
        CaseStatus.ON_HOLD,
        CaseStatus.RESOLVED,
        CaseStatus.CLOSED,
    ),
    CaseStatus.ON_HOLD: (CaseStatus.IN_PROGRESS, CaseStatus.CLOSED),
    CaseStatus.RESOLVED: (CaseStatus.CLOSED, CaseStatus.IN_PROGRESS),
    CaseStatus.CLOSED: (CaseStatus.ARCHIVED, CaseStatus.OPEN),
    CaseStatus.ARCHIVED: (),
}

# Terminal / final statuses.
TERMINAL_STATUSES = frozenset({CaseStatus.CLOSED, CaseStatus.ARCHIVED})

# Statuses reachable from any (except archived).
_OPENING_STATUSES = frozenset(
    {CaseStatus.OPEN, CaseStatus.IN_PROGRESS, CaseStatus.ON_HOLD}
)


class CaseWorkflowError(RuntimeError):
    """Base error for workflow operations."""


class CaseWorkflowManager:
    """
    Validates case status transitions against the defined state machine.
    """

    def can_transition(
        self, current: CaseStatus, target: CaseStatus
    ) -> bool:
        """Return whether ``current -> target`` is a legal transition."""
        if not isinstance(current, CaseStatus) or not isinstance(
            target, CaseStatus
        ):
            return False
        allowed = _TRANSITIONS.get(current, ())
        return target in allowed

    def assert_transition(
        self, current: CaseStatus, target: CaseStatus
    ) -> None:
        """Raise if the transition is illegal."""
        if not self.can_transition(current, target):
            raise CaseWorkflowError(
                f"Illegal case transition {current.value} -> {target.value}"
            )

    def next_statuses(self, current: CaseStatus) -> Sequence[CaseStatus]:
        """Return the legal next statuses for ``current``."""
        return tuple(_TRANSITIONS.get(current, ()))

    def is_terminal(self, status: CaseStatus) -> bool:
        return status in TERMINAL_STATUSES

    def is_open(self, status: CaseStatus) -> bool:
        return status in _OPENING_STATUSES


case_workflow = CaseWorkflowManager()


__all__ = [
    "CaseStatus",
    "CaseWorkflowManager",
    "CaseWorkflowError",
    "TERMINAL_STATUSES",
    "case_workflow",
]

"""
PhishGuard_AI - Quarantine Retention Policy

Defines retention rules and computes expiry for quarantined items so that
storage remains bounded and legal holds are respected.

Security / reliability principles:
- Retention is expressed in whole days with validated bounds.
- Legal holds override expiry and must be explicit.
- Policy is deterministic and pure (no I/O); it only computes outcomes.
- Defaults are conservative (shorter retention for high-severity content).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

MIN_RETENTION_DAYS = 1
MAX_RETENTION_DAYS = 3650  # ~10 years
DEFAULT_RETENTION_DAYS = 90

# Maps risk tiers (0.0-1.0) to default retention in days.
_DEFAULT_TIERS = (
    (0.0, 14),    # benign / review
    (0.30, 45),   # low
    (0.60, 90),   # medium
    (0.85, 180),  # high
)


class RetentionAction(str, enum.Enum):
    """Outcome computed by the retention policy for an item."""

    KEEP = "keep"
    PURGE = "purge"
    NOTIFY = "notify"


class RetentionPolicyError(RuntimeError):
    """Base error for retention policy."""


class RetentionPolicy(BaseModel):
    """Validation-safe retention configuration."""

    model_config = ConfigDict(extra="forbid")

    default_days: int = Field(
        default=DEFAULT_RETENTION_DAYS,
        ge=MIN_RETENTION_DAYS,
        le=MAX_RETENTION_DAYS,
    )
    # Optional tier overrides: list of (risk_threshold, days).
    tiers: list = Field(default_factory=lambda: list(_DEFAULT_TIERS))

    def model_post_init(self, __context) -> None:
        tiers = []
        for threshold, days in self.tiers:
            threshold = float(threshold)
            days = int(days)
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("Tier threshold out of range.")
            if not (MIN_RETENTION_DAYS <= days <= MAX_RETENTION_DAYS):
                raise ValueError("Tier retention out of range.")
            tiers.append((threshold, days))
        tiers.sort(key=lambda t: t[0], reverse=True)
        object.__setattr__(self, "tiers", tiers)


@dataclass(frozen=True, slots=True)
class RetentionAssessment:
    """Result of a retention policy evaluation for a single item."""

    quarantine_id: str
    action: RetentionAction
    expires_utc: Optional[str]
    retention_days: int
    reason: str

    def to_dict(self) -> dict:
        return {
            "quarantine_id": self.quarantine_id,
            "action": self.action.value,
            "expires_utc": self.expires_utc,
            "retention_days": self.retention_days,
            "reason": self.reason,
        }


class RetentionPolicyEngine:
    """
    Pure, deterministic retention computation.

    Evaluates a quarantined item against the configured tiers, producing either
    a concrete expiry (KEEP) or a purge decision.
    """

    def __init__(
        self,
        *,
        policy: Optional[RetentionPolicy] = None,
        now: Optional[datetime] = None,
    ) -> None:
        self.policy = policy or RetentionPolicy()
        self._now = now or datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        *,
        quarantine_id: str,
        created_utc: str,
        risk_score: float = 0.0,
        legal_hold: bool = False,
        purge_requested: bool = False,
    ) -> RetentionAssessment:
        """Compute the retention outcome for a single quarantined item."""
        quarantine_id = str(quarantine_id)[:128]
        created = self._parse_created(created_utc)
        days = self._retention_days(risk_score)

        if purge_requested:
            return RetentionAssessment(
                quarantine_id=quarantine_id,
                action=RetentionAction.PURGE,
                expires_utc=None,
                retention_days=days,
                reason="explicit purge requested",
            )

        if legal_hold:
            return RetentionAssessment(
                quarantine_id=quarantine_id,
                action=RetentionAction.KEEP,
                expires_utc=None,
                retention_days=days,
                reason="legal hold active",
            )

        expiry = created + timedelta(days=days)
        if expiry <= self._now:
            return RetentionAssessment(
                quarantine_id=quarantine_id,
                action=RetentionAction.PURGE,
                expires_utc=None,
                retention_days=days,
                reason="retention period elapsed",
            )

        near_expiry = (expiry - self._now) <= timedelta(days=3)
        return RetentionAssessment(
            quarantine_id=quarantine_id,
            action=RetentionAction.NOTIFY if near_expiry else RetentionAction.KEEP,
            expires_utc=expiry.isoformat(timespec="seconds"),
            retention_days=days,
            reason="retention active",
        )

    def assess_many(
        self, items: list
    ) -> list:
        """Assess a batch of quarantined items; returns retention outcomes."""
        results = []
        for item in items:
            try:
                results.append(
                    self.assess(
                        quarantine_id=item.get("quarantine_id", ""),
                        created_utc=item.get("created_utc", self._now.isoformat()),
                        risk_score=float(item.get("risk_score", 0.0)),
                        legal_hold=bool(item.get("legal_hold", False)),
                        purge_requested=bool(item.get("purge_requested", False)),
                    )
                )
            except Exception:  # noqa: BLE001 - batch resilience
                continue
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retention_days(self, risk_score: float) -> int:
        try:
            risk_score = float(risk_score)
        except (TypeError, ValueError):
            risk_score = 0.0
        risk_score = min(1.0, max(0.0, risk_score))
        for threshold, days in self.policy.tiers:
            if risk_score >= threshold:
                return days
        return self.policy.default_days

    @staticmethod
    def _parse_created(value: str) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.now(timezone.utc) - timedelta(days=1)


# ============================================================================
# SINGLETON
# ============================================================================

retention_policy = RetentionPolicyEngine()


__all__ = [
    "RetentionAction",
    "RetentionPolicy",
    "RetentionAssessment",
    "RetentionPolicyEngine",
    "RetentionPolicyError",
    "retention_policy",
]

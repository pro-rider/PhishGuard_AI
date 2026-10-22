"""
PhishGuard_AI - Policy Engine

Maps a normalized risk score and optional findings to a disposition decision
(allow / review / quarantine / block). Policies are threshold-based, bounded,
and tenant-aware.

Security / reliability principles:
- Fail-closed: any unhandled anomaly maps to ``block``.
- Thresholds are validated and monotonic; boundaries are explicit.
- Decisions carry a machine-readable reason and are audit-friendly.
- The engine is pure and deterministic (no I/O, no side effects).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# ENUMS
# ============================================================================


class Disposition(str, enum.Enum):
    """Final disposition produced by the policy engine."""

    ALLOW = "allow"
    REVIEW = "review"
    QUARANTINE = "quarantine"
    BLOCK = "block"


class PolicyAction(str, enum.Enum):
    """A concrete action the caller should take for a disposition."""

    DELIVER = "deliver"
    REVIEW_MANUALLY = "review_manually"
    HOLD_FOR_QUARANTINE = "hold_for_quarantine"
    REJECT = "reject"
    QUARANTINE_AND_ALERT = "quarantine_and_alert"


# ============================================================================
# MODELS
# ============================================================================


class PolicyConfig(BaseModel):
    """Threshold configuration mapping risk score to disposition."""

    model_config = ConfigDict(extra="forbid")

    # Risk score is expected in [0.0, 1.0].
    block_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    quarantine_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    review_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    block_on_high_confidence_blocklist: bool = True
    quarantine_on_unverified_attachment: bool = True
    default_disposition: Disposition = Disposition.REVIEW

    def model_post_init(self, __context: Any) -> None:
        # Enforce monotonic thresholds: block > quarantine > review.
        if not (
            self.review_threshold
            <= self.quarantine_threshold
            <= self.block_threshold
        ):
            raise ValueError(
                "Thresholds must satisfy review <= quarantine <= block."
            )


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """The decision produced for a given risk assessment."""

    disposition: Disposition
    action: PolicyAction
    reason: str
    risk_score: float
    matched_policy: str = "default"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "disposition": self.disposition.value,
            "action": self.action.value,
            "reason": self.reason,
            "risk_score": self.risk_score,
            "matched_policy": self.matched_policy,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# ENGINE
# ============================================================================


class PolicyEngine:
    """
    Deterministic, fail-closed policy engine.

    The engine never raises on inputs; instead it returns a conservative
    ``BLOCK`` decision when inputs are malformed or out of range.
    """

    def __init__(
        self,
        *,
        config: Optional[PolicyConfig] = None,
    ) -> None:
        self.config = config or PolicyConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        risk_score: float,
        blocklisted: bool = False,
        blocklist_confidence: float = 1.0,
        has_unverified_attachment: bool = False,
        finding_count: int = 0,
        entity_kind: str = "email",
    ) -> PolicyDecision:
        """Evaluate a risk assessment to a policy decision."""
        score = self._clamp_risk(risk_score)

        # Fail-closed on NaN.
        if score != score:
            return self._block("risk_score is not a number")

        # Explicit blocklist wins when enabled and confident.
        if (
            self.config.block_on_high_confidence_blocklist
            and blocklisted
            and blocklist_confidence >= 0.8
        ):
            return self._block("high-confidence blocklist match")

        # Attachment policy.
        if (
            self.config.quarantine_on_unverified_attachment
            and has_unverified_attachment
            and score >= self.config.review_threshold
        ):
            return self._quarantine("unverified attachment present")

        if score >= self.config.block_threshold:
            return self._block("risk score exceeds block threshold")
        if score >= self.config.quarantine_threshold:
            return self._quarantine("risk score exceeds quarantine threshold")
        if score >= self.config.review_threshold:
            return self._review("risk score exceeds review threshold")

        return PolicyDecision(
            disposition=Disposition.ALLOW,
            action=PolicyAction.DELIVER,
            reason="risk score within acceptable bounds",
            risk_score=score,
            metadata={"entity_kind": entity_kind, "finding_count": int(finding_count)},
        )

    def evaluate_many(
        self, assessments: Sequence[Mapping[str, Any]]
    ) -> List[PolicyDecision]:
        """Evaluate a batch of risk assessments."""
        return [
            self.evaluate(
                risk_score=float(a.get("risk_score", 0.0)),
                blocklisted=bool(a.get("blocklisted", False)),
                blocklist_confidence=float(a.get("blocklist_confidence", 1.0)),
                has_unverified_attachment=bool(
                    a.get("has_unverified_attachment", False)
                ),
                finding_count=int(a.get("finding_count", 0)),
                entity_kind=str(a.get("entity_kind", "email")),
            )
            for a in assessments
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_risk(value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float("nan")
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _block(self, reason: str) -> PolicyDecision:
        return PolicyDecision(
            disposition=Disposition.BLOCK,
            action=PolicyAction.REJECT,
            reason=reason,
            risk_score=1.0,
            matched_policy="block",
        )

    def _quarantine(self, reason: str) -> PolicyDecision:
        return PolicyDecision(
            disposition=Disposition.QUARANTINE,
            action=PolicyAction.QUARANTINE_AND_ALERT,
            reason=reason,
            risk_score=self.config.quarantine_threshold,
            matched_policy="quarantine",
        )

    def _review(self, reason: str) -> PolicyDecision:
        return PolicyDecision(
            disposition=Disposition.REVIEW,
            action=PolicyAction.REVIEW_MANUALLY,
            reason=reason,
            risk_score=self.config.review_threshold,
            matched_policy="review",
        )


# ============================================================================
# SINGLETON
# ============================================================================

default_policy_engine = PolicyEngine()


__all__ = [
    "Disposition",
    "PolicyAction",
    "PolicyConfig",
    "PolicyDecision",
    "PolicyEngine",
    "default_policy_engine",
]

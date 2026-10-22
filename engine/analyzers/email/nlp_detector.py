"""
PhishGuard_AI - NLP Detector Analyzer

Linguistic / heuristic analysis of email body text for social-engineering
indicators (urgency, credential bait, spoofed identity, link pressure).

Security / reliability principles:
- Works purely on bounded in-memory text; no external calls by default.
- Rule signals are deterministic and bounded in count/length.
- Results are surfaced as structured findings with a bounded score contribution.
"""

from __future__ import annotations

import re
import time
from typing import List, Optional, Set, Tuple

from engine.orchestrator import (
    AnalyzerExecutionStatus,
    AnalyzerResourceClass,
    AnalyzerResult,
    BaseAnalyzer,
    ConstrainedMetadata,
    Finding,
    ScanContext,
    ScanType,
    SecurityVerdict,
)

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

ANALYZER_NAME = "nlp_detector"
ANALYZER_VERSION = "1.1.0"
MAX_BODY_CHARS = 512_000
MAX_FINDINGS = 20
SCORE_CAP = 100.0

_URGENCY_TERMS = (
    "immediately", "asap", "urgent", "act now", "right away", "within 24 hours",
    "account suspended", "will be deactivated", "expires today", "final notice",
    "overdue", "action required", "your account will be closed",
)
_CREDENTIAL_BAIT = (
    "verify your password", "confirm your password", "update your account",
    "enter your credentials", "click to verify", "re-enter your password",
    "unlock your account", "reset your password", "verify identity",
)
_PRESSURING_TERMS = (
    "failure to respond", "will result in", "legal action", "lawsuit",
    "authorities", "police report", "fraud investigation", "if you do not act",
)
_MONEY_TERMS = (
    "lottery", "inheritance", "wire transfer", "western union", "bitcoin",
    "cryptocurrency", "you have won", "prize", "refund", "gift card",
)

_URL_PRESSURE = re.compile(r"(click|follow|open)\s+(?:the\s+)?(?:this\s+)?link", re.I)
_LINK_TAG = re.compile(r"https?://\S+", re.I)


class NLPDetectorAnalyzer(BaseAnalyzer):
    """Heuristic NLP detector for email body text."""

    name: str = ANALYZER_NAME
    version: str = ANALYZER_VERSION
    supported_scan_types: Set[ScanType] = {
        ScanType.EMAIL,
        ScanType.COMPREHENSIVE,
    }
    required_inputs: Set[str] = {"email"}
    resource_class: AnalyzerResourceClass = AnalyzerResourceClass.ASYNC_IO

    async def analyze(
        self, context: ScanContext, deadline_monotonic: float
    ) -> AnalyzerResult:
        start = time.monotonic()
        text = self._resolve_body(context)
        if text is None:
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SKIPPED,
                execution_time_ms=0.0,
                score_contribution=0.0,
                sanitized_error="email body text unavailable",
            )

        findings: List[Finding] = []
        score = 0.0
        norm = text[:MAX_BODY_CHARS].lower()

        score += self._score_lexicon(
            norm, _URGENCY_TERMS, "urgency", findings, 0.20
        )
        score += self._score_lexicon(
            norm, _CREDENTIAL_BAIT, "credential_bait", findings, 0.25
        )
        score += self._score_lexicon(
            norm, _PRESSURING_TERMS, "pressure", findings, 0.20
        )
        score += self._score_lexicon(
            norm, _MONEY_TERMS, "money_bait", findings, 0.15
        )

        if _URL_PRESSURE.search(text[:MAX_BODY_CHARS]):
            score += 0.15
            findings.append(
                self._finding(
                    "nlp.link_pressure",
                    "message pressures clicking a link",
                    0.7,
                )
            )

        link_count = len(_LINK_TAG.findall(text[:MAX_BODY_CHARS]))
        if link_count >= 2:
            score += 0.10
            findings.append(
                self._finding(
                    "nlp.multiple_links",
                    f"message contains {link_count} links",
                    0.6,
                )
            )

        findings = findings[:MAX_FINDINGS]
        contribution = min(score, 1.0) * SCORE_CAP
        execution_ms = (time.monotonic() - start) * 1000.0

        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.SUCCESS,
            execution_time_ms=execution_ms,
            score_contribution=contribution,
            confidence=self._confidence(contribution),
            findings=findings,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_body(context: ScanContext) -> Optional[str]:
        if context.email_payload is None:
            return None
        storage_key = context.email_payload.storage_key
        if storage_key:
            try:
                from storage.object_storage import object_storage

                raw = object_storage.get_bytes(storage_key)
                return raw.decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                return None
        metadata = context.metadata.data or {}
        return str(metadata.get("email_text") or "")

    def _score_lexicon(
        self,
        norm: str,
        terms: Tuple[str, ...],
        category: str,
        findings: List[Finding],
        weight: float,
    ) -> float:
        hits = [t for t in terms if t in norm]
        if hits:
            findings.append(
                self._finding(
                    f"nlp.{category}",
                    f"matched terms: {', '.join(hits[:3])}",
                    0.75,
                )
            )
            return weight
        return 0.0

    @staticmethod
    def _finding(rule_id: str, message: str, confidence: float) -> Finding:
        return Finding(
            rule_id=rule_id,
            category="nlp_social_engineering",
            severity=SecurityVerdict.MEDIUM,
            confidence=confidence,
            message=message,
            metadata=ConstrainedMetadata(),
        )

    @staticmethod
    def _confidence(contribution: float) -> float:
        return min(1.0, max(0.0, contribution / SCORE_CAP))


nlp_detector = NLPDetectorAnalyzer()


__all__ = ["NLPDetectorAnalyzer", "nlp_detector"]

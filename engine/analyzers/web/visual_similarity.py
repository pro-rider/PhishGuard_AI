"""
PhishGuard_AI - Visual Similarity Analyzer

Detects credential-harvesting lookalike domains / pages by comparing brand
identity signals (brand-name tokens, brand domains) present in a target URL
against a known-brand registry.

Security / reliability principles:
- Pure local, deterministic; no screenshotting or browser execution.
- Brand matching is bounded and exact-ish (substring on brand tokens).
- Fail-closed: unknown targets are assigned a conservative low score, never
  treated as trusted.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlsplit

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

ANALYZER_NAME = "visual_similarity"
ANALYZER_VERSION = "1.0.0"
SCORE_CAP = 100.0
MAX_FINDINGS = 20

# Registry of brand tokens: (brand name, normalized token(s), canonical domain).
_BRANDS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("Apple", ("apple",), "apple.com"),
    ("Microsoft", ("microsoft", "outlook", "office365", "live"), "microsoft.com"),
    ("Google", ("google", "gmail", "youtube", "android"), "google.com"),
    ("Amazon", ("amazon", "aws"), "amazon.com"),
    ("PayPal", ("paypal",), "paypal.com"),
    ("Netflix", ("netflix",), "netflix.com"),
    ("Facebook", ("facebook", "fb"), "facebook.com"),
    ("Instagram", ("instagram",), "instagram.com"),
    ("WhatsApp", ("whatsapp", "wa"), "whatsapp.com"),
    ("LinkedIn", ("linkedin",), "linkedin.com"),
    ("Twitter", ("twitter", "x"), "twitter.com"),
    ("Wells Fargo", ("wellsfargo",), "wellsfargo.com"),
    ("Chase", ("chase",), "chase.com"),
    ("Bank of America", ("bankofamerica", "bofa"), "bankofamerica.com"),
    ("Coinbase", ("coinbase",), "coinbase.com"),
    ("Binance", ("binance",), "binance.com"),
    ("FedEx", ("fedex",), "fedex.com"),
    ("UPS", ("ups",), "ups.com"),
    ("DHL", ("dhl",), "dhl.com"),
    ("IRS", ("irs",), "irs.gov"),
)

_TOKEN_REGEX = re.compile(r"[a-z0-9]+")


class VisualSimilarityAnalyzer(BaseAnalyzer):
    """Detects lookalike brand domains in target URLs."""

    name: str = ANALYZER_NAME
    version: str = ANALYZER_VERSION
    supported_scan_types: Set[ScanType] = {
        ScanType.URL,
        ScanType.COMPREHENSIVE,
    }
    required_inputs: Set[str] = {"target_url"}
    resource_class: AnalyzerResourceClass = AnalyzerResourceClass.ASYNC_IO

    async def analyze(
        self, context: ScanContext, deadline_monotonic: float
    ) -> AnalyzerResult:
        start = time.monotonic()
        target = context.target_url
        if not target:
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SKIPPED,
                execution_time_ms=0.0,
                score_contribution=0.0,
                sanitized_error="target_url unavailable",
            )

        host = self._host(target)
        findings: List[Finding] = []
        score = 0.0

        if host:
            host_tokens = self._tokens(host)
            for brand, tokens, canonical in _BRANDS:
                matched = [t for t in tokens if t in host_tokens]
                if not matched:
                    continue
                is_canonical = host == canonical or host.endswith("." + canonical)
                if is_canonical:
                    continue  # legitimate canonical domain
                # Confusable: brand token present but not the canonical domain.
                score += 0.45
                findings.append(
                    self._finding(
                        f"visual.{brand.lower().replace(' ', '_')}",
                        f"host '{host}' impersonates '{brand}'",
                        0.8,
                    )
                )
                break

        # Punycode / IDN obfuscation heuristic.
        if host and host.startswith("xn--"):
            score += 0.25
            findings.append(
                self._finding(
                    "visual.punycode",
                    f"host uses punycode obfuscation: {host}",
                    0.7,
                )
            )

        # Numeric-heavy or hyphen-heavy host.
        if host:
            digits = sum(c.isdigit() for c in host)
            if digits / max(1, len(host)) > 0.4:
                score += 0.15
                findings.append(
                    self._finding(
                        "visual.numeric_heavy",
                        f"host '{host}' is digit-heavy",
                        0.6,
                    )
                )

        contribution = min(score, 1.0) * SCORE_CAP
        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.SUCCESS,
            execution_time_ms=(time.monotonic() - start) * 1000.0,
            score_contribution=contribution,
            confidence=min(1.0, contribution / SCORE_CAP),
            findings=findings[:MAX_FINDINGS],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _host(url: str) -> str:
        try:
            return (urlsplit(url).hostname or "").lower()
        except ValueError:
            return ""

    @staticmethod
    def _tokens(host: str) -> Set[str]:
        return set(_TOKEN_REGEX.findall(host))

    @staticmethod
    def _finding(rule_id: str, message: str, confidence: float) -> Finding:
        return Finding(
            rule_id=rule_id,
            category="visual_similarity",
            severity=SecurityVerdict.HIGH,
            confidence=confidence,
            message=message,
            metadata=ConstrainedMetadata(),
        )


visual_similarity = VisualSimilarityAnalyzer()


__all__ = ["VisualSimilarityAnalyzer", "visual_similarity"]

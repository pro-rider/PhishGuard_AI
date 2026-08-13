"""
PhishGuard_AI - Threat Scoring & Risk Engine

Transforms aggregated analyzer signals, findings, and confidence metrics from 
the Detection Orchestrator into calibrated, deterministic security scores and verdicts.

Key Architectural & Security Controls:
--------------------------------------
1. Evidence Family Correlation & Family Caps: Groups correlated sub-analyzers
   (e.g., Domain, DNS, URL) into evidence families and enforces strict contribution caps
   to prevent noisy subsystems from dominating the verdict.
2. Single-Role Mathematical Calibration: Eliminates double-application of weights and
   confidence exponents using normalized, bounded probabilistic dampening per family.
3. Explicit Hard Override Allowlist: Replaces untrusted analyzer metadata flags with
   a strict, rule-id-based trusted override registry.
4. Signal Contradiction & Authentication Model: Evaluates discrepancies between sender
   infrastructure authenticity (SPF/DKIM) and payload threat indicators, applying score
   adjustments and confidence penalties.
5. Analysis Coverage & Completeness Metric: Quantifies confidence bounds based on the ratio
   of executed analyzers versus expected analyzers, penalizing incomplete scans.
6. Cryptographic Provenance & Lineage: Generates SHA-256 hashes of canonical scoring input
   trees for forensic auditability and SIEM integration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from engine.orchestrator import (
    AnalyzerExecutionStatus,
    AnalyzerResult,
    Finding,
    PipelineExecutionState,
    PipelineResult,
    ScanType,
    SecurityVerdict,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & SCORING VERSIONS
# ============================================================================

SCORING_ALGORITHM_VERSION = "2.0.0"
SCORING_POLICY_VERSION = "2026.08.1"

MAX_RISK_FACTORS = 20
MAX_RISK_FACTOR_MSG_LEN = 256


def is_finite_number(val: Any) -> bool:
    """Validate that a numeric value is real and finite (not NaN or Inf)."""
    return isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val)


# ============================================================================
# ENUMERATIONS & TAXONOMY
# ============================================================================


class EvidenceFamily(str, Enum):
    """Correlated taxonomy families preventing single-vector score inflation."""

    EMAIL_AUTHENTICATION = "email_authentication"
    URL_STRUCTURE = "url_structure"
    DOMAIN_REPUTATION = "domain_reputation"
    WEB_CONTENT = "web_content"
    ATTACHMENT_ANALYSIS = "attachment_analysis"
    MALWARE_SIGNATURE = "malware_signature"
    SANDBOX_EXECUTION = "sandbox_execution"


# Maximum risk score allowed per evidence family
DEFAULT_FAMILY_CAPS: Dict[EvidenceFamily, float] = {
    EvidenceFamily.EMAIL_AUTHENTICATION: 30.0,
    EvidenceFamily.URL_STRUCTURE: 35.0,
    EvidenceFamily.DOMAIN_REPUTATION: 35.0,
    EvidenceFamily.WEB_CONTENT: 50.0,
    EvidenceFamily.ATTACHMENT_ANALYSIS: 60.0,
    EvidenceFamily.MALWARE_SIGNATURE: 100.0,
    EvidenceFamily.SANDBOX_EXECUTION: 100.0,
}

# Mapping analyzers to their primary evidence family
ANALYZER_FAMILY_MAP: Dict[str, EvidenceFamily] = {
    "header_analyzer": EvidenceFamily.EMAIL_AUTHENTICATION,
    "body_analyzer": EvidenceFamily.WEB_CONTENT,
    "nlp_detector": EvidenceFamily.WEB_CONTENT,
    "attachment_detector": EvidenceFamily.ATTACHMENT_ANALYSIS,
    "url_parser": EvidenceFamily.URL_STRUCTURE,
    "domain_analyzer": EvidenceFamily.DOMAIN_REPUTATION,
    "dns_analyzer": EvidenceFamily.DOMAIN_REPUTATION,
    "ssl_checker": EvidenceFamily.DOMAIN_REPUTATION,
    "secure_fetcher": EvidenceFamily.WEB_CONTENT,
    "dom_analyzer": EvidenceFamily.WEB_CONTENT,
    "visual_similarity": EvidenceFamily.WEB_CONTENT,
    "browser_sandbox": EvidenceFamily.SANDBOX_EXECUTION,
    "zip_bomb_guard": EvidenceFamily.ATTACHMENT_ANALYSIS,
    "pdf_analyzer": EvidenceFamily.ATTACHMENT_ANALYSIS,
    "macro_deobfuscator": EvidenceFamily.ATTACHMENT_ANALYSIS,
    "yara_scanner": EvidenceFamily.MALWARE_SIGNATURE,
}

DEFAULT_VERDICT_THRESHOLDS: Dict[SecurityVerdict, float] = {
    SecurityVerdict.CRITICAL: 80.0,
    SecurityVerdict.HIGH: 60.0,
    SecurityVerdict.MEDIUM: 40.0,
    SecurityVerdict.LOW: 20.0,
}


# ============================================================================
# TRUSTED HARD OVERRIDE REGISTRY
# ============================================================================


class TrustedHardOverrideRegistry:
    """Explicit registry of authorized rule IDs permitted to trigger hard overrides."""

    _TRUSTED_RULE_IDS: Set[str] = {
        "YARA.MALWARE.CONFIRMED",
        "SANDBOX.PAYLOAD.EXECUTED",
        "DOM.CREDENTIAL_HARVEST_KIT",
        "MACRO.MALICIOUS_DROPPER",
        "ZIP.BOMB_EXPLOIT",
    }

    @classmethod
    def is_authorized(cls, rule_id: str) -> bool:
        return rule_id in cls._TRUSTED_RULE_IDS


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True, slots=True)
class FamilyScoreContribution:
    """Aggregated score contribution for a correlated evidence family."""

    family: EvidenceFamily
    uncapped_score: float
    capped_score: float
    family_cap: float
    contributing_analyzers: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScoringBreakdown:
    """Immutable audit breakdown capturing complete scoring lineage."""

    provenance_hash: str
    algorithm_version: str
    policy_version: str
    raw_accumulated_score: float
    final_score: float
    overall_confidence: float
    coverage_score: float
    family_contributions: Tuple[FamilyScoreContribution, ...]
    hard_overrides: Tuple[str, ...]
    conflicts_detected: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ThreatAssessment:
    """Final Output produced by ThreatScoringEngine."""

    scan_id: str
    verdict: SecurityVerdict
    score: float
    confidence: float
    coverage: float
    is_malicious: bool
    breakdown: ScoringBreakdown
    risk_factors: Tuple[str, ...] = field(default_factory=tuple)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Immutable, validated runtime configuration for threat scoring."""

    family_caps: Mapping[EvidenceFamily, float] = field(
        default_factory=lambda: MappingProxyType(DEFAULT_FAMILY_CAPS)
    )
    verdict_thresholds: Mapping[SecurityVerdict, float] = field(
        default_factory=lambda: MappingProxyType(DEFAULT_VERDICT_THRESHOLDS)
    )
    enable_contradiction_penalties: bool = True
    contradiction_confidence_penalty: float = 0.20
    contradiction_score_penalty: float = 15.0
    fail_closed_on_total_failure: bool = True

    def __post_init__(self) -> None:
        # Validate ordering: CRITICAL > HIGH > MEDIUM > LOW
        crit = self.verdict_thresholds[SecurityVerdict.CRITICAL]
        high = self.verdict_thresholds[SecurityVerdict.HIGH]
        med = self.verdict_thresholds[SecurityVerdict.MEDIUM]
        low = self.verdict_thresholds[SecurityVerdict.LOW]

        if not (100.0 >= crit > high > med > low >= 0.0):
            raise ValueError("Verdict thresholds must follow strict descending order: CRITICAL > HIGH > MEDIUM > LOW.")

        for family, cap in self.family_caps.items():
            if not is_finite_number(cap) or not (0.0 <= cap <= 100.0):
                raise ValueError(f"Family cap for {family} must be finite number between 0 and 100.")


# ============================================================================
# THREAT SCORING ENGINE
# ============================================================================


class ThreatScoringEngine:
    """Production threat scoring engine with family correlation and provenance tracking."""

    def __init__(self, config: Optional[ScoringConfig] = None) -> None:
        self.config = config or ScoringConfig()

    def evaluate(self, pipeline_result: PipelineResult) -> ThreatAssessment:
        """Evaluate PipelineResult and produce a deterministic ThreatAssessment."""
        # 1. Handle pipeline execution failure or fail-closed state
        if (
            pipeline_result.execution_state == PipelineExecutionState.FAILED
            and self.config.fail_closed_on_total_failure
        ):
            return self._build_fail_closed_assessment(pipeline_result, "Pipeline execution failed completely.")

        # 2. Extract findings and classify into evidence families
        family_scores, contributing_analyzers = self._aggregate_by_family(pipeline_result)

        # 3. Apply family caps and compute family-level contributions
        family_contributions: List[FamilyScoreContribution] = []
        total_dampened_score = 0.0

        for family, raw_score in family_scores.items():
            cap = self.config.family_caps.get(family, 100.0)
            capped_score = min(raw_score, cap)
            family_contributions.append(
                FamilyScoreContribution(
                    family=family,
                    uncapped_score=round(raw_score, 2),
                    capped_score=round(capped_score, 2),
                    family_cap=cap,
                    contributing_analyzers=tuple(contributing_analyzers.get(family, [])),
                )
            )
            total_dampened_score += capped_score

        # 4. Check for trusted hard overrides
        hard_overrides = self._extract_authorized_hard_overrides(pipeline_result)

        # 5. Detect signal contradictions
        conflicts = self._detect_conflicts(pipeline_result)

        # 6. Apply contradiction penalties
        score_penalty = 0.0
        confidence_penalty = 0.0
        if conflicts and self.config.enable_contradiction_penalties:
            score_penalty = self.config.contradiction_score_penalty
            confidence_penalty = self.config.contradiction_confidence_penalty

        # 7. Calculate coverage score (analysis completeness)
        coverage_score = self._calculate_coverage(pipeline_result)

        # 8. Compute final score & verdict
        if hard_overrides:
            final_score = 100.0
            verdict = SecurityVerdict.CRITICAL
        else:
            adjusted_score = total_dampened_score + score_penalty
            final_score = min(max(adjusted_score, 0.0), 100.0)
            verdict = self._score_to_verdict(final_score)

        # Calibrate overall confidence
        base_confidence = pipeline_result.confidence_score * coverage_score
        final_confidence = min(max(base_confidence - confidence_penalty, 0.0), 1.0)

        # Extract sanitized risk factors
        risk_factors = self._extract_risk_factors(pipeline_result)

        # 9. Build cryptographic scoring lineage
        breakdown = self._build_breakdown(
            scan_id=pipeline_result.scan_id,
            raw_accumulated_score=round(total_dampened_score, 2),
            final_score=round(final_score, 2),
            overall_confidence=round(final_confidence, 2),
            coverage_score=round(coverage_score, 2),
            family_contributions=family_contributions,
            hard_overrides=hard_overrides,
            conflicts=conflicts,
        )

        is_malicious = verdict in (SecurityVerdict.HIGH, SecurityVerdict.CRITICAL)

        return ThreatAssessment(
            scan_id=pipeline_result.scan_id,
            verdict=verdict,
            score=round(final_score, 2),
            confidence=round(final_confidence, 2),
            coverage=round(coverage_score, 2),
            is_malicious=is_malicious,
            breakdown=breakdown,
            risk_factors=tuple(risk_factors),
        )

    # ========================================================================
    # INTERNAL CALCULATIONS & CORRELATION
    # ========================================================================

    def _aggregate_by_family(
        self, pipeline_result: PipelineResult
    ) -> Tuple[Dict[EvidenceFamily, float], Dict[EvidenceFamily, List[str]]]:
        """Groups analyzer scores into evidence families using non-linear dampening per family."""
        family_analyzer_scores: Dict[EvidenceFamily, List[float]] = {f: [] for f in EvidenceFamily}
        family_analyzers: Dict[EvidenceFamily, List[str]] = {f: [] for f in EvidenceFamily}

        for name, res in pipeline_result.analyzer_results.items():
            if res.status != AnalyzerExecutionStatus.SUCCESS:
                continue

            family = ANALYZER_FAMILY_MAP.get(name, EvidenceFamily.WEB_CONTENT)
            # Effective signal contribution = score * confidence
            effective_signal = res.score_contribution * res.confidence

            family_analyzer_scores[family].append(effective_signal)
            family_analyzers[family].append(name)

        family_scores: Dict[EvidenceFamily, float] = {}
        for family, scores in family_analyzer_scores.items():
            if not scores:
                family_scores[family] = 0.0
                continue

            # Family non-linear probabilistic combination: S = 100 * (1 - PROD(1 - s_i/100))
            prob_clean = 1.0
            for s in scores:
                norm_s = min(max(s / 100.0, 0.0), 1.0)
                prob_clean *= (1.0 - (norm_s * 0.90))

            family_scores[family] = (1.0 - prob_clean) * 100.0

        return family_scores, family_analyzers

    def _extract_authorized_hard_overrides(self, pipeline_result: PipelineResult) -> List[str]:
        """Inspects findings against the explicit TrustedHardOverrideRegistry."""
        authorized_overrides: List[str] = []

        for name, res in pipeline_result.analyzer_results.items():
            if res.status != AnalyzerExecutionStatus.SUCCESS:
                continue

            for finding in res.findings:
                if TrustedHardOverrideRegistry.is_authorized(finding.rule_id):
                    override_entry = f"[{name}] Authorized Hard Override: {finding.rule_id}"
                    if override_entry not in authorized_overrides:
                        authorized_overrides.append(override_entry)

        return authorized_overrides

    def _detect_conflicts(self, pipeline_result: PipelineResult) -> List[str]:
        """Detects contradictions between authentication signals and threat payloads."""
        conflicts: List[str] = []
        has_auth_pass = False
        has_malicious_payload = False

        header_res = pipeline_result.analyzer_results.get("header_analyzer")
        if header_res and header_res.status == AnalyzerExecutionStatus.SUCCESS:
            for finding in header_res.findings:
                if finding.rule_id in ("SPF_PASS", "DKIM_PASS", "DMARC_PASS"):
                    has_auth_pass = True

        yara_res = pipeline_result.analyzer_results.get("yara_scanner")
        dom_res = pipeline_result.analyzer_results.get("dom_analyzer")

        if (yara_res and yara_res.score_contribution > 60.0) or (
            dom_res and dom_res.score_contribution > 60.0
        ):
            has_malicious_payload = True

        if has_auth_pass and has_malicious_payload:
            conflicts.append("CONFLICT: Valid email authentication combined with high-risk payload.")

        return conflicts

    def _calculate_coverage(self, pipeline_result: PipelineResult) -> float:
        """Calculates analysis completeness score (ratio of executed vs expected analyzers)."""
        total_analyzers = len(pipeline_result.analyzer_results)
        if total_analyzers == 0:
            return 0.0

        successful_count = sum(
            1 for res in pipeline_result.analyzer_results.values()
            if res.status == AnalyzerExecutionStatus.SUCCESS
        )

        return successful_count / total_analyzers

    def _score_to_verdict(self, score: float) -> SecurityVerdict:
        if score >= self.config.verdict_thresholds[SecurityVerdict.CRITICAL]:
            return SecurityVerdict.CRITICAL
        elif score >= self.config.verdict_thresholds[SecurityVerdict.HIGH]:
            return SecurityVerdict.HIGH
        elif score >= self.config.verdict_thresholds[SecurityVerdict.MEDIUM]:
            return SecurityVerdict.MEDIUM
        elif score >= self.config.verdict_thresholds[SecurityVerdict.LOW]:
            return SecurityVerdict.LOW
        return SecurityVerdict.INFORMATIONAL

    def _extract_risk_factors(self, pipeline_result: PipelineResult) -> List[str]:
        """Extracts sanitized, deduplicated risk factors bounded in length."""
        risk_factors: List[str] = []

        for name, res in pipeline_result.analyzer_results.items():
            if res.status != AnalyzerExecutionStatus.SUCCESS:
                continue

            for finding in res.findings:
                if finding.severity in (SecurityVerdict.HIGH, SecurityVerdict.CRITICAL):
                    msg = f"[{name}] {finding.rule_id}: {finding.message[:MAX_RISK_FACTOR_MSG_LEN]}"
                    if msg not in risk_factors:
                        risk_factors.append(msg)

                if len(risk_factors) >= MAX_RISK_FACTORS:
                    break

        return risk_factors

    def _build_breakdown(
        self,
        scan_id: str,
        raw_accumulated_score: float,
        final_score: float,
        overall_confidence: float,
        coverage_score: float,
        family_contributions: List[FamilyScoreContribution],
        hard_overrides: List[str],
        conflicts: List[str],
    ) -> ScoringBreakdown:
        """Generates an immutable ScoringBreakdown with SHA-256 provenance lineage."""
        lineage_payload = {
            "algorithm_version": SCORING_ALGORITHM_VERSION,
            "policy_version": SCORING_POLICY_VERSION,
            "scan_id": scan_id,
            "raw_accumulated_score": raw_accumulated_score,
            "final_score": final_score,
            "overall_confidence": overall_confidence,
            "coverage_score": coverage_score,
            "family_contributions": [
                {
                    "family": fc.family.value,
                    "uncapped": fc.uncapped_score,
                    "capped": fc.capped_score,
                    "cap": fc.family_cap,
                    "analyzers": list(fc.contributing_analyzers),
                }
                for fc in family_contributions
            ],
            "hard_overrides": hard_overrides,
            "conflicts": conflicts,
        }

        canonical_json = json.dumps(lineage_payload, sort_keys=True)
        provenance_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

        return ScoringBreakdown(
            provenance_hash=provenance_hash,
            algorithm_version=SCORING_ALGORITHM_VERSION,
            policy_version=SCORING_POLICY_VERSION,
            raw_accumulated_score=raw_accumulated_score,
            final_score=final_score,
            overall_confidence=overall_confidence,
            coverage_score=coverage_score,
            family_contributions=tuple(family_contributions),
            hard_overrides=tuple(hard_overrides),
            conflicts_detected=tuple(conflicts),
        )

    def _build_fail_closed_assessment(
        self, pipeline_result: PipelineResult, reason: str
    ) -> ThreatAssessment:
        breakdown = self._build_breakdown(
            scan_id=pipeline_result.scan_id,
            raw_accumulated_score=100.0,
            final_score=100.0,
            overall_confidence=0.0,
            coverage_score=0.0,
            family_contributions=[],
            hard_overrides=["FAIL-CLOSED TRIGGERED"],
            conflicts=[],
        )

        return ThreatAssessment(
            scan_id=pipeline_result.scan_id,
            verdict=SecurityVerdict.CRITICAL,
            score=100.0,
            confidence=0.0,
            coverage=0.0,
            is_malicious=True,
            breakdown=breakdown,
            risk_factors=(f"FAIL-CLOSED: {reason}",),
        )


# ============================================================================
# EXPORTS
# ============================================================================

scoring_engine = ThreatScoringEngine()

__all__ = [
    "ANALYZER_FAMILY_MAP",
    "DEFAULT_FAMILY_CAPS",
    "DEFAULT_VERDICT_THRESHOLDS",
    "EvidenceFamily",
    "FamilyScoreContribution",
    "SCORING_ALGORITHM_VERSION",
    "SCORING_POLICY_VERSION",
    "ScoringBreakdown",
    "ScoringConfig",
    "ThreatAssessment",
    "ThreatScoringEngine",
    "TrustedHardOverrideRegistry",
    "scoring_engine",
]
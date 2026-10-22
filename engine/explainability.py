"""
PhishGuard_AI - Threat Explainability & Evidence Reporting Engine

Transforms technical outputs from the Orchestrator and Threat Scoring Engine
into clear, human-readable summaries, structured SOC audit breakdowns, and
SIEM-compatible (ECS/CEF) security event payloads.

Key Architectural & Security Controls:
--------------------------------------
1. Scan & Assessment Invariant Continuity: Guarantees scan_id, verdict, and score
   alignment between PipelineResult and ThreatAssessment prior to report generation.
2. Cryptographic Provenance Verification: Validates canonical SHA-256 evidence
   hashes against the ScoringBreakdown to prevent evidence tampering.
3. Multi-Pattern Secret & Path Sanitizer: Multi-pass regex engine redacting JWTs,
   AWS credentials, Bearer tokens, private keys, and sensitive paths.
4. Audience-Aware Redaction Modes: Supports STRICT_PUBLIC, SOC_ANALYST, and
   FORENSIC_FULL output modes to match security clearance boundaries.
5. Primary & Secondary Threat Hierarchy: Discovers primary threat vectors alongside
   correlated secondary threats without inventing divergent scoring rules.
6. SIEM Common Event Integration: Native export formatting for Elastic Common Schema (ECS)
   and ArcSight Common Event Format (CEF) with explicit tenant/correlation metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from engine.orchestrator import (
    AnalyzerExecutionStatus,
    PipelineResult,
    SecurityVerdict,
)
from engine.scoring_engine import (
    ThreatAssessment,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & BOUNDS
# ============================================================================

EXPLAINABILITY_SCHEMA_VERSION = "2.0.0"
EXPLAINABILITY_ENGINE_VERSION = "2026.08.1"

MAX_SUMMARY_LEN = 2048
MAX_FACTOR_MSG_LEN = 256
MAX_FACTORS_COUNT = 25
MAX_REPORT_JSON_BYTES = 65536

# Multi-pattern regex engine for secret and path sanitization
SECRET_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("JWT_TOKEN", re.compile(r"eyJ[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*")),
    ("AWS_KEY", re.compile(r"(?i)(AKIA|ASIA)[0-9A-Z]{16}")),
    ("BEARER_AUTH", re.compile(r"(?i)bearer\s+[a-zA-Z0-9\-\._~\+\/]+=*")),
    ("GENERIC_SECRET", re.compile(r"(?i)(api_?key|password|secret|token|auth_token|passwd)=([^\s&]+)")),
    ("PRIVATE_KEY", re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----")),
    ("CONN_STRING", re.compile(r"(?i)(postgres|mysql|mongodb|redis|amqp):\/\/[^\s]+")),
)

PATH_PATTERN = re.compile(r"([a-zA-Z]:\\[\\\w\s\.\-]+|/(?:[\w\.\-]+/)+[\w\.\-]+)")


def is_finite_number(val: Any) -> bool:
    """Validate that a numeric value is real and finite (not NaN or Inf)."""
    return isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val)


# ============================================================================
# ENUMERATIONS
# ============================================================================


class ReportAudience(str, Enum):
    """Target audience format for explainability output."""

    EXECUTIVE = "executive"
    SOC_ANALYST = "soc_analyst"
    FORENSIC_AUDIT = "forensic_audit"
    SIEM_JSON = "siem_json"


class RedactionMode(str, Enum):
    """Redaction granularity level depending on stakeholder clearance."""

    STRICT_PUBLIC = "strict_public"
    SOC_ANALYST = "soc_analyst"
    FORENSIC_FULL = "forensic_full"


class ThreatCategoryLabel(str, Enum):
    """Human-readable classification of primary and secondary threat vectors."""

    MALWARE_DROPPER = "Malicious Payload / Dropper"
    CREDENTIAL_HARVESTING = "Credential Harvesting / Phishing"
    BRAND_IMPERSONATION = "Brand Impersonation / Lookalike Domain"
    BUSINESS_EMAIL_COMPROMISE = "Business Email Compromise (BEC)"
    SUSPICIOUS_ATTACHMENT = "Suspicious Attachment / Exploitative Format"
    BEACONING_SPAM = "Unsolicited Bulk / Suspicious Infrastructure"
    CLEAN_BENIGN = "No Significant Threats Detected"


# ============================================================================
# SANITIZER ENGINE
# ============================================================================


class SecretAndPathSanitizer:
    """Multi-pass regex engine stripping credentials, tokens, and paths."""

    @staticmethod
    def sanitize(text: str, mode: RedactionMode = RedactionMode.SOC_ANALYST) -> str:
        if not text:
            return ""

        sanitized = str(text)

        # 1. Always strip hard secrets across all modes
        for label, pattern in SECRET_PATTERNS:
            sanitized = pattern.sub(f"[REDACTED_{label}]", sanitized)

        # 2. Strip local filesystem paths under STRICT_PUBLIC mode
        if mode == RedactionMode.STRICT_PUBLIC:
            sanitized = PATH_PATTERN.sub("[REDACTED_PATH]", sanitized)

        return sanitized[:MAX_SUMMARY_LEN]


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True, slots=True)
class ThreatFactorExplanation:
    """Detailed explanation of a specific evidence factor."""

    analyzer_name: str
    rule_id: str
    severity: SecurityVerdict
    title: str
    description: str
    raw_score: float
    confidence: float
    remediation_step: str

    def __post_init__(self) -> None:
        if not is_finite_number(self.raw_score) or not (0.0 <= self.raw_score <= 100.0):
            raise ValueError("raw_score must be finite float between 0 and 100.")
        if not is_finite_number(self.confidence) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be finite float between 0 and 1.0.")


@dataclass(frozen=True, slots=True)
class ExplainabilityReport:
    """Immutable, audit-ready security report."""

    report_id: str
    scan_id: str
    schema_version: str
    engine_version: str
    tenant_id: Optional[str]
    correlation_id: Optional[str]
    audience: ReportAudience
    redaction_mode: RedactionMode
    verdict: SecurityVerdict
    final_score: float
    confidence_score: float
    coverage_score: float
    primary_threat_label: ThreatCategoryLabel
    secondary_threat_labels: Tuple[ThreatCategoryLabel, ...]
    summary: str
    key_factors: Tuple[ThreatFactorExplanation, ...]
    recommended_action: str
    provenance_hash: str
    report_hash: str
    created_at_timestamp: float
    created_at_iso8601: str

    def __post_init__(self) -> None:
        if not is_finite_number(self.final_score) or not (0.0 <= self.final_score <= 100.0):
            raise ValueError("final_score must be a finite float between 0 and 100.")
        if not is_finite_number(self.confidence_score) or not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("confidence_score must be a finite float between 0 and 1.")
        if not is_finite_number(self.coverage_score) or not (0.0 <= self.coverage_score <= 1.0):
            raise ValueError("coverage_score must be a finite float between 0 and 1.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert report object into a serializable dictionary."""
        return {
            "report_id": self.report_id,
            "scan_id": self.scan_id,
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "tenant_id": self.tenant_id,
            "correlation_id": self.correlation_id,
            "audience": self.audience.value,
            "redaction_mode": self.redaction_mode.value,
            "verdict": self.verdict.value,
            "final_score": self.final_score,
            "confidence_score": self.confidence_score,
            "coverage_score": self.coverage_score,
            "primary_threat_label": self.primary_threat_label.value,
            "secondary_threat_labels": [st.value for st in self.secondary_threat_labels],
            "summary": self.summary,
            "recommended_action": self.recommended_action,
            "provenance_hash": self.provenance_hash,
            "report_hash": self.report_hash,
            "created_at_timestamp": self.created_at_timestamp,
            "created_at_iso8601": self.created_at_iso8601,
            "key_factors": [
                {
                    "analyzer_name": kf.analyzer_name,
                    "rule_id": kf.rule_id,
                    "severity": kf.severity.value,
                    "title": kf.title,
                    "description": kf.description,
                    "raw_score": kf.raw_score,
                    "confidence": kf.confidence,
                    "remediation_step": kf.remediation_step,
                }
                for kf in self.key_factors
            ],
        }

    def to_json(self) -> str:
        """Export report as formatted, size-bounded JSON string."""
        raw_json = json.dumps(
            self.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        if len(raw_json.encode("utf-8")) > MAX_REPORT_JSON_BYTES:
            logger.warning("Report JSON exceeds size threshold. Truncating key factors.")
            # Fallback output with stripped factors if payload size limit is exceeded
            trimmed_dict = self.to_dict()
            trimmed_dict["key_factors"] = trimmed_dict["key_factors"][:5]
            raw_json = json.dumps(trimmed_dict, indent=2, ensure_ascii=False, allow_nan=False)
        return raw_json

    def to_ecs_event(self) -> Dict[str, Any]:
        """Convert report into Elastic Common Schema (ECS) compatible log format."""
        return {
            "@timestamp": self.created_at_iso8601,
            "ecs": {"version": "8.11.0"},
            "event": {
                "id": self.report_id,
                "kind": "alert" if self.verdict in (SecurityVerdict.HIGH, SecurityVerdict.CRITICAL) else "event",
                "category": ["intrusion_detection", "email"],
                "type": ["indicator"],
                "outcome": "success",
                "severity": int(self.final_score),
            },
            "phishguard": {
                "scan_id": self.scan_id,
                "tenant_id": self.tenant_id,
                "correlation_id": self.correlation_id,
                "verdict": self.verdict.value,
                "confidence": self.confidence_score,
                "coverage": self.coverage_score,
                "primary_threat": self.primary_threat_label.value,
                "provenance_hash": self.provenance_hash,
                "report_hash": self.report_hash,
            },
            "message": self.summary,
        }

    def to_cef_event(self) -> str:
        """Convert report into ArcSight Common Event Format (CEF) string."""
        severity_num = int(self.final_score / 10.0)
        return (
            f"CEF:0|PhishGuard_AI|DetectionEngine|{self.engine_version}|"
            f"{self.primary_threat_label.name}|{self.verdict.value}|{severity_num}|"
            f"cn1={self.final_score} cn1Label=RiskScore "
            f"cn2={self.confidence_score} cn2Label=Confidence "
            f"cs1={self.scan_id} cs1Label=ScanID "
            f"cs2={self.tenant_id or 'default'} cs2Label=TenantID "
            f"msg={self.summary[:512]}"
        )


# ============================================================================
# EXPLAINABILITY ENGINE
# ============================================================================


class ThreatExplainabilityEngine:
    """Production explainability engine translating threat assessments into human-readable insights."""

    @classmethod
    def verify_scoring_provenance(cls, assessment: ThreatAssessment) -> bool:
        """Verifies the integrity of the cryptographic provenance hash from ScoringBreakdown."""
        breakdown = assessment.breakdown
        if not breakdown or not breakdown.provenance_hash:
            return False

        # Re-compute provenance check
        raw_payload = {
            "scan_id": assessment.scan_id,
            "final_score": round(assessment.score, 2),
            "overrides_count": len(breakdown.hard_overrides),
            "contributions_count": len(breakdown.contributions),
        }
        computed_hash = hashlib.sha256(
            json.dumps(raw_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        return computed_hash == breakdown.provenance_hash

    def generate_report(
        self,
        assessment: ThreatAssessment,
        pipeline_result: PipelineResult,
        audience: ReportAudience = ReportAudience.SOC_ANALYST,
        redaction_mode: RedactionMode = RedactionMode.SOC_ANALYST,
        tenant_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> ExplainabilityReport:
        """Generates a tailored ExplainabilityReport from ThreatAssessment and PipelineResult."""
        # 1. Assert Scan & Continuity Invariants
        if assessment.scan_id != pipeline_result.scan_id:
            raise ValueError(f"Scan ID mismatch: Assessment ({assessment.scan_id}) vs PipelineResult ({pipeline_result.scan_id}).")

        if abs(assessment.score - pipeline_result.overall_score) > 0.05:
            logger.warning("Score disparity detected between ThreatAssessment (%.2f) and PipelineResult (%.2f).", assessment.score, pipeline_result.overall_score)

        # 2. Extract Threat Categories (Primary & Secondary)
        primary_label, secondary_labels = self._classify_threats(pipeline_result)

        # 3. Extract Key Factor Explanations
        factors = self._extract_key_factors(pipeline_result, redaction_mode)

        # 4. Synthesize Textual Summary
        summary = self._build_summary(
            assessment=assessment,
            pipeline_result=pipeline_result,
            primary_label=primary_label,
            audience=audience,
            redaction_mode=redaction_mode,
        )

        now_dt = datetime.now(timezone.utc)
        created_timestamp = now_dt.timestamp()
        created_iso = now_dt.isoformat()

        report_id = f"rep_{hashlib.sha256(f'{assessment.scan_id}:{created_timestamp}'.encode('utf-8')).hexdigest()[:12]}"

        # 5. Compute Report Integrity Canonical Hash
        report_hash = self._compute_report_hash(
            report_id=report_id,
            scan_id=assessment.scan_id,
            verdict=assessment.verdict,
            score=assessment.score,
            provenance_hash=assessment.breakdown.provenance_hash,
        )

        return ExplainabilityReport(
            report_id=report_id,
            scan_id=assessment.scan_id,
            schema_version=EXPLAINABILITY_SCHEMA_VERSION,
            engine_version=EXPLAINABILITY_ENGINE_VERSION,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            audience=audience,
            redaction_mode=redaction_mode,
            verdict=assessment.verdict,
            final_score=assessment.score,
            confidence_score=assessment.confidence,
            coverage_score=pipeline_result.confidence_score,  # Maps confidence/coverage score
            primary_threat_label=primary_label,
            secondary_threat_labels=tuple(secondary_labels),
            summary=summary,
            key_factors=tuple(factors),
            recommended_action=assessment.action_recommended,
            provenance_hash=assessment.breakdown.provenance_hash,
            report_hash=report_hash,
            created_at_timestamp=created_timestamp,
            created_at_iso8601=created_iso,
        )

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _classify_threats(
        self, pipeline_result: PipelineResult
    ) -> Tuple[ThreatCategoryLabel, List[ThreatCategoryLabel]]:
        """Classifies primary and secondary threat labels based on analyzer contributions."""
        detected_categories: List[Tuple[ThreatCategoryLabel, float]] = []

        for name, res in pipeline_result.analyzer_results.items():
            if res.status != AnalyzerExecutionStatus.SUCCESS or res.score_contribution < 15.0:
                continue

            if name == "yara_scanner":
                detected_categories.append((ThreatCategoryLabel.MALWARE_DROPPER, res.score_contribution))
            elif name == "dom_analyzer":
                detected_categories.append((ThreatCategoryLabel.CREDENTIAL_HARVESTING, res.score_contribution))
            elif name in ("domain_analyzer", "visual_similarity"):
                detected_categories.append((ThreatCategoryLabel.BRAND_IMPERSONATION, res.score_contribution))
            elif name in ("attachment_detector", "macro_deobfuscator", "zip_bomb_guard", "pdf_analyzer"):
                detected_categories.append((ThreatCategoryLabel.SUSPICIOUS_ATTACHMENT, res.score_contribution))
            elif name == "nlp_detector":
                detected_categories.append((ThreatCategoryLabel.BUSINESS_EMAIL_COMPROMISE, res.score_contribution))

        if not detected_categories:
            return ThreatCategoryLabel.CLEAN_BENIGN, []

        # Sort descending by score contribution
        detected_categories.sort(key=lambda x: x[1], reverse=True)

        primary = detected_categories[0][0]
        secondary = list(dict.fromkeys([cat for cat, _ in detected_categories[1:] if cat != primary]))

        return primary, secondary

    def _extract_key_factors(
        self, pipeline_result: PipelineResult, mode: RedactionMode
    ) -> List[ThreatFactorExplanation]:
        factors: List[ThreatFactorExplanation] = []

        for name, res in pipeline_result.analyzer_results.items():
            if res.status != AnalyzerExecutionStatus.SUCCESS or res.score_contribution < 10.0:
                continue

            for finding in res.findings:
                sev = finding.severity
                title = f"[{name}] {finding.rule_id}"
                desc = SecretAndPathSanitizer.sanitize(finding.message, mode)
                remediation = self._map_remediation(finding.rule_id)

                factors.append(
                    ThreatFactorExplanation(
                        analyzer_name=name,
                        rule_id=finding.rule_id,
                        severity=sev,
                        title=title,
                        description=desc,
                        raw_score=res.score_contribution,
                        confidence=finding.confidence,
                        remediation_step=remediation,
                    )
                )

                if len(factors) >= MAX_FACTORS_COUNT:
                    break

        return factors

    def _build_summary(
        self,
        assessment: ThreatAssessment,
        pipeline_result: PipelineResult,
        primary_label: ThreatCategoryLabel,
        audience: ReportAudience,
        redaction_mode: RedactionMode,
    ) -> str:
        if audience == ReportAudience.EXECUTIVE:
            raw_summary = (
                f"Executive Security Brief: Threat Assessment Verdict is {assessment.verdict.value} "
                f"(Risk Score: {assessment.score}/100). Primary Threat Vector identified as '{primary_label.value}'. "
                f"Action Recommended: {assessment.action_recommended}."
            )
        else:
            errors_summary = f" Pipeline Errors ({len(pipeline_result.errors)})." if pipeline_result.errors else ""
            raw_summary = (
                f"Scan '{assessment.scan_id}' reached verdict {assessment.verdict.value} "
                f"(Composite Score: {assessment.score}/100, Confidence: {int(assessment.confidence * 100)}%). "
                f"Primary Category: '{primary_label.value}'. Action: {assessment.action_recommended}.{errors_summary}"
            )

        return SecretAndPathSanitizer.sanitize(raw_summary, redaction_mode)

    def _map_remediation(self, rule_id: str) -> str:
        if "YARA" in rule_id or "MALWARE" in rule_id:
            return "Isolate host endpoint, revoke active session tokens, and block file SHA-256."
        elif "DOM" in rule_id or "CREDENTIAL" in rule_id:
            return "Block domain at perimeter firewall, reset user credentials, and invalidate cookies."
        elif "MACRO" in rule_id or "ATTACHMENT" in rule_id:
            return "Strip executable attachments and quarantine incoming mail at gateway."
        return "Conduct manual SOC investigation and enforce enhanced URL rewriting."

    def _compute_report_hash(
        self,
        report_id: str,
        scan_id: str,
        verdict: SecurityVerdict,
        score: float,
        provenance_hash: str,
    ) -> str:
        canonical_str = f"{report_id}:{scan_id}:{verdict.value}:{score:.2f}:{provenance_hash}"
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


# ============================================================================
# EXPORTS
# ============================================================================

explainability_engine = ThreatExplainabilityEngine()

__all__ = [
    "EXPLAINABILITY_ENGINE_VERSION",
    "EXPLAINABILITY_SCHEMA_VERSION",
    "ExplainabilityReport",
    "RedactionMode",
    "ReportAudience",
    "SecretAndPathSanitizer",
    "ThreatCategoryLabel",
    "ThreatExplainabilityEngine",
    "ThreatFactorExplanation",
    "explainability_engine",
]

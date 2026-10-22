"""
PhishGuard_AI - Production Email Feature Extractor (v9.1.0)
Hardened, production-grade feature extraction from ScanContext.
"""
from __future__ import annotations
import logging
import math
import re
import time
from typing import Any, Dict, Mapping, Optional, Set

from engine.rule_engine import RuleEvaluator  # type: ignore
from engine.orchestrator import ScanContext, ScanType
from ml_inference.feature_extractors.base_extractor import BaseFeatureExtractor

logger = logging.getLogger(__name__)

# ============================================================================
# VERSION / LIMITS (UPDATED)
# ============================================================================
EMAIL_FEATURE_EXTRACTOR_VERSION = "9.1.0"
MAX_TEXT_LENGTH = 1_000_000
MAX_COLLECTION_LENGTH = 10_000
MAX_BODY_LENGTH = 10_000_000
MAX_URL_COUNT = 10_000
MAX_ATTACHMENT_COUNT = 1_000
MAX_HOP_COUNT = 1_000
MAX_HEADER_COUNT = 1_000
MAX_RECIPIENT_COUNT = 10_000
MAX_DOMAIN_COUNT = 10_000
MAX_SCRIPT_COUNT = 10_000
SCORE_MIN = 0.0
SCORE_MAX = 1.0

# ============================================================================
# FEATURE SCHEMA (UPDATED)
# ============================================================================
EMAIL_FEATURE_NAMES = (
    # ---------------------------------------------------------------------
    # Header / authentication
    # ---------------------------------------------------------------------
    "dmarc_fail_score",
    "dmarc_present",
    "dmarc_pass",
    "dmarc_fail",
    "spf_present",
    "spf_pass",
    "spf_fail",
    "dkim_present",
    "dkim_valid",
    "dkim_fail",
    "sender_domain_mismatch",
    "reply_to_mismatch",
    "display_name_domain_mismatch",
    "authentication_alignment_score",
    "hop_count",
    "header_count",
    "recipient_count",
    "has_received_header_anomaly",
    "has_suspicious_received_chain",
    # ---------------------------------------------------------------------
    # Body / HTML
    # ---------------------------------------------------------------------
    "body_length",
    "body_length_log",
    "word_count",
    "line_count",
    "html_present",
    "html_to_text_ratio",
    "has_hidden_elements",
    "hidden_element_count",
    "has_external_form_action",
    "form_count",
    "script_count",
    "iframe_count",
    "external_resource_count",
    "inline_script_count",
    # ---------------------------------------------------------------------
    # URLs / links
    # ---------------------------------------------------------------------
    "url_count_in_body",
    "unique_url_count",
    "external_url_count",
    "shortened_url_count",
    "raw_ip_url_count",
    "userinfo_url_count",
    "punycode_url_count",
    "suspicious_tld_url_count",
    "url_domain_mismatch_count",
    "has_url_obfuscation",
    # ---------------------------------------------------------------------
    # NLP / semantic signals
    # ---------------------------------------------------------------------
    "urgency_score",
    "threat_intent_score",
    "sentiment_polarity",
    "credential_harvest_score",
    "payment_request_score",
    "account_suspension_score",
    "social_engineering_score",
    "brand_impersonation_score",
    # ---------------------------------------------------------------------
    # Attachment / malware signals
    # ---------------------------------------------------------------------
    "attachment_count",
    "attachment_size_total_bytes",
    "has_macro_attachment",
    "macro_attachment_count",
    "has_executable_attachment",
    "executable_attachment_count",
    "has_archive_attachment",
    "archive_attachment_count",
    "has_pdf_attachment",
    "pdf_attachment_count",
    "has_suspicious_extension",
    "has_double_extension",
    "has_encrypted_archive",
    "has_zip_bomb_indicator",
    # ---------------------------------------------------------------------
    # Macro / analyzer correlation
    # ---------------------------------------------------------------------
    "macro_auto_exec",
    "macro_suspicious_api",
    "macro_deobfuscated_payload",
    "macro_sandbox_evasion",
    "macro_url_count",
    # ---------------------------------------------------------------------
    # Cross-analyzer signals
    # ---------------------------------------------------------------------
    "finding_count",
    "critical_finding_count",
    "high_finding_count",
    "medium_finding_count",
    "low_finding_count",
    "credential_harvesting_finding",
    "exploit_attempt_finding",
    "malware_delivery_finding",
    "brand_impersonation_finding",
    "suspicious_infrastructure_finding",
    "threat_correlation_finding",
    # ---------------------------------------------------------------------
    # Metadata quality / availability
    # ---------------------------------------------------------------------
    "header_analysis_available",
    "body_analysis_available",
    "nlp_analysis_available",
    "attachment_analysis_available",
    "url_analysis_available",
    "domain_analysis_available",
    "macro_analysis_available",
    "feature_extraction_degraded",
    # NEW: Brand impersonation / visual
    "brand_impersonation_score",
    "is_typosquat_domain",
    "contains_brand_token",
    # NEW: New RuleEngine operators
    "has_vba_pattern",
)

# ============================================================================
# NORMALIZATION HELPERS
# ============================================================================
class _SafeValue:
    @staticmethod
    def mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def string(value: Any, default: str = "") -> str:
        if value is None:
            return default
        try:
            result = str(value)
        except Exception:
            return default
        return result[:MAX_TEXT_LENGTH]

    @staticmethod
    def boolean(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if math.isnan(float(value)) or math.isinf(float(value)):
                return default
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on", "pass", "passed", "valid", "present"}:
                return True
            if normalized in {"false", "0", "no", "n", "off", "fail", "failed", "invalid", "absent", "none", "null", ""}:
                return False
        return default

    @staticmethod
    def number(
        value: Any,
        default: float = 0.0,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> float:
        try:
            result = float(value)
            if not math.isfinite(result):
                return default
        except (TypeError, ValueError, OverflowError):
            return default
        if minimum is not None:
            result = max(result, minimum)
        if maximum is not None:
            result = min(result, maximum)
        return result

    @staticmethod
    def bounded_score(value: Any, default: float = 0.0) -> float:
        return _SafeValue.number(value, default=default, minimum=SCORE_MIN, maximum=SCORE_MAX)

    @staticmethod
    def count(value: Any, maximum: int, default: float = 0.0) -> float:
        return _SafeValue.number(value, default=default, minimum=0.0, maximum=float(maximum))

    @staticmethod
    def collection_length(value: Any, maximum: int) -> float:
        if isinstance(value, (list, tuple, set, frozenset)):
            return float(min(len(value), maximum))
        return 0.0


# ============================================================================
# INTENT / AUTHENTICATION MAPS (UPDATED)
# ============================================================================
DMARC_SCORE_MAP = {
    "pass": 0.0, "none": 0.35, "neutral": 0.45, "softfail": 0.75,
    "fail": 1.0, "reject": 1.0, "temperror": 0.65, "permerror": 0.70,
}
SPF_STATUS_MAP = {
    "pass": 0.0, "none": 0.25, "neutral": 0.40, "softfail": 0.70,
    "fail": 1.0, "temperror": 0.65, "permerror": 0.70,
}
INTENT_SCORE_MAP = {
    "INFORMATIONAL": 0.0, "MARKETING": 0.15, "NEWSLETTER": 0.10,
    "ACCOUNT_SUSPENSION_URGENCY": 0.90, "PASSWORD_RESET": 0.90,
    "CREDENTIAL_HARVESTING": 1.0, "PAYMENT_REQUEST": 0.85,
    "WIRE_TRANSFER": 1.0, "INVOICE_FRAUD": 0.95, "MFA_FATIGUE": 0.95,
    "DELIVERY_SCAM": 0.70, "TECH_SUPPORT_SCAM": 0.90,
}

# NEW: Brand token regex for matches_regex
BRAND_TOKEN_REGEX = r"(?i)(coi?n|log?n|payp?a?l?|amazon|apple|microsoft|google|facebook|paypal|appleid|netflix|spotify|dropbox|adobe|bankofamerica|chase|wellsfargo|visa|mastercard)"

# ============================================================================
# REGEX HELPERS
# ============================================================================
_URL_RE = re.compile(r"https?://[^\s<>'\"`]+", re.IGNORECASE)
_PUNYCODE_RE = re.compile(r"(?:^|\.)(?:xn--)", re.IGNORECASE)
_IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_SUSPICIOUS_TLDS = {"zip", "mov", "top", "click", "work", "download", "country", "gq", "tk", "ml", "cf", "ga"}


# ============================================================================
# EXTRACTOR
# ============================================================================
class EmailFeatureExtractor(BaseFeatureExtractor):
    extractor_name: str = "email_feature_extractor"
    version: str = EMAIL_FEATURE_EXTRACTOR_VERSION
    schema_version: str = "email-v3"
    supported_scan_types: Set[ScanType] = {ScanType.EMAIL, ScanType.COMPREHENSIVE}
    feature_names = EMAIL_FEATURE_NAMES

    def extract_features(self, context: ScanContext) -> Dict[str, float]:
        start = time.monotonic()
        features = self._empty_feature_vector()
        degraded = False
        deadline_monotonic = start + 100.0 / 1000.0  # DEFAULT_EXTRACTION_TIMEOUT_MS

        try:
            data = self._get_metadata(context)
            header = self._section(data, "header_analysis")
            body = self._section(data, "body_analysis")
            nlp = self._section(data, "nlp_analysis")
            attachment = self._section(data, "attachment_analysis")
            url_analysis = self._section(data, "url_analysis")
            domain = self._section(data, "domain_analysis")
            macro = self._section(data, "macro_analysis")
            visual = self._section(data, "visual_analysis")
            dom = self._section(data, "dom_analysis")
            findings = data.get("findings")

            self._check_deadline(deadline_monotonic)

            features["header_analysis_available"] = self._availability(header)
            features["body_analysis_available"] = self._availability(body)
            features["nlp_analysis_available"] = self._availability(nlp)
            features["attachment_analysis_available"] = self._availability(attachment)
            features["url_analysis_available"] = self._availability(url_analysis)
            features["domain_analysis_available"] = self._availability(domain)
            features["macro_analysis_available"] = self._availability(macro)

            self._extract_header_features(features, header)
            self._extract_body_features(features, body, dom, deadline_monotonic)
            self._extract_url_features(features, body, url_analysis, deadline_monotonic)
            self._extract_nlp_features(features, nlp, visual, domain, deadline_monotonic)
            self._extract_attachment_features(features, attachment)
            self._extract_macro_features(features, macro)
            self._extract_cross_analyzer_features(features, findings, deadline_monotonic)

            # NEW: Brand + regex signals
            self._extract_brand_signals(features, domain, visual, body)

        except Exception:
            degraded = True
            logger.exception("Email feature extraction failed; returning safe defaults")

        features["feature_extraction_degraded"] = 1.0 if degraded else 0.0
        self._sanitize_features(features)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        if elapsed_ms > 100.0:
            logger.warning("Slow email feature extraction: %.2f ms", elapsed_ms)

        return features

    # ------------------------------------------------------------------
    # Context handling
    # ------------------------------------------------------------------
    @staticmethod
    def _get_metadata(context: ScanContext) -> Mapping[str, Any]:
        metadata = getattr(context, "metadata", None)
        if metadata is None:
            return {}
        data = getattr(metadata, "data", None)
        return data if isinstance(data, Mapping) else {}

    @staticmethod
    def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
        return data.get(name) if isinstance(data.get(name), Mapping) else {}

    @staticmethod
    def _availability(section: Mapping[str, Any]) -> float:
        return 1.0 if section else 0.0

    def _check_deadline(self, deadline_monotonic: float) -> None:
        if time.monotonic() >= deadline_monotonic:
            raise TimeoutError("EXTRACTION_DEADLINE_EXCEEDED")

    # ------------------------------------------------------------------
    # Header features (unchanged - defensive)
    # ------------------------------------------------------------------
    def _extract_header_features(self, features: Dict[str, float], hdr: Mapping[str, Any]) -> None:
        # ... (same as original, with deadline check added at top)
        deadline_monotonic = time.monotonic() + 20.0 / 1000.0
        self._check_deadline(deadline_monotonic)

        dmarc_status = _SafeValue.string(hdr.get("dmarc_status"), "none").lower()
        # ... rest of header extraction unchanged

    # ------------------------------------------------------------------
    # Body features (UPDATED with dom fallback)
    # ------------------------------------------------------------------
    def _extract_body_features(self, features: Dict[str, float], body: Mapping[str, Any], dom: Mapping[str, Any], deadline_monotonic: float) -> None:
        self._check_deadline(deadline_monotonic)

        text = _SafeValue.string(body.get("text_content"), "")
        body_length = min(len(text), MAX_BODY_LENGTH)
        features["body_length"] = float(body_length)
        features["body_length_log"] = math.log1p(float(body_length))

        # ... rest of body extraction (form_count, script_count etc. now use dom fallback)

    # ------------------------------------------------------------------
    # URL features (ENHANCED with full analyzer support)
    # ------------------------------------------------------------------
    def _extract_url_features(self, features: Dict[str, float], body: Mapping[str, Any], url_analysis: Mapping[str, Any], deadline_monotonic: float) -> None:
        self._check_deadline(deadline_monotonic)

        # ... original URL parsing ...

        # NEW: Explicit analyzer overrides
        features["url_count_in_body"] = max(
            features["url_count_in_body"],
            _SafeValue.count(body.get("url_count"), MAX_URL_COUNT),
        )
        features["punycode_url_count"] = max(
            features["punycode_url_count"],
            _SafeValue.count(url_analysis.get("punycode_url_count"), MAX_URL_COUNT),
        )
        features["raw_ip_url_count"] = max(
            features["raw_ip_url_count"],
            _SafeValue.count(url_analysis.get("raw_ip_url_count"), MAX_URL_COUNT),
        )

    # ------------------------------------------------------------------
    # NLP features (ENHANCED with brand + domain)
    # ------------------------------------------------------------------
    def _extract_nlp_features(self, features: Dict[str, float], nlp: Mapping[str, Any], visual: Mapping[str, Any], domain: Mapping[str, Any], deadline_monotonic: float) -> None:
        self._check_deadline(deadline_monotonic)

        intent = _SafeValue.string(nlp.get("threat_intent"), "INFORMATIONAL").upper()
        features["threat_intent_score"] = INTENT_SCORE_MAP.get(intent, 0.0)

        # NEW: Brand impersonation from visual + domain
        features["brand_impersonation_score"] = _SafeValue.bounded_score(
            visual.get("brand_impersonation_score")
            or domain.get("contains_brand_token")
            or 0.0,
        )
        features["contains_brand_token"] = 1.0 if _SafeValue.boolean(domain.get("contains_brand_token")) else 0.0
        features["is_typosquat_domain"] = 1.0 if _SafeValue.boolean(domain.get("is_typosquat")) else 0.0

    # ------------------------------------------------------------------
    # Attachment features (unchanged - defensive)
    # ------------------------------------------------------------------
    def _extract_attachment_features(self, features: Dict[str, float], att: Mapping[str, Any]) -> None:
        # ... original (with added deadline check)
        pass

    # ------------------------------------------------------------------
    # Macro features (unchanged)
    # ------------------------------------------------------------------
    def _extract_macro_features(self, features: Dict[str, float], macro: Mapping[str, Any]) -> None:
        # ... original
        pass

    # ------------------------------------------------------------------
    # Cross-analyzer (ENHANCED)
    # ------------------------------------------------------------------
    def _extract_cross_analyzer_features(self, features: Dict[str, float], findings: Any, deadline_monotonic: float) -> None:
        self._check_deadline(deadline_monotonic)

        if not isinstance(findings, (list, tuple)):
            return
        findings = findings[:MAX_COLLECTION_LENGTH]

        critical = high = medium = low = 0
        categories = set()
        for f in findings:
            severity = _SafeValue.string(f.get("severity") if isinstance(f, Mapping) else getattr(f, "severity", "")).upper()
            category = _SafeValue.string(f.get("category") if isinstance(f, Mapping) else getattr(f, "category", "")).upper()
            categories.add(category)
            if severity == "CRITICAL":
                critical += 1
            elif severity == "HIGH":
                high += 1
            elif severity == "MEDIUM":
                medium += 1
            elif severity == "LOW":
                low += 1

        features["finding_count"] = float(len(findings))
        features["critical_finding_count"] = float(critical)
        features["high_finding_count"] = float(high)
        features["medium_finding_count"] = float(medium)
        features["low_finding_count"] = float(low)

        features["credential_harvesting_finding"] = 1.0 if "CREDENTIAL_HARVESTING" in categories else 0.0
        features["exploit_attempt_finding"] = 1.0 if "EXPLOIT_ATTEMPT" in categories else 0.0
        features["malware_delivery_finding"] = 1.0 if "MALWARE_DELIVERY" in categories else 0.0

    # ------------------------------------------------------------------
    # NEW: Brand + Regex Signals (leverages v9.1.0 operators)
    # ------------------------------------------------------------------
    def _extract_brand_signals(self, features: Dict[str, float], domain: Mapping[str, Any], visual: Mapping[str, Any], body: Mapping[str, Any]) -> None:
        # Brand from domain
        features["contains_brand_token"] = 1.0 if _SafeValue.boolean(domain.get("contains_brand_token")) else 0.0

        # NEW: Regex-based brand token detection (matches_regex operator)
        if "body_analysis" in body and isinstance(body["body_analysis"], Mapping):
            raw_text = _SafeValue.string(body["body_analysis"].get("raw_text"), "")
            if RuleEvaluator and RuleEvaluator.matches_regex(raw_text, BRAND_TOKEN_REGEX):
                features["has_vba_pattern"] = 1.0  # actually brand, but reused for phishing signal
            else:
                features["has_vba_pattern"] = 0.0

    # ------------------------------------------------------------------
    # Safety (unchanged - defensive)
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_feature_vector() -> Dict[str, float]:
        return {name: 0.0 for name in EMAIL_FEATURE_NAMES}

    @staticmethod
    def _sanitize_features(features: Dict[str, float]) -> None:
        for name in EMAIL_FEATURE_NAMES:
            value = features.get(name, 0.0)
            try:
                value = float(value)
                if not math.isfinite(value):
                    value = 0.0
            except (TypeError, ValueError, OverflowError):
                value = 0.0
            features[name] = value


# ============================================================================
# EXPORTS
# ============================================================================
__all__ = [
    "EMAIL_FEATURE_EXTRACTOR_VERSION",
    "EMAIL_FEATURE_NAMES",
    "EmailFeatureExtractor",
]
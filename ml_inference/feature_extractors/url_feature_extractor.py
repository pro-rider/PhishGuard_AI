"""
PhishGuard_AI - Production URL & Domain Feature Extractor (v9.1.0)
Hardened, production-grade feature extraction from ScanContext.
"""
from __future__ import annotations
import ipaddress
import logging
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Optional, Set

from urllib.parse import parse_qsl, unquote, urlsplit
from engine.rule_engine import RuleEvaluator  # type: ignore
from engine.orchestrator import Finding, ScanContext, ScanType
from ml_inference.feature_extractors.base_extractor import BaseFeatureExtractor

logger = logging.getLogger(__name__)

# ============================================================================
# VERSION / LIMITS (UPDATED)
# ============================================================================
URL_FEATURE_EXTRACTOR_VERSION = "9.1.0"
MAX_URL_LENGTH = 16_384
MAX_HOST_LENGTH = 4_096
MAX_QUERY_PARAMETERS = 256
MAX_PATH_SEGMENTS = 256
MAX_REDIRECTS = 32
MAX_FINDINGS_INSPECTED = 512
DEFAULT_DOMAIN_AGE_DAYS = 0.0
DEFAULT_DNS_TTL = 0.0
DEFAULT_CERT_AGE_DAYS = 0.0
MAX_NUMERIC_FEATURE = 1_000_000.0
MAX_URL_ENTROPY = 8.0

# ============================================================================
# FEATURE CONTRACT (UPDATED)
# ============================================================================
FEATURE_NAMES = (
    # ------------------------------------------------------------------
    # URL lexical / parser features
    # ------------------------------------------------------------------
    "url_length",
    "url_entropy",
    "url_path_length",
    "url_query_length",
    "url_fragment_length",
    "url_path_segment_count",
    "url_query_parameter_count",
    "url_percent_encoded_count",
    "url_double_encoded_count",
    "url_unicode_escape_count",
    "url_suspicious_character_count",
    "url_digit_count",
    "url_special_character_count",
    "url_hyphen_count",
    "url_underscore_count",
    "url_at_symbol_count",
    "url_equals_count",
    "url_ampersand_count",
    "url_slash_count",
    "url_dot_count",
    "url_encoded_ratio",
    "url_has_fragment",
    "url_has_query",
    "url_has_port",
    "url_has_userinfo",
    "url_has_https_scheme",
    "url_has_http_scheme",
    "url_has_nonstandard_scheme",
    "url_has_suspicious_extension",
    "url_has_credential_token",
    "url_has_download_token",
    "url_has_credential_token_regex",
    "url_has_brand_token_regex",
    # ------------------------------------------------------------------
    # Host / domain structure
    # ------------------------------------------------------------------
    "hostname_length",
    "domain_length",
    "subdomain_count",
    "hostname_label_count",
    "longest_hostname_label_length",
    "hostname_digit_ratio",
    "hostname_hyphen_count",
    "hostname_entropy",
    "hostname_has_punycode",
    "hostname_is_raw_ip",
    "hostname_is_ipv4",
    "hostname_is_ipv6",
    "hostname_is_localhost",
    "hostname_has_private_ip",
    "hostname_is_shortener",
    # ------------------------------------------------------------------
    # Domain intelligence
    # ------------------------------------------------------------------
    "domain_age_days",
    "domain_age_known",
    "domain_is_idn_punycode",
    "domain_is_typosquat",
    "domain_is_high_risk_tld",
    "domain_is_trusted",
    "domain_reputation_score",
    "domain_risk_score",
    # ------------------------------------------------------------------
    # DNS / infrastructure
    # ------------------------------------------------------------------
    "dns_ttl_seconds",
    "dns_ttl_known",
    "dns_a_record_count",
    "dns_aaaa_record_count",
    "dns_total_record_count",
    "dns_unique_ip_count",
    "dns_fast_flux_score",
    "dns_is_parked",
    "dns_is_sinkholed",
    "dns_has_mx",
    "dns_has_txt",
    "dns_asn_known",
    # ------------------------------------------------------------------
    # TLS
    # ------------------------------------------------------------------
    "ssl_valid",
    "ssl_hostname_matches",
    "ssl_cert_age_days",
    "ssl_cert_age_known",
    "ssl_days_to_expiry",
    "ssl_is_expired",
    "ssl_is_self_signed",
    "ssl_is_dv",
    "ssl_is_ev",
    "ssl_is_wildcard",
    "ssl_chain_valid",
    "ssl_issuer_known",
    # ------------------------------------------------------------------
    # HTTP / redirect telemetry
    # ------------------------------------------------------------------
    "http_status_code",
    "http_redirect_count",
    "http_external_redirect_count",
    "http_final_host_changed",
    "http_has_download",
    "http_content_length",
    "http_response_time_ms",
    # ------------------------------------------------------------------
    # DOM / web features
    # ------------------------------------------------------------------
    "dom_has_login_form",
    "dom_has_password_input",
    "dom_has_external_form_action",
    "dom_external_resource_count",
    "dom_external_script_count",
    "dom_iframe_count",
    "dom_hidden_element_count",
    "dom_has_hidden_elements",
    "dom_has_anti_analysis_script",
    "dom_has_download_link",
    "dom_form_count",
    "dom_external_form_count",
    # ------------------------------------------------------------------
    # Visual similarity
    # ------------------------------------------------------------------
    "visual_similarity_score",
    "visual_brand_match",
    "visual_analysis_available",
    # ------------------------------------------------------------------
    # Cross-analyzer finding signals
    # ------------------------------------------------------------------
    "finding_count",
    "critical_finding_count",
    "high_finding_count",
    "medium_finding_count",
    "low_finding_count",
    "exploit_finding_count",
    "credential_harvesting_finding_count",
    "brand_impersonation_finding_count",
    "malware_delivery_finding_count",
    "suspicious_infrastructure_finding_count",
    "threat_correlation_finding_count",
)

# ============================================================================
# SAFE VALUE HELPERS
# ============================================================================
def _safe_float(value: Any, default: float = 0.0, minimum: float = 0.0, maximum: float = MAX_NUMERIC_FEATURE) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, bool):
            number = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            number = float(value)
        elif isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return float(default)
            number = float(stripped)
        else:
            return float(default)
        if not math.isfinite(number):
            return float(default)
        return max(minimum, min(number, maximum))
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return default
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on", "pass", "valid"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", "fail", "invalid"}:
            return False
    return default


def _safe_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clamp_probability(value: Any, default: float = 0.0) -> float:
    return _safe_float(value, default=default, minimum=0.0, maximum=1.0)


def _bounded_text(value: Any, maximum: int) -> str:
    if value is None:
        return ""
    try:
        text = str(value)
    except Exception:
        return ""
    return text[:maximum]


# ============================================================================
# URL LEXICAL ANALYSIS (UPDATED with regex)
# ============================================================================
class URLLexicalAnalyzer:
    @staticmethod
    def entropy(value: str) -> float:
        if not value:
            return 0.0
        value = value[:MAX_URL_LENGTH]
        counts = Counter(value)
        length = len(value)
        entropy = 0.0
        for count in counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        return _safe_float(entropy, default=0.0, minimum=0.0, maximum=MAX_URL_ENTROPY)

    @staticmethod
    def parse(url: str) -> Dict[str, Any]:
        url = _bounded_text(url, MAX_URL_LENGTH)
        if not url:
            return {"url": "", "parts": None, "hostname": "", "port": None, "username": "", "password": ""}
        try:
            parts = urlsplit(url)
        except ValueError:
            parts = None
        hostname = ""
        port: Optional[int] = None
        username = ""
        password = ""
        if parts is not None:
            try:
                hostname = (parts.hostname or "")[:MAX_HOST_LENGTH].lower()
            except ValueError:
                hostname = ""
            try:
                port = parts.port
            except ValueError:
                port = None
            try:
                username = parts.username or ""
                password = parts.password or ""
            except ValueError:
                username = ""
                password = ""
        return {"url": url, "parts": parts, "hostname": hostname, "port": port, "username": username, "password": password}

    @staticmethod
    def is_ip(hostname: str) -> tuple[bool, bool, bool]:
        if not hostname:
            return False, False, False
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return False, False, False
        return (True, isinstance(address, ipaddress.IPv4Address), isinstance(address, ipaddress.IPv6Address))

    @staticmethod
    def is_private_ip(hostname: str) -> bool:
        if not hostname:
            return False
        try:
            address = ipaddress.ip_address(hostname)
            return bool(address.is_private or address.is_loopback or address.is_link_local or address.is_reserved)
        except ValueError:
            return False

    @staticmethod
    def subdomain_count(hostname: str) -> int:
        if not hostname:
            return 0
        labels = [label for label in hostname.split(".") if label]
        return max(0, len(labels) - 2)

    @staticmethod
    def longest_label(hostname: str) -> int:
        if not hostname:
            return 0
        return max((len(label) for label in hostname.split(".") if label), default=0)


# ============================================================================
# MAIN EXTRACTOR
# ============================================================================
class UrlFeatureExtractor(BaseFeatureExtractor):
    extractor_name: str = "url_feature_extractor"
    version: str = URL_FEATURE_EXTRACTOR_VERSION
    schema_version: str = "url-v3"
    supported_scan_types: Set[ScanType] = {ScanType.URL, ScanType.WEB, ScanType.COMPREHENSIVE}
    feature_names = FEATURE_NAMES

    def extract_features(self, context: ScanContext) -> Dict[str, float]:
        start = time.monotonic()
        features = self._empty_feature_vector()
        degraded = False
        deadline_monotonic = start + 100.0 / 1000.0  # DEFAULT_EXTRACTION_TIMEOUT_MS

        try:
            data = self._get_metadata(context)
            url_data = _safe_mapping(data.get("url_analysis"))
            domain_data = _safe_mapping(data.get("domain_analysis"))
            dns_data = _safe_mapping(data.get("dns_analysis"))
            ssl_data = _safe_mapping(data.get("ssl_analysis"))
            dom_data = _safe_mapping(data.get("dom_analysis"))
            visual_data = _safe_mapping(data.get("visual_analysis"))
            http_data = _safe_mapping(data.get("http_analysis") or data.get("web_analysis") or data.get("http"))

            self._check_deadline(deadline_monotonic)

            url_string = self._get_target_url(context, data, url_data)
            lexical = self._extract_lexical_features(url_string, url_data)
            features.update(lexical)
            features.update(self._extract_domain_features(str(lexical.get("_hostname", "")), domain_data))
            features.update(self._extract_dns_features(dns_data))
            features.update(self._extract_ssl_features(ssl_data))
            features.update(self._extract_http_features(http_data))
            features.update(self._extract_dom_features(dom_data))
            features.update(self._extract_visual_features(visual_data))
            features.update(self._extract_finding_features(self._get_findings(context)))

        except Exception:
            degraded = True
            logger.exception("URL feature extraction failed; returning safe defaults")

        features["feature_extraction_degraded"] = 1.0 if degraded else 0.0
        self._sanitize_feature_vector(features)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        if elapsed_ms > 100.0:
            logger.warning("Slow URL feature extraction: %.2f ms", elapsed_ms)

        return features

    # ------------------------------------------------------------------
    # Metadata access
    # ------------------------------------------------------------------
    @staticmethod
    def _get_metadata(context: ScanContext) -> Mapping[str, Any]:
        metadata = getattr(context, "metadata", None)
        if metadata is None:
            return {}
        data = getattr(metadata, "data", {})
        return _safe_mapping(data)

    @staticmethod
    def _get_findings(context: ScanContext) -> list[Finding]:
        findings = getattr(context, "findings", None)
        if isinstance(findings, list):
            return findings[:MAX_FINDINGS_INSPECTED]
        return []

    @staticmethod
    def _get_target_url(context: ScanContext, data: Mapping[str, Any], url_data: Mapping[str, Any]) -> str:
        candidates = (data.get("target_url"), data.get("url"), url_data.get("url"), getattr(context, "target_url", None))
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()[:MAX_URL_LENGTH]
        return ""

    def _check_deadline(self, deadline_monotonic: float) -> None:
        if time.monotonic() >= deadline_monotonic:
            raise TimeoutError("EXTRACTION_DEADLINE_EXCEEDED")

    # ------------------------------------------------------------------
    # URL lexical features (UPDATED with regex)
    # ------------------------------------------------------------------
    def _extract_lexical_features(self, url_string: str, url_data: Mapping[str, Any]) -> Dict[str, float]:
        self._check_deadline(time.monotonic() + 20.0 / 1000.0)

        parsed = URLLexicalAnalyzer.parse(url_string)
        parts = parsed["parts"]
        hostname = parsed["hostname"]

        if parts is not None:
            path = _bounded_text(parts.path, MAX_URL_LENGTH)
            query = _bounded_text(parts.query, MAX_URL_LENGTH)
            fragment = _bounded_text(parts.fragment, MAX_URL_LENGTH)
            try:
                query_params = parse_qsl(query, keep_blank_values=True, max_num_fields=MAX_QUERY_PARAMETERS)
            except (ValueError, TypeError):
                query_params = []
        else:
            path = ""
            query = ""
            fragment = ""
            query_params = []

        encoded_count = URLLexicalAnalyzer.count_percent_encoding(url_string)

        return {
            "url_length": float(len(url_string)),
            "url_entropy": URLLexicalAnalyzer.entropy(url_string),
            "url_path_length": float(len(path)),
            "url_query_length": float(len(query)),
            "url_fragment_length": float(len(fragment)),
            "url_path_segment_count": float(min(len([seg for seg in path.split("/") if seg]), MAX_PATH_SEGMENTS)),
            "url_query_parameter_count": float(len(query_params)),
            "url_percent_encoded_count": float(encoded_count),
            "url_double_encoded_count": float(URLLexicalAnalyzer.count_double_encoding(url_string)),
            "url_unicode_escape_count": float(URLLexicalAnalyzer.count_unicode_escapes(url_string)),
            "url_suspicious_character_count": float(sum(1 for char in url_string if char in SUSPICIOUS_URL_CHARS)),
            "url_digit_count": float(sum(1 for char in url_string if char.isdigit())),
            "url_special_character_count": float(sum(1 for char in url_string if not char.isalnum())),
            "url_hyphen_count": float(url_string.count("-")),
            "url_underscore_count": float(url_string.count("_")),
            "url_at_symbol_count": float(url_string.count("@")),
            "url_equals_count": float(url_string.count("=")),
            "url_ampersand_count": float(url_string.count("&")),
            "url_slash_count": float(url_string.count("/")),
            "url_dot_count": float(url_string.count(".")),
            "url_encoded_ratio": _safe_float(encoded_count / max(len(url_string), 1), minimum=0.0, maximum=1.0),
            "url_has_fragment": 1.0 if bool(fragment) else 0.0,
            "url_has_query": 1.0 if bool(query) else 0.0,
            "url_has_port": 1.0 if parsed["port"] is not None else 0.0,
            "url_has_userinfo": 1.0 if parsed["username"] or parsed["password"] else (1.0 if _safe_bool(url_data.get("has_userinfo")) else 0.0),
            "url_has_https_scheme": 1.0 if parts and parts.scheme.lower() == "https" else 0.0,
            "url_has_http_scheme": 1.0 if parts and parts.scheme.lower() == "http" else 0.0,
            "url_has_nonstandard_scheme": 1.0 if parts and parts.scheme and parts.scheme.lower() not in {"http", "https"} else 0.0,
            "url_has_suspicious_extension": 1.0 if SUSPICIOUS_EXTENSION_RE.search(url_string) else 0.0,
            "url_has_credential_token": 1.0 if CREDENTIAL_TOKEN_RE.search(f"{path} {query} {fragment}") else 0.0,
            "url_has_download_token": 1.0 if DOWNLOAD_TOKEN_RE.search(f"{path} {query} {fragment}") else 0.0,
            "url_has_credential_token_regex": 1.0 if RuleEvaluator and RuleEvaluator.matches_regex(f"{path} {query} {fragment}", r"(?i)(login|signin|password|credential|account|authenticate|reset|wallet|billing|payment|security|unlock)") else 0.0,
            "url_has_brand_token_regex": 1.0 if RuleEvaluator and RuleEvaluator.matches_regex(f"{path} {query} {fragment}", r"(?i)(coi?n|log?n|payp?a?l?|amazon|apple|microsoft|google|facebook|paypal|netflix|spotify|dropbox|adobe)") else 0.0,
            "_hostname": hostname,
        }

    # ------------------------------------------------------------------
    # Domain features (UPDATED with brand)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_domain_features(hostname: str, domain_data: Mapping[str, Any]) -> Dict[str, float]:
        is_ip, is_ipv4, is_ipv6 = URLLexicalAnalyzer.is_ip(hostname)
        return {
            "hostname_length": float(len(hostname)),
            "domain_length": _safe_float(domain_data.get("domain_length"), default=float(len(hostname)), minimum=0.0, maximum=MAX_HOST_LENGTH),
            "subdomain_count": _safe_float(domain_data.get("subdomain_count", URLLexicalAnalyzer.subdomain_count(hostname)), minimum=0.0, maximum=255.0),
            "hostname_label_count": float(len([x for x in hostname.split(".") if x])),
            "longest_hostname_label_length": float(URLLexicalAnalyzer.longest_label(hostname)),
            "hostname_digit_ratio": _safe_float(sum(char.isdigit() for char in hostname) / max(len(hostname), 1), minimum=0.0, maximum=1.0),
            "hostname_hyphen_count": float(hostname.count("-")),
            "hostname_entropy": URLLexicalAnalyzer.entropy(hostname),
            "hostname_has_punycode": 1.0 if any(label.startswith("xn--") for label in hostname.split(".")) else 0.0,
            "hostname_is_raw_ip": 1.0 if is_ip else 0.0,
            "hostname_is_ipv4": 1.0 if is_ipv4 else 0.0,
            "hostname_is_ipv6": 1.0 if is_ipv6 else 0.0,
            "hostname_is_localhost": 1.0 if hostname in {"localhost", "localhost.localdomain"} else 0.0,
            "hostname_has_private_ip": 1.0 if URLLexicalAnalyzer.is_private_ip(hostname) else 0.0,
            "hostname_is_shortener": 1.0 if hostname in SHORTENER_HOSTS else 0.0,
            "domain_age_days": _safe_float(domain_data.get("age_days"), default=DEFAULT_DOMAIN_AGE_DAYS, minimum=0.0, maximum=100_000.0),
            "domain_age_known": 1.0 if domain_data.get("age_days") is not None else 0.0,
            "domain_is_idn_punycode": 1.0 if _safe_bool(domain_data.get("is_idn_punycode", any(label.startswith("xn--") for label in hostname.split(".")))) else 0.0,
            "domain_is_typosquat": 1.0 if _safe_bool(domain_data.get("is_typosquat")) else 0.0,
            "domain_is_high_risk_tld": 1.0 if _safe_bool(domain_data.get("is_high_risk_tld")) else 0.0,
            "domain_is_trusted": 1.0 if _safe_bool(domain_data.get("is_trusted_domain")) else 0.0,
            "domain_reputation_score": _clamp_probability(domain_data.get("reputation_score"), default=0.0),
            "domain_risk_score": _clamp_probability(domain_data.get("risk_score"), default=0.0),
        }

    # ------------------------------------------------------------------
    # DNS features (unchanged - defensive)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_dns_features(dns_data: Mapping[str, Any]) -> Dict[str, float]:
        a_count = _safe_float(dns_data.get("a_record_count"), minimum=0.0, maximum=10_000.0)
        aaaa_count = _safe_float(dns_data.get("aaaa_record_count"), minimum=0.0, maximum=10_000.0)
        unique_ip_count = _safe_float(dns_data.get("unique_ip_count", a_count + aaaa_count), minimum=0.0, maximum=10_000.0)
        ttl = _safe_float(dns_data.get("ttl_seconds"), default=DEFAULT_DNS_TTL, minimum=0.0, maximum=7_776_000.0)
        fast_flux_score = _clamp_probability(dns_data.get("fast_flux_score", 1.0 if ttl > 0 and ttl < 60 else 0.0))
        return {
            "dns_ttl_seconds": ttl,
            "dns_ttl_known": 1.0 if dns_data.get("ttl_seconds") is not None else 0.0,
            "dns_a_record_count": a_count,
            "dns_aaaa_record_count": aaaa_count,
            "dns_total_record_count": _safe_float(dns_data.get("total_record_count", a_count + aaaa_count), minimum=0.0, maximum=20_000.0),
            "dns_unique_ip_count": unique_ip_count,
            "dns_fast_flux_score": fast_flux_score,
            "dns_is_parked": 1.0 if _safe_bool(dns_data.get("is_parked")) else 0.0,
            "dns_is_sinkholed": 1.0 if _safe_bool(dns_data.get("is_sinkholed")) else 0.0,
            "dns_has_mx": 1.0 if _safe_bool(dns_data.get("has_mx")) else 0.0,
            "dns_has_txt": 1.0 if _safe_bool(dns_data.get("has_txt")) else 0.0,
            "dns_asn_known": 1.0 if dns_data.get("asn") is not None else 0.0,
        }

    # ------------------------------------------------------------------
    # TLS features (unchanged - defensive)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_ssl_features(ssl_data: Mapping[str, Any]) -> Dict[str, float]:
        cert_age = _safe_float(ssl_data.get("cert_age_days"), default=DEFAULT_CERT_AGE_DAYS, minimum=0.0, maximum=20_000.0)
        days_to_expiry = _safe_float(ssl_data.get("days_to_expiry"), default=0.0, minimum=-20_000.0, maximum=20_000.0)
        return {
            "ssl_valid": 1.0 if _safe_bool(ssl_data.get("valid")) else 0.0,
            "ssl_hostname_matches": 1.0 if _safe_bool(ssl_data.get("hostname_matches")) else 0.0,
            "ssl_cert_age_days": cert_age,
            "ssl_cert_age_known": 1.0 if ssl_data.get("cert_age_days") is not None else 0.0,
            "ssl_days_to_expiry": days_to_expiry,
            "ssl_is_expired": 1.0 if _safe_bool(ssl_data.get("is_expired", days_to_expiry < 0)) else 0.0,
            "ssl_is_self_signed": 1.0 if _safe_bool(ssl_data.get("is_self_signed")) else 0.0,
            "ssl_is_dv": 1.0 if _safe_bool(ssl_data.get("is_dv")) else 0.0,
            "ssl_is_ev": 1.0 if _safe_bool(ssl_data.get("is_ev")) else 0.0,
            "ssl_is_wildcard": 1.0 if _safe_bool(ssl_data.get("is_wildcard")) else 0.0,
            "ssl_chain_valid": 1.0 if _safe_bool(ssl_data.get("chain_valid")) else 0.0,
            "ssl_issuer_known": 1.0 if ssl_data.get("issuer") else 0.0,
        }

    # ------------------------------------------------------------------
    # HTTP / redirect features (unchanged - defensive)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_http_features(http_data: Mapping[str, Any]) -> Dict[str, float]:
        return {
            "http_status_code": _safe_float(http_data.get("status_code"), minimum=0.0, maximum=599.0),
            "http_redirect_count": _safe_float(http_data.get("redirect_count"), minimum=0.0, maximum=MAX_REDIRECTS),
            "http_external_redirect_count": _safe_float(http_data.get("external_redirect_count"), minimum=0.0, maximum=MAX_REDIRECTS),
            "http_final_host_changed": 1.0 if _safe_bool(http_data.get("final_host_changed")) else 0.0,
            "http_has_download": 1.0 if _safe_bool(http_data.get("has_download")) else 0.0,
            "http_content_length": _safe_float(http_data.get("content_length"), minimum=0.0, maximum=2_000_000_000.0),
            "http_response_time_ms": _safe_float(http_data.get("response_time_ms"), minimum=0.0, maximum=300_000.0),
        }

    # ------------------------------------------------------------------
    # DOM features (unchanged - defensive)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_dom_features(dom_data: Mapping[str, Any]) -> Dict[str, float]:
        return {
            "dom_has_login_form": 1.0 if _safe_bool(dom_data.get("has_login_form")) else 0.0,
            "dom_has_password_input": 1.0 if _safe_bool(dom_data.get("has_password_input")) else 0.0,
            "dom_has_external_form_action": 1.0 if _safe_bool(dom_data.get("has_external_form_action")) else 0.0,
            "dom_external_resource_count": _safe_float(dom_data.get("external_resource_count"), minimum=0.0, maximum=100_000.0),
            "dom_external_script_count": _safe_float(dom_data.get("external_script_count"), minimum=0.0, maximum=100_000.0),
            "dom_iframe_count": _safe_float(dom_data.get("iframe_count"), minimum=0.0, maximum=100_000.0),
            "dom_hidden_element_count": _safe_float(dom_data.get("hidden_element_count"), minimum=0.0, maximum=100_000.0),
            "dom_has_hidden_elements": 1.0 if _safe_bool(dom_data.get("has_hidden_elements")) else 0.0,
            "dom_has_anti_analysis_script": 1.0 if _safe_bool(dom_data.get("has_anti_analysis_script")) else 0.0,
            "dom_has_download_link": 1.0 if _safe_bool(dom_data.get("has_download_link")) else 0.0,
            "dom_form_count": _safe_float(dom_data.get("form_count"), minimum=0.0, maximum=100_000.0),
            "dom_external_form_count": _safe_float(dom_data.get("external_form_count"), minimum=0.0, maximum=100_000.0),
        }

    # ------------------------------------------------------------------
    # Visual features (UPDATED with brand)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_visual_features(visual_data: Mapping[str, Any]) -> Dict[str, float]:
        return {
            "visual_similarity_score": _clamp_probability(visual_data.get("similarity_score"), default=0.0),
            "visual_brand_match": 1.0 if _safe_bool(visual_data.get("brand_match")) else 0.0,
            "visual_analysis_available": 1.0 if visual_data else 0.0,
        }

    # ------------------------------------------------------------------
    # Finding aggregation (UPDATED)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_finding_features(findings: Iterable[Finding]) -> Dict[str, float]:
        finding_list = list(findings)[:MAX_FINDINGS_INSPECTED]
        critical = high = medium = low = exploit = credential = brand = malware = infrastructure = correlation = 0
        for f in finding_list:
            severity = str(getattr(f, "severity", "")).upper()
            category = str(getattr(f, "category", "")).upper()
            if "CRITICAL" in severity:
                critical += 1
            elif "HIGH" in severity:
                high += 1
            elif "MEDIUM" in severity:
                medium += 1
            elif "LOW" in severity:
                low += 1
            if category == "EXPLOIT_ATTEMPT":
                exploit += 1
            elif category == "CREDENTIAL_HARVESTING":
                credential += 1
            elif category == "BRAND_IMPERSONATION":
                brand += 1
            elif category == "MALWARE_DELIVERY":
                malware += 1
            elif category == "SUSPICIOUS_INFRASTRUCTURE":
                infrastructure += 1
            elif category == "THREAT_CORRELATION":
                correlation += 1
        return {
            "finding_count": float(len(finding_list)),
            "critical_finding_count": float(critical),
            "high_finding_count": float(high),
            "medium_finding_count": float(medium),
            "low_finding_count": float(low),
            "exploit_finding_count": float(exploit),
            "credential_harvesting_finding_count": float(credential),
            "brand_impersonation_finding_count": float(brand),
            "malware_delivery_finding_count": float(malware),
            "suspicious_infrastructure_finding_count": float(infrastructure),
            "threat_correlation_finding_count": float(correlation),
        }

    # ------------------------------------------------------------------
    # Vector construction (UPDATED)
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_feature_vector() -> Dict[str, float]:
        return {name: 0.0 for name in FEATURE_NAMES}

    @staticmethod
    def _sanitize_feature_vector(features: Mapping[str, Any]) -> Dict[str, float]:
        sanitized: Dict[str, float] = {}
        for name in FEATURE_NAMES:
            value = features.get(name, 0.0)
            sanitized[name] = _safe_float(value, default=0.0, minimum=-MAX_NUMERIC_FEATURE, maximum=MAX_NUMERIC_FEATURE)
        return sanitized


# ============================================================================
# EXPORTS
# ============================================================================
__all__ = [
    "FEATURE_NAMES",
    "URL_FEATURE_EXTRACTOR_VERSION",
    "UrlFeatureExtractor",
    "URLLexicalAnalyzer",
]
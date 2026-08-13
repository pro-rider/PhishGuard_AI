"""
PhishGuard_AI - Enterprise URL Structural Parser & Threat Engine
Production-grade analyzer performing canonical URL extraction, RFC 3986
normalization, public-suffix domain deconstruction, Unicode homograph
spoofing detection, SSRF protection, Shannon entropy analysis, and
defanged URL recovery.

Key Security Controls
---------------------
1. RFC 3986 canonicalization + robust defanged URL recovery.
2. Homograph / Punycode threat detection (expanded).
3. SSRF & cloud metadata defense (IPv4/IPv6 + encoded forms).
4. Public-suffix deconstruction via tldextract (with safe fallback).
5. Shannon entropy analysis on subdomains.
6. Zero-leakage credential redaction.
7. Calibrated per-category scoring with damping.
"""

from __future__ import annotations

import html
import ipaddress
import logging
import math
import re
import time
import unicodedata
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

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

logger = logging.getLogger(__name__)

# Optional but recommended
try:
    import tldextract
    _TLDEXTRACT_AVAILABLE = True
except ImportError:
    tldextract = None  # type: ignore
    _TLDEXTRACT_AVAILABLE = False

try:
    import idna
except ImportError:
    idna = None  # type: ignore

# ============================================================================
# CONSTANTS & BOUNDS
# ============================================================================

URL_PARSER_VERSION = "2.3.0"
RULESET_VERSION = "2026.08.13"

MAX_URL_COUNT = 80
MAX_URL_LENGTH = 2048
MAX_FINDINGS = 25
HIGH_ENTROPY_THRESHOLD = 4.35

# Fallback multi-part suffixes when tldextract is unavailable
FALLBACK_PUBLIC_SUFFIXES: Set[str] = {
    "co.uk", "org.uk", "gov.uk", "me.uk", "ac.uk",
    "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "co.jp", "ne.jp", "or.jp", "go.jp",
    "co.nz", "net.nz", "org.nz",
    "com.br", "gov.br", "co.in", "net.in", "gov.in",
    "github.io", "gitlab.io", "pages.dev", "vercel.app",
    "azurewebsites.net", "azureedge.net", "amazonaws.com",
    "cloudfront.net", "appspot.com", "herokuapp.com",
}

DANGEROUS_SCHEMES: Set[str] = {
    "data", "javascript", "file", "view-source", "vbscript",
    "about", "blob", "ws", "wss",
}

# Expanded homoglyph map
HOMOGLYPH_ASCII_MAP: Dict[str, str] = {
    # Cyrillic
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "һ": "h", "ԁ": "d", "ɡ": "g",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I",
    # Greek
    "α": "a", "β": "b", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o",
    "ρ": "p", "τ": "t", "υ": "y", "χ": "x",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Full-width
    **{chr(0xFF21 + i): chr(65 + i) for i in range(26)},
    **{chr(0xFF41 + i): chr(97 + i) for i in range(26)},
}

# URL extraction (more complete)
RAW_URL_REGEX = re.compile(
    r"""(?ix)
    (?:
        (?:https?|ftp|hxxps?|h__ps?|fxp|htxps?):// |
        (?<=\s)www\.
    )
    [^\s<>"'()\[\]{}]+
    """
)

# Defang patterns
DEFANGED_SCHEME_RE = re.compile(
    r"(?i)^(?:hxxps?|h__ps?|hxxp|fxp|htxps?)[:\(\[\{]*/*"
)
DEFANGED_DOT_RE = re.compile(
    r"\s*(?:\[\.\]|\(\.\)|\[dot\]|\(dot\)|\s+dot\s+)\s*", re.I
)
DEFANGED_AT_RE = re.compile(
    r"\s*(?:\[@\]|\(@\)|\[at\]|\(at\)|\s+at\s+)\s*", re.I
)
DEFANGED_COLON_RE = re.compile(r"\[:\]|\(:\)", re.I)

# ============================================================================
# DATA MODELS
# ============================================================================

class HostType(str, Enum):
    DOMAIN = "domain"
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class ParsedURLStructure:
    raw_url: str
    canonical_url: str
    original_url: str
    scheme: str
    host_raw: str
    host_normalized: str
    host_type: HostType
    subdomain: str
    org_domain: str
    public_suffix: str
    port: Optional[int]
    path: str
    query_string: str
    has_user_info: bool
    is_defanged: bool
    is_dangerous_scheme: bool
    is_homograph: bool
    is_punycode: bool
    is_ssrf_risk: bool
    ssrf_reason: str
    entropy_score: float


# ============================================================================
# HELPERS
# ============================================================================

def redact_userinfo(url: str) -> str:
    """Strip credentials from a URL for safe logging / metadata."""
    try:
        p = urllib.parse.urlsplit(url)
        if not (p.username or p.password):
            return url
        # Rebuild without userinfo
        netloc = p.hostname or ""
        if p.port:
            netloc = f"{netloc}:{p.port}"
        return urllib.parse.urlunsplit((p.scheme, netloc, p.path, p.query, p.fragment))
    except Exception:
        return re.sub(r"://[^/]*@", "://***:***@", url)


# ============================================================================
# DEFANGED URL DECODER
# ============================================================================

class DefangedURLDecoder:
    @classmethod
    def recover(cls, raw: str) -> Tuple[str, bool]:
        if not raw:
            return "", False
        working = raw.strip()
        is_defanged = False

        if (
            DEFANGED_SCHEME_RE.search(working)
            or DEFANGED_DOT_RE.search(working)
            or DEFANGED_COLON_RE.search(working)
            or DEFANGED_AT_RE.search(working)
        ):
            is_defanged = True
            working = DEFANGED_SCHEME_RE.sub(
                lambda m: "https://" if "s" in m.group(0).lower() else "http://",
                working,
            )
            working = DEFANGED_DOT_RE.sub(".", working)
            working = DEFANGED_AT_RE.sub("@", working)
            working = DEFANGED_COLON_RE.sub(":", working)

        # Backslash confusion
        if "://" not in working and ":\\\\" in working:
            working = working.replace(":\\\\", "://")
            is_defanged = True
        if "\\" in working and "://" in working:
            working = working.replace("\\", "/")
            is_defanged = True

        return working, is_defanged


# ============================================================================
# UNICODE / HOMOGRAPH DETECTOR
# ============================================================================

class UnicodeThreatDetector:
    @classmethod
    def analyze(cls, domain: str) -> Tuple[bool, bool]:
        if not domain:
            return False, False

        is_punycode = "xn--" in domain.lower()
        is_homograph = False
        scripts: Set[str] = set()

        for ch in domain:
            if ch in HOMOGLYPH_ASCII_MAP:
                is_homograph = True
            if ch.isalpha():
                try:
                    name = unicodedata.name(ch, "")
                    if name:
                        scripts.add(name.split()[0])
                except Exception:
                    pass

        if len(scripts) > 1 and ({"CYRILLIC", "GREEK", "ARMENIAN"} & scripts):
            is_homograph = True

        return is_homograph, is_punycode


# ============================================================================
# SSRF PROTECTION
# ============================================================================

class SSRFProtectionEngine:
    CLOUD_IMDS = {
        "169.254.169.254",
        "169.254.170.2",          # AWS ECS
        "fd00:ec2::254",          # AWS IPv6 IMDS
    }

    @classmethod
    def evaluate(cls, host: str, host_type: HostType) -> Tuple[bool, str]:
        if not host:
            return False, "NONE"

        h = host.strip().lower().rstrip(".")

        if h in ("localhost", "localhost.localdomain") or h.endswith(".localhost"):
            return True, "LOOPBACK_HOSTNAME"
        if h in cls.CLOUD_IMDS:
            return True, "CLOUD_METADATA_IMDS"

        # Try direct IP parse
        try:
            ip = ipaddress.ip_address(h)
            if ip.is_private:
                return True, "PRIVATE_IP"
            if ip.is_loopback:
                return True, "LOOPBACK_IP"
            if ip.is_link_local:
                return True, "LINK_LOCAL_IP"
            if ip.is_reserved:
                return True, "RESERVED_IP"
            return False, "NONE"
        except ValueError:
            pass

        # Decimal / octal / hex encoded IPv4 tricks (e.g. 2130706433, 0x7f000001)
        if host_type == HostType.DOMAIN and re.fullmatch(r"(?:0x[0-9a-f]+|\d+)", h):
            try:
                # Possible integer form of IPv4
                val = int(h, 0)
                if 0 <= val <= 0xFFFFFFFF:
                    ip = ipaddress.IPv4Address(val)
                    if ip.is_private or ip.is_loopback or ip.is_link_local:
                        return True, "ENCODED_PRIVATE_IP"
            except Exception:
                pass

        return False, "NONE"


# ============================================================================
# ENTROPY
# ============================================================================

class URLEntropyAnalyzer:
    @classmethod
    def shannon(cls, text: str) -> float:
        if not text or len(text) < 4:
            return 0.0
        freq: Dict[str, int] = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1
        length = len(text)
        entropy = -sum(
            (c / length) * math.log2(c / length) for c in freq.values()
        )
        return round(entropy, 3)


# ============================================================================
# PUBLIC SUFFIX ENGINE
# ============================================================================

class PublicSuffixEngine:
    @classmethod
    def deconstruct(cls, raw_host: str) -> Tuple[str, HostType, str, str, str]:
        """
        Returns: (normalized_host, host_type, subdomain, org_domain, public_suffix)
        """
        if not raw_host:
            return "", HostType.MALFORMED, "", "", ""

        host = raw_host.strip().lower().rstrip(".")
        # Strip zone ID from IPv6
        if "%" in host:
            host = host.split("%", 1)[0]

        # IP?
        try:
            ip = ipaddress.ip_address(host)
            htype = HostType.IPV4 if isinstance(ip, ipaddress.IPv4Address) else HostType.IPV6
            return host, htype, "", host, ""
        except ValueError:
            pass

        # IDNA normalize
        try:
            if idna:
                norm = idna.encode(host).decode("ascii")
            else:
                norm = host
        except Exception:
            norm = host

        if _TLDEXTRACT_AVAILABLE and tldextract is not None:
            ext = tldextract.extract(norm)
            subdomain = ext.subdomain or ""
            org_domain = f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else norm
            public_suffix = ext.suffix or ""
            return norm, HostType.DOMAIN, subdomain, org_domain, public_suffix

        # Fallback hand-rolled
        parts = norm.split(".")
        if len(parts) == 1:
            return norm, HostType.DOMAIN, "", norm, ""

        two = ".".join(parts[-2:])
        if two in FALLBACK_PUBLIC_SUFFIXES and len(parts) >= 3:
            return (
                norm,
                HostType.DOMAIN,
                ".".join(parts[:-3]),
                ".".join(parts[-3:]),
                two,
            )
        return (
            norm,
            HostType.DOMAIN,
            ".".join(parts[:-2]),
            ".".join(parts[-2:]),
            parts[-1],
        )


# ============================================================================
# URL CANONICALIZER
# ============================================================================

class URLCanonicalizer:
    @classmethod
    @lru_cache(maxsize=2048)
    def parse(cls, raw_url: str) -> Optional[ParsedURLStructure]:
        if not raw_url or len(raw_url) > MAX_URL_LENGTH:
            return None

        recovered, is_defanged = DefangedURLDecoder.recover(raw_url)
        working = html.unescape(html.unescape(recovered)).strip()

        if working.startswith("//"):
            working = "http:" + working
        if working.lower().startswith("www.") and "://" not in working:
            working = "http://" + working

        try:
            parsed = urllib.parse.urlsplit(working)
        except Exception:
            return None

        scheme = (parsed.scheme or "http").lower()
        is_dangerous = scheme in DANGEROUS_SCHEMES

        host_str = (parsed.hostname or "").strip()
        port = parsed.port

        # Userinfo detection (including encoded forms)
        has_user_info = bool(parsed.username or parsed.password)
        netloc_decoded = urllib.parse.unquote(parsed.netloc or "")
        if "@" in netloc_decoded or "%40" in (parsed.netloc or "").lower():
            has_user_info = True

        norm_host, host_type, subdomain, org_domain, public_suffix = (
            PublicSuffixEngine.deconstruct(host_str)
        )

        is_homograph, is_punycode = UnicodeThreatDetector.analyze(norm_host)
        is_ssrf, ssrf_reason = SSRFProtectionEngine.evaluate(norm_host, host_type)

        # Entropy prefers subdomain; fall back carefully
        entropy_target = subdomain if len(subdomain) >= 6 else ""
        entropy_score = URLEntropyAnalyzer.shannon(entropy_target)

        # Path normalisation
        raw_path = urllib.parse.unquote(parsed.path or "/")
        # Collapse multiple slashes and resolve simple traversal for analysis
        norm_path = re.sub(r"/+", "/", raw_path)
        norm_path = norm_path.replace("\\", "/")

        port_part = f":{port}" if port and port not in (80, 443) else ""
        canonical = f"{scheme}://{norm_host}{port_part}{norm_path}"
        if parsed.query:
            canonical += f"?{parsed.query}"

        return ParsedURLStructure(
            raw_url=raw_url,
            canonical_url=redact_userinfo(canonical),
            original_url=redact_userinfo(working),
            scheme=scheme,
            host_raw=host_str,
            host_normalized=norm_host,
            host_type=host_type,
            subdomain=subdomain,
            org_domain=org_domain,
            public_suffix=public_suffix,
            port=port,
            path=norm_path,
            query_string=parsed.query or "",
            has_user_info=has_user_info,
            is_defanged=is_defanged,
            is_dangerous_scheme=is_dangerous,
            is_homograph=is_homograph,
            is_punycode=is_punycode,
            is_ssrf_risk=is_ssrf,
            ssrf_reason=ssrf_reason,
            entropy_score=entropy_score,
        )


# ============================================================================
# PAYLOAD STORE
# ============================================================================

class URLPayloadStore:
    @staticmethod
    def resolve(context: ScanContext) -> List[str]:
        candidates: List[str] = []
        meta = getattr(context, "metadata", None)
        data: Dict[str, Any] = getattr(meta, "data", {}) if meta else {}

        # 1. Explicit URL (pure URL scan)
        for key in ("url", "target_url", "primary_url"):
            val = data.get(key) or getattr(context, key, None)
            if isinstance(val, str) and val.strip():
                candidates.append(val.strip())

        # 2. Pre-extracted list
        explicit = data.get("extracted_urls") or data.get("urls") or []
        if isinstance(explicit, list):
            candidates.extend(str(u) for u in explicit if isinstance(u, str))

        # 3. Body extraction
        plain = data.get("body_plain") or data.get("text_plain") or ""
        html_body = data.get("body_html") or data.get("text_html") or ""
        combined = f"{plain}\n{html_body}"
        if combined.strip():
            candidates.extend(RAW_URL_REGEX.findall(combined))

        # Dedup + limit
        seen: Set[str] = set()
        out: List[str] = []
        for u in candidates:
            u = u.strip()
            if not u or u in seen or len(u) > MAX_URL_LENGTH:
                continue
            seen.add(u)
            out.append(u)
            if len(out) >= MAX_URL_COUNT:
                break
        return out


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class URLParser(BaseAnalyzer):
    name: str = "url_parser"
    version: str = URL_PARSER_VERSION
    supported_scan_types: Set[ScanType] = {
        ScanType.URL, ScanType.EMAIL, ScanType.COMPREHENSIVE
    }
    required_inputs: Set[str] = set()          # flexible: url or email body
    resource_class: AnalyzerResourceClass = AnalyzerResourceClass.CPU_BOUND
    priority: int = 5
    can_hard_override: bool = False

    async def analyze(
        self, context: ScanContext, deadline_monotonic: float
    ) -> AnalyzerResult:
        start = time.monotonic()

        if time.monotonic() >= deadline_monotonic:
            return self._timeout(start)

        candidates = URLPayloadStore.resolve(context)
        if not candidates:
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=0.0,
                confidence=0.88,
                findings=[],
            )

        findings: List[Finding] = []
        cat_scores: Dict[str, float] = {
            "IMPERSONATION": 0.0,
            "EXPLOIT_ATTEMPT": 0.0,
            "DECEPTION": 0.0,
            "SUSPICIOUS_INFRASTRUCTURE": 0.0,
            "OBFUSCATION": 0.0,
        }
        seen_rules: Set[Tuple[str, str]] = set()

        def emit(
            rule_id: str,
            category: str,
            severity: SecurityVerdict,
            confidence: float,
            message: str,
            points: float,
            domain_key: str,
            meta: Optional[Dict[str, Any]] = None,
        ) -> None:
            key = (rule_id, domain_key)
            if key in seen_rules or len(findings) >= MAX_FINDINGS:
                return
            seen_rules.add(key)
            damp = 0.7 ** sum(1 for f in findings if f.rule_id == rule_id)
            cat_scores[category] = cat_scores.get(category, 0.0) + points * damp
            findings.append(
                Finding(
                    rule_id=rule_id,
                    category=category,
                    severity=severity,
                    confidence=confidence,
                    message=message,
                    metadata=ConstrainedMetadata(meta or {}),
                )
            )

        try:
            for raw in candidates:
                if time.monotonic() >= deadline_monotonic:
                    return self._timeout(start)

                parsed = URLCanonicalizer.parse(raw)
                if not parsed:
                    continue

                domain_key = parsed.org_domain or parsed.host_normalized

                # Dangerous scheme
                if parsed.is_dangerous_scheme:
                    emit(
                        "URL.SCHEME.DANGEROUS",
                        "EXPLOIT_ATTEMPT",
                        SecurityVerdict.HIGH,
                        0.97,
                        f"Dangerous URI scheme '{parsed.scheme}' in URL.",
                        40.0,
                        domain_key,
                        {"scheme": parsed.scheme, "host": parsed.host_normalized},
                    )

                # Defanged
                if parsed.is_defanged:
                    emit(
                        "URL.OBFUSCATION.DEFANGED",
                        "OBFUSCATION",
                        SecurityVerdict.LOW,
                        0.90,
                        f"Defanged/obfuscated URL recovered for host '{parsed.host_normalized}'.",
                        8.0,
                        domain_key,
                        {"host": parsed.host_normalized},
                    )

                # Homograph
                if parsed.is_homograph:
                    emit(
                        "URL.IDN.HOMOGRAPH_ATTACK",
                        "IMPERSONATION",
                        SecurityVerdict.CRITICAL,
                        0.98,
                        f"Homograph spoofing detected in '{parsed.host_normalized}'.",
                        48.0,
                        domain_key,
                        {
                            "domain": parsed.host_normalized,
                            "is_punycode": parsed.is_punycode,
                            "ruleset_version": RULESET_VERSION,
                        },
                    )

                # SSRF
                if parsed.is_ssrf_risk:
                    emit(
                        "URL.SSRF.PRIVATE_OR_METADATA",
                        "EXPLOIT_ATTEMPT",
                        SecurityVerdict.HIGH,
                        0.96,
                        f"SSRF risk ({parsed.ssrf_reason}) for host '{parsed.host_normalized}'.",
                        44.0,
                        domain_key,
                        {
                            "target_host": parsed.host_normalized,
                            "reason": parsed.ssrf_reason,
                        },
                    )

                # Userinfo deception
                if parsed.has_user_info:
                    emit(
                        "URL.DECEPTION.USERINFO",
                        "DECEPTION",
                        SecurityVerdict.HIGH,
                        0.95,
                        f"Userinfo / '@' deception routing to '{parsed.host_normalized}'.",
                        36.0,
                        domain_key,
                        {"target_host": parsed.host_normalized},
                    )

                # Path traversal
                if ".." in parsed.path or "%2e%2e" in parsed.path.lower():
                    emit(
                        "URL.PATH.TRAVERSAL",
                        "EXPLOIT_ATTEMPT",
                        SecurityVerdict.MEDIUM,
                        0.90,
                        f"Path traversal sequence in '{parsed.path[:80]}'.",
                        22.0,
                        domain_key,
                        {"path_sample": parsed.path[:80]},
                    )

                # High entropy subdomain
                if parsed.entropy_score >= HIGH_ENTROPY_THRESHOLD:
                    emit(
                        "URL.STRUCTURE.HIGH_ENTROPY",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.MEDIUM,
                        0.84,
                        f"High entropy ({parsed.entropy_score}) in subdomain of '{parsed.host_normalized}'.",
                        16.0,
                        domain_key,
                        {
                            "domain": parsed.host_normalized,
                            "subdomain": parsed.subdomain[:64],
                            "entropy": parsed.entropy_score,
                        },
                    )

            total = (
                min(cat_scores["IMPERSONATION"], 50.0)
                + min(cat_scores["EXPLOIT_ATTEMPT"], 50.0)
                + min(cat_scores["DECEPTION"], 40.0)
                + min(cat_scores["SUSPICIOUS_INFRASTRUCTURE"], 25.0)
                + min(cat_scores["OBFUSCATION"], 15.0)
            )
            score = min(max(total, 0.0), 100.0)

            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=score,
                confidence=0.94 if findings else 0.86,
                findings=findings,
            )

        except Exception as exc:
            logger.exception(
                "URLParser failure scan_id=%s", getattr(context, "scan_id", "?")
            )
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.FAILED,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                sanitized_error=f"URL parsing error: {type(exc).__name__}",
            )

    def _timeout(self, start: float) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.TIMEOUT,
            execution_time_ms=(time.monotonic() - start) * 1000.0,
            sanitized_error="Deadline exceeded during URL structural parsing.",
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "URLParser",
    "URLCanonicalizer",
    "DefangedURLDecoder",
    "PublicSuffixEngine",
    "SSRFProtectionEngine",
    "UnicodeThreatDetector",
    "URLEntropyAnalyzer",
    "URLPayloadStore",
    "ParsedURLStructure",
    "HostType",
    "URL_PARSER_VERSION",
    "redact_userinfo",
]
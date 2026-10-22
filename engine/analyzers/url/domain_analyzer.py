"""
PhishGuard_AI - Enterprise Domain Threat & Brand Protection Engine
Production-grade domain analyzer for brand impersonation, typosquatting,
IDN homographs, high-entropy names, risky TLDs, newly-registered domains,
and subdomain abuse.

Key Controls
------------
1. Bounded Damerau-Levenshtein + keyboard / visual similarity.
2. Expanded IDN / homograph detection with script mixing.
3. Multi-token brand + action-word impersonation engine.
4. Public-suffix aware TLD risk scoring.
5. Subdomain depth & entropy analysis.
6. Calibrated multi-factor scoring with damping.
7. Safe thread-pool lifecycle.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import re
import time
import unicodedata
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

try:
    import idna
except ImportError:
    idna = None  # type: ignore

try:
    import tldextract
    _HAS_TLDEXTRACT = True
except ImportError:
    tldextract = None  # type: ignore
    _HAS_TLDEXTRACT = False

# ============================================================================
# CONSTANTS
# ============================================================================

DOMAIN_ANALYZER_VERSION = "2.3.0"
RULESET_VERSION = "2026.08.13"

MAX_DOMAINS = 40
MAX_SUBDOMAIN_DEPTH = 5
MAX_EDIT_DISTANCE = 2
MAX_FINDINGS = 20
HIGH_ENTROPY_THRESHOLD = 4.30

# Risky / frequently abused TLDs (final label or multi-part)
TLD_RISK_SCORES: Dict[str, float] = {
    "zip": 0.92, "mov": 0.90, "top": 0.85, "xyz": 0.82, "cfd": 0.85,
    "work": 0.75, "club": 0.72, "country": 0.80, "kim": 0.75, "stream": 0.75,
    "gq": 0.88, "ml": 0.88, "cf": 0.88, "ga": 0.88, "tk": 0.88, "cc": 0.70,
    "icu": 0.78, "sbs": 0.80, "cyou": 0.78, "buzz": 0.70, "rest": 0.75,
    "com": 0.05, "org": 0.05, "net": 0.08, "edu": 0.01, "gov": 0.01,
    "io": 0.18, "co": 0.25, "ai": 0.20, "app": 0.15, "dev": 0.12,
}

# High-value brands → legitimate registered domains
PROTECTED_BRANDS: Dict[str, Set[str]] = {
    "paypal": {"paypal.com", "paypal-objects.com", "paypalobjects.com"},
    "microsoft": {
        "microsoft.com", "office.com", "office365.com", "live.com",
        "outlook.com", "azure.com", "microsoftonline.com", "msn.com",
    },
    "google": {
        "google.com", "gmail.com", "googlemail.com", "accounts.google.com",
        "googleworkspace.com", "withgoogle.com",
    },
    "apple": {"apple.com", "icloud.com", "me.com", "mac.com"},
    "amazon": {"amazon.com", "aws.amazon.com", "amazonaws.com"},
    "docusign": {"docusign.com", "docusign.net"},
    "chase": {"chase.com"},
    "binance": {"binance.com"},
    "coinbase": {"coinbase.com"},
    "netflix": {"netflix.com"},
    "facebook": {"facebook.com", "fb.com", "meta.com"},
    "linkedin": {"linkedin.com"},
    "dropbox": {"dropbox.com"},
    "adobe": {"adobe.com"},
    "okta": {"okta.com"},
}

ACTION_TOKENS: Set[str] = {
    "login", "signin", "sign-in", "log-in", "verify", "verification",
    "secure", "security", "account", "update", "billing", "support",
    "service", "portal", "auth", "authenticate", "confirm", "validation",
    "wallet", "recover", "unlock", "limited", "suspended",
}

# Homoglyph / look-alike map (expanded)
HOMOGLYPH_MAP: Dict[str, str] = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "һ": "h", "ԁ": "d", "ɡ": "g",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I",
    "α": "a", "β": "b", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o",
    "ρ": "p", "τ": "t", "υ": "y", "χ": "x",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    **{chr(0xFF21 + i): chr(65 + i) for i in range(26)},
    **{chr(0xFF41 + i): chr(97 + i) for i in range(26)},
}

# Common visual / keyboard substitutions for typosquat
VISUAL_CONFUSABLES: Dict[str, str] = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
    "7": "t", "@": "a", "$": "s", "!": "i",
    "l": "i", "I": "l", "O": "0", "S": "5",
}

# Free / dynamic DNS & bulletproof-ish providers (high risk when used for auth pages)
RISKY_DNS_PROVIDERS: Set[str] = {
    "duckdns.org", "no-ip.com", "no-ip.org", "hopto.org", "zapto.org",
    "sytes.net", "ddns.net", "dynu.com", "ngrok.io", "localtunnel.me",
    "serveo.net", "pagekite.me",
}

# ============================================================================
# NORMALISATION HELPERS
# ============================================================================

def normalize_label(text: str) -> str:
    """Lowercase, strip homoglyphs, map visual confusables, remove hyphens for comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    out = []
    for ch in text:
        ch = HOMOGLYPH_MAP.get(ch, ch)
        ch = VISUAL_CONFUSABLES.get(ch, ch)
        if ch.isalnum():
            out.append(ch)
    return "".join(out)


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in re.split(r"[\._\-\s]+", text.lower()) if t and len(t) > 1]


# ============================================================================
# BOUNDED DAMERAU-LEVENSHTEIN
# ============================================================================

class TyposquatEngine:
    """Space-efficient bounded Damerau-Levenshtein with early exit."""

    @classmethod
    @lru_cache(maxsize=8192)
    def distance(cls, s1: str, s2: str, max_dist: int = MAX_EDIT_DISTANCE) -> int:
        if s1 == s2:
            return 0
        n, m = len(s1), len(s2)
        if abs(n - m) > max_dist:
            return max_dist + 1
        if n == 0:
            return min(m, max_dist + 1)
        if m == 0:
            return min(n, max_dist + 1)

        # Ensure n >= m for slight optimisation
        if n < m:
            s1, s2 = s2, s1
            n, m = m, n

        prev = list(range(m + 1))
        curr = [0] * (m + 1)
        prev2 = [0] * (m + 1)

        for i in range(1, n + 1):
            curr[0] = i
            min_row = i
            for j in range(1, m + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                deletion = prev[j] + 1
                insertion = curr[j - 1] + 1
                substitution = prev[j - 1] + cost
                best = min(deletion, insertion, substitution)

                # Transposition
                if (
                    i > 1
                    and j > 1
                    and s1[i - 1] == s2[j - 2]
                    and s1[i - 2] == s2[j - 1]
                ):
                    best = min(best, prev2[j - 2] + 1)

                curr[j] = best
                if best < min_row:
                    min_row = best

            if min_row > max_dist:
                return max_dist + 1
            prev2, prev, curr = prev, curr, prev2

        return prev[m] if prev[m] <= max_dist else max_dist + 1


# ============================================================================
# HOMOGRAPH ENGINE
# ============================================================================

class UnicodeHomographEngine:
    @classmethod
    def analyze(cls, domain: str) -> Tuple[bool, bool, str]:
        if not domain:
            return False, False, ""

        clean = domain.strip().lower()
        is_punycode = "xn--" in clean
        decoded = clean

        if is_punycode and idna is not None:
            try:
                decoded = idna.decode(clean)
            except Exception:
                pass

        is_homograph = False
        scripts: Set[str] = set()

        for ch in decoded:
            if ch in HOMOGLYPH_MAP:
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

        return is_homograph, is_punycode, decoded


# ============================================================================
# BRAND PROTECTION
# ============================================================================

class BrandProtectionEngine:
    @classmethod
    def evaluate(
        cls, org_domain: str, subdomain: str
    ) -> Optional[Tuple[str, float, str]]:
        """
        Returns (brand, confidence, reason) or None.
        """
        if not org_domain:
            return None

        full = f"{subdomain}.{org_domain}".lower().strip(".")
        org_norm = normalize_label(org_domain.split(".")[0] if "." in org_domain else org_domain)
        tokens = tokenize(full)
        org_tokens = tokenize(org_domain)

        for brand, legit_domains in PROTECTED_BRANDS.items():
            # Skip if this *is* a legitimate domain for the brand
            if any(
                org_domain == d or org_domain.endswith("." + d)
                for d in legit_domains
            ):
                continue

            brand_norm = normalize_label(brand)

            # 1. Exact brand token present
            if brand in tokens or brand_norm in [normalize_label(t) for t in tokens]:
                has_action = bool(ACTION_TOKENS & set(tokens))
                conf = 0.96 if has_action else 0.89
                reason = (
                    "Brand token + social-engineering action word"
                    if has_action
                    else "Brand token embedded in domain"
                )
                return brand, conf, reason

            # 2. Typosquat on the registrable label
            if len(org_norm) >= 4 and len(brand_norm) >= 4:
                dist = TyposquatEngine.distance(org_norm, brand_norm, max_dist=2)
                if dist <= 2:
                    return (
                        brand,
                        0.93 if dist == 1 else 0.88,
                        f"Registrable label edit-distance {dist} from '{brand}'",
                    )

            # 3. Typosquat on individual tokens
            for tok in org_tokens:
                tnorm = normalize_label(tok)
                if len(tnorm) < 4:
                    continue
                dist = TyposquatEngine.distance(tnorm, brand_norm, max_dist=2)
                if dist <= 2:
                    return (
                        brand,
                        0.91 if dist == 1 else 0.86,
                        f"Token '{tok}' edit-distance {dist} from '{brand}'",
                    )

        return None


# ============================================================================
# ENTROPY
# ============================================================================

class DomainEntropyEngine:
    @classmethod
    def shannon(cls, text: str) -> float:
        if not text or len(text) < 5:
            return 0.0
        freq: Dict[str, int] = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1
        n = len(text)
        return round(-sum((c / n) * math.log2(c / n) for c in freq.values()), 3)


# ============================================================================
# PAYLOAD RESOLUTION
# ============================================================================

class DomainPayloadStore:
    @staticmethod
    def resolve(context: ScanContext) -> List[Tuple[str, str, str]]:
        """
        Returns list of (subdomain, org_domain, public_suffix).
        """
        results: List[Tuple[str, str, str]] = []
        seen: Set[str] = set()

        meta = getattr(context, "metadata", None)
        data: Dict[str, Any] = getattr(meta, "data", {}) if meta else {}

        def add(sub: str, org: str, suffix: str) -> None:
            org = (org or "").lower().strip(".")
            if not org or org in seen:
                return
            seen.add(org)
            results.append(((sub or "").lower(), org, (suffix or "").lower()))

        # 1. From upstream url_parser structures
        for item in data.get("parsed_urls") or []:
            if not isinstance(item, dict):
                continue
            add(
                str(item.get("subdomain") or ""),
                str(item.get("org_domain") or item.get("registered_domain") or ""),
                str(item.get("public_suffix") or item.get("suffix") or ""),
            )

        # 2. Explicit domain / domains list
        for key in ("domain", "org_domain", "registered_domain"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                sub, org, suf = DomainPayloadStore._split(val.strip())
                add(sub, org, suf)

        for val in data.get("domains") or data.get("extracted_domains") or []:
            if isinstance(val, str) and val.strip():
                sub, org, suf = DomainPayloadStore._split(val.strip())
                add(sub, org, suf)

        # 3. Fallback: raw URL host
        for key in ("url", "target_url"):
            val = data.get(key) or getattr(context, key, None)
            if isinstance(val, str) and "://" in val:
                try:
                    from urllib.parse import urlparse
                    host = urlparse(val).hostname or ""
                    if host:
                        sub, org, suf = DomainPayloadStore._split(host)
                        add(sub, org, suf)
                except Exception:
                    pass

        return results[:MAX_DOMAINS]

    @staticmethod
    def _split(host: str) -> Tuple[str, str, str]:
        host = host.lower().strip().rstrip(".")
        if _HAS_TLDEXTRACT and tldextract is not None:
            ext = tldextract.extract(host)
            sub = ext.subdomain or ""
            org = f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else host
            return sub, org, ext.suffix or ""
        parts = host.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:-2]), ".".join(parts[-2:]), parts[-1]
        return "", host, ""


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class DomainAnalyzer(BaseAnalyzer):
    name: str = "domain_analyzer"
    version: str = DOMAIN_ANALYZER_VERSION
    supported_scan_types: Set[ScanType] = {
        ScanType.DOMAIN, ScanType.URL, ScanType.EMAIL, ScanType.COMPREHENSIVE
    }
    required_inputs: Set[str] = set()
    resource_class: AnalyzerResourceClass = AnalyzerResourceClass.CPU_BOUND
    priority: int = 10
    can_hard_override: bool = False

    def __init__(self, thread_pool_workers: int = 2) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, thread_pool_workers),
            thread_name_prefix="phishguard_domain",
        )

    def __del__(self) -> None:
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def analyze(
        self, context: ScanContext, deadline_monotonic: float
    ) -> AnalyzerResult:
        start = time.monotonic()

        if time.monotonic() >= deadline_monotonic:
            return self._timeout(start)

        domains = DomainPayloadStore.resolve(context)
        if not domains:
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=0.0,
                confidence=0.85,
                findings=[],
            )

        findings: List[Finding] = []
        cat_scores: Dict[str, float] = {
            "IMPERSONATION": 0.0,
            "SUSPICIOUS_INFRASTRUCTURE": 0.0,
            "DECEPTION": 0.0,
        }
        seen: Set[Tuple[str, str]] = set()

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
            if key in seen or len(findings) >= MAX_FINDINGS:
                return
            seen.add(key)
            damp = 0.72 ** sum(1 for f in findings if f.rule_id == rule_id)
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
            loop = asyncio.get_running_loop()

            for subdomain, org_domain, public_suffix in domains:
                if time.monotonic() >= deadline_monotonic:
                    return self._timeout(start)

                domain_key = org_domain
                full_host = f"{subdomain}.{org_domain}".strip(".")

                # ----- Homograph -----
                is_homo, is_puny, decoded = UnicodeHomographEngine.analyze(org_domain)
                if is_homo:
                    emit(
                        "DOMAIN.IDN.HOMOGRAPH_ATTACK",
                        "IMPERSONATION",
                        SecurityVerdict.HIGH,
                        0.97,
                        f"IDN/homograph spoofing in '{org_domain}' (decoded: '{decoded}').",
                        42.0,
                        domain_key,
                        {
                            "org_domain": org_domain,
                            "decoded": decoded,
                            "punycode": is_puny,
                        },
                    )

                # ----- Brand impersonation (CPU) -----
                remaining = max(0.05, deadline_monotonic - time.monotonic())
                try:
                    brand_result = await asyncio.wait_for(
                        loop.run_in_executor(
                            self._executor,
                            BrandProtectionEngine.evaluate,
                            org_domain,
                            subdomain,
                        ),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    brand_result = None

                if brand_result:
                    brand, conf, reason = brand_result
                    emit(
                        "DOMAIN.BRAND_IMPERSONATION",
                        "IMPERSONATION",
                        SecurityVerdict.HIGH,
                        conf,
                        f"Domain '{full_host}' impersonates '{brand}' ({reason}).",
                        40.0,
                        domain_key,
                        {
                            "org_domain": org_domain,
                            "brand": brand,
                            "reason": reason,
                            "ruleset_version": RULESET_VERSION,
                        },
                    )

                # ----- Risky TLD -----
                # Prefer full public suffix when available
                tld_key = public_suffix.lower() if public_suffix else org_domain.split(".")[-1]
                tld_risk = TLD_RISK_SCORES.get(tld_key, TLD_RISK_SCORES.get(tld_key.split(".")[-1], 0.05))
                if tld_risk >= 0.70:
                    emit(
                        "DOMAIN.TLD.HIGH_RISK",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.MEDIUM,
                        0.90,
                        f"High-risk TLD '.{tld_key}' used by '{org_domain}'.",
                        18.0,
                        domain_key,
                        {"public_suffix": tld_key, "risk": tld_risk},
                    )

                # ----- Free / dynamic DNS -----
                if any(org_domain == p or org_domain.endswith("." + p) for p in RISKY_DNS_PROVIDERS):
                    emit(
                        "DOMAIN.DNS.DYNAMIC_OR_FREE",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.MEDIUM,
                        0.88,
                        f"Domain uses free/dynamic DNS provider: '{org_domain}'.",
                        16.0,
                        domain_key,
                        {"provider": org_domain},
                    )

                # ----- Newly registered / age -----
                age_map = None
                meta_data = getattr(getattr(context, "metadata", None), "data", {}) or {}
                age_map = meta_data.get("domain_ages") or {}
                creation_ts = (
                    age_map.get(org_domain)
                    or meta_data.get("domain_creation_timestamp")
                )
                if creation_ts is not None:
                    try:
                        c_ts = float(creation_ts)
                        now = time.time()
                        if c_ts > now + 86400:
                            emit(
                                "DOMAIN.AGE.INVALID_FUTURE",
                                "SUSPICIOUS_INFRASTRUCTURE",
                                SecurityVerdict.MEDIUM,
                                0.94,
                                f"Domain '{org_domain}' has future registration timestamp.",
                                20.0,
                                domain_key,
                                {"creation_ts": c_ts},
                            )
                        else:
                            age_days = int((now - c_ts) / 86400)
                            if age_days < 30:
                                sev = (
                                    SecurityVerdict.HIGH
                                    if age_days < 7
                                    else SecurityVerdict.MEDIUM
                                )
                                emit(
                                    "DOMAIN.AGE.NEWLY_REGISTERED",
                                    "SUSPICIOUS_INFRASTRUCTURE",
                                    sev,
                                    0.94,
                                    f"Domain '{org_domain}' is newly registered ({age_days} days).",
                                    28.0 if age_days < 7 else 20.0,
                                    domain_key,
                                    {"org_domain": org_domain, "age_days": age_days},
                                )
                    except (ValueError, TypeError):
                        pass

                # ----- Subdomain depth -----
                depth = len([p for p in subdomain.split(".") if p]) if subdomain else 0
                if depth >= MAX_SUBDOMAIN_DEPTH:
                    emit(
                        "DOMAIN.SUBDOMAIN.EXCESSIVE_DEPTH",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.LOW,
                        0.80,
                        f"Excessive subdomain depth ({depth}) on '{full_host}'.",
                        10.0,
                        domain_key,
                        {"depth": depth, "host": full_host[:80]},
                    )

                # ----- Entropy (prefer long subdomain) -----
                entropy_target = subdomain if len(subdomain) >= 8 else ""
                if not entropy_target and len(org_domain.split(".")[0]) >= 10:
                    entropy_target = org_domain.split(".")[0]
                entropy = DomainEntropyEngine.shannon(entropy_target)
                if entropy >= HIGH_ENTROPY_THRESHOLD:
                    emit(
                        "DOMAIN.HIGH_ENTROPY",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.MEDIUM,
                        0.84,
                        f"High entropy ({entropy}) in '{entropy_target[:40]}…'.",
                        14.0,
                        domain_key,
                        {"entropy": entropy, "target": entropy_target[:64]},
                    )

            # Final calibrated score
            total = (
                min(cat_scores["IMPERSONATION"], 55.0)
                + min(cat_scores["SUSPICIOUS_INFRASTRUCTURE"], 45.0)
                + min(cat_scores["DECEPTION"], 25.0)
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
                "DomainAnalyzer failure scan_id=%s", getattr(context, "scan_id", "?")
            )
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.FAILED,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                sanitized_error=f"Domain analysis error: {type(exc).__name__}",
            )

    def _timeout(self, start: float) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.TIMEOUT,
            execution_time_ms=(time.monotonic() - start) * 1000.0,
            sanitized_error="Deadline exceeded during domain analysis.",
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "DomainAnalyzer",
    "BrandProtectionEngine",
    "TyposquatEngine",
    "UnicodeHomographEngine",
    "DomainEntropyEngine",
    "DomainPayloadStore",
    "DOMAIN_ANALYZER_VERSION",
]

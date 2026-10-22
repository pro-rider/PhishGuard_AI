"""
PhishGuard_AI - Heuristic Content Analyzer

Self-contained, offline heuristic analyzer that scores raw email body text and
target URLs for common phishing indicators.

Unlike the RFC822-oriented ``HeaderAnalyzer`` / ``BodyAnalyzer`` (which expect
structured email headers and payload storage), this analyzer is designed for the
plain text the API gateway actually receives, so it always has something to work
with and never touches the network, a browser, or an ML model.

Security / reliability principles:
- Pure in-memory heuristics; zero external dependency and zero network egress.
- Deterministic, bounded rules; findings are capped in number/length.
- Contribution score is clamped to [0, 100] and combined with a data-driven
  confidence so it aggregates cleanly with the other registered analyzers.
"""

from __future__ import annotations

import math
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
# CONSTANTS & BOUNDS
# ============================================================================

ANALYZER_NAME = "heuristic_analyzer"
ANALYZER_VERSION = "1.0.0"
MAX_TEXT_CHARS = 64_000
MAX_FINDINGS = 40
SCORE_CAP = 100.0
CONF_DEFAULT = 0.82

# Email lexicon groups (word/phrase -> indicator weight).
_URGENCY_TERMS = (
    "immediately", "asap", "urgent", "act now", "right away", "within 24 hours",
    "account suspended", "will be deactivated", "expires today", "final notice",
    "overdue", "action required", "your account will be closed", "security alert",
)
_CREDENTIAL_BAIT_TERMS = (
    "verify your password", "confirm your password", "update your account",
    "enter your credentials", "click to verify", "re-enter your password",
    "unlock your account", "reset your password", "verify identity",
    "account has been suspended", "confirm your identity",
    "verify your account", "verify your account now", "verify your email",
    "click here to verify", "account verification", "confirm your account",
    "validate your account", "account will be locked",
)
_PRESSURE_TERMS = (
    "failure to respond", "will result in", "legal action", "lawsuit",
    "authorities", "police report", "fraud investigation", "if you do not act",
    "account will be terminated", "permanently blocked", "suspend your account",
)
_MONEY_TERMS = (
    "lottery", "inheritance", "wire transfer", "western union", "bitcoin",
    "cryptocurrency", "you have won", "prize", "refund", "gift card",
    "bank account", "account number", "ssn", "social security number",
)

# URL indicator terms.
_BAD_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".club", ".icu", ".zip",
    ".mov", ".click", ".link", ".stream", ".download", ".country", ".kim",
    ".work", ".party", ".rest", ".trade", ".loan", ".date", ".racing", ".win",
    ".bid", ".mom", ".monster", ".gdn",
}
_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "cutt.ly", "rb.gy",
    "ow.ly", "tiny.cc", "tr.im", "v.gd", "soo.gd", "shorturl.at", "buff.ly",
    "lnkd.in", "rebrand.ly", "bl.ink",
}
_PHISH_KEYWORDS = (
    "verify", "secure", "account", "login", "signin", "update", "confirm",
    "bank", "paypal", "apple", "microsoft", "support", "unlock", "suspend",
    "alert", "billing", "invoice", "password", "webscr", "signin",
)

# Regexes.
_LINK_PRESSURE_RE = re.compile(r"(click|follow|open)\s+(?:the\s+)?(?:this\s+)?(?:secure\s+)?link", re.I)
_LINK_TAG_RE = re.compile(r"https?://\S+", re.I)
_URL_SCHEME_RE = re.compile(r"^(?P<scheme>[a-z][a-z0-9+.\-]*):", re.I)
_IPV4_RE = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}")
_IPV6_RE = re.compile(r"[0-9a-f:]+:[0-9a-f:]+", re.I)
_UNICODE_DOMAIN_RE = re.compile(r"[^\x00-\x7f]", re.I)


def _host_entropy(host: str) -> float:
    """Shannon entropy of the host label; high entropy signals machine-generated names."""
    if not host:
        return 0.0
    counts = {}
    for ch in host.lower():
        counts[ch] = counts.get(ch, 0) + 1
    length = len(host)
    entropy = -sum((c / length) * math.log2(c / length) for c in counts.values())
    return entropy


class HeuristicContentAnalyzer(BaseAnalyzer):
    """Scores plain email body text and/or a target URL for phishing indicators."""

    name: str = ANALYZER_NAME
    version: str = ANALYZER_VERSION
    supported_scan_types: Set[ScanType] = {ScanType.COMPREHENSIVE}
    required_inputs: Set[str] = set()
    optional_inputs: Set[str] = {"email", "target_url"}
    resource_class: AnalyzerResourceClass = AnalyzerResourceClass.ASYNC_IO
    priority: int = 30
    can_hard_override: bool = False

    async def analyze(
        self, context: ScanContext, deadline_monotonic: float
    ) -> AnalyzerResult:
        start = time.monotonic()

        findings: List[Finding] = []
        email_text = self._resolve_email_text(context)
        target_url = getattr(context, "target_url", None) or ""

        if time.monotonic() >= deadline_monotonic:
            return self._timeout(start)

        score = 0.0
        if email_text:
            score += self._score_email(email_text[:MAX_TEXT_CHARS], findings)
        if target_url:
            score += self._score_url(target_url[:MAX_TEXT_CHARS], findings)

        findings = findings[:MAX_FINDINGS]
        contribution = min(max(score, 0.0), 1.0) * SCORE_CAP
        confidence = self._confidence(contribution, bool(email_text), bool(target_url))

        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.SUCCESS,
            execution_time_ms=(time.monotonic() - start) * 1000.0,
            score_contribution=round(contribution, 2),
            confidence=confidence,
            findings=findings,
        )

    # ------------------------------------------------------------------
    # Email heuristics
    # ------------------------------------------------------------------

    def _score_email(self, text: str, findings: List[Finding]) -> float:
        norm = text.lower()
        score = 0.0

        score += self._lexicon(norm, _URGENCY_TERMS, "email.urgency", 0.15, findings)
        score += self._lexicon(norm, _CREDENTIAL_BAIT_TERMS, "email.credential_bait", 0.30, findings)
        score += self._lexicon(norm, _PRESSURE_TERMS, "email.pressure", 0.22, findings)
        score += self._lexicon(norm, _MONEY_TERMS, "email.money_bait", 0.18, findings)

        if _LINK_PRESSURE_RE.search(norm):
            score += 0.12
            findings.append(
                self._finding(
                    "email.link_pressure",
                    "message pressures the recipient to click a link",
                    SecurityVerdict.HIGH,
                    0.7,
                )
            )

        link_count = len(_LINK_TAG_RE.findall(norm))
        if link_count >= 2:
            score += 0.06
            findings.append(
                self._finding(
                    "email.multiple_links",
                    f"message embeds {link_count} links",
                    SecurityVerdict.LOW,
                    0.5,
                )
            )

        return score

    @staticmethod
    def _lexicon(
        norm: str,
        terms: Tuple[str, ...],
        rule_id: str,
        weight: float,
        findings: List[Finding],
    ) -> float:
        hits = [t for t in terms if t in norm]
        if not hits:
            return 0.0
        findings.append(
            Finding(
                rule_id=rule_id,
                category="social_engineering",
                severity=SecurityVerdict.MEDIUM,
                confidence=0.75,
                message=f"matched indicators: {', '.join(hits[:3])}",
                metadata=ConstrainedMetadata(),
            )
        )
        return weight

    # ------------------------------------------------------------------
    # URL heuristics
    # ------------------------------------------------------------------

    def _score_url(self, raw: str, findings: List[Finding]) -> float:
        raw = raw.strip()
        if not raw:
            return 0.0
        score = 0.0

        # 1. Non-HTTP scheme is inherently risky in a scan context.
        m = _URL_SCHEME_RE.match(raw)
        scheme = m.group("scheme").lower() if m else ""
        if scheme and scheme not in ("http", "https"):
            score += 0.35
            findings.append(
                self._finding("url.non_http_scheme", f"suspicious scheme '{scheme}:'", SecurityVerdict.HIGH, 0.9)
            )

        # 2. Strip scheme and userinfo to isolate the host.
        authority = raw
        if "://" in authority:
            authority = authority.split("://", 1)[1]
        host = authority.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]

        # 3. Userinfo / '@' before the host.
        if "@" in host:
            score += 0.25
            findings.append(
                self._finding("url.userinfo_at", "host prefixed with '@' userinfo", SecurityVerdict.HIGH, 0.85)
            )
            host = host.rsplit("@", 1)[-1]

        # 4. Strip port for host checks, but flag nonstandard ports.
        host_bare = host
        port: Optional[int] = None
        if ":" in host and host.count(":") == 1:
            head, maybe_port = host.split(":", 1)
            if maybe_port.isdigit():
                host_bare = head
                port = int(maybe_port)
        if port is not None and port not in (80, 443):
            score += 0.10
            findings.append(
                self._finding("url.nonstandard_port", f"URL uses nonstandard port {port}", SecurityVerdict.MEDIUM, 0.6)
            )

        # 5. IP-literal host (direct-to-IP phishing).
        if _IPV4_RE.match(host_bare) or _IPV6_RE.match(host_bare):
            score += 0.30
            findings.append(
                self._finding("url.ip_literal_host", "host is a raw IP address", SecurityVerdict.HIGH, 0.9)
            )
        else:
            # 6. Entropy-based random host detection.
            if _host_entropy(host_bare) > 2.6:
                score += 0.15
                findings.append(
                    self._finding("url.high_entropy_host", "host name has high randomness", SecurityVerdict.MEDIUM, 0.6)
                )

            # 7. Suspicious TLD.
            tld_hit = next((t for t in _BAD_TLDS if host_bare.lower().endswith(t)), None)
            if tld_hit:
                score += 0.25
                findings.append(
                    self._finding("url.bad_tld", f"host uses risky TLD '{tld_hit}'", SecurityVerdict.HIGH, 0.85)
                )

            # 8. Known URL shortener.
            if any(s in host_bare.lower() for s in _SHORTENERS):
                score += 0.20
                findings.append(
                    self._finding("url.shortener", "host is a known URL shortener", SecurityVerdict.MEDIUM, 0.7)
                )

            # 9. Phishing keywords in the domain.
            kw_hits = [k for k in _PHISH_KEYWORDS if k in host_bare.lower()]
            if kw_hits:
                add = min(0.30, 0.15 * len(kw_hits))
                score += add
                findings.append(
                    self._finding(
                        "url.phishing_keywords",
                        f"domain contains trust keywords: {', '.join(kw_hits[:4])}",
                        SecurityVerdict.MEDIUM,
                        0.7,
                    )
                )

            # 10. Excessive subdomains / overlong host.
            subdomains = host_bare.split(".")[:-1]
            if len(subdomains) > 3:
                score += 0.12
                findings.append(
                    self._finding("url.many_subdomains", f"host has {len(subdomains)} subdomain labels", SecurityVerdict.MEDIUM, 0.6)
                )
            if len(host_bare) > 60:
                score += 0.08
                findings.append(
                    self._finding("url.long_host", "host name is unusually long", SecurityVerdict.LOW, 0.5)
                )

            # 11. Unicode / punycode homograph.
            if _UNICODE_DOMAIN_RE.search(host_bare):
                score += 0.10
                findings.append(
                    self._finding("url.unicode_host", "host contains non-ASCII (possible homograph)", SecurityVerdict.MEDIUM, 0.6)
                )

        return score

    # ------------------------------------------------------------------
    # Content resolution & helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_email_text(context: ScanContext) -> str:
        metadata = context.metadata.data or {}
        text = metadata.get("email_text")
        if text:
            return str(text)
        if getattr(context, "email_payload", None):
            try:
                from storage.object_storage import object_storage

                key = getattr(context.email_payload, "storage_key", None)
                if key:
                    raw = object_storage.get_bytes(key)
                    return raw.decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                return ""
        return ""

    @staticmethod
    def _finding(
        rule_id: str, message: str, severity: SecurityVerdict, confidence: float
    ) -> Finding:
        return Finding(
            rule_id=rule_id,
            category="heuristic",
            severity=severity,
            confidence=confidence,
            message=message,
            metadata=ConstrainedMetadata(),
        )

    @staticmethod
    def _confidence(contribution: float, has_email: bool, has_url: bool) -> float:
        base = CONF_DEFAULT
        if has_email and has_url:
            base = min(0.95, base + 0.10)
        # Very low signal -> lower confidence; strong signal -> higher confidence.
        base = min(0.98, max(0.55, base * (0.55 + 0.45 * min(contribution / SCORE_CAP, 1.0))))
        return round(base, 3)

    def _timeout(self, start: float) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.TIMEOUT,
            execution_time_ms=(time.monotonic() - start) * 1000.0,
            sanitized_error="Deadline exceeded during heuristic analysis.",
        )


heuristic_analyzer = HeuristicContentAnalyzer()


__all__ = ["HeuristicContentAnalyzer", "heuristic_analyzer"]

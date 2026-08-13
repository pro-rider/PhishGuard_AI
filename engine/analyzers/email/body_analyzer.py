"""
PhishGuard_AI - Email Body Security Analyzer
Production-grade sub-analyzer inspecting email body content (Plain Text and HTML)
for credential harvesting indicators, hidden obfuscated text, link-text discrepancies,
coercive urgency patterns, embedded form/password inputs, and brand mismatch.

Key Security Controls:
----------------------
1. Safe Email Payload Store Resolution from PayloadReference buffers.
2. Hardened event-driven HTML parser with depth, tag-count, and deadline bounds.
3. Multi-pattern CSS + Unicode obfuscation detection.
4. Comprehensive link-text deception + dangerous URI scheme inspection.
5. Public-suffix-aware domain extraction.
6. Calibrated per-category scoring with asymptotic damping.
"""

from __future__ import annotations

import html
import html.parser
import idna
import logging
import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Tuple

from engine.orchestrator import (
    AnalyzerExecutionStatus,
    AnalyzerResourceClass,
    AnalyzerResult,
    BaseAnalyzer,
    ConstrainedMetadata,
    Finding,
    PayloadReference,
    ScanContext,
    ScanType,
    SecurityVerdict,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & BOUNDS
# ============================================================================

MAX_BODY_SIZE_BYTES = 1_048_576          # 1 MB
MAX_EXTRACTED_URLS = 60
MAX_HTML_TREE_DEPTH = 60
MAX_HTML_TAGS = 6_000
MAX_HIDDEN_SNIPPETS = 25
MAX_FINDINGS_PER_RULE = 5

ZERO_WIDTH_CHARS_REGEX = re.compile(
    r"[\u200B-\u200D\uFEFF\u200E\u200F\u00AD\u2060\u180E]"
)
RTL_OVERRIDE_REGEX = re.compile(r"[\u202A-\u202E\u2066-\u2069]")

DEFANGED_URL_REGEX = re.compile(
    r"(?i)\b(?:hxxps?|h__ps?|hxxp|f_p|fxp|htxps?)://"
    r"[\w\.-]+(?:\.[\w\.-]+)+(?:[\w\-\._~:/?#[\]@!$&'()*+,;=%]*)?"
)

URL_EXTRACTION_REGEX = re.compile(
    r"(?i)\b(?:https?|ftp)://"
    r"(?:[-\w.@]+)(?::\d{1,5})?"
    r"(?:/[-\w._~:/?#[\]@!$&'()*+,;=%]*)?"
)

DOMAIN_LIKE_REGEX = re.compile(
    r"(?i)\b(?:https?://)?(?:www\.)?([a-z0-9][-a-z0-9]*\.)+[a-z]{2,}\b"
)

DANGEROUS_SCHEME_REGEX = re.compile(
    r"(?i)^\s*(?:javascript|data|vbscript|file|blob|about):"
)

MULTI_PART_TLDS: Set[str] = {
    "co.uk", "org.uk", "gov.uk", "me.uk", "ac.uk",
    "com.au", "net.au", "org.au", "edu.au",
    "co.jp", "ne.jp", "or.jp", "go.jp",
    "co.nz", "org.nz", "govt.nz",
    "com.br", "com.mx", "co.in", "gov.in",
    "github.io", "blogspot.com", "herokuapp.com",
}

BRAND_REGISTRY: Dict[str, Set[str]] = {
    "microsoft": {
        "microsoft.com", "office.com", "office365.com", "live.com",
        "outlook.com", "azure.com", "microsoftonline.com", "msn.com",
    },
    "google": {
        "google.com", "gmail.com", "accounts.google.com",
        "googlemail.com", "googleworkspace.com", "withgoogle.com",
    },
    "paypal": {"paypal.com", "paypal-objects.com", "paypalobjects.com"},
    "docusign": {"docusign.com", "docusign.net"},
    "apple": {"apple.com", "icloud.com", "me.com", "mac.com"},
    "amazon": {"amazon.com", "aws.amazon.com", "amazonaws.com"},
    "netflix": {"netflix.com"},
    "facebook": {"facebook.com", "fb.com", "meta.com"},
    "linkedin": {"linkedin.com"},
    "dropbox": {"dropbox.com"},
}

URGENCY_PATTERNS = [
    re.compile(r"(?i)\b(?:account\s+(?:suspended|terminated|locked|restricted|disabled))\b"),
    re.compile(r"(?i)\b(?:verify\s+your\s+(?:identity|account|credentials|billing|payment))\b"),
    re.compile(r"(?i)\b(?:immediate\s+action\s+required|unauthorized\s+(?:login|access)\s+attempt)\b"),
    re.compile(r"(?i)\b(?:update\s+(?:your\s+)?(?:payment|billing)\s+method|wire\s+transfer|gift\s+card)\b"),
    re.compile(r"(?i)\b(?:password\s+(?:expires?|expiring)\s+today|security\s+alert|unusual\s+activity)\b"),
    re.compile(r"(?i)\b(?:click\s+(?:here|below)\s+to\s+(?:verify|confirm|secure|unlock))\b"),
]

VOID_TAGS: Set[str] = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}

# ============================================================================
# DOMAIN EXTRACTOR
# ============================================================================

class PublicSuffixDomainExtractor:
    """Extract organizational (registered) domain with multi-part TLD awareness."""

    @classmethod
    def extract_org_domain(cls, url_or_domain: str) -> str:
        if not url_or_domain:
            return ""

        candidate = url_or_domain.strip().lower()

        # Strip credentials / port
        if "://" in candidate:
            try:
                parsed = urllib.parse.urlparse(candidate)
                candidate = (parsed.netloc or parsed.path.split("/")[0]).split("@")[-1]
            except Exception:
                return ""
        candidate = candidate.split(":")[0].split("/")[0]

        if not candidate or "." not in candidate:
            return candidate

        try:
            candidate = idna.encode(candidate).decode("ascii")
        except Exception:
            pass

        parts = candidate.split(".")
        if len(parts) <= 2:
            return candidate

        two = ".".join(parts[-2:])
        if two in MULTI_PART_TLDS and len(parts) >= 3:
            return ".".join(parts[-3:])
        return two

# ============================================================================
# CSS OBFUSCATION DETECTOR
# ============================================================================

class CSSObfuscationDetector:
    """Detects common inline CSS hiding techniques."""

    HIDDEN_PATTERNS = [
        re.compile(r"display\s*:\s*none", re.I),
        re.compile(r"visibility\s*:\s*hidden", re.I),
        re.compile(r"font-size\s*:\s*0(?:px|em|rem|%)?", re.I),
        re.compile(r"opacity\s*:\s*0(?:\.0+)?\b", re.I),
        re.compile(r"color\s*:\s*(?:transparent|rgba?\([^)]*,\s*0(?:\.\d+)?\s*\))", re.I),
        re.compile(
            r"color\s*:\s*(?:#fff(?:fff)?|white)\s*;\s*background(?:-color)?\s*:\s*(?:#fff(?:fff)?|white)",
            re.I,
        ),
        re.compile(r"position\s*:\s*absolute\s*;\s*(?:left|top)\s*:\s*-\d{3,}px", re.I),
        re.compile(r"clip(?:-path)?\s*:\s*(?:rect\s*\(\s*0|inset\s*\(\s*100%)", re.I),
        re.compile(r"mso-hide\s*:\s*all", re.I),
    ]

    @classmethod
    def is_style_hidden(cls, style_attr: str) -> bool:
        if not style_attr:
            return False
        return any(p.search(style_attr) for p in cls.HIDDEN_PATTERNS)

# ============================================================================
# HARDENED HTML PARSER
# ============================================================================

class HardenedHTMLParser(html.parser.HTMLParser):
    """Resource-bounded, deadline-aware HTML parser."""

    def __init__(self, deadline_monotonic: float) -> None:
        super().__init__(convert_charrefs=True)
        self.deadline_monotonic = deadline_monotonic
        self.tag_count = 0
        self.tag_stack: List[str] = []
        self.links: List[Tuple[str, str]] = []
        self.dangerous_uris: List[Tuple[str, str]] = []
        self.has_form = False
        self.has_password_input = False
        self.hidden_text_snippets: List[str] = []
        self.raw_text_chunks: List[str] = []
        self._current_href: Optional[str] = None
        self._current_anchor_parts: List[str] = []
        self._hidden_depth = 0
        self._aborted = False

    def _check_limits(self) -> None:
        if self._aborted:
            raise ValueError("Parser already aborted")
        if time.monotonic() >= self.deadline_monotonic:
            self._aborted = True
            raise TimeoutError("Deadline exceeded during HTML parsing")
        if self.tag_count > MAX_HTML_TAGS:
            self._aborted = True
            raise ValueError("HTML tag count limit exceeded")
        if len(self.tag_stack) > MAX_HTML_TREE_DEPTH:
            self._aborted = True
            raise ValueError("HTML nesting depth limit exceeded")

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self._check_limits()
        self.tag_count += 1

        tag_l = tag.lower()
        attr = {k.lower(): (v or "") for k, v in attrs}

        if tag_l not in VOID_TAGS:
            self.tag_stack.append(tag_l)

        style = attr.get("style", "")
        if CSSObfuscationDetector.is_style_hidden(style):
            self._hidden_depth += 1

        if tag_l == "form":
            self.has_form = True
        if tag_l == "input" and attr.get("type", "").lower() == "password":
            self.has_password_input = True

        if tag_l == "a":
            href = attr.get("href", "").strip()
            self._current_href = href
            self._current_anchor_parts = []
            if href and DANGEROUS_SCHEME_REGEX.match(href):
                scheme = href.split(":", 1)[0].lower()
                self.dangerous_uris.append((scheme, href[:96]))

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()

        # Pop matching open tag (tolerant of mismatch)
        if self.tag_stack and self.tag_stack[-1] == tag_l:
            self.tag_stack.pop()
        elif tag_l in self.tag_stack:
            # best-effort recovery
            while self.tag_stack and self.tag_stack[-1] != tag_l:
                self.tag_stack.pop()
            if self.tag_stack:
                self.tag_stack.pop()

        if tag_l == "a" and self._current_href is not None:
            anchor = html.unescape("".join(self._current_anchor_parts)).strip()
            self.links.append((self._current_href, anchor))
            self._current_href = None
            self._current_anchor_parts = []

        if self._hidden_depth > 0:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._aborted or not data:
            return
        cleaned = data.strip()
        if not cleaned:
            return

        self.raw_text_chunks.append(cleaned)

        if self._current_href is not None:
            self._current_anchor_parts.append(cleaned)

        if self._hidden_depth > 0 and len(self.hidden_text_snippets) < MAX_HIDDEN_SNIPPETS:
            self.hidden_text_snippets.append(cleaned[:120])

# ============================================================================
# PAYLOAD RESOLUTION
# ============================================================================

class EmailPayloadStore:
    """Safely resolve plain-text and HTML bodies."""

    @staticmethod
    def resolve_bodies(context: ScanContext) -> Tuple[str, str]:
        plain = ""
        html_body = ""

        meta = getattr(context, "metadata", None)
        data: Dict[str, Any] = getattr(meta, "data", {}) if meta else {}

        # 1. Direct string keys (most common orchestrator path)
        plain = (
            data.get("body_plain")
            or data.get("text_plain")
            or data.get("plain")
            or ""
        )
        html_body = (
            data.get("body_html")
            or data.get("text_html")
            or data.get("html")
            or ""
        )

        # 2. PayloadReference objects
        for key, target in (
            ("body_plain_ref", "plain"),
            ("plain_ref", "plain"),
            ("body_html_ref", "html"),
            ("html_ref", "html"),
        ):
            ref = data.get(key)
            if not isinstance(ref, PayloadReference):
                continue
            try:
                raw = None
                if hasattr(ref, "read"):
                    raw = ref.read(max_bytes=MAX_BODY_SIZE_BYTES)  # type: ignore[call-arg]
                elif hasattr(ref, "content"):
                    raw = ref.content
                if raw is None:
                    continue
                text = (
                    raw.decode("utf-8", errors="replace")
                    if isinstance(raw, (bytes, bytearray))
                    else str(raw)
                )
                if target == "plain":
                    plain = text
                else:
                    html_body = text
            except Exception as exc:
                logger.debug("Failed reading PayloadReference %s: %s", key, type(exc).__name__)

        # 3. Optional top-level email_payload (guarded)
        email_payload = getattr(context, "email_payload", None)
        if email_payload is not None and not (plain or html_body):
            storage_key = getattr(email_payload, "storage_key", None)
            if storage_key:
                logger.debug("Body storage_key present but no direct content resolved: %s", storage_key)

        return str(plain or ""), str(html_body or "")

# ============================================================================
# MAIN ANALYZER
# ============================================================================

class BodyAnalyzer(BaseAnalyzer):
    """Production email-body phishing / obfuscation analyzer."""

    name: str = "body_analyzer"
    version: str = "2.2.0"
    supported_scan_types: Set[ScanType] = {ScanType.EMAIL, ScanType.COMPREHENSIVE}
    required_inputs: Set[str] = {"email"}
    resource_class: AnalyzerResourceClass = AnalyzerResourceClass.CPU_BOUND
    priority: int = 20
    can_hard_override: bool = False

    async def analyze(self, context: ScanContext, deadline_monotonic: float) -> AnalyzerResult:
        start_mono = time.monotonic()

        if time.monotonic() >= deadline_monotonic:
            return self._timeout(start_mono)

        plain_text, html_text = EmailPayloadStore.resolve_bodies(context)

        total_bytes = (
            len(plain_text.encode("utf-8", errors="ignore"))
            + len(html_text.encode("utf-8", errors="ignore"))
        )

        if total_bytes == 0:
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start_mono) * 1000.0,
                score_contribution=0.0,
                confidence=0.75,
                findings=[],
            )

        if total_bytes > MAX_BODY_SIZE_BYTES:
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.INVALID_INPUT,
                execution_time_ms=(time.monotonic() - start_mono) * 1000.0,
                sanitized_error=(
                    f"Email body size ({total_bytes} B) exceeds limit "
                    f"({MAX_BODY_SIZE_BYTES} B)."
                ),
            )

        findings: List[Finding] = []
        cat_scores: Dict[str, float] = {
            "OBFUSCATION": 0.0,
            "CREDENTIAL_HARVESTING": 0.0,
            "DECEPTION": 0.0,
            "IMPERSONATION": 0.0,
            "SOCIAL_ENGINEERING": 0.0,
        }
        rule_counts: Dict[str, int] = {}

        def emit(
            rule_id: str,
            category: str,
            severity: SecurityVerdict,
            confidence: float,
            message: str,
            points: float,
            meta: Optional[Dict[str, Any]] = None,
        ) -> None:
            cnt = rule_counts.get(rule_id, 0)
            if cnt >= MAX_FINDINGS_PER_RULE:
                return
            rule_counts[rule_id] = cnt + 1
            # asymptotic damping
            effective = points * (0.65 ** cnt)
            cat_scores[category] = cat_scores.get(category, 0.0) + effective
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
            combined = f"{plain_text}\n{html_text}"

            # ----- Unicode obfuscation -----
            zw = ZERO_WIDTH_CHARS_REGEX.findall(combined)
            rtl = RTL_OVERRIDE_REGEX.findall(combined)
            if len(zw) > 12 or len(rtl) > 0:
                emit(
                    "EMAIL.BODY.UNICODE_OBFUSCATION",
                    "OBFUSCATION",
                    SecurityVerdict.MEDIUM,
                    0.90,
                    f"Zero-width chars={len(zw)}, RTL overrides={len(rtl)}.",
                    22.0,
                    {"zero_width_count": len(zw), "rtl_count": len(rtl)},
                )

            if time.monotonic() >= deadline_monotonic:
                return self._timeout(start_mono)

            # ----- HTML parse -----
            parser = HardenedHTMLParser(deadline_monotonic)
            if html_text:
                try:
                    # hard truncate before feed
                    parser.feed(html_text[:MAX_BODY_SIZE_BYTES])
                    parser.close()
                except (ValueError, TimeoutError) as err:
                    logger.warning(
                        "HTML parser aborted scan_id=%s: %s",
                        getattr(context, "scan_id", "?"),
                        err,
                    )
                except Exception as err:
                    logger.warning(
                        "Unexpected HTML parse error scan_id=%s: %s",
                        getattr(context, "scan_id", "?"),
                        type(err).__name__,
                    )

            # ----- Credential harvest signals -----
            if parser.has_password_input:
                emit(
                    "EMAIL.BODY.EMBEDDED_PASSWORD_INPUT",
                    "CREDENTIAL_HARVESTING",
                    SecurityVerdict.HIGH,
                    0.97,
                    "Embedded password <input> detected.",
                    42.0,
                    {"has_form": parser.has_form},
                )
            elif parser.has_form:
                emit(
                    "EMAIL.BODY.EMBEDDED_FORM",
                    "CREDENTIAL_HARVESTING",
                    SecurityVerdict.MEDIUM,
                    0.82,
                    "Embedded <form> detected (no password field).",
                    14.0,
                    {},
                )

            for scheme, snippet in parser.dangerous_uris[:MAX_FINDINGS_PER_RULE]:
                emit(
                    "EMAIL.BODY.DANGEROUS_URI_SCHEME",
                    "CREDENTIAL_HARVESTING",
                    SecurityVerdict.HIGH,
                    0.95,
                    f"Dangerous URI scheme '{scheme}' in hyperlink.",
                    36.0,
                    {"scheme": scheme, "snippet": snippet[:64]},
                )

            if parser.hidden_text_snippets:
                emit(
                    "EMAIL.BODY.HIDDEN_CSS_TEXT",
                    "OBFUSCATION",
                    SecurityVerdict.MEDIUM,
                    0.88,
                    f"{len(parser.hidden_text_snippets)} hidden text block(s) via CSS.",
                    20.0,
                    {"hidden_snippet_count": len(parser.hidden_text_snippets)},
                )

            if time.monotonic() >= deadline_monotonic:
                return self._timeout(start_mono)

            # ----- Defanged URLs -----
            defanged = DEFANGED_URL_REGEX.findall(combined)
            if defanged:
                emit(
                    "EMAIL.BODY.DEFANGED_URL_DETECTED",
                    "OBFUSCATION",
                    SecurityVerdict.LOW,
                    0.85,
                    f"{len(defanged)} defanged URL pattern(s) found.",
                    10.0,
                    {"count": len(defanged), "samples": defanged[:3]},
                )

            # ----- Links + mismatch -----
            all_links: List[Tuple[str, str]] = list(parser.links)
            for u in URL_EXTRACTION_REGEX.findall(plain_text)[:MAX_EXTRACTED_URLS]:
                all_links.append((u, u))

            extracted_domains: Set[str] = set()

            for href, anchor in all_links[:MAX_EXTRACTED_URLS]:
                if not href:
                    continue
                if DANGEROUS_SCHEME_REGEX.match(href):
                    continue  # already handled

                href_dom = PublicSuffixDomainExtractor.extract_org_domain(href)
                if href_dom:
                    extracted_domains.add(href_dom)

                # Improved mismatch: domain-like anchor text
                anchor_norm = html.unescape(anchor or "").strip()
                if len(anchor_norm) < 4:
                    continue

                m = DOMAIN_LIKE_REGEX.search(anchor_norm)
                if m:
                    anchor_dom = PublicSuffixDomainExtractor.extract_org_domain(
                        m.group(0) if "://" in m.group(0) else f"https://{m.group(0)}"
                    )
                    if (
                        anchor_dom
                        and href_dom
                        and anchor_dom != href_dom
                        and not href_dom.endswith("." + anchor_dom)
                        and not anchor_dom.endswith("." + href_dom)
                    ):
                        emit(
                            "EMAIL.BODY.LINK_TEXT_MISMATCH",
                            "DECEPTION",
                            SecurityVerdict.HIGH,
                            0.93,
                            f"Anchor suggests '{anchor_dom}' but targets '{href_dom}'.",
                            32.0,
                            {
                                "anchor_domain": anchor_dom,
                                "target_domain": href_dom,
                                "anchor_sample": anchor_norm[:80],
                            },
                        )

            if time.monotonic() >= deadline_monotonic:
                return self._timeout(start_mono)

            # ----- Brand vs destination -----
            prose = " ".join(parser.raw_text_chunks) if parser.raw_text_chunks else plain_text
            prose_l = prose.lower()

            for brand, legit in BRAND_REGISTRY.items():
                if not re.search(rf"\b{re.escape(brand)}\b", prose_l):
                    continue
                if not extracted_domains:
                    continue
                has_legit = any(
                    any(d == ld or d.endswith("." + ld) for ld in legit)
                    for d in extracted_domains
                )
                if not has_legit:
                    emit(
                        "EMAIL.BODY.BRAND_LINK_MISMATCH",
                        "IMPERSONATION",
                        SecurityVerdict.HIGH,
                        0.86,
                        f"Mentions '{brand.title()}' but no links to known legitimate domains.",
                        26.0,
                        {
                            "brand": brand,
                            "observed_domains": sorted(extracted_domains)[:10],
                        },
                    )

            # ----- Urgency / social engineering -----
            urgency_hits = sum(1 for p in URGENCY_PATTERNS if p.search(prose_l))
            if urgency_hits >= 2:
                emit(
                    "EMAIL.BODY.COERCIVE_URGENCY",
                    "SOCIAL_ENGINEERING",
                    SecurityVerdict.MEDIUM,
                    0.83,
                    f"{urgency_hits} coercive urgency phrases detected.",
                    14.0,
                    {"urgency_count": urgency_hits},
                )

            # ----- Final calibrated score -----
            total = (
                min(cat_scores["OBFUSCATION"], 35.0)
                + min(cat_scores["CREDENTIAL_HARVESTING"], 45.0)
                + min(cat_scores["DECEPTION"], 40.0)
                + min(cat_scores["IMPERSONATION"], 30.0)
                + min(cat_scores["SOCIAL_ENGINEERING"], 20.0)
            )
            score = min(max(total, 0.0), 100.0)
            exec_ms = (time.monotonic() - start_mono) * 1000.0

            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=exec_ms,
                score_contribution=score,
                confidence=0.90 if findings else 0.78,
                findings=findings,
            )

        except Exception as exc:
            logger.exception(
                "BodyAnalyzer failure scan_id=%s", getattr(context, "scan_id", "?")
            )
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.FAILED,
                execution_time_ms=(time.monotonic() - start_mono) * 1000.0,
                sanitized_error=f"Body analysis error: {type(exc).__name__}",
            )

    def _timeout(self, start_mono: float) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.TIMEOUT,
            execution_time_ms=(time.monotonic() - start_mono) * 1000.0,
            sanitized_error="Deadline exceeded during email body processing.",
        )

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "BodyAnalyzer",
    "CSSObfuscationDetector",
    "EmailPayloadStore",
    "HardenedHTMLParser",
    "PublicSuffixDomainExtractor",
]
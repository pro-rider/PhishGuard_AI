"""
PhishGuard_AI - Advanced DOM Security & Phishing Structural Inspector
Production DOM analyzer for forms, credential harvest patterns, hidden overlays,
brand impersonation, script obfuscation, and evasive structure.

Key controls
------------
1. Form + formaction/formmethod + formless password detection.
2. Origin / IP / scheme checks on form actions and base href.
3. Hidden iframe / overlay heuristics.
4. Brand claim vs registrable domain correlation (with password context).
5. Obfuscated JS & sensitive DOM API signatures.
6. Meta-refresh / base-hijack / dangerous URI schemes.
7. Bounds, deadline checks, per-category scoring.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag

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

# Prefer lxml when available
try:
    BeautifulSoup("<p></p>", "lxml")
    _BS_PARSER = "lxml"
except Exception:
    _BS_PARSER = "html.parser"

# ============================================================================
# CONSTANTS
# ============================================================================

DOM_ANALYZER_VERSION = "6.2.0"
RULESET_VERSION = "2026.08.13"

MAX_DOM_BYTES = 3 * 1024 * 1024
MAX_FORMS = 40
MAX_SCRIPTS = 80
MAX_LINKS = 300
MAX_ATTR_LEN = 800
MAX_FINDINGS = 30
MAX_TEXT_SAMPLE = 12_000

TARGET_BRANDS: Dict[str, Set[str]] = {
    "microsoft": {"microsoft.com", "office.com", "live.com", "outlook.com", "azure.com", "microsoftonline.com"},
    "google": {"google.com", "gmail.com", "youtube.com", "googlemail.com"},
    "apple": {"apple.com", "icloud.com", "me.com"},
    "paypal": {"paypal.com", "paypalobjects.com"},
    "amazon": {"amazon.com", "amazonaws.com"},
    "netflix": {"netflix.com"},
    "chase": {"chase.com"},
    "wellsfargo": {"wellsfargo.com"},
    "binance": {"binance.com"},
    "coinbase": {"coinbase.com"},
    "docusign": {"docusign.com", "docusign.net"},
    "dropbox": {"dropbox.com"},
    "adobe": {"adobe.com"},
    "facebook": {"facebook.com", "fb.com", "meta.com"},
    "linkedin": {"linkedin.com"},
    "okta": {"okta.com"},
}

USERNAME_NAME_HINTS = re.compile(
    r"(user|login|email|e-mail|account|identifier|id$|phone|mobile)", re.I
)
PASSWORD_NAME_HINTS = re.compile(r"(pass|pwd|secret|pin)", re.I)

RE_PACKED_JS = re.compile(r"eval\s*\(\s*function\s*\(", re.I)
RE_UNESCAPE_WRITE = re.compile(r"document\s*\.\s*write\s*\(\s*unescape\s*\(", re.I)
RE_FROM_CHAR_CODE = re.compile(r"String\s*\.\s*fromCharCode\s*\(\s*(?:\d+\s*,\s*){4,}", re.I)
RE_HEX_ENCODED = re.compile(r"(?:\\x[0-9a-fA-F]{2}){10,}")
RE_BASE64_EXEC = re.compile(r"atob\s*\(\s*['\"][A-Za-z0-9+/]{32,}", re.I)
RE_FUNCTION_CONSTRUCT = re.compile(r"new\s+Function\s*\(", re.I)
RE_SETTIMEOUT_STR = re.compile(r"set(?:Timeout|Interval)\s*\(\s*['\"]", re.I)
RE_COOKIE_ACCESS = re.compile(r"document\s*\.\s*cookie", re.I)
RE_STORAGE_ACCESS = re.compile(r"(?:localStorage|sessionStorage)\s*\.", re.I)
RE_LOCATION_NAV = re.compile(
    r"(?:window\.|document\.)?location(?:\.href|\.replace|\.assign)?\s*=", re.I
)
RE_META_REFRESH_URL = re.compile(r"url\s*=\s*['\"]?([^'\"\s;]+)", re.I)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass(slots=True)
class FormInspectionResult:
    action_url: str
    method: str
    has_password_field: bool = False
    has_username_field: bool = False
    has_get_password_hazard: bool = False
    is_cross_origin: bool = False
    is_unencrypted_http: bool = False
    is_raw_ip_action: bool = False
    is_dangerous_scheme: bool = False
    overridden_by_control: bool = False
    hidden_inputs_count: int = 0


@dataclass(slots=True)
class DOMStructureAnalysis:
    title: str = ""
    form_count: int = 0
    password_input_count: int = 0
    formless_password_count: int = 0
    hidden_iframe_count: int = 0
    external_script_count: int = 0
    inline_script_count: int = 0
    obfuscated_script_matches: List[str] = field(default_factory=list)
    detected_brand_claims: List[str] = field(default_factory=list)
    forms: List[FormInspectionResult] = field(default_factory=list)
    has_meta_refresh: bool = False
    meta_refresh_target: Optional[str] = None
    has_base_href_hijack: bool = False
    dangerous_uri_links: int = 0
    inline_handler_count: int = 0


# ============================================================================
# ORIGIN UTILS
# ============================================================================

class OriginSecurityUtils:
    @classmethod
    def normalize_host(cls, host: str) -> str:
        if not host:
            return ""
        clean = host.lower().rstrip(".")
        if idna is not None:
            try:
                return idna.decode(clean)
            except Exception:
                pass
        return clean

    @classmethod
    def parse_origin(cls, url: str) -> Tuple[str, str, int]:
        try:
            p = urllib.parse.urlparse(url)
            scheme = (p.scheme or "http").lower()
            host = cls.normalize_host(p.hostname or "")
            port = p.port or (443 if scheme in {"https", "wss"} else 80)
            return scheme, host, port
        except Exception:
            return "http", "", 80

    @classmethod
    def is_cross_origin(cls, a: str, b: str) -> bool:
        s1, h1, p1 = cls.parse_origin(a)
        s2, h2, p2 = cls.parse_origin(b)
        if not h1 or not h2:
            return False
        return s1 != s2 or h1 != h2 or p1 != p2

    @classmethod
    def is_raw_ip(cls, host: str) -> bool:
        if not host:
            return False
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            return False


# ============================================================================
# FORM ENGINE
# ============================================================================

class FormSecurityEngine:
    @classmethod
    def _is_username_input(cls, inp: Tag) -> bool:
        t = (inp.get("type") or "text").lower()
        if t in {"email", "tel"}:
            return True
        name = (inp.get("name") or "") + " " + (inp.get("id") or "") + " " + (inp.get("autocomplete") or "")
        return bool(USERNAME_NAME_HINTS.search(name))

    @classmethod
    def inspect_form(cls, form: Tag, page_url: str) -> FormInspectionResult:
        method = (form.get("method") or "GET").upper().strip() or "GET"
        raw_action = (form.get("action") or "").strip()

        overridden = None
        for ctrl in form.find_all(["button", "input"]):
            fa = (ctrl.get("formaction") or "").strip()
            if fa:
                overridden = fa
                fm = (ctrl.get("formmethod") or "").upper().strip()
                if fm:
                    method = fm
                break

        effective = overridden if overridden else raw_action
        action_url = urllib.parse.urljoin(page_url, effective) if effective else page_url

        result = FormInspectionResult(
            action_url=action_url,
            method=method,
            overridden_by_control=overridden is not None,
        )

        scheme = (urllib.parse.urlparse(action_url).scheme or "").lower()
        if scheme in {"javascript", "data", "blob", "vbscript"}:
            result.is_dangerous_scheme = True

        for inp in form.find_all("input"):
            t = (inp.get("type") or "text").lower()
            if t == "password" or PASSWORD_NAME_HINTS.search(
                (inp.get("name") or "") + (inp.get("id") or "")
            ):
                result.has_password_field = True
            elif cls._is_username_input(inp):
                result.has_username_field = True
            elif t == "hidden":
                result.hidden_inputs_count += 1

        if result.has_password_field and method == "GET":
            result.has_get_password_hazard = True

        result.is_cross_origin = OriginSecurityUtils.is_cross_origin(page_url, action_url)
        _, action_host, _ = OriginSecurityUtils.parse_origin(action_url)
        result.is_raw_ip_action = OriginSecurityUtils.is_raw_ip(action_host)

        page_scheme, _, _ = OriginSecurityUtils.parse_origin(page_url)
        act_scheme, _, _ = OriginSecurityUtils.parse_origin(action_url)
        if page_scheme == "https" and act_scheme == "http":
            result.is_unencrypted_http = True

        return result

    @classmethod
    def formless_password_count(cls, soup: BeautifulSoup) -> int:
        n = 0
        for pwd in soup.find_all("input", attrs={"type": "password"}):
            if not pwd.find_parent("form"):
                n += 1
        return n


# ============================================================================
# VISUAL / IFRAME
# ============================================================================

class VisualElementInspector:
    HIDDEN_STYLE_MARKERS = (
        "display:none", "display: none",
        "visibility:hidden", "visibility: hidden",
        "opacity:0", "opacity: 0", "opacity:0.0",
        "left:-999", "left: -999", "top:-999",
        "clip:rect(0", "clip-path:inset(100%",
        "transform:scale(0", "height:0", "width:0",
        "pointer-events:none",
    )

    @classmethod
    def _is_hidden(cls, tag: Tag) -> bool:
        if tag.has_attr("hidden"):
            return True
        if (tag.get("aria-hidden") or "").lower() == "true":
            return True
        style = (tag.get("style") or "").lower().replace(" ", "")
        if any(m.replace(" ", "") in style for m in cls.HIDDEN_STYLE_MARKERS):
            return True
        for attr in ("width", "height"):
            v = (tag.get(attr) or "").strip().lower()
            if v in {"0", "0px", "1", "1px"}:
                return True
        return False

    @classmethod
    def hidden_iframe_count(cls, soup: BeautifulSoup) -> int:
        return sum(1 for iframe in soup.find_all("iframe") if cls._is_hidden(iframe))

    @classmethod
    def dangerous_iframe_count(cls, soup: BeautifulSoup) -> int:
        """Iframes without sandbox or with allow-scripts + allow-same-origin."""
        n = 0
        for iframe in soup.find_all("iframe"):
            sandbox = iframe.get("sandbox")
            if sandbox is None:
                # no sandbox attribute on cross-origin-looking iframe
                src = (iframe.get("src") or "").strip()
                if src and not src.startswith(("#", "about:blank")):
                    n += 1
            else:
                tokens = set(sandbox.lower().split())
                if "allow-scripts" in tokens and "allow-same-origin" in tokens:
                    n += 1
        return n


# ============================================================================
# SCRIPT EVASION
# ============================================================================

class ScriptEvasionAnalyzer:
    @classmethod
    def _script_text(cls, script: Tag) -> str:
        parts: List[str] = []
        for c in script.contents:
            if isinstance(c, NavigableString):
                parts.append(str(c))
            elif isinstance(c, Tag) and c.name is None:
                parts.append(c.get_text() or "")
        if not parts:
            return script.get_text() or ""
        return "".join(parts)

    @classmethod
    def analyze(cls, soup: BeautifulSoup) -> Tuple[int, int, List[str]]:
        external = 0
        inline = 0
        matches: Set[str] = set()

        for script in soup.find_all("script")[:MAX_SCRIPTS]:
            src = (script.get("src") or "").strip()
            if src:
                external += 1
                # raw IP script host
                try:
                    host = urllib.parse.urlparse(urllib.parse.urljoin("https://x/", src)).hostname
                    if host and OriginSecurityUtils.is_raw_ip(host):
                        matches.add("SCRIPT_FROM_RAW_IP")
                except Exception:
                    pass
                continue

            inline += 1
            content = cls._script_text(script)
            if not content or len(content) < 8:
                continue
            if RE_PACKED_JS.search(content):
                matches.add("PACKED_JS_EVAL")
            if RE_UNESCAPE_WRITE.search(content):
                matches.add("UNESCAPE_DOCUMENT_WRITE")
            if RE_FROM_CHAR_CODE.search(content):
                matches.add("FROM_CHAR_CODE")
            if RE_HEX_ENCODED.search(content):
                matches.add("HEX_ENCODED_PAYLOAD")
            if RE_BASE64_EXEC.search(content):
                matches.add("BASE64_ATOB")
            if RE_FUNCTION_CONSTRUCT.search(content):
                matches.add("FUNCTION_CONSTRUCTOR")
            if RE_SETTIMEOUT_STR.search(content):
                matches.add("SETTIMEOUT_STRING")
            if RE_COOKIE_ACCESS.search(content):
                matches.add("DOCUMENT_COOKIE")
            if RE_STORAGE_ACCESS.search(content):
                matches.add("WEB_STORAGE")
            if RE_LOCATION_NAV.search(content):
                matches.add("LOCATION_MANIPULATION")

        return external, inline, sorted(matches)


# ============================================================================
# BRAND ENGINE
# ============================================================================

class BrandImpersonationEngine:
    @classmethod
    def detect(cls, soup: BeautifulSoup, page_url: str) -> List[str]:
        _, host, _ = OriginSecurityUtils.parse_origin(page_url)
        if not host:
            return []

        title = ""
        t = soup.find("title")
        if t:
            title = t.get_text(" ", strip=True).lower()

        # Prefer visible text sample over full body dump
        body = soup.body.get_text(" ", strip=True)[:MAX_TEXT_SAMPLE].lower() if soup.body else ""
        blob = f"{title}\n{body}"

        found: List[str] = []
        for brand, domains in TARGET_BRANDS.items():
            # word-ish match to cut pure substring FPs
            if not re.search(rf"\b{re.escape(brand)}\b", blob):
                continue
            official = any(host == d or host.endswith("." + d) for d in domains)
            if not official:
                found.append(brand)
        return found


# ============================================================================
# LINK / HANDLER HELPERS
# ============================================================================

def count_dangerous_uris(soup: BeautifulSoup) -> int:
    n = 0
    for tag in soup.find_all(["a", "area", "form", "iframe", "object", "embed"]):
        for attr in ("href", "src", "action", "data"):
            v = (tag.get(attr) or "").strip().lower()
            if v.startswith(("javascript:", "data:text/html", "vbscript:")):
                n += 1
    return n


def count_inline_handlers(soup: BeautifulSoup) -> int:
    n = 0
    for tag in soup.find_all(True):
        for attr in list(tag.attrs or {}):
            if isinstance(attr, str) and attr.lower().startswith("on"):
                n += 1
                if n >= 50:
                    return n
    return n


# ============================================================================
# PAYLOAD STORE
# ============================================================================

class DOMAnalyzerPayloadStore:
    @staticmethod
    def resolve(context: ScanContext) -> List[Tuple[str, str]]:
        sources: List[Tuple[str, str]] = []
        seen: Set[str] = set()
        meta = getattr(context, "metadata", None)
        data: Dict[str, Any] = getattr(meta, "data", {}) if meta else {}

        def add(url: str, dom: str) -> None:
            if not url or not dom:
                return
            key = f"{url}:{hash(dom[:2048])}"
            if key in seen:
                return
            seen.add(key)
            sources.append((url, dom))

        for item in data.get("rendered_browser_pages") or []:
            if isinstance(item, dict):
                add(
                    str(item.get("final_url") or item.get("url") or ""),
                    str(item.get("rendered_dom") or ""),
                )

        if not sources:
            for item in data.get("fetched_web_pages") or []:
                if isinstance(item, dict):
                    add(
                        str(item.get("final_url") or item.get("url") or ""),
                        str(item.get("body_text") or ""),
                    )

        # Direct HTML buffer
        for key in ("dom_html", "page_html", "html"):
            html = data.get(key)
            url = data.get("url") or data.get("target_url") or "https://unknown.invalid/"
            if isinstance(html, str) and html.strip():
                add(str(url), html)

        return sources


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class DOMAnalyzer(BaseAnalyzer):
    name: str = "dom_analyzer"
    version: str = DOM_ANALYZER_VERSION
    supported_scan_types: Set[ScanType] = {ScanType.URL, ScanType.COMPREHENSIVE}
    required_inputs: Set[str] = set()
    resource_class: AnalyzerResourceClass = AnalyzerResourceClass.COMPUTE
    priority: int = 12
    can_hard_override: bool = False

    async def _analyze_one(
        self, page_url: str, raw_dom: str, deadline: float
    ) -> Tuple[DOMStructureAnalysis, List[Finding]]:
        findings: List[Finding] = []
        analysis = DOMStructureAnalysis()
        if time.monotonic() >= deadline:
            return analysis, findings

        raw = raw_dom
        if len(raw.encode("utf-8", errors="ignore")) > MAX_DOM_BYTES:
            raw = raw.encode("utf-8", errors="ignore")[:MAX_DOM_BYTES].decode(
                "utf-8", errors="ignore"
            )

        try:
            soup = BeautifulSoup(raw, _BS_PARSER)
        except Exception as exc:
            logger.warning("DOM parse failed for %s: %s", page_url, type(exc).__name__)
            return analysis, findings

        t = soup.find("title")
        analysis.title = t.get_text(" ", strip=True) if t else ""

        # Meta refresh
        meta = soup.find("meta", attrs={"http-equiv": re.compile(r"^refresh$", re.I)})
        if meta:
            content = (meta.get("content") or "").strip()
            analysis.has_meta_refresh = True
            m = RE_META_REFRESH_URL.search(content)
            analysis.meta_refresh_target = m.group(1) if m else content
            findings.append(
                Finding(
                    rule_id="DOM.META.REFRESH",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.MEDIUM,
                    confidence=0.88,
                    message=f"Meta-refresh present ({content[:80]}).",
                    metadata=ConstrainedMetadata({
                        "page_url": page_url[:200],
                        "content": content[:200],
                    }),
                )
            )

        # Base hijack
        base = soup.find("base", href=True)
        if base:
            href = (base.get("href") or "").strip()
            if href and OriginSecurityUtils.is_cross_origin(page_url, urllib.parse.urljoin(page_url, href)):
                analysis.has_base_href_hijack = True
                findings.append(
                    Finding(
                        rule_id="DOM.BASE.CROSS_ORIGIN",
                        category="EXPLOIT_ATTEMPT",
                        severity=SecurityVerdict.HIGH,
                        confidence=0.93,
                        message=f"Cross-origin <base href='{href[:120]}'>.",
                        metadata=ConstrainedMetadata({
                            "page_url": page_url[:200],
                            "base_href": href[:200],
                        }),
                    )
                )

        # Forms
        forms = soup.find_all("form")
        analysis.form_count = len(forms)
        for form in forms[:MAX_FORMS]:
            if time.monotonic() >= deadline:
                break
            fr = FormSecurityEngine.inspect_form(form, page_url)
            analysis.forms.append(fr)
            if not fr.has_password_field:
                continue
            analysis.password_input_count += 1

            if fr.has_get_password_hazard:
                findings.append(
                    Finding(
                        rule_id="DOM.FORM.PASSWORD_VIA_GET",
                        category="CREDENTIAL_HARVESTING",
                        severity=SecurityVerdict.HIGH,
                        confidence=0.95,
                        message="Password field submitted via GET.",
                        metadata=ConstrainedMetadata({"page_url": page_url[:200]}),
                    )
                )
            if fr.is_dangerous_scheme:
                findings.append(
                    Finding(
                        rule_id="DOM.FORM.DANGEROUS_ACTION_SCHEME",
                        category="CREDENTIAL_HARVESTING",
                        severity=SecurityVerdict.HIGH,
                        confidence=0.96,
                        message=f"Form action uses dangerous scheme: {fr.action_url[:120]}",
                        metadata=ConstrainedMetadata({
                            "page_url": page_url[:200],
                            "action": fr.action_url[:200],
                        }),
                    )
                )
            if fr.is_cross_origin:
                findings.append(
                    Finding(
                        rule_id="DOM.FORM.CROSS_ORIGIN_PASSWORD",
                        category="CREDENTIAL_HARVESTING",
                        severity=SecurityVerdict.HIGH,
                        confidence=0.97,
                        message=f"Password form posts cross-origin to {fr.action_url[:120]}",
                        metadata=ConstrainedMetadata({
                            "page_url": page_url[:200],
                            "action": fr.action_url[:200],
                        }),
                    )
                )
            if fr.is_raw_ip_action:
                findings.append(
                    Finding(
                        rule_id="DOM.FORM.PASSWORD_TO_RAW_IP",
                        category="SUSPICIOUS_INFRASTRUCTURE",
                        severity=SecurityVerdict.HIGH,
                        confidence=0.94,
                        message=f"Password form posts to raw IP: {fr.action_url[:120]}",
                        metadata=ConstrainedMetadata({
                            "page_url": page_url[:200],
                            "action": fr.action_url[:200],
                        }),
                    )
                )
            if fr.is_unencrypted_http:
                findings.append(
                    Finding(
                        rule_id="DOM.FORM.HTTPS_TO_HTTP_ACTION",
                        category="CREDENTIAL_HARVESTING",
                        severity=SecurityVerdict.HIGH,
                        confidence=0.95,
                        message="HTTPS page posts credentials to HTTP action.",
                        metadata=ConstrainedMetadata({
                            "page_url": page_url[:200],
                            "action": fr.action_url[:200],
                        }),
                    )
                )

        analysis.formless_password_count = FormSecurityEngine.formless_password_count(soup)
        if analysis.formless_password_count:
            findings.append(
                Finding(
                    rule_id="DOM.FORM.FORMLESS_PASSWORD",
                    category="CREDENTIAL_HARVESTING",
                    severity=SecurityVerdict.MEDIUM,
                    confidence=0.88,
                    message=f"{analysis.formless_password_count} password field(s) outside <form>.",
                    metadata=ConstrainedMetadata({
                        "page_url": page_url[:200],
                        "count": analysis.formless_password_count,
                    }),
                )
            )

        # Hidden iframes
        analysis.hidden_iframe_count = VisualElementInspector.hidden_iframe_count(soup)
        if analysis.hidden_iframe_count:
            findings.append(
                Finding(
                    rule_id="DOM.IFRAME.HIDDEN",
                    category="EXPLOIT_ATTEMPT",
                    severity=SecurityVerdict.MEDIUM,
                    confidence=0.90,
                    message=f"{analysis.hidden_iframe_count} hidden iframe overlay(s).",
                    metadata=ConstrainedMetadata({
                        "page_url": page_url[:200],
                        "count": analysis.hidden_iframe_count,
                    }),
                )
            )

        dangerous_iframes = VisualElementInspector.dangerous_iframe_count(soup)
        if dangerous_iframes >= 2:
            findings.append(
                Finding(
                    rule_id="DOM.IFRAME.UNSANDBOXED",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.LOW,
                    confidence=0.80,
                    message=f"{dangerous_iframes} iframe(s) without tight sandbox.",
                    metadata=ConstrainedMetadata({
                        "page_url": page_url[:200],
                        "count": dangerous_iframes,
                    }),
                )
            )

        # Scripts
        ext, inline, matches = ScriptEvasionAnalyzer.analyze(soup)
        analysis.external_script_count = ext
        analysis.inline_script_count = inline
        analysis.obfuscated_script_matches = matches
        if matches:
            findings.append(
                Finding(
                    rule_id="DOM.SCRIPT.OBFUSCATION",
                    category="EXPLOIT_ATTEMPT",
                    severity=SecurityVerdict.HIGH,
                    confidence=0.91,
                    message=f"Obfuscated/suspicious JS signatures: {', '.join(matches[:6])}",
                    metadata=ConstrainedMetadata({
                        "page_url": page_url[:200],
                        "signatures": matches[:10],
                    }),
                )
            )

        # Brand + password context
        brands = BrandImpersonationEngine.detect(soup, page_url)
        analysis.detected_brand_claims = brands
        if brands and (analysis.password_input_count or analysis.formless_password_count):
            findings.append(
                Finding(
                    rule_id="DOM.BRAND.SPOOFED_LOGIN",
                    category="IMPERSONATION",
                    severity=SecurityVerdict.HIGH,
                    confidence=0.95,
                    message=(
                        f"Brand claim(s) {brands} with password fields on "
                        f"unofficial host."
                    ),
                    metadata=ConstrainedMetadata({
                        "page_url": page_url[:200],
                        "brands": brands,
                    }),
                )
            )

        # Dangerous URI schemes in links/actions
        analysis.dangerous_uri_links = count_dangerous_uris(soup)
        if analysis.dangerous_uri_links:
            findings.append(
                Finding(
                    rule_id="DOM.URI.DANGEROUS_SCHEME",
                    category="EXPLOIT_ATTEMPT",
                    severity=SecurityVerdict.HIGH,
                    confidence=0.93,
                    message=f"{analysis.dangerous_uri_links} javascript:/data: URI(s) in DOM.",
                    metadata=ConstrainedMetadata({
                        "page_url": page_url[:200],
                        "count": analysis.dangerous_uri_links,
                    }),
                )
            )

        analysis.inline_handler_count = count_inline_handlers(soup)
        if analysis.inline_handler_count >= 15:
            findings.append(
                Finding(
                    rule_id="DOM.EVENT.INLINE_HANDLER_DENSITY",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.LOW,
                    confidence=0.75,
                    message=f"High inline event-handler density ({analysis.inline_handler_count}).",
                    metadata=ConstrainedMetadata({
                        "page_url": page_url[:200],
                        "count": analysis.inline_handler_count,
                    }),
                )
            )

        return analysis, findings

    async def analyze(
        self, context: ScanContext, deadline_monotonic: float
    ) -> AnalyzerResult:
        start = time.monotonic()
        if time.monotonic() >= deadline_monotonic:
            return self._timeout(start)

        sources = DOMAnalyzerPayloadStore.resolve(context)
        if not sources:
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=0.0,
                confidence=0.85,
                findings=[],
            )

        all_findings: List[Finding] = []
        seen: Set[Tuple[str, str]] = set()
        cat = {
            "CREDENTIAL_HARVESTING": 0.0,
            "IMPERSONATION": 0.0,
            "EXPLOIT_ATTEMPT": 0.0,
            "SUSPICIOUS_INFRASTRUCTURE": 0.0,
        }

        try:
            for page_url, raw_dom in sources:
                if time.monotonic() >= deadline_monotonic:
                    break
                _, findings = await self._analyze_one(page_url, raw_dom, deadline_monotonic)
                for f in findings:
                    key = (f.rule_id, page_url)
                    if key in seen or len(all_findings) >= MAX_FINDINGS:
                        continue
                    seen.add(key)
                    all_findings.append(f)
                    if f.severity == SecurityVerdict.HIGH:
                        if "FORM" in f.rule_id or "CREDENTIAL" in f.category:
                            cat["CREDENTIAL_HARVESTING"] += 38.0
                        elif "BRAND" in f.rule_id or "IMPERSONATION" in f.category:
                            cat["IMPERSONATION"] += 36.0
                        else:
                            cat["EXPLOIT_ATTEMPT"] += 32.0
                    elif f.severity == SecurityVerdict.MEDIUM:
                        cat["SUSPICIOUS_INFRASTRUCTURE"] += 14.0
                    else:
                        cat["SUSPICIOUS_INFRASTRUCTURE"] += 5.0

            total = (
                min(cat["CREDENTIAL_HARVESTING"], 50.0)
                + min(cat["IMPERSONATION"], 40.0)
                + min(cat["EXPLOIT_ATTEMPT"], 40.0)
                + min(cat["SUSPICIOUS_INFRASTRUCTURE"], 25.0)
            )
            score = min(max(total, 0.0), 100.0)

            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=score,
                confidence=0.94 if all_findings else 0.85,
                findings=all_findings,
            )

        except Exception as exc:
            logger.exception(
                "DOMAnalyzer failure scan_id=%s", getattr(context, "scan_id", "?")
            )
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.FAILED,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                sanitized_error=f"DOM analyzer error: {type(exc).__name__}",
            )

    def _timeout(self, start: float) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.TIMEOUT,
            execution_time_ms=(time.monotonic() - start) * 1000.0,
            sanitized_error="Deadline exceeded during DOM structural analysis.",
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "DOMAnalyzer",
    "FormSecurityEngine",
    "BrandImpersonationEngine",
    "ScriptEvasionAnalyzer",
    "VisualElementInspector",
    "OriginSecurityUtils",
    "DOMAnalyzerPayloadStore",
    "DOMStructureAnalysis",
    "FormInspectionResult",
    "DOM_ANALYZER_VERSION",
]

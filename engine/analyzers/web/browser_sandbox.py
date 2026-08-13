"""
PhishGuard_AI - Hardened Sandboxed Browser Execution Engine
Production Playwright/Chromium engine for hostile pages.

Security goals
--------------
* Isolated BrowserContext per target (no shared storage / SW).
* Chromium sandbox kept enabled (no --no-sandbox).
* Application-level request interception as defense-in-depth
  (NOT a replacement for network egress controls).
* Block private / metadata / dangerous-scheme destinations.
* Block downloads, popups, service workers.
* Detect password forms, third-party form actions, canvas probes,
  automation probes, third-party websockets.
* Hard bounds on DOM, network log, screenshot, console, errors.
* Strict deadline + context cleanup on all paths.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import time
import urllib.parse
from dataclasses import dataclass, field
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
    from playwright.async_api import (
        Browser,
        BrowserContext,
        ConsoleMessage,
        Error as PlaywrightError,
        Page,
        Playwright,
        Request,
        Route,
        WebSocket,
        async_playwright,
    )
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = BrowserContext = Page = Playwright = Any  # type: ignore
    ConsoleMessage = PlaywrightError = Request = Route = WebSocket = Any  # type: ignore

# ============================================================================
# CONSTANTS
# ============================================================================

BROWSER_SANDBOX_VERSION = "6.2.0"
RULESET_VERSION = "2026.08.13"

NAVIGATION_TIMEOUT_MS = 10_000
POST_LOAD_WAIT_MS = 1_200
MAX_CONCURRENT_PAGES = 3
MAX_TARGETS = 5
MAX_NETWORK_ENTRIES = 400
MAX_WS_ENTRIES = 80
MAX_CONSOLE_ENTRIES = 150
MAX_JS_ERRORS = 80
MAX_DOM_BYTES = 1_500_000
MAX_SCREENSHOT_BYTES = 2_500_000
MAX_FINDINGS = 25

# Web ports only (8080/8443 allowed — common for kits)
ALLOWED_PORTS: Set[int] = {80, 443, 8000, 8080, 8443, 8888}

LOCALHOST_HOSTS: Set[str] = {
    "localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback",
}
CLOUD_METADATA_IPS: Set[str] = {
    "169.254.169.254", "169.254.170.2", "168.63.129.16", "fd00:ec2::254",
}
CLOUD_METADATA_HOSTS: Set[str] = {
    "metadata.google.internal", "metadata.azure.com", "instance-data",
}
K8S_CIDR = ipaddress.ip_network("10.96.0.0/12")
DOCKER_CIDR = ipaddress.ip_network("172.17.0.0/16")

CHROMIUM_ARGS: List[str] = [
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--no-first-run",
    "--disable-background-networking",
    "--disable-default-apps",
    "--disable-extensions",
    "--disable-sync",
    "--disable-translate",
    "--metrics-recording-only",
    "--mute-audio",
    "--no-default-browser-check",
    "--disable-popup-blocking",  # we kill popups ourselves
]

# Broader runtime probes (still best-effort; kits can evade)
RUNTIME_MONITOR_JS = r"""
(() => {
  try {
    window.__pg = window.__pg || {
      canvas: false,
      webdriver: false,
      permissions: false,
      plugins: false,
      languages: false
    };

    const flag = (k) => { try { window.__pg[k] = true; } catch(e) {} };

    // Canvas
    const patch = (obj, name) => {
      try {
        const orig = obj[name];
        if (typeof orig !== 'function') return;
        obj[name] = function() { flag('canvas'); return orig.apply(this, arguments); };
      } catch(e) {}
    };
    patch(HTMLCanvasElement.prototype, 'toDataURL');
    patch(HTMLCanvasElement.prototype, 'toBlob');
    if (window.CanvasRenderingContext2D) {
      patch(CanvasRenderingContext2D.prototype, 'getImageData');
    }

    // webdriver
    try {
      Object.defineProperty(navigator, 'webdriver', {
        get: function() { flag('webdriver'); return undefined; },
        configurable: true
      });
    } catch(e) {}

    // permissions / plugins / languages probes (common anti-bot)
    try {
      const desc = Object.getOwnPropertyDescriptor(Navigator.prototype, 'plugins');
      if (desc && desc.get) {
        Object.defineProperty(Navigator.prototype, 'plugins', {
          get: function() { flag('plugins'); return desc.get.call(this); }
        });
      }
    } catch(e) {}
    try {
      const ld = Object.getOwnPropertyDescriptor(Navigator.prototype, 'languages');
      if (ld && ld.get) {
        Object.defineProperty(Navigator.prototype, 'languages', {
          get: function() { flag('languages'); return ld.get.call(this); }
        });
      }
    } catch(e) {}
  } catch (e) {}
})();
"""

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass(slots=True)
class NetworkTrafficEntry:
    url: str
    method: str
    resource_type: str
    status_code: Optional[int] = None
    is_third_party: bool = False
    blocked: bool = False
    block_reason: Optional[str] = None


@dataclass(slots=True)
class WebSocketEventEntry:
    url: str
    event_type: str
    is_third_party: bool = False


@dataclass(slots=True)
class ConsoleLogEntry:
    level: str
    text: str


@dataclass(slots=True)
class SandboxRenderResult:
    target_url: str
    final_url: str = ""
    status_code: int = 0
    rendered_dom: str = ""
    dom_truncated: bool = False
    screenshot_bytes: bytes = b""
    title: str = ""
    redirect_chain: List[str] = field(default_factory=list)
    network_requests: List[NetworkTrafficEntry] = field(default_factory=list)
    websocket_events: List[WebSocketEventEntry] = field(default_factory=list)
    console_logs: List[ConsoleLogEntry] = field(default_factory=list)
    js_errors: List[str] = field(default_factory=list)
    popup_urls: List[str] = field(default_factory=list)
    download_attempts: List[str] = field(default_factory=list)
    has_password_fields: bool = False
    has_unencrypted_password_form: bool = False
    third_party_form_actions: List[str] = field(default_factory=list)
    has_canvas_fingerprinting: bool = False
    has_anti_analysis_probes: bool = False
    iframe_srcs: List[str] = field(default_factory=list)
    network_overflow: bool = False
    render_time_ms: float = 0.0
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# NETWORK GUARD
# ============================================================================

class ContextNetworkRouter:
    @classmethod
    def evaluate(cls, target_url: str) -> Tuple[bool, Optional[str]]:
        """Return (should_block, reason)."""
        if not target_url:
            return True, "EMPTY_URL"
        try:
            p = urllib.parse.urlparse(target_url)
            scheme = (p.scheme or "").lower()
            if scheme not in {"http", "https", "ws", "wss"}:
                return True, f"SCHEME_{scheme or 'none'}"
            if not p.hostname:
                return True, "NO_HOST"
            host = p.hostname.lower().rstrip(".")
            if host in LOCALHOST_HOSTS or host in CLOUD_METADATA_HOSTS:
                return True, "LOCAL_OR_METADATA_HOST"
            port = p.port or (443 if scheme in {"https", "wss"} else 80)
            if port not in ALLOWED_PORTS:
                return True, f"PORT_{port}"
            try:
                ip = ipaddress.ip_address(host)
                if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
                    ip = ip.ipv4_mapped
                if ip.is_loopback:
                    return True, "LOOPBACK"
                if ip.is_link_local:
                    return True, "LINK_LOCAL"
                if str(ip) in CLOUD_METADATA_IPS:
                    return True, "METADATA_IP"
                if ip.is_private:
                    if isinstance(ip, ipaddress.IPv4Address):
                        if ip in K8S_CIDR:
                            return True, "K8S"
                        if ip in DOCKER_CIDR:
                            return True, "DOCKER"
                    return True, "PRIVATE"
                if ip.is_reserved or ip.is_multicast:
                    return True, "RESERVED"
            except ValueError:
                pass
        except Exception:
            return True, "PARSE_FAIL"
        return False, None


# ============================================================================
# BROWSER MANAGER
# ============================================================================

class StrictSandboxManager:
    def __init__(self) -> None:
        self._pw: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._lock = asyncio.Lock()
        self._init_failed = False

    async def initialize(self) -> None:
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not installed")
        if self._init_failed:
            raise RuntimeError("Browser init previously failed")
        async with self._lock:
            if self._browser is not None:
                return
            try:
                self._pw = await async_playwright().start()
                self._browser = await self._pw.chromium.launch(
                    headless=True,
                    args=CHROMIUM_ARGS,
                    # sandbox stays ON (do not pass --no-sandbox)
                )
            except Exception:
                self._init_failed = True
                await self._force_close()
                raise

    async def new_context(self) -> BrowserContext:
        await self.initialize()
        assert self._browser is not None
        ctx = await self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            ignore_https_errors=False,   # do not blanket-ignore TLS errors
            java_script_enabled=True,
            accept_downloads=False,
            service_workers="block",
            bypass_csp=False,
        )
        await ctx.add_init_script(RUNTIME_MONITOR_JS)
        return ctx

    async def _force_close(self) -> None:
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        try:
            if self._pw:
                await self._pw.stop()
        except Exception:
            pass
        self._browser = None
        self._pw = None

    async def close(self) -> None:
        async with self._lock:
            await self._force_close()
            self._init_failed = False


# ============================================================================
# RENDER ENGINE
# ============================================================================

class SandboxRenderEngine:
    def __init__(self, manager: StrictSandboxManager) -> None:
        self.manager = manager

    async def render(
        self, target_url: str, timeout_ms: int = NAVIGATION_TIMEOUT_MS
    ) -> SandboxRenderResult:
        start = time.monotonic()
        result = SandboxRenderResult(target_url=target_url)

        if not PLAYWRIGHT_AVAILABLE:
            result.error_message = "PLAYWRIGHT_MISSING"
            result.render_time_ms = (time.monotonic() - start) * 1000.0
            return result

        block, reason = ContextNetworkRouter.evaluate(target_url)
        if block:
            result.error_message = f"SSRF_BLOCKED:{reason}"
            result.render_time_ms = (time.monotonic() - start) * 1000.0
            return result

        context: Optional[BrowserContext] = None
        page: Optional[Page] = None
        pending_tasks: List[asyncio.Task] = []

        try:
            context = await self.manager.new_context()
            origin_host = (urllib.parse.urlparse(target_url).hostname or "").lower()

            async def on_route(route: Route, request: Request) -> None:
                should_block, why = ContextNetworkRouter.evaluate(request.url)
                third = False
                try:
                    rh = (urllib.parse.urlparse(request.url).hostname or "").lower()
                    third = bool(origin_host and rh and rh != origin_host)
                except Exception:
                    pass
                if len(result.network_requests) < MAX_NETWORK_ENTRIES:
                    result.network_requests.append(
                        NetworkTrafficEntry(
                            url=request.url[:500],
                            method=request.method,
                            resource_type=request.resource_type,
                            is_third_party=third,
                            blocked=should_block,
                            block_reason=why,
                        )
                    )
                else:
                    result.network_overflow = True
                if should_block:
                    await route.abort("blockedbyclient")
                else:
                    await route.continue_()

            await context.route("**/*", on_route)
            page = await context.new_page()

            def on_console(msg: ConsoleMessage) -> None:
                if len(result.console_logs) < MAX_CONSOLE_ENTRIES:
                    result.console_logs.append(
                        ConsoleLogEntry(level=msg.type, text=(msg.text or "")[:300])
                    )

            def on_pageerror(err: PlaywrightError) -> None:
                if len(result.js_errors) < MAX_JS_ERRORS:
                    result.js_errors.append(str(err)[:400])

            async def on_popup(popup: Page) -> None:
                try:
                    result.popup_urls.append((popup.url or "")[:300])
                    await popup.close()
                except Exception:
                    pass

            async def on_download(download: Any) -> None:
                try:
                    result.download_attempts.append((download.url or "")[:300])
                    await download.cancel()
                except Exception:
                    pass

            def on_ws(ws: WebSocket) -> None:
                if len(result.websocket_events) >= MAX_WS_ENTRIES:
                    return
                third = False
                try:
                    rh = (urllib.parse.urlparse(ws.url).hostname or "").lower()
                    third = bool(origin_host and rh and rh != origin_host)
                except Exception:
                    pass
                result.websocket_events.append(
                    WebSocketEventEntry(
                        url=(ws.url or "")[:400],
                        event_type="opened",
                        is_third_party=third,
                    )
                )

            page.on("console", on_console)
            page.on("pageerror", on_pageerror)
            page.on("websocket", on_ws)

            # Track popup/download with managed tasks
            def _popup_handler(popup: Page) -> None:
                pending_tasks.append(asyncio.create_task(on_popup(popup)))

            def _dl_handler(download: Any) -> None:
                pending_tasks.append(asyncio.create_task(on_download(download)))

            page.on("popup", _popup_handler)
            page.on("download", _dl_handler)

            response = await page.goto(
                target_url,
                timeout=timeout_ms,
                wait_until="domcontentloaded",
            )
            if response:
                result.status_code = response.status
            result.final_url = page.url or target_url

            # Bounded post-load wait (JS kits)
            try:
                await page.wait_for_timeout(POST_LOAD_WAIT_MS)
            except Exception:
                pass

            result.title = (await page.title()) or ""

            # DOM
            try:
                dom = await page.content()
                encoded = dom.encode("utf-8", errors="replace")
                if len(encoded) > MAX_DOM_BYTES:
                    result.rendered_dom = encoded[:MAX_DOM_BYTES].decode(
                        "utf-8", errors="replace"
                    )
                    result.dom_truncated = True
                else:
                    result.rendered_dom = dom
            except Exception:
                pass

            # Password fields + form actions
            try:
                pwd = await page.query_selector_all("input[type='password']")
                result.has_password_fields = len(pwd) > 0
                if result.has_password_fields and result.final_url.startswith("http://"):
                    result.has_unencrypted_password_form = True

                actions = await page.eval_on_selector_all(
                    "form",
                    "els => els.map(e => e.getAttribute('action') || '')",
                )
                for action in actions or []:
                    if not action:
                        continue
                    abs_action = urllib.parse.urljoin(result.final_url, action)
                    ah = (urllib.parse.urlparse(abs_action).hostname or "").lower()
                    if origin_host and ah and ah != origin_host:
                        result.third_party_form_actions.append(abs_action[:400])
            except Exception:
                pass

            # Iframes
            try:
                iframes = await page.eval_on_selector_all(
                    "iframe",
                    "els => els.map(e => e.src || '').filter(Boolean)",
                )
                result.iframe_srcs = [s[:400] for s in (iframes or [])[:20]]
            except Exception:
                pass

            # Runtime flags
            try:
                flags = await page.evaluate(
                    """() => ({
                        canvas: !!(window.__pg && window.__pg.canvas),
                        webdriver: !!(window.__pg && window.__pg.webdriver),
                        plugins: !!(window.__pg && window.__pg.plugins),
                        languages: !!(window.__pg && window.__pg.languages)
                    })"""
                )
                result.has_canvas_fingerprinting = bool(flags.get("canvas"))
                result.has_anti_analysis_probes = bool(
                    flags.get("webdriver")
                    or flags.get("plugins")
                    or flags.get("languages")
                )
            except Exception:
                pass

            # Screenshot (bounded)
            try:
                shot = await page.screenshot(type="png", full_page=False)
                if shot and len(shot) <= MAX_SCREENSHOT_BYTES:
                    result.screenshot_bytes = shot
                elif shot:
                    result.screenshot_bytes = shot[:MAX_SCREENSHOT_BYTES]
            except Exception:
                pass

        except asyncio.TimeoutError:
            result.error_message = "NAVIGATION_TIMEOUT"
        except Exception as exc:
            result.error_message = f"RENDER_{type(exc).__name__}"
        finally:
            for t in pending_tasks:
                if not t.done():
                    t.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            try:
                if page:
                    await page.close()
            except Exception:
                pass
            try:
                if context:
                    await context.close()
            except Exception:
                pass

        result.render_time_ms = (time.monotonic() - start) * 1000.0
        return result


# ============================================================================
# PAYLOAD STORE
# ============================================================================

class BrowserSandboxPayloadStore:
    @staticmethod
    def resolve(context: ScanContext) -> List[str]:
        urls: Set[str] = set()
        meta = getattr(context, "metadata", None)
        data: Dict[str, Any] = getattr(meta, "data", {}) if meta else {}

        for key in ("target_url", "url", "primary_url"):
            v = data.get(key) or getattr(context, key, None)
            if isinstance(v, str) and v.startswith(("http://", "https://")):
                urls.add(v.strip())

        for item in data.get("parsed_urls") or []:
            if isinstance(item, dict):
                for k in ("raw_url", "canonical_url", "original_url"):
                    v = item.get(k)
                    if isinstance(v, str) and v.startswith(("http://", "https://")):
                        urls.add(v.strip())

        # Prefer pages already fetched if present (same targets)
        for page in data.get("fetched_web_pages") or []:
            if isinstance(page, dict):
                for k in ("final_url", "url"):
                    v = page.get(k)
                    if isinstance(v, str) and v.startswith(("http://", "https://")):
                        urls.add(v.strip())

        out = []
        for u in urls:
            block, _ = ContextNetworkRouter.evaluate(u)
            if not block:
                out.append(u)
            if len(out) >= MAX_TARGETS:
                break
        return out


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class BrowserSandboxAnalyzer(BaseAnalyzer):
    name: str = "browser_sandbox"
    version: str = BROWSER_SANDBOX_VERSION
    supported_scan_types: Set[ScanType] = {ScanType.URL, ScanType.COMPREHENSIVE}
    required_inputs: Set[str] = set()
    resource_class: AnalyzerResourceClass = AnalyzerResourceClass.COMPUTE
    priority: int = 8
    can_hard_override: bool = False

    def __init__(self) -> None:
        self.manager = StrictSandboxManager()
        self.engine = SandboxRenderEngine(self.manager)
        self._sem: Optional[asyncio.Semaphore] = None

    def _sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(MAX_CONCURRENT_PAGES)
        return self._sem

    async def close(self) -> None:
        await self.manager.close()

    async def _render_one(
        self, url: str, deadline: float
    ) -> Tuple[Optional[SandboxRenderResult], List[Finding]]:
        findings: List[Finding] = []
        if time.monotonic() >= deadline:
            return None, findings

        remaining_ms = int(max(1.0, deadline - time.monotonic()) * 1000)
        timeout_ms = min(NAVIGATION_TIMEOUT_MS, remaining_ms)

        async with self._sem():
            if time.monotonic() >= deadline:
                return None, findings
            res = await self.engine.render(url, timeout_ms=timeout_ms)

        if res.error_message:
            sev = SecurityVerdict.HIGH if "SSRF" in res.error_message else SecurityVerdict.LOW
            findings.append(
                Finding(
                    rule_id="BROWSER.RENDER.FAILED"
                    if "SSRF" not in res.error_message
                    else "BROWSER.SSRF.BLOCKED",
                    category="EXPLOIT_ATTEMPT"
                    if "SSRF" in res.error_message
                    else "SUSPICIOUS_INFRASTRUCTURE",
                    severity=sev,
                    confidence=0.95 if "SSRF" in res.error_message else 0.80,
                    message=f"Sandbox render issue for '{url}': {res.error_message}",
                    metadata=ConstrainedMetadata({
                        "url": url[:200],
                        "error": res.error_message,
                    }),
                )
            )
            return res, findings

        if res.has_unencrypted_password_form:
            findings.append(
                Finding(
                    rule_id="BROWSER.FORM.UNENCRYPTED_PASSWORD",
                    category="CREDENTIAL_HARVESTING",
                    severity=SecurityVerdict.HIGH,
                    confidence=0.97,
                    message=f"Password field on HTTP page '{res.final_url}'.",
                    metadata=ConstrainedMetadata({"url": res.final_url[:200]}),
                )
            )

        if res.third_party_form_actions:
            findings.append(
                Finding(
                    rule_id="BROWSER.FORM.THIRD_PARTY_ACTION",
                    category="CREDENTIAL_HARVESTING",
                    severity=SecurityVerdict.HIGH,
                    confidence=0.93,
                    message=(
                        f"Form posts to third-party host: "
                        f"{res.third_party_form_actions[0][:120]}"
                    ),
                    metadata=ConstrainedMetadata({
                        "url": res.final_url[:200],
                        "action": res.third_party_form_actions[0][:300],
                    }),
                )
            )

        if res.has_canvas_fingerprinting:
            findings.append(
                Finding(
                    rule_id="BROWSER.JS.CANVAS_FINGERPRINT",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.MEDIUM,
                    confidence=0.88,
                    message=f"Canvas fingerprinting on '{res.final_url}'.",
                    metadata=ConstrainedMetadata({"url": res.final_url[:200]}),
                )
            )

        if res.has_anti_analysis_probes:
            findings.append(
                Finding(
                    rule_id="BROWSER.EVASION.AUTOMATION_PROBE",
                    category="EXPLOIT_ATTEMPT",
                    severity=SecurityVerdict.HIGH,
                    confidence=0.90,
                    message=f"Anti-automation probes on '{res.final_url}'.",
                    metadata=ConstrainedMetadata({"url": res.final_url[:200]}),
                )
            )

        tp_ws = [w for w in res.websocket_events if w.is_third_party]
        if tp_ws:
            findings.append(
                Finding(
                    rule_id="BROWSER.WS.THIRD_PARTY",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.HIGH,
                    confidence=0.91,
                    message=f"Third-party WebSocket: {tp_ws[0].url[:120]}",
                    metadata=ConstrainedMetadata({
                        "page": res.final_url[:200],
                        "ws": tp_ws[0].url[:300],
                    }),
                )
            )

        if res.popup_urls:
            findings.append(
                Finding(
                    rule_id="BROWSER.UI.POPUP",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.MEDIUM,
                    confidence=0.85,
                    message=f"Popup(s) opened from '{res.final_url}'.",
                    metadata=ConstrainedMetadata({
                        "url": res.final_url[:200],
                        "popups": res.popup_urls[:5],
                    }),
                )
            )

        if res.download_attempts:
            findings.append(
                Finding(
                    rule_id="BROWSER.DOWNLOAD.ATTEMPT",
                    category="MALWARE_PAYLOAD",
                    severity=SecurityVerdict.HIGH,
                    confidence=0.92,
                    message=f"Download attempt from '{res.final_url}'.",
                    metadata=ConstrainedMetadata({
                        "url": res.final_url[:200],
                        "downloads": res.download_attempts[:5],
                    }),
                )
            )

        blocked = [n for n in res.network_requests if n.blocked]
        if len(blocked) >= 3:
            findings.append(
                Finding(
                    rule_id="BROWSER.NETWORK.BLOCKED_DESTINATIONS",
                    category="EXPLOIT_ATTEMPT",
                    severity=SecurityVerdict.MEDIUM,
                    confidence=0.90,
                    message=f"{len(blocked)} blocked network destinations from page.",
                    metadata=ConstrainedMetadata({
                        "url": res.final_url[:200],
                        "blocked_count": len(blocked),
                        "sample": [b.url[:120] for b in blocked[:3]],
                    }),
                )
            )

        return res, findings

    async def analyze(
        self, context: ScanContext, deadline_monotonic: float
    ) -> AnalyzerResult:
        start = time.monotonic()

        if not PLAYWRIGHT_AVAILABLE:
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.FAILED,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                sanitized_error="Playwright not installed in runtime.",
            )

        if time.monotonic() >= deadline_monotonic:
            return self._timeout(start)

        urls = BrowserSandboxPayloadStore.resolve(context)
        if not urls:
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=0.0,
                confidence=0.85,
                findings=[],
            )

        all_findings: List[Finding] = []
        buffers: List[Dict[str, Any]] = []
        cat = {
            "CREDENTIAL_HARVESTING": 0.0,
            "EXPLOIT_ATTEMPT": 0.0,
            "SUSPICIOUS_INFRASTRUCTURE": 0.0,
            "MALWARE_PAYLOAD": 0.0,
        }

        try:
            tasks = [self._render_one(u, deadline_monotonic) for u in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for item in results:
                if not isinstance(item, tuple):
                    continue
                res, findings = item
                for f in findings:
                    if len(all_findings) >= MAX_FINDINGS:
                        break
                    all_findings.append(f)
                    if f.severity == SecurityVerdict.HIGH:
                        if "FORM" in f.rule_id or "CREDENTIAL" in f.category:
                            cat["CREDENTIAL_HARVESTING"] += 40.0
                        elif "DOWNLOAD" in f.rule_id:
                            cat["MALWARE_PAYLOAD"] += 38.0
                        else:
                            cat["EXPLOIT_ATTEMPT"] += 36.0
                    elif f.severity == SecurityVerdict.MEDIUM:
                        cat["SUSPICIOUS_INFRASTRUCTURE"] += 16.0
                    else:
                        cat["SUSPICIOUS_INFRASTRUCTURE"] += 6.0

                if res and not res.error_message:
                    buffers.append({
                        "url": res.target_url,
                        "final_url": res.final_url,
                        "title": res.title,
                        "rendered_dom": res.rendered_dom,
                        "screenshot_bytes": res.screenshot_bytes,
                        "has_password_fields": res.has_password_fields,
                        "status_code": res.status_code,
                        "network_sample": [
                            {
                                "url": n.url[:200],
                                "method": n.method,
                                "third_party": n.is_third_party,
                                "blocked": n.blocked,
                            }
                            for n in res.network_requests[:40]
                        ],
                    })

            if buffers:
                meta = getattr(context, "metadata", None)
                if meta is not None and hasattr(meta, "data"):
                    meta.data["rendered_browser_pages"] = buffers

            total = (
                min(cat["CREDENTIAL_HARVESTING"], 50.0)
                + min(cat["EXPLOIT_ATTEMPT"], 45.0)
                + min(cat["MALWARE_PAYLOAD"], 40.0)
                + min(cat["SUSPICIOUS_INFRASTRUCTURE"], 30.0)
            )
            score = min(max(total, 0.0), 100.0)

            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=score,
                confidence=0.94 if buffers else 0.84,
                findings=all_findings,
            )

        except Exception as exc:
            logger.exception(
                "BrowserSandbox failure scan_id=%s", getattr(context, "scan_id", "?")
            )
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.FAILED,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                sanitized_error=f"Browser sandbox error: {type(exc).__name__}",
            )

    def _timeout(self, start: float) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.TIMEOUT,
            execution_time_ms=(time.monotonic() - start) * 1000.0,
            sanitized_error="Deadline exceeded during sandboxed browser execution.",
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "BrowserSandboxAnalyzer",
    "SandboxRenderEngine",
    "StrictSandboxManager",
    "ContextNetworkRouter",
    "SandboxRenderResult",
    "BrowserSandboxPayloadStore",
    "BROWSER_SANDBOX_VERSION",
]
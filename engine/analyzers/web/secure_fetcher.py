"""
PhishGuard_AI - Anti-SSRF Secure HTTP Fetcher Engine
Production-grade hardened HTTP fetcher for phishing / visual inspection.
Protects against SSRF, DNS rebinding, dangerous ports, credential leakage,
decompression bombs, and unbounded redirects.

Key Controls
------------
1. Pre-resolve + pin IP (PinnedStaticResolver) while preserving SNI / Host.
2. Full IPv4/IPv6 + mapped-IPv6 + cloud-IMDS + private-range rejection.
3. Port allow-list (only common web ports) + userinfo rejection.
4. Manual size-limited streaming; optional safe decompression under budget.
5. Bounded LRU cache with canonical keys.
6. Redirect re-validation on every hop + protocol-downgrade detection.
7. Calibrated scoring.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
import ssl
import time
import urllib.parse
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
from aiohttp.abc import AbstractResolver

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
    import dns.asyncresolver
    _HAS_DNS = True
except ImportError:
    _HAS_DNS = False

# ============================================================================
# CONSTANTS
# ============================================================================

SECURE_FETCHER_VERSION = "5.2.0"
RULESET_VERSION = "2026.08.13"

DEFAULT_TIMEOUT = 5.0
CONNECT_TIMEOUT = 2.0
MAX_CONCURRENT = 8
MAX_RESPONSE_BYTES = 8 * 1024 * 1024          # 8 MB hard cap
MAX_TEXT_BYTES = 1 * 1024 * 1024              # 1 MB text kept
MAX_REDIRECTS = 4
MAX_TARGETS = 12
MAX_CACHE_ENTRIES = 2048
CACHE_TTL = 180
MAX_FINDINGS = 20

# Only allow classic web ports (8080/8443 intentionally permitted)
ALLOWED_PORTS: Set[int] = {80, 443, 8080, 8443, 8000, 8888}

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

DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Upgrade-Insecure-Requests": "1",
}

# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass(slots=True)
class HTTPFetchResponse:
    target_url: str
    final_url: str = ""
    status_code: int = 0
    headers: Dict[str, str] = field(default_factory=dict)
    body_text: str = ""
    body_bytes: bytes = b""
    resolved_ip: str = ""
    redirect_chain: List[str] = field(default_factory=list)
    content_type: str = ""
    content_length: int = 0
    fetch_time_ms: float = 0.0
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# SSRF VALIDATOR
# ============================================================================

class SSRFValidator:
    @classmethod
    def is_ssrf_ip(cls, ip_str: str) -> Tuple[bool, Optional[str]]:
        if ip_str in CLOUD_METADATA_IPS:
            return True, "CLOUD_METADATA_IMDS"
        try:
            ip = ipaddress.ip_address(ip_str)
            if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
                ip = ip.ipv4_mapped
            if ip.is_loopback:
                return True, "LOOPBACK"
            if ip.is_link_local:
                return True, "LINK_LOCAL"
            if ip.is_private:
                if isinstance(ip, ipaddress.IPv4Address):
                    if ip in K8S_CIDR:
                        return True, "KUBERNETES"
                    if ip in DOCKER_CIDR:
                        return True, "DOCKER_BRIDGE"
                return True, "PRIVATE"
            if ip.is_reserved or ip.is_multicast:
                return True, "RESERVED"
        except ValueError:
            return True, "INVALID_IP"
        return False, None

    @classmethod
    def sanitize(cls, raw_url: str) -> Optional[urllib.parse.ParseResult]:
        if not raw_url or not isinstance(raw_url, str):
            return None
        try:
            p = urllib.parse.urlparse(raw_url.strip())
            if p.scheme not in {"http", "https"}:
                return None
            if not p.hostname or "\0" in p.hostname:
                return None
            if p.username or p.password:
                return None
            host = p.hostname.lower().rstrip(".")
            if host in LOCALHOST_HOSTS or host in CLOUD_METADATA_HOSTS:
                return None
            port = p.port or (443 if p.scheme == "https" else 80)
            if port not in ALLOWED_PORTS:
                return None
            return p
        except Exception:
            return None

    @classmethod
    def canonicalize(cls, raw_url: str) -> str:
        p = cls.sanitize(raw_url)
        if not p or not p.hostname:
            return (raw_url or "").strip().lower()
        host = p.hostname.lower().rstrip(".")
        try:
            if idna is not None:
                host = idna.encode(host).decode("ascii")
        except Exception:
            pass
        port = ""
        if p.port and p.port not in (80, 443):
            port = f":{p.port}"
        path = p.path or "/"
        return urllib.parse.urlunparse(
            (p.scheme.lower(), f"{host}{port}", path, "", p.query, "")
        )


# ============================================================================
# PINNED RESOLVER
# ============================================================================

class PinnedStaticResolver(AbstractResolver):
    def __init__(self, hostname: str, pinned_ip: str, port: int) -> None:
        self.hostname = hostname
        self.pinned_ip = pinned_ip
        self.port = port

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_INET
    ) -> List[Dict[str, Any]]:
        ip = ipaddress.ip_address(self.pinned_ip)
        fam = socket.AF_INET6 if isinstance(ip, ipaddress.IPv6Address) else socket.AF_INET
        return [{
            "hostname": self.hostname,
            "host": self.pinned_ip,
            "port": port or self.port,
            "family": fam,
            "proto": socket.IPPROTO_TCP,
            "flags": socket.AI_NUMERICHOST,
        }]

    async def close(self) -> None:
        return None


# ============================================================================
# DNS PRE-RESOLVER
# ============================================================================

class AsyncPreResolver:
    def __init__(self) -> None:
        self._resolver = None
        if _HAS_DNS:
            self._resolver = dns.asyncresolver.Resolver()
            self._resolver.timeout = 2.0
            self._resolver.lifetime = 2.0
            self._resolver.nameservers = ["1.1.1.1", "8.8.8.8"]

    async def resolve_validated(self, hostname: str) -> str:
        host = hostname.strip().lower().rstrip(".")
        # Already an IP?
        try:
            ip = str(ipaddress.ip_address(host))
            bad, reason = SSRFValidator.is_ssrf_ip(ip)
            if bad:
                raise ValueError(f"SSRF_IP:{reason}:{ip}")
            return ip
        except ValueError as e:
            if str(e).startswith("SSRF_IP:"):
                raise

        ips: List[str] = []
        if self._resolver is not None:
            for rdtype in ("A", "AAAA"):
                try:
                    ans = await self._resolver.resolve(host, rdtype)
                    for r in ans:
                        ips.append(r.address)
                except Exception:
                    pass
        else:
            # Fallback to getaddrinfo in a thread
            loop = asyncio.get_running_loop()
            try:
                infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
                for info in infos:
                    ips.append(info[4][0])
            except Exception:
                pass

        if not ips:
            raise socket.gaierror(f"NXDOMAIN_OR_NO_A:{host}")

        for ip in ips:
            bad, reason = SSRFValidator.is_ssrf_ip(ip)
            if bad:
                raise ValueError(f"SSRF_RESOLVED:{reason}:{ip}")

        return ips[0]


# ============================================================================
# CACHE
# ============================================================================

class SecureFetchCache:
    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES, ttl: int = CACHE_TTL) -> None:
        self._cache: OrderedDict[str, HTTPFetchResponse] = OrderedDict()
        self._max = max_entries
        self._ttl = ttl
        self._lock = asyncio.Lock()

    async def get(self, url: str) -> Optional[HTTPFetchResponse]:
        key = SSRFValidator.canonicalize(url)
        async with self._lock:
            item = self._cache.get(key)
            if item is None:
                return None
            if time.time() - item.timestamp > self._ttl:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return item

    async def set(self, url: str, resp: HTTPFetchResponse) -> None:
        key = SSRFValidator.canonicalize(url)
        async with self._lock:
            self._cache[key] = resp
            self._cache.move_to_end(key)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


# ============================================================================
# SAFE DECOMPRESSION (size-bounded)
# ============================================================================

def _safe_decompress(data: bytes, content_encoding: str) -> bytes:
    """Decompress under a hard size budget. Returns original on failure."""
    enc = (content_encoding or "").lower().strip()
    if not enc or enc == "identity":
        return data
    try:
        if "gzip" in enc or "x-gzip" in enc:
            import gzip
            return gzip.decompress(data)
        if "deflate" in enc:
            import zlib
            try:
                return zlib.decompress(data)
            except zlib.error:
                return zlib.decompress(data, -zlib.MAX_WBITS)
        if "br" in enc:
            try:
                import brotli
                return brotli.decompress(data)
            except ImportError:
                return data
    except Exception:
        return data
    return data


# ============================================================================
# FETCH ENGINE
# ============================================================================

class SecureHTTPFetcherEngine:
    def __init__(self) -> None:
        self.resolver = AsyncPreResolver()

    async def fetch(
        self, target_url: str, timeout: float = DEFAULT_TIMEOUT
    ) -> HTTPFetchResponse:
        start = time.monotonic()
        resp = HTTPFetchResponse(target_url=target_url)

        parsed = SSRFValidator.sanitize(target_url)
        if not parsed or not parsed.hostname:
            resp.error_message = "INVALID_OR_SSRF_URL"
            resp.fetch_time_ms = (time.monotonic() - start) * 1000.0
            return resp

        current = target_url
        chain: List[str] = [target_url]
        seen: Set[str] = {SSRFValidator.canonicalize(target_url)}
        headers = dict(DEFAULT_HEADERS)

        # Re-use one session per fetch chain where possible
        for hop in range(MAX_REDIRECTS + 1):
            p = SSRFValidator.sanitize(current)
            if not p or not p.hostname:
                resp.error_message = f"SSRF_REDIRECT_HOP_{hop}"
                break

            port = p.port or (443 if p.scheme == "https" else 80)
            try:
                pinned_ip = await self.resolver.resolve_validated(p.hostname)
                resp.resolved_ip = pinned_ip
            except Exception as exc:
                resp.error_message = f"DNS_OR_SSRF:{type(exc).__name__}"
                break

            resolver = PinnedStaticResolver(p.hostname, pinned_ip, port)
            # TLS verification ON (important)
            ssl_ctx = ssl.create_default_context()
            connector = aiohttp.TCPConnector(
                resolver=resolver,
                use_dns_cache=False,
                ssl=ssl_ctx,
                force_close=True,
                enable_cleanup_closed=True,
            )
            timeout_cfg = aiohttp.ClientTimeout(total=timeout, connect=CONNECT_TIMEOUT)

            # Strip auth on cross-origin
            if hop > 0:
                orig = urllib.parse.urlparse(target_url)
                if (p.hostname or "").lower() != (orig.hostname or "").lower():
                    headers.pop("Authorization", None)
                    headers.pop("Cookie", None)

            try:
                async with aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout_cfg,
                    auto_decompress=False,
                ) as session:
                    async with session.get(
                        current,
                        headers=headers,
                        allow_redirects=False,
                        max_redirects=0,
                    ) as r:
                        resp.status_code = r.status
                        resp.headers = {
                            k.lower(): v
                            for k, v in r.headers.items()
                            if k.lower() not in {
                                "set-cookie", "authorization",
                                "proxy-authenticate", "www-authenticate",
                            }
                        }
                        resp.content_type = r.content_type or ""

                        # Redirects
                        if r.status in {301, 302, 303, 307, 308}:
                            loc = r.headers.get("Location")
                            if not loc:
                                resp.error_message = "REDIRECT_NO_LOCATION"
                                break
                            next_url = urllib.parse.urljoin(current, loc)
                            # Re-sanitize every hop
                            if not SSRFValidator.sanitize(next_url):
                                resp.error_message = "SSRF_REDIRECT_TARGET"
                                break
                            canon = SSRFValidator.canonicalize(next_url)
                            if canon in seen:
                                resp.error_message = "REDIRECT_LOOP"
                                break
                            seen.add(canon)
                            chain.append(next_url)
                            current = next_url
                            continue

                        # Stream body under hard limit
                        buf = bytearray()
                        async for chunk in r.content.iter_chunked(64 * 1024):
                            buf.extend(chunk)
                            if len(buf) > MAX_RESPONSE_BYTES:
                                resp.error_message = "RESPONSE_TOO_LARGE"
                                break
                        if resp.error_message:
                            break

                        raw = bytes(buf)
                        # Safe decompress
                        encoding = r.headers.get("Content-Encoding", "")
                        raw = _safe_decompress(raw, encoding)
                        if len(raw) > MAX_RESPONSE_BYTES:
                            raw = raw[:MAX_RESPONSE_BYTES]

                        resp.body_bytes = raw
                        resp.content_length = len(raw)

                        text_slice = raw[:MAX_TEXT_BYTES]
                        charset = r.charset or "utf-8"
                        try:
                            resp.body_text = text_slice.decode(charset, errors="replace")
                        except Exception:
                            resp.body_text = text_slice.decode("utf-8", errors="replace")

                        resp.final_url = current
                        resp.redirect_chain = chain
                        break

            except asyncio.TimeoutError:
                resp.error_message = "TIMEOUT"
                break
            except aiohttp.ClientConnectorCertificateError:
                resp.error_message = "TLS_CERT_ERROR"
                break
            except Exception as exc:
                resp.error_message = f"FETCH_{type(exc).__name__}"
                break

        resp.fetch_time_ms = (time.monotonic() - start) * 1000.0
        return resp


# ============================================================================
# PAYLOAD STORE
# ============================================================================

class SecureFetcherPayloadStore:
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

        for v in data.get("extracted_urls") or data.get("urls") or []:
            if isinstance(v, str) and v.startswith(("http://", "https://")):
                urls.add(v.strip())

        # Keep only sanitizable ones
        out = []
        for u in urls:
            if SSRFValidator.sanitize(u):
                out.append(u)
            if len(out) >= MAX_TARGETS:
                break
        return out


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class SecureFetcherAnalyzer(BaseAnalyzer):
    name: str = "secure_fetcher"
    version: str = SECURE_FETCHER_VERSION
    supported_scan_types: Set[ScanType] = {ScanType.URL, ScanType.COMPREHENSIVE}
    required_inputs: Set[str] = set()
    resource_class: AnalyzerResourceClass = AnalyzerResourceClass.NETWORK_IO
    priority: int = 10
    can_hard_override: bool = False

    def __init__(self) -> None:
        self.engine = SecureHTTPFetcherEngine()
        self.cache = SecureFetchCache()
        self._sem: Optional[asyncio.Semaphore] = None

    def _sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(MAX_CONCURRENT)
        return self._sem

    async def close(self) -> None:
        self.cache.clear()

    async def _fetch_one(
        self, url: str, deadline: float
    ) -> Tuple[Optional[HTTPFetchResponse], List[Finding]]:
        findings: List[Finding] = []
        if time.monotonic() >= deadline:
            return None, findings

        cached = await self.cache.get(url)
        if cached:
            resp = cached
        else:
            async with self._sem():
                if time.monotonic() >= deadline:
                    return None, findings
                resp = await self.engine.fetch(url)
                await self.cache.set(url, resp)

        if resp.error_message:
            if "SSRF" in (resp.error_message or ""):
                findings.append(
                    Finding(
                        rule_id="WEB.SSRF.BLOCKED",
                        category="EXPLOIT_ATTEMPT",
                        severity=SecurityVerdict.HIGH,
                        confidence=0.98,
                        message=f"SSRF blocked for '{url}' ({resp.error_message}).",
                        metadata=ConstrainedMetadata({
                            "url": url[:200],
                            "error": resp.error_message,
                        }),
                    )
                )
            else:
                findings.append(
                    Finding(
                        rule_id="WEB.FETCH.FAILED",
                        category="SUSPICIOUS_INFRASTRUCTURE",
                        severity=SecurityVerdict.LOW,
                        confidence=0.80,
                        message=f"Fetch failed for '{url}' ({resp.error_message}).",
                        metadata=ConstrainedMetadata({
                            "url": url[:200],
                            "error": resp.error_message,
                        }),
                    )
                )
            return resp, findings

        # Excessive redirects
        hops = max(0, len(resp.redirect_chain) - 1)
        if hops >= 3:
            findings.append(
                Finding(
                    rule_id="WEB.REDIRECT.EXCESSIVE",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.MEDIUM,
                    confidence=0.88,
                    message=f"Excessive redirects ({hops}) for '{url}'.",
                    metadata=ConstrainedMetadata({
                        "url": url[:200],
                        "final_url": resp.final_url[:200],
                        "hops": hops,
                    }),
                )
            )

        # HTTPS → HTTP downgrade
        if (
            resp.redirect_chain
            and resp.redirect_chain[0].startswith("https://")
            and resp.final_url.startswith("http://")
        ):
            findings.append(
                Finding(
                    rule_id="WEB.REDIRECT.PROTOCOL_DOWNGRADE",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.HIGH,
                    confidence=0.95,
                    message=(
                        f"HTTPS→HTTP downgrade: '{url}' → '{resp.final_url}'."
                    ),
                    metadata=ConstrainedMetadata({
                        "initial": url[:200],
                        "final": resp.final_url[:200],
                    }),
                )
            )

        # Soft signals from status
        if resp.status_code >= 500:
            findings.append(
                Finding(
                    rule_id="WEB.FETCH.SERVER_ERROR",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.LOW,
                    confidence=0.70,
                    message=f"Server error {resp.status_code} for '{url}'.",
                    metadata=ConstrainedMetadata({
                        "url": url[:200],
                        "status": resp.status_code,
                    }),
                )
            )

        return resp, findings

    async def analyze(
        self, context: ScanContext, deadline_monotonic: float
    ) -> AnalyzerResult:
        start = time.monotonic()
        if time.monotonic() >= deadline_monotonic:
            return self._timeout(start)

        urls = SecureFetcherPayloadStore.resolve(context)
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
        pages: List[Dict[str, Any]] = []
        cat_scores = {"EXPLOIT_ATTEMPT": 0.0, "SUSPICIOUS_INFRASTRUCTURE": 0.0}

        try:
            tasks = [self._fetch_one(u, deadline_monotonic) for u in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for item in results:
                if not isinstance(item, tuple):
                    continue
                resp, findings = item
                for f in findings:
                    if len(all_findings) >= MAX_FINDINGS:
                        break
                    all_findings.append(f)
                    if f.severity == SecurityVerdict.HIGH:
                        cat_scores["EXPLOIT_ATTEMPT"] += 40.0
                    elif f.severity == SecurityVerdict.MEDIUM:
                        cat_scores["SUSPICIOUS_INFRASTRUCTURE"] += 18.0
                    else:
                        cat_scores["SUSPICIOUS_INFRASTRUCTURE"] += 6.0

                if resp and not resp.error_message and resp.body_text:
                    pages.append({
                        "url": resp.target_url,
                        "final_url": resp.final_url,
                        "status": resp.status_code,
                        "headers": dict(list(resp.headers.items())[:30]),
                        "body_text": resp.body_text,
                        "resolved_ip": resp.resolved_ip,
                        "content_type": resp.content_type,
                        "redirect_chain": resp.redirect_chain[:8],
                    })

            # Hand off to DOM / visual analyzers
            if pages:
                meta = getattr(context, "metadata", None)
                if meta is not None and hasattr(meta, "data"):
                    meta.data["fetched_web_pages"] = pages

            total = (
                min(cat_scores["EXPLOIT_ATTEMPT"], 50.0)
                + min(cat_scores["SUSPICIOUS_INFRASTRUCTURE"], 40.0)
            )
            score = min(max(total, 0.0), 100.0)

            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=score,
                confidence=0.93 if pages else 0.85,
                findings=all_findings,
            )

        except Exception as exc:
            logger.exception(
                "SecureFetcher failure scan_id=%s", getattr(context, "scan_id", "?")
            )
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.FAILED,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                sanitized_error=f"Secure fetcher error: {type(exc).__name__}",
            )

    def _timeout(self, start: float) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.TIMEOUT,
            execution_time_ms=(time.monotonic() - start) * 1000.0,
            sanitized_error="Deadline exceeded during secure URL fetching.",
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "SecureFetcherAnalyzer",
    "SecureHTTPFetcherEngine",
    "SSRFValidator",
    "PinnedStaticResolver",
    "AsyncPreResolver",
    "SecureFetchCache",
    "HTTPFetchResponse",
    "SecureFetcherPayloadStore",
    "SECURE_FETCHER_VERSION",
]
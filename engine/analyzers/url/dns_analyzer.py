"""
PhishGuard_AI - Async DNS Intelligence & Threat Analyzer
Production-grade DNS analyzer for fast-flux, dangling CNAME takeover,
DNS rebinding / SSRF, Dynamic DNS abuse, Null MX, and basic infrastructure risk.

Key Controls
------------
1. Concurrent async resolution (A/AAAA/CNAME/MX/NS/TXT) with timeouts & semaphore.
2. SSRF / cloud metadata / private-IP detection (including rebinding patterns).
3. Improved fast-flux heuristics (TTL + subnet diversity + CDN exclusion).
4. Dangling CNAME takeover detection (fingerprint + NXDOMAIN/no-A verification).
5. Dynamic DNS provider detection.
6. Bounded in-memory cache with size + TTL limits.
7. Calibrated scoring with per-category caps.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import dns.asyncresolver
import dns.exception
import dns.rdatatype
import dns.resolver

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

# ============================================================================
# CONSTANTS
# ============================================================================

DNS_ANALYZER_VERSION = "3.2.0"
RULESET_VERSION = "2026.08.13"

DNS_QUERY_TIMEOUT = 2.0
MAX_CONCURRENT = 8
MAX_DOMAINS = 20
FAST_FLUX_TTL = 300
CACHE_MAX_ENTRIES = 2048
CACHE_DEFAULT_TTL = 180
MAX_FINDINGS = 25

CLOUD_METADATA_IPS: Set[str] = {
    "169.254.169.254",   # AWS / GCP / Azure IMDS
    "169.254.170.2",     # AWS ECS
    "168.63.129.16",     # Azure Wire Server
    "fd00:ec2::254",     # AWS IPv6 IMDS
}

CLOUD_METADATA_HOSTS: Set[str] = {
    "metadata.google.internal",
    "metadata.azure.com",
    "instance-data",
}

DDNS_SUFFIXES: Set[str] = {
    "duckdns.org", "no-ip.com", "noip.me", "no-ip.org", "ddns.net",
    "ngrok.io", "ngrok-free.app", "loca.lt", "pagekite.me", "serveo.net",
    "hopto.org", "zapto.org", "sytes.net", "dynu.com", "freedns.afraid.org",
}

# CNAME fingerprints that are commonly vulnerable to takeover
# (value = human-readable service name)
TAKEOVER_FINGERPRINTS: Dict[str, str] = {
    "s3.amazonaws.com": "AWS S3",
    "s3-website": "AWS S3 Website",
    "github.io": "GitHub Pages",
    "githubusercontent.com": "GitHub",
    "azurewebsites.net": "Azure App Service",
    "cloudapp.net": "Azure CloudApp",
    "herokuapp.com": "Heroku",
    "herokudns.com": "Heroku",
    "myshopify.com": "Shopify",
    "zendesk.com": "Zendesk",
    "pantheonsite.io": "Pantheon",
    "bitbucket.io": "Bitbucket",
    "ghost.io": "Ghost",
    "cargo.site": "Cargo",
    "surge.sh": "Surge",
    "netlify.com": "Netlify",
    "pages.dev": "Cloudflare Pages",
    "web.app": "Firebase",
    "firebaseapp.com": "Firebase",
}

# Known large CDN / cloud prefixes – used to reduce fast-flux false positives
CDN_HINTS: Set[str] = {
    "cloudflare", "cloudfront", "akamai", "fastly", "azureedge",
    "googleusercontent", "amazonaws", "edgekey", "edgesuite",
}

# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass(slots=True)
class DNSResolutionResult:
    domain: str
    a_records: List[str] = field(default_factory=list)
    aaaa_records: List[str] = field(default_factory=list)
    cname_records: List[str] = field(default_factory=list)
    mx_records: List[str] = field(default_factory=list)
    ns_records: List[str] = field(default_factory=list)
    txt_records: List[str] = field(default_factory=list)
    min_ttl: int = 86400
    has_dnskey: bool = False
    is_nxdomain: bool = False
    is_null_mx: bool = False
    resolution_error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    cache_ttl: int = CACHE_DEFAULT_TTL


# ============================================================================
# BOUNDED LRU CACHE
# ============================================================================

class DNSCacheManager:
    """Simple TTL + size bounded LRU cache."""

    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES) -> None:
        self._cache: OrderedDict[str, DNSResolutionResult] = OrderedDict()
        self._max = max_entries

    def get(self, domain: str) -> Optional[DNSResolutionResult]:
        key = domain.lower()
        res = self._cache.get(key)
        if res is None:
            return None
        if time.time() - res.timestamp > res.cache_ttl:
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return res

    def set(self, domain: str, result: DNSResolutionResult) -> None:
        key = domain.lower()
        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


# ============================================================================
# ASYNC RESOLVER
# ============================================================================

class AsyncDNSResolverEngine:
    def __init__(self) -> None:
        self.resolver = dns.asyncresolver.Resolver(configure=True)
        self.resolver.timeout = DNS_QUERY_TIMEOUT
        self.resolver.lifetime = DNS_QUERY_TIMEOUT
        self.fallback_nameservers = ["1.1.1.1", "8.8.8.8", "9.9.9.9"]

    async def _query(
        self, domain: str, rdtype: dns.rdatatype.RdataType
    ) -> Tuple[List[str], int]:
        records: List[str] = []
        ttl = 86400
        try:
            answers = await asyncio.wait_for(
                self.resolver.resolve(domain, rdtype),
                timeout=DNS_QUERY_TIMEOUT + 0.4,
            )
            for rdata in answers:
                records.append(str(rdata).strip().rstrip("."))
            if answers.rrset is not None and answers.rrset.ttl is not None:
                ttl = int(answers.rrset.ttl)
        except (dns.resolver.NoAnswer, dns.resolver.NoNameservers, asyncio.TimeoutError):
            pass
        except dns.resolver.NXDOMAIN:
            raise
        except Exception as exc:
            logger.debug("DNS %s %s: %s", domain, rdtype, type(exc).__name__)
        return records, ttl

    async def resolve(self, domain: str) -> DNSResolutionResult:
        result = DNSResolutionResult(domain=domain)
        clean = domain.strip().lower().rstrip(".")
        if not clean or len(clean) > 253:
            result.resolution_error = "INVALID_HOSTNAME"
            return result

        try:
            tasks = [
                self._query(clean, dns.rdatatype.A),
                self._query(clean, dns.rdatatype.AAAA),
                self._query(clean, dns.rdatatype.CNAME),
                self._query(clean, dns.rdatatype.MX),
                self._query(clean, dns.rdatatype.NS),
                self._query(clean, dns.rdatatype.TXT),
                self._query(clean, dns.rdatatype.DNSKEY),
            ]
            gathered = await asyncio.gather(*tasks, return_exceptions=True)

            def take(idx: int) -> Tuple[List[str], int]:
                r = gathered[idx]
                if isinstance(r, Exception):
                    return [], 86400
                return r  # type: ignore[return-value]

            a, ttl_a = take(0)
            aaaa, ttl_aaaa = take(1)
            cname, _ = take(2)
            mx, _ = take(3)
            ns, _ = take(4)
            txt, _ = take(5)
            dnskey, _ = take(6)

            result.a_records = a
            result.aaaa_records = aaaa
            result.cname_records = cname
            result.mx_records = mx
            result.ns_records = ns
            result.txt_records = txt
            result.min_ttl = min(ttl_a, ttl_aaaa, 86400)
            result.has_dnskey = bool(dnskey)

            # Proper Null MX detection: priority 0 and exchange "."
            for m in mx:
                # dnspython usually returns "0 ." or "0 ."
                parts = m.split()
                if len(parts) >= 2 and parts[0] == "0" and parts[1] in {".", ""}:
                    result.is_null_mx = True
                    break
                if m.strip() in {"0 .", "0."}:
                    result.is_null_mx = True
                    break

            # Cache TTL = min of observed TTLs, clamped
            result.cache_ttl = max(30, min(result.min_ttl, CACHE_DEFAULT_TTL))

        except dns.resolver.NXDOMAIN:
            result.is_nxdomain = True
            result.resolution_error = "NXDOMAIN"
        except Exception as exc:
            result.resolution_error = type(exc).__name__

        return result


# ============================================================================
# SSRF / REBINDING
# ============================================================================

class SSRFDefenseEngine:
    K8S_CIDR = ipaddress.ip_network("10.96.0.0/12")
    DOCKER_CIDR = ipaddress.ip_network("172.17.0.0/16")

    @classmethod
    def evaluate(
        cls, domain: str, ips: List[str]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        d = domain.lower()
        if d in CLOUD_METADATA_HOSTS:
            return True, "CLOUD_METADATA_HOSTNAME", d

        private_count = 0
        public_count = 0
        first_private: Optional[str] = None

        for ip_str in ips:
            if ip_str in CLOUD_METADATA_IPS:
                return True, "CLOUD_METADATA_IMDS", ip_str
            try:
                ip = ipaddress.ip_address(ip_str)
                if ip.is_loopback:
                    return True, "LOOPBACK_IP", ip_str
                if ip.is_link_local:
                    return True, "LINK_LOCAL_IP", ip_str
                if ip.is_private or ip.is_reserved:
                    private_count += 1
                    if first_private is None:
                        first_private = ip_str
                    if isinstance(ip, ipaddress.IPv4Address):
                        if ip in cls.K8S_CIDR:
                            return True, "KUBERNETES_SERVICE_IP", ip_str
                        if ip in cls.DOCKER_CIDR:
                            return True, "DOCKER_BRIDGE_IP", ip_str
                else:
                    public_count += 1
            except ValueError:
                continue

        # DNS rebinding: mix of public + private
        if private_count and public_count:
            return True, "DNS_REBINDING_MIXED", first_private
        if private_count and not public_count:
            return True, "RFC1918_PRIVATE_IP", first_private

        return False, None, None


# ============================================================================
# FAST-FLUX
# ============================================================================

class FastFluxEngine:
    @classmethod
    def subnet_entropy(cls, ips: List[str]) -> float:
        if len(ips) < 2:
            return 0.0
        buckets: Dict[str, int] = {}
        for ip in ips:
            try:
                parts = ip.split(".")
                if len(parts) == 4:
                    key = f"{parts[0]}.{parts[1]}"
                    buckets[key] = buckets.get(key, 0) + 1
            except Exception:
                pass
        if not buckets:
            return 0.0
        n = sum(buckets.values())
        return round(-sum((c / n) * math.log2(c / n) for c in buckets.values()), 2)

    @classmethod
    def is_likely_cdn(cls, cnames: List[str], ns: List[str]) -> bool:
        blob = " ".join(cnames + ns).lower()
        return any(h in blob for h in CDN_HINTS)

    @classmethod
    def evaluate(
        cls, a_records: List[str], min_ttl: int, cnames: List[str], ns: List[str]
    ) -> Tuple[bool, float]:
        if len(a_records) < 4 or min_ttl > FAST_FLUX_TTL:
            return False, 0.0
        if cls.is_likely_cdn(cnames, ns):
            return False, 0.0
        entropy = cls.subnet_entropy(a_records)
        # Require both diversity and low TTL
        if entropy >= 1.5 and min_ttl <= FAST_FLUX_TTL:
            return True, entropy
        return False, entropy


# ============================================================================
# CNAME TAKEOVER
# ============================================================================

class CNAMETakeoverEngine:
    @classmethod
    async def evaluate(
        cls,
        domain: str,
        cnames: List[str],
        resolver: AsyncDNSResolverEngine,
    ) -> Optional[Tuple[str, str, str]]:
        """
        Returns (cname_target, service_name, reason) if dangling.
        """
        for cname in cnames:
            c = cname.lower().rstrip(".")
            for pattern, service in TAKEOVER_FINGERPRINTS.items():
                if pattern in c or c.endswith(pattern):
                    # Verify dangling: target should not resolve to an address
                    try:
                        target_res = await resolver.resolve(c)
                        if target_res.is_nxdomain or (
                            not target_res.a_records and not target_res.aaaa_records
                        ):
                            return cname, service, "NXDOMAIN_OR_NO_A"
                    except Exception:
                        # Resolution failure also treated as suspicious
                        return cname, service, "RESOLUTION_FAILED"
        return None


# ============================================================================
# PAYLOAD STORE
# ============================================================================

class DNSPayloadStore:
    @staticmethod
    def resolve(context: ScanContext) -> List[str]:
        domains: Set[str] = set()
        meta = getattr(context, "metadata", None)
        data: Dict[str, Any] = getattr(meta, "data", {}) if meta else {}

        for item in data.get("parsed_urls") or []:
            if isinstance(item, dict):
                for key in ("org_domain", "host_normalized", "registered_domain"):
                    v = item.get(key)
                    if isinstance(v, str) and v.strip():
                        domains.add(v.strip().lower().rstrip("."))

        for key in ("domain", "org_domain", "registered_domain"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                domains.add(v.strip().lower().rstrip("."))

        for v in data.get("domains") or data.get("extracted_domains") or []:
            if isinstance(v, str) and v.strip():
                domains.add(v.strip().lower().rstrip("."))

        # Pure URL scan
        for key in ("url", "target_url"):
            v = data.get(key) or getattr(context, key, None)
            if isinstance(v, str) and "://" in v:
                try:
                    from urllib.parse import urlparse
                    host = urlparse(v).hostname
                    if host:
                        domains.add(host.lower().rstrip("."))
                except Exception:
                    pass

        # Filter obvious garbage
        out = [
            d for d in domains
            if d and "." in d and len(d) < 253 and not d.startswith(".")
        ]
        return out[:MAX_DOMAINS]


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class DNSAnalyzer(BaseAnalyzer):
    name: str = "dns_analyzer"
    version: str = DNS_ANALYZER_VERSION
    supported_scan_types: Set[ScanType] = {
        ScanType.DOMAIN, ScanType.URL, ScanType.EMAIL, ScanType.COMPREHENSIVE
    }
    required_inputs: Set[str] = set()
    resource_class: AnalyzerResourceClass = AnalyzerResourceClass.ASYNC_IO
    priority: int = 12
    can_hard_override: bool = False

    def __init__(self) -> None:
        self.resolver = AsyncDNSResolverEngine()
        self.cache = DNSCacheManager()
        self._sem: Optional[asyncio.Semaphore] = None

    def _get_sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(MAX_CONCURRENT)
        return self._sem

    async def close(self) -> None:
        self.cache.clear()

    async def _analyze_one(
        self, domain: str, deadline: float
    ) -> List[Finding]:
        findings: List[Finding] = []
        if time.monotonic() >= deadline:
            return findings

        cached = self.cache.get(domain)
        if cached:
            res = cached
        else:
            async with self._get_sem():
                if time.monotonic() >= deadline:
                    return findings
                res = await self.resolver.resolve(domain)
                self.cache.set(domain, res)

        # NXDOMAIN
        if res.is_nxdomain:
            findings.append(
                Finding(
                    rule_id="DNS.RESOLUTION.NXDOMAIN",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.LOW,
                    confidence=0.95,
                    message=f"Domain '{domain}' is NXDOMAIN.",
                    metadata=ConstrainedMetadata({"domain": domain}),
                )
            )
            return findings

        all_ips = res.a_records + res.aaaa_records

        # SSRF / rebinding
        is_ssrf, ssrf_type, target = SSRFDefenseEngine.evaluate(domain, all_ips)
        if is_ssrf and target:
            findings.append(
                Finding(
                    rule_id="DNS.SSRF.PRIVATE_OR_REBINDING",
                    category="EXPLOIT_ATTEMPT",
                    severity=SecurityVerdict.HIGH,
                    confidence=0.97,
                    message=f"DNS resolved '{domain}' to hazardous target ({ssrf_type}: {target}).",
                    metadata=ConstrainedMetadata({
                        "domain": domain,
                        "ssrf_type": str(ssrf_type),
                        "target": str(target),
                    }),
                )
            )

        # Fast-flux
        is_flux, entropy = FastFluxEngine.evaluate(
            res.a_records, res.min_ttl, res.cname_records, res.ns_records
        )
        if is_flux:
            findings.append(
                Finding(
                    rule_id="DNS.INFRASTRUCTURE.FAST_FLUX",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.HIGH,
                    confidence=0.90,
                    message=(
                        f"Possible fast-flux on '{domain}' "
                        f"(TTL={res.min_ttl}s, subnet_entropy={entropy}, ips={len(res.a_records)})."
                    ),
                    metadata=ConstrainedMetadata({
                        "domain": domain,
                        "min_ttl": res.min_ttl,
                        "entropy": entropy,
                        "ip_count": len(res.a_records),
                    }),
                )
            )

        # Null MX
        if res.is_null_mx:
            findings.append(
                Finding(
                    rule_id="DNS.MAIL.NULL_MX",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.LOW,
                    confidence=0.95,
                    message=f"Domain '{domain}' publishes Null MX (rejects mail).",
                    metadata=ConstrainedMetadata({"domain": domain}),
                )
            )

        # Dynamic DNS
        for suffix in DDNS_SUFFIXES:
            if domain == suffix or domain.endswith("." + suffix):
                findings.append(
                    Finding(
                        rule_id="DNS.INFRASTRUCTURE.DYNAMIC_DNS",
                        category="SUSPICIOUS_INFRASTRUCTURE",
                        severity=SecurityVerdict.MEDIUM,
                        confidence=0.94,
                        message=f"Domain '{domain}' uses Dynamic DNS provider '{suffix}'.",
                        metadata=ConstrainedMetadata({
                            "domain": domain,
                            "provider": suffix,
                        }),
                    )
                )
                break

        # Dangling CNAME takeover (async verification)
        if res.cname_records and time.monotonic() < deadline:
            takeover = await CNAMETakeoverEngine.evaluate(
                domain, res.cname_records, self.resolver
            )
            if takeover:
                cname, service, reason = takeover
                findings.append(
                    Finding(
                        rule_id="DNS.TAKEOVER.DANGLING_CNAME",
                        category="EXPLOIT_ATTEMPT",
                        severity=SecurityVerdict.HIGH,
                        confidence=0.91,
                        message=(
                            f"Dangling CNAME on '{domain}' → '{cname}' "
                            f"({service}, {reason}). Possible subdomain takeover."
                        ),
                        metadata=ConstrainedMetadata({
                            "domain": domain,
                            "cname": cname,
                            "service": service,
                            "reason": reason,
                        }),
                    )
                )

        return findings

    async def analyze(
        self, context: ScanContext, deadline_monotonic: float
    ) -> AnalyzerResult:
        start = time.monotonic()
        if time.monotonic() >= deadline_monotonic:
            return self._timeout(start)

        domains = DNSPayloadStore.resolve(context)
        if not domains:
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=0.0,
                confidence=0.86,
                findings=[],
            )

        findings_map: Dict[Tuple[str, str], Finding] = {}
        cat_scores: Dict[str, float] = {
            "EXPLOIT_ATTEMPT": 0.0,
            "SUSPICIOUS_INFRASTRUCTURE": 0.0,
        }

        try:
            tasks = [self._analyze_one(d, deadline_monotonic) for d in domains]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for res in results:
                if not isinstance(res, list):
                    continue
                for f in res:
                    if len(findings_map) >= MAX_FINDINGS:
                        break
                    dom = ""
                    if f.metadata and getattr(f.metadata, "data", None):
                        dom = str(f.metadata.data.get("domain", ""))
                    key = (f.rule_id, dom)
                    if key in findings_map:
                        continue
                    findings_map[key] = f
                    if f.severity == SecurityVerdict.HIGH:
                        cat_scores["EXPLOIT_ATTEMPT"] += 38.0
                    elif f.severity == SecurityVerdict.MEDIUM:
                        cat_scores["SUSPICIOUS_INFRASTRUCTURE"] += 22.0
                    else:
                        cat_scores["SUSPICIOUS_INFRASTRUCTURE"] += 8.0

            total = (
                min(cat_scores["EXPLOIT_ATTEMPT"], 55.0)
                + min(cat_scores["SUSPICIOUS_INFRASTRUCTURE"], 40.0)
            )
            score = min(max(total, 0.0), 100.0)

            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=score,
                confidence=0.94 if findings_map else 0.87,
                findings=list(findings_map.values()),
            )

        except Exception as exc:
            logger.exception(
                "DNSAnalyzer failure scan_id=%s", getattr(context, "scan_id", "?")
            )
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.FAILED,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                sanitized_error=f"DNS analysis error: {type(exc).__name__}",
            )

    def _timeout(self, start: float) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.TIMEOUT,
            execution_time_ms=(time.monotonic() - start) * 1000.0,
            sanitized_error="Deadline exceeded during DNS resolution analysis.",
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "DNSAnalyzer",
    "AsyncDNSResolverEngine",
    "DNSCacheManager",
    "DNSResolutionResult",
    "SSRFDefenseEngine",
    "FastFluxEngine",
    "CNAMETakeoverEngine",
    "DNSPayloadStore",
    "DNS_ANALYZER_VERSION",
]
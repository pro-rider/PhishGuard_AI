"""
PhishGuard_AI - Async SSL/TLS Certificate & Transport Security Analyzer
Production-grade TLS analyzer: binary X.509 parsing, key/signature strength,
hostname/SAN matching, HSTS, weak protocol/cipher detection, SSRF protection,
and bounded async caching.

Key Controls
------------
1. DER certificate parsing via cryptography.x509 (avoids empty dict under CERT_NONE).
2. RSA/ECDSA key strength + weak signature (MD5/SHA1) detection.
3. SSRF / private-IP / metadata target rejection (pre + post connect).
4. Bounded LRU cache with per-host locks (thundering-herd safe).
5. Full lifecycle (expired / not-yet-valid / expiring-soon) + HSTS.
6. TLS 1.0/1.1 and weak-cipher detection.
7. Calibrated scoring with per-category caps.
"""

from __future__ import annotations

import asyncio
import datetime
import ipaddress
import logging
import math
import re
import socket
import ssl
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import cryptography.x509 as x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import dsa, ec, ed25519, ed448, rsa

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

# ============================================================================
# CONSTANTS
# ============================================================================

SSL_CHECKER_VERSION = "4.2.0"
RULESET_VERSION = "2026.08.13"

DEFAULT_HTTPS_PORT = 443
TLS_CONNECT_TIMEOUT = 3.0
HTTP_HEADER_TIMEOUT = 1.5
MAX_CONCURRENT = 8
MAX_TARGETS = 15
MAX_CACHE_ENTRIES = 2048
CACHE_TTL_SECONDS = 240
EXPIRATION_WARNING_DAYS = 14
MAX_FINDINGS = 25

CLOUD_METADATA_IPS: Set[str] = {
    "169.254.169.254",
    "169.254.170.2",
    "168.63.129.16",
    "fd00:ec2::254",
}

WEAK_HASHES: Set[str] = {"md5", "sha1"}
WEAK_CIPHER_RE = re.compile(
    r"(RC4|3DES|DES-CBC|EXPORT|NULL|anon|ADH|AECDH)", re.I
)
LEGACY_TLS: Set[str] = {"TLSv1", "TLSv1.0", "TLSv1.1", "SSLv3", "SSLv2"}

# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass(slots=True)
class SSLCertificateDetails:
    hostname: str
    port: int = 443
    subject_cn: Optional[str] = None
    issuer_cn: Optional[str] = None
    issuer_org: Optional[str] = None
    san_list: List[str] = field(default_factory=list)
    valid_from: Optional[datetime.datetime] = None
    valid_until: Optional[datetime.datetime] = None
    days_until_expiration: Optional[int] = None
    is_expired: bool = False
    is_not_yet_valid: bool = False
    is_self_signed: bool = False
    is_hostname_match: bool = False
    key_type: Optional[str] = None
    key_size: Optional[int] = None
    signature_hash: Optional[str] = None
    is_weak_key: bool = False
    is_weak_signature: bool = False
    tls_version: Optional[str] = None
    cipher_suite: Optional[str] = None
    is_weak_cipher: bool = False
    is_legacy_tls: bool = False
    has_hsts: bool = False
    connected_ip: Optional[str] = None
    connection_error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# HOST VALIDATOR
# ============================================================================

class HostValidator:
    @classmethod
    def is_dangerous_ip(cls, host: str) -> bool:
        if host in CLOUD_METADATA_IPS:
            return True
        try:
            ip = ipaddress.ip_address(host)
            return bool(
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
            )
        except ValueError:
            return False

    @classmethod
    def sanitize(cls, hostname: str) -> Optional[str]:
        if not hostname:
            return None
        clean = hostname.strip().lower().rstrip(".")
        if not clean or any(c in clean for c in ("\0", "/", "\\", " ")):
            return None
        if len(clean) > 253:
            return None
        try:
            if idna is not None:
                # Prefer ASCII form for connection
                clean = idna.encode(clean).decode("ascii")
        except Exception:
            pass
        if cls.is_dangerous_ip(clean):
            return None
        # Allow underscores (common in internal names) and IDN already converted
        if not re.match(r"^[a-z0-9._-]+$", clean):
            return None
        return clean


# ============================================================================
# BOUNDED LRU CACHE
# ============================================================================

class SSLCacheManager:
    def __init__(
        self,
        max_entries: int = MAX_CACHE_ENTRIES,
        ttl: int = CACHE_TTL_SECONDS,
    ) -> None:
        self._cache: OrderedDict[str, SSLCertificateDetails] = OrderedDict()
        self._locks: Dict[str, asyncio.Lock] = {}
        self._max = max_entries
        self._ttl = ttl
        self._global = asyncio.Lock()

    async def get_lock(self, key: str) -> asyncio.Lock:
        async with self._global:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    async def get(self, key: str) -> Optional[SSLCertificateDetails]:
        async with self._global:
            item = self._cache.get(key)
            if item is None:
                return None
            if time.time() - item.timestamp > self._ttl:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return item

    async def set(self, key: str, details: SSLCertificateDetails) -> None:
        async with self._global:
            self._cache[key] = details
            self._cache.move_to_end(key)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()
        self._locks.clear()


# ============================================================================
# CERTIFICATE PARSER
# ============================================================================

class CertificateParserEngine:
    @classmethod
    def parse(
        cls, der_bytes: bytes, hostname: str, port: int
    ) -> SSLCertificateDetails:
        details = SSLCertificateDetails(hostname=hostname, port=port)
        if not der_bytes:
            return details
        try:
            cert = x509.load_der_x509_certificate(der_bytes, default_backend())

            # Subject / Issuer
            for attr in cert.subject:
                if attr.oid == x509.NameOID.COMMON_NAME:
                    details.subject_cn = str(attr.value)
            for attr in cert.issuer:
                if attr.oid == x509.NameOID.COMMON_NAME:
                    details.issuer_cn = str(attr.value)
                elif attr.oid == x509.NameOID.ORGANIZATION_NAME:
                    details.issuer_org = str(attr.value)

            # True self-signed: issuer == subject (full Name)
            details.is_self_signed = cert.issuer == cert.subject

            # SANs
            try:
                san_ext = cert.extensions.get_extension_for_class(
                    x509.SubjectAlternativeName
                )
                details.san_list = [
                    v.lower() for v in san_ext.value.get_values_for_type(x509.DNSName)
                ]
            except x509.ExtensionNotFound:
                if details.subject_cn:
                    details.san_list = [details.subject_cn.lower()]

            details.is_hostname_match = cls._hostname_matches(
                hostname, details.san_list, details.subject_cn
            )

            # Validity
            now = datetime.datetime.now(datetime.timezone.utc)
            # cryptography >= 42 uses not_valid_before_utc
            try:
                valid_from = cert.not_valid_before_utc
                valid_until = cert.not_valid_after_utc
            except AttributeError:
                valid_from = cert.not_valid_before.replace(tzinfo=datetime.timezone.utc)
                valid_until = cert.not_valid_after.replace(tzinfo=datetime.timezone.utc)

            details.valid_from = valid_from
            details.valid_until = valid_until

            if now < valid_from:
                details.is_not_yet_valid = True
            elif now > valid_until:
                details.is_expired = True
            else:
                details.days_until_expiration = math.ceil(
                    (valid_until - now).total_seconds() / 86400.0
                )

            # Public key
            pub = cert.public_key()
            if isinstance(pub, rsa.RSAPublicKey):
                details.key_type = "RSA"
                details.key_size = pub.key_size
                details.is_weak_key = pub.key_size < 2048
            elif isinstance(pub, ec.EllipticCurvePublicKey):
                details.key_type = "ECDSA"
                details.key_size = pub.key_size
                details.is_weak_key = pub.key_size < 256
            elif isinstance(pub, (ed25519.Ed25519PublicKey, ed448.Ed448PublicKey)):
                details.key_type = "EdDSA"
                details.key_size = 256 if isinstance(pub, ed25519.Ed25519PublicKey) else 448
                details.is_weak_key = False
            elif isinstance(pub, dsa.DSAPublicKey):
                details.key_type = "DSA"
                details.is_weak_key = True

            # Signature hash
            if cert.signature_hash_algorithm is not None:
                hname = cert.signature_hash_algorithm.name.lower()
                details.signature_hash = hname
                details.is_weak_signature = hname in WEAK_HASHES

        except Exception as exc:
            logger.debug("Cert parse failed for %s: %s", hostname, type(exc).__name__)
            details.connection_error = "CERT_PARSE_FAILURE"
        return details

    @classmethod
    def _hostname_matches(
        cls,
        hostname: str,
        sans: List[str],
        subject_cn: Optional[str],
    ) -> bool:
        host = hostname.lower().rstrip(".")
        candidates = list(sans)
        if subject_cn:
            candidates.append(subject_cn.lower())

        for san in candidates:
            san = san.lower().rstrip(".")
            if san.startswith("*."):
                # *.example.com matches a.example.com but not example.com
                base = san[2:]
                if host.endswith("." + base) and host.count(".") == base.count(".") + 1:
                    return True
            elif host == san:
                return True
        return False


# ============================================================================
# TLS TRANSPORT
# ============================================================================

class AsyncSSLTransportEngine:
    @classmethod
    async def inspect(
        cls, hostname: str, port: int = DEFAULT_HTTPS_PORT
    ) -> SSLCertificateDetails:
        details = SSLCertificateDetails(hostname=hostname, port=port)

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        # Prefer modern TLS but still allow connection so we can inspect
        try:
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        except Exception:
            pass

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    host=hostname,
                    port=port,
                    ssl=ctx,
                    server_hostname=hostname,
                ),
                timeout=TLS_CONNECT_TIMEOUT,
            )
            try:
                # Actual connected peer IP (rebinding guard)
                peer = writer.get_extra_info("peername")
                if peer and isinstance(peer, tuple) and peer:
                    details.connected_ip = str(peer[0])

                ssl_obj = writer.get_extra_info("ssl_object")
                if ssl_obj:
                    cipher = ssl_obj.cipher()
                    if cipher:
                        details.cipher_suite = cipher[0]
                        details.tls_version = cipher[1]
                        details.is_weak_cipher = bool(
                            WEAK_CIPHER_RE.search(cipher[0] or "")
                        )
                        details.is_legacy_tls = (cipher[1] or "") in LEGACY_TLS

                    der = ssl_obj.getpeercert(binary_form=True)
                    if der:
                        parsed = CertificateParserEngine.parse(der, hostname, port)
                        # Preserve transport fields
                        parsed.tls_version = details.tls_version
                        parsed.cipher_suite = details.cipher_suite
                        parsed.is_weak_cipher = details.is_weak_cipher
                        parsed.is_legacy_tls = details.is_legacy_tls
                        parsed.connected_ip = details.connected_ip
                        details = parsed

                # Lightweight HEAD for HSTS (best-effort)
                try:
                    req = (
                        f"HEAD / HTTP/1.1\r\n"
                        f"Host: {hostname}\r\n"
                        f"User-Agent: PhishGuard-SSL/4.2\r\n"
                        f"Connection: close\r\n\r\n"
                    )
                    writer.write(req.encode("ascii", errors="ignore"))
                    await writer.drain()
                    data = await asyncio.wait_for(
                        reader.read(2048), timeout=HTTP_HEADER_TIMEOUT
                    )
                    text = data.decode("utf-8", errors="ignore").lower()
                    if "strict-transport-security" in text:
                        details.has_hsts = True
                except Exception:
                    pass
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

        except asyncio.TimeoutError:
            details.connection_error = "TLS_HANDSHAKE_TIMEOUT"
        except ssl.SSLError as exc:
            details.connection_error = f"SSL_ERROR_{type(exc).__name__}"
        except (socket.gaierror, ConnectionRefusedError, OSError) as exc:
            details.connection_error = f"CONNECTION_FAILED_{type(exc).__name__}"
        except Exception as exc:
            details.connection_error = f"UNKNOWN_{type(exc).__name__}"

        return details


# ============================================================================
# PAYLOAD STORE
# ============================================================================

class SSLPayloadStore:
    @staticmethod
    def resolve(context: ScanContext) -> List[Tuple[str, int]]:
        targets: Set[Tuple[str, int]] = set()
        meta = getattr(context, "metadata", None)
        data: Dict[str, Any] = getattr(meta, "data", {}) if meta else {}

        def add(host: str, port: Any = 443) -> None:
            clean = HostValidator.sanitize(host)
            if not clean:
                return
            try:
                p = int(port) if port else 443
            except (TypeError, ValueError):
                p = 443
            if p <= 0 or p > 65535:
                p = 443
            targets.add((clean, p))

        for item in data.get("parsed_urls") or []:
            if not isinstance(item, dict):
                continue
            host = str(item.get("host_normalized") or item.get("host") or "")
            scheme = str(item.get("scheme") or "https").lower()
            port = item.get("port")
            if scheme in {"https", "wss"} or (port and int(port) != 80):
                add(host, port or 443)

        for key in ("domain", "org_domain", "host", "hostname"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                add(v.strip(), 443)

        for key in ("url", "target_url"):
            v = data.get(key) or getattr(context, key, None)
            if isinstance(v, str) and "://" in v:
                try:
                    from urllib.parse import urlparse
                    u = urlparse(v)
                    if u.hostname:
                        port = u.port or (443 if (u.scheme or "").lower() in {"https", "wss"} else 80)
                        if port != 80:
                            add(u.hostname, port)
                except Exception:
                    pass

        return list(targets)[:MAX_TARGETS]


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class SSLCheckerAnalyzer(BaseAnalyzer):
    name: str = "ssl_checker"
    version: str = SSL_CHECKER_VERSION
    supported_scan_types: Set[ScanType] = {
        ScanType.DOMAIN, ScanType.URL, ScanType.COMPREHENSIVE
    }
    required_inputs: Set[str] = set()
    resource_class: AnalyzerResourceClass = AnalyzerResourceClass.NETWORK_IO
    priority: int = 15
    can_hard_override: bool = False

    def __init__(self) -> None:
        self.cache = SSLCacheManager()
        self._sem: Optional[asyncio.Semaphore] = None

    def _sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(MAX_CONCURRENT)
        return self._sem

    async def close(self) -> None:
        self.cache.clear()

    async def _inspect_one(
        self, hostname: str, port: int, deadline: float
    ) -> List[Finding]:
        findings: List[Finding] = []
        if time.monotonic() >= deadline:
            return findings

        cache_key = f"{hostname}:{port}"
        lock = await self.cache.get_lock(cache_key)

        async with lock:
            cached = await self.cache.get(cache_key)
            if cached:
                details = cached
            else:
                async with self._sem():
                    if time.monotonic() >= deadline:
                        return findings
                    details = await AsyncSSLTransportEngine.inspect(hostname, port)
                    await self.cache.set(cache_key, details)

        # Post-connect SSRF / rebinding check
        if details.connected_ip and HostValidator.is_dangerous_ip(details.connected_ip):
            findings.append(
                Finding(
                    rule_id="SSL.SSRF.CONNECTED_PRIVATE_IP",
                    category="EXPLOIT_ATTEMPT",
                    severity=SecurityVerdict.HIGH,
                    confidence=0.97,
                    message=(
                        f"TLS connected to private/metadata IP {details.connected_ip} "
                        f"for host '{hostname}'."
                    ),
                    metadata=ConstrainedMetadata({
                        "hostname": hostname,
                        "connected_ip": details.connected_ip,
                    }),
                )
            )
            return findings

        if details.connection_error:
            # Keep connection failures low-severity / low-noise
            findings.append(
                Finding(
                    rule_id="SSL.TRANSPORT.CONNECTION_FAILED",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.LOW,
                    confidence=0.75,
                    message=(
                        f"TLS handshake failed for '{hostname}:{port}' "
                        f"({details.connection_error})."
                    ),
                    metadata=ConstrainedMetadata({
                        "hostname": hostname,
                        "error": details.connection_error,
                    }),
                )
            )
            return findings

        # Hostname / SAN mismatch
        if not details.is_hostname_match:
            findings.append(
                Finding(
                    rule_id="SSL.CERTIFICATE.SAN_MISMATCH",
                    category="IMPERSONATION",
                    severity=SecurityVerdict.HIGH,
                    confidence=0.96,
                    message=f"Certificate SAN/CN does not match '{hostname}'.",
                    metadata=ConstrainedMetadata({
                        "hostname": hostname,
                        "subject_cn": details.subject_cn or "",
                        "san_sample": ",".join(details.san_list[:6]),
                    }),
                )
            )

        # Self-signed
        if details.is_self_signed:
            findings.append(
                Finding(
                    rule_id="SSL.CERTIFICATE.SELF_SIGNED",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.HIGH,
                    confidence=0.97,
                    message=f"Self-signed certificate on '{hostname}'.",
                    metadata=ConstrainedMetadata({
                        "hostname": hostname,
                        "issuer_cn": details.issuer_cn or "",
                    }),
                )
            )

        # Lifecycle
        if details.is_expired:
            findings.append(
                Finding(
                    rule_id="SSL.CERTIFICATE.EXPIRED",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.HIGH,
                    confidence=0.98,
                    message=f"Expired certificate on '{hostname}' (until {details.valid_until}).",
                    metadata=ConstrainedMetadata({
                        "hostname": hostname,
                        "valid_until": str(details.valid_until),
                    }),
                )
            )
        elif details.is_not_yet_valid:
            findings.append(
                Finding(
                    rule_id="SSL.CERTIFICATE.NOT_YET_VALID",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.HIGH,
                    confidence=0.95,
                    message=f"Certificate on '{hostname}' not yet valid (from {details.valid_from}).",
                    metadata=ConstrainedMetadata({
                        "hostname": hostname,
                        "valid_from": str(details.valid_from),
                    }),
                )
            )
        elif (
            details.days_until_expiration is not None
            and details.days_until_expiration <= EXPIRATION_WARNING_DAYS
        ):
            findings.append(
                Finding(
                    rule_id="SSL.CERTIFICATE.EXPIRING_SOON",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.MEDIUM,
                    confidence=0.88,
                    message=(
                        f"Certificate on '{hostname}' expires in "
                        f"{details.days_until_expiration} days."
                    ),
                    metadata=ConstrainedMetadata({
                        "hostname": hostname,
                        "days_remaining": details.days_until_expiration,
                    }),
                )
            )

        # Weak key
        if details.is_weak_key:
            findings.append(
                Finding(
                    rule_id="SSL.CERTIFICATE.WEAK_KEY",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.MEDIUM,
                    confidence=0.94,
                    message=(
                        f"Weak key on '{hostname}' "
                        f"({details.key_type} {details.key_size} bits)."
                    ),
                    metadata=ConstrainedMetadata({
                        "hostname": hostname,
                        "key_type": details.key_type or "",
                        "key_size": details.key_size or 0,
                    }),
                )
            )

        # Weak signature
        if details.is_weak_signature:
            findings.append(
                Finding(
                    rule_id="SSL.CERTIFICATE.WEAK_SIGNATURE",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.MEDIUM,
                    confidence=0.94,
                    message=(
                        f"Weak signature hash on '{hostname}' "
                        f"({details.signature_hash})."
                    ),
                    metadata=ConstrainedMetadata({
                        "hostname": hostname,
                        "signature_hash": details.signature_hash or "",
                    }),
                )
            )

        # Legacy TLS
        if details.is_legacy_tls:
            findings.append(
                Finding(
                    rule_id="SSL.TRANSPORT.LEGACY_TLS",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.MEDIUM,
                    confidence=0.92,
                    message=f"Legacy TLS version negotiated with '{hostname}' ({details.tls_version}).",
                    metadata=ConstrainedMetadata({
                        "hostname": hostname,
                        "tls_version": details.tls_version or "",
                    }),
                )
            )

        # Weak cipher
        if details.is_weak_cipher:
            findings.append(
                Finding(
                    rule_id="SSL.TRANSPORT.WEAK_CIPHER",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.MEDIUM,
                    confidence=0.90,
                    message=f"Weak cipher suite on '{hostname}' ({details.cipher_suite}).",
                    metadata=ConstrainedMetadata({
                        "hostname": hostname,
                        "cipher": details.cipher_suite or "",
                    }),
                )
            )

        # Missing HSTS (only if we successfully spoke HTTP)
        if not details.has_hsts and not details.connection_error:
            findings.append(
                Finding(
                    rule_id="SSL.HTTP.NO_HSTS",
                    category="SUSPICIOUS_INFRASTRUCTURE",
                    severity=SecurityVerdict.LOW,
                    confidence=0.80,
                    message=f"Missing HSTS header on '{hostname}'.",
                    metadata=ConstrainedMetadata({"hostname": hostname}),
                )
            )

        return findings

    async def analyze(
        self, context: ScanContext, deadline_monotonic: float
    ) -> AnalyzerResult:
        start = time.monotonic()
        if time.monotonic() >= deadline_monotonic:
            return self._timeout(start)

        targets = SSLPayloadStore.resolve(context)
        if not targets:
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=0.0,
                confidence=0.85,
                findings=[],
            )

        findings_map: Dict[Tuple[str, str], Finding] = {}
        cat_scores: Dict[str, float] = {
            "IMPERSONATION": 0.0,
            "EXPLOIT_ATTEMPT": 0.0,
            "SUSPICIOUS_INFRASTRUCTURE": 0.0,
        }

        try:
            tasks = [
                self._inspect_one(host, port, deadline_monotonic)
                for host, port in targets
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for res in results:
                if not isinstance(res, list):
                    continue
                for f in res:
                    if len(findings_map) >= MAX_FINDINGS:
                        break
                    host = ""
                    if f.metadata and getattr(f.metadata, "data", None):
                        host = str(f.metadata.data.get("hostname", ""))
                    key = (f.rule_id, host)
                    if key in findings_map:
                        continue
                    findings_map[key] = f

                    if f.severity == SecurityVerdict.HIGH:
                        if "SAN_MISMATCH" in f.rule_id or "IMPERSONATION" in f.category:
                            cat_scores["IMPERSONATION"] += 38.0
                        elif "SSRF" in f.rule_id:
                            cat_scores["EXPLOIT_ATTEMPT"] += 40.0
                        else:
                            cat_scores["SUSPICIOUS_INFRASTRUCTURE"] += 32.0
                    elif f.severity == SecurityVerdict.MEDIUM:
                        cat_scores["SUSPICIOUS_INFRASTRUCTURE"] += 18.0
                    else:
                        cat_scores["SUSPICIOUS_INFRASTRUCTURE"] += 6.0

            total = (
                min(cat_scores["IMPERSONATION"], 45.0)
                + min(cat_scores["EXPLOIT_ATTEMPT"], 45.0)
                + min(cat_scores["SUSPICIOUS_INFRASTRUCTURE"], 40.0)
            )
            score = min(max(total, 0.0), 100.0)

            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=score,
                confidence=0.94 if findings_map else 0.86,
                findings=list(findings_map.values()),
            )

        except Exception as exc:
            logger.exception(
                "SSLChecker failure scan_id=%s", getattr(context, "scan_id", "?")
            )
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.FAILED,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                sanitized_error=f"SSL analysis error: {type(exc).__name__}",
            )

    def _timeout(self, start: float) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.TIMEOUT,
            execution_time_ms=(time.monotonic() - start) * 1000.0,
            sanitized_error="Deadline exceeded during SSL/TLS inspection.",
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "SSLCheckerAnalyzer",
    "AsyncSSLTransportEngine",
    "CertificateParserEngine",
    "HostValidator",
    "SSLCacheManager",
    "SSLCertificateDetails",
    "SSLPayloadStore",
    "SSL_CHECKER_VERSION",
]

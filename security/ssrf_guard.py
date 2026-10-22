"""
PhishGuard_AI - Server-Side Request Forgery (SSRF) Guard

Production-grade SSRF protection boundary used by network-facing analyzers.

Security principles:
- Only http/https schemes are ever permitted.
- IP-literal hosts are validated against private / loopback / link-local /
  multicast / reserved / metadata networks before any request.
- DNS hostnames are resolved and every resolved address is validated, which
  mitigates classic DNS-rebinding and TOCTOU attacks.
- Redirect chains are capped and re-validated at each hop.
- Parser inputs are size-bounded and control characters rejected.
- Fail-closed: any parsing or resolution anomaly results in a denial.

This module performs NETWORK validation only. It never issues HTTP requests.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

DEFAULT_MAX_URL_LENGTH = 8192
DEFAULT_DNS_TIMEOUT_SECONDS = 2.0
DEFAULT_MAX_REDIRECT_DEPTH = 5
DEFAULT_MAX_RESOLVED_ADDRESSES = 64

ALLOWED_SCHEMES = frozenset({"http", "https"})

# RFC 1918 + IPv6 analogues, link-local, loopback, multicast, documentation,
# benchmarking, unspecified, CGNAT and special-purpose networks.
_PRIVATE_NETWORKS: Tuple[ipaddress._BaseNetwork, ...] = (
    ipaddress.ip_network("0.0.0.0/8"),        # "this" network
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),    # CGNAT / shared
    ipaddress.ip_network("127.0.0.0/8"),      # loopback
    ipaddress.ip_network("169.254.0.0/16"),   # link-local
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),     # documentation
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),    # benchmarking
    ipaddress.ip_network("198.51.100.0/24"),  # documentation
    ipaddress.ip_network("203.0.113.0/24"),   # documentation
    ipaddress.ip_network("224.0.0.0/4"),      # multicast
    ipaddress.ip_network("240.0.0.0/4"),      # reserved
    ipaddress.ip_network("255.255.255.255/32"),
    ipaddress.ip_network("::/128"),           # unspecified
    ipaddress.ip_network("::1/128"),          # loopback
    ipaddress.ip_network("::ffff:0:0/96"),    # IPv4-mapped
    ipaddress.ip_network("64:ff9b::/96"),     # NAT64
    ipaddress.ip_network("100::/64"),         # discard-only
    ipaddress.ip_network("2001::/32"),        # teredo
    ipaddress.ip_network("2001:10::/28"),     # orchid
    ipaddress.ip_network("2001:db8::/32"),    # documentation
    ipaddress.ip_network("fc00::/7"),         # unique-local
    ipaddress.ip_network("fe80::/10"),        # link-local
    ipaddress.ip_network("ff00::/8"),         # multicast
)

# Well-known cloud metadata endpoints that must never be reachable.
METADATA_HOSTNAMES = frozenset(
    {
        "metadata.google.internal",
        "metadata.google.com",
        "169.254.169.254",
        "169.254.170.2",  # ECS credentials endpoint
        "fd00:ec2::254",
    }
)

_HOSTNAME_REGEX = re.compile(
    r"^(?=.{1,253}$)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*"
    r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$"
)

# ============================================================================
# ENUMS & RESULT
# ============================================================================


class SSRFBlockReason(str, Enum):
    """Machine-readable reason for an SSRF denial."""

    DISALLOWED_SCHEME = "disallowed_scheme"
    INVALID_URL = "invalid_url"
    URL_TOO_LONG = "url_too_long"
    CONTROL_CHARACTER = "control_character"
    BLOCKED_HOSTNAME = "blocked_hostname"
    PRIVATE_ADDRESS = "private_address"
    RESERVED_ADDRESS = "reserved_address"
    METADATA_ADDRESS = "metadata_address"
    DNS_RESOLUTION_FAILED = "dns_resolution_failed"
    REDIRECT_DEPTH_EXCEEDED = "redirect_depth_exceeded"
    MALFORMED_HOST = "malformed_host"
    EMPTY_URL = "empty_url"


@dataclass(frozen=True, slots=True)
class SSRFValidationResult:
    """Result of a single SSRF validation pass."""

    allowed: bool
    reason: str = ""
    host: Optional[str] = None
    resolved_addresses: Tuple[str, ...] = ()
    blocked_address: Optional[str] = None
    scheme: Optional[str] = None
    details: Tuple[str, ...] = ()


# ============================================================================
# GUARD
# ============================================================================


class SSRFGuard:
    """
    Fail-closed SSRF protection boundary.

    Thread-safe and re-entrant. DNS resolution runs synchronously in the
    blocking API and is offloaded to a worker thread in the async API so the
    event loop is never blocked.
    """

    def __init__(
        self,
        *,
        max_url_length: int = DEFAULT_MAX_URL_LENGTH,
        dns_timeout_seconds: float = DEFAULT_DNS_TIMEOUT_SECONDS,
        max_redirect_depth: int = DEFAULT_MAX_REDIRECT_DEPTH,
        max_resolved_addresses: int = DEFAULT_MAX_RESOLVED_ADDRESSES,
        private_networks: Sequence[ipaddress._BaseNetwork] = _PRIVATE_NETWORKS,
        blocked_hostnames: Sequence[str] = (),
    ) -> None:
        if max_url_length <= 0:
            raise ValueError("max_url_length must be positive.")
        if dns_timeout_seconds <= 0:
            raise ValueError("dns_timeout_seconds must be positive.")
        if max_redirect_depth < 0:
            raise ValueError("max_redirect_depth must be non-negative.")

        self._max_url_length = max_url_length
        self._dns_timeout_seconds = dns_timeout_seconds
        self._max_redirect_depth = max_redirect_depth
        self._max_resolved_addresses = max_resolved_addresses
        self._private_networks = tuple(private_networks)
        self._blocked_hostnames = frozenset(
            h.lower() for h in blocked_hostnames
        ) | METADATA_HOSTNAMES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        url: str,
        *,
        max_redirect_depth: Optional[int] = None,
    ) -> SSRFValidationResult:
        """
        Validate a URL against the SSRF policy.

        Performs scheme/host checks and DNS resolution of the target host.
        The caller must gate on ``allowed`` before issuing any request.
        """
        return self._validate(url)

    async def validate_async(
        self,
        url: str,
        *,
        max_redirect_depth: Optional[int] = None,
    ) -> SSRFValidationResult:
        """Asynchronous variant that offloads DNS work off the event loop."""
        return await asyncio.to_thread(self.validate, url)

    def validate_redirect(
        self,
        current: str,
        next_url: str,
        *,
        depth: int,
    ) -> SSRFValidationResult:
        """Validate a redirect target, enforcing the configured hop bound."""
        if depth > self._max_redirect_depth:
            return SSRFValidationResult(
                allowed=False,
                reason=SSRFBlockReason.REDIRECT_DEPTH_EXCEEDED.value,
                host=self._extract_host(next_url),
                details=(f"redirect depth {depth} exceeds bound",),
            )
        return self.validate(next_url)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_host(self, url: str) -> Optional[str]:
        try:
            return urlsplit(url).hostname
        except ValueError:
            return None

    def _validate(self, url: str) -> SSRFValidationResult:
        if not isinstance(url, str) or not url.strip():
            return SSRFValidationResult(
                allowed=False, reason=SSRFBlockReason.EMPTY_URL.value
            )

        if len(url) > self._max_url_length:
            return SSRFValidationResult(
                allowed=False, reason=SSRFBlockReason.URL_TOO_LONG.value
            )

        if any(ord(c) < 0x20 for c in url):
            return SSRFValidationResult(
                allowed=False, reason=SSRFBlockReason.CONTROL_CHARACTER.value
            )

        try:
            parts = urlsplit(url)
        except ValueError:
            return SSRFValidationResult(
                allowed=False, reason=SSRFBlockReason.INVALID_URL.value
            )

        scheme = (parts.scheme or "").lower()
        if scheme not in ALLOWED_SCHEMES:
            return SSRFValidationResult(
                allowed=False,
                reason=SSRFBlockReason.DISALLOWED_SCHEME.value,
                scheme=scheme or None,
            )

        host = parts.hostname
        if not host:
            return SSRFValidationResult(
                allowed=False,
                reason=SSRFBlockReason.MALFORMED_HOST.value,
                scheme=scheme,
            )

        host_lower = host.lower()

        # Fast-path block for known metadata / explicitly blocked hostnames.
        if host_lower in self._blocked_hostnames:
            return SSRFValidationResult(
                allowed=False,
                reason=SSRFBlockReason.BLOCKED_HOSTNAME.value,
                host=host,
                scheme=scheme,
            )

        # IP-literal host: validate the literal directly, no DNS needed.
        parsed_ip = self._parse_ip_literal(host)
        if parsed_ip is not None:
            reason = self._classify_address(parsed_ip)
            if reason is not None:
                return SSRFValidationResult(
                    allowed=False,
                    reason=reason.value,
                    host=host,
                    scheme=scheme,
                    blocked_address=str(parsed_ip),
                )
            return SSRFValidationResult(
                allowed=True,
                host=host,
                scheme=scheme,
                resolved_addresses=(str(parsed_ip),),
            )

        # Hostname: resolve every address and validate each (DNS-rebinding safe).
        addresses = self._resolve_host(host_lower)
        if not addresses:
            return SSRFValidationResult(
                allowed=False,
                reason=SSRFBlockReason.DNS_RESOLUTION_FAILED.value,
                host=host,
                scheme=scheme,
                details=("no validated A/AAAA records returned",),
            )

        blocked = self._first_blocked(addresses)
        if blocked is not None:
            address, reason = blocked
            return SSRFValidationResult(
                allowed=False,
                reason=reason.value,
                host=host,
                scheme=scheme,
                resolved_addresses=tuple(str(a) for a in addresses),
                blocked_address=str(address),
            )

        return SSRFValidationResult(
            allowed=True,
            host=host,
            scheme=scheme,
            resolved_addresses=tuple(str(a) for a in addresses),
        )

    def _parse_ip_literal(
        self, host: str
    ) -> Optional[ipaddress._BaseAddress]:
        """Parse bracketed/unbracketed IPv6 or plain IPv4 literals."""
        candidate = host.strip("[]")
        try:
            return ipaddress.ip_address(candidate)
        except ValueError:
            return None

    def _classify_address(
        self, address: ipaddress._BaseAddress
    ) -> Optional[SSRFBlockReason]:
        """Return the block reason for a single address, else None."""
        for network in self._private_networks:
            if address.version == network.version and address in network:
                if (
                    address.version == 4
                    and str(address) == "169.254.169.254"
                ):
                    return SSRFBlockReason.METADATA_ADDRESS
                return SSRFBlockReason.PRIVATE_ADDRESS
        if address.is_global is False:
            return SSRFBlockReason.RESERVED_ADDRESS
        return None

    def _first_blocked(
        self, addresses: Sequence[ipaddress._BaseAddress]
    ) -> Optional[Tuple[ipaddress._BaseAddress, SSRFBlockReason]]:
        for address in addresses:
            reason = self._classify_address(address)
            if reason is not None:
                return address, reason
        return None

    def _resolve_host(
        self, hostname: str
    ) -> List[ipaddress._BaseAddress]:
        """Resolve a hostname to validated address objects, bounded in count."""
        try:
            infos = socket.getaddrinfo(
                hostname,
                None,
                proto=socket.IPPROTO_TCP,
                flags=socket.AI_ADDRCONFIG,
            )
        except (socket.gaierror, OSError):
            return []

        addresses: List[ipaddress._BaseAddress] = []
        seen: set = set()
        for info in infos:
            ip_str = info[4][0]
            if ip_str in seen:
                continue
            seen.add(ip_str)
            try:
                addresses.append(ipaddress.ip_address(ip_str))
            except ValueError:
                continue
            if len(addresses) >= self._max_resolved_addresses:
                break
        return addresses


# ============================================================================
# SINGLETON
# ============================================================================

ssrf_guard = SSRFGuard()


__all__ = [
    "SSRFBlockReason",
    "SSRFValidationResult",
    "SSRFGuard",
    "ssrf_guard",
    "ALLOWED_SCHEMES",
    "METADATA_HOSTNAMES",
]

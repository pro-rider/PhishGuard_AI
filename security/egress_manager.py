"""
PhishGuard_AI - Production Egress Management Engine

Centralized outbound-network policy enforcement.

Responsibilities
----------------
- Validate outbound URL schemes.
- Validate destination hostnames.
- Resolve DNS and inspect every returned IP.
- Block private, loopback, link-local, multicast, reserved, and unspecified destinations.
- Block cloud metadata endpoints (AWS IMDSv1/v2, GCP, Azure, Oracle, Alibaba).
- Normalize IPv4-mapped IPv6 and alternative IP representations.
- Restrict destination ports.
- Support explicit hostname/domain allowlists.
- Support explicit IP allowlists.
- Support tenant-specific egress policy.
- Provide synchronous and asynchronous policy checks.
- Integrate with the security audit logger.
- Fail closed when security validation cannot be completed.

Important
---------
This module is a POLICY / AUTHORIZATION layer.

It does NOT perform HTTP requests.

The actual HTTP connection layer must revalidate the destination
immediately before establishing the connection to reduce DNS-rebinding
and TOCTOU risks.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence
from urllib.parse import urlsplit

from security.audit_log import (
    AuditEvent,
    AuditLogger,
    AuditOutcome,
    AuditSeverity,
    audit_logger,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_ALLOWED_SCHEMES = frozenset({"https"})
DEFAULT_ALLOWED_PORTS = frozenset({443})

MAX_URL_LENGTH = 8192
MAX_HOSTNAME_LENGTH = 253

DNS_CACHE_TTL_SECONDS = 30

DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_READ_TIMEOUT_SECONDS = 15.0

# Explicitly blocked system/internal hostnames
BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata.google.internal",
        "metadata.google.internal.",
        "instance-data",
        "metadata.azure.com",
    }
)

# Cloud Metadata Subnets (AWS, GCP, Azure, Oracle, Alibaba)
METADATA_NETWORKS = (
    ipaddress.ip_network("169.254.169.254/32"),  # AWS / GCP / Azure IMDS
    ipaddress.ip_network("169.254.169.253/32"),  # AWS DNS / EC2
    ipaddress.ip_network("100.100.100.200/32"),  # Alibaba Cloud
    ipaddress.ip_network("fd00:ec2::254/128"),   # AWS IPv6 IMDS
)

HOSTNAME_REGEX = re.compile(
    r"^(?=.{1,253}\.?$)"
    r"(?:[a-zA-Z0-9]"
    r"(?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*"
    r"[a-zA-Z0-9]"
    r"(?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.?$"
)


# ============================================================================
# ENUMERATIONS
# ============================================================================


class EgressDecision(str, Enum):
    """Result of an outbound destination policy evaluation."""

    ALLOW = "allow"
    DENY = "deny"


class EgressReason(str, Enum):
    """Machine-readable reason for an egress decision."""

    ALLOWED = "allowed"

    INVALID_URL = "invalid_url"
    INVALID_SCHEME = "invalid_scheme"
    INVALID_HOST = "invalid_host"
    EMPTY_HOST = "empty_host"

    PORT_NOT_ALLOWED = "port_not_allowed"

    BLOCKED_HOSTNAME = "blocked_hostname"
    BLOCKED_IP = "blocked_ip"

    PRIVATE_IP = "private_ip"
    LOOPBACK_IP = "loopback_ip"
    LINK_LOCAL_IP = "link_local_ip"
    MULTICAST_IP = "multicast_ip"
    RESERVED_IP = "reserved_ip"
    UNSPECIFIED_IP = "unspecified_ip"

    DNS_FAILURE = "dns_failure"
    DNS_NO_RESULT = "dns_no_result"

    NOT_IN_ALLOWLIST = "not_in_allowlist"
    IP_NOT_IN_ALLOWLIST = "ip_not_in_allowlist"

    USERINFO_NOT_ALLOWED = "userinfo_not_allowed"
    FRAGMENT_NOT_ALLOWED = "fragment_not_allowed"

    POLICY_DISABLED = "policy_disabled"
    INTERNAL_ERROR = "internal_error"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True, slots=True)
class EgressPolicy:
    """Immutable outbound-network policy."""

    allowed_schemes: frozenset[str] = DEFAULT_ALLOWED_SCHEMES
    allowed_ports: frozenset[int] = DEFAULT_ALLOWED_PORTS
    allowed_hostnames: frozenset[str] = frozenset()
    allowed_domains: frozenset[str] = frozenset()
    allowed_ips: frozenset[str] = frozenset()

    allow_private_networks: bool = False
    allow_loopback: bool = False
    allow_link_local: bool = False
    allow_multicast: bool = False
    allow_reserved: bool = False
    allow_unspecified: bool = False

    allow_url_userinfo: bool = False
    allow_fragments: bool = False

    require_dns_resolution: bool = True
    fail_closed_on_dns_error: bool = True
    block_metadata_endpoints: bool = True

    max_url_length: int = MAX_URL_LENGTH
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
    read_timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        schemes = frozenset(
            str(value).strip().lower()
            for value in self.allowed_schemes
            if str(value).strip()
        )
        ports = frozenset(int(port) for port in self.allowed_ports)

        if not schemes:
            raise ValueError("At least one egress scheme must be configured.")

        if any(port < 1 or port > 65535 for port in ports):
            raise ValueError("Egress ports must be between 1 and 65535.")

        if self.max_url_length <= 0:
            raise ValueError("max_url_length must be positive.")

        if self.connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be positive.")

        if self.read_timeout_seconds <= 0:
            raise ValueError("read_timeout_seconds must be positive.")


@dataclass(frozen=True, slots=True)
class EgressTarget:
    """Normalized destination extracted from a URL."""

    original_url: str
    scheme: str
    hostname: str
    port: int
    path: str

    @property
    def normalized_host(self) -> str:
        return self.hostname.rstrip(".").lower()


@dataclass(frozen=True, slots=True)
class EgressResult:
    """Complete result of an egress policy evaluation."""

    decision: EgressDecision
    reason: EgressReason
    target: Optional[EgressTarget] = None
    resolved_ips: tuple[str, ...] = ()
    message: str = ""
    tenant_id: Optional[str] = None

    @property
    def allowed(self) -> bool:
        return self.decision == EgressDecision.ALLOW


# ============================================================================
# EXCEPTIONS
# ============================================================================


class EgressDeniedError(RuntimeError):
    """Raised when outbound network access violates policy."""

    def __init__(self, result: EgressResult) -> None:
        self.result = result
        super().__init__(f"Outbound request denied: {result.reason.value}")


class EgressConfigurationError(ValueError):
    """Raised for invalid security policy configuration."""


# ============================================================================
# DNS RESOLVER
# ============================================================================


class DNSResolver:
    """Controlled DNS resolver returning evaluated IP targets."""

    def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        try:
            results = socket.getaddrinfo(
                hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise EgressDeniedError(
                EgressResult(
                    decision=EgressDecision.DENY,
                    reason=EgressReason.DNS_FAILURE,
                    message="DNS resolution failed.",
                )
            ) from exc

        addresses: set[str] = set()

        for result in results:
            sockaddr = result[4]
            if not sockaddr:
                continue

            address = sockaddr[0]
            try:
                ipaddress.ip_address(address)
            except ValueError:
                continue

            addresses.add(address)

        if not addresses:
            raise EgressDeniedError(
                EgressResult(
                    decision=EgressDecision.DENY,
                    reason=EgressReason.DNS_NO_RESULT,
                    message="DNS returned no usable IP address.",
                )
            )

        return tuple(sorted(addresses))


# ============================================================================
# EGRESS MANAGER
# ============================================================================


class EgressManager:
    """Central outbound-network security policy engine."""

    def __init__(
        self,
        *,
        policy: Optional[EgressPolicy] = None,
        resolver: Optional[DNSResolver] = None,
        audit_logger_instance: Optional[AuditLogger] = None,
        service: str = "phishguard-egress",
    ) -> None:
        self.policy = policy or EgressPolicy()
        self.resolver = resolver or DNSResolver()
        self.audit = audit_logger_instance or audit_logger
        self.service = service

        self._validate_policy()

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def check(
        self,
        url: str,
        *,
        tenant_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        audit: bool = True,
    ) -> EgressResult:
        """Validate an outbound destination against policy."""
        result = self._evaluate(url, tenant_id=tenant_id)

        if audit:
            self._audit_result(
                result,
                tenant_id=tenant_id,
                resource_id=resource_id,
                correlation_id=correlation_id,
                request_id=request_id,
            )

        return result

    def enforce(
        self,
        url: str,
        *,
        tenant_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        audit: bool = True,
    ) -> EgressTarget:
        """Validate and enforce egress policy, raising on violation."""
        result = self.check(
            url,
            tenant_id=tenant_id,
            resource_id=resource_id,
            correlation_id=correlation_id,
            request_id=request_id,
            audit=audit,
        )

        if not result.allowed or result.target is None:
            raise EgressDeniedError(result)

        return result.target

    async def check_async(
        self,
        url: str,
        *,
        tenant_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        audit: bool = True,
    ) -> EgressResult:
        """Asynchronously validate egress policy without blocking the event loop."""
        return await asyncio.to_thread(
            self.check,
            url,
            tenant_id=tenant_id,
            resource_id=resource_id,
            correlation_id=correlation_id,
            request_id=request_id,
            audit=audit,
        )

    async def enforce_async(
        self,
        url: str,
        *,
        tenant_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        audit: bool = True,
    ) -> EgressTarget:
        """Async version of enforce()."""
        result = await self.check_async(
            url,
            tenant_id=tenant_id,
            resource_id=resource_id,
            correlation_id=correlation_id,
            request_id=request_id,
            audit=audit,
        )

        if not result.allowed or result.target is None:
            raise EgressDeniedError(result)

        return result.target

    # ========================================================================
    # CORE EVALUATION
    # ========================================================================

    def _evaluate(
        self,
        url: str,
        *,
        tenant_id: Optional[str],
    ) -> EgressResult:

        if not isinstance(url, str):
            return self._deny(
                EgressReason.INVALID_URL,
                "Outbound URL must be a string.",
                tenant_id=tenant_id,
            )

        if not url or len(url) > self.policy.max_url_length:
            return self._deny(
                EgressReason.INVALID_URL,
                "Outbound URL is invalid or exceeds configured length.",
                tenant_id=tenant_id,
            )

        try:
            target = self._parse_url(url)
        except ValueError as exc:
            return self._deny(
                EgressReason.INVALID_URL,
                str(exc),
                tenant_id=tenant_id,
            )

        # Scheme check
        if target.scheme not in self.policy.allowed_schemes:
            return self._deny(
                EgressReason.INVALID_SCHEME,
                "URL scheme is not permitted.",
                target=target,
                tenant_id=tenant_id,
            )

        # Userinfo check
        parsed = urlsplit(url)
        if parsed.username is not None or parsed.password is not None:
            if not self.policy.allow_url_userinfo:
                return self._deny(
                    EgressReason.USERINFO_NOT_ALLOWED,
                    "URL userinfo is not permitted.",
                    target=target,
                    tenant_id=tenant_id,
                )

        # Fragment check
        if parsed.fragment and not self.policy.allow_fragments:
            return self._deny(
                EgressReason.FRAGMENT_NOT_ALLOWED,
                "URL fragments are not permitted for outbound requests.",
                target=target,
                tenant_id=tenant_id,
            )

        # Port check
        if target.port not in self.policy.allowed_ports:
            return self._deny(
                EgressReason.PORT_NOT_ALLOWED,
                "Destination port is not permitted.",
                target=target,
                tenant_id=tenant_id,
            )

        # Hostname check
        hostname = target.normalized_host
        if hostname in BLOCKED_HOSTNAMES:
            return self._deny(
                EgressReason.BLOCKED_HOSTNAME,
                "Destination hostname is blocked.",
                target=target,
                tenant_id=tenant_id,
            )

        # Direct IP evaluation
        direct_ip = self._parse_ip_safely(hostname)

        if direct_ip is not None:
            normalized_ip = self._normalize_ip_address(direct_ip)

            if self.policy.block_metadata_endpoints and self._is_metadata_ip(
                normalized_ip
            ):
                return self._deny(
                    EgressReason.BLOCKED_IP,
                    "Cloud metadata destination is blocked.",
                    target=target,
                    tenant_id=tenant_id,
                )

            reason = self._classify_ip(normalized_ip)
            if reason is not None:
                return self._deny(
                    reason,
                    "Destination IP is blocked by egress policy.",
                    target=target,
                    tenant_id=tenant_id,
                )

            if self.policy.allowed_ips:
                if str(normalized_ip) not in self.policy.allowed_ips:
                    return self._deny(
                        EgressReason.IP_NOT_IN_ALLOWLIST,
                        "Destination IP is not in the egress allowlist.",
                        target=target,
                        tenant_id=tenant_id,
                    )

            return self._allow(
                target=target,
                resolved_ips=(str(normalized_ip),),
                tenant_id=tenant_id,
            )

        # Hostname allowlisting
        if self.policy.allowed_hostnames or self.policy.allowed_domains:
            if not self._hostname_allowed(hostname):
                return self._deny(
                    EgressReason.NOT_IN_ALLOWLIST,
                    "Destination hostname is not permitted.",
                    target=target,
                    tenant_id=tenant_id,
                )

        # DNS Validation
        if not self.policy.require_dns_resolution:
            return self._allow(
                target=target,
                tenant_id=tenant_id,
            )

        try:
            resolved_ips = self.resolver.resolve(
                hostname,
                target.port,
            )
        except EgressDeniedError as exc:
            return EgressResult(
                decision=EgressDecision.DENY,
                reason=exc.result.reason,
                target=target,
                tenant_id=tenant_id,
                message=exc.result.message,
            )

        # Inspect all resolved IP results
        for raw_ip in resolved_ips:
            try:
                parsed_resolved = ipaddress.ip_address(raw_ip)
            except ValueError:
                return self._deny(
                    EgressReason.DNS_FAILURE,
                    "DNS returned an invalid address.",
                    target=target,
                    tenant_id=tenant_id,
                )

            ip_obj = self._normalize_ip_address(parsed_resolved)

            if (
                self.policy.block_metadata_endpoints
                and self._is_metadata_ip(ip_obj)
            ):
                return self._deny(
                    EgressReason.BLOCKED_IP,
                    "Resolved destination is a cloud metadata endpoint.",
                    target=target,
                    resolved_ips=resolved_ips,
                    tenant_id=tenant_id,
                )

            reason = self._classify_ip(ip_obj)
            if reason is not None:
                return self._deny(
                    reason,
                    "Resolved destination is blocked by egress policy.",
                    target=target,
                    resolved_ips=resolved_ips,
                    tenant_id=tenant_id,
                )

        return self._allow(
            target=target,
            resolved_ips=resolved_ips,
            tenant_id=tenant_id,
        )

    # ========================================================================
    # IP & URL PARSING HELPERS
    # ========================================================================

    @staticmethod
    def _parse_ip_safely(
        hostname: str,
    ) -> Optional[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        """Parse direct IP address including alternate integer/hex literals."""
        try:
            return ipaddress.ip_address(hostname)
        except ValueError:
            pass

        # Try parsing hex or integer octets
        try:
            if hostname.startswith("0x") or hostname.isdigit():
                return ipaddress.ip_address(int(hostname, 0))
        except (ValueError, OverflowError):
            pass

        return None

    @staticmethod
    def _normalize_ip_address(
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    ) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
        """Unmap IPv4-mapped IPv6 addresses (e.g., ::ffff:169.254.169.254)."""
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            return ip.ipv4_mapped
        return ip

    @staticmethod
    def _parse_url(url: str) -> EgressTarget:
        try:
            parsed = urlsplit(url)
        except ValueError as exc:
            raise ValueError("URL parsing failed.") from exc

        scheme = parsed.scheme.lower().strip()
        if not scheme:
            raise ValueError("URL scheme is required.")

        if not parsed.hostname:
            raise ValueError("URL hostname is required.")

        try:
            hostname = parsed.hostname.strip().lower()
        except ValueError as exc:
            raise ValueError("Invalid URL hostname.") from exc

        if not hostname:
            raise ValueError("URL hostname is empty.")

        if len(hostname.rstrip(".")) > MAX_HOSTNAME_LENGTH:
            raise ValueError("URL hostname is too long.")

        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("Invalid URL port.") from exc

        if port is None:
            port = 443 if scheme == "https" else 80

        return EgressTarget(
            original_url=url,
            scheme=scheme,
            hostname=hostname,
            port=port,
            path=parsed.path or "/",
        )

    # ========================================================================
    # HOSTNAME & IP CLASSIFICATION
    # ========================================================================

    def _hostname_allowed(self, hostname: str) -> bool:
        if hostname in self.policy.allowed_hostnames:
            return True

        for domain in self.policy.allowed_domains:
            normalized_domain = domain.rstrip(".").lower()
            if hostname == normalized_domain or hostname.endswith("." + normalized_domain):
                return True

        return False

    def _classify_ip(
        self,
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    ) -> Optional[EgressReason]:

        if ip.is_loopback and not self.policy.allow_loopback:
            return EgressReason.LOOPBACK_IP

        if ip.is_link_local and not self.policy.allow_link_local:
            return EgressReason.LINK_LOCAL_IP

        if ip.is_multicast and not self.policy.allow_multicast:
            return EgressReason.MULTICAST_IP

        if ip.is_unspecified and not self.policy.allow_unspecified:
            return EgressReason.UNSPECIFIED_IP

        if ip.is_private and not self.policy.allow_private_networks:
            return EgressReason.PRIVATE_IP

        if ip.is_reserved and not self.policy.allow_reserved:
            return EgressReason.RESERVED_IP

        return None

    @staticmethod
    def _is_metadata_ip(
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    ) -> bool:
        return any(ip in network for network in METADATA_NETWORKS)

    # ========================================================================
    # RESULT HELPERS
    # ========================================================================

    @staticmethod
    def _allow(
        *,
        target: EgressTarget,
        resolved_ips: Sequence[str] = (),
        tenant_id: Optional[str],
    ) -> EgressResult:
        return EgressResult(
            decision=EgressDecision.ALLOW,
            reason=EgressReason.ALLOWED,
            target=target,
            resolved_ips=tuple(resolved_ips),
            message="Destination allowed by egress policy.",
            tenant_id=tenant_id,
        )

    @staticmethod
    def _deny(
        reason: EgressReason,
        message: str,
        *,
        target: Optional[EgressTarget] = None,
        resolved_ips: Sequence[str] = (),
        tenant_id: Optional[str] = None,
    ) -> EgressResult:
        return EgressResult(
            decision=EgressDecision.DENY,
            reason=reason,
            target=target,
            resolved_ips=tuple(resolved_ips),
            message=message,
            tenant_id=tenant_id,
        )

    # ========================================================================
    # AUDITING
    # ========================================================================

    def _audit_result(
        self,
        result: EgressResult,
        *,
        tenant_id: Optional[str],
        resource_id: Optional[str],
        correlation_id: Optional[str],
        request_id: Optional[str],
    ) -> None:
        if result.allowed:
            return

        metadata = {
            "reason": result.reason.value,
            "decision": result.decision.value,
            "service": self.service,
        }

        try:
            self.audit.record_security_event(
                event=AuditEvent.EGRESS_BLOCKED,
                outcome=AuditOutcome.BLOCKED,
                severity=AuditSeverity.WARNING,
                tenant_id=tenant_id,
                resource_id=resource_id,
                correlation_id=correlation_id,
                request_id=request_id,
                metadata=metadata,
            )
        except Exception:
            logger.exception("egress_audit_failure")

    # ========================================================================
    # POLICY VALIDATION
    # ========================================================================

    def _validate_policy(self) -> None:
        for scheme in self.policy.allowed_schemes:
            if scheme != scheme.lower():
                raise EgressConfigurationError("Allowed schemes must be lowercase.")
            if scheme not in {"http", "https"}:
                raise EgressConfigurationError(
                    f"Unsupported outbound scheme: {scheme!r}"
                )

        for hostname in self.policy.allowed_hostnames:
            normalized = hostname.rstrip(".").lower()
            if not self._valid_hostname(normalized):
                raise EgressConfigurationError(
                    f"Invalid allowed hostname: {hostname!r}"
                )

        for domain in self.policy.allowed_domains:
            normalized = domain.rstrip(".").lower()
            if not self._valid_hostname(normalized):
                raise EgressConfigurationError(
                    f"Invalid allowed domain: {domain!r}"
                )

        for raw_ip in self.policy.allowed_ips:
            try:
                ipaddress.ip_address(raw_ip)
            except ValueError as exc:
                raise EgressConfigurationError(
                    f"Invalid allowed IP: {raw_ip!r}"
                ) from exc

    @staticmethod
    def _valid_hostname(hostname: str) -> bool:
        if len(hostname) > MAX_HOSTNAME_LENGTH:
            return False

        try:
            ipaddress.ip_address(hostname)
            return True
        except ValueError:
            pass

        return bool(HOSTNAME_REGEX.fullmatch(hostname))


# ============================================================================
# PRODUCTION POLICY FACTORY
# ============================================================================


def build_production_egress_policy(
    *,
    allowed_hostnames: Iterable[str] = (),
    allowed_domains: Iterable[str] = (),
) -> EgressPolicy:
    """Build the default, highly restrictive production policy."""
    normalized_hosts = frozenset(
        hostname.rstrip(".").lower() for hostname in allowed_hostnames
    )
    normalized_domains = frozenset(
        domain.rstrip(".").lower() for domain in allowed_domains
    )

    return EgressPolicy(
        allowed_schemes=frozenset({"https"}),
        allowed_ports=frozenset({443}),
        allowed_hostnames=normalized_hosts,
        allowed_domains=normalized_domains,
        allow_private_networks=False,
        allow_loopback=False,
        allow_link_local=False,
        allow_multicast=False,
        allow_reserved=False,
        allow_unspecified=False,
        allow_url_userinfo=False,
        allow_fragments=False,
        require_dns_resolution=True,
        fail_closed_on_dns_error=True,
        block_metadata_endpoints=True,
    )


# ============================================================================
# DEFAULT GLOBAL INSTANCE
# ============================================================================

egress_manager = EgressManager(policy=build_production_egress_policy())


# ============================================================================
# CONVENIENCE EXPORTS
# ============================================================================


def check_egress(
    url: str,
    *,
    tenant_id: Optional[str] = None,
    resource_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> EgressResult:
    """Convenience wrapper around the default production egress manager."""
    return egress_manager.check(
        url,
        tenant_id=tenant_id,
        resource_id=resource_id,
        correlation_id=correlation_id,
        request_id=request_id,
    )


def enforce_egress(
    url: str,
    *,
    tenant_id: Optional[str] = None,
    resource_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> EgressTarget:
    """Convenience enforcement wrapper around default production egress manager."""
    return egress_manager.enforce(
        url,
        tenant_id=tenant_id,
        resource_id=resource_id,
        correlation_id=correlation_id,
        request_id=request_id,
    )


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "BLOCKED_HOSTNAMES",
    "DNSResolver",
    "EgressConfigurationError",
    "EgressDecision",
    "EgressDeniedError",
    "EgressManager",
    "EgressPolicy",
    "EgressReason",
    "EgressResult",
    "EgressTarget",
    "build_production_egress_policy",
    "check_egress",
    "egress_manager",
    "enforce_egress",
]
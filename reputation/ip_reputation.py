"""
PhishGuard AI - IP Reputation & Risk Intelligence Engine
=========================================================

Version: 15.0.0
File Path: reputation/ip_reputation.py

Purpose
-------
Production-grade IP reputation analysis for PhishGuard AI.

This module combines:

    * strict IPv4 / IPv6 canonicalization
    * IPv4-mapped IPv6 normalization
    * special-purpose address classification
    * threat-intelligence aggregation
    * optional RDAP / registration intelligence
    * reverse DNS / PTR analysis
    * forward-confirmed reverse DNS validation
    * PTR hostname quality analysis
    * deterministic bounded scoring
    * independent confidence calculation
    * degraded-state propagation
    * bounded async concurrency
    * per-component concurrency isolation
    * cache + request coalescing
    * cache configuration fingerprinting
    * bounded evidence
    * structured operational metrics
    * dependency injection for testing

Security goals
--------------
- Never treat PTR presence as proof of ownership.
- Never treat private/reserved IP space as malicious by itself.
- Never allow one weak signal to dominate reputation.
- Never pass IPv4-mapped IPv6 text to providers when the embedded IPv4
  is the canonical provider lookup target.
- Never perform arbitrary HTTP requests from this module.
- Never expose provider secrets or raw provider payloads.
- Bound all external evidence.
- Bound DNS concurrency.
- Bound provider concurrency independently.
- Fail closed for malformed input.
- Fail safely for provider failures.
- Keep risk and confidence independent.
- Preserve explainability.

Architecture
------------
                    ┌──────────────────────────────┐
                    │      IP Reputation API       │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
             IP Canonicalizer              Cache / Coalescer
                    │                             │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
        Threat Intel             RDAP               Reverse DNS
        isolated pool        isolated pool          isolated pool
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                                   ▼
                         Signal / Evidence Engine
                                   │
                          ┌────────┴────────┐
                          ▼                 ▼
                       Risk              Confidence
                          │                 │
                          └────────┬────────┘
                                   ▼
                         IPReputationAssessment
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import math
import re
import socket
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from pydantic import BaseModel, ConfigDict, Field, field_validator

from threat_intel.base import IOCType, ThreatSeverity

try:
    from threat_intel.aggregator import (
        AggregatedThreatSummary,
        ThreatIntelAggregator,
    )
except ImportError:  # pragma: no cover
    AggregatedThreatSummary = Any  # type: ignore
    ThreatIntelAggregator = Any  # type: ignore


logger = logging.getLogger("phishguard.reputation.ip")


# =============================================================================
# VERSION / CONSTANTS
# =============================================================================

MODULE_VERSION = "15.0.0"

DEFAULT_CACHE_TTL_SECONDS = 900
DEFAULT_CACHE_MAX_ENTRIES = 10_000

DEFAULT_MAX_EVIDENCE_ITEMS = 64
DEFAULT_MAX_REASONS = 32
DEFAULT_MAX_REASON_LENGTH = 256
DEFAULT_MAX_TAG_LENGTH = 128
DEFAULT_MAX_PTR_HOSTNAMES = 16

DEFAULT_REVERSE_DNS_TIMEOUT_SECONDS = 2.0
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 5.0
DEFAULT_REGISTRATION_TIMEOUT_SECONDS = 5.0
DEFAULT_THREAT_INTEL_TIMEOUT_SECONDS = 8.0

DEFAULT_THREAT_INTEL_CONCURRENCY = 32
DEFAULT_REGISTRATION_CONCURRENCY = 16
DEFAULT_REVERSE_DNS_CONCURRENCY = 32

DEFAULT_NEWLY_ALLOCATED_DAYS = 30
DEFAULT_VERY_NEWLY_ALLOCATED_DAYS = 7

DEFAULT_MAX_HOSTNAME_LENGTH = 253
DEFAULT_MAX_LABEL_LENGTH = 63

# ---------------------------------------------------------------------------
# Risk category weights.
#
# These are category-level weights. Individual signals have a local
# reliability weight. Category and signal weights are combined in the
# secondary evidence model.
# ---------------------------------------------------------------------------

RISK_WEIGHTS: Mapping[str, float] = {
    "threat_intel": 0.55,
    "registration": 0.15,
    "reverse_dns": 0.10,
    "network": 0.10,
    "structural": 0.10,
}

# Signals are intentionally not considered proof of maliciousness.
# They become stronger when multiple independent observations agree.

MALICIOUS_RISK_THRESHOLD = 0.75
SUSPICIOUS_RISK_THRESHOLD = 0.40

HIGH_RISK_THRESHOLD = 0.65
CRITICAL_RISK_THRESHOLD = 0.85

MIN_CONFIDENCE_FOR_MALICIOUS = 0.50
MIN_CONFIDENCE_FOR_SUSPICIOUS = 0.30


# =============================================================================
# ENUMS
# =============================================================================

class IPReputationStatus(str, Enum):
    COMPLETE = "complete"
    DEGRADED = "degraded"
    INVALID_INPUT = "invalid_input"
    UNKNOWN = "unknown"


class IPRiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CLEAN = "clean"
    UNKNOWN = "unknown"


class IPVersion(str, Enum):
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    IPV4_MAPPED = "ipv4_mapped"


class IPSpecialPurpose(str, Enum):
    NONE = "none"
    PRIVATE = "private"
    LOOPBACK = "loopback"
    LINK_LOCAL = "link_local"
    UNSPECIFIED = "unspecified"
    MULTICAST = "multicast"
    RESERVED = "reserved"
    DOCUMENTATION = "documentation"
    SHARED = "shared"
    BENCHMARK = "benchmark"
    UNIQUE_LOCAL = "unique_local"
    TRANSITION = "transition"
    OTHER_SPECIAL = "other_special"


class IPReputationEvidenceType(str, Enum):
    THREAT_INTEL = "threat_intel"
    REGISTRATION = "registration"
    REVERSE_DNS = "reverse_dns"
    NETWORK = "network"
    STRUCTURAL = "structural"
    INPUT = "input"


# =============================================================================
# EXCEPTIONS
# =============================================================================

class IPReputationError(Exception):
    """Base exception for IP reputation failures."""


class InvalidIPError(IPReputationError):
    """Raised when an IP address cannot be safely canonicalized."""


class IPResolutionError(IPReputationError):
    """Raised when reverse DNS resolution explicitly fails."""


class IPReputationConfigurationError(IPReputationError):
    """Raised when engine configuration is invalid."""


# =============================================================================
# MODELS
# =============================================================================

class IPIdentity(BaseModel):
    """Canonical identity of an IPv4/IPv6 address."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    original: str = Field(min_length=1, max_length=256)

    canonical_ip: str = Field(
        min_length=1,
        max_length=45,
    )

    lookup_ip: str = Field(
        min_length=1,
        max_length=45,
    )

    version: IPVersion

    mapped_ipv4: Optional[str] = Field(
        default=None,
        max_length=15,
    )

    is_private: bool = False
    is_global: bool = False
    is_reserved: bool = False
    is_loopback: bool = False
    is_link_local: bool = False
    is_multicast: bool = False
    is_unspecified: bool = False

    special_purpose: IPSpecialPurpose = IPSpecialPurpose.NONE

    @property
    def cache_key(self) -> str:
        digest = hashlib.sha256(
            self.lookup_ip.encode("ascii")
        ).hexdigest()

        return (
            f"ip-reputation:v{MODULE_VERSION}:"
            f"{digest}"
        )


class ReverseDNSObservation(BaseModel):
    """Reverse DNS and forward-confirmation result."""

    model_config = ConfigDict(extra="forbid")

    performed: bool = False

    queried_ip: Optional[str] = None

    ptr_hostnames: List[str] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_PTR_HOSTNAMES,
    )

    ptr_present: bool = False

    forward_confirmed: bool = False

    forward_confirmed_hostnames: List[str] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_PTR_HOSTNAMES,
    )

    mismatch_detected: bool = False

    hostname_count: int = Field(
        default=0,
        ge=0,
    )

    dynamic_hostname_suspected: bool = False
    infrastructure_hostname_suspected: bool = False

    resolver: Optional[str] = None

    timed_out: bool = False

    error: Optional[str] = Field(
        default=None,
        max_length=128,
    )


class IPRegistrationObservation(BaseModel):
    """
    Normalized IP/RIR/RDAP registration information.

    Providers should populate allocation_date when available.
    The engine calculates age flags itself so providers cannot accidentally
    omit them.
    """

    model_config = ConfigDict(extra="forbid")

    available: bool = False

    allocation_date: Optional[datetime] = None

    registry: Optional[str] = Field(
        default=None,
        max_length=128,
    )

    organization: Optional[str] = Field(
        default=None,
        max_length=256,
    )

    network_name: Optional[str] = Field(
        default=None,
        max_length=256,
    )

    country: Optional[str] = Field(
        default=None,
        max_length=16,
    )

    asn: Optional[int] = Field(
        default=None,
        ge=0,
    )

    status: List[str] = Field(
        default_factory=list,
        max_length=32,
    )

    age_days: Optional[int] = Field(
        default=None,
        ge=0,
    )

    newly_allocated: bool = False
    very_newly_allocated: bool = False

    error: Optional[str] = Field(
        default=None,
        max_length=128,
    )


class IPSignal(BaseModel):
    """One bounded, explainable IP reputation signal."""

    model_config = ConfigDict(extra="forbid")

    type: IPReputationEvidenceType

    name: str = Field(
        min_length=1,
        max_length=128,
    )

    value: float = Field(
        ge=0.0,
        le=1.0,
    )

    weight: float = Field(
        ge=0.0,
        le=1.0,
    )

    description: str = Field(
        min_length=1,
        max_length=DEFAULT_MAX_REASON_LENGTH,
    )


class IPReputationMetrics(BaseModel):
    """Operational metrics captured for one analysis."""

    model_config = ConfigDict(extra="forbid")

    total_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    threat_intel_duration_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    registration_duration_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    reverse_dns_duration_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    cache_hit: bool = False

    coalesced_request: bool = False

    provider_timeout_count: int = Field(
        default=0,
        ge=0,
    )

    degraded_component_count: int = Field(
        default=0,
        ge=0,
    )


class IPReputationAssessment(BaseModel):
    """Final bounded IP reputation assessment."""

    model_config = ConfigDict(extra="forbid")

    engine_version: str = MODULE_VERSION

    status: IPReputationStatus

    ip: IPIdentity

    risk_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    risk_level: IPRiskLevel

    severity: ThreatSeverity

    is_malicious: bool
    is_suspicious: bool

    reasons: List[str] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_REASONS,
    )

    signals: List[IPSignal] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_EVIDENCE_ITEMS,
    )

    reverse_dns: ReverseDNSObservation = Field(
        default_factory=ReverseDNSObservation,
    )

    registration: IPRegistrationObservation = Field(
        default_factory=IPRegistrationObservation,
    )

    threat_intel_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    providers_queried: int = Field(
        default=0,
        ge=0,
    )

    providers_successful: int = Field(
        default=0,
        ge=0,
    )

    degraded_components: List[str] = Field(
        default_factory=list,
        max_length=16,
    )

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metrics: IPReputationMetrics = Field(
        default_factory=IPReputationMetrics
    )

    cache_hit: bool = False

    @property
    def cache_key(self) -> str:
        return self.ip.cache_key


# =============================================================================
# CACHE
# =============================================================================

@dataclass
class _CacheEntry:
    value: IPReputationAssessment
    expires_at: float
    last_access: float


class IPReputationCache:
    """
    Bounded async LRU/TTL cache.

    This cache is intentionally process-local. Distributed deployments
    should place a shared cache above this component.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        max_entries: int = DEFAULT_CACHE_MAX_ENTRIES,
    ) -> None:
        if ttl_seconds < 0:
            raise IPReputationConfigurationError(
                "ttl_seconds must be >= 0"
            )

        if max_entries < 1:
            raise IPReputationConfigurationError(
                "max_entries must be >= 1"
            )

        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

        self._entries: OrderedDict[
            str,
            _CacheEntry,
        ] = OrderedDict()

        self._lock = asyncio.Lock()

        self.hits = 0
        self.misses = 0
        self.evictions = 0

    async def get(
        self,
        key: str,
    ) -> Optional[IPReputationAssessment]:
        async with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                self.misses += 1
                return None

            now = time.monotonic()

            if (
                self.ttl_seconds == 0
                or now >= entry.expires_at
            ):
                self._entries.pop(key, None)
                self.misses += 1
                return None

            entry.last_access = now

            self._entries.move_to_end(key)

            self.hits += 1

            return entry.value.model_copy(
                deep=True
            )

    async def set(
        self,
        key: str,
        value: IPReputationAssessment,
    ) -> None:
        async with self._lock:
            if self.ttl_seconds == 0:
                return

            now = time.monotonic()

            if key in self._entries:
                self._entries.pop(key, None)

            self._entries[key] = _CacheEntry(
                value=value.model_copy(deep=True),
                expires_at=now + self.ttl_seconds,
                last_access=now,
            )

            self._entries.move_to_end(key)

            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self.evictions += 1

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()

    async def stats(self) -> Mapping[str, int]:
        async with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }


# =============================================================================
# IP CANONICALIZER
# =============================================================================

class IPCanonicalizer:
    """Strict canonicalization and classification of IP addresses."""

    @staticmethod
    def canonicalize(value: str) -> IPIdentity:
        if not isinstance(value, str):
            raise InvalidIPError(
                "IP address must be a string."
            )

        original = value

        raw = value.strip()

        if not raw:
            raise InvalidIPError(
                "IP address cannot be empty."
            )

        if len(raw) > 256:
            raise InvalidIPError(
                "IP address input is too long."
            )

        if "%" in raw:
            raise InvalidIPError(
                "IPv6 zone indexes are not accepted."
            )

        try:
            parsed = ipaddress.ip_address(raw)
        except ValueError as exc:
            raise InvalidIPError(
                "Invalid IPv4 or IPv6 address."
            ) from exc

        mapped = getattr(
            parsed,
            "ipv4_mapped",
            None,
        )

        if mapped is not None:
            lookup_ip = str(mapped)
            canonical_ip = parsed.compressed

            return IPIdentity(
                original=original,
                canonical_ip=canonical_ip,
                lookup_ip=lookup_ip,
                version=IPVersion.IPV4_MAPPED,
                mapped_ipv4=str(mapped),
                is_private=mapped.is_private,
                is_global=mapped.is_global,
                is_reserved=mapped.is_reserved,
                is_loopback=mapped.is_loopback,
                is_link_local=mapped.is_link_local,
                is_multicast=mapped.is_multicast,
                is_unspecified=mapped.is_unspecified,
                special_purpose=IPCanonicalizer.classify_special(
                    mapped
                ),
            )

        if isinstance(
            parsed,
            ipaddress.IPv4Address,
        ):
            version = IPVersion.IPV4
        else:
            version = IPVersion.IPV6

        return IPIdentity(
            original=original,
            canonical_ip=parsed.compressed,
            lookup_ip=parsed.compressed,
            version=version,
            is_private=parsed.is_private,
            is_global=parsed.is_global,
            is_reserved=parsed.is_reserved,
            is_loopback=parsed.is_loopback,
            is_link_local=parsed.is_link_local,
            is_multicast=parsed.is_multicast,
            is_unspecified=parsed.is_unspecified,
            special_purpose=IPCanonicalizer.classify_special(
                parsed
            ),
        )

    @staticmethod
    def classify_special(
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    ) -> IPSpecialPurpose:

        if ip.is_unspecified:
            return IPSpecialPurpose.UNSPECIFIED

        if ip.is_loopback:
            return IPSpecialPurpose.LOOPBACK

        if ip.is_link_local:
            return IPSpecialPurpose.LINK_LOCAL

        if ip.is_multicast:
            return IPSpecialPurpose.MULTICAST

        if ip.is_private:
            # Distinguish IPv4 shared space.
            if isinstance(ip, ipaddress.IPv4Address):
                if ip in ipaddress.ip_network(
                    "100.64.0.0/10"
                ):
                    return IPSpecialPurpose.SHARED

            if isinstance(ip, ipaddress.IPv6Address):
                if ip in ipaddress.ip_network(
                    "fc00::/7"
                ):
                    return IPSpecialPurpose.UNIQUE_LOCAL

            return IPSpecialPurpose.PRIVATE

        if ip.is_reserved:
            return IPSpecialPurpose.RESERVED

        # Documentation networks.
        documentation_networks = (
            "192.0.2.0/24",
            "198.51.100.0/24",
            "203.0.113.0/24",
            "2001:db8::/32",
        )

        for network in documentation_networks:
            if ip in ipaddress.ip_network(network):
                return IPSpecialPurpose.DOCUMENTATION

        # Benchmark/testing.
        benchmark_networks = (
            "198.18.0.0/15",
        )

        for network in benchmark_networks:
            if ip in ipaddress.ip_network(network):
                return IPSpecialPurpose.BENCHMARK

        # IPv6 transition technologies.
        transition_networks = (
            "2002::/16",       # 6to4
            "2001::/32",       # Teredo
            "64:ff9b::/96",    # Well-known NAT64
        )

        for network in transition_networks:
            if ip in ipaddress.ip_network(network):
                return IPSpecialPurpose.TRANSITION

        return IPSpecialPurpose.NONE


# =============================================================================
# REVERSE DNS
# =============================================================================

ReverseDNSResolver = Callable[
    [str],
    Awaitable[ReverseDNSObservation],
]


_DYNAMIC_PTR_PATTERNS = (
    re.compile(
        r"(?:^|[-.])(?:dhcp|dynamic|dyn|pool|pppoe|dsl|cable)"
        r"(?:[-.]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[-.])(?:customer|client|host|user)"
        r"(?:[-.]|$)",
        re.IGNORECASE,
    ),
)


class SafeReverseDNSResolver:
    """
    Reverse DNS resolver with forward confirmation.

    The resolver does NOT make HTTP requests.

    PTR is considered forward-confirmed only when at least one returned PTR
    hostname resolves back to the queried address.
    """

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_REVERSE_DNS_TIMEOUT_SECONDS,
        max_ptr_hostnames: int = DEFAULT_MAX_PTR_HOSTNAMES,
    ) -> None:
        if timeout_seconds <= 0:
            raise IPReputationConfigurationError(
                "timeout_seconds must be > 0"
            )

        self.timeout_seconds = timeout_seconds

        self.max_ptr_hostnames = max(
            1,
            min(max_ptr_hostnames, 64),
        )

    async def resolve(
        self,
        ip: str,
    ) -> ReverseDNSObservation:

        started = time.perf_counter()

        try:
            ptr_names = await asyncio.wait_for(
                asyncio.to_thread(
                    socket.gethostbyaddr,
                    ip,
                ),
                timeout=self.timeout_seconds,
            )

            hostname = (
                ptr_names[0]
                if ptr_names
                else None
            )

            aliases = (
                list(ptr_names[1])
                if len(ptr_names) > 1
                else []
            )

            candidates: List[str] = []

            for item in [
                hostname,
                *aliases,
            ]:
                if not isinstance(item, str):
                    continue

                cleaned = self._normalize_hostname(item)

                if (
                    cleaned
                    and cleaned not in candidates
                ):
                    candidates.append(cleaned)

                if (
                    len(candidates)
                    >= self.max_ptr_hostnames
                ):
                    break

            if not candidates:
                return ReverseDNSObservation(
                    performed=True,
                    queried_ip=ip,
                    ptr_present=False,
                    hostname_count=0,
                    resolver="system",
                )

            confirmed: List[str] = []

            for candidate in candidates:
                if await self._forward_confirms(
                    candidate,
                    ip,
                ):
                    confirmed.append(candidate)

            dynamic = any(
                self._looks_dynamic(host)
                for host in candidates
            )

            infrastructure = any(
                self._looks_infrastructure(host)
                for host in candidates
            )

            return ReverseDNSObservation(
                performed=True,
                queried_ip=ip,
                ptr_hostnames=candidates[
                    : self.max_ptr_hostnames
                ],
                ptr_present=True,
                forward_confirmed=bool(confirmed),
                forward_confirmed_hostnames=confirmed[
                    : self.max_ptr_hostnames
                ],
                mismatch_detected=(
                    bool(candidates)
                    and not bool(confirmed)
                ),
                hostname_count=len(candidates),
                dynamic_hostname_suspected=dynamic,
                infrastructure_hostname_suspected=infrastructure,
                resolver="system",
            )

        except asyncio.TimeoutError:
            return ReverseDNSObservation(
                performed=True,
                queried_ip=ip,
                timed_out=True,
                resolver="system",
                error="reverse_dns_timeout",
            )

        except (
            socket.herror,
            socket.gaierror,
        ) as exc:
            return ReverseDNSObservation(
                performed=True,
                queried_ip=ip,
                ptr_present=False,
                resolver="system",
                error=type(exc).__name__,
            )

        except Exception as exc:
            logger.warning(
                "Reverse DNS resolution failed",
                extra={
                    "ip": ip,
                    "error_type": type(exc).__name__,
                    "duration_ms": (
                        time.perf_counter()
                        - started
                    ) * 1000.0,
                },
            )

            return ReverseDNSObservation(
                performed=True,
                queried_ip=ip,
                resolver="system",
                error=type(exc).__name__,
            )

    async def _forward_confirms(
        self,
        hostname: str,
        expected_ip: str,
    ) -> bool:

        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    socket.getaddrinfo,
                    hostname,
                    None,
                    type=socket.SOCK_STREAM,
                ),
                timeout=self.timeout_seconds,
            )

            expected = ipaddress.ip_address(
                expected_ip
            )

            for result in results:
                sockaddr = result[4]

                if not sockaddr:
                    continue

                candidate_ip = sockaddr[0]

                try:
                    parsed = ipaddress.ip_address(
                        candidate_ip
                    )
                except ValueError:
                    continue

                if parsed == expected:
                    return True

            return False

        except Exception:
            return False

    @staticmethod
    def _normalize_hostname(
        hostname: str,
    ) -> Optional[str]:

        hostname = hostname.strip().rstrip(".")

        if not hostname:
            return None

        if len(hostname) > DEFAULT_MAX_HOSTNAME_LENGTH:
            return None

        labels = hostname.split(".")

        if any(
            not label
            or len(label) > DEFAULT_MAX_LABEL_LENGTH
            for label in labels
        ):
            return None

        try:
            return hostname.encode(
                "idna"
            ).decode("ascii").lower()
        except UnicodeError:
            return None

    @staticmethod
    def _looks_dynamic(
        hostname: str,
    ) -> bool:

        return any(
            pattern.search(hostname)
            for pattern in _DYNAMIC_PTR_PATTERNS
        )

    @staticmethod
    def _looks_infrastructure(
        hostname: str,
    ) -> bool:

        infrastructure_tokens = (
            "amazonaws.com",
            "cloudfront.net",
            "googleusercontent.com",
            "azure",
            "cloudapp.azure.com",
            "digitalocean",
            "linode",
            "vultr",
            "hetzner",
            "oraclecloud",
        )

        return any(
            token in hostname
            for token in infrastructure_tokens
        )


# =============================================================================
# REGISTRATION CONTRACT
# =============================================================================

class IPRegistrationProvider:
    """
    Optional RDAP/RIR provider contract.

    Implementations must return normalized observations.

    IP RDAP lookups should use the canonical provider lookup address.
    """

    async def lookup(
        self,
        ip: str,
    ) -> IPRegistrationObservation:
        raise NotImplementedError


# =============================================================================
# REGISTRATION SIGNAL ANALYZER
# =============================================================================

class IPRegistrationSignalAnalyzer:
    """Derives age-related signals centrally."""

    def __init__(
        self,
        newly_allocated_days: int = DEFAULT_NEWLY_ALLOCATED_DAYS,
        very_newly_allocated_days: int = (
            DEFAULT_VERY_NEWLY_ALLOCATED_DAYS
        ),
    ) -> None:

        if newly_allocated_days < 1:
            raise IPReputationConfigurationError(
                "newly_allocated_days must be >= 1"
            )

        if very_newly_allocated_days < 0:
            raise IPReputationConfigurationError(
                "very_newly_allocated_days must be >= 0"
            )

        self.newly_allocated_days = newly_allocated_days

        self.very_newly_allocated_days = min(
            very_newly_allocated_days,
            newly_allocated_days,
        )

    def normalize(
        self,
        observation: IPRegistrationObservation,
    ) -> IPRegistrationObservation:

        if observation.allocation_date is None:
            return observation

        allocation = observation.allocation_date

        if allocation.tzinfo is None:
            allocation = allocation.replace(
                tzinfo=timezone.utc
            )
        else:
            allocation = allocation.astimezone(
                timezone.utc
            )

        now = datetime.now(timezone.utc)

        age = (
            now - allocation
        ).total_seconds() / 86400.0

        if age < 0:
            # Future dates should not produce negative age.
            age_days = 0
        else:
            age_days = int(age)

        return observation.model_copy(
            update={
                "age_days": age_days,
                "newly_allocated": (
                    age_days
                    <= self.newly_allocated_days
                ),
                "very_newly_allocated": (
                    age_days
                    <= self.very_newly_allocated_days
                ),
            }
        )

    def analyze(
        self,
        registration: IPRegistrationObservation,
    ) -> List[IPSignal]:

        signals: List[IPSignal] = []

        if registration.very_newly_allocated:
            signals.append(
                IPSignal(
                    type=IPReputationEvidenceType.REGISTRATION,
                    name="very_new_allocation",
                    value=0.55,
                    weight=0.80,
                    description=(
                        "The IP allocation appears to be within "
                        "the configured very-new allocation window."
                    ),
                )
            )

        elif registration.newly_allocated:
            signals.append(
                IPSignal(
                    type=IPReputationEvidenceType.REGISTRATION,
                    name="new_allocation",
                    value=0.30,
                    weight=0.65,
                    description=(
                        "The IP allocation is relatively recent."
                    ),
                )
            )

        if (
            registration.available
            and registration.allocation_date is None
        ):
            signals.append(
                IPSignal(
                    type=IPReputationEvidenceType.REGISTRATION,
                    name="allocation_date_unavailable",
                    value=0.05,
                    weight=0.10,
                    description=(
                        "Registration data is available but "
                        "allocation date is unavailable."
                    ),
                )
            )

        return signals


# =============================================================================
# SIGNAL ANALYZER
# =============================================================================

class IPSignalAnalyzer:
    """Deterministic local IP signal generation."""

    @staticmethod
    def analyze_identity(
        identity: IPIdentity,
    ) -> List[IPSignal]:

        signals: List[IPSignal] = []

        if identity.special_purpose != IPSpecialPurpose.NONE:

            # These are classification signals, not maliciousness proof.
            description_map = {
                IPSpecialPurpose.PRIVATE:
                    "Address belongs to private address space.",
                IPSpecialPurpose.LOOPBACK:
                    "Address is a loopback address.",
                IPSpecialPurpose.LINK_LOCAL:
                    "Address belongs to link-local space.",
                IPSpecialPurpose.UNSPECIFIED:
                    "Address is an unspecified address.",
                IPSpecialPurpose.MULTICAST:
                    "Address belongs to multicast space.",
                IPSpecialPurpose.RESERVED:
                    "Address belongs to reserved address space.",
                IPSpecialPurpose.DOCUMENTATION:
                    "Address belongs to documentation-only space.",
                IPSpecialPurpose.SHARED:
                    "Address belongs to shared address space.",
                IPSpecialPurpose.BENCHMARK:
                    "Address belongs to benchmark/testing space.",
                IPSpecialPurpose.UNIQUE_LOCAL:
                    "Address belongs to IPv6 unique-local space.",
                IPSpecialPurpose.TRANSITION:
                    "Address belongs to an IPv6 transition mechanism.",
                IPSpecialPurpose.OTHER_SPECIAL:
                    "Address belongs to special-purpose space.",
            }

            # Non-global classification should never be interpreted as
            # maliciousness. A small diagnostic value is retained so that
            # explainability and confidence calculations can see it.
            value = {
                IPSpecialPurpose.LOOPBACK: 0.05,
                IPSpecialPurpose.PRIVATE: 0.02,
                IPSpecialPurpose.LINK_LOCAL: 0.05,
                IPSpecialPurpose.UNSPECIFIED: 0.05,
                IPSpecialPurpose.MULTICAST: 0.02,
                IPSpecialPurpose.RESERVED: 0.04,
                IPSpecialPurpose.DOCUMENTATION: 0.00,
                IPSpecialPurpose.SHARED: 0.03,
                IPSpecialPurpose.BENCHMARK: 0.00,
                IPSpecialPurpose.UNIQUE_LOCAL: 0.02,
                IPSpecialPurpose.TRANSITION: 0.03,
                IPSpecialPurpose.OTHER_SPECIAL: 0.02,
            }.get(
                identity.special_purpose,
                0.0,
            )

            signals.append(
                IPSignal(
                    type=IPReputationEvidenceType.NETWORK,
                    name=(
                        f"special_purpose:"
                        f"{identity.special_purpose.value}"
                    ),
                    value=value,
                    weight=0.20,
                    description=description_map.get(
                        identity.special_purpose,
                        "Special-purpose IP classification.",
                    ),
                )
            )

        if identity.version == IPVersion.IPV4_MAPPED:
            signals.append(
                IPSignal(
                    type=IPReputationEvidenceType.STRUCTURAL,
                    name="ipv4_mapped_ipv6",
                    value=0.0,
                    weight=0.05,
                    description=(
                        "Input is an IPv4-mapped IPv6 representation; "
                        "external reputation lookups use the embedded IPv4."
                    ),
                )
            )

        return signals

    @staticmethod
    def analyze_reverse_dns(
        observation: ReverseDNSObservation,
    ) -> List[IPSignal]:

        signals: List[IPSignal] = []

        if not observation.performed:
            return signals

        if observation.ptr_present:
            if observation.forward_confirmed:
                signals.append(
                    IPSignal(
                        type=IPReputationEvidenceType.REVERSE_DNS,
                        name="forward_confirmed_ptr",
                        value=0.00,
                        weight=0.35,
                        description=(
                            "At least one PTR hostname resolves back "
                            "to the queried IP address."
                        ),
                    )
                )
            else:
                signals.append(
                    IPSignal(
                        type=IPReputationEvidenceType.REVERSE_DNS,
                        name="unconfirmed_ptr",
                        value=0.12,
                        weight=0.40,
                        description=(
                            "PTR data exists but no returned hostname "
                            "was forward-confirmed to the queried IP."
                        ),
                    )
                )

        if observation.mismatch_detected:
            signals.append(
                IPSignal(
                    type=IPReputationEvidenceType.REVERSE_DNS,
                    name="ptr_forward_mismatch",
                    value=0.20,
                    weight=0.50,
                    description=(
                        "Returned PTR hostname data did not "
                        "forward-confirm to the queried IP."
                    ),
                )
            )

        if observation.dynamic_hostname_suspected:
            signals.append(
                IPSignal(
                    type=IPReputationEvidenceType.REVERSE_DNS,
                    name="dynamic_ptr_hostname",
                    value=0.18,
                    weight=0.30,
                    description=(
                        "PTR hostname resembles dynamically assigned "
                        "or customer access infrastructure."
                    ),
                )

        if observation.hostname_count > 4:
            signals.append(
                IPSignal(
                    type=IPReputationEvidenceType.REVERSE_DNS,
                    name="multiple_ptr_hostnames",
                    value=0.08,
                    weight=0.15,
                    description=(
                        "Multiple PTR hostnames were returned for the address."
                    ),
                )
            )

        return signals


# =============================================================================
# CONCURRENCY / METRICS
# =============================================================================

class _Metrics:
    """Small lock-protected in-process operational counter set."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

        self.total_requests = 0
        self.cache_hits = 0
        self.coalesced_requests = 0

        self.degraded_requests = 0
        self.timeout_count = 0

        self.risk_levels: Counter[str] = Counter()

    async def request(self) -> None:
        async with self._lock:
            self.total_requests += 1

    async def cache_hit(self) -> None:
        async with self._lock:
            self.cache_hits += 1

    async def coalesced(self) -> None:
        async with self._lock:
            self.coalesced_requests += 1

    async def degraded(self) -> None:
        async with self._lock:
            self.degraded_requests += 1

    async def timeout(self) -> None:
        async with self._lock:
            self.timeout_count += 1

    async def risk_level(
        self,
        value: IPRiskLevel,
    ) -> None:
        async with self._lock:
            self.risk_levels[value.value] += 1

    async def snapshot(self) -> Mapping[str, Any]:
        async with self._lock:
            return {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "coalesced_requests": self.coalesced_requests,
                "degraded_requests": self.degraded_requests,
                "timeout_count": self.timeout_count,
                "risk_levels": dict(self.risk_levels),
            }


# =============================================================================
# ENGINE
# =============================================================================

class IPReputationEngine:
    """
    Production IP reputation engine.

    External dependencies are isolated by component-specific concurrency
    pools. This prevents slow registration/RDNS work from starving the
    highest-value threat-intelligence path.
    """

    def __init__(
        self,
        threat_intel: Optional["ThreatIntelAggregator"] = None,
        registration_provider: Optional[
            IPRegistrationProvider
        ] = None,
        reverse_dns_resolver: Optional[
            ReverseDNSResolver
        ] = None,
        *,
        enable_reverse_dns: bool = True,
        enable_registration: bool = True,
        cache: Optional[IPReputationCache] = None,
        provider_timeout_seconds: float = (
            DEFAULT_PROVIDER_TIMEOUT_SECONDS
        ),
        threat_intel_timeout_seconds: float = (
            DEFAULT_THREAT_INTEL_TIMEOUT_SECONDS
        ),
        registration_timeout_seconds: float = (
            DEFAULT_REGISTRATION_TIMEOUT_SECONDS
        ),
        reverse_dns_timeout_seconds: float = (
            DEFAULT_REVERSE_DNS_TIMEOUT_SECONDS
        ),
        threat_intel_concurrency: int = (
            DEFAULT_THREAT_INTEL_CONCURRENCY
        ),
        registration_concurrency: int = (
            DEFAULT_REGISTRATION_CONCURRENCY
        ),
        reverse_dns_concurrency: int = (
            DEFAULT_REVERSE_DNS_CONCURRENCY
        ),
        newly_allocated_days: int = (
            DEFAULT_NEWLY_ALLOCATED_DAYS
        ),
        very_newly_allocated_days: int = (
            DEFAULT_VERY_NEWLY_ALLOCATED_DAYS
        ),
    ) -> None:

        self.threat_intel = threat_intel
        self.registration_provider = registration_provider

        self.reverse_dns_resolver = (
            reverse_dns_resolver
            or SafeReverseDNSResolver(
                timeout_seconds=reverse_dns_timeout_seconds,
            ).resolve
        )

        self.enable_reverse_dns = enable_reverse_dns

        self.enable_registration = (
            enable_registration
            and registration_provider is not None
        )

        self.cache = cache

        self.provider_timeout_seconds = (
            self._validate_timeout(
                provider_timeout_seconds,
                "provider_timeout_seconds",
            )
        )

        self.threat_intel_timeout_seconds = (
            self._validate_timeout(
                threat_intel_timeout_seconds,
                "threat_intel_timeout_seconds",
            )
        )

        self.registration_timeout_seconds = (
            self._validate_timeout(
                registration_timeout_seconds,
                "registration_timeout_seconds",
            )
        )

        self.reverse_dns_timeout_seconds = (
            self._validate_timeout(
                reverse_dns_timeout_seconds,
                "reverse_dns_timeout_seconds",
            )
        )

        self._threat_intel_semaphore = asyncio.Semaphore(
            self._validate_concurrency(
                threat_intel_concurrency,
                "threat_intel_concurrency",
            )
        )

        self._registration_semaphore = asyncio.Semaphore(
            self._validate_concurrency(
                registration_concurrency,
                "registration_concurrency",
            )
        )

        self._reverse_dns_semaphore = asyncio.Semaphore(
            self._validate_concurrency(
                reverse_dns_concurrency,
                "reverse_dns_concurrency",
            )
        )

        self.registration_analyzer = (
            IPRegistrationSignalAnalyzer(
                newly_allocated_days=newly_allocated_days,
                very_newly_allocated_days=(
                    very_newly_allocated_days
                ),
            )
        )

        # Request coalescing for normal requests.
        self._inflight: Dict[
            str,
            asyncio.Task[IPReputationAssessment],
        ] = {}

        self._inflight_lock = asyncio.Lock()

        self._metrics = _Metrics()

        self._config_fingerprint = self._build_config_fingerprint()

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    async def analyze(
        self,
        ip: str,
        *,
        force_refresh: bool = False,
        resolve_reverse_dns: Optional[bool] = None,
        lookup_registration: Optional[bool] = None,
    ) -> IPReputationAssessment:

        await self._metrics.request()

        started = time.perf_counter()

        identity = IPCanonicalizer.canonicalize(ip)

        use_reverse_dns = (
            self.enable_reverse_dns
            if resolve_reverse_dns is None
            else bool(resolve_reverse_dns)
        )

        use_registration = (
            self.enable_registration
            if lookup_registration is None
            else bool(lookup_registration)
        )

        cache_key = self._cache_key(
            identity,
            use_reverse_dns=use_reverse_dns,
            use_registration=use_registration,
        )

        if self.cache and not force_refresh:
            cached = await self.cache.get(cache_key)

            if cached is not None:
                cached.cache_hit = True

                await self._metrics.cache_hit()

                return cached

        # Force-refresh calls intentionally execute independently.
        #
        # Normal requests use request coalescing.
        if force_refresh:
            return await self._analyze_uncached(
                identity=identity,
                use_reverse_dns=use_reverse_dns,
                use_registration=use_registration,
                started=started,
                cache_key=cache_key,
                coalesced_request=False,
                force_refresh=True,
            )

        async with self._inflight_lock:

            existing = self._inflight.get(
                cache_key
            )

            if existing is not None:
                await self._metrics.coalesced()

                task = existing

            else:
                task = asyncio.create_task(
                    self._analyze_uncached(
                        identity=identity,
                        use_reverse_dns=use_reverse_dns,
                        use_registration=use_registration,
                        started=started,
                        cache_key=cache_key,
                        coalesced_request=False,
                        force_refresh=False,
                    )
                )

                self._inflight[cache_key] = task

        try:
            result = await asyncio.shield(task)

            if task is not asyncio.current_task():
                result = result.model_copy(
                    deep=True
                )

            return result

        finally:
            if (
                task is not None
                and task.done()
            ):
                async with self._inflight_lock:
                    current = self._inflight.get(
                        cache_key
                    )

                    if current is task:
                        self._inflight.pop(
                            cache_key,
                            None,
                        )

    # -------------------------------------------------------------------------
    # ANALYSIS CORE
    # -------------------------------------------------------------------------

    async def _analyze_uncached(
        self,
        *,
        identity: IPIdentity,
        use_reverse_dns: bool,
        use_registration: bool,
        started: float,
        cache_key: str,
        coalesced_request: bool,
        force_refresh: bool,
    ) -> IPReputationAssessment:

        degraded: List[str] = []

        signals: List[IPSignal] = []

        signals.extend(
            IPSignalAnalyzer.analyze_identity(
                identity
            )
        )

        reverse_dns = ReverseDNSObservation(
            performed=False
        )

        registration = IPRegistrationObservation(
            available=False
        )

        threat_report: Optional[
            AggregatedThreatSummary
        ] = None

        threat_duration: Optional[float] = None
        registration_duration: Optional[float] = None
        reverse_dns_duration: Optional[float] = None

        timeout_count = 0

        # ---------------------------------------------------------------------
        # Build independent tasks.
        # ---------------------------------------------------------------------

        tasks: List[
            Tuple[
                str,
                Awaitable[Any],
            ]
        ] = []

        if self.threat_intel is not None:
            tasks.append(
                (
                    "threat_intel",
                    self._run_threat_intel(
                        identity.lookup_ip,
                        force_refresh=force_refresh,
                    ),
                )
            )

        if (
            use_registration
            and self.registration_provider is not None
        ):
            tasks.append(
                (
                    "registration",
                    self._run_registration(
                        identity.lookup_ip
                    ),
                )
            )

        if (
            use_reverse_dns
            and self.reverse_dns_resolver is not None
        ):
            tasks.append(
                (
                    "reverse_dns",
                    self._run_reverse_dns(
                        identity.lookup_ip
                    ),
                )
            )

        if tasks:
            task_results = await asyncio.gather(
                *[
                    awaitable
                    for _, awaitable in tasks
                ],
                return_exceptions=True,
            )
        else:
            task_results = []

        observations = {
            name: result
            for (name, _),
            result in zip(tasks, task_results)
        }

        # ---------------------------------------------------------------------
        # Threat intelligence
        # ---------------------------------------------------------------------

        threat_result = observations.get(
            "threat_intel"
        )

        if isinstance(
            threat_result,
            _TimedResult,
        ):
            threat_duration = (
                threat_result.duration_ms
            )
            threat_report = threat_result.value

            if threat_result.timed_out:
                timeout_count += 1
                degraded.append("threat_intel")

        elif isinstance(
            threat_result,
            Exception,
        ):
            degraded.append("threat_intel")

        elif threat_result is not None:
            threat_report = threat_result

        # ---------------------------------------------------------------------
        # Registration
        # ---------------------------------------------------------------------

        registration_result = observations.get(
            "registration"
        )

        if isinstance(
            registration_result,
            _TimedResult,
        ):
            registration_duration = (
                registration_result.duration_ms
            )

            if registration_result.value is not None:
                registration = (
                    self.registration_analyzer.normalize(
                        registration_result.value
                    )
                )

            if registration_result.timed_out:
                timeout_count += 1
                degraded.append("registration")

        elif isinstance(
            registration_result,
            IPRegistrationObservation,
        ):
            registration = (
                self.registration_analyzer.normalize(
                    registration_result
                )
            )

        elif isinstance(
            registration_result,
            Exception,
        ):
            degraded.append("registration")

        # ---------------------------------------------------------------------
        # Reverse DNS
        # ---------------------------------------------------------------------

        reverse_result = observations.get(
            "reverse_dns"
        )

        if isinstance(
            reverse_result,
            _TimedResult,
        ):
            reverse_dns_duration = (
                reverse_result.duration_ms
            )

            if reverse_result.value is not None:
                reverse_dns = reverse_result.value

            if reverse_result.timed_out:
                timeout_count += 1
                degraded.append("reverse_dns")

        elif isinstance(
            reverse_result,
            ReverseDNSObservation,
        ):
            reverse_dns = reverse_result

        elif isinstance(
            reverse_result,
            Exception,
        ):
            degraded.append("reverse_dns")

        # ---------------------------------------------------------------------
        # Local evidence
        # ---------------------------------------------------------------------

        signals.extend(
            self.registration_analyzer.analyze(
                registration
            )
        )

        signals.extend(
            IPSignalAnalyzer.analyze_reverse_dns(
                reverse_dns
            )
        )

        # ---------------------------------------------------------------------
        # Threat intelligence signal
        # ---------------------------------------------------------------------

        threat_score: Optional[float] = None

        providers_queried = 0
        providers_successful = 0

        if threat_report is not None:

            threat_score = self._safe_float(
                getattr(
                    threat_report,
                    "aggregated_risk_score",
                    None,
                )
            )

            providers_queried = (
                self._safe_int(
                    getattr(
                        threat_report,
                        "total_providers_queried",
                        0,
                    )
                )
            )

            providers_successful = (
                self._safe_int(
                    getattr(
                        threat_report,
                        "successful_provider_responses",
                        0,
                    )
                )
            )

            if threat_score is not None:
                signals.append(
                    IPSignal(
                        type=(
                            IPReputationEvidenceType
                            .THREAT_INTEL
                        ),
                        name="aggregated_provider_reputation",
                        value=threat_score,
                        weight=1.0,
                        description=(
                            "Aggregated multi-provider "
                            "threat-intelligence reputation score."
                        ),
                    )
                )

        # ---------------------------------------------------------------------
        # Risk / confidence
        # ---------------------------------------------------------------------

        risk_score = self._calculate_risk(
            identity=identity,
            threat_score=threat_score,
            signals=signals,
        )

        confidence = self._calculate_confidence(
            identity=identity,
            threat_score=threat_score,
            providers_queried=providers_queried,
            providers_successful=providers_successful,
            registration=registration,
            reverse_dns=reverse_dns,
            signals=signals,
            degraded_components=degraded,
        )

        risk_level = self._risk_level(
            risk_score,
            confidence,
        )

        severity = self._severity(
            risk_score,
            confidence,
        )

        is_malicious = (
            risk_score
            >= MALICIOUS_RISK_THRESHOLD
            and confidence
            >= MIN_CONFIDENCE_FOR_MALICIOUS
        )

        is_suspicious = (
            not is_malicious
            and risk_score
            >= SUSPICIOUS_RISK_THRESHOLD
            and confidence
            >= MIN_CONFIDENCE_FOR_SUSPICIOUS
        )

        reasons = self._build_reasons(
            signals=signals,
            risk_score=risk_score,
            confidence=confidence,
            threat_report=threat_report,
        )

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        # ---------------------------------------------------------------------
        # Status
        # ---------------------------------------------------------------------

        if degraded:
            status = IPReputationStatus.DEGRADED
        else:
            status = IPReputationStatus.COMPLETE

        if not signals:
            status = IPReputationStatus.UNKNOWN

        metrics = IPReputationMetrics(
            total_duration_ms=elapsed_ms,
            threat_intel_duration_ms=(
                threat_duration
            ),
            registration_duration_ms=(
                registration_duration
            ),
            reverse_dns_duration_ms=(
                reverse_dns_duration
            ),
            cache_hit=False,
            coalesced_request=coalesced_request,
            provider_timeout_count=timeout_count,
            degraded_component_count=len(
                set(degraded)
            ),
        )

        assessment = IPReputationAssessment(
            status=status,
            ip=identity,
            risk_score=risk_score,
            confidence=confidence,
            risk_level=risk_level,
            severity=severity,
            is_malicious=is_malicious,
            is_suspicious=is_suspicious,
            reasons=reasons,
            signals=self._bound_signals(
                signals
            ),
            reverse_dns=reverse_dns,
            registration=registration,
            threat_intel_score=threat_score,
            providers_queried=providers_queried,
            providers_successful=providers_successful,
            degraded_components=sorted(
                set(degraded)
            ),
            metrics=metrics,
        )

        if self.cache:
            await self.cache.set(
                cache_key,
                assessment,
            )

        if degraded:
            await self._metrics.degraded()

        for _ in range(timeout_count):
            await self._metrics.timeout()

        await self._metrics.risk_level(
            risk_level
        )

        return assessment

    # -------------------------------------------------------------------------
    # EXTERNAL OPERATIONS
    # -------------------------------------------------------------------------

    async def _run_threat_intel(
        self,
        ip: str,
        *,
        force_refresh: bool,
    ) -> "_TimedResult":

        started = time.perf_counter()

        try:
            async with self._threat_intel_semaphore:
                result = await asyncio.wait_for(
                    self.threat_intel.analyze_ioc(
                        ip,
                        IOCType.IP,
                        force_refresh=force_refresh,
                    ),
                    timeout=self.threat_intel_timeout_seconds,
                )

            return _TimedResult(
                value=result,
                duration_ms=(
                    time.perf_counter()
                    - started
                ) * 1000.0,
            )

        except asyncio.TimeoutError:
            return _TimedResult(
                value=None,
                duration_ms=(
                    time.perf_counter()
                    - started
                ) * 1000.0,
                timed_out=True,
            )

    async def _run_registration(
        self,
        ip: str,
    ) -> "_TimedResult":

        started = time.perf_counter()

        try:
            async with self._registration_semaphore:
                result = await asyncio.wait_for(
                    self.registration_provider.lookup(
                        ip
                    ),
                    timeout=self.registration_timeout_seconds,
                )

            return _TimedResult(
                value=result,
                duration_ms=(
                    time.perf_counter()
                    - started
                ) * 1000.0,
            )

        except asyncio.TimeoutError:
            return _TimedResult(
                value=None,
                duration_ms=(
                    time.perf_counter()
                    - started
                ) * 1000.0,
                timed_out=True,
            )

    async def _run_reverse_dns(
        self,
        ip: str,
    ) -> "_TimedResult":

        started = time.perf_counter()

        try:
            async with self._reverse_dns_semaphore:
                result = await asyncio.wait_for(
                    self.reverse_dns_resolver(ip),
                    timeout=self.reverse_dns_timeout_seconds,
                )

            return _TimedResult(
                value=result,
                duration_ms=(
                    time.perf_counter()
                    - started
                ) * 1000.0,
            )

        except asyncio.TimeoutError:
            return _TimedResult(
                value=None,
                duration_ms=(
                    time.perf_counter()
                    - started
                ) * 1000.0,
                timed_out=True,
            )

    # -------------------------------------------------------------------------
    # RISK ENGINE
    # -------------------------------------------------------------------------

    @staticmethod
    def _calculate_risk(
        *,
        identity: IPIdentity,
        threat_score: Optional[float],
        signals: Sequence[IPSignal],
    ) -> float:
        """
        Calculate bounded risk without an artificial non-TI ceiling.

        Threat intelligence is dominant when present, but local evidence
        retains meaningful influence.

        The secondary evidence is aggregated by category first, preventing
        repeated signals from one category from overpowering the model.
        """

        category_scores: Dict[
            str,
            List[float],
        ] = {}

        for signal in signals:

            if signal.type == (
                IPReputationEvidenceType.THREAT_INTEL
            ):
                continue

            category = signal.type.value

            category_scores.setdefault(
                category,
                [],
            ).append(
                signal.value * signal.weight
            )

        secondary_category_scores: Dict[
            str,
            float,
        ] = {}

        for category, values in category_scores.items():
            if not values:
                continue

            # Saturating combination:
            #
            # 1 - Π(1-x)
            #
            # avoids linear over-counting of repeated correlated signals.
            residual = 1.0

            for value in values:
                bounded = max(
                    0.0,
                    min(1.0, value),
                )

                residual *= (
                    1.0 - bounded
                )

            secondary_category_scores[
                category
            ] = 1.0 - residual

        weighted_sum = 0.0
        total_weight = 0.0

        for category, category_score in (
            secondary_category_scores.items()
        ):
            category_weight = RISK_WEIGHTS.get(
                category,
                0.05,
            )

            weighted_sum += (
                category_score
                * category_weight
            )

            total_weight += category_weight

        if total_weight > 0:
            secondary_score = (
                weighted_sum
                / total_weight
            )
        else:
            secondary_score = 0.0

        # Infrastructure / special-purpose addresses are not intrinsically
        # malicious. Do not manufacture risk from classification alone.
        if not identity.is_global:
            secondary_score *= 0.50

        if threat_score is None:
            # No artificial 0.65 cap.
            #
            # However, purely local evidence is still deliberately bounded
            # by the strength of the observations.
            return max(
                0.0,
                min(
                    1.0,
                    secondary_score * 0.85,
                ),
            )

        ti_weight = RISK_WEIGHTS[
            "threat_intel"
        ]

        risk = (
            threat_score * ti_weight
            + secondary_score
            * (1.0 - ti_weight)
        )

        # Strong independent corroboration can slightly increase risk,
        # but never by more than 0.10.
        corroborating_categories = sum(
            1
            for category_score
            in secondary_category_scores.values()
            if category_score >= 0.25
        )

        if corroborating_categories >= 3:
            risk += 0.05
        elif corroborating_categories == 2:
            risk += 0.025

        return max(
            0.0,
            min(1.0, risk),
        )

    # -------------------------------------------------------------------------
    # CONFIDENCE
    # -------------------------------------------------------------------------

    @staticmethod
    def _calculate_confidence(
        *,
        identity: IPIdentity,
        threat_score: Optional[float],
        providers_queried: int,
        providers_successful: int,
        registration: IPRegistrationObservation,
        reverse_dns: ReverseDNSObservation,
        signals: Sequence[IPSignal],
        degraded_components: Sequence[str],
    ) -> float:

        evidence: List[float] = []

        # Local parsing/classification is reliable but does not deserve
        # a huge confidence boost.
        evidence.append(0.35)

        if threat_score is not None:
            if providers_queried > 0:
                coverage = (
                    providers_successful
                    / max(
                        providers_queried,
                        1,
                    )
                )

                coverage = max(
                    0.0,
                    min(1.0, coverage),
                )

                evidence.append(
                    0.60
                    + 0.35 * coverage
                )
            else:
                evidence.append(0.55)

        if registration.available:
            evidence.append(0.70)

        if reverse_dns.performed:
            if reverse_dns.forward_confirmed:
                evidence.append(0.72)
            elif reverse_dns.ptr_present:
                evidence.append(0.50)
            else:
                evidence.append(0.35)

        independent_signal_types = {
            signal.type
            for signal in signals
            if signal.value > 0
        }

        if len(independent_signal_types) >= 3:
            evidence.append(0.78)
        elif len(independent_signal_types) == 2:
            evidence.append(0.65)
        elif len(independent_signal_types) == 1:
            evidence.append(0.48)

        confidence = (
            sum(evidence)
            / max(len(evidence), 1)
        )

        # Degraded components explicitly reduce confidence.
        if degraded_components:
            reduction = min(
                0.30,
                0.08
                * len(set(degraded_components)),
            )

            confidence -= reduction

        # Non-global classification can be very certain as classification,
        # but it does not make a reputation verdict highly certain.
        if (
            not identity.is_global
            and threat_score is None
        ):
            confidence = min(
                confidence,
                0.60,
            )

        return max(
            0.0,
            min(1.0, confidence),
        )

    # -------------------------------------------------------------------------
    # DECISION
    # -------------------------------------------------------------------------

    @staticmethod
    def _risk_level(
        risk_score: float,
        confidence: float,
    ) -> IPRiskLevel:

        if confidence < 0.20:
            return IPRiskLevel.UNKNOWN

        if risk_score >= CRITICAL_RISK_THRESHOLD:
            return IPRiskLevel.CRITICAL

        if risk_score >= HIGH_RISK_THRESHOLD:
            return IPRiskLevel.HIGH

        if risk_score >= SUSPICIOUS_RISK_THRESHOLD:
            return IPRiskLevel.MEDIUM

        if risk_score >= 0.15:
            return IPRiskLevel.LOW

        return IPRiskLevel.CLEAN

    @staticmethod
    def _severity(
        risk_score: float,
        confidence: float,
    ) -> ThreatSeverity:

        if confidence < 0.20:
            return ThreatSeverity.UNKNOWN

        if risk_score >= CRITICAL_RISK_THRESHOLD:
            return ThreatSeverity.CRITICAL

        if risk_score >= HIGH_RISK_THRESHOLD:
            return ThreatSeverity.HIGH

        if risk_score >= SUSPICIOUS_RISK_THRESHOLD:
            return ThreatSeverity.MEDIUM

        if risk_score >= 0.15:
            return ThreatSeverity.LOW

        return ThreatSeverity.CLEAN

    # -------------------------------------------------------------------------
    # REASONS
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_reasons(
        *,
        signals: Sequence[IPSignal],
        risk_score: float,
        confidence: float,
        threat_report: Optional[Any],
    ) -> List[str]:

        reasons: List[str] = []

        for signal in signals:
            if signal.value <= 0:
                continue

            reasons.append(
                signal.description[:DEFAULT_MAX_REASON_LENGTH]
            )

        if threat_report is not None:
            provider_tags = getattr(
                threat_report,
                "combined_raw_tags",
                [],
            )

            if isinstance(
                provider_tags,
                Iterable,
            ):
                for tag in list(
                    provider_tags
                )[:16]:

                    if not isinstance(
                        tag,
                        str,
                    ):
                        continue

                    cleaned = (
                        tag.strip()
                        .replace("\n", " ")
                        .replace("\r", " ")
                    )

                    if not cleaned:
                        continue

                    reasons.append(
                        f"Threat-intelligence tag: "
                        f"{cleaned[:DEFAULT_MAX_TAG_LENGTH]}"
                    )

        if confidence < 0.35:
            reasons.append(
                "Evidence confidence is limited."
            )

        if risk_score >= CRITICAL_RISK_THRESHOLD:
            reasons.append(
                "Overall risk score is critical."
            )
        elif risk_score >= HIGH_RISK_THRESHOLD:
            reasons.append(
                "Overall risk score is high."
            )
        elif risk_score >= SUSPICIOUS_RISK_THRESHOLD:
            reasons.append(
                "Overall risk score is medium."
            )

        unique: List[str] = []

        seen = set()

        for reason in reasons:
            normalized = reason.strip()

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            unique.append(normalized)

            if len(unique) >= DEFAULT_MAX_REASONS:
                break

        return unique

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _cache_key(
        self,
        identity: IPIdentity,
        *,
        use_reverse_dns: bool,
        use_registration: bool,
    ) -> str:

        payload = (
            f"{identity.lookup_ip}|"
            f"rdns={int(use_reverse_dns)}|"
            f"registration={int(use_registration)}|"
            f"cfg={self._config_fingerprint}"
        )

        digest = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

        return (
            f"ip-reputation:"
            f"v{MODULE_VERSION}:"
            f"{digest}"
        )

    def _build_config_fingerprint(self) -> str:

        payload = "|".join(
            [
                str(
                    self.registration_analyzer
                    .newly_allocated_days
                ),
                str(
                    self.registration_analyzer
                    .very_newly_allocated_days
                ),
                str(
                    self.enable_reverse_dns
                ),
                str(
                    self.enable_registration
                ),
                str(
                    self.threat_intel_timeout_seconds
                ),
                str(
                    self.registration_timeout_seconds
                ),
                str(
                    self.reverse_dns_timeout_seconds
                ),
            ]
        )

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()[:16]

    @staticmethod
    def _bound_signals(
        signals: Sequence[IPSignal],
    ) -> List[IPSignal]:

        # Remove zero-information signals unless they are useful
        # for explicit structural diagnostics.
        filtered = [
            signal
            for signal in signals
            if (
                signal.value > 0
                or signal.weight > 0
            )
        ]

        # Deterministic ordering:
        # highest effective contribution first.
        filtered.sort(
            key=lambda signal: (
                -(
                    signal.value
                    * signal.weight
                ),
                signal.type.value,
                signal.name,
            )
        )

        return [
            signal.model_copy(
                deep=True
            )
            for signal in filtered[
                :DEFAULT_MAX_EVIDENCE_ITEMS
            ]
        ]

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> Optional[float]:

        if value is None:
            return None

        try:
            number = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

        if not math.isfinite(number):
            return None

        return max(
            0.0,
            min(1.0, number),
        )

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> int:

        try:
            number = int(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0

        return max(0, number)

    @staticmethod
    def _validate_timeout(
        value: float,
        name: str,
    ) -> float:

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise IPReputationConfigurationError(
                f"{name} must be numeric."
            ) from exc

        if (
            not math.isfinite(value)
            or value <= 0
            or value > 120
        ):
            raise IPReputationConfigurationError(
                f"{name} must be > 0 and <= 120 seconds."
            )

        return value

    @staticmethod
    def _validate_concurrency(
        value: int,
        name: str,
    ) -> int:

        try:
            value = int(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise IPReputationConfigurationError(
                f"{name} must be an integer."
            ) from exc

        if value < 1 or value > 1024:
            raise IPReputationConfigurationError(
                f"{name} must be between 1 and 1024."
            )

        return value

    async def metrics_snapshot(
        self,
    ) -> Mapping[str, Any]:

        return await self._metrics.snapshot()


# =============================================================================
# INTERNAL TIMING RESULT
# =============================================================================

@dataclass
class _TimedResult:
    value: Any
    duration_ms: float
    timed_out: bool = False


# =============================================================================
# CONVENIENCE API
# =============================================================================

async def assess_ip_reputation(
    ip: str,
    *,
    threat_intel: Optional[
        "ThreatIntelAggregator"
    ] = None,
    registration_provider: Optional[
        IPRegistrationProvider
    ] = None,
    reverse_dns_resolver: Optional[
        ReverseDNSResolver
    ] = None,
    enable_reverse_dns: bool = True,
    enable_registration: bool = True,
    force_refresh: bool = False,
) -> IPReputationAssessment:
    """
    Convenience one-shot API.

    For high-QPS applications, construct one IPReputationEngine and reuse it
    so that its cache, concurrency pools, request coalescing and metrics
    remain effective.
    """

    engine = IPReputationEngine(
        threat_intel=threat_intel,
        registration_provider=registration_provider,
        reverse_dns_resolver=reverse_dns_resolver,
        enable_reverse_dns=enable_reverse_dns,
        enable_registration=enable_registration,
    )

    return await engine.analyze(
        ip,
        force_refresh=force_refresh,
    )


# =============================================================================
# PUBLIC EXPORTS
# =============================================================================

__all__ = [
    "MODULE_VERSION",
    "IPReputationStatus",
    "IPRiskLevel",
    "IPVersion",
    "IPSpecialPurpose",
    "IPReputationEvidenceType",
    "IPReputationError",
    "InvalidIPError",
    "IPResolutionError",
    "IPReputationConfigurationError",
    "IPIdentity",
    "ReverseDNSObservation",
    "IPRegistrationObservation",
    "IPSignal",
    "IPReputationMetrics",
    "IPReputationAssessment",
    "IPReputationCache",
    "IPCanonicalizer",
    "SafeReverseDNSResolver",
    "IPRegistrationProvider",
    "IPRegistrationSignalAnalyzer",
    "IPSignalAnalyzer",
    "IPReputationEngine",
    "assess_ip_reputation",
]
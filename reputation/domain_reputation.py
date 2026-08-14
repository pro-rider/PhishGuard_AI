"""
PhishGuard AI - Domain Reputation & Risk Intelligence Engine
=============================================================

Version: 15.0.0
File Path:
    reputation/domain_reputation.py

Purpose
-------
Production-grade domain reputation analysis.

Design goals
------------
- Strict hostname / URL canonicalization
- IDNA-safe normalization
- Public Suffix List aware registrable-domain extraction
- Unicode mixed-script / homograph detection
- Bounded typo-squatting detection
- Tokenized lexical analysis
- Entropy / structural analysis
- Threat-intelligence aggregation
- Optional RDAP / registration intelligence
- Bounded DNS analysis
- CNAME-chain inspection
- Optional NS / MX / TXT observations
- Explicit DNS failure classification
- Analysis deadlines
- Provider timeout isolation
- Bounded concurrency
- Deterministic risk scoring
- Independent confidence scoring
- Explainable evidence
- Evidence deduplication
- Configuration-aware cache keys
- Bounded LRU + TTL cache
- Cache stampede protection
- Cancellation-safe async behavior
- No secret leakage
- No arbitrary URL fetching
- No provider-specific scoring assumptions

Important
---------
This module performs DNS lookups only. It does NOT fetch HTTP/HTTPS
content and must not be used as a substitute for the project's
SSRF/Egress controls.

PSL
---
A Public Suffix List resolver should be supplied for production.
`tldextract` is supported when installed.

Threat Intel
------------
The engine expects the existing ThreatIntelAggregator interface:

    await threat_intel.analyze_ioc(
        domain,
        IOCType.DOMAIN,
        force_refresh=force_refresh,
    )

Provider-specific details are intentionally not required by this module.
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
import unicodedata
from collections import OrderedDict
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
from urllib.parse import urlparse

from threat_intel.base import IOCType, ThreatSeverity

try:
    from threat_intel.aggregator import (
        AggregatedThreatSummary,
        ThreatIntelAggregator,
    )
except ImportError:  # pragma: no cover
    AggregatedThreatSummary = Any  # type: ignore
    ThreatIntelAggregator = Any  # type: ignore


logger = logging.getLogger("phishguard.reputation.domain")


# =============================================================================
# VERSION / LIMITS
# =============================================================================

MODULE_VERSION = "15.0.0"

DEFAULT_CACHE_TTL_SECONDS = 900
DEFAULT_CACHE_MAX_ENTRIES = 10_000

DEFAULT_MAX_INPUT_LENGTH = 2048
DEFAULT_MAX_DOMAIN_LENGTH = 253
DEFAULT_MAX_LABEL_LENGTH = 63

DEFAULT_MAX_SIGNALS = 64
DEFAULT_MAX_REASONS = 32
DEFAULT_MAX_PROVIDER_TAGS = 16
DEFAULT_MAX_TAG_LENGTH = 128

DEFAULT_MAX_DNS_ADDRESSES = 32
DEFAULT_MAX_CNAME_DEPTH = 8
DEFAULT_MAX_NAMESERVERS = 16
DEFAULT_MAX_MX_RECORDS = 16
DEFAULT_MAX_TXT_RECORDS = 16

DEFAULT_DNS_TIMEOUT_SECONDS = 2.0
DEFAULT_ANALYSIS_TIMEOUT_SECONDS = 8.0
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 4.0

DEFAULT_MAX_CONCURRENT_DNS = 64
DEFAULT_MAX_CONCURRENT_ANALYSIS = 256

DEFAULT_NEW_DOMAIN_DAYS = 30
DEFAULT_VERY_NEW_DOMAIN_DAYS = 7

DEFAULT_MAX_EDIT_DISTANCE = 2
DEFAULT_MAX_EDIT_DISTANCE_LENGTH = 32

DEFAULT_HIGH_DIGIT_COUNT = 5
DEFAULT_HIGH_HYPHEN_COUNT = 4

DEFAULT_ENTROPY_THRESHOLD = 3.8
DEFAULT_LONG_LABEL_LENGTH = 30

DEFAULT_MAX_LABEL_ENTROPY = 5.5


# =============================================================================
# RISK WEIGHTS
# =============================================================================

RISK_WEIGHTS: Mapping[str, float] = {
    "threat_intel": 0.65,
    "registration": 0.10,
    "dns": 0.08,
    "lexical": 0.10,
    "structural": 0.07,
}


# =============================================================================
# DEFAULT INTELLIGENCE
# =============================================================================

DEFAULT_SUSPICIOUS_TOKENS = frozenset(
    {
        "login",
        "signin",
        "sign-in",
        "verify",
        "verification",
        "secure",
        "account",
        "update",
        "confirm",
        "password",
        "credential",
        "credentials",
        "wallet",
        "billing",
        "payment",
        "invoice",
        "support",
        "unlock",
        "authenticate",
        "authentication",
    }
)


DEFAULT_BRAND_NAMES = frozenset(
    {
        "microsoft",
        "office365",
        "paypal",
        "apple",
        "appleid",
        "icloud",
        "google",
        "facebook",
        "instagram",
        "netflix",
        "amazon",
    }
)


DEFAULT_SUSPICIOUS_TLDS = frozenset(
    {
        "zip",
        "mov",
        "click",
        "top",
        "xyz",
        "work",
        "support",
        "country",
        "stream",
        "download",
        "review",
        "gq",
        "tk",
        "ml",
        "ga",
        "cf",
    }
)


DEFAULT_INFRASTRUCTURE_SUFFIXES = frozenset(
    {
        "cloudfront.net",
        "amazonaws.com",
        "azurewebsites.net",
        "github.io",
        "pages.dev",
        "workers.dev",
        "vercel.app",
        "netlify.app",
    }
)


# =============================================================================
# ENUMS
# =============================================================================

class DomainReputationStatus(str, Enum):
    COMPLETE = "complete"
    DEGRADED = "degraded"
    INVALID_INPUT = "invalid_input"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class DomainRiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CLEAN = "clean"
    UNKNOWN = "unknown"


class DomainEvidenceType(str, Enum):
    THREAT_INTEL = "threat_intel"
    REGISTRATION = "registration"
    DNS = "dns"
    LEXICAL = "lexical"
    STRUCTURAL = "structural"
    INPUT = "input"


class DNSFailureType(str, Enum):
    NONE = "none"
    NXDOMAIN = "nxdomain"
    TIMEOUT = "timeout"
    SERVFAIL = "servfail"
    REFUSED = "refused"
    NETWORK = "network"
    INVALID = "invalid"
    UNKNOWN = "unknown"


# =============================================================================
# EXCEPTIONS
# =============================================================================

class DomainReputationError(Exception):
    """Base domain reputation exception."""


class InvalidDomainError(DomainReputationError):
    """Invalid or unsafe domain input."""


class DomainResolutionError(DomainReputationError):
    """DNS resolution failure."""


class DomainReputationConfigurationError(DomainReputationError):
    """Invalid engine configuration."""


# =============================================================================
# HELPERS
# =============================================================================

def _bounded_text(
    value: Any,
    *,
    max_length: int,
) -> Optional[str]:
    """
    Convert arbitrary external text into bounded safe text.

    This is intentionally conservative. Provider responses must never be
    allowed to create unbounded evidence or reason strings.
    """

    if value is None:
        return None

    if not isinstance(value, str):
        value = str(value)

    value = unicodedata.normalize("NFC", value)
    value = value.replace("\x00", "")
    value = " ".join(value.split())

    if not value:
        return None

    return value[:max_length]


def _safe_float(
    value: Any,
    *,
    default: Optional[float] = None,
) -> Optional[float]:
    if value is None:
        return default

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return max(0.0, min(1.0, result))


def _unique_preserve_order(
    values: Iterable[str],
    *,
    limit: int,
) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()

    for value in values:
        if not isinstance(value, str):
            continue

        if value in seen:
            continue

        seen.add(value)
        result.append(value)

        if len(result) >= limit:
            break

    return result


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0

    counts: Dict[str, int] = {}

    for char in value:
        counts[char] = counts.get(char, 0) + 1

    length = len(value)

    entropy = 0.0

    for count in counts.values():
        probability = count / length

        entropy -= probability * math.log2(probability)

    return entropy


def _bounded_levenshtein(
    left: str,
    right: str,
    *,
    max_distance: int,
) -> Optional[int]:
    """
    Bounded Levenshtein distance.

    Returns None as soon as the distance is provably greater than the
    requested bound. This prevents pathological CPU usage on attacker-
    controlled labels.
    """

    if left == right:
        return 0

    if abs(len(left) - len(right)) > max_distance:
        return None

    if not left:
        return len(right) if len(right) <= max_distance else None

    if not right:
        return len(left) if len(left) <= max_distance else None

    if len(left) > len(right):
        left, right = right, left

    previous = list(range(len(left) + 1))

    for row_index, right_char in enumerate(right, start=1):
        current = [row_index]

        row_min = current[0]

        for col_index, left_char in enumerate(left, start=1):
            insertion = current[col_index - 1] + 1
            deletion = previous[col_index] + 1
            substitution = previous[col_index - 1] + (
                left_char != right_char
            )

            value = min(
                insertion,
                deletion,
                substitution,
            )

            current.append(value)
            row_min = min(row_min, value)

        if row_min > max_distance:
            return None

        previous = current

    distance = previous[-1]

    return distance if distance <= max_distance else None


# =============================================================================
# DATA MODELS
# =============================================================================

class DomainIdentity(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    original: str = Field(
        min_length=1,
        max_length=DEFAULT_MAX_INPUT_LENGTH,
    )

    ascii_domain: str = Field(
        min_length=1,
        max_length=DEFAULT_MAX_DOMAIN_LENGTH,
    )

    unicode_domain: str = Field(
        min_length=1,
        max_length=DEFAULT_MAX_DOMAIN_LENGTH,
    )

    registrable_domain: Optional[str] = Field(
        default=None,
        max_length=DEFAULT_MAX_DOMAIN_LENGTH,
    )

    public_suffix: Optional[str] = Field(
        default=None,
        max_length=DEFAULT_MAX_DOMAIN_LENGTH,
    )

    tld: Optional[str] = Field(
        default=None,
        max_length=DEFAULT_MAX_LABEL_LENGTH,
    )

    label_count: int = Field(
        default=1,
        ge=1,
    )

    is_idn: bool = False
    is_punycode: bool = False
    has_trailing_dot: bool = False
    is_subdomain: bool = False
    is_single_label: bool = False

    @property
    def cache_key(self) -> str:
        digest = hashlib.sha256(
            self.ascii_domain.encode("utf-8")
        ).hexdigest()

        return f"domain-reputation:{MODULE_VERSION}:{digest}"


class DNSObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    performed: bool = False

    resolver: Optional[str] = None

    a_records: List[str] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_DNS_ADDRESSES,
    )

    aaaa_records: List[str] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_DNS_ADDRESSES,
    )

    cname_chain: List[str] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_CNAME_DEPTH,
    )

    nameservers: List[str] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_NAMESERVERS,
    )

    mx_records: List[str] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_MX_RECORDS,
    )

    txt_records: List[str] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_TXT_RECORDS,
    )

    resolved: bool = False

    nx_domain: bool = False

    failure_type: DNSFailureType = DNSFailureType.NONE

    has_ipv4: bool = False
    has_ipv6: bool = False

    private_ip_detected: bool = False
    loopback_ip_detected: bool = False
    link_local_ip_detected: bool = False
    multicast_ip_detected: bool = False
    unspecified_ip_detected: bool = False

    error: Optional[str] = Field(
        default=None,
        max_length=128,
    )


class RegistrationObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool = False

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    registrar: Optional[str] = Field(
        default=None,
        max_length=256,
    )

    registry: Optional[str] = Field(
        default=None,
        max_length=256,
    )

    nameservers: List[str] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_NAMESERVERS,
    )

    age_days: Optional[int] = Field(
        default=None,
        ge=0,
    )

    days_until_expiry: Optional[int] = None

    newly_registered: bool = False
    very_new: bool = False

    short_expiry: bool = False
    privacy_protected: bool = False

    status: List[str] = Field(
        default_factory=list,
        max_length=32,
    )

    error: Optional[str] = Field(
        default=None,
        max_length=128,
    )


class DomainSignal(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    type: DomainEvidenceType

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

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    description: str = Field(
        min_length=1,
        max_length=256,
    )


class DomainReputationAssessment(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    engine_version: str = MODULE_VERSION

    status: DomainReputationStatus

    domain: DomainIdentity

    risk_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    risk_level: DomainRiskLevel

    severity: ThreatSeverity

    is_malicious: bool
    is_suspicious: bool

    reasons: List[str] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_REASONS,
    )

    signals: List[DomainSignal] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_SIGNALS,
    )

    dns: DNSObservation = Field(
        default_factory=DNSObservation,
    )

    registration: RegistrationObservation = Field(
        default_factory=RegistrationObservation,
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
    )

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    analysis_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    cache_hit: bool = False

    config_fingerprint: str = ""

    @property
    def cache_key(self) -> str:
        return self.domain.cache_key


# =============================================================================
# PUBLIC SUFFIX RESOLVER
# =============================================================================

class PublicSuffixResolver:
    """
    Public Suffix List abstraction.

    Production deployments should provide a PSL-backed implementation.

    `tldextract` is supported without network fetching at analysis time.
    The bundled PSL must be updated through the application's dependency
    / deployment process.

    The fallback is intentionally conservative and should only be used for
    environments where the PSL package is unavailable.
    """

    def __init__(
        self,
        *,
        use_private_suffixes: bool = True,
    ) -> None:
        self.use_private_suffixes = use_private_suffixes

        self._extractor = None

        try:
            import tldextract

            self._extractor = tldextract.TLDExtract(
                suffix_list_urls=(),
                include_psl_private_domains=use_private_suffixes,
            )
        except ImportError:
            self._extractor = None

    def extract(
        self,
        ascii_domain: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        if self._extractor is not None:
            result = self._extractor(ascii_domain)

            suffix = result.suffix or None

            if result.domain:
                registrable = (
                    f"{result.domain}.{suffix}"
                    if suffix
                    else result.domain
                )
            else:
                registrable = None

            return registrable, suffix

        # Minimal fallback.
        # This is deliberately not advertised as PSL-equivalent.
        labels = ascii_domain.split(".")

        if len(labels) < 2:
            return ascii_domain, labels[-1]

        return (
            ".".join(labels[-2:]),
            labels[-1],
        )


# =============================================================================
# CACHE
# =============================================================================

@dataclass
class _CacheEntry:
    value: DomainReputationAssessment
    expires_at: float
    created_at: float


class DomainReputationCache:
    """
    Bounded TTL + LRU cache.

    The cache is process-local by design.

    Multi-worker deployments should place a distributed cache above this
    component when cross-worker consistency is required.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        max_entries: int = DEFAULT_CACHE_MAX_ENTRIES,
    ) -> None:
        if ttl_seconds < 0:
            raise DomainReputationConfigurationError(
                "ttl_seconds must be >= 0"
            )

        if max_entries < 1:
            raise DomainReputationConfigurationError(
                "max_entries must be >= 1"
            )

        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

        self._hits = 0
        self._misses = 0

    async def get(
        self,
        key: str,
    ) -> Optional[DomainReputationAssessment]:
        async with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                self._misses += 1
                return None

            if (
                self.ttl_seconds <= 0
                or time.monotonic() >= entry.expires_at
            ):
                self._entries.pop(key, None)
                self._misses += 1
                return None

            self._entries.move_to_end(key)
            self._hits += 1

            return entry.value.model_copy(deep=True)

    async def set(
        self,
        key: str,
        value: DomainReputationAssessment,
    ) -> None:
        async with self._lock:
            if self.ttl_seconds <= 0:
                return

            now = time.monotonic()

            self._entries[key] = _CacheEntry(
                value=value.model_copy(deep=True),
                expires_at=now + self.ttl_seconds,
                created_at=now,
            )

            self._entries.move_to_end(key)

            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()

    async def stats(self) -> Mapping[str, int]:
        async with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
            }


# =============================================================================
# DOMAIN CANONICALIZER
# =============================================================================

class DomainCanonicalizer:
    """
    Strict domain canonicalization.

    Canonical representation:
        ASCII A-label / lowercase / no trailing dot.

    Unicode representation:
        Normalized U-label representation.

    No network operations occur here.
    """

    DOMAIN_LABEL_RE = re.compile(
        r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$",
        re.IGNORECASE,
    )

    @classmethod
    def canonicalize(
        cls,
        value: str,
        *,
        suffix_resolver: Optional[PublicSuffixResolver] = None,
        allow_single_label: bool = False,
    ) -> DomainIdentity:
        if not isinstance(value, str):
            raise InvalidDomainError(
                "Domain must be a string."
            )

        original = value

        if len(value) > DEFAULT_MAX_INPUT_LENGTH:
            raise InvalidDomainError(
                "Domain input exceeds maximum length."
            )

        value = unicodedata.normalize(
            "NFC",
            value,
        ).strip()

        if not value:
            raise InvalidDomainError(
                "Domain cannot be empty."
            )

        # -------------------------------------------------------------
        # URL input
        # -------------------------------------------------------------

        if "://" in value:
            parsed = urlparse(value)

            if parsed.scheme.lower() not in {
                "http",
                "https",
            }:
                raise InvalidDomainError(
                    "Only HTTP/HTTPS URLs are accepted."
                )

            if parsed.username or parsed.password:
                raise InvalidDomainError(
                    "Credential-bearing URLs are not accepted."
                )

            if parsed.fragment:
                # Fragment never reaches a server but accepting it here
                # would make canonicalization semantics ambiguous.
                raise InvalidDomainError(
                    "URL fragments are not accepted."
                )

            host = parsed.hostname

            if not host:
                raise InvalidDomainError(
                    "URL does not contain a hostname."
                )

            value = host

        else:
            # Hostname-only input.
            if any(
                character in value
                for character in "/?#@"
            ):
                raise InvalidDomainError(
                    "Input contains path, query, fragment, or credentials."
                )

        # -------------------------------------------------------------
        # IP rejection
        # -------------------------------------------------------------

        ip_candidate = value.strip("[]")

        try:
            ipaddress.ip_address(
                ip_candidate.rstrip(".")
            )
        except ValueError:
            pass
        else:
            raise InvalidDomainError(
                "IP addresses must use the IP reputation engine."
            )

        had_trailing_dot = value.endswith(".")

        value = value.rstrip(".")

        if not value:
            raise InvalidDomainError(
                "Invalid root domain."
            )

        # Reject control characters.
        if any(
            unicodedata.category(ch) == "Cc"
            for ch in value
        ):
            raise InvalidDomainError(
                "Domain contains control characters."
            )

        # -------------------------------------------------------------
        # IDNA conversion
        # -------------------------------------------------------------

        try:
            ascii_domain = (
                value.encode("idna")
                .decode("ascii")
                .lower()
            )
        except UnicodeError as exc:
            raise InvalidDomainError(
                "Domain could not be IDNA encoded."
            ) from exc

        ascii_domain = ascii_domain.rstrip(".")

        if (
            not ascii_domain
            or len(ascii_domain)
            > DEFAULT_MAX_DOMAIN_LENGTH
        ):
            raise InvalidDomainError(
                "Domain exceeds DNS length limits."
            )

        labels = ascii_domain.split(".")

        if (
            not allow_single_label
            and len(labels) < 2
        ):
            raise InvalidDomainError(
                "Single-label domains are not accepted."
            )

        for label in labels:
            if not label:
                raise InvalidDomainError(
                    "Domain contains an empty label."
                )

            if len(label) > DEFAULT_MAX_LABEL_LENGTH:
                raise InvalidDomainError(
                    "Domain label exceeds 63 characters."
                )

            if not cls.DOMAIN_LABEL_RE.fullmatch(label):
                raise InvalidDomainError(
                    f"Invalid domain label: {label!r}"
                )

        # -------------------------------------------------------------
        # IDNA round-trip verification
        # -------------------------------------------------------------

        try:
            unicode_domain = (
                ascii_domain.encode("ascii")
                .decode("idna")
            )

            round_trip = (
                unicode_domain.encode("idna")
                .decode("ascii")
                .lower()
            )
        except UnicodeError as exc:
            raise InvalidDomainError(
                "Invalid IDNA round-trip."
            ) from exc

        if round_trip != ascii_domain:
            raise InvalidDomainError(
                "IDNA canonicalization is not stable."
            )

        # -------------------------------------------------------------
        # PSL
        # -------------------------------------------------------------

        suffix_resolver = (
            suffix_resolver
            or PublicSuffixResolver()
        )

        registrable_domain, public_suffix = (
            suffix_resolver.extract(ascii_domain)
        )

        if not registrable_domain:
            raise InvalidDomainError(
                "Unable to determine registrable domain."
            )

        labels = ascii_domain.split(".")

        return DomainIdentity(
            original=original,
            ascii_domain=ascii_domain,
            unicode_domain=unicode_domain,
            registrable_domain=registrable_domain,
            public_suffix=public_suffix,
            tld=labels[-1] if labels else None,
            label_count=len(labels),
            is_idn=ascii_domain != unicode_domain,
            is_punycode=any(
                label.startswith("xn--")
                for label in labels
            ),
            has_trailing_dot=had_trailing_dot,
            is_subdomain=(
                ascii_domain != registrable_domain
            ),
            is_single_label=len(labels) == 1,
        )


# =============================================================================
# UNICODE / HOMOGRAPH ANALYSIS
# =============================================================================

def _script_of_character(character: str) -> str:
    """
    Lightweight script classification.

    This intentionally avoids third-party Unicode security dependencies.
    """

    codepoint = ord(character)

    if "a" <= character.lower() <= "z":
        return "latin"

    if 0x0400 <= codepoint <= 0x04FF:
        return "cyrillic"

    if 0x0370 <= codepoint <= 0x03FF:
        return "greek"

    if 0x4E00 <= codepoint <= 0x9FFF:
        return "han"

    if 0x3040 <= codepoint <= 0x30FF:
        return "japanese"

    if 0xAC00 <= codepoint <= 0xD7AF:
        return "hangul"

    if character.isdigit():
        return "digit"

    if character in ".-":
        return "punctuation"

    return "other"


class UnicodeRiskAnalyzer:

    @staticmethod
    def analyze(
        identity: DomainIdentity,
    ) -> List[DomainSignal]:
        signals: List[DomainSignal] = []

        unicode_domain = identity.unicode_domain

        scripts = {
            _script_of_character(ch)
            for ch in unicode_domain
            if ch.isalpha()
        }

        scripts.discard("other")

        if len(scripts) >= 2:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.LEXICAL,
                    name="mixed_unicode_scripts",
                    value=0.65,
                    weight=0.70,
                    confidence=0.90,
                    description=(
                        "Domain contains characters from multiple "
                        "writing systems."
                    ),
                )
            )

        if identity.is_punycode:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.LEXICAL,
                    name="idna_punycode",
                    value=0.15,
                    weight=0.15,
                    confidence=0.95,
                    description=(
                        "Domain contains an IDNA/Punycode label."
                    ),
                )
            )

        # A simple confusable-character detector for common Latin/Cyrillic
        # lookalikes. This is evidence, not a standalone verdict.
        confusable_pairs = {
            ("а", "a"),
            ("е", "e"),
            ("о", "o"),
            ("р", "p"),
            ("с", "c"),
            ("х", "x"),
            ("у", "y"),
        }

        confusable_found = False

        for character in unicode_domain.lower():
            if any(
                character == left
                for left, _ in confusable_pairs
            ):
                confusable_found = True
                break

        if confusable_found:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.LEXICAL,
                    name="unicode_confusable_character",
                    value=0.70,
                    weight=0.80,
                    confidence=0.85,
                    description=(
                        "Domain contains a Unicode character with a "
                        "common visual confusable."
                    ),
                )
            )

        return signals


# =============================================================================
# LEXICAL ANALYZER
# =============================================================================

class DomainLexicalAnalyzer:

    def __init__(
        self,
        *,
        suspicious_tokens: Iterable[str] = DEFAULT_SUSPICIOUS_TOKENS,
        brand_names: Iterable[str] = DEFAULT_BRAND_NAMES,
        suspicious_tlds: Iterable[str] = DEFAULT_SUSPICIOUS_TLDS,
        max_edit_distance: int = DEFAULT_MAX_EDIT_DISTANCE,
    ) -> None:
        self.suspicious_tokens = frozenset(
            token.lower()
            for token in suspicious_tokens
            if token
        )

        self.brand_names = frozenset(
            brand.lower()
            for brand in brand_names
            if brand
        )

        self.suspicious_tlds = frozenset(
            tld.lower().lstrip(".")
            for tld in suspicious_tlds
            if tld
        )

        self.max_edit_distance = max(
            0,
            min(max_edit_distance, 3),
        )

    @staticmethod
    def _tokenize_label(
        label: str,
    ) -> List[str]:
        """
        Tokenization avoids naive substring matches.

        Example:
            insecure-login
        becomes:
            ["insecure", "login"]
        """

        pieces = re.split(
            r"[-_.]+",
            label.lower(),
        )

        return [
            piece
            for piece in pieces
            if piece
        ]

    def analyze(
        self,
        identity: DomainIdentity,
    ) -> List[DomainSignal]:
        signals: List[DomainSignal] = []

        labels = identity.ascii_domain.split(".")

        tokens: List[str] = []

        for label in labels:
            tokens.extend(
                self._tokenize_label(label)
            )

        # -------------------------------------------------------------
        # Suspicious semantic tokens
        # -------------------------------------------------------------

        matched_tokens = sorted(
            set(tokens).intersection(
                self.suspicious_tokens
            )
        )

        if matched_tokens:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.LEXICAL,
                    name="suspicious_semantic_tokens",
                    value=min(
                        1.0,
                        0.18 * len(matched_tokens),
                    ),
                    weight=0.50,
                    confidence=0.90,
                    description=(
                        "Domain contains authentication, account, "
                        "payment, or other security-sensitive terms."
                    ),
                )
            )

        # -------------------------------------------------------------
        # Exact brand token
        # -------------------------------------------------------------

        brand_hits = sorted(
            set(tokens).intersection(
                self.brand_names
            )
        )

        if brand_hits:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.LEXICAL,
                    name="brand_token",
                    value=min(
                        1.0,
                        0.45 + 0.10 * (
                            len(brand_hits) - 1
                        ),
                    ),
                    weight=0.65,
                    confidence=0.80,
                    description=(
                        "Domain contains a recognized brand token."
                    ),
                )
            )

        # -------------------------------------------------------------
        # Brand typo-squatting
        # -------------------------------------------------------------

        if (
            self.max_edit_distance > 0
            and len(tokens) <= 32
        ):
            for token in tokens:
                if not token:
                    continue

                if len(token) > DEFAULT_MAX_EDIT_DISTANCE_LENGTH:
                    continue

                for brand in self.brand_names:
                    if token == brand:
                        continue

                    if abs(
                        len(token) - len(brand)
                    ) > self.max_edit_distance:
                        continue

                    distance = _bounded_levenshtein(
                        token,
                        brand,
                        max_distance=self.max_edit_distance,
                    )

                    if distance is not None:
                        signals.append(
                            DomainSignal(
                                type=DomainEvidenceType.LEXICAL,
                                name="possible_brand_typosquat",
                                value=0.75
                                if distance == 1
                                else 0.60,
                                weight=0.75,
                                confidence=0.75,
                                description=(
                                    "A domain label is close to a "
                                    "known brand name by edit distance."
                                ),
                            )
                        )
                        break

        # -------------------------------------------------------------
        # Suspicious TLD
        # -------------------------------------------------------------

        if (
            identity.tld
            and identity.tld.lower()
            in self.suspicious_tlds
        ):
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.LEXICAL,
                    name="higher_risk_tld",
                    value=0.30,
                    weight=0.25,
                    confidence=0.70,
                    description=(
                        "The TLD is associated with elevated abuse "
                        "observations; this is supporting evidence only."
                    ),
                )
            )

        # -------------------------------------------------------------
        # Digits
        # -------------------------------------------------------------

        digit_count = sum(
            char.isdigit()
            for char in identity.ascii_domain
        )

        if digit_count >= DEFAULT_HIGH_DIGIT_COUNT:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.LEXICAL,
                    name="high_digit_density",
                    value=min(
                        0.60,
                        digit_count / 20.0,
                    ),
                    weight=0.20,
                    confidence=0.70,
                    description=(
                        "Domain contains an unusually high number "
                        "of numeric characters."
                    ),
                )
            )

        # -------------------------------------------------------------
        # Hyphens
        # -------------------------------------------------------------

        hyphen_count = identity.ascii_domain.count("-")

        if hyphen_count >= DEFAULT_HIGH_HYPHEN_COUNT:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.LEXICAL,
                    name="high_hyphen_density",
                    value=min(
                        0.60,
                        hyphen_count / 12.0,
                    ),
                    weight=0.15,
                    confidence=0.70,
                    description=(
                        "Domain contains an unusually high number "
                        "of hyphens."
                    ),
                )
            )

        return signals


# =============================================================================
# DNS RESOLVER
# =============================================================================

DNSResolver = Callable[
    [str],
    Awaitable[DNSObservation],
]


class SafeDNSResolver:
    """
    Bounded DNS resolver.

    Preferred backend:
        dnspython

    Fallback:
        asyncio.get_running_loop().getaddrinfo()

    The fallback is deliberately limited to A/AAAA observations.

    Python's asynchronous getaddrinfo uses the event loop's default thread
    pool internally, therefore a concurrency semaphore is used here to avoid
    allowing attacker-controlled domain inputs to consume the executor
    without bound. :contentReference[oaicite:2]{index=2}
    """

    def __init__(
        self,
        *,
        max_addresses: int = DEFAULT_MAX_DNS_ADDRESSES,
        max_cname_depth: int = DEFAULT_MAX_CNAME_DEPTH,
        timeout_seconds: float = DEFAULT_DNS_TIMEOUT_SECONDS,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT_DNS,
        include_ns: bool = True,
        include_mx: bool = True,
        include_txt: bool = False,
    ) -> None:
        if max_addresses < 1:
            raise DomainReputationConfigurationError(
                "max_addresses must be >= 1"
            )

        if max_cname_depth < 0:
            raise DomainReputationConfigurationError(
                "max_cname_depth must be >= 0"
            )

        if timeout_seconds <= 0:
            raise DomainReputationConfigurationError(
                "timeout_seconds must be > 0"
            )

        self.max_addresses = min(
            max_addresses,
            256,
        )

        self.max_cname_depth = min(
            max_cname_depth,
            32,
        )

        self.timeout_seconds = timeout_seconds

        self._semaphore = asyncio.Semaphore(
            max(1, max_concurrent)
        )

        self.include_ns = include_ns
        self.include_mx = include_mx
        self.include_txt = include_txt

        try:
            import dns.asyncresolver

            self._dns_available = True
        except ImportError:
            self._dns_available = False

    async def resolve(
        self,
        domain: str,
    ) -> DNSObservation:
        async with self._semaphore:
            try:
                async with asyncio.timeout(
                    self.timeout_seconds
                ):
                    if self._dns_available:
                        return await self._resolve_dnspython(
                            domain
                        )

                    return await self._resolve_socket(
                        domain
                    )

            except TimeoutError:
                return DNSObservation(
                    performed=True,
                    resolver="bounded",
                    failure_type=DNSFailureType.TIMEOUT,
                    error="dns_timeout",
                )

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                logger.warning(
                    "Domain DNS resolution failed",
                    extra={
                        "domain": domain,
                        "error_type": type(exc).__name__,
                    },
                )

                return DNSObservation(
                    performed=True,
                    resolver="bounded",
                    failure_type=DNSFailureType.UNKNOWN,
                    error=(
                        f"dns_error:{type(exc).__name__}"
                    ),
                )

    async def _resolve_dnspython(
        self,
        domain: str,
    ) -> DNSObservation:
        import dns.asyncresolver
        import dns.exception
        import dns.resolver

        resolver = dns.asyncresolver.Resolver()

        resolver.timeout = self.timeout_seconds
        resolver.lifetime = self.timeout_seconds

        observation = DNSObservation(
            performed=True,
            resolver="dnspython",
        )

        # -------------------------------------------------------------
        # A / AAAA
        # -------------------------------------------------------------

        async def query(
            record_type: str,
        ) -> Any:
            return await resolver.resolve(
                domain,
                record_type,
                raise_on_no_answer=False,
            )

        for record_type in ("A", "AAAA"):
            try:
                answer = await query(record_type)

                for item in answer:
                    address = str(item)

                    if record_type == "A":
                        if address not in observation.a_records:
                            observation.a_records.append(
                                address
                            )
                    else:
                        if address not in observation.aaaa_records:
                            observation.aaaa_records.append(
                                address
                            )

                    if (
                        len(observation.a_records)
                        + len(observation.aaaa_records)
                        >= self.max_addresses
                    ):
                        break

            except dns.resolver.NXDOMAIN:
                observation.nx_domain = True
                observation.failure_type = (
                    DNSFailureType.NXDOMAIN
                )
                break

            except dns.resolver.NoAnswer:
                continue

            except dns.exception.Timeout:
                observation.failure_type = (
                    DNSFailureType.TIMEOUT
                )
                observation.error = "dns_timeout"
                break

            except dns.resolver.NoNameservers:
                observation.failure_type = (
                    DNSFailureType.SERVFAIL
                )
                observation.error = "dns_no_nameservers"
                break

            except Exception as exc:
                observation.failure_type = (
                    DNSFailureType.UNKNOWN
                )
                observation.error = (
                    f"dns_query:{type(exc).__name__}"
                )
                break

        # -------------------------------------------------------------
        # CNAME chain
        # -------------------------------------------------------------

        if self.max_cname_depth > 0:
            current = domain

            for _ in range(
                self.max_cname_depth
            ):
                try:
                    answer = await resolver.resolve(
                        current,
                        "CNAME",
                        raise_on_no_answer=False,
                    )

                    if not answer:
                        break

                    target = str(
                        answer[0]
                    ).rstrip(".").lower()

                    if not target:
                        break

                    if target in observation.cname_chain:
                        break

                    observation.cname_chain.append(
                        target
                    )

                    current = target

                except (
                    dns.resolver.NoAnswer,
                    dns.resolver.NXDOMAIN,
                ):
                    break

                except Exception:
                    break

        # -------------------------------------------------------------
        # NS
        # -------------------------------------------------------------

        if self.include_ns:
            try:
                answer = await resolver.resolve(
                    domain,
                    "NS",
                    raise_on_no_answer=False,
                )

                for item in answer:
                    if len(observation.nameservers) >= (
                        DEFAULT_MAX_NAMESERVERS
                    ):
                        break

                    value = str(item).rstrip(".").lower()

                    if value not in observation.nameservers:
                        observation.nameservers.append(
                            value
                        )

            except Exception:
                pass

        # -------------------------------------------------------------
        # MX
        # -------------------------------------------------------------

        if self.include_mx:
            try:
                answer = await resolver.resolve(
                    domain,
                    "MX",
                    raise_on_no_answer=False,
                )

                for item in answer:
                    if len(observation.mx_records) >= (
                        DEFAULT_MAX_MX_RECORDS
                    ):
                        break

                    value = str(
                        getattr(
                            item,
                            "exchange",
                            item,
                        )
                    ).rstrip(".").lower()

                    if value not in observation.mx_records:
                        observation.mx_records.append(
                            value
                        )

            except Exception:
                pass

        # -------------------------------------------------------------
        # TXT
        # -------------------------------------------------------------

        if self.include_txt:
            try:
                answer = await resolver.resolve(
                    domain,
                    "TXT",
                    raise_on_no_answer=False,
                )

                for item in answer:
                    if len(observation.txt_records) >= (
                        DEFAULT_MAX_TXT_RECORDS
                    ):
                        break

                    value = str(item)

                    if len(value) > 512:
                        value = value[:512]

                    observation.txt_records.append(
                        value
                    )

            except Exception:
                pass

        self._classify_addresses(observation)

        observation.resolved = bool(
            observation.a_records
            or observation.aaaa_records
            or observation.cname_chain
        )

        return observation

    async def _resolve_socket(
        self,
        domain: str,
    ) -> DNSObservation:
        observation = DNSObservation(
            performed=True,
            resolver="system",
        )

        loop = asyncio.get_running_loop()

        try:
            results = await loop.getaddrinfo(
                domain,
                443,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
            )

        except socket.gaierror as exc:
            observation.nx_domain = True
            observation.failure_type = (
                DNSFailureType.NXDOMAIN
            )
            observation.error = (
                f"getaddrinfo:{getattr(exc, 'errno', 'unknown')}"
            )
            return observation

        except OSError as exc:
            observation.failure_type = (
                DNSFailureType.NETWORK
            )
            observation.error = (
                f"getaddrinfo:{type(exc).__name__}"
            )
            return observation

        for result in results:
            sockaddr = result[4]

            if not sockaddr:
                continue

            address = str(
                sockaddr[0]
            )

            try:
                parsed = ipaddress.ip_address(
                    address
                )
            except ValueError:
                continue

            if isinstance(
                parsed,
                ipaddress.IPv4Address,
            ):
                if address not in observation.a_records:
                    observation.a_records.append(
                        address
                    )
            else:
                if address not in observation.aaaa_records:
                    observation.aaaa_records.append(
                        address
                    )

            if (
                len(observation.a_records)
                + len(observation.aaaa_records)
                >= self.max_addresses
            ):
                break

        self._classify_addresses(
            observation
        )

        observation.resolved = bool(
            observation.a_records
            or observation.aaaa_records
        )

        return observation

    @staticmethod
    def _classify_addresses(
        observation: DNSObservation,
    ) -> None:
        for address in [
            *observation.a_records,
            *observation.aaaa_records,
        ]:
            try:
                parsed = ipaddress.ip_address(
                    address
                )
            except ValueError:
                continue

            observation.private_ip_detected |= (
                parsed.is_private
            )

            observation.loopback_ip_detected |= (
                parsed.is_loopback
            )

            observation.link_local_ip_detected |= (
                parsed.is_link_local
            )

            observation.multicast_ip_detected |= (
                parsed.is_multicast
            )

            observation.unspecified_ip_detected |= (
                parsed.is_unspecified
            )


# =============================================================================
# REGISTRATION CONTRACT
# =============================================================================

class RegistrationProvider:
    """
    Provider contract.

    Implementations should normalize RDAP/WHOIS information into
    RegistrationObservation before returning it.

    RDAP domain queries use the standardized domain query model defined
    by RFC 9082. :contentReference[oaicite:3]{index=3}
    """

    async def lookup(
        self,
        domain: str,
    ) -> RegistrationObservation:
        raise NotImplementedError


# =============================================================================
# REGISTRATION SIGNAL ANALYZER
# =============================================================================

class RegistrationSignalAnalyzer:

    def __init__(
        self,
        *,
        new_domain_days: int = DEFAULT_NEW_DOMAIN_DAYS,
        very_new_domain_days: int = DEFAULT_VERY_NEW_DOMAIN_DAYS,
    ) -> None:
        if new_domain_days < 1:
            raise DomainReputationConfigurationError(
                "new_domain_days must be >= 1"
            )

        if very_new_domain_days < 0:
            raise DomainReputationConfigurationError(
                "very_new_domain_days must be >= 0"
            )

        self.new_domain_days = new_domain_days

        self.very_new_domain_days = min(
            very_new_domain_days,
            new_domain_days,
        )

    def normalize(
        self,
        observation: RegistrationObservation,
    ) -> RegistrationObservation:
        """
        Derive age and expiry fields when timestamps are available.

        Future creation dates are rejected rather than producing negative
        domain ages.
        """

        now = datetime.now(timezone.utc)

        created_at = observation.created_at

        if created_at is not None:
            if created_at.tzinfo is None:
                created_at = created_at.replace(
                    tzinfo=timezone.utc
                )

            age_seconds = (
                now - created_at
            ).total_seconds()

            if age_seconds >= 0:
                observation.age_days = int(
                    age_seconds // 86400
                )

                observation.very_new = (
                    observation.age_days
                    <= self.very_new_domain_days
                )

                observation.newly_registered = (
                    observation.age_days
                    <= self.new_domain_days
                )
            else:
                observation.age_days = None
                observation.very_new = False
                observation.newly_registered = False

        expires_at = observation.expires_at

        if expires_at is not None:
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(
                    tzinfo=timezone.utc
                )

            seconds_until_expiry = (
                expires_at - now
            ).total_seconds()

            observation.days_until_expiry = int(
                seconds_until_expiry // 86400
            )

            observation.short_expiry = (
                0
                <= observation.days_until_expiry
                <= 30
            )

        return observation

    def analyze(
        self,
        registration: RegistrationObservation,
    ) -> List[DomainSignal]:
        registration = self.normalize(
            registration
        )

        signals: List[DomainSignal] = []

        if registration.very_new:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.REGISTRATION,
                    name="very_new_domain",
                    value=0.65,
                    weight=0.80,
                    confidence=0.90,
                    description=(
                        "Domain registration age is within the configured "
                        "very-new window."
                    ),
                )
            )

        elif registration.newly_registered:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.REGISTRATION,
                    name="newly_registered",
                    value=0.40,
                    weight=0.70,
                    confidence=0.90,
                    description=(
                        "Domain was registered recently."
                    ),
                )
            )

        if registration.short_expiry:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.REGISTRATION,
                    name="short_registration_expiry",
                    value=0.20,
                    weight=0.20,
                    confidence=0.75,
                    description=(
                        "Domain has a relatively short remaining "
                        "registration period."
                    ),
                )
            )

        if (
            registration.available
            and registration.created_at is None
        ):
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.REGISTRATION,
                    name="registration_date_unavailable",
                    value=0.05,
                    weight=0.10,
                    confidence=0.60,
                    description=(
                        "Registration data is available but the creation "
                        "date is missing."
                    ),
                )
            )

        return signals


# =============================================================================
# ENGINE CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class DomainReputationConfig:
    """
    Immutable engine configuration.

    The fingerprint is included in cache identity so configuration changes
    cannot accidentally reuse assessments calculated under a different
    policy.
    """

    enable_dns: bool = True
    enable_registration: bool = True

    analysis_timeout_seconds: float = (
        DEFAULT_ANALYSIS_TIMEOUT_SECONDS
    )

    provider_timeout_seconds: float = (
        DEFAULT_PROVIDER_TIMEOUT_SECONDS
    )

    allow_single_label: bool = False

    max_signals: int = DEFAULT_MAX_SIGNALS
    max_reasons: int = DEFAULT_MAX_REASONS

    suspicious_tokens: Tuple[str, ...] = tuple(
        sorted(DEFAULT_SUSPICIOUS_TOKENS)
    )

    brand_names: Tuple[str, ...] = tuple(
        sorted(DEFAULT_BRAND_NAMES)
    )

    suspicious_tlds: Tuple[str, ...] = tuple(
        sorted(DEFAULT_SUSPICIOUS_TLDS)
    )

    infrastructure_suffixes: Tuple[str, ...] = tuple(
        sorted(DEFAULT_INFRASTRUCTURE_SUFFIXES)
    )

    max_edit_distance: int = DEFAULT_MAX_EDIT_DISTANCE

    @property
    def fingerprint(self) -> str:
        raw = "|".join(
            [
                MODULE_VERSION,
                str(self.enable_dns),
                str(self.enable_registration),
                str(self.analysis_timeout_seconds),
                str(self.provider_timeout_seconds),
                str(self.allow_single_label),
                str(self.max_signals),
                str(self.max_reasons),
                str(self.max_edit_distance),
                ",".join(self.suspicious_tokens),
                ",".join(self.brand_names),
                ",".join(self.suspicious_tlds),
                ",".join(self.infrastructure_suffixes),
            ]
        )

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()[:32]


# =============================================================================
# ENGINE
# =============================================================================

class DomainReputationEngine:

    def __init__(
        self,
        threat_intel: Optional[
            "ThreatIntelAggregator"
        ] = None,
        registration_provider: Optional[
            RegistrationProvider
        ] = None,
        dns_resolver: Optional[
            DNSResolver
        ] = None,
        *,
        enable_dns: bool = True,
        enable_registration: bool = True,
        cache: Optional[
            DomainReputationCache
        ] = None,
        config: Optional[
            DomainReputationConfig
        ] = None,
        suffix_resolver: Optional[
            PublicSuffixResolver
        ] = None,
        new_domain_days: int = DEFAULT_NEW_DOMAIN_DAYS,
        very_new_domain_days: int = DEFAULT_VERY_NEW_DOMAIN_DAYS,
    ) -> None:

        self.config = config or DomainReputationConfig(
            enable_dns=enable_dns,
            enable_registration=(
                enable_registration
            ),
        )

        if self.config.analysis_timeout_seconds <= 0:
            raise DomainReputationConfigurationError(
                "analysis_timeout_seconds must be > 0"
            )

        if self.config.provider_timeout_seconds <= 0:
            raise DomainReputationConfigurationError(
                "provider_timeout_seconds must be > 0"
            )

        self.threat_intel = threat_intel

        self.registration_provider = (
            registration_provider
        )

        self.dns_resolver = (
            dns_resolver
            or SafeDNSResolver().resolve
        )

        self.cache = cache

        self.suffix_resolver = (
            suffix_resolver
            or PublicSuffixResolver()
        )

        self.enable_dns = (
            self.config.enable_dns
        )

        self.enable_registration = (
            self.config.enable_registration
            and registration_provider is not None
        )

        self.lexical_analyzer = (
            DomainLexicalAnalyzer(
                suspicious_tokens=(
                    self.config.suspicious_tokens
                ),
                brand_names=(
                    self.config.brand_names
                ),
                suspicious_tlds=(
                    self.config.suspicious_tlds
                ),
                max_edit_distance=(
                    self.config.max_edit_distance
                ),
            )
        )

        self.registration_analyzer = (
            RegistrationSignalAnalyzer(
                new_domain_days=new_domain_days,
                very_new_domain_days=very_new_domain_days,
            )
        )

        self._analysis_semaphore = asyncio.Semaphore(
            DEFAULT_MAX_CONCURRENT_ANALYSIS
        )

        # In-flight request deduplication prevents a cache stampede when
        # many workers ask for the same uncached domain simultaneously.
        self._inflight: Dict[
            str,
            asyncio.Task[
                DomainReputationAssessment
            ],
        ] = {}

        self._inflight_lock = asyncio.Lock()

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def analyze(
        self,
        domain: str,
        *,
        force_refresh: bool = False,
        resolve_dns: Optional[bool] = None,
        lookup_registration: Optional[bool] = None,
    ) -> DomainReputationAssessment:

        started = time.perf_counter()

        identity = DomainCanonicalizer.canonicalize(
            domain,
            suffix_resolver=self.suffix_resolver,
            allow_single_label=(
                self.config.allow_single_label
            ),
        )

        use_dns = (
            self.enable_dns
            if resolve_dns is None
            else bool(resolve_dns)
        )

        use_registration = (
            self.enable_registration
            if lookup_registration is None
            else bool(lookup_registration)
        )

        cache_key = self._build_cache_key(
            identity,
            use_dns=use_dns,
            use_registration=use_registration,
        )

        if self.cache and not force_refresh:
            cached = await self.cache.get(
                cache_key
            )

            if cached is not None:
                cached.cache_hit = True
                return cached

        async with self._inflight_lock:
            existing = (
                self._inflight.get(cache_key)
                if not force_refresh
                else None
            )

            if existing is not None:
                result = await asyncio.shield(
                    existing
                )

                result.cache_hit = True
                return result

            task = asyncio.create_task(
                self._analyze_uncached(
                    identity,
                    use_dns=use_dns,
                    use_registration=(
                        use_registration
                    ),
                    force_refresh=force_refresh,
                    started=started,
                )
            )

            self._inflight[cache_key] = task

        try:
            return await asyncio.shield(task)

        finally:
            async with self._inflight_lock:
                current = self._inflight.get(
                    cache_key
                )

                if current is task:
                    self._inflight.pop(
                        cache_key,
                        None,
                    )

    # =========================================================================
    # UNCACHED ANALYSIS
    # =========================================================================

    async def _analyze_uncached(
        self,
        identity: DomainIdentity,
        *,
        use_dns: bool,
        use_registration: bool,
        force_refresh: bool,
        started: float,
    ) -> DomainReputationAssessment:

        async with self._analysis_semaphore:
            try:
                async with asyncio.timeout(
                    self.config.analysis_timeout_seconds
                ):
                    return await self._perform_analysis(
                        identity,
                        use_dns=use_dns,
                        use_registration=(
                            use_registration
                        ),
                        force_refresh=force_refresh,
                        started=started,
                    )

            except TimeoutError:
                elapsed_ms = (
                    time.perf_counter()
                    - started
                ) * 1000.0

                assessment = (
                    self._timeout_assessment(
                        identity,
                        elapsed_ms,
                    )
                )

                return assessment

            except asyncio.CancelledError:
                raise

    async def _perform_analysis(
        self,
        identity: DomainIdentity,
        *,
        use_dns: bool,
        use_registration: bool,
        force_refresh: bool,
        started: float,
    ) -> DomainReputationAssessment:

        degraded_components: List[str] = []

        # -------------------------------------------------------------
        # Local signals
        # -------------------------------------------------------------

        signals: List[DomainSignal] = []

        signals.extend(
            self.lexical_analyzer.analyze(
                identity
            )
        )

        signals.extend(
            UnicodeRiskAnalyzer.analyze(
                identity
            )
        )

        signals.extend(
            self._analyze_structural_signals(
                identity
            )
        )

        # -------------------------------------------------------------
        # External tasks
        # -------------------------------------------------------------

        tasks: Dict[
            str,
            asyncio.Task[Any],
        ] = {}

        if use_dns and self.dns_resolver:
            tasks["dns"] = asyncio.create_task(
                self.dns_resolver(
                    identity.ascii_domain
                )
            )

        if (
            use_registration
            and self.registration_provider
        ):
            tasks["registration"] = asyncio.create_task(
                self.registration_provider.lookup(
                    identity.ascii_domain
                )
            )

        if self.threat_intel:
            tasks["threat_intel"] = asyncio.create_task(
                self.threat_intel.analyze_ioc(
                    identity.ascii_domain,
                    IOCType.DOMAIN,
                    force_refresh=force_refresh,
                )
            )

        # -------------------------------------------------------------
        # Isolated collection
        # -------------------------------------------------------------

        observations: Dict[
            str,
            Any,
        ] = {}

        for name, task in tasks.items():
            try:
                if name == "threat_intel":
                    observations[name] = await asyncio.wait_for(
                        asyncio.shield(task),
                        timeout=(
                            self.config.provider_timeout_seconds
                        ),
                    )
                else:
                    observations[name] = await task

            except TimeoutError:
                degraded_components.append(
                    name
                )

                task.cancel()

                observations[name] = None

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                degraded_components.append(
                    name
                )

                logger.warning(
                    "Domain reputation component failed",
                    extra={
                        "component": name,
                        "error_type": (
                            type(exc).__name__
                        ),
                    },
                )

                observations[name] = None

        # -------------------------------------------------------------
        # DNS
        # -------------------------------------------------------------

        dns = observations.get(
            "dns"
        )

        if not isinstance(
            dns,
            DNSObservation,
        ):
            dns = DNSObservation(
                performed=False
                if "dns" not in tasks
                else True,
                error=(
                    "component_unavailable"
                    if "dns" in tasks
                    else None
                ),
            )

        signals.extend(
            self._analyze_dns_signals(
                dns
            )
        )

        # -------------------------------------------------------------
        # Registration
        # -------------------------------------------------------------

        registration = observations.get(
            "registration"
        )

        if not isinstance(
            registration,
            RegistrationObservation,
        ):
            registration = RegistrationObservation(
                available=False,
                error=(
                    "component_unavailable"
                    if "registration" in tasks
                    else None
                ),
            )

        signals.extend(
            self.registration_analyzer.analyze(
                registration
            )
        )

        # -------------------------------------------------------------
        # Threat intelligence
        # -------------------------------------------------------------

        threat_report = observations.get(
            "threat_intel"
        )

        threat_score: Optional[float] = None

        providers_queried = 0
        providers_successful = 0

        if threat_report is not None:
            try:
                threat_score = _safe_float(
                    getattr(
                        threat_report,
                        "aggregated_risk_score",
                        None,
                    )
                )

                providers_queried = max(
                    0,
                    int(
                        getattr(
                            threat_report,
                            "total_providers_queried",
                            0,
                        )
                    ),
                )

                providers_successful = max(
                    0,
                    int(
                        getattr(
                            threat_report,
                            "successful_provider_responses",
                            0,
                        )
                    ),
                )

                providers_successful = min(
                    providers_successful,
                    providers_queried,
                )

                if threat_score is not None:
                    signals.append(
                        DomainSignal(
                            type=(
                                DomainEvidenceType
                                .THREAT_INTEL
                            ),
                            name=(
                                "aggregated_provider_reputation"
                            ),
                            value=threat_score,
                            weight=1.0,
                            confidence=(
                                self._provider_confidence(
                                    providers_queried,
                                    providers_successful,
                                )
                            ),
                            description=(
                                "Aggregated multi-provider threat "
                                "intelligence reputation score."
                            ),
                        )
                    )

            except Exception:
                degraded_components.append(
                    "threat_intel"
                )

        # -------------------------------------------------------------
        # Bound and deduplicate
        # -------------------------------------------------------------

        signals = self._deduplicate_signals(
            signals
        )

        # -------------------------------------------------------------
        # Risk / confidence
        # -------------------------------------------------------------

        risk_score = self._calculate_risk(
            threat_score=threat_score,
            signals=signals,
            identity=identity,
        )

        confidence = self._calculate_confidence(
            threat_score=threat_score,
            providers_queried=providers_queried,
            providers_successful=(
                providers_successful
            ),
            registration=registration,
            dns=dns,
            signals=signals,
            degraded_components=(
                degraded_components
            ),
        )

        risk_level = self._risk_level(
            risk_score=risk_score,
            confidence=confidence,
        )

        severity = self._severity(
            risk_score,
            confidence,
        )

        is_malicious = (
            risk_score >= 0.75
            and confidence >= 0.45
        )

        is_suspicious = (
            not is_malicious
            and risk_score >= 0.40
            and confidence >= 0.30
        )

        reasons = self._build_reasons(
            signals=signals,
            risk_score=risk_score,
            confidence=confidence,
            threat_report=threat_report,
        )

        status = (
            DomainReputationStatus.DEGRADED
            if degraded_components
            else DomainReputationStatus.COMPLETE
        )

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        assessment = DomainReputationAssessment(
            status=status,
            domain=identity,
            risk_score=risk_score,
            confidence=confidence,
            risk_level=risk_level,
            severity=severity,
            is_malicious=is_malicious,
            is_suspicious=is_suspicious,
            reasons=reasons,
            signals=signals[
                :self.config.max_signals
            ],
            dns=dns,
            registration=registration,
            threat_intel_score=threat_score,
            providers_queried=providers_queried,
            providers_successful=(
                providers_successful
            ),
            degraded_components=sorted(
                set(degraded_components)
            ),
            analysis_duration_ms=max(
                0.0,
                elapsed_ms,
            ),
            config_fingerprint=(
                self.config.fingerprint
            ),
        )

        if self.cache:
            await self.cache.set(
                self._build_cache_key(
                    identity,
                    use_dns=use_dns,
                    use_registration=(
                        use_registration
                    ),
                ),
                assessment,
            )

        return assessment

    # =========================================================================
    # CACHE IDENTITY
    # =========================================================================

    def _build_cache_key(
        self,
        identity: DomainIdentity,
        *,
        use_dns: bool,
        use_registration: bool,
    ) -> str:
        raw = "|".join(
            [
                identity.ascii_domain,
                self.config.fingerprint,
                str(use_dns),
                str(use_registration),
            ]
        )

        digest = hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

        return (
            f"domain-reputation:"
            f"{MODULE_VERSION}:"
            f"{digest}"
        )

    # =========================================================================
    # DNS SIGNALS
    # =========================================================================

    @staticmethod
    def _analyze_dns_signals(
        dns: DNSObservation,
    ) -> List[DomainSignal]:

        signals: List[DomainSignal] = []

        if not dns.performed:
            return signals

        if dns.nx_domain:
            # NXDOMAIN is not malicious.
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.DNS,
                    name="nxdomain",
                    value=0.05,
                    weight=0.05,
                    confidence=0.80,
                    description=(
                        "Domain did not resolve through the configured "
                        "DNS resolver."
                    ),
                )
            )

        if (
            dns.failure_type
            == DNSFailureType.TIMEOUT
        ):
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.DNS,
                    name="dns_timeout",
                    value=0.0,
                    weight=0.0,
                    confidence=0.30,
                    description=(
                        "DNS resolution timed out."
                    ),
                )
            )

        if dns.private_ip_detected:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.DNS,
                    name="private_ip_resolution",
                    value=0.25,
                    weight=0.20,
                    confidence=0.90,
                    description=(
                        "Domain resolved to a private address."
                    ),
                )
            )

        if dns.loopback_ip_detected:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.DNS,
                    name="loopback_resolution",
                    value=0.45,
                    weight=0.30,
                    confidence=0.95,
                    description=(
                        "Domain resolved to a loopback address."
                    ),
                )
            )

        if dns.link_local_ip_detected:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.DNS,
                    name="link_local_resolution",
                    value=0.35,
                    weight=0.25,
                    confidence=0.90,
                    description=(
                        "Domain resolved to a link-local address."
                    ),
                )
            )

        if dns.unspecified_ip_detected:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.DNS,
                    name="unspecified_ip_resolution",
                    value=0.30,
                    weight=0.20,
                    confidence=0.90,
                    description=(
                        "Domain resolved to an unspecified IP address."
                    ),
                )
            )

        if dns.multicast_ip_detected:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.DNS,
                    name="multicast_ip_resolution",
                    value=0.35,
                    weight=0.20,
                    confidence=0.90,
                    description=(
                        "Domain resolved to a multicast address."
                    ),
                )
            )

        if dns.cname_chain:
            if len(dns.cname_chain) >= 4:
                signals.append(
                    DomainSignal(
                        type=DomainEvidenceType.DNS,
                        name="deep_cname_chain",
                        value=0.20,
                        weight=0.15,
                        confidence=0.75,
                        description=(
                            "Domain uses a relatively deep CNAME chain."
                        ),
                    )
                )

        return signals

    # =========================================================================
    # STRUCTURAL SIGNALS
    # =========================================================================

    def _analyze_structural_signals(
        self,
        identity: DomainIdentity,
    ) -> List[DomainSignal]:

        signals: List[DomainSignal] = []

        labels = identity.ascii_domain.split(".")

        # -------------------------------------------------------------
        # Deep subdomain
        # -------------------------------------------------------------

        if identity.label_count >= 5:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.STRUCTURAL,
                    name="deep_subdomain",
                    value=min(
                        0.50,
                        0.10
                        * (
                            identity.label_count
                            - 4
                        ),
                    ),
                    weight=0.20,
                    confidence=0.75,
                    description=(
                        "Domain contains a deep subdomain hierarchy."
                    ),
                )
            )

        # -------------------------------------------------------------
        # Long label
        # -------------------------------------------------------------

        if any(
            len(label)
            >= DEFAULT_LONG_LABEL_LENGTH
            for label in labels
        ):
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.STRUCTURAL,
                    name="long_domain_label",
                    value=0.20,
                    weight=0.15,
                    confidence=0.75,
                    description=(
                        "Domain contains an unusually long label."
                    ),
                )
            )

        # -------------------------------------------------------------
        # Entropy
        # -------------------------------------------------------------

        max_entropy = max(
            (
                _shannon_entropy(label)
                for label in labels
                if label
            ),
            default=0.0,
        )

        if max_entropy >= DEFAULT_ENTROPY_THRESHOLD:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.STRUCTURAL,
                    name="high_label_entropy",
                    value=min(
                        0.50,
                        max_entropy / 10.0,
                    ),
                    weight=0.20,
                    confidence=0.70,
                    description=(
                        "A domain label has unusually high character "
                        "entropy."
                    ),
                )
            )

        # -------------------------------------------------------------
        # Infrastructure namespace
        # -------------------------------------------------------------

        registrable = (
            identity.registrable_domain
            or ""
        ).lower()

        if any(
            registrable == suffix
            or registrable.endswith(
                "." + suffix
            )
            for suffix in (
                self.config.infrastructure_suffixes
            )
        ):
            # Important:
            # Infrastructure is NOT assigned negative risk.
            # We only add an explanatory zero-risk signal.
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.STRUCTURAL,
                    name="common_infrastructure_namespace",
                    value=0.0,
                    weight=0.0,
                    confidence=0.95,
                    description=(
                        "Domain belongs to a commonly used hosting or "
                        "developer-infrastructure namespace."
                    ),
                )
            )

        # -------------------------------------------------------------
        # Subdomain present
        # -------------------------------------------------------------

        if identity.is_subdomain:
            signals.append(
                DomainSignal(
                    type=DomainEvidenceType.STRUCTURAL,
                    name="subdomain_present",
                    value=0.02,
                    weight=0.02,
                    confidence=0.95,
                    description=(
                        "Input is a subdomain of the registrable domain."
                    ),
                )
            )

        return signals

    # =========================================================================
    # RISK
    # =========================================================================

    @staticmethod
    def _calculate_risk(
        *,
        threat_score: Optional[float],
        signals: Sequence[DomainSignal],
        identity: DomainIdentity,
    ) -> float:

        category_scores: Dict[str, List[float]] = {}

        for signal in signals:
            if signal.type == (
                DomainEvidenceType.THREAT_INTEL
            ):
                continue

            if signal.value <= 0:
                continue

            category = signal.type.value

            category_scores.setdefault(
                category,
                [],
            ).append(
                signal.value
                * signal.weight
                * signal.confidence
            )

        secondary_components: List[
            Tuple[float, float]
        ] = []

        for category, values in category_scores.items():
            if not values:
                continue

            # Root-combine repeated signals to reduce double-counting.
            combined = 1.0

            for value in values:
                combined *= (
                    1.0 - max(
                        0.0,
                        min(1.0, value),
                    )
                )

            category_score = 1.0 - combined

            category_weight = RISK_WEIGHTS.get(
                category,
                0.05,
            )

            secondary_components.append(
                (
                    category_score,
                    category_weight,
                )
            )

        if secondary_components:
            total_weight = sum(
                weight
                for _, weight
                in secondary_components
            )

            secondary_score = (
                sum(
                    score * weight
                    for score, weight
                    in secondary_components
                )
                / total_weight
                if total_weight > 0
                else 0.0
            )
        else:
            secondary_score = 0.0

        # -------------------------------------------------------------
        # Threat-intelligence dominance
        # -------------------------------------------------------------

        if threat_score is not None:
            risk = (
                threat_score
                * RISK_WEIGHTS["threat_intel"]
                + secondary_score
                * (
                    1.0
                    - RISK_WEIGHTS[
                        "threat_intel"
                    ]
                )
            )
        else:
            # Without external reputation we deliberately cap the
            # contribution of heuristic-only evidence.
            risk = min(
                0.70,
                secondary_score * 0.70,
            )

        # Infrastructure hosting does not reduce risk and does not
        # increase trust.
        _ = identity

        return max(
            0.0,
            min(
                1.0,
                float(risk),
            ),
        )

    # =========================================================================
    # CONFIDENCE
    # =========================================================================

    @staticmethod
    def _provider_confidence(
        providers_queried: int,
        providers_successful: int,
    ) -> float:

        if providers_queried <= 0:
            return 0.35

        coverage = (
            providers_successful
            / providers_queried
        )

        coverage = max(
            0.0,
            min(1.0, coverage),
        )

        return (
            0.45
            + 0.45 * coverage
        )

    @classmethod
    def _calculate_confidence(
        cls,
        *,
        threat_score: Optional[float],
        providers_queried: int,
        providers_successful: int,
        registration: RegistrationObservation,
        dns: DNSObservation,
        signals: Sequence[DomainSignal],
        degraded_components: Sequence[str],
    ) -> float:

        evidence_quality: List[
            Tuple[float, float]
        ] = []

        if threat_score is not None:
            evidence_quality.append(
                (
                    cls._provider_confidence(
                        providers_queried,
                        providers_successful,
                    ),
                    0.50,
                )
            )

        if registration.available:
            evidence_quality.append(
                (
                    0.75
                    if registration.created_at
                    else 0.55,
                    0.20,
                )
            )

        if dns.performed:
            if dns.failure_type == (
                DNSFailureType.TIMEOUT
            ):
                evidence_quality.append(
                    (0.20, 0.10)
                )
            elif dns.resolved:
                evidence_quality.append(
                    (0.75, 0.15)
                )
            else:
                evidence_quality.append(
                    (0.45, 0.10)
                )

        nonzero_signals = [
            signal
            for signal in signals
            if signal.value > 0
        ]

        if nonzero_signals:
            average_signal_confidence = (
                sum(
                    signal.confidence
                    for signal in nonzero_signals
                )
                / len(nonzero_signals)
            )

            evidence_quality.append(
                (
                    average_signal_confidence,
                    0.15,
                )
            )

        if not evidence_quality:
            return 0.10

        total_weight = sum(
            weight
            for _, weight
            in evidence_quality
        )

        confidence = (
            sum(
                quality * weight
                for quality, weight
                in evidence_quality
            )
            / total_weight
            if total_weight > 0
            else 0.10
        )

        # Missing external components reduce confidence, not risk.
        if degraded_components:
            penalty = min(
                0.30,
                0.08
                * len(
                    set(degraded_components)
                ),
            )

            confidence -= penalty

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    # =========================================================================
    # SIGNAL DEDUPLICATION
    # =========================================================================

    @staticmethod
    def _deduplicate_signals(
        signals: Sequence[DomainSignal],
    ) -> List[DomainSignal]:

        selected: Dict[
            Tuple[str, str],
            DomainSignal,
        ] = {}

        for signal in signals:
            key = (
                signal.type.value,
                signal.name,
            )

            existing = selected.get(key)

            if existing is None:
                selected[key] = signal
                continue

            # Keep the strongest observation.
            if (
                signal.value
                * signal.weight
                * signal.confidence
            ) > (
                existing.value
                * existing.weight
                * existing.confidence
            ):
                selected[key] = signal

        ordered = list(
            selected.values()
        )

        ordered.sort(
            key=lambda item: (
                -(
                    item.value
                    * item.weight
                    * item.confidence
                ),
                item.type.value,
                item.name,
            )
        )

        return ordered[
            :DEFAULT_MAX_SIGNALS
        ]

    # =========================================================================
    # REASONS
    # =========================================================================

    @staticmethod
    def _build_reasons(
        *,
        signals: Sequence[DomainSignal],
        risk_score: float,
        confidence: float,
        threat_report: Optional[Any],
    ) -> List[str]:

        reasons: List[str] = []

        for signal in signals:
            if signal.value <= 0:
                continue

            reasons.append(
                signal.name
            )

        # Provider tags are untrusted external strings.
        if threat_report is not None:
            provider_tags = getattr(
                threat_report,
                "combined_raw_tags",
                [],
            )

            if isinstance(
                provider_tags,
                (list, tuple, set),
            ):
                for tag in list(
                    provider_tags
                )[
                    :DEFAULT_MAX_PROVIDER_TAGS
                ]:
                    safe_tag = _bounded_text(
                        tag,
                        max_length=DEFAULT_MAX_TAG_LENGTH,
                    )

                    if safe_tag:
                        reasons.append(
                            "provider_tag:"
                            + safe_tag
                        )

        if confidence < 0.35:
            reasons.append(
                "low_evidence_confidence"
            )

        if risk_score >= 0.85:
            reasons.append(
                "critical_risk_score"
            )
        elif risk_score >= 0.65:
            reasons.append(
                "high_risk_score"
            )
        elif risk_score >= 0.40:
            reasons.append(
                "medium_risk_score"
            )

        return _unique_preserve_order(
            reasons,
            limit=DEFAULT_MAX_REASONS,
        )

    # =========================================================================
    # RISK LEVEL
    # =========================================================================

    @staticmethod
    def _risk_level(
        *,
        risk_score: float,
        confidence: float,
    ) -> DomainRiskLevel:

        if confidence < 0.20:
            return DomainRiskLevel.UNKNOWN

        if risk_score >= 0.85:
            return DomainRiskLevel.CRITICAL

        if risk_score >= 0.65:
            return DomainRiskLevel.HIGH

        if risk_score >= 0.40:
            return DomainRiskLevel.MEDIUM

        if risk_score >= 0.15:
            return DomainRiskLevel.LOW

        return DomainRiskLevel.CLEAN

    # =========================================================================
    # SEVERITY
    # =========================================================================

    @staticmethod
    def _severity(
        risk_score: float,
        confidence: float,
    ) -> ThreatSeverity:

        if confidence < 0.20:
            return ThreatSeverity.UNKNOWN

        if risk_score >= 0.85:
            return ThreatSeverity.CRITICAL

        if risk_score >= 0.65:
            return ThreatSeverity.HIGH

        if risk_score >= 0.40:
            return ThreatSeverity.MEDIUM

        if risk_score >= 0.15:
            return ThreatSeverity.LOW

        return ThreatSeverity.CLEAN

    # =========================================================================
    # TIMEOUT ASSESSMENT
    # =========================================================================

    def _timeout_assessment(
        self,
        identity: DomainIdentity,
        elapsed_ms: float,
    ) -> DomainReputationAssessment:

        return DomainReputationAssessment(
            status=DomainReputationStatus.TIMEOUT,
            domain=identity,
            risk_score=0.0,
            confidence=0.05,
            risk_level=DomainRiskLevel.UNKNOWN,
            severity=ThreatSeverity.UNKNOWN,
            is_malicious=False,
            is_suspicious=False,
            reasons=[
                "analysis_timeout",
            ],
            signals=[],
            dns=DNSObservation(
                performed=False,
            ),
            registration=RegistrationObservation(
                available=False,
            ),
            degraded_components=[
                "analysis",
            ],
            analysis_duration_ms=max(
                0.0,
                elapsed_ms,
            ),
            config_fingerprint=(
                self.config.fingerprint
            ),
        )


# =============================================================================
# CONVENIENCE API
# =============================================================================

async def assess_domain_reputation(
    domain: str,
    *,
    threat_intel: Optional[
        "ThreatIntelAggregator"
    ] = None,
    registration_provider: Optional[
        RegistrationProvider
    ] = None,
    enable_dns: bool = True,
    enable_registration: bool = True,
    force_refresh: bool = False,
    cache: Optional[
        DomainReputationCache
    ] = None,
) -> DomainReputationAssessment:

    engine = DomainReputationEngine(
        threat_intel=threat_intel,
        registration_provider=(
            registration_provider
        ),
        enable_dns=enable_dns,
        enable_registration=(
            enable_registration
        ),
        cache=cache,
    )

    return await engine.analyze(
        domain,
        force_refresh=force_refresh,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MODULE_VERSION",
    "DomainReputationStatus",
    "DomainRiskLevel",
    "DomainEvidenceType",
    "DNSFailureType",
    "DomainReputationError",
    "InvalidDomainError",
    "DomainResolutionError",
    "DomainReputationConfigurationError",
    "DomainIdentity",
    "DNSObservation",
    "RegistrationObservation",
    "DomainSignal",
    "DomainReputationAssessment",
    "PublicSuffixResolver",
    "DomainReputationCache",
    "DomainCanonicalizer",
    "UnicodeRiskAnalyzer",
    "DomainLexicalAnalyzer",
    "SafeDNSResolver",
    "RegistrationProvider",
    "RegistrationSignalAnalyzer",
    "DomainReputationConfig",
    "DomainReputationEngine",
    "assess_domain_reputation",
]
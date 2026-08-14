"""
PhishGuard AI - Threat Intelligence Core Contracts & Normalized Schemas
========================================================================

Version: 14.0.0
File Path: threat_intel/base.py

Production-grade foundation for the PhishGuard AI threat-intelligence subsystem.

Responsibilities
----------------
This module defines:

    * canonical IOC types
    * normalized threat severity
    * provider capabilities
    * structured threat-intelligence exceptions
    * validated normalized provider reports
    * IOC canonicalization utilities
    * deterministic cache/report identities
    * provider health contracts
    * provider query dispatching
    * defensive raw-response handling

Security Principles
-------------------
1. Never trust provider responses.
2. Never use raw provider values directly as canonical IOC identifiers.
3. Validate IOC syntax before network requests.
4. Keep API credentials out of normalized reports.
5. Bound externally supplied data.
6. Treat URL parsing as parsing, not validation.
7. Preserve deterministic normalization.
8. Do not silently convert malformed indicators into different indicators.
9. Distinguish operational failures from malicious verdicts.
10. Keep provider-specific implementation details outside this contract layer.

Compatibility
-------------
Designed for Python 3.11+ and Pydantic v2+.

The module intentionally keeps query_url(), query_domain(), and query_ip()
methods available for compatibility with existing PhishGuard providers while
also providing a generic query() dispatcher.

"""

from __future__ import annotations

import abc
import enum
import hashlib
import ipaddress
import math
import re
import secrets
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Set
from urllib.parse import SplitResult, urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# =============================================================================
# VERSION / LIMITS
# =============================================================================

__version__ = "14.0.0"

NORMALIZATION_VERSION = "1"

DEFAULT_TIMEOUT_SECONDS = 5.0
MIN_TIMEOUT_SECONDS = 0.1
MAX_TIMEOUT_SECONDS = 120.0

MAX_PROVIDER_NAME_LENGTH = 64
MAX_IOC_LENGTH = 8192
MAX_TAG_LENGTH = 128
MAX_CATEGORY_LENGTH = 128
MAX_TAGS = 128
MAX_CATEGORIES = 64
MAX_RAW_RESPONSE_KEYS = 512

MIN_SHA256_LENGTH = 64
MIN_MD5_LENGTH = 32

# Conservative hostname limits based on DNS practical limits.
MAX_DOMAIN_LENGTH = 253
MAX_DOMAIN_LABEL_LENGTH = 63

# URL schemes supported by the threat-intelligence layer.
SUPPORTED_URL_SCHEMES = frozenset({"http", "https"})

# Characters that should never appear in canonical domain names.
DOMAIN_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

# DNS hostname syntax.
DOMAIN_LABEL_RE = re.compile(
    r"^(?=.{1,63}$)(?!-)[A-Za-z0-9-]+(?<!-)$"
)

# Cryptographic hash syntax.
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
MD5_RE = re.compile(r"^[0-9a-fA-F]{32}$")


# =============================================================================
# ENUMERATIONS
# =============================================================================


class ThreatSeverity(str, enum.Enum):
    """
    Normalized threat severity.

    The ordering is represented numerically by severity_rank().
    """

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    CLEAN = "CLEAN"
    UNKNOWN = "UNKNOWN"


class IOCType(str, enum.Enum):
    """Supported Indicator of Compromise categories."""

    URL = "url"
    DOMAIN = "domain"
    IP = "ip"
    HASH_SHA256 = "hash_sha256"
    HASH_MD5 = "hash_md5"


class ProviderStatus(str, enum.Enum):
    """Operational status returned by provider health checks."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    AUTHENTICATION_FAILED = "authentication_failed"
    RATE_LIMITED = "rate_limited"
    UNKNOWN = "unknown"


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


class ThreatIntelException(Exception):
    """
    Base exception for all threat-intelligence operations.

    Attributes
    ----------
    provider:
        Provider identifier when known.
    retryable:
        Whether retrying may succeed.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        retryable: bool = False,
    ) -> None:
        self.provider = provider
        self.retryable = retryable
        super().__init__(message)


class InvalidIOCError(ThreatIntelException):
    """Raised when an IOC is malformed or unsupported."""

    def __init__(
        self,
        ioc_value: str,
        ioc_type: IOCType,
        reason: str,
    ) -> None:
        self.ioc_type = ioc_type
        self.reason = reason

        # Do not place the complete IOC in the exception if it could contain
        # sensitive URL query data.
        safe_preview = _safe_preview(ioc_value)

        super().__init__(
            f"Invalid {ioc_type.value} IOC ({safe_preview}): {reason}",
            retryable=False,
        )


class UnsupportedIOCTypeError(ThreatIntelException):
    """Raised when a provider does not support the requested IOC type."""

    def __init__(
        self,
        provider: str,
        ioc_type: IOCType,
    ) -> None:
        self.ioc_type = ioc_type

        super().__init__(
            f"[{provider}] IOC type '{ioc_type.value}' is not supported.",
            provider=provider,
            retryable=False,
        )


class ThreatIntelAPIError(ThreatIntelException):
    """Raised when a provider returns an unhandled API failure."""

    def __init__(
        self,
        provider: str,
        status_code: int,
        message: str,
        *,
        retryable: Optional[bool] = None,
        request_id: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.status_code = int(status_code)
        self.message = message
        self.request_id = request_id

        if retryable is None:
            retryable = status_code == 429 or 500 <= status_code <= 599

        super().__init__(
            f"[{provider}] API Error {status_code}: {message}",
            provider=provider,
            retryable=retryable,
        )


class RateLimitExceededError(ThreatIntelException):
    """Raised when a provider rate limit is reached."""

    def __init__(
        self,
        provider: str,
        retry_after: Optional[float] = None,
    ) -> None:
        self.provider = provider
        self.retry_after = retry_after

        message = f"[{provider}] Rate limit exceeded."

        if retry_after is not None:
            message += f" Retry after {retry_after:g} seconds."

        super().__init__(
            message,
            provider=provider,
            retryable=True,
        )


class ProviderTimeoutError(ThreatIntelException):
    """Raised when a provider request exceeds its configured timeout."""

    def __init__(
        self,
        provider: str,
        timeout_seconds: float,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = float(timeout_seconds)

        super().__init__(
            f"[{provider}] Request timed out after "
            f"{timeout_seconds:g}s.",
            provider=provider,
            retryable=True,
        )


class ProviderAuthenticationError(ThreatIntelException):
    """Raised when provider authentication fails."""

    def __init__(self, provider: str) -> None:
        self.provider = provider

        super().__init__(
            f"[{provider}] Provider authentication failed.",
            provider=provider,
            retryable=False,
        )


class ProviderUnavailableError(ThreatIntelException):
    """Raised when a provider is unavailable."""

    def __init__(
        self,
        provider: str,
        reason: str = "Provider unavailable.",
    ) -> None:
        self.provider = provider
        self.reason = reason

        super().__init__(
            f"[{provider}] {reason}",
            provider=provider,
            retryable=True,
        )


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _safe_preview(value: str, max_length: int = 96) -> str:
    """
    Produce a safe short representation for exception messages.

    This intentionally avoids logging complete attacker-controlled URLs.
    """
    value = str(value).replace("\r", "\\r").replace("\n", "\\n")

    if len(value) <= max_length:
        return repr(value)

    return repr(value[:max_length] + "...")


def _utc_now() -> datetime:
    """Return an explicit timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _normalize_string(value: str) -> str:
    """Normalize ordinary metadata strings without changing semantics."""
    if not isinstance(value, str):
        raise TypeError("Expected string value.")

    value = value.strip()

    if not value:
        raise ValueError("Value must not be empty.")

    if DOMAIN_CONTROL_RE.search(value):
        raise ValueError("Value contains control characters.")

    return value


def _normalize_token(
    value: str,
    *,
    max_length: int,
) -> str:
    """
    Normalize a provider tag/category token.

    Whitespace is collapsed and surrounding whitespace removed.
    """
    value = _normalize_string(value)

    if len(value) > max_length:
        raise ValueError(
            f"Value exceeds maximum length of {max_length} characters."
        )

    return value


def _deduplicate_strings(
    values: Iterable[str],
    *,
    max_items: int,
    max_length: int,
) -> List[str]:
    """
    Normalize, deduplicate, and bound string collections.
    """
    result: List[str] = []
    seen: Set[str] = set()

    for raw in values:
        normalized = _normalize_token(
            str(raw),
            max_length=max_length,
        )

        key = normalized.casefold()

        if key in seen:
            continue

        seen.add(key)
        result.append(normalized)

        if len(result) >= max_items:
            break

    return result


def _severity_for_score(score: float) -> ThreatSeverity:
    """Convert a normalized risk score into a deterministic severity."""
    if score >= 0.85:
        return ThreatSeverity.CRITICAL

    if score >= 0.65:
        return ThreatSeverity.HIGH

    if score >= 0.40:
        return ThreatSeverity.MEDIUM

    if score >= 0.15:
        return ThreatSeverity.LOW

    return ThreatSeverity.CLEAN


def _severity_rank(severity: ThreatSeverity) -> int:
    """Return an ordinal severity ranking."""
    ranks = {
        ThreatSeverity.UNKNOWN: -1,
        ThreatSeverity.CLEAN: 0,
        ThreatSeverity.LOW: 1,
        ThreatSeverity.MEDIUM: 2,
        ThreatSeverity.HIGH: 3,
        ThreatSeverity.CRITICAL: 4,
    }

    return ranks[severity]


# =============================================================================
# IOC NORMALIZATION
# =============================================================================


def normalize_ip(ip_value: str) -> str:
    """
    Canonicalize an IPv4/IPv6 address.

    IPv4 is returned in standard dotted-decimal form.
    IPv6 is compressed using ipaddress' canonical representation.
    """
    if not isinstance(ip_value, str):
        raise InvalidIOCError(
            str(ip_value),
            IOCType.IP,
            "IOC must be a string.",
        )

    value = ip_value.strip()

    if not value:
        raise InvalidIOCError(
            value,
            IOCType.IP,
            "IOC must not be empty.",
        )

    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise InvalidIOCError(
            value,
            IOCType.IP,
            "Invalid IPv4 or IPv6 address.",
        ) from exc

    return str(address)


def normalize_domain(domain: str) -> str:
    """
    Canonicalize a DNS domain name.

    Accepted examples
    ------------------
    example.com
    EXAMPLE.COM.
    https://example.com/path

    Rejected examples
    -----------------
    user:password@example.com
    example.com/path
    malformed hostnames
    IPv4/IPv6 literals

    Notes
    -----
    IDNs are converted to ASCII using IDNA where supported by Python.
    """
    if not isinstance(domain, str):
        raise InvalidIOCError(
            str(domain),
            IOCType.DOMAIN,
            "IOC must be a string.",
        )

    original = domain.strip()

    if not original:
        raise InvalidIOCError(
            original,
            IOCType.DOMAIN,
            "Domain must not be empty.",
        )

    if len(original) > MAX_IOC_LENGTH:
        raise InvalidIOCError(
            original,
            IOCType.DOMAIN,
            "Domain exceeds maximum IOC length.",
        )

    try:
        # If a scheme exists, parse it as a URL.
        candidate = original

        if "://" in candidate:
            parsed = urlsplit(candidate)

            if parsed.scheme.lower() not in SUPPORTED_URL_SCHEMES:
                raise InvalidIOCError(
                    original,
                    IOCType.DOMAIN,
                    "Unsupported URL scheme.",
                )

            if parsed.username is not None or parsed.password is not None:
                raise InvalidIOCError(
                    original,
                    IOCType.DOMAIN,
                    "Credentials are not permitted in domain IOCs.",
                )

            hostname = parsed.hostname

        else:
            # A bare domain must not contain path/query/fragment syntax.
            if any(char in original for char in "/?#"):
                raise InvalidIOCError(
                    original,
                    IOCType.DOMAIN,
                    "Domain IOC must not contain path, query, or fragment.",
                )

            hostname = original

        if not hostname:
            raise InvalidIOCError(
                original,
                IOCType.DOMAIN,
                "Missing hostname.",
            )

        # urlsplit().hostname is lower-case according to Python's parser.
        hostname = hostname.rstrip(".").lower()

        # Reject IP literals from the domain type.
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            pass
        else:
            raise InvalidIOCError(
                original,
                IOCType.DOMAIN,
                "IP addresses must use IOCType.IP.",
            )

        # Convert internationalized names to ASCII.
        try:
            hostname = hostname.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise InvalidIOCError(
                original,
                IOCType.DOMAIN,
                "Invalid internationalized domain name.",
            ) from exc

    except InvalidIOCError:
        raise

    except (ValueError, UnicodeError) as exc:
        raise InvalidIOCError(
            original,
            IOCType.DOMAIN,
            "Unable to parse domain.",
        ) from exc

    if len(hostname) > MAX_DOMAIN_LENGTH:
        raise InvalidIOCError(
            original,
            IOCType.DOMAIN,
            "Domain exceeds 253 characters.",
        )

    if hostname.startswith(".") or hostname.endswith("."):
        raise InvalidIOCError(
            original,
            IOCType.DOMAIN,
            "Malformed domain.",
        )

    labels = hostname.split(".")

    for label in labels:
        if not DOMAIN_LABEL_RE.fullmatch(label):
            raise InvalidIOCError(
                original,
                IOCType.DOMAIN,
                f"Invalid domain label: {_safe_preview(label)}",
            )

    return hostname


def normalize_url(url: str) -> str:
    """
    Canonicalize a URL sufficiently for deterministic IOC identity.

    Security restrictions
    ---------------------
    * only HTTP/HTTPS
    * no username/password
    * valid hostname
    * valid port
    * no control characters
    * absolute URL required

    The URL is NOT fetched by this function.
    """
    if not isinstance(url, str):
        raise InvalidIOCError(
            str(url),
            IOCType.URL,
            "URL must be a string.",
        )

    value = url.strip()

    if not value:
        raise InvalidIOCError(
            value,
            IOCType.URL,
            "URL must not be empty.",
        )

    if len(value) > MAX_IOC_LENGTH:
        raise InvalidIOCError(
            value,
            IOCType.URL,
            "URL exceeds maximum IOC length.",
        )

    if DOMAIN_CONTROL_RE.search(value):
        raise InvalidIOCError(
            value,
            IOCType.URL,
            "URL contains control characters.",
        )

    try:
        parsed: SplitResult = urlsplit(value)
    except ValueError as exc:
        raise InvalidIOCError(
            value,
            IOCType.URL,
            "Malformed URL.",
        ) from exc

    scheme = parsed.scheme.lower()

    if scheme not in SUPPORTED_URL_SCHEMES:
        raise InvalidIOCError(
            value,
            IOCType.URL,
            "Only HTTP and HTTPS URLs are supported.",
        )

    if not parsed.netloc:
        raise InvalidIOCError(
            value,
            IOCType.URL,
            "URL must contain a hostname.",
        )

    if parsed.username is not None or parsed.password is not None:
        raise InvalidIOCError(
            value,
            IOCType.URL,
            "Credentials are not permitted in threat-intelligence URLs.",
        )

    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise InvalidIOCError(
            value,
            IOCType.URL,
            "Invalid URL host or port.",
        ) from exc

    if not hostname:
        raise InvalidIOCError(
            value,
            IOCType.URL,
            "URL hostname is empty.",
        )

    # Validate the hostname without losing IPv6 support.
    try:
        ipaddress.ip_address(hostname)
        canonical_host = str(ipaddress.ip_address(hostname))

        # IPv6 literals need brackets when serialized into URLs.
        if ":" in canonical_host:
            canonical_host = f"[{canonical_host}]"

    except ValueError:
        canonical_host = normalize_domain(hostname)

    # Preserve explicit ports while normalizing default ports away.
    if port is not None:
        if not 1 <= port <= 65535:
            raise InvalidIOCError(
                value,
                IOCType.URL,
                "Port is outside valid TCP range.",
            )

        if not (
            (scheme == "http" and port == 80)
            or (scheme == "https" and port == 443)
        ):
            authority = f"{canonical_host}:{port}"
        else:
            authority = canonical_host
    else:
        authority = canonical_host

    # urlsplit keeps path/query/fragment separate.
    # We intentionally preserve them because they may identify different URLs.
    path = parsed.path or "/"

    # Remove fragment from canonical threat identity.
    #
    # Fragments are client-side and are normally not sent to the HTTP server.
    query = f"?{parsed.query}" if parsed.query else ""

    return f"{scheme}://{authority}{path}{query}"


def normalize_hash(
    hash_value: str,
    ioc_type: IOCType,
) -> str:
    """Canonicalize MD5 or SHA-256 hexadecimal hashes."""
    if not isinstance(hash_value, str):
        raise InvalidIOCError(
            str(hash_value),
            ioc_type,
            "Hash must be a string.",
        )

    value = hash_value.strip().lower()

    if ioc_type is IOCType.HASH_SHA256:
        if not SHA256_RE.fullmatch(value):
            raise InvalidIOCError(
                value,
                ioc_type,
                "Expected a 64-character hexadecimal SHA-256 hash.",
            )

        return value

    if ioc_type is IOCType.HASH_MD5:
        if not MD5_RE.fullmatch(value):
            raise InvalidIOCError(
                value,
                ioc_type,
                "Expected a 32-character hexadecimal MD5 hash.",
            )

        return value

    raise InvalidIOCError(
        value,
        ioc_type,
        "Unsupported hash IOC type.",
    )


def canonicalize_ioc(
    ioc_value: str,
    ioc_type: IOCType,
) -> str:
    """
    Canonicalize an IOC according to its declared IOC type.

    This is the single canonicalization entry point that should be used
    before provider queries, caching, persistence, and aggregation.
    """
    if not isinstance(ioc_type, IOCType):
        try:
            ioc_type = IOCType(ioc_type)
        except ValueError as exc:
            raise InvalidIOCError(
                str(ioc_value),
                IOCType.DOMAIN,
                "Unknown IOC type.",
            ) from exc

    if ioc_type is IOCType.URL:
        return normalize_url(ioc_value)

    if ioc_type is IOCType.DOMAIN:
        return normalize_domain(ioc_value)

    if ioc_type is IOCType.IP:
        return normalize_ip(ioc_value)

    if ioc_type in {
        IOCType.HASH_MD5,
        IOCType.HASH_SHA256,
    }:
        return normalize_hash(ioc_value, ioc_type)

    raise InvalidIOCError(
        str(ioc_value),
        ioc_type,
        "IOC type is not supported.",
    )


# =============================================================================
# NORMALIZED THREAT REPORT
# =============================================================================


class NormalizedThreatReport(BaseModel):
    """
    Provider-independent normalized threat-intelligence report.

    All external provider responses MUST be transformed into this model before
    reaching the orchestration/aggregation layer.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------

    provider_name: str = Field(
        ...,
        min_length=1,
        max_length=MAX_PROVIDER_NAME_LENGTH,
        description="Canonical provider identifier.",
    )

    ioc_value: str = Field(
        ...,
        min_length=1,
        max_length=MAX_IOC_LENGTH,
        description="Canonical IOC value.",
    )

    ioc_type: IOCType = Field(
        ...,
        description="IOC category.",
    )

    normalization_version: str = Field(
        default=NORMALIZATION_VERSION,
        min_length=1,
        max_length=16,
        description="IOC normalization contract version.",
    )

    # -------------------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------------------

    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized risk probability in [0, 1].",
    )

    severity: ThreatSeverity = Field(
        default=ThreatSeverity.UNKNOWN,
        description="Severity derived from risk score.",
    )

    is_malicious: bool = Field(
        default=False,
        description="Provider-normalized malicious verdict.",
    )

    # -------------------------------------------------------------------------
    # Detection statistics
    # -------------------------------------------------------------------------

    positive_detections: int = Field(
        default=0,
        ge=0,
        description="Number of malicious detections.",
    )

    total_engines: int = Field(
        default=0,
        ge=0,
        description="Total number of engines.",
    )

    # -------------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------------

    threat_categories: List[str] = Field(
        default_factory=list,
        description="Normalized threat categories.",
    )

    raw_tags: List[str] = Field(
        default_factory=list,
        description="Provider tags after basic normalization.",
    )

    # -------------------------------------------------------------------------
    # Operational metadata
    # -------------------------------------------------------------------------

    query_timestamp: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp at which provider data was obtained.",
    )

    response_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Provider response duration.",
    )

    cached_response: bool = Field(
        default=False,
        description="Whether this report came from cache.",
    )

    # -------------------------------------------------------------------------
    # Provider response
    # -------------------------------------------------------------------------

    raw_response: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Provider response payload. Do not store credentials or secrets."
        ),
    )

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------

    @field_validator("provider_name")
    @classmethod
    def validate_provider_name(cls, value: str) -> str:
        """Normalize provider names into stable identifiers."""
        value = _normalize_string(value).lower()

        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", value):
            raise ValueError(
                "Provider name must contain only letters, numbers, "
                "'.', '_' or '-'."
            )

        return value

    @field_validator("risk_score")
    @classmethod
    def validate_risk_score(cls, value: float) -> float:
        """Reject NaN and infinity despite numeric type compatibility."""
        value = float(value)

        if not math.isfinite(value):
            raise ValueError("risk_score must be finite.")

        return value

    @field_validator("response_time_ms")
    @classmethod
    def validate_response_time(cls, value: float) -> float:
        """Reject NaN/infinite response durations."""
        value = float(value)

        if not math.isfinite(value):
            raise ValueError("response_time_ms must be finite.")

        return value

    @field_validator("ioc_value")
    @classmethod
    def validate_ioc_value(cls, value: str) -> str:
        """Reject empty/control-character IOC values."""
        value = _normalize_string(value)

        if len(value) > MAX_IOC_LENGTH:
            raise ValueError(
                f"IOC exceeds maximum length of {MAX_IOC_LENGTH}."
            )

        return value

    @field_validator("threat_categories")
    @classmethod
    def normalize_categories(cls, value: List[str]) -> List[str]:
        """Normalize and bound threat categories."""
        return _deduplicate_strings(
            value,
            max_items=MAX_CATEGORIES,
            max_length=MAX_CATEGORY_LENGTH,
        )

    @field_validator("raw_tags")
    @classmethod
    def normalize_tags(cls, value: List[str]) -> List[str]:
        """Normalize and bound provider tags."""
        return _deduplicate_strings(
            value,
            max_items=MAX_TAGS,
            max_length=MAX_TAG_LENGTH,
        )

    @field_validator("raw_response")
    @classmethod
    def validate_raw_response(
        cls,
        value: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply shallow structural limits to provider payloads.

        Deep recursive sanitization belongs in the provider adapter because
        different providers have different response structures.
        """
        if len(value) > MAX_RAW_RESPONSE_KEYS:
            raise ValueError(
                f"raw_response contains more than "
                f"{MAX_RAW_RESPONSE_KEYS} top-level keys."
            )

        return value

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "NormalizedThreatReport":
        """
        Enforce invariants spanning multiple fields.

        Important:
        Severity is ALWAYS derived from risk_score. A provider cannot create
        an internally contradictory report by sending a different severity.
        """
        if self.positive_detections > self.total_engines:
            raise ValueError(
                "positive_detections cannot exceed total_engines."
            )

        # Synchronize the severity instead of trusting provider-supplied text.
        self.severity = _severity_for_score(self.risk_score)

        return self

    # -------------------------------------------------------------------------
    # Public helpers
    # -------------------------------------------------------------------------

    @property
    def detection_ratio(self) -> Optional[float]:
        """
        Return positive detection ratio.

        None means that the provider supplied no engine count.
        """
        if self.total_engines == 0:
            return None

        return self.positive_detections / self.total_engines

    @property
    def severity_rank(self) -> int:
        """Return an ordinal severity rank."""
        return _severity_rank(self.severity)

    def get_cache_key(self) -> str:
        """
        Generate a deterministic cache key.

        Normalization version is included so future canonicalization changes
        do not silently collide with old cache entries.
        """
        canonical_ioc = canonicalize_ioc(
            self.ioc_value,
            self.ioc_type,
        )

        raw_key = (
            f"{NORMALIZATION_VERSION}:"
            f"{self.provider_name}:"
            f"{self.ioc_type.value}:"
            f"{canonical_ioc}"
        )

        digest = hashlib.sha256(
            raw_key.encode("utf-8")
        ).hexdigest()

        return f"threat_cache:{digest}"

    def get_report_id(self) -> str:
        """
        Generate a deterministic report identity.

        Unlike cache keys, this is intended for persistence/auditing.
        """
        canonical_ioc = canonicalize_ioc(
            self.ioc_value,
            self.ioc_type,
        )

        material = (
            f"{self.provider_name}|"
            f"{self.ioc_type.value}|"
            f"{canonical_ioc}|"
            f"{self.normalization_version}"
        )

        return hashlib.sha256(
            material.encode("utf-8")
        ).hexdigest()

    def is_high_confidence_malicious(
        self,
        *,
        minimum_score: float = 0.85,
        minimum_detections: int = 2,
    ) -> bool:
        """
        Convenience predicate for orchestration policies.

        This does not replace the central risk decision engine.
        """
        return (
            self.is_malicious
            and self.risk_score >= minimum_score
            and self.positive_detections >= minimum_detections
        )


# =============================================================================
# PROVIDER HEALTH MODEL
# =============================================================================


class ProviderHealth(BaseModel):
    """Normalized provider health information."""

    model_config = ConfigDict(extra="forbid")

    provider_name: str = Field(
        ...,
        min_length=1,
        max_length=MAX_PROVIDER_NAME_LENGTH,
    )

    status: ProviderStatus = Field(
        default=ProviderStatus.UNKNOWN,
    )

    checked_at: datetime = Field(
        default_factory=_utc_now,
    )

    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    message: Optional[str] = Field(
        default=None,
        max_length=512,
    )

    retry_after_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
    )


# =============================================================================
# BASE THREAT PROVIDER
# =============================================================================


class BaseThreatProvider(abc.ABC):
    """
    Abstract contract for threat-intelligence providers.

    Provider implementations should focus on:

        API communication
        authentication
        provider-specific response parsing
        provider-specific rate limiting

    This base class owns:

        IOC validation
        canonicalization
        capability checking
        generic dispatch
        timeout configuration
        provider identity
    """

    # Provider implementations can override this to declare whether
    # authentication is mandatory.
    requires_api_key: ClassVar[bool] = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not math.isfinite(float(timeout_seconds)):
            raise ValueError(
                "timeout_seconds must be finite."
            )

        if not (
            MIN_TIMEOUT_SECONDS
            <= float(timeout_seconds)
            <= MAX_TIMEOUT_SECONDS
        ):
            raise ValueError(
                f"timeout_seconds must be between "
                f"{MIN_TIMEOUT_SECONDS} and "
                f"{MAX_TIMEOUT_SECONDS}."
            )

        if api_key is not None:
            api_key = api_key.strip()

            if not api_key:
                api_key = None

        if self.requires_api_key and api_key is None:
            # We deliberately do not raise here.
            #
            # Some applications instantiate providers before secret injection.
            # check_configuration() can be used when startup validation is needed.
            pass

        self._api_key = api_key
        self._timeout_seconds = float(timeout_seconds)

    # -------------------------------------------------------------------------
    # Provider identity
    # -------------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Unique provider identifier."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def supported_ioc_types(self) -> Set[IOCType]:
        """IOC types supported by this provider."""
        raise NotImplementedError

    @property
    def timeout_seconds(self) -> float:
        """Configured provider timeout."""
        return self._timeout_seconds

    @property
    def has_api_key(self) -> bool:
        """Return whether a provider credential is configured."""
        return bool(self._api_key)

    # -------------------------------------------------------------------------
    # Secret handling
    # -------------------------------------------------------------------------

    def check_configuration(self) -> None:
        """
        Validate provider startup configuration.

        API keys are intentionally not returned or logged.
        """
        if self.requires_api_key and not self._api_key:
            raise ProviderAuthenticationError(self.provider_name)

    # -------------------------------------------------------------------------
    # Abstract provider operations
    # -------------------------------------------------------------------------

    @abc.abstractmethod
    async def query_url(
        self,
        url: str,
    ) -> NormalizedThreatReport:
        """Query URL intelligence."""
        raise NotImplementedError

    @abc.abstractmethod
    async def query_domain(
        self,
        domain: str,
    ) -> NormalizedThreatReport:
        """Query domain intelligence."""
        raise NotImplementedError

    @abc.abstractmethod
    async def query_ip(
        self,
        ip_address: str,
    ) -> NormalizedThreatReport:
        """Query IP intelligence."""
        raise NotImplementedError

    @abc.abstractmethod
    async def check_health(self) -> bool:
        """Check provider availability."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Generic dispatcher
    # -------------------------------------------------------------------------

    async def query(
        self,
        ioc_value: str,
        ioc_type: IOCType,
    ) -> NormalizedThreatReport:
        """
        Canonical generic provider query entry point.

        This method prevents every caller from implementing its own IOC
        validation and dispatch logic.
        """
        try:
            ioc_type = IOCType(ioc_type)
        except ValueError as exc:
            raise InvalidIOCError(
                str(ioc_value),
                IOCType.DOMAIN,
                "Unsupported IOC type.",
            ) from exc

        if ioc_type not in self.supported_ioc_types:
            raise UnsupportedIOCTypeError(
                self.provider_name,
                ioc_type,
            )

        canonical_ioc = canonicalize_ioc(
            ioc_value,
            ioc_type,
        )

        if ioc_type is IOCType.URL:
            return await self.query_url(canonical_ioc)

        if ioc_type is IOCType.DOMAIN:
            return await self.query_domain(canonical_ioc)

        if ioc_type is IOCType.IP:
            return await self.query_ip(canonical_ioc)

        # Hash querying is intentionally not represented by a mandatory
        # abstract method to preserve compatibility with existing providers.
        #
        # Providers supporting hashes should override query() and/or provide
        # a dedicated query_hash implementation.
        raise UnsupportedIOCTypeError(
            self.provider_name,
            ioc_type,
        )

    # -------------------------------------------------------------------------
    # IOC helper utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def sanitize_domain(domain: str) -> str:
        """
        Backwards-compatible alias for canonical domain normalization.
        """
        return normalize_domain(domain)

    @staticmethod
    def validate_ip(ip_str: str) -> bool:
        """Return True when the value is a valid IPv4/IPv6 address."""
        try:
            normalize_ip(ip_str)
            return True
        except InvalidIOCError:
            return False

    @staticmethod
    def is_sha256(hash_str: str) -> bool:
        """Return True when the value is a valid SHA-256 hex digest."""
        if not isinstance(hash_str, str):
            return False

        return bool(SHA256_RE.fullmatch(hash_str.strip()))

    @staticmethod
    def is_md5(hash_str: str) -> bool:
        """Return True when the value is a valid MD5 hex digest."""
        if not isinstance(hash_str, str):
            return False

        return bool(MD5_RE.fullmatch(hash_str.strip()))

    @staticmethod
    def canonicalize(
        ioc_value: str,
        ioc_type: IOCType,
    ) -> str:
        """Public canonicalization helper."""
        return canonicalize_ioc(
            ioc_value,
            ioc_type,
        )


# =============================================================================
# PUBLIC EXPORTS
# =============================================================================


__all__ = [
    # Version
    "__version__",
    "NORMALIZATION_VERSION",

    # Enums
    "ThreatSeverity",
    "IOCType",
    "ProviderStatus",

    # Models
    "NormalizedThreatReport",
    "ProviderHealth",

    # Exceptions
    "ThreatIntelException",
    "InvalidIOCError",
    "UnsupportedIOCTypeError",
    "ThreatIntelAPIError",
    "RateLimitExceededError",
    "ProviderTimeoutError",
    "ProviderAuthenticationError",
    "ProviderUnavailableError",

    # IOC normalization
    "normalize_ip",
    "normalize_domain",
    "normalize_url",
    "normalize_hash",
    "canonicalize_ioc",

    # Provider contract
    "BaseThreatProvider",
]
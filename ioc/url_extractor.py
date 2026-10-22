"""
PhishGuard_AI - Production URL IOC Extractor
Version: 19.0.0

Security-focused URL extraction and canonicalization engine.

Design principles
-----------------
1. Extraction is NOT reputation.
2. No DNS, network, filesystem, subprocess, or external lookups.
3. Regexes are candidate generators, never validators.
4. Every candidate passes strict structural validation.
5. Host validation uses ipaddress + conservative DNS-name validation.
6. IPv4-mapped IPv6 addresses are explicitly recognized.
7. Userinfo is never persisted in canonical IOC values.
8. Defanging is bounded, recursive, and revalidated.
9. Percent encoding is validated before optional decoding.
10. Bare-host extraction is intentionally conservative.
11. Unicode / IDNA handling is fail-closed.
12. Resource usage is bounded.
13. Deterministic output ordering and IOC identifiers.
14. Rejection reasons are optionally observable without leaking data.
15. No raw credentials are emitted in results or diagnostics.

This module intentionally does not:
- perform DNS resolution
- perform WHOIS/RDAP
- query threat intelligence
- determine maliciousness
- dereference URLs
- follow redirects
- access the filesystem

Those responsibilities belong to later pipeline stages.

Standards / references:
- RFC 3986 - URI Generic Syntax
- WHATWG URL Standard
- Python urllib.parse documentation
- Python ipaddress documentation
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import math
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)
from urllib.parse import SplitResult, urlsplit, urlunsplit


# ============================================================================
# VERSION
# ============================================================================

__version__ = "19.0.0"


# ============================================================================
# ENUMERATIONS
# ============================================================================


class URLSourceType(str, Enum):
    EXPLICIT_URL = "explicit_url"
    DEFACTO_URL = "defacto_url"
    BARE_HOST = "bare_host"


class URLValidationStatus(str, Enum):
    VALID = "valid"

    EMPTY = "empty"
    TOO_LONG = "too_long"
    INVALID_SCHEME = "invalid_scheme"
    MISSING_AUTHORITY = "missing_authority"
    INVALID_AUTHORITY = "invalid_authority"
    USERINFO_PRESENT = "userinfo_present"
    INVALID_HOST = "invalid_host"
    INVALID_DOMAIN = "invalid_domain"
    INVALID_IP = "invalid_ip"
    INVALID_PORT = "invalid_port"
    INVALID_PERCENT_ENCODING = "invalid_percent_encoding"
    INVALID_UNICODE = "invalid_unicode"
    INVALID_CONTROL_CHARACTER = "invalid_control_character"
    INVALID_PATH = "invalid_path"
    INVALID_URL = "invalid_url"
    NON_WEB_SCHEME = "non_web_scheme"
    CANDIDATE_REJECTED = "candidate_rejected"
    TIME_BUDGET_EXCEEDED = "time_budget_exceeded"


class HostKind(str, Enum):
    DOMAIN = "domain"
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    IPV4_MAPPED_IPV6 = "ipv4_mapped_ipv6"


class NormalizationAction(str, Enum):
    NONE = "none"

    DEFLANGED = "defanged"
    SCHEME_RESTORED = "scheme_restored"
    BRACKET_WRAPPER_REMOVED = "bracket_wrapper_removed"
    ANGLE_WRAPPER_REMOVED = "angle_wrapper_removed"
    QUOTE_WRAPPER_REMOVED = "quote_wrapper_removed"
    WRAPPER_REMOVED = "wrapper_removed"
    PERCENT_DECODED = "percent_decoded"
    FRAGMENT_REMOVED = "fragment_removed"
    USERINFO_REMOVED = "userinfo_removed"
    HOST_NORMALIZED = "host_normalized"
    DEFAULT_PORT_REMOVED = "default_port_removed"
    TRAILING_DOT_REMOVED = "trailing_dot_removed"


class URLRiskFlag(str, Enum):
    NONE = "none"

    USERINFO_PRESENT = "userinfo_present"
    USERINFO_PASSWORD_PRESENT = "userinfo_password_present"

    SCHEME_SYNTHESIZED = "scheme_synthesized"

    NON_DEFAULT_PORT = "non_default_port"
    WELL_KNOWN_PORT = "well_known_port"

    IPV4_LITERAL = "ipv4_literal"
    IPV6_LITERAL = "ipv6_literal"
    IPV4_MAPPED_IPV6 = "ipv4_mapped_ipv6"

    PRIVATE_IP = "private_ip"
    LOOPBACK_IP = "loopback_ip"
    LINK_LOCAL_IP = "link_local_ip"
    MULTICAST_IP = "multicast_ip"
    RESERVED_IP = "reserved_ip"
    UNSPECIFIED_IP = "unspecified_ip"
    NON_GLOBAL_IP = "non_global_ip"

    IDN_HOST = "idn_host"
    UNICODE_HOST = "unicode_host"

    PUNYCODE_HOST = "punycode_host"

    SUSPICIOUS_PORT = "suspicious_port"
    LONG_HOST = "long_host"
    MANY_SUBDOMAINS = "many_subdomains"

    DEFANGED_INPUT = "defanged_input"
    PERCENT_ENCODED = "percent_encoded"

    URL_WRAPPER = "url_wrapper"
    FRAGMENT_PRESENT = "fragment_present"

    HOST_CASE_NORMALIZED = "host_case_normalized"

    ODD_PATH = "odd_path"
    CONTROL_CHARACTER_ATTEMPT = "control_character_attempt"


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class URLExtractorConfig:
    """
    Immutable production configuration.

    Defaults are deliberately conservative.
    """

    # Resource limits --------------------------------------------------------
    max_input_length: int = 1_000_000
    max_candidate_length: int = 16_384
    max_results: int = 512
    max_candidates: int = 2_048
    max_extraction_seconds: float = 0.250

    # Context ---------------------------------------------------------------
    context_radius: int = 160
    max_context_length: int = 320

    # URL length ------------------------------------------------------------
    max_url_length: int = 16_384
    max_host_length: int = 253
    max_label_length: int = 63
    max_path_length: int = 8_192
    max_query_length: int = 8_192

    # Scheme ----------------------------------------------------------------
    allowed_schemes: frozenset[str] = frozenset(
        {
            "http",
            "https",
            "ws",
            "wss",
        }
    )

    # Common defanged schemes -----------------------------------------------
    defanged_scheme_aliases: frozenset[str] = frozenset(
        {
            "hxxp",
            "hxxps",
            "hXXp",
            "hXXps",
        }
    )

    # Bare hosts ------------------------------------------------------------
    extract_bare_hosts: bool = True
    require_known_tld_for_bare_hosts: bool = False

    # This is intentionally a small maintained list rather than pretending
    # to be a full Public Suffix List.
    common_public_suffixes: frozenset[str] = frozenset(
        {
            "com",
            "org",
            "net",
            "edu",
            "gov",
            "mil",
            "int",
            "io",
            "ai",
            "app",
            "dev",
            "info",
            "biz",
            "name",
            "me",
            "co",
            "us",
            "uk",
            "ca",
            "au",
            "nz",
            "de",
            "fr",
            "es",
            "it",
            "nl",
            "be",
            "ch",
            "se",
            "no",
            "fi",
            "dk",
            "pl",
            "cz",
            "sk",
            "at",
            "in",
            "cn",
            "jp",
            "kr",
            "sg",
            "hk",
            "tw",
            "br",
            "mx",
            "ru",
            "za",
            "xyz",
            "online",
            "site",
            "top",
            "club",
            "shop",
            "store",
            "cloud",
            "tech",
            "live",
            "pro",
            "website",
            "work",
            "link",
            "today",
        }
    )

    # Userinfo --------------------------------------------------------------
    reject_userinfo: bool = False

    # Fragment --------------------------------------------------------------
    remove_fragment: bool = True

    # Percent decoding ------------------------------------------------------
    percent_decode_path_query: bool = False
    max_percent_decode_rounds: int = 2

    # Unicode ---------------------------------------------------------------
    normalize_unicode_nfkc: bool = True
    reject_bidi_controls: bool = True
    reject_zero_width_chars: bool = True

    # Canonicalization ------------------------------------------------------
    remove_default_ports: bool = True
    remove_trailing_host_dot: bool = True

    # Path normalization is deliberately conservative.
    normalize_dot_segments: bool = False
    collapse_repeated_slashes: bool = False

    # Validation ------------------------------------------------------------
    allow_localhost: bool = False
    allow_single_label_hosts: bool = False

    # Metrics ---------------------------------------------------------------
    collect_metrics: bool = True

    # IOC identity ----------------------------------------------------------
    ioc_id_prefix: str = "url"

    def __post_init__(self) -> None:
        if self.max_input_length <= 0:
            raise ValueError("max_input_length must be > 0")

        if self.max_candidate_length <= 0:
            raise ValueError("max_candidate_length must be > 0")

        if self.max_results <= 0:
            raise ValueError("max_results must be > 0")

        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be > 0")

        if self.max_extraction_seconds <= 0:
            raise ValueError("max_extraction_seconds must be > 0")

        if self.max_url_length <= 0:
            raise ValueError("max_url_length must be > 0")

        if not self.allowed_schemes:
            raise ValueError("allowed_schemes cannot be empty")


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class URLHostInfo:
    host: str
    normalized_host: str
    kind: HostKind

    is_private: bool = False
    is_global: bool = False
    is_loopback: bool = False
    is_link_local: bool = False
    is_reserved: bool = False
    is_multicast: bool = False
    is_unspecified: bool = False

    ipv4_mapped: Optional[str] = None

    idna_normalized: bool = False
    unicode_present: bool = False
    punycode_present: bool = False

    label_count: int = 0


@dataclass(frozen=True)
class URLCandidate:
    value: str
    start: int
    end: int
    source_type: URLSourceType
    explicit_scheme: bool = True


@dataclass(frozen=True)
class URLExtraction:
    """
    Immutable extracted URL IOC.

    raw_value:
        Exact candidate observed in the source text.

    canonical_value:
        Safe canonical IOC representation.

    host:
        Normalized host only.

    normalized_value:
        Alias retained for compatibility with earlier extractor versions.
    """

    ioc_id: str

    raw_value: str
    canonical_value: str
    normalized_value: str

    start: int
    end: int

    source_type: URLSourceType

    scheme: str
    host: str
    port: Optional[int]

    host_info: URLHostInfo

    confidence: float

    validation_status: URLValidationStatus = URLValidationStatus.VALID

    normalization_actions: Tuple[NormalizationAction, ...] = ()
    risk_flags: Tuple[URLRiskFlag, ...] = ()

    context: str = ""

    # Security-preserving metadata.
    fragment_present: bool = False
    userinfo_present: bool = False
    password_present: bool = False
    scheme_synthesized: bool = False
    percent_encoded: bool = False

    # Never store credentials.
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class URLExtractionMetrics:
    candidates_seen: int = 0
    candidates_accepted: int = 0
    candidates_rejected: int = 0

    urls_explicit: int = 0
    urls_bare: int = 0

    defanged_candidates: int = 0
    percent_encoded_candidates: int = 0
    userinfo_candidates: int = 0

    ipv4_candidates: int = 0
    ipv6_candidates: int = 0
    mapped_ipv6_candidates: int = 0
    domain_candidates: int = 0

    timeout_events: int = 0
    input_too_large: int = 0

    rejection_counts: Dict[str, int] = field(default_factory=dict)

    total_extraction_seconds: float = 0.0
    extraction_calls: int = 0

    def record_rejection(self, reason: URLValidationStatus) -> None:
        key = reason.value
        self.rejection_counts[key] = self.rejection_counts.get(key, 0) + 1


class MetricsRegistry:
    """
    Thread-safe metrics registry.

    Unlike the old global-only implementation, metrics can be injected.
    This allows isolated tests and isolated tenants/services.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics = URLExtractionMetrics()

    def increment(self, field_name: str, amount: int = 1) -> None:
        with self._lock:
            if hasattr(self._metrics, field_name):
                value = getattr(self._metrics, field_name)
                if isinstance(value, int):
                    setattr(self._metrics, field_name, value + amount)

    def record_rejection(self, reason: URLValidationStatus) -> None:
        with self._lock:
            self._metrics.record_rejection(reason)

    def add_duration(self, seconds: float) -> None:
        with self._lock:
            self._metrics.total_extraction_seconds += max(0.0, seconds)
            self._metrics.extraction_calls += 1

    def snapshot(self) -> Mapping[str, Any]:
        with self._lock:
            m = self._metrics

            return {
                "candidates_seen": m.candidates_seen,
                "candidates_accepted": m.candidates_accepted,
                "candidates_rejected": m.candidates_rejected,
                "urls_explicit": m.urls_explicit,
                "urls_bare": m.urls_bare,
                "defanged_candidates": m.defanged_candidates,
                "percent_encoded_candidates": m.percent_encoded_candidates,
                "userinfo_candidates": m.userinfo_candidates,
                "ipv4_candidates": m.ipv4_candidates,
                "ipv6_candidates": m.ipv6_candidates,
                "mapped_ipv6_candidates": m.mapped_ipv6_candidates,
                "domain_candidates": m.domain_candidates,
                "timeout_events": m.timeout_events,
                "input_too_large": m.input_too_large,
                "rejection_counts": dict(m.rejection_counts),
                "total_extraction_seconds": m.total_extraction_seconds,
                "extraction_calls": m.extraction_calls,
            }

    def reset(self) -> None:
        with self._lock:
            self._metrics = URLExtractionMetrics()


_GLOBAL_METRICS = MetricsRegistry()


# ============================================================================
# CONSTANTS / REGEXES
# ============================================================================


# RFC-style scheme candidate.
_SCHEME_RE = re.compile(
    r"(?i)\b([a-z][a-z0-9+.-]{1,31}):\/\/"
)


# Defanged scheme candidates.
_DEFANGED_SCHEME_RE = re.compile(
    r"(?i)\b("
    r"hxxps?|"
    r"https?|"
    r"[a-z][a-z0-9+.-]{1,20}"
    r")"
    r"(?:\s*"
    r"(?:\[:\]|\[:\]|:|\[\:\])"
    r")"
    r"(?:\s*"
    r"(?:\/\/|\\\\|//)"
    r")"
)


# Explicit URL candidate:
#
# Deliberately broad enough to find candidates, but bounded.
# It is NOT the validator.
_EXPLICIT_URL_RE = re.compile(
    r"(?i)"
    r"\b[a-z][a-z0-9+.-]{1,31}"
    r"(?:\s*(?::|\[\:\]|\[:\]))"
    r"(?:\s*(?:/{2}|\\{2}|\[/\]\[/\]))"
    r"[^\s<>'\"`]+"
)


# More reliable normal web scheme candidate.
_WEB_SCHEME_RE = re.compile(
    r"(?i)\b(?:https?|wss?)\s*:\s*(?:/|\[\/\])\s*(?:/|\[\/\])"
    r"[^\s<>'\"`]+"
)


# Bare hostname candidate.
#
# Important:
# - does NOT attempt to parse arbitrary dotted text
# - requires DNS-looking labels
# - validation occurs later
_BARE_HOST_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?:"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)\."
    r"){1,20}"
    r"(?:[A-Za-z]{2,63})"
    r"(?![A-Za-z0-9_-])",
    re.ASCII,
)


# Trailing prose punctuation.
_TRAILING_PUNCTUATION = frozenset(
    {
        ".",
        ",",
        ";",
        ":",
        "!",
        "?",
        ")",
        "]",
        "}",
        ">",
        "\"",
        "'",
        "`",
    }
)


# Unicode characters that frequently act as URL look-alikes.
_UNICODE_DOT_MAP = {
    "\u3002": ".",
    "\uff0e": ".",
    "\uff61": ".",
}

_UNICODE_COLON_MAP = {
    "\uff1a": ":",
}

_ZERO_WIDTH_CHARS = frozenset(
    {
        "\u200b",
        "\u200c",
        "\u200d",
        "\u2060",
        "\ufeff",
    }
)

_BIDI_CONTROLS = frozenset(
    {
        "\u061c",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)


# Dangerous / non-web schemes.
_NON_WEB_SCHEMES = frozenset(
    {
        "javascript",
        "data",
        "vbscript",
        "file",
        "about",
        "blob",
        "chrome",
        "chrome-extension",
        "view-source",
        "ms-excel",
        "ms-word",
        "ms-outlook",
        "search-ms",
        "shell",
        "mk",
        "res",
    }
)


_DEFAULT_PORTS = {
    "http": 80,
    "https": 443,
    "ws": 80,
    "wss": 443,
}


# Ports that deserve a security flag but are not automatically malicious.
_SUSPICIOUS_PORTS = frozenset(
    {
        21,
        22,
        23,
        25,
        53,
        110,
        111,
        135,
        139,
        143,
        445,
        587,
        993,
        995,
        1433,
        1521,
        2049,
        2375,
        2376,
        3306,
        3389,
        5432,
        5900,
        6379,
        8080,
        8443,
        9200,
        27017,
    }
)


# ============================================================================
# EXCEPTIONS
# ============================================================================


class URLExtractionError(Exception):
    """Base extractor error."""


class URLInputTooLargeError(URLExtractionError):
    """Input exceeded configured resource limits."""


class URLCandidateTooLargeError(URLExtractionError):
    """Candidate exceeded configured limits."""


class URLNormalizationError(URLExtractionError):
    """Candidate normalization failed."""


# ============================================================================
# NORMALIZATION
# ============================================================================


class URLNormalizer:
    """
    Bounded URL defanging / wrapper normalizer.

    The normalizer is deliberately deterministic and never performs
    arbitrary regex replacement supplied by callers.
    """

    _MAX_PASSES = 5

    @classmethod
    def normalize_candidate(
        cls,
        value: str,
    ) -> Tuple[str, Tuple[NormalizationAction, ...]]:
        if not value:
            return "", ()

        actions: List[NormalizationAction] = []
        current = value

        # ---------------------------------------------------------------
        # Unicode normalization
        # ---------------------------------------------------------------
        normalized = unicodedata.normalize("NFKC", current)

        if normalized != current:
            actions.append(NormalizationAction.DEFLANGED)

        current = normalized

        # ---------------------------------------------------------------
        # Normalize common Unicode punctuation used for defanging.
        # ---------------------------------------------------------------
        for source, target in {
            **_UNICODE_DOT_MAP,
            **_UNICODE_COLON_MAP,
        }.items():
            if source in current:
                current = current.replace(source, target)
                if NormalizationAction.DEFLANGED not in actions:
                    actions.append(NormalizationAction.DEFLANGED)

        # ---------------------------------------------------------------
        # Remove common invisible characters only when they are inside
        # obvious scheme / authority defanging patterns.
        #
        # We do NOT globally delete arbitrary Unicode controls.
        # ---------------------------------------------------------------
        for _ in range(cls._MAX_PASSES):
            before = current

            # Scheme defanging:
            #
            # hxxp://
            # hxxps://
            # hxxp[:]//
            # hxxps[://]
            # hxxps : //
            current = re.sub(
                r"(?i)\bhxxps?\b"
                r"\s*(?:\[\s*:\s*\]|\(\s*:\s*\)|:)"
                r"\s*(?:/{2}|\\{2})",
                lambda m: (
                    "https://"
                    if m.group(0).lower().startswith("hxxps")
                    else "http://"
                ),
                current,
            )

            current = re.sub(
                r"(?i)\bhxxps?\b"
                r"\s*(?:\[\s*:\s*\]|\(\s*:\s*\)|:)"
                r"\s*"
                r"(?:\[\s*/\s*\]\s*\[\s*/\s*\])",
                lambda m: (
                    "https://"
                    if m.group(0).lower().startswith("hxxps")
                    else "http://"
                ),
                current,
            )

            # https[://example.com]
            current = re.sub(
                r"(?i)\b(https?|hxxps?)\s*\[\s*:\s*/\s*/\s*\]",
                lambda m: (
                    "https://"
                    if m.group(1).lower() in {"https", "hxxps"}
                    else "http://"
                ),
                current,
            )

            # Replace bracketed dot / parenthesized dot.
            current = re.sub(
                r"\[\s*\.\s*\]|\(\s*\.\s*\)|\{\s*\.\s*\}",
                ".",
                current,
            )

            # Replace bracketed colon.
            current = re.sub(
                r"\[\s*:\s*\]|\(\s*:\s*\)|\{\s*:\s*\}",
                ":",
                current,
            )

            # Replace bracketed slash.
            current = re.sub(
                r"\[\s*/\s*\]",
                "/",
                current,
            )

            # Common slash defanging.
            current = re.sub(
                r"(?i)\[\s*slash\s*\]",
                "/",
                current,
            )

            # Optional whitespace inside a web scheme.
            current = re.sub(
                r"(?i)\b(https?|hxxps?)\s*:\s*/\s*/",
                lambda m: (
                    "https://"
                    if m.group(1).lower() in {"https", "hxxps"}
                    else "http://"
                ),
                current,
            )

            # Remove a single pair of outer URL wrappers.
            stripped = current.strip()

            if (
                len(stripped) >= 2
                and stripped[0] in "<[({\"'"
                and stripped[-1] in ">])}\"'"
            ):
                current = stripped[1:-1].strip()
                if (
                    NormalizationAction.WRAPPER_REMOVED
                    not in actions
                ):
                    actions.append(
                        NormalizationAction.WRAPPER_REMOVED
                    )
            else:
                current = stripped

            if current != before:
                if NormalizationAction.DEFLANGED not in actions:
                    actions.append(
                        NormalizationAction.DEFLANGED
                    )
                continue

            break

        return current, tuple(dict.fromkeys(actions))


# ============================================================================
# PERCENT ENCODING
# ============================================================================


def _contains_invalid_percent_encoding(value: str) -> bool:
    """
    Detect malformed percent sequences.

    A '%' must always be followed by two ASCII hexadecimal digits.
    """

    for index, char in enumerate(value):
        if char != "%":
            continue

        if index + 2 >= len(value):
            return True

        pair = value[index + 1 : index + 3]

        if not re.fullmatch(r"[0-9A-Fa-f]{2}", pair):
            return True

    return False


def _bounded_percent_decode(
    value: str,
    rounds: int,
) -> Tuple[str, bool]:
    """
    Bounded percent decoding.

    We only decode UTF-8-compatible byte sequences and never repeatedly
    decode indefinitely.
    """

    if rounds <= 0 or "%" not in value:
        return value, False

    current = value
    changed = False

    for _ in range(rounds):
        if _contains_invalid_percent_encoding(current):
            raise URLNormalizationError(
                "invalid percent encoding"
            )

        if "%" not in current:
            break

        try:
            from urllib.parse import unquote

            decoded = unquote(
                current,
                encoding="utf-8",
                errors="strict",
            )
        except (UnicodeDecodeError, ValueError) as exc:
            raise URLNormalizationError(
                "invalid UTF-8 percent encoding"
            ) from exc

        if decoded == current:
            break

        current = decoded
        changed = True

    return current, changed


# ============================================================================
# UNICODE / CONTROL VALIDATION
# ============================================================================


def _contains_forbidden_unicode(
    value: str,
    config: URLExtractorConfig,
) -> bool:
    for char in value:
        category = unicodedata.category(char)

        if category == "Cc":
            # Permit no ASCII controls in extracted URLs.
            return True

        if config.reject_zero_width_chars and char in _ZERO_WIDTH_CHARS:
            return True

        if config.reject_bidi_controls and char in _BIDI_CONTROLS:
            return True

    return False


# ============================================================================
# HOST VALIDATION
# ============================================================================


def _normalize_idna_hostname(
    hostname: str,
    config: URLExtractorConfig,
) -> Tuple[str, bool, bool, bool]:
    """
    Return:
        normalized_host,
        idna_changed,
        unicode_present,
        punycode_present
    """

    host = hostname.strip()

    if not host:
        raise ValueError("empty hostname")

    if _contains_forbidden_unicode(host, config):
        raise ValueError("forbidden unicode")

    unicode_present = any(ord(c) > 127 for c in host)
    punycode_present = any(
        label.lower().startswith("xn--")
        for label in host.split(".")
    )

    # DNS host names are case insensitive.
    lower_host = host.lower()

    # Strip exactly one terminal root dot for canonicalization.
    if lower_host.endswith("."):
        lower_host = lower_host[:-1]

    if not lower_host:
        raise ValueError("empty hostname")

    try:
        # Python's built-in IDNA codec is intentionally used here rather
        # than network-dependent PSL/DNS functionality.
        ascii_host = lower_host.encode(
            "idna"
        ).decode("ascii")
    except UnicodeError as exc:
        raise ValueError("invalid idna hostname") from exc

    if len(ascii_host) > config.max_host_length:
        raise ValueError("hostname too long")

    labels = ascii_host.split(".")

    if any(not label for label in labels):
        raise ValueError("empty hostname label")

    for label in labels:
        if len(label) > config.max_label_length:
            raise ValueError("hostname label too long")

        if label.startswith("-") or label.endswith("-"):
            raise ValueError("hostname label begins/ends with hyphen")

        if not re.fullmatch(
            r"[a-z0-9-]+",
            label,
            re.IGNORECASE,
        ):
            raise ValueError("invalid hostname label")

    changed = ascii_host != hostname

    return (
        ascii_host,
        changed,
        unicode_present,
        punycode_present,
    )


def _validate_ip(
    host: str,
) -> Optional[URLHostInfo]:
    """
    Validate an IP literal.

    Returns None for a DNS hostname.

    ipaddress is deliberately the final authority for IP syntax instead
    of a handwritten IPv6 regex.
    """

    candidate = host

    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]

    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return None

    if isinstance(address, ipaddress.IPv4Address):
        return URLHostInfo(
            host=host,
            normalized_host=str(address),
            kind=HostKind.IPV4,
            is_private=address.is_private,
            is_global=address.is_global,
            is_loopback=address.is_loopback,
            is_link_local=address.is_link_local,
            is_reserved=address.is_reserved,
            is_multicast=address.is_multicast,
            is_unspecified=address.is_unspecified,
            label_count=1,
        )

    mapped = getattr(address, "ipv4_mapped", None)

    if mapped is not None:
        return URLHostInfo(
            host=host,
            normalized_host=address.compressed.lower(),
            kind=HostKind.IPV4_MAPPED_IPV6,
            is_private=mapped.is_private,
            is_global=mapped.is_global,
            is_loopback=mapped.is_loopback,
            is_link_local=mapped.is_link_local,
            is_reserved=mapped.is_reserved,
            is_multicast=mapped.is_multicast,
            is_unspecified=mapped.is_unspecified,
            ipv4_mapped=str(mapped),
            label_count=1,
        )

    return URLHostInfo(
        host=host,
        normalized_host=address.compressed.lower(),
        kind=HostKind.IPV6,
        is_private=address.is_private,
        is_global=address.is_global,
        is_loopback=address.is_loopback,
        is_link_local=address.is_link_local,
        is_reserved=address.is_reserved,
        is_multicast=address.is_multicast,
        is_unspecified=address.is_unspecified,
        label_count=1,
    )


def _validate_domain_host(
    host: str,
    config: URLExtractorConfig,
) -> URLHostInfo:
    normalized, idna_changed, unicode_present, punycode_present = (
        _normalize_idna_hostname(host, config)
    )

    labels = normalized.split(".")

    if len(labels) == 1:
        if normalized == "localhost" and config.allow_localhost:
            return URLHostInfo(
                host=host,
                normalized_host=normalized,
                kind=HostKind.DOMAIN,
                idna_normalized=idna_changed,
                unicode_present=unicode_present,
                punycode_present=punycode_present,
                label_count=1,
            )

        if not config.allow_single_label_hosts:
            raise ValueError("single-label host rejected")

    if (
        config.require_known_tld_for_bare_hosts
        and len(labels) >= 2
        and labels[-1].lower()
        not in config.common_public_suffixes
    ):
        raise ValueError("unknown public suffix")

    return URLHostInfo(
        host=host,
        normalized_host=normalized,
        kind=HostKind.DOMAIN,
        idna_normalized=idna_changed,
        unicode_present=unicode_present,
        punycode_present=punycode_present,
        label_count=len(labels),
    )


# ============================================================================
# URL CANONICALIZATION
# ============================================================================


def _remove_dot_segments(path: str) -> str:
    """
    RFC 3986-style dot segment removal.

    Enabled only when explicitly configured because aggressive URL
    canonicalization can erase forensic distinctions.
    """

    if not path:
        return path

    output: List[str] = []

    for segment in path.split("/"):
        if segment == ".":
            continue

        if segment == "..":
            if output:
                output.pop()
            continue

        output.append(segment)

    result = "/".join(output)

    if path.startswith("/") and not result.startswith("/"):
        result = "/" + result

    if path.endswith("/") and not result.endswith("/"):
        result += "/"

    return result


def _normalize_path(
    path: str,
    config: URLExtractorConfig,
) -> str:
    if len(path) > config.max_path_length:
        raise ValueError("path too long")

    normalized = path

    if config.normalize_dot_segments:
        normalized = _remove_dot_segments(normalized)

    if config.collapse_repeated_slashes:
        normalized = re.sub(r"/{2,}", "/", normalized)

    return normalized


def _normalize_query(
    query: str,
    config: URLExtractorConfig,
) -> str:
    if len(query) > config.max_query_length:
        raise ValueError("query too long")

    return query


# ============================================================================
# URL EXTRACTOR
# ============================================================================


class URLExtractor:
    """
    Production URL IOC extractor.

    Thread-safe:
        configuration is immutable.
        metrics are synchronized.

    CPU-bound extraction:
        async_extract() offloads the synchronous parser to a worker thread.

    No external side effects.
    """

    def __init__(
        self,
        config: Optional[URLExtractorConfig] = None,
        *,
        metrics: Optional[MetricsRegistry] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or URLExtractorConfig()

        self.metrics = (
            metrics
            if metrics is not None
            else MetricsRegistry()
        )

        self._clock = clock

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def extract(
        self,
        text: str,
    ) -> List[URLExtraction]:
        """
        Extract validated URLs from text.

        Returns only successfully validated IOCs.
        """

        started = self._clock()

        try:
            if not isinstance(text, str):
                raise TypeError("text must be a string")

            if not text:
                return []

            if len(text) > self.config.max_input_length:
                self.metrics.increment("input_too_large")
                raise URLInputTooLargeError(
                    "input exceeds configured maximum length"
                )

            deadline = (
                started
                + self.config.max_extraction_seconds
            )

            candidates = self._find_candidates(
                text,
                deadline,
            )

            results: List[URLExtraction] = []

            for candidate in candidates:
                if self._clock() >= deadline:
                    self.metrics.increment("timeout_events")
                    break

                self.metrics.increment("candidates_seen")

                try:
                    extracted = self._extract_one(
                        text,
                        candidate,
                    )
                except URLExtractionError as exc:
                    status = self._error_to_status(exc)

                    self.metrics.increment(
                        "candidates_rejected"
                    )
                    self.metrics.record_rejection(status)
                    continue
                except Exception:
                    # Fail closed. Do not leak candidate or exception details.
                    self.metrics.increment(
                        "candidates_rejected"
                    )
                    self.metrics.record_rejection(
                        URLValidationStatus.INVALID_URL
                    )
                    continue

                if extracted is None:
                    self.metrics.increment(
                        "candidates_rejected"
                    )
                    self.metrics.record_rejection(
                        URLValidationStatus.CANDIDATE_REJECTED
                    )
                    continue

                results.append(extracted)

                if len(results) >= self.config.max_results:
                    break

            return self._deduplicate(results)

        finally:
            self.metrics.add_duration(
                self._clock() - started
            )

    async def async_extract(
        self,
        text: str,
    ) -> List[URLExtraction]:
        """
        Async-safe CPU offload.

        The parser itself remains synchronous and deterministic.
        """

        return await asyncio.to_thread(
            self.extract,
            text,
        )

    def validate_url(
        self,
        value: str,
        *,
        source_type: URLSourceType = URLSourceType.EXPLICIT_URL,
    ) -> Optional[URLExtraction]:
        """
        Validate one URL candidate.

        This is cheaper than scanning a complete document.
        """

        if not isinstance(value, str):
            return None

        candidate = URLCandidate(
            value=value,
            start=0,
            end=len(value),
            source_type=source_type,
            explicit_scheme=(
                source_type != URLSourceType.BARE_HOST
            ),
        )

        try:
            return self._extract_one(
                value,
                candidate,
            )
        except Exception:
            return None

    def metrics_snapshot(self) -> Mapping[str, Any]:
        return self.metrics.snapshot()

    # ----------------------------------------------------------------------
    # Candidate extraction
    # ----------------------------------------------------------------------

    def _find_candidates(
        self,
        text: str,
        deadline: float,
    ) -> List[URLCandidate]:

        candidates: List[URLCandidate] = []

        occupied: List[Tuple[int, int]] = []

        # ---------------------------------------------------------------
        # Explicit URLs
        # ---------------------------------------------------------------
        for match in _EXPLICIT_URL_RE.finditer(text):

            if self._clock() >= deadline:
                self.metrics.increment("timeout_events")
                break

            if len(candidates) >= self.config.max_candidates:
                break

            value = match.group(0)

            if len(value) > self.config.max_candidate_length:
                continue

            candidate = URLCandidate(
                value=value,
                start=match.start(),
                end=match.end(),
                source_type=URLSourceType.EXPLICIT_URL,
                explicit_scheme=True,
            )

            candidates.append(candidate)
            occupied.append(
                (match.start(), match.end())
            )

            if len(candidates) >= self.config.max_candidates:
                break

        # ---------------------------------------------------------------
        # Defanged / special explicit URL candidates
        # ---------------------------------------------------------------
        if len(candidates) < self.config.max_candidates:

            for match in _WEB_SCHEME_RE.finditer(text):

                if self._clock() >= deadline:
                    self.metrics.increment(
                        "timeout_events"
                    )
                    break

                if len(candidates) >= self.config.max_candidates:
                    break

                if self._overlaps(
                    match.start(),
                    match.end(),
                    occupied,
                ):
                    continue

                value = match.group(0)

                if len(value) > self.config.max_candidate_length:
                    continue

                candidates.append(
                    URLCandidate(
                        value=value,
                        start=match.start(),
                        end=match.end(),
                        source_type=URLSourceType.EXPLICIT_URL,
                        explicit_scheme=True,
                    )
                )

                occupied.append(
                    (match.start(), match.end())
                )

        # ---------------------------------------------------------------
        # Bare hosts
        # ---------------------------------------------------------------
        if (
            self.config.extract_bare_hosts
            and len(candidates)
            < self.config.max_candidates
        ):

            for match in _BARE_HOST_RE.finditer(text):

                if self._clock() >= deadline:
                    self.metrics.increment(
                        "timeout_events"
                    )
                    break

                if len(candidates) >= self.config.max_candidates:
                    break

                if self._overlaps(
                    match.start(),
                    match.end(),
                    occupied,
                ):
                    continue

                value = match.group(0)

                if len(value) > self.config.max_candidate_length:
                    continue

                # Cheap false-positive filters before full parsing.
                if self._looks_like_non_domain_token(
                    value,
                    text,
                    match.start(),
                    match.end(),
                ):
                    continue

                candidates.append(
                    URLCandidate(
                        value=value,
                        start=match.start(),
                        end=match.end(),
                        source_type=URLSourceType.BARE_HOST,
                        explicit_scheme=False,
                    )
                )

                occupied.append(
                    (match.start(), match.end())
                )

        # Deterministic ordering.
        candidates.sort(
            key=lambda item: (
                item.start,
                0
                if item.source_type
                == URLSourceType.EXPLICIT_URL
                else 1,
                -(item.end - item.start),
            )
        )

        return candidates

    @staticmethod
    def _overlaps(
        start: int,
        end: int,
        ranges: Sequence[Tuple[int, int]],
    ) -> bool:
        for other_start, other_end in ranges:
            if start < other_end and end > other_start:
                return True
        return False

    def _looks_like_non_domain_token(
        self,
        value: str,
        text: str,
        start: int,
        end: int,
    ) -> bool:

        lower = value.lower()

        # Version-like tokens.
        if re.fullmatch(
            r"v?\d+(?:\.\d+){1,6}(?:-[a-z0-9.-]+)?",
            lower,
            re.IGNORECASE,
        ):
            return True

        # File-like names.
        if lower.endswith(
            (
                ".exe",
                ".dll",
                ".so",
                ".py",
                ".js",
                ".ts",
                ".json",
                ".yaml",
                ".yml",
                ".log",
                ".txt",
                ".csv",
                ".xml",
                ".zip",
                ".tar",
                ".gz",
            )
        ):
            return True

        # Package / import style tokens.
        if "_" in value:
            return True

        # If directly adjacent to a URL scheme delimiter, let explicit
        # URL extraction handle it instead.
        left = text[max(0, start - 3) : start].lower()

        if left.endswith("://"):
            return True

        return False

    # ----------------------------------------------------------------------
    # Candidate parser
    # ----------------------------------------------------------------------

    def _extract_one(
        self,
        original_text: str,
        candidate: URLCandidate,
    ) -> Optional[URLExtraction]:

        raw = candidate.value

        if not raw:
            raise URLNormalizationError(
                "empty candidate"
            )

        if len(raw) > self.config.max_candidate_length:
            raise URLCandidateTooLargeError(
                "candidate too long"
            )

        # ---------------------------------------------------------------
        # First-pass cheap structural check.
        # ---------------------------------------------------------------
        stripped = raw.strip()

        if not stripped:
            raise URLNormalizationError(
                "empty candidate"
            )

        if (
            candidate.source_type
            == URLSourceType.EXPLICIT_URL
            and not self._has_possible_scheme(stripped)
        ):
            raise URLNormalizationError(
                "missing scheme"
            )

        # ---------------------------------------------------------------
        # Defanging / wrappers.
        # ---------------------------------------------------------------
        normalized, actions = (
            URLNormalizer.normalize_candidate(
                stripped
            )
        )

        if not normalized:
            raise URLNormalizationError(
                "empty normalized candidate"
            )

        if len(normalized) > self.config.max_url_length:
            raise URLCandidateTooLargeError(
                "normalized URL too long"
            )

        if actions:
            self.metrics.increment(
                "defanged_candidates"
            )

        # ---------------------------------------------------------------
        # Revalidate after normalization.
        # ---------------------------------------------------------------
        if _contains_forbidden_unicode(
            normalized,
            self.config,
        ):
            raise URLNormalizationError(
                "forbidden unicode/control"
            )

        if _contains_invalid_percent_encoding(
            normalized
        ):
            raise URLNormalizationError(
                "invalid percent encoding"
            )

        percent_encoded = "%" in normalized

        if percent_encoded:
            self.metrics.increment(
                "percent_encoded_candidates"
            )

        # ---------------------------------------------------------------
        # Optional bounded decoding.
        # ---------------------------------------------------------------
        if self.config.percent_decode_path_query:

            parsed_pre = self._parse_for_validation(
                normalized
            )

            decoded_path, changed_path = (
                _bounded_percent_decode(
                    parsed_pre.path,
                    self.config.max_percent_decode_rounds,
                )
            )

            decoded_query, changed_query = (
                _bounded_percent_decode(
                    parsed_pre.query,
                    self.config.max_percent_decode_rounds,
                )
            )

            if changed_path or changed_query:

                normalized = urlunsplit(
                    (
                        parsed_pre.scheme,
                        parsed_pre.netloc,
                        decoded_path,
                        decoded_query,
                        parsed_pre.fragment,
                    )
                )

                actions = tuple(
                    dict.fromkeys(
                        [
                            *actions,
                            NormalizationAction.PERCENT_DECODED,
                        ]
                    )
                )

                # Critical: validate the decoded candidate again.
                if _contains_forbidden_unicode(
                    normalized,
                    self.config,
                ):
                    raise URLNormalizationError(
                        "decoded value contains forbidden unicode"
                    )

                if _contains_invalid_percent_encoding(
                    normalized
                ):
                    raise URLNormalizationError(
                        "decoded value contains invalid percent encoding"
                    )

        # ---------------------------------------------------------------
        # Bare-host synthesis.
        # ---------------------------------------------------------------
        scheme_synthesized = False

        if candidate.source_type == URLSourceType.BARE_HOST:
            normalized = "https://" + normalized
            scheme_synthesized = True

            actions = tuple(
                dict.fromkeys(
                    [
                        *actions,
                        NormalizationAction.SCHEME_RESTORED,
                    ]
                )
            )

        elif "://" not in normalized:
            # A candidate which was expected to be explicit but does not
            # contain authority syntax is rejected.
            raise URLNormalizationError(
                "explicit URL lacks authority delimiter"
            )

        # ---------------------------------------------------------------
        # Parse.
        # ---------------------------------------------------------------
        parsed = self._parse_for_validation(
            normalized
        )

        scheme = parsed.scheme.lower()

        if scheme in _NON_WEB_SCHEMES:
            raise URLNormalizationError(
                "non-web scheme"
            )

        if scheme not in self.config.allowed_schemes:
            raise URLNormalizationError(
                "scheme not allowed"
            )

        if not parsed.netloc:
            raise URLNormalizationError(
                "missing authority"
            )

        # ---------------------------------------------------------------
        # Userinfo.
        # ---------------------------------------------------------------
        userinfo_present = (
            parsed.username is not None
            or parsed.password is not None
        )

        password_present = (
            parsed.password is not None
        )

        if userinfo_present:
            self.metrics.increment(
                "userinfo_candidates"
            )

            if self.config.reject_userinfo:
                raise URLNormalizationError(
                    "userinfo rejected"
                )

        # ---------------------------------------------------------------
        # Host.
        # ---------------------------------------------------------------
        try:
            hostname = parsed.hostname
        except ValueError as exc:
            raise URLNormalizationError(
                "invalid hostname"
            ) from exc

        if not hostname:
            raise URLNormalizationError(
                "empty hostname"
            )

        # Explicitly reject host percent encoding. WHATWG treats
        # percent-encoded host bytes specially and they are dangerous
        # for IOC identity if silently decoded. :contentReference[oaicite:1]{index=1}
        if "%" in hostname:
            raise URLNormalizationError(
                "percent encoded hostname"
            )

        ip_info = _validate_ip(hostname)

        if ip_info is not None:
            host_info = ip_info
        else:
            try:
                host_info = _validate_domain_host(
                    hostname,
                    self.config,
                )
            except ValueError as exc:
                raise URLNormalizationError(
                    "invalid domain"
                ) from exc

        # ---------------------------------------------------------------
        # Port.
        # ---------------------------------------------------------------
        try:
            port = parsed.port
        except ValueError as exc:
            raise URLNormalizationError(
                "invalid port"
            ) from exc

        if port is not None:
            if port < 1 or port > 65535:
                raise URLNormalizationError(
                    "port outside valid range"
                )

        # ---------------------------------------------------------------
        # Path/query.
        # ---------------------------------------------------------------
        try:
            path = _normalize_path(
                parsed.path,
                self.config,
            )

            query = _normalize_query(
                parsed.query,
                self.config,
            )
        except ValueError as exc:
            raise URLNormalizationError(
                "invalid path or query"
            ) from exc

        # ---------------------------------------------------------------
        # Fragment.
        # ---------------------------------------------------------------
        fragment_present = bool(parsed.fragment)

        if fragment_present:
            actions = tuple(
                dict.fromkeys(
                    [
                        *actions,
                        NormalizationAction.FRAGMENT_REMOVED,
                    ]
                )
            )

        fragment = (
            ""
            if self.config.remove_fragment
            else parsed.fragment
        )

        # ---------------------------------------------------------------
        # Host canonicalization.
        # ---------------------------------------------------------------
        normalized_host = host_info.normalized_host

        canonical_port = port

        if (
            self.config.remove_default_ports
            and port is not None
            and _DEFAULT_PORTS.get(scheme) == port
        ):
            canonical_port = None

            actions = tuple(
                dict.fromkeys(
                    [
                        *actions,
                        NormalizationAction.DEFAULT_PORT_REMOVED,
                    ]
                )
            )

        # IPv6 must remain bracketed in authority.
        authority_host = normalized_host

        if host_info.kind in {
            HostKind.IPV6,
            HostKind.IPV4_MAPPED_IPV6,
        }:
            authority_host = f"[{normalized_host}]"

        netloc = authority_host

        if canonical_port is not None:
            netloc += f":{canonical_port}"

        # ---------------------------------------------------------------
        # Canonical URL.
        # ---------------------------------------------------------------
        canonical_value = urlunsplit(
            (
                scheme,
                netloc,
                path,
                query,
                fragment,
            )
        )

        if len(canonical_value) > self.config.max_url_length:
            raise URLCandidateTooLargeError(
                "canonical URL too long"
            )

        # ---------------------------------------------------------------
        # Security flags.
        # ---------------------------------------------------------------
        risk_flags = self._build_risk_flags(
            host_info=host_info,
            port=port,
            scheme=scheme,
            userinfo_present=userinfo_present,
            password_present=password_present,
            scheme_synthesized=scheme_synthesized,
            defanged=bool(actions),
            percent_encoded=percent_encoded,
            fragment_present=fragment_present,
            original_host=hostname,
        )

        # ---------------------------------------------------------------
        # Confidence.
        # ---------------------------------------------------------------
        confidence = self._calculate_confidence(
            candidate=candidate,
            host_info=host_info,
            canonical_value=canonical_value,
            userinfo_present=userinfo_present,
            scheme_synthesized=scheme_synthesized,
            defanged=bool(actions),
        )

        # ---------------------------------------------------------------
        # Provenance context.
        # ---------------------------------------------------------------
        context = self._bounded_context(
            original_text,
            candidate.start,
            candidate.end,
        )

        # ---------------------------------------------------------------
        # IOC identity.
        # ---------------------------------------------------------------
        ioc_id = self._make_ioc_id(
            canonical_value
        )

        metadata: Dict[str, Any] = {
            "source_type": candidate.source_type.value,
            "explicit_scheme": candidate.explicit_scheme,
            "scheme_synthesized": scheme_synthesized,
            "fragment_present": fragment_present,
            "userinfo_present": userinfo_present,
            "password_present": password_present,
            "percent_encoded": percent_encoded,
            "host_kind": host_info.kind.value,
            "ipv4_mapped": host_info.ipv4_mapped,
            "idna_normalized": host_info.idna_normalized,
            "unicode_host": host_info.unicode_present,
            "punycode_host": host_info.punycode_present,
        }

        result = URLExtraction(
            ioc_id=ioc_id,
            raw_value=raw,
            canonical_value=canonical_value,
            normalized_value=canonical_value,
            start=candidate.start,
            end=candidate.end,
            source_type=candidate.source_type,
            scheme=scheme,
            host=normalized_host,
            port=port,
            host_info=host_info,
            confidence=confidence,
            validation_status=URLValidationStatus.VALID,
            normalization_actions=tuple(
                dict.fromkeys(actions)
            ),
            risk_flags=tuple(
                dict.fromkeys(risk_flags)
            ),
            context=context,
            fragment_present=fragment_present,
            userinfo_present=userinfo_present,
            password_present=password_present,
            scheme_synthesized=scheme_synthesized,
            percent_encoded=percent_encoded,
            metadata=metadata,
        )

        self.metrics.increment(
            "candidates_accepted"
        )

        if candidate.source_type == URLSourceType.EXPLICIT_URL:
            self.metrics.increment(
                "urls_explicit"
            )
        else:
            self.metrics.increment(
                "urls_bare"
            )

        if host_info.kind == HostKind.DOMAIN:
            self.metrics.increment(
                "domain_candidates"
            )
        elif host_info.kind == HostKind.IPV4:
            self.metrics.increment(
                "ipv4_candidates"
            )
        elif host_info.kind == HostKind.IPV6:
            self.metrics.increment(
                "ipv6_candidates"
            )
        elif (
            host_info.kind
            == HostKind.IPV4_MAPPED_IPV6
        ):
            self.metrics.increment(
                "mapped_ipv6_candidates"
            )

        return result

    # ----------------------------------------------------------------------
    # Parser
    # ----------------------------------------------------------------------

    @staticmethod
    def _parse_for_validation(
        value: str,
    ) -> SplitResult:
        """
        Defensive wrapper around urlsplit().

        urllib.parse explicitly warns that urlsplit() is not a validator,
        therefore all security validation remains outside this method.
        """

        try:
            parsed = urlsplit(
                value,
                allow_fragments=True,
            )
        except ValueError as exc:
            raise URLNormalizationError(
                "URL parse failed"
            ) from exc

        return parsed

    @staticmethod
    def _has_possible_scheme(
        value: str,
    ) -> bool:
        return bool(
            re.match(
                r"^[A-Za-z][A-Za-z0-9+.-]{1,31}\s*:",
                value,
            )
        )

    # ----------------------------------------------------------------------
    # Risk flags
    # ----------------------------------------------------------------------

    def _build_risk_flags(
        self,
        *,
        host_info: URLHostInfo,
        port: Optional[int],
        scheme: str,
        userinfo_present: bool,
        password_present: bool,
        scheme_synthesized: bool,
        defanged: bool,
        percent_encoded: bool,
        fragment_present: bool,
        original_host: str,
    ) -> List[URLRiskFlag]:

        flags: List[URLRiskFlag] = []

        if userinfo_present:
            flags.append(
                URLRiskFlag.USERINFO_PRESENT
            )

        if password_present:
            flags.append(
                URLRiskFlag.USERINFO_PASSWORD_PRESENT
            )

        if scheme_synthesized:
            flags.append(
                URLRiskFlag.SCHEME_SYNTHESIZED
            )

        if port is not None:
            if port != _DEFAULT_PORTS.get(scheme):
                flags.append(
                    URLRiskFlag.NON_DEFAULT_PORT
                )

            if port < 1024:
                flags.append(
                    URLRiskFlag.WELL_KNOWN_PORT
                )

            if port in _SUSPICIOUS_PORTS:
                flags.append(
                    URLRiskFlag.SUSPICIOUS_PORT
                )

        if host_info.kind == HostKind.IPV4:
            flags.append(
                URLRiskFlag.IPV4_LITERAL
            )

        if host_info.kind == HostKind.IPV6:
            flags.append(
                URLRiskFlag.IPV6_LITERAL
            )

        if (
            host_info.kind
            == HostKind.IPV4_MAPPED_IPV6
        ):
            flags.append(
                URLRiskFlag.IPV4_MAPPED_IPV6
            )

        if host_info.is_private:
            flags.append(
                URLRiskFlag.PRIVATE_IP
            )

        if host_info.is_loopback:
            flags.append(
                URLRiskFlag.LOOPBACK_IP
            )

        if host_info.is_link_local:
            flags.append(
                URLRiskFlag.LINK_LOCAL_IP
            )

        if host_info.is_multicast:
            flags.append(
                URLRiskFlag.MULTICAST_IP
            )

        if host_info.is_reserved:
            flags.append(
                URLRiskFlag.RESERVED_IP
            )

        if host_info.is_unspecified:
            flags.append(
                URLRiskFlag.UNSPECIFIED_IP
            )

        if (
            host_info.kind
            in {
                HostKind.IPV4,
                HostKind.IPV6,
                HostKind.IPV4_MAPPED_IPV6,
            }
            and not host_info.is_global
        ):
            flags.append(
                URLRiskFlag.NON_GLOBAL_IP
            )

        if host_info.unicode_present:
            flags.append(
                URLRiskFlag.UNICODE_HOST
            )

        if host_info.idna_normalized:
            flags.append(
                URLRiskFlag.IDN_HOST
            )

        if host_info.punycode_present:
            flags.append(
                URLRiskFlag.PUNYCODE_HOST
            )

        if host_info.label_count >= 5:
            flags.append(
                URLRiskFlag.MANY_SUBDOMAINS
            )

        if len(host_info.normalized_host) >= 100:
            flags.append(
                URLRiskFlag.LONG_HOST
            )

        if defanged:
            flags.append(
                URLRiskFlag.DEFANGED_INPUT
            )

        if percent_encoded:
            flags.append(
                URLRiskFlag.PERCENT_ENCODED
            )

        if fragment_present:
            flags.append(
                URLRiskFlag.FRAGMENT_PRESENT
            )

        if original_host.lower() != (
            host_info.normalized_host.lower()
        ):
            flags.append(
                URLRiskFlag.HOST_CASE_NORMALIZED
            )

        return flags

    # ----------------------------------------------------------------------
    # Confidence
    # ----------------------------------------------------------------------

    def _calculate_confidence(
        self,
        *,
        candidate: URLCandidate,
        host_info: URLHostInfo,
        canonical_value: str,
        userinfo_present: bool,
        scheme_synthesized: bool,
        defanged: bool,
    ) -> float:

        score = 0.80

        # Explicit complete URL is stronger than bare host.
        if candidate.source_type == URLSourceType.EXPLICIT_URL:
            score += 0.10
        else:
            score -= 0.08

        # IP literals are structurally unambiguous.
        if host_info.kind in {
            HostKind.IPV4,
            HostKind.IPV6,
            HostKind.IPV4_MAPPED_IPV6,
        }:
            score += 0.04

        if host_info.kind == HostKind.DOMAIN:
            if host_info.label_count >= 2:
                score += 0.03

        # Defanged input requires additional normalization.
        if defanged:
            score -= 0.03

        # Synthesized scheme is weaker evidence than an explicit scheme.
        if scheme_synthesized:
            score -= 0.05

        # Userinfo is still a valid URL but deserves slightly lower
        # extraction confidence.
        if userinfo_present:
            score -= 0.04

        # Excessively long URLs are noisier.
        if len(canonical_value) > 4096:
            score -= 0.05

        if not math.isfinite(score):
            return 0.0

        return max(
            0.0,
            min(1.0, score),
        )

    # ----------------------------------------------------------------------
    # Context
    # ----------------------------------------------------------------------

    def _bounded_context(
        self,
        text: str,
        start: int,
        end: int,
    ) -> str:

        radius = self.config.context_radius

        left = max(0, start - radius)
        right = min(
            len(text),
            end + radius,
        )

        context = text[left:right]

        if len(context) > self.config.max_context_length:
            context = context[
                : self.config.max_context_length
            ]

        # Prevent accidentally persisting unbounded secrets in context.
        return context

    # ----------------------------------------------------------------------
    # IOC ID
    # ----------------------------------------------------------------------

    def _make_ioc_id(
        self,
        canonical_value: str,
    ) -> str:

        digest = hashlib.sha256(
            canonical_value.encode(
                "utf-8",
                errors="strict",
            )
        ).hexdigest()

        return (
            f"{self.config.ioc_id_prefix}_"
            f"{digest[:32]}"
        )

    # ----------------------------------------------------------------------
    # Deduplication
    # ----------------------------------------------------------------------

    def _deduplicate(
        self,
        results: Sequence[URLExtraction],
    ) -> List[URLExtraction]:

        by_canonical: Dict[
            str,
            URLExtraction,
        ] = {}

        for result in results:

            existing = by_canonical.get(
                result.canonical_value
            )

            if existing is None:
                by_canonical[
                    result.canonical_value
                ] = result
                continue

            preferred = self._prefer_result(
                existing,
                result,
            )

            # Preserve provenance without storing duplicate raw credentials.
            metadata = dict(
                preferred.metadata
            )

            seen_sources = list(
                metadata.get(
                    "seen_source_types",
                    [],
                )
            )

            for source in (
                existing.source_type.value,
                result.source_type.value,
            ):
                if source not in seen_sources:
                    seen_sources.append(source)

            metadata[
                "seen_source_types"
            ] = tuple(seen_sources)

            preferred = URLExtraction(
                **{
                    **preferred.__dict__,
                    "metadata": metadata,
                }
            )

            by_canonical[
                result.canonical_value
            ] = preferred

        output = list(
            by_canonical.values()
        )

        output.sort(
            key=lambda item: (
                item.start,
                item.end,
                item.canonical_value,
            )
        )

        return output[: self.config.max_results]

    @staticmethod
    def _prefer_result(
        first: URLExtraction,
        second: URLExtraction,
    ) -> URLExtraction:

        # Explicit URLs beat synthesized bare hosts.
        if (
            first.source_type
            != second.source_type
        ):
            if (
                first.source_type
                == URLSourceType.EXPLICIT_URL
            ):
                return first

            if (
                second.source_type
                == URLSourceType.EXPLICIT_URL
            ):
                return second

        # Higher confidence wins.
        if second.confidence > first.confidence:
            return second

        # Longer source span generally carries stronger provenance.
        first_len = first.end - first.start
        second_len = second.end - second.start

        if second_len > first_len:
            return second

        # Stable deterministic tie-breaker.
        if second.raw_value < first.raw_value:
            return second

        return first

    # ----------------------------------------------------------------------
    # Errors
    # ----------------------------------------------------------------------

    @staticmethod
    def _error_to_status(
        error: URLExtractionError,
    ) -> URLValidationStatus:

        message = str(error).lower()

        if "empty" in message:
            return URLValidationStatus.EMPTY

        if "too long" in message:
            return URLValidationStatus.TOO_LONG

        if "scheme" in message:
            return URLValidationStatus.INVALID_SCHEME

        if "authority" in message:
            return URLValidationStatus.INVALID_AUTHORITY

        if "userinfo" in message:
            return URLValidationStatus.USERINFO_PRESENT

        if "hostname" in message:
            return URLValidationStatus.INVALID_HOST

        if "domain" in message:
            return URLValidationStatus.INVALID_DOMAIN

        if "port" in message:
            return URLValidationStatus.INVALID_PORT

        if "percent" in message:
            return URLValidationStatus.INVALID_PERCENT_ENCODING

        if "unicode" in message:
            return URLValidationStatus.INVALID_UNICODE

        if "control" in message:
            return URLValidationStatus.INVALID_CONTROL_CHARACTER

        if "path" in message:
            return URLValidationStatus.INVALID_PATH

        if "non-web" in message:
            return URLValidationStatus.NON_WEB_SCHEME

        return URLValidationStatus.INVALID_URL


# ============================================================================
# DEFAULT EXTRACTOR
# ============================================================================


_default_extractor: Optional[URLExtractor] = None
_default_extractor_lock = threading.Lock()


def get_default_url_extractor() -> URLExtractor:
    """
    Return the process-local default extractor.

    The instance is lazy-created and thread-safe.
    """

    global _default_extractor

    if _default_extractor is None:
        with _default_extractor_lock:
            if _default_extractor is None:
                _default_extractor = URLExtractor(
                    config=URLExtractorConfig(),
                    metrics=_GLOBAL_METRICS,
                )

    return _default_extractor


# ============================================================================
# CONVENIENCE APIs
# ============================================================================


def extract_urls(
    text: str,
    *,
    config: Optional[URLExtractorConfig] = None,
    metrics: Optional[MetricsRegistry] = None,
) -> List[URLExtraction]:
    """
    Convenience extraction API.

    If custom config or metrics are supplied, an isolated extractor is used.
    Otherwise the shared production extractor is reused.
    """

    if config is None and metrics is None:
        return get_default_url_extractor().extract(text)

    return URLExtractor(
        config=config,
        metrics=metrics,
    ).extract(text)


async def async_extract_urls(
    text: str,
    *,
    config: Optional[URLExtractorConfig] = None,
    metrics: Optional[MetricsRegistry] = None,
) -> List[URLExtraction]:
    """
    Async convenience API.
    """

    if config is None and metrics is None:
        return await get_default_url_extractor().async_extract(
            text
        )

    extractor = URLExtractor(
        config=config,
        metrics=metrics,
    )

    return await extractor.async_extract(text)


def validate_url(
    value: str,
    *,
    config: Optional[URLExtractorConfig] = None,
) -> Optional[URLExtraction]:
    """
    Lightweight single-value validation helper.
    """

    extractor = URLExtractor(
        config=config,
    )

    return extractor.validate_url(value)


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "__version__",

    # Configuration
    "URLExtractorConfig",

    # Enums
    "URLSourceType",
    "URLValidationStatus",
    "HostKind",
    "NormalizationAction",
    "URLRiskFlag",

    # Models
    "URLCandidate",
    "URLHostInfo",
    "URLExtraction",
    "URLExtractionMetrics",

    # Metrics
    "MetricsRegistry",

    # Exceptions
    "URLExtractionError",
    "URLInputTooLargeError",
    "URLCandidateTooLargeError",
    "URLNormalizationError",

    # Engine
    "URLExtractor",
    "get_default_url_extractor",

    # Convenience
    "extract_urls",
    "async_extract_urls",
    "validate_url",
]
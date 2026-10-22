"""
PhishGuard_AI - Production IP IOC Extractor
Version: 19.0.0

Purpose
-------
Extract IPv4 / IPv6 indicators from untrusted text without performing
network, DNS, filesystem, or subprocess operations.

Security properties
-------------------
- Extraction is strictly local and deterministic.
- No DNS/network access.
- Bounded input, candidate count, output count, and context size.
- Regexes are intentionally conservative and avoid nested ambiguous
  quantifiers where possible.
- ipaddress is the final authority for IP syntax.
- Explicit IPv4-mapped IPv6 handling.
- Explicit IPv6 zone-index detection.
- Defanged IPv4 / IPv6 support.
- Bracketed IPv6 support.
- Optional URL/scheme and port stripping.
- Optional alternative IPv4 representation support.
- Special-purpose classification.
- Stable IOC IDs.
- Provenance-preserving normalization metadata.
- Overlapping candidate deconfliction.
- Thread-safe metrics with thread-local batching.
- Async API offloads CPU work.
- No exception details from internal failures are exposed by default.

Important
---------
This module performs IOC EXTRACTION only.

It does not:
- perform reputation checks
- perform DNS lookups
- perform RDAP/WHOIS
- connect to IPs
- decide whether an IP is malicious

External reputation belongs in:
    reputation/ip_reputation.py
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Iterator, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# Constants
# ============================================================================

MODULE_NAME = "ioc.ip_extractor"
MODULE_VERSION = "19.0.0"

DEFAULT_MAX_INPUT_CHARS = 1_000_000
DEFAULT_MAX_IOCS = 512
DEFAULT_MAX_CANDIDATES = 4_096
DEFAULT_CONTEXT_RADIUS = 180
DEFAULT_MAX_CONTEXT_CHARS = 360
DEFAULT_MAX_EXTRACTION_SECONDS = 0.250
DEFAULT_MAX_CANDIDATE_CHARS = 128

IOC_ID_BYTES = 16


# ============================================================================
# Enums
# ============================================================================


class IPVersion(str, Enum):
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    IPV4_MAPPED_IPV6 = "ipv4_mapped_ipv6"


class IPAddressScope(str, Enum):
    GLOBAL = "global"
    PRIVATE = "private"
    SHARED = "shared"
    LOOPBACK = "loopback"
    LINK_LOCAL = "link_local"
    MULTICAST = "multicast"
    UNSPECIFIED = "unspecified"
    RESERVED = "reserved"
    DOCUMENTATION = "documentation"
    BENCHMARK = "benchmark"
    UNIQUE_LOCAL = "unique_local"
    TEREDO = "teredo"
    SIXTOFOUR = "6to4"
    ORCHID = "orchid"
    TRANSLATION = "translation"
    SPECIAL = "special"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class CandidateKind(str, Enum):
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    BRACKETED_IPV6 = "bracketed_ipv6"
    URL_HOST = "url_host"
    DEFANGED_IPV4 = "defanged_ipv4"
    DEFANGED_IPV6 = "defanged_ipv6"
    ALTERNATIVE_IPV4 = "alternative_ipv4"


class RejectionReason(str, Enum):
    EMPTY = "empty"
    TOO_LONG = "too_long"
    INVALID_IP = "invalid_ip"
    INVALID_IPV4 = "invalid_ipv4"
    INVALID_IPV6 = "invalid_ipv6"
    ZONE_INDEX = "zone_index"
    UNSUPPORTED_ALTERNATIVE = "unsupported_alternative"
    AMBIGUOUS_ALTERNATIVE = "ambiguous_alternative"
    NON_IP_TOKEN = "non_ip_token"
    CONTROL_CHARACTER = "control_character"
    PRIVATE_DISABLED = "private_disabled"
    MULTICAST_DISABLED = "multicast_disabled"
    SPECIAL_DISABLED = "special_disabled"
    NOT_ALLOWED = "not_allowed"
    DUPLICATE = "duplicate"
    LIMIT_REACHED = "limit_reached"
    TIME_BUDGET = "time_budget"
    INTERNAL_ERROR = "internal_error"


class NormalizationAction(str, Enum):
    NONE = "none"
    UNICODE_NORMALIZED = "unicode_normalized"
    WHITESPACE_REMOVED = "whitespace_removed"
    DOT_RESTORED = "dot_restored"
    COLON_RESTORED = "colon_restored"
    BRACKETS_REMOVED = "brackets_removed"
    SCHEME_REMOVED = "scheme_removed"
    PORT_REMOVED = "port_removed"
    WRAPPER_REMOVED = "wrapper_removed"
    ZONE_DETECTED = "zone_detected"
    MAPPED_IPV4_DERIVED = "mapped_ipv4_derived"
    ALTERNATIVE_IPV4_DECODED = "alternative_ipv4_decoded"


# ============================================================================
# Models
# ============================================================================


class IPExtractorConfig(BaseModel):
    """
    Immutable extraction configuration.

    Conservative defaults are intentional.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_input_chars: int = Field(
        DEFAULT_MAX_INPUT_CHARS,
        ge=1_024,
        le=10_000_000,
    )

    max_iocs: int = Field(
        DEFAULT_MAX_IOCS,
        ge=1,
        le=100_000,
    )

    max_candidates: int = Field(
        DEFAULT_MAX_CANDIDATES,
        ge=1,
        le=1_000_000,
    )

    max_candidate_chars: int = Field(
        DEFAULT_MAX_CANDIDATE_CHARS,
        ge=16,
        le=1_024,
    )

    context_radius: int = Field(
        DEFAULT_CONTEXT_RADIUS,
        ge=0,
        le=4_096,
    )

    max_context_chars: int = Field(
        DEFAULT_MAX_CONTEXT_CHARS,
        ge=0,
        le=8_192,
    )

    max_extraction_seconds: float = Field(
        DEFAULT_MAX_EXTRACTION_SECONDS,
        gt=0.001,
        le=30.0,
    )

    allow_private: bool = True
    allow_multicast: bool = True
    allow_special: bool = True

    reject_zone_indices: bool = True

    """
    If enabled, parse:
        2130706433       -> 127.0.0.1
        0x7f000001       -> 127.0.0.1
        0177.0.0.1      -> 127.0.0.1

    Disabled by default because these representations are highly
    ambiguous in natural language.
    """
    allow_alternative_ipv4_formats: bool = False

    emit_mapped_ipv4: bool = True

    strip_url_schemes: bool = True
    strip_url_ports: bool = True

    """
    Bare decimal DWORDs are extremely noisy.

    When alternative formats are enabled, require the token to have
    explicit evidence that it is intended as an IP representation.
    """
    require_alt_ipv4_context: bool = True

    normalize_unicode: bool = True

    preserve_context: bool = True

    max_defang_passes: int = Field(
        3,
        ge=1,
        le=8,
    )


class IPAddressMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: IPVersion
    canonical: str

    embedded_ipv4: Optional[str] = None

    is_global: bool = False
    is_private: bool = False
    is_shared: bool = False
    is_loopback: bool = False
    is_link_local: bool = False
    is_multicast: bool = False
    is_unspecified: bool = False
    is_reserved: bool = False

    scope: IPAddressScope = IPAddressScope.UNKNOWN

    is_documentation: bool = False
    is_benchmark: bool = False
    is_unique_local: bool = False

    is_teredo: bool = False
    is_6to4: bool = False
    is_orchid: bool = False
    is_translation: bool = False

    zone_index_present: bool = False


class IPIOC(BaseModel):
    """
    Canonical extracted IP IOC.

    start/end always refer to the ORIGINAL source text.

    normalized_value is the value actually suitable for downstream
    reputation processing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ioc_id: str

    kind: str = "ip"
    version: IPVersion

    raw_value: str
    normalized_value: str

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    confidence: float = Field(ge=0.0, le=1.0)

    scope: IPAddressScope
    metadata: IPAddressMetadata

    normalization_actions: tuple[NormalizationAction, ...] = ()

    context: Optional[str] = None

    source: str = "text"

    derived_from: Optional[str] = None

    """
    For IPv4-mapped IPv6:
        mapped_ipv4 = 192.0.2.1

    The extractor may additionally emit a separate IPv4 IOC when
    configured to do so.
    """
    mapped_ipv4: Optional[str] = None


class IPExtractionFailure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: RejectionReason
    count: int = Field(default=1, ge=1)


class IPExtractionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool

    iocs: tuple[IPIOC, ...] = ()

    total_candidates: int = 0
    accepted_candidates: int = 0
    rejected_candidates: int = 0

    truncated: bool = False
    timed_out: bool = False

    failures: tuple[IPExtractionFailure, ...] = ()

    duration_ms: float = 0.0

    extractor_version: str = MODULE_VERSION


# ============================================================================
# Metrics
# ============================================================================


@dataclass
class _MetricsSnapshot:
    calls: int = 0
    candidates_seen: int = 0
    candidates_accepted: int = 0
    candidates_rejected: int = 0
    ipv4: int = 0
    ipv6: int = 0
    mapped_ipv6: int = 0
    alternative_ipv4: int = 0
    duplicates: int = 0
    zone_rejections: int = 0
    timeout_count: int = 0
    errors: int = 0
    context_truncations: int = 0
    mapped_ipv4_emissions: int = 0


class IPExtractorMetrics:
    """
    Thread-safe metrics.

    Rather than taking a lock for every counter increment, each thread
    accumulates counters locally and periodically flushes them.

    This greatly reduces lock contention under concurrent extraction.
    """

    _FIELDS = tuple(_MetricsSnapshot.__annotations__.keys())

    def __init__(self, flush_every: int = 128) -> None:
        self._lock = threading.Lock()
        self._global = _MetricsSnapshot()
        self._local = threading.local()
        self._flush_every = max(1, flush_every)

    def _get_local(self) -> dict[str, int]:
        counters = getattr(self._local, "counters", None)

        if counters is None:
            counters = {field: 0 for field in self._FIELDS}
            counters["_updates"] = 0
            self._local.counters = counters

        return counters

    def increment(self, field: str, amount: int = 1) -> None:
        if field not in self._FIELDS:
            return

        counters = self._get_local()
        counters[field] += amount
        counters["_updates"] += 1

        if counters["_updates"] >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        counters = getattr(self._local, "counters", None)

        if not counters:
            return

        updates = counters.get("_updates", 0)

        if updates <= 0:
            return

        with self._lock:
            for field in self._FIELDS:
                self._global.__dict__[field] += counters[field]

        for field in self._FIELDS:
            counters[field] = 0

        counters["_updates"] = 0

    def snapshot(self) -> dict[str, int]:
        self.flush()

        with self._lock:
            return dict(self._global.__dict__)

    def reset(self) -> None:
        self.flush()

        with self._lock:
            self._global = _MetricsSnapshot()


_GLOBAL_METRICS = IPExtractorMetrics()


# ============================================================================
# Candidate representation
# ============================================================================


@dataclass(frozen=True)
class _Candidate:
    start: int
    end: int
    raw: str
    kind: CandidateKind
    priority: int


# ============================================================================
# Regexes
# ============================================================================

# Normal IPv4 candidate.
#
# IMPORTANT:
# Do not use nested \s* structures here.
#
# The candidate is deliberately broad enough for defanged forms, while
# actual validation is performed by ipaddress.
_IPV4_RE = re.compile(
    r"""
    (?<![A-Za-z0-9_])
    (?P<ipv4>
        \d{1,3}
        (?:
            (?:\s*(?:\.|\[\.\]|\(\.\)|\{\.\}|\[dot\]|\(dot\)|\{dot\})\s*)
            \d{1,3}
        ){3}
    )
    (?![A-Za-z0-9_])
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bracketed IPv6:
#   [2001:db8::1]
#   [2001:db8::1]:443
_BRACKETED_IPV6_RE = re.compile(
    r"""
    (?<![A-Za-z0-9_])
    \[
        (?P<ipv6>
            [0-9A-Fa-f:.%]+
        )
    \]
    """,
    re.VERBOSE,
)

# IPv6 candidates.
#
# This intentionally uses a broad bounded candidate rather than attempting
# to reproduce the entire IPv6 grammar in regex. ipaddress.IPv6Address()
# is the authoritative parser.
_IPV6_RE = re.compile(
    r"""
    (?<![A-Za-z0-9_])
    (?P<ipv6>
        (?:
            [0-9A-Fa-f]{0,4}:){1,7}
            [0-9A-Fa-f:.%]{1,39}
        )
    (?![A-Za-z0-9_])
    """,
    re.IGNORECASE | re.VERBOSE,
)

# URL-like host candidates.
#
# The actual URL parser is not performed here because this module is an IP
# extractor. This regex simply discovers likely IP-looking host material.
_URL_SCHEME_RE = re.compile(
    r"""
    (?ix)
    (?:
        hxxps?|https?|ftp
    )
    \s*
    (?:
        :\s*/\s*/|
        \[:\s*/\s*/\]|
        \[\:\]\s*/\s*/
    )
    (?P<host>
        \[
            [0-9A-Fa-f:.%]+\
        \]
        |
        [0-9A-Fa-f:.%]+
    )
    (?:
        \s*:\s*\d{1,5}
    )?
    """
)

# Explicit hexadecimal DWORD:
#   0x7f000001
_HEX_DWORD_RE = re.compile(
    r"(?<![A-Za-z0-9_])0[xX][0-9A-Fa-f]{8}(?![A-Za-z0-9_])"
)

# Explicit octal DWORD:
#   017700000001
_OCTAL_DWORD_RE = re.compile(
    r"(?<![A-Za-z0-9_])0[0-7]{8,11}(?![A-Za-z0-9_])"
)

# Dotted octal:
#   0177.0000.0000.0001
_OCTAL_DOTTED_RE = re.compile(
    r"""
    (?<![A-Za-z0-9_])
    0[0-7]{1,3}
    \.
    0[0-7]{1,3}
    \.
    0[0-7]{1,3}
    \.
    0[0-7]{1,3}
    (?![A-Za-z0-9_])
    """,
    re.VERBOSE,
)

# Decimal DWORD.
#
# Deliberately narrow and disabled by default.
_DECIMAL_DWORD_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[1-9][0-9]{8,9})(?![A-Za-z0-9_])"
)

# Zone identifier:
#   fe80::1%eth0
_ZONE_RE = re.compile(r"%[A-Za-z0-9_.~-]{1,32}")

# URL scheme.
_SCHEME_RE = re.compile(
    r"^(?:https?|ftp)\s*:\s*/\s*/",
    re.IGNORECASE,
)

# Wrappers commonly seen in reports.
_WRAPPER_CHARS = "\"'`<>(),;{}"

# Defanged replacements.
#
# Order matters: longer/more explicit forms first.
_DEFANG_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(?i)\[\s*dot\s*\]|\(\s*dot\s*\)|\{\s*dot\s*\}"),
        ".",
    ),
    (
        re.compile(r"(?i)\[\s*:\s*\]|\(\s*:\s*\)|\{\s*:\s*\}"),
        ":",
    ),
    (
        re.compile(r"(?i)\[\s*/\s*\]|\(\s*/\s*\)|\{\s*/\s*\}"),
        "/",
    ),
    (
        re.compile(r"(?i)\[\s*\.\s*\]|\(\s*\.\s*\)|\{\s*\.\s*\}"),
        ".",
    ),
    (
        re.compile(r"(?i)\[\s*colon\s*\]|\(\s*colon\s*\)"),
        ":",
    ),
    (
        re.compile(r"(?i)\[\s*slash\s*\]|\(\s*slash\s*\)"),
        "/",
    ),
    (
        re.compile(r"(?i)hxxps"),
        "https",
    ),
    (
        re.compile(r"(?i)hxxp"),
        "http",
    ),
    (
        re.compile(r"(?i)h\s*t\s*t\s*p\s*s"),
        "https",
    ),
    (
        re.compile(r"(?i)h\s*t\s*t\s*p"),
        "http",
    ),
    (
        re.compile(r"(?i)https\s*\[\s*:\s*\]"),
        "https:",
    ),
    (
        re.compile(r"(?i)http\s*\[\s*:\s*\]"),
        "http:",
    ),
    (
        re.compile(r"[．。｡﹒﹕]"),
        ".",
    ),
    (
        re.compile(r"[\u200b\u200c\u200d\ufeff]"),
        "",
    ),
)


# ============================================================================
# Special-purpose networks
# ============================================================================


def _network_contains(
    address: ipaddress._BaseAddress,
    networks: Sequence[ipaddress._BaseNetwork],
) -> bool:
    for network in networks:
        if address in network:
            return True
    return False


# RFC 5737 documentation ranges.
_DOCUMENTATION_V4 = (
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
)

# IPv6 documentation prefix RFC 3849.
_DOCUMENTATION_V6 = (
    ipaddress.ip_network("2001:db8::/32"),
)

# RFC 6598 shared address space.
_SHARED_V4 = (
    ipaddress.ip_network("100.64.0.0/10"),
)

# RFC 1918 private address space.
_PRIVATE_V4 = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)

# RFC 4193 ULA.
_UNIQUE_LOCAL_V6 = (
    ipaddress.ip_network("fc00::/7"),
)

# RFC 2544 benchmark testing.
_BENCHMARK_V4 = (
    ipaddress.ip_network("198.18.0.0/15"),
)

# 6to4.
_SIXTOFOUR_V6 = (
    ipaddress.ip_network("2002::/16"),
)

# Teredo.
_TEREDO_V6 = (
    ipaddress.ip_network("2001::/32"),
)

# ORCHID / ORCHIDv2.
_ORCHID_V6 = (
    ipaddress.ip_network("2001:10::/28"),
    ipaddress.ip_network("2001:20::/28"),
)

# IPv6 translation-related ranges.
_TRANSLATION_V6 = (
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
)


# ============================================================================
# Utility functions
# ============================================================================


def _safe_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "strict")).hexdigest()[:32]


def _stable_ioc_id(
    normalized_value: str,
    version: IPVersion,
) -> str:
    material = f"{MODULE_NAME}|{version.value}|{normalized_value}"
    return _safe_hash(material)


def _is_control_heavy(value: str) -> bool:
    """
    Reject dangerous control characters.

    Newlines/tabs are allowed in surrounding source text but not inside
    the actual candidate.
    """
    for char in value:
        code = ord(char)

        if code < 32 or code == 127:
            if char not in ("\t", " "):
                return True

    return False


def _strip_wrappers(value: str) -> tuple[str, bool]:
    original = value
    value = value.strip()

    changed = value != original

    while value and value[0] in _WRAPPER_CHARS:
        value = value[1:]
        changed = True

    while value and value[-1] in _WRAPPER_CHARS:
        value = value[:-1]
        changed = True

    return value, changed


def _normalize_unicode(value: str) -> tuple[str, bool]:
    normalized = unicodedata.normalize("NFKC", value)
    return normalized, normalized != value


def _defang(value: str, max_passes: int) -> tuple[str, set[NormalizationAction]]:
    actions: set[NormalizationAction] = set()

    current = value

    for _ in range(max_passes):
        previous = current

        for pattern, replacement in _DEFANG_REPLACEMENTS:
            new_value, count = pattern.subn(replacement, current)

            if count:
                current = new_value

                if replacement == ".":
                    actions.add(NormalizationAction.DOT_RESTORED)
                elif replacement == ":":
                    actions.add(NormalizationAction.COLON_RESTORED)

        if current.lower() != previous.lower():
            actions.add(NormalizationAction.WRAPPER_REMOVED)

        if current == previous:
            break

    # Collapse whitespace only around separators, not arbitrary text.
    compact = re.sub(r"\s*(?:\.)\s*", ".", current)

    if compact != current:
        actions.add(NormalizationAction.WHITESPACE_REMOVED)
        current = compact

    compact = re.sub(r"\s*:\s*", ":", current)

    if compact != current:
        actions.add(NormalizationAction.WHITESPACE_REMOVED)
        current = compact

    compact = re.sub(r"\s*/\s*", "/", current)

    if compact != current:
        actions.add(NormalizationAction.WHITESPACE_REMOVED)
        current = compact

    return current.strip(), actions


def _extract_context(
    text: str,
    start: int,
    end: int,
    radius: int,
    max_chars: int,
) -> Optional[str]:
    if max_chars <= 0:
        return None

    left = max(0, start - radius)
    right = min(len(text), end + radius)

    context = text[left:right]

    # Avoid leaking terminal/control characters into evidence stores.
    context = "".join(
        char if (char.isprintable() or char in "\t\r\n") else " "
        for char in context
    )

    context = context.replace("\r", " ").replace("\n", " ")

    if len(context) > max_chars:
        center = start - left

        half = max_chars // 2

        begin = max(0, center - half)
        finish = min(len(context), begin + max_chars)

        context = context[begin:finish]

    return context


def _strip_url_container(value: str) -> tuple[str, set[NormalizationAction]]:
    actions: set[NormalizationAction] = set()

    candidate = value.strip()

    # Remove URL scheme.
    if _SCHEME_RE.match(candidate):
        candidate = _SCHEME_RE.sub("", candidate, count=1)
        actions.add(NormalizationAction.SCHEME_REMOVED)

    # Remove trailing URL path/query from an IP host.
    #
    # This is intentionally done only after the candidate has been
    # identified as IP-like. It prevents accidental modification of
    # legitimate IPv6 colons.
    slash_index = candidate.find("/")

    if slash_index >= 0:
        candidate = candidate[:slash_index]

    # Query/fragment without slash.
    for delimiter in ("?", "#"):
        index = candidate.find(delimiter)

        if index >= 0:
            candidate = candidate[:index]

    # Bracketed IPv6 + optional port.
    if candidate.startswith("["):
        close = candidate.find("]")

        if close > 0:
            inside = candidate[1:close]

            if close + 1 < len(candidate):
                rest = candidate[close + 1 :]

                if re.fullmatch(r":\d{1,5}", rest):
                    actions.add(NormalizationAction.PORT_REMOVED)

            candidate = inside
            actions.add(NormalizationAction.BRACKETS_REMOVED)

            return candidate, actions

    # IPv4 / unbracketed IPv6.
    #
    # Only strip a port when the remaining prefix is an unambiguous
    # IPv4 address. For IPv6, colon is part of the address.
    port_match = re.fullmatch(
        r"(?P<host>\d{1,3}(?:\.\d{1,3}){3}):(?P<port>\d{1,5})",
        candidate,
    )

    if port_match:
        candidate = port_match.group("host")
        actions.add(NormalizationAction.PORT_REMOVED)

    return candidate, actions


def _contains_url_scheme_context(text: str, start: int) -> bool:
    """
    Cheap local check used for alternative IPv4 forms.

    It prevents a bare decimal number such as:
        1700000000

    from automatically becoming an IP IOC.
    """
    left = max(0, start - 32)
    prefix = text[left:start]

    return bool(
        re.search(
            r"(?i)(?:https?|ftp|hxxps?|hxxp)\s*:\s*/\s*/\s*$",
            prefix,
        )
    )


def _decode_dword(value: str, base: int) -> Optional[str]:
    try:
        number = int(value, base)

        if not 0 <= number <= 0xFFFFFFFF:
            return None

        return str(ipaddress.IPv4Address(number))

    except (ValueError, TypeError, OverflowError):
        return None


def _decode_dotted_octal(value: str) -> Optional[str]:
    parts = value.split(".")

    if len(parts) != 4:
        return None

    octets: list[int] = []

    try:
        for part in parts:
            octet = int(part, 8)

            if not 0 <= octet <= 255:
                return None

            octets.append(octet)

        return ".".join(str(x) for x in octets)

    except (ValueError, TypeError):
        return None


def _classify_address(
    address: ipaddress._BaseAddress,
    zone_present: bool = False,
) -> IPAddressMetadata:
    is_v4 = address.version == 4

    mapped = getattr(address, "ipv4_mapped", None)

    if mapped is not None:
        embedded = str(mapped)

        # For mapped addresses, security-relevant classification must use
        # the embedded IPv4 semantics.
        effective = mapped
        version = IPVersion.IPV4_MAPPED_IPV6
    else:
        effective = address
        embedded = None
        version = IPVersion.IPV4 if is_v4 else IPVersion.IPV6

    is_shared = _network_contains(effective, _SHARED_V4)

    is_documentation = (
        _network_contains(effective, _DOCUMENTATION_V4)
        if effective.version == 4
        else _network_contains(effective, _DOCUMENTATION_V6)
    )

    is_benchmark = (
        effective.version == 4
        and _network_contains(effective, _BENCHMARK_V4)
    )

    is_unique_local = (
        effective.version == 6
        and _network_contains(effective, _UNIQUE_LOCAL_V6)
    )

    is_teredo = (
        effective.version == 6
        and _network_contains(effective, _TEREDO_V6)
    )

    is_6to4 = (
        effective.version == 6
        and _network_contains(effective, _SIXTOFOUR_V6)
    )

    is_orchid = (
        effective.version == 6
        and _network_contains(effective, _ORCHID_V6)
    )

    is_translation = (
        effective.version == 6
        and _network_contains(effective, _TRANSLATION_V6)
    )

    # Python's ipaddress implementation is used as the base classification.
    is_private = bool(effective.is_private)
    is_global = bool(effective.is_global)
    is_loopback = bool(effective.is_loopback)
    is_link_local = bool(effective.is_link_local)
    is_multicast = bool(effective.is_multicast)
    is_unspecified = bool(effective.is_unspecified)
    is_reserved = bool(effective.is_reserved)

    if is_documentation:
        scope = IPAddressScope.DOCUMENTATION
    elif is_benchmark:
        scope = IPAddressScope.BENCHMARK
    elif is_shared:
        scope = IPAddressScope.SHARED
    elif is_loopback:
        scope = IPAddressScope.LOOPBACK
    elif is_link_local:
        scope = IPAddressScope.LINK_LOCAL
    elif is_multicast:
        scope = IPAddressScope.MULTICAST
    elif is_unspecified:
        scope = IPAddressScope.UNSPECIFIED
    elif is_unique_local:
        scope = IPAddressScope.UNIQUE_LOCAL
    elif is_teredo:
        scope = IPAddressScope.TEREDO
    elif is_6to4:
        scope = IPAddressScope.SIXTOFOUR
    elif is_orchid:
        scope = IPAddressScope.ORCHID
    elif is_translation:
        scope = IPAddressScope.TRANSLATION
    elif is_private:
        scope = IPAddressScope.PRIVATE
    elif is_reserved:
        scope = IPAddressScope.RESERVED
    elif is_global:
        scope = IPAddressScope.GLOBAL
    else:
        scope = IPAddressScope.SPECIAL

    return IPAddressMetadata(
        version=version,
        canonical=str(address),
        embedded_ipv4=embedded,
        is_global=is_global,
        is_private=is_private,
        is_shared=is_shared,
        is_loopback=is_loopback,
        is_link_local=is_link_local,
        is_multicast=is_multicast,
        is_unspecified=is_unspecified,
        is_reserved=is_reserved,
        scope=scope,
        is_documentation=is_documentation,
        is_benchmark=is_benchmark,
        is_unique_local=is_unique_local,
        is_teredo=is_teredo,
        is_6tofour=False if False else is_6to4,
        is_orchid=is_orchid,
        is_translation=is_translation,
        zone_index_present=zone_present,
    )


# ============================================================================
# Main extractor
# ============================================================================


class IPExtractor:
    """
    Production IP IOC extractor.

    Instances are immutable with respect to configuration and can safely
    be shared across threads.
    """

    def __init__(
        self,
        config: Optional[IPExtractorConfig] = None,
        metrics: Optional[IPExtractorMetrics] = None,
    ) -> None:
        self.config = config or IPExtractorConfig()
        self.metrics = metrics or _GLOBAL_METRICS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        *,
        source: str = "text",
    ) -> IPExtractionResult:
        started = time.perf_counter()

        self.metrics.increment("calls")

        if not isinstance(text, str):
            return IPExtractionResult(
                success=False,
                failures=(
                    IPExtractionFailure(
                        reason=RejectionReason.INVALID_IP,
                    ),
                ),
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        if not text:
            return IPExtractionResult(
                success=True,
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        if len(text) > self.config.max_input_chars:
            return IPExtractionResult(
                success=False,
                failures=(
                    IPExtractionFailure(
                        reason=RejectionReason.TOO_LONG,
                    ),
                ),
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        deadline = started + self.config.max_extraction_seconds

        try:
            candidates = self._discover_candidates(
                text,
                deadline,
            )

            timed_out = time.perf_counter() >= deadline

            iocs: list[IPIOC] = []
            failures: dict[RejectionReason, int] = {}

            seen: dict[str, IPIOC] = {}

            for candidate in candidates:
                if time.perf_counter() >= deadline:
                    timed_out = True
                    break

                if len(iocs) >= self.config.max_iocs:
                    break

                self.metrics.increment("candidates_seen")

                try:
                    result = self._parse_candidate(
                        text,
                        candidate,
                        source=source,
                    )

                except Exception:
                    # Fail closed. Never expose internal exception details.
                    self.metrics.increment("errors")
                    failures[RejectionReason.INTERNAL_ERROR] = (
                        failures.get(RejectionReason.INTERNAL_ERROR, 0) + 1
                    )
                    continue

                if result is None:
                    self.metrics.increment("candidates_rejected")
                    continue

                primary, mapped = result

                self.metrics.increment("candidates_accepted")

                existing = seen.get(primary.normalized_value)

                if existing is not None:
                    self.metrics.increment("duplicates")

                    # Preserve the most informative candidate.
                    winner = self._merge_ioc(existing, primary)

                    seen[primary.normalized_value] = winner

                    continue

                seen[primary.normalized_value] = primary
                iocs.append(primary)

                if mapped is not None and self.config.emit_mapped_ipv4:
                    mapped_existing = seen.get(mapped.normalized_value)

                    if mapped_existing is None:
                        seen[mapped.normalized_value] = mapped
                        iocs.append(mapped)
                        self.metrics.increment("mapped_ipv4_emissions")

            # Reconstruct deterministic source order.
            iocs = sorted(
                seen.values(),
                key=lambda item: (
                    item.start,
                    item.end,
                    item.normalized_value,
                ),
            )

            if len(iocs) > self.config.max_iocs:
                iocs = iocs[: self.config.max_iocs]

            rejected = sum(failures.values())

            if timed_out:
                self.metrics.increment("timeout_count")

            return IPExtractionResult(
                success=True,
                iocs=tuple(iocs),
                total_candidates=len(candidates),
                accepted_candidates=len(iocs),
                rejected_candidates=rejected,
                truncated=len(iocs) >= self.config.max_iocs,
                timed_out=timed_out,
                failures=tuple(
                    IPExtractionFailure(
                        reason=reason,
                        count=count,
                    )
                    for reason, count in sorted(
                        failures.items(),
                        key=lambda item: item[0].value,
                    )
                ),
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        except Exception:
            self.metrics.increment("errors")

            return IPExtractionResult(
                success=False,
                failures=(
                    IPExtractionFailure(
                        reason=RejectionReason.INTERNAL_ERROR,
                    ),
                ),
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

    async def extract_async(
        self,
        text: str,
        *,
        source: str = "text",
    ) -> IPExtractionResult:
        return await asyncio.to_thread(
            self.extract,
            text,
            source=source,
        )

    def extract_values(self, text: str) -> tuple[str, ...]:
        result = self.extract(text)
        return tuple(ioc.normalized_value for ioc in result.iocs)

    def metrics_snapshot(self) -> dict[str, int]:
        return self.metrics.snapshot()

    # ------------------------------------------------------------------
    # Candidate discovery
    # ------------------------------------------------------------------

    def _discover_candidates(
        self,
        text: str,
        deadline: float,
    ) -> list[_Candidate]:
        """
        Discover broad candidates, then deconflict intervals.

        We deliberately avoid trying to make one giant regex understand
        every IPv4/IPv6 grammar variant. The final parser is ipaddress.
        """

        candidates: list[_Candidate] = []

        def add_matches(
            pattern: re.Pattern[str],
            kind: CandidateKind,
            priority: int,
            group: Optional[str] = None,
        ) -> None:
            if len(candidates) >= self.config.max_candidates:
                return

            for match in pattern.finditer(text):
                if time.perf_counter() >= deadline:
                    return

                if len(candidates) >= self.config.max_candidates:
                    return

                value = (
                    match.group(group)
                    if group is not None
                    else match.group(0)
                )

                start = (
                    match.start(group)
                    if group is not None
                    else match.start()
                )

                end = (
                    match.end(group)
                    if group is not None
                    else match.end()
                )

                if not value:
                    continue

                if len(value) > self.config.max_candidate_chars:
                    continue

                candidates.append(
                    _Candidate(
                        start=start,
                        end=end,
                        raw=value,
                        kind=kind,
                        priority=priority,
                    )
                )

        # URL candidates receive highest priority.
        add_matches(
            _URL_SCHEME_RE,
            CandidateKind.URL_HOST,
            priority=100,
            group="host",
        )

        # Bracketed IPv6 must outrank generic IPv6.
        add_matches(
            _BRACKETED_IPV6_RE,
            CandidateKind.BRACKETED_IPV6,
            priority=90,
            group="ipv6",
        )

        add_matches(
            _IPV4_RE,
            CandidateKind.IPV4,
            priority=80,
            group="ipv4",
        )

        add_matches(
            _IPV6_RE,
            CandidateKind.IPV6,
            priority=70,
            group="ipv6",
        )

        if self.config.allow_alternative_ipv4_formats:
            add_matches(
                _HEX_DWORD_RE,
                CandidateKind.ALTERNATIVE_IPV4,
                priority=60,
            )

            add_matches(
                _OCTAL_DOTTED_RE,
                CandidateKind.ALTERNATIVE_IPV4,
                priority=60,
            )

            add_matches(
                _OCTAL_DWORD_RE,
                CandidateKind.ALTERNATIVE_IPV4,
                priority=50,
            )

            add_matches(
                _DECIMAL_DWORD_RE,
                CandidateKind.ALTERNATIVE_IPV4,
                priority=40,
            )

        return self._deconflict_candidates(candidates)

    @staticmethod
    def _deconflict_candidates(
        candidates: Iterable[_Candidate],
    ) -> list[_Candidate]:
        """
        Greedy interval deconfliction.

        Higher priority wins. For equal priority:
        - longer span wins
        - earlier span wins
        """

        ordered = sorted(
            candidates,
            key=lambda item: (
                -item.priority,
                -(item.end - item.start),
                item.start,
                item.end,
            ),
        )

        selected: list[_Candidate] = []
        occupied: list[tuple[int, int]] = []

        for candidate in ordered:
            overlaps = False

            for start, end in occupied:
                if candidate.start < end and candidate.end > start:
                    overlaps = True
                    break

            if overlaps:
                continue

            selected.append(candidate)
            occupied.append((candidate.start, candidate.end))

        return sorted(
            selected,
            key=lambda item: (
                item.start,
                item.end,
                -item.priority,
            ),
        )

    # ------------------------------------------------------------------
    # Candidate parsing
    # ------------------------------------------------------------------

    def _parse_candidate(
        self,
        text: str,
        candidate: _Candidate,
        *,
        source: str,
    ) -> Optional[tuple[IPIOC, Optional[IPIOC]]]:
        raw = candidate.raw

        if not raw:
            return None

        if len(raw) > self.config.max_candidate_chars:
            return None

        if _is_control_heavy(raw):
            return None

        value = raw

        actions: set[NormalizationAction] = set()

        if self.config.normalize_unicode:
            value, changed = _normalize_unicode(value)

            if changed:
                actions.add(NormalizationAction.UNICODE_NORMALIZED)

        value, defang_actions = _defang(
            value,
            self.config.max_defang_passes,
        )

        actions.update(defang_actions)

        value, wrapper_changed = _strip_wrappers(value)

        if wrapper_changed:
            actions.add(NormalizationAction.WRAPPER_REMOVED)

        # URL container removal.
        value, url_actions = _strip_url_container(value)
        actions.update(url_actions)

        if not value:
            return None

        # Explicitly detect zone identifiers before parsing.
        zone_present = bool(_ZONE_RE.search(value))

        if zone_present:
            if self.config.reject_zone_indices:
                self.metrics.increment("zone_rejections")
                return None

            actions.add(NormalizationAction.ZONE_DETECTED)

        # ipaddress accepts IPv6Address("fe80::1%eth0") in modern Python
        # in some contexts, but IOC normalization should not preserve a
        # zone identifier as an Internet-wide identity.
        if "%" in value:
            value_without_zone = value.split("%", 1)[0]
        else:
            value_without_zone = value

        # Alternative representations are handled before normal parsing.
        alternative = self._parse_alternative_ipv4(
            text,
            candidate,
            value_without_zone,
        )

        if alternative is not None:
            value_without_zone = alternative
            actions.add(NormalizationAction.ALTERNATIVE_IPV4_DECODED)

        try:
            address = ipaddress.ip_address(value_without_zone)
        except ValueError:
            return None

        # Canonical textual representation.
        normalized = str(address)

        # IPv4-mapped IPv6.
        mapped_ipv4 = getattr(address, "ipv4_mapped", None)

        if mapped_ipv4 is not None:
            actions.add(NormalizationAction.MAPPED_IPV4_DERIVED)

        metadata = _classify_address(
            address,
            zone_present=zone_present,
        )

        version = metadata.version

        # Policy filtering.
        if metadata.is_private and not self.config.allow_private:
            return None

        if metadata.is_multicast and not self.config.allow_multicast:
            return None

        special_scope = {
            IPAddressScope.DOCUMENTATION,
            IPAddressScope.BENCHMARK,
            IPAddressScope.RESERVED,
            IPAddressScope.SPECIAL,
            IPAddressScope.TRANSLATION,
            IPAddressScope.TEREDO,
            IPAddressScope.SIXTOFOUR,
            IPAddressScope.ORCHID,
            IPAddressScope.SHARED,
            IPAddressScope.UNIQUE_LOCAL,
        }

        if (
            metadata.scope in special_scope
            and not self.config.allow_special
        ):
            return None

        confidence = self._calculate_confidence(
            raw=raw,
            normalized=normalized,
            candidate=candidate,
            metadata=metadata,
            actions=actions,
        )

        context = None

        if self.config.preserve_context:
            context = _extract_context(
                text,
                candidate.start,
                candidate.end,
                self.config.context_radius,
                self.config.max_context_chars,
            )

        primary = IPIOC(
            ioc_id=_stable_ioc_id(normalized, version),
            version=version,
            raw_value=raw,
            normalized_value=normalized,
            start=candidate.start,
            end=candidate.end,
            confidence=confidence,
            scope=metadata.scope,
            metadata=metadata,
            normalization_actions=tuple(
                sorted(actions, key=lambda item: item.value)
            ),
            context=context,
            source=source,
            mapped_ipv4=(
                str(mapped_ipv4)
                if mapped_ipv4 is not None
                else None
            ),
        )

        self._record_version_metric(version)

        mapped_ioc: Optional[IPIOC] = None

        # Important:
        # IPv4-mapped IPv6 is an IPv6 textual IOC but its embedded IPv4
        # should also be independently visible to downstream IP reputation.
        if (
            mapped_ipv4 is not None
            and self.config.emit_mapped_ipv4
        ):
            mapped_text = str(mapped_ipv4)

            mapped_metadata = _classify_address(mapped_ipv4)

            mapped_ioc = IPIOC(
                ioc_id=_stable_ioc_id(
                    mapped_text,
                    IPVersion.IPV4,
                ),
                version=IPVersion.IPV4,
                raw_value=raw,
                normalized_value=mapped_text,
                start=candidate.start,
                end=candidate.end,
                confidence=max(
                    0.0,
                    min(1.0, confidence - 0.01),
                ),
                scope=mapped_metadata.scope,
                metadata=mapped_metadata,
                normalization_actions=tuple(
                    sorted(
                        actions | {
                            NormalizationAction.MAPPED_IPV4_DERIVED
                        },
                        key=lambda item: item.value,
                    )
                ),
                context=context,
                source=source,
                derived_from=normalized,
                mapped_ipv4=None,
            )

        return primary, mapped_ioc

    # ------------------------------------------------------------------
    # Alternative IPv4 parsing
    # ------------------------------------------------------------------

    def _parse_alternative_ipv4(
        self,
        text: str,
        candidate: _Candidate,
        value: str,
    ) -> Optional[str]:
        if not self.config.allow_alternative_ipv4_formats:
            return None

        if candidate.kind != CandidateKind.ALTERNATIVE_IPV4:
            return None

        # Hex DWORD.
        if re.fullmatch(r"0[xX][0-9A-Fa-f]{8}", value):
            return _decode_dword(value[2:], 16)

        # Dotted octal.
        if re.fullmatch(
            r"0[0-7]{1,3}(?:\.0[0-7]{1,3}){3}",
            value,
        ):
            return _decode_dotted_octal(value)

        # Octal DWORD.
        if re.fullmatch(r"0[0-7]{8,11}", value):
            return _decode_dword(value, 8)

        # Decimal DWORD.
        if re.fullmatch(r"[1-9][0-9]{8,9}", value):
            if self.config.require_alt_ipv4_context:
                if not _contains_url_scheme_context(
                    text,
                    candidate.start,
                ):
                    return None

            return _decode_dword(value, 10)

        return None

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_confidence(
        *,
        raw: str,
        normalized: str,
        candidate: _Candidate,
        metadata: IPAddressMetadata,
        actions: set[NormalizationAction],
    ) -> float:
        """
        Extraction confidence, NOT maliciousness.

        This measures how confidently we believe the source token really
        represents an IP address.

        It must never encode reputation.
        """

        score = 0.90

        if candidate.kind == CandidateKind.URL_HOST:
            score += 0.07

        elif candidate.kind == CandidateKind.BRACKETED_IPV6:
            score += 0.06

        elif candidate.kind == CandidateKind.ALTERNATIVE_IPV4:
            score -= 0.20

        if metadata.version == IPVersion.IPV4_MAPPED_IPV6:
            score -= 0.01

        if NormalizationAction.DOT_RESTORED in actions:
            score -= 0.03

        if NormalizationAction.COLON_RESTORED in actions:
            score -= 0.03

        if NormalizationAction.UNICODE_NORMALIZED in actions:
            score -= 0.02

        if NormalizationAction.ALTERNATIVE_IPV4_DECODED in actions:
            score -= 0.15

        if raw != normalized:
            score -= 0.01

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Merge / provenance
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_ioc(
        existing: IPIOC,
        incoming: IPIOC,
    ) -> IPIOC:
        """
        Preserve the best representation while retaining the most useful
        provenance.

        The canonical IOC ID remains stable.
        """

        # Prefer:
        # 1. explicit URL candidate
        # 2. bracketed IPv6
        # 3. normal IP
        # 4. alternative representation
        def rank(ioc: IPIOC) -> tuple[int, float, int]:
            actions = set(ioc.normalization_actions)

            if NormalizationAction.ALTERNATIVE_IPV4_DECODED in actions:
                kind_rank = 1
            elif NormalizationAction.SCHEME_REMOVED in actions:
                kind_rank = 4
            elif NormalizationAction.BRACKETS_REMOVED in actions:
                kind_rank = 3
            else:
                kind_rank = 2

            return (
                kind_rank,
                ioc.confidence,
                ioc.end - ioc.start,
            )

        winner = max(
            (existing, incoming),
            key=rank,
        )

        # If the incoming source gives a larger context, preserve it.
        context = winner.context

        if (
            incoming.context
            and (
                not context
                or len(incoming.context) > len(context)
            )
        ):
            context = incoming.context

        actions = tuple(
            sorted(
                set(existing.normalization_actions)
                | set(incoming.normalization_actions),
                key=lambda item: item.value,
            )
        )

        return winner.model_copy(
            update={
                "normalization_actions": actions,
                "context": context,
            }
        )

    def _record_version_metric(
        self,
        version: IPVersion,
    ) -> None:
        if version == IPVersion.IPV4:
            self.metrics.increment("ipv4")
        elif version == IPVersion.IPV6:
            self.metrics.increment("ipv6")
        elif version == IPVersion.IPV4_MAPPED_IPV6:
            self.metrics.increment("mapped_ipv6")


# ============================================================================
# Singleton / convenience API
# ============================================================================


_default_extractor: Optional[IPExtractor] = None
_default_extractor_lock = threading.Lock()


def get_default_ip_extractor() -> IPExtractor:
    global _default_extractor

    if _default_extractor is None:
        with _default_extractor_lock:
            if _default_extractor is None:
                _default_extractor = IPExtractor()

    return _default_extractor


def extract_ip_iocs(
    text: str,
    *,
    config: Optional[IPExtractorConfig] = None,
    source: str = "text",
) -> IPExtractionResult:
    """
    Convenience API.

    Default configuration uses the shared singleton so metrics remain
    process-wide and inexpensive.

    A custom configuration creates an independent extractor because the
    configuration changes extraction semantics.
    """

    if config is None:
        extractor = get_default_ip_extractor()
    else:
        extractor = IPExtractor(config=config)

    return extractor.extract(
        text,
        source=source,
    )


async def extract_ip_iocs_async(
    text: str,
    *,
    config: Optional[IPExtractorConfig] = None,
    source: str = "text",
) -> IPExtractionResult:
    if config is None:
        extractor = get_default_ip_extractor()
    else:
        extractor = IPExtractor(config=config)

    return await extractor.extract_async(
        text,
        source=source,
    )


def extract_ip_values(
    text: str,
    *,
    config: Optional[IPExtractorConfig] = None,
) -> tuple[str, ...]:
    result = extract_ip_iocs(
        text,
        config=config,
    )

    return tuple(
        ioc.normalized_value
        for ioc in result.iocs
    )


# ============================================================================
# Compatibility helpers
# ============================================================================


def normalize_ip(value: str) -> Optional[str]:
    """
    Lightweight canonicalization helper.

    This intentionally does not accept alternative representations.
    """

    if not isinstance(value, str):
        return None

    candidate = value.strip()

    candidate, _ = _defang(
        candidate,
        max_passes=3,
    )

    candidate, _ = _strip_wrappers(candidate)

    candidate, _ = _strip_url_container(candidate)

    if "%" in candidate:
        return None

    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return None


def is_valid_ip(value: str) -> bool:
    return normalize_ip(value) is not None


def classify_ip(value: str) -> Optional[IPAddressMetadata]:
    normalized = normalize_ip(value)

    if normalized is None:
        return None

    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return None

    return _classify_address(address)


__all__ = [
    "MODULE_NAME",
    "MODULE_VERSION",
    "IPVersion",
    "IPAddressScope",
    "CandidateKind",
    "RejectionReason",
    "NormalizationAction",
    "IPExtractorConfig",
    "IPAddressMetadata",
    "IPIOC",
    "IPExtractionFailure",
    "IPExtractionResult",
    "IPExtractorMetrics",
    "IPExtractor",
    "get_default_ip_extractor",
    "extract_ip_iocs",
    "extract_ip_iocs_async",
    "extract_ip_values",
    "normalize_ip",
    "is_valid_ip",
    "classify_ip",
]
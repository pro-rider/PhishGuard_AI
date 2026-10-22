"""
PhishGuard AI - Hash Reputation & File Identity Intelligence Engine
====================================================================

Version: 16.0.0
File Path: reputation/hash_reputation.py

Purpose
-------
Production-grade hash reputation analysis for PhishGuard AI.

Design principles
-----------------
1. NOT OBSERVED != CLEAN
2. Threat-intelligence evidence is stronger than local heuristics.
3. Hash algorithm quality affects confidence, never maliciousness by itself.
4. Exact hash matches and fuzzy matches are separate evidence classes.
5. Provider failures are explicit degraded states.
6. File-level analysis correlates multiple hashes.
7. Cache identity includes relevant configuration.
8. Request coalescing prevents duplicate provider requests.
9. Force-refresh requests cannot corrupt normal in-flight state.
10. All externally supplied provider data is bounded and normalized.
11. File hashing supports TOCTOU detection.
12. Legacy hashes remain supported for IOC interoperability but are
    explicitly identified as legacy.
13. No provider-specific scoring assumptions are required.
14. The engine is deterministic for a fixed evidence set.
15. No network requests are performed by this module itself.

Cryptographic references
------------------------
SHA-224/256/384/512 and SHA-512/224/256 are specified by NIST FIPS 180-4.
SHA3-224/256/384/512 are specified by NIST FIPS 202.

This module intentionally does not implement cryptographic algorithms itself;
it delegates to Python's hashlib implementation.

Security note
-------------
A hash is an identity/fingerprint. A hash reputation result is not proof
that a file is malicious unless the underlying evidence supports that
conclusion. In particular:

    provider returned "not found"
        !=
    provider proved the object is benign.

"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from threat_intel.base import IOCType, ThreatSeverity
except ImportError:  # pragma: no cover
    IOCType = Any  # type: ignore
    ThreatSeverity = Any  # type: ignore

try:
    from threat_intel.aggregator import (
        AggregatedThreatSummary,
        ThreatIntelAggregator,
    )
except ImportError:  # pragma: no cover
    AggregatedThreatSummary = Any  # type: ignore
    ThreatIntelAggregator = Any  # type: ignore


# =============================================================================
# VERSION / LIMITS
# =============================================================================

MODULE_VERSION = "16.0.0"

DEFAULT_CACHE_TTL_SECONDS = 900
DEFAULT_CACHE_MAX_ENTRIES = 10_000

DEFAULT_MAX_HASH_LENGTH = 256
DEFAULT_MAX_REASONS = 32
DEFAULT_MAX_REASON_LENGTH = 256
DEFAULT_MAX_SIGNALS = 64
DEFAULT_MAX_PROVIDER_OBSERVATIONS = 64
DEFAULT_MAX_PROVIDER_TAGS = 16
DEFAULT_MAX_PROVIDER_NAME_LENGTH = 128

DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024
DEFAULT_MAX_FILE_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB

DEFAULT_PROVIDER_TIMEOUT_SECONDS = 8.0
DEFAULT_FUZZY_TIMEOUT_SECONDS = 5.0

DEFAULT_TI_CONCURRENCY = 32
DEFAULT_FUZZY_CONCURRENCY = 8
DEFAULT_FILE_HASH_CONCURRENCY = 4

DEFAULT_TOC_TOCTOU_RECHECK = True


# =============================================================================
# HASH ALGORITHMS
# =============================================================================

class HashAlgorithm(str, Enum):
    """Supported fixed-length digest algorithms."""

    MD5 = "md5"
    SHA1 = "sha1"

    SHA224 = "sha224"
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"

    SHA512_224 = "sha512_224"
    SHA512_256 = "sha512_256"

    SHA3_224 = "sha3_224"
    SHA3_256 = "sha3_256"
    SHA3_384 = "sha3_384"
    SHA3_512 = "sha3_512"

    BLAKE2S = "blake2s"
    BLAKE2B = "blake2b"


class HashMatchType(str, Enum):
    """
    Reputation match classification.

    EXACT:
        At least one trusted TI provider positively matched the digest.

    FUZZY:
        A fuzzy provider produced a bounded similarity match.

    NONE:
        Providers responded successfully but no positive match was observed.

    UNKNOWN:
        Evidence is insufficient or provider coverage is unavailable.

    ALLOWLIST:
        A local explicit allowlist matched.

    """

    EXACT = "exact"
    FUZZY = "fuzzy"
    NONE = "none"
    UNKNOWN = "unknown"
    ALLOWLIST = "allowlist"


class HashReputationStatus(str, Enum):
    COMPLETE = "complete"
    DEGRADED = "degraded"
    INVALID_INPUT = "invalid_input"
    UNKNOWN = "unknown"


class HashRiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CLEAN = "clean"
    UNKNOWN = "unknown"


class HashEvidenceType(str, Enum):
    THREAT_INTEL = "threat_intel"
    FUZZY = "fuzzy"
    ALLOWLIST = "allowlist"
    ALGORITHM = "algorithm"
    FILE = "file"
    INPUT = "input"


# =============================================================================
# ALGORITHM METADATA
# =============================================================================

@dataclass(frozen=True)
class _AlgorithmSpec:
    name: HashAlgorithm
    digest_length: int
    hashlib_name: str
    legacy: bool
    security_confidence: float


ALGORITHM_SPECS: Mapping[HashAlgorithm, _AlgorithmSpec] = {
    HashAlgorithm.MD5: _AlgorithmSpec(
        HashAlgorithm.MD5, 32, "md5", True, 0.45
    ),
    HashAlgorithm.SHA1: _AlgorithmSpec(
        HashAlgorithm.SHA1, 40, "sha1", True, 0.55
    ),
    HashAlgorithm.SHA224: _AlgorithmSpec(
        HashAlgorithm.SHA224, 56, "sha224", False, 0.82
    ),
    HashAlgorithm.SHA256: _AlgorithmSpec(
        HashAlgorithm.SHA256, 64, "sha256", False, 0.95
    ),
    HashAlgorithm.SHA384: _AlgorithmSpec(
        HashAlgorithm.SHA384, 96, "sha384", False, 0.96
    ),
    HashAlgorithm.SHA512: _AlgorithmSpec(
        HashAlgorithm.SHA512, 128, "sha512", False, 0.97
    ),
    HashAlgorithm.SHA512_224: _AlgorithmSpec(
        HashAlgorithm.SHA512_224, 56, "sha512_224", False, 0.88
    ),
    HashAlgorithm.SHA512_256: _AlgorithmSpec(
        HashAlgorithm.SHA512_256, 64, "sha512_256", False, 0.92
    ),
    HashAlgorithm.SHA3_224: _AlgorithmSpec(
        HashAlgorithm.SHA3_224, 56, "sha3_224", False, 0.86
    ),
    HashAlgorithm.SHA3_256: _AlgorithmSpec(
        HashAlgorithm.SHA3_256, 64, "sha3_256", False, 0.95
    ),
    HashAlgorithm.SHA3_384: _AlgorithmSpec(
        HashAlgorithm.SHA3_384, 96, "sha3_384", False, 0.96
    ),
    HashAlgorithm.SHA3_512: _AlgorithmSpec(
        HashAlgorithm.SHA3_512, 128, "sha3_512", False, 0.98
    ),
    HashAlgorithm.BLAKE2S: _AlgorithmSpec(
        HashAlgorithm.BLAKE2S, 64, "blake2s", False, 0.92
    ),
    HashAlgorithm.BLAKE2B: _AlgorithmSpec(
        HashAlgorithm.BLAKE2B, 128, "blake2b", False, 0.97
    ),
}


DEFAULT_ALGORITHM_PRIORITY: Tuple[HashAlgorithm, ...] = (
    HashAlgorithm.SHA256,
    HashAlgorithm.SHA512,
    HashAlgorithm.SHA3_256,
    HashAlgorithm.SHA3_512,
    HashAlgorithm.SHA384,
    HashAlgorithm.SHA3_384,
    HashAlgorithm.SHA224,
    HashAlgorithm.SHA3_224,
    HashAlgorithm.SHA512_256,
    HashAlgorithm.SHA512_224,
    HashAlgorithm.BLAKE2B,
    HashAlgorithm.BLAKE2S,
    HashAlgorithm.SHA1,
    HashAlgorithm.MD5,
)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class HashReputationError(Exception):
    """Base exception."""


class InvalidHashError(HashReputationError):
    """Invalid digest or algorithm."""


class UnsupportedHashAlgorithmError(HashReputationError):
    """Unsupported algorithm."""


class HashResolutionError(HashReputationError):
    """Hash provider resolution error."""


class HashConfigurationError(HashReputationError):
    """Invalid engine configuration."""


class FileHashingError(HashReputationError):
    """File hashing failed."""


class FileChangedDuringHashingError(FileHashingError):
    """File changed during hashing."""


# =============================================================================
# INPUT / DIGEST MODELS
# =============================================================================

class HashIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: HashAlgorithm
    digest: str

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        value = value.strip().lower()

        if not value:
            raise ValueError("Digest cannot be empty.")

        if len(value) > DEFAULT_MAX_HASH_LENGTH:
            raise ValueError("Digest is too long.")

        if not re.fullmatch(r"[0-9a-f]+", value):
            raise ValueError("Digest must contain hexadecimal characters only.")

        return value

    @property
    def spec(self) -> _AlgorithmSpec:
        return ALGORITHM_SPECS[self.algorithm]

    @property
    def cache_key(self) -> str:
        digest_hash = hashlib.sha256(
            self.digest.encode("ascii")
        ).hexdigest()

        return (
            f"hash-reputation:v{MODULE_VERSION}:"
            f"{self.algorithm.value}:{digest_hash}"
        )


class HashDigestSet(BaseModel):
    """
    Complete digest set produced from one file.

    All supported fixed-length algorithms are represented.
    """

    model_config = ConfigDict(extra="forbid")

    md5: Optional[str] = None
    sha1: Optional[str] = None

    sha224: Optional[str] = None
    sha256: Optional[str] = None
    sha384: Optional[str] = None
    sha512: Optional[str] = None

    sha512_224: Optional[str] = None
    sha512_256: Optional[str] = None

    sha3_224: Optional[str] = None
    sha3_256: Optional[str] = None
    sha3_384: Optional[str] = None
    sha3_512: Optional[str] = None

    blake2s: Optional[str] = None
    blake2b: Optional[str] = None


class HashFileMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Optional[str] = None
    filename: Optional[str] = None
    size_bytes: int = Field(ge=0)
    mtime_ns: Optional[int] = Field(default=None, ge=0)

    inode: Optional[int] = Field(default=None, ge=0)

    content_changed_during_hash: bool = False
    hash_duration_ms: float = Field(default=0.0, ge=0.0)


# =============================================================================
# PROVIDER MODELS
# =============================================================================

class ProviderObservation(BaseModel):
    """
    Normalized provider evidence.

    This deliberately contains no secret material.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        min_length=1,
        max_length=DEFAULT_MAX_PROVIDER_NAME_LENGTH,
    )

    matched: bool = False
    malicious: bool = False

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    severity: Optional[str] = None

    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)

    latency_ms: float = Field(default=0.0, ge=0.0)

    error: Optional[str] = None


class FuzzyHashMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=128)
    algorithm: str = Field(min_length=1, max_length=32)

    similarity: float = Field(ge=0.0, le=1.0)

    matched: bool = False
    malicious: bool = False

    target_identifier: Optional[str] = Field(
        default=None,
        max_length=256,
    )

    latency_ms: float = Field(default=0.0, ge=0.0)

    tags: List[str] = Field(default_factory=list)


class HashSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: HashEvidenceType
    name: str

    value: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)

    description: str = Field(
        min_length=1,
        max_length=DEFAULT_MAX_REASON_LENGTH,
    )


# =============================================================================
# FINAL ASSESSMENT
# =============================================================================

class HashReputationAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_version: str = MODULE_VERSION

    status: HashReputationStatus

    identity: HashIdentity

    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    risk_level: HashRiskLevel

    severity: Any

    match_type: HashMatchType

    is_malicious: bool = False
    is_suspicious: bool = False
    known_good: bool = False

    reasons: List[str] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_REASONS,
    )

    signals: List[HashSignal] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_SIGNALS,
    )

    provider_observations: List[ProviderObservation] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_PROVIDER_OBSERVATIONS,
    )

    fuzzy_matches: List[FuzzyHashMatch] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_PROVIDER_OBSERVATIONS,
    )

    threat_intel_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    providers_queried: int = Field(default=0, ge=0)
    providers_successful: int = Field(default=0, ge=0)
    providers_positive: int = Field(default=0, ge=0)

    degraded_components: List[str] = Field(default_factory=list)

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    analysis_duration_ms: float = Field(default=0.0, ge=0.0)

    cache_hit: bool = False

    @property
    def cache_key(self) -> str:
        return self.identity.cache_key


# =============================================================================
# FILE-LEVEL CORRELATION
# =============================================================================

class FileReputationAssessment(BaseModel):
    """
    Correlated reputation of a file represented by multiple digests.

    This is intentionally separate from individual hash assessments.
    """

    model_config = ConfigDict(extra="forbid")

    engine_version: str = MODULE_VERSION

    status: HashReputationStatus

    file_metadata: HashFileMetadata
    digests: HashDigestSet

    assessments: List[HashReputationAssessment] = Field(
        default_factory=list,
        max_length=DEFAULT_MAX_PROVIDER_OBSERVATIONS,
    )

    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    risk_level: HashRiskLevel

    is_malicious: bool = False
    is_suspicious: bool = False
    known_good: bool = False

    reasons: List[str] = Field(default_factory=list)

    corroborating_algorithms: List[HashAlgorithm] = Field(
        default_factory=list
    )

    disagreement_detected: bool = False

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# =============================================================================
# ALLOWLIST
# =============================================================================

class HashAllowlist(Protocol):
    async def contains(
        self,
        identity: HashIdentity,
    ) -> bool:
        ...


class InMemoryHashAllowlist:
    """
    Explicit local allowlist.

    IMPORTANT:
        Allowlist is intentionally explicit. A provider's "not found"
        response never inserts anything here.
    """

    def __init__(
        self,
        entries: Optional[Iterable[Tuple[HashAlgorithm, str]]] = None,
    ) -> None:
        self._entries: set[Tuple[HashAlgorithm, str]] = set()

        if entries:
            for algorithm, digest in entries:
                identity = normalize_hash(
                    digest,
                    algorithm,
                )
                self._entries.add(
                    (
                        identity.algorithm,
                        identity.digest,
                    )
                )

    async def contains(
        self,
        identity: HashIdentity,
    ) -> bool:
        return (
            identity.algorithm,
            identity.digest,
        ) in self._entries

    def add(
        self,
        algorithm: HashAlgorithm,
        digest: str,
    ) -> None:
        identity = normalize_hash(
            digest,
            algorithm,
        )
        self._entries.add(
            (
                identity.algorithm,
                identity.digest,
            )
        )

    def remove(
        self,
        algorithm: HashAlgorithm,
        digest: str,
    ) -> None:
        identity = normalize_hash(
            digest,
            algorithm,
        )
        self._entries.discard(
            (
                identity.algorithm,
                identity.digest,
            )
        )


# =============================================================================
# FUZZY PROVIDER
# =============================================================================

class FuzzyHashProvider(Protocol):
    """
    Optional fuzzy-hash adapter.

    Implementations may use ssdeep, TLSH, or another approved system.

    The provider must return normalized FuzzyHashMatch objects or objects
    convertible into them.
    """

    async def lookup(
        self,
        identity: HashIdentity,
    ) -> Sequence[FuzzyHashMatch]:
        ...


# =============================================================================
# CACHE
# =============================================================================

@dataclass
class _CacheEntry:
    value: HashReputationAssessment
    expires_at: float


class HashReputationCache:
    """
    Bounded asynchronous TTL cache.

    Eviction uses oldest expiry first. This is intentionally simple and
    predictable. A distributed cache should live above this component.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        max_entries: int = DEFAULT_CACHE_MAX_ENTRIES,
    ) -> None:
        if ttl_seconds < 0:
            raise HashConfigurationError(
                "ttl_seconds must be >= 0"
            )

        if max_entries < 1:
            raise HashConfigurationError(
                "max_entries must be >= 1"
            )

        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

        self._entries: Dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

        self.hits = 0
        self.misses = 0
        self.evictions = 0

    async def get(
        self,
        key: str,
    ) -> Optional[HashReputationAssessment]:
        async with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                self.misses += 1
                return None

            if (
                self.ttl_seconds == 0
                or time.monotonic() >= entry.expires_at
            ):
                self._entries.pop(key, None)
                self.misses += 1
                return None

            self.hits += 1

            result = entry.value.model_copy(deep=True)
            result.cache_hit = True

            return result

    async def set(
        self,
        key: str,
        value: HashReputationAssessment,
    ) -> None:
        async with self._lock:
            if self.ttl_seconds == 0:
                return

            if len(self._entries) >= self.max_entries:
                oldest = min(
                    self._entries,
                    key=lambda k: self._entries[k].expires_at,
                )

                self._entries.pop(oldest, None)
                self.evictions += 1

            self._entries[key] = _CacheEntry(
                value=value.model_copy(deep=True),
                expires_at=(
                    time.monotonic()
                    + self.ttl_seconds
                ),
            )

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()

    async def size(self) -> int:
        async with self._lock:
            return len(self._entries)


# =============================================================================
# HASH NORMALIZATION
# =============================================================================

def _coerce_algorithm(
    algorithm: HashAlgorithm | str,
) -> HashAlgorithm:
    if isinstance(algorithm, HashAlgorithm):
        return algorithm

    try:
        return HashAlgorithm(str(algorithm).strip().lower())
    except ValueError as exc:
        raise UnsupportedHashAlgorithmError(
            f"Unsupported hash algorithm: {algorithm!r}"
        ) from exc


def normalize_hash(
    digest: str,
    algorithm: HashAlgorithm | str,
) -> HashIdentity:
    """
    Strict digest normalization.

    Hash equality is case-insensitive because hexadecimal representation is
    case-insensitive. Canonical form is lowercase.
    """

    if not isinstance(digest, str):
        raise InvalidHashError("Hash digest must be a string.")

    algorithm_enum = _coerce_algorithm(algorithm)

    digest = digest.strip().lower()

    spec = ALGORITHM_SPECS[algorithm_enum]

    if len(digest) != spec.digest_length:
        raise InvalidHashError(
            f"{algorithm_enum.value} requires "
            f"{spec.digest_length} hexadecimal characters."
        )

    if not re.fullmatch(
        rf"[0-9a-f]{{{spec.digest_length}}}",
        digest,
    ):
        raise InvalidHashError(
            "Hash digest contains non-hexadecimal characters."
        )

    return HashIdentity(
        algorithm=algorithm_enum,
        digest=digest,
    )


def infer_hash_algorithms(
    digest: str,
    *,
    preferred: Sequence[HashAlgorithm] = DEFAULT_ALGORITHM_PRIORITY,
) -> List[HashAlgorithm]:
    """
    Return all algorithms compatible with digest length.

    Important:
        A 64-character digest is ambiguous. It may be SHA-256,
        SHA3-256, SHA-512/256, BLAKE2S, etc.

    The caller may use preferred order for best-effort ingestion, but this
    function never pretends that length uniquely identifies the algorithm.
    """

    if not isinstance(digest, str):
        raise InvalidHashError("Digest must be a string.")

    normalized = digest.strip().lower()

    if not normalized:
        raise InvalidHashError("Digest cannot be empty.")

    if not re.fullmatch(r"[0-9a-f]+", normalized):
        raise InvalidHashError(
            "Digest must contain hexadecimal characters only."
        )

    result = [
        algorithm
        for algorithm in preferred
        if len(normalized)
        == ALGORITHM_SPECS[algorithm].digest_length
    ]

    if not result:
        raise InvalidHashError(
            f"No supported algorithm matches digest length {len(normalized)}."
        )

    return result


# =============================================================================
# STREAMING FILE HASHER
# =============================================================================

class StreamingFileHasher:
    """
    Streaming multi-algorithm file hasher.

    The implementation never loads the entire file into memory.
    """

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
        executor: Optional[Executor] = None,
    ) -> None:
        if chunk_size < 4096:
            raise HashConfigurationError(
                "chunk_size must be >= 4096"
            )

        self.chunk_size = chunk_size
        self.executor = executor

    def _hash_sync(
        self,
        path: Path,
        algorithms: Sequence[HashAlgorithm],
    ) -> Tuple[HashDigestSet, HashFileMetadata]:
        try:
            stat_before = path.stat()
        except OSError as exc:
            raise FileHashingError(
                f"Unable to stat file: {type(exc).__name__}"
            ) from exc

        if not path.is_file():
            raise FileHashingError(
                "Input path is not a regular file."
            )

        if stat_before.st_size > DEFAULT_MAX_FILE_SIZE:
            raise FileHashingError(
                "File exceeds configured maximum size."
            )

        hashers: Dict[HashAlgorithm, Any] = {}

        for algorithm in algorithms:
            spec = ALGORITHM_SPECS[algorithm]

            try:
                hashers[algorithm] = hashlib.new(
                    spec.hashlib_name,
                    usedforsecurity=not spec.legacy,
                )
            except TypeError:
                # Older Python/OpenSSL builds may not expose usedforsecurity.
                hashers[algorithm] = hashlib.new(
                    spec.hashlib_name
                )

        started = time.perf_counter()

        try:
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(self.chunk_size)

                    if not chunk:
                        break

                    for hasher in hashers.values():
                        hasher.update(chunk)

        except OSError as exc:
            raise FileHashingError(
                f"File hashing failed: {type(exc).__name__}"
            ) from exc

        stat_after = path.stat()

        changed = (
            stat_before.st_size != stat_after.st_size
            or stat_before.st_mtime_ns != stat_after.st_mtime_ns
            or (
                getattr(stat_before, "st_ino", None)
                != getattr(stat_after, "st_ino", None)
            )
        )

        digest_values = {
            algorithm.value: hasher.hexdigest()
            for algorithm, hasher in hashers.items()
        }

        digest_set = HashDigestSet(
            **digest_values,
        )

        metadata = HashFileMetadata(
            path=str(path),
            filename=path.name,
            size_bytes=stat_after.st_size,
            mtime_ns=stat_after.st_mtime_ns,
            inode=getattr(stat_after, "st_ino", None),
            content_changed_during_hash=changed,
            hash_duration_ms=(
                time.perf_counter() - started
            ) * 1000.0,
        )

        return digest_set, metadata

    async def hash_file(
        self,
        path: str | Path,
        algorithms: Sequence[HashAlgorithm],
    ) -> Tuple[HashDigestSet, HashFileMetadata]:
        if not algorithms:
            raise HashConfigurationError(
                "At least one hash algorithm is required."
            )

        unique_algorithms = list(dict.fromkeys(algorithms))

        path_obj = Path(path)

        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            self.executor,
            self._hash_sync,
            path_obj,
            unique_algorithms,
        )


# =============================================================================
# ENGINE
# =============================================================================

class HashReputationEngine:
    """
    Production hash reputation engine.

    Threat intelligence and fuzzy providers are deliberately independent.

    TI concurrency and fuzzy concurrency are separate so that a slow fuzzy
    provider cannot starve the primary reputation path.
    """

    def __init__(
        self,
        threat_intel: Optional["ThreatIntelAggregator"] = None,
        fuzzy_provider: Optional[FuzzyHashProvider] = None,
        allowlist: Optional[HashAllowlist] = None,
        cache: Optional[HashReputationCache] = None,
        *,
        provider_timeout_seconds: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
        fuzzy_timeout_seconds: float = DEFAULT_FUZZY_TIMEOUT_SECONDS,
        ti_concurrency: int = DEFAULT_TI_CONCURRENCY,
        fuzzy_concurrency: int = DEFAULT_FUZZY_CONCURRENCY,
        file_hash_concurrency: int = DEFAULT_FILE_HASH_CONCURRENCY,
        enable_fuzzy: bool = True,
        allow_ambiguous_inference: bool = False,
        cache_config_version: str = "1",
    ) -> None:
        if provider_timeout_seconds <= 0:
            raise HashConfigurationError(
                "provider_timeout_seconds must be > 0"
            )

        if fuzzy_timeout_seconds <= 0:
            raise HashConfigurationError(
                "fuzzy_timeout_seconds must be > 0"
            )

        if ti_concurrency < 1:
            raise HashConfigurationError(
                "ti_concurrency must be >= 1"
            )

        if fuzzy_concurrency < 1:
            raise HashConfigurationError(
                "fuzzy_concurrency must be >= 1"
            )

        if file_hash_concurrency < 1:
            raise HashConfigurationError(
                "file_hash_concurrency must be >= 1"
            )

        self.threat_intel = threat_intel
        self.fuzzy_provider = fuzzy_provider
        self.allowlist = allowlist
        self.cache = cache

        self.provider_timeout_seconds = provider_timeout_seconds
        self.fuzzy_timeout_seconds = fuzzy_timeout_seconds

        self.enable_fuzzy = (
            enable_fuzzy
            and fuzzy_provider is not None
        )

        self.allow_ambiguous_inference = (
            allow_ambiguous_inference
        )

        self.cache_config_version = str(
            cache_config_version
        )

        self._ti_semaphore = asyncio.Semaphore(
            ti_concurrency
        )

        self._fuzzy_semaphore = asyncio.Semaphore(
            fuzzy_concurrency
        )

        self._file_semaphore = asyncio.Semaphore(
            file_hash_concurrency
        )

        # Normal request coalescing.
        self._inflight: Dict[
            str,
            asyncio.Task[HashReputationAssessment],
        ] = {}

        self._inflight_lock = asyncio.Lock()

        self._metrics_lock = asyncio.Lock()

        self._metrics = Counter()

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def analyze(
        self,
        digest: str,
        algorithm: HashAlgorithm | str | None = None,
        *,
        force_refresh: bool = False,
        include_fuzzy: Optional[bool] = None,
    ) -> HashReputationAssessment:
        """
        Analyze one hash.

        If algorithm is omitted:
            - strict mode raises InvalidHashError for ambiguous lengths;
            - allow_ambiguous_inference mode uses the first configured
              preferred algorithm.

        This prevents silent misclassification of 64-character digests.
        """

        identity = self._resolve_identity(
            digest,
            algorithm,
        )

        if self.cache and not force_refresh:
            cached = await self.cache.get(
                self._cache_key(identity)
            )

            if cached is not None:
                await self._metric("cache_hit")
                return cached

            await self._metric("cache_miss")

        # Force refresh deliberately bypasses coalescing.
        #
        # This prevents a forced request from accidentally returning an
        # older in-flight result.
        if force_refresh:
            return await self._analyze_uncached(
                identity,
                force_refresh=True,
                include_fuzzy=include_fuzzy,
            )

        key = self._cache_key(identity)

        async with self._inflight_lock:
            existing = self._inflight.get(key)

            if existing is not None:
                await self._metric("coalesced")
                task = existing

            else:
                task = asyncio.create_task(
                    self._analyze_uncached(
                        identity,
                        force_refresh=False,
                        include_fuzzy=include_fuzzy,
                    )
                )

                self._inflight[key] = task

        try:
            return await asyncio.shield(task)

        finally:
            if task.done():
                async with self._inflight_lock:
                    if self._inflight.get(key) is task:
                        self._inflight.pop(key, None)

    async def analyze_file(
        self,
        path: str | Path,
        *,
        algorithms: Optional[
            Sequence[HashAlgorithm]
        ] = None,
        force_refresh: bool = False,
        include_fuzzy: Optional[bool] = None,
    ) -> FileReputationAssessment:
        """
        Hash a file and return a correlated file-level assessment.

        Unlike the older implementation, this does not merely return a list
        of unrelated hash assessments.
        """

        algorithms = list(
            algorithms
            or DEFAULT_ALGORITHM_PRIORITY[:]
        )

        async with self._file_semaphore:
            digest_set, metadata = (
                await StreamingFileHasher().hash_file(
                    path,
                    algorithms,
                )
            )

        if metadata.content_changed_during_hash:
            await self._metric("file_changed_during_hash")

        pairs = self._digest_pairs(
            digest_set,
            algorithms,
        )

        assessments: List[HashReputationAssessment] = []

        for algorithm, digest in pairs:
            result = await self.analyze(
                digest,
                algorithm,
                force_refresh=force_refresh,
                include_fuzzy=include_fuzzy,
            )

            assessments.append(result)

        return self._correlate_file(
            metadata,
            digest_set,
            assessments,
        )

    # =========================================================================
    # IDENTITY
    # =========================================================================

    def _resolve_identity(
        self,
        digest: str,
        algorithm: HashAlgorithm | str | None,
    ) -> HashIdentity:
        if algorithm is not None:
            return normalize_hash(
                digest,
                algorithm,
            )

        candidates = infer_hash_algorithms(digest)

        if len(candidates) > 1:
            if not self.allow_ambiguous_inference:
                raise InvalidHashError(
                    "Hash algorithm is required because the digest length "
                    "matches multiple algorithms."
                )

            return normalize_hash(
                digest,
                candidates[0],
            )

        return normalize_hash(
            digest,
            candidates[0],
        )

    def _cache_key(
        self,
        identity: HashIdentity,
    ) -> str:
        config_digest = hashlib.sha256(
            self.cache_config_version.encode("utf-8")
        ).hexdigest()[:16]

        return (
            f"{identity.cache_key}:"
            f"cfg-{config_digest}"
        )

    # =========================================================================
    # UNCACHED ANALYSIS
    # =========================================================================

    async def _analyze_uncached(
        self,
        identity: HashIdentity,
        *,
        force_refresh: bool,
        include_fuzzy: Optional[bool],
    ) -> HashReputationAssessment:
        started = time.perf_counter()

        degraded: set[str] = set()

        signals: List[HashSignal] = []

        provider_observations: List[
            ProviderObservation
        ] = []

        fuzzy_matches: List[FuzzyHashMatch] = []

        # ---------------------------------------------------------------------
        # Local allowlist
        # ---------------------------------------------------------------------

        allowlisted = False

        if self.allowlist is not None:
            try:
                allowlisted = bool(
                    await self.allowlist.contains(identity)
                )
            except Exception:
                degraded.add("allowlist")

        if allowlisted:
            signals.append(
                HashSignal(
                    type=HashEvidenceType.ALLOWLIST,
                    name="local_allowlist_match",
                    value=0.0,
                    weight=1.0,
                    description=(
                        "Hash matched an explicit local allowlist entry."
                    ),
                )
            )

        # ---------------------------------------------------------------------
        # Algorithm signal
        # ---------------------------------------------------------------------

        spec = identity.spec

        if spec.legacy:
            signals.append(
                HashSignal(
                    type=HashEvidenceType.ALGORITHM,
                    name="legacy_hash_algorithm",
                    value=0.0,
                    weight=0.0,
                    description=(
                        f"{identity.algorithm.value} is retained for IOC "
                        "interoperability but is a legacy identity algorithm."
                    ),
                )
            )

        # ---------------------------------------------------------------------
        # Threat intelligence
        # ---------------------------------------------------------------------

        threat_report: Any = None

        if self.threat_intel is not None:
            try:
                async with self._ti_semaphore:
                    threat_report = await asyncio.wait_for(
                        self.threat_intel.analyze_ioc(
                            identity.digest,
                            IOCType.HASH,
                            force_refresh=force_refresh,
                        ),
                        timeout=self.provider_timeout_seconds,
                    )

            except asyncio.TimeoutError:
                degraded.add("threat_intel_timeout")
                await self._metric("ti_timeout")

            except Exception as exc:
                degraded.add("threat_intel")
                await self._metric("ti_error")

                # Never leak provider exception content.
                _ = exc

        # ---------------------------------------------------------------------
        # Normalize TI
        # ---------------------------------------------------------------------

        (
            threat_score,
            providers_queried,
            providers_successful,
            providers_positive,
            normalized_observations,
            provider_tags,
        ) = self._normalize_threat_report(
            threat_report
        )

        provider_observations.extend(
            normalized_observations
        )

        if threat_report is not None and not normalized_observations:
            degraded.add("threat_intel_normalization")

        # ---------------------------------------------------------------------
        # Fuzzy reputation
        # ---------------------------------------------------------------------

        fuzzy_enabled = (
            self.enable_fuzzy
            if include_fuzzy is None
            else bool(include_fuzzy)
        )

        if fuzzy_enabled and self.fuzzy_provider is not None:
            try:
                async with self._fuzzy_semaphore:
                    raw_fuzzy = await asyncio.wait_for(
                        self.fuzzy_provider.lookup(
                            identity
                        ),
                        timeout=self.fuzzy_timeout_seconds,
                    )

                fuzzy_matches = self._normalize_fuzzy_matches(
                    raw_fuzzy
                )

            except asyncio.TimeoutError:
                degraded.add("fuzzy_timeout")
                await self._metric("fuzzy_timeout")

            except Exception:
                degraded.add("fuzzy")
                await self._metric("fuzzy_error")

        # ---------------------------------------------------------------------
        # Signals
        # ---------------------------------------------------------------------

        if threat_score is not None:
            signals.append(
                HashSignal(
                    type=HashEvidenceType.THREAT_INTEL,
                    name="aggregated_threat_intelligence",
                    value=threat_score,
                    weight=1.0,
                    description=(
                        "Normalized multi-provider threat-intelligence "
                        "reputation score."
                    ),
                )
            )

        best_fuzzy = self._best_fuzzy_match(
            fuzzy_matches
        )

        if best_fuzzy is not None:
            fuzzy_risk = (
                best_fuzzy.similarity
                if best_fuzzy.malicious
                else 0.0
            )

            signals.append(
                HashSignal(
                    type=HashEvidenceType.FUZZY,
                    name="fuzzy_similarity",
                    value=fuzzy_risk,
                    weight=0.35,
                    description=(
                        "Fuzzy-hash similarity evidence. Similarity alone "
                        "does not prove maliciousness."
                    ),
                )
            )

        # ---------------------------------------------------------------------
        # Risk
        # ---------------------------------------------------------------------

        risk_score = self._calculate_risk(
            threat_score=threat_score,
            providers_positive=providers_positive,
            fuzzy_matches=fuzzy_matches,
            allowlisted=allowlisted,
        )

        # ---------------------------------------------------------------------
        # Confidence
        # ---------------------------------------------------------------------

        confidence = self._calculate_confidence(
            identity=identity,
            threat_report=threat_report,
            providers_queried=providers_queried,
            providers_successful=providers_successful,
            providers_positive=providers_positive,
            fuzzy_matches=fuzzy_matches,
            allowlisted=allowlisted,
        )

        # ---------------------------------------------------------------------
        # Critical semantic rule:
        #
        # A successful TI lookup with zero positives does NOT automatically
        # prove the hash is clean.
        # ---------------------------------------------------------------------

        known_good = False

        # Explicit local allowlist is the only deterministic local
        # known-good mechanism.
        if allowlisted:
            known_good = True

        # Providers must explicitly communicate benign/known-good evidence.
        if (
            threat_report is not None
            and providers_successful > 0
            and self._explicit_known_good(
                threat_report
            )
        ):
            known_good = True

        # ---------------------------------------------------------------------
        # Match type
        # ---------------------------------------------------------------------

        if allowlisted:
            match_type = HashMatchType.ALLOWLIST

        elif providers_positive > 0:
            match_type = HashMatchType.EXACT

        elif any(
            match.matched
            for match in fuzzy_matches
        ):
            match_type = HashMatchType.FUZZY

        elif (
            threat_report is not None
            and providers_successful > 0
        ):
            match_type = HashMatchType.NONE

        else:
            match_type = HashMatchType.UNKNOWN

        # ---------------------------------------------------------------------
        # Unknown semantic protection
        # ---------------------------------------------------------------------

        if (
            match_type == HashMatchType.UNKNOWN
            and not allowlisted
        ):
            # No evidence means unknown, regardless of heuristic score.
            risk_score = min(
                risk_score,
                0.39,
            )

        # A provider saying "not observed" is still not equivalent to clean.
        if (
            match_type == HashMatchType.NONE
            and not known_good
        ):
            risk_score = min(
                risk_score,
                0.39,
            )

        risk_level = self._risk_level(
            risk_score,
            confidence,
            known_good=known_good,
            match_type=match_type,
        )

        severity = self._severity(
            risk_level
        )

        is_malicious = (
            not known_good
            and risk_score >= 0.75
            and confidence >= 0.50
            and (
                match_type in {
                    HashMatchType.EXACT,
                    HashMatchType.FUZZY,
                }
            )
        )

        is_suspicious = (
            not is_malicious
            and not known_good
            and risk_score >= 0.40
        )

        # ---------------------------------------------------------------------
        # Reasons
        # ---------------------------------------------------------------------

        reasons = self._build_reasons(
            identity=identity,
            signals=signals,
            match_type=match_type,
            known_good=known_good,
            confidence=confidence,
            risk_score=risk_score,
            provider_tags=provider_tags,
        )

        status = (
            HashReputationStatus.DEGRADED
            if degraded
            else HashReputationStatus.COMPLETE
        )

        duration_ms = (
            time.perf_counter() - started
        ) * 1000.0

        assessment = HashReputationAssessment(
            status=status,
            identity=identity,
            risk_score=risk_score,
            confidence=confidence,
            risk_level=risk_level,
            severity=severity,
            match_type=match_type,
            is_malicious=is_malicious,
            is_suspicious=is_suspicious,
            known_good=known_good,
            reasons=reasons,
            signals=signals[:DEFAULT_MAX_SIGNALS],
            provider_observations=provider_observations[
                :DEFAULT_MAX_PROVIDER_OBSERVATIONS
            ],
            fuzzy_matches=fuzzy_matches[
                :DEFAULT_MAX_PROVIDER_OBSERVATIONS
            ],
            threat_intel_score=threat_score,
            providers_queried=providers_queried,
            providers_successful=providers_successful,
            providers_positive=providers_positive,
            degraded_components=sorted(degraded),
            analysis_duration_ms=duration_ms,
        )

        if self.cache:
            await self.cache.set(
                self._cache_key(identity),
                assessment,
            )

        await self._metric("analyses")

        if is_malicious:
            await self._metric("malicious")

        if status == HashReputationStatus.DEGRADED:
            await self._metric("degraded")

        return assessment

    # =========================================================================
    # THREAT INTEL NORMALIZATION
    # =========================================================================

    @classmethod
    def _normalize_threat_report(
        cls,
        report: Any,
    ) -> Tuple[
        Optional[float],
        int,
        int,
        int,
        List[ProviderObservation],
        List[str],
    ]:
        if report is None:
            return (
                None,
                0,
                0,
                0,
                [],
                [],
            )

        threat_score = cls._safe_float(
            getattr(
                report,
                "aggregated_risk_score",
                None,
            )
        )

        queried = cls._safe_nonnegative_int(
            getattr(
                report,
                "total_providers_queried",
                0,
            )
        )

        successful = cls._safe_nonnegative_int(
            getattr(
                report,
                "successful_provider_responses",
                0,
            )
        )

        positive = cls._safe_nonnegative_int(
            getattr(
                report,
                "positive_provider_matches",
                getattr(
                    report,
                    "providers_positive",
                    0,
                ),
            )
        )

        observations: List[ProviderObservation] = []

        raw_observations = getattr(
            report,
            "provider_observations",
            getattr(
                report,
                "observations",
                [],
            ),
        )

        if isinstance(raw_observations, Sequence):
            for raw in list(
                raw_observations
            )[:DEFAULT_MAX_PROVIDER_OBSERVATIONS]:

                normalized = cls._normalize_provider_observation(
                    raw
                )

                if normalized is not None:
                    observations.append(normalized)

        tags: List[str] = []

        raw_tags = getattr(
            report,
            "combined_raw_tags",
            [],
        )

        if isinstance(raw_tags, Sequence):
            for tag in list(raw_tags)[
                :DEFAULT_MAX_PROVIDER_TAGS
            ]:
                if not isinstance(tag, str):
                    continue

                clean = cls._sanitize_text(
                    tag,
                    max_length=128,
                )

                if clean:
                    tags.append(clean)

        return (
            threat_score,
            queried,
            successful,
            positive,
            observations,
            tags,
        )

    @classmethod
    def _normalize_provider_observation(
        cls,
        raw: Any,
    ) -> Optional[ProviderObservation]:
        if isinstance(raw, ProviderObservation):
            return raw

        provider = cls._sanitize_text(
            getattr(
                raw,
                "provider",
                getattr(
                    raw,
                    "name",
                    "unknown",
                ),
            ),
            max_length=DEFAULT_MAX_PROVIDER_NAME_LENGTH,
        )

        if not provider:
            provider = "unknown"

        return ProviderObservation(
            provider=provider,
            matched=bool(
                getattr(
                    raw,
                    "matched",
                    getattr(
                        raw,
                        "positive",
                        False,
                    ),
                )
            ),
            malicious=bool(
                getattr(
                    raw,
                    "malicious",
                    False,
                )
            ),
            confidence=(
                cls._safe_float(
                    getattr(
                        raw,
                        "confidence",
                        0.0,
                    )
                )
                or 0.0
            ),
            risk_score=cls._safe_float(
                getattr(
                    raw,
                    "risk_score",
                    getattr(
                        raw,
                        "score",
                        None,
                    ),
                )
            ),
            severity=cls._sanitize_text(
                getattr(
                    raw,
                    "severity",
                    None,
                ),
                max_length=64,
            ),
            tags=cls._bounded_strings(
                getattr(
                    raw,
                    "tags",
                    [],
                ),
                DEFAULT_MAX_PROVIDER_TAGS,
                128,
            ),
            categories=cls._bounded_strings(
                getattr(
                    raw,
                    "categories",
                    [],
                ),
                DEFAULT_MAX_PROVIDER_TAGS,
                128,
            ),
            latency_ms=max(
                0.0,
                cls._safe_float(
                    getattr(
                        raw,
                        "latency_ms",
                        0.0,
                    )
                )
                or 0.0,
            ),
        )

    # =========================================================================
    # FUZZY
    # =========================================================================

    @classmethod
    def _normalize_fuzzy_matches(
        cls,
        values: Sequence[Any],
    ) -> List[FuzzyHashMatch]:
        result: List[FuzzyHashMatch] = []

        if not isinstance(values, Sequence):
            return result

        for value in list(values)[
            :DEFAULT_MAX_PROVIDER_OBSERVATIONS
        ]:
            if isinstance(value, FuzzyHashMatch):
                result.append(value)
                continue

            provider = cls._sanitize_text(
                getattr(
                    value,
                    "provider",
                    "unknown",
                ),
                max_length=128,
            ) or "unknown"

            algorithm = cls._sanitize_text(
                getattr(
                    value,
                    "algorithm",
                    "unknown",
                ),
                max_length=32,
            ) or "unknown"

            similarity = (
                cls._safe_float(
                    getattr(
                        value,
                        "similarity",
                        0.0,
                    )
                )
                or 0.0
            )

            result.append(
                FuzzyHashMatch(
                    provider=provider,
                    algorithm=algorithm,
                    similarity=similarity,
                    matched=bool(
                        getattr(
                            value,
                            "matched",
                            similarity >= 0.90,
                        )
                    ),
                    malicious=bool(
                        getattr(
                            value,
                            "malicious",
                            False,
                        )
                    ),
                    target_identifier=cls._sanitize_text(
                        getattr(
                            value,
                            "target_identifier",
                            None,
                        ),
                        max_length=256,
                    ),
                    latency_ms=max(
                        0.0,
                        cls._safe_float(
                            getattr(
                                value,
                                "latency_ms",
                                0.0,
                            )
                        )
                        or 0.0,
                    ),
                    tags=cls._bounded_strings(
                        getattr(
                            value,
                            "tags",
                            [],
                        ),
                        DEFAULT_MAX_PROVIDER_TAGS,
                        128,
                    ),
                )
            )

        return result

    @staticmethod
    def _best_fuzzy_match(
        matches: Sequence[FuzzyHashMatch],
    ) -> Optional[FuzzyHashMatch]:
        malicious = [
            match
            for match in matches
            if match.matched and match.malicious
        ]

        if not malicious:
            return None

        return max(
            malicious,
            key=lambda item: item.similarity,
        )

    # =========================================================================
    # RISK ENGINE
    # =========================================================================

    @staticmethod
    def _calculate_risk(
        *,
        threat_score: Optional[float],
        providers_positive: int,
        fuzzy_matches: Sequence[FuzzyHashMatch],
        allowlisted: bool,
    ) -> float:
        if allowlisted:
            return 0.0

        # Exact TI evidence remains dominant.
        if threat_score is not None:
            ti = threat_score

            # Exact provider corroboration gives a modest additional boost,
            # but never exceeds the bounded maximum.
            corroboration = min(
                0.15,
                0.05 * max(
                    0,
                    providers_positive - 1,
                ),
            )

            best_fuzzy = HashReputationEngine._best_fuzzy_match(
                fuzzy_matches
            )

            fuzzy_component = (
                best_fuzzy.similarity * 0.15
                if best_fuzzy is not None
                else 0.0
            )

            return max(
                0.0,
                min(
                    1.0,
                    ti * 0.75
                    + corroboration
                    + fuzzy_component,
                ),
            )

        # No TI:
        #
        # Fuzzy evidence can still be meaningful, but fuzzy similarity alone
        # must not be treated as an exact malicious-hash match.
        best_fuzzy = HashReputationEngine._best_fuzzy_match(
            fuzzy_matches
        )

        if best_fuzzy is not None:
            return min(
                0.75,
                0.50
                + 0.25 * best_fuzzy.similarity,
            )

        return 0.0

    # =========================================================================
    # CONFIDENCE
    # =========================================================================

    @staticmethod
    def _calculate_confidence(
        *,
        identity: HashIdentity,
        threat_report: Any,
        providers_queried: int,
        providers_successful: int,
        providers_positive: int,
        fuzzy_matches: Sequence[FuzzyHashMatch],
        allowlisted: bool,
    ) -> float:
        if allowlisted:
            return 0.99

        components: List[float] = []

        if threat_report is not None:
            if providers_queried > 0:
                coverage = (
                    providers_successful
                    / max(
                        1,
                        providers_queried,
                    )
                )

                provider_confidence = (
                    0.45
                    + 0.45 * min(
                        1.0,
                        coverage,
                    )
                )

                if providers_positive >= 2:
                    provider_confidence += 0.05

                components.append(
                    min(
                        0.95,
                        provider_confidence,
                    )
                )

            else:
                components.append(0.25)

        # Algorithm identity quality affects confidence, not risk.
        components.append(
            identity.spec.security_confidence
        )

        if fuzzy_matches:
            best = max(
                fuzzy_matches,
                key=lambda x: x.similarity,
            )

            components.append(
                0.35
                + 0.50 * best.similarity
            )

        if not components:
            return 0.10

        return max(
            0.0,
            min(
                1.0,
                sum(components)
                / len(components),
            ),
        )

    # =========================================================================
    # EXPLICIT KNOWN-GOOD
    # =========================================================================

    @staticmethod
    def _explicit_known_good(
        report: Any,
    ) -> bool:
        """
        Only accept an explicit provider assertion.

        We intentionally do NOT infer known-good from:
            score == 0
            no positive matches
            provider responded successfully
        """

        if report is None:
            return False

        for field in (
            "known_good",
            "explicitly_benign",
            "benign",
            "is_clean",
        ):
            value = getattr(
                report,
                field,
                None,
            )

            if value is True:
                return True

        observations = getattr(
            report,
            "provider_observations",
            [],
        )

        if isinstance(observations, Sequence):
            for observation in observations:
                if bool(
                    getattr(
                        observation,
                        "known_good",
                        False,
                    )
                ):
                    return True

                if bool(
                    getattr(
                        observation,
                        "explicitly_benign",
                        False,
                    )
                ):
                    return True

        return False

    # =========================================================================
    # FILE CORRELATION
    # =========================================================================

    @staticmethod
    def _digest_pairs(
        digest_set: HashDigestSet,
        algorithms: Sequence[HashAlgorithm],
    ) -> List[Tuple[HashAlgorithm, str]]:
        pairs: List[
            Tuple[HashAlgorithm, str]
        ] = []

        for algorithm in algorithms:
            digest = getattr(
                digest_set,
                algorithm.value,
                None,
            )

            if digest:
                pairs.append(
                    (
                        algorithm,
                        digest,
                    )
                )

        return pairs

    @staticmethod
    def _correlate_file(
        metadata: HashFileMetadata,
        digest_set: HashDigestSet,
        assessments: Sequence[
            HashReputationAssessment
        ],
    ) -> FileReputationAssessment:
        malicious = [
            result
            for result in assessments
            if result.is_malicious
        ]

        suspicious = [
            result
            for result in assessments
            if result.is_suspicious
        ]

        allowlisted = [
            result
            for result in assessments
            if result.known_good
        ]

        exact_positive = [
            result
            for result in assessments
            if result.match_type
            == HashMatchType.EXACT
        ]

        unknown_count = sum(
            result.match_type
            == HashMatchType.UNKNOWN
            for result in assessments
        )

        if malicious:
            file_risk = max(
                result.risk_score
                for result in malicious
            )

            file_confidence = max(
                result.confidence
                for result in malicious
            )

            level = HashRiskLevel.CRITICAL \
                if file_risk >= 0.90 \
                else HashRiskLevel.HIGH \
                if file_risk >= 0.75 \
                else HashRiskLevel.MEDIUM

            is_malicious = True
            is_suspicious = False

        elif suspicious:
            file_risk = max(
                result.risk_score
                for result in suspicious
            )

            file_confidence = max(
                result.confidence
                for result in suspicious
            )

            level = (
                HashRiskLevel.MEDIUM
                if file_risk >= 0.40
                else HashRiskLevel.LOW
            )

            is_malicious = False
            is_suspicious = True

        elif allowlisted and unknown_count == 0:
            file_risk = 0.0
            file_confidence = max(
                result.confidence
                for result in allowlisted
            )

            level = HashRiskLevel.CLEAN

            is_malicious = False
            is_suspicious = False

        elif unknown_count == len(assessments):
            file_risk = 0.0
            file_confidence = (
                sum(
                    result.confidence
                    for result in assessments
                )
                / max(
                    1,
                    len(assessments),
                )
            )

            level = HashRiskLevel.UNKNOWN

            is_malicious = False
            is_suspicious = False

        else:
            file_risk = max(
                (
                    result.risk_score
                    for result in assessments
                ),
                default=0.0,
            )

            file_confidence = (
                sum(
                    result.confidence
                    for result in assessments
                )
                / max(
                    1,
                    len(assessments),
                )
            )

            level = (
                HashRiskLevel.LOW
                if file_risk >= 0.15
                else HashRiskLevel.UNKNOWN
            )

            is_malicious = False
            is_suspicious = False

        reasons: List[str] = []

        if exact_positive:
            reasons.append(
                "one_or_more_exact_hash_matches"
            )

        if len(exact_positive) >= 2:
            reasons.append(
                "cross_algorithm_corrobation"
            )

        if malicious:
            reasons.append(
                "malicious_hash_evidence"
            )

        if unknown_count:
            reasons.append(
                "some_hashes_unresolved"
            )

        if metadata.content_changed_during_hash:
            reasons.append(
                "file_changed_during_hashing"
            )

        disagreement = (
            any(
                result.is_malicious
                for result in assessments
            )
            and any(
                result.known_good
                for result in assessments
            )
        )

        if disagreement:
            reasons.append(
                "cross_hash_reputation_disagreement"
            )

        return FileReputationAssessment(
            status=(
                HashReputationStatus.DEGRADED
                if metadata.content_changed_during_hash
                or any(
                    result.status
                    == HashReputationStatus.DEGRADED
                    for result in assessments
                )
                else HashReputationStatus.COMPLETE
            ),
            file_metadata=metadata,
            digests=digest_set,
            assessments=list(assessments),
            risk_score=max(
                0.0,
                min(
                    1.0,
                    file_risk,
                ),
            ),
            confidence=max(
                0.0,
                min(
                    1.0,
                    file_confidence,
                ),
            ),
            risk_level=level,
            is_malicious=is_malicious,
            is_suspicious=is_suspicious,
            known_good=bool(
                allowlisted
                and not malicious
            ),
            reasons=list(
                dict.fromkeys(reasons)
            ),
            corroborating_algorithms=[
                result.identity.algorithm
                for result in exact_positive
            ],
            disagreement_detected=disagreement,
        )

    # =========================================================================
    # RISK LEVEL / SEVERITY
    # =========================================================================

    @staticmethod
    def _risk_level(
        risk_score: float,
        confidence: float,
        *,
        known_good: bool,
        match_type: HashMatchType,
    ) -> HashRiskLevel:
        if known_good:
            return HashRiskLevel.CLEAN

        if (
            match_type
            in {
                HashMatchType.UNKNOWN,
                HashMatchType.NONE,
            }
            and risk_score < 0.40
        ):
            return HashRiskLevel.UNKNOWN

        if confidence < 0.20:
            return HashRiskLevel.UNKNOWN

        if risk_score >= 0.90:
            return HashRiskLevel.CRITICAL

        if risk_score >= 0.75:
            return HashRiskLevel.HIGH

        if risk_score >= 0.40:
            return HashRiskLevel.MEDIUM

        if risk_score >= 0.15:
            return HashRiskLevel.LOW

        return HashRiskLevel.UNKNOWN

    @staticmethod
    def _severity(
        level: HashRiskLevel,
    ) -> Any:
        """
        Keep compatibility with the project's ThreatSeverity enum.

        The fallback also makes isolated unit testing possible.
        """

        try:
            mapping = {
                HashRiskLevel.CRITICAL: ThreatSeverity.CRITICAL,
                HashRiskLevel.HIGH: ThreatSeverity.HIGH,
                HashRiskLevel.MEDIUM: ThreatSeverity.MEDIUM,
                HashRiskLevel.LOW: ThreatSeverity.LOW,
                HashRiskLevel.CLEAN: ThreatSeverity.CLEAN,
                HashRiskLevel.UNKNOWN: ThreatSeverity.UNKNOWN,
            }

            return mapping[level]

        except Exception:
            return level.value

    # =========================================================================
    # REASONS
    # =========================================================================

    @classmethod
    def _build_reasons(
        cls,
        *,
        identity: HashIdentity,
        signals: Sequence[HashSignal],
        match_type: HashMatchType,
        known_good: bool,
        confidence: float,
        risk_score: float,
        provider_tags: Sequence[str],
    ) -> List[str]:
        reasons: List[str] = []

        if known_good:
            reasons.append(
                "explicit_known_good_evidence"
            )

        if match_type == HashMatchType.EXACT:
            reasons.append(
                "exact_threat_intelligence_match"
            )

        elif match_type == HashMatchType.FUZZY:
            reasons.append(
                "fuzzy_hash_similarity_match"
            )

        elif match_type == HashMatchType.NONE:
            reasons.append(
                "no_positive_provider_match"
            )

        elif match_type == HashMatchType.UNKNOWN:
            reasons.append(
                "insufficient_reputation_evidence"
            )

        if identity.spec.legacy:
            reasons.append(
                "legacy_hash_algorithm"
            )

        for signal in signals:
            if signal.value > 0:
                reasons.append(
                    signal.name
                )

        # Provider tags are retained only as bounded diagnostic reasons.
        for tag in provider_tags[
            :DEFAULT_MAX_PROVIDER_TAGS
        ]:
            clean = cls._sanitize_text(
                tag,
                max_length=128,
            )

            if clean:
                reasons.append(
                    f"intel_category:{clean}"
                )

        if confidence < 0.35:
            reasons.append(
                "low_reputation_confidence"
            )

        if risk_score >= 0.90:
            reasons.append(
                "critical_risk_score"
            )

        elif risk_score >= 0.75:
            reasons.append(
                "high_risk_score"
            )

        elif risk_score >= 0.40:
            reasons.append(
                "medium_risk_score"
            )

        return list(
            dict.fromkeys(
                reasons
            )
        )[:DEFAULT_MAX_REASONS]

    # =========================================================================
    # METRICS
    # =========================================================================

    async def _metric(
        self,
        name: str,
        value: int = 1,
    ) -> None:
        async with self._metrics_lock:
            self._metrics[name] += value

    async def metrics_snapshot(self) -> Dict[str, int]:
        async with self._metrics_lock:
            return dict(self._metrics)

    # =========================================================================
    # SAFE HELPERS
    # =========================================================================

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
            min(
                1.0,
                number,
            ),
        )

    @staticmethod
    def _safe_nonnegative_int(
        value: Any,
    ) -> int:
        try:
            number = int(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0

        return max(
            0,
            number,
        )

    @staticmethod
    def _sanitize_text(
        value: Any,
        *,
        max_length: int,
    ) -> Optional[str]:
        if not isinstance(value, str):
            return None

        value = value.strip()

        if not value:
            return None

        value = re.sub(
            r"[\x00-\x1f\x7f]",
            " ",
            value,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value[:max_length]

    @classmethod
    def _bounded_strings(
        cls,
        values: Any,
        limit: int,
        max_length: int,
    ) -> List[str]:
        if not isinstance(values, Sequence):
            return []

        result: List[str] = []

        for value in list(values)[:limit]:
            clean = cls._sanitize_text(
                value,
                max_length=max_length,
            )

            if clean:
                result.append(clean)

        return list(
            dict.fromkeys(result)
        )

    # =========================================================================
    # CONSTANT-TIME DIGEST COMPARISON
    # =========================================================================

    @staticmethod
    def secure_digest_equal(
        left: str,
        right: str,
    ) -> bool:
        """
        Constant-time comparison helper.

        Useful when comparing a supplied digest against a sensitive local
        reference. Normal dictionary/set lookups do not require this helper.
        """

        if not isinstance(left, str):
            return False

        if not isinstance(right, str):
            return False

        return hmac.compare_digest(
            left.lower(),
            right.lower(),
        )


# =============================================================================
# CONVENIENCE API
# =============================================================================

async def assess_hash_reputation(
    digest: str,
    algorithm: HashAlgorithm | str | None = None,
    *,
    threat_intel: Optional[
        "ThreatIntelAggregator"
    ] = None,
    fuzzy_provider: Optional[
        FuzzyHashProvider
    ] = None,
    allowlist: Optional[
        HashAllowlist
    ] = None,
    force_refresh: bool = False,
) -> HashReputationAssessment:
    """
    One-shot API.

    For high-throughput applications, instantiate HashReputationEngine once
    and reuse it so cache, request coalescing, semaphores, and metrics remain
    effective.
    """

    engine = HashReputationEngine(
        threat_intel=threat_intel,
        fuzzy_provider=fuzzy_provider,
        allowlist=allowlist,
    )

    return await engine.analyze(
        digest,
        algorithm,
        force_refresh=force_refresh,
    )


async def assess_file_reputation(
    path: str | Path,
    *,
    algorithms: Optional[
        Sequence[HashAlgorithm]
    ] = None,
    threat_intel: Optional[
        "ThreatIntelAggregator"
    ] = None,
    fuzzy_provider: Optional[
        FuzzyHashProvider
    ] = None,
    allowlist: Optional[
        HashAllowlist
    ] = None,
) -> FileReputationAssessment:
    """
    One-shot file reputation API.
    """

    engine = HashReputationEngine(
        threat_intel=threat_intel,
        fuzzy_provider=fuzzy_provider,
        allowlist=allowlist,
    )

    return await engine.analyze_file(
        path,
        algorithms=algorithms,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MODULE_VERSION",

    # enums
    "HashAlgorithm",
    "HashMatchType",
    "HashReputationStatus",
    "HashRiskLevel",
    "HashEvidenceType",

    # exceptions
    "HashReputationError",
    "InvalidHashError",
    "UnsupportedHashAlgorithmError",
    "HashResolutionError",
    "HashConfigurationError",
    "FileHashingError",
    "FileChangedDuringHashingError",

    # models
    "HashIdentity",
    "HashDigestSet",
    "HashFileMetadata",
    "ProviderObservation",
    "FuzzyHashMatch",
    "HashSignal",
    "HashReputationAssessment",
    "FileReputationAssessment",

    # providers / adapters
    "HashAllowlist",
    "InMemoryHashAllowlist",
    "FuzzyHashProvider",

    # cache
    "HashReputationCache",

    # hashing
    "StreamingFileHasher",

    # helpers
    "normalize_hash",
    "infer_hash_algorithms",

    # engine
    "HashReputationEngine",

    # convenience
    "assess_hash_reputation",
    "assess_file_reputation",
]
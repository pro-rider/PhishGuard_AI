"""
PhishGuard_AI - Production Hash IOC Extractor
Version: 1.0.0

Purpose
-------
Extract cryptographic hash indicators from untrusted text without performing
any network, filesystem, or subprocess operation.

Supported digest families
-------------------------
MD5, SHA-1, SHA-224, SHA-256, SHA-384, SHA-512, SHA3-224/256/384/512,
BLAKE2S-256, BLAKE2B-512.

Additional formats
------------------
- Explicit algorithm prefixes: ``sha256:<digest>``, ``blake2b:<digest>``, ...
- SRI: ``sha256-<base64>``, ``sha384-<base64>``, ``sha512-<base64>``
- Defanged hex digests: ``abc[.]def`` / ``abc(dot)def`` / ``abc [.] def``

Security properties
-------------------
- Extraction is strictly local and deterministic.
- Bounded input, candidate count, output count, and context size.
- ReDoS-resistant candidate discovery.
- Exact original-text span mapping and stable IOC IDs.
- No exception details from internal failures are exposed by default.
"""

from __future__ import annotations

import base64
import hashlib
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# CONSTANTS
# ============================================================================

MODULE_NAME = "ioc.hash_extractor"
MODULE_VERSION = "1.0.0"

DEFAULT_MAX_INPUT_CHARS = 1_000_000
DEFAULT_MAX_IOCS = 512
DEFAULT_MAX_CANDIDATES = 4_096
DEFAULT_CONTEXT_RADIUS = 180
DEFAULT_MAX_CANDIDATE_CHARS = 160
DEFAULT_MAX_EXTRACTION_SECONDS = 0.250

_ALGORITHM_PREFIXES: Mapping[str, int] = {
    "md5": 32,
    "sha1": 40,
    "sha224": 56,
    "sha256": 64,
    "sha384": 96,
    "sha512": 128,
    "sha3-224": 56,
    "sha3-256": 64,
    "sha3-384": 96,
    "sha3-512": 128,
    "blake2s-256": 64,
    "blake2b-512": 128,
}

_SRI_PREFIXES: Mapping[str, int] = {
    "sha256": 44,
    "sha384": 64,
    "sha512": 88,
}

_HEX32 = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32}(?![0-9a-fA-F])")
_HEX40 = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_HEX56 = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{56}(?![0-9a-fA-F])")
_HEX64 = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_HEX96 = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{96}(?![0-9a-fA-F])")
_HEX128 = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{128}(?![0-9a-fA-F])")

_SRI_REGEX = re.compile(
    r"(?<![A-Za-z0-9_-])(sha256|sha384|sha512)-"
    r"([A-Za-z0-9+/]{20,}={0,2})(?![A-Za-z0-9+/=])"
)

_PREFIX_REGEX = re.compile(
    r"(?<![A-Za-z0-9])(md5|sha1|sha224|sha256|sha384|sha512|"
    r"sha3-224|sha3-256|sha3-384|sha3-512|blake2s-256|blake2b-512)"
    r":([0-9a-fA-F]{32,128})(?![0-9a-fA-F])"
)

_DEFANG_REGEX = re.compile(
    r"\b([0-9a-fA-F]{8,})(?:\[\.\]|\(dot\)|\s*\[\.\]\s*)"
    r"([0-9a-fA-F]{8,})\b"
)

# ============================================================================
# ENUMS
# ============================================================================


class HashAlgorithm(str, Enum):
    """Supported digest families."""

    MD5 = "md5"
    SHA1 = "sha1"
    SHA224 = "sha224"
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    SHA3_224 = "sha3-224"
    SHA3_256 = "sha3-256"
    SHA3_384 = "sha3-384"
    SHA3_512 = "sha3-512"
    BLAKE2S_256 = "blake2s-256"
    BLAKE2B_512 = "blake2b-512"


class HashEncoding(str, Enum):
    """How the digest was represented in the source text."""

    HEX = "hex"
    SRI = "sri"
    DEFANGED = "defanged"
    PREFIXED = "prefixed"


class HashValidationStatus(str, Enum):
    """Validation outcome for a candidate."""

    VALID = "valid"
    REJECTED = "rejected"


# ============================================================================
# MODELS
# ============================================================================


class HashExtractorConfig(BaseModel):
    """Configuration for the hash IOC extractor."""

    model_config = ConfigDict(extra="forbid")

    max_input_chars: int = Field(default=DEFAULT_MAX_INPUT_CHARS, ge=1, le=10_000_000)
    max_iocs: int = Field(default=DEFAULT_MAX_IOCS, ge=1, le=10_000)
    max_candidates: int = Field(default=DEFAULT_MAX_CANDIDATES, ge=1, le=100_000)
    context_radius: int = Field(default=DEFAULT_CONTEXT_RADIUS, ge=0, le=10_000)
    max_extraction_seconds: float = Field(
        default=DEFAULT_MAX_EXTRACTION_SECONDS, ge=0.001, le=30.0
    )
    include_sri: bool = True
    include_defanged: bool = True
    include_prefixed: bool = True


class HashIOC(BaseModel):
    """A single extracted hash indicator."""

    model_config = ConfigDict(extra="forbid")

    ioc_id: str
    algorithm: HashAlgorithm
    encoding: HashEncoding
    digest: str
    raw_value: str
    start: int
    end: int
    confidence: float = Field(ge=0.0, le=1.0)
    validation_status: HashValidationStatus = HashValidationStatus.VALID
    context: str = ""


class HashExtractionResult(BaseModel):
    """The complete result of a hash extraction pass."""

    model_config = ConfigDict(extra="forbid")

    iocs: List[HashIOC] = Field(default_factory=list)
    candidate_count: int = 0
    rejected_count: int = 0
    truncated: bool = False
    extraction_ms: float = 0.0


class HashExtractionError(Exception):
    """Base error for hash extraction."""


class HashInputTooLargeError(HashExtractionError):
    """Input exceeded the configured maximum length."""


# ============================================================================
# METRICS
# ============================================================================


class HashExtractorMetrics:
    """Thread-safe, thread-local-batched metrics for the extractor."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict = {}
        self._local = threading.local()

    def increment(self, field_name: str, amount: int = 1) -> None:
        bucket = getattr(self._local, "b", None)
        if bucket is None:
            bucket = {}
            self._local.b = bucket
        bucket[field_name] = bucket.get(field_name, 0) + amount

    def add_duration(self, seconds: float) -> None:
        bucket = getattr(self._local, "b", None)
        if bucket is None:
            bucket = {}
            self._local.b = bucket
        bucket["total_extraction_seconds"] = (
            bucket.get("total_extraction_seconds", 0.0) + seconds
        )

    def snapshot(self) -> Mapping[str, Any]:
        with self._lock:
            merged: dict = {
                "extraction_calls": 0,
                "total_extraction_seconds": 0.0,
            }
            bucket = getattr(self._local, "b", None) or {}
            for k, v in bucket.items():
                if k == "total_extraction_seconds":
                    merged[k] = merged.get(k, 0.0) + v
                else:
                    merged[k] = merged.get(k, 0) + v
            merged["extraction_calls"] = merged.get("extraction_calls", 0) + (
                1 if bucket else 0
            )
            return merged


# ============================================================================
# EXTRACTOR
# ============================================================================


class HashExtractor:
    """
    Production hash IOC extractor.

    Pure local, deterministic, thread-safe extraction of cryptographic digests
    from untrusted text. The parser enforces strict candidate and output bounds.
    """

    def __init__(
        self,
        config: Optional[HashExtractorConfig] = None,
        *,
        metrics: Optional[HashExtractorMetrics] = None,
        clock=time.monotonic,
    ) -> None:
        self.config = config or HashExtractorConfig()
        self.metrics = metrics or HashExtractorMetrics()
        self._clock = clock

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> List[HashIOC]:
        """Extract validated hash IOCs from ``text``, in appearance order."""
        started = self._clock()

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return []

        if len(text) > self.config.max_input_chars:
            self.metrics.increment("input_too_large")
            raise HashInputTooLargeError(
                "input exceeds configured maximum length"
            )

        candidates = self._find_candidates(text)
        valid = self._validate_candidates(text, candidates)

        self.metrics.increment("extraction_calls")
        self.metrics.add_duration(self._clock() - started)
        return valid

    def extract_full(self, text: str) -> HashExtractionResult:
        """Extract and also return candidate/rejection statistics."""
        started = self._clock()
        try:
            iocs = self.extract(text)
        except HashExtractionError:
            raise
        candidates = self._find_candidates(text)
        return HashExtractionResult(
            iocs=iocs,
            candidate_count=len(candidates),
            rejected_count=max(0, len(candidates) - len(iocs)),
            truncated=len(iocs) >= self.config.max_iocs,
            extraction_ms=(self._clock() - started) * 1000.0,
        )

    def metrics_snapshot(self) -> Mapping[str, Any]:
        """Return a copy of the extractor metrics."""
        return self.metrics.snapshot()

    # ------------------------------------------------------------------
    # Candidate discovery
    # ------------------------------------------------------------------

    def _find_candidates(self, text: str) -> List[HashIOC]:
        candidates: List[HashIOC] = []
        seen: set = set()

        def emit(
            raw: str,
            algorithm: HashAlgorithm,
            encoding: HashEncoding,
            start: int,
            end: int,
            digest: str,
            confidence: float,
        ) -> None:
            dedup_key = (digest, algorithm.value, encoding.value)
            if dedup_key in seen:
                return
            seen.add(dedup_key)
            candidates.append(
                HashIOC(
                    ioc_id=f"hash_{self._stable_id(digest)}",
                    algorithm=algorithm,
                    encoding=encoding,
                    digest=digest,
                    raw_value=raw,
                    start=start,
                    end=end,
                    confidence=confidence,
                    context=self._bounded_context(text, start, end),
                )
            )

        self._scan_plain_hex(text, emit)
        if self.config.include_sri:
            self._scan_sri(text, emit)
        if self.config.include_prefixed:
            self._scan_prefixed(text, emit)
        if self.config.include_defanged:
            self._scan_defanged(text, emit)

        candidates.sort(key=lambda c: (c.start, c.end))
        return candidates[: self.config.max_candidates]

    def _scan_plain_hex(self, text: str, emit) -> None:
        for length, regex, algorithm in (
            (32, _HEX32, HashAlgorithm.MD5),
            (40, _HEX40, HashAlgorithm.SHA1),
            (56, _HEX56, HashAlgorithm.SHA224),
            (64, _HEX64, HashAlgorithm.SHA256),
            (96, _HEX96, HashAlgorithm.SHA384),
            (128, _HEX128, HashAlgorithm.SHA512),
        ):
            for match in regex.finditer(text):
                emit(
                    raw=match.group(0),
                    algorithm=algorithm,
                    encoding=HashEncoding.HEX,
                    start=match.start(),
                    end=match.end(),
                    digest=match.group(0).lower(),
                    confidence=0.90,
                )

    def _scan_sri(self, text: str, emit) -> None:
        algorithm_map = {
            "sha256": HashAlgorithm.SHA256,
            "sha384": HashAlgorithm.SHA384,
            "sha512": HashAlgorithm.SHA512,
        }
        for match in _SRI_REGEX.finditer(text):
            algo = match.group(1).lower()
            b64 = match.group(2)
            if len(b64) != _SRI_PREFIXES.get(algo):
                continue
            digest = self._b64_to_hex(b64)
            if digest is None:
                continue
            emit(
                raw=match.group(0),
                algorithm=algorithm_map[algo],
                encoding=HashEncoding.SRI,
                start=match.start(),
                end=match.end(),
                digest=digest,
                confidence=0.95,
            )

    def _scan_prefixed(self, text: str, emit) -> None:
        for match in _PREFIX_REGEX.finditer(text):
            algo_name = match.group(1).lower()
            digest = match.group(2).lower()
            if len(digest) != _ALGORITHM_PREFIXES.get(algo_name):
                continue
            try:
                algorithm = HashAlgorithm(algo_name)
            except ValueError:
                continue
            emit(
                raw=match.group(0),
                algorithm=algorithm,
                encoding=HashEncoding.PREFIXED,
                start=match.start(),
                end=match.end(),
                digest=digest,
                confidence=0.98,
            )

    def _scan_defanged(self, text: str, emit) -> None:
        algorithm_map = {
            32: HashAlgorithm.MD5,
            40: HashAlgorithm.SHA1,
            64: HashAlgorithm.SHA256,
        }
        for match in _DEFANG_REGEX.finditer(text):
            joined = (match.group(1) + match.group(2)).lower()
            algorithm = algorithm_map.get(len(joined))
            if algorithm is None:
                continue
            emit(
                raw=match.group(0),
                algorithm=algorithm,
                encoding=HashEncoding.DEFANGED,
                start=match.start(),
                end=match.end(),
                digest=joined,
                confidence=0.88,
            )

    # ------------------------------------------------------------------
    # Validation & helpers
    # ------------------------------------------------------------------

    def _validate_candidates(
        self, text: str, candidates: Sequence[HashIOC]
    ) -> List[HashIOC]:
        valid: List[HashIOC] = []
        for candidate in candidates:
            if self._is_standalone(text, candidate):
                valid.append(candidate)
            if len(valid) >= self.config.max_iocs:
                break
        return valid

    @staticmethod
    def _is_standalone(text: str, candidate: HashIOC) -> bool:
        before = text[max(0, candidate.start - 1): candidate.start]
        after = text[candidate.end: candidate.end + 1]
        return not (before.isalnum() or after.isalnum())

    def _bounded_context(self, text: str, start: int, end: int) -> str:
        radius = self.config.context_radius
        left = max(0, start - radius)
        right = min(len(text), end + radius)
        return text[left:right]

    @staticmethod
    def _stable_id(digest: str) -> str:
        return hashlib.sha256(digest.encode("ascii")).hexdigest()[:16]

    @staticmethod
    def _b64_to_hex(b64: str) -> Optional[str]:
        try:
            padded = b64 + "=" * (-len(b64) % 4)
            return base64.b64decode(padded, validate=True).hex()
        except Exception:  # noqa: BLE001
            return None


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "HashAlgorithm",
    "HashEncoding",
    "HashValidationStatus",
    "HashExtractorConfig",
    "HashIOC",
    "HashExtractionResult",
    "HashExtractionError",
    "HashInputTooLargeError",
    "HashExtractorMetrics",
    "HashExtractor",
    "MODULE_VERSION",
]

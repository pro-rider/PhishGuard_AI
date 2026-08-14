"""
PhishGuard AI - Threat Intelligence Multi-Provider Aggregator
==============================================================

Version: 14.0.0
File Path: threat_intel/aggregator.py

Production-grade multi-provider threat-intelligence aggregation engine.

Responsibilities
----------------
* Canonical IOC normalization and validation
* Provider capability routing
* Concurrent provider fan-out
* Per-provider resilience/circuit-breaker integration
* Request-level concurrency control
* Weighted consensus calculation
* Provider-confidence / coverage calculation
* Degraded-operation detection
* Bounded TTL L1 caching
* Cache stampede protection
* Deterministic result ordering
* Provider-result validation
* Safe error isolation
* Health-check orchestration
* Operational cache statistics
* Explicit lifecycle management

Security properties
-------------------
* No API credentials are stored here.
* IOC values are never included in cache logs.
* Cache keys are SHA-256 derived.
* Cache size is bounded.
* Provider failures cannot crash the complete aggregation request.
* Unsupported IOC types are never routed to unrelated provider methods.
* Malformed provider responses are rejected safely.
* Cache entries are copied before returning to callers.
* User-controlled IOC values are validated before provider dispatch.

Compatibility
-------------
This module is intentionally compatible with the existing:

    BaseThreatProvider
    IOCType
    NormalizedThreatReport
    ThreatSeverity
    CircuitBreaker
    execute_with_resilience

contracts from PhishGuard AI.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from pydantic import BaseModel, Field, field_validator

from threat_intel.base import (
    BaseThreatProvider,
    IOCType,
    NormalizedThreatReport,
    ThreatSeverity,
)
from threat_intel.resilience import CircuitBreaker, execute_with_resilience


logger = logging.getLogger("phishguard.threat_intel.aggregator")


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_PROVIDER_WEIGHTS: Dict[str, float] = {
    "virustotal": 1.00,
    "urlhaus": 0.90,
    "abuseipdb": 0.80,
}

DEFAULT_CACHE_TTL_SECONDS = 3600
DEFAULT_MAX_CACHE_ENTRIES = 4096

DEFAULT_MAX_CONCURRENT_PROVIDER_QUERIES = 16
DEFAULT_MAX_CONCURRENT_HEALTH_CHECKS = 8

DEFAULT_MAX_RETRIES_PER_PROVIDER = 2

MIN_IOC_LENGTH = 1
MAX_IOC_LENGTH = 8192

CACHE_KEY_PREFIX = "phishguard:ti:v14"

# Consensus thresholds.
MALICIOUS_THRESHOLD = 0.40
CRITICAL_THRESHOLD = 0.85

# Confidence thresholds.
MIN_CONFIDENCE_FOR_STRONG_DECISION = 0.70

# A provider response marked with this tag is treated as degraded and
# contributes no authority weight.
DEGRADED_FALLBACK_TAG = "degraded_fallback"

# Reports carrying these tags are not considered successful authoritative
# intelligence.
NON_AUTHORITATIVE_TAGS = {
    DEGRADED_FALLBACK_TAG,
    "unsupported_ioc_type",
    "provider_unavailable",
    "provider_timeout",
    "provider_error",
    "circuit_open",
}


# =============================================================================
# DATA MODELS
# =============================================================================

class AggregatedThreatSummary(BaseModel):
    """
    Provider-agnostic aggregate threat-intelligence result.

    This model intentionally contains operational coverage information in
    addition to the threat score so downstream detection code can distinguish:

        "clean because providers agree"

    from:

        "clean because providers were unavailable."
    """

    ioc_value: str = Field(..., min_length=MIN_IOC_LENGTH, max_length=MAX_IOC_LENGTH)
    ioc_type: IOCType

    # -------------------------------------------------------------------------
    # Consensus
    # -------------------------------------------------------------------------

    aggregated_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)

    overall_severity: ThreatSeverity = Field(
        default=ThreatSeverity.UNKNOWN
    )

    is_malicious: bool = False

    max_single_provider_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    # -------------------------------------------------------------------------
    # Provider coverage
    # -------------------------------------------------------------------------

    total_providers_queried: int = Field(default=0, ge=0)

    successful_provider_responses: int = Field(default=0, ge=0)

    failed_provider_responses: int = Field(default=0, ge=0)

    positive_detection_count: int = Field(default=0, ge=0)

    authoritative_provider_count: int = Field(default=0, ge=0)

    provider_coverage_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    decision_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    # -------------------------------------------------------------------------
    # Threat metadata
    # -------------------------------------------------------------------------

    combined_threat_categories: List[str] = Field(default_factory=list)

    combined_raw_tags: List[str] = Field(default_factory=list)

    # -------------------------------------------------------------------------
    # Execution metadata
    # -------------------------------------------------------------------------

    total_execution_time_ms: float = Field(default=0.0, ge=0.0)

    is_degraded: bool = False

    cache_hit: bool = False

    generated_at: float = Field(default_factory=time.time, ge=0.0)

    # -------------------------------------------------------------------------
    # Provider reports
    # -------------------------------------------------------------------------

    provider_reports: Dict[str, NormalizedThreatReport] = Field(
        default_factory=dict
    )

    @field_validator("aggregated_risk_score", "max_single_provider_score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        """Reject NaN and infinity even though Pydantic bounds are present."""
        value = float(value)

        if not math.isfinite(value):
            raise ValueError("Threat score must be finite.")

        return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class _CacheEntry:
    """
    Immutable internal L1 cache entry.
    """

    summary: AggregatedThreatSummary
    expires_at: float


@dataclass(frozen=True)
class _ProviderResult:
    """
    Internal normalized provider execution result.
    """

    provider_name: str
    report: Optional[NormalizedThreatReport]
    error: Optional[BaseException]
    elapsed_ms: float


# =============================================================================
# AGGREGATOR
# =============================================================================

class ThreatIntelAggregator:
    """
    Production multi-provider threat-intelligence aggregation engine.

    The aggregator is intentionally provider-agnostic. Provider-specific API
    behavior belongs in provider implementations, while resilience and
    consensus policy belong here.
    """

    DEFAULT_PROVIDER_WEIGHTS = DEFAULT_PROVIDER_WEIGHTS

    def __init__(
        self,
        providers: Sequence[BaseThreatProvider],
        provider_weights: Optional[Mapping[str, float]] = None,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        max_retries_per_provider: int = DEFAULT_MAX_RETRIES_PER_PROVIDER,
        max_cache_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
        max_concurrent_provider_queries: int = (
            DEFAULT_MAX_CONCURRENT_PROVIDER_QUERIES
        ),
        max_concurrent_health_checks: int = (
            DEFAULT_MAX_CONCURRENT_HEALTH_CHECKS
        ),
    ) -> None:
        """
        Initialize the threat-intelligence aggregator.

        Args:
            providers:
                Registered threat-intelligence providers.

            provider_weights:
                Optional provider authority weights.

            cache_ttl_seconds:
                L1 cache TTL.

            max_retries_per_provider:
                Maximum retries delegated to resilience.py.

            max_cache_entries:
                Hard upper bound for in-memory cache entries.

            max_concurrent_provider_queries:
                Maximum simultaneous provider requests generated by this
                aggregator instance.

            max_concurrent_health_checks:
                Maximum simultaneous provider health checks.
        """

        if not providers:
            raise ValueError(
                "ThreatIntelAggregator requires at least one provider."
            )

        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds cannot be negative.")

        if max_retries_per_provider < 0:
            raise ValueError(
                "max_retries_per_provider cannot be negative."
            )

        if max_cache_entries <= 0:
            raise ValueError("max_cache_entries must be greater than zero.")

        if max_concurrent_provider_queries <= 0:
            raise ValueError(
                "max_concurrent_provider_queries must be greater than zero."
            )

        if max_concurrent_health_checks <= 0:
            raise ValueError(
                "max_concurrent_health_checks must be greater than zero."
            )

        # ---------------------------------------------------------------------
        # Provider registration
        # ---------------------------------------------------------------------

        provider_map: Dict[str, BaseThreatProvider] = {}

        for provider in providers:
            if not isinstance(provider, BaseThreatProvider):
                raise TypeError(
                    "All providers must implement BaseThreatProvider."
                )

            name = provider.provider_name

            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "Provider name must be a non-empty string."
                )

            name = name.strip().lower()

            if name in provider_map:
                raise ValueError(
                    f"Duplicate threat-intelligence provider: {name}"
                )

            supported_types = provider.supported_ioc_types

            if not supported_types:
                logger.warning(
                    "Provider '%s' advertises no supported IOC types.",
                    name,
                )

            provider_map[name] = provider

        self.providers: Dict[str, BaseThreatProvider] = dict(
            sorted(provider_map.items())
        )

        # ---------------------------------------------------------------------
        # Provider weights
        # ---------------------------------------------------------------------

        supplied_weights = dict(provider_weights or {})

        merged_weights: Dict[str, float] = {}

        for provider_name in self.providers:
            configured = supplied_weights.get(
                provider_name,
                self.DEFAULT_PROVIDER_WEIGHTS.get(
                    provider_name,
                    1.0,
                ),
            )

            try:
                weight = float(configured)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid provider weight for '{provider_name}'."
                ) from exc

            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(
                    f"Provider weight for '{provider_name}' "
                    "must be finite and greater than zero."
                )

            merged_weights[provider_name] = weight

        self.provider_weights = merged_weights

        # ---------------------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------------------

        self.cache_ttl_seconds = int(cache_ttl_seconds)
        self.max_retries = int(max_retries_per_provider)
        self.max_cache_entries = int(max_cache_entries)

        # ---------------------------------------------------------------------
        # Resilience
        # ---------------------------------------------------------------------

        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            name: CircuitBreaker(provider_name=name)
            for name in self.providers
        }

        # ---------------------------------------------------------------------
        # Concurrency controls
        # ---------------------------------------------------------------------

        self._provider_semaphore = asyncio.Semaphore(
            max_concurrent_provider_queries
        )

        self._health_semaphore = asyncio.Semaphore(
            max_concurrent_health_checks
        )

        # ---------------------------------------------------------------------
        # Cache
        # ---------------------------------------------------------------------

        self._l1_cache: "OrderedDict[str, _CacheEntry]" = OrderedDict()

        # Async lock protects both cache mutation and in-flight request map.
        self._cache_lock = asyncio.Lock()

        # Cache stampede prevention.
        self._inflight_requests: Dict[
            str,
            asyncio.Task[AggregatedThreatSummary],
        ] = {}

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------

        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def analyze_ioc(
        self,
        ioc_value: str,
        ioc_type: Optional[IOCType] = None,
        force_refresh: bool = False,
    ) -> AggregatedThreatSummary:
        """
        Analyze an IOC across all applicable providers.

        The method is safe for concurrent callers and includes cache
        stampede protection.
        """

        clean_ioc = self._canonicalize_ioc(ioc_value)

        resolved_type = (
            IOCType(ioc_type)
            if ioc_type is not None
            else self._infer_ioc_type(clean_ioc)
        )

        self._validate_ioc_for_type(
            clean_ioc,
            resolved_type,
        )

        cache_key = self._build_cache_key(
            clean_ioc,
            resolved_type,
        )

        if not force_refresh:
            cached = await self._cache_get(cache_key)

            if cached is not None:
                self._cache_hits += 1

                return self._clone_summary(
                    cached,
                    cache_hit=True,
                )

        self._cache_misses += 1

        # ---------------------------------------------------------------------
        # Stampede protection
        # ---------------------------------------------------------------------

        async with self._cache_lock:
            if not force_refresh:
                existing = self._l1_cache.get(cache_key)

                if existing is not None:
                    if existing.expires_at > time.monotonic():
                        self._l1_cache.move_to_end(cache_key)

                        return self._clone_summary(
                            existing.summary,
                            cache_hit=True,
                        )

                    self._l1_cache.pop(cache_key, None)

            existing_task = self._inflight_requests.get(cache_key)

            if existing_task is not None and not existing_task.done():
                task = existing_task
            else:
                task = asyncio.create_task(
                    self._analyze_ioc_uncached(
                        clean_ioc=clean_ioc,
                        resolved_type=resolved_type,
                    )
                )

                self._inflight_requests[cache_key] = task

        try:
            result = await asyncio.shield(task)

            return self._clone_summary(
                result,
                cache_hit=False,
            )

        finally:
            async with self._cache_lock:
                current = self._inflight_requests.get(cache_key)

                if current is task:
                    self._inflight_requests.pop(
                        cache_key,
                        None,
                    )

    async def check_all_provider_health(self) -> Dict[str, bool]:
        """
        Check health of every registered provider concurrently.

        One provider's exception cannot prevent other health checks.
        """

        async def check(
            name: str,
            provider: BaseThreatProvider,
        ) -> Tuple[str, bool]:
            async with self._health_semaphore:
                try:
                    result = await provider.check_health()

                    return name, bool(result)

                except asyncio.CancelledError:
                    raise

                except Exception:
                    logger.exception(
                        "Threat-intel provider health check failed: %s",
                        name,
                    )

                    return name, False

        tasks = [
            asyncio.create_task(
                check(name, provider)
            )
            for name, provider in self.providers.items()
        ]

        results = await asyncio.gather(*tasks)

        return dict(sorted(results))

    async def clear_cache(self) -> None:
        """
        Clear the entire L1 cache.
        """

        async with self._cache_lock:
            self._l1_cache.clear()

    async def invalidate_ioc(
        self,
        ioc_value: str,
        ioc_type: Optional[IOCType] = None,
    ) -> bool:
        """
        Remove a specific IOC from the L1 cache.
        """

        clean_ioc = self._canonicalize_ioc(ioc_value)

        resolved_type = (
            IOCType(ioc_type)
            if ioc_type is not None
            else self._infer_ioc_type(clean_ioc)
        )

        cache_key = self._build_cache_key(
            clean_ioc,
            resolved_type,
        )

        async with self._cache_lock:
            return self._l1_cache.pop(
                cache_key,
                None,
            ) is not None

    def cache_stats(self) -> Dict[str, int]:
        """
        Return cache operational counters.
        """

        return {
            "entries": len(self._l1_cache),
            "max_entries": self.max_cache_entries,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "evictions": self._cache_evictions,
            "inflight_requests": len(self._inflight_requests),
        }

    # =========================================================================
    # UNCACHED ANALYSIS
    # =========================================================================

    async def _analyze_ioc_uncached(
        self,
        clean_ioc: str,
        resolved_type: IOCType,
    ) -> AggregatedThreatSummary:
        """
        Execute a fresh multi-provider analysis.
        """

        start_time = time.perf_counter()

        eligible_providers = [
            provider
            for provider in self.providers.values()
            if resolved_type in provider.supported_ioc_types
        ]

        if not eligible_providers:
            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000.0

            return AggregatedThreatSummary(
                ioc_value=clean_ioc,
                ioc_type=resolved_type,
                aggregated_risk_score=0.0,
                overall_severity=ThreatSeverity.UNKNOWN,
                is_malicious=False,
                total_providers_queried=0,
                successful_provider_responses=0,
                failed_provider_responses=0,
                positive_detection_count=0,
                authoritative_provider_count=0,
                provider_coverage_ratio=0.0,
                decision_confidence=0.0,
                total_execution_time_ms=elapsed_ms,
                is_degraded=True,
            )

        # Deterministic ordering.
        eligible_providers.sort(
            key=lambda provider: provider.provider_name.lower()
        )

        tasks = [
            asyncio.create_task(
                self._query_provider_resilient(
                    provider=provider,
                    ioc_value=clean_ioc,
                    ioc_type=resolved_type,
                )
            )
            for provider in eligible_providers
        ]

        raw_results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        provider_results: List[_ProviderResult] = []

        for provider, result in zip(
            eligible_providers,
            raw_results,
        ):
            if isinstance(result, BaseException):
                provider_results.append(
                    _ProviderResult(
                        provider_name=provider.provider_name,
                        report=None,
                        error=result,
                        elapsed_ms=0.0,
                    )
                )

            else:
                provider_results.append(result)

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        summary = self._aggregate_reports(
            ioc_value=clean_ioc,
            ioc_type=resolved_type,
            provider_results=provider_results,
            total_execution_time_ms=elapsed_ms,
        )

        await self._cache_put(
            self._build_cache_key(
                clean_ioc,
                resolved_type,
            ),
            summary,
        )

        return summary

    # =========================================================================
    # PROVIDER EXECUTION
    # =========================================================================

    async def _query_provider_resilient(
        self,
        provider: BaseThreatProvider,
        ioc_value: str,
        ioc_type: IOCType,
    ) -> _ProviderResult:
        """
        Execute one provider behind concurrency and resilience boundaries.
        """

        provider_name = provider.provider_name
        start_time = time.perf_counter()

        try:
            async with self._provider_semaphore:

                coro_factory = self._build_provider_query(
                    provider=provider,
                    ioc_value=ioc_value,
                    ioc_type=ioc_type,
                )

                report = await execute_with_resilience(
                    query_coro_func=coro_factory,
                    provider_name=provider_name,
                    circuit_breaker=self.circuit_breakers[
                        provider_name
                    ],
                    ioc_value=ioc_value,
                    ioc_type=ioc_type,
                    max_retries=self.max_retries,
                    fallback_on_error=True,
                )

            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000.0

            validated_report = self._validate_provider_report(
                report=report,
                expected_provider=provider_name,
                expected_ioc=ioc_value,
                expected_ioc_type=ioc_type,
            )

            return _ProviderResult(
                provider_name=provider_name,
                report=validated_report,
                error=None,
                elapsed_ms=elapsed_ms,
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000.0

            logger.warning(
                "Threat-intel provider failed: %s",
                provider_name,
                exc_info=True,
            )

            return _ProviderResult(
                provider_name=provider_name,
                report=None,
                error=exc,
                elapsed_ms=elapsed_ms,
            )

    @staticmethod
    def _build_provider_query(
        provider: BaseThreatProvider,
        ioc_value: str,
        ioc_type: IOCType,
    ):
        """
        Return the correct provider coroutine factory.

        Unsupported IOC types are rejected rather than silently routed to an
        unrelated provider method.
        """

        if ioc_type == IOCType.URL:
            if not hasattr(provider, "query_url"):
                raise TypeError(
                    f"Provider '{provider.provider_name}' does not "
                    "implement URL querying."
                )

            return lambda: provider.query_url(ioc_value)

        if ioc_type == IOCType.DOMAIN:
            if not hasattr(provider, "query_domain"):
                raise TypeError(
                    f"Provider '{provider.provider_name}' does not "
                    "implement domain querying."
                )

            return lambda: provider.query_domain(ioc_value)

        if ioc_type == IOCType.IP:
            if not hasattr(provider, "query_ip"):
                raise TypeError(
                    f"Provider '{provider.provider_name}' does not "
                    "implement IP querying."
                )

            return lambda: provider.query_ip(ioc_value)

        raise ValueError(
            f"IOC type '{ioc_type.value}' is not supported by the "
            "current aggregator dispatch contract."
        )

    # =========================================================================
    # REPORT VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_provider_report(
        report: Any,
        expected_provider: str,
        expected_ioc: str,
        expected_ioc_type: IOCType,
    ) -> NormalizedThreatReport:
        """
        Validate and normalize provider output before it enters consensus.
        """

        if not isinstance(report, NormalizedThreatReport):
            raise TypeError(
                f"Provider '{expected_provider}' returned "
                f"unexpected result type: {type(report).__name__}"
            )

        if report.provider_name.strip().lower() != expected_provider.strip().lower():
            raise ValueError(
                f"Provider identity mismatch: expected "
                f"'{expected_provider}', got '{report.provider_name}'."
            )

        if report.ioc_type != expected_ioc:
            raise ValueError(
                f"IOC type mismatch from provider '{expected_provider}'."
            )

        # IOC values may be canonicalized differently by providers, therefore
        # compare canonical values rather than requiring raw-string equality.
        if (
            ThreatIntelAggregator._canonicalize_ioc(
                report.ioc_value
            )
            != expected_ioc
        ):
            raise ValueError(
                f"IOC value mismatch from provider '{expected_provider}'."
            )

        if not math.isfinite(report.risk_score):
            raise ValueError(
                f"Provider '{expected_provider}' returned non-finite risk."
            )

        if report.positive_detections > report.total_engines:
            # A provider can legitimately expose report counts rather than
            # scanner counts, but within this normalized schema the invariant
            # must hold.
            raise ValueError(
                f"Provider '{expected_provider}' returned "
                "positive_detections > total_engines."
            )

        return report

    # =========================================================================
    # AGGREGATION
    # =========================================================================

    def _aggregate_reports(
        self,
        ioc_value: str,
        ioc_type: IOCType,
        provider_results: Sequence[_ProviderResult],
        total_execution_time_ms: float,
    ) -> AggregatedThreatSummary:
        """
        Compute weighted consensus and operational confidence.
        """

        provider_reports: Dict[str, NormalizedThreatReport] = {}

        weighted_score_sum = 0.0
        total_weight = 0.0

        categories: Set[str] = set()
        tags: Set[str] = set()

        successful = 0
        failed = 0
        authoritative = 0
        positive_detections = 0

        max_provider_score = 0.0

        degraded = False

        # Deterministic provider ordering.
        ordered_results = sorted(
            provider_results,
            key=lambda item: item.provider_name.lower(),
        )

        eligible_weight = sum(
            self.provider_weights.get(
                result.provider_name,
                1.0,
            )
            for result in ordered_results
        )

        for result in ordered_results:
            provider_name = result.provider_name

            if result.report is None:
                failed += 1
                degraded = True
                continue

            report = result.report

            provider_reports[provider_name] = report

            report_tags = {
                str(tag).strip().lower()
                for tag in report.raw_tags
                if str(tag).strip()
            }

            tags.update(report.raw_tags)
            categories.update(report.threat_categories)

            is_degraded = bool(
                report_tags.intersection(
                    NON_AUTHORITATIVE_TAGS
                )
            )

            if is_degraded:
                degraded = True
                continue

            successful += 1
            authoritative += 1

            max_provider_score = max(
                max_provider_score,
                report.risk_score,
            )

            if report.is_malicious:
                positive_detections += 1

            weight = self.provider_weights.get(
                provider_name,
                1.0,
            )

            weighted_score_sum += (
                report.risk_score * weight
            )

            total_weight += weight

        # ---------------------------------------------------------------------
        # Consensus score
        # ---------------------------------------------------------------------

        if total_weight > 0:
            consensus_score = (
                weighted_score_sum / total_weight
            )
        else:
            consensus_score = 0.0

        # ---------------------------------------------------------------------
        # Trusted critical signal
        # ---------------------------------------------------------------------
        #
        # Do not blindly let any provider force a critical result.
        #
        # The strongest provider score can act as a severity floor only when
        # the provider is authoritative and sufficiently weighted.
        # ---------------------------------------------------------------------

        critical_scores = [
            report.risk_score
            for provider_name, report in provider_reports.items()
            if report.risk_score >= CRITICAL_THRESHOLD
            and self.provider_weights.get(
                provider_name,
                1.0,
            ) >= 0.8
            and not set(report.raw_tags).intersection(
                NON_AUTHORITATIVE_TAGS
            )
        ]

        if critical_scores:
            consensus_score = max(
                consensus_score,
                min(critical_scores),
            )

        consensus_score = min(
            1.0,
            max(
                0.0,
                consensus_score,
            ),
        )

        # ---------------------------------------------------------------------
        # Severity
        # ---------------------------------------------------------------------

        if authoritative == 0:
            severity = ThreatSeverity.UNKNOWN
        else:
            severity = NormalizedThreatReport.calculate_severity(
                consensus_score
            )

        # ---------------------------------------------------------------------
        # Coverage
        # ---------------------------------------------------------------------

        total_queried = len(provider_results)

        if total_queried > 0:
            coverage_ratio = (
                authoritative / total_queried
            )
        else:
            coverage_ratio = 0.0

        # ---------------------------------------------------------------------
        # Decision confidence
        # ---------------------------------------------------------------------
        #
        # Confidence reflects:
        #   1. provider coverage
        #   2. authority-weight coverage
        #   3. agreement among provider scores
        #
        # It deliberately does not simply mirror risk_score.
        # ---------------------------------------------------------------------

        if authoritative == 0:
            confidence = 0.0

        else:
            authority_coverage = (
                total_weight / eligible_weight
                if eligible_weight > 0
                else 0.0
            )

            provider_scores = [
                report.risk_score
                for report in provider_reports.values()
                if not set(report.raw_tags).intersection(
                    NON_AUTHORITATIVE_TAGS
                )
            ]

            if len(provider_scores) <= 1:
                agreement = 1.0
            else:
                score_range = (
                    max(provider_scores)
                    - min(provider_scores)
                )

                agreement = max(
                    0.0,
                    1.0 - score_range,
                )

            confidence = (
                0.45 * coverage_ratio
                + 0.35 * authority_coverage
                + 0.20 * agreement
            )

            confidence = min(
                1.0,
                max(
                    0.0,
                    confidence,
                ),
            )

        # ---------------------------------------------------------------------
        # Malicious decision
        # ---------------------------------------------------------------------
        #
        # A positive provider result alone is not enough to override a fully
        # clean consensus. This prevents a single low-authority false positive
        # from dominating the aggregate.
        #
        # However, a sufficiently strong consensus score remains malicious.
        # ---------------------------------------------------------------------

        is_malicious = (
            consensus_score >= MALICIOUS_THRESHOLD
            or (
                positive_detections >= 2
                and authoritative >= 2
            )
            or (
                max_provider_score >= CRITICAL_THRESHOLD
                and authoritative >= 1
            )
        )

        # If there was no authoritative evidence, explicitly avoid presenting
        # UNKNOWN coverage as a clean result.
        if authoritative == 0:
            is_malicious = False

        return AggregatedThreatSummary(
            ioc_value=ioc_value,
            ioc_type=ioc_type,
            aggregated_risk_score=consensus_score,
            overall_severity=severity,
            is_malicious=is_malicious,
            max_single_provider_score=max_provider_score,
            total_providers_queried=total_queried,
            successful_provider_responses=successful,
            failed_provider_responses=failed,
            positive_detection_count=positive_detections,
            authoritative_provider_count=authoritative,
            provider_coverage_ratio=coverage_ratio,
            decision_confidence=confidence,
            combined_threat_categories=sorted(
                categories
            ),
            combined_raw_tags=sorted(
                tags
            ),
            total_execution_time_ms=max(
                0.0,
                float(total_execution_time_ms),
            ),
            is_degraded=degraded or failed > 0,
            cache_hit=False,
            provider_reports=dict(
                sorted(provider_reports.items())
            ),
        )

    # =========================================================================
    # CACHE
    # =========================================================================

    async def _cache_get(
        self,
        cache_key: str,
    ) -> Optional[AggregatedThreatSummary]:
        """
        Get a non-expired cache item.
        """

        now = time.monotonic()

        async with self._cache_lock:
            entry = self._l1_cache.get(cache_key)

            if entry is None:
                return None

            if entry.expires_at <= now:
                self._l1_cache.pop(
                    cache_key,
                    None,
                )

                return None

            self._l1_cache.move_to_end(cache_key)

            return entry.summary

    async def _cache_put(
        self,
        cache_key: str,
        summary: AggregatedThreatSummary,
    ) -> None:
        """
        Insert an entry using bounded LRU-style eviction.
        """

        if self.cache_ttl_seconds <= 0:
            return

        entry = _CacheEntry(
            summary=self._clone_summary(
                summary,
                cache_hit=False,
            ),
            expires_at=(
                time.monotonic()
                + self.cache_ttl_seconds
            ),
        )

        async with self._cache_lock:
            self._l1_cache[cache_key] = entry
            self._l1_cache.move_to_end(cache_key)

            # Bounded memory protection.
            while len(self._l1_cache) > self.max_cache_entries:
                self._l1_cache.popitem(
                    last=False
                )

                self._cache_evictions += 1

    # =========================================================================
    # IOC NORMALIZATION
    # =========================================================================

    @staticmethod
    def _canonicalize_ioc(ioc_value: str) -> str:
        """
        Canonicalize IOC text without performing DNS or network activity.

        Network access must never occur during canonicalization.
        """

        if not isinstance(ioc_value, str):
            raise TypeError("IOC value must be a string.")

        value = ioc_value.strip()

        if not value:
            raise ValueError("IOC value cannot be empty.")

        if len(value) > MAX_IOC_LENGTH:
            raise ValueError(
                f"IOC value exceeds maximum length of "
                f"{MAX_IOC_LENGTH} characters."
            )

        # Avoid accidental cache-key divergence from surrounding whitespace.
        return value

    @staticmethod
    def _infer_ioc_type(
        ioc_value: str,
    ) -> IOCType:
        """
        Infer IOC type conservatively.
        """

        value = ioc_value.strip()

        # Hashes first.
        if BaseThreatProvider.is_sha256(value):
            return IOCType.HASH_SHA256

        # IP addresses.
        if BaseThreatProvider.validate_ip(value):
            return IOCType.IP

        lowered = value.lower()

        # URLs should contain an explicit scheme.
        if "://" in lowered:
            return IOCType.URL

        # URLs with path/query syntax but no scheme are treated as URLs.
        if "/" in value or "?" in value or "#" in value:
            return IOCType.URL

        return IOCType.DOMAIN

    @staticmethod
    def _validate_ioc_for_type(
        ioc_value: str,
        ioc_type: IOCType,
    ) -> None:
        """
        Perform cheap local IOC validation before provider execution.
        """

        if ioc_type == IOCType.IP:
            if not BaseThreatProvider.validate_ip(ioc_value):
                raise ValueError(
                    f"Invalid IP address: {ioc_value!r}"
                )

        elif ioc_type == IOCType.HASH_SHA256:
            if not BaseThreatProvider.is_sha256(ioc_value):
                raise ValueError(
                    "Invalid SHA-256 IOC."
                )

        elif ioc_type == IOCType.URL:
            if len(ioc_value) > MAX_IOC_LENGTH:
                raise ValueError(
                    "URL exceeds maximum supported length."
                )

        elif ioc_type == IOCType.DOMAIN:
            if len(ioc_value) > MAX_IOC_LENGTH:
                raise ValueError(
                    "Domain exceeds maximum supported length."
                )

        elif ioc_type == IOCType.HASH_MD5:
            if len(ioc_value) != 32:
                raise ValueError(
                    "Invalid MD5 IOC."
                )

    # =========================================================================
    # CACHE KEY
    # =========================================================================

    @staticmethod
    def _build_cache_key(
        ioc_value: str,
        ioc_type: IOCType,
    ) -> str:
        """
        Build a bounded deterministic cache key.

        The raw IOC is never placed into the cache key itself.
        """

        canonical = (
            f"{ioc_type.value}:"
            f"{ioc_value.strip().lower()}"
        )

        digest = hashlib.sha256(
            canonical.encode(
                "utf-8",
                errors="strict",
            )
        ).hexdigest()

        return (
            f"{CACHE_KEY_PREFIX}:"
            f"{digest}"
        )

    # =========================================================================
    # SAFE RESULT COPYING
    # =========================================================================

    @staticmethod
    def _clone_summary(
        summary: AggregatedThreatSummary,
        cache_hit: bool,
    ) -> AggregatedThreatSummary:
        """
        Return an independent Pydantic model copy.

        This prevents callers from mutating cached objects.
        """

        try:
            cloned = summary.model_copy(
                deep=True
            )
        except AttributeError:
            # Compatibility with older Pydantic releases.
            cloned = copy.deepcopy(summary)

        cloned.cache_hit = cache_hit

        return cloned

    # =========================================================================
    # REPRESENTATION / DIAGNOSTICS
    # =========================================================================

    def registered_provider_names(self) -> List[str]:
        """
        Return deterministic list of registered provider names.
        """

        return list(self.providers.keys())

    def provider_capabilities(self) -> Dict[str, List[str]]:
        """
        Return provider -> supported IOC type mapping.
        """

        return {
            name: sorted(
                ioc_type.value
                for ioc_type in provider.supported_ioc_types
            )
            for name, provider in self.providers.items()
        }

    def provider_weights_snapshot(self) -> Dict[str, float]:
        """
        Return a defensive copy of configured provider weights.
        """

        return dict(self.provider_weights)
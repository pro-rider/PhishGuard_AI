"""
PhishGuard AI - Threat Intelligence Resilience & Fault Tolerance Engine
======================================================================

Version: 14.0.0
File Path: threat_intel/resilience.py

Production resilience primitives for Threat Intelligence providers.

Design goals
------------
- Prevent cascading failures.
- Bound retry amplification.
- Prevent retry storms.
- Protect providers with circuit breakers.
- Restrict HALF_OPEN probe concurrency.
- Enforce operation and attempt deadlines.
- Support Retry-After from rate-limited providers.
- Provide bulkhead/concurrency protection.
- Preserve cancellation semantics.
- Avoid leaking secrets through logs/fallback payloads.
- Provide deterministic testing hooks.
- Remain compatible with the previous v13 public API.

Patterns implemented
--------------------
1. Circuit Breaker
   CLOSED -> OPEN -> HALF_OPEN -> CLOSED

2. Exponential Backoff + Full Jitter

3. Retry-After aware rate-limit handling

4. Global operation deadline

5. Per-attempt timeout

6. Bulkhead / concurrency isolation

7. Failure classification

8. Slow-call detection

9. Half-open probe limiting

10. Safe degraded fallback

This module intentionally contains no network I/O.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from threat_intel.base import (
    IOCType,
    NormalizedThreatReport,
    ProviderTimeoutError,
    RateLimitExceededError,
    ThreatIntelAPIError,
    ThreatIntelException,
    ThreatSeverity,
)

logger = logging.getLogger("phishguard.threat_intel.resilience")

T = TypeVar("T")

ExceptionTuple = Tuple[Type[BaseException], ...]
AsyncCallable = Callable[[], Awaitable[T]]
SleepCallable = Callable[[float], Awaitable[None]]
RandomCallable = Callable[[float, float], float]


# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_FAILURE_THRESHOLD = 5
DEFAULT_RECOVERY_TIMEOUT_SECONDS = 60.0
DEFAULT_SUCCESS_THRESHOLD_HALF_OPEN = 2

DEFAULT_MAX_RETRIES = 2
DEFAULT_BASE_BACKOFF_SECONDS = 0.5
DEFAULT_MAX_BACKOFF_SECONDS = 5.0

DEFAULT_OPERATION_TIMEOUT_SECONDS = 15.0
DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 5.0

DEFAULT_MAX_CONCURRENT_CALLS = 32
DEFAULT_HALF_OPEN_MAX_PROBES = 1

DEFAULT_SLOW_CALL_THRESHOLD_SECONDS = 5.0

MAX_PROVIDER_NAME_LENGTH = 128
MAX_IOC_LOG_LENGTH = 256


# ==============================================================================
# CIRCUIT STATE
# ==============================================================================

class CircuitState(str, enum.Enum):
    """Operational state of a circuit breaker."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


# ==============================================================================
# FAILURE CLASSIFICATION
# ==============================================================================

class FailureKind(str, enum.Enum):
    """
    Classification used by the resilience layer.

    TRANSIENT
        Usually safe to retry.

    RATE_LIMITED
        Retryable, but should honor Retry-After when available.

    PERMANENT
        Should normally not be retried.

    CANCELLATION
        Task cancellation. Never swallow or retry.

    UNKNOWN
        Conservative classification; treated as non-retryable by default.
    """

    TRANSIENT = "TRANSIENT"
    RATE_LIMITED = "RATE_LIMITED"
    PERMANENT = "PERMANENT"
    CANCELLATION = "CANCELLATION"
    UNKNOWN = "UNKNOWN"


def classify_exception(exc: BaseException) -> FailureKind:
    """
    Classify provider exceptions for retry/circuit-breaker decisions.

    Important:
        asyncio.CancelledError must never be swallowed as an ordinary
        provider failure.
    """

    if isinstance(exc, asyncio.CancelledError):
        return FailureKind.CANCELLATION

    if isinstance(exc, RateLimitExceededError):
        return FailureKind.RATE_LIMITED

    if isinstance(
        exc,
        (
            ProviderTimeoutError,
            TimeoutError,
            asyncio.TimeoutError,
            ConnectionError,
        ),
    ):
        return FailureKind.TRANSIENT

    if isinstance(exc, ThreatIntelAPIError):
        # HTTP 408/425/429 and 5xx are generally transient.
        status = getattr(exc, "status_code", None)

        if status == 429:
            return FailureKind.RATE_LIMITED

        if status in {
            408,
            425,
            500,
            502,
            503,
            504,
        }:
            return FailureKind.TRANSIENT

        return FailureKind.PERMANENT

    return FailureKind.UNKNOWN


def is_retryable_exception(exc: BaseException) -> bool:
    """Return whether an exception is considered retryable."""

    return classify_exception(exc) in {
        FailureKind.TRANSIENT,
        FailureKind.RATE_LIMITED,
    }


# ==============================================================================
# CIRCUIT BREAKER EXCEPTION
# ==============================================================================

class CircuitBreakerOpenException(ThreatIntelException):
    """
    Raised when execution is rejected by the circuit breaker.
    """

    def __init__(
        self,
        provider_name: str,
        reset_remaining_seconds: float,
    ) -> None:
        self.provider_name = provider_name
        self.reset_remaining_seconds = max(0.0, float(reset_remaining_seconds))

        super().__init__(
            f"[{provider_name}] Circuit breaker is OPEN. "
            f"Fast-failing query. "
            f"Reset in {self.reset_remaining_seconds:.1f}s."
        )


class BulkheadRejectedException(ThreatIntelException):
    """
    Raised when the provider concurrency limit is exhausted.
    """

    def __init__(
        self,
        provider_name: str,
        max_concurrent_calls: int,
    ) -> None:
        self.provider_name = provider_name
        self.max_concurrent_calls = max_concurrent_calls

        super().__init__(
            f"[{provider_name}] Bulkhead rejected execution. "
            f"Maximum concurrent calls: {max_concurrent_calls}."
        )


class RetryBudgetExceededException(ThreatIntelException):
    """
    Raised when a retry operation cannot continue because its deadline
    has been exhausted.
    """

    def __init__(
        self,
        provider_name: str,
    ) -> None:
        self.provider_name = provider_name

        super().__init__(
            f"[{provider_name}] Retry operation deadline exhausted."
        )


# ==============================================================================
# CIRCUIT SNAPSHOT
# ==============================================================================

@dataclass(frozen=True, slots=True)
class CircuitSnapshot:
    """
    Immutable diagnostic snapshot of circuit-breaker state.

    Intended for metrics, health endpoints, and debugging.
    """

    provider_name: str
    state: CircuitState
    consecutive_failures: int
    consecutive_successes_half_open: int
    total_calls: int
    total_successes: int
    total_failures: int
    total_rejections: int
    total_slow_calls: int
    half_open_in_flight: int
    failure_rate: float
    slow_call_rate: float
    open_for_seconds: float


# ==============================================================================
# CIRCUIT BREAKER
# ==============================================================================

class CircuitBreaker:
    """
    Async-safe circuit breaker with bounded HALF_OPEN probing.

    Important improvements over the previous implementation:

    - HALF_OPEN no longer allows unlimited concurrent probes.
    - State changes happen under one lock.
    - Metrics are maintained atomically.
    - Optional sliding outcome window is supported.
    - Slow calls can count as failures.
    - Manual reset and force-open are supported.
    - Cancellation is not automatically counted as a provider failure.
    """

    def __init__(
        self,
        provider_name: str,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        recovery_timeout_seconds: float = DEFAULT_RECOVERY_TIMEOUT_SECONDS,
        success_threshold_half_open: int = DEFAULT_SUCCESS_THRESHOLD_HALF_OPEN,
        *,
        half_open_max_calls: int = DEFAULT_HALF_OPEN_MAX_PROBES,
        failure_window_size: Optional[int] = None,
        failure_rate_threshold: Optional[float] = None,
        slow_call_threshold_seconds: Optional[float] = None,
        slow_call_counts_as_failure: bool = False,
    ) -> None:

        if not provider_name or not provider_name.strip():
            raise ValueError("provider_name must be non-empty.")

        if len(provider_name) > MAX_PROVIDER_NAME_LENGTH:
            raise ValueError(
                f"provider_name exceeds {MAX_PROVIDER_NAME_LENGTH} characters."
            )

        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1.")

        if recovery_timeout_seconds <= 0:
            raise ValueError(
                "recovery_timeout_seconds must be > 0."
            )

        if success_threshold_half_open < 1:
            raise ValueError(
                "success_threshold_half_open must be >= 1."
            )

        if half_open_max_calls < 1:
            raise ValueError(
                "half_open_max_calls must be >= 1."
            )

        if failure_window_size is not None and failure_window_size < 1:
            raise ValueError(
                "failure_window_size must be >= 1 when configured."
            )

        if failure_rate_threshold is not None and not (
            0.0 < failure_rate_threshold <= 1.0
        ):
            raise ValueError(
                "failure_rate_threshold must be in (0, 1]."
            )

        if slow_call_threshold_seconds is not None and (
            slow_call_threshold_seconds <= 0
        ):
            raise ValueError(
                "slow_call_threshold_seconds must be > 0."
            )

        self.provider_name = provider_name.strip()
        self.failure_threshold = int(failure_threshold)
        self.recovery_timeout_seconds = float(
            recovery_timeout_seconds
        )
        self.success_threshold_half_open = int(
            success_threshold_half_open
        )
        self.half_open_max_calls = int(half_open_max_calls)

        self.failure_window_size = failure_window_size
        self.failure_rate_threshold = failure_rate_threshold

        self.slow_call_threshold_seconds = (
            float(slow_call_threshold_seconds)
            if slow_call_threshold_seconds is not None
            else None
        )
        self.slow_call_counts_as_failure = bool(
            slow_call_counts_as_failure
        )

        self._state = CircuitState.CLOSED

        self._consecutive_failures = 0
        self._consecutive_successes_half_open = 0

        self._opened_at: Optional[float] = None

        self._half_open_in_flight = 0

        self._total_calls = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_rejections = 0
        self._total_slow_calls = 0

        self._outcomes: Deque[bool] = deque(
            maxlen=failure_window_size or 1
        )

        self._lock = asyncio.Lock()

    # --------------------------------------------------------------------------
    # STATE
    # --------------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """
        Read-only current state.

        This property intentionally performs no state mutation.

        State transitions requiring synchronization are handled by
        allow_execution().
        """

        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self._state == CircuitState.HALF_OPEN

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    # --------------------------------------------------------------------------
    # PERMISSION ACQUISITION
    # --------------------------------------------------------------------------

    async def allow_execution(self) -> bool:
        """
        Acquire permission to execute one provider call.

        In HALF_OPEN state only a bounded number of probes are permitted.

        Returns:
            True when execution is allowed.

        Raises:
            CircuitBreakerOpenException
            when the circuit is OPEN or another HALF_OPEN probe is active.
        """

        async with self._lock:
            now = time.monotonic()

            # CLOSED -----------------------------------------------------------
            if self._state == CircuitState.CLOSED:
                self._total_calls += 1
                return True

            # OPEN -------------------------------------------------------------
            if self._state == CircuitState.OPEN:
                assert self._opened_at is not None

                elapsed = now - self._opened_at
                remaining = self.recovery_timeout_seconds - elapsed

                if remaining > 0:
                    self._total_rejections += 1

                    raise CircuitBreakerOpenException(
                        self.provider_name,
                        remaining,
                    )

                # Cooldown complete -> HALF_OPEN.
                self._state = CircuitState.HALF_OPEN
                self._consecutive_successes_half_open = 0
                self._half_open_in_flight = 0

                logger.info(
                    "[%s] Circuit transition OPEN -> HALF_OPEN.",
                    self.provider_name,
                )

            # HALF_OPEN --------------------------------------------------------
            if self._state == CircuitState.HALF_OPEN:

                if (
                    self._half_open_in_flight
                    >= self.half_open_max_calls
                ):
                    self._total_rejections += 1

                    raise CircuitBreakerOpenException(
                        self.provider_name,
                        0.0,
                    )

                self._half_open_in_flight += 1
                self._total_calls += 1

                return True

            raise RuntimeError(
                f"Unknown circuit state: {self._state!r}"
            )

    # --------------------------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------------------------

    async def record_success(
        self,
        *,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """
        Record a successful provider call.

        Slow calls can optionally count as failures.
        """

        async with self._lock:

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_in_flight = max(
                    0,
                    self._half_open_in_flight - 1,
                )

            is_slow = self._is_slow_call(duration_seconds)

            if is_slow:
                self._total_slow_calls += 1

            if (
                is_slow
                and self.slow_call_counts_as_failure
            ):
                self._record_failure_locked(
                    count_toward_circuit=True
                )
                return

            self._total_successes += 1
            self._consecutive_failures = 0

            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes_half_open += 1

                if (
                    self._consecutive_successes_half_open
                    >= self.success_threshold_half_open
                    and self._half_open_in_flight == 0
                ):
                    self._state = CircuitState.CLOSED
                    self._opened_at = None
                    self._consecutive_failures = 0
                    self._consecutive_successes_half_open = 0

                    self._outcomes.clear()

                    logger.info(
                        "[%s] Circuit restored HALF_OPEN -> CLOSED.",
                        self.provider_name,
                    )

            elif self._state == CircuitState.CLOSED:
                self._append_outcome_locked(True)

    # --------------------------------------------------------------------------
    # FAILURE
    # --------------------------------------------------------------------------

    async def record_failure(
        self,
        error: BaseException,
        *,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """
        Record a provider failure.

        Cancellation is deliberately ignored because task cancellation is
        control flow, not evidence that the provider is unhealthy.
        """

        if isinstance(error, asyncio.CancelledError):
            return

        async with self._lock:

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_in_flight = max(
                    0,
                    self._half_open_in_flight - 1,
                )

            is_slow = self._is_slow_call(duration_seconds)

            if is_slow:
                self._total_slow_calls += 1

            self._record_failure_locked(
                count_toward_circuit=True
            )

            logger.warning(
                "[%s] Provider failure recorded: type=%s state=%s",
                self.provider_name,
                type(error).__name__,
                self._state.value,
            )

    def _record_failure_locked(
        self,
        *,
        count_toward_circuit: bool,
    ) -> None:

        self._total_failures += 1

        if not count_toward_circuit:
            return

        self._consecutive_failures += 1

        self._append_outcome_locked(False)

        # HALF_OPEN failure immediately reopens the circuit.
        if self._state == CircuitState.HALF_OPEN:
            self._transition_to_open_locked()
            return

        # Consecutive failure threshold.
        if (
            self._state == CircuitState.CLOSED
            and self._consecutive_failures
            >= self.failure_threshold
        ):
            self._transition_to_open_locked()
            return

        # Optional failure-rate window.
        if (
            self._state == CircuitState.CLOSED
            and self.failure_window_size
            and self.failure_rate_threshold is not None
        ):
            if len(self._outcomes) >= self.failure_window_size:
                failures = sum(
                    not outcome
                    for outcome in self._outcomes
                )

                failure_rate = (
                    failures / len(self._outcomes)
                )

                if failure_rate >= self.failure_rate_threshold:
                    self._transition_to_open_locked()

    # --------------------------------------------------------------------------
    # STATE TRANSITION
    # --------------------------------------------------------------------------

    def _transition_to_open_locked(self) -> None:
        if self._state == CircuitState.OPEN:
            return

        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._consecutive_successes_half_open = 0
        self._half_open_in_flight = 0

        logger.error(
            "[%s] Circuit transition -> OPEN.",
            self.provider_name,
        )

    # --------------------------------------------------------------------------
    # MANUAL CONTROLS
    # --------------------------------------------------------------------------

    async def reset(self) -> None:
        """
        Manually reset the breaker to CLOSED.
        """

        async with self._lock:
            previous = self._state

            self._state = CircuitState.CLOSED
            self._opened_at = None
            self._consecutive_failures = 0
            self._consecutive_successes_half_open = 0
            self._half_open_in_flight = 0
            self._outcomes.clear()

            logger.warning(
                "[%s] Circuit manually reset: %s -> CLOSED.",
                self.provider_name,
                previous.value,
            )

    async def force_open(self) -> None:
        """
        Force the breaker OPEN.

        Useful for operational kill-switches.
        """

        async with self._lock:
            self._transition_to_open_locked()

    # --------------------------------------------------------------------------
    # METRICS
    # --------------------------------------------------------------------------

    async def snapshot(self) -> CircuitSnapshot:
        """
        Return an immutable circuit metrics snapshot.
        """

        async with self._lock:
            total_recorded = (
                self._total_successes
                + self._total_failures
            )

            failure_rate = (
                self._total_failures / total_recorded
                if total_recorded
                else 0.0
            )

            slow_rate = (
                self._total_slow_calls / self._total_calls
                if self._total_calls
                else 0.0
            )

            open_for = 0.0

            if self._opened_at is not None:
                open_for = max(
                    0.0,
                    time.monotonic() - self._opened_at,
                )

            return CircuitSnapshot(
                provider_name=self.provider_name,
                state=self._state,
                consecutive_failures=self._consecutive_failures,
                consecutive_successes_half_open=(
                    self._consecutive_successes_half_open
                ),
                total_calls=self._total_calls,
                total_successes=self._total_successes,
                total_failures=self._total_failures,
                total_rejections=self._total_rejections,
                total_slow_calls=self._total_slow_calls,
                half_open_in_flight=self._half_open_in_flight,
                failure_rate=failure_rate,
                slow_call_rate=slow_rate,
                open_for_seconds=open_for,
            )

    # --------------------------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------------------------

    def _append_outcome_locked(self, success: bool) -> None:
        if self.failure_window_size:
            self._outcomes.append(success)

    def _is_slow_call(
        self,
        duration_seconds: Optional[float],
    ) -> bool:
        if (
            duration_seconds is None
            or self.slow_call_threshold_seconds is None
        ):
            return False

        return duration_seconds >= self.slow_call_threshold_seconds


# ==============================================================================
# BULKHEAD / CONCURRENCY LIMITER
# ==============================================================================

class ProviderBulkhead:
    """
    Bounded async concurrency limiter.

    Prevents a single degraded provider from consuming all application
    concurrency.

    This is intentionally fail-fast instead of maintaining an unbounded wait
    queue.
    """

    def __init__(
        self,
        provider_name: str,
        max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS,
    ) -> None:

        if not provider_name or not provider_name.strip():
            raise ValueError("provider_name must be non-empty.")

        if max_concurrent_calls < 1:
            raise ValueError(
                "max_concurrent_calls must be >= 1."
            )

        self.provider_name = provider_name.strip()
        self.max_concurrent_calls = int(
            max_concurrent_calls
        )

        self._semaphore = asyncio.BoundedSemaphore(
            self.max_concurrent_calls
        )

        self._in_flight = 0
        self._total_acquired = 0
        self._total_rejected = 0

        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        """
        Non-blocking best-effort acquisition.

        Returns:
            True if capacity was acquired.
        """

        # asyncio.Semaphore has no public non-blocking acquire.
        # We inspect its private counter only inside this tiny abstraction.
        # This keeps the public interface deterministic and fail-fast.
        async with self._lock:

            value = getattr(
                self._semaphore,
                "_value",
                0,
            )

            if value <= 0:
                self._total_rejected += 1
                return False

            await self._semaphore.acquire()

            self._in_flight += 1
            self._total_acquired += 1

            return True

    async def acquire(self) -> None:
        """
        Fail-fast acquisition.
        """

        acquired = await self.try_acquire()

        if not acquired:
            raise BulkheadRejectedException(
                self.provider_name,
                self.max_concurrent_calls,
            )

    async def release(self) -> None:
        """
        Release one concurrency slot.
        """

        async with self._lock:

            if self._in_flight <= 0:
                return

            self._in_flight -= 1
            self._semaphore.release()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def available_capacity(self) -> int:
        return max(
            0,
            self.max_concurrent_calls - self._in_flight,
        )

    @property
    def total_rejected(self) -> int:
        return self._total_rejected


# ==============================================================================
# RETRY CONFIGURATION
# ==============================================================================

@dataclass(frozen=True, slots=True)
class RetryConfig:
    """
    Immutable retry policy.

    max_retries means retries AFTER the first attempt.

    Example:
        max_retries=2

        attempt 1
        retry
        attempt 2
        retry
        attempt 3
        fail
    """

    max_retries: int = DEFAULT_MAX_RETRIES
    base_backoff_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS
    max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS

    operation_timeout_seconds: Optional[
        float
    ] = DEFAULT_OPERATION_TIMEOUT_SECONDS

    attempt_timeout_seconds: Optional[
        float
    ] = DEFAULT_ATTEMPT_TIMEOUT_SECONDS

    retryable_exceptions: ExceptionTuple = (
        ProviderTimeoutError,
        RateLimitExceededError,
        TimeoutError,
        asyncio.TimeoutError,
        ConnectionError,
    )

    def __post_init__(self) -> None:

        if self.max_retries < 0:
            raise ValueError(
                "max_retries must be >= 0."
            )

        if self.base_backoff_seconds < 0:
            raise ValueError(
                "base_backoff_seconds must be >= 0."
            )

        if self.max_backoff_seconds < 0:
            raise ValueError(
                "max_backoff_seconds must be >= 0."
            )

        if (
            self.max_backoff_seconds
            < self.base_backoff_seconds
        ):
            raise ValueError(
                "max_backoff_seconds must be >= "
                "base_backoff_seconds."
            )

        if (
            self.operation_timeout_seconds is not None
            and self.operation_timeout_seconds <= 0
        ):
            raise ValueError(
                "operation_timeout_seconds must be > 0."
            )

        if (
            self.attempt_timeout_seconds is not None
            and self.attempt_timeout_seconds <= 0
        ):
            raise ValueError(
                "attempt_timeout_seconds must be > 0."
            )


# ==============================================================================
# RETRY-AFTER PARSING
# ==============================================================================

def _get_retry_after_seconds(
    exc: BaseException,
) -> Optional[float]:
    """
    Extract Retry-After from RateLimitExceededError.

    The existing exception stores retry_after as seconds.
    """

    if not isinstance(exc, RateLimitExceededError):
        return None

    retry_after = getattr(
        exc,
        "retry_after",
        None,
    )

    if retry_after is None:
        return None

    try:
        value = float(retry_after)

        if math.isfinite(value) and value >= 0:
            return value

    except (TypeError, ValueError):
        pass

    return None


def calculate_backoff(
    *,
    attempt: int,
    base_backoff_seconds: float,
    max_backoff_seconds: float,
    retry_after_seconds: Optional[float] = None,
    random_func: RandomCallable = random.uniform,
) -> float:
    """
    Calculate bounded exponential full-jitter delay.

    Formula:

        cap = min(
            max_backoff,
            base_backoff * 2**attempt
        )

        delay = random(0, cap)

    If Retry-After is supplied, it becomes the minimum delay.

    The result is always bounded by max_backoff.
    """

    if attempt < 0:
        raise ValueError("attempt must be >= 0.")

    cap = min(
        max_backoff_seconds,
        base_backoff_seconds * (2 ** attempt),
    )

    jittered = (
        random_func(0.0, cap)
        if cap > 0
        else 0.0
    )

    if retry_after_seconds is not None:
        retry_after_seconds = max(
            0.0,
            float(retry_after_seconds),
        )

        jittered = max(
            jittered,
            retry_after_seconds,
        )

    return min(
        max_backoff_seconds,
        max(0.0, jittered),
    )


# ==============================================================================
# ASYNC RETRY ENGINE
# ==============================================================================

async def async_retry_with_backoff(
    coro_func: Callable[[], Awaitable[T]],
    provider_name: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
    max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
    retryable_exceptions: ExceptionTuple = (
        ProviderTimeoutError,
        RateLimitExceededError,
        TimeoutError,
        asyncio.TimeoutError,
        ConnectionError,
    ),
    *,
    operation_timeout_seconds: Optional[
        float
    ] = DEFAULT_OPERATION_TIMEOUT_SECONDS,
    attempt_timeout_seconds: Optional[
        float
    ] = DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
    sleep_func: SleepCallable = asyncio.sleep,
    random_func: RandomCallable = random.uniform,
) -> T:
    """
    Execute an async operation with bounded retries.

    Security/reliability properties
    --------------------------------
    - Never retries CancelledError.
    - Never retries arbitrary exceptions by default.
    - Honors RateLimitExceededError.retry_after.
    - Enforces total operation deadline.
    - Optionally enforces per-attempt timeout.
    - Uses full jitter.
    - Never sleeps beyond the remaining operation deadline.

    This avoids the retry amplification problem where multiple layers
    independently retry the same failed request.
    """

    if not provider_name or not provider_name.strip():
        raise ValueError(
            "provider_name must be non-empty."
        )

    if max_retries < 0:
        raise ValueError(
            "max_retries must be >= 0."
        )

    if base_backoff_seconds < 0:
        raise ValueError(
            "base_backoff_seconds must be >= 0."
        )

    if max_backoff_seconds < base_backoff_seconds:
        raise ValueError(
            "max_backoff_seconds must be >= "
            "base_backoff_seconds."
        )

    if (
        operation_timeout_seconds is not None
        and operation_timeout_seconds <= 0
    ):
        raise ValueError(
            "operation_timeout_seconds must be > 0."
        )

    if (
        attempt_timeout_seconds is not None
        and attempt_timeout_seconds <= 0
    ):
        raise ValueError(
            "attempt_timeout_seconds must be > 0."
        )

    started_at = time.monotonic()

    def remaining_budget() -> Optional[float]:
        if operation_timeout_seconds is None:
            return None

        return (
            operation_timeout_seconds
            - (time.monotonic() - started_at)
        )

    attempt = 0

    while True:

        remaining = remaining_budget()

        if (
            remaining is not None
            and remaining <= 0
        ):
            raise RetryBudgetExceededException(
                provider_name
            )

        try:

            if attempt_timeout_seconds is not None:

                effective_timeout = (
                    attempt_timeout_seconds
                )

                if remaining is not None:
                    effective_timeout = min(
                        effective_timeout,
                        remaining,
                    )

                if effective_timeout <= 0:
                    raise RetryBudgetExceededException(
                        provider_name
                    )

                return await asyncio.wait_for(
                    coro_func(),
                    timeout=effective_timeout,
                )

            return await coro_func()

        except asyncio.CancelledError:
            # Never swallow cancellation.
            raise

        except retryable_exceptions as exc:

            attempt += 1

            if attempt > max_retries:
                logger.warning(
                    "[%s] Retry budget exhausted after %d retries.",
                    provider_name,
                    max_retries,
                )
                raise

            remaining = remaining_budget()

            if (
                remaining is not None
                and remaining <= 0
            ):
                raise RetryBudgetExceededException(
                    provider_name
                ) from exc

            retry_after = _get_retry_after_seconds(exc)

            delay = calculate_backoff(
                attempt=attempt - 1,
                base_backoff_seconds=base_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
                retry_after_seconds=retry_after,
                random_func=random_func,
            )

            if remaining is not None:
                delay = min(
                    delay,
                    max(0.0, remaining),
                )

            logger.warning(
                "[%s] Retry %d/%d after %s; "
                "delay=%.3fs.",
                provider_name,
                attempt,
                max_retries,
                type(exc).__name__,
                delay,
            )

            if delay > 0:
                await sleep_func(delay)

        except Exception:
            # Permanent/unknown failures propagate immediately.
            raise


# ==============================================================================
# FALLBACK REPORT
# ==============================================================================

def build_degraded_fallback_report(
    *,
    provider_name: str,
    ioc_value: str,
    ioc_type: IOCType,
    elapsed_ms: float,
    error: Optional[BaseException] = None,
) -> NormalizedThreatReport:
    """
    Build a safe degraded report.

    Important:
        A provider failure must NEVER be interpreted as CLEAN.

    Therefore:
        risk_score = 0
        severity = UNKNOWN
        is_malicious = False

    UNKNOWN means "not enough evidence", not "safe".
    """

    error_type = (
        type(error).__name__
        if error is not None
        else "unknown"
    )

    return NormalizedThreatReport(
        provider_name=provider_name,
        ioc_value=ioc_value,
        ioc_type=ioc_type,
        risk_score=0.0,
        severity=ThreatSeverity.UNKNOWN,
        is_malicious=False,
        positive_detections=0,
        total_engines=0,
        threat_categories=[],
        raw_tags=[
            "degraded_fallback",
            f"error_type:{error_type}",
        ],
        response_time_ms=max(
            0.0,
            float(elapsed_ms),
        ),
        cached_response=False,
        raw_response={
            "fallback_triggered": True,
            "error_type": error_type,
            "provider_available": False,
        },
    )


# ==============================================================================
# RESILIENT PROVIDER EXECUTION
# ==============================================================================

async def execute_with_resilience(
    query_coro_func: Callable[
        [],
        Awaitable[NormalizedThreatReport],
    ],
    provider_name: str,
    circuit_breaker: CircuitBreaker,
    ioc_value: str,
    ioc_type: IOCType,
    max_retries: int = DEFAULT_MAX_RETRIES,
    fallback_on_error: bool = True,
    *,
    retry_config: Optional[RetryConfig] = None,
    bulkhead: Optional[ProviderBulkhead] = None,
) -> NormalizedThreatReport:
    """
    Execute a threat-intelligence query through the complete resilience stack.

    Execution order
    ---------------
    1. Circuit breaker permission
    2. Bulkhead permission
    3. Bounded retry engine
    4. Success/failure recording
    5. Safe degraded fallback

    Important:
        Circuit breaker permission is acquired before retries, so one logical
        query consumes one circuit permission instead of making every retry
        look like an independent logical request.

    Args:
        query_coro_func:
            Zero-argument coroutine factory.

        provider_name:
            Provider identifier.

        circuit_breaker:
            Provider-specific CircuitBreaker.

        ioc_value:
            IOC being queried.

        ioc_type:
            IOC type.

        max_retries:
            Backward-compatible retry count.

        fallback_on_error:
            Whether to return UNKNOWN degraded result.

        retry_config:
            Optional complete retry policy.

        bulkhead:
            Optional provider-specific concurrency limiter.

    Returns:
        NormalizedThreatReport.

    Raises:
        Original exception when fallback_on_error=False.
    """

    if retry_config is None:
        retry_config = RetryConfig(
            max_retries=max_retries
        )

    started_at = time.monotonic()

    circuit_permission_acquired = False
    bulkhead_acquired = False

    try:

        # ----------------------------------------------------------------------
        # 1. CIRCUIT BREAKER
        # ----------------------------------------------------------------------

        await circuit_breaker.allow_execution()
        circuit_permission_acquired = True

        # ----------------------------------------------------------------------
        # 2. BULKHEAD
        # ----------------------------------------------------------------------

        if bulkhead is not None:
            await bulkhead.acquire()
            bulkhead_acquired = True

        # ----------------------------------------------------------------------
        # 3. RETRY
        # ----------------------------------------------------------------------

        report = await async_retry_with_backoff(
            coro_func=query_coro_func,
            provider_name=provider_name,
            max_retries=retry_config.max_retries,
            base_backoff_seconds=(
                retry_config.base_backoff_seconds
            ),
            max_backoff_seconds=(
                retry_config.max_backoff_seconds
            ),
            retryable_exceptions=(
                retry_config.retryable_exceptions
            ),
            operation_timeout_seconds=(
                retry_config.operation_timeout_seconds
            ),
            attempt_timeout_seconds=(
                retry_config.attempt_timeout_seconds
            ),
        )

        # ----------------------------------------------------------------------
        # 4. SUCCESS
        # ----------------------------------------------------------------------

        duration = time.monotonic() - started_at

        await circuit_breaker.record_success(
            duration_seconds=duration
        )

        return report

    except asyncio.CancelledError:
        # Cancellation must propagate.
        raise

    except Exception as exc:

        duration = time.monotonic() - started_at

        # ----------------------------------------------------------------------
        # 5. FAILURE
        # ----------------------------------------------------------------------

        if circuit_permission_acquired:
            await circuit_breaker.record_failure(
                exc,
                duration_seconds=duration,
            )

        if not fallback_on_error:
            raise

        elapsed_ms = duration * 1000.0

        # Do not log the raw IOC or exception message here.
        # IOC values may contain sensitive data and exception strings may
        # contain credentials returned by upstream HTTP clients.
        logger.error(
            "[%s] Threat-intelligence execution degraded. "
            "error_type=%s",
            provider_name,
            type(exc).__name__,
        )

        return build_degraded_fallback_report(
            provider_name=provider_name,
            ioc_value=ioc_value,
            ioc_type=ioc_type,
            elapsed_ms=elapsed_ms,
            error=exc,
        )

    finally:

        if bulkhead_acquired and bulkhead is not None:
            await bulkhead.release()


# ==============================================================================
# RESILIENCE POLICY FACTORY
# ==============================================================================

@dataclass(frozen=True, slots=True)
class ResiliencePolicy:
    """
    Combined provider resilience configuration.

    This is useful for configuration-driven provider registration.
    """

    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    recovery_timeout_seconds: float = (
        DEFAULT_RECOVERY_TIMEOUT_SECONDS
    )
    success_threshold_half_open: int = (
        DEFAULT_SUCCESS_THRESHOLD_HALF_OPEN
    )
    half_open_max_calls: int = (
        DEFAULT_HALF_OPEN_MAX_PROBES
    )

    max_concurrent_calls: int = (
        DEFAULT_MAX_CONCURRENT_CALLS
    )

    max_retries: int = DEFAULT_MAX_RETRIES
    base_backoff_seconds: float = (
        DEFAULT_BASE_BACKOFF_SECONDS
    )
    max_backoff_seconds: float = (
        DEFAULT_MAX_BACKOFF_SECONDS
    )

    operation_timeout_seconds: Optional[
        float
    ] = DEFAULT_OPERATION_TIMEOUT_SECONDS

    attempt_timeout_seconds: Optional[
        float
    ] = DEFAULT_ATTEMPT_TIMEOUT_SECONDS

    slow_call_threshold_seconds: Optional[
        float
    ] = DEFAULT_SLOW_CALL_THRESHOLD_SECONDS

    slow_call_counts_as_failure: bool = False

    failure_window_size: Optional[int] = None
    failure_rate_threshold: Optional[float] = None

    def create_circuit_breaker(
        self,
        provider_name: str,
    ) -> CircuitBreaker:

        return CircuitBreaker(
            provider_name=provider_name,
            failure_threshold=self.failure_threshold,
            recovery_timeout_seconds=(
                self.recovery_timeout_seconds
            ),
            success_threshold_half_open=(
                self.success_threshold_half_open
            ),
            half_open_max_calls=(
                self.half_open_max_calls
            ),
            failure_window_size=(
                self.failure_window_size
            ),
            failure_rate_threshold=(
                self.failure_rate_threshold
            ),
            slow_call_threshold_seconds=(
                self.slow_call_threshold_seconds
            ),
            slow_call_counts_as_failure=(
                self.slow_call_counts_as_failure
            ),
        )

    def create_bulkhead(
        self,
        provider_name: str,
    ) -> ProviderBulkhead:

        return ProviderBulkhead(
            provider_name=provider_name,
            max_concurrent_calls=(
                self.max_concurrent_calls
            ),
        )

    def create_retry_config(self) -> RetryConfig:

        return RetryConfig(
            max_retries=self.max_retries,
            base_backoff_seconds=(
                self.base_backoff_seconds
            ),
            max_backoff_seconds=(
                self.max_backoff_seconds
            ),
            operation_timeout_seconds=(
                self.operation_timeout_seconds
            ),
            attempt_timeout_seconds=(
                self.attempt_timeout_seconds
            ),
        )


# ==============================================================================
# PUBLIC EXPORTS
# ==============================================================================

__all__ = [
    "CircuitState",
    "FailureKind",
    "CircuitBreakerOpenException",
    "BulkheadRejectedException",
    "RetryBudgetExceededException",
    "CircuitSnapshot",
    "CircuitBreaker",
    "ProviderBulkhead",
    "RetryConfig",
    "ResiliencePolicy",
    "classify_exception",
    "is_retryable_exception",
    "calculate_backoff",
    "async_retry_with_backoff",
    "build_degraded_fallback_report",
    "execute_with_resilience",
]
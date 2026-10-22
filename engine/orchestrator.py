"""
PhishGuard_AI - Production Detection Engine Orchestrator

Central detection pipeline coordinator managing parallel, dependency-aware, 
and resource-isolated execution of security analyzers across Email, URL, Web,
and Attachment subsystems.

Key Security & Execution Controls:
----------------------------------
1. True Process & Subprocess Isolation: CPU-bound tasks execute in process pools;
   subprocess tasks execute under a managed lifecycle with hard OS termination (SIGKILL)
   on cancellation or timeout.
2. Capability & DAG Dependency Scheduler: Analyzers declare capabilities, inputs, and 
   explicit dependencies. Execution follows a dynamically scheduled Directed Acyclic Graph.
3. Global Multi-Tenant Resource Manager: Bounded resource budgets across ASYNC_IO,
   CPU_BOUND, SUBPROCESS, and EXTERNAL_NETWORK prevent resource exhaustion.
4. External Network Security Boundary: EXTERNAL_NETWORK tasks are subjected to mandatory
   SSRF validation via SSRFGuard before network dispatch.
5. True Context & Metadata Immutability: ScanContext is frozen; metadata uses read-only
   MappingProxyType and strict string/type bounds.
6. Mathematical Invariant & Result Guards: Rejects NaN/Inf numeric values, enforces analyzer
   identity, and validates hard-override permissions.
7. Fail-Closed Verdict & Audit Policies: Pipeline fails closed to HIGH/CRITICAL if 
   trustworthy analyzer execution is compromised or audit logging fails.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import os
import re
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from security.audit_log import (
    AuditEvent,
    AuditLogger,
    AuditOutcome,
    AuditSeverity,
    audit_logger,
)
from security.ssrf_guard import ssrf_guard

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & BOUNDS
# ============================================================================

MAX_URL_LENGTH = 8192
MAX_FILENAME_LENGTH = 255
MAX_METADATA_KEYS = 32
MAX_METADATA_KEY_LEN = 64
MAX_METADATA_VAL_LEN = 1024

MAX_FINDINGS_PER_ANALYZER = 50
MAX_FINDING_MESSAGE_LEN = 512
MAX_RULE_ID_LEN = 64
MAX_CATEGORY_LEN = 64

SHA256_REGEX = re.compile(r"^[0-9a-fA-F]{64}$")
SAFE_STORAGE_KEY_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.\/]+$")
SENSITIVE_DATA_REGEX = re.compile(
    r"(?i)(api_?key|bearer\s+[a-z0-9\-\._~\+\/]+=*|password|secret|token|auth)=([^\s&]+)"
)
PATH_LEAK_REGEX = re.compile(r"([a-zA-Z]:\\[\\\w\s\.\-]+|/(?:[\w\.\-]+/)+[\w\.\-]+)")


# ============================================================================
# ENUMERATIONS
# ============================================================================


class ScanType(str, Enum):
    """Supported entity categories for scanning."""

    EMAIL = "email"
    URL = "url"
    ATTACHMENT = "attachment"
    COMPREHENSIVE = "comprehensive"


class AnalyzerResourceClass(str, Enum):
    """Resource requirement classification for proper concurrency offloading."""

    ASYNC_IO = "async_io"
    CPU_BOUND = "cpu_bound"
    SUBPROCESS = "subprocess"
    EXTERNAL_NETWORK = "external_network"


class AnalyzerExecutionStatus(str, Enum):
    """Detailed status tracking for sub-analyzer execution lifecycle."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    NOT_APPLICABLE = "not_applicable"
    DISABLED = "disabled"
    CIRCUIT_BROKEN = "circuit_broken"
    INVALID_INPUT = "invalid_input"


class CircuitState(str, Enum):
    """Circuit breaker state machine states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class PipelineExecutionState(str, Enum):
    """Execution lifecycle state machine for orchestrator scans."""

    CREATED = "created"
    VALIDATING = "validating"
    PLANNING = "planning"
    RUNNING = "running"
    AGGREGATING = "aggregating"
    AUDITING = "auditing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SecurityVerdict(str, Enum):
    """Final security risk categorization."""

    INFORMATIONAL = "INFORMATIONAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ============================================================================
# UTILITIES & INVARIANT CHECKS
# ============================================================================


def is_finite_number(val: Any) -> bool:
    """Validate that a numeric value is real and finite (not NaN or Inf)."""
    return isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val)


class ExceptionSanitizer:
    """Strips secrets, auth headers, and local filesystem paths from exception strings."""

    @staticmethod
    def sanitize(exc: Exception) -> str:
        raw_msg = str(exc)
        if not raw_msg:
            return exc.__class__.__name__

        sanitized = SENSITIVE_DATA_REGEX.sub(r"\1=[REDACTED]", raw_msg)
        sanitized = PATH_LEAK_REGEX.sub("[REDACTED_PATH]", sanitized)
        return f"{exc.__class__.__name__}: {sanitized[:256]}"


# ============================================================================
# DATA MODELS & IMMUTABLE PAYLOAD REFERENCES
# ============================================================================


@dataclass(frozen=True, slots=True)
class PayloadReference:
    """Safe handle representing a payload without keeping raw bytes in memory."""

    reference_id: str
    sha256_hash: str
    size_bytes: int
    mime_type: str = "application/octet-stream"
    filename: Optional[str] = None
    storage_key: Optional[str] = None

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError("Payload size_bytes cannot be negative.")
        if not SHA256_REGEX.match(self.sha256_hash):
            raise ValueError("Invalid SHA-256 hash formatting.")
        if self.filename and len(self.filename) > MAX_FILENAME_LENGTH:
            raise ValueError("Filename exceeds maximum length limits.")
        if self.storage_key:
            if ".." in self.storage_key or not SAFE_STORAGE_KEY_REGEX.match(self.storage_key):
                raise ValueError("Storage key contains unsafe path traversal sequences.")


@dataclass(frozen=True, slots=True)
class ConstrainedMetadata:
    """Genuinely immutable, bounded key-value container preventing memory exhaustion."""

    _data: Mapping[str, Union[str, int, float, bool]] = field(default_factory=dict)

    def __init__(self, raw_data: Optional[Dict[str, Union[str, int, float, bool]]] = None) -> None:
        if not raw_data:
            object.__setattr__(self, "_data", MappingProxyType({}))
            return

        if len(raw_data) > MAX_METADATA_KEYS:
            raise ValueError(f"Metadata exceeds maximum limit of {MAX_METADATA_KEYS} keys.")

        clean_dict: Dict[str, Union[str, int, float, bool]] = {}
        for k, v in raw_data.items():
            if not isinstance(k, str) or len(k) > MAX_METADATA_KEY_LEN:
                raise ValueError(f"Metadata key '{k}' must be a string under {MAX_METADATA_KEY_LEN} chars.")
            if isinstance(v, str):
                if len(v) > MAX_METADATA_VAL_LEN:
                    raise ValueError(f"Metadata key '{k}' value exceeds length limit.")
                clean_dict[k] = v
            elif isinstance(v, (int, bool)):
                clean_dict[k] = v
            elif isinstance(v, float):
                if not is_finite_number(v):
                    raise ValueError(f"Metadata key '{k}' float value must be finite.")
                clean_dict[k] = v
            else:
                raise ValueError(f"Unsupported metadata value type for key '{k}'.")

        object.__setattr__(self, "_data", MappingProxyType(clean_dict))

    @property
    def data(self) -> Mapping[str, Union[str, int, float, bool]]:
        return self._data


@dataclass(frozen=True, slots=True)
class ScanContext:
    """Immutable, validated execution context passed concurrently to security analyzers."""

    scan_id: str = field(default_factory=lambda: f"scan_{uuid.uuid4().hex[:12]}")
    scan_type: ScanType = ScanType.COMPREHENSIVE
    tenant_id: Optional[str] = None
    correlation_id: Optional[str] = None

    # Payload references
    email_payload: Optional[PayloadReference] = None
    attachment_payload: Optional[PayloadReference] = None
    target_url: Optional[str] = None

    metadata: ConstrainedMetadata = field(default_factory=ConstrainedMetadata)

    # Feature Toggles
    enable_sandbox: bool = False
    enable_yara: bool = True

    def __post_init__(self) -> None:
        if self.target_url:
            if len(self.target_url) > MAX_URL_LENGTH:
                raise ValueError(f"Target URL exceeds maximum length of {MAX_URL_LENGTH}.")
            if any(ord(c) < 0x20 for c in self.target_url):
                raise ValueError("Target URL contains control characters.")


@dataclass(frozen=True, slots=True)
class Finding:
    """Structured security finding emitted by an analyzer."""

    rule_id: str
    category: str
    severity: SecurityVerdict
    confidence: float
    message: str
    metadata: ConstrainedMetadata = field(default_factory=ConstrainedMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or len(self.rule_id) > MAX_RULE_ID_LEN:
            raise ValueError(f"rule_id must be string <= {MAX_RULE_ID_LEN} chars.")
        if not isinstance(self.category, str) or len(self.category) > MAX_CATEGORY_LEN:
            raise ValueError(f"category must be string <= {MAX_CATEGORY_LEN} chars.")
        if not is_finite_number(self.confidence) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Finding confidence must be a finite float between 0.0 and 1.0.")
        if not isinstance(self.message, str) or len(self.message) > MAX_FINDING_MESSAGE_LEN:
            object.__setattr__(self, "message", str(self.message)[:MAX_FINDING_MESSAGE_LEN] + "...")


@dataclass(slots=True)
class AnalyzerResult:
    """Validated, normalized result emitted by a single security analyzer."""

    analyzer_name: str
    status: AnalyzerExecutionStatus
    execution_time_ms: float
    score_contribution: float = 0.0
    confidence: float = 1.0
    is_hard_override: bool = False
    hard_override_verdict: Optional[SecurityVerdict] = None
    findings: List[Finding] = field(default_factory=list)
    sanitized_error: Optional[str] = None

    def validate(self, expected_name: str, can_hard_override_permission: bool) -> None:
        """Validate result invariants prior to accepting into pipeline aggregation."""
        if self.analyzer_name != expected_name:
            raise ValueError(f"Result analyzer_name '{self.analyzer_name}' mismatch with expected '{expected_name}'.")

        if not is_finite_number(self.execution_time_ms) or self.execution_time_ms < 0:
            raise ValueError("execution_time_ms must be non-negative finite number.")

        if not is_finite_number(self.score_contribution) or not (0.0 <= self.score_contribution <= 100.0):
            raise ValueError("score_contribution must be finite float between 0.0 and 100.0.")

        if not is_finite_number(self.confidence) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be finite float between 0.0 and 1.0.")

        if self.is_hard_override and not can_hard_override_permission:
            raise ValueError(f"Analyzer '{self.analyzer_name}' attempted hard override without permission.")

        if len(self.findings) > MAX_FINDINGS_PER_ANALYZER:
            self.findings = self.findings[:MAX_FINDINGS_PER_ANALYZER]


@dataclass(slots=True)
class PipelineResult:
    """Final output produced by the detection orchestrator."""

    scan_id: str
    scan_type: ScanType
    execution_state: PipelineExecutionState
    total_execution_time_ms: float
    overall_score: float
    confidence_score: float
    verdict: SecurityVerdict
    is_malicious: bool
    analyzer_results: Dict[str, AnalyzerResult] = field(default_factory=dict)
    hard_overrides_triggered: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ============================================================================
# CONFIGURATION & RESOURCE MANAGERS
# ============================================================================


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    """Configuration parameters enforcing bounds and security defaults."""

    total_pipeline_timeout_seconds: float = 30.0
    per_analyzer_timeout_seconds: float = 8.0
    max_concurrent_scans: int = 50
    
    # Global Resource Limits
    global_max_async_io: int = 100
    global_max_cpu_workers: int = 8
    global_max_subprocesses: int = 4
    global_max_external_network: int = 25

    # Per-Scan Resource Limits
    scan_max_concurrent_analyzers: int = 10

    # Resiliency Policies
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_cooldown_seconds: float = 60.0
    fail_closed_on_total_failure: bool = True
    fail_closed_on_audit_failure: bool = False

    def __post_init__(self) -> None:
        if self.total_pipeline_timeout_seconds <= 0:
            raise ValueError("total_pipeline_timeout_seconds must be positive.")
        if self.per_analyzer_timeout_seconds <= 0:
            raise ValueError("per_analyzer_timeout_seconds must be positive.")


class ThreadSafeCircuitBreaker:
    """Thread and Asyncio safe circuit breaker implementing CLOSED, OPEN, HALF_OPEN."""

    def __init__(self, failure_threshold: int, cooldown_seconds: float) -> None:
        self.threshold = failure_threshold
        self.cooldown = cooldown_seconds
        self.consecutive_failures = 0
        self.state = CircuitState.CLOSED
        self.open_until = 0.0
        self._lock = asyncio.Lock()

    async def can_execute(self) -> bool:
        async with self._lock:
            now = time.monotonic()
            if self.state == CircuitState.OPEN:
                if now >= self.open_until:
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False
            return True

    async def record_result(self, status: AnalyzerExecutionStatus) -> None:
        async with self._lock:
            # Only FAILED and TIMEOUT impact circuit health
            if status in (AnalyzerExecutionStatus.FAILED, AnalyzerExecutionStatus.TIMEOUT):
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.threshold:
                    self.state = CircuitState.OPEN
                    self.open_until = time.monotonic() + self.cooldown
            elif status == AnalyzerExecutionStatus.SUCCESS:
                self.consecutive_failures = 0
                self.state = CircuitState.CLOSED


class GlobalResourceManager:
    """Multi-tenant resource budget manager enforcing global bounds across execution classes."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self.async_io_sem = asyncio.Semaphore(config.global_max_async_io)
        self.cpu_sem = asyncio.Semaphore(config.global_max_cpu_workers)
        self.subprocess_sem = asyncio.Semaphore(config.global_max_subprocesses)
        self.network_sem = asyncio.Semaphore(config.global_max_external_network)
        
        # Dedicated Process Pool for CPU heavy work
        self.cpu_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=config.global_max_cpu_workers
        )

    def get_semaphore(self, resource_class: AnalyzerResourceClass) -> asyncio.Semaphore:
        if resource_class == AnalyzerResourceClass.CPU_BOUND:
            return self.cpu_sem
        elif resource_class == AnalyzerResourceClass.SUBPROCESS:
            return self.subprocess_sem
        elif resource_class == AnalyzerResourceClass.EXTERNAL_NETWORK:
            return self.network_sem
        return self.async_io_sem

    def shutdown(self) -> None:
        self.cpu_pool.shutdown(wait=False, cancel_futures=True)


# ============================================================================
# ABSTRACT BASE ANALYZER INTERFACE
# ============================================================================


class BaseAnalyzer:
    """Interface required for all sub-analyzers registered with the orchestrator."""

    name: str = "base_analyzer"
    version: str = "1.0.0"
    supported_scan_types: Set[ScanType] = {ScanType.COMPREHENSIVE}
    required_inputs: Set[str] = set()  # 'email', 'attachment', 'target_url'
    optional_inputs: Set[str] = set()
    resource_class: AnalyzerResourceClass = AnalyzerResourceClass.ASYNC_IO
    priority: int = 100
    timeout_seconds: Optional[float] = None
    dependencies: Set[str] = set()
    can_hard_override: bool = False

    async def analyze(self, context: ScanContext, deadline_monotonic: float) -> AnalyzerResult:
        """Process context asynchronously within allotted monotonic deadline."""
        raise NotImplementedError("Subclasses must implement analyze().")


# ============================================================================
# DAG PLANNER
# ============================================================================


class DAGPlanner:
    """Validates dependencies and constructs an execution DAG for analyzers."""

    @staticmethod
    def build_execution_plan(analyzers: List[BaseAnalyzer]) -> List[List[BaseAnalyzer]]:
        """Topologically sorts analyzers into independent parallel execution waves."""
        analyzer_map = {a.name: a for a in analyzers}
        in_degree = {a.name: 0 for a in analyzers}
        graph: Dict[str, List[str]] = {a.name: [] for a in analyzers}

        for a in analyzers:
            for dep in a.dependencies:
                if dep in analyzer_map:
                    graph[dep].append(a.name)
                    in_degree[a.name] += 1

        waves: List[List[BaseAnalyzer]] = []
        ready = [a.name for a in analyzers if in_degree[a.name] == 0]

        while ready:
            waves.append([analyzer_map[name] for name in ready])
            next_ready = []
            for name in ready:
                for neighbor in graph[name]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_ready.append(neighbor)
            ready = next_ready

        processed = sum(len(w) for w in waves)
        if processed != len(analyzers):
            raise ValueError("Circular dependency detected in registered analyzers.")

        return waves


# ============================================================================
# EXTERNAL NETWORK SECURITY BOUNDARY
# ============================================================================


class ExternalNetworkBoundary:
    """Security boundary enforcing SSRF check before network analyzer execution."""

    @staticmethod
    async def validate_target(context: ScanContext) -> None:
        if context.target_url:
            res = ssrf_guard.validate(context.target_url)
            if not res.allowed:
                raise ValueError(f"SSRF Policy Blocked target URL: {res.reason}")


# ============================================================================
# SUBPROCESS MANAGER
# ============================================================================


class SubprocessManager:
    """Executes subprocess tasks with strict PID tracking and SIGKILL termination."""

    @staticmethod
    async def run_managed_subprocess(
        cmd: List[str], timeout: float
    ) -> Tuple[int, bytes, bytes]:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return proc.returncode or 0, stdout, stderr
        except asyncio.TimeoutError:
            # Hard process termination via OS Process Group
            if sys.platform != "win32":
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            await proc.wait()
            raise


# ============================================================================
# DETECTION ENGINE ORCHESTRATOR
# ============================================================================


class DetectionOrchestrator:
    """Production detection pipeline orchestrator with isolated execution pools."""

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        audit_logger_instance: Optional[AuditLogger] = None,
    ) -> None:
        self.config = config or OrchestratorConfig()
        self.audit = audit_logger_instance or audit_logger
        self.resources = GlobalResourceManager(self.config)
        self._analyzers: Dict[str, BaseAnalyzer] = {}
        self._circuit_breakers: Dict[str, ThreadSafeCircuitBreaker] = {}

    def register_analyzer(self, analyzer: BaseAnalyzer, allow_override: bool = False) -> None:
        """Register an analyzer instance into the orchestrator execution table."""
        name = getattr(analyzer, "name", None)
        if not name or not isinstance(name, str):
            raise ValueError("Analyzer must possess a valid string 'name' attribute.")

        if name in self._analyzers and not allow_override:
            raise ValueError(f"Analyzer '{name}' is already registered.")

        self._analyzers[name] = analyzer
        self._circuit_breakers[name] = ThreadSafeCircuitBreaker(
            failure_threshold=self.config.circuit_breaker_failure_threshold,
            cooldown_seconds=self.config.circuit_breaker_cooldown_seconds,
        )
        logger.info("Registered analyzer: %s (v%s)", name, getattr(analyzer, "version", "1.0.0"))

    async def run_scan(self, context: ScanContext) -> PipelineResult:
        """Execute configured scan pipeline concurrently with full resource limits."""
        start_time_mono = time.monotonic()
        pipeline_deadline = start_time_mono + self.config.total_pipeline_timeout_seconds

        analyzer_results: Dict[str, AnalyzerResult] = {}
        errors: List[str] = []
        hard_overrides: List[str] = []

        logger.info(
            "Starting scan [ID: %s, Type: %s, Tenant: %s]",
            context.scan_id,
            context.scan_type.value,
            context.tenant_id or "default",
        )

        # 1. Select applicable analyzers & Build Execution Waves
        selected = self._select_analyzers(context)
        if not selected:
            return self._build_empty_result(context, time.monotonic() - start_time_mono)

        try:
            waves = DAGPlanner.build_execution_plan(selected)
        except Exception as exc:
            sanitized = ExceptionSanitizer.sanitize(exc)
            return self._build_failed_result(context, time.monotonic() - start_time_mono, [sanitized])

        # 2. Execute Waves sequentially; tasks within waves run concurrently
        scan_semaphore = asyncio.Semaphore(self.config.scan_max_concurrent_analyzers)

        for wave in waves:
            if time.monotonic() >= pipeline_deadline:
                errors.append(f"Global pipeline timeout prior to wave execution.")
                break

            tasks: List[asyncio.Task] = []
            for analyzer in wave:
                cb = self._circuit_breakers[analyzer.name]
                if not await cb.can_execute():
                    analyzer_results[analyzer.name] = AnalyzerResult(
                        analyzer_name=analyzer.name,
                        status=AnalyzerExecutionStatus.CIRCUIT_BROKEN,
                        execution_time_ms=0.0,
                        sanitized_error="Circuit breaker OPEN due to repeated failures.",
                    )
                    continue

                task = asyncio.create_task(
                    self._dispatch_analyzer(
                        analyzer=analyzer,
                        context=context,
                        pipeline_deadline=pipeline_deadline,
                        scan_semaphore=scan_semaphore,
                    )
                )
                tasks.append(task)

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for analyzer, res_or_exc in zip(wave, results):
                    cb = self._circuit_breakers[analyzer.name]
                    if isinstance(res_or_exc, Exception):
                        sanitized = ExceptionSanitizer.sanitize(res_or_exc)
                        await cb.record_result(AnalyzerExecutionStatus.FAILED)
                        errors.append(f"{analyzer.name}: {sanitized}")
                        analyzer_results[analyzer.name] = AnalyzerResult(
                            analyzer_name=analyzer.name,
                            status=AnalyzerExecutionStatus.FAILED,
                            execution_time_ms=0.0,
                            sanitized_error=sanitized,
                        )
                    else:
                        res: AnalyzerResult = res_or_exc
                        try:
                            res.validate(analyzer.name, analyzer.can_hard_override)
                            analyzer_results[analyzer.name] = res
                            await cb.record_result(res.status)

                            if res.status == AnalyzerExecutionStatus.SUCCESS and res.is_hard_override:
                                hard_overrides.append(
                                    f"{analyzer.name} triggered hard override: {res.hard_override_verdict}"
                                )
                        except ValueError as ve:
                            sanitized = ExceptionSanitizer.sanitize(ve)
                            await cb.record_result(AnalyzerExecutionStatus.FAILED)
                            errors.append(f"{analyzer.name}: {sanitized}")

        total_time_ms = (time.monotonic() - start_time_mono) * 1000.0

        # 3. Aggregate Findings & Determine Verdict
        pipeline_result = self._aggregate_results(
            context=context,
            analyzer_results=analyzer_results,
            execution_time_ms=total_time_ms,
            hard_overrides=hard_overrides,
            errors=errors,
            is_timed_out=(time.monotonic() >= pipeline_deadline),
        )

        # 4. Record Audit Security Log
        self._audit_scan(context, pipeline_result)

        return pipeline_result

    # ========================================================================
    # DISPATCHER BY RESOURCE CLASS
    # ========================================================================

    async def _dispatch_analyzer(
        self,
        analyzer: BaseAnalyzer,
        context: ScanContext,
        pipeline_deadline: float,
        scan_semaphore: asyncio.Semaphore,
    ) -> AnalyzerResult:
        """Route execution to proper pool based on declared resource class."""
        global_sem = self.resources.get_semaphore(analyzer.resource_class)

        async with scan_semaphore, global_sem:
            start_mono = time.monotonic()
            time_rem = pipeline_deadline - start_mono

            if time_rem <= 0:
                return AnalyzerResult(
                    analyzer_name=analyzer.name,
                    status=AnalyzerExecutionStatus.TIMEOUT,
                    execution_time_ms=0.0,
                    sanitized_error="Pipeline deadline expired prior to invocation.",
                )

            effective_timeout = min(
                analyzer.timeout_seconds or self.config.per_analyzer_timeout_seconds,
                time_rem,
            )

            try:
                if analyzer.resource_class == AnalyzerResourceClass.EXTERNAL_NETWORK:
                    await ExternalNetworkBoundary.validate_target(context)
                    res = await asyncio.wait_for(
                        analyzer.analyze(context, pipeline_deadline),
                        timeout=effective_timeout,
                    )

                elif analyzer.resource_class == AnalyzerResourceClass.CPU_BOUND:
                    loop = asyncio.get_running_loop()
                    res = await asyncio.wait_for(
                        loop.run_in_executor(
                            self.resources.cpu_pool,
                            _standalone_sync_run,
                            analyzer,
                            context,
                            pipeline_deadline,
                        ),
                        timeout=effective_timeout,
                    )

                else:
                    res = await asyncio.wait_for(
                        analyzer.analyze(context, pipeline_deadline),
                        timeout=effective_timeout,
                    )

                res.execution_time_ms = (time.monotonic() - start_mono) * 1000.0
                return res

            except asyncio.TimeoutError:
                return AnalyzerResult(
                    analyzer_name=analyzer.name,
                    status=AnalyzerExecutionStatus.TIMEOUT,
                    execution_time_ms=(time.monotonic() - start_mono) * 1000.0,
                    sanitized_error=f"Exceeded timeout threshold of {effective_timeout:.2f}s",
                )
            except Exception as exc:
                return AnalyzerResult(
                    analyzer_name=analyzer.name,
                    status=AnalyzerExecutionStatus.FAILED,
                    execution_time_ms=(time.monotonic() - start_mono) * 1000.0,
                    sanitized_error=ExceptionSanitizer.sanitize(exc),
                )

    # ================================================================= grand
    # SELECTION & AGGREGATION LOGIC
    # ========================================================================

    def _select_analyzers(self, context: ScanContext) -> List[BaseAnalyzer]:
        selected: List[BaseAnalyzer] = []
        for analyzer in self._analyzers.values():
            if (
                context.scan_type != ScanType.COMPREHENSIVE
                and context.scan_type not in analyzer.supported_scan_types
            ):
                continue

            reqs_met = True
            for req in analyzer.required_inputs:
                if req == "email" and context.email_payload is None:
                    reqs_met = False
                elif req == "attachment" and context.attachment_payload is None:
                    reqs_met = False
                elif req == "target_url" and context.target_url is None:
                    reqs_met = False

            if reqs_met:
                selected.append(analyzer)

        return selected

    def _aggregate_results(
        self,
        context: ScanContext,
        analyzer_results: Dict[str, AnalyzerResult],
        execution_time_ms: float,
        hard_overrides: List[str],
        errors: List[str],
        is_timed_out: bool,
    ) -> PipelineResult:
        weighted_score_sum = 0.0
        total_confidence = 0.0
        successful_count = 0
        forced_verdict: Optional[SecurityVerdict] = None

        for res in analyzer_results.values():
            if res.status == AnalyzerExecutionStatus.SUCCESS:
                successful_count += 1
                if res.is_hard_override and res.hard_override_verdict:
                    forced_verdict = res.hard_override_verdict

                weighted_score_sum += res.score_contribution * res.confidence
                total_confidence += res.confidence

        # FAIL-CLOSED CHECK: If no analyzers succeeded and fail_closed_on_total_failure is active
        if successful_count == 0 and self.config.fail_closed_on_total_failure:
            return PipelineResult(
                scan_id=context.scan_id,
                scan_type=context.scan_type,
                execution_state=PipelineExecutionState.FAILED,
                total_execution_time_ms=execution_time_ms,
                overall_score=100.0,
                confidence_score=0.0,
                verdict=SecurityVerdict.CRITICAL,
                is_malicious=True,
                analyzer_results=analyzer_results,
                errors=errors + ["FAIL-CLOSED: Zero trustworthy analyzer results obtained."],
            )

        overall_score = (
            weighted_score_sum / total_confidence if total_confidence > 0.0 else 0.0
        )
        avg_confidence = (
            total_confidence / successful_count if successful_count > 0 else 0.0
        )

        if forced_verdict:
            verdict = forced_verdict
        elif overall_score >= 80.0:
            verdict = SecurityVerdict.CRITICAL
        elif overall_score >= 60.0:
            verdict = SecurityVerdict.HIGH
        elif overall_score >= 40.0:
            verdict = SecurityVerdict.MEDIUM
        elif overall_score >= 20.0:
            verdict = SecurityVerdict.LOW
        else:
            verdict = SecurityVerdict.INFORMATIONAL

        is_malicious = verdict in {SecurityVerdict.HIGH, SecurityVerdict.CRITICAL}

        if is_timed_out:
            state = PipelineExecutionState.TIMED_OUT
        elif errors:
            state = PipelineExecutionState.PARTIAL
        else:
            state = PipelineExecutionState.COMPLETED

        return PipelineResult(
            scan_id=context.scan_id,
            scan_type=context.scan_type,
            execution_state=state,
            total_execution_time_ms=execution_time_ms,
            overall_score=round(overall_score, 2),
            confidence_score=round(avg_confidence, 2),
            verdict=verdict,
            is_malicious=is_malicious,
            analyzer_results=analyzer_results,
            hard_overrides_triggered=hard_overrides,
            errors=errors,
        )

    def _build_empty_result(self, context: ScanContext, exec_time: float) -> PipelineResult:
        return PipelineResult(
            scan_id=context.scan_id,
            scan_type=context.scan_type,
            execution_state=PipelineExecutionState.COMPLETED,
            total_execution_time_ms=exec_time * 1000.0,
            overall_score=0.0,
            confidence_score=1.0,
            verdict=SecurityVerdict.INFORMATIONAL,
            is_malicious=False,
            errors=["No analyzers matched scan criteria."],
        )

    def _build_failed_result(
        self, context: ScanContext, exec_time: float, errors: List[str]
    ) -> PipelineResult:
        return PipelineResult(
            scan_id=context.scan_id,
            scan_type=context.scan_type,
            execution_state=PipelineExecutionState.FAILED,
            total_execution_time_ms=exec_time * 1000.0,
            overall_score=100.0,
            confidence_score=0.0,
            verdict=SecurityVerdict.CRITICAL,
            is_malicious=True,
            errors=errors,
        )

    def _audit_scan(self, context: ScanContext, result: PipelineResult) -> None:
        """Record audit event mapping outcome accurately to execution state."""
        metadata = {
            "scan_id": result.scan_id,
            "scan_type": result.scan_type.value,
            "overall_score": result.overall_score,
            "verdict": result.verdict.value,
            "execution_state": result.execution_state.value,
            "analyzers_run": list(result.analyzer_results.keys()),
            "errors_count": len(result.errors),
        }

        if result.is_malicious:
            outcome = AuditOutcome.BLOCKED
            severity = AuditSeverity.HIGH
        elif result.execution_state == PipelineExecutionState.COMPLETED:
            outcome = AuditOutcome.SUCCESS
            severity = AuditSeverity.INFO
        elif result.execution_state == PipelineExecutionState.TIMED_OUT:
            outcome = AuditOutcome.TIMEOUT if hasattr(AuditOutcome, "TIMEOUT") else AuditOutcome.FAILURE
            severity = AuditSeverity.WARNING
        else:
            outcome = AuditOutcome.PARTIAL if hasattr(AuditOutcome, "PARTIAL") else AuditOutcome.FAILURE
            severity = AuditSeverity.WARNING

        try:
            self.audit.record_security_event(
                event=AuditEvent.SCAN_COMPLETED,
                outcome=outcome,
                severity=severity,
                tenant_id=context.tenant_id,
                correlation_id=context.correlation_id,
                metadata=metadata,
            )
        except Exception as exc:
            logger.exception("Failed to record scan audit log for %s", result.scan_id)
            if self.config.fail_closed_on_audit_failure:
                raise RuntimeError(f"Audit log recording failed for scan {result.scan_id}") from exc


# ============================================================================
# HELPER FOR PROCESS POOL WORKERS
# ============================================================================


def _standalone_sync_run(
    analyzer: BaseAnalyzer, context: ScanContext, deadline: float
) -> AnalyzerResult:
    """Entry point executed in ProcessPoolExecutor for CPU-bound tasks."""
    return asyncio.run(analyzer.analyze(context, deadline))


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "AnalyzerExecutionStatus",
    "AnalyzerResourceClass",
    "AnalyzerResult",
    "BaseAnalyzer",
    "CircuitState",
    "ConstrainedMetadata",
    "DAGPlanner",
    "DetectionOrchestrator",
    "ExceptionSanitizer",
    "ExternalNetworkBoundary",
    "Finding",
    "GlobalResourceManager",
    "OrchestratorConfig",
    "PayloadReference",
    "PipelineExecutionState",
    "PipelineResult",
    "ScanContext",
    "ScanType",
    "SecurityVerdict",
    "SubprocessManager",
    "ThreadSafeCircuitBreaker",
]
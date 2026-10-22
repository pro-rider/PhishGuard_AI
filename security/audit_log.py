"""
PhishGuard_AI - Enterprise Security Audit Logging Engine

Production-grade, tamper-evident security audit trail for authentication,
scanning, security controls, quarantine, case management, and administrative actions.

Features:
- Guaranteed single-path persistence (record vs. record_async) preventing double writes.
- Multi-tenant, stream-isolated hash chaining with sequence numbers and startup seeding.
- Constant-time hash verifications (hmac.compare_digest) and external SIEM anchoring.
- Strict scalar bounds and depth protection against deeply nested/malicious metadata.
- Configurable failure modes (FAIL_OPEN vs FAIL_CLOSED) with deadlock-free error logging.
- Auto-integration with config.internal.settings and structlog contextvars.
- Expanded security event taxonomy for full lifecycle traceability.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Protocol, Set, Tuple

import structlog.contextvars

from config.internal.settings import settings
from security.pii_redactor import PIIRedactor

# ============================================================================
# CONSTANTS & SECURITY BOUNDS
# ============================================================================

MAX_EVENT_NAME_LENGTH = 128
MAX_ACTOR_ID_LENGTH = 256
MAX_TENANT_ID_LENGTH = 256
MAX_RESOURCE_ID_LENGTH = 512
MAX_CORRELATION_ID_LENGTH = 256
MAX_SOURCE_LENGTH = 128

# Metadata Security Guardrails
MAX_METADATA_KEYS = 100
MAX_METADATA_KEY_LENGTH = 128
MAX_METADATA_VALUE_LENGTH = 4096
MAX_METADATA_DEPTH = 5
MAX_METADATA_TOTAL_BYTES = 32768
MAX_COLLECTION_ITEMS = 100

EVENT_NAME_REGEX = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")
IDENTIFIER_REGEX = re.compile(r"^[\w.@:/+\-=]{1,512}$", re.UNICODE)

# Allowed scalar types for metadata and canonical serialization
APPROVED_METADATA_TYPES = (
    str,
    int,
    float,
    bool,
    type(None),
)

# ============================================================================
# ENUMERATIONS
# ============================================================================


class AuditSeverity(str, Enum):
    """Describes security significance rather than diagnostic verbosity."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditOutcome(str, Enum):
    """Result of the audited operation."""

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    BLOCKED = "blocked"
    ERROR = "error"
    UNKNOWN = "unknown"


class AuditCategory(str, Enum):
    """High-level classification used by SIEM and audit consumers."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    SCANNING = "scanning"
    SECURITY = "security"
    POLICY = "policy"
    QUARANTINE = "quarantine"
    CASE_MANAGEMENT = "case_management"
    THREAT_INTELLIGENCE = "threat_intelligence"
    DATA_ACCESS = "data_access"
    MODEL = "model"
    SYSTEM = "system"
    ADMINISTRATION = "administration"


class AuditFailureMode(str, Enum):
    """Behavior when audit storage persistence fails."""

    FAIL_OPEN = "fail_open"  # Log error safely and allow execution to continue
    FAIL_CLOSED = "fail_closed"  # Raise RuntimeError to abort sensitive operation


class AuditWriteStatus(str, Enum):
    """Result status from storage write operations."""

    PERSISTED = "persisted"
    DUPLICATE = "duplicate"
    FAILED = "failed"


# ============================================================================
# AUDIT EVENT TAXONOMY
# ============================================================================


class AuditEvent(str, Enum):
    """Canonical PhishGuard security events."""

    # Authentication & Authorization
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_REFRESH = "auth.token.refresh"
    AUTHZ_ACCESS_GRANTED = "authz.access.granted"
    AUTHZ_ACCESS_DENIED = "authz.access.denied"
    AUTHZ_ROLE_CHANGED = "authz.role.changed"

    # Core Scan Lifecycle
    SCAN_SUBMITTED = "scan.submitted"
    SCAN_STARTED = "scan.started"
    SCAN_COMPLETED = "scan.completed"
    SCAN_FAILED = "scan.failed"
    SCAN_CANCELLED = "scan.cancelled"

    # Specific Target Pipeline Events
    ATTACHMENT_UPLOADED = "attachment.uploaded"
    ATTACHMENT_BLOCKED = "attachment.blocked"
    ATTACHMENT_QUARANTINED = "attachment.quarantined"
    URL_ANALYSIS_STARTED = "url.analysis.started"
    URL_ANALYSIS_COMPLETED = "url.analysis.completed"
    URL_ANALYSIS_BLOCKED = "url.analysis.blocked"
    EMAIL_ANALYSIS_STARTED = "email.analysis.started"
    EMAIL_ANALYSIS_COMPLETED = "email.analysis.completed"
    IOC_EXTRACTED = "ioc.extracted"

    # Sandbox Events
    SANDBOX_EXECUTION_STARTED = "sandbox.execution.started"
    SANDBOX_EXECUTION_COMPLETED = "sandbox.execution.completed"
    SANDBOX_EXECUTION_BLOCKED = "sandbox.execution.blocked"

    # Threat Intelligence & Models
    THREAT_INTEL_QUERY = "threat_intel.query"
    THREAT_INTEL_RESULT = "threat_intel.result"
    THREAT_INTEL_FAILURE = "threat_intel.failure"
    THREAT_INTEL_RATE_LIMITED = "threat_intel.rate_limited"
    MODEL_LOADED = "model.loaded"
    MODEL_CHANGED = "model.changed"
    MODEL_PREDICTION_GENERATED = "model.prediction.generated"
    MODEL_PREDICTION_REJECTED = "model.prediction.rejected"
    MODEL_DRIFT_DETECTED = "model.drift.detected"

    # Security Controls
    SSRF_BLOCKED = "security.ssrf.blocked"
    EGRESS_BLOCKED = "security.egress.blocked"
    RATE_LIMIT_TRIGGERED = "security.rate_limit.triggered"
    INVALID_SIGNATURE = "security.invalid_signature"
    SUSPICIOUS_REQUEST = "security.suspicious_request"
    SECURITY_POLICY_VIOLATION = "security.policy.violation"

    # Quarantine, Case, & Policy
    QUARANTINE_CREATED = "quarantine.created"
    QUARANTINE_RELEASED = "quarantine.released"
    QUARANTINE_DELETED = "quarantine.deleted"
    CASE_CREATED = "case.created"
    CASE_UPDATED = "case.updated"
    CASE_CLOSED = "case.closed"
    POLICY_CREATED = "policy.created"
    POLICY_UPDATED = "policy.updated"
    POLICY_DELETED = "policy.deleted"

    # Data & Administration
    DATA_READ = "data.read"
    DATA_CREATED = "data.created"
    DATA_UPDATED = "data.updated"
    DATA_DELETED = "data.deleted"
    ADMIN_USER_CREATED = "admin.user.created"
    ADMIN_USER_DISABLED = "admin.user.disabled"
    ADMIN_PERMISSION_CHANGED = "admin.permission.changed"
    ADMIN_CONFIGURATION_CHANGED = "admin.configuration.changed"
    ADMIN_SECRET_ROTATED = "admin.secret.rotated"


# ============================================================================
# CATEGORY MAPPING
# ============================================================================

_EVENT_CATEGORY_MAP: Dict[AuditEvent, AuditCategory] = {
    AuditEvent.AUTH_LOGIN_SUCCESS: AuditCategory.AUTHENTICATION,
    AuditEvent.AUTH_LOGIN_FAILURE: AuditCategory.AUTHENTICATION,
    AuditEvent.AUTH_LOGOUT: AuditCategory.AUTHENTICATION,
    AuditEvent.AUTH_TOKEN_REFRESH: AuditCategory.AUTHENTICATION,
    AuditEvent.AUTHZ_ACCESS_GRANTED: AuditCategory.AUTHORIZATION,
    AuditEvent.AUTHZ_ACCESS_DENIED: AuditCategory.AUTHORIZATION,
    AuditEvent.AUTHZ_ROLE_CHANGED: AuditCategory.AUTHORIZATION,
    AuditEvent.SCAN_SUBMITTED: AuditCategory.SCANNING,
    AuditEvent.SCAN_STARTED: AuditCategory.SCANNING,
    AuditEvent.SCAN_COMPLETED: AuditCategory.SCANNING,
    AuditEvent.SCAN_FAILED: AuditCategory.SCANNING,
    AuditEvent.SCAN_CANCELLED: AuditCategory.SCANNING,
    AuditEvent.ATTACHMENT_UPLOADED: AuditCategory.SCANNING,
    AuditEvent.ATTACHMENT_BLOCKED: AuditCategory.SECURITY,
    AuditEvent.ATTACHMENT_QUARANTINED: AuditCategory.QUARANTINE,
    AuditEvent.URL_ANALYSIS_STARTED: AuditCategory.SCANNING,
    AuditEvent.URL_ANALYSIS_COMPLETED: AuditCategory.SCANNING,
    AuditEvent.URL_ANALYSIS_BLOCKED: AuditCategory.SECURITY,
    AuditEvent.EMAIL_ANALYSIS_STARTED: AuditCategory.SCANNING,
    AuditEvent.EMAIL_ANALYSIS_COMPLETED: AuditCategory.SCANNING,
    AuditEvent.IOC_EXTRACTED: AuditCategory.THREAT_INTELLIGENCE,
    AuditEvent.SANDBOX_EXECUTION_STARTED: AuditCategory.SECURITY,
    AuditEvent.SANDBOX_EXECUTION_COMPLETED: AuditCategory.SECURITY,
    AuditEvent.SANDBOX_EXECUTION_BLOCKED: AuditCategory.SECURITY,
    AuditEvent.THREAT_INTEL_QUERY: AuditCategory.THREAT_INTELLIGENCE,
    AuditEvent.THREAT_INTEL_RESULT: AuditCategory.THREAT_INTELLIGENCE,
    AuditEvent.THREAT_INTEL_FAILURE: AuditCategory.THREAT_INTELLIGENCE,
    AuditEvent.THREAT_INTEL_RATE_LIMITED: AuditCategory.THREAT_INTELLIGENCE,
    AuditEvent.MODEL_LOADED: AuditCategory.MODEL,
    AuditEvent.MODEL_CHANGED: AuditCategory.MODEL,
    AuditEvent.MODEL_PREDICTION_GENERATED: AuditCategory.MODEL,
    AuditEvent.MODEL_PREDICTION_REJECTED: AuditCategory.MODEL,
    AuditEvent.MODEL_DRIFT_DETECTED: AuditCategory.MODEL,
    AuditEvent.SSRF_BLOCKED: AuditCategory.SECURITY,
    AuditEvent.EGRESS_BLOCKED: AuditCategory.SECURITY,
    AuditEvent.RATE_LIMIT_TRIGGERED: AuditCategory.SECURITY,
    AuditEvent.INVALID_SIGNATURE: AuditCategory.SECURITY,
    AuditEvent.SUSPICIOUS_REQUEST: AuditCategory.SECURITY,
    AuditEvent.SECURITY_POLICY_VIOLATION: AuditCategory.SECURITY,
    AuditEvent.QUARANTINE_CREATED: AuditCategory.QUARANTINE,
    AuditEvent.QUARANTINE_RELEASED: AuditCategory.QUARANTINE,
    AuditEvent.QUARANTINE_DELETED: AuditCategory.QUARANTINE,
    AuditEvent.CASE_CREATED: AuditCategory.CASE_MANAGEMENT,
    AuditEvent.CASE_UPDATED: AuditCategory.CASE_MANAGEMENT,
    AuditEvent.CASE_CLOSED: AuditCategory.CASE_MANAGEMENT,
    AuditEvent.POLICY_CREATED: AuditCategory.POLICY,
    AuditEvent.POLICY_UPDATED: AuditCategory.POLICY,
    AuditEvent.POLICY_DELETED: AuditCategory.POLICY,
    AuditEvent.DATA_READ: AuditCategory.DATA_ACCESS,
    AuditEvent.DATA_CREATED: AuditCategory.DATA_ACCESS,
    AuditEvent.DATA_UPDATED: AuditCategory.DATA_ACCESS,
    AuditEvent.DATA_DELETED: AuditCategory.DATA_ACCESS,
    AuditEvent.ADMIN_USER_CREATED: AuditCategory.ADMINISTRATION,
    AuditEvent.ADMIN_USER_DISABLED: AuditCategory.ADMINISTRATION,
    AuditEvent.ADMIN_PERMISSION_CHANGED: AuditCategory.ADMINISTRATION,
    AuditEvent.ADMIN_CONFIGURATION_CHANGED: AuditCategory.ADMINISTRATION,
    AuditEvent.ADMIN_SECRET_ROTATED: AuditCategory.ADMINISTRATION,
}

_RESERVED_FIELDS: Set[str] = {
    "event_id",
    "timestamp",
    "event_name",
    "category",
    "severity",
    "outcome",
    "actor_id",
    "tenant_id",
    "resource_id",
    "correlation_id",
    "request_id",
    "source",
    "service",
    "environment",
    "metadata",
    "stream_key",
    "sequence_number",
    "previous_hash",
    "event_hash",
    "schema_version",
}

# ============================================================================
# STORAGE INTERFACE
# ============================================================================


@dataclass(frozen=True, slots=True)
class AuditWriteResult:
    """Explicit response contract for storage layer operations."""

    status: AuditWriteStatus
    event_id: str
    error_message: Optional[str] = None


class AuditStorage(Protocol):
    """
    Storage abstraction protocol.
    Implementations handle persistence to PostgreSQL, OpenSearch, S3, or SIEMs.
    """

    def write(self, event: AuditRecord) -> AuditWriteResult:
        ...

    async def write_async(self, event: AuditRecord) -> AuditWriteResult:
        ...


# ============================================================================
# IMMUTABLE AUDIT RECORD
# ============================================================================


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """
    Immutable, validated security audit record representation.
    """

    event_id: str
    timestamp: str

    event_name: str
    category: str
    severity: str
    outcome: str

    actor_id: Optional[str]
    tenant_id: Optional[str]
    resource_id: Optional[str]

    correlation_id: Optional[str]
    request_id: Optional[str]

    source: str
    service: str
    environment: str

    metadata: Mapping[str, Any] = field(default_factory=dict)

    stream_key: str = "global:system"
    sequence_number: int = 1
    previous_hash: Optional[str] = None
    event_hash: Optional[str] = None

    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Enforces schema invariants on record initialization."""
        # Validate timestamp format
        try:
            dt = datetime.fromisoformat(self.timestamp)
            if dt.tzinfo is None or dt.tzinfo != timezone.utc:
                raise ValueError("Timestamp must be a UTC ISO-8601 string ending in +00:00 or Z")
        except Exception as e:
            raise ValueError(f"Invalid audit record timestamp '{self.timestamp}': {e}")

        if self.sequence_number < 1:
            raise ValueError("sequence_number must be greater than or equal to 1")

    def to_dict(self, include_hash: bool = True) -> Dict[str, Any]:
        """Convert record to a JSON-compatible dictionary."""
        d: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_name": self.event_name,
            "category": self.category,
            "severity": self.severity,
            "outcome": self.outcome,
            "actor_id": self.actor_id,
            "tenant_id": self.tenant_id,
            "resource_id": self.resource_id,
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
            "source": self.source,
            "service": self.service,
            "environment": self.environment,
            "metadata": dict(self.metadata),
            "stream_key": self.stream_key,
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
        }
        if include_hash:
            d["event_hash"] = self.event_hash
        return d

    def to_canonical_json(self) -> str:
        """
        Serialize event deterministically for SHA-256 hashing.
        Strictly excludes 'event_hash' to guarantee hash reproducibility.
        """
        return json.dumps(
            self.to_dict(include_hash=False),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def to_json(self) -> str:
        """Serialize complete record including event_hash."""
        return json.dumps(
            self.to_dict(include_hash=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


# ============================================================================
# CENTRAL AUDIT LOGGER
# ============================================================================


class AuditLogger:
    """Central security audit logging engine."""

    def __init__(
        self,
        *,
        service: Optional[str] = None,
        environment: Optional[str] = None,
        storage: Optional[AuditStorage] = None,
        enable_hash_chain: bool = True,
        default_failure_mode: AuditFailureMode = AuditFailureMode.FAIL_OPEN,
    ) -> None:
        # Resolve service and environment from settings if omitted
        raw_service = service or settings.APP_NAME
        raw_env = environment or settings.ENVIRONMENT.value

        self.service = self._validate_identifier(
            raw_service, "service", MAX_SOURCE_LENGTH
        )
        self.environment = self._validate_identifier(
            raw_env, "environment", MAX_SOURCE_LENGTH
        )

        self.storage = storage
        self.enable_hash_chain = enable_hash_chain
        self.default_failure_mode = default_failure_mode

        # Stream key -> (sequence_number, event_hash)
        self._stream_chains: Dict[str, Tuple[int, str]] = {}
        self._lock = threading.RLock()

    def seed_stream_state(
        self, stream_key: str, sequence_number: int, last_event_hash: str
    ) -> None:
        """
        Initializes in-memory stream state from durable DB on application startup.
        """
        valid_stream = self._validate_identifier(
            stream_key, "stream_key", MAX_TENANT_ID_LENGTH
        )
        if sequence_number < 1:
            raise ValueError("sequence_number must be >= 1")
        if not last_event_hash or len(last_event_hash) != 64:
            raise ValueError("last_event_hash must be a valid 64-char SHA-256 hex string")

        with self._lock:
            self._stream_chains[valid_stream] = (sequence_number, last_event_hash)

    # ========================================================================
    # PUBLIC RECORDING API
    # ========================================================================

    def record(
        self,
        *,
        event: AuditEvent | str,
        outcome: AuditOutcome | str,
        severity: AuditSeverity | str = AuditSeverity.INFO,
        source: str = "audit_logger",
        actor_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        failure_mode: Optional[AuditFailureMode] = None,
    ) -> AuditRecord:
        """
        Synchronously builds, hashes, and persists an audit record exactly once.
        """
        record = self._build_record(
            event=event,
            outcome=outcome,
            severity=severity,
            source=source,
            actor_id=actor_id,
            tenant_id=tenant_id,
            resource_id=resource_id,
            correlation_id=correlation_id,
            request_id=request_id,
            metadata=metadata,
        )

        mode = failure_mode or self.default_failure_mode
        self._persist_sync(record, mode)
        return record

    async def record_async(
        self,
        *,
        event: AuditEvent | str,
        outcome: AuditOutcome | str,
        severity: AuditSeverity | str = AuditSeverity.INFO,
        source: str = "audit_logger",
        actor_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        failure_mode: Optional[AuditFailureMode] = None,
    ) -> AuditRecord:
        """
        Asynchronously builds, hashes, and persists an audit record exactly once.
        """
        record = self._build_record(
            event=event,
            outcome=outcome,
            severity=severity,
            source=source,
            actor_id=actor_id,
            tenant_id=tenant_id,
            resource_id=resource_id,
            correlation_id=correlation_id,
            request_id=request_id,
            metadata=metadata,
        )

        mode = failure_mode or self.default_failure_mode
        await self._persist_async(record, mode)
        return record

    # ========================================================================
    # INTERNAL BUILD & PERSISTENCE
    # ========================================================================

    def _build_record(
        self,
        *,
        event: AuditEvent | str,
        outcome: AuditOutcome | str,
        severity: AuditSeverity | str,
        source: str,
        actor_id: Optional[str],
        tenant_id: Optional[str],
        resource_id: Optional[str],
        correlation_id: Optional[str],
        request_id: Optional[str],
        metadata: Optional[Mapping[str, Any]],
    ) -> AuditRecord:
        """Pure helper that builds, normalizes, and hashes an AuditRecord."""
        # Auto-bind contextvars if parameters are omitted
        contextvars_dict = structlog.contextvars.get_contextvars()
        correlation_id = correlation_id or contextvars_dict.get("correlation_id")
        request_id = request_id or contextvars_dict.get("request_id")
        tenant_id = tenant_id or contextvars_dict.get("tenant_id")
        actor_id = actor_id or contextvars_dict.get("actor_id")

        normalized_event = self._normalize_event_name(event)
        normalized_severity = self._normalize_enum(
            severity, AuditSeverity, "severity"
        )
        normalized_outcome = self._normalize_enum(
            outcome, AuditOutcome, "outcome"
        )
        category = self._resolve_category(normalized_event)

        safe_source = self._validate_identifier(source, "source", MAX_SOURCE_LENGTH)
        safe_actor_id = self._validate_optional_identifier(
            actor_id, "actor_id", MAX_ACTOR_ID_LENGTH
        )
        safe_tenant_id = self._validate_optional_identifier(
            tenant_id, "tenant_id", MAX_TENANT_ID_LENGTH
        )
        safe_resource_id = self._validate_optional_identifier(
            resource_id, "resource_id", MAX_RESOURCE_ID_LENGTH
        )
        safe_correlation_id = self._validate_optional_identifier(
            correlation_id, "correlation_id", MAX_CORRELATION_ID_LENGTH
        )
        safe_request_id = self._validate_optional_identifier(
            request_id, "request_id", MAX_CORRELATION_ID_LENGTH
        )

        safe_metadata = self._sanitize_metadata(metadata)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        event_id = str(uuid.uuid4())

        # Stream key isolation
        stream_key = f"tenant:{safe_tenant_id}" if safe_tenant_id else "global:system"

        with self._lock:
            sequence_number = 1
            previous_hash: Optional[str] = None

            if self.enable_hash_chain:
                if stream_key in self._stream_chains:
                    last_seq, last_hash = self._stream_chains[stream_key]
                    sequence_number = last_seq + 1
                    previous_hash = last_hash

            unsigned_record = AuditRecord(
                event_id=event_id,
                timestamp=timestamp,
                event_name=normalized_event,
                category=category.value,
                severity=normalized_severity.value,
                outcome=normalized_outcome.value,
                actor_id=safe_actor_id,
                tenant_id=safe_tenant_id,
                resource_id=safe_resource_id,
                correlation_id=safe_correlation_id,
                request_id=safe_request_id,
                source=safe_source,
                service=self.service,
                environment=self.environment,
                metadata=safe_metadata,
                stream_key=stream_key,
                sequence_number=sequence_number,
                previous_hash=previous_hash,
                event_hash=None,
            )

            event_hash = self._calculate_hash(unsigned_record)

            signed_record = AuditRecord(
                event_id=unsigned_record.event_id,
                timestamp=unsigned_record.timestamp,
                event_name=unsigned_record.event_name,
                category=unsigned_record.category,
                severity=unsigned_record.severity,
                outcome=unsigned_record.outcome,
                actor_id=unsigned_record.actor_id,
                tenant_id=unsigned_record.tenant_id,
                resource_id=unsigned_record.resource_id,
                correlation_id=unsigned_record.correlation_id,
                request_id=unsigned_record.request_id,
                source=unsigned_record.source,
                service=unsigned_record.service,
                environment=unsigned_record.environment,
                metadata=unsigned_record.metadata,
                stream_key=unsigned_record.stream_key,
                sequence_number=unsigned_record.sequence_number,
                previous_hash=unsigned_record.previous_hash,
                event_hash=event_hash,
                schema_version=unsigned_record.schema_version,
            )

            if self.enable_hash_chain:
                self._stream_chains[stream_key] = (sequence_number, event_hash)

        return signed_record

    def _persist_sync(self, record: AuditRecord, mode: AuditFailureMode) -> None:
        if self.storage is None:
            return

        try:
            res = self.storage.write(record)
            if res.status == AuditWriteStatus.FAILED:
                self._handle_failure(record, mode, res.error_message)
        except Exception as exc:
            self._handle_failure(record, mode, str(exc))

    async def _persist_async(self, record: AuditRecord, mode: AuditFailureMode) -> None:
        if self.storage is None:
            return

        try:
            if hasattr(self.storage, "write_async"):
                res = await self.storage.write_async(record)
            else:
                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(None, self.storage.write, record)

            if res.status == AuditWriteStatus.FAILED:
                self._handle_failure(record, mode, res.error_message)
        except Exception as exc:
            self._handle_failure(record, mode, str(exc))

    @staticmethod
    def _handle_failure(
        record: AuditRecord, mode: AuditFailureMode, error_details: Optional[str]
    ) -> None:
        """
        Ultra-safe, deadlock-free failure logging that bypasses application loggers.
        Outputs directly to sys.stderr to prevent recursive logging crashes.
        """
        safe_err = {
            "error": "audit_storage_failure",
            "event_id": record.event_id,
            "event_name": record.event_name,
            "timestamp": record.timestamp,
        }
        sys.stderr.write(json.dumps(safe_err) + "\n")
        sys.stderr.flush()

        if mode == AuditFailureMode.FAIL_CLOSED:
            raise RuntimeError(
                f"Audit persistence failed for critical event '{record.event_name}' (ID: {record.event_id})."
            )

    # ========================================================================
    # SPECIALIZED CONVENIENCE HELPERS
    # ========================================================================

    def record_security_event(
        self,
        *,
        event: AuditEvent,
        outcome: AuditOutcome,
        severity: AuditSeverity = AuditSeverity.WARNING,
        source: str = "security_control",
        tenant_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        failure_mode: AuditFailureMode = AuditFailureMode.FAIL_CLOSED,
    ) -> AuditRecord:
        """Helper for security control events. Defaults to FAIL_CLOSED."""
        return self.record(
            event=event,
            outcome=outcome,
            severity=severity,
            source=source,
            tenant_id=tenant_id,
            resource_id=resource_id,
            correlation_id=correlation_id,
            request_id=request_id,
            metadata=metadata,
            failure_mode=failure_mode,
        )

    def record_scan_event(
        self,
        *,
        event: AuditEvent,
        scan_id: str,
        outcome: AuditOutcome,
        source: str = "scan_pipeline",
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AuditRecord:
        """Helper for scan lifecycle events."""
        scan_metadata = dict(metadata or {})
        scan_metadata["scan_id"] = scan_id

        return self.record(
            event=event,
            outcome=outcome,
            severity=severity,
            source=source,
            actor_id=actor_id,
            tenant_id=tenant_id,
            resource_id=scan_id,
            correlation_id=correlation_id,
            request_id=request_id,
            metadata=scan_metadata,
        )

    # ========================================================================
    # VALIDATION & SANITIZATION HELPERS
    # ========================================================================

    @staticmethod
    def _normalize_event_name(event: AuditEvent | str) -> str:
        if isinstance(event, AuditEvent):
            value = event.value
        elif isinstance(event, str):
            value = event.strip()
        else:
            raise TypeError("event must be an AuditEvent or string")

        if not value:
            raise ValueError("Audit event name cannot be empty")

        if len(value) > MAX_EVENT_NAME_LENGTH:
            raise ValueError("Audit event name exceeds maximum length")

        if not EVENT_NAME_REGEX.fullmatch(value):
            raise ValueError(f"Invalid audit event name: {value!r}")

        return value

    @staticmethod
    def _normalize_enum(
        value: Any, enum_type: type[Enum], field_name: str
    ) -> Enum:
        if isinstance(value, enum_type):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            for member in enum_type:
                if (
                    member.value.lower() == normalized
                    or member.name.lower() == normalized
                ):
                    return member

        raise ValueError(f"Invalid {field_name}: {value!r}")

    @staticmethod
    def _validate_identifier(
        value: str, field_name: str, max_length: int
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")

        value = value.strip()
        if not value:
            raise ValueError(f"{field_name} cannot be empty")

        if len(value) > max_length:
            raise ValueError(f"{field_name} exceeds maximum length of {max_length}")

        if not IDENTIFIER_REGEX.fullmatch(value):
            raise ValueError(f"Invalid characters in {field_name}")

        return value

    @classmethod
    def _validate_optional_identifier(
        cls, value: Optional[str], field_name: str, max_length: int
    ) -> Optional[str]:
        if value is None:
            return None

        if not isinstance(value, (str, int, uuid.UUID)):
            raise TypeError(
                f"{field_name} must be a str, int, or UUID instance (got {type(value).__name__})"
            )

        str_val = str(value).strip()
        if not str_val:
            return None

        return cls._validate_identifier(str_val, field_name, max_length)

    @classmethod
    def _sanitize_metadata(
        cls, metadata: Optional[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        """Enforces scalar type bounds, key length, total bytes, and depth bounds."""
        if metadata is None:
            return {}

        if not isinstance(metadata, Mapping):
            raise TypeError("Audit metadata must be a mapping")

        if len(metadata) > MAX_METADATA_KEYS:
            raise ValueError(f"Audit metadata exceeds limit of {MAX_METADATA_KEYS} keys")

        sanitized: Dict[str, Any] = {}
        for key, value in metadata.items():
            key_string = str(key).strip()

            if len(key_string) > MAX_METADATA_KEY_LENGTH:
                raise ValueError(
                    f"Metadata key '{key_string[:20]}...' exceeds max length of {MAX_METADATA_KEY_LENGTH}"
                )

            if key_string in _RESERVED_FIELDS:
                raise ValueError(
                    f"Reserved audit field cannot be placed in metadata: {key_string}"
                )

            safe_value = cls._sanitize_metadata_value(key_string, value, depth=1)
            sanitized[key_string] = safe_value

        # Validate total serialized bytes
        serialized_bytes = len(json.dumps(sanitized, default=str).encode("utf-8"))
        if serialized_bytes > MAX_METADATA_TOTAL_BYTES:
            raise ValueError(
                f"Metadata serialized payload size ({serialized_bytes} bytes) exceeds limit of {MAX_METADATA_TOTAL_BYTES} bytes"
            )

        return sanitized

    @classmethod
    def _sanitize_metadata_value(
        cls, key_context: str, value: Any, depth: int
    ) -> Any:
        if depth > MAX_METADATA_DEPTH:
            raise ValueError(
                f"Metadata depth at key '{key_context}' exceeds maximum allowed depth of {MAX_METADATA_DEPTH}"
            )

        if isinstance(value, APPROVED_METADATA_TYPES):
            if isinstance(value, str):
                if len(value) > MAX_METADATA_VALUE_LENGTH:
                    raise ValueError(
                        f"Metadata value at '{key_context}' exceeds max length of {MAX_METADATA_VALUE_LENGTH}"
                    )
                return PIIRedactor.sanitize_value(key_context, value)
            return PIIRedactor.sanitize_value(key_context, value)

        if isinstance(value, (list, tuple, set)):
            if len(value) > MAX_COLLECTION_ITEMS:
                raise ValueError(
                    f"Collection at metadata key '{key_context}' exceeds limit of {MAX_COLLECTION_ITEMS} items"
                )
            return [
                cls._sanitize_metadata_value(f"{key_context}[{idx}]", item, depth + 1)
                for idx, item in enumerate(value)
            ]

        if isinstance(value, Mapping):
            if len(value) > MAX_METADATA_KEYS:
                raise ValueError(
                    f"Nested dictionary at '{key_context}' exceeds limit of {MAX_METADATA_KEYS} keys"
                )
            res: Dict[str, Any] = {}
            for k, v in value.items():
                k_str = str(k).strip()
                if len(k_str) > MAX_METADATA_KEY_LENGTH:
                    raise ValueError(
                        f"Nested key '{k_str[:20]}' exceeds max length of {MAX_METADATA_KEY_LENGTH}"
                    )
                res[k_str] = cls._sanitize_metadata_value(
                    f"{key_context}.{k_str}", v, depth + 1
                )
            return res

        if isinstance(value, (uuid.UUID, datetime)):
            return str(value)

        raise TypeError(
            f"Unsupported type '{type(value).__name__}' in audit metadata key '{key_context}'. "
            "Only approved scalars, lists, and dicts are permitted."
        )

    @staticmethod
    def _resolve_category(event_name: str) -> AuditCategory:
        try:
            event = AuditEvent(event_name)
            return _EVENT_CATEGORY_MAP.get(event, AuditCategory.SYSTEM)
        except ValueError:
            prefix = event_name.split(".", 1)[0].lower()
            mapping = {
                "auth": AuditCategory.AUTHENTICATION,
                "authz": AuditCategory.AUTHORIZATION,
                "scan": AuditCategory.SCANNING,
                "security": AuditCategory.SECURITY,
                "policy": AuditCategory.POLICY,
                "quarantine": AuditCategory.QUARANTINE,
                "case": AuditCategory.CASE_MANAGEMENT,
                "model": AuditCategory.MODEL,
                "data": AuditCategory.DATA_ACCESS,
                "admin": AuditCategory.ADMINISTRATION,
                "threat_intel": AuditCategory.THREAT_INTELLIGENCE,
                "attachment": AuditCategory.SCANNING,
                "url": AuditCategory.SCANNING,
                "email": AuditCategory.SCANNING,
                "sandbox": AuditCategory.SECURITY,
            }
            return mapping.get(prefix, AuditCategory.SYSTEM)

    @staticmethod
    def _calculate_hash(record: AuditRecord) -> str:
        """Calculate deterministic SHA-256 hash from canonical JSON string."""
        canonical = record.to_canonical_json().encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


# Default module-level instance
audit_logger = AuditLogger()


# ============================================================================
# VERIFICATION & ANCHORING UTILITIES
# ============================================================================


def verify_event_hash(record: AuditRecord) -> bool:
    """
    Verifies SHA-256 digest integrity of a single audit record using constant-time comparison.
    """
    if not record.event_hash:
        return False

    expected_hash = AuditLogger._calculate_hash(record)
    return hmac.compare_digest(expected_hash, record.event_hash)


def verify_hash_chain(records: List[AuditRecord]) -> bool:
    """
    Verifies sequence numbers, stream keys, and hash continuity across an ordered list of audit records.
    """
    if not records:
        return True

    stream_keys = {r.stream_key for r in records}
    if len(stream_keys) > 1:
        raise ValueError("Cannot verify hash chain across mixed stream_keys. Filter records by stream_key first.")

    previous_hash: Optional[str] = records[0].previous_hash
    expected_sequence = records[0].sequence_number

    for record in records:
        if record.sequence_number != expected_sequence:
            return False

        if record.previous_hash != previous_hash:
            return False

        if not verify_event_hash(record):
            return False

        previous_hash = record.event_hash
        expected_sequence += 1

    return True


def create_checkpoint_anchor(records: List[AuditRecord]) -> Dict[str, Any]:
    """
    Produces a summary checkpoint anchor designed for periodic commit to WORM/immutable storage.
    """
    if not records:
        raise ValueError("Cannot create checkpoint anchor for empty record set")

    if not verify_hash_chain(records):
        raise ValueError("Cannot anchor a compromised or invalid audit chain")

    first_record = records[0]
    last_record = records[-1]

    anchor_payload = {
        "stream_key": first_record.stream_key,
        "first_sequence_number": first_record.sequence_number,
        "last_sequence_number": last_record.sequence_number,
        "first_event_id": first_record.event_id,
        "last_event_id": last_record.event_id,
        "checkpoint_hash": last_record.event_hash,
        "record_count": len(records),
        "anchored_at": datetime.now(timezone.utc).isoformat(),
    }

    canonical = json.dumps(anchor_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    anchor_signature = hashlib.sha256(canonical).hexdigest()
    anchor_payload["anchor_signature"] = anchor_signature

    return anchor_payload


def audit_summary(record: AuditRecord) -> Dict[str, Any]:
    """Produces an operational summary without metadata payload."""
    return {
        "event_id": record.event_id,
        "timestamp": record.timestamp,
        "event_name": record.event_name,
        "category": record.category,
        "severity": record.severity,
        "outcome": record.outcome,
        "actor_id": record.actor_id,
        "tenant_id": record.tenant_id,
        "resource_id": record.resource_id,
        "correlation_id": record.correlation_id,
        "request_id": record.request_id,
        "source": record.source,
        "service": record.service,
        "environment": record.environment,
        "stream_key": record.stream_key,
        "sequence_number": record.sequence_number,
    }


__all__ = [
    "AuditCategory",
    "AuditEvent",
    "AuditFailureMode",
    "AuditLogger",
    "AuditOutcome",
    "AuditRecord",
    "AuditSeverity",
    "AuditStorage",
    "AuditWriteResult",
    "AuditWriteStatus",
    "audit_logger",
    "audit_summary",
    "create_checkpoint_anchor",
    "verify_event_hash",
    "verify_hash_chain",
]



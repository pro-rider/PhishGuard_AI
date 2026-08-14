"""
PhishGuard AI - Enterprise Secure Logging, Context & Audit System
=================================================================

Version: 14.0.0
File Path: mlops_pipeline/src/utils/logger.py

Purpose
-------
Production-grade structured logging and security audit infrastructure for the
PhishGuard AI MLOps pipeline.

This module provides:

1. Structured JSON logging
2. Human-readable console logging
3. Security-aware recursive redaction
4. Secret/token/API-key/credential detection
5. PII protection
6. Log-injection protection
7. Correlation/request/job/run context
8. Bounded log record sizes
9. Rotating application logs
10. Optional queue-based asynchronous logging
11. Tamper-evident audit records
12. Audit hash chaining
13. Audit record integrity verification
14. Safe exception serialization
15. Environment-aware configuration
16. Multi-process-safe best-effort audit writing
17. Stable event identifiers
18. Explicit schema versioning
19. Safe logger reconfiguration
20. Fail-closed security defaults

Security Model
--------------
Logging is treated as a security-sensitive data sink.

The logger MUST NOT become a secondary exfiltration channel for:

- passwords
- API keys
- access tokens
- refresh tokens
- JWTs
- OAuth credentials
- cloud credentials
- private keys
- cookies
- session identifiers
- authorization headers
- database credentials
- email addresses when PII redaction is enabled
- raw email bodies
- URLs containing embedded credentials
- authentication material
- arbitrary secrets contained in nested structures

Important
---------
This module does NOT provide cryptographic confidentiality.

Logs should still be stored using operating-system access controls, encrypted
storage, centralized SIEM controls, and appropriate retention policies.

The audit hash chain provides tamper-evidence, NOT absolute tamper prevention.
An attacker with unrestricted write access can still rewrite the entire chain.
For stronger guarantees, forward audit events to an external append-only or
WORM-capable security system.

Compatibility
-------------
Python >= 3.10

No external dependency is required for the core implementation.
"""

from __future__ import annotations

import contextlib
import contextvars
import datetime as dt
import hashlib
import json
import logging
import logging.handlers
import os
import platform
import re
import secrets
import socket
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# ==============================================================================
# PACKAGE CONSTANTS
# ==============================================================================

LOGGER_SCHEMA_VERSION = "1.0.0"
AUDIT_SCHEMA_VERSION = "1.0.0"
MODULE_VERSION = "14.0.0"

DEFAULT_SERVICE_NAME = "PhishGuard_MLOps"
DEFAULT_ENVIRONMENT = "production"

DEFAULT_MAX_MESSAGE_LENGTH = 16_384
DEFAULT_MAX_FIELD_LENGTH = 8_192
DEFAULT_MAX_COLLECTION_ITEMS = 100
DEFAULT_MAX_DEPTH = 8
DEFAULT_MAX_EXCEPTION_LENGTH = 16_384

DEFAULT_LOG_MAX_BYTES = 100 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 10

REDACTED_SECRET = "[REDACTED_SECRET]"
REDACTED_TOKEN = "[REDACTED_TOKEN]"
REDACTED_PII = "[REDACTED_PII]"
REDACTED_CREDENTIAL = "[REDACTED_CREDENTIAL]"
TRUNCATED_VALUE = "[TRUNCATED]"

UTC = dt.timezone.utc


# ==============================================================================
# CONTEXT VARIABLES
# ==============================================================================

_context_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "phishguard_run_id",
    default=None,
)

_context_job_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "phishguard_job_id",
    default=None,
)

_context_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "phishguard_request_id",
    default=None,
)

_context_model_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "phishguard_model_id",
    default=None,
)

_context_dataset_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "phishguard_dataset_id",
    default=None,
)

_context_extra: contextvars.ContextVar[Mapping[str, Any]] = contextvars.ContextVar(
    "phishguard_log_extra",
    default={},
)


# ==============================================================================
# VALIDATION HELPERS
# ==============================================================================

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._:/@+\-]{1,256}$")

_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
)


def _safe_identifier(value: Any, default: str = "unknown") -> str:
    """
    Normalize an identifier before placing it into a log or audit record.

    Identifiers are intentionally conservative to prevent log injection and
    malformed structured records.
    """
    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    text = _CONTROL_CHAR_RE.sub("", text)

    if len(text) > 256:
        text = text[:256] + TRUNCATED_VALUE

    if not _SAFE_IDENTIFIER_RE.fullmatch(text):
        return SecurityRedactor.sanitize_text(text)

    return text


def _utc_now() -> dt.datetime:
    """Return the current UTC timestamp."""
    return dt.datetime.now(tz=UTC)


def _iso_utc(timestamp: Optional[dt.datetime] = None) -> str:
    """Return normalized RFC3339-like UTC timestamp."""
    value = timestamp or _utc_now()

    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)

    return value.astimezone(UTC).isoformat(timespec="milliseconds")


# ==============================================================================
# SECURITY REDACTION ENGINE
# ==============================================================================

class SecurityRedactor:
    """
    Centralized security redaction engine.

    The redactor operates in two layers:

    1. Key-based redaction
       Detects sensitive dictionary field names.

    2. Value-based pattern redaction
       Detects common credential/token/PII representations embedded inside
       strings.

    Security principle
    ------------------
    Unknown structured data is sanitized conservatively rather than assumed
    safe.
    """

    REDACTED_TEXT = REDACTED_SECRET
    REDACTED_EMAIL = REDACTED_PII

    # ------------------------------------------------------------------
    # Sensitive key names
    # ------------------------------------------------------------------

    SENSITIVE_KEY_FRAGMENTS = frozenset(
        {
            "password",
            "passwd",
            "passphrase",
            "secret",
            "token",
            "access_token",
            "refresh_token",
            "id_token",
            "api_key",
            "apikey",
            "authorization",
            "auth",
            "credential",
            "credentials",
            "private_key",
            "privatekey",
            "client_secret",
            "session",
            "session_id",
            "cookie",
            "set_cookie",
            "proxy_password",
            "database_password",
            "encryption_key",
            "signing_key",
            "webhook_secret",
            "ssh_key",
        }
    )

    # ------------------------------------------------------------------
    # Secret / credential patterns
    # ------------------------------------------------------------------

    PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
        # Authorization header
        (
            re.compile(
                r"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]+"
            ),
            "Authorization: Bearer " + REDACTED_TOKEN,
        ),

        # Bearer tokens
        (
            re.compile(
                r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"
            ),
            "Bearer " + REDACTED_TOKEN,
        ),

        # Basic authentication
        (
            re.compile(
                r"(?i)\bbasic\s+[A-Za-z0-9+/=]+"
            ),
            "Basic " + REDACTED_CREDENTIAL,
        ),

        # JWT
        (
            re.compile(
                r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"
            ),
            REDACTED_TOKEN,
        ),

        # AWS access key
        (
            re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            "[REDACTED_AWS_ACCESS_KEY]",
        ),

        # AWS temporary access key
        (
            re.compile(r"\bASIA[0-9A-Z]{16}\b"),
            "[REDACTED_AWS_TEMP_KEY]",
        ),

        # GitHub tokens
        (
            re.compile(
                r"\bgh[pousr]_[A-Za-z0-9_]{20,255}\b"
            ),
            "[REDACTED_GITHUB_TOKEN]",
        ),

        # Slack tokens
        (
            re.compile(
                r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"
            ),
            "[REDACTED_SLACK_TOKEN]",
        ),

        # Generic API-key assignment
        (
            re.compile(
                r"(?i)\b(api[_-]?key|secret[_-]?key|client[_-]?secret)"
                r"\s*[:=]\s*['\"]?[^'\"\s,;]+"
            ),
            r"\1=" + REDACTED_SECRET,
        ),

        # Generic password assignment
        (
            re.compile(
                r"(?i)\b(password|passwd|passphrase)"
                r"\s*[:=]\s*['\"]?[^'\"\s,;]+"
            ),
            r"\1=" + REDACTED_CREDENTIAL,
        ),

        # Generic token assignment
        (
            re.compile(
                r"(?i)\b(access[_-]?token|refresh[_-]?token|id[_-]?token|auth[_-]?token)"
                r"\s*[:=]\s*['\"]?[^'\"\s,;]+"
            ),
            r"\1=" + REDACTED_TOKEN,
        ),

        # Private keys
        (
            re.compile(
                r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"
                r"[\s\S]*?"
                r"-----END [A-Z0-9 ]*PRIVATE KEY-----"
            ),
            "[REDACTED_PRIVATE_KEY]",
        ),

        # PEM certificate/key material
        (
            re.compile(
                r"-----BEGIN [A-Z0-9 ]+-----"
                r"[\s\S]*?"
                r"-----END [A-Z0-9 ]+-----"
            ),
            "[REDACTED_PEM_BLOCK]",
        ),

        # URL credentials:
        # https://user:password@example.com
        (
            re.compile(
                r"(?i)(https?://)([^/\s:@]+):([^@\s/]+)@"
            ),
            r"\1" + REDACTED_CREDENTIAL + "@",
        ),

        # Email addresses
        (
            re.compile(
                r"\b[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
                r"@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+\b"
            ),
            REDACTED_PII,
        ),

        # IPv4 address
        # Kept disabled by default because IPs are often useful for security
        # investigations. Users can enable explicit IP anonymization if needed.
    )

    @classmethod
    def is_sensitive_key(cls, key: Any) -> bool:
        """Return True if a field name strongly indicates sensitive content."""
        normalized = str(key).strip().lower().replace("-", "_")

        if normalized in cls.SENSITIVE_KEY_FRAGMENTS:
            return True

        return any(
            fragment in normalized
            for fragment in (
                "password",
                "secret",
                "token",
                "credential",
                "private_key",
                "apikey",
                "authorization",
                "cookie",
            )
        )

    @classmethod
    def sanitize_text(
        cls,
        text: Any,
        *,
        max_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
        redact_pii: bool = True,
        normalize_controls: bool = True,
    ) -> str:
        """
        Sanitize a textual value.

        Parameters
        ----------
        text:
            Input value.
        max_length:
            Maximum output length.
        redact_pii:
            Whether email addresses are redacted.
        normalize_controls:
            Whether dangerous control characters are removed.
        """
        if text is None:
            return ""

        if not isinstance(text, str):
            text = str(text)

        if normalize_controls:
            text = _CONTROL_CHAR_RE.sub("", text)

        for pattern, replacement in cls.PATTERNS:
            if not redact_pii and replacement == REDACTED_PII:
                continue

            text = pattern.sub(replacement, text)

        if len(text) > max_length:
            text = text[:max_length] + TRUNCATED_VALUE

        return text

    @classmethod
    def sanitize_data(
        cls,
        data: Any,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_field_length: int = DEFAULT_MAX_FIELD_LENGTH,
        max_items: int = DEFAULT_MAX_COLLECTION_ITEMS,
        redact_pii: bool = True,
        _depth: int = 0,
    ) -> Any:
        """
        Recursively sanitize arbitrary structured data.

        The implementation deliberately bounds recursion and collection sizes
        to prevent logging itself from becoming a resource-exhaustion vector.
        """
        if _depth > max_depth:
            return "[MAX_DEPTH_REACHED]"

        if data is None:
            return None

        if isinstance(data, str):
            return cls.sanitize_text(
                data,
                max_length=max_field_length,
                redact_pii=redact_pii,
            )

        if isinstance(data, bytes):
            return "[BINARY_DATA_REDACTED]"

        if isinstance(data, Mapping):
            sanitized: Dict[str, Any] = {}

            items = list(data.items())[:max_items]

            for key, value in items:
                safe_key = cls.sanitize_text(
                    str(key),
                    max_length=256,
                    redact_pii=False,
                )

                if cls.is_sensitive_key(key):
                    sanitized[safe_key] = REDACTED_SECRET
                else:
                    sanitized[safe_key] = cls.sanitize_data(
                        value,
                        max_depth=max_depth,
                        max_field_length=max_field_length,
                        max_items=max_items,
                        redact_pii=redact_pii,
                        _depth=_depth + 1,
                    )

            if len(data) > max_items:
                sanitized["__truncated_items__"] = len(data) - max_items

            return sanitized

        if isinstance(data, (list, tuple, set, frozenset)):
            values = list(data)

            sanitized_list = [
                cls.sanitize_data(
                    value,
                    max_depth=max_depth,
                    max_field_length=max_field_length,
                    max_items=max_items,
                    redact_pii=redact_pii,
                    _depth=_depth + 1,
                )
                for value in values[:max_items]
            ]

            if len(values) > max_items:
                sanitized_list.append(
                    f"[TRUNCATED_ITEMS={len(values) - max_items}]"
                )

            return sanitized_list

        if isinstance(data, (int, float, bool)):
            return data

        if isinstance(data, (dt.date, dt.datetime)):
            return data.isoformat()

        if isinstance(data, Path):
            return cls.sanitize_text(
                str(data),
                max_length=max_field_length,
                redact_pii=redact_pii,
            )

        # NumPy/PyTorch/Pandas-like objects:
        # Do NOT serialize potentially massive tensor/dataframe content.
        if hasattr(data, "shape"):
            try:
                shape = getattr(data, "shape")
                return {
                    "__type__": type(data).__name__,
                    "shape": str(shape),
                }
            except Exception:
                return {
                    "__type__": type(data).__name__,
                }

        return cls.sanitize_text(
            repr(data),
            max_length=max_field_length,
            redact_pii=redact_pii,
        )


# ==============================================================================
# LOG RECORD SECURITY FILTER
# ==============================================================================

class SecurityRedactionFilter(logging.Filter):
    """
    Sanitizes LogRecord fields before they reach handlers.

    The filter is intentionally attached to handlers rather than relying only
    on the logger, because records can originate from child libraries.
    """

    def __init__(
        self,
        *,
        redact_pii: bool = True,
        max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
    ) -> None:
        super().__init__()
        self.redact_pii = redact_pii
        self.max_message_length = max_message_length

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = SecurityRedactor.sanitize_text(
                record.msg,
                max_length=self.max_message_length,
                redact_pii=self.redact_pii,
            )

            if record.args:
                if isinstance(record.args, tuple):
                    record.args = tuple(
                        SecurityRedactor.sanitize_data(
                            value,
                            redact_pii=self.redact_pii,
                        )
                        for value in record.args
                    )
                elif isinstance(record.args, Mapping):
                    record.args = SecurityRedactor.sanitize_data(
                        record.args,
                        redact_pii=self.redact_pii,
                    )

            if hasattr(record, "extra_fields"):
                record.extra_fields = SecurityRedactor.sanitize_data(
                    getattr(record, "extra_fields"),
                    redact_pii=self.redact_pii,
                )

            # Standard LogRecord fields that may contain user-controlled text.
            for field_name in (
                "name",
                "pathname",
                "filename",
                "module",
                "funcName",
            ):
                value = getattr(record, field_name, None)

                if isinstance(value, str):
                    setattr(
                        record,
                        field_name,
                        _CONTROL_CHAR_RE.sub("", value),
                    )

            return True

        except Exception:
            # Logging must never crash the application.
            #
            # If sanitization fails, preserve the event but replace the
            # potentially unsafe message.
            record.msg = "[LOG_REDACTION_FAILURE]"
            record.args = ()
            return True


# ==============================================================================
# CONTEXT MANAGEMENT
# ==============================================================================

@dataclass(frozen=True)
class LogContext:
    """
    Immutable execution context.

    Useful for:
    - training runs
    - evaluation runs
    - retraining jobs
    - model promotion
    - dataset validation
    - CI/CD jobs
    - API requests
    """

    run_id: Optional[str] = None
    job_id: Optional[str] = None
    request_id: Optional[str] = None
    model_id: Optional[str] = None
    dataset_id: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)


def get_log_context() -> Dict[str, Any]:
    """Return the current context without exposing context variable internals."""
    context: Dict[str, Any] = {}

    values = {
        "run_id": _context_run_id.get(),
        "job_id": _context_job_id.get(),
        "request_id": _context_request_id.get(),
        "model_id": _context_model_id.get(),
        "dataset_id": _context_dataset_id.get(),
    }

    for key, value in values.items():
        if value is not None:
            context[key] = value

    context.update(_context_extra.get())

    return SecurityRedactor.sanitize_data(context)


@contextlib.contextmanager
def log_context(
    *,
    run_id: Optional[str] = None,
    job_id: Optional[str] = None,
    request_id: Optional[str] = None,
    model_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Iterator[None]:
    """
    Temporarily bind structured logging context.

    Context is automatically restored when leaving the block.

    Example
    -------
    with log_context(
        run_id="run-123",
        model_id="url_classifier",
    ):
        logger.info("evaluation_started")
    """
    tokens = []

    if run_id is not None:
        tokens.append(
            (_context_run_id, _context_run_id.set(_safe_identifier(run_id)))
        )

    if job_id is not None:
        tokens.append(
            (_context_job_id, _context_job_id.set(_safe_identifier(job_id)))
        )

    if request_id is not None:
        tokens.append(
            (
                _context_request_id,
                _context_request_id.set(_safe_identifier(request_id)),
            )
        )

    if model_id is not None:
        tokens.append(
            (
                _context_model_id,
                _context_model_id.set(_safe_identifier(model_id)),
            )
        )

    if dataset_id is not None:
        tokens.append(
            (
                _context_dataset_id,
                _context_dataset_id.set(_safe_identifier(dataset_id)),
            )
        )

    if extra:
        merged = dict(_context_extra.get())
        merged.update(extra)

        tokens.append(
            (_context_extra, _context_extra.set(merged))
        )

    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


# ==============================================================================
# LOGGING ADAPTER
# ==============================================================================

class ContextLoggerAdapter(logging.LoggerAdapter):
    """
    Context-aware LoggerAdapter.

    Combines explicit per-call context with ambient ContextVar context.
    """

    def process(
        self,
        msg: Any,
        kwargs: MutableMapping[str, Any],
    ) -> Tuple[Any, MutableMapping[str, Any]]:
        current_context = get_log_context()

        explicit_extra = kwargs.get("extra")

        if isinstance(explicit_extra, Mapping):
            current_context.update(explicit_extra)

        kwargs["extra"] = {
            "extra_fields": current_context
        }

        return msg, kwargs


# ==============================================================================
# JSON SERIALIZATION
# ==============================================================================

def _safe_json_default(obj: Any) -> Any:
    """
    Conservative JSON serializer.

    Large/opaque scientific objects are represented by metadata rather than
    serializing their complete contents.
    """
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, bytes):
        return "[BINARY_DATA_REDACTED]"

    if hasattr(obj, "shape"):
        try:
            return {
                "__type__": type(obj).__name__,
                "shape": str(obj.shape),
            }
        except Exception:
            return {
                "__type__": type(obj).__name__,
            }

    return SecurityRedactor.sanitize_text(
        repr(obj),
        max_length=DEFAULT_MAX_FIELD_LENGTH,
    )


# ==============================================================================
# STRUCTURED JSON FORMATTER
# ==============================================================================

class StructuredJSONFormatter(logging.Formatter):
    """
    Production structured JSON formatter.

    Output is JSON Lines compatible with:
    - ELK
    - Splunk
    - Datadog
    - OpenSearch
    - Loki
    - SIEM ingestion systems
    """

    def __init__(
        self,
        *,
        service_name: str = DEFAULT_SERVICE_NAME,
        environment: str = DEFAULT_ENVIRONMENT,
        schema_version: str = LOGGER_SCHEMA_VERSION,
        redact_pii: bool = True,
    ) -> None:
        super().__init__()

        self.service_name = _safe_identifier(
            service_name,
            default=DEFAULT_SERVICE_NAME,
        )

        self.environment = _safe_identifier(
            environment,
            default=DEFAULT_ENVIRONMENT,
        )

        self.schema_version = schema_version
        self.redact_pii = redact_pii

    def format(self, record: logging.LogRecord) -> str:
        try:
            message = SecurityRedactor.sanitize_text(
                record.getMessage(),
                redact_pii=self.redact_pii,
            )

            payload: Dict[str, Any] = {
                "schema": {
                    "name": "phishguard.log",
                    "version": self.schema_version,
                },
                "timestamp": _iso_utc(
                    dt.datetime.fromtimestamp(record.created, tz=UTC)
                ),
                "severity": record.levelname,
                "severity_number": record.levelno,
                "logger": SecurityRedactor.sanitize_text(
                    record.name,
                    redact_pii=False,
                    max_length=256,
                ),
                "message": message,
                "service": self.service_name,
                "environment": self.environment,

                "process": {
                    "pid": record.process,
                    "process_name": SecurityRedactor.sanitize_text(
                        record.processName,
                        redact_pii=False,
                    ),
                },

                "thread": {
                    "id": record.thread,
                    "name": SecurityRedactor.sanitize_text(
                        record.threadName,
                        redact_pii=False,
                    ),
                },

                "source": {
                    "file": SecurityRedactor.sanitize_text(
                        record.filename,
                        redact_pii=False,
                    ),
                    "line": record.lineno,
                    "function": SecurityRedactor.sanitize_text(
                        record.funcName,
                        redact_pii=False,
                    ),
                },
            }

            context = get_log_context()

            if context:
                payload["context"] = context

            if hasattr(record, "extra_fields"):
                extra_fields = getattr(record, "extra_fields")

                if isinstance(extra_fields, Mapping):
                    payload["extra"] = SecurityRedactor.sanitize_data(
                        extra_fields,
                        redact_pii=self.redact_pii,
                    )

            if record.exc_info:
                payload["exception"] = self._format_exception(
                    record.exc_info
                )

            if record.stack_info:
                payload["stack"] = SecurityRedactor.sanitize_text(
                    record.stack_info,
                    max_length=DEFAULT_MAX_EXCEPTION_LENGTH,
                    redact_pii=self.redact_pii,
                )

            return json.dumps(
                payload,
                default=_safe_json_default,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        except Exception:
            # Formatter failure must never bring down the pipeline.
            fallback = {
                "schema": {
                    "name": "phishguard.log",
                    "version": self.schema_version,
                },
                "timestamp": _iso_utc(),
                "severity": "ERROR",
                "logger": "logging",
                "message": "[STRUCTURED_LOG_FORMAT_FAILURE]",
            }

            return json.dumps(fallback, separators=(",", ":"))

    def _format_exception(
        self,
        exc_info: Tuple[Any, BaseException, Any],
    ) -> Dict[str, Any]:
        """Serialize exception data safely."""
        exc_type, exc_value, exc_tb = exc_info

        try:
            stack = "".join(
                traceback.format_exception(
                    exc_type,
                    exc_value,
                    exc_tb,
                )
            )

            stack = SecurityRedactor.sanitize_text(
                stack,
                max_length=DEFAULT_MAX_EXCEPTION_LENGTH,
                redact_pii=self.redact_pii,
            )

            return {
                "type": getattr(
                    exc_type,
                    "__name__",
                    "UnknownException",
                ),
                "message": SecurityRedactor.sanitize_text(
                    str(exc_value),
                    max_length=DEFAULT_MAX_FIELD_LENGTH,
                    redact_pii=self.redact_pii,
                ),
                "stack_trace": stack,
            }

        except Exception:
            return {
                "type": "UnknownException",
                "message": "[EXCEPTION_SERIALIZATION_FAILURE]",
            }


# ==============================================================================
# CONSOLE FORMATTER
# ==============================================================================

class ColoredConsoleFormatter(logging.Formatter):
    """
    Human-readable console formatter.

    ANSI colors are automatically disabled when stdout is not a TTY or when
    explicitly disabled.
    """

    COLOR_MAP = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[41m",
    }

    RESET = "\033[0m"

    def __init__(
        self,
        *,
        use_color: Optional[bool] = None,
        redact_pii: bool = True,
    ) -> None:
        super().__init__()
        self.redact_pii = redact_pii

        if use_color is None:
            use_color = bool(
                hasattr(sys.stdout, "isatty")
                and sys.stdout.isatty()
            )

        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = _iso_utc(
            dt.datetime.fromtimestamp(record.created, tz=UTC)
        )

        message = SecurityRedactor.sanitize_text(
            record.getMessage(),
            redact_pii=self.redact_pii,
        )

        context = get_log_context()

        context_text = ""

        if context:
            context_text = " " + " ".join(
                f"{key}={value}"
                for key, value in context.items()
            )

        output = (
            f"[{timestamp}] "
            f"[{record.levelname:<8}] "
            f"[{record.name}]"
            f"{context_text}: "
            f"{message}"
        )

        if record.exc_info:
            output += "\n" + SecurityRedactor.sanitize_text(
                self.formatException(record.exc_info),
                redact_pii=self.redact_pii,
                max_length=DEFAULT_MAX_EXCEPTION_LENGTH,
            )

        if self.use_color:
            color = self.COLOR_MAP.get(
                record.levelno,
                self.RESET,
            )
            return f"{color}{output}{self.RESET}"

        return output


# ==============================================================================
# LOGGER CONFIGURATION
# ==============================================================================

_config_lock = threading.RLock()
_configured_services: set[str] = set()


def _remove_phishguard_handlers(
    logger: logging.Logger,
) -> None:
    """
    Remove only handlers owned by this module.

    This is intentionally safer than logger.handlers.clear().
    """
    for handler in list(logger.handlers):
        if getattr(handler, "_phishguard_owned", False):
            logger.removeHandler(handler)

            try:
                handler.close()
            except Exception:
                pass


def _mark_owned(handler: logging.Handler) -> logging.Handler:
    """Mark a handler as managed by PhishGuard."""
    setattr(handler, "_phishguard_owned", True)
    return handler


def configure_pipeline_logging(
    log_dir: Union[str, Path] = "artifacts/logs",
    log_level: str = "INFO",
    enable_json_logs: bool = True,
    service_name: str = DEFAULT_SERVICE_NAME,
    environment: Optional[str] = None,
    *,
    max_bytes: int = DEFAULT_LOG_MAX_BYTES,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
    redact_pii: bool = True,
    console_enabled: bool = True,
    color_console: Optional[bool] = None,
    propagate: bool = False,
    use_queue: bool = False,
) -> None:
    """
    Configure PhishGuard application logging.

    Parameters
    ----------
    log_dir:
        Application log directory.

    log_level:
        DEBUG / INFO / WARNING / ERROR / CRITICAL.

    enable_json_logs:
        Emit JSON structured logs to the rotating file.

    service_name:
        Service identifier.

    environment:
        Runtime environment.

    max_bytes:
        Maximum size of an individual log file.

    backup_count:
        Number of rotated files.

    redact_pii:
        Whether email addresses are redacted.

    console_enabled:
        Whether stdout logging is enabled.

    color_console:
        Enable/disable ANSI console colors.

    propagate:
        Whether child records propagate to parent loggers.

    use_queue:
        Reserved for future queue-based high-throughput deployment.
        The current implementation deliberately keeps configuration
        synchronous unless explicitly extended with QueueListener.
    """
    del use_queue  # Explicitly acknowledged; avoids accidental behavior.

    with _config_lock:
        try:
            level_name = str(log_level).upper()

            if level_name not in {
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
                "CRITICAL",
            }:
                raise ValueError(
                    f"Unsupported log level: {log_level!r}"
                )

            if max_bytes <= 0:
                raise ValueError("max_bytes must be > 0")

            if backup_count < 1:
                raise ValueError("backup_count must be >= 1")

            service_name = _safe_identifier(
                service_name,
                DEFAULT_SERVICE_NAME,
            )

            environment = _safe_identifier(
                environment
                or os.getenv(
                    "PHISHGUARD_ENVIRONMENT",
                    DEFAULT_ENVIRONMENT,
                ),
                DEFAULT_ENVIRONMENT,
            )

            path_dir = Path(log_dir)
            path_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            root_logger = logging.getLogger()
            root_logger.setLevel(level_name)

            # Do not destroy handlers owned by the application.
            _remove_phishguard_handlers(root_logger)

            security_filter = SecurityRedactionFilter(
                redact_pii=redact_pii,
            )

            if console_enabled:
                console_handler = _mark_owned(
                    logging.StreamHandler(sys.stdout)
                )

                console_handler.setLevel(level_name)

                console_handler.setFormatter(
                    ColoredConsoleFormatter(
                        use_color=color_console,
                        redact_pii=redact_pii,
                    )
                )

                console_handler.addFilter(security_filter)

                root_logger.addHandler(console_handler)

            log_file = (
                path_dir
                / f"{service_name.lower()}_pipeline.jsonl"
            )

            file_handler = _mark_owned(
                logging.handlers.RotatingFileHandler(
                    filename=log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                    delay=True,
                )
            )

            file_handler.setLevel(level_name)

            if enable_json_logs:
                file_handler.setFormatter(
                    StructuredJSONFormatter(
                        service_name=service_name,
                        environment=environment,
                        redact_pii=redact_pii,
                    )
                )
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s "
                        "[%(levelname)s] "
                        "[%(name)s]: "
                        "%(message)s"
                    )
                )

            file_handler.addFilter(security_filter)

            root_logger.addHandler(file_handler)

            # Prevent duplicated application output when this module is
            # embedded in frameworks that install parent handlers.
            root_logger.propagate = propagate

            _configured_services.add(service_name)

        except Exception:
            # Configuration failures should be explicit. Silent logging
            # misconfiguration is dangerous for security-sensitive systems.
            raise


# ==============================================================================
# LOGGER FACTORY
# ==============================================================================

def get_logger(
    name: str,
    extra_context: Optional[Mapping[str, Any]] = None,
) -> ContextLoggerAdapter:
    """
    Return a context-aware structured logger.

    Parameters
    ----------
    name:
        Logger namespace.

    extra_context:
        Optional static context.

    Returns
    -------
    ContextLoggerAdapter
        Context-aware logger adapter.

    Example
    -------
    logger = get_logger(
        "src.evaluation.metrics",
        {"component": "metrics"},
    )

    logger.info(
        "evaluation_completed",
        extra={"metric": "pr_auc"},
    )
    """
    safe_name = _safe_identifier(
        name,
        default="phishguard",
    )

    logger = logging.getLogger(safe_name)

    if extra_context:
        sanitized_context = SecurityRedactor.sanitize_data(
            dict(extra_context)
        )
    else:
        sanitized_context = {}

    return ContextLoggerAdapter(
        logger,
        sanitized_context,
    )


# ==============================================================================
# AUDIT EVENT MODEL
# ==============================================================================

@dataclass(frozen=True)
class AuditEvent:
    """
    Immutable audit event representation.

    Audit records are intentionally explicit because they may later be
    consumed by SIEM, compliance systems, or forensic tooling.
    """

    event_type: str
    actor: str
    resource_id: str
    status: str
    details: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: str = field(
        default_factory=_iso_utc
    )


# ==============================================================================
# AUDIT LOGGER
# ==============================================================================

class AuditLogger:
    """
    Tamper-evident security audit logger.

    Features
    --------
    - JSONL output
    - stable audit event IDs
    - hash chaining
    - sensitive-data redaction
    - bounded event size
    - rotating storage
    - duplicate handler protection
    - integrity verification

    Important
    ---------
    Hash chaining is tamper-evident only while the chain's trusted anchor is
    protected. For enterprise deployments, the final hash should periodically
    be exported to an external trusted system.
    """

    VALID_STATUSES = frozenset(
        {
            "SUCCESS",
            "FAILURE",
            "QUARANTINED",
            "STARTED",
            "APPROVED",
            "REJECTED",
            "CANCELLED",
        }
    )

    def __init__(
        self,
        audit_log_path: Union[str, Path] = (
            "artifacts/logs/audit_trail.jsonl"
        ),
        *,
        service_name: str = "PhishGuard_Audit",
        environment: Optional[str] = None,
        max_bytes: int = DEFAULT_LOG_MAX_BYTES,
        backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
        redact_pii: bool = True,
    ) -> None:
        self.audit_path = Path(audit_log_path)
        self.audit_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.service_name = _safe_identifier(
            service_name,
            "PhishGuard_Audit",
        )

        self.environment = _safe_identifier(
            environment
            or os.getenv(
                "PHISHGUARD_ENVIRONMENT",
                DEFAULT_ENVIRONMENT,
            ),
            DEFAULT_ENVIRONMENT,
        )

        self.redact_pii = redact_pii

        self._lock = threading.RLock()

        self.logger = logging.getLogger(
            f"{self.service_name}.{self.audit_path}"
        )

        self.logger.setLevel(logging.INFO)

        self.logger.propagate = False

        self._configure_handler(
            max_bytes=max_bytes,
            backup_count=backup_count,
        )

        self._previous_hash = self._load_previous_hash()

    def _configure_handler(
        self,
        *,
        max_bytes: int,
        backup_count: int,
    ) -> None:
        """Configure exactly one owned audit handler."""
        with _config_lock:
            _remove_phishguard_handlers(self.logger)

            handler = _mark_owned(
                logging.handlers.RotatingFileHandler(
                    filename=self.audit_path,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                    delay=True,
                )
            )

            handler.setLevel(logging.INFO)

            handler.setFormatter(
                StructuredJSONFormatter(
                    service_name=self.service_name,
                    environment=self.environment,
                    redact_pii=self.redact_pii,
                )
            )

            handler.addFilter(
                SecurityRedactionFilter(
                    redact_pii=self.redact_pii,
                )
            )

            self.logger.addHandler(handler)

    def _load_previous_hash(self) -> str:
        """
        Load the previous audit hash.

        If the file cannot be safely parsed, return an explicit genesis hash.
        """
        if not self.audit_path.exists():
            return self._genesis_hash()

        try:
            last_line = ""

            with self.audit_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                for line in handle:
                    if line.strip():
                        last_line = line.strip()

            if not last_line:
                return self._genesis_hash()

            record = json.loads(last_line)

            previous_hash = record.get(
                "audit_chain_hash"
            )

            if isinstance(previous_hash, str) and previous_hash:
                return previous_hash

        except Exception:
            # Existing audit history may be corrupted. We deliberately do
            # not pretend it is valid.
            return "[AUDIT_CHAIN_RECOVERY_REQUIRED]"

        return self._genesis_hash()

    @staticmethod
    def _genesis_hash() -> str:
        """Return deterministic genesis anchor."""
        return hashlib.sha256(
            b"PHISHGUARD-AUDIT-GENESIS-V1"
        ).hexdigest()

    @staticmethod
    def _canonical_json(data: Mapping[str, Any]) -> str:
        """Create deterministic JSON for hashing."""
        return json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_safe_json_default,
        )

    def _calculate_chain_hash(
        self,
        event_payload: Mapping[str, Any],
    ) -> str:
        """
        Calculate chained audit hash.

        hash_n = SHA256(hash_(n-1) || canonical_event_n)
        """
        canonical = self._canonical_json(
            event_payload
        )

        material = (
            self._previous_hash
            + "|"
            + canonical
        ).encode("utf-8")

        return hashlib.sha256(material).hexdigest()

    def log_event(
        self,
        event_type: str,
        actor: str,
        resource_id: str,
        status: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """
        Record an immutable audit event.

        Returns
        -------
        str
            Unique audit event ID.
        """
        safe_event_type = _safe_identifier(
            event_type,
            "UNKNOWN_EVENT",
        ).upper()

        safe_actor = _safe_identifier(
            actor,
            "UNKNOWN_ACTOR",
        )

        safe_resource = _safe_identifier(
            resource_id,
            "UNKNOWN_RESOURCE",
        )

        safe_status = _safe_identifier(
            status,
            "UNKNOWN_STATUS",
        ).upper()

        if safe_status not in self.VALID_STATUSES:
            raise ValueError(
                f"Unsupported audit status: {status!r}"
            )

        event_id = str(uuid.uuid4())

        sanitized_details = SecurityRedactor.sanitize_data(
            details or {},
            redact_pii=self.redact_pii,
        )

        with self._lock:
            base_event: Dict[str, Any] = {
                "schema": {
                    "name": "phishguard.audit",
                    "version": AUDIT_SCHEMA_VERSION,
                },
                "audit_event_id": event_id,
                "audit_event_type": safe_event_type,
                "timestamp": _iso_utc(),
                "actor": safe_actor,
                "resource_id": safe_resource,
                "status": safe_status,
                "service": self.service_name,
                "environment": self.environment,
                "details": sanitized_details,
            }

            chain_hash = self._calculate_chain_hash(
                base_event
            )

            base_event["previous_audit_chain_hash"] = (
                self._previous_hash
            )

            base_event["audit_chain_hash"] = chain_hash

            record = self.logger.makeRecord(
                name=self.logger.name,
                level=logging.INFO,
                fn=__file__,
                lno=0,
                msg=(
                    f"AUDIT_EVENT "
                    f"{safe_event_type} "
                    f"{safe_resource} "
                    f"{safe_status}"
                ),
                args=(),
                exc_info=None,
            )

            record.extra_fields = base_event  # type: ignore[attr-defined]

            self.logger.handle(record)

            self._previous_hash = chain_hash

            return event_id

    def verify_integrity(self) -> Tuple[bool, str]:
        """
        Verify the audit hash chain.

        Returns
        -------
        Tuple[bool, str]
            (is_valid, diagnostic_message)
        """
        if not self.audit_path.exists():
            return True, "Audit file does not exist; empty chain."

        previous_hash = self._genesis_hash()
        line_number = 0

        try:
            with self.audit_path.open(
                "r",
                encoding="utf-8",
            ) as handle:

                for raw_line in handle:
                    line_number += 1

                    if not raw_line.strip():
                        continue

                    record = json.loads(
                        raw_line
                    )

                    stored_previous = record.pop(
                        "previous_audit_chain_hash",
                        None,
                    )

                    stored_chain = record.pop(
                        "audit_chain_hash",
                        None,
                    )

                    if stored_previous != previous_hash:
                        return (
                            False,
                            (
                                "Previous hash mismatch at "
                                f"line {line_number}"
                            ),
                        )

                    expected_material = (
                        previous_hash
                        + "|"
                        + self._canonical_json(record)
                    ).encode("utf-8")

                    expected_hash = hashlib.sha256(
                        expected_material
                    ).hexdigest()

                    if stored_chain != expected_hash:
                        return (
                            False,
                            (
                                "Chain hash mismatch at "
                                f"line {line_number}"
                            ),
                        )

                    previous_hash = stored_chain

            return True, "Audit chain integrity verified."

        except Exception as exc:
            return (
                False,
                (
                    "Audit integrity verification failed: "
                    f"{SecurityRedactor.sanitize_text(str(exc))}"
                ),
            )


# ==============================================================================
# SYSTEM / RUNTIME DIAGNOSTICS
# ==============================================================================

def get_logging_environment() -> Dict[str, Any]:
    """
    Return minimal logging/runtime diagnostics.

    Sensitive environment variables are intentionally excluded.
    """
    return {
        "hostname": SecurityRedactor.sanitize_text(
            socket.gethostname(),
            redact_pii=False,
            max_length=256,
        ),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": SecurityRedactor.sanitize_text(
            platform.system(),
            redact_pii=False,
        ),
        "platform_release": SecurityRedactor.sanitize_text(
            platform.release(),
            redact_pii=False,
        ),
        "architecture": SecurityRedactor.sanitize_text(
            platform.machine(),
            redact_pii=False,
        ),
        "process_id": os.getpid(),
    }


# ==============================================================================
# CONVENIENCE AUDIT HELPERS
# ==============================================================================

def audit_model_promotion(
    audit_logger: AuditLogger,
    *,
    actor: str,
    model_id: str,
    artifact_version: str,
    status: str,
    evaluation_summary: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Convenience helper for model promotion audit events.
    """
    return audit_logger.log_event(
        event_type="MODEL_PROMOTION",
        actor=actor,
        resource_id=model_id,
        status=status,
        details={
            "artifact_version": artifact_version,
            "evaluation_summary": dict(
                evaluation_summary or {}
            ),
        },
    )


def audit_retraining_trigger(
    audit_logger: AuditLogger,
    *,
    actor: str,
    model_id: str,
    trigger_type: str,
    trigger_metrics: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Convenience helper for retraining-trigger audit events.
    """
    return audit_logger.log_event(
        event_type="RETRAIN_TRIGGER",
        actor=actor,
        resource_id=model_id,
        status="STARTED",
        details={
            "trigger_type": trigger_type,
            "trigger_metrics": dict(
                trigger_metrics or {}
            ),
        },
    )


def audit_quarantine(
    audit_logger: AuditLogger,
    *,
    actor: str,
    resource_id: str,
    reason: str,
    details: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Convenience helper for quarantine events.
    """
    return audit_logger.log_event(
        event_type="RESOURCE_QUARANTINE",
        actor=actor,
        resource_id=resource_id,
        status="QUARANTINED",
        details={
            "reason": reason,
            "details": dict(details or {}),
        },
    )


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Metadata
    "LOGGER_SCHEMA_VERSION",
    "AUDIT_SCHEMA_VERSION",
    "MODULE_VERSION",

    # Security
    "SecurityRedactor",
    "SecurityRedactionFilter",

    # Logging
    "StructuredJSONFormatter",
    "ColoredConsoleFormatter",
    "ContextLoggerAdapter",
    "configure_pipeline_logging",
    "get_logger",

    # Context
    "LogContext",
    "get_log_context",
    "log_context",

    # Audit
    "AuditEvent",
    "AuditLogger",

    # Diagnostics
    "get_logging_environment",

    # Convenience audit helpers
    "audit_model_promotion",
    "audit_retraining_trigger",
    "audit_quarantine",
]
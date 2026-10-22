"""
PhishGuard_AI - Production Structured Logging Engine

Features:
- Single-source-of-truth PII and credential sanitization via security.pii_redactor.PIIRedactor.
- Strict event name and severity level validation for SIEM integration.
- Protected event helpers (namespaced details/context) preventing keyword collision crashes.
- Global service metadata enrichment (service, environment, version).
- Non-destructive root logger handler registration preserving third-party handlers.
- Explicit, zero-side-effect initialization designed for application boot sequences.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any, Dict, Set

import structlog

from config.internal.settings import LogFormat, LogLevel, settings
from security.pii_redactor import PIIRedactor

# ============================================================================
# VALIDATION PATTERNS & CONSTANTS
# ============================================================================

# Enforces safe, normalized, lower-case event names (1-128 chars, no CRLF or whitespace)
EVENT_NAME_REGEX = re.compile(r"^[a-z0-9_.-]{1,128}$")

# Strict whitelist of allowed security event severity levels
ALLOWED_SECURITY_LEVELS: Set[str] = {
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
}

# Reserved top-level structlog key names to prevent pipeline collision
RESERVED_KEYS: Set[str] = {
    "event",
    "level",
    "timestamp",
    "logger",
    "severity",
    "service",
    "environment",
    "version",
}

# ============================================================================
# STRUCTLOG PROCESSORS & ENRICHMENT
# ============================================================================


def add_service_metadata_processor(
    logger: Any,
    method_name: str,
    event_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Injects process-level metadata for distributed observability."""
    event_dict["service"] = "phishguard-ai"
    event_dict["environment"] = settings.ENVIRONMENT.value
    event_dict["version"] = settings.VERSION
    return event_dict


def redact_pii_processor(
    logger: Any,
    method_name: str,
    event_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Structlog processor that delegates deep sanitization to PIIRedactor.
    Skipped if PII redaction is explicitly disabled in settings.
    """
    if not settings.ENABLE_PII_REDACTION:
        return event_dict

    return {
        str(k): PIIRedactor.sanitize_value(str(k), v)
        for k, v in event_dict.items()
    }


# ============================================================================
# LOGGING CONFIGURATION ENGINE
# ============================================================================

_LOGGING_CONFIGURED = False


def configure_logging() -> None:
    """
    Configures application-wide structured logging for structlog & stdlib.
    Must be called explicitly from core/entrypoint.py or main application boot.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    # Determine log level from settings
    log_level_str = (
        settings.LOG_LEVEL.value
        if isinstance(settings.LOG_LEVEL, LogLevel)
        else str(settings.LOG_LEVEL)
    )
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # Pipeline Processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        add_service_metadata_processor,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        redact_pii_processor,
    ]

    # Select Renderer (JSON in Production, Colorized Console in Development)
    if settings.LOG_FORMAT == LogFormat.JSON:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=settings.is_development())

    # Configure Structlog Engine
    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Standard Library Processor Formatter Bridge
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Register Root Handler safely without clobbering third-party handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    has_phishguard_handler = any(
        isinstance(h, logging.StreamHandler)
        and getattr(h, "_phishguard_handler", False)
        for h in root_logger.handlers
    )

    if not has_phishguard_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler._phishguard_handler = True
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    # Third-Party Logger Suppression & Propagation Controls
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.setLevel(logging.WARNING)
    uvicorn_access.propagate = False

    uvicorn_error = logging.getLogger("uvicorn.error")
    uvicorn_error.setLevel(logging.INFO)

    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


def get_logger(name: str = "phishguard") -> structlog.stdlib.BoundLogger:
    """
    Retrieves a structured logger instance.
    Example:
        logger = get_logger(__name__)
        logger.info("scan.started", scan_id=scan_id)
    """
    if not _LOGGING_CONFIGURED:
        configure_logging()

    return structlog.get_logger(name)


# ============================================================================
# STRUCTURED EVENT HELPERS
# ============================================================================


def log_security_event(
    event_name: str,
    severity: str,
    details: Dict[str, Any],
    logger_name: str = "phishguard.security",
) -> None:
    """
    Standardized helper for SIEM security events (SSRF blocks, auth failures).

    Validates event_name and severity, and namespaces arbitrary details dictionary
    to eliminate keyword collision risk.
    """
    normalized_event = event_name.lower().strip()
    if not EVENT_NAME_REGEX.match(normalized_event):
        raise ValueError(
            f"Invalid security event_name '{event_name}'. "
            "Must match regex ^[a-z0-9_.-]{1,128}$"
        )

    upper_severity = severity.upper().strip()
    if upper_severity not in ALLOWED_SECURITY_LEVELS:
        raise ValueError(
            f"Invalid security severity '{severity}'. "
            f"Must be one of {sorted(ALLOWED_SECURITY_LEVELS)}"
        )

    if not isinstance(details, dict):
        raise ValueError("Security event 'details' must be a dictionary.")

    logger = get_logger(logger_name)
    log_func_map = {
        "DEBUG": logger.debug,
        "INFO": logger.info,
        "WARNING": logger.warning,
        "ERROR": logger.error,
        "CRITICAL": logger.critical,
    }
    log_func = log_func_map[upper_severity]

    log_func(
        f"security_event.{normalized_event}",
        severity=upper_severity,
        details=details,
    )


def log_scan_event(
    event_name: str,
    scan_id: str,
    target_type: str,
    status: str,
    logger_name: str = "phishguard.scan",
    **extra_context: Any,
) -> None:
    """
    Standardized helper for core scan pipeline events.

    Namespaces extra_context under 'context' to avoid keyword argument collisions.
    """
    normalized_event = event_name.lower().strip()
    if not EVENT_NAME_REGEX.match(normalized_event):
        raise ValueError(
            f"Invalid scan event_name '{event_name}'. "
            "Must match regex ^[a-z0-9_.-]{1,128}$"
        )

    logger = get_logger(logger_name)
    logger.info(
        f"scan.{normalized_event}",
        scan_id=scan_id,
        target_type=target_type,
        status=status,
        context=extra_context,
    )

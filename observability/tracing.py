"""
PhishGuard_AI - Distributed Tracing Bootstrap

OpenTelemetry-compatible tracing setup used for distributed observability of
the detection pipeline.

Security / reliability principles:
- Instrumentation is optional and fail-open: absent SDKs never break the app.
- Span names and attribute keys/values are validated and bounded.
- High-cardinality or sensitive values are excluded by default.
- Exporters are configured to be non-blocking (async batch) where available.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Mapping, Optional

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
        ConsoleSpanExporter,
    )

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _otel_trace = None
    _OTEL_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

MAX_SPAN_NAME_LENGTH = 128
MAX_ATTRIBUTE_KEY_LENGTH = 64
MAX_ATTRIBUTE_VALUE_LENGTH = 2048
MAX_ATTRIBUTES_PER_SPAN = 64
SERVICE_NAME_DEFAULT = "phishguard-ai"

_SPAN_NAME_REGEX = re.compile(r"^[A-Za-z0-9_./:-]{1,128}$")
_ATTR_KEY_REGEX = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class TracingError(RuntimeError):
    """Base error for tracing setup."""


# ============================================================================
# TRACING BOOTSTRAP
# ============================================================================


class TracingManager:
    """
    Bounded, fail-open tracing manager.

    Wraps the OpenTelemetry global tracer provider. When the SDK is absent all
    operations degrade to safe no-ops so instrumentation never blocks scanning.
    """

    def __init__(
        self,
        *,
        service_name: str = SERVICE_NAME_DEFAULT,
        enabled: bool = True,
    ) -> None:
        self._service_name = service_name
        self._enabled = enabled and _OTEL_AVAILABLE
        self._initialized = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def configure(
        self,
        *,
        resource_attributes: Optional[Mapping[str, Any]] = None,
        use_console_exporter: bool = False,
        batch: bool = True,
    ) -> None:
        """Initialize the global tracer provider once."""
        if self._initialized:
            return
        self._initialized = True

        if not self._enabled:
            if _otel_trace is not None:
                # Fall back to a no-op provider so ``tracer()`` still works.
                _otel_trace.set_tracer_provider(_otel_trace.ProxyTracerProvider())
            return

        attributes: Dict[str, Any] = {SERVICE_NAME: self._service_name}
        if resource_attributes:
            attributes.update(dict(resource_attributes))

        resource = Resource.create(attributes)
        provider = TracerProvider(resource=resource)

        if use_console_exporter:
            processor = (
                BatchSpanProcessor(ConsoleSpanExporter())
                if batch
                else SimpleSpanProcessor(ConsoleSpanExporter())
            )
            provider.add_span_processor(processor)

        _otel_trace.set_tracer_provider(provider)

    def tracer(self, name: str = "phishguard-ai"):
        """Return a bounded tracer for ``name``."""
        if not self._enabled or _otel_trace is None:
            return None
        try:
            return _otel_trace.get_tracer(name)
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # Span helpers
    # ------------------------------------------------------------------

    def start_span(self, tracer: Any, name: str, attributes: Optional[Mapping[str, Any]] = None):
        """Start a span with a validated name and bounded attributes."""
        if tracer is None or not _OTEL_AVAILABLE:
            return None
        if not isinstance(name, str) or not _SPAN_NAME_REGEX.match(name):
            name = "span"
        try:
            return tracer.start_as_current_span(
                name, attributes=self._sanitize_attributes(attributes)
            )
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _sanitize_attributes(
        attributes: Optional[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not attributes:
            return None
        result: Dict[str, Any] = {}
        for key, value in attributes.items():
            if len(result) >= MAX_ATTRIBUTES_PER_SPAN:
                break
            if not isinstance(key, str) or not _ATTR_KEY_REGEX.match(key):
                continue
            if len(key) > MAX_ATTRIBUTE_KEY_LENGTH:
                continue
            if value is None:
                continue
            if isinstance(value, str) and len(value) > MAX_ATTRIBUTE_VALUE_LENGTH:
                value = value[:MAX_ATTRIBUTE_VALUE_LENGTH]
            if isinstance(value, (str, int, float, bool)):
                result[key] = value
        return result or None


# ============================================================================
# SINGLETON
# ============================================================================

tracing = TracingManager()


def configure_tracing(**kwargs: Any) -> None:
    """One-shot configuration helper for application boot."""
    tracing.configure(**kwargs)


__all__ = [
    "TracingManager",
    "TracingError",
    "tracing",
    "configure_tracing",
    "MAX_ATTRIBUTES_PER_SPAN",
    "MAX_ATTRIBUTE_VALUE_LENGTH",
]

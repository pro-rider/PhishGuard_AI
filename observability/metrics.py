"""
PhishGuard_AI - Metrics Emission Layer

Prometheus-compatible metrics registry used across the detection pipeline.

Security / reliability principles:
- Metric names and labels are validated to prevent Prometheus injection and
  unbounded cardinality.
- Label values are bounded in length and count.
- Emitting metrics never raises: instrumentation is fail-open.
- Help strings are fixed at registration to avoid drift.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    CollectorRegistry = None
    Counter = None
    Gauge = None
    Histogram = None
    generate_latest = None
    _PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

DEFAULT_NAMESPACE = "phishguard"
MAX_NAME_LENGTH = 128
MAX_LABEL_NAME_LENGTH = 64
MAX_LABEL_VALUE_LENGTH = 256
MAX_LABELS_PER_METRIC = 16
DEFAULT_HISTOGRAM_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 60.0)

_NAME_REGEX = None  # validated at write time by prometheus; we pre-validate below
import re

_METRIC_NAME_REGEX = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_NAME_REGEX = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class MetricsError(RuntimeError):
    """Base error for metrics construction."""


# ============================================================================
# METRICS MANAGER
# ============================================================================


class MetricsManager:
    """
    Thread-safe, namespaced metrics registry.

    Uses a dedicated :class:`CollectorRegistry` so instrumented metrics are
    isolated from any pre-existing Prometheus default registry.
    """

    def __init__(
        self,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        registry: Any = None,
    ) -> None:
        if not _PROMETHEUS_AVAILABLE:
            raise MetricsError("prometheus_client is not installed.")
        self._namespace = self._validate_namespace(namespace)
        self._registry = registry or CollectorRegistry(auto_describe=False)
        self._lock = None  # registry operations are internally safe

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_namespace(namespace: str) -> str:
        if not namespace or not _METRIC_NAME_REGEX.match(namespace):
            raise MetricsError(f"Invalid metric namespace: {namespace!r}")
        return namespace

    @staticmethod
    def _validate_labels(labels: Sequence[str]) -> List[str]:
        if len(labels) > MAX_LABELS_PER_METRIC:
            raise MetricsError("Too many labels per metric.")
        normalized: List[str] = []
        for label in labels:
            if not isinstance(label, str) or not _LABEL_NAME_REGEX.match(label):
                raise MetricsError(f"Invalid label name: {label!r}")
            if len(label) > MAX_LABEL_NAME_LENGTH:
                raise MetricsError(f"Label name too long: {label!r}")
            if label in normalized:
                raise MetricsError(f"Duplicate label: {label!r}")
            normalized.append(label)
        return normalized

    @staticmethod
    def _validate_label_values(
        labels: Sequence[str], values: Mapping[str, str]
    ) -> List[str]:
        sanitized: List[str] = []
        for label in labels:
            raw = values.get(label, "")
            if not isinstance(raw, str):
                raw = str(raw)
            if len(raw) > MAX_LABEL_VALUE_LENGTH:
                raw = raw[:MAX_LABEL_VALUE_LENGTH]
            sanitized.append(raw)
        return sanitized

    # ------------------------------------------------------------------
    # Metric constructors
    # ------------------------------------------------------------------

    def counter(
        self, name: str, help_text: str, labels: Sequence[str] = ()
    ) -> Callable[..., None]:
        """Create and register a Counter; returns an increment callable."""
        metric_name = self._full_name(name)
        label_list = self._validate_labels(labels)
        counter = Counter(
            metric_name,
            help_text,
            labelnames=label_list,
            registry=self._registry,
        )

        def increment(*, amount: float = 1.0, **label_values: Any) -> None:
            try:
                vals = self._validate_label_values(label_list, label_values)
                if vals:
                    counter.labels(*vals).inc(amount)
                else:
                    counter.inc(amount)
            except Exception:  # noqa: BLE001 - instrumentation is fail-open
                logger.warning("Counter increment failed.", exc_info=True)

        return increment

    def gauge(
        self, name: str, help_text: str, labels: Sequence[str] = ()
    ) -> Callable[..., None]:
        """Create and register a Gauge; returns a set callable."""
        metric_name = self._full_name(name)
        label_list = self._validate_labels(labels)
        gauge = Gauge(
            metric_name,
            help_text,
            labelnames=label_list,
            registry=self._registry,
        )

        def set_value(*, value: float, **label_values: Any) -> None:
            try:
                vals = self._validate_label_values(label_list, label_values)
                if vals:
                    gauge.labels(*vals).set(value)
                else:
                    gauge.set(value)
            except Exception:  # noqa: BLE001
                logger.warning("Gauge set failed.", exc_info=True)

        return set_value

    def histogram(
        self,
        name: str,
        help_text: str,
        labels: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_HISTOGRAM_BUCKETS,
    ) -> Callable[..., None]:
        """Create and register a Histogram; returns an observe callable."""
        metric_name = self._full_name(name)
        label_list = self._validate_labels(labels)
        histogram = Histogram(
            metric_name,
            help_text,
            labelnames=label_list,
            buckets=list(buckets),
            registry=self._registry,
        )

        def observe(*, value: float, **label_values: Any) -> None:
            try:
                vals = self._validate_label_values(label_list, label_values)
                if vals:
                    histogram.labels(*vals).observe(value)
                else:
                    histogram.observe(value)
            except Exception:  # noqa: BLE001
                logger.warning("Histogram observe failed.", exc_info=True)

        return observe

    # ------------------------------------------------------------------
    # Timers
    # ------------------------------------------------------------------

    def timer(self, name: str, help_text: str, labels: Sequence[str] = ()):
        """Context-manager timer that records seconds into a histogram."""

        class _Timer:
            def __init__(self, observe: Callable[..., None]) -> None:
                self._observe = observe
                self._start = 0.0

            def __enter__(self) -> "_Timer":
                self._start = time.perf_counter()
                return self

            def __exit__(self, *exc: Any) -> None:
                self._observe(value=time.perf_counter() - self._start)

        return _Timer(self.histogram(name, help_text, labels))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def render(self) -> bytes:
        """Render all registered metrics in Prometheus text format."""
        return generate_latest(self._registry)

    def snapshot(self) -> Dict[str, Any]:
        """Return a lightweight, JSON-safe metric listing."""
        result: Dict[str, Any] = {"namespace": self._namespace, "metrics": []}
        for metric in self._registry.collect():
            result["metrics"].append(metric.name)
        return result

    def _full_name(self, name: str) -> str:
        if not isinstance(name, str) or not _METRIC_NAME_REGEX.match(name):
            raise MetricsError(f"Invalid metric name: {name!r}")
        if len(name) > MAX_NAME_LENGTH:
            raise MetricsError(f"Metric name too long: {name!r}")
        return f"{self._namespace}_{name}"


# ============================================================================
# SINGLETON
# ============================================================================

metrics = MetricsManager()


__all__ = [
    "MetricsManager",
    "MetricsError",
    "metrics",
    "DEFAULT_HISTOGRAM_BUCKETS",
    "MAX_LABEL_VALUE_LENGTH",
]

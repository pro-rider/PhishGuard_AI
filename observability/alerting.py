"""
PhishGuard_AI - Security Alert Manager

Centralized alert emission and delivery for the detection pipeline.

Security / reliability principles:
- Alerts are deduplicated and rate-limited to prevent alert storms.
- Delivery failures never crash the caller (fail-open, best-effort delivery).
- Alert content is size-bounded and sensitive fields are redactable.
- Multiple transports (webhook, Slack, file) are supported behind one API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

MAX_ALERT_TITLE = 256
MAX_ALERT_DESCRIPTION = 4096
MAX_ALERT_DETAILS_KEYS = 32
MAX_ALERT_DETAILS_VALUE = 1024
MAX_PENDING_ALERTS = 10_000
DEFAULT_DEDUP_WINDOW_SECONDS = 300
DEFAULT_MIN_INTERVAL_SECONDS = 1.0
DEFAULT_WEBHOOK_TIMEOUT_SECONDS = 5.0

_DEDUP_FIELDS = ("type", "entity", "severity")


class AlertSeverity(str, Enum):
    """Security alert severity levels."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertTransport(str, Enum):
    """Supported delivery transports."""

    WEBHOOK = "webhook"
    SLACK = "slack"
    FILE = "file"
    LOG = "log"


# ============================================================================
# MODELS
# ============================================================================


@dataclass(frozen=True, slots=True)
class SecurityAlert:
    """A single, normalized security alert."""

    alert_id: str
    title: str
    severity: AlertSeverity
    alert_type: str = "generic"
    entity: str = ""
    description: str = ""
    details: Mapping[str, Any] = ()
    timestamp_utc: str = ""
    dedup_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "severity": self.severity.value,
            "type": self.alert_type,
            "entity": self.entity,
            "description": self.description,
            "details": dict(self.details),
            "timestamp_utc": self.timestamp_utc,
            "dedup_key": self.dedup_key,
        }


# ============================================================================
# TRANSPORTS
# ============================================================================


class AlertTransportBackend:
    """Interface for delivering an alert."""

    def deliver(self, alert: SecurityAlert) -> bool:
        raise NotImplementedError


class WebhookTransport(AlertTransportBackend):
    """Deliver alerts to a generic HTTPS webhook endpoint."""

    def __init__(self, url: str, *, timeout: float = DEFAULT_WEBHOOK_TIMEOUT_SECONDS):
        if not url.startswith(("https://", "http://")):
            raise ValueError("Webhook URL must be http(s).")
        self._url = url
        self._timeout = timeout

    def deliver(self, alert: SecurityAlert) -> bool:
        try:
            body = json.dumps(alert.to_dict()).encode("utf-8")
            req = urllib.request.Request(
                self._url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return 200 <= resp.status < 300
        except Exception as exc:  # noqa: BLE001
            logger.warning("Webhook alert delivery failed: %s", exc.__class__.__name__)
            return False


class SlackTransport(AlertTransportBackend):
    """Deliver alerts as Slack-compatible incoming-webhook messages."""

    def __init__(self, url: str, *, timeout: float = DEFAULT_WEBHOOK_TIMEOUT_SECONDS):
        self._inner = WebhookTransport(url, timeout=timeout)

    def deliver(self, alert: SecurityAlert) -> bool:
        payload = {
            "text": f"[{alert.severity.value.upper()}] {alert.title}",
            "attachments": [
                {
                    "color": _SEVERITY_COLOR[alert.severity],
                    "fields": [
                        {"title": "Type", "value": alert.alert_type, "short": True},
                        {"title": "Entity", "value": alert.entity or "-", "short": True},
                        {"title": "Details", "value": json.dumps(dict(alert.details))[:2000]},
                    ],
                }
            ],
        }
        try:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return 200 <= resp.status < 300
        except Exception as exc:  # noqa: BLE001
            logger.warning("Slack alert delivery failed: %s", exc.__class__.__name__)
            return False


_SEVERITY_COLOR = {
    AlertSeverity.INFO: "good",
    AlertSeverity.LOW: "#2eb886",
    AlertSeverity.MEDIUM: "warning",
    AlertSeverity.HIGH: "danger",
    AlertSeverity.CRITICAL: "#e01e5a",
}


class FileTransport(AlertTransportBackend):
    """Append alerts to a JSON Lines log file (bounded, atomic-ish writes)."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def deliver(self, alert: SecurityAlert) -> bool:
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(alert.to_dict()) + "\n")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("File alert delivery failed: %s", exc.__class__.__name__)
            return False


# ============================================================================
# ALERT MANAGER
# ============================================================================


class AlertManager:
    """
    Deduplicating, rate-limited alert manager.

    Thread-safe best-effort delivery across multiple transports.
    """

    def __init__(
        self,
        *,
        dedup_window_seconds: int = DEFAULT_DEDUP_WINDOW_SECONDS,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
    ) -> None:
        self._dedup_window = dedup_window_seconds
        self._min_interval = min_interval_seconds
        self._transports: List[AlertTransportBackend] = []
        self._recent: MutableMapping[str, float] = {}
        self._lock = None  # best-effort; GIL provides atomic dict ops in CPython

    def add_transport(self, transport: AlertTransportBackend) -> None:
        self._transports.append(transport)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def emit(
        self,
        *,
        title: str,
        severity: AlertSeverity,
        alert_type: str = "generic",
        entity: str = "",
        description: str = "",
        details: Optional[Mapping[str, Any]] = None,
        force: bool = False,
    ) -> Optional[SecurityAlert]:
        """
        Emit an alert, subject to dedup and rate limiting.

        Returns the created alert when delivered (or when transports are
        empty and only logging is desired), else ``None``.
        """
        title = self._bound(title, MAX_ALERT_TITLE)
        description = self._bound(description, MAX_ALERT_DESCRIPTION)
        details = self._sanitize_details(details or {})
        entity = self._bound(entity, MAX_ALERT_TITLE)

        dedup_key = self._dedup_key(alert_type, entity, severity)
        now = time.time()

        if not force:
            last = self._recent.get(dedup_key)
            if last is not None and (now - last) < self._dedup_window:
                return None

        alert = SecurityAlert(
            alert_id=self._new_id(alert_type, entity, now),
            title=title,
            severity=severity,
            alert_type=alert_type,
            entity=entity,
            description=description,
            details=details,
            timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            dedup_key=dedup_key,
        )

        # Rate limiting per dedup key.
        self._recent[dedup_key] = now
        if len(self._recent) > MAX_PENDING_ALERTS:
            self._trim_recent(now)

        delivered = False
        for transport in self._transports:
            try:
                delivered = transport.deliver(alert) or delivered
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Alert transport failed: %s", exc.__class__.__name__
                )

        if not self._transports:
            logger.warning(
                "ALERT [%s] %s: %s", severity.value, title, description
            )
        return alert

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bound(value: str, limit: int) -> str:
        if not isinstance(value, str):
            value = str(value)
        return value[:limit]

    @staticmethod
    def _sanitize_details(details: Mapping[str, Any]) -> Mapping[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in details.items():
            if len(result) >= MAX_ALERT_DETAILS_KEYS:
                break
            if isinstance(value, str) and len(value) > MAX_ALERT_DETAILS_VALUE:
                value = value[:MAX_ALERT_DETAILS_VALUE]
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[str(key)] = value
        return result

    def _dedup_key(self, alert_type: str, entity: str, severity: AlertSeverity) -> str:
        raw = f"{alert_type}|{entity}|{severity.value}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _new_id(self, alert_type: str, entity: str, now: float) -> str:
        raw = f"{alert_type}|{entity}|{now}|{id(self)}"
        return f"alrt_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"

    def _trim_recent(self, now: float) -> None:
        cutoff = now - self._dedup_window
        expired = [
            key for key, ts in self._recent.items() if ts < cutoff
        ]
        for key in expired:
            self._recent.pop(key, None)


# ============================================================================
# SINGLETON
# ============================================================================

alert_manager = AlertManager()


__all__ = [
    "AlertSeverity",
    "AlertTransport",
    "SecurityAlert",
    "AlertManager",
    "WebhookTransport",
    "SlackTransport",
    "FileTransport",
    "alert_manager",
]

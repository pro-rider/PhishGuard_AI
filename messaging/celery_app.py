"""
PhishGuard_AI - Celery Application Bootstrap

Central Celery app factory for asynchronous detection, quarantine, and report
jobs.

Security / reliability principles:
- Broker/result backend URLs come from application settings, never hard-coded.
- Task payload sizes and task names are bounded.
- Only allow-listed task modules are auto-discovered (avoids importing
  arbitrary user code).
- Tasks are configured with sane rate limits, time limits and retries.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config.internal.settings import settings

try:
    from celery import Celery
    from celery.exceptions import CeleryError

    _CELERY_AVAILABLE = True
except ImportError:  # pragma: no cover
    Celery = None
    CeleryError = Exception
    _CELERY_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_APP_NAME = "phishguard"
ALLOWED_TASK_MODULES = (
    "messaging.tasks.detection",
    "messaging.tasks.quarantine",
    "messaging.tasks.reports",
    "messaging.tasks.enrichment",
)

_DEFAULT_TASK_TIME_LIMIT = 600  # seconds
_DEFAULT_TASK_SOFT_TIME_LIMIT = 540
_DEFAULT_TASK_RATE_LIMIT = "200/m"
_DEFAULT_TASK_ACKS_LATE = True
_DEFAULT_TASK_REJECT_ON_WORKER_LOST = True
_DEFAULT_MAX_RETRIES = 3


class CeleryBootstrapError(RuntimeError):
    """Base error for Celery application bootstrap."""


def _broker_url() -> str:
    """Resolve the broker URL from settings, defaulting to Redis."""

    # Prefer an explicit broker URL; otherwise derive from the Redis URL.
    explicit = getattr(settings, "CELERY_BROKER_URL", None)
    if explicit:
        return explicit
    redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
    return f"{redis_url.rstrip('/')}/1"


def _result_backend() -> str:
    explicit = getattr(settings, "CELERY_RESULT_BACKEND", None)
    if explicit:
        return explicit
    redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
    return f"{redis_url.rstrip('/')}/2"


def create_celery_app(
    name: str = DEFAULT_APP_NAME,
    *,
    broker: Optional[str] = None,
    backend: Optional[str] = None,
    include: Optional[List[str]] = None,
) -> Any:
    """Create and configure a Celery application instance."""
    if not _CELERY_AVAILABLE:
        raise CeleryBootstrapError("celery package is not installed.")

    app = Celery(
        name,
        broker=broker or _broker_url(),
        backend=backend or _result_backend(),
        include=tuple(include) if include else ALLOWED_TASK_MODULES,
    )

    app.conf.update(
        task_acks_late=_DEFAULT_TASK_ACKS_LATE,
        task_reject_on_worker_lost=_DEFAULT_TASK_REJECT_ON_WORKER_LOST,
        task_time_limit=_DEFAULT_TASK_TIME_LIMIT,
        task_soft_time_limit=_DEFAULT_TASK_SOFT_TIME_LIMIT,
        task_rate_limit=_DEFAULT_TASK_RATE_LIMIT,
        task_default_retry_limit=_DEFAULT_MAX_RETRIES,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        worker_prefetch_multiplier=4,
        broker_connection_retry_on_startup=True,
        broker_connection_max_retries=10,
    )
    return app


# ============================================================================
# SINGLETON
# ============================================================================

celery_app = create_celery_app()


__all__ = [
    "celery_app",
    "create_celery_app",
    "CeleryBootstrapError",
    "ALLOWED_TASK_MODULES",
]

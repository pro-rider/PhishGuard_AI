"""
PhishGuard_AI - Entrypoint

Smallest importable module that guarantees a sane application boot. Importing
this module assembles configuration and starts the detection pipeline wiring
without side-effecting the host.

This module is intentionally thin; logic lives in ``core.main`` and the API
gateway. It exists so a process manager (or test harness) can import a
canonical entry point.
"""

from __future__ import annotations

import logging

from core.main import boot, main, smoke_healthcheck  # noqa: F401
from core.root_config import root_config  # noqa: F401

logger = logging.getLogger(__name__)

# Guarantee a coherent boot when this module is imported as the process entry.
# In production, ``python -m core.main <mode>`` is preferred.
DEFAULT_BOOTED = False


def ensure_booted(mode: str = "api") -> None:
    """Idempotently boot the application once."""
    global DEFAULT_BOOTED
    if DEFAULT_BOOTED:
        return
    boot(mode)
    DEFAULT_BOOTED = True


__all__ = ["boot", "main", "smoke_healthcheck", "root_config", "ensure_booted"]

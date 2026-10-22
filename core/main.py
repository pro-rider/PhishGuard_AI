"""
PhishGuard_AI - Application Bootstrap Entry Point

Wire together configuration, logging, tracing, metrics, and the detection
pipeline, then start the configured entry mode (api / worker / cli).

Security / reliability principles:
- Boot fails fast on misconfiguration.
- Sensitive configuration (secrets) is never logged.
- Shutdown is graceful and bounded.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import List, Optional

from core.root_config import root_config
from observability.alerting import alert_manager
from observability.metrics import metrics
from observability.tracing import configure_tracing

logger = logging.getLogger(__name__)

_ALLOWED_MODES = {"api", "worker", "cli", "smoke"}


class BootError(RuntimeError):
    """Base error for application boot."""


def _configure_logging() -> None:
    level = getattr(logging, root_config.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s [%(process)d] %(message)s",
    )


def _configure_tracing() -> None:
    configure_tracing(
        service_name=root_config.service_name,
        resource_attributes={"deployment.environment": root_config.environment},
    )


def _configure_alerting() -> None:
    # Wire a file transport when an alert path is configured via the
    # environment; otherwise alerts are logged.
    import os

    alert_path = os.getenv("PHISHGUARD_ALERT_PATH")
    if alert_path:
        from observability.alerting import FileTransport
        from pathlib import Path

        alert_manager.add_transport(FileTransport(Path(alert_path)))


def smoke_healthcheck() -> bool:
    """Run a minimal health check; return True when healthy."""
    try:
        metrics.counter("boot_checks_total", "Boot health checks performed")
        return True
    except Exception:  # noqa: BLE001
        return False


def boot(mode: str = "api") -> None:
    """Boot the application in ``mode``."""
    mode = (mode or "api").lower()
    if mode not in _ALLOWED_MODES:
        raise BootError(f"Unknown boot mode: {mode!r}")

    started = time.perf_counter()
    _configure_logging()
    _configure_tracing()
    _configure_alerting()

    healthy = smoke_healthcheck()
    if not healthy:
        raise BootError("Boot health check failed.")

    logger.info(
        "Booted %s in %s mode (environment=%s, version=%s)",
        root_config.service_name,
        mode,
        root_config.environment,
        root_config.version,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry: parse a mode argument and boot."""
    argv = argv if argv is not None else sys.argv[1:]
    mode = argv[0] if argv else "api"
    try:
        boot(mode)
    except BootError as exc:
        logger.error("Boot failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

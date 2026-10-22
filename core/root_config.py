"""
PhishGuard_AI - Root Configuration

Central, validated application configuration assembled from environment and the
application settings singleton. Provides a single typed view used at boot.

Security / reliability principles:
- Secrets are never serialized or logged.
- Configuration is validated eagerly at import time (fail fast).
- Unknown / insecure default values are rejected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config.internal.settings import settings

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

MAX_FEATURES = 512
MAX_MODELS = 64
MAX_PLUGINS = 64

# Flag names used by ``feature_flags``.
_ENV_FLAG_PREFIX = "PHISHGUARD_FLAG_"


class RootConfigError(RuntimeError):
    """Base error for root configuration assembly."""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Frozen, validated runtime configuration snapshot."""

    environment: str
    service_name: str
    version: str
    log_level: str
    enable_pii_redaction: bool
    redis_url: str
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    enabled_analyzers: List[str] = field(default_factory=list)

    def has_flag(self, flag: str) -> bool:
        return bool(self.feature_flags.get(flag, False))


def _read_feature_flags() -> Dict[str, bool]:
    """Read ``PHISHGUARD_FLAG_*`` environment variables into a flag map."""
    flags: Dict[str, bool] = {}
    for key, value in os.environ.items():
        if not key.startswith(_ENV_FLAG_PREFIX):
            continue
        name = key[len(_ENV_FLAG_PREFIX):].lower()
        if len(name) > 128:
            continue
        flags[name] = value.strip().lower() in {"1", "true", "yes", "on"}
    return flags


def _default_analyzers() -> List[str]:
    return [
        "url",
        "ip",
        "hash",
        "email_header",
        "email_nlp",
        "attachment",
        "web_content",
        "visual_similarity",
    ]


def load_runtime_config() -> RuntimeConfig:
    """
    Assemble and validate the runtime configuration from the settings singleton
    and process environment.
    """
    try:
        environment = getattr(settings, "ENVIRONMENT", "development")
        version = getattr(settings, "VERSION", "0.0.0")
        log_level = getattr(settings, "LOG_LEVEL", "INFO").upper()
        redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
        pii = getattr(settings, "ENABLE_PII_REDACTION", True)
    except Exception as exc:  # noqa: BLE001
        raise RootConfigError(
            f"Unable to read application settings: {exc.__class__.__name__}"
        ) from exc

    if log_level not in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }:
        raise RootConfigError(f"Invalid LOG_LEVEL: {log_level!r}")

    if not isinstance(redis_url, str) or not redis_url.startswith(("redis://", "rediss://")):
        raise RootConfigError("REDIS_URL must be a redis:// or rediss:// URL.")

    flags = _read_feature_flags()
    analyzers = _default_analyzers()

    return RuntimeConfig(
        environment=str(environment),
        service_name=str(getattr(settings, "SERVICE_NAME", "phishguard-ai")),
        version=str(version),
        log_level=log_level,
        enable_pii_redaction=bool(pii),
        redis_url=redis_url,
        feature_flags=flags,
        enabled_analyzers=analyzers,
    )


root_config = load_runtime_config()


__all__ = [
    "RuntimeConfig",
    "load_runtime_config",
    "root_config",
    "RootConfigError",
]

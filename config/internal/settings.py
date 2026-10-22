
"""
PhishGuard_AI - Production Application Settings

Centralized, strongly typed configuration for the application.

Security principles:
- No production secrets are hardcoded.
- Secrets use SecretStr.
- Production configuration is validated at startup.
- Database/Redis credentials are URL encoded.
- Dangerous configuration combinations are rejected.
- Network, upload, archive, and browser limits are centralized.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote, urlparse

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================================
# PROJECT PATHS
# ============================================================================

# Anchor project root deterministically regardless of execution CWD
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = PROJECT_ROOT / ".env"


# ============================================================================
# ENUMS
# ============================================================================


class EnvironmentOption(str, Enum):
    """Supported application environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Supported logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Supported logging output formats."""

    TEXT = "text"
    JSON = "json"


# ============================================================================
# SETTINGS
# ============================================================================


class Settings(BaseSettings):
    """
    Hardened root configuration for PhishGuard_AI.

    Configuration is loaded from:
    1. Environment variables
    2. .env file (anchored to PROJECT_ROOT)
    3. Class defaults

    Environment variables take precedence over .env values.
    """

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
        validate_assignment=True,
    )

    # ========================================================================
    # APPLICATION
    # ========================================================================

    APP_NAME: str = Field(
        default="PhishGuard_AI",
        min_length=1,
        max_length=100,
    )

    APP_VERSION: str = Field(
        default="1.0.0",
        min_length=1,
        max_length=30,
    )

    CONFIG_VERSION: str = "1"

    ENVIRONMENT: EnvironmentOption = EnvironmentOption.DEVELOPMENT

    DEBUG: bool = False

    # ========================================================================
    # SECURITY / APPLICATION SECRET
    # ========================================================================

    SECRET_KEY: SecretStr = Field(
        default=SecretStr(
            "CHANGE_ME_DEV_ONLY_SECRET_DO_NOT_USE_IN_PRODUCTION"
        ),
        min_length=32,
    )

    # JWT configuration
    JWT_ALGORITHM: str = "HS256"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=5,
        le=24 * 60,
    )

    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        ge=1,
        le=90,
    )

    # ========================================================================
    # API SERVER
    # ========================================================================

    API_V1_STR: str = "/api/v1"

    HOST: str = "0.0.0.0"

    PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
    )

    ALLOWED_HOSTS: list[str] = Field(
        default_factory=lambda: [
            "localhost",
            "127.0.0.1",
        ]
    )

    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8501",
        ]
    )

    CORS_ALLOW_CREDENTIALS: bool = False

    # ========================================================================
    # DATABASE
    # ========================================================================

    POSTGRES_USER: str = Field(
        default="phishguard",
        min_length=1,
        max_length=100,
    )

    POSTGRES_PASSWORD: SecretStr = Field(
        default=SecretStr("CHANGE_ME_DEV_DATABASE_PASSWORD"),
        min_length=8,
    )

    POSTGRES_SERVER: str = "localhost"

    POSTGRES_PORT: int = Field(
        default=5432,
        ge=1,
        le=65535,
    )

    POSTGRES_DB: str = Field(
        default="phishguard_db",
        min_length=1,
        max_length=100,
    )

    SQLALCHEMY_DATABASE_URI: str | None = None

    DATABASE_POOL_SIZE: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    DATABASE_MAX_OVERFLOW: int = Field(
        default=20,
        ge=0,
        le=100,
    )

    DATABASE_POOL_TIMEOUT: int = Field(
        default=30,
        ge=1,
        le=300,
    )

    DATABASE_POOL_RECYCLE: int = Field(
        default=1800,
        ge=60,
        le=86400,
    )

    DATABASE_ECHO: bool = False

    # ========================================================================
    # REDIS
    # ========================================================================

    REDIS_HOST: str = "localhost"

    REDIS_PORT: int = Field(
        default=6379,
        ge=1,
        le=65535,
    )

    REDIS_DB: int = Field(
        default=0,
        ge=0,
        le=15,
    )

    REDIS_PASSWORD: SecretStr | None = None

    REDIS_URL: str | None = None

    REDIS_SSL: bool = False

    REDIS_SOCKET_TIMEOUT: int = Field(
        default=5,
        ge=1,
        le=60,
    )

    REDIS_CONNECT_TIMEOUT: int = Field(
        default=5,
        ge=1,
        le=60,
    )

    # ========================================================================
    # CELERY / WORKERS
    # ========================================================================

    CELERY_BROKER_URL: str | None = None

    CELERY_RESULT_BACKEND: str | None = None

    CELERY_WORKER_CONCURRENCY: int = Field(
        default=4,
        ge=1,
        le=64,
    )

    CELERY_TASK_TIME_LIMIT: int = Field(
        default=300,
        ge=10,
        le=3600,
    )

    CELERY_TASK_SOFT_TIME_LIMIT: int = Field(
        default=270,
        ge=5,
        le=3500,
    )

    # ========================================================================
    # THREAT INTELLIGENCE
    # ========================================================================

    VIRUSTOTAL_API_KEY: SecretStr | None = None

    ABUSEIPDB_API_KEY: SecretStr | None = None

    URLHAUS_API_KEY: SecretStr | None = None

    THREAT_INTEL_TIMEOUT_SECONDS: int = Field(
        default=10,
        ge=1,
        le=60,
    )

    THREAT_INTEL_MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
    )

    # ========================================================================
    # API GATEWAY AUTHENTICATION
    # ========================================================================

    # Comma-separated static API keys accepted by AuthGuardMiddleware. Business
    # endpoints (/scan, /cases, /quarantine, ...) reject requests without a
    # matching key sent in the X-API-Key header (or Authorization: Bearer).
    # Leave unset to run in locked-down mode (only exempt paths are reachable).
    API_KEYS: str | None = Field(
        default=None,
        description="Comma-separated API keys accepted by the auth guard.",
    )

    # Optional: comma-separated entries of "username:password:role1,role2:tenant"
    # (entries separated by ';'). When set, JWT auth via POST /auth/token is
    # enabled and principals carry roles (admin/analyst/viewer) + a tenant scope.
    # Leave unset to keep open mode (no auth) alongside the static API keys.
    AUTH_USERS: str | None = Field(
        default=None,
        description="Semicolon-separated user registry enabling JWT + RBAC.",
    )

    # Extra offline analyzers (e.g. url_parser) that add structural signals but
    # change aggregation. Keep OFF by default until verdict baselines are
    # re-verified for your corpus. Enabling is safe (graceful degradation), but
    # re-baseline phishing vs benign verdicts before promoting to production.
    ENABLE_EXTRA_ANALYZERS: bool = False

    # ========================================================================
    # SSRF / OUTBOUND NETWORK SECURITY
    # ========================================================================

    ENFORCE_SSRF_PROTECTION: bool = True

    ALLOW_PRIVATE_NETWORK_ACCESS: bool = False

    ALLOW_LOOPBACK_ACCESS: bool = False

    ALLOW_LINK_LOCAL_ACCESS: bool = False

    ALLOW_FILE_SCHEME: bool = False

    ALLOW_DATA_SCHEME: bool = False

    MAX_REDIRECTS: int = Field(
        default=5,
        ge=0,
        le=20,
    )

    OUTBOUND_REQUEST_TIMEOUT_SECONDS: int = Field(
        default=15,
        ge=1,
        le=120,
    )

    DNS_RESOLUTION_TIMEOUT_SECONDS: int = Field(
        default=5,
        ge=1,
        le=60,
    )

    # ========================================================================
    # FILE / ATTACHMENT SECURITY
    # ========================================================================

    MAX_ATTACHMENT_SIZE_MB: int = Field(
        default=25,
        ge=1,
        le=500,
    )

    MAX_EMAIL_SIZE_MB: int = Field(
        default=50,
        ge=1,
        le=500,
    )

    MAX_FILES_PER_SCAN: int = Field(
        default=20,
        ge=1,
        le=100,
    )

    # Archive protection
    ZIP_BOMB_MAX_RATIO: int = Field(
        default=100,
        ge=1,
        le=10000,
    )

    ZIP_MAX_UNCOMPRESSED_SIZE_MB: int = Field(
        default=250,
        ge=1,
        le=2000,
    )

    ZIP_MAX_RECURSION_DEPTH: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    MAX_ARCHIVE_FILES: int = Field(
        default=1000,
        ge=1,
        le=100000,
    )

    # ========================================================================
    # WEB ANALYSIS / BROWSER SANDBOX
    # ========================================================================

    BROWSER_ENABLED: bool = True

    BROWSER_HEADLESS: bool = True

    BROWSER_TIMEOUT_SECONDS: int = Field(
        default=30,
        ge=5,
        le=180,
    )

    BROWSER_MAX_PAGES: int = Field(
        default=5,
        ge=1,
        le=50,
    )

    BROWSER_SCREENSHOT_ENABLED: bool = True

    BROWSER_MAX_SCREENSHOT_SIZE_MB: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    BROWSER_ALLOW_JAVASCRIPT: bool = True

    # ========================================================================
    # OBJECT STORAGE
    # ========================================================================

    S3_ENDPOINT_URL: str | None = None

    S3_BUCKET_NAME: str = "phishguard-artifacts"

    S3_REGION: str = "us-east-1"

    AWS_ACCESS_KEY_ID: str | None = None

    AWS_SECRET_ACCESS_KEY: SecretStr | None = None

    S3_SERVER_SIDE_ENCRYPTION: bool = True

    OBJECT_RETENTION_DAYS: int = Field(
        default=30,
        ge=1,
        le=3650,
    )

    # ========================================================================
    # OBSERVABILITY
    # ========================================================================

    LOG_LEVEL: LogLevel = LogLevel.INFO

    LOG_FORMAT: LogFormat = LogFormat.JSON

    ENABLE_PII_REDACTION: bool = True

    PROMETHEUS_METRICS_ENABLED: bool = True

    OTEL_ENABLED: bool = True

    OTEL_SERVICE_NAME: str = "phishguard-ai"

    OTEL_EXPORTER_ENDPOINT: str | None = None

    # ========================================================================
    # RATE LIMITING
    # ========================================================================

    RATE_LIMIT_ENABLED: bool = True

    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=60,
        ge=1,
        le=10000,
    )

    RATE_LIMIT_SCAN_REQUESTS_PER_MINUTE: int = Field(
        default=20,
        ge=1,
        le=1000,
    )

    # ========================================================================
    # SCAN LIMITS
    # ========================================================================

    MAX_URL_LENGTH: int = Field(
        default=4096,
        ge=256,
        le=32768,
    )

    MAX_EMAIL_BODY_SIZE_MB: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    MAX_IOCS_PER_SCAN: int = Field(
        default=1000,
        ge=10,
        le=100000,
    )

    # ========================================================================
    # MODEL / ML
    # ========================================================================

    MODEL_REGISTRY_PATH: str = "config/external/model_registry"

    MODEL_CACHE_DIR: str = ".cache/models"

    ML_INFERENCE_TIMEOUT_SECONDS: int = Field(
        default=10,
        ge=1,
        le=120,
    )

    ML_MODEL_VERSION: str = "1.0.0"

    ML_DRIFT_MONITORING_ENABLED: bool = True

    # ========================================================================
    # MODEL DECISION THRESHOLDS
    # ========================================================================

    PHISHING_THRESHOLD: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
    )

    MALICIOUS_THRESHOLD: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
    )

    # ========================================================================
    # FIELD VALIDATORS
    # ========================================================================

    @field_validator("APP_NAME", "APP_VERSION", "POSTGRES_SERVER", "POSTGRES_DB")
    @classmethod
    def reject_empty_strings(cls, value: str) -> str:
        """Prevent silently accepting empty configuration values or whitespace."""
        value = value.strip()
        if not value:
            raise ValueError("Configuration value cannot be empty.")
        return value

    @field_validator("API_V1_STR")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        """Ensure API prefix has a valid format."""
        value = value.strip()
        if not value.startswith("/"):
            raise ValueError("API_V1_STR must start with '/'.")
        if value != "/" and value.endswith("/"):
            value = value.rstrip("/")
        return value

    @field_validator("ALLOWED_HOSTS")
    @classmethod
    def normalize_allowed_hosts(cls, value: list[str]) -> list[str]:
        """Normalize host configuration."""
        cleaned = []
        for host in value:
            host = host.strip().lower()
            if host:
                cleaned.append(host)
        if not cleaned:
            raise ValueError("ALLOWED_HOSTS must contain at least one host.")
        return list(dict.fromkeys(cleaned))

    @field_validator("CORS_ORIGINS")
    @classmethod
    def normalize_cors_origins(cls, value: list[str]) -> list[str]:
        """Normalize CORS origins."""
        cleaned = []
        for origin in value:
            origin = origin.strip().rstrip("/")
            if origin:
                cleaned.append(origin)
        return list(dict.fromkeys(cleaned))

    @field_validator("S3_ENDPOINT_URL", "OTEL_EXPORTER_ENDPOINT")
    @classmethod
    def validate_optional_urls(cls, value: str | None) -> str | None:
        """Validate optional URL fields if provided."""
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Endpoints must specify a valid http or https scheme.")
        return value

    # ========================================================================
    # CROSS-FIELD VALIDATION
    # ========================================================================

    @model_validator(mode="after")
    def validate_security_configuration(self) -> "Settings":
        """
        Perform cross-field security validation.
        Intentionally executed during startup so dangerous configuration fails fast.
        """

        # PRODUCTION SECRET VALIDATION
        insecure_secret = "CHANGE_ME_DEV_ONLY_SECRET_DO_NOT_USE_IN_PRODUCTION"

        if self.ENVIRONMENT in {
            EnvironmentOption.STAGING,
            EnvironmentOption.PRODUCTION,
        }:
            if self.SECRET_KEY.get_secret_value() == insecure_secret:
                raise ValueError("SECRET_KEY must be explicitly configured in staging/production.")

            if len(self.SECRET_KEY.get_secret_value()) < 32:
                raise ValueError("SECRET_KEY must contain at least 32 characters.")

        # DEBUG MUST NEVER BE ENABLED IN PRODUCTION
        if self.ENVIRONMENT == EnvironmentOption.PRODUCTION and self.DEBUG:
            raise ValueError("DEBUG=True is forbidden in production.")

        # PII REDACTION
        if self.ENVIRONMENT == EnvironmentOption.PRODUCTION and not self.ENABLE_PII_REDACTION:
            raise ValueError("PII redaction cannot be disabled in production.")

        # SSRF PROTECTION
        if self.ENVIRONMENT == EnvironmentOption.PRODUCTION:
            if not self.ENFORCE_SSRF_PROTECTION:
                raise ValueError("SSRF protection must be enabled in production.")

            if self.ALLOW_PRIVATE_NETWORK_ACCESS:
                raise ValueError("Private network access must not be enabled for production URL scanning.")

        # CORS ORIGINS
        if self.ENVIRONMENT == EnvironmentOption.PRODUCTION:
            for origin in self.CORS_ORIGINS:
                if "*" in origin:
                    raise ValueError("Wildcard CORS origins are forbidden in production.")

        # DATABASE PASSWORD
        if self.ENVIRONMENT in {
            EnvironmentOption.STAGING,
            EnvironmentOption.PRODUCTION,
        }:
            db_password = self.POSTGRES_PASSWORD.get_secret_value()
            if db_password in {"", "CHANGE_ME_DEV_DATABASE_PASSWORD"}:
                raise ValueError("A real POSTGRES_PASSWORD is required in staging/production.")

        # REDIS SSL & PASSWORD
        if self.ENVIRONMENT == EnvironmentOption.PRODUCTION:
            if not self.REDIS_SSL:
                raise ValueError("REDIS_SSL must be enabled in production.")
            if not self.REDIS_PASSWORD or not self.REDIS_PASSWORD.get_secret_value():
                raise ValueError("REDIS_PASSWORD is required in production.")

        # BROWSER SANDBOX
        if self.ENVIRONMENT == EnvironmentOption.PRODUCTION:
            if self.BROWSER_ENABLED and not self.BROWSER_HEADLESS:
                raise ValueError("Production browser scanning must run headless.")

        # ML THRESHOLDS
        if self.MALICIOUS_THRESHOLD < self.PHISHING_THRESHOLD:
            raise ValueError("MALICIOUS_THRESHOLD cannot be lower than PHISHING_THRESHOLD.")

        # CELERY TIME LIMITS
        if self.CELERY_TASK_SOFT_TIME_LIMIT >= self.CELERY_TASK_TIME_LIMIT:
            raise ValueError("CELERY_TASK_SOFT_TIME_LIMIT must be lower than CELERY_TASK_TIME_LIMIT.")

        # SSRF + BROWSER
        if self.BROWSER_ENABLED and not self.ENFORCE_SSRF_PROTECTION:
            if self.ENVIRONMENT != EnvironmentOption.DEVELOPMENT:
                raise ValueError("Browser scanning requires SSRF protection outside development.")

        # BUILD DATABASE URL
        if not self.SQLALCHEMY_DATABASE_URI:
            self.SQLALCHEMY_DATABASE_URI = self._build_database_url()

        # BUILD REDIS URL
        if not self.REDIS_URL:
            self.REDIS_URL = self._build_redis_url()

        # CELERY CONFIG
        if not self.CELERY_BROKER_URL:
            self.CELERY_BROKER_URL = self.REDIS_URL

        if not self.CELERY_RESULT_BACKEND:
            self.CELERY_RESULT_BACKEND = self.REDIS_URL

        return self

    # ========================================================================
    # URL BUILDERS
    # ========================================================================

    def _build_database_url(self) -> str:
        """
        Safely construct the async PostgreSQL connection URL.
        Credentials are URL encoded to prevent special characters.
        """
        username = quote(self.POSTGRES_USER, safe="")
        password = quote(self.POSTGRES_PASSWORD.get_secret_value(), safe="")

        return (
            f"postgresql+asyncpg://"
            f"{username}:{password}"
            f"@{self.POSTGRES_SERVER}:"
            f"{self.POSTGRES_PORT}/"
            f"{self.POSTGRES_DB}"
        )

    def _build_redis_url(self) -> str:
        """
        Safely construct the Redis connection URL.
        """
        scheme = "rediss" if self.REDIS_SSL else "redis"

        if self.REDIS_PASSWORD and self.REDIS_PASSWORD.get_secret_value():
            password = quote(self.REDIS_PASSWORD.get_secret_value(), safe="")
            return f"{scheme}://:{password}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

        return f"{scheme}://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ========================================================================
    # SAFE HELPERS
    # ========================================================================

    def is_production(self) -> bool:
        """Return True when running in production."""
        return self.ENVIRONMENT == EnvironmentOption.PRODUCTION

    def is_testing(self) -> bool:
        """Return True when running in test mode."""
        return self.ENVIRONMENT == EnvironmentOption.TESTING

    def is_development(self) -> bool:
        """Return True when running locally."""
        return self.ENVIRONMENT == EnvironmentOption.DEVELOPMENT

    def get_sanitized_config(self) -> Dict[str, Any]:
        """
        Return a safe representation of current settings with secrets redacted.
        Safe for logging and diagnostic dumps.
        """
        data = self.model_dump()
        data["SQLALCHEMY_DATABASE_URI"] = "[REDACTED]"
        data["REDIS_URL"] = "[REDACTED]"
        data["CELERY_BROKER_URL"] = "[REDACTED]"
        data["CELERY_RESULT_BACKEND"] = "[REDACTED]"

        for key, val in data.items():
            if isinstance(val, SecretStr):
                data[key] = "[REDACTED]"

        return data


# ============================================================================
# GLOBAL SETTINGS SINGLETON
# ============================================================================

settings = Settings()

"""
PhishGuard_AI - Production Secrets Management Client

Centralized secrets abstraction for the PhishGuard_AI security layer.

Responsibilities:
    - Retrieve secrets from configured providers (AWS Secrets Manager & Environment).
    - Prevent secret values from entering log messages, metrics, exceptions, or tracebacks.
    - Provide thread-safe synchronous and non-blocking asynchronous access.
    - Bounded in-memory TTL caching with explicit rotation and invalidation support.
    - Strict identifier validation, payload size enforcement, and JSON parsing guardrails.
    - Fail-closed behavior for missing, malformed, or unauthorized production secrets.

Security Principles:
    - Zero secret exposure in logs, exceptions, or repr/str outputs.
    - Strict environment isolation with no silent substitution of missing secrets.
    - Defense-in-depth against payload memory consumption and malformed inputs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol

from pydantic import SecretStr

try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover
    boto3 = None
    BotoConfig = None
    BotoCoreError = Exception
    ClientError = Exception


logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS & BOUNDS
# ============================================================================

MAX_SECRET_NAME_LENGTH = 256
MAX_SECRET_VALUE_LENGTH = 1024 * 1024  # 1 MiB max payload
MAX_JSON_DEPTH = 5                     # Max recursion depth for JSON secrets

DEFAULT_CACHE_TTL_SECONDS = 300
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 5

MAX_CACHE_ENTRIES = 256

SECRET_NAME_REGEX = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._/@+=:-]{0,255}$"
)

ENV_SECRET_PREFIX = "PHISHGUARD_SECRET_"


# ============================================================================
# ENUMERATIONS
# ============================================================================


class SecretProvider(str, Enum):
    """Supported secret backends."""

    ENVIRONMENT = "environment"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"


class SecretFormat(str, Enum):
    """Expected format of secret content."""

    TEXT = "text"
    JSON = "json"


# ============================================================================
# EXCEPTIONS
# ============================================================================


class SecretsError(RuntimeError):
    """
    Base exception for all secrets-management failures.

    VULNERABILITY PREVENTION: Never include secret values in exception messages.
    """


class SecretNotFoundError(SecretsError):
    """Raised when a requested secret does not exist in the provider."""


class SecretProviderError(SecretsError):
    """Raised when the underlying secrets manager client or transport fails."""


class SecretValidationError(SecretsError):
    """Raised when a secret identifier or payload fails schema or size validation."""


class SecretConfigurationError(SecretsError):
    """Raised when the secrets client is misconfigured or lacks prerequisites."""


# ============================================================================
# CACHE ENTRY
# ============================================================================


@dataclass(slots=True)
class _CacheEntry:
    """
    Internal bounded TTL cache entry.

    The secret value is wrapped in SecretStr to prevent accidental print/repr leaks.
    """

    value: SecretStr
    expires_at: float

    def is_expired(self, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.monotonic()
        return now >= self.expires_at


# ============================================================================
# PROVIDER PROTOCOL
# ============================================================================


class SecretProviderClient(Protocol):
    """
    Interface definition for secret provider implementations.
    """

    def get_secret(self, name: str) -> SecretStr:
        ...

    async def get_secret_async(self, name: str) -> SecretStr:
        ...

    def delete_secret(self, name: str) -> None:
        ...


# ============================================================================
# ENVIRONMENT PROVIDER
# ============================================================================


class EnvironmentSecretProvider:
    """
    Development and local test secret provider.

    Maps secret names to environment variables with PHISHGUARD_SECRET_ prefix.
    Example: 'DATABASE_PASSWORD' -> 'PHISHGUARD_SECRET_DATABASE_PASSWORD'
    """

    def __init__(
        self,
        *,
        prefix: str = ENV_SECRET_PREFIX,
        allow_empty: bool = False,
    ) -> None:
        self.prefix = prefix
        self.allow_empty = allow_empty

    def _environment_key(self, name: str) -> str:
        validated_name = validate_secret_name(name)
        normalized = validated_name.replace("/", "_").replace("-", "_").replace(".", "_")
        return f"{self.prefix}{normalized}".upper()

    def get_secret(self, name: str) -> SecretStr:
        environment_key = self._environment_key(name)
        value = os.getenv(environment_key)

        if value is None:
            raise SecretNotFoundError(
                f"Secret '{name}' was not found in process environment (Key: {environment_key})."
            )

        if not value and not self.allow_empty:
            raise SecretValidationError(
                f"Secret '{name}' exists in environment but contains an empty string."
            )

        validate_secret_value(value)
        return SecretStr(value)

    async def get_secret_async(self, name: str) -> SecretStr:
        return self.get_secret(name)

    def delete_secret(self, name: str) -> None:
        raise SecretProviderError(
            "Environment secret provider does not support runtime deletion. Invalidate cache instead."
        )


# ============================================================================
# AWS SECRETS MANAGER PROVIDER
# ============================================================================


class AWSSecretsManagerProvider:
    """
    Production AWS Secrets Manager provider. Uses AWS SDK credential chain.
    """

    def __init__(
        self,
        *,
        region_name: Optional[str] = None,
        timeout_seconds: int = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
        max_attempts: int = 3,
    ) -> None:

        if boto3 is None or BotoConfig is None:
            raise SecretConfigurationError(
                "boto3 package is required for AWS Secrets Manager integration."
            )

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")

        config = BotoConfig(
            connect_timeout=timeout_seconds,
            read_timeout=timeout_seconds,
            retries={
                "max_attempts": max_attempts,
                "mode": "standard",
            },
        )

        try:
            self._client = boto3.client(
                "secretsmanager",
                region_name=region_name,
                config=config,
            )
        except Exception as exc:
            raise SecretConfigurationError(
                f"Failed to initialize AWS Secrets Manager client: {exc}"
            ) from None

    def get_secret(self, name: str) -> SecretStr:
        validated_name = validate_secret_name(name)

        try:
            response = self._client.get_secret_value(SecretId=validated_name)

        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "Unknown")

            if error_code == "ResourceNotFoundException":
                raise SecretNotFoundError(
                    f"Secret '{name}' was not found in AWS Secrets Manager."
                ) from None

            logger.error(
                "aws_secrets_manager_error",
                extra={
                    "provider": SecretProvider.AWS_SECRETS_MANAGER.value,
                    "secret_name": name,
                    "error_code": error_code,
                },
            )

            raise SecretProviderError(
                f"AWS Secrets Manager failed to retrieve secret '{name}' (Error Code: {error_code})."
            ) from None

        except BotoCoreError:
            logger.error(
                "aws_secrets_manager_transport_error",
                extra={
                    "provider": SecretProvider.AWS_SECRETS_MANAGER.value,
                    "secret_name": name,
                },
            )

            raise SecretProviderError(
                f"AWS Secrets Manager transport or connection failure for secret '{name}'."
            ) from None

        value = response.get("SecretString")

        if value is None:
            binary_value = response.get("SecretBinary")

            if binary_value is None:
                raise SecretValidationError(
                    f"Secret '{name}' returned no text or binary payload."
                )

            if isinstance(binary_value, bytes):
                try:
                    value = binary_value.decode("utf-8")
                except UnicodeDecodeError:
                    raise SecretValidationError(
                        f"Secret '{name}' binary payload is not valid UTF-8 encoded text."
                    ) from None
            else:
                raise SecretValidationError(
                    f"Secret '{name}' binary payload returned unsupported type '{type(binary_value).__name__}'."
                )

        validate_secret_value(value)
        return SecretStr(value)

    async def get_secret_async(self, name: str) -> SecretStr:
        """Runs blocking boto3 calls safely outside the asyncio event loop."""
        return await asyncio.to_thread(self.get_secret, name)

    def delete_secret(self, name: str) -> None:
        validated_name = validate_secret_name(name)

        try:
            self._client.delete_secret(
                SecretId=validated_name,
                RecoveryWindowInDays=7,
            )

        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "Unknown")
            logger.error(
                "aws_secrets_manager_delete_failed",
                extra={
                    "provider": SecretProvider.AWS_SECRETS_MANAGER.value,
                    "secret_name": name,
                    "error_code": error_code,
                },
            )

            raise SecretProviderError(
                f"Unable to delete AWS secret '{name}' (Error Code: {error_code})."
            ) from None

        except BotoCoreError:
            raise SecretProviderError(
                f"AWS Secrets Manager transport failure while deleting secret '{name}'."
            ) from None


# ============================================================================
# INPUT & PAYLOAD VALIDATION
# ============================================================================


def validate_secret_name(name: str) -> str:
    """
    Validates provider-independent secret identifiers against standard security rules.
    """

    if not isinstance(name, str):
        raise SecretValidationError("Secret identifier must be a string.")

    cleaned_name = name.strip()

    if not cleaned_name:
        raise SecretValidationError("Secret identifier cannot be empty or whitespace-only.")

    if len(cleaned_name) > MAX_SECRET_NAME_LENGTH:
        raise SecretValidationError(
            f"Secret identifier exceeds maximum permitted length of {MAX_SECRET_NAME_LENGTH} characters."
        )

    if "\r" in cleaned_name or "\n" in cleaned_name:
        raise SecretValidationError("Secret identifier cannot contain newline characters.")

    if not SECRET_NAME_REGEX.fullmatch(cleaned_name):
        raise SecretValidationError(
            f"Secret identifier '{cleaned_name[:16]}...' contains illegal characters."
        )

    return cleaned_name


def validate_secret_value(value: str) -> None:
    """
    Validates secret payload size and string structure without exposing the content.
    """

    if not isinstance(value, str):
        raise SecretValidationError("Secret payload must be a string.")

    if len(value) > MAX_SECRET_VALUE_LENGTH:
        raise SecretValidationError(
            f"Secret payload size ({len(value)} bytes) exceeds the limit of {MAX_SECRET_VALUE_LENGTH} bytes."
        )


# ============================================================================
# MAIN SECRETS CLIENT
# ============================================================================


class SecretsClient:
    """
    Central production secrets-management service for PhishGuard_AI.
    """

    def __init__(
        self,
        *,
        provider: SecretProvider | str = SecretProvider.ENVIRONMENT,
        region_name: Optional[str] = None,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        max_cache_entries: int = MAX_CACHE_ENTRIES,
        provider_timeout_seconds: int = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
        provider_max_attempts: int = 3,
    ) -> None:

        self.provider_type = self._normalize_provider(provider)

        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds cannot be negative.")

        if max_cache_entries < 1:
            raise ValueError("max_cache_entries must be at least 1.")

        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_cache_entries = max_cache_entries

        if self.provider_type == SecretProvider.ENVIRONMENT:
            self._provider: SecretProviderClient = EnvironmentSecretProvider()

        elif self.provider_type == SecretProvider.AWS_SECRETS_MANAGER:
            self._provider = AWSSecretsManagerProvider(
                region_name=region_name,
                timeout_seconds=provider_timeout_seconds,
                max_attempts=provider_max_attempts,
            )

        else:
            raise SecretConfigurationError(
                f"Unsupported secret provider type: {self.provider_type.value}"
            )

        self._cache: dict[str, _CacheEntry] = {}
        self._lock = threading.RLock()

    # ========================================================================
    # PROVIDER NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_provider(provider: SecretProvider | str) -> SecretProvider:
        if isinstance(provider, SecretProvider):
            return provider

        if isinstance(provider, str):
            normalized = provider.strip().lower()
            try:
                return SecretProvider(normalized)
            except ValueError:
                pass

        raise SecretConfigurationError(f"Unsupported secret provider configuration: {provider!r}")

    # ========================================================================
    # SYNCHRONOUS RETRIEVAL
    # ========================================================================

    def get_secret(
        self,
        name: str,
        *,
        use_cache: bool = True,
    ) -> SecretStr:
        """Retrieve a secret as a protected Pydantic SecretStr object."""
        name = validate_secret_name(name)

        if use_cache:
            cached = self._get_cached(name)
            if cached is not None:
                return cached

        try:
            secret = self._provider.get_secret(name)
        except SecretsError:
            raise
        except Exception:
            logger.error(
                "unexpected_secret_retrieval_failure",
                extra={
                    "secret_name": name,
                    "provider": self.provider_type.value,
                },
            )
            raise SecretProviderError(
                f"Unexpected system error while retrieving secret '{name}'."
            ) from None

        validate_secret_value(secret.get_secret_value())

        if use_cache:
            self._cache_secret(name, secret)

        return secret

    def get_secret_value(
        self,
        name: str,
        *,
        use_cache: bool = True,
    ) -> str:
        """
        Retrieve raw string secret value.

        WARNING: Callers must prevent this return value from being logged or cached.
        """
        return self.get_secret(name, use_cache=use_cache).get_secret_value()

    # ========================================================================
    # ASYNC RETRIEVAL
    # ========================================================================

    async def get_secret_async(
        self,
        name: str,
        *,
        use_cache: bool = True,
    ) -> SecretStr:
        """Asynchronous secret retrieval."""
        name = validate_secret_name(name)

        if use_cache:
            cached = self._get_cached(name)
            if cached is not None:
                return cached

        try:
            secret = await self._provider.get_secret_async(name)
        except SecretsError:
            raise
        except Exception:
            logger.error(
                "unexpected_secret_async_retrieval_failure",
                extra={
                    "secret_name": name,
                    "provider": self.provider_type.value,
                },
            )
            raise SecretProviderError(
                f"Unexpected system error during async retrieval for secret '{name}'."
            ) from None

        validate_secret_value(secret.get_secret_value())

        if use_cache:
            self._cache_secret(name, secret)

        return secret

    async def get_secret_value_async(
        self,
        name: str,
        *,
        use_cache: bool = True,
    ) -> str:
        """Asynchronous raw string retrieval."""
        secret = await self.get_secret_async(name, use_cache=use_cache)
        return secret.get_secret_value()

    # ========================================================================
    # JSON PARSING & STRUCTURED SECRETS
    # ========================================================================

    def get_json_secret(
        self,
        name: str,
        *,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Retrieves and parses a JSON-encoded secret payload into a dictionary."""
        secret = self.get_secret(name, use_cache=use_cache)
        return self._parse_json_payload(name, secret.get_secret_value())

    async def get_json_secret_async(
        self,
        name: str,
        *,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Asynchronously retrieves and parses a JSON-encoded secret dictionary."""
        secret = await self.get_secret_async(name, use_cache=use_cache)
        return self._parse_json_payload(name, secret.get_secret_value())

    @staticmethod
    def _parse_json_payload(name: str, raw_payload: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw_payload)
        except json.JSONDecodeError:
            raise SecretValidationError(
                f"Secret '{name}' payload does not contain valid JSON."
            ) from None

        if not isinstance(parsed, dict):
            raise SecretValidationError(
                f"Secret '{name}' must contain a JSON object (got '{type(parsed).__name__}')."
            )

        return parsed

    # ========================================================================
    # CACHE & TTL MANAGEMENT
    # ========================================================================

    def _get_cached(self, name: str) -> Optional[SecretStr]:
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(name)

            if entry is None:
                return None

            if entry.is_expired(now):
                self._cache.pop(name, None)
                return None

            return entry.value

    def _cache_secret(self, name: str, secret: SecretStr) -> None:
        if self.cache_ttl_seconds == 0:
            return

        now = time.monotonic()

        with self._lock:
            if len(self._cache) >= self.max_cache_entries:
                self._evict_expired_locked(now)

            if len(self._cache) >= self.max_cache_entries:
                # Evict oldest entry based on expiration time
                oldest_name = min(
                    self._cache,
                    key=lambda k: self._cache[k].expires_at,
                )
                self._cache.pop(oldest_name, None)

            self._cache[name] = _CacheEntry(
                value=secret,
                expires_at=now + self.cache_ttl_seconds,
            )

    def _evict_expired_locked(self, now: float) -> None:
        expired = [
            key for key, entry in self._cache.items() if entry.is_expired(now)
        ]
        for key in expired:
            self._cache.pop(key, None)

    def invalidate(self, name: str) -> None:
        """Invalidates a single secret in the local memory cache."""
        cleaned_name = validate_secret_name(name)
        with self._lock:
            self._cache.pop(cleaned_name, None)

    def invalidate_all(self) -> None:
        """Flushes all entries from local memory cache."""
        with self._lock:
            self._cache.clear()

    # ========================================================================
    # ROTATION & DELETION
    # ========================================================================

    def rotate(self, name: str) -> None:
        """Triggered when an external platform signals secret rotation."""
        cleaned_name = validate_secret_name(name)
        self.invalidate(cleaned_name)
        logger.info(
            "secret_cache_invalidated_for_rotation",
            extra={
                "secret_name": cleaned_name,
                "provider": self.provider_type.value,
            },
        )

    def delete_secret(self, name: str) -> None:
        """Deletes secret from managed backend provider and invalidates cache."""
        cleaned_name = validate_secret_name(name)
        self._provider.delete_secret(cleaned_name)
        self.invalidate(cleaned_name)

        logger.warning(
            "secret_deleted_from_provider",
            extra={
                "secret_name": cleaned_name,
                "provider": self.provider_type.value,
            },
        )


# ============================================================================
# FACTORY & MODULE INSTANCE
# ============================================================================


def _build_default_client() -> SecretsClient:
    provider_name = os.getenv(
        "PHISHGUARD_SECRET_PROVIDER",
        SecretProvider.ENVIRONMENT.value,
    )
    region = os.getenv("AWS_REGION")

    return SecretsClient(
        provider=provider_name,
        region_name=region,
    )


# Singleton module instance
secrets_client = _build_default_client()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def secret_present(name: str) -> bool:
    """
    Checks if a secret exists and can be retrieved without exposing its content.
    """
    try:
        secrets_client.get_secret(name)
        return True
    except (SecretNotFoundError, SecretsError):
        return False


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================


__all__ = [
    "AWSSecretsManagerProvider",
    "EnvironmentSecretProvider",
    "SecretConfigurationError",
    "SecretFormat",
    "SecretNotFoundError",
    "SecretProvider",
    "SecretProviderClient",
    "SecretProviderError",
    "SecretValidationError",
    "SecretsClient",
    "SecretsError",
    "secret_present",
    "secrets_client",
    "validate_secret_name",
    "validate_secret_value",
]

"""
PhishGuard AI - Zero-Trust Data Schemas & Feature Contracts
============================================================

Version: 14.0.0
File Path: src/data/schemas.py

Purpose
-------
This module defines the canonical data contracts used by the PhishGuard AI
data-ingestion, validation, preprocessing, training, evaluation, registry,
and inference layers.

Design goals
------------
1. Zero-trust validation.
2. Strict rejection of unknown fields.
3. Strong type and range validation.
4. Cross-field consistency validation.
5. Deterministic schema fingerprints.
6. Dataset schema manifests.
7. Model-to-schema compatibility resolution.
8. Schema version management.
9. Safe JSON serialization.
10. DataFrame schema validation.
11. Batch record validation.
12. Schema drift detection.
13. Production-oriented observability.
14. Explicit failure instead of dangerous fallback behaviour.
15. Backward-compatible helper APIs where practical.

Security notes
-------------
This module validates structure and semantics. It does NOT prove that a URL,
email, attachment, sender, domain, or artifact is safe. Security decisions
must remain in the appropriate analysis engines.

Pydantic
--------
This implementation targets Pydantic v2.

Recommended dependency:
    pydantic>=2.10,<3

The module intentionally uses:
    - ConfigDict
    - field_validator
    - model_validator
    - model_validate
    - model_dump
    - model_json_schema

instead of legacy Pydantic v1 APIs.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from src.utils.logger import get_logger

logger = get_logger("src.data.schemas")


# =============================================================================
# 0. MODULE CONSTANTS
# =============================================================================

SCHEMA_MODULE_VERSION = "14.0.0"
SCHEMA_API_VERSION = "1"
SCHEMA_MANIFEST_VERSION = "14.0.0"

DEFAULT_TARGET_COLUMN = "label"

SUPPORTED_SCHEMA_KINDS = frozenset(
    {
        "email",
        "url",
        "attachment",
        "multimodal",
    }
)

SUPPORTED_LABEL_VALUES = frozenset({0, 1})

SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:@+\-/]{0,255}$"
)

DOMAIN_PATTERN = re.compile(
    r"^(?=.{1,253}$)"
    r"(?:[A-Za-z0-9]"
    r"(?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}\.?$"
)

EXTENSION_PATTERN = re.compile(
    r"^\.[a-z0-9][a-z0-9._+-]{0,31}$"
)

URL_SCHEME_PATTERN = re.compile(
    r"^(https?|ftp)://",
    re.IGNORECASE,
)

MODEL_ID_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9._-]{2,127}$"
)


# =============================================================================
# 1. INTERNAL NORMALIZATION HELPERS
# =============================================================================

def _require_string(value: Any, field_name: str) -> str:
    """
    Validate that a value is actually string-like.

    We deliberately do not blindly call str(value), because converting arbitrary
    objects into strings can hide upstream schema corruption.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string; received {type(value).__name__}"
        )

    value = value.strip()

    if not value:
        raise ValueError(f"{field_name} cannot be empty or whitespace.")

    return value


def _normalize_identifier(value: Any, field_name: str) -> str:
    """Normalize and validate a bounded pipeline identifier."""
    value = _require_string(value, field_name)

    if len(value) > 256:
        raise ValueError(
            f"{field_name} exceeds the maximum length of 256 characters."
        )

    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} contains unsupported identifier characters."
        )

    return value


def _normalize_domain(value: Any, field_name: str = "domain") -> str:
    """
    Normalize a DNS domain.

    This intentionally rejects IP addresses because a domain field and an IP
    field should represent different concepts.
    """
    value = _require_string(value, field_name).lower().rstrip(".")

    if len(value) > 253:
        raise ValueError(f"{field_name} exceeds 253 characters.")

    try:
        ipaddress.ip_address(value)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"{field_name} must be a DNS domain, not an IP address."
        )

    if not DOMAIN_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} is not a valid normalized DNS domain."
        )

    return value


def _normalize_sha256(value: Any, field_name: str) -> str:
    """Normalize and strictly validate a SHA-256 hexadecimal digest."""
    value = _require_string(value, field_name).lower()

    if not SHA256_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must be exactly 64 hexadecimal characters."
        )

    return value


def _normalize_extension(value: Any) -> str:
    """
    Normalize file extension.

    Examples:
        PDF -> .pdf
        .PDF -> .pdf
    """
    value = _require_string(value, "file_extension").lower()

    if not value.startswith("."):
        value = "." + value

    if not EXTENSION_PATTERN.fullmatch(value):
        raise ValueError(
            f"Invalid file extension '{value}'."
        )

    return value


def _validate_finite_number(
    value: float,
    field_name: str,
) -> float:
    """Reject NaN and +/- infinity."""
    if not math.isfinite(value):
        raise ValueError(
            f"{field_name} must be finite; NaN and infinity are forbidden."
        )

    return float(value)


def _canonical_json(data: Any) -> str:
    """
    Produce deterministic JSON suitable for schema fingerprints.

    The canonical representation is deliberately compact and sorted.
    """
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _sha256_text(text: str) -> str:
    """Compute SHA-256 over UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _schema_fingerprint(model: Type[BaseModel]) -> str:
    """
    Generate a deterministic fingerprint from Pydantic's JSON schema.

    This is useful for detecting accidental contract changes between training
    and production.
    """
    schema = model.model_json_schema()
    canonical = _canonical_json(schema)
    return _sha256_text(canonical)


# =============================================================================
# 2. COMMON PYDANTIC BASE MODEL
# =============================================================================

class PhishGuardBaseModel(BaseModel):
    """
    Base model for all security-sensitive data contracts.

    Configuration intentionally forbids unknown fields. Silent acceptance of
    unexpected fields is dangerous in an MLOps pipeline because producers can
    drift without the consumer noticing.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
        str_strip_whitespace=True,
        hide_input_in_errors=True,
        populate_by_name=False,
    )

    SCHEMA_VERSION: ClassVar[str] = SCHEMA_MODULE_VERSION

    @classmethod
    def schema_fingerprint(cls) -> str:
        """Return deterministic SHA-256 fingerprint of the JSON schema."""
        return _schema_fingerprint(cls)

    @classmethod
    def schema_id(cls) -> str:
        """Return canonical schema identifier."""
        return cls.__name__.lower()

    def canonical_dict(self) -> Dict[str, Any]:
        """Return deterministic model representation."""
        return self.model_dump(
            mode="json",
            exclude_none=False,
            exclude_unset=False,
        )

    def canonical_json(self) -> str:
        """Return deterministic JSON representation."""
        return _canonical_json(self.canonical_dict())

    def content_fingerprint(self) -> str:
        """
        Return a fingerprint of this particular validated record.

        This is different from schema_fingerprint():
            schema_fingerprint = contract identity
            content_fingerprint = record identity
        """
        return _sha256_text(self.canonical_json())


# =============================================================================
# 3. COMMON IDENTIFIERS
# =============================================================================

class IdentifierMixin(PhishGuardBaseModel):
    """Common metadata contract for records."""

    source: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Logical data source identifier.",
    )

    source_version: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Producer/source version.",
    )

    @field_validator("source", "source_version")
    @classmethod
    def validate_optional_metadata(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        value = value.strip()

        if not value:
            raise ValueError("Metadata fields cannot be blank.")

        return value


# =============================================================================
# 4. EMAIL FEATURE SCHEMA
# =============================================================================

class EmailFeatureSchema(IdentifierMixin):
    """
    Canonical email feature contract.

    Intended for extracted metadata/features, not raw message storage.
    """

    schema_kind: ClassVar[str] = "email"

    email_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Stable unique identifier for the email.",
    )

    sender_domain: str = Field(
        ...,
        min_length=3,
        max_length=253,
        description="Normalized sender DNS domain.",
    )

    subject_length: int = Field(
        ...,
        ge=0,
        le=1_000_000,
        description="Subject length in characters.",
    )

    body_length: int = Field(
        ...,
        ge=0,
        le=100_000_000,
        description="Body length in characters.",
    )

    has_html: bool = Field(
        default=False,
        description="Whether HTML content was detected.",
    )

    urgent_words_count: int = Field(
        default=0,
        ge=0,
        le=1_000_000,
        description="Count of urgency-related lexical indicators.",
    )

    spf_pass: bool = Field(
        default=False,
        description="Whether SPF validation passed.",
    )

    dkim_pass: bool = Field(
        default=False,
        description="Whether DKIM validation passed.",
    )

    dmarc_pass: bool = Field(
        default=False,
        description="Whether DMARC validation passed.",
    )

    sender_domain_is_punycode: bool = Field(
        default=False,
        description="Whether sender domain contains an IDNA/punycode label.",
    )

    reply_to_domain: Optional[str] = Field(
        default=None,
        max_length=253,
        description="Normalized Reply-To domain when available.",
    )

    attachment_count: int = Field(
        default=0,
        ge=0,
        le=100_000,
        description="Number of attachments associated with the message.",
    )

    url_count: int = Field(
        default=0,
        ge=0,
        le=10_000_000,
        description="Number of URLs extracted from the message.",
    )

    @field_validator("email_id")
    @classmethod
    def validate_email_id(cls, value: str) -> str:
        return _normalize_identifier(value, "email_id")

    @field_validator("sender_domain")
    @classmethod
    def validate_sender_domain(cls, value: str) -> str:
        return _normalize_domain(value, "sender_domain")

    @field_validator("reply_to_domain")
    @classmethod
    def validate_reply_to_domain(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None

        return _normalize_domain(value, "reply_to_domain")

    @model_validator(mode="after")
    def validate_email_consistency(self) -> "EmailFeatureSchema":
        """
        Validate relationships between fields.

        These checks catch structurally valid but semantically suspicious
        records.
        """
        if self.body_length == 0 and self.has_html:
            logger.debug(
                "Email record %s declares HTML but has zero body length.",
                self.email_id,
            )

        if self.url_count == 0 and self.attachment_count < 0:
            raise ValueError(
                "attachment_count cannot be negative."
            )

        return self


# =============================================================================
# 5. URL FEATURE SCHEMA
# =============================================================================

class URLFeatureSchema(IdentifierMixin):
    """
    Canonical URL feature contract.

    The schema validates URL structure but does not perform network access.
    Network access must never occur as a side effect of model validation.
    """

    schema_kind: ClassVar[str] = "url"

    url_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Stable URL identifier, preferably SHA-256.",
    )

    url: str = Field(
        ...,
        min_length=1,
        max_length=2_000_000,
        description="Target URL.",
    )

    url_length: int = Field(
        ...,
        ge=1,
        le=2_000_000,
        description="Length of normalized URL.",
    )

    domain_age_days: int = Field(
        default=-1,
        ge=-1,
        le=100_000,
        description="Domain age in days; -1 means unavailable.",
    )

    num_subdomains: int = Field(
        default=0,
        ge=0,
        le=1_000,
        description="Number of detected subdomain labels.",
    )

    has_ip_address: bool = Field(
        default=False,
        description="Whether the URL host is an IP address.",
    )

    use_shortener: bool = Field(
        default=False,
        description="Whether a known URL-shortening service was detected.",
    )

    entropy: float = Field(
        default=0.0,
        ge=0.0,
        le=8.0,
        description="Shannon entropy estimate in bits per symbol.",
    )

    scheme: str = Field(
        default="https",
        description="URL scheme.",
    )

    has_port: bool = Field(
        default=False,
        description="Whether a non-default explicit port is present.",
    )

    path_length: int = Field(
        default=0,
        ge=0,
        le=2_000_000,
        description="URL path length.",
    )

    query_length: int = Field(
        default=0,
        ge=0,
        le=2_000_000,
        description="URL query length.",
    )

    fragment_length: int = Field(
        default=0,
        ge=0,
        le=2_000_000,
        description="URL fragment length.",
    )

    @field_validator("url_id")
    @classmethod
    def validate_url_id(cls, value: str) -> str:
        return _normalize_identifier(value, "url_id")

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        value = _require_string(value, "url")

        if len(value) > 2_000_000:
            raise ValueError("URL exceeds maximum supported length.")

        if not URL_SCHEME_PATTERN.match(value):
            raise ValueError(
                "URL must include an explicit http://, https://, or ftp:// scheme."
            )

        # Prevent control characters that can create parser ambiguity.
        if any(ord(char) < 32 for char in value):
            raise ValueError(
                "URL contains ASCII control characters."
            )

        return value

    @field_validator("scheme")
    @classmethod
    def validate_scheme(cls, value: str) -> str:
        value = value.lower().strip()

        if value not in {"http", "https", "ftp"}:
            raise ValueError(
                "Unsupported URL scheme. Allowed: http, https, ftp."
            )

        return value

    @field_validator("entropy")
    @classmethod
    def validate_entropy(cls, value: float) -> float:
        return _validate_finite_number(value, "entropy")

    @model_validator(mode="after")
    def validate_url_consistency(self) -> "URLFeatureSchema":
        if self.url_length != len(self.url):
            logger.debug(
                "URL length metadata differs from raw URL length for %s.",
                self.url_id,
            )

        if self.has_ip_address:
            # This is a semantic consistency hint, not a network operation.
            try:
                host = self._extract_host()
                ipaddress.ip_address(host)
            except (ValueError, IndexError):
                raise ValueError(
                    "has_ip_address=True but URL host is not a valid IP address."
                )

        if self.domain_age_days == -1:
            # Unknown is explicitly represented by -1.
            pass

        return self

    def _extract_host(self) -> str:
        """
        Extract URL host without performing DNS resolution.

        This deliberately uses only local string parsing.
        """
        from urllib.parse import urlsplit

        parsed = urlsplit(self.url)

        if not parsed.hostname:
            raise ValueError("URL does not contain a valid hostname.")

        return parsed.hostname.strip("[]").lower()


# =============================================================================
# 6. ATTACHMENT FEATURE SCHEMA
# =============================================================================

class AttachmentFeatureSchema(IdentifierMixin):
    """
    Canonical attachment metadata contract.

    The schema does not execute or inspect the attachment. It represents
    previously extracted security telemetry.
    """

    schema_kind: ClassVar[str] = "attachment"

    attachment_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Stable attachment identifier, preferably SHA-256.",
    )

    file_extension: str = Field(
        ...,
        min_length=2,
        max_length=32,
        description="Normalized file extension including leading dot.",
    )

    file_size_bytes: int = Field(
        ...,
        ge=0,
        le=10_000_000_000,
        description="Attachment size in bytes.",
    )

    contains_macros: bool = Field(
        default=False,
        description="Whether macro content was detected.",
    )

    is_executable: bool = Field(
        default=False,
        description="Whether executable characteristics were detected.",
    )

    entropy: float = Field(
        default=0.0,
        ge=0.0,
        le=8.0,
        description="Payload entropy.",
    )

    mime_type: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Detected MIME type.",
    )

    sha256: Optional[str] = Field(
        default=None,
        description="SHA-256 content digest.",
    )

    magic_type: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Detected magic/file signature classification.",
    )

    nested_archive: bool = Field(
        default=False,
        description="Whether nested archive content was detected.",
    )

    encrypted: bool = Field(
        default=False,
        description="Whether the attachment appears encrypted/password protected.",
    )

    @field_validator("attachment_id")
    @classmethod
    def validate_attachment_id(cls, value: str) -> str:
        return _normalize_identifier(value, "attachment_id")

    @field_validator("file_extension")
    @classmethod
    def validate_file_extension(cls, value: str) -> str:
        return _normalize_extension(value)

    @field_validator("entropy")
    @classmethod
    def validate_entropy(cls, value: float) -> float:
        return _validate_finite_number(value, "entropy")

    @field_validator("sha256")
    @classmethod
    def validate_sha256(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None

        return _normalize_sha256(value, "sha256")

    @field_validator("mime_type", "magic_type")
    @classmethod
    def normalize_optional_text(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None

        value = value.strip()

        if not value:
            return None

        return value

    @model_validator(mode="after")
    def validate_attachment_consistency(
        self,
    ) -> "AttachmentFeatureSchema":
        """
        Detect contradictory metadata.
        """
        if self.file_size_bytes == 0 and self.entropy > 0:
            raise ValueError(
                "Zero-byte attachment cannot have non-zero entropy."
            )

        if self.is_executable and self.file_extension in {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
        }:
            logger.warning(
                "Executable indicator conflicts with image extension: %s",
                self.file_extension,
            )

        return self


# =============================================================================
# 7. LABEL SCHEMA
# =============================================================================

class TargetLabelSchema(PhishGuardBaseModel):
    """
    Canonical supervised-learning target contract.
    """

    label: int = Field(
        ...,
        description="Binary classification label: 0 or 1.",
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: int) -> int:
        if value not in SUPPORTED_LABEL_VALUES:
            raise ValueError(
                "label must be exactly 0 or 1."
            )

        return value


# =============================================================================
# 8. DATASET METADATA CONTRACT
# =============================================================================

class DatasetMetadataSchema(PhishGuardBaseModel):
    """
    Metadata attached to a dataset or dataset snapshot.
    """

    dataset_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
    )

    dataset_version: str = Field(
        ...,
        min_length=1,
        max_length=64,
    )

    source: str = Field(
        ...,
        min_length=1,
        max_length=256,
    )

    created_at: datetime = Field(
        ...,
        description="UTC dataset creation timestamp.",
    )

    row_count: int = Field(
        ...,
        ge=0,
    )

    feature_count: int = Field(
        ...,
        ge=0,
    )

    content_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 dataset artifact hash.",
    )

    @field_validator("dataset_id", "dataset_version", "source")
    @classmethod
    def normalize_metadata_text(cls, value: str) -> str:
        return _require_string(value, "dataset metadata")

    @field_validator("created_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError(
                "created_at must be timezone-aware."
            )

        return value.astimezone(timezone.utc)

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None

        return _normalize_sha256(value, "content_hash")


# =============================================================================
# 9. FEATURE DEFINITION
# =============================================================================

class FeatureDefinition(PhishGuardBaseModel):
    """
    Machine-readable feature contract.

    This provides a richer representation than a plain list[str] and enables
    schema-drift detection and documentation generation.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    dtype: str = Field(
        ...,
        min_length=1,
        max_length=64,
    )

    required: bool = True

    nullable: bool = False

    description: str = Field(
        default="",
        max_length=2048,
    )

    domain: Optional[str] = Field(
        default=None,
        max_length=128,
    )

    default: Optional[Any] = None

    @field_validator("name")
    @classmethod
    def validate_feature_name(cls, value: str) -> str:
        value = _require_string(value, "feature name")

        if not re.fullmatch(
            r"^[A-Za-z_][A-Za-z0-9_]{0,127}$",
            value,
        ):
            raise ValueError(
                f"Invalid feature name '{value}'."
            )

        return value

    @field_validator("dtype")
    @classmethod
    def normalize_dtype(cls, value: str) -> str:
        return _require_string(value, "dtype").lower()


# =============================================================================
# 10. DATASET SCHEMA MANIFEST
# =============================================================================

@dataclass
class DatasetSchemaManifest:
    """
    Canonical multi-modal dataset schema manifest.

    Unlike the previous implementation, the manifest performs consistency
    checks and has deterministic serialization/fingerprinting.
    """

    schema_version: str = SCHEMA_MANIFEST_VERSION

    email_features: List[str] = field(
        default_factory=lambda: [
            "email_id",
            "sender_domain",
            "subject_length",
            "body_length",
            "has_html",
            "urgent_words_count",
            "spf_pass",
            "dkim_pass",
            "dmarc_pass",
            "sender_domain_is_punycode",
            "reply_to_domain",
            "attachment_count",
            "url_count",
        ]
    )

    url_features: List[str] = field(
        default_factory=lambda: [
            "url_id",
            "url",
            "url_length",
            "domain_age_days",
            "num_subdomains",
            "has_ip_address",
            "use_shortener",
            "entropy",
            "scheme",
            "has_port",
            "path_length",
            "query_length",
            "fragment_length",
        ]
    )

    attachment_features: List[str] = field(
        default_factory=lambda: [
            "attachment_id",
            "file_extension",
            "file_size_bytes",
            "contains_macros",
            "is_executable",
            "entropy",
            "mime_type",
            "sha256",
            "magic_type",
            "nested_archive",
            "encrypted",
        ]
    )

    target_column: str = DEFAULT_TARGET_COLUMN

    metadata_columns: List[str] = field(
        default_factory=lambda: [
            "source",
            "source_version",
        ]
    )

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    description: str = (
        "PhishGuard AI canonical multi-modal dataset schema."
    )

    def __post_init__(self) -> None:
        self.schema_version = _require_string(
            self.schema_version,
            "schema_version",
        )

        self.target_column = _require_string(
            self.target_column,
            "target_column",
        )

        self.description = self.description.strip()

        self.email_features = self._normalize_feature_list(
            self.email_features,
            "email_features",
        )

        self.url_features = self._normalize_feature_list(
            self.url_features,
            "url_features",
        )

        self.attachment_features = self._normalize_feature_list(
            self.attachment_features,
            "attachment_features",
        )

        self.metadata_columns = self._normalize_feature_list(
            self.metadata_columns,
            "metadata_columns",
        )

        self.validate()

    @staticmethod
    def _normalize_feature_list(
        values: Sequence[str],
        field_name: str,
    ) -> List[str]:
        if not isinstance(values, Sequence) or isinstance(
            values,
            (str, bytes),
        ):
            raise TypeError(
                f"{field_name} must be a sequence of strings."
            )

        normalized: List[str] = []

        for value in values:
            value = _require_string(value, field_name)

            if value in normalized:
                raise ValueError(
                    f"Duplicate feature '{value}' found in {field_name}."
                )

            normalized.append(value)

        return normalized

    def validate(self) -> None:
        """
        Validate manifest-level invariants.
        """
        all_features = (
            self.email_features
            + self.url_features
            + self.attachment_features
            + self.metadata_columns
        )

        duplicates = {
            feature
            for feature in all_features
            if all_features.count(feature) > 1
        }

        # Some multimodal pipelines may intentionally reuse `entropy`.
        # Therefore duplicate detection is exposed but not automatically fatal
        # for known shared features.
        allowed_shared = {
            "entropy",
            "source",
            "source_version",
        }

        invalid_duplicates = duplicates - allowed_shared

        if invalid_duplicates:
            raise ValueError(
                "Manifest contains duplicate feature names across modalities: "
                + ", ".join(sorted(invalid_duplicates))
            )

        if self.target_column in all_features:
            raise ValueError(
                f"Target column '{self.target_column}' cannot also be a feature."
            )

        if not self.schema_version:
            raise ValueError("schema_version cannot be empty.")

    def get_all_feature_names(
        self,
        include_metadata: bool = True,
    ) -> List[str]:
        """
        Return flattened feature list.

        Duplicates are removed while preserving declaration order.
        """
        groups = [
            self.email_features,
            self.url_features,
            self.attachment_features,
        ]

        if include_metadata:
            groups.append(self.metadata_columns)

        result: List[str] = []

        for group in groups:
            for feature in group:
                if feature not in result:
                    result.append(feature)

        return result

    def feature_count(self) -> int:
        """Return number of unique non-target features."""
        return len(self.get_all_feature_names())

    def modality_feature_count(self) -> Dict[str, int]:
        """Return feature counts by modality."""
        return {
            "email": len(self.email_features),
            "url": len(self.url_features),
            "attachment": len(self.attachment_features),
            "metadata": len(self.metadata_columns),
            "total_unique": self.feature_count(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to a plain dictionary."""
        self.validate()
        return asdict(self)

    def canonical_dict(self) -> Dict[str, Any]:
        """
        Return deterministic manifest data.

        The creation timestamp is retained intentionally because this is the
        complete artifact representation.
        """
        return self.to_dict()

    def canonical_json(self) -> str:
        """Return deterministic JSON representation."""
        return _canonical_json(self.canonical_dict())

    def fingerprint(self) -> str:
        """
        Compute SHA-256 fingerprint of the manifest.

        SHA-256 is used here because the fingerprint is an integrity identity,
        not a password hash. Python's hashlib provides SHA-256 as a standard
        cryptographic hash. :contentReference[oaicite:3]{index=3}
        """
        return _sha256_text(self.canonical_json())

    def export_json(
        self,
        filepath: Union[str, Path],
        *,
        include_fingerprint: bool = True,
        indent: int = 2,
    ) -> Path:
        """
        Export manifest safely.

        A temporary file is used before replacement to avoid leaving a
        half-written manifest after a process interruption.
        """
        if indent < 0:
            raise ValueError("indent must be non-negative.")

        path = Path(filepath).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = self.to_dict()

        if include_fingerprint:
            payload["manifest_fingerprint"] = self.fingerprint()

        temporary = path.with_name(
            f".{path.name}.tmp"
        )

        try:
            with temporary.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                json.dump(
                    payload,
                    handle,
                    indent=indent,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                handle.write("\n")
                handle.flush()

                # Best-effort durability.
                try:
                    import os

                    os.fsync(handle.fileno())
                except OSError:
                    logger.debug(
                        "Filesystem fsync unavailable for manifest '%s'.",
                        path,
                    )

            temporary.replace(path)

        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

        logger.info(
            "Exported dataset schema manifest to '%s' fingerprint=%s",
            path,
            self.fingerprint(),
        )

        return path

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "DatasetSchemaManifest":
        """Create a manifest from a dictionary."""
        if not isinstance(payload, Mapping):
            raise TypeError("Manifest payload must be a mapping.")

        data = dict(payload)

        # Fingerprints are metadata, not constructor fields.
        data.pop("manifest_fingerprint", None)

        return cls(**data)

    @classmethod
    def from_json(
        cls,
        filepath: Union[str, Path],
    ) -> "DatasetSchemaManifest":
        """Load a manifest from JSON."""
        path = Path(filepath).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Schema manifest not found: '{path}'"
            )

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        return cls.from_dict(payload)

    def verify_fingerprint(
        self,
        expected_fingerprint: str,
    ) -> bool:
        """Constant-time-ish string equality for fingerprint comparison."""
        import hmac

        if not isinstance(expected_fingerprint, str):
            return False

        return hmac.compare_digest(
            self.fingerprint().lower(),
            expected_fingerprint.strip().lower(),
        )


# =============================================================================
# 11. DATAFRAME SCHEMA REPORT
# =============================================================================

@dataclass
class SchemaValidationReport:
    """
    Result of DataFrame schema validation.

    This object is intentionally separate from Pydantic's ValidationError so
    batch/dataset-level failures can be reported without stopping after the
    first invalid row.
    """

    is_valid: bool

    expected_columns: List[str] = field(default_factory=list)

    actual_columns: List[str] = field(default_factory=list)

    missing_columns: List[str] = field(default_factory=list)

    unexpected_columns: List[str] = field(default_factory=list)

    duplicate_columns: List[str] = field(default_factory=list)

    dtype_mismatches: Dict[str, Dict[str, str]] = field(
        default_factory=dict
    )

    null_columns: Dict[str, int] = field(
        default_factory=dict
    )

    errors: List[str] = field(default_factory=list)

    warnings: List[str] = field(default_factory=list)

    row_count: int = 0

    schema_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation report to JSON-compatible dictionary."""
        return asdict(self)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)


# =============================================================================
# 12. SCHEMA VALIDATOR
# =============================================================================

class DataFrameSchemaValidator:
    """
    Dataset-level schema validator.

    This validator complements the row-level Pydantic contracts.

    Row-level validation:
        EmailFeatureSchema(...)

    Dataset-level validation:
        DataFrameSchemaValidator(...).validate(df)
    """

    def __init__(
        self,
        manifest: Optional[DatasetSchemaManifest] = None,
        *,
        allow_extra_columns: bool = False,
        allow_missing_optional_columns: bool = True,
        required_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.manifest = manifest or DatasetSchemaManifest()

        self.allow_extra_columns = allow_extra_columns

        self.allow_missing_optional_columns = (
            allow_missing_optional_columns
        )

        self.required_columns = list(
            required_columns
            if required_columns is not None
            else self.manifest.get_all_feature_names()
        )

    def validate(
        self,
        df: pd.DataFrame,
    ) -> SchemaValidationReport:
        """
        Validate DataFrame structure.

        No data mutation occurs.
        """
        if not isinstance(df, pd.DataFrame):
            return SchemaValidationReport(
                is_valid=False,
                errors=[
                    "Input must be a pandas DataFrame."
                ],
            )

        actual_columns = [str(column) for column in df.columns]

        duplicate_columns = [
            column
            for column in df.columns[
                df.columns.duplicated()
            ].tolist()
        ]

        expected_columns = list(
            self.required_columns
        )

        missing_columns = [
            column
            for column in expected_columns
            if column not in df.columns
        ]

        unexpected_columns = [
            column
            for column in actual_columns
            if column not in expected_columns
        ]

        errors: List[str] = []
        warnings: List[str] = []

        if duplicate_columns:
            errors.append(
                "Duplicate DataFrame column names detected: "
                + ", ".join(map(str, duplicate_columns))
            )

        if missing_columns:
            errors.append(
                "Missing required schema columns: "
                + ", ".join(missing_columns)
            )

        if unexpected_columns:
            message = (
                "Unexpected DataFrame columns detected: "
                + ", ".join(unexpected_columns)
            )

            if self.allow_extra_columns:
                warnings.append(message)
            else:
                errors.append(message)

        null_columns: Dict[str, int] = {}

        for column in actual_columns:
            if column in df.columns:
                count = int(df[column].isna().sum())

                if count:
                    null_columns[column] = count

        report = SchemaValidationReport(
            is_valid=not errors,
            expected_columns=expected_columns,
            actual_columns=actual_columns,
            missing_columns=missing_columns,
            unexpected_columns=unexpected_columns,
            duplicate_columns=duplicate_columns,
            null_columns=null_columns,
            errors=errors,
            warnings=warnings,
            row_count=len(df),
            schema_fingerprint=self.manifest.fingerprint(),
        )

        return report


# =============================================================================
# 13. RECORD VALIDATION HELPERS
# =============================================================================

def validate_email_record(
    data: Mapping[str, Any],
) -> EmailFeatureSchema:
    """
    Validate one email record.

    Raises:
        pydantic.ValidationError
    """
    return EmailFeatureSchema.model_validate(data)


def validate_url_record(
    data: Mapping[str, Any],
) -> URLFeatureSchema:
    """Validate one URL record."""
    return URLFeatureSchema.model_validate(data)


def validate_attachment_record(
    data: Mapping[str, Any],
) -> AttachmentFeatureSchema:
    """Validate one attachment record."""
    return AttachmentFeatureSchema.model_validate(data)


def validate_target_record(
    data: Mapping[str, Any],
) -> TargetLabelSchema:
    """Validate one supervised-learning target record."""
    return TargetLabelSchema.model_validate(data)


# =============================================================================
# 14. BATCH VALIDATION
# =============================================================================

@dataclass
class BatchValidationResult:
    """Result of validating multiple records."""

    schema_name: str

    total_records: int

    valid_records: int = 0

    invalid_records: int = 0

    errors: List[Dict[str, Any]] = field(
        default_factory=list
    )

    max_errors: int = 100

    @property
    def is_valid(self) -> bool:
        return self.invalid_records == 0

    @property
    def error_rate(self) -> float:
        if self.total_records == 0:
            return 0.0

        return self.invalid_records / self.total_records

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "error_rate": self.error_rate,
            "is_valid": self.is_valid,
            "errors": self.errors,
        }


def validate_records(
    records: Iterable[Mapping[str, Any]],
    model: Type[BaseModel],
    *,
    max_errors: int = 100,
) -> BatchValidationResult:
    """
    Validate records against a Pydantic model.

    Validation continues after an individual record fails, which is useful for
    ingestion pipelines where a complete batch report is required.
    """
    if max_errors <= 0:
        raise ValueError("max_errors must be greater than zero.")

    result = BatchValidationResult(
        schema_name=model.__name__,
        total_records=0,
        max_errors=max_errors,
    )

    for index, record in enumerate(records):
        result.total_records += 1

        try:
            model.model_validate(record)
            result.valid_records += 1

        except ValidationError as exc:
            result.invalid_records += 1

            if len(result.errors) < max_errors:
                result.errors.append(
                    {
                        "index": index,
                        "errors": exc.errors(
                            include_input=False
                        ),
                    }
                )

    return result


# =============================================================================
# 15. SCHEMA DRIFT DETECTION
# =============================================================================

@dataclass
class SchemaDriftReport:
    """Report describing differences between expected and observed columns."""

    drift_detected: bool

    added_columns: List[str] = field(
        default_factory=list
    )

    removed_columns: List[str] = field(
        default_factory=list
    )

    reordered_columns: bool = False

    duplicate_columns: List[str] = field(
        default_factory=list
    )

    expected_fingerprint: Optional[str] = None

    notes: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_schema_drift(
    expected_columns: Sequence[str],
    actual_columns: Sequence[str],
    *,
    expected_fingerprint: Optional[str] = None,
) -> SchemaDriftReport:
    """
    Detect structural schema drift.

    This function does not infer semantic compatibility from names alone.
    """
    expected = list(expected_columns)
    actual = list(actual_columns)

    expected_set = set(expected)
    actual_set = set(actual)

    added = [
        column
        for column in actual
        if column not in expected_set
    ]

    removed = [
        column
        for column in expected
        if column not in actual_set
    ]

    duplicate_columns = [
        column
        for column in dict.fromkeys(actual)
        if actual.count(column) > 1
    ]

    reordered = (
        not added
        and not removed
        and expected != actual
    )

    drift = bool(
        added
        or removed
        or duplicate_columns
        or reordered
    )

    notes: List[str] = []

    if added:
        notes.append(
            f"{len(added)} column(s) added."
        )

    if removed:
        notes.append(
            f"{len(removed)} column(s) removed."
        )

    if reordered:
        notes.append(
            "Columns are identical but their order changed."
        )

    if duplicate_columns:
        notes.append(
            "Duplicate columns detected."
        )

    return SchemaDriftReport(
        drift_detected=drift,
        added_columns=added,
        removed_columns=removed,
        reordered_columns=reordered,
        duplicate_columns=duplicate_columns,
        expected_fingerprint=expected_fingerprint,
        notes=notes,
    )


# =============================================================================
# 16. MODEL-SCHEMA REGISTRY ENTRY
# =============================================================================

@dataclass(frozen=True)
class SchemaRegistryEntry:
    """
    Immutable model-to-schema registration metadata.
    """

    model_id: str
    schema_name: str
    schema_kind: str
    schema_version: str
    model_class: Type[BaseModel]
    aliases: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not MODEL_ID_PATTERN.fullmatch(
            self.model_id
        ):
            raise ValueError(
                f"Invalid model_id '{self.model_id}'."
            )

        if self.schema_kind not in SUPPORTED_SCHEMA_KINDS:
            raise ValueError(
                f"Unsupported schema kind '{self.schema_kind}'."
            )

        if not issubclass(
            self.model_class,
            BaseModel,
        ):
            raise TypeError(
                "model_class must inherit from BaseModel."
            )

    @property
    def fingerprint(self) -> str:
        """Return registered schema fingerprint."""
        return _schema_fingerprint(self.model_class)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "schema_name": self.schema_name,
            "schema_kind": self.schema_kind,
            "schema_version": self.schema_version,
            "aliases": list(self.aliases),
            "fingerprint": self.fingerprint,
        }


# =============================================================================
# 17. SCHEMA REGISTRY
# =============================================================================

class SchemaRegistry:
    """
    Explicit model/schema registry.

    Unknown identifiers raise KeyError. Silent fallback is intentionally
    prohibited because a typo in a model identifier must not silently switch
    the data contract.
    """

    def __init__(
        self,
        entries: Optional[
            Iterable[SchemaRegistryEntry]
        ] = None,
    ) -> None:
        self._entries: Dict[
            str,
            SchemaRegistryEntry,
        ] = {}

        self._aliases: Dict[str, str] = {}

        if entries:
            for entry in entries:
                self.register(entry)

    def register(
        self,
        entry: SchemaRegistryEntry,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register one schema entry.
        """
        model_id = entry.model_id.lower()

        if (
            model_id in self._entries
            and not overwrite
        ):
            raise KeyError(
                f"Schema model ID '{model_id}' is already registered."
            )

        self._entries[model_id] = entry

        for alias in entry.aliases:
            alias_normalized = alias.strip().lower()

            if not alias_normalized:
                continue

            existing = self._aliases.get(
                alias_normalized
            )

            if (
                existing is not None
                and existing != model_id
                and not overwrite
            ):
                raise KeyError(
                    f"Schema alias '{alias}' already belongs to "
                    f"'{existing}'."
                )

            self._aliases[alias_normalized] = model_id

        logger.info(
            "Registered schema model_id=%s kind=%s version=%s fingerprint=%s",
            model_id,
            entry.schema_kind,
            entry.schema_version,
            entry.fingerprint,
        )

    def resolve(
        self,
        model_id: str,
    ) -> SchemaRegistryEntry:
        """
        Resolve model ID or alias.

        Raises:
            KeyError: unknown model ID.
        """
        normalized = _require_string(
            model_id,
            "model_id",
        ).lower()

        if normalized in self._entries:
            return self._entries[normalized]

        alias_target = self._aliases.get(
            normalized
        )

        if alias_target:
            return self._entries[alias_target]

        available = sorted(
            self._entries.keys()
        )

        raise KeyError(
            f"Unknown schema model ID '{model_id}'. "
            f"Registered IDs: {available}"
        )

    def get_model_class(
        self,
        model_id: str,
    ) -> Type[BaseModel]:
        """Return Pydantic class for model ID."""
        return self.resolve(model_id).model_class

    def contains(
        self,
        model_id: str,
    ) -> bool:
        """Return whether model ID/alias exists."""
        try:
            self.resolve(model_id)
            return True
        except KeyError:
            return False

    def list_model_ids(self) -> List[str]:
        """Return sorted canonical model IDs."""
        return sorted(self._entries.keys())

    def list_entries(self) -> List[SchemaRegistryEntry]:
        """Return registered entries."""
        return list(
            self._entries.values()
        )

    def metadata(self) -> Dict[str, Dict[str, Any]]:
        """Return registry metadata."""
        return {
            entry.model_id: entry.to_dict()
            for entry in self._entries.values()
        }

    def as_read_only_mapping(
        self,
    ) -> Mapping[str, SchemaRegistryEntry]:
        """Return immutable registry view."""
        return MappingProxyType(
            dict(self._entries)
        )


# =============================================================================
# 18. DEFAULT REGISTRY
# =============================================================================

DEFAULT_SCHEMA_REGISTRY = SchemaRegistry(
    entries=[
        SchemaRegistryEntry(
            model_id="phishguard_v14_email",
            schema_name="EmailFeatureSchema",
            schema_kind="email",
            schema_version=SCHEMA_MODULE_VERSION,
            model_class=EmailFeatureSchema,
            aliases=(
                "email",
                "phishguard_email",
            ),
        ),
        SchemaRegistryEntry(
            model_id="phishguard_v14_url",
            schema_name="URLFeatureSchema",
            schema_kind="url",
            schema_version=SCHEMA_MODULE_VERSION,
            model_class=URLFeatureSchema,
            aliases=(
                "url",
                "phishguard_url",
            ),
        ),
        SchemaRegistryEntry(
            model_id="phishguard_v14_attachment",
            schema_name="AttachmentFeatureSchema",
            schema_kind="attachment",
            schema_version=SCHEMA_MODULE_VERSION,
            model_class=AttachmentFeatureSchema,
            aliases=(
                "attachment",
                "phishguard_attachment",
            ),
        ),
    ]
)


# =============================================================================
# 19. MODEL ID RESOLUTION
# =============================================================================

def get_schema_by_model_id(
    model_id: str,
) -> Type[BaseModel]:
    """
    Resolve a model identifier to a Pydantic schema class.

    Unlike the previous implementation, this function does NOT silently return
    the multimodal manifest for unknown IDs.

    Example:
        schema = get_schema_by_model_id("phishguard_v14_url")
        record = schema.model_validate(payload)
    """
    entry = DEFAULT_SCHEMA_REGISTRY.resolve(
        model_id
    )

    return entry.model_class


def get_schema_registry_entry(
    model_id: str,
) -> SchemaRegistryEntry:
    """Return complete registry metadata for a model ID."""
    return DEFAULT_SCHEMA_REGISTRY.resolve(
        model_id
    )


# =============================================================================
# 20. SCHEMA COMPATIBILITY
# =============================================================================

@dataclass
class SchemaCompatibilityResult:
    """Result of comparing two schema contracts."""

    compatible: bool

    expected_schema: str

    actual_schema: str

    expected_fingerprint: str

    actual_fingerprint: str

    reasons: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compare_schema_compatibility(
    expected: Type[BaseModel],
    actual: Type[BaseModel],
) -> SchemaCompatibilityResult:
    """
    Compare two Pydantic schema contracts using JSON schema fingerprints.

    Exact fingerprint equality is intentionally conservative.
    """
    expected_fp = _schema_fingerprint(
        expected
    )

    actual_fp = _schema_fingerprint(
        actual
    )

    reasons: List[str] = []

    compatible = expected_fp == actual_fp

    if not compatible:
        reasons.append(
            "Schema fingerprints differ."
        )

    return SchemaCompatibilityResult(
        compatible=compatible,
        expected_schema=expected.__name__,
        actual_schema=actual.__name__,
        expected_fingerprint=expected_fp,
        actual_fingerprint=actual_fp,
        reasons=reasons,
    )


# =============================================================================
# 21. JSON SCHEMA EXPORT
# =============================================================================

def export_json_schema(
    model: Type[BaseModel],
    filepath: Union[str, Path],
) -> Path:
    """
    Export Pydantic JSON schema atomically.

    Useful for:
        - API contracts
        - documentation
        - CI schema snapshots
        - downstream validation
        - data-contract review
    """
    if not issubclass(model, BaseModel):
        raise TypeError(
            "model must be a Pydantic BaseModel subclass."
        )

    path = Path(filepath).expanduser().resolve()
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = model.model_json_schema()

    temporary = path.with_name(
        f".{path.name}.tmp"
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")
            handle.flush()

            try:
                import os

                os.fsync(handle.fileno())
            except OSError:
                logger.debug(
                    "Could not fsync JSON schema '%s'.",
                    path,
                )

        temporary.replace(path)

    except Exception:
        try:
            temporary.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        raise

    logger.info(
        "Exported JSON schema '%s' fingerprint=%s",
        path,
        _schema_fingerprint(model),
    )

    return path


# =============================================================================
# 22. SCHEMA SNAPSHOT
# =============================================================================

@dataclass
class SchemaSnapshot:
    """
    Immutable-ish snapshot metadata used by CI/CD and MLOps pipelines.
    """

    schema_name: str
    schema_version: str
    fingerprint: str
    captured_at: str
    fields: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_schema_snapshot(
    model: Type[BaseModel],
) -> SchemaSnapshot:
    """
    Capture a deterministic schema snapshot.
    """
    if not issubclass(model, BaseModel):
        raise TypeError(
            "model must inherit from BaseModel."
        )

    schema = model.model_json_schema()

    properties = schema.get(
        "properties",
        {},
    )

    fields = sorted(
        str(name)
        for name in properties
    )

    return SchemaSnapshot(
        schema_name=model.__name__,
        schema_version=getattr(
            model,
            "SCHEMA_VERSION",
            SCHEMA_MODULE_VERSION,
        ),
        fingerprint=_schema_fingerprint(
            model
        ),
        captured_at=datetime.now(
            timezone.utc
        ).isoformat(),
        fields=fields,
    )


# =============================================================================
# 23. SCHEMA HEALTH CHECK
# =============================================================================

@dataclass
class SchemaHealthReport:
    """Operational health report for the schema subsystem."""

    healthy: bool

    pydantic_version: str

    registered_models: List[str]

    fingerprints: Dict[str, str]

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def schema_health_check() -> SchemaHealthReport:
    """
    Verify that registered schema contracts can be built and fingerprinted.

    This is useful as a startup/liveness check in training and inference
    services.
    """
    import pydantic

    warnings: List[str] = []
    errors: List[str] = []
    fingerprints: Dict[str, str] = {}

    registered_models = (
        DEFAULT_SCHEMA_REGISTRY.list_model_ids()
    )

    for model_id in registered_models:
        try:
            entry = DEFAULT_SCHEMA_REGISTRY.resolve(
                model_id
            )

            fingerprints[model_id] = (
                entry.fingerprint
            )

        except Exception as exc:
            errors.append(
                f"{model_id}: {exc}"
            )

    healthy = not errors

    return SchemaHealthReport(
        healthy=healthy,
        pydantic_version=pydantic.__version__,
        registered_models=registered_models,
        fingerprints=fingerprints,
        warnings=warnings,
        errors=errors,
    )


# =============================================================================
# 24. DATAFRAME RECORD EXTRACTION
# =============================================================================

def dataframe_to_validated_records(
    df: pd.DataFrame,
    model: Type[BaseModel],
    *,
    max_errors: int = 100,
) -> Tuple[List[BaseModel], BatchValidationResult]:
    """
    Validate every DataFrame row and return validated model instances.

    Invalid records are omitted from the returned valid-record list.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if not issubclass(model, BaseModel):
        raise TypeError(
            "model must be a Pydantic BaseModel subclass."
        )

    valid_records: List[BaseModel] = []

    result = BatchValidationResult(
        schema_name=model.__name__,
        total_records=0,
        max_errors=max_errors,
    )

    for index, row in df.iterrows():
        result.total_records += 1

        payload = row.to_dict()

        try:
            validated = model.model_validate(
                payload
            )

            valid_records.append(
                validated
            )

            result.valid_records += 1

        except ValidationError as exc:
            result.invalid_records += 1

            if len(result.errors) < max_errors:
                result.errors.append(
                    {
                        "index": index,
                        "errors": exc.errors(
                            include_input=False
                        ),
                    }
                )

    return valid_records, result


# =============================================================================
# 25. REQUIRED FEATURE HELPERS
# =============================================================================

def get_required_features(
    model: Type[BaseModel],
) -> List[str]:
    """
    Return fields required by a Pydantic model.

    This uses Pydantic's generated JSON schema rather than private internals.
    """
    if not issubclass(model, BaseModel):
        raise TypeError(
            "model must inherit from BaseModel."
        )

    schema = model.model_json_schema()

    return list(
        schema.get(
            "required",
            [],
        )
    )


def get_optional_features(
    model: Type[BaseModel],
) -> List[str]:
    """Return fields that are not required by the JSON schema."""
    if not issubclass(model, BaseModel):
        raise TypeError(
            "model must inherit from BaseModel."
        )

    schema = model.model_json_schema()

    required = set(
        schema.get(
            "required",
            [],
        )
    )

    properties = schema.get(
        "properties",
        {},
    )

    return [
        str(field_name)
        for field_name in properties
        if field_name not in required
    ]


def get_schema_fields(
    model: Type[BaseModel],
) -> List[str]:
    """Return all declared JSON-schema properties."""
    if not issubclass(model, BaseModel):
        raise TypeError(
            "model must inherit from BaseModel."
        )

    schema = model.model_json_schema()

    return list(
        schema.get(
            "properties",
            {},
        ).keys()
    )


# =============================================================================
# 26. MANIFEST / MODEL CONSISTENCY
# =============================================================================

def validate_manifest_against_model(
    manifest: DatasetSchemaManifest,
    model: Type[BaseModel],
) -> SchemaValidationReport:
    """
    Compare manifest feature declarations against a Pydantic model.

    This is particularly useful in CI to catch a situation where a model class
    was changed but the dataset manifest was not updated.
    """
    if not isinstance(
        manifest,
        DatasetSchemaManifest,
    ):
        raise TypeError(
            "manifest must be DatasetSchemaManifest."
        )

    expected = set(
        manifest.get_all_feature_names()
    )

    actual = set(
        get_schema_fields(model)
    )

    missing = sorted(
        expected - actual
    )

    unexpected = sorted(
        actual - expected
    )

    errors: List[str] = []

    if missing:
        errors.append(
            "Manifest features missing from model: "
            + ", ".join(missing)
        )

    if unexpected:
        errors.append(
            "Model fields missing from manifest: "
            + ", ".join(unexpected)
        )

    return SchemaValidationReport(
        is_valid=not errors,
        expected_columns=sorted(expected),
        actual_columns=sorted(actual),
        missing_columns=missing,
        unexpected_columns=unexpected,
        errors=errors,
        row_count=0,
        schema_fingerprint=_schema_fingerprint(
            model
        ),
    )


# =============================================================================
# 27. MODEL-SCHEMA CONTRACT ASSERTION
# =============================================================================

def assert_schema_contract(
    model_id: str,
    expected_fingerprint: str,
) -> None:
    """
    Fail fast if the deployed schema differs from the expected fingerprint.

    Intended for startup checks and CI/CD deployment gates.
    """
    entry = get_schema_registry_entry(
        model_id
    )

    actual = entry.fingerprint

    if actual.lower() != expected_fingerprint.lower():
        raise RuntimeError(
            "Schema contract fingerprint mismatch for "
            f"'{model_id}'. Expected={expected_fingerprint}, "
            f"actual={actual}."
        )

    logger.info(
        "Schema contract verified: model_id=%s fingerprint=%s",
        model_id,
        actual,
    )


# =============================================================================
# 28. PUBLIC EXPORT CONTRACT
# =============================================================================

__all__ = [
    # Constants
    "SCHEMA_MODULE_VERSION",
    "SCHEMA_API_VERSION",
    "SCHEMA_MANIFEST_VERSION",
    "DEFAULT_TARGET_COLUMN",
    "SUPPORTED_SCHEMA_KINDS",
    "SUPPORTED_LABEL_VALUES",

    # Base model
    "PhishGuardBaseModel",

    # Feature schemas
    "EmailFeatureSchema",
    "URLFeatureSchema",
    "AttachmentFeatureSchema",
    "TargetLabelSchema",

    # Dataset metadata
    "DatasetMetadataSchema",
    "FeatureDefinition",

    # Manifest
    "DatasetSchemaManifest",

    # DataFrame validation
    "SchemaValidationReport",
    "DataFrameSchemaValidator",

    # Batch validation
    "BatchValidationResult",
    "validate_records",
    "dataframe_to_validated_records",

    # Record helpers
    "validate_email_record",
    "validate_url_record",
    "validate_attachment_record",
    "validate_target_record",

    # Drift detection
    "SchemaDriftReport",
    "detect_schema_drift",

    # Registry
    "SchemaRegistryEntry",
    "SchemaRegistry",
    "DEFAULT_SCHEMA_REGISTRY",
    "get_schema_by_model_id",
    "get_schema_registry_entry",

    # Compatibility
    "SchemaCompatibilityResult",
    "compare_schema_compatibility",

    # JSON schema / snapshots
    "export_json_schema",
    "SchemaSnapshot",
    "create_schema_snapshot",

    # Health
    "SchemaHealthReport",
    "schema_health_check",

    # Feature inspection
    "get_required_features",
    "get_optional_features",
    "get_schema_fields",

    # Manifest compatibility
    "validate_manifest_against_model",
    "assert_schema_contract",
]
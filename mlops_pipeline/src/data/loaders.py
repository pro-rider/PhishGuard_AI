"""
PhishGuard AI - Secure High-Performance Data Loading Engine
============================================================

Version: 13.0.0
File Path: mlops_pipeline/src/data/loaders.py

Purpose
-------
Production-grade, security-conscious data ingestion layer for the PhishGuard AI
MLOps pipeline.

This module provides:

    * Strict dataset path validation
    * Optional symlink restrictions
    * File-size and row-count resource guards
    * SHA-256 integrity verification
    * Dataset fingerprinting
    * Parquet / Feather / CSV / JSON / JSONL ingestion
    * PyArrow accelerated Parquet batch streaming
    * Pandas chunked CSV/JSONL streaming
    * Controlled JSON document ingestion
    * Column projection
    * Encoding and malformed-record policies
    * Duplicate-column detection
    * Basic dataframe structural validation
    * Domain-specific Email / URL / Attachment loaders
    * Batch-level validation hooks
    * Ingestion statistics and provenance
    * Deterministic configuration
    * Defensive error handling
    * Safe logging with no raw payload logging
    * Extensible loader contracts for future modalities

Security Model
--------------
Datasets entering the ML pipeline are treated as untrusted input.

The loader therefore assumes that:

    1. Files may be malformed.
    2. Files may be unexpectedly large.
    3. Files may contain malicious or adversarial content.
    4. Column names may be duplicated or unexpected.
    5. Encodings may be invalid.
    6. JSON documents may be structurally different from JSONL.
    7. A file may be replaced between validation and reading.
    8. A path may be a symlink to an unintended location.
    9. A dataset may have been tampered with.
   10. A dataset may intentionally attempt resource exhaustion.

Important
---------
This module does NOT inspect or log raw email bodies, URLs, attachment contents,
credentials, authentication tokens, or other sensitive payloads.

The loader records metadata and structural information only.

Compatibility
-------------
Python: 3.10+
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAS_PYARROW: bool = True
except ImportError:
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]
    HAS_PYARROW = False

from src.utils.hashing import compute_file_sha256
from src.utils.logger import get_logger


logger = get_logger("src.data.loaders")


# ==============================================================================
# 1. MODULE CONSTANTS
# ==============================================================================

MODULE_VERSION: str = "13.0.0"

DEFAULT_BATCH_SIZE: int = 10_000

DEFAULT_MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024 * 1024
DEFAULT_MAX_ROWS: int = 50_000_000
DEFAULT_MAX_COLUMNS: int = 2_000

DEFAULT_HASH_CHUNK_SIZE: int = 1024 * 1024

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".parquet",
        ".feather",
        ".csv",
        ".txt",
        ".json",
        ".jsonl",
        ".ndjson",
    }
)

SUPPORTED_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".csv",
        ".txt",
        ".json",
        ".jsonl",
        ".ndjson",
    }
)

DEFAULT_ALLOWED_ENCODINGS: Tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
)

# Column names should remain reasonably constrained. This is deliberately
# conservative because column names eventually become feature identifiers.
COLUMN_NAME_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.\-]{0,255}$"
)

# Control characters in column names are almost always accidental or malicious.
CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")

# These are common fields that should never be emitted in logs.
SENSITIVE_COLUMN_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i).*password.*"),
    re.compile(r"(?i).*secret.*"),
    re.compile(r"(?i).*token.*"),
    re.compile(r"(?i).*api[_\-]?key.*"),
    re.compile(r"(?i).*private[_\-]?key.*"),
    re.compile(r"(?i).*authorization.*"),
    re.compile(r"(?i).*cookie.*"),
)


# ==============================================================================
# 2. EXCEPTIONS
# ==============================================================================


class DataLoaderError(RuntimeError):
    """Base exception for secure data-loading failures."""


class DatasetPathError(DataLoaderError):
    """Raised when a dataset path violates path-policy requirements."""


class DatasetFormatError(DataLoaderError):
    """Raised when a dataset format is unsupported or malformed."""


class DatasetIntegrityError(DataLoaderError):
    """Raised when dataset integrity verification fails."""


class DatasetResourceLimitError(DataLoaderError):
    """Raised when dataset resource limits are exceeded."""


class DatasetSchemaError(DataLoaderError):
    """Raised when dataset structural requirements are violated."""


class DatasetBatchError(DataLoaderError):
    """Raised when an individual streamed batch fails validation."""


# ==============================================================================
# 3. CONFIGURATION DATACLASSES
# ==============================================================================


@dataclass(frozen=True)
class LoaderSecurityPolicy:
    """
    Security and resource policy governing dataset ingestion.

    The defaults intentionally favor fail-closed behavior.
    """

    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
    max_rows: int = DEFAULT_MAX_ROWS
    max_columns: int = DEFAULT_MAX_COLUMNS

    allow_symlinks: bool = False
    require_regular_file: bool = True

    verify_sha256: bool = False
    expected_sha256: Optional[str] = None

    calculate_sha256: bool = True

    allowed_extensions: frozenset[str] = SUPPORTED_EXTENSIONS

    reject_duplicate_columns: bool = True
    reject_invalid_column_names: bool = True
    reject_control_characters: bool = True

    require_non_empty_dataset: bool = True

    allowed_encodings: Tuple[str, ...] = DEFAULT_ALLOWED_ENCODINGS

    max_cell_text_length: Optional[int] = 5_000_000

    fail_on_parser_warning: bool = False


@dataclass(frozen=True)
class LoaderOptions:
    """
    Runtime options shared by dataset loaders.
    """

    batch_size: int = DEFAULT_BATCH_SIZE

    csv_encoding: str = "utf-8"
    csv_separator: str = ","

    csv_on_bad_lines: str = "error"

    json_encoding: str = "utf-8"

    parquet_memory_map: bool = True

    columns: Optional[Tuple[str, ...]] = None

    copy_batches: bool = True

    deterministic_column_order: bool = False

    drop_empty_rows: bool = False


@dataclass
class DatasetStatistics:
    """
    Structural ingestion statistics.

    Raw payload values are intentionally excluded.
    """

    dataset_path: str
    dataset_name: str
    extension: str

    file_size_bytes: int = 0
    row_count: int = 0
    column_count: int = 0

    columns: List[str] = field(default_factory=list)

    sha256: Optional[str] = None

    batches_read: int = 0

    started_at_epoch: float = field(default_factory=time.time)
    completed_at_epoch: Optional[float] = None

    duration_seconds: Optional[float] = None

    backend: Optional[str] = None

    success: bool = False

    error_type: Optional[str] = None

    @property
    def rows_per_second(self) -> Optional[float]:
        """Returns ingestion throughput when timing information exists."""
        if self.duration_seconds is None or self.duration_seconds <= 0:
            return None

        return self.row_count / self.duration_seconds


@dataclass(frozen=True)
class DatasetManifest:
    """
    Immutable dataset provenance record.

    This can be persisted into MLflow, an artifact manifest, or an audit store.
    """

    dataset_name: str
    dataset_path: str
    extension: str

    sha256: Optional[str]

    file_size_bytes: int

    row_count: int
    column_count: int

    columns: Tuple[str, ...]

    loader_module_version: str

    created_at_epoch: float

    backend: Optional[str]


# ==============================================================================
# 4. TYPE ALIASES
# ==============================================================================

BatchValidator = Callable[[pd.DataFrame], None]


# ==============================================================================
# 5. INTERNAL UTILITY FUNCTIONS
# ==============================================================================


def _validate_positive_integer(value: int, name: str) -> None:
    """Validate a positive integer configuration value."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def _normalize_columns(
    columns: Optional[Sequence[str]],
) -> Optional[Tuple[str, ...]]:
    """Normalize and validate requested column projections."""

    if columns is None:
        return None

    normalized: List[str] = []

    for column in columns:
        if not isinstance(column, str):
            raise TypeError("All requested column names must be strings.")

        if not column:
            raise ValueError("Column names cannot be empty.")

        normalized.append(column)

    if len(normalized) != len(set(normalized)):
        raise ValueError("Requested columns contain duplicates.")

    return tuple(normalized)


def _validate_sha256(value: str) -> str:
    """Validate and normalize a SHA-256 hexadecimal digest."""

    if not isinstance(value, str):
        raise TypeError("SHA-256 digest must be a string.")

    normalized = value.strip().lower()

    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(
            "Expected SHA-256 digest must contain exactly 64 hexadecimal characters."
        )

    return normalized


def _safe_display_path(path: Path) -> str:
    """
    Return a safe path representation for logs.

    This does not expose dataset contents.
    """

    try:
        return str(path)
    except Exception:
        return "<unrepresentable-path>"


def _validate_extension(
    path: Path,
    allowed_extensions: Iterable[str],
) -> str:
    """Validate and return normalized dataset extension."""

    extension = path.suffix.lower()

    allowed = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in allowed_extensions
    }

    if extension not in allowed:
        raise DatasetFormatError(
            f"Unsupported dataset extension '{extension}'. "
            f"Allowed extensions: {sorted(allowed)}"
        )

    return extension


def _validate_dataset_path(
    filepath: Union[str, Path],
    policy: LoaderSecurityPolicy,
) -> Path:
    """
    Validate dataset path before opening it.

    Security checks include:

        * existence
        * regular-file requirement
        * symlink policy
        * file-size limit
        * supported extension
    """

    if isinstance(filepath, Path):
        raw_path = filepath
    elif isinstance(filepath, str):
        if not filepath.strip():
            raise DatasetPathError("Dataset path cannot be empty.")
        raw_path = Path(filepath)
    else:
        raise TypeError("filepath must be a string or pathlib.Path.")

    try:
        path = raw_path.expanduser()
    except Exception as exc:
        raise DatasetPathError(
            "Unable to normalize dataset path."
        ) from exc

    if not path.exists():
        raise FileNotFoundError(
            f"Data source path does not exist: '{_safe_display_path(path)}'"
        )

    if policy.require_regular_file and not path.is_file():
        raise DatasetPathError(
            f"Dataset path is not a regular file: '{_safe_display_path(path)}'"
        )

    if not policy.allow_symlinks and path.is_symlink():
        raise DatasetPathError(
            f"Symlink dataset paths are disabled by policy: "
            f"'{_safe_display_path(path)}'"
        )

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise DatasetPathError(
            f"Unable to inspect dataset metadata: '{_safe_display_path(path)}'"
        ) from exc

    if file_size > policy.max_file_size_bytes:
        raise DatasetResourceLimitError(
            f"Dataset exceeds maximum permitted file size. "
            f"size={file_size} bytes, "
            f"limit={policy.max_file_size_bytes} bytes."
        )

    _validate_extension(path, policy.allowed_extensions)

    return path.resolve()


def _validate_dataframe_structure(
    df: pd.DataFrame,
    policy: LoaderSecurityPolicy,
    requested_columns: Optional[Sequence[str]] = None,
) -> None:
    """
    Perform structural dataframe validation.

    This function deliberately does not inspect individual sensitive values.
    """

    if not isinstance(df, pd.DataFrame):
        raise DatasetSchemaError(
            f"Loader returned unexpected object type: {type(df).__name__}"
        )

    if len(df.columns) > policy.max_columns:
        raise DatasetResourceLimitError(
            f"Dataset contains {len(df.columns)} columns, "
            f"exceeding limit of {policy.max_columns}."
        )

    if len(df) > policy.max_rows:
        raise DatasetResourceLimitError(
            f"Dataset contains {len(df)} rows, "
            f"exceeding limit of {policy.max_rows}."
        )

    if policy.require_non_empty_dataset and df.empty:
        raise DatasetSchemaError("Dataset is empty.")

    column_names = [str(column) for column in df.columns]

    if policy.reject_duplicate_columns:
        duplicates = [
            column
            for column in set(column_names)
            if column_names.count(column) > 1
        ]

        if duplicates:
            raise DatasetSchemaError(
                f"Dataset contains duplicate column names: {duplicates[:20]}"
            )

    for column in column_names:
        if policy.reject_control_characters and CONTROL_CHARACTER_PATTERN.search(
            column
        ):
            raise DatasetSchemaError(
                f"Column name contains control characters: {column!r}"
            )

        if policy.reject_invalid_column_names and not COLUMN_NAME_PATTERN.fullmatch(
            column
        ):
            raise DatasetSchemaError(
                f"Invalid column name: {column!r}"
            )

    if requested_columns is not None:
        missing = [
            column
            for column in requested_columns
            if column not in df.columns
        ]

        if missing:
            raise DatasetSchemaError(
                f"Requested columns are missing from dataset: {missing[:20]}"
            )

    if policy.max_cell_text_length is not None:
        # Only inspect object/string-like columns.
        # We deliberately avoid logging offending values.
        for column in df.columns:
            series = df[column]

            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(
                series
            ):
                try:
                    max_length = series.dropna().astype(str).str.len().max()

                    if pd.notna(max_length) and max_length > policy.max_cell_text_length:
                        raise DatasetResourceLimitError(
                            f"Column '{column}' contains a text cell exceeding "
                            f"the configured maximum length."
                        )
                except DatasetResourceLimitError:
                    raise
                except Exception:
                    # Structural validation should not accidentally make
                    # otherwise valid extension dtypes unusable.
                    logger.debug(
                        "Unable to perform text-length validation for column.",
                        extra={
                            "extra_fields": {
                                "column": column
                                if not any(
                                    pattern.match(column)
                                    for pattern in SENSITIVE_COLUMN_PATTERNS
                                )
                                else "[REDACTED_COLUMN]"
                            }
                        },
                    )


def _drop_empty_rows_if_requested(
    df: pd.DataFrame,
    enabled: bool,
) -> pd.DataFrame:
    """Optionally remove completely empty rows."""

    if not enabled:
        return df

    if df.empty:
        return df

    return df.dropna(how="all").reset_index(drop=True)


def _stable_column_order(df: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministic lexicographic column ordering."""

    return df.reindex(sorted(df.columns, key=lambda value: str(value)), axis=1)


def _dataset_statistics_from_dataframe(
    path: Path,
    df: pd.DataFrame,
    backend: str,
) -> DatasetStatistics:
    """Create structural statistics for a fully loaded dataframe."""

    try:
        file_size = path.stat().st_size
    except OSError:
        file_size = 0

    return DatasetStatistics(
        dataset_path=str(path),
        dataset_name=path.name,
        extension=path.suffix.lower(),
        file_size_bytes=file_size,
        row_count=len(df),
        column_count=len(df.columns),
        columns=[str(column) for column in df.columns],
        backend=backend,
    )


# ==============================================================================
# 6. BASE DATA LOADER
# ==============================================================================


class BaseDataLoader(ABC):
    """
    Abstract contract for all domain-specific dataset loaders.

    Domain-specific loaders should not duplicate low-level file parsing logic.
    They inherit secure path validation, integrity checking, structural
    validation, and provenance facilities from this class.
    """

    loader_name: str = "base"

    def __init__(
        self,
        data_path: Union[str, Path],
        *,
        security_policy: Optional[LoaderSecurityPolicy] = None,
        options: Optional[LoaderOptions] = None,
        batch_validator: Optional[BatchValidator] = None,
        expected_sha256: Optional[str] = None,
    ) -> None:

        self.security_policy = security_policy or LoaderSecurityPolicy()
        self.options = options or LoaderOptions()
        self.batch_validator = batch_validator

        self.options_columns = _normalize_columns(self.options.columns)

        if expected_sha256 is not None:
            normalized_hash = _validate_sha256(expected_sha256)

            self.security_policy = LoaderSecurityPolicy(
                **{
                    **asdict(self.security_policy),
                    "verify_sha256": True,
                    "expected_sha256": normalized_hash,
                }
            )

        _validate_positive_integer(
            self.options.batch_size,
            "batch_size",
        )

        self.data_path = _validate_dataset_path(
            data_path,
            self.security_policy,
        )

        self._lock = threading.RLock()

        self._last_statistics: Optional[DatasetStatistics] = None

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load the complete dataset into memory."""
        raise NotImplementedError

    @abstractmethod
    def stream_batches(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Stream the dataset in bounded batches.

        Implementations must ensure that each yielded batch has passed
        structural validation before it reaches the caller.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    def verify_integrity(self) -> bool:
        """
        Verify dataset SHA-256 against configured expected digest.

        Returns
        -------
        bool
            True when verification succeeds or no expected checksum is configured.
        """

        if not self.security_policy.verify_sha256:
            return True

        expected = self.security_policy.expected_sha256

        if expected is None:
            raise DatasetIntegrityError(
                "SHA-256 verification enabled but expected_sha256 is missing."
            )

        expected = _validate_sha256(expected)

        actual = compute_file_sha256(
            self.data_path,
            chunk_size=DEFAULT_HASH_CHUNK_SIZE,
        )

        if not hmac_compare_digest(actual, expected):
            logger.error(
                "Dataset integrity verification failed.",
                extra={
                    "extra_fields": {
                        "dataset": self.data_path.name,
                        "integrity_status": "FAILED",
                    }
                },
            )

            raise DatasetIntegrityError(
                f"SHA-256 integrity verification failed for "
                f"'{self.data_path.name}'."
            )

        logger.info(
            "Dataset integrity verification succeeded.",
            extra={
                "extra_fields": {
                    "dataset": self.data_path.name,
                    "integrity_status": "VERIFIED",
                }
            },
        )

        return True

    # ------------------------------------------------------------------
    # Fingerprinting
    # ------------------------------------------------------------------

    def compute_fingerprint(self) -> Optional[str]:
        """
        Compute SHA-256 dataset fingerprint when enabled by policy.
        """

        if not self.security_policy.calculate_sha256:
            return None

        return compute_file_sha256(
            self.data_path,
            chunk_size=DEFAULT_HASH_CHUNK_SIZE,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_batch(self, batch: pd.DataFrame) -> None:
        """
        Validate a batch before it is exposed to downstream pipeline stages.
        """

        _validate_dataframe_structure(
            batch,
            self.security_policy,
            self.options_columns,
        )

        if self.batch_validator is not None:
            try:
                self.batch_validator(batch)
            except Exception as exc:
                raise DatasetBatchError(
                    "Custom dataset batch validator rejected a batch."
                ) from exc

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def last_statistics(self) -> Optional[DatasetStatistics]:
        """Return statistics from the latest completed load operation."""

        with self._lock:
            return self._last_statistics

    def _finalize_statistics(
        self,
        statistics: DatasetStatistics,
        *,
        success: bool,
        error: Optional[BaseException] = None,
    ) -> DatasetStatistics:

        statistics.completed_at_epoch = time.time()
        statistics.duration_seconds = (
            statistics.completed_at_epoch - statistics.started_at_epoch
        )

        statistics.success = success

        if error is not None:
            statistics.error_type = type(error).__name__

        with self._lock:
            self._last_statistics = statistics

        return statistics

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def build_manifest(
        self,
        statistics: DatasetStatistics,
    ) -> DatasetManifest:
        """
        Construct an immutable provenance manifest.
        """

        return DatasetManifest(
            dataset_name=statistics.dataset_name,
            dataset_path=statistics.dataset_path,
            extension=statistics.extension,
            sha256=statistics.sha256,
            file_size_bytes=statistics.file_size_bytes,
            row_count=statistics.row_count,
            column_count=statistics.column_count,
            columns=tuple(statistics.columns),
            loader_module_version=MODULE_VERSION,
            created_at_epoch=time.time(),
            backend=statistics.backend,
        )


# ==============================================================================
# 7. STANDARD DATASET LOADER
# ==============================================================================


class GenericDataLoader(BaseDataLoader):
    """
    Generic implementation supporting all formats exposed by load_raw_dataset.
    """

    loader_name = "generic"

    def load(self) -> pd.DataFrame:
        """Load a complete dataset with secure validation."""

        started = time.time()

        statistics = DatasetStatistics(
            dataset_path=str(self.data_path),
            dataset_name=self.data_path.name,
            extension=self.data_path.suffix.lower(),
            file_size_bytes=self.data_path.stat().st_size,
            backend="auto",
            started_at_epoch=started,
        )

        try:
            self.verify_integrity()

            df = load_raw_dataset(
                self.data_path,
                columns=list(self.options_columns)
                if self.options_columns
                else None,
                security_policy=self.security_policy,
                options=self.options,
            )

            self.validate_batch(df)

            if self.options.deterministic_column_order:
                df = _stable_column_order(df)

            statistics.row_count = len(df)
            statistics.column_count = len(df.columns)
            statistics.columns = [str(column) for column in df.columns]
            statistics.batches_read = 1
            statistics.sha256 = self.compute_fingerprint()

            statistics.backend = _infer_backend(self.data_path)

            self._finalize_statistics(
                statistics,
                success=True,
            )

            logger.info(
                "Dataset loaded successfully.",
                extra={
                    "extra_fields": {
                        "dataset": self.data_path.name,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "backend": statistics.backend,
                        "duration_seconds": statistics.duration_seconds,
                    }
                },
            )

            return df

        except Exception as exc:
            self._finalize_statistics(
                statistics,
                success=False,
                error=exc,
            )

            logger.error(
                "Dataset loading failed.",
                extra={
                    "extra_fields": {
                        "dataset": self.data_path.name,
                        "error_type": type(exc).__name__,
                    }
                },
                exc_info=True,
            )

            raise

    def stream_batches(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:

        effective_batch_size = (
            batch_size if batch_size is not None else self.options.batch_size
        )

        _validate_positive_integer(
            effective_batch_size,
            "batch_size",
        )

        self.verify_integrity()

        started = time.time()

        statistics = DatasetStatistics(
            dataset_path=str(self.data_path),
            dataset_name=self.data_path.name,
            extension=self.data_path.suffix.lower(),
            file_size_bytes=self.data_path.stat().st_size,
            backend=_infer_backend(self.data_path),
            started_at_epoch=started,
        )

        total_rows = 0

        try:
            for batch in _stream_dataset_batches(
                self.data_path,
                batch_size=effective_batch_size,
                security_policy=self.security_policy,
                options=self.options,
            ):

                if self.options.drop_empty_rows:
                    batch = _drop_empty_rows_if_requested(
                        batch,
                        enabled=True,
                    )

                if self.options.deterministic_column_order:
                    batch = _stable_column_order(batch)

                self.validate_batch(batch)

                total_rows += len(batch)

                if total_rows > self.security_policy.max_rows:
                    raise DatasetResourceLimitError(
                        f"Streaming dataset exceeded maximum row limit "
                        f"of {self.security_policy.max_rows}."
                    )

                statistics.row_count = total_rows
                statistics.column_count = len(batch.columns)

                if not statistics.columns:
                    statistics.columns = [
                        str(column) for column in batch.columns
                    ]

                statistics.batches_read += 1

                yield batch.copy() if self.options.copy_batches else batch

            statistics.sha256 = self.compute_fingerprint()

            self._finalize_statistics(
                statistics,
                success=True,
            )

        except Exception as exc:
            self._finalize_statistics(
                statistics,
                success=False,
                error=exc,
            )

            logger.error(
                "Dataset streaming failed.",
                extra={
                    "extra_fields": {
                        "dataset": self.data_path.name,
                        "batches_read": statistics.batches_read,
                        "rows_read": statistics.row_count,
                        "error_type": type(exc).__name__,
                    }
                },
                exc_info=True,
            )

            raise


# ==============================================================================
# 8. EMAIL DATA LOADER
# ==============================================================================


class EmailDataLoader(GenericDataLoader):
    """
    Secure loader for email feature datasets.

    Expected downstream domains may include:

        * header-derived features
        * authentication alignment
        * sender reputation
        * display-name spoofing
        * text-derived features
        * HTML statistics
        * URL density features
        * analyst labels

    Raw email bodies should not be logged by this loader.
    """

    loader_name = "email"

    DEFAULT_EXPECTED_COLUMNS: frozenset[str] = frozenset(
        {
            "sample_id",
            "label",
        }
    )

    def load(self) -> pd.DataFrame:
        logger.info(
            "Beginning email dataset ingestion.",
            extra={
                "extra_fields": {
                    "dataset": self.data_path.name,
                    "domain": "email",
                }
            },
        )

        return super().load()

    def stream_batches(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:

        for batch in super().stream_batches(batch_size):
            yield batch


# ==============================================================================
# 9. URL DATA LOADER
# ==============================================================================


class URLDataLoader(GenericDataLoader):
    """
    Secure loader for URL/domain intelligence datasets.

    Typical feature groups include:

        * hostname entropy
        * lexical URL features
        * domain age
        * TLD risk
        * redirect statistics
        * certificate metadata
        * DNS-derived signals
        * labels

    Raw URLs should never be written to logs.
    """

    loader_name = "url"

    def load(self) -> pd.DataFrame:
        logger.info(
            "Beginning URL dataset ingestion.",
            extra={
                "extra_fields": {
                    "dataset": self.data_path.name,
                    "domain": "url",
                }
            },
        )

        return super().load()

    def stream_batches(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:

        for batch in super().stream_batches(batch_size):
            yield batch


# ==============================================================================
# 10. ATTACHMENT DATA LOADER
# ==============================================================================


class AttachmentDataLoader(GenericDataLoader):
    """
    Secure loader for attachment/file-analysis datasets.

    Expected features may include:

        * file type
        * MIME metadata
        * entropy
        * archive ratio
        * macro indicators
        * executable indicators
        * PE metadata
        * sandbox-derived features
        * malware labels

    Raw attachment bytes are never loaded by this module unless represented
    explicitly by an upstream feature-extraction process.
    """

    loader_name = "attachment"

    def load(self) -> pd.DataFrame:
        logger.info(
            "Beginning attachment dataset ingestion.",
            extra={
                "extra_fields": {
                    "dataset": self.data_path.name,
                    "domain": "attachment",
                }
            },
        )

        return super().load()

    def stream_batches(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:

        for batch in super().stream_batches(batch_size):
            yield batch


# ==============================================================================
# 11. BACKEND INFERENCE
# ==============================================================================


def _infer_backend(path: Path) -> str:
    """Infer the underlying ingestion backend."""

    extension = path.suffix.lower()

    if extension == ".parquet":
        return "pyarrow.parquet" if HAS_PYARROW else "pandas.parquet"

    if extension == ".feather":
        return "pandas.feather"

    if extension in {".csv", ".txt"}:
        return "pandas.csv"

    if extension in {".json", ".jsonl", ".ndjson"}:
        return "pandas.json"

    return "unknown"


# ==============================================================================
# 12. STREAMING IMPLEMENTATION
# ==============================================================================


def _stream_dataset_batches(
    path: Path,
    *,
    batch_size: int,
    security_policy: LoaderSecurityPolicy,
    options: LoaderOptions,
) -> Generator[pd.DataFrame, None, None]:

    extension = path.suffix.lower()

    if extension == ".parquet":
        yield from _stream_parquet(
            path,
            batch_size=batch_size,
            security_policy=security_policy,
            options=options,
        )
        return

    if extension in {".csv", ".txt"}:
        yield from _stream_csv(
            path,
            batch_size=batch_size,
            security_policy=security_policy,
            options=options,
        )
        return

    if extension in {".jsonl", ".ndjson"}:
        yield from _stream_jsonl(
            path,
            batch_size=batch_size,
            security_policy=security_policy,
            options=options,
        )
        return

    if extension == ".json":
        # Standard JSON documents do not necessarily support true row-wise
        # streaming. We therefore enforce a bounded full-document read.
        dataframe = _load_json_document(
            path,
            columns=options.columns,
            security_policy=security_policy,
            options=options,
        )

        for start in range(0, len(dataframe), batch_size):
            yield dataframe.iloc[start:start + batch_size].copy()

        return

    # Feather generally does not expose reliable row-group streaming.
    dataframe = _load_feather(
        path,
        columns=options.columns,
        security_policy=security_policy,
        options=options,
    )

    for start in range(0, len(dataframe), batch_size):
        yield dataframe.iloc[start:start + batch_size].copy()


def _stream_parquet(
    path: Path,
    *,
    batch_size: int,
    security_policy: LoaderSecurityPolicy,
    options: LoaderOptions,
) -> Generator[pd.DataFrame, None, None]:

    if not HAS_PYARROW:
        logger.warning(
            "PyArrow unavailable; falling back to pandas Parquet reader."
        )

        dataframe = pd.read_parquet(
            path,
            columns=list(options.columns)
            if options.columns
            else None,
        )

        for start in range(0, len(dataframe), batch_size):
            yield dataframe.iloc[start:start + batch_size].copy()

        return

    parquet_file = pq.ParquetFile(str(path))

    metadata = parquet_file.metadata

    if metadata is not None:
        total_rows = metadata.num_rows

        if total_rows > security_policy.max_rows:
            raise DatasetResourceLimitError(
                f"Parquet metadata reports {total_rows} rows, "
                f"exceeding limit of {security_policy.max_rows}."
            )

    column_projection = (
        list(options.columns)
        if options.columns is not None
        else None
    )

    for record_batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=column_projection,
        use_threads=True,
    ):
        yield record_batch.to_pandas()


def _stream_csv(
    path: Path,
    *,
    batch_size: int,
    security_policy: LoaderSecurityPolicy,
    options: LoaderOptions,
) -> Generator[pd.DataFrame, None, None]:

    encoding = options.csv_encoding

    if encoding.lower() not in {
        value.lower()
        for value in security_policy.allowed_encodings
    }:
        raise DatasetFormatError(
            f"Encoding '{encoding}' is not permitted by the loader policy."
        )

    reader = pd.read_csv(
        path,
        usecols=list(options.columns)
        if options.columns
        else None,
        chunksize=batch_size,
        encoding=encoding,
        sep=options.csv_separator,
        on_bad_lines=options.csv_on_bad_lines,
        low_memory=True,
    )

    total_rows = 0

    for chunk in reader:
        total_rows += len(chunk)

        if total_rows > security_policy.max_rows:
            raise DatasetResourceLimitError(
                "CSV dataset exceeded configured row limit."
            )

        yield chunk


def _stream_jsonl(
    path: Path,
    *,
    batch_size: int,
    security_policy: LoaderSecurityPolicy,
    options: LoaderOptions,
) -> Generator[pd.DataFrame, None, None]:

    encoding = options.json_encoding

    if encoding.lower() not in {
        value.lower()
        for value in security_policy.allowed_encodings
    }:
        raise DatasetFormatError(
            f"Encoding '{encoding}' is not permitted by the loader policy."
        )

    reader = pd.read_json(
        path,
        lines=True,
        chunksize=batch_size,
        encoding=encoding,
    )

    total_rows = 0

    for chunk in reader:

        if options.columns:
            missing = [
                column
                for column in options.columns
                if column not in chunk.columns
            ]

            if missing:
                raise DatasetSchemaError(
                    f"Requested JSONL columns are missing: {missing[:20]}"
                )

            chunk = chunk[list(options.columns)]

        total_rows += len(chunk)

        if total_rows > security_policy.max_rows:
            raise DatasetResourceLimitError(
                "JSONL dataset exceeded configured row limit."
            )

        yield chunk


# ==============================================================================
# 13. FORMAT-SPECIFIC LOADERS
# ==============================================================================


def _load_json_document(
    path: Path,
    *,
    columns: Optional[Sequence[str]],
    security_policy: LoaderSecurityPolicy,
    options: LoaderOptions,
) -> pd.DataFrame:

    file_size = path.stat().st_size

    if file_size > security_policy.max_file_size_bytes:
        raise DatasetResourceLimitError(
            "JSON document exceeds configured file-size limit."
        )

    dataframe = pd.read_json(
        path,
        lines=False,
        encoding=options.json_encoding,
    )

    if columns:
        missing = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing:
            raise DatasetSchemaError(
                f"Requested JSON columns are missing: {missing[:20]}"
            )

        dataframe = dataframe[list(columns)]

    _validate_dataframe_structure(
        dataframe,
        security_policy,
        columns,
    )

    return dataframe


def _load_feather(
    path: Path,
    *,
    columns: Optional[Sequence[str]],
    security_policy: LoaderSecurityPolicy,
    options: LoaderOptions,
) -> pd.DataFrame:

    dataframe = pd.read_feather(
        path,
        columns=list(columns)
        if columns
        else None,
    )

    _validate_dataframe_structure(
        dataframe,
        security_policy,
        columns,
    )

    return dataframe


# ==============================================================================
# 14. UNIFIED INGESTION FACTORY
# ==============================================================================


def load_raw_dataset(
    filepath: Union[str, Path],
    columns: Optional[List[str]] = None,
    *,
    security_policy: Optional[LoaderSecurityPolicy] = None,
    options: Optional[LoaderOptions] = None,
    expected_sha256: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Secure unified dataset ingestion factory.

    Supported formats
    -----------------
    * Parquet
    * Feather
    * CSV
    * TXT
    * JSON
    * JSONL
    * NDJSON

    Parameters
    ----------
    filepath:
        Dataset path.

    columns:
        Optional projected columns.

    security_policy:
        Resource and integrity policy.

    options:
        Loader behavior configuration.

    expected_sha256:
        Optional integrity checksum.

    kwargs:
        Controlled reader-specific parameters.

    Notes
    -----
    Arbitrary keyword arguments are intentionally passed only to explicitly
    supported readers. Callers should not use this function as an unrestricted
    wrapper around pandas.
    """

    policy = security_policy or LoaderSecurityPolicy()

    if expected_sha256 is not None:
        normalized_hash = _validate_sha256(expected_sha256)

        policy = LoaderSecurityPolicy(
            **{
                **asdict(policy),
                "verify_sha256": True,
                "expected_sha256": normalized_hash,
            }
        )

    normalized_columns = _normalize_columns(columns)

    effective_options = options or LoaderOptions(
        columns=normalized_columns,
    )

    if options is not None and columns is not None:
        effective_options = LoaderOptions(
            **{
                **asdict(options),
                "columns": normalized_columns,
            }
        )

    path = _validate_dataset_path(
        filepath,
        policy,
    )

    if policy.verify_sha256:
        expected = policy.expected_sha256

        if expected is None:
            raise DatasetIntegrityError(
                "Integrity verification enabled without expected SHA-256."
            )

        actual = compute_file_sha256(path)

        if not hmac_compare_digest(actual, expected):
            raise DatasetIntegrityError(
                f"Dataset checksum mismatch for '{path.name}'."
            )

    extension = path.suffix.lower()

    start_time = time.time()

    try:

        if extension == ".parquet":

            reader_kwargs: Dict[str, Any] = {}

            if "engine" in kwargs:
                reader_kwargs["engine"] = kwargs["engine"]

            if "filters" in kwargs:
                reader_kwargs["filters"] = kwargs["filters"]

            if "use_threads" in kwargs:
                reader_kwargs["use_threads"] = kwargs["use_threads"]

            dataframe = pd.read_parquet(
                path,
                columns=list(normalized_columns)
                if normalized_columns
                else None,
                **reader_kwargs,
            )

            backend = "pandas.parquet"

        elif extension == ".feather":

            dataframe = pd.read_feather(
                path,
                columns=list(normalized_columns)
                if normalized_columns
                else None,
            )

            backend = "pandas.feather"

        elif extension in {".csv", ".txt"}:

            encoding = kwargs.pop(
                "encoding",
                effective_options.csv_encoding,
            )

            sep = kwargs.pop(
                "sep",
                effective_options.csv_separator,
            )

            on_bad_lines = kwargs.pop(
                "on_bad_lines",
                effective_options.csv_on_bad_lines,
            )

            dataframe = pd.read_csv(
                path,
                usecols=list(normalized_columns)
                if normalized_columns
                else None,
                encoding=encoding,
                sep=sep,
                on_bad_lines=on_bad_lines,
                **kwargs,
            )

            backend = "pandas.csv"

        elif extension in {".jsonl", ".ndjson"}:

            dataframe = pd.read_json(
                path,
                lines=True,
                encoding=effective_options.json_encoding,
                **kwargs,
            )

            if normalized_columns:
                dataframe = dataframe[list(normalized_columns)]

            backend = "pandas.jsonl"

        elif extension == ".json":

            dataframe = _load_json_document(
                path,
                columns=normalized_columns,
                security_policy=policy,
                options=effective_options,
            )

            backend = "pandas.json"

        else:
            raise DatasetFormatError(
                f"Unsupported dataset format: '{extension}'."
            )

        dataframe = _drop_empty_rows_if_requested(
            dataframe,
            effective_options.drop_empty_rows,
        )

        if effective_options.deterministic_column_order:
            dataframe = _stable_column_order(dataframe)

        _validate_dataframe_structure(
            dataframe,
            policy,
            normalized_columns,
        )

        elapsed = time.time() - start_time

        logger.info(
            "Dataset ingestion completed.",
            extra={
                "extra_fields": {
                    "dataset": path.name,
                    "format": extension,
                    "backend": backend,
                    "rows": len(dataframe),
                    "columns": len(dataframe.columns),
                    "duration_seconds": round(elapsed, 6),
                }
            },
        )

        return dataframe

    except DatasetLoaderError:
        raise

    except Exception as exc:
        logger.error(
            "Dataset ingestion failed.",
            extra={
                "extra_fields": {
                    "dataset": path.name,
                    "format": extension,
                    "error_type": type(exc).__name__,
                }
            },
            exc_info=True,
        )

        raise DataLoaderError(
            f"Failed to ingest dataset '{path.name}'."
        ) from exc


# ==============================================================================
# 15. DATASET LOADER FACTORY
# ==============================================================================


def create_data_loader(
    model_id: str,
    data_path: Union[str, Path],
    *,
    security_policy: Optional[LoaderSecurityPolicy] = None,
    options: Optional[LoaderOptions] = None,
    batch_validator: Optional[BatchValidator] = None,
    expected_sha256: Optional[str] = None,
) -> BaseDataLoader:
    """
    Construct a domain-specific loader from a model identifier.

    Supported model IDs
    -------------------
    email_classifier
    url_classifier
    attachment_classifier
    """

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string.")

    normalized = model_id.strip().lower()

    loader_mapping = {
        "email_classifier": EmailDataLoader,
        "email": EmailDataLoader,
        "url_classifier": URLDataLoader,
        "url": URLDataLoader,
        "attachment_classifier": AttachmentDataLoader,
        "attachment": AttachmentDataLoader,
    }

    loader_class = loader_mapping.get(normalized)

    if loader_class is None:
        raise ValueError(
            f"Unsupported model_id '{model_id}'. "
            f"Supported models: {sorted(loader_mapping)}"
        )

    return loader_class(
        data_path,
        security_policy=security_policy,
        options=options,
        batch_validator=batch_validator,
        expected_sha256=expected_sha256,
    )


# ==============================================================================
# 16. DATASET INSPECTION
# ==============================================================================


def inspect_dataset(
    filepath: Union[str, Path],
    *,
    security_policy: Optional[LoaderSecurityPolicy] = None,
) -> Dict[str, Any]:
    """
    Perform lightweight dataset inspection without exposing raw records.

    Returns structural metadata suitable for pre-ingestion checks.
    """

    policy = security_policy or LoaderSecurityPolicy()

    path = _validate_dataset_path(
        filepath,
        policy,
    )

    stat = path.stat()

    result: Dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "size_megabytes": round(
            stat.st_size / (1024 * 1024),
            4,
        ),
        "modified_time_epoch": stat.st_mtime,
        "is_regular_file": path.is_file(),
        "is_symlink": path.is_symlink(),
        "pyarrow_available": HAS_PYARROW,
        "loader_module_version": MODULE_VERSION,
    }

    if policy.calculate_sha256:
        result["sha256"] = compute_file_sha256(path)

    if path.suffix.lower() == ".parquet" and HAS_PYARROW:
        parquet_file = pq.ParquetFile(str(path))

        metadata = parquet_file.metadata

        if metadata is not None:
            result["row_count"] = metadata.num_rows
            result["row_groups"] = metadata.num_row_groups
            result["columns"] = [
                metadata.schema.column(index).name
                for index in range(metadata.schema.num_columns)
            ]

    return result


# ==============================================================================
# 17. DATASET MANIFEST CREATION
# ==============================================================================


def create_dataset_manifest(
    loader: BaseDataLoader,
) -> DatasetManifest:
    """
    Load a dataset and generate immutable structural provenance.
    """

    dataframe = loader.load()

    statistics = loader.last_statistics

    if statistics is None:
        raise DataLoaderError(
            "Loader did not produce ingestion statistics."
        )

    if not statistics.success:
        raise DataLoaderError(
            "Cannot create a manifest from an unsuccessful ingestion."
        )

    # Prevent unused-variable ambiguity while preserving a straightforward
    # manifest contract.
    del dataframe

    return loader.build_manifest(statistics)


# ==============================================================================
# 18. CONSTANT-TIME COMPARISON HELPER
# ==============================================================================


def hmac_compare_digest(value_a: str, value_b: str) -> bool:
    """
    Constant-time string comparison wrapper.

    Dataset checksums are not normally secret, but constant-time comparison
    is a harmless defensive default and prevents callers from introducing
    unsafe equality checks into integrity workflows.
    """

    import hmac

    return hmac.compare_digest(
        value_a.lower(),
        value_b.lower(),
    )


# ==============================================================================
# 19. PUBLIC API
# ==============================================================================


__all__ = [
    # Metadata
    "MODULE_VERSION",
    "HAS_PYARROW",
    "SUPPORTED_EXTENSIONS",

    # Exceptions
    "DataLoaderError",
    "DatasetPathError",
    "DatasetFormatError",
    "DatasetIntegrityError",
    "DatasetResourceLimitError",
    "DatasetSchemaError",
    "DatasetBatchError",

    # Configuration
    "LoaderSecurityPolicy",
    "LoaderOptions",

    # Provenance
    "DatasetStatistics",
    "DatasetManifest",

    # Loader contracts
    "BaseDataLoader",
    "GenericDataLoader",

    # Domain loaders
    "EmailDataLoader",
    "URLDataLoader",
    "AttachmentDataLoader",

    # Factory
    "create_data_loader",

    # Unified ingestion
    "load_raw_dataset",

    # Inspection
    "inspect_dataset",
    "create_dataset_manifest",

    # Integrity helper
    "hmac_compare_digest",
]
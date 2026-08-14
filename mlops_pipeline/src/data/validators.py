"""
PhishGuard AI - Zero-Trust Data Quality & Schema Validation Engine
===================================================================

Version: 14.0.0
File Path: src/data/validators.py

Purpose
-------
Enterprise-grade validation and data-quality enforcement for the PhishGuard AI
MLOps pipeline.

This module sits at the trust boundary between externally supplied datasets and
the ML training/evaluation pipeline. It therefore follows a zero-trust model:

    UNTRUSTED DATA
          |
          v
    Structural validation
          |
          v
    Schema validation
          |
          v
    Type validation
          |
          v
    Null / NaN / Inf validation
          |
          v
    Duplicate / cardinality checks
          |
          v
    Range / domain validation
          |
          v
    String / text validation
          |
          v
    Statistical sanity checks
          |
          v
    Dataset integrity decision
          |
          v
    TRAIN / REJECT / QUARANTINE

Security principles
-------------------
1. Never trust externally supplied DataFrames.
2. Never silently accept an unknown validation rule.
3. Never expose raw sensitive values in validation logs.
4. Prefer deterministic validation output.
5. Separate errors from warnings.
6. Support strict and permissive validation policies.
7. Provide machine-readable validation reports.
8. Detect structural data corruption before ML processing.
9. Detect suspicious statistical anomalies before training.
10. Preserve enough provenance to reproduce validation decisions.

The module intentionally does NOT modify the input DataFrame. Validation should
be side-effect free.

Optional dependencies
---------------------
Only pandas and NumPy are required.

The implementation is designed to work with:
    - pandas NumPy dtypes
    - pandas extension dtypes
    - nullable integers
    - nullable booleans
    - categorical columns
    - datetime columns
    - timezone-aware datetime columns
    - object/string columns

Typical usage
-------------

    validator = SchemaValidator(
        required_columns=["url", "label"],
        column_types={
            "url": "string",
            "label": "binary",
        },
    )

    result = validator.validate(df)

    if not result.is_valid:
        raise ValueError(result.summary())

or:

    result = validate_dataset_integrity(
        df,
        required_columns=["url", "label"],
        column_types={
            "url": "string",
            "label": "binary",
        },
        max_null_rate=0.20,
    )

    if result.is_valid:
        train(df)

"""

from __future__ import annotations

# ============================================================================
# STANDARD LIBRARY
# ============================================================================

import math
import re
import statistics
import time
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# ============================================================================
# THIRD-PARTY
# ============================================================================

import numpy as np
import pandas as pd

# ============================================================================
# INTERNAL
# ============================================================================

from src.utils.hashing import compute_object_hash
from src.utils.logger import get_logger

logger = get_logger("src.data.validators")


# ============================================================================
# MODULE CONSTANTS
# ============================================================================

VALIDATION_ENGINE_VERSION = "14.0.0"

DEFAULT_MAX_NULL_RATE = 0.20
DEFAULT_MAX_DUPLICATE_RATE = 0.50
DEFAULT_MAX_CARDINALITY_RATIO = 0.95
DEFAULT_MAX_STRING_LENGTH = 1_000_000
DEFAULT_SAMPLE_SIZE = 10_000

# Protect the validator itself from accidentally processing pathological
# configurations.
MAX_VALIDATION_RULES = 10_000
MAX_ALLOWED_VALUES = 100_000
MAX_COLUMNS_FOR_DETAILED_REPORT = 10_000

SUPPORTED_COLUMN_TYPES: Set[str] = {
    "numeric",
    "integer",
    "float",
    "number",
    "string",
    "text",
    "object",
    "boolean",
    "bool",
    "datetime",
    "date",
    "categorical",
    "category",
    "binary",
    "email",
    "url",
}

COLUMN_NAME_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.\-]{0,254}$"
)

# Common names which frequently indicate target/label columns.
DEFAULT_LABEL_NAMES: Set[str] = {
    "label",
    "target",
    "class",
    "y",
    "is_phishing",
    "is_malicious",
    "malicious",
    "phishing",
}


# ============================================================================
# ENUMERATIONS
# ============================================================================


class ValidationSeverity(str, Enum):
    """Severity assigned to an individual validation finding."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ValidationStatus(str, Enum):
    """Overall validation execution status."""

    PASSED = "PASSED"
    PASSED_WITH_WARNINGS = "PASSED_WITH_WARNINGS"
    FAILED = "FAILED"
    ERROR = "ERROR"


# ============================================================================
# VALIDATION ISSUE
# ============================================================================


@dataclass
class ValidationIssue:
    """
    Machine-readable representation of a single validation finding.

    Important
    ---------
    This class intentionally does not contain raw offending values. Validation
    reports may be uploaded to centralized logging/SIEM systems and therefore
    must not become a source of data leakage.
    """

    code: str
    message: str
    severity: ValidationSeverity
    column: Optional[str] = None
    category: str = "general"
    observed: Optional[Any] = None
    expected: Optional[Any] = None
    count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "column": self.column,
            "category": self.category,
            "observed": self.observed,
            "expected": self.expected,
            "count": self.count,
            "metadata": self.metadata,
        }


# ============================================================================
# VALIDATION RESULT
# ============================================================================


@dataclass
class ValidationResult:
    """
    Complete validation report.

    Backward-compatible fields from the original implementation are retained,
    while additional enterprise-grade metrics are included.
    """

    is_valid: bool

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    missing_columns: List[str] = field(default_factory=list)

    type_mismatches: Dict[str, str] = field(default_factory=dict)

    null_rates: Dict[str, float] = field(default_factory=dict)

    out_of_bounds_counts: Dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Extended structural metrics
    # ------------------------------------------------------------------

    row_count: int = 0
    column_count: int = 0

    duplicate_row_count: int = 0
    duplicate_row_rate: float = 0.0

    duplicate_column_names: List[str] = field(default_factory=list)

    empty_string_counts: Dict[str, int] = field(default_factory=dict)

    infinite_counts: Dict[str, int] = field(default_factory=dict)

    nan_counts: Dict[str, int] = field(default_factory=dict)

    unique_counts: Dict[str, int] = field(default_factory=dict)

    cardinality_rates: Dict[str, float] = field(default_factory=dict)

    constant_columns: List[str] = field(default_factory=list)

    suspicious_columns: List[str] = field(default_factory=list)

    string_length_violations: Dict[str, int] = field(default_factory=dict)

    invalid_category_counts: Dict[str, int] = field(default_factory=dict)

    issues: List[ValidationIssue] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Dataset fingerprint / execution metadata
    # ------------------------------------------------------------------

    validation_id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )

    validation_engine_version: str = VALIDATION_ENGINE_VERSION

    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0

    dataset_fingerprint: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Computed helpers
    # ------------------------------------------------------------------

    @property
    def error_count(self) -> int:
        """Number of ERROR/CRITICAL findings."""

        return sum(
            1
            for issue in self.issues
            if issue.severity in {
                ValidationSeverity.ERROR,
                ValidationSeverity.CRITICAL,
            }
        )

    @property
    def warning_count(self) -> int:
        """Number of warning findings."""

        return sum(
            1
            for issue in self.issues
            if issue.severity == ValidationSeverity.WARNING
        )

    @property
    def status(self) -> ValidationStatus:
        """Return the overall validation status."""

        if not self.is_valid:
            return ValidationStatus.FAILED

        if self.warning_count > 0 or self.warnings:
            return ValidationStatus.PASSED_WITH_WARNINGS

        return ValidationStatus.PASSED

    def add_issue(
        self,
        code: str,
        message: str,
        severity: ValidationSeverity,
        *,
        column: Optional[str] = None,
        category: str = "general",
        observed: Optional[Any] = None,
        expected: Optional[Any] = None,
        count: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Add a structured issue and update compatibility error/warning lists.
        """

        issue = ValidationIssue(
            code=code,
            message=message,
            severity=severity,
            column=column,
            category=category,
            observed=observed,
            expected=expected,
            count=count,
            metadata=dict(metadata or {}),
        )

        self.issues.append(issue)

        if severity in {
            ValidationSeverity.ERROR,
            ValidationSeverity.CRITICAL,
        }:
            self.errors.append(message)
            self.is_valid = False

        elif severity == ValidationSeverity.WARNING:
            self.warnings.append(message)

    def summary(self) -> str:
        """Return concise human-readable validation summary."""

        return (
            f"Validation status={self.status.value}; "
            f"valid={self.is_valid}; "
            f"rows={self.row_count}; "
            f"columns={self.column_count}; "
            f"errors={self.error_count}; "
            f"warnings={self.warning_count}; "
            f"duplicates={self.duplicate_row_count}; "
            f"validation_id={self.validation_id}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert report into a JSON-compatible dictionary.

        Structured issues are retained for downstream MLOps systems.
        """

        return {
            "validation_id": self.validation_id,
            "validation_engine_version": self.validation_engine_version,
            "status": self.status.value,
            "is_valid": self.is_valid,

            "error_count": self.error_count,
            "warning_count": self.warning_count,

            "errors": list(self.errors),
            "warnings": list(self.warnings),

            "missing_columns": list(self.missing_columns),
            "type_mismatches": dict(self.type_mismatches),

            "null_rates": dict(self.null_rates),
            "out_of_bounds_counts": dict(
                self.out_of_bounds_counts
            ),

            "row_count": self.row_count,
            "column_count": self.column_count,

            "duplicate_row_count": self.duplicate_row_count,
            "duplicate_row_rate": self.duplicate_row_rate,

            "duplicate_column_names": list(
                self.duplicate_column_names
            ),

            "empty_string_counts": dict(
                self.empty_string_counts
            ),

            "infinite_counts": dict(
                self.infinite_counts
            ),

            "nan_counts": dict(self.nan_counts),

            "unique_counts": dict(self.unique_counts),

            "cardinality_rates": dict(
                self.cardinality_rates
            ),

            "constant_columns": list(
                self.constant_columns
            ),

            "suspicious_columns": list(
                self.suspicious_columns
            ),

            "string_length_violations": dict(
                self.string_length_violations
            ),

            "invalid_category_counts": dict(
                self.invalid_category_counts
            ),

            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],

            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,

            "dataset_fingerprint": self.dataset_fingerprint,

            "metadata": dict(self.metadata),
        }


# ============================================================================
# SCHEMA VALIDATOR
# ============================================================================


class SchemaValidator:
    """
    Enterprise schema enforcement engine.

    Responsibilities
    ----------------
    - Required columns
    - Unexpected columns
    - Duplicate column names
    - Column naming policy
    - Broad dtype validation
    - Numeric bounds
    - Allowed categorical values
    - String lengths
    - Cardinality
    - Binary labels
    - Nullability
    - Optional strict mode

    The validator does not mutate the DataFrame.
    """

    def __init__(
        self,
        required_columns: Sequence[str],
        column_types: Optional[Mapping[str, str]] = None,
        value_bounds: Optional[
            Mapping[str, Tuple[Optional[float], Optional[float]]]
        ] = None,
        *,
        allowed_columns: Optional[Sequence[str]] = None,
        allow_extra_columns: bool = True,
        nullable_columns: Optional[Mapping[str, bool]] = None,
        allowed_values: Optional[
            Mapping[str, Iterable[Any]]
        ] = None,
        max_string_lengths: Optional[Mapping[str, int]] = None,
        min_string_lengths: Optional[Mapping[str, int]] = None,
        max_cardinality: Optional[
            Mapping[str, int]
        ] = None,
        max_cardinality_rate: Optional[
            Mapping[str, float]
        ] = None,
        strict: bool = True,
        fail_on_type_mismatch: Optional[bool] = None,
        fail_on_bounds_violation: bool = True,
        fail_on_invalid_categories: bool = True,
        validate_column_names: bool = True,
    ) -> None:

        self.required_columns = list(required_columns)

        self.column_types = dict(column_types or {})

        self.value_bounds = dict(value_bounds or {})

        self.allowed_columns = (
            set(allowed_columns)
            if allowed_columns is not None
            else None
        )

        self.allow_extra_columns = allow_extra_columns

        self.nullable_columns = dict(
            nullable_columns or {}
        )

        self.allowed_values = {
            key: set(values)
            for key, values in (
                allowed_values or {}
            ).items()
        }

        self.max_string_lengths = dict(
            max_string_lengths or {}
        )

        self.min_string_lengths = dict(
            min_string_lengths or {}
        )

        self.max_cardinality = dict(
            max_cardinality or {}
        )

        self.max_cardinality_rate = dict(
            max_cardinality_rate or {}
        )

        self.strict = strict

        self.fail_on_type_mismatch = (
            strict
            if fail_on_type_mismatch is None
            else fail_on_type_mismatch
        )

        self.fail_on_bounds_violation = (
            fail_on_bounds_violation
        )

        self.fail_on_invalid_categories = (
            fail_on_invalid_categories
        )

        self.validate_column_names = (
            validate_column_names
        )

        self._validate_configuration()

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------

    def _validate_configuration(self) -> None:
        """Validate validator configuration before processing data."""

        if len(self.required_columns) > MAX_VALIDATION_RULES:
            raise ValueError(
                "Too many required-column rules configured."
            )

        if len(set(self.required_columns)) != len(
            self.required_columns
        ):
            raise ValueError(
                "required_columns contains duplicate names."
            )

        unknown_types = set(
            self.column_types.values()
        ) - SUPPORTED_COLUMN_TYPES

        if unknown_types:
            raise ValueError(
                "Unsupported expected column types: "
                + ", ".join(sorted(unknown_types))
            )

        for column, bounds in self.value_bounds.items():
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(
                    f"Bounds for '{column}' must be "
                    "(min_value, max_value)."
                )

            minimum, maximum = bounds

            if (
                minimum is not None
                and maximum is not None
                and minimum > maximum
            ):
                raise ValueError(
                    f"Invalid bounds for '{column}': "
                    "minimum is greater than maximum."
                )

        for column, rate in self.max_cardinality_rate.items():
            if not 0.0 <= rate <= 1.0:
                raise ValueError(
                    f"Cardinality rate for '{column}' "
                    "must be between 0 and 1."
                )

        for column, length in self.max_string_lengths.items():
            if length < 0:
                raise ValueError(
                    f"Maximum string length for '{column}' "
                    "cannot be negative."
                )

        for column, length in self.min_string_lengths.items():
            if length < 0:
                raise ValueError(
                    f"Minimum string length for '{column}' "
                    "cannot be negative."
                )

        if len(self.allowed_values) > MAX_VALIDATION_RULES:
            raise ValueError(
                "Too many allowed-value validation rules."
            )

        for column, values in self.allowed_values.items():
            if len(values) > MAX_ALLOWED_VALUES:
                raise ValueError(
                    f"Too many allowed values configured "
                    f"for column '{column}'."
                )

    # ------------------------------------------------------------------
    # Main validation
    # ------------------------------------------------------------------

    def validate(
        self,
        df: pd.DataFrame,
        result: Optional[ValidationResult] = None,
    ) -> ValidationResult:
        """Validate DataFrame against the configured schema."""

        if result is None:
            result = ValidationResult(is_valid=True)

        if not isinstance(df, pd.DataFrame):
            result.add_issue(
                "INVALID_DATAFRAME",
                "Input dataset is not a pandas DataFrame.",
                ValidationSeverity.CRITICAL,
                category="structure",
            )
            return result

        result.row_count = len(df)
        result.column_count = len(df.columns)

        # Structural validation first.
        self._validate_structure(df, result)

        # Stop expensive checks if the structure is unusable.
        if result.missing_columns:
            return result

        self._validate_types(df, result)
        self._validate_nullability(df, result)
        self._validate_bounds(df, result)
        self._validate_allowed_values(df, result)
        self._validate_string_lengths(df, result)
        self._validate_cardinality(df, result)

        return result

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def _validate_structure(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:

        columns = list(df.columns)

        # Duplicate columns are dangerous because df["column"] semantics
        # become ambiguous.
        duplicated_mask = pd.Index(columns).duplicated(
            keep=False
        )

        duplicates = sorted(
            {
                str(column)
                for column, duplicated in zip(
                    columns,
                    duplicated_mask,
                )
                if duplicated
            }
        )

        if duplicates:
            result.duplicate_column_names = duplicates

            result.add_issue(
                "DUPLICATE_COLUMN_NAMES",
                "Dataset contains duplicate column names.",
                ValidationSeverity.CRITICAL,
                category="structure",
                count=len(duplicates),
                metadata={
                    "duplicate_column_count": len(
                        duplicates
                    )
                },
            )

        missing = [
            column
            for column in self.required_columns
            if column not in df.columns
        ]

        if missing:
            result.missing_columns.extend(missing)

            for column in missing:
                result.add_issue(
                    "MISSING_REQUIRED_COLUMN",
                    f"Required column missing: '{column}'.",
                    ValidationSeverity.CRITICAL,
                    column=column,
                    category="schema",
                )

        if self.allowed_columns is not None:
            unexpected = [
                str(column)
                for column in df.columns
                if column not in self.allowed_columns
            ]

            if unexpected and not self.allow_extra_columns:
                result.add_issue(
                    "UNEXPECTED_COLUMNS",
                    "Dataset contains columns outside "
                    "the allowed schema.",
                    ValidationSeverity.ERROR,
                    category="schema",
                    count=len(unexpected),
                    metadata={
                        "unexpected_columns": unexpected[
                            :100
                        ]
                    },
                )

        if self.validate_column_names:
            for column in df.columns:
                column_str = str(column)

                if not COLUMN_NAME_PATTERN.fullmatch(
                    column_str
                ):
                    result.add_issue(
                        "INVALID_COLUMN_NAME",
                        (
                            f"Column name '{column_str}' "
                            "does not satisfy the "
                            "dataset naming policy."
                        ),
                        (
                            ValidationSeverity.ERROR
                            if self.strict
                            else ValidationSeverity.WARNING
                        ),
                        column=column_str,
                        category="schema",
                    )

    # ------------------------------------------------------------------
    # Types
    # ------------------------------------------------------------------

    def _validate_types(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:

        for column, expected_type in self.column_types.items():

            if column not in df.columns:
                continue

            series = df[column]

            if self._check_column_type(
                series,
                expected_type,
            ):
                continue

            actual_type = str(series.dtype)

            result.type_mismatches[column] = (
                f"Expected '{expected_type}', "
                f"got '{actual_type}'"
            )

            severity = (
                ValidationSeverity.ERROR
                if self.fail_on_type_mismatch
                else ValidationSeverity.WARNING
            )

            result.add_issue(
                "TYPE_MISMATCH",
                (
                    f"Column '{column}' type mismatch: "
                    f"expected '{expected_type}', "
                    f"found '{actual_type}'."
                ),
                severity,
                column=column,
                category="schema",
                observed=actual_type,
                expected=expected_type,
            )

    # ------------------------------------------------------------------
    # Nullability
    # ------------------------------------------------------------------

    def _validate_nullability(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:

        for column, nullable in self.nullable_columns.items():

            if column not in df.columns:
                continue

            null_count = int(
                df[column].isna().sum()
            )

            if null_count == 0:
                continue

            if not nullable:
                result.add_issue(
                    "NULL_NOT_ALLOWED",
                    (
                        f"Column '{column}' contains "
                        f"{null_count} null values but "
                        "nulls are not allowed."
                    ),
                    ValidationSeverity.ERROR,
                    column=column,
                    category="quality",
                    count=null_count,
                )

    # ------------------------------------------------------------------
    # Numeric bounds
    # ------------------------------------------------------------------

    def _validate_bounds(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:

        for column, (
            minimum,
            maximum,
        ) in self.value_bounds.items():

            if column not in df.columns:
                continue

            series = df[column]

            if not pd.api.types.is_numeric_dtype(
                series
            ):
                result.add_issue(
                    "BOUNDS_NON_NUMERIC_COLUMN",
                    (
                        f"Bounds configured for "
                        f"non-numeric column '{column}'."
                    ),
                    ValidationSeverity.ERROR,
                    column=column,
                    category="schema",
                )
                continue

            numeric = series.dropna()

            if numeric.empty:
                continue

            mask = pd.Series(
                False,
                index=numeric.index,
            )

            if minimum is not None:
                mask |= numeric < minimum

            if maximum is not None:
                mask |= numeric > maximum

            count = int(mask.sum())

            if count == 0:
                continue

            result.out_of_bounds_counts[
                column
            ] = count

            severity = (
                ValidationSeverity.ERROR
                if self.fail_on_bounds_violation
                else ValidationSeverity.WARNING
            )

            result.add_issue(
                "VALUE_OUT_OF_BOUNDS",
                (
                    f"Column '{column}' contains "
                    f"{count} values outside the "
                    f"allowed range [{minimum}, {maximum}]."
                ),
                severity,
                column=column,
                category="quality",
                count=count,
                expected={
                    "minimum": minimum,
                    "maximum": maximum,
                },
            )

    # ------------------------------------------------------------------
    # Allowed categorical values
    # ------------------------------------------------------------------

    def _validate_allowed_values(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:

        for column, allowed in self.allowed_values.items():

            if column not in df.columns:
                continue

            series = df[column].dropna()

            if series.empty:
                continue

            invalid_mask = ~series.isin(allowed)

            invalid_count = int(
                invalid_mask.sum()
            )

            if invalid_count == 0:
                continue

            result.invalid_category_counts[
                column
            ] = invalid_count

            severity = (
                ValidationSeverity.ERROR
                if self.fail_on_invalid_categories
                else ValidationSeverity.WARNING
            )

            result.add_issue(
                "INVALID_CATEGORY",
                (
                    f"Column '{column}' contains "
                    f"{invalid_count} values outside "
                    "the configured allowed-value set."
                ),
                severity,
                column=column,
                category="domain",
                count=invalid_count,
                metadata={
                    "allowed_value_count": len(
                        allowed
                    )
                },
            )

    # ------------------------------------------------------------------
    # String lengths
    # ------------------------------------------------------------------

    def _validate_string_lengths(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:

        columns = set(
            self.max_string_lengths
        ).union(
            self.min_string_lengths
        )

        for column in columns:

            if column not in df.columns:
                continue

            series = df[column].dropna()

            if series.empty:
                continue

            lengths = series.astype(str).str.len()

            violation_mask = pd.Series(
                False,
                index=lengths.index,
            )

            minimum = self.min_string_lengths.get(
                column
            )
            maximum = self.max_string_lengths.get(
                column
            )

            if minimum is not None:
                violation_mask |= lengths < minimum

            if maximum is not None:
                violation_mask |= lengths > maximum

            count = int(
                violation_mask.sum()
            )

            if count == 0:
                continue

            result.string_length_violations[
                column
            ] = count

            result.add_issue(
                "STRING_LENGTH_VIOLATION",
                (
                    f"Column '{column}' contains "
                    f"{count} strings outside the "
                    "configured length constraints."
                ),
                ValidationSeverity.ERROR,
                column=column,
                category="quality",
                count=count,
                expected={
                    "minimum": minimum,
                    "maximum": maximum,
                },
            )

    # ------------------------------------------------------------------
    # Cardinality
    # ------------------------------------------------------------------

    def _validate_cardinality(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:

        configured_columns = (
            set(self.max_cardinality)
            | set(self.max_cardinality_rate)
        )

        for column in configured_columns:

            if column not in df.columns:
                continue

            series = df[column].dropna()

            unique_count = int(
                series.nunique(dropna=True)
            )

            result.unique_counts[
                column
            ] = unique_count

            if len(series) > 0:
                rate = unique_count / len(series)
            else:
                rate = 0.0

            result.cardinality_rates[
                column
            ] = round(rate, 6)

            maximum = self.max_cardinality.get(
                column
            )

            if (
                maximum is not None
                and unique_count > maximum
            ):
                result.add_issue(
                    "MAX_CARDINALITY_EXCEEDED",
                    (
                        f"Column '{column}' has "
                        f"{unique_count} unique values, "
                        f"exceeding maximum {maximum}."
                    ),
                    ValidationSeverity.WARNING,
                    column=column,
                    category="quality",
                    observed=unique_count,
                    expected=maximum,
                )

            maximum_rate = (
                self.max_cardinality_rate.get(
                    column
                )
            )

            if (
                maximum_rate is not None
                and rate > maximum_rate
            ):
                result.add_issue(
                    "MAX_CARDINALITY_RATE_EXCEEDED",
                    (
                        f"Column '{column}' cardinality "
                        f"rate ({rate:.2%}) exceeds "
                        f"maximum ({maximum_rate:.2%})."
                    ),
                    ValidationSeverity.WARNING,
                    column=column,
                    category="quality",
                    observed=round(rate, 6),
                    expected=maximum_rate,
                )

    # ------------------------------------------------------------------
    # Type helper
    # ------------------------------------------------------------------

    @staticmethod
    def _check_column_type(
        series: pd.Series,
        expected_type: str,
    ) -> bool:

        expected = expected_type.lower().strip()

        if expected not in SUPPORTED_COLUMN_TYPES:
            # Never silently accept an unknown rule.
            raise ValueError(
                f"Unsupported expected column type: "
                f"'{expected_type}'."
            )

        if expected in {
            "numeric",
            "number",
        }:
            return pd.api.types.is_numeric_dtype(
                series
            )

        if expected == "integer":
            return pd.api.types.is_integer_dtype(
                series
            )

        if expected == "float":
            return pd.api.types.is_float_dtype(
                series
            )

        if expected in {
            "string",
            "text",
        }:
            return (
                pd.api.types.is_string_dtype(series)
                or pd.api.types.is_object_dtype(
                    series
                )
            )

        if expected == "object":
            return pd.api.types.is_object_dtype(
                series
            )

        if expected in {
            "boolean",
            "bool",
            "binary",
        }:
            if expected == "binary":
                return SchemaValidator._is_binary_series(
                    series
                )

            return pd.api.types.is_bool_dtype(
                series
            )

        if expected in {
            "datetime",
            "date",
        }:
            return pd.api.types.is_datetime64_any_dtype(
                series
            )

        if expected in {
            "categorical",
            "category",
        }:
            return isinstance(
                series.dtype,
                pd.CategoricalDtype,
            )

        if expected == "email":
            return (
                pd.api.types.is_string_dtype(series)
                or pd.api.types.is_object_dtype(
                    series
                )
            )

        if expected == "url":
            return (
                pd.api.types.is_string_dtype(series)
                or pd.api.types.is_object_dtype(
                    series
                )
            )

        return False

    @staticmethod
    def _is_binary_series(
        series: pd.Series,
    ) -> bool:

        values = series.dropna()

        if values.empty:
            return True

        if pd.api.types.is_bool_dtype(
            series
        ):
            return True

        unique_values = set(
            values.astype(str).str.lower()
        )

        return unique_values.issubset(
            {
                "0",
                "1",
                "true",
                "false",
            }
        )


# ============================================================================
# DATA QUALITY VALIDATOR
# ============================================================================


class DataQualityValidator:
    """
    Advanced dataset quality validation engine.

    Detects:
        - Empty datasets
        - Null rates
        - NaN values
        - Infinite values
        - Empty strings
        - Duplicate rows
        - Constant columns
        - High cardinality
        - Numeric anomalies
        - Invalid numeric values
        - Extremely large values
        - Text-length anomalies

    This validator is intentionally configurable because different datasets
    have different legitimate characteristics.
    """

    def __init__(
        self,
        max_allowed_null_rate: float = DEFAULT_MAX_NULL_RATE,
        allow_empty_dataframe: bool = False,
        *,
        max_allowed_duplicate_rate: float = DEFAULT_MAX_DUPLICATE_RATE,
        fail_on_null_rate: bool = False,
        fail_on_duplicate_rate: bool = False,
        fail_on_constant_columns: bool = False,
        detect_empty_strings: bool = True,
        fail_on_infinite: bool = True,
        detect_constant_columns: bool = True,
        minimum_rows: Optional[int] = None,
        maximum_rows: Optional[int] = None,
        minimum_columns: Optional[int] = None,
        maximum_columns: Optional[int] = None,
        check_numeric_extremes: bool = True,
        numeric_zscore_threshold: float = 12.0,
        max_string_length: int = DEFAULT_MAX_STRING_LENGTH,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
    ) -> None:

        self.max_allowed_null_rate = (
            max_allowed_null_rate
        )

        self.allow_empty_dataframe = (
            allow_empty_dataframe
        )

        self.max_allowed_duplicate_rate = (
            max_allowed_duplicate_rate
        )

        self.fail_on_null_rate = (
            fail_on_null_rate
        )

        self.fail_on_duplicate_rate = (
            fail_on_duplicate_rate
        )

        self.fail_on_constant_columns = (
            fail_on_constant_columns
        )

        self.detect_empty_strings = (
            detect_empty_strings
        )

        self.fail_on_infinite = (
            fail_on_infinite
        )

        self.detect_constant_columns = (
            detect_constant_columns
        )

        self.minimum_rows = minimum_rows
        self.maximum_rows = maximum_rows

        self.minimum_columns = minimum_columns
        self.maximum_columns = maximum_columns

        self.check_numeric_extremes = (
            check_numeric_extremes
        )

        self.numeric_zscore_threshold = (
            numeric_zscore_threshold
        )

        self.max_string_length = (
            max_string_length
        )

        self.sample_size = sample_size

        self._validate_configuration()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _validate_configuration(self) -> None:

        if not 0.0 <= self.max_allowed_null_rate <= 1.0:
            raise ValueError(
                "max_allowed_null_rate must be "
                "between 0 and 1."
            )

        if not 0.0 <= self.max_allowed_duplicate_rate <= 1.0:
            raise ValueError(
                "max_allowed_duplicate_rate must be "
                "between 0 and 1."
            )

        if self.minimum_rows is not None:
            if self.minimum_rows < 0:
                raise ValueError(
                    "minimum_rows cannot be negative."
                )

        if self.maximum_rows is not None:
            if self.maximum_rows < 0:
                raise ValueError(
                    "maximum_rows cannot be negative."
                )

        if (
            self.minimum_rows is not None
            and self.maximum_rows is not None
            and self.minimum_rows
            > self.maximum_rows
        ):
            raise ValueError(
                "minimum_rows cannot exceed maximum_rows."
            )

        if self.minimum_columns is not None:
            if self.minimum_columns < 0:
                raise ValueError(
                    "minimum_columns cannot be negative."
                )

        if self.maximum_columns is not None:
            if self.maximum_columns < 0:
                raise ValueError(
                    "maximum_columns cannot be negative."
                )

        if self.numeric_zscore_threshold <= 0:
            raise ValueError(
                "numeric_zscore_threshold must be > 0."
            )

        if self.max_string_length < 0:
            raise ValueError(
                "max_string_length cannot be negative."
            )

        if self.sample_size <= 0:
            raise ValueError(
                "sample_size must be greater than zero."
            )

    # ------------------------------------------------------------------
    # Main validation
    # ------------------------------------------------------------------

    def validate(
        self,
        df: pd.DataFrame,
        result: Optional[ValidationResult] = None,
    ) -> ValidationResult:

        if result is None:
            result = ValidationResult(
                is_valid=True
            )

        if not isinstance(df, pd.DataFrame):
            result.add_issue(
                "INVALID_DATAFRAME",
                "Dataset is not a pandas DataFrame.",
                ValidationSeverity.CRITICAL,
                category="structure",
            )
            return result

        result.row_count = len(df)
        result.column_count = len(df.columns)

        self._validate_shape(
            df,
            result,
        )

        if len(df) == 0:
            if not self.allow_empty_dataframe:
                result.add_issue(
                    "EMPTY_DATASET",
                    "Dataset contains zero rows.",
                    ValidationSeverity.CRITICAL,
                    category="quality",
                )
            return result

        self._validate_null_rates(
            df,
            result,
        )

        self._validate_numeric_values(
            df,
            result,
        )

        self._validate_empty_strings(
            df,
            result,
        )

        self._validate_duplicates(
            df,
            result,
        )

        self._validate_constant_columns(
            df,
            result,
        )

        self._validate_string_extremes(
            df,
            result,
        )

        return result

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    def _validate_shape(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:

        rows = len(df)
        columns = len(df.columns)

        if (
            self.minimum_rows is not None
            and rows < self.minimum_rows
        ):
            result.add_issue(
                "MINIMUM_ROWS_NOT_MET",
                (
                    f"Dataset contains {rows} rows, "
                    f"below minimum {self.minimum_rows}."
                ),
                ValidationSeverity.ERROR,
                category="structure",
            )

        if (
            self.maximum_rows is not None
            and rows > self.maximum_rows
        ):
            result.add_issue(
                "MAXIMUM_ROWS_EXCEEDED",
                (
                    f"Dataset contains {rows} rows, "
                    f"above maximum {self.maximum_rows}."
                ),
                ValidationSeverity.ERROR,
                category="structure",
            )

        if (
            self.minimum_columns is not None
            and columns < self.minimum_columns
        ):
            result.add_issue(
                "MINIMUM_COLUMNS_NOT_MET",
                (
                    f"Dataset contains {columns} columns, "
                    f"below minimum "
                    f"{self.minimum_columns}."
                ),
                ValidationSeverity.ERROR,
                category="structure",
            )

        if (
            self.maximum_columns is not None
            and columns > self.maximum_columns
        ):
            result.add_issue(
                "MAXIMUM_COLUMNS_EXCEEDED",
                (
                    f"Dataset contains {columns} columns, "
                    f"above maximum "
                    f"{self.maximum_columns}."
                ),
                ValidationSeverity.ERROR,
                category="structure",
            )

    # ------------------------------------------------------------------
    # Null rates
    # ------------------------------------------------------------------

    def _validate_null_rates(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:

        total_rows = len(df)

        null_counts = df.isna().sum()

        for column in df.columns:

            null_count = int(
                null_counts[column]
            )

            rate = (
                null_count / total_rows
                if total_rows
                else 0.0
            )

            rate = round(rate, 6)

            result.null_rates[
                str(column)
            ] = rate

            if rate <= self.max_allowed_null_rate:
                continue

            severity = (
                ValidationSeverity.ERROR
                if self.fail_on_null_rate
                else ValidationSeverity.WARNING
            )

            result.add_issue(
                "NULL_RATE_EXCEEDED",
                (
                    f"Column '{column}' null rate "
                    f"({rate:.2%}) exceeds "
                    f"maximum threshold "
                    f"({self.max_allowed_null_rate:.2%})."
                ),
                severity,
                column=str(column),
                category="quality",
                count=null_count,
                observed=rate,
                expected=self.max_allowed_null_rate,
            )

    # ------------------------------------------------------------------
    # Numeric validation
    # ------------------------------------------------------------------

    def _validate_numeric_values(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:

        numeric_columns = df.select_dtypes(
            include=[np.number]
        ).columns

        for column in numeric_columns:

            series = df[column]

            # NaN
            nan_count = int(
                series.isna().sum()
            )

            if nan_count:
                result.nan_counts[
                    str(column)
                ] = nan_count

            # Infinity
            try:
                inf_count = int(
                    np.isinf(
                        series.to_numpy(
                            dtype=float,
                            na_value=np.nan,
                        )
                    ).sum()
                )
            except (TypeError, ValueError):
                values = pd.to_numeric(
                    series,
                    errors="coerce",
                )

                inf_count = int(
                    np.isinf(
                        values.to_numpy(
                            dtype=float
                        )
                    ).sum()
                )

            if inf_count:
                result.infinite_counts[
                    str(column)
                ] = inf_count

                severity = (
                    ValidationSeverity.ERROR
                    if self.fail_on_infinite
                    else ValidationSeverity.WARNING
                )

                result.add_issue(
                    "INFINITE_VALUES",
                    (
                        f"Column '{column}' contains "
                        f"{inf_count} infinite "
                        "values."
                    ),
                    severity,
                    column=str(column),
                    category="quality",
                    count=inf_count,
                )

            if (
                self.check_numeric_extremes
                and inf_count == 0
            ):
                self._detect_numeric_extremes(
                    series,
                    str(column),
                    result,
                )

    # ------------------------------------------------------------------
    # Statistical extremes
    # ------------------------------------------------------------------

    def _detect_numeric_extremes(
        self,
        series: pd.Series,
        column: str,
        result: ValidationResult,
    ) -> None:

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        if len(numeric) < 10:
            return

        # Avoid overflow and pathological calculations.
        sample = (
            numeric.sample(
                n=min(
                    len(numeric),
                    self.sample_size,
                ),
                random_state=42,
            )
            if len(numeric) > self.sample_size
            else numeric
        )

        values = sample.to_numpy(
            dtype=float
        )

        if not np.all(
            np.isfinite(values)
        ):
            return

        mean = float(
            np.mean(values)
        )

        std = float(
            np.std(values)
        )

        if std == 0.0 or not math.isfinite(std):
            return

        max_abs_z = float(
            np.max(
                np.abs(
                    (values - mean) / std
                )
            )
        )

        if (
            max_abs_z
            > self.numeric_zscore_threshold
        ):
            result.suspicious_columns.append(
                column
            )

            result.add_issue(
                "NUMERIC_EXTREME_VALUES",
                (
                    f"Column '{column}' contains "
                    "statistically extreme values "
                    f"(maximum absolute z-score "
                    f"≈ {max_abs_z:.2f})."
                ),
                ValidationSeverity.WARNING,
                column=column,
                category="statistics",
                observed=round(
                    max_abs_z,
                    4,
                ),
                expected=(
                    self.numeric_zscore_threshold
                ),
            )

    # ------------------------------------------------------------------
    # Empty strings
    # ------------------------------------------------------------------

    def _validate_empty_strings(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:

        if not self.detect_empty_strings:
            return

        for column in df.columns:

            series = df[column]

            if not (
                pd.api.types.is_object_dtype(
                    series
                )
                or pd.api.types.is_string_dtype(
                    series
                )
            ):
                continue

            try:
                empty_mask = (
                    series.astype("string")
                    .str.strip()
                    .eq("")
                )

                count = int(
                    empty_mask.fillna(False)
                    .sum()
                )
            except Exception:
                continue

            if count:
                result.empty_string_counts[
                    str(column)
                ] = count

                result.add_issue(
                    "EMPTY_STRINGS",
                    (
                        f"Column '{column}' contains "
                        f"{count} empty or whitespace-only "
                        "string values."
                    ),
                    ValidationSeverity.WARNING,
                    column=str(column),
                    category="quality",
                    count=count,
                )

    # ------------------------------------------------------------------
    # Duplicate rows
    # ------------------------------------------------------------------

    def _validate_duplicates(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:

        try:
            duplicate_mask = df.duplicated(
                keep="first"
            )

            duplicate_count = int(
                duplicate_mask.sum()
            )

        except (TypeError, ValueError):
            # Some object columns can contain unhashable Python objects.
            # Avoid crashing the complete validation engine.
            result.add_issue(
                "DUPLICATE_CHECK_FAILED",
                (
                    "Duplicate-row detection could not "
                    "be completed because one or more "
                    "columns contain unsupported "
                    "unhashable values."
                ),
                ValidationSeverity.WARNING,
                category="quality",
            )
            return

        result.duplicate_row_count = (
            duplicate_count
        )

        result.duplicate_row_rate = round(
            duplicate_count / len(df),
            6,
        )

        if (
            result.duplicate_row_rate
            <= self.max_allowed_duplicate_rate
        ):
            return

        severity = (
            ValidationSeverity.ERROR
            if self.fail_on_duplicate_rate
            else ValidationSeverity.WARNING
        )

        result.add_issue(
            "DUPLICATE_RATE_EXCEEDED",
            (
                f"Dataset duplicate-row rate "
                f"({result.duplicate_row_rate:.2%}) "
                f"exceeds configured threshold "
                f"({self.max_allowed_duplicate_rate:.2%})."
            ),
            severity,
            category="quality",
            count=duplicate_count,
            observed=result.duplicate_row_rate,
            expected=self.max_allowed_duplicate_rate,
        )

    # ------------------------------------------------------------------
    # Constant columns
    # ------------------------------------------------------------------

    def _validate_constant_columns(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:

        if not self.detect_constant_columns:
            return

        for column in df.columns:

            try:
                unique_count = int(
                    df[column].nunique(
                        dropna=False
                    )
                )
            except (TypeError, ValueError):
                continue

            if unique_count != 1:
                continue

            result.constant_columns.append(
                str(column)
            )

            severity = (
                ValidationSeverity.ERROR
                if self.fail_on_constant_columns
                else ValidationSeverity.WARNING
            )

            result.add_issue(
                "CONSTANT_COLUMN",
                (
                    f"Column '{column}' contains "
                    "only one unique value."
                ),
                severity,
                column=str(column),
                category="quality",
            )

    # ------------------------------------------------------------------
    # String extremes
    # ------------------------------------------------------------------

    def _validate_string_extremes(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:

        for column in df.columns:

            series = df[column]

            if not (
                pd.api.types.is_object_dtype(
                    series
                )
                or pd.api.types.is_string_dtype(
                    series
                )
            ):
                continue

            try:
                lengths = (
                    series.dropna()
                    .astype(str)
                    .str.len()
                )
            except Exception:
                continue

            if lengths.empty:
                continue

            max_length = int(
                lengths.max()
            )

            if (
                max_length
                > self.max_string_length
            ):
                result.add_issue(
                    "PATHOLOGICAL_STRING_LENGTH",
                    (
                        f"Column '{column}' contains "
                        "string values exceeding the "
                        f"maximum safety length of "
                        f"{self.max_string_length}."
                    ),
                    ValidationSeverity.ERROR,
                    column=str(column),
                    category="security",
                    observed=max_length,
                    expected=self.max_string_length,
                )


# ============================================================================
# DATASET FINGERPRINTING
# ============================================================================


def compute_dataset_fingerprint(
    df: pd.DataFrame,
    *,
    include_values: bool = True,
    max_rows: int = DEFAULT_SAMPLE_SIZE,
) -> str:
    """
    Generate a deterministic fingerprint for a DataFrame.

    The fingerprint can be used for:
        - validation provenance
        - dataset lineage
        - experiment tracking
        - regression detection
        - model-training provenance

    Parameters
    ----------
    df:
        DataFrame to fingerprint.

    include_values:
        If True, include sampled dataset values.

    max_rows:
        Maximum rows incorporated into the fingerprint.

    Notes
    -----
    This function is intentionally deterministic and does not return the raw
    dataset. It returns only a SHA-256-style object fingerprint.

    For very large datasets, a bounded sample is used to prevent validation
    itself from becoming an excessive memory consumer.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    metadata: Dict[str, Any] = {
        "shape": [
            int(df.shape[0]),
            int(df.shape[1]),
        ],
        "columns": [
            str(column)
            for column in df.columns
        ],
        "dtypes": {
            str(column): str(dtype)
            for column, dtype in df.dtypes.items()
        },
    }

    if include_values and not df.empty:

        if len(df) > max_rows:
            sample = df.sample(
                n=max_rows,
                random_state=42,
            )
        else:
            sample = df

        # Convert to deterministic records while avoiding non-serializable
        # pandas objects.
        records = sample.astype(
            object
        ).where(
            pd.notna(sample),
            None,
        ).to_dict(
            orient="records"
        )

        metadata["sample"] = records

    return compute_object_hash(
        metadata
    )


# ============================================================================
# DATASET INTEGRITY CONVENIENCE FUNCTION
# ============================================================================


def validate_dataset_integrity(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    column_types: Optional[
        Mapping[str, str]
    ] = None,
    max_null_rate: float = DEFAULT_MAX_NULL_RATE,
    *,
    value_bounds: Optional[
        Mapping[
            str,
            Tuple[
                Optional[float],
                Optional[float],
            ],
        ]
    ] = None,
    allowed_columns: Optional[
        Sequence[str]
    ] = None,
    allow_extra_columns: bool = True,
    nullable_columns: Optional[
        Mapping[str, bool]
    ] = None,
    allowed_values: Optional[
        Mapping[str, Iterable[Any]]
    ] = None,
    max_string_lengths: Optional[
        Mapping[str, int]
    ] = None,
    min_string_lengths: Optional[
        Mapping[str, int]
    ] = None,
    max_cardinality: Optional[
        Mapping[str, int]
    ] = None,
    max_cardinality_rate: Optional[
        Mapping[str, float]
    ] = None,
    max_duplicate_rate: float = DEFAULT_MAX_DUPLICATE_RATE,
    allow_empty_dataframe: bool = False,
    strict_schema: bool = True,
    fail_on_null_rate: bool = False,
    fail_on_duplicate_rate: bool = False,
    fail_on_constant_columns: bool = False,
    detect_empty_strings: bool = True,
    compute_fingerprint: bool = True,
    fingerprint_include_values: bool = False,
) -> ValidationResult:
    """
    Execute the complete schema + data-quality integrity pipeline.

    This is the recommended high-level API for training and evaluation jobs.

    The function is intentionally side-effect free with respect to the
    DataFrame. It never modifies or cleans the input.
    """

    started = time.perf_counter()

    validation_result = ValidationResult(
        is_valid=True,
        started_at=datetime.now(
            timezone.utc
        ).isoformat(),
    )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    if not isinstance(df, pd.DataFrame):

        validation_result.add_issue(
            "INVALID_DATAFRAME",
            "Input dataset is not a pandas DataFrame.",
            ValidationSeverity.CRITICAL,
            category="structure",
        )

        validation_result.completed_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        validation_result.duration_seconds = (
            time.perf_counter() - started
        )

        logger.error(
            "Dataset validation failed: "
            "input is not a pandas DataFrame."
        )

        return validation_result

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    schema_validator = SchemaValidator(
        required_columns=required_columns,
        column_types=column_types,
        value_bounds=value_bounds,
        allowed_columns=allowed_columns,
        allow_extra_columns=allow_extra_columns,
        nullable_columns=nullable_columns,
        allowed_values=allowed_values,
        max_string_lengths=max_string_lengths,
        min_string_lengths=min_string_lengths,
        max_cardinality=max_cardinality,
        max_cardinality_rate=max_cardinality_rate,
        strict=strict_schema,
    )

    schema_validator.validate(
        df,
        validation_result,
    )

    # ------------------------------------------------------------------
    # Quality validation
    # ------------------------------------------------------------------

    quality_validator = DataQualityValidator(
        max_allowed_null_rate=max_null_rate,
        allow_empty_dataframe=allow_empty_dataframe,
        max_allowed_duplicate_rate=max_duplicate_rate,
        fail_on_null_rate=fail_on_null_rate,
        fail_on_duplicate_rate=fail_on_duplicate_rate,
        fail_on_constant_columns=fail_on_constant_columns,
        detect_empty_strings=detect_empty_strings,
    )

    quality_validator.validate(
        df,
        validation_result,
    )

    # ------------------------------------------------------------------
    # Dataset fingerprint
    # ------------------------------------------------------------------

    if compute_fingerprint:

        try:
            validation_result.dataset_fingerprint = (
                compute_dataset_fingerprint(
                    df,
                    include_values=fingerprint_include_values,
                )
            )
        except Exception as exc:
            # Fingerprinting failure should not hide the actual data
            # validation result.
            validation_result.add_issue(
                "FINGERPRINT_FAILED",
                (
                    "Dataset fingerprint generation "
                    "failed; validation results remain "
                    "available."
                ),
                ValidationSeverity.WARNING,
                category="provenance",
                metadata={
                    "exception_type": type(
                        exc
                    ).__name__
                },
            )

    # ------------------------------------------------------------------
    # Finalize metadata
    # ------------------------------------------------------------------

    validation_result.metadata.update(
        {
            "required_column_count": len(
                required_columns
            ),
            "configured_type_rules": len(
                column_types or {}
            ),
            "configured_bound_rules": len(
                value_bounds or {}
            ),
            "strict_schema": strict_schema,
        }
    )

    validation_result.completed_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    validation_result.duration_seconds = round(
        time.perf_counter() - started,
        6,
    )

    # ------------------------------------------------------------------
    # Security-conscious logging
    # ------------------------------------------------------------------

    if validation_result.is_valid:

        if validation_result.warning_count:

            logger.warning(
                (
                    "Dataset integrity validation "
                    "PASSED WITH WARNINGS: "
                    "validation_id=%s rows=%d columns=%d "
                    "warnings=%d"
                ),
                validation_result.validation_id,
                validation_result.row_count,
                validation_result.column_count,
                validation_result.warning_count,
            )

        else:

            logger.info(
                (
                    "Dataset integrity validation "
                    "PASSED: validation_id=%s "
                    "rows=%d columns=%d"
                ),
                validation_result.validation_id,
                validation_result.row_count,
                validation_result.column_count,
            )

    else:

        logger.error(
            (
                "Dataset integrity validation "
                "FAILED: validation_id=%s "
                "errors=%d warnings=%d"
            ),
            validation_result.validation_id,
            validation_result.error_count,
            validation_result.warning_count,
        )

    return validation_result


# ============================================================================
# SPECIALIZED VALIDATION HELPERS
# ============================================================================


def validate_binary_target(
    df: pd.DataFrame,
    target_column: str,
    *,
    allowed_values: Optional[
        Iterable[Any]
    ] = None,
    require_both_classes: bool = True,
) -> ValidationResult:
    """
    Validate an ML classification target.

    This is especially important for phishing detection because a training
    dataset containing only benign or only malicious examples can produce a
    misleading training/evaluation pipeline.
    """

    result = ValidationResult(
        is_valid=True,
        started_at=datetime.now(
            timezone.utc
        ).isoformat(),
    )

    if not isinstance(df, pd.DataFrame):
        result.add_issue(
            "INVALID_DATAFRAME",
            "Input dataset is not a pandas DataFrame.",
            ValidationSeverity.CRITICAL,
            category="target",
        )
        return result

    result.row_count = len(df)
    result.column_count = len(df.columns)

    if target_column not in df.columns:

        result.add_issue(
            "TARGET_COLUMN_MISSING",
            (
                f"Target column '{target_column}' "
                "is missing."
            ),
            ValidationSeverity.CRITICAL,
            column=target_column,
            category="target",
        )

        return result

    target = df[target_column]

    null_count = int(
        target.isna().sum()
    )

    if null_count:

        result.add_issue(
            "TARGET_NULLS",
            (
                f"Target column '{target_column}' "
                f"contains {null_count} null values."
            ),
            ValidationSeverity.ERROR,
            column=target_column,
            category="target",
            count=null_count,
        )

    values = target.dropna()

    if values.empty:

        result.add_issue(
            "TARGET_EMPTY",
            (
                f"Target column '{target_column}' "
                "contains no usable values."
            ),
            ValidationSeverity.CRITICAL,
            column=target_column,
            category="target",
        )

        return result

    unique_values = set(
        values.tolist()
    )

    if allowed_values is None:

        # Default binary representation.
        normalized = {
            str(value).strip().lower()
            for value in unique_values
        }

        accepted = {
            "0",
            "1",
            "true",
            "false",
            "benign",
            "malicious",
            "phishing",
            "legitimate",
        }

        invalid = normalized - accepted

        if invalid:

            result.add_issue(
                "INVALID_BINARY_TARGET",
                (
                    f"Target column '{target_column}' "
                    "contains values outside the "
                    "recognized binary target domain."
                ),
                ValidationSeverity.ERROR,
                column=target_column,
                category="target",
            )

    else:

        allowed = set(
            allowed_values
        )

        invalid_count = int(
            (~values.isin(allowed)).sum()
        )

        if invalid_count:

            result.add_issue(
                "INVALID_TARGET_VALUES",
                (
                    f"Target column '{target_column}' "
                    f"contains {invalid_count} values "
                    "outside the allowed target domain."
                ),
                ValidationSeverity.ERROR,
                column=target_column,
                category="target",
                count=invalid_count,
            )

    if (
        require_both_classes
        and len(unique_values) < 2
    ):

        result.add_issue(
            "SINGLE_CLASS_TARGET",
            (
                f"Target column '{target_column}' "
                "contains only one observed class."
            ),
            ValidationSeverity.ERROR,
            column=target_column,
            category="target",
            observed=len(unique_values),
            expected=2,
        )

    result.completed_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    return result


# ============================================================================
# EMAIL-SPECIFIC VALIDATION
# ============================================================================


def validate_email_dataset(
    df: pd.DataFrame,
    *,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    max_text_length: int = 5_000_000,
) -> ValidationResult:
    """
    Security-oriented validation for email datasets.

    This does not inspect or log actual email content.
    """

    required = []

    if text_column:
        required.append(text_column)

    if label_column:
        required.append(label_column)

    result = validate_dataset_integrity(
        df,
        required_columns=required,
        column_types=(
            {
                text_column: "string"
            }
            if text_column
            else None
        ),
        max_string_lengths=(
            {
                text_column: max_text_length
            }
            if text_column
            else None
        ),
        compute_fingerprint=True,
        fingerprint_include_values=False,
    )

    if label_column:

        target_result = validate_binary_target(
            df,
            label_column,
        )

        result.errors.extend(
            target_result.errors
        )

        result.warnings.extend(
            target_result.warnings
        )

        result.issues.extend(
            target_result.issues
        )

        if not target_result.is_valid:
            result.is_valid = False

    return result


# ============================================================================
# URL-SPECIFIC VALIDATION
# ============================================================================


def validate_url_dataset(
    df: pd.DataFrame,
    *,
    url_column: str = "url",
    label_column: Optional[str] = None,
    max_url_length: int = 16_384,
) -> ValidationResult:
    """
    Security-oriented validation for URL datasets.

    The function validates structure and length only. It deliberately does not
    normalize, resolve, fetch, or contact URLs. Network interaction belongs to
    a separate controlled subsystem.
    """

    required = [url_column]

    if label_column:
        required.append(label_column)

    result = validate_dataset_integrity(
        df,
        required_columns=required,
        column_types={
            url_column: "url",
        },
        max_string_lengths={
            url_column: max_url_length,
        },
        compute_fingerprint=True,
        fingerprint_include_values=False,
    )

    if label_column:

        target_result = validate_binary_target(
            df,
            label_column,
        )

        result.errors.extend(
            target_result.errors
        )

        result.warnings.extend(
            target_result.warnings
        )

        result.issues.extend(
            target_result.issues
        )

        if not target_result.is_valid:
            result.is_valid = False

    return result


# ============================================================================
# ATTACHMENT DATASET VALIDATION
# ============================================================================


def validate_attachment_dataset(
    df: pd.DataFrame,
    *,
    file_name_column: Optional[str] = None,
    size_column: Optional[str] = None,
    label_column: Optional[str] = None,
    maximum_file_size: Optional[int] = None,
) -> ValidationResult:
    """
    Validate attachment metadata datasets.

    Important
    ---------
    This validator operates on metadata. It does not execute, open, unpack,
    parse, or otherwise process attachment payloads.
    """

    required: List[str] = []

    if file_name_column:
        required.append(file_name_column)

    if size_column:
        required.append(size_column)

    if label_column:
        required.append(label_column)

    bounds = None

    if (
        size_column
        and maximum_file_size is not None
    ):
        bounds = {
            size_column: (
                0,
                maximum_file_size,
            )
        }

    result = validate_dataset_integrity(
        df,
        required_columns=required,
        column_types=(
            {
                size_column: "numeric"
            }
            if size_column
            else None
        ),
        value_bounds=bounds,
        compute_fingerprint=True,
        fingerprint_include_values=False,
    )

    if label_column:

        target_result = validate_binary_target(
            df,
            label_column,
        )

        result.errors.extend(
            target_result.errors
        )

        result.warnings.extend(
            target_result.warnings
        )

        result.issues.extend(
            target_result.issues
        )

        if not target_result.is_valid:
            result.is_valid = False

    return result


# ============================================================================
# EXPORT CONTRACT
# ============================================================================


__all__ = [
    # Version
    "VALIDATION_ENGINE_VERSION",

    # Enumerations
    "ValidationSeverity",
    "ValidationStatus",

    # Result structures
    "ValidationIssue",
    "ValidationResult",

    # Core validators
    "SchemaValidator",
    "DataQualityValidator",

    # Fingerprinting
    "compute_dataset_fingerprint",

    # Unified validation
    "validate_dataset_integrity",

    # Domain-specific validation
    "validate_binary_target",
    "validate_email_dataset",
    "validate_url_dataset",
    "validate_attachment_dataset",

    # Constants
    "DEFAULT_MAX_NULL_RATE",
    "DEFAULT_MAX_DUPLICATE_RATE",
    "SUPPORTED_COLUMN_TYPES",
]
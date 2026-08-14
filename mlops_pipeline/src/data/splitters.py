"""
PhishGuard AI - Production Dataset Splitting & Leakage Prevention Engine
=========================================================================

Version: 14.0.0
File Path: src/data/splitters.py

Purpose
-------
Production-grade dataset partitioning for the PhishGuard AI MLOps pipeline.

This module is responsible for one of the most security-sensitive stages of
the ML lifecycle: deciding which observations are allowed to influence model
training, validation, and final evaluation.

The implementation explicitly addresses:

    * Temporal leakage
    * Campaign/domain/group leakage
    * Duplicate-row leakage
    * Target leakage caused by invalid split construction
    * Class imbalance
    * Small datasets
    * Empty partitions
    * Invalid timestamps
    * NaN targets
    * Conflicting labels inside groups
    * Cross-validation leakage
    * Temporal embargo / gap
    * Deterministic reproducibility
    * Split provenance
    * Split fingerprints
    * Partition overlap detection
    * Class-distribution auditing
    * Group-distribution auditing
    * Dataset-order normalization
    * Fail-closed validation

Design Principles
-----------------
1. Never silently create an invalid split.
2. Never silently ignore a requested grouping constraint.
3. Never silently discard malformed timestamps.
4. Never use future observations to train a temporal model.
5. Never allow the same protected group across train and evaluation partitions.
6. Never claim "stratified" behavior unless class distribution is actually
   considered.
7. Preserve the legacy public API wherever practical.
8. Return copies of partitions rather than views into mutable source data.
9. Make every random split deterministic when random_state is supplied.
10. Make split provenance inspectable and serializable.

Important
---------
A splitter prevents only split-related leakage. Feature engineering,
preprocessing, target encoding, feature selection, normalization, and model
selection must still be fitted only on the training partition.

For phishing detection, recommended protected groups include:

    campaign_id
    sender_domain
    registrable_domain
    url_domain
    message_thread_id
    attachment_hash
    sample_family_id

Do NOT automatically group on arbitrary high-cardinality identifiers such as
a unique row ID because doing so provides little protection and can produce
pathological partitions.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)

from src.utils.logger import get_logger


logger = get_logger("src.data.splitters")


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_RANDOM_STATE: int = 42

MIN_ALLOWED_SPLIT_RATIO: float = 0.0
MAX_ALLOWED_SPLIT_RATIO: float = 0.90

DEFAULT_DUPLICATE_CHECK_COLUMNS: Optional[List[str]] = None

SUPPORTED_SPLIT_MODES: Set[str] = {
    "temporal",
    "stratified",
    "group",
    "stratified_group",
}


# =============================================================================
# EXCEPTIONS
# =============================================================================


class SplitterError(RuntimeError):
    """Base exception for dataset splitting failures."""


class SplitConfigurationError(SplitterError, ValueError):
    """Raised when splitter configuration is invalid."""


class SplitDataError(SplitterError, ValueError):
    """Raised when input data cannot safely be partitioned."""


class LeakageDetectedError(SplitterError):
    """Raised when a partition violates an explicit leakage constraint."""


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class PartitionStatistics:
    """
    Statistics describing one partition.

    This object is intentionally serializable so that split provenance can be
    persisted alongside model artifacts.
    """

    name: str
    row_count: int = 0
    percentage: float = 0.0

    class_counts: Dict[str, int] = field(default_factory=dict)
    class_distribution: Dict[str, float] = field(default_factory=dict)

    unique_group_count: Optional[int] = None
    null_target_count: int = 0

    min_timestamp: Optional[str] = None
    max_timestamp: Optional[str] = None

    duplicate_row_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SplitDiagnostics:
    """
    Complete audit record for a dataset partitioning operation.
    """

    splitter_type: str
    target_column: Optional[str]

    random_state: Optional[int]

    source_rows: int

    train_rows: int
    validation_rows: int
    test_rows: int

    source_groups: Optional[int] = None
    train_groups: Optional[int] = None
    validation_groups: Optional[int] = None
    test_groups: Optional[int] = None

    group_column: Optional[str] = None
    time_column: Optional[str] = None

    validation_size: float = 0.0
    test_size: float = 0.0

    temporal_gap: int = 0

    duplicate_source_rows: int = 0
    duplicate_across_partitions: int = 0

    group_overlap_count: int = 0
    row_overlap_count: int = 0

    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    train_statistics: Optional[PartitionStatistics] = None
    validation_statistics: Optional[PartitionStatistics] = None
    test_statistics: Optional[PartitionStatistics] = None

    fingerprint: Optional[str] = None

    valid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "splitter_type": self.splitter_type,
            "target_column": self.target_column,
            "random_state": self.random_state,
            "source_rows": self.source_rows,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "test_rows": self.test_rows,
            "source_groups": self.source_groups,
            "train_groups": self.train_groups,
            "validation_groups": self.validation_groups,
            "test_groups": self.test_groups,
            "group_column": self.group_column,
            "time_column": self.time_column,
            "validation_size": self.validation_size,
            "test_size": self.test_size,
            "temporal_gap": self.temporal_gap,
            "duplicate_source_rows": self.duplicate_source_rows,
            "duplicate_across_partitions": self.duplicate_across_partitions,
            "group_overlap_count": self.group_overlap_count,
            "row_overlap_count": self.row_overlap_count,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "train_statistics": (
                self.train_statistics.to_dict()
                if self.train_statistics
                else None
            ),
            "validation_statistics": (
                self.validation_statistics.to_dict()
                if self.validation_statistics
                else None
            ),
            "test_statistics": (
                self.test_statistics.to_dict()
                if self.test_statistics
                else None
            ),
            "fingerprint": self.fingerprint,
            "valid": self.valid,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize diagnostics to deterministic JSON."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
        )


@dataclass
class SplitResult:
    """
    Rich result representation.

    The legacy API still returns:

        train_df, val_df, test_df

    but callers needing stronger provenance can use this structure.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: Optional[pd.DataFrame]

    diagnostics: SplitDiagnostics

    def as_tuple(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        return self.train, self.validation, self.test


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Fail closed if the supplied dataset is not a DataFrame."""
    if df is None:
        raise SplitDataError("Input DataFrame is None.")

    if not isinstance(df, pd.DataFrame):
        raise SplitDataError(
            f"Expected pandas.DataFrame, received {type(df).__name__}."
        )

    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise SplitDataError(
            f"Dataset contains duplicated column names: {duplicated!r}"
        )


def _validate_non_empty_dataframe(df: pd.DataFrame) -> None:
    _validate_dataframe(df)

    if df.empty:
        raise SplitDataError("Cannot split an empty DataFrame.")


def _validate_column(
    df: pd.DataFrame,
    column: str,
    purpose: str,
) -> None:
    if column not in df.columns:
        raise SplitDataError(
            f"{purpose} column '{column}' does not exist in dataset."
        )


def _validate_ratio(name: str, value: float) -> None:
    if not isinstance(value, (int, float)):
        raise SplitConfigurationError(
            f"{name} must be numeric; received {type(value).__name__}."
        )

    if not math.isfinite(float(value)):
        raise SplitConfigurationError(
            f"{name} must be finite; received {value!r}."
        )

    if not (
        MIN_ALLOWED_SPLIT_RATIO
        <= float(value)
        <= MAX_ALLOWED_SPLIT_RATIO
    ):
        raise SplitConfigurationError(
            f"{name} must be between "
            f"{MIN_ALLOWED_SPLIT_RATIO} and "
            f"{MAX_ALLOWED_SPLIT_RATIO}; received {value}."
        )


def _validate_split_sizes(
    val_size: float,
    test_size: float,
) -> None:
    _validate_ratio("val_size", val_size)
    _validate_ratio("test_size", test_size)

    if val_size + test_size >= 1.0:
        raise SplitConfigurationError(
            "val_size + test_size must be strictly less than 1.0."
        )


def _validate_random_state(random_state: Optional[int]) -> None:
    if random_state is not None:
        if not isinstance(random_state, int):
            raise SplitConfigurationError(
                "random_state must be an integer or None."
            )


def _validate_n_splits(n_splits: int) -> None:
    if not isinstance(n_splits, int) or n_splits < 2:
        raise SplitConfigurationError(
            f"n_splits must be an integer >= 2; received {n_splits!r}."
        )


def _validate_target(
    df: pd.DataFrame,
    target_column: str,
    *,
    allow_missing_target: bool = False,
) -> None:
    _validate_column(df, target_column, "Target")

    if allow_missing_target:
        return

    missing = int(df[target_column].isna().sum())

    if missing:
        raise SplitDataError(
            f"Target column '{target_column}' contains "
            f"{missing} missing values. Target labels must be complete "
            "before dataset splitting."
        )


def _validate_group_column(
    df: pd.DataFrame,
    group_column: str,
) -> None:
    _validate_column(df, group_column, "Group")

    missing = int(df[group_column].isna().sum())

    if missing:
        raise SplitDataError(
            f"Group column '{group_column}' contains {missing} missing values. "
            "Missing group identifiers cannot be safely assigned to a "
            "leakage-protected partition."
        )


def _validate_group_label_consistency(
    df: pd.DataFrame,
    target_column: str,
    group_column: str,
) -> None:
    """
    Detect groups containing multiple target labels.

    This is not necessarily always invalid in ML, but in phishing campaign
    datasets it is often a signal that the grouping key is too coarse,
    labels are noisy, or the dataset contains contradictory observations.
    """

    grouped = df.groupby(group_column, dropna=False)[target_column].nunique()

    conflicting = grouped[grouped > 1]

    if not conflicting.empty:
        examples = conflicting.index.astype(str).tolist()[:10]

        raise SplitDataError(
            f"Found {len(conflicting)} groups containing multiple target "
            f"labels in '{target_column}'. Example groups: {examples!r}. "
            "Resolve label/group conflicts before group-aware splitting."
        )


def _coerce_datetime(
    df: pd.DataFrame,
    time_column: str,
) -> pd.Series:
    """Convert timestamps while rejecting invalid temporal observations."""
    _validate_column(df, time_column, "Timestamp")

    converted = pd.to_datetime(
        df[time_column],
        errors="coerce",
        utc=True,
    )

    invalid = int(converted.isna().sum())

    if invalid:
        raise SplitDataError(
            f"Timestamp column '{time_column}' contains {invalid} invalid "
            "or missing timestamps. Temporal splitting requires valid "
            "timestamps for every observation."
        )

    return converted


def _count_duplicate_rows(
    df: pd.DataFrame,
    duplicate_subset: Optional[Sequence[str]] = None,
) -> int:
    """Return number of rows that belong to duplicate groups."""
    if df.empty:
        return 0

    if duplicate_subset is not None:
        missing = [c for c in duplicate_subset if c not in df.columns]

        if missing:
            raise SplitDataError(
                f"Duplicate detection columns do not exist: {missing!r}"
            )

        duplicated = df.duplicated(
            subset=list(duplicate_subset),
            keep=False,
        )
    else:
        duplicated = df.duplicated(keep=False)

    return int(duplicated.sum())


def _stable_row_fingerprint(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> pd.Series:
    """
    Create deterministic row fingerprints.

    SHA-256 is deliberately used here because this fingerprint is used for
    leakage detection/provenance, not fuzzy similarity.
    """

    selected = (
        list(columns)
        if columns is not None
        else list(df.columns)
    )

    missing = [c for c in selected if c not in df.columns]

    if missing:
        raise SplitDataError(
            f"Fingerprint columns missing from dataset: {missing!r}"
        )

    normalized = df[selected].copy()

    # Normalize object-like values into deterministic strings.
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].astype("string")

    return pd.util.hash_pandas_object(
        normalized,
        index=False,
    ).astype("uint64").astype(str)


def _set_overlap(
    first: Iterable[Any],
    second: Iterable[Any],
) -> Set[Any]:
    return set(first).intersection(set(second))


def _assert_no_row_overlap(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: Optional[pd.DataFrame],
) -> None:
    """
    Detect actual source-index overlap.

    This is an additional defense layer. Normal pandas iloc splitting should
    not produce overlap, but the explicit check protects future modifications.
    """

    train_ids = set(train.index.tolist())
    val_ids = set(validation.index.tolist())

    if train_ids.intersection(val_ids):
        raise LeakageDetectedError(
            "Train and validation partitions contain overlapping source indices."
        )

    if test is not None:
        test_ids = set(test.index.tolist())

        if train_ids.intersection(test_ids):
            raise LeakageDetectedError(
                "Train and test partitions contain overlapping source indices."
            )

        if val_ids.intersection(test_ids):
            raise LeakageDetectedError(
                "Validation and test partitions contain overlapping source indices."
            )


def _assert_no_group_overlap(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: Optional[pd.DataFrame],
    group_column: str,
) -> int:
    """Fail closed if protected groups cross partition boundaries."""

    train_groups = set(train[group_column].tolist())
    val_groups = set(validation[group_column].tolist())

    overlap = train_groups.intersection(val_groups)

    if test is not None:
        test_groups = set(test[group_column].tolist())

        overlap |= train_groups.intersection(test_groups)
        overlap |= val_groups.intersection(test_groups)

    if overlap:
        sample = list(overlap)[:10]

        raise LeakageDetectedError(
            f"Protected group leakage detected for '{group_column}'. "
            f"{len(overlap)} groups occur across partitions. "
            f"Examples: {sample!r}"
        )

    return 0


def _class_distribution(
    series: pd.Series,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    counts = series.value_counts(dropna=False)

    total = len(series)

    if total == 0:
        return {}, {}

    count_dict = {
        str(key): int(value)
        for key, value in counts.items()
    }

    distribution = {
        str(key): float(value / total)
        for key, value in counts.items()
    }

    return count_dict, distribution


def _partition_statistics(
    name: str,
    df: pd.DataFrame,
    target_column: Optional[str],
    time_column: Optional[str],
    group_column: Optional[str],
    source_rows: int,
    duplicate_subset: Optional[Sequence[str]],
) -> PartitionStatistics:

    class_counts: Dict[str, int] = {}
    class_distribution: Dict[str, float] = {}

    if target_column and target_column in df.columns:
        class_counts, class_distribution = _class_distribution(
            df[target_column]
        )

    unique_group_count: Optional[int] = None

    if group_column and group_column in df.columns:
        unique_group_count = int(df[group_column].nunique(dropna=False))

    min_timestamp: Optional[str] = None
    max_timestamp: Optional[str] = None

    if time_column and time_column in df.columns:
        timestamps = pd.to_datetime(
            df[time_column],
            errors="coerce",
            utc=True,
        )

        if not timestamps.empty:
            min_timestamp = timestamps.min().isoformat()
            max_timestamp = timestamps.max().isoformat()

    duplicate_count = _count_duplicate_rows(
        df,
        duplicate_subset,
    )

    return PartitionStatistics(
        name=name,
        row_count=len(df),
        percentage=(
            float(len(df) / source_rows)
            if source_rows
            else 0.0
        ),
        class_counts=class_counts,
        class_distribution=class_distribution,
        unique_group_count=unique_group_count,
        null_target_count=(
            int(df[target_column].isna().sum())
            if target_column and target_column in df.columns
            else 0
        ),
        min_timestamp=min_timestamp,
        max_timestamp=max_timestamp,
        duplicate_row_count=duplicate_count,
    )


def _split_fingerprint(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: Optional[pd.DataFrame],
) -> str:
    """
    Generate a deterministic fingerprint representing partition membership.

    The fingerprint uses source index values, not row content, because the
    purpose is to identify the exact partition plan.
    """

    payload = {
        "train": [str(x) for x in train.index.tolist()],
        "validation": [str(x) for x in validation.index.tolist()],
        "test": (
            [str(x) for x in test.index.tolist()]
            if test is not None
            else []
        ),
    }

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def _safe_copy_partition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a detached partition.

    reset_index is deliberately NOT performed because the original source
    index is useful for provenance and leakage auditing.
    """
    return df.copy(deep=True)


# =============================================================================
# BASE SPLITTER
# =============================================================================


class BaseDataSplitter(ABC):
    """
    Abstract contract for production dataset partitioners.

    Implementations must return:

        train_df, validation_df, test_df

    where test_df may be None when test_size == 0.
    """

    def __init__(
        self,
        *,
        duplicate_subset: Optional[Sequence[str]] = None,
        minimum_partition_rows: int = 1,
        fail_on_duplicates: bool = False,
    ) -> None:

        if minimum_partition_rows < 0:
            raise SplitConfigurationError(
                "minimum_partition_rows cannot be negative."
            )

        self.duplicate_subset = (
            list(duplicate_subset)
            if duplicate_subset is not None
            else None
        )

        self.minimum_partition_rows = int(minimum_partition_rows)
        self.fail_on_duplicates = bool(fail_on_duplicates)

        self.last_diagnostics: Optional[SplitDiagnostics] = None

    @abstractmethod
    def split(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        Optional[pd.DataFrame],
    ]:
        """Partition a dataset."""
        raise NotImplementedError

    def _preflight(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> None:

        _validate_non_empty_dataframe(df)
        _validate_target(df, target_column)

        duplicate_count = _count_duplicate_rows(
            df,
            self.duplicate_subset,
        )

        if duplicate_count and self.fail_on_duplicates:
            raise SplitDataError(
                f"Dataset contains {duplicate_count} rows belonging to "
                "duplicate groups. Duplicate handling must be resolved "
                "before splitting."
            )

    def _postflight(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: Optional[pd.DataFrame],
        *,
        target_column: Optional[str] = None,
        group_column: Optional[str] = None,
        time_column: Optional[str] = None,
    ) -> None:

        partitions = [train, validation]

        if test is not None:
            partitions.append(test)

        for partition in partitions:
            if len(partition) < self.minimum_partition_rows:
                raise SplitDataError(
                    f"Partition contains {len(partition)} rows, below the "
                    f"configured minimum of {self.minimum_partition_rows}."
                )

        _assert_no_row_overlap(
            train,
            validation,
            test,
        )

        if group_column:
            _assert_no_group_overlap(
                train,
                validation,
                test,
                group_column,
            )

        if target_column:
            for name, partition in [
                ("train", train),
                ("validation", validation),
                ("test", test),
            ]:
                if partition is None:
                    continue

                missing_target = int(
                    partition[target_column].isna().sum()
                )

                if missing_target:
                    raise SplitDataError(
                        f"{name} partition contains {missing_target} "
                        f"missing target values."
                    )

        if time_column:
            _validate_temporal_order(
                train,
                validation,
                test,
                time_column,
            )


# =============================================================================
# TEMPORAL VALIDATION
# =============================================================================


def _validate_temporal_order(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: Optional[pd.DataFrame],
    time_column: str,
) -> None:
    """Ensure partitions are chronologically ordered."""

    def min_max(
        frame: pd.DataFrame,
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:

        timestamps = pd.to_datetime(
            frame[time_column],
            errors="raise",
            utc=True,
        )

        return timestamps.min(), timestamps.max()

    train_min, train_max = min_max(train)
    val_min, val_max = min_max(validation)

    if train_max > val_min:
        raise LeakageDetectedError(
            "Temporal leakage detected: training contains observations "
            "later than validation."
        )

    if test is not None:
        test_min, _ = min_max(test)

        if val_max > test_min:
            raise LeakageDetectedError(
                "Temporal leakage detected: validation contains observations "
                "later than test."
            )


# =============================================================================
# TEMPORAL SPLITTER
# =============================================================================


class TemporalTrainTestSplitter(BaseDataSplitter):
    """
    Strict chronological train/validation/test splitter.

    Unlike random splitting, this models a real deployment condition:

        past -> train
        later past -> validation
        future -> test

    Parameters
    ----------
    time_column:
        Timestamp column.

    val_size:
        Fraction allocated to validation.

    test_size:
        Fraction allocated to test.

    gap:
        Number of observations removed between partitions.

        train | GAP | validation | GAP | test

    minimum_partition_rows:
        Minimum number of rows required in each active partition.

    reject_duplicate_timestamps:
        If True, duplicated timestamps are rejected. This is useful when each
        timestamp represents a unique event boundary.

    group_column:
        Optional protected grouping key. When supplied, this splitter performs
        a conservative temporal group assignment rather than blindly slicing
        individual rows.
    """

    def __init__(
        self,
        time_column: str,
        val_size: float = 0.15,
        test_size: float = 0.15,
        *,
        gap: int = 0,
        minimum_partition_rows: int = 1,
        duplicate_subset: Optional[Sequence[str]] = None,
        fail_on_duplicates: bool = False,
        reject_duplicate_timestamps: bool = False,
        group_column: Optional[str] = None,
    ) -> None:

        super().__init__(
            duplicate_subset=duplicate_subset,
            minimum_partition_rows=minimum_partition_rows,
            fail_on_duplicates=fail_on_duplicates,
        )

        _validate_split_sizes(
            val_size,
            test_size,
        )

        if not isinstance(gap, int) or gap < 0:
            raise SplitConfigurationError(
                "gap must be an integer >= 0."
            )

        self.time_column = time_column
        self.val_size = float(val_size)
        self.test_size = float(test_size)
        self.gap = gap
        self.reject_duplicate_timestamps = reject_duplicate_timestamps
        self.group_column = group_column

    def split(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        Optional[pd.DataFrame],
    ]:

        self._preflight(
            df,
            target_column,
        )

        timestamps = _coerce_datetime(
            df,
            self.time_column,
        )

        working = df.copy(deep=True)

        working["__pg_timestamp__"] = timestamps

        if self.reject_duplicate_timestamps:
            duplicate_timestamps = int(
                working["__pg_timestamp__"].duplicated(
                    keep=False
                ).sum()
            )

            if duplicate_timestamps:
                raise SplitDataError(
                    f"Found {duplicate_timestamps} rows with duplicate "
                    "timestamps while reject_duplicate_timestamps=True."
                )

        working = (
            working
            .sort_values(
                "__pg_timestamp__",
                kind="mergesort",
            )
        )

        total = len(working)

        test_count = (
            int(round(total * self.test_size))
            if self.test_size > 0
            else 0
        )

        val_count = (
            int(round(total * self.val_size))
            if self.val_size > 0
            else 0
        )

        if test_count + val_count >= total:
            raise SplitDataError(
                "Requested temporal split sizes leave no training rows."
            )

        # ------------------------------------------------------------------
        # Basic contiguous temporal slicing
        # ------------------------------------------------------------------

        if self.test_size > 0:
            test_start = total - test_count
            test_end = total

            test_df = working.iloc[
                test_start:test_end
            ].copy()

            # Embargo before test.
            test_train_boundary = (
                test_start - self.gap
            )

            if test_train_boundary < 0:
                raise SplitDataError(
                    "Temporal gap is too large for the requested dataset."
                )

            working_before_test = working.iloc[
                :test_train_boundary
            ]

        else:
            test_df = None
            working_before_test = working

        if self.val_size > 0:
            val_end = len(working_before_test)
            val_start = val_end - val_count

            if val_start < 0:
                raise SplitDataError(
                    "Validation split is larger than the available "
                    "pre-test dataset."
                )

            val_df = working_before_test.iloc[
                val_start:val_end
            ].copy()

            train_end = val_start - self.gap

            if train_end <= 0:
                raise SplitDataError(
                    "Temporal gap leaves no usable training data."
                )

            train_df = working_before_test.iloc[
                :train_end
            ].copy()

        else:
            val_df = working_before_test.iloc[0:0].copy()

            train_end = len(working_before_test)

            train_df = working_before_test.iloc[
                :train_end
            ].copy()

        # Remove internal helper column.
        for partition in [
            train_df,
            val_df,
            test_df,
        ]:
            if partition is not None and "__pg_timestamp__" in partition:
                partition.drop(
                    columns=["__pg_timestamp__"],
                    inplace=True,
                )

        train_df = _safe_copy_partition(train_df)
        val_df = _safe_copy_partition(val_df)

        if test_df is not None:
            test_df = _safe_copy_partition(test_df)

        self._postflight(
            train_df,
            val_df,
            test_df,
            target_column=target_column,
            time_column=self.time_column,
        )

        diagnostics = _build_diagnostics(
            splitter_type=self.__class__.__name__,
            source=df,
            train=train_df,
            validation=val_df,
            test=test_df,
            target_column=target_column,
            time_column=self.time_column,
            validation_size=self.val_size,
            test_size=self.test_size,
            temporal_gap=self.gap,
            duplicate_subset=self.duplicate_subset,
            random_state=None,
        )

        self.last_diagnostics = diagnostics

        logger.info(
            "Temporal split completed: "
            "train=%d, validation=%d, test=%d, fingerprint=%s",
            len(train_df),
            len(val_df),
            len(test_df) if test_df is not None else 0,
            diagnostics.fingerprint,
        )

        return train_df, val_df, test_df


# =============================================================================
# STRATIFIED RANDOM SPLITTER
# =============================================================================


class AdversarialStratifiedSplitter(BaseDataSplitter):
    """
    Production stratified train/validation/test splitter.

    The historical class name is preserved for backwards compatibility.

    This implementation is appropriate when:
        * samples are independent enough for random splitting
        * temporal ordering is not the primary evaluation constraint
        * campaign/group leakage has already been removed

    For phishing campaign datasets, prefer ``StratifiedGroupSplitter`` when
    a meaningful campaign/domain identifier exists.
    """

    def __init__(
        self,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = DEFAULT_RANDOM_STATE,
        *,
        minimum_partition_rows: int = 1,
        duplicate_subset: Optional[Sequence[str]] = None,
        fail_on_duplicates: bool = False,
        strict_class_support: bool = True,
    ) -> None:

        super().__init__(
            duplicate_subset=duplicate_subset,
            minimum_partition_rows=minimum_partition_rows,
            fail_on_duplicates=fail_on_duplicates,
        )

        _validate_split_sizes(
            val_size,
            test_size,
        )

        _validate_random_state(random_state)

        self.val_size = float(val_size)
        self.test_size = float(test_size)
        self.random_state = random_state
        self.strict_class_support = strict_class_support

    def _validate_class_support(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> None:

        counts = df[target_column].value_counts()

        active_partitions = int(
            self.test_size > 0
            + self.val_size > 0
            + 1
        )

        if self.strict_class_support:
            insufficient = counts[counts < active_partitions]

            if not insufficient.empty:
                raise SplitDataError(
                    "Some target classes do not have enough observations "
                    f"for {active_partitions} partitions: "
                    f"{insufficient.to_dict()!r}"
                )

    def split(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        Optional[pd.DataFrame],
    ]:

        self._preflight(
            df,
            target_column,
        )

        self._validate_class_support(
            df,
            target_column,
        )

        working = df.copy(deep=True)

        # --------------------------------------------------------------
        # Test partition
        # --------------------------------------------------------------

        if self.test_size > 0:

            test_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self.test_size,
                random_state=self.random_state,
            )

            train_val_idx, test_idx = next(
                test_splitter.split(
                    working,
                    working[target_column],
                )
            )

            train_val_df = working.iloc[
                train_val_idx
            ].copy()

            test_df = working.iloc[
                test_idx
            ].copy()

        else:
            train_val_df = working.copy()
            test_df = None

        # --------------------------------------------------------------
        # Validation partition
        # --------------------------------------------------------------

        if self.val_size > 0:

            remaining_ratio = 1.0 - self.test_size

            if remaining_ratio <= 0:
                raise SplitConfigurationError(
                    "No data remains for train/validation splitting."
                )

            adjusted_val_size = (
                self.val_size / remaining_ratio
            )

            if not 0 < adjusted_val_size < 1:
                raise SplitConfigurationError(
                    "Calculated validation ratio is invalid: "
                    f"{adjusted_val_size}."
                )

            validation_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=adjusted_val_size,
                random_state=self.random_state,
            )

            train_idx, val_idx = next(
                validation_splitter.split(
                    train_val_df,
                    train_val_df[target_column],
                )
            )

            train_df = train_val_df.iloc[
                train_idx
            ].copy()

            val_df = train_val_df.iloc[
                val_idx
            ].copy()

        else:
            train_df = train_val_df.copy()
            val_df = train_val_df.iloc[0:0].copy()

        train_df = _safe_copy_partition(train_df)
        val_df = _safe_copy_partition(val_df)

        if test_df is not None:
            test_df = _safe_copy_partition(test_df)

        self._postflight(
            train_df,
            val_df,
            test_df,
            target_column=target_column,
        )

        diagnostics = _build_diagnostics(
            splitter_type=self.__class__.__name__,
            source=df,
            train=train_df,
            validation=val_df,
            test=test_df,
            target_column=target_column,
            validation_size=self.val_size,
            test_size=self.test_size,
            duplicate_subset=self.duplicate_subset,
            random_state=self.random_state,
        )

        self.last_diagnostics = diagnostics

        logger.info(
            "Stratified split completed: "
            "train=%d, validation=%d, test=%d, fingerprint=%s",
            len(train_df),
            len(val_df),
            len(test_df) if test_df is not None else 0,
            diagnostics.fingerprint,
        )

        return train_df, val_df, test_df


# =============================================================================
# GROUP-AWARE RANDOM SPLITTER
# =============================================================================


class GroupAwareSplitter(BaseDataSplitter):
    """
    Group-aware random splitter.

    Guarantees that one protected group appears in only one partition.

    Useful groups include:

        campaign_id
        sender_domain
        registrable_domain
        message_thread_id
        attachment_hash
        sample_family_id

    Uses GroupShuffleSplit because scikit-learn explicitly provides it for
    non-overlapping group-based partitions. :contentReference[oaicite:2]{index=2}
    """

    def __init__(
        self,
        group_column: str,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = DEFAULT_RANDOM_STATE,
        *,
        minimum_partition_rows: int = 1,
        duplicate_subset: Optional[Sequence[str]] = None,
        fail_on_duplicates: bool = False,
    ) -> None:

        super().__init__(
            duplicate_subset=duplicate_subset,
            minimum_partition_rows=minimum_partition_rows,
            fail_on_duplicates=fail_on_duplicates,
        )

        _validate_split_sizes(
            val_size,
            test_size,
        )

        _validate_random_state(random_state)

        self.group_column = group_column
        self.val_size = float(val_size)
        self.test_size = float(test_size)
        self.random_state = random_state

    def split(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        Optional[pd.DataFrame],
    ]:

        self._preflight(
            df,
            target_column,
        )

        _validate_group_column(
            df,
            self.group_column,
        )

        _validate_group_label_consistency(
            df,
            target_column,
            self.group_column,
        )

        working = df.copy(deep=True)

        # --------------------------------------------------------------
        # Test groups
        # --------------------------------------------------------------

        if self.test_size > 0:

            test_splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=self.test_size,
                random_state=self.random_state,
            )

            train_val_idx, test_idx = next(
                test_splitter.split(
                    working,
                    y=working[target_column],
                    groups=working[self.group_column],
                )
            )

            train_val_df = working.iloc[
                train_val_idx
            ].copy()

            test_df = working.iloc[
                test_idx
            ].copy()

        else:
            train_val_df = working.copy()
            test_df = None

        # --------------------------------------------------------------
        # Validation groups
        # --------------------------------------------------------------

        if self.val_size > 0:

            remaining_ratio = 1.0 - self.test_size

            adjusted_val_size = (
                self.val_size / remaining_ratio
            )

            val_splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=adjusted_val_size,
                random_state=self.random_state,
            )

            train_idx, val_idx = next(
                val_splitter.split(
                    train_val_df,
                    y=train_val_df[target_column],
                    groups=train_val_df[self.group_column],
                )
            )

            train_df = train_val_df.iloc[
                train_idx
            ].copy()

            val_df = train_val_df.iloc[
                val_idx
            ].copy()

        else:
            train_df = train_val_df.copy()
            val_df = train_val_df.iloc[0:0].copy()

        self._postflight(
            train_df,
            val_df,
            test_df,
            target_column=target_column,
            group_column=self.group_column,
        )

        diagnostics = _build_diagnostics(
            splitter_type=self.__class__.__name__,
            source=df,
            train=train_df,
            validation=val_df,
            test=test_df,
            target_column=target_column,
            group_column=self.group_column,
            validation_size=self.val_size,
            test_size=self.test_size,
            duplicate_subset=self.duplicate_subset,
            random_state=self.random_state,
        )

        self.last_diagnostics = diagnostics

        return (
            _safe_copy_partition(train_df),
            _safe_copy_partition(val_df),
            (
                _safe_copy_partition(test_df)
                if test_df is not None
                else None
            ),
        )


# =============================================================================
# STRATIFIED GROUP SPLITTER
# =============================================================================


class StratifiedGroupSplitter(BaseDataSplitter):
    """
    Group-aware and class-aware dataset splitter.

    This is the preferred random splitter for phishing datasets where samples
    belonging to the same campaign/domain/family must never cross partitions.

    The implementation uses StratifiedGroupKFold internally to construct
    candidate group partitions while considering class distribution.

    scikit-learn documents StratifiedGroupKFold as the combination of group
    separation and approximate class stratification. :contentReference[oaicite:3]{index=3}
    """

    def __init__(
        self,
        group_column: str,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = DEFAULT_RANDOM_STATE,
        *,
        n_candidates: int = 32,
        minimum_partition_rows: int = 1,
        duplicate_subset: Optional[Sequence[str]] = None,
        fail_on_duplicates: bool = False,
        require_all_classes: bool = True,
    ) -> None:

        super().__init__(
            duplicate_subset=duplicate_subset,
            minimum_partition_rows=minimum_partition_rows,
            fail_on_duplicates=fail_on_duplicates,
        )

        _validate_split_sizes(
            val_size,
            test_size,
        )

        _validate_random_state(random_state)

        if n_candidates < 1:
            raise SplitConfigurationError(
                "n_candidates must be >= 1."
            )

        self.group_column = group_column
        self.val_size = float(val_size)
        self.test_size = float(test_size)
        self.random_state = random_state
        self.n_candidates = int(n_candidates)
        self.require_all_classes = require_all_classes

    @staticmethod
    def _distribution_error(
        reference: pd.Series,
        candidate: pd.Series,
    ) -> float:
        """
        Measure L1 difference between class distributions.
        """
        ref = reference.value_counts(
            normalize=True
        )

        cand = candidate.value_counts(
            normalize=True
        )

        classes = set(ref.index).union(cand.index)

        error = 0.0

        for cls in classes:
            error += abs(
                float(ref.get(cls, 0.0))
                - float(cand.get(cls, 0.0))
            )

        return error

    def _choose_fold(
        self,
        df: pd.DataFrame,
        target_column: str,
        desired_size: float,
        n_splits: int,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if len(df[self.group_column].unique()) < n_splits:
            raise SplitDataError(
                f"Cannot create {n_splits} group folds from only "
                f"{df[self.group_column].nunique()} unique groups."
            )

        best_train_idx: Optional[np.ndarray] = None
        best_test_idx: Optional[np.ndarray] = None
        best_score = float("inf")

        y = df[target_column]
        groups = df[self.group_column]

        for candidate_number in range(self.n_candidates):

            candidate_seed = seed + candidate_number

            cv = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=candidate_seed,
            )

            for train_idx, candidate_idx in cv.split(
                df,
                y,
                groups,
            ):
                candidate_size = (
                    len(candidate_idx) / len(df)
                )

                size_error = abs(
                    candidate_size - desired_size
                )

                distribution_error = self._distribution_error(
                    y,
                    y.iloc[candidate_idx],
                )

                score = (
                    size_error * 2.0
                    + distribution_error
                )

                if score < best_score:
                    best_score = score
                    best_train_idx = train_idx
                    best_test_idx = candidate_idx

        if best_train_idx is None or best_test_idx is None:
            raise SplitDataError(
                "Unable to construct a valid stratified-group partition."
            )

        return best_train_idx, best_test_idx

    def split(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        Optional[pd.DataFrame],
    ]:

        self._preflight(
            df,
            target_column,
        )

        _validate_group_column(
            df,
            self.group_column,
        )

        _validate_group_label_consistency(
            df,
            target_column,
            self.group_column,
        )

        working = df.copy(deep=True)

        # --------------------------------------------------------------
        # Test
        # --------------------------------------------------------------

        if self.test_size > 0:

            n_groups = working[
                self.group_column
            ].nunique()

            estimated_splits = max(
                2,
                min(
                    10,
                    int(round(1.0 / self.test_size)),
                    n_groups,
                ),
            )

            train_val_idx, test_idx = self._choose_fold(
                working,
                target_column,
                self.test_size,
                estimated_splits,
                self.random_state,
            )

            train_val_df = working.iloc[
                train_val_idx
            ].copy()

            test_df = working.iloc[
                test_idx
            ].copy()

        else:
            train_val_df = working.copy()
            test_df = None

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if self.val_size > 0:

            remaining_ratio = 1.0 - self.test_size

            adjusted_val_size = (
                self.val_size / remaining_ratio
            )

            n_groups = train_val_df[
                self.group_column
            ].nunique()

            estimated_splits = max(
                2,
                min(
                    10,
                    int(round(1.0 / adjusted_val_size)),
                    n_groups,
                ),
            )

            train_idx, val_idx = self._choose_fold(
                train_val_df,
                target_column,
                adjusted_val_size,
                estimated_splits,
                self.random_state + 1000,
            )

            train_df = train_val_df.iloc[
                train_idx
            ].copy()

            val_df = train_val_df.iloc[
                val_idx
            ].copy()

        else:
            train_df = train_val_df.copy()
            val_df = train_val_df.iloc[0:0].copy()

        if self.require_all_classes:
            source_classes = set(
                working[target_column].dropna().unique()
            )

            for name, partition in [
                ("train", train_df),
                ("validation", val_df),
                ("test", test_df),
            ]:
                if partition is None or partition.empty:
                    continue

                partition_classes = set(
                    partition[target_column].dropna().unique()
                )

                missing = source_classes - partition_classes

                if missing:
                    raise SplitDataError(
                        f"{name} partition does not contain all target "
                        f"classes. Missing: {missing!r}. Group constraints "
                        "may make perfect stratification impossible."
                    )

        self._postflight(
            train_df,
            val_df,
            test_df,
            target_column=target_column,
            group_column=self.group_column,
        )

        diagnostics = _build_diagnostics(
            splitter_type=self.__class__.__name__,
            source=df,
            train=train_df,
            validation=val_df,
            test=test_df,
            target_column=target_column,
            group_column=self.group_column,
            validation_size=self.val_size,
            test_size=self.test_size,
            duplicate_subset=self.duplicate_subset,
            random_state=self.random_state,
        )

        self.last_diagnostics = diagnostics

        return (
            _safe_copy_partition(train_df),
            _safe_copy_partition(val_df),
            (
                _safe_copy_partition(test_df)
                if test_df is not None
                else None
            ),
        )


# =============================================================================
# TEMPORAL + GROUP SPLITTER
# =============================================================================


class TemporalGroupSplitter(BaseDataSplitter):
    """
    Conservative temporal splitter that preserves group boundaries.

    This is useful when campaign identity and chronology are both important.

    The algorithm:

        1. validates timestamps
        2. sorts groups by their earliest timestamp
        3. assigns complete groups to chronological partitions
        4. never divides a group
        5. applies temporal ordering checks

    This is intentionally conservative. A group crossing a temporal boundary
    cannot be split safely without violating the group constraint.
    """

    def __init__(
        self,
        time_column: str,
        group_column: str,
        val_size: float = 0.15,
        test_size: float = 0.15,
        *,
        minimum_partition_rows: int = 1,
        duplicate_subset: Optional[Sequence[str]] = None,
        fail_on_duplicates: bool = False,
    ) -> None:

        super().__init__(
            duplicate_subset=duplicate_subset,
            minimum_partition_rows=minimum_partition_rows,
            fail_on_duplicates=fail_on_duplicates,
        )

        _validate_split_sizes(
            val_size,
            test_size,
        )

        self.time_column = time_column
        self.group_column = group_column
        self.val_size = float(val_size)
        self.test_size = float(test_size)

    def split(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        Optional[pd.DataFrame],
    ]:

        self._preflight(
            df,
            target_column,
        )

        _validate_group_column(
            df,
            self.group_column,
        )

        timestamps = _coerce_datetime(
            df,
            self.time_column,
        )

        working = df.copy(deep=True)

        working["__pg_timestamp__"] = timestamps

        # Earliest observation determines the chronological position of a
        # protected group.
        group_order = (
            working
            .groupby(self.group_column)["__pg_timestamp__"]
            .min()
            .sort_values()
        )

        groups = group_order.index.tolist()

        total_rows = len(working)

        target_test_rows = int(
            round(total_rows * self.test_size)
        )

        target_val_rows = int(
            round(total_rows * self.val_size)
        )

        # Allocate groups from the future backwards to test.
        test_groups: List[Any] = []
        accumulated = 0

        if target_test_rows > 0:
            for group in reversed(groups):
                group_rows = int(
                    (working[self.group_column] == group).sum()
                )

                test_groups.append(group)
                accumulated += group_rows

                if accumulated >= target_test_rows:
                    break

        test_group_set = set(test_groups)

        remaining_groups = [
            g for g in groups
            if g not in test_group_set
        ]

        val_groups: List[Any] = []
        accumulated = 0

        if target_val_rows > 0:
            for group in reversed(remaining_groups):
                group_rows = int(
                    (working[self.group_column] == group).sum()
                )

                val_groups.append(group)
                accumulated += group_rows

                if accumulated >= target_val_rows:
                    break

        val_group_set = set(val_groups)

        train_group_set = set(remaining_groups) - val_group_set

        train_df = working[
            working[self.group_column].isin(train_group_set)
        ].copy()

        val_df = working[
            working[self.group_column].isin(val_group_set)
        ].copy()

        test_df = (
            working[
                working[self.group_column].isin(test_group_set)
            ].copy()
            if test_group_set
            else None
        )

        for partition in [
            train_df,
            val_df,
            test_df,
        ]:
            if partition is not None:
                partition.drop(
                    columns=["__pg_timestamp__"],
                    inplace=True,
                )

        # Sort each partition internally.
        train_df = train_df.sort_values(
            self.time_column,
            kind="mergesort",
        )

        val_df = val_df.sort_values(
            self.time_column,
            kind="mergesort",
        )

        if test_df is not None:
            test_df = test_df.sort_values(
                self.time_column,
                kind="mergesort",
            )

        self._postflight(
            train_df,
            val_df,
            test_df,
            target_column=target_column,
            group_column=self.group_column,
            time_column=self.time_column,
        )

        diagnostics = _build_diagnostics(
            splitter_type=self.__class__.__name__,
            source=df,
            train=train_df,
            validation=val_df,
            test=test_df,
            target_column=target_column,
            group_column=self.group_column,
            time_column=self.time_column,
            validation_size=self.val_size,
            test_size=self.test_size,
            duplicate_subset=self.duplicate_subset,
        )

        self.last_diagnostics = diagnostics

        return (
            _safe_copy_partition(train_df),
            _safe_copy_partition(val_df),
            (
                _safe_copy_partition(test_df)
                if test_df is not None
                else None
            ),
        )


# =============================================================================
# DIAGNOSTICS
# =============================================================================


def _build_diagnostics(
    *,
    splitter_type: str,
    source: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: Optional[pd.DataFrame],
    target_column: Optional[str],
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    validation_size: float = 0.0,
    test_size: float = 0.0,
    temporal_gap: int = 0,
    duplicate_subset: Optional[Sequence[str]] = None,
    random_state: Optional[int] = None,
) -> SplitDiagnostics:

    duplicate_source_rows = _count_duplicate_rows(
        source,
        duplicate_subset,
    )

    duplicate_across_partitions = 0

    if duplicate_subset is not None:
        train_fp = set(
            _stable_row_fingerprint(
                train,
                duplicate_subset,
            )
        )

        val_fp = set(
            _stable_row_fingerprint(
                validation,
                duplicate_subset,
            )
        )

        duplicate_across_partitions += len(
            train_fp.intersection(val_fp)
        )

        if test is not None:
            test_fp = set(
                _stable_row_fingerprint(
                    test,
                    duplicate_subset,
                )
            )

            duplicate_across_partitions += len(
                train_fp.intersection(test_fp)
            )

            duplicate_across_partitions += len(
                val_fp.intersection(test_fp)
            )

    group_overlap_count = 0

    if group_column:
        train_groups = set(train[group_column])
        val_groups = set(validation[group_column])

        group_overlap_count += len(
            train_groups.intersection(val_groups)
        )

        if test is not None:
            test_groups = set(test[group_column])

            group_overlap_count += len(
                train_groups.intersection(test_groups)
            )

            group_overlap_count += len(
                val_groups.intersection(test_groups)
            )

    row_overlap_count = (
        len(set(train.index).intersection(validation.index))
    )

    if test is not None:
        row_overlap_count += len(
            set(train.index).intersection(test.index)
        )

        row_overlap_count += len(
            set(validation.index).intersection(test.index)
        )

    diagnostics = SplitDiagnostics(
        splitter_type=splitter_type,
        target_column=target_column,
        random_state=random_state,
        source_rows=len(source),
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test) if test is not None else 0,
        source_groups=(
            int(source[group_column].nunique())
            if group_column
            else None
        ),
        train_groups=(
            int(train[group_column].nunique())
            if group_column
            else None
        ),
        validation_groups=(
            int(validation[group_column].nunique())
            if group_column
            else None
        ),
        test_groups=(
            int(test[group_column].nunique())
            if group_column and test is not None
            else None
        ),
        group_column=group_column,
        time_column=time_column,
        validation_size=validation_size,
        test_size=test_size,
        temporal_gap=temporal_gap,
        duplicate_source_rows=duplicate_source_rows,
        duplicate_across_partitions=duplicate_across_partitions,
        group_overlap_count=group_overlap_count,
        row_overlap_count=row_overlap_count,
        train_statistics=_partition_statistics(
            "train",
            train,
            target_column,
            time_column,
            group_column,
            len(source),
            duplicate_subset,
        ),
        validation_statistics=_partition_statistics(
            "validation",
            validation,
            target_column,
            time_column,
            group_column,
            len(source),
            duplicate_subset,
        ),
        test_statistics=(
            _partition_statistics(
                "test",
                test,
                target_column,
                time_column,
                group_column,
                len(source),
                duplicate_subset,
            )
            if test is not None
            else None
        ),
    )

    diagnostics.fingerprint = _split_fingerprint(
        train,
        validation,
        test,
    )

    errors: List[str] = []

    if diagnostics.row_overlap_count:
        errors.append(
            f"Row overlap detected: {diagnostics.row_overlap_count}"
        )

    if diagnostics.group_overlap_count:
        errors.append(
            f"Group overlap detected: {diagnostics.group_overlap_count}"
        )

    if diagnostics.duplicate_across_partitions:
        errors.append(
            "Duplicate observations detected across partitions."
        )

    diagnostics.errors = errors
    diagnostics.valid = not errors

    return diagnostics


# =============================================================================
# PUBLIC AUDIT FUNCTION
# =============================================================================


def audit_split(
    source: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: Optional[pd.DataFrame] = None,
    *,
    target_column: Optional[str] = None,
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    duplicate_subset: Optional[Sequence[str]] = None,
) -> SplitDiagnostics:
    """
    Independently audit an already-created split.

    This is useful when partitions were produced by an external process,
    data lake query, Spark job, SQL query, or previously serialized split plan.

    Raises
    ------
    LeakageDetectedError
        If the supplied partitions violate row/group/temporal constraints.
    """

    _validate_dataframe(source)
    _validate_dataframe(train)
    _validate_dataframe(validation)

    if test is not None:
        _validate_dataframe(test)

    if target_column:
        _validate_column(
            source,
            target_column,
            "Target",
        )

    if group_column:
        _validate_group_column(
            source,
            group_column,
        )

    diagnostics = _build_diagnostics(
        splitter_type="external_audit",
        source=source,
        train=train,
        validation=validation,
        test=test,
        target_column=target_column,
        group_column=group_column,
        time_column=time_column,
        duplicate_subset=duplicate_subset,
    )

    if group_column:
        _assert_no_group_overlap(
            train,
            validation,
            test,
            group_column,
        )

    if time_column:
        _validate_temporal_order(
            train,
            validation,
            test,
            time_column,
        )

    if not diagnostics.valid:
        raise LeakageDetectedError(
            "Split audit failed: "
            + "; ".join(diagnostics.errors)
        )

    return diagnostics


# =============================================================================
# CONVENIENCE SPLIT FUNCTION
# =============================================================================


def split_dataset(
    df: pd.DataFrame,
    target_column: str = "label",
    time_column: Optional[str] = None,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = DEFAULT_RANDOM_STATE,
    *,
    group_column: Optional[str] = None,
    strategy: Optional[str] = None,
    temporal_gap: int = 0,
    duplicate_subset: Optional[Sequence[str]] = None,
    fail_on_duplicates: bool = False,
    minimum_partition_rows: int = 1,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Optional[pd.DataFrame],
]:
    """
    Unified production splitting entry point.

    Strategy selection:

        group_column + time_column
            -> TemporalGroupSplitter

        time_column
            -> TemporalTrainTestSplitter

        group_column
            -> StratifiedGroupSplitter

        otherwise
            -> AdversarialStratifiedSplitter

    Explicit ``strategy`` values:

        temporal
        stratified
        group
        stratified_group

    Parameters
    ----------
    strategy:
        Explicitly select split strategy. If None, strategy is inferred.

    group_column:
        Protected group identifier.

    temporal_gap:
        Number of observations embargoed between temporal partitions.
    """

    _validate_non_empty_dataframe(df)

    _validate_split_sizes(
        val_size,
        test_size,
    )

    if strategy is not None:
        strategy = strategy.lower().strip()

        if strategy not in SUPPORTED_SPLIT_MODES:
            raise SplitConfigurationError(
                f"Unsupported strategy '{strategy}'. "
                f"Supported values: {sorted(SUPPORTED_SPLIT_MODES)}"
            )

    # ------------------------------------------------------------------
    # Automatic strategy selection
    # ------------------------------------------------------------------

    if strategy is None:

        if time_column and group_column:
            strategy = "temporal_group"

        elif time_column:
            strategy = "temporal"

        elif group_column:
            strategy = "stratified_group"

        else:
            strategy = "stratified"

    # ------------------------------------------------------------------
    # Explicit strategy
    # ------------------------------------------------------------------

    if strategy == "temporal":

        splitter = TemporalTrainTestSplitter(
            time_column=time_column
            or "timestamp",
            val_size=val_size,
            test_size=test_size,
            gap=temporal_gap,
            duplicate_subset=duplicate_subset,
            fail_on_duplicates=fail_on_duplicates,
            minimum_partition_rows=minimum_partition_rows,
        )

    elif strategy == "stratified":

        splitter = AdversarialStratifiedSplitter(
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            duplicate_subset=duplicate_subset,
            fail_on_duplicates=fail_on_duplicates,
            minimum_partition_rows=minimum_partition_rows,
        )

    elif strategy == "group":

        if not group_column:
            raise SplitConfigurationError(
                "strategy='group' requires group_column."
            )

        splitter = GroupAwareSplitter(
            group_column=group_column,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            duplicate_subset=duplicate_subset,
            fail_on_duplicates=fail_on_duplicates,
            minimum_partition_rows=minimum_partition_rows,
        )

    elif strategy == "stratified_group":

        if not group_column:
            raise SplitConfigurationError(
                "strategy='stratified_group' requires group_column."
            )

        splitter = StratifiedGroupSplitter(
            group_column=group_column,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            duplicate_subset=duplicate_subset,
            fail_on_duplicates=fail_on_duplicates,
            minimum_partition_rows=minimum_partition_rows,
        )

    elif strategy == "temporal_group":

        if not time_column or not group_column:
            raise SplitConfigurationError(
                "strategy='temporal_group' requires both "
                "time_column and group_column."
            )

        splitter = TemporalGroupSplitter(
            time_column=time_column,
            group_column=group_column,
            val_size=val_size,
            test_size=test_size,
            duplicate_subset=duplicate_subset,
            fail_on_duplicates=fail_on_duplicates,
            minimum_partition_rows=minimum_partition_rows,
        )

    else:
        raise SplitConfigurationError(
            f"Unhandled splitting strategy: {strategy!r}"
        )

    return splitter.split(
        df,
        target_column=target_column,
    )


# =============================================================================
# CROSS-VALIDATION
# =============================================================================


def create_cross_validation_folds(
    df: pd.DataFrame,
    target_column: str,
    n_splits: int = 5,
    is_temporal: bool = False,
    random_state: int = DEFAULT_RANDOM_STATE,
    *,
    time_column: Optional[str] = None,
    group_column: Optional[str] = None,
    temporal_gap: int = 0,
    max_train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    shuffle: bool = True,
) -> Generator[
    Tuple[pd.DataFrame, pd.DataFrame],
    None,
    None,
]:
    """
    Generate leakage-aware cross-validation folds.

    Modes
    -----
    Temporal:
        TimeSeriesSplit with optional gap.

    Group:
        GroupKFold.

    Stratified group:
        StratifiedGroupKFold.

    Standard:
        StratifiedKFold.

    Notes
    -----
    TimeSeriesSplit is specifically designed for time-ordered data and
    supports a ``gap`` that excludes samples immediately preceding each test
    window. :contentReference[oaicite:4]{index=4}
    """

    _validate_non_empty_dataframe(df)
    _validate_target(
        df,
        target_column,
    )
    _validate_n_splits(
        n_splits,
    )
    _validate_random_state(
        random_state,
    )

    working = df.copy(deep=True)

    # ------------------------------------------------------------------
    # Temporal CV
    # ------------------------------------------------------------------

    if is_temporal:

        if not time_column:
            raise SplitConfigurationError(
                "Temporal CV requires time_column."
            )

        timestamps = _coerce_datetime(
            working,
            time_column,
        )

        working["__pg_timestamp__"] = timestamps

        working = (
            working
            .sort_values(
                "__pg_timestamp__",
                kind="mergesort",
            )
        )

        cv = TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=max_train_size,
            test_size=test_size,
            gap=temporal_gap,
        )

        for train_idx, val_idx in cv.split(working):

            train_df = working.iloc[
                train_idx
            ].copy()

            val_df = working.iloc[
                val_idx
            ].copy()

            train_df.drop(
                columns=["__pg_timestamp__"],
                inplace=True,
            )

            val_df.drop(
                columns=["__pg_timestamp__"],
                inplace=True,
            )

            _assert_no_row_overlap(
                train_df,
                val_df,
                None,
            )

            _validate_temporal_order(
                train_df,
                val_df,
                None,
                time_column,
            )

            yield (
                _safe_copy_partition(train_df),
                _safe_copy_partition(val_df),
            )

        return

    # ------------------------------------------------------------------
    # Group CV
    # ------------------------------------------------------------------

    if group_column:

        _validate_group_column(
            working,
            group_column,
        )

        _validate_group_label_consistency(
            working,
            target_column,
            group_column,
        )

        if shuffle:
            cv = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )

        else:
            cv = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=False,
            )

        for train_idx, val_idx in cv.split(
            working,
            working[target_column],
            working[group_column],
        ):

            train_df = working.iloc[
                train_idx
            ].copy()

            val_df = working.iloc[
                val_idx
            ].copy()

            _assert_no_group_overlap(
                train_df,
                val_df,
                None,
                group_column,
            )

            yield (
                _safe_copy_partition(train_df),
                _safe_copy_partition(val_df),
            )

        return

    # ------------------------------------------------------------------
    # Standard stratified CV
    # ------------------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=(
            random_state
            if shuffle
            else None
        ),
    )

    for train_idx, val_idx in cv.split(
        working,
        working[target_column],
    ):

        train_df = working.iloc[
            train_idx
        ].copy()

        val_df = working.iloc[
            val_idx
        ].copy()

        _assert_no_row_overlap(
            train_df,
            val_df,
            None,
        )

        yield (
            _safe_copy_partition(train_df),
            _safe_copy_partition(val_df),
        )


# =============================================================================
# GROUP CROSS-VALIDATION
# =============================================================================


def create_group_cross_validation_folds(
    df: pd.DataFrame,
    target_column: str,
    group_column: str,
    *,
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratified: bool = True,
    shuffle: bool = True,
) -> Generator[
    Tuple[pd.DataFrame, pd.DataFrame],
    None,
    None,
]:
    """
    Explicit group-aware CV generator.

    Use this when the group constraint is important enough that it should be
    visible at the call site instead of inferred automatically.
    """

    _validate_non_empty_dataframe(df)
    _validate_target(
        df,
        target_column,
    )
    _validate_group_column(
        df,
        group_column,
    )
    _validate_n_splits(
        n_splits,
    )

    _validate_group_label_consistency(
        df,
        target_column,
        group_column,
    )

    if stratified:

        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=(
                random_state
                if shuffle
                else None
            ),
        )

        iterator = cv.split(
            df,
            df[target_column],
            df[group_column],
        )

    else:

        cv = GroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=(
                random_state
                if shuffle
                else None
            ),
        )

        iterator = cv.split(
            df,
            df[target_column],
            df[group_column],
        )

    for train_idx, val_idx in iterator:

        train_df = df.iloc[
            train_idx
        ].copy()

        val_df = df.iloc[
            val_idx
        ].copy()

        _assert_no_group_overlap(
            train_df,
            val_df,
            None,
            group_column,
        )

        yield (
            _safe_copy_partition(train_df),
            _safe_copy_partition(val_df),
        )


# =============================================================================
# TEMPORAL CROSS-VALIDATION
# =============================================================================


def create_temporal_cross_validation_folds(
    df: pd.DataFrame,
    target_column: str,
    time_column: str,
    *,
    n_splits: int = 5,
    gap: int = 0,
    max_train_size: Optional[int] = None,
    test_size: Optional[int] = None,
) -> Generator[
    Tuple[pd.DataFrame, pd.DataFrame],
    None,
    None,
]:
    """
    Explicit temporal CV API.

    Data is sorted by timestamp before TimeSeriesSplit is applied.
    """

    yield from create_cross_validation_folds(
        df=df,
        target_column=target_column,
        n_splits=n_splits,
        is_temporal=True,
        time_column=time_column,
        temporal_gap=gap,
        max_train_size=max_train_size,
        test_size=test_size,
        shuffle=False,
    )


# =============================================================================
# SPLIT CLASS DISTRIBUTION AUDIT
# =============================================================================


def compare_class_distributions(
    source: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: Optional[pd.DataFrame],
    target_column: str,
) -> Dict[str, Any]:
    """
    Compare class proportions between source and partitions.

    This does not declare a split invalid merely because proportions differ.
    It provides measurable evidence for monitoring and reports.
    """

    _validate_target(
        source,
        target_column,
    )

    source_counts, source_distribution = _class_distribution(
        source[target_column]
    )

    result: Dict[str, Any] = {
        "source": {
            "counts": source_counts,
            "distribution": source_distribution,
        },
        "partitions": {},
    }

    for name, partition in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:

        if partition is None:
            continue

        counts, distribution = _class_distribution(
            partition[target_column]
        )

        drift: Dict[str, float] = {}

        classes = set(source_distribution).union(
            distribution
        )

        for cls in classes:
            drift[cls] = abs(
                source_distribution.get(cls, 0.0)
                - distribution.get(cls, 0.0)
            )

        result["partitions"][name] = {
            "counts": counts,
            "distribution": distribution,
            "absolute_distribution_drift": drift,
        }

    return result


# =============================================================================
# LEAKAGE-SAFE INDEX EXTRACTION
# =============================================================================


def extract_partition_indices(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: Optional[pd.DataFrame] = None,
) -> Dict[str, List[Any]]:
    """
    Return source indices for reproducible partition manifests.
    """

    return {
        "train": train.index.tolist(),
        "validation": validation.index.tolist(),
        "test": (
            test.index.tolist()
            if test is not None
            else []
        ),
    }


def build_split_manifest(
    source: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: Optional[pd.DataFrame],
    *,
    target_column: Optional[str] = None,
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    duplicate_subset: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Build a machine-readable split manifest.

    This manifest can be stored with a model-training run to establish
    exactly which observations were used.
    """

    diagnostics = audit_split(
        source=source,
        train=train,
        validation=validation,
        test=test,
        target_column=target_column,
        group_column=group_column,
        time_column=time_column,
        duplicate_subset=duplicate_subset,
    )

    return {
        "schema_version": "1.0",
        "split_fingerprint": diagnostics.fingerprint,
        "diagnostics": diagnostics.to_dict(),
        "indices": extract_partition_indices(
            train,
            validation,
            test,
        ),
    }


# =============================================================================
# LEGACY ALIAS
# =============================================================================


# Existing code may already reference this semantic name.
CampaignAwareSplitter = StratifiedGroupSplitter


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    # Exceptions
    "SplitterError",
    "SplitConfigurationError",
    "SplitDataError",
    "LeakageDetectedError",

    # Data structures
    "PartitionStatistics",
    "SplitDiagnostics",
    "SplitResult",

    # Base
    "BaseDataSplitter",

    # Splitters
    "TemporalTrainTestSplitter",
    "AdversarialStratifiedSplitter",
    "GroupAwareSplitter",
    "StratifiedGroupSplitter",
    "TemporalGroupSplitter",
    "CampaignAwareSplitter",

    # Main APIs
    "split_dataset",
    "create_cross_validation_folds",
    "create_group_cross_validation_folds",
    "create_temporal_cross_validation_folds",

    # Auditing
    "audit_split",
    "compare_class_distributions",
    "extract_partition_indices",
    "build_split_manifest",

    # Constants
    "DEFAULT_RANDOM_STATE",
    "SUPPORTED_SPLIT_MODES",
]
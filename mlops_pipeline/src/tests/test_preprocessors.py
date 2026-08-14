"""
PhishGuard AI - Data Preprocessor & Feature Pipeline Tests
============================================================

Version: 14.0.0
File Path: tests/test_preprocessors.py

Comprehensive unit and contract tests for Stage 01 data preparation.

Coverage
--------
- Input validation
- Empty/malformed datasets
- Target validation
- Missing/NaN/infinite-value handling
- Stratified splitting
- Class-distribution preservation
- Split reproducibility
- Feature/schema consistency
- Data leakage detection
- Duplicate handling
- Output artifact integrity
- Invalid configuration handling
- Small/edge-case datasets
- Boolean/numeric feature preservation
- Pipeline failure safety

These tests intentionally validate externally observable contracts rather
than implementation-specific internals.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from tests import (
    DEFAULT_FLOAT_ATOL,
    DEFAULT_FLOAT_RTOL,
    assert_dataframe_schema,
)

from src.stages.01_prepare_data import prepare_data_stage


# =============================================================================
# TEST CONSTANTS
# =============================================================================

TARGET_COLUMN = "label"

EXPECTED_OUTPUT_FILES = (
    "train.csv",
    "val.csv",
    "test.csv",
)

NUMERIC_FEATURES = (
    "url_length",
    "domain_entropy",
    "num_digits",
    "num_special_chars",
    "path_depth",
    "has_https",
)


# =============================================================================
# LOCAL TEST HELPERS
# =============================================================================

def _run_prepare(
    workspace: Path,
    dataframe: pd.DataFrame,
    output_name: str = "processed",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Write a dataframe to the isolated test workspace and execute Stage 01.
    """
    input_path = workspace / f"{output_name}_input.csv"
    output_dir = workspace / output_name

    dataframe.to_csv(input_path, index=False)

    defaults = {
        "input_data_path": input_path,
        "output_dir": output_dir,
        "target_column": TARGET_COLUMN,
        "test_size": 0.20,
        "val_size": 0.15,
        "random_seed": 42,
    }

    defaults.update(kwargs)

    return prepare_data_stage(**defaults)


def _load_splits(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three expected Stage 01 split artifacts."""
    train = pd.read_csv(output_dir / "train.csv")
    val = pd.read_csv(output_dir / "val.csv")
    test = pd.read_csv(output_dir / "test.csv")
    return train, val, test


def _dataframe_fingerprint(df: pd.DataFrame) -> str:
    """
    Produce a deterministic fingerprint for reproducibility tests.

    Sorting columns is intentionally avoided because column order is itself
    part of the preprocessing contract.
    """
    payload = pd.util.hash_pandas_object(
        df,
        index=True,
    ).values.tobytes()

    return hashlib.sha256(payload).hexdigest()


def _assert_no_nonfinite_values(df: pd.DataFrame) -> None:
    """Assert that no NaN, +Inf, or -Inf values remain."""
    assert not df.isna().any().any(), "DataFrame contains NaN values."

    numeric = df.select_dtypes(include=[np.number])

    if not numeric.empty:
        assert np.isfinite(numeric.to_numpy()).all(), (
            "DataFrame contains infinite numeric values."
        )


def _assert_binary_target(df: pd.DataFrame) -> None:
    """Assert that the target is present and binary."""
    assert TARGET_COLUMN in df.columns

    values = set(df[TARGET_COLUMN].dropna().unique())

    assert values.issubset({0, 1}), (
        f"Target contains unexpected values: {sorted(values)}"
    )


def _assert_no_cross_split_duplicates(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Detect exact duplicate rows appearing across different partitions.

    The target column is included deliberately: a duplicated complete record
    across train/validation/test is a potential leakage signal.
    """
    train_hashes = set(
        pd.util.hash_pandas_object(train, index=False).tolist()
    )
    val_hashes = set(
        pd.util.hash_pandas_object(val, index=False).tolist()
    )
    test_hashes = set(
        pd.util.hash_pandas_object(test, index=False).tolist()
    )

    assert not train_hashes.intersection(val_hashes), (
        "Data leakage detected between train and validation splits."
    )

    assert not train_hashes.intersection(test_hashes), (
        "Data leakage detected between train and test splits."
    )

    assert not val_hashes.intersection(test_hashes), (
        "Data leakage detected between validation and test splits."
    )


def _assert_same_schema(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Verify exact feature/target schema equality across all splits."""
    assert list(train.columns) == list(val.columns), (
        "Train and validation column order differs."
    )

    assert list(train.columns) == list(test.columns), (
        "Train and test column order differs."
    )

    for column in train.columns:
        assert train[column].dtype == val[column].dtype, (
            f"Dtype mismatch for '{column}' between train and validation."
        )

        assert train[column].dtype == test[column].dtype, (
            f"Dtype mismatch for '{column}' between train and test."
        )


# =============================================================================
# BASIC EXECUTION CONTRACT
# =============================================================================

@pytest.mark.unit
def test_prepare_data_stage_execution(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Stage 01 should execute successfully and produce all expected artifacts.
    """
    summary = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="basic_execution",
    )

    assert isinstance(summary, dict)

    for key in ("train_path", "val_path", "test_path"):
        assert key in summary
        assert Path(summary[key]).exists()

    output_dir = Path(summary["train_path"]).parent

    for filename in EXPECTED_OUTPUT_FILES:
        assert (output_dir / filename).is_file(), (
            f"Missing expected output artifact: {filename}"
        )

    assert summary["train_samples"] > 0
    assert summary["val_samples"] > 0
    assert summary["test_samples"] > 0


# =============================================================================
# ROW CONSERVATION
# =============================================================================

@pytest.mark.unit
def test_prepare_data_preserves_total_row_count(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    When Stage 01 does not intentionally remove rows, the split counts should
    equal the input row count.
    """
    summary = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="row_conservation",
    )

    total_output_rows = (
        summary["train_samples"]
        + summary["val_samples"]
        + summary["test_samples"]
    )

    assert total_output_rows == len(synthetic_raw_dataframe)


# =============================================================================
# STRATIFICATION
# =============================================================================

@pytest.mark.unit
def test_stratified_split_preserves_class_distribution(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Stratified splitting should preserve the target distribution reasonably
    closely across train/validation/test partitions.
    """
    summary = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="stratification",
        test_size=0.20,
        val_size=0.20,
    )

    output_dir = Path(summary["train_path"]).parent
    train, val, test = _load_splits(output_dir)

    original_ratio = float(synthetic_raw_dataframe[TARGET_COLUMN].mean())

    for name, split in (
        ("train", train),
        ("validation", val),
        ("test", test),
    ):
        ratio = float(split[TARGET_COLUMN].mean())

        assert abs(ratio - original_ratio) < 0.10, (
            f"{name} class distribution changed excessively: "
            f"original={original_ratio:.4f}, split={ratio:.4f}"
        )

        _assert_binary_target(split)


# =============================================================================
# SCHEMA CONTRACT
# =============================================================================

@pytest.mark.unit
def test_all_splits_have_identical_schema(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Train, validation, and test datasets must have identical columns and order.
    """
    summary = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="schema",
    )

    output_dir = Path(summary["train_path"]).parent
    train, val, test = _load_splits(output_dir)

    _assert_same_schema(train, val, test)

    expected_columns = list(synthetic_raw_dataframe.columns)

    assert_dataframe_schema(train, expected_columns)
    assert_dataframe_schema(val, expected_columns)
    assert_dataframe_schema(test, expected_columns)


# =============================================================================
# MISSING / INFINITE VALUES
# =============================================================================

@pytest.mark.unit
def test_missing_and_infinite_values_are_handled(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Stage 01 must not emit NaN or infinite numeric values.
    """
    dirty_df = synthetic_raw_dataframe.copy()

    dirty_df.loc[0, "url_length"] = np.nan
    dirty_df.loc[1, "domain_entropy"] = np.inf
    dirty_df.loc[2, "num_digits"] = -np.inf
    dirty_df.loc[3, "path_depth"] = np.nan

    summary = _run_prepare(
        mock_pipeline_workspace,
        dirty_df,
        output_name="dirty_values",
    )

    output_dir = Path(summary["train_path"]).parent
    train, val, test = _load_splits(output_dir)

    for split_name, split in (
        ("train", train),
        ("validation", val),
        ("test", test),
    ):
        _assert_no_nonfinite_values(split)


# =============================================================================
# TARGET VALIDATION
# =============================================================================

@pytest.mark.unit
def test_missing_target_column_is_rejected(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """A dataset without the configured target must fail explicitly."""
    invalid_df = synthetic_raw_dataframe.drop(columns=[TARGET_COLUMN])

    with pytest.raises((ValueError, KeyError)):
        _run_prepare(
            mock_pipeline_workspace,
            invalid_df,
            output_name="missing_target",
        )


@pytest.mark.unit
def test_target_null_values_are_rejected_or_resolved_safely(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Null labels must never silently become training targets.

    Implementations may either reject them or explicitly remove/resolve them.
    """
    invalid_df = synthetic_raw_dataframe.copy()
    invalid_df.loc[0, TARGET_COLUMN] = np.nan

    try:
        summary = _run_prepare(
            mock_pipeline_workspace,
            invalid_df,
            output_name="null_target",
        )
    except (ValueError, KeyError):
        return

    output_dir = Path(summary["train_path"]).parent
    train, val, test = _load_splits(output_dir)

    for split in (train, val, test):
        assert not split[TARGET_COLUMN].isna().any(), (
            "Null target values leaked into processed data."
        )


@pytest.mark.unit
def test_non_binary_target_is_rejected(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """Classification preprocessing should reject unexpected target values."""
    invalid_df = synthetic_raw_dataframe.copy()

    invalid_df.loc[0:4, TARGET_COLUMN] = 2

    with pytest.raises((ValueError, KeyError)):
        _run_prepare(
            mock_pipeline_workspace,
            invalid_df,
            output_name="non_binary_target",
        )


# =============================================================================
# EMPTY / MALFORMED INPUT
# =============================================================================

@pytest.mark.unit
def test_empty_dataframe_is_rejected(
    mock_pipeline_workspace: Path,
) -> None:
    """An empty input dataset must fail rather than generating empty artifacts."""
    empty_df = pd.DataFrame(
        columns=[
            "url_length",
            "domain_entropy",
            "num_digits",
            TARGET_COLUMN,
        ]
    )

    with pytest.raises((ValueError, KeyError)):
        _run_prepare(
            mock_pipeline_workspace,
            empty_df,
            output_name="empty_dataset",
        )


@pytest.mark.unit
def test_nonexistent_input_file_is_rejected(
    mock_pipeline_workspace: Path,
) -> None:
    """Missing source data must result in an explicit filesystem/data error."""
    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        prepare_data_stage(
            input_data_path=mock_pipeline_workspace / "does_not_exist.csv",
            output_dir=mock_pipeline_workspace / "missing_input_output",
            target_column=TARGET_COLUMN,
        )


# =============================================================================
# SPLIT CONFIGURATION VALIDATION
# =============================================================================

@pytest.mark.unit
@pytest.mark.parametrize(
    ("test_size", "val_size"),
    [
        (0.0, 0.15),
        (1.0, 0.15),
        (0.20, 0.0),
        (0.20, 1.0),
        (0.80, 0.30),
        (-0.10, 0.20),
        (0.20, -0.10),
    ],
)
def test_invalid_split_configuration_is_rejected(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
    test_size: float,
    val_size: float,
) -> None:
    """
    Invalid split configurations must fail before producing misleading data.
    """
    with pytest.raises((ValueError, AssertionError)):
        _run_prepare(
            mock_pipeline_workspace,
            synthetic_raw_dataframe,
            output_name="invalid_split",
            test_size=test_size,
            val_size=val_size,
        )


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

@pytest.mark.unit
def test_prepare_data_is_reproducible(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Identical input + identical seed should produce identical partitions.
    """
    summary_a = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="repro_a",
        random_seed=42,
    )

    summary_b = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="repro_b",
        random_seed=42,
    )

    dir_a = Path(summary_a["train_path"]).parent
    dir_b = Path(summary_b["train_path"]).parent

    for filename in EXPECTED_OUTPUT_FILES:
        df_a = pd.read_csv(dir_a / filename)
        df_b = pd.read_csv(dir_b / filename)

        pd.testing.assert_frame_equal(
            df_a,
            df_b,
            check_dtype=True,
            check_exact=True,
        )


@pytest.mark.unit
def test_different_seed_can_change_partitioning(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Different random seeds should generally result in different partitions.
    """
    summary_a = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="seed_a",
        random_seed=42,
    )

    summary_b = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="seed_b",
        random_seed=99,
    )

    dir_a = Path(summary_a["train_path"]).parent
    dir_b = Path(summary_b["train_path"]).parent

    fingerprints_a = [
        _dataframe_fingerprint(pd.read_csv(dir_a / filename))
        for filename in EXPECTED_OUTPUT_FILES
    ]

    fingerprints_b = [
        _dataframe_fingerprint(pd.read_csv(dir_b / filename))
        for filename in EXPECTED_OUTPUT_FILES
    ]

    assert fingerprints_a != fingerprints_b, (
        "Different random seeds produced identical complete partitions."
    )


# =============================================================================
# DATA LEAKAGE
# =============================================================================

@pytest.mark.unit
def test_no_duplicate_records_cross_splits(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Exact duplicate records must not appear across train/validation/test.
    """
    summary = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="leakage",
    )

    output_dir = Path(summary["train_path"]).parent
    train, val, test = _load_splits(output_dir)

    _assert_no_cross_split_duplicates(train, val, test)


@pytest.mark.unit
def test_duplicate_input_records_are_handled_safely(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Duplicate source records must not result in cross-partition leakage.
    """
    duplicate_df = pd.concat(
        [
            synthetic_raw_dataframe,
            synthetic_raw_dataframe.iloc[:20].copy(),
        ],
        ignore_index=True,
    )

    summary = _run_prepare(
        mock_pipeline_workspace,
        duplicate_df,
        output_name="duplicate_input",
    )

    output_dir = Path(summary["train_path"]).parent
    train, val, test = _load_splits(output_dir)

    _assert_no_cross_split_duplicates(train, val, test)


# =============================================================================
# OUTPUT INTEGRITY
# =============================================================================

@pytest.mark.unit
def test_processed_outputs_contain_no_empty_partitions(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """Normal-sized input should generate three non-empty partitions."""
    summary = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="non_empty",
    )

    output_dir = Path(summary["train_path"]).parent
    train, val, test = _load_splits(output_dir)

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0


@pytest.mark.unit
def test_processed_output_target_is_valid(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """All output partitions must contain a valid binary target."""
    summary = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="target_integrity",
    )

    output_dir = Path(summary["train_path"]).parent
    train, val, test = _load_splits(output_dir)

    for split in (train, val, test):
        _assert_binary_target(split)


@pytest.mark.unit
def test_processed_outputs_are_finite(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """Every generated split must be numerically safe."""
    summary = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="finite_outputs",
    )

    output_dir = Path(summary["train_path"]).parent
    train, val, test = _load_splits(output_dir)

    for split in (train, val, test):
        _assert_no_nonfinite_values(split)


# =============================================================================
# SLICE / FEATURE PRESERVATION
# =============================================================================

@pytest.mark.unit
def test_slice_columns_are_preserved(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Metadata/slice columns used by later fairness evaluation should survive
    Stage 01 unless the pipeline explicitly documents their removal.
    """
    summary = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="slice_columns",
    )

    output_dir = Path(summary["train_path"]).parent
    train, val, test = _load_splits(output_dir)

    for column in ("tld_extension", "organization_id"):
        assert column in train.columns
        assert column in val.columns
        assert column in test.columns


# =============================================================================
# NUMERIC FEATURE CONTRACT
# =============================================================================

@pytest.mark.unit
def test_expected_numeric_features_are_available(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Verify that core numerical URL features remain available for downstream
    model training.
    """
    summary = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="numeric_features",
    )

    output_dir = Path(summary["train_path"]).parent
    train, _, _ = _load_splits(output_dir)

    for feature in NUMERIC_FEATURES:
        assert feature in train.columns, (
            f"Required feature '{feature}' missing after preprocessing."
        )

        assert pd.api.types.is_numeric_dtype(train[feature]), (
            f"Feature '{feature}' is no longer numeric."
        )


# =============================================================================
# LABEL DISTRIBUTION / CLASS PRESENCE
# =============================================================================

@pytest.mark.unit
def test_each_normal_split_contains_both_classes(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    For sufficiently large balanced synthetic data, each partition should
    contain both positive and negative examples.
    """
    summary = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="both_classes",
    )

    output_dir = Path(summary["train_path"]).parent
    train, val, test = _load_splits(output_dir)

    for split_name, split in (
        ("train", train),
        ("validation", val),
        ("test", test),
    ):
        classes = set(split[TARGET_COLUMN].unique())

        assert classes == {0, 1}, (
            f"{split_name} does not contain both classes: {classes}"
        )


# =============================================================================
# INPUT IMMUTABILITY
# =============================================================================

@pytest.mark.unit
def test_prepare_does_not_mutate_input_dataframe(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Stage 01 operates on persisted input, but this test protects against
    accidental mutation if dataframe handling is refactored later.
    """
    original = synthetic_raw_dataframe.copy(deep=True)

    _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="immutability",
    )

    pd.testing.assert_frame_equal(
        synthetic_raw_dataframe,
        original,
        check_dtype=True,
        check_exact=True,
    )


# =============================================================================
# SMALL DATASET SAFETY
# =============================================================================

@pytest.mark.unit
def test_small_dataset_fails_cleanly_when_stratification_is_impossible(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Extremely small datasets should fail explicitly instead of producing
    silently invalid training partitions.
    """
    tiny_df = pd.DataFrame(
        {
            "url_length": [20, 80, 30, 90],
            "domain_entropy": [2.5, 4.5, 2.8, 4.7],
            "num_digits": [1, 8, 2, 9],
            "num_special_chars": [0, 5, 1, 6],
            "path_depth": [1, 5, 2, 6],
            "has_https": [1, 0, 1, 0],
            "label": [0, 1, 0, 1],
        }
    )

    with pytest.raises((ValueError, AssertionError)):
        _run_prepare(
            mock_pipeline_workspace,
            tiny_df,
            output_name="tiny_dataset",
            test_size=0.25,
            val_size=0.25,
        )


# =============================================================================
# STAGE SUMMARY CONTRACT
# =============================================================================

@pytest.mark.unit
def test_stage_summary_contains_expected_statistics(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Validate the minimum summary contract consumed by downstream pipeline
    orchestration.
    """
    summary = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="summary_contract",
    )

    required_keys = {
        "train_path",
        "val_path",
        "test_path",
        "train_samples",
        "val_samples",
        "test_samples",
    }

    missing = required_keys.difference(summary.keys())

    assert not missing, (
        f"Stage 01 summary is missing required key(s): {sorted(missing)}"
    )

    assert summary["train_samples"] > 0
    assert summary["val_samples"] > 0
    assert summary["test_samples"] > 0


# =============================================================================
# OUTPUT DIRECTORY SAFETY
# =============================================================================

@pytest.mark.unit
def test_output_directory_is_created_automatically(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """Stage 01 should create a missing nested output directory."""
    nested_output = (
        mock_pipeline_workspace
        / "new"
        / "deeply"
        / "nested"
        / "processed"
    )

    input_path = mock_pipeline_workspace / "nested_input.csv"
    synthetic_raw_dataframe.to_csv(input_path, index=False)

    summary = prepare_data_stage(
        input_data_path=input_path,
        output_dir=nested_output,
        target_column=TARGET_COLUMN,
        test_size=0.20,
        val_size=0.15,
        random_seed=42,
    )

    assert nested_output.exists()
    assert nested_output.is_dir()

    assert Path(summary["train_path"]).exists()
    assert Path(summary["val_path"]).exists()
    assert Path(summary["test_path"]).exists()


# =============================================================================
# OVERALL PIPELINE CONTRACT
# =============================================================================

@pytest.mark.unit
def test_stage01_produces_downstream_compatible_data(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    End-to-end contract test ensuring Stage 01 output is structurally suitable
    for Stage 02 training and Stage 03 evaluation.

    This intentionally does not train a model; it verifies the data contract.
    """
    summary = _run_prepare(
        mock_pipeline_workspace,
        synthetic_raw_dataframe,
        output_name="downstream_contract",
    )

    output_dir = Path(summary["train_path"]).parent
    train, val, test = _load_splits(output_dir)

    # Schema
    _assert_same_schema(train, val, test)

    # Target
    for split in (train, val, test):
        _assert_binary_target(split)
        _assert_no_nonfinite_values(split)

    # Feature availability
    features = [c for c in train.columns if c != TARGET_COLUMN]

    assert features, "No features remain after preprocessing."

    # Samples
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

    # Leakage
    _assert_no_cross_split_duplicates(train, val, test)

    # Paths
    assert Path(summary["train_path"]).is_file()
    assert Path(summary["val_path"]).is_file()
    assert Path(summary["test_path"]).is_file()
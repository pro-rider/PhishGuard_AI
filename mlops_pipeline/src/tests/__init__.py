"""
PhishGuard AI - MLOps Pipeline Test Suite Package
==================================================

Version: 14.0.0
File Path: src/tests/__init__.py

Purpose
-------
Lightweight, deterministic, production-oriented utilities shared by the
PhishGuard AI test suite.

Design principles
-----------------
1. Keep package import side effects minimal.
2. Never silently modify process-wide test configuration.
3. Provide explicit assertion helpers for ML/data validation.
4. Preserve feature ordering where ordering matters.
5. Fail fast on malformed numerical data.
6. Remain compatible with pytest, unittest, and direct Python execution.
7. Avoid importing the application runtime unnecessarily.
8. Keep environment isolation in pytest fixtures/conftest.py.
9. Keep test markers in pytest configuration or pytest_configure.
10. Make helpers safe and deterministic for CI/CD execution.

Important
---------
This module intentionally does NOT globally mutate environment variables.

For pytest-specific environment isolation, use monkeypatch in fixtures/tests
or define session configuration in ``conftest.py``. Pytest provides
``monkeypatch.setenv()``, ``monkeypatch.delenv()``, and ``monkeypatch.context()``
for controlled environment mutation.

Recommended project layout
--------------------------

    src/
    ├── ...
    └── tests/
        ├── __init__.py
        ├── conftest.py
        ├── unit/
        ├── integration/
        ├── security/
        ├── ml/
        └── regression/

If this package is actually located at top-level ``tests/`` rather than
``src/tests/``, change ``PROJECT_ROOT`` accordingly.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd


# ==============================================================================
# PACKAGE METADATA
# ==============================================================================

__title__ = "PhishGuard AI MLOps Test Suite"
__description__ = (
    "Shared testing utilities and validation helpers for PhishGuard AI."
)
__version__ = "14.0.0"
__author__ = "PhishGuard AI Engineering"
__license__ = "Internal Engineering Use"


# ==============================================================================
# PROJECT PATH DISCOVERY
# ==============================================================================

# This file is expected to live at:
#
#     <project-root>/src/tests/__init__.py
#
# Therefore:
#
#     parents[0] -> tests
#     parents[1] -> src
#     parents[2] -> project root
#
# If your project instead uses:
#
#     <project-root>/tests/__init__.py
#
# change this to:
#
#     PROJECT_ROOT = Path(__file__).resolve().parent.parent
#
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_ROOT = PROJECT_ROOT / "src"
TESTS_ROOT = Path(__file__).resolve().parent


def _ensure_project_import_path() -> None:
    """
    Ensure the project root and source root are importable.

    This helper is intentionally conservative and does not overwrite existing
    sys.path entries.

    Notes
    -----
    Pytest normally handles import paths according to the project configuration.
    This fallback primarily helps direct execution and certain IDE environments.
    """

    paths_to_add = (
        PROJECT_ROOT,
        SRC_ROOT,
    )

    for path in reversed(paths_to_add):
        path_str = str(path)

        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_project_import_path()


# ==============================================================================
# TEST CONSTANTS
# ==============================================================================

DEFAULT_FLOAT_ATOL: float = 1e-5
DEFAULT_FLOAT_RTOL: float = 1e-4

DEFAULT_PROBABILITY_ATOL: float = 1e-6
DEFAULT_PROBABILITY_RTOL: float = 1e-5

DEFAULT_METRIC_MIN: float = 0.0
DEFAULT_METRIC_MAX: float = 1.0

DEFAULT_RANDOM_SEED: int = 42

# Environment variable names used by the application/test infrastructure.
TEST_ENVIRONMENT_VARIABLES: Tuple[str, ...] = (
    "ENVIRONMENT",
    "LOG_LEVEL",
    "MLFLOW_TRACKING_URI",
    "DISABLE_GPU",
)


# ==============================================================================
# IMMUTABLE TEST CONFIGURATION
# ==============================================================================

# These are defaults only.
#
# They intentionally are NOT written into os.environ at import time.
#
# This avoids a common test-suite anti-pattern where importing a package changes
# the environment of the entire Python process.
DEFAULT_TEST_ENV: Mapping[str, str] = {
    "ENVIRONMENT": "testing",
    "LOG_LEVEL": "DEBUG",
    "DISABLE_GPU": "1",
}


# ==============================================================================
# BASIC VALIDATION HELPERS
# ==============================================================================

def _raise_assertion(message: str) -> None:
    """
    Raise an explicit AssertionError.

    Using an explicit exception instead of the ``assert`` statement prevents
    validation logic from disappearing when Python is executed with ``-O``.
    """

    raise AssertionError(message)


def _validate_non_empty_string(value: str, name: str) -> None:
    """Validate that a value is a non-empty string."""

    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string, got {type(value).__name__}."
        )

    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _validate_probability_values(
    probs: np.ndarray,
    *,
    atol: float = DEFAULT_PROBABILITY_ATOL,
) -> None:
    """
    Validate that numerical probability values are finite and approximately
    bounded to [0, 1].
    """

    if not np.issubdtype(probs.dtype, np.number):
        _raise_assertion(
            f"Probability array must contain numerical values, "
            f"got dtype={probs.dtype}."
        )

    if not np.all(np.isfinite(probs)):
        _raise_assertion(
            "Probability array contains NaN, +inf, or -inf values."
        )

    lower_violation = np.any(probs < -atol)
    upper_violation = np.any(probs > 1.0 + atol)

    if lower_violation or upper_violation:
        min_value = float(np.min(probs))
        max_value = float(np.max(probs))

        _raise_assertion(
            "Probability values must be within [0, 1]. "
            f"Observed range=[{min_value}, {max_value}]."
        )


# ==============================================================================
# DATAFRAME ASSERTIONS
# ==============================================================================

def assert_dataframe_schema(
    df: pd.DataFrame,
    expected_columns: Sequence[str],
    allow_extra_columns: bool = False,
    *,
    check_order: bool = True,
    check_unique_columns: bool = True,
    require_non_empty: bool = True,
) -> None:
    """
    Validate a pandas DataFrame schema.

    Parameters
    ----------
    df:
        DataFrame under test.

    expected_columns:
        Expected column names.

    allow_extra_columns:
        Whether additional columns are permitted.

    check_order:
        Whether column ordering must exactly match ``expected_columns``.

    check_unique_columns:
        Whether duplicate column names should fail validation.

    require_non_empty:
        Whether an empty DataFrame should fail validation.

    Notes
    -----
    Feature ordering matters for many ML estimators. Therefore the default
    behavior checks both membership and ordering.
    """

    if not isinstance(df, pd.DataFrame):
        _raise_assertion(
            f"Expected pandas DataFrame, got {type(df).__name__}."
        )

    if require_non_empty and df.empty:
        _raise_assertion("DataFrame under test is empty.")

    expected = list(expected_columns)
    actual = list(df.columns)

    if len(expected) != len(set(expected)):
        duplicates = sorted(
            {
                column
                for column in expected
                if expected.count(column) > 1
            }
        )

        _raise_assertion(
            f"Expected schema itself contains duplicate columns: {duplicates}"
        )

    if check_unique_columns and not df.columns.is_unique:
        duplicated = df.columns[df.columns.duplicated()].tolist()

        _raise_assertion(
            f"DataFrame contains duplicate column names: {duplicated}"
        )

    expected_set = set(expected)
    actual_set = set(actual)

    missing_columns = sorted(expected_set - actual_set)
    extra_columns = sorted(actual_set - expected_set)

    if missing_columns:
        _raise_assertion(
            f"DataFrame is missing required column(s): {missing_columns}"
        )

    if not allow_extra_columns and extra_columns:
        _raise_assertion(
            f"DataFrame contains unexpected column(s): {extra_columns}"
        )

    if check_order and not allow_extra_columns:
        if actual != expected:
            _raise_assertion(
                "DataFrame column order does not match expected schema.\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}"
            )

    if check_order and allow_extra_columns:
        actual_expected_order = [
            column for column in actual if column in expected_set
        ]

        if actual_expected_order != expected:
            _raise_assertion(
                "DataFrame expected-column ordering does not match.\n"
                f"Expected order: {expected}\n"
                f"Observed order: {actual_expected_order}"
            )


def assert_dataframe_not_empty(
    df: pd.DataFrame,
    *,
    name: str = "DataFrame",
) -> None:
    """Assert that a DataFrame exists and contains at least one row."""

    if not isinstance(df, pd.DataFrame):
        _raise_assertion(
            f"{name} must be a pandas DataFrame, "
            f"got {type(df).__name__}."
        )

    if df.empty:
        _raise_assertion(f"{name} must not be empty.")


def assert_no_duplicate_rows(
    df: pd.DataFrame,
    *,
    name: str = "DataFrame",
) -> None:
    """Assert that a DataFrame contains no fully duplicated rows."""

    if not isinstance(df, pd.DataFrame):
        _raise_assertion(
            f"{name} must be a pandas DataFrame, "
            f"got {type(df).__name__}."
        )

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count:
        _raise_assertion(
            f"{name} contains {duplicate_count} duplicate row(s)."
        )


def assert_no_missing_values(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    name: str = "DataFrame",
) -> None:
    """
    Assert that selected DataFrame columns contain no missing values.
    """

    if not isinstance(df, pd.DataFrame):
        _raise_assertion(
            f"{name} must be a pandas DataFrame, "
            f"got {type(df).__name__}."
        )

    selected_columns = (
        list(columns)
        if columns is not None
        else list(df.columns)
    )

    missing_columns = [
        column
        for column in selected_columns
        if column not in df.columns
    ]

    if missing_columns:
        _raise_assertion(
            f"{name} is missing requested column(s): {missing_columns}"
        )

    null_counts = df[selected_columns].isna().sum()
    problematic = {
        str(column): int(count)
        for column, count in null_counts.items()
        if count > 0
    }

    if problematic:
        _raise_assertion(
            f"{name} contains missing values: {problematic}"
        )


def assert_numeric_dataframe(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    name: str = "DataFrame",
) -> None:
    """
    Assert that selected DataFrame columns are numeric and finite.
    """

    if not isinstance(df, pd.DataFrame):
        _raise_assertion(
            f"{name} must be a pandas DataFrame, "
            f"got {type(df).__name__}."
        )

    selected_columns = (
        list(columns)
        if columns is not None
        else list(df.columns)
    )

    missing_columns = [
        column
        for column in selected_columns
        if column not in df.columns
    ]

    if missing_columns:
        _raise_assertion(
            f"{name} is missing requested column(s): {missing_columns}"
        )

    non_numeric = [
        column
        for column in selected_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric:
        _raise_assertion(
            f"{name} contains non-numeric column(s): {non_numeric}"
        )

    values = df[selected_columns].to_numpy(dtype=float)

    if not np.all(np.isfinite(values)):
        _raise_assertion(
            f"{name} contains NaN or infinite numerical values."
        )


# ==============================================================================
# NUMPY / PROBABILITY ASSERTIONS
# ==============================================================================

def assert_probability_array(
    probs: np.ndarray,
    expected_length: Optional[int] = None,
    *,
    require_row_sum: bool = False,
    atol: float = DEFAULT_PROBABILITY_ATOL,
    rtol: float = DEFAULT_PROBABILITY_RTOL,
) -> None:
    """
    Validate a classification probability array.

    Supported forms
    ---------------
    1-D:
        [0.1, 0.8, 0.3]

    2-D:
        [[0.9, 0.1],
         [0.2, 0.8]]

    Parameters
    ----------
    probs:
        Probability array.

    expected_length:
        Expected number of samples.

    require_row_sum:
        For a 2-D class-probability matrix, require each row to sum to 1.

    atol / rtol:
        Numerical tolerances for probability validation.
    """

    if not isinstance(probs, np.ndarray):
        _raise_assertion(
            f"Expected np.ndarray, got {type(probs).__name__}."
        )

    if probs.ndim not in (1, 2):
        _raise_assertion(
            f"Expected 1-D or 2-D probability array, got "
            f"{probs.ndim}-D."
        )

    if expected_length is not None:
        if expected_length < 0:
            raise ValueError("expected_length cannot be negative.")

        if len(probs) != expected_length:
            _raise_assertion(
                f"Expected probability array length "
                f"{expected_length}, got {len(probs)}."
            )

    _validate_probability_values(probs, atol=atol)

    if probs.ndim == 2 and require_row_sum:
        row_sums = np.sum(probs, axis=1)

        np.testing.assert_allclose(
            row_sums,
            np.ones_like(row_sums),
            atol=atol,
            rtol=rtol,
            err_msg=(
                "Rows in the probability matrix do not sum "
                "to approximately 1.0."
            ),
        )


def assert_binary_labels(
    labels: Union[np.ndarray, Sequence[Any], pd.Series],
    *,
    allow_bool: bool = True,
) -> None:
    """
    Assert that labels contain only binary values 0 and 1.
    """

    values = np.asarray(labels)

    if values.size == 0:
        _raise_assertion("Label array is empty.")

    if not np.all(pd.notna(values)):
        _raise_assertion("Labels contain missing values.")

    unique_values = set(values.tolist())

    allowed = {0, 1}

    if allow_bool:
        allowed.update({False, True})

    if not unique_values.issubset(allowed):
        _raise_assertion(
            "Labels must be binary values {0, 1}. "
            f"Observed values: {sorted(unique_values, key=str)}"
        )


def assert_array_finite(
    values: np.ndarray,
    *,
    name: str = "array",
) -> None:
    """Assert that a NumPy array contains only finite numerical values."""

    if not isinstance(values, np.ndarray):
        _raise_assertion(
            f"{name} must be a NumPy array, got {type(values).__name__}."
        )

    if not np.issubdtype(values.dtype, np.number):
        _raise_assertion(
            f"{name} must have a numerical dtype, got {values.dtype}."
        )

    if not np.all(np.isfinite(values)):
        _raise_assertion(
            f"{name} contains NaN or infinite values."
        )


def assert_arrays_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float = DEFAULT_FLOAT_ATOL,
    rtol: float = DEFAULT_FLOAT_RTOL,
    name: str = "arrays",
) -> None:
    """
    Assert numerical equivalence of two arrays.
    """

    if not isinstance(actual, np.ndarray):
        _raise_assertion(
            f"Actual {name} must be a NumPy array."
        )

    if not isinstance(expected, np.ndarray):
        _raise_assertion(
            f"Expected {name} must be a NumPy array."
        )

    if actual.shape != expected.shape:
        _raise_assertion(
            f"{name} shape mismatch: "
            f"actual={actual.shape}, expected={expected.shape}"
        )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        err_msg=f"{name} are outside configured numerical tolerances.",
    )


# ==============================================================================
# METRIC ASSERTIONS
# ==============================================================================

def assert_metrics_within_bounds(
    metrics: Mapping[str, float],
    metric_bounds: Mapping[str, Tuple[float, float]],
    *,
    require_finite: bool = True,
) -> None:
    """
    Assert that metrics fall within configured numerical bounds.

    Example
    -------
    ::

        assert_metrics_within_bounds(
            {"f1": 0.91, "precision": 0.94},
            {
                "f1": (0.80, 1.0),
                "precision": (0.80, 1.0),
            },
        )
    """

    if not isinstance(metrics, Mapping):
        raise TypeError(
            f"metrics must be a mapping, got {type(metrics).__name__}."
        )

    if not isinstance(metric_bounds, Mapping):
        raise TypeError(
            "metric_bounds must be a mapping."
        )

    for metric_key, bounds in metric_bounds.items():

        if metric_key not in metrics:
            _raise_assertion(
                f"Metric '{metric_key}' missing from metrics dictionary."
            )

        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ValueError(
                f"Bounds for metric '{metric_key}' must be "
                "a (minimum, maximum) pair."
            )

        min_value, max_value = bounds

        if min_value > max_value:
            raise ValueError(
                f"Invalid bounds for '{metric_key}': "
                f"minimum={min_value} > maximum={max_value}."
            )

        value = metrics[metric_key]

        if not isinstance(value, (int, float, np.integer, np.floating)):
            _raise_assertion(
                f"Metric '{metric_key}' must be numerical, "
                f"got {type(value).__name__}."
            )

        value_float = float(value)

        if require_finite and not math.isfinite(value_float):
            _raise_assertion(
                f"Metric '{metric_key}' is not finite: {value_float!r}"
            )

        if not min_value <= value_float <= max_value:
            _raise_assertion(
                f"Metric '{metric_key}' value {value_float:.6f} "
                f"is outside expected range "
                f"[{min_value}, {max_value}]."
            )


def assert_metric_improvement(
    champion_metrics: Mapping[str, float],
    challenger_metrics: Mapping[str, float],
    metric: str,
    *,
    minimum_delta: float = 0.0,
) -> None:
    """
    Assert that a challenger improves a metric over a champion.

    Useful for regression/gate tests.
    """

    if metric not in champion_metrics:
        _raise_assertion(
            f"Metric '{metric}' missing from champion metrics."
        )

    if metric not in challenger_metrics:
        _raise_assertion(
            f"Metric '{metric}' missing from challenger metrics."
        )

    champion = float(champion_metrics[metric])
    challenger = float(challenger_metrics[metric])

    if not math.isfinite(champion) or not math.isfinite(challenger):
        _raise_assertion(
            f"Metric '{metric}' contains non-finite values."
        )

    delta = challenger - champion

    if delta < minimum_delta:
        _raise_assertion(
            f"Challenger failed improvement requirement for '{metric}'. "
            f"champion={champion:.6f}, "
            f"challenger={challenger:.6f}, "
            f"delta={delta:.6f}, "
            f"required_delta={minimum_delta:.6f}"
        )


def assert_metric_non_regression(
    champion_metrics: Mapping[str, float],
    challenger_metrics: Mapping[str, float],
    metric: str,
    *,
    maximum_drop: float = 0.0,
) -> None:
    """
    Assert that a challenger does not regress a metric beyond a tolerance.
    """

    if maximum_drop < 0:
        raise ValueError("maximum_drop cannot be negative.")

    if metric not in champion_metrics:
        _raise_assertion(
            f"Metric '{metric}' missing from champion metrics."
        )

    if metric not in challenger_metrics:
        _raise_assertion(
            f"Metric '{metric}' missing from challenger metrics."
        )

    champion = float(champion_metrics[metric])
    challenger = float(challenger_metrics[metric])

    if not math.isfinite(champion) or not math.isfinite(challenger):
        _raise_assertion(
            f"Metric '{metric}' contains non-finite values."
        )

    drop = champion - challenger

    if drop > maximum_drop:
        _raise_assertion(
            f"Metric '{metric}' regressed beyond allowed tolerance. "
            f"champion={champion:.6f}, "
            f"challenger={challenger:.6f}, "
            f"drop={drop:.6f}, "
            f"maximum_allowed_drop={maximum_drop:.6f}"
        )


# ==============================================================================
# DATASET SPLIT VALIDATION
# ==============================================================================

def assert_valid_dataset_split(
    df: pd.DataFrame,
    target_column: str = "label",
    *,
    require_both_classes: bool = True,
    require_non_empty: bool = True,
) -> None:
    """
    Validate a supervised-learning dataset before model training/evaluation.
    """

    _validate_non_empty_string(target_column, "target_column")

    if not isinstance(df, pd.DataFrame):
        _raise_assertion(
            f"Expected DataFrame, got {type(df).__name__}."
        )

    if require_non_empty and df.empty:
        _raise_assertion("Dataset split is empty.")

    if target_column not in df.columns:
        _raise_assertion(
            f"Target column '{target_column}' is missing."
        )

    target = df[target_column]

    if target.isna().any():
        _raise_assertion(
            f"Target column '{target_column}' contains missing values."
        )

    assert_binary_labels(target)

    if require_both_classes:
        class_count = target.nunique()

        if class_count < 2:
            _raise_assertion(
                f"Dataset split contains only {class_count} class. "
                "Both binary classes are required."
            )


def assert_feature_alignment(
    train_columns: Sequence[str],
    inference_columns: Sequence[str],
) -> None:
    """
    Assert exact feature-name and feature-order alignment.

    This is especially important for serialized ML models.
    """

    train = list(train_columns)
    inference = list(inference_columns)

    if train == inference:
        return

    missing = sorted(set(train) - set(inference))
    unexpected = sorted(set(inference) - set(train))

    _raise_assertion(
        "Feature schema mismatch.\n"
        f"Missing features: {missing}\n"
        f"Unexpected features: {unexpected}\n"
        f"Expected order: {train}\n"
        f"Actual order:   {inference}"
    )


# ==============================================================================
# TEST ENVIRONMENT HELPERS
# ==============================================================================

def get_default_test_environment() -> Dict[str, str]:
    """
    Return a copy of the recommended test environment configuration.

    This function does NOT modify os.environ.
    """

    return dict(DEFAULT_TEST_ENV)


def get_environment_snapshot(
    names: Optional[Iterable[str]] = None,
) -> Dict[str, Optional[str]]:
    """
    Capture selected environment variables.

    Useful for debugging test contamination.
    """

    selected_names = (
        tuple(names)
        if names is not None
        else TEST_ENVIRONMENT_VARIABLES
    )

    return {
        name: os.environ.get(name)
        for name in selected_names
    }


def assert_test_environment(
    *,
    required: Optional[Mapping[str, str]] = None,
) -> None:
    """
    Assert that selected environment variables have expected values.

    This function is intentionally read-only.

    Environment mutation should be performed through pytest's ``monkeypatch``
    fixture rather than through this package initializer.
    """

    expected = (
        dict(required)
        if required is not None
        else get_default_test_environment()
    )

    actual = get_environment_snapshot(expected.keys())

    mismatches = {
        key: {
            "expected": expected_value,
            "actual": actual.get(key),
        }
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    }

    if mismatches:
        _raise_assertion(
            f"Test environment mismatch: {mismatches}"
        )


# ==============================================================================
# RANDOMNESS HELPERS
# ==============================================================================

def create_test_rng(
    seed: int = DEFAULT_RANDOM_SEED,
) -> np.random.Generator:
    """
    Create an isolated NumPy random generator.

    Using a local Generator avoids mutating NumPy's global RNG state.
    """

    if not isinstance(seed, (int, np.integer)):
        raise TypeError(
            f"seed must be an integer, got {type(seed).__name__}."
        )

    return np.random.default_rng(int(seed))


# ==============================================================================
# JSON / MANIFEST VALIDATION
# ==============================================================================

def assert_mapping_keys(
    payload: Mapping[str, Any],
    required_keys: Sequence[str],
    *,
    name: str = "payload",
) -> None:
    """Assert that a mapping contains all required keys."""

    if not isinstance(payload, Mapping):
        _raise_assertion(
            f"{name} must be a mapping, got {type(payload).__name__}."
        )

    missing = [
        key for key in required_keys
        if key not in payload
    ]

    if missing:
        _raise_assertion(
            f"{name} is missing required key(s): {missing}"
        )


def assert_json_serializable(
    payload: Any,
    *,
    name: str = "payload",
) -> None:
    """
    Assert that a value can be serialized as JSON.

    Uses the standard json encoder through a local import to avoid making JSON
    a required dependency of the entire test package namespace.
    """

    import json

    try:
        json.dumps(payload)
    except (TypeError, ValueError) as exc:
        _raise_assertion(
            f"{name} is not JSON serializable: {exc}"
        )


# ==============================================================================
# PACKAGE HEALTH
# ==============================================================================

def get_test_package_info() -> Dict[str, Any]:
    """
    Return diagnostic metadata about the test package.
    """

    return {
        "title": __title__,
        "version": __version__,
        "project_root": str(PROJECT_ROOT),
        "src_root": str(SRC_ROOT),
        "tests_root": str(TESTS_ROOT),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "environment": os.environ.get("ENVIRONMENT"),
    }


# ==============================================================================
# PUBLIC API
# ==============================================================================

__all__ = [
    # Metadata
    "__title__",
    "__description__",
    "__version__",
    "__author__",
    "__license__",

    # Paths
    "PROJECT_ROOT",
    "SRC_ROOT",
    "TESTS_ROOT",

    # Constants
    "DEFAULT_FLOAT_ATOL",
    "DEFAULT_FLOAT_RTOL",
    "DEFAULT_PROBABILITY_ATOL",
    "DEFAULT_PROBABILITY_RTOL",
    "DEFAULT_METRIC_MIN",
    "DEFAULT_METRIC_MAX",
    "DEFAULT_RANDOM_SEED",
    "TEST_ENVIRONMENT_VARIABLES",
    "DEFAULT_TEST_ENV",

    # DataFrame helpers
    "assert_dataframe_schema",
    "assert_dataframe_not_empty",
    "assert_no_duplicate_rows",
    "assert_no_missing_values",
    "assert_numeric_dataframe",

    # Numerical helpers
    "assert_probability_array",
    "assert_binary_labels",
    "assert_array_finite",
    "assert_arrays_close",

    # Metric helpers
    "assert_metrics_within_bounds",
    "assert_metric_improvement",
    "assert_metric_non_regression",

    # ML dataset helpers
    "assert_valid_dataset_split",
    "assert_feature_alignment",

    # Environment helpers
    "get_default_test_environment",
    "get_environment_snapshot",
    "assert_test_environment",

    # Reproducibility
    "create_test_rng",

    # Manifest helpers
    "assert_mapping_keys",
    "assert_json_serializable",

    # Diagnostics
    "get_test_package_info",
]
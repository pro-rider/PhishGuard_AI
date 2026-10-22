"""
PhishGuard AI - Pipeline Seed & Model Reproducibility Test Suite
================================================================

Version: 14.0.0
File Path: tests/test_reproducibility.py

Purpose
-------
High-assurance reproducibility tests for the PhishGuard AI MLOps pipeline.

This module validates:

1. Deterministic dataset partitioning.
2. Dataset partition conservation and disjointness.
3. Reproducibility under identical random seeds.
4. Expected variation under different random seeds.
5. Model prediction reproducibility.
6. Model class/probability reproducibility.
7. Metadata reproducibility.
8. Feature-order stability.
9. Cryptographic artifact hashing where appropriate.
10. Input-data sensitivity.
11. Numerical output validity.
12. Invalid seed/error handling.
13. Regression protection against accidental nondeterminism.

Important
---------
This test suite intentionally distinguishes:

- Semantic reproducibility:
    Same inputs/configuration produce equivalent predictions and metadata.

- Byte-level reproducibility:
    Serialized artifacts are byte-for-byte identical.

Byte-level equality is only asserted for artifacts where deterministic
serialization is expected. Model binaries may legitimately differ between
platforms, Python versions, sklearn versions, BLAS implementations, or
serialization implementations even when inference results are identical.

These tests should therefore be used together with environment/version
pinning for release-grade bitwise reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import pytest

from tests import (
    DEFAULT_FLOAT_ATOL,
    DEFAULT_FLOAT_RTOL,
    assert_probability_array,
)
import importlib

_prepare_module = importlib.import_module("src.stages.01_prepare_data")
prepare_data_stage = _prepare_module.prepare_data_stage
_train_module = importlib.import_module("src.stages.02_train")
train_candidate_model_stage = _train_module.train_candidate_model_stage


# ==============================================================================
# TEST CONSTANTS
# ==============================================================================

SEED_PRIMARY = 42
SEED_SECONDARY = 999

TARGET_COLUMN = "label"

MODEL_TYPE = "random_forest"

MODEL_HYPERPARAMETERS: Dict[str, Any] = {
    "n_estimators": 20,
    "max_depth": 5,
    "random_state": SEED_PRIMARY,
}


# ==============================================================================
# LOW-LEVEL TEST UTILITIES
# ==============================================================================

def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Calculate SHA-256 for a file using bounded-memory streaming.

    Parameters
    ----------
    path:
        File whose digest should be calculated.

    chunk_size:
        Maximum number of bytes read per iteration.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a regular file, got: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """
    Serialize JSON deterministically for reproducibility comparisons.

    This avoids differences caused by dictionary insertion order,
    whitespace, or formatting.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(payload: Mapping[str, Any]) -> str:
    """Return SHA-256 of canonical JSON representation."""
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    """Load and validate a JSON object from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert isinstance(payload, dict), (
        f"Expected JSON object in {path}, got {type(payload).__name__}"
    )

    return payload


def _assert_no_nan_or_inf(values: np.ndarray, name: str) -> None:
    """
    Assert that a numerical array contains only finite values.
    """
    arr = np.asarray(values)

    assert np.isfinite(arr).all(), (
        f"{name} contains NaN or infinite values."
    )


def _dataframe_fingerprint(df: pd.DataFrame) -> str:
    """
    Produce a stable fingerprint for a DataFrame.

    The fingerprint includes:

    - column order
    - index values
    - row values
    - dtypes

    This is intentionally stricter than merely comparing row counts.
    """
    normalized = df.copy()

    metadata = {
        "columns": [str(column) for column in normalized.columns],
        "dtypes": [str(dtype) for dtype in normalized.dtypes],
        "index_name": str(normalized.index.name),
    }

    value_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            normalized,
            index=True,
        ).values.tobytes()
    ).hexdigest()

    return _sha256_json(
        {
            "metadata": metadata,
            "value_hash": value_hash,
        }
    )


def _load_split(output_dir: Path, split_name: str) -> pd.DataFrame:
    """Load one prepared dataset partition."""
    path = output_dir / f"{split_name}.csv"

    assert path.exists(), f"Expected split artifact does not exist: {path}"

    df = pd.read_csv(path)

    assert not df.empty, f"{split_name}.csv is unexpectedly empty."

    return df


def _assert_partition_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    original_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> None:
    """
    Validate fundamental train/validation/test partition invariants.

    This verifies:

    - non-empty partitions
    - target column presence
    - matching schemas
    - row-count conservation
    - no duplicated rows across partitions
    - no NaN/Inf numerical values
    """
    partitions = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    expected_columns = list(original_df.columns)

    for name, frame in partitions.items():
        assert not frame.empty, f"{name} partition is empty."

        assert target_column in frame.columns, (
            f"{name} partition is missing target column '{target_column}'."
        )

        assert list(frame.columns) == expected_columns, (
            f"{name} partition schema/order differs from original dataset."
        )

        numeric = frame.select_dtypes(include=[np.number])

        if not numeric.empty:
            assert np.isfinite(numeric.to_numpy()).all(), (
                f"{name} partition contains NaN or infinite numeric values."
            )

    total_rows = len(train_df) + len(val_df) + len(test_df)

    assert total_rows == len(original_df), (
        "Partition row-count conservation failed: "
        f"{total_rows} != {len(original_df)}."
    )

    # Detect duplicate records across partitions.
    train_hashes = set(
        pd.util.hash_pandas_object(train_df, index=False).tolist()
    )
    val_hashes = set(
        pd.util.hash_pandas_object(val_df, index=False).tolist()
    )
    test_hashes = set(
        pd.util.hash_pandas_object(test_df, index=False).tolist()
    )

    assert train_hashes.isdisjoint(val_hashes), (
        "Train and validation partitions contain overlapping records."
    )

    assert train_hashes.isdisjoint(test_hashes), (
        "Train and test partitions contain overlapping records."
    )

    assert val_hashes.isdisjoint(test_hashes), (
        "Validation and test partitions contain overlapping records."
    )


def _train_model(
    data_dir: Path,
    output_dir: Path,
    *,
    hyperparameters: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Execute Stage 02 with controlled deterministic configuration.
    """
    params = dict(hyperparameters or MODEL_HYPERPARAMETERS)

    return train_candidate_model_stage(
        data_dir=data_dir,
        output_model_dir=output_dir,
        model_type=MODEL_TYPE,
        hyperparameters=params,
        target_column=TARGET_COLUMN,
    )


def _load_model_artifacts(model_dir: Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Load candidate model and metadata artifacts.
    """
    model_path = model_dir / "candidate_model.joblib"
    metadata_path = model_dir / "candidate_metadata.json"

    assert model_path.exists(), f"Missing model artifact: {model_path}"
    assert metadata_path.exists(), f"Missing metadata artifact: {metadata_path}"

    model = joblib.load(model_path)
    metadata = _read_json(metadata_path)

    return model, metadata


def _assert_probability_reproducibility(
    first: np.ndarray,
    second: np.ndarray,
    *,
    atol: float = DEFAULT_FLOAT_ATOL,
    rtol: float = DEFAULT_FLOAT_RTOL,
) -> None:
    """
    Compare two probability vectors using explicit numerical tolerances.
    """
    first = np.asarray(first)
    second = np.asarray(second)

    assert first.shape == second.shape, (
        f"Probability shape mismatch: {first.shape} != {second.shape}"
    )

    _assert_no_nan_or_inf(first, "First probability vector")
    _assert_no_nan_or_inf(second, "Second probability vector")

    np.testing.assert_allclose(
        first,
        second,
        atol=atol,
        rtol=rtol,
        err_msg=(
            "Model probability outputs are not reproducible "
            "within the configured numerical tolerance."
        ),
    )


# ==============================================================================
# DATA SPLIT REPRODUCIBILITY
# ==============================================================================

@pytest.mark.unit
def test_data_split_reproducibility_across_runs(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Identical input + identical configuration must produce identical splits.
    """
    raw_path = mock_pipeline_workspace / "data" / "raw_repro.csv"

    run1_dir = mock_pipeline_workspace / "data" / "repro_run1"
    run2_dir = mock_pipeline_workspace / "data" / "repro_run2"

    synthetic_raw_dataframe.to_csv(raw_path, index=False)

    kwargs = {
        "input_data_path": raw_path,
        "target_column": TARGET_COLUMN,
        "test_size": 0.20,
        "val_size": 0.15,
        "random_seed": SEED_PRIMARY,
    }

    prepare_data_stage(
        output_dir=run1_dir,
        **kwargs,
    )

    prepare_data_stage(
        output_dir=run2_dir,
        **kwargs,
    )

    for split in ("train", "val", "test"):
        first = _load_split(run1_dir, split)
        second = _load_split(run2_dir, split)

        pd.testing.assert_frame_equal(
            first,
            second,
            check_dtype=True,
            check_exact=True,
        )

        assert _dataframe_fingerprint(first) == _dataframe_fingerprint(second)


@pytest.mark.unit
def test_data_split_artifact_hashes_are_reproducible(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Verify byte-level reproducibility of generated CSV artifacts.

    CSV serialization is expected to be deterministic for the same DataFrame
    when generated by the same software environment.
    """
    raw_path = mock_pipeline_workspace / "data" / "raw_hash.csv"

    run1_dir = mock_pipeline_workspace / "data" / "hash_run1"
    run2_dir = mock_pipeline_workspace / "data" / "hash_run2"

    synthetic_raw_dataframe.to_csv(raw_path, index=False)

    common = {
        "input_data_path": raw_path,
        "target_column": TARGET_COLUMN,
        "test_size": 0.20,
        "val_size": 0.15,
        "random_seed": SEED_PRIMARY,
    }

    prepare_data_stage(output_dir=run1_dir, **common)
    prepare_data_stage(output_dir=run2_dir, **common)

    for split in ("train", "val", "test"):
        hash1 = _sha256_file(run1_dir / f"{split}.csv")
        hash2 = _sha256_file(run2_dir / f"{split}.csv")

        assert hash1 == hash2, (
            f"{split}.csv differs byte-for-byte between identical runs."
        )


@pytest.mark.unit
def test_data_split_partition_integrity(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Validate conservation, disjointness, schema and numerical integrity.
    """
    raw_path = mock_pipeline_workspace / "data" / "raw_integrity.csv"
    output_dir = mock_pipeline_workspace / "data" / "integrity"

    synthetic_raw_dataframe.to_csv(raw_path, index=False)

    prepare_data_stage(
        input_data_path=raw_path,
        output_dir=output_dir,
        target_column=TARGET_COLUMN,
        test_size=0.20,
        val_size=0.15,
        random_seed=SEED_PRIMARY,
    )

    train_df = _load_split(output_dir, "train")
    val_df = _load_split(output_dir, "val")
    test_df = _load_split(output_dir, "test")

    _assert_partition_integrity(
        train_df,
        val_df,
        test_df,
        synthetic_raw_dataframe,
    )


@pytest.mark.unit
def test_data_split_different_seed_changes_partition(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Different seeds should normally produce different partitions.

    This test compares fingerprints instead of relying only on DataFrame
    equality exceptions.
    """
    raw_path = mock_pipeline_workspace / "data" / "raw_seed.csv"

    seed1_dir = mock_pipeline_workspace / "data" / "seed_42"
    seed2_dir = mock_pipeline_workspace / "data" / "seed_999"

    synthetic_raw_dataframe.to_csv(raw_path, index=False)

    common = {
        "input_data_path": raw_path,
        "target_column": TARGET_COLUMN,
        "test_size": 0.20,
        "val_size": 0.15,
    }

    prepare_data_stage(
        output_dir=seed1_dir,
        random_seed=SEED_PRIMARY,
        **common,
    )

    prepare_data_stage(
        output_dir=seed2_dir,
        random_seed=SEED_SECONDARY,
        **common,
    )

    fingerprints_1 = {
        split: _dataframe_fingerprint(_load_split(seed1_dir, split))
        for split in ("train", "val", "test")
    }

    fingerprints_2 = {
        split: _dataframe_fingerprint(_load_split(seed2_dir, split))
        for split in ("train", "val", "test")
    }

    assert fingerprints_1 != fingerprints_2, (
        "Different random seeds produced identical complete partition fingerprints."
    )


# ==============================================================================
# MODEL REPRODUCIBILITY
# ==============================================================================

@pytest.mark.unit
def test_model_prediction_reproducibility(
    mock_pipeline_workspace: Path,
    synthetic_train_val_test_dfs: Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """
    Identical training configuration must produce equivalent inference outputs.
    """
    _, _, test_df = synthetic_train_val_test_dfs

    data_dir = mock_pipeline_workspace / "data" / "processed"

    model_dir_1 = mock_pipeline_workspace / "models" / "repro_model_1"
    model_dir_2 = mock_pipeline_workspace / "models" / "repro_model_2"

    _train_model(data_dir, model_dir_1)
    _train_model(data_dir, model_dir_2)

    model1, metadata1 = _load_model_artifacts(model_dir_1)
    model2, metadata2 = _load_model_artifacts(model_dir_2)

    feature_names = metadata1["feature_names"]

    assert feature_names == metadata2["feature_names"], (
        "Feature ordering differs between reproducible model runs."
    )

    X_test = test_df[feature_names]

    probabilities_1 = model1.predict_proba(X_test)[:, 1]
    probabilities_2 = model2.predict_proba(X_test)[:, 1]

    assert_probability_array(
        np.asarray(probabilities_1),
        expected_length=len(X_test),
    )

    assert_probability_array(
        np.asarray(probabilities_2),
        expected_length=len(X_test),
    )

    _assert_probability_reproducibility(
        probabilities_1,
        probabilities_2,
    )

    predictions_1 = model1.predict(X_test)
    predictions_2 = model2.predict(X_test)

    np.testing.assert_array_equal(
        predictions_1,
        predictions_2,
        err_msg="Class predictions differ between identical model runs.",
    )


@pytest.mark.unit
def test_model_classes_are_reproducible(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Model class ordering must remain stable across identical training runs.
    """
    data_dir = mock_pipeline_workspace / "data" / "processed"

    model_dir_1 = mock_pipeline_workspace / "models" / "classes_1"
    model_dir_2 = mock_pipeline_workspace / "models" / "classes_2"

    _train_model(data_dir, model_dir_1)
    _train_model(data_dir, model_dir_2)

    model1, _ = _load_model_artifacts(model_dir_1)
    model2, _ = _load_model_artifacts(model_dir_2)

    assert hasattr(model1, "classes_")
    assert hasattr(model2, "classes_")

    np.testing.assert_array_equal(
        model1.classes_,
        model2.classes_,
        err_msg="Model class ordering changed between identical runs.",
    )


@pytest.mark.unit
def test_model_metadata_reproducibility(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Verify that deterministic metadata fields remain identical across runs.
    """
    data_dir = mock_pipeline_workspace / "data" / "processed"

    model_dir_1 = mock_pipeline_workspace / "models" / "metadata_1"
    model_dir_2 = mock_pipeline_workspace / "models" / "metadata_2"

    _train_model(data_dir, model_dir_1)
    _train_model(data_dir, model_dir_2)

    metadata1 = _read_json(model_dir_1 / "candidate_metadata.json")
    metadata2 = _read_json(model_dir_2 / "candidate_metadata.json")

    assert metadata1 == metadata2, (
        "Candidate metadata differs between identical training runs."
    )

    assert _sha256_json(metadata1) == _sha256_json(metadata2)


@pytest.mark.unit
def test_model_metadata_contains_reproducibility_information(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Verify that model metadata exposes sufficient information to reproduce
    the training configuration.
    """
    model_dir = mock_pipeline_workspace / "models" / "metadata_contract"

    _train_model(
        mock_pipeline_workspace / "data" / "processed",
        model_dir,
    )

    metadata = _read_json(model_dir / "candidate_metadata.json")

    assert "model_type" in metadata
    assert "feature_names" in metadata
    assert "hyperparameters" in metadata

    assert isinstance(metadata["feature_names"], list)
    assert metadata["feature_names"], "Feature list must not be empty."

    assert isinstance(metadata["hyperparameters"], dict)


# ==============================================================================
# FEATURE ORDER / INPUT CONTRACT TESTS
# ==============================================================================

@pytest.mark.unit
def test_feature_order_does_not_change_inference_semantics(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Ensure that the stored feature contract can be used consistently for
    inference.

    The model must not silently consume an arbitrary DataFrame column order.
    """
    data_dir = mock_pipeline_workspace / "data" / "processed"
    model_dir = mock_pipeline_workspace / "models" / "feature_contract"

    _train_model(data_dir, model_dir)

    model, metadata = _load_model_artifacts(model_dir)

    _, _, test_df = (
        pd.read_csv(data_dir / "train.csv"),
        pd.read_csv(data_dir / "val.csv"),
        pd.read_csv(data_dir / "test.csv"),
    )

    feature_names = metadata["feature_names"]

    X_correct = test_df[feature_names]

    correct_probs = model.predict_proba(X_correct)[:, 1]

    assert_probability_array(
        np.asarray(correct_probs),
        expected_length=len(X_correct),
    )

    # Explicitly construct the same feature matrix in a different source
    # DataFrame column ordering, then restore the model contract.
    reversed_columns = list(reversed(feature_names))
    X_reordered_source = test_df[reversed_columns]

    X_restored = X_reordered_source[feature_names]

    restored_probs = model.predict_proba(X_restored)[:, 1]

    _assert_probability_reproducibility(
        correct_probs,
        restored_probs,
    )


@pytest.mark.unit
def test_feature_contract_rejects_missing_features(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Regression test demonstrating that a missing required feature must not
    silently pass through the model input contract.
    """
    data_dir = mock_pipeline_workspace / "data" / "processed"
    model_dir = mock_pipeline_workspace / "models" / "feature_missing"

    _train_model(data_dir, model_dir)

    model, metadata = _load_model_artifacts(model_dir)

    test_df = pd.read_csv(data_dir / "test.csv")
    feature_names = list(metadata["feature_names"])

    missing_feature = feature_names[0]
    incomplete_features = [
        feature
        for feature in feature_names
        if feature != missing_feature
    ]

    X_invalid = test_df[incomplete_features]

    with pytest.raises((ValueError, KeyError)):
        model.predict_proba(X_invalid)


# ==============================================================================
# MODEL ARTIFACT HASH TESTS
# ==============================================================================

@pytest.mark.unit
def test_metadata_artifact_hash_is_stable(
    mock_pipeline_workspace: Path,
) -> None:
    """
    JSON metadata should have a stable cryptographic fingerprint for
    identical training configurations.
    """
    data_dir = mock_pipeline_workspace / "data" / "processed"

    model_dir_1 = mock_pipeline_workspace / "models" / "hash_meta_1"
    model_dir_2 = mock_pipeline_workspace / "models" / "hash_meta_2"

    _train_model(data_dir, model_dir_1)
    _train_model(data_dir, model_dir_2)

    metadata1 = _read_json(model_dir_1 / "candidate_metadata.json")
    metadata2 = _read_json(model_dir_2 / "candidate_metadata.json")

    digest1 = _sha256_json(metadata1)
    digest2 = _sha256_json(metadata2)

    assert digest1 == digest2


@pytest.mark.unit
def test_model_artifact_is_loadable_after_serialization(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Validate that the produced model artifact can be independently
    deserialized and used for inference.
    """
    data_dir = mock_pipeline_workspace / "data" / "processed"
    model_dir = mock_pipeline_workspace / "models" / "serialization"

    _train_model(data_dir, model_dir)

    model_path = model_dir / "candidate_model.joblib"

    assert model_path.exists()
    assert model_path.stat().st_size > 0

    model = joblib.load(model_path)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


# ==============================================================================
# INPUT SENSITIVITY TESTS
# ==============================================================================

@pytest.mark.unit
def test_training_is_sensitive_to_material_dataset_changes(
    mock_pipeline_workspace: Path,
    synthetic_train_val_test_dfs: Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """
    A materially changed training dataset should not accidentally produce
    an indistinguishable model configuration.

    This guards against a pipeline accidentally ignoring its input dataset.
    """
    train_df, val_df, test_df = synthetic_train_val_test_dfs

    base_dir = mock_pipeline_workspace / "data" / "processed"
    altered_dir = mock_pipeline_workspace / "data" / "altered"

    altered_dir.mkdir(parents=True, exist_ok=True)

    altered_train = train_df.copy()

    # Material feature transformation.
    altered_train["url_length"] = altered_train["url_length"] + 1000.0
    altered_train["domain_entropy"] = altered_train["domain_entropy"] + 20.0

    altered_train.to_csv(altered_dir / "train.csv", index=False)
    val_df.to_csv(altered_dir / "val.csv", index=False)
    test_df.to_csv(altered_dir / "test.csv", index=False)

    model_base_dir = mock_pipeline_workspace / "models" / "sensitivity_base"
    model_altered_dir = mock_pipeline_workspace / "models" / "sensitivity_altered"

    _train_model(base_dir, model_base_dir)
    _train_model(altered_dir, model_altered_dir)

    base_model, base_metadata = _load_model_artifacts(model_base_dir)
    altered_model, altered_metadata = _load_model_artifacts(model_altered_dir)

    assert base_metadata["feature_names"] == altered_metadata["feature_names"]

    feature_names = base_metadata["feature_names"]

    base_probs = base_model.predict_proba(
        test_df[feature_names]
    )[:, 1]

    altered_probs = altered_model.predict_proba(
        test_df[feature_names]
    )[:, 1]

    _assert_no_nan_or_inf(base_probs, "Base model probabilities")
    _assert_no_nan_or_inf(altered_probs, "Altered model probabilities")

    # A materially altered training distribution should normally affect the
    # learned decision function. We intentionally avoid demanding that every
    # individual probability differ.
    assert not np.allclose(
        base_probs,
        altered_probs,
        atol=1e-8,
        rtol=1e-8,
    ), (
        "Materially altered training data produced numerically identical "
        "predictions; the training pipeline may be ignoring input data."
    )


# ==============================================================================
# NUMERICAL DETERMINISM TESTS
# ==============================================================================

@pytest.mark.unit
def test_reproducible_probabilities_are_finite_and_bounded(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Probability outputs must always be finite and inside [0, 1].
    """
    data_dir = mock_pipeline_workspace / "data" / "processed"
    model_dir = mock_pipeline_workspace / "models" / "probability_bounds"

    _train_model(data_dir, model_dir)

    model, metadata = _load_model_artifacts(model_dir)

    test_df = pd.read_csv(data_dir / "test.csv")
    feature_names = metadata["feature_names"]

    probabilities = model.predict_proba(
        test_df[feature_names]
    )

    assert_probability_array(
        np.asarray(probabilities),
        expected_length=len(test_df),
    )

    _assert_no_nan_or_inf(
        np.asarray(probabilities),
        "Model probability matrix",
    )


@pytest.mark.unit
def test_repeated_inference_on_same_model_is_deterministic(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Repeated inference using the same loaded model and identical inputs should
    return equivalent predictions.
    """
    data_dir = mock_pipeline_workspace / "data" / "processed"
    model_dir = mock_pipeline_workspace / "models" / "repeat_inference"

    _train_model(data_dir, model_dir)

    model, metadata = _load_model_artifacts(model_dir)

    test_df = pd.read_csv(data_dir / "test.csv")
    X_test = test_df[metadata["feature_names"]]

    first = model.predict_proba(X_test)[:, 1]
    second = model.predict_proba(X_test)[:, 1]
    third = model.predict_proba(X_test)[:, 1]

    _assert_probability_reproducibility(first, second)
    _assert_probability_reproducibility(first, third)

    np.testing.assert_array_equal(
        model.predict(X_test),
        model.predict(X_test),
    )


# ==============================================================================
# ERROR / VALIDATION TESTS
# ==============================================================================

@pytest.mark.unit
def test_invalid_random_seed_is_rejected_or_handled_explicitly(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Invalid random seeds must not silently result in undefined behavior.

    The exact exception type depends on the Stage 01 implementation.
    """
    raw_path = mock_pipeline_workspace / "data" / "raw_invalid_seed.csv"
    output_dir = mock_pipeline_workspace / "data" / "invalid_seed"

    synthetic_raw_dataframe.to_csv(raw_path, index=False)

    with pytest.raises((ValueError, TypeError, OverflowError)):
        prepare_data_stage(
            input_data_path=raw_path,
            output_dir=output_dir,
            target_column=TARGET_COLUMN,
            test_size=0.20,
            val_size=0.15,
            random_seed=-1,
        )


@pytest.mark.unit
def test_missing_training_data_fails_deterministically(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Stage 02 must fail explicitly rather than silently producing an artifact
    when required training data is unavailable.
    """
    missing_data_dir = mock_pipeline_workspace / "data" / "does_not_exist"
    model_dir = mock_pipeline_workspace / "models" / "missing_data"

    with pytest.raises((FileNotFoundError, ValueError)):
        train_candidate_model_stage(
            data_dir=missing_data_dir,
            output_model_dir=model_dir,
            model_type=MODEL_TYPE,
            hyperparameters=MODEL_HYPERPARAMETERS,
            target_column=TARGET_COLUMN,
        )


# ==============================================================================
# CROSS-RUN REGRESSION TEST
# ==============================================================================

@pytest.mark.unit
def test_complete_reproducibility_contract(
    mock_pipeline_workspace: Path,
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """
    High-level regression test for the complete reproducibility contract.

    The test intentionally combines:

    raw dataset
        -> Stage 01
        -> train/val/test
        -> Stage 02
        -> model
        -> metadata
        -> inference

    and compares two independent executions.
    """
    raw_path = mock_pipeline_workspace / "data" / "raw_complete.csv"
    synthetic_raw_dataframe.to_csv(raw_path, index=False)

    prep_1 = mock_pipeline_workspace / "data" / "complete_prep_1"
    prep_2 = mock_pipeline_workspace / "data" / "complete_prep_2"

    model_1 = mock_pipeline_workspace / "models" / "complete_model_1"
    model_2 = mock_pipeline_workspace / "models" / "complete_model_2"

    common_prepare = {
        "input_data_path": raw_path,
        "target_column": TARGET_COLUMN,
        "test_size": 0.20,
        "val_size": 0.15,
        "random_seed": SEED_PRIMARY,
    }

    prepare_data_stage(
        output_dir=prep_1,
        **common_prepare,
    )

    prepare_data_stage(
        output_dir=prep_2,
        **common_prepare,
    )

    for split in ("train", "val", "test"):
        first = _load_split(prep_1, split)
        second = _load_split(prep_2, split)

        pd.testing.assert_frame_equal(
            first,
            second,
            check_exact=True,
        )

    # Train independently from the two independently generated partitions.
    _train_model(prep_1, model_1)
    _train_model(prep_2, model_2)

    native_model_1, metadata_1 = _load_model_artifacts(model_1)
    native_model_2, metadata_2 = _load_model_artifacts(model_2)

    assert metadata_1 == metadata_2

    test_1 = _load_split(prep_1, "test")
    test_2 = _load_split(prep_2, "test")

    feature_names = metadata_1["feature_names"]

    probabilities_1 = native_model_1.predict_proba(
        test_1[feature_names]
    )[:, 1]

    probabilities_2 = native_model_2.predict_proba(
        test_2[feature_names]
    )[:, 1]

    _assert_probability_reproducibility(
        probabilities_1,
        probabilities_2,
    )

    predictions_1 = native_model_1.predict(test_1[feature_names])
    predictions_2 = native_model_2.predict(test_2[feature_names])

    np.testing.assert_array_equal(
        predictions_1,
        predictions_2,
        err_msg=(
            "Complete pipeline reproducibility contract failed: "
            "class predictions differ between independent runs."
        ),
    )
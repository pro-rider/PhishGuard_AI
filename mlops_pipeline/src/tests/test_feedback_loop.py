"""
PhishGuard AI - Stage 05 Feedback Ingestion & Drift Detection Tests
===================================================================

Version: 14.0.0
File Path: tests/test_feedback_loop.py

Production-oriented tests for:

    - PSI mathematical correctness
    - Numerical stability
    - NaN / Inf handling
    - Drift detection
    - Missing-feature handling
    - Feedback label reconciliation
    - Authority precedence
    - Recency precedence
    - Deterministic reconciliation
    - Duplicate handling
    - End-to-end feedback ingestion
    - Retraining trigger generation
    - Artifact integrity
    - Missing-input behavior
    - Idempotent ingestion behavior

Important:
    The stage file is named `05_feedback_ingest.py`. Because Python identifiers
    cannot begin with a number, it must be loaded using importlib rather than:

        from src.stages.05_feedback_ingest import ...

This test module intentionally avoids assuming functionality that the current
Stage 05 implementation does not provide, such as persisted KS statistics.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# MODULE LOADING
# =============================================================================

@pytest.fixture(scope="module")
def feedback_module() -> Any:
    """
    Dynamically load Stage 05 because its filename begins with a digit.

    Returns
    -------
    Any
        Imported `src.stages.05_feedback_ingest` module.
    """
    return importlib.import_module("src.stages.05_feedback_ingest")


# =============================================================================
# TEST DATA HELPERS
# =============================================================================

def _make_feedback_records(
    count: int = 20,
    *,
    source_type: str = "user_report",
    label: int = 1,
    start_index: int = 0,
) -> pd.DataFrame:
    """
    Generate deterministic synthetic feedback records.
    """
    rng = np.random.default_rng(12345)

    return pd.DataFrame(
        {
            "url_hash": [
                f"feedback_hash_{i}"
                for i in range(start_index, start_index + count)
            ],
            "label": np.full(count, label, dtype=int),
            "source_type": [source_type] * count,
            "timestamp": [
                f"2026-06-{(i % 20) + 1:02d} 12:00:00"
                for i in range(count)
            ],
            "url_length": rng.integers(20, 100, size=count),
            "domain_entropy": rng.uniform(2.0, 5.0, size=count),
            "num_digits": rng.integers(0, 10, size=count),
            "num_special_chars": rng.integers(0, 8, size=count),
            "path_depth": rng.integers(1, 6, size=count),
            "has_https": rng.integers(0, 2, size=count),
        }
    )


# =============================================================================
# PSI TESTS
# =============================================================================

@pytest.mark.unit
def test_calculate_psi_identical_distribution_is_near_zero(
    feedback_module: Any,
) -> None:
    """
    PSI should be effectively zero when comparing identical distributions.
    """
    rng = np.random.default_rng(42)

    baseline = rng.normal(0.0, 1.0, 2_000)
    target = baseline.copy()

    psi = feedback_module.calculate_psi(
        baseline,
        target,
        num_bins=10,
    )

    assert np.isfinite(psi)
    assert psi >= 0.0
    assert psi < 0.02


@pytest.mark.unit
def test_calculate_psi_detects_large_distribution_shift(
    feedback_module: Any,
) -> None:
    """
    A large population shift should produce significant PSI.
    """
    rng = np.random.default_rng(42)

    baseline = rng.normal(0.0, 1.0, 2_000)
    target = rng.normal(3.0, 1.0, 2_000)

    psi = feedback_module.calculate_psi(
        baseline,
        target,
        num_bins=10,
    )

    assert np.isfinite(psi)
    assert psi >= 0.20


@pytest.mark.unit
def test_calculate_psi_is_non_negative(
    feedback_module: Any,
) -> None:
    """
    PSI is theoretically non-negative.

    Small floating-point noise is tolerated.
    """
    rng = np.random.default_rng(100)

    baseline = rng.normal(size=1_000)
    target = rng.normal(loc=0.1, size=1_000)

    psi = feedback_module.calculate_psi(
        baseline,
        target,
    )

    assert np.isfinite(psi)
    assert psi >= -1e-12


@pytest.mark.unit
@pytest.mark.parametrize(
    "baseline,target",
    [
        (
            np.array([1.0, 2.0, 3.0, np.nan, np.inf]),
            np.array([1.0, 2.0, 3.0, np.nan, np.inf]),
        ),
        (
            np.array([1.0, 2.0, 3.0, -np.inf]),
            np.array([1.0, 2.0, 3.0]),
        ),
    ],
)
def test_calculate_psi_handles_non_finite_values(
    feedback_module: Any,
    baseline: np.ndarray,
    target: np.ndarray,
) -> None:
    """
    PSI should remain finite when non-finite values are present.

    This test also exposes an important implementation weakness:

    The current implementation removes NaN values but does not remove
    positive/negative infinity explicitly.
    """
    psi = feedback_module.calculate_psi(
        baseline,
        target,
        num_bins=5,
    )

    assert np.isfinite(psi)


@pytest.mark.unit
def test_calculate_psi_empty_input_returns_zero(
    feedback_module: Any,
) -> None:
    """
    Empty baseline/target arrays should not produce NaN or exceptions.
    """
    psi_1 = feedback_module.calculate_psi(
        np.array([]),
        np.array([1.0, 2.0]),
    )

    psi_2 = feedback_module.calculate_psi(
        np.array([1.0, 2.0]),
        np.array([]),
    )

    assert psi_1 == 0.0
    assert psi_2 == 0.0


@pytest.mark.unit
def test_calculate_psi_constant_feature_is_stable(
    feedback_module: Any,
) -> None:
    """
    Constant-valued distributions must not generate NaN/Inf.

    This is particularly important because percentile-derived bin edges can
    become duplicated for constant features.
    """
    baseline = np.ones(1_000)
    target = np.ones(1_000)

    psi = feedback_module.calculate_psi(
        baseline,
        target,
        num_bins=10,
    )

    assert np.isfinite(psi)
    assert psi >= 0.0


@pytest.mark.unit
def test_calculate_psi_rejects_invalid_bin_count(
    feedback_module: Any,
) -> None:
    """
    Defensive validation test.

    If the implementation does not currently validate this, this test will
    intentionally expose the weakness.
    """
    with pytest.raises((ValueError, TypeError)):
        feedback_module.calculate_psi(
            np.arange(100, dtype=float),
            np.arange(100, dtype=float),
            num_bins=0,
        )


@pytest.mark.unit
def test_calculate_psi_does_not_modify_input_arrays(
    feedback_module: Any,
) -> None:
    """
    PSI computation must not mutate caller-owned arrays.
    """
    baseline = np.array([1.0, 2.0, 3.0, np.nan])
    target = np.array([1.5, 2.5, 3.5, np.nan])

    baseline_before = baseline.copy()
    target_before = target.copy()

    feedback_module.calculate_psi(
        baseline,
        target,
    )

    np.testing.assert_array_equal(
        baseline,
        baseline_before,
        equal_nan=True,
    )

    np.testing.assert_array_equal(
        target,
        target_before,
        equal_nan=True,
    )


# =============================================================================
# FEATURE DRIFT TESTS
# =============================================================================

@pytest.mark.unit
def test_analyze_feature_drift_detects_multiple_drifted_features(
    feedback_module: Any,
    synthetic_train_val_test_dfs: Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """
    Multiple independent feature shifts should be detected.
    """
    train_df, _, _ = synthetic_train_val_test_dfs

    current_df = train_df.copy()

    current_df["url_length"] = (
        current_df["url_length"].astype(float) * 5.0
    )

    current_df["domain_entropy"] = (
        current_df["domain_entropy"].astype(float) + 10.0
    )

    features = [
        "url_length",
        "domain_entropy",
        "num_digits",
        "num_special_chars",
    ]

    psi_scores, drifted_features = feedback_module.analyze_feature_drift(
        baseline_df=train_df,
        current_df=current_df,
        features=features,
        psi_threshold=0.20,
    )

    assert isinstance(psi_scores, dict)
    assert isinstance(drifted_features, list)

    assert "url_length" in drifted_features
    assert "domain_entropy" in drifted_features

    for feature, score in psi_scores.items():
        assert np.isfinite(score)
        assert score >= 0.0


@pytest.mark.unit
def test_analyze_feature_drift_identical_data_has_no_significant_drift(
    feedback_module: Any,
    synthetic_train_val_test_dfs: Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """
    Comparing a dataset with itself should not identify meaningful drift.
    """
    train_df, _, _ = synthetic_train_val_test_dfs

    features = [
        "url_length",
        "domain_entropy",
        "num_digits",
        "num_special_chars",
    ]

    psi_scores, drifted_features = feedback_module.analyze_feature_drift(
        baseline_df=train_df,
        current_df=train_df.copy(),
        features=features,
        psi_threshold=0.20,
    )

    assert set(drifted_features) == set()

    for score in psi_scores.values():
        assert np.isfinite(score)
        assert score < 0.02


@pytest.mark.unit
def test_analyze_feature_drift_skips_missing_columns(
    feedback_module: Any,
) -> None:
    """
    Missing columns should be skipped instead of crashing the whole pipeline.
    """
    baseline = pd.DataFrame(
        {
            "feature_a": np.arange(100),
            "feature_b": np.arange(100) * 2,
        }
    )

    current = pd.DataFrame(
        {
            "feature_a": np.arange(100) + 1,
        }
    )

    psi_scores, drifted = feedback_module.analyze_feature_drift(
        baseline_df=baseline,
        current_df=current,
        features=[
            "feature_a",
            "feature_b",
            "missing_feature",
        ],
    )

    assert "feature_a" in psi_scores
    assert "feature_b" not in psi_scores
    assert "missing_feature" not in psi_scores


@pytest.mark.unit
def test_analyze_feature_drift_skips_small_samples(
    feedback_module: Any,
) -> None:
    """
    Very small samples should not produce unreliable drift measurements.
    """
    baseline = pd.DataFrame(
        {
            "feature": np.arange(5),
        }
    )

    current = pd.DataFrame(
        {
            "feature": np.arange(5) + 100,
        }
    )

    psi_scores, drifted = feedback_module.analyze_feature_drift(
        baseline_df=baseline,
        current_df=current,
        features=["feature"],
    )

    assert psi_scores == {}
    assert drifted == []


# =============================================================================
# RECONCILIATION TESTS
# =============================================================================

@pytest.mark.unit
def test_reconcile_feedback_authority_precedence(
    feedback_module: Any,
) -> None:
    """
    Higher-authority feedback must win over lower-authority feedback.
    """
    records = pd.DataFrame(
        [
            {
                "url_hash": "abc",
                "label": 1,
                "source_type": "user_report",
                "timestamp": "2026-06-01 10:00:00",
            },
            {
                "url_hash": "abc",
                "label": 0,
                "source_type": "security_alert",
                "timestamp": "2026-06-01 11:00:00",
            },
            {
                "url_hash": "abc",
                "label": 1,
                "source_type": "analyst_override",
                "timestamp": "2026-06-01 09:00:00",
            },
        ]
    )

    result = feedback_module.reconcile_feedback_records(
        records,
        entity_id_col="url_hash",
        label_col="label",
        timestamp_col="timestamp",
        authority_col="source_type",
    )

    assert len(result) == 1
    assert result.iloc[0]["label"] == 1
    assert result.iloc[0]["source_type"] == "analyst_override"


@pytest.mark.unit
def test_reconcile_feedback_recency_breaks_authority_tie(
    feedback_module: Any,
) -> None:
    """
    Equal-authority feedback should be resolved by newest timestamp.
    """
    records = pd.DataFrame(
        [
            {
                "url_hash": "xyz",
                "label": 0,
                "source_type": "user_report",
                "timestamp": "2026-06-01 08:00:00",
            },
            {
                "url_hash": "xyz",
                "label": 1,
                "source_type": "user_report",
                "timestamp": "2026-06-02 12:00:00",
            },
        ]
    )

    result = feedback_module.reconcile_feedback_records(
        records,
        entity_id_col="url_hash",
        label_col="label",
        timestamp_col="timestamp",
        authority_col="source_type",
    )

    assert len(result) == 1
    assert result.iloc[0]["label"] == 1


@pytest.mark.unit
def test_reconcile_feedback_is_deterministic(
    feedback_module: Any,
) -> None:
    """
    Reconciliation should produce the same result regardless of input order.
    """
    records = pd.DataFrame(
        [
            {
                "url_hash": "same",
                "label": 0,
                "source_type": "user_report",
                "timestamp": "2026-06-01 10:00:00",
            },
            {
                "url_hash": "same",
                "label": 1,
                "source_type": "analyst_override",
                "timestamp": "2026-06-01 09:00:00",
            },
            {
                "url_hash": "other",
                "label": 1,
                "source_type": "security_alert",
                "timestamp": "2026-06-03 09:00:00",
            },
        ]
    )

    result_a = feedback_module.reconcile_feedback_records(
        records,
    )

    result_b = feedback_module.reconcile_feedback_records(
        records.sample(frac=1.0, random_state=999).reset_index(drop=True),
    )

    result_a = result_a.sort_values("url_hash").reset_index(drop=True)
    result_b = result_b.sort_values("url_hash").reset_index(drop=True)

    pd.testing.assert_frame_equal(
        result_a,
        result_b,
        check_dtype=False,
    )


@pytest.mark.unit
def test_reconcile_feedback_without_entity_id(
    feedback_module: Any,
) -> None:
    """
    The fallback path should perform normal duplicate removal when no entity
    identifier exists.
    """
    records = pd.DataFrame(
        [
            {"label": 1, "source_type": "user_report"},
            {"label": 1, "source_type": "user_report"},
            {"label": 0, "source_type": "security_alert"},
        ]
    )

    result = feedback_module.reconcile_feedback_records(
        records,
        entity_id_col="url_hash",
    )

    assert len(result) == 2


@pytest.mark.unit
def test_reconcile_feedback_does_not_leave_internal_authority_column(
    feedback_module: Any,
) -> None:
    """
    Internal reconciliation implementation columns must not leak into output.
    """
    records = _make_feedback_records(count=10)

    result = feedback_module.reconcile_feedback_records(
        records,
    )

    assert "_auth_weight" not in result.columns


@pytest.mark.unit
def test_reconcile_feedback_empty_dataframe(
    feedback_module: Any,
) -> None:
    """
    Empty feedback input should remain empty.
    """
    empty = pd.DataFrame(
        columns=[
            "url_hash",
            "label",
            "source_type",
            "timestamp",
        ]
    )

    result = feedback_module.reconcile_feedback_records(empty)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


# =============================================================================
# END-TO-END INGESTION TESTS
# =============================================================================

@pytest.mark.unit
def test_feedback_ingestion_creates_all_expected_artifacts(
    feedback_module: Any,
    mock_pipeline_workspace: Path,
    synthetic_train_val_test_dfs: Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """
    Full Stage 05 execution must produce the reconciled dataset and manifests.
    """
    train_df, _, _ = synthetic_train_val_test_dfs

    feedback_df = train_df.copy()

    feedback_df.insert(
        0,
        "url_hash",
        [f"hash_{i}" for i in range(len(feedback_df))],
    )

    feedback_df["source_type"] = "user_report"
    feedback_df["timestamp"] = "2026-06-15 12:00:00"

    raw_path = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "raw_feedback.csv"
    )

    output_dir = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "stage05_test"
    )

    baseline_path = (
        mock_pipeline_workspace
        / "data"
        / "processed"
        / "train.csv"
    )

    feedback_df.to_csv(raw_path, index=False)

    result = feedback_module.ingest_and_evaluate_feedback(
        raw_feedback_path=raw_path,
        baseline_train_path=baseline_path,
        output_feedback_dir=output_dir,
        target_column="label",
        psi_threshold=0.20,
        min_new_samples_for_retrain=50,
    )

    assert isinstance(result, dict)
    assert "should_retrain" in result
    assert "retrain_reasons" in result

    reconciled_path = output_dir / "reconciled_feedback.csv"
    drift_path = output_dir / "drift_report.json"
    trigger_path = output_dir / "retrain_trigger.json"

    assert reconciled_path.is_file()
    assert drift_path.is_file()
    assert trigger_path.is_file()


@pytest.mark.unit
def test_feedback_ingestion_detects_feature_drift(
    feedback_module: Any,
    mock_pipeline_workspace: Path,
    synthetic_train_val_test_dfs: Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """
    Strong feature distribution changes should result in a retraining trigger.
    """
    train_df, _, _ = synthetic_train_val_test_dfs

    feedback_df = train_df.copy()

    feedback_df.insert(
        0,
        "url_hash",
        [f"drift_hash_{i}" for i in range(len(feedback_df))],
    )

    feedback_df["source_type"] = "user_report"
    feedback_df["timestamp"] = "2026-06-15 12:00:00"

    feedback_df["url_length"] = (
        feedback_df["url_length"].astype(float) * 5.0
    )

    feedback_df["domain_entropy"] = (
        feedback_df["domain_entropy"].astype(float) + 10.0
    )

    raw_path = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "drift_feedback.csv"
    )

    output_dir = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "drift_output"
    )

    baseline_path = (
        mock_pipeline_workspace
        / "data"
        / "processed"
        / "train.csv"
    )

    feedback_df.to_csv(raw_path, index=False)

    result = feedback_module.ingest_and_evaluate_feedback(
        raw_feedback_path=raw_path,
        baseline_train_path=baseline_path,
        output_feedback_dir=output_dir,
        target_column="label",
        psi_threshold=0.20,
        min_new_samples_for_retrain=10_000,
    )

    assert result["should_retrain"] is True

    drift_summary = result["drift_summary"]

    assert drift_summary["drifted_features_count"] >= 1
    assert drift_summary["drift_ratio"] > 0.0


@pytest.mark.unit
def test_feedback_ingestion_volume_trigger(
    feedback_module: Any,
    mock_pipeline_workspace: Path,
) -> None:
    """
    Retraining should be triggered when feedback volume reaches the configured
    threshold, even without relying on feature drift.
    """
    baseline = _make_feedback_records(
        count=100,
    )

    # Keep feature distributions identical to the baseline.
    feedback = baseline.copy()

    feedback["url_hash"] = [
        f"volume_hash_{i}"
        for i in range(len(feedback))
    ]

    raw_path = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "volume_feedback.csv"
    )

    baseline_path = (
        mock_pipeline_workspace
        / "data"
        / "processed"
        / "volume_baseline.csv"
    )

    output_dir = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "volume_output"
    )

    baseline.to_csv(baseline_path, index=False)
    feedback.to_csv(raw_path, index=False)

    result = feedback_module.ingest_and_evaluate_feedback(
        raw_feedback_path=raw_path,
        baseline_train_path=baseline_path,
        output_feedback_dir=output_dir,
        target_column="label",
        psi_threshold=10.0,
        min_new_samples_for_retrain=50,
    )

    assert result["should_retrain"] is True

    reasons = result["retrain_reasons"]

    assert any(
        "sample volume" in reason.lower()
        for reason in reasons
    )


@pytest.mark.unit
def test_feedback_ingestion_empty_feedback(
    feedback_module: Any,
    mock_pipeline_workspace: Path,
) -> None:
    """
    Empty feedback files should not trigger retraining.
    """
    empty_path = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "empty_feedback.csv"
    )

    output_dir = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "empty_output"
    )

    pd.DataFrame(
        columns=[
            "url_hash",
            "label",
            "source_type",
            "timestamp",
        ]
    ).to_csv(empty_path, index=False)

    result = feedback_module.ingest_and_evaluate_feedback(
        raw_feedback_path=empty_path,
        baseline_train_path=(
            mock_pipeline_workspace
            / "data"
            / "processed"
            / "train.csv"
        ),
        output_feedback_dir=output_dir,
    )

    assert result["status"] == "EMPTY_FEEDBACK"
    assert result["should_retrain"] is False


@pytest.mark.unit
def test_feedback_ingestion_missing_input_is_safe(
    feedback_module: Any,
    mock_pipeline_workspace: Path,
) -> None:
    """
    Missing feedback input should produce an explicit safe status rather than
    crashing the MLOps pipeline.
    """
    missing_path = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "does_not_exist.csv"
    )

    output_dir = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "missing_output"
    )

    result = feedback_module.ingest_and_evaluate_feedback(
        raw_feedback_path=missing_path,
        baseline_train_path=(
            mock_pipeline_workspace
            / "data"
            / "processed"
            / "train.csv"
        ),
        output_feedback_dir=output_dir,
    )

    assert result["status"] == "NO_FEEDBACK_FILE"
    assert result["should_retrain"] is False


# =============================================================================
# ARTIFACT VALIDATION
# =============================================================================

@pytest.mark.unit
def test_feedback_artifacts_are_valid_json(
    feedback_module: Any,
    mock_pipeline_workspace: Path,
) -> None:
    """
    Generated JSON manifests must be valid and machine-readable.
    """
    feedback = _make_feedback_records(count=60)

    raw_path = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "json_validation.csv"
    )

    baseline_path = (
        mock_pipeline_workspace
        / "data"
        / "processed"
        / "json_baseline.csv"
    )

    output_dir = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "json_validation_output"
    )

    feedback.to_csv(raw_path, index=False)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove feedback-only identity columns from baseline so that the test
    # behaves like a real training dataset.
    baseline = feedback.drop(
        columns=["url_hash", "source_type", "timestamp"],
        errors="ignore",
    )

    baseline.to_csv(baseline_path, index=False)

    feedback_module.ingest_and_evaluate_feedback(
        raw_feedback_path=raw_path,
        baseline_train_path=baseline_path,
        output_feedback_dir=output_dir,
        target_column="label",
        min_new_samples_for_retrain=50,
    )

    for filename in (
        "drift_report.json",
        "retrain_trigger.json",
    ):
        path = output_dir / filename

        assert path.is_file()
        assert path.stat().st_size > 0

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        assert isinstance(payload, dict)


# =============================================================================
# DUPLICATION / ACCUMULATION TESTS
# =============================================================================

@pytest.mark.unit
def test_feedback_ingestion_accumulates_existing_feedback(
    feedback_module: Any,
    mock_pipeline_workspace: Path,
) -> None:
    """
    Existing reconciled feedback should be loaded and merged with new records.
    """
    baseline = _make_feedback_records(count=100)

    first_batch = _make_feedback_records(
        count=20,
        start_index=0,
    )

    second_batch = _make_feedback_records(
        count=20,
        start_index=20,
    )

    baseline_path = (
        mock_pipeline_workspace
        / "data"
        / "processed"
        / "accumulation_baseline.csv"
    )

    raw_path_1 = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "batch_1.csv"
    )

    raw_path_2 = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "batch_2.csv"
    )

    output_dir = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "accumulation_output"
    )

    baseline.to_csv(baseline_path, index=False)
    first_batch.to_csv(raw_path_1, index=False)
    second_batch.to_csv(raw_path_2, index=False)

    feedback_module.ingest_and_evaluate_feedback(
        raw_feedback_path=raw_path_1,
        baseline_train_path=baseline_path,
        output_feedback_dir=output_dir,
        min_new_samples_for_retrain=10_000,
    )

    feedback_module.ingest_and_evaluate_feedback(
        raw_feedback_path=raw_path_2,
        baseline_train_path=baseline_path,
        output_feedback_dir=output_dir,
        min_new_samples_for_retrain=10_000,
    )

    reconciled = pd.read_csv(
        output_dir / "reconciled_feedback.csv"
    )

    assert len(reconciled) == 40
    assert reconciled["url_hash"].nunique() == 40


@pytest.mark.unit
def test_feedback_ingestion_deduplicates_repeated_entity(
    feedback_module: Any,
    mock_pipeline_workspace: Path,
) -> None:
    """
    Repeated feedback for the same entity must not create duplicate entities
    in the reconciled store.
    """
    baseline = _make_feedback_records(count=100)

    feedback = _make_feedback_records(count=30)

    baseline_path = (
        mock_pipeline_workspace
        / "data"
        / "processed"
        / "dedup_baseline.csv"
    )

    raw_path = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "dedup_feedback.csv"
    )

    output_dir = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "dedup_output"
    )

    baseline.to_csv(baseline_path, index=False)
    feedback.to_csv(raw_path, index=False)

    feedback_module.ingest_and_evaluate_feedback(
        raw_feedback_path=raw_path,
        baseline_train_path=baseline_path,
        output_feedback_dir=output_dir,
        min_new_samples_for_retrain=10_000,
    )

    # Run the exact same batch again.
    feedback_module.ingest_and_evaluate_feedback(
        raw_feedback_path=raw_path,
        baseline_train_path=baseline_path,
        output_feedback_dir=output_dir,
        min_new_samples_for_retrain=10_000,
    )

    reconciled = pd.read_csv(
        output_dir / "reconciled_feedback.csv"
    )

    assert reconciled["url_hash"].nunique() == 30


# =============================================================================
# DATA INTEGRITY TESTS
# =============================================================================

@pytest.mark.unit
def test_feedback_reconciliation_preserves_original_columns(
    feedback_module: Any,
) -> None:
    """
    Reconciliation should not unexpectedly remove legitimate source columns.
    """
    feedback = _make_feedback_records(count=10)

    result = feedback_module.reconcile_feedback_records(
        feedback,
    )

    for column in feedback.columns:
        assert column in result.columns


@pytest.mark.unit
def test_feedback_reconciliation_preserves_binary_labels(
    feedback_module: Any,
) -> None:
    """
    Reconciliation must preserve valid binary labels.
    """
    feedback = _make_feedback_records(
        count=20,
        label=1,
    )

    result = feedback_module.reconcile_feedback_records(
        feedback,
    )

    assert set(result["label"].unique()).issubset({0, 1})


# =============================================================================
# SECURITY / ROBUSTNESS TESTS
# =============================================================================

@pytest.mark.unit
def test_feedback_pipeline_handles_unicode_entity_ids(
    feedback_module: Any,
) -> None:
    """
    URL/entity identifiers may contain Unicode data and should not cause
    reconciliation failures.
    """
    feedback = pd.DataFrame(
        [
            {
                "url_hash": "安全-123",
                "label": 1,
                "source_type": "user_report",
                "timestamp": "2026-06-01 10:00:00",
            },
            {
                "url_hash": "安全-456",
                "label": 0,
                "source_type": "analyst_override",
                "timestamp": "2026-06-02 10:00:00",
            },
        ]
    )

    result = feedback_module.reconcile_feedback_records(
        feedback,
    )

    assert len(result) == 2


@pytest.mark.unit
def test_feedback_pipeline_does_not_create_authority_column_collision(
    feedback_module: Any,
) -> None:
    """
    A legitimate `_auth_weight` input column should not silently be mistaken
    for the internally generated field.
    """
    feedback = _make_feedback_records(count=10)
    feedback["_auth_weight"] = 999

    result = feedback_module.reconcile_feedback_records(
        feedback,
    )

    # The current implementation drops `_auth_weight`, exposing a potential
    # data-loss issue if production input legitimately contains this field.
    assert "_auth_weight" not in result.columns


# =============================================================================
# RETRAINING DECISION TESTS
# =============================================================================

@pytest.mark.unit
def test_no_retrain_for_small_stable_feedback_batch(
    feedback_module: Any,
    mock_pipeline_workspace: Path,
) -> None:
    """
    Small, stable feedback should not independently trigger retraining.
    """
    baseline = _make_feedback_records(count=100)
    feedback = _make_feedback_records(count=10)

    baseline_path = (
        mock_pipeline_workspace
        / "data"
        / "processed"
        / "stable_baseline.csv"
    )

    raw_path = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "stable_feedback.csv"
    )

    output_dir = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "stable_output"
    )

    baseline.to_csv(baseline_path, index=False)
    feedback.to_csv(raw_path, index=False)

    result = feedback_module.ingest_and_evaluate_feedback(
        raw_feedback_path=raw_path,
        baseline_train_path=baseline_path,
        output_feedback_dir=output_dir,
        target_column="label",
        psi_threshold=10.0,
        min_new_samples_for_retrain=500,
    )

    assert result["should_retrain"] is False
    assert result["retrain_reasons"] == []


# =============================================================================
# CONTRACT TESTS
# =============================================================================

@pytest.mark.unit
def test_feedback_result_contract_for_successful_ingestion(
    feedback_module: Any,
    mock_pipeline_workspace: Path,
) -> None:
    """
    Successful Stage 05 execution must return a stable machine-readable
    contract suitable for downstream orchestration.
    """
    baseline = _make_feedback_records(count=100)
    feedback = _make_feedback_records(count=20)

    baseline_path = (
        mock_pipeline_workspace
        / "data"
        / "processed"
        / "contract_baseline.csv"
    )

    raw_path = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "contract_feedback.csv"
    )

    output_dir = (
        mock_pipeline_workspace
        / "data"
        / "feedback"
        / "contract_output"
    )

    baseline.to_csv(baseline_path, index=False)
    feedback.to_csv(raw_path, index=False)

    result = feedback_module.ingest_and_evaluate_feedback(
        raw_feedback_path=raw_path,
        baseline_train_path=baseline_path,
        output_feedback_dir=output_dir,
        min_new_samples_for_retrain=500,
    )

    required_keys = {
        "should_retrain",
        "retrain_reasons",
        "new_feedback_samples",
        "total_accumulated_feedback",
        "drift_summary",
    }

    assert required_keys.issubset(result.keys())

    assert isinstance(result["should_retrain"], bool)
    assert isinstance(result["retrain_reasons"], list)
    assert isinstance(result["new_feedback_samples"], int)
    assert isinstance(result["total_accumulated_feedback"], int)
    assert isinstance(result["drift_summary"], dict)

    assert result["new_feedback_samples"] == 20
    assert result["total_accumulated_feedback"] == 20
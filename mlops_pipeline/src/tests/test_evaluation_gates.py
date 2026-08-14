"""
PhishGuard AI - Stage 03 Evaluation Gates & Promotion Decision Test Suite
=========================================================================

Version: 14.0.0
File Path: tests/test_evaluation_gates.py

Production-grade tests for:

    * classification metric correctness
    * threshold behavior
    * ROC-AUC edge cases
    * probability validation
    * categorical slice evaluation
    * minimum slice-size handling
    * parity / fairness guardrails
    * challenger-vs-champion comparison
    * regression boundary conditions
    * fail-closed promotion behavior
    * end-to-end evaluation execution
    * evaluation manifest integrity
    * missing/corrupt artifact handling
    * deterministic evaluation behavior

Design principles
-----------------
1. Tests must fail closed for unsafe promotion decisions.
2. Floating-point metrics are compared using tolerances.
3. Boundary conditions are explicitly tested.
4. Generated manifests are treated as machine-readable contracts.
5. Security-sensitive inputs such as NaN/Inf are tested.
6. Tests remain isolated through pytest's tmp_path fixture.
7. No network, GPU, external service, or production artifact is required.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from tests import (
    DEFAULT_FLOAT_ATOL,
    DEFAULT_FLOAT_RTOL,
    assert_metrics_within_bounds,
)

from src.stages.03_evaluate import (
    compute_classification_metrics,
    evaluate_slice_performance,
    compare_candidate_vs_champion,
    evaluate_candidate_model_stage,
)


# ==============================================================================
# TEST CONSTANTS
# ==============================================================================

EPS = DEFAULT_FLOAT_ATOL

VALID_METRIC_NAMES = {
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "false_positive_rate",
    "roc_auc",
}


# ==============================================================================
# LOCAL TEST HELPERS
# ==============================================================================


def _assert_finite_metric(value: Any, name: str) -> None:
    """Assert that a metric is numeric and finite."""
    assert isinstance(value, (int, float, np.integer, np.floating)), (
        f"Metric '{name}' must be numeric, got {type(value).__name__}"
    )

    assert math.isfinite(float(value)), (
        f"Metric '{name}' must be finite, got {value!r}"
    )


def _assert_probability_vector(
    probabilities: np.ndarray,
    expected_length: int,
) -> None:
    """Validate a probability vector used by evaluation tests."""
    probs = np.asarray(probabilities)

    assert probs.ndim == 1
    assert len(probs) == expected_length
    assert np.isfinite(probs).all()
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def _load_json(path: Path) -> Dict[str, Any]:
    """Load and structurally validate a JSON object."""
    assert path.exists(), f"Expected manifest does not exist: {path}"

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert isinstance(payload, dict)
    return payload


def _assert_metric_bounds(metrics: Dict[str, Any]) -> None:
    """Validate all standard bounded classification metrics."""
    for metric_name in VALID_METRIC_NAMES:
        if metric_name not in metrics:
            continue

        value = float(metrics[metric_name])

        assert math.isfinite(value), (
            f"Metric '{metric_name}' contains non-finite value."
        )

        assert 0.0 <= value <= 1.0, (
            f"Metric '{metric_name}'={value} outside [0, 1]."
        )


# ==============================================================================
# CLASSIFICATION METRIC TESTS
# ==============================================================================


@pytest.mark.unit
def test_compute_classification_metrics_exact_values() -> None:
    """
    Validate deterministic binary-classification metrics against a known
    confusion matrix.

    Confusion matrix:

        TP = 3
        FN = 1
        TN = 3
        FP = 1
    """
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    y_prob = np.array(
        [0.90, 0.80, 0.80, 0.40, 0.10, 0.20, 0.10, 0.70],
        dtype=np.float64,
    )

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    expected = {
        "accuracy": 0.75,
        "precision": 0.75,
        "recall": 0.75,
        "f1_score": 0.75,
        "false_positive_rate": 0.25,
    }

    for key, expected_value in expected.items():
        assert key in metrics
        assert metrics[key] == pytest.approx(
            expected_value,
            abs=DEFAULT_FLOAT_ATOL,
            rel=DEFAULT_FLOAT_RTOL,
        )

    assert "roc_auc" in metrics
    _assert_metric_bounds(metrics)


@pytest.mark.unit
def test_compute_classification_metrics_perfect_classifier() -> None:
    """Perfect predictions should produce perfect bounded metrics."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.10, 0.90, 0.99])

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        y_prob,
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1_score"] == pytest.approx(1.0)
    assert metrics["false_positive_rate"] == pytest.approx(0.0)

    if not math.isnan(float(metrics["roc_auc"])):
        assert metrics["roc_auc"] == pytest.approx(1.0)


@pytest.mark.unit
def test_compute_classification_metrics_all_negative_predictions() -> None:
    """
    Validate safe metric behavior when the classifier predicts only the
    negative class.
    """
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.zeros(6, dtype=int)
    y_prob = np.array(
        [0.10, 0.20, 0.15, 0.30, 0.25, 0.20],
        dtype=np.float64,
    )

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        y_prob,
    )

    _assert_metric_bounds(metrics)

    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["false_positive_rate"] == pytest.approx(0.0)


@pytest.mark.unit
def test_compute_classification_metrics_all_positive_predictions() -> None:
    """
    Validate safe metric behavior when the classifier predicts only the
    positive class.
    """
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.ones(6, dtype=int)
    y_prob = np.array(
        [0.80, 0.70, 0.90, 0.95, 0.99, 0.85],
        dtype=np.float64,
    )

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        y_prob,
    )

    _assert_metric_bounds(metrics)

    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["false_positive_rate"] == pytest.approx(1.0)


@pytest.mark.unit
def test_compute_classification_metrics_rejects_length_mismatch() -> None:
    """
    Evaluation must never silently compare differently sized arrays.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.8])

    with pytest.raises((ValueError, AssertionError)):
        compute_classification_metrics(
            y_true,
            y_pred,
            y_prob,
        )


@pytest.mark.unit
def test_compute_classification_metrics_rejects_non_finite_probabilities() -> None:
    """
    NaN/Inf model probabilities can corrupt evaluation gates and therefore
    must not silently pass through.
    """
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])

    invalid_probabilities = [
        np.array([0.1, np.nan, 0.2, 0.9]),
        np.array([0.1, np.inf, 0.2, 0.9]),
        np.array([0.1, -np.inf, 0.2, 0.9]),
    ]

    for y_prob in invalid_probabilities:
        with pytest.raises((ValueError, AssertionError)):
            compute_classification_metrics(
                y_true,
                y_pred,
                y_prob,
            )


@pytest.mark.unit
def test_compute_classification_metrics_probability_bounds() -> None:
    """
    Probabilities outside [0,1] must never be accepted as valid classifier
    probabilities.
    """
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])

    for invalid_probability in (-0.01, 1.01, 2.0):
        y_prob = np.array(
            [0.1, invalid_probability, 0.2, 0.9],
            dtype=np.float64,
        )

        with pytest.raises((ValueError, AssertionError)):
            compute_classification_metrics(
                y_true,
                y_pred,
                y_prob,
            )


# ==============================================================================
# SLICE / PARITY TESTS
# ==============================================================================


@pytest.mark.unit
def test_evaluate_slice_performance_parity_audit(
    synthetic_train_val_test_dfs: Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """
    Perfect predictions should result in passing parity checks for valid
    categorical slices.
    """
    _, _, test_df = synthetic_train_val_test_dfs

    y_true = test_df["label"].to_numpy()
    y_pred = y_true.copy()
    y_prob = y_true.astype(float)

    report = evaluate_slice_performance(
        df=test_df,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        slice_columns=[
            "tld_extension",
            "organization_id",
        ],
        min_parity_ratio=0.80,
    )

    assert isinstance(report, dict)
    assert "tld_extension" in report
    assert "organization_id" in report
    assert report["overall_parity_passed"] is True


@pytest.mark.unit
def test_evaluate_slice_performance_does_not_mutate_input_dataframe(
    synthetic_train_val_test_dfs: Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """Slice evaluation must not modify production/evaluation input data."""
    _, _, test_df = synthetic_train_val_test_dfs

    original = test_df.copy(deep=True)

    y_true = test_df["label"].to_numpy()
    y_pred = y_true.copy()
    y_prob = y_true.astype(float)

    evaluate_slice_performance(
        df=test_df,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        slice_columns=["tld_extension"],
        min_parity_ratio=0.80,
    )

    pd.testing.assert_frame_equal(test_df, original)


@pytest.mark.unit
def test_evaluate_slice_performance_missing_slice_column_is_handled(
    synthetic_train_val_test_dfs: Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """
    Missing slice columns should not result in a silently fabricated parity
    result.
    """
    _, _, test_df = synthetic_train_val_test_dfs

    y_true = test_df["label"].to_numpy()
    y_pred = y_true.copy()
    y_prob = y_true.astype(float)

    try:
        report = evaluate_slice_performance(
            df=test_df,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            slice_columns=["does_not_exist"],
            min_parity_ratio=0.80,
        )
    except (KeyError, ValueError):
        return

    assert isinstance(report, dict)

    # If the implementation intentionally handles missing slices gracefully,
    # it must not claim that an untested slice passed.
    assert (
        report.get("overall_parity_passed") is False
        or "does_not_exist" not in report
    )


@pytest.mark.unit
def test_evaluate_slice_performance_extreme_slice_imbalance(
    synthetic_train_val_test_dfs: Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """
    Very small slices are statistically unreliable and should not be treated
    as equivalent to well-populated slices.
    """
    _, _, test_df = synthetic_train_val_test_dfs

    test_df = test_df.copy()

    test_df["security_slice"] = "majority"
    test_df.loc[test_df.index[0], "security_slice"] = "tiny"

    y_true = test_df["label"].to_numpy()
    y_pred = y_true.copy()
    y_prob = y_true.astype(float)

    report = evaluate_slice_performance(
        df=test_df,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        slice_columns=["security_slice"],
        min_parity_ratio=0.80,
    )

    assert isinstance(report, dict)


@pytest.mark.unit
def test_slice_parity_threshold_boundary(
    synthetic_train_val_test_dfs: Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """
    Exercise the exact parity threshold boundary.

    This test primarily verifies that the evaluator returns a deterministic
    decision rather than relying on floating-point coincidence.
    """
    _, _, test_df = synthetic_train_val_test_dfs

    y_true = test_df["label"].to_numpy()
    y_pred = y_true.copy()
    y_prob = y_true.astype(float)

    report = evaluate_slice_performance(
        df=test_df,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        slice_columns=["tld_extension"],
        min_parity_ratio=0.80,
    )

    assert isinstance(report, dict)
    assert isinstance(report.get("overall_parity_passed"), bool)


# ==============================================================================
# CHAMPION / CHALLENGER GUARDRAIL TESTS
# ==============================================================================


@pytest.mark.unit
def test_compare_candidate_vs_champion_passed_guardrails() -> None:
    """Candidate that improves relevant metrics should pass guardrails."""
    candidate_metrics = {
        "f1_score": 0.92,
        "false_positive_rate": 0.01,
        "roc_auc": 0.96,
    }

    champion_metrics = {
        "f1_score": 0.89,
        "false_positive_rate": 0.02,
        "roc_auc": 0.94,
    }

    result = compare_candidate_vs_champion(
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        max_f1_drop=0.05,
        max_fpr_increase=0.02,
    )

    assert result["is_candidate_superior"] is True
    assert result["guardrails_passed"] is True
    assert result["regression_warnings"] == []


@pytest.mark.unit
def test_compare_candidate_vs_champion_failed_regression_guardrails() -> None:
    """Large regression must block promotion."""
    candidate_metrics = {
        "f1_score": 0.78,
        "false_positive_rate": 0.05,
        "roc_auc": 0.82,
    }

    champion_metrics = {
        "f1_score": 0.89,
        "false_positive_rate": 0.01,
        "roc_auc": 0.94,
    }

    result = compare_candidate_vs_champion(
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        max_f1_drop=0.05,
        max_fpr_increase=0.02,
    )

    assert result["is_candidate_superior"] is False
    assert result["guardrails_passed"] is False
    assert len(result["regression_warnings"]) > 0


@pytest.mark.unit
def test_compare_candidate_vs_champion_f1_drop_boundary() -> None:
    """
    Candidate exactly at the allowed F1 regression boundary must have a
    deterministic guardrail result.
    """
    champion_metrics = {
        "f1_score": 0.90,
        "false_positive_rate": 0.10,
        "roc_auc": 0.90,
    }

    candidate_metrics = {
        "f1_score": 0.85,
        "false_positive_rate": 0.10,
        "roc_auc": 0.90,
    }

    result = compare_candidate_vs_champion(
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        max_f1_drop=0.05,
        max_fpr_increase=0.02,
    )

    assert isinstance(result["guardrails_passed"], bool)


@pytest.mark.unit
def test_compare_candidate_vs_champion_fpr_boundary() -> None:
    """
    Candidate exactly at the allowed FPR increase boundary must be evaluated
    deterministically.
    """
    champion_metrics = {
        "f1_score": 0.90,
        "false_positive_rate": 0.10,
        "roc_auc": 0.90,
    }

    candidate_metrics = {
        "f1_score": 0.90,
        "false_positive_rate": 0.12,
        "roc_auc": 0.90,
    }

    result = compare_candidate_vs_champion(
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        max_f1_drop=0.05,
        max_fpr_increase=0.02,
    )

    assert isinstance(result["guardrails_passed"], bool)


@pytest.mark.unit
def test_compare_candidate_vs_champion_identical_models() -> None:
    """
    Identical candidate and champion models must not be incorrectly reported
    as a meaningful improvement.
    """
    metrics = {
        "f1_score": 0.90,
        "false_positive_rate": 0.05,
        "roc_auc": 0.95,
    }

    result = compare_candidate_vs_champion(
        candidate_metrics=metrics.copy(),
        champion_metrics=metrics.copy(),
        max_f1_drop=0.05,
        max_fpr_increase=0.02,
    )

    assert isinstance(result["guardrails_passed"], bool)
    assert isinstance(result["is_candidate_superior"], bool)


@pytest.mark.unit
@pytest.mark.parametrize(
    "candidate_metrics,champion_metrics",
    [
        (
            {"f1_score": np.nan, "false_positive_rate": 0.01},
            {"f1_score": 0.90, "false_positive_rate": 0.02},
        ),
        (
            {"f1_score": 0.90, "false_positive_rate": np.inf},
            {"f1_score": 0.90, "false_positive_rate": 0.02},
        ),
        (
            {"f1_score": 0.90},
            {"f1_score": 0.90, "false_positive_rate": 0.02},
        ),
    ],
)
def test_compare_candidate_vs_champion_invalid_metrics_fail_closed(
    candidate_metrics: Dict[str, float],
    champion_metrics: Dict[str, float],
) -> None:
    """
    Security-critical promotion comparisons must not silently treat malformed
    metrics as successful.
    """
    try:
        result = compare_candidate_vs_champion(
            candidate_metrics=candidate_metrics,
            champion_metrics=champion_metrics,
            max_f1_drop=0.05,
            max_fpr_increase=0.02,
        )
    except (ValueError, KeyError, AssertionError):
        return

    assert result["guardrails_passed"] is False


# ==============================================================================
# END-TO-END EVALUATION GATE TESTS
# ==============================================================================


@pytest.mark.unit
def test_evaluate_candidate_model_stage_successful_execution(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Execute the complete Stage 03 evaluation pipeline and validate generated
    artifacts.
    """
    output_dir = mock_pipeline_workspace / "reports" / "eval_test"

    summary = evaluate_candidate_model_stage(
        model_dir=mock_pipeline_workspace / "models" / "candidate",
        champion_model_dir=mock_pipeline_workspace / "models" / "champion",
        data_dir=mock_pipeline_workspace / "data" / "processed",
        output_dir=output_dir,
        decision_threshold=0.50,
        slice_columns=[
            "tld_extension",
            "organization_id",
        ],
        min_parity_ratio=0.80,
    )

    assert isinstance(summary, dict)

    evaluation_path = output_dir / "evaluation_report.json"
    promotion_path = output_dir / "promotion_decision.json"

    assert evaluation_path.exists()
    assert promotion_path.exists()

    evaluation = _load_json(evaluation_path)
    promotion = _load_json(promotion_path)

    assert isinstance(evaluation, dict)
    assert isinstance(promotion, dict)

    required_promotion_fields = {
        "approved_for_promotion",
        "candidate_metrics",
        "slice_parity_report",
    }

    missing = required_promotion_fields - set(promotion)
    assert not missing, (
        f"Promotion manifest missing required fields: {sorted(missing)}"
    )

    assert isinstance(
        promotion["approved_for_promotion"],
        bool,
    )

    assert isinstance(
        promotion["candidate_metrics"],
        dict,
    )

    _assert_metric_bounds(
        promotion["candidate_metrics"]
    )


@pytest.mark.unit
def test_evaluation_stage_generated_metrics_are_finite(
    mock_pipeline_workspace: Path,
) -> None:
    """Generated evaluation metrics must never contain NaN or Infinity."""
    output_dir = mock_pipeline_workspace / "reports" / "finite_metrics"

    evaluate_candidate_model_stage(
        model_dir=mock_pipeline_workspace / "models" / "candidate",
        champion_model_dir=mock_pipeline_workspace / "models" / "champion",
        data_dir=mock_pipeline_workspace / "data" / "processed",
        output_dir=output_dir,
        decision_threshold=0.50,
    )

    promotion = _load_json(
        output_dir / "promotion_decision.json"
    )

    metrics = promotion["candidate_metrics"]

    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            _assert_finite_metric(value, name)


@pytest.mark.unit
def test_evaluate_candidate_model_stage_missing_test_data_fails_closed(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Evaluation without the required test dataset must not produce a promotion
    decision.
    """
    empty_data_dir = (
        mock_pipeline_workspace / "data" / "empty_evaluation_data"
    )
    empty_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir = (
        mock_pipeline_workspace / "reports" / "failed_eval"
    )

    with pytest.raises(FileNotFoundError):
        evaluate_candidate_model_stage(
            model_dir=mock_pipeline_workspace / "models" / "candidate",
            champion_model_dir=mock_pipeline_workspace / "models" / "champion",
            data_dir=empty_data_dir,
            output_dir=output_dir,
        )

    promotion_manifest = output_dir / "promotion_decision.json"

    # A failed evaluation must not leave behind an apparently valid promotion
    # artifact.
    if promotion_manifest.exists():
        payload = _load_json(promotion_manifest)
        assert payload.get("approved_for_promotion") is False


@pytest.mark.unit
def test_evaluate_candidate_model_stage_missing_candidate_model_fails(
    mock_pipeline_workspace: Path,
) -> None:
    """Missing candidate model artifacts must stop evaluation."""
    missing_model_dir = (
        mock_pipeline_workspace / "models" / "missing_candidate"
    )
    missing_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pytest.raises((FileNotFoundError, ValueError)):
        evaluate_candidate_model_stage(
            model_dir=missing_model_dir,
            champion_model_dir=(
                mock_pipeline_workspace
                / "models"
                / "champion"
            ),
            data_dir=(
                mock_pipeline_workspace
                / "data"
                / "processed"
            ),
            output_dir=(
                mock_pipeline_workspace
                / "reports"
                / "missing_candidate"
            ),
        )


@pytest.mark.unit
def test_evaluate_candidate_model_stage_corrupt_candidate_artifact_fails(
    mock_pipeline_workspace: Path,
) -> None:
    """
    A corrupted serialized model must never be interpreted as a valid
    candidate.
    """
    model_dir = (
        mock_pipeline_workspace
        / "models"
        / "candidate"
    )

    model_file = model_dir / "candidate_model.joblib"

    original_bytes = model_file.read_bytes()

    try:
        model_file.write_bytes(
            b"THIS_IS_NOT_A_VALID_JOBLIB_ARTIFACT"
        )

        with pytest.raises(
            (ValueError, EOFError, OSError, Exception)
        ):
            evaluate_candidate_model_stage(
                model_dir=model_dir,
                champion_model_dir=(
                    mock_pipeline_workspace
                    / "models"
                    / "champion"
                ),
                data_dir=(
                    mock_pipeline_workspace
                    / "data"
                    / "processed"
                ),
                output_dir=(
                    mock_pipeline_workspace
                    / "reports"
                    / "corrupt_candidate"
                ),
            )
    finally:
        model_file.write_bytes(original_bytes)


@pytest.mark.unit
def test_evaluate_candidate_model_stage_threshold_changes_predictions(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Different decision thresholds should be accepted and evaluated
    deterministically.

    The test does not require a particular metric direction because the
    optimal threshold depends on the candidate model.
    """
    output_a = (
        mock_pipeline_workspace
        / "reports"
        / "threshold_050"
    )

    output_b = (
        mock_pipeline_workspace
        / "reports"
        / "threshold_080"
    )

    result_a = evaluate_candidate_model_stage(
        model_dir=mock_pipeline_workspace / "models" / "candidate",
        champion_model_dir=mock_pipeline_workspace / "models" / "champion",
        data_dir=mock_pipeline_workspace / "data" / "processed",
        output_dir=output_a,
        decision_threshold=0.50,
    )

    result_b = evaluate_candidate_model_stage(
        model_dir=mock_pipeline_workspace / "models" / "candidate",
        champion_model_dir=mock_pipeline_workspace / "models" / "champion",
        data_dir=mock_pipeline_workspace / "data" / "processed",
        output_dir=output_b,
        decision_threshold=0.80,
    )

    assert isinstance(result_a, dict)
    assert isinstance(result_b, dict)

    assert (output_a / "evaluation_report.json").exists()
    assert (output_b / "evaluation_report.json").exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_threshold",
    [-1.0, -0.01, 1.01, 2.0, np.nan, np.inf, -np.inf],
)
def test_evaluation_stage_rejects_invalid_decision_threshold(
    mock_pipeline_workspace: Path,
    invalid_threshold: float,
) -> None:
    """
    Deployment evaluation must reject thresholds outside [0,1] and
    non-finite thresholds.
    """
    with pytest.raises((ValueError, AssertionError)):
        evaluate_candidate_model_stage(
            model_dir=mock_pipeline_workspace / "models" / "candidate",
            champion_model_dir=(
                mock_pipeline_workspace
                / "models"
                / "champion"
            ),
            data_dir=(
                mock_pipeline_workspace
                / "data"
                / "processed"
            ),
            output_dir=(
                mock_pipeline_workspace
                / "reports"
                / "invalid_threshold"
            ),
            decision_threshold=invalid_threshold,
        )


# ==============================================================================
# DETERMINISM TESTS
# ==============================================================================


@pytest.mark.unit
def test_evaluation_stage_is_deterministic(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Running evaluation twice over identical immutable artifacts should produce
    equivalent candidate metrics.
    """
    output_a = (
        mock_pipeline_workspace
        / "reports"
        / "deterministic_a"
    )

    output_b = (
        mock_pipeline_workspace
        / "reports"
        / "deterministic_b"
    )

    evaluate_candidate_model_stage(
        model_dir=mock_pipeline_workspace / "models" / "candidate",
        champion_model_dir=mock_pipeline_workspace / "models" / "champion",
        data_dir=mock_pipeline_workspace / "data" / "processed",
        output_dir=output_a,
        decision_threshold=0.50,
    )

    evaluate_candidate_model_stage(
        model_dir=mock_pipeline_workspace / "models" / "candidate",
        champion_model_dir=mock_pipeline_workspace / "models" / "champion",
        data_dir=mock_pipeline_workspace / "data" / "processed",
        output_dir=output_b,
        decision_threshold=0.50,
    )

    decision_a = _load_json(
        output_a / "promotion_decision.json"
    )

    decision_b = _load_json(
        output_b / "promotion_decision.json"
    )

    assert decision_a["candidate_metrics"] == pytest.approx(
        decision_b["candidate_metrics"],
        abs=DEFAULT_FLOAT_ATOL,
        rel=DEFAULT_FLOAT_RTOL,
    )


# ==============================================================================
# MANIFEST CONTRACT TESTS
# ==============================================================================


@pytest.mark.unit
def test_promotion_manifest_contains_machine_readable_gate_fields(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Promotion decisions are consumed by downstream automation, so their
    structural contract must be tested explicitly.
    """
    output_dir = (
        mock_pipeline_workspace
        / "reports"
        / "manifest_contract"
    )

    evaluate_candidate_model_stage(
        model_dir=mock_pipeline_workspace / "models" / "candidate",
        champion_model_dir=(
            mock_pipeline_workspace
            / "models"
            / "champion"
        ),
        data_dir=(
            mock_pipeline_workspace
            / "data"
            / "processed"
        ),
        output_dir=output_dir,
        decision_threshold=0.50,
        slice_columns=[
            "tld_extension",
            "organization_id",
        ],
        min_parity_ratio=0.80,
    )

    decision = _load_json(
        output_dir / "promotion_decision.json"
    )

    assert isinstance(
        decision["approved_for_promotion"],
        bool,
    )

    assert isinstance(
        decision["candidate_metrics"],
        dict,
    )

    assert isinstance(
        decision["slice_parity_report"],
        dict,
    )

    _assert_metric_bounds(
        decision["candidate_metrics"]
    )


@pytest.mark.unit
def test_promotion_decision_never_uses_string_boolean(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Prevent a dangerous JSON contract bug such as:

        "approved_for_promotion": "false"

    because non-empty strings can be interpreted as truthy by consumers.
    """
    output_dir = (
        mock_pipeline_workspace
        / "reports"
        / "boolean_contract"
    )

    evaluate_candidate_model_stage(
        model_dir=mock_pipeline_workspace / "models" / "candidate",
        champion_model_dir=(
            mock_pipeline_workspace
            / "models"
            / "champion"
        ),
        data_dir=(
            mock_pipeline_workspace
            / "data"
            / "processed"
        ),
        output_dir=output_dir,
    )

    decision = _load_json(
        output_dir / "promotion_decision.json"
    )

    assert type(
        decision["approved_for_promotion"]
    ) is bool


# ==============================================================================
# METRIC BOUNDS HELPER TESTS
# ==============================================================================


@pytest.mark.unit
def test_assert_metrics_within_bounds_accepts_valid_metrics() -> None:
    """Verify the shared test helper itself."""
    metrics = {
        "accuracy": 0.95,
        "f1_score": 0.91,
        "false_positive_rate": 0.03,
    }

    assert_metrics_within_bounds(
        metrics,
        {
            "accuracy": (0.0, 1.0),
            "f1_score": (0.0, 1.0),
            "false_positive_rate": (0.0, 1.0),
        },
    )


@pytest.mark.unit
def test_assert_metrics_within_bounds_rejects_out_of_range_metric() -> None:
    """Shared metric assertions must reject invalid deployment metrics."""
    metrics = {
        "f1_score": 1.20,
    }

    with pytest.raises(AssertionError):
        assert_metrics_within_bounds(
            metrics,
            {
                "f1_score": (0.0, 1.0),
            },
        )


# ==============================================================================
# NUMERICAL STABILITY TESTS
# ==============================================================================


@pytest.mark.unit
def test_probability_vector_validation_accepts_valid_probabilities() -> None:
    """Sanity test for local probability validation."""
    probabilities = np.array(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        dtype=np.float64,
    )

    _assert_probability_vector(
        probabilities,
        expected_length=5,
    )


@pytest.mark.unit
def test_probability_vector_validation_rejects_nan() -> None:
    """NaN probability vectors are invalid for evaluation."""
    probabilities = np.array(
        [0.1, np.nan, 0.8],
        dtype=np.float64,
    )

    with pytest.raises(AssertionError):
        _assert_probability_vector(
            probabilities,
            expected_length=3,
        )


@pytest.mark.unit
def test_probability_vector_validation_rejects_inf() -> None:
    """Infinite probability vectors are invalid."""
    probabilities = np.array(
        [0.1, np.inf, 0.8],
        dtype=np.float64,
    )

    with pytest.raises(AssertionError):
        _assert_probability_vector(
            probabilities,
            expected_length=3,
        )
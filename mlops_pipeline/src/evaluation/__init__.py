"""
PhishGuard AI - Evaluation Engine & Quality Assurance Suite
============================================================

Version: 14.0.0
Package: src.evaluation
File Path: mlops_pipeline/src/evaluation/__init__.py

Production-grade evaluation orchestration layer for the PhishGuard AI
MLOps ecosystem.

Responsibilities
----------------
- Centralized evaluation-suite orchestration.
- Strict evaluation-input validation.
- Classification, calibration, fairness, adversarial, and regression evaluation.
- Structured evaluator lifecycle/status tracking.
- Failure isolation between evaluation modules.
- Deterministic report generation.
- JSON-safe report serialization.
- Evaluation quality gates.
- Dataset alignment and integrity diagnostics.
- Execution timing and operational telemetry.
- Backward-compatible public exports.

Security / MLOps Principles
---------------------------
1. Evaluation data is treated as untrusted input.
2. No evaluator is allowed to silently mutate X_test/y_test.
3. Evaluators are isolated so one failure does not destroy the whole report.
4. Thresholds and binary labels are validated before model evaluation.
5. Reports contain no raw samples or feature payloads.
6. NumPy/Pandas scalar values are normalized before serialization.
7. Evaluation failures are explicitly represented rather than hidden.
8. Champion/challenger comparison is treated as a separate quality gate.
9. Fairness/adversarial results are optional, but their absence is visible.
10. The final report is suitable for downstream MLOps registration/auditing.

Compatibility
-------------
Designed for the existing PhishGuard AI evaluation modules:

- src.evaluation.metrics
- src.evaluation.calibration
- src.evaluation.adversarial
- src.evaluation.fairness
- src.evaluation.regression
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

# ==============================================================================
# RE-EXPORTS: CORE EVALUATION MODULES
# ==============================================================================

from src.evaluation.adversarial import (
    AdversarialEvaluator,
    evaluate_adversarial_robustness,
)

from src.evaluation.calibration import (
    CalibrationEvaluator,
    calculate_expected_calibration_error,
    evaluate_calibration,
)

from src.evaluation.fairness import (
    FairnessEvaluator,
    evaluate_slice_fairness,
)

from src.evaluation.metrics import (
    ClassificationMetrics,
    calculate_classification_metrics,
    generate_confusion_matrix_report,
)

from src.evaluation.regression import (
    RegressionEvaluator,
    compare_champion_vs_challenger,
)

from src.utils.logger import get_logger


logger = get_logger("src.evaluation")


# ==============================================================================
# PACKAGE CONSTANTS
# ==============================================================================

EVALUATION_ENGINE_VERSION = "14.0.0"

DEFAULT_DECISION_THRESHOLD = 0.50
DEFAULT_CALIBRATION_BINS = 10

MIN_RECOMMENDED_EVALUATION_SAMPLES = 30

STATUS_IN_PROGRESS = "IN_PROGRESS"
STATUS_COMPLETED = "COMPLETED"
STATUS_COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
STATUS_FAILED = "FAILED"

STATUS_PASSED = "PASSED"
STATUS_FAILED_EVALUATOR = "FAILED"
STATUS_SKIPPED = "SKIPPED"
STATUS_ERROR = "ERROR"

# Evaluator names are deliberately stable because downstream MLOps systems
# may use these keys for dashboards, gates, or artifact metadata.
EVALUATOR_CLASSIFICATION = "classification"
EVALUATOR_CALIBRATION = "calibration"
EVALUATOR_FAIRNESS = "fairness"
EVALUATOR_ADVERSARIAL = "adversarial_robustness"
EVALUATOR_REGRESSION = "regression_benchmark"


# ==============================================================================
# TYPE ALIASES
# ==============================================================================

ArrayLike = Union[
    pd.DataFrame,
    pd.Series,
    np.ndarray,
    Sequence[Any],
]

SliceMapping = Mapping[str, Union[pd.Series, np.ndarray, Sequence[Any]]]

AdversarialMapping = Mapping[
    str,
    Union[pd.DataFrame, np.ndarray],
]


# ==============================================================================
# INTERNAL REPORT STRUCTURES
# ==============================================================================

@dataclass
class EvaluatorResult:
    """
    Structured result for a single evaluation component.

    This prevents one evaluator failure from corrupting the entire evaluation
    report and provides machine-readable operational status.
    """

    name: str
    status: str
    duration_seconds: float = 0.0
    result: Any = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluator result into a JSON-safe dictionary."""
        return {
            "name": self.name,
            "status": self.status,
            "duration_seconds": round(float(self.duration_seconds), 6),
            "result": _to_json_safe(self.result),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


# ==============================================================================
# INPUT VALIDATION
# ==============================================================================

def _validate_threshold(threshold: float) -> float:
    """
    Validate and normalize a classification decision threshold.
    """
    if isinstance(threshold, bool):
        raise TypeError("decision_threshold must be a numeric value, not bool.")

    try:
        threshold_float = float(threshold)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "decision_threshold must be a numeric value in the range [0, 1]."
        ) from exc

    if not math.isfinite(threshold_float):
        raise ValueError("decision_threshold must be finite.")

    if not 0.0 <= threshold_float <= 1.0:
        raise ValueError(
            f"decision_threshold must be between 0 and 1; "
            f"received {threshold_float}."
        )

    return threshold_float


def _validate_test_data(
    X_test: ArrayLike,
    y_test: ArrayLike,
) -> Tuple[int, List[str]]:
    """
    Validate basic evaluation dataset integrity.

    Returns
    -------
    Tuple[int, List[str]]
        Sample count and non-fatal diagnostics.
    """
    warnings: List[str] = []

    if X_test is None:
        raise ValueError("X_test cannot be None.")

    if y_test is None:
        raise ValueError("y_test cannot be None.")

    try:
        sample_count = len(X_test)
    except TypeError as exc:
        raise TypeError("X_test must be a sized array-like object.") from exc

    try:
        target_count = len(y_test)
    except TypeError as exc:
        raise TypeError("y_test must be a sized array-like object.") from exc

    if sample_count == 0:
        raise ValueError("Evaluation dataset contains zero samples.")

    if sample_count != target_count:
        raise ValueError(
            "X_test and y_test length mismatch: "
            f"X_test={sample_count}, y_test={target_count}."
        )

    # Convert labels only for validation. Do not mutate caller's data.
    if isinstance(y_test, pd.Series):
        y_values = y_test.to_numpy(copy=True)
    else:
        y_values = np.asarray(y_test)

    if y_values.ndim != 1:
        y_values = y_values.reshape(-1)

    # Reject missing target labels.
    try:
        missing_mask = pd.isna(y_values)
        if bool(np.any(missing_mask)):
            raise ValueError(
                "y_test contains missing/null target labels."
            )
    except TypeError:
        # Some object arrays can produce unusual pd.isna behavior.
        pass

    unique_labels = pd.unique(y_values)

    if len(unique_labels) < 2:
        warnings.append(
            "y_test contains fewer than two classes; "
            "ROC-AUC/PR-AUC and calibration diagnostics may be unreliable."
        )

    # PhishGuard binary classification contract.
    if not set(unique_labels).issubset({0, 1, False, True}):
        raise ValueError(
            "y_test must contain binary labels represented by 0/1 or "
            f"False/True. Observed labels: {unique_labels.tolist()!r}"
        )

    if sample_count < MIN_RECOMMENDED_EVALUATION_SAMPLES:
        warnings.append(
            f"Evaluation dataset contains only {sample_count} samples. "
            f"At least {MIN_RECOMMENDED_EVALUATION_SAMPLES} samples are "
            "recommended for meaningful operational evaluation."
        )

    return sample_count, warnings


def _validate_model_interface(model: Any) -> None:
    """
    Validate the minimum model interface required by the evaluation suite.
    """
    if model is None:
        raise ValueError("model cannot be None.")

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Model must expose a 'predict_proba(X)' method."
        )

    if not callable(getattr(model, "predict_proba")):
        raise TypeError(
            "Model 'predict_proba' attribute must be callable."
        )


def _validate_slice_inputs(
    sensitive_slices: SliceMapping,
    sample_count: int,
) -> List[str]:
    """
    Validate fairness slice definitions without modifying them.
    """
    warnings: List[str] = []

    if not isinstance(sensitive_slices, Mapping):
        raise TypeError("sensitive_slices must be a mapping.")

    for slice_name, slice_values in sensitive_slices.items():
        if not isinstance(slice_name, str) or not slice_name.strip():
            raise ValueError(
                "Fairness slice names must be non-empty strings."
            )

        try:
            slice_count = len(slice_values)
        except TypeError as exc:
            raise TypeError(
                f"Fairness slice '{slice_name}' is not sized."
            ) from exc

        if slice_count != sample_count:
            raise ValueError(
                f"Fairness slice '{slice_name}' length mismatch: "
                f"expected {sample_count}, got {slice_count}."
            )

        if slice_count == 0:
            warnings.append(
                f"Fairness slice '{slice_name}' contains zero samples."
            )

    return warnings


def _validate_adversarial_inputs(
    adversarial_samples: AdversarialMapping,
    expected_sample_count: int,
) -> List[str]:
    """
    Validate adversarial datasets.

    Adversarial datasets are allowed to have a different number of rows than
    the clean evaluation set because perturbation suites commonly contain
    subsets or attack-specific samples.
    """
    warnings: List[str] = []

    if not isinstance(adversarial_samples, Mapping):
        raise TypeError("adversarial_samples must be a mapping.")

    for attack_name, attack_data in adversarial_samples.items():
        if not isinstance(attack_name, str) or not attack_name.strip():
            raise ValueError(
                "Adversarial attack names must be non-empty strings."
            )

        if attack_data is None:
            raise ValueError(
                f"Adversarial dataset '{attack_name}' cannot be None."
            )

        try:
            count = len(attack_data)
        except TypeError as exc:
            raise TypeError(
                f"Adversarial dataset '{attack_name}' is not sized."
            ) from exc

        if count == 0:
            warnings.append(
                f"Adversarial dataset '{attack_name}' contains zero samples."
            )

        if count != expected_sample_count:
            warnings.append(
                f"Adversarial dataset '{attack_name}' contains {count} samples "
                f"instead of the clean evaluation count {expected_sample_count}."
            )

    return warnings


# ==============================================================================
# JSON-SAFE SERIALIZATION
# ==============================================================================

def _to_json_safe(value: Any) -> Any:
    """
    Recursively convert common scientific Python objects into JSON-safe types.

    This is intentionally defensive because evaluation modules may return:
    - NumPy scalars
    - NumPy arrays
    - Pandas Series
    - Pandas DataFrames
    - NaN / Inf
    - nested dictionaries/lists
    """
    if value is None:
        return None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, np.generic):
        return _to_json_safe(value.item())

    if isinstance(value, np.ndarray):
        return [_to_json_safe(item) for item in value.tolist()]

    if isinstance(value, pd.Series):
        return [_to_json_safe(item) for item in value.tolist()]

    if isinstance(value, pd.DataFrame):
        return [
            {
                str(key): _to_json_safe(item)
                for key, item in row.items()
            }
            for row in value.to_dict(orient="records")
        ]

    if isinstance(value, Mapping):
        return {
            str(key): _to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    # Last-resort representation. This avoids JSON serialization crashes while
    # making the fallback explicit.
    return str(value)


# ==============================================================================
# EVALUATOR EXECUTION WRAPPER
# ==============================================================================

def _run_evaluator(
    name: str,
    evaluator_callable: Any,
    *,
    warnings: Optional[List[str]] = None,
) -> EvaluatorResult:
    """
    Execute an evaluator with isolated error handling and timing.

    Evaluation errors are captured as structured report data rather than
    silently swallowed.
    """
    started = time.perf_counter()
    local_warnings = list(warnings or [])

    try:
        result = evaluator_callable()

        duration = time.perf_counter() - started

        return EvaluatorResult(
            name=name,
            status=STATUS_PASSED,
            duration_seconds=duration,
            result=_to_json_safe(result),
            warnings=local_warnings,
        )

    except Exception as exc:
        duration = time.perf_counter() - started

        logger.exception(
            "Evaluation component '%s' failed: %s",
            name,
            exc,
        )

        return EvaluatorResult(
            name=name,
            status=STATUS_ERROR,
            duration_seconds=duration,
            result=None,
            warnings=local_warnings,
            errors=[
                f"{type(exc).__name__}: {str(exc)}"
            ],
        )


def _skipped_evaluator(
    name: str,
    reason: str,
) -> EvaluatorResult:
    """
    Build a consistent skipped evaluator result.
    """
    return EvaluatorResult(
        name=name,
        status=STATUS_SKIPPED,
        duration_seconds=0.0,
        result=None,
        warnings=[reason],
    )


# ==============================================================================
# EVALUATION QUALITY GATES
# ==============================================================================

def _build_quality_gate(
    evaluator_results: Mapping[str, EvaluatorResult],
) -> Dict[str, Any]:
    """
    Build a machine-readable evaluation quality gate.

    The gate is deliberately conservative:
    - evaluator errors fail the suite;
    - classification failure fails the suite;
    - skipped optional evaluators do not fail the suite;
    - warnings alone do not fail the suite.
    """
    errors: List[str] = []
    warnings: List[str] = []

    required_evaluators = {
        EVALUATOR_CLASSIFICATION,
        EVALUATOR_CALIBRATION,
    }

    required_failures: List[str] = []

    for name, evaluator in evaluator_results.items():
        if evaluator.status == STATUS_ERROR:
            errors.extend(
                f"{name}: {message}"
                for message in evaluator.errors
            )

        warnings.extend(
            f"{name}: {message}"
            for message in evaluator.warnings
        )

        if (
            name in required_evaluators
            and evaluator.status != STATUS_PASSED
        ):
            required_failures.append(name)

    gate_passed = (
        not errors
        and not required_failures
    )

    return {
        "passed": gate_passed,
        "status": STATUS_PASSED if gate_passed else STATUS_FAILED,
        "required_evaluator_failures": required_failures,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


# ==============================================================================
# MASTER EVALUATION SUITE
# ==============================================================================

def run_full_evaluation_suite(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    champion_model: Optional[Any] = None,
    sensitive_slices: Optional[Dict[str, pd.Series]] = None,
    adversarial_samples: Optional[
        Dict[str, Union[pd.DataFrame, np.ndarray]]
    ] = None,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    *,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    fail_fast: bool = False,
    raise_on_error: bool = False,
    include_confusion_matrix: bool = False,
) -> Dict[str, Any]:
    """
    Execute the complete PhishGuard AI evaluation suite.

    Parameters
    ----------
    model:
        Candidate/challenger model exposing predict_proba().

    X_test:
        Clean, holdout evaluation feature matrix.

    y_test:
        Binary ground-truth labels.

    champion_model:
        Optional currently deployed/champion model.

    sensitive_slices:
        Optional subgroup definitions for fairness analysis.

    adversarial_samples:
        Optional attack-name -> adversarial feature matrix mapping.

    decision_threshold:
        Probability threshold for positive/phishing classification.

    calibration_bins:
        Number of bins used by the calibration evaluator.

    fail_fast:
        If True, stop after the first evaluator error.
        If False, continue evaluating independent components.

    raise_on_error:
        If True, raise RuntimeError after report generation when one or more
        evaluators fail.

    include_confusion_matrix:
        If True, additionally attempt to generate a confusion-matrix report.

    Returns
    -------
    Dict[str, Any]
        Versioned, structured, JSON-safe evaluation report.

    Notes
    -----
    The caller is responsible for ensuring that X_test is a true holdout set
    that was not used for model fitting, feature fitting, preprocessing fitting,
    threshold optimization, or hyperparameter selection.
    """

    started_at = datetime.now(timezone.utc)
    suite_started = time.perf_counter()

    # --------------------------------------------------------------------------
    # 1. Validate configuration
    # --------------------------------------------------------------------------

    threshold = _validate_threshold(decision_threshold)

    if isinstance(calibration_bins, bool):
        raise TypeError("calibration_bins must be an integer.")

    if not isinstance(calibration_bins, int):
        raise TypeError("calibration_bins must be an integer.")

    if calibration_bins < 2:
        raise ValueError(
            "calibration_bins must be >= 2."
        )

    if calibration_bins > 100:
        raise ValueError(
            "calibration_bins must be <= 100."
        )

    _validate_model_interface(model)

    sample_count, dataset_warnings = _validate_test_data(
        X_test=X_test,
        y_test=y_test,
    )

    # --------------------------------------------------------------------------
    # 2. Optional input validation
    # --------------------------------------------------------------------------

    optional_warnings: List[str] = []

    if sensitive_slices is not None:
        optional_warnings.extend(
            _validate_slice_inputs(
                sensitive_slices,
                sample_count,
            )
        )

    if adversarial_samples is not None:
        optional_warnings.extend(
            _validate_adversarial_inputs(
                adversarial_samples,
                sample_count,
            )
        )

    # --------------------------------------------------------------------------
    # 3. Initialize report
    # --------------------------------------------------------------------------

    logger.info("=" * 72)
    logger.info(
        "STARTING PHISHGUARD AI EVALUATION SUITE | engine=%s",
        EVALUATION_ENGINE_VERSION,
    )
    logger.info(
        "Evaluation dataset: %d samples | threshold=%.4f",
        sample_count,
        threshold,
    )

    evaluator_results: Dict[str, EvaluatorResult] = {}

    report: Dict[str, Any] = {
        "evaluation_engine": {
            "name": "PhishGuard AI Evaluation Engine",
            "version": EVALUATION_ENGINE_VERSION,
        },
        "evaluation_status": STATUS_IN_PROGRESS,
        "started_at": started_at.isoformat(),
        "completed_at": None,
        "duration_seconds": None,
        "sample_count": sample_count,
        "decision_threshold": threshold,
        "calibration_bins": calibration_bins,
        "configuration": {
            "fail_fast": bool(fail_fast),
            "raise_on_error": bool(raise_on_error),
            "include_confusion_matrix": bool(include_confusion_matrix),
            "has_champion_model": champion_model is not None,
            "has_fairness_slices": sensitive_slices is not None,
            "has_adversarial_sets": adversarial_samples is not None,
        },
        "dataset_diagnostics": {
            "warnings": dataset_warnings + optional_warnings,
        },
        "evaluators": {},
        "quality_gate": None,
        "summary": {},
    }

    # --------------------------------------------------------------------------
    # 4. Classification metrics
    # --------------------------------------------------------------------------

    logger.info(
        "[1/5] Running core classification evaluation..."
    )

    classification_result = _run_evaluator(
        EVALUATOR_CLASSIFICATION,
        lambda: calculate_classification_metrics(
            model=model,
            X=X_test,
            y_true=y_test,
            threshold=threshold,
        ),
    )

    evaluator_results[EVALUATOR_CLASSIFICATION] = classification_result

    if include_confusion_matrix:
        confusion_result = _run_evaluator(
            "confusion_matrix",
            lambda: generate_confusion_matrix_report(
                model=model,
                X=X_test,
                y_true=y_test,
                threshold=threshold,
            ),
        )
        evaluator_results["confusion_matrix"] = confusion_result

    if (
        fail_fast
        and classification_result.status == STATUS_ERROR
    ):
        logger.error(
            "Classification evaluator failed and fail_fast=True."
        )

        report["evaluators"] = {
            name: result.to_dict()
            for name, result in evaluator_results.items()
        }

        report["evaluation_status"] = STATUS_FAILED
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report["duration_seconds"] = round(
            time.perf_counter() - suite_started,
            6,
        )
        report["quality_gate"] = _build_quality_gate(
            evaluator_results
        )

        if raise_on_error:
            raise RuntimeError(
                "PhishGuard evaluation suite failed during classification "
                "evaluation."
            )

        return _to_json_safe(report)

    # --------------------------------------------------------------------------
    # 5. Calibration
    # --------------------------------------------------------------------------

    logger.info(
        "[2/5] Running probability calibration evaluation..."
    )

    calibration_result = _run_evaluator(
        EVALUATOR_CALIBRATION,
        lambda: evaluate_calibration(
            model=model,
            X=X_test,
            y_true=y_test,
            n_bins=calibration_bins,
        ),
    )

    evaluator_results[EVALUATOR_CALIBRATION] = calibration_result

    if (
        fail_fast
        and calibration_result.status == STATUS_ERROR
    ):
        logger.error(
            "Calibration evaluator failed and fail_fast=True."
        )

        report["evaluators"] = {
            name: result.to_dict()
            for name, result in evaluator_results.items()
        }

        report["evaluation_status"] = STATUS_FAILED
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report["duration_seconds"] = round(
            time.perf_counter() - suite_started,
            6,
        )
        report["quality_gate"] = _build_quality_gate(
            evaluator_results
        )

        if raise_on_error:
            raise RuntimeError(
                "PhishGuard evaluation suite failed during calibration "
                "evaluation."
            )

        return _to_json_safe(report)

    # --------------------------------------------------------------------------
    # 6. Fairness / slice evaluation
    # --------------------------------------------------------------------------

    logger.info(
        "[3/5] Running fairness/slice evaluation..."
    )

    if sensitive_slices:
        fairness_result = _run_evaluator(
            EVALUATOR_FAIRNESS,
            lambda: evaluate_slice_fairness(
                model=model,
                X=X_test,
                y_true=y_test,
                slices=sensitive_slices,
                threshold=threshold,
            ),
        )
    else:
        fairness_result = _skipped_evaluator(
            EVALUATOR_FAIRNESS,
            "No sensitive_slices were supplied.",
        )

    evaluator_results[EVALUATOR_FAIRNESS] = fairness_result

    if (
        fail_fast
        and fairness_result.status == STATUS_ERROR
    ):
        logger.error(
            "Fairness evaluator failed and fail_fast=True."
        )

        report["evaluators"] = {
            name: result.to_dict()
            for name, result in evaluator_results.items()
        }

        report["evaluation_status"] = STATUS_FAILED
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report["duration_seconds"] = round(
            time.perf_counter() - suite_started,
            6,
        )
        report["quality_gate"] = _build_quality_gate(
            evaluator_results
        )

        if raise_on_error:
            raise RuntimeError(
                "PhishGuard evaluation suite failed during fairness "
                "evaluation."
            )

        return _to_json_safe(report)

    # --------------------------------------------------------------------------
    # 7. Adversarial robustness evaluation
    # --------------------------------------------------------------------------

    logger.info(
        "[4/5] Running adversarial robustness evaluation..."
    )

    if adversarial_samples:
        adversarial_result = _run_evaluator(
            EVALUATOR_ADVERSARIAL,
            lambda: evaluate_adversarial_robustness(
                model=model,
                adversarial_sets=adversarial_samples,
                threshold=threshold,
            ),
        )
    else:
        adversarial_result = _skipped_evaluator(
            EVALUATOR_ADVERSARIAL,
            "No adversarial_samples were supplied.",
        )

    evaluator_results[EVALUATOR_ADVERSARIAL] = adversarial_result

    # --------------------------------------------------------------------------
    # 8. Champion vs challenger
    # --------------------------------------------------------------------------

    logger.info(
        "[5/5] Running champion/challenger regression evaluation..."
    )

    if champion_model is not None:
        _validate_model_interface(champion_model)

        regression_result = _run_evaluator(
            EVALUATOR_REGRESSION,
            lambda: compare_champion_vs_challenger(
                champion=champion_model,
                challenger=model,
                X_test=X_test,
                y_test=y_test,
                threshold=threshold,
            ),
        )
    else:
        regression_result = _skipped_evaluator(
            EVALUATOR_REGRESSION,
            "No champion_model was supplied.",
        )

    evaluator_results[EVALUATOR_REGRESSION] = regression_result

    # --------------------------------------------------------------------------
    # 9. Finalize report
    # --------------------------------------------------------------------------

    completed_at = datetime.now(timezone.utc)
    duration_seconds = time.perf_counter() - suite_started

    report["evaluators"] = {
        name: result.to_dict()
        for name, result in evaluator_results.items()
    }

    quality_gate = _build_quality_gate(
        evaluator_results
    )

    report["quality_gate"] = quality_gate

    status_counts: Dict[str, int] = {}

    for evaluator in evaluator_results.values():
        status_counts[evaluator.status] = (
            status_counts.get(evaluator.status, 0) + 1
        )

    evaluator_errors = sum(
        len(evaluator.errors)
        for evaluator in evaluator_results.values()
    )

    evaluator_warnings = sum(
        len(evaluator.warnings)
        for evaluator in evaluator_results.values()
    )

    report["summary"] = {
        "total_evaluators": len(evaluator_results),
        "status_counts": status_counts,
        "error_count": evaluator_errors,
        "warning_count": evaluator_warnings,
        "quality_gate_passed": bool(
            quality_gate["passed"]
        ),
    }

    if quality_gate["passed"]:
        if evaluator_warnings > 0:
            final_status = STATUS_COMPLETED_WITH_WARNINGS
        else:
            final_status = STATUS_COMPLETED
    else:
        final_status = STATUS_FAILED

    report["evaluation_status"] = final_status
    report["completed_at"] = completed_at.isoformat()
    report["duration_seconds"] = round(
        duration_seconds,
        6,
    )

    # --------------------------------------------------------------------------
    # 10. Operational logging
    # --------------------------------------------------------------------------

    logger.info(
        "Evaluation suite completed | status=%s | duration=%.3fs | "
        "evaluators=%d | errors=%d | warnings=%d",
        final_status,
        duration_seconds,
        len(evaluator_results),
        evaluator_errors,
        evaluator_warnings,
    )

    logger.info(
        "Evaluation quality gate: %s",
        quality_gate["status"],
    )

    logger.info("=" * 72)

    # --------------------------------------------------------------------------
    # 11. Optional strict failure mode
    # --------------------------------------------------------------------------

    if raise_on_error and not quality_gate["passed"]:
        raise RuntimeError(
            "PhishGuard AI evaluation quality gate failed. "
            f"Errors={quality_gate['error_count']}, "
            f"required_failures="
            f"{quality_gate['required_evaluator_failures']}"
        )

    return _to_json_safe(report)


# ==============================================================================
# REPORT VALIDATION / PERSISTENCE HELPERS
# ==============================================================================

def validate_evaluation_report(
    report: Mapping[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Validate the structural integrity of an evaluation report.

    This does not judge model quality. It only verifies that the evaluation
    artifact has the expected structural contract.
    """
    errors: List[str] = []

    if not isinstance(report, Mapping):
        return False, ["Evaluation report must be a mapping/dictionary."]

    required_fields = {
        "evaluation_engine",
        "evaluation_status",
        "started_at",
        "completed_at",
        "duration_seconds",
        "sample_count",
        "decision_threshold",
        "evaluators",
        "quality_gate",
        "summary",
    }

    missing = sorted(
        required_fields.difference(report.keys())
    )

    if missing:
        errors.append(
            f"Missing required report fields: {missing}"
        )

    if "decision_threshold" in report:
        try:
            _validate_threshold(
                report["decision_threshold"]
            )
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))

    if "sample_count" in report:
        try:
            if int(report["sample_count"]) <= 0:
                errors.append(
                    "sample_count must be greater than zero."
                )
        except (TypeError, ValueError):
            errors.append(
                "sample_count must be an integer."
            )

    if "evaluators" in report and not isinstance(
        report["evaluators"],
        Mapping,
    ):
        errors.append(
            "evaluators must be a mapping."
        )

    if "quality_gate" in report and not isinstance(
        report["quality_gate"],
        Mapping,
    ):
        errors.append(
            "quality_gate must be a mapping."
        )

    return len(errors) == 0, errors


def save_evaluation_report(
    report: Mapping[str, Any],
    filepath: Union[str, Path],
    *,
    overwrite: bool = False,
) -> Path:
    """
    Persist a JSON-safe evaluation report to disk.

    Parameters
    ----------
    report:
        Evaluation report returned by run_full_evaluation_suite.

    filepath:
        Destination JSON file.

    overwrite:
        Whether an existing report may be replaced.

    Returns
    -------
    Path
        Absolute report path.
    """
    is_valid, errors = validate_evaluation_report(report)

    if not is_valid:
        raise ValueError(
            "Cannot persist invalid evaluation report: "
            + "; ".join(errors)
        )

    path = Path(filepath).expanduser().resolve()

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Evaluation report already exists: '{path}'. "
            "Set overwrite=True to replace it."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_report = _to_json_safe(dict(report))

    # Use a temporary file first to reduce the risk of leaving a partially
    # written report if the process crashes during serialization.
    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                safe_report,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
            handle.flush()

        temp_path.replace(path)

    except Exception:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass

        raise

    logger.info(
        "Evaluation report persisted to '%s'",
        path,
    )

    return path


# ==============================================================================
# CONVENIENCE ACCESSORS
# ==============================================================================

def evaluation_passed(
    report: Mapping[str, Any],
) -> bool:
    """
    Return whether the evaluation quality gate passed.
    """
    quality_gate = report.get("quality_gate")

    if not isinstance(quality_gate, Mapping):
        return False

    return bool(
        quality_gate.get("passed", False)
    )


def get_evaluation_errors(
    report: Mapping[str, Any],
) -> List[str]:
    """
    Extract all evaluator errors from a report.
    """
    errors: List[str] = []

    evaluators = report.get("evaluators", {})

    if not isinstance(evaluators, Mapping):
        return errors

    for evaluator_name, evaluator_data in evaluators.items():
        if not isinstance(evaluator_data, Mapping):
            continue

        evaluator_errors = evaluator_data.get(
            "errors",
            [],
        )

        if isinstance(evaluator_errors, list):
            errors.extend(
                f"{evaluator_name}: {error}"
                for error in evaluator_errors
            )

    return errors


# ==============================================================================
# PUBLIC EXPORT CONTRACT
# ==============================================================================

__all__ = [
    # Engine metadata
    "EVALUATION_ENGINE_VERSION",

    # Result structures
    "EvaluatorResult",

    # Master orchestrator
    "run_full_evaluation_suite",

    # Report utilities
    "validate_evaluation_report",
    "save_evaluation_report",
    "evaluation_passed",
    "get_evaluation_errors",

    # Core classification metrics
    "ClassificationMetrics",
    "calculate_classification_metrics",
    "generate_confusion_matrix_report",

    # Calibration diagnostics
    "CalibrationEvaluator",
    "calculate_expected_calibration_error",
    "evaluate_calibration",

    # Adversarial robustness
    "AdversarialEvaluator",
    "evaluate_adversarial_robustness",

    # Fairness
    "FairnessEvaluator",
    "evaluate_slice_fairness",

    # Regression / champion vs challenger
    "RegressionEvaluator",
    "compare_champion_vs_challenger",
]
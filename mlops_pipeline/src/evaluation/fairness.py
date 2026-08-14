"""
PhishGuard AI - Slice-Based Fairness & Subgroup Parity Module
==============================================================

Version: 14.0.0
File Path: src/evaluation/fairness.py

Production-grade subgroup and slice evaluation for phishing classifiers.

This module evaluates model behavior across categorical slices such as:

    - URL TLD
    - language / locale
    - organization / tenant
    - email source
    - attachment type
    - provider
    - traffic source
    - model routing group

The implementation intentionally treats "fairness" as a model-risk and
performance-stability problem rather than assuming that a single F1 parity
ratio is sufficient.

Supported diagnostics include:

    * Accuracy
    * Precision
    * Recall / TPR
    * F1
    * F-beta
    * Specificity / TNR
    * False-positive rate
    * False-negative rate
    * Selection / positive prediction rate
    * Disparate impact ratio
    * TPR parity ratio
    * FPR ratio
    * F1 parity ratio
    * Worst-slice identification
    * Minimum sample-size filtering
    * Optional bootstrap confidence intervals
    * Multiple-slice auditing
    * Explicit handling of undefined metrics
    * JSON-safe reports
    * MLOps fairness gates

Important:
-----------
Fairness metrics are contextual. For cybersecurity classifiers, some
"protected-attribute" concepts from conventional fairness literature may
not apply directly. This module therefore supports generic operational
slices and does not automatically declare a slice "unfair" merely because
its metrics differ.

A fairness gate should be interpreted as a model-risk policy decision,
not as a legal determination.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from src.utils.logger import get_logger

logger = get_logger("src.evaluation.fairness")


# ==============================================================================
# CONSTANTS
# ==============================================================================

MODULE_VERSION = "14.0.0"

DEFAULT_MIN_PARITY_RATIO = 0.80
DEFAULT_MIN_SAMPLES_PER_SLICE = 30
DEFAULT_MAX_FPR_RATIO = 1.25
DEFAULT_MAX_TPR_GAP = 0.20
DEFAULT_MAX_FPR_GAP = 0.20
DEFAULT_MAX_SELECTION_RATE_RATIO = 1.25

SUPPORTED_LABELS = {0, 1}


# ==============================================================================
# INTERNAL VALIDATION HELPERS
# ==============================================================================


def _to_1d_array(
    values: Union[pd.Series, np.ndarray, Sequence[Any]],
    name: str,
) -> np.ndarray:
    """Convert an input sequence to a one-dimensional NumPy array."""
    if values is None:
        raise ValueError(f"'{name}' must not be None.")

    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError(
            f"'{name}' must be one-dimensional; received shape={array.shape}."
        )

    if len(array) == 0:
        raise ValueError(f"'{name}' must contain at least one sample.")

    return array


def _validate_binary_labels(
    y_true: Union[pd.Series, np.ndarray, Sequence[Any]],
) -> np.ndarray:
    """Validate and normalize binary ground-truth labels."""
    arr = _to_1d_array(y_true, "y_true")

    if pd.isna(arr).any():
        raise ValueError("'y_true' contains missing values.")

    try:
        arr_float = arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("'y_true' must contain binary numeric labels 0/1.") from exc

    if not np.all(np.isin(arr_float, [0.0, 1.0])):
        unique_values = np.unique(arr_float).tolist()
        raise ValueError(
            "'y_true' must contain only binary labels 0 and 1. "
            f"Observed values: {unique_values}"
        )

    return arr_float.astype(np.int8)


def _validate_binary_predictions(
    y_pred: Union[pd.Series, np.ndarray, Sequence[Any]],
) -> np.ndarray:
    """Validate and normalize binary predictions."""
    arr = _to_1d_array(y_pred, "y_pred")

    if pd.isna(arr).any():
        raise ValueError("'y_pred' contains missing values.")

    try:
        arr_float = arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("'y_pred' must contain binary numeric predictions 0/1.") from exc

    if not np.all(np.isin(arr_float, [0.0, 1.0])):
        unique_values = np.unique(arr_float).tolist()
        raise ValueError(
            "'y_pred' must contain only binary predictions 0 and 1. "
            f"Observed values: {unique_values}"
        )

    return arr_float.astype(np.int8)


def _validate_probability_vector(
    y_prob: Union[pd.Series, np.ndarray, Sequence[Any]],
) -> np.ndarray:
    """Validate a probability vector."""
    arr = _to_1d_array(y_prob, "y_prob").astype(float)

    if not np.all(np.isfinite(arr)):
        raise ValueError("'y_prob' contains NaN or infinite values.")

    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("'y_prob' values must be within [0, 1].")

    return arr


def _validate_threshold(threshold: float) -> float:
    """Validate classification threshold."""
    threshold = float(threshold)

    if not math.isfinite(threshold):
        raise ValueError("Decision threshold must be finite.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Decision threshold must be between 0 and 1. Received {threshold}."
        )

    return threshold


def _safe_ratio(
    numerator: float,
    denominator: float,
    *,
    undefined_value: Optional[float] = None,
) -> Optional[float]:
    """Calculate a safe ratio without silently producing infinity."""
    if denominator == 0:
        return undefined_value

    return float(numerator / denominator)


def _safe_gap(
    first: Optional[float],
    second: Optional[float],
) -> Optional[float]:
    """Absolute gap between two optional metrics."""
    if first is None or second is None:
        return None

    return float(abs(first - second))


def _json_safe(value: Any) -> Any:
    """
    Recursively convert NumPy / pandas values into JSON-safe Python objects.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not math.isfinite(value) else value

    if isinstance(value, float):
        return None if not math.isfinite(value) else value

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if pd.isna(value) if np.isscalar(value) else False:
        return None

    return value


# ==============================================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ==============================================================================


def _bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    *,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate a percentile bootstrap confidence interval.

    Metrics that become undefined for a resample are skipped.
    """
    if n_bootstrap <= 0:
        return None, None

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    n = len(y_true)

    if n < 2:
        return None, None

    values: List[float] = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        yt = y_true[indices]
        yp = y_pred[indices]

        try:
            if metric == "accuracy":
                value = accuracy_score(yt, yp)

            elif metric == "precision":
                value = precision_score(
                    yt,
                    yp,
                    zero_division=0,
                )

            elif metric == "recall":
                if np.sum(yt == 1) == 0:
                    continue

                value = recall_score(
                    yt,
                    yp,
                    zero_division=0,
                )

            elif metric == "f1":
                if np.sum(yt == 1) == 0:
                    continue

                value = f1_score(
                    yt,
                    yp,
                    zero_division=0,
                )

            else:
                raise ValueError(f"Unsupported bootstrap metric: {metric}")

            if math.isfinite(float(value)):
                values.append(float(value))

        except Exception:
            continue

    if not values:
        return None, None

    alpha = 1.0 - confidence_level

    lower = float(np.quantile(values, alpha / 2))
    upper = float(np.quantile(values, 1.0 - alpha / 2))

    return lower, upper


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================


@dataclass
class SliceMetricResult:
    """
    Detailed performance metrics for one slice category.

    None indicates that a metric was mathematically undefined rather than
    silently converting the result to zero.
    """

    slice_name: str
    category_value: str

    sample_count: int
    positive_count: int
    negative_count: int

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    accuracy: float
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    f0_5: Optional[float]

    specificity: Optional[float]
    false_positive_rate: Optional[float]
    false_negative_rate: Optional[float]

    selection_rate: float

    accuracy_ci_lower: Optional[float] = None
    accuracy_ci_upper: Optional[float] = None

    precision_ci_lower: Optional[float] = None
    precision_ci_upper: Optional[float] = None

    recall_ci_lower: Optional[float] = None
    recall_ci_upper: Optional[float] = None

    f1_ci_lower: Optional[float] = None
    f1_ci_upper: Optional[float] = None

    meets_minimum_sample_size: bool = True
    has_positive_examples: bool = True
    has_negative_examples: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe dictionary representation."""
        return _json_safe(asdict(self))


@dataclass
class FairnessReport:
    """
    Aggregated fairness and subgroup-stability report for one slice dimension.
    """

    slice_name: str

    overall_f1_parity_ratio: Optional[float]

    min_f1_category: str
    max_f1_category: str

    min_parity_threshold: float
    passed_fairness_gate: bool

    category_results: Dict[str, SliceMetricResult]

    # Selection-rate / disparate-impact analysis
    disparate_impact_ratio: Optional[float]
    min_selection_rate_category: str
    max_selection_rate_category: str

    # TPR / recall parity
    tpr_gap: Optional[float]
    tpr_parity_ratio: Optional[float]

    # FPR parity
    fpr_gap: Optional[float]
    fpr_ratio: Optional[float]

    # FNR parity
    fnr_gap: Optional[float]

    # Selection-rate gap
    selection_rate_gap: Optional[float]

    # Worst slice
    worst_slice_by_f1: str
    worst_slice_by_fpr: str
    worst_slice_by_recall: str

    # Coverage
    evaluated_category_count: int
    excluded_category_count: int

    # Gate details
    passed_f1_gate: bool
    passed_disparate_impact_gate: bool
    passed_tpr_gate: bool
    passed_fpr_gate: bool

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe dictionary representation."""
        return _json_safe(asdict(self))

    def export_json(self, filepath: Union[str, Path]) -> Path:
        """Export fairness report to JSON."""
        path = Path(filepath).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                self.to_dict(),
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        logger.info(
            "Exported fairness report for '%s' to '%s'.",
            self.slice_name,
            path,
        )

        return path


# ==============================================================================
# FAIRNESS EVALUATOR
# ==============================================================================


class FairnessEvaluator:
    """
    Production-grade subgroup fairness and performance-stability evaluator.

    The evaluator intentionally supports several independent gates because
    F1 parity alone can hide operationally important problems.

    Example:
        evaluator = FairnessEvaluator(
            min_parity_ratio=0.80,
            min_samples_per_slice=30,
            min_disparate_impact=0.80,
            max_tpr_gap=0.20,
            max_fpr_gap=0.20,
        )
    """

    def __init__(
        self,
        min_parity_ratio: float = DEFAULT_MIN_PARITY_RATIO,
        min_samples_per_slice: int = DEFAULT_MIN_SAMPLES_PER_SLICE,
        min_disparate_impact: float = 0.80,
        max_tpr_gap: float = DEFAULT_MAX_TPR_GAP,
        max_fpr_gap: float = DEFAULT_MAX_FPR_GAP,
        max_fpr_ratio: float = DEFAULT_MAX_FPR_RATIO,
        bootstrap_samples: int = 0,
        confidence_level: float = 0.95,
        random_state: int = 42,
        include_undefined_slices: bool = True,
    ) -> None:
        self._validate_configuration(
            min_parity_ratio=min_parity_ratio,
            min_samples_per_slice=min_samples_per_slice,
            min_disparate_impact=min_disparate_impact,
            max_tpr_gap=max_tpr_gap,
            max_fpr_gap=max_fpr_gap,
            max_fpr_ratio=max_fpr_ratio,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
        )

        self.min_parity_ratio = float(min_parity_ratio)
        self.min_samples_per_slice = int(min_samples_per_slice)
        self.min_disparate_impact = float(min_disparate_impact)
        self.max_tpr_gap = float(max_tpr_gap)
        self.max_fpr_gap = float(max_fpr_gap)
        self.max_fpr_ratio = float(max_fpr_ratio)

        self.bootstrap_samples = int(bootstrap_samples)
        self.confidence_level = float(confidence_level)
        self.random_state = int(random_state)
        self.include_undefined_slices = bool(include_undefined_slices)

    @staticmethod
    def _validate_configuration(
        *,
        min_parity_ratio: float,
        min_samples_per_slice: int,
        min_disparate_impact: float,
        max_tpr_gap: float,
        max_fpr_gap: float,
        max_fpr_ratio: float,
        bootstrap_samples: int,
        confidence_level: float,
    ) -> None:
        if not 0.0 <= min_parity_ratio <= 1.0:
            raise ValueError("min_parity_ratio must be within [0, 1].")

        if min_samples_per_slice < 1:
            raise ValueError("min_samples_per_slice must be >= 1.")

        if not 0.0 <= min_disparate_impact <= 1.0:
            raise ValueError("min_disparate_impact must be within [0, 1].")

        if not 0.0 <= max_tpr_gap <= 1.0:
            raise ValueError("max_tpr_gap must be within [0, 1].")

        if not 0.0 <= max_fpr_gap <= 1.0:
            raise ValueError("max_fpr_gap must be within [0, 1].")

        if max_fpr_ratio < 1.0:
            raise ValueError("max_fpr_ratio must be >= 1.")

        if bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be >= 0.")

        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1.")

    # --------------------------------------------------------------------------
    # SLICE METRICS
    # --------------------------------------------------------------------------

    def _calculate_slice_metrics(
        self,
        slice_name: str,
        category_value: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> SliceMetricResult:
        """Calculate comprehensive metrics for one slice."""
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))

        total = len(y_true)
        positive_count = tp + fn
        negative_count = tn + fp

        accuracy = float(accuracy_score(y_true, y_pred))

        precision: Optional[float]
        recall: Optional[float]
        f1: Optional[float]
        f0_5: Optional[float]
        specificity: Optional[float]
        fpr: Optional[float]
        fnr: Optional[float]

        if tp + fp > 0:
            precision = float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            )
        else:
            precision = None

        if positive_count > 0:
            recall = float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            )
            f1 = float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            )
            f0_5 = float(
                fbeta_score(
                    y_true,
                    y_pred,
                    beta=0.5,
                    zero_division=0,
                )
            )
            fnr = float(fn / positive_count)
        else:
            recall = None
            f1 = None
            f0_5 = None
            fnr = None

        if negative_count > 0:
            specificity = float(tn / negative_count)
            fpr = float(fp / negative_count)
        else:
            specificity = None
            fpr = None

        selection_rate = float(np.mean(y_pred == 1)) if total > 0 else 0.0

        result = SliceMetricResult(
            slice_name=slice_name,
            category_value=category_value,
            sample_count=total,
            positive_count=positive_count,
            negative_count=negative_count,
            true_positive=tp,
            false_positive=fp,
            true_negative=tn,
            false_negative=fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            f0_5=f0_5,
            specificity=specificity,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            selection_rate=selection_rate,
            meets_minimum_sample_size=total >= self.min_samples_per_slice,
            has_positive_examples=positive_count > 0,
            has_negative_examples=negative_count > 0,
        )

        if self.bootstrap_samples > 0:
            (
                result.accuracy_ci_lower,
                result.accuracy_ci_upper,
            ) = _bootstrap_metric_ci(
                y_true,
                y_pred,
                "accuracy",
                n_bootstrap=self.bootstrap_samples,
                confidence_level=self.confidence_level,
                random_state=self.random_state,
            )

            (
                result.precision_ci_lower,
                result.precision_ci_upper,
            ) = _bootstrap_metric_ci(
                y_true,
                y_pred,
                "precision",
                n_bootstrap=self.bootstrap_samples,
                confidence_level=self.confidence_level,
                random_state=self.random_state + 1,
            )

            (
                result.recall_ci_lower,
                result.recall_ci_upper,
            ) = _bootstrap_metric_ci(
                y_true,
                y_pred,
                "recall",
                n_bootstrap=self.bootstrap_samples,
                confidence_level=self.confidence_level,
                random_state=self.random_state + 2,
            )

            (
                result.f1_ci_lower,
                result.f1_ci_upper,
            ) = _bootstrap_metric_ci(
                y_true,
                y_pred,
                "f1",
                n_bootstrap=self.bootstrap_samples,
                confidence_level=self.confidence_level,
                random_state=self.random_state + 3,
            )

        return result

    # --------------------------------------------------------------------------
    # SINGLE SLICE
    # --------------------------------------------------------------------------

    def evaluate_slice(
        self,
        slice_name: str,
        slice_series: Union[pd.Series, np.ndarray, Sequence[Any]],
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
    ) -> FairnessReport:
        """
        Evaluate one categorical slice dimension.

        Categories with fewer than `min_samples_per_slice` samples are reported
        but excluded from parity gates by default.
        """
        if not slice_name or not str(slice_name).strip():
            raise ValueError("slice_name must be a non-empty string.")

        y_true_arr = _validate_binary_labels(y_true)
        y_pred_arr = _validate_binary_predictions(y_pred)
        slice_arr = _to_1d_array(slice_series, "slice_series")

        if len(slice_arr) != len(y_true_arr):
            raise ValueError(
                "Length mismatch: slice_series, y_true, and y_pred must contain "
                "the same number of samples."
            )

        if pd.isna(slice_arr).any():
            if self.include_undefined_slices:
                slice_arr = slice_arr.astype(object)
                slice_arr[pd.isna(slice_arr)] = "__MISSING__"
            else:
                raise ValueError(
                    f"Slice '{slice_name}' contains missing values."
                )

        # Preserve deterministic category order.
        categories = sorted(
            {str(value) for value in slice_arr},
            key=lambda item: item,
        )

        category_results: Dict[str, SliceMetricResult] = {}

        for category in categories:
            mask = np.asarray(
                [str(value) == category for value in slice_arr],
                dtype=bool,
            )

            if not np.any(mask):
                continue

            result = self._calculate_slice_metrics(
                slice_name=slice_name,
                category_value=category,
                y_true=y_true_arr[mask],
                y_pred=y_pred_arr[mask],
            )

            category_results[category] = result

        valid_results = {
            category: result
            for category, result in category_results.items()
            if result.meets_minimum_sample_size
        }

        if not valid_results:
            logger.warning(
                "No categories in slice '%s' met minimum sample size=%d.",
                slice_name,
                self.min_samples_per_slice,
            )

            return FairnessReport(
                slice_name=slice_name,
                overall_f1_parity_ratio=None,
                min_f1_category="N/A",
                max_f1_category="N/A",
                min_parity_threshold=self.min_parity_ratio,
                passed_fairness_gate=False,
                category_results=category_results,
                disparate_impact_ratio=None,
                min_selection_rate_category="N/A",
                max_selection_rate_category="N/A",
                tpr_gap=None,
                tpr_parity_ratio=None,
                fpr_gap=None,
                fpr_ratio=None,
                fnr_gap=None,
                selection_rate_gap=None,
                worst_slice_by_f1="N/A",
                worst_slice_by_fpr="N/A",
                worst_slice_by_recall="N/A",
                evaluated_category_count=0,
                excluded_category_count=len(category_results),
                passed_f1_gate=False,
                passed_disparate_impact_gate=False,
                passed_tpr_gate=False,
                passed_fpr_gate=False,
                metadata={
                    "status": "INSUFFICIENT_DATA",
                    "minimum_samples_per_slice": self.min_samples_per_slice,
                },
            )

        # ------------------------------------------------------------------
        # F1 PARITY
        # ------------------------------------------------------------------

        f1_values = {
            category: result.f1
            for category, result in valid_results.items()
            if result.f1 is not None
        }

        if f1_values:
            min_f1_category = min(
                f1_values,
                key=lambda category: f1_values[category],  # type: ignore
            )
            max_f1_category = max(
                f1_values,
                key=lambda category: f1_values[category],  # type: ignore
            )

            min_f1 = float(f1_values[min_f1_category])
            max_f1 = float(f1_values[max_f1_category])

            f1_parity_ratio = (
                float(min_f1 / max_f1)
                if max_f1 > 0
                else (1.0 if min_f1 == 0 else None)
            )

            passed_f1_gate = (
                f1_parity_ratio is not None
                and f1_parity_ratio >= self.min_parity_ratio
            )
        else:
            min_f1_category = "N/A"
            max_f1_category = "N/A"
            f1_parity_ratio = None
            passed_f1_gate = False

        # ------------------------------------------------------------------
        # SELECTION RATE / DISPARATE IMPACT
        # ------------------------------------------------------------------

        selection_rates = {
            category: result.selection_rate
            for category, result in valid_results.items()
        }

        min_selection_category = min(
            selection_rates,
            key=selection_rates.get,  # type: ignore
        )

        max_selection_category = max(
            selection_rates,
            key=selection_rates.get,  # type: ignore
        )

        min_selection_rate = selection_rates[min_selection_category]
        max_selection_rate = selection_rates[max_selection_category]

        if max_selection_rate > 0:
            disparate_impact_ratio = float(
                min_selection_rate / max_selection_rate
            )
        else:
            disparate_impact_ratio = 1.0

        passed_disparate_impact_gate = (
            disparate_impact_ratio >= self.min_disparate_impact
        )

        selection_rate_gap = float(
            max_selection_rate - min_selection_rate
        )

        # ------------------------------------------------------------------
        # TPR / RECALL PARITY
        # ------------------------------------------------------------------

        tpr_values = {
            category: result.recall
            for category, result in valid_results.items()
            if result.recall is not None
        }

        if len(tpr_values) >= 2:
            min_tpr = float(min(tpr_values.values()))
            max_tpr = float(max(tpr_values.values()))

            tpr_gap = max_tpr - min_tpr

            tpr_parity_ratio = (
                min_tpr / max_tpr
                if max_tpr > 0
                else 1.0
            )

            passed_tpr_gate = tpr_gap <= self.max_tpr_gap
        else:
            tpr_gap = None
            tpr_parity_ratio = None
            passed_tpr_gate = False

        # ------------------------------------------------------------------
        # FPR PARITY
        # ------------------------------------------------------------------

        fpr_values = {
            category: result.false_positive_rate
            for category, result in valid_results.items()
            if result.false_positive_rate is not None
        }

        if len(fpr_values) >= 2:
            min_fpr = float(min(fpr_values.values()))
            max_fpr = float(max(fpr_values.values()))

            fpr_gap = max_fpr - min_fpr

            if min_fpr > 0:
                fpr_ratio = max_fpr / min_fpr
            elif max_fpr == 0:
                fpr_ratio = 1.0
            else:
                fpr_ratio = None

            passed_fpr_gate = (
                fpr_gap <= self.max_fpr_gap
                and (
                    fpr_ratio is None
                    or fpr_ratio <= self.max_fpr_ratio
                )
            )
        else:
            fpr_gap = None
            fpr_ratio = None
            passed_fpr_gate = False

        # ------------------------------------------------------------------
        # FNR PARITY
        # ------------------------------------------------------------------

        fnr_values = {
            category: result.false_negative_rate
            for category, result in valid_results.items()
            if result.false_negative_rate is not None
        }

        if len(fnr_values) >= 2:
            fnr_gap = float(
                max(fnr_values.values()) - min(fnr_values.values())
            )
        else:
            fnr_gap = None

        # ------------------------------------------------------------------
        # WORST-SLICE ANALYSIS
        # ------------------------------------------------------------------

        if f1_values:
            worst_slice_by_f1 = min(
                f1_values,
                key=lambda category: f1_values[category],  # type: ignore
            )
        else:
            worst_slice_by_f1 = "N/A"

        if fpr_values:
            worst_slice_by_fpr = max(
                fpr_values,
                key=lambda category: fpr_values[category],  # type: ignore
            )
        else:
            worst_slice_by_fpr = "N/A"

        if tpr_values:
            worst_slice_by_recall = min(
                tpr_values,
                key=lambda category: tpr_values[category],  # type: ignore
            )
        else:
            worst_slice_by_recall = "N/A"

        # ------------------------------------------------------------------
        # FINAL GATE
        # ------------------------------------------------------------------

        passed_all_gates = bool(
            passed_f1_gate
            and passed_disparate_impact_gate
            and passed_tpr_gate
            and passed_fpr_gate
        )

        report = FairnessReport(
            slice_name=slice_name,
            overall_f1_parity_ratio=f1_parity_ratio,
            min_f1_category=min_f1_category,
            max_f1_category=max_f1_category,
            min_parity_threshold=self.min_parity_ratio,
            passed_fairness_gate=passed_all_gates,
            category_results=category_results,
            disparate_impact_ratio=disparate_impact_ratio,
            min_selection_rate_category=min_selection_category,
            max_selection_rate_category=max_selection_category,
            tpr_gap=tpr_gap,
            tpr_parity_ratio=tpr_parity_ratio,
            fpr_gap=fpr_gap,
            fpr_ratio=fpr_ratio,
            fnr_gap=fnr_gap,
            selection_rate_gap=selection_rate_gap,
            worst_slice_by_f1=worst_slice_by_f1,
            worst_slice_by_fpr=worst_slice_by_fpr,
            worst_slice_by_recall=worst_slice_by_recall,
            evaluated_category_count=len(valid_results),
            excluded_category_count=(
                len(category_results) - len(valid_results)
            ),
            passed_f1_gate=passed_f1_gate,
            passed_disparate_impact_gate=passed_disparate_impact_gate,
            passed_tpr_gate=passed_tpr_gate,
            passed_fpr_gate=passed_fpr_gate,
            metadata={
                "module_version": MODULE_VERSION,
                "minimum_samples_per_slice": self.min_samples_per_slice,
                "min_disparate_impact": self.min_disparate_impact,
                "max_tpr_gap": self.max_tpr_gap,
                "max_fpr_gap": self.max_fpr_gap,
                "max_fpr_ratio": self.max_fpr_ratio,
                "bootstrap_samples": self.bootstrap_samples,
                "confidence_level": self.confidence_level,
            },
        )

        logger.info(
            "Fairness audit '%s': F1 parity=%s, DI=%s, TPR gap=%s, "
            "FPR gap=%s, passed=%s.",
            slice_name,
            (
                f"{f1_parity_ratio:.4f}"
                if f1_parity_ratio is not None
                else "N/A"
            ),
            f"{disparate_impact_ratio:.4f}",
            (
                f"{tpr_gap:.4f}"
                if tpr_gap is not None
                else "N/A"
            ),
            (
                f"{fpr_gap:.4f}"
                if fpr_gap is not None
                else "N/A"
            ),
            passed_all_gates,
        )

        return report


# ==============================================================================
# MULTI-SLICE EVALUATION
# ==============================================================================


def evaluate_slice_fairness(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y_true: Union[pd.Series, np.ndarray],
    slices: Mapping[str, Union[pd.Series, np.ndarray, Sequence[Any]]],
    threshold: float = 0.50,
    min_parity_ratio: float = DEFAULT_MIN_PARITY_RATIO,
    min_samples_per_slice: int = DEFAULT_MIN_SAMPLES_PER_SLICE,
    min_disparate_impact: float = 0.80,
    max_tpr_gap: float = DEFAULT_MAX_TPR_GAP,
    max_fpr_gap: float = DEFAULT_MAX_FPR_GAP,
    max_fpr_ratio: float = DEFAULT_MAX_FPR_RATIO,
    bootstrap_samples: int = 0,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate one or more slice dimensions.

    Returns
    -------
    Dict[str, Any]
        Mapping of slice names to fairness reports.

    Notes
    -----
    The model must expose ``predict_proba`` and return either:

        shape (N, 2)

    or:

        shape (N,)

    where the values represent the positive-class probability.
    """
    threshold = _validate_threshold(threshold)

    y_true_arr = _validate_binary_labels(y_true)

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "Provided model does not possess a 'predict_proba' method."
        )

    probabilities = np.asarray(model.predict_proba(X))

    if probabilities.ndim == 2:
        if probabilities.shape[0] != len(y_true_arr):
            raise ValueError(
                "Model probability output and y_true have different sample counts."
            )

        if probabilities.shape[1] < 2:
            raise ValueError(
                "2D predict_proba output must contain at least two class columns."
            )

        y_prob = probabilities[:, 1]

    elif probabilities.ndim == 1:
        if len(probabilities) != len(y_true_arr):
            raise ValueError(
                "Model probability output and y_true have different sample counts."
            )

        y_prob = probabilities

    else:
        raise ValueError(
            f"Unsupported predict_proba output shape: {probabilities.shape}"
        )

    y_prob = _validate_probability_vector(y_prob)

    y_pred = (y_prob >= threshold).astype(np.int8)

    evaluator = FairnessEvaluator(
        min_parity_ratio=min_parity_ratio,
        min_samples_per_slice=min_samples_per_slice,
        min_disparate_impact=min_disparate_impact,
        max_tpr_gap=max_tpr_gap,
        max_fpr_gap=max_fpr_gap,
        max_fpr_ratio=max_fpr_ratio,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        random_state=random_state,
    )

    results: Dict[str, Any] = {}

    for slice_name, slice_series in slices.items():
        if not slice_name:
            raise ValueError("Slice names must be non-empty.")

        report = evaluator.evaluate_slice(
            slice_name=str(slice_name),
            slice_series=slice_series,
            y_true=y_true_arr,
            y_pred=y_pred,
        )

        results[str(slice_name)] = report.to_dict()

    return results


# ==============================================================================
# AGGREGATED FAIRNESS AUDIT
# ==============================================================================


def summarize_fairness_results(
    results: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Produce a high-level summary across multiple slice dimensions.

    This is useful for MLOps deployment gates and dashboards.
    """
    if not results:
        return {
            "slice_dimension_count": 0,
            "passed_all_fairness_gates": True,
            "failed_dimensions": [],
            "worst_f1_parity_ratio": None,
            "worst_disparate_impact_ratio": None,
        }

    failed_dimensions: List[str] = []

    f1_ratios: List[float] = []
    di_ratios: List[float] = []

    for slice_name, report in results.items():
        passed = bool(report.get("passed_fairness_gate", False))

        if not passed:
            failed_dimensions.append(slice_name)

        f1_ratio = report.get("overall_f1_parity_ratio")
        if f1_ratio is not None:
            f1_ratios.append(float(f1_ratio))

        di_ratio = report.get("disparate_impact_ratio")
        if di_ratio is not None:
            di_ratios.append(float(di_ratio))

    return {
        "slice_dimension_count": len(results),
        "passed_all_fairness_gates": len(failed_dimensions) == 0,
        "failed_dimensions": failed_dimensions,
        "worst_f1_parity_ratio": (
            min(f1_ratios)
            if f1_ratios
            else None
        ),
        "worst_disparate_impact_ratio": (
            min(di_ratios)
            if di_ratios
            else None
        ),
    }


# ==============================================================================
# PUBLIC API
# ==============================================================================

__all__ = [
    "SliceMetricResult",
    "FairnessReport",
    "FairnessEvaluator",
    "evaluate_slice_fairness",
    "summarize_fairness_results",
]
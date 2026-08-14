"""
PhishGuard AI - Production Classification Metrics & Evaluation Engine
=====================================================================

Version: 14.0.0
File Path: src/evaluation/metrics.py

Purpose
-------
Production-grade binary classification evaluation for phishing detection.

Design goals
------------
- Strict input validation
- No silent metric failures
- Explicit binary-label semantics: 0=benign, 1=phishing
- Probability validation and finite-value protection
- ROC-AUC and Average Precision / PR-AUC
- Threshold-dependent operational metrics
- Threshold optimization with configurable constraints
- Cost-sensitive threshold selection
- Confusion-matrix diagnostics
- Matthews correlation coefficient
- Balanced accuracy
- NPV/FNR/FDR/FPR/specificity
- Optional bootstrap confidence intervals
- Threshold sweep reports
- JSON-safe serialization
- Deterministic evaluation where possible
- Compatibility with sklearn-like and BasePhishingModel models
- Defensive handling of small/degenerate datasets

Important
---------
Threshold optimization MUST be performed on a validation/calibration set,
not on the final untouched test set, otherwise the reported test performance
becomes optimistically biased.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger


logger = get_logger("src.evaluation.metrics")


# ==============================================================================
# CONSTANTS
# ==============================================================================

METRICS_VERSION = "14.0.0"

DEFAULT_THRESHOLD = 0.50
MIN_THRESHOLD = 0.0
MAX_THRESHOLD = 1.0

SUPPORTED_THRESHOLD_METRICS = {
    "f1",
    "f0.5",
    "f2",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "mcc",
    "accuracy",
}

EPSILON = 1e-12


# ==============================================================================
# INTERNAL VALIDATION HELPERS
# ==============================================================================

def _validate_threshold(threshold: float) -> float:
    """Validate and normalize a probability threshold."""
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a numeric value.") from exc

    if not math.isfinite(value):
        raise ValueError("threshold must be finite.")

    if not MIN_THRESHOLD <= value <= MAX_THRESHOLD:
        raise ValueError(
            f"threshold must be between {MIN_THRESHOLD} and {MAX_THRESHOLD}."
        )

    return value


def _validate_binary_labels(
    y_true: Union[pd.Series, np.ndarray, Sequence[Any]],
) -> np.ndarray:
    """
    Validate binary ground-truth labels.

    PhishGuard convention:
        0 = benign
        1 = phishing/malicious
    """
    if y_true is None:
        raise ValueError("y_true must not be None.")

    arr = np.asarray(y_true)

    if arr.ndim != 1:
        arr = arr.reshape(-1)

    if arr.size == 0:
        raise ValueError("y_true cannot be empty.")

    if pd.isna(arr).any():
        raise ValueError("y_true contains missing/NaN labels.")

    # Prevent accidental conversion of arbitrary values such as "abc" to int.
    unique = set(np.unique(arr).tolist())

    if not unique.issubset({0, 1, False, True}):
        raise ValueError(
            "y_true must contain only binary labels {0, 1}. "
            f"Observed labels: {sorted(map(str, unique))}"
        )

    return arr.astype(np.int8, copy=False)


def _validate_probabilities(
    y_prob: Union[pd.Series, np.ndarray, Sequence[float]],
    expected_length: Optional[int] = None,
) -> np.ndarray:
    """
    Validate positive-class probability scores.

    Accepted forms:
        [0.1, 0.8, 0.2]
        [[0.9, 0.1], [0.2, 0.8]]
    """
    if y_prob is None:
        raise ValueError("y_prob must not be None.")

    arr = np.asarray(y_prob, dtype=float)

    if arr.size == 0:
        raise ValueError("y_prob cannot be empty.")

    if arr.ndim == 2:
        if arr.shape[1] == 2:
            arr = arr[:, 1]
        elif arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            raise ValueError(
                "2-D y_prob must have exactly 1 or 2 columns."
            )
    elif arr.ndim != 1:
        raise ValueError(
            f"y_prob must be 1-D or 2-D. Received shape={arr.shape}."
        )

    arr = arr.reshape(-1)

    if expected_length is not None and len(arr) != expected_length:
        raise ValueError(
            f"y_prob length ({len(arr)}) does not match "
            f"y_true length ({expected_length})."
        )

    if not np.isfinite(arr).all():
        raise ValueError("y_prob contains NaN or infinite values.")

    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(
            "Probability values must lie in the inclusive range [0, 1]."
        )

    return arr.astype(np.float64, copy=False)


def _safe_divide(
    numerator: Union[int, float],
    denominator: Union[int, float],
    default: float = 0.0,
) -> float:
    """Division helper that never emits NaN/inf."""
    if denominator == 0:
        return float(default)

    result = float(numerator) / float(denominator)

    if not math.isfinite(result):
        return float(default)

    return result


def _extract_positive_probability(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """Extract positive-class probabilities from a fitted model."""
    if model is None:
        raise ValueError("model cannot be None when y_prob is not supplied.")

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"Model '{type(model).__name__}' does not expose predict_proba()."
        )

    probabilities = model.predict_proba(X)

    return _validate_probabilities(probabilities)


# ==============================================================================
# CONFIDENCE INTERVAL HELPERS
# ==============================================================================

def _bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    metric_name: str,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Optional[float]]:
    """
    Bootstrap confidence interval for a threshold-dependent metric.

    Returns None bounds when the requested metric cannot be evaluated
    for enough bootstrap samples.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be greater than zero.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    rng = np.random.default_rng(random_state)

    n = len(y_true)

    if n < 2:
        return {
            "lower": None,
            "upper": None,
            "confidence_level": confidence_level,
            "n_bootstrap": n_bootstrap,
        }

    values: List[float] = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)

        y_b = y_true[indices]
        p_b = y_prob[indices]

        # ROC-AUC and similar metrics are undefined for one-class samples.
        if metric_name == "roc_auc" and len(np.unique(y_b)) < 2:
            continue

        try:
            if metric_name == "accuracy":
                value = accuracy_score(
                    y_b,
                    (p_b >= threshold).astype(int),
                )
            elif metric_name == "precision":
                value = precision_score(
                    y_b,
                    (p_b >= threshold).astype(int),
                    zero_division=0,
                )
            elif metric_name == "recall":
                value = recall_score(
                    y_b,
                    (p_b >= threshold).astype(int),
                    zero_division=0,
                )
            elif metric_name == "f1":
                value = f1_score(
                    y_b,
                    (p_b >= threshold).astype(int),
                    zero_division=0,
                )
            elif metric_name == "roc_auc":
                value = roc_auc_score(y_b, p_b)
            else:
                continue

            if math.isfinite(float(value)):
                values.append(float(value))

        except ValueError:
            continue

    if len(values) < max(10, n_bootstrap // 20):
        return {
            "lower": None,
            "upper": None,
            "confidence_level": confidence_level,
            "n_bootstrap": n_bootstrap,
        }

    alpha = 1.0 - confidence_level

    lower = float(np.quantile(values, alpha / 2.0))
    upper = float(np.quantile(values, 1.0 - alpha / 2.0))

    return {
        "lower": lower,
        "upper": upper,
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
    }


# ==============================================================================
# DATA STRUCTURE
# ==============================================================================

@dataclass
class ClassificationMetrics:
    """
    Production binary classification metrics.

    All class-specific metrics follow:

        0 = benign
        1 = phishing/malicious
    """

    threshold: float

    accuracy: float
    balanced_accuracy: float

    precision: float
    recall: float
    f1: float
    f0_5: float
    f2: float

    specificity: float
    sensitivity: float

    fpr: float
    fnr: float
    npv: float
    fdr: float

    mcc: float

    roc_auc: Optional[float]
    pr_auc: Optional[float]
    average_precision: Optional[float]

    positive_rate: float
    prevalence: float

    tp: int
    fp: int
    tn: int
    fn: int

    total_samples: int
    positive_samples: int
    negative_samples: int

    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the metrics object to a JSON-compatible dictionary."""
        return _json_safe(asdict(self))

    def export_json(
        self,
        filepath: Union[str, Path],
        indent: int = 2,
    ) -> Path:
        """Persist metrics as JSON."""
        path = Path(filepath).expanduser().resolve()

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                self.to_dict(),
                handle,
                indent=indent,
                ensure_ascii=False,
                sort_keys=True,
            )

        logger.info(
            "Exported classification metrics report to '%s'.",
            path,
        )

        return path


# ==============================================================================
# JSON SERIALIZATION
# ==============================================================================

def _json_safe(value: Any) -> Any:
    """Recursively convert numpy/dataclass values into JSON-safe primitives."""
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return value


# ==============================================================================
# CORE METRICS
# ==============================================================================

def calculate_classification_metrics(
    model: Optional[Any] = None,
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_true: Optional[Union[pd.Series, np.ndarray]] = None,
    y_prob: Optional[Union[pd.Series, np.ndarray]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    include_bootstrap_ci: bool = False,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Calculate comprehensive binary classification metrics.

    Either:

        model + X + y_true

    or:

        y_prob + y_true

    must be supplied.

    Notes
    -----
    ROC-AUC and PR-AUC are threshold-independent.
    Precision/recall/F1/etc. depend on `threshold`.

    For production evaluation, calculate the threshold on validation data
    and evaluate the final threshold once on the untouched test set.
    """
    threshold = _validate_threshold(threshold)

    if y_true is None:
        raise ValueError("Ground-truth labels 'y_true' are required.")

    y_true_arr = _validate_binary_labels(y_true)

    if y_prob is None:
        if model is None or X is None:
            raise ValueError(
                "Provide either y_prob or both model and X."
            )

        y_prob_arr = _extract_positive_probability(model, X)
    else:
        y_prob_arr = _validate_probabilities(
            y_prob,
            expected_length=len(y_true_arr),
        )

    if len(y_prob_arr) != len(y_true_arr):
        raise ValueError(
            "y_true and y_prob must contain the same number of samples."
        )

    y_pred_arr = (y_prob_arr >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
    ).ravel()

    tn = int(tn)
    fp = int(fp)
    fn = int(fn)
    tp = int(tp)

    total = tn + fp + fn + tp

    positive_samples = int(np.sum(y_true_arr == 1))
    negative_samples = int(np.sum(y_true_arr == 0))

    accuracy = float(
        accuracy_score(y_true_arr, y_pred_arr)
    )

    balanced_accuracy = float(
        balanced_accuracy_score(y_true_arr, y_pred_arr)
    )

    precision = float(
        precision_score(
            y_true_arr,
            y_pred_arr,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_true_arr,
            y_pred_arr,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            zero_division=0,
        )
    )

    f0_5 = float(
        fbeta_score(
            y_true_arr,
            y_pred_arr,
            beta=0.5,
            zero_division=0,
        )
    )

    f2 = float(
        fbeta_score(
            y_true_arr,
            y_pred_arr,
            beta=2.0,
            zero_division=0,
        )
    )

    specificity = _safe_divide(tn, tn + fp)
    sensitivity = recall

    fpr = _safe_divide(fp, fp + tn)
    fnr = _safe_divide(fn, fn + tp)

    npv = _safe_divide(tn, tn + fn)
    fdr = _safe_divide(fp, fp + tp)

    try:
        mcc = float(
            matthews_corrcoef(
                y_true_arr,
                y_pred_arr,
            )
        )
    except ValueError:
        mcc = 0.0

    # ROC-AUC is mathematically undefined when only one class exists.
    if len(np.unique(y_true_arr)) >= 2:
        try:
            roc_auc: Optional[float] = float(
                roc_auc_score(y_true_arr, y_prob_arr)
            )
        except ValueError:
            roc_auc = None
    else:
        roc_auc = None

    # Average Precision is preferable to relying solely on trapezoidal
    # PR-AUC for imbalanced phishing datasets.
    if len(np.unique(y_true_arr)) >= 2:
        try:
            average_precision: Optional[float] = float(
                average_precision_score(
                    y_true_arr,
                    y_prob_arr,
                )
            )
        except ValueError:
            average_precision = None

        try:
            precision_curve, recall_curve, _ = precision_recall_curve(
                y_true_arr,
                y_prob_arr,
            )

            pr_auc: Optional[float] = float(
                np.trapezoid(
                    precision_curve[::-1],
                    recall_curve[::-1],
                )
            )
        except (ValueError, AttributeError):
            # Compatibility with older NumPy versions.
            try:
                precision_curve, recall_curve, _ = precision_recall_curve(
                    y_true_arr,
                    y_prob_arr,
                )
                pr_auc = float(
                    np.trapz(
                        precision_curve[::-1],
                        recall_curve[::-1],
                    )
                )
            except ValueError:
                pr_auc = None
    else:
        average_precision = None
        pr_auc = None

    prevalence = _safe_divide(
        positive_samples,
        total,
    )

    predicted_positive = int(np.sum(y_pred_arr == 1))

    positive_rate = _safe_divide(
        predicted_positive,
        total,
    )

    metadata: Dict[str, Any] = {
        "metrics_version": METRICS_VERSION,
        "positive_class": 1,
        "negative_class": 0,
        "positive_class_name": "phishing",
        "negative_class_name": "benign",
        "predicted_positive": predicted_positive,
        "predicted_negative": int(total - predicted_positive),
    }

    if include_bootstrap_ci:
        logger.info(
            "Calculating bootstrap confidence intervals "
            "(n=%d, confidence=%.3f).",
            n_bootstrap,
            confidence_level,
        )

        metadata["confidence_intervals"] = {
            "accuracy": _bootstrap_metric_ci(
                y_true_arr,
                y_prob_arr,
                threshold,
                "accuracy",
                n_bootstrap,
                confidence_level,
                random_state,
            ),
            "precision": _bootstrap_metric_ci(
                y_true_arr,
                y_prob_arr,
                threshold,
                "precision",
                n_bootstrap,
                confidence_level,
                random_state + 1,
            ),
            "recall": _bootstrap_metric_ci(
                y_true_arr,
                y_prob_arr,
                threshold,
                "recall",
                n_bootstrap,
                confidence_level,
                random_state + 2,
            ),
            "f1": _bootstrap_metric_ci(
                y_true_arr,
                y_prob_arr,
                threshold,
                "f1",
                n_bootstrap,
                confidence_level,
                random_state + 3,
            ),
            "roc_auc": _bootstrap_metric_ci(
                y_true_arr,
                y_prob_arr,
                threshold,
                "roc_auc",
                n_bootstrap,
                confidence_level,
                random_state + 4,
            ),
        }

    result = ClassificationMetrics(
        threshold=threshold,

        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,

        precision=precision,
        recall=recall,
        f1=f1,
        f0_5=f0_5,
        f2=f2,

        specificity=specificity,
        sensitivity=sensitivity,

        fpr=fpr,
        fnr=fnr,
        npv=npv,
        fdr=fdr,

        mcc=mcc,

        roc_auc=roc_auc,
        pr_auc=pr_auc,
        average_precision=average_precision,

        positive_rate=positive_rate,
        prevalence=prevalence,

        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,

        total_samples=total,
        positive_samples=positive_samples,
        negative_samples=negative_samples,

        additional_metadata=metadata,
    )

    return result.to_dict()


# ==============================================================================
# CONFUSION MATRIX REPORT
# ==============================================================================

def generate_confusion_matrix_report(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate an operational confusion-matrix report.

    Default labels:
        Benign
        Phishing
    """
    y_true_arr = _validate_binary_labels(y_true)
    y_pred_arr = _validate_binary_labels(y_pred)

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            "y_true and y_pred must contain the same number of samples."
        )

    if labels is None:
        labels = ["Benign", "Phishing"]

    if len(labels) != 2:
        raise ValueError(
            "labels must contain exactly two class names."
        )

    cm = confusion_matrix(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    total = int(cm.sum())

    actual_positive = int(tp + fn)
    actual_negative = int(tn + fp)

    predicted_positive = int(tp + fp)
    predicted_negative = int(tn + fn)

    return {
        "metrics_version": METRICS_VERSION,
        "class_labels": {
            "negative": labels[0],
            "positive": labels[1],
        },
        "label_mapping": {
            "0": labels[0],
            "1": labels[1],
        },
        "raw_matrix": cm.tolist(),
        "matrix_layout": [
            [labels[0] + "_predicted", labels[1] + "_predicted"],
            [labels[0] + "_actual", labels[1] + "_actual"],
        ],
        "breakdown": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "support": {
            "total": total,
            "actual_positive": actual_positive,
            "actual_negative": actual_negative,
            "predicted_positive": predicted_positive,
            "predicted_negative": predicted_negative,
        },
        "rates": {
            "tn_rate": _safe_divide(tn, total),
            "fp_rate": _safe_divide(fp, total),
            "fn_rate": _safe_divide(fn, total),
            "tp_rate": _safe_divide(tp, total),
            "specificity": _safe_divide(tn, tn + fp),
            "sensitivity": _safe_divide(tp, tp + fn),
            "fpr": _safe_divide(fp, fp + tn),
            "fnr": _safe_divide(fn, fn + tp),
            "precision": _safe_divide(tp, tp + fp),
            "npv": _safe_divide(tn, tn + fn),
        },
    }


# ==============================================================================
# THRESHOLD METRIC EVALUATION
# ==============================================================================

def _calculate_threshold_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    metric: str,
) -> float:
    """Calculate one threshold-dependent metric."""
    predictions = (y_prob >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    metric = metric.lower().strip()

    if metric == "f1":
        return float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        )

    if metric in {"f0.5", "f0_5"}:
        return float(
            fbeta_score(
                y_true,
                predictions,
                beta=0.5,
                zero_division=0,
            )
        )

    if metric == "f2":
        return float(
            fbeta_score(
                y_true,
                predictions,
                beta=2.0,
                zero_division=0,
            )
        )

    if metric == "precision":
        return float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        )

    if metric == "recall":
        return float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        )

    if metric == "specificity":
        return _safe_divide(tn, tn + fp)

    if metric == "balanced_accuracy":
        return float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        )

    if metric == "mcc":
        try:
            return float(
                matthews_corrcoef(
                    y_true,
                    predictions,
                )
            )
        except ValueError:
            return 0.0

    if metric == "accuracy":
        return float(
            accuracy_score(
                y_true,
                predictions,
            )
        )

    raise ValueError(
        f"Unsupported threshold metric '{metric}'. "
        f"Supported metrics: {sorted(SUPPORTED_THRESHOLD_METRICS)}"
    )


# ==============================================================================
# THRESHOLD OPTIMIZATION
# ==============================================================================

def find_optimal_threshold(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    metric: str = "f1",
    target_max_fpr: Optional[float] = None,
    min_recall: Optional[float] = None,
    min_precision: Optional[float] = None,
    min_specificity: Optional[float] = None,
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """
    Find an operational decision threshold.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.

    y_prob:
        Positive-class probabilities.

    metric:
        Metric to maximize.

    target_max_fpr:
        Optional maximum permitted false-positive rate.

    min_recall:
        Optional minimum recall constraint.

    min_precision:
        Optional minimum precision constraint.

    min_specificity:
        Optional minimum specificity constraint.

    thresholds:
        Optional explicit candidate threshold list.

    Returns
    -------
    (threshold, score)

    Important
    ---------
    This function should normally operate on a validation/calibration dataset,
    NOT on the final test set.
    """
    y_true_arr = _validate_binary_labels(y_true)

    y_prob_arr = _validate_probabilities(
        y_prob,
        expected_length=len(y_true_arr),
    )

    metric = metric.lower().strip()

    if metric not in SUPPORTED_THRESHOLD_METRICS:
        raise ValueError(
            f"Unsupported metric '{metric}'. "
            f"Supported metrics: {sorted(SUPPORTED_THRESHOLD_METRICS)}"
        )

    if target_max_fpr is not None:
        target_max_fpr = float(target_max_fpr)

        if not 0.0 <= target_max_fpr <= 1.0:
            raise ValueError(
                "target_max_fpr must be between 0 and 1."
            )

    for name, value in (
        ("min_recall", min_recall),
        ("min_precision", min_precision),
        ("min_specificity", min_specificity),
    ):
        if value is not None:
            value = float(value)

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

    if thresholds is None:
        # Include unique observed probability values because those are the
        # only points where classification actually changes.
        observed = np.unique(y_prob_arr)

        candidate_thresholds = np.concatenate(
            [
                np.array([0.0, DEFAULT_THRESHOLD, 1.0]),
                observed,
            ]
        )

        candidate_thresholds = np.unique(
            np.clip(candidate_thresholds, 0.0, 1.0)
        )
    else:
        candidate_thresholds = np.asarray(
            list(thresholds),
            dtype=float,
        )

        if candidate_thresholds.size == 0:
            raise ValueError("thresholds cannot be empty.")

        if not np.isfinite(candidate_thresholds).all():
            raise ValueError("thresholds contains NaN or infinite values.")

        if np.any(candidate_thresholds < 0) or np.any(
            candidate_thresholds > 1
        ):
            raise ValueError(
                "All threshold candidates must lie between 0 and 1."
            )

        candidate_thresholds = np.unique(candidate_thresholds)

    best_threshold: Optional[float] = None
    best_score = -np.inf

    # Deterministic tie-breaking:
    # when two thresholds achieve exactly the same score, prefer the higher
    # threshold because it is generally more conservative for phishing alerts.
    for threshold in candidate_thresholds:
        threshold = float(threshold)

        predictions = (y_prob_arr >= threshold).astype(np.int8)

        tn, fp, fn, tp = confusion_matrix(
            y_true_arr,
            predictions,
            labels=[0, 1],
        ).ravel()

        fpr = _safe_divide(fp, fp + tn)
        recall = _safe_divide(tp, tp + fn)
        precision = _safe_divide(tp, tp + fp)
        specificity = _safe_divide(tn, tn + fp)

        if target_max_fpr is not None and fpr > target_max_fpr:
            continue

        if min_recall is not None and recall < min_recall:
            continue

        if min_precision is not None and precision < min_precision:
            continue

        if min_specificity is not None and specificity < min_specificity:
            continue

        score = _calculate_threshold_metric(
            y_true_arr,
            y_prob_arr,
            threshold,
            metric,
        )

        if (
            score > best_score + EPSILON
            or (
                abs(score - best_score) <= EPSILON
                and (
                    best_threshold is None
                    or threshold > best_threshold
                )
            )
        ):
            best_score = score
            best_threshold = threshold

    if best_threshold is None:
        raise RuntimeError(
            "No candidate threshold satisfies the supplied operational "
            "constraints. Relax the constraints or provide more validation data."
        )

    logger.info(
        "Optimal threshold discovered: %.6f | metric=%s | score=%.6f",
        best_threshold,
        metric,
        best_score,
    )

    return float(best_threshold), float(best_score)


# ==============================================================================
# COST-SENSITIVE THRESHOLD OPTIMIZATION
# ==============================================================================

def find_cost_optimal_threshold(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 5.0,
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """
    Find a threshold minimizing expected classification error cost.

    Cost model:

        total_cost =
            FP * false_positive_cost
            +
            FN * false_negative_cost

    This is particularly useful for phishing detection where missing a
    malicious email may be substantially more expensive than generating
    an additional alert.
    """
    if false_positive_cost < 0:
        raise ValueError("false_positive_cost cannot be negative.")

    if false_negative_cost < 0:
        raise ValueError("false_negative_cost cannot be negative.")

    y_true_arr = _validate_binary_labels(y_true)

    y_prob_arr = _validate_probabilities(
        y_prob,
        expected_length=len(y_true_arr),
    )

    if thresholds is None:
        candidate_thresholds = np.unique(
            np.concatenate(
                [
                    np.array([0.0, DEFAULT_THRESHOLD, 1.0]),
                    y_prob_arr,
                ]
            )
        )
    else:
        candidate_thresholds = np.asarray(
            list(thresholds),
            dtype=float,
        )

        if candidate_thresholds.size == 0:
            raise ValueError("thresholds cannot be empty.")

        if not np.isfinite(candidate_thresholds).all():
            raise ValueError("thresholds contains non-finite values.")

    best_threshold = 0.50
    best_cost = float("inf")

    for threshold in candidate_thresholds:
        threshold = _validate_threshold(float(threshold))

        predictions = (y_prob_arr >= threshold).astype(np.int8)

        tn, fp, fn, tp = confusion_matrix(
            y_true_arr,
            predictions,
            labels=[0, 1],
        ).ravel()

        cost = (
            float(fp) * float(false_positive_cost)
            + float(fn) * float(false_negative_cost)
        )

        if (
            cost < best_cost - EPSILON
            or (
                abs(cost - best_cost) <= EPSILON
                and threshold > best_threshold
            )
        ):
            best_cost = cost
            best_threshold = threshold

    logger.info(
        "Cost-optimal threshold discovered: %.6f | cost=%.4f "
        "| FP_cost=%.4f | FN_cost=%.4f",
        best_threshold,
        best_cost,
        false_positive_cost,
        false_negative_cost,
    )

    return float(best_threshold), float(best_cost)


# ==============================================================================
# THRESHOLD SWEEP
# ==============================================================================

def generate_threshold_report(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    thresholds: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Generate a threshold-performance table.

    Useful for:
    - Security policy selection
    - ROC/PR operating point analysis
    - SOC alert-volume analysis
    - Model governance
    """
    y_true_arr = _validate_binary_labels(y_true)

    y_prob_arr = _validate_probabilities(
        y_prob,
        expected_length=len(y_true_arr),
    )

    if thresholds is None:
        thresholds_arr = np.linspace(
            0.01,
            0.99,
            99,
        )
    else:
        thresholds_arr = np.asarray(
            list(thresholds),
            dtype=float,
        )

    rows: List[Dict[str, Any]] = []

    for threshold in thresholds_arr:
        threshold = _validate_threshold(float(threshold))

        predictions = (y_prob_arr >= threshold).astype(np.int8)

        tn, fp, fn, tp = confusion_matrix(
            y_true_arr,
            predictions,
            labels=[0, 1],
        ).ravel()

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        specificity = _safe_divide(tn, tn + fp)

        f1 = float(
            f1_score(
                y_true_arr,
                predictions,
                zero_division=0,
            )
        )

        rows.append(
            {
                "threshold": threshold,
                "accuracy": float(
                    accuracy_score(
                        y_true_arr,
                        predictions,
                    )
                ),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "fpr": _safe_divide(fp, fp + tn),
                "fnr": _safe_divide(fn, fn + tp),
                "f1": f1,
                "mcc": float(
                    matthews_corrcoef(
                        y_true_arr,
                        predictions,
                    )
                ),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "alert_rate": _safe_divide(
                    int(np.sum(predictions == 1)),
                    len(predictions),
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by="threshold"
    ).reset_index(drop=True)


# ==============================================================================
# ROC CURVE DATA
# ==============================================================================

def generate_roc_curve_report(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
) -> Dict[str, Any]:
    """
    Generate serializable ROC curve coordinates.
    """
    y_true_arr = _validate_binary_labels(y_true)

    y_prob_arr = _validate_probabilities(
        y_prob,
        expected_length=len(y_true_arr),
    )

    if len(np.unique(y_true_arr)) < 2:
        return {
            "available": False,
            "reason": "ROC curve requires both positive and negative classes.",
            "fpr": [],
            "tpr": [],
            "thresholds": [],
            "roc_auc": None,
        }

    fpr, tpr, thresholds = roc_curve(
        y_true_arr,
        y_prob_arr,
    )

    return {
        "available": True,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "roc_auc": float(
            roc_auc_score(
                y_true_arr,
                y_prob_arr,
            )
        ),
    }


# ==============================================================================
# PRECISION-RECALL CURVE DATA
# ==============================================================================

def generate_precision_recall_report(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
) -> Dict[str, Any]:
    """
    Generate serializable precision-recall curve coordinates.
    """
    y_true_arr = _validate_binary_labels(y_true)

    y_prob_arr = _validate_probabilities(
        y_prob,
        expected_length=len(y_true_arr),
    )

    if len(np.unique(y_true_arr)) < 2:
        return {
            "available": False,
            "reason": (
                "Precision-recall diagnostics require both positive "
                "and negative classes."
            ),
            "precision": [],
            "recall": [],
            "thresholds": [],
            "average_precision": None,
        }

    precision, recall, thresholds = precision_recall_curve(
        y_true_arr,
        y_prob_arr,
    )

    return {
        "available": True,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "average_precision": float(
            average_precision_score(
                y_true_arr,
                y_prob_arr,
            )
        ),
        "positive_prevalence": float(
            np.mean(y_true_arr)
        ),
    }


# ==============================================================================
# PUBLIC EXPORT CONTRACT
# ==============================================================================

__all__ = [
    "ClassificationMetrics",
    "calculate_classification_metrics",
    "generate_confusion_matrix_report",
    "find_optimal_threshold",
    "find_cost_optimal_threshold",
    "generate_threshold_report",
    "generate_roc_curve_report",
    "generate_precision_recall_report",
]
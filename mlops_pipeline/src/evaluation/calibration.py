"""
PhishGuard AI - Probability Calibration & Reliability Analytics
===============================================================

Version: 13.1.0
File Path: src/evaluation/calibration.py

Production-oriented probability calibration evaluation for binary phishing
classification models.

Security / MLOps goals
----------------------
- Strict binary-label and probability validation.
- Reject NaN / +/-Inf prediction scores.
- Prevent silent length mismatches.
- Support uniform and quantile reliability bins.
- Calculate ECE, MCE, Brier score, Brier skill score and log loss.
- Calculate calibration intercept and slope.
- Produce explicit bin-level reliability statistics.
- Detect statistically weak / under-populated bins.
- Optional deterministic bootstrap confidence intervals.
- Support direct probabilities or model.predict_proba().
- Produce JSON-safe reports suitable for artifact registration.
- Never mutate caller-owned data.
- Avoid logging sensitive email/URL/attachment contents.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss

from src.utils.logger import get_logger

logger = get_logger("src.evaluation.calibration")

VERSION = "13.1.0"

ArrayLike = Union[pd.Series, np.ndarray, Sequence[float]]
LabelLike = Union[pd.Series, np.ndarray, Sequence[int]]
FeatureMatrix = Union[pd.DataFrame, np.ndarray]


# ==============================================================================
# VALIDATION HELPERS
# ==============================================================================

def _validate_probability_vector(
    y_prob: ArrayLike,
    *,
    name: str = "y_prob",
) -> np.ndarray:
    """
    Convert probabilities into a validated one-dimensional float array.

    Raises
    ------
    ValueError
        If probabilities are empty, non-finite, non-numeric, or outside [0, 1].
    """
    try:
        arr = np.asarray(y_prob, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric probability values.") from exc

    if arr.ndim != 1:
        arr = arr.reshape(-1)

    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    if np.any((arr < 0.0) | (arr > 1.0)):
        min_value = float(np.min(arr))
        max_value = float(np.max(arr))
        raise ValueError(
            f"{name} must contain probabilities in [0, 1]. "
            f"Observed range [{min_value:.6f}, {max_value:.6f}]."
        )

    return arr


def _validate_binary_labels(
    y_true: LabelLike,
    *,
    name: str = "y_true",
) -> np.ndarray:
    """
    Convert and strictly validate binary labels.

    Only 0 and 1 are accepted.
    """
    try:
        arr_raw = np.asarray(y_true)
    except Exception as exc:
        raise ValueError(f"{name} could not be converted to an array.") from exc

    if arr_raw.ndim != 1:
        arr_raw = arr_raw.reshape(-1)

    if arr_raw.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if pd.isna(arr_raw).any():
        raise ValueError(f"{name} contains missing values.")

    try:
        arr_float = arr_raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain binary numeric labels 0 and 1."
        ) from exc

    if not np.all(np.isfinite(arr_float)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    if not np.all(np.isin(arr_float, [0.0, 1.0])):
        unique_values = np.unique(arr_float)
        raise ValueError(
            f"{name} must contain only binary labels 0 and 1. "
            f"Observed values: {unique_values.tolist()}"
        )

    return arr_float.astype(np.int8)


def _validate_inputs(
    y_true: LabelLike,
    y_prob: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and align binary labels and probabilities."""
    labels = _validate_binary_labels(y_true)
    probabilities = _validate_probability_vector(y_prob)

    if len(labels) != len(probabilities):
        raise ValueError(
            "Length mismatch between y_true and y_prob: "
            f"{len(labels)} != {len(probabilities)}."
        )

    return labels, probabilities


def _validate_n_bins(n_bins: int) -> int:
    """Validate reliability-bin count."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
        raise TypeError("n_bins must be an integer.")

    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")

    if n_bins > 100:
        raise ValueError("n_bins cannot exceed 100.")

    return int(n_bins)


def _validate_strategy(strategy: str) -> str:
    """Validate calibration binning strategy."""
    normalized = str(strategy).strip().lower()

    if normalized not in {"uniform", "quantile"}:
        raise ValueError(
            f"Invalid binning strategy '{strategy}'. "
            "Supported strategies: 'uniform', 'quantile'."
        )

    return normalized


def _validate_threshold(
    threshold: float,
    *,
    name: str = "threshold",
) -> float:
    """Validate a probability threshold."""
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc

    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1].")

    return value


# ==============================================================================
# JSON SERIALIZATION HELPERS
# ==============================================================================

def _json_safe(value: Any) -> Any:
    """
    Convert NumPy / pandas / Path values into JSON-safe primitives.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value_float = float(value)
        return value_float if math.isfinite(value_float) else None

    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return value


# ==============================================================================
# RELIABILITY BIN
# ==============================================================================

@dataclass
class ReliabilityBin:
    """
    Represents one probability calibration bin.

    Empty bins are explicitly represented with ``None`` statistics rather than
    fabricated accuracy/confidence values.
    """

    bin_index: int
    lower_bound: float
    upper_bound: float
    sample_count: int
    sample_fraction: float
    mean_predicted_probability: Optional[float]
    empirical_positive_rate: Optional[float]
    absolute_calibration_error: Optional[float]
    signed_calibration_error: Optional[float]
    min_probability: Optional[float]
    max_probability: Optional[float]

    @property
    def is_empty(self) -> bool:
        return self.sample_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


# ==============================================================================
# CALIBRATION REPORT
# ==============================================================================

@dataclass
class CalibrationReport:
    """
    Complete probability calibration diagnostic report.

    The report is intentionally serializable so it can be persisted beside
    trained model artifacts.
    """

    ece: float
    mce: float
    brier_score: float
    brier_skill_score: float
    log_loss: float

    calibration_intercept: Optional[float]
    calibration_slope: Optional[float]

    n_bins: int
    bin_strategy: str

    prob_true: List[Optional[float]]
    prob_pred: List[Optional[float]]
    bin_counts: List[int]

    reliability_bins: List[Dict[str, Any]]

    is_well_calibrated: bool

    positive_rate: float
    total_samples: int
    positive_samples: int
    negative_samples: int

    confidence_interval: Optional[Dict[str, Any]] = None

    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return _json_safe(asdict(self))

    def export_json(
        self,
        filepath: Union[str, Path],
        *,
        indent: int = 2,
    ) -> Path:
        """
        Persist calibration report as JSON.

        Uses UTF-8 and rejects non-finite JSON values.
        """
        path = Path(filepath).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = self.to_dict()

        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=indent,
                ensure_ascii=False,
                allow_nan=False,
            )

        logger.info("Exported calibration report to '%s'.", path)
        return path


# ==============================================================================
# BIN CONSTRUCTION
# ==============================================================================

def _build_bin_edges(
    y_prob: np.ndarray,
    n_bins: int,
    strategy: str,
) -> np.ndarray:
    """
    Build deterministic probability bin edges.

    Quantile edges are deduplicated because repeated model scores can produce
    zero-width quantile bins.
    """
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(y_prob, quantiles)

    edges = np.unique(edges)

    # Ensure probability-space boundaries remain explicit.
    if edges[0] > 0.0:
        edges = np.insert(edges, 0, 0.0)

    if edges[-1] < 1.0:
        edges = np.append(edges, 1.0)

    return edges


def _calculate_reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int,
    strategy: str,
) -> Tuple[List[ReliabilityBin], np.ndarray]:
    """
    Construct reliability statistics for every probability bin.
    """
    edges = _build_bin_edges(y_prob, n_bins, strategy)

    actual_bin_count = len(edges) - 1
    sample_count = len(y_true)

    # searchsorted with right=True allows p=1.0 to fall in the final bin.
    bin_ids = np.searchsorted(edges, y_prob, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, actual_bin_count - 1)

    bins: List[ReliabilityBin] = []

    for index in range(actual_bin_count):
        mask = bin_ids == index
        count = int(np.sum(mask))

        lower = float(edges[index])
        upper = float(edges[index + 1])

        if count == 0:
            bins.append(
                ReliabilityBin(
                    bin_index=index,
                    lower_bound=lower,
                    upper_bound=upper,
                    sample_count=0,
                    sample_fraction=0.0,
                    mean_predicted_probability=None,
                    empirical_positive_rate=None,
                    absolute_calibration_error=None,
                    signed_calibration_error=None,
                    min_probability=None,
                    max_probability=None,
                )
            )
            continue

        probabilities = y_prob[mask]
        labels = y_true[mask]

        confidence = float(np.mean(probabilities))
        observed_rate = float(np.mean(labels))
        signed_error = float(observed_rate - confidence)

        bins.append(
            ReliabilityBin(
                bin_index=index,
                lower_bound=lower,
                upper_bound=upper,
                sample_count=count,
                sample_fraction=float(count / sample_count),
                mean_predicted_probability=confidence,
                empirical_positive_rate=observed_rate,
                absolute_calibration_error=float(abs(signed_error)),
                signed_calibration_error=signed_error,
                min_probability=float(np.min(probabilities)),
                max_probability=float(np.max(probabilities)),
            )
        )

    return bins, edges


# ==============================================================================
# ECE / MCE
# ==============================================================================

def calculate_expected_calibration_error(
    y_true: LabelLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[float, float, List[Optional[float]], List[Optional[float]], List[int]]:
    """
    Calculate Expected Calibration Error (ECE) and Maximum Calibration Error.

    ECE
        Weighted mean of absolute calibration error.

    MCE
        Maximum absolute calibration error among non-empty bins.

    Empty bins are excluded from MCE and contribute zero to ECE.

    Returns
    -------
    tuple
        ``(ece, mce, prob_true, prob_pred, bin_counts)``
    """
    labels, probabilities = _validate_inputs(y_true, y_prob)
    n_bins = _validate_n_bins(n_bins)
    strategy = _validate_strategy(strategy)

    bins, _ = _calculate_reliability_bins(
        labels,
        probabilities,
        n_bins=n_bins,
        strategy=strategy,
    )

    ece = 0.0
    mce = 0.0

    prob_true: List[Optional[float]] = []
    prob_pred: List[Optional[float]] = []
    counts: List[int] = []

    for calibration_bin in bins:
        count = calibration_bin.sample_count

        prob_true.append(calibration_bin.empirical_positive_rate)
        prob_pred.append(calibration_bin.mean_predicted_probability)
        counts.append(count)

        if count == 0:
            continue

        error = calibration_bin.absolute_calibration_error or 0.0

        ece += calibration_bin.sample_fraction * error
        mce = max(mce, error)

    return (
        float(ece),
        float(mce),
        prob_true,
        prob_pred,
        counts,
    )


# ==============================================================================
# CALIBRATION INTERCEPT / SLOPE
# ==============================================================================

def _calculate_calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate calibration intercept and slope using logistic recalibration.

    A perfectly calibrated model approximately has:
        intercept = 0
        slope = 1

    Probability values are clipped only for the logit transformation.
    """
    if len(np.unique(y_true)) < 2:
        return None, None

    eps = np.finfo(float).eps
    clipped = np.clip(y_prob, eps, 1.0 - eps)

    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    try:
        recalibration_model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=1000,
        )
    except TypeError:
        # Compatibility with older sklearn versions.
        recalibration_model = LogisticRegression(
            penalty="none",
            solver="lbfgs",
            max_iter=1000,
        )

    try:
        recalibration_model.fit(logits, y_true)

        intercept = float(recalibration_model.intercept_[0])
        slope = float(recalibration_model.coef_[0][0])

        return intercept, slope

    except Exception as exc:
        logger.warning(
            "Unable to calculate calibration intercept/slope: %s",
            exc,
        )
        return None, None


# ==============================================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ==============================================================================

def _bootstrap_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int,
    strategy: str,
    n_bootstrap: int,
    random_state: int,
    confidence_level: float,
) -> Dict[str, Any]:
    """
    Estimate a percentile bootstrap confidence interval for ECE.

    This is intentionally optional because bootstrapping can be expensive on
    very large evaluation datasets.
    """
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >= 100.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    rng = np.random.default_rng(random_state)

    n_samples = len(y_true)
    bootstrap_scores = np.empty(n_bootstrap, dtype=float)

    for index in range(n_bootstrap):
        sample_indices = rng.integers(
            0,
            n_samples,
            size=n_samples,
        )

        sampled_y = y_true[sample_indices]
        sampled_p = y_prob[sample_indices]

        ece, _, _, _, _ = calculate_expected_calibration_error(
            sampled_y,
            sampled_p,
            n_bins=n_bins,
            strategy=strategy,
        )

        bootstrap_scores[index] = ece

    alpha = 1.0 - confidence_level

    lower = float(np.quantile(bootstrap_scores, alpha / 2.0))
    upper = float(np.quantile(bootstrap_scores, 1.0 - alpha / 2.0))

    return {
        "metric": "ece",
        "method": "percentile_bootstrap",
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
        "random_state": random_state,
        "lower": lower,
        "upper": upper,
    }


# ==============================================================================
# CALIBRATION EVALUATOR
# ==============================================================================

class CalibrationEvaluator:
    """
    Stateful calibration evaluator.

    This class is suitable for both offline model evaluation and MLOps
    validation gates.
    """

    def __init__(
        self,
        n_bins: int = 10,
        strategy: str = "uniform",
        ece_threshold: float = 0.05,
        min_samples_per_bin: int = 20,
        bootstrap_iterations: int = 0,
        bootstrap_confidence_level: float = 0.95,
        random_state: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        n_bins:
            Number of reliability bins.

        strategy:
            ``uniform`` or ``quantile``.

        ece_threshold:
            Maximum ECE accepted by the calibration quality gate.

        min_samples_per_bin:
            Minimum recommended observations per populated bin. This does not
            invalidate the report; it produces a warning.

        bootstrap_iterations:
            Number of bootstrap samples. ``0`` disables bootstrap CIs.

        bootstrap_confidence_level:
            Confidence level for bootstrap ECE interval.

        random_state:
            Deterministic bootstrap seed.
        """
        self.n_bins = _validate_n_bins(n_bins)
        self.strategy = _validate_strategy(strategy)

        if not math.isfinite(float(ece_threshold)):
            raise ValueError("ece_threshold must be finite.")

        if not 0.0 <= float(ece_threshold) <= 1.0:
            raise ValueError("ece_threshold must be within [0, 1].")

        if isinstance(min_samples_per_bin, bool) or min_samples_per_bin < 1:
            raise ValueError("min_samples_per_bin must be >= 1.")

        if bootstrap_iterations < 0:
            raise ValueError("bootstrap_iterations cannot be negative.")

        if bootstrap_iterations and bootstrap_iterations < 100:
            raise ValueError(
                "bootstrap_iterations must be 0 or >= 100."
            )

        if not 0.0 < bootstrap_confidence_level < 1.0:
            raise ValueError(
                "bootstrap_confidence_level must be between 0 and 1."
            )

        self.ece_threshold = float(ece_threshold)
        self.min_samples_per_bin = int(min_samples_per_bin)
        self.bootstrap_iterations = int(bootstrap_iterations)
        self.bootstrap_confidence_level = float(
            bootstrap_confidence_level
        )
        self.random_state = int(random_state)

    def evaluate(
        self,
        y_true: LabelLike,
        y_prob: ArrayLike,
    ) -> CalibrationReport:
        """
        Run a complete calibration audit.
        """
        labels, probabilities = _validate_inputs(y_true, y_prob)

        ece, mce, prob_true, prob_pred, bin_counts = (
            calculate_expected_calibration_error(
                labels,
                probabilities,
                n_bins=self.n_bins,
                strategy=self.strategy,
            )
        )

        # ------------------------------------------------------------------
        # Brier score
        # ------------------------------------------------------------------

        brier = float(
            brier_score_loss(
                labels,
                probabilities,
            )
        )

        # ------------------------------------------------------------------
        # Brier skill score
        #
        # Reference model = climatology / empirical positive prevalence.
        # ------------------------------------------------------------------

        prevalence = float(np.mean(labels))

        reference_brier = float(
            brier_score_loss(
                labels,
                np.full_like(
                    probabilities,
                    prevalence,
                    dtype=float,
                ),
            )
        )

        if reference_brier > 0.0:
            brier_skill_score = float(
                1.0 - (brier / reference_brier)
            )
        else:
            brier_skill_score = 0.0

        # ------------------------------------------------------------------
        # Log loss
        # ------------------------------------------------------------------

        try:
            model_log_loss = float(
                log_loss(
                    labels,
                    probabilities,
                    labels=[0, 1],
                )
            )
        except ValueError:
            model_log_loss = float("nan")

        # ------------------------------------------------------------------
        # Calibration slope / intercept
        # ------------------------------------------------------------------

        intercept, slope = _calculate_calibration_slope_intercept(
            labels,
            probabilities,
        )

        # ------------------------------------------------------------------
        # Reliability bins
        # ------------------------------------------------------------------

        reliability_bins, _ = _calculate_reliability_bins(
            labels,
            probabilities,
            n_bins=self.n_bins,
            strategy=self.strategy,
        )

        # ------------------------------------------------------------------
        # Quality warnings
        # ------------------------------------------------------------------

        warnings: List[str] = []

        if len(np.unique(labels)) < 2:
            warnings.append(
                "Evaluation set contains only one class; calibration "
                "diagnostics are statistically unreliable."
            )

        populated_bins = [
            item for item in reliability_bins
            if item.sample_count > 0
        ]

        underpopulated_bins = [
            item for item in populated_bins
            if item.sample_count < self.min_samples_per_bin
        ]

        if underpopulated_bins:
            warnings.append(
                f"{len(underpopulated_bins)} populated calibration bins "
                f"contain fewer than {self.min_samples_per_bin} samples."
            )

        empty_bins = [
            item for item in reliability_bins
            if item.sample_count == 0
        ]

        if empty_bins:
            warnings.append(
                f"{len(empty_bins)} calibration bins are empty."
            )

        if prevalence <= 0.01 or prevalence >= 0.99:
            warnings.append(
                "Evaluation set has extreme class prevalence; calibration "
                "metrics should be interpreted with additional caution."
            )

        if ece > self.ece_threshold:
            warnings.append(
                f"ECE {ece:.6f} exceeds configured threshold "
                f"{self.ece_threshold:.6f}."
            )

        # ------------------------------------------------------------------
        # Optional bootstrap CI
        # ------------------------------------------------------------------

        confidence_interval: Optional[Dict[str, Any]] = None

        if self.bootstrap_iterations > 0:
            confidence_interval = _bootstrap_ece(
                labels,
                probabilities,
                n_bins=self.n_bins,
                strategy=self.strategy,
                n_bootstrap=self.bootstrap_iterations,
                random_state=self.random_state,
                confidence_level=self.bootstrap_confidence_level,
            )

        # ------------------------------------------------------------------
        # Report
        # ------------------------------------------------------------------

        report = CalibrationReport(
            ece=ece,
            mce=mce,
            brier_score=brier,
            brier_skill_score=brier_skill_score,
            log_loss=model_log_loss,
            calibration_intercept=intercept,
            calibration_slope=slope,
            n_bins=len(reliability_bins),
            bin_strategy=self.strategy,
            prob_true=prob_true,
            prob_pred=prob_pred,
            bin_counts=bin_counts,
            reliability_bins=[
                item.to_dict()
                for item in reliability_bins
            ],
            is_well_calibrated=(
                ece <= self.ece_threshold
                and len(underpopulated_bins) == 0
            ),
            positive_rate=prevalence,
            total_samples=len(labels),
            positive_samples=int(np.sum(labels == 1)),
            negative_samples=int(np.sum(labels == 0)),
            confidence_interval=confidence_interval,
            warnings=warnings,
            metadata={
                "module_version": VERSION,
                "ece_threshold": self.ece_threshold,
                "min_samples_per_bin": self.min_samples_per_bin,
                "bootstrap_enabled": self.bootstrap_iterations > 0,
                "random_state": self.random_state,
            },
        )

        logger.info(
            "Calibration evaluation completed: "
            "ECE=%.6f MCE=%.6f Brier=%.6f BSS=%.6f LogLoss=%.6f "
            "Samples=%d",
            ece,
            mce,
            brier,
            brier_skill_score,
            model_log_loss,
            len(labels),
        )

        if warnings:
            logger.warning(
                "Calibration evaluation generated %d warning(s).",
                len(warnings),
            )

        return report


# ==============================================================================
# MODEL PROBABILITY EXTRACTION
# ==============================================================================

def _extract_positive_class_probability(
    model: Any,
    X: FeatureMatrix,
) -> np.ndarray:
    """
    Safely extract positive-class probabilities from a fitted estimator.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "Provided model does not expose predict_proba()."
        )

    probabilities = model.predict_proba(X)

    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim == 1:
        return _validate_probability_vector(probabilities)

    if probabilities.ndim != 2:
        raise ValueError(
            "predict_proba() must return a 1-D or 2-D probability array."
        )

    if probabilities.shape[1] == 2:
        return _validate_probability_vector(
            probabilities[:, 1]
        )

    # A single-column probability matrix can represent a positive-class
    # probability for some custom estimators.
    if probabilities.shape[1] == 1:
        return _validate_probability_vector(
            probabilities[:, 0]
        )

    raise ValueError(
        "Expected binary classifier probabilities with one or two columns; "
        f"received shape {probabilities.shape}."
    )


# ==============================================================================
# HIGH-LEVEL FUNCTIONAL API
# ==============================================================================

def evaluate_calibration(
    model: Optional[Any] = None,
    X: Optional[FeatureMatrix] = None,
    y_true: Optional[LabelLike] = None,
    y_prob: Optional[ArrayLike] = None,
    n_bins: int = 10,
    strategy: str = "uniform",
    ece_threshold: float = 0.05,
    min_samples_per_bin: int = 20,
    bootstrap_iterations: int = 0,
    bootstrap_confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    High-level calibration evaluation API.

    Either provide:

        y_prob + y_true

    or:

        model + X + y_true

    Parameters
    ----------
    model:
        Fitted binary classifier exposing predict_proba().

    X:
        Evaluation feature matrix.

    y_true:
        Ground-truth binary labels.

    y_prob:
        Pre-computed positive-class probabilities.

    n_bins:
        Reliability-bin count.

    strategy:
        ``uniform`` or ``quantile``.

    ece_threshold:
        Calibration quality-gate threshold.

    min_samples_per_bin:
        Minimum recommended samples per populated bin.

    bootstrap_iterations:
        Optional number of bootstrap iterations. 0 disables bootstrap.

    bootstrap_confidence_level:
        Bootstrap confidence level.

    random_state:
        Deterministic random seed.

    Returns
    -------
    dict
        JSON-safe calibration report.
    """
    if y_true is None:
        raise ValueError(
            "Ground-truth labels 'y_true' must be provided."
        )

    if y_prob is not None and model is not None:
        logger.warning(
            "Both model and y_prob were provided; using explicit y_prob."
        )

    if y_prob is None:
        if model is None or X is None:
            raise ValueError(
                "Provide either y_prob or both model and X."
            )

        y_prob_arr = _extract_positive_class_probability(
            model,
            X,
        )
    else:
        y_prob_arr = _validate_probability_vector(y_prob)

    evaluator = CalibrationEvaluator(
        n_bins=n_bins,
        strategy=strategy,
        ece_threshold=ece_threshold,
        min_samples_per_bin=min_samples_per_bin,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_confidence_level=bootstrap_confidence_level,
        random_state=random_state,
    )

    report = evaluator.evaluate(
        y_true=y_true,
        y_prob=y_prob_arr,
    )

    return report.to_dict()


# ==============================================================================
# OPTIONAL DIRECT RELIABILITY DATA API
# ==============================================================================

def generate_reliability_data(
    y_true: LabelLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> List[Dict[str, Any]]:
    """
    Generate bin-level reliability data without calculating the full report.

    Useful for plotting reliability diagrams or dashboards.
    """
    labels, probabilities = _validate_inputs(y_true, y_prob)

    n_bins = _validate_n_bins(n_bins)
    strategy = _validate_strategy(strategy)

    bins, _ = _calculate_reliability_bins(
        labels,
        probabilities,
        n_bins=n_bins,
        strategy=strategy,
    )

    return [item.to_dict() for item in bins]


# ==============================================================================
# PUBLIC API
# ==============================================================================

__all__ = [
    "CalibrationReport",
    "ReliabilityBin",
    "CalibrationEvaluator",
    "calculate_expected_calibration_error",
    "evaluate_calibration",
    "generate_reliability_data",
]
"""
PhishGuard AI - Adversarial Robustness & Evasion Diagnostics
=============================================================

Version: 14.0.0
File Path: src/evaluation/adversarial.py

Production-grade adversarial robustness evaluation for phishing classifiers.

This module provides:

- deterministic synthetic perturbations;
- homoglyph and zero-width character transformations;
- conservative URL obfuscation transformations;
- clean-vs-adversarial prediction comparison;
- recall degradation analysis;
- evasion-success analysis;
- confidence-drop statistics;
- attack-vector security gates;
- aggregate robustness scoring;
- input/schema validation;
- JSON-safe report serialization;
- reproducible random state handling.

Important
---------
This module evaluates defensive model robustness. It does not perform live
network requests, exploit execution, credential collection, malware execution,
or interaction with external targets.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.logger import get_logger

logger = get_logger("src.evaluation.adversarial")


# ============================================================================
# CONSTANTS
# ============================================================================

MODULE_VERSION = "14.0.0"

ArrayLike = Union[
    pd.Series,
    pd.DataFrame,
    np.ndarray,
    Sequence[Any],
]

FeatureMatrix = Union[
    pd.DataFrame,
    np.ndarray,
]


# ============================================================================
# VALIDATION HELPERS
# ============================================================================


def _validate_threshold(value: float, name: str = "threshold") -> float:
    """
    Validate a probability/decision threshold.
    """
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric value.") from exc

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0.")

    return value


def _validate_non_negative_float(value: float, name: str) -> float:
    """
    Validate a non-negative floating-point configuration.
    """
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if value < 0.0:
        raise ValueError(f"{name} must be >= 0.")

    return value


def _validate_binary_labels(y_true: ArrayLike) -> np.ndarray:
    """
    Convert and validate binary target labels.
    """
    if isinstance(y_true, pd.DataFrame):
        if y_true.shape[1] != 1:
            raise ValueError("y_true DataFrame must contain exactly one column.")
        values = y_true.iloc[:, 0].to_numpy()
    elif isinstance(y_true, pd.Series):
        values = y_true.to_numpy()
    else:
        values = np.asarray(y_true)

    if values.ndim != 1:
        values = values.reshape(-1)

    if values.size == 0:
        raise ValueError("y_true cannot be empty.")

    if pd.isna(values).any():
        raise ValueError("y_true contains missing values.")

    try:
        values = values.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true must contain binary integer-compatible labels.") from exc

    unique = set(np.unique(values).tolist())

    if not unique.issubset({0, 1}):
        raise ValueError(
            f"y_true must contain only binary labels 0 and 1; "
            f"received {sorted(unique)}."
        )

    return values


def _validate_probability_vector(
    probabilities: Any,
    expected_length: Optional[int] = None,
) -> np.ndarray:
    """
    Normalize model probability output into a 1-D positive-class vector.
    """
    arr = np.asarray(probabilities, dtype=float)

    if arr.ndim == 1:
        positive = arr

    elif arr.ndim == 2:
        if arr.shape[1] == 2:
            positive = arr[:, 1]

        elif arr.shape[1] == 1:
            positive = arr[:, 0]

        else:
            raise ValueError(
                "predict_proba() must return shape (N,), (N, 1), or (N, 2). "
                f"Received {arr.shape}."
            )
    else:
        raise ValueError(
            "Probability output must be one- or two-dimensional; "
            f"received shape {arr.shape}."
        )

    if expected_length is not None and len(positive) != expected_length:
        raise ValueError(
            f"Probability length mismatch: expected {expected_length}, "
            f"received {len(positive)}."
        )

    if len(positive) == 0:
        raise ValueError("Probability vector cannot be empty.")

    if not np.all(np.isfinite(positive)):
        raise ValueError("Probability vector contains NaN or infinite values.")

    # Small numerical tolerance is allowed, but values outside [0, 1]
    # are rejected rather than silently clipped.
    if np.any(positive < -1e-8) or np.any(positive > 1.0 + 1e-8):
        raise ValueError("Probability values must lie within [0, 1].")

    return np.clip(positive, 0.0, 1.0)


def _validate_feature_matrix(
    X: FeatureMatrix,
    expected_rows: Optional[int] = None,
    reference_columns: Optional[Sequence[Any]] = None,
    name: str = "X",
) -> None:
    """
    Validate feature matrix shape and schema.
    """
    if isinstance(X, pd.DataFrame):
        if X.ndim != 2:
            raise ValueError(f"{name} must be a 2-dimensional DataFrame.")

        if X.shape[1] == 0:
            raise ValueError(f"{name} contains zero features.")

        if X.columns.duplicated().any():
            raise ValueError(f"{name} contains duplicated column names.")

        if X.isnull().values.any():
            raise ValueError(f"{name} contains missing values.")

        if not np.isfinite(
            X.select_dtypes(include=[np.number]).to_numpy(
                dtype=float,
                copy=False,
            )
        ).all():
            raise ValueError(f"{name} contains NaN or infinite numeric values.")

        if expected_rows is not None and len(X) != expected_rows:
            raise ValueError(
                f"{name} row count mismatch: expected {expected_rows}, "
                f"received {len(X)}."
            )

        if reference_columns is not None:
            if list(X.columns) != list(reference_columns):
                raise ValueError(
                    f"{name} feature schema does not match the clean dataset."
                )

    else:
        arr = np.asarray(X)

        if arr.ndim != 2:
            raise ValueError(
                f"{name} must be a 2-dimensional array; received shape {arr.shape}."
            )

        if arr.shape[1] == 0:
            raise ValueError(f"{name} contains zero features.")

        if expected_rows is not None and len(arr) != expected_rows:
            raise ValueError(
                f"{name} row count mismatch: expected {expected_rows}, "
                f"received {len(arr)}."
            )

        if np.issubdtype(arr.dtype, np.number):
            if not np.all(np.isfinite(arr.astype(float))):
                raise ValueError(f"{name} contains NaN or infinite values.")


def _safe_attack_name(name: str) -> str:
    """
    Produce a safe canonical identifier for an attack-vector name.
    """
    if not isinstance(name, str):
        raise TypeError("attack_name must be a string.")

    name = name.strip()

    if not name:
        raise ValueError("attack_name cannot be empty.")

    if len(name) > 128:
        raise ValueError("attack_name cannot exceed 128 characters.")

    if not re.fullmatch(r"[A-Za-z0-9_.:/-]+", name):
        raise ValueError(
            "attack_name contains unsupported characters. "
            "Use letters, numbers, '.', ':', '/', '_' or '-'."
        )

    return name


# ============================================================================
# MODEL PROBABILITY INTERFACE
# ============================================================================


def _predict_positive_probability(
    model: Any,
    X: FeatureMatrix,
) -> np.ndarray:
    """
    Obtain validated positive-class probabilities from a model.
    """
    if model is None:
        raise ValueError("model cannot be None.")

    predict_proba = getattr(model, "predict_proba", None)

    if not callable(predict_proba):
        raise AttributeError(
            "The supplied model must expose a callable predict_proba() method."
        )

    raw_probabilities = predict_proba(X)

    return _validate_probability_vector(
        raw_probabilities,
        expected_length=len(X),
    )


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class ConfidenceStatistics:
    """
    Confidence-change statistics for positive samples.

    Values are defined as:

        confidence_drop = clean_probability - adversarial_probability
    """

    mean: float = 0.0
    median: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    p90: float = 0.0
    p95: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            key: float(value)
            for key, value in asdict(self).items()
        }


@dataclass
class AttackVectorResult:
    """
    Detailed result for one adversarial attack vector.
    """

    attack_name: str
    sample_count: int

    positive_sample_count: int
    clean_detected_positive_count: int
    adversarial_detected_positive_count: int

    baseline_accuracy: float
    adversarial_accuracy: float

    baseline_precision: float
    adversarial_precision: float

    baseline_recall: float
    adversarial_recall: float

    baseline_f1: float
    adversarial_f1: float

    recall_decay_rate: float
    recall_absolute_drop: float

    evasion_success_rate: float
    confidence_statistics: ConfidenceStatistics

    passed_security_gate: bool

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["confidence_statistics"] = (
            self.confidence_statistics.to_dict()
        )
        return result


@dataclass
class AdversarialReport:
    """
    Aggregated adversarial robustness report.
    """

    overall_robustness_score: float
    weighted_robustness_score: float

    max_acceptable_recall_decay: float
    max_acceptable_evasion_rate: float

    passed_all_gates: bool

    vector_results: Dict[str, AttackVectorResult]

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_robustness_score": float(
                self.overall_robustness_score
            ),
            "weighted_robustness_score": float(
                self.weighted_robustness_score
            ),
            "max_acceptable_recall_decay": float(
                self.max_acceptable_recall_decay
            ),
            "max_acceptable_evasion_rate": float(
                self.max_acceptable_evasion_rate
            ),
            "passed_all_gates": bool(self.passed_all_gates),
            "vector_results": {
                name: result.to_dict()
                for name, result in self.vector_results.items()
            },
            "metadata": self.metadata,
        }

    def export_json(
        self,
        filepath: Union[str, Path],
    ) -> Path:
        """
        Export report as JSON.
        """
        path = Path(filepath).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                self.to_dict(),
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        logger.info(
            "Exported adversarial robustness report to '%s'.",
            path,
        )

        return path


# ============================================================================
# SYNTHETIC ADVERSARIAL PERTURBER
# ============================================================================


class AdversarialPerturber:
    """
    Deterministic, local-only perturbation generator.

    These transformations operate exclusively on supplied strings and do not
    perform network access.

    Parameters
    ----------
    random_state:
        Seed for reproducible stochastic perturbations.
    """

    HOMOGLYPH_MAP: Mapping[str, str] = {
        "a": "а",
        "e": "е",
        "o": "о",
        "p": "р",
        "c": "с",
        "i": "і",
        "x": "х",
        "y": "у",
    }

    ZERO_WIDTH_CHARACTERS: Tuple[str, ...] = (
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
    )

    def __init__(
        self,
        random_state: Optional[int] = 42,
    ) -> None:
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def apply_homoglyphs(
        self,
        text: str,
        replacement_ratio: float = 0.30,
    ) -> str:
        """
        Replace selected Latin characters with visually similar Unicode
        characters.

        This is useful for testing Unicode-normalization defenses.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        replacement_ratio = _validate_threshold(
            replacement_ratio,
            "replacement_ratio",
        )

        chars = list(text)

        for index, char in enumerate(chars):
            replacement = self.HOMOGLYPH_MAP.get(char)

            if (
                replacement is not None
                and self._rng.random() < replacement_ratio
            ):
                chars[index] = replacement

        return "".join(chars)

    def inject_zero_width_spaces(
        self,
        text: str,
        frequency: int = 5,
        character: str = "\u200b",
    ) -> str:
        """
        Inject zero-width characters at deterministic intervals.

        Intended for defensive testing of text canonicalization.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not isinstance(frequency, int) or frequency <= 0:
            raise ValueError("frequency must be a positive integer.")

        if character not in self.ZERO_WIDTH_CHARACTERS:
            raise ValueError(
                "character must be an approved zero-width Unicode character."
            )

        if len(text) <= frequency:
            return text

        return character.join(
            text[index:index + frequency]
            for index in range(0, len(text), frequency)
        )

    @staticmethod
    def unicode_decompose(text: str) -> str:
        """
        Produce a decomposed Unicode representation.

        Useful for testing whether downstream feature extraction is sensitive
        to canonical Unicode representation.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        return unicodedata.normalize("NFKD", text)

    @staticmethod
    def insert_separator_obfuscation(
        text: str,
        separator: str = "[.]",
    ) -> str:
        """
        Replace literal dots in domain-like text with a harmless textual
        separator.

        Example:

            example.com -> example[.]com

        This transformation is intentionally local and non-networked.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not separator:
            raise ValueError("separator cannot be empty.")

        return text.replace(".", separator)

    @staticmethod
    def obfuscate_urls(text: str) -> str:
        """
        Apply a local URL representation transformation.

        This intentionally does NOT introduce a third-party redirect domain
        or make any network request. It simply changes URL separators so that
        URL parsing/canonicalization logic can be tested.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        url_pattern = re.compile(
            r"https?://[^\s<>'\"]+",
            flags=re.IGNORECASE,
        )

        def replace_url(match: re.Match[str]) -> str:
            original = match.group(0)

            return original.replace(".", "[.]")

        return url_pattern.sub(replace_url, text)


# ============================================================================
# ADVERSARIAL EVALUATOR
# ============================================================================


class AdversarialEvaluator:
    """
    Production-grade evaluator for adversarial model robustness.
    """

    def __init__(
        self,
        max_acceptable_recall_decay: float = 0.15,
        decision_threshold: float = 0.50,
        max_acceptable_evasion_rate: float = 0.10,
        minimum_positive_samples: int = 1,
        require_clean_detection: bool = True,
    ) -> None:
        self.max_acceptable_recall_decay = (
            _validate_non_negative_float(
                max_acceptable_recall_decay,
                "max_acceptable_recall_decay",
            )
        )

        if self.max_acceptable_recall_decay > 1.0:
            raise ValueError(
                "max_acceptable_recall_decay cannot exceed 1.0."
            )

        self.decision_threshold = _validate_threshold(
            decision_threshold,
            "decision_threshold",
        )

        self.max_acceptable_evasion_rate = _validate_threshold(
            max_acceptable_evasion_rate,
            "max_acceptable_evasion_rate",
        )

        if not isinstance(minimum_positive_samples, int):
            raise TypeError("minimum_positive_samples must be an integer.")

        if minimum_positive_samples < 1:
            raise ValueError(
                "minimum_positive_samples must be >= 1."
            )

        self.minimum_positive_samples = minimum_positive_samples
        self.require_clean_detection = bool(require_clean_detection)

    # ----------------------------------------------------------------------
    # Prediction helper
    # ----------------------------------------------------------------------

    def _predict(
        self,
        model: Any,
        X: FeatureMatrix,
    ) -> Tuple[np.ndarray, np.ndarray]:
        probabilities = _predict_positive_probability(model, X)

        predictions = (
            probabilities >= self.decision_threshold
        ).astype(np.int8)

        return probabilities, predictions

    # ----------------------------------------------------------------------
    # Single-vector evaluation
    # ----------------------------------------------------------------------

    def evaluate_vector(
        self,
        model: Any,
        X_clean: FeatureMatrix,
        X_adv: FeatureMatrix,
        y_true: ArrayLike,
        attack_name: str,
    ) -> AttackVectorResult:
        """
        Evaluate one adversarial transformation.

        The clean and adversarial feature matrices must contain exactly the
        same number of samples and, for DataFrames, the same feature schema.
        """
        attack_name = _safe_attack_name(attack_name)

        y_true_arr = _validate_binary_labels(y_true)

        _validate_feature_matrix(
            X_clean,
            expected_rows=len(y_true_arr),
            name="X_clean",
        )

        reference_columns = (
            list(X_clean.columns)
            if isinstance(X_clean, pd.DataFrame)
            else None
        )

        _validate_feature_matrix(
            X_adv,
            expected_rows=len(y_true_arr),
            reference_columns=reference_columns,
            name="X_adv",
        )

        if len(X_clean) != len(X_adv):
            raise ValueError(
                "X_clean and X_adv must contain identical sample counts."
            )

        positive_mask = y_true_arr == 1
        positive_count = int(np.sum(positive_mask))

        if positive_count < self.minimum_positive_samples:
            raise ValueError(
                f"Attack vector '{attack_name}' contains only "
                f"{positive_count} positive samples; at least "
                f"{self.minimum_positive_samples} are required."
            )

        # --------------------------------------------------------------
        # Clean predictions
        # --------------------------------------------------------------

        clean_probabilities, clean_predictions = self._predict(
            model,
            X_clean,
        )

        # --------------------------------------------------------------
        # Adversarial predictions
        # --------------------------------------------------------------

        adversarial_probabilities, adversarial_predictions = self._predict(
            model,
            X_adv,
        )

        # --------------------------------------------------------------
        # Standard metrics
        # --------------------------------------------------------------

        baseline_accuracy = float(
            accuracy_score(
                y_true_arr,
                clean_predictions,
            )
        )

        adversarial_accuracy = float(
            accuracy_score(
                y_true_arr,
                adversarial_predictions,
            )
        )

        baseline_precision = float(
            precision_score(
                y_true_arr,
                clean_predictions,
                zero_division=0,
            )
        )

        adversarial_precision = float(
            precision_score(
                y_true_arr,
                adversarial_predictions,
                zero_division=0,
            )
        )

        baseline_recall = float(
            recall_score(
                y_true_arr,
                clean_predictions,
                zero_division=0,
            )
        )

        adversarial_recall = float(
            recall_score(
                y_true_arr,
                adversarial_predictions,
                zero_division=0,
            )
        )

        baseline_f1 = float(
            f1_score(
                y_true_arr,
                clean_predictions,
                zero_division=0,
            )
        )

        adversarial_f1 = float(
            f1_score(
                y_true_arr,
                adversarial_predictions,
                zero_division=0,
            )
        )

        # --------------------------------------------------------------
        # Recall degradation
        # --------------------------------------------------------------

        recall_absolute_drop = max(
            0.0,
            baseline_recall - adversarial_recall,
        )

        if baseline_recall > 0.0:
            recall_decay_rate = (
                recall_absolute_drop / baseline_recall
            )
        else:
            recall_decay_rate = 0.0

        recall_decay_rate = float(
            np.clip(recall_decay_rate, 0.0, 1.0)
        )

        # --------------------------------------------------------------
        # Evasion analysis
        # --------------------------------------------------------------
        #
        # Conventional evasion denominator:
        #
        # true-positive samples detected on clean input.
        #
        # This measures attack-induced successful evasion instead of simply
        # counting all positive samples.
        # --------------------------------------------------------------

        clean_detected_positive_mask = (
            positive_mask & (clean_predictions == 1)
        )

        clean_detected_positive_count = int(
            np.sum(clean_detected_positive_mask)
        )

        adversarial_detected_positive_count = int(
            np.sum(
                positive_mask
                & (adversarial_predictions == 1)
            )
        )

        if clean_detected_positive_count > 0:
            successful_evasions = int(
                np.sum(
                    clean_detected_positive_mask
                    & (adversarial_predictions == 0)
                )
            )

            evasion_success_rate = (
                successful_evasions
                / clean_detected_positive_count
            )

        else:
            successful_evasions = 0
            evasion_success_rate = 0.0

        # --------------------------------------------------------------
        # Confidence degradation
        # --------------------------------------------------------------

        confidence_drops = (
            clean_probabilities[positive_mask]
            - adversarial_probabilities[positive_mask]
        )

        confidence_statistics = ConfidenceStatistics(
            mean=float(np.mean(confidence_drops)),
            median=float(np.median(confidence_drops)),
            minimum=float(np.min(confidence_drops)),
            maximum=float(np.max(confidence_drops)),
            p90=float(np.percentile(confidence_drops, 90)),
            p95=float(np.percentile(confidence_drops, 95)),
        )

        # --------------------------------------------------------------
        # Security gates
        # --------------------------------------------------------------

        recall_gate = (
            recall_decay_rate
            <= self.max_acceptable_recall_decay
        )

        evasion_gate = (
            evasion_success_rate
            <= self.max_acceptable_evasion_rate
        )

        clean_detection_gate = True

        if self.require_clean_detection:
            clean_detection_gate = (
                clean_detected_positive_count > 0
            )

        passed_security_gate = bool(
            recall_gate
            and evasion_gate
            and clean_detection_gate
        )

        logger.info(
            "Adversarial vector '%s': "
            "baseline_recall=%.4f, adversarial_recall=%.4f, "
            "recall_decay=%.2f%%, evasion_rate=%.2f%%, "
            "passed=%s",
            attack_name,
            baseline_recall,
            adversarial_recall,
            recall_decay_rate * 100.0,
            evasion_success_rate * 100.0,
            passed_security_gate,
        )

        return AttackVectorResult(
            attack_name=attack_name,
            sample_count=len(y_true_arr),
            positive_sample_count=positive_count,
            clean_detected_positive_count=(
                clean_detected_positive_count
            ),
            adversarial_detected_positive_count=(
                adversarial_detected_positive_count
            ),
            baseline_accuracy=baseline_accuracy,
            adversarial_accuracy=adversarial_accuracy,
            baseline_precision=baseline_precision,
            adversarial_precision=adversarial_precision,
            baseline_recall=baseline_recall,
            adversarial_recall=adversarial_recall,
            baseline_f1=baseline_f1,
            adversarial_f1=adversarial_f1,
            recall_decay_rate=recall_decay_rate,
            recall_absolute_drop=float(recall_absolute_drop),
            evasion_success_rate=float(
                np.clip(evasion_success_rate, 0.0, 1.0)
            ),
            confidence_statistics=confidence_statistics,
            passed_security_gate=passed_security_gate,
            metadata={
                "successful_evasions": successful_evasions,
                "recall_gate_passed": recall_gate,
                "evasion_gate_passed": evasion_gate,
                "clean_detection_gate_passed": clean_detection_gate,
                "decision_threshold": self.decision_threshold,
            },
        )

    # ----------------------------------------------------------------------
    # Suite evaluation
    # ----------------------------------------------------------------------

    def evaluate_suite(
        self,
        model: Any,
        X_clean: FeatureMatrix,
        y_true: ArrayLike,
        adversarial_sets: Mapping[
            str,
            FeatureMatrix,
        ],
    ) -> AdversarialReport:
        """
        Evaluate multiple adversarial attack vectors.
        """
        if not isinstance(adversarial_sets, Mapping):
            raise TypeError(
                "adversarial_sets must be a mapping of attack names to datasets."
            )

        if not adversarial_sets:
            raise ValueError(
                "adversarial_sets cannot be empty."
            )

        y_true_arr = _validate_binary_labels(y_true)

        results: Dict[str, AttackVectorResult] = {}

        for attack_name, X_adv in adversarial_sets.items():
            result = self.evaluate_vector(
                model=model,
                X_clean=X_clean,
                X_adv=X_adv,
                y_true=y_true_arr,
                attack_name=attack_name,
            )

            results[result.attack_name] = result

        if not results:
            raise RuntimeError(
                "No adversarial attack vectors were successfully evaluated."
            )

        # --------------------------------------------------------------
        # Aggregate robustness
        # --------------------------------------------------------------

        decay_rates = np.asarray(
            [
                result.recall_decay_rate
                for result in results.values()
            ],
            dtype=float,
        )

        positive_counts = np.asarray(
            [
                result.positive_sample_count
                for result in results.values()
            ],
            dtype=float,
        )

        evasion_rates = np.asarray(
            [
                result.evasion_success_rate
                for result in results.values()
            ],
            dtype=float,
        )

        # Equal-vector robustness
        overall_robustness_score = float(
            np.clip(
                1.0 - np.mean(decay_rates),
                0.0,
                1.0,
            )
        )

        # Sample-weighted robustness
        weight_sum = float(np.sum(positive_counts))

        if weight_sum > 0:
            weighted_decay = float(
                np.sum(decay_rates * positive_counts)
                / weight_sum
            )
        else:
            weighted_decay = 0.0

        weighted_robustness_score = float(
            np.clip(
                1.0 - weighted_decay,
                0.0,
                1.0,
            )
        )

        passed_all_gates = all(
            result.passed_security_gate
            for result in results.values()
        )

        return AdversarialReport(
            overall_robustness_score=overall_robustness_score,
            weighted_robustness_score=weighted_robustness_score,
            max_acceptable_recall_decay=(
                self.max_acceptable_recall_decay
            ),
            max_acceptable_evasion_rate=(
                self.max_acceptable_evasion_rate
            ),
            passed_all_gates=bool(passed_all_gates),
            vector_results=results,
            metadata={
                "module_version": MODULE_VERSION,
                "attack_vector_count": len(results),
                "total_test_samples": int(len(y_true_arr)),
                "total_positive_samples": int(
                    np.sum(y_true_arr == 1)
                ),
                "decision_threshold": self.decision_threshold,
                "minimum_positive_samples": (
                    self.minimum_positive_samples
                ),
                "require_clean_detection": (
                    self.require_clean_detection
                ),
                "mean_recall_decay": float(
                    np.mean(decay_rates)
                ),
                "maximum_recall_decay": float(
                    np.max(decay_rates)
                ),
                "mean_evasion_success_rate": float(
                    np.mean(evasion_rates)
                ),
                "maximum_evasion_success_rate": float(
                    np.max(evasion_rates)
                ),
            },
        )


# ============================================================================
# HIGH-LEVEL FUNCTIONAL API
# ============================================================================


def evaluate_adversarial_robustness(
    model: Any,
    adversarial_sets: Mapping[
        str,
        FeatureMatrix,
    ],
    X_clean: Optional[FeatureMatrix] = None,
    y_true: Optional[ArrayLike] = None,
    max_recall_decay: float = 0.15,
    threshold: float = 0.50,
    max_evasion_rate: float = 0.10,
    minimum_positive_samples: int = 1,
    require_clean_detection: bool = True,
) -> Dict[str, Any]:
    """
    Functional API for adversarial robustness evaluation.

    Parameters
    ----------
    model:
        Fitted classifier exposing predict_proba().

    adversarial_sets:
        Mapping:

            attack_name -> adversarial feature matrix

    X_clean:
        Clean baseline feature matrix.

    y_true:
        Ground-truth binary labels.

    max_recall_decay:
        Maximum permitted relative recall degradation.

    threshold:
        Classification decision threshold.

    max_evasion_rate:
        Maximum permitted clean-positive -> adversarial-negative flip rate.

    minimum_positive_samples:
        Minimum number of positive examples required per attack vector.

    require_clean_detection:
        If True, an attack vector cannot pass if the model detects none of
        its positive clean samples.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable adversarial robustness report.
    """
    if X_clean is None:
        raise ValueError(
            "X_clean is required for adversarial evaluation."
        )

    if y_true is None:
        raise ValueError(
            "y_true is required for adversarial evaluation."
        )

    evaluator = AdversarialEvaluator(
        max_acceptable_recall_decay=max_recall_decay,
        decision_threshold=threshold,
        max_acceptable_evasion_rate=max_evasion_rate,
        minimum_positive_samples=minimum_positive_samples,
        require_clean_detection=require_clean_detection,
    )

    report = evaluator.evaluate_suite(
        model=model,
        X_clean=X_clean,
        y_true=y_true,
        adversarial_sets=adversarial_sets,
    )

    return report.to_dict()


# ============================================================================
# PUBLIC EXPORT CONTRACT
# ============================================================================


__all__ = [
    "MODULE_VERSION",
    "ConfidenceStatistics",
    "AttackVectorResult",
    "AdversarialReport",
    "AdversarialPerturber",
    "AdversarialEvaluator",
    "evaluate_adversarial_robustness",
]
"""
PhishGuard_AI - Regression Evaluation Utilities
===============================================

Champion-vs-challenger regression testing for model deployment gates.

Security / reliability principles:
- Metrics are computed deterministically from finite probability vectors.
- Guardrail deltas are signed and bounded.
- ``passed_all_guardrails`` is the single deployment-gate boolean.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("src.evaluation.regression")

ArrayLike = Union[np.ndarray, "list[float]"]


def _to_probabilities(scores: ArrayLike) -> np.ndarray:
    return np.asarray(scores, dtype=float).ravel()


def _binary_metrics(y_true: ArrayLike, proba: ArrayLike, threshold: float) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float).ravel()
    p = _to_probabilities(proba)
    pred = (p >= threshold).astype(int)

    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = float(np.mean(pred == y))
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "accuracy": round(accuracy, 6),
    }


class RegressionEvaluator:
    """Evaluate a challenger model against a champion model on shared data."""

    def __init__(
        self,
        champion: Any,
        challenger: Any,
        threshold: float = 0.50,
    ) -> None:
        self.champion = champion
        self.challenger = challenger
        self.threshold = float(threshold)

    def predict_proba(self, model: Any, X: Any) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return proba[:, 1] if proba.ndim == 2 else _to_probabilities(proba)
        return _to_probabilities(model.predict(X))

    def evaluate(
        self,
        X_test: Any,
        y_test: Any,
        guardrails: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        champion_proba = self.predict_proba(self.champion, X_test)
        challenger_proba = self.predict_proba(self.challenger, X_test)

        champ_metrics = _binary_metrics(y_test, champion_proba, self.threshold)
        chal_metrics = _binary_metrics(y_test, challenger_proba, self.threshold)

        deltas = {
            key: round(chal_metrics[key] - champ_metrics[key], 6)
            for key in champ_metrics
        }

        guardrail_results: Dict[str, bool] = {}
        guardrails = guardrails or {}
        for name, margin in guardrails.items():
            delta = deltas.get(name, 0.0)
            guardrail_results[name] = bool(delta >= -float(margin))

        passed_all = bool(guardrail_results) and all(guardrail_results.values())

        return {
            "champion_metrics": champ_metrics,
            "challenger_metrics": chal_metrics,
            "deltas": deltas,
            "threshold": self.threshold,
            "guardrails": dict(guardrails),
            "guardrail_results": guardrail_results,
            "passed_all_guardrails": passed_all,
        }


def compare_champion_vs_challenger(
    *,
    champion: Any,
    challenger: Any,
    X_test: Any,
    y_test: Any,
    threshold: float = 0.50,
    guardrails: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compare a challenger against a champion and apply deployment guardrails.

    Returns a dictionary with ``passed_all_guardrails`` set to ``True`` only
    when every configured guardrail margin is met.
    """
    evaluator = RegressionEvaluator(
        champion=champion,
        challenger=challenger,
        threshold=threshold,
    )
    return evaluator.evaluate(
        X_test=X_test,
        y_test=y_test,
        guardrails=guardrails,
    )


__all__ = [
    "RegressionEvaluator",
    "compare_champion_vs_challenger",
]

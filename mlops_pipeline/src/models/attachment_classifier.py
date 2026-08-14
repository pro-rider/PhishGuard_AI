"""
PhishGuard AI - Attachment Phishing & Malware Classifier
=========================================================

Version: 14.0.0
File Path: src/models/attachment_classifier.py

Production-oriented attachment security classifier.

Responsibilities
----------------
* Classify attachment telemetry into benign / malicious classes.
* Enforce strict train/inference feature contracts.
* Detect schema drift and invalid numeric values.
* Support RandomForest and ExtraTrees engines.
* Preserve deterministic model configuration.
* Provide probability predictions and threshold-based decisions.
* Provide normalized feature importance.
* Produce validation metrics suitable for imbalanced security datasets.
* Preserve training metadata and class-distribution information.
* Avoid target leakage through strict separation of X and y.
* Provide safe estimator reinitialization for retraining.

Security Model
--------------
0 = benign
1 = malicious / phishing-related attachment

Important
---------
This classifier should operate on extracted metadata/features rather than
opening or executing attachment payloads. Dynamic file analysis belongs in
a sandbox / static-analysis subsystem, not inside the ML classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.base_model import BasePhishingModel
from src.utils.logger import get_logger


logger = get_logger("src.models.attachment_classifier")


# =============================================================================
# CONSTANTS
# =============================================================================

MODEL_VERSION = "14.0.0"

BENIGN_LABEL = 0
MALICIOUS_LABEL = 1

SUPPORTED_LABELS = frozenset({BENIGN_LABEL, MALICIOUS_LABEL})


# =============================================================================
# INTERNAL DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class AttachmentValidationMetrics:
    """
    Immutable validation metrics generated after model training.

    These metrics are intentionally stored separately from the estimator
    because they describe an evaluation event rather than the model itself.
    """

    sample_count: int
    positive_count: int
    negative_count: int
    positive_rate: float

    accuracy: float
    balanced_accuracy: float

    precision: float
    recall: float
    f1: float

    roc_auc: Optional[float]
    pr_auc: Optional[float]

    decision_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-compatible representation."""
        return {
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "positive_rate": self.positive_rate,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "decision_threshold": self.decision_threshold,
        }


# =============================================================================
# ATTACHMENT CLASSIFIER
# =============================================================================

class AttachmentPhishingClassifier(BasePhishingModel):
    """
    Production attachment phishing/malware classifier.

    The model expects a numeric feature matrix representing attachment
    telemetry such as:

        * file size
        * entropy
        * macro indicators
        * executable indicators
        * MIME mismatch flags
        * extension risk indicators
        * archive depth
        * embedded object counts
        * script indicators
        * structural PE/ELF metadata

    The classifier deliberately does not inspect or execute file contents.
    """

    def __init__(
        self,
        model_name: str = "attachment_phishing_classifier",
        version: str = MODEL_VERSION,
        decision_threshold: float = 0.50,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the attachment classifier.

        Parameters
        ----------
        model_name:
            Stable artifact/model identifier.

        version:
            Model implementation/version identifier.

        decision_threshold:
            Probability threshold for malicious classification.

        model_params:
            Estimator configuration.

            Supported control parameter:

                use_extra_trees: bool

            All remaining parameters are passed to the selected sklearn
            estimator.
        """

        self._validate_threshold(decision_threshold)

        default_params: Dict[str, Any] = {
            "n_estimators": 300,
            "max_depth": 16,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "random_state": 42,
            "n_jobs": -1,
            "bootstrap": True,
            "use_extra_trees": False,
        }

        if model_params:
            default_params.update(dict(model_params))

        # Do not mutate the caller's dictionary.
        normalized_params = dict(default_params)

        super().__init__(
            model_name=model_name,
            version=version,
            decision_threshold=decision_threshold,
            model_params=normalized_params,
        )

        self.validation_metrics_: Optional[Dict[str, Any]] = None

        self.training_class_distribution_: Dict[int, int] = {}

        self.training_sample_count_: int = 0

        self.training_feature_count_: int = 0

        self.training_dtypes_: Dict[str, str] = {}

        self.feature_statistics_: Dict[str, Dict[str, float]] = {}

        self.estimator_type_: str = ""

        self._init_estimator()

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def _init_estimator(self) -> None:
        """
        Create the underlying sklearn estimator.

        The configuration flag `use_extra_trees` is consumed without mutating
        `self.model_params`.
        """

        use_extra_trees = bool(
            self.model_params.get("use_extra_trees", False)
        )

        estimator_params = {
            key: value
            for key, value in self.model_params.items()
            if key != "use_extra_trees"
        }

        if use_extra_trees:
            logger.info(
                "Initializing ExtraTreesClassifier for attachment detection."
            )

            self.model = ExtraTreesClassifier(
                **estimator_params
            )

            self.estimator_type_ = "ExtraTreesClassifier"

        else:
            logger.info(
                "Initializing RandomForestClassifier for attachment detection."
            )

            self.model = RandomForestClassifier(
                **estimator_params
            )

            self.estimator_type_ = "RandomForestClassifier"

    # =========================================================================
    # VALIDATION HELPERS
    # =========================================================================

    @staticmethod
    def _validate_threshold(threshold: float) -> None:
        """Validate classification threshold."""

        if not isinstance(threshold, (float, int)):
            raise TypeError(
                "decision_threshold must be a numeric value."
            )

        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(
                "decision_threshold must be between 0.0 and 1.0."
            )

    @staticmethod
    def _validate_dataframe_or_array(
        X: Union[pd.DataFrame, np.ndarray],
        *,
        name: str = "X",
    ) -> None:
        """
        Validate the basic shape and numerical safety of a feature matrix.
        """

        if X is None:
            raise ValueError(f"{name} cannot be None.")

        if isinstance(X, pd.DataFrame):

            if X.ndim != 2:
                raise ValueError(
                    f"{name} must be two-dimensional."
                )

            if X.shape[1] == 0:
                raise ValueError(
                    f"{name} contains zero features."
                )

        elif isinstance(X, np.ndarray):

            if X.ndim != 2:
                raise ValueError(
                    f"{name} must be a two-dimensional numpy array."
                )

            if X.shape[1] == 0:
                raise ValueError(
                    f"{name} contains zero features."
                )

        else:
            raise TypeError(
                f"{name} must be a pandas DataFrame or numpy ndarray; "
                f"received {type(X).__name__}."
            )

    @staticmethod
    def _validate_target(
        y: Union[pd.Series, np.ndarray],
    ) -> np.ndarray:
        """
        Validate and normalize binary target labels.

        Only labels 0 and 1 are accepted.
        """

        if y is None:
            raise ValueError("Target vector y cannot be None.")

        if isinstance(y, pd.Series):
            values = y.to_numpy()

        elif isinstance(y, np.ndarray):
            values = y

        else:
            values = np.asarray(y)

        if values.ndim != 1:
            values = values.reshape(-1)

        if values.size == 0:
            raise ValueError("Target vector y is empty.")

        if pd.isna(values).any():
            raise ValueError(
                "Target vector contains missing/null labels."
            )

        # Convert numpy integer-like values safely.
        try:
            numeric_values = values.astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Target labels must contain binary integer values 0 and 1."
            ) from exc

        unique_values = set(np.unique(numeric_values).tolist())

        if not unique_values.issubset(SUPPORTED_LABELS):
            raise ValueError(
                "Attachment classifier supports only binary labels "
                f"{sorted(SUPPORTED_LABELS)}; received {sorted(unique_values)}."
            )

        if len(unique_values) < 2:
            raise ValueError(
                "Training requires both benign (0) and malicious (1) samples."
            )

        return numeric_values

    @staticmethod
    def _validate_numeric_values(
        X: Union[pd.DataFrame, np.ndarray],
        *,
        name: str = "X",
    ) -> None:
        """
        Reject NaN and infinite values.

        RandomForest/ExtraTrees should not silently receive corrupted
        numerical telemetry.
        """

        if isinstance(X, pd.DataFrame):

            non_numeric = X.select_dtypes(
                exclude=[np.number, "bool"]
            ).columns.tolist()

            if non_numeric:
                raise TypeError(
                    f"{name} contains unsupported non-numeric columns: "
                    f"{non_numeric}"
                )

            numeric_array = X.to_numpy(dtype=np.float64)

        else:
            try:
                numeric_array = np.asarray(X, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{name} must contain numeric values only."
                ) from exc

        if not np.isfinite(numeric_array).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

    # =========================================================================
    # FEATURE CONTRACT
    # =========================================================================

    def _capture_feature_contract(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> None:
        """
        Capture feature names, ordering and basic training statistics.

        This protects the model from silent feature-order drift.
        """

        self.training_sample_count_ = int(X.shape[0])
        self.training_feature_count_ = int(X.shape[1])

        if isinstance(X, pd.DataFrame):

            self.feature_names_ = [
                str(column)
                for column in X.columns
            ]

            self.training_dtypes_ = {
                str(column): str(dtype)
                for column, dtype in X.dtypes.items()
            }

            numeric_df = X.select_dtypes(
                include=[np.number, "bool"]
            )

            statistics: Dict[str, Dict[str, float]] = {}

            for column in numeric_df.columns:

                series = pd.to_numeric(
                    numeric_df[column],
                    errors="coerce",
                )

                statistics[str(column)] = {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=0)),
                }

            self.feature_statistics_ = statistics

        else:

            self.feature_names_ = [
                f"feature_{index}"
                for index in range(X.shape[1])
            ]

            self.training_dtypes_ = {}

            array = np.asarray(X, dtype=np.float64)

            self.feature_statistics_ = {}

            for index in range(array.shape[1]):

                values = array[:, index]

                self.feature_statistics_[
                    f"feature_{index}"
                ] = {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

    def _validate_inference_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> None:
        """
        Verify inference feature compatibility with the training contract.
        """

        if X.shape[1] != self.training_feature_count_:
            raise ValueError(
                "Feature count mismatch: "
                f"model expects {self.training_feature_count_}, "
                f"received {X.shape[1]}."
            )

        if isinstance(X, pd.DataFrame):

            incoming_names = [
                str(column)
                for column in X.columns
            ]

            if incoming_names != self.feature_names_:
                raise ValueError(
                    "Feature schema/order mismatch detected. "
                    f"Expected {self.feature_names_}, "
                    f"received {incoming_names}."
                )

    # =========================================================================
    # TRAINING
    # =========================================================================

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        validation_data: Optional[
            Tuple[
                Union[pd.DataFrame, np.ndarray],
                Union[pd.Series, np.ndarray],
            ]
        ] = None,
        **kwargs: Any,
    ) -> "AttachmentPhishingClassifier":
        """
        Train the attachment classifier.

        Parameters
        ----------
        X:
            Numeric attachment feature matrix.

        y:
            Binary labels:
                0 = benign
                1 = malicious

        validation_data:
            Optional holdout validation tuple.

        kwargs:
            Additional sklearn fit parameters.

        Returns
        -------
        AttachmentPhishingClassifier
            Fitted classifier.
        """

        logger.info(
            "Starting training for attachment classifier '%s'.",
            self.model_name,
        )

        self._validate_dataframe_or_array(X, name="X")

        y_array = self._validate_target(y)

        if len(X) != len(y_array):
            raise ValueError(
                "Feature matrix and target vector have different "
                f"sample counts: X={len(X)}, y={len(y_array)}."
            )

        self._validate_numeric_values(X)

        if validation_data is not None:

            X_val, y_val = validation_data

            self._validate_dataframe_or_array(
                X_val,
                name="X_val",
            )

            y_val_array = self._validate_target(y_val)

            if len(X_val) != len(y_val_array):
                raise ValueError(
                    "Validation feature matrix and target vector have "
                    f"different sample counts: "
                    f"X_val={len(X_val)}, y_val={len(y_val_array)}."
                )

            self._validate_numeric_values(
                X_val,
                name="X_val",
            )

        # Capture training schema before fitting.
        self._capture_feature_contract(X)

        # Store class distribution for auditability.
        unique_labels, counts = np.unique(
            y_array,
            return_counts=True,
        )

        self.training_class_distribution_ = {
            int(label): int(count)
            for label, count in zip(unique_labels, counts)
        }

        # Always ensure a fresh estimator for a complete retraining cycle.
        self._init_estimator()

        self.model.fit(
            X,
            y_array,
            **kwargs,
        )

        self._update_fit_metadata(X)

        # Explicitly preserve our feature contract after base metadata update.
        self.training_feature_count_ = len(self.feature_names_)

        logger.info(
            "Attachment classifier '%s' trained successfully: "
            "samples=%d, features=%d, malicious_samples=%d, engine=%s.",
            self.model_name,
            self.training_sample_count_,
            self.training_feature_count_,
            self.training_class_distribution_.get(
                MALICIOUS_LABEL,
                0,
            ),
            self.estimator_type_,
        )

        if validation_data is not None:

            X_val, y_val = validation_data

            # Validate schema against training contract.
            self._validate_inference_features(X_val)

            metrics = self._evaluate_validation_set(
                X_val,
                y_val,
            )

            self.validation_metrics_ = metrics.to_dict()

            logger.info(
                "Attachment classifier validation completed: "
                "F1=%.4f, precision=%.4f, recall=%.4f, PR-AUC=%s.",
                metrics.f1,
                metrics.precision,
                metrics.recall,
                (
                    f"{metrics.pr_auc:.4f}"
                    if metrics.pr_auc is not None
                    else "N/A"
                ),
            )

        return self

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _evaluate_validation_set(
        self,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
    ) -> AttachmentValidationMetrics:
        """
        Evaluate the fitted classifier on a holdout dataset.
        """

        y_true = self._validate_target(y_val)

        probabilities = self.predict_proba(X_val)

        positive_probabilities = probabilities[:, 1]

        predictions = (
            positive_probabilities
            >= self.decision_threshold
        ).astype(np.int8)

        positive_count = int(
            np.sum(y_true == MALICIOUS_LABEL)
        )

        negative_count = int(
            np.sum(y_true == BENIGN_LABEL)
        )

        positive_rate = float(
            positive_count / len(y_true)
        )

        roc_auc: Optional[float]

        if len(np.unique(y_true)) == 2:
            roc_auc = float(
                roc_auc_score(
                    y_true,
                    positive_probabilities,
                )
            )
        else:
            roc_auc = None

        try:
            pr_auc = float(
                average_precision_score(
                    y_true,
                    positive_probabilities,
                )
            )
        except ValueError:
            pr_auc = None

        return AttachmentValidationMetrics(
            sample_count=len(y_true),
            positive_count=positive_count,
            negative_count=negative_count,
            positive_rate=positive_rate,
            accuracy=float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),
            balanced_accuracy=float(
                balanced_accuracy_score(
                    y_true,
                    predictions,
                )
            ),
            precision=float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            recall=float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            f1=float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            decision_threshold=float(
                self.decision_threshold
            ),
        )

    # =========================================================================
    # INFERENCE
    # =========================================================================

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict benign/malicious probabilities.

        Returns
        -------
        numpy.ndarray
            Array with shape `(N, 2)`:

                [:, 0] = benign probability
                [:, 1] = malicious probability
        """

        if not self.is_fitted:
            raise RuntimeError(
                f"Model '{self.model_name}' must be fitted "
                "before running inference."
            )

        self._validate_dataframe_or_array(X)
        self._validate_numeric_values(X)

        self._validate_inference_features(X)

        probabilities = self.model.predict_proba(X)

        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        if probabilities.ndim != 2:
            raise RuntimeError(
                "Underlying estimator returned an invalid probability "
                f"shape: {probabilities.shape}."
            )

        if probabilities.shape[1] != 2:
            raise RuntimeError(
                "Attachment classifier requires a binary probability "
                f"matrix with two columns; received {probabilities.shape}."
            )

        if not np.isfinite(probabilities).all():
            raise RuntimeError(
                "Estimator produced NaN or infinite probabilities."
            )

        # Numerical safety.
        probabilities = np.clip(
            probabilities,
            0.0,
            1.0,
        )

        return probabilities

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict benign/malicious labels using the configured threshold.

        This override provides stricter validation than the base implementation.
        """

        probabilities = self.predict_proba(X)

        positive_probability = probabilities[:, 1]

        return (
            positive_probability
            >= self.decision_threshold
        ).astype(np.int8)

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    def get_feature_importance(
        self,
    ) -> Dict[str, float]:
        """
        Return normalized tree-based feature importance.

        Returns
        -------
        Dict[str, float]
            Feature name -> normalized importance.

        Notes
        -----
        Native impurity importance should not be interpreted as causal
        importance. For high-stakes analysis, SHAP/permutation importance
        should be evaluated separately.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Model is not fitted. Cannot retrieve feature importances."
            )

        if not hasattr(
            self.model,
            "feature_importances_",
        ):
            logger.warning(
                "Estimator '%s' does not expose feature_importances_.",
                self.estimator_type_,
            )

            return {}

        raw_importances = np.asarray(
            self.model.feature_importances_,
            dtype=np.float64,
        )

        if len(raw_importances) != len(
            self.feature_names_
        ):
            raise RuntimeError(
                "Feature importance count does not match "
                "the trained feature contract."
            )

        if not np.isfinite(raw_importances).all():
            raise RuntimeError(
                "Feature importance contains NaN or infinite values."
            )

        raw_importances = np.maximum(
            raw_importances,
            0.0,
        )

        total = float(
            raw_importances.sum()
        )

        if total <= 0.0:
            normalized = np.zeros_like(
                raw_importances
            )
        else:
            normalized = raw_importances / total

        importance_map = {
            feature: float(score)
            for feature, score in zip(
                self.feature_names_,
                normalized,
            )
        }

        return dict(
            sorted(
                importance_map.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    # =========================================================================
    # MODEL METADATA
    # =========================================================================

    def get_metadata(self) -> Dict[str, Any]:
        """
        Extend base metadata with attachment-specific audit information.
        """

        metadata = super().get_metadata()

        metadata.update(
            {
                "classifier_type": "attachment_security",
                "estimator_type": self.estimator_type_,
                "training_sample_count": self.training_sample_count_,
                "training_feature_count": self.training_feature_count_,
                "training_class_distribution": dict(
                    self.training_class_distribution_
                ),
                "training_dtypes": dict(
                    self.training_dtypes_
                ),
                "feature_statistics": dict(
                    self.feature_statistics_
                ),
                "validation_metrics": (
                    dict(self.validation_metrics_)
                    if self.validation_metrics_
                    else None
                ),
                "security_labels": {
                    "0": "benign",
                    "1": "malicious",
                },
            }
        )

        return metadata

    # =========================================================================
    # MODEL HEALTH
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Return a lightweight runtime health report.

        This is useful before putting a loaded model into production.
        """

        healthy = True
        problems: List[str] = []

        if not self.is_fitted:
            healthy = False
            problems.append(
                "Model is not fitted."
            )

        if self.model is None:
            healthy = False
            problems.append(
                "Underlying estimator is missing."
            )

        if self.training_feature_count_ <= 0:
            healthy = False
            problems.append(
                "Training feature count is invalid."
            )

        if not self.feature_names_:
            healthy = False
            problems.append(
                "Feature contract is missing."
            )

        try:
            self._validate_threshold(
                self.decision_threshold
            )
        except (TypeError, ValueError) as exc:
            healthy = False
            problems.append(str(exc))

        return {
            "healthy": healthy,
            "model_name": self.model_name,
            "version": self.version,
            "estimator_type": self.estimator_type_,
            "is_fitted": self.is_fitted,
            "feature_count": self.training_feature_count_,
            "problems": problems,
        }

    # =========================================================================
    # THRESHOLD MANAGEMENT
    # =========================================================================

    def set_decision_threshold(
        self,
        threshold: float,
    ) -> None:
        """
        Change the operational classification threshold.

        Important:
            This does not retrain the model.

        It changes only the conversion from probability to class label.
        """

        self._validate_threshold(threshold)

        old_threshold = self.decision_threshold

        self.decision_threshold = float(
            threshold
        )

        logger.info(
            "Attachment classifier decision threshold changed "
            "from %.4f to %.4f.",
            old_threshold,
            self.decision_threshold,
        )

    # =========================================================================
    # SAFE MODEL SUMMARY
    # =========================================================================

    def summary(self) -> Dict[str, Any]:
        """
        Return a concise operational model summary.
        """

        return {
            "model_name": self.model_name,
            "version": self.version,
            "estimator": self.estimator_type_,
            "fitted": self.is_fitted,
            "decision_threshold": self.decision_threshold,
            "features": len(self.feature_names_),
            "training_samples": self.training_sample_count_,
            "class_distribution": dict(
                self.training_class_distribution_
            ),
            "validation": (
                dict(self.validation_metrics_)
                if self.validation_metrics_
                else None
            ),
        }


# =============================================================================
# PUBLIC EXPORT CONTRACT
# =============================================================================

__all__ = [
    "AttachmentPhishingClassifier",
    "AttachmentValidationMetrics",
]
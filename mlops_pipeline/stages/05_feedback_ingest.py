"""
PhishGuard AI - Stage 05: Production Feedback Ingestion, Label Reconciliation,
Data Quality, Drift Detection & Retraining Governance
==========================================================================

Version: 14.0.0
File Path: src/stages/05_feedback_ingest.py

Production responsibilities
---------------------------
1. Safely ingest production feedback.
2. Validate and normalize feedback schema.
3. Reject malformed / unsafe records.
4. Reconcile duplicate and conflicting labels.
5. Prevent repeated counting of previously ingested feedback.
6. Maintain an append/update historical feedback store.
7. Calculate numerical distribution drift using PSI and KS tests.
8. Calculate categorical distribution drift.
9. Perform basic data-quality checks.
10. Detect feedback-label imbalance.
11. Detect suspicious feedback concentration.
12. Generate retraining recommendations.
13. Separate "drift detected" from "retraining approved".
14. Export deterministic, auditable manifests.
15. Use atomic writes for critical artifacts.
16. Preserve source fingerprints and lineage metadata.

Security model
-------------
Production feedback is treated as UNTRUSTED DATA.

User reports alone must never automatically become authoritative ground truth.
Authority levels are configurable, and unknown feedback sources receive the
lowest trust level.

This module does NOT automatically train or deploy a model.

Recommended production flow:

    production feedback
            |
            v
    Stage 05 ingestion
            |
            +--> validation
            +--> reconciliation
            +--> drift detection
            +--> poisoning indicators
            |
            v
    retrain_trigger.json
            |
            v
    Stage 02 training / approval workflow
            |
            v
    Stage 03 evaluation
            |
            v
    Stage 04 ONNX export
            |
            v
    deployment / registry

Notes
-----
- This module intentionally does not load joblib/pickle models.
- Feedback should preferably contain a stable non-sensitive entity identifier
  such as a SHA-256 URL hash rather than the raw URL.
- Never store secrets, authentication tokens, cookies, passwords, or raw
  credentials in feedback records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger("src.stages.05_feedback_ingest")


# ============================================================================
# CONSTANTS
# ============================================================================

MODULE_VERSION = "14.0.0"

SUPPORTED_INPUT_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".ndjson",
    ".parquet",
}

SUPPORTED_LABELS = {
    0,
    1,
}

DEFAULT_AUTHORITY_WEIGHTS: Dict[str, int] = {
    "analyst_override": 100,
    "security_alert": 80,
    "trusted_feed": 70,
    "verified_customer": 50,
    "user_report": 20,
    "unknown": 0,
}

DEFAULT_MAX_FILE_SIZE_MB = 250
DEFAULT_MIN_DRIFT_SAMPLE_SIZE = 30
DEFAULT_MIN_EFFECTIVE_SAMPLE_SIZE = 30

DEFAULT_PSI_THRESHOLD = 0.20
DEFAULT_KS_PVALUE_THRESHOLD = 0.05
DEFAULT_CATEGORICAL_PVALUE_THRESHOLD = 0.05
DEFAULT_DRIFT_RATIO_THRESHOLD = 0.25

DEFAULT_MIN_NEW_SAMPLES_FOR_RETRAIN = 500
DEFAULT_MIN_TRUSTED_SAMPLES_FOR_RETRAIN = 100

DEFAULT_MAX_UNKNOWN_SOURCE_RATIO = 0.50
DEFAULT_MAX_SINGLE_ENTITY_RATIO = 0.05
DEFAULT_MAX_LABEL_DOMINANCE_RATIO = 0.98

DEFAULT_PSI_BINS = 10
DEFAULT_PSI_EPSILON = 1e-6

DEFAULT_RETRAIN_COOLDOWN_HOURS = 24


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class DriftMetric:
    """
    Per-feature drift result.

    The object is deliberately immutable so reports cannot accidentally be
    modified after creation.
    """

    feature: str
    feature_type: str
    baseline_count: int
    current_count: int
    psi: Optional[float]
    ks_statistic: Optional[float]
    ks_pvalue: Optional[float]
    categorical_pvalue: Optional[float]
    drift_detected: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class FeedbackQualityReport:
    """
    Data quality and feedback trust assessment.
    """

    input_records: int
    accepted_records: int
    rejected_records: int
    duplicate_records: int
    invalid_label_records: int
    missing_entity_records: int
    missing_timestamp_records: int
    unknown_authority_records: int
    unknown_authority_ratio: float
    dominant_label_ratio: float
    dominant_entity_ratio: float
    quality_passed: bool
    poisoning_risk_detected: bool
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrainingDecision:
    """
    Governance decision.

    should_retrain is intentionally conservative.

    A high drift score alone does not automatically mean retraining should
    happen. The decision additionally considers data volume, trust, quality,
    poisoning indicators and cooldown policy.
    """

    should_retrain: bool
    status: str
    reasons: Tuple[str, ...]
    blockers: Tuple[str, ...]
    trusted_new_samples: int
    total_new_samples: int
    drift_ratio: float
    quality_passed: bool
    poisoning_risk_detected: bool
    cooldown_active: bool


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return pd.Timestamp.utcnow().isoformat()


def _json_safe(value: Any) -> Any:
    """
    Convert common NumPy/Pandas objects into JSON-compatible values.
    """

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)

    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def _atomic_write_json(
    payload: Mapping[str, Any],
    path: Union[str, Path],
) -> Path:
    """
    Atomically write JSON.

    Writes to a temporary file first and then replaces the target. This avoids
    partially-written manifests when a process crashes during serialization.
    """

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
        text=True,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(dict(payload)),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_name, target)

    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

    return target


def _atomic_write_csv(
    dataframe: pd.DataFrame,
    path: Union[str, Path],
) -> Path:
    """
    Atomically write a CSV artifact.
    """

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
        text=True,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            dataframe.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_name, target)

    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

    return target


def calculate_file_sha256(
    filepath: Union[str, Path],
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate SHA-256 fingerprint of an input artifact.

    This provides basic lineage and repeat-ingestion detection.
    """

    path = Path(filepath).resolve()

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def validate_input_file(
    filepath: Union[str, Path],
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
) -> Path:
    """
    Validate feedback input path before reading it.

    Checks:
    - existence
    - regular file
    - supported extension
    - maximum file size
    """

    path = Path(filepath).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Feedback file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Feedback path is not a regular file: {path}")

    if path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        raise ValueError(
            f"Unsupported feedback format '{path.suffix}'. "
            f"Supported formats: {sorted(SUPPORTED_INPUT_EXTENSIONS)}"
        )

    size_bytes = path.stat().st_size
    max_bytes = max_file_size_mb * 1024 * 1024

    if size_bytes > max_bytes:
        raise ValueError(
            f"Feedback file exceeds configured size limit: "
            f"{size_bytes / (1024 * 1024):.2f} MB > {max_file_size_mb} MB"
        )

    return path


# ============================================================================
# INPUT LOADING
# ============================================================================

def load_feedback_dataframe(
    filepath: Union[str, Path],
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
) -> pd.DataFrame:
    """
    Load feedback data from CSV, JSON, JSONL/NDJSON or Parquet.
    """

    path = validate_input_file(filepath, max_file_size_mb=max_file_size_mb)

    logger.info("Loading feedback data from '%s'", path)

    suffix = path.suffix.lower()

    if suffix == ".csv":
        dataframe = pd.read_csv(
            path,
            low_memory=False,
        )

    elif suffix == ".parquet":
        dataframe = pd.read_parquet(path)

    elif suffix == ".json":
        dataframe = pd.read_json(path)

    elif suffix in {".jsonl", ".ndjson"}:
        dataframe = pd.read_json(path, lines=True)

    else:
        raise ValueError(f"Unsupported input format: {suffix}")

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Feedback loader did not produce a pandas DataFrame.")

    return dataframe


# ============================================================================
# SCHEMA / LABEL NORMALIZATION
# ============================================================================

def normalize_binary_label(value: Any) -> Optional[int]:
    """
    Normalize common binary label representations.

    Accepted positive forms:
        1, "1", "true", "phishing", "malicious", "yes"

    Accepted negative forms:
        0, "0", "false", "legitimate", "benign", "no"

    Unknown values become None and are rejected by validation.
    """

    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(value, (int, np.integer)):
        value_int = int(value)
        return value_int if value_int in SUPPORTED_LABELS else None

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None

        if float(value).is_integer():
            integer = int(value)

            if integer in SUPPORTED_LABELS:
                return integer

        return None

    normalized = str(value).strip().lower()

    positive = {
        "1",
        "true",
        "yes",
        "phishing",
        "phish",
        "malicious",
        "malware",
        "fraud",
    }

    negative = {
        "0",
        "false",
        "no",
        "legitimate",
        "legit",
        "benign",
        "safe",
    }

    if normalized in positive:
        return 1

    if normalized in negative:
        return 0

    return None


def validate_feedback_schema(
    dataframe: pd.DataFrame,
    entity_id_col: str,
    label_col: str,
    timestamp_col: str,
    authority_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and normalize incoming feedback.

    Invalid records are removed from the accepted dataset rather than being
    silently coerced into potentially dangerous labels.
    """

    if dataframe.empty:
        return dataframe.copy(), {
            "input_records": 0,
            "accepted_records": 0,
            "rejected_records": 0,
        }

    df = dataframe.copy()

    required = [label_col]

    missing_required = [
        column for column in required
        if column not in df.columns
    ]

    if missing_required:
        raise KeyError(
            f"Required feedback columns are missing: {missing_required}"
        )

    input_count = len(df)

    # ------------------------------------------------------------------
    # Normalize labels
    # ------------------------------------------------------------------

    df["_normalized_label"] = df[label_col].map(normalize_binary_label)

    invalid_label_mask = df["_normalized_label"].isna()

    invalid_label_records = int(invalid_label_mask.sum())

    df = df.loc[~invalid_label_mask].copy()
    df[label_col] = df["_normalized_label"].astype(int)
    df.drop(columns=["_normalized_label"], inplace=True)

    # ------------------------------------------------------------------
    # Entity identifier
    # ------------------------------------------------------------------

    missing_entity_records = 0

    if entity_id_col in df.columns:
        missing_entity_records = int(
            df[entity_id_col].isna().sum()
        )

        df[entity_id_col] = (
            df[entity_id_col]
            .astype("string")
            .str.strip()
        )

        # Empty strings are treated as missing.
        df.loc[
            df[entity_id_col].isin(["", "nan", "none"]),
            entity_id_col,
        ] = pd.NA

    # ------------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------------

    missing_timestamp_records = 0

    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(
            df[timestamp_col],
            errors="coerce",
            utc=True,
        )

        missing_timestamp_records = int(
            df[timestamp_col].isna().sum()
        )

    # ------------------------------------------------------------------
    # Authority
    # ------------------------------------------------------------------

    unknown_authority_records = 0

    if authority_col in df.columns:
        normalized_authority = (
            df[authority_col]
            .astype("string")
            .fillna("unknown")
            .str.strip()
            .str.lower()
        )

        unknown_mask = ~normalized_authority.isin(
            DEFAULT_AUTHORITY_WEIGHTS.keys()
        )

        unknown_authority_records = int(unknown_mask.sum())

        df[authority_col] = normalized_authority.mask(
            unknown_mask,
            "unknown",
        )

    # ------------------------------------------------------------------
    # Remove exact duplicates
    # ------------------------------------------------------------------

    before_dedup = len(df)

    df = df.drop_duplicates(
        keep="last"
    ).reset_index(drop=True)

    duplicate_records = before_dedup - len(df)

    quality = {
        "input_records": input_count,
        "accepted_records": len(df),
        "rejected_records": input_count - len(df),
        "invalid_label_records": invalid_label_records,
        "missing_entity_records": missing_entity_records,
        "missing_timestamp_records": missing_timestamp_records,
        "unknown_authority_records": unknown_authority_records,
        "duplicate_records": duplicate_records,
    }

    return df, quality


# ============================================================================
# FEEDBACK RECONCILIATION
# ============================================================================

def reconcile_feedback_records(
    raw_df: pd.DataFrame,
    entity_id_col: str = "url_hash",
    label_col: str = "label",
    timestamp_col: str = "timestamp",
    authority_col: str = "source_type",
    authority_weights: Optional[Mapping[str, int]] = None,
) -> pd.DataFrame:
    """
    Reconcile feedback records into one effective label per entity.

    Reconciliation policy:

        1. Highest trusted authority wins.
        2. Within the same authority, newest timestamp wins.
        3. If timestamp is unavailable, original ordering is retained.

    This is deliberately deterministic.

    IMPORTANT:
    This function should be used for ground-truth construction only after
    input schema validation.
    """

    if raw_df.empty:
        return raw_df.copy()

    weights = dict(
        authority_weights or DEFAULT_AUTHORITY_WEIGHTS
    )

    df = raw_df.copy()

    if authority_col in df.columns:
        df["_authority_weight"] = (
            df[authority_col]
            .astype("string")
            .str.lower()
            .map(weights)
            .fillna(weights.get("unknown", 0))
            .astype(int)
        )
    else:
        df["_authority_weight"] = weights.get("unknown", 0)

    if timestamp_col in df.columns:
        df["_reconciliation_timestamp"] = pd.to_datetime(
            df[timestamp_col],
            errors="coerce",
            utc=True,
        )
    else:
        df["_reconciliation_timestamp"] = pd.NaT

    # Preserve original row order as deterministic final tie-breaker.
    df["_input_order"] = np.arange(len(df))

    if entity_id_col not in df.columns:
        logger.warning(
            "Entity ID column '%s' missing. "
            "Only exact duplicate reconciliation is possible.",
            entity_id_col,
        )

        result = (
            df.sort_values(
                by=[
                    "_authority_weight",
                    "_reconciliation_timestamp",
                    "_input_order",
                ],
                ascending=[False, False, True],
                na_position="last",
            )
            .drop_duplicates()
            .copy()
        )

    else:
        valid_entity = (
            df[entity_id_col]
            .notna()
            & (
                df[entity_id_col]
                .astype(str)
                .str.strip()
                != ""
            )
        )

        identified = df.loc[valid_entity].copy()
        unidentified = df.loc[~valid_entity].copy()

        identified = identified.sort_values(
            by=[
                entity_id_col,
                "_authority_weight",
                "_reconciliation_timestamp",
                "_input_order",
            ],
            ascending=[True, False, False, True],
            na_position="last",
        )

        identified = identified.drop_duplicates(
            subset=[entity_id_col],
            keep="first",
        )

        result = pd.concat(
            [identified, unidentified],
            ignore_index=True,
        )

    result = result.drop(
        columns=[
            "_authority_weight",
            "_reconciliation_timestamp",
            "_input_order",
        ],
        errors="ignore",
    )

    result = result.reset_index(drop=True)

    logger.info(
        "Feedback reconciliation complete: %d -> %d records",
        len(raw_df),
        len(result),
    )

    return result


# ============================================================================
# PSI
# ============================================================================

def _safe_quantile_edges(
    baseline: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    """
    Create robust quantile edges.

    Duplicate quantile boundaries are removed to avoid zero-width histogram
    bins.
    """

    if len(baseline) == 0:
        return np.array([-np.inf, np.inf], dtype=float)

    quantiles = np.linspace(
        0.0,
        1.0,
        num_bins + 1,
    )

    edges = np.quantile(
        baseline,
        quantiles,
    )

    edges = np.unique(edges)

    if len(edges) < 2:
        value = float(edges[0])
        return np.array(
            [-np.inf, value, np.inf],
            dtype=float,
        )

    edges = edges.astype(float)

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


def calculate_psi(
    baseline: np.ndarray,
    target: np.ndarray,
    num_bins: int = DEFAULT_PSI_BINS,
    eps: float = DEFAULT_PSI_EPSILON,
) -> float:
    """
    Calculate robust Population Stability Index.

    PSI:

        sum((target_pct - baseline_pct)
            * ln(target_pct / baseline_pct))

    Important improvement over the original implementation:
    duplicate quantile boundaries are removed before histogramming.
    """

    base = np.asarray(baseline, dtype=float)
    current = np.asarray(target, dtype=float)

    base = base[np.isfinite(base)]
    current = current[np.isfinite(current)]

    if len(base) == 0 or len(current) == 0:
        return float("nan")

    edges = _safe_quantile_edges(
        base,
        num_bins=num_bins,
    )

    base_counts, _ = np.histogram(
        base,
        bins=edges,
    )

    current_counts, _ = np.histogram(
        current,
        bins=edges,
    )

    base_pct = base_counts.astype(float) / len(base)
    current_pct = current_counts.astype(float) / len(current)

    base_pct = np.clip(base_pct, eps, None)
    current_pct = np.clip(current_pct, eps, None)

    psi = np.sum(
        (current_pct - base_pct)
        * np.log(current_pct / base_pct)
    )

    return float(max(psi, 0.0))


# ============================================================================
# NUMERICAL DRIFT
# ============================================================================

def calculate_numeric_drift(
    baseline: Sequence[float],
    current: Sequence[float],
    feature: str,
    psi_threshold: float,
    ks_pvalue_threshold: float,
    min_sample_size: int = DEFAULT_MIN_DRIFT_SAMPLE_SIZE,
    psi_bins: int = DEFAULT_PSI_BINS,
) -> DriftMetric:
    """
    Calculate PSI + two-sample KS drift for one numerical feature.
    """

    base = np.asarray(baseline, dtype=float)
    curr = np.asarray(current, dtype=float)

    base = base[np.isfinite(base)]
    curr = curr[np.isfinite(curr)]

    if (
        len(base) < min_sample_size
        or len(curr) < min_sample_size
    ):
        return DriftMetric(
            feature=feature,
            feature_type="numeric",
            baseline_count=len(base),
            current_count=len(curr),
            psi=None,
            ks_statistic=None,
            ks_pvalue=None,
            categorical_pvalue=None,
            drift_detected=False,
            reason="insufficient_samples",
        )

    psi = calculate_psi(
        base,
        curr,
        num_bins=psi_bins,
    )

    ks_result = ks_2samp(
        base,
        curr,
        alternative="two-sided",
        method="auto",
    )

    ks_stat = float(ks_result.statistic)
    ks_pvalue = float(ks_result.pvalue)

    psi_drift = (
        np.isfinite(psi)
        and psi >= psi_threshold
    )

    ks_drift = (
        np.isfinite(ks_pvalue)
        and ks_pvalue < ks_pvalue_threshold
    )

    reasons: List[str] = []

    if psi_drift:
        reasons.append(
            f"psi>={psi_threshold}"
        )

    if ks_drift:
        reasons.append(
            f"ks_pvalue<{ks_pvalue_threshold}"
        )

    return DriftMetric(
        feature=feature,
        feature_type="numeric",
        baseline_count=len(base),
        current_count=len(curr),
        psi=float(psi) if np.isfinite(psi) else None,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pvalue,
        categorical_pvalue=None,
        drift_detected=bool(
            psi_drift or ks_drift
        ),
        reason=";".join(reasons) if reasons else None,
    )


# ============================================================================
# CATEGORICAL DRIFT
# ============================================================================

def calculate_categorical_drift(
    baseline: pd.Series,
    current: pd.Series,
    feature: str,
    pvalue_threshold: float = DEFAULT_CATEGORICAL_PVALUE_THRESHOLD,
    min_sample_size: int = DEFAULT_MIN_DRIFT_SAMPLE_SIZE,
) -> DriftMetric:
    """
    Detect categorical distribution changes using a chi-square test.

    Very rare categories are retained because collapsing categories can hide
    meaningful security-domain changes.
    """

    base = (
        baseline
        .astype("string")
        .fillna("<MISSING>")
    )

    curr = (
        current
        .astype("string")
        .fillna("<MISSING>")
    )

    if (
        len(base) < min_sample_size
        or len(curr) < min_sample_size
    ):
        return DriftMetric(
            feature=feature,
            feature_type="categorical",
            baseline_count=len(base),
            current_count=len(curr),
            psi=None,
            ks_statistic=None,
            ks_pvalue=None,
            categorical_pvalue=None,
            drift_detected=False,
            reason="insufficient_samples",
        )

    categories = sorted(
        set(base.unique()).union(
            set(curr.unique())
        )
    )

    base_counts = (
        base.value_counts()
        .reindex(categories, fill_value=0)
        .astype(float)
    )

    current_counts = (
        curr.value_counts()
        .reindex(categories, fill_value=0)
        .astype(float)
    )

    contingency = np.vstack(
        [
            base_counts.values,
            current_counts.values,
        ]
    )

    # Chi-square requires at least two categories.
    if contingency.shape[1] < 2:
        return DriftMetric(
            feature=feature,
            feature_type="categorical",
            baseline_count=len(base),
            current_count=len(curr),
            psi=None,
            ks_statistic=None,
            ks_pvalue=None,
            categorical_pvalue=None,
            drift_detected=False,
            reason="single_category",
        )

    try:
        _, pvalue, _, _ = chi2_contingency(
            contingency
        )

        pvalue = float(pvalue)

    except ValueError as exc:
        logger.warning(
            "Categorical drift test failed for '%s': %s",
            feature,
            exc,
        )

        return DriftMetric(
            feature=feature,
            feature_type="categorical",
            baseline_count=len(base),
            current_count=len(curr),
            psi=None,
            ks_statistic=None,
            ks_pvalue=None,
            categorical_pvalue=None,
            drift_detected=False,
            reason="statistical_test_failed",
        )

    drifted = (
        np.isfinite(pvalue)
        and pvalue < pvalue_threshold
    )

    return DriftMetric(
        feature=feature,
        feature_type="categorical",
        baseline_count=len(base),
        current_count=len(curr),
        psi=None,
        ks_statistic=None,
        ks_pvalue=None,
        categorical_pvalue=pvalue,
        drift_detected=bool(drifted),
        reason=(
            f"chi_square_pvalue<{pvalue_threshold}"
            if drifted
            else None
        ),
    )


# ============================================================================
# FEATURE DRIFT ENGINE
# ============================================================================

def analyze_feature_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    ks_pvalue_threshold: float = DEFAULT_KS_PVALUE_THRESHOLD,
    categorical_pvalue_threshold: float = DEFAULT_CATEGORICAL_PVALUE_THRESHOLD,
    min_sample_size: int = DEFAULT_MIN_DRIFT_SAMPLE_SIZE,
    psi_bins: int = DEFAULT_PSI_BINS,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Analyze both numerical and categorical feature drift.

    Returns
    -------
    drift_metrics:
        Detailed per-feature metrics.
    drifted_features:
        List of features that crossed configured thresholds.
    """

    if features is None:
        common_columns = sorted(
            set(baseline_df.columns)
            & set(current_df.columns)
        )
    else:
        common_columns = [
            feature
            for feature in features
            if feature in baseline_df.columns
            and feature in current_df.columns
        ]

    metrics: Dict[str, Dict[str, Any]] = {}
    drifted_features: List[str] = []

    for feature in common_columns:

        baseline_series = baseline_df[feature]
        current_series = current_df[feature]

        if (
            pd.api.types.is_numeric_dtype(
                baseline_series
            )
            and pd.api.types.is_numeric_dtype(
                current_series
            )
        ):
            result = calculate_numeric_drift(
                baseline=baseline_series.values,
                current=current_series.values,
                feature=feature,
                psi_threshold=psi_threshold,
                ks_pvalue_threshold=ks_pvalue_threshold,
                min_sample_size=min_sample_size,
                psi_bins=psi_bins,
            )

        else:
            result = calculate_categorical_drift(
                baseline=baseline_series,
                current=current_series,
                feature=feature,
                pvalue_threshold=categorical_pvalue_threshold,
                min_sample_size=min_sample_size,
            )

        result_dict = asdict(result)

        metrics[feature] = result_dict

        if result.drift_detected:
            drifted_features.append(feature)

            logger.warning(
                "Distribution drift detected: feature='%s', type='%s', reason='%s'",
                feature,
                result.feature_type,
                result.reason,
            )

    return metrics, drifted_features


# ============================================================================
# FEEDBACK QUALITY / POISONING ANALYSIS
# ============================================================================

def assess_feedback_quality(
    dataframe: pd.DataFrame,
    entity_id_col: str,
    label_col: str,
    authority_col: str,
    max_unknown_source_ratio: float = DEFAULT_MAX_UNKNOWN_SOURCE_RATIO,
    max_single_entity_ratio: float = DEFAULT_MAX_SINGLE_ENTITY_RATIO,
    max_label_dominance_ratio: float = DEFAULT_MAX_LABEL_DOMINANCE_RATIO,
) -> FeedbackQualityReport:
    """
    Assess feedback quality and basic poisoning indicators.

    This is NOT a full adversarial-ML detector.

    It catches obvious signals such as:
    - excessive unknown-source feedback
    - one entity dominating the batch
    - extreme label concentration
    """

    if dataframe.empty:
        return FeedbackQualityReport(
            input_records=0,
            accepted_records=0,
            rejected_records=0,
            duplicate_records=0,
            invalid_label_records=0,
            missing_entity_records=0,
            missing_timestamp_records=0,
            unknown_authority_records=0,
            unknown_authority_ratio=0.0,
            dominant_label_ratio=0.0,
            dominant_entity_ratio=0.0,
            quality_passed=False,
            poisoning_risk_detected=False,
            warnings=("feedback_dataset_empty",),
        )

    warnings: List[str] = []

    unknown_authority_records = 0

    if authority_col in dataframe.columns:
        authority = (
            dataframe[authority_col]
            .astype("string")
            .fillna("unknown")
            .str.lower()
        )

        unknown_authority_records = int(
            (~authority.isin(
                DEFAULT_AUTHORITY_WEIGHTS.keys()
            )).sum()
        )

        unknown_authority_records += int(
            (authority == "unknown").sum()
        )

    unknown_ratio = (
        unknown_authority_records
        / max(len(dataframe), 1)
    )

    if unknown_ratio > max_unknown_source_ratio:
        warnings.append(
            "excessive_unknown_feedback_sources"
        )

    dominant_label_ratio = 0.0

    if label_col in dataframe.columns:
        counts = dataframe[label_col].value_counts(
            normalize=True
        )

        if not counts.empty:
            dominant_label_ratio = float(
                counts.iloc[0]
            )

    if dominant_label_ratio >= max_label_dominance_ratio:
        warnings.append(
            "extreme_label_concentration"
        )

    dominant_entity_ratio = 0.0

    if entity_id_col in dataframe.columns:
        entity_counts = (
            dataframe[entity_id_col]
            .astype("string")
            .value_counts(
                normalize=True
            )
        )

        if not entity_counts.empty:
            dominant_entity_ratio = float(
                entity_counts.iloc[0]
            )

    if dominant_entity_ratio >= max_single_entity_ratio:
        warnings.append(
            "single_entity_dominates_feedback"
        )

    quality_passed = len(warnings) == 0

    poisoning_risk_detected = bool(
        "excessive_unknown_feedback_sources" in warnings
        or "single_entity_dominates_feedback" in warnings
    )

    return FeedbackQualityReport(
        input_records=len(dataframe),
        accepted_records=len(dataframe),
        rejected_records=0,
        duplicate_records=0,
        invalid_label_records=0,
        missing_entity_records=(
            int(dataframe[entity_id_col].isna().sum())
            if entity_id_col in dataframe.columns
            else len(dataframe)
        ),
        missing_timestamp_records=(
            int(dataframe["_parsed_timestamp"].isna().sum())
            if "_parsed_timestamp" in dataframe.columns
            else 0
        ),
        unknown_authority_records=unknown_authority_records,
        unknown_authority_ratio=float(
            unknown_ratio
        ),
        dominant_label_ratio=float(
            dominant_label_ratio
        ),
        dominant_entity_ratio=float(
            dominant_entity_ratio
        ),
        quality_passed=quality_passed,
        poisoning_risk_detected=poisoning_risk_detected,
        warnings=tuple(warnings),
    )


# ============================================================================
# TRUSTED SAMPLE COUNT
# ============================================================================

def count_trusted_feedback(
    dataframe: pd.DataFrame,
    authority_col: str,
    minimum_authority_weight: int = 50,
) -> int:
    """
    Count feedback records from sufficiently trusted sources.

    User reports are intentionally not automatically treated as equivalent to
    analyst-confirmed labels.
    """

    if dataframe.empty:
        return 0

    if authority_col not in dataframe.columns:
        return 0

    weights = (
        dataframe[authority_col]
        .astype("string")
        .str.lower()
        .map(DEFAULT_AUTHORITY_WEIGHTS)
        .fillna(0)
    )

    return int(
        (weights >= minimum_authority_weight).sum()
    )


# ============================================================================
# RETRAINING COOLDOWN
# ============================================================================

def check_retraining_cooldown(
    state_file: Union[str, Path],
    cooldown_hours: int = DEFAULT_RETRAIN_COOLDOWN_HOURS,
) -> bool:
    """
    Return True if retraining is currently inside the configured cooldown.

    The state file is intentionally small and human-readable.
    """

    path = Path(state_file).resolve()

    if not path.exists():
        return False

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            state = json.load(handle)

        last_trigger = pd.to_datetime(
            state.get("last_retrain_triggered_at"),
            errors="coerce",
            utc=True,
        )

        if pd.isna(last_trigger):
            return False

        now = pd.Timestamp.now(tz="UTC")

        elapsed_hours = (
            now - last_trigger
        ).total_seconds() / 3600.0

        return elapsed_hours < cooldown_hours

    except Exception as exc:
        logger.warning(
            "Unable to evaluate retraining cooldown: %s",
            exc,
        )

        # Fail closed: do not block retraining due to an unreadable state file.
        return False


def update_retraining_state(
    state_file: Union[str, Path],
    trigger_payload: Mapping[str, Any],
) -> None:
    """
    Update retraining state only after an approved retraining decision.
    """

    payload = {
        "last_retrain_triggered_at": _utc_now_iso(),
        "trigger": dict(trigger_payload),
    }

    _atomic_write_json(
        payload,
        state_file,
    )


# ============================================================================
# MAIN STAGE PIPELINE
# ============================================================================

def ingest_and_evaluate_feedback(
    raw_feedback_path: Union[str, Path],
    baseline_train_path: Union[str, Path],
    output_feedback_dir: Union[str, Path],
    target_column: str = "label",
    entity_id_col: str = "url_hash",
    timestamp_col: str = "timestamp",
    authority_col: str = "source_type",
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    ks_pvalue_threshold: float = DEFAULT_KS_PVALUE_THRESHOLD,
    categorical_pvalue_threshold: float = DEFAULT_CATEGORICAL_PVALUE_THRESHOLD,
    drift_ratio_threshold: float = DEFAULT_DRIFT_RATIO_THRESHOLD,
    min_new_samples_for_retrain: int = DEFAULT_MIN_NEW_SAMPLES_FOR_RETRAIN,
    min_trusted_samples_for_retrain: int = DEFAULT_MIN_TRUSTED_SAMPLES_FOR_RETRAIN,
    min_drift_sample_size: int = DEFAULT_MIN_DRIFT_SAMPLE_SIZE,
    retrain_cooldown_hours: int = DEFAULT_RETRAIN_COOLDOWN_HOURS,
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    max_unknown_source_ratio: float = DEFAULT_MAX_UNKNOWN_SOURCE_RATIO,
    max_single_entity_ratio: float = DEFAULT_MAX_SINGLE_ENTITY_RATIO,
    max_label_dominance_ratio: float = DEFAULT_MAX_LABEL_DOMINANCE_RATIO,
    psi_bins: int = DEFAULT_PSI_BINS,
) -> Dict[str, Any]:
    """
    Execute the complete production feedback ingestion pipeline.

    Returns a JSON-serializable governance report.

    The function deliberately does NOT train or deploy anything.
    """

    out_dir = Path(
        output_feedback_dir
    ).resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    feedback_file = Path(
        raw_feedback_path
    ).resolve()

    # ------------------------------------------------------------------
    # Input availability
    # ------------------------------------------------------------------

    if not feedback_file.exists():
        logger.warning(
            "No feedback file found at '%s'.",
            feedback_file,
        )

        payload = {
            "status": "NO_FEEDBACK_FILE",
            "should_retrain": False,
            "module_version": MODULE_VERSION,
            "generated_at": _utc_now_iso(),
        }

        _atomic_write_json(
            payload,
            out_dir / "retrain_trigger.json",
        )

        return payload

    # ------------------------------------------------------------------
    # Fingerprint input
    # ------------------------------------------------------------------

    source_sha256 = calculate_file_sha256(
        feedback_file
    )

    logger.info(
        "Feedback source SHA-256: %s",
        source_sha256,
    )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    raw_df = load_feedback_dataframe(
        feedback_file,
        max_file_size_mb=max_file_size_mb,
    )

    if raw_df.empty:
        payload = {
            "status": "EMPTY_FEEDBACK",
            "should_retrain": False,
            "module_version": MODULE_VERSION,
            "source_sha256": source_sha256,
            "generated_at": _utc_now_iso(),
        }

        _atomic_write_json(
            payload,
            out_dir / "retrain_trigger.json",
        )

        return payload

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    validated_df, validation_stats = validate_feedback_schema(
        dataframe=raw_df,
        entity_id_col=entity_id_col,
        label_col=target_column,
        timestamp_col=timestamp_col,
        authority_col=authority_col,
    )

    if validated_df.empty:
        payload = {
            "status": "NO_VALID_FEEDBACK",
            "should_retrain": False,
            "module_version": MODULE_VERSION,
            "source_sha256": source_sha256,
            "validation": validation_stats,
            "generated_at": _utc_now_iso(),
        }

        _atomic_write_json(
            payload,
            out_dir / "retrain_trigger.json",
        )

        return payload

    # ------------------------------------------------------------------
    # Parse timestamp for quality analysis
    # ------------------------------------------------------------------

    if timestamp_col in validated_df.columns:
        validated_df["_parsed_timestamp"] = pd.to_datetime(
            validated_df[timestamp_col],
            errors="coerce",
            utc=True,
        )
    else:
        validated_df["_parsed_timestamp"] = pd.NaT

    # ------------------------------------------------------------------
    # Quality / poisoning assessment
    # ------------------------------------------------------------------

    quality_report = assess_feedback_quality(
        dataframe=validated_df,
        entity_id_col=entity_id_col,
        label_col=target_column,
        authority_col=authority_col,
        max_unknown_source_ratio=max_unknown_source_ratio,
        max_single_entity_ratio=max_single_entity_ratio,
        max_label_dominance_ratio=max_label_dominance_ratio,
    )

    # ------------------------------------------------------------------
    # Remove internal timestamp helper before reconciliation
    # ------------------------------------------------------------------

    validated_for_reconciliation = validated_df.drop(
        columns=["_parsed_timestamp"],
        errors="ignore",
    )

    # ------------------------------------------------------------------
    # Reconcile current batch
    # ------------------------------------------------------------------

    reconciled_current = reconcile_feedback_records(
        raw_df=validated_for_reconciliation,
        entity_id_col=entity_id_col,
        label_col=target_column,
        timestamp_col=timestamp_col,
        authority_col=authority_col,
    )

    # ------------------------------------------------------------------
    # Existing historical store
    # ------------------------------------------------------------------

    reconciled_file = (
        out_dir
        / "reconciled_feedback.csv"
    )

    if reconciled_file.exists():
        logger.info(
            "Loading existing feedback store: %s",
            reconciled_file,
        )

        historical_df = pd.read_csv(
            reconciled_file,
            low_memory=False,
        )

        historical_before = len(
            historical_df
        )

    else:
        historical_df = pd.DataFrame()
        historical_before = 0

    # ------------------------------------------------------------------
    # Determine genuinely new entities
    # ------------------------------------------------------------------

    if (
        entity_id_col in reconciled_current.columns
        and entity_id_col in historical_df.columns
    ):
        historical_entities = set(
            historical_df[entity_id_col]
            .dropna()
            .astype(str)
        )

        current_entities = (
            reconciled_current[entity_id_col]
            .dropna()
            .astype(str)
        )

        new_entity_mask = ~current_entities.isin(
            historical_entities
        )

        new_feedback_df = reconciled_current.loc[
            new_entity_mask.values
        ].copy()

    else:
        # Without a stable entity identifier, we can only use exact-row
        # deduplication. This is intentionally conservative.
        if historical_df.empty:
            new_feedback_df = reconciled_current.copy()

        else:
            historical_hashes = set(
                pd.util.hash_pandas_object(
                    historical_df.astype(str),
                    index=False,
                ).astype(str)
            )

            current_hashes = pd.util.hash_pandas_object(
                reconciled_current.astype(str),
                index=False,
            ).astype(str)

            new_feedback_df = reconciled_current.loc[
                ~current_hashes.isin(
                    historical_hashes
                )
            ].copy()

    # ------------------------------------------------------------------
    # Merge historical store
    # ------------------------------------------------------------------

    if historical_df.empty:
        combined_df = reconciled_current.copy()

    else:
        combined_df = pd.concat(
            [
                historical_df,
                reconciled_current,
            ],
            ignore_index=True,
        )

        combined_df = reconcile_feedback_records(
            raw_df=combined_df,
            entity_id_col=entity_id_col,
            label_col=target_column,
            timestamp_col=timestamp_col,
            authority_col=authority_col,
        )

    # ------------------------------------------------------------------
    # Persist historical feedback
    # ------------------------------------------------------------------

    _atomic_write_csv(
        combined_df,
        reconciled_file,
    )

    # ------------------------------------------------------------------
    # Baseline drift analysis
    # ------------------------------------------------------------------

    drift_metrics: Dict[str, Dict[str, Any]] = {}
    drifted_features: List[str] = []

    baseline_file = Path(
        baseline_train_path
    ).resolve()

    if baseline_file.exists():
        logger.info(
            "Loading baseline training data from '%s'.",
            baseline_file,
        )

        baseline_df = pd.read_csv(
            baseline_file,
            low_memory=False,
        )

        candidate_features = sorted(
            set(baseline_df.columns)
            & set(new_feedback_df.columns)
        )

        candidate_features = [
            feature
            for feature in candidate_features
            if feature != target_column
        ]

        drift_metrics, drifted_features = (
            analyze_feature_drift(
                baseline_df=baseline_df,
                current_df=new_feedback_df,
                features=candidate_features,
                psi_threshold=psi_threshold,
                ks_pvalue_threshold=ks_pvalue_threshold,
                categorical_pvalue_threshold=categorical_pvalue_threshold,
                min_sample_size=min_drift_sample_size,
                psi_bins=psi_bins,
            )
        )

    else:
        logger.warning(
            "Baseline training file does not exist: '%s'",
            baseline_file,
        )

    evaluated_features = len(
        drift_metrics
    )

    drift_ratio = (
        len(drifted_features)
        / evaluated_features
        if evaluated_features
        else 0.0
    )

    # ------------------------------------------------------------------
    # Trusted feedback
    # ------------------------------------------------------------------

    trusted_new_samples = count_trusted_feedback(
        dataframe=new_feedback_df,
        authority_col=authority_col,
        minimum_authority_weight=50,
    )

    total_new_samples = len(
        new_feedback_df
    )

    # ------------------------------------------------------------------
    # Retraining decision
    # ------------------------------------------------------------------

    reasons: List[str] = []
    blockers: List[str] = []

    if total_new_samples >= min_new_samples_for_retrain:
        reasons.append(
            f"new_feedback_volume={total_new_samples}>="
            f"{min_new_samples_for_retrain}"
        )

    if trusted_new_samples >= min_trusted_samples_for_retrain:
        reasons.append(
            f"trusted_feedback_volume={trusted_new_samples}>="
            f"{min_trusted_samples_for_retrain}"
        )

    if drift_ratio >= drift_ratio_threshold:
        reasons.append(
            f"feature_drift_ratio={drift_ratio:.4f}>="
            f"{drift_ratio_threshold:.4f}"
        )

    if not quality_report.quality_passed:
        blockers.extend(
            quality_report.warnings
        )

    if quality_report.poisoning_risk_detected:
        blockers.append(
            "feedback_poisoning_risk"
        )

    # Cooldown state
    state_file = (
        out_dir
        / "retraining_state.json"
    )

    cooldown_active = check_retraining_cooldown(
        state_file=state_file,
        cooldown_hours=retrain_cooldown_hours,
    )

    if cooldown_active:
        blockers.append(
            "retraining_cooldown_active"
        )

    # ------------------------------------------------------------------
    # Conservative governance policy
    #
    # Retraining requires:
    #
    #   A) enough new feedback
    #   B) enough trusted feedback
    #      OR significant drift
    #   C) quality passed
    #   D) no poisoning signal
    #   E) cooldown inactive
    #
    # ------------------------------------------------------------------

    enough_volume = (
        total_new_samples
        >= min_new_samples_for_retrain
    )

    enough_trusted_volume = (
        trusted_new_samples
        >= min_trusted_samples_for_retrain
    )

    significant_drift = (
        drift_ratio
        >= drift_ratio_threshold
    )

    governance_approval = (
        enough_volume
        and (
            enough_trusted_volume
            or significant_drift
        )
    )

    should_retrain = bool(
        governance_approval
        and quality_report.quality_passed
        and not quality_report.poisoning_risk_detected
        and not cooldown_active
    )

    if should_retrain:
        status = "RETRAIN_RECOMMENDED"

    elif blockers:
        status = "RETRAIN_BLOCKED"

    elif not enough_volume:
        status = "INSUFFICIENT_NEW_FEEDBACK"

    else:
        status = "NO_RETRAIN_REQUIRED"

    # ------------------------------------------------------------------
    # Trigger payload
    # ------------------------------------------------------------------

    generated_at = _utc_now_iso()

    trigger_payload: Dict[str, Any] = {
        "status": status,
        "should_retrain": should_retrain,

        "module": {
            "name": "phishguard_feedback_ingestion",
            "version": MODULE_VERSION,
        },

        "generated_at": generated_at,

        "source": {
            "path": str(feedback_file),
            "sha256": source_sha256,
            "format": feedback_file.suffix.lower(),
        },

        "validation": validation_stats,

        "feedback_quality": asdict(
            quality_report
        ),

        "reconciliation": {
            "current_batch_records": len(
                validated_df
            ),
            "reconciled_current_records": len(
                reconciled_current
            ),
            "historical_records_before": historical_before,
            "historical_records_after": len(
                combined_df
            ),
            "new_entity_records": total_new_samples,
            "trusted_new_samples": trusted_new_samples,
        },

        "drift": {
            "evaluated_features_count": evaluated_features,
            "drifted_features_count": len(
                drifted_features
            ),
            "drifted_features": drifted_features,
            "drift_ratio": drift_ratio,
            "psi_threshold": psi_threshold,
            "ks_pvalue_threshold": ks_pvalue_threshold,
            "categorical_pvalue_threshold": (
                categorical_pvalue_threshold
            ),
            "metrics": drift_metrics,
        },

        "retraining_policy": {
            "min_new_samples": min_new_samples_for_retrain,
            "min_trusted_samples": min_trusted_samples_for_retrain,
            "drift_ratio_threshold": drift_ratio_threshold,
            "cooldown_hours": retrain_cooldown_hours,
            "governance_approval": governance_approval,
            "cooldown_active": cooldown_active,
        },

        "decision": {
            "should_retrain": should_retrain,
            "status": status,
            "reasons": reasons,
            "blockers": blockers,
        },

        "artifacts": {
            "reconciled_feedback": str(
                reconciled_file
            ),
            "drift_report": str(
                out_dir / "drift_report.json"
            ),
            "retrain_trigger": str(
                out_dir / "retrain_trigger.json"
            ),
        },
    }

    # ------------------------------------------------------------------
    # Drift manifest
    # ------------------------------------------------------------------

    drift_report = {
        "generated_at": generated_at,
        "module_version": MODULE_VERSION,
        "source_sha256": source_sha256,
        "evaluated_features_count": evaluated_features,
        "drifted_features_count": len(
            drifted_features
        ),
        "drifted_features": drifted_features,
        "drift_ratio": drift_ratio,
        "psi_threshold": psi_threshold,
        "ks_pvalue_threshold": ks_pvalue_threshold,
        "categorical_pvalue_threshold": (
            categorical_pvalue_threshold
        ),
        "metrics": drift_metrics,
    }

    _atomic_write_json(
        drift_report,
        out_dir / "drift_report.json",
    )

    # ------------------------------------------------------------------
    # Trigger manifest
    # ------------------------------------------------------------------

    _atomic_write_json(
        trigger_payload,
        out_dir / "retrain_trigger.json",
    )

    # ------------------------------------------------------------------
    # Update cooldown state only after a positive recommendation.
    #
    # This records that a retraining recommendation occurred. A more mature
    # system can move this update to Stage 02 after training actually begins.
    # ------------------------------------------------------------------

    if should_retrain:
        update_retraining_state(
            state_file=state_file,
            trigger_payload={
                "status": status,
                "source_sha256": source_sha256,
                "new_samples": total_new_samples,
                "trusted_samples": trusted_new_samples,
                "drift_ratio": drift_ratio,
            },
        )

    logger.info(
        "Stage 05 completed: status=%s, should_retrain=%s, "
        "new_samples=%d, trusted_samples=%d, drift_ratio=%.3f",
        status,
        should_retrain,
        total_new_samples,
        trusted_new_samples,
        drift_ratio,
    )

    return trigger_payload


# ============================================================================
# STAGE ENTRYPOINT
# ============================================================================

def main(
    config_path: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Stage 05 programmatic entrypoint.

    Configuration example:

        feedback_ingestion:
          raw_feedback_path: data/feedback/raw_feedback.csv
          output_dir: data/feedback/processed

          entity_id_column: url_hash
          timestamp_column: timestamp
          authority_column: source_type

          psi_threshold: 0.20
          ks_pvalue_threshold: 0.05
          categorical_pvalue_threshold: 0.05

          drift_ratio_threshold: 0.25

          min_new_samples_for_retrain: 500
          min_trusted_samples_for_retrain: 100

          min_drift_sample_size: 30
          retrain_cooldown_hours: 24

          max_file_size_mb: 250
          max_unknown_source_ratio: 0.50
          max_single_entity_ratio: 0.05
          max_label_dominance_ratio: 0.98

          psi_bins: 10
    """

    config = (
        load_config(config_path)
        if config_path
        else {}
    )

    feedback_cfg = config.get(
        "feedback_ingestion",
        {},
    )

    data_cfg = config.get(
        "data_preparation",
        {},
    )

    data_dir = data_cfg.get(
        "output_dir",
        "data/processed",
    )

    raw_feedback_path = (
        kwargs.get("raw_feedback_path")
        or feedback_cfg.get(
            "raw_feedback_path",
            "data/feedback/raw_feedback.csv",
        )
    )

    baseline_train_path = (
        kwargs.get("baseline_train_path")
        or feedback_cfg.get(
            "baseline_train_path",
            str(
                Path(data_dir)
                / "train.csv"
            ),
        )
    )

    output_feedback_dir = (
        kwargs.get("output_feedback_dir")
        or feedback_cfg.get(
            "output_dir",
            "data/feedback/processed",
        )
    )

    target_column = feedback_cfg.get(
        "target_column",
        data_cfg.get(
            "target_column",
            "label",
        ),
    )

    entity_id_col = feedback_cfg.get(
        "entity_id_column",
        "url_hash",
    )

    timestamp_col = feedback_cfg.get(
        "timestamp_column",
        "timestamp",
    )

    authority_col = feedback_cfg.get(
        "authority_column",
        "source_type",
    )

    return ingest_and_evaluate_feedback(
        raw_feedback_path=raw_feedback_path,
        baseline_train_path=baseline_train_path,
        output_feedback_dir=output_feedback_dir,
        target_column=target_column,
        entity_id_col=entity_id_col,
        timestamp_col=timestamp_col,
        authority_col=authority_col,
        psi_threshold=float(
            feedback_cfg.get(
                "psi_threshold",
                DEFAULT_PSI_THRESHOLD,
            )
        ),
        ks_pvalue_threshold=float(
            feedback_cfg.get(
                "ks_pvalue_threshold",
                DEFAULT_KS_PVALUE_THRESHOLD,
            )
        ),
        categorical_pvalue_threshold=float(
            feedback_cfg.get(
                "categorical_pvalue_threshold",
                DEFAULT_CATEGORICAL_PVALUE_THRESHOLD,
            )
        ),
        drift_ratio_threshold=float(
            feedback_cfg.get(
                "drift_ratio_threshold",
                DEFAULT_DRIFT_RATIO_THRESHOLD,
            )
        ),
        min_new_samples_for_retrain=int(
            feedback_cfg.get(
                "min_new_samples_for_retrain",
                DEFAULT_MIN_NEW_SAMPLES_FOR_RETRAIN,
            )
        ),
        min_trusted_samples_for_retrain=int(
            feedback_cfg.get(
                "min_trusted_samples_for_retrain",
                DEFAULT_MIN_TRUSTED_SAMPLES_FOR_RETRAIN,
            )
        ),
        min_drift_sample_size=int(
            feedback_cfg.get(
                "min_drift_sample_size",
                DEFAULT_MIN_DRIFT_SAMPLE_SIZE,
            )
        ),
        retrain_cooldown_hours=int(
            feedback_cfg.get(
                "retrain_cooldown_hours",
                DEFAULT_RETRAIN_COOLDOWN_HOURS,
            )
        ),
        max_file_size_mb=int(
            feedback_cfg.get(
                "max_file_size_mb",
                DEFAULT_MAX_FILE_SIZE_MB,
            )
        ),
        max_unknown_source_ratio=float(
            feedback_cfg.get(
                "max_unknown_source_ratio",
                DEFAULT_MAX_UNKNOWN_SOURCE_RATIO,
            )
        ),
        max_single_entity_ratio=float(
            feedback_cfg.get(
                "max_single_entity_ratio",
                DEFAULT_MAX_SINGLE_ENTITY_RATIO,
            )
        ),
        max_label_dominance_ratio=float(
            feedback_cfg.get(
                "max_label_dominance_ratio",
                DEFAULT_MAX_LABEL_DOMINANCE_RATIO,
            )
        ),
        psi_bins=int(
            feedback_cfg.get(
                "psi_bins",
                DEFAULT_PSI_BINS,
            )
        ),
    )


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Stage 05: Production Feedback Ingestion, "
            "Reconciliation, Drift Detection & Retraining Governance"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="params.yaml",
        help="Path to pipeline configuration YAML.",
    )

    parser.add_argument(
        "--feedback",
        type=str,
        default=None,
        help="Optional override for raw feedback path.",
    )

    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Optional override for baseline training dataset.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional override for feedback output directory.",
    )

    args = parser.parse_args()

    cli_kwargs: Dict[str, Any] = {}

    if args.feedback:
        cli_kwargs["raw_feedback_path"] = args.feedback

    if args.baseline:
        cli_kwargs["baseline_train_path"] = args.baseline

    if args.output:
        cli_kwargs["output_feedback_dir"] = args.output

    main(
        config_path=args.config,
        **cli_kwargs,
    )


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "MODULE_VERSION",
    "DriftMetric",
    "FeedbackQualityReport",
    "RetrainingDecision",
    "calculate_file_sha256",
    "validate_input_file",
    "load_feedback_dataframe",
    "normalize_binary_label",
    "validate_feedback_schema",
    "reconcile_feedback_records",
    "calculate_psi",
    "calculate_numeric_drift",
    "calculate_categorical_drift",
    "analyze_feature_drift",
    "assess_feedback_quality",
    "count_trusted_feedback",
    "check_retraining_cooldown",
    "update_retraining_state",
    "ingest_and_evaluate_feedback",
    "main",
]



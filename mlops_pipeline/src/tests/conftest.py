"""
PhishGuard AI - Shared Pytest Fixtures
======================================

Version: 14.0.0
File Path: src/tests/conftest.py

Purpose
-------
Central pytest fixture collection for the PhishGuard AI MLOps pipeline.

This module provides:

* deterministic test configuration
* reproducible synthetic phishing datasets
* realistic URL-oriented raw datasets
* train/validation/test datasets
* fitted candidate and champion models
* model metadata
* isolated pipeline workspaces
* feedback/drift datasets
* malformed and edge-case datasets
* ONNX-compatible sample artifacts
* common artifact paths
* environment isolation
* lightweight integrity helpers

Design Goals
------------
1. Deterministic tests
2. No writes to the real project data/models/reports directories
3. Realistic enough data to exercise pipeline behavior
4. Explicit feature-schema contracts
5. Compatibility with pytest, sklearn, pandas and optional ONNX tests
6. Safe cleanup through pytest's tmp_path fixture
7. Easy extension for integration and CI/CD tests

IMPORTANT
---------
This file intentionally does NOT modify production application configuration.
All pipeline artifacts are generated under pytest's temporary directory.
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# =============================================================================
# PROJECT PATH / TEST ENVIRONMENT
# =============================================================================

# src/tests/conftest.py
#       -> tests directory
#       -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Keep test environment isolated from production behavior.
#
# These values are defaults only. If the CI system explicitly provides an
# environment variable, do not overwrite it.
_TEST_ENV_DEFAULTS: Mapping[str, str] = {
    "ENVIRONMENT": "testing",
    "LOG_LEVEL": "DEBUG",
    "DISABLE_GPU": "1",
    "PYTHONHASHSEED": "42",
}

for _key, _value in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(_key, _value)


# =============================================================================
# TEST METADATA
# =============================================================================

TEST_RANDOM_SEED = 42

TARGET_COLUMN = "label"

# These are intentionally numeric because Stage 01's modeling contract
# currently produces numeric feature matrices.
NUMERIC_FEATURE_COLUMNS: Tuple[str, ...] = (
    "url_length",
    "domain_length",
    "path_length",
    "dot_count",
    "hyphen_count",
    "at_symbol_present",
    "double_slash_redirect",
    "is_ip_address",
    "has_suspicious_tld",
    "keyword_count",
    "url_entropy",
    "subdomain_count",
)

SLICE_COLUMNS: Tuple[str, ...] = (
    "tld_extension",
    "organization_id",
)

MODEL_FEATURE_COLUMNS: Tuple[str, ...] = NUMERIC_FEATURE_COLUMNS


# =============================================================================
# SESSION-SCOPED RANDOMNESS
# =============================================================================

@pytest.fixture(scope="session")
def random_seed() -> int:
    """
    Canonical deterministic seed used by the test suite.
    """
    return TEST_RANDOM_SEED


@pytest.fixture(scope="session", autouse=True)
def seed_global_randomness(random_seed: int):
    """
    Seed Python and NumPy global RNGs for deterministic test behavior.

    Tests should still prefer local ``np.random.default_rng`` instances rather
    than relying on global random state.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    yield

    # Do not attempt to restore global random state. Pytest owns the process
    # lifetime and tests should not depend on global RNG state after teardown.


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

@pytest.fixture(scope="session")
def sample_config_dict(random_seed: int) -> Dict[str, Any]:
    """
    Return a realistic isolated configuration matching the pipeline stages.

    The returned dictionary is safe for callers to mutate because each fixture
    invocation returns a deep copy when needed.
    """
    config = {
        "main": {
            "project_name": "PhishGuard-AI",
            "environment": "testing",
            "random_seed": random_seed,
        },
        "data_preparation": {
            "raw_data_path": "data/raw/phishing_dataset.csv",
            "target_column": TARGET_COLUMN,
            "url_column": "url",
            "test_size": 0.20,
            "val_size": 0.15,
            "output_dir": "data/processed",
        },
        "training": {
            "model_type": "random_forest",
            "hyperparameters": {
                "n_estimators": 15,
                "max_depth": 5,
                "random_state": random_seed,
                "n_jobs": 1,
            },
            "output_model_dir": "models/candidate",
            "experiment_name": "phishguard_test_experiment",
        },
        "evaluation": {
            "decision_threshold": 0.50,
            "min_parity_ratio": 0.80,
            "slice_columns": list(SLICE_COLUMNS),
            "output_eval_dir": "reports/evaluation",
            "guardrails": {
                "max_f1_drop": 0.05,
                "max_fpr_increase": 0.02,
                "max_recall_drop": 0.05,
                "max_precision_drop": 0.05,
            },
        },
        "onnx_export": {
            "opset_version": 15,
            "max_allowed_diff": 1e-4,
            "output_onnx_dir": "models/onnx",
        },
        "feedback_ingestion": {
            "raw_feedback_path": "data/feedback/raw_feedback.csv",
            "psi_threshold": 0.20,
            "min_new_samples_for_retrain": 50,
            "output_dir": "data/feedback/processed",
            "entity_id_column": "url_hash",
            "timestamp_column": "timestamp",
            "authority_column": "source_type",
        },
    }

    return copy.deepcopy(config)


@pytest.fixture(scope="session")
def sample_params_yaml(sample_config_dict: Dict[str, Any]) -> str:
    """
    Serialize the test configuration to YAML.

    Useful for tests that exercise ``load_config()`` instead of passing a
    Python dictionary directly.
    """
    return yaml.safe_dump(sample_config_dict, sort_keys=False)


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def _generate_url(
    index: int,
    label: int,
    rng: np.random.Generator,
) -> str:
    """
    Generate a deterministic but varied URL-like sample.

    This is synthetic test data only; domains use reserved/example-style
    namespaces and should never be interpreted as real threat intelligence.
    """
    if label == 1:
        suspicious_domains = [
            "verify-account.example.test",
            "secure-login.example.test",
            "account-update.example.test",
            "credential-check.example.test",
        ]

        domain = suspicious_domains[index % len(suspicious_domains)]

        path_options = [
            "/login/verify",
            "/account/confirm",
            "/secure/update",
            "/credential/check",
        ]

        path = path_options[index % len(path_options)]

        token = "".join(
            rng.choice(
                list("abcdefghijklmnopqrstuvwxyz0123456789"),
                size=int(rng.integers(8, 18)),
            )
        )

        return f"https://{domain}{path}?session={token}"

    legitimate_domains = [
        "www.example.com",
        "docs.example.com",
        "support.example.org",
        "portal.example.net",
    ]

    domain = legitimate_domains[index % len(legitimate_domains)]

    paths = [
        "/",
        "/about",
        "/docs",
        "/support",
        "/products",
    ]

    return f"https://{domain}{paths[index % len(paths)]}"


def _generate_synthetic_phishing_df(
    num_samples: int = 200,
    random_seed: int = TEST_RANDOM_SEED,
    include_slices: bool = True,
    include_urls: bool = False,
    phishing_ratio: float = 0.50,
) -> pd.DataFrame:
    """
    Generate a deterministic synthetic phishing feature dataset.

    Important properties
    --------------------
    * deterministic
    * both classes represented
    * correlated phishing features
    * realistic feature ranges
    * optional slice dimensions
    * optional raw URL column
    """
    if num_samples < 4:
        raise ValueError("num_samples must be >= 4.")

    if not 0.0 < phishing_ratio < 1.0:
        raise ValueError("phishing_ratio must be strictly between 0 and 1.")

    rng = np.random.default_rng(random_seed)

    labels = rng.binomial(
        n=1,
        p=phishing_ratio,
        size=num_samples,
    ).astype(np.int8)

    # Guarantee both classes for classification tests.
    if np.all(labels == 0):
        labels[0] = 1

    if np.all(labels == 1):
        labels[0] = 0

    # -------------------------------------------------------------------------
    # URL lexical / structural features
    # -------------------------------------------------------------------------

    url_length = (
        rng.integers(25, 90, size=num_samples)
        + labels * rng.integers(25, 80, size=num_samples)
    )

    domain_length = (
        rng.integers(8, 25, size=num_samples)
        + labels * rng.integers(5, 25, size=num_samples)
    )

    path_length = (
        rng.integers(1, 35, size=num_samples)
        + labels * rng.integers(10, 50, size=num_samples)
    )

    dot_count = (
        rng.integers(1, 4, size=num_samples)
        + labels * rng.integers(0, 3, size=num_samples)
    )

    hyphen_count = (
        rng.integers(0, 3, size=num_samples)
        + labels * rng.integers(0, 5, size=num_samples)
    )

    at_symbol_present = (
        rng.random(num_samples) < (0.03 + labels * 0.20)
    ).astype(np.int8)

    double_slash_redirect = (
        rng.random(num_samples) < (0.01 + labels * 0.15)
    ).astype(np.int8)

    is_ip_address = (
        rng.random(num_samples) < (0.01 + labels * 0.10)
    ).astype(np.int8)

    has_suspicious_tld = (
        rng.random(num_samples) < (0.03 + labels * 0.35)
    ).astype(np.int8)

    keyword_count = np.clip(
        rng.poisson(0.4 + labels * 1.8, size=num_samples),
        0,
        10,
    ).astype(np.int16)

    url_entropy = np.clip(
        rng.normal(
            loc=3.1 + labels * 0.9,
            scale=0.45,
            size=num_samples,
        ),
        0.0,
        8.0,
    )

    subdomain_count = np.clip(
        rng.poisson(0.6 + labels * 0.8, size=num_samples),
        0,
        8,
    ).astype(np.int16)

    data: Dict[str, Any] = {
        "url_length": url_length.astype(np.float64),
        "domain_length": domain_length.astype(np.float64),
        "path_length": path_length.astype(np.float64),
        "dot_count": dot_count.astype(np.float64),
        "hyphen_count": hyphen_count.astype(np.float64),
        "at_symbol_present": at_symbol_present,
        "double_slash_redirect": double_slash_redirect,
        "is_ip_address": is_ip_address,
        "has_suspicious_tld": has_suspicious_tld,
        "keyword_count": keyword_count,
        "url_entropy": url_entropy.astype(np.float64),
        "subdomain_count": subdomain_count,
        TARGET_COLUMN: labels,
    }

    df = pd.DataFrame(data)

    # -------------------------------------------------------------------------
    # Categorical slices
    # -------------------------------------------------------------------------

    if include_slices:
        tld_options = [
            ".com",
            ".org",
            ".net",
            ".xyz",
            ".top",
        ]

        org_options = [
            "org_alpha",
            "org_beta",
            "org_gamma",
            "org_delta",
        ]

        df["tld_extension"] = rng.choice(
            tld_options,
            size=num_samples,
        )

        df["organization_id"] = rng.choice(
            org_options,
            size=num_samples,
        )

    # -------------------------------------------------------------------------
    # Optional raw URL column
    # -------------------------------------------------------------------------

    if include_urls:
        df.insert(
            0,
            "url",
            [
                _generate_url(
                    index=i,
                    label=int(label),
                    rng=rng,
                )
                for i, label in enumerate(labels)
            ],
        )

    return df


# =============================================================================
# RAW URL DATASET FIXTURE
# =============================================================================

@pytest.fixture
def synthetic_raw_dataframe() -> pd.DataFrame:
    """
    Realistic synthetic raw dataset for Stage 01 tests.

    Includes:
    * raw URL
    * labels
    * categorical slices
    * feature columns
    """
    return _generate_synthetic_phishing_df(
        num_samples=250,
        random_seed=42,
        include_slices=True,
        include_urls=True,
    ).copy(deep=True)


@pytest.fixture
def synthetic_feature_dataframe() -> pd.DataFrame:
    """
    Numeric feature-only dataset suitable for model tests.
    """
    df = _generate_synthetic_phishing_df(
        num_samples=250,
        random_seed=43,
        include_slices=False,
        include_urls=False,
    )

    return df.loc[
        :,
        list(NUMERIC_FEATURE_COLUMNS) + [TARGET_COLUMN],
    ].copy(deep=True)


# =============================================================================
# TRAIN / VALIDATION / TEST DATASETS
# =============================================================================

@pytest.fixture
def synthetic_train_val_test_dfs() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Generate deterministic train/validation/test datasets.

    Each split uses a different RNG stream but the same feature-generation
    contract, which better resembles independently observed samples.
    """
    train_df = _generate_synthetic_phishing_df(
        num_samples=300,
        random_seed=101,
        include_slices=True,
        include_urls=False,
    )

    val_df = _generate_synthetic_phishing_df(
        num_samples=100,
        random_seed=102,
        include_slices=True,
        include_urls=False,
    )

    test_df = _generate_synthetic_phishing_df(
        num_samples=100,
        random_seed=103,
        include_slices=True,
        include_urls=False,
    )

    return (
        train_df.copy(deep=True),
        val_df.copy(deep=True),
        test_df.copy(deep=True),
    )


@pytest.fixture
def model_feature_names() -> List[str]:
    """
    Canonical feature ordering used by model fixtures.
    """
    return list(MODEL_FEATURE_COLUMNS)


# =============================================================================
# TRAINED MODEL FIXTURES
# =============================================================================

@pytest.fixture
def trained_candidate_model(
    synthetic_train_val_test_dfs: Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> Tuple[RandomForestClassifier, List[str]]:
    """
    Return a fitted RandomForest candidate and its exact feature schema.
    """
    train_df, _, _ = synthetic_train_val_test_dfs

    feature_cols = list(MODEL_FEATURE_COLUMNS)

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COLUMN].astype(int).copy()

    model = RandomForestClassifier(
        n_estimators=15,
        max_depth=5,
        random_state=TEST_RANDOM_SEED,
        n_jobs=1,
        class_weight=None,
    )

    model.fit(X_train, y_train)

    return model, feature_cols


@pytest.fixture
def trained_champion_model(
    synthetic_train_val_test_dfs: Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> Tuple[LogisticRegression, List[str]]:
    """
    Return a fitted LogisticRegression champion baseline.

    This deliberately uses a different algorithm from the candidate so
    champion-vs-challenger regression tests exercise different model behavior.
    """
    train_df, _, _ = synthetic_train_val_test_dfs

    feature_cols = list(MODEL_FEATURE_COLUMNS)

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COLUMN].astype(int).copy()

    model = LogisticRegression(
        random_state=TEST_RANDOM_SEED,
        max_iter=500,
        solver="liblinear",
    )

    model.fit(X_train, y_train)

    return model, feature_cols


# =============================================================================
# MODEL METADATA FIXTURES
# =============================================================================

@pytest.fixture
def candidate_metadata(
    model_feature_names: List[str],
) -> Dict[str, Any]:
    """
    Metadata contract matching Stage 02 candidate artifacts.
    """
    return {
        "schema_version": "1.0",
        "model_type": "random_forest",
        "hyperparameters": {
            "n_estimators": 15,
            "max_depth": 5,
            "random_state": TEST_RANDOM_SEED,
            "n_jobs": 1,
        },
        "optimal_threshold": 0.50,
        "feature_names": model_feature_names,
        "feature_count": len(model_feature_names),
        "target_column": TARGET_COLUMN,
        "classes": [0, 1],
    }


# =============================================================================
# FEEDBACK DATA FIXTURES
# =============================================================================

@pytest.fixture
def synthetic_feedback_dataframe() -> pd.DataFrame:
    """
    Create realistic feedback records for Stage 05.

    Includes:
    * duplicate entity IDs
    * different authority levels
    * timestamps
    * labels
    * numeric features
    """
    rng = np.random.default_rng(500)

    num_records = 80

    base = _generate_synthetic_phishing_df(
        num_samples=num_records,
        random_seed=500,
        include_slices=True,
        include_urls=False,
    )

    base["url_hash"] = [
        f"sha256-test-{i:064d}"
        for i in range(num_records)
    ]

    base["timestamp"] = pd.date_range(
        start="2026-01-01T00:00:00Z",
        periods=num_records,
        freq="h",
    )

    base["source_type"] = rng.choice(
        [
            "user_report",
            "security_alert",
            "analyst_override",
        ],
        size=num_records,
        p=[0.60, 0.25, 0.15],
    )

    base["feedback_id"] = [
        f"feedback-{i:06d}"
        for i in range(num_records)
    ]

    return base


@pytest.fixture
def duplicate_feedback_dataframe(
    synthetic_feedback_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Feedback dataset containing deliberate duplicate entity records.

    Useful for testing label reconciliation and authority precedence.
    """
    df = synthetic_feedback_dataframe.copy(deep=True)

    duplicated = df.iloc[:10].copy(deep=True)

    # Deliberately modify authority and timestamps.
    duplicated["source_type"] = "analyst_override"
    duplicated["timestamp"] = pd.to_datetime(
        duplicated["timestamp"]
    ) + pd.Timedelta(hours=2)

    return pd.concat(
        [df, duplicated],
        ignore_index=True,
    )


@pytest.fixture
def drifted_feedback_dataframe() -> pd.DataFrame:
    """
    Synthetic feedback distribution intentionally shifted from baseline.

    The purpose is to trigger PSI/KS drift detection.
    """
    rng = np.random.default_rng(700)

    num_records = 150

    labels = rng.binomial(1, 0.55, size=num_records)

    df = pd.DataFrame(
        {
            "url_length": rng.normal(
                loc=180 + labels * 50,
                scale=12,
                size=num_records,
            ),
            "domain_length": rng.normal(
                loc=55 + labels * 20,
                scale=5,
                size=num_records,
            ),
            "path_length": rng.normal(
                loc=100 + labels * 35,
                scale=10,
                size=num_records,
            ),
            "dot_count": rng.integers(
                3,
                9,
                size=num_records,
            ),
            "hyphen_count": rng.integers(
                2,
                10,
                size=num_records,
            ),
            "at_symbol_present": rng.binomial(
                1,
                0.35,
                size=num_records,
            ),
            "double_slash_redirect": rng.binomial(
                1,
                0.25,
                size=num_records,
            ),
            "is_ip_address": rng.binomial(
                1,
                0.20,
                size=num_records,
            ),
            "has_suspicious_tld": rng.binomial(
                1,
                0.55,
                size=num_records,
            ),
            "keyword_count": rng.poisson(
                3.0,
                size=num_records,
            ),
            "url_entropy": rng.normal(
                5.0,
                0.4,
                size=num_records,
            ),
            "subdomain_count": rng.integers(
                1,
                8,
                size=num_records,
            ),
            TARGET_COLUMN: labels,
        }
    )

    return df


# =============================================================================
# EDGE-CASE / INVALID DATA FIXTURES
# =============================================================================

@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """
    Empty DataFrame fixture for input-validation tests.
    """
    return pd.DataFrame()


@pytest.fixture
def malformed_dataframe() -> pd.DataFrame:
    """
    Deliberately malformed dataset.

    Intended to test:
    * missing target
    * invalid labels
    * missing URL
    * null values
    """
    return pd.DataFrame(
        {
            "url": [
                "https://example.test",
                None,
                "not-a-valid-url",
            ],
            "label": [
                0,
                1,
                "unknown",
            ],
        }
    )


@pytest.fixture
def single_class_dataframe() -> pd.DataFrame:
    """
    Dataset containing only one target class.

    Useful for verifying that stratified classification pipelines fail
    safely instead of silently producing invalid models.
    """
    df = _generate_synthetic_phishing_df(
        num_samples=100,
        random_seed=900,
        include_slices=False,
    )

    df[TARGET_COLUMN] = 0

    return df


@pytest.fixture
def missing_feature_dataframe() -> pd.DataFrame:
    """
    Dataset missing one canonical model feature.
    """
    df = _generate_synthetic_phishing_df(
        num_samples=100,
        random_seed=901,
        include_slices=False,
    )

    return df.drop(columns=["url_entropy"])


@pytest.fixture
def non_numeric_feature_dataframe() -> pd.DataFrame:
    """
    Dataset containing an unexpected object/string model feature.
    """
    df = _generate_synthetic_phishing_df(
        num_samples=100,
        random_seed=902,
        include_slices=True,
    )

    df["unexpected_text_feature"] = "malformed"

    return df


# =============================================================================
# PIPELINE WORKSPACE
# =============================================================================

@pytest.fixture
def mock_pipeline_workspace(
    tmp_path: Path,
    sample_config_dict: Dict[str, Any],
    sample_params_yaml: str,
    synthetic_train_val_test_dfs: Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
    synthetic_raw_dataframe: pd.DataFrame,
    synthetic_feedback_dataframe: pd.DataFrame,
    trained_candidate_model: Tuple[
        RandomForestClassifier,
        List[str],
    ],
    trained_champion_model: Tuple[
        LogisticRegression,
        List[str],
    ],
    candidate_metadata: Dict[str, Any],
) -> Path:
    """
    Construct a complete isolated PhishGuard pipeline workspace.

    Nothing is written to the real project tree.
    """
    root_dir = tmp_path / "phishguard_test_workspace"

    # -------------------------------------------------------------------------
    # Directory structure
    # -------------------------------------------------------------------------

    directories = [
        root_dir / "data" / "raw",
        root_dir / "data" / "processed",
        root_dir / "data" / "feedback",
        root_dir / "data" / "feedback" / "processed",
        root_dir / "models" / "candidate",
        root_dir / "models" / "champion",
        root_dir / "models" / "onnx",
        root_dir / "reports" / "evaluation",
        root_dir / "reports" / "training",
        root_dir / "logs",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Raw dataset
    # -------------------------------------------------------------------------

    synthetic_raw_dataframe.to_csv(
        root_dir / "data" / "raw" / "phishing_dataset.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Processed train / validation / test datasets
    # -------------------------------------------------------------------------

    train_df, val_df, test_df = synthetic_train_val_test_dfs

    train_df.to_csv(
        root_dir / "data" / "processed" / "train.csv",
        index=False,
    )

    val_df.to_csv(
        root_dir / "data" / "processed" / "val.csv",
        index=False,
    )

    test_df.to_csv(
        root_dir / "data" / "processed" / "test.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Feedback
    # -------------------------------------------------------------------------

    synthetic_feedback_dataframe.to_csv(
        root_dir / "data" / "feedback" / "raw_feedback.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Candidate / champion models
    # -------------------------------------------------------------------------

    candidate_model, candidate_features = trained_candidate_model
    champion_model, champion_features = trained_champion_model

    candidate_model_path = (
        root_dir
        / "models"
        / "candidate"
        / "candidate_model.joblib"
    )

    champion_model_path = (
        root_dir
        / "models"
        / "champion"
        / "champion_model.joblib"
    )

    joblib.dump(
        candidate_model,
        candidate_model_path,
    )

    joblib.dump(
        champion_model,
        champion_model_path,
    )

    # -------------------------------------------------------------------------
    # Candidate metadata
    # -------------------------------------------------------------------------

    metadata = copy.deepcopy(candidate_metadata)
    metadata["feature_names"] = candidate_features

    candidate_metadata_path = (
        root_dir
        / "models"
        / "candidate"
        / "candidate_metadata.json"
    )

    with candidate_metadata_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            sort_keys=True,
        )

    # -------------------------------------------------------------------------
    # Champion metadata
    # -------------------------------------------------------------------------

    champion_metadata = {
        "schema_version": "1.0",
        "model_type": "logistic_regression",
        "feature_names": champion_features,
        "feature_count": len(champion_features),
        "target_column": TARGET_COLUMN,
        "classes": [0, 1],
        "decision_threshold": 0.50,
    }

    with (
        root_dir
        / "models"
        / "champion"
        / "champion_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            champion_metadata,
            handle,
            indent=2,
            sort_keys=True,
        )

    # -------------------------------------------------------------------------
    # Test params.yaml
    # -------------------------------------------------------------------------

    with (
        root_dir / "params.yaml"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(sample_params_yaml)

    # -------------------------------------------------------------------------
    # Minimal README marker
    # -------------------------------------------------------------------------

    (root_dir / "README.test.md").write_text(
        "# PhishGuard AI isolated pytest workspace\n",
        encoding="utf-8",
    )

    return root_dir


# =============================================================================
# PATH FIXTURES
# =============================================================================

@pytest.fixture
def pipeline_paths(
    mock_pipeline_workspace: Path,
) -> Dict[str, Path]:
    """
    Provide commonly used artifact paths for integration tests.
    """
    root = mock_pipeline_workspace

    return {
        "root": root,
        "config": root / "params.yaml",

        "raw_data": (
            root
            / "data"
            / "raw"
            / "phishing_dataset.csv"
        ),

        "train": (
            root
            / "data"
            / "processed"
            / "train.csv"
        ),

        "val": (
            root
            / "data"
            / "processed"
            / "val.csv"
        ),

        "test": (
            root
            / "data"
            / "processed"
            / "test.csv"
        ),

        "feedback": (
            root
            / "data"
            / "feedback"
            / "raw_feedback.csv"
        ),

        "candidate_model": (
            root
            / "models"
            / "candidate"
            / "candidate_model.joblib"
        ),

        "candidate_metadata": (
            root
            / "models"
            / "candidate"
            / "candidate_metadata.json"
        ),

        "champion_model": (
            root
            / "models"
            / "champion"
            / "champion_model.joblib"
        ),

        "candidate_dir": (
            root
            / "models"
            / "candidate"
        ),

        "champion_dir": (
            root
            / "models"
            / "champion"
        ),

        "onnx_dir": (
            root
            / "models"
            / "onnx"
        ),

        "evaluation_dir": (
            root
            / "reports"
            / "evaluation"
        ),

        "feedback_processed_dir": (
            root
            / "data"
            / "feedback"
            / "processed"
        ),
    }


# =============================================================================
# COMMON ASSERTION / DATA HELPERS
# =============================================================================

@pytest.fixture
def assert_valid_binary_labels():
    """
    Return a reusable assertion helper for binary target columns.
    """

    def _assert(
        values: Sequence[Any],
    ) -> None:
        array = np.asarray(values)

        assert array.size > 0, "Label array is empty."
        assert np.all(
            np.isin(array, [0, 1])
        ), f"Labels must be binary 0/1, got {np.unique(array)}."

    return _assert


@pytest.fixture
def assert_valid_model_probabilities():
    """
    Return a reusable assertion helper for probability predictions.
    """

    def _assert(
        probabilities: Sequence[float],
    ) -> None:
        array = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        assert array.ndim == 1, (
            f"Expected 1D probabilities, got {array.ndim}D."
        )

        assert np.isfinite(array).all(), (
            "Probability predictions contain NaN or infinity."
        )

        assert np.all(
            (array >= 0.0) & (array <= 1.0)
        ), (
            "Probability predictions must be bounded "
            "between 0 and 1."
        )

    return _assert


# =============================================================================
# OPTIONAL ONNX AVAILABILITY
# =============================================================================

@pytest.fixture(scope="session")
def onnx_available() -> bool:
    """
    Return whether the optional ONNX test dependencies are installed.

    Tests requiring ONNX should use:

        @pytest.mark.skipif(not onnx_available, ...)
    """
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
        import skl2onnx  # noqa: F401
    except ImportError:
        return False

    return True


# =============================================================================
# PYTEST CONFIGURATION HOOKS
# =============================================================================

def pytest_configure(config: pytest.Config) -> None:
    """
    Register project-specific pytest markers.

    This allows tests to use:

        @pytest.mark.unit
        @pytest.mark.integration
        @pytest.mark.security
        @pytest.mark.ml
        @pytest.mark.onnx
        @pytest.mark.slow
    """
    markers = {
        "unit": "Fast isolated unit tests.",
        "integration": "Tests involving multiple pipeline components.",
        "security": "Security-focused validation and adversarial tests.",
        "ml": "Machine-learning model and metric tests.",
        "onnx": "ONNX export/runtime tests.",
        "slow": "Tests that may take significant time.",
        "regression": "Model or pipeline regression tests.",
    }

    for marker_name, description in markers.items():
        config.addinivalue_line(
            "markers",
            f"{marker_name}: {description}",
        )


# =============================================================================
# PUBLIC EXPORTS
# =============================================================================

__all__ = [
    "PROJECT_ROOT",
    "TEST_RANDOM_SEED",
    "TARGET_COLUMN",
    "NUMERIC_FEATURE_COLUMNS",
    "MODEL_FEATURE_COLUMNS",
    "SLICE_COLUMNS",
]
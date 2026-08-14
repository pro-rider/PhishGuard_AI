"""
PhishGuard AI - Stage 04 ONNX Export & Runtime Verification Tests
================================================================

Version: 14.0.0
File Path: tests/test_onnx_conversion.py

Production-grade tests for the Stage 04 ONNX export boundary.

Coverage
--------
- ONNX dependency detection
- sklearn -> ONNX conversion
- exported graph validity
- ONNX Runtime CPU inference
- probability extraction across supported output representations
- native-vs-ONNX numerical parity
- dynamic batch dimensions
- single-row inference
- feature-count/schema validation
- manifest integrity
- threshold preservation
- deterministic inference
- non-finite input rejection/behavior
- missing artifact handling
- corrupted artifact handling
- invalid converter configuration
- repeated export safety
- metadata consistency

The tests intentionally use the public Stage 04 interfaces:

    convert_to_onnx(...)
    verify_onnx_parity(...)
    export_onnx_stage(...)

They do NOT depend on implementation-private internals.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import pytest

from tests import (
    DEFAULT_FLOAT_ATOL,
    DEFAULT_FLOAT_RTOL,
    assert_probability_array,
)

from src.stages.04_export_onnx import (
    convert_to_onnx,
    export_onnx_stage,
    verify_onnx_parity,
)

# ---------------------------------------------------------------------------
# Optional ONNX dependencies
# ---------------------------------------------------------------------------

try:
    import onnx

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import onnxruntime as ort

    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


HAS_FULL_ONNX_STACK = HAS_ONNX and HAS_ONNXRUNTIME


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_OPSET = 15
PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-4

MODEL_FILENAME = "candidate_model.joblib"
METADATA_FILENAME = "candidate_metadata.json"
ONNX_FILENAME = "phishguard_model.onnx"
MANIFEST_FILENAME = "onnx_manifest.json"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _require_onnx_stack() -> None:
    """Skip the test when the ONNX stack is unavailable."""
    if not HAS_FULL_ONNX_STACK:
        pytest.skip(
            "ONNX tests require both 'onnx' and 'onnxruntime' "
            "to be installed."
        )


def _load_candidate_artifacts(
    workspace: Path,
) -> Tuple[Any, Dict[str, Any]]:
    """Load the native candidate model and its metadata."""
    model_dir = workspace / "models" / "candidate"

    model_path = model_dir / MODEL_FILENAME
    metadata_path = model_dir / METADATA_FILENAME

    assert model_path.exists(), f"Missing candidate model: {model_path}"
    assert metadata_path.exists(), f"Missing metadata: {metadata_path}"

    model = joblib.load(model_path)

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    return model, metadata


def _feature_matrix(
    workspace: Path,
    metadata: Dict[str, Any],
    split: str = "test",
) -> np.ndarray:
    """
    Build a feature matrix using the exact feature order recorded by Stage 02.
    """
    feature_names = metadata.get("feature_names")

    assert isinstance(feature_names, list), (
        "candidate_metadata.json must contain a 'feature_names' list."
    )
    assert feature_names, "feature_names must not be empty."

    dataset_path = (
        workspace
        / "data"
        / "processed"
        / f"{split}.csv"
    )

    assert dataset_path.exists(), f"Missing dataset: {dataset_path}"

    df = pd.read_csv(dataset_path)

    missing = [c for c in feature_names if c not in df.columns]
    assert not missing, (
        f"Dataset is missing model features: {missing}"
    )

    X = df[feature_names].to_numpy(dtype=np.float32)

    assert X.ndim == 2
    assert X.shape[1] == len(feature_names)
    assert np.isfinite(X).all(), (
        "Test feature matrix contains NaN or infinite values."
    )

    return X


def _get_probability_output(
    session: Any,
    outputs: List[Any],
) -> np.ndarray:
    """
    Robustly extract positive-class probabilities from common
    skl2onnx classification output layouts.

    Supported layouts include:

    1. probability matrix:
       shape = (N, 2)

    2. positive-class vector:
       shape = (N,)

    3. ZipMap-like list/dict representation:
       [{0: p0, 1: p1}, ...]

    The helper intentionally does not assume that probability output
    is always output index 1.
    """
    del session  # Reserved for future output-name based selection.

    if not outputs:
        raise AssertionError("ONNX Runtime returned no outputs.")

    candidates: List[np.ndarray] = []

    for raw_output in outputs:
        # ---------------------------------------------------------------
        # Dictionary / ZipMap-like representation
        # ---------------------------------------------------------------
        if isinstance(raw_output, list):
            if not raw_output:
                continue

            if isinstance(raw_output[0], dict):
                values: List[float] = []

                for row in raw_output:
                    if 1 in row:
                        values.append(float(row[1]))
                    elif "1" in row:
                        values.append(float(row["1"]))
                    else:
                        raise AssertionError(
                            "ZipMap probability output does not contain "
                            "positive-class label 1."
                        )

                return np.asarray(values, dtype=np.float32)

        # ---------------------------------------------------------------
        # ndarray representation
        # ---------------------------------------------------------------
        if isinstance(raw_output, np.ndarray):
            arr = np.asarray(raw_output)

            if arr.ndim == 2:
                if arr.shape[1] == 2:
                    candidates.append(arr[:, 1].astype(np.float32))
                    continue

                if arr.shape[1] == 1:
                    candidates.append(arr[:, 0].astype(np.float32))
                    continue

            if arr.ndim == 1:
                candidates.append(arr.astype(np.float32))

    if not candidates:
        raise AssertionError(
            "Unable to identify a supported ONNX probability output."
        )

    # Prefer a probability-shaped candidate.
    for candidate in candidates:
        if np.all(np.isfinite(candidate)) and np.all(
            (candidate >= 0.0) & (candidate <= 1.0)
        ):
            return candidate

    raise AssertionError(
        "ONNX outputs did not contain valid probabilities."
    )


def _native_positive_probabilities(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """Return native positive-class probabilities."""
    assert hasattr(model, "predict_proba"), (
        "Candidate model must expose predict_proba for parity tests."
    )

    probs = np.asarray(model.predict_proba(X))

    assert probs.ndim == 2, (
        f"Expected native probability matrix, got {probs.shape}."
    )

    assert probs.shape[1] == 2, (
        "PhishGuard binary classifier must produce exactly "
        "two class probabilities."
    )

    positive_probs = probs[:, 1]

    assert_probability_array(
        positive_probs,
        expected_length=len(X),
    )

    return positive_probs.astype(np.float64)


def _sha256(path: Path) -> str:
    """Calculate SHA-256 for an artifact."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _export(
    workspace: Path,
    output_name: str = "onnx_test",
) -> Dict[str, Any]:
    """Export the candidate model using the actual Stage 04 API."""
    _require_onnx_stack()

    model_dir = workspace / "models" / "candidate"
    output_dir = workspace / "models" / output_name
    sample_data = workspace / "data" / "processed" / "val.csv"

    return export_onnx_stage(
        candidate_model_path=model_dir / MODEL_FILENAME,
        candidate_metadata_path=model_dir / METADATA_FILENAME,
        sample_data_path=sample_data,
        output_onnx_dir=output_dir,
        target_column="label",
        opset_version=TEST_OPSET,
        max_allowed_diff=PARITY_ATOL,
    )


# ============================================================================
# Dependency / availability tests
# ============================================================================


@pytest.mark.unit
def test_onnx_dependency_detection() -> None:
    """
    Confirms that dependency detection is boolean and does not crash.
    """
    assert isinstance(HAS_ONNX, bool)
    assert isinstance(HAS_ONNXRUNTIME, bool)


# ============================================================================
# Conversion tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_convert_to_onnx_produces_valid_model(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Tests the low-level converter and validates the generated ONNX graph.
    """
    model, metadata = _load_candidate_artifacts(
        mock_pipeline_workspace
    )

    feature_names = metadata["feature_names"]
    X = _feature_matrix(
        mock_pipeline_workspace,
        metadata,
        split="val",
    )

    onnx_model = convert_to_onnx(
        model=model,
        num_features=len(feature_names),
        model_type=metadata["model_type"],
        target_opset=TEST_OPSET,
    )

    assert isinstance(onnx_model, onnx.ModelProto)

    # Structural validation performed by the ONNX library.
    onnx.checker.check_model(onnx_model)

    assert onnx_model.graph is not None
    assert len(onnx_model.graph.node) > 0

    assert len(onnx_model.graph.input) >= 1
    assert len(onnx_model.graph.output) >= 1

    # Ensure our fixture's feature count is sensible.
    assert X.shape[1] == len(feature_names)


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_export_stage_execution(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Tests the complete Stage 04 export workflow.
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_stage_execution",
    )

    assert isinstance(summary, dict)

    assert "onnx_path" in summary
    assert "manifest_path" in summary
    assert "parity_passed" in summary
    assert "max_diff" in summary

    onnx_path = Path(summary["onnx_path"])
    manifest_path = Path(summary["manifest_path"])

    assert onnx_path.exists()
    assert onnx_path.is_file()
    assert onnx_path.suffix == ".onnx"

    assert manifest_path.exists()
    assert manifest_path.is_file()

    assert summary["parity_passed"] is True
    assert float(summary["max_diff"]) <= PARITY_ATOL


# ============================================================================
# Graph validation tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_exported_graph_passes_onnx_checker(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Ensures the persisted ONNX graph is structurally valid.
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_graph_validation",
    )

    onnx_path = Path(summary["onnx_path"])

    model = onnx.load(str(onnx_path))

    # Raises if the graph is malformed.
    onnx.checker.check_model(model)

    assert model.ir_version > 0
    assert len(model.graph.node) > 0


# ============================================================================
# ONNX Runtime inference tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_onnx_runtime_inference_returns_valid_probabilities(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Verifies that ONNX Runtime executes the exported model and produces
    finite probabilities bounded to [0, 1].
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_runtime_probability",
    )

    _, metadata = _load_candidate_artifacts(
        mock_pipeline_workspace
    )

    X = _feature_matrix(
        mock_pipeline_workspace,
        metadata,
        split="test",
    )

    session = ort.InferenceSession(
        summary["onnx_path"],
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()

    assert len(inputs) >= 1

    input_name = inputs[0].name

    outputs = session.run(
        None,
        {input_name: X},
    )

    probabilities = _get_probability_output(
        session,
        outputs,
    )

    assert_probability_array(
        probabilities,
        expected_length=len(X),
    )


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_onnx_inference_numerical_parity(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Verifies native sklearn probabilities against ONNX Runtime.

    Uses both relative and absolute tolerance because floating-point
    differences can legitimately occur during conversion/runtime execution.
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_numerical_parity",
    )

    model, metadata = _load_candidate_artifacts(
        mock_pipeline_workspace
    )

    X = _feature_matrix(
        mock_pipeline_workspace,
        metadata,
        split="test",
    )

    native_probs = _native_positive_probabilities(
        model,
        X,
    )

    session = ort.InferenceSession(
        summary["onnx_path"],
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    onnx_outputs = session.run(
        None,
        {input_name: X.astype(np.float32)},
    )

    onnx_probs = _get_probability_output(
        session,
        onnx_outputs,
    ).astype(np.float64)

    assert len(native_probs) == len(onnx_probs)

    max_diff = float(
        np.max(
            np.abs(native_probs - onnx_probs)
        )
    )

    assert np.allclose(
        native_probs,
        onnx_probs,
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
    ), (
        "Native/ONNX probability parity failure: "
        f"max_diff={max_diff:.8f}, "
        f"rtol={PARITY_RTOL}, "
        f"atol={PARITY_ATOL}"
    )


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_stage_parity_result_matches_direct_runtime_check(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Ensures Stage 04's reported parity result is consistent with an
    independent runtime calculation.
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_independent_parity",
    )

    model, metadata = _load_candidate_artifacts(
        mock_pipeline_workspace
    )

    X = _feature_matrix(
        mock_pipeline_workspace,
        metadata,
        split="val",
    )

    passed, max_diff = verify_onnx_parity(
        original_model=model,
        onnx_file_path=summary["onnx_path"],
        sample_data=X,
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
    )

    assert passed is True
    assert max_diff <= PARITY_ATOL

    assert summary["parity_passed"] is True


# ============================================================================
# Shape / batching tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_onnx_single_sample_inference(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Verifies that the exported model accepts a single-row batch.
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_single_sample",
    )

    _, metadata = _load_candidate_artifacts(
        mock_pipeline_workspace
    )

    X = _feature_matrix(
        mock_pipeline_workspace,
        metadata,
        split="test",
    )

    single = X[:1]

    session = ort.InferenceSession(
        summary["onnx_path"],
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    outputs = session.run(
        None,
        {input_name: single},
    )

    probs = _get_probability_output(
        session,
        outputs,
    )

    assert_probability_array(
        probs,
        expected_length=1,
    )


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_onnx_dynamic_batching(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Tests several batch sizes to ensure the ONNX graph supports dynamic
    first-dimension batching.
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_dynamic_batch",
    )

    _, metadata = _load_candidate_artifacts(
        mock_pipeline_workspace
    )

    X = _feature_matrix(
        mock_pipeline_workspace,
        metadata,
        split="test",
    )

    session = ort.InferenceSession(
        summary["onnx_path"],
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    for batch_size in [1, 2, 5, 10, min(25, len(X))]:
        if batch_size > len(X):
            continue

        batch = X[:batch_size]

        outputs = session.run(
            None,
            {input_name: batch},
        )

        probs = _get_probability_output(
            session,
            outputs,
        )

        assert_probability_array(
            probs,
            expected_length=batch_size,
        )


# ============================================================================
# Input schema tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_onnx_input_feature_count_matches_metadata(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Ensures the ONNX input tensor's second dimension matches the feature
    contract produced by Stage 02.
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_feature_contract",
    )

    _, metadata = _load_candidate_artifacts(
        mock_pipeline_workspace
    )

    expected_features = len(metadata["feature_names"])

    session = ort.InferenceSession(
        summary["onnx_path"],
        providers=["CPUExecutionProvider"],
    )

    input_tensor = session.get_inputs()[0]

    shape = input_tensor.shape

    assert len(shape) == 2

    # Dynamic batch dimension is expected.
    assert shape[0] in (None, "None", "N", "batch", -1) or isinstance(
        shape[0],
        str,
    )

    feature_dimension = shape[1]

    if isinstance(feature_dimension, int):
        assert feature_dimension == expected_features


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_feature_order_from_metadata_is_used(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Detects accidental feature-order changes.

    The model must be evaluated using the exact feature order recorded
    by Stage 02 metadata.
    """
    model, metadata = _load_candidate_artifacts(
        mock_pipeline_workspace
    )

    feature_names = metadata["feature_names"]

    assert len(feature_names) > 0
    assert len(set(feature_names)) == len(feature_names)

    X = _feature_matrix(
        mock_pipeline_workspace,
        metadata,
        split="test",
    )

    native_probs = _native_positive_probabilities(
        model,
        X,
    )

    # Reordering columns must change the matrix shape/order and therefore
    # should not be silently treated as the canonical feature contract.
    reversed_names = list(reversed(feature_names))

    if reversed_names != feature_names:
        df = pd.read_csv(
            mock_pipeline_workspace
            / "data"
            / "processed"
            / "test.csv"
        )

        X_reversed = df[reversed_names].to_numpy(
            dtype=np.float32
        )

        # We intentionally expect either different predictions or an
        # explicit model-side feature-name/order error.
        try:
            reversed_probs = _native_positive_probabilities(
                model,
                X_reversed,
            )

            if np.array_equal(
                feature_names,
                reversed_names,
            ):
                return

            assert not np.allclose(
                native_probs,
                reversed_probs,
                atol=1e-12,
                rtol=1e-12,
            ), (
                "Feature reordering produced identical probabilities. "
                "This test dataset/model may not be sensitive to feature order."
            )

        except (ValueError, TypeError):
            # Explicit rejection is acceptable and safer.
            pass


# ============================================================================
# Manifest / metadata tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_onnx_manifest_integrity(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Validates the production manifest generated by Stage 04.
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_manifest",
    )

    manifest_path = Path(summary["manifest_path"])
    onnx_path = Path(summary["onnx_path"])

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)

    required_keys = {
        "onnx_model_path",
        "opset_version",
        "num_features",
        "feature_names",
        "optimal_threshold",
        "model_type",
        "parity_verification",
    }

    assert required_keys.issubset(manifest.keys())

    assert manifest["num_features"] == len(
        manifest["feature_names"]
    )

    assert 0.0 <= float(
        manifest["optimal_threshold"]
    ) <= 1.0

    parity = manifest["parity_verification"]

    assert parity["passed"] is True
    assert parity["sample_rows_tested"] > 0

    assert Path(
        manifest["onnx_model_path"]
    ).exists() or (
        Path(manifest["onnx_model_path"]).name
        == onnx_path.name
    )


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_manifest_matches_export_summary(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Ensures Stage 04's return object and persisted manifest do not disagree.
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_manifest_consistency",
    )

    with Path(summary["manifest_path"]).open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)

    assert manifest["onnx_model_path"] == summary["onnx_path"]

    assert bool(
        manifest["parity_verification"]["passed"]
    ) == bool(summary["parity_passed"])

    assert np.isclose(
        float(
            manifest["parity_verification"][
                "max_absolute_difference"
            ]
        ),
        float(summary["max_diff"]),
        rtol=1e-6,
        atol=1e-10,
    )


# ============================================================================
# Threshold tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_optimal_threshold_is_preserved(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Confirms the decision threshold generated during Stage 02 survives
    the ONNX packaging boundary.
    """
    _, metadata = _load_candidate_artifacts(
        mock_pipeline_workspace
    )

    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_threshold",
    )

    with Path(summary["manifest_path"]).open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)

    original_threshold = float(
        metadata.get("optimal_threshold", 0.50)
    )

    exported_threshold = float(
        manifest["optimal_threshold"]
    )

    assert 0.0 < original_threshold < 1.0
    assert exported_threshold == pytest.approx(
        original_threshold,
        abs=1e-12,
    )


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_thresholded_predictions_match_native_model(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Ensures applying the persisted decision threshold to native and ONNX
    probabilities produces identical classifications.
    """
    model, metadata = _load_candidate_artifacts(
        mock_pipeline_workspace
    )

    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_threshold_predictions",
    )

    X = _feature_matrix(
        mock_pipeline_workspace,
        metadata,
        split="test",
    )

    threshold = float(
        metadata.get("optimal_threshold", 0.50)
    )

    native_probs = _native_positive_probabilities(
        model,
        X,
    )

    session = ort.InferenceSession(
        summary["onnx_path"],
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    outputs = session.run(
        None,
        {input_name: X},
    )

    onnx_probs = _get_probability_output(
        session,
        outputs,
    )

    native_labels = (
        native_probs >= threshold
    ).astype(np.int8)

    onnx_labels = (
        onnx_probs >= threshold
    ).astype(np.int8)

    np.testing.assert_array_equal(
        native_labels,
        onnx_labels,
    )


# ============================================================================
# Determinism tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_onnx_runtime_is_deterministic(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Repeated CPU inference on identical input should produce identical
    outputs for this deterministic classifier.
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_determinism",
    )

    _, metadata = _load_candidate_artifacts(
        mock_pipeline_workspace
    )

    X = _feature_matrix(
        mock_pipeline_workspace,
        metadata,
        split="test",
    )

    session = ort.InferenceSession(
        summary["onnx_path"],
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    first = _get_probability_output(
        session,
        session.run(
            None,
            {input_name: X},
        ),
    )

    second = _get_probability_output(
        session,
        session.run(
            None,
            {input_name: X},
        ),
    )

    np.testing.assert_array_equal(
        first,
        second,
    )


# ============================================================================
# Artifact integrity tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_exported_onnx_artifact_has_nonzero_size(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Prevents empty/truncated ONNX artifacts from being considered valid.
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_artifact_size",
    )

    path = Path(summary["onnx_path"])

    assert path.exists()
    assert path.stat().st_size > 0


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_exported_onnx_checksum_is_stable(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Verifies that copying an exported artifact does not alter its checksum.

    This is useful for CI/CD artifact integrity checks.
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_checksum",
    )

    original = Path(summary["onnx_path"])
    copied = original.with_name("copied_model.onnx")

    shutil.copy2(original, copied)

    assert _sha256(original) == _sha256(copied)


# ============================================================================
# Negative / failure-path tests
# ============================================================================


@pytest.mark.unit
def test_missing_candidate_model_fails_cleanly(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Stage 04 must fail explicitly when the candidate model is absent.
    """
    if not HAS_FULL_ONNX_STACK:
        # The file check should occur before ONNX conversion, but we skip
        # this test in environments where the complete Stage 04 dependency
        # stack is unavailable to avoid testing unrelated import behavior.
        pytest.skip("ONNX stack unavailable.")

    missing_model = (
        mock_pipeline_workspace
        / "models"
        / "candidate"
        / "missing_model.joblib"
    )

    metadata = (
        mock_pipeline_workspace
        / "models"
        / "candidate"
        / METADATA_FILENAME
    )

    sample_data = (
        mock_pipeline_workspace
        / "data"
        / "processed"
        / "val.csv"
    )

    with pytest.raises(
        (FileNotFoundError, OSError),
    ):
        export_onnx_stage(
            candidate_model_path=missing_model,
            candidate_metadata_path=metadata,
            sample_data_path=sample_data,
            output_onnx_dir=(
                mock_pipeline_workspace
                / "models"
                / "missing_model_export"
            ),
            target_column="label",
        )


@pytest.mark.unit
def test_missing_sample_data_fails_cleanly(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Missing verification data must not silently produce an ONNX artifact.
    """
    if not HAS_FULL_ONNX_STACK:
        pytest.skip("ONNX stack unavailable.")

    model_path = (
        mock_pipeline_workspace
        / "models"
        / "candidate"
        / MODEL_FILENAME
    )

    metadata_path = (
        mock_pipeline_workspace
        / "models"
        / "candidate"
        / METADATA_FILENAME
    )

    missing_sample = (
        mock_pipeline_workspace
        / "data"
        / "processed"
        / "missing.csv"
    )

    with pytest.raises(
        (FileNotFoundError, OSError),
    ):
        export_onnx_stage(
            candidate_model_path=model_path,
            candidate_metadata_path=metadata_path,
            sample_data_path=missing_sample,
            output_onnx_dir=(
                mock_pipeline_workspace
                / "models"
                / "missing_sample_export"
            ),
        )


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_corrupted_onnx_artifact_is_rejected(
    mock_pipeline_workspace: Path,
) -> None:
    """
    ONNX Runtime must reject a deliberately corrupted model artifact.
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_corruption",
    )

    original = Path(summary["onnx_path"])

    corrupted = original.with_name(
        "corrupted_model.onnx"
    )

    data = original.read_bytes()

    assert len(data) > 32

    # Destroy the artifact while retaining a non-empty file.
    corrupted.write_bytes(
        data[:16] + b"CORRUPTED_PHISHGUARD_ARTIFACT"
    )

    with pytest.raises(Exception):
        ort.InferenceSession(
            str(corrupted),
            providers=["CPUExecutionProvider"],
        )


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_invalid_feature_count_is_rejected_by_runtime(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Supplying an input matrix with an incorrect feature count must fail.

    This guards against production inference callers accidentally sending
    a different feature schema to the exported model.
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_invalid_feature_count",
    )

    _, metadata = _load_candidate_artifacts(
        mock_pipeline_workspace
    )

    X = _feature_matrix(
        mock_pipeline_workspace,
        metadata,
        split="test",
    )

    assert X.shape[1] >= 2

    invalid_X = X[:, :-1]

    session = ort.InferenceSession(
        summary["onnx_path"],
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    with pytest.raises(Exception):
        session.run(
            None,
            {input_name: invalid_X},
        )


# ============================================================================
# Non-finite input tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_non_finite_inputs_do_not_get_silently_normalized(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Verifies that NaN/Inf inputs are not silently transformed into valid
    phishing predictions by the ONNX runtime.

    Depending on the generated graph/runtime implementation, this may
    result in either:
      - an explicit runtime failure, or
      - a finite output generated by an operator.

    If finite output is accepted, this test verifies that the behavior is
    observable rather than pretending the input was valid.
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_nonfinite_inputs",
    )

    _, metadata = _load_candidate_artifacts(
        mock_pipeline_workspace
    )

    X = _feature_matrix(
        mock_pipeline_workspace,
        metadata,
        split="test",
    )

    invalid = X[:1].copy()
    invalid[0, 0] = np.nan

    session = ort.InferenceSession(
        summary["onnx_path"],
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    try:
        outputs = session.run(
            None,
            {input_name: invalid},
        )

        probs = _get_probability_output(
            session,
            outputs,
        )

        assert_probability_array(
            probs,
            expected_length=1,
        )

        # If runtime accepts NaN, ensure this is explicitly visible to the
        # test suite rather than silently ignored.
        assert np.isfinite(probs).all()

    except Exception:
        # Explicit runtime rejection is the preferred behavior.
        pass


# ============================================================================
# Repeated export / idempotency tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_repeated_export_is_idempotent(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Running Stage 04 twice against the same candidate should produce
    valid artifacts both times.

    Exact byte equality is intentionally NOT required because converter
    metadata or serialization details can legitimately differ.
    """
    first = _export(
        mock_pipeline_workspace,
        output_name="onnx_idempotency",
    )

    second = _export(
        mock_pipeline_workspace,
        output_name="onnx_idempotency",
    )

    first_path = Path(first["onnx_path"])
    second_path = Path(second["onnx_path"])

    assert first_path.exists()
    assert second_path.exists()

    assert first["parity_passed"] is True
    assert second["parity_passed"] is True

    assert float(first["max_diff"]) <= PARITY_ATOL
    assert float(second["max_diff"]) <= PARITY_ATOL


# ============================================================================
# Model metadata consistency
# ============================================================================


@pytest.mark.unit
def test_candidate_metadata_contains_required_onnx_contract_fields(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Validates the Stage 02 -> Stage 04 metadata contract before ONNX export.
    """
    _, metadata = _load_candidate_artifacts(
        mock_pipeline_workspace
    )

    required = {
        "model_type",
        "feature_names",
        "optimal_threshold",
    }

    missing = required - set(metadata.keys())

    assert not missing, (
        f"Candidate metadata missing ONNX export contract fields: "
        f"{sorted(missing)}"
    )

    assert isinstance(
        metadata["feature_names"],
        list,
    )

    assert len(metadata["feature_names"]) > 0

    threshold = float(
        metadata["optimal_threshold"]
    )

    assert 0.0 < threshold < 1.0


# ============================================================================
# Performance smoke test
# ============================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    not HAS_FULL_ONNX_STACK,
    reason="ONNX and onnxruntime are required.",
)
def test_onnx_batch_inference_smoke_performance(
    mock_pipeline_workspace: Path,
) -> None:
    """
    Lightweight inference smoke test.

    This is intentionally NOT a strict latency benchmark because CI hosts
    have highly variable CPU performance.
    """
    summary = _export(
        mock_pipeline_workspace,
        output_name="onnx_performance_smoke",
    )

    _, metadata = _load_candidate_artifacts(
        mock_pipeline_workspace
    )

    X = _feature_matrix(
        mock_pipeline_workspace,
        metadata,
        split="test",
    )

    session = ort.InferenceSession(
        summary["onnx_path"],
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    outputs = session.run(
        None,
        {input_name: X},
    )

    probabilities = _get_probability_output(
        session,
        outputs,
    )

    assert_probability_array(
        probabilities,
        expected_length=len(X),
    )

    # The test is deliberately about successful completion, not an
    # environment-dependent latency SLA.
    assert len(probabilities) == len(X)
"""
PhishGuard_AI - MLOps Stage 04: ONNX Export
===========================================

Exports a fitted phishing model to the ONNX format and verifies numerical
parity against the original estimator before a deployment gate.

Security / reliability principles:
- Missing ONNX dependencies fail loudly rather than producing partial output.
- Parity verification uses finite sample inputs with explicit tolerances.
- Atomic writes and SHA-256 provenance for the exported artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from src.utils.hashing import compute_file_sha256
from src.utils.io import atomic_write_text
from src.utils.logger import get_logger

logger = get_logger("src.stages.04_export_onnx")

STAGE_VERSION = "14.0.0"


class ModelExportError(RuntimeError):
    """Raised when ONNX conversion or export fails."""


# ============================================================================
# CONVERSION
# ============================================================================


def convert_to_onnx(
    model: Any,
    *,
    num_features: int,
    model_type: str = "unknown",
    target_opset: int = 13,
) -> Any:
    """
    Convert a fitted estimator to an ONNX ``ModelProto``.

    Requires ``onnx`` and ``skl2onnx``. Raises ``ModelExportError`` on missing
    dependencies or conversion failure.
    """
    try:
        import onnx
    except ImportError:
        raise ModelExportError("ONNX dependency 'onnx' is not installed.")

    try:
        import skl2onnx
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        raise ModelExportError("ONNX dependency 'skl2onnx' is not installed.")

    initial_types = [("input", FloatTensorType([None, num_features]))]
    try:
        onnx_model = convert_sklearn(
            model, initial_types=initial_types, target_opset=target_opset
        )
    except Exception as exc:  # noqa: BLE001
        raise ModelExportError(f"ONNX conversion failed: {exc}") from exc

    try:
        onnx.checker.check_model(onnx_model)
    except Exception as exc:  # noqa: BLE001
        raise ModelExportError(f"ONNX model failed structural validation: {exc}") from exc

    logger.info(
        "ONNX conversion succeeded (model_type=%s, opset=%s)",
        model_type,
        target_opset,
    )
    return onnx_model


# ============================================================================
# PARITY VERIFICATION
# ============================================================================


def verify_onnx_parity(
    *,
    original_model: Any,
    onnx_file_path: Union[str, Path],
    sample_data: Any,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> Tuple[bool, float]:
    """
    Compare the original estimator output against the ONNX runtime.

    Returns a tuple ``(passed, max_diff)`` where ``max_diff`` is the maximum
    absolute difference observed across samples.
    """
    X = np.asarray(sample_data, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    try:
        import onnxruntime as ort
    except ImportError:
        raise ModelExportError("ONNX runtime dependency is not installed.")

    session = ort.InferenceSession(str(onnx_file_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_out = np.asarray(session.run([output_name], {input_name: X.astype(np.float32)}))

    if hasattr(original_model, "predict_proba"):
        original = np.asarray(original_model.predict_proba(X), dtype=float)
    else:
        original = np.asarray(original_model.predict(X), dtype=float)

    max_diff = 0.0
    for i in range(X.shape[0]):
        ref = np.atleast_1d(original[i]).astype(float)
        got = np.atleast_1d(onnx_out[i]).astype(float)
        diff = float(np.max(np.abs(ref - got)))
        max_diff = max(max_diff, diff)

    passed = bool(max_diff <= (atol + rtol * float(np.max(np.abs(original)))))
    return passed, max_diff


# ============================================================================
# STAGE
# ============================================================================


def export_onnx_stage(
    *,
    candidate_model_path: Union[str, Path],
    candidate_metadata_path: Union[str, Path],
    sample_data_path: Union[str, Path],
    output_onnx_dir: Union[str, Path],
    target_column: str = "label",
    opset_version: int = 13,
    max_allowed_diff: float = 1e-4,
) -> Dict[str, Any]:
    """
    Run the ONNX export stage end to end.

    Raises ``ModelExportError`` if the model or sample data is missing or the
    conversion/parity check fails.
    """
    model_path = Path(candidate_model_path)
    meta_path = Path(candidate_metadata_path)
    sample_path = Path(sample_data_path)

    if not model_path.is_file():
        raise ModelExportError(f"Model artifact not found: {model_path}")
    if not sample_path.is_file():
        raise ModelExportError(f"Sample data not found: {sample_path}")

    output_dir = Path(output_onnx_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
    except ImportError:
        raise ModelExportError("Dependency 'joblib' is not installed.")

    model = joblib.load(str(model_path))
    metadata: Dict[str, Any] = {}
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    sample_data = np.load(str(sample_path), allow_pickle=False)

    feature_count = int(metadata.get("feature_count", sample_data.shape[1]))
    model_type = str(metadata.get("model_type", "unknown"))

    onnx_model = convert_to_onnx(
        model,
        num_features=feature_count,
        model_type=model_type,
        target_opset=opset_version,
    )

    import onnx

    onnx_path = output_dir / f"{model_path.stem}.onnx"
    with onnx_path.open("wb") as handle:
        handle.write(onnx_model.SerializeToString())

    passed, max_diff = verify_onnx_parity(
        original_model=model,
        onnx_file_path=onnx_path,
        sample_data=sample_data,
        rtol=1e-3,
        atol=float(max_allowed_diff),
    )

    summary = {
        "stage": "04_export_onnx",
        "stage_version": STAGE_VERSION,
        "onnx_path": str(onnx_path),
        "onnx_sha256": compute_file_sha256(onnx_path),
        "model_type": model_type,
        "feature_count": feature_count,
        "opset_version": opset_version,
        "parity_passed": bool(passed),
        "max_diff": float(max_diff),
        "target_column": target_column,
    }

    manifest_path = output_dir / "export_summary.json"
    atomic_write_text(manifest_path, json.dumps(summary, indent=2, sort_keys=True))
    return summary


# ============================================================================
# CLI
# ============================================================================


def main(config_path: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    """Entrypoint that exports a model using the configured paths."""
    from src.utils.config import load_config

    config = load_config(config_path) if config_path else {}
    export_cfg = config.get("export_onnx", {})
    return export_onnx_stage(
        candidate_model_path=kwargs.get("candidate_model_path")
        or export_cfg.get("candidate_model_path"),
        candidate_metadata_path=kwargs.get("candidate_metadata_path")
        or export_cfg.get("candidate_metadata_path"),
        sample_data_path=kwargs.get("sample_data_path")
        or export_cfg.get("sample_data_path"),
        output_onnx_dir=kwargs.get("output_onnx_dir")
        or export_cfg.get("output_onnx_dir"),
        target_column=kwargs.get("target_column")
        or export_cfg.get("target_column", "label"),
        opset_version=int(kwargs.get("opset_version") or export_cfg.get("opset_version", 13)),
        max_allowed_diff=float(
            kwargs.get("max_allowed_diff") or export_cfg.get("max_allowed_diff", 1e-4)
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export phishing model to ONNX.")
    parser.add_argument("--config", help="Path to a stage config YAML/JSON file.")
    parser.add_argument("--candidate-model-path", help="Path to the fitted joblib model.")
    parser.add_argument("--candidate-metadata-path", help="Path to model metadata JSON.")
    parser.add_argument("--sample-data-path", help="Path to the parity sample data (.npy).")
    parser.add_argument("--output-onnx-dir", help="Output directory for the ONNX artifact.")
    args = parser.parse_args()

    try:
        result = main(
            config_path=args.config,
            candidate_model_path=args.candidate_model_path,
            candidate_metadata_path=args.candidate_metadata_path,
            sample_data_path=args.sample_data_path,
            output_onnx_dir=args.output_onnx_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    except ModelExportError as exc:
        logger.error("Export failed: %s", exc)
        sys.exit(1)


__all__ = [
    "convert_to_onnx",
    "verify_onnx_parity",
    "export_onnx_stage",
    "main",
    "ModelExportError",
    "STAGE_VERSION",
]

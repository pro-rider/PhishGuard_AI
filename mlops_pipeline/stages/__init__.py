"""
PhishGuard_AI - MLOps Pipeline Stages Package
=============================================

The stages package owns the end-to-end pipeline orchestration modules:

    01_prepare_data   -> dataset loading, validation, feature engineering
    02_train          -> candidate model training and validation
    03_evaluate       -> champion-vs-challenger evaluation and gates
    04_export_onnx    -> ONNX export and parity verification
    05_feedback_ingest-> runtime feedback ingestion

Stages are imported lazily so that package import never fails when optional
ML dependencies are absent.
"""

from __future__ import annotations

from typing import Callable, Dict

PIPELINE_VERSION = "14.0.0"

# Ordered stage metadata for orchestration and reporting.
STAGE_REGISTRY: Dict[str, Dict[str, object]] = {
    "prepare_data": {"module": "src.stages.01_prepare_data", "order": 1},
    "train": {"module": "src.stages.02_train", "order": 2},
    "evaluate": {"module": "src.stages.03_evaluate", "order": 3},
    "export_onnx": {"module": "src.stages.04_export_onnx", "order": 4},
    "feedback_ingest": {"module": "src.stages.05_feedback_ingest", "order": 5},
}


def ordered_stage_names() -> "list[str]":
    """Return stage names sorted by pipeline execution order."""
    return sorted(
        STAGE_REGISTRY, key=lambda name: int(STAGE_REGISTRY[name]["order"])
    )


def load_stage(name: str) -> Callable:
    """
    Lazily import a stage module by its registry name.

    Raises ``KeyError`` for unknown names and ``ImportError`` when the stage
    module cannot be imported.
    """
    if name not in STAGE_REGISTRY:
        raise KeyError(f"Unknown stage name: {name}")
    module_name = str(STAGE_REGISTRY[name]["module"])
    module = __import__(module_name, fromlist=["main"])
    return getattr(module, "main")


__all__ = [
    "PIPELINE_VERSION",
    "STAGE_REGISTRY",
    "ordered_stage_names",
    "load_stage",
]

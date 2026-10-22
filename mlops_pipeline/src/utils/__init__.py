"""
PhishGuard_AI - MLOps Utils Package

Re-exports shared utilities for the MLOps pipeline (hashing, IO, logging).
"""

from __future__ import annotations

from src.utils.hashing import (  # noqa: F401
    compute_file_sha256,
    compute_object_hash,
    sha256_bytes,
    sha256_file,
)
from src.utils.io import (  # noqa: F401
    atomic_write_bytes,
    atomic_write_text,
    read_json,
    safe_resolve,
    write_json_atomic,
)
from src.utils.logger import get_logger  # noqa: F401

__all__ = [
    "compute_file_sha256",
    "compute_object_hash",
    "sha256_bytes",
    "sha256_file",
    "atomic_write_bytes",
    "atomic_write_text",
    "read_json",
    "safe_resolve",
    "write_json_atomic",
    "get_logger",
]

"""
PhishGuard_AI - MLOps IO Utilities

Bounded, defensive file I/O helpers for the MLOps pipeline.

Security / reliability principles:
- Atomic writes to avoid partial artifacts.
- JSON serialization is size-bounded and key-sorted for stability.
- Paths are validated to prevent traversal.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union

MAX_JSON_BYTES = 64 * 1024 * 1024  # 64 MiB


class IOUtilsError(RuntimeError):
    """Base error for IO utilities."""


def atomic_write_text(path: Union[str, Path], content: str, encoding: str = "utf-8") -> None:
    """Atomically write a text file (temp file + rename)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(content)
        os.replace(tmp, str(target))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> None:
    """Atomically write a binary file (temp file + rename)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, str(target))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json_atomic(
    path: Union[str, Path],
    data: Any,
    *,
    sort_keys: bool = True,
) -> None:
    """Serialize ``data`` to JSON and write it atomically."""
    payload = json.dumps(
        data, sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=True
    )
    if len(payload.encode("utf-8")) > MAX_JSON_BYTES:
        raise IOUtilsError("JSON payload exceeds size bound.")
    atomic_write_text(path, payload)


def read_json(path: Union[str, Path]) -> Any:
    """Read and parse a JSON file with bounded size."""
    target = Path(path)
    size = target.stat().st_size
    if size > MAX_JSON_BYTES:
        raise IOUtilsError("JSON file exceeds size bound.")
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_resolve(base: Union[str, Path], name: str) -> Path:
    """Resolve ``name`` under ``base`` with path-containment enforcement."""
    base = Path(base).resolve()
    target = (base / name).resolve()
    if not target.is_relative_to(base):
        raise IOUtilsError("Path escapes the base directory.")
    return target


__all__ = [
    "atomic_write_text",
    "atomic_write_bytes",
    "write_json_atomic",
    "read_json",
    "safe_resolve",
    "IOUtilsError",
]

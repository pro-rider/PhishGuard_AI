"""
PhishGuard_AI - MLOps Hashing Utilities

Deterministic, bounded hashing helpers for data integrity and provenance.

Security / reliability principles:
- Streaming SHA-256 for large files to bound memory.
- ``compute_object_hash`` is deterministic for JSON-safe objects.
- All hashes are lowercase hex strings.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Union

CHUNK_SIZE = 1024 * 1024  # 1 MiB
MAX_OBJECT_HASH_BYTES = 16 * 1024 * 1024


def sha256_file(path: Union[str, Path], chunk_size: int = CHUNK_SIZE) -> str:
    """Compute the SHA-256 of a file by streaming it in bounded chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Public alias for streaming SHA-256 of a file."""
    return sha256_file(path)


def compute_object_hash(obj: Any) -> str:
    """
    Compute a deterministic SHA-256 over a JSON-safe object.

    The object is canonicalized with sorted keys to yield a stable hash.
    """
    payload = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute the SHA-256 of an in-memory byte payload."""
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "sha256_file",
    "compute_file_sha256",
    "compute_object_hash",
    "sha256_bytes",
    "CHUNK_SIZE",
]

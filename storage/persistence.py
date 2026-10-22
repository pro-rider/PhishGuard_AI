"""
PhishGuard_AI - SQLite Persistence Store

Write-through SQLite persistence for the in-memory case / quarantine managers so
records survive process restarts (the #1 production hardening priority).

Design / security notes:
- A thin adapter over the stdlib ``sqlite3`` (no external DB dependency) with a
  single per-process connection (``check_same_thread=False``) guarded by a lock.
- Whole ``to_dict()`` rows are stored as JSON blobs; the in-memory dicts remain
  the live read index and this store is the durable write-through + load source.
- Table names are validated against an allow-list to prevent injection; every
  parameter is bound.
- Record payloads may contain sensitive content: the DB file and its directory
  are intended to be gitignored and access-restricted. Production deployments
  should replace this with PostgreSQL + Alembic and real key management.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# CONSTANTS
# ============================================================================

_TABLES = ("cases", "quarantine")


class SqliteStore:
    """Minimal durable KV store keyed by record id, JSON-serialized rows."""

    def __init__(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS quarantine (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Row persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _check_table(table: str) -> str:
        if table not in _TABLES:
            raise ValueError(f"Unsupported persistence table: {table!r}")
        return table

    def upsert(self, table: str, record_id: str, data: Dict[str, Any]) -> None:
        table = self._check_table(table)
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {table} (id, data) VALUES (?, ?)",
                (record_id, json.dumps(data, ensure_ascii=False)),
            )
            self._conn.commit()

    def load_all(self, table: str) -> List[Tuple[str, Dict[str, Any]]]:
        table = self._check_table(table)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, data FROM {table}"
            ).fetchall()
        return [(r["id"], json.loads(r["data"])) for r in rows]

    # ------------------------------------------------------------------
    # Key-value metadata (e.g. persisted encryption key id / material)
    # ------------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, value),
            )
            self._conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return str(row["value"]) if row is not None else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

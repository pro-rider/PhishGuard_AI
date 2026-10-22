"""
PhishGuard_AI - Database Engine & Session Factory

Resolves the SQLAlchemy connection string and provides a thread-safe engine /
session factory. Defaults to a local SQLite file so the project runs with zero
external services; a production PostgreSQL URL is used automatically when the
``SQLALCHEMY_DATABASE_URI`` environment variable is set.

Resolution order (first match wins):
1. ``SQLALCHEMY_DATABASE_URI`` env  -> e.g. ``postgresql+psycopg2://user:pass@host/db``
2. An explicit ``path_or_dsn`` that already looks like a URL (contains ``://``).
3. ``PHISHGUARD_DB`` env (a file path) -> SQLite at that path.
4. Default SQLite file: ``<project>/storage/data/phishguard.db``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from storage.models import Base

# Anchor the project root deterministically (independent of CWD).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = str(PROJECT_ROOT / "storage" / "data" / "phishguard.db")


def resolve_dsn(path_or_dsn: Optional[str] = None) -> str:
    """Return the effective SQLAlchemy connection string."""
    env_dsn = os.environ.get("SQLALCHEMY_DATABASE_URI", "").strip()
    if env_dsn:
        return env_dsn

    if path_or_dsn and "://" in path_or_dsn:
        return path_or_dsn

    path = path_or_dsn or os.environ.get("PHISHGUARD_DB", "").strip() or DEFAULT_DB_PATH
    return f"sqlite:///{path}"


def create_app_engine(path_or_dsn: Optional[str] = None) -> Engine:
    """Create a configured engine for the resolved DSN."""
    dsn = resolve_dsn(path_or_dsn)
    if dsn.startswith("sqlite"):
        return create_engine(dsn, connect_args={"check_same_thread": False})
    # PostgreSQL / other servers: enable a bounded pool + reconnect.
    return create_engine(
        dsn,
        pool_pre_ping=True,
        pool_size=int(os.environ.get("DATABASE_POOL_SIZE", "10")),
        max_overflow=int(os.environ.get("DATABASE_MAX_OVERFLOW", "20")),
    )


_ENGINE: Optional[Engine] = None
_SESSION_FACTORY: Optional[sessionmaker] = None
_ENGINE_LOCK = threading.Lock()


def get_engine(path_or_dsn: Optional[str] = None) -> Engine:
    """Return a process-wide engine (lazily created, cached)."""
    global _ENGINE, _SESSION_FACTORY
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = create_app_engine(path_or_dsn)
            _SESSION_FACTORY = sessionmaker(
                bind=_ENGINE, autoflush=False, expire_on_commit=False
            )
    return _ENGINE


def get_session_factory(path_or_dsn: Optional[str] = None) -> sessionmaker:
    """Return the cached session factory (creating the engine if needed)."""
    get_engine(path_or_dsn)
    assert _SESSION_FACTORY is not None
    return _SESSION_FACTORY


def new_session(path_or_dsn: Optional[str] = None) -> Session:
    """Open a new session bound to the process-wide engine."""
    return get_session_factory(path_or_dsn)()


def init_db(path_or_dsn: Optional[str] = None) -> None:
    """Create all tables idempotently (safe on existing schemas)."""
    Base.metadata.create_all(get_engine(path_or_dsn))


__all__ = [
    "resolve_dsn",
    "create_app_engine",
    "get_engine",
    "get_session_factory",
    "new_session",
    "init_db",
]

"""
PhishGuard_AI - SQLAlchemy ORM Models

Durable relational schema for the persistence layer. Rows are stored as a
primary key plus a JSON ``data`` blob (mirrors the previous SQLite adapter's
layout so existing databases keep working), which lets the in-memory managers
stay the fast read index while the DB is the durable write-through + load source.

Tables:
- ``cases`` / ``quarantine``  : incident + quarantine records (JSON row).
- ``scan_jobs``               : async scan job lifecycle + result.
- ``audit_log``               : immutable, append-only audit trail.
- ``meta``                    : key/value metadata (e.g. encryption key material).

Migrate with Alembic (see ``alembic.ini`` / ``migrations/``) or let the engine
create tables idempotently at startup.
"""

from __future__ import annotations

from sqlalchemy import String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class CaseRecord(Base):
    __tablename__ = "cases"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    data: Mapped[str] = mapped_column(Text, nullable=False)


class QuarantineRecord(Base):
    __tablename__ = "quarantine"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    data: Mapped[str] = mapped_column(Text, nullable=False)


class ScanJobRecord(Base):
    __tablename__ = "scan_jobs"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    data: Mapped[str] = mapped_column(Text, nullable=False)


class AuditLogRecord(Base):
    __tablename__ = "audit_log"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    data: Mapped[str] = mapped_column(Text, nullable=False)


class MetaRecord(Base):
    __tablename__ = "meta"
    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)


# Tables managed through the generic (id, data) store interface.
DATA_TABLES = ("cases", "quarantine", "scan_jobs", "audit_log")


__all__ = [
    "Base",
    "CaseRecord",
    "QuarantineRecord",
    "ScanJobRecord",
    "AuditLogRecord",
    "MetaRecord",
    "DATA_TABLES",
]

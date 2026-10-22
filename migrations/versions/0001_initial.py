"""initial schema: cases, quarantine, scan_jobs, audit_log, meta

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-22
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cases",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("data", sa.Text(), nullable=False),
    )
    op.create_table(
        "quarantine",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("data", sa.Text(), nullable=False),
    )
    op.create_table(
        "scan_jobs",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("data", sa.Text(), nullable=False),
    )
    op.create_table(
        "audit_log",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("data", sa.Text(), nullable=False),
    )
    op.create_table(
        "meta",
        sa.Column("key", sa.String(256), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("meta")
    op.drop_table("audit_log")
    op.drop_table("scan_jobs")
    op.drop_table("quarantine")
    op.drop_table("cases")

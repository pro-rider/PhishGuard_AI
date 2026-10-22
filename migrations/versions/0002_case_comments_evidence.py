"""add case_comments + case_evidence tables

Revision ID: 0002_case_comments_evidence
Revises: 0001_initial
Create Date: 2026-10-22
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0002_case_comments_evidence"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "case_comments",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("data", sa.Text(), nullable=False),
    )
    op.create_table(
        "case_evidence",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("data", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("case_evidence")
    op.drop_table("case_comments")

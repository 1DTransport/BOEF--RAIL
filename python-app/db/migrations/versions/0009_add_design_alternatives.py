"""Add persistent design alternatives.

Revision ID: 0009_add_design_alternatives
Revises: 0008_add_project_design_defaults
Create Date: 2026-05-03 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "0009_add_design_alternatives"
down_revision = "0008_add_project_design_defaults"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "design_alternatives",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.Integer(), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column(
            "track_config_id",
            sa.Integer(),
            sa.ForeignKey("track_configs.id"),
            nullable=False,
        ),
        sa.Column("load_case_id", sa.Integer(), sa.ForeignKey("load_cases.id"), nullable=True),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=True),
        sa.Column("source_type", sa.String(length=40), nullable=False),
        sa.Column("analysis_type", sa.String(length=40), nullable=False),
        sa.Column("changed_parameters_json", sa.Text(), nullable=False),
        sa.Column("input_snapshot_json", sa.Text(), nullable=False),
        sa.Column("metrics_json", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_design_alternatives_project_id",
        "design_alternatives",
        ["project_id"],
    )
    op.create_index(
        "ix_design_alternatives_track_config_id",
        "design_alternatives",
        ["track_config_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_design_alternatives_track_config_id", table_name="design_alternatives")
    op.drop_index("ix_design_alternatives_project_id", table_name="design_alternatives")
    op.drop_table("design_alternatives")

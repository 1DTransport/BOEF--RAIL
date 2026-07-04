"""Create BOEF tables for materials and projects."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_create_boef_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rails",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("elastic_modulus_pa", sa.Float(), nullable=False),
        sa.Column("moment_inertia_m4", sa.Float(), nullable=False),
        sa.Column("section_modulus_m3", sa.Float(), nullable=False),
        sa.Column("mass_kg_per_m", sa.Float(), nullable=False),
    )
    op.create_table(
        "sleepers",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("elastic_modulus_pa", sa.Float(), nullable=False),
        sa.Column("length_m", sa.Float(), nullable=False),
        sa.Column("width_m", sa.Float(), nullable=False),
        sa.Column("height_m", sa.Float(), nullable=False),
        sa.Column("mass_kg", sa.Float(), nullable=False),
    )
    op.create_table(
        "pads",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("stiffness_newtons_per_meter", sa.Float(), nullable=False),
        sa.Column("thickness_m", sa.Float(), nullable=False),
    )
    op.create_table(
        "support_profiles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("stiffness_newtons_per_meter", sa.Float(), nullable=False),
    )
    op.create_table(
        "projects",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_table(
        "track_configs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("rail_id", sa.Integer(), nullable=False),
        sa.Column("sleeper_id", sa.Integer(), nullable=False),
        sa.Column("pad_id", sa.Integer(), nullable=False),
        sa.Column("support_profile_id", sa.Integer(), nullable=False),
        sa.Column("sleeper_spacing_m", sa.Float(), nullable=False),
        sa.Column("gauge_m", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["rail_id"], ["rails.id"]),
        sa.ForeignKeyConstraint(["sleeper_id"], ["sleepers.id"]),
        sa.ForeignKeyConstraint(["pad_id"], ["pads.id"]),
        sa.ForeignKeyConstraint(["support_profile_id"], ["support_profiles.id"]),
    )
    op.create_table(
        "load_cases",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("load_newtons", sa.Float(), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=True),
    )
    op.create_table(
        "results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("track_config_id", sa.Integer(), nullable=False),
        sa.Column("load_case_id", sa.Integer(), nullable=False),
        sa.Column("max_deflection_m", sa.Float(), nullable=False),
        sa.Column("max_moment_nm", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["track_config_id"], ["track_configs.id"]),
        sa.ForeignKeyConstraint(["load_case_id"], ["load_cases.id"]),
    )


def downgrade() -> None:
    op.drop_table("results")
    op.drop_table("load_cases")
    op.drop_table("track_configs")
    op.drop_table("projects")
    op.drop_table("support_profiles")
    op.drop_table("pads")
    op.drop_table("sleepers")
    op.drop_table("rails")

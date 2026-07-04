"""Add dynamic track parameter reference table."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_add_dynamic_track_parameters"
down_revision = "0004_add_rail_dimension_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dynamic_track_parameters",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("rail_bending_stiffness_nm2", sa.Float(), nullable=False),
        sa.Column("unsprung_wheel_mass_kg", sa.Float(), nullable=False),
        sa.Column("hertzian_contact_stiffness_n_per_m", sa.Float(), nullable=False),
        sa.Column("track_mass_single_beam_kg_per_m", sa.Float(), nullable=False),
        sa.Column("rail_mass_double_beam_kg_per_m", sa.Float(), nullable=False),
        sa.Column("sleeper_mass_double_beam_kg_per_m", sa.Float(), nullable=False),
        sa.Column("track_stiffness_single_beam_n_per_m2", sa.Float(), nullable=False),
        sa.Column("pad_stiffness_double_beam_n_per_m2", sa.Float(), nullable=False),
        sa.Column("foundation_stiffness_double_beam_n_per_m2", sa.Float(), nullable=False),
        sa.Column("track_damping_single_beam_n_s_per_m2", sa.Float(), nullable=False),
        sa.Column("pad_damping_double_beam_n_s_per_m2", sa.Float(), nullable=False),
        sa.Column("foundation_damping_double_beam_n_s_per_m2", sa.Float(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("dynamic_track_parameters")

"""Add dipped joint reference set table."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_add_dipped_joint_reference_sets"
down_revision = "0005_add_dynamic_track_parameters"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dipped_joint_reference_sets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("hertzian_contact_stiffness_n_per_m", sa.Float(), nullable=True),
        sa.Column("unsprung_mass_kg", sa.Float(), nullable=True),
        sa.Column("track_mass_p1_kg", sa.Float(), nullable=True),
        sa.Column("track_mass_p2_kg", sa.Float(), nullable=True),
        sa.Column("track_stiffness_p2_n_per_m", sa.Float(), nullable=True),
        sa.Column("track_damping_p2_n_s_per_m", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("dipped_joint_reference_sets")

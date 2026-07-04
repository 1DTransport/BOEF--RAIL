"""Rename support profile foundation modulus column."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_rename_support_profile_modulus"
down_revision = "0001_create_boef_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("support_profiles") as batch_op:
        batch_op.alter_column(
            "stiffness_newtons_per_meter",
            new_column_name="foundation_modulus_n_per_m2",
            existing_type=sa.Float(),
            existing_nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("support_profiles") as batch_op:
        batch_op.alter_column(
            "foundation_modulus_n_per_m2",
            new_column_name="stiffness_newtons_per_meter",
            existing_type=sa.Float(),
            existing_nullable=False,
        )

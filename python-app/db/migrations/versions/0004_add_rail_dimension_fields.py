"""Add additional rail dimension fields."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_add_rail_dimension_fields"
down_revision = "0003_add_rail_profiles_and_reference_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("rails") as batch_op:
        batch_op.add_column(sa.Column("head_height_mm", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("web_thickness_mm", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("rails") as batch_op:
        batch_op.drop_column("web_thickness_mm")
        batch_op.drop_column("head_height_mm")

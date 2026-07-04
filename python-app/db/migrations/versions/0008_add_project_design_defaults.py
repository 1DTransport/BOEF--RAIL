"""Add project design defaults for speed and wheel radius.

Revision ID: 0008_add_project_design_defaults
Revises: 0007_add_project_vehicle_fields
Create Date: 2025-02-14 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "0008_add_project_design_defaults"
down_revision = "0007_add_project_vehicle_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("projects", sa.Column("design_speed_kmh", sa.Float(), nullable=True))
    op.add_column("projects", sa.Column("design_wheel_radius_mm", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("projects", "design_wheel_radius_mm")
    op.drop_column("projects", "design_speed_kmh")

"""Add vehicle type/subtype fields to projects.

Revision ID: 0007_add_project_vehicle_fields
Revises: 0006_add_dipped_joint_reference_sets
Create Date: 2025-01-01 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0007_add_project_vehicle_fields"
down_revision = "0006_add_dipped_joint_reference_sets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("projects", sa.Column("vehicle_type", sa.String(length=120), nullable=True))
    op.add_column("projects", sa.Column("vehicle_subtype", sa.String(length=160), nullable=True))


def downgrade() -> None:
    op.drop_column("projects", "vehicle_subtype")
    op.drop_column("projects", "vehicle_type")

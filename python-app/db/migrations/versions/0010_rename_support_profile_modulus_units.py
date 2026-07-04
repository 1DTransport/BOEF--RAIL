"""Rename support profile display units to MN per square metre.

Revision ID: 0010_rename_support_profile_modulus_units
Revises: 0009_add_design_alternatives
Create Date: 2026-05-24 00:00:00.000000
"""

from alembic import op

revision = "0010_rename_support_profile_modulus_units"
down_revision = "0009_add_design_alternatives"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE support_profiles SET name = 'Ballast 30 MN/m²' WHERE name = 'Ballast 30 MN/m'")
    op.execute("UPDATE support_profiles SET name = 'Ballast 50 MN/m²' WHERE name = 'Ballast 50 MN/m'")
    op.execute("UPDATE support_profiles SET name = 'Ballast 80 MN/m²' WHERE name = 'Ballast 80 MN/m'")


def downgrade() -> None:
    op.execute("UPDATE support_profiles SET name = 'Ballast 30 MN/m' WHERE name = 'Ballast 30 MN/m²'")
    op.execute("UPDATE support_profiles SET name = 'Ballast 50 MN/m' WHERE name = 'Ballast 50 MN/m²'")
    op.execute("UPDATE support_profiles SET name = 'Ballast 80 MN/m' WHERE name = 'Ballast 80 MN/m²'")

"""Add rail profile geometry and reference steel tables."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_add_rail_profiles_and_reference_tables"
down_revision = "0002_rename_support_profile_modulus"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("rails") as batch_op:
        batch_op.add_column(sa.Column("height_mm", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("head_width_mm", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("foot_width_mm", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("area_cm2", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("moment_inertia_z_m4", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("section_modulus_head_m3", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("section_modulus_foot_m3", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("section_modulus_z_m3", sa.Float(), nullable=True))

    op.create_table(
        "rail_steel_properties",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("elastic_modulus_pa", sa.Float(), nullable=False),
        sa.Column("poisson_ratio", sa.Float(), nullable=False),
        sa.Column("thermal_expansion_per_c", sa.Float(), nullable=False),
        sa.Column("density_kg_per_m3", sa.Float(), nullable=False),
    )
    op.create_table(
        "rail_admissible_stress",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tensile_strength_mpa", sa.Float(), nullable=False),
        sa.Column("yield_stress_mpa", sa.Float(), nullable=False),
        sa.Column("residual_stress_mpa", sa.Float(), nullable=False),
        sa.Column("temperature_stress_mpa", sa.Float(), nullable=False),
        sa.Column("incidental_stress_mpa", sa.Float(), nullable=False),
        sa.Column("repeated_stress_mpa", sa.Float(), nullable=False),
    )
    op.create_table(
        "rail_admissible_shear_stress",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tensile_strength_mpa", sa.Float(), nullable=False),
        sa.Column("incidental_shear_mpa", sa.Float(), nullable=False),
        sa.Column("repeated_shear_mpa", sa.Float(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("rail_admissible_shear_stress")
    op.drop_table("rail_admissible_stress")
    op.drop_table("rail_steel_properties")

    with op.batch_alter_table("rails") as batch_op:
        batch_op.drop_column("section_modulus_z_m3")
        batch_op.drop_column("section_modulus_foot_m3")
        batch_op.drop_column("section_modulus_head_m3")
        batch_op.drop_column("moment_inertia_z_m4")
        batch_op.drop_column("area_cm2")
        batch_op.drop_column("foot_width_mm")
        batch_op.drop_column("head_width_mm")
        batch_op.drop_column("height_mm")

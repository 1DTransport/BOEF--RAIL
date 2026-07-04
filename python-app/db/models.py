"""SQLAlchemy models for BOEF materials and projects."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""


class Rail(Base):
    """Rail material properties (SI units)."""

    __tablename__ = "rails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    elastic_modulus_pa: Mapped[float] = mapped_column(Float, nullable=False)
    moment_inertia_m4: Mapped[float] = mapped_column(Float, nullable=False)
    section_modulus_m3: Mapped[float] = mapped_column(Float, nullable=False)
    mass_kg_per_m: Mapped[float] = mapped_column(Float, nullable=False)
    height_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    head_width_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    foot_width_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    head_height_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    web_thickness_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    area_cm2: Mapped[float | None] = mapped_column(Float, nullable=True)
    moment_inertia_z_m4: Mapped[float | None] = mapped_column(Float, nullable=True)
    section_modulus_head_m3: Mapped[float | None] = mapped_column(Float, nullable=True)
    section_modulus_foot_m3: Mapped[float | None] = mapped_column(Float, nullable=True)
    section_modulus_z_m3: Mapped[float | None] = mapped_column(Float, nullable=True)


class RailSteelProperties(Base):
    """Reference rail steel properties (SI units)."""

    __tablename__ = "rail_steel_properties"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    elastic_modulus_pa: Mapped[float] = mapped_column(Float, nullable=False)
    poisson_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    thermal_expansion_per_c: Mapped[float] = mapped_column(Float, nullable=False)
    density_kg_per_m3: Mapped[float] = mapped_column(Float, nullable=False)


class RailAdmissibleStress(Base):
    """Reference admissible dynamic stress ranges (MPa)."""

    __tablename__ = "rail_admissible_stress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tensile_strength_mpa: Mapped[float] = mapped_column(Float, nullable=False)
    yield_stress_mpa: Mapped[float] = mapped_column(Float, nullable=False)
    residual_stress_mpa: Mapped[float] = mapped_column(Float, nullable=False)
    temperature_stress_mpa: Mapped[float] = mapped_column(Float, nullable=False)
    incidental_stress_mpa: Mapped[float] = mapped_column(Float, nullable=False)
    repeated_stress_mpa: Mapped[float] = mapped_column(Float, nullable=False)


class RailAdmissibleShearStress(Base):
    """Reference admissible shear stress limits in rail head (MPa)."""

    __tablename__ = "rail_admissible_shear_stress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tensile_strength_mpa: Mapped[float] = mapped_column(Float, nullable=False)
    incidental_shear_mpa: Mapped[float] = mapped_column(Float, nullable=False)
    repeated_shear_mpa: Mapped[float] = mapped_column(Float, nullable=False)


class Sleeper(Base):
    """Sleeper material properties (SI units)."""

    __tablename__ = "sleepers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    elastic_modulus_pa: Mapped[float] = mapped_column(Float, nullable=False)
    length_m: Mapped[float] = mapped_column(Float, nullable=False)
    width_m: Mapped[float] = mapped_column(Float, nullable=False)
    height_m: Mapped[float] = mapped_column(Float, nullable=False)
    mass_kg: Mapped[float] = mapped_column(Float, nullable=False)


class Pad(Base):
    """Pad stiffness properties (SI units)."""

    __tablename__ = "pads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    stiffness_newtons_per_meter: Mapped[float] = mapped_column(Float, nullable=False)
    thickness_m: Mapped[float] = mapped_column(Float, nullable=False)


class SupportProfile(Base):
    """Support profile properties (SI units)."""

    __tablename__ = "support_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    foundation_modulus_n_per_m2: Mapped[float] = mapped_column(Float, nullable=False)

    track_configs: Mapped[list["TrackConfig"]] = relationship(
        back_populates="support_profile"
    )


class DynamicTrackParameter(Base):
    """Reference dynamic track parameters (SI units)."""

    __tablename__ = "dynamic_track_parameters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    rail_bending_stiffness_nm2: Mapped[float] = mapped_column(Float, nullable=False)
    unsprung_wheel_mass_kg: Mapped[float] = mapped_column(Float, nullable=False)
    hertzian_contact_stiffness_n_per_m: Mapped[float] = mapped_column(Float, nullable=False)
    track_mass_single_beam_kg_per_m: Mapped[float] = mapped_column(Float, nullable=False)
    rail_mass_double_beam_kg_per_m: Mapped[float] = mapped_column(Float, nullable=False)
    sleeper_mass_double_beam_kg_per_m: Mapped[float] = mapped_column(Float, nullable=False)
    track_stiffness_single_beam_n_per_m2: Mapped[float] = mapped_column(Float, nullable=False)
    pad_stiffness_double_beam_n_per_m2: Mapped[float] = mapped_column(Float, nullable=False)
    foundation_stiffness_double_beam_n_per_m2: Mapped[float] = mapped_column(Float, nullable=False)
    track_damping_single_beam_n_s_per_m2: Mapped[float] = mapped_column(Float, nullable=False)
    pad_damping_double_beam_n_s_per_m2: Mapped[float] = mapped_column(Float, nullable=False)
    foundation_damping_double_beam_n_s_per_m2: Mapped[float] = mapped_column(Float, nullable=False)


class DippedJointReferenceSet(Base):
    """Reference dipped-joint parameter sets (SI units)."""

    __tablename__ = "dipped_joint_reference_sets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    hertzian_contact_stiffness_n_per_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    unsprung_mass_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    track_mass_p1_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    track_mass_p2_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    track_stiffness_p2_n_per_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    track_damping_p2_n_s_per_m: Mapped[float | None] = mapped_column(Float, nullable=True)


class Project(Base):
    """Project metadata."""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500))
    vehicle_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    vehicle_subtype: Mapped[str | None] = mapped_column(String(160), nullable=True)
    design_speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    design_wheel_radius_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )

    track_configs: Mapped[list["TrackConfig"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    results: Mapped[list["Result"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    design_alternatives: Mapped[list["DesignAlternative"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class TrackConfig(Base):
    """Track configuration that references material selections (SI units)."""

    __tablename__ = "track_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    rail_id: Mapped[int] = mapped_column(ForeignKey("rails.id"), nullable=False)
    sleeper_id: Mapped[int] = mapped_column(ForeignKey("sleepers.id"), nullable=False)
    pad_id: Mapped[int] = mapped_column(ForeignKey("pads.id"), nullable=False)
    support_profile_id: Mapped[int] = mapped_column(
        ForeignKey("support_profiles.id"), nullable=False
    )
    sleeper_spacing_m: Mapped[float] = mapped_column(Float, nullable=False)
    gauge_m: Mapped[float] = mapped_column(Float, nullable=False)

    project: Mapped[Project] = relationship(back_populates="track_configs")
    rail: Mapped[Rail] = relationship()
    sleeper: Mapped[Sleeper] = relationship()
    pad: Mapped[Pad] = relationship()
    support_profile: Mapped[SupportProfile] = relationship(
        back_populates="track_configs"
    )
    results: Mapped[list["Result"]] = relationship(
        back_populates="track_config", cascade="all, delete-orphan"
    )
    design_alternatives: Mapped[list["DesignAlternative"]] = relationship(
        back_populates="track_config", cascade="all, delete-orphan"
    )


class LoadCase(Base):
    """Load case definition (SI units)."""

    __tablename__ = "load_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    load_newtons: Mapped[float] = mapped_column(Float, nullable=False)
    description: Mapped[str | None] = mapped_column(String(500))

    results: Mapped[list["Result"]] = relationship(
        back_populates="load_case", cascade="all, delete-orphan"
    )
    design_alternatives: Mapped[list["DesignAlternative"]] = relationship(
        back_populates="load_case"
    )


class Result(Base):
    """Analysis results (SI units)."""

    __tablename__ = "results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    track_config_id: Mapped[int] = mapped_column(
        ForeignKey("track_configs.id"), nullable=False
    )
    load_case_id: Mapped[int] = mapped_column(
        ForeignKey("load_cases.id"), nullable=False
    )
    max_deflection_m: Mapped[float] = mapped_column(Float, nullable=False)
    max_moment_nm: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )

    project: Mapped[Project] = relationship(back_populates="results")
    track_config: Mapped[TrackConfig] = relationship(back_populates="results")
    load_case: Mapped[LoadCase] = relationship(back_populates="results")


class DesignAlternative(Base):
    """Persisted design-decision snapshot for project alternatives."""

    __tablename__ = "design_alternatives"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    track_config_id: Mapped[int] = mapped_column(
        ForeignKey("track_configs.id"), nullable=False
    )
    load_case_id: Mapped[int | None] = mapped_column(
        ForeignKey("load_cases.id"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    source_type: Mapped[str] = mapped_column(String(40), nullable=False)
    analysis_type: Mapped[str] = mapped_column(String(40), nullable=False)
    changed_parameters_json: Mapped[str] = mapped_column(Text, nullable=False)
    input_snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )

    project: Mapped[Project] = relationship(back_populates="design_alternatives")
    track_config: Mapped[TrackConfig] = relationship(back_populates="design_alternatives")
    load_case: Mapped[LoadCase | None] = relationship(back_populates="design_alternatives")

"""CRUD helpers for BOEF database entities with unit validation."""

from __future__ import annotations

import json
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import (
    DesignAlternative,
    DippedJointReferenceSet,
    DynamicTrackParameter,
    LoadCase,
    Pad,
    Project,
    Rail,
    Result,
    Sleeper,
    SupportProfile,
    TrackConfig,
)

DESIGN_ALTERNATIVE_SOURCE_TYPES = {"manual", "sensitivity", "analysis"}
DESIGN_ALTERNATIVE_ANALYSIS_TYPES = {"static", "transition", "dynamic", "special"}
DESIGN_ALTERNATIVE_STATUSES = {"ok", "warning", "fail", "draft"}


def _require_non_empty(value: str, field: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field} is required")


def _require_positive(value: float, field: str, unit: str) -> None:
    if value is None or value <= 0:
        raise ValueError(f"{field} must be > 0 {unit}")


def _require_non_negative(value: float, field: str, unit: str) -> None:
    if value is None or value < 0:
        raise ValueError(f"{field} must be >= 0 {unit}")


def _require_optional_positive(value: float | None, field: str, unit: str) -> None:
    if value is None:
        return
    _require_positive(value, field, unit)


def _require_id(value: int, field: str) -> None:
    if value is None or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def validate_rail_data(
    name: str,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
    section_modulus_m3: float,
    mass_kg_per_m: float,
) -> None:
    """Validate rail data in SI units."""
    _require_non_empty(name, "name")
    _require_positive(elastic_modulus_pa, "elastic_modulus_pa", "Pa")
    _require_positive(moment_inertia_m4, "moment_inertia_m4", "m^4")
    _require_positive(section_modulus_m3, "section_modulus_m3", "m^3")
    _require_positive(mass_kg_per_m, "mass_kg_per_m", "kg/m")


def validate_optional_rail_data(
    *,
    height_mm: float | None,
    head_width_mm: float | None,
    foot_width_mm: float | None,
    head_height_mm: float | None,
    web_thickness_mm: float | None,
    area_cm2: float | None,
    moment_inertia_z_m4: float | None,
    section_modulus_head_m3: float | None,
    section_modulus_foot_m3: float | None,
    section_modulus_z_m3: float | None,
) -> None:
    """Validate optional rail geometry data."""
    _require_optional_positive(height_mm, "height_mm", "mm")
    _require_optional_positive(head_width_mm, "head_width_mm", "mm")
    _require_optional_positive(foot_width_mm, "foot_width_mm", "mm")
    _require_optional_positive(head_height_mm, "head_height_mm", "mm")
    _require_optional_positive(web_thickness_mm, "web_thickness_mm", "mm")
    _require_optional_positive(area_cm2, "area_cm2", "cm^2")
    _require_optional_positive(moment_inertia_z_m4, "moment_inertia_z_m4", "m^4")
    _require_optional_positive(section_modulus_head_m3, "section_modulus_head_m3", "m^3")
    _require_optional_positive(section_modulus_foot_m3, "section_modulus_foot_m3", "m^3")
    _require_optional_positive(section_modulus_z_m3, "section_modulus_z_m3", "m^3")


def validate_sleeper_data(
    name: str,
    elastic_modulus_pa: float,
    length_m: float,
    width_m: float,
    height_m: float,
    mass_kg: float,
) -> None:
    """Validate sleeper data in SI units."""
    _require_non_empty(name, "name")
    _require_positive(elastic_modulus_pa, "elastic_modulus_pa", "Pa")
    _require_positive(length_m, "length_m", "m")
    _require_positive(width_m, "width_m", "m")
    _require_positive(height_m, "height_m", "m")
    _require_positive(mass_kg, "mass_kg", "kg")


def validate_pad_data(
    name: str, stiffness_newtons_per_meter: float, thickness_m: float
) -> None:
    """Validate pad data in SI units."""
    _require_non_empty(name, "name")
    _require_positive(stiffness_newtons_per_meter, "stiffness_newtons_per_meter", "N/m")
    _require_positive(thickness_m, "thickness_m", "m")


def validate_support_profile_data(name: str, foundation_modulus_n_per_m2: float) -> None:
    """Validate support profile data in SI units."""
    _require_non_empty(name, "name")
    _require_positive(
        foundation_modulus_n_per_m2, "foundation_modulus_n_per_m2", "N/m²"
    )


def validate_project_data(name: str) -> None:
    """Validate project data."""
    _require_non_empty(name, "name")


def validate_track_config_data(
    name: str,
    project_id: int,
    rail_id: int,
    sleeper_id: int,
    pad_id: int,
    support_profile_id: int,
    sleeper_spacing_m: float,
    gauge_m: float,
) -> None:
    """Validate track configuration data in SI units."""
    _require_non_empty(name, "name")
    _require_id(project_id, "project_id")
    _require_id(rail_id, "rail_id")
    _require_id(sleeper_id, "sleeper_id")
    _require_id(pad_id, "pad_id")
    _require_id(support_profile_id, "support_profile_id")
    _require_positive(sleeper_spacing_m, "sleeper_spacing_m", "m")
    _require_positive(gauge_m, "gauge_m", "m")


def validate_load_case_data(name: str, load_newtons: float) -> None:
    """Validate load case data in SI units."""
    _require_non_empty(name, "name")
    _require_positive(load_newtons, "load_newtons", "N")


def validate_result_data(
    project_id: int,
    track_config_id: int,
    load_case_id: int,
    max_deflection_m: float,
    max_moment_nm: float,
) -> None:
    """Validate result data in SI units."""
    _require_id(project_id, "project_id")
    _require_id(track_config_id, "track_config_id")
    _require_id(load_case_id, "load_case_id")
    _require_non_negative(max_deflection_m, "max_deflection_m", "m")
    _require_non_negative(max_moment_nm, "max_moment_nm", "N·m")


def _json_text(value: object, field: str) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON serializable") from exc


def _validate_json_text(value: str, field: str) -> None:
    try:
        json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} must contain valid JSON") from exc


def validate_design_alternative_data(
    *,
    project_id: int,
    track_config_id: int,
    load_case_id: int | None,
    name: str,
    source_type: str,
    analysis_type: str,
    changed_parameters_json: str,
    input_snapshot_json: str,
    metrics_json: str,
    status: str,
    score: float | None,
) -> None:
    """Validate persisted design-alternative data."""
    _require_id(project_id, "project_id")
    _require_id(track_config_id, "track_config_id")
    if load_case_id is not None:
        _require_id(load_case_id, "load_case_id")
    _require_non_empty(name, "name")
    if source_type not in DESIGN_ALTERNATIVE_SOURCE_TYPES:
        raise ValueError("source_type must be manual, sensitivity, or analysis")
    if analysis_type not in DESIGN_ALTERNATIVE_ANALYSIS_TYPES:
        raise ValueError("analysis_type must be static, transition, dynamic, or special")
    if status not in DESIGN_ALTERNATIVE_STATUSES:
        raise ValueError("status must be ok, warning, fail, or draft")
    _validate_json_text(changed_parameters_json, "changed_parameters_json")
    _validate_json_text(input_snapshot_json, "input_snapshot_json")
    _validate_json_text(metrics_json, "metrics_json")
    if score is not None:
        float(score)


def create_rail(
    session: Session,
    *,
    name: str,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
    section_modulus_m3: float,
    mass_kg_per_m: float,
    height_mm: float | None = None,
    head_width_mm: float | None = None,
    foot_width_mm: float | None = None,
    head_height_mm: float | None = None,
    web_thickness_mm: float | None = None,
    area_cm2: float | None = None,
    moment_inertia_z_m4: float | None = None,
    section_modulus_head_m3: float | None = None,
    section_modulus_foot_m3: float | None = None,
    section_modulus_z_m3: float | None = None,
) -> Rail:
    validate_rail_data(
        name,
        elastic_modulus_pa,
        moment_inertia_m4,
        section_modulus_m3,
        mass_kg_per_m,
    )
    validate_optional_rail_data(
        height_mm=height_mm,
        head_width_mm=head_width_mm,
        foot_width_mm=foot_width_mm,
        head_height_mm=head_height_mm,
        web_thickness_mm=web_thickness_mm,
        area_cm2=area_cm2,
        moment_inertia_z_m4=moment_inertia_z_m4,
        section_modulus_head_m3=section_modulus_head_m3,
        section_modulus_foot_m3=section_modulus_foot_m3,
        section_modulus_z_m3=section_modulus_z_m3,
    )
    rail = Rail(
        name=name.strip(),
        elastic_modulus_pa=elastic_modulus_pa,
        moment_inertia_m4=moment_inertia_m4,
        section_modulus_m3=section_modulus_m3,
        mass_kg_per_m=mass_kg_per_m,
        height_mm=height_mm,
        head_width_mm=head_width_mm,
        foot_width_mm=foot_width_mm,
        head_height_mm=head_height_mm,
        web_thickness_mm=web_thickness_mm,
        area_cm2=area_cm2,
        moment_inertia_z_m4=moment_inertia_z_m4,
        section_modulus_head_m3=section_modulus_head_m3,
        section_modulus_foot_m3=section_modulus_foot_m3,
        section_modulus_z_m3=section_modulus_z_m3,
    )
    session.add(rail)
    session.commit()
    session.refresh(rail)
    return rail


def list_rails(session: Session) -> list[Rail]:
    return list(session.scalars(select(Rail).order_by(Rail.name)))


def update_rail(session: Session, rail: Rail, **updates: float | str | None) -> Rail:
    data = {
        "name": updates.get("name", rail.name),
        "elastic_modulus_pa": updates.get("elastic_modulus_pa", rail.elastic_modulus_pa),
        "moment_inertia_m4": updates.get("moment_inertia_m4", rail.moment_inertia_m4),
        "section_modulus_m3": updates.get(
            "section_modulus_m3", rail.section_modulus_m3
        ),
        "mass_kg_per_m": updates.get("mass_kg_per_m", rail.mass_kg_per_m),
    }
    optional_data = {
        "height_mm": updates.get("height_mm", rail.height_mm),
        "head_width_mm": updates.get("head_width_mm", rail.head_width_mm),
        "foot_width_mm": updates.get("foot_width_mm", rail.foot_width_mm),
        "head_height_mm": updates.get("head_height_mm", rail.head_height_mm),
        "web_thickness_mm": updates.get("web_thickness_mm", rail.web_thickness_mm),
        "area_cm2": updates.get("area_cm2", rail.area_cm2),
        "moment_inertia_z_m4": updates.get("moment_inertia_z_m4", rail.moment_inertia_z_m4),
        "section_modulus_head_m3": updates.get(
            "section_modulus_head_m3", rail.section_modulus_head_m3
        ),
        "section_modulus_foot_m3": updates.get(
            "section_modulus_foot_m3", rail.section_modulus_foot_m3
        ),
        "section_modulus_z_m3": updates.get("section_modulus_z_m3", rail.section_modulus_z_m3),
    }
    validate_rail_data(
        data["name"],
        float(data["elastic_modulus_pa"]),
        float(data["moment_inertia_m4"]),
        float(data["section_modulus_m3"]),
        float(data["mass_kg_per_m"]),
    )
    validate_optional_rail_data(
        height_mm=optional_data["height_mm"],
        head_width_mm=optional_data["head_width_mm"],
        foot_width_mm=optional_data["foot_width_mm"],
        head_height_mm=optional_data["head_height_mm"],
        web_thickness_mm=optional_data["web_thickness_mm"],
        area_cm2=optional_data["area_cm2"],
        moment_inertia_z_m4=optional_data["moment_inertia_z_m4"],
        section_modulus_head_m3=optional_data["section_modulus_head_m3"],
        section_modulus_foot_m3=optional_data["section_modulus_foot_m3"],
        section_modulus_z_m3=optional_data["section_modulus_z_m3"],
    )
    rail.name = str(data["name"]).strip()
    rail.elastic_modulus_pa = float(data["elastic_modulus_pa"])
    rail.moment_inertia_m4 = float(data["moment_inertia_m4"])
    rail.section_modulus_m3 = float(data["section_modulus_m3"])
    rail.mass_kg_per_m = float(data["mass_kg_per_m"])
    rail.height_mm = (
        float(optional_data["height_mm"]) if optional_data["height_mm"] is not None else None
    )
    rail.head_width_mm = (
        float(optional_data["head_width_mm"])
        if optional_data["head_width_mm"] is not None
        else None
    )
    rail.foot_width_mm = (
        float(optional_data["foot_width_mm"])
        if optional_data["foot_width_mm"] is not None
        else None
    )
    rail.head_height_mm = (
        float(optional_data["head_height_mm"])
        if optional_data["head_height_mm"] is not None
        else None
    )
    rail.web_thickness_mm = (
        float(optional_data["web_thickness_mm"])
        if optional_data["web_thickness_mm"] is not None
        else None
    )
    rail.area_cm2 = float(optional_data["area_cm2"]) if optional_data["area_cm2"] is not None else None
    rail.moment_inertia_z_m4 = (
        float(optional_data["moment_inertia_z_m4"])
        if optional_data["moment_inertia_z_m4"] is not None
        else None
    )
    rail.section_modulus_head_m3 = (
        float(optional_data["section_modulus_head_m3"])
        if optional_data["section_modulus_head_m3"] is not None
        else None
    )
    rail.section_modulus_foot_m3 = (
        float(optional_data["section_modulus_foot_m3"])
        if optional_data["section_modulus_foot_m3"] is not None
        else None
    )
    rail.section_modulus_z_m3 = (
        float(optional_data["section_modulus_z_m3"])
        if optional_data["section_modulus_z_m3"] is not None
        else None
    )
    session.commit()
    session.refresh(rail)
    return rail


def delete_rail(session: Session, rail: Rail) -> None:
    session.delete(rail)
    session.commit()


def create_sleeper(
    session: Session,
    *,
    name: str,
    elastic_modulus_pa: float,
    length_m: float,
    width_m: float,
    height_m: float,
    mass_kg: float,
) -> Sleeper:
    validate_sleeper_data(
        name, elastic_modulus_pa, length_m, width_m, height_m, mass_kg
    )
    sleeper = Sleeper(
        name=name.strip(),
        elastic_modulus_pa=elastic_modulus_pa,
        length_m=length_m,
        width_m=width_m,
        height_m=height_m,
        mass_kg=mass_kg,
    )
    session.add(sleeper)
    session.commit()
    session.refresh(sleeper)
    return sleeper


def list_sleepers(session: Session) -> list[Sleeper]:
    return list(session.scalars(select(Sleeper).order_by(Sleeper.name)))


def update_sleeper(session: Session, sleeper: Sleeper, **updates: float | str) -> Sleeper:
    data = {
        "name": updates.get("name", sleeper.name),
        "elastic_modulus_pa": updates.get(
            "elastic_modulus_pa", sleeper.elastic_modulus_pa
        ),
        "length_m": updates.get("length_m", sleeper.length_m),
        "width_m": updates.get("width_m", sleeper.width_m),
        "height_m": updates.get("height_m", sleeper.height_m),
        "mass_kg": updates.get("mass_kg", sleeper.mass_kg),
    }
    validate_sleeper_data(
        data["name"],
        float(data["elastic_modulus_pa"]),
        float(data["length_m"]),
        float(data["width_m"]),
        float(data["height_m"]),
        float(data["mass_kg"]),
    )
    sleeper.name = str(data["name"]).strip()
    sleeper.elastic_modulus_pa = float(data["elastic_modulus_pa"])
    sleeper.length_m = float(data["length_m"])
    sleeper.width_m = float(data["width_m"])
    sleeper.height_m = float(data["height_m"])
    sleeper.mass_kg = float(data["mass_kg"])
    session.commit()
    session.refresh(sleeper)
    return sleeper


def delete_sleeper(session: Session, sleeper: Sleeper) -> None:
    session.delete(sleeper)
    session.commit()


def create_pad(
    session: Session,
    *,
    name: str,
    stiffness_newtons_per_meter: float,
    thickness_m: float,
) -> Pad:
    validate_pad_data(name, stiffness_newtons_per_meter, thickness_m)
    pad = Pad(
        name=name.strip(),
        stiffness_newtons_per_meter=stiffness_newtons_per_meter,
        thickness_m=thickness_m,
    )
    session.add(pad)
    session.commit()
    session.refresh(pad)
    return pad


def list_pads(session: Session) -> list[Pad]:
    return list(session.scalars(select(Pad).order_by(Pad.name)))


def update_pad(session: Session, pad: Pad, **updates: float | str) -> Pad:
    data = {
        "name": updates.get("name", pad.name),
        "stiffness_newtons_per_meter": updates.get(
            "stiffness_newtons_per_meter", pad.stiffness_newtons_per_meter
        ),
        "thickness_m": updates.get("thickness_m", pad.thickness_m),
    }
    validate_pad_data(
        str(data["name"]),
        float(data["stiffness_newtons_per_meter"]),
        float(data["thickness_m"]),
    )
    pad.name = str(data["name"]).strip()
    pad.stiffness_newtons_per_meter = float(data["stiffness_newtons_per_meter"])
    pad.thickness_m = float(data["thickness_m"])
    session.commit()
    session.refresh(pad)
    return pad


def delete_pad(session: Session, pad: Pad) -> None:
    session.delete(pad)
    session.commit()


def create_support_profile(
    session: Session, *, name: str, foundation_modulus_n_per_m2: float
) -> SupportProfile:
    validate_support_profile_data(name, foundation_modulus_n_per_m2)
    profile = SupportProfile(
        name=name.strip(), foundation_modulus_n_per_m2=foundation_modulus_n_per_m2
    )
    session.add(profile)
    session.commit()
    session.refresh(profile)
    return profile


def list_support_profiles(session: Session) -> list[SupportProfile]:
    return list(session.scalars(select(SupportProfile).order_by(SupportProfile.name)))


def list_dynamic_track_parameters(session: Session) -> list[DynamicTrackParameter]:
    return list(
        session.scalars(
            select(DynamicTrackParameter).order_by(DynamicTrackParameter.name)
        )
    )


def list_dipped_joint_reference_sets(session: Session) -> list[DippedJointReferenceSet]:
    return list(
        session.scalars(
            select(DippedJointReferenceSet).order_by(DippedJointReferenceSet.name)
        )
    )


def update_support_profile(
    session: Session, profile: SupportProfile, **updates: float | str
) -> SupportProfile:
    data = {
        "name": updates.get("name", profile.name),
        "foundation_modulus_n_per_m2": updates.get(
            "foundation_modulus_n_per_m2", profile.foundation_modulus_n_per_m2
        ),
    }
    validate_support_profile_data(
        str(data["name"]), float(data["foundation_modulus_n_per_m2"])
    )
    profile.name = str(data["name"]).strip()
    profile.foundation_modulus_n_per_m2 = float(data["foundation_modulus_n_per_m2"])
    session.commit()
    session.refresh(profile)
    return profile


def delete_support_profile(session: Session, profile: SupportProfile) -> None:
    session.delete(profile)
    session.commit()


def create_project(
    session: Session,
    *,
    name: str,
    description: str | None = None,
    vehicle_type: str | None = None,
    vehicle_subtype: str | None = None,
    design_speed_kmh: float | None = None,
    design_wheel_radius_mm: float | None = None,
) -> Project:
    validate_project_data(name)
    existing = session.scalar(select(Project).where(Project.name == name.strip()))
    if existing is not None:
        raise ValueError(f"Project '{name}' already exists")
    if design_speed_kmh is not None and design_speed_kmh < 0:
        raise ValueError("design_speed_kmh must be >= 0")
    if design_wheel_radius_mm is not None and design_wheel_radius_mm < 0:
        raise ValueError("design_wheel_radius_mm must be >= 0")
    project = Project(
        name=name.strip(),
        description=description,
        vehicle_type=vehicle_type.strip() if vehicle_type else None,
        vehicle_subtype=vehicle_subtype.strip() if vehicle_subtype else None,
        design_speed_kmh=float(design_speed_kmh) if design_speed_kmh is not None else None,
        design_wheel_radius_mm=float(design_wheel_radius_mm)
        if design_wheel_radius_mm is not None
        else None,
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


def list_projects(session: Session) -> list[Project]:
    return list(session.scalars(select(Project).order_by(Project.name)))


def update_project(
    session: Session, project: Project, **updates: str | float | None
) -> Project:
    name = updates.get("name", project.name)
    validate_project_data(str(name))
    existing = session.scalar(
        select(Project).where(Project.name == str(name).strip(), Project.id != project.id)
    )
    if existing is not None:
        raise ValueError(f"Project '{name}' already exists")
    project.name = str(name).strip()
    if "description" in updates:
        project.description = updates.get("description")
    if "vehicle_type" in updates:
        value = updates.get("vehicle_type")
        project.vehicle_type = value.strip() if value else None
    if "vehicle_subtype" in updates:
        value = updates.get("vehicle_subtype")
        project.vehicle_subtype = value.strip() if value else None
    if "design_speed_kmh" in updates:
        value = updates.get("design_speed_kmh")
        if value is None:
            project.design_speed_kmh = None
        else:
            speed = float(value)
            if speed < 0:
                raise ValueError("design_speed_kmh must be >= 0")
            project.design_speed_kmh = speed
    if "design_wheel_radius_mm" in updates:
        value = updates.get("design_wheel_radius_mm")
        if value is None:
            project.design_wheel_radius_mm = None
        else:
            radius = float(value)
            if radius < 0:
                raise ValueError("design_wheel_radius_mm must be >= 0")
            project.design_wheel_radius_mm = radius
    session.commit()
    session.refresh(project)
    return project


def delete_project(session: Session, project: Project) -> None:
    session.delete(project)
    session.commit()


def create_track_config(
    session: Session,
    *,
    name: str,
    project_id: int,
    rail_id: int,
    sleeper_id: int,
    pad_id: int,
    support_profile_id: int,
    sleeper_spacing_m: float,
    gauge_m: float,
) -> TrackConfig:
    validate_track_config_data(
        name,
        project_id,
        rail_id,
        sleeper_id,
        pad_id,
        support_profile_id,
        sleeper_spacing_m,
        gauge_m,
    )
    config = TrackConfig(
        name=name.strip(),
        project_id=project_id,
        rail_id=rail_id,
        sleeper_id=sleeper_id,
        pad_id=pad_id,
        support_profile_id=support_profile_id,
        sleeper_spacing_m=sleeper_spacing_m,
        gauge_m=gauge_m,
    )
    session.add(config)
    session.commit()
    session.refresh(config)
    return config


def list_track_configs(session: Session) -> list[TrackConfig]:
    return list(session.scalars(select(TrackConfig).order_by(TrackConfig.name)))


def update_track_config(
    session: Session, config: TrackConfig, **updates: float | str | int
) -> TrackConfig:
    data = {
        "name": updates.get("name", config.name),
        "project_id": updates.get("project_id", config.project_id),
        "rail_id": updates.get("rail_id", config.rail_id),
        "sleeper_id": updates.get("sleeper_id", config.sleeper_id),
        "pad_id": updates.get("pad_id", config.pad_id),
        "support_profile_id": updates.get(
            "support_profile_id", config.support_profile_id
        ),
        "sleeper_spacing_m": updates.get(
            "sleeper_spacing_m", config.sleeper_spacing_m
        ),
        "gauge_m": updates.get("gauge_m", config.gauge_m),
    }
    validate_track_config_data(
        str(data["name"]),
        int(data["project_id"]),
        int(data["rail_id"]),
        int(data["sleeper_id"]),
        int(data["pad_id"]),
        int(data["support_profile_id"]),
        float(data["sleeper_spacing_m"]),
        float(data["gauge_m"]),
    )
    config.name = str(data["name"]).strip()
    config.project_id = int(data["project_id"])
    config.rail_id = int(data["rail_id"])
    config.sleeper_id = int(data["sleeper_id"])
    config.pad_id = int(data["pad_id"])
    config.support_profile_id = int(data["support_profile_id"])
    config.sleeper_spacing_m = float(data["sleeper_spacing_m"])
    config.gauge_m = float(data["gauge_m"])
    session.commit()
    session.refresh(config)
    return config


def delete_track_config(session: Session, config: TrackConfig) -> None:
    session.delete(config)
    session.commit()


def create_load_case(
    session: Session,
    *,
    name: str,
    load_newtons: float,
    description: str | None = None,
) -> LoadCase:
    validate_load_case_data(name, load_newtons)
    load_case = LoadCase(
        name=name.strip(), load_newtons=load_newtons, description=description
    )
    session.add(load_case)
    session.commit()
    session.refresh(load_case)
    return load_case


def list_load_cases(session: Session) -> list[LoadCase]:
    return list(session.scalars(select(LoadCase).order_by(LoadCase.name)))


def update_load_case(
    session: Session, load_case: LoadCase, **updates: float | str | None
) -> LoadCase:
    data = {
        "name": updates.get("name", load_case.name),
        "load_newtons": updates.get("load_newtons", load_case.load_newtons),
        "description": updates.get("description", load_case.description),
    }
    validate_load_case_data(str(data["name"]), float(data["load_newtons"]))
    load_case.name = str(data["name"]).strip()
    load_case.load_newtons = float(data["load_newtons"])
    load_case.description = data["description"]
    session.commit()
    session.refresh(load_case)
    return load_case


def delete_load_case(session: Session, load_case: LoadCase) -> None:
    session.delete(load_case)
    session.commit()


def create_result(
    session: Session,
    *,
    project_id: int,
    track_config_id: int,
    load_case_id: int,
    max_deflection_m: float,
    max_moment_nm: float,
) -> Result:
    validate_result_data(
        project_id, track_config_id, load_case_id, max_deflection_m, max_moment_nm
    )
    result = Result(
        project_id=project_id,
        track_config_id=track_config_id,
        load_case_id=load_case_id,
        max_deflection_m=max_deflection_m,
        max_moment_nm=max_moment_nm,
    )
    session.add(result)
    session.commit()
    session.refresh(result)
    return result


def list_results(session: Session) -> list[Result]:
    return list(session.scalars(select(Result).order_by(Result.created_at.desc())))


def update_result(session: Session, result: Result, **updates: float | int) -> Result:
    data = {
        "project_id": updates.get("project_id", result.project_id),
        "track_config_id": updates.get("track_config_id", result.track_config_id),
        "load_case_id": updates.get("load_case_id", result.load_case_id),
        "max_deflection_m": updates.get("max_deflection_m", result.max_deflection_m),
        "max_moment_nm": updates.get("max_moment_nm", result.max_moment_nm),
    }
    validate_result_data(
        int(data["project_id"]),
        int(data["track_config_id"]),
        int(data["load_case_id"]),
        float(data["max_deflection_m"]),
        float(data["max_moment_nm"]),
    )
    result.project_id = int(data["project_id"])
    result.track_config_id = int(data["track_config_id"])
    result.load_case_id = int(data["load_case_id"])
    result.max_deflection_m = float(data["max_deflection_m"])
    result.max_moment_nm = float(data["max_moment_nm"])
    session.commit()
    session.refresh(result)
    return result


def delete_result(session: Session, result: Result) -> None:
    session.delete(result)
    session.commit()


def create_design_alternative(
    session: Session,
    *,
    project_id: int,
    track_config_id: int,
    load_case_id: int | None = None,
    name: str,
    description: str | None = None,
    source_type: str,
    analysis_type: str,
    changed_parameters: object,
    input_snapshot: object,
    metrics: object,
    status: str,
    score: float | None = None,
) -> DesignAlternative:
    """Create a design alternative snapshot."""
    changed_parameters_json = _json_text(changed_parameters, "changed_parameters_json")
    input_snapshot_json = _json_text(input_snapshot, "input_snapshot_json")
    metrics_json = _json_text(metrics, "metrics_json")
    validate_design_alternative_data(
        project_id=project_id,
        track_config_id=track_config_id,
        load_case_id=load_case_id,
        name=name,
        source_type=source_type,
        analysis_type=analysis_type,
        changed_parameters_json=changed_parameters_json,
        input_snapshot_json=input_snapshot_json,
        metrics_json=metrics_json,
        status=status,
        score=score,
    )
    alternative = DesignAlternative(
        project_id=project_id,
        track_config_id=track_config_id,
        load_case_id=load_case_id,
        name=name.strip(),
        description=description,
        source_type=source_type,
        analysis_type=analysis_type,
        changed_parameters_json=changed_parameters_json,
        input_snapshot_json=input_snapshot_json,
        metrics_json=metrics_json,
        status=status,
        score=float(score) if score is not None else None,
    )
    session.add(alternative)
    session.commit()
    session.refresh(alternative)
    return alternative


def list_design_alternatives(
    session: Session,
    *,
    project_id: int | None = None,
) -> list[DesignAlternative]:
    statement = select(DesignAlternative).order_by(DesignAlternative.created_at.desc())
    if project_id is not None:
        statement = statement.where(DesignAlternative.project_id == project_id)
    return list(session.scalars(statement))


def update_design_alternative(
    session: Session,
    alternative: DesignAlternative,
    **updates: object,
) -> DesignAlternative:
    data = {
        "project_id": updates.get("project_id", alternative.project_id),
        "track_config_id": updates.get("track_config_id", alternative.track_config_id),
        "load_case_id": updates.get("load_case_id", alternative.load_case_id),
        "name": updates.get("name", alternative.name),
        "description": updates.get("description", alternative.description),
        "source_type": updates.get("source_type", alternative.source_type),
        "analysis_type": updates.get("analysis_type", alternative.analysis_type),
        "changed_parameters_json": (
            _json_text(updates["changed_parameters"], "changed_parameters_json")
            if "changed_parameters" in updates
            else alternative.changed_parameters_json
        ),
        "input_snapshot_json": (
            _json_text(updates["input_snapshot"], "input_snapshot_json")
            if "input_snapshot" in updates
            else alternative.input_snapshot_json
        ),
        "metrics_json": (
            _json_text(updates["metrics"], "metrics_json")
            if "metrics" in updates
            else alternative.metrics_json
        ),
        "status": updates.get("status", alternative.status),
        "score": updates.get("score", alternative.score),
    }
    validate_design_alternative_data(
        project_id=int(data["project_id"]),
        track_config_id=int(data["track_config_id"]),
        load_case_id=int(data["load_case_id"]) if data["load_case_id"] is not None else None,
        name=str(data["name"]),
        source_type=str(data["source_type"]),
        analysis_type=str(data["analysis_type"]),
        changed_parameters_json=str(data["changed_parameters_json"]),
        input_snapshot_json=str(data["input_snapshot_json"]),
        metrics_json=str(data["metrics_json"]),
        status=str(data["status"]),
        score=float(data["score"]) if data["score"] is not None else None,
    )
    alternative.project_id = int(data["project_id"])
    alternative.track_config_id = int(data["track_config_id"])
    alternative.load_case_id = int(data["load_case_id"]) if data["load_case_id"] is not None else None
    alternative.name = str(data["name"]).strip()
    alternative.description = data["description"] if data["description"] is None else str(data["description"])
    alternative.source_type = str(data["source_type"])
    alternative.analysis_type = str(data["analysis_type"])
    alternative.changed_parameters_json = str(data["changed_parameters_json"])
    alternative.input_snapshot_json = str(data["input_snapshot_json"])
    alternative.metrics_json = str(data["metrics_json"])
    alternative.status = str(data["status"])
    alternative.score = float(data["score"]) if data["score"] is not None else None
    session.commit()
    session.refresh(alternative)
    return alternative


def delete_design_alternative(session: Session, alternative: DesignAlternative) -> None:
    session.delete(alternative)
    session.commit()


def upsert_named(
    session: Session, model: type[Rail | Sleeper | Pad | SupportProfile | LoadCase], items: Iterable[dict]
) -> None:
    """Insert items by name if they do not already exist."""
    existing = {
        row.name for row in session.scalars(select(model).where(model.name.is_not(None)))
    }
    for item in items:
        if item["name"] in existing:
            continue
        session.add(model(**item))
    session.commit()

"""Seed data for BOEF database."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from db import crud
from db.models import (
    DippedJointReferenceSet,
    DynamicTrackParameter,
    LoadCase,
    Pad,
    Project,
    Rail,
    RailAdmissibleShearStress,
    RailAdmissibleStress,
    RailSteelProperties,
    Result,
    Sleeper,
    SupportProfile,
    TrackConfig,
)


def _exists_by_name(
    session: Session,
    model: type[
        Rail
        | DippedJointReferenceSet
        | DynamicTrackParameter
        | Sleeper
        | SupportProfile
        | LoadCase
        | Pad
        | Project
        | RailSteelProperties
    ],
    name: str,
) -> bool:
    return session.scalar(select(model).where(model.name == name)) is not None


def _exists_by_tensile_strength(
    session: Session,
    model: type[RailAdmissibleStress | RailAdmissibleShearStress],
    tensile_strength_mpa: float,
) -> bool:
    return (
        session.scalar(
            select(model).where(model.tensile_strength_mpa == tensile_strength_mpa)
        )
        is not None
    )


def _rail_dimensions(
    *,
    name: str,
    height_mm: float,
    foot_width_mm: float | None,
    head_width_mm: float | None,
    head_height_mm: float | None,
    web_thickness_mm: float | None,
    area_cm2: float | None,
    mass_kg_per_m: float,
    elastic_modulus_pa: float | None = None,
    moment_inertia_cm4: tuple[float, float] | None = None,
    section_modulus_z_cm3: float | None = None,
    section_modulus_head_cm3: float | None = None,
    section_modulus_foot_cm3: float | None = None,
) -> dict:
    return {
        "name": name,
        "height_mm": height_mm,
        "foot_width_mm": foot_width_mm,
        "head_width_mm": head_width_mm,
        "head_height_mm": head_height_mm,
        "web_thickness_mm": web_thickness_mm,
        "area_cm2": area_cm2,
        "mass_kg_per_m": mass_kg_per_m,
        "elastic_modulus_pa": elastic_modulus_pa,
        "moment_inertia_cm4": moment_inertia_cm4,
        "section_modulus_z_cm3": section_modulus_z_cm3,
        "section_modulus_head_cm3": section_modulus_head_cm3,
        "section_modulus_foot_cm3": section_modulus_foot_cm3,
    }


def _modern_rail_profile(
    *,
    name: str,
    height_mm: float,
    head_width_mm: float,
    foot_width_mm: float,
    area_cm2: float,
    mass_kg_per_m: float,
    iy_cm4: float,
    iz_cm4: float,
    wyh_cm3: float,
    wyf_cm3: float,
    wz_cm3: float,
    elastic_modulus_pa: float = 2.10e11,
    head_height_mm: float | None = None,
    web_thickness_mm: float | None = None,
) -> dict:
    section_modulus_m3 = min(wyh_cm3, wyf_cm3) * 1.0e-6
    spec = {
        "name": name,
        "elastic_modulus_pa": elastic_modulus_pa,
        "moment_inertia_m4": iy_cm4 * 1.0e-8,
        "moment_inertia_z_m4": iz_cm4 * 1.0e-8,
        "section_modulus_m3": section_modulus_m3,
        "section_modulus_head_m3": wyh_cm3 * 1.0e-6,
        "section_modulus_foot_m3": wyf_cm3 * 1.0e-6,
        "section_modulus_z_m3": wz_cm3 * 1.0e-6,
        "mass_kg_per_m": mass_kg_per_m,
        "height_mm": height_mm,
        "head_width_mm": head_width_mm,
        "foot_width_mm": foot_width_mm,
        "area_cm2": area_cm2,
    }
    if head_height_mm is not None:
        spec["head_height_mm"] = head_height_mm
    if web_thickness_mm is not None:
        spec["web_thickness_mm"] = web_thickness_mm
    return spec


def _approximate_section_properties(area_cm2: float, height_mm: float) -> tuple[float, float]:
    """Approximate Iy and W using a rectangular section with the same area and height."""
    area_m2 = area_cm2 * 1.0e-4
    height_m = height_mm / 1000.0
    inertia_m4 = area_m2 * height_m**2 / 12.0
    section_modulus_m3 = inertia_m4 / (height_m / 2.0)
    return inertia_m4, section_modulus_m3


def _upsert_rail(session: Session, spec: dict) -> None:
    rail = session.scalar(select(Rail).where(Rail.name == spec["name"]))
    if rail is None:
        inertia_m4, section_modulus_m3, optional_create, _ = _resolve_section_properties(spec)
        crud.create_rail(
            session,
            name=spec["name"],
            elastic_modulus_pa=spec.get("elastic_modulus_pa") or 2.1e11,
            moment_inertia_m4=inertia_m4,
            section_modulus_m3=section_modulus_m3,
            mass_kg_per_m=spec["mass_kg_per_m"],
            height_mm=spec["height_mm"],
            head_width_mm=spec["head_width_mm"],
            foot_width_mm=spec["foot_width_mm"],
            head_height_mm=spec["head_height_mm"],
            web_thickness_mm=spec["web_thickness_mm"],
            area_cm2=spec["area_cm2"],
            **optional_create,
        )
        return
    inertia_m4, section_modulus_m3, _, optional_update = _resolve_section_properties(spec)
    crud.update_rail(
        session,
        rail,
        moment_inertia_m4=inertia_m4,
        section_modulus_m3=section_modulus_m3,
        height_mm=spec["height_mm"],
        head_width_mm=spec["head_width_mm"],
        foot_width_mm=spec["foot_width_mm"],
        head_height_mm=spec["head_height_mm"],
        web_thickness_mm=spec["web_thickness_mm"],
        area_cm2=spec["area_cm2"],
        mass_kg_per_m=spec["mass_kg_per_m"],
        **optional_update,
    )
    if spec.get("elastic_modulus_pa") is not None:
        crud.update_rail(
            session,
            rail,
            elastic_modulus_pa=spec["elastic_modulus_pa"],
        )


def _upsert_rail_exact(session: Session, spec: dict) -> None:
    rail = session.scalar(select(Rail).where(Rail.name == spec["name"]))
    optional_fields = (
        "height_mm",
        "head_width_mm",
        "foot_width_mm",
        "head_height_mm",
        "web_thickness_mm",
        "area_cm2",
        "moment_inertia_z_m4",
        "section_modulus_head_m3",
        "section_modulus_foot_m3",
        "section_modulus_z_m3",
    )
    optional_data = {key: spec[key] for key in optional_fields if key in spec}
    if rail is None:
        crud.create_rail(
            session,
            name=spec["name"],
            elastic_modulus_pa=spec["elastic_modulus_pa"],
            moment_inertia_m4=spec["moment_inertia_m4"],
            section_modulus_m3=spec["section_modulus_m3"],
            mass_kg_per_m=spec["mass_kg_per_m"],
            **optional_data,
        )
        return
    crud.update_rail(
        session,
        rail,
        elastic_modulus_pa=spec["elastic_modulus_pa"],
        moment_inertia_m4=spec["moment_inertia_m4"],
        section_modulus_m3=spec["section_modulus_m3"],
        mass_kg_per_m=spec["mass_kg_per_m"],
        **optional_data,
    )


def _resolve_section_properties(spec: dict) -> tuple[float, float, dict, dict]:
    inertia_cm4 = spec.get("moment_inertia_cm4")
    section_modulus_head_cm3 = spec.get("section_modulus_head_cm3")
    section_modulus_foot_cm3 = spec.get("section_modulus_foot_cm3")
    section_modulus_z_cm3 = spec.get("section_modulus_z_cm3")
    optional_create: dict[str, float] = {}
    optional_update: dict[str, float] = {}
    approx_inertia_m4 = None
    approx_section_modulus_m3 = None
    if inertia_cm4:
        inertia_low = min(inertia_cm4)
        inertia_high = max(inertia_cm4)
        inertia_m4 = inertia_low * 1.0e-8
        optional_create["moment_inertia_z_m4"] = inertia_high * 1.0e-8
        optional_update["moment_inertia_z_m4"] = inertia_high * 1.0e-8
    else:
        approx_inertia_m4, approx_section_modulus_m3 = _approximate_section_properties(
            spec["area_cm2"],
            spec["height_mm"],
        )
        inertia_m4 = approx_inertia_m4
    if section_modulus_head_cm3 is not None and section_modulus_foot_cm3 is not None:
        section_modulus_low = min(section_modulus_head_cm3, section_modulus_foot_cm3)
        section_modulus_high = max(section_modulus_head_cm3, section_modulus_foot_cm3)
        section_modulus_m3 = section_modulus_low * 1.0e-6
        optional_create["section_modulus_head_m3"] = section_modulus_head_cm3 * 1.0e-6
        optional_create["section_modulus_foot_m3"] = section_modulus_foot_cm3 * 1.0e-6
        optional_create["section_modulus_z_m3"] = section_modulus_high * 1.0e-6
        optional_update["section_modulus_head_m3"] = section_modulus_head_cm3 * 1.0e-6
        optional_update["section_modulus_foot_m3"] = section_modulus_foot_cm3 * 1.0e-6
        optional_update["section_modulus_z_m3"] = section_modulus_high * 1.0e-6
    elif section_modulus_z_cm3 is not None:
        section_modulus_m3 = section_modulus_z_cm3 * 1.0e-6
        optional_create["section_modulus_z_m3"] = section_modulus_m3
        optional_update["section_modulus_z_m3"] = section_modulus_m3
    else:
        if approx_section_modulus_m3 is None:
            _, approx_section_modulus_m3 = _approximate_section_properties(
                spec["area_cm2"],
                spec["height_mm"],
            )
        section_modulus_m3 = approx_section_modulus_m3
        optional_create["section_modulus_head_m3"] = section_modulus_m3
        optional_create["section_modulus_foot_m3"] = section_modulus_m3
    return inertia_m4, section_modulus_m3, optional_create, optional_update

def seed_database(session: Session) -> None:
    """Seed baseline materials and load cases."""
    dipped_joint_reference = [
        DippedJointReferenceSet(
            name="Modern Railway Track – Table 6.1 (baseline)",
            hertzian_contact_stiffness_n_per_m=1.4e9,
            unsprung_mass_kg=350.0,
            track_mass_p1_kg=None,
            track_mass_p2_kg=None,
            track_stiffness_p2_n_per_m=None,
            track_damping_p2_n_s_per_m=None,
        )
    ]
    dynamic_reference = [
        DynamicTrackParameter(
            name="Table 6.1 reference",
            rail_bending_stiffness_nm2=4.5e6,
            unsprung_wheel_mass_kg=350.0,
            hertzian_contact_stiffness_n_per_m=1.4e9,
            track_mass_single_beam_kg_per_m=119.0,
            rail_mass_double_beam_kg_per_m=54.43,
            sleeper_mass_double_beam_kg_per_m=157.0,
            track_stiffness_single_beam_n_per_m2=4.0e7,
            pad_stiffness_double_beam_n_per_m2=2.5e8,
            foundation_stiffness_double_beam_n_per_m2=4.0e7,
            track_damping_single_beam_n_s_per_m2=1.2e5,
            pad_damping_double_beam_n_s_per_m2=9.1e4,
            foundation_damping_double_beam_n_s_per_m2=1.2e5,
        )
    ]
    for reference in dynamic_reference:
        if not _exists_by_name(session, DynamicTrackParameter, reference.name):
            session.add(reference)
    for reference in dipped_joint_reference:
        if not _exists_by_name(session, DippedJointReferenceSet, reference.name):
            session.add(reference)
    session.commit()

    rails = [
        _rail_dimensions(
            name="AS60",
            height_mm=170.0,
            foot_width_mm=146.0,
            head_width_mm=70.0,
            head_height_mm=49.0,
            web_thickness_mm=16.5,
            area_cm2=77.25,
            mass_kg_per_m=60.60,
        ),
        _rail_dimensions(
            name="AS68",
            height_mm=185.70,
            foot_width_mm=152.40,
            head_width_mm=74.60,
            head_height_mm=49.20,
            web_thickness_mm=17.50,
            area_cm2=86.02,
            mass_kg_per_m=67.50,
        ),
        _rail_dimensions(
            name="39E1 (BS 80A)",
            height_mm=133.35,
            foot_width_mm=117.47,
            head_width_mm=63.50,
            head_height_mm=42.47,
            web_thickness_mm=13.10,
            area_cm2=50.66,
            mass_kg_per_m=39.77,
        ),
        _rail_dimensions(
            name="45E1 (BS 90A)",
            height_mm=142.88,
            foot_width_mm=127.00,
            head_width_mm=66.67,
            head_height_mm=46.04,
            web_thickness_mm=13.89,
            area_cm2=57.46,
            mass_kg_per_m=45.11,
        ),
        _rail_dimensions(
            name="45E3 (RN 45)",
            height_mm=142.00,
            foot_width_mm=130.00,
            head_width_mm=66.00,
            head_height_mm=40.50,
            web_thickness_mm=15.00,
            area_cm2=57.05,
            mass_kg_per_m=44.79,
        ),
        _rail_dimensions(
            name="46E1",
            height_mm=145.00,
            foot_width_mm=125.00,
            head_width_mm=65.00,
            head_height_mm=45.00,
            web_thickness_mm=14.00,
            area_cm2=58.82,
            mass_kg_per_m=46.17,
        ),
        _rail_dimensions(
            name="46E2 (U33)",
            height_mm=149.00,
            foot_width_mm=125.00,
            head_width_mm=62.00,
            head_height_mm=47.00,
            web_thickness_mm=15.00,
            area_cm2=58.94,
            mass_kg_per_m=46.66,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(1605.9, 307.5),
            section_modulus_head_cm3=224.2,
            section_modulus_foot_cm3=228.2,
            section_modulus_z_cm3=228.2,
        ),
        _rail_dimensions(
            name="46E3",
            height_mm=149.00,
            foot_width_mm=125.00,
            head_width_mm=73.72,
            head_height_mm=14.18,
            web_thickness_mm=6.81,
            area_cm2=59.78,
            mass_kg_per_m=46.90,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(1688.0, 338.6),
            section_modulus_head_cm3=221.6,
            section_modulus_foot_cm3=245.2,
            section_modulus_z_cm3=245.2,
        ),
        _rail_dimensions(
            name="46E4",
            height_mm=149.00,
            foot_width_mm=125.00,
            head_width_mm=65.00,
            head_height_mm=13.75,
            web_thickness_mm=5.48,
            area_cm2=62.92,
            mass_kg_per_m=49.39,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(1816.0, 319.1),
            section_modulus_head_cm3=240.3,
            section_modulus_foot_cm3=247.5,
            section_modulus_z_cm3=247.5,
        ),
        _rail_dimensions(
            name="MAV48",
            height_mm=148.00,
            foot_width_mm=120.00,
            head_width_mm=66.80,
            head_height_mm=50.00,
            web_thickness_mm=14.00,
            area_cm2=61.78,
            mass_kg_per_m=48.50,
        ),
        _modern_rail_profile(
            name="S41",
            height_mm=138.0,
            head_width_mm=67.0,
            foot_width_mm=125.0,
            area_cm2=52.7,
            mass_kg_per_m=41.3,
            iy_cm4=1368.0,
            iz_cm4=276.0,
            wyh_cm3=196.0,
            wyf_cm3=200.5,
            wz_cm3=44.2,
        ),
        _modern_rail_profile(
            name="S49",
            height_mm=149.0,
            head_width_mm=67.0,
            foot_width_mm=125.0,
            area_cm2=63.0,
            mass_kg_per_m=49.4,
            iy_cm4=1819.0,
            iz_cm4=320.0,
            wyh_cm3=240.0,
            wyf_cm3=248.0,
            wz_cm3=51.2,
            head_height_mm=51.50,
            web_thickness_mm=14.00,
        ),
        _rail_dimensions(
            name="49E1",
            height_mm=149.00,
            foot_width_mm=125.00,
            head_width_mm=67.00,
            head_height_mm=14.00,
            web_thickness_mm=7.06,
            area_cm2=62.55,
            mass_kg_per_m=49.10,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(1796.3, 318.4),
            section_modulus_head_cm3=239.4,
            section_modulus_foot_cm3=246.2,
            section_modulus_z_cm3=246.2,
        ),
        _rail_dimensions(
            name="49E5",
            height_mm=149.00,
            foot_width_mm=125.00,
            head_width_mm=67.00,
            head_height_mm=51.50,
            web_thickness_mm=14.00,
            area_cm2=62.59,
            mass_kg_per_m=50.37,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(1987.8, 365.0),
            section_modulus_head_cm3=246.7,
            section_modulus_foot_cm3=274.4,
            section_modulus_z_cm3=274.4,
        ),
        _rail_dimensions(
            name="50E1",
            height_mm=152.00,
            foot_width_mm=125.00,
            head_width_mm=65.00,
            head_height_mm=13.58,
            web_thickness_mm=8.21,
            area_cm2=63.65,
            mass_kg_per_m=49.97,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(1988.8, 408.4),
            section_modulus_head_cm3=248.5,
            section_modulus_foot_cm3=280.3,
            section_modulus_z_cm3=280.3,
        ),
        _rail_dimensions(
            name="50E2",
            height_mm=152.00,
            foot_width_mm=125.00,
            head_width_mm=72.00,
            head_height_mm=44.00,
            web_thickness_mm=15.00,
            area_cm2=63.65,
            mass_kg_per_m=50.02,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(2057.8, 351.3),
            section_modulus_head_cm3=259.5,
            section_modulus_foot_cm3=271.8,
            section_modulus_z_cm3=271.8,
        ),
        _rail_dimensions(
            name="50E3",
            height_mm=152.00,
            foot_width_mm=125.00,
            head_width_mm=70.00,
            head_height_mm=48.00,
            web_thickness_mm=14.00,
            area_cm2=63.71,
            mass_kg_per_m=50.17,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(1931.0, 314.7),
            section_modulus_head_cm3=251.4,
            section_modulus_foot_cm3=256.8,
            section_modulus_z_cm3=256.8,
        ),
        _rail_dimensions(
            name="50E4",
            height_mm=152.00,
            foot_width_mm=125.00,
            head_width_mm=70.00,
            head_height_mm=14.10,
            web_thickness_mm=11.49,
            area_cm2=63.62,
            mass_kg_per_m=49.90,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(1844.0, 362.4),
            section_modulus_head_cm3=242.1,
            section_modulus_foot_cm3=256.6,
            section_modulus_z_cm3=256.6,
        ),
        _rail_dimensions(
            name="50E5",
            height_mm=152.00,
            foot_width_mm=125.00,
            head_width_mm=70.00,
            head_height_mm=14.28,
            web_thickness_mm=7.06,
            area_cm2=64.84,
            mass_kg_per_m=50.90,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(2017.8, 396.8),
            section_modulus_head_cm3=248.3,
            section_modulus_foot_cm3=281.3,
            section_modulus_z_cm3=281.3,
        ),
        _rail_dimensions(
            name="50E6 (U50)",
            height_mm=152.00,
            foot_width_mm=125.00,
            head_width_mm=65.00,
            head_height_mm=49.00,
            web_thickness_mm=15.50,
            area_cm2=64.84,
            mass_kg_per_m=52.15,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(1970.9, 434.2),
            section_modulus_head_cm3=247.1,
            section_modulus_foot_cm3=280.6,
            section_modulus_z_cm3=280.6,
        ),
        _modern_rail_profile(
            name="NP46",
            height_mm=142.0,
            head_width_mm=72.0,
            foot_width_mm=120.0,
            area_cm2=59.3,
            mass_kg_per_m=46.6,
            iy_cm4=1605.0,
            iz_cm4=310.0,
            wyh_cm3=224.0,
            wyf_cm3=228.0,
            wz_cm3=52.0,
        ),
        _modern_rail_profile(
            name="UIC54",
            height_mm=159.0,
            head_width_mm=70.0,
            foot_width_mm=140.0,
            area_cm2=69.3,
            mass_kg_per_m=54.4,
            iy_cm4=2346.0,
            iz_cm4=418.0,
            wyh_cm3=279.0,
            wyf_cm3=313.0,
            wz_cm3=60.0,
            head_height_mm=49.40,
            web_thickness_mm=16.00,
        ),
        _rail_dimensions(
            name="52E1",
            height_mm=159.00,
            foot_width_mm=125.00,
            head_width_mm=72.00,
            head_height_mm=14.30,
            web_thickness_mm=6.61,
            area_cm2=69.77,
            mass_kg_per_m=54.77,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(2337.9, 419.2),
            section_modulus_head_cm3=278.7,
            section_modulus_foot_cm3=311.2,
            section_modulus_z_cm3=311.2,
        ),
        _rail_dimensions(
            name="54E2 (UIC54E)",
            height_mm=161.00,
            foot_width_mm=125.00,
            head_width_mm=67.00,
            head_height_mm=51.40,
            web_thickness_mm=16.00,
            area_cm2=68.56,
            mass_kg_per_m=53.82,
        ),
        _rail_dimensions(
            name="54E3 (S54)",
            height_mm=159.00,
            foot_width_mm=125.00,
            head_width_mm=67.00,
            head_height_mm=49.40,
            web_thickness_mm=16.00,
            area_cm2=69.52,
            mass_kg_per_m=54.57,
        ),
        _rail_dimensions(
            name="54E4",
            height_mm=154.00,
            foot_width_mm=125.00,
            head_width_mm=67.00,
            head_height_mm=55.00,
            web_thickness_mm=16.00,
            area_cm2=69.19,
            mass_kg_per_m=54.31,
        ),
        _rail_dimensions(
            name="54E5",
            height_mm=159.00,
            foot_width_mm=140.00,
            head_width_mm=70.20,
            head_height_mm=49.40,
            web_thickness_mm=16.00,
            area_cm2=69.32,
            mass_kg_per_m=56.03,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(2150.4, 418.4),
            section_modulus_head_cm3=255.2,
            section_modulus_foot_cm3=304.0,
            section_modulus_z_cm3=304.0,
        ),
        _rail_dimensions(
            name="55E1",
            height_mm=159.00,
            foot_width_mm=140.00,
            head_width_mm=67.00,
            head_height_mm=14.28,
            web_thickness_mm=7.06,
            area_cm2=71.69,
            mass_kg_per_m=56.30,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(2321.0, 421.6),
            section_modulus_head_cm3=275.5,
            section_modulus_foot_cm3=311.5,
            section_modulus_z_cm3=311.5,
        ),
        _rail_dimensions(
            name="56E1",
            height_mm=159.00,
            foot_width_mm=140.00,
            head_width_mm=69.85,
            head_height_mm=49.21,
            web_thickness_mm=20.00,
            area_cm2=71.69,
            mass_kg_per_m=60.21,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(3038.3, 512.3),
            section_modulus_head_cm3=333.6,
            section_modulus_foot_cm3=375.5,
            section_modulus_z_cm3=375.5,
        ),
        _modern_rail_profile(
            name="UIC60",
            height_mm=172.0,
            head_width_mm=72.0,
            foot_width_mm=150.0,
            area_cm2=76.9,
            mass_kg_per_m=60.3,
            iy_cm4=3055.0,
            iz_cm4=513.0,
            wyh_cm3=336.0,
            wyf_cm3=377.0,
            wz_cm3=68.0,
            head_height_mm=51.00,
            web_thickness_mm=16.50,
        ),
        _modern_rail_profile(
            name="Ri60",
            height_mm=180.0,
            head_width_mm=113.0,
            foot_width_mm=180.0,
            area_cm2=77.1,
            mass_kg_per_m=60.5,
            iy_cm4=3334.0,
            iz_cm4=884.0,
            wyh_cm3=387.0,
            wyf_cm3=355.0,
            wz_cm3=135.0,
        ),
        _rail_dimensions(
            name="60E1",
            height_mm=172.00,
            foot_width_mm=150.00,
            head_width_mm=72.00,
            head_height_mm=14.30,
            web_thickness_mm=7.28,
            area_cm2=76.48,
            mass_kg_per_m=60.03,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(3021.5, 510.5),
            section_modulus_head_cm3=330.8,
            section_modulus_foot_cm3=374.5,
            section_modulus_z_cm3=374.5,
        ),
        _rail_dimensions(
            name="60E2",
            height_mm=172.00,
            foot_width_mm=150.00,
            head_width_mm=72.00,
            head_height_mm=51.00,
            web_thickness_mm=16.50,
            area_cm2=76.48,
            mass_kg_per_m=60.03,
            elastic_modulus_pa=210.0e9,
            moment_inertia_cm4=(3021.5, 510.5),
            section_modulus_head_cm3=330.8,
            section_modulus_foot_cm3=374.5,
            section_modulus_z_cm3=374.5,
        ),
        _rail_dimensions(
            name="90ARA-A (TR45)",
            height_mm=142.90,
            foot_width_mm=130.20,
            head_width_mm=65.10,
            head_height_mm=37.30,
            web_thickness_mm=14.30,
            area_cm2=56.90,
            mass_kg_per_m=44.65,
        ),
        _rail_dimensions(
            name="100RE",
            height_mm=152.40,
            foot_width_mm=136.52,
            head_width_mm=68.26,
            head_height_mm=42.07,
            web_thickness_mm=14.29,
            area_cm2=64.19,
            mass_kg_per_m=50.35,
        ),
        _rail_dimensions(
            name="115RE (TR57)",
            height_mm=168.30,
            foot_width_mm=139.70,
            head_width_mm=69.10,
            head_height_mm=42.90,
            web_thickness_mm=15.90,
            area_cm2=72.32,
            mass_kg_per_m=56.73,
        ),
        _rail_dimensions(
            name="AREMA136RE",
            height_mm=185.70,
            foot_width_mm=152.40,
            head_width_mm=74.60,
            head_height_mm=49.20,
            web_thickness_mm=17.50,
            area_cm2=85.93,
            mass_kg_per_m=67.40,
        ),
    ]
    steel_properties = [
        {
            "name": "Standard rail steel",
            "elastic_modulus_pa": 2.1e11,
            "poisson_ratio": 0.3,
            "thermal_expansion_per_c": 1.15e-5,
            "density_kg_per_m3": 7850.0,
        }
    ]
    admissible_stress = [
        {
            "tensile_strength_mpa": 700.0,
            "yield_stress_mpa": 450.0,
            "residual_stress_mpa": 220.0,
            "temperature_stress_mpa": 100.0,
            "incidental_stress_mpa": 450.0,
            "repeated_stress_mpa": 55.0,
        },
        {
            "tensile_strength_mpa": 900.0,
            "yield_stress_mpa": 580.0,
            "residual_stress_mpa": 220.0,
            "temperature_stress_mpa": 100.0,
            "incidental_stress_mpa": 580.0,
            "repeated_stress_mpa": 220.0,
        },
    ]
    admissible_shear = [
        {
            "tensile_strength_mpa": 700.0,
            "incidental_shear_mpa": 260.0,
            "repeated_shear_mpa": 200.0,
        },
        {
            "tensile_strength_mpa": 900.0,
            "incidental_shear_mpa": 340.0,
            "repeated_shear_mpa": 260.0,
        },
    ]
    sleepers = [
        {
            "name": "Concrete B70",
            "elastic_modulus_pa": 3.5e10,
            "length_m": 2.6,
            "width_m": 0.26,
            "height_m": 0.22,
            "mass_kg": 290.0,
        },
        {
            "name": "Concrete B58",
            "elastic_modulus_pa": 3.3e10,
            "length_m": 2.5,
            "width_m": 0.25,
            "height_m": 0.21,
            "mass_kg": 260.0,
        },
        {
            "name": "Timber 2.4m",
            "elastic_modulus_pa": 1.2e10,
            "length_m": 2.4,
            "width_m": 0.23,
            "height_m": 0.18,
            "mass_kg": 180.0,
        },
    ]
    profiles = [
        {"name": "Ballast 30 MN/m²", "foundation_modulus_n_per_m2": 3.0e7},
        {"name": "Ballast 50 MN/m²", "foundation_modulus_n_per_m2": 5.0e7},
        {"name": "Ballast 80 MN/m²", "foundation_modulus_n_per_m2": 8.0e7},
    ]
    load_cases = [
        {
            "name": "20 t axle load",
            "load_newtons": 196200.0,
            "description": "Single axle load at rail seat.",
        },
        {
            "name": "25 t axle load",
            "load_newtons": 245250.0,
            "description": "Heavy haul axle load.",
        },
    ]
    pads = [
        {
            "name": "Standard Pad",
            "stiffness_newtons_per_meter": 120_000_000.0,
            "thickness_m": 0.01,
        }
    ]

    for rail in rails:
        if "moment_inertia_m4" in rail:
            _upsert_rail_exact(session, rail)
        else:
            _upsert_rail(session, rail)
    for props in steel_properties:
        if not _exists_by_name(session, RailSteelProperties, props["name"]):
            session.add(RailSteelProperties(**props))
    for stress in admissible_stress:
        if not _exists_by_tensile_strength(
            session, RailAdmissibleStress, stress["tensile_strength_mpa"]
        ):
            session.add(RailAdmissibleStress(**stress))
    for stress in admissible_shear:
        if not _exists_by_tensile_strength(
            session, RailAdmissibleShearStress, stress["tensile_strength_mpa"]
        ):
            session.add(RailAdmissibleShearStress(**stress))
    for sleeper in sleepers:
        if not _exists_by_name(session, Sleeper, sleeper["name"]):
            crud.create_sleeper(session, **sleeper)
    for profile in profiles:
        if not _exists_by_name(session, SupportProfile, profile["name"]):
            crud.create_support_profile(session, **profile)
    for load_case in load_cases:
        if not _exists_by_name(session, LoadCase, load_case["name"]):
            crud.create_load_case(session, **load_case)
    for pad in pads:
        if not _exists_by_name(session, Pad, pad["name"]):
            crud.create_pad(session, **pad)

    seed_example_projects(session)


def seed_example_projects(session: Session) -> None:
    """Seed example projects with track configs and result metadata."""
    rail_uic60 = session.scalar(select(Rail).where(Rail.name == "UIC60"))
    rail_uic54 = session.scalar(select(Rail).where(Rail.name == "UIC54"))
    rail_s49 = session.scalar(select(Rail).where(Rail.name == "S49"))
    rail_ri60 = session.scalar(select(Rail).where(Rail.name == "Ri60"))
    sleeper_b70 = session.scalar(select(Sleeper).where(Sleeper.name == "Concrete B70"))
    sleeper_b58 = session.scalar(select(Sleeper).where(Sleeper.name == "Concrete B58"))
    sleeper_timber = session.scalar(select(Sleeper).where(Sleeper.name == "Timber 2.4m"))
    pad = session.scalar(select(Pad).where(Pad.name == "Standard Pad"))
    ballast_30 = session.scalar(
        select(SupportProfile).where(SupportProfile.name == "Ballast 30 MN/m²")
    )
    ballast_50 = session.scalar(
        select(SupportProfile).where(SupportProfile.name == "Ballast 50 MN/m²")
    )
    ballast_80 = session.scalar(
        select(SupportProfile).where(SupportProfile.name == "Ballast 80 MN/m²")
    )
    load_20t = session.scalar(select(LoadCase).where(LoadCase.name == "20 t axle load"))
    load_25t = session.scalar(select(LoadCase).where(LoadCase.name == "25 t axle load"))

    if not all(
        [
            rail_uic60,
            rail_uic54,
            rail_s49,
            rail_ri60,
            sleeper_b70,
            sleeper_b58,
            sleeper_timber,
            pad,
            ballast_30,
            ballast_50,
            ballast_80,
            load_20t,
            load_25t,
        ]
    ):
        raise ValueError("Seed prerequisites missing for example projects")

    def get_or_create_project(
        name: str,
        description: str,
        *,
        vehicle_type: str | None = None,
        vehicle_subtype: str | None = None,
        design_speed_kmh: float | None = None,
        design_wheel_radius_mm: float | None = None,
    ) -> Project:
        project = session.scalar(select(Project).where(Project.name == name))
        if project is None:
            project = crud.create_project(
                session,
                name=name,
                description=description,
                vehicle_type=vehicle_type,
                vehicle_subtype=vehicle_subtype,
                design_speed_kmh=design_speed_kmh,
                design_wheel_radius_mm=design_wheel_radius_mm,
            )
            return project
        updates: dict[str, str | float | None] = {}
        if vehicle_type and not project.vehicle_type:
            updates["vehicle_type"] = vehicle_type
        if vehicle_subtype and not project.vehicle_subtype:
            updates["vehicle_subtype"] = vehicle_subtype
        if design_speed_kmh is not None and project.design_speed_kmh is None:
            updates["design_speed_kmh"] = design_speed_kmh
        if design_wheel_radius_mm is not None and project.design_wheel_radius_mm is None:
            updates["design_wheel_radius_mm"] = design_wheel_radius_mm
        if updates:
            project = crud.update_project(session, project, **updates)
        return project

    def ensure_track_config(
        *,
        project: Project,
        name: str,
        rail: Rail,
        sleeper: Sleeper,
        pad: Pad,
        support: SupportProfile,
        sleeper_spacing_m: float,
        gauge_m: float,
    ) -> TrackConfig:
        existing = session.scalar(
            select(TrackConfig).where(
                TrackConfig.project_id == project.id,
                TrackConfig.name == name,
            )
        )
        if existing is not None:
            return existing
        config = TrackConfig(
            name=name,
            project_id=project.id,
            rail_id=rail.id,
            sleeper_id=sleeper.id,
            pad_id=pad.id,
            support_profile_id=support.id,
            sleeper_spacing_m=sleeper_spacing_m,
            gauge_m=gauge_m,
        )
        session.add(config)
        session.flush()
        return config

    def ensure_result(
        *,
        project: Project,
        config: TrackConfig,
        load_case: LoadCase,
        max_deflection_m: float,
        max_moment_nm: float,
    ) -> None:
        existing = session.scalar(
            select(Result).where(
                Result.project_id == project.id,
                Result.track_config_id == config.id,
                Result.load_case_id == load_case.id,
            )
        )
        if existing is not None:
            return
        session.add(
            Result(
                project_id=project.id,
                track_config_id=config.id,
                load_case_id=load_case.id,
                max_deflection_m=max_deflection_m,
                max_moment_nm=max_moment_nm,
            )
        )

    project_fast = get_or_create_project(
        "High Speed Baseline",
        "Baseline high-speed alignment with concrete sleepers.",
        vehicle_type="high_speed",
        design_wheel_radius_mm=0.5 * 985.0,
    )
    config_fast = ensure_track_config(
        project=project_fast,
        name="HS Concrete Track",
        rail=rail_uic60,
        sleeper=sleeper_b70,
        pad=pad,
        support=ballast_80,
        sleeper_spacing_m=0.6,
        gauge_m=1.435,
    )
    ensure_result(
        project=project_fast,
        config=config_fast,
        load_case=load_20t,
        max_deflection_m=0.0024,
        max_moment_nm=18_500.0,
    )
    ensure_track_config(
        project=project_fast,
        name="HS Ballast Track",
        rail=rail_uic60,
        sleeper=sleeper_b58,
        pad=pad,
        support=ballast_50,
        sleeper_spacing_m=0.6,
        gauge_m=1.435,
    )

    project_heavy = get_or_create_project(
        "Heavy Haul Baseline",
        "Heavy haul reference track for freight loading.",
        vehicle_type="freight_heavy_haul",
        design_wheel_radius_mm=0.5 * 920.0,
    )
    config_heavy = ensure_track_config(
        project=project_heavy,
        name="Freight Concrete Track",
        rail=rail_uic60,
        sleeper=sleeper_b70,
        pad=pad,
        support=ballast_50,
        sleeper_spacing_m=0.65,
        gauge_m=1.435,
    )
    ensure_result(
        project=project_heavy,
        config=config_heavy,
        load_case=load_25t,
        max_deflection_m=0.0031,
        max_moment_nm=24_800.0,
    )
    ensure_track_config(
        project=project_heavy,
        name="Freight Timber Track",
        rail=rail_uic54,
        sleeper=sleeper_timber,
        pad=pad,
        support=ballast_30,
        sleeper_spacing_m=0.65,
        gauge_m=1.435,
    )

    project_metro = get_or_create_project(
        "Metro Heavy",
        "Metro heavy-duty alignment with reduced maintenance windows.",
        vehicle_type="heavy_metro",
        design_wheel_radius_mm=0.5 * 885.0,
    )
    ensure_track_config(
        project=project_metro,
        name="Metro Concrete Track",
        rail=rail_ri60,
        sleeper=sleeper_b70,
        pad=pad,
        support=ballast_80,
        sleeper_spacing_m=0.6,
        gauge_m=1.435,
    )
    ensure_track_config(
        project=project_metro,
        name="Metro Ballast Track",
        rail=rail_uic60,
        sleeper=sleeper_b58,
        pad=pad,
        support=ballast_50,
        sleeper_spacing_m=0.6,
        gauge_m=1.435,
    )

    project_lrt = get_or_create_project(
        "LRT",
        "Light rail transit baseline for mixed-traffic operation.",
        vehicle_type="lrt",
        design_wheel_radius_mm=0.5 * 735.0,
    )
    ensure_track_config(
        project=project_lrt,
        name="LRT Ballast Track",
        rail=rail_s49,
        sleeper=sleeper_b58,
        pad=pad,
        support=ballast_50,
        sleeper_spacing_m=0.6,
        gauge_m=1.435,
    )
    ensure_track_config(
        project=project_lrt,
        name="LRT Timber Track",
        rail=rail_uic54,
        sleeper=sleeper_timber,
        pad=pad,
        support=ballast_30,
        sleeper_spacing_m=0.6,
        gauge_m=1.435,
    )

    session.commit()

"""Analysis pipeline helpers for BOEF plotting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from core.model import (
    PointLoad,
    beam_parameter_beta,
    contraflexure_distance,
    deflection_at,
    max_deflection_single_load,
    rail_seat_load_from_deflection,
    rail_base_stress,
    moment_at,
    reaction_at,
    shear_at,
    sleeper_seat_loads,
    zero_moment_distance,
)

ADMISSIBLE_BENDING_STRESS_MPA = {
    700.0: {"repeated": 55.0, "incidental": 450.0},
    900.0: {"repeated": 220.0, "incidental": 580.0},
}

ADMISSIBLE_SHEAR_STRESS_MPA = {
    700.0: {"repeated": 200.0, "incidental": 260.0},
    900.0: {"repeated": 260.0, "incidental": 340.0},
}

MTM_TRACK_CLASS_TO_VQI = {
    1: 45.0,
    2: 50.0,
    3: 65.0,
    4: 75.0,
    5: 75.0,
}


@dataclass(frozen=True)
class DesignInputs:
    """Rail design parameters for stress checks."""

    speed_kmh: float
    track_factor: float
    probability_factor: float
    wheel_radius_mm: float
    tensile_strength_mpa: float = 700.0
    curve_load_factor: float = 1.2
    on_curve: bool = False
    track_class: int | None = None
    vqi: float | None = None
    confidence_limit: float | None = None
    curve_radius_m: float | None = None
    factor_of_safety_f1: float | None = None
    factor_of_safety_f2: float = 1.0
    ballast_depth_m: float | None = None
    fill_depth_m: float = 0.0
    rail_centres_m: float | None = None


@dataclass(frozen=True)
class A3902Checks:
    """A3902 quasi-static design values (SI units)."""

    static_vertical_wheel_load_n: float
    dynamic_vertical_wheel_load_n: float
    dynamic_factor_phi: float
    track_condition_factor_delta: float
    velocity_factor_eta: float
    confidence_limit_tc: float
    vqi: float
    rail_deflection_max_m: float
    rail_seat_load_n: float
    ballast_contact_pressure_pa: float
    formation_pressure_pa: float | None
    subgrade_pressure_pa: float | None
    effective_bearing_length_m: float


@dataclass(frozen=True)
class DesignSummary:
    """Rail design outputs (SI units unless noted)."""

    daf: float
    effective_load_n: float
    characteristic_length_m: float
    mean_bending_stress_pa: float
    max_bending_stress_pa: float
    admissible_bending_stress_pa: float
    admissible_bending_repeated_pa: float
    admissible_bending_incidental_pa: float
    bending_pass: bool
    max_shear_stress_pa: float
    permissible_shear_stress_pa: float
    admissible_shear_repeated_pa: float
    admissible_shear_incidental_pa: float
    shear_pass: bool
    combined_head_stress_pa: float
    combined_foot_stress_pa: float
    influence_stress_pa: float | None
    a3902_checks: A3902Checks | None = None


@dataclass(frozen=True)
class AnalysisInputs:
    """Inputs required for a track analysis run (SI units)."""

    loads: Sequence[PointLoad]
    foundation_modulus_n_per_m2: float
    elastic_modulus_pa: float
    moment_inertia_m4: float
    section_modulus_m3: float
    sleeper_spacing_m: float
    sleeper_length_m: float
    sleeper_width_m: float
    sample_count: int = 401
    x_domain_m: tuple[float, float] | None = None
    section_modulus_head_m3: float | None = None
    section_modulus_foot_m3: float | None = None
    area_m2: float | None = None
    discrete_support_stiffness_n_per_m: float | None = None
    design_inputs: DesignInputs | None = None


@dataclass(frozen=True)
class AnalysisResult:
    """Analysis output data (SI units)."""

    x_m: list[float]
    deflection_m: list[float]
    moment_nm: list[float]
    shear_n: list[float]
    reaction_n_per_m: list[float]
    sleeper_positions_m: list[float]
    sleeper_loads_n: list[float]
    sleeper_pressures_pa: list[float]
    summary: "AnalysisSummary"
    slope_rad: list[float] | None = None
    rotation_rad: list[float] | None = None
    shear_angle_rad: list[float] | None = None
    winkler_reaction_n_per_m: list[float] | None = None
    pasternak_shear_reaction_n_per_m: list[float] | None = None
    left_deflection_m: list[float] | None = None
    right_deflection_m: list[float] | None = None
    left_moment_nm: list[float] | None = None
    right_moment_nm: list[float] | None = None
    left_shear_n: list[float] | None = None
    right_shear_n: list[float] | None = None
    left_reaction_n_per_m: list[float] | None = None
    right_reaction_n_per_m: list[float] | None = None
    left_sleeper_loads_n: list[float] | None = None
    right_sleeper_loads_n: list[float] | None = None
    railpad_force_n_per_m: list[float] | None = None
    trackbed_force_n_per_m: list[float] | None = None
    sleeper_deflection_m: list[float] | None = None


@dataclass(frozen=True)
class Extremum:
    """Extreme magnitude value and its position."""

    value: float
    position_m: float


@dataclass(frozen=True)
class AnalysisSummary:
    """Summary values for quick review (SI units)."""

    beta_per_m: float
    zero_moment_distance_m: float
    contraflexure_distance_m: float
    max_deflection: Extremum
    max_moment: Extremum
    max_shear: Extremum
    max_reaction: Extremum
    max_sleeper_load: Extremum
    max_sleeper_pressure: Extremum
    max_rail_base_stress_pa: float
    design_summary: DesignSummary | None = None
    support_model: str | None = None
    support_k_eq_n_per_m2: float | None = None
    max_railpad_force_n_per_m: float | None = None
    max_trackbed_force_n_per_m: float | None = None
    max_sleeper_deflection_m: float | None = None


def compute_track_response(inputs: AnalysisInputs) -> AnalysisResult:
    """Compute response arrays for plotting and reporting."""
    _require_positive(inputs.foundation_modulus_n_per_m2, "foundation_modulus_n_per_m2")
    _require_positive(inputs.elastic_modulus_pa, "elastic_modulus_pa")
    _require_positive(inputs.moment_inertia_m4, "moment_inertia_m4")
    _require_positive(inputs.section_modulus_m3, "section_modulus_m3")
    _require_positive(inputs.sleeper_spacing_m, "sleeper_spacing_m")
    _require_positive(inputs.sleeper_length_m, "sleeper_length_m")
    _require_positive(inputs.sleeper_width_m, "sleeper_width_m")
    _require_sample_count(inputs.sample_count)

    beta = beam_parameter_beta(
        inputs.foundation_modulus_n_per_m2,
        inputs.elastic_modulus_pa,
        inputs.moment_inertia_m4,
    )
    x_values, domain_start, domain_end = _build_analysis_grid(
        beta,
        inputs.sample_count,
        inputs.x_domain_m,
    )

    deflections = [
        deflection_at(
            x,
            inputs.loads,
            inputs.foundation_modulus_n_per_m2,
            inputs.elastic_modulus_pa,
            inputs.moment_inertia_m4,
        )
        for x in x_values
    ]
    moments = [
        moment_at(
            x,
            inputs.loads,
            inputs.foundation_modulus_n_per_m2,
            inputs.elastic_modulus_pa,
            inputs.moment_inertia_m4,
        )
        for x in x_values
    ]
    shears = [
        shear_at(
            x,
            inputs.loads,
            inputs.foundation_modulus_n_per_m2,
            inputs.elastic_modulus_pa,
            inputs.moment_inertia_m4,
        )
        for x in x_values
    ]
    reactions = [
        reaction_at(
            x,
            inputs.loads,
            inputs.foundation_modulus_n_per_m2,
            inputs.elastic_modulus_pa,
            inputs.moment_inertia_m4,
        )
        for x in x_values
    ]
    slopes = _first_derivative(x_values, deflections)

    sleeper_start, sleeper_end = _build_sleeper_domain(inputs.loads, beta)
    sleeper_positions = _build_sleeper_positions(
        start=sleeper_start,
        end=sleeper_end,
        spacing=inputs.sleeper_spacing_m,
    )
    sleeper_loads = sleeper_seat_loads(
        sleeper_positions,
        inputs.sleeper_spacing_m,
        inputs.loads,
        inputs.foundation_modulus_n_per_m2,
        inputs.elastic_modulus_pa,
        inputs.moment_inertia_m4,
    )
    bearing_area = inputs.sleeper_length_m * inputs.sleeper_width_m
    sleeper_pressures = [load / bearing_area for load in sleeper_loads]

    max_deflection = _max_abs_with_position(x_values, deflections)
    max_moment = _max_abs_with_position(x_values, moments)
    max_shear = _max_abs_with_position(x_values, shears)
    max_reaction = _max_abs_with_position(x_values, reactions)
    max_sleeper_load = _max_abs_with_position(sleeper_positions, sleeper_loads)
    max_sleeper_pressure = _max_abs_with_position(sleeper_positions, sleeper_pressures)
    max_stress = abs(rail_base_stress(max_moment.value, inputs.section_modulus_m3))

    design_summary = build_design_summary(inputs, beta, max_moment_nm=max_moment.value)

    summary = AnalysisSummary(
        beta_per_m=beta,
        zero_moment_distance_m=zero_moment_distance(beta),
        contraflexure_distance_m=contraflexure_distance(beta),
        max_deflection=max_deflection,
        max_moment=max_moment,
        max_shear=max_shear,
        max_reaction=max_reaction,
        max_sleeper_load=max_sleeper_load,
        max_sleeper_pressure=max_sleeper_pressure,
        max_rail_base_stress_pa=max_stress,
        design_summary=design_summary,
    )

    return AnalysisResult(
        x_m=x_values,
        deflection_m=deflections,
        moment_nm=moments,
        shear_n=shears,
        reaction_n_per_m=reactions,
        sleeper_positions_m=sleeper_positions,
        sleeper_loads_n=sleeper_loads,
        sleeper_pressures_pa=sleeper_pressures,
        summary=summary,
        slope_rad=slopes,
        rotation_rad=slopes,
        winkler_reaction_n_per_m=reactions,
        pasternak_shear_reaction_n_per_m=[0.0 for _ in reactions],
    )


def build_load_domain(
    *,
    loads: Sequence[PointLoad],
    foundation_modulus_n_per_m2: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
    margin_factor: float = 5.0,
) -> tuple[float, float]:
    """Compute an analysis domain that spans all loads with a decay margin."""
    if not loads:
        raise ValueError("At least one load is required to build the analysis domain.")
    _require_positive(margin_factor, "margin_factor")
    beta = beam_parameter_beta(
        foundation_modulus_n_per_m2,
        elastic_modulus_pa,
        moment_inertia_m4,
    )
    margin = margin_factor / beta
    positions = [load.position_m for load in loads]
    left = min(positions) - margin
    right = max(positions) + margin
    if right <= left:
        raise ValueError("Analysis domain must be positive length.")
    return left, right


def eisenmann_dynamic_amplification(
    speed_kmh: float,
    probability_factor: float,
    track_factor: float,
) -> float:
    """Compute Eisenmann dynamic amplification factor (DAF)."""
    eta = eisenmann_velocity_dependent_factor(speed_kmh)
    return eisenmann_dynamic_factor(track_factor, eta, probability_factor)


def vqi_for_track_class(track_class: int) -> float:
    """Return the MTM VQI value for track class 1..5."""
    if track_class not in MTM_TRACK_CLASS_TO_VQI:
        raise ValueError("track_class must be one of 1, 2, 3, 4, 5")
    return MTM_TRACK_CLASS_TO_VQI[track_class]


def eisenmann_track_condition_factor(vqi: float) -> float:
    """Compute A3902 track condition factor: delta = VQI / 200."""
    _require_positive(vqi, "vqi")
    return vqi / 200.0


def eisenmann_velocity_dependent_factor(speed_kmh: float) -> float:
    """Compute A3902 velocity factor eta."""
    _require_non_negative(speed_kmh, "speed_kmh")
    if speed_kmh <= 60.0:
        return 1.0
    return 1.0 + (speed_kmh - 60.0) / 140.0


def eisenmann_dynamic_factor(
    track_condition_factor: float,
    velocity_factor: float,
    confidence_limit: float,
) -> float:
    """Compute A3902 dynamic factor phi = 1 + delta * eta * tc."""
    _require_positive(track_condition_factor, "track_condition_factor")
    _require_positive(velocity_factor, "velocity_factor")
    _require_positive(confidence_limit, "confidence_limit")
    return 1.0 + track_condition_factor * velocity_factor * confidence_limit


def dynamic_vertical_wheel_load(static_vertical_wheel_load_n: float, dynamic_factor_phi: float) -> float:
    """Compute A3902 dynamic vertical wheel load PDV = PSV * phi."""
    _require_non_negative(static_vertical_wheel_load_n, "static_vertical_wheel_load_n")
    _require_positive(dynamic_factor_phi, "dynamic_factor_phi")
    return static_vertical_wheel_load_n * dynamic_factor_phi


def ballast_contact_pressure_a3902(
    *,
    rail_seat_load_n: float,
    sleeper_width_m: float,
    sleeper_length_m: float,
    rail_centres_m: float | None,
    factor_of_safety_f2: float = 1.0,
) -> tuple[float, float]:
    """Compute A3902 ballast pressure using the effective sleeper area."""
    _require_non_negative(rail_seat_load_n, "rail_seat_load_n")
    _require_positive(sleeper_width_m, "sleeper_width_m")
    _require_positive(sleeper_length_m, "sleeper_length_m")
    _require_positive(factor_of_safety_f2, "factor_of_safety_f2")
    if rail_centres_m is None:
        raise ValueError("rail_centres_m must be provided for A3902 pressure checks")
    _require_positive(rail_centres_m, "rail_centres_m")
    effective_length = sleeper_length_m - rail_centres_m
    _require_positive(effective_length, "effective_bearing_length_m")
    return (
        factor_of_safety_f2 * rail_seat_load_n / (sleeper_width_m * effective_length),
        effective_length,
    )


def formation_pressure_a3902(
    *,
    ballast_contact_pressure_pa: float,
    ballast_depth_m: float,
    sleeper_width_m: float,
    effective_bearing_length_m: float,
) -> float:
    """Compute A3902 formation pressure PF at ballast depth."""
    _require_non_negative(ballast_contact_pressure_pa, "ballast_contact_pressure_pa")
    _require_positive(ballast_depth_m, "ballast_depth_m")
    _require_positive(sleeper_width_m, "sleeper_width_m")
    _require_positive(effective_bearing_length_m, "effective_bearing_length_m")
    equivalent_area_radius_sq = (sleeper_width_m * effective_bearing_length_m) / math.pi
    spread_term = (equivalent_area_radius_sq + ballast_depth_m**2) ** 1.5
    return ballast_contact_pressure_pa * (1.0 - (ballast_depth_m**3) / spread_term)


def subgrade_pressure_a3902(
    *,
    ballast_contact_pressure_pa: float,
    ballast_depth_m: float,
    fill_depth_m: float,
    sleeper_width_m: float,
    effective_bearing_length_m: float,
) -> float:
    """Compute A3902 subgrade pressure PS at ballast + fill depth."""
    _require_non_negative(ballast_contact_pressure_pa, "ballast_contact_pressure_pa")
    _require_positive(ballast_depth_m, "ballast_depth_m")
    _require_non_negative(fill_depth_m, "fill_depth_m")
    _require_positive(sleeper_width_m, "sleeper_width_m")
    _require_positive(effective_bearing_length_m, "effective_bearing_length_m")
    total_depth = ballast_depth_m + fill_depth_m
    _require_positive(total_depth, "ballast_depth_m + fill_depth_m")
    equivalent_area_radius_sq = (sleeper_width_m * effective_bearing_length_m) / math.pi
    spread_term = (equivalent_area_radius_sq + total_depth**2) ** 1.5
    return ballast_contact_pressure_pa * (1.0 - (total_depth**3) / spread_term)


def _mean_bending_stress(
    load_n: float, characteristic_length_m: float, section_modulus_m3: float
) -> float:
    _require_non_negative(load_n, "load_n")
    _require_positive(characteristic_length_m, "characteristic_length_m")
    _require_positive(section_modulus_m3, "section_modulus_m3")
    return load_n * characteristic_length_m / (4.0 * section_modulus_m3)


def _max_contact_shear_stress(load_kn: float, wheel_radius_mm: float) -> float:
    if load_kn <= 0:
        raise ValueError("load_kn must be positive")
    if wheel_radius_mm <= 0:
        raise ValueError("wheel_radius_mm must be positive")
    shear_mpa = 412.0 * math.sqrt(load_kn / wheel_radius_mm)
    return shear_mpa * 1_000_000.0


def _admissible_limits_mpa(
    table: dict[float, dict[str, float]],
    tensile_strength_mpa: float,
) -> dict[str, float]:
    if tensile_strength_mpa in table:
        return table[tensile_strength_mpa]
    return table[min(table.keys())]


def admissible_bending_stresses_mpa(tensile_strength_mpa: float) -> tuple[float, float]:
    limits = _admissible_limits_mpa(ADMISSIBLE_BENDING_STRESS_MPA, tensile_strength_mpa)
    return limits["repeated"], limits["incidental"]


def admissible_shear_stresses_mpa(tensile_strength_mpa: float) -> tuple[float, float]:
    limits = _admissible_limits_mpa(ADMISSIBLE_SHEAR_STRESS_MPA, tensile_strength_mpa)
    return limits["repeated"], limits["incidental"]


def _permissible_shear_stress(tensile_strength_mpa: float) -> float:
    repeated_mpa, _ = admissible_shear_stresses_mpa(tensile_strength_mpa)
    return repeated_mpa * 1_000_000.0


def _admissible_fatigue_stress(tensile_strength_mpa: float) -> float:
    repeated_mpa, _ = admissible_bending_stresses_mpa(tensile_strength_mpa)
    return repeated_mpa * 1_000_000.0


def _combined_stress_envelope(
    *,
    base_bending_pa: float,
    residual_mpa: float,
    temperature_mpa: float,
    curve_mpa: float,
    sign_flip_temperature: bool = False,
) -> float:
    base_mpa = abs(base_bending_pa) / 1_000_000.0
    residual = abs(residual_mpa)
    temp = abs(temperature_mpa)
    curve = abs(curve_mpa)
    candidates: list[float] = []
    for temp_sign in (-1.0, 1.0):
        for curve_sign in (-1.0, 1.0):
            temp_component = temp_sign * temp
            if sign_flip_temperature:
                temp_component *= -1.0
            value = -residual + 0.9 * (base_mpa + temp_component + curve_sign * curve)
            candidates.append(value)
    return max(abs(value) for value in candidates) * 1_000_000.0


def build_design_summary(
    inputs: AnalysisInputs,
    beta: float,
    *,
    max_moment_nm: float | None = None,
) -> DesignSummary | None:
    """Build design summary values using either max load or combined moment results."""
    if inputs.design_inputs is None:
        return None
    design = inputs.design_inputs
    if not inputs.loads:
        return None
    section_modulus_foot = inputs.section_modulus_foot_m3 or inputs.section_modulus_m3
    section_modulus_head = inputs.section_modulus_head_m3 or inputs.section_modulus_m3
    curve_factor = design.curve_load_factor if design.on_curve else 1.0
    characteristic_length = 1.0 / beta
    # A3902 PSV should remain the static wheel load before curve multipliers.
    base_vertical_wheel_load_n = max(load.load_newtons for load in inputs.loads)
    tc = design.confidence_limit if design.confidence_limit is not None else design.probability_factor
    delta, vqi = _resolve_track_condition(design)
    eta = eisenmann_velocity_dependent_factor(design.speed_kmh)
    daf = eisenmann_dynamic_factor(delta, eta, tc)
    if max_moment_nm is not None:
        combined_moment = abs(max_moment_nm) * curve_factor
        mean_stress = abs(rail_base_stress(combined_moment, section_modulus_foot))
        mean_stress_head = abs(rail_base_stress(combined_moment, section_modulus_head))
        effective_load_n = 4.0 * beta * abs(max_moment_nm) * curve_factor
    else:
        effective_load_n = base_vertical_wheel_load_n * curve_factor
        mean_stress = _mean_bending_stress(
            effective_load_n,
            characteristic_length,
            section_modulus_foot,
        )
        mean_stress_head = _mean_bending_stress(
            effective_load_n,
            characteristic_length,
            section_modulus_head,
        )
    max_stress = mean_stress * daf
    admissible_repeated_mpa, admissible_incidental_mpa = admissible_bending_stresses_mpa(
        design.tensile_strength_mpa
    )
    admissible_stress = _admissible_fatigue_stress(design.tensile_strength_mpa)
    admissible_incidental_pa = admissible_incidental_mpa * 1_000_000.0
    bending_pass = max_stress <= admissible_stress
    load_kn = effective_load_n / 1000.0
    shear_stress = _max_contact_shear_stress(load_kn, design.wheel_radius_mm)
    permissible_repeated_mpa, permissible_incidental_mpa = admissible_shear_stresses_mpa(
        design.tensile_strength_mpa
    )
    permissible_shear = _permissible_shear_stress(design.tensile_strength_mpa)
    permissible_incidental_pa = permissible_incidental_mpa * 1_000_000.0
    shear_pass = shear_stress <= permissible_shear

    combined_head = _combined_stress_envelope(
        base_bending_pa=mean_stress_head * daf,
        residual_mpa=40.0,
        temperature_mpa=100.0,
        curve_mpa=25.0,
    )
    combined_foot = _combined_stress_envelope(
        base_bending_pa=max_stress,
        residual_mpa=60.0,
        temperature_mpa=100.0,
        curve_mpa=50.0,
        sign_flip_temperature=True,
    )

    influence_stress = None
    if inputs.area_m2 and inputs.discrete_support_stiffness_n_per_m:
        inertia = inputs.moment_inertia_m4
        area = inputs.area_m2
        term = (area * inertia ** 0.25) / (4.0 * section_modulus_foot)
        stiffness_term = (4.0 * inputs.elastic_modulus_pa * inputs.sleeper_spacing_m /
                          inputs.discrete_support_stiffness_n_per_m) ** 0.25
        influence_stress = (effective_load_n / area) * term * stiffness_term

    a3902_checks = _build_a3902_checks(
        inputs=inputs,
        design=design,
        beta=beta,
        static_vertical_wheel_load_n=base_vertical_wheel_load_n,
        vqi=vqi,
        delta=delta,
        eta=eta,
        tc=tc,
        phi=daf,
    )

    return DesignSummary(
        daf=daf,
        effective_load_n=effective_load_n,
        characteristic_length_m=characteristic_length,
        mean_bending_stress_pa=mean_stress,
        max_bending_stress_pa=max_stress,
        admissible_bending_stress_pa=admissible_stress,
        admissible_bending_repeated_pa=admissible_stress,
        admissible_bending_incidental_pa=admissible_incidental_pa,
        bending_pass=bending_pass,
        max_shear_stress_pa=shear_stress,
        permissible_shear_stress_pa=permissible_shear,
        admissible_shear_repeated_pa=permissible_shear,
        admissible_shear_incidental_pa=permissible_incidental_pa,
        shear_pass=shear_pass,
        combined_head_stress_pa=combined_head,
        combined_foot_stress_pa=combined_foot,
        influence_stress_pa=influence_stress,
        a3902_checks=a3902_checks,
    )


def _resolve_track_condition(design: DesignInputs) -> tuple[float, float]:
    if design.track_class is not None:
        vqi = vqi_for_track_class(design.track_class)
        return eisenmann_track_condition_factor(vqi), vqi
    if design.vqi is not None:
        vqi = design.vqi
        return eisenmann_track_condition_factor(vqi), vqi
    delta = design.track_factor
    _require_positive(delta, "track_factor")
    return delta, delta * 200.0


def _build_a3902_checks(
    *,
    inputs: AnalysisInputs,
    design: DesignInputs,
    beta: float,
    static_vertical_wheel_load_n: float,
    vqi: float,
    delta: float,
    eta: float,
    tc: float,
    phi: float,
) -> A3902Checks:
    pdv = dynamic_vertical_wheel_load(static_vertical_wheel_load_n, phi)
    ymax = max_deflection_single_load(
        pdv,
        inputs.foundation_modulus_n_per_m2,
        beta,
    )
    f1 = _resolve_factor_of_safety_f1(design)
    qr = rail_seat_load_from_deflection(
        sleeper_spacing_m=inputs.sleeper_spacing_m,
        foundation_modulus_n_per_m2=inputs.foundation_modulus_n_per_m2,
        max_deflection_m=ymax,
        factor=f1,
    )
    pa, effective_bearing_length_m = ballast_contact_pressure_a3902(
        rail_seat_load_n=qr,
        sleeper_width_m=inputs.sleeper_width_m,
        sleeper_length_m=inputs.sleeper_length_m,
        rail_centres_m=design.rail_centres_m,
        factor_of_safety_f2=design.factor_of_safety_f2,
    )
    pf = None
    ps = None
    if design.ballast_depth_m is not None and design.ballast_depth_m > 0.0:
        pf = formation_pressure_a3902(
            ballast_contact_pressure_pa=pa,
            ballast_depth_m=design.ballast_depth_m,
            sleeper_width_m=inputs.sleeper_width_m,
            effective_bearing_length_m=effective_bearing_length_m,
        )
        ps = subgrade_pressure_a3902(
            ballast_contact_pressure_pa=pa,
            ballast_depth_m=design.ballast_depth_m,
            fill_depth_m=design.fill_depth_m,
            sleeper_width_m=inputs.sleeper_width_m,
            effective_bearing_length_m=effective_bearing_length_m,
        )
    return A3902Checks(
        static_vertical_wheel_load_n=static_vertical_wheel_load_n,
        dynamic_vertical_wheel_load_n=pdv,
        dynamic_factor_phi=phi,
        track_condition_factor_delta=delta,
        velocity_factor_eta=eta,
        confidence_limit_tc=tc,
        vqi=vqi,
        rail_deflection_max_m=ymax,
        rail_seat_load_n=qr,
        ballast_contact_pressure_pa=pa,
        formation_pressure_pa=pf,
        subgrade_pressure_pa=ps,
        effective_bearing_length_m=effective_bearing_length_m,
    )


def _resolve_factor_of_safety_f1(design: DesignInputs) -> float:
    if design.factor_of_safety_f1 is not None:
        _require_positive(design.factor_of_safety_f1, "factor_of_safety_f1")
        return design.factor_of_safety_f1
    if design.curve_radius_m is not None:
        _require_positive(design.curve_radius_m, "curve_radius_m")
        return 1.25 if design.curve_radius_m >= 450.0 else 1.35
    if design.on_curve:
        return 1.35
    return 1.25


def _build_sleeper_positions(start: float, end: float, spacing: float) -> list[float]:
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    count = int(math.floor((end - start) / spacing))
    return [start + i * spacing for i in range(count + 1)]


def _build_sleeper_domain(
    loads: Sequence[PointLoad],
    beta: float,
    *,
    margin_factor: float = 10.0,
) -> tuple[float, float]:
    if not loads:
        raise ValueError("At least one load is required to build the sleeper domain.")
    _require_positive(margin_factor, "margin_factor")
    _require_positive(beta, "beta")
    margin = margin_factor / beta
    positions = [load.position_m for load in loads]
    left = min(positions) - margin
    right = max(positions) + margin
    if right <= left:
        raise ValueError("Sleeper domain must be positive length.")
    return left, right


def _build_analysis_grid(
    beta: float,
    sample_count: int,
    domain_m: tuple[float, float] | None,
) -> tuple[list[float], float, float]:
    _require_positive(beta, "beta")
    _require_sample_count(sample_count)
    if domain_m is None:
        length = 10.0 / beta
        step = (2.0 * length) / (sample_count - 1)
        x_values = [(-length + i * step) for i in range(sample_count)]
        return x_values, -length, length
    start, end = domain_m
    if end <= start:
        raise ValueError("x_domain_m must define a positive-length domain")
    step = (end - start) / (sample_count - 1)
    x_values = [start + i * step for i in range(sample_count)]
    return x_values, start, end


def _max_abs_with_position(x_values: Sequence[float], values: Sequence[float]) -> Extremum:
    if not values or len(values) != len(x_values):
        raise ValueError("x_values and values must be non-empty and equal length")
    best_index = max(range(len(values)), key=lambda i: abs(values[i]))
    return Extremum(value=values[best_index], position_m=x_values[best_index])


def _first_derivative(x_values: Sequence[float], values: Sequence[float]) -> list[float]:
    if len(values) != len(x_values):
        raise ValueError("values must match x_values length")
    if len(values) < 2:
        return [0.0 for _ in values]
    derivative = [0.0 for _ in values]
    for i in range(1, len(values) - 1):
        derivative[i] = (values[i + 1] - values[i - 1]) / (x_values[i + 1] - x_values[i - 1])
    derivative[0] = (values[1] - values[0]) / (x_values[1] - x_values[0])
    derivative[-1] = (values[-1] - values[-2]) / (x_values[-1] - x_values[-2])
    return derivative


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_sample_count(sample_count: int) -> None:
    if sample_count < 3:
        raise ValueError("sample_count must be >= 3")

"""Adapter layer for closed-form and numerical analysis backends."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Sequence
import logging
import math

from core.analysis import (
    AnalysisInputs,
    AnalysisResult,
    AnalysisSummary,
    DesignInputs,
    DesignSummary,
    Extremum,
    build_design_summary,
    compute_track_response,
)
from core.foundation_profiles import ramp_profile, step_profile
from core.model import (
    PointLoad,
    beam_parameter_beta,
    contraflexure_distance,
    rail_base_stress,
    zero_moment_distance,
)
from core.foundation.base import (
    DampingModel,
    equivalent_series_damping,
    equivalent_series_stiffness,
    per_support_to_per_length,
    series_layer_response_per_length,
)
from core.solver import (
    build_discrete_supports,
    solve_static_beam_fdm,
    solve_static_beam_timoshenko_fdm,
    solve_two_rail_static_fdm,
)

LOGGER = logging.getLogger(__name__)


class AnalysisMode(str, Enum):
    CLOSED_FORM = "closed_form"
    NUMERICAL = "numerical"


class FoundationProfileType(str, Enum):
    UNIFORM = "uniform"
    STEP = "step"
    RAMP = "ramp"


class FoundationModelType(str, Enum):
    WINKLER = "winkler"
    SERIES = "series"
    SLEEPER_MASS = "sleeper_mass"


class BeamTheory(str, Enum):
    EULER = "euler"
    TIMOSHENKO = "timoshenko"


@dataclass(frozen=True)
class AnalysisConfig:
    """Analysis configuration shared across solver backends (SI units)."""

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
    foundation_profile_n_per_m2: Sequence[float] | None = None
    foundation_profile_type: FoundationProfileType = FoundationProfileType.UNIFORM
    foundation_profile_k1_n_per_m2: float | None = None
    foundation_profile_k2_n_per_m2: float | None = None
    foundation_profile_x_start_m: float | None = None
    foundation_profile_x_end_m: float | None = None
    pasternak_shear_n: float = 0.0
    use_discrete_supports: bool = False
    pad_stiffness_n_per_m: float | None = None
    pad_damping_n_s_per_m: float | None = None
    nodes_between_sleepers: int = 10
    use_two_rail: bool = False
    right_loads: Sequence[PointLoad] | None = None
    coupling_stiffness_n_per_m: float | None = None
    foundation_model: FoundationModelType = FoundationModelType.WINKLER
    railpad_stiffness_n_per_m: float | None = None
    railpad_damping_n_s_per_m: float | None = None
    trackbed_stiffness_n_per_m: float | None = None
    trackbed_damping_n_s_per_m: float | None = None
    foundation_damping_model: DampingModel = DampingModel.VISCOUS
    railpad_loss_factor: float | None = None
    trackbed_loss_factor: float | None = None
    pad_loss_factor: float | None = None
    sleeper_mass_kg: float | None = None
    beam_theory: BeamTheory = BeamTheory.EULER
    shear_modulus_pa: float | None = None
    shear_correction_factor: float = 0.4
    rail_area_m2: float | None = None


class ClosedFormBackend:
    """Closed-form Winkler backend (legacy)."""

    def run(self, config: AnalysisConfig) -> AnalysisResult:
        inputs = AnalysisInputs(
            loads=config.loads,
            foundation_modulus_n_per_m2=config.foundation_modulus_n_per_m2,
            elastic_modulus_pa=config.elastic_modulus_pa,
            moment_inertia_m4=config.moment_inertia_m4,
            section_modulus_m3=config.section_modulus_m3,
            sleeper_spacing_m=config.sleeper_spacing_m,
            sleeper_length_m=config.sleeper_length_m,
            sleeper_width_m=config.sleeper_width_m,
            sample_count=config.sample_count,
            x_domain_m=config.x_domain_m,
            section_modulus_head_m3=config.section_modulus_head_m3,
            section_modulus_foot_m3=config.section_modulus_foot_m3,
            area_m2=config.area_m2,
            discrete_support_stiffness_n_per_m=config.discrete_support_stiffness_n_per_m,
            design_inputs=config.design_inputs,
        )
        return compute_track_response(inputs)


class NumericalBackend:
    """Finite-difference numerical backend (Winkler/Pasternak)."""

    def run(self, config: AnalysisConfig) -> AnalysisResult:
        _require_positive(config.foundation_modulus_n_per_m2, "foundation_modulus_n_per_m2")
        _require_positive(config.elastic_modulus_pa, "elastic_modulus_pa")
        _require_positive(config.moment_inertia_m4, "moment_inertia_m4")
        _require_positive(config.section_modulus_m3, "section_modulus_m3")
        _require_positive(config.sleeper_spacing_m, "sleeper_spacing_m")
        _require_positive(config.sleeper_length_m, "sleeper_length_m")
        _require_positive(config.sleeper_width_m, "sleeper_width_m")
        _require_non_negative(config.pasternak_shear_n, "pasternak_shear_n")

        if config.beam_theory == BeamTheory.TIMOSHENKO and config.pasternak_shear_n > 0:
            raise ValueError(
                "Pasternak shear layer is not supported with Timoshenko beam theory."
            )

        support_model_name = None
        support_k_eq = None
        series_pad_per_length = None
        series_bed_per_length = None

        if config.foundation_model != FoundationModelType.WINKLER:
            if config.use_discrete_supports:
                raise ValueError(
                    "Discrete supports are not supported with multilayer foundation models."
                )
            if config.foundation_profile_type != FoundationProfileType.UNIFORM:
                raise ValueError(
                    "Nonuniform foundation profiles are not supported with multilayer models."
                )
            k_pad = _require_value(config.railpad_stiffness_n_per_m, "railpad_stiffness_n_per_m")
            k_bed = _require_value(config.trackbed_stiffness_n_per_m, "trackbed_stiffness_n_per_m")
            c_pad = config.railpad_damping_n_s_per_m or 0.0
            c_bed = config.trackbed_damping_n_s_per_m or 0.0
            series_pad_per_length = per_support_to_per_length(k_pad, config.sleeper_spacing_m)
            series_bed_per_length = per_support_to_per_length(k_bed, config.sleeper_spacing_m)
            support_k_eq = equivalent_series_stiffness(series_pad_per_length, series_bed_per_length)
            config = replace(
                config,
                foundation_modulus_n_per_m2=support_k_eq,
            )
            support_model_name = (
                "Series (railpad + trackbed)"
                if config.foundation_model == FoundationModelType.SERIES
                else "Sleeper-mass (static equivalent)"
            )

        beta = beam_parameter_beta(
            config.foundation_modulus_n_per_m2,
            config.elastic_modulus_pa,
            config.moment_inertia_m4,
        )

        x_values, domain_start, domain_end, dx = _build_grid(
            beta,
            config.sample_count,
            config.sleeper_spacing_m,
            config.use_discrete_supports or config.use_two_rail,
            config.nodes_between_sleepers,
            config.x_domain_m,
        )

        sleeper_positions = _build_sleeper_positions(
            start=domain_start,
            end=domain_end,
            spacing=config.sleeper_spacing_m,
        )

        foundation_profile = _resolve_foundation_profile(config, x_values)

        discrete_supports = None
        if config.use_discrete_supports:
            pad_stiffness = _require_value(config.pad_stiffness_n_per_m, "pad_stiffness_n_per_m")
            discrete_supports = build_discrete_supports(
                x_m=x_values,
                sleeper_positions_m=sleeper_positions,
                pad_stiffness_n_per_m=pad_stiffness,
            )

        if config.beam_theory == BeamTheory.TIMOSHENKO and config.use_two_rail:
            raise ValueError("Timoshenko beam theory is not supported for two-rail analysis.")

        if config.use_two_rail:
            coupling_stiffness = _require_value(
                config.coupling_stiffness_n_per_m,
                "coupling_stiffness_n_per_m",
            )
            coupling_nodes = _resolve_coupling_nodes(x_values, sleeper_positions, discrete_supports)
            right_loads = list(config.right_loads) if config.right_loads is not None else list(config.loads)
            solution = solve_two_rail_static_fdm(
                x_m=x_values,
                left_loads=config.loads,
                right_loads=right_loads,
                foundation_modulus_n_per_m2=foundation_profile,
                elastic_modulus_pa=config.elastic_modulus_pa,
                moment_inertia_m4=config.moment_inertia_m4,
                coupling_stiffness_n_per_m=coupling_stiffness,
                coupling_nodes=coupling_nodes,
                pasternak_shear_n=config.pasternak_shear_n,
                discrete_supports=discrete_supports,
            )

            left = solution.left
            right = solution.right
            left_sleeper_loads = _sleeper_loads_from_reaction(
                x_values,
                left.reaction_n_per_m,
                sleeper_positions,
                config.sleeper_spacing_m,
            )
            right_sleeper_loads = _sleeper_loads_from_reaction(
                x_values,
                right.reaction_n_per_m,
                sleeper_positions,
                config.sleeper_spacing_m,
            )
            total_sleeper_loads = [
                left_sleeper_loads[i] + right_sleeper_loads[i]
                for i in range(len(left_sleeper_loads))
            ]
            bearing_area = config.sleeper_length_m * config.sleeper_width_m
            sleeper_pressures = [load / bearing_area for load in total_sleeper_loads]

            summary = _build_summary_two_rail(
                x_values,
                left,
                right,
                sleeper_positions,
                total_sleeper_loads,
                sleeper_pressures,
                beta,
                config.section_modulus_m3,
                config,
            )

            _check_solution_finite(
                "left rail",
                left.deflection_m,
                left.moment_nm,
                left.shear_n,
                left.reaction_n_per_m,
            )
            _check_solution_finite(
                "right rail",
                right.deflection_m,
                right.moment_nm,
                right.shear_n,
                right.reaction_n_per_m,
            )
            _check_equilibrium("left rail", x_values, left.reaction_n_per_m, config.loads, dx)
            _check_equilibrium("right rail", x_values, right.reaction_n_per_m, right_loads, dx)

            return AnalysisResult(
                x_m=list(x_values),
                deflection_m=list(left.deflection_m),
                moment_nm=list(left.moment_nm),
                shear_n=list(left.shear_n),
                reaction_n_per_m=list(left.reaction_n_per_m),
                sleeper_positions_m=sleeper_positions,
                sleeper_loads_n=total_sleeper_loads,
                sleeper_pressures_pa=sleeper_pressures,
                summary=summary,
                slope_rad=list(left.slope_rad) if left.slope_rad is not None else None,
                rotation_rad=list(left.rotation_rad) if left.rotation_rad is not None else None,
                shear_angle_rad=list(left.shear_angle_rad) if left.shear_angle_rad is not None else None,
                winkler_reaction_n_per_m=(
                    list(left.winkler_reaction_n_per_m)
                    if left.winkler_reaction_n_per_m is not None
                    else None
                ),
                pasternak_shear_reaction_n_per_m=(
                    list(left.pasternak_shear_reaction_n_per_m)
                    if left.pasternak_shear_reaction_n_per_m is not None
                    else None
                ),
                left_deflection_m=list(left.deflection_m),
                right_deflection_m=list(right.deflection_m),
                left_moment_nm=list(left.moment_nm),
                right_moment_nm=list(right.moment_nm),
                left_shear_n=list(left.shear_n),
                right_shear_n=list(right.shear_n),
                left_reaction_n_per_m=list(left.reaction_n_per_m),
                right_reaction_n_per_m=list(right.reaction_n_per_m),
                left_sleeper_loads_n=left_sleeper_loads,
                right_sleeper_loads_n=right_sleeper_loads,
            )

        if config.beam_theory == BeamTheory.TIMOSHENKO:
            shear_modulus = _require_value(config.shear_modulus_pa, "shear_modulus_pa")
            area_m2 = _require_value(config.rail_area_m2, "rail_area_m2")
            solution = solve_static_beam_timoshenko_fdm(
                x_m=x_values,
                loads=config.loads,
                foundation_modulus_n_per_m2=foundation_profile,
                elastic_modulus_pa=config.elastic_modulus_pa,
                moment_inertia_m4=config.moment_inertia_m4,
                shear_modulus_pa=shear_modulus,
                shear_correction_factor=config.shear_correction_factor,
                area_m2=area_m2,
                discrete_supports=discrete_supports,
            )
        else:
            solution = solve_static_beam_fdm(
                x_m=x_values,
                loads=config.loads,
                foundation_modulus_n_per_m2=foundation_profile,
                elastic_modulus_pa=config.elastic_modulus_pa,
                moment_inertia_m4=config.moment_inertia_m4,
                pasternak_shear_n=config.pasternak_shear_n,
                discrete_supports=discrete_supports,
            )

        sleeper_loads = _sleeper_loads_from_reaction(
            x_values,
            solution.reaction_n_per_m,
            sleeper_positions,
            config.sleeper_spacing_m,
        )
        bearing_area = config.sleeper_length_m * config.sleeper_width_m
        sleeper_pressures = [load / bearing_area for load in sleeper_loads]

        railpad_force = None
        trackbed_force = None
        sleeper_deflection = None
        if series_pad_per_length is not None and series_bed_per_length is not None:
            railpad_force = list(solution.reaction_n_per_m)
            trackbed_force = list(solution.reaction_n_per_m)
            sleeper_deflection = [
                series_layer_response_per_length(
                    reaction_n_per_m=reaction,
                    trackbed_stiffness_n_per_m2=series_bed_per_length,
                )
                for reaction in solution.reaction_n_per_m
            ]

        summary = _build_summary(
            x_values,
            solution.deflection_m,
            solution.moment_nm,
            solution.shear_n,
            solution.reaction_n_per_m,
            sleeper_positions,
            sleeper_loads,
            sleeper_pressures,
            beta,
            config.section_modulus_m3,
            config,
            support_model=support_model_name,
            support_k_eq_n_per_m2=support_k_eq,
            max_railpad_force_n_per_m=_max_abs(railpad_force),
            max_trackbed_force_n_per_m=_max_abs(trackbed_force),
            max_sleeper_deflection_m=_max_abs(sleeper_deflection),
        )

        _check_solution_finite(
            "single rail",
            solution.deflection_m,
            solution.moment_nm,
            solution.shear_n,
            solution.reaction_n_per_m,
        )
        _check_equilibrium("single rail", x_values, solution.reaction_n_per_m, config.loads, dx)

        return AnalysisResult(
            x_m=list(x_values),
            deflection_m=list(solution.deflection_m),
            moment_nm=list(solution.moment_nm),
            shear_n=list(solution.shear_n),
            reaction_n_per_m=list(solution.reaction_n_per_m),
            sleeper_positions_m=sleeper_positions,
            sleeper_loads_n=sleeper_loads,
            sleeper_pressures_pa=sleeper_pressures,
            summary=summary,
            slope_rad=list(solution.slope_rad) if solution.slope_rad is not None else None,
            rotation_rad=list(solution.rotation_rad) if solution.rotation_rad is not None else None,
            shear_angle_rad=(
                list(solution.shear_angle_rad) if solution.shear_angle_rad is not None else None
            ),
            winkler_reaction_n_per_m=(
                list(solution.winkler_reaction_n_per_m)
                if solution.winkler_reaction_n_per_m is not None
                else None
            ),
            pasternak_shear_reaction_n_per_m=(
                list(solution.pasternak_shear_reaction_n_per_m)
                if solution.pasternak_shear_reaction_n_per_m is not None
                else None
            ),
            railpad_force_n_per_m=railpad_force,
            trackbed_force_n_per_m=trackbed_force,
            sleeper_deflection_m=sleeper_deflection,
        )


def run_analysis(config: AnalysisConfig, *, mode: AnalysisMode = AnalysisMode.CLOSED_FORM) -> AnalysisResult:
    """Run analysis using the requested backend."""
    if mode == AnalysisMode.CLOSED_FORM:
        return ClosedFormBackend().run(config)
    if mode == AnalysisMode.NUMERICAL:
        return NumericalBackend().run(config)
    raise ValueError(f"Unsupported analysis mode: {mode}")


def build_foundation_profile(
    config: AnalysisConfig,
    x_values: Sequence[float],
) -> list[float] | None:
    """Build a non-uniform foundation profile if configured."""
    if config.foundation_profile_n_per_m2 is not None:
        return list(config.foundation_profile_n_per_m2)

    profile_type = config.foundation_profile_type
    if profile_type == FoundationProfileType.UNIFORM:
        return None

    k1 = _require_value(config.foundation_profile_k1_n_per_m2, "foundation_profile_k1_n_per_m2")
    k2 = _require_value(config.foundation_profile_k2_n_per_m2, "foundation_profile_k2_n_per_m2")
    x_start = _require_value(config.foundation_profile_x_start_m, "foundation_profile_x_start_m")
    x_end = config.foundation_profile_x_end_m

    if profile_type == FoundationProfileType.STEP:
        return step_profile(
            x_m=x_values,
            left_modulus_n_per_m2=k1,
            right_modulus_n_per_m2=k2,
            step_location_m=x_start,
        )
    if profile_type == FoundationProfileType.RAMP:
        ramp_end = _require_value(x_end, "foundation_profile_x_end_m")
        return ramp_profile(
            x_m=x_values,
            start_modulus_n_per_m2=k1,
            end_modulus_n_per_m2=k2,
            ramp_start_m=x_start,
            ramp_end_m=ramp_end,
        )

    raise ValueError(f"Unsupported foundation profile type: {profile_type}")


def _resolve_foundation_profile(config: AnalysisConfig, x_values: Sequence[float]) -> float | list[float]:
    profile = build_foundation_profile(config, x_values)
    if profile is None:
        return config.foundation_modulus_n_per_m2
    if len(profile) != len(x_values):
        raise ValueError("foundation_profile_n_per_m2 must match x_m length")
    return profile


def _build_grid(
    beta: float,
    sample_count: int,
    sleeper_spacing_m: float,
    align_to_sleepers: bool,
    nodes_between_sleepers: int,
    domain_m: tuple[float, float] | None,
) -> tuple[list[float], float, float, float]:
    _require_positive(beta, "beta")
    if align_to_sleepers:
        _require_positive(sleeper_spacing_m, "sleeper_spacing_m")
        if nodes_between_sleepers < 2:
            raise ValueError("nodes_between_sleepers must be >= 2")
        dx = sleeper_spacing_m / nodes_between_sleepers
        if domain_m is None:
            length = 10.0 / beta
            half_nodes = int(math.floor(length / dx))
            if half_nodes < 2:
                raise ValueError("grid length too short for discrete supports")
            actual_length = half_nodes * dx
            x_values = [(-actual_length + i * dx) for i in range(2 * half_nodes + 1)]
            return x_values, -actual_length, actual_length, dx
        start, end = domain_m
        if end <= start:
            raise ValueError("x_domain_m must define a positive-length domain")
        start = math.floor(start / dx) * dx
        end = math.ceil(end / dx) * dx
        if end - start < 2 * dx:
            raise ValueError("grid length too short for discrete supports")
        node_count = int(round((end - start) / dx))
        x_values = [start + i * dx for i in range(node_count + 1)]
        return x_values, start, end, dx

    _require_sample_count(sample_count)
    if domain_m is None:
        length = 10.0 / beta
        step = (2.0 * length) / (sample_count - 1)
        x_values = [(-length + i * step) for i in range(sample_count)]
        return x_values, -length, length, step
    start, end = domain_m
    if end <= start:
        raise ValueError("x_domain_m must define a positive-length domain")
    step = (end - start) / (sample_count - 1)
    x_values = [start + i * step for i in range(sample_count)]
    return x_values, start, end, step


def _build_sleeper_positions(start: float, end: float, spacing: float) -> list[float]:
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    count = int(math.floor((end - start) / spacing))
    return [start + i * spacing for i in range(count + 1)]


def _resolve_coupling_nodes(
    x_values: Sequence[float],
    sleeper_positions: Sequence[float],
    discrete_supports: dict[int, float] | None,
) -> list[int]:
    if discrete_supports:
        return sorted(discrete_supports.keys())
    return _map_positions_to_nodes(x_values, sleeper_positions)


def _map_positions_to_nodes(x_values: Sequence[float], positions: Sequence[float]) -> list[int]:
    dx = x_values[1] - x_values[0]
    nodes: list[int] = []
    for position in positions:
        index = min(range(len(x_values)), key=lambda i: abs(x_values[i] - position))
        if abs(x_values[index] - position) > 0.51 * dx:
            raise ValueError("sleeper position must align with grid spacing")
        nodes.append(index)
    return nodes


def _sleeper_loads_from_reaction(
    x_values: Sequence[float],
    reaction_n_per_m: Sequence[float],
    sleeper_positions_m: Sequence[float],
    spacing_m: float,
) -> list[float]:
    half = 0.5 * spacing_m
    return [
        _integrate_segment(
            x_values,
            reaction_n_per_m,
            position - half,
            position + half,
        )
        for position in sleeper_positions_m
    ]


def _integrate_segment(
    x_values: Sequence[float],
    y_values: Sequence[float],
    start: float,
    end: float,
) -> float:
    if end <= start:
        return 0.0
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must be the same length")

    x_min = x_values[0]
    x_max = x_values[-1]
    segment_start = max(start, x_min)
    segment_end = min(end, x_max)
    if segment_end <= segment_start:
        return 0.0

    total = 0.0
    for i in range(len(x_values) - 1):
        x0 = x_values[i]
        x1 = x_values[i + 1]
        if x1 <= segment_start or x0 >= segment_end:
            continue
        seg_start = max(segment_start, x0)
        seg_end = min(segment_end, x1)
        if seg_end <= seg_start:
            continue
        y0 = _linear_interpolate(x0, x1, y_values[i], y_values[i + 1], seg_start)
        y1 = _linear_interpolate(x0, x1, y_values[i], y_values[i + 1], seg_end)
        total += 0.5 * (y0 + y1) * (seg_end - seg_start)
    return total


def _linear_interpolate(x0: float, x1: float, y0: float, y1: float, x: float) -> float:
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _build_summary(
    x_values: Sequence[float],
    deflections: Sequence[float],
    moments: Sequence[float],
    shears: Sequence[float],
    reactions: Sequence[float],
    sleeper_positions: Sequence[float],
    sleeper_loads: Sequence[float],
    sleeper_pressures: Sequence[float],
    beta: float,
    section_modulus_m3: float,
    config: AnalysisConfig,
    support_model: str | None = None,
    support_k_eq_n_per_m2: float | None = None,
    max_railpad_force_n_per_m: float | None = None,
    max_trackbed_force_n_per_m: float | None = None,
    max_sleeper_deflection_m: float | None = None,
) -> AnalysisSummary:
    max_deflection = _max_abs_with_position(x_values, deflections)
    max_moment = _max_abs_with_position(x_values, moments)
    max_shear = _max_abs_with_position(x_values, shears)
    max_reaction = _max_abs_with_position(x_values, reactions)
    max_sleeper_load = _max_abs_with_position(sleeper_positions, sleeper_loads)
    max_sleeper_pressure = _max_abs_with_position(sleeper_positions, sleeper_pressures)
    max_stress = abs(rail_base_stress(max_moment.value, section_modulus_m3))

    design_summary = _build_design_summary(config, beta, max_moment.value)

    return AnalysisSummary(
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
        support_model=support_model,
        support_k_eq_n_per_m2=support_k_eq_n_per_m2,
        max_railpad_force_n_per_m=max_railpad_force_n_per_m,
        max_trackbed_force_n_per_m=max_trackbed_force_n_per_m,
        max_sleeper_deflection_m=max_sleeper_deflection_m,
    )


def _max_abs(values: Sequence[float] | None) -> float | None:
    if values is None:
        return None
    if not values:
        return None
    return max(abs(value) for value in values)


def _build_summary_two_rail(
    x_values: Sequence[float],
    left: "StaticBeamSolution",
    right: "StaticBeamSolution",
    sleeper_positions: Sequence[float],
    sleeper_loads: Sequence[float],
    sleeper_pressures: Sequence[float],
    beta: float,
    section_modulus_m3: float,
    config: AnalysisConfig,
) -> AnalysisSummary:
    max_deflection = _pick_extremum(
        _max_abs_with_position(x_values, left.deflection_m),
        _max_abs_with_position(x_values, right.deflection_m),
    )
    max_moment = _pick_extremum(
        _max_abs_with_position(x_values, left.moment_nm),
        _max_abs_with_position(x_values, right.moment_nm),
    )
    max_shear = _pick_extremum(
        _max_abs_with_position(x_values, left.shear_n),
        _max_abs_with_position(x_values, right.shear_n),
    )
    max_reaction = _pick_extremum(
        _max_abs_with_position(x_values, left.reaction_n_per_m),
        _max_abs_with_position(x_values, right.reaction_n_per_m),
    )
    max_sleeper_load = _max_abs_with_position(sleeper_positions, sleeper_loads)
    max_sleeper_pressure = _max_abs_with_position(sleeper_positions, sleeper_pressures)
    max_stress = abs(rail_base_stress(max_moment.value, section_modulus_m3))

    design_summary = _build_design_summary(config, beta, max_moment.value)

    return AnalysisSummary(
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


def _build_design_summary(
    config: AnalysisConfig,
    beta: float,
    max_moment_nm: float | None = None,
) -> DesignSummary | None:
    if config.design_inputs is None:
        return None
    inputs = AnalysisInputs(
        loads=config.loads,
        foundation_modulus_n_per_m2=config.foundation_modulus_n_per_m2,
        elastic_modulus_pa=config.elastic_modulus_pa,
        moment_inertia_m4=config.moment_inertia_m4,
        section_modulus_m3=config.section_modulus_m3,
        sleeper_spacing_m=config.sleeper_spacing_m,
        sleeper_length_m=config.sleeper_length_m,
        sleeper_width_m=config.sleeper_width_m,
        sample_count=config.sample_count,
        x_domain_m=config.x_domain_m,
        section_modulus_head_m3=config.section_modulus_head_m3,
        section_modulus_foot_m3=config.section_modulus_foot_m3,
        area_m2=config.area_m2,
        discrete_support_stiffness_n_per_m=config.discrete_support_stiffness_n_per_m,
        design_inputs=config.design_inputs,
    )
    return build_design_summary(inputs, beta, max_moment_nm=max_moment_nm)


def _max_abs_with_position(x_values: Sequence[float], values: Sequence[float]) -> Extremum:
    if not values or len(values) != len(x_values):
        raise ValueError("x_values and values must be non-empty and equal length")
    best_index = max(range(len(values)), key=lambda i: abs(values[i]))
    return Extremum(value=values[best_index], position_m=x_values[best_index])


def _pick_extremum(left: Extremum, right: Extremum) -> Extremum:
    return left if abs(left.value) >= abs(right.value) else right


def _check_solution_finite(
    label: str,
    deflections: Iterable[float],
    moments: Iterable[float],
    shears: Iterable[float],
    reactions: Iterable[float],
) -> None:
    series = {
        "deflection": deflections,
        "moment": moments,
        "shear": shears,
        "reaction": reactions,
    }
    for name, values in series.items():
        if any(not math.isfinite(value) for value in values):
            LOGGER.warning("%s output contains non-finite %s values.", label, name)


def _check_equilibrium(
    label: str,
    x_values: Sequence[float],
    reaction_n_per_m: Sequence[float],
    loads: Sequence[PointLoad],
    dx: float,
) -> None:
    if not loads:
        return
    total_load = sum(load.load_newtons for load in loads)
    if math.isclose(total_load, 0.0, abs_tol=1e-12):
        return
    total_reaction = _integrate_segment(x_values, reaction_n_per_m, x_values[0], x_values[-1])
    error = abs(total_reaction - total_load) / abs(total_load)
    if error > 0.05:
        LOGGER.warning(
            "%s equilibrium check deviates by %.1f%% (reaction=%.3f N, load=%.3f N).",
            label,
            error * 100.0,
            total_reaction,
            total_load,
        )


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_sample_count(sample_count: int) -> None:
    if sample_count < 5:
        raise ValueError("sample_count must be >= 5")


def _require_value(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    return value

"""Quasi-static moving-load envelope analysis for BOEF."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import math
from typing import Callable, Iterable, Sequence

from core.analysis import AnalysisInputs, AnalysisResult, DesignSummary, Extremum, build_design_summary
from core.analysis_engine import AnalysisConfig, AnalysisMode, run_analysis
from core.load_builder import (
    AS5100RailLoadConfig,
    as5100_load_metadata,
    build_as5100_group_spacing_candidates,
    build_as5100_rail_loads,
)
from core.model import (
    PointLoad,
    beam_parameter_beta,
    contraflexure_distance,
    deflection_at,
    moment_at,
    reaction_at,
    shear_at,
    zero_moment_distance,
    sleeper_seat_loads,
    rail_base_stress,
)

LOGGER = logging.getLogger(__name__)


class EnvelopeCancelled(RuntimeError):
    """Raised when the envelope analysis is cancelled."""


@dataclass(frozen=True)
class EnvelopeConfig:
    """Inputs for quasi-static envelope analysis (SI units)."""

    analysis_config: AnalysisConfig
    x_ref_start_m: float
    x_ref_end_m: float
    x_ref_step_m: float
    x_domain_m: tuple[float, float]
    bearing_width_m: float
    bearing_length_m: float
    depth_m: Sequence[float]
    rail_count: int = 2
    mode: AnalysisMode = AnalysisMode.CLOSED_FORM
    as5100_sweep: "AS5100EnvelopeSweep | None" = None
    run_metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class AS5100EnvelopeSweep:
    """Compact AS5100 arrangement sweep definition for envelope runs."""

    model: str
    selected_group_count: int
    selected_group_spacing_m: float
    reference_position_m: float = 0.0

    @property
    def group_count_candidates(self) -> tuple[int, ...]:
        if self.selected_group_count <= 1:
            return (1,)
        return tuple(range(1, self.selected_group_count + 1))

    @property
    def group_spacing_candidates_m(self) -> tuple[float, ...]:
        return tuple(build_as5100_group_spacing_candidates(self.selected_group_spacing_m))


@dataclass(frozen=True)
class EnvelopeSummary:
    """Envelope summary values (SI units)."""

    beta_per_m: float
    zero_moment_distance_m: float
    contraflexure_distance_m: float
    max_deflection: Extremum
    max_moment: Extremum
    max_shear: Extremum
    max_reaction: Extremum
    max_sleeper_load: Extremum
    max_ballast_pressure: Extremum
    max_rail_base_stress_pa: float
    max_formation_stress_by_depth_pa: dict[float, float]
    design_summary: DesignSummary | None = None


@dataclass(frozen=True)
class EnvelopeResult:
    """Envelope results (SI units)."""

    x_m: list[float]
    deflection_max_m: list[float]
    deflection_min_m: list[float]
    moment_max_nm: list[float]
    moment_min_nm: list[float]
    shear_max_n: list[float]
    shear_min_n: list[float]
    reaction_max_n_per_m: list[float]
    reaction_min_n_per_m: list[float]
    sleeper_positions_m: list[float]
    sleeper_loads_max_n: list[float]
    sleeper_loads_min_n: list[float]
    ballast_pressure_max_pa: list[float]
    ballast_pressure_min_pa: list[float]
    formation_stress_max_pa_by_depth: dict[float, list[float]]
    formation_stress_min_pa_by_depth: dict[float, list[float]]
    summary: EnvelopeSummary
    left_deflection_max_m: list[float] | None = None
    left_deflection_min_m: list[float] | None = None
    right_deflection_max_m: list[float] | None = None
    right_deflection_min_m: list[float] | None = None
    left_moment_max_nm: list[float] | None = None
    left_moment_min_nm: list[float] | None = None
    right_moment_max_nm: list[float] | None = None
    right_moment_min_nm: list[float] | None = None
    left_shear_max_n: list[float] | None = None
    left_shear_min_n: list[float] | None = None
    right_shear_max_n: list[float] | None = None
    right_shear_min_n: list[float] | None = None
    left_reaction_max_n_per_m: list[float] | None = None
    left_reaction_min_n_per_m: list[float] | None = None
    right_reaction_max_n_per_m: list[float] | None = None
    right_reaction_min_n_per_m: list[float] | None = None
    left_sleeper_loads_max_n: list[float] | None = None
    left_sleeper_loads_min_n: list[float] | None = None
    right_sleeper_loads_max_n: list[float] | None = None
    right_sleeper_loads_min_n: list[float] | None = None
    run_metadata: dict[str, object] | None = None

    @property
    def deflection_abs_max_m(self) -> list[float]:
        return _max_abs_envelope(self.deflection_max_m, self.deflection_min_m)

    @property
    def moment_abs_max_nm(self) -> list[float]:
        return _max_abs_envelope(self.moment_max_nm, self.moment_min_nm)

    @property
    def shear_abs_max_n(self) -> list[float]:
        return _max_abs_envelope(self.shear_max_n, self.shear_min_n)

    @property
    def reaction_abs_max_n_per_m(self) -> list[float]:
        return _max_abs_envelope(self.reaction_max_n_per_m, self.reaction_min_n_per_m)


def run_envelope(
    config: EnvelopeConfig,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> EnvelopeResult:
    """Run a quasi-static moving-load envelope analysis."""
    if config.as5100_sweep is not None:
        return _run_as5100_envelope_sweep(
            config,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
    _require_positive(config.x_ref_step_m, "x_ref_step_m")
    if config.x_ref_end_m < config.x_ref_start_m:
        raise ValueError("x_ref_end_m must be greater than or equal to x_ref_start_m")
    if config.bearing_width_m <= 0 or config.bearing_length_m <= 0:
        raise ValueError("bearing dimensions must be positive")
    if config.rail_count < 1:
        raise ValueError("rail_count must be >= 1")
    if not config.depth_m:
        raise ValueError("depth_m must include at least one depth value")
    if any(depth <= 0 for depth in config.depth_m):
        raise ValueError("depth_m values must be positive")

    base = config.analysis_config
    if config.mode == AnalysisMode.CLOSED_FORM and base.use_two_rail:
        raise ValueError("Two-rail coupling is not supported for closed-form envelope analysis.")
    _require_loads(base.loads)

    x_values: list[float] | None = None
    sleeper_positions: list[float] | None = None
    if config.mode == AnalysisMode.CLOSED_FORM:
        x_values = _build_grid(base.sample_count, config.x_domain_m)
        sleeper_positions = _build_sleeper_positions(
            start=config.x_domain_m[0],
            end=config.x_domain_m[1],
            spacing=base.sleeper_spacing_m,
        )

    step_positions = _build_ref_positions(
        config.x_ref_start_m,
        config.x_ref_end_m,
        config.x_ref_step_m,
    )

    deflection_max: list[float] | None = None
    deflection_min: list[float] | None = None
    moment_max: list[float] | None = None
    moment_min: list[float] | None = None
    shear_max: list[float] | None = None
    shear_min: list[float] | None = None
    reaction_max: list[float] | None = None
    reaction_min: list[float] | None = None

    sleeper_max: list[float] | None = None
    sleeper_min: list[float] | None = None
    ballast_max: list[float] | None = None
    ballast_min: list[float] | None = None

    formation_max: dict[float, list[float]] | None = None
    formation_min: dict[float, list[float]] | None = None

    left_deflection_max, left_deflection_min = None, None
    right_deflection_max, right_deflection_min = None, None
    left_moment_max, left_moment_min = None, None
    right_moment_max, right_moment_min = None, None
    left_shear_max, left_shear_min = None, None
    right_shear_max, right_shear_min = None, None
    left_reaction_max, left_reaction_min = None, None
    right_reaction_max, right_reaction_min = None, None
    left_sleeper_max, left_sleeper_min = None, None
    right_sleeper_max, right_sleeper_min = None, None

    total_steps = len(step_positions)
    for index, x_ref in enumerate(step_positions):
        if cancel_callback is not None and cancel_callback():
            raise EnvelopeCancelled("Envelope analysis cancelled.")
        shifted_loads = _shift_loads(base.loads, x_ref)
        if config.mode == AnalysisMode.CLOSED_FORM:
            deflections = [
                deflection_at(
                    x,
                    shifted_loads,
                    base.foundation_modulus_n_per_m2,
                    base.elastic_modulus_pa,
                    base.moment_inertia_m4,
                )
                for x in x_values
            ]
            moments = [
                moment_at(
                    x,
                    shifted_loads,
                    base.foundation_modulus_n_per_m2,
                    base.elastic_modulus_pa,
                    base.moment_inertia_m4,
                )
                for x in x_values
            ]
            shears = [
                shear_at(
                    x,
                    shifted_loads,
                    base.foundation_modulus_n_per_m2,
                    base.elastic_modulus_pa,
                    base.moment_inertia_m4,
                )
                for x in x_values
            ]
            reactions = [
                reaction_at(
                    x,
                    shifted_loads,
                    base.foundation_modulus_n_per_m2,
                    base.elastic_modulus_pa,
                    base.moment_inertia_m4,
                )
                for x in x_values
            ]
            sleeper_loads = sleeper_seat_loads(
                sleeper_positions,
                base.sleeper_spacing_m,
                shifted_loads,
                base.foundation_modulus_n_per_m2,
                base.elastic_modulus_pa,
                base.moment_inertia_m4,
            )
            left_deflections = None
            right_deflections = None
            left_moments = None
            right_moments = None
            left_shears = None
            right_shears = None
            left_reactions = None
            right_reactions = None
            left_sleeper_loads = None
            right_sleeper_loads = None
        else:
            right_loads = None
            if base.right_loads is not None:
                right_loads = _shift_loads(base.right_loads, x_ref)
            step_config = replace(
                base,
                loads=shifted_loads,
                right_loads=right_loads,
            )
            result = run_analysis(step_config, mode=config.mode)
            cropped = _crop_analysis_result_to_domain(result, config.x_domain_m)
            deflections = cropped["deflection_m"]
            moments = cropped["moment_nm"]
            shears = cropped["shear_n"]
            reactions = cropped["reaction_n_per_m"]
            sleeper_loads = cropped["sleeper_loads_n"]
            left_deflections = cropped["left_deflection_m"]
            right_deflections = cropped["right_deflection_m"]
            left_moments = cropped["left_moment_nm"]
            right_moments = cropped["right_moment_nm"]
            left_shears = cropped["left_shear_n"]
            right_shears = cropped["right_shear_n"]
            left_reactions = cropped["left_reaction_n_per_m"]
            right_reactions = cropped["right_reaction_n_per_m"]
            left_sleeper_loads = cropped["left_sleeper_loads_n"]
            right_sleeper_loads = cropped["right_sleeper_loads_n"]

            if x_values is None:
                x_values = cropped["x_m"]
            if sleeper_positions is None:
                sleeper_positions = cropped["sleeper_positions_m"]
            if cropped["x_m"] != x_values:
                raise ValueError("Envelope analysis requires a consistent x grid across steps.")
            if cropped["sleeper_positions_m"] != sleeper_positions:
                raise ValueError("Envelope analysis requires consistent sleeper positions across steps.")

        if x_values is None or sleeper_positions is None:
            raise ValueError("Envelope analysis failed to initialize spatial grids.")
        if deflection_max is None:
            deflection_max = _init_extrema(len(x_values), -math.inf)
            deflection_min = _init_extrema(len(x_values), math.inf)
            moment_max = _init_extrema(len(x_values), -math.inf)
            moment_min = _init_extrema(len(x_values), math.inf)
            shear_max = _init_extrema(len(x_values), -math.inf)
            shear_min = _init_extrema(len(x_values), math.inf)
            reaction_max = _init_extrema(len(x_values), -math.inf)
            reaction_min = _init_extrema(len(x_values), math.inf)
            sleeper_max = _init_extrema(len(sleeper_positions), -math.inf)
            sleeper_min = _init_extrema(len(sleeper_positions), math.inf)
            ballast_max = _init_extrema(len(sleeper_positions), -math.inf)
            ballast_min = _init_extrema(len(sleeper_positions), math.inf)
            formation_max = {
                depth: _init_extrema(len(sleeper_positions), -math.inf) for depth in config.depth_m
            }
            formation_min = {
                depth: _init_extrema(len(sleeper_positions), math.inf) for depth in config.depth_m
            }

        _update_extrema(deflection_max, deflection_min, deflections)
        _update_extrema(moment_max, moment_min, moments)
        _update_extrema(shear_max, shear_min, shears)
        _update_extrema(reaction_max, reaction_min, reactions)

        total_sleeper_loads = _scale_sleeper_loads(
            sleeper_loads,
            left_sleeper_loads,
            right_sleeper_loads,
            config.rail_count,
        )
        _update_extrema(sleeper_max, sleeper_min, total_sleeper_loads)

        ballast_pressures = [
            load / (config.bearing_width_m * config.bearing_length_m)
            for load in total_sleeper_loads
        ]
        _update_extrema(ballast_max, ballast_min, ballast_pressures)

        for depth in config.depth_m:
            stresses = [
                _formation_stress_2to1(
                    pressure,
                    config.bearing_width_m,
                    config.bearing_length_m,
                    depth,
                )
                for pressure in ballast_pressures
            ]
            if formation_max is None or formation_min is None:
                raise ValueError("Envelope formation buffers were not initialized.")
            _update_extrema(formation_max[depth], formation_min[depth], stresses)

        if left_deflections is not None and right_deflections is not None:
            if left_deflection_max is None:
                left_deflection_max = _init_extrema(len(left_deflections), -math.inf)
                left_deflection_min = _init_extrema(len(left_deflections), math.inf)
                right_deflection_max = _init_extrema(len(right_deflections), -math.inf)
                right_deflection_min = _init_extrema(len(right_deflections), math.inf)
                left_moment_max = _init_extrema(len(left_deflections), -math.inf)
                left_moment_min = _init_extrema(len(left_deflections), math.inf)
                right_moment_max = _init_extrema(len(right_deflections), -math.inf)
                right_moment_min = _init_extrema(len(right_deflections), math.inf)
                left_shear_max = _init_extrema(len(left_deflections), -math.inf)
                left_shear_min = _init_extrema(len(left_deflections), math.inf)
                right_shear_max = _init_extrema(len(right_deflections), -math.inf)
                right_shear_min = _init_extrema(len(right_deflections), math.inf)
                left_reaction_max = _init_extrema(len(left_deflections), -math.inf)
                left_reaction_min = _init_extrema(len(left_deflections), math.inf)
                right_reaction_max = _init_extrema(len(right_deflections), -math.inf)
                right_reaction_min = _init_extrema(len(right_deflections), math.inf)
            _update_extrema(left_deflection_max, left_deflection_min, left_deflections)
            _update_extrema(right_deflection_max, right_deflection_min, right_deflections)
            if left_moments is not None and right_moments is not None:
                _update_extrema(left_moment_max, left_moment_min, left_moments)
                _update_extrema(right_moment_max, right_moment_min, right_moments)
            if left_shears is not None and right_shears is not None:
                _update_extrema(left_shear_max, left_shear_min, left_shears)
                _update_extrema(right_shear_max, right_shear_min, right_shears)
            if left_reactions is not None and right_reactions is not None:
                _update_extrema(left_reaction_max, left_reaction_min, left_reactions)
                _update_extrema(right_reaction_max, right_reaction_min, right_reactions)
            if left_sleeper_loads is not None and right_sleeper_loads is not None:
                if left_sleeper_max is None:
                    left_sleeper_max = _init_extrema(len(left_sleeper_loads), -math.inf)
                    left_sleeper_min = _init_extrema(len(left_sleeper_loads), math.inf)
                    right_sleeper_max = _init_extrema(len(right_sleeper_loads), -math.inf)
                    right_sleeper_min = _init_extrema(len(right_sleeper_loads), math.inf)
                _update_extrema(left_sleeper_max, left_sleeper_min, left_sleeper_loads)
                _update_extrema(right_sleeper_max, right_sleeper_min, right_sleeper_loads)

        if progress_callback is not None:
            progress_callback(index + 1, total_steps)
        elif index % 200 == 0 and index > 0:
            LOGGER.info("Envelope progress: %d / %d", index, total_steps)

    if (
        x_values is None
        or sleeper_positions is None
        or deflection_max is None
        or deflection_min is None
        or moment_max is None
        or moment_min is None
        or shear_max is None
        or shear_min is None
        or reaction_max is None
        or reaction_min is None
        or sleeper_max is None
        or sleeper_min is None
        or ballast_max is None
        or ballast_min is None
        or formation_max is None
        or formation_min is None
    ):
        raise ValueError("Envelope analysis did not produce complete results.")

    summary = _build_envelope_summary(
        config,
        x_values,
        deflection_max,
        deflection_min,
        moment_max,
        moment_min,
        shear_max,
        shear_min,
        reaction_max,
        reaction_min,
        sleeper_positions,
        sleeper_max,
        sleeper_min,
        ballast_max,
        ballast_min,
        formation_max,
    )

    return EnvelopeResult(
        x_m=list(x_values),
        deflection_max_m=deflection_max,
        deflection_min_m=deflection_min,
        moment_max_nm=moment_max,
        moment_min_nm=moment_min,
        shear_max_n=shear_max,
        shear_min_n=shear_min,
        reaction_max_n_per_m=reaction_max,
        reaction_min_n_per_m=reaction_min,
        sleeper_positions_m=sleeper_positions,
        sleeper_loads_max_n=sleeper_max,
        sleeper_loads_min_n=sleeper_min,
        ballast_pressure_max_pa=ballast_max,
        ballast_pressure_min_pa=ballast_min,
        formation_stress_max_pa_by_depth=formation_max,
        formation_stress_min_pa_by_depth=formation_min,
        summary=summary,
        left_deflection_max_m=left_deflection_max,
        left_deflection_min_m=left_deflection_min,
        right_deflection_max_m=right_deflection_max,
        right_deflection_min_m=right_deflection_min,
        left_moment_max_nm=left_moment_max,
        left_moment_min_nm=left_moment_min,
        right_moment_max_nm=right_moment_max,
        right_moment_min_nm=right_moment_min,
        left_shear_max_n=left_shear_max,
        left_shear_min_n=left_shear_min,
        right_shear_max_n=right_shear_max,
        right_shear_min_n=right_shear_min,
        left_reaction_max_n_per_m=left_reaction_max,
        left_reaction_min_n_per_m=left_reaction_min,
        right_reaction_max_n_per_m=right_reaction_max,
        right_reaction_min_n_per_m=right_reaction_min,
        left_sleeper_loads_max_n=left_sleeper_max,
        left_sleeper_loads_min_n=left_sleeper_min,
        right_sleeper_loads_max_n=right_sleeper_max,
        right_sleeper_loads_min_n=right_sleeper_min,
        run_metadata=dict(config.run_metadata) if config.run_metadata is not None else None,
    )


def _run_as5100_envelope_sweep(
    config: EnvelopeConfig,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> EnvelopeResult:
    sweep = config.as5100_sweep
    if sweep is None:
        raise ValueError("AS5100 sweep config is missing.")

    arrangements = [
        AS5100RailLoadConfig(
            model=sweep.model,
            group_count=group_count,
            group_spacing_m=group_spacing_m,
            reference_position_m=sweep.reference_position_m,
        )
        for group_count in sweep.group_count_candidates
        for group_spacing_m in sweep.group_spacing_candidates_m
    ]
    if not arrangements:
        raise ValueError("AS5100 sweep did not define any candidate arrangements.")

    step_positions = _build_ref_positions(
        config.x_ref_start_m,
        config.x_ref_end_m,
        config.x_ref_step_m,
    )
    steps_per_run = max(1, len(step_positions))
    total_progress_steps = steps_per_run * len(arrangements)
    governing_result: EnvelopeResult | None = None
    governing_score: tuple[float, float, float, float] | None = None
    governing_arrangement: AS5100RailLoadConfig | None = None
    governing_loads: list[PointLoad] | None = None
    candidate_summaries: list[dict[str, object]] = []

    for arrangement_index, arrangement in enumerate(arrangements):
        if cancel_callback is not None and cancel_callback():
            raise EnvelopeCancelled("Envelope analysis cancelled.")
        loads = build_as5100_rail_loads(arrangement)
        child_config = replace(
            config,
            analysis_config=replace(config.analysis_config, loads=loads),
            as5100_sweep=None,
            run_metadata=as5100_load_metadata(arrangement, loads=loads),
        )

        def child_progress(current: int, _total: int) -> None:
            if progress_callback is None:
                return
            completed = arrangement_index * steps_per_run + current
            progress_callback(min(completed, total_progress_steps), total_progress_steps)

        result = run_envelope(
            child_config,
            progress_callback=child_progress if progress_callback is not None else None,
            cancel_callback=cancel_callback,
        )
        score = _envelope_governing_score(result)
        candidate_summaries.append(
            {
                "group_count": arrangement.group_count,
                "group_spacing_m": arrangement.group_spacing_m,
                "axle_count": len(loads),
                "max_abs_moment_nm": abs(result.summary.max_moment.value),
                "max_abs_deflection_m": abs(result.summary.max_deflection.value),
                "max_abs_reaction_n_per_m": abs(result.summary.max_reaction.value),
                "max_abs_shear_n": abs(result.summary.max_shear.value),
            }
        )
        if governing_score is None or score > governing_score:
            governing_result = result
            governing_score = score
            governing_arrangement = arrangement
            governing_loads = loads

    if progress_callback is not None:
        progress_callback(total_progress_steps, total_progress_steps)
    if governing_result is None or governing_arrangement is None or governing_loads is None:
        raise ValueError("AS5100 governing sweep did not produce an envelope result.")

    run_metadata = as5100_load_metadata(
        governing_arrangement,
        loads=governing_loads,
        arrangement="governing_envelope_sweep",
        extra={
            "selected_group_count": sweep.selected_group_count,
            "selected_group_spacing_m": sweep.selected_group_spacing_m,
            "sweep_group_count_candidates": list(sweep.group_count_candidates),
            "sweep_group_spacing_candidates_m": list(sweep.group_spacing_candidates_m),
            "sweep_candidate_count": len(arrangements),
            "governing_metric": "max_abs_moment_nm",
            "candidate_summaries": candidate_summaries,
        },
    )
    return replace(governing_result, run_metadata=run_metadata)


def _crop_analysis_result_to_domain(
    result: AnalysisResult,
    domain_m: tuple[float, float],
) -> dict[str, list[float] | None]:
    x_indices = _indices_in_domain(result.x_m, domain_m, "x_m")
    sleeper_indices = _indices_in_domain(result.sleeper_positions_m, domain_m, "sleeper_positions_m")
    return {
        "x_m": _take_indices(result.x_m, x_indices),
        "deflection_m": _take_indices(result.deflection_m, x_indices),
        "moment_nm": _take_indices(result.moment_nm, x_indices),
        "shear_n": _take_indices(result.shear_n, x_indices),
        "reaction_n_per_m": _take_indices(result.reaction_n_per_m, x_indices),
        "sleeper_positions_m": _take_indices(result.sleeper_positions_m, sleeper_indices),
        "sleeper_loads_n": _take_indices(result.sleeper_loads_n, sleeper_indices),
        "left_deflection_m": _take_optional_indices(result.left_deflection_m, x_indices),
        "right_deflection_m": _take_optional_indices(result.right_deflection_m, x_indices),
        "left_moment_nm": _take_optional_indices(result.left_moment_nm, x_indices),
        "right_moment_nm": _take_optional_indices(result.right_moment_nm, x_indices),
        "left_shear_n": _take_optional_indices(result.left_shear_n, x_indices),
        "right_shear_n": _take_optional_indices(result.right_shear_n, x_indices),
        "left_reaction_n_per_m": _take_optional_indices(result.left_reaction_n_per_m, x_indices),
        "right_reaction_n_per_m": _take_optional_indices(result.right_reaction_n_per_m, x_indices),
        "left_sleeper_loads_n": _take_optional_indices(result.left_sleeper_loads_n, sleeper_indices),
        "right_sleeper_loads_n": _take_optional_indices(result.right_sleeper_loads_n, sleeper_indices),
    }


def _build_envelope_summary(
    config: EnvelopeConfig,
    x_values: Sequence[float],
    deflection_max: Sequence[float],
    deflection_min: Sequence[float],
    moment_max: Sequence[float],
    moment_min: Sequence[float],
    shear_max: Sequence[float],
    shear_min: Sequence[float],
    reaction_max: Sequence[float],
    reaction_min: Sequence[float],
    sleeper_positions: Sequence[float],
    sleeper_max: Sequence[float],
    sleeper_min: Sequence[float],
    ballast_max: Sequence[float],
    ballast_min: Sequence[float],
    formation_max: dict[float, Sequence[float]],
) -> EnvelopeSummary:
    max_deflection = _max_abs_with_position(x_values, deflection_max, deflection_min)
    max_moment = _max_abs_with_position(x_values, moment_max, moment_min)
    max_shear = _max_abs_with_position(x_values, shear_max, shear_min)
    max_reaction = _max_abs_with_position(x_values, reaction_max, reaction_min)
    max_sleeper_load = _max_abs_with_position(sleeper_positions, sleeper_max, sleeper_min)
    max_ballast = _max_abs_with_position(sleeper_positions, ballast_max, ballast_min)
    max_stress = abs(rail_base_stress(max_moment.value, config.analysis_config.section_modulus_m3))

    inputs = AnalysisInputs(
        loads=config.analysis_config.loads,
        foundation_modulus_n_per_m2=config.analysis_config.foundation_modulus_n_per_m2,
        elastic_modulus_pa=config.analysis_config.elastic_modulus_pa,
        moment_inertia_m4=config.analysis_config.moment_inertia_m4,
        section_modulus_m3=config.analysis_config.section_modulus_m3,
        sleeper_spacing_m=config.analysis_config.sleeper_spacing_m,
        sleeper_length_m=config.analysis_config.sleeper_length_m,
        sleeper_width_m=config.analysis_config.sleeper_width_m,
        sample_count=config.analysis_config.sample_count,
        x_domain_m=config.analysis_config.x_domain_m,
        section_modulus_head_m3=config.analysis_config.section_modulus_head_m3,
        section_modulus_foot_m3=config.analysis_config.section_modulus_foot_m3,
        area_m2=config.analysis_config.area_m2,
        discrete_support_stiffness_n_per_m=config.analysis_config.discrete_support_stiffness_n_per_m,
        design_inputs=config.analysis_config.design_inputs,
    )
    beta = beam_parameter_beta(
        config.analysis_config.foundation_modulus_n_per_m2,
        config.analysis_config.elastic_modulus_pa,
        config.analysis_config.moment_inertia_m4,
    )
    design_summary = build_design_summary(inputs, beta, max_moment_nm=max_moment.value)
    max_formation = {
        depth: max(values) if values else 0.0 for depth, values in formation_max.items()
    }

    return EnvelopeSummary(
        beta_per_m=beta,
        zero_moment_distance_m=zero_moment_distance(beta),
        contraflexure_distance_m=contraflexure_distance(beta),
        max_deflection=max_deflection,
        max_moment=max_moment,
        max_shear=max_shear,
        max_reaction=max_reaction,
        max_sleeper_load=max_sleeper_load,
        max_ballast_pressure=max_ballast,
        max_rail_base_stress_pa=max_stress,
        max_formation_stress_by_depth_pa=max_formation,
        design_summary=design_summary,
    )


def _build_grid(sample_count: int, domain_m: tuple[float, float]) -> list[float]:
    _require_sample_count(sample_count)
    start, end = domain_m
    if end <= start:
        raise ValueError("x_domain_m must define a positive-length domain")
    step = (end - start) / (sample_count - 1)
    return [start + i * step for i in range(sample_count)]


def _build_sleeper_positions(start: float, end: float, spacing: float) -> list[float]:
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    count = int(math.floor((end - start) / spacing))
    return [start + i * spacing for i in range(count + 1)]


def _build_ref_positions(start: float, end: float, step: float) -> list[float]:
    positions: list[float] = []
    current = start
    while current <= end + 1.0e-12:
        positions.append(current)
        current += step
    if not positions:
        raise ValueError("No envelope steps generated.")
    return positions


def _shift_loads(loads: Sequence[PointLoad], offset_m: float) -> list[PointLoad]:
    return [
        PointLoad(position_m=load.position_m + offset_m, load_newtons=load.load_newtons)
        for load in loads
    ]


def _scale_sleeper_loads(
    sleeper_loads: Sequence[float],
    left: Sequence[float] | None,
    right: Sequence[float] | None,
    rail_count: int,
) -> list[float]:
    if left is not None and right is not None:
        return [left[i] + right[i] for i in range(len(left))]
    return [value * rail_count for value in sleeper_loads]


def _indices_in_domain(
    positions: Sequence[float],
    domain_m: tuple[float, float],
    label: str,
) -> list[int]:
    start, end = domain_m
    tolerance = 1.0e-12
    indices = [
        index
        for index, position in enumerate(positions)
        if start - tolerance <= position <= end + tolerance
    ]
    if not indices:
        raise ValueError(f"Envelope analysis produced no {label} values inside the plot domain.")
    return indices


def _take_indices(values: Sequence[float], indices: Sequence[int]) -> list[float]:
    return [values[index] for index in indices]


def _take_optional_indices(
    values: Sequence[float] | None,
    indices: Sequence[int],
) -> list[float] | None:
    if values is None:
        return None
    return _take_indices(values, indices)


def _update_extrema(max_values: list[float], min_values: list[float], values: Sequence[float]) -> None:
    if len(max_values) != len(values) or len(min_values) != len(values):
        raise ValueError("Envelope arrays must be the same length")
    for i, value in enumerate(values):
        if value > max_values[i]:
            max_values[i] = value
        if value < min_values[i]:
            min_values[i] = value


def _init_extrema(length: int, value: float) -> list[float]:
    return [value for _ in range(length)]


def _max_abs_with_position(
    positions: Sequence[float],
    max_values: Sequence[float],
    min_values: Sequence[float],
) -> Extremum:
    if not positions or len(positions) != len(max_values) or len(max_values) != len(min_values):
        raise ValueError("Envelope summary arrays must be non-empty and equal length")
    best_index = 0
    best_value = max_values[0]
    best_abs = abs(max_values[0])
    for i in range(len(positions)):
        max_val = max_values[i]
        min_val = min_values[i]
        max_abs = abs(max_val)
        min_abs = abs(min_val)
        if max_abs >= min_abs and max_abs > best_abs:
            best_abs = max_abs
            best_index = i
            best_value = max_val
        if min_abs > max_abs and min_abs > best_abs:
            best_abs = min_abs
            best_index = i
            best_value = min_val
    return Extremum(value=best_value, position_m=positions[best_index])


def _max_abs_envelope(max_values: Sequence[float], min_values: Sequence[float]) -> list[float]:
    if len(max_values) != len(min_values):
        raise ValueError("Envelope max/min arrays must be the same length")
    return [max(abs(max_val), abs(min_val)) for max_val, min_val in zip(max_values, min_values)]


def _envelope_governing_score(result: EnvelopeResult) -> tuple[float, float, float, float]:
    summary = result.summary
    return (
        abs(summary.max_moment.value),
        abs(summary.max_deflection.value),
        abs(summary.max_reaction.value),
        abs(summary.max_shear.value),
    )


def _formation_stress_2to1(
    ballast_pressure_pa: float,
    bearing_width_m: float,
    bearing_length_m: float,
    depth_m: float,
) -> float:
    # Soil/ballast contact stress is compressive-only in this simplified spread model.
    ballast_pressure_pa = max(0.0, ballast_pressure_pa)
    numerator = bearing_width_m * bearing_length_m
    denominator = (bearing_width_m + 2.0 * depth_m) * (bearing_length_m + 2.0 * depth_m)
    return ballast_pressure_pa * numerator / denominator


def _require_loads(loads: Iterable[PointLoad]) -> None:
    if not list(loads):
        raise ValueError("loads must not be empty")


def _require_sample_count(sample_count: int) -> None:
    if sample_count < 3:
        raise ValueError("sample_count must be >= 3")


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")

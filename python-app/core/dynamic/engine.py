"""Dynamic analysis engine (GUI adapter)."""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Sequence

import numpy as np

from core.dynamic.config import (
    DippedJointConfig,
    DynamicConfig,
    DynamicMode,
    DynamicTransitionConfig,
    DynamicTransitionProfileType,
    DynamicTransitionRunMode,
)
from core.dynamic.results import (
    DippedJointResult,
    DynamicParameterTrace,
    DynamicResult,
    DynamicSummary,
    DynamicTransitionMetrics,
    DynamicTransitionResult,
    DynamicTransitionSeries,
    Extremum,
    TransitionRiskMetrics,
    TransitionRiskPoint,
    WavelengthBandMetric,
)
from core.dynamic.solver import (
    build_spatial_grid,
    build_time_series,
    build_transition_stiffness_profile,
    solve_dipped_joint_forces,
    solve_spatial_response,
    solve_transition_spatial_response,
)
from core.dynamic.validation import (
    require_domain_length,
    require_non_empty,
    require_non_negative,
    require_positive,
    validate_dynamic_transition_config,
    validate_dynamic_advanced_options,
    validate_time_window_coverage,
)
from core.model import beam_parameter_beta, deflection_at

DynamicAnalysisConfig = DynamicConfig | DippedJointConfig | DynamicTransitionConfig
DynamicAnalysisResult = DynamicResult | DippedJointResult | DynamicTransitionResult


def run_dynamic_analysis(config: DynamicAnalysisConfig, *, mode: DynamicMode) -> DynamicAnalysisResult:
    """Run dynamic analysis.

    Note: steady-state and time-history modes share the moving-load solver.
    """
    if mode == DynamicMode.DIPPED_JOINT:
        if not isinstance(config, DippedJointConfig):
            raise TypeError("Dipped joint analysis requires DippedJointConfig (total_dip_angle_rad).")
        _validate_dipped_joint_config(config)
        return solve_dipped_joint_forces(config)
    if mode == DynamicMode.TRANSITION:
        if not isinstance(config, DynamicTransitionConfig):
            raise TypeError("Dynamic transition analysis requires DynamicTransitionConfig.")
        return run_dynamic_transition_analysis(config)
    if not isinstance(config, DynamicConfig):
        raise TypeError("Moving-load dynamic analysis requires DynamicConfig.")
    _validate_config(config)
    spatial = solve_spatial_response(config)
    probes = build_time_series(config, spatial)
    summary = _build_summary(spatial)
    wavelength_band_metrics = _build_wavelength_band_metrics(config, probes)
    transition_risk_metrics = _build_transition_risk_metrics(config, spatial, summary)
    parameter_trace = _build_parameter_trace(config, transition_risk_metrics)
    return DynamicResult(
        spatial=spatial,
        probes=probes,
        summary=summary,
        wavelength_band_metrics=wavelength_band_metrics,
        transition_risk_metrics=transition_risk_metrics,
        parameter_trace=parameter_trace,
    )


def run_dynamic_transition_analysis(config: DynamicTransitionConfig) -> DynamicTransitionResult:
    validate_dynamic_transition_config(config)

    if config.run_mode == DynamicTransitionRunMode.SINGLE:
        single_result = _run_dynamic_transition_single(config, x_ref=config.x_ref_m)
        metrics = _build_dynamic_transition_metrics(single_result, config.x_ref_m)
        series = DynamicTransitionSeries(
            x_m=list(single_result.spatial.xi_m),
            k_profile_n_per_m2=_build_transition_profile_for_result(config, single_result.spatial.xi_m),
            deflection_m=list(single_result.spatial.deflection_m),
            moment_nm=list(single_result.spatial.moment_nm),
            shear_n=list(single_result.spatial.shear_n),
            reaction_n_per_m=list(single_result.spatial.reaction_n_per_m),
        )
        return DynamicTransitionResult(
            solver_fidelity=config.solver_fidelity,
            profile_type=config.profile_type.value,
            run_mode=config.run_mode.value,
            k1_n_per_m2=config.k1_n_per_m2,
            k2_n_per_m2=config.k2_n_per_m2,
            transition_length_m=config.transition_length_m,
            segment_length_m=config.segment_length_m,
            x_ref_m=config.x_ref_m,
            x_ref_start_m=None,
            x_ref_end_m=None,
            x_ref_step_m=None,
            metrics=metrics,
            series=series,
            representative=single_result,
            envelope_count=1,
        )

    if config.x_ref_start_m is None or config.x_ref_end_m is None or config.x_ref_step_m is None:
        raise ValueError("Envelope transition mode requires x_ref_start_m, x_ref_end_m, and x_ref_step_m.")

    x_refs: list[float] = []
    value = config.x_ref_start_m
    while value <= config.x_ref_end_m + 1.0e-12:
        x_refs.append(value)
        value += config.x_ref_step_m
    if not x_refs:
        raise ValueError("No x_ref samples were generated for dynamic transition envelope.")

    results_by_ref: list[tuple[float, DynamicResult]] = [
        (x_ref, _run_dynamic_transition_single(config, x_ref=x_ref))
        for x_ref in x_refs
    ]
    representative_x_ref, representative_result = _pick_governing_transition_result(results_by_ref)

    xi = representative_result.spatial.xi_m
    deflection_max = np.full(len(xi), -np.inf, dtype=float)
    deflection_min = np.full(len(xi), np.inf, dtype=float)
    moment_max = np.full(len(xi), -np.inf, dtype=float)
    moment_min = np.full(len(xi), np.inf, dtype=float)
    shear_max = np.full(len(xi), -np.inf, dtype=float)
    shear_min = np.full(len(xi), np.inf, dtype=float)
    reaction_max = np.full(len(xi), -np.inf, dtype=float)
    reaction_min = np.full(len(xi), np.inf, dtype=float)

    for _x_ref, result in results_by_ref:
        deflection = np.asarray(result.spatial.deflection_m, dtype=float)
        moment = np.asarray(result.spatial.moment_nm, dtype=float)
        shear = np.asarray(result.spatial.shear_n, dtype=float)
        reaction = np.asarray(result.spatial.reaction_n_per_m, dtype=float)
        deflection_max = np.maximum(deflection_max, deflection)
        deflection_min = np.minimum(deflection_min, deflection)
        moment_max = np.maximum(moment_max, moment)
        moment_min = np.minimum(moment_min, moment)
        shear_max = np.maximum(shear_max, shear)
        shear_min = np.minimum(shear_min, shear)
        reaction_max = np.maximum(reaction_max, reaction)
        reaction_min = np.minimum(reaction_min, reaction)

    metrics = _build_dynamic_transition_metrics(
        representative_result,
        representative_x_ref,
        envelope_bounds={
            "deflection": (deflection_max, deflection_min),
            "moment": (moment_max, moment_min),
            "shear": (shear_max, shear_min),
            "reaction": (reaction_max, reaction_min),
        },
    )
    series = DynamicTransitionSeries(
        x_m=list(xi),
        k_profile_n_per_m2=_build_transition_profile_for_result(config, xi),
        deflection_max_m=list(deflection_max),
        deflection_min_m=list(deflection_min),
        moment_max_nm=list(moment_max),
        moment_min_nm=list(moment_min),
        shear_max_n=list(shear_max),
        shear_min_n=list(shear_min),
        reaction_max_n_per_m=list(reaction_max),
        reaction_min_n_per_m=list(reaction_min),
    )
    return DynamicTransitionResult(
        solver_fidelity=config.solver_fidelity,
        profile_type=config.profile_type.value,
        run_mode=config.run_mode.value,
        k1_n_per_m2=config.k1_n_per_m2,
        k2_n_per_m2=config.k2_n_per_m2,
        transition_length_m=config.transition_length_m,
        segment_length_m=config.segment_length_m,
        x_ref_m=representative_x_ref,
        x_ref_start_m=config.x_ref_start_m,
        x_ref_end_m=config.x_ref_end_m,
        x_ref_step_m=config.x_ref_step_m,
        metrics=metrics,
        series=series,
        representative=representative_result,
        envelope_count=len(results_by_ref),
    )


def _run_dynamic_transition_single(config: DynamicTransitionConfig, *, x_ref: float) -> DynamicResult:
    loads = [
        replace(load, position_m=load.position_m + (x_ref - config.x_ref_m))
        for load in config.loads
    ]
    if (
        config.solver_fidelity == "screening"
        or config.profile_type == DynamicTransitionProfileType.UNIFORM
    ):
        dynamic_config = DynamicConfig(
            loads=loads,
            elastic_modulus_pa=config.elastic_modulus_pa,
            moment_inertia_m4=config.moment_inertia_m4,
            section_modulus_m3=config.section_modulus_m3,
            mass_kg_per_m=config.mass_kg_per_m,
            foundation_modulus_n_per_m2=config.k1_n_per_m2,
            foundation_damping_n_s_per_m2=config.foundation_damping_n_s_per_m2,
            speed_m_per_s=config.speed_m_per_s,
            domain_length_m=config.domain_length_m,
            spatial_step_m=config.spatial_step_m,
            probe_positions_m=config.probe_positions_m,
            time_window_s=config.time_window_s,
            sample_rate_hz=config.sample_rate_hz,
            foundation_damping_model=config.foundation_damping_model,
            foundation_loss_factor=config.foundation_loss_factor,
            pasternak_shear_n=config.pasternak_shear_n,
            psd_segment_length=config.psd_segment_length,
            psd_overlap=config.psd_overlap,
            excitation_mode=config.excitation_mode,
            boundary_mode=config.boundary_mode,
            oscillator_unsprung_mass_kg=config.oscillator_unsprung_mass_kg,
            oscillator_suspension_stiffness_n_per_m=config.oscillator_suspension_stiffness_n_per_m,
            oscillator_suspension_damping_n_s_per_m=config.oscillator_suspension_damping_n_s_per_m,
            irregularity_input=config.irregularity_input,
            transition_stiffness_ratio=config.transition_stiffness_ratio,
        )
        _validate_config(dynamic_config)
        spatial = solve_spatial_response(dynamic_config)
        probes = build_time_series(dynamic_config, spatial)
        summary = _build_summary(spatial)
        wavelength_band_metrics = _build_wavelength_band_metrics(dynamic_config, probes)
        transition_risk_metrics = _build_transition_risk_metrics(dynamic_config, spatial, summary)
        parameter_trace = _build_parameter_trace(dynamic_config, transition_risk_metrics)
        return DynamicResult(
            spatial=spatial,
            probes=probes,
            summary=summary,
            wavelength_band_metrics=wavelength_band_metrics,
            transition_risk_metrics=transition_risk_metrics,
            parameter_trace=parameter_trace,
        )

    profile_seed = DynamicConfig(
        loads=loads,
        elastic_modulus_pa=config.elastic_modulus_pa,
        moment_inertia_m4=config.moment_inertia_m4,
        section_modulus_m3=config.section_modulus_m3,
        mass_kg_per_m=config.mass_kg_per_m,
        foundation_modulus_n_per_m2=config.k1_n_per_m2,
        foundation_damping_n_s_per_m2=config.foundation_damping_n_s_per_m2,
        speed_m_per_s=config.speed_m_per_s,
        domain_length_m=config.domain_length_m,
        spatial_step_m=config.spatial_step_m,
        probe_positions_m=config.probe_positions_m,
        time_window_s=config.time_window_s,
        sample_rate_hz=config.sample_rate_hz,
        foundation_damping_model=config.foundation_damping_model,
        foundation_loss_factor=config.foundation_loss_factor,
        pasternak_shear_n=config.pasternak_shear_n,
        psd_segment_length=config.psd_segment_length,
        psd_overlap=config.psd_overlap,
        excitation_mode=config.excitation_mode,
        boundary_mode=config.boundary_mode,
        oscillator_unsprung_mass_kg=config.oscillator_unsprung_mass_kg,
        oscillator_suspension_stiffness_n_per_m=config.oscillator_suspension_stiffness_n_per_m,
        oscillator_suspension_damping_n_s_per_m=config.oscillator_suspension_damping_n_s_per_m,
        irregularity_input=config.irregularity_input,
        transition_stiffness_ratio=config.transition_stiffness_ratio,
    )
    grid = build_spatial_grid(config.domain_length_m, config.spatial_step_m)
    xi = grid.xi_m
    k_profile = build_transition_stiffness_profile(
        x_values=list(xi),
        profile_type=config.profile_type,
        k1_n_per_m2=config.k1_n_per_m2,
        k2_n_per_m2=config.k2_n_per_m2,
        transition_length_m=config.transition_length_m,
        segment_length_m=config.segment_length_m,
    )
    transition_config = replace(config, loads=loads)
    spatial = solve_transition_spatial_response(
        transition_config,
        foundation_profile_n_per_m2=k_profile,
    )
    probes = build_time_series(profile_seed, spatial)
    summary = _build_summary(spatial)
    wavelength_band_metrics = _build_wavelength_band_metrics(profile_seed, probes)
    transition_risk_metrics = _build_transition_risk_metrics(profile_seed, spatial, summary)
    parameter_trace = _build_parameter_trace(profile_seed, transition_risk_metrics)
    return DynamicResult(
        spatial=spatial,
        probes=probes,
        summary=summary,
        wavelength_band_metrics=wavelength_band_metrics,
        transition_risk_metrics=transition_risk_metrics,
        parameter_trace=parameter_trace,
    )


def _build_transition_profile_for_result(
    config: DynamicTransitionConfig,
    x_values: Sequence[float],
) -> list[float]:
    return build_transition_stiffness_profile(
        x_values=x_values,
        profile_type=config.profile_type,
        k1_n_per_m2=config.k1_n_per_m2,
        k2_n_per_m2=config.k2_n_per_m2,
        transition_length_m=config.transition_length_m,
        segment_length_m=config.segment_length_m,
    )


def _pick_governing_transition_result(
    candidates: Sequence[tuple[float, DynamicResult]],
) -> tuple[float, DynamicResult]:
    if not candidates:
        raise ValueError("No transition candidates were generated.")
    return max(
        candidates,
        key=lambda item: abs(item[1].summary.max_deflection.value),
    )


def _build_dynamic_transition_metrics(
    result: DynamicResult,
    governing_x_ref_m: float,
    *,
    envelope_bounds: dict[str, tuple[Sequence[float], Sequence[float]]] | None = None,
) -> DynamicTransitionMetrics:
    if envelope_bounds is None:
        max_deflection = abs(result.summary.max_deflection.value)
        max_moment = abs(result.summary.max_moment.value)
        max_shear = abs(result.summary.max_shear.value)
        max_reaction = abs(result.summary.max_reaction.value)
    else:
        max_deflection = _max_abs_envelope(*envelope_bounds["deflection"])
        max_moment = _max_abs_envelope(*envelope_bounds["moment"])
        max_shear = _max_abs_envelope(*envelope_bounds["shear"])
        max_reaction = _max_abs_envelope(*envelope_bounds["reaction"])

    risk = result.transition_risk_metrics
    if risk is None:
        risk_index = 0.0
        critical_speed_ratio = 0.0
        dynamic_amplification = 0.0
        stiffness_ratio = None
    else:
        risk_index = risk.risk_index
        critical_speed_ratio = risk.critical_speed_ratio
        dynamic_amplification = risk.dynamic_amplification
        stiffness_ratio = risk.transition_stiffness_ratio
    return DynamicTransitionMetrics(
        max_deflection_m=max_deflection,
        max_moment_nm=max_moment,
        max_shear_n=max_shear,
        max_reaction_n_per_m=max_reaction,
        governing_x_ref_m=governing_x_ref_m,
        risk_index=risk_index,
        critical_speed_ratio=critical_speed_ratio,
        dynamic_amplification=dynamic_amplification,
        transition_stiffness_ratio=stiffness_ratio,
    )


def _max_abs_envelope(
    max_series: Sequence[float],
    min_series: Sequence[float],
) -> float:
    max_values = np.asarray(max_series, dtype=float)
    min_values = np.asarray(min_series, dtype=float)
    upper = float(np.max(np.abs(max_values))) if max_values.size else 0.0
    lower = float(np.max(np.abs(min_values))) if min_values.size else 0.0
    return max(upper, lower)


def _build_summary(spatial) -> DynamicSummary:
    deflection = spatial.deflection_m
    moment = spatial.moment_nm
    shear = spatial.shear_n
    reaction = spatial.reaction_n_per_m
    xi = spatial.xi_m

    max_deflection = _extremum(deflection, xi)
    max_moment = _extremum(moment, xi)
    max_shear = _extremum(shear, xi)
    max_reaction = _extremum(reaction, xi)

    return DynamicSummary(
        max_deflection=max_deflection,
        max_moment=max_moment,
        max_shear=max_shear,
        max_reaction=max_reaction,
    )


def _extremum(values: list[float], positions: list[float]) -> Extremum:
    if not values:
        return Extremum(0.0, 0.0)
    idx = max(range(len(values)), key=lambda i: abs(values[i]))
    return Extremum(values[idx], positions[idx])


def _validate_config(config: DynamicConfig) -> None:
    require_non_empty(config.loads, "loads")
    require_positive(config.elastic_modulus_pa, "elastic_modulus_pa")
    require_positive(config.moment_inertia_m4, "moment_inertia_m4")
    require_positive(config.section_modulus_m3, "section_modulus_m3")
    require_positive(config.mass_kg_per_m, "mass_kg_per_m")
    require_positive(config.foundation_modulus_n_per_m2, "foundation_modulus_n_per_m2")
    require_non_negative(config.foundation_damping_n_s_per_m2, "foundation_damping_n_s_per_m2")
    require_non_negative(config.foundation_loss_factor, "foundation_loss_factor")
    require_non_negative(config.speed_m_per_s, "speed_m_per_s")
    require_positive(config.domain_length_m, "domain_length_m")
    require_positive(config.spatial_step_m, "spatial_step_m")
    require_non_empty(config.probe_positions_m, "probe_positions_m")
    require_positive(config.time_window_s, "time_window_s")
    require_positive(config.sample_rate_hz, "sample_rate_hz")
    require_domain_length(
        config.domain_length_m,
        foundation_modulus_n_per_m2=config.foundation_modulus_n_per_m2,
        elastic_modulus_pa=config.elastic_modulus_pa,
        moment_inertia_m4=config.moment_inertia_m4,
    )
    validate_dynamic_advanced_options(config)
    validate_time_window_coverage(config)
    _warn_critical_speed(config)


def _build_wavelength_band_metrics(
    config: DynamicConfig,
    probes,
) -> list[WavelengthBandMetric] | None:
    if not probes:
        return None
    if config.speed_m_per_s <= 0:
        return None
    probe = probes[0]
    freq = probe.fft_frequency_hz
    amp = probe.fft_amplitude
    if not freq or not amp:
        return None

    wavelength_amp_pairs: list[tuple[float, float]] = []
    for f_hz, amplitude in zip(freq, amp):
        if f_hz <= 0:
            continue
        wavelength_m = config.speed_m_per_s / f_hz
        wavelength_amp_pairs.append((wavelength_m, amplitude))
    if not wavelength_amp_pairs:
        return None

    bands = [
        ("short", 0.5, 3.0),
        ("medium", 3.0, 25.0),
        ("long", 25.0, 200.0),
    ]
    metrics: list[WavelengthBandMetric] = []
    for name, lower, upper in bands:
        values = [a for wavelength, a in wavelength_amp_pairs if lower <= wavelength < upper]
        if values:
            rms = math.sqrt(sum(v * v for v in values) / len(values))
        else:
            rms = 0.0
        metrics.append(
            WavelengthBandMetric(
                band_name=name,
                min_wavelength_m=lower,
                max_wavelength_m=upper,
                rms_amplitude_m=rms,
            )
        )
    return metrics


def _build_transition_risk_metrics(
    config: DynamicConfig,
    spatial,
    summary: DynamicSummary,
) -> TransitionRiskMetrics:
    stiffness_term = config.elastic_modulus_pa * config.moment_inertia_m4 * config.foundation_modulus_n_per_m2
    critical_speed = math.sqrt((2.0 * math.sqrt(stiffness_term)) / config.mass_kg_per_m)
    critical_speed_ratio = config.speed_m_per_s / critical_speed if critical_speed > 0 else 0.0

    static_profile = [
        abs(
            deflection_at(
                x,
                config.loads,
                config.foundation_modulus_n_per_m2,
                config.elastic_modulus_pa,
                config.moment_inertia_m4,
            )
        )
        for x in spatial.xi_m
    ]
    static_peak = max(static_profile, default=0.0)
    dynamic_peak = abs(summary.max_deflection.value)
    dynamic_amplification = dynamic_peak / static_peak if static_peak > 0 else 0.0
    risk_index = max(critical_speed_ratio, dynamic_amplification)

    stiffness_ratio = config.transition_stiffness_ratio
    risk_map: list[TransitionRiskPoint] = []
    if stiffness_ratio is not None and stiffness_ratio > 0:
        for speed_ratio in (0.5, 0.75, 1.0, 1.25):
            scaled_critical = speed_ratio * critical_speed_ratio / (stiffness_ratio ** 0.25)
            map_risk = max(scaled_critical, dynamic_amplification * speed_ratio)
            risk_map.append(
                TransitionRiskPoint(
                    speed_ratio=speed_ratio,
                    stiffness_ratio=stiffness_ratio,
                    risk_index=map_risk,
                )
            )

    return TransitionRiskMetrics(
        critical_speed_m_per_s=critical_speed,
        critical_speed_ratio=critical_speed_ratio,
        dynamic_amplification=dynamic_amplification,
        transition_stiffness_ratio=stiffness_ratio,
        risk_index=risk_index,
        risk_map=risk_map,
    )


def _build_parameter_trace(
    config: DynamicConfig,
    risk: TransitionRiskMetrics,
) -> DynamicParameterTrace:
    beta = beam_parameter_beta(
        config.foundation_modulus_n_per_m2,
        config.elastic_modulus_pa,
        config.moment_inertia_m4,
    )
    damping_ratio = config.foundation_damping_n_s_per_m2 / (
        2.0 * math.sqrt(config.foundation_modulus_n_per_m2 * config.mass_kg_per_m)
    )
    return DynamicParameterTrace(
        flexural_rigidity_nm2=config.elastic_modulus_pa * config.moment_inertia_m4,
        foundation_modulus_n_per_m2=config.foundation_modulus_n_per_m2,
        foundation_damping_n_s_per_m2=config.foundation_damping_n_s_per_m2,
        damping_ratio=damping_ratio,
        mass_kg_per_m=config.mass_kg_per_m,
        beta_per_m=beta,
        characteristic_length_m=1.0 / beta,
        spatial_step_m=config.spatial_step_m,
        critical_speed_m_per_s=risk.critical_speed_m_per_s,
        critical_speed_ratio=risk.critical_speed_ratio,
        dynamic_amplification=risk.dynamic_amplification,
    )


def _validate_dipped_joint_config(config: DippedJointConfig) -> None:
    require_positive(config.static_wheel_load_n, "static_wheel_load_n")
    require_non_negative(config.total_dip_angle_rad, "total_dip_angle_rad")
    require_non_negative(config.speed_m_per_s, "speed_m_per_s")
    require_positive(config.hertzian_stiffness_n_per_m, "hertzian_stiffness_n_per_m")
    require_positive(config.track_mass_p1_kg, "track_mass_p1_kg")
    require_positive(config.unsprung_mass_kg, "unsprung_mass_kg")
    require_positive(config.track_mass_p2_kg, "track_mass_p2_kg")
    require_positive(config.track_stiffness_p2_n_per_m, "track_stiffness_p2_n_per_m")
    require_non_negative(config.track_damping_p2_n_s_per_m, "track_damping_p2_n_s_per_m")


def _warn_critical_speed(config: DynamicConfig) -> None:
    if config.mass_kg_per_m <= 0:
        return
    # v_cr^2 = (2 / m) * sqrt(k * EI)
    stiffness_term = config.elastic_modulus_pa * config.moment_inertia_m4 * config.foundation_modulus_n_per_m2
    critical_speed = math.sqrt((2.0 * math.sqrt(stiffness_term)) / config.mass_kg_per_m)
    if config.speed_m_per_s > 0.8 * critical_speed and LOGGER.hasHandlers():
        LOGGER.warning(
            "Dynamic speed %.2f m/s exceeds 0.8× critical speed (v_cr≈%.2f m/s).",
            config.speed_m_per_s,
            critical_speed,
        )

LOGGER = logging.getLogger(__name__)

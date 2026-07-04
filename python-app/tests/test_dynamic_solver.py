import math
from dataclasses import replace

import numpy as np
import pytest

from core.dynamic.config import (
    DynamicBoundaryMode,
    DynamicConfig,
    DynamicExcitationMode,
    DynamicMode,
    IrregularityInput,
    IrregularityMode,
)
from core.foundation.base import DampingModel
from core.load_builder import AS5100RailLoadConfig, build_as5100_rail_loads
from core.dynamic.engine import run_dynamic_analysis
from core.model import PointLoad, beam_parameter_beta, deflection_at


def _base_config() -> DynamicConfig:
    return DynamicConfig(
        loads=[PointLoad(position_m=0.0, load_newtons=100_000.0)],
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.85e-5,
        section_modulus_m3=3.2e-5,
        mass_kg_per_m=60.0,
        foundation_modulus_n_per_m2=10_000_000.0,
        foundation_damping_n_s_per_m2=0.0,
        foundation_damping_model=DampingModel.VISCOUS,
        foundation_loss_factor=0.0,
        speed_m_per_s=0.0,
        domain_length_m=120.0,
        spatial_step_m=0.05,
        probe_positions_m=[0.0],
        time_window_s=2.0,
        sample_rate_hz=200.0,
        psd_segment_length=256,
        psd_overlap=0.5,
    )


def test_dynamic_solver_static_limit_matches_closed_form() -> None:
    config = _base_config()
    result = run_dynamic_analysis(config, mode=DynamicMode.STEADY_STATE)
    spatial = result.spatial
    xi = np.asarray(spatial.xi_m)
    deflection = np.asarray(spatial.deflection_m)
    idx = int(np.argmin(np.abs(xi)))
    dynamic_deflection = deflection[idx]

    static_deflection = deflection_at(
        0.0,
        config.loads,
        config.foundation_modulus_n_per_m2,
        config.elastic_modulus_pa,
        config.moment_inertia_m4,
    )

    assert math.isclose(dynamic_deflection, static_deflection, rel_tol=0.05, abs_tol=1e-6)


def test_dynamic_solver_symmetry_for_stationary_load() -> None:
    config = _base_config()
    result = run_dynamic_analysis(config, mode=DynamicMode.STEADY_STATE)
    spatial = result.spatial
    deflection = np.asarray(spatial.deflection_m)
    moment = np.asarray(spatial.moment_nm)
    shear = np.asarray(spatial.shear_n)
    reaction = np.asarray(spatial.reaction_n_per_m)

    center = len(deflection) // 2
    window = min(200, center - 1)
    left = slice(center - window, center)
    right = slice(center + 1, center + 1 + window)

    assert np.allclose(deflection[left][::-1], deflection[right], rtol=0.01, atol=1e-6)
    assert np.allclose(moment[left][::-1], moment[right], rtol=0.01, atol=1e-3)
    assert np.allclose(shear[left][::-1], -shear[right], rtol=0.01, atol=1e-2)
    assert np.allclose(reaction[left][::-1], reaction[right], rtol=0.01, atol=1e-3)


def test_dynamic_solver_reaction_equilibrium() -> None:
    config = _base_config()
    result = run_dynamic_analysis(config, mode=DynamicMode.STEADY_STATE)
    spatial = result.spatial
    xi = np.asarray(spatial.xi_m)
    reaction = np.asarray(spatial.reaction_n_per_m)
    total_reaction = np.trapezoid(reaction, xi)
    total_load = sum(load.load_newtons for load in config.loads)

    assert math.isclose(total_reaction, total_load, rel_tol=0.02)


def test_dynamic_time_series_shapes() -> None:
    config = _base_config()
    result = run_dynamic_analysis(config, mode=DynamicMode.TIME_HISTORY)
    probe = result.probes[0]

    assert len(probe.time_s) == len(probe.deflection_m)
    assert len(probe.time_s) == len(probe.moment_nm)
    assert len(probe.time_s) == len(probe.shear_n)
    assert len(probe.time_s) == len(probe.reaction_n_per_m)
    assert len(probe.time_s) == len(probe.damping_force_n_per_m)
    assert len(probe.fft_frequency_hz) == len(probe.fft_amplitude)
    assert len(probe.psd_frequency_hz) == len(probe.psd)
    assert len(probe.psd_frequency_hz) == len(probe.psd_ci_lower)
    assert len(probe.psd_frequency_hz) == len(probe.psd_ci_upper)
    assert len(probe.impedance_frequency_hz) == len(probe.impedance_magnitude_n_per_m2)
    assert len(probe.impedance_frequency_hz) == len(probe.impedance_phase_deg)


def test_dynamic_solver_accepts_as5100_fixed_loads() -> None:
    config = replace(
        _base_config(),
        loads=build_as5100_rail_loads(
            AS5100RailLoadConfig(
                model="300LA",
                group_count=1,
                group_spacing_m=12.0,
                reference_position_m=0.0,
            )
        ),
        domain_length_m=80.0,
        spatial_step_m=0.1,
    )

    result = run_dynamic_analysis(config, mode=DynamicMode.STEADY_STATE)

    assert len(config.loads) == 5
    assert result.summary.max_deflection.value > 0.0


def test_dynamic_solver_equilibrium_with_damping_and_speed() -> None:
    config = replace(
        _base_config(),
        foundation_damping_n_s_per_m2=2_000.0,
        speed_m_per_s=25.0,
    )
    result = run_dynamic_analysis(config, mode=DynamicMode.STEADY_STATE)
    spatial = result.spatial
    xi = np.asarray(spatial.xi_m)
    reaction = np.asarray(spatial.reaction_n_per_m)
    total_reaction = np.trapezoid(reaction, xi)
    total_load = sum(load.load_newtons for load in config.loads)

    assert math.isclose(total_reaction, total_load, rel_tol=0.05)


def test_dynamic_solver_hysteretic_damping_force_nonzero() -> None:
    config = replace(
        _base_config(),
        foundation_damping_model=DampingModel.HYSTERETIC,
        foundation_loss_factor=0.1,
    )
    result = run_dynamic_analysis(config, mode=DynamicMode.STEADY_STATE)
    damping_force = np.asarray(result.spatial.damping_force_n_per_m)
    assert np.max(np.abs(damping_force)) > 0.0


def test_dynamic_solver_hysteretic_reaction_includes_damping() -> None:
    config = replace(
        _base_config(),
        foundation_damping_model=DampingModel.HYSTERETIC,
        foundation_loss_factor=0.15,
    )
    result = run_dynamic_analysis(config, mode=DynamicMode.STEADY_STATE)
    deflection = np.asarray(result.spatial.deflection_m)
    reaction = np.asarray(result.spatial.reaction_n_per_m)
    damping_force = np.asarray(result.spatial.damping_force_n_per_m)
    expected_reaction = config.foundation_modulus_n_per_m2 * deflection + damping_force
    residual = reaction - expected_reaction
    max_residual = float(np.max(np.abs(residual)))
    max_reaction = float(np.max(np.abs(reaction)))

    assert max_residual <= 1e-6 + 1e-4 * max_reaction


def test_dynamic_solver_accepts_multiple_loads() -> None:
    config = replace(
        _base_config(),
        loads=[
            PointLoad(position_m=-1.5, load_newtons=80_000.0),
            PointLoad(position_m=1.5, load_newtons=60_000.0),
        ],
    )
    result = run_dynamic_analysis(config, mode=DynamicMode.STEADY_STATE)
    spatial = result.spatial
    xi = np.asarray(spatial.xi_m)
    reaction = np.asarray(spatial.reaction_n_per_m)
    total_reaction = np.trapezoid(reaction, xi)
    total_load = sum(load.load_newtons for load in config.loads)

    assert math.isclose(total_reaction, total_load, rel_tol=0.05)


def test_dynamic_domain_length_validation() -> None:
    config = replace(_base_config(), domain_length_m=1.0)
    with pytest.raises(ValueError, match="domain_length_m must be at least"):
        run_dynamic_analysis(config, mode=DynamicMode.STEADY_STATE)


def test_dynamic_boundary_periodic_wrap_avoids_tail_zeroing() -> None:
    long_window = 8.0
    zero_pad = replace(
        _base_config(),
        speed_m_per_s=30.0,
        time_window_s=long_window,
        boundary_mode=DynamicBoundaryMode.ZERO_PAD,
    )
    periodic = replace(
        zero_pad,
        boundary_mode=DynamicBoundaryMode.PERIODIC_WRAP,
    )

    zero_result = run_dynamic_analysis(zero_pad, mode=DynamicMode.TIME_HISTORY)
    periodic_result = run_dynamic_analysis(periodic, mode=DynamicMode.TIME_HISTORY)
    zero_probe = np.asarray(zero_result.probes[0].deflection_m)
    periodic_probe = np.asarray(periodic_result.probes[0].deflection_m)

    zero_tail = zero_probe[int(0.75 * len(zero_probe)) :]
    periodic_tail = periodic_probe[int(0.75 * len(periodic_probe)) :]
    assert np.mean(np.abs(zero_tail)) < 1.0e-7
    assert np.mean(np.abs(periodic_tail)) > 1.0e-6


def test_moving_oscillator_reduces_to_moving_load_in_limit() -> None:
    base = replace(
        _base_config(),
        speed_m_per_s=20.0,
        boundary_mode=DynamicBoundaryMode.PERIODIC_WRAP,
    )
    moving_load = run_dynamic_analysis(base, mode=DynamicMode.STEADY_STATE)
    moving_oscillator = run_dynamic_analysis(
        replace(
            base,
            excitation_mode=DynamicExcitationMode.MOVING_OSCILLATOR,
            oscillator_unsprung_mass_kg=1.0e-6,
            oscillator_suspension_stiffness_n_per_m=1.0e12,
            oscillator_suspension_damping_n_s_per_m=0.0,
        ),
        mode=DynamicMode.STEADY_STATE,
    )
    assert np.allclose(
        moving_oscillator.spatial.deflection_m,
        moving_load.spatial.deflection_m,
        rtol=1.0e-3,
        atol=1.0e-8,
    )


def test_synthetic_irregularity_is_deterministic_for_seed() -> None:
    config = replace(
        _base_config(),
        speed_m_per_s=15.0,
        irregularity_input=IrregularityInput(
            mode=IrregularityMode.SYNTHETIC_PSD,
            psd_level_m3=1.0e-6,
            seed=42,
        ),
    )
    first = run_dynamic_analysis(config, mode=DynamicMode.STEADY_STATE)
    second = run_dynamic_analysis(config, mode=DynamicMode.STEADY_STATE)
    assert np.allclose(first.spatial.deflection_m, second.spatial.deflection_m, rtol=0.0, atol=1.0e-12)


def test_dynamic_result_includes_optional_metrics_blocks() -> None:
    config = replace(_base_config(), speed_m_per_s=20.0, transition_stiffness_ratio=1.4)
    result = run_dynamic_analysis(config, mode=DynamicMode.STEADY_STATE)
    assert result.transition_risk_metrics is not None
    assert result.transition_risk_metrics.critical_speed_m_per_s > 0.0
    assert result.transition_risk_metrics.transition_stiffness_ratio == pytest.approx(1.4)
    assert result.wavelength_band_metrics is not None
    assert len(result.wavelength_band_metrics) >= 1


def test_dynamic_result_includes_parameter_trace_for_review() -> None:
    config = replace(_base_config(), foundation_damping_n_s_per_m2=2_000.0, speed_m_per_s=20.0)
    result = run_dynamic_analysis(config, mode=DynamicMode.STEADY_STATE)

    trace = result.parameter_trace
    assert trace is not None
    beta = beam_parameter_beta(
        config.foundation_modulus_n_per_m2,
        config.elastic_modulus_pa,
        config.moment_inertia_m4,
    )
    expected_zeta = config.foundation_damping_n_s_per_m2 / (
        2.0 * math.sqrt(config.foundation_modulus_n_per_m2 * config.mass_kg_per_m)
    )

    assert trace.flexural_rigidity_nm2 == pytest.approx(config.elastic_modulus_pa * config.moment_inertia_m4)
    assert trace.foundation_modulus_n_per_m2 == pytest.approx(config.foundation_modulus_n_per_m2)
    assert trace.foundation_damping_n_s_per_m2 == pytest.approx(config.foundation_damping_n_s_per_m2)
    assert trace.damping_ratio == pytest.approx(expected_zeta)
    assert trace.mass_kg_per_m == pytest.approx(config.mass_kg_per_m)
    assert trace.beta_per_m == pytest.approx(beta)
    assert trace.characteristic_length_m == pytest.approx(1.0 / beta)
    assert trace.spatial_step_m == pytest.approx(config.spatial_step_m)
    assert trace.dynamic_amplification == pytest.approx(result.transition_risk_metrics.dynamic_amplification)


def test_dynamic_amplification_uses_true_static_peak_for_offset_load() -> None:
    config = replace(
        _base_config(),
        loads=[PointLoad(position_m=20.0, load_newtons=100_000.0)],
        speed_m_per_s=0.0,
    )
    result = run_dynamic_analysis(config, mode=DynamicMode.STEADY_STATE)
    assert result.transition_risk_metrics is not None
    assert result.transition_risk_metrics.dynamic_amplification == pytest.approx(1.0, rel=0.05)


def test_single_axle_benchmark_is_mesh_convergent() -> None:
    coarse = replace(_base_config(), spatial_step_m=0.10)
    medium = replace(_base_config(), spatial_step_m=0.05)
    fine = replace(_base_config(), spatial_step_m=0.025)

    coarse_result = run_dynamic_analysis(coarse, mode=DynamicMode.STEADY_STATE)
    medium_result = run_dynamic_analysis(medium, mode=DynamicMode.STEADY_STATE)
    fine_result = run_dynamic_analysis(fine, mode=DynamicMode.STEADY_STATE)

    static_peak = deflection_at(
        0.0,
        fine.loads,
        fine.foundation_modulus_n_per_m2,
        fine.elastic_modulus_pa,
        fine.moment_inertia_m4,
    )
    peaks = [
        abs(coarse_result.summary.max_deflection.value),
        abs(medium_result.summary.max_deflection.value),
        abs(fine_result.summary.max_deflection.value),
    ]

    assert peaks[1] == pytest.approx(peaks[2], rel=0.01)
    assert peaks[2] == pytest.approx(static_peak, rel=0.02)
    assert abs(fine_result.summary.max_deflection.position_m) <= fine.spatial_step_m


def test_irregularity_profile_validation_rejects_mismatched_lengths() -> None:
    config = replace(
        _base_config(),
        irregularity_input=IrregularityInput(
            mode=IrregularityMode.PROFILE,
            profile_x_m=[-1.0, 0.0, 1.0],
            profile_z_m=[0.0, 0.001],
        ),
    )
    with pytest.raises(ValueError, match="lengths must match"):
        run_dynamic_analysis(config, mode=DynamicMode.STEADY_STATE)

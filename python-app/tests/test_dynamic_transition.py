from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from core.dynamic.config import (
    DynamicMode,
    IrregularityInput,
    IrregularityMode,
    DynamicTransitionConfig,
    DynamicTransitionProfileType,
    DynamicTransitionRunMode,
)
from core.dynamic import engine as dynamic_engine
from core.dynamic.engine import run_dynamic_analysis
from core.dynamic.results import DynamicResult, DynamicSpatialResult, DynamicSummary, Extremum
from core.foundation.base import DampingModel
from core.load_builder import AS5100RailLoadConfig, build_as5100_rail_loads
from core.model import PointLoad, deflection_at


def _base_transition_config() -> DynamicTransitionConfig:
    return DynamicTransitionConfig(
        loads=[PointLoad(position_m=0.0, load_newtons=100_000.0)],
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.85e-5,
        section_modulus_m3=3.2e-5,
        mass_kg_per_m=60.0,
        foundation_modulus_n_per_m2=10_000_000.0,
        foundation_damping_n_s_per_m2=0.0,
        speed_m_per_s=20.0,
        domain_length_m=120.0,
        spatial_step_m=0.05,
        probe_positions_m=[0.0],
        time_window_s=2.0,
        sample_rate_hz=200.0,
        profile_type=DynamicTransitionProfileType.RAMP,
        run_mode=DynamicTransitionRunMode.SINGLE,
        solver_fidelity="screening",
        k1_n_per_m2=10_000_000.0,
        k2_n_per_m2=20_000_000.0,
        transition_length_m=10.0,
        x_ref_m=0.0,
    )


def _mock_dynamic_result(
    *,
    deflection: float,
    moment: float,
    shear: float,
    reaction: float,
) -> DynamicResult:
    spatial = DynamicSpatialResult(
        xi_m=[-1.0, 0.0, 1.0],
        deflection_m=[0.0, deflection, 0.0],
        moment_nm=[0.0, moment, 0.0],
        shear_n=[0.0, shear, 0.0],
        reaction_n_per_m=[0.0, reaction, 0.0],
        damping_force_n_per_m=[0.0, 0.0, 0.0],
    )
    summary = DynamicSummary(
        max_deflection=Extremum(deflection, 0.0),
        max_moment=Extremum(moment, 0.0),
        max_shear=Extremum(shear, 0.0),
        max_reaction=Extremum(reaction, 0.0),
    )
    return DynamicResult(spatial=spatial, probes=[], summary=summary)


def test_dynamic_transition_single_screening_returns_metrics() -> None:
    config = _base_transition_config()
    result = run_dynamic_analysis(config, mode=DynamicMode.TRANSITION)

    assert result.run_mode == DynamicTransitionRunMode.SINGLE.value
    assert result.solver_fidelity == "screening"
    assert result.envelope_count == 1
    assert result.metrics.max_deflection_m > 0.0
    assert len(result.series.x_m) == len(result.series.k_profile_n_per_m2)


def test_dynamic_transition_accepts_as5100_fixed_loads() -> None:
    config = replace(
        _base_transition_config(),
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

    result = run_dynamic_analysis(config, mode=DynamicMode.TRANSITION)

    assert len(config.loads) == 5
    assert result.metrics.max_deflection_m > 0.0


def test_dynamic_transition_screening_series_keeps_configured_k_profile_shape() -> None:
    config = _base_transition_config()
    result = run_dynamic_analysis(config, mode=DynamicMode.TRANSITION)
    k_profile = np.asarray(result.series.k_profile_n_per_m2)

    assert k_profile.size == len(result.series.x_m)
    assert float(np.max(k_profile) - np.min(k_profile)) > 0.0
    assert np.isclose(float(np.min(k_profile)), config.k1_n_per_m2, rtol=0.0, atol=1.0e-9)
    assert np.isclose(float(np.max(k_profile)), config.k2_n_per_m2, rtol=0.0, atol=1.0e-9)


def test_dynamic_transition_envelope_sweeps_reference_positions() -> None:
    config = replace(
        _base_transition_config(),
        run_mode=DynamicTransitionRunMode.ENVELOPE,
        x_ref_start_m=-2.0,
        x_ref_end_m=2.0,
        x_ref_step_m=1.0,
    )
    result = run_dynamic_analysis(config, mode=DynamicMode.TRANSITION)

    assert result.run_mode == DynamicTransitionRunMode.ENVELOPE.value
    assert result.envelope_count == 5
    assert result.series.deflection_max_m is not None
    assert result.series.deflection_min_m is not None
    assert result.x_ref_m is not None
    assert config.x_ref_start_m <= result.x_ref_m <= config.x_ref_end_m


def test_dynamic_transition_full_profile_uniform_matches_base_solver() -> None:
    transition = replace(
        _base_transition_config(),
        solver_fidelity="full_profile",
        profile_type=DynamicTransitionProfileType.UNIFORM,
        k2_n_per_m2=None,
        speed_m_per_s=10.0,
    )
    baseline = replace(
        transition,
        solver_fidelity="screening",
    )

    full_result = run_dynamic_analysis(transition, mode=DynamicMode.TRANSITION)
    base_result = run_dynamic_analysis(baseline, mode=DynamicMode.TRANSITION)

    full_deflection = np.asarray(full_result.representative.spatial.deflection_m)
    base_deflection = np.asarray(base_result.representative.spatial.deflection_m)
    assert np.allclose(full_deflection, base_deflection, rtol=0.08, atol=1.0e-7)


def test_dynamic_transition_static_limit_at_zero_speed() -> None:
    config = replace(_base_transition_config(), speed_m_per_s=0.0, solver_fidelity="screening")
    result = run_dynamic_analysis(config, mode=DynamicMode.TRANSITION)
    xi = np.asarray(result.representative.spatial.xi_m)
    deflection = np.asarray(result.representative.spatial.deflection_m)
    idx = int(np.argmin(np.abs(xi)))

    static_deflection = deflection_at(
        0.0,
        config.loads,
        config.k1_n_per_m2,
        config.elastic_modulus_pa,
        config.moment_inertia_m4,
    )
    assert math.isclose(deflection[idx], static_deflection, rel_tol=0.05, abs_tol=1.0e-6)


def test_dynamic_transition_rejects_moving_oscillator_mode() -> None:
    from core.dynamic.config import DynamicExcitationMode

    config = replace(_base_transition_config(), excitation_mode=DynamicExcitationMode.MOVING_OSCILLATOR)
    with pytest.raises(ValueError, match="Moving oscillator excitation"):
        run_dynamic_analysis(config, mode=DynamicMode.TRANSITION)


def test_dynamic_transition_rejects_periodic_wrap_in_full_profile_mode() -> None:
    from core.dynamic.config import DynamicBoundaryMode

    config = replace(
        _base_transition_config(),
        solver_fidelity="full_profile",
        boundary_mode=DynamicBoundaryMode.PERIODIC_WRAP,
    )
    with pytest.raises(ValueError, match="Periodic boundary mode is not supported"):
        run_dynamic_analysis(config, mode=DynamicMode.TRANSITION)


def test_dynamic_transition_rejects_irregularity_in_full_profile_mode() -> None:
    config = replace(
        _base_transition_config(),
        solver_fidelity="full_profile",
        irregularity_input=IrregularityInput(
            mode=IrregularityMode.SYNTHETIC_PSD,
            psd_level_m3=1.0e-8,
            seed=7,
        ),
    )
    with pytest.raises(ValueError, match="Irregularity excitation is not supported"):
        run_dynamic_analysis(config, mode=DynamicMode.TRANSITION)


def test_dynamic_transition_rejects_hysteretic_in_full_profile_mode() -> None:
    config = replace(
        _base_transition_config(),
        solver_fidelity="full_profile",
        foundation_damping_model=DampingModel.HYSTERETIC,
        foundation_loss_factor=0.1,
    )
    with pytest.raises(ValueError, match="Hysteretic damping is not supported"):
        run_dynamic_analysis(config, mode=DynamicMode.TRANSITION)


def test_dynamic_transition_screening_allows_periodic_wrap_and_irregularity() -> None:
    from core.dynamic.config import DynamicBoundaryMode

    config = replace(
        _base_transition_config(),
        solver_fidelity="screening",
        boundary_mode=DynamicBoundaryMode.PERIODIC_WRAP,
        irregularity_input=IrregularityInput(
            mode=IrregularityMode.SYNTHETIC_PSD,
            psd_level_m3=1.0e-8,
            seed=11,
        ),
    )
    result = run_dynamic_analysis(config, mode=DynamicMode.TRANSITION)
    assert result.solver_fidelity == "screening"
    assert result.representative.spatial.deflection_m


def test_dynamic_transition_screening_allows_hysteretic_damping() -> None:
    config = replace(
        _base_transition_config(),
        solver_fidelity="screening",
        foundation_damping_model=DampingModel.HYSTERETIC,
        foundation_loss_factor=0.05,
    )
    result = run_dynamic_analysis(config, mode=DynamicMode.TRANSITION)
    damping_force = np.asarray(result.representative.spatial.damping_force_n_per_m, dtype=float)
    assert np.max(np.abs(damping_force)) > 0.0


def test_dynamic_transition_envelope_metrics_use_envelope_extrema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        _base_transition_config(),
        run_mode=DynamicTransitionRunMode.ENVELOPE,
        x_ref_start_m=0.0,
        x_ref_end_m=1.0,
        x_ref_step_m=1.0,
    )
    by_ref = {
        0.0: _mock_dynamic_result(deflection=10.0, moment=1.0, shear=2.0, reaction=3.0),
        1.0: _mock_dynamic_result(deflection=2.0, moment=40.0, shear=50.0, reaction=60.0),
    }

    def _fake_single(_config: DynamicTransitionConfig, *, x_ref: float) -> DynamicResult:
        return by_ref[x_ref]

    monkeypatch.setattr(dynamic_engine, "_run_dynamic_transition_single", _fake_single)

    result = dynamic_engine.run_dynamic_transition_analysis(config)

    assert result.metrics.max_deflection_m == 10.0
    assert result.metrics.max_moment_nm == 40.0
    assert result.metrics.max_shear_n == 50.0
    assert result.metrics.max_reaction_n_per_m == 60.0

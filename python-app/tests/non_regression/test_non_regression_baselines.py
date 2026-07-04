"""Baseline non-regression checks for canonical BOEF outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.analysis import AnalysisInputs, compute_track_response
from core.dynamic.config import DynamicConfig, DynamicMode
from core.dynamic.engine import run_dynamic_analysis
from core.foundation.base import DampingModel
from core.model import PointLoad
from core.transition import (
    TransitionProfileType,
    build_transition_profile,
    compute_metrics_from_series,
)

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"


def _load_fixture(name: str) -> dict:
    return json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))


def test_closed_form_static_baseline() -> None:
    fixture = _load_fixture("non_regression_static_closed_form.json")
    inputs = fixture["inputs"]
    result = compute_track_response(
        AnalysisInputs(
            loads=[PointLoad(**entry) for entry in inputs["loads"]],
            foundation_modulus_n_per_m2=inputs["foundation_modulus_n_per_m2"],
            elastic_modulus_pa=inputs["elastic_modulus_pa"],
            moment_inertia_m4=inputs["moment_inertia_m4"],
            section_modulus_m3=inputs["section_modulus_m3"],
            sleeper_spacing_m=inputs["sleeper_spacing_m"],
            sleeper_length_m=inputs["sleeper_length_m"],
            sleeper_width_m=inputs["sleeper_width_m"],
            sample_count=inputs["sample_count"],
        )
    )
    assert np.allclose(result.x_m, fixture["x_m"], rtol=0.0, atol=1.0e-12)
    assert np.allclose(result.deflection_m, fixture["deflection_m"], rtol=1.0e-9, atol=1.0e-12)
    assert np.allclose(result.moment_nm, fixture["moment_nm"], rtol=1.0e-9, atol=1.0e-8)
    assert np.allclose(result.shear_n, fixture["shear_n"], rtol=1.0e-9, atol=1.0e-8)
    assert np.allclose(result.reaction_n_per_m, fixture["reaction_n_per_m"], rtol=1.0e-9, atol=1.0e-8)


def test_transition_metrics_baseline() -> None:
    static_fixture = _load_fixture("non_regression_static_closed_form.json")
    fixture = _load_fixture("non_regression_transition_metrics.json")
    inputs = static_fixture["inputs"]
    result = compute_track_response(
        AnalysisInputs(
            loads=[PointLoad(**entry) for entry in inputs["loads"]],
            foundation_modulus_n_per_m2=inputs["foundation_modulus_n_per_m2"],
            elastic_modulus_pa=inputs["elastic_modulus_pa"],
            moment_inertia_m4=inputs["moment_inertia_m4"],
            section_modulus_m3=inputs["section_modulus_m3"],
            sleeper_spacing_m=inputs["sleeper_spacing_m"],
            sleeper_length_m=inputs["sleeper_length_m"],
            sleeper_width_m=inputs["sleeper_width_m"],
            sample_count=inputs["sample_count"],
        )
    )
    k_profile = build_transition_profile(
        x_values=result.x_m,
        profile_type=TransitionProfileType.RAMP,
        k1_n_per_m2=45_000_000.0,
        k2_n_per_m2=85_000_000.0,
        transition_length_m=3.0,
        segment_length_m=None,
    )
    assert np.allclose(k_profile, fixture["k_profile_n_per_m2"], rtol=0.0, atol=1.0e-12)

    metrics = compute_metrics_from_series(
        x_values=result.x_m,
        deflection_m=result.deflection_m,
        moment_nm=result.moment_nm,
        reaction_n_per_m=result.reaction_n_per_m,
        sleeper_positions_m=result.sleeper_positions_m,
        sleeper_loads_n=result.sleeper_loads_n,
        sleeper_spacing_m=inputs["sleeper_spacing_m"],
        elastic_modulus_pa=inputs["elastic_modulus_pa"],
        moment_inertia_m4=inputs["moment_inertia_m4"],
    )
    for key, expected in fixture["metrics"].items():
        assert np.isclose(getattr(metrics, key), expected, rtol=1.0e-9, atol=1.0e-12), key


def test_dynamic_static_limit_baseline() -> None:
    fixture = _load_fixture("non_regression_dynamic_static_limit.json")
    config_data = fixture["config"]
    result = run_dynamic_analysis(
        DynamicConfig(
            loads=[PointLoad(**entry) for entry in config_data["loads"]],
            elastic_modulus_pa=config_data["elastic_modulus_pa"],
            moment_inertia_m4=config_data["moment_inertia_m4"],
            section_modulus_m3=config_data["section_modulus_m3"],
            mass_kg_per_m=config_data["mass_kg_per_m"],
            foundation_modulus_n_per_m2=config_data["foundation_modulus_n_per_m2"],
            foundation_damping_n_s_per_m2=config_data["foundation_damping_n_s_per_m2"],
            foundation_damping_model=DampingModel.VISCOUS,
            foundation_loss_factor=0.0,
            speed_m_per_s=config_data["speed_m_per_s"],
            domain_length_m=config_data["domain_length_m"],
            spatial_step_m=config_data["spatial_step_m"],
            probe_positions_m=[0.0],
            time_window_s=2.0,
            sample_rate_hz=200.0,
            psd_segment_length=256,
            psd_overlap=0.5,
        ),
        mode=DynamicMode.STEADY_STATE,
    )
    assert np.allclose(result.spatial.xi_m, fixture["xi_m"], rtol=0.0, atol=1.0e-12)
    assert np.allclose(result.spatial.deflection_m, fixture["deflection_m"], rtol=1.0e-9, atol=1.0e-12)
    assert np.allclose(result.spatial.moment_nm, fixture["moment_nm"], rtol=1.0e-9, atol=1.0e-8)
    assert np.allclose(result.spatial.shear_n, fixture["shear_n"], rtol=1.0e-9, atol=1.0e-8)
    assert np.allclose(result.spatial.reaction_n_per_m, fixture["reaction_n_per_m"], rtol=1.0e-9, atol=1.0e-8)

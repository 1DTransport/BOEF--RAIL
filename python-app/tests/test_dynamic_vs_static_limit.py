"""Validation-only comparison between dynamic and static modules."""

import math

import numpy as np

from core.analysis import AnalysisInputs, compute_track_response
from core.dynamic.config import DynamicConfig, DynamicMode
from core.foundation.base import DampingModel
from core.dynamic.engine import run_dynamic_analysis
from core.model import PointLoad


def test_dynamic_static_limit_matches_analysis_module() -> None:
    """Static limit cross-check (not a runtime dependency)."""
    config = DynamicConfig(
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
    analysis_inputs = AnalysisInputs(
        loads=config.loads,
        foundation_modulus_n_per_m2=config.foundation_modulus_n_per_m2,
        elastic_modulus_pa=config.elastic_modulus_pa,
        moment_inertia_m4=config.moment_inertia_m4,
        section_modulus_m3=config.section_modulus_m3,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=401,
    )
    analysis_result = compute_track_response(analysis_inputs)
    static_deflection = analysis_result.deflection_m[len(analysis_result.deflection_m) // 2]

    dynamic_result = run_dynamic_analysis(config, mode=DynamicMode.STEADY_STATE)
    xi = np.asarray(dynamic_result.spatial.xi_m)
    deflection = np.asarray(dynamic_result.spatial.deflection_m)
    idx = int(np.argmin(np.abs(xi)))
    dynamic_deflection = deflection[idx]

    assert math.isclose(dynamic_deflection, static_deflection, rel_tol=0.05, abs_tol=1e-6)

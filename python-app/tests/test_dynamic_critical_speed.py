import logging
import math

import pytest

from core.dynamic.config import DynamicConfig
from core.dynamic.engine import _warn_critical_speed
from core.foundation.base import DampingModel
from core.model import PointLoad


def _config_with_speed(speed_m_per_s: float, *, elastic_modulus_pa: float, moment_inertia_m4: float, foundation_modulus_n_per_m2: float, mass_kg_per_m: float) -> DynamicConfig:
    return DynamicConfig(
        loads=[PointLoad(position_m=0.0, load_newtons=100_000.0)],
        elastic_modulus_pa=elastic_modulus_pa,
        moment_inertia_m4=moment_inertia_m4,
        section_modulus_m3=3.2e-5,
        mass_kg_per_m=mass_kg_per_m,
        foundation_modulus_n_per_m2=foundation_modulus_n_per_m2,
        foundation_damping_n_s_per_m2=0.0,
        foundation_damping_model=DampingModel.VISCOUS,
        foundation_loss_factor=0.0,
        speed_m_per_s=speed_m_per_s,
        domain_length_m=120.0,
        spatial_step_m=0.05,
        probe_positions_m=[0.0],
        time_window_s=2.0,
        sample_rate_hz=200.0,
        psd_segment_length=256,
        psd_overlap=0.5,
    )


def _critical_speed(elastic_modulus_pa: float, moment_inertia_m4: float, foundation_modulus_n_per_m2: float, mass_kg_per_m: float) -> float:
    stiffness_term = elastic_modulus_pa * moment_inertia_m4 * foundation_modulus_n_per_m2
    return math.sqrt((2.0 * math.sqrt(stiffness_term)) / mass_kg_per_m)


def test_critical_speed_monotonicity_and_warning(caplog: pytest.LogCaptureFixture) -> None:
    base = dict(
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.85e-5,
        foundation_modulus_n_per_m2=10_000_000.0,
        mass_kg_per_m=60.0,
    )
    v_cr = _critical_speed(**base)

    higher_k = _critical_speed(
        elastic_modulus_pa=base["elastic_modulus_pa"],
        moment_inertia_m4=base["moment_inertia_m4"],
        foundation_modulus_n_per_m2=base["foundation_modulus_n_per_m2"] * 2.0,
        mass_kg_per_m=base["mass_kg_per_m"],
    )
    higher_m = _critical_speed(
        elastic_modulus_pa=base["elastic_modulus_pa"],
        moment_inertia_m4=base["moment_inertia_m4"],
        foundation_modulus_n_per_m2=base["foundation_modulus_n_per_m2"],
        mass_kg_per_m=base["mass_kg_per_m"] * 2.0,
    )
    higher_ei = _critical_speed(
        elastic_modulus_pa=base["elastic_modulus_pa"] * 1.5,
        moment_inertia_m4=base["moment_inertia_m4"],
        foundation_modulus_n_per_m2=base["foundation_modulus_n_per_m2"],
        mass_kg_per_m=base["mass_kg_per_m"],
    )

    assert higher_k > v_cr
    assert higher_ei > v_cr
    assert higher_m < v_cr

    config = _config_with_speed(
        v_cr * 0.81,
        elastic_modulus_pa=base["elastic_modulus_pa"],
        moment_inertia_m4=base["moment_inertia_m4"],
        foundation_modulus_n_per_m2=base["foundation_modulus_n_per_m2"],
        mass_kg_per_m=base["mass_kg_per_m"],
    )
    with caplog.at_level(logging.WARNING):
        _warn_critical_speed(config)

    assert any("critical speed" in record.message for record in caplog.records)

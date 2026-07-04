import math

import pytest

from core.special.config import FloatingSlabConfig
from core.special.solver import solve_floating_slab


def test_floating_slab_basic_values() -> None:
    k_eff = 5.0e7
    mass = 20_000.0
    zeta = 0.1
    c = 2.0 * zeta * math.sqrt(k_eff * mass)
    config = FloatingSlabConfig(
        slab_mass_kg=mass,
        isolator_stiffness_n_per_m=k_eff,
        isolator_damping_n_s_per_m=c,
        static_load_n=100_000.0,
        frequency_min_hz=0.0,
        frequency_max_hz=50.0,
        frequency_points=5,
    )
    result = solve_floating_slab(config)

    expected_fn = math.sqrt(k_eff / mass) / (2.0 * math.pi)
    assert result.natural_frequency_hz == pytest.approx(expected_fn, rel=1.0e-6)
    assert result.damping_ratio == pytest.approx(zeta, rel=1.0e-6)
    assert result.static_deflection_m == pytest.approx(0.002, rel=1.0e-6)
    assert len(result.frequency_hz) == 5
    assert len(result.transmissibility) == 5
    assert len(result.attenuation_db) == 5

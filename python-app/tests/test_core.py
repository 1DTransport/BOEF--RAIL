import math

import pytest

from core.model import (
    PointLoad,
    contraflexure_distance,
    compute_deflection,
    deflection_at,
    max_deflection_single_load,
    max_moment_single_load,
    moment_at,
    rail_base_stress,
    rail_seat_load_from_deflection,
    reaction_at,
    shear_at,
    sleeper_seat_loads,
    zero_moment_distance,
)
from core.units import (
    kn_to_n,
    kpa_to_pa,
    m3_to_mm3,
    m4_to_mm4,
    mm3_to_m3,
    mm4_to_m4,
    mm_to_m,
    m_to_mm,
    mpa_to_pa,
    n_to_kn,
    pa_to_kpa,
    pa_to_mpa,
)


def test_compute_deflection_returns_expected_value() -> None:
    assert compute_deflection(1000.0, 2000.0) == 0.5


def test_compute_deflection_rejects_negative_load() -> None:
    with pytest.raises(ValueError, match="load_newtons must be non-negative"):
        compute_deflection(-1.0, 2000.0)


def test_compute_deflection_rejects_non_positive_stiffness() -> None:
    with pytest.raises(ValueError, match="stiffness_newtons_per_meter must be positive"):
        compute_deflection(1000.0, 0.0)


def test_point_load_symmetry_about_origin() -> None:
    loads = [PointLoad(position_m=0.0, load_newtons=10000.0)]
    foundation_modulus = 40_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    x = 0.8

    assert math.isclose(
        deflection_at(x, loads, foundation_modulus, elastic_modulus, moment_inertia),
        deflection_at(-x, loads, foundation_modulus, elastic_modulus, moment_inertia),
        rel_tol=1e-9,
    )
    assert math.isclose(
        moment_at(x, loads, foundation_modulus, elastic_modulus, moment_inertia),
        moment_at(-x, loads, foundation_modulus, elastic_modulus, moment_inertia),
        rel_tol=1e-9,
    )
    assert math.isclose(
        reaction_at(x, loads, foundation_modulus, elastic_modulus, moment_inertia),
        reaction_at(-x, loads, foundation_modulus, elastic_modulus, moment_inertia),
        rel_tol=1e-9,
    )
    assert math.isclose(
        shear_at(x, loads, foundation_modulus, elastic_modulus, moment_inertia),
        -shear_at(-x, loads, foundation_modulus, elastic_modulus, moment_inertia),
        rel_tol=1e-9,
    )


def test_reaction_equilibrium_matches_total_load() -> None:
    loads = [
        PointLoad(position_m=-0.6, load_newtons=12_000.0),
        PointLoad(position_m=0.9, load_newtons=8_000.0),
    ]
    foundation_modulus = 35_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5

    beta = (foundation_modulus / (4.0 * elastic_modulus * moment_inertia)) ** 0.25
    length = 10.0 / beta
    samples = 4000
    dx = 2.0 * length / samples
    x_start = -length
    reaction_sum = 0.0
    for i in range(samples + 1):
        x = x_start + i * dx
        weight = 0.5 if i in (0, samples) else 1.0
        reaction_sum += weight * reaction_at(
            x, loads, foundation_modulus, elastic_modulus, moment_inertia
        )
    reaction_sum *= dx

    assert math.isclose(
        reaction_sum,
        sum(load.load_newtons for load in loads),
        rel_tol=1e-3,
    )


def test_two_point_loads_superposition_matches_combined_response() -> None:
    load_a = PointLoad(position_m=-0.4, load_newtons=11_000.0)
    load_b = PointLoad(position_m=0.9, load_newtons=7_500.0)
    loads = [load_a, load_b]
    foundation_modulus = 40_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    positions = [-1.2, -0.1, 0.5, 1.3]

    for x in positions:
        y_sum = deflection_at(
            x, [load_a], foundation_modulus, elastic_modulus, moment_inertia
        ) + deflection_at(x, [load_b], foundation_modulus, elastic_modulus, moment_inertia)
        m_sum = moment_at(
            x, [load_a], foundation_modulus, elastic_modulus, moment_inertia
        ) + moment_at(x, [load_b], foundation_modulus, elastic_modulus, moment_inertia)
        v_sum = shear_at(
            x, [load_a], foundation_modulus, elastic_modulus, moment_inertia
        ) + shear_at(x, [load_b], foundation_modulus, elastic_modulus, moment_inertia)
        p_sum = reaction_at(
            x, [load_a], foundation_modulus, elastic_modulus, moment_inertia
        ) + reaction_at(x, [load_b], foundation_modulus, elastic_modulus, moment_inertia)

        assert math.isclose(
            y_sum,
            deflection_at(x, loads, foundation_modulus, elastic_modulus, moment_inertia),
            rel_tol=1e-9,
        )
        assert math.isclose(
            m_sum,
            moment_at(x, loads, foundation_modulus, elastic_modulus, moment_inertia),
            rel_tol=1e-9,
        )
        assert math.isclose(
            v_sum,
            shear_at(x, loads, foundation_modulus, elastic_modulus, moment_inertia),
            rel_tol=1e-9,
        )
        assert math.isclose(
            p_sum,
            reaction_at(x, loads, foundation_modulus, elastic_modulus, moment_inertia),
            rel_tol=1e-9,
        )


def test_moment_and_shear_match_deflection_derivatives() -> None:
    loads = [PointLoad(position_m=0.3, load_newtons=9_500.0)]
    foundation_modulus = 42_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    x = 1.1
    h = 1e-4

    y_plus = deflection_at(x + h, loads, foundation_modulus, elastic_modulus, moment_inertia)
    y = deflection_at(x, loads, foundation_modulus, elastic_modulus, moment_inertia)
    y_minus = deflection_at(x - h, loads, foundation_modulus, elastic_modulus, moment_inertia)

    second_derivative = (y_plus - 2.0 * y + y_minus) / (h**2)
    moment_numeric = -elastic_modulus * moment_inertia * second_derivative
    assert math.isclose(
        moment_numeric,
        moment_at(x, loads, foundation_modulus, elastic_modulus, moment_inertia),
        rel_tol=2e-3,
    )

    m_plus = moment_at(x + h, loads, foundation_modulus, elastic_modulus, moment_inertia)
    m_minus = moment_at(x - h, loads, foundation_modulus, elastic_modulus, moment_inertia)
    shear_numeric = (m_plus - m_minus) / (2.0 * h)
    assert math.isclose(
        shear_numeric,
        shear_at(x, loads, foundation_modulus, elastic_modulus, moment_inertia),
        rel_tol=2e-3,
    )


def test_sleeper_seat_loads_sum_to_applied_loads() -> None:
    loads = [
        PointLoad(position_m=0.0, load_newtons=12_000.0),
        PointLoad(position_m=1.2, load_newtons=9_000.0),
    ]
    foundation_modulus = 38_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    tributary = 0.6
    sleeper_positions = [i * tributary for i in range(-15, 16)]

    seat_loads = sleeper_seat_loads(
        sleeper_positions,
        tributary,
        loads,
        foundation_modulus,
        elastic_modulus,
        moment_inertia,
    )

    assert math.isclose(
        sum(seat_loads),
        sum(load.load_newtons for load in loads),
        rel_tol=1e-3,
    )


def test_single_load_maxima_and_distances_match_closed_form() -> None:
    load = PointLoad(position_m=0.0, load_newtons=12_000.0)
    foundation_modulus = 40_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    beta = (foundation_modulus / (4.0 * elastic_modulus * moment_inertia)) ** 0.25

    assert math.isclose(
        max_deflection_single_load(load.load_newtons, foundation_modulus, beta),
        deflection_at(0.0, [load], foundation_modulus, elastic_modulus, moment_inertia),
        rel_tol=1e-9,
    )
    assert math.isclose(
        max_moment_single_load(load.load_newtons, beta),
        moment_at(0.0, [load], foundation_modulus, elastic_modulus, moment_inertia),
        rel_tol=1e-9,
    )
    assert math.isclose(
        zero_moment_distance(beta),
        math.pi / (4.0 * beta),
        rel_tol=1e-12,
    )
    assert math.isclose(
        contraflexure_distance(beta),
        3.0 * math.pi / (4.0 * beta),
        rel_tol=1e-12,
    )


def test_rail_base_stress_and_seat_load_helpers() -> None:
    moment = 25_000.0
    section_modulus = 3.2e-5
    stress = rail_base_stress(moment, section_modulus)

    assert math.isclose(stress, moment / section_modulus, rel_tol=1e-12)

    load = rail_seat_load_from_deflection(
        sleeper_spacing_m=0.6,
        foundation_modulus_n_per_m2=45_000_000.0,
        max_deflection_m=0.002,
        factor=1.1,
    )
    assert math.isclose(load, 0.6 * 45_000_000.0 * 0.002 * 1.1, rel_tol=1e-12)


def test_rejects_non_finite_loads() -> None:
    foundation_modulus = 40_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5

    with pytest.raises(ValueError, match="load position must be finite"):
        deflection_at(
            0.0,
            [PointLoad(position_m=math.nan, load_newtons=10_000.0)],
            foundation_modulus,
            elastic_modulus,
            moment_inertia,
        )

    with pytest.raises(ValueError, match="load_newtons must be finite"):
        deflection_at(
            0.0,
            [PointLoad(position_m=0.0, load_newtons=math.nan)],
            foundation_modulus,
            elastic_modulus,
            moment_inertia,
        )


def test_unit_conversions_round_trip() -> None:
    assert math.isclose(m_to_mm(1.2), 1200.0)
    assert math.isclose(mm_to_m(1200.0), 1.2)
    assert math.isclose(kn_to_n(12.5), 12_500.0)
    assert math.isclose(n_to_kn(12_500.0), 12.5)
    assert math.isclose(mpa_to_pa(210.0), 210_000_000.0)
    assert math.isclose(pa_to_mpa(210_000_000.0), 210.0)
    assert math.isclose(mm3_to_m3(1_000_000_000.0), 1.0)
    assert math.isclose(m3_to_mm3(1.0), 1_000_000_000.0)
    assert math.isclose(mm4_to_m4(1_000_000_000_000.0), 1.0)
    assert math.isclose(m4_to_mm4(1.0), 1_000_000_000_000.0)
    assert math.isclose(kpa_to_pa(12.5), 12_500.0)
    assert math.isclose(pa_to_kpa(12_500.0), 12.5)

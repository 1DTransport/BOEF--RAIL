import math

import pytest

from core.model import PointLoad, beam_parameter_beta, deflection_at
from core.track_modulus import (
    estimate_track_modulus_deflection_area,
    estimate_track_modulus_deflection_area_delta,
    estimate_track_modulus_single_deflection,
    synthesize_track_modulus_from_springs,
    track_modulus_from_deflection,
    track_modulus_from_spring_constant,
    track_spring_constant_from_deflection,
)


def _build_grid(length: float, samples: int) -> list[float]:
    step = (2.0 * length) / (samples - 1)
    return [-length + i * step for i in range(samples)]


def test_deflection_area_estimate_matches_known_modulus() -> None:
    load = PointLoad(position_m=0.0, load_newtons=10_000.0)
    foundation_modulus = 45_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    beta = beam_parameter_beta(foundation_modulus, elastic_modulus, moment_inertia)
    length = 10.0 / beta
    x_values = _build_grid(length, 801)

    deflections = [
        deflection_at(x, [load], foundation_modulus, elastic_modulus, moment_inertia)
        for x in x_values
    ]

    estimate = estimate_track_modulus_deflection_area(
        total_load_newtons=load.load_newtons,
        x_m=x_values,
        deflection_m=deflections,
    )

    assert math.isclose(estimate.modulus_n_per_m2, foundation_modulus, rel_tol=2e-2)


def test_deflection_area_rejects_non_finite_profiles() -> None:
    with pytest.raises(ValueError, match="x_m must contain only finite values"):
        estimate_track_modulus_deflection_area(
            total_load_newtons=10_000.0,
            x_m=[0.0, math.nan, 1.0],
            deflection_m=[0.1, 0.1, 0.1],
        )

    with pytest.raises(ValueError, match="values must contain only finite values"):
        estimate_track_modulus_deflection_area(
            total_load_newtons=10_000.0,
            x_m=[0.0, 1.0, 2.0],
            deflection_m=[0.1, math.nan, 0.1],
        )


def test_deflection_area_delta_estimate_matches_known_modulus() -> None:
    load_low = PointLoad(position_m=0.0, load_newtons=8_000.0)
    load_high = PointLoad(position_m=0.0, load_newtons=12_000.0)
    foundation_modulus = 38_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    beta = beam_parameter_beta(foundation_modulus, elastic_modulus, moment_inertia)
    length = 10.0 / beta
    x_values = _build_grid(length, 801)

    deflections_low = [
        deflection_at(x, [load_low], foundation_modulus, elastic_modulus, moment_inertia)
        for x in x_values
    ]
    deflections_high = [
        deflection_at(x, [load_high], foundation_modulus, elastic_modulus, moment_inertia)
        for x in x_values
    ]

    estimate = estimate_track_modulus_deflection_area_delta(
        load_high_newtons=load_high.load_newtons,
        deflection_high_m=deflections_high,
        load_low_newtons=load_low.load_newtons,
        deflection_low_m=deflections_low,
        x_m=x_values,
    )

    assert math.isclose(estimate.modulus_n_per_m2, foundation_modulus, rel_tol=2e-2)


def test_single_deflection_estimate_matches_known_modulus() -> None:
    load = PointLoad(position_m=0.0, load_newtons=10_000.0)
    foundation_modulus = 50_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    w0 = deflection_at(0.0, [load], foundation_modulus, elastic_modulus, moment_inertia)

    estimate = estimate_track_modulus_single_deflection(
        load_newtons=load.load_newtons,
        deflection_m=w0,
        elastic_modulus_pa=elastic_modulus,
        moment_inertia_m4=moment_inertia,
    )

    assert math.isclose(estimate.modulus_n_per_m2, foundation_modulus, rel_tol=5e-3)


def test_synthesize_track_modulus_from_springs() -> None:
    sleeper_spacing = 0.6
    rail_seat = 90_000_000.0
    pad = 80_000_000.0

    modulus = synthesize_track_modulus_from_springs(
        sleeper_spacing_m=sleeper_spacing,
        rail_seat_stiffness_n_per_m=rail_seat,
        series_stiffnesses_n_per_m=[pad],
    )

    combined = 1.0 / (1.0 / rail_seat + 1.0 / pad)
    expected = combined / sleeper_spacing
    assert math.isclose(modulus, expected, rel_tol=1e-12)


def test_single_deflection_requires_bracket_root() -> None:
    with pytest.raises(ValueError, match="bracket does not contain a root"):
        estimate_track_modulus_single_deflection(
            load_newtons=10_000.0,
            deflection_m=1e-6,
            elastic_modulus_pa=210_000_000_000.0,
            moment_inertia_m4=3.05e-5,
            bracket_n_per_m2=(1.0e9, 1.0e10),
        )


def test_single_deflection_rejects_non_positive_iteration_controls() -> None:
    with pytest.raises(ValueError, match="tolerance must be positive"):
        estimate_track_modulus_single_deflection(
            load_newtons=10_000.0,
            deflection_m=1e-6,
            elastic_modulus_pa=210_000_000_000.0,
            moment_inertia_m4=3.05e-5,
            tolerance=0.0,
        )

    with pytest.raises(ValueError, match="max_iterations must be positive"):
        estimate_track_modulus_single_deflection(
            load_newtons=10_000.0,
            deflection_m=1e-6,
            elastic_modulus_pa=210_000_000_000.0,
            moment_inertia_m4=3.05e-5,
            max_iterations=0,
        )


def test_track_modulus_from_deflection_helpers() -> None:
    load = 110_000.0
    deflection = 0.0022
    spacing = 0.6
    spring_constant = track_spring_constant_from_deflection(
        load_newtons=load,
        deflection_m=deflection,
    )
    modulus = track_modulus_from_spring_constant(
        spring_constant_n_per_m=spring_constant,
        sleeper_spacing_m=spacing,
    )
    direct = track_modulus_from_deflection(
        load_newtons=load,
        deflection_m=deflection,
        sleeper_spacing_m=spacing,
    )

    assert math.isclose(spring_constant, load / deflection, rel_tol=1e-12)
    assert math.isclose(modulus, (load / deflection) / spacing, rel_tol=1e-12)
    assert math.isclose(direct, modulus, rel_tol=1e-12)

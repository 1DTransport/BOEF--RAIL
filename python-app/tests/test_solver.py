import math

import pytest

from core.foundation_profiles import ramp_profile, step_profile
from core.model import PointLoad, deflection_at
from core.solver import (
    build_discrete_supports,
    equivalent_continuous_modulus,
    solve_static_beam_fdm,
    solve_static_beam_timoshenko_fdm,
    solve_two_rail_static_fdm,
)


def _build_grid(length: float, samples: int) -> list[float]:
    step = (2.0 * length) / (samples - 1)
    return [-length + i * step for i in range(samples)]


def _trapezoid_integral(x_values: list[float], values: list[float]) -> float:
    return sum(
        0.5 * (values[i] + values[i + 1]) * (x_values[i + 1] - x_values[i])
        for i in range(len(x_values) - 1)
    )


def _first_derivative(values: list[float], dx: float) -> list[float]:
    derivative = [0.0 for _ in values]
    for i in range(1, len(values) - 1):
        derivative[i] = (values[i + 1] - values[i - 1]) / (2.0 * dx)
    derivative[0] = (values[1] - values[0]) / dx
    derivative[-1] = (values[-1] - values[-2]) / dx
    return derivative


def _second_derivative(values: list[float], dx: float) -> list[float]:
    derivative = [0.0 for _ in values]
    for i in range(1, len(values) - 1):
        derivative[i] = (values[i - 1] - 2.0 * values[i] + values[i + 1]) / (dx**2)
    derivative[0] = (values[0] - 2.0 * values[1] + values[2]) / (dx**2)
    derivative[-1] = (values[-3] - 2.0 * values[-2] + values[-1]) / (dx**2)
    return derivative


def _pasternak_fourier_deflection(
    *,
    x_m: float,
    load_newtons: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
    foundation_modulus_n_per_m2: float,
    pasternak_shear_n: float,
    wave_number_max: float = 200.0,
    samples: int = 20_000,
) -> float:
    """Reference infinite-beam integral for a point load on a Pasternak foundation."""
    ei = elastic_modulus_pa * moment_inertia_m4
    da = wave_number_max / samples
    integral = 0.0
    for i in range(samples + 1):
        wave_number = i * da
        weight = 0.5 if i in (0, samples) else 1.0
        denominator = (
            ei * wave_number**4
            + pasternak_shear_n * wave_number**2
            + foundation_modulus_n_per_m2
        )
        integral += weight * math.cos(wave_number * x_m) / denominator
    return load_newtons * integral * da / math.pi


def test_fdm_matches_closed_form_winkler() -> None:
    loads = [PointLoad(position_m=0.0, load_newtons=10_000.0)]
    foundation_modulus = 40_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    beta = (foundation_modulus / (4.0 * elastic_modulus * moment_inertia)) ** 0.25
    length = 10.0 / beta

    x_values = _build_grid(length, 401)
    solution = solve_static_beam_fdm(
        x_m=x_values,
        loads=loads,
        foundation_modulus_n_per_m2=foundation_modulus,
        elastic_modulus_pa=elastic_modulus,
        moment_inertia_m4=moment_inertia,
    )

    for x in (-1.2, -0.4, 0.0, 0.7, 1.3):
        closed = deflection_at(x, loads, foundation_modulus, elastic_modulus, moment_inertia)
        index = min(range(len(solution.x_m)), key=lambda i: abs(solution.x_m[i] - x))
        numeric = solution.deflection_m[index]
        assert math.isclose(numeric, closed, rel_tol=5e-2, abs_tol=1e-6)


def test_pasternak_reduces_to_winkler_when_shear_zero() -> None:
    loads = [PointLoad(position_m=0.2, load_newtons=8_000.0)]
    foundation_modulus = 32_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    beta = (foundation_modulus / (4.0 * elastic_modulus * moment_inertia)) ** 0.25
    length = 10.0 / beta

    x_values = _build_grid(length, 221)
    solution = solve_static_beam_fdm(
        x_m=x_values,
        loads=loads,
        foundation_modulus_n_per_m2=foundation_modulus,
        elastic_modulus_pa=elastic_modulus,
        moment_inertia_m4=moment_inertia,
        pasternak_shear_n=0.0,
    )

    target = deflection_at(0.4, loads, foundation_modulus, elastic_modulus, moment_inertia)
    index = min(range(len(solution.x_m)), key=lambda i: abs(solution.x_m[i] - 0.4))
    assert math.isclose(solution.deflection_m[index], target, rel_tol=2e-2, abs_tol=1e-6)


def test_pasternak_matches_nonzero_shear_reference() -> None:
    load = PointLoad(position_m=0.2, load_newtons=8_000.0)
    foundation_modulus = 32_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    pasternak_shear = 2_000_000.0
    beta = (foundation_modulus / (4.0 * elastic_modulus * moment_inertia)) ** 0.25
    length = 12.0 / beta

    x_values = _build_grid(length, 801)
    solution = solve_static_beam_fdm(
        x_m=x_values,
        loads=[load],
        foundation_modulus_n_per_m2=foundation_modulus,
        elastic_modulus_pa=elastic_modulus,
        moment_inertia_m4=moment_inertia,
        pasternak_shear_n=pasternak_shear,
    )

    for x in (0.0, 0.2, 0.6, 1.0):
        index = min(range(len(solution.x_m)), key=lambda i: abs(solution.x_m[i] - x))
        reference = _pasternak_fourier_deflection(
            x_m=x - load.position_m,
            load_newtons=load.load_newtons,
            elastic_modulus_pa=elastic_modulus,
            moment_inertia_m4=moment_inertia,
            foundation_modulus_n_per_m2=foundation_modulus,
            pasternak_shear_n=pasternak_shear,
        )
        assert math.isclose(solution.deflection_m[index], reference, rel_tol=1e-2, abs_tol=1e-7)


def test_pasternak_reaction_components_and_coherence() -> None:
    load = PointLoad(position_m=0.0, load_newtons=9_000.0)
    foundation_modulus = 35_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    pasternak_shear = 1_500_000.0
    beta = (foundation_modulus / (4.0 * elastic_modulus * moment_inertia)) ** 0.25
    x_values = _build_grid(12.0 / beta, 801)
    dx = x_values[1] - x_values[0]

    solution = solve_static_beam_fdm(
        x_m=x_values,
        loads=[load],
        foundation_modulus_n_per_m2=foundation_modulus,
        elastic_modulus_pa=elastic_modulus,
        moment_inertia_m4=moment_inertia,
        pasternak_shear_n=pasternak_shear,
    )

    assert solution.slope_rad is not None
    assert solution.winkler_reaction_n_per_m is not None
    assert solution.pasternak_shear_reaction_n_per_m is not None
    curvature = _second_derivative(solution.deflection_m, dx)
    shear_from_moment = _first_derivative(solution.moment_nm, dx)
    for index in range(4, len(x_values) - 4, 31):
        assert math.isclose(
            solution.moment_nm[index],
            -elastic_modulus * moment_inertia * curvature[index],
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        assert math.isclose(solution.shear_n[index], shear_from_moment[index], rel_tol=1e-9, abs_tol=1e-6)
        assert math.isclose(
            solution.winkler_reaction_n_per_m[index],
            foundation_modulus * solution.deflection_m[index],
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        assert math.isclose(
            solution.pasternak_shear_reaction_n_per_m[index],
            -pasternak_shear * curvature[index],
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        assert math.isclose(
            solution.reaction_n_per_m[index],
            solution.winkler_reaction_n_per_m[index]
            + solution.pasternak_shear_reaction_n_per_m[index],
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
    assert math.isclose(
        _trapezoid_integral(solution.x_m, solution.reaction_n_per_m),
        load.load_newtons,
        rel_tol=5e-3,
    )


def test_timoshenko_realistic_shear_close_to_euler_without_fallback() -> None:
    loads = [PointLoad(position_m=0.0, load_newtons=10_000.0)]
    foundation_modulus = 40_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    area_m2 = 7.7e-3
    shear_modulus = elastic_modulus / (2.0 * (1.0 + 0.3))
    kappa = 0.9
    beta = (foundation_modulus / (4.0 * elastic_modulus * moment_inertia)) ** 0.25
    length = 10.0 / beta

    x_values = _build_grid(length, 401)
    euler = solve_static_beam_fdm(
        x_m=x_values,
        loads=loads,
        foundation_modulus_n_per_m2=foundation_modulus,
        elastic_modulus_pa=elastic_modulus,
        moment_inertia_m4=moment_inertia,
    )
    timoshenko = solve_static_beam_timoshenko_fdm(
        x_m=x_values,
        loads=loads,
        foundation_modulus_n_per_m2=foundation_modulus,
        elastic_modulus_pa=elastic_modulus,
        moment_inertia_m4=moment_inertia,
        shear_modulus_pa=shear_modulus,
        shear_correction_factor=kappa,
        area_m2=area_m2,
    )
    idx = min(range(len(x_values)), key=lambda i: abs(x_values[i] - 0.0))
    assert math.isclose(
        timoshenko.deflection_m[idx],
        euler.deflection_m[idx],
        rel_tol=5e-2,
        abs_tol=1e-6,
    )


@pytest.mark.parametrize(
    ("samples", "kappa"),
    [
        (241, 0.7),
        (801, 0.5),
    ],
)
def test_timoshenko_converges_to_euler_across_grid_and_kappa(samples: int, kappa: float) -> None:
    loads = [PointLoad(position_m=0.0, load_newtons=10_000.0)]
    foundation_modulus = 40_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    area_m2 = 7.7e-3
    shear_modulus = elastic_modulus / (2.0 * (1.0 + 0.3))
    beta = (foundation_modulus / (4.0 * elastic_modulus * moment_inertia)) ** 0.25
    length = 10.0 / beta

    x_values = _build_grid(length, samples)
    euler = solve_static_beam_fdm(
        x_m=x_values,
        loads=loads,
        foundation_modulus_n_per_m2=foundation_modulus,
        elastic_modulus_pa=elastic_modulus,
        moment_inertia_m4=moment_inertia,
    )
    timoshenko = solve_static_beam_timoshenko_fdm(
        x_m=x_values,
        loads=loads,
        foundation_modulus_n_per_m2=foundation_modulus,
        elastic_modulus_pa=elastic_modulus,
        moment_inertia_m4=moment_inertia,
        shear_modulus_pa=shear_modulus,
        shear_correction_factor=kappa,
        area_m2=area_m2,
    )

    idx = min(range(len(x_values)), key=lambda i: abs(x_values[i]))
    assert math.isclose(
        timoshenko.deflection_m[idx],
        euler.deflection_m[idx],
        rel_tol=5e-2,
        abs_tol=1e-6,
    )


def test_timoshenko_rotation_shear_and_equilibrium_are_coherent() -> None:
    load = PointLoad(position_m=0.0, load_newtons=10_000.0)
    foundation_modulus = 40_000_000.0
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    area_m2 = 7.7e-3
    shear_modulus = elastic_modulus / (2.0 * (1.0 + 0.3))
    kappa = 0.4
    beta = (foundation_modulus / (4.0 * elastic_modulus * moment_inertia)) ** 0.25
    x_values = _build_grid(12.0 / beta, 601)
    dx = x_values[1] - x_values[0]
    shear_stiffness = kappa * shear_modulus * area_m2

    solution = solve_static_beam_timoshenko_fdm(
        x_m=x_values,
        loads=[load],
        foundation_modulus_n_per_m2=foundation_modulus,
        elastic_modulus_pa=elastic_modulus,
        moment_inertia_m4=moment_inertia,
        shear_modulus_pa=shear_modulus,
        shear_correction_factor=kappa,
        area_m2=area_m2,
    )

    mid = min(range(len(x_values)), key=lambda i: abs(x_values[i]))
    assert solution.deflection_m[mid] > 0.0
    assert solution.rotation_rad is not None
    assert solution.slope_rad is not None
    assert solution.shear_angle_rad is not None
    assert math.isclose(
        _trapezoid_integral(solution.x_m, solution.reaction_n_per_m),
        load.load_newtons,
        rel_tol=5e-3,
    )
    moment_from_rotation = [
        elastic_modulus * moment_inertia * value
        for value in _first_derivative(solution.rotation_rad, dx)
    ]
    for index in range(4, len(x_values) - 4, 29):
        assert math.isclose(
            solution.shear_angle_rad[index],
            solution.rotation_rad[index] - solution.slope_rad[index],
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        assert math.isclose(
            solution.shear_n[index],
            shear_stiffness * solution.shear_angle_rad[index],
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        assert math.isclose(solution.moment_nm[index], moment_from_rotation[index], rel_tol=1e-9, abs_tol=1e-6)


def test_discrete_supports_close_to_equivalent_continuous() -> None:
    loads = [PointLoad(position_m=0.0, load_newtons=9_000.0)]
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    sleeper_spacing = 0.6
    pad_stiffness = 70_000_000.0

    length = 6.0
    x_values = _build_grid(length, 241)
    sleeper_positions = [i * sleeper_spacing for i in range(-8, 9)]
    discrete_supports = build_discrete_supports(
        x_m=x_values,
        sleeper_positions_m=sleeper_positions,
        pad_stiffness_n_per_m=pad_stiffness,
    )
    solution_discrete = solve_static_beam_fdm(
        x_m=x_values,
        loads=loads,
        foundation_modulus_n_per_m2=0.0,
        elastic_modulus_pa=elastic_modulus,
        moment_inertia_m4=moment_inertia,
        discrete_supports=discrete_supports,
    )

    equivalent_k = equivalent_continuous_modulus(
        pad_stiffness_n_per_m=pad_stiffness,
        sleeper_spacing_m=sleeper_spacing,
    )
    solution_continuous = solve_static_beam_fdm(
        x_m=x_values,
        loads=loads,
        foundation_modulus_n_per_m2=equivalent_k,
        elastic_modulus_pa=elastic_modulus,
        moment_inertia_m4=moment_inertia,
    )

    idx = min(range(len(x_values)), key=lambda i: abs(x_values[i] - 0.0))
    diff = abs(solution_discrete.deflection_m[idx] - solution_continuous.deflection_m[idx])
    assert diff / abs(solution_continuous.deflection_m[idx]) < 0.2


def test_two_rail_symmetry_under_equal_loading() -> None:
    loads = [PointLoad(position_m=0.0, load_newtons=12_000.0)]
    elastic_modulus = 210_000_000_000.0
    moment_inertia = 3.05e-5
    foundation_modulus = 35_000_000.0
    coupling_stiffness = 50_000_000.0
    sleeper_spacing = 0.6

    length = 6.0
    x_values = _build_grid(length, 241)
    sleeper_nodes = [
        min(range(len(x_values)), key=lambda i: abs(x_values[i] - pos))
        for pos in [i * sleeper_spacing for i in range(-8, 9)]
    ]

    solution = solve_two_rail_static_fdm(
        x_m=x_values,
        left_loads=loads,
        right_loads=loads,
        foundation_modulus_n_per_m2=foundation_modulus,
        elastic_modulus_pa=elastic_modulus,
        moment_inertia_m4=moment_inertia,
        coupling_stiffness_n_per_m=coupling_stiffness,
        coupling_nodes=sleeper_nodes,
    )

    for left, right in zip(solution.left.deflection_m, solution.right.deflection_m):
        assert math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-9)


def test_two_rail_requires_positive_section_properties() -> None:
    x_values = _build_grid(4.0, 121)
    loads = [PointLoad(position_m=0.0, load_newtons=5_000.0)]

    with pytest.raises(ValueError, match="elastic_modulus_pa must be positive"):
        solve_two_rail_static_fdm(
            x_m=x_values,
            left_loads=loads,
            right_loads=loads,
            foundation_modulus_n_per_m2=35_000_000.0,
            elastic_modulus_pa=0.0,
            moment_inertia_m4=3.05e-5,
            coupling_stiffness_n_per_m=50_000_000.0,
            coupling_nodes=[60],
        )

    with pytest.raises(ValueError, match="moment_inertia_m4 must be positive"):
        solve_two_rail_static_fdm(
            x_m=x_values,
            left_loads=loads,
            right_loads=loads,
            foundation_modulus_n_per_m2=35_000_000.0,
            elastic_modulus_pa=210_000_000_000.0,
            moment_inertia_m4=0.0,
            coupling_stiffness_n_per_m=50_000_000.0,
            coupling_nodes=[60],
        )


def test_non_uniform_profiles_build() -> None:
    x_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    step = step_profile(
        x_m=x_values,
        left_modulus_n_per_m2=20_000_000.0,
        right_modulus_n_per_m2=50_000_000.0,
        step_location_m=0.5,
    )
    ramp = ramp_profile(
        x_m=x_values,
        start_modulus_n_per_m2=20_000_000.0,
        end_modulus_n_per_m2=50_000_000.0,
        ramp_start_m=-1.0,
        ramp_end_m=1.0,
    )
    assert step[0] == 20_000_000.0
    assert step[-1] == 50_000_000.0
    assert ramp[0] == 20_000_000.0
    assert ramp[-1] == 50_000_000.0


@pytest.mark.parametrize("samples", [241, 281])
def test_point_loads_must_be_within_domain(samples: int) -> None:
    x_values = _build_grid(5.0, samples)
    with pytest.raises(ValueError, match="point load position must fall within x_m range"):
        solve_static_beam_fdm(
            x_m=x_values,
            loads=[PointLoad(position_m=10.0, load_newtons=1.0)],
            foundation_modulus_n_per_m2=1.0,
            elastic_modulus_pa=210_000_000_000.0,
            moment_inertia_m4=3.05e-5,
        )


def test_fdm_rejects_non_finite_inputs() -> None:
    x_values = _build_grid(5.0, 241)
    x_values[10] = math.nan
    with pytest.raises(ValueError, match="x_m must contain only finite values"):
        solve_static_beam_fdm(
            x_m=x_values,
            loads=[PointLoad(position_m=0.0, load_newtons=1.0)],
            foundation_modulus_n_per_m2=1.0,
            elastic_modulus_pa=210_000_000_000.0,
            moment_inertia_m4=3.05e-5,
        )

    with pytest.raises(ValueError, match="load_newtons must be finite"):
        solve_static_beam_fdm(
            x_m=_build_grid(5.0, 241),
            loads=[PointLoad(position_m=0.0, load_newtons=math.nan)],
            foundation_modulus_n_per_m2=1.0,
            elastic_modulus_pa=210_000_000_000.0,
            moment_inertia_m4=3.05e-5,
        )

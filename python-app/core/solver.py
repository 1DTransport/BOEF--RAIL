"""Numerical solvers for BOEF track models (static).

All computations are in SI units. Governing equations follow standard BOEF
formulations for Winkler and Pasternak foundations.

References:
- NPTEL "Beam on Elastic Foundation" (Winkler beam, beta definition).
- Cai, Raymond & Bathurst (1994), TRR 1470 (rail BOEF governing equation,
  track modulus context, and two-parameter foundation notes).
- Lamprea-Pineda et al. (2022) review for two-parameter foundation variants.
- Zimmermann equivalent continuous support approach (e.g., Prakoso 2012 notes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import bisect
import math

from core.model import PointLoad


@dataclass(frozen=True)
class StaticBeamSolution:
    """Static beam solution outputs (SI units)."""

    x_m: list[float]
    deflection_m: list[float]
    moment_nm: list[float]
    shear_n: list[float]
    reaction_n_per_m: list[float]
    slope_rad: list[float] | None = None
    rotation_rad: list[float] | None = None
    shear_angle_rad: list[float] | None = None
    winkler_reaction_n_per_m: list[float] | None = None
    pasternak_shear_reaction_n_per_m: list[float] | None = None


@dataclass(frozen=True)
class TwoRailSolution:
    """Static two-rail solution outputs (SI units)."""

    x_m: list[float]
    left: StaticBeamSolution
    right: StaticBeamSolution


def solve_static_beam_fdm(
    *,
    x_m: Sequence[float],
    loads: Sequence[PointLoad],
    foundation_modulus_n_per_m2: float | Sequence[float],
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
    pasternak_shear_n: float = 0.0,
    distributed_load_n_per_m: Sequence[float] | None = None,
    discrete_supports: dict[int, float] | None = None,
) -> StaticBeamSolution:
    """Solve static BOEF using a finite-difference scheme on a uniform grid.

    Governing equation (Pasternak; Winkler when kg=0):
      EI * w''''(x) - kg * w''(x) + ks * w(x) = q(x)

    The discrete system uses a five-point stencil for w'''' and a three-point
    stencil for w''. Boundary conditions approximate decay on an infinite beam
    by enforcing w=0 at the first two and last two nodes.

    References:
    - Cai, Raymond & Bathurst (1994), TRR 1470 (governing equation form).
    - NPTEL "Beam on Elastic Foundation" (Winkler special case).
    """
    _require_positive(elastic_modulus_pa, "elastic_modulus_pa")
    _require_positive(moment_inertia_m4, "moment_inertia_m4")
    _require_non_negative(pasternak_shear_n, "pasternak_shear_n")

    x_values = _validate_uniform_grid(x_m)
    n = len(x_values)
    dx = x_values[1] - x_values[0]

    foundation_profile = _expand_profile(foundation_modulus_n_per_m2, n, "foundation_modulus_n_per_m2")
    _require_all_non_negative(foundation_profile, "foundation_modulus_n_per_m2")

    q_values = [0.0 for _ in range(n)]
    if distributed_load_n_per_m is not None:
        if len(distributed_load_n_per_m) != n:
            raise ValueError("distributed_load_n_per_m must match x_m length")
        q_values = [float(q) for q in distributed_load_n_per_m]

    q_values = _add_point_loads_to_distributed(x_values, dx, q_values, loads)

    discrete_per_length = [0.0 for _ in range(n)]
    if discrete_supports:
        for index, stiffness in discrete_supports.items():
            _require_non_negative(stiffness, "discrete_support_stiffness_n_per_m")
            if index < 0 or index >= n:
                raise ValueError("discrete support index out of range")
            discrete_per_length[index] += stiffness / dx

    total_support = [foundation_profile[i] + discrete_per_length[i] for i in range(n)]

    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    rhs = [float(q) for q in q_values]

    for boundary_index in (0, 1, n - 2, n - 1):
        matrix[boundary_index][boundary_index] = 1.0
        rhs[boundary_index] = 0.0

    coeff_4 = elastic_modulus_pa * moment_inertia_m4 / (dx**4)
    coeff_2 = pasternak_shear_n / (dx**2)

    for i in range(2, n - 2):
        matrix[i][i - 2] = coeff_4
        matrix[i][i - 1] = -4.0 * coeff_4 - coeff_2
        matrix[i][i] = 6.0 * coeff_4 + 2.0 * coeff_2 + total_support[i]
        matrix[i][i + 1] = -4.0 * coeff_4 - coeff_2
        matrix[i][i + 2] = coeff_4

    deflections = _solve_linear_system(matrix, rhs)
    second_derivative = _second_derivative(deflections, dx)
    moments = [-(elastic_modulus_pa * moment_inertia_m4) * value for value in second_derivative]
    shears = _first_derivative(moments, dx)
    slopes = _first_derivative(deflections, dx)

    winkler_reactions = [total_support[i] * deflections[i] for i in range(n)]
    pasternak_reactions = [-pasternak_shear_n * second_derivative[i] for i in range(n)]
    reactions = [winkler_reactions[i] + pasternak_reactions[i] for i in range(n)]

    return StaticBeamSolution(
        x_m=list(x_values),
        deflection_m=deflections,
        moment_nm=moments,
        shear_n=shears,
        reaction_n_per_m=reactions,
        slope_rad=slopes,
        rotation_rad=slopes,
        winkler_reaction_n_per_m=winkler_reactions,
        pasternak_shear_reaction_n_per_m=pasternak_reactions,
    )


def solve_static_beam_timoshenko_fdm(
    *,
    x_m: Sequence[float],
    loads: Sequence[PointLoad],
    foundation_modulus_n_per_m2: float | Sequence[float],
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
    shear_modulus_pa: float,
    shear_correction_factor: float,
    area_m2: float,
    distributed_load_n_per_m: Sequence[float] | None = None,
    discrete_supports: dict[int, float] | None = None,
) -> StaticBeamSolution:
    """Solve static Timoshenko beam on Winkler foundation using finite differences.

    Unknowns are deflection w and rotation phi at each node. The formulation uses:
      EI * phi'' = kappa * G * A * (phi - w')
      (kappa * G * A * (phi - w'))' + q(x) - k_s * w = 0
    """
    _require_positive(elastic_modulus_pa, "elastic_modulus_pa")
    _require_positive(moment_inertia_m4, "moment_inertia_m4")
    _require_positive(shear_modulus_pa, "shear_modulus_pa")
    _require_positive(area_m2, "area_m2")
    _require_positive(shear_correction_factor, "shear_correction_factor")

    x_values = _validate_uniform_grid(x_m)
    n = len(x_values)
    dx = x_values[1] - x_values[0]

    foundation_profile = _expand_profile(
        foundation_modulus_n_per_m2, n, "foundation_modulus_n_per_m2"
    )
    _require_all_non_negative(foundation_profile, "foundation_modulus_n_per_m2")

    q_values = [0.0 for _ in range(n)]
    if distributed_load_n_per_m is not None:
        if len(distributed_load_n_per_m) != n:
            raise ValueError("distributed_load_n_per_m must match x_m length")
        q_values = [float(q) for q in distributed_load_n_per_m]

    q_values = _add_point_loads_to_distributed(x_values, dx, q_values, loads)

    discrete_per_length = [0.0 for _ in range(n)]
    if discrete_supports:
        for index, stiffness in discrete_supports.items():
            _require_non_negative(stiffness, "discrete_support_stiffness_n_per_m")
            if index < 0 or index >= n:
                raise ValueError("discrete support index out of range")
            discrete_per_length[index] += stiffness / dx

    total_support = [foundation_profile[i] + discrete_per_length[i] for i in range(n)]

    size = 2 * n
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    rhs = [0.0 for _ in range(size)]

    shear_stiffness = shear_correction_factor * shear_modulus_pa * area_m2
    boundary_nodes = {0, 1, n - 2, n - 1}
    for i in range(n):
        if i in boundary_nodes:
            matrix[i][i] = 1.0
            rhs[i] = 0.0
            matrix[n + i][n + i] = 1.0
            rhs[n + i] = 0.0
            continue

        # Equation 1: EI * phi'' - kGA * (phi - w') = 0
        row_phi = i
        matrix[row_phi][n + i - 1] = elastic_modulus_pa * moment_inertia_m4 / (dx**2)
        matrix[row_phi][n + i] = -2.0 * elastic_modulus_pa * moment_inertia_m4 / (dx**2) - shear_stiffness
        matrix[row_phi][n + i + 1] = elastic_modulus_pa * moment_inertia_m4 / (dx**2)
        matrix[row_phi][i - 1] = -shear_stiffness / (2.0 * dx)
        matrix[row_phi][i + 1] = shear_stiffness / (2.0 * dx)

        # Equation 2: (kGA * (phi - w'))' + k_s * w = q
        # With constant kGA: kGA*(phi' - w'') + k_s*w = q.
        row_w = n + i
        coeff_phi = shear_stiffness / (2.0 * dx)
        coeff_w = shear_stiffness / (dx**2)
        matrix[row_w][n + i + 1] += coeff_phi
        matrix[row_w][n + i - 1] += -coeff_phi
        matrix[row_w][i + 1] += -coeff_w
        matrix[row_w][i] += 2.0 * coeff_w
        matrix[row_w][i - 1] += -coeff_w
        matrix[row_w][i] += total_support[i]
        rhs[row_w] = q_values[i]

    solution = _solve_linear_system(matrix, rhs)
    deflections = solution[:n]
    rotations = solution[n:]

    slope = _first_derivative(deflections, dx)
    moments = [elastic_modulus_pa * moment_inertia_m4 * value for value in _first_derivative(rotations, dx)]
    shears = [
        shear_stiffness * (rotations[i] - slope[i])
        for i in range(n)
    ]
    shear_angles = [rotations[i] - slope[i] for i in range(n)]
    reactions = [total_support[i] * deflections[i] for i in range(n)]

    return StaticBeamSolution(
        x_m=list(x_values),
        deflection_m=deflections,
        moment_nm=moments,
        shear_n=shears,
        reaction_n_per_m=reactions,
        slope_rad=slope,
        rotation_rad=rotations,
        shear_angle_rad=shear_angles,
        winkler_reaction_n_per_m=reactions,
        pasternak_shear_reaction_n_per_m=[0.0 for _ in range(n)],
    )


def solve_two_rail_static_fdm(
    *,
    x_m: Sequence[float],
    left_loads: Sequence[PointLoad],
    right_loads: Sequence[PointLoad],
    foundation_modulus_n_per_m2: float | Sequence[float],
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
    coupling_stiffness_n_per_m: float,
    coupling_nodes: Sequence[int],
    pasternak_shear_n: float = 0.0,
    distributed_left_n_per_m: Sequence[float] | None = None,
    distributed_right_n_per_m: Sequence[float] | None = None,
    discrete_supports: dict[int, float] | None = None,
) -> TwoRailSolution:
    """Solve a two-rail BOEF model with sleeper coupling springs.

    The coupling is modeled as a spring between left/right rail deflections
    at specified nodes (typically sleeper seats).

    References:
    - Cai, Raymond & Bathurst (1994), TRR 1470 (rail + tie coupling context).
    - Lamprea-Pineda et al. (2022) review (two-rail modeling considerations).
    """
    _require_positive(elastic_modulus_pa, "elastic_modulus_pa")
    _require_positive(moment_inertia_m4, "moment_inertia_m4")
    _require_positive(coupling_stiffness_n_per_m, "coupling_stiffness_n_per_m")
    _require_non_negative(pasternak_shear_n, "pasternak_shear_n")

    x_values = _validate_uniform_grid(x_m)
    n = len(x_values)
    dx = x_values[1] - x_values[0]

    foundation_profile = _expand_profile(foundation_modulus_n_per_m2, n, "foundation_modulus_n_per_m2")
    _require_all_non_negative(foundation_profile, "foundation_modulus_n_per_m2")

    left_q = [0.0 for _ in range(n)]
    if distributed_left_n_per_m is not None:
        if len(distributed_left_n_per_m) != n:
            raise ValueError("distributed_left_n_per_m must match x_m length")
        left_q = [float(q) for q in distributed_left_n_per_m]

    right_q = [0.0 for _ in range(n)]
    if distributed_right_n_per_m is not None:
        if len(distributed_right_n_per_m) != n:
            raise ValueError("distributed_right_n_per_m must match x_m length")
        right_q = [float(q) for q in distributed_right_n_per_m]

    left_q = _add_point_loads_to_distributed(x_values, dx, left_q, left_loads)
    right_q = _add_point_loads_to_distributed(x_values, dx, right_q, right_loads)

    discrete_per_length = [0.0 for _ in range(n)]
    if discrete_supports:
        for index, stiffness in discrete_supports.items():
            _require_non_negative(stiffness, "discrete_support_stiffness_n_per_m")
            if index < 0 or index >= n:
                raise ValueError("discrete support index out of range")
            discrete_per_length[index] += stiffness / dx

    total_support = [foundation_profile[i] + discrete_per_length[i] for i in range(n)]

    size = 2 * n
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    rhs = left_q + right_q

    coeff_4 = elastic_modulus_pa * moment_inertia_m4 / (dx**4)
    coeff_2 = pasternak_shear_n / (dx**2)

    for boundary_index in (0, 1, n - 2, n - 1):
        matrix[boundary_index][boundary_index] = 1.0
        rhs[boundary_index] = 0.0

    for boundary_index in (n, n + 1, size - 2, size - 1):
        matrix[boundary_index][boundary_index] = 1.0
        rhs[boundary_index] = 0.0

    for i in range(2, n - 2):
        base = i
        matrix[base][base - 2] = coeff_4
        matrix[base][base - 1] = -4.0 * coeff_4 - coeff_2
        matrix[base][base] = 6.0 * coeff_4 + 2.0 * coeff_2 + total_support[i]
        matrix[base][base + 1] = -4.0 * coeff_4 - coeff_2
        matrix[base][base + 2] = coeff_4

        right_base = n + i
        matrix[right_base][right_base - 2] = coeff_4
        matrix[right_base][right_base - 1] = -4.0 * coeff_4 - coeff_2
        matrix[right_base][right_base] = 6.0 * coeff_4 + 2.0 * coeff_2 + total_support[i]
        matrix[right_base][right_base + 1] = -4.0 * coeff_4 - coeff_2
        matrix[right_base][right_base + 2] = coeff_4

    coupling_per_length = coupling_stiffness_n_per_m / dx
    for node in coupling_nodes:
        if node < 0 or node >= n:
            raise ValueError("coupling node index out of range")
        left_index = node
        right_index = n + node
        matrix[left_index][left_index] += coupling_per_length
        matrix[right_index][right_index] += coupling_per_length
        matrix[left_index][right_index] -= coupling_per_length
        matrix[right_index][left_index] -= coupling_per_length

    solution = _solve_linear_system(matrix, rhs)

    left_deflection = solution[:n]
    right_deflection = solution[n:]

    left = _build_solution_from_deflection(
        x_values,
        left_deflection,
        total_support,
        elastic_modulus_pa,
        moment_inertia_m4,
        pasternak_shear_n,
    )
    right = _build_solution_from_deflection(
        x_values,
        right_deflection,
        total_support,
        elastic_modulus_pa,
        moment_inertia_m4,
        pasternak_shear_n,
    )

    return TwoRailSolution(x_m=list(x_values), left=left, right=right)


def build_discrete_supports(
    *,
    x_m: Sequence[float],
    sleeper_positions_m: Sequence[float],
    pad_stiffness_n_per_m: float,
) -> dict[int, float]:
    """Map sleeper positions to nearest grid nodes with pad stiffness.

    The returned dictionary maps node indices to discrete spring stiffness
    values (N/m). Use equivalent_continuous_modulus for Zimmermann-style
    verification comparisons.

    References:
    - Zimmermann equivalent continuous support approach (Prakoso 2012 notes).
    """
    _require_positive(pad_stiffness_n_per_m, "pad_stiffness_n_per_m")
    x_values = _validate_uniform_grid(x_m)
    dx = x_values[1] - x_values[0]
    supports: dict[int, float] = {}
    for position in sleeper_positions_m:
        index = bisect.bisect_left(x_values, position)
        if index == len(x_values):
            index = len(x_values) - 1
        if index > 0 and abs(x_values[index - 1] - position) < abs(x_values[index] - position):
            index = index - 1
        if abs(x_values[index] - position) > 0.51 * dx:
            raise ValueError("sleeper position must align with grid spacing")
        supports[index] = supports.get(index, 0.0) + pad_stiffness_n_per_m
    return supports


def equivalent_continuous_modulus(
    *,
    pad_stiffness_n_per_m: float,
    sleeper_spacing_m: float,
) -> float:
    """Compute equivalent continuous modulus for discrete sleeper supports.

    Uses the Zimmermann-style equivalence: k_eq = k_s / a, where k_s is rail-seat
    stiffness (N/m) and a is sleeper spacing (m).

    References:
    - Zimmermann equivalent continuous support approach (Prakoso 2012 notes).
    """
    _require_positive(pad_stiffness_n_per_m, "pad_stiffness_n_per_m")
    _require_positive(sleeper_spacing_m, "sleeper_spacing_m")
    return pad_stiffness_n_per_m / sleeper_spacing_m


def _build_solution_from_deflection(
    x_values: Sequence[float],
    deflection: Sequence[float],
    support_profile: Sequence[float],
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
    pasternak_shear_n: float,
) -> StaticBeamSolution:
    dx = x_values[1] - x_values[0]
    second_derivative = _second_derivative(deflection, dx)
    moments = [-(elastic_modulus_pa * moment_inertia_m4) * value for value in second_derivative]
    shears = _first_derivative(moments, dx)
    slopes = _first_derivative(deflection, dx)
    winkler_reactions = [support_profile[i] * deflection[i] for i in range(len(deflection))]
    pasternak_reactions = [-pasternak_shear_n * second_derivative[i] for i in range(len(deflection))]
    reactions = [winkler_reactions[i] + pasternak_reactions[i] for i in range(len(deflection))]
    return StaticBeamSolution(
        x_m=list(x_values),
        deflection_m=list(deflection),
        moment_nm=moments,
        shear_n=shears,
        reaction_n_per_m=reactions,
        slope_rad=slopes,
        rotation_rad=slopes,
        winkler_reaction_n_per_m=winkler_reactions,
        pasternak_shear_reaction_n_per_m=pasternak_reactions,
    )


def _validate_uniform_grid(x_m: Sequence[float]) -> list[float]:
    if len(x_m) < 5:
        raise ValueError("x_m must contain at least 5 points")
    x_values = [float(x) for x in x_m]
    if any(not math.isfinite(x) for x in x_values):
        raise ValueError("x_m must contain only finite values")
    dx = x_values[1] - x_values[0]
    if dx <= 0:
        raise ValueError("x_m must be strictly increasing")
    for i in range(1, len(x_values) - 1):
        delta = x_values[i + 1] - x_values[i]
        if not math.isclose(delta, dx, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("x_m must be uniformly spaced")
    return x_values


def _expand_profile(profile: float | Sequence[float], n: int, name: str) -> list[float]:
    if isinstance(profile, (int, float)):
        return [float(profile) for _ in range(n)]
    values = [float(value) for value in profile]
    if len(values) != n:
        raise ValueError(f"{name} must have length {n}")
    return values


def _add_point_loads_to_distributed(
    x_values: Sequence[float],
    dx: float,
    q_values: list[float],
    loads: Sequence[PointLoad],
) -> list[float]:
    if not loads:
        return q_values
    for load in loads:
        if not math.isfinite(load.load_newtons):
            raise ValueError("load_newtons must be finite")
        if not math.isfinite(load.position_m):
            raise ValueError("load position must be finite")
        if load.load_newtons < 0:
            raise ValueError("load_newtons must be non-negative")
        if load.position_m < x_values[0] or load.position_m > x_values[-1]:
            raise ValueError("point load position must fall within x_m range")
        index = bisect.bisect_right(x_values, load.position_m) - 1
        if index >= len(x_values) - 1:
            index = len(x_values) - 2
        x0 = x_values[index]
        x1 = x_values[index + 1]
        if x1 == x0:
            raise ValueError("x_m spacing must be positive")
        t = (load.position_m - x0) / (x1 - x0)
        q_values[index] += load.load_newtons * (1.0 - t) / dx
        q_values[index + 1] += load.load_newtons * t / dx
    return q_values


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")
    a = [row[:] for row in matrix]
    b = rhs[:]

    for pivot in range(n):
        max_row = max(range(pivot, n), key=lambda r: abs(a[r][pivot]))
        if math.isclose(a[max_row][pivot], 0.0, abs_tol=1e-12):
            raise ValueError("singular matrix")
        if max_row != pivot:
            a[pivot], a[max_row] = a[max_row], a[pivot]
            b[pivot], b[max_row] = b[max_row], b[pivot]

        pivot_val = a[pivot][pivot]
        for col in range(pivot, n):
            a[pivot][col] /= pivot_val
        b[pivot] /= pivot_val

        for row in range(pivot + 1, n):
            factor = a[row][pivot]
            if math.isclose(factor, 0.0, abs_tol=1e-14):
                continue
            for col in range(pivot, n):
                a[row][col] -= factor * a[pivot][col]
            b[row] -= factor * b[pivot]

    solution = [0.0 for _ in range(n)]
    for row in range(n - 1, -1, -1):
        solution[row] = b[row] - sum(a[row][col] * solution[col] for col in range(row + 1, n))
    return solution


def _first_derivative(values: Sequence[float], dx: float) -> list[float]:
    n = len(values)
    derivative = [0.0 for _ in range(n)]
    for i in range(1, n - 1):
        derivative[i] = (values[i + 1] - values[i - 1]) / (2.0 * dx)
    derivative[0] = (values[1] - values[0]) / dx
    derivative[-1] = (values[-1] - values[-2]) / dx
    return derivative


def _second_derivative(values: Sequence[float], dx: float) -> list[float]:
    n = len(values)
    derivative = [0.0 for _ in range(n)]
    for i in range(1, n - 1):
        derivative[i] = (values[i - 1] - 2.0 * values[i] + values[i + 1]) / (dx**2)
    derivative[0] = (values[0] - 2.0 * values[1] + values[2]) / (dx**2)
    derivative[-1] = (values[-3] - 2.0 * values[-2] + values[-1]) / (dx**2)
    return derivative


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_all_non_negative(values: Iterable[float], name: str) -> None:
    if any(value < 0 for value in values):
        raise ValueError(f"{name} must be non-negative")

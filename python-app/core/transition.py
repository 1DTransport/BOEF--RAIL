"""Transition-zone performance metrics for static BOEF analysis (SI units)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence


class TransitionProfileType(str, Enum):
    UNIFORM = "uniform"
    STEP = "step"
    RAMP = "ramp"
    EXPONENTIAL = "exponential"
    SEGMENT = "segment"


class TransitionRunMode(str, Enum):
    SINGLE = "single"
    ENVELOPE = "envelope"


TRANSITION_ENERGY_METHOD = "static_winkler_elastic_strain_energy"
TRANSITION_ENERGY_EQUATIONS = (
    "u_rail=M^2/(2EI); "
    "u_foundation=0.5*k(x)*w^2; "
    "u_total=u_rail+u_foundation; "
    "U=int(u)dx"
)
TRANSITION_ENERGY_SCOPE = "post_processing_only_winkler_static_response"


@dataclass(frozen=True)
class TransitionMetrics:
    delta_w_s_m: float
    delta_w_s_position_m: float
    delta_w_1m_m: float
    delta_w_1m_position_m: float
    curvature_max_per_m: float
    curvature_max_position_m: float
    moment_max_nm: float
    moment_max_position_m: float
    energy_bending_j: float
    reaction_gradient_max_n_per_m2: float
    reaction_gradient_position_m: float
    sleeper_load_max_n: float
    sleeper_load_position_m: float


@dataclass(frozen=True)
class TransitionEnergyMetrics:
    energy_rail_j: float
    energy_foundation_j: float
    energy_total_j: float
    energy_partition_eta: float
    u_total_max_j_per_m: float
    u_total_max_position_m: float
    du_dx_max_j_per_m2: float
    du_dx_max_position_m: float
    window_length_target_m: float
    window_energy_max_j: float
    window_energy_max_position_m: float
    window_avg_max_j_per_m: float
    window_avg_max_position_m: float
    window_effective_length_min_m: float
    window_effective_length_max_m: float
    is_envelope_upper_bound: bool
    boundary_peak_flag: bool
    boundary_gradient_peak_flag: bool
    p_ref_n: float | None = None
    energy_total_over_p_ref_m: float | None = None
    u_total_max_over_p_ref: float | None = None
    energy_method: str = TRANSITION_ENERGY_METHOD
    energy_equations: str = TRANSITION_ENERGY_EQUATIONS
    energy_scope: str = TRANSITION_ENERGY_SCOPE


@dataclass(frozen=True)
class TransitionEnergySeries:
    u_rail_j_per_m: list[float]
    u_foundation_j_per_m: list[float]
    u_total_j_per_m: list[float]
    du_dx_j_per_m2: list[float]
    window_energy_j: list[float]
    window_avg_j_per_m: list[float]
    window_effective_length_m: list[float]


@dataclass(frozen=True)
class TransitionSeries:
    x_m: list[float]
    k_profile_n_per_m2: list[float]
    deflection_m: list[float] | None = None
    moment_nm: list[float] | None = None
    shear_n: list[float] | None = None
    reaction_n_per_m: list[float] | None = None
    deflection_max_m: list[float] | None = None
    deflection_min_m: list[float] | None = None
    moment_max_nm: list[float] | None = None
    moment_min_nm: list[float] | None = None
    shear_max_n: list[float] | None = None
    shear_min_n: list[float] | None = None
    reaction_max_n_per_m: list[float] | None = None
    reaction_min_n_per_m: list[float] | None = None


@dataclass(frozen=True)
class TransitionRunResult:
    mode: TransitionRunMode
    profile_type: TransitionProfileType
    k1_n_per_m2: float
    k2_n_per_m2: float | None
    transition_length_m: float | None
    segment_length_m: float | None
    domain_length_m: float
    metrics: TransitionMetrics
    series: TransitionSeries
    template_name: str | None = None
    preset_name: str | None = None
    k_units: str = "N/m^2"
    k_representation: str = "continuous_per_unit_length"
    foundation_reaction_law: str = "q_f(x)=k(x)w(x) [N/m]"
    transition_metrics_schema_version: int = 2
    energy_metrics: TransitionEnergyMetrics | None = None
    energy_series: TransitionEnergySeries | None = None


def build_transition_profile(
    *,
    x_values: Sequence[float],
    profile_type: TransitionProfileType,
    k1_n_per_m2: float,
    k2_n_per_m2: float | None,
    transition_length_m: float | None,
    segment_length_m: float | None,
) -> list[float]:
    if k1_n_per_m2 <= 0:
        raise ValueError("k1_n_per_m2 must be positive")
    if profile_type == TransitionProfileType.UNIFORM:
        return [k1_n_per_m2 for _ in x_values]
    if k2_n_per_m2 is None or k2_n_per_m2 <= 0:
        raise ValueError("k2_n_per_m2 must be positive for non-uniform profiles")
    if profile_type == TransitionProfileType.STEP:
        return [k1_n_per_m2 if x < 0 else k2_n_per_m2 for x in x_values]
    if profile_type == TransitionProfileType.RAMP:
        if transition_length_m is None or transition_length_m <= 0:
            raise ValueError("transition_length_m must be positive for ramp profile")
        values: list[float] = []
        for x in x_values:
            if x < 0:
                values.append(k1_n_per_m2)
            elif x <= transition_length_m:
                ratio = x / transition_length_m
                values.append(k1_n_per_m2 + (k2_n_per_m2 - k1_n_per_m2) * ratio)
            else:
                values.append(k2_n_per_m2)
        return values
    if profile_type == TransitionProfileType.EXPONENTIAL:
        if transition_length_m is None or transition_length_m <= 0:
            raise ValueError("transition_length_m must be positive for exponential profile")
        values = []
        for x in x_values:
            if x < 0:
                values.append(k1_n_per_m2)
            else:
                values.append(
                    k1_n_per_m2 + (k2_n_per_m2 - k1_n_per_m2) * (1.0 - math.exp(-x / transition_length_m))
                )
        return values
    if profile_type == TransitionProfileType.SEGMENT:
        if segment_length_m is None or segment_length_m <= 0:
            raise ValueError("segment_length_m must be positive for segment profile")
        half = segment_length_m / 2.0
        return [k2_n_per_m2 if abs(x) <= half else k1_n_per_m2 for x in x_values]
    raise ValueError(f"Unsupported profile type: {profile_type}")


def compute_metrics_from_series(
    *,
    x_values: Sequence[float],
    deflection_m: Sequence[float],
    moment_nm: Sequence[float],
    reaction_n_per_m: Sequence[float],
    sleeper_positions_m: Sequence[float],
    sleeper_loads_n: Sequence[float],
    sleeper_spacing_m: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
) -> TransitionMetrics:
    _require_series_lengths(x_values, deflection_m, moment_nm, reaction_n_per_m)
    delta_w_s, delta_w_s_x = _max_delta_w(x_values, deflection_m, sleeper_spacing_m)
    delta_w_1m, delta_w_1m_x = _max_delta_w(x_values, deflection_m, 1.0)

    curvature = [moment / (elastic_modulus_pa * moment_inertia_m4) for moment in moment_nm]
    curvature_max, curvature_x = _max_abs_with_position(x_values, curvature)
    moment_max, moment_x = _max_abs_with_position(x_values, moment_nm)
    energy = _integrate_energy(x_values, moment_nm, elastic_modulus_pa, moment_inertia_m4)
    grad_p, grad_x = _max_gradient(x_values, reaction_n_per_m)
    sleeper_max, sleeper_x = _max_abs_with_position(sleeper_positions_m, sleeper_loads_n)

    return TransitionMetrics(
        delta_w_s_m=delta_w_s,
        delta_w_s_position_m=delta_w_s_x,
        delta_w_1m_m=delta_w_1m,
        delta_w_1m_position_m=delta_w_1m_x,
        curvature_max_per_m=curvature_max,
        curvature_max_position_m=curvature_x,
        moment_max_nm=moment_max,
        moment_max_position_m=moment_x,
        energy_bending_j=energy,
        reaction_gradient_max_n_per_m2=grad_p,
        reaction_gradient_position_m=grad_x,
        sleeper_load_max_n=sleeper_max,
        sleeper_load_position_m=sleeper_x,
    )


def compute_energy_from_series(
    *,
    x_values: Sequence[float],
    k_profile_n_per_m2: Sequence[float],
    deflection_m: Sequence[float],
    moment_nm: Sequence[float],
    sleeper_spacing_m: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
    p_ref_n: float | None = None,
    gradient_smoothing_points: int = 0,
) -> tuple[TransitionEnergyMetrics, TransitionEnergySeries]:
    _require_energy_lengths(x_values, k_profile_n_per_m2, deflection_m, moment_nm)
    _require_monotonic_increasing(x_values)
    if sleeper_spacing_m <= 0:
        raise ValueError("sleeper_spacing_m must be positive")
    if elastic_modulus_pa <= 0 or moment_inertia_m4 <= 0:
        raise ValueError("elastic_modulus_pa and moment_inertia_m4 must be positive")

    ei = elastic_modulus_pa * moment_inertia_m4
    u_rail = [moment * moment / (2.0 * ei) for moment in moment_nm]
    u_foundation = [0.5 * k * w * w for k, w in zip(k_profile_n_per_m2, deflection_m)]
    return _build_energy_outputs(
        x_values=x_values,
        u_rail=u_rail,
        u_foundation=u_foundation,
        window_length_target_m=sleeper_spacing_m,
        is_envelope_upper_bound=False,
        p_ref_n=p_ref_n,
        gradient_smoothing_points=gradient_smoothing_points,
    )


def compute_metrics_from_envelope(
    *,
    x_values: Sequence[float],
    deflection_max_m: Sequence[float],
    deflection_min_m: Sequence[float],
    moment_max_nm: Sequence[float],
    moment_min_nm: Sequence[float],
    reaction_max_n_per_m: Sequence[float],
    reaction_min_n_per_m: Sequence[float],
    sleeper_positions_m: Sequence[float],
    sleeper_loads_max_n: Sequence[float],
    sleeper_spacing_m: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
) -> TransitionMetrics:
    _require_series_lengths(x_values, deflection_max_m, moment_max_nm, reaction_max_n_per_m)
    delta_w_s_max, delta_w_s_x = _max_delta_w(x_values, deflection_max_m, sleeper_spacing_m)
    delta_w_s_min, delta_w_s_x_min = _max_delta_w(x_values, deflection_min_m, sleeper_spacing_m)
    delta_w_s = max(delta_w_s_max, delta_w_s_min)
    delta_w_s_x = delta_w_s_x if delta_w_s_max >= delta_w_s_min else delta_w_s_x_min

    delta_w_1m_max, delta_w_1m_x = _max_delta_w(x_values, deflection_max_m, 1.0)
    delta_w_1m_min, delta_w_1m_x_min = _max_delta_w(x_values, deflection_min_m, 1.0)
    delta_w_1m = max(delta_w_1m_max, delta_w_1m_min)
    delta_w_1m_x = delta_w_1m_x if delta_w_1m_max >= delta_w_1m_min else delta_w_1m_x_min

    curvature_max_val, curvature_x = _max_abs_with_position(
        x_values,
        [m / (elastic_modulus_pa * moment_inertia_m4) for m in moment_max_nm],
        [m / (elastic_modulus_pa * moment_inertia_m4) for m in moment_min_nm],
    )
    moment_max_val, moment_x = _max_abs_with_position(x_values, moment_max_nm, moment_min_nm)
    abs_moment = [max(abs(m_max), abs(m_min)) for m_max, m_min in zip(moment_max_nm, moment_min_nm)]
    energy = _integrate_energy(x_values, abs_moment, elastic_modulus_pa, moment_inertia_m4)

    grad_max, grad_x_max = _max_gradient(x_values, reaction_max_n_per_m)
    grad_min, grad_x_min = _max_gradient(x_values, reaction_min_n_per_m)
    grad_p = max(grad_max, grad_min)
    grad_x = grad_x_max if grad_max >= grad_min else grad_x_min

    sleeper_max, sleeper_x = _max_abs_with_position(sleeper_positions_m, sleeper_loads_max_n)

    return TransitionMetrics(
        delta_w_s_m=delta_w_s,
        delta_w_s_position_m=delta_w_s_x,
        delta_w_1m_m=delta_w_1m,
        delta_w_1m_position_m=delta_w_1m_x,
        curvature_max_per_m=curvature_max_val,
        curvature_max_position_m=curvature_x,
        moment_max_nm=moment_max_val,
        moment_max_position_m=moment_x,
        energy_bending_j=energy,
        reaction_gradient_max_n_per_m2=grad_p,
        reaction_gradient_position_m=grad_x,
        sleeper_load_max_n=sleeper_max,
        sleeper_load_position_m=sleeper_x,
    )


def compute_energy_from_envelope(
    *,
    x_values: Sequence[float],
    k_profile_n_per_m2: Sequence[float],
    deflection_max_m: Sequence[float],
    deflection_min_m: Sequence[float],
    moment_max_nm: Sequence[float],
    moment_min_nm: Sequence[float],
    sleeper_spacing_m: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
    p_ref_n: float | None = None,
    gradient_smoothing_points: int = 0,
) -> tuple[TransitionEnergyMetrics, TransitionEnergySeries]:
    _require_energy_lengths(x_values, k_profile_n_per_m2, deflection_max_m, moment_max_nm)
    if len(deflection_min_m) != len(x_values):
        raise ValueError("deflection_min_m length must match x_values")
    if len(moment_min_nm) != len(x_values):
        raise ValueError("moment_min_nm length must match x_values")
    _require_monotonic_increasing(x_values)
    if sleeper_spacing_m <= 0:
        raise ValueError("sleeper_spacing_m must be positive")
    if elastic_modulus_pa <= 0 or moment_inertia_m4 <= 0:
        raise ValueError("elastic_modulus_pa and moment_inertia_m4 must be positive")

    ei = elastic_modulus_pa * moment_inertia_m4
    abs_moment = [max(abs(m_max), abs(m_min)) for m_max, m_min in zip(moment_max_nm, moment_min_nm)]
    abs_deflection = [max(abs(w_max), abs(w_min)) for w_max, w_min in zip(deflection_max_m, deflection_min_m)]
    u_rail = [moment * moment / (2.0 * ei) for moment in abs_moment]
    u_foundation = [0.5 * k * w * w for k, w in zip(k_profile_n_per_m2, abs_deflection)]
    return _build_energy_outputs(
        x_values=x_values,
        u_rail=u_rail,
        u_foundation=u_foundation,
        window_length_target_m=sleeper_spacing_m,
        is_envelope_upper_bound=True,
        p_ref_n=p_ref_n,
        gradient_smoothing_points=gradient_smoothing_points,
    )


def build_series_from_single(
    *,
    x_values: Sequence[float],
    k_profile_n_per_m2: Sequence[float],
    deflection_m: Sequence[float],
    moment_nm: Sequence[float],
    reaction_n_per_m: Sequence[float],
    shear_n: Sequence[float] | None = None,
) -> TransitionSeries:
    return TransitionSeries(
        x_m=list(x_values),
        k_profile_n_per_m2=list(k_profile_n_per_m2),
        deflection_m=list(deflection_m),
        moment_nm=list(moment_nm),
        shear_n=list(shear_n) if shear_n is not None else None,
        reaction_n_per_m=list(reaction_n_per_m),
    )


def build_series_from_envelope(
    *,
    x_values: Sequence[float],
    k_profile_n_per_m2: Sequence[float],
    deflection_max_m: Sequence[float],
    deflection_min_m: Sequence[float],
    moment_max_nm: Sequence[float],
    moment_min_nm: Sequence[float],
    reaction_max_n_per_m: Sequence[float],
    reaction_min_n_per_m: Sequence[float],
    shear_max_n: Sequence[float] | None = None,
    shear_min_n: Sequence[float] | None = None,
) -> TransitionSeries:
    return TransitionSeries(
        x_m=list(x_values),
        k_profile_n_per_m2=list(k_profile_n_per_m2),
        deflection_max_m=list(deflection_max_m),
        deflection_min_m=list(deflection_min_m),
        moment_max_nm=list(moment_max_nm),
        moment_min_nm=list(moment_min_nm),
        shear_max_n=list(shear_max_n) if shear_max_n is not None else None,
        shear_min_n=list(shear_min_n) if shear_min_n is not None else None,
        reaction_max_n_per_m=list(reaction_max_n_per_m),
        reaction_min_n_per_m=list(reaction_min_n_per_m),
    )


def _max_delta_w(
    x_values: Sequence[float],
    deflection_m: Sequence[float],
    offset_m: float,
) -> tuple[float, float]:
    if offset_m <= 0:
        raise ValueError("offset_m must be positive")
    max_delta = 0.0
    max_x = x_values[0] if x_values else 0.0
    for i, x in enumerate(x_values):
        target = x + offset_m
        if target > x_values[-1]:
            break
        w_target = _linear_interpolate(x_values, deflection_m, target)
        delta = abs(deflection_m[i] - w_target)
        if delta > max_delta:
            max_delta = delta
            max_x = x
    return max_delta, max_x


def _linear_interpolate(x_values: Sequence[float], y_values: Sequence[float], x: float) -> float:
    if not x_values:
        return 0.0
    if x <= x_values[0]:
        return y_values[0]
    if x >= x_values[-1]:
        return y_values[-1]
    left = 0
    right = len(x_values) - 1
    while right - left > 1:
        mid = (left + right) // 2
        if x_values[mid] <= x:
            left = mid
        else:
            right = mid
    x0 = x_values[left]
    x1 = x_values[right]
    y0 = y_values[left]
    y1 = y_values[right]
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _integrate_energy(
    x_values: Sequence[float],
    moment_nm: Sequence[float],
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
) -> float:
    if len(x_values) != len(moment_nm) or len(x_values) < 2:
        return 0.0
    total = 0.0
    for i in range(len(x_values) - 1):
        dx = x_values[i + 1] - x_values[i]
        m0 = moment_nm[i]
        m1 = moment_nm[i + 1]
        total += 0.5 * (m0 * m0 + m1 * m1) * dx
    return total / (2.0 * elastic_modulus_pa * moment_inertia_m4)


def _max_gradient(
    x_values: Sequence[float],
    values: Sequence[float],
) -> tuple[float, float]:
    if len(x_values) != len(values) or len(values) < 2:
        return 0.0, x_values[0] if x_values else 0.0
    max_grad = 0.0
    max_x = x_values[0]
    for i in range(1, len(values) - 1):
        dx = x_values[i + 1] - x_values[i - 1]
        if dx == 0:
            continue
        grad = abs((values[i + 1] - values[i - 1]) / dx)
        if grad > max_grad:
            max_grad = grad
            max_x = x_values[i]
    return max_grad, max_x


def _max_abs_with_position(
    x_values: Sequence[float],
    values: Sequence[float],
    alt_values: Sequence[float] | None = None,
) -> tuple[float, float]:
    max_val = 0.0
    max_x = x_values[0] if x_values else 0.0
    for i, x in enumerate(x_values):
        candidates = [values[i]]
        if alt_values is not None:
            candidates.append(alt_values[i])
        for value in candidates:
            if abs(value) > abs(max_val):
                max_val = value
                max_x = x
    return max_val, max_x


def _require_series_lengths(
    x_values: Sequence[float],
    deflection_m: Sequence[float],
    moment_nm: Sequence[float],
    reaction_n_per_m: Sequence[float],
) -> None:
    if not x_values:
        raise ValueError("x_values must not be empty")
    if len(deflection_m) != len(x_values):
        raise ValueError("deflection_m length must match x_values")
    if len(moment_nm) != len(x_values):
        raise ValueError("moment_nm length must match x_values")
    if len(reaction_n_per_m) != len(x_values):
        raise ValueError("reaction_n_per_m length must match x_values")


def _require_energy_lengths(
    x_values: Sequence[float],
    k_profile_n_per_m2: Sequence[float],
    deflection_m: Sequence[float],
    moment_nm: Sequence[float],
) -> None:
    if not x_values:
        raise ValueError("x_values must not be empty")
    if len(k_profile_n_per_m2) != len(x_values):
        raise ValueError("k_profile_n_per_m2 length must match x_values")
    if len(deflection_m) != len(x_values):
        raise ValueError("deflection_m length must match x_values")
    if len(moment_nm) != len(x_values):
        raise ValueError("moment_nm length must match x_values")


def _require_monotonic_increasing(x_values: Sequence[float]) -> None:
    if len(x_values) < 2:
        raise ValueError("x_values must contain at least 2 points")
    for i in range(len(x_values) - 1):
        if x_values[i + 1] <= x_values[i]:
            raise ValueError("x_values must be strictly increasing")


def _build_energy_outputs(
    *,
    x_values: Sequence[float],
    u_rail: Sequence[float],
    u_foundation: Sequence[float],
    window_length_target_m: float,
    is_envelope_upper_bound: bool,
    p_ref_n: float | None,
    gradient_smoothing_points: int,
) -> tuple[TransitionEnergyMetrics, TransitionEnergySeries]:
    u_total = [u_r + u_f for u_r, u_f in zip(u_rail, u_foundation)]
    u_for_gradient = _smooth_series(u_total, gradient_smoothing_points)
    du_dx = _first_derivative_nonuniform(x_values, u_for_gradient)
    window_energy, window_avg, window_effective_len = _compute_windowed_energy(
        x_values=x_values,
        values=u_total,
        window_length_target_m=window_length_target_m,
    )

    energy_rail = _integrate_trapezoid(x_values, u_rail)
    energy_foundation = _integrate_trapezoid(x_values, u_foundation)
    energy_total = energy_rail + energy_foundation
    energy_partition_eta = energy_foundation / energy_total if energy_total > 0.0 else 0.0

    u_total_max, u_total_max_x, u_total_max_i = _max_value_with_position(x_values, u_total)
    du_dx_max, du_dx_max_x, du_dx_max_i = _max_abs_value_with_position(x_values, du_dx)

    window_energy_max, window_energy_max_x, _ = _max_value_with_position(x_values, window_energy)
    window_avg_max, window_avg_max_x, _ = _max_value_with_position(x_values, window_avg)
    window_effective_length_min = min(window_effective_len) if window_effective_len else 0.0
    window_effective_length_max = max(window_effective_len) if window_effective_len else 0.0

    boundary_peak_flag = u_total_max_i in (0, len(x_values) - 1)
    boundary_gradient_peak_flag = du_dx_max_i in (0, len(x_values) - 1)

    energy_total_over_p_ref_m: float | None = None
    u_total_max_over_p_ref: float | None = None
    if p_ref_n is not None and p_ref_n > 0.0:
        energy_total_over_p_ref_m = energy_total / p_ref_n
        u_total_max_over_p_ref = u_total_max / p_ref_n

    energy_metrics = TransitionEnergyMetrics(
        energy_rail_j=energy_rail,
        energy_foundation_j=energy_foundation,
        energy_total_j=energy_total,
        energy_partition_eta=energy_partition_eta,
        u_total_max_j_per_m=u_total_max,
        u_total_max_position_m=u_total_max_x,
        du_dx_max_j_per_m2=du_dx_max,
        du_dx_max_position_m=du_dx_max_x,
        window_length_target_m=window_length_target_m,
        window_energy_max_j=window_energy_max,
        window_energy_max_position_m=window_energy_max_x,
        window_avg_max_j_per_m=window_avg_max,
        window_avg_max_position_m=window_avg_max_x,
        window_effective_length_min_m=window_effective_length_min,
        window_effective_length_max_m=window_effective_length_max,
        is_envelope_upper_bound=is_envelope_upper_bound,
        boundary_peak_flag=boundary_peak_flag,
        boundary_gradient_peak_flag=boundary_gradient_peak_flag,
        p_ref_n=p_ref_n if p_ref_n is not None and p_ref_n > 0.0 else None,
        energy_total_over_p_ref_m=energy_total_over_p_ref_m,
        u_total_max_over_p_ref=u_total_max_over_p_ref,
    )
    energy_series = TransitionEnergySeries(
        u_rail_j_per_m=list(u_rail),
        u_foundation_j_per_m=list(u_foundation),
        u_total_j_per_m=u_total,
        du_dx_j_per_m2=du_dx,
        window_energy_j=window_energy,
        window_avg_j_per_m=window_avg,
        window_effective_length_m=window_effective_len,
    )
    return energy_metrics, energy_series


def _integrate_trapezoid(x_values: Sequence[float], values: Sequence[float]) -> float:
    if len(x_values) != len(values) or len(values) < 2:
        return 0.0
    total = 0.0
    for i in range(len(values) - 1):
        dx = x_values[i + 1] - x_values[i]
        total += 0.5 * (values[i] + values[i + 1]) * dx
    return total


def _first_derivative_nonuniform(x_values: Sequence[float], values: Sequence[float]) -> list[float]:
    if len(x_values) != len(values) or len(values) < 2:
        return [0.0 for _ in values]
    derivative = [0.0 for _ in values]
    derivative[0] = (values[1] - values[0]) / (x_values[1] - x_values[0])
    derivative[-1] = (values[-1] - values[-2]) / (x_values[-1] - x_values[-2])
    for i in range(1, len(values) - 1):
        h_s = x_values[i] - x_values[i - 1]
        h_d = x_values[i + 1] - x_values[i]
        c_prev = -h_d / (h_s * (h_s + h_d))
        c_curr = (h_d - h_s) / (h_s * h_d)
        c_next = h_s / (h_d * (h_s + h_d))
        derivative[i] = c_prev * values[i - 1] + c_curr * values[i] + c_next * values[i + 1]
    return derivative


def _smooth_series(values: Sequence[float], points: int) -> list[float]:
    if points <= 1 or len(values) < 3:
        return list(values)
    window = points
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    half = window // 2
    smoothed: list[float] = []
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        segment = values[start:end]
        smoothed.append(sum(segment) / len(segment))
    return smoothed


def _compute_windowed_energy(
    *,
    x_values: Sequence[float],
    values: Sequence[float],
    window_length_target_m: float,
) -> tuple[list[float], list[float], list[float]]:
    if window_length_target_m <= 0:
        raise ValueError("window_length_target_m must be positive")
    x_min = x_values[0]
    x_max = x_values[-1]
    window_energy: list[float] = []
    window_avg: list[float] = []
    window_effective_len: list[float] = []
    half = window_length_target_m / 2.0
    for x_center in x_values:
        a = max(x_min, x_center - half)
        b = min(x_max, x_center + half)
        effective_len = max(0.0, b - a)
        energy = 0.0
        if effective_len > 0.0:
            energy = _integrate_between(x_values, values, a, b)
        window_energy.append(energy)
        window_avg.append(energy / effective_len if effective_len > 0.0 else 0.0)
        window_effective_len.append(effective_len)
    return window_energy, window_avg, window_effective_len


def _integrate_between(
    x_values: Sequence[float],
    values: Sequence[float],
    start_x: float,
    end_x: float,
) -> float:
    if end_x <= start_x:
        return 0.0
    xs = [start_x]
    ys = [_linear_interpolate(x_values, values, start_x)]
    for x, y in zip(x_values, values):
        if start_x < x < end_x:
            xs.append(x)
            ys.append(y)
    xs.append(end_x)
    ys.append(_linear_interpolate(x_values, values, end_x))
    return _integrate_trapezoid(xs, ys)


def _max_value_with_position(
    x_values: Sequence[float],
    values: Sequence[float],
) -> tuple[float, float, int]:
    max_val = values[0] if values else 0.0
    max_x = x_values[0] if x_values else 0.0
    max_i = 0
    for i, value in enumerate(values):
        if value > max_val:
            max_val = value
            max_x = x_values[i]
            max_i = i
    return max_val, max_x, max_i


def _max_abs_value_with_position(
    x_values: Sequence[float],
    values: Sequence[float],
) -> tuple[float, float, int]:
    max_val = abs(values[0]) if values else 0.0
    max_x = x_values[0] if x_values else 0.0
    max_i = 0
    for i, value in enumerate(values):
        abs_val = abs(value)
        if abs_val > max_val:
            max_val = abs_val
            max_x = x_values[i]
            max_i = i
    return max_val, max_x, max_i

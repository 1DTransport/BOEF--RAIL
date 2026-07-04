"""Core engineering calculations for BOEF analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PointLoad:
    """Point load applied to an infinite beam on an elastic foundation."""

    position_m: float
    load_newtons: float


def compute_deflection(load_newtons: float, stiffness_newtons_per_meter: float) -> float:
    """Compute deflection in meters using a simple linear foundation model.

    Args:
        load_newtons: Applied load in newtons (N).
        stiffness_newtons_per_meter: Foundation stiffness in N/m.

    Returns:
        Deflection in meters (m).
    """
    if load_newtons < 0:
        raise ValueError("load_newtons must be non-negative")
    if stiffness_newtons_per_meter <= 0:
        raise ValueError("stiffness_newtons_per_meter must be positive")
    return load_newtons / stiffness_newtons_per_meter


def beam_parameter_beta(
    foundation_modulus_n_per_m2: float, elastic_modulus_pa: float, moment_inertia_m4: float
) -> float:
    """Compute the beam parameter beta for an infinite beam on Winkler foundation.

    Reference: NPTEL "Beam on Elastic Foundation" (beta definition).
    """
    _require_positive(foundation_modulus_n_per_m2, "foundation_modulus_n_per_m2")
    _require_positive(elastic_modulus_pa, "elastic_modulus_pa")
    _require_positive(moment_inertia_m4, "moment_inertia_m4")
    return (foundation_modulus_n_per_m2 / (4.0 * elastic_modulus_pa * moment_inertia_m4)) ** 0.25


def zero_moment_distance(beta: float) -> float:
    """Distance to the first zero bending moment from a point load.

    x1 = pi / (4 * beta).
    """
    _require_positive(beta, "beta")
    return math.pi / (4.0 * beta)


def contraflexure_distance(beta: float) -> float:
    """Distance to rail contraflexure from a point load (3 * x1)."""
    return 3.0 * zero_moment_distance(beta)


def max_moment_single_load(load_newtons: float, beta: float) -> float:
    """Maximum bending moment under a single point load: M0 = P / (4 * beta)."""
    _require_non_negative(load_newtons, "load_newtons")
    _require_positive(beta, "beta")
    return load_newtons / (4.0 * beta)


def max_deflection_single_load(
    load_newtons: float,
    foundation_modulus_n_per_m2: float,
    beta: float,
) -> float:
    """Maximum deflection under a single point load: y0 = P * beta / (2 * k)."""
    _require_non_negative(load_newtons, "load_newtons")
    _require_positive(foundation_modulus_n_per_m2, "foundation_modulus_n_per_m2")
    _require_positive(beta, "beta")
    return load_newtons * beta / (2.0 * foundation_modulus_n_per_m2)


def rail_base_stress(moment_nm: float, section_modulus_m3: float) -> float:
    """Compute rail base stress from bending moment (sigma = M / Z)."""
    _require_positive(section_modulus_m3, "section_modulus_m3")
    return moment_nm / section_modulus_m3


def rail_seat_load_from_deflection(
    *,
    sleeper_spacing_m: float,
    foundation_modulus_n_per_m2: float,
    max_deflection_m: float,
    factor: float = 1.0,
) -> float:
    """Approximate rail seat load from max deflection: Q = S * k * y * F1."""
    _require_positive(sleeper_spacing_m, "sleeper_spacing_m")
    _require_positive(foundation_modulus_n_per_m2, "foundation_modulus_n_per_m2")
    _require_non_negative(max_deflection_m, "max_deflection_m")
    _require_positive(factor, "factor")
    return sleeper_spacing_m * foundation_modulus_n_per_m2 * max_deflection_m * factor


def deflection_at(
    x_m: float,
    loads: Sequence[PointLoad],
    foundation_modulus_n_per_m2: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
) -> float:
    """Compute deflection at position x for multiple point loads via superposition.

    Reference: NPTEL "Beam on Elastic Foundation" (infinite beam point-load solution).
    """
    beta = beam_parameter_beta(
        foundation_modulus_n_per_m2, elastic_modulus_pa, moment_inertia_m4
    )
    return sum(
        _point_load_deflection(
            x_m, load, beta, elastic_modulus_pa, moment_inertia_m4
        )
        for load in _require_loads(loads)
    )


def moment_at(
    x_m: float,
    loads: Sequence[PointLoad],
    foundation_modulus_n_per_m2: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
) -> float:
    """Compute bending moment at position x for multiple point loads.

    Reference: NPTEL "Beam on Elastic Foundation" (moment from deflection).
    """
    beta = beam_parameter_beta(
        foundation_modulus_n_per_m2, elastic_modulus_pa, moment_inertia_m4
    )
    return sum(_point_load_moment(x_m, load, beta) for load in _require_loads(loads))


def shear_at(
    x_m: float,
    loads: Sequence[PointLoad],
    foundation_modulus_n_per_m2: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
) -> float:
    """Compute shear at position x for multiple point loads.

    Reference: NPTEL "Beam on Elastic Foundation" (shear from moment).
    """
    beta = beam_parameter_beta(
        foundation_modulus_n_per_m2, elastic_modulus_pa, moment_inertia_m4
    )
    return sum(_point_load_shear(x_m, load, beta) for load in _require_loads(loads))


def reaction_at(
    x_m: float,
    loads: Sequence[PointLoad],
    foundation_modulus_n_per_m2: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
) -> float:
    """Compute foundation reaction (per unit length) at position x for point loads.

    Reference: NPTEL "Beam on Elastic Foundation" (reaction = k * w).
    """
    beta = beam_parameter_beta(
        foundation_modulus_n_per_m2, elastic_modulus_pa, moment_inertia_m4
    )
    return sum(
        _point_load_reaction(
            x_m, load, foundation_modulus_n_per_m2, beta, elastic_modulus_pa, moment_inertia_m4
        )
        for load in _require_loads(loads)
    )


def sleeper_seat_loads(
    sleeper_positions_m: Sequence[float],
    tributary_length_m: float,
    loads: Sequence[PointLoad],
    foundation_modulus_n_per_m2: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
) -> list[float]:
    """Compute sleeper seat loads by integrating reaction over tributary length.

    Reference: Cai, Raymond & Bathurst (1994), TRR 1470 (rail-seat load estimation).
    """
    _require_positive(tributary_length_m, "tributary_length_m")
    beta = beam_parameter_beta(
        foundation_modulus_n_per_m2, elastic_modulus_pa, moment_inertia_m4
    )
    validated_loads = _require_loads(loads)
    return [
        sum(
            _point_load_reaction_integral(
                sleeper_position_m,
                tributary_length_m,
                load,
                foundation_modulus_n_per_m2,
                beta,
                elastic_modulus_pa,
                moment_inertia_m4,
            )
            for load in validated_loads
        )
        for sleeper_position_m in sleeper_positions_m
    ]


def _point_load_deflection(
    x_m: float,
    load: PointLoad,
    beta: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
) -> float:
    """Closed-form deflection for a point load on an infinite Winkler beam.

    Reference: NPTEL "Beam on Elastic Foundation" (point-load Green's function).
    """
    r = abs(x_m - load.position_m)
    coefficient = load.load_newtons / (8.0 * beta**3 * elastic_modulus_pa * moment_inertia_m4)
    return coefficient * math.exp(-beta * r) * (math.cos(beta * r) + math.sin(beta * r))


def _point_load_moment(x_m: float, load: PointLoad, beta: float) -> float:
    """Closed-form bending moment for a point load on an infinite Winkler beam."""
    r = abs(x_m - load.position_m)
    return (
        load.load_newtons
        / (4.0 * beta)
        * math.exp(-beta * r)
        * (math.cos(beta * r) - math.sin(beta * r))
    )


def _point_load_shear(x_m: float, load: PointLoad, beta: float) -> float:
    """Closed-form shear for a point load on an infinite Winkler beam."""
    r = abs(x_m - load.position_m)
    direction = 1.0 if x_m >= load.position_m else -1.0
    return -direction * (load.load_newtons / 2.0) * math.exp(-beta * r) * math.cos(beta * r)


def _point_load_reaction(
    x_m: float,
    load: PointLoad,
    foundation_modulus_n_per_m2: float,
    beta: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
) -> float:
    return foundation_modulus_n_per_m2 * _point_load_deflection(
        x_m, load, beta, elastic_modulus_pa, moment_inertia_m4
    )


def _point_load_reaction_integral(
    sleeper_position_m: float,
    tributary_length_m: float,
    load: PointLoad,
    foundation_modulus_n_per_m2: float,
    beta: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
) -> float:
    """Integrate foundation reaction over a sleeper tributary length.

    Reference: Cai, Raymond & Bathurst (1994), TRR 1470 (rail-seat load via reaction).
    """
    half_length = tributary_length_m / 2.0
    left = sleeper_position_m - half_length
    right = sleeper_position_m + half_length
    coefficient = (
        foundation_modulus_n_per_m2
        * load.load_newtons
        / (8.0 * beta**3 * elastic_modulus_pa * moment_inertia_m4)
    )

    if right <= load.position_m:
        return coefficient * _integral_exp_cos_sin(
            load.position_m - right, load.position_m - left, beta
        )
    if left >= load.position_m:
        return coefficient * _integral_exp_cos_sin(
            left - load.position_m, right - load.position_m, beta
        )

    left_contribution = coefficient * _integral_exp_cos_sin(
        0.0, load.position_m - left, beta
    )
    right_contribution = coefficient * _integral_exp_cos_sin(
        0.0, right - load.position_m, beta
    )
    return left_contribution + right_contribution


def _integral_exp_cos_sin(start: float, end: float, beta: float) -> float:
    if end < start:
        raise ValueError("integration bounds must be ascending")
    _require_non_negative(start, "start")
    _require_non_negative(end, "end")
    _require_positive(beta, "beta")
    return (-math.exp(-beta * end) * math.cos(beta * end) + math.exp(-beta * start) * math.cos(beta * start)) / beta


def _require_loads(loads: Iterable[PointLoad]) -> list[PointLoad]:
    validated = list(loads)
    if not validated:
        raise ValueError("loads must not be empty")
    for load in validated:
        if not math.isfinite(load.position_m):
            raise ValueError("load position must be finite")
        if not math.isfinite(load.load_newtons):
            raise ValueError("load_newtons must be finite")
        if load.load_newtons < 0:
            raise ValueError("load_newtons must be non-negative")
    return validated


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")

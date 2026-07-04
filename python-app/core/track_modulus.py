"""Track modulus estimation and synthesis utilities (SI units).

References:
- Cai, Raymond & Bathurst (1994), TRR 1470 (deflection-area and single-deflection
  methods; component synthesis discussion).
- NPTEL "Beam on Elastic Foundation" (infinite-beam closed form).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from core.model import PointLoad, beam_parameter_beta, deflection_at


@dataclass(frozen=True)
class TrackModulusEstimate:
    """Result container for track modulus estimation."""

    modulus_n_per_m2: float
    iterations: int | None = None


def estimate_track_modulus_deflection_area(
    *,
    total_load_newtons: float,
    x_m: Sequence[float],
    deflection_m: Sequence[float],
) -> TrackModulusEstimate:
    """Estimate track modulus using deflection-area equilibrium.

    k = P / ∫ w(x) dx, approximated by the trapezoidal rule.

    Reference: Cai, Raymond & Bathurst (1994), TRR 1470.
    """
    _require_positive(total_load_newtons, "total_load_newtons")
    _require_equal_length(x_m, deflection_m)
    area = _trapz(x_m, deflection_m)
    if area <= 0:
        raise ValueError("deflection area must be positive")
    return TrackModulusEstimate(modulus_n_per_m2=total_load_newtons / area)


def estimate_track_modulus_deflection_area_delta(
    *,
    load_high_newtons: float,
    deflection_high_m: Sequence[float],
    load_low_newtons: float,
    deflection_low_m: Sequence[float],
    x_m: Sequence[float],
) -> TrackModulusEstimate:
    """Estimate track modulus using the two-load deflection-area method.

    k = (P1 - P2) / ∫ (w1(x) - w2(x)) dx, which reduces slack.

    Reference: Cai, Raymond & Bathurst (1994), TRR 1470.
    """
    _require_positive(load_high_newtons, "load_high_newtons")
    _require_positive(load_low_newtons, "load_low_newtons")
    if load_high_newtons <= load_low_newtons:
        raise ValueError("load_high_newtons must exceed load_low_newtons")
    _require_equal_length(x_m, deflection_high_m)
    _require_equal_length(x_m, deflection_low_m)
    delta = [high - low for high, low in zip(deflection_high_m, deflection_low_m)]
    area = _trapz(x_m, delta)
    if area <= 0:
        raise ValueError("deflection delta area must be positive")
    return TrackModulusEstimate(
        modulus_n_per_m2=(load_high_newtons - load_low_newtons) / area
    )


def estimate_track_modulus_single_deflection(
    *,
    load_newtons: float,
    deflection_m: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
    bracket_n_per_m2: tuple[float, float] = (1.0e4, 1.0e9),
    tolerance: float = 1.0e-6,
    max_iterations: int = 80,
) -> TrackModulusEstimate:
    """Estimate track modulus from single deflection at the load point.

    Solves for k in the closed-form Winkler solution:
      w(0) = P / (8 * beta^3 * E * I), beta = (k / (4 E I))^(1/4).

    Reference: NPTEL "Beam on Elastic Foundation"; Cai et al. (1994).
    """
    _require_positive(load_newtons, "load_newtons")
    _require_positive(deflection_m, "deflection_m")
    _require_positive(elastic_modulus_pa, "elastic_modulus_pa")
    _require_positive(moment_inertia_m4, "moment_inertia_m4")
    _require_positive(tolerance, "tolerance")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    lower, upper = bracket_n_per_m2
    if lower <= 0 or upper <= 0 or lower >= upper:
        raise ValueError("bracket_n_per_m2 must be positive and increasing")

    def residual(modulus: float) -> float:
        beta = beam_parameter_beta(modulus, elastic_modulus_pa, moment_inertia_m4)
        predicted = load_newtons / (8.0 * (beta**3) * elastic_modulus_pa * moment_inertia_m4)
        return predicted - deflection_m

    low_val = residual(lower)
    high_val = residual(upper)
    if low_val * high_val > 0:
        raise ValueError("bracket does not contain a root")

    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        mid = 0.5 * (lower + upper)
        mid_val = residual(mid)
        if abs(mid_val) <= tolerance:
            return TrackModulusEstimate(modulus_n_per_m2=mid, iterations=iterations)
        if low_val * mid_val <= 0:
            upper = mid
            high_val = mid_val
        else:
            lower = mid
            low_val = mid_val

    return TrackModulusEstimate(modulus_n_per_m2=mid, iterations=iterations)


def synthesize_track_modulus_from_springs(
    *,
    sleeper_spacing_m: float,
    rail_seat_stiffness_n_per_m: float,
    series_stiffnesses_n_per_m: Sequence[float] | None = None,
) -> float:
    """Synthesize equivalent track modulus from rail-seat springs.

    Rail-seat stiffness is combined with optional series layers (e.g., pad, tie,
    ballast) using linear spring-in-series superposition, then converted to
    equivalent continuous modulus using k = k_seat / a.

    Reference: Cai, Raymond & Bathurst (1994), TRR 1470.
    """
    _require_positive(sleeper_spacing_m, "sleeper_spacing_m")
    _require_positive(rail_seat_stiffness_n_per_m, "rail_seat_stiffness_n_per_m")
    stiffnesses = [rail_seat_stiffness_n_per_m]
    if series_stiffnesses_n_per_m is not None:
        for stiffness in series_stiffnesses_n_per_m:
            _require_positive(stiffness, "series_stiffnesses_n_per_m")
            stiffnesses.append(stiffness)

    combined = 1.0 / sum(1.0 / stiffness for stiffness in stiffnesses)
    return combined / sleeper_spacing_m


def track_spring_constant_from_deflection(
    *,
    load_newtons: float,
    deflection_m: float,
) -> float:
    """Compute track spring constant from a static load/deflection: D = P / Y."""
    _require_positive(load_newtons, "load_newtons")
    _require_positive(deflection_m, "deflection_m")
    return load_newtons / deflection_m


def track_modulus_from_spring_constant(
    *,
    spring_constant_n_per_m: float,
    sleeper_spacing_m: float,
) -> float:
    """Compute track modulus from spring constant: k = D / S."""
    _require_positive(spring_constant_n_per_m, "spring_constant_n_per_m")
    _require_positive(sleeper_spacing_m, "sleeper_spacing_m")
    return spring_constant_n_per_m / sleeper_spacing_m


def track_modulus_from_deflection(
    *,
    load_newtons: float,
    deflection_m: float,
    sleeper_spacing_m: float,
) -> float:
    """Compute track modulus from static deflection: k = P / (Y * S)."""
    spring_constant = track_spring_constant_from_deflection(
        load_newtons=load_newtons,
        deflection_m=deflection_m,
    )
    return track_modulus_from_spring_constant(
        spring_constant_n_per_m=spring_constant,
        sleeper_spacing_m=sleeper_spacing_m,
    )


def back_calculate_modulus_single_point(
    *,
    load: PointLoad,
    deflection_m: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
) -> TrackModulusEstimate:
    """Convenience wrapper for single-point track modulus estimation."""
    return estimate_track_modulus_single_deflection(
        load_newtons=load.load_newtons,
        deflection_m=deflection_m,
        elastic_modulus_pa=elastic_modulus_pa,
        moment_inertia_m4=moment_inertia_m4,
    )


def compute_deflection_profile(
    *,
    x_m: Sequence[float],
    loads: Sequence[PointLoad],
    foundation_modulus_n_per_m2: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
) -> list[float]:
    """Compute deflection profile using the closed-form Winkler response.

    This is a helper for validation and estimation workflows.

    Reference: NPTEL "Beam on Elastic Foundation" (infinite beam).
    """
    return [
        deflection_at(
            x,
            loads,
            foundation_modulus_n_per_m2,
            elastic_modulus_pa,
            moment_inertia_m4,
        )
        for x in x_m
    ]


def _trapz(x_m: Sequence[float], values: Sequence[float]) -> float:
    _require_equal_length(x_m, values)
    total = 0.0
    for i in range(len(x_m) - 1):
        if not math.isfinite(x_m[i]) or not math.isfinite(x_m[i + 1]):
            raise ValueError("x_m must contain only finite values")
        if not math.isfinite(values[i]) or not math.isfinite(values[i + 1]):
            raise ValueError("values must contain only finite values")
        dx = x_m[i + 1] - x_m[i]
        if dx <= 0:
            raise ValueError("x_m must be strictly increasing")
        total += 0.5 * dx * (values[i] + values[i + 1])
    return total


def _require_equal_length(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) != len(b):
        raise ValueError("input lengths must match")


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")

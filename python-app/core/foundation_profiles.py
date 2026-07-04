"""Foundation modulus profiles for non-uniform support (SI units).

References:
- Cai, Raymond & Bathurst (1994), TRR 1470 (track modulus definition).
- Lamprea-Pineda et al. (2022) review (transition modeling notes).
"""

from __future__ import annotations

from typing import Sequence


def step_profile(
    *,
    x_m: Sequence[float],
    left_modulus_n_per_m2: float,
    right_modulus_n_per_m2: float,
    step_location_m: float,
) -> list[float]:
    """Build a step-change modulus profile for k(x)."""
    _require_non_negative(left_modulus_n_per_m2, "left_modulus_n_per_m2")
    _require_non_negative(right_modulus_n_per_m2, "right_modulus_n_per_m2")
    return [
        left_modulus_n_per_m2 if x < step_location_m else right_modulus_n_per_m2
        for x in x_m
    ]


def ramp_profile(
    *,
    x_m: Sequence[float],
    start_modulus_n_per_m2: float,
    end_modulus_n_per_m2: float,
    ramp_start_m: float,
    ramp_end_m: float,
) -> list[float]:
    """Build a linear ramp modulus profile for k(x)."""
    _require_non_negative(start_modulus_n_per_m2, "start_modulus_n_per_m2")
    _require_non_negative(end_modulus_n_per_m2, "end_modulus_n_per_m2")
    if ramp_end_m <= ramp_start_m:
        raise ValueError("ramp_end_m must exceed ramp_start_m")

    values = []
    span = ramp_end_m - ramp_start_m
    for x in x_m:
        if x <= ramp_start_m:
            values.append(start_modulus_n_per_m2)
        elif x >= ramp_end_m:
            values.append(end_modulus_n_per_m2)
        else:
            t = (x - ramp_start_m) / span
            values.append(start_modulus_n_per_m2 + t * (end_modulus_n_per_m2 - start_modulus_n_per_m2))
    return values


def _require_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")

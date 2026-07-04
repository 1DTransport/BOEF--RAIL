"""Foundation model utilities for multilayer support (SI units)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DampingModel(str, Enum):
    VISCOUS = "viscous"
    HYSTERETIC = "hysteretic"


@dataclass(frozen=True)
class SeriesFoundationInputs:
    """Series spring inputs per support point (SI units)."""

    railpad_stiffness_n_per_m: float
    railpad_damping_n_s_per_m: float
    trackbed_stiffness_n_per_m: float
    trackbed_damping_n_s_per_m: float


def per_support_to_per_length(value_n_per_m: float, sleeper_spacing_m: float) -> float:
    if sleeper_spacing_m <= 0:
        raise ValueError("sleeper_spacing_m must be positive")
    if value_n_per_m <= 0:
        raise ValueError("support stiffness must be positive")
    return value_n_per_m / sleeper_spacing_m


def equivalent_series_stiffness(k_pad: float, k_bed: float) -> float:
    if k_pad <= 0 or k_bed <= 0:
        raise ValueError("series stiffness values must be positive")
    return (k_pad * k_bed) / (k_pad + k_bed)


def equivalent_series_damping(c_pad: float, c_bed: float) -> float:
    if c_pad <= 0 or c_bed <= 0:
        return 0.0
    return (c_pad * c_bed) / (c_pad + c_bed)


def series_layer_response_per_length(
    *,
    reaction_n_per_m: float,
    trackbed_stiffness_n_per_m2: float,
) -> float:
    """Return sleeper deflection from trackbed stiffness (per length)."""
    if trackbed_stiffness_n_per_m2 <= 0:
        raise ValueError("trackbed_stiffness_n_per_m2 must be positive")
    return reaction_n_per_m / trackbed_stiffness_n_per_m2

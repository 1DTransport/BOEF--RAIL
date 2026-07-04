"""Solvers for special analyses (SI units)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from core.foundation.base import equivalent_series_stiffness
from core.special.config import FloatingSlabConfig
from core.special.results import FloatingSlabResult
from core.special.validation import require_non_negative, require_positive


def _build_frequency_grid(
    *,
    f_min_hz: float,
    f_max_hz: float,
    points: int,
) -> np.ndarray:
    require_non_negative(f_min_hz, "frequency_min_hz")
    require_positive(f_max_hz, "frequency_max_hz")
    if f_max_hz <= f_min_hz:
        raise ValueError("frequency_max_hz must exceed frequency_min_hz")
    if points < 2:
        raise ValueError("frequency_points must be at least 2")
    return np.linspace(f_min_hz, f_max_hz, points)


def solve_floating_slab(config: FloatingSlabConfig) -> FloatingSlabResult:
    """Solve a floating slab isolation model (SDOF)."""
    require_positive(config.slab_mass_kg, "slab_mass_kg")
    require_positive(config.isolator_stiffness_n_per_m, "isolator_stiffness_n_per_m")
    require_non_negative(config.isolator_damping_n_s_per_m, "isolator_damping_n_s_per_m")
    require_non_negative(config.static_load_n, "static_load_n")

    k_eff = config.isolator_stiffness_n_per_m
    if config.railpad_stiffness_n_per_m is not None:
        require_positive(config.railpad_stiffness_n_per_m, "railpad_stiffness_n_per_m")
        k_eff = equivalent_series_stiffness(k_eff, config.railpad_stiffness_n_per_m)

    mass = config.slab_mass_kg
    damping = config.isolator_damping_n_s_per_m

    omega_n = math.sqrt(k_eff / mass)
    natural_frequency_hz = omega_n / (2.0 * math.pi)
    damping_ratio = 0.0
    if k_eff > 0.0 and mass > 0.0:
        damping_ratio = damping / (2.0 * math.sqrt(k_eff * mass))

    static_deflection_m = config.static_load_n / k_eff

    frequency_hz = _build_frequency_grid(
        f_min_hz=config.frequency_min_hz,
        f_max_hz=config.frequency_max_hz,
        points=config.frequency_points,
    )
    r = frequency_hz / max(natural_frequency_hz, 1.0e-12)
    numerator = 1.0 + (2.0 * damping_ratio * r) ** 2
    denominator = (1.0 - r**2) ** 2 + (2.0 * damping_ratio * r) ** 2
    transmissibility = np.sqrt(numerator / np.maximum(denominator, 1.0e-24))
    attenuation_db = 20.0 * np.log10(np.maximum(transmissibility, 1.0e-24))

    return FloatingSlabResult(
        natural_frequency_hz=natural_frequency_hz,
        damping_ratio=damping_ratio,
        static_deflection_m=static_deflection_m,
        frequency_hz=list(frequency_hz),
        transmissibility=list(transmissibility),
        attenuation_db=list(attenuation_db),
    )

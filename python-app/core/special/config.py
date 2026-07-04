"""Special analysis configuration inputs (SI units)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SpecialMode(str, Enum):
    FLOATING_SLAB = "floating_slab"


@dataclass(frozen=True)
class FloatingSlabConfig:
    """Floating slab isolation inputs (SI units)."""

    slab_mass_kg: float
    isolator_stiffness_n_per_m: float
    isolator_damping_n_s_per_m: float
    static_load_n: float
    frequency_min_hz: float = 0.0
    frequency_max_hz: float = 50.0
    frequency_points: int = 200
    railpad_stiffness_n_per_m: float | None = None

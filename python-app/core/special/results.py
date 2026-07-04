"""Special analysis outputs (SI units)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FloatingSlabResult:
    natural_frequency_hz: float
    damping_ratio: float
    static_deflection_m: float
    frequency_hz: list[float]
    transmissibility: list[float]
    attenuation_db: list[float]

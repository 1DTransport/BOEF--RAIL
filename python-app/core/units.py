"""Unit conversion helpers for BOEF."""

from __future__ import annotations


def mm_to_m(length_mm: float) -> float:
    """Convert millimeters to meters."""
    return length_mm / 1000.0


def m_to_mm(length_m: float) -> float:
    """Convert meters to millimeters."""
    return length_m * 1000.0


def kn_to_n(force_kn: float) -> float:
    """Convert kilonewtons to newtons."""
    return force_kn * 1000.0


def n_to_kn(force_n: float) -> float:
    """Convert newtons to kilonewtons."""
    return force_n / 1000.0


def mpa_to_pa(stress_mpa: float) -> float:
    """Convert megapascals to pascals."""
    return stress_mpa * 1_000_000.0


def pa_to_mpa(stress_pa: float) -> float:
    """Convert pascals to megapascals."""
    return stress_pa / 1_000_000.0


def mn_per_m2_to_n_per_m2(modulus_mn_per_m2: float) -> float:
    """Convert support modulus from MN/m² to N/m²."""
    return modulus_mn_per_m2 * 1_000_000.0


def n_per_m2_to_mn_per_m2(modulus_n_per_m2: float) -> float:
    """Convert support modulus from N/m² to MN/m²."""
    return modulus_n_per_m2 / 1_000_000.0


def mm3_to_m3(volume_mm3: float) -> float:
    """Convert cubic millimeters to cubic meters."""
    return volume_mm3 / 1_000_000_000.0


def m3_to_mm3(volume_m3: float) -> float:
    """Convert cubic meters to cubic millimeters."""
    return volume_m3 * 1_000_000_000.0


def mm4_to_m4(inertia_mm4: float) -> float:
    """Convert fourth-power millimeters to fourth-power meters."""
    return inertia_mm4 / 1_000_000_000_000.0


def m4_to_mm4(inertia_m4: float) -> float:
    """Convert fourth-power meters to fourth-power millimeters."""
    return inertia_m4 * 1_000_000_000_000.0


def cm2_to_m2(area_cm2: float) -> float:
    """Convert square centimeters to square meters."""
    return area_cm2 / 10_000.0


def m2_to_cm2(area_m2: float) -> float:
    """Convert square meters to square centimeters."""
    return area_m2 * 10_000.0


def cm3_to_m3(volume_cm3: float) -> float:
    """Convert cubic centimeters to cubic meters."""
    return volume_cm3 / 1_000_000.0


def m3_to_cm3(volume_m3: float) -> float:
    """Convert cubic meters to cubic centimeters."""
    return volume_m3 * 1_000_000.0


def cm4_to_m4(inertia_cm4: float) -> float:
    """Convert fourth-power centimeters to fourth-power meters."""
    return inertia_cm4 / 100_000_000.0


def m4_to_cm4(inertia_m4: float) -> float:
    """Convert fourth-power meters to fourth-power centimeters."""
    return inertia_m4 * 100_000_000.0


def kpa_to_pa(pressure_kpa: float) -> float:
    """Convert kilopascals to pascals."""
    return pressure_kpa * 1_000.0


def pa_to_kpa(pressure_pa: float) -> float:
    """Convert pascals to kilopascals."""
    return pressure_pa / 1_000.0

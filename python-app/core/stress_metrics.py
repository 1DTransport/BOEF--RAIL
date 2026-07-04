"""Stress and pressure post-processing helpers for BOEF plots/exports (SI units)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BearingGeometry:
    """Effective bearing geometry used for ballast/capping pressure calculations."""

    width_m: float
    length_m: float
    area_m2: float
    provenance: str


@dataclass(frozen=True)
class StressMetadata:
    """Metadata attached to computed stress series."""

    ballast_thickness_m: float | None
    stress_model: str
    pressure_sign_convention: str
    bending_sign_convention: str
    bearing_geometry_provenance: str | None
    pressure_available: bool


@dataclass(frozen=True)
class StressResults:
    """Structured stress outputs for plotting/export."""

    x_m: list[float]
    sigma_top_fiber_pa: list[float]
    sigma_bottom_fiber_pa: list[float]
    sleeper_positions_m: list[float]
    q_ballast_signed_pa: list[float] | None
    q_ballast_comp_pa: list[float] | None
    q_capping_signed_pa: list[float] | None
    q_capping_comp_pa: list[float] | None
    metadata: StressMetadata


def get_bearing_geometry(
    *,
    sleeper_width_m: float,
    sleeper_length_m: float,
    bearing_width_override_m: float | None = None,
    bearing_length_override_m: float | None = None,
    override_provenance: str | None = None,
) -> BearingGeometry:
    """Resolve effective bearing geometry from sleeper defaults or explicit overrides."""
    width_m = bearing_width_override_m if bearing_width_override_m is not None else sleeper_width_m
    length_m = bearing_length_override_m if bearing_length_override_m is not None else sleeper_length_m
    _require_positive(width_m, "bearing_width_m")
    _require_positive(length_m, "bearing_length_m")
    area_m2 = width_m * length_m
    _require_positive(area_m2, "bearing_area_m2")
    provenance = (
        override_provenance
        if bearing_width_override_m is not None or bearing_length_override_m is not None
        else "sleeper_geometry"
    )
    if not provenance:
        provenance = "effective_bearing_geometry"
    return BearingGeometry(width_m=width_m, length_m=length_m, area_m2=area_m2, provenance=provenance)


def stress_top_bottom_from_moment(
    *,
    moment_nm: float,
    section_modulus_top_m3: float,
    section_modulus_bottom_m3: float,
) -> tuple[float, float]:
    """Compute signed top/bottom-fibre bending stresses from moment.

    Convention:
    - Positive moment -> positive (compressive) stress at top fibre.
    - Positive moment -> negative (tensile) stress at bottom fibre.
    """
    _require_positive(section_modulus_top_m3, "section_modulus_top_m3")
    _require_positive(section_modulus_bottom_m3, "section_modulus_bottom_m3")
    sigma_top = moment_nm / section_modulus_top_m3
    sigma_bottom = -moment_nm / section_modulus_bottom_m3
    return sigma_top, sigma_bottom


def stress_top_bottom_series_from_moment(
    *,
    moments_nm: Sequence[float],
    section_modulus_top_m3: float,
    section_modulus_bottom_m3: float,
) -> tuple[list[float], list[float]]:
    top: list[float] = []
    bottom: list[float] = []
    for moment in moments_nm:
        sigma_top, sigma_bottom = stress_top_bottom_from_moment(
            moment_nm=moment,
            section_modulus_top_m3=section_modulus_top_m3,
            section_modulus_bottom_m3=section_modulus_bottom_m3,
        )
        top.append(sigma_top)
        bottom.append(sigma_bottom)
    return top, bottom


def ballast_pressure_from_sleeper_load(
    *,
    sleeper_load_n: float,
    bearing_area_m2: float,
) -> float:
    _require_positive(bearing_area_m2, "bearing_area_m2")
    return sleeper_load_n / bearing_area_m2


def capping_pressure_2to1_load_conserving(
    *,
    ballast_pressure_pa: float,
    bearing_width_m: float,
    bearing_length_m: float,
    ballast_thickness_m: float,
) -> float:
    """Compute capping-top pressure using 2:1 spread with explicit load conservation."""
    _require_positive(bearing_width_m, "bearing_width_m")
    _require_positive(bearing_length_m, "bearing_length_m")
    _require_non_negative(ballast_thickness_m, "ballast_thickness_m")
    area_ballast = bearing_width_m * bearing_length_m
    area_capping = (bearing_width_m + 2.0 * ballast_thickness_m) * (bearing_length_m + 2.0 * ballast_thickness_m)
    _require_positive(area_ballast, "area_ballast_m2")
    _require_positive(area_capping, "area_capping_m2")
    conserved_load_n = ballast_pressure_pa * area_ballast
    return conserved_load_n / area_capping


def ensure_positive_compression(pressure_pa: float) -> float:
    """Return compressive-only pressure under convention: positive == compression."""
    return max(0.0, pressure_pa)


def max_abs_envelope(
    *,
    upper_series: Sequence[float],
    lower_series: Sequence[float],
) -> list[float]:
    _require_equal_lengths(upper_series, lower_series, "upper_series", "lower_series")
    return [max(abs(upper), abs(lower)) for upper, lower in zip(upper_series, lower_series)]


def max_compressive_envelope(
    *,
    upper_series: Sequence[float],
    lower_series: Sequence[float],
) -> list[float]:
    _require_equal_lengths(upper_series, lower_series, "upper_series", "lower_series")
    return [ensure_positive_compression(max(upper, lower)) for upper, lower in zip(upper_series, lower_series)]


def build_stress_results_from_single(
    *,
    x_m: Sequence[float],
    moment_nm: Sequence[float],
    sleeper_positions_m: Sequence[float],
    sleeper_loads_n: Sequence[float],
    section_modulus_top_m3: float,
    section_modulus_bottom_m3: float,
    bearing_geometry: BearingGeometry,
    ballast_thickness_m: float,
) -> StressResults:
    _require_equal_lengths(x_m, moment_nm, "x_m", "moment_nm")
    _require_equal_lengths(sleeper_positions_m, sleeper_loads_n, "sleeper_positions_m", "sleeper_loads_n")
    sigma_top, sigma_bottom = stress_top_bottom_series_from_moment(
        moments_nm=moment_nm,
        section_modulus_top_m3=section_modulus_top_m3,
        section_modulus_bottom_m3=section_modulus_bottom_m3,
    )
    ballast_signed = [
        ballast_pressure_from_sleeper_load(
            sleeper_load_n=load,
            bearing_area_m2=bearing_geometry.area_m2,
        )
        for load in sleeper_loads_n
    ]
    ballast_comp = [ensure_positive_compression(value) for value in ballast_signed]
    capping_signed = [
        capping_pressure_2to1_load_conserving(
            ballast_pressure_pa=value,
            bearing_width_m=bearing_geometry.width_m,
            bearing_length_m=bearing_geometry.length_m,
            ballast_thickness_m=ballast_thickness_m,
        )
        for value in ballast_signed
    ]
    capping_comp = [ensure_positive_compression(value) for value in capping_signed]
    metadata = StressMetadata(
        ballast_thickness_m=ballast_thickness_m,
        stress_model="M/Z + 2:1 spread (load-conserving)",
        pressure_sign_convention="positive_compression",
        bending_sign_convention=(
            "positive_moment=>top_fiber_compression_positive; bottom_fiber_tension_negative"
        ),
        bearing_geometry_provenance=bearing_geometry.provenance,
        pressure_available=True,
    )
    return StressResults(
        x_m=list(x_m),
        sigma_top_fiber_pa=sigma_top,
        sigma_bottom_fiber_pa=sigma_bottom,
        sleeper_positions_m=list(sleeper_positions_m),
        q_ballast_signed_pa=ballast_signed,
        q_ballast_comp_pa=ballast_comp,
        q_capping_signed_pa=capping_signed,
        q_capping_comp_pa=capping_comp,
        metadata=metadata,
    )


def build_stress_results_from_envelope(
    *,
    x_m: Sequence[float],
    moment_max_nm: Sequence[float],
    moment_min_nm: Sequence[float],
    sleeper_positions_m: Sequence[float],
    sleeper_loads_max_n: Sequence[float],
    sleeper_loads_min_n: Sequence[float],
    section_modulus_top_m3: float,
    section_modulus_bottom_m3: float,
    bearing_geometry: BearingGeometry,
    ballast_thickness_m: float,
) -> StressResults:
    _require_equal_lengths(x_m, moment_max_nm, "x_m", "moment_max_nm")
    _require_equal_lengths(x_m, moment_min_nm, "x_m", "moment_min_nm")
    _require_equal_lengths(sleeper_positions_m, sleeper_loads_max_n, "sleeper_positions_m", "sleeper_loads_max_n")
    _require_equal_lengths(sleeper_positions_m, sleeper_loads_min_n, "sleeper_positions_m", "sleeper_loads_min_n")

    sigma_top_max, sigma_bottom_max = stress_top_bottom_series_from_moment(
        moments_nm=moment_max_nm,
        section_modulus_top_m3=section_modulus_top_m3,
        section_modulus_bottom_m3=section_modulus_bottom_m3,
    )
    sigma_top_min, sigma_bottom_min = stress_top_bottom_series_from_moment(
        moments_nm=moment_min_nm,
        section_modulus_top_m3=section_modulus_top_m3,
        section_modulus_bottom_m3=section_modulus_bottom_m3,
    )
    sigma_top = max_abs_envelope(upper_series=sigma_top_max, lower_series=sigma_top_min)
    sigma_bottom = max_abs_envelope(upper_series=sigma_bottom_max, lower_series=sigma_bottom_min)

    ballast_signed_max = [
        ballast_pressure_from_sleeper_load(
            sleeper_load_n=load,
            bearing_area_m2=bearing_geometry.area_m2,
        )
        for load in sleeper_loads_max_n
    ]
    ballast_signed_min = [
        ballast_pressure_from_sleeper_load(
            sleeper_load_n=load,
            bearing_area_m2=bearing_geometry.area_m2,
        )
        for load in sleeper_loads_min_n
    ]
    ballast_comp = max_compressive_envelope(
        upper_series=ballast_signed_max,
        lower_series=ballast_signed_min,
    )
    capping_signed = [
        capping_pressure_2to1_load_conserving(
            ballast_pressure_pa=pressure,
            bearing_width_m=bearing_geometry.width_m,
            bearing_length_m=bearing_geometry.length_m,
            ballast_thickness_m=ballast_thickness_m,
        )
        for pressure in ballast_signed_max
    ]
    capping_comp = [
        ensure_positive_compression(
            capping_pressure_2to1_load_conserving(
                ballast_pressure_pa=pressure,
                bearing_width_m=bearing_geometry.width_m,
                bearing_length_m=bearing_geometry.length_m,
                ballast_thickness_m=ballast_thickness_m,
            )
        )
        for pressure in ballast_comp
    ]
    metadata = StressMetadata(
        ballast_thickness_m=ballast_thickness_m,
        stress_model="M/Z + 2:1 spread (load-conserving)",
        pressure_sign_convention="positive_compression",
        bending_sign_convention=(
            "positive_moment=>top_fiber_compression_positive; bottom_fiber_tension_negative"
        ),
        bearing_geometry_provenance=bearing_geometry.provenance,
        pressure_available=True,
    )
    return StressResults(
        x_m=list(x_m),
        sigma_top_fiber_pa=sigma_top,
        sigma_bottom_fiber_pa=sigma_bottom,
        sleeper_positions_m=list(sleeper_positions_m),
        q_ballast_signed_pa=ballast_signed_max,
        q_ballast_comp_pa=ballast_comp,
        q_capping_signed_pa=capping_signed,
        q_capping_comp_pa=capping_comp,
        metadata=metadata,
    )


def build_rail_only_stress_results(
    *,
    x_m: Sequence[float],
    moment_nm: Sequence[float],
    section_modulus_top_m3: float,
    section_modulus_bottom_m3: float,
) -> StressResults:
    _require_equal_lengths(x_m, moment_nm, "x_m", "moment_nm")
    sigma_top, sigma_bottom = stress_top_bottom_series_from_moment(
        moments_nm=moment_nm,
        section_modulus_top_m3=section_modulus_top_m3,
        section_modulus_bottom_m3=section_modulus_bottom_m3,
    )
    metadata = StressMetadata(
        ballast_thickness_m=None,
        stress_model="M/Z (rail bending only)",
        pressure_sign_convention="positive_compression",
        bending_sign_convention=(
            "positive_moment=>top_fiber_compression_positive; bottom_fiber_tension_negative"
        ),
        bearing_geometry_provenance=None,
        pressure_available=False,
    )
    return StressResults(
        x_m=list(x_m),
        sigma_top_fiber_pa=sigma_top,
        sigma_bottom_fiber_pa=sigma_bottom,
        sleeper_positions_m=[],
        q_ballast_signed_pa=None,
        q_ballast_comp_pa=None,
        q_capping_signed_pa=None,
        q_capping_comp_pa=None,
        metadata=metadata,
    )


def build_rail_only_stress_results_from_envelope(
    *,
    x_m: Sequence[float],
    moment_max_nm: Sequence[float],
    moment_min_nm: Sequence[float],
    section_modulus_top_m3: float,
    section_modulus_bottom_m3: float,
) -> StressResults:
    _require_equal_lengths(x_m, moment_max_nm, "x_m", "moment_max_nm")
    _require_equal_lengths(x_m, moment_min_nm, "x_m", "moment_min_nm")
    top_max, bottom_max = stress_top_bottom_series_from_moment(
        moments_nm=moment_max_nm,
        section_modulus_top_m3=section_modulus_top_m3,
        section_modulus_bottom_m3=section_modulus_bottom_m3,
    )
    top_min, bottom_min = stress_top_bottom_series_from_moment(
        moments_nm=moment_min_nm,
        section_modulus_top_m3=section_modulus_top_m3,
        section_modulus_bottom_m3=section_modulus_bottom_m3,
    )
    metadata = StressMetadata(
        ballast_thickness_m=None,
        stress_model="M/Z (rail bending only envelope)",
        pressure_sign_convention="positive_compression",
        bending_sign_convention=(
            "positive_moment=>top_fiber_compression_positive; bottom_fiber_tension_negative"
        ),
        bearing_geometry_provenance=None,
        pressure_available=False,
    )
    return StressResults(
        x_m=list(x_m),
        sigma_top_fiber_pa=max_abs_envelope(upper_series=top_max, lower_series=top_min),
        sigma_bottom_fiber_pa=max_abs_envelope(upper_series=bottom_max, lower_series=bottom_min),
        sleeper_positions_m=[],
        q_ballast_signed_pa=None,
        q_ballast_comp_pa=None,
        q_capping_signed_pa=None,
        q_capping_comp_pa=None,
        metadata=metadata,
    )


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_equal_lengths(
    a: Sequence[float],
    b: Sequence[float],
    a_name: str,
    b_name: str,
) -> None:
    if len(a) != len(b):
        raise ValueError(f"{a_name} length must match {b_name}")

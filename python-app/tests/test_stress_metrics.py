from __future__ import annotations

import pytest

from core.stress_metrics import (
    ballast_pressure_from_sleeper_load,
    build_rail_only_stress_results,
    build_stress_results_from_envelope,
    build_stress_results_from_single,
    capping_pressure_2to1_load_conserving,
    ensure_positive_compression,
    get_bearing_geometry,
    max_abs_envelope,
    max_compressive_envelope,
    stress_top_bottom_from_moment,
)


def test_stress_top_bottom_from_moment_sign_convention() -> None:
    sigma_top, sigma_bottom = stress_top_bottom_from_moment(
        moment_nm=10_000.0,
        section_modulus_top_m3=5.0e-5,
        section_modulus_bottom_m3=5.0e-5,
    )
    assert sigma_top > 0.0
    assert sigma_bottom < 0.0

    sigma_top_neg, sigma_bottom_neg = stress_top_bottom_from_moment(
        moment_nm=-10_000.0,
        section_modulus_top_m3=5.0e-5,
        section_modulus_bottom_m3=5.0e-5,
    )
    assert sigma_top_neg < 0.0
    assert sigma_bottom_neg > 0.0


def test_get_bearing_geometry_validates_area() -> None:
    geometry = get_bearing_geometry(
        sleeper_width_m=0.25,
        sleeper_length_m=2.6,
    )
    assert geometry.area_m2 == pytest.approx(0.65)
    assert geometry.provenance == "sleeper_geometry"

    with pytest.raises(ValueError, match="bearing_width_m"):
        get_bearing_geometry(sleeper_width_m=0.0, sleeper_length_m=2.6)


def test_2to1_spread_load_is_conserved() -> None:
    geometry = get_bearing_geometry(sleeper_width_m=0.25, sleeper_length_m=2.6)
    ballast_pressure = 200_000.0
    capping_pressure = capping_pressure_2to1_load_conserving(
        ballast_pressure_pa=ballast_pressure,
        bearing_width_m=geometry.width_m,
        bearing_length_m=geometry.length_m,
        ballast_thickness_m=0.3,
    )
    load_ballast = ballast_pressure * geometry.area_m2
    capping_area = (geometry.width_m + 2.0 * 0.3) * (geometry.length_m + 2.0 * 0.3)
    load_capping = capping_pressure * capping_area
    assert load_capping == pytest.approx(load_ballast, rel=1e-12)


def test_pressure_compressive_filter_never_negative() -> None:
    assert ensure_positive_compression(-100.0) == 0.0
    assert ensure_positive_compression(0.0) == 0.0
    assert ensure_positive_compression(120.0) == pytest.approx(120.0)


def test_envelope_helpers_max_abs_and_compressive() -> None:
    assert max_abs_envelope(upper_series=[2.0, -3.0], lower_series=[-4.0, 1.0]) == [4.0, 3.0]
    assert max_compressive_envelope(upper_series=[2.0, -3.0], lower_series=[-4.0, 1.0]) == [2.0, 1.0]


def test_build_stress_results_from_single_and_envelope() -> None:
    geometry = get_bearing_geometry(
        sleeper_width_m=0.25,
        sleeper_length_m=2.6,
    )
    single = build_stress_results_from_single(
        x_m=[-1.0, 0.0, 1.0],
        moment_nm=[0.0, 10_000.0, 0.0],
        sleeper_positions_m=[-0.6, 0.0, 0.6],
        sleeper_loads_n=[0.0, 100_000.0, 0.0],
        section_modulus_top_m3=5.0e-5,
        section_modulus_bottom_m3=5.0e-5,
        bearing_geometry=geometry,
        ballast_thickness_m=0.3,
    )
    assert len(single.sigma_top_fiber_pa) == 3
    assert single.q_ballast_comp_pa is not None
    assert single.q_capping_comp_pa is not None
    assert single.q_capping_comp_pa[1] < single.q_ballast_comp_pa[1]
    assert single.metadata.pressure_sign_convention == "positive_compression"

    envelope = build_stress_results_from_envelope(
        x_m=[-1.0, 0.0, 1.0],
        moment_max_nm=[0.0, 8_000.0, 0.0],
        moment_min_nm=[0.0, -10_000.0, 0.0],
        sleeper_positions_m=[-0.6, 0.0, 0.6],
        sleeper_loads_max_n=[0.0, 100_000.0, 0.0],
        sleeper_loads_min_n=[0.0, -30_000.0, 0.0],
        section_modulus_top_m3=5.0e-5,
        section_modulus_bottom_m3=5.0e-5,
        bearing_geometry=geometry,
        ballast_thickness_m=0.3,
    )
    assert envelope.sigma_top_fiber_pa[1] == pytest.approx(200_000_000.0)
    assert envelope.sigma_bottom_fiber_pa[1] == pytest.approx(200_000_000.0)
    assert envelope.q_ballast_comp_pa is not None
    assert envelope.q_ballast_comp_pa[1] > 0.0
    assert envelope.q_capping_comp_pa is not None
    assert envelope.q_capping_comp_pa[1] < envelope.q_ballast_comp_pa[1]


def test_envelope_capping_signed_series_preserves_signed_semantics() -> None:
    geometry = get_bearing_geometry(
        sleeper_width_m=0.25,
        sleeper_length_m=2.6,
    )
    envelope = build_stress_results_from_envelope(
        x_m=[0.0, 1.0],
        moment_max_nm=[0.0, 8_000.0],
        moment_min_nm=[0.0, -10_000.0],
        sleeper_positions_m=[0.0, 1.0],
        sleeper_loads_max_n=[0.0, -10_000.0],
        sleeper_loads_min_n=[0.0, -30_000.0],
        section_modulus_top_m3=5.0e-5,
        section_modulus_bottom_m3=5.0e-5,
        bearing_geometry=geometry,
        ballast_thickness_m=0.3,
    )
    assert envelope.q_capping_signed_pa is not None
    assert envelope.q_capping_comp_pa is not None
    assert envelope.q_capping_signed_pa[1] < 0.0
    assert envelope.q_capping_comp_pa[1] == 0.0


def test_build_rail_only_stress_results_marks_pressure_unavailable() -> None:
    result = build_rail_only_stress_results(
        x_m=[0.0, 1.0],
        moment_nm=[0.0, 10_000.0],
        section_modulus_top_m3=5.0e-5,
        section_modulus_bottom_m3=5.0e-5,
    )
    assert result.metadata.pressure_available is False
    assert result.q_ballast_comp_pa is None


def test_ballast_pressure_from_sleeper_load_validates_area() -> None:
    with pytest.raises(ValueError, match="bearing_area_m2"):
        ballast_pressure_from_sleeper_load(sleeper_load_n=1.0, bearing_area_m2=0.0)

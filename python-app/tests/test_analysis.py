import math

import pytest

from core.analysis import (
    AnalysisInputs,
    DesignInputs,
    ballast_contact_pressure_a3902,
    admissible_bending_stresses_mpa,
    admissible_shear_stresses_mpa,
    build_design_summary,
    dynamic_vertical_wheel_load,
    eisenmann_dynamic_factor,
    build_load_domain,
    compute_track_response,
    eisenmann_dynamic_amplification,
    eisenmann_track_condition_factor,
    eisenmann_velocity_dependent_factor,
    formation_pressure_a3902,
    subgrade_pressure_a3902,
    vqi_for_track_class,
)
from core.model import PointLoad, beam_parameter_beta, rail_base_stress


def test_compute_track_response_builds_pressure_from_loads() -> None:
    inputs = AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=101,
    )

    result = compute_track_response(inputs)

    assert len(result.x_m) == inputs.sample_count
    assert len(result.deflection_m) == inputs.sample_count
    assert len(result.sleeper_positions_m) == len(result.sleeper_loads_n)
    assert len(result.sleeper_loads_n) == len(result.sleeper_pressures_pa)

    area = inputs.sleeper_length_m * inputs.sleeper_width_m
    for load, pressure in zip(result.sleeper_loads_n, result.sleeper_pressures_pa):
        assert math.isclose(pressure, load / area, rel_tol=1e-12)


def test_compute_track_response_preserves_default_domain() -> None:
    inputs = AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=101,
    )

    result = compute_track_response(inputs)

    beta = beam_parameter_beta(
        inputs.foundation_modulus_n_per_m2,
        inputs.elastic_modulus_pa,
        inputs.moment_inertia_m4,
    )
    length = 10.0 / beta
    assert result.x_m[0] == pytest.approx(-length)
    assert result.x_m[-1] == pytest.approx(length)


def test_compute_track_response_superposition_matches_individual_loads() -> None:
    loads = [
        PointLoad(position_m=-1.0, load_newtons=80_000.0),
        PointLoad(position_m=1.5, load_newtons=60_000.0),
    ]
    domain = build_load_domain(
        loads=loads,
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
    )
    inputs = AnalysisInputs(
        loads=loads,
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=121,
        x_domain_m=domain,
    )

    combined = compute_track_response(inputs)

    single_results = [
        compute_track_response(
            AnalysisInputs(
                loads=[load],
                foundation_modulus_n_per_m2=inputs.foundation_modulus_n_per_m2,
                elastic_modulus_pa=inputs.elastic_modulus_pa,
                moment_inertia_m4=inputs.moment_inertia_m4,
                section_modulus_m3=inputs.section_modulus_m3,
                sleeper_spacing_m=inputs.sleeper_spacing_m,
                sleeper_length_m=inputs.sleeper_length_m,
                sleeper_width_m=inputs.sleeper_width_m,
                sample_count=inputs.sample_count,
                x_domain_m=inputs.x_domain_m,
            )
        )
        for load in loads
    ]

    for index in range(0, len(combined.x_m), 20):
        summed_deflection = sum(result.deflection_m[index] for result in single_results)
        summed_moment = sum(result.moment_nm[index] for result in single_results)
        summed_shear = sum(result.shear_n[index] for result in single_results)
        summed_reaction = sum(result.reaction_n_per_m[index] for result in single_results)

        assert math.isclose(combined.deflection_m[index], summed_deflection, rel_tol=1e-12)
        assert math.isclose(combined.moment_nm[index], summed_moment, rel_tol=1e-12)
        assert math.isclose(combined.shear_n[index], summed_shear, rel_tol=1e-12)
        assert math.isclose(combined.reaction_n_per_m[index], summed_reaction, rel_tol=1e-12)


def test_eisenmann_dynamic_amplification_ranges() -> None:
    assert math.isclose(eisenmann_dynamic_amplification(50.0, 2.0, 0.2), 1.0 + 2.0 * 0.2)
    assert math.isclose(
        eisenmann_dynamic_amplification(100.0, 1.0, 0.3),
        1.0 + 1.0 * 0.3 * (1.0 + (100.0 - 60.0) / 140.0),
    )
    assert math.isclose(
        eisenmann_dynamic_amplification(240.0, 1.0, 0.1),
        1.0 + 1.0 * 0.1 * (1.0 + (240.0 - 60.0) / 140.0),
    )


def test_a3902_track_class_to_vqi_and_delta() -> None:
    assert vqi_for_track_class(1) == pytest.approx(45.0)
    assert vqi_for_track_class(5) == pytest.approx(75.0)
    assert eisenmann_track_condition_factor(50.0) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="track_class"):
        vqi_for_track_class(6)


def test_a3902_velocity_factor_piecewise() -> None:
    assert eisenmann_velocity_dependent_factor(60.0) == pytest.approx(1.0)
    assert eisenmann_velocity_dependent_factor(120.0) == pytest.approx(1.0 + (120.0 - 60.0) / 140.0)


def test_a3902_dynamic_load_and_factor() -> None:
    phi = eisenmann_dynamic_factor(0.25, 1.2, 3.0)
    assert phi == pytest.approx(1.9)
    assert dynamic_vertical_wheel_load(100_000.0, phi) == pytest.approx(190_000.0)


def test_a3902_pressure_chain_is_monotonic_with_depth() -> None:
    ballast_pressure_pa, effective_length = ballast_contact_pressure_a3902(
        rail_seat_load_n=120_000.0,
        sleeper_width_m=0.25,
        sleeper_length_m=2.6,
        rail_centres_m=1.5,
        factor_of_safety_f2=1.0,
    )
    pf = formation_pressure_a3902(
        ballast_contact_pressure_pa=ballast_pressure_pa,
        ballast_depth_m=0.30,
        sleeper_width_m=0.25,
        effective_bearing_length_m=effective_length,
    )
    ps = subgrade_pressure_a3902(
        ballast_contact_pressure_pa=ballast_pressure_pa,
        ballast_depth_m=0.30,
        fill_depth_m=0.50,
        sleeper_width_m=0.25,
        effective_bearing_length_m=effective_length,
    )
    assert ballast_pressure_pa > pf > ps > 0.0


def test_a3902_ballast_pressure_requires_rail_centres() -> None:
    with pytest.raises(ValueError, match="rail_centres_m"):
        ballast_contact_pressure_a3902(
            rail_seat_load_n=120_000.0,
            sleeper_width_m=0.25,
            sleeper_length_m=2.6,
            rail_centres_m=None,
            factor_of_safety_f2=1.0,
        )


def test_a3902_static_wheel_load_does_not_include_curve_factor() -> None:
    inputs_curve = AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        design_inputs=DesignInputs(
            speed_kmh=80.0,
            track_factor=0.2,
            probability_factor=3.0,
            wheel_radius_mm=460.0,
            on_curve=True,
            curve_load_factor=1.2,
            factor_of_safety_f1=1.0,
            rail_centres_m=1.5,
        ),
    )
    inputs_straight = AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        design_inputs=DesignInputs(
            speed_kmh=80.0,
            track_factor=0.2,
            probability_factor=3.0,
            wheel_radius_mm=460.0,
            on_curve=False,
            factor_of_safety_f1=1.0,
            rail_centres_m=1.5,
        ),
    )
    beta_curve = beam_parameter_beta(
        inputs_curve.foundation_modulus_n_per_m2,
        inputs_curve.elastic_modulus_pa,
        inputs_curve.moment_inertia_m4,
    )
    beta_straight = beam_parameter_beta(
        inputs_straight.foundation_modulus_n_per_m2,
        inputs_straight.elastic_modulus_pa,
        inputs_straight.moment_inertia_m4,
    )
    summary_curve = build_design_summary(inputs_curve, beta_curve)
    summary_straight = build_design_summary(inputs_straight, beta_straight)
    assert summary_curve is not None and summary_curve.a3902_checks is not None
    assert summary_straight is not None and summary_straight.a3902_checks is not None

    assert summary_curve.effective_load_n == pytest.approx(12_000.0)
    assert summary_curve.a3902_checks.static_vertical_wheel_load_n == pytest.approx(10_000.0)
    assert summary_curve.a3902_checks.rail_seat_load_n == pytest.approx(
        summary_straight.a3902_checks.rail_seat_load_n
    )


def test_subgrade_pressure_a3902_validates_inputs() -> None:
    with pytest.raises(ValueError, match="ballast_depth_m"):
        subgrade_pressure_a3902(
            ballast_contact_pressure_pa=100_000.0,
            ballast_depth_m=0.0,
            fill_depth_m=0.2,
            sleeper_width_m=0.25,
            effective_bearing_length_m=1.2,
        )
    with pytest.raises(ValueError, match="sleeper_width_m"):
        subgrade_pressure_a3902(
            ballast_contact_pressure_pa=100_000.0,
            ballast_depth_m=0.3,
            fill_depth_m=0.2,
            sleeper_width_m=0.0,
            effective_bearing_length_m=1.2,
        )
    with pytest.raises(ValueError, match="effective_bearing_length_m"):
        subgrade_pressure_a3902(
            ballast_contact_pressure_pa=100_000.0,
            ballast_depth_m=0.3,
            fill_depth_m=0.2,
            sleeper_width_m=0.25,
            effective_bearing_length_m=0.0,
        )


def test_a3902_depth_chain_skips_at_zero_ballast_depth() -> None:
    inputs = AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        design_inputs=DesignInputs(
            speed_kmh=80.0,
            track_factor=0.2,
            probability_factor=1.0,
            wheel_radius_mm=460.0,
            ballast_depth_m=0.0,
            rail_centres_m=1.5,
        ),
    )
    beta = beam_parameter_beta(
        inputs.foundation_modulus_n_per_m2,
        inputs.elastic_modulus_pa,
        inputs.moment_inertia_m4,
    )
    summary = build_design_summary(inputs, beta)
    assert summary is not None
    assert summary.a3902_checks is not None
    assert summary.a3902_checks.formation_pressure_pa is None
    assert summary.a3902_checks.subgrade_pressure_pa is None


def test_admissible_design_limits_700_mpa() -> None:
    repeated_bending, incidental_bending = admissible_bending_stresses_mpa(700.0)
    repeated_shear, incidental_shear = admissible_shear_stresses_mpa(700.0)

    assert repeated_bending == pytest.approx(55.0)
    assert incidental_bending == pytest.approx(450.0)
    assert repeated_shear == pytest.approx(200.0)
    assert incidental_shear == pytest.approx(260.0)


def test_admissible_design_limits_900_mpa() -> None:
    repeated_bending, incidental_bending = admissible_bending_stresses_mpa(900.0)
    repeated_shear, incidental_shear = admissible_shear_stresses_mpa(900.0)

    assert repeated_bending == pytest.approx(220.0)
    assert incidental_bending == pytest.approx(580.0)
    assert repeated_shear == pytest.approx(260.0)
    assert incidental_shear == pytest.approx(340.0)


def test_design_summary_applies_curve_factor() -> None:
    inputs = AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        section_modulus_foot_m3=3.0e-5,
        section_modulus_head_m3=3.4e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        design_inputs=DesignInputs(
            speed_kmh=80.0,
            track_factor=0.2,
            probability_factor=1.0,
            wheel_radius_mm=460.0,
            tensile_strength_mpa=700.0,
            on_curve=True,
            rail_centres_m=1.5,
        ),
    )
    beta = beam_parameter_beta(
        inputs.foundation_modulus_n_per_m2,
        inputs.elastic_modulus_pa,
        inputs.moment_inertia_m4,
    )
    summary = build_design_summary(inputs, beta)
    assert summary is not None
    assert summary.effective_load_n == pytest.approx(12_000.0)
    assert summary.shear_pass
    assert summary.a3902_checks is not None
    assert summary.a3902_checks.dynamic_vertical_wheel_load_n > summary.a3902_checks.static_vertical_wheel_load_n


def test_design_summary_uses_combined_moment_envelope() -> None:
    inputs = AnalysisInputs(
        loads=[
            PointLoad(position_m=0.0, load_newtons=10_000.0),
            PointLoad(position_m=0.0, load_newtons=10_000.0),
        ],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        section_modulus_foot_m3=3.0e-5,
        section_modulus_head_m3=3.4e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=121,
        design_inputs=DesignInputs(
            speed_kmh=80.0,
            track_factor=0.2,
            probability_factor=1.0,
            wheel_radius_mm=460.0,
            tensile_strength_mpa=700.0,
            on_curve=False,
            rail_centres_m=1.5,
        ),
    )

    result = compute_track_response(inputs)
    summary = result.summary
    design_summary = summary.design_summary

    assert design_summary is not None
    expected_effective_load = 4.0 * summary.beta_per_m * abs(summary.max_moment.value)
    assert design_summary.effective_load_n == pytest.approx(expected_effective_load)
    assert design_summary.effective_load_n > max(load.load_newtons for load in inputs.loads)

    expected_mean_stress = abs(
        rail_base_stress(summary.max_moment.value, inputs.section_modulus_foot_m3)
    )
    expected_max_stress = expected_mean_stress * design_summary.daf
    assert design_summary.max_bending_stress_pa == pytest.approx(expected_max_stress)


def test_design_summary_uses_track_class_when_provided() -> None:
    inputs = AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        design_inputs=DesignInputs(
            speed_kmh=80.0,
            track_factor=0.2,
            probability_factor=3.0,
            wheel_radius_mm=460.0,
            track_class=4,
            rail_centres_m=1.5,
        ),
    )
    beta = beam_parameter_beta(
        inputs.foundation_modulus_n_per_m2,
        inputs.elastic_modulus_pa,
        inputs.moment_inertia_m4,
    )
    summary = build_design_summary(inputs, beta)
    assert summary is not None
    assert summary.a3902_checks is not None
    assert summary.a3902_checks.vqi == pytest.approx(75.0)
    assert summary.a3902_checks.track_condition_factor_delta == pytest.approx(0.375)


def test_sleeper_positions_ignore_domain_shift() -> None:
    inputs = AnalysisInputs(
        loads=[PointLoad(position_m=1.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=121,
        x_domain_m=(-10.0, 10.0),
    )
    shifted_inputs = AnalysisInputs(
        loads=inputs.loads,
        foundation_modulus_n_per_m2=inputs.foundation_modulus_n_per_m2,
        elastic_modulus_pa=inputs.elastic_modulus_pa,
        moment_inertia_m4=inputs.moment_inertia_m4,
        section_modulus_m3=inputs.section_modulus_m3,
        sleeper_spacing_m=inputs.sleeper_spacing_m,
        sleeper_length_m=inputs.sleeper_length_m,
        sleeper_width_m=inputs.sleeper_width_m,
        sample_count=inputs.sample_count,
        x_domain_m=(-5.0, 15.0),
    )

    base_result = compute_track_response(inputs)
    shifted_result = compute_track_response(shifted_inputs)

    assert base_result.sleeper_positions_m == shifted_result.sleeper_positions_m


def test_compute_track_response_rejects_invalid_samples() -> None:
    inputs = AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=2,
    )

    with pytest.raises(ValueError, match="sample_count must be >= 3"):
        compute_track_response(inputs)


def test_compute_track_response_uses_sleeper_length_for_pressure() -> None:
    inputs = AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=12_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.4,
        sleeper_width_m=0.3,
        sample_count=101,
    )

    result = compute_track_response(inputs)

    bearing_area = inputs.sleeper_length_m * inputs.sleeper_width_m
    for load, pressure in zip(result.sleeper_loads_n, result.sleeper_pressures_pa):
        assert math.isclose(pressure, load / bearing_area, rel_tol=1e-12)


def test_compute_track_response_summary_matches_peaks() -> None:
    inputs = AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=15_000.0)],
        foundation_modulus_n_per_m2=42_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.1e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=121,
    )

    result = compute_track_response(inputs)
    summary = result.summary

    assert summary.beta_per_m > 0
    assert summary.zero_moment_distance_m > 0
    assert summary.contraflexure_distance_m > summary.zero_moment_distance_m
    assert summary.max_deflection.value in result.deflection_m
    assert summary.max_moment.value in result.moment_nm
    assert summary.max_shear.value in result.shear_n
    assert summary.max_reaction.value in result.reaction_n_per_m

import math

from core.transition import (
    TRANSITION_ENERGY_EQUATIONS,
    TRANSITION_ENERGY_METHOD,
    TRANSITION_ENERGY_SCOPE,
    TransitionProfileType,
    build_transition_profile,
    compute_energy_from_envelope,
    compute_energy_from_series,
    compute_metrics_from_envelope,
    compute_metrics_from_series,
)


def test_build_transition_profile_exponential_is_monotone():
    x_values = [0.0, 1.0, 2.0, 3.0]
    profile = build_transition_profile(
        x_values=x_values,
        profile_type=TransitionProfileType.EXPONENTIAL,
        k1_n_per_m2=1.0,
        k2_n_per_m2=5.0,
        transition_length_m=2.0,
        segment_length_m=None,
    )
    assert all(profile[i] <= profile[i + 1] for i in range(len(profile) - 1))
    assert math.isclose(profile[0], 1.0, rel_tol=0.0, abs_tol=1.0e-12)


def test_build_transition_profile_ramp_shows_stiffness_gradient():
    x_values = [-1.0, 0.0, 1.0, 2.0, 3.0]
    profile = build_transition_profile(
        x_values=x_values,
        profile_type=TransitionProfileType.RAMP,
        k1_n_per_m2=40_000_000.0,
        k2_n_per_m2=80_000_000.0,
        transition_length_m=2.0,
        segment_length_m=None,
    )

    assert profile == [
        40_000_000.0,
        40_000_000.0,
        60_000_000.0,
        80_000_000.0,
        80_000_000.0,
    ]
    assert len(set(profile)) > 2


def test_compute_metrics_from_series_linear_deflection():
    x_values = [0.0, 1.0, 2.0, 3.0]
    deflection = [0.0, 1.0, 2.0, 3.0]
    moment = [0.0, 0.0, 0.0, 0.0]
    reaction = [0.0, 0.0, 0.0, 0.0]
    sleeper_positions = [0.0, 1.0, 2.0, 3.0]
    sleeper_loads = [0.0, 0.0, 0.0, 0.0]

    metrics = compute_metrics_from_series(
        x_values=x_values,
        deflection_m=deflection,
        moment_nm=moment,
        reaction_n_per_m=reaction,
        sleeper_positions_m=sleeper_positions,
        sleeper_loads_n=sleeper_loads,
        sleeper_spacing_m=1.0,
        elastic_modulus_pa=1.0,
        moment_inertia_m4=1.0,
    )

    assert math.isclose(metrics.delta_w_s_m, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(metrics.delta_w_1m_m, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(metrics.curvature_max_per_m, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(metrics.moment_max_nm, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(metrics.energy_bending_j, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(metrics.reaction_gradient_max_n_per_m2, 0.0, rel_tol=0.0, abs_tol=1.0e-12)


def test_compute_metrics_from_envelope_symmetric_deflection():
    x_values = [0.0, 1.0, 2.0]
    deflection_max = [0.0, 1.0, 2.0]
    deflection_min = [0.0, -1.0, -2.0]
    moment_max = [0.0, 0.0, 0.0]
    moment_min = [0.0, 0.0, 0.0]
    reaction_max = [0.0, 0.0, 0.0]
    reaction_min = [0.0, 0.0, 0.0]
    sleeper_positions = [0.0, 1.0, 2.0]
    sleeper_loads_max = [0.0, 0.0, 0.0]

    metrics = compute_metrics_from_envelope(
        x_values=x_values,
        deflection_max_m=deflection_max,
        deflection_min_m=deflection_min,
        moment_max_nm=moment_max,
        moment_min_nm=moment_min,
        reaction_max_n_per_m=reaction_max,
        reaction_min_n_per_m=reaction_min,
        sleeper_positions_m=sleeper_positions,
        sleeper_loads_max_n=sleeper_loads_max,
        sleeper_spacing_m=1.0,
        elastic_modulus_pa=1.0,
        moment_inertia_m4=1.0,
    )

    assert math.isclose(metrics.delta_w_s_m, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(metrics.delta_w_1m_m, 1.0, rel_tol=0.0, abs_tol=1.0e-12)


def test_compute_metrics_from_envelope_bending_energy_uses_pointwise_max_absolute_moment():
    metrics = compute_metrics_from_envelope(
        x_values=[0.0, 1.0, 2.0],
        deflection_max_m=[0.0, 0.0, 0.0],
        deflection_min_m=[0.0, 0.0, 0.0],
        moment_max_nm=[10.0, 0.0, 10.0],
        moment_min_nm=[0.0, -10.0, 0.0],
        reaction_max_n_per_m=[0.0, 0.0, 0.0],
        reaction_min_n_per_m=[0.0, 0.0, 0.0],
        sleeper_positions_m=[0.0, 1.0, 2.0],
        sleeper_loads_max_n=[0.0, 0.0, 0.0],
        sleeper_spacing_m=1.0,
        elastic_modulus_pa=1.0,
        moment_inertia_m4=1.0,
    )

    assert math.isclose(metrics.energy_bending_j, 100.0, rel_tol=0.0, abs_tol=1.0e-12)


def test_compute_energy_from_series_zero_response():
    energy_metrics, energy_series = compute_energy_from_series(
        x_values=[0.0, 1.0, 2.0],
        k_profile_n_per_m2=[40_000_000.0, 40_000_000.0, 40_000_000.0],
        deflection_m=[0.0, 0.0, 0.0],
        moment_nm=[0.0, 0.0, 0.0],
        sleeper_spacing_m=0.6,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
    )

    assert math.isclose(energy_metrics.energy_rail_j, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(energy_metrics.energy_foundation_j, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(energy_metrics.energy_total_j, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(energy_metrics.energy_partition_eta, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert all(math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-12) for value in energy_series.u_total_j_per_m)
    assert not energy_metrics.is_envelope_upper_bound
    assert energy_metrics.energy_method == TRANSITION_ENERGY_METHOD
    assert energy_metrics.energy_equations == TRANSITION_ENERGY_EQUATIONS
    assert energy_metrics.energy_scope == TRANSITION_ENERGY_SCOPE


def test_compute_energy_from_series_known_integrals_and_normalization():
    energy_metrics, _ = compute_energy_from_series(
        x_values=[0.0, 1.0, 2.0],
        k_profile_n_per_m2=[2.0, 2.0, 2.0],
        deflection_m=[1.0, 1.0, 1.0],
        moment_nm=[2.0, 2.0, 2.0],
        sleeper_spacing_m=1.0,
        elastic_modulus_pa=1.0,
        moment_inertia_m4=1.0,
        p_ref_n=3.0,
    )

    assert math.isclose(energy_metrics.energy_rail_j, 4.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(energy_metrics.energy_foundation_j, 2.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(energy_metrics.energy_total_j, 6.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(energy_metrics.energy_partition_eta, 1.0 / 3.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(energy_metrics.u_total_max_j_per_m, 3.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(energy_metrics.energy_total_over_p_ref_m or 0.0, 2.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(energy_metrics.u_total_max_over_p_ref or 0.0, 1.0, rel_tol=0.0, abs_tol=1.0e-12)


def test_compute_energy_from_envelope_uses_max_absolute_upper_bound():
    energy_metrics, energy_series = compute_energy_from_envelope(
        x_values=[0.0, 1.0],
        k_profile_n_per_m2=[1.0, 1.0],
        deflection_max_m=[0.1, 0.2],
        deflection_min_m=[-0.3, -0.1],
        moment_max_nm=[1.0, 2.0],
        moment_min_nm=[-4.0, -1.0],
        sleeper_spacing_m=1.0,
        elastic_modulus_pa=2.0,
        moment_inertia_m4=2.0,
    )

    expected_u0 = (4.0**2) / (2.0 * 4.0) + 0.5 * 1.0 * (0.3**2)
    expected_u1 = (2.0**2) / (2.0 * 4.0) + 0.5 * 1.0 * (0.2**2)
    assert math.isclose(energy_series.u_total_j_per_m[0], expected_u0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(energy_series.u_total_j_per_m[1], expected_u1, rel_tol=0.0, abs_tol=1.0e-12)
    assert energy_metrics.is_envelope_upper_bound


def test_compute_energy_from_series_nonuniform_derivative_is_correct_at_interior():
    x_values = [0.0, 0.5, 1.5, 3.0]
    energy_metrics, energy_series = compute_energy_from_series(
        x_values=x_values,
        k_profile_n_per_m2=[2.0, 2.0, 2.0, 2.0],
        deflection_m=[0.0, 0.5, 1.5, 3.0],
        moment_nm=[0.0, 0.0, 0.0, 0.0],
        sleeper_spacing_m=1.0,
        elastic_modulus_pa=1.0,
        moment_inertia_m4=1.0,
    )

    assert math.isclose(energy_series.du_dx_j_per_m2[1], 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(energy_series.du_dx_j_per_m2[2], 3.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(energy_metrics.du_dx_max_j_per_m2, 4.5, rel_tol=0.0, abs_tol=1.0e-12)
    assert energy_metrics.boundary_gradient_peak_flag


def test_compute_energy_from_series_window_clips_to_domain_edges():
    energy_metrics, energy_series = compute_energy_from_series(
        x_values=[0.0, 1.0, 2.0],
        k_profile_n_per_m2=[1.0, 1.0, 1.0],
        deflection_m=[0.0, 0.0, 0.0],
        moment_nm=[2.0, 2.0, 2.0],
        sleeper_spacing_m=2.0,
        elastic_modulus_pa=1.0,
        moment_inertia_m4=1.0,
    )

    assert energy_series.window_effective_length_m == [1.0, 2.0, 1.0]
    assert energy_series.window_energy_j == [2.0, 4.0, 2.0]
    assert energy_series.window_avg_j_per_m == [2.0, 2.0, 2.0]
    assert math.isclose(energy_metrics.window_effective_length_min_m, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(energy_metrics.window_effective_length_max_m, 2.0, rel_tol=0.0, abs_tol=1.0e-12)


def test_compute_energy_from_series_rail_energy_cross_check_with_curvature_form():
    x_values = [0.0, 1.0, 2.0]
    elastic_modulus_pa = 2.0
    moment_inertia_m4 = 3.0
    ei = elastic_modulus_pa * moment_inertia_m4
    curvature = [1.0, 1.0, 1.0]
    moment = [ei * value for value in curvature]
    energy_metrics, _ = compute_energy_from_series(
        x_values=x_values,
        k_profile_n_per_m2=[1.0, 1.0, 1.0],
        deflection_m=[0.0, 0.0, 0.0],
        moment_nm=moment,
        sleeper_spacing_m=1.0,
        elastic_modulus_pa=elastic_modulus_pa,
        moment_inertia_m4=moment_inertia_m4,
    )

    u_from_curvature = [0.5 * ei * (kappa**2) for kappa in curvature]
    expected_rail_energy = 0.5 * (u_from_curvature[0] + u_from_curvature[1]) + 0.5 * (
        u_from_curvature[1] + u_from_curvature[2]
    )
    assert math.isclose(energy_metrics.energy_rail_j, expected_rail_energy, rel_tol=0.0, abs_tol=1.0e-12)

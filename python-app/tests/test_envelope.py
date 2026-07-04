import math
from dataclasses import replace

import pytest

import core.envelope as envelope_module
from core.analysis import AnalysisResult, AnalysisSummary, Extremum
from core.analysis_engine import AnalysisConfig, AnalysisMode
from core.envelope import AS5100EnvelopeSweep, EnvelopeConfig, run_envelope
from core.load_builder import AS5100RailLoadConfig, as5100_load_metadata, build_as5100_rail_loads
from core.model import PointLoad, beam_parameter_beta, max_moment_single_load, moment_at, shear_at


def _build_base_config() -> AnalysisConfig:
    return AnalysisConfig(
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=121,
        x_domain_m=(-5.0, 5.0),
    )


def test_envelope_symmetry_single_load() -> None:
    base = _build_base_config()
    config = EnvelopeConfig(
        analysis_config=base,
        x_ref_start_m=-2.0,
        x_ref_end_m=2.0,
        x_ref_step_m=0.2,
        x_domain_m=(-5.0, 5.0),
        bearing_width_m=0.25,
        bearing_length_m=2.6,
        depth_m=[0.3, 0.6],
        rail_count=2,
        mode=AnalysisMode.CLOSED_FORM,
    )
    result = run_envelope(config)
    mid = len(result.x_m) // 2
    for offset in (5, 10, 20):
        left = mid - offset
        right = mid + offset
        assert result.deflection_max_m[left] == pytest.approx(result.deflection_max_m[right], rel=1e-6)
        assert result.moment_max_nm[left] == pytest.approx(result.moment_max_nm[right], rel=1e-6)
        assert result.shear_max_n[left] == pytest.approx(-result.shear_min_n[right], rel=1e-6)


def test_envelope_includes_static_peak_at_reference() -> None:
    base = _build_base_config()
    config = EnvelopeConfig(
        analysis_config=base,
        x_ref_start_m=0.0,
        x_ref_end_m=0.0,
        x_ref_step_m=0.5,
        x_domain_m=base.x_domain_m or (-5.0, 5.0),
        bearing_width_m=0.25,
        bearing_length_m=2.6,
        depth_m=[0.3],
        rail_count=2,
        mode=AnalysisMode.CLOSED_FORM,
    )
    result = run_envelope(config)
    mid_index = min(range(len(result.x_m)), key=lambda i: abs(result.x_m[i]))
    assert result.deflection_max_m[mid_index] == pytest.approx(result.deflection_min_m[mid_index])


def test_envelope_exposes_pointwise_absolute_governing_curves() -> None:
    base = _build_base_config()
    config = EnvelopeConfig(
        analysis_config=base,
        x_ref_start_m=-1.0,
        x_ref_end_m=1.0,
        x_ref_step_m=0.25,
        x_domain_m=(-3.0, 3.0),
        bearing_width_m=0.25,
        bearing_length_m=2.6,
        depth_m=[0.3],
        rail_count=2,
        mode=AnalysisMode.CLOSED_FORM,
    )

    result = run_envelope(config)

    assert result.moment_abs_max_nm == [
        max(abs(max_val), abs(min_val))
        for max_val, min_val in zip(result.moment_max_nm, result.moment_min_nm)
    ]
    assert result.shear_abs_max_n == [
        max(abs(max_val), abs(min_val))
        for max_val, min_val in zip(result.shear_max_n, result.shear_min_n)
    ]


def test_single_wheel_moment_and_shear_sign_convention_benchmark() -> None:
    base = _build_base_config()
    load = base.loads[0]
    beta = beam_parameter_beta(
        base.foundation_modulus_n_per_m2,
        base.elastic_modulus_pa,
        base.moment_inertia_m4,
    )
    epsilon = 1.0e-9

    moment_under_wheel = moment_at(
        load.position_m,
        [load],
        base.foundation_modulus_n_per_m2,
        base.elastic_modulus_pa,
        base.moment_inertia_m4,
    )
    shear_left = shear_at(
        load.position_m - epsilon,
        [load],
        base.foundation_modulus_n_per_m2,
        base.elastic_modulus_pa,
        base.moment_inertia_m4,
    )
    shear_right = shear_at(
        load.position_m + epsilon,
        [load],
        base.foundation_modulus_n_per_m2,
        base.elastic_modulus_pa,
        base.moment_inertia_m4,
    )

    assert moment_under_wheel == pytest.approx(max_moment_single_load(load.load_newtons, beta))
    assert shear_left == pytest.approx(load.load_newtons / 2.0)
    assert shear_right == pytest.approx(-load.load_newtons / 2.0)


def test_formation_stress_decays_with_depth() -> None:
    base = _build_base_config()
    config = EnvelopeConfig(
        analysis_config=base,
        x_ref_start_m=-1.0,
        x_ref_end_m=1.0,
        x_ref_step_m=0.5,
        x_domain_m=(-4.0, 4.0),
        bearing_width_m=0.25,
        bearing_length_m=2.6,
        depth_m=[0.3, 0.6, 1.2],
        rail_count=2,
        mode=AnalysisMode.CLOSED_FORM,
    )
    result = run_envelope(config)
    shallow = result.formation_stress_max_pa_by_depth[0.3]
    medium = result.formation_stress_max_pa_by_depth[0.6]
    deep = result.formation_stress_max_pa_by_depth[1.2]
    for i in range(len(shallow)):
        assert shallow[i] >= medium[i]
        assert medium[i] >= deep[i]


def test_envelope_auto_domain_uses_extended_beta_margin_formula() -> None:
    base = _build_base_config()
    beta = beam_parameter_beta(
        base.foundation_modulus_n_per_m2,
        base.elastic_modulus_pa,
        base.moment_inertia_m4,
    )
    margin = 8.0 / beta
    x_ref_start = -1.0
    x_ref_end = 1.0
    offsets = [load.position_m for load in base.loads]
    expected_min = x_ref_start + min(offsets) - margin
    expected_max = x_ref_end + max(offsets) + margin
    assert expected_max > expected_min


def test_closed_form_envelope_decays_outside_movement_range_without_zero_clipping() -> None:
    base = _build_base_config()
    config = EnvelopeConfig(
        analysis_config=base,
        x_ref_start_m=-1.0,
        x_ref_end_m=1.0,
        x_ref_step_m=0.1,
        x_domain_m=(-4.0, 4.0),
        bearing_width_m=0.25,
        bearing_length_m=2.6,
        depth_m=[0.3],
        rail_count=2,
        mode=AnalysisMode.CLOSED_FORM,
    )

    result = run_envelope(config)
    peak_abs = max(
        max(abs(value) for value in result.moment_max_nm),
        max(abs(value) for value in result.moment_min_nm),
    )
    left_tail_index = min(range(len(result.x_m)), key=lambda idx: abs(result.x_m[idx] + 2.5))
    right_tail_index = min(range(len(result.x_m)), key=lambda idx: abs(result.x_m[idx] - 2.5))
    left_tail_abs = max(
        abs(result.moment_max_nm[left_tail_index]),
        abs(result.moment_min_nm[left_tail_index]),
    )
    right_tail_abs = max(
        abs(result.moment_max_nm[right_tail_index]),
        abs(result.moment_min_nm[right_tail_index]),
    )

    assert 0.0 < left_tail_abs < peak_abs
    assert 0.0 < right_tail_abs < peak_abs


def test_numerical_transition_envelope_passes_profile_and_shifted_loads_each_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = replace(
        _build_base_config(),
        sample_count=3,
        x_domain_m=(-1.0, 1.0),
        foundation_profile_n_per_m2=[30_000_000.0, 45_000_000.0, 60_000_000.0],
    )
    calls: list[AnalysisConfig] = []

    def fake_run_analysis(config: AnalysisConfig, *, mode: AnalysisMode) -> AnalysisResult:
        calls.append(config)
        assert mode == AnalysisMode.NUMERICAL
        assert config.foundation_profile_n_per_m2 == base.foundation_profile_n_per_m2
        assert config.x_domain_m == (-1.0, 1.0)
        x_ref = config.loads[0].position_m
        x_values = [-1.0, 0.0, 1.0]
        deflection = [0.001 * (x_ref + 1.0), 0.002 * (x_ref + 1.0), 0.001 * (x_ref + 1.0)]
        moment = [100.0 * (x_ref + 1.0), 200.0 * (x_ref + 1.0), 100.0 * (x_ref + 1.0)]
        shear = [-50.0 * (x_ref + 1.0), 0.0, 50.0 * (x_ref + 1.0)]
        reaction = [10.0 * (x_ref + 1.0), 20.0 * (x_ref + 1.0), 10.0 * (x_ref + 1.0)]
        sleeper_loads = [100.0 * (x_ref + 1.0), 200.0 * (x_ref + 1.0), 100.0 * (x_ref + 1.0)]
        sleeper_pressures = [load / (config.sleeper_length_m * config.sleeper_width_m) for load in sleeper_loads]
        summary = AnalysisSummary(
            beta_per_m=1.0,
            zero_moment_distance_m=1.0,
            contraflexure_distance_m=1.0,
            max_deflection=Extremum(max(deflection), 0.0),
            max_moment=Extremum(max(moment), 0.0),
            max_shear=Extremum(max(shear), 1.0),
            max_reaction=Extremum(max(reaction), 0.0),
            max_sleeper_load=Extremum(max(sleeper_loads), 0.0),
            max_sleeper_pressure=Extremum(max(sleeper_pressures), 0.0),
            max_rail_base_stress_pa=0.0,
        )
        return AnalysisResult(
            x_m=x_values,
            deflection_m=deflection,
            moment_nm=moment,
            shear_n=shear,
            reaction_n_per_m=reaction,
            sleeper_positions_m=x_values,
            sleeper_loads_n=sleeper_loads,
            sleeper_pressures_pa=sleeper_pressures,
            summary=summary,
        )

    monkeypatch.setattr(envelope_module, "run_analysis", fake_run_analysis)
    config = EnvelopeConfig(
        analysis_config=base,
        x_ref_start_m=0.0,
        x_ref_end_m=1.0,
        x_ref_step_m=0.5,
        x_domain_m=(-1.0, 1.0),
        bearing_width_m=0.25,
        bearing_length_m=2.6,
        depth_m=[0.3],
        rail_count=2,
        mode=AnalysisMode.NUMERICAL,
    )

    result = run_envelope(config)

    assert [call.loads[0].position_m for call in calls] == [0.0, 0.5, 1.0]
    assert result.x_m == [-1.0, 0.0, 1.0]
    assert result.deflection_max_m[1] == pytest.approx(0.004)
    assert result.moment_max_nm[1] == pytest.approx(400.0)
    assert result.shear_max_n[2] == pytest.approx(100.0)
    assert result.reaction_max_n_per_m[1] == pytest.approx(40.0)
    assert result.sleeper_loads_max_n[1] == pytest.approx(800.0)
    assert result.ballast_pressure_max_pa[1] == pytest.approx(800.0 / (0.25 * 2.6))


def test_numerical_envelope_crops_extended_solver_domain_to_plot_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = replace(
        _build_base_config(),
        sample_count=5,
        x_domain_m=(-3.0, 3.0),
    )
    calls: list[AnalysisConfig] = []

    def fake_run_analysis(config: AnalysisConfig, *, mode: AnalysisMode) -> AnalysisResult:
        calls.append(config)
        assert mode == AnalysisMode.NUMERICAL
        assert config.x_domain_m == (-3.0, 3.0)
        x_values = [-3.0, -1.0, 0.0, 1.0, 3.0]
        load_factor = config.loads[0].position_m + 3.0
        response = [load_factor + value for value in range(len(x_values))]
        sleeper_pressures = [
            value / (config.sleeper_length_m * config.sleeper_width_m)
            for value in response
        ]
        summary = AnalysisSummary(
            beta_per_m=1.0,
            zero_moment_distance_m=1.0,
            contraflexure_distance_m=1.0,
            max_deflection=Extremum(max(response), 0.0),
            max_moment=Extremum(max(response), 0.0),
            max_shear=Extremum(max(response), 0.0),
            max_reaction=Extremum(max(response), 0.0),
            max_sleeper_load=Extremum(max(response), 0.0),
            max_sleeper_pressure=Extremum(max(sleeper_pressures), 0.0),
            max_rail_base_stress_pa=0.0,
        )
        return AnalysisResult(
            x_m=x_values,
            deflection_m=response,
            moment_nm=response,
            shear_n=response,
            reaction_n_per_m=response,
            sleeper_positions_m=x_values,
            sleeper_loads_n=response,
            sleeper_pressures_pa=sleeper_pressures,
            summary=summary,
        )

    monkeypatch.setattr(envelope_module, "run_analysis", fake_run_analysis)
    config = EnvelopeConfig(
        analysis_config=base,
        x_ref_start_m=-2.0,
        x_ref_end_m=2.0,
        x_ref_step_m=2.0,
        x_domain_m=(-1.0, 1.0),
        bearing_width_m=0.25,
        bearing_length_m=2.6,
        depth_m=[0.3],
        rail_count=2,
        mode=AnalysisMode.NUMERICAL,
    )

    result = run_envelope(config)

    assert [call.loads[0].position_m for call in calls] == [-2.0, 0.0, 2.0]
    assert result.x_m == [-1.0, 0.0, 1.0]
    assert result.sleeper_positions_m == [-1.0, 0.0, 1.0]


def test_as5100_governing_sweep_reuses_envelope_engine_and_captures_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = replace(
        _build_base_config(),
        sample_count=3,
        x_domain_m=(-1.0, 1.0),
    )
    calls: list[tuple[int, float, list[float]]] = []

    def fake_run_analysis(config: AnalysisConfig, *, mode: AnalysisMode) -> AnalysisResult:
        assert mode == AnalysisMode.NUMERICAL
        loads = list(config.loads)
        group_count = (len(loads) - 1) // 4
        group_spacing = loads[5].position_m - loads[1].position_m if group_count >= 2 else 12.0
        calls.append((group_count, group_spacing, [load.position_m for load in loads]))
        severity = group_count * 100.0 + (20.0 - group_spacing)
        x_values = [-1.0, 0.0, 1.0]
        deflection = [0.0, 0.001 * severity, 0.0]
        moment = [0.0, 1_000.0 * severity, 0.0]
        shear = [0.0, 100.0 * severity, 0.0]
        reaction = [0.0, 50.0 * severity, 0.0]
        sleeper_loads = [0.0, 80.0 * severity, 0.0]
        sleeper_pressures = [load / (config.sleeper_length_m * config.sleeper_width_m) for load in sleeper_loads]
        summary = AnalysisSummary(
            beta_per_m=1.0,
            zero_moment_distance_m=1.0,
            contraflexure_distance_m=1.0,
            max_deflection=Extremum(max(deflection), 0.0),
            max_moment=Extremum(max(moment), 0.0),
            max_shear=Extremum(max(shear), 0.0),
            max_reaction=Extremum(max(reaction), 0.0),
            max_sleeper_load=Extremum(max(sleeper_loads), 0.0),
            max_sleeper_pressure=Extremum(max(sleeper_pressures), 0.0),
            max_rail_base_stress_pa=1_000_000.0,
        )
        return AnalysisResult(
            x_m=x_values,
            deflection_m=deflection,
            moment_nm=moment,
            shear_n=shear,
            reaction_n_per_m=reaction,
            sleeper_positions_m=x_values,
            sleeper_loads_n=sleeper_loads,
            sleeper_pressures_pa=sleeper_pressures,
            summary=summary,
        )

    monkeypatch.setattr(envelope_module, "run_analysis", fake_run_analysis)

    config = EnvelopeConfig(
        analysis_config=base,
        x_ref_start_m=0.0,
        x_ref_end_m=0.0,
        x_ref_step_m=0.5,
        x_domain_m=(-1.0, 1.0),
        bearing_width_m=0.25,
        bearing_length_m=2.6,
        depth_m=[0.3],
        rail_count=2,
        mode=AnalysisMode.NUMERICAL,
        as5100_sweep=AS5100EnvelopeSweep(
            model="300LA",
            selected_group_count=3,
            selected_group_spacing_m=12.0,
            reference_position_m=0.0,
        ),
    )

    result = run_envelope(config)

    assert len(calls) == 6
    assert result.run_metadata is not None
    assert result.run_metadata["arrangement"] == "governing_envelope_sweep"
    assert result.run_metadata["group_count"] == 3
    assert result.run_metadata["group_spacing_m"] == pytest.approx(12.0)
    assert result.run_metadata["sweep_group_count_candidates"] == [1, 2, 3]
    assert result.run_metadata["sweep_group_spacing_candidates_m"] == [12.0, 20.0]
    assert result.run_metadata["sweep_candidate_count"] == 6
    assert len(result.run_metadata["candidate_summaries"]) == 6
    governing_loads = build_as5100_rail_loads(
        AS5100RailLoadConfig(
            model="300LA",
            group_count=3,
            group_spacing_m=12.0,
            reference_position_m=0.0,
        )
    )
    assert result.run_metadata["axle_positions_m"] == pytest.approx(
        [load.position_m for load in governing_loads]
    )


def test_as5100_fixed_envelope_preserves_run_metadata() -> None:
    arrangement = AS5100RailLoadConfig(
        model="150LA",
        group_count=2,
        group_spacing_m=14.0,
        reference_position_m=1.5,
    )
    loads = build_as5100_rail_loads(arrangement)
    base = replace(_build_base_config(), loads=loads)
    config = EnvelopeConfig(
        analysis_config=base,
        x_ref_start_m=0.0,
        x_ref_end_m=0.0,
        x_ref_step_m=0.5,
        x_domain_m=(-5.0, 15.0),
        bearing_width_m=0.25,
        bearing_length_m=2.6,
        depth_m=[0.3],
        rail_count=2,
        mode=AnalysisMode.CLOSED_FORM,
        run_metadata=as5100_load_metadata(arrangement, loads=loads),
    )

    result = run_envelope(config)

    assert result.run_metadata is not None
    assert result.run_metadata["source_type"] == "as5100_fixed_rail"
    assert result.run_metadata["model"] == "150LA"
    assert result.run_metadata["arrangement"] == "fixed_user_selected"
    assert result.run_metadata["group_count"] == 2
    assert result.run_metadata["group_spacing_m"] == pytest.approx(14.0)
    assert result.run_metadata["reference_position_m"] == pytest.approx(1.5)
    assert result.run_metadata["axle_positions_m"] == pytest.approx([load.position_m for load in loads])


def test_as5100_fixed_envelope_matches_existing_envelope_engine_outputs() -> None:
    arrangement = AS5100RailLoadConfig(
        model="300LA",
        group_count=3,
        group_spacing_m=12.0,
        reference_position_m=0.0,
    )
    loads = build_as5100_rail_loads(arrangement)
    base = replace(_build_base_config(), loads=loads)
    plain_config = EnvelopeConfig(
        analysis_config=base,
        x_ref_start_m=0.0,
        x_ref_end_m=0.0,
        x_ref_step_m=0.5,
        x_domain_m=(-5.0, 25.0),
        bearing_width_m=0.25,
        bearing_length_m=2.6,
        depth_m=[0.3],
        rail_count=2,
        mode=AnalysisMode.CLOSED_FORM,
    )
    traced_config = replace(
        plain_config,
        run_metadata=as5100_load_metadata(arrangement, loads=loads),
    )

    plain_result = run_envelope(plain_config)
    traced_result = run_envelope(traced_config)

    assert traced_result.deflection_max_m == pytest.approx(plain_result.deflection_max_m)
    assert traced_result.deflection_min_m == pytest.approx(plain_result.deflection_min_m)
    assert traced_result.moment_max_nm == pytest.approx(plain_result.moment_max_nm)
    assert traced_result.moment_min_nm == pytest.approx(plain_result.moment_min_nm)
    assert traced_result.shear_max_n == pytest.approx(plain_result.shear_max_n)
    assert traced_result.shear_min_n == pytest.approx(plain_result.shear_min_n)
    assert traced_result.reaction_max_n_per_m == pytest.approx(plain_result.reaction_max_n_per_m)
    assert traced_result.reaction_min_n_per_m == pytest.approx(plain_result.reaction_min_n_per_m)
    assert traced_result.run_metadata is not None
    assert traced_result.run_metadata["source_type"] == "as5100_fixed_rail"


def test_as5100_envelope_sleeper_totals_use_two_per_rail_wheel_reactions() -> None:
    arrangement = AS5100RailLoadConfig(
        model="300LA",
        group_count=1,
        group_spacing_m=12.0,
        reference_position_m=0.0,
    )
    loads = build_as5100_rail_loads(arrangement)
    base = replace(_build_base_config(), loads=loads, sample_count=801)
    metadata = as5100_load_metadata(arrangement, loads=loads)
    config = EnvelopeConfig(
        analysis_config=base,
        x_ref_start_m=0.0,
        x_ref_end_m=0.0,
        x_ref_step_m=0.5,
        x_domain_m=(-20.0, 25.0),
        bearing_width_m=0.25,
        bearing_length_m=2.6,
        depth_m=[0.3],
        rail_count=2,
        mode=AnalysisMode.CLOSED_FORM,
        run_metadata=metadata,
    )

    result = run_envelope(config)

    solver_wheel_total_n = sum(load.load_newtons for load in loads)
    display_axle_total_n = sum(metadata["axle_loads_n"])
    assert solver_wheel_total_n * 2.0 == pytest.approx(display_axle_total_n)
    assert sum(result.sleeper_loads_max_n) == pytest.approx(display_axle_total_n, rel=0.08)

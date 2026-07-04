from dataclasses import replace

import pytest

from core.analysis import DesignInputs
from core.analysis_engine import AnalysisConfig, AnalysisMode
from core.load_builder import AS5100RailLoadConfig, build_as5100_rail_loads
from core.model import PointLoad
from core.sensitivity import (
    DesignCriteria,
    SensitivityMetrics,
    SensitivityScenario,
    SensitivityScenarioResult,
    SensitivityRunMode,
    SensitivityTransitionContext,
    SensitivityVariable,
    evaluate_design_criteria,
    _scenario_name,
    _score_result,
    _score_results,
    build_scenarios,
    rescore_sensitivity_result,
    run_sensitivity,
)
from core.transition import TransitionProfileType, TransitionRunMode


def _base_config() -> AnalysisConfig:
    return AnalysisConfig(
        loads=[PointLoad(position_m=0.0, load_newtons=100_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=4.1e-4,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.5,
        sleeper_width_m=0.25,
        railpad_stiffness_n_per_m=70_000_000.0,
        pad_stiffness_n_per_m=70_000_000.0,
        design_inputs=DesignInputs(
            speed_kmh=80.0,
            track_factor=1.0,
            probability_factor=1.0,
            wheel_radius_mm=430.0,
            ballast_depth_m=0.3,
            rail_centres_m=1.5,
        ),
    )


def test_baseline_scenario_matches_selected_config_values() -> None:
    config = _base_config()
    scenario = build_scenarios(
        base_config=config,
        variables=[SensitivityVariable.SUPPORT_STIFFNESS],
    )[0]
    assert scenario.name == "Baseline"
    assert scenario.analysis_config.foundation_modulus_n_per_m2 == pytest.approx(40_000_000.0)
    assert scenario.analysis_config.sleeper_spacing_m == pytest.approx(0.6)
    assert scenario.analysis_config.loads[0].load_newtons == pytest.approx(100_000.0)


def test_one_variable_at_a_time_sweep_creates_expected_scenario_count() -> None:
    scenarios = build_scenarios(
        base_config=_base_config(),
        variables=[
            SensitivityVariable.SUPPORT_STIFFNESS,
            SensitivityVariable.SLEEPER_SPACING,
            SensitivityVariable.WHEEL_LOAD,
        ],
    )
    assert len(scenarios) == 1 + 3 * 4


def test_scenario_names_report_percent_change() -> None:
    assert _scenario_name("support_stiffness", 1.2) == "support stiffness +20%"
    assert _scenario_name("sleeper_spacing", 0.8) == "sleeper spacing -20%"


def test_sweep_changes_only_selected_solver_input() -> None:
    base = _base_config()
    scenario = build_scenarios(
        base_config=base,
        variables=[SensitivityVariable.SLEEPER_SPACING],
    )[1]

    assert scenario.analysis_config.sleeper_spacing_m == pytest.approx(0.48)
    assert scenario.analysis_config.foundation_modulus_n_per_m2 == pytest.approx(
        base.foundation_modulus_n_per_m2
    )
    assert scenario.analysis_config.loads[0].load_newtons == pytest.approx(
        base.loads[0].load_newtons
    )
    assert scenario.analysis_config.design_inputs.speed_kmh == pytest.approx(
        base.design_inputs.speed_kmh
    )


def test_wheel_load_sensitivity_scales_every_as5100_axle() -> None:
    as5100_loads = build_as5100_rail_loads(
        AS5100RailLoadConfig(
            model="300LA",
            group_count=2,
            group_spacing_m=12.0,
            reference_position_m=0.0,
        )
    )
    base = _base_config().__class__(
        **{**_base_config().__dict__, "loads": as5100_loads}
    )

    scenario = build_scenarios(
        base_config=base,
        variables=[SensitivityVariable.WHEEL_LOAD],
    )[1]

    assert len(scenario.analysis_config.loads) == len(as5100_loads)
    assert [load.position_m for load in scenario.analysis_config.loads] == pytest.approx(
        [load.position_m for load in as5100_loads]
    )
    assert [load.load_newtons for load in scenario.analysis_config.loads] == pytest.approx(
        [0.8 * load.load_newtons for load in as5100_loads]
    )


def test_as5100_position_sensitivity_shifts_every_axle_without_scaling_loads() -> None:
    as5100_loads = build_as5100_rail_loads(
        AS5100RailLoadConfig(
            model="300LA",
            group_count=2,
            group_spacing_m=12.0,
            reference_position_m=0.0,
        )
    )
    base = replace(_base_config(), loads=as5100_loads)

    scenarios = build_scenarios(
        base_config=base,
        variables=[SensitivityVariable.AS5100_POSITION],
    )
    shifted = next(scenario for scenario in scenarios if scenario.parameter_value == 0.5)

    assert len(scenarios) == 5
    assert shifted.name == "AS5100 train shifted +0.5 m"
    assert shifted.changed_parameter == "as5100_position"
    assert shifted.factor == pytest.approx(1.0)
    assert [load.position_m for load in shifted.analysis_config.loads] == pytest.approx(
        [load.position_m + 0.5 for load in as5100_loads]
    )
    assert [load.load_newtons for load in shifted.analysis_config.loads] == pytest.approx(
        [load.load_newtons for load in as5100_loads]
    )


def test_invalid_sweep_values_are_rejected_before_solver_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_run_analysis(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("solver should not run")

    monkeypatch.setattr("core.sensitivity.run_analysis", fake_run_analysis)
    bad_config = _base_config()
    bad_config = bad_config.__class__(
        **{**bad_config.__dict__, "foundation_modulus_n_per_m2": -1.0}
    )
    result = run_sensitivity(base_config=bad_config, variables=[])
    assert calls == 0
    assert result.baseline.warning


def test_missing_metrics_do_not_crash_scoring() -> None:
    score = _score_result(
        SensitivityMetrics(max_deflection_m=1.0),
        SensitivityMetrics(max_moment_nm=10.0),
    )
    assert score is None


def test_weight_redistribution_uses_available_metrics_only() -> None:
    score = _score_result(
        SensitivityMetrics(max_deflection_m=2.0),
        SensitivityMetrics(max_deflection_m=1.0),
    )
    assert score == pytest.approx(0.5)


def test_static_metrics_include_ballast_formation_and_subgrade_pressures() -> None:
    result = run_sensitivity(
        base_config=_base_config(),
        variables=[SensitivityVariable.SUPPORT_STIFFNESS],
    )

    metrics = result.baseline.metrics
    assert metrics.ballast_pressure_pa is not None
    assert metrics.formation_pressure_pa is not None
    assert metrics.subgrade_pressure_pa is not None
    assert metrics.deep_subgrade_pressure_pa is not None
    assert metrics.ballast_pressure_pa > 0.0
    assert metrics.formation_pressure_pa > 0.0
    assert metrics.formation_pressure_pa >= metrics.subgrade_pressure_pa > 0.0
    assert metrics.subgrade_pressure_pa >= metrics.deep_subgrade_pressure_pa > 0.0


def test_pressure_metrics_participate_in_score() -> None:
    score = _score_result(
        SensitivityMetrics(ballast_pressure_pa=100.0, formation_pressure_pa=50.0),
        SensitivityMetrics(ballast_pressure_pa=80.0, formation_pressure_pa=25.0),
    )

    assert score == pytest.approx(0.65)


def test_design_criteria_utilization_statuses() -> None:
    criteria = DesignCriteria(max_deflection_m=0.01, rail_stress_pa=100.0)

    passing = evaluate_design_criteria(
        SensitivityMetrics(max_deflection_m=0.004, rail_stress_pa=50.0),
        criteria,
    )
    warning = evaluate_design_criteria(
        SensitivityMetrics(max_deflection_m=0.009, rail_stress_pa=50.0),
        criteria,
    )
    failing = evaluate_design_criteria(
        SensitivityMetrics(max_deflection_m=0.004, rail_stress_pa=125.0),
        criteria,
    )

    assert passing.status == "pass"
    assert warning.status == "warning"
    assert failing.status == "fail"
    assert failing.governing_criterion == "rail stress"


def test_rescore_sensitivity_result_updates_criteria_without_solver_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = run_sensitivity(
        base_config=_base_config(),
        variables=[SensitivityVariable.SUPPORT_STIFFNESS],
        criteria=DesignCriteria(max_deflection_m=0.01),
    )

    def fake_run_analysis(*_args, **_kwargs):
        raise AssertionError("solver should not rerun when only criteria change")

    monkeypatch.setattr("core.sensitivity.run_analysis", fake_run_analysis)
    rescored = rescore_sensitivity_result(
        result,
        criteria=DesignCriteria(max_deflection_m=1.0e-9),
    )

    assert rescored.criteria == DesignCriteria(max_deflection_m=1.0e-9)
    assert rescored.baseline.metrics == result.baseline.metrics
    assert rescored.baseline.decision.status == "fail"
    assert rescored.scenarios[0].decision.status == "fail"


def test_ranking_prefers_passing_scenario_over_lower_failed_engineering_score() -> None:
    baseline = SensitivityScenarioResult(
        scenario_name="Baseline",
        changed_parameter="baseline",
        parameter_value=None,
        factor=1.0,
        metrics=SensitivityMetrics(max_deflection_m=1.0, rail_stress_pa=10.0),
        score=None,
        percent_improvement=None,
        warning="",
    )
    passing = SensitivityScenarioResult(
        scenario_name="Passing",
        changed_parameter="support_stiffness",
        parameter_value=1.0,
        factor=1.1,
        metrics=SensitivityMetrics(max_deflection_m=0.8, rail_stress_pa=0.8),
        score=None,
        percent_improvement=None,
        warning="",
    )
    failed_but_low_score = SensitivityScenarioResult(
        scenario_name="Failed",
        changed_parameter="support_stiffness",
        parameter_value=1.0,
        factor=1.2,
        metrics=SensitivityMetrics(max_deflection_m=0.6, rail_stress_pa=1.1),
        score=None,
        percent_improvement=None,
        warning="",
    )

    scored = _score_results(
        baseline,
        [baseline, failed_but_low_score, passing],
        criteria=DesignCriteria(max_deflection_m=1.0, rail_stress_pa=1.0),
    )
    rank_by_name = {item.scenario_name: item.rank for item in scored}

    assert rank_by_name["Passing"] == 1
    assert next(item for item in scored if item.scenario_name == "Passing").decision.status == "pass"
    assert next(item for item in scored if item.scenario_name == "Failed").decision.status == "fail"


def test_additional_database_scenarios_are_run_and_ranked() -> None:
    base = _base_config()
    support_option = SensitivityScenario(
        name="support profile Ballast 80 MN/m²",
        changed_parameter="support_profile",
        parameter_value=80_000_000.0,
        factor=2.0,
        analysis_config=base.__class__(
            **{**base.__dict__, "foundation_modulus_n_per_m2": 80_000_000.0}
        ),
        apply_payload={"support_profile_id": 2},
    )

    result = run_sensitivity(
        base_config=base,
        variables=[],
        criteria=DesignCriteria(max_deflection_m=0.01),
        additional_scenarios=[support_option],
    )

    added = next(item for item in result.scenarios if item.changed_parameter == "support_profile")
    assert added.apply_payload == {"support_profile_id": 2}
    assert added.rank is not None
    assert added.decision.status == "pass"


def test_warning_scenarios_receive_penalty() -> None:
    baseline = SensitivityScenarioResult(
        scenario_name="Baseline",
        changed_parameter="baseline",
        parameter_value=None,
        factor=1.0,
        metrics=SensitivityMetrics(max_deflection_m=1.0),
        score=None,
        percent_improvement=None,
        warning="",
    )
    warning = SensitivityScenarioResult(
        scenario_name="Warning",
        changed_parameter="support_stiffness",
        parameter_value=1.0,
        factor=1.2,
        metrics=SensitivityMetrics(max_deflection_m=0.8),
        score=None,
        percent_improvement=None,
        warning="check warning",
    )
    scored = _score_results(baseline, [baseline, warning])
    assert scored[1].score == pytest.approx(0.9)


def test_ranking_selects_lowest_valid_score() -> None:
    baseline = SensitivityScenarioResult(
        scenario_name="Baseline",
        changed_parameter="baseline",
        parameter_value=None,
        factor=1.0,
        metrics=SensitivityMetrics(max_deflection_m=1.0),
        score=None,
        percent_improvement=None,
        warning="",
    )
    better = SensitivityScenarioResult(
        scenario_name="Better",
        changed_parameter="support_stiffness",
        parameter_value=1.2,
        factor=1.2,
        metrics=SensitivityMetrics(max_deflection_m=0.8),
        score=None,
        percent_improvement=None,
        warning="",
    )
    worse = SensitivityScenarioResult(
        scenario_name="Worse",
        changed_parameter="support_stiffness",
        parameter_value=0.8,
        factor=0.8,
        metrics=SensitivityMetrics(max_deflection_m=1.2),
        score=None,
        percent_improvement=None,
        warning="",
    )
    scored = _score_results(baseline, [baseline, worse, better])
    rank_by_name = {item.scenario_name: item.rank for item in scored}
    assert rank_by_name["Better"] == 1
    assert rank_by_name["Baseline"] == 2
    assert rank_by_name["Worse"] == 3


def test_transition_length_sensitivity_populates_transition_metric() -> None:
    config = _base_config()
    config = config.__class__(**{**config.__dict__, "x_domain_m": (-2.0, 4.0)})
    context = SensitivityTransitionContext(
        run_mode=TransitionRunMode.SINGLE,
        profile_type=TransitionProfileType.RAMP,
        template_name="Test",
        preset_name="Test",
        k1_n_per_m2=40_000_000.0,
        k2_n_per_m2=80_000_000.0,
        transition_length_m=3.0,
        segment_length_m=None,
        domain_m=(-2.0, 4.0),
        analysis_config=config,
        analysis_mode=AnalysisMode.NUMERICAL,
    )

    result = run_sensitivity(
        base_config=config,
        variables=[SensitivityVariable.TRANSITION_LENGTH],
        mode=SensitivityRunMode.TRANSITION,
        transition_context=context,
    )

    assert len(result.scenarios) == 5
    assert result.baseline.warning == ""
    assert result.baseline.metrics.transition_metric_m is not None


def test_transition_sweep_preserves_context_units_and_mode() -> None:
    config = _base_config()
    config = config.__class__(**{**config.__dict__, "x_domain_m": (-2.0, 4.0)})
    context = SensitivityTransitionContext(
        run_mode=TransitionRunMode.SINGLE,
        profile_type=TransitionProfileType.RAMP,
        template_name="Test",
        preset_name="Test",
        k1_n_per_m2=40_000_000.0,
        k2_n_per_m2=80_000_000.0,
        transition_length_m=3.0,
        segment_length_m=None,
        domain_m=(-2.0, 4.0),
        analysis_config=config,
        analysis_mode=AnalysisMode.NUMERICAL,
    )

    scenarios = build_scenarios(
        base_config=config,
        variables=[SensitivityVariable.TRANSITION_LENGTH],
        mode=SensitivityRunMode.TRANSITION,
        transition_context=context,
    )
    changed = scenarios[1]

    assert changed.transition_context.k1_n_per_m2 == pytest.approx(40_000_000.0)
    assert changed.transition_context.k2_n_per_m2 == pytest.approx(80_000_000.0)
    assert changed.transition_context.transition_length_m == pytest.approx(2.4)
    assert changed.transition_context.domain_m == pytest.approx((-2.0, 4.0))
    assert changed.transition_context.analysis_mode is AnalysisMode.NUMERICAL
    assert changed.analysis_config.loads[0].load_newtons == pytest.approx(100_000.0)

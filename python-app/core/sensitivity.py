"""Sensitivity and design-screening helpers for BOEF analyses.

All calculation inputs and outputs use SI units. This module deliberately
reuses the existing static and transition analysis paths; it only owns sweep
generation, lightweight metric extraction, scoring, and recommendation text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import csv
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

from core.analysis import AnalysisResult, DesignInputs, subgrade_pressure_a3902
from core.analysis_engine import AnalysisConfig, AnalysisMode, run_analysis
from core.model import PointLoad
from core.transition import (
    TransitionProfileType,
    TransitionRunMode,
    TransitionRunResult,
    build_series_from_single,
    build_transition_profile,
    compute_energy_from_series,
    compute_metrics_from_series,
)


SWEEP_FACTORS: tuple[float, ...] = (0.80, 0.90, 1.0, 1.10, 1.20)
AS5100_POSITION_OFFSETS_M: tuple[float, ...] = (-1.0, -0.5, 0.5, 1.0)
WARNING_SCORE_PENALTY = 0.10
DEFAULT_WEIGHTS = {
    "deflection": 0.30,
    "moment_stress": 0.25,
    "sleeper_load": 0.15,
    "ballast_pressure": 0.10,
    "formation_pressure": 0.10,
    "subgrade_pressure": 0.08,
    "deep_subgrade_pressure": 0.02,
    "transition_metric": 0.15,
}
DECISION_STATUS_ORDER = {"pass": 0, "warning": 1, "fail": 2, "unrated": 3}
CONSTRUCTABILITY_DIFFICULTY = {
    "baseline": 0.0,
    "support_stiffness": 0.45,
    "support_profile": 0.45,
    "ballast_depth": 0.50,
    "sleeper_spacing": 0.65,
    "pad_stiffness": 0.65,
    "pad": 0.65,
    "wheel_load": 0.85,
    "as5100_position": 0.25,
    "speed": 0.85,
    "rail": 0.90,
    "sleeper": 0.90,
    "transition_length": 0.50,
}


class SensitivityRunMode(str, Enum):
    STATIC = "static"
    TRANSITION = "transition"


class SensitivityVariable(str, Enum):
    SUPPORT_STIFFNESS = "support_stiffness"
    SLEEPER_SPACING = "sleeper_spacing"
    PAD_STIFFNESS = "pad_stiffness"
    WHEEL_LOAD = "wheel_load"
    AS5100_POSITION = "as5100_position"
    SPEED = "speed"
    BALLAST_DEPTH = "ballast_depth"
    TRANSITION_LENGTH = "transition_length"


@dataclass(frozen=True)
class SensitivityTransitionContext:
    run_mode: TransitionRunMode
    profile_type: TransitionProfileType
    template_name: str
    preset_name: str
    k1_n_per_m2: float
    k2_n_per_m2: float | None
    transition_length_m: float | None
    segment_length_m: float | None
    domain_m: tuple[float, float]
    analysis_config: AnalysisConfig
    analysis_mode: AnalysisMode
    k_profile_n_per_m2: list[float] | None = None


@dataclass(frozen=True)
class SensitivityMetrics:
    max_deflection_m: float | None = None
    max_moment_nm: float | None = None
    max_sleeper_load_n: float | None = None
    rail_stress_pa: float | None = None
    ballast_pressure_pa: float | None = None
    formation_pressure_pa: float | None = None
    subgrade_pressure_pa: float | None = None
    deep_subgrade_pressure_pa: float | None = None
    transition_metric_m: float | None = None


@dataclass(frozen=True)
class DesignCriteria:
    max_deflection_m: float | None = None
    rail_stress_pa: float | None = None
    max_sleeper_load_n: float | None = None
    ballast_pressure_pa: float | None = None
    formation_pressure_pa: float | None = None
    subgrade_pressure_pa: float | None = None
    deep_subgrade_pressure_pa: float | None = None
    transition_metric_m: float | None = None


@dataclass(frozen=True)
class DesignDecision:
    status: str = "unrated"
    governing_criterion: str = ""
    max_utilization: float | None = None
    utilizations: dict[str, float] | None = None


@dataclass(frozen=True)
class SensitivityScenario:
    name: str
    changed_parameter: str
    parameter_value: float | None
    factor: float
    analysis_config: AnalysisConfig
    transition_context: SensitivityTransitionContext | None = None
    apply_payload: dict[str, object] | None = None
    constructability_class: str = "screening"


@dataclass(frozen=True)
class SensitivityScenarioResult:
    scenario_name: str
    changed_parameter: str
    parameter_value: float | None
    factor: float
    metrics: SensitivityMetrics
    score: float | None
    percent_improvement: float | None
    warning: str
    rank: int | None = None
    decision: DesignDecision = DesignDecision()
    constructability_score: float | None = None
    combined_score: float | None = None
    apply_payload: dict[str, object] | None = None


@dataclass(frozen=True)
class SensitivityRecommendation:
    best_option: str
    most_sensitive_parameter: str
    worst_option: str
    next_design_adjustment: str
    key_warning: str


@dataclass(frozen=True)
class SensitivityRunResult:
    mode: SensitivityRunMode
    baseline: SensitivityScenarioResult
    scenarios: list[SensitivityScenarioResult]
    recommendation: SensitivityRecommendation
    criteria: DesignCriteria | None = None


@dataclass(frozen=True)
class _TransitionScenarioArtifacts:
    transition: TransitionRunResult
    analysis: AnalysisResult


def build_scenarios(
    *,
    base_config: AnalysisConfig,
    variables: Sequence[SensitivityVariable],
    mode: SensitivityRunMode = SensitivityRunMode.STATIC,
    transition_context: SensitivityTransitionContext | None = None,
    additional_scenarios: Sequence[SensitivityScenario] | None = None,
) -> list[SensitivityScenario]:
    """Build one-variable-at-a-time sweep scenarios."""
    if mode == SensitivityRunMode.TRANSITION and transition_context is None:
        raise ValueError("transition_context is required for transition sensitivity")

    selected = list(dict.fromkeys(variables))
    scenarios = [
        SensitivityScenario(
            name="Baseline",
            changed_parameter="baseline",
            parameter_value=None,
            factor=1.0,
            analysis_config=base_config,
            transition_context=transition_context,
        )
    ]
    for variable in selected:
        if variable == SensitivityVariable.TRANSITION_LENGTH and mode != SensitivityRunMode.TRANSITION:
            continue
        if variable == SensitivityVariable.AS5100_POSITION:
            for offset_m in AS5100_POSITION_OFFSETS_M:
                scenarios.append(
                    _build_as5100_position_scenario(
                        base_config=base_config,
                        offset_m=offset_m,
                        transition_context=transition_context,
                    )
                )
            continue
        for factor in SWEEP_FACTORS:
            if math.isclose(factor, 1.0):
                continue
            scenarios.append(
                _build_factor_scenario(
                    base_config=base_config,
                    variable=variable,
                    factor=factor,
                    transition_context=transition_context,
                )
            )
    scenarios.extend(additional_scenarios or [])
    return scenarios


def run_sensitivity(
    *,
    base_config: AnalysisConfig,
    variables: Sequence[SensitivityVariable],
    mode: SensitivityRunMode = SensitivityRunMode.STATIC,
    transition_context: SensitivityTransitionContext | None = None,
    criteria: DesignCriteria | None = None,
    additional_scenarios: Sequence[SensitivityScenario] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> SensitivityRunResult:
    scenarios = build_scenarios(
        base_config=base_config,
        variables=variables,
        mode=mode,
        transition_context=transition_context,
        additional_scenarios=additional_scenarios,
    )
    total = len(scenarios)
    raw_results: list[SensitivityScenarioResult] = []
    for index, scenario in enumerate(scenarios, start=1):
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Sensitivity run cancelled.")
        if progress_callback is not None:
            progress_callback(index, total, scenario.name)
        raw_results.append(_run_scenario(scenario, mode=mode))

    baseline = raw_results[0]
    scored = _score_results(baseline, raw_results, criteria=criteria)
    recommendation = build_recommendation(scored[0], scored[1:])
    return SensitivityRunResult(
        mode=mode,
        baseline=scored[0],
        scenarios=scored,
        recommendation=recommendation,
        criteria=criteria,
    )


def rescore_sensitivity_result(
    result: SensitivityRunResult,
    *,
    criteria: DesignCriteria | None,
) -> SensitivityRunResult:
    """Re-evaluate screening criteria without rerunning solver scenarios."""
    if not result.scenarios:
        return replace(result, criteria=criteria)
    scored = _score_results(result.scenarios[0], result.scenarios, criteria=criteria)
    recommendation = build_recommendation(scored[0], scored[1:])
    return replace(
        result,
        baseline=scored[0],
        scenarios=scored,
        recommendation=recommendation,
        criteria=criteria,
    )


def write_sensitivity_csv(path: str | Path, result: SensitivityRunResult) -> None:
    """Write sensitivity results and recommendation text to CSV."""
    export_path = Path(path)
    with export_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "field", "value"])
        writer.writerow(["metadata", "mode", result.mode.value])
        writer.writerow(["baseline", "scenario", result.baseline.scenario_name])
        writer.writerow(["recommendation", "best_option", result.recommendation.best_option])
        writer.writerow(
            ["recommendation", "most_sensitive_parameter", result.recommendation.most_sensitive_parameter]
        )
        writer.writerow(["recommendation", "worst_option", result.recommendation.worst_option])
        writer.writerow(
            ["recommendation", "next_design_adjustment", result.recommendation.next_design_adjustment]
        )
        writer.writerow(["recommendation", "key_warning", result.recommendation.key_warning])
        writer.writerow([])
        writer.writerow(
            [
                "scenario_name",
                "changed_parameter",
                "parameter_value_si",
                "factor",
                "max_deflection_m",
                "max_moment_nm",
                "max_sleeper_load_n",
                "rail_stress_pa",
                "ballast_pressure_pa",
                "formation_pressure_pa",
                "subgrade_pressure_pa",
                "deep_subgrade_pressure_pa",
                "transition_metric_m",
                "score",
                "constructability_score",
                "combined_score",
                "percent_improvement",
                "decision_status",
                "governing_criterion",
                "max_utilization",
                "warning_status",
                "rank",
            ]
        )
        for scenario in result.scenarios:
            writer.writerow(
                [
                    scenario.scenario_name,
                    scenario.changed_parameter,
                    _format_optional(scenario.parameter_value),
                    _format_optional(scenario.factor),
                    _format_optional(scenario.metrics.max_deflection_m),
                    _format_optional(scenario.metrics.max_moment_nm),
                    _format_optional(scenario.metrics.max_sleeper_load_n),
                    _format_optional(scenario.metrics.rail_stress_pa),
                    _format_optional(scenario.metrics.ballast_pressure_pa),
                    _format_optional(scenario.metrics.formation_pressure_pa),
                    _format_optional(scenario.metrics.subgrade_pressure_pa),
                    _format_optional(scenario.metrics.deep_subgrade_pressure_pa),
                    _format_optional(scenario.metrics.transition_metric_m),
                    _format_optional(scenario.score),
                    _format_optional(scenario.constructability_score),
                    _format_optional(scenario.combined_score),
                    _format_optional(scenario.percent_improvement),
                    scenario.decision.status,
                    scenario.decision.governing_criterion,
                    _format_optional(scenario.decision.max_utilization),
                    scenario.warning,
                    "" if scenario.rank is None else str(scenario.rank),
                ]
            )


def build_recommendation(
    baseline: SensitivityScenarioResult,
    scenarios: Sequence[SensitivityScenarioResult],
) -> SensitivityRecommendation:
    valid = [scenario for scenario in scenarios if scenario.score is not None and math.isfinite(scenario.score)]
    if not valid:
        return SensitivityRecommendation(
            best_option="No valid non-baseline option.",
            most_sensitive_parameter="No valid sensitivity ranking available.",
            worst_option="No valid non-baseline option.",
            next_design_adjustment="Review baseline inputs and rerun the screening study.",
            key_warning="No valid sensitivity scenarios completed.",
        )
    best = min(valid, key=_ranking_key)
    worst = max(valid, key=lambda item: item.decision.max_utilization or item.combined_score or item.score or 0.0)
    sensitivity_by_parameter: dict[str, float] = {}
    for scenario in valid:
        if scenario.percent_improvement is None:
            continue
        sensitivity_by_parameter[scenario.changed_parameter] = max(
            sensitivity_by_parameter.get(scenario.changed_parameter, 0.0),
            abs(scenario.percent_improvement),
        )
    most_sensitive = max(sensitivity_by_parameter.items(), key=lambda item: item[1])[0]
    warning = next((scenario.warning for scenario in valid if scenario.warning), "")
    if not warning:
        warning = "Screening recommendation only; confirm with detailed design checks."
    adjustment = _adjustment_text(best)
    if best.decision.governing_criterion:
        adjustment = f"{adjustment} Governing criterion: {best.decision.governing_criterion}."
    return SensitivityRecommendation(
        best_option=_scenario_label(best),
        most_sensitive_parameter=most_sensitive.replace("_", " "),
        worst_option=_scenario_label(worst),
        next_design_adjustment=adjustment,
        key_warning=warning,
    )


def _build_factor_scenario(
    *,
    base_config: AnalysisConfig,
    variable: SensitivityVariable,
    factor: float,
    transition_context: SensitivityTransitionContext | None,
) -> SensitivityScenario:
    config = base_config
    context = transition_context
    value: float | None
    if variable == SensitivityVariable.SUPPORT_STIFFNESS:
        value = _positive_scaled(config.foundation_modulus_n_per_m2, factor, variable.value)
        config = replace(config, foundation_modulus_n_per_m2=value)
        if context is not None:
            context = replace(
                context,
                k1_n_per_m2=value,
                analysis_config=replace(context.analysis_config, foundation_modulus_n_per_m2=value),
                k_profile_n_per_m2=None,
            )
    elif variable == SensitivityVariable.SLEEPER_SPACING:
        value = _positive_scaled(config.sleeper_spacing_m, factor, variable.value)
        config = replace(config, sleeper_spacing_m=value)
        if context is not None:
            context = replace(context, analysis_config=replace(context.analysis_config, sleeper_spacing_m=value))
        apply_payload = {"sleeper_spacing_m": value}
    elif variable == SensitivityVariable.PAD_STIFFNESS:
        pad_value = config.pad_stiffness_n_per_m or config.railpad_stiffness_n_per_m
        if pad_value is None:
            raise ValueError("pad_stiffness is unavailable for the selected analysis context")
        value = _positive_scaled(pad_value, factor, variable.value)
        config = replace(config, pad_stiffness_n_per_m=value, railpad_stiffness_n_per_m=value)
        if context is not None:
            context = replace(
                context,
                analysis_config=replace(
                    context.analysis_config,
                    pad_stiffness_n_per_m=value,
                    railpad_stiffness_n_per_m=value,
                ),
            )
    elif variable == SensitivityVariable.WHEEL_LOAD:
        value = None
        loads = []
        for load in config.loads:
            scaled_load = _positive_scaled(abs(load.load_newtons), factor, variable.value)
            loads.append(PointLoad(position_m=load.position_m, load_newtons=math.copysign(scaled_load, load.load_newtons)))
        config = replace(config, loads=loads)
        value = max(abs(load.load_newtons) for load in loads) if loads else None
        if context is not None:
            context = replace(context, analysis_config=replace(context.analysis_config, loads=loads))
    elif variable == SensitivityVariable.SPEED:
        design_inputs = config.design_inputs
        if design_inputs is None:
            raise ValueError("speed is unavailable because design inputs are missing")
        value = _positive_scaled(design_inputs.speed_kmh, factor, variable.value)
        updated = replace(design_inputs, speed_kmh=value)
        config = replace(config, design_inputs=updated)
        if context is not None:
            context = replace(context, analysis_config=replace(context.analysis_config, design_inputs=updated))
    elif variable == SensitivityVariable.BALLAST_DEPTH:
        design_inputs = _require_design_inputs(config)
        baseline = design_inputs.ballast_depth_m
        if baseline is None:
            raise ValueError("ballast_depth is unavailable because the baseline value is missing")
        value = _positive_scaled(baseline, factor, variable.value)
        updated = replace(design_inputs, ballast_depth_m=value)
        config = replace(config, design_inputs=updated)
        if context is not None:
            context = replace(context, analysis_config=replace(context.analysis_config, design_inputs=updated))
    elif variable == SensitivityVariable.TRANSITION_LENGTH:
        if context is None:
            raise ValueError("transition_length is only available for transition sensitivity")
        if context.transition_length_m is None:
            raise ValueError("transition_length is unavailable for this transition profile")
        value = _positive_scaled(context.transition_length_m, factor, variable.value)
        context = replace(context, transition_length_m=value, k_profile_n_per_m2=None)
    else:
        raise ValueError(f"Unsupported sensitivity variable: {variable}")

    if variable != SensitivityVariable.SLEEPER_SPACING:
        apply_payload = None
    return SensitivityScenario(
        name=_scenario_name(variable.value, factor),
        changed_parameter=variable.value,
        parameter_value=value,
        factor=factor,
        analysis_config=config,
        transition_context=context,
        apply_payload=apply_payload,
    )


def _build_as5100_position_scenario(
    *,
    base_config: AnalysisConfig,
    offset_m: float,
    transition_context: SensitivityTransitionContext | None,
) -> SensitivityScenario:
    loads = [
        PointLoad(position_m=load.position_m + offset_m, load_newtons=load.load_newtons)
        for load in base_config.loads
    ]
    config = replace(base_config, loads=loads)
    context = transition_context
    if context is not None:
        context = replace(context, analysis_config=replace(context.analysis_config, loads=loads))
    return SensitivityScenario(
        name=f"AS5100 train shifted {offset_m:+.1f} m",
        changed_parameter=SensitivityVariable.AS5100_POSITION.value,
        parameter_value=offset_m,
        factor=1.0,
        analysis_config=config,
        transition_context=context,
        constructability_class="load_position_screening",
    )


def _run_scenario(
    scenario: SensitivityScenario,
    *,
    mode: SensitivityRunMode,
) -> SensitivityScenarioResult:
    try:
        _validate_config(scenario.analysis_config)
        if mode == SensitivityRunMode.TRANSITION:
            result = _run_transition_scenario(scenario)
            metrics = _metrics_from_transition(result, scenario.analysis_config)
        else:
            result = run_analysis(scenario.analysis_config, mode=AnalysisMode.NUMERICAL if _needs_numerical(scenario.analysis_config) else AnalysisMode.CLOSED_FORM)
            metrics = _metrics_from_static(result, scenario.analysis_config)
        warning = ""
    except Exception as exc:
        metrics = SensitivityMetrics()
        warning = str(exc)
    return SensitivityScenarioResult(
        scenario_name=scenario.name,
        changed_parameter=scenario.changed_parameter,
        parameter_value=scenario.parameter_value,
        factor=scenario.factor,
        metrics=metrics,
        score=None,
        percent_improvement=None,
        warning=warning,
        apply_payload=scenario.apply_payload,
        constructability_score=_constructability_score(
            scenario.changed_parameter,
            scenario.factor,
            scenario.apply_payload,
        ),
    )


def _run_transition_scenario(scenario: SensitivityScenario) -> _TransitionScenarioArtifacts:
    context = scenario.transition_context
    if context is None:
        raise ValueError("transition context is missing")
    if context.run_mode != TransitionRunMode.SINGLE:
        raise ValueError("Stage 1 transition sensitivity supports single transition runs.")
    config = context.analysis_config
    k_profile = None
    if context.profile_type != TransitionProfileType.UNIFORM:
        x_values = _preview_x_values(config)
        k_profile = build_transition_profile(
            x_values=x_values,
            profile_type=context.profile_type,
            k1_n_per_m2=context.k1_n_per_m2,
            k2_n_per_m2=context.k2_n_per_m2,
            transition_length_m=context.transition_length_m,
            segment_length_m=context.segment_length_m,
        )
        config = replace(config, foundation_profile_n_per_m2=k_profile)
    result = run_analysis(config, mode=context.analysis_mode)
    if k_profile is None or len(k_profile) != len(result.x_m):
        k_profile = build_transition_profile(
            x_values=result.x_m,
            profile_type=context.profile_type,
            k1_n_per_m2=context.k1_n_per_m2,
            k2_n_per_m2=context.k2_n_per_m2,
            transition_length_m=context.transition_length_m,
            segment_length_m=context.segment_length_m,
        )
    metrics = compute_metrics_from_series(
        x_values=result.x_m,
        deflection_m=result.deflection_m,
        moment_nm=result.moment_nm,
        reaction_n_per_m=result.reaction_n_per_m,
        sleeper_positions_m=result.sleeper_positions_m,
        sleeper_loads_n=result.sleeper_loads_n,
        sleeper_spacing_m=config.sleeper_spacing_m,
        elastic_modulus_pa=config.elastic_modulus_pa,
        moment_inertia_m4=config.moment_inertia_m4,
    )
    energy_metrics = None
    energy_series = None
    try:
        p_ref_n = max((abs(load.load_newtons) for load in config.loads), default=0.0) or None
        energy_metrics, energy_series = compute_energy_from_series(
            x_values=result.x_m,
            k_profile_n_per_m2=k_profile,
            deflection_m=result.deflection_m,
            moment_nm=result.moment_nm,
            sleeper_spacing_m=config.sleeper_spacing_m,
            elastic_modulus_pa=config.elastic_modulus_pa,
            moment_inertia_m4=config.moment_inertia_m4,
            p_ref_n=p_ref_n,
        )
    except ValueError:
        energy_metrics = None
        energy_series = None
    series = build_series_from_single(
        x_values=result.x_m,
        k_profile_n_per_m2=k_profile,
        deflection_m=result.deflection_m,
        moment_nm=result.moment_nm,
        reaction_n_per_m=result.reaction_n_per_m,
        shear_n=result.shear_n,
    )
    transition = TransitionRunResult(
        mode=context.run_mode,
        profile_type=context.profile_type,
        template_name=context.template_name,
        preset_name=context.preset_name,
        k1_n_per_m2=context.k1_n_per_m2,
        k2_n_per_m2=context.k2_n_per_m2,
        transition_length_m=context.transition_length_m,
        segment_length_m=context.segment_length_m,
        domain_length_m=context.domain_m[1] - context.domain_m[0],
        metrics=metrics,
        series=series,
        energy_metrics=energy_metrics,
        energy_series=energy_series,
    )
    return _TransitionScenarioArtifacts(transition=transition, analysis=result)


def _score_results(
    baseline: SensitivityScenarioResult,
    results: Sequence[SensitivityScenarioResult],
    *,
    criteria: DesignCriteria | None = None,
) -> list[SensitivityScenarioResult]:
    scored: list[SensitivityScenarioResult] = []
    for result in results:
        score = _score_result(baseline.metrics, result.metrics)
        if score is not None and result.warning:
            score += WARNING_SCORE_PENALTY
        decision = evaluate_design_criteria(result.metrics, criteria)
        constructability = result.constructability_score
        combined = _combined_score(score, constructability, decision)
        improvement = None
        if score is not None:
            improvement = (1.0 - score) * 100.0
        scored.append(
            replace(
                result,
                score=score,
                percent_improvement=improvement,
                decision=decision,
                constructability_score=constructability,
                combined_score=combined,
            )
        )

    ranked = sorted(
        [item for item in scored if item.score is not None and not item.warning],
        key=_ranking_key,
    )
    rank_by_name = {item.scenario_name: index for index, item in enumerate(ranked, start=1)}
    return [
        replace(item, rank=rank_by_name.get(item.scenario_name))
        for item in scored
    ]


def _score_result(
    baseline: SensitivityMetrics,
    metrics: SensitivityMetrics,
) -> float | None:
    pairs = {
        "deflection": (baseline.max_deflection_m, metrics.max_deflection_m),
        "moment_stress": (
            baseline.rail_stress_pa or baseline.max_moment_nm,
            metrics.rail_stress_pa or metrics.max_moment_nm,
        ),
        "sleeper_load": (baseline.max_sleeper_load_n, metrics.max_sleeper_load_n),
        "ballast_pressure": (baseline.ballast_pressure_pa, metrics.ballast_pressure_pa),
        "formation_pressure": (baseline.formation_pressure_pa, metrics.formation_pressure_pa),
        "subgrade_pressure": (baseline.subgrade_pressure_pa, metrics.subgrade_pressure_pa),
        "deep_subgrade_pressure": (
            baseline.deep_subgrade_pressure_pa,
            metrics.deep_subgrade_pressure_pa,
        ),
        "transition_metric": (baseline.transition_metric_m, metrics.transition_metric_m),
    }
    available = {
        key: (base, value)
        for key, (base, value) in pairs.items()
        if _usable_metric(base) and _usable_metric(value)
    }
    if not available:
        return None
    weight_total = sum(DEFAULT_WEIGHTS[key] for key in available)
    if weight_total <= 0.0:
        return None
    score = 0.0
    for key, (base, value) in available.items():
        weight = DEFAULT_WEIGHTS[key] / weight_total
        score += weight * (abs(value) / abs(base))
    return score


def evaluate_design_criteria(
    metrics: SensitivityMetrics,
    criteria: DesignCriteria | None,
) -> DesignDecision:
    if criteria is None:
        return DesignDecision()
    pairs = {
        "deflection": (metrics.max_deflection_m, criteria.max_deflection_m),
        "rail stress": (metrics.rail_stress_pa, criteria.rail_stress_pa),
        "sleeper load": (metrics.max_sleeper_load_n, criteria.max_sleeper_load_n),
        "ballast pressure": (metrics.ballast_pressure_pa, criteria.ballast_pressure_pa),
        "formation stress": (metrics.formation_pressure_pa, criteria.formation_pressure_pa),
        "subgrade stress": (metrics.subgrade_pressure_pa, criteria.subgrade_pressure_pa),
        "deep subgrade stress": (
            metrics.deep_subgrade_pressure_pa,
            criteria.deep_subgrade_pressure_pa,
        ),
        "transition metric": (metrics.transition_metric_m, criteria.transition_metric_m),
    }
    utilizations = {
        key: abs(actual) / abs(limit)
        for key, (actual, limit) in pairs.items()
        if _usable_metric(actual) and _usable_metric(limit)
    }
    if not utilizations:
        return DesignDecision()
    governing, max_utilization = max(utilizations.items(), key=lambda item: item[1])
    if max_utilization > 1.0:
        status = "fail"
    elif max_utilization >= 0.85:
        status = "warning"
    else:
        status = "pass"
    return DesignDecision(
        status=status,
        governing_criterion=governing,
        max_utilization=max_utilization,
        utilizations=utilizations,
    )


def _constructability_score(
    changed_parameter: str,
    factor: float,
    apply_payload: Mapping[str, object] | None,
) -> float:
    if changed_parameter == "baseline":
        return 0.0
    difficulty = CONSTRUCTABILITY_DIFFICULTY.get(changed_parameter, 0.70)
    magnitude = min(abs(factor - 1.0), 1.0)
    if apply_payload:
        difficulty *= 0.85
    return min(1.0, difficulty * (0.75 + magnitude))


def _combined_score(
    engineering_score: float | None,
    constructability_score: float | None,
    decision: DesignDecision,
) -> float | None:
    if engineering_score is None:
        return None
    constructability = constructability_score if constructability_score is not None else 0.70
    utilization = decision.max_utilization if decision.max_utilization is not None else 1.0
    status_penalty = {
        "pass": 0.0,
        "warning": 0.15,
        "fail": 0.50,
        "unrated": 0.05,
    }.get(decision.status, 0.05)
    return 0.65 * engineering_score + 0.20 * constructability + 0.15 * utilization + status_penalty


def _ranking_key(scenario: SensitivityScenarioResult) -> tuple[int, float, float]:
    return (
        DECISION_STATUS_ORDER.get(scenario.decision.status, 3),
        scenario.decision.max_utilization if scenario.decision.max_utilization is not None else math.inf,
        scenario.combined_score if scenario.combined_score is not None else scenario.score or math.inf,
    )


def _metrics_from_static(result: AnalysisResult, config: AnalysisConfig) -> SensitivityMetrics:
    pressure_metrics = _pressure_metrics_from_analysis(result, config)
    return SensitivityMetrics(
        max_deflection_m=abs(result.summary.max_deflection.value),
        max_moment_nm=abs(result.summary.max_moment.value),
        max_sleeper_load_n=abs(result.summary.max_sleeper_load.value),
        rail_stress_pa=abs(result.summary.max_rail_base_stress_pa),
        ballast_pressure_pa=pressure_metrics["ballast_pressure_pa"],
        formation_pressure_pa=pressure_metrics["formation_pressure_pa"],
        subgrade_pressure_pa=pressure_metrics["subgrade_pressure_pa"],
        deep_subgrade_pressure_pa=pressure_metrics["deep_subgrade_pressure_pa"],
    )


def _metrics_from_transition(
    result: _TransitionScenarioArtifacts,
    config: AnalysisConfig,
) -> SensitivityMetrics:
    pressure_metrics = _pressure_metrics_from_analysis(result.analysis, config)
    return SensitivityMetrics(
        max_deflection_m=_max_abs(result.transition.series.deflection_m),
        max_moment_nm=abs(result.transition.metrics.moment_max_nm),
        max_sleeper_load_n=abs(result.transition.metrics.sleeper_load_max_n),
        rail_stress_pa=abs(result.analysis.summary.max_rail_base_stress_pa),
        ballast_pressure_pa=pressure_metrics["ballast_pressure_pa"],
        formation_pressure_pa=pressure_metrics["formation_pressure_pa"],
        subgrade_pressure_pa=pressure_metrics["subgrade_pressure_pa"],
        deep_subgrade_pressure_pa=pressure_metrics["deep_subgrade_pressure_pa"],
        transition_metric_m=abs(result.transition.metrics.delta_w_s_m),
    )


def _pressure_metrics_from_analysis(
    result: AnalysisResult,
    config: AnalysisConfig,
) -> dict[str, float | None]:
    design_summary = result.summary.design_summary
    a3902 = design_summary.a3902_checks if design_summary is not None else None
    deep_subgrade = None
    design_inputs = config.design_inputs
    if a3902 is not None and design_inputs is not None and design_inputs.ballast_depth_m:
        try:
            deep_subgrade = subgrade_pressure_a3902(
                ballast_contact_pressure_pa=a3902.ballast_contact_pressure_pa,
                ballast_depth_m=design_inputs.ballast_depth_m,
                fill_depth_m=max(design_inputs.fill_depth_m, 1.0),
                sleeper_width_m=config.sleeper_width_m,
                effective_bearing_length_m=a3902.effective_bearing_length_m,
            )
        except ValueError:
            deep_subgrade = None
    return {
        "ballast_pressure_pa": abs(result.summary.max_sleeper_pressure.value),
        "formation_pressure_pa": (
            abs(a3902.formation_pressure_pa)
            if a3902 is not None and a3902.formation_pressure_pa is not None
            else None
        ),
        "subgrade_pressure_pa": (
            abs(a3902.subgrade_pressure_pa)
            if a3902 is not None and a3902.subgrade_pressure_pa is not None
            else None
        ),
        "deep_subgrade_pressure_pa": abs(deep_subgrade) if deep_subgrade is not None else None,
    }


def _needs_numerical(config: AnalysisConfig) -> bool:
    return (
        config.foundation_profile_n_per_m2 is not None
        or config.foundation_profile_type.value != "uniform"
        or config.pasternak_shear_n > 0.0
        or config.use_discrete_supports
        or config.use_two_rail
        or config.foundation_model.value != "winkler"
        or config.beam_theory.value != "euler"
    )


def _preview_x_values(config: AnalysisConfig) -> list[float]:
    if config.x_domain_m is None:
        return [0.0]
    sample_count = max(config.sample_count, 2)
    start, end = config.x_domain_m
    step = (end - start) / (sample_count - 1)
    return [start + index * step for index in range(sample_count)]


def _validate_config(config: AnalysisConfig) -> None:
    positive_values = {
        "foundation_modulus_n_per_m2": config.foundation_modulus_n_per_m2,
        "elastic_modulus_pa": config.elastic_modulus_pa,
        "moment_inertia_m4": config.moment_inertia_m4,
        "section_modulus_m3": config.section_modulus_m3,
        "sleeper_spacing_m": config.sleeper_spacing_m,
        "sleeper_length_m": config.sleeper_length_m,
        "sleeper_width_m": config.sleeper_width_m,
    }
    for name, value in positive_values.items():
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be a positive finite value")
    if not config.loads:
        raise ValueError("at least one wheel load is required")
    for load in config.loads:
        if load.load_newtons == 0.0 or not math.isfinite(load.load_newtons):
            raise ValueError("wheel loads must be non-zero finite values")


def _require_design_inputs(config: AnalysisConfig) -> DesignInputs:
    if config.design_inputs is None:
        raise ValueError("design inputs are missing")
    return config.design_inputs


def _positive_scaled(value: float, factor: float, name: str) -> float:
    candidate = value * factor
    if candidate <= 0.0 or not math.isfinite(candidate):
        raise ValueError(f"{name} sweep produced an invalid value")
    return candidate


def _usable_metric(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and abs(value) > 0.0


def _max_abs(values: Sequence[float] | None) -> float | None:
    if not values:
        return None
    return max(abs(value) for value in values)


def _scenario_label(scenario: SensitivityScenarioResult) -> str:
    return scenario.scenario_name


def _scenario_name(parameter: str, factor: float) -> str:
    change = (factor - 1.0) * 100.0
    sign = "+" if change > 0.0 else ""
    return f"{parameter.replace('_', ' ')} {sign}{change:.0f}%"


def _adjustment_text(best: SensitivityScenarioResult) -> str:
    direction = "increase" if best.factor > 1.0 else "decrease"
    parameter = best.changed_parameter.replace("_", " ")
    if best.changed_parameter == "wheel_load":
        return "Review wheel load assumptions; load reduction is operational rather than a track design change."
    if best.changed_parameter == "as5100_position":
        return "Review the AS5100 train reference position used for the governing fixed-arrangement check."
    if best.changed_parameter == "speed":
        return "Review operating speed limits for this baseline, then confirm with detailed checks."
    return f"Consider a controlled {direction} in {parameter}, then rerun detailed static and transition checks."


def _format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.10g}"


def result_payload(result: SensitivityRunResult) -> dict[str, object]:
    """Return a JSON-serializable result payload for metadata sidecars."""
    return asdict(result)

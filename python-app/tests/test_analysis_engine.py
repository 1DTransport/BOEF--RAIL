import math
from dataclasses import replace

import pytest

from core.analysis import AnalysisInputs, compute_track_response
from core.analysis_engine import (
    AnalysisConfig,
    AnalysisMode,
    BeamTheory,
    FoundationModelType,
    FoundationProfileType,
    run_analysis,
)
from core.foundation.base import equivalent_series_stiffness, per_support_to_per_length
from core.load_builder import TrainLoadConfig, build_train_loads
from core.model import PointLoad


def _build_config() -> AnalysisConfig:
    return AnalysisConfig(
        loads=[PointLoad(position_m=0.0, load_newtons=80_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=201,
    )


def test_closed_form_backend_matches_legacy() -> None:
    config = _build_config()
    legacy = compute_track_response(
        AnalysisInputs(
            loads=config.loads,
            foundation_modulus_n_per_m2=config.foundation_modulus_n_per_m2,
            elastic_modulus_pa=config.elastic_modulus_pa,
            moment_inertia_m4=config.moment_inertia_m4,
            section_modulus_m3=config.section_modulus_m3,
            sleeper_spacing_m=config.sleeper_spacing_m,
            sleeper_length_m=config.sleeper_length_m,
            sleeper_width_m=config.sleeper_width_m,
            sample_count=config.sample_count,
        )
    )

    result = run_analysis(config, mode=AnalysisMode.CLOSED_FORM)

    assert result.x_m == legacy.x_m
    assert result.deflection_m == legacy.deflection_m
    assert result.moment_nm == legacy.moment_nm
    assert result.shear_n == legacy.shear_n
    assert result.reaction_n_per_m == legacy.reaction_n_per_m


def test_numerical_backend_uniform_is_finite() -> None:
    config = _build_config()

    result = run_analysis(config, mode=AnalysisMode.NUMERICAL)

    assert len(result.x_m) == config.sample_count
    assert len(result.deflection_m) == config.sample_count
    assert len(result.moment_nm) == config.sample_count
    assert len(result.shear_n) == config.sample_count
    assert len(result.reaction_n_per_m) == config.sample_count


def test_timoshenko_rejects_pasternak_layer() -> None:
    config = replace(
        _build_config(),
        beam_theory=BeamTheory.TIMOSHENKO,
        pasternak_shear_n=10_000.0,
    )
    with pytest.raises(ValueError, match="Pasternak shear layer is not supported"):
        run_analysis(config, mode=AnalysisMode.NUMERICAL)


def test_numerical_backend_matches_closed_form_within_tolerance() -> None:
    config = _build_config()

    closed_form = run_analysis(config, mode=AnalysisMode.CLOSED_FORM)
    numerical = run_analysis(config, mode=AnalysisMode.NUMERICAL)

    def max_abs(values: list[float]) -> float:
        return max(values, key=lambda value: abs(value))

    assert math.isclose(
        max_abs(numerical.deflection_m),
        max_abs(closed_form.deflection_m),
        rel_tol=0.2,
    )
    assert math.isclose(
        max_abs(numerical.moment_nm),
        max_abs(closed_form.moment_nm),
        rel_tol=0.2,
    )


def test_nonuniform_step_profile_is_asymmetric() -> None:
    config = _build_config()
    config = replace(
        config,
        foundation_profile_type=FoundationProfileType.STEP,
        foundation_profile_k1_n_per_m2=40_000_000.0,
        foundation_profile_k2_n_per_m2=80_000_000.0,
        foundation_profile_x_start_m=0.0,
    )

    result = run_analysis(config, mode=AnalysisMode.NUMERICAL)

    index_left = min(range(len(result.x_m)), key=lambda i: abs(result.x_m[i] + 1.0))
    index_right = min(range(len(result.x_m)), key=lambda i: abs(result.x_m[i] - 1.0))
    assert not math.isclose(
        result.deflection_m[index_left],
        result.deflection_m[index_right],
        rel_tol=1e-3,
    )


def test_discrete_sleepers_return_seat_loads() -> None:
    config = replace(
        _build_config(),
        use_discrete_supports=True,
        pad_stiffness_n_per_m=120_000_000.0,
        nodes_between_sleepers=8,
    )

    result = run_analysis(config, mode=AnalysisMode.NUMERICAL)

    assert len(result.sleeper_positions_m) == len(result.sleeper_loads_n)
    assert len(result.sleeper_loads_n) > 0
    total_load = sum(load.load_newtons for load in config.loads)
    total_sleeper = sum(result.sleeper_loads_n)
    assert math.isclose(total_sleeper, total_load, rel_tol=0.2)


def test_two_rail_outputs_differ_for_asymmetric_loads() -> None:
    config = replace(
        _build_config(),
        use_two_rail=True,
        coupling_stiffness_n_per_m=50_000_000.0,
        right_loads=[PointLoad(position_m=0.0, load_newtons=20_000.0)],
    )

    result = run_analysis(config, mode=AnalysisMode.NUMERICAL)

    assert result.left_deflection_m is not None
    assert result.right_deflection_m is not None
    assert any(
        abs(left - right) > 1e-9
        for left, right in zip(result.left_deflection_m, result.right_deflection_m)
    )


def test_two_rail_symmetric_axle_source_applies_half_load_to_each_rail() -> None:
    loads = build_train_loads(
        TrainLoadConfig(
            axle_load_n=120_000.0,
            bogie_count=1,
            bogie_spacing_m=0.0,
            axles_per_bogie=1,
            axle_spacing_m=0.0,
        )
    )
    config = replace(
        _build_config(),
        loads=loads,
        use_two_rail=True,
        coupling_stiffness_n_per_m=50_000_000.0,
    )

    result = run_analysis(config, mode=AnalysisMode.NUMERICAL)

    assert [load.load_newtons for load in loads] == pytest.approx([60_000.0])
    assert result.left_deflection_m is not None
    assert result.right_deflection_m is not None
    peak_deflection = max(abs(value) for value in result.left_deflection_m)
    max_left_right_delta = max(
        abs(left - right)
        for left, right in zip(result.left_deflection_m, result.right_deflection_m)
    )
    assert max_left_right_delta <= peak_deflection * 1.0e-3
    assert result.left_sleeper_loads_n is not None
    assert result.right_sleeper_loads_n is not None
    assert sum(result.left_sleeper_loads_n) == pytest.approx(60_000.0, rel=0.05)
    assert sum(result.right_sleeper_loads_n) == pytest.approx(60_000.0, rel=0.05)
    assert sum(result.sleeper_loads_n) == pytest.approx(120_000.0, rel=0.05)


def test_pasternak_shear_changes_response() -> None:
    config = _build_config()
    base = run_analysis(config, mode=AnalysisMode.NUMERICAL)
    with_shear = run_analysis(
        replace(config, pasternak_shear_n=2_000_000.0),
        mode=AnalysisMode.NUMERICAL,
    )

    mid = len(base.x_m) // 2
    assert not math.isclose(
        base.deflection_m[mid],
        with_shear.deflection_m[mid],
        rel_tol=1e-4,
    )


def test_series_foundation_uses_equivalent_stiffness() -> None:
    config = replace(
        _build_config(),
        foundation_model=FoundationModelType.SERIES,
        railpad_stiffness_n_per_m=20_000_000.0,
        trackbed_stiffness_n_per_m=10_000_000.0,
    )
    result = run_analysis(config, mode=AnalysisMode.NUMERICAL)
    k_pad = per_support_to_per_length(config.railpad_stiffness_n_per_m, config.sleeper_spacing_m)
    k_bed = per_support_to_per_length(config.trackbed_stiffness_n_per_m, config.sleeper_spacing_m)
    expected = equivalent_series_stiffness(k_pad, k_bed)
    assert result.summary.support_k_eq_n_per_m2 == pytest.approx(expected)
    assert result.summary.support_model == "Series (railpad + trackbed)"


def test_series_foundation_changes_deflection_vs_winkler() -> None:
    base = _build_config()
    winkler = run_analysis(base, mode=AnalysisMode.NUMERICAL)
    series = run_analysis(
        replace(
            base,
            foundation_model=FoundationModelType.SERIES,
            railpad_stiffness_n_per_m=5_000_000.0,
            trackbed_stiffness_n_per_m=5_000_000.0,
        ),
        mode=AnalysisMode.NUMERICAL,
    )
    mid = len(series.x_m) // 2
    assert not math.isclose(
        winkler.deflection_m[mid],
        series.deflection_m[mid],
        rel_tol=1e-3,
    )

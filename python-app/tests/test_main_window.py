import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from matplotlib.text import Annotation

if os.environ.get("BOEF_ENABLE_GUI_TESTS", "").lower() not in {"1", "true", "yes"}:
    pytest.skip("Set BOEF_ENABLE_GUI_TESTS=1 to run PySide GUI tests.", allow_module_level=True)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip(
    "PySide6.QtWidgets",
    reason="PySide6 (libGL) is required for GUI tests.",
    exc_type=ImportError,
)
QApplication = QtWidgets.QApplication
QtCore = pytest.importorskip(
    "PySide6.QtCore",
    reason="PySide6 (QtCore) is required for GUI tests.",
    exc_type=ImportError,
)
Qt = QtCore.Qt

from app import main
from app import sensitivity_dialog
from core.analysis_engine import (
    AnalysisConfig,
    AnalysisMode,
    FoundationModelType,
    FoundationProfileType,
    run_analysis,
)
from core.analysis import Extremum as StaticExtremum, compute_track_response
from core.envelope import EnvelopeConfig, EnvelopeResult, EnvelopeSummary
from core.dynamic.config import (
    DippedJointConfig,
    DynamicConfig,
    DynamicMode,
    DynamicTransitionConfig,
    DynamicTransitionProfileType,
    DynamicTransitionRunMode,
)
from core.dynamic.results import (
    DippedJointResult,
    DynamicParameterTrace,
    DynamicResult,
    DynamicSpatialResult,
    DynamicSummary,
    DynamicTimeSeries,
    DynamicTransitionMetrics,
    DynamicTransitionResult,
    DynamicTransitionSeries,
    Extremum,
)
from core.model import PointLoad, beam_parameter_beta
from core.sensitivity import (
    SensitivityMetrics,
    SensitivityRecommendation,
    SensitivityRunMode,
    SensitivityRunResult,
    SensitivityScenarioResult,
    SensitivityVariable,
)
from core.stress_metrics import StressMetadata, StressResults
from core.transition import (
    TransitionEnergySeries,
    TransitionEnergyMetrics,
    TransitionMetrics,
    TransitionProfileType as StaticTransitionProfileType,
    TransitionRunMode as StaticTransitionRunMode,
    TransitionRunResult,
    TransitionSeries,
)
from db import crud
from db.models import DesignAlternative, LoadCase, Pad, Project, Rail, Sleeper, SupportProfile, TrackConfig
from sqlalchemy import select


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _build_dynamic_result(scale: float = 1.0, *, probes: int = 1) -> DynamicResult:
    spatial = DynamicSpatialResult(
        xi_m=[0.0, 1.0, 2.0],
        deflection_m=[0.0, 0.001 * scale, 0.0],
        moment_nm=[0.0, 12.0 * scale, 0.0],
        shear_n=[0.0, 6.0 * scale, 0.0],
        reaction_n_per_m=[0.0, 2.5 * scale, 0.0],
        damping_force_n_per_m=[0.0, 1.5 * scale, 0.0],
    )
    probe_series: list[DynamicTimeSeries] = []
    for index in range(probes):
        factor = scale * (index + 1.0)
        probe_series.append(
            DynamicTimeSeries(
                position_m=float(index),
                time_s=[0.0, 0.5, 1.0],
                deflection_m=[0.0, 0.001 * factor, 0.0],
                moment_nm=[0.0, 12.0 * factor, 0.0],
                shear_n=[0.0, 6.0 * factor, 0.0],
                reaction_n_per_m=[0.0, 2.5 * factor, 0.0],
                damping_force_n_per_m=[0.0, 1.5 * factor, 0.0],
                fft_frequency_hz=[0.0, 5.0, 10.0],
                fft_amplitude=[0.0, 0.001 * factor, 0.0004 * factor],
                psd_frequency_hz=[0.0, 5.0, 10.0],
                psd=[0.0, 0.000001 * factor, 0.0000004 * factor],
                psd_ci_lower=[0.0, 0.0000005 * factor, 0.0000002 * factor],
                psd_ci_upper=[0.0, 0.0000015 * factor, 0.0000006 * factor],
                impedance_frequency_hz=[0.0, 5.0, 10.0],
                impedance_magnitude_n_per_m2=[1.0 * factor, 1.2 * factor, 1.4 * factor],
                impedance_phase_deg=[0.0, 8.0, 16.0],
            )
        )
    summary = DynamicSummary(
        max_deflection=Extremum(0.001 * scale, 1.0),
        max_moment=Extremum(12.0 * scale, 1.0),
        max_shear=Extremum(6.0 * scale, 1.0),
        max_reaction=Extremum(2.5 * scale, 1.0),
    )
    trace = DynamicParameterTrace(
        flexural_rigidity_nm2=8.085e6,
        foundation_modulus_n_per_m2=10_000_000.0,
        foundation_damping_n_s_per_m2=2_000.0,
        damping_ratio=0.0408,
        mass_kg_per_m=60.0,
        beta_per_m=0.7459,
        characteristic_length_m=1.3407,
        spatial_step_m=0.05,
        critical_speed_m_per_s=94.0,
        critical_speed_ratio=0.25,
        dynamic_amplification=1.02,
    )
    return DynamicResult(spatial=spatial, probes=probe_series, summary=summary, parameter_trace=trace)


def _build_dynamic_transition_result() -> DynamicTransitionResult:
    representative = _build_dynamic_result(scale=1.0, probes=1)
    xi = list(representative.spatial.xi_m)
    k_profile = [40_000_000.0 if x < 0.0 else 80_000_000.0 for x in xi]
    return DynamicTransitionResult(
        solver_fidelity="screening",
        profile_type=DynamicTransitionProfileType.STEP.value,
        run_mode=DynamicTransitionRunMode.SINGLE.value,
        k1_n_per_m2=40_000_000.0,
        k2_n_per_m2=80_000_000.0,
        transition_length_m=None,
        segment_length_m=None,
        x_ref_m=0.0,
        x_ref_start_m=None,
        x_ref_end_m=None,
        x_ref_step_m=None,
        metrics=DynamicTransitionMetrics(
            max_deflection_m=0.001,
            max_moment_nm=12.0,
            max_shear_n=6.0,
            max_reaction_n_per_m=2.5,
            governing_x_ref_m=0.0,
            risk_index=1.1,
            critical_speed_ratio=0.9,
            dynamic_amplification=1.05,
            transition_stiffness_ratio=2.0,
        ),
        series=DynamicTransitionSeries(
            x_m=xi,
            k_profile_n_per_m2=k_profile,
            deflection_m=list(representative.spatial.deflection_m),
            moment_nm=list(representative.spatial.moment_nm),
            shear_n=list(representative.spatial.shear_n),
            reaction_n_per_m=list(representative.spatial.reaction_n_per_m),
        ),
        representative=representative,
        envelope_count=1,
    )


def _build_plot_envelope_result() -> EnvelopeResult:
    x_values = [float(index) for index in range(6)]
    max_deflection = [0.010 if index % 2 == 0 else 0.005 for index in range(len(x_values))]
    min_deflection = [-value for value in max_deflection]
    max_moment = [23_000.0 if index % 2 == 0 else 17_000.0 for index in range(len(x_values))]
    min_moment = [-value for value in max_moment]
    max_shear = [63_000.0 if index % 2 == 0 else 56_000.0 for index in range(len(x_values))]
    min_shear = [-value for value in max_shear]
    max_reaction = [45_000.0 if index % 2 == 0 else 35_000.0 for index in range(len(x_values))]
    min_reaction = [-value for value in max_reaction]
    max_sleeper = [120_000.0 if index % 2 == 0 else 80_000.0 for index in range(len(x_values))]
    min_sleeper = [-value for value in max_sleeper]
    max_ballast = [300_000.0 if index % 2 == 0 else 200_000.0 for index in range(len(x_values))]
    min_ballast = [-value for value in max_ballast]
    max_formation = [200_000.0 if index % 2 == 0 else 160_000.0 for index in range(len(x_values))]
    min_formation = [-value for value in max_formation]

    summary = EnvelopeSummary(
        beta_per_m=0.7,
        zero_moment_distance_m=1.0,
        contraflexure_distance_m=2.0,
        max_deflection=StaticExtremum(value=max(max_deflection), position_m=0.0),
        max_moment=StaticExtremum(value=max(max_moment), position_m=0.0),
        max_shear=StaticExtremum(value=max(max_shear), position_m=0.0),
        max_reaction=StaticExtremum(value=max(max_reaction), position_m=0.0),
        max_sleeper_load=StaticExtremum(value=max(max_sleeper), position_m=0.0),
        max_ballast_pressure=StaticExtremum(value=max(max_ballast), position_m=0.0),
        max_rail_base_stress_pa=10_000_000.0,
        max_formation_stress_by_depth_pa={0.3: max(max_formation)},
    )
    return EnvelopeResult(
        x_m=x_values,
        deflection_max_m=max_deflection,
        deflection_min_m=min_deflection,
        moment_max_nm=max_moment,
        moment_min_nm=min_moment,
        shear_max_n=max_shear,
        shear_min_n=min_shear,
        reaction_max_n_per_m=max_reaction,
        reaction_min_n_per_m=min_reaction,
        sleeper_positions_m=x_values,
        sleeper_loads_max_n=max_sleeper,
        sleeper_loads_min_n=min_sleeper,
        ballast_pressure_max_pa=max_ballast,
        ballast_pressure_min_pa=min_ballast,
        formation_stress_max_pa_by_depth={0.3: max_formation},
        formation_stress_min_pa_by_depth={0.3: min_formation},
        summary=summary,
        left_deflection_max_m=list(max_deflection),
        left_deflection_min_m=list(min_deflection),
        right_deflection_max_m=list(max_deflection),
        right_deflection_min_m=list(min_deflection),
        left_moment_max_nm=list(max_moment),
        left_moment_min_nm=list(min_moment),
        right_moment_max_nm=list(max_moment),
        right_moment_min_nm=list(min_moment),
    )


def _build_plot_envelope_config(step_m: float) -> EnvelopeConfig:
    analysis_config = AnalysisConfig(
        loads=[PointLoad(position_m=0.0, load_newtons=80_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.85e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=6,
        x_domain_m=(0.0, 5.0),
    )
    return EnvelopeConfig(
        analysis_config=analysis_config,
        x_ref_start_m=-1.0,
        x_ref_end_m=1.0,
        x_ref_step_m=step_m,
        x_domain_m=(0.0, 5.0),
        bearing_width_m=0.25,
        bearing_length_m=2.6,
        depth_m=[0.3],
        rail_count=2,
        mode=AnalysisMode.CLOSED_FORM,
    )


def test_analysis_worker_emits_safe_error_and_invokes_gui_handler(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    warning_calls: dict[str, str] = {}

    def fake_warning(parent, title, message) -> None:
        warning_calls["title"] = title
        warning_calls["message"] = message

    monkeypatch.setattr(main.QMessageBox, "warning", fake_warning)
    monkeypatch.setattr(main, "run_analysis", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    window = main.MainWindow()
    inputs = main.AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
    )
    config = AnalysisConfig(
        loads=inputs.loads,
        foundation_modulus_n_per_m2=inputs.foundation_modulus_n_per_m2,
        elastic_modulus_pa=inputs.elastic_modulus_pa,
        moment_inertia_m4=inputs.moment_inertia_m4,
        section_modulus_m3=inputs.section_modulus_m3,
        sleeper_spacing_m=inputs.sleeper_spacing_m,
        sleeper_length_m=inputs.sleeper_length_m,
        sleeper_width_m=inputs.sleeper_width_m,
        sample_count=inputs.sample_count,
    )
    worker = main.AnalysisWorker(config, inputs, AnalysisMode.CLOSED_FORM)
    worker.failed.connect(window._handle_analysis_error)

    worker.run()

    assert warning_calls["message"] == main.SAFE_ANALYSIS_ERROR_MESSAGE
    assert window.statusBar().currentMessage() == "Analysis failed"
    window.close()


def test_dynamic_result_type_guard_warns(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    warning_calls: dict[str, str] = {}

    def fake_warning(parent, title, message) -> None:
        warning_calls["title"] = title
        warning_calls["message"] = message

    monkeypatch.setattr(main.QMessageBox, "warning", fake_warning)
    window = main.MainWindow()
    window.run_button.setEnabled(False)

    window._handle_dynamic_result("unexpected")

    assert warning_calls["title"] == "Analysis error"
    assert warning_calls["message"] == "Dynamic analysis returned an unexpected result type."
    assert window.statusBar().currentMessage() == "Analysis failed"
    assert window.run_button.isEnabled()
    window.close()


def test_sensitivity_button_appears_and_tracks_project_selection(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()

    assert window.sensitivity_button.text() == "Sensitivity / Design"
    project_item = window.project_tree.topLevelItem(0)
    assert project_item is not None
    window.project_tree.setCurrentItem(project_item)
    assert window.sensitivity_button.isEnabled() == (project_item.childCount() > 0)

    if project_item.childCount() > 0:
        config_item = project_item.child(0)
        window.project_tree.setCurrentItem(config_item)
        assert window.apply_config_button.isEnabled()
        assert window.sensitivity_button.isEnabled()

    window.close()


def test_project_selection_does_not_mutate_current_load_defaults(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    project = window.session.scalar(select(Project).order_by(Project.id))
    assert project is not None
    project.vehicle_type = "freight_heavy_haul"
    window.session.commit()
    window.load_magnitude_input.set_value(77.0)

    window._refresh_project_tree()
    window.project_tree.setCurrentItem(window.project_tree.topLevelItem(0))

    assert window.load_magnitude_input.value() == pytest.approx(77.0)
    window.close()


def test_sensitivity_static_context_uses_selected_track_config_and_load(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    config = window.session.scalar(select(TrackConfig).order_by(TrackConfig.id))
    assert config is not None

    analysis_config = window._build_sensitivity_static_context(config, 123_000.0)

    assert analysis_config.loads[0].load_newtons == pytest.approx(123_000.0)
    assert analysis_config.foundation_modulus_n_per_m2 == pytest.approx(
        config.support_profile.foundation_modulus_n_per_m2
    )
    assert analysis_config.sleeper_spacing_m == pytest.approx(config.sleeper_spacing_m)
    assert analysis_config.railpad_stiffness_n_per_m == pytest.approx(
        config.pad.stiffness_newtons_per_meter
    )
    window.close()


def test_sensitivity_button_uses_design_alternative_track_config(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    config = window.session.scalar(select(TrackConfig).order_by(TrackConfig.id))
    assert config is not None
    alternative = crud.create_design_alternative(
        window.session,
        project_id=config.project_id,
        track_config_id=config.id,
        name="Alternative for sensitivity",
        source_type="manual",
        analysis_type="static",
        changed_parameters={},
        input_snapshot={},
        metrics={
            "max_deflection_m": 0.001,
            "ballast_pressure_pa": 180_000.0,
            "formation_pressure_pa": 95_000.0,
            "subgrade_pressure_pa": 60_000.0,
            "deep_subgrade_pressure_pa": 35_000.0,
        },
        status="ok",
    )
    window._refresh_project_tree()

    def find_alt_item(parent):
        for index in range(parent.childCount()):
            child = parent.child(index)
            payload = child.data(0, Qt.UserRole) or {}
            if payload.get("type") == "alternative" and payload.get("id") == alternative.id:
                return child
            nested = find_alt_item(child)
            if nested is not None:
                return nested
        return None

    alt_item = None
    for top_index in range(window.project_tree.topLevelItemCount()):
        alt_item = find_alt_item(window.project_tree.topLevelItem(top_index))
        if alt_item is not None:
            break
    assert alt_item is not None
    window.project_tree.setCurrentItem(alt_item)

    assert window.sensitivity_button.isEnabled()
    project, selected_config = window._selected_project_or_config()
    assert project.id == config.project_id
    assert selected_config.id == config.id
    window.close()


def test_sensitivity_button_falls_back_to_active_config_when_tree_selection_is_empty(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    config = window.session.scalar(select(TrackConfig).order_by(TrackConfig.id))
    assert config is not None

    window._apply_track_config(config)
    window.project_tree.clearSelection()
    window._handle_project_tree_selection()

    assert window.sensitivity_button.isEnabled()
    project, selected_config = window._selected_project_or_config()
    assert project.id == config.project_id
    assert selected_config.id == config.id
    window.close()


def test_sensitivity_dialog_filters_and_colors_result_rows(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    config = window.session.scalar(select(TrackConfig).order_by(TrackConfig.id))
    assert config is not None
    baseline = SensitivityScenarioResult(
        scenario_name="Baseline",
        changed_parameter="baseline",
        parameter_value=None,
        factor=1.0,
        metrics=SensitivityMetrics(
            max_deflection_m=0.001,
            ballast_pressure_pa=200_000.0,
            formation_pressure_pa=120_000.0,
            subgrade_pressure_pa=70_000.0,
            deep_subgrade_pressure_pa=40_000.0,
        ),
        score=1.0,
        percent_improvement=0.0,
        warning="",
        rank=2,
    )
    best = SensitivityScenarioResult(
        scenario_name="support stiffness +20%",
        changed_parameter="support_stiffness",
        parameter_value=48_000_000.0,
        factor=1.2,
        metrics=SensitivityMetrics(
            max_deflection_m=0.0008,
            ballast_pressure_pa=180_000.0,
            formation_pressure_pa=100_000.0,
            subgrade_pressure_pa=60_000.0,
            deep_subgrade_pressure_pa=35_000.0,
        ),
        score=0.8,
        percent_improvement=20.0,
        warning="",
        rank=1,
    )
    warning = SensitivityScenarioResult(
        scenario_name="pad stiffness -20%",
        changed_parameter="pad_stiffness",
        parameter_value=10.0,
        factor=0.8,
        metrics=SensitivityMetrics(),
        score=None,
        percent_improvement=None,
        warning="pad unavailable",
        rank=None,
    )
    result = SensitivityRunResult(
        mode=SensitivityRunMode.STATIC,
        baseline=baseline,
        scenarios=[baseline, best, warning],
        recommendation=SensitivityRecommendation(
            best_option="support stiffness +20%",
            most_sensitive_parameter="support stiffness",
            worst_option="pad stiffness -20%",
            next_design_adjustment="Increase support stiffness.",
            key_warning="Screening only.",
        ),
    )
    dialog = main.SensitivityDialog(
        session=window.session,
        project=config.project,
        track_config=config,
        current_load_n=100_000.0,
        build_static_context=window._build_sensitivity_static_context,
        build_transition_context=window._build_sensitivity_transition_context,
        parent=window,
    )

    dialog._populate_results(result)

    assert dialog.table.item(0, 0).background().color().name() == "#f1f3f5"
    assert dialog.table.item(1, 0).background().color().name() == "#d8f3dc"
    assert dialog.table.item(1, 7).text() == "180"
    assert dialog.table.item(1, 8).text() == "100"
    assert dialog.table.item(1, 9).text() == "60"
    assert dialog.table.item(1, 10).text() == "35"
    assert dialog.table.item(2, 0).background().color().name() == "#fff3bf"
    dialog.result_filter_combo.setCurrentIndex(dialog.result_filter_combo.findData("warnings"))
    assert dialog.table.isRowHidden(0)
    assert dialog.table.isRowHidden(1)
    assert not dialog.table.isRowHidden(2)
    dialog.close()
    window.close()


def test_as5100_load_mode_is_mutually_exclusive_and_collects_standard_loads(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()

    window.train_loads_checkbox.setChecked(True)
    window.as5100_loads_checkbox.setChecked(True)

    assert window.as5100_loads_checkbox.isChecked()
    assert not window.train_loads_checkbox.isChecked()
    assert not window.several_loads_checkbox.isChecked()
    assert not window.as5100_loads_group.isHidden()
    assert not window.load_case_combo.isEnabled()

    window.as5100_model_combo.setCurrentIndex(window.as5100_model_combo.findData(main.AS5100_MODEL_300LA))
    window.as5100_group_count_input.setValue(2)
    window.as5100_group_spacing_input.set_value(12.0)
    window.as5100_reference_input.set_value(0.0)
    loads = window._collect_analysis_loads()

    assert [load.position_m for load in loads] == pytest.approx(
        [0.0, 2.0, 3.7, 4.8, 6.5, 14.0, 15.7, 16.8, 18.5]
    )
    assert max(load.load_newtons for load in loads) == pytest.approx(180_000.0)
    metadata = window._current_load_source_metadata()
    assert metadata["source_type"] == "as5100_fixed_rail"
    assert metadata["standard"] == "AS5100.2:2017"
    assert metadata["model"] == "300LA"
    assert metadata["max_axle_load_n"] == pytest.approx(360_000.0)
    assert metadata["max_wheel_load_n_per_rail"] == pytest.approx(180_000.0)

    config = AnalysisConfig(
        loads=loads,
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
    )
    window._last_analysis_config = config
    window._last_analysis_load_source = metadata
    markers = window._build_load_markers_from_analysis_config(config)
    assert "P_wheel = 180.0 kN/rail" in markers[0].label
    assert "axle = 360.0 kN" in markers[0].label
    window.close()


def test_train_builder_treats_input_as_axle_load_and_marks_wheel_solver_load(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()

    window.train_loads_checkbox.setChecked(True)
    window.train_axle_load_input.set_value(120.0)
    window.train_bogie_count_input.setValue(1)
    window.train_axles_per_bogie_input.setValue(2)
    window.train_axle_spacing_input.set_value(1600.0)
    loads = window._collect_analysis_loads()
    metadata = window._current_load_source_metadata()

    assert [load.load_newtons for load in loads] == pytest.approx([60_000.0, 60_000.0])
    assert metadata["load_basis"] == "axle_load_split_to_two_rails"
    assert metadata["max_axle_load_n"] == pytest.approx(120_000.0)
    assert metadata["max_wheel_load_n_per_rail"] == pytest.approx(60_000.0)

    config = AnalysisConfig(
        loads=loads,
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
    )
    window._last_analysis_config = config
    window._last_analysis_load_source = metadata
    markers = window._build_load_markers_from_analysis_config(config)

    assert markers
    assert all("P_wheel = 60.0 kN/rail" in marker.label for marker in markers)
    assert all("axle = 120.0 kN" in marker.label for marker in markers)
    window.close()


def test_as5100_load_source_is_captured_with_completed_static_result(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    rail = window.rail_combo.currentData()
    sleeper = window.sleeper_combo.currentData()
    support = window.support_combo.currentData()
    assert all([rail, sleeper, support])

    window.as5100_loads_checkbox.setChecked(True)
    config, inputs, mode = window._build_analysis_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )
    metadata = window._capture_load_source_metadata()
    window._pending_analysis_load_source = metadata
    window.worker = main.AnalysisWorker(config, inputs, mode)
    window._handle_analysis_result(compute_track_response(inputs))

    window.as5100_loads_checkbox.setChecked(False)
    window.load_magnitude_input.set_value(123.0)

    captured_metadata = window._last_static_load_source_metadata()
    assert captured_metadata is not None
    assert captured_metadata["source_type"] == "as5100_fixed_rail"
    assert captured_metadata["model"] == "300LA"

    window._write_analysis_snapshot(inputs, config, load_source=captured_metadata)
    snapshot_path = tmp_path / ".boef" / "analysis_inputs_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["load_source"]["source_type"] == "as5100_fixed_rail"
    assert snapshot["load_source"]["model"] == "300LA"
    window.close()


def test_as5100_governing_envelope_summary_and_annotations_show_provenance(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window._last_envelope_result = _build_plot_envelope_result()
    window._last_envelope_config = _build_plot_envelope_config(step_m=0.2)
    window._last_envelope_load_source = {
        "source_type": "as5100_fixed_rail",
        "arrangement": "governing_envelope_sweep",
        "model": "300LA",
        "group_count": 3,
        "group_spacing_m": 12.0,
        "selected_group_count": 3,
        "selected_group_spacing_m": 16.0,
        "sweep_candidate_count": 9,
    }

    window.summary_panel.update_envelope_summary(
        window._last_envelope_result,
        load_source=window._last_envelope_load_source,
    )
    annotations = window._build_envelope_chart_annotations(
        window._last_envelope_result,
        chart_title="Stress envelope",
    )

    assert "300LA governing sweep" in window.summary_panel._fields["as5100_summary"].text()
    summary_text = window.summary_panel._fields["as5100_summary"].text()
    assert "AS5100.2:2017; x0=0.000 m" in summary_text
    assert "|M|max" in summary_text
    assert "|w|max" in summary_text
    metadata_text = annotations[0][0]
    assert "Load source: AS5100 300LA governing sweep" in metadata_text
    assert "Standard: AS5100.2:2017; x0=0.000 m" in metadata_text
    assert "Governing arrangement: 3 group(s) @ 12.00 m" in metadata_text
    assert "Selected upper bound: 3 group(s) @ 16.00 m" in metadata_text
    window.close()


def test_as5100_fixed_envelope_summary_and_annotations_show_provenance(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window._last_envelope_result = _build_plot_envelope_result()
    window._last_envelope_config = _build_plot_envelope_config(step_m=0.2)
    window._last_envelope_load_source = {
        "source_type": "as5100_fixed_rail",
        "standard": "AS5100.2:2017",
        "arrangement": "fixed_user_selected",
        "model": "150LA",
        "group_count": 2,
        "group_spacing_m": 14.0,
        "reference_position_m": 1.5,
        "axle_count": 9,
    }

    window.summary_panel.update_envelope_summary(
        window._last_envelope_result,
        load_source=window._last_envelope_load_source,
    )
    annotations = window._build_envelope_chart_annotations(
        window._last_envelope_result,
        chart_title="Deflection envelope",
    )

    summary_text = window.summary_panel._fields["as5100_summary"].text()
    assert "150LA fixed selected arrangement" in summary_text
    assert "AS5100.2:2017; x0=1.500 m" in summary_text
    assert "|M|max" in summary_text
    metadata_text = annotations[0][0]
    assert "Load source: AS5100 150LA fixed arrangement" in metadata_text
    assert "Standard: AS5100.2:2017; x0=1.500 m" in metadata_text
    assert "AS5100 axles: 9, 2 group(s) @ 14.00 m" in metadata_text
    window.close()


def test_as5100_envelope_summary_payload_carries_export_ready_summary(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window._last_envelope_result = _build_plot_envelope_result()
    load_source = {
        "source_type": "as5100_fixed_rail",
        "standard": "AS5100.2:2017",
        "arrangement": "governing_envelope_sweep",
        "model": "300LA",
        "group_count": 3,
        "group_spacing_m": 12.0,
        "selected_group_count": 3,
        "selected_group_spacing_m": 16.0,
        "reference_position_m": 0.0,
        "axle_count": 13,
        "max_axle_load_n": 360_000.0,
        "max_wheel_load_n_per_rail": 180_000.0,
        "sweep_candidate_count": 9,
    }

    payload = window._as5100_envelope_summary_payload(window._last_envelope_result, load_source)

    assert payload is not None
    assert payload["arrangement"] == "governing_envelope_sweep"
    assert payload["sweep_candidate_count"] == 9
    assert "|M|max" in payload["text"]
    window.close()


def test_as5100_150la_group_count_can_be_edited(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.as5100_model_combo.setCurrentIndex(window.as5100_model_combo.findData(main.AS5100_MODEL_150LA))
    window.as5100_group_count_input.setValue(4)

    assert window.as5100_group_count_input.value() == 4
    assert "4 group(s)" in window.as5100_summary_label.text()
    window.close()


def test_chart_label_controls_filter_input_and_output_overlay_categories(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    result = _build_plot_envelope_result()

    annotations = window._build_envelope_chart_annotations(result, chart_title="Deflection envelope")
    assert len(annotations) == 2
    assert "Analysis: Static envelope" in annotations[0][0]
    assert "|w|max" in annotations[1][0]

    window.chart_output_labels_checkbox.setChecked(False)
    annotations = window._build_envelope_chart_annotations(result, chart_title="Deflection envelope")
    assert len(annotations) == 1
    assert "Analysis: Static envelope" in annotations[0][0]
    assert "|w|max" not in annotations[0][0]

    window.chart_output_labels_checkbox.setChecked(True)
    window.chart_input_labels_checkbox.setChecked(False)
    annotations = window._build_envelope_chart_annotations(result, chart_title="Deflection envelope")
    assert len(annotations) == 1
    assert "|w|max" in annotations[0][0]
    assert "Analysis: Static envelope" not in annotations[0][0]

    window.chart_output_labels_checkbox.setChecked(False)
    assert window._build_envelope_chart_annotations(result, chart_title="Deflection envelope") == []
    window.close()


def test_chart_input_label_control_hides_load_marker_annotations(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    markers = [main.LoadMarker(x_m=0.0, load_kn=100.0, label="P = 100.0 kN")]

    assert window._chart_input_load_markers(markers) == markers

    window.chart_input_labels_checkbox.setChecked(False)
    assert window._chart_input_load_markers(markers) is None
    window.close()


def test_sensitivity_dialog_uses_current_as5100_load_selection_and_labels_scale(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    config = window.session.scalar(select(TrackConfig).order_by(TrackConfig.id))
    assert config is not None
    window.as5100_loads_checkbox.setChecked(True)
    current_loads = window._collect_analysis_loads()
    metadata = window._current_load_source_metadata()
    dialog = main.SensitivityDialog(
        session=window.session,
        project=config.project,
        track_config=config,
        current_load_n=100_000.0,
        build_static_context=window._build_sensitivity_static_context,
        build_transition_context=window._build_sensitivity_transition_context,
        current_loads=current_loads,
        current_load_source_metadata=metadata,
        parent=window,
    )

    base_config = dialog.build_static_context(config, dialog._baseline_load_n(config))
    updated, transition_context = dialog._apply_current_loads_to_context(base_config, None)

    assert transition_context is None
    assert len(updated.loads) == len(current_loads)
    assert [load.load_newtons for load in updated.loads] == pytest.approx(
        [load.load_newtons for load in current_loads]
    )
    assert dialog.variable_checks[SensitivityVariable.AS5100_POSITION].isEnabled()
    assert not dialog.variable_checks[SensitivityVariable.AS5100_POSITION].isChecked()
    assert dialog._scenario_change_label("wheel_load", 1.2) == "AS5100 load scale +20%"
    assert dialog._scenario_change_label("as5100_position", 1.0, 0.5) == "AS5100 train shifted +0.5 m"
    dialog.use_current_load_selection_checkbox.setChecked(False)
    assert not dialog.variable_checks[SensitivityVariable.AS5100_POSITION].isEnabled()
    assert "AS5100 300LA" in dialog.summary_label.text()
    dialog.close()
    window.close()


def test_sensitivity_dialog_rescores_when_criteria_change(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    config = window.session.scalar(select(TrackConfig).order_by(TrackConfig.id))
    assert config is not None
    baseline = SensitivityScenarioResult(
        scenario_name="Baseline",
        changed_parameter="baseline",
        parameter_value=None,
        factor=1.0,
        metrics=SensitivityMetrics(max_deflection_m=0.001),
        score=1.0,
        percent_improvement=0.0,
        warning="",
        rank=1,
    )
    result = SensitivityRunResult(
        mode=SensitivityRunMode.STATIC,
        baseline=baseline,
        scenarios=[baseline],
        recommendation=SensitivityRecommendation(
            best_option="No valid non-baseline option.",
            most_sensitive_parameter="No valid sensitivity ranking available.",
            worst_option="No valid non-baseline option.",
            next_design_adjustment="Review baseline inputs and rerun the screening study.",
            key_warning="No valid sensitivity scenarios completed.",
        ),
    )
    dialog = main.SensitivityDialog(
        session=window.session,
        project=config.project,
        track_config=config,
        current_load_n=100_000.0,
        build_static_context=window._build_sensitivity_static_context,
        build_transition_context=window._build_sensitivity_transition_context,
        parent=window,
    )

    dialog._populate_results(result)
    dialog.criteria_inputs["max_deflection_mm"].setValue(0.5)

    assert dialog.result is not None
    assert dialog.result.baseline.decision.status == "fail"
    assert dialog.table.item(0, 18).text() == "fail"
    assert "re-scored without rerunning" in dialog.status_label.text()
    assert dialog.export_button.isEnabled()
    dialog.close()
    window.close()


def test_sensitivity_dialog_marks_results_stale_when_inputs_change(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    config = window.session.scalar(select(TrackConfig).order_by(TrackConfig.id))
    assert config is not None
    baseline = SensitivityScenarioResult(
        scenario_name="Baseline",
        changed_parameter="baseline",
        parameter_value=None,
        factor=1.0,
        metrics=SensitivityMetrics(max_deflection_m=0.001),
        score=1.0,
        percent_improvement=0.0,
        warning="",
        rank=1,
    )
    result = SensitivityRunResult(
        mode=SensitivityRunMode.STATIC,
        baseline=baseline,
        scenarios=[baseline],
        recommendation=SensitivityRecommendation(
            best_option="No valid non-baseline option.",
            most_sensitive_parameter="No valid sensitivity ranking available.",
            worst_option="No valid non-baseline option.",
            next_design_adjustment="Review baseline inputs and rerun the screening study.",
            key_warning="No valid sensitivity scenarios completed.",
        ),
    )
    dialog = main.SensitivityDialog(
        session=window.session,
        project=config.project,
        track_config=config,
        current_load_n=100_000.0,
        build_static_context=window._build_sensitivity_static_context,
        build_transition_context=window._build_sensitivity_transition_context,
        parent=window,
    )

    dialog._populate_results(result)
    dialog.preset_combo.setCurrentIndex(dialog.preset_combo.findData("rail"))

    assert dialog.result is result
    assert dialog._result_stale
    assert "Run sensitivity again" in dialog.status_label.text()
    assert not dialog.export_button.isEnabled()
    assert not dialog.save_alternative_button.isEnabled()
    dialog.close()
    window.close()


def test_sensitivity_dialog_opens_large_selected_plot(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    captured: dict[str, object] = {}

    def fake_exec(self):
        captured["title"] = self.windowTitle()
        captured["canvas_count"] = len(self.findChildren(sensitivity_dialog.FigureCanvas))
        return 0

    monkeypatch.setattr(sensitivity_dialog.QDialog, "exec", fake_exec)
    window = main.MainWindow()
    config = window.session.scalar(select(TrackConfig).order_by(TrackConfig.id))
    assert config is not None
    baseline = SensitivityScenarioResult(
        scenario_name="Baseline",
        changed_parameter="baseline",
        parameter_value=None,
        factor=1.0,
        metrics=SensitivityMetrics(max_deflection_m=0.001),
        score=1.0,
        percent_improvement=0.0,
        warning="",
        rank=2,
    )
    best = SensitivityScenarioResult(
        scenario_name="support stiffness +20%",
        changed_parameter="support_stiffness",
        parameter_value=48_000_000.0,
        factor=1.2,
        metrics=SensitivityMetrics(max_deflection_m=0.0008),
        score=0.8,
        percent_improvement=20.0,
        warning="",
        rank=1,
    )
    result = SensitivityRunResult(
        mode=SensitivityRunMode.STATIC,
        baseline=baseline,
        scenarios=[baseline, best],
        recommendation=SensitivityRecommendation(
            best_option="support stiffness +20%",
            most_sensitive_parameter="support stiffness",
            worst_option="support stiffness +20%",
            next_design_adjustment="Increase support stiffness.",
            key_warning="Screening only.",
        ),
    )
    dialog = main.SensitivityDialog(
        session=window.session,
        project=config.project,
        track_config=config,
        current_load_n=100_000.0,
        build_static_context=window._build_sensitivity_static_context,
        build_transition_context=window._build_sensitivity_transition_context,
        parent=window,
    )

    dialog._populate_results(result)
    dialog.plots.setCurrentIndex(1)
    dialog._open_large_plot()

    assert captured == {"title": "Tornado", "canvas_count": 1}
    dialog.close()
    window.close()


def test_design_alternatives_appear_under_project_and_current_result_can_save(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(main.QMessageBox, "information", lambda *_args, **_kwargs: None)
    window = main.MainWindow()
    config = window.session.scalar(select(TrackConfig).order_by(TrackConfig.id))
    assert config is not None
    inputs = main.AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=100_000.0)],
        foundation_modulus_n_per_m2=config.support_profile.foundation_modulus_n_per_m2,
        elastic_modulus_pa=config.rail.elastic_modulus_pa,
        moment_inertia_m4=config.rail.moment_inertia_m4,
        section_modulus_m3=config.rail.section_modulus_m3,
        sleeper_spacing_m=config.sleeper_spacing_m,
        sleeper_length_m=config.sleeper.length_m,
        sleeper_width_m=config.sleeper.width_m,
    )
    window._last_analysis_result = compute_track_response(inputs)
    window.project_tree.setCurrentItem(window.project_tree.topLevelItem(0).child(0))

    window._save_current_result_as_alternative()

    alternatives = window.session.scalars(select(DesignAlternative)).all()
    assert len(alternatives) == 1
    window._refresh_project_tree()
    project_item = window.project_tree.topLevelItem(0)
    labels = [project_item.child(index).text(0) for index in range(project_item.childCount())]
    assert any(label.startswith("Design Alternatives") for label in labels)
    window.close()


def test_alternatives_comparison_requires_two_alternatives(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    warning_calls: dict[str, str] = {}
    monkeypatch.setattr(
        main.QMessageBox,
        "warning",
        lambda _parent, title, message: warning_calls.update({"title": title, "message": message}),
    )
    window = main.MainWindow()
    config = window.session.scalar(select(TrackConfig).order_by(TrackConfig.id))
    assert config is not None
    crud.create_design_alternative(
        window.session,
        project_id=config.project_id,
        track_config_id=config.id,
        name="Only Alternative",
        source_type="manual",
        analysis_type="static",
        changed_parameters={},
        input_snapshot={},
        metrics={
            "max_deflection_m": 0.001,
            "ballast_pressure_pa": 180_000.0,
            "formation_pressure_pa": 95_000.0,
            "subgrade_pressure_pa": 60_000.0,
            "deep_subgrade_pressure_pa": 35_000.0,
        },
        status="draft",
    )
    window._refresh_project_tree()
    window.project_tree.setCurrentItem(window.project_tree.topLevelItem(0))

    window._open_alternatives_comparison()

    assert warning_calls["message"] == "At least two alternatives are required for comparison."
    window.close()


def test_alternatives_comparison_table_and_export_state(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    config = window.session.scalar(select(TrackConfig).order_by(TrackConfig.id))
    assert config is not None
    first = crud.create_design_alternative(
        window.session,
        project_id=config.project_id,
        track_config_id=config.id,
        name="Alt 1",
        source_type="manual",
        analysis_type="static",
        changed_parameters={"support_stiffness": 1.1},
        input_snapshot={},
        metrics={
            "max_deflection_m": 0.001,
            "ballast_pressure_pa": 180_000.0,
            "formation_pressure_pa": 95_000.0,
            "subgrade_pressure_pa": 60_000.0,
            "deep_subgrade_pressure_pa": 35_000.0,
        },
        status="ok",
        score=0.9,
    )
    second = crud.create_design_alternative(
        window.session,
        project_id=config.project_id,
        track_config_id=config.id,
        name="Alt 2",
        source_type="manual",
        analysis_type="static",
        changed_parameters={"support_stiffness": 0.9},
        input_snapshot={},
        metrics={"max_deflection_m": 0.0014},
        status="warning",
        score=1.1,
    )

    dialog = main.AlternativeComparisonDialog(
        project=config.project,
        alternatives=[first, second],
        parent=window,
    )

    assert dialog.table.rowCount() == 2
    assert dialog.table.item(0, 0).background().color().name() == "#d8f3dc"
    assert dialog.table.item(0, 5).text() == "180"
    assert dialog.table.item(0, 6).text() == "95"
    assert dialog.table.item(0, 7).text() == "60"
    assert dialog.table.item(0, 8).text() == "35"
    assert dialog.table.item(1, 0).background().color().name() == "#fff3bf"
    assert not dialog.export_button.isEnabled()
    dialog.table.selectAll()
    assert dialog.export_button.isEnabled()
    dialog.close()
    window.close()


def test_all_charts_thumbnail_capture_scales_and_restores_plot_style(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()

    x_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    series = [
        (x_values, [-8.0, -2.0, 12.0, -2.0, -8.0], "Primary"),
        (x_values, [-6.0, -1.0, 10.0, -1.0, -6.0], "Overlay"),
    ]
    window.moment_plot.update_multi_plot(
        series,
        title="Bending moment",
        xlabel="x (m)",
        ylabel="M (kN·m)",
    )

    legend_before = window.moment_plot.axes.get_legend()
    assert legend_before is not None
    title_font_before = window.moment_plot.axes.title.get_fontsize()
    legend_font_before = legend_before.get_texts()[0].get_fontsize()
    frame_alpha_before = legend_before.get_frame().get_alpha()

    # Simulate a tiny hidden-tab canvas; thumbnail rendering should still be stable.
    window.moment_plot.canvas.resize(120, 90)
    pixmap = window._capture_plot_thumbnail(window.moment_plot)

    assert pixmap is not None
    assert not pixmap.isNull()
    assert pixmap.width() >= 500
    assert pixmap.height() >= 400

    legend_after = window.moment_plot.axes.get_legend()
    assert legend_after is not None
    assert window.moment_plot.axes.title.get_fontsize() == pytest.approx(title_font_before)
    assert legend_after.get_texts()[0].get_fontsize() == pytest.approx(legend_font_before)
    assert legend_after.get_frame().get_alpha() == frame_alpha_before
    window.close()


def test_plot_panel_choose_label_offset_spreads_dense_labels(qapp: QApplication) -> None:
    panel = main.PlotPanel()
    offsets = [(12, 0), (-12, 0), (0, 12), (0, -12)]
    first = panel._choose_label_offset(
        axis=panel.axes,
        base_display_xy=(200.0, 160.0),
        offsets=offsets,
        seed=0,
        min_clearance_px=6.0,
    )
    second = panel._choose_label_offset(
        axis=panel.axes,
        base_display_xy=(200.0, 160.0),
        offsets=offsets,
        seed=1,
        min_clearance_px=6.0,
    )
    third = panel._choose_label_offset(
        axis=panel.axes,
        base_display_xy=(200.0, 160.0),
        offsets=offsets,
        seed=2,
        min_clearance_px=6.0,
    )
    assert len({first, second, third}) == 3
    panel.close()


def test_stress_chart_annotations_are_draggable_for_multi_load_case(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    loads = [
        PointLoad(position_m=-0.24, load_newtons=80_000.0),
        PointLoad(position_m=-0.12, load_newtons=80_000.0),
        PointLoad(position_m=0.0, load_newtons=80_000.0),
        PointLoad(position_m=0.12, load_newtons=80_000.0),
        PointLoad(position_m=0.24, load_newtons=80_000.0),
    ]
    window._last_analysis_config = AnalysisConfig(
        loads=loads,
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=5,
    )
    stress = StressResults(
        x_m=[-0.6, -0.3, 0.0, 0.3, 0.6],
        sigma_top_fiber_pa=[0.0, 2_000_000.0, 4_000_000.0, 2_000_000.0, 0.0],
        sigma_bottom_fiber_pa=[0.0, -2_000_000.0, -4_000_000.0, -2_000_000.0, 0.0],
        sleeper_positions_m=[-0.6, -0.3, 0.0, 0.3, 0.6],
        q_ballast_signed_pa=[0.0, 200_000.0, 300_000.0, 200_000.0, 0.0],
        q_ballast_comp_pa=[0.0, 200_000.0, 300_000.0, 200_000.0, 0.0],
        q_capping_signed_pa=[0.0, 100_000.0, 160_000.0, 100_000.0, 0.0],
        q_capping_comp_pa=[0.0, 100_000.0, 160_000.0, 100_000.0, 0.0],
        metadata=StressMetadata(
            ballast_thickness_m=0.3,
            stress_model="M/Z + 2:1 spread (load-conserving)",
            pressure_sign_convention="positive=compression",
            bending_sign_convention="positive moment -> top compression",
            bearing_geometry_provenance="sleeper_geometry",
            pressure_available=True,
        ),
    )

    window._render_stress_results(stress, title="Stress")
    assert len(window.stress_plot.figure.axes) == 2
    assert window.stress_plot.axes.get_ylabel() == "Rail stress (MPa)"
    assert window.stress_plot.figure.axes[1].get_ylabel() == "Pressure (MPa)"

    draggable_annotations = [
        artist
        for artist in window.stress_plot.axes.texts
        if isinstance(artist, Annotation)
        and artist.get_text().strip()
        and getattr(artist, "_draggable", None) is not None
    ]
    assert draggable_annotations
    window.close()


def test_main_window_exports_csv_after_analysis(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    assert not window.export_analysis_button.isEnabled()
    assert not window.export_sleeper_button.isEnabled()
    assert not window.export_dynamic_time_button.isEnabled()
    assert not window.export_dynamic_fft_button.isEnabled()
    assert not window.export_dynamic_psd_button.isEnabled()
    assert not window.export_dipped_joint_button.isEnabled()
    assert not window.export_config_button.isEnabled()


def test_advanced_analysis_toggles_nonuniform_and_discrete_inputs(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()

    window.advanced_solver_checkbox.setChecked(True)
    window._toggle_advanced_controls(True)
    window.nonuniform_profile_checkbox.setChecked(False)
    window.discrete_supports_checkbox.setChecked(False)

    rail = window.rail_combo.currentData()
    sleeper = window.sleeper_combo.currentData()
    support = window.support_combo.currentData()
    assert rail and sleeper and support

    config_off, _, _mode_off = window._build_analysis_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )

    window.nonuniform_profile_checkbox.setChecked(True)
    window.profile_type_combo.setCurrentIndex(2)
    window.profile_k1_input.set_value(55.0)
    window.profile_k2_input.set_value(85.0)
    window.profile_x_start_input.set_value(-200.0)
    window.profile_x_end_input.set_value(600.0)
    window.discrete_supports_checkbox.setChecked(True)
    window.nodes_between_sleepers_input.setValue(8)

    config_on, _, _mode_on = window._build_analysis_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )

    assert config_off.foundation_profile_k1_n_per_m2 is None
    assert config_on.foundation_profile_k1_n_per_m2 == pytest.approx(
        main.mn_per_m2_to_n_per_m2(55.0)
    )
    assert config_off.foundation_profile_k2_n_per_m2 is None
    assert config_on.foundation_profile_k2_n_per_m2 == pytest.approx(
        main.mn_per_m2_to_n_per_m2(85.0)
    )
    assert config_off.foundation_profile_x_start_m is None
    assert config_on.foundation_profile_x_start_m == pytest.approx(
        main.mm_to_m(-200.0)
    )
    assert config_off.foundation_profile_x_end_m is None
    assert config_on.foundation_profile_x_end_m == pytest.approx(
        main.mm_to_m(600.0)
    )
    assert not config_off.use_discrete_supports
    assert config_on.use_discrete_supports
    assert config_off.nodes_between_sleepers != config_on.nodes_between_sleepers
    assert config_on.nodes_between_sleepers == 8

    window.close()


def test_nonuniform_checkbox_defaults_profile_to_step(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.advanced_solver_checkbox.setChecked(True)
    window._toggle_advanced_controls(True)

    uniform_index = window.profile_type_combo.findData(FoundationProfileType.UNIFORM)
    assert uniform_index >= 0
    window.profile_type_combo.setCurrentIndex(uniform_index)
    window.nonuniform_profile_checkbox.setChecked(True)

    assert window.profile_type_combo.currentData() == FoundationProfileType.STEP
    window.close()


def test_nonuniform_uniform_profile_is_rejected(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.advanced_solver_checkbox.setChecked(True)
    window._toggle_advanced_controls(True)
    window.nonuniform_profile_checkbox.setChecked(True)

    uniform_index = window.profile_type_combo.findData(FoundationProfileType.UNIFORM)
    assert uniform_index >= 0
    window.profile_type_combo.setCurrentIndex(uniform_index)

    rail = window.rail_combo.currentData()
    sleeper = window.sleeper_combo.currentData()
    support = window.support_combo.currentData()
    assert rail and sleeper and support

    with pytest.raises(ValueError, match="Select Step or Ramp"):
        window._build_analysis_context(rail=rail, sleeper=sleeper, support=support)
    window.close()


def test_nonuniform_profile_changes_static_numerical_response(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.advanced_solver_checkbox.setChecked(True)
    window._toggle_advanced_controls(True)
    window.nonuniform_profile_checkbox.setChecked(False)

    rail = window.rail_combo.currentData()
    sleeper = window.sleeper_combo.currentData()
    support = window.support_combo.currentData()
    assert rail and sleeper and support

    config_uniform, _, mode_uniform = window._build_analysis_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )

    window.nonuniform_profile_checkbox.setChecked(True)
    step_index = window.profile_type_combo.findData(FoundationProfileType.STEP)
    assert step_index >= 0
    window.profile_type_combo.setCurrentIndex(step_index)
    window.profile_k1_input.set_value(35.0)
    window.profile_k2_input.set_value(95.0)
    window.profile_x_start_input.set_value(0.0)
    window.profile_x_end_input.set_value(2000.0)

    config_nonuniform, _, mode_nonuniform = window._build_analysis_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )

    assert mode_uniform == AnalysisMode.NUMERICAL
    assert mode_nonuniform == AnalysisMode.NUMERICAL

    uniform_result = run_analysis(config_uniform, mode=mode_uniform)
    nonuniform_result = run_analysis(config_nonuniform, mode=mode_nonuniform)

    assert any(
        not math.isclose(u, n, rel_tol=1e-4, abs_tol=1e-9)
        for u, n in zip(uniform_result.deflection_m, nonuniform_result.deflection_m)
    )
    window.close()


def test_apply_track_config_updates_inputs(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    session = window.session

    project = crud.create_project(session, name="Config Apply Project", description=None)
    rail = session.scalar(select(Rail).where(Rail.name == "UIC60"))
    sleeper = session.scalar(select(Sleeper).where(Sleeper.name == "Concrete B70"))
    pad = session.scalar(select(Pad).where(Pad.name == "Standard Pad"))
    support = session.scalar(select(SupportProfile).where(SupportProfile.name == "Ballast 80 MN/m²"))
    assert all([rail, sleeper, pad, support])

    config = crud.create_track_config(
        session,
        name="Apply Config",
        project_id=project.id,
        rail_id=rail.id,
        sleeper_id=sleeper.id,
        pad_id=pad.id,
        support_profile_id=support.id,
        sleeper_spacing_m=0.65,
        gauge_m=1.435,
    )
    load_case = crud.create_load_case(
        session,
        name="Config Load",
        load_newtons=200_000.0,
        description=None,
    )
    crud.create_result(
        session,
        project_id=project.id,
        track_config_id=config.id,
        load_case_id=load_case.id,
        max_deflection_m=0.001,
        max_moment_nm=10_000.0,
    )

    window._apply_track_config(config)

    assert window.rail_combo.currentData().id == rail.id
    assert window.sleeper_combo.currentData().id == sleeper.id
    assert window.pad_combo.currentData().id == pad.id
    assert window.support_combo.currentData().id == support.id
    assert window.sleeper_spacing_input.value() == pytest.approx(650.0)
    assert window.load_case_combo.currentData().id == load_case.id
    assert window.load_magnitude_input.value() == pytest.approx(200.0)
    assert window._active_track_gauge_m == pytest.approx(1.435)

    config_ctx, _, _mode = window._build_analysis_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )
    assert config_ctx.design_inputs is not None
    expected_rail_centres_m = 1.435 + (
        main.mm_to_m(rail.head_width_mm) if rail.head_width_mm and rail.head_width_mm > 0 else 0.0
    )
    assert config_ctx.design_inputs.rail_centres_m == pytest.approx(expected_rail_centres_m)
    window.close()


def test_analysis_context_sets_zero_ballast_depth_to_none_for_a3902(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    rail = window.rail_combo.currentData()
    sleeper = window.sleeper_combo.currentData()
    support = window.support_combo.currentData()
    assert all([rail, sleeper, support])

    window.ballast_thickness_input.set_value(0.0)
    config, _inputs, _mode = window._build_analysis_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )
    assert config.design_inputs is not None
    assert config.design_inputs.ballast_depth_m is None
    assert config.design_inputs.rail_centres_m is not None
    assert config.design_inputs.rail_centres_m > 0.0
    window.close()


def test_project_vehicle_defaults_display_without_mutating_inputs_on_selection(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    session = window.session
    project = crud.create_project(
        session,
        name="Vehicle Defaults Project",
        description=None,
        vehicle_type="heavy_metro",
        design_speed_kmh=95.0,
        design_wheel_radius_mm=445.0,
    )
    window._refresh_project_tree()

    selected_item = None
    for index in range(window.project_tree.topLevelItemCount()):
        item = window.project_tree.topLevelItem(index)
        payload = item.data(0, Qt.UserRole) or {}
        if payload.get("type") == "project" and payload.get("id") == project.id:
            selected_item = item
            break

    assert selected_item is not None
    original_load = window.load_magnitude_input.value()
    original_wheel_radius = window.wheel_radius_input.value()
    original_axle_load = window.train_axle_load_input.value()
    original_bogie_spacing = window.train_bogie_spacing_input.value()
    original_design_speed = window.design_speed_input.value()
    window.project_tree.setCurrentItem(selected_item)

    defaults = main.VEHICLE_DEFAULTS["heavy_metro"]
    assert f"Wheel load: {defaults.wheel_load_kn:.0f} kN" in window.project_detail_label.text()
    assert "Design speed: 95 km/h" in window.project_detail_label.text()
    assert window.load_magnitude_input.value() == pytest.approx(original_load)
    assert window.wheel_radius_input.value() == pytest.approx(original_wheel_radius)
    assert window.train_axle_load_input.value() == pytest.approx(original_axle_load)
    assert window.train_bogie_spacing_input.value() == pytest.approx(original_bogie_spacing)
    assert window.design_speed_input.value() == pytest.approx(original_design_speed)
    window.close()


def test_analysis_snapshot_written(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    rail = window.rail_combo.currentData()
    sleeper = window.sleeper_combo.currentData()
    support = window.support_combo.currentData()
    assert all([rail, sleeper, support])

    window.advanced_solver_checkbox.setChecked(True)
    config, inputs, _mode = window._build_analysis_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )
    window._write_analysis_snapshot(inputs, config)
    snapshot_path = tmp_path / ".boef" / "analysis_inputs_snapshot.json"
    assert snapshot_path.exists()
    window.close()
    analysis_inputs = main.AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=5,
    )
    result = compute_track_response(analysis_inputs)
    config = AnalysisConfig(
        loads=analysis_inputs.loads,
        foundation_modulus_n_per_m2=analysis_inputs.foundation_modulus_n_per_m2,
        elastic_modulus_pa=analysis_inputs.elastic_modulus_pa,
        moment_inertia_m4=analysis_inputs.moment_inertia_m4,
        section_modulus_m3=analysis_inputs.section_modulus_m3,
        sleeper_spacing_m=analysis_inputs.sleeper_spacing_m,
        sleeper_length_m=analysis_inputs.sleeper_length_m,
        sleeper_width_m=analysis_inputs.sleeper_width_m,
        sample_count=analysis_inputs.sample_count,
    )
    window.worker = main.AnalysisWorker(config, analysis_inputs, AnalysisMode.CLOSED_FORM)
    window._handle_analysis_result(result)
    assert window.export_analysis_button.isEnabled()
    assert window.export_sleeper_button.isEnabled()
    assert not window.export_dynamic_time_button.isEnabled()
    assert not window.export_dynamic_fft_button.isEnabled()
    assert not window.export_dynamic_psd_button.isEnabled()
    assert not window.export_dipped_joint_button.isEnabled()

    analysis_path = tmp_path / "analysis.csv"
    sleeper_path = tmp_path / "sleepers.csv"
    save_paths = [str(analysis_path), str(sleeper_path)]

    def fake_get_save_file_name(*_args, **_kwargs):
        return save_paths.pop(0), "CSV Files (*.csv)"

    monkeypatch.setattr(main.QFileDialog, "getSaveFileName", fake_get_save_file_name)

    window.export_analysis_button.click()
    qapp.processEvents()
    window.export_sleeper_button.click()
    qapp.processEvents()

    assert analysis_path.exists()
    assert sleeper_path.exists()
    window.close()


def test_plot_panel_includes_navigation_toolbar(qapp: QApplication) -> None:
    panel = main.PlotPanel()

    assert panel.toolbar is not None
    assert panel.layout().itemAt(0).widget() is panel.toolbar
    assert panel.layout().itemAt(1).widget() is panel.canvas


def test_plot_panel_clears_secondary_axes_after_custom_chart(qapp: QApplication) -> None:
    panel = main.PlotPanel()
    panel.update_comparison_plot(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        title="Source",
        xlabel="x (m)",
        ylabel="y (mm)",
        left_label="A",
        right_label="B",
    )
    panel.render_custom_chart(
        selections=[
            main.CustomChartSelection(series_id="chart:left", axis_target="L1", legend_label="A"),
            main.CustomChartSelection(series_id="chart:right", axis_target="R1", legend_label="B"),
        ],
        title="Source - custom",
    )

    assert len(panel.figure.axes) == 2

    panel.update_plot(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 0.0],
        title="Reset",
        xlabel="x (m)",
        ylabel="y (mm)",
    )

    assert len(panel.figure.axes) == 1


def test_plot_panel_overlay_labels_are_relocatable(qapp: QApplication) -> None:
    panel = main.PlotPanel()
    panel.update_plot(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 0.0],
        title="Relocatable",
        xlabel="x",
        ylabel="y",
        annotations=[
            (
                "Move me",
                (0.02, 0.98),
                {
                    "ha": "left",
                    "va": "top",
                    "fontsize": 8,
                    "bbox": {"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
                },
            )
        ],
    )
    labels = [
        text
        for text in panel.axes.texts
        if text.get_text() == "Move me" and getattr(text, "_boef_relocatable_text", False)
    ]
    assert labels

    panel._drag_text_artist = labels[0]
    x_px, y_px = panel.axes.transAxes.transform((0.45, 0.55))
    event = type("Event", (), {"x": x_px, "y": y_px})()
    panel._handle_text_drag_motion(event)
    assert labels[0].get_position() == pytest.approx((0.45, 0.55))
    panel._handle_text_drag_release(event)
    assert panel._drag_text_artist is None
    panel.close()
    panel.close()


def test_envelope_smoothing_flattens_alias_spikes_for_upper_lower(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    x_values = [0.0, 1.0, 2.0, 3.0, 4.0]
    upper = [10.0, 5.0, 10.0, 5.0, 10.0]
    lower = [-10.0, -5.0, -10.0, -5.0, -10.0]

    smooth_upper, smooth_lower = window._smooth_envelope_pair(
        x_values,
        upper,
        lower,
        2.0,
        chart_family="test",
    )

    assert smooth_upper == [10.0, 10.0, 10.0, 10.0, 10.0]
    assert smooth_lower == [-10.0, -10.0, -10.0, -10.0, -10.0]
    assert upper == [10.0, 5.0, 10.0, 5.0, 10.0]
    assert lower == [-10.0, -5.0, -10.0, -5.0, -10.0]
    window.close()


def test_envelope_smoothing_noop_when_step_is_fine(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    x_values = [0.0, 1.0, 2.0, 3.0]
    upper = [10.0, 5.0, 10.0, 5.0]
    lower = [-10.0, -5.0, -10.0, -5.0]

    smooth_upper, smooth_lower = window._smooth_envelope_pair(
        x_values,
        upper,
        lower,
        1.0,
        chart_family="test",
    )

    assert smooth_upper == upper
    assert smooth_lower == lower
    window.close()


def test_transition_visual_smoothing_can_smooth_fine_step_aliasing(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    x_values = [0.0, 1.0, 2.0, 3.0]
    upper = [10.0, 5.0, 10.0, 5.0]
    lower = [-10.0, -5.0, -10.0, -5.0]

    smooth_upper, smooth_lower = window._smooth_envelope_pair(
        x_values,
        upper,
        lower,
        1.0,
        chart_family="transition_test",
        visual_min_radius=1,
    )

    assert smooth_upper != upper
    assert smooth_lower != lower
    assert upper == [10.0, 5.0, 10.0, 5.0]
    assert lower == [-10.0, -5.0, -10.0, -5.0]
    window.close()


def test_render_envelope_result_uses_smoothed_plot_series_but_preserves_raw(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window._last_envelope_config = _build_plot_envelope_config(2.0)
    result = _build_plot_envelope_result()
    raw_moment = list(result.moment_max_nm)
    raw_shear = list(result.shear_max_n)

    window._render_envelope_result(result)

    moment_rendered = next(series for series in window.moment_plot.rendered_series() if series.label == "Max")
    shear_rendered = next(series for series in window.shear_plot.rendered_series() if series.label == "Max")
    moment_abs_rendered = next(series for series in window.moment_plot.rendered_series() if series.label == "|Max|")
    shear_abs_rendered = next(series for series in window.shear_plot.rendered_series() if series.label == "|Max|")
    assert len({round(value, 6) for value in moment_rendered.y}) == 1
    assert len({round(value, 6) for value in shear_rendered.y}) == 1
    assert all(value >= 0.0 for value in moment_abs_rendered.y)
    assert all(value >= 0.0 for value in shear_abs_rendered.y)
    assert result.moment_max_nm == raw_moment
    assert result.shear_max_n == raw_shear
    assert len(set(result.moment_max_nm)) > 1
    assert len(set(result.shear_max_n)) > 1
    window.close()


def test_transition_envelope_render_path_applies_same_smoothing(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window._last_envelope_config = _build_plot_envelope_config(2.0)
    result = _build_plot_envelope_result()
    transition_result = TransitionRunResult(
        mode=StaticTransitionRunMode.ENVELOPE,
        profile_type=StaticTransitionProfileType.UNIFORM,
        k1_n_per_m2=40_000_000.0,
        k2_n_per_m2=None,
        transition_length_m=None,
        segment_length_m=None,
        domain_length_m=5.0,
        metrics=TransitionMetrics(
            delta_w_s_m=0.01,
            delta_w_s_position_m=0.0,
            delta_w_1m_m=0.01,
            delta_w_1m_position_m=0.0,
            curvature_max_per_m=0.1,
            curvature_max_position_m=0.0,
            moment_max_nm=23_000.0,
            moment_max_position_m=0.0,
            energy_bending_j=1.0,
            reaction_gradient_max_n_per_m2=2.0,
            reaction_gradient_position_m=0.0,
            sleeper_load_max_n=120_000.0,
            sleeper_load_position_m=0.0,
        ),
        series=TransitionSeries(
            x_m=list(result.x_m),
            k_profile_n_per_m2=[40_000_000.0 for _ in result.x_m],
            deflection_max_m=list(result.deflection_max_m),
            deflection_min_m=list(result.deflection_min_m),
            moment_max_nm=list(result.moment_max_nm),
            moment_min_nm=list(result.moment_min_nm),
            reaction_max_n_per_m=list(result.reaction_max_n_per_m),
            reaction_min_n_per_m=list(result.reaction_min_n_per_m),
        ),
        energy_metrics=TransitionEnergyMetrics(
            energy_rail_j=1.0,
            energy_foundation_j=2.0,
            energy_total_j=3.0,
            energy_partition_eta=2.0 / 3.0,
            u_total_max_j_per_m=4.0,
            u_total_max_position_m=1.0,
            du_dx_max_j_per_m2=5.0,
            du_dx_max_position_m=2.0,
            window_length_target_m=0.6,
            window_energy_max_j=1.5,
            window_energy_max_position_m=1.0,
            window_avg_max_j_per_m=2.5,
            window_avg_max_position_m=1.0,
            window_effective_length_min_m=0.3,
            window_effective_length_max_m=0.6,
            is_envelope_upper_bound=True,
            boundary_peak_flag=True,
            boundary_gradient_peak_flag=False,
        ),
        energy_series=TransitionEnergySeries(
            u_rail_j_per_m=[1.0 for _ in result.x_m],
            u_foundation_j_per_m=[3.0 for _ in result.x_m],
            u_total_j_per_m=[4.0 for _ in result.x_m],
            du_dx_j_per_m2=[0.0 for _ in result.x_m],
            window_energy_j=[1.5 for _ in result.x_m],
            window_avg_j_per_m=[2.5 for _ in result.x_m],
            window_effective_length_m=[0.6 for _ in result.x_m],
        ),
    )

    window._last_transition_result = transition_result
    window._render_transition_result(transition_result, envelope_result=result)

    moment_rendered = next(series for series in window.moment_plot.rendered_series() if series.label == "Max")
    assert len({round(value, 6) for value in moment_rendered.y}) == 1
    deflection_abs_rendered = next(
        series for series in window.deflection_plot.rendered_series() if series.label == "|Max|"
    )
    moment_abs_rendered = next(series for series in window.moment_plot.rendered_series() if series.label == "|Max|")
    shear_abs_rendered = next(series for series in window.shear_plot.rendered_series() if series.label == "|Max|")
    assert deflection_abs_rendered.color_hint == "#1f77b4"
    assert moment_abs_rendered.color_hint == "#1f77b4"
    assert shear_abs_rendered.color_hint == "#1f77b4"
    transition_profile_series = window.transition_profile_plot.rendered_series()
    assert any(series.label == "u_total" and series.y_unit == "J/m" for series in transition_profile_series)
    assert window.transition_summary_panel._fields["energy_total"].text() == "0.003"
    assert window.transition_summary_panel._fields["energy_eta"].text() == "0.667"
    assert window.transition_summary_panel._fields["u_total_max"].text() == "4.000 @ 1.000 m"
    assert window.transition_summary_panel._fields["du_dx_max"].text() == "5.000 @ 2.000 m"
    assert "upper-bound" in window.transition_summary_panel.interpretation_label.text().lower()
    assert "boundary artifact" in window.transition_summary_panel.interpretation_label.text().lower()

    transition_panels = [
        window.deflection_plot,
        window.moment_plot,
        window.shear_plot,
        window.reaction_plot,
        window.sleeper_plot,
        window.pressure_plot,
        window.transition_profile_plot,
    ]
    for panel in transition_panels:
        panel_text = "\n".join(text.get_text() for text in panel.axes.texts)
        assert "Analysis: Static envelope" not in panel_text
        assert "Template: Custom" in panel_text
        assert "Δw(s):" in panel_text
        assert "Max |dp/dx|:" in panel_text
        assert "u_max:" in panel_text
        assert "Max |du/dx|:" in panel_text

    stress_text = "\n".join(text.get_text() for text in window.stress_plot.axes.texts)
    assert "Template: Custom" in stress_text
    assert "Δw(s):" in stress_text
    assert any("P_ref =" in text.get_text() for text in window.moment_plot.axes.texts)
    window.close()


def test_transition_k_plot_falls_back_without_blank_summary(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    result = _build_plot_envelope_result()
    transition_result = TransitionRunResult(
        mode=StaticTransitionRunMode.ENVELOPE,
        profile_type=StaticTransitionProfileType.RAMP,
        k1_n_per_m2=10.0,
        k2_n_per_m2=20.0,
        transition_length_m=2.0,
        segment_length_m=None,
        domain_length_m=5.0,
        metrics=TransitionMetrics(
            delta_w_s_m=0.01,
            delta_w_s_position_m=0.0,
            delta_w_1m_m=0.01,
            delta_w_1m_position_m=0.0,
            curvature_max_per_m=0.1,
            curvature_max_position_m=0.0,
            moment_max_nm=23_000.0,
            moment_max_position_m=0.0,
            energy_bending_j=1.0,
            reaction_gradient_max_n_per_m2=2.0,
            reaction_gradient_position_m=0.0,
            sleeper_load_max_n=120_000.0,
            sleeper_load_position_m=0.0,
        ),
        series=TransitionSeries(
            x_m=list(result.x_m),
            k_profile_n_per_m2=[10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            deflection_max_m=list(result.deflection_max_m),
            deflection_min_m=list(result.deflection_min_m),
            moment_max_nm=list(result.moment_max_nm),
            moment_min_nm=list(result.moment_min_nm),
            reaction_max_n_per_m=list(result.reaction_max_n_per_m),
            reaction_min_n_per_m=list(result.reaction_min_n_per_m),
        ),
        energy_series=TransitionEnergySeries(
            u_rail_j_per_m=[1.0 for _ in result.x_m],
            u_foundation_j_per_m=[3.0 for _ in result.x_m],
            u_total_j_per_m=[4.0 for _ in result.x_m],
            du_dx_j_per_m2=[0.0 for _ in result.x_m],
            window_energy_j=[1.5 for _ in result.x_m],
            window_avg_j_per_m=[2.5 for _ in result.x_m],
            window_effective_length_m=[0.6 for _ in result.x_m],
        ),
    )

    def fail_dual_axis(*_args, **_kwargs) -> None:
        raise RuntimeError("dual axis failed")

    monkeypatch.setattr(window.transition_profile_plot, "update_multi_plot_dual_axis", fail_dual_axis)

    window._render_transition_result(transition_result, envelope_result=result)

    assert window.transition_summary_panel._fields["mode"].text() == "Envelope"
    k_series = window.transition_profile_plot.rendered_series()[0]
    assert k_series.label == "Foundation modulus profile k(x)"
    assert 2.0 in k_series.x
    assert k_series.y[k_series.x.index(2.0)] == pytest.approx(0.02)
    window.close()


def test_transition_envelope_handler_keeps_transition_tabs_visible_with_result(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    result = _build_plot_envelope_result()
    envelope_config = _build_plot_envelope_config(2.0)
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.STATIC))
    window._last_envelope_config = envelope_config
    window._pending_transition_context = main.TransitionContext(
        run_mode=StaticTransitionRunMode.ENVELOPE,
        profile_type=StaticTransitionProfileType.UNIFORM,
        template_name="Custom",
        preset_name="Custom",
        k1_n_per_m2=envelope_config.analysis_config.foundation_modulus_n_per_m2,
        k2_n_per_m2=None,
        transition_length_m=None,
        segment_length_m=None,
        domain_m=envelope_config.x_domain_m,
        analysis_config=envelope_config.analysis_config,
        analysis_mode=envelope_config.mode,
        k_profile_n_per_m2=None,
    )
    window.transition_group.setChecked(False)

    window._handle_transition_envelope_result(result)

    assert window._last_transition_result is not None
    assert window.tab_widget.isTabVisible(window.transition_profile_tab_index)
    assert window.tab_widget.isTabVisible(window.transition_summary_tab_index)
    assert window.transition_profile_plot.axes.lines
    assert window.transition_summary_panel._fields["mode"].text() != "—"
    window.close()


def test_transition_result_disables_energy_metrics_for_non_winkler_foundation(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    analysis_inputs = main.AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=5,
    )
    result = compute_track_response(analysis_inputs)
    config = AnalysisConfig(
        loads=analysis_inputs.loads,
        foundation_modulus_n_per_m2=analysis_inputs.foundation_modulus_n_per_m2,
        elastic_modulus_pa=analysis_inputs.elastic_modulus_pa,
        moment_inertia_m4=analysis_inputs.moment_inertia_m4,
        section_modulus_m3=analysis_inputs.section_modulus_m3,
        sleeper_spacing_m=analysis_inputs.sleeper_spacing_m,
        sleeper_length_m=analysis_inputs.sleeper_length_m,
        sleeper_width_m=analysis_inputs.sleeper_width_m,
        sample_count=analysis_inputs.sample_count,
        foundation_model=FoundationModelType.SERIES,
    )
    context = main.TransitionContext(
        run_mode=StaticTransitionRunMode.SINGLE,
        profile_type=StaticTransitionProfileType.UNIFORM,
        template_name="Custom",
        preset_name="Custom",
        k1_n_per_m2=analysis_inputs.foundation_modulus_n_per_m2,
        k2_n_per_m2=None,
        transition_length_m=None,
        segment_length_m=None,
        domain_m=(result.x_m[0], result.x_m[-1]),
        analysis_config=config,
        analysis_mode=AnalysisMode.NUMERICAL,
        k_profile_n_per_m2=None,
    )

    transition_result = window._build_transition_result_from_analysis(result, context)

    assert transition_result.energy_metrics is None
    assert transition_result.energy_series is None
    assert transition_result.k_representation == "model_dependent_per_unit_length"
    assert "disabled for non-Winkler" in transition_result.foundation_reaction_law
    window.close()


def test_main_window_plots_include_deflection_moment_and_shear(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    analysis_inputs = main.AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=5,
    )
    result = compute_track_response(analysis_inputs)
    config = AnalysisConfig(
        loads=analysis_inputs.loads,
        foundation_modulus_n_per_m2=analysis_inputs.foundation_modulus_n_per_m2,
        elastic_modulus_pa=analysis_inputs.elastic_modulus_pa,
        moment_inertia_m4=analysis_inputs.moment_inertia_m4,
        section_modulus_m3=analysis_inputs.section_modulus_m3,
        sleeper_spacing_m=analysis_inputs.sleeper_spacing_m,
        sleeper_length_m=analysis_inputs.sleeper_length_m,
        sleeper_width_m=analysis_inputs.sleeper_width_m,
        sample_count=analysis_inputs.sample_count,
    )
    window.worker = main.AnalysisWorker(config, analysis_inputs, AnalysisMode.CLOSED_FORM)
    window._handle_analysis_result(result)

    plots = [
        (window.deflection_plot, "y (mm)"),
        (window.moment_plot, "M (kN·m)"),
        (window.shear_plot, "V (kN)"),
    ]
    for plot, ylabel in plots:
        assert plot.axes.get_xlabel() == "x (m)"
        assert plot.axes.get_ylabel() == ylabel
        assert len(plot.axes.lines) == 1
    deflection_texts = [text.get_text() for text in window.deflection_plot.axes.texts]
    assert any("P = 10.0 kN" in text for text in deflection_texts)
    window.close()


def test_main_window_advanced_toggle_switches_analysis_mode(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    rail = window.rail_combo.currentData()
    sleeper = window.sleeper_combo.currentData()
    support = window.support_combo.currentData()
    assert rail is not None
    assert sleeper is not None
    assert support is not None

    config, _inputs, mode = window._build_analysis_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )
    assert mode is AnalysisMode.CLOSED_FORM
    assert not config.use_two_rail

    window.advanced_solver_checkbox.setChecked(True)
    config, _inputs, mode = window._build_analysis_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )
    assert mode is AnalysisMode.NUMERICAL
    window.close()


def test_dynamic_worker_dispatches_engine(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    called: dict[str, DynamicConfig | DynamicMode] = {}

    def fake_run_dynamic_analysis(config, *, mode):
        called["config"] = config
        called["mode"] = mode
        spatial = DynamicSpatialResult(
            xi_m=[0.0],
            deflection_m=[0.0],
            moment_nm=[0.0],
            shear_n=[0.0],
            reaction_n_per_m=[0.0],
            damping_force_n_per_m=[0.0],
        )
        series = DynamicTimeSeries(
            position_m=0.0,
            time_s=[0.0],
            deflection_m=[0.0],
            moment_nm=[0.0],
            shear_n=[0.0],
            reaction_n_per_m=[0.0],
            damping_force_n_per_m=[0.0],
            fft_frequency_hz=[0.0],
            fft_amplitude=[0.0],
            psd_frequency_hz=[0.0],
            psd=[0.0],
            psd_ci_lower=[0.0],
            psd_ci_upper=[0.0],
            impedance_frequency_hz=[0.0],
            impedance_magnitude_n_per_m2=[0.0],
            impedance_phase_deg=[0.0],
        )
        summary = DynamicSummary(
            max_deflection=Extremum(0.0, 0.0),
            max_moment=Extremum(0.0, 0.0),
            max_shear=Extremum(0.0, 0.0),
            max_reaction=Extremum(0.0, 0.0),
        )
        return DynamicResult(spatial=spatial, probes=[series], summary=summary)

    monkeypatch.setattr(main, "run_dynamic_analysis", fake_run_dynamic_analysis)

    window = main.MainWindow()
    rail = window.rail_combo.currentData()
    sleeper = window.sleeper_combo.currentData()
    support = window.support_combo.currentData()
    assert rail is not None
    assert sleeper is not None
    assert support is not None

    window.analysis_type_combo.setCurrentIndex(
        window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC)
    )
    assert window.export_dynamic_time_button.isVisible()
    assert window.export_dynamic_fft_button.isVisible()
    assert window.export_dynamic_psd_button.isVisible()
    assert not window.export_dipped_joint_button.isVisible()
    assert not window.export_analysis_button.isVisible()
    assert not window.export_sleeper_button.isVisible()

    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.DIPPED_JOINT))
    assert not window.export_dynamic_time_button.isVisible()
    assert window.export_dipped_joint_button.isVisible()
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.STEADY_STATE))

    config, mode = window._build_dynamic_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )
    worker = main.DynamicAnalysisWorker(config, mode)
    worker.run()

    assert "config" in called
    assert called["mode"] == DynamicMode.STEADY_STATE
    window.close()


def test_dipped_joint_mode_builds_config_and_enables_dipped_joint_export(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(
        window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC)
    )
    window.dynamic_mode_combo.setCurrentIndex(
        window.dynamic_mode_combo.findData(DynamicMode.DIPPED_JOINT)
    )

    rail = window.rail_combo.currentData()
    sleeper = window.sleeper_combo.currentData()
    support = window.support_combo.currentData()
    assert rail is not None
    assert sleeper is not None
    assert support is not None

    config, mode = window._build_dynamic_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )
    assert isinstance(config, DippedJointConfig)
    assert mode == DynamicMode.DIPPED_JOINT

    result = DippedJointResult(
        static_load_n=100_000.0,
        p1_n=120_000.0,
        p2_n=140_000.0,
        p1_dynamic_amplification=1.2,
        p2_dynamic_amplification=1.4,
    )
    window._handle_dynamic_result(result)

    assert not window.export_dynamic_time_button.isEnabled()
    assert not window.export_dynamic_fft_button.isEnabled()
    assert not window.export_dynamic_psd_button.isEnabled()
    assert window.export_dipped_joint_button.isEnabled()
    window.close()


def test_dipped_joint_zero_speed_is_allowed(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(
        window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC)
    )
    window.dynamic_mode_combo.setCurrentIndex(
        window.dynamic_mode_combo.findData(DynamicMode.DIPPED_JOINT)
    )
    window.speed_input.set_value(0.0)

    rail = window.rail_combo.currentData()
    sleeper = window.sleeper_combo.currentData()
    support = window.support_combo.currentData()
    assert rail is not None
    assert sleeper is not None
    assert support is not None

    config, mode = window._build_dynamic_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )
    assert mode == DynamicMode.DIPPED_JOINT
    assert config.speed_m_per_s == 0.0
    window.close()


def test_dipped_joint_mode_hides_load_position_and_disables_load_case(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(
        window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC)
    )
    window.dynamic_mode_combo.setCurrentIndex(
        window.dynamic_mode_combo.findData(DynamicMode.DIPPED_JOINT)
    )
    analysis_layout = window.analysis_layout
    label = analysis_layout.labelForField(window.load_position_input)
    assert label is not None

    assert not window.load_position_input.isVisible()
    assert not label.isVisible()
    assert not window.load_case_combo.isEnabled()

    window.dynamic_mode_combo.setCurrentIndex(
        window.dynamic_mode_combo.findData(DynamicMode.STEADY_STATE)
    )
    assert window.load_position_input.isVisible()
    assert label.isVisible()
    assert window.load_case_combo.isEnabled()
    window.close()


def test_dynamic_speed_validation_blocks_zero_speed(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(
        window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC)
    )
    window.dynamic_mode_combo.setCurrentIndex(
        window.dynamic_mode_combo.findData(DynamicMode.TIME_HISTORY)
    )
    window.speed_input.set_value(0.0)

    rail = window.rail_combo.currentData()
    sleeper = window.sleeper_combo.currentData()
    support = window.support_combo.currentData()
    assert rail is not None
    assert sleeper is not None
    assert support is not None

    with pytest.raises(ValueError, match="Speed must be positive"):
        window._build_dynamic_context(rail=rail, sleeper=sleeper, support=support)
    window.close()


def test_dipped_joint_export_includes_inputs(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(
        window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC)
    )
    window.dynamic_mode_combo.setCurrentIndex(
        window.dynamic_mode_combo.findData(DynamicMode.DIPPED_JOINT)
    )
    window._last_dipped_joint_config = DippedJointConfig(
        static_wheel_load_n=100_000.0,
        total_dip_angle_rad=0.01,
        speed_m_per_s=25.0,
        hertzian_stiffness_n_per_m=1.4e9,
        track_mass_p1_kg=150.0,
        unsprung_mass_kg=350.0,
        track_mass_p2_kg=100.0,
        track_stiffness_p2_n_per_m=8.0e7,
        track_damping_p2_n_s_per_m=50_000.0,
    )
    window._last_dipped_joint_result = DippedJointResult(
        static_load_n=100_000.0,
        p1_n=120_000.0,
        p2_n=140_000.0,
        p1_dynamic_amplification=1.2,
        p2_dynamic_amplification=1.4,
    )

    export_path = tmp_path / "dipped_joint.csv"

    def fake_get_save_file_name(*_args, **_kwargs):
        return str(export_path), "CSV Files (*.csv)"

    monkeypatch.setattr(main.QFileDialog, "getSaveFileName", fake_get_save_file_name)

    window.export_dipped_joint_button.setEnabled(True)
    window.export_dipped_joint_button.click()
    qapp.processEvents()

    contents = export_path.read_text(encoding="utf-8")
    assert "Hertzian stiffness" in contents
    assert "Equivalent track damping" in contents
    assert "Peak force P₁" in contents
    window.close()


def test_dynamic_plots_use_mm_units(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    spatial = DynamicSpatialResult(
        xi_m=[0.0, 1.0],
        deflection_m=[0.0, 0.001],
        moment_nm=[0.0, 10.0],
        shear_n=[0.0, 5.0],
        reaction_n_per_m=[0.0, 2.0],
        damping_force_n_per_m=[0.0, 1.0],
    )
    series = DynamicTimeSeries(
        position_m=0.0,
        time_s=[0.0, 1.0],
        deflection_m=[0.0, 0.001],
        moment_nm=[0.0, 10.0],
        shear_n=[0.0, 5.0],
        reaction_n_per_m=[0.0, 2.0],
        damping_force_n_per_m=[0.0, 1.0],
        fft_frequency_hz=[0.0, 1.0],
        fft_amplitude=[0.0, 0.001],
        psd_frequency_hz=[0.0, 1.0],
        psd=[0.0, 0.000001],
        psd_ci_lower=[0.0, 0.0000005],
        psd_ci_upper=[0.0, 0.0000015],
        impedance_frequency_hz=[0.0, 1.0],
        impedance_magnitude_n_per_m2=[1.0, 1.0],
        impedance_phase_deg=[0.0, 0.0],
    )
    summary = DynamicSummary(
        max_deflection=Extremum(0.001, 1.0),
        max_moment=Extremum(10.0, 1.0),
        max_shear=Extremum(5.0, 1.0),
        max_reaction=Extremum(2.0, 1.0),
    )
    result = DynamicResult(spatial=spatial, probes=[series], summary=summary)
    window._handle_dynamic_result(result)

    assert window.dynamic_fft_plot.axes.get_ylabel() == "|W(f)| (mm)"
    assert window.dynamic_psd_plot.axes.get_ylabel() == "PSD (mm²/Hz)"
    window.close()


def test_right_load_inputs_sync_when_asymmetric_disabled(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()

    window.advanced_solver_checkbox.setChecked(True)
    window.two_rail_checkbox.setChecked(True)

    window.load_magnitude_input.set_value(150.0)
    window.load_position_input.set_value(250.0)

    assert window.right_load_magnitude_input.value() == 150.0
    assert window.right_load_position_input.value() == 250.0
    window.close()


def test_two_rail_axle_source_uses_per_rail_loads_and_preserves_asymmetric_right_input(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()

    window.as5100_loads_checkbox.setChecked(True)
    window.advanced_solver_checkbox.setChecked(True)
    window.two_rail_checkbox.setChecked(True)

    rail = window.rail_combo.currentData()
    sleeper = window.sleeper_combo.currentData()
    support = window.support_combo.currentData()
    assert rail is not None
    assert sleeper is not None
    assert support is not None

    config, _inputs, mode = window._build_analysis_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )

    assert mode is AnalysisMode.NUMERICAL
    assert config.use_two_rail
    assert config.right_loads is None
    assert max(load.load_newtons for load in config.loads) == pytest.approx(180_000.0)

    window.asymmetric_load_checkbox.setChecked(True)
    window.right_load_magnitude_input.set_value(45.0)
    window.right_load_position_input.set_value(125.0)

    asymmetric_config, _inputs, _mode = window._build_analysis_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )

    assert asymmetric_config.use_two_rail
    assert asymmetric_config.right_loads is not None
    assert len(asymmetric_config.right_loads) == 1
    assert asymmetric_config.right_loads[0].load_newtons == pytest.approx(45_000.0)
    assert asymmetric_config.right_loads[0].position_m == pytest.approx(0.125)
    window.close()


def test_dynamic_overlay_adds_second_line_set(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.STEADY_STATE))
    window.overlay_checkbox.setChecked(True)

    primary = _build_dynamic_result(scale=1.0)
    overlay = _build_dynamic_result(scale=1.4)
    window._handle_dynamic_result(primary)
    window._handle_dynamic_result(overlay)

    assert len(window._dynamic_overlay_results) == 1
    assert len(window.dynamic_deflection_plot.axes.lines) == 2
    assert len(window.dynamic_fft_plot.axes.lines) == 2
    window.close()


def test_static_and_dynamic_overlays_do_not_mix(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()

    def build_static_result(load_n: float):
        analysis_inputs = main.AnalysisInputs(
            loads=[PointLoad(position_m=0.0, load_newtons=load_n)],
            foundation_modulus_n_per_m2=40_000_000.0,
            elastic_modulus_pa=210_000_000_000.0,
            moment_inertia_m4=3.05e-5,
            section_modulus_m3=3.2e-5,
            sleeper_spacing_m=0.6,
            sleeper_length_m=2.6,
            sleeper_width_m=0.25,
            sample_count=5,
        )
        result = compute_track_response(analysis_inputs)
        config = AnalysisConfig(
            loads=analysis_inputs.loads,
            foundation_modulus_n_per_m2=analysis_inputs.foundation_modulus_n_per_m2,
            elastic_modulus_pa=analysis_inputs.elastic_modulus_pa,
            moment_inertia_m4=analysis_inputs.moment_inertia_m4,
            section_modulus_m3=analysis_inputs.section_modulus_m3,
            sleeper_spacing_m=analysis_inputs.sleeper_spacing_m,
            sleeper_length_m=analysis_inputs.sleeper_length_m,
            sleeper_width_m=analysis_inputs.sleeper_width_m,
            sample_count=analysis_inputs.sample_count,
        )
        return result, config, analysis_inputs

    window.overlay_checkbox.setChecked(True)
    result_a, config_a, inputs_a = build_static_result(10_000.0)
    window.worker = main.AnalysisWorker(config_a, inputs_a, AnalysisMode.CLOSED_FORM)
    window._handle_analysis_result(result_a)
    result_b, config_b, inputs_b = build_static_result(12_000.0)
    window.worker = main.AnalysisWorker(config_b, inputs_b, AnalysisMode.CLOSED_FORM)
    window._handle_analysis_result(result_b)
    assert len(window._overlay_results) == 1

    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    assert len(window._overlay_results) == 0
    assert len(window._dynamic_overlay_results) == 0

    window.overlay_checkbox.setChecked(True)
    window._handle_dynamic_result(_build_dynamic_result(scale=1.0))
    window._handle_dynamic_result(_build_dynamic_result(scale=1.3))
    assert len(window._dynamic_overlay_results) == 1
    assert len(window._overlay_results) == 0
    window.close()


def test_static_mode_change_rerenders_without_stale_overlay_lines(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()

    def build_static_result(load_n: float):
        analysis_inputs = main.AnalysisInputs(
            loads=[PointLoad(position_m=0.0, load_newtons=load_n)],
            foundation_modulus_n_per_m2=40_000_000.0,
            elastic_modulus_pa=210_000_000_000.0,
            moment_inertia_m4=3.05e-5,
            section_modulus_m3=3.2e-5,
            sleeper_spacing_m=0.6,
            sleeper_length_m=2.6,
            sleeper_width_m=0.25,
            sample_count=5,
        )
        result = compute_track_response(analysis_inputs)
        config = AnalysisConfig(
            loads=analysis_inputs.loads,
            foundation_modulus_n_per_m2=analysis_inputs.foundation_modulus_n_per_m2,
            elastic_modulus_pa=analysis_inputs.elastic_modulus_pa,
            moment_inertia_m4=analysis_inputs.moment_inertia_m4,
            section_modulus_m3=analysis_inputs.section_modulus_m3,
            sleeper_spacing_m=analysis_inputs.sleeper_spacing_m,
            sleeper_length_m=analysis_inputs.sleeper_length_m,
            sleeper_width_m=analysis_inputs.sleeper_width_m,
            sample_count=analysis_inputs.sample_count,
        )
        return result, config, analysis_inputs

    window.overlay_checkbox.setChecked(True)
    result_a, config_a, inputs_a = build_static_result(10_000.0)
    window.worker = main.AnalysisWorker(config_a, inputs_a, AnalysisMode.CLOSED_FORM)
    window._handle_analysis_result(result_a)
    result_b, config_b, inputs_b = build_static_result(12_000.0)
    window.worker = main.AnalysisWorker(config_b, inputs_b, AnalysisMode.CLOSED_FORM)
    window._handle_analysis_result(result_b)

    assert len(window._overlay_results) == 1
    assert len(window.deflection_plot.axes.lines) == 2

    window.static_mode_combo.setCurrentIndex(
        window.static_mode_combo.findData(main.StaticMode.ENVELOPE_CLOSED_FORM)
    )
    qapp.processEvents()

    assert len(window._overlay_results) == 0
    assert len(window.deflection_plot.axes.lines) == 1
    window.close()


def test_new_non_overlay_analysis_clears_stale_graphs(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    analysis_inputs = main.AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=5,
    )
    config = AnalysisConfig(
        loads=analysis_inputs.loads,
        foundation_modulus_n_per_m2=analysis_inputs.foundation_modulus_n_per_m2,
        elastic_modulus_pa=analysis_inputs.elastic_modulus_pa,
        moment_inertia_m4=analysis_inputs.moment_inertia_m4,
        section_modulus_m3=analysis_inputs.section_modulus_m3,
        sleeper_spacing_m=analysis_inputs.sleeper_spacing_m,
        sleeper_length_m=analysis_inputs.sleeper_length_m,
        sleeper_width_m=analysis_inputs.sleeper_width_m,
        sample_count=analysis_inputs.sample_count,
    )
    window.worker = main.AnalysisWorker(config, analysis_inputs, AnalysisMode.CLOSED_FORM)
    window._handle_analysis_result(compute_track_response(analysis_inputs))
    assert len(window.deflection_plot.axes.lines) == 1

    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window._handle_dynamic_result(_build_dynamic_result(scale=1.0))

    assert len(window.deflection_plot.axes.lines) == 0
    assert len(window.dynamic_deflection_plot.axes.lines) == 1
    window.close()


def test_overlays_clear_on_analysis_type_change(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.overlay_checkbox.setChecked(True)
    window._handle_dynamic_result(_build_dynamic_result(scale=1.0))
    window._handle_dynamic_result(_build_dynamic_result(scale=1.2))
    assert len(window._dynamic_overlay_results) == 1

    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.STATIC))
    assert len(window._dynamic_overlay_results) == 0
    assert len(window._overlay_results) == 0
    window.close()


def test_reset_application_clears_runtime_results_without_deleting_database_records(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    rail_count = len(crud.list_rails(window.session))
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.STEADY_STATE))
    window.overlay_checkbox.setChecked(True)
    window._handle_dynamic_result(_build_dynamic_result(scale=1.0))
    window._handle_dynamic_result(_build_dynamic_result(scale=1.2))
    assert len(window._dynamic_overlay_results) == 1
    assert len(window.dynamic_deflection_plot.axes.lines) == 2

    window._reset_application()

    assert len(window._dynamic_overlay_results) == 0
    assert len(window._overlay_results) == 0
    assert len(window.dynamic_deflection_plot.axes.lines) == 0
    assert len(window.deflection_plot.axes.lines) == 0
    assert not window.export_analysis_button.isEnabled()
    assert not window.export_dynamic_time_button.isEnabled()
    assert len(crud.list_rails(window.session)) == rail_count
    assert window.statusBar().currentMessage() == "Application reset to starting state."
    window.close()


def test_dynamic_overlays_clear_on_dynamic_mode_change(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.STEADY_STATE))
    window.overlay_checkbox.setChecked(True)
    window._handle_dynamic_result(_build_dynamic_result(scale=1.0))
    window._handle_dynamic_result(_build_dynamic_result(scale=1.2))
    assert len(window._dynamic_overlay_results) == 1

    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.TIME_HISTORY))
    assert len(window._dynamic_overlay_results) == 0
    window.close()


def test_dipped_joint_mode_clears_and_disables_dynamic_overlay(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.STEADY_STATE))
    window.overlay_checkbox.setChecked(True)
    window._handle_dynamic_result(_build_dynamic_result(scale=1.0))
    window._handle_dynamic_result(_build_dynamic_result(scale=1.1))
    assert len(window._dynamic_overlay_results) == 1

    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.DIPPED_JOINT))
    assert len(window._dynamic_overlay_results) == 0
    assert not window.overlay_checkbox.isEnabled()
    assert not window.overlay_checkbox.isChecked()
    window.close()


def test_dynamic_overlay_probe_switch_updates_overlay_probe_charts(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.STEADY_STATE))
    window.probe_locations_input.setText("0.0, 1.0")
    window._refresh_probe_selection()
    window.overlay_checkbox.setChecked(True)
    window._handle_dynamic_result(_build_dynamic_result(scale=1.0, probes=2))
    window._handle_dynamic_result(_build_dynamic_result(scale=1.5, probes=2))

    window.probe_selection_combo.setCurrentIndex(1)
    window._update_dynamic_probe_plots()

    assert len(window.dynamic_time_plot.axes.lines) == 2
    first_line = window.dynamic_time_plot.axes.lines[0].get_ydata().tolist()
    second_line = window.dynamic_time_plot.axes.lines[1].get_ydata().tolist()
    assert max(first_line) == pytest.approx(2.0)
    assert max(second_line) == pytest.approx(3.0)
    window.close()


def test_dynamic_chart_help_actions_exist_and_open_chart_help(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()

    dynamic_plots = [
        window.dynamic_deflection_plot,
        window.dynamic_moment_plot,
        window.dynamic_shear_plot,
        window.dynamic_reaction_plot,
        window.dynamic_damping_plot,
        window.dynamic_time_plot,
        window.dynamic_fft_plot,
        window.dynamic_psd_plot,
        window.dynamic_impedance_plot,
    ]
    for plot in dynamic_plots:
        assert any(action.text() == "Help" for action in plot.toolbar.actions())

    help_action = next(
        action for action in window.dynamic_deflection_plot.toolbar.actions() if action.text() == "Help"
    )
    help_action.trigger()
    assert window.dynamic_help_dialog is not None
    assert window.dynamic_help_browser is not None
    assert "Dynamic Deflection" in window.dynamic_help_browser.toPlainText()
    assert "What it shows" in window.dynamic_help_browser.toPlainText()
    window.close()


def test_dynamic_mode_combo_includes_transition_mode(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()

    assert window.dynamic_mode_combo.findData(DynamicMode.TRANSITION) >= 0
    window.close()


def test_custom_chart_actions_exist_on_all_plot_panels(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()

    for entry in window._chart_registry:
        assert any(action.text() == "Custom chart" for action in entry.plot_panel.toolbar.actions())
    window.close()


def test_envelope_auto_extents_include_decay_tail(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    rail = window.rail_combo.currentData()
    support = window.support_combo.currentData()
    if rail is None or support is None:
        window.close()
        pytest.skip("Seed rail/support data not available.")

    window.envelope_reference_input.set_value(0.0)
    window.load_position_input.set_value(0.0)
    window.envelope_range_auto_checkbox.setChecked(True)
    window.envelope_domain_auto_checkbox.setChecked(True)
    window._update_envelope_domain_auto()

    beta = beam_parameter_beta(
        support.foundation_modulus_n_per_m2,
        rail.elastic_modulus_pa,
        rail.moment_inertia_m4,
    )
    movement = main.ENVELOPE_AUTO_MOVEMENT_FACTOR / beta
    tail = main.ENVELOPE_AUTO_DECAY_MARGIN_FACTOR / beta

    assert window.envelope_range_start_input.value() == pytest.approx(-movement, abs=0.001)
    assert window.envelope_range_end_input.value() == pytest.approx(movement, abs=0.001)
    assert window.envelope_domain_start_input.value() == pytest.approx(-(movement + tail), abs=0.001)
    assert window.envelope_domain_end_input.value() == pytest.approx(movement + tail, abs=0.001)
    window.close()


def test_internal_movement_range_extends_beyond_plot_domain_by_buffer() -> None:
    start, end = main.MainWindow._extend_movement_range_for_plot_domain(
        x_ref_start_m=-1.0,
        x_ref_end_m=1.0,
        x_domain_m=(-4.0, 4.0),
        load_offsets_m=[-0.5, 0.5],
        beta_per_m=2.0,
    )
    buffer = main.ENVELOPE_MOVEMENT_BUFFER_FACTOR / 2.0

    assert start == pytest.approx(-4.0 - 0.5 - buffer)
    assert end == pytest.approx(4.0 + 0.5 + buffer)


def test_transition_k_chart_series_inserts_profile_breakpoints() -> None:
    ramp_x, ramp_k = main.MainWindow._transition_k_chart_series(
        x_values=[-1.0, 0.5, 1.0],
        profile_type=StaticTransitionProfileType.RAMP,
        k1_n_per_m2=10.0,
        k2_n_per_m2=20.0,
        transition_length_m=2.0,
        segment_length_m=None,
    )
    assert 0.0 in ramp_x
    assert 2.0 in ramp_x
    assert ramp_k[ramp_x.index(0.0)] == pytest.approx(10.0)
    assert ramp_k[ramp_x.index(2.0)] == pytest.approx(20.0)

    step_x, step_k = main.MainWindow._transition_k_chart_series(
        x_values=[-1.0, 1.0],
        profile_type=StaticTransitionProfileType.STEP,
        k1_n_per_m2=10.0,
        k2_n_per_m2=20.0,
        transition_length_m=None,
        segment_length_m=None,
    )
    zero_indices = [index for index, value in enumerate(step_x) if value == 0.0]
    assert len(zero_indices) == 2
    assert [step_k[index] for index in zero_indices] == pytest.approx([10.0, 20.0])

    dynamic_x, dynamic_k = main.MainWindow._transition_k_chart_series(
        x_values=[-1.0, 0.5, 1.0],
        profile_type=DynamicTransitionProfileType.RAMP.value,
        k1_n_per_m2=10.0,
        k2_n_per_m2=20.0,
        transition_length_m=2.0,
        segment_length_m=None,
    )
    assert 2.0 in dynamic_x
    assert dynamic_k[dynamic_x.index(2.0)] == pytest.approx(20.0)


def test_static_transition_profile_plot_uses_chart_breakpoints(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    result = TransitionRunResult(
        mode=StaticTransitionRunMode.SINGLE,
        profile_type=StaticTransitionProfileType.RAMP,
        k1_n_per_m2=10.0,
        k2_n_per_m2=20.0,
        transition_length_m=2.0,
        segment_length_m=None,
        domain_length_m=2.0,
        metrics=TransitionMetrics(
            delta_w_s_m=0.0,
            delta_w_s_position_m=0.0,
            delta_w_1m_m=0.0,
            delta_w_1m_position_m=0.0,
            curvature_max_per_m=0.0,
            curvature_max_position_m=0.0,
            moment_max_nm=0.0,
            moment_max_position_m=0.0,
            energy_bending_j=0.0,
            reaction_gradient_max_n_per_m2=0.0,
            reaction_gradient_position_m=0.0,
            sleeper_load_max_n=0.0,
            sleeper_load_position_m=0.0,
        ),
        series=TransitionSeries(
            x_m=[-1.0, 0.5, 1.0],
            k_profile_n_per_m2=[10.0, 12.5, 15.0],
            deflection_m=[0.0, 0.0, 0.0],
            moment_nm=[0.0, 0.0, 0.0],
            reaction_n_per_m=[0.0, 0.0, 0.0],
        ),
    )

    window._render_transition_result(result)

    k_series = window.transition_profile_plot.rendered_series()[0]
    assert 2.0 in k_series.x
    assert k_series.y[k_series.x.index(2.0)] == pytest.approx(0.02)
    window.close()


def test_solver_domain_extends_to_cover_buffered_load_positions() -> None:
    start, end = main.MainWindow._solver_domain_for_movement_range(
        x_ref_start_m=-6.5,
        x_ref_end_m=6.5,
        plot_domain_m=(-4.0, 4.0),
        load_offsets_m=[-0.5, 0.5],
        beta_per_m=2.0,
    )
    buffer = main.ENVELOPE_MOVEMENT_BUFFER_FACTOR / 2.0

    assert start == pytest.approx(-6.5 - 0.5 - buffer)
    assert end == pytest.approx(6.5 + 0.5 + buffer)


def test_static_transition_context_coerces_string_mode_values(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.transition_group.setChecked(True)

    rail = window.rail_combo.currentData()
    sleeper = window.sleeper_combo.currentData()
    support = window.support_combo.currentData()
    assert rail is not None
    assert sleeper is not None
    assert support is not None

    monkeypatch.setattr(
        window.transition_profile_combo,
        "currentData",
        lambda *_args, **_kwargs: StaticTransitionProfileType.RAMP.value,
    )
    monkeypatch.setattr(
        window.transition_run_mode_combo,
        "currentData",
        lambda *_args, **_kwargs: StaticTransitionRunMode.ENVELOPE.value,
    )

    context, _, envelope_config = window._build_transition_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )

    assert context.profile_type == StaticTransitionProfileType.RAMP
    assert context.run_mode == StaticTransitionRunMode.ENVELOPE
    assert envelope_config is not None
    window.close()


def test_as5100_governing_sweep_is_enabled_for_transition_envelope(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.transition_group.setChecked(True)
    window.as5100_loads_checkbox.setChecked(True)
    window.as5100_arrangement_mode_combo.setCurrentIndex(
        window.as5100_arrangement_mode_combo.findData(main.AS5100ArrangementMode.GOVERNING_SWEEP)
    )
    window.as5100_group_count_input.setValue(3)
    window.as5100_group_spacing_input.set_value(16.0)
    window.transition_run_mode_combo.setCurrentIndex(
        window.transition_run_mode_combo.findData(StaticTransitionRunMode.ENVELOPE)
    )

    rail = window.rail_combo.currentData()
    sleeper = window.sleeper_combo.currentData()
    support = window.support_combo.currentData()
    assert rail is not None
    assert sleeper is not None
    assert support is not None

    context, _, envelope_config = window._build_transition_context(
        rail=rail,
        sleeper=sleeper,
        support=support,
    )

    assert context.run_mode == StaticTransitionRunMode.ENVELOPE
    assert envelope_config is not None
    assert envelope_config.as5100_sweep is not None
    assert envelope_config.as5100_sweep.selected_group_count == 3
    assert envelope_config.as5100_sweep.selected_group_spacing_m == pytest.approx(16.0)
    assert envelope_config.run_metadata is not None
    assert envelope_config.run_metadata["arrangement"] == "governing_envelope_sweep_requested"
    assert envelope_config.run_metadata["sweep_group_count_candidates"] == [1, 2, 3]
    assert envelope_config.run_metadata["sweep_group_spacing_candidates_m"] == pytest.approx([12.0, 16.0, 20.0])
    window.close()


def test_transition_envelope_handler_uses_resolved_as5100_governing_metadata(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    selected_config = main.AS5100RailLoadConfig(
        model=main.AS5100_MODEL_300LA,
        group_count=3,
        group_spacing_m=16.0,
        reference_position_m=0.0,
    )
    governing_config = main.AS5100RailLoadConfig(
        model=main.AS5100_MODEL_300LA,
        group_count=2,
        group_spacing_m=12.0,
        reference_position_m=0.0,
    )
    selected_metadata = main.as5100_load_metadata(
        selected_config,
        loads=main.build_as5100_rail_loads(selected_config),
        arrangement="governing_envelope_sweep_requested",
    )
    governing_metadata = main.as5100_load_metadata(
        governing_config,
        loads=main.build_as5100_rail_loads(governing_config),
        arrangement="governing_envelope_sweep",
        extra={
            "selected_group_count": selected_config.group_count,
            "selected_group_spacing_m": selected_config.group_spacing_m,
            "sweep_candidate_count": 9,
            "governing_metric": "max_abs_moment_nm",
        },
    )
    result = replace(_build_plot_envelope_result(), run_metadata=governing_metadata)
    envelope_config = replace(
        _build_plot_envelope_config(2.0),
        run_metadata=selected_metadata,
    )
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.STATIC))
    window.envelope_worker = main.EnvelopeAnalysisWorker(envelope_config)
    window._pending_transition_load_source = selected_metadata
    window._pending_transition_context = main.TransitionContext(
        run_mode=StaticTransitionRunMode.ENVELOPE,
        profile_type=StaticTransitionProfileType.UNIFORM,
        template_name="Custom",
        preset_name="Custom",
        k1_n_per_m2=envelope_config.analysis_config.foundation_modulus_n_per_m2,
        k2_n_per_m2=None,
        transition_length_m=None,
        segment_length_m=None,
        domain_m=envelope_config.x_domain_m,
        analysis_config=envelope_config.analysis_config,
        analysis_mode=envelope_config.mode,
        k_profile_n_per_m2=None,
    )

    window._handle_transition_envelope_result(result)

    expected_loads = main.build_as5100_rail_loads(governing_config)
    assert window._last_transition_load_source is not None
    assert window._last_transition_load_source["arrangement"] == "governing_envelope_sweep"
    assert window._last_transition_load_source["group_count"] == 2
    assert window._last_transition_load_source["group_spacing_m"] == pytest.approx(12.0)
    assert window._last_envelope_config is not None
    assert [load.position_m for load in window._last_envelope_config.analysis_config.loads] == pytest.approx(
        [load.position_m for load in expected_loads]
    )
    assert window._last_transition_context is not None
    assert [load.load_newtons for load in window._last_transition_context.analysis_config.loads] == pytest.approx(
        [load.load_newtons for load in expected_loads]
    )
    window.close()


def test_static_transition_annotations_accept_string_mode_values(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    result = TransitionRunResult(
        mode=StaticTransitionRunMode.ENVELOPE.value,
        profile_type=StaticTransitionProfileType.RAMP.value,
        k1_n_per_m2=10.0,
        k2_n_per_m2=20.0,
        transition_length_m=2.0,
        segment_length_m=None,
        domain_length_m=4.0,
        metrics=TransitionMetrics(
            delta_w_s_m=0.0,
            delta_w_s_position_m=0.0,
            delta_w_1m_m=0.0,
            delta_w_1m_position_m=0.0,
            curvature_max_per_m=0.0,
            curvature_max_position_m=0.0,
            moment_max_nm=0.0,
            moment_max_position_m=0.0,
            energy_bending_j=0.0,
            reaction_gradient_max_n_per_m2=0.0,
            reaction_gradient_position_m=0.0,
            sleeper_load_max_n=0.0,
            sleeper_load_position_m=0.0,
        ),
        series=TransitionSeries(
            x_m=[-1.0, 0.0, 1.0],
            k_profile_n_per_m2=[10.0, 15.0, 20.0],
        ),
    )

    annotations = window._build_transition_chart_annotations(result, chart_title="Bending moment")
    annotation_text = "\n".join(text for text, _, _ in annotations)

    assert "Mode: Envelope" in annotation_text
    assert "Profile: Ramp" in annotation_text
    window.close()


def test_visual_envelope_extrema_connector_lifts_sawtooth_valleys() -> None:
    smoothed = main.MainWindow._connect_visual_envelope_extrema(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [10.0, 5.0, 9.0, 4.0, 8.0],
        mode="max",
    )

    assert smoothed[0] == pytest.approx(10.0)
    assert smoothed[2] == pytest.approx(9.0)
    assert smoothed[4] == pytest.approx(8.0)
    assert smoothed[1] == pytest.approx(9.5)
    assert smoothed[3] == pytest.approx(8.5)


def test_collect_custom_chart_series_for_active_analysis_mixes_static_metrics(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()

    x_values = [-1.0, 0.0, 1.0]
    window.deflection_plot.update_plot(
        x_values,
        [0.0, 1.0, 0.0],
        title="Deflection",
        xlabel="x (m)",
        ylabel="w (mm)",
    )
    window.moment_plot.update_plot(
        x_values,
        [0.0, 2.0, 0.0],
        title="Moment",
        xlabel="x (m)",
        ylabel="M (kN·m)",
    )
    window.shear_plot.update_plot(
        x_values,
        [0.0, 3.0, 0.0],
        title="Shear",
        xlabel="x (m)",
        ylabel="V (kN)",
    )

    series = window._collect_custom_chart_series_for_active_analysis()
    series_ids = {item.series_id for item in series}

    assert "deflection:primary" in series_ids
    assert "moment:primary" in series_ids
    assert "shear:primary" in series_ids
    window.close()


def test_build_dynamic_transition_context_coerces_string_mode_values(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.TRANSITION))
    window.probe_locations_input.setText("0.0")

    rail = window.rail_combo.currentData()
    support = window.support_combo.currentData()
    assert rail is not None
    assert support is not None

    monkeypatch.setattr(
        window.dynamic_transition_profile_combo,
        "currentData",
        lambda *_args, **_kwargs: DynamicTransitionProfileType.RAMP.value,
    )
    monkeypatch.setattr(
        window.dynamic_transition_run_mode_combo,
        "currentData",
        lambda *_args, **_kwargs: DynamicTransitionRunMode.SINGLE.value,
    )

    context = window._build_dynamic_transition_context(rail=rail, support=support)
    assert context.profile_type == DynamicTransitionProfileType.RAMP
    assert context.run_mode == DynamicTransitionRunMode.SINGLE
    window.close()


def test_build_dynamic_transition_context_uses_k1_for_ratio_damping(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.TRANSITION))
    window.probe_locations_input.setText("0.0")
    window.damping_mode_combo.setCurrentIndex(window.damping_mode_combo.findData("ratio"))
    window.foundation_damping_model_combo.setCurrentIndex(
        window.foundation_damping_model_combo.findData(main.DampingModel.VISCOUS)
    )
    window.damping_ratio_input.set_value(0.23)
    window.dynamic_transition_k1_input.set_value(55.0)

    rail = window.rail_combo.currentData()
    support = window.support_combo.currentData()
    assert rail is not None
    assert support is not None
    assert not math.isclose(
        support.foundation_modulus_n_per_m2,
        main.mn_per_m2_to_n_per_m2(window.dynamic_transition_k1_input.value()),
        rel_tol=1.0e-12,
    )

    context = window._build_dynamic_transition_context(rail=rail, support=support)
    expected = 2.0 * window.damping_ratio_input.value() * math.sqrt(
        main.mn_per_m2_to_n_per_m2(window.dynamic_transition_k1_input.value()) * rail.mass_kg_per_m
    )
    assert context.foundation_modulus_n_per_m2 == pytest.approx(
        main.mn_per_m2_to_n_per_m2(window.dynamic_transition_k1_input.value())
    )
    assert context.foundation_damping_n_s_per_m2 == pytest.approx(expected)
    window.close()


def test_dynamic_transition_full_profile_enforces_supported_advanced_options(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.TRANSITION))
    window.dynamic_advanced_group.setChecked(True)
    window.dynamic_excitation_mode_combo.setCurrentIndex(
        window.dynamic_excitation_mode_combo.findData(main.DynamicExcitationMode.MOVING_OSCILLATOR)
    )
    window.dynamic_boundary_mode_combo.setCurrentIndex(
        window.dynamic_boundary_mode_combo.findData(main.DynamicBoundaryMode.PERIODIC_WRAP)
    )
    window.irregularity_mode_combo.setCurrentIndex(
        window.irregularity_mode_combo.findData(main.IrregularityMode.SYNTHETIC_PSD)
    )
    window.foundation_damping_model_combo.setCurrentIndex(
        window.foundation_damping_model_combo.findData(main.DampingModel.HYSTERETIC)
    )
    window.dynamic_transition_solver_fidelity_combo.setCurrentIndex(
        window.dynamic_transition_solver_fidelity_combo.findData("full_profile")
    )

    window._enforce_dynamic_transition_advanced_constraints()

    assert window.dynamic_excitation_mode_combo.currentData() == main.DynamicExcitationMode.MOVING_LOAD
    assert not window.dynamic_excitation_mode_combo.isEnabled()
    assert window.dynamic_boundary_mode_combo.currentData() == main.DynamicBoundaryMode.ZERO_PAD
    assert window.irregularity_mode_combo.currentData() is None
    assert window.foundation_damping_model_combo.currentData() == main.DampingModel.VISCOUS
    window.close()


def test_build_dynamic_transition_context_rejects_invalid_string_mode_values(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.TRANSITION))
    window.probe_locations_input.setText("0.0")

    rail = window.rail_combo.currentData()
    support = window.support_combo.currentData()
    assert rail is not None
    assert support is not None

    monkeypatch.setattr(
        window.dynamic_transition_profile_combo,
        "currentData",
        lambda *_args, **_kwargs: "invalid-profile",
    )
    with pytest.raises(ValueError, match="valid dynamic transition profile type"):
        window._build_dynamic_transition_context(rail=rail, support=support)

    monkeypatch.setattr(
        window.dynamic_transition_profile_combo,
        "currentData",
        lambda *_args, **_kwargs: DynamicTransitionProfileType.UNIFORM.value,
    )
    monkeypatch.setattr(
        window.dynamic_transition_run_mode_combo,
        "currentData",
        lambda *_args, **_kwargs: "invalid-run-mode",
    )
    with pytest.raises(ValueError, match="valid dynamic transition run mode"):
        window._build_dynamic_transition_context(rail=rail, support=support)
    window.close()


def test_build_dynamic_context_coerces_string_dynamic_mode_to_transition(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.probe_locations_input.setText("0.0")

    rail = window.rail_combo.currentData()
    sleeper = window.sleeper_combo.currentData()
    support = window.support_combo.currentData()
    assert rail is not None
    assert sleeper is not None
    assert support is not None

    monkeypatch.setattr(
        window.dynamic_mode_combo,
        "currentData",
        lambda *_args, **_kwargs: DynamicMode.TRANSITION.value,
    )
    monkeypatch.setattr(
        window.dynamic_transition_profile_combo,
        "currentData",
        lambda *_args, **_kwargs: DynamicTransitionProfileType.UNIFORM.value,
    )
    monkeypatch.setattr(
        window.dynamic_transition_run_mode_combo,
        "currentData",
        lambda *_args, **_kwargs: DynamicTransitionRunMode.SINGLE.value,
    )

    config, mode = window._build_dynamic_context(rail=rail, sleeper=sleeper, support=support)
    assert mode == DynamicMode.TRANSITION
    assert isinstance(config, DynamicTransitionConfig)
    window.close()


def test_main_window_title_uses_trademark_name(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()

    assert window.windowTitle() == "1DTransport.com: BOEF Calculation Tool"
    window.close()


def test_ballast_thickness_input_defaults_and_updates_capping_pressure(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    assert window.ballast_thickness_input.value() == pytest.approx(300.0)

    analysis_inputs = main.AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=5,
    )
    config = AnalysisConfig(
        loads=analysis_inputs.loads,
        foundation_modulus_n_per_m2=analysis_inputs.foundation_modulus_n_per_m2,
        elastic_modulus_pa=analysis_inputs.elastic_modulus_pa,
        moment_inertia_m4=analysis_inputs.moment_inertia_m4,
        section_modulus_m3=analysis_inputs.section_modulus_m3,
        sleeper_spacing_m=analysis_inputs.sleeper_spacing_m,
        sleeper_length_m=analysis_inputs.sleeper_length_m,
        sleeper_width_m=analysis_inputs.sleeper_width_m,
        sample_count=analysis_inputs.sample_count,
    )
    window._last_analysis_config = config
    result = compute_track_response(analysis_inputs)

    window.ballast_thickness_input.set_value(300.0)
    stress_300 = window._build_stress_from_analysis_result(result)
    window.ballast_thickness_input.set_value(600.0)
    stress_600 = window._build_stress_from_analysis_result(result)
    assert stress_300 is not None
    assert stress_600 is not None
    assert stress_300.q_capping_comp_pa is not None
    assert stress_600.q_capping_comp_pa is not None
    assert max(stress_600.q_capping_comp_pa) < max(stress_300.q_capping_comp_pa)
    window.close()


def test_static_pressure_chart_shows_ballast_and_below_ballast_series(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    analysis_inputs = main.AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=80_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=21,
    )
    window._last_analysis_config = AnalysisConfig(
        loads=analysis_inputs.loads,
        foundation_modulus_n_per_m2=analysis_inputs.foundation_modulus_n_per_m2,
        elastic_modulus_pa=analysis_inputs.elastic_modulus_pa,
        moment_inertia_m4=analysis_inputs.moment_inertia_m4,
        section_modulus_m3=analysis_inputs.section_modulus_m3,
        sleeper_spacing_m=analysis_inputs.sleeper_spacing_m,
        sleeper_length_m=analysis_inputs.sleeper_length_m,
        sleeper_width_m=analysis_inputs.sleeper_width_m,
        sample_count=analysis_inputs.sample_count,
    )
    window.ballast_thickness_input.set_value(300.0)

    result = compute_track_response(analysis_inputs)
    window._last_analysis_result = result
    window._render_analysis_result(result)

    series = window.pressure_plot.rendered_series()
    labels = [item.label for item in series]
    assert "Ballast top / sleeper contact" in labels
    assert any(label.startswith("Below ballast / capping top") for label in labels)
    ballast = next(item for item in series if item.label == "Ballast top / sleeper contact")
    below_ballast = next(item for item in series if item.label.startswith("Below ballast / capping top"))
    assert max(below_ballast.y) < max(ballast.y)
    assert window.pressure_plot.axes.get_title() == "Ballast top and depth pressures"
    pressure_footer_left, pressure_footer_right = window.pressure_plot.footer_texts()
    assert "Analysis: Static" in pressure_footer_left
    assert "Chart: Ballast top and depth pressures" in pressure_footer_left
    assert "|w|max:" in pressure_footer_right
    stress_footer_left, stress_footer_right = window.stress_plot.footer_texts()
    assert "Analysis: Static" in stress_footer_left
    assert "Chart: Stress" in stress_footer_left
    assert "|M|max:" in stress_footer_right
    window.close()


def test_dynamic_custom_chart_source_ids_include_stress(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()

    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    dynamic_ids = window._custom_chart_active_chart_ids()
    assert "stress" in dynamic_ids

    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.SPECIAL))
    special_ids = window._custom_chart_active_chart_ids()
    assert "stress" in special_ids
    window.close()


def test_handle_dynamic_result_renders_peak_dynamic_stress_with_unavailable_note(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.STEADY_STATE))

    window._handle_dynamic_result(_build_dynamic_result(scale=1.0))

    labels = [line.get_label() for line in window.stress_plot.axes.lines]
    texts = [text.get_text() for text in window.stress_plot.axes.texts]
    assert any("top fibre" in label for label in labels)
    assert any("bottom fibre" in label for label in labels)
    assert any("dynamic mode" in text for text in texts)
    window.close()


def test_dynamic_chart_annotations_include_critical_values_for_non_deflection_charts(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.STEADY_STATE))

    window._handle_dynamic_result(_build_dynamic_result(scale=1.0))

    moment_texts = [text.get_text() for text in window.dynamic_moment_plot.axes.texts]
    shear_texts = [text.get_text() for text in window.dynamic_shear_plot.axes.texts]
    reaction_texts = [text.get_text() for text in window.dynamic_reaction_plot.axes.texts]
    assert any("M_max" in text and "|M|_max" in text for text in moment_texts)
    assert any("V_max" in text and "|V|_max" in text for text in shear_texts)
    assert any("R_max" in text and "|R|_max" in text for text in reaction_texts)
    assert any("EI:" in text and "DAF:" in text and "Δξ:" in text for text in moment_texts)
    window.close()


def test_dynamic_probe_charts_show_annotation_with_probe_and_critical_values(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.STEADY_STATE))

    window._handle_dynamic_result(_build_dynamic_result(scale=1.0, probes=2))
    window.probe_selection_combo.setCurrentIndex(1)
    window._update_dynamic_probe_plots()

    time_texts = [text.get_text() for text in window.dynamic_time_plot.axes.texts]
    fft_texts = [text.get_text() for text in window.dynamic_fft_plot.axes.texts]
    assert any("Probe: x=1.000 m" in text for text in time_texts)
    assert any("w_max" in text and "|w|_max" in text for text in time_texts)
    assert any("|W|_max" in text and "f range" in text for text in fft_texts)
    window.close()


def test_dynamic_compact_annotations_include_as5100_traceability_without_full_trace(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.STEADY_STATE))
    window.dynamic_annotation_mode_combo.setCurrentIndex(
        window.dynamic_annotation_mode_combo.findData(main.DynamicAnnotationMode.COMPACT)
    )
    window.as5100_loads_checkbox.setChecked(True)
    window.as5100_model_combo.setCurrentIndex(window.as5100_model_combo.findData(main.AS5100_MODEL_300LA))
    window.as5100_group_count_input.setValue(3)
    window.as5100_group_spacing_input.set_value(12.0)
    window.as5100_reference_input.set_value(0.25)
    window._last_dynamic_result = _build_dynamic_result(scale=1.0)
    window._last_dynamic_mode = DynamicMode.STEADY_STATE
    window._last_dynamic_load_source = window._capture_load_source_metadata()

    annotations = window._build_dynamic_chart_annotations(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 0.0],
        value_symbol="M",
        value_unit="kN·m",
        axis_symbol="ξ",
        axis_unit="m",
        chart_title="Dynamic moment",
    )

    metadata_text = annotations[0][0]
    kpi_text = annotations[1][0]
    assert "Load source: AS5100 300LA fixed (AS5100.2:2017)" in metadata_text
    assert "Axles: 13, max 360 kN; 3 group(s) @ 12.00 m" in metadata_text
    assert "Solver load: 180 kN/rail" in metadata_text
    assert "x0=0.250 m; no automatic DLA" in metadata_text
    assert "EI:" not in metadata_text
    assert "DAF:" in kpi_text
    assert "Global |w|" in kpi_text
    window.close()


def test_dynamic_chart_label_controls_can_show_only_output_overlay(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window._last_dynamic_result = _build_dynamic_result(scale=1.0)
    window._last_dynamic_mode = DynamicMode.STEADY_STATE

    window.chart_input_labels_checkbox.setChecked(False)
    annotations = window._build_dynamic_chart_annotations(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 0.0],
        value_symbol="w",
        value_unit="mm",
        axis_symbol="ξ",
        axis_unit="m",
        chart_title="Dynamic deflection",
    )

    assert len(annotations) == 1
    assert "Analysis:" not in annotations[0][0]
    assert "w_max" in annotations[0][0]

    window.chart_output_labels_checkbox.setChecked(False)
    assert (
        window._build_dynamic_chart_annotations(
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 0.0],
            value_symbol="w",
            value_unit="mm",
            axis_symbol="ξ",
            axis_unit="m",
            chart_title="Dynamic deflection",
        )
        == []
    )
    window.close()


def test_chart_max_min_label_control_rerenders_dynamic_point_labels(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.STEADY_STATE))

    window._handle_dynamic_result(_build_dynamic_result(scale=1.0))
    assert any(isinstance(text, Annotation) for text in window.dynamic_deflection_plot.axes.texts)

    window.chart_extrema_labels_checkbox.setChecked(False)
    assert not any(isinstance(text, Annotation) for text in window.dynamic_deflection_plot.axes.texts)
    window.close()


def test_dynamic_annotation_mode_off_returns_no_annotations(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_annotation_mode_combo.setCurrentIndex(
        window.dynamic_annotation_mode_combo.findData(main.DynamicAnnotationMode.OFF)
    )

    annotations = window._build_dynamic_chart_annotations(
        [0.0, 1.0],
        [0.0, 1.0],
        value_symbol="w",
        value_unit="mm",
        axis_symbol="ξ",
        axis_unit="m",
        chart_title="Dynamic deflection",
    )

    assert annotations == []
    window.close()


def test_dynamic_transition_annotation_metadata_uses_transition_run_config(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.TRANSITION))
    window.speed_input.set_value(99.0)
    window.load_magnitude_input.set_value(222.0)

    rail = window.rail_combo.currentData()
    support = window.support_combo.currentData()
    assert rail is not None
    assert support is not None
    transition_cfg = window._build_dynamic_transition_context(rail=rail, support=support)
    transition_cfg = replace(
        transition_cfg,
        speed_m_per_s=12.5,
        loads=[PointLoad(position_m=0.0, load_newtons=123_000.0)],
    )
    window._last_dynamic_mode = DynamicMode.TRANSITION
    window._last_dynamic_transition_config = transition_cfg
    window._last_dynamic_config = None

    annotations = window._build_dynamic_chart_annotations(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 0.0],
        value_symbol="M",
        value_unit="kN·m",
        axis_symbol="ξ",
        axis_unit="m",
        chart_title="Dynamic moment",
    )
    metadata_text = annotations[0][0]
    assert "Mode: Transition" in metadata_text
    assert "Speed: 12.50 m/s" in metadata_text
    assert "Wheel load: 123.00 kN" in metadata_text
    window.close()


def test_handle_dynamic_transition_result_shows_transition_k_profile_tab(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    window.analysis_type_combo.setCurrentIndex(window.analysis_type_combo.findData(main.AnalysisType.DYNAMIC))
    window.dynamic_mode_combo.setCurrentIndex(window.dynamic_mode_combo.findData(DynamicMode.TRANSITION))

    window._handle_dynamic_result(_build_dynamic_transition_result())

    assert window.tab_widget.isTabVisible(window.transition_profile_tab_index)
    assert window.transition_profile_plot.axes.lines
    texts = [text.get_text() for text in window.transition_profile_plot.axes.texts]
    assert any("Analysis: Dynamic transition" in text for text in texts)
    assert any("Fidelity: Screening" in text for text in texts)
    window.close()


def test_handle_dipped_joint_result_marks_stress_unavailable(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)
    window = main.MainWindow()
    result = DippedJointResult(
        static_load_n=100_000.0,
        p1_n=120_000.0,
        p2_n=110_000.0,
        p1_dynamic_amplification=1.2,
        p2_dynamic_amplification=1.1,
    )

    window._handle_dipped_joint_result(result)

    texts = [text.get_text() for text in window.stress_plot.axes.texts]
    assert any("dipped-joint mode" in text for text in texts)
    window.close()

"""Sensitivity and design recommendation dialog."""

from __future__ import annotations

from dataclasses import asdict, replace
import textwrap
from typing import Callable, Sequence

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.export_helpers import write_export_metadata
from core.analysis_engine import AnalysisConfig
from core.model import PointLoad
from core.sensitivity import (
    DesignCriteria,
    SensitivityScenario,
    SensitivityRunMode,
    SensitivityRunResult,
    SensitivityTransitionContext,
    SensitivityVariable,
    build_scenarios,
    result_payload,
    rescore_sensitivity_result,
    run_sensitivity,
    write_sensitivity_csv,
)
from core.units import cm2_to_m2, n_per_m2_to_mn_per_m2, n_to_kn, m_to_mm
from db.models import LoadCase, Pad, Project, Rail, RailAdmissibleStress, Result, Sleeper, SupportProfile, TrackConfig
from db import crud


StaticContextBuilder = Callable[[TrackConfig, float], AnalysisConfig]
TransitionContextBuilder = Callable[[TrackConfig, float], SensitivityTransitionContext]


COLOR_BASELINE = QColor("#f1f3f5")
COLOR_BEST = QColor("#d8f3dc")
COLOR_IMPROVED = QColor("#eaf7ed")
COLOR_WORSE = QColor("#fde2e2")
COLOR_WARNING = QColor("#fff3bf")
COLOR_TEXT = QColor("#1f2933")
CHART_GREEN = "#2f855a"
CHART_RED = "#c2410c"
CHART_BLUE = "#2b6cb0"
CHART_AMBER = "#b7791f"
CHART_GREY = "#718096"
CHART_DARK = "#1a202c"


def scenario_change_label(parameter: str, factor: float) -> str:
    """Return a readable scenario change label."""
    if parameter == "baseline":
        return "Baseline"
    change = (factor - 1.0) * 100.0
    sign = "+" if change > 0 else ""
    return f"{parameter.replace('_', ' ')} {sign}{change:.0f}%"


def scenario_visual_state(scenario) -> str:
    """Classify a sensitivity row for table and chart styling."""
    if scenario.changed_parameter == "baseline":
        return "baseline"
    if scenario.warning:
        return "warning"
    if getattr(scenario, "decision", None) is not None and scenario.decision.status == "fail":
        return "worse"
    if getattr(scenario, "decision", None) is not None and scenario.decision.status == "warning":
        return "warning"
    if scenario.rank == 1:
        return "best"
    if scenario.percent_improvement is not None and scenario.percent_improvement >= 0.0:
        return "improved"
    return "worse"


def scenario_row_color(state: str) -> QColor:
    if state == "baseline":
        return COLOR_BASELINE
    if state == "best":
        return COLOR_BEST
    if state == "improved":
        return COLOR_IMPROVED
    if state == "warning":
        return COLOR_WARNING
    return COLOR_WORSE


class SensitivityWorker(QObject):
    """Background worker for sensitivity sweeps."""

    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        *,
        base_config: AnalysisConfig,
        variables: Sequence[SensitivityVariable],
        mode: SensitivityRunMode,
        transition_context: SensitivityTransitionContext | None,
        criteria: DesignCriteria | None,
        additional_scenarios: Sequence[SensitivityScenario] | None,
    ) -> None:
        super().__init__()
        self.base_config = base_config
        self.variables = list(variables)
        self.mode = mode
        self.transition_context = transition_context
        self.criteria = criteria
        self.additional_scenarios = list(additional_scenarios or [])
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            result = run_sensitivity(
                base_config=self.base_config,
                variables=self.variables,
                mode=self.mode,
                transition_context=self.transition_context,
                criteria=self.criteria,
                additional_scenarios=self.additional_scenarios,
                progress_callback=self.progress.emit,
                cancel_check=lambda: self._cancel_requested,
            )
        except RuntimeError as exc:
            if "cancelled" in str(exc).lower():
                self.cancelled.emit()
                return
            self.failed.emit(str(exc))
            return
        except ValueError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive GUI boundary
            self.failed.emit(f"{exc.__class__.__name__}: {exc}")
            return
        self.finished.emit(result)


class SensitivityDialog(QDialog):
    """Standalone Stage 1 sensitivity and design screening workflow."""

    alternatives_saved = Signal()

    def __init__(
        self,
        *,
        session: Session,
        project: Project | None,
        track_config: TrackConfig | None,
        current_load_n: float,
        build_static_context: StaticContextBuilder,
        build_transition_context: TransitionContextBuilder,
        current_loads: Sequence[PointLoad] | None = None,
        current_load_source_metadata: dict[str, object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sensitivity + Design Recommendation")
        self.resize(1180, 760)
        self.session = session
        self.project = project or (track_config.project if track_config is not None else None)
        self.track_config = track_config
        self.current_load_n = current_load_n
        self.current_loads = list(current_loads or [PointLoad(position_m=0.0, load_newtons=current_load_n)])
        self.current_load_source_metadata = current_load_source_metadata or {
            "source_type": "single_point_load"
        }
        self.build_static_context = build_static_context
        self.build_transition_context = build_transition_context
        self.worker_thread: QThread | None = None
        self.worker: SensitivityWorker | None = None
        self.result: SensitivityRunResult | None = None
        self._result_stale = False

        self.config_combo = QComboBox()
        self.preset_combo = QComboBox()
        self.preset_status_label = QLabel()
        self.static_mode_radio = QRadioButton("Static sensitivity")
        self.transition_mode_radio = QRadioButton("Transition design sensitivity")
        self.dynamic_mode_radio = QRadioButton("Dynamic sensitivity (future)")
        self.use_current_load_selection_checkbox = QCheckBox("Use current load selection")
        self.use_current_load_selection_checkbox.setChecked(True)
        self.use_current_load_selection_checkbox.toggled.connect(self._mark_results_stale)
        self.use_current_load_selection_checkbox.toggled.connect(self._sync_variable_availability)
        self.use_current_load_selection_checkbox.toggled.connect(lambda _checked: self._sync_baseline_summary())
        self.baseline_source_label = QLabel()
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.result_filter_combo = QComboBox()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status_label = QLabel("Ready.")
        self.run_button = QPushButton("Run sensitivity")
        self.cancel_button = QPushButton("Cancel")
        self.export_button = QPushButton("Export sensitivity CSV")
        self.save_alternative_button = QPushButton("Save selected as Design Alternative")
        self.apply_best_button = QPushButton("Apply best as new track config")
        self.recommendation_label = QLabel(
            "Run sensitivity to generate screening recommendations. Results are not final design acceptance."
        )
        self.recommendation_label.setWordWrap(True)
        self.table = QTableWidget()
        self.plots = QTabWidget()
        self.variable_checks: dict[SensitivityVariable, QCheckBox] = {}
        self.criteria_inputs: dict[str, QDoubleSpinBox] = {}

        self._build_ui()
        self._populate_configs()
        self._populate_presets()
        self._sync_baseline_summary()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        top_layout = QGridLayout()
        top_layout.addWidget(QLabel("Track config"), 0, 0)
        top_layout.addWidget(self.config_combo, 0, 1)
        self.config_combo.currentIndexChanged.connect(self._handle_config_changed)
        top_layout.addWidget(QLabel("Preset"), 0, 2)
        top_layout.addWidget(self.preset_combo, 0, 3)
        self.preset_combo.currentIndexChanged.connect(self._handle_preset_changed)
        self.preset_status_label.setStyleSheet("color: #4a5568;")
        top_layout.addWidget(self.preset_status_label, 1, 2, 1, 2)
        top_layout.addWidget(self.use_current_load_selection_checkbox, 2, 0, 1, 2)

        self.static_mode_radio.setChecked(True)
        self.dynamic_mode_radio.setEnabled(False)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.static_mode_radio)
        mode_layout.addWidget(self.transition_mode_radio)
        mode_layout.addWidget(self.dynamic_mode_radio)
        mode_layout.addStretch()
        top_layout.addLayout(mode_layout, 1, 0, 1, 2)
        self.static_mode_radio.toggled.connect(self._sync_variable_availability)
        self.static_mode_radio.toggled.connect(self._sync_baseline_summary)
        self.transition_mode_radio.toggled.connect(self._sync_variable_availability)
        self.transition_mode_radio.toggled.connect(self._sync_baseline_summary)
        main_layout.addLayout(top_layout)

        self.baseline_source_label.setStyleSheet(
            "background: #edf2f7; color: #1f2933; padding: 5px; border: 1px solid #cbd5e0;"
        )
        self.baseline_source_label.setWordWrap(True)
        main_layout.addWidget(self.baseline_source_label)

        self.summary_label.setStyleSheet("color: #4f4f4f;")
        main_layout.addWidget(self.summary_label)

        variables_group = QGroupBox("Sweep variables")
        variables_layout = QGridLayout(variables_group)
        labels = {
            SensitivityVariable.SUPPORT_STIFFNESS: "Support stiffness k",
            SensitivityVariable.SLEEPER_SPACING: "Sleeper spacing",
            SensitivityVariable.PAD_STIFFNESS: "Pad stiffness",
            SensitivityVariable.WHEEL_LOAD: "Wheel load",
            SensitivityVariable.AS5100_POSITION: "AS5100 train position",
            SensitivityVariable.SPEED: "Speed",
            SensitivityVariable.BALLAST_DEPTH: "Ballast depth",
            SensitivityVariable.TRANSITION_LENGTH: "Transition length Lt",
        }
        for index, variable in enumerate(SensitivityVariable):
            check = QCheckBox(labels[variable])
            check.setChecked(
                variable
                not in {
                    SensitivityVariable.AS5100_POSITION,
                    SensitivityVariable.TRANSITION_LENGTH,
                }
            )
            if variable == SensitivityVariable.TRANSITION_LENGTH:
                check.setToolTip("Available for transition sensitivity with ramp or exponential profiles and valid Lt.")
            if variable == SensitivityVariable.AS5100_POSITION:
                check.setToolTip(
                    "AS5100 only: shifts the selected fixed train arrangement by +/-0.5 m and +/-1.0 m."
                )
            self.variable_checks[variable] = check
            check.toggled.connect(self._mark_results_stale)
            variables_layout.addWidget(check, index // 4, index % 4)
        main_layout.addWidget(variables_group)
        self._sync_variable_availability()

        criteria_group = QGroupBox("Design Criteria (screening limits)")
        criteria_layout = QGridLayout(criteria_group)
        criteria_specs = [
            ("max_deflection_mm", "Max deflection", "mm", 6.0, 0.1, 1000.0),
            ("rail_stress_mpa", "Rail stress", "MPa", self._default_rail_stress_limit_mpa(), 1.0, 2000.0),
            ("max_sleeper_load_kn", "Sleeper load", "kN", 250.0, 1.0, 10000.0),
            ("ballast_pressure_kpa", "Ballast pressure", "kPa", 300.0, 1.0, 10000.0),
            ("formation_pressure_kpa", "Formation stress", "kPa", 150.0, 1.0, 10000.0),
            ("subgrade_pressure_kpa", "Subgrade stress", "kPa", 100.0, 1.0, 10000.0),
            ("deep_subgrade_pressure_kpa", "Deep subgrade", "kPa", 70.0, 1.0, 10000.0),
            ("transition_metric_mm", "Transition metric", "mm", 1.5, 0.01, 1000.0),
        ]
        for index, (key, label, unit, value, minimum, maximum) in enumerate(criteria_specs):
            criteria_layout.addWidget(QLabel(label), index // 4 * 2, index % 4)
            spin = QDoubleSpinBox()
            spin.setDecimals(3 if unit in {"mm"} else 2)
            spin.setRange(minimum, maximum)
            spin.setValue(value)
            spin.setSuffix(f" {unit}")
            spin.setToolTip("Screening limit only; confirm final acceptance with project design criteria.")
            spin.valueChanged.connect(self._handle_criteria_changed)
            self.criteria_inputs[key] = spin
            criteria_layout.addWidget(spin, index // 4 * 2 + 1, index % 4)
        main_layout.addWidget(criteria_group)

        run_layout = QHBoxLayout()
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.cancel_button)
        run_layout.addWidget(self.progress, stretch=1)
        run_layout.addWidget(self.status_label)
        run_layout.addWidget(self.save_alternative_button)
        run_layout.addWidget(self.apply_best_button)
        run_layout.addWidget(self.export_button)
        main_layout.addLayout(run_layout)
        self.run_button.clicked.connect(self._run)
        self.cancel_button.clicked.connect(self._cancel)
        self.export_button.clicked.connect(self._export)
        self.save_alternative_button.clicked.connect(self._save_selected_alternatives)
        self.apply_best_button.clicked.connect(self._apply_best_as_track_config)
        self.cancel_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.save_alternative_button.setEnabled(False)
        self.apply_best_button.setEnabled(False)

        self.table.setColumnCount(21)
        self.table.setHorizontalHeaderLabels(
            [
                "Scenario",
                "Parameter",
                "Value",
                "Deflection (mm)",
                "Moment (kN.m)",
                "Sleeper load (kN)",
                "Rail stress (MPa)",
                "Ballast pressure (kPa)",
                "Formation stress (kPa)",
                "Subgrade stress (kPa)",
                "Deep subgrade (kPa)",
                "Transition metric (mm)",
                "Engineering score",
                "Constructability",
                "Combined score",
                "Improvement (%)",
                "Max utilization",
                "Governing criterion",
                "Decision",
                "Warning/status",
                "Rank",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Result filter"))
        self.result_filter_combo.addItem("All scenarios", "all")
        self.result_filter_combo.addItem("Valid only", "valid")
        self.result_filter_combo.addItem("Warnings only", "warnings")
        self.result_filter_combo.addItem("Best and worst", "best_worst")
        self.result_filter_combo.currentIndexChanged.connect(self._apply_result_filter)
        filter_layout.addWidget(self.result_filter_combo)
        filter_layout.addStretch()

        self.plots.addTab(self._plot_widget("improvement"), "Improvement")
        self.plots.addTab(self._plot_widget("tornado"), "Tornado")
        self.plots.addTab(self._plot_widget("comparison"), "Baseline vs best")
        self.open_large_plot_button = QPushButton("Open larger plot")
        self.open_large_plot_button.setEnabled(False)
        self.open_large_plot_button.clicked.connect(self._open_large_plot)

        tabs = QTabWidget()
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        table_layout.addLayout(filter_layout)
        table_layout.addWidget(self.table)
        plots_tab = QWidget()
        plots_layout = QVBoxLayout(plots_tab)
        plots_layout.addWidget(self.plots)
        plots_layout.addWidget(self.open_large_plot_button, alignment=Qt.AlignRight)
        rec_tab = QWidget()
        rec_layout = QVBoxLayout(rec_tab)
        rec_layout.addWidget(self.recommendation_label)
        rec_layout.addStretch()
        tabs.addTab(table_tab, "Results")
        tabs.addTab(plots_tab, "Plots")
        tabs.addTab(rec_tab, "Recommendation")
        main_layout.addWidget(tabs, stretch=1)

    def _plot_widget(self, name: str) -> FigureCanvas:
        figure = Figure(figsize=(6, 3.4), tight_layout=True)
        canvas = FigureCanvas(figure)
        canvas.setObjectName(name)
        return canvas

    def _default_rail_stress_limit_mpa(self) -> float:
        value = self.session.scalar(
            select(RailAdmissibleStress.repeated_stress_mpa)
            .order_by(RailAdmissibleStress.repeated_stress_mpa)
        )
        return float(value) if value is not None else 55.0

    def _populate_presets(self) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Custom", "custom")
        self.preset_combo.addItem("Formation pressure", "formation")
        self.preset_combo.addItem("Rail stress", "rail")
        self.preset_combo.addItem("Transition design", "transition")
        self.preset_combo.addItem("Cost-aware screening", "cost")
        self.preset_combo.blockSignals(False)
        self._handle_preset_changed()

    def _handle_preset_changed(self) -> None:
        preset = self.preset_combo.currentData()
        if preset is None:
            return
        variables_by_preset = {
            "custom": None,
            "formation": {
                SensitivityVariable.BALLAST_DEPTH,
                SensitivityVariable.SUPPORT_STIFFNESS,
                SensitivityVariable.SLEEPER_SPACING,
                SensitivityVariable.PAD_STIFFNESS,
                SensitivityVariable.WHEEL_LOAD,
            },
            "rail": {
                SensitivityVariable.WHEEL_LOAD,
                SensitivityVariable.SUPPORT_STIFFNESS,
                SensitivityVariable.SPEED,
            },
            "transition": {
                SensitivityVariable.SUPPORT_STIFFNESS,
                SensitivityVariable.TRANSITION_LENGTH,
                SensitivityVariable.SLEEPER_SPACING,
                SensitivityVariable.PAD_STIFFNESS,
            },
            "cost": {
                SensitivityVariable.SLEEPER_SPACING,
            },
        }
        selected = variables_by_preset.get(str(preset))
        if selected is not None:
            for variable, check in self.variable_checks.items():
                check.setChecked(variable in selected and check.isEnabled())
        self._sync_preset_status()
        self._mark_results_stale()

    def _sync_preset_status(self) -> None:
        config = self.track_config
        supports = self.session.scalars(select(SupportProfile).order_by(SupportProfile.foundation_modulus_n_per_m2)).all()
        pads = self.session.scalars(select(Pad).order_by(Pad.stiffness_newtons_per_meter)).all()
        rails = self.session.scalars(select(Rail).order_by(Rail.mass_kg_per_m)).all()
        sleepers = self.session.scalars(select(Sleeper).order_by(Sleeper.length_m)).all()
        parts = [
            f"{len(supports)} support profiles",
            f"{len(pads)} pads",
            f"{len(rails)} rails",
            f"{len(sleepers)} sleepers",
        ]
        if config is not None and len(pads) <= 1:
            parts.append("pad material alternatives unavailable")
        self.preset_status_label.setText("Database preset source: " + "; ".join(parts) + ".")

    def _populate_configs(self) -> None:
        self.config_combo.blockSignals(True)
        self.config_combo.clear()
        configs: list[TrackConfig] = []
        if self.track_config is not None:
            configs = [self.track_config]
        elif self.project is not None:
            configs = list(sorted(self.project.track_configs, key=lambda item: item.name))
        for config in configs:
            self.config_combo.addItem(config.name, config)
        self.config_combo.blockSignals(False)
        if self.config_combo.count() > 0:
            self.config_combo.setCurrentIndex(0)
            self.track_config = self.config_combo.currentData()
        self._sync_preset_status()

    def _handle_config_changed(self) -> None:
        self.track_config = self.config_combo.currentData()
        if self.result is None:
            self._set_result_actions_enabled(False)
            self.table.setRowCount(0)
        else:
            self._mark_results_stale()
        self._sync_preset_status()
        self._sync_baseline_summary()

    def _sync_variable_availability(self) -> None:
        is_transition = self.transition_mode_radio.isChecked()
        self.variable_checks[SensitivityVariable.TRANSITION_LENGTH].setEnabled(is_transition)
        self.variable_checks[SensitivityVariable.AS5100_POSITION].setEnabled(self._as5100_position_sensitivity_available())
        if is_transition:
            self.variable_checks[SensitivityVariable.TRANSITION_LENGTH].setChecked(True)
        else:
            self.variable_checks[SensitivityVariable.TRANSITION_LENGTH].setChecked(False)
        if not self.variable_checks[SensitivityVariable.AS5100_POSITION].isEnabled():
            self.variable_checks[SensitivityVariable.AS5100_POSITION].setChecked(False)
        self._handle_preset_changed()
        self._mark_results_stale()

    def _sync_baseline_summary(self) -> None:
        config = self.track_config
        if config is None:
            self.baseline_source_label.setText("Baseline source: no track config selected.")
            self.summary_label.setText("Select a track config before running sensitivity.")
            self.run_button.setEnabled(False)
            return
        self.session.refresh(config)
        load_case = self._latest_load_case(config)
        if self.use_current_load_selection_checkbox.isChecked():
            load_text = self._current_load_selection_label()
            load_source = "current UI load selection"
        else:
            load_text = (
                f"{load_case.name}: {n_to_kn(load_case.load_newtons):.2f} kN"
                if load_case is not None
                else f"Current UI temporary load: {n_to_kn(self.current_load_n):.2f} kN"
            )
            load_source = "latest related load case" if load_case is not None else "current UI temporary load"
        mode_text = "transition design sensitivity" if self.transition_mode_radio.isChecked() else "static sensitivity"
        self.baseline_source_label.setText(
            f"Baseline source: {mode_text}; config '{config.name}'; load from {load_source}."
        )
        self.summary_label.setText(
            "\n".join(
                [
                    f"Project: {config.project.name if config.project else 'Unknown'}",
                    f"Track config: {config.name}",
                    f"Rail: {config.rail.name if config.rail else 'Unknown'}",
                    f"Sleeper: {config.sleeper.name if config.sleeper else 'Unknown'}",
                    f"Pad: {config.pad.name if config.pad else 'Unknown'}",
                    f"Support: {config.support_profile.name if config.support_profile else 'Unknown'}",
                    f"Sleeper spacing: {m_to_mm(config.sleeper_spacing_m):.1f} mm",
                    f"Baseline load: {load_text}",
                    "Recommendation is screening guidance only, not final design acceptance.",
                ]
            )
        )
        self.run_button.setEnabled(True)

    def _latest_load_case(self, config: TrackConfig) -> LoadCase | None:
        result = self.session.scalar(
            select(Result)
            .where(Result.track_config_id == config.id)
            .order_by(Result.id.desc())
        )
        if result is None:
            return None
        return self.session.get(LoadCase, result.load_case_id)

    def _baseline_load_n(self, config: TrackConfig) -> float:
        if self.use_current_load_selection_checkbox.isChecked():
            return max(abs(load.load_newtons) for load in self.current_loads)
        load_case = self._latest_load_case(config)
        if load_case is not None:
            return load_case.load_newtons
        return self.current_load_n

    def _current_load_selection_label(self) -> str:
        source_type = self.current_load_source_metadata.get("source_type")
        if source_type == "as5100_fixed_rail":
            model = self.current_load_source_metadata.get("model", "AS5100")
            axle_count = self.current_load_source_metadata.get("axle_count", len(self.current_loads))
            max_axle = self.current_load_source_metadata.get(
                "max_axle_load_n",
                max(abs(load.load_newtons) for load in self.current_loads),
            )
            max_wheel = self.current_load_source_metadata.get(
                "max_wheel_load_n_per_rail",
                max(abs(load.load_newtons) for load in self.current_loads),
            )
            return (
                f"AS5100 {model}: {axle_count} axles, "
                f"max axle {n_to_kn(float(max_axle)):.2f} kN -> "
                f"wheel {n_to_kn(float(max_wheel)):.2f} kN/rail"
            )
        if source_type == "train_builder":
            axle_count = self.current_load_source_metadata.get("axle_count", len(self.current_loads))
            max_axle = self.current_load_source_metadata.get(
                "max_axle_load_n",
                max(abs(load.load_newtons) for load in self.current_loads),
            )
            max_wheel = self.current_load_source_metadata.get(
                "max_wheel_load_n_per_rail",
                max(abs(load.load_newtons) for load in self.current_loads),
            )
            return (
                f"Train builder: {axle_count} axles, "
                f"max axle {n_to_kn(float(max_axle)):.2f} kN -> "
                f"wheel {n_to_kn(float(max_wheel)):.2f} kN/rail"
            )
        if len(self.current_loads) == 1:
            load = self.current_loads[0]
            return f"Current UI load: {n_to_kn(load.load_newtons):.2f} kN @ x={load.position_m:.3f} m"
        max_load = max(abs(load.load_newtons) for load in self.current_loads)
        return f"Current UI loads: {len(self.current_loads)} loads, max {n_to_kn(max_load):.2f} kN"

    def _loads_for_context(self) -> list[PointLoad] | None:
        if not self.use_current_load_selection_checkbox.isChecked():
            return None
        return list(self.current_loads)

    def _apply_current_loads_to_context(
        self,
        base_config: AnalysisConfig,
        transition_context: SensitivityTransitionContext | None,
    ) -> tuple[AnalysisConfig, SensitivityTransitionContext | None]:
        loads = self._loads_for_context()
        if loads is None:
            return base_config, transition_context
        updated_config = replace(base_config, loads=loads)
        if transition_context is None:
            return updated_config, None
        updated_transition_config = replace(transition_context.analysis_config, loads=loads)
        return updated_config, replace(
            transition_context,
            analysis_config=updated_transition_config,
        )

    def _active_load_source_metadata(self) -> dict[str, object]:
        if self.use_current_load_selection_checkbox.isChecked():
            return dict(self.current_load_source_metadata)
        return {"source_type": "sensitivity_baseline_load"}

    def _as5100_position_sensitivity_available(self) -> bool:
        return (
            self.use_current_load_selection_checkbox.isChecked()
            and self.current_load_source_metadata.get("source_type") == "as5100_fixed_rail"
        )

    def _scenario_change_label(
        self,
        parameter: str,
        factor: float,
        parameter_value: float | None = None,
    ) -> str:
        if (
            parameter == SensitivityVariable.AS5100_POSITION.value
            and self._active_load_source_metadata().get("source_type") == "as5100_fixed_rail"
        ):
            offset_m = 0.0 if parameter_value is None else parameter_value
            return f"AS5100 train shifted {offset_m:+.1f} m"
        if (
            parameter == SensitivityVariable.WHEEL_LOAD.value
            and self._active_load_source_metadata().get("source_type") == "as5100_fixed_rail"
        ):
            change = (factor - 1.0) * 100.0
            sign = "+" if change > 0 else ""
            return f"AS5100 load scale {sign}{change:.0f}%"
        return scenario_change_label(parameter, factor)

    def _selected_variables(self) -> list[SensitivityVariable]:
        return [
            variable
            for variable, check in self.variable_checks.items()
            if check.isEnabled() and check.isChecked()
        ]

    def _selected_mode(self) -> SensitivityRunMode:
        if self.transition_mode_radio.isChecked():
            return SensitivityRunMode.TRANSITION
        return SensitivityRunMode.STATIC

    def _criteria(self) -> DesignCriteria:
        return DesignCriteria(
            max_deflection_m=self.criteria_inputs["max_deflection_mm"].value() / 1000.0,
            rail_stress_pa=self.criteria_inputs["rail_stress_mpa"].value() * 1_000_000.0,
            max_sleeper_load_n=self.criteria_inputs["max_sleeper_load_kn"].value() * 1000.0,
            ballast_pressure_pa=self.criteria_inputs["ballast_pressure_kpa"].value() * 1000.0,
            formation_pressure_pa=self.criteria_inputs["formation_pressure_kpa"].value() * 1000.0,
            subgrade_pressure_pa=self.criteria_inputs["subgrade_pressure_kpa"].value() * 1000.0,
            deep_subgrade_pressure_pa=self.criteria_inputs["deep_subgrade_pressure_kpa"].value() * 1000.0,
            transition_metric_m=self.criteria_inputs["transition_metric_mm"].value() / 1000.0,
        )

    def _database_scenarios(
        self,
        *,
        base_config: AnalysisConfig,
        transition_context: SensitivityTransitionContext | None,
    ) -> list[SensitivityScenario]:
        config = self.track_config
        if config is None:
            return []
        preset = str(self.preset_combo.currentData() or "custom")
        scenarios: list[SensitivityScenario] = []
        if preset in {"formation", "transition", "cost"}:
            scenarios.extend(self._support_profile_scenarios(base_config, transition_context))
            scenarios.extend(self._pad_scenarios(base_config, transition_context))
        if preset in {"rail", "cost"}:
            scenarios.extend(self._rail_scenarios(base_config, transition_context))
        if preset == "cost":
            scenarios.extend(self._sleeper_scenarios(base_config, transition_context))
        return scenarios

    def _scenario_with_context(
        self,
        *,
        name: str,
        changed_parameter: str,
        parameter_value: float,
        factor: float,
        config: AnalysisConfig,
        transition_context: SensitivityTransitionContext | None,
        apply_payload: dict[str, object],
    ) -> SensitivityScenario:
        context = None
        if transition_context is not None:
            context = replace(transition_context, analysis_config=config)
            if changed_parameter == "support_profile":
                context = replace(
                    context,
                    k1_n_per_m2=config.foundation_modulus_n_per_m2,
                    k_profile_n_per_m2=None,
                )
        return SensitivityScenario(
            name=name,
            changed_parameter=changed_parameter,
            parameter_value=parameter_value,
            factor=factor,
            analysis_config=config,
            transition_context=context,
            apply_payload=apply_payload,
        )

    def _support_profile_scenarios(
        self,
        base_config: AnalysisConfig,
        transition_context: SensitivityTransitionContext | None,
    ) -> list[SensitivityScenario]:
        current = self.track_config
        if current is None:
            return []
        profiles = self.session.scalars(
            select(SupportProfile).order_by(SupportProfile.foundation_modulus_n_per_m2)
        ).all()
        scenarios = []
        for profile in profiles:
            if profile.id == current.support_profile_id:
                continue
            value = profile.foundation_modulus_n_per_m2
            config = replace(base_config, foundation_modulus_n_per_m2=value)
            scenarios.append(
                self._scenario_with_context(
                    name=f"support profile {profile.name}",
                    changed_parameter="support_profile",
                    parameter_value=value,
                    factor=value / base_config.foundation_modulus_n_per_m2,
                    config=config,
                    transition_context=transition_context,
                    apply_payload={"support_profile_id": profile.id},
                )
            )
        return scenarios

    def _pad_scenarios(
        self,
        base_config: AnalysisConfig,
        transition_context: SensitivityTransitionContext | None,
    ) -> list[SensitivityScenario]:
        current = self.track_config
        if current is None:
            return []
        pads = self.session.scalars(select(Pad).order_by(Pad.stiffness_newtons_per_meter)).all()
        scenarios = []
        for pad in pads:
            if pad.id == current.pad_id:
                continue
            value = pad.stiffness_newtons_per_meter
            baseline = base_config.pad_stiffness_n_per_m or base_config.railpad_stiffness_n_per_m or value
            config = replace(base_config, pad_stiffness_n_per_m=value, railpad_stiffness_n_per_m=value)
            scenarios.append(
                self._scenario_with_context(
                    name=f"pad {pad.name}",
                    changed_parameter="pad",
                    parameter_value=value,
                    factor=value / baseline,
                    config=config,
                    transition_context=transition_context,
                    apply_payload={"pad_id": pad.id},
                )
            )
        return scenarios

    def _rail_scenarios(
        self,
        base_config: AnalysisConfig,
        transition_context: SensitivityTransitionContext | None,
    ) -> list[SensitivityScenario]:
        current = self.track_config
        if current is None or current.rail is None:
            return []
        rails = [
            rail
            for rail in self.session.scalars(select(Rail).order_by(Rail.mass_kg_per_m)).all()
            if rail.id != current.rail_id
        ]
        rails = sorted(
            rails,
            key=lambda rail: abs(rail.section_modulus_m3 - current.rail.section_modulus_m3),
        )[:4]
        scenarios = []
        for rail in rails:
            area_m2 = cm2_to_m2(rail.area_cm2) if rail.area_cm2 is not None else base_config.area_m2
            config = replace(
                base_config,
                elastic_modulus_pa=rail.elastic_modulus_pa,
                moment_inertia_m4=rail.moment_inertia_m4,
                section_modulus_m3=rail.section_modulus_m3,
                section_modulus_head_m3=rail.section_modulus_head_m3,
                section_modulus_foot_m3=rail.section_modulus_foot_m3,
                area_m2=area_m2,
                rail_area_m2=area_m2,
            )
            scenarios.append(
                self._scenario_with_context(
                    name=f"rail {rail.name}",
                    changed_parameter="rail",
                    parameter_value=rail.section_modulus_m3,
                    factor=rail.section_modulus_m3 / current.rail.section_modulus_m3,
                    config=config,
                    transition_context=transition_context,
                    apply_payload={"rail_id": rail.id},
                )
            )
        return scenarios

    def _sleeper_scenarios(
        self,
        base_config: AnalysisConfig,
        transition_context: SensitivityTransitionContext | None,
    ) -> list[SensitivityScenario]:
        current = self.track_config
        if current is None or current.sleeper is None:
            return []
        sleepers = [
            sleeper
            for sleeper in self.session.scalars(select(Sleeper).order_by(Sleeper.length_m)).all()
            if sleeper.id != current.sleeper_id
        ]
        scenarios = []
        for sleeper in sleepers:
            config = replace(
                base_config,
                sleeper_length_m=sleeper.length_m,
                sleeper_width_m=sleeper.width_m,
            )
            scenarios.append(
                self._scenario_with_context(
                    name=f"sleeper {sleeper.name}",
                    changed_parameter="sleeper",
                    parameter_value=sleeper.length_m,
                    factor=sleeper.length_m / current.sleeper.length_m,
                    config=config,
                    transition_context=transition_context,
                    apply_payload={"sleeper_id": sleeper.id},
                )
            )
        return scenarios

    def _run(self) -> None:
        if self.worker_thread is not None:
            QMessageBox.information(self, "Sensitivity running", "A sensitivity run is already active.")
            return
        config = self.track_config
        if config is None:
            QMessageBox.warning(self, "Missing config", "Select a track config before running sensitivity.")
            return
        variables = self._selected_variables()
        if not variables:
            QMessageBox.warning(self, "No variables", "Select at least one sensitivity variable.")
            return
        load_n = self._baseline_load_n(config)
        try:
            base_config = self.build_static_context(config, load_n)
            transition_context = None
            mode = self._selected_mode()
            if mode == SensitivityRunMode.TRANSITION:
                transition_context = self.build_transition_context(config, load_n)
                if (
                    SensitivityVariable.TRANSITION_LENGTH in variables
                    and transition_context.transition_length_m is None
                ):
                    QMessageBox.warning(
                        self,
                        "Transition length unavailable",
                        "Transition length sensitivity requires a ramp or exponential transition profile with a valid Lt.",
                    )
                    return
            base_config, transition_context = self._apply_current_loads_to_context(
                base_config,
                transition_context,
            )
            additional_scenarios = self._database_scenarios(
                base_config=base_config,
                transition_context=transition_context,
            )
            criteria = self._criteria()
            scenario_count = len(
                build_scenarios(
                    base_config=base_config,
                    variables=variables,
                    mode=mode,
                    transition_context=transition_context,
                    additional_scenarios=additional_scenarios,
                )
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Validation error", str(exc))
            return

        self.progress.setRange(0, scenario_count)
        self.progress.setValue(0)
        self.status_label.setText(f"Scenario 0 of {scenario_count}")
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.save_alternative_button.setEnabled(False)
        self.result = None
        self._result_stale = False
        self.open_large_plot_button.setEnabled(False)

        self.worker_thread = QThread(self)
        self.worker = SensitivityWorker(
            base_config=base_config,
            variables=variables,
            mode=mode,
            transition_context=transition_context,
            criteria=criteria,
            additional_scenarios=additional_scenarios,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._handle_progress)
        self.worker.finished.connect(self._handle_result)
        self.worker.failed.connect(self._handle_failed)
        self.worker.cancelled.connect(self._handle_cancelled)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker.cancelled.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.worker.cancelled.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.status_label.setText("Cancelling...")
            self.cancel_button.setEnabled(False)

    def _handle_progress(self, index: int, total: int, name: str) -> None:
        self.progress.setValue(index)
        self.status_label.setText(f"Scenario {index} of {total}: {name}")

    def _handle_result(self, result: object) -> None:
        self.result = result if isinstance(result, SensitivityRunResult) else None
        self._result_stale = False
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._set_result_actions_enabled(self.result is not None)
        self.status_label.setText("Sensitivity complete.")
        if self.result is not None:
            self._populate_results(self.result)

    def _handle_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Sensitivity failed.")
        self._set_result_actions_enabled(False)
        QMessageBox.warning(self, "Sensitivity error", message)

    def _handle_cancelled(self) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Sensitivity cancelled.")
        self._set_result_actions_enabled(False)

    def _clear_worker(self) -> None:
        self.worker = None
        self.worker_thread = None

    def _populate_results(self, result: SensitivityRunResult) -> None:
        result = self._with_as5100_display_labels(result)
        self.result = result
        self.open_large_plot_button.setEnabled(True)
        self.table.setRowCount(len(result.scenarios))
        for row, scenario in enumerate(result.scenarios):
            label = self._scenario_change_label(
                scenario.changed_parameter,
                scenario.factor,
                scenario.parameter_value,
            )
            values = [
                label,
                scenario.changed_parameter,
                self._format_parameter(scenario.changed_parameter, scenario.parameter_value),
                self._format_metric(scenario.metrics.max_deflection_m, 1000.0),
                self._format_metric(scenario.metrics.max_moment_nm, 0.001),
                self._format_metric(scenario.metrics.max_sleeper_load_n, 0.001),
                self._format_metric(scenario.metrics.rail_stress_pa, 1.0 / 1_000_000.0),
                self._format_metric(scenario.metrics.ballast_pressure_pa, 1.0 / 1000.0),
                self._format_metric(scenario.metrics.formation_pressure_pa, 1.0 / 1000.0),
                self._format_metric(scenario.metrics.subgrade_pressure_pa, 1.0 / 1000.0),
                self._format_metric(scenario.metrics.deep_subgrade_pressure_pa, 1.0 / 1000.0),
                self._format_metric(scenario.metrics.transition_metric_m, 1000.0),
                self._format_metric(scenario.score, 1.0),
                self._format_metric(scenario.constructability_score, 1.0),
                self._format_metric(scenario.combined_score, 1.0),
                self._format_metric(scenario.percent_improvement, 1.0),
                self._format_metric(scenario.decision.max_utilization, 1.0),
                scenario.decision.governing_criterion or "-",
                scenario.decision.status,
                scenario.warning or "ok",
                "" if scenario.rank is None else str(scenario.rank),
            ]
            state = scenario_visual_state(scenario)
            color = scenario_row_color(state)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, row)
                if column in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20}:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item.setBackground(color)
                item.setForeground(COLOR_TEXT)
                if state in {"warning", "worse"}:
                    item.setToolTip(scenario.warning or "Scenario is worse than the baseline score.")
                elif state == "best":
                    item.setToolTip("Best valid scenario by decision status, utilization, and combined score.")
                elif state == "baseline":
                    item.setToolTip("Baseline scenario.")
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self._apply_result_filter()
        self.recommendation_label.setText(
            "\n".join(
                [
                    f"Best option: {result.recommendation.best_option}",
                    f"Most sensitive parameter: {result.recommendation.most_sensitive_parameter}",
                    f"Worst option: {result.recommendation.worst_option}",
                    f"Recommended next design adjustment: {result.recommendation.next_design_adjustment}",
                    f"Key warning / limitation: {result.recommendation.key_warning}",
                ]
            )
        )
        self._draw_plots(result)

    def _with_as5100_display_labels(self, result: SensitivityRunResult) -> SensitivityRunResult:
        if self._active_load_source_metadata().get("source_type") != "as5100_fixed_rail":
            return result
        scenarios = [
            replace(
                scenario,
                scenario_name=self._scenario_change_label(
                    scenario.changed_parameter,
                    scenario.factor,
                    scenario.parameter_value,
                ),
            )
            if scenario.changed_parameter
            in {
                SensitivityVariable.WHEEL_LOAD.value,
                SensitivityVariable.AS5100_POSITION.value,
            }
            else scenario
            for scenario in result.scenarios
        ]
        baseline = scenarios[0] if scenarios else result.baseline
        replacements = {
            "wheel load": "AS5100 load scale",
            "as5100 position": "AS5100 train position",
        }

        def replace_as5100_terms(text: str) -> str:
            for source, target in replacements.items():
                text = text.replace(source, target)
            return text

        recommendation = replace(
            result.recommendation,
            best_option=replace_as5100_terms(result.recommendation.best_option),
            worst_option=replace_as5100_terms(result.recommendation.worst_option),
            most_sensitive_parameter=replace_as5100_terms(result.recommendation.most_sensitive_parameter),
            next_design_adjustment=replace_as5100_terms(result.recommendation.next_design_adjustment),
        )
        return replace(result, baseline=baseline, scenarios=scenarios, recommendation=recommendation)

    def _set_result_actions_enabled(self, enabled: bool) -> None:
        enabled = enabled and not self._result_stale
        self.export_button.setEnabled(enabled)
        self.save_alternative_button.setEnabled(enabled)
        self.apply_best_button.setEnabled(enabled and self._best_applicable_scenario() is not None)
        if hasattr(self, "open_large_plot_button"):
            self.open_large_plot_button.setEnabled(self.result is not None)

    def _mark_results_stale(self, *_: object) -> None:
        if self.result is None or self.worker_thread is not None:
            return
        self._result_stale = True
        self.status_label.setText("Inputs changed. Run sensitivity again to refresh these results.")
        self._set_result_actions_enabled(False)

    def _handle_criteria_changed(self, *_: object) -> None:
        if self.result is None or self.worker_thread is not None:
            return
        was_stale = self._result_stale
        self.result = rescore_sensitivity_result(self.result, criteria=self._criteria())
        self._result_stale = was_stale
        if was_stale:
            self.status_label.setText("Criteria updated, but inputs changed. Run sensitivity again.")
        else:
            self.status_label.setText("Criteria updated. Results re-scored without rerunning analysis.")
        self._populate_results(self.result)
        self._set_result_actions_enabled(not was_stale)

    def _apply_result_filter(self) -> None:
        if self.result is None:
            return
        mode = self.result_filter_combo.currentData()
        valid_scenarios = [
            item
            for item in self.result.scenarios
            if item.changed_parameter != "baseline" and item.rank is not None and not item.warning
        ]
        best_name = min(valid_scenarios, key=lambda item: item.rank or 999999).scenario_name if valid_scenarios else None
        worst_name = max(valid_scenarios, key=lambda item: item.rank or -1).scenario_name if valid_scenarios else None
        for row, scenario in enumerate(self.result.scenarios):
            show = True
            if mode == "valid":
                show = scenario.score is not None and not scenario.warning
            elif mode == "warnings":
                show = bool(scenario.warning)
            elif mode == "best_worst":
                show = scenario.changed_parameter == "baseline" or scenario.scenario_name in {best_name, worst_name}
            self.table.setRowHidden(row, not show)

    def _draw_plots(self, result: SensitivityRunResult) -> None:
        self._draw_improvement(result)
        self._draw_tornado(result)
        self._draw_comparison(result)

    @staticmethod
    def _style_axes(axes) -> None:
        axes.grid(True, axis="y", color="#e2e8f0", linewidth=0.8)
        axes.set_axisbelow(True)
        for spine in axes.spines.values():
            spine.set_color("#cbd5e0")
            spine.set_linewidth(0.8)

    @staticmethod
    def _wrapped_labels(labels: Sequence[str], width: int = 18) -> list[str]:
        return ["\n".join(textwrap.wrap(label, width=width)) or label for label in labels]

    def _draw_improvement(self, result: SensitivityRunResult, canvas: FigureCanvas | None = None) -> None:
        canvas = canvas or self.plots.widget(0)
        figure = canvas.figure
        figure.clear()
        axes = figure.add_subplot(111)
        scenarios = [item for item in result.scenarios if item.scenario_name != "Baseline"]
        labels = [
            self._scenario_change_label(item.changed_parameter, item.factor, item.parameter_value)
            for item in scenarios
        ]
        values = [item.percent_improvement or 0.0 for item in scenarios]
        colors = [
            CHART_RED if item.decision.status == "fail" else CHART_AMBER if item.warning or item.decision.status == "warning" else (CHART_GREEN if value >= 0.0 else CHART_RED)
            for item, value in zip(scenarios, values)
        ]
        edges = [CHART_DARK if item.rank == 1 else "none" for item in scenarios]
        axes.bar(range(len(values)), values, color=colors, edgecolor=edges, linewidth=1.2)
        axes.axhline(0.0, color=CHART_DARK, linewidth=0.9)
        axes.set_ylabel("Improvement (%)")
        axes.set_title("Sensitivity improvement by scenario")
        axes.set_xticks(range(len(labels)))
        axes.set_xticklabels(self._wrapped_labels(labels), rotation=45, ha="right", fontsize=8)
        self._style_axes(axes)
        canvas.draw_idle()

    def _draw_tornado(self, result: SensitivityRunResult, canvas: FigureCanvas | None = None) -> None:
        canvas = canvas or self.plots.widget(1)
        figure = canvas.figure
        figure.clear()
        axes = figure.add_subplot(111)
        spans: dict[str, float] = {}
        for scenario in result.scenarios:
            if scenario.changed_parameter == "baseline" or scenario.percent_improvement is None:
                continue
            spans[scenario.changed_parameter] = max(
                spans.get(scenario.changed_parameter, 0.0),
                abs(scenario.percent_improvement),
            )
        labels = list(spans)
        values = [spans[label] for label in labels]
        colors = [CHART_BLUE if index % 2 == 0 else CHART_GREEN for index, _ in enumerate(values)]
        axes.barh(range(len(values)), values, color=colors)
        axes.set_yticks(range(len(labels)))
        axes.set_yticklabels([label.replace("_", " ") for label in labels])
        axes.set_xlabel("Maximum absolute score movement (%)")
        axes.set_title("Most influential parameters")
        self._style_axes(axes)
        canvas.draw_idle()

    def _draw_comparison(self, result: SensitivityRunResult, canvas: FigureCanvas | None = None) -> None:
        canvas = canvas or self.plots.widget(2)
        figure = canvas.figure
        figure.clear()
        axes = figure.add_subplot(111)
        valid = [
            item
            for item in result.scenarios
            if item.scenario_name != "Baseline" and item.rank is not None and not item.warning
        ]
        if not valid:
            axes.text(0.5, 0.5, "No valid best scenario", ha="center", va="center")
            canvas.draw_idle()
            return
        best = min(valid, key=lambda item: item.rank or 999999)
        baseline = result.baseline
        labels = [
            "Deflection",
            "Moment",
            "Sleeper",
            "Ballast",
            "Formation",
            "Subgrade",
            "Deep subgrade",
            "Transition",
        ]
        base_values = [
            baseline.metrics.max_deflection_m,
            baseline.metrics.max_moment_nm,
            baseline.metrics.max_sleeper_load_n,
            baseline.metrics.ballast_pressure_pa,
            baseline.metrics.formation_pressure_pa,
            baseline.metrics.subgrade_pressure_pa,
            baseline.metrics.deep_subgrade_pressure_pa,
            baseline.metrics.transition_metric_m,
        ]
        best_values = [
            best.metrics.max_deflection_m,
            best.metrics.max_moment_nm,
            best.metrics.max_sleeper_load_n,
            best.metrics.ballast_pressure_pa,
            best.metrics.formation_pressure_pa,
            best.metrics.subgrade_pressure_pa,
            best.metrics.deep_subgrade_pressure_pa,
            best.metrics.transition_metric_m,
        ]
        available_labels: list[str] = []
        ratios: list[float] = []
        for label, base, value in zip(labels, base_values, best_values):
            if base and value is not None:
                available_labels.append(label)
                ratios.append(value / base)
        if not available_labels:
            axes.text(0.5, 0.5, "No comparable metrics", ha="center", va="center")
            canvas.draw_idle()
            return
        axes.bar(
            [x - 0.18 for x in range(len(available_labels))],
            [1.0] * len(available_labels),
            width=0.36,
            label="Baseline",
            color=CHART_GREY,
        )
        axes.bar(
            [x + 0.18 for x in range(len(available_labels))],
            ratios,
            width=0.36,
            label="Best",
            color=[CHART_GREEN if value <= 1.0 else CHART_RED for value in ratios],
        )
        axes.axhline(1.0, color=CHART_DARK, linewidth=0.8)
        axes.set_xticks(range(len(available_labels)))
        axes.set_xticklabels(available_labels)
        axes.set_ylabel("Ratio to baseline")
        axes.set_title("Baseline vs best scenario")
        axes.legend()
        self._style_axes(axes)
        canvas.draw_idle()

    def _open_large_plot(self) -> None:
        if self.result is None:
            return
        index = self.plots.currentIndex()
        title = self.plots.tabText(index) or "Sensitivity plot"
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(1000, 720)
        layout = QVBoxLayout(dialog)
        canvas = FigureCanvas(Figure(figsize=(10, 6), tight_layout=True))
        layout.addWidget(canvas)
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, alignment=Qt.AlignRight)
        if index == 0:
            self._draw_improvement(self.result, canvas)
        elif index == 1:
            self._draw_tornado(self.result, canvas)
        else:
            self._draw_comparison(self.result, canvas)
        dialog.exec()

    def _export(self) -> None:
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export sensitivity CSV",
            "sensitivity_results.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            write_sensitivity_csv(path, self.result)
            write_export_metadata(
                path,
                solver_mode=f"sensitivity_{self.result.mode.value}",
                inputs_payload={
                    "project_id": self.project.id if self.project else None,
                    "track_config_id": self.track_config.id if self.track_config else None,
                    "result": result_payload(self.result),
                    "track_config": self._config_snapshot(),
                    "load_source": self._active_load_source_metadata(),
                },
                units="SI",
                parameter_trace=asdict(self.result.recommendation),
            )
        except OSError:
            QMessageBox.critical(self, "Export error", "Unable to write the CSV file.")

    def _save_selected_alternatives(self) -> None:
        if self.result is None or self.track_config is None:
            return
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        if not rows:
            QMessageBox.warning(
                self,
                "Design Alternative",
                "Select one or more sensitivity scenarios first.",
            )
            return
        load_case = self._latest_load_case(self.track_config)
        saved = 0
        analysis_type = "transition" if self.result.mode == SensitivityRunMode.TRANSITION else "static"
        for row in rows:
            item = self.table.item(row, 0)
            scenario_index = item.data(Qt.UserRole) if item is not None else row
            scenario = self.result.scenarios[int(scenario_index)]
            if scenario.changed_parameter == "baseline":
                continue
            status = "warning" if scenario.warning else _alternative_status_from_decision(scenario.decision.status)
            name = (
                f"{self._scenario_change_label(scenario.changed_parameter, scenario.factor, scenario.parameter_value)}"
                f" - {status}"
            )
            crud.create_design_alternative(
                self.session,
                project_id=self.track_config.project_id,
                track_config_id=self.track_config.id,
                load_case_id=load_case.id if load_case is not None else None,
                name=name,
                description=scenario.warning or "Saved from sensitivity screening.",
                source_type="sensitivity",
                analysis_type=analysis_type,
                changed_parameters={
                    "parameter": scenario.changed_parameter,
                    "factor": scenario.factor,
                    "parameter_value_si": scenario.parameter_value,
                },
                input_snapshot={
                    "track_config": self._config_snapshot(),
                    "scenario_name": scenario.scenario_name,
                    "recommendation": asdict(self.result.recommendation),
                    "criteria": asdict(self.result.criteria) if self.result.criteria else None,
                    "load_source": self._active_load_source_metadata(),
                },
                metrics={
                    **asdict(scenario.metrics),
                    "governing_criterion": scenario.decision.governing_criterion,
                    "max_utilization": scenario.decision.max_utilization,
                    "decision_status": scenario.decision.status,
                    "constructability_score": scenario.constructability_score,
                    "combined_score": scenario.combined_score,
                },
                status=status,
                score=scenario.combined_score or scenario.score,
            )
            saved += 1
        if saved == 0:
            QMessageBox.warning(
                self,
                "Design Alternative",
                "Select at least one non-baseline scenario to save.",
            )
            return
        self.session.commit()
        self.alternatives_saved.emit()
        QMessageBox.information(
            self,
            "Design Alternative",
            f"Saved {saved} design alternative{'s' if saved != 1 else ''}.",
        )

    def _best_applicable_scenario(self):
        if self.result is None:
            return None
        candidates = [
            scenario
            for scenario in self.result.scenarios
            if scenario.changed_parameter != "baseline"
            and scenario.apply_payload
            and scenario.score is not None
            and not scenario.warning
            and scenario.decision.status != "fail"
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda scenario: (
                {"pass": 0, "warning": 1, "fail": 2, "unrated": 3}.get(scenario.decision.status, 3),
                scenario.decision.max_utilization if scenario.decision.max_utilization is not None else 999.0,
                scenario.combined_score if scenario.combined_score is not None else scenario.score or 999.0,
            ),
        )

    def _apply_best_as_track_config(self) -> None:
        scenario = self._best_applicable_scenario()
        config = self.track_config
        if scenario is None or config is None or not scenario.apply_payload:
            QMessageBox.warning(
                self,
                "Apply design option",
                "No applicable non-baseline design scenario is available.",
            )
            return
        payload = dict(scenario.apply_payload)
        name = self._unique_track_config_name(f"Design option - {scenario.scenario_name}")
        new_config = crud.create_track_config(
            self.session,
            name=name,
            project_id=config.project_id,
            rail_id=int(payload.get("rail_id", config.rail_id)),
            sleeper_id=int(payload.get("sleeper_id", config.sleeper_id)),
            pad_id=int(payload.get("pad_id", config.pad_id)),
            support_profile_id=int(payload.get("support_profile_id", config.support_profile_id)),
            sleeper_spacing_m=float(payload.get("sleeper_spacing_m", config.sleeper_spacing_m)),
            gauge_m=config.gauge_m,
        )
        load_case = self._latest_load_case(config)
        if self.result is not None:
            crud.create_design_alternative(
                self.session,
                project_id=new_config.project_id,
                track_config_id=new_config.id,
                load_case_id=load_case.id if load_case is not None else None,
                name=f"{scenario.scenario_name} - applied",
                description="Created from sensitivity/design screening. Original config was not modified.",
                source_type="sensitivity",
                analysis_type="transition" if self.result.mode == SensitivityRunMode.TRANSITION else "static",
                changed_parameters={
                    "parameter": scenario.changed_parameter,
                    "factor": scenario.factor,
                    "parameter_value_si": scenario.parameter_value,
                    "apply_payload": payload,
                },
                input_snapshot={
                    "source_track_config": self._config_snapshot(),
                    "criteria": asdict(self.result.criteria) if self.result.criteria else None,
                    "recommendation": asdict(self.result.recommendation),
                    "load_source": self._active_load_source_metadata(),
                },
                metrics={
                    **asdict(scenario.metrics),
                    "governing_criterion": scenario.decision.governing_criterion,
                    "max_utilization": scenario.decision.max_utilization,
                    "decision_status": scenario.decision.status,
                    "constructability_score": scenario.constructability_score,
                    "combined_score": scenario.combined_score,
                },
                status=_alternative_status_from_decision(scenario.decision.status),
                score=scenario.combined_score or scenario.score,
            )
        self.alternatives_saved.emit()
        QMessageBox.information(
            self,
            "Apply design option",
            f"Created new track config '{new_config.name}'. The original config was not changed.",
        )

    def _unique_track_config_name(self, base_name: str) -> str:
        project_id = self.track_config.project_id if self.track_config is not None else None
        existing = {
            name
            for name in self.session.scalars(
                select(TrackConfig.name).where(TrackConfig.project_id == project_id)
            )
        }
        candidate = base_name[:110]
        if candidate not in existing:
            return candidate
        index = 2
        while True:
            suffix = f" ({index})"
            candidate = f"{base_name[:110 - len(suffix)]}{suffix}"
            if candidate not in existing:
                return candidate
            index += 1

    def _config_snapshot(self) -> dict[str, object]:
        config = self.track_config
        if config is None:
            return {}
        return {
            "project": config.project.name if config.project else None,
            "track_config": config.name,
            "rail": config.rail.name if config.rail else None,
            "sleeper": config.sleeper.name if config.sleeper else None,
            "pad": config.pad.name if config.pad else None,
            "support": config.support_profile.name if config.support_profile else None,
            "sleeper_spacing_m": config.sleeper_spacing_m,
            "gauge_m": config.gauge_m,
        }

    @staticmethod
    def _format_metric(value: float | None, factor: float) -> str:
        if value is None:
            return "-"
        return f"{value * factor:.4g}"

    @staticmethod
    def _format_parameter(parameter: str, value: float | None) -> str:
        if value is None:
            return "-"
        if parameter == SensitivityVariable.SUPPORT_STIFFNESS.value:
            return f"{n_per_m2_to_mn_per_m2(value):.3g} MN/m²"
        if parameter == SensitivityVariable.SLEEPER_SPACING.value:
            return f"{m_to_mm(value):.1f} mm"
        if parameter == SensitivityVariable.PAD_STIFFNESS.value:
            return f"{n_to_kn(value):.3g} kN/m"
        if parameter == SensitivityVariable.WHEEL_LOAD.value:
            return f"{n_to_kn(value):.3g} kN"
        if parameter == SensitivityVariable.AS5100_POSITION.value:
            return f"{value:+.3g} m"
        if parameter == SensitivityVariable.SPEED.value:
            return f"{value:.1f} km/h"
        if parameter in {SensitivityVariable.BALLAST_DEPTH.value, SensitivityVariable.TRANSITION_LENGTH.value}:
            return f"{value:.3g} m"
        return f"{value:.4g}"


def _alternative_status_from_decision(status: str) -> str:
    if status == "pass":
        return "ok"
    if status in {"warning", "fail"}:
        return status
    return "draft"

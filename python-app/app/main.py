"""PySide6 GUI entry point for the BOEF desktop application."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import io
import math
import json
import logging
import os
from pathlib import Path
import sys
import warnings
from typing import Callable, Literal, Sequence

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, QLibraryInfo, QTimer
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QTextBrowser,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTableWidget,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QSizePolicy,
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from core.analysis import (
    AnalysisInputs,
    AnalysisResult,
    DesignInputs,
    build_load_domain,
    subgrade_pressure_a3902,
)
from core.analysis_engine import (
    AnalysisConfig,
    AnalysisMode,
    BeamTheory,
    FoundationModelType,
    FoundationProfileType,
    _build_grid,
    run_analysis,
)
from core.envelope import (
    AS5100EnvelopeSweep,
    EnvelopeCancelled,
    EnvelopeConfig,
    EnvelopeResult,
    run_envelope,
)
from core.transition import (
    TransitionProfileType,
    TransitionRunMode,
    TransitionRunResult,
    build_series_from_envelope,
    build_series_from_single,
    build_transition_profile,
    compute_energy_from_envelope,
    compute_energy_from_series,
    compute_metrics_from_envelope,
    compute_metrics_from_series,
)
from core.dynamic.config import (
    DippedJointConfig,
    DynamicBoundaryMode,
    DynamicConfig,
    DynamicExcitationMode,
    DynamicMode,
    DynamicTransitionConfig,
    DynamicTransitionProfileType,
    DynamicTransitionRunMode,
    IrregularityInput,
    IrregularityMode,
)
from core.dynamic.engine import run_dynamic_analysis
from core.dynamic.results import DippedJointResult, DynamicResult, DynamicTransitionResult
from core.foundation.base import DampingModel, equivalent_series_stiffness, per_support_to_per_length
from core.load_builder import (
    AS5100_MODEL_150LA,
    AS5100_MODEL_300LA,
    AS5100RailLoadConfig,
    as5100_load_metadata,
    axle_load_to_wheel_load,
    build_as5100_rail_loads,
    TrainLoadConfig,
    build_train_loads,
)
from core.model import PointLoad, beam_parameter_beta
from core.special.config import FloatingSlabConfig, SpecialMode
from core.special.engine import run_special_analysis
from core.special.results import FloatingSlabResult
from core.stress_metrics import (
    BearingGeometry,
    StressResults,
    build_rail_only_stress_results,
    build_rail_only_stress_results_from_envelope,
    build_stress_results_from_envelope,
    build_stress_results_from_single,
    get_bearing_geometry,
)
from core.units import (
    kn_to_n,
    cm2_to_m2,
    cm3_to_m3,
    cm4_to_m4,
    m3_to_cm3,
    m4_to_cm4,
    m3_to_mm3,
    m4_to_mm4,
    mm3_to_m3,
    mm4_to_m4,
    mm_to_m,
    m_to_mm,
    mpa_to_pa,
    mn_per_m2_to_n_per_m2,
    n_to_kn,
    n_per_m2_to_mn_per_m2,
    pa_to_kpa,
    pa_to_mpa,
)
from db import crud
from db.migration import run_migrations
from db.models import (
    DesignAlternative,
    LoadCase,
    Pad,
    Project,
    Rail,
    RailAdmissibleStress,
    Result,
    Sleeper,
    SupportProfile,
    TrackConfig,
)
from db.seed import seed_database
from app.export_helpers import (
    compute_inputs_hash,
    write_analysis_csv_from_result,
    write_export_metadata,
    write_envelope_analysis_csv,
    write_envelope_sleeper_csv,
    write_dynamic_fft_csv,
    write_dynamic_psd_csv,
    write_dynamic_transition_metrics_csv,
    write_dynamic_transition_series_csv,
    write_dynamic_time_history_csv,
    write_dipped_joint_csv,
    write_sleeper_csv_from_result,
    write_transition_metrics_csv,
    write_transition_series_csv,
)
from app.design_alternatives_dialog import AlternativeComparisonDialog
from app.help_content import build_dynamic_chart_help_markdown, build_help_markdown
from app.sensitivity_dialog import SensitivityDialog
from core.sensitivity import SensitivityTransitionContext
from app.custom_chart import (
    ChartAxisFamily,
    CustomChartDialog,
    CustomChartSelection,
    RenderedSeries,
    axis_label_with_unit,
    build_resampled_series,
    custom_chart_color,
)

LOGGER = logging.getLogger(__name__)


def _is_deleted_qt_object_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "already deleted" in message or "internal c++ object" in message


def _install_safe_canvas_idle_draw() -> None:
    if getattr(FigureCanvas, "_boef_safe_idle_draw_installed", False):
        return
    original_draw_idle = getattr(FigureCanvas, "_draw_idle", None)
    if original_draw_idle is None:
        return

    def safe_draw_idle(self: FigureCanvas) -> None:
        try:
            original_draw_idle(self)
        except RuntimeError as exc:
            if not _is_deleted_qt_object_error(exc):
                raise
            try:
                setattr(self, "_draw_pending", False)
            except RuntimeError:
                pass
            LOGGER.debug("Ignored queued Matplotlib draw for a deleted Qt canvas.")

    setattr(FigureCanvas, "_draw_idle", safe_draw_idle)
    setattr(FigureCanvas, "_boef_safe_idle_draw_installed", True)


_install_safe_canvas_idle_draw()
SAFE_ANALYSIS_ERROR_MESSAGE = "An unexpected error occurred. Please try again."
CHART_TILE_THUMBNAIL_DPI = 120
CHART_TILE_TEXT_SCALE = 0.72
CHART_TILE_MIN_FONT_SIZE = 6.0
CHART_TILE_TEXT_ALPHA = 0.72
CHART_TILE_ANNOTATION_ALPHA = 0.28
CHART_TILE_LEGEND_FRAME_ALPHA = 0.24
CHART_TILE_LEGEND_EDGE_ALPHA = 0.16
APP_LOG_RELATIVE_PATH = Path(".boef") / "logs" / "boef.log"
DEFAULT_TRACK_GAUGE_M = 1.435
OVERLAY_BADGE_TEXT_GID = "boef-overlay-badge"
ENVELOPE_AUTO_MOVEMENT_FACTOR = 12.0
ENVELOPE_AUTO_DECAY_MARGIN_FACTOR = 8.0
ENVELOPE_MOVEMENT_BUFFER_FACTOR = 5.0


@dataclass(frozen=True)
class VehicleDefaults:
    display_name: str
    axle_load_kn: float
    wheel_load_kn: float
    wheel_diameter_mm: float
    bogie_spacing_m: float
    axles_per_bogie: int
    arrangement: str


VEHICLE_DEFAULTS: dict[str, VehicleDefaults] = {
    "freight_heavy_haul": VehicleDefaults(
        display_name="Freight (Heavy Haul)",
        axle_load_kn=265.0,
        wheel_load_kn=263.0,
        wheel_diameter_mm=920.0,
        bogie_spacing_m=9.0,
        axles_per_bogie=3,
        arrangement="Co-Co / 2x2-axle bogies",
    ),
    "heavy_metro": VehicleDefaults(
        display_name="Heavy Metro (Urban)",
        axle_load_kn=105.0,
        wheel_load_kn=105.0,
        wheel_diameter_mm=885.0,
        bogie_spacing_m=2.25,
        axles_per_bogie=2,
        arrangement="Bo-Bo / 2-axle bogies",
    ),
    "high_speed": VehicleDefaults(
        display_name="High Speed Rail",
        axle_load_kn=200.0,
        wheel_load_kn=200.0,
        wheel_diameter_mm=985.0,
        bogie_spacing_m=18.5,
        axles_per_bogie=2,
        arrangement="Bo-Bo / articulated EMU bogies",
    ),
    "lrt": VehicleDefaults(
        display_name="Light Rail Transit (LRT)",
        axle_load_kn=100.0,
        wheel_load_kn=100.0,
        wheel_diameter_mm=735.0,
        bogie_spacing_m=2.5,
        axles_per_bogie=2,
        arrangement="2-axle trucks",
    ),
}


class AnalysisType(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    SPECIAL = "special"


class StaticMode(str, Enum):
    SINGLE = "single"
    ENVELOPE_CLOSED_FORM = "envelope_closed_form"
    ENVELOPE_NUMERICAL = "envelope_numerical"


class AS5100ArrangementMode(str, Enum):
    FIXED_SELECTED = "fixed_selected"
    GOVERNING_SWEEP = "governing_sweep"


class DynamicAnnotationMode(str, Enum):
    COMPACT = "compact"
    FULL = "full"
    OFF = "off"


def _enum_value(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


@dataclass(frozen=True)
class TransitionContext:
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
    k_profile_n_per_m2: list[float] | None


@dataclass(frozen=True)
class LoadMarker:
    x_m: float
    load_kn: float
    label: str


@dataclass(frozen=True)
class MaterialField:
    key: str
    label: str
    unit: str
    to_si: Callable[[float], float]
    from_si: Callable[[float], float]
    decimals: int = 3
    minimum: float = 0.0
    maximum: float = 1.0e12
    optional: bool = False


class UnitInput(QWidget):
    """Numeric input with a unit suffix label."""

    def __init__(self, unit: str, *, decimals: int = 3, minimum: float = 0.0, maximum: float = 1.0e12) -> None:
        super().__init__()
        self._requested_visible = True
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setDecimals(decimals)
        self.spinbox.setRange(minimum, maximum)
        self.spinbox.setSingleStep(0.1)
        self.spinbox.setMinimumWidth(120)
        self.spinbox.setMaximumWidth(220)
        unit_label = QLabel(unit)
        unit_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.spinbox, stretch=1)
        layout.addWidget(unit_label)

    def value(self) -> float:
        return float(self.spinbox.value())

    def set_value(self, value: float) -> None:
        self.spinbox.setValue(float(value))

    def setVisible(self, visible: bool) -> None:  # noqa: N802 - Qt API override
        self._requested_visible = bool(visible)
        super().setVisible(visible)

    def isVisible(self) -> bool:  # noqa: N802 - Qt API override
        return self._requested_visible


class WheelLoadsWidget(QWidget):
    """Table widget for configuring multiple wheel loads."""

    def __init__(self) -> None:
        super().__init__()
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Q (kN)", "x (m)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setMaximumWidth(420)

        self.add_button = QPushButton("Add load")
        self.remove_button = QPushButton("Remove load")

        self.add_button.clicked.connect(self.add_row)
        self.remove_button.clicked.connect(self.remove_selected_row)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(button_layout)

    def add_row(self, load_kn: float = 100.0, position_m: float = 0.0) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        load_input = QDoubleSpinBox()
        load_input.setDecimals(2)
        load_input.setRange(0.01, 1.0e6)
        load_input.setValue(load_kn)

        position_input = QDoubleSpinBox()
        position_input.setDecimals(3)
        position_input.setRange(-1.0e5, 1.0e5)
        position_input.setValue(position_m)

        self.table.setCellWidget(row, 0, load_input)
        self.table.setCellWidget(row, 1, position_input)

    def remove_selected_row(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def rows(self) -> int:
        return self.table.rowCount()

    def loads(self) -> list[tuple[float, float]]:
        loads: list[tuple[float, float]] = []
        for row in range(self.table.rowCount()):
            load_widget = self.table.cellWidget(row, 0)
            position_widget = self.table.cellWidget(row, 1)
            if not isinstance(load_widget, QDoubleSpinBox) or not isinstance(position_widget, QDoubleSpinBox):
                continue
            loads.append((float(load_widget.value()), float(position_widget.value())))
        return loads

    def clear(self) -> None:
        self.table.setRowCount(0)


class VisibilityStateLabel(QLabel):
    """Label whose requested visibility is testable before its parent window is shown."""

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self._requested_visible = True

    def setVisible(self, visible: bool) -> None:  # noqa: N802 - Qt API override
        self._requested_visible = bool(visible)
        super().setVisible(visible)

    def isVisible(self) -> bool:  # noqa: N802 - Qt API override
        return self._requested_visible


class VisibilityStatePushButton(QPushButton):
    """Push button whose requested visibility is testable before its parent window is shown."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self._requested_visible = True

    def setVisible(self, visible: bool) -> None:  # noqa: N802 - Qt API override
        self._requested_visible = bool(visible)
        super().setVisible(visible)

    def isVisible(self) -> bool:  # noqa: N802 - Qt API override
        return self._requested_visible


class MaterialDialog(QDialog):
    """Generic CRUD dialog for material records."""

    error_style = "border: 1px solid #d93025;"

    def __init__(
        self,
        session: Session,
        *,
        title: str,
        list_items: Callable[[Session], list],
        create_item: Callable[..., object],
        update_item: Callable[..., object],
        delete_item: Callable[..., None],
        fields: Sequence[MaterialField],
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.session = session
        self.list_items = list_items
        self.create_item = create_item
        self.update_item = update_item
        self.delete_item = delete_item
        self.fields = fields

        self.list_widget = QListWidget()
        self.name_input = QLineEdit()
        self.field_inputs: dict[str, UnitInput] = {}
        self.validation_labels: dict[str, QLabel] = {}
        self._touched_fields: set[str] = set()

        form_layout = QFormLayout()
        form_layout.addRow("Name", self.name_input)
        self._add_validation_label(form_layout, "name")
        for field in self.fields:
            input_widget = UnitInput(
                field.unit,
                decimals=field.decimals,
                minimum=field.minimum,
                maximum=field.maximum,
            )
            self.field_inputs[field.key] = input_widget
            form_layout.addRow(field.label, input_widget)
            self._add_validation_label(form_layout, field.key)

        self.add_button = QPushButton("New")
        self.save_button = QPushButton("Save")
        self.delete_button = QPushButton("Delete")
        self.close_button = QPushButton("Close")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        right_layout = QVBoxLayout()
        right_layout.addLayout(form_layout)
        right_layout.addStretch()
        right_layout.addLayout(button_layout)

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.list_widget, stretch=1)
        main_layout.addLayout(right_layout, stretch=2)

        self.list_widget.currentItemChanged.connect(self._load_selected)
        self.add_button.clicked.connect(self._clear_selection)
        self.save_button.clicked.connect(self._save)
        self.delete_button.clicked.connect(self._delete)
        self.close_button.clicked.connect(self.accept)

        self.setMinimumWidth(720)
        self.list_widget.setMinimumWidth(220)
        self.name_input.setPlaceholderText("Required")

        self._bind_validation()
        self._refresh_list()
        self._update_action_buttons()

    def _add_validation_label(self, layout: QFormLayout, key: str) -> None:
        label = VisibilityStateLabel()
        label.setStyleSheet("color: #d93025;")
        label.setVisible(False)
        label.setWordWrap(True)
        self.validation_labels[key] = label
        layout.addRow("", label)

    def _bind_validation(self) -> None:
        self.name_input.textChanged.connect(lambda _value: self._mark_touched("name"))
        for field in self.fields:
            self.field_inputs[field.key].spinbox.valueChanged.connect(
                lambda _value, key=field.key: self._mark_touched(key)
            )

    def _mark_touched(self, key: str) -> None:
        self._touched_fields.add(key)
        self._validate_fields()

    def _set_widget_invalid(self, widget: QWidget, invalid: bool) -> None:
        widget.setStyleSheet(self.error_style if invalid else "")

    def _set_field_error(self, key: str, widget: QWidget, message: str | None, *, force: bool) -> None:
        label = self.validation_labels[key]
        should_show = force or key in self._touched_fields
        if message and should_show:
            label.setText(message)
            label.setVisible(True)
            self._set_widget_invalid(widget, True)
        else:
            label.setText("")
            label.setVisible(False)
            self._set_widget_invalid(widget, False)

    def _reset_validation(self) -> None:
        self._touched_fields.clear()
        for key, label in self.validation_labels.items():
            label.setText("")
            label.setVisible(False)
        self._set_widget_invalid(self.name_input, False)
        for field in self.fields:
            self._set_widget_invalid(self.field_inputs[field.key].spinbox, False)

    def _validate_fields(self, *, force: bool = False) -> bool:
        valid = True
        name = self.name_input.text().strip()
        name_error = "Name is required" if not name else None
        self._set_field_error("name", self.name_input, name_error, force=force)
        if name_error:
            valid = False

        for field in self.fields:
            value = self.field_inputs[field.key].value()
            if field.optional and value <= 0:
                error = None
            else:
                error = f"{field.label} must be > 0 {field.unit}" if value <= 0 else None
            self._set_field_error(
                field.key, self.field_inputs[field.key].spinbox, error, force=force
            )
            if error:
                valid = False
        return valid

    def _show_db_error(self, action: str) -> None:
        QMessageBox.critical(
            self,
            "Database error",
            f"Unable to {action}. Please try again.",
        )

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        for item in self.list_items(self.session):
            list_item = QListWidgetItem(item.name)
            list_item.setData(Qt.UserRole, item)
            self.list_widget.addItem(list_item)

    def _current_item(self) -> object | None:
        current = self.list_widget.currentItem()
        if current is None:
            return None
        return current.data(Qt.UserRole)

    def _clear_selection(self) -> None:
        self.list_widget.setCurrentItem(None)
        self.name_input.clear()
        for field in self.fields:
            self.field_inputs[field.key].set_value(0.0)
        self.name_input.setFocus()
        self._reset_validation()
        self._update_action_buttons()

    def _load_selected(self, current: QListWidgetItem | None) -> None:
        if current is None:
            self._update_action_buttons()
            return
        item = current.data(Qt.UserRole)
        if item is None:
            self._update_action_buttons()
            return
        self.name_input.setText(item.name)
        for field in self.fields:
            value = getattr(item, field.key)
            if value is None:
                self.field_inputs[field.key].set_value(0.0)
            else:
                self.field_inputs[field.key].set_value(field.from_si(float(value)))
        self._reset_validation()
        self._update_action_buttons()

    def _update_action_buttons(self) -> None:
        has_selection = self._current_item() is not None
        self.delete_button.setEnabled(has_selection)

    def _save(self) -> None:
        name = self.name_input.text().strip()
        values: dict[str, float | None] = {}
        for field in self.fields:
            value = self.field_inputs[field.key].value()
            if field.optional and value <= 0:
                values[field.key] = None
            else:
                values[field.key] = field.to_si(value)
        if not self._validate_fields(force=True):
            return
        try:
            current = self._current_item()
            if current is None:
                self.create_item(self.session, name=name, **values)
            else:
                self.update_item(self.session, current, name=name, **values)
        except ValueError as exc:
            QMessageBox.warning(self, "Validation error", str(exc))
            return
        except SQLAlchemyError:
            self.session.rollback()
            self._show_db_error("save changes")
            return
        self._refresh_list()

    def _delete(self) -> None:
        current = self._current_item()
        if current is None:
            return
        confirm = QMessageBox.question(
            self,
            "Delete",
            f"Delete '{current.name}'?",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self.delete_item(self.session, current)
        except SQLAlchemyError:
            self.session.rollback()
            self._show_db_error("delete the item")
            return
        self._refresh_list()
        self._clear_selection()


class LoadCaseDialog(QDialog):
    """CRUD dialog for load case records."""

    error_style = "border: 1px solid #d93025;"

    def __init__(self, session: Session) -> None:
        super().__init__()
        self.setWindowTitle("Load cases")
        self.session = session

        self.list_widget = QListWidget()
        self.name_input = QLineEdit()
        self.load_input = UnitInput("kN", decimals=2, minimum=0.01, maximum=1.0e6)
        self.description_input = QLineEdit()
        self.validation_labels: dict[str, QLabel] = {}
        self._touched_fields: set[str] = set()

        form_layout = QFormLayout()
        form_layout.addRow("Name", self.name_input)
        self._add_validation_label(form_layout, "name")
        form_layout.addRow("Load", self.load_input)
        self._add_validation_label(form_layout, "load")
        form_layout.addRow("Description", self.description_input)

        self.add_button = QPushButton("New")
        self.save_button = QPushButton("Save")
        self.delete_button = QPushButton("Delete")
        self.close_button = QPushButton("Close")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        right_layout = QVBoxLayout()
        right_layout.addLayout(form_layout)
        right_layout.addStretch()
        right_layout.addLayout(button_layout)

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.list_widget, stretch=1)
        main_layout.addLayout(right_layout, stretch=2)

        self.list_widget.currentItemChanged.connect(self._load_selected)
        self.add_button.clicked.connect(self._clear_selection)
        self.save_button.clicked.connect(self._save)
        self.delete_button.clicked.connect(self._delete)
        self.close_button.clicked.connect(self.accept)

        self.setMinimumWidth(680)
        self.list_widget.setMinimumWidth(220)
        self.name_input.setPlaceholderText("Required")
        self.description_input.setPlaceholderText("Optional")

        self._bind_validation()
        self._refresh_list()
        self._update_action_buttons()

    def _add_validation_label(self, layout: QFormLayout, key: str) -> None:
        label = VisibilityStateLabel()
        label.setStyleSheet("color: #d93025;")
        label.setVisible(False)
        label.setWordWrap(True)
        self.validation_labels[key] = label
        layout.addRow("", label)

    def _bind_validation(self) -> None:
        self.name_input.textChanged.connect(lambda _value: self._mark_touched("name"))
        self.load_input.spinbox.valueChanged.connect(lambda _value: self._mark_touched("load"))

    def _mark_touched(self, key: str) -> None:
        self._touched_fields.add(key)
        self._validate_fields()

    def _set_widget_invalid(self, widget: QWidget, invalid: bool) -> None:
        widget.setStyleSheet(self.error_style if invalid else "")

    def _set_field_error(self, key: str, widget: QWidget, message: str | None, *, force: bool) -> None:
        label = self.validation_labels[key]
        should_show = force or key in self._touched_fields
        if message and should_show:
            label.setText(message)
            label.setVisible(True)
            self._set_widget_invalid(widget, True)
        else:
            label.setText("")
            label.setVisible(False)
            self._set_widget_invalid(widget, False)

    def _reset_validation(self) -> None:
        self._touched_fields.clear()
        for key, label in self.validation_labels.items():
            label.setText("")
            label.setVisible(False)
        self._set_widget_invalid(self.name_input, False)
        self._set_widget_invalid(self.load_input.spinbox, False)

    def _validate_fields(self, *, force: bool = False) -> bool:
        valid = True
        name = self.name_input.text().strip()
        name_error = "Name is required" if not name else None
        self._set_field_error("name", self.name_input, name_error, force=force)
        if name_error:
            valid = False

        load_value = self.load_input.value()
        load_error = "Load must be > 0 kN" if load_value <= 0 else None
        self._set_field_error("load", self.load_input.spinbox, load_error, force=force)
        if load_error:
            valid = False
        return valid

    def _show_db_error(self, action: str) -> None:
        QMessageBox.critical(
            self,
            "Database error",
            f"Unable to {action}. Please try again.",
        )

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        for item in crud.list_load_cases(self.session):
            list_item = QListWidgetItem(item.name)
            list_item.setData(Qt.UserRole, item)
            self.list_widget.addItem(list_item)

    def _current_item(self) -> object | None:
        current = self.list_widget.currentItem()
        if current is None:
            return None
        return current.data(Qt.UserRole)

    def _clear_selection(self) -> None:
        self.list_widget.setCurrentItem(None)
        self.name_input.clear()
        self.load_input.set_value(0.0)
        self.description_input.clear()
        self.name_input.setFocus()
        self._reset_validation()
        self._update_action_buttons()

    def _load_selected(self, current: QListWidgetItem | None) -> None:
        if current is None:
            self._update_action_buttons()
            return
        item = current.data(Qt.UserRole)
        if item is None:
            self._update_action_buttons()
            return
        self.name_input.setText(item.name)
        self.load_input.set_value(n_to_kn(item.load_newtons))
        self.description_input.setText(item.description or "")
        self._reset_validation()
        self._update_action_buttons()

    def _update_action_buttons(self) -> None:
        has_selection = self._current_item() is not None
        self.delete_button.setEnabled(has_selection)

    def _save(self) -> None:
        name = self.name_input.text().strip()
        load_newtons = kn_to_n(self.load_input.value())
        description = self.description_input.text().strip() or None
        if not self._validate_fields(force=True):
            return
        try:
            current = self._current_item()
            if current is None:
                crud.create_load_case(
                    self.session,
                    name=name,
                    load_newtons=load_newtons,
                    description=description,
                )
            else:
                crud.update_load_case(
                    self.session,
                    current,
                    name=name,
                    load_newtons=load_newtons,
                    description=description,
                )
        except ValueError as exc:
            QMessageBox.warning(self, "Validation error", str(exc))
            return
        except SQLAlchemyError:
            self.session.rollback()
            self._show_db_error("save changes")
            return
        self._refresh_list()

    def _delete(self) -> None:
        current = self._current_item()
        if current is None:
            return
        confirm = QMessageBox.question(
            self,
            "Delete",
            f"Delete '{current.name}'?",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            crud.delete_load_case(self.session, current)
        except SQLAlchemyError:
            self.session.rollback()
            self._show_db_error("delete the item")
            return
        self._refresh_list()
        self._clear_selection()


class TrackConfigDialog(QDialog):
    """CRUD dialog for track configuration records."""

    error_style = "border: 1px solid #d93025;"

    @staticmethod
    def _set_compact_combo(combo: QComboBox, *, max_width: int = 240) -> None:
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        combo.setMinimumContentsLength(16)
        combo.setMaximumWidth(max_width)

    def __init__(self, session: Session) -> None:
        super().__init__()
        self.setWindowTitle("Track configurations")
        self.session = session

        self.list_widget = QListWidget()
        self.name_input = QLineEdit()
        self.project_combo = QComboBox()
        self.rail_combo = QComboBox()
        self.sleeper_combo = QComboBox()
        self.pad_combo = QComboBox()
        self.support_combo = QComboBox()
        self._set_compact_combo(self.rail_combo)
        self._set_compact_combo(self.sleeper_combo)
        self._set_compact_combo(self.pad_combo)
        self._set_compact_combo(self.support_combo)
        self.sleeper_spacing_input = UnitInput(
            "mm", decimals=1, minimum=100.0, maximum=2000.0
        )
        self.gauge_input = UnitInput("mm", decimals=1, minimum=500.0, maximum=2000.0)
        self.validation_labels: dict[str, QLabel] = {}
        self._touched_fields: set[str] = set()

        form_layout = QFormLayout()
        form_layout.addRow("Name", self.name_input)
        self._add_validation_label(form_layout, "name")
        form_layout.addRow("Project", self.project_combo)
        self._add_validation_label(form_layout, "project")
        form_layout.addRow("Rail", self.rail_combo)
        self._add_validation_label(form_layout, "rail")
        form_layout.addRow("Sleeper", self.sleeper_combo)
        self._add_validation_label(form_layout, "sleeper")
        form_layout.addRow("Pad", self.pad_combo)
        self._add_validation_label(form_layout, "pad")
        form_layout.addRow("Support profile", self.support_combo)
        self._add_validation_label(form_layout, "support")
        form_layout.addRow("Sleeper spacing", self.sleeper_spacing_input)
        self._add_validation_label(form_layout, "sleeper_spacing")
        form_layout.addRow("Gauge", self.gauge_input)
        self._add_validation_label(form_layout, "gauge")

        self.add_button = QPushButton("New")
        self.save_button = QPushButton("Save")
        self.delete_button = QPushButton("Delete")
        self.close_button = QPushButton("Close")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        right_layout = QVBoxLayout()
        right_layout.addLayout(form_layout)
        right_layout.addStretch()
        right_layout.addLayout(button_layout)

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.list_widget, stretch=1)
        main_layout.addLayout(right_layout, stretch=2)

        self.list_widget.currentItemChanged.connect(self._load_selected)
        self.add_button.clicked.connect(self._clear_selection)
        self.save_button.clicked.connect(self._save)
        self.delete_button.clicked.connect(self._delete)
        self.close_button.clicked.connect(self.accept)

        self.setMinimumWidth(760)
        self.list_widget.setMinimumWidth(240)
        self.name_input.setPlaceholderText("Required")

        self._bind_validation()
        self._refresh_materials()
        self._refresh_list()
        self._update_action_buttons()

    def _add_validation_label(self, layout: QFormLayout, key: str) -> None:
        label = VisibilityStateLabel()
        label.setStyleSheet("color: #d93025;")
        label.setVisible(False)
        label.setWordWrap(True)
        self.validation_labels[key] = label
        layout.addRow("", label)

    def _bind_validation(self) -> None:
        self.name_input.textChanged.connect(lambda _value: self._mark_touched("name"))
        self.project_combo.currentIndexChanged.connect(
            lambda _value: self._mark_touched("project")
        )
        self.rail_combo.currentIndexChanged.connect(
            lambda _value: self._mark_touched("rail")
        )
        self.sleeper_combo.currentIndexChanged.connect(
            lambda _value: self._mark_touched("sleeper")
        )
        self.pad_combo.currentIndexChanged.connect(lambda _value: self._mark_touched("pad"))
        self.support_combo.currentIndexChanged.connect(
            lambda _value: self._mark_touched("support")
        )
        self.sleeper_spacing_input.spinbox.valueChanged.connect(
            lambda _value: self._mark_touched("sleeper_spacing")
        )
        self.gauge_input.spinbox.valueChanged.connect(
            lambda _value: self._mark_touched("gauge")
        )

    def _mark_touched(self, key: str) -> None:
        self._touched_fields.add(key)
        self._validate_fields()

    def _set_widget_invalid(self, widget: QWidget, invalid: bool) -> None:
        widget.setStyleSheet(self.error_style if invalid else "")

    def _set_field_error(self, key: str, widget: QWidget, message: str | None, *, force: bool) -> None:
        label = self.validation_labels[key]
        should_show = force or key in self._touched_fields
        if message and should_show:
            label.setText(message)
            label.setVisible(True)
            self._set_widget_invalid(widget, True)
        else:
            label.setText("")
            label.setVisible(False)
            self._set_widget_invalid(widget, False)

    def _reset_validation(self) -> None:
        self._touched_fields.clear()
        for key, label in self.validation_labels.items():
            label.setText("")
            label.setVisible(False)
        self._set_widget_invalid(self.name_input, False)
        self._set_widget_invalid(self.project_combo, False)
        self._set_widget_invalid(self.rail_combo, False)
        self._set_widget_invalid(self.sleeper_combo, False)
        self._set_widget_invalid(self.pad_combo, False)
        self._set_widget_invalid(self.support_combo, False)
        self._set_widget_invalid(self.sleeper_spacing_input.spinbox, False)
        self._set_widget_invalid(self.gauge_input.spinbox, False)

    def _validate_fields(self, *, force: bool = False) -> bool:
        valid = True
        name = self.name_input.text().strip()
        name_error = "Name is required" if not name else None
        self._set_field_error("name", self.name_input, name_error, force=force)
        if name_error:
            valid = False

        combo_fields = [
            ("project", self.project_combo, "Project is required"),
            ("rail", self.rail_combo, "Rail is required"),
            ("sleeper", self.sleeper_combo, "Sleeper is required"),
            ("pad", self.pad_combo, "Pad is required"),
            ("support", self.support_combo, "Support profile is required"),
        ]
        for key, combo, message in combo_fields:
            error = message if combo.currentData() is None else None
            self._set_field_error(key, combo, error, force=force)
            if error:
                valid = False

        spacing_value = self.sleeper_spacing_input.value()
        spacing_error = "Sleeper spacing must be > 0 mm" if spacing_value <= 0 else None
        self._set_field_error(
            "sleeper_spacing",
            self.sleeper_spacing_input.spinbox,
            spacing_error,
            force=force,
        )
        if spacing_error:
            valid = False

        gauge_value = self.gauge_input.value()
        gauge_error = "Gauge must be > 0 mm" if gauge_value <= 0 else None
        self._set_field_error(
            "gauge",
            self.gauge_input.spinbox,
            gauge_error,
            force=force,
        )
        if gauge_error:
            valid = False
        return valid

    def _show_db_error(self, action: str) -> None:
        QMessageBox.critical(
            self,
            "Database error",
            f"Unable to {action}. Please try again.",
        )

    def _refresh_materials(self) -> None:
        self._fill_combo(self.project_combo, crud.list_projects(self.session))
        self._fill_combo(self.rail_combo, crud.list_rails(self.session))
        self._fill_combo(self.sleeper_combo, crud.list_sleepers(self.session))
        self._fill_combo(self.pad_combo, crud.list_pads(self.session))
        self._fill_combo(self.support_combo, crud.list_support_profiles(self.session))

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        for item in crud.list_track_configs(self.session):
            project_name = item.project.name if item.project else "Unknown project"
            label = f"{item.name} ({project_name})"
            list_item = QListWidgetItem(label)
            list_item.setData(Qt.UserRole, item)
            self.list_widget.addItem(list_item)

    def _fill_combo(self, combo: QComboBox, items: Sequence) -> None:
        combo.clear()
        for item in items:
            combo.addItem(item.name, item)

    def _current_item(self) -> object | None:
        current = self.list_widget.currentItem()
        if current is None:
            return None
        return current.data(Qt.UserRole)

    def _clear_selection(self) -> None:
        self.list_widget.setCurrentItem(None)
        self.name_input.clear()
        self._reset_combo(self.project_combo)
        self._reset_combo(self.rail_combo)
        self._reset_combo(self.sleeper_combo)
        self._reset_combo(self.pad_combo)
        self._reset_combo(self.support_combo)
        self.sleeper_spacing_input.set_value(600.0)
        self.gauge_input.set_value(1435.0)
        self.name_input.setFocus()
        self._reset_validation()
        self._update_action_buttons()

    def _reset_combo(self, combo: QComboBox) -> None:
        if combo.count() > 0:
            combo.setCurrentIndex(0)

    def _select_combo_by_id(self, combo: QComboBox, item_id: int) -> None:
        for index in range(combo.count()):
            data = combo.itemData(index)
            if data is not None and getattr(data, "id", None) == item_id:
                combo.setCurrentIndex(index)
                return

    def _load_selected(self, current: QListWidgetItem | None) -> None:
        if current is None:
            self._update_action_buttons()
            return
        item = current.data(Qt.UserRole)
        if item is None:
            self._update_action_buttons()
            return
        self.name_input.setText(item.name)
        self._select_combo_by_id(self.project_combo, item.project_id)
        self._select_combo_by_id(self.rail_combo, item.rail_id)
        self._select_combo_by_id(self.sleeper_combo, item.sleeper_id)
        self._select_combo_by_id(self.pad_combo, item.pad_id)
        self._select_combo_by_id(self.support_combo, item.support_profile_id)
        self.sleeper_spacing_input.set_value(m_to_mm(item.sleeper_spacing_m))
        self.gauge_input.set_value(m_to_mm(item.gauge_m))
        self._reset_validation()
        self._update_action_buttons()

    def _update_action_buttons(self) -> None:
        has_selection = self._current_item() is not None
        self.delete_button.setEnabled(has_selection)

    def _require_selection(self, combo: QComboBox, label: str) -> object:
        selection = combo.currentData()
        if selection is None:
            raise ValueError(f"{label} is required")
        return selection

    def _save(self) -> None:
        name = self.name_input.text().strip()
        if not self._validate_fields(force=True):
            return
        try:
            project = self._require_selection(self.project_combo, "Project")
            rail = self._require_selection(self.rail_combo, "Rail")
            sleeper = self._require_selection(self.sleeper_combo, "Sleeper")
            pad = self._require_selection(self.pad_combo, "Pad")
            support = self._require_selection(self.support_combo, "Support profile")
            sleeper_spacing_m = mm_to_m(self.sleeper_spacing_input.value())
            gauge_m = mm_to_m(self.gauge_input.value())
            current = self._current_item()
            if current is None:
                crud.create_track_config(
                    self.session,
                    name=name,
                    project_id=project.id,
                    rail_id=rail.id,
                    sleeper_id=sleeper.id,
                    pad_id=pad.id,
                    support_profile_id=support.id,
                    sleeper_spacing_m=sleeper_spacing_m,
                    gauge_m=gauge_m,
                )
            else:
                crud.update_track_config(
                    self.session,
                    current,
                    name=name,
                    project_id=project.id,
                    rail_id=rail.id,
                    sleeper_id=sleeper.id,
                    pad_id=pad.id,
                    support_profile_id=support.id,
                    sleeper_spacing_m=sleeper_spacing_m,
                    gauge_m=gauge_m,
                )
        except ValueError as exc:
            QMessageBox.warning(self, "Validation error", str(exc))
            return
        except SQLAlchemyError:
            self.session.rollback()
            self._show_db_error("save changes")
            return
        self._refresh_list()

    def _delete(self) -> None:
        current = self._current_item()
        if current is None:
            return
        confirm = QMessageBox.question(
            self,
            "Delete",
            f"Delete '{current.name}'?",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            crud.delete_track_config(self.session, current)
        except SQLAlchemyError:
            self.session.rollback()
            self._show_db_error("delete the item")
            return
        self._refresh_list()
        self._clear_selection()


class ProjectDialog(QDialog):
    """CRUD dialog for project records."""

    error_style = "border: 1px solid #d93025;"

    def __init__(self, session: Session) -> None:
        super().__init__()
        self.setWindowTitle("Projects")
        self.session = session

        self.list_widget = QListWidget()
        self.name_input = QLineEdit()
        self.description_input = QPlainTextEdit()
        self.vehicle_type_combo = QComboBox()
        self.vehicle_subtype_input = QLineEdit()
        self.vehicle_defaults_label = QLabel("Defaults: —")
        self.vehicle_defaults_label.setWordWrap(True)
        self.project_speed_input = UnitInput("km/h", decimals=1, minimum=0.0, maximum=400.0)
        self.project_wheel_radius_input = UnitInput("mm", decimals=1, minimum=0.0, maximum=10_000.0)
        self.validation_labels: dict[str, QLabel] = {}
        self._touched_fields: set[str] = set()

        form_layout = QFormLayout()
        form_layout.addRow("Name", self.name_input)
        self._add_validation_label(form_layout, "name")
        form_layout.addRow("Description", self.description_input)
        form_layout.addRow("Vehicle type", self.vehicle_type_combo)
        form_layout.addRow("Vehicle subtype", self.vehicle_subtype_input)
        form_layout.addRow("Rolling stock defaults", self.vehicle_defaults_label)
        form_layout.addRow("Design speed (optional)", self.project_speed_input)
        form_layout.addRow("Wheel radius override (optional)", self.project_wheel_radius_input)

        self.add_button = QPushButton("New")
        self.save_button = QPushButton("Save")
        self.delete_button = QPushButton("Delete")
        self.close_button = QPushButton("Close")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        right_layout = QVBoxLayout()
        right_layout.addLayout(form_layout)
        right_layout.addStretch()
        right_layout.addLayout(button_layout)

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.list_widget, stretch=1)
        main_layout.addLayout(right_layout, stretch=2)

        self.list_widget.currentItemChanged.connect(self._load_selected)
        self.add_button.clicked.connect(self._clear_selection)
        self.save_button.clicked.connect(self._save)
        self.delete_button.clicked.connect(self._delete)
        self.close_button.clicked.connect(self.accept)

        self.setMinimumWidth(720)
        self.list_widget.setMinimumWidth(240)
        self.name_input.setPlaceholderText("Required")
        self.description_input.setPlaceholderText("Optional")
        self.vehicle_subtype_input.setPlaceholderText("Optional")
        self.project_speed_input.set_value(0.0)
        self.project_wheel_radius_input.set_value(0.0)

        self._bind_validation()
        self._populate_vehicle_types()
        self._refresh_list()
        self._update_action_buttons()

    def _add_validation_label(self, layout: QFormLayout, key: str) -> None:
        label = VisibilityStateLabel()
        label.setStyleSheet("color: #d93025;")
        label.setVisible(False)
        label.setWordWrap(True)
        self.validation_labels[key] = label
        layout.addRow("", label)

    def _bind_validation(self) -> None:
        self.name_input.textChanged.connect(lambda _value: self._mark_touched("name"))
        self.vehicle_type_combo.currentIndexChanged.connect(self._update_vehicle_defaults)

    def _mark_touched(self, key: str) -> None:
        self._touched_fields.add(key)
        self._validate_fields()

    def _set_widget_invalid(self, widget: QWidget, invalid: bool) -> None:
        widget.setStyleSheet(self.error_style if invalid else "")

    def _set_field_error(self, key: str, widget: QWidget, message: str | None, *, force: bool) -> None:
        label = self.validation_labels[key]
        should_show = force or key in self._touched_fields
        if message and should_show:
            label.setText(message)
            label.setVisible(True)
            self._set_widget_invalid(widget, True)
        else:
            label.setText("")
            label.setVisible(False)
            self._set_widget_invalid(widget, False)

    def _reset_validation(self) -> None:
        self._touched_fields.clear()
        for label in self.validation_labels.values():
            label.setText("")
            label.setVisible(False)
        self._set_widget_invalid(self.name_input, False)

    def _validate_fields(self, *, force: bool = False) -> bool:
        valid = True
        name = self.name_input.text().strip()
        name_error = "Name is required" if not name else None
        self._set_field_error("name", self.name_input, name_error, force=force)
        if name_error:
            valid = False
        return valid

    def _show_db_error(self, action: str) -> None:
        QMessageBox.critical(
            self,
            "Database error",
            f"Unable to {action}. Please try again.",
        )

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        for project in crud.list_projects(self.session):
            list_item = QListWidgetItem(project.name)
            list_item.setData(Qt.UserRole, project)
            self.list_widget.addItem(list_item)

    def _current_item(self) -> Project | None:
        current = self.list_widget.currentItem()
        if current is None:
            return None
        return current.data(Qt.UserRole)

    def _clear_selection(self) -> None:
        self.list_widget.setCurrentItem(None)
        self.name_input.clear()
        self.description_input.clear()
        self.vehicle_subtype_input.clear()
        self._reset_combo(self.vehicle_type_combo)
        self.project_speed_input.set_value(0.0)
        self.project_wheel_radius_input.set_value(0.0)
        self.name_input.setFocus()
        self._reset_validation()
        self._update_action_buttons()

    def _load_selected(self, current: QListWidgetItem | None) -> None:
        if current is None:
            self._update_action_buttons()
            return
        item = current.data(Qt.UserRole)
        if item is None:
            self._update_action_buttons()
            return
        self.name_input.setText(item.name)
        self.description_input.setPlainText(item.description or "")
        self._select_combo_by_text(self.vehicle_type_combo, item.vehicle_type)
        self.vehicle_subtype_input.setText(item.vehicle_subtype or "")
        self.project_speed_input.set_value(item.design_speed_kmh or 0.0)
        self.project_wheel_radius_input.set_value(item.design_wheel_radius_mm or 0.0)
        self._update_vehicle_defaults()
        self._reset_validation()
        self._update_action_buttons()

    def _update_action_buttons(self) -> None:
        has_selection = self._current_item() is not None
        self.delete_button.setEnabled(has_selection)

    def _save(self) -> None:
        name = self.name_input.text().strip()
        description = self.description_input.toPlainText().strip() or None
        vehicle_type = self.vehicle_type_combo.currentData()
        vehicle_subtype = self.vehicle_subtype_input.text().strip() or None
        design_speed_kmh = self.project_speed_input.value()
        wheel_radius_mm = self.project_wheel_radius_input.value()
        if design_speed_kmh <= 0:
            design_speed_kmh = None
        if wheel_radius_mm <= 0:
            wheel_radius_mm = None
        if not self._validate_fields(force=True):
            return
        try:
            current = self._current_item()
            if current is None:
                crud.create_project(
                    self.session,
                    name=name,
                    description=description,
                    vehicle_type=vehicle_type,
                    vehicle_subtype=vehicle_subtype,
                    design_speed_kmh=design_speed_kmh,
                    design_wheel_radius_mm=wheel_radius_mm,
                )
            else:
                crud.update_project(
                    self.session,
                    current,
                    name=name,
                    description=description,
                    vehicle_type=vehicle_type,
                    vehicle_subtype=vehicle_subtype,
                    design_speed_kmh=design_speed_kmh,
                    design_wheel_radius_mm=wheel_radius_mm,
                )
        except ValueError as exc:
            QMessageBox.warning(self, "Validation error", str(exc))
            return
        except SQLAlchemyError:
            self.session.rollback()
            self._show_db_error("save changes")
            return
        self._refresh_list()

    def _populate_vehicle_types(self) -> None:
        self.vehicle_type_combo.clear()
        self.vehicle_type_combo.addItem("—", None)
        for key, defaults in VEHICLE_DEFAULTS.items():
            self.vehicle_type_combo.addItem(defaults.display_name, key)

    def _update_vehicle_defaults(self) -> None:
        key = self.vehicle_type_combo.currentData()
        if key is None:
            self.vehicle_defaults_label.setText("Defaults: —")
            return
        defaults = VEHICLE_DEFAULTS[key]
        self.vehicle_defaults_label.setText(
            "Defaults: "
            f"{defaults.axle_load_kn:.0f} kN axle, "
            f"{defaults.wheel_load_kn:.0f} kN wheel, "
            f"{defaults.wheel_diameter_mm:.0f} mm wheel, "
            f"{defaults.bogie_spacing_m:.1f} m bogie spacing"
        )

    def _select_combo_by_text(self, combo: QComboBox, text: str | None) -> None:
        if text is None:
            return
        for index in range(combo.count()):
            if combo.itemData(index) == text:
                combo.setCurrentIndex(index)
                return

    def _reset_combo(self, combo: QComboBox) -> None:
        if combo.count() > 0:
            combo.setCurrentIndex(0)

    def _delete(self) -> None:
        current = self._current_item()
        if current is None:
            return
        confirm = QMessageBox.question(
            self,
            "Delete",
            (
                f"Delete '{current.name}'?\n\n"
                "This will also remove its track configs and results."
            ),
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            crud.delete_project(self.session, current)
        except SQLAlchemyError:
            self.session.rollback()
            self._show_db_error("delete the item")
            return
        self._refresh_list()
        self._clear_selection()


class PlotPanel(QWidget):
    """Matplotlib plot panel."""

    _CHART_PRIMARY_COLORS: dict[str, str] = {
        "deflection": "#1f77b4",
        "moment": "#2a9d8f",
        "shear": "#f4a261",
        "reaction": "#2ca02c",
        "sleeper": "#9467bd",
        "pressure": "#d62728",
        "stress": "#7f3c8d",
        "rail_deflection": "#1f77b4",
        "rail_moment": "#2a9d8f",
        "transition_profile": "#5e60ce",
        "dynamic_deflection": "#1f77b4",
        "dynamic_moment": "#2a9d8f",
        "dynamic_shear": "#f4a261",
        "dynamic_reaction": "#2ca02c",
        "dynamic_damping": "#8c564b",
        "dynamic_time": "#1f77b4",
        "dynamic_fft": "#5e60ce",
        "dynamic_psd": "#d62728",
        "dynamic_impedance": "#7f3c8d",
        "special_floating_slab_transmissibility": "#2ca02c",
        "special_floating_slab_attenuation": "#d62728",
    }
    _SERIES_COLOR_CYCLE: tuple[str, ...] = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#17becf",
        "#bcbd22",
    )
    _SERIES_LINESTYLE_CYCLE: tuple[object, ...] = (
        "-",
        "--",
        "-.",
        ":",
        (0, (5, 2)),
        (0, (3, 1, 1, 1)),
        (0, (1, 1)),
    )

    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(figsize=(5, 4), dpi=120, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self._help_action: QAction | None = None
        self._custom_chart_action: QAction | None = None
        self._chart_id: str = "chart"
        self._chart_title: str = "Chart"
        self._rendered_series: list[RenderedSeries] = []
        self._drag_text_artist: object | None = None
        self._closed = False
        self.axes = self.figure.add_subplot(111)
        self.footer_widget = QFrame()
        self.footer_widget.setObjectName("chartFooter")
        self.footer_widget.setVisible(False)
        self.footer_widget.setStyleSheet(
            "#chartFooter { border-top: 1px solid #dddddd; background: #ffffff; }"
        )
        footer_layout = QHBoxLayout(self.footer_widget)
        footer_layout.setContentsMargins(8, 6, 8, 4)
        footer_layout.setSpacing(24)
        self.footer_left_label = QLabel()
        self.footer_right_label = QLabel()
        for label in (self.footer_left_label, self.footer_right_label):
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignTop)
            label.setStyleSheet("color: #333333; font-size: 11px;")
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.footer_right_label.setAlignment(Qt.AlignTop | Qt.AlignRight)
        footer_layout.addWidget(self.footer_left_label, stretch=3)
        footer_layout.addWidget(self.footer_right_label, stretch=2)

        self.axes.set_facecolor("#fbfbfb")
        self.figure.patch.set_facecolor("#ffffff")
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.footer_widget)
        self.canvas.mpl_connect("button_press_event", self._handle_text_drag_press)
        self.canvas.mpl_connect("motion_notify_event", self._handle_text_drag_motion)
        self.canvas.mpl_connect("button_release_event", self._handle_text_drag_release)

    @staticmethod
    def _is_deleted_qt_object_error(exc: RuntimeError) -> bool:
        return _is_deleted_qt_object_error(exc)

    def _cancel_pending_draw(self) -> None:
        try:
            if hasattr(self.canvas, "_draw_pending"):
                setattr(self.canvas, "_draw_pending", False)
        except RuntimeError as exc:
            if not self._is_deleted_qt_object_error(exc):
                raise

    def prepare_for_close(self) -> None:
        self._closed = True
        self._cancel_pending_draw()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.prepare_for_close()
        super().closeEvent(event)

    def request_draw_idle(self) -> None:
        if self._closed:
            return
        try:
            self.canvas.draw_idle()
        except RuntimeError as exc:
            if self._is_deleted_qt_object_error(exc):
                self._closed = True
                return
            raise

    def draw_now(self) -> None:
        if self._closed:
            return
        try:
            self.canvas.draw()
        except RuntimeError as exc:
            if self._is_deleted_qt_object_error(exc):
                self._closed = True
                return
            raise

    def clear_plot(self) -> None:
        self._reset_axes()
        self._rendered_series = []
        self._drag_text_artist = None
        self.axes.set_title("")
        self.axes.set_xlabel("")
        self.axes.set_ylabel("")
        self.request_draw_idle()

    def set_footer_texts(self, left_text: str, right_text: str = "") -> None:
        self.footer_left_label.setText(left_text)
        self.footer_right_label.setText(right_text)
        self.footer_right_label.setVisible(bool(right_text.strip()))
        self.footer_widget.setVisible(bool(left_text.strip() or right_text.strip()))

    def clear_footer_texts(self) -> None:
        self.set_footer_texts("", "")

    def footer_texts(self) -> tuple[str, str]:
        return self.footer_left_label.text(), self.footer_right_label.text()

    @staticmethod
    def _is_relocatable_text(text_artist: object) -> bool:
        return bool(getattr(text_artist, "_boef_relocatable_text", False))

    def make_text_relocatable(self, text_artist: object) -> None:
        setattr(text_artist, "_boef_relocatable_text", True)
        try:
            text_artist.set_picker(True)
        except AttributeError:
            pass

    def add_relocatable_text(self, *args: object, **kwargs: object) -> object:
        text_artist = self.axes.text(*args, **kwargs)
        self.make_text_relocatable(text_artist)
        return text_artist

    def _handle_text_drag_press(self, event: object) -> None:
        if getattr(event, "button", None) != 1:
            return
        for axis in reversed(list(self.figure.axes)):
            for text_artist in reversed(list(axis.texts)):
                if not self._is_relocatable_text(text_artist):
                    continue
                contains = getattr(text_artist, "contains", None)
                if not callable(contains):
                    continue
                try:
                    hit, _details = contains(event)
                except Exception:
                    hit = False
                if hit:
                    self._drag_text_artist = text_artist
                    return

    def _handle_text_drag_motion(self, event: object) -> None:
        text_artist = self._drag_text_artist
        if text_artist is None:
            return
        axis = getattr(text_artist, "axes", None)
        if axis is None or getattr(event, "x", None) is None or getattr(event, "y", None) is None:
            return
        try:
            x_pos, y_pos = axis.transAxes.inverted().transform((event.x, event.y))
            text_artist.set_position((float(x_pos), float(y_pos)))
            text_artist.set_transform(axis.transAxes)
            self.request_draw_idle()
        except Exception:
            self._drag_text_artist = None

    def _handle_text_drag_release(self, _event: object) -> None:
        self._drag_text_artist = None

    def _chart_primary_color(self) -> str:
        return self._CHART_PRIMARY_COLORS.get(self._chart_id, "#1f77b4")

    def _default_series_style(self, *, index: int, primary_index: int) -> dict[str, object]:
        color = self._chart_primary_color() if index == primary_index else self._SERIES_COLOR_CYCLE[
            (index - primary_index - 1) % len(self._SERIES_COLOR_CYCLE)
        ]
        linestyle = self._SERIES_LINESTYLE_CYCLE[index % len(self._SERIES_LINESTYLE_CYCLE)]
        linewidth = 2.2 if index == primary_index else 1.9
        return {"color": color, "linestyle": linestyle, "linewidth": linewidth}

    def set_chart_context(self, *, chart_id: str, title: str) -> None:
        self._chart_id = chart_id
        self._chart_title = title

    def rendered_series(self) -> list[RenderedSeries]:
        return list(self._rendered_series)

    def configure_help_action(
        self,
        *,
        callback: Callable[[], None] | None,
        text: str = "Help",
        tooltip: str = "",
    ) -> None:
        """Configure an optional help action on the plot toolbar."""
        if callback is None:
            if self._help_action is not None:
                self.toolbar.removeAction(self._help_action)
                self._help_action = None
            return
        if self._help_action is None:
            self._help_action = self.toolbar.addAction(text)
        self._help_action.setText(text)
        self._help_action.setToolTip(tooltip or text)
        try:
            self._help_action.triggered.disconnect()
        except (TypeError, RuntimeError):
            pass
        self._help_action.triggered.connect(lambda _checked=False: callback())

    def configure_custom_chart_action(
        self,
        *,
        callback: Callable[[], None] | None,
        text: str = "Custom chart",
        tooltip: str = "",
    ) -> None:
        if callback is None:
            if self._custom_chart_action is not None:
                self.toolbar.removeAction(self._custom_chart_action)
                self._custom_chart_action = None
            return
        if self._custom_chart_action is None:
            self._custom_chart_action = self.toolbar.addAction(text)
        self._custom_chart_action.setText(text)
        self._custom_chart_action.setToolTip(tooltip or text)
        try:
            self._custom_chart_action.triggered.disconnect()
        except (TypeError, RuntimeError):
            pass
        self._custom_chart_action.triggered.connect(lambda _checked=False: callback())

    def update_plot(
        self,
        x: list[float],
        y: list[float],
        *,
        title: str,
        xlabel: str,
        ylabel: str,
        annotations: Sequence[tuple[str, tuple[float, float], dict[str, object]]] | None = None,
        critical_labels: bool = False,
        load_markers: Sequence[LoadMarker] | None = None,
    ) -> None:
        self._reset_axes()
        style = self._default_series_style(index=0, primary_index=0)
        line = self.axes.plot(x, y, antialiased=True, **style)[0]
        self.axes.set_title(title, pad=10)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self._set_axis_limits(x, y)
        self._apply_sleek_grid(
            axis=self.axes,
            x_values=x,
            y_values=[y],
        )
        if critical_labels:
            self._annotate_series_critical_points(
                x_values=x,
                y_values=y,
                color=line.get_color() or "#1f77b4",
                series_label=None,
                include_zero_crossings=True,
                max_zero_crossings=2,
                seed=0,
            )
        if annotations:
            for text, (x_pos, y_pos), style in annotations:
                text_artist = self.add_relocatable_text(
                    x_pos,
                    y_pos,
                    text,
                    transform=self.axes.transAxes,
                    **style,
                )
                text_artist.set_gid(OVERLAY_BADGE_TEXT_GID)
        self._render_load_markers(load_markers)
        self._cache_rendered_series(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            series=[
                (
                    f"{self._chart_id}:primary",
                    title,
                    x,
                    y,
                    line.get_color(),
                    line.get_linestyle(),
                )
            ],
        )
        self.request_draw_idle()

    def update_comparison_plot(
        self,
        x: list[float],
        left: list[float],
        right: list[float],
        *,
        title: str,
        xlabel: str,
        ylabel: str,
        left_label: str = "Left rail",
        right_label: str = "Right rail",
        critical_labels: bool = False,
        load_markers: Sequence[LoadMarker] | None = None,
    ) -> None:
        self._reset_axes()
        left_style = self._default_series_style(index=0, primary_index=0)
        left_style["label"] = left_label
        right_style = self._default_series_style(index=1, primary_index=0)
        right_style["label"] = right_label
        left_line = self.axes.plot(x, left, **left_style)[0]
        right_line = self.axes.plot(x, right, **right_style)[0]
        self.axes.set_title(title, pad=10)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self._set_axis_limits_multi(x, [left, right])
        self._apply_sleek_grid(
            axis=self.axes,
            x_values=x,
            y_values=[left, right],
        )
        if critical_labels:
            seed = 0
            seed = self._annotate_series_critical_points(
                x_values=x,
                y_values=left,
                color=left_line.get_color() or "#1f77b4",
                series_label=left_label,
                include_zero_crossings=True,
                max_zero_crossings=1,
                seed=seed,
            )
            seed = self._annotate_series_critical_points(
                x_values=x,
                y_values=right,
                color=right_line.get_color() or "#ff7f0e",
                series_label=right_label,
                include_zero_crossings=True,
                max_zero_crossings=1,
                seed=seed,
            )
            self._annotate_intersections_between_series(
                x_values=x,
                y_primary=left,
                y_secondary=right,
                color="#555555",
                label="L/R int",
                max_points=2,
                seed=seed,
            )
        self._render_load_markers(load_markers)
        self.axes.legend(loc="best", fontsize=8)
        self._cache_rendered_series(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            series=[
                (
                    f"{self._chart_id}:left",
                    left_label,
                    x,
                    left,
                    left_line.get_color(),
                    left_line.get_linestyle(),
                ),
                (
                    f"{self._chart_id}:right",
                    right_label,
                    x,
                    right,
                    right_line.get_color(),
                    right_line.get_linestyle(),
                ),
            ],
        )
        self.request_draw_idle()

    def update_multi_plot(
        self,
        series: Sequence[tuple[list[float], list[float], str]],
        *,
        title: str,
        xlabel: str,
        ylabel: str,
        primary_index: int = 0,
        styles: Sequence[dict[str, object]] | None = None,
        critical_labels: bool = False,
        load_markers: Sequence[LoadMarker] | None = None,
    ) -> None:
        self._reset_axes()
        plotted: list[tuple[str, str, list[float], list[float], str | None, str | None]] = []
        drawn_series: list[tuple[list[float], list[float], str, str]] = []
        for index, (x_values, y_values, label) in enumerate(series):
            style: dict[str, object] = self._default_series_style(index=index, primary_index=primary_index)
            style["label"] = label
            if styles and index < len(styles):
                style.update(styles[index])
            line = self.axes.plot(x_values, y_values, **style)[0]
            plotted.append(
                (
                    f"{self._chart_id}:{index}",
                    label,
                    list(x_values),
                    list(y_values),
                    line.get_color(),
                    line.get_linestyle(),
                )
            )
            drawn_series.append(
                (
                    list(x_values),
                    list(y_values),
                    label,
                    line.get_color() or "#1f77b4",
                )
            )
        self.axes.set_title(title, pad=10)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self._set_axis_limits_series(series)
        self._apply_sleek_grid(
            axis=self.axes,
            x_values=drawn_series[0][0] if drawn_series else [],
            y_values=[item[1] for item in drawn_series],
        )
        if critical_labels:
            seed = 0
            for x_values, y_values, label, color in drawn_series:
                seed = self._annotate_series_critical_points(
                    x_values=x_values,
                    y_values=y_values,
                    color=color,
                    series_label=label if len(drawn_series) > 1 else None,
                    include_zero_crossings=True,
                    max_zero_crossings=1,
                    seed=seed,
                )
            if 0 <= primary_index < len(drawn_series):
                primary_x, primary_y, _, _ = drawn_series[primary_index]
                for index, (x_values, y_values, label, _) in enumerate(drawn_series):
                    if index == primary_index:
                        continue
                    seed = self._annotate_intersections_between_series(
                        x_values=primary_x,
                        y_primary=primary_y,
                        y_secondary=y_values,
                        color="#555555",
                        label=f"int {label}",
                        max_points=1,
                        seed=seed,
                        x_secondary=x_values,
                    )
        self._render_load_markers(load_markers)
        if len(series) > 1:
            self.axes.legend(loc="best", fontsize=8)
        self._cache_rendered_series(title=title, xlabel=xlabel, ylabel=ylabel, series=plotted)
        self.request_draw_idle()

    def update_multi_plot_dual_axis(
        self,
        primary_series: Sequence[tuple[list[float], list[float], str]],
        secondary_series: Sequence[tuple[list[float], list[float], str]],
        *,
        title: str,
        xlabel: str,
        primary_ylabel: str,
        secondary_ylabel: str,
        primary_styles: Sequence[dict[str, object]] | None = None,
        secondary_styles: Sequence[dict[str, object]] | None = None,
        critical_labels: bool = False,
        load_markers: Sequence[LoadMarker] | None = None,
    ) -> None:
        self._reset_axes()
        secondary_axis = self.axes.twinx()
        plotted: list[tuple[str, str, list[float], list[float], str | None, str | None]] = []
        primary_drawn: list[tuple[list[float], list[float], str, str]] = []
        secondary_drawn: list[tuple[list[float], list[float], str, str]] = []
        legend_lines: list[object] = []
        legend_labels: list[str] = []

        for index, (x_values, y_values, label) in enumerate(primary_series):
            style: dict[str, object] = self._default_series_style(index=index, primary_index=0)
            style["label"] = label
            if primary_styles and index < len(primary_styles):
                style.update(primary_styles[index])
            line = self.axes.plot(x_values, y_values, **style)[0]
            plotted.append(
                (
                    f"{self._chart_id}:primary:{index}",
                    label,
                    list(x_values),
                    list(y_values),
                    line.get_color(),
                    line.get_linestyle(),
                )
            )
            primary_drawn.append((list(x_values), list(y_values), label, line.get_color() or "#1f77b4"))
            legend_lines.append(line)
            legend_labels.append(label)

        for index, (x_values, y_values, label) in enumerate(secondary_series):
            style: dict[str, object] = self._default_series_style(index=index + len(primary_series), primary_index=0)
            style["label"] = label
            if secondary_styles and index < len(secondary_styles):
                style.update(secondary_styles[index])
            line = secondary_axis.plot(x_values, y_values, **style)[0]
            plotted.append(
                (
                    f"{self._chart_id}:secondary:{index}",
                    label,
                    list(x_values),
                    list(y_values),
                    line.get_color(),
                    line.get_linestyle(),
                )
            )
            secondary_drawn.append((list(x_values), list(y_values), label, line.get_color() or "#1f77b4"))
            legend_lines.append(line)
            legend_labels.append(label)

        self.axes.set_title(title, pad=10)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(primary_ylabel)
        secondary_axis.set_ylabel(secondary_ylabel)

        self._set_axis_limits_series_on_axis(self.axes, primary_series)
        self._set_axis_limits_series_on_axis(secondary_axis, secondary_series)
        primary_x = primary_drawn[0][0] if primary_drawn else (secondary_drawn[0][0] if secondary_drawn else [])
        self._apply_sleek_grid(
            axis=self.axes,
            x_values=primary_x,
            y_values=[item[1] for item in primary_drawn],
        )
        self._apply_sleek_grid(
            axis=secondary_axis,
            x_values=primary_x,
            y_values=[item[1] for item in secondary_drawn],
            apply_x=False,
        )
        if secondary_drawn:
            secondary_color = secondary_drawn[0][3]
            secondary_axis.tick_params(axis="y", colors=secondary_color)
            secondary_axis.yaxis.label.set_color(secondary_color)

        if critical_labels:
            seed = 0
            for x_values, y_values, label, color in primary_drawn:
                seed = self._annotate_series_critical_points(
                    x_values=x_values,
                    y_values=y_values,
                    color=color,
                    series_label=label if len(primary_drawn) > 1 else None,
                    include_zero_crossings=True,
                    max_zero_crossings=1,
                    seed=seed,
                    axis=self.axes,
                )
            for x_values, y_values, label, color in secondary_drawn:
                seed = self._annotate_series_critical_points(
                    x_values=x_values,
                    y_values=y_values,
                    color=color,
                    series_label=label if len(secondary_drawn) > 1 else None,
                    include_zero_crossings=True,
                    max_zero_crossings=1,
                    seed=seed,
                    axis=secondary_axis,
                )
            if len(secondary_drawn) > 1:
                secondary_x, secondary_y, _, _ = secondary_drawn[0]
                for index in range(1, len(secondary_drawn)):
                    x_values, y_values, label, _ = secondary_drawn[index]
                    seed = self._annotate_intersections_between_series(
                        x_values=secondary_x,
                        y_primary=secondary_y,
                        y_secondary=y_values,
                        color="#555555",
                        label=f"int {label}",
                        max_points=1,
                        seed=seed,
                        x_secondary=x_values,
                        axis=secondary_axis,
                    )

        self._render_load_markers(load_markers)
        if legend_lines:
            self.axes.legend(legend_lines, legend_labels, loc="best", fontsize=8)
        self._cache_rendered_series(
            title=title,
            xlabel=xlabel,
            ylabel=primary_ylabel,
            series=plotted,
            series_ylabels=(
                [primary_ylabel for _ in primary_series]
                + [secondary_ylabel for _ in secondary_series]
            ),
        )
        self.request_draw_idle()

    def render_custom_chart(
        self,
        *,
        selections: Sequence[CustomChartSelection],
        title: str,
        source_series: Sequence[RenderedSeries] | None = None,
    ) -> None:
        series_pool = list(source_series) if source_series is not None else self._rendered_series
        x_values, _x_label, _family, resampled = build_resampled_series(
            source_series=series_pool,
            selections=selections,
        )

        self._reset_axes()
        axis_map: dict[str, object] = {"L1": self.axes}
        first_series_for_axis: dict[str, RenderedSeries] = {}
        rendered_by_axis: dict[str, list[tuple[list[float], list[float], str, str]]] = {}
        lines = []

        def ensure_axis(target: str):
            if target in axis_map:
                return axis_map[target]
            if target == "R1":
                axis_map[target] = self.axes.twinx()
            elif target == "R2":
                axis = self.axes.twinx()
                axis.spines["right"].set_position(("outward", 55))
                axis_map[target] = axis
            elif target == "L2":
                axis = self.axes.twinx()
                axis.spines["left"].set_position(("outward", 55))
                axis.spines["left"].set_visible(True)
                axis.spines["right"].set_visible(False)
                axis.yaxis.set_label_position("left")
                axis.yaxis.tick_left()
                axis_map[target] = axis
            else:
                axis_map[target] = self.axes
            return axis_map[target]

        for index, item in enumerate(resampled):
            axis = ensure_axis(item.selection.axis_target)
            style: dict[str, object] = {"linewidth": 2.1, "color": custom_chart_color(index)}
            if item.series.linestyle_hint:
                style["linestyle"] = item.series.linestyle_hint
            line = axis.plot(x_values, item.y_values, label=item.selection.legend_label, **style)[0]
            lines.append(line)
            first_series_for_axis.setdefault(item.selection.axis_target, item.series)
            rendered_by_axis.setdefault(item.selection.axis_target, []).append(
                (
                    list(x_values),
                    list(item.y_values),
                    item.selection.legend_label,
                    line.get_color() or "#1f77b4",
                )
            )

        self.axes.set_title(title, pad=10)
        if resampled:
            self.axes.set_xlabel(axis_label_with_unit(resampled[0].series.x_label, resampled[0].series.x_unit))
        self.axes.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.4)
        for target, series in first_series_for_axis.items():
            axis = ensure_axis(target)
            axis.set_ylabel(axis_label_with_unit(series.y_label, series.y_unit))
            axis_series = rendered_by_axis.get(target, [])
            self._set_axis_limits_series_on_axis(
                axis,
                [(x_axis, y_axis, label) for x_axis, y_axis, label, _ in axis_series],
            )
            self._apply_sleek_grid(
                axis=axis,
                x_values=list(x_values),
                y_values=[y_values for _, y_values, _, _ in axis_series],
                apply_x=target == "L1",
            )
            if target != "L1":
                color = series.color_hint or "#222222"
                axis.tick_params(axis="y", colors=color)
                axis.yaxis.label.set_color(color)
        seed = 0
        for target, series_items in rendered_by_axis.items():
            axis = ensure_axis(target)
            for x_series, y_series, label, color in series_items:
                seed = self._annotate_series_critical_points(
                    x_values=x_series,
                    y_values=y_series,
                    color=color,
                    series_label=label,
                    include_zero_crossings=True,
                    max_zero_crossings=2,
                    seed=seed,
                    axis=axis,
                    detailed_labels=True,
                )
            if len(series_items) > 1:
                primary_x, primary_y, _, _ = series_items[0]
                for index in range(1, len(series_items)):
                    x_series, y_series, label, _ = series_items[index]
                    seed = self._annotate_intersections_between_series(
                        x_values=primary_x,
                        y_primary=primary_y,
                        y_secondary=y_series,
                        color="#555555",
                        label=f"int {label}",
                        max_points=1,
                        seed=seed,
                        x_secondary=x_series,
                        axis=axis,
                        detailed_labels=True,
                    )
        if lines:
            self.axes.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)
        self.request_draw_idle()

    def _reset_axes(self) -> None:
        self.clear_footer_texts()
        for axis in list(self.figure.axes):
            if hasattr(axis, "_boef_label_positions_px"):
                setattr(axis, "_boef_label_positions_px", [])
            if axis is self.axes:
                axis.clear()
            else:
                axis.remove()
        if self.axes not in self.figure.axes:
            self.axes = self.figure.add_subplot(111)

    @staticmethod
    def _sample_load_markers(markers: Sequence[LoadMarker], *, max_count: int = 8) -> list[LoadMarker]:
        ordered = sorted(markers, key=lambda item: item.x_m)
        if len(ordered) <= max_count:
            return ordered
        if max_count <= 2:
            return [ordered[0], ordered[-1]]
        kept_indices: list[int] = [0, len(ordered) - 1]
        slots = max_count - 2
        for index in range(1, slots + 1):
            candidate = round(index * (len(ordered) - 1) / (slots + 1))
            if candidate not in kept_indices:
                kept_indices.append(candidate)
        kept_indices.sort()
        return [ordered[index] for index in kept_indices]

    def _render_load_markers(self, load_markers: Sequence[LoadMarker] | None) -> None:
        if not load_markers:
            return
        x_limits = self.axes.get_xlim()
        if len(x_limits) != 2:
            return
        x_min = min(float(x_limits[0]), float(x_limits[1]))
        x_max = max(float(x_limits[0]), float(x_limits[1]))
        span = x_max - x_min
        tolerance = 0.01 * span if span > 0.0 else 0.0
        visible = [
            marker
            for marker in load_markers
            if math.isfinite(marker.x_m)
            and math.isfinite(marker.load_kn)
            and marker.x_m >= (x_min - tolerance)
            and marker.x_m <= (x_max + tolerance)
        ]
        if not visible:
            return
        markers = self._sample_load_markers(visible, max_count=8)
        x_mid = 0.5 * (x_min + x_max)
        font_size = max(6.5, min(8.2, self._critical_label_font_size() - 0.1))
        for index, marker in enumerate(markers):
            x_value = float(marker.x_m)
            self.axes.annotate(
                "",
                xy=(x_value, 0.79),
                xytext=(x_value, 0.93),
                xycoords=("data", "axes fraction"),
                textcoords=("data", "axes fraction"),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#d94841",
                    "lw": 1.5,
                    "alpha": 0.9,
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                },
            )
            base_x_px = float(self.axes.transData.transform((x_value, 0.0))[0])
            base_y_px = float(self.axes.transAxes.transform((0.0, 0.93))[1])
            if x_value <= x_mid:
                offsets = [(26, -2), (26, -12), (26, -22), (36, -6), (14, -6), (26, 8), (34, -30), (16, -30)]
            else:
                offsets = [(-26, -2), (-26, -12), (-26, -22), (-36, -6), (-14, -6), (-26, 8), (-34, -30), (-16, -30)]
            text_offset_x, text_offset_y = self._choose_label_offset(
                axis=self.axes,
                base_display_xy=(base_x_px, base_y_px),
                offsets=offsets,
                seed=index,
                min_clearance_px=26.0,
            )
            text_artist = self.axes.annotate(
                marker.label,
                xy=(x_value, 0.93),
                xycoords=("data", "axes fraction"),
                xytext=(text_offset_x, text_offset_y),
                textcoords="offset points",
                ha="left" if text_offset_x > 0 else "right",
                va="top",
                fontsize=font_size,
                color="#4a4a4a",
                alpha=0.9,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#7a7a7a",
                    "lw": 0.7,
                    "alpha": 0.55,
                },
                bbox={
                    "facecolor": "white",
                    "alpha": 0.42,
                    "edgecolor": "none",
                    "boxstyle": "round,pad=0.12",
                },
                clip_on=True,
            )
            self._enable_annotation_drag(text_artist)

    @staticmethod
    def _axis_label_position_cache(axis: object) -> list[tuple[float, float]]:
        cache = getattr(axis, "_boef_label_positions_px", None)
        if isinstance(cache, list):
            return cache
        cache = []
        setattr(axis, "_boef_label_positions_px", cache)
        return cache

    def _choose_label_offset(
        self,
        *,
        axis: object,
        base_display_xy: tuple[float, float],
        offsets: Sequence[tuple[int, int]],
        seed: int,
        min_clearance_px: float,
    ) -> tuple[int, int]:
        if not offsets:
            return 0, 0
        cache = self._axis_label_position_cache(axis)
        ordered_offsets = list(offsets)
        seed_index = seed % len(ordered_offsets)
        ordered_offsets = ordered_offsets[seed_index:] + ordered_offsets[:seed_index]

        best_offset = ordered_offsets[0]
        best_distance_sq = -1.0
        min_clearance_sq = min_clearance_px * min_clearance_px
        for x_offset, y_offset in ordered_offsets:
            candidate_x = base_display_xy[0] + float(x_offset)
            candidate_y = base_display_xy[1] + float(y_offset)
            if cache:
                min_distance_sq = min(
                    ((candidate_x - used_x) ** 2) + ((candidate_y - used_y) ** 2)
                    for used_x, used_y in cache
                )
            else:
                min_distance_sq = float("inf")
            if min_distance_sq >= min_clearance_sq:
                cache.append((candidate_x, candidate_y))
                return x_offset, y_offset
            if min_distance_sq > best_distance_sq:
                best_distance_sq = min_distance_sq
                best_offset = (x_offset, y_offset)

        cache.append((base_display_xy[0] + float(best_offset[0]), base_display_xy[1] + float(best_offset[1])))
        return best_offset

    @staticmethod
    def _enable_annotation_drag(annotation: object) -> None:
        drag_fn = getattr(annotation, "draggable", None)
        if not callable(drag_fn):
            return
        try:
            drag_fn(use_blit=True)
        except TypeError:
            try:
                drag_fn()
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _series_span(values: Sequence[float]) -> float | None:
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if len(finite) < 2:
            return None
        span = max(finite) - min(finite)
        return span if span > 0.0 else None

    @staticmethod
    def _nice_step(value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            return 1.0
        exponent = math.floor(math.log10(value))
        fraction = value / (10.0 ** exponent)
        if fraction <= 1.0:
            base = 1.0
        elif fraction <= 2.0:
            base = 2.0
        elif fraction <= 5.0:
            base = 5.0
        else:
            base = 10.0
        return base * (10.0 ** exponent)

    def _estimate_minor_step(self, span: float | None) -> float | None:
        if span is None or span <= 0.0:
            return None
        if span >= 8.0 and span <= 300.0:
            return 1.0
        if span < 8.0:
            return self._nice_step(span / 8.0)
        return self._nice_step(span / 300.0)

    def _apply_sleek_grid(
        self,
        *,
        axis: object,
        x_values: Sequence[float],
        y_values: Sequence[Sequence[float]],
        apply_x: bool = True,
    ) -> None:
        from matplotlib.ticker import AutoMinorLocator, MultipleLocator

        x_span = self._series_span(x_values)
        y_span_candidates = [self._series_span(series) for series in y_values]
        y_span = max((span for span in y_span_candidates if span is not None), default=None)

        if apply_x:
            x_step = self._estimate_minor_step(x_span)
            if x_step is not None:
                axis.xaxis.set_minor_locator(MultipleLocator(x_step))
            else:
                axis.xaxis.set_minor_locator(AutoMinorLocator(5))
        y_step = self._estimate_minor_step(y_span)
        if y_step is not None:
            axis.yaxis.set_minor_locator(MultipleLocator(y_step))
        else:
            axis.yaxis.set_minor_locator(AutoMinorLocator(5))

        axis.grid(True, linestyle="-", linewidth=0.45, alpha=0.18)
        axis.grid(True, which="minor", linestyle="-", linewidth=0.28, alpha=0.10)
        axis.tick_params(axis="both", labelsize=9)
        axis.tick_params(axis="both", which="minor", length=2.0, width=0.35)

    def _critical_label_font_size(self) -> float:
        min_dim = max(1, min(self.canvas.width(), self.canvas.height()))
        return max(6.0, min(8.5, min_dim / 90.0))

    @staticmethod
    def _format_critical_value(value: float) -> str:
        if not math.isfinite(value):
            return "nan"
        magnitude = abs(value)
        if magnitude >= 1000.0 or (0.0 < magnitude < 0.01):
            return f"{value:.2e}"
        if magnitude >= 100.0:
            return f"{value:.1f}"
        if magnitude >= 1.0:
            return f"{value:.2f}"
        return f"{value:.3f}"

    @staticmethod
    def _compact_series_label(label: str | None) -> str:
        if not label:
            return ""
        cleaned = label.strip()
        if len(cleaned) <= 10:
            return cleaned
        return f"{cleaned[:9]}."

    @staticmethod
    def _find_zero_crossings(x_values: Sequence[float], y_values: Sequence[float]) -> list[float]:
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return []
        zero_crossings: list[float] = []
        for idx in range(len(x_values) - 1):
            x0 = float(x_values[idx])
            x1 = float(x_values[idx + 1])
            y0 = float(y_values[idx])
            y1 = float(y_values[idx + 1])
            if not math.isfinite(x0) or not math.isfinite(x1) or not math.isfinite(y0) or not math.isfinite(y1):
                continue
            if math.isclose(y0, 0.0, abs_tol=1e-12):
                zero_crossings.append(x0)
            if y0 * y1 < 0.0:
                t = -y0 / (y1 - y0)
                zero_crossings.append(x0 + t * (x1 - x0))
            elif math.isclose(y1, 0.0, abs_tol=1e-12):
                zero_crossings.append(x1)
        deduped: list[float] = []
        for value in zero_crossings:
            if not any(math.isclose(value, existing, rel_tol=0.0, abs_tol=1e-9) for existing in deduped):
                deduped.append(value)
        return deduped

    @staticmethod
    def _find_intersections(
        x_primary: Sequence[float],
        y_primary: Sequence[float],
        x_secondary: Sequence[float],
        y_secondary: Sequence[float],
    ) -> list[float]:
        if (
            len(x_primary) != len(y_primary)
            or len(x_secondary) != len(y_secondary)
            or len(x_primary) != len(x_secondary)
            or len(x_primary) < 2
        ):
            return []
        for xp, xs in zip(x_primary, x_secondary):
            if not math.isclose(float(xp), float(xs), rel_tol=1e-7, abs_tol=1e-9):
                return []
        intersections: list[float] = []
        for idx in range(len(x_primary) - 1):
            x0 = float(x_primary[idx])
            x1 = float(x_primary[idx + 1])
            d0 = float(y_primary[idx]) - float(y_secondary[idx])
            d1 = float(y_primary[idx + 1]) - float(y_secondary[idx + 1])
            if not math.isfinite(d0) or not math.isfinite(d1):
                continue
            if math.isclose(d0, 0.0, abs_tol=1e-12):
                intersections.append(x0)
            if d0 * d1 < 0.0:
                t = -d0 / (d1 - d0)
                intersections.append(x0 + t * (x1 - x0))
            elif math.isclose(d1, 0.0, abs_tol=1e-12):
                intersections.append(x1)
        deduped: list[float] = []
        for value in intersections:
            if not any(math.isclose(value, existing, rel_tol=0.0, abs_tol=1e-9) for existing in deduped):
                deduped.append(value)
        return deduped

    @staticmethod
    def _interpolate_y_at_x(x_values: Sequence[float], y_values: Sequence[float], x_target: float) -> float:
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        for idx in range(len(x_values) - 1):
            x0 = float(x_values[idx])
            x1 = float(x_values[idx + 1])
            if (x0 <= x_target <= x1) or (x1 <= x_target <= x0):
                y0 = float(y_values[idx])
                y1 = float(y_values[idx + 1])
                if math.isclose(x1, x0, abs_tol=1e-12):
                    return y0
                t = (x_target - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return float(y_values[-1])

    def _annotate_point(
        self,
        *,
        x_value: float,
        y_value: float,
        text: str,
        color: str,
        seed: int,
        axis: object | None = None,
    ) -> None:
        offsets = [
            (8, 8),
            (8, -12),
            (-62, 8),
            (-62, -12),
            (10, 20),
            (-64, 20),
            (10, -24),
            (-64, -24),
            (0, 26),
            (0, -28),
            (18, 4),
            (-70, 4),
        ]
        font_size = self._critical_label_font_size()
        marker_size = max(2.0, min(4.2, font_size * 0.55))
        target_axis = axis if axis is not None else self.axes
        base_x_px, base_y_px = target_axis.transData.transform((x_value, y_value))
        x_offset, y_offset = self._choose_label_offset(
            axis=target_axis,
            base_display_xy=(float(base_x_px), float(base_y_px)),
            offsets=offsets,
            seed=seed,
            min_clearance_px=32.0,
        )
        target_axis.scatter(
            [x_value],
            [y_value],
            s=marker_size * marker_size,
            color=color,
            zorder=6,
            alpha=0.85,
        )
        text_artist = target_axis.annotate(
            text,
            xy=(x_value, y_value),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=font_size,
            color=color,
            alpha=0.92,
            ha="left" if x_offset >= 0 else "right",
            va="bottom" if y_offset >= 0 else "top",
            bbox={
                "facecolor": "white",
                "alpha": 0.45,
                "edgecolor": "none",
                "boxstyle": "round,pad=0.15",
            },
            clip_on=True,
        )
        self._enable_annotation_drag(text_artist)

    def _annotate_series_critical_points(
        self,
        *,
        x_values: Sequence[float],
        y_values: Sequence[float],
        color: str,
        series_label: str | None,
        include_zero_crossings: bool,
        max_zero_crossings: int,
        seed: int,
        axis: object | None = None,
        detailed_labels: bool = False,
    ) -> int:
        if len(x_values) != len(y_values) or not x_values:
            return seed
        prefix = self._compact_series_label(series_label)
        detailed_prefix = series_label.strip() if series_label else ""
        prefix_text = f"{prefix} " if prefix else ""
        detailed_prefix_text = f"{detailed_prefix} " if detailed_prefix else ""
        max_index = max(range(len(y_values)), key=lambda i: y_values[i])
        min_index = min(range(len(y_values)), key=lambda i: y_values[i])
        max_x = float(x_values[max_index])
        max_y = float(y_values[max_index])
        min_x = float(x_values[min_index])
        min_y = float(y_values[min_index])
        max_text = (
            f"{detailed_prefix_text}max\nx={self._format_critical_value(max_x)}, y={self._format_critical_value(max_y)}"
            if detailed_labels
            else f"{prefix_text}max {self._format_critical_value(max_y)} @ {self._format_critical_value(max_x)}"
        )
        critical_points = [
            (
                max_x,
                max_y,
                max_text,
            )
        ]
        if min_index != max_index:
            min_text = (
                f"{detailed_prefix_text}min\nx={self._format_critical_value(min_x)}, y={self._format_critical_value(min_y)}"
                if detailed_labels
                else f"{prefix_text}min {self._format_critical_value(min_y)} @ {self._format_critical_value(min_x)}"
            )
            critical_points.append(
                (
                    min_x,
                    min_y,
                    min_text,
                )
            )
        if include_zero_crossings:
            zero_crossings = self._find_zero_crossings(x_values, y_values)
            if zero_crossings:
                center = 0.5 * (float(min(x_values)) + float(max(x_values)))
                zero_crossings.sort(key=lambda value: abs(value - center))
                for x_zero in zero_crossings[: max(0, max_zero_crossings)]:
                    zero_text = (
                        f"{detailed_prefix_text}x0\nx={self._format_critical_value(float(x_zero))}, y=0"
                        if detailed_labels
                        else f"{prefix_text}x0 {self._format_critical_value(float(x_zero))}"
                    )
                    critical_points.append(
                        (
                            float(x_zero),
                            0.0,
                            zero_text,
                        )
                    )
        for x_point, y_point, text in critical_points:
            self._annotate_point(
                x_value=x_point,
                y_value=y_point,
                text=text,
                color=color,
                seed=seed,
                axis=axis,
            )
            seed += 1
        return seed

    def _annotate_intersections_between_series(
        self,
        *,
        x_values: Sequence[float],
        y_primary: Sequence[float],
        y_secondary: Sequence[float],
        color: str,
        label: str,
        max_points: int,
        seed: int,
        x_secondary: Sequence[float] | None = None,
        axis: object | None = None,
        detailed_labels: bool = False,
    ) -> int:
        secondary_x = x_secondary if x_secondary is not None else x_values
        intersections = self._find_intersections(
            x_values,
            y_primary,
            secondary_x,
            y_secondary,
        )
        if not intersections:
            return seed
        center = 0.5 * (float(min(x_values)) + float(max(x_values)))
        intersections.sort(key=lambda value: abs(value - center))
        for x_value in intersections[: max(0, max_points)]:
            y_value = self._interpolate_y_at_x(x_values, y_primary, x_value)
            text = (
                f"{label}\nx={self._format_critical_value(x_value)}, y={self._format_critical_value(y_value)}"
                if detailed_labels
                else f"{label} {self._format_critical_value(x_value)}"
            )
            self._annotate_point(
                x_value=x_value,
                y_value=y_value,
                text=text,
                color=color,
                seed=seed,
                axis=axis,
            )
            seed += 1
        return seed

    @staticmethod
    def _split_axis_label(label: str) -> tuple[str, str]:
        cleaned = label.strip()
        if "(" not in cleaned or ")" not in cleaned:
            return cleaned, ""
        open_idx = cleaned.rfind("(")
        close_idx = cleaned.rfind(")")
        if open_idx == -1 or close_idx < open_idx:
            return cleaned, ""
        base = cleaned[:open_idx].strip()
        unit = cleaned[open_idx + 1 : close_idx].strip()
        return base or cleaned, unit

    def _infer_axis_family(self, *, title: str, xlabel: str) -> ChartAxisFamily:
        x_lower = xlabel.lower()
        title_lower = title.lower()
        chart_id = self._chart_id.lower()
        if "f (" in x_lower or x_lower.startswith("f "):
            return ChartAxisFamily.FREQUENCY
        if "t (" in x_lower or x_lower.startswith("t "):
            return ChartAxisFamily.TIME
        if "xi" in x_lower or "ξ" in xlabel or chart_id.startswith("dynamic_"):
            if "time" not in chart_id and "fft" not in chart_id and "psd" not in chart_id and "impedance" not in chart_id:
                return ChartAxisFamily.DYNAMIC_XI
        if "sleeper position" in x_lower or chart_id in {"sleeper", "pressure"}:
            return ChartAxisFamily.SLEEPER_SPATIAL
        if "transition" in chart_id or "transition" in title_lower:
            return ChartAxisFamily.TRANSITION_SPATIAL
        return ChartAxisFamily.STATIC_SPATIAL

    def _cache_rendered_series(
        self,
        *,
        title: str,
        xlabel: str,
        ylabel: str,
        series: Sequence[tuple[str, str, list[float], list[float], str | None, str | None]],
        series_ylabels: Sequence[str] | None = None,
    ) -> None:
        x_label_name, x_unit = self._split_axis_label(xlabel)
        y_label_name, y_unit = self._split_axis_label(ylabel)
        family = self._infer_axis_family(title=title, xlabel=xlabel)
        cached: list[RenderedSeries] = []
        for index, (series_id, label, x_values, y_values, color, linestyle) in enumerate(series):
            series_y_label = y_label_name
            series_y_unit = y_unit
            if series_ylabels is not None and index < len(series_ylabels):
                series_y_label, series_y_unit = self._split_axis_label(series_ylabels[index])
            cached.append(
                RenderedSeries(
                    series_id=series_id,
                    source_chart_id=self._chart_title,
                    label=label,
                    x=list(x_values),
                    y=list(y_values),
                    x_label=x_label_name,
                    y_label=series_y_label,
                    x_unit=x_unit,
                    y_unit=series_y_unit,
                    axis_family=family,
                    color_hint=color,
                    linestyle_hint=linestyle,
                )
            )
        self._rendered_series = cached

    def _set_axis_limits(self, x: list[float], y: list[float]) -> None:
        if not x or not y:
            return
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)
        x_pad = 1.0 if math.isclose(x_min, x_max) else 0.05 * (x_max - x_min)
        y_pad = 1.0 if math.isclose(y_min, y_max) else 0.05 * (y_max - y_min)
        self.axes.set_xlim(x_min - x_pad, x_max + x_pad)
        self.axes.set_ylim(y_min - y_pad, y_max + y_pad)

    def _set_axis_limits_multi(self, x: list[float], y_series: list[list[float]]) -> None:
        if not x or not y_series or any(not series for series in y_series):
            return
        x_min = min(x)
        x_max = max(x)
        y_min = min(min(series) for series in y_series)
        y_max = max(max(series) for series in y_series)
        x_pad = 1.0 if math.isclose(x_min, x_max) else 0.05 * (x_max - x_min)
        y_pad = 1.0 if math.isclose(y_min, y_max) else 0.05 * (y_max - y_min)
        self.axes.set_xlim(x_min - x_pad, x_max + x_pad)
        self.axes.set_ylim(y_min - y_pad, y_max + y_pad)

    def _set_axis_limits_series(
        self, series: Sequence[tuple[list[float], list[float], str]]
    ) -> None:
        self._set_axis_limits_series_on_axis(self.axes, series)

    @staticmethod
    def _set_axis_limits_series_on_axis(
        axis: object,
        series: Sequence[tuple[list[float], list[float], str]],
    ) -> None:
        if not series or any(not x_values or not y_values for x_values, y_values, _ in series):
            return
        x_min = min(min(x_values) for x_values, _, _ in series)
        x_max = max(max(x_values) for x_values, _, _ in series)
        y_min = min(min(y_values) for _, y_values, _ in series)
        y_max = max(max(y_values) for _, y_values, _ in series)
        x_pad = 1.0 if math.isclose(x_min, x_max) else 0.05 * (x_max - x_min)
        y_pad = 1.0 if math.isclose(y_min, y_max) else 0.05 * (y_max - y_min)
        axis.set_xlim(x_min - x_pad, x_max + x_pad)
        axis.set_ylim(y_min - y_pad, y_max + y_pad)


@dataclass(frozen=True)
class ChartRegistryEntry:
    chart_id: str
    title: str
    tab_index: int
    plot_panel: PlotPanel


class ChartTile(QWidget):
    clicked = Signal(str)

    def __init__(self, chart_id: str, title: str) -> None:
        super().__init__()
        self.chart_id = chart_id
        self._pixmap: QPixmap | None = None

        self.setObjectName("chartTile")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.title_label.setStyleSheet("font-weight: 600; font-size: 11px;")

        self.image_label = QLabel("Not produced in this mode")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: #666666;")
        self.image_label.setMinimumHeight(140)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, stretch=1)

        self.setStyleSheet(
            "#chartTile { background: #fafafa; border: 1px solid #dddddd; border-radius: 6px; }"
        )

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        self._pixmap = pixmap
        if pixmap is None or pixmap.isNull():
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Not produced in this mode")
            self.image_label.setStyleSheet("color: #666666;")
        else:
            self.image_label.setText("")
            self.image_label.setStyleSheet("")
            self._update_scaled_pixmap()

    def set_placeholder(self, text: str) -> None:
        self._pixmap = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(text)
        self.image_label.setStyleSheet("color: #666666;")

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap is None:
            return
        target_size = self.image_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        scaled = self._pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        if self._pixmap is not None:
            self._update_scaled_pixmap()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.chart_id)
        super().mousePressEvent(event)


def _summary_value_label() -> QLabel:
    label = QLabel("—")
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    label.setWordWrap(True)
    label.setMinimumWidth(160)
    return label


def _add_summary_section(
    root: QVBoxLayout,
    title: str,
    rows: Sequence[tuple[str, str]],
    fields: dict[str, QLabel],
) -> None:
    group = QGroupBox(title)
    group.setStyleSheet(
        "QGroupBox { font-weight: 600; margin-top: 10px; } "
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
    )
    grid = QGridLayout(group)
    grid.setContentsMargins(10, 14, 10, 10)
    grid.setHorizontalSpacing(18)
    grid.setVerticalSpacing(6)
    metric_header = QLabel("Item")
    metric_header.setStyleSheet("font-weight: 600; color: #555555;")
    value_header = QLabel("Value")
    value_header.setStyleSheet("font-weight: 600; color: #555555;")
    grid.addWidget(metric_header, 0, 0)
    grid.addWidget(value_header, 0, 1)
    for row_index, (key, label_text) in enumerate(rows, start=1):
        metric = QLabel(label_text)
        metric.setWordWrap(True)
        value = _summary_value_label()
        grid.addWidget(metric, row_index, 0)
        grid.addWidget(value, row_index, 1)
        fields[key] = value
    grid.setColumnStretch(0, 2)
    grid.setColumnStretch(1, 1)
    root.addWidget(group)


def _build_as5100_envelope_summary_payload(
    result: EnvelopeResult,
    load_source: dict[str, object] | None,
) -> dict[str, object] | None:
    if not load_source or load_source.get("source_type") != "as5100_fixed_rail":
        return None
    summary = result.summary
    design = summary.design_summary
    arrangement = str(load_source.get("arrangement", "fixed_user_selected"))
    payload: dict[str, object] = {
        "model": str(load_source.get("model", "AS5100")),
        "standard": str(load_source.get("standard", "AS5100.2:2017")),
        "arrangement": arrangement,
        "reference_position_m": float(load_source.get("reference_position_m", 0.0) or 0.0),
        "group_count": int(load_source.get("group_count", 0) or 0),
        "group_spacing_m": float(load_source.get("group_spacing_m", 0.0) or 0.0),
        "axle_count": int(load_source.get("axle_count", 0) or 0),
        "max_axle_load_kn": n_to_kn(float(load_source.get("max_axle_load_n", 0.0) or 0.0)),
        "max_wheel_load_kn_per_rail": n_to_kn(
            float(load_source.get("max_wheel_load_n_per_rail", 0.0) or 0.0)
        ),
        "max_abs_deflection_mm": m_to_mm(abs(summary.max_deflection.value)),
        "max_abs_moment_kn_m": abs(summary.max_moment.value) / 1000.0,
        "max_abs_shear_kn": n_to_kn(abs(summary.max_shear.value)),
        "max_abs_reaction_kn_per_m": n_to_kn(abs(summary.max_reaction.value)),
        "max_ballast_pressure_kpa": pa_to_kpa(summary.max_ballast_pressure.value),
        "max_rail_base_stress_mpa": pa_to_mpa(summary.max_rail_base_stress_pa),
        "daf": design.daf if design is not None else None,
    }
    if arrangement == "governing_envelope_sweep":
        payload.update(
            {
                "selected_group_count": int(load_source.get("selected_group_count", 0) or 0),
                "selected_group_spacing_m": float(
                    load_source.get("selected_group_spacing_m", 0.0) or 0.0
                ),
                "sweep_candidate_count": int(load_source.get("sweep_candidate_count", 0) or 0),
                "governing_metric": str(load_source.get("governing_metric", "max_abs_moment_nm")),
            }
        )
    return payload


def _format_as5100_envelope_summary_text(
    result: EnvelopeResult,
    load_source: dict[str, object] | None,
) -> str:
    payload = _build_as5100_envelope_summary_payload(result, load_source)
    if payload is None:
        return "—"
    model = str(payload["model"])
    standard = str(payload["standard"])
    arrangement = str(payload["arrangement"])
    reference_position = float(payload["reference_position_m"])
    max_axle_kn = float(payload["max_axle_load_kn"])
    max_wheel_kn = float(payload["max_wheel_load_kn_per_rail"])
    max_moment_kn_m = float(payload["max_abs_moment_kn_m"])
    max_deflection_mm = float(payload["max_abs_deflection_mm"])
    daf = payload.get("daf")
    if arrangement == "governing_envelope_sweep":
        return (
            f"{model} governing sweep\n"
            f"{standard}; x0={reference_position:.3f} m\n"
            f"Governing arrangement: {int(payload['group_count'])} group(s) @ "
            f"{float(payload['group_spacing_m']):.2f} m\n"
            f"Selected upper bound: {int(payload.get('selected_group_count', 0) or 0)} group(s) @ "
            f"{float(payload.get('selected_group_spacing_m', 0.0) or 0.0):.2f} m\n"
            f"|M|max {max_moment_kn_m:.3f} kN·m; |w|max {max_deflection_mm:.3f} mm"
            + (f"; DAF {float(daf):.3f}" if daf is not None else "")
            + "\n"
            f"Max axle {max_axle_kn:.0f} kN -> wheel {max_wheel_kn:.0f} kN/rail; "
            f"{int(payload.get('sweep_candidate_count', 0) or 0)} candidates; no automatic DLA applied"
        )
    return (
        f"{model} fixed selected arrangement\n"
        f"{standard}; x0={reference_position:.3f} m\n"
        f"{int(payload['group_count'])} group(s) @ {float(payload['group_spacing_m']):.2f} m\n"
        f"|M|max {max_moment_kn_m:.3f} kN·m; |w|max {max_deflection_mm:.3f} mm"
        + (f"; DAF {float(daf):.3f}" if daf is not None else "")
        + "\n"
        f"{int(payload['axle_count'])} axles; max axle {max_axle_kn:.0f} kN -> wheel "
        f"{max_wheel_kn:.0f} kN/rail; no automatic DLA applied"
    )


class SummaryPanel(QWidget):
    """Summary panel for key analysis outputs."""

    def __init__(self) -> None:
        super().__init__()
        self._fields: dict[str, QLabel] = {}
        self._sanity_fields: dict[str, QLabel] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        _add_summary_section(
            root,
            "Main Rail Response",
            [
                ("deflection", "Max |deflection| (mm) @ x (m)"),
                ("moment", "Max |moment| (kN·m) @ x (m)"),
                ("shear", "Max |shear| (kN) @ x (m)"),
                ("reaction", "Max |rail support reaction| (kN/m) @ x (m)"),
                ("stress", "Max rail base stress (MPa)"),
            ],
            self._fields,
        )
        _add_summary_section(
            root,
            "Track Support And Pressure",
            [
                ("sleeper_load", "Max |sleeper load| (kN) @ x (m)"),
                ("pressure", "Max |sleeper-ballast contact pressure| (kPa) @ x (m)"),
                ("ballast_pressure", "Max ballast pressure, compression (kPa) @ x (m)"),
                ("formation_stress", "Max formation stress by depth (kPa)"),
                ("support_model", "Foundation model"),
                ("support_k_eq", "Equivalent support k (MN/m²)"),
                ("support_pad_force", "Max railpad force (kN/m)"),
                ("support_bed_force", "Max trackbed force (kN/m)"),
                ("support_sleeper_deflection", "Max sleeper deflection (mm)"),
            ],
            self._fields,
        )
        _add_summary_section(
            root,
            "Design Checks",
            [
                ("as5100_summary", "AS5100 envelope summary"),
                ("daf", "Dynamic amplification factor (DAF)"),
                ("bending_stress", "Max bending stress (MPa)"),
                ("bending_allowable", "Admissible repeated bending stress (MPa)"),
                ("bending_allowable_incidental", "Admissible incidental bending stress (MPa)"),
                ("bending_pass", "Bending check"),
                ("shear_stress", "Max shear stress (MPa)"),
                ("shear_allowable", "Admissible repeated rail-head shear stress (MPa)"),
                ("shear_allowable_incidental", "Admissible incidental rail-head shear stress (MPa)"),
                ("shear_pass", "Shear check"),
                ("combined_head", "Combined head stress (MPa)"),
                ("combined_foot", "Combined foot stress (MPa)"),
                ("influence_stress", "Discrete support stress (MPa)"),
            ],
            self._fields,
        )
        _add_summary_section(
            root,
            "Model Parameters And Checks",
            [
                ("beta", "Beam parameter β (1/m)"),
                ("x1", "Zero moment distance x₁ (m)"),
                ("x2", "Contraflexure distance x₂ (m)"),
                ("combined_deflection", "Combined max |w| (mm)"),
                ("combined_moment", "Combined max |M| (kN·m)"),
                ("combined_shear", "Combined max |V| (kN)"),
                ("combined_reaction", "Combined max R_support(x) (kN/m)"),
            ],
            self._fields,
        )
        _add_summary_section(
            root,
            "Sanity Checks",
            [
                ("equilibrium", "Equilibrium error"),
                ("symmetry", "Symmetry error"),
                ("moment_coherence", "Moment/curvature coherence"),
                ("shear_coherence", "Shear/gradient coherence"),
            ],
            self._sanity_fields,
        )
        root.addStretch(1)

    def _add_field(self, layout: QFormLayout, key: str, label: str) -> None:
        value_label = QLabel("—")
        value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addRow(label, value_label)
        self._fields[key] = value_label

    def _add_sanity_field(self, layout: QFormLayout, key: str, label: str) -> None:
        value_label = QLabel("—")
        value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addRow(label, value_label)
        self._sanity_fields[key] = value_label

    def update_sanity_checks(self, checks: dict[str, tuple[str, bool | None]] | None) -> None:
        if not checks:
            for label in self._sanity_fields.values():
                label.setText("—")
                label.setStyleSheet("color: #666666;")
            return
        for key, label in self._sanity_fields.items():
            text, ok = checks.get(key, ("—", None))
            if ok is True:
                label.setStyleSheet("color: #2e7d32;")
            elif ok is False:
                label.setStyleSheet("color: #b26a00;")
            else:
                label.setStyleSheet("color: #666666;")
            label.setText(text)

    def update_summary(self, result: AnalysisResult) -> None:
        summary = result.summary
        self._fields["as5100_summary"].setText("—")
        self._fields["beta"].setText(f"{summary.beta_per_m:.6f} 1/m")
        self._fields["x1"].setText(f"{summary.zero_moment_distance_m:.3f} m")
        self._fields["x2"].setText(f"{summary.contraflexure_distance_m:.3f} m")
        self._fields["deflection"].setText(
            f"{m_to_mm(abs(summary.max_deflection.value)):.3f} mm @ {summary.max_deflection.position_m:.3f} m"
        )
        self._fields["moment"].setText(
            f"{abs(summary.max_moment.value) / 1000.0:.3f} kN·m @ {summary.max_moment.position_m:.3f} m"
        )
        self._fields["shear"].setText(
            f"{n_to_kn(abs(summary.max_shear.value)):.3f} kN @ {summary.max_shear.position_m:.3f} m"
        )
        self._fields["reaction"].setText(
            f"{n_to_kn(abs(summary.max_reaction.value)):.3f} kN/m @ {summary.max_reaction.position_m:.3f} m"
        )
        self._fields["sleeper_load"].setText(
            f"{n_to_kn(abs(summary.max_sleeper_load.value)):.3f} kN @ {summary.max_sleeper_load.position_m:.3f} m"
        )
        self._fields["pressure"].setText(
            f"{pa_to_kpa(abs(summary.max_sleeper_pressure.value)):.3f} kPa @ {summary.max_sleeper_pressure.position_m:.3f} m"
        )
        self._fields["ballast_pressure"].setText("—")
        self._fields["formation_stress"].setText("—")
        self._fields["stress"].setText(f"{pa_to_mpa(summary.max_rail_base_stress_pa):.3f} MPa")
        if result.deflection_m:
            max_deflection = max((abs(value) for value in result.deflection_m), default=0.0)
            self._fields["combined_deflection"].setText(f"{m_to_mm(max_deflection):.3f} mm")
        if result.moment_nm:
            max_moment = max((abs(value) for value in result.moment_nm), default=0.0)
            self._fields["combined_moment"].setText(f"{max_moment / 1000.0:.3f} kN·m")
        if result.shear_n:
            max_shear = max((abs(value) for value in result.shear_n), default=0.0)
            self._fields["combined_shear"].setText(f"{n_to_kn(max_shear):.3f} kN")
        if result.reaction_n_per_m:
            max_reaction = max((abs(value) for value in result.reaction_n_per_m), default=0.0)
            self._fields["combined_reaction"].setText(f"{n_to_kn(max_reaction):.3f} kN/m")
        design = summary.design_summary
        if design is None:
            for key in (
                "daf",
                "bending_stress",
                "bending_allowable",
                "bending_allowable_incidental",
                "bending_pass",
                "shear_stress",
                "shear_allowable",
                "shear_allowable_incidental",
                "shear_pass",
                "combined_head",
                "combined_foot",
                "influence_stress",
            ):
                self._fields[key].setText("—")
        else:
            self._fields["daf"].setText(f"{design.daf:.3f}")
            self._fields["bending_stress"].setText(f"{pa_to_mpa(design.max_bending_stress_pa):.1f} MPa")
            self._fields["bending_allowable"].setText(
                f"{pa_to_mpa(design.admissible_bending_stress_pa):.1f} MPa"
            )
            self._fields["bending_allowable_incidental"].setText(
                f"{pa_to_mpa(design.admissible_bending_incidental_pa):.1f} MPa"
            )
            self._fields["bending_pass"].setText("Pass" if design.bending_pass else "Fail")
            self._fields["shear_stress"].setText(f"{pa_to_mpa(design.max_shear_stress_pa):.1f} MPa")
            self._fields["shear_allowable"].setText(
                f"{pa_to_mpa(design.admissible_shear_repeated_pa):.1f} MPa"
            )
            self._fields["shear_allowable_incidental"].setText(
                f"{pa_to_mpa(design.admissible_shear_incidental_pa):.1f} MPa"
            )
            self._fields["shear_pass"].setText("Pass" if design.shear_pass else "Fail")
            self._fields["combined_head"].setText(f"{pa_to_mpa(design.combined_head_stress_pa):.1f} MPa")
            self._fields["combined_foot"].setText(f"{pa_to_mpa(design.combined_foot_stress_pa):.1f} MPa")
            if design.influence_stress_pa is None:
                self._fields["influence_stress"].setText("—")
            else:
                self._fields["influence_stress"].setText(
                    f"{pa_to_mpa(design.influence_stress_pa):.1f} MPa"
                )
            if design.a3902_checks is not None:
                checks = design.a3902_checks
                self._fields["ballast_pressure"].setText(
                    f"{pa_to_kpa(checks.ballast_contact_pressure_pa):.3f} kPa"
                )
                formation_lines: list[str] = []
                if checks.formation_pressure_pa is not None:
                    formation_lines.append(f"Formation: {pa_to_kpa(checks.formation_pressure_pa):.2f} kPa")
                if checks.subgrade_pressure_pa is not None:
                    formation_lines.append(f"Subgrade: {pa_to_kpa(checks.subgrade_pressure_pa):.2f} kPa")
                self._fields["formation_stress"].setText("; ".join(formation_lines) if formation_lines else "—")

        if summary.support_model is None:
            for key in (
                "support_model",
                "support_k_eq",
                "support_pad_force",
                "support_bed_force",
                "support_sleeper_deflection",
            ):
                self._fields[key].setText("—")
        else:
            self._fields["support_model"].setText(summary.support_model)
            if summary.support_k_eq_n_per_m2 is None:
                self._fields["support_k_eq"].setText("—")
            else:
                self._fields["support_k_eq"].setText(
                    f"{n_per_m2_to_mn_per_m2(summary.support_k_eq_n_per_m2):.2f} MN/m²"
                )
            if summary.max_railpad_force_n_per_m is None:
                self._fields["support_pad_force"].setText("—")
            else:
                self._fields["support_pad_force"].setText(
                    f"{n_to_kn(summary.max_railpad_force_n_per_m):.2f} kN/m"
                )
            if summary.max_trackbed_force_n_per_m is None:
                self._fields["support_bed_force"].setText("—")
            else:
                self._fields["support_bed_force"].setText(
                    f"{n_to_kn(summary.max_trackbed_force_n_per_m):.2f} kN/m"
                )
            if summary.max_sleeper_deflection_m is None:
                self._fields["support_sleeper_deflection"].setText("—")
            else:
                self._fields["support_sleeper_deflection"].setText(
                    f"{m_to_mm(summary.max_sleeper_deflection_m):.3f} mm"
                )

    def update_envelope_summary(
        self,
        result: EnvelopeResult,
        *,
        load_source: dict[str, object] | None = None,
    ) -> None:
        summary = result.summary
        self._fields["beta"].setText(f"{summary.beta_per_m:.6f} 1/m")
        self._fields["x1"].setText(f"{summary.zero_moment_distance_m:.3f} m")
        self._fields["x2"].setText(f"{summary.contraflexure_distance_m:.3f} m")
        self._fields["deflection"].setText(
            f"{m_to_mm(abs(summary.max_deflection.value)):.3f} mm @ {summary.max_deflection.position_m:.3f} m"
        )
        self._fields["moment"].setText(
            f"{abs(summary.max_moment.value) / 1000.0:.3f} kN·m @ {summary.max_moment.position_m:.3f} m"
        )
        self._fields["shear"].setText(
            f"{n_to_kn(abs(summary.max_shear.value)):.3f} kN @ {summary.max_shear.position_m:.3f} m"
        )
        self._fields["reaction"].setText(
            f"{n_to_kn(abs(summary.max_reaction.value)):.3f} kN/m @ {summary.max_reaction.position_m:.3f} m"
        )
        self._fields["sleeper_load"].setText(
            f"{n_to_kn(abs(summary.max_sleeper_load.value)):.3f} kN @ {summary.max_sleeper_load.position_m:.3f} m"
        )
        self._fields["pressure"].setText("—")
        self._fields["ballast_pressure"].setText(
            f"{pa_to_kpa(summary.max_ballast_pressure.value):.3f} kPa @ "
            f"{summary.max_ballast_pressure.position_m:.3f} m"
        )
        formation_entries = [
            f"z={depth:.2f} m: {pa_to_kpa(value):.2f} kPa"
            for depth, value in summary.max_formation_stress_by_depth_pa.items()
        ]
        self._fields["formation_stress"].setText("; ".join(formation_entries) if formation_entries else "—")
        self._fields["stress"].setText(f"{pa_to_mpa(summary.max_rail_base_stress_pa):.3f} MPa")

        max_deflection = max(
            max((abs(value) for value in result.deflection_max_m), default=0.0),
            max((abs(value) for value in result.deflection_min_m), default=0.0),
        )
        self._fields["combined_deflection"].setText(f"{m_to_mm(max_deflection):.3f} mm")
        max_moment = max(
            max((abs(value) for value in result.moment_max_nm), default=0.0),
            max((abs(value) for value in result.moment_min_nm), default=0.0),
        )
        self._fields["combined_moment"].setText(f"{max_moment / 1000.0:.3f} kN·m")
        max_shear = max(
            max((abs(value) for value in result.shear_max_n), default=0.0),
            max((abs(value) for value in result.shear_min_n), default=0.0),
        )
        self._fields["combined_shear"].setText(f"{n_to_kn(max_shear):.3f} kN")
        max_reaction = max((abs(value) for value in result.reaction_max_n_per_m), default=0.0)
        self._fields["combined_reaction"].setText(f"{n_to_kn(max_reaction):.3f} kN/m")
        if load_source and load_source.get("source_type") == "as5100_fixed_rail":
            self._fields["as5100_summary"].setText(
                _format_as5100_envelope_summary_text(result, load_source)
            )
        else:
            self._fields["as5100_summary"].setText("—")
        design = summary.design_summary
        if design is None:
            for key in (
                "daf",
                "bending_stress",
                "bending_allowable",
                "bending_allowable_incidental",
                "bending_pass",
                "shear_stress",
                "shear_allowable",
                "shear_allowable_incidental",
                "shear_pass",
                "combined_head",
                "combined_foot",
                "influence_stress",
            ):
                self._fields[key].setText("—")
        else:
            self._fields["daf"].setText(f"{design.daf:.3f}")
            self._fields["bending_stress"].setText(f"{pa_to_mpa(design.max_bending_stress_pa):.1f} MPa")
            self._fields["bending_allowable"].setText(
                f"{pa_to_mpa(design.admissible_bending_stress_pa):.1f} MPa"
            )
            self._fields["bending_allowable_incidental"].setText(
                f"{pa_to_mpa(design.admissible_bending_incidental_pa):.1f} MPa"
            )
            self._fields["bending_pass"].setText("Pass" if design.bending_pass else "Fail")
            self._fields["shear_stress"].setText(f"{pa_to_mpa(design.max_shear_stress_pa):.1f} MPa")
            self._fields["shear_allowable"].setText(
                f"{pa_to_mpa(design.admissible_shear_repeated_pa):.1f} MPa"
            )
            self._fields["shear_allowable_incidental"].setText(
                f"{pa_to_mpa(design.admissible_shear_incidental_pa):.1f} MPa"
            )
            self._fields["shear_pass"].setText("Pass" if design.shear_pass else "Fail")
            self._fields["combined_head"].setText(f"{pa_to_mpa(design.combined_head_stress_pa):.1f} MPa")
            self._fields["combined_foot"].setText(f"{pa_to_mpa(design.combined_foot_stress_pa):.1f} MPa")
            if design.influence_stress_pa is None:
                self._fields["influence_stress"].setText("—")
            else:
                self._fields["influence_stress"].setText(
                    f"{pa_to_mpa(design.influence_stress_pa):.1f} MPa"
                )
            if design.a3902_checks is not None:
                checks = design.a3902_checks
                self._fields["ballast_pressure"].setText(
                    f"{pa_to_kpa(checks.ballast_contact_pressure_pa):.3f} kPa"
                )
                formation_lines: list[str] = []
                if checks.formation_pressure_pa is not None:
                    formation_lines.append(f"Formation: {pa_to_kpa(checks.formation_pressure_pa):.2f} kPa")
                if checks.subgrade_pressure_pa is not None:
                    formation_lines.append(f"Subgrade: {pa_to_kpa(checks.subgrade_pressure_pa):.2f} kPa")
                self._fields["formation_stress"].setText("; ".join(formation_lines) if formation_lines else "—")


class TransitionSummaryPanel(QWidget):
    """Summary panel for transition-zone performance metrics."""

    def __init__(self) -> None:
        super().__init__()
        self._fields: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        _add_summary_section(
            layout,
            "Configuration",
            [
                ("template", "Template"),
                ("preset", "Preset"),
                ("mode", "Run mode"),
                ("profile", "Profile type"),
                ("k1", "k1 (MN/m²)"),
                ("k2", "k2 (MN/m²)"),
                ("transition_length", "Transition length Lt (m)"),
                ("segment_length", "Segment length Lc (m)"),
                ("domain_length", "Domain length (m)"),
            ],
            self._fields,
        )
        _add_summary_section(
            layout,
            "Response And Forces",
            [
                ("delta_ws", "Delta w at sleeper spacing (mm) @ x (m)"),
                ("delta_w1", "Delta w over 1 m (mm) @ x (m)"),
                ("curvature", "Max curvature (1/m) @ x (m)"),
                ("moment", "Max moment (kN·m) @ x (m)"),
                ("gradient", "Max support reaction gradient (kN/m²) @ x (m)"),
                ("sleeper_load", "Max sleeper load (kN) @ x (m)"),
            ],
            self._fields,
        )
        _add_summary_section(
            layout,
            "Energy",
            [
                ("energy", "Bending energy Ub (kJ)"),
                ("energy_total", "Total energy Utotal (kJ)"),
                ("energy_eta", "Eta = foundation / total energy"),
                ("u_total_max", "Max energy density u_total (J/m) @ x (m)"),
                ("du_dx_max", "Max energy gradient |du/dx| (J/m²) @ x (m)"),
            ],
            self._fields,
        )

        self.interpretation_label = QLabel("—")
        self.interpretation_label.setWordWrap(True)
        self.interpretation_label.setStyleSheet("color: #444444;")
        layout.addWidget(self.interpretation_label)
        layout.addStretch(1)

    def _add_field(self, layout: QFormLayout, key: str, label: str) -> None:
        value_label = QLabel("—")
        value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addRow(label, value_label)
        self._fields[key] = value_label

    @staticmethod
    def _format_enum_value(value: object) -> str:
        if isinstance(value, Enum):
            text = value.value
        else:
            text = str(value)
        return text.replace("_", " ").title()

    def update_summary(self, result: TransitionRunResult) -> None:
        metrics = result.metrics
        self._fields["template"].setText(result.template_name or "Custom")
        self._fields["preset"].setText(result.preset_name or "Custom")
        self._fields["mode"].setText(self._format_enum_value(result.mode))
        self._fields["profile"].setText(self._format_enum_value(result.profile_type))
        self._fields["k1"].setText(f"{n_per_m2_to_mn_per_m2(result.k1_n_per_m2):.2f}")
        self._fields["k2"].setText(
            "—" if result.k2_n_per_m2 is None else f"{n_per_m2_to_mn_per_m2(result.k2_n_per_m2):.2f}"
        )
        self._fields["transition_length"].setText(
            "—" if result.transition_length_m is None else f"{result.transition_length_m:.3f}"
        )
        self._fields["segment_length"].setText(
            "—" if result.segment_length_m is None else f"{result.segment_length_m:.3f}"
        )
        self._fields["domain_length"].setText(f"{result.domain_length_m:.3f}")

        self._fields["delta_ws"].setText(
            f"{m_to_mm(metrics.delta_w_s_m):.3f} mm @ {metrics.delta_w_s_position_m:.3f} m"
        )
        self._fields["delta_w1"].setText(
            f"{m_to_mm(metrics.delta_w_1m_m):.3f} mm @ {metrics.delta_w_1m_position_m:.3f} m"
        )
        self._fields["curvature"].setText(
            f"{metrics.curvature_max_per_m:.6e} @ {metrics.curvature_max_position_m:.3f} m"
        )
        self._fields["moment"].setText(
            f"{metrics.moment_max_nm / 1000.0:.3f} kN·m @ {metrics.moment_max_position_m:.3f} m"
        )
        self._fields["energy"].setText(f"{metrics.energy_bending_j / 1000.0:.3f}")
        self._fields["gradient"].setText(
            f"{n_to_kn(metrics.reaction_gradient_max_n_per_m2):.3f} kN/m² @ "
            f"{metrics.reaction_gradient_position_m:.3f} m"
        )
        self._fields["sleeper_load"].setText(
            f"{n_to_kn(metrics.sleeper_load_max_n):.3f} kN @ {metrics.sleeper_load_position_m:.3f} m"
        )
        energy_metrics = result.energy_metrics
        if energy_metrics is None:
            self._fields["energy_total"].setText("—")
            self._fields["energy_eta"].setText("—")
            self._fields["u_total_max"].setText("—")
            self._fields["du_dx_max"].setText("—")
        else:
            self._fields["energy_total"].setText(f"{energy_metrics.energy_total_j / 1000.0:.3f}")
            self._fields["energy_eta"].setText(f"{energy_metrics.energy_partition_eta:.3f}")
            self._fields["u_total_max"].setText(
                f"{energy_metrics.u_total_max_j_per_m:.3f} @ {energy_metrics.u_total_max_position_m:.3f} m"
            )
            self._fields["du_dx_max"].setText(
                f"{energy_metrics.du_dx_max_j_per_m2:.3f} @ {energy_metrics.du_dx_max_position_m:.3f} m"
            )

        interpretation_lines = [
            "Interpretation:",
            f"- Peak Δw(s) at x={metrics.delta_w_s_position_m:.2f} m highlights local geometry sensitivity.",
            f"- Peak |dp/dx| at x={metrics.reaction_gradient_position_m:.2f} m flags abrupt reaction transfer.",
            f"- κ_max and M_max indicate bending severity near x={metrics.moment_max_position_m:.2f} m.",
        ]
        if energy_metrics is not None:
            interpretation_lines.append(
                "- Energy uses elastic strain-energy post-processing: rail bending plus Winkler foundation spring energy."
            )
            if energy_metrics.is_envelope_upper_bound:
                interpretation_lines.append("- Energy metrics are conservative envelope upper-bound values.")
            if energy_metrics.boundary_peak_flag or energy_metrics.boundary_gradient_peak_flag:
                interpretation_lines.append(
                    "- Boundary artifact check: one energy peak occurs at a domain edge; review domain length."
                )
        else:
            interpretation_lines.append(
                "- Energy metrics are available only for Winkler foundation-model transition runs."
            )
        interpretation = "\n".join(interpretation_lines)
        self.interpretation_label.setText(interpretation)


class DynamicSummaryPanel(QWidget):
    """Summary panel for dynamic outputs."""

    def __init__(self) -> None:
        super().__init__()
        self._fields: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        _add_summary_section(
            layout,
            "Rail Response",
            [
                ("deflection", "Max |deflection| (mm) @ xi (m)"),
                ("moment", "Max |moment| (kN·m) @ xi (m)"),
                ("shear", "Max |shear| (kN) @ xi (m)"),
                ("reaction", "Max |rail support reaction| (kN/m) @ xi (m)"),
            ],
            self._fields,
        )
        _add_summary_section(
            layout,
            "Transition Metrics",
            [
                ("transition_fidelity", "Transition fidelity"),
                ("transition_x_ref", "Governing x_ref (m)"),
                ("transition_risk", "Transition risk index"),
                ("transition_speed_ratio", "Critical speed ratio"),
                ("transition_amplification", "Dynamic amplification"),
            ],
            self._fields,
        )
        layout.addStretch(1)

    def _add_field(self, layout: QFormLayout, key: str, label: str) -> None:
        value_label = QLabel("—")
        value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addRow(label, value_label)
        self._fields[key] = value_label

    def update_summary(self, result: DynamicResult | DynamicTransitionResult) -> None:
        dynamic_result = result.representative if isinstance(result, DynamicTransitionResult) else result
        summary = dynamic_result.summary
        self._fields["deflection"].setText(
            f"{m_to_mm(abs(summary.max_deflection.value)):.3f} mm @ {summary.max_deflection.position_m:.3f} m"
        )
        self._fields["moment"].setText(
            f"{abs(summary.max_moment.value) / 1000.0:.3f} kN·m @ {summary.max_moment.position_m:.3f} m"
        )
        self._fields["shear"].setText(
            f"{n_to_kn(abs(summary.max_shear.value)):.3f} kN @ {summary.max_shear.position_m:.3f} m"
        )
        self._fields["reaction"].setText(
            f"{n_to_kn(abs(summary.max_reaction.value)):.3f} kN/m @ {summary.max_reaction.position_m:.3f} m"
        )
        if isinstance(result, DynamicTransitionResult):
            self._fields["transition_fidelity"].setText(result.solver_fidelity.replace("_", " ").title())
            self._fields["transition_x_ref"].setText(f"{result.metrics.governing_x_ref_m:.3f}")
            self._fields["transition_risk"].setText(f"{result.metrics.risk_index:.3f}")
            self._fields["transition_speed_ratio"].setText(f"{result.metrics.critical_speed_ratio:.3f}")
            self._fields["transition_amplification"].setText(
                f"{result.metrics.dynamic_amplification:.3f}"
            )
        else:
            self._fields["transition_fidelity"].setText("—")
            self._fields["transition_x_ref"].setText("—")
            self._fields["transition_risk"].setText("—")
            self._fields["transition_speed_ratio"].setText("—")
            self._fields["transition_amplification"].setText("—")


class DippedJointSummaryPanel(QWidget):
    """Summary panel for dipped joint wheel/rail force outputs."""

    def __init__(self) -> None:
        super().__init__()
        self._fields: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        _add_summary_section(
            layout,
            "Wheel/Rail Forces",
            [
                ("p0", "Static wheel load P0 (kN)"),
                ("p1", "Peak impact force P1 (kN)"),
                ("p2", "Peak secondary force P2 (kN)"),
            ],
            self._fields,
        )
        _add_summary_section(
            layout,
            "Amplification",
            [
                ("daf_p1", "Dynamic amplification factor P1"),
                ("daf_p2", "Dynamic amplification factor P2"),
            ],
            self._fields,
        )
        layout.addStretch(1)

    def _add_field(self, layout: QFormLayout, key: str, label: str) -> None:
        value_label = QLabel("—")
        value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addRow(label, value_label)
        self._fields[key] = value_label

    def update_summary(self, result: DippedJointResult) -> None:
        self._fields["p0"].setText(f"{n_to_kn(result.static_load_n):.3f}")
        self._fields["p1"].setText(f"{n_to_kn(result.p1_n):.3f}")
        self._fields["p2"].setText(f"{n_to_kn(result.p2_n):.3f}")
        self._fields["daf_p1"].setText(f"{result.p1_dynamic_amplification:.2f}")
        self._fields["daf_p2"].setText(f"{result.p2_dynamic_amplification:.2f}")


class SpecialSummaryPanel(QWidget):
    """Summary panel for special analysis outputs."""

    def __init__(self) -> None:
        super().__init__()
        self._fields: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        _add_summary_section(
            layout,
            "Floating Slab Response",
            [
                ("natural_frequency", "Natural frequency fn (Hz)"),
                ("damping_ratio", "Damping ratio zeta"),
                ("static_deflection", "Static deflection delta (mm)"),
            ],
            self._fields,
        )
        layout.addStretch(1)

    def _add_field(self, layout: QFormLayout, key: str, label: str) -> None:
        value_label = QLabel("—")
        value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addRow(label, value_label)
        self._fields[key] = value_label

    def update_summary(self, result: FloatingSlabResult) -> None:
        self._fields["natural_frequency"].setText(f"{result.natural_frequency_hz:.3f}")
        self._fields["damping_ratio"].setText(f"{result.damping_ratio:.3f}")
        self._fields["static_deflection"].setText(f"{m_to_mm(result.static_deflection_m):.3f}")

class AnalysisWorker(QObject):
    """Background worker for analysis computation."""

    finished = Signal(AnalysisResult)
    failed = Signal(str)

    def __init__(self, config: AnalysisConfig, analysis_inputs: AnalysisInputs, mode: AnalysisMode) -> None:
        super().__init__()
        self.config = config
        self.analysis_inputs = analysis_inputs
        self.mode = mode

    def run(self) -> None:
        try:
            result = run_analysis(self.config, mode=self.mode)
        except ValueError as exc:
            self.failed.emit(str(exc))
            return
        except Exception:
            if LOGGER.hasHandlers():
                LOGGER.exception("Analysis failed due to an unexpected error.")
            self.failed.emit(SAFE_ANALYSIS_ERROR_MESSAGE)
            return
        self.finished.emit(result)


class DynamicAnalysisWorker(QObject):
    """Background worker for dynamic analysis computation."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        config: DippedJointConfig | DynamicConfig | DynamicTransitionConfig,
        mode: DynamicMode,
    ) -> None:
        super().__init__()
        self.config = config
        self.mode = mode

    def run(self) -> None:
        try:
            result = run_dynamic_analysis(self.config, mode=self.mode)
        except ValueError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            LOGGER.exception("Dynamic analysis failed due to an unexpected error.")
            detail = f"{exc.__class__.__name__}: {exc}" if str(exc) else exc.__class__.__name__
            self.failed.emit(f"{SAFE_ANALYSIS_ERROR_MESSAGE}\n{detail}\nSee log: {_app_log_path()}")
            return
        self.finished.emit(result)


class SpecialAnalysisWorker(QObject):
    """Background worker for special analysis computation."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config: FloatingSlabConfig, mode: SpecialMode) -> None:
        super().__init__()
        self.config = config
        self.mode = mode

    def run(self) -> None:
        try:
            result = run_special_analysis(self.config, mode=self.mode)
        except ValueError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            LOGGER.exception("Special analysis failed due to an unexpected error.")
            detail = f"{exc.__class__.__name__}: {exc}" if str(exc) else exc.__class__.__name__
            self.failed.emit(f"{SAFE_ANALYSIS_ERROR_MESSAGE}\n{detail}\nSee log: {_app_log_path()}")
            return
        self.finished.emit(result)


class EnvelopeAnalysisWorker(QObject):
    """Background worker for quasi-static envelope computation."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)
    cancelled = Signal()

    def __init__(self, config: EnvelopeConfig) -> None:
        super().__init__()
        self.config = config
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancel_requested(self) -> bool:
        return self._cancel_requested

    def run(self) -> None:
        try:
            result = run_envelope(
                self.config,
                progress_callback=self._emit_progress,
                cancel_callback=self._is_cancel_requested,
            )
        except EnvelopeCancelled:
            self.cancelled.emit()
            return
        except ValueError as exc:
            self.failed.emit(str(exc))
            return
        except Exception:
            if LOGGER.hasHandlers():
                LOGGER.exception("Envelope analysis failed due to an unexpected error.")
            self.failed.emit(SAFE_ANALYSIS_ERROR_MESSAGE)
            return
        self.finished.emit(result)

    def _emit_progress(self, current: int, total: int) -> None:
        self.progress.emit(current, total)

class MainWindow(QMainWindow):
    """Main window for BOEF analysis UI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("1DTransport.com: BOEF Calculation Tool")
        self.resize(1400, 800)
        self.setMinimumSize(1200, 720)

        self.session = self._init_database()
        self._last_analysis_result: AnalysisResult | None = None
        self._last_analysis_inputs: AnalysisInputs | None = None
        self._last_analysis_config: AnalysisConfig | None = None
        self._last_analysis_mode: AnalysisMode | None = None
        self._pending_analysis_load_source: dict[str, object] | None = None
        self._last_analysis_load_source: dict[str, object] | None = None
        self._last_envelope_result: EnvelopeResult | None = None
        self._last_envelope_config: EnvelopeConfig | None = None
        self._pending_envelope_load_source: dict[str, object] | None = None
        self._last_envelope_load_source: dict[str, object] | None = None
        self._last_analysis_stress: StressResults | None = None
        self._last_envelope_stress: StressResults | None = None
        self._last_dynamic_stress: StressResults | None = None
        self._last_rendered_stress: StressResults | None = None
        self._last_stress_title: str = "Stress"
        self._last_stress_unavailable_note: str | None = None
        self._last_static_mode: StaticMode | None = None
        self._last_transition_result: TransitionRunResult | None = None
        self._last_transition_context: TransitionContext | None = None
        self._pending_transition_context: TransitionContext | None = None
        self._pending_transition_load_source: dict[str, object] | None = None
        self._last_transition_load_source: dict[str, object] | None = None
        self._overlay_results: list[tuple[str, AnalysisResult]] = []
        self._primary_label: str | None = None
        self._dynamic_overlay_results: list[tuple[str, DynamicResult]] = []
        self._dynamic_primary_label: str | None = None
        self._dynamic_overlay_mode: DynamicMode | None = None
        self._last_dynamic_result: DynamicResult | None = None
        self._last_dynamic_config: DynamicConfig | None = None
        self._pending_dynamic_load_source: dict[str, object] | None = None
        self._last_dynamic_load_source: dict[str, object] | None = None
        self._last_dynamic_transition_result: DynamicTransitionResult | None = None
        self._last_dynamic_transition_config: DynamicTransitionConfig | None = None
        self._last_dynamic_transition_load_source: dict[str, object] | None = None
        self._last_dynamic_mode: DynamicMode | None = None
        self._last_special_result: FloatingSlabResult | None = None
        self._last_special_config: FloatingSlabConfig | None = None
        self._last_dipped_joint_result: DippedJointResult | None = None
        self._last_dipped_joint_config: DippedJointConfig | None = None
        self._active_track_config_id: int | None = None
        self._active_project_id: int | None = None
        self._active_track_config_name: str | None = None
        self._active_track_gauge_m: float | None = None
        self.worker: AnalysisWorker | None = None
        self.dynamic_worker: DynamicAnalysisWorker | None = None
        self.special_worker: SpecialAnalysisWorker | None = None
        self.envelope_worker: EnvelopeAnalysisWorker | None = None
        self.thread: QThread | None = None
        self.help_dialog: QDialog | None = None
        self.help_browser: QTextBrowser | None = None
        self.dynamic_help_dialog: QDialog | None = None
        self.dynamic_help_browser: QTextBrowser | None = None
        self._chart_thumbnail_cache: dict[str, tuple[int, int, QPixmap]] = {}
        self._chart_registry: list[ChartRegistryEntry] = []
        self._chart_registry_by_id: dict[str, ChartRegistryEntry] = {}
        self._chart_hidden_entries: list[ChartRegistryEntry] = []
        self._chart_view_syncing = False
        self._last_single_tab_index: int | None = None
        self._chart_result_token = 0
        self._chart_probe_token = 0
        self._chart_refresh_timer: QTimer | None = None
        self._pending_chart_refresh_ids: set[str] | None = set()
        self._chart_probe_chart_ids: set[str] = {
            "dynamic_time",
            "dynamic_damping",
            "dynamic_fft",
            "dynamic_psd",
            "dynamic_impedance",
        }
        self.main_splitter: QSplitter | None = None
        self.cancel_button: QPushButton | None = None

        self.main_splitter = QSplitter(Qt.Horizontal)
        sidebar = self._build_sidebar_panel()
        plot_panel = self._build_plot_panel()
        self.main_splitter.addWidget(sidebar)
        self.main_splitter.addWidget(plot_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setHandleWidth(14)
        self.main_splitter.setChildrenCollapsible(True)
        self.main_splitter.setSizes([340, 960])
        self.setCentralWidget(self.main_splitter)

        self.statusBar().showMessage("Ready")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._chart_refresh_timer is not None:
            self._chart_refresh_timer.stop()
        for entry in getattr(self, "_chart_registry", []):
            entry.plot_panel.prepare_for_close()
        self.session.close()
        super().closeEvent(event)

    def _init_database(self) -> Session:
        data_dir = Path.home() / ".boef"
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = data_dir / "boef.sqlite"
        database_url = f"sqlite:///{db_path}"
        run_migrations(database_url)
        engine = create_engine(database_url, future=True)
        session_factory = sessionmaker(engine)
        session = session_factory()
        seed_database(session)
        self._ensure_default_pad(session)
        self._ensure_default_project(session)
        return session

    def _ensure_default_pad(self, session: Session) -> None:
        if session.scalars(select(Pad)).first() is None:
            crud.create_pad(
                session,
                name="Standard Pad",
                stiffness_newtons_per_meter=120_000_000.0,
                thickness_m=0.01,
            )

    def _ensure_default_project(self, session: Session) -> None:
        if session.scalars(select(Project)).first() is None:
            crud.create_project(session, name="Sample Project", description="Default project")

    def _build_project_panel(self) -> QGroupBox:
        group = QGroupBox("Projects")
        group.setCheckable(True)
        group.setChecked(True)
        self._set_compact_group(group)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(6, 6, 6, 6)
        group_layout.setSpacing(6)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.project_tree = QTreeWidget()
        self.project_tree.setHeaderHidden(True)
        self.project_tree.setMinimumWidth(200)
        self.project_tree.itemSelectionChanged.connect(self._handle_project_tree_selection)
        self.project_tree.itemDoubleClicked.connect(self._apply_selected_project_config)
        layout.addWidget(self.project_tree, stretch=1)

        self.project_detail_label = QLabel("Select a project or track config to view details.")
        self.project_detail_label.setWordWrap(True)
        self.project_detail_label.setStyleSheet("color: #4f4f4f;")
        layout.addWidget(self.project_detail_label)

        self.apply_config_button = QPushButton("Apply selected config")
        self.apply_config_button.setEnabled(False)
        self.apply_config_button.clicked.connect(self._apply_selected_project_config)
        layout.addWidget(self.apply_config_button)

        self.sensitivity_button = QPushButton("Sensitivity / Design")
        self.sensitivity_button.setEnabled(False)
        self.sensitivity_button.clicked.connect(self._open_sensitivity_dialog)
        layout.addWidget(self.sensitivity_button)

        alternatives_layout = QHBoxLayout()
        self.save_current_alternative_button = QPushButton("Save current result as design alternative")
        self.save_current_alternative_button.setEnabled(False)
        self.save_current_alternative_button.clicked.connect(self._save_current_result_as_alternative)
        alternatives_layout.addWidget(self.save_current_alternative_button)
        self.compare_alternatives_button = QPushButton("Compare alternatives")
        self.compare_alternatives_button.setEnabled(False)
        self.compare_alternatives_button.clicked.connect(self._open_alternatives_comparison)
        alternatives_layout.addWidget(self.compare_alternatives_button)
        layout.addLayout(alternatives_layout)

        manage_layout = QHBoxLayout()
        manage_projects_button = QPushButton("Manage projects")
        manage_projects_button.clicked.connect(self._open_project_dialog)
        manage_layout.addWidget(manage_projects_button)
        manage_configs_button = QPushButton("Manage track configs")
        manage_configs_button.clicked.connect(self._open_track_config_dialog)
        manage_layout.addWidget(manage_configs_button)
        layout.addLayout(manage_layout)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._refresh_project_tree)
        layout.addWidget(refresh_button)

        group_layout.addWidget(content)
        group.toggled.connect(content.setVisible)

        self._refresh_project_tree()
        return group

    def _refresh_project_tree(self) -> None:
        self.project_tree.clear()
        projects = self.session.scalars(select(Project).order_by(Project.name)).all()
        for project in projects:
            project_item = QTreeWidgetItem([f"{project.name} ({len(project.track_configs)})"])
            project_item.setData(0, Qt.UserRole, {"type": "project", "id": project.id})
            self.project_tree.addTopLevelItem(project_item)
            for config in sorted(project.track_configs, key=lambda item: item.name):
                config_item = QTreeWidgetItem(project_item, [config.name])
                config_item.setData(0, Qt.UserRole, {"type": "config", "id": config.id})
            alternatives = sorted(
                project.design_alternatives,
                key=lambda item: item.created_at,
                reverse=True,
            )
            if alternatives:
                alternatives_item = QTreeWidgetItem(project_item, [f"Design Alternatives ({len(alternatives)})"])
                alternatives_item.setData(0, Qt.UserRole, {"type": "alternatives", "id": project.id})
                for alternative in alternatives:
                    alt_item = QTreeWidgetItem(alternatives_item, [alternative.name])
                    alt_item.setData(0, Qt.UserRole, {"type": "alternative", "id": alternative.id})
        self._handle_project_tree_selection()

    def _handle_project_tree_selection(self) -> None:
        selected = self.project_tree.selectedItems()
        if not selected:
            active_project, active_config = self._active_project_or_config()
            if active_config is not None:
                self.project_detail_label.setText(
                    f"Active track config: {active_config.name}\n"
                    "No project tree item is selected."
                )
            else:
                self.project_detail_label.setText("Select a project or track config to view details.")
            self.apply_config_button.setEnabled(False)
            self.sensitivity_button.setEnabled(active_config is not None)
            self._update_alternative_action_buttons(project=active_project, config=active_config)
            return
        item = selected[0]
        payload = item.data(0, Qt.UserRole) or {}
        item_type = payload.get("type")
        item_id = payload.get("id")
        if item_type == "project":
            project = self.session.get(Project, item_id)
            if project is None:
                self.project_detail_label.setText("Project not found.")
                self.apply_config_button.setEnabled(False)
                self.sensitivity_button.setEnabled(False)
                self.save_current_alternative_button.setEnabled(False)
                self.compare_alternatives_button.setEnabled(False)
                return
            description = project.description or "No description provided."
            vehicle_defaults = VEHICLE_DEFAULTS.get(project.vehicle_type or "")
            vehicle_lines = []
            if project.vehicle_type:
                label = project.vehicle_type
                if vehicle_defaults:
                    label = vehicle_defaults.display_name
                vehicle_lines.append(f"Vehicle type: {label}")
            if project.vehicle_subtype:
                vehicle_lines.append(f"Subtype: {project.vehicle_subtype}")
            if vehicle_defaults:
                vehicle_lines.append(f"Arrangement: {vehicle_defaults.arrangement}")
                vehicle_lines.append(
                    f"Wheel load: {vehicle_defaults.wheel_load_kn:.0f} kN"
                )
                vehicle_lines.append(
                    f"Wheel diameter: {vehicle_defaults.wheel_diameter_mm:.0f} mm"
                )
            if project.design_wheel_radius_mm:
                vehicle_lines.append(
                    f"Wheel radius override: {project.design_wheel_radius_mm:.0f} mm"
                )
            if project.design_speed_kmh:
                vehicle_lines.append(f"Design speed: {project.design_speed_kmh:.0f} km/h")
            vehicle_text = "\n".join(vehicle_lines)
            detail_text = f"{project.name}\n{description}"
            if vehicle_text:
                detail_text = f"{detail_text}\n{vehicle_text}"
            self.project_detail_label.setText(detail_text)
            self.apply_config_button.setEnabled(False)
            self.sensitivity_button.setEnabled(bool(project.track_configs))
            self._update_alternative_action_buttons(project=project, config=None)
            return
        if item_type == "config":
            config = self.session.get(TrackConfig, item_id)
            if config is None:
                self.project_detail_label.setText("Track config not found.")
                self.apply_config_button.setEnabled(False)
                self.sensitivity_button.setEnabled(False)
                self.save_current_alternative_button.setEnabled(False)
                self.compare_alternatives_button.setEnabled(False)
                return
            sleeper_spacing_mm = m_to_mm(config.sleeper_spacing_m)
            gauge_mm = m_to_mm(config.gauge_m)
            details = (
                f"{config.name}\n"
                f"Rail: {config.rail.name if config.rail else 'Unknown'}\n"
                f"Sleeper: {config.sleeper.name if config.sleeper else 'Unknown'}\n"
                f"Support: {config.support_profile.name if config.support_profile else 'Unknown'}\n"
                f"Sleeper spacing: {sleeper_spacing_mm:.1f} mm\n"
                f"Gauge: {gauge_mm:.1f} mm"
            )
            self.project_detail_label.setText(details)
            self.apply_config_button.setEnabled(True)
            self.sensitivity_button.setEnabled(True)
            self._update_alternative_action_buttons(project=config.project, config=config)
            return
        if item_type == "alternatives":
            project = self.session.get(Project, item_id)
            if project is None:
                self.project_detail_label.setText("Project not found.")
                self.apply_config_button.setEnabled(False)
                self.sensitivity_button.setEnabled(False)
                self.save_current_alternative_button.setEnabled(False)
                self.compare_alternatives_button.setEnabled(False)
                return
            self.project_detail_label.setText(
                f"Design Alternatives\nProject: {project.name}\n"
                f"Alternatives: {len(project.design_alternatives)}"
            )
            self.apply_config_button.setEnabled(False)
            self.sensitivity_button.setEnabled(bool(project.track_configs))
            self._update_alternative_action_buttons(project=project, config=None)
            return
        if item_type == "alternative":
            alternative = self.session.get(DesignAlternative, item_id)
            if alternative is None:
                self.project_detail_label.setText("Design alternative not found.")
                self.apply_config_button.setEnabled(False)
                self.sensitivity_button.setEnabled(False)
                self.save_current_alternative_button.setEnabled(False)
                self.compare_alternatives_button.setEnabled(False)
                return
            self.project_detail_label.setText(self._format_design_alternative_detail(alternative))
            self.apply_config_button.setEnabled(False)
            self.sensitivity_button.setEnabled(alternative.track_config is not None)
            self._update_alternative_action_buttons(
                project=alternative.project,
                config=alternative.track_config,
            )
            return
        self.project_detail_label.setText("Select a project or track config to view details.")
        self.apply_config_button.setEnabled(False)
        self.sensitivity_button.setEnabled(False)
        self.save_current_alternative_button.setEnabled(False)
        self.compare_alternatives_button.setEnabled(False)

    def _apply_selected_project_config(self) -> None:
        selected = self.project_tree.selectedItems()
        if not selected:
            return
        payload = selected[0].data(0, Qt.UserRole) or {}
        if payload.get("type") != "config":
            return
        config = self.session.get(TrackConfig, payload.get("id"))
        if config is None:
            QMessageBox.warning(self, "Config missing", "Selected track config no longer exists.")
            return
        self._apply_track_config(config)

    def _selected_project_or_config(self) -> tuple[Project | None, TrackConfig | None]:
        selected = self.project_tree.selectedItems()
        if not selected:
            return self._active_project_or_config()
        payload = selected[0].data(0, Qt.UserRole) or {}
        item_type = payload.get("type")
        item_id = payload.get("id")
        if item_type == "config":
            config = self.session.get(TrackConfig, item_id)
            return (config.project if config is not None else None), config
        if item_type == "alternative":
            alternative = self.session.get(DesignAlternative, item_id)
            if alternative is None:
                return None, None
            return alternative.project, alternative.track_config
        if item_type == "alternatives":
            return self.session.get(Project, item_id), None
        if item_type == "project":
            return self.session.get(Project, item_id), None
        return self._active_project_or_config()

    def _active_project_or_config(self) -> tuple[Project | None, TrackConfig | None]:
        config = None
        if self._active_track_config_id is not None:
            config = self.session.get(TrackConfig, self._active_track_config_id)
        if config is not None:
            return config.project, config
        if self._active_project_id is not None:
            return self.session.get(Project, self._active_project_id), None
        return None, None

    def _selected_project_for_alternatives(self) -> Project | None:
        project, config = self._selected_project_or_config()
        if project is not None:
            return project
        return config.project if config is not None else None

    def _has_current_analysis_snapshot(self) -> bool:
        return any(
            item is not None
            for item in (
                self._last_analysis_result,
                self._last_envelope_result,
                self._last_transition_result,
                self._last_dynamic_result,
                self._last_dynamic_transition_result,
                self._last_dipped_joint_result,
                self._last_special_result,
            )
        )

    def _update_alternative_action_buttons(
        self,
        *,
        project: Project | None,
        config: TrackConfig | None,
    ) -> None:
        self.save_current_alternative_button.setEnabled(
            project is not None and self._has_current_analysis_snapshot()
        )
        self.compare_alternatives_button.setEnabled(
            project is not None and len(project.design_alternatives) >= 2
        )

    def _refresh_alternative_action_buttons_for_selection(self) -> None:
        project, config = self._selected_project_or_config()
        if project is None and config is not None:
            project = config.project
        self.sensitivity_button.setEnabled(config is not None or bool(project and project.track_configs))
        self._update_alternative_action_buttons(project=project, config=config)

    def _format_design_alternative_detail(self, alternative: DesignAlternative) -> str:
        changed = self._json_summary(alternative.changed_parameters_json)
        metrics = self._json_summary(alternative.metrics_json)
        created = alternative.created_at.isoformat(sep=" ", timespec="seconds")
        return (
            f"{alternative.name}\n"
            f"Source: {alternative.source_type} / {alternative.analysis_type}\n"
            f"Status: {alternative.status}\n"
            f"Track config: {alternative.track_config.name if alternative.track_config else 'Unknown'}\n"
            f"Changed parameters: {changed}\n"
            f"Metrics: {metrics}\n"
            f"Notes: {alternative.description or '-'}\n"
            f"Created: {created}"
        )

    @staticmethod
    def _json_summary(text: str) -> str:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return "-"
        if not isinstance(payload, dict) or not payload:
            return "-"
        parts = []
        for key, value in list(payload.items())[:5]:
            parts.append(f"{key}={value}")
        return "; ".join(parts)

    def _open_sensitivity_dialog(self) -> None:
        project, config = self._selected_project_or_config()
        if project is None and config is None:
            QMessageBox.warning(
                self,
                "Sensitivity / Design",
                "Select a project or track config before opening sensitivity.",
            )
            return
        if config is None and project is not None and not project.track_configs:
            QMessageBox.warning(
                self,
                "Sensitivity / Design",
                "This project does not have a track config to analyse.",
            )
            return
        if config is not None:
            try:
                current_loads = self._collect_analysis_loads()
                current_load_source = self._current_load_source_metadata()
            except ValueError:
                current_loads = [PointLoad(position_m=0.0, load_newtons=kn_to_n(self.load_magnitude_input.value()))]
                current_load_source = {"source_type": "single_point_load"}
            self._apply_track_config(config)
        elif project is not None:
            self._active_project_id = project.id
            try:
                current_loads = self._collect_analysis_loads()
                current_load_source = self._current_load_source_metadata()
            except ValueError:
                current_loads = [PointLoad(position_m=0.0, load_newtons=kn_to_n(self.load_magnitude_input.value()))]
                current_load_source = {"source_type": "single_point_load"}
        else:
            current_loads = [PointLoad(position_m=0.0, load_newtons=kn_to_n(self.load_magnitude_input.value()))]
            current_load_source = {"source_type": "single_point_load"}
        dialog = SensitivityDialog(
            session=self.session,
            project=project,
            track_config=config,
            current_load_n=kn_to_n(self.load_magnitude_input.value()),
            current_loads=current_loads,
            current_load_source_metadata=current_load_source,
            build_static_context=self._build_sensitivity_static_context,
            build_transition_context=self._build_sensitivity_transition_context,
            parent=self,
        )
        dialog.alternatives_saved.connect(self._refresh_project_tree)
        dialog.exec()
        self._refresh_project_tree()

    def _save_current_result_as_alternative(self) -> None:
        project, config = self._selected_project_or_config()
        if project is None:
            QMessageBox.warning(
                self,
                "Design Alternative",
                "Select a project or track config before saving an alternative.",
            )
            return
        if config is None:
            if len(project.track_configs) == 1:
                config = project.track_configs[0]
            else:
                QMessageBox.warning(
                    self,
                    "Design Alternative",
                    "Select the track config that belongs to this completed result.",
                )
                return
        snapshot = self._current_result_alternative_snapshot()
        if snapshot is None:
            QMessageBox.warning(
                self,
                "Design Alternative",
                "Run an analysis before saving a design alternative.",
            )
            return
        analysis_type, metrics, status, description = snapshot
        load_source = self._current_result_load_source_metadata()
        load_case = self.load_case_combo.currentData()
        if load_case is not None and not isinstance(load_case, LoadCase):
            load_case = None
        try:
            crud.create_design_alternative(
                self.session,
                project_id=project.id,
                track_config_id=config.id,
                load_case_id=load_case.id if load_case is not None else None,
                name=f"Current {analysis_type} result - {status}",
                description=description,
                source_type="analysis",
                analysis_type=analysis_type,
                changed_parameters={"source": "current_analysis"},
                input_snapshot={
                    "project": project.name,
                    "track_config": config.name,
                    "load_case": load_case.name if load_case is not None else None,
                    "active_track_config_name": self._active_track_config_name,
                    "load_source": load_source,
                },
                metrics=metrics,
                status=status,
                score=None,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Design Alternative", str(exc))
            return
        self._refresh_project_tree()
        QMessageBox.information(self, "Design Alternative", "Saved current result as a design alternative.")

    def _current_result_load_source_metadata(self) -> dict[str, object] | None:
        if (
            self._last_transition_result is not None
            or self._last_envelope_result is not None
            or self._last_analysis_result is not None
        ):
            return self._last_static_load_source_metadata()
        if self._last_dynamic_transition_result is not None or self._last_dynamic_result is not None:
            return self._last_dynamic_load_source_metadata()
        return None

    def _current_result_alternative_snapshot(
        self,
    ) -> tuple[str, dict[str, float | None], str, str] | None:
        if self._last_transition_result is not None:
            metrics = self._transition_alternative_metrics(self._last_transition_result)
            return "transition", metrics, "ok", "Saved from current transition result."
        if self._last_dynamic_transition_result is not None:
            result = self._last_dynamic_transition_result
            metrics = {
                "max_deflection_m": result.metrics.max_deflection_m,
                "max_moment_nm": result.metrics.max_moment_nm,
                "max_reaction_n_per_m": result.metrics.max_reaction_n_per_m,
                "transition_metric_m": result.metrics.risk_index,
            }
            return "dynamic", metrics, "ok", "Saved from current dynamic transition result."
        if self._last_dipped_joint_result is not None:
            result = self._last_dipped_joint_result
            metrics = {
                "p1_n": result.p1_n,
                "p2_n": result.p2_n,
                "p1_dynamic_amplification": result.p1_dynamic_amplification,
                "p2_dynamic_amplification": result.p2_dynamic_amplification,
            }
            return "dynamic", metrics, "ok", "Saved from current dipped-joint result."
        if self._last_dynamic_result is not None:
            result = self._last_dynamic_result
            metrics = {
                "max_deflection_m": result.summary.max_deflection.value,
                "max_moment_nm": result.summary.max_moment.value,
                "max_reaction_n_per_m": result.summary.max_reaction.value,
            }
            return "dynamic", metrics, "ok", "Saved from current dynamic result."
        if self._last_special_result is not None:
            result = self._last_special_result
            metrics = {
                "natural_frequency_hz": result.natural_frequency_hz,
                "damping_ratio": result.damping_ratio,
                "static_deflection_m": result.static_deflection_m,
            }
            return "special", metrics, "ok", "Saved from current special analysis result."
        if self._last_envelope_result is not None:
            result = self._last_envelope_result
            formation_depths = sorted(result.summary.max_formation_stress_by_depth_pa)
            formation_pressure = (
                abs(result.summary.max_formation_stress_by_depth_pa[formation_depths[0]])
                if formation_depths
                else None
            )
            subgrade_pressure = (
                abs(result.summary.max_formation_stress_by_depth_pa[formation_depths[-1]])
                if formation_depths
                else None
            )
            deep_subgrade_pressure = subgrade_pressure
            metrics = {
                "max_deflection_m": abs(result.summary.max_deflection.value),
                "max_moment_nm": abs(result.summary.max_moment.value),
                "max_sleeper_load_n": abs(result.summary.max_sleeper_load.value),
                "rail_stress_pa": abs(result.summary.max_rail_base_stress_pa),
                "ballast_pressure_pa": abs(result.summary.max_ballast_pressure.value),
                "formation_pressure_pa": formation_pressure,
                "subgrade_pressure_pa": subgrade_pressure,
                "deep_subgrade_pressure_pa": deep_subgrade_pressure,
            }
            return "static", metrics, "ok", "Saved from current envelope result."
        if self._last_analysis_result is not None:
            result = self._last_analysis_result
            design_summary = result.summary.design_summary
            a3902 = design_summary.a3902_checks if design_summary is not None else None
            deep_subgrade_pressure = None
            _project, config = self._selected_project_or_config()
            ballast_depth_m = self._a3902_ballast_depth_m()
            if a3902 is not None and config is not None and config.sleeper is not None and ballast_depth_m:
                try:
                    deep_subgrade_pressure = subgrade_pressure_a3902(
                        ballast_contact_pressure_pa=a3902.ballast_contact_pressure_pa,
                        ballast_depth_m=ballast_depth_m,
                        fill_depth_m=1.0,
                        sleeper_width_m=config.sleeper.width_m,
                        effective_bearing_length_m=a3902.effective_bearing_length_m,
                    )
                except (AttributeError, ValueError):
                    deep_subgrade_pressure = None
            metrics = {
                "max_deflection_m": abs(result.summary.max_deflection.value),
                "max_moment_nm": abs(result.summary.max_moment.value),
                "max_sleeper_load_n": abs(result.summary.max_sleeper_load.value),
                "rail_stress_pa": abs(result.summary.max_rail_base_stress_pa),
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
                "deep_subgrade_pressure_pa": (
                    abs(deep_subgrade_pressure)
                    if deep_subgrade_pressure is not None
                    else None
                ),
            }
            return "static", metrics, "ok", "Saved from current static result."
        return None

    @staticmethod
    def _transition_alternative_metrics(result: TransitionRunResult) -> dict[str, float | None]:
        return {
            "max_deflection_m": (
                max(abs(value) for value in result.series.deflection_m)
                if result.series.deflection_m
                else None
            ),
            "max_moment_nm": abs(result.metrics.moment_max_nm),
            "max_sleeper_load_n": abs(result.metrics.sleeper_load_max_n),
            "transition_metric_m": abs(result.metrics.delta_w_s_m),
        }

    def _open_alternatives_comparison(self) -> None:
        project = self._selected_project_for_alternatives()
        if project is None:
            QMessageBox.warning(
                self,
                "Design Alternatives",
                "Select a project or design alternative first.",
            )
            return
        alternatives = list(project.design_alternatives)
        if len(alternatives) < 2:
            QMessageBox.warning(
                self,
                "Design Alternatives",
                "At least two alternatives are required for comparison.",
            )
            return
        dialog = AlternativeComparisonDialog(
            project=project,
            alternatives=alternatives,
            parent=self,
        )
        dialog.exec()

    def _build_sensitivity_static_context(
        self,
        config: TrackConfig,
        load_n: float,
    ) -> AnalysisConfig:
        self.session.refresh(config)
        if not all([config.rail, config.sleeper, config.support_profile]):
            raise ValueError("Selected track config is missing rail, sleeper, or support data.")
        rail = config.rail
        sleeper = config.sleeper
        support = config.support_profile
        area_m2 = cm2_to_m2(rail.area_cm2) if rail.area_cm2 is not None else None
        design_inputs = DesignInputs(
            speed_kmh=self.design_speed_input.value(),
            track_factor=self.track_quality_combo.currentData(),
            probability_factor=self.probability_combo.currentData(),
            wheel_radius_mm=self.wheel_radius_input.value(),
            tensile_strength_mpa=self.tensile_strength_combo.currentData(),
            on_curve=self.curve_checkbox.isChecked(),
            ballast_depth_m=self._a3902_ballast_depth_m(),
            rail_centres_m=self._resolve_a3902_rail_centres_m(rail),
        )
        loads = [PointLoad(position_m=0.0, load_newtons=load_n)]
        return AnalysisConfig(
            loads=loads,
            foundation_modulus_n_per_m2=support.foundation_modulus_n_per_m2,
            elastic_modulus_pa=rail.elastic_modulus_pa,
            moment_inertia_m4=rail.moment_inertia_m4,
            section_modulus_m3=rail.section_modulus_m3,
            sleeper_spacing_m=config.sleeper_spacing_m,
            sleeper_length_m=sleeper.length_m,
            sleeper_width_m=sleeper.width_m,
            sample_count=401,
            x_domain_m=None,
            section_modulus_head_m3=rail.section_modulus_head_m3,
            section_modulus_foot_m3=rail.section_modulus_foot_m3,
            area_m2=area_m2,
            discrete_support_stiffness_n_per_m=(
                config.pad.stiffness_newtons_per_meter if config.pad is not None else None
            ),
            design_inputs=design_inputs,
            railpad_stiffness_n_per_m=(
                config.pad.stiffness_newtons_per_meter if config.pad is not None else None
            ),
            pad_stiffness_n_per_m=(
                config.pad.stiffness_newtons_per_meter if config.pad is not None else None
            ),
        )

    def _build_sensitivity_transition_context(
        self,
        config: TrackConfig,
        load_n: float,
    ) -> SensitivityTransitionContext:
        analysis_config = self._build_sensitivity_static_context(config, load_n)
        profile_type = self._coerce_transition_profile_type(self.transition_profile_combo.currentData())
        k1_n_per_m2 = mn_per_m2_to_n_per_m2(self.transition_k1_input.value())
        if k1_n_per_m2 <= 0.0:
            k1_n_per_m2 = analysis_config.foundation_modulus_n_per_m2
        k2_n_per_m2: float | None = None
        transition_length_m: float | None = None
        segment_length_m: float | None = None
        if profile_type != TransitionProfileType.UNIFORM:
            k2_n_per_m2 = mn_per_m2_to_n_per_m2(self.transition_k2_input.value())
            if k2_n_per_m2 <= 0.0:
                raise ValueError("k2 must be positive for transition sensitivity.")
        if profile_type in (TransitionProfileType.RAMP, TransitionProfileType.EXPONENTIAL):
            transition_length_m = self.transition_length_input.value()
            if transition_length_m <= 0.0:
                raise ValueError("Transition length Lt must be positive for transition sensitivity.")
        if profile_type == TransitionProfileType.SEGMENT:
            segment_length_m = self.transition_segment_length_input.value()
            if segment_length_m <= 0.0:
                raise ValueError("Segment length must be positive for transition sensitivity.")
        x_min = self.transition_domain_start_input.value()
        x_max = self.transition_domain_end_input.value()
        if x_max <= x_min:
            raise ValueError("Transition domain end must be greater than start.")
        self._validate_transition_domain_covers_profile(
            profile_type=profile_type,
            domain_m=(x_min, x_max),
            transition_length_m=transition_length_m,
            segment_length_m=segment_length_m,
        )
        analysis_config = replace(
            analysis_config,
            foundation_modulus_n_per_m2=k1_n_per_m2,
            x_domain_m=(x_min, x_max),
        )
        analysis_mode = AnalysisMode.NUMERICAL if profile_type != TransitionProfileType.UNIFORM else AnalysisMode.CLOSED_FORM
        return SensitivityTransitionContext(
            run_mode=TransitionRunMode.SINGLE,
            profile_type=profile_type,
            template_name=self.transition_template_combo.currentText(),
            preset_name=self.transition_preset_combo.currentText(),
            k1_n_per_m2=k1_n_per_m2,
            k2_n_per_m2=k2_n_per_m2,
            transition_length_m=transition_length_m,
            segment_length_m=segment_length_m,
            domain_m=(x_min, x_max),
            analysis_config=analysis_config,
            analysis_mode=analysis_mode,
        )

    def _apply_track_config(self, config: TrackConfig) -> None:
        self._select_combo_by_id(self.rail_combo, config.rail_id)
        self._select_combo_by_id(self.sleeper_combo, config.sleeper_id)
        self._select_combo_by_id(self.pad_combo, config.pad_id)
        self._select_combo_by_id(self.support_combo, config.support_profile_id)
        self.sleeper_spacing_input.set_value(m_to_mm(config.sleeper_spacing_m))
        self._active_track_config_id = config.id
        self._active_project_id = config.project_id
        self._active_track_config_name = config.name
        self._active_track_gauge_m = config.gauge_m
        if config.project is not None:
            self._apply_project_defaults(config.project, apply_loads=False)
        load_case = self._default_load_case_for_config(config)
        if load_case is not None:
            if not self._combo_contains_id(self.load_case_combo, load_case.id):
                self._refresh_load_cases()
            self._select_combo_by_id(self.load_case_combo, load_case.id)
            if self.several_loads_checkbox.isChecked():
                self.several_loads_checkbox.setChecked(False)
            if self.train_loads_checkbox.isChecked():
                self.train_loads_checkbox.setChecked(False)
            if hasattr(self, "as5100_loads_checkbox") and self.as5100_loads_checkbox.isChecked():
                self.as5100_loads_checkbox.setChecked(False)
        self.statusBar().showMessage(f"Applied track config: {config.name}", 5000)

    def _apply_project_defaults(
        self,
        project: Project,
        *,
        apply_loads: bool,
    ) -> VehicleDefaults | None:
        defaults = VEHICLE_DEFAULTS.get(project.vehicle_type or "")
        if defaults and apply_loads:
            self.load_magnitude_input.set_value(defaults.wheel_load_kn)
            self.train_axle_load_input.set_value(defaults.axle_load_kn)
            self.train_bogie_spacing_input.set_value(defaults.bogie_spacing_m * 1000.0)
            self.train_axles_per_bogie_input.setValue(defaults.axles_per_bogie)
            self.train_bogie_count_input.setValue(2)

        wheel_radius_mm = project.design_wheel_radius_mm
        if wheel_radius_mm is None and defaults is not None:
            wheel_radius_mm = defaults.wheel_diameter_mm / 2.0
        if wheel_radius_mm is not None and wheel_radius_mm > 0:
            self.wheel_radius_input.set_value(wheel_radius_mm)

        if project.design_speed_kmh is not None and project.design_speed_kmh > 0:
            self.design_speed_input.set_value(project.design_speed_kmh)

        if defaults and apply_loads:
            self.statusBar().showMessage(
                f"Applied rolling stock defaults: {defaults.display_name}", 5000
            )
        return defaults

    def _default_load_case_for_config(self, config: TrackConfig) -> LoadCase | None:
        result = self.session.scalar(
            select(Result)
            .where(Result.track_config_id == config.id)
            .order_by(Result.id.desc())
        )
        if result is None:
            return None
        return self.session.get(LoadCase, result.load_case_id)

    def _combo_contains_id(self, combo: QComboBox, item_id: int) -> bool:
        for index in range(combo.count()):
            data = combo.itemData(index)
            if data is not None and getattr(data, "id", None) == item_id:
                return True
        return False

    def _select_combo_by_id(self, combo: QComboBox, item_id: int) -> None:
        for index in range(combo.count()):
            data = combo.itemData(index)
            if data is not None and getattr(data, "id", None) == item_id:
                combo.setCurrentIndex(index)
                return

    def _build_input_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container.setMinimumWidth(0)
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        layout.addWidget(self._build_material_group())
        layout.addWidget(self._build_analysis_group())
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def _build_material_group(self) -> QGroupBox:
        group = QGroupBox("Materials")
        self._set_compact_group(group)
        layout = QFormLayout(group)
        self._configure_form_layout(layout)

        self.rail_combo = QComboBox()
        self.sleeper_combo = QComboBox()
        self.pad_combo = QComboBox()
        self.support_combo = QComboBox()

        self._refresh_material_combos()
        self.pad_combo.currentIndexChanged.connect(self._sync_pad_inputs)
        self.rail_combo.currentIndexChanged.connect(lambda _value: self._update_envelope_estimate())
        self.support_combo.currentIndexChanged.connect(lambda _value: self._update_envelope_estimate())
        self.rail_combo.currentIndexChanged.connect(lambda _value: self._update_validation_hints())
        self.support_combo.currentIndexChanged.connect(lambda _value: self._update_validation_hints())

        layout.addRow(self._combo_with_button("Rail", self.rail_combo, self._open_rail_dialog))
        layout.addRow(self._combo_with_button("Sleeper", self.sleeper_combo, self._open_sleeper_dialog))
        layout.addRow(self._combo_with_button("Pad", self.pad_combo, self._open_pad_dialog))
        layout.addRow(self._combo_with_button("Support profile", self.support_combo, self._open_support_dialog))

        return group

    def _combo_with_button(self, label: str, combo: QComboBox, handler: Callable[[], None]) -> QWidget:
        container = QWidget()
        container.setMinimumWidth(0)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        manage_button = QPushButton("Manage")
        manage_button.clicked.connect(handler)
        layout.addWidget(combo, stretch=1)
        layout.addWidget(manage_button)
        return container

    def _configure_form_layout(self, layout: QFormLayout) -> None:
        layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        layout.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

    def _set_compact_combo(self, combo: QComboBox, *, max_width: int = 420) -> None:
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        combo.setMaximumWidth(max_width)

    def _set_compact_group(self, group: QGroupBox, *, max_width: int = 520) -> None:
        group.setMaximumWidth(max_width)
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    def _set_compact_line_edit(self, line_edit: QLineEdit, *, max_width: int = 420) -> None:
        line_edit.setMaximumWidth(max_width)

    def _build_sidebar_panel(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setMinimumWidth(360)
        sidebar.setMaximumWidth(560)
        sidebar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self._build_project_panel())
        layout.addWidget(self._build_input_panel(), stretch=1)
        return sidebar

    def _build_analysis_group(self) -> QGroupBox:
        group = QGroupBox("Analysis inputs")
        self._set_compact_group(group)
        layout = QFormLayout(group)
        self._configure_form_layout(layout)
        self.analysis_layout = layout

        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItem("Static", AnalysisType.STATIC)
        self.analysis_type_combo.addItem("Dynamic", AnalysisType.DYNAMIC)
        self.analysis_type_combo.addItem("Special", AnalysisType.SPECIAL)
        self.analysis_type_combo.currentIndexChanged.connect(self._toggle_analysis_type)
        self._set_compact_combo(self.analysis_type_combo, max_width=220)
        layout.addRow("Analysis type", self.analysis_type_combo)

        self.static_mode_combo = QComboBox()
        self.static_mode_combo.addItem("Static (single position)", StaticMode.SINGLE)
        self.static_mode_combo.addItem("Quasi-static Envelope (Closed-form)", StaticMode.ENVELOPE_CLOSED_FORM)
        self.static_mode_combo.addItem("Quasi-static Envelope (Advanced static)", StaticMode.ENVELOPE_NUMERICAL)
        self.static_mode_combo.currentIndexChanged.connect(self._toggle_static_mode)
        self._set_compact_combo(self.static_mode_combo, max_width=300)
        layout.addRow("Static mode", self.static_mode_combo)

        self.dynamic_mode_combo = QComboBox()
        self.dynamic_mode_combo.addItem(
            "Steady-state moving load (travelling wave)",
            DynamicMode.STEADY_STATE,
        )
        self.dynamic_mode_combo.addItem("Time-history (explicit)", DynamicMode.TIME_HISTORY)
        self.dynamic_mode_combo.addItem("Transition (dynamic)", DynamicMode.TRANSITION)
        self.dynamic_mode_combo.addItem("Dipped joint (wheel/rail forces)", DynamicMode.DIPPED_JOINT)
        self.dynamic_mode_combo.currentIndexChanged.connect(self._toggle_dynamic_mode)
        self._set_compact_combo(self.dynamic_mode_combo, max_width=280)
        layout.addRow("Dynamic mode", self.dynamic_mode_combo)

        self.dynamic_annotation_mode_combo = QComboBox()
        self.dynamic_annotation_mode_combo.addItem("Full traceability", DynamicAnnotationMode.FULL)
        self.dynamic_annotation_mode_combo.addItem("Compact", DynamicAnnotationMode.COMPACT)
        self.dynamic_annotation_mode_combo.addItem("Off", DynamicAnnotationMode.OFF)
        self.dynamic_annotation_mode_combo.currentIndexChanged.connect(self._handle_dynamic_annotation_mode_changed)
        self._set_compact_combo(self.dynamic_annotation_mode_combo, max_width=220)
        layout.addRow("Dynamic annotations", self.dynamic_annotation_mode_combo)

        self.chart_label_controls = QWidget()
        chart_label_layout = QHBoxLayout(self.chart_label_controls)
        chart_label_layout.setContentsMargins(0, 0, 0, 0)
        chart_label_layout.setSpacing(8)
        self.chart_input_labels_checkbox = QCheckBox("Inputs")
        self.chart_input_labels_checkbox.setToolTip(
            "Show input/provenance badges and load-position labels on charts."
        )
        self.chart_output_labels_checkbox = QCheckBox("Outputs")
        self.chart_output_labels_checkbox.setToolTip(
            "Show output/result summary badges on charts."
        )
        self.chart_extrema_labels_checkbox = QCheckBox("Max/min")
        self.chart_extrema_labels_checkbox.setToolTip(
            "Show max/min point labels on the plotted series."
        )
        for checkbox in (
            self.chart_input_labels_checkbox,
            self.chart_output_labels_checkbox,
            self.chart_extrema_labels_checkbox,
        ):
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._handle_chart_label_visibility_changed)
            chart_label_layout.addWidget(checkbox)
        chart_label_layout.addStretch(1)
        layout.addRow("Chart labels", self.chart_label_controls)

        self.special_mode_combo = QComboBox()
        self.special_mode_combo.addItem("Floating slab isolation", SpecialMode.FLOATING_SLAB)
        self.special_mode_combo.currentIndexChanged.connect(self._toggle_special_mode)
        self._set_compact_combo(self.special_mode_combo, max_width=280)
        layout.addRow("Special mode", self.special_mode_combo)

        self.load_case_combo = QComboBox()
        self.load_case_combo.currentIndexChanged.connect(self._sync_load_case)
        self._set_compact_combo(self.load_case_combo, max_width=280)

        self.load_magnitude_input = UnitInput("kN", decimals=2, minimum=0.0, maximum=1.0e6)
        self.load_magnitude_input.set_value(100.0)
        self.load_magnitude_input.spinbox.valueChanged.connect(lambda _value: self._sync_right_load_inputs())
        self.load_position_input = UnitInput("mm", decimals=1, minimum=-10_000.0, maximum=10_000.0)
        self.load_position_input.set_value(0.0)
        self.load_position_input.spinbox.valueChanged.connect(lambda _value: self._sync_right_load_inputs())
        self.sleeper_spacing_input = UnitInput("mm", decimals=1, minimum=100.0, maximum=2000.0)
        self.sleeper_spacing_input.set_value(600.0)
        self.ballast_thickness_input = UnitInput("mm", decimals=1, minimum=0.0, maximum=5000.0)
        self.ballast_thickness_input.set_value(300.0)
        self.pad_combo.currentIndexChanged.connect(self._sync_multilayer_defaults)
        self.support_combo.currentIndexChanged.connect(self._sync_multilayer_defaults)
        self.sleeper_spacing_input.spinbox.valueChanged.connect(
            lambda _value: self._sync_multilayer_defaults()
        )
        self.sleeper_spacing_input.spinbox.valueChanged.connect(
            lambda _value: self._update_validation_hints()
        )
        self.ballast_thickness_input.spinbox.valueChanged.connect(
            lambda _value: self._update_validation_hints()
        )

        layout.addRow(
            "Load case",
            self._combo_with_button("Load case", self.load_case_combo, self._open_load_case_dialog),
        )
        layout.addRow("Point load", self.load_magnitude_input)
        self.load_position_label = VisibilityStateLabel("Load position")
        layout.addRow(self.load_position_label, self.load_position_input)

        self.several_loads_checkbox = QCheckBox("Several wheel loads (superposition)")
        self.several_loads_checkbox.toggled.connect(self._toggle_several_loads)
        layout.addRow(self.several_loads_checkbox)

        self.wheel_loads_group = QGroupBox("Wheel loads")
        self._set_compact_group(self.wheel_loads_group, max_width=520)
        wheel_loads_layout = QVBoxLayout(self.wheel_loads_group)
        self.wheel_loads_widget = WheelLoadsWidget()
        wheel_loads_layout.addWidget(self.wheel_loads_widget)
        self.wheel_loads_group.setVisible(False)
        layout.addRow(self.wheel_loads_group)

        self.train_loads_checkbox = QCheckBox("Train/axle load builder")
        self.train_loads_checkbox.toggled.connect(self._toggle_train_loads)
        layout.addRow(self.train_loads_checkbox)

        self.train_loads_group = QGroupBox("Train loads")
        self._set_compact_group(self.train_loads_group, max_width=520)
        train_layout = QFormLayout(self.train_loads_group)
        train_layout.setContentsMargins(10, 5, 10, 5)
        self._configure_form_layout(train_layout)

        self.train_axle_load_input = UnitInput("kN", decimals=2, minimum=0.01, maximum=1.0e6)
        self.train_axle_load_input.set_value(100.0)
        self.train_bogie_count_input = QSpinBox()
        self.train_bogie_count_input.setRange(1, 200)
        self.train_bogie_count_input.setValue(2)
        self.train_bogie_spacing_input = UnitInput("mm", decimals=1, minimum=0.0, maximum=1.0e6)
        self.train_bogie_spacing_input.set_value(2500.0)
        self.train_axles_per_bogie_input = QSpinBox()
        self.train_axles_per_bogie_input.setRange(1, 6)
        self.train_axles_per_bogie_input.setValue(2)
        self.train_axle_spacing_input = UnitInput("mm", decimals=1, minimum=0.0, maximum=1.0e6)
        self.train_axle_spacing_input.set_value(1600.0)
        self.train_reference_input = UnitInput("mm", decimals=1, minimum=-1.0e6, maximum=1.0e6)
        self.train_reference_input.set_value(0.0)
        self.train_reference_input.setToolTip("Reference position for the first bogie center (x₀).")

        train_layout.addRow("Axle load", self.train_axle_load_input)
        train_layout.addRow("Bogie count", self.train_bogie_count_input)
        train_layout.addRow("Bogie spacing", self.train_bogie_spacing_input)
        train_layout.addRow("Axles per bogie", self.train_axles_per_bogie_input)
        train_layout.addRow("Axle spacing", self.train_axle_spacing_input)
        train_layout.addRow("Bogie center x₀", self.train_reference_input)

        self.train_loads_group.setVisible(False)
        layout.addRow(self.train_loads_group)

        self.as5100_loads_checkbox = QCheckBox("AS5100 rail loading")
        self.as5100_loads_checkbox.toggled.connect(self._toggle_as5100_loads)
        layout.addRow(self.as5100_loads_checkbox)

        self.as5100_loads_group = QGroupBox("AS5100 vertical rail loading")
        self._set_compact_group(self.as5100_loads_group, max_width=548)
        as5100_layout = QFormLayout(self.as5100_loads_group)
        as5100_layout.setContentsMargins(12, 8, 12, 10)
        self._configure_form_layout(as5100_layout)
        as5100_layout.setHorizontalSpacing(12)
        as5100_layout.setVerticalSpacing(8)

        self.as5100_model_combo = QComboBox()
        self.as5100_model_combo.addItem("300LA (primary)", AS5100_MODEL_300LA)
        self.as5100_model_combo.addItem("150LA (light rail)", AS5100_MODEL_150LA)
        self.as5100_model_combo.currentIndexChanged.connect(self._sync_as5100_model_defaults)
        self._set_compact_combo(self.as5100_model_combo, max_width=260)
        self.as5100_arrangement_mode_combo = QComboBox()
        self.as5100_arrangement_mode_combo.addItem(
            "Fixed selected arrangement",
            AS5100ArrangementMode.FIXED_SELECTED,
        )
        self.as5100_arrangement_mode_combo.addItem(
            "Governing envelope sweep",
            AS5100ArrangementMode.GOVERNING_SWEEP,
        )
        self.as5100_arrangement_mode_combo.currentIndexChanged.connect(self._refresh_as5100_summary_label)
        self._set_compact_combo(self.as5100_arrangement_mode_combo, max_width=260)
        self.as5100_group_count_input = QSpinBox()
        self.as5100_group_count_input.setRange(1, 200)
        self.as5100_group_count_input.setValue(2)
        self.as5100_group_count_input.setMinimumWidth(72)
        self.as5100_group_count_input.setMaximumWidth(92)
        self.as5100_group_count_input.valueChanged.connect(self._refresh_as5100_summary_label)
        self.as5100_group_spacing_input = UnitInput("m", decimals=2, minimum=12.0, maximum=20.0)
        self.as5100_group_spacing_input.set_value(12.0)
        self.as5100_group_spacing_input.spinbox.setMinimumWidth(155)
        self.as5100_group_spacing_input.spinbox.setMaximumWidth(185)
        self.as5100_group_spacing_input.spinbox.valueChanged.connect(self._refresh_as5100_summary_label)
        self.as5100_reference_input = UnitInput("m", decimals=3, minimum=-1.0e6, maximum=1.0e6)
        self.as5100_reference_input.set_value(0.0)
        self.as5100_reference_input.spinbox.setMinimumWidth(155)
        self.as5100_reference_input.spinbox.setMaximumWidth(185)
        self.as5100_reference_input.spinbox.valueChanged.connect(self._refresh_as5100_summary_label)
        self.as5100_summary_label = QLabel()
        self.as5100_summary_label.setWordWrap(True)
        self.as5100_summary_label.setMinimumWidth(285)
        self.as5100_summary_label.setMaximumWidth(350)
        self.as5100_summary_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)

        as5100_layout.addRow("Model", self.as5100_model_combo)
        as5100_layout.addRow("Arrangement mode", self.as5100_arrangement_mode_combo)
        as5100_layout.addRow("Axle group count", self.as5100_group_count_input)
        as5100_layout.addRow("Group spacing", self.as5100_group_spacing_input)
        as5100_layout.addRow("Leading axle x₀", self.as5100_reference_input)
        as5100_layout.addRow("Traceability", self.as5100_summary_label)
        self.as5100_loads_group.setVisible(False)
        layout.addRow(self.as5100_loads_group)
        self._sync_as5100_model_defaults()

        self.overlay_checkbox = QCheckBox("Overlay results on existing charts")
        self.overlay_checkbox.toggled.connect(self._update_overlay_state)
        layout.addRow(self.overlay_checkbox)

        self.clear_overlay_button = QPushButton("Clear overlays")
        self.clear_overlay_button.clicked.connect(self._clear_overlays)
        layout.addRow(self.clear_overlay_button)

        self.design_group = QGroupBox("Design parameters")
        self._set_compact_group(self.design_group, max_width=520)
        design_layout = QFormLayout(self.design_group)
        design_layout.setContentsMargins(10, 5, 10, 5)
        self._configure_form_layout(design_layout)

        self.track_quality_combo = QComboBox()
        self.track_quality_combo.addItem("Very good (φ=0.1)", 0.1)
        self.track_quality_combo.addItem("Good (φ=0.2)", 0.2)
        self.track_quality_combo.addItem("Bad (φ=0.3)", 0.3)
        self.track_quality_combo.setCurrentIndex(1)
        self._set_compact_combo(self.track_quality_combo, max_width=220)

        self.probability_combo = QComboBox()
        self.probability_combo.addItem("68.3% (t=1)", 1.0)
        self.probability_combo.addItem("95.4% (t=2)", 2.0)
        self.probability_combo.addItem("99.7% (t=3)", 3.0)
        self._set_compact_combo(self.probability_combo, max_width=220)

        self.design_speed_input = UnitInput("km/h", decimals=1, minimum=0.0, maximum=400.0)
        self.design_speed_input.set_value(80.0)

        self.wheel_radius_input = UnitInput("mm", decimals=1, minimum=100.0, maximum=10_000.0)
        self.wheel_radius_input.set_value(460.0)

        self.curve_checkbox = QCheckBox("Apply curve load factor (1.2×)")
        self.curve_checkbox.setChecked(False)

        self.tensile_strength_combo = QComboBox()
        for tensile_strength in self._load_tensile_strengths():
            self.tensile_strength_combo.addItem(f"{tensile_strength:.0f} MPa", tensile_strength)
        self._set_compact_combo(self.tensile_strength_combo, max_width=240)

        design_layout.addRow("Track quality", self.track_quality_combo)
        design_layout.addRow("Probability level", self.probability_combo)
        design_layout.addRow("Speed", self.design_speed_input)
        design_layout.addRow("Wheel radius", self.wheel_radius_input)
        design_layout.addRow("Curve analysis", self.curve_checkbox)
        design_layout.addRow("Steel tensile strength", self.tensile_strength_combo)

        layout.addRow(self.design_group)

        layout.addRow("Sleeper spacing", self.sleeper_spacing_input)
        layout.addRow("Ballast thickness", self.ballast_thickness_input)

        self.envelope_group = QGroupBox("Envelope settings")
        self._set_compact_group(self.envelope_group, max_width=520)
        envelope_layout = QFormLayout(self.envelope_group)
        envelope_layout.setContentsMargins(10, 5, 10, 5)
        self._configure_form_layout(envelope_layout)

        self.envelope_reference_input = UnitInput("m", decimals=3, minimum=-1.0e6, maximum=1.0e6)
        self.envelope_reference_input.set_value(0.0)
        envelope_layout.addRow("Reference position x_ref", self.envelope_reference_input)

        self.envelope_range_start_input = UnitInput("m", decimals=3, minimum=-1.0e6, maximum=1.0e6)
        self.envelope_range_end_input = UnitInput("m", decimals=3, minimum=-1.0e6, maximum=1.0e6)
        self.envelope_range_start_input.set_value(-10.0)
        self.envelope_range_end_input.set_value(10.0)
        self.envelope_range_auto_checkbox = QCheckBox("Auto")
        self.envelope_range_auto_checkbox.setChecked(True)
        self.envelope_range_auto_checkbox.toggled.connect(self._toggle_envelope_range_auto)
        range_container = QWidget()
        range_layout = QHBoxLayout(range_container)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.addWidget(self.envelope_range_start_input, stretch=1)
        range_layout.addWidget(QLabel("to"))
        range_layout.addWidget(self.envelope_range_end_input, stretch=1)
        range_layout.addWidget(self.envelope_range_auto_checkbox)
        envelope_layout.addRow("Movement range x_ref", range_container)

        self.envelope_step_input = UnitInput("m", decimals=3, minimum=0.001, maximum=100.0)
        self.envelope_step_input.set_value(0.2)
        self.envelope_step_input.spinbox.valueChanged.connect(
            lambda _value: self._update_envelope_estimate()
        )
        envelope_layout.addRow("Movement increment Δx_ref", self.envelope_step_input)

        self.envelope_domain_start_input = UnitInput("m", decimals=3, minimum=-1.0e6, maximum=1.0e6)
        self.envelope_domain_end_input = UnitInput("m", decimals=3, minimum=-1.0e6, maximum=1.0e6)
        self.envelope_domain_start_input.set_value(-15.0)
        self.envelope_domain_end_input.set_value(15.0)
        self.envelope_domain_auto_checkbox = QCheckBox("Auto")
        self.envelope_domain_auto_checkbox.setChecked(True)
        self.envelope_domain_auto_checkbox.toggled.connect(self._toggle_envelope_domain_auto)
        self.envelope_range_start_input.spinbox.valueChanged.connect(
            lambda _value: self._update_envelope_estimate()
        )
        self.envelope_range_end_input.spinbox.valueChanged.connect(
            lambda _value: self._update_envelope_estimate()
        )
        self.envelope_domain_start_input.spinbox.valueChanged.connect(
            lambda _value: self._update_envelope_estimate()
        )
        self.envelope_domain_end_input.spinbox.valueChanged.connect(
            lambda _value: self._update_envelope_estimate()
        )
        self.envelope_domain_start_input.spinbox.valueChanged.connect(
            lambda _value: self._update_validation_hints()
        )
        self.envelope_domain_end_input.spinbox.valueChanged.connect(
            lambda _value: self._update_validation_hints()
        )
        domain_container = QWidget()
        domain_layout = QHBoxLayout(domain_container)
        domain_layout.setContentsMargins(0, 0, 0, 0)
        domain_layout.addWidget(self.envelope_domain_start_input, stretch=1)
        domain_layout.addWidget(QLabel("to"))
        domain_layout.addWidget(self.envelope_domain_end_input, stretch=1)
        domain_layout.addWidget(self.envelope_domain_auto_checkbox)
        envelope_layout.addRow("Analysis domain x", domain_container)

        self.envelope_depths_input = QLineEdit()
        self.envelope_depths_input.setPlaceholderText("Comma-separated, e.g., 0.3, 0.6, 1.0")
        self.envelope_depths_input.setText("0.3, 0.6, 1.0")
        self._set_compact_line_edit(self.envelope_depths_input, max_width=300)
        envelope_layout.addRow("Formation depths z", self.envelope_depths_input)

        self.envelope_use_sleeper_geometry_checkbox = QCheckBox("Use sleeper geometry")
        self.envelope_use_sleeper_geometry_checkbox.setChecked(True)
        self.envelope_use_sleeper_geometry_checkbox.toggled.connect(
            self._sync_envelope_bearing_defaults
        )
        envelope_layout.addRow(self.envelope_use_sleeper_geometry_checkbox)

        self.envelope_bearing_width_input = UnitInput("m", decimals=3, minimum=0.01, maximum=10.0)
        self.envelope_bearing_length_input = UnitInput("m", decimals=3, minimum=0.01, maximum=20.0)
        self.envelope_bearing_width_input.spinbox.valueChanged.connect(
            lambda _value: self._update_envelope_bearing_area()
        )
        self.envelope_bearing_length_input.spinbox.valueChanged.connect(
            lambda _value: self._update_envelope_bearing_area()
        )
        envelope_layout.addRow("Effective width B₀", self.envelope_bearing_width_input)
        envelope_layout.addRow("Effective length L₀", self.envelope_bearing_length_input)

        self.envelope_bearing_area_label = QLabel("—")
        self.envelope_bearing_area_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        envelope_layout.addRow("Effective area A₀", self.envelope_bearing_area_label)

        self.envelope_rail_count_combo = QComboBox()
        self.envelope_rail_count_combo.addItem("1", 1)
        self.envelope_rail_count_combo.addItem("2", 2)
        self.envelope_rail_count_combo.setCurrentIndex(1)
        self._set_compact_combo(self.envelope_rail_count_combo, max_width=120)
        envelope_layout.addRow("Rail count (total load)", self.envelope_rail_count_combo)

        self.envelope_note_label = QLabel(
            "Note: Load positions are treated as offsets from x_ref in envelope mode."
        )
        self.envelope_note_label.setWordWrap(False)
        self.envelope_note_label.setStyleSheet("color: #555555;")
        envelope_layout.addRow(self.envelope_note_label)

        self.envelope_estimate_label = QLabel("—")
        self.envelope_estimate_label.setWordWrap(False)
        self.envelope_estimate_label.setStyleSheet("color: #555555;")
        envelope_layout.addRow("Run time estimate", self.envelope_estimate_label)

        self.envelope_group.setVisible(False)
        layout.addRow(self.envelope_group)
        self.sleeper_combo.currentIndexChanged.connect(self._sync_envelope_bearing_defaults)

        self.transition_group = QGroupBox("Transition zones (Design metrics)")
        self._set_compact_group(self.transition_group, max_width=520)
        self.transition_group.setCheckable(True)
        self.transition_group.setChecked(False)
        self.transition_group.toggled.connect(self._toggle_transition_mode)
        transition_layout = QFormLayout(self.transition_group)
        transition_layout.setContentsMargins(10, 5, 10, 5)
        self._configure_form_layout(transition_layout)

        self.transition_template_combo = QComboBox()
        self.transition_template_combo.addItem("Custom (manual)", "custom")
        self.transition_template_combo.addItem("Ballast → slab transition", "ballast_slab")
        self.transition_template_combo.addItem("Bridge → earthworks transition", "bridge_earthworks")
        self.transition_template_combo.addItem("Formation thickness change", "formation_change")
        self.transition_template_combo.addItem("Local stiff spot", "local_stiff")
        self.transition_template_combo.currentIndexChanged.connect(self._apply_transition_template)
        self._set_compact_combo(self.transition_template_combo, max_width=300)
        transition_layout.addRow("Template", self.transition_template_combo)

        self.transition_preset_combo = QComboBox()
        self.transition_preset_combo.addItem("Custom", "custom")
        self.transition_preset_combo.addItem("PWI ballasted reference (k₁=76.9 MN/m²)", "pwi")
        self.transition_preset_combo.currentIndexChanged.connect(self._apply_transition_preset)
        self._set_compact_combo(self.transition_preset_combo, max_width=300)
        transition_layout.addRow("Preset", self.transition_preset_combo)

        self.transition_run_mode_combo = QComboBox()
        self.transition_run_mode_combo.addItem("Single position", TransitionRunMode.SINGLE)
        self.transition_run_mode_combo.addItem("Worst-case envelope", TransitionRunMode.ENVELOPE)
        self.transition_run_mode_combo.currentIndexChanged.connect(self._toggle_transition_run_mode)
        self.transition_run_mode_combo.currentIndexChanged.connect(
            lambda _value: self._update_transition_estimate()
        )
        self._set_compact_combo(self.transition_run_mode_combo, max_width=240)
        transition_layout.addRow("Run mode", self.transition_run_mode_combo)

        self.transition_reference_input = UnitInput("m", decimals=3, minimum=-1.0e6, maximum=1.0e6)
        self.transition_reference_input.set_value(0.0)
        self.transition_reference_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_range_auto()
        )
        self.transition_reference_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_domain_auto()
        )
        self.transition_reference_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_estimate()
        )
        transition_layout.addRow("Reference position x_ref", self.transition_reference_input)

        self.transition_range_start_input = UnitInput("m", decimals=3, minimum=-1.0e6, maximum=1.0e6)
        self.transition_range_end_input = UnitInput("m", decimals=3, minimum=-1.0e6, maximum=1.0e6)
        self.transition_range_start_input.set_value(-10.0)
        self.transition_range_end_input.set_value(10.0)
        self.transition_range_start_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_domain_auto()
        )
        self.transition_range_end_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_domain_auto()
        )
        self.transition_range_auto_checkbox = QCheckBox("Auto")
        self.transition_range_auto_checkbox.setChecked(True)
        self.transition_range_auto_checkbox.toggled.connect(self._toggle_transition_range_auto)
        self.transition_range_start_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_estimate()
        )
        self.transition_range_end_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_estimate()
        )
        self.transition_range_container = QWidget()
        transition_range_layout = QHBoxLayout(self.transition_range_container)
        transition_range_layout.setContentsMargins(0, 0, 0, 0)
        transition_range_layout.addWidget(self.transition_range_start_input, stretch=1)
        transition_range_layout.addWidget(QLabel("to"))
        transition_range_layout.addWidget(self.transition_range_end_input, stretch=1)
        transition_range_layout.addWidget(self.transition_range_auto_checkbox)
        transition_layout.addRow("Movement range x_ref", self.transition_range_container)

        self.transition_step_input = UnitInput("m", decimals=3, minimum=0.001, maximum=100.0)
        self.transition_step_input.set_value(0.2)
        self.transition_step_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_estimate()
        )
        transition_layout.addRow("Movement increment Δx_ref", self.transition_step_input)

        self.transition_domain_start_input = UnitInput("m", decimals=3, minimum=-1.0e6, maximum=1.0e6)
        self.transition_domain_end_input = UnitInput("m", decimals=3, minimum=-1.0e6, maximum=1.0e6)
        self.transition_domain_start_input.set_value(-15.0)
        self.transition_domain_end_input.set_value(15.0)
        self.transition_domain_auto_checkbox = QCheckBox("Auto")
        self.transition_domain_auto_checkbox.setChecked(True)
        self.transition_domain_auto_checkbox.toggled.connect(self._toggle_transition_domain_auto)
        self.transition_domain_start_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_estimate()
        )
        self.transition_domain_end_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_estimate()
        )
        self.transition_domain_start_input.spinbox.valueChanged.connect(
            lambda _value: self._update_validation_hints()
        )
        self.transition_domain_end_input.spinbox.valueChanged.connect(
            lambda _value: self._update_validation_hints()
        )
        self.transition_domain_container = QWidget()
        transition_domain_layout = QHBoxLayout(self.transition_domain_container)
        transition_domain_layout.setContentsMargins(0, 0, 0, 0)
        transition_domain_layout.addWidget(self.transition_domain_start_input, stretch=1)
        transition_domain_layout.addWidget(QLabel("to"))
        transition_domain_layout.addWidget(self.transition_domain_end_input, stretch=1)
        transition_domain_layout.addWidget(self.transition_domain_auto_checkbox)
        transition_layout.addRow("Analysis domain x", self.transition_domain_container)

        self.transition_profile_combo = QComboBox()
        self.transition_profile_combo.addItem("Uniform", TransitionProfileType.UNIFORM)
        self.transition_profile_combo.addItem("Step change", TransitionProfileType.STEP)
        self.transition_profile_combo.addItem("Linear ramp", TransitionProfileType.RAMP)
        self.transition_profile_combo.addItem("Exponential ramp", TransitionProfileType.EXPONENTIAL)
        self.transition_profile_combo.addItem("Local stiff segment", TransitionProfileType.SEGMENT)
        self.transition_profile_combo.currentIndexChanged.connect(self._update_transition_profile_visibility)
        self.transition_profile_combo.currentIndexChanged.connect(
            lambda _value: self._update_transition_estimate()
        )
        self.transition_profile_combo.currentIndexChanged.connect(
            lambda _value: self._update_validation_hints()
        )
        self._set_compact_combo(self.transition_profile_combo, max_width=240)
        transition_layout.addRow("Profile type", self.transition_profile_combo)

        self.transition_k1_input = UnitInput("MN/m²", decimals=2, minimum=0.0, maximum=1.0e6)
        self.transition_k1_input.set_value(40.0)
        self.transition_k1_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_domain_auto()
        )
        self.transition_k1_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_range_auto()
        )
        self.transition_k1_input.spinbox.valueChanged.connect(
            lambda _value: self._update_validation_hints()
        )
        self.transition_k2_input = UnitInput("MN/m²", decimals=2, minimum=0.0, maximum=1.0e6)
        self.transition_k2_input.set_value(80.0)
        self.transition_k2_input.spinbox.valueChanged.connect(
            lambda _value: self._update_validation_hints()
        )
        transition_layout.addRow("k₁", self.transition_k1_input)
        transition_layout.addRow("k₂", self.transition_k2_input)

        self.transition_length_input = UnitInput("m", decimals=3, minimum=0.1, maximum=1.0e6)
        self.transition_length_input.set_value(10.0)
        self.transition_length_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_range_auto()
        )
        self.transition_length_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_domain_auto()
        )
        self.transition_length_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_estimate()
        )
        self.transition_length_auto_checkbox = QCheckBox("Auto")
        self.transition_length_auto_checkbox.setChecked(True)
        self.transition_length_auto_checkbox.toggled.connect(self._toggle_transition_length_auto)
        self.transition_length_container = QWidget()
        transition_length_layout = QHBoxLayout(self.transition_length_container)
        transition_length_layout.setContentsMargins(0, 0, 0, 0)
        transition_length_layout.addWidget(self.transition_length_input, stretch=1)
        transition_length_layout.addWidget(self.transition_length_auto_checkbox)
        transition_layout.addRow("Transition length Lₜ", self.transition_length_container)

        self.transition_segment_length_input = UnitInput("m", decimals=3, minimum=0.1, maximum=1.0e6)
        self.transition_segment_length_input.set_value(2.0)
        self.transition_segment_length_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_domain_auto()
        )
        self.transition_segment_length_input.spinbox.valueChanged.connect(
            lambda _value: self._update_transition_estimate()
        )
        transition_layout.addRow("Segment length L_c", self.transition_segment_length_input)

        self.transition_note_label = QLabel(
            "Envelope runs treat load positions as offsets from x_ref within the movement range."
        )
        self.transition_note_label.setWordWrap(False)
        self.transition_note_label.setStyleSheet("color: #555555;")
        transition_layout.addRow(self.transition_note_label)

        self.transition_estimate_label = QLabel("—")
        self.transition_estimate_label.setWordWrap(False)
        self.transition_estimate_label.setStyleSheet("color: #555555;")
        transition_layout.addRow("Run time estimate", self.transition_estimate_label)

        self.transition_group.setVisible(False)
        layout.addRow(self.transition_group)
        self.support_combo.currentIndexChanged.connect(self._sync_transition_defaults)
        self.support_combo.currentIndexChanged.connect(lambda _value: self._update_validation_hints())

        self.advanced_solver_checkbox = QCheckBox("Advanced solver (numerical)")
        self.advanced_solver_checkbox.toggled.connect(self._toggle_advanced_controls)
        layout.addRow(self.advanced_solver_checkbox)

        self.solver_backend_label = QLabel("Closed-form")
        layout.addRow("Solver backend", self.solver_backend_label)

        self.dynamic_container = QWidget()
        dynamic_layout = QFormLayout(self.dynamic_container)
        dynamic_layout.setContentsMargins(10, 0, 0, 0)
        self._configure_form_layout(dynamic_layout)

        self.speed_input = UnitInput("m/s", decimals=2, minimum=0.0, maximum=120.0)
        self.speed_input.set_value(30.0)
        dynamic_layout.addRow("Train speed", self.speed_input)

        self.damping_mode_combo = QComboBox()
        self.damping_mode_combo.addItem("Coefficient cₓ", "coefficient")
        self.damping_mode_combo.addItem("Ratio ζ", "ratio")
        self.damping_mode_combo.currentIndexChanged.connect(self._toggle_damping_inputs)
        self._set_compact_combo(self.damping_mode_combo, max_width=220)
        dynamic_layout.addRow("Foundation damping", self.damping_mode_combo)

        self.foundation_damping_model_combo = QComboBox()
        self.foundation_damping_model_combo.addItem("Viscous (c)", DampingModel.VISCOUS)
        self.foundation_damping_model_combo.addItem("Hysteretic (η)", DampingModel.HYSTERETIC)
        self.foundation_damping_model_combo.currentIndexChanged.connect(self._toggle_damping_inputs)
        self._set_compact_combo(self.foundation_damping_model_combo, max_width=220)
        dynamic_layout.addRow("Damping model", self.foundation_damping_model_combo)

        self.damping_coefficient_input = UnitInput("kN·s/m²", decimals=2, minimum=0.0, maximum=1.0e9)
        self.damping_coefficient_input.set_value(1_000.0)
        self.damping_ratio_input = UnitInput("—", decimals=3, minimum=0.0, maximum=5.0)
        self.damping_ratio_input.set_value(0.05)
        self.damping_loss_factor_input = UnitInput("—", decimals=3, minimum=0.0, maximum=5.0)
        self.damping_loss_factor_input.set_value(0.0)
        dynamic_layout.addRow("Damping coefficient cₓ", self.damping_coefficient_input)
        dynamic_layout.addRow("Damping ratio ζ", self.damping_ratio_input)
        dynamic_layout.addRow("Loss factor η", self.damping_loss_factor_input)

        self.dynamic_transition_group = QGroupBox("Dynamic transition settings")
        self._set_compact_group(self.dynamic_transition_group, max_width=520)
        transition_dynamic_layout = QFormLayout(self.dynamic_transition_group)
        transition_dynamic_layout.setContentsMargins(10, 5, 10, 5)
        self._configure_form_layout(transition_dynamic_layout)

        self.dynamic_transition_run_mode_combo = QComboBox()
        self.dynamic_transition_run_mode_combo.addItem("Single position", DynamicTransitionRunMode.SINGLE)
        self.dynamic_transition_run_mode_combo.addItem("Worst-case envelope", DynamicTransitionRunMode.ENVELOPE)
        self.dynamic_transition_run_mode_combo.currentIndexChanged.connect(
            self._toggle_dynamic_transition_run_mode
        )
        transition_dynamic_layout.addRow("Run mode", self.dynamic_transition_run_mode_combo)

        self.dynamic_transition_profile_combo = QComboBox()
        self.dynamic_transition_profile_combo.addItem("Uniform", DynamicTransitionProfileType.UNIFORM)
        self.dynamic_transition_profile_combo.addItem("Step change", DynamicTransitionProfileType.STEP)
        self.dynamic_transition_profile_combo.addItem("Linear ramp", DynamicTransitionProfileType.RAMP)
        self.dynamic_transition_profile_combo.addItem("Exponential ramp", DynamicTransitionProfileType.EXPONENTIAL)
        self.dynamic_transition_profile_combo.addItem("Local stiff segment", DynamicTransitionProfileType.SEGMENT)
        self.dynamic_transition_profile_combo.currentIndexChanged.connect(
            self._update_dynamic_transition_profile_visibility
        )
        transition_dynamic_layout.addRow("Profile type", self.dynamic_transition_profile_combo)

        self.dynamic_transition_solver_fidelity_combo = QComboBox()
        self.dynamic_transition_solver_fidelity_combo.addItem("Screening", "screening")
        self.dynamic_transition_solver_fidelity_combo.addItem("Full profile", "full_profile")
        self.dynamic_transition_solver_fidelity_combo.currentIndexChanged.connect(
            self._enforce_dynamic_transition_advanced_constraints
        )
        self.dynamic_transition_solver_fidelity_combo.setCurrentIndex(
            self.dynamic_transition_solver_fidelity_combo.findData("full_profile")
        )
        self.dynamic_transition_solver_fidelity_combo.setToolTip(
            "Full profile resolves non-uniform k(x) transitions; Screening is a faster uniform-k1 approximation."
        )
        transition_dynamic_layout.addRow("Solver fidelity", self.dynamic_transition_solver_fidelity_combo)

        self.dynamic_transition_k1_input = UnitInput("MN/m²", decimals=2, minimum=0.001, maximum=1.0e6)
        self.dynamic_transition_k1_input.set_value(40.0)
        self.dynamic_transition_k2_input = UnitInput("MN/m²", decimals=2, minimum=0.001, maximum=1.0e6)
        self.dynamic_transition_k2_input.set_value(80.0)
        transition_dynamic_layout.addRow("k₁", self.dynamic_transition_k1_input)
        transition_dynamic_layout.addRow("k₂", self.dynamic_transition_k2_input)

        self.dynamic_transition_length_input = UnitInput("m", decimals=3, minimum=0.001, maximum=1.0e6)
        self.dynamic_transition_length_input.set_value(10.0)
        self.dynamic_transition_segment_length_input = UnitInput("m", decimals=3, minimum=0.001, maximum=1.0e6)
        self.dynamic_transition_segment_length_input.set_value(2.0)
        transition_dynamic_layout.addRow("Transition length Lₜ", self.dynamic_transition_length_input)
        transition_dynamic_layout.addRow("Segment length L_c", self.dynamic_transition_segment_length_input)

        self.dynamic_transition_x_ref_input = UnitInput("m", decimals=3, minimum=-1.0e6, maximum=1.0e6)
        self.dynamic_transition_x_ref_input.set_value(0.0)
        transition_dynamic_layout.addRow("Reference position x_ref", self.dynamic_transition_x_ref_input)

        self.dynamic_transition_range_start_input = UnitInput("m", decimals=3, minimum=-1.0e6, maximum=1.0e6)
        self.dynamic_transition_range_end_input = UnitInput("m", decimals=3, minimum=-1.0e6, maximum=1.0e6)
        self.dynamic_transition_range_start_input.set_value(-10.0)
        self.dynamic_transition_range_end_input.set_value(10.0)
        self.dynamic_transition_range_container = QWidget()
        dynamic_range_layout = QHBoxLayout(self.dynamic_transition_range_container)
        dynamic_range_layout.setContentsMargins(0, 0, 0, 0)
        dynamic_range_layout.addWidget(self.dynamic_transition_range_start_input, stretch=1)
        dynamic_range_layout.addWidget(QLabel("to"))
        dynamic_range_layout.addWidget(self.dynamic_transition_range_end_input, stretch=1)
        transition_dynamic_layout.addRow("Movement range x_ref", self.dynamic_transition_range_container)

        self.dynamic_transition_step_input = UnitInput("m", decimals=3, minimum=0.001, maximum=100.0)
        self.dynamic_transition_step_input.set_value(0.2)
        transition_dynamic_layout.addRow("Movement increment Δx_ref", self.dynamic_transition_step_input)

        self.dynamic_transition_note_label = QLabel(
            "Envelope runs treat load positions as offsets from x_ref within the movement range."
        )
        self.dynamic_transition_note_label.setStyleSheet("color: #555555;")
        self.dynamic_transition_note_label.setWordWrap(False)
        transition_dynamic_layout.addRow(self.dynamic_transition_note_label)

        self.dynamic_transition_group.setVisible(False)
        dynamic_layout.addRow(self.dynamic_transition_group)

        self.dipped_joint_group = QGroupBox("Dipped joint inputs")
        self._set_compact_group(self.dipped_joint_group, max_width=520)
        dipped_layout = QFormLayout(self.dipped_joint_group)
        dipped_layout.setContentsMargins(10, 5, 10, 5)

        self.dipped_joint_reference_combo = QComboBox()
        self._refresh_dipped_joint_reference_combo()
        self._set_compact_combo(self.dipped_joint_reference_combo, max_width=220)
        apply_reference_button = QPushButton("Apply")
        apply_reference_button.clicked.connect(self._apply_dipped_joint_reference)
        reference_container = QWidget()
        reference_layout = QHBoxLayout(reference_container)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        reference_layout.addWidget(self.dipped_joint_reference_combo, stretch=1)
        reference_layout.addWidget(apply_reference_button)
        dipped_layout.addRow("Reference set", reference_container)

        self.dip_angle_input = UnitInput("mrad", decimals=3, minimum=0.0, maximum=100.0)
        self.dip_angle_input.setToolTip(
            "Total dip angle at the joint (2α).\n"
            "Typical values: 2-10 mrad for insulated joints.\n"
            "Use 0 for static case."
        )
        self.hertzian_stiffness_input = UnitInput("MN/m", decimals=2, minimum=0.0, maximum=5_000.0)
        self.hertzian_stiffness_input.setToolTip(
            "Hertzian contact stiffness between wheel and rail (kₕ).\n"
            "Typical values: 1000-2000 MN/m."
        )
        self.unsprung_mass_input = UnitInput("kg", decimals=2, minimum=0.0, maximum=2_000.0)
        self.unsprung_mass_input.setToolTip(
            "Unsprung mass of the wheel/axle assembly (mᵤ).\n"
            "Typical values: 250-500 kg per wheel."
        )
        self.track_mass_p1_input = UnitInput("kg", decimals=2, minimum=0.0, maximum=2_000.0)
        self.track_mass_p1_input.setToolTip(
            "Effective track mass at contact point P₁ (mᵀ₁).\n"
            "Typically rail mass within ~1m: 50-200 kg."
        )
        self.track_mass_p2_input = UnitInput("kg", decimals=2, minimum=0.0, maximum=2_000.0)
        self.track_mass_p2_input.setToolTip(
            "Equivalent track mass at second stage P₂ (mᵀ₂).\n"
            "Includes rail and sleeper contribution: 50-150 kg."
        )
        self.track_stiffness_p2_input = UnitInput("MN/m", decimals=2, minimum=0.0, maximum=500.0)
        self.track_stiffness_p2_input.setToolTip(
            "Equivalent track stiffness at second stage (kᵀ₂).\n"
            "Foundation stiffness: 50-150 MN/m."
        )
        self.track_damping_p2_input = UnitInput("kN·s/m", decimals=2, minimum=0.0, maximum=500.0)
        self.track_damping_p2_input.setToolTip(
            "Equivalent track damping coefficient (cᵀ).\n"
            "Typical values: 20-100 kN·s/m."
        )
        self.dip_angle_input.set_value(5.0)
        self.hertzian_stiffness_input.set_value(1_400.0)
        self.unsprung_mass_input.set_value(350.0)
        self.track_mass_p1_input.set_value(150.0)
        self.track_mass_p2_input.set_value(100.0)
        self.track_stiffness_p2_input.set_value(80.0)
        self.track_damping_p2_input.set_value(50.0)

        dipped_layout.addRow("Total dip angle (2α)", self.dip_angle_input)
        dipped_layout.addRow("Hertzian stiffness kₕ", self.hertzian_stiffness_input)
        dipped_layout.addRow("Unsprung mass mᵤ", self.unsprung_mass_input)
        dipped_layout.addRow("Effective track mass mᵀ₁", self.track_mass_p1_input)
        dipped_layout.addRow("Equivalent track mass mᵀ₂", self.track_mass_p2_input)
        dipped_layout.addRow("Equivalent track stiffness kᵀ₂", self.track_stiffness_p2_input)
        dipped_layout.addRow("Equivalent track damping cᵀ", self.track_damping_p2_input)

        self.dipped_joint_group.setVisible(False)
        dynamic_layout.addRow(self.dipped_joint_group)

        self.probe_locations_input = QLineEdit()
        self.probe_locations_input.setPlaceholderText("Comma-separated, e.g., 0, -5, 5")
        self.probe_locations_input.setText("0, -5, 5")
        self.probe_locations_input.textChanged.connect(self._refresh_probe_selection)
        self._set_compact_line_edit(self.probe_locations_input, max_width=280)
        dynamic_layout.addRow("Probe locations xₚ", self.probe_locations_input)

        self.probe_selection_combo = QComboBox()
        self.probe_selection_combo.currentIndexChanged.connect(self._update_dynamic_probe_plots)
        self._set_compact_combo(self.probe_selection_combo, max_width=240)
        dynamic_layout.addRow("Plot probe", self.probe_selection_combo)

        self.time_window_auto_checkbox = QCheckBox("Auto")
        self.time_window_auto_checkbox.setChecked(True)
        self.time_window_auto_checkbox.toggled.connect(self._toggle_time_window_auto)
        self.time_window_container = QWidget()
        time_window_layout = QHBoxLayout(self.time_window_container)
        time_window_layout.setContentsMargins(0, 0, 0, 0)
        self.time_window_input = UnitInput("s", decimals=2, minimum=0.1, maximum=1.0e4)
        time_window_layout.addWidget(self.time_window_input, stretch=1)
        time_window_layout.addWidget(self.time_window_auto_checkbox)
        self.sample_rate_input = UnitInput("Hz", decimals=1, minimum=1.0, maximum=5_000.0)
        self.sample_rate_input.set_value(250.0)
        dynamic_layout.addRow("Time window T", self.time_window_container)
        dynamic_layout.addRow("Sampling rate fₛ", self.sample_rate_input)

        self.dynamic_advanced_group = QGroupBox("Dynamic advanced")
        self._set_compact_group(self.dynamic_advanced_group, max_width=520)
        self.dynamic_advanced_group.setCheckable(True)
        self.dynamic_advanced_group.setChecked(False)
        self.dynamic_advanced_group.toggled.connect(self._toggle_dynamic_advanced_controls)
        advanced_dynamic_layout = QFormLayout(self.dynamic_advanced_group)
        self._configure_form_layout(advanced_dynamic_layout)

        self.domain_length_input = UnitInput("m", decimals=1, minimum=10.0, maximum=1.0e5)
        self.domain_length_input.set_value(100.0)
        self.spatial_step_input = UnitInput("m", decimals=3, minimum=0.001, maximum=10.0)
        self.spatial_step_input.set_value(0.05)
        self.psd_segment_length_input = QSpinBox()
        self.psd_segment_length_input.setRange(16, 8192)
        self.psd_segment_length_input.setValue(256)
        self.psd_overlap_input = UnitInput("—", decimals=2, minimum=0.0, maximum=0.9)
        self.psd_overlap_input.set_value(0.5)
        self.dynamic_excitation_mode_combo = QComboBox()
        self.dynamic_excitation_mode_combo.addItem("Moving load", DynamicExcitationMode.MOVING_LOAD)
        self.dynamic_excitation_mode_combo.addItem("Moving oscillator (advanced)", DynamicExcitationMode.MOVING_OSCILLATOR)
        self.dynamic_excitation_mode_combo.currentIndexChanged.connect(self._toggle_dynamic_extra_inputs)
        self.dynamic_boundary_mode_combo = QComboBox()
        self.dynamic_boundary_mode_combo.addItem("Zero pad (legacy default)", DynamicBoundaryMode.ZERO_PAD)
        self.dynamic_boundary_mode_combo.addItem("Periodic wrap", DynamicBoundaryMode.PERIODIC_WRAP)
        self.oscillator_unsprung_mass_input = UnitInput("kg", decimals=2, minimum=0.0, maximum=5_000.0)
        self.oscillator_unsprung_mass_input.set_value(350.0)
        self.oscillator_stiffness_input = UnitInput("MN/m", decimals=2, minimum=0.0, maximum=5_000.0)
        self.oscillator_stiffness_input.set_value(120.0)
        self.oscillator_damping_input = UnitInput("kN·s/m", decimals=2, minimum=0.0, maximum=2_000.0)
        self.oscillator_damping_input.set_value(20.0)
        self.irregularity_mode_combo = QComboBox()
        self.irregularity_mode_combo.addItem("Off", None)
        self.irregularity_mode_combo.addItem("Measured profile", IrregularityMode.PROFILE)
        self.irregularity_mode_combo.addItem("Synthetic PSD", IrregularityMode.SYNTHETIC_PSD)
        self.irregularity_mode_combo.currentIndexChanged.connect(self._toggle_dynamic_extra_inputs)
        self.irregularity_profile_x_input = QLineEdit()
        self.irregularity_profile_x_input.setPlaceholderText("x positions, m (e.g., -20,-10,0,10,20)")
        self._set_compact_line_edit(self.irregularity_profile_x_input, max_width=420)
        self.irregularity_profile_z_input = QLineEdit()
        self.irregularity_profile_z_input.setPlaceholderText("z levels, mm (e.g., 0,0.2,-0.1,0.15,0)")
        self._set_compact_line_edit(self.irregularity_profile_z_input, max_width=420)
        self.irregularity_psd_level_input = UnitInput("m³", decimals=8, minimum=0.0, maximum=1.0)
        self.irregularity_psd_level_input.set_value(1.0e-6)
        self.irregularity_seed_input = QSpinBox()
        self.irregularity_seed_input.setRange(0, 1_000_000)
        self.irregularity_seed_input.setValue(0)

        default_time_window = self.domain_length_input.value() / max(self.speed_input.value(), 0.1) * 1.2
        self.time_window_input.set_value(default_time_window)
        self.time_window_input.setEnabled(False)
        self.speed_input.spinbox.valueChanged.connect(self._update_time_window_auto)
        self.domain_length_input.spinbox.valueChanged.connect(self._update_time_window_auto)

        advanced_dynamic_layout.addRow("Domain length Lξ", self.domain_length_input)
        advanced_dynamic_layout.addRow("Spatial resolution Δξ", self.spatial_step_input)
        advanced_dynamic_layout.addRow("PSD segment length", self.psd_segment_length_input)
        advanced_dynamic_layout.addRow("PSD overlap", self.psd_overlap_input)
        advanced_dynamic_layout.addRow("Excitation mode", self.dynamic_excitation_mode_combo)
        advanced_dynamic_layout.addRow("Boundary mode", self.dynamic_boundary_mode_combo)
        advanced_dynamic_layout.addRow("Oscillator unsprung mass", self.oscillator_unsprung_mass_input)
        advanced_dynamic_layout.addRow("Oscillator suspension stiffness", self.oscillator_stiffness_input)
        advanced_dynamic_layout.addRow("Oscillator suspension damping", self.oscillator_damping_input)
        advanced_dynamic_layout.addRow("Irregularity mode", self.irregularity_mode_combo)
        advanced_dynamic_layout.addRow("Irregularity profile x", self.irregularity_profile_x_input)
        advanced_dynamic_layout.addRow("Irregularity profile z", self.irregularity_profile_z_input)
        advanced_dynamic_layout.addRow("Synthetic PSD level", self.irregularity_psd_level_input)
        advanced_dynamic_layout.addRow("Synthetic seed", self.irregularity_seed_input)
        self._toggle_dynamic_extra_inputs()

        dynamic_layout.addRow(self.dynamic_advanced_group)

        layout.addRow(self.dynamic_container)

        self.special_container = QWidget()
        special_layout = QFormLayout(self.special_container)
        special_layout.setContentsMargins(10, 0, 0, 0)
        self._configure_form_layout(special_layout)

        self.floating_slab_group = QGroupBox("Floating slab isolation")
        self._set_compact_group(self.floating_slab_group, max_width=520)
        slab_layout = QFormLayout(self.floating_slab_group)
        slab_layout.setContentsMargins(10, 5, 10, 5)
        self._configure_form_layout(slab_layout)

        self.floating_slab_mass_input = UnitInput("kg", decimals=1, minimum=1.0, maximum=1.0e9)
        self.floating_slab_mass_input.set_value(20_000.0)
        slab_layout.addRow("Slab mass", self.floating_slab_mass_input)

        self.floating_slab_stiffness_input = UnitInput("kN/m", decimals=2, minimum=0.0, maximum=1.0e9)
        self.floating_slab_stiffness_input.set_value(50_000.0)
        slab_layout.addRow("Isolator stiffness", self.floating_slab_stiffness_input)

        self.floating_slab_damping_input = UnitInput("kN·s/m", decimals=2, minimum=0.0, maximum=1.0e9)
        self.floating_slab_damping_input.set_value(200.0)
        slab_layout.addRow("Isolator damping", self.floating_slab_damping_input)

        self.floating_slab_static_load_input = UnitInput("kN", decimals=2, minimum=0.0, maximum=1.0e9)
        self.floating_slab_static_load_input.set_value(100.0)
        slab_layout.addRow("Static load", self.floating_slab_static_load_input)

        self.floating_slab_freq_min_input = UnitInput("Hz", decimals=2, minimum=0.0, maximum=1.0e4)
        self.floating_slab_freq_min_input.set_value(0.0)
        self.floating_slab_freq_max_input = UnitInput("Hz", decimals=2, minimum=0.0, maximum=1.0e4)
        self.floating_slab_freq_max_input.set_value(50.0)
        slab_layout.addRow("Frequency min f_min", self.floating_slab_freq_min_input)
        slab_layout.addRow("Frequency max f_max", self.floating_slab_freq_max_input)

        self.floating_slab_freq_points_input = QSpinBox()
        self.floating_slab_freq_points_input.setRange(10, 2000)
        self.floating_slab_freq_points_input.setValue(200)
        slab_layout.addRow("Frequency points", self.floating_slab_freq_points_input)

        special_layout.addRow(self.floating_slab_group)
        layout.addRow(self.special_container)

        self.advanced_container = QWidget()
        advanced_layout = QFormLayout(self.advanced_container)
        advanced_layout.setContentsMargins(10, 0, 0, 0)
        self._configure_form_layout(advanced_layout)

        self.foundation_model_combo = QComboBox()
        self.foundation_model_combo.addItem("Winkler (single-layer)", FoundationModelType.WINKLER)
        self.foundation_model_combo.addItem("Series (railpad + trackbed)", FoundationModelType.SERIES)
        self.foundation_model_combo.addItem("Sleeper-mass (3-layer)", FoundationModelType.SLEEPER_MASS)
        self.foundation_model_combo.currentIndexChanged.connect(self._toggle_foundation_model_inputs)
        advanced_layout.addRow("Foundation model", self.foundation_model_combo)

        self.foundation_model_note_label = QLabel(
            "Sleeper-mass uses a static equivalent stiffness; mass effects are ignored in static."
        )
        self.foundation_model_note_label.setWordWrap(True)
        self.foundation_model_note_label.setStyleSheet("color: #555555;")
        self.foundation_model_note_label.setVisible(False)
        advanced_layout.addRow(self.foundation_model_note_label)

        self.railpad_stiffness_input = UnitInput("kN/m", decimals=2, minimum=0.0, maximum=1.0e9)
        self.railpad_stiffness_input.set_value(120_000.0)
        self.railpad_damping_input = UnitInput("kN·s/m", decimals=2, minimum=0.0, maximum=1.0e9)
        self.railpad_damping_input.set_value(0.0)
        self.railpad_loss_factor_input = UnitInput("—", decimals=3, minimum=0.0, maximum=5.0)
        self.railpad_loss_factor_input.set_value(0.0)
        self.trackbed_stiffness_input = UnitInput("kN/m", decimals=2, minimum=0.0, maximum=1.0e9)
        self.trackbed_stiffness_input.set_value(80_000.0)
        self.trackbed_damping_input = UnitInput("kN·s/m", decimals=2, minimum=0.0, maximum=1.0e9)
        self.trackbed_damping_input.set_value(0.0)
        self.trackbed_loss_factor_input = UnitInput("—", decimals=3, minimum=0.0, maximum=5.0)
        self.trackbed_loss_factor_input.set_value(0.0)
        self.foundation_damping_model_static_combo = QComboBox()
        self.foundation_damping_model_static_combo.addItem("Viscous (c)", DampingModel.VISCOUS)
        self.foundation_damping_model_static_combo.addItem("Hysteretic (η)", DampingModel.HYSTERETIC)
        self.foundation_damping_model_static_combo.currentIndexChanged.connect(
            self._toggle_foundation_damping_inputs
        )
        self.sleeper_mass_input = UnitInput("kg", decimals=2, minimum=0.0, maximum=1.0e6)
        self.sleeper_mass_input.set_value(250.0)
        advanced_layout.addRow("Railpad stiffness k_p", self.railpad_stiffness_input)
        advanced_layout.addRow("Railpad damping c_p", self.railpad_damping_input)
        advanced_layout.addRow("Railpad loss factor η_p", self.railpad_loss_factor_input)
        advanced_layout.addRow("Trackbed stiffness k_b", self.trackbed_stiffness_input)
        advanced_layout.addRow("Trackbed damping c_b", self.trackbed_damping_input)
        advanced_layout.addRow("Trackbed loss factor η_b", self.trackbed_loss_factor_input)
        advanced_layout.addRow("Foundation damping model", self.foundation_damping_model_static_combo)
        advanced_layout.addRow("Sleeper mass m_s", self.sleeper_mass_input)

        self.beam_theory_combo = QComboBox()
        self.beam_theory_combo.addItem("Euler–Bernoulli", BeamTheory.EULER)
        self.beam_theory_combo.addItem("Timoshenko", BeamTheory.TIMOSHENKO)
        self.beam_theory_combo.currentIndexChanged.connect(self._toggle_beam_theory_inputs)
        advanced_layout.addRow("Beam theory", self.beam_theory_combo)

        self.poisson_ratio_input = UnitInput("—", decimals=3, minimum=0.0, maximum=0.49)
        self.poisson_ratio_input.set_value(0.3)
        self.kappa_input = UnitInput("—", decimals=3, minimum=0.1, maximum=1.0)
        self.kappa_input.set_value(0.4)
        self.rail_area_input = UnitInput("cm²", decimals=3, minimum=0.0, maximum=1.0e6)
        self.rail_area_input.set_value(0.0)
        advanced_layout.addRow("Poisson ratio ν", self.poisson_ratio_input)
        advanced_layout.addRow("Shear correction κ", self.kappa_input)
        advanced_layout.addRow("Rail area override A", self.rail_area_input)

        self.nonuniform_profile_checkbox = QCheckBox("Use nonuniform foundation k(x)")
        self.nonuniform_profile_checkbox.toggled.connect(self._toggle_foundation_profile_controls)
        advanced_layout.addRow(self.nonuniform_profile_checkbox)

        self.profile_type_combo = QComboBox()
        self.profile_type_combo.addItem("Uniform", FoundationProfileType.UNIFORM)
        self.profile_type_combo.addItem("Step", FoundationProfileType.STEP)
        self.profile_type_combo.addItem("Ramp", FoundationProfileType.RAMP)
        self.profile_type_combo.currentIndexChanged.connect(self._update_profile_input_visibility)
        advanced_layout.addRow("Profile type", self.profile_type_combo)

        self.profile_k1_input = UnitInput("MN/m²", decimals=2, minimum=0.0, maximum=1.0e6)
        self.profile_k1_input.set_value(40.0)
        self.profile_k2_input = UnitInput("MN/m²", decimals=2, minimum=0.0, maximum=1.0e6)
        self.profile_k2_input.set_value(80.0)
        self.profile_x_start_input = UnitInput("mm", decimals=1, minimum=-10_000.0, maximum=10_000.0)
        self.profile_x_start_input.set_value(0.0)
        self.profile_x_end_input = UnitInput("mm", decimals=1, minimum=-10_000.0, maximum=10_000.0)
        self.profile_x_end_input.set_value(2000.0)
        advanced_layout.addRow("k₁", self.profile_k1_input)
        advanced_layout.addRow("k₂", self.profile_k2_input)
        advanced_layout.addRow("Transition start", self.profile_x_start_input)
        advanced_layout.addRow("Transition end", self.profile_x_end_input)

        self.discrete_supports_checkbox = QCheckBox("Discrete sleepers/pads")
        self.discrete_supports_checkbox.toggled.connect(self._toggle_discrete_controls)
        advanced_layout.addRow(self.discrete_supports_checkbox)

        self.pad_stiffness_input = UnitInput("kN/m", decimals=2, minimum=0.0, maximum=1.0e9)
        self.pad_stiffness_input.set_value(120_000.0)
        self.pad_damping_input = UnitInput("kN·s/m", decimals=2, minimum=0.0, maximum=1.0e9)
        self.pad_damping_input.set_value(0.0)
        self.pad_loss_factor_input = UnitInput("—", decimals=3, minimum=0.0, maximum=5.0)
        self.pad_loss_factor_input.set_value(0.0)
        self.nodes_between_sleepers_input = QSpinBox()
        self.nodes_between_sleepers_input.setRange(2, 50)
        self.nodes_between_sleepers_input.setValue(10)
        self.nodes_between_sleepers_input.valueChanged.connect(
            lambda _value: self._update_envelope_estimate()
        )
        self.nodes_between_sleepers_input.valueChanged.connect(
            lambda _value: self._update_transition_estimate()
        )
        advanced_layout.addRow("Pad stiffness Kz", self.pad_stiffness_input)
        advanced_layout.addRow("Pad damping Cz", self.pad_damping_input)
        advanced_layout.addRow("Pad loss factor ηz", self.pad_loss_factor_input)
        advanced_layout.addRow("Nodes between sleepers", self.nodes_between_sleepers_input)

        self.two_rail_checkbox = QCheckBox("Two-rail coupled analysis")
        self.two_rail_checkbox.toggled.connect(self._toggle_two_rail_controls)
        advanced_layout.addRow(self.two_rail_checkbox)

        self.coupling_stiffness_input = UnitInput("kN/m", decimals=2, minimum=0.0, maximum=1.0e9)
        self.coupling_stiffness_input.set_value(50_000.0)
        advanced_layout.addRow("Coupling stiffness", self.coupling_stiffness_input)

        self.asymmetric_load_checkbox = QCheckBox("Use asymmetric right rail load")
        self.asymmetric_load_checkbox.toggled.connect(self._toggle_right_load_controls)
        advanced_layout.addRow(self.asymmetric_load_checkbox)

        self.right_load_magnitude_input = UnitInput("kN", decimals=2, minimum=0.0, maximum=1.0e6)
        self.right_load_magnitude_input.set_value(100.0)
        self.right_load_position_input = UnitInput("mm", decimals=1, minimum=-10_000.0, maximum=10_000.0)
        self.right_load_position_input.set_value(0.0)
        advanced_layout.addRow("Right rail load", self.right_load_magnitude_input)
        advanced_layout.addRow("Right rail position", self.right_load_position_input)

        self._refresh_load_cases()

        self.pasternak_checkbox = QCheckBox("Use Pasternak shear layer (k_g)")
        self.pasternak_checkbox.toggled.connect(self._toggle_pasternak_controls)
        advanced_layout.addRow(self.pasternak_checkbox)
        self.pasternak_warning_label = QLabel(
            "Pasternak shear layer is not supported with Timoshenko beam theory."
        )
        self.pasternak_warning_label.setStyleSheet("color: #b3261e;")
        self.pasternak_warning_label.setWordWrap(True)
        self.pasternak_warning_label.setVisible(False)
        advanced_layout.addRow("", self.pasternak_warning_label)

        self.pasternak_input = UnitInput("kN", decimals=2, minimum=0.0, maximum=1.0e9)
        self.pasternak_input.set_value(10_000.0)
        advanced_layout.addRow("k_g", self.pasternak_input)

        layout.addRow(self.advanced_container)

        self.run_button = QPushButton("Run analysis")
        self.run_button.clicked.connect(self._run_analysis)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_long_run)
        self.reset_button = QPushButton("Reset application")
        self.reset_button.clicked.connect(self._reset_application)
        run_container = QWidget()
        run_layout = QHBoxLayout(run_container)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.cancel_button)
        run_layout.addWidget(self.reset_button)
        run_layout.addStretch()
        layout.addRow(run_container)

        self.validation_hint_label = QLabel("—")
        self.validation_hint_label.setWordWrap(False)
        self.validation_hint_label.setStyleSheet("color: #8a5a00;")
        layout.addRow(self.validation_hint_label)

        export_layout = QHBoxLayout()
        self.export_analysis_button = VisibilityStatePushButton("Export analysis CSV")
        self.export_analysis_button.setEnabled(False)
        self.export_analysis_button.clicked.connect(self._export_analysis_csv)
        self.export_config_button = VisibilityStatePushButton("Export run config (JSON)")
        self.export_config_button.setEnabled(False)
        self.export_config_button.clicked.connect(self._export_analysis_config)
        self.export_sleeper_button = VisibilityStatePushButton("Export sleeper CSV")
        self.export_sleeper_button.setEnabled(False)
        self.export_sleeper_button.clicked.connect(self._export_sleeper_csv)
        self.export_transition_metrics_button = VisibilityStatePushButton("Export transition metrics CSV")
        self.export_transition_metrics_button.setEnabled(False)
        self.export_transition_metrics_button.clicked.connect(self._export_transition_metrics_csv)
        self.export_transition_series_button = VisibilityStatePushButton("Export transition series CSV")
        self.export_transition_series_button.setEnabled(False)
        self.export_transition_series_button.clicked.connect(self._export_transition_series_csv)
        self.export_transition_config_button = VisibilityStatePushButton("Export transition run JSON")
        self.export_transition_config_button.setEnabled(False)
        self.export_transition_config_button.clicked.connect(self._export_transition_config)
        self.export_dynamic_time_button = VisibilityStatePushButton("Export dynamic time CSV")
        self.export_dynamic_time_button.setEnabled(False)
        self.export_dynamic_time_button.clicked.connect(self._export_dynamic_time_csv)
        self.export_dynamic_fft_button = VisibilityStatePushButton("Export dynamic FFT CSV")
        self.export_dynamic_fft_button.setEnabled(False)
        self.export_dynamic_fft_button.clicked.connect(self._export_dynamic_fft_csv)
        self.export_dynamic_psd_button = VisibilityStatePushButton("Export dynamic PSD CSV")
        self.export_dynamic_psd_button.setEnabled(False)
        self.export_dynamic_psd_button.clicked.connect(self._export_dynamic_psd_csv)
        self.export_dynamic_transition_metrics_button = VisibilityStatePushButton("Export dyn transition metrics CSV")
        self.export_dynamic_transition_metrics_button.setEnabled(False)
        self.export_dynamic_transition_metrics_button.clicked.connect(
            self._export_dynamic_transition_metrics_csv
        )
        self.export_dynamic_transition_series_button = VisibilityStatePushButton("Export dyn transition series CSV")
        self.export_dynamic_transition_series_button.setEnabled(False)
        self.export_dynamic_transition_series_button.clicked.connect(
            self._export_dynamic_transition_series_csv
        )
        self.export_dynamic_transition_config_button = VisibilityStatePushButton("Export dyn transition run JSON")
        self.export_dynamic_transition_config_button.setEnabled(False)
        self.export_dynamic_transition_config_button.clicked.connect(
            self._export_dynamic_transition_config
        )
        self.export_dipped_joint_button = VisibilityStatePushButton("Export dipped joint CSV")
        self.export_dipped_joint_button.setEnabled(False)
        self.export_dipped_joint_button.clicked.connect(self._export_dipped_joint_csv)
        export_layout.addWidget(self.export_analysis_button)
        export_layout.addWidget(self.export_config_button)
        export_layout.addWidget(self.export_sleeper_button)
        export_layout.addWidget(self.export_transition_metrics_button)
        export_layout.addWidget(self.export_transition_series_button)
        export_layout.addWidget(self.export_transition_config_button)
        export_layout.addWidget(self.export_dynamic_time_button)
        export_layout.addWidget(self.export_dynamic_fft_button)
        export_layout.addWidget(self.export_dynamic_psd_button)
        export_layout.addWidget(self.export_dynamic_transition_metrics_button)
        export_layout.addWidget(self.export_dynamic_transition_series_button)
        export_layout.addWidget(self.export_dynamic_transition_config_button)
        export_layout.addWidget(self.export_dipped_joint_button)
        layout.addRow(export_layout)

        self._toggle_advanced_controls(False)
        self._toggle_dynamic_advanced_controls(False)
        self._toggle_damping_inputs()
        self._toggle_foundation_damping_inputs()
        self._update_dynamic_transition_profile_visibility()
        self._toggle_dynamic_transition_run_mode()
        self._refresh_probe_selection()
        self._toggle_dynamic_mode()
        self._toggle_analysis_type()
        self._sync_envelope_bearing_defaults()
        self._sync_transition_defaults()
        self._update_transition_profile_visibility()
        self._toggle_transition_run_mode()
        self._update_transition_length_auto()
        self._update_transition_domain_auto()
        self._update_solver_backend_label()
        self._update_overlay_state()
        self._update_envelope_estimate()
        self._update_transition_estimate()
        self._update_validation_hints()
        return group

    def _toggle_advanced_controls(self, enabled: bool) -> None:
        self.advanced_container.setVisible(enabled)
        self._update_solver_backend_label()
        if not enabled:
            self.foundation_model_combo.setCurrentIndex(0)
            self.beam_theory_combo.setCurrentIndex(0)
            self.nonuniform_profile_checkbox.setChecked(False)
            self.discrete_supports_checkbox.setChecked(False)
            self.two_rail_checkbox.setChecked(False)
            self.asymmetric_load_checkbox.setChecked(False)
            self.pasternak_checkbox.setChecked(False)
            self.foundation_damping_model_static_combo.setCurrentIndex(0)
            self._toggle_foundation_model_inputs()
            self._toggle_beam_theory_inputs()
            self._toggle_foundation_profile_controls(False)
            self._toggle_discrete_controls(False)
            self._toggle_two_rail_controls(False)
            self._toggle_right_load_controls(False)
            self._toggle_pasternak_controls(False)
            self._toggle_foundation_damping_inputs()
        self._update_envelope_estimate()
        self._update_transition_estimate()
        self._toggle_foundation_model_inputs()
        self._toggle_beam_theory_inputs()
        self._toggle_foundation_profile_controls(self.nonuniform_profile_checkbox.isChecked())
        self._toggle_discrete_controls(self.discrete_supports_checkbox.isChecked())
        self._toggle_two_rail_controls(self.two_rail_checkbox.isChecked())
        self._toggle_right_load_controls(
            self.two_rail_checkbox.isChecked() and self.asymmetric_load_checkbox.isChecked()
        )
        self._toggle_pasternak_controls(self.pasternak_checkbox.isChecked())
        self._toggle_foundation_damping_inputs()

    def _update_solver_backend_label(self) -> None:
        if not hasattr(self, "solver_backend_label"):
            return
        backend = "Closed-form"
        if hasattr(self, "transition_group") and self.transition_group.isChecked():
            if self._transition_requires_numerical() or self.advanced_solver_checkbox.isChecked():
                backend = "Numerical (Transition)"
            else:
                backend = "Closed-form (Transition)"
            self.solver_backend_label.setText(backend)
            return
        if hasattr(self, "static_mode_combo"):
            static_mode = self.static_mode_combo.currentData()
            if static_mode == StaticMode.ENVELOPE_CLOSED_FORM:
                backend = "Closed-form (Envelope)"
            elif static_mode == StaticMode.ENVELOPE_NUMERICAL:
                backend = "Numerical (Envelope)"
            elif self.advanced_solver_checkbox.isChecked():
                backend = "Numerical"
        elif self.advanced_solver_checkbox.isChecked():
            backend = "Numerical"
        self.solver_backend_label.setText(backend)

    def _toggle_foundation_model_inputs(self) -> None:
        model = self.foundation_model_combo.currentData()
        is_multilayer = model in (FoundationModelType.SERIES, FoundationModelType.SLEEPER_MASS)
        self.railpad_stiffness_input.setEnabled(is_multilayer)
        self.railpad_damping_input.setEnabled(is_multilayer)
        self.railpad_loss_factor_input.setEnabled(is_multilayer)
        self.trackbed_stiffness_input.setEnabled(is_multilayer)
        self.trackbed_damping_input.setEnabled(is_multilayer)
        self.sleeper_mass_input.setEnabled(model == FoundationModelType.SLEEPER_MASS)
        if hasattr(self, "foundation_model_note_label"):
            self.foundation_model_note_label.setVisible(model == FoundationModelType.SLEEPER_MASS)
        self.nonuniform_profile_checkbox.setEnabled(not is_multilayer)
        self.discrete_supports_checkbox.setEnabled(not is_multilayer)
        if is_multilayer:
            self.nonuniform_profile_checkbox.setChecked(False)
            self.discrete_supports_checkbox.setChecked(False)
        if is_multilayer:
            self._sync_multilayer_defaults()
        self._toggle_foundation_damping_inputs()

    def _toggle_beam_theory_inputs(self) -> None:
        is_timoshenko = self.beam_theory_combo.currentData() == BeamTheory.TIMOSHENKO
        self.poisson_ratio_input.setEnabled(is_timoshenko)
        self.kappa_input.setEnabled(is_timoshenko)
        self.rail_area_input.setEnabled(is_timoshenko)
        self.two_rail_checkbox.setEnabled(not is_timoshenko)
        if is_timoshenko:
            self.two_rail_checkbox.setChecked(False)
        if is_timoshenko and self.pasternak_checkbox.isChecked():
            self.pasternak_checkbox.setChecked(False)
        self._update_pasternak_warning()

    def _sync_multilayer_defaults(self) -> None:
        pad = self.pad_combo.currentData()
        support = self.support_combo.currentData()
        spacing_m = mm_to_m(self.sleeper_spacing_input.value())
        if pad is not None:
            self.railpad_stiffness_input.set_value(n_to_kn(pad.stiffness_newtons_per_meter))
        if support is not None and spacing_m > 0:
            trackbed_support_n_per_m = support.foundation_modulus_n_per_m2 * spacing_m
            self.trackbed_stiffness_input.set_value(n_to_kn(trackbed_support_n_per_m))

    def _toggle_several_loads(self, enabled: bool) -> None:
        if enabled:
            self.train_loads_checkbox.blockSignals(True)
            self.train_loads_checkbox.setChecked(False)
            self.train_loads_checkbox.blockSignals(False)
            if hasattr(self, "as5100_loads_checkbox"):
                self.as5100_loads_checkbox.blockSignals(True)
                self.as5100_loads_checkbox.setChecked(False)
                self.as5100_loads_checkbox.blockSignals(False)
        self._sync_load_input_state()

    def _toggle_train_loads(self, enabled: bool) -> None:
        if enabled:
            self.several_loads_checkbox.blockSignals(True)
            self.several_loads_checkbox.setChecked(False)
            self.several_loads_checkbox.blockSignals(False)
            if hasattr(self, "as5100_loads_checkbox"):
                self.as5100_loads_checkbox.blockSignals(True)
                self.as5100_loads_checkbox.setChecked(False)
                self.as5100_loads_checkbox.blockSignals(False)
        self._sync_load_input_state()

    def _toggle_as5100_loads(self, enabled: bool) -> None:
        if enabled:
            self.several_loads_checkbox.blockSignals(True)
            self.several_loads_checkbox.setChecked(False)
            self.several_loads_checkbox.blockSignals(False)
            self.train_loads_checkbox.blockSignals(True)
            self.train_loads_checkbox.setChecked(False)
            self.train_loads_checkbox.blockSignals(False)
            self._sync_as5100_model_defaults()
        self._sync_load_input_state()

    def _sync_load_input_state(self) -> None:
        train_enabled = self.train_loads_checkbox.isChecked()
        multiple_enabled = self.several_loads_checkbox.isChecked()
        as5100_enabled = (
            self.as5100_loads_checkbox.isChecked()
            if hasattr(self, "as5100_loads_checkbox")
            else False
        )
        if train_enabled and multiple_enabled:
            self.several_loads_checkbox.blockSignals(True)
            self.several_loads_checkbox.setChecked(False)
            self.several_loads_checkbox.blockSignals(False)
            multiple_enabled = False
        if as5100_enabled and (train_enabled or multiple_enabled):
            self.train_loads_checkbox.blockSignals(True)
            self.train_loads_checkbox.setChecked(False)
            self.train_loads_checkbox.blockSignals(False)
            self.several_loads_checkbox.blockSignals(True)
            self.several_loads_checkbox.setChecked(False)
            self.several_loads_checkbox.blockSignals(False)
            train_enabled = False
            multiple_enabled = False

        self.several_loads_checkbox.setEnabled(not train_enabled and not as5100_enabled)
        self.train_loads_checkbox.setEnabled(not multiple_enabled and not as5100_enabled)
        if hasattr(self, "as5100_loads_checkbox"):
            self.as5100_loads_checkbox.setEnabled(not train_enabled and not multiple_enabled)

        self.train_loads_group.setVisible(train_enabled)
        self.wheel_loads_group.setVisible(multiple_enabled)
        if hasattr(self, "as5100_loads_group"):
            self.as5100_loads_group.setVisible(as5100_enabled)

        single_enabled = not train_enabled and not multiple_enabled and not as5100_enabled
        self.load_case_combo.setEnabled(single_enabled)
        self.load_magnitude_input.setEnabled(single_enabled)
        self.load_position_input.setEnabled(single_enabled)

        if train_enabled:
            self.train_axle_load_input.set_value(self.load_magnitude_input.value())
        if multiple_enabled and self.wheel_loads_widget.rows() == 0:
            self.wheel_loads_widget.add_row(
                load_kn=self.load_magnitude_input.value(),
                position_m=mm_to_m(self.load_position_input.value()),
            )
        if hasattr(self, "static_mode_combo"):
            is_envelope = self.static_mode_combo.currentData() in (
                StaticMode.ENVELOPE_CLOSED_FORM,
                StaticMode.ENVELOPE_NUMERICAL,
            )
            self._toggle_envelope_train_reference(is_envelope)
        if hasattr(self, "transition_group") and self.transition_group.isChecked():
            self._update_transition_domain_auto()
        self._refresh_as5100_summary_label()

    def _sync_as5100_model_defaults(self, *_: object) -> None:
        if not hasattr(self, "as5100_model_combo"):
            return
        self._refresh_as5100_summary_label()

    def _refresh_as5100_summary_label(self, *_: object) -> None:
        if hasattr(self, "as5100_summary_label"):
            self.as5100_summary_label.setText(self._format_as5100_load_source_summary())

    @staticmethod
    def _format_as5100_group_count_range(group_counts: Sequence[int]) -> str:
        if not group_counts:
            return "[]"
        first = int(group_counts[0])
        last = int(group_counts[-1])
        if len(group_counts) <= 4:
            return "[" + ", ".join(str(int(value)) for value in group_counts) + "]"
        return f"[{first}..{last}]"

    def _sync_envelope_bearing_defaults(self, *_: object) -> None:
        if not hasattr(self, "envelope_use_sleeper_geometry_checkbox"):
            return
        use_sleeper = self.envelope_use_sleeper_geometry_checkbox.isChecked()
        self.envelope_bearing_width_input.setEnabled(not use_sleeper)
        self.envelope_bearing_length_input.setEnabled(not use_sleeper)
        if use_sleeper:
            sleeper = self.sleeper_combo.currentData()
            if sleeper is not None:
                self.envelope_bearing_width_input.set_value(sleeper.width_m)
                self.envelope_bearing_length_input.set_value(sleeper.length_m)
        self._update_envelope_bearing_area()

    def _update_envelope_bearing_area(self) -> None:
        if not hasattr(self, "envelope_bearing_area_label"):
            return
        width = self.envelope_bearing_width_input.value()
        length = self.envelope_bearing_length_input.value()
        if width <= 0 or length <= 0:
            self.envelope_bearing_area_label.setText("—")
            return
        area = width * length
        self.envelope_bearing_area_label.setText(f"{area:.4f} m²")

    def _update_overlay_state(self, *_: object) -> None:
        if not hasattr(self, "overlay_checkbox"):
            return
        analysis_type = self.analysis_type_combo.currentData()
        try:
            dynamic_mode = self._coerce_dynamic_mode(self.dynamic_mode_combo.currentData())
        except ValueError:
            dynamic_mode = None
        static_enabled = analysis_type == AnalysisType.STATIC and not (
            hasattr(self, "transition_group") and self.transition_group.isChecked()
        ) and self.static_mode_combo.currentData() == StaticMode.SINGLE
        dynamic_enabled = (
            analysis_type == AnalysisType.DYNAMIC
            and dynamic_mode in (DynamicMode.STEADY_STATE, DynamicMode.TIME_HISTORY)
        )
        enabled = static_enabled or dynamic_enabled
        self.overlay_checkbox.setEnabled(enabled)
        if not enabled and self.overlay_checkbox.isChecked():
            self.overlay_checkbox.blockSignals(True)
            self.overlay_checkbox.setChecked(False)
            self.overlay_checkbox.blockSignals(False)
        if not self.overlay_checkbox.isChecked() and (
            self._overlay_results or self._dynamic_overlay_results
        ):
            self._clear_all_overlays(render=True)
        self.clear_overlay_button.setEnabled(bool(self._overlay_results or self._dynamic_overlay_results))

    def _clear_overlays(self) -> None:
        self._clear_all_overlays(render=True)
        self._update_overlay_state()

    def _clear_static_overlays(self, *, render: bool = True) -> None:
        self._overlay_results = []
        self._primary_label = None
        if render and self._last_analysis_result is not None:
            self._render_analysis_result(self._last_analysis_result)

    def _clear_dynamic_overlays(self, *, render: bool = True) -> None:
        self._dynamic_overlay_results = []
        self._dynamic_primary_label = None
        self._dynamic_overlay_mode = None
        if render and self._last_dynamic_result is not None:
            self._render_dynamic_result(self._last_dynamic_result)

    def _clear_all_overlays(self, *, render: bool = True) -> None:
        self._clear_static_overlays(render=render)
        self._clear_dynamic_overlays(render=render)

    def _toggle_foundation_profile_controls(self, enabled: bool) -> None:
        self.profile_type_combo.setEnabled(enabled)
        self.profile_k1_input.setEnabled(enabled)
        self.profile_k2_input.setEnabled(enabled)
        self.profile_x_start_input.setEnabled(enabled)
        self.profile_x_end_input.setEnabled(enabled)
        if enabled:
            if self.profile_type_combo.currentData() == FoundationProfileType.UNIFORM:
                step_index = self.profile_type_combo.findData(FoundationProfileType.STEP)
                if step_index >= 0:
                    self.profile_type_combo.setCurrentIndex(step_index)
            self._update_profile_input_visibility()

    def _update_profile_input_visibility(self) -> None:
        profile_type = self.profile_type_combo.currentData()
        is_ramp = profile_type == FoundationProfileType.RAMP
        self.profile_x_end_input.setVisible(is_ramp)
        layout = self.advanced_container.layout()
        if isinstance(layout, QFormLayout):
            label = layout.labelForField(self.profile_x_end_input)
            if label is not None:
                label.setVisible(is_ramp)

    def _toggle_discrete_controls(self, enabled: bool) -> None:
        self.pad_stiffness_input.setEnabled(enabled)
        self.pad_damping_input.setEnabled(enabled)
        self.nodes_between_sleepers_input.setEnabled(enabled)
        self._toggle_foundation_damping_inputs()
        self._update_envelope_estimate()
        self._update_transition_estimate()

    def _toggle_two_rail_controls(self, enabled: bool) -> None:
        self.coupling_stiffness_input.setEnabled(enabled)
        self.asymmetric_load_checkbox.setEnabled(enabled)
        if not enabled:
            self.asymmetric_load_checkbox.setChecked(False)
        self._toggle_right_load_controls(enabled and self.asymmetric_load_checkbox.isChecked())
        self._update_envelope_rail_count_state()
        self._update_envelope_estimate()
        self._update_transition_estimate()

    def _toggle_right_load_controls(self, enabled: bool) -> None:
        self.right_load_magnitude_input.setEnabled(enabled)
        self.right_load_position_input.setEnabled(enabled)
        if not enabled:
            self.right_load_magnitude_input.set_value(self.load_magnitude_input.value())
            self.right_load_position_input.set_value(self.load_position_input.value())

    def _toggle_pasternak_controls(self, enabled: bool) -> None:
        self.pasternak_input.setEnabled(enabled)
        self._update_pasternak_warning()

    def _update_pasternak_warning(self) -> None:
        if not hasattr(self, "pasternak_warning_label"):
            return
        is_timoshenko = self.beam_theory_combo.currentData() == BeamTheory.TIMOSHENKO
        show_warning = is_timoshenko and self.pasternak_checkbox.isChecked()
        self.pasternak_warning_label.setVisible(show_warning)

    def _toggle_analysis_type(self, *_: object) -> None:
        analysis_type = self.analysis_type_combo.currentData()
        is_dynamic = analysis_type == AnalysisType.DYNAMIC
        is_special = analysis_type == AnalysisType.SPECIAL
        is_static = analysis_type == AnalysisType.STATIC
        self._clear_all_overlays(render=False)
        self.dynamic_mode_combo.setVisible(is_dynamic)
        if hasattr(self, "dynamic_annotation_mode_combo"):
            self.dynamic_annotation_mode_combo.setVisible(is_dynamic)
        self.dynamic_container.setVisible(is_dynamic)
        if hasattr(self, "special_mode_combo"):
            self.special_mode_combo.setVisible(is_special)
        if hasattr(self, "special_container"):
            self.special_container.setVisible(is_special)
        self.advanced_solver_checkbox.setVisible(is_static)
        self.static_mode_combo.setVisible(is_static)
        self.design_group.setVisible(is_static)
        if hasattr(self, "transition_group"):
            self.transition_group.setVisible(is_static)
        if hasattr(self, "overlay_checkbox"):
            show_overlay = not is_special
            self.overlay_checkbox.setVisible(show_overlay)
            self.clear_overlay_button.setVisible(show_overlay)
            if not show_overlay and self.overlay_checkbox.isChecked():
                self.overlay_checkbox.blockSignals(True)
                self.overlay_checkbox.setChecked(False)
                self.overlay_checkbox.blockSignals(False)
            self._update_overlay_state()
        if (is_dynamic or is_special) and hasattr(self, "envelope_group"):
            self.envelope_group.setVisible(False)
        if hasattr(self, "several_loads_checkbox"):
            self.several_loads_checkbox.setVisible(True)
            self.train_loads_checkbox.setVisible(True)
            self.as5100_loads_checkbox.setVisible(True)
            self._sync_load_input_state()
        if isinstance(self.analysis_layout, QFormLayout) and hasattr(self, "solver_backend_label"):
            self._set_form_row_visible(self.analysis_layout, self.solver_backend_label, is_static)
            self._set_form_row_visible(self.analysis_layout, self.static_mode_combo, is_static)
            if hasattr(self, "dynamic_mode_combo"):
                self._set_form_row_visible(self.analysis_layout, self.dynamic_mode_combo, is_dynamic)
            if hasattr(self, "dynamic_annotation_mode_combo"):
                self._set_form_row_visible(self.analysis_layout, self.dynamic_annotation_mode_combo, is_dynamic)
            if is_static:
                self._update_solver_backend_label()
        self.export_analysis_button.setVisible(is_static)
        self.export_config_button.setVisible(is_static)
        self.export_sleeper_button.setVisible(is_static)
        self.export_dynamic_time_button.setVisible(is_dynamic)
        self.export_dynamic_fft_button.setVisible(is_dynamic)
        self.export_dynamic_psd_button.setVisible(is_dynamic)
        self.export_dynamic_transition_metrics_button.setVisible(is_dynamic)
        self.export_dynamic_transition_series_button.setVisible(is_dynamic)
        self.export_dynamic_transition_config_button.setVisible(is_dynamic)
        self.export_dipped_joint_button.setVisible(is_dynamic)
        if hasattr(self, "tab_widget"):
            self._set_dynamic_tabs_visible(is_dynamic)
            if hasattr(self, "_set_special_tabs_visible"):
                self._set_special_tabs_visible(is_special)
            if is_dynamic:
                self._toggle_dynamic_mode()
            elif is_special:
                self._toggle_special_mode()
            else:
                self._switch_to_appropriate_tab()
        if is_dynamic:
            self.advanced_solver_checkbox.setChecked(False)
            self._toggle_advanced_controls(False)
        elif is_static:
            self._toggle_advanced_controls(self.advanced_solver_checkbox.isChecked())
            self._toggle_static_mode()
        else:
            self._toggle_advanced_controls(False)
        self._update_validation_hints()
        if hasattr(self, "transition_group"):
            self._toggle_transition_mode(self.transition_group.isChecked())
        self._set_export_buttons_enabled(
            is_static
            and (
                self._last_analysis_result is not None
                or self._last_envelope_result is not None
                or self._last_transition_result is not None
            )
        )
        self._set_dynamic_export_buttons_enabled(
            is_dynamic
            and (
                self._last_dynamic_result is not None
                or self._last_dynamic_transition_result is not None
                or self._last_dipped_joint_result is not None
            )
        )

    def _toggle_static_mode(self, *_: object) -> None:
        if not hasattr(self, "static_mode_combo"):
            return
        had_overlays = bool(self._overlay_results or self._dynamic_overlay_results)
        self._clear_all_overlays(render=had_overlays)
        if hasattr(self, "transition_group") and self.transition_group.isChecked():
            self._update_solver_backend_label()
            return
        mode = self.static_mode_combo.currentData()
        is_envelope = mode in (StaticMode.ENVELOPE_CLOSED_FORM, StaticMode.ENVELOPE_NUMERICAL)
        if hasattr(self, "envelope_group"):
            self.envelope_group.setVisible(is_envelope)
        if is_envelope:
            self.advanced_solver_checkbox.blockSignals(True)
            self.advanced_solver_checkbox.setChecked(False)
            self.advanced_solver_checkbox.blockSignals(False)
            self.advanced_solver_checkbox.setEnabled(False)
            if mode == StaticMode.ENVELOPE_CLOSED_FORM:
                self._toggle_advanced_controls(False)
                self.two_rail_checkbox.setChecked(False)
                self.two_rail_checkbox.setEnabled(False)
                self._toggle_two_rail_controls(False)
            else:
                self.two_rail_checkbox.setEnabled(True)
                self._toggle_advanced_controls(True)
            self._sync_envelope_bearing_defaults()
        else:
            self.advanced_solver_checkbox.setEnabled(True)
            self._toggle_advanced_controls(self.advanced_solver_checkbox.isChecked())
            self.two_rail_checkbox.setEnabled(True)
            self._toggle_two_rail_controls(self.two_rail_checkbox.isChecked())

        if hasattr(self, "overlay_checkbox"):
            self.overlay_checkbox.setEnabled(not is_envelope)
            if is_envelope and self.overlay_checkbox.isChecked():
                self.overlay_checkbox.blockSignals(True)
                self.overlay_checkbox.setChecked(False)
                self.overlay_checkbox.blockSignals(False)
            self._update_overlay_state()

        self._toggle_envelope_train_reference(is_envelope)
        self._update_envelope_rail_count_state()
        if is_envelope:
            self._update_envelope_range_auto()
            self._update_envelope_domain_auto()
            self._update_envelope_estimate()
        else:
            self._update_envelope_estimate()
        self._update_validation_hints()
        self._update_solver_backend_label()

    def _toggle_envelope_train_reference(self, is_envelope: bool) -> None:
        if not hasattr(self, "train_reference_input"):
            return
        if is_envelope:
            self.train_reference_input.set_value(0.0)
            self.train_reference_input.setEnabled(False)
            self.train_reference_input.setToolTip(
                "Envelope mode treats load positions as offsets from x_ref."
            )
        else:
            self.train_reference_input.setEnabled(True)
            self.train_reference_input.setToolTip("Reference position for the first bogie center (x₀).")

    def _update_envelope_rail_count_state(self) -> None:
        if not hasattr(self, "envelope_rail_count_combo"):
            return
        mode = self.static_mode_combo.currentData()
        if mode == StaticMode.ENVELOPE_NUMERICAL and self.two_rail_checkbox.isChecked():
            self.envelope_rail_count_combo.setCurrentIndex(1)
            self.envelope_rail_count_combo.setEnabled(False)
        else:
            self.envelope_rail_count_combo.setEnabled(True)

    def _update_transition_tabs_visible(self) -> None:
        if not hasattr(self, "tab_widget"):
            return
        if not hasattr(self, "transition_group"):
            return
        show = (
            self.analysis_type_combo.currentData() == AnalysisType.STATIC
            and (self.transition_group.isChecked() or self._last_transition_result is not None)
        )
        self.tab_widget.setTabVisible(self.transition_profile_tab_index, show)
        self.tab_widget.setTabVisible(self.transition_summary_tab_index, show)
        self._refresh_chart_grid_if_visible()

    def _transition_requires_numerical(self) -> bool:
        if not hasattr(self, "transition_group") or not self.transition_group.isChecked():
            return False
        try:
            profile_type = self._coerce_transition_profile_type(self.transition_profile_combo.currentData())
        except ValueError:
            return False
        return profile_type != TransitionProfileType.UNIFORM

    def _toggle_transition_mode(self, enabled: bool) -> None:
        if not hasattr(self, "transition_group"):
            return
        self._clear_all_overlays(render=False)
        analysis_type = self.analysis_type_combo.currentData()
        is_dynamic = analysis_type == AnalysisType.DYNAMIC
        is_special = analysis_type == AnalysisType.SPECIAL
        self.transition_group.setVisible(not is_dynamic and not is_special)
        if is_dynamic or is_special:
            self._update_transition_tabs_visible()
            return
        self._update_transition_tabs_visible()
        if not enabled:
            if isinstance(self.analysis_layout, QFormLayout):
                self._set_form_row_visible(self.analysis_layout, self.static_mode_combo, True)
            self.static_mode_combo.setEnabled(True)
            self._toggle_static_mode()
            self.advanced_solver_checkbox.setEnabled(True)
            self.foundation_model_combo.setEnabled(True)
            if hasattr(self, "nonuniform_profile_checkbox"):
                self.nonuniform_profile_checkbox.setEnabled(True)
            if hasattr(self, "overlay_checkbox"):
                self.overlay_checkbox.setEnabled(True)
            self._update_solver_backend_label()
            self._update_validation_hints()
            return

        if isinstance(self.analysis_layout, QFormLayout):
            self._set_form_row_visible(self.analysis_layout, self.static_mode_combo, False)
        if hasattr(self, "envelope_group"):
            self.envelope_group.setVisible(False)
        self.static_mode_combo.setEnabled(False)
        self.static_mode_combo.setCurrentIndex(0)

        if hasattr(self, "overlay_checkbox"):
            self.overlay_checkbox.blockSignals(True)
            self.overlay_checkbox.setChecked(False)
            self.overlay_checkbox.blockSignals(False)
            self.overlay_checkbox.setEnabled(False)
            self._update_overlay_state()

        self._toggle_transition_run_mode()
        self._update_transition_profile_visibility()
        self._update_transition_length_auto()
        self._update_transition_range_auto()
        self._update_transition_domain_auto()
        self._update_transition_estimate()
        self._refresh_chart_grid_if_visible()
        self._update_validation_hints()

        force_numerical = self._transition_requires_numerical()
        if force_numerical:
            self.advanced_solver_checkbox.blockSignals(True)
            self.advanced_solver_checkbox.setChecked(True)
            self.advanced_solver_checkbox.blockSignals(False)
            self.advanced_solver_checkbox.setEnabled(False)
        else:
            self.advanced_solver_checkbox.setEnabled(True)

        self._toggle_advanced_controls(self.advanced_solver_checkbox.isChecked())

        try:
            profile_type = self._coerce_transition_profile_type(self.transition_profile_combo.currentData())
        except ValueError:
            profile_type = TransitionProfileType.UNIFORM
        if profile_type != TransitionProfileType.UNIFORM:
            self.foundation_model_combo.setCurrentIndex(0)
            self.foundation_model_combo.setEnabled(False)
        else:
            self.foundation_model_combo.setEnabled(True)
        if hasattr(self, "nonuniform_profile_checkbox"):
            self.nonuniform_profile_checkbox.setChecked(False)
            self.nonuniform_profile_checkbox.setEnabled(False)

        self._update_solver_backend_label()

    @staticmethod
    def _coerce_transition_profile_type(raw: object) -> TransitionProfileType:
        if isinstance(raw, TransitionProfileType):
            return raw
        if isinstance(raw, str):
            try:
                return TransitionProfileType(raw)
            except ValueError:
                pass
        raise ValueError("Select a valid transition profile type.")

    @staticmethod
    def _coerce_transition_run_mode(raw: object) -> TransitionRunMode:
        if isinstance(raw, TransitionRunMode):
            return raw
        if isinstance(raw, str):
            try:
                return TransitionRunMode(raw)
            except ValueError:
                pass
        raise ValueError("Select a valid transition run mode.")

    def _toggle_transition_run_mode(self, *_: object) -> None:
        if not hasattr(self, "transition_run_mode_combo"):
            return
        try:
            run_mode = self._coerce_transition_run_mode(self.transition_run_mode_combo.currentData())
        except ValueError:
            run_mode = TransitionRunMode.SINGLE
        is_envelope = run_mode == TransitionRunMode.ENVELOPE
        layout = self.transition_group.layout()
        if isinstance(layout, QFormLayout):
            self._set_form_row_visible(layout, self.transition_reference_input, is_envelope)
            self._set_form_row_visible(layout, self.transition_range_container, is_envelope)
            self._set_form_row_visible(layout, self.transition_step_input, is_envelope)
            self.transition_note_label.setVisible(is_envelope)
        if is_envelope:
            self._update_transition_range_auto()
        self._update_transition_domain_auto()
        self._update_transition_estimate()

    def _toggle_dynamic_transition_run_mode(self, *_: object) -> None:
        if not hasattr(self, "dynamic_transition_group"):
            return
        run_mode = self._coerce_dynamic_transition_run_mode(
            self.dynamic_transition_run_mode_combo.currentData()
        )
        is_envelope = run_mode == DynamicTransitionRunMode.ENVELOPE
        layout = self.dynamic_transition_group.layout()
        if isinstance(layout, QFormLayout):
            self._set_form_row_visible(layout, self.dynamic_transition_x_ref_input, is_envelope)
            self._set_form_row_visible(layout, self.dynamic_transition_range_container, is_envelope)
            self._set_form_row_visible(layout, self.dynamic_transition_step_input, is_envelope)
            self.dynamic_transition_note_label.setVisible(is_envelope)

    def _update_dynamic_transition_profile_visibility(self, *_: object) -> None:
        if not hasattr(self, "dynamic_transition_group"):
            return
        profile = self._coerce_dynamic_transition_profile_type(
            self.dynamic_transition_profile_combo.currentData()
        )
        layout = self.dynamic_transition_group.layout()
        if isinstance(layout, QFormLayout):
            self._set_form_row_visible(
                layout,
                self.dynamic_transition_k2_input,
                profile != DynamicTransitionProfileType.UNIFORM,
            )
            self._set_form_row_visible(
                layout,
                self.dynamic_transition_length_input,
                profile in (DynamicTransitionProfileType.RAMP, DynamicTransitionProfileType.EXPONENTIAL),
            )
            self._set_form_row_visible(
                layout,
                self.dynamic_transition_segment_length_input,
                profile == DynamicTransitionProfileType.SEGMENT,
            )
        self._update_solver_backend_label()

    @staticmethod
    def _coerce_dynamic_transition_profile_type(raw: object) -> DynamicTransitionProfileType:
        if isinstance(raw, DynamicTransitionProfileType):
            return raw
        if isinstance(raw, str):
            try:
                return DynamicTransitionProfileType(raw)
            except ValueError:
                pass
        raise ValueError("Select a valid dynamic transition profile type.")

    @staticmethod
    def _coerce_dynamic_transition_run_mode(raw: object) -> DynamicTransitionRunMode:
        if isinstance(raw, DynamicTransitionRunMode):
            return raw
        if isinstance(raw, str):
            try:
                return DynamicTransitionRunMode(raw)
            except ValueError:
                pass
        raise ValueError("Select a valid dynamic transition run mode.")

    @staticmethod
    def _coerce_dynamic_mode(raw: object) -> DynamicMode:
        if isinstance(raw, DynamicMode):
            return raw
        if isinstance(raw, str):
            try:
                return DynamicMode(raw)
            except ValueError:
                pass
        raise ValueError("Select a valid dynamic mode.")

    def _toggle_transition_range_auto(self, enabled: bool) -> None:
        self.transition_range_start_input.setEnabled(not enabled)
        self.transition_range_end_input.setEnabled(not enabled)
        if enabled:
            self._update_transition_range_auto()
            self._update_transition_domain_auto()
        self._update_transition_estimate()

    def _update_transition_range_auto(self) -> None:
        if not self.transition_range_auto_checkbox.isChecked():
            return
        try:
            run_mode = self._coerce_transition_run_mode(self.transition_run_mode_combo.currentData())
        except ValueError:
            return
        if run_mode != TransitionRunMode.ENVELOPE:
            return
        rail = self.rail_combo.currentData()
        if rail is None:
            return
        k1_n_per_m2 = mn_per_m2_to_n_per_m2(self.transition_k1_input.value())
        if k1_n_per_m2 <= 0:
            return
        k_extent_n_per_m2 = k1_n_per_m2
        try:
            profile_type = self._coerce_transition_profile_type(self.transition_profile_combo.currentData())
        except ValueError:
            return
        if profile_type != TransitionProfileType.UNIFORM:
            k2_n_per_m2 = mn_per_m2_to_n_per_m2(self.transition_k2_input.value())
            if k2_n_per_m2 > 0:
                k_extent_n_per_m2 = min(k_extent_n_per_m2, k2_n_per_m2)
        beta = beam_parameter_beta(
            k_extent_n_per_m2,
            rail.elastic_modulus_pa,
            rail.moment_inertia_m4,
        )
        length = 10.0 / beta
        transition_length = self.transition_length_input.value()
        extent = max(length, transition_length) if transition_length > 0 else length
        start = -extent
        end = extent
        if profile_type in (TransitionProfileType.RAMP, TransitionProfileType.EXPONENTIAL):
            end = max(end, transition_length + extent)
        self.transition_range_start_input.set_value(start)
        self.transition_range_end_input.set_value(end)
        self._update_transition_estimate()

    def _toggle_transition_domain_auto(self, enabled: bool) -> None:
        self.transition_domain_start_input.setEnabled(not enabled)
        self.transition_domain_end_input.setEnabled(not enabled)
        if enabled:
            self._update_transition_domain_auto()
        self._update_transition_estimate()

    def _transition_zone_bounds(self) -> tuple[float, float]:
        try:
            profile_type = self._coerce_transition_profile_type(self.transition_profile_combo.currentData())
        except ValueError:
            return 0.0, 0.0
        if profile_type in (TransitionProfileType.RAMP, TransitionProfileType.EXPONENTIAL):
            length = self.transition_length_input.value()
            if length > 0:
                return 0.0, length
        if profile_type == TransitionProfileType.SEGMENT:
            length = self.transition_segment_length_input.value()
            if length > 0:
                half = 0.5 * length
                return -half, half
        return 0.0, 0.0

    def _update_transition_domain_auto(self) -> None:
        if not self.transition_domain_auto_checkbox.isChecked():
            return
        try:
            loads = self._collect_analysis_loads()
        except ValueError:
            return
        if not loads:
            return
        rail = self.rail_combo.currentData()
        if rail is None:
            return
        k1_n_per_m2 = mn_per_m2_to_n_per_m2(self.transition_k1_input.value())
        if k1_n_per_m2 <= 0:
            return
        k_extent_n_per_m2 = k1_n_per_m2
        try:
            profile_type = self._coerce_transition_profile_type(self.transition_profile_combo.currentData())
        except ValueError:
            return
        if profile_type != TransitionProfileType.UNIFORM:
            k2_n_per_m2 = mn_per_m2_to_n_per_m2(self.transition_k2_input.value())
            if k2_n_per_m2 > 0:
                k_extent_n_per_m2 = min(k_extent_n_per_m2, k2_n_per_m2)
        beta = beam_parameter_beta(
            k_extent_n_per_m2,
            rail.elastic_modulus_pa,
            rail.moment_inertia_m4,
        )
        margin = 5.0 / beta
        offsets = [load.position_m for load in loads]
        x_ref_start = 0.0
        x_ref_end = 0.0
        try:
            run_mode = self._coerce_transition_run_mode(self.transition_run_mode_combo.currentData())
        except ValueError:
            run_mode = TransitionRunMode.SINGLE
        if run_mode == TransitionRunMode.ENVELOPE:
            if self.transition_range_auto_checkbox.isChecked():
                self._update_transition_range_auto()
            reference = self.transition_reference_input.value()
            x_ref_start = reference + self.transition_range_start_input.value()
            x_ref_end = reference + self.transition_range_end_input.value()
        x_min = x_ref_start + min(offsets) - margin
        x_max = x_ref_end + max(offsets) + margin
        zone_min, zone_max = self._transition_zone_bounds()
        x_min = min(x_min, zone_min - margin)
        x_max = max(x_max, zone_max + margin)
        if x_max <= x_min:
            return
        self.transition_domain_start_input.set_value(x_min)
        self.transition_domain_end_input.set_value(x_max)
        self._update_transition_estimate()

    def _update_validation_hints(self) -> None:
        if not hasattr(self, "validation_hint_label"):
            return
        if self.analysis_type_combo.currentData() in (AnalysisType.DYNAMIC, AnalysisType.SPECIAL):
            self.validation_hint_label.setText("—")
            self.validation_hint_label.setStyleSheet("color: #666666;")
            return
        hints: list[str] = []

        spacing_m = mm_to_m(self.sleeper_spacing_input.value())
        if spacing_m < 0.3 or spacing_m > 1.0:
            hints.append(f"Sleeper spacing {spacing_m:.3f} m is outside 0.3–1.0 m (heuristic).")

        k_values_mn: list[float] = []
        if hasattr(self, "transition_group") and self.transition_group.isChecked():
            k1 = self.transition_k1_input.value()
            k_values_mn.append(k1)
            try:
                profile_type = self._coerce_transition_profile_type(self.transition_profile_combo.currentData())
            except ValueError:
                profile_type = TransitionProfileType.UNIFORM
            if profile_type != TransitionProfileType.UNIFORM:
                k_values_mn.append(self.transition_k2_input.value())
        else:
            support = self.support_combo.currentData()
            if support is not None:
                k_values_mn.append(n_per_m2_to_mn_per_m2(support.foundation_modulus_n_per_m2))

        for k in k_values_mn:
            if k < 5 or k > 400:
                hints.append(f"Foundation modulus k={k:.1f} MN/m² outside 5–400 MN/m² (heuristic).")
                break

        rail = self.rail_combo.currentData()
        domain_length = None
        k_for_beta = None
        if hasattr(self, "transition_group") and self.transition_group.isChecked():
            x_min = self.transition_domain_start_input.value()
            x_max = self.transition_domain_end_input.value()
            domain_length = x_max - x_min
            k_for_beta = mn_per_m2_to_n_per_m2(self.transition_k1_input.value())
            try:
                profile_type = self._coerce_transition_profile_type(self.transition_profile_combo.currentData())
            except ValueError:
                profile_type = TransitionProfileType.UNIFORM
            if profile_type != TransitionProfileType.UNIFORM:
                k2 = mn_per_m2_to_n_per_m2(self.transition_k2_input.value())
                if k2 > 0:
                    k_for_beta = min(k_for_beta, k2)
        elif hasattr(self, "envelope_group") and self.envelope_group.isVisible():
            x_min = self.envelope_domain_start_input.value()
            x_max = self.envelope_domain_end_input.value()
            domain_length = x_max - x_min
            support = self.support_combo.currentData()
            if support is not None:
                k_for_beta = support.foundation_modulus_n_per_m2

        if rail is not None and k_for_beta is not None and domain_length and domain_length > 0:
            beta = beam_parameter_beta(
                k_for_beta,
                rail.elastic_modulus_pa,
                rail.moment_inertia_m4,
            )
            recommended = 8.0 / beta
            if domain_length < recommended:
                hints.append(
                    f"Domain length {domain_length:.2f} m < 8/β ({recommended:.2f} m)."
                )

        if hints:
            text = "Heuristic checks:\n" + "\n".join(f"• {hint}" for hint in hints)
            self.validation_hint_label.setText(text)
            self.validation_hint_label.setStyleSheet("color: #8a5a00;")
        else:
            self.validation_hint_label.setText("Heuristic checks: OK")
            self.validation_hint_label.setStyleSheet("color: #4f6f52;")

    def _estimate_grid_points(
        self,
        *,
        rail: Rail | None,
        k_n_per_m2: float | None,
        domain: tuple[float, float],
        align_to_sleepers: bool,
    ) -> int:
        if rail is None or k_n_per_m2 is None or k_n_per_m2 <= 0:
            return 0
        start, end = domain
        if end <= start:
            return 0
        beta = beam_parameter_beta(k_n_per_m2, rail.elastic_modulus_pa, rail.moment_inertia_m4)
        x_values, _, _, _ = _build_grid(
            beta,
            401,
            mm_to_m(self.sleeper_spacing_input.value()),
            align_to_sleepers,
            self.nodes_between_sleepers_input.value(),
            domain,
        )
        return len(x_values)

    @staticmethod
    def _format_runtime_estimate(steps: int, grid_points: int, is_numerical: bool) -> str:
        if steps <= 0 or grid_points <= 0:
            return "—"
        work_units = steps * grid_points
        rate = 250_000 if not is_numerical else 120_000
        seconds = work_units / rate
        if seconds < 1.0:
            time_text = f"~{seconds * 1000.0:.0f} ms"
        elif seconds < 90.0:
            time_text = f"~{seconds:.1f} s"
        else:
            time_text = f"~{seconds / 60.0:.1f} min"
        return f"{time_text} ({steps} steps × {grid_points} pts)"

    @staticmethod
    def _extend_movement_range_for_plot_domain(
        *,
        x_ref_start_m: float,
        x_ref_end_m: float,
        x_domain_m: tuple[float, float],
        load_offsets_m: Sequence[float],
        beta_per_m: float,
    ) -> tuple[float, float]:
        if beta_per_m <= 0.0:
            raise ValueError("beta_per_m must be positive")
        if not load_offsets_m:
            raise ValueError("load_offsets_m must not be empty")
        buffer_m = ENVELOPE_MOVEMENT_BUFFER_FACTOR / beta_per_m
        domain_start, domain_end = x_domain_m
        extended_start = min(x_ref_start_m, domain_start - max(load_offsets_m) - buffer_m)
        extended_end = max(x_ref_end_m, domain_end - min(load_offsets_m) + buffer_m)
        if extended_end <= extended_start:
            raise ValueError("Extended movement range is invalid.")
        return extended_start, extended_end

    @staticmethod
    def _solver_domain_for_movement_range(
        *,
        x_ref_start_m: float,
        x_ref_end_m: float,
        plot_domain_m: tuple[float, float],
        load_offsets_m: Sequence[float],
        beta_per_m: float,
    ) -> tuple[float, float]:
        if beta_per_m <= 0.0:
            raise ValueError("beta_per_m must be positive")
        if not load_offsets_m:
            raise ValueError("load_offsets_m must not be empty")
        buffer_m = ENVELOPE_MOVEMENT_BUFFER_FACTOR / beta_per_m
        plot_start, plot_end = plot_domain_m
        solver_start = min(plot_start, x_ref_start_m + min(load_offsets_m) - buffer_m)
        solver_end = max(plot_end, x_ref_end_m + max(load_offsets_m) + buffer_m)
        if solver_end <= solver_start:
            raise ValueError("Solver domain is invalid for the selected movement range.")
        return solver_start, solver_end

    @staticmethod
    def _transition_k_chart_series(
        *,
        x_values: Sequence[float],
        profile_type: object,
        k1_n_per_m2: float,
        k2_n_per_m2: float | None,
        transition_length_m: float | None,
        segment_length_m: float | None,
    ) -> tuple[list[float], list[float]]:
        if not x_values:
            raise ValueError("x_values must not be empty")
        if k1_n_per_m2 <= 0.0:
            raise ValueError("k1_n_per_m2 must be positive")
        profile_value = getattr(profile_type, "value", str(profile_type))
        x_min = min(x_values)
        x_max = max(x_values)
        span = max(x_max - x_min, 1.0)
        pad = max(0.05 * span, 1.0)
        anchors: list[float] = [x_min, x_max]

        if profile_value == "uniform":
            chart_x = sorted(set(float(x) for x in x_values))
            return chart_x, [k1_n_per_m2 for _ in chart_x]

        if k2_n_per_m2 is None or k2_n_per_m2 <= 0.0:
            raise ValueError("k2_n_per_m2 must be positive for non-uniform profiles")

        if profile_value == "step":
            x_min = min(x_min, -pad)
            x_max = max(x_max, pad)
            base = sorted({float(x) for x in x_values if x_min <= x <= x_max} | {x_min, x_max})
            chart_x = [x for x in base if x < 0.0] + [0.0, 0.0] + [x for x in base if x > 0.0]
            chart_y = []
            zero_seen = False
            for x in chart_x:
                if x == 0.0:
                    chart_y.append(k2_n_per_m2 if zero_seen else k1_n_per_m2)
                    zero_seen = True
                else:
                    chart_y.append(k2_n_per_m2 if x > 0.0 else k1_n_per_m2)
            return chart_x, chart_y

        if profile_value in {"ramp", "exponential"}:
            if transition_length_m is None or transition_length_m <= 0.0:
                raise ValueError("transition_length_m must be positive for ramp profiles")
            extent_end = transition_length_m if profile_value == "ramp" else 5.0 * transition_length_m
            x_min = min(x_min, 0.0)
            x_max = max(x_max, extent_end)
            anchors.extend([0.0, transition_length_m, extent_end, x_min, x_max])
            chart_x = sorted({float(x) for x in x_values if x_min <= x <= x_max} | set(anchors))
            if profile_value == "ramp":
                chart_y = []
                for x in chart_x:
                    if x < 0.0:
                        chart_y.append(k1_n_per_m2)
                    elif x <= transition_length_m:
                        ratio = x / transition_length_m
                        chart_y.append(k1_n_per_m2 + (k2_n_per_m2 - k1_n_per_m2) * ratio)
                    else:
                        chart_y.append(k2_n_per_m2)
                return chart_x, chart_y
            return chart_x, [
                (
                    k1_n_per_m2
                    if x < 0.0
                    else k1_n_per_m2
                    + (k2_n_per_m2 - k1_n_per_m2) * (1.0 - math.exp(-x / transition_length_m))
                )
                for x in chart_x
            ]

        if profile_value == "segment":
            if segment_length_m is None or segment_length_m <= 0.0:
                raise ValueError("segment_length_m must be positive for segment profiles")
            half = 0.5 * segment_length_m
            x_min = min(x_min, -half - pad)
            x_max = max(x_max, half + pad)
            base = sorted({float(x) for x in x_values if x_min <= x <= x_max} | {x_min, x_max})
            chart_x = (
                [x for x in base if x < -half]
                + [-half, -half]
                + [x for x in base if -half < x < half]
                + [half, half]
                + [x for x in base if x > half]
            )
            chart_y = []
            neg_half_seen = False
            pos_half_seen = False
            for index, x in enumerate(chart_x):
                if x == -half:
                    chart_y.append(k2_n_per_m2 if neg_half_seen else k1_n_per_m2)
                    neg_half_seen = True
                elif x == half:
                    chart_y.append(k1_n_per_m2 if pos_half_seen else k2_n_per_m2)
                    pos_half_seen = True
                elif -half < x < half:
                    chart_y.append(k2_n_per_m2)
                else:
                    chart_y.append(k1_n_per_m2)
            return chart_x, chart_y

        raise ValueError(f"Unsupported transition profile type: {profile_value}")

    def _update_envelope_estimate(self) -> None:
        if not hasattr(self, "envelope_estimate_label"):
            return
        mode = self.static_mode_combo.currentData()
        if mode not in (StaticMode.ENVELOPE_CLOSED_FORM, StaticMode.ENVELOPE_NUMERICAL):
            self.envelope_estimate_label.setText("—")
            return
        step = self.envelope_step_input.value()
        start = self.envelope_range_start_input.value()
        end = self.envelope_range_end_input.value()
        if step <= 0 or end <= start:
            self.envelope_estimate_label.setText("—")
            return
        steps = int(math.floor((end - start) / step)) + 1
        domain = (self.envelope_domain_start_input.value(), self.envelope_domain_end_input.value())
        support = self.support_combo.currentData()
        k_value = support.foundation_modulus_n_per_m2 if support is not None else None
        align_to_sleepers = False
        grid_points = 401
        if mode == StaticMode.ENVELOPE_NUMERICAL:
            align_to_sleepers = (
                self.discrete_supports_checkbox.isChecked() or self.two_rail_checkbox.isChecked()
            )
            grid_points = self._estimate_grid_points(
                rail=self.rail_combo.currentData(),
                k_n_per_m2=k_value,
                domain=domain,
                align_to_sleepers=align_to_sleepers,
            )
        estimate = self._format_runtime_estimate(steps, grid_points, mode == StaticMode.ENVELOPE_NUMERICAL)
        self.envelope_estimate_label.setText(estimate)

    def _update_transition_estimate(self) -> None:
        if not hasattr(self, "transition_estimate_label"):
            return
        if not hasattr(self, "transition_group") or not self.transition_group.isChecked():
            self.transition_estimate_label.setText("—")
            return
        try:
            run_mode = self._coerce_transition_run_mode(self.transition_run_mode_combo.currentData())
        except ValueError:
            self.transition_estimate_label.setText("—")
            return
        step = self.transition_step_input.value()
        if run_mode == TransitionRunMode.ENVELOPE:
            start = self.transition_range_start_input.value()
            end = self.transition_range_end_input.value()
            if step <= 0 or end <= start:
                self.transition_estimate_label.setText("—")
                return
            steps = int(math.floor((end - start) / step)) + 1
        else:
            steps = 1
        domain = (self.transition_domain_start_input.value(), self.transition_domain_end_input.value())
        try:
            profile_type = self._coerce_transition_profile_type(self.transition_profile_combo.currentData())
        except ValueError:
            self.transition_estimate_label.setText("—")
            return
        k_value = mn_per_m2_to_n_per_m2(self.transition_k1_input.value())
        if profile_type != TransitionProfileType.UNIFORM:
            k2 = mn_per_m2_to_n_per_m2(self.transition_k2_input.value())
            if k2 > 0:
                k_value = min(k_value, k2)
        is_numerical = self.advanced_solver_checkbox.isChecked() or profile_type != TransitionProfileType.UNIFORM
        grid_points = 401
        if is_numerical:
            align_to_sleepers = (
                self.discrete_supports_checkbox.isChecked() or self.two_rail_checkbox.isChecked()
            )
            grid_points = self._estimate_grid_points(
                rail=self.rail_combo.currentData(),
                k_n_per_m2=k_value,
                domain=domain,
                align_to_sleepers=align_to_sleepers,
            )
        estimate = self._format_runtime_estimate(steps, grid_points, is_numerical)
        self.transition_estimate_label.setText(estimate)

    def _update_transition_profile_visibility(self, *_: object) -> None:
        if not hasattr(self, "transition_profile_combo"):
            return
        try:
            profile_type = self._coerce_transition_profile_type(self.transition_profile_combo.currentData())
        except ValueError:
            return
        layout = self.transition_group.layout()
        if isinstance(layout, QFormLayout):
            self._set_form_row_visible(layout, self.transition_k2_input, profile_type != TransitionProfileType.UNIFORM)
            self._set_form_row_visible(
                layout,
                self.transition_length_container,
                profile_type in (TransitionProfileType.RAMP, TransitionProfileType.EXPONENTIAL),
            )
            self._set_form_row_visible(
                layout,
                self.transition_segment_length_input,
                profile_type == TransitionProfileType.SEGMENT,
            )
        self._update_transition_length_auto()
        self._update_transition_domain_auto()
        self._update_solver_backend_label()

    def _toggle_transition_length_auto(self, enabled: bool) -> None:
        self.transition_length_input.setEnabled(not enabled)
        if enabled:
            self._update_transition_length_auto()

    def _update_transition_length_auto(self) -> None:
        if not self.transition_length_auto_checkbox.isChecked():
            return
        try:
            profile_type = self._coerce_transition_profile_type(self.transition_profile_combo.currentData())
        except ValueError:
            return
        if profile_type not in (TransitionProfileType.RAMP, TransitionProfileType.EXPONENTIAL):
            return
        speed_m_per_s = self.design_speed_input.value() * (1000.0 / 3600.0)
        bogie_spacing_m = mm_to_m(self.train_bogie_spacing_input.value())
        if speed_m_per_s <= 0 and bogie_spacing_m <= 0:
            return
        default_length = max(speed_m_per_s, bogie_spacing_m)
        if default_length <= 0:
            return
        self.transition_length_input.set_value(default_length)
        self._update_transition_range_auto()
        self._update_transition_domain_auto()

    def _apply_transition_template(self) -> None:
        template = self.transition_template_combo.currentData()
        if template == "custom":
            return
        if template == "local_stiff":
            self.transition_profile_combo.setCurrentIndex(
                self.transition_profile_combo.findData(TransitionProfileType.SEGMENT)
            )
            self.transition_segment_length_input.set_value(
                max(mm_to_m(self.sleeper_spacing_input.value()), 0.1)
            )
        else:
            self.transition_profile_combo.setCurrentIndex(
                self.transition_profile_combo.findData(TransitionProfileType.RAMP)
            )
            self.transition_length_auto_checkbox.setChecked(True)
        self._update_transition_profile_visibility()
        self._update_transition_length_auto()
        self._update_transition_domain_auto()

    def _apply_transition_preset(self) -> None:
        preset = self.transition_preset_combo.currentData()
        if preset == "pwi":
            self.transition_k1_input.set_value(76.9)
        self._update_transition_domain_auto()

    def _sync_transition_defaults(self, *_: object) -> None:
        support = self.support_combo.currentData()
        if support is None:
            return
        if self.transition_preset_combo.currentData() == "custom":
            self.transition_k1_input.set_value(n_per_m2_to_mn_per_m2(support.foundation_modulus_n_per_m2))
        if hasattr(self, "dynamic_transition_k1_input"):
            self.dynamic_transition_k1_input.set_value(
                n_per_m2_to_mn_per_m2(support.foundation_modulus_n_per_m2)
            )
        self._update_transition_domain_auto()
    def _toggle_dynamic_mode(self, *_: object) -> None:
        self._clear_all_overlays(render=False)
        try:
            current_mode = self._coerce_dynamic_mode(self.dynamic_mode_combo.currentData())
        except ValueError:
            current_mode = DynamicMode.STEADY_STATE
        self._last_dynamic_result = None
        self._last_dynamic_config = None
        self._last_dynamic_load_source = None
        self._last_dynamic_transition_result = None
        self._last_dynamic_transition_config = None
        self._last_dynamic_transition_load_source = None
        self._last_dipped_joint_result = None
        self._last_dipped_joint_config = None
        self._set_dynamic_export_buttons_enabled(False)
        dynamic_layout = self.dynamic_container.layout()
        if not isinstance(dynamic_layout, QFormLayout):
            return
        show_time_controls = current_mode in (DynamicMode.TIME_HISTORY, DynamicMode.TRANSITION)
        show_moving_controls = current_mode in (
            DynamicMode.TIME_HISTORY,
            DynamicMode.STEADY_STATE,
            DynamicMode.TRANSITION,
        )
        show_dipped_joint = current_mode == DynamicMode.DIPPED_JOINT
        show_transition = current_mode == DynamicMode.TRANSITION

        self._set_form_row_visible(dynamic_layout, self.damping_mode_combo, show_moving_controls)
        self._set_form_row_visible(dynamic_layout, self.foundation_damping_model_combo, show_moving_controls)
        if show_moving_controls:
            self._toggle_damping_inputs()
        else:
            self._set_form_row_visible(dynamic_layout, self.damping_coefficient_input, False)
            self._set_form_row_visible(dynamic_layout, self.damping_ratio_input, False)
            self._set_form_row_visible(dynamic_layout, self.damping_loss_factor_input, False)
        self._set_form_row_visible(dynamic_layout, self.probe_locations_input, show_moving_controls)
        self._set_form_row_visible(dynamic_layout, self.probe_selection_combo, show_moving_controls)
        self._set_form_row_visible(dynamic_layout, self.time_window_container, show_time_controls)
        self._set_form_row_visible(dynamic_layout, self.sample_rate_input, show_time_controls)
        self._set_form_row_visible(dynamic_layout, self.dynamic_advanced_group, show_moving_controls)
        self._set_form_row_visible(dynamic_layout, self.dynamic_transition_group, show_transition)
        if show_transition:
            self._update_dynamic_transition_profile_visibility()
            self._toggle_dynamic_transition_run_mode()
        self._set_form_row_visible(dynamic_layout, self.dipped_joint_group, show_dipped_joint)
        analysis_layout = getattr(self, "analysis_layout", None)
        if isinstance(analysis_layout, QFormLayout):
            self._set_form_row_visible(analysis_layout, self.load_position_input, not show_dipped_joint)
        if show_dipped_joint:
            for checkbox in (
                self.several_loads_checkbox,
                self.train_loads_checkbox,
                self.as5100_loads_checkbox,
            ):
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.blockSignals(False)
            self.several_loads_checkbox.setVisible(False)
            self.train_loads_checkbox.setVisible(False)
            self.as5100_loads_checkbox.setVisible(False)
            self.wheel_loads_group.setVisible(False)
            self.train_loads_group.setVisible(False)
            self.as5100_loads_group.setVisible(False)
            self.load_case_combo.setEnabled(False)
            self.load_magnitude_input.setEnabled(False)
            self.load_position_input.setEnabled(False)
        else:
            self.several_loads_checkbox.setVisible(True)
            self.train_loads_checkbox.setVisible(True)
            self.as5100_loads_checkbox.setVisible(True)
            self._sync_load_input_state()

        if hasattr(self, "tab_widget"):
            self._set_dynamic_tabs_visible(
                self.analysis_type_combo.currentData() == AnalysisType.DYNAMIC,
                mode=current_mode,
            )
            self._enforce_dynamic_transition_advanced_constraints()
            if show_moving_controls:
                show_frequency = current_mode in (DynamicMode.STEADY_STATE, DynamicMode.TRANSITION)
                self.tab_widget.setTabVisible(self.dynamic_fft_tab_index, show_frequency)
                self.tab_widget.setTabVisible(self.dynamic_psd_tab_index, show_frequency)
            if self.analysis_type_combo.currentData() == AnalysisType.DYNAMIC:
                self._switch_to_appropriate_tab()
        show_transition_exports = current_mode == DynamicMode.TRANSITION
        show_dipped_export = current_mode == DynamicMode.DIPPED_JOINT
        self.export_dynamic_time_button.setVisible(not show_dipped_export)
        self.export_dynamic_fft_button.setVisible(not show_dipped_export)
        self.export_dynamic_psd_button.setVisible(not show_dipped_export)
        self.export_dynamic_transition_metrics_button.setVisible(show_transition_exports)
        self.export_dynamic_transition_series_button.setVisible(show_transition_exports)
        self.export_dynamic_transition_config_button.setVisible(show_transition_exports)
        self.export_dipped_joint_button.setVisible(show_dipped_export)
        self._update_overlay_state()
        self._refresh_chart_grid_if_visible()

    def _handle_dynamic_annotation_mode_changed(self, *_: object) -> None:
        if self._last_dynamic_transition_result is not None:
            self._render_dynamic_transition_result(self._last_dynamic_transition_result)
            return
        if self._last_dynamic_result is None:
            return
        if self._dynamic_overlay_results:
            self._render_dynamic_overlay_plots()
            return
        self._render_dynamic_result(self._last_dynamic_result)

    def _chart_input_labels_visible(self) -> bool:
        checkbox = getattr(self, "chart_input_labels_checkbox", None)
        return not isinstance(checkbox, QCheckBox) or checkbox.isChecked()

    def _chart_output_labels_visible(self) -> bool:
        checkbox = getattr(self, "chart_output_labels_checkbox", None)
        return not isinstance(checkbox, QCheckBox) or checkbox.isChecked()

    def _chart_extrema_labels_visible(self) -> bool:
        checkbox = getattr(self, "chart_extrema_labels_checkbox", None)
        return not isinstance(checkbox, QCheckBox) or checkbox.isChecked()

    def _chart_input_load_markers(
        self,
        load_markers: Sequence[LoadMarker] | None,
    ) -> Sequence[LoadMarker] | None:
        return load_markers if self._chart_input_labels_visible() else None

    def _handle_chart_label_visibility_changed(self, *_: object) -> None:
        self._refresh_rendered_chart_labels()

    def _refresh_rendered_chart_labels(self) -> None:
        if self._last_transition_result is not None:
            if self._last_envelope_result is not None:
                self._render_transition_result(
                    self._last_transition_result,
                    envelope_result=self._last_envelope_result,
                )
            elif self._last_analysis_result is not None:
                self._render_transition_result(
                    self._last_transition_result,
                    analysis_result=self._last_analysis_result,
                )
            else:
                self._apply_transition_annotations_to_visible_charts(self._last_transition_result)
                self._rerender_last_stress_chart()
            self._refresh_chart_grid_if_visible()
            return

        if self._last_envelope_result is not None:
            self._render_envelope_result(self._last_envelope_result)
            self._refresh_chart_grid_if_visible()
            return

        if self._last_analysis_result is not None:
            if self._overlay_results:
                self._render_overlay_plots()
            else:
                self._render_analysis_result(self._last_analysis_result)
            self._refresh_chart_grid_if_visible()
            return

        if self._last_dynamic_transition_result is not None:
            self._render_dynamic_transition_result(self._last_dynamic_transition_result)
            self._refresh_chart_grid_if_visible()
            return

        if self._last_dynamic_result is not None:
            if self._dynamic_overlay_results:
                self._render_dynamic_overlay_plots()
            else:
                self._render_dynamic_result(self._last_dynamic_result)
            self._refresh_chart_grid_if_visible()
            return

        if self._last_special_result is not None:
            self._render_special_result(self._last_special_result)
            self._refresh_chart_grid_if_visible()
            return

        self._rerender_last_stress_chart()
        self._refresh_chart_grid_if_visible()

    def _toggle_special_mode(self, *_: object) -> None:
        if not hasattr(self, "special_container"):
            return
        mode = self.special_mode_combo.currentData()
        show_floating_slab = mode == SpecialMode.FLOATING_SLAB
        self._clear_special_state()
        self.floating_slab_group.setVisible(show_floating_slab)
        if hasattr(self, "tab_widget"):
            self._set_special_tabs_visible(
                self.analysis_type_combo.currentData() == AnalysisType.SPECIAL
            )
            if self.analysis_type_combo.currentData() == AnalysisType.SPECIAL:
                self._switch_to_appropriate_tab()
        self._refresh_chart_grid_if_visible()

    def _toggle_damping_inputs(self, *_: object) -> None:
        mode = self.damping_mode_combo.currentData()
        damping_model = self.foundation_damping_model_combo.currentData()
        dynamic_layout = self.dynamic_container.layout()
        if not isinstance(dynamic_layout, QFormLayout):
            return
        is_viscous = damping_model == DampingModel.VISCOUS
        self._set_form_row_visible(
            dynamic_layout, self.damping_coefficient_input, is_viscous and mode == "coefficient"
        )
        self._set_form_row_visible(
            dynamic_layout, self.damping_ratio_input, is_viscous and mode == "ratio"
        )
        self._set_form_row_visible(dynamic_layout, self.damping_loss_factor_input, not is_viscous)

    def _toggle_foundation_damping_inputs(self, *_: object) -> None:
        is_hysteretic = (
            self.foundation_damping_model_static_combo.currentData() == DampingModel.HYSTERETIC
        )
        foundation_model = self.foundation_model_combo.currentData()
        is_multilayer = foundation_model in (
            FoundationModelType.SERIES,
            FoundationModelType.SLEEPER_MASS,
        )
        discrete_enabled = self.discrete_supports_checkbox.isChecked()

        self.railpad_damping_input.setEnabled(not is_hysteretic and is_multilayer)
        self.trackbed_damping_input.setEnabled(not is_hysteretic and is_multilayer)
        self.pad_damping_input.setEnabled(not is_hysteretic and discrete_enabled)
        self.railpad_loss_factor_input.setEnabled(is_hysteretic and is_multilayer)
        self.trackbed_loss_factor_input.setEnabled(is_hysteretic)
        self.pad_loss_factor_input.setEnabled(is_hysteretic and discrete_enabled)

    def _toggle_dynamic_advanced_controls(self, enabled: bool) -> None:
        self.domain_length_input.setEnabled(enabled)
        self.spatial_step_input.setEnabled(enabled)
        self.psd_segment_length_input.setEnabled(enabled)
        self.psd_overlap_input.setEnabled(enabled)
        self.dynamic_excitation_mode_combo.setEnabled(enabled)
        self.dynamic_boundary_mode_combo.setEnabled(enabled)
        self.irregularity_mode_combo.setEnabled(enabled)
        self.irregularity_seed_input.setEnabled(enabled)
        self._toggle_dynamic_extra_inputs()
        self._enforce_dynamic_transition_advanced_constraints()

    def _enforce_dynamic_transition_advanced_constraints(self, *_: object) -> None:
        if (
            not hasattr(self, "dynamic_mode_combo")
            or not hasattr(self, "dynamic_advanced_group")
            or not hasattr(self, "dynamic_excitation_mode_combo")
            or not hasattr(self, "dynamic_boundary_mode_combo")
            or not hasattr(self, "irregularity_mode_combo")
            or not hasattr(self, "foundation_damping_model_combo")
            or not hasattr(self, "dynamic_transition_solver_fidelity_combo")
        ):
            return

        try:
            mode = self._coerce_dynamic_mode(self.dynamic_mode_combo.currentData())
        except ValueError:
            return

        advanced_enabled = self.dynamic_advanced_group.isChecked()

        def _set_combo_value(combo: QComboBox, value: object) -> None:
            index = combo.findData(value)
            if index < 0 or combo.currentIndex() == index:
                return
            blocked = combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(blocked)

        if mode != DynamicMode.TRANSITION:
            self.dynamic_excitation_mode_combo.setEnabled(advanced_enabled)
            self.dynamic_boundary_mode_combo.setEnabled(advanced_enabled)
            self.irregularity_mode_combo.setEnabled(advanced_enabled)
            self.foundation_damping_model_combo.setEnabled(True)
            self.irregularity_seed_input.setEnabled(advanced_enabled)
            self._toggle_dynamic_extra_inputs()
            self._toggle_damping_inputs()
            return

        # Dynamic transition currently supports moving-load response only.
        _set_combo_value(self.dynamic_excitation_mode_combo, DynamicExcitationMode.MOVING_LOAD)
        self.dynamic_excitation_mode_combo.setEnabled(False)

        full_profile = self.dynamic_transition_solver_fidelity_combo.currentData() == "full_profile"
        if full_profile:
            _set_combo_value(self.dynamic_boundary_mode_combo, DynamicBoundaryMode.ZERO_PAD)
            _set_combo_value(self.irregularity_mode_combo, None)
            _set_combo_value(self.foundation_damping_model_combo, DampingModel.VISCOUS)

        self.dynamic_boundary_mode_combo.setEnabled(advanced_enabled and not full_profile)
        self.irregularity_mode_combo.setEnabled(advanced_enabled and not full_profile)
        self.foundation_damping_model_combo.setEnabled(not full_profile)
        self.irregularity_seed_input.setEnabled(advanced_enabled and not full_profile)
        self._toggle_dynamic_extra_inputs()
        self._toggle_damping_inputs()

    def _toggle_dynamic_extra_inputs(self, *_: object) -> None:
        advanced_enabled = self.dynamic_advanced_group.isChecked()
        oscillator_enabled = (
            self.dynamic_excitation_mode_combo.currentData() == DynamicExcitationMode.MOVING_OSCILLATOR
            and advanced_enabled
        )
        for widget in (
            self.oscillator_unsprung_mass_input,
            self.oscillator_stiffness_input,
            self.oscillator_damping_input,
        ):
            widget.setEnabled(oscillator_enabled)

        irregularity_mode = self.irregularity_mode_combo.currentData()
        profile_enabled = irregularity_mode == IrregularityMode.PROFILE and advanced_enabled
        synthetic_enabled = irregularity_mode == IrregularityMode.SYNTHETIC_PSD and advanced_enabled
        self.irregularity_profile_x_input.setEnabled(profile_enabled)
        self.irregularity_profile_z_input.setEnabled(profile_enabled)
        self.irregularity_psd_level_input.setEnabled(synthetic_enabled)

    def _toggle_time_window_auto(self, enabled: bool) -> None:
        self.time_window_input.setEnabled(not enabled)
        if enabled:
            self._update_time_window_auto()

    def _update_time_window_auto(self, *_: object) -> None:
        if not self.time_window_auto_checkbox.isChecked():
            return
        speed = max(self.speed_input.value(), 0.1)
        default_time_window = self.domain_length_input.value() / speed * 1.2
        self.time_window_input.set_value(default_time_window)

    def _toggle_envelope_range_auto(self, enabled: bool) -> None:
        self.envelope_range_start_input.setEnabled(not enabled)
        self.envelope_range_end_input.setEnabled(not enabled)
        if enabled:
            self._update_envelope_range_auto()
        self._update_envelope_estimate()

    def _update_envelope_range_auto(self) -> None:
        if not self.envelope_range_auto_checkbox.isChecked():
            return
        rail = self.rail_combo.currentData()
        support = self.support_combo.currentData()
        if rail is None or support is None:
            return
        beta = beam_parameter_beta(
            support.foundation_modulus_n_per_m2,
            rail.elastic_modulus_pa,
            rail.moment_inertia_m4,
        )
        length = ENVELOPE_AUTO_MOVEMENT_FACTOR / beta
        self.envelope_range_start_input.set_value(-length)
        self.envelope_range_end_input.set_value(length)
        self._update_envelope_estimate()

    def _toggle_envelope_domain_auto(self, enabled: bool) -> None:
        self.envelope_domain_start_input.setEnabled(not enabled)
        self.envelope_domain_end_input.setEnabled(not enabled)
        if enabled:
            self._update_envelope_domain_auto()
        self._update_envelope_estimate()

    def _update_envelope_domain_auto(self) -> None:
        if not self.envelope_domain_auto_checkbox.isChecked():
            return
        try:
            loads = self._collect_analysis_loads()
        except ValueError:
            return
        if not loads:
            return
        rail = self.rail_combo.currentData()
        support = self.support_combo.currentData()
        if rail is None or support is None:
            return
        if self.envelope_range_auto_checkbox.isChecked():
            self._update_envelope_range_auto()
        reference = self.envelope_reference_input.value()
        x_ref_start = reference + self.envelope_range_start_input.value()
        x_ref_end = reference + self.envelope_range_end_input.value()
        beta = beam_parameter_beta(
            support.foundation_modulus_n_per_m2,
            rail.elastic_modulus_pa,
            rail.moment_inertia_m4,
        )
        margin = ENVELOPE_AUTO_DECAY_MARGIN_FACTOR / beta
        offsets = [load.position_m for load in loads]
        x_min = x_ref_start + min(offsets) - margin
        x_max = x_ref_end + max(offsets) + margin
        if x_max <= x_min:
            return
        self.envelope_domain_start_input.set_value(x_min)
        self.envelope_domain_end_input.set_value(x_max)
        self._update_envelope_estimate()

    def _refresh_probe_selection(self, *_: object) -> None:
        positions = self._safe_parse_probe_positions()
        self.probe_selection_combo.blockSignals(True)
        self.probe_selection_combo.clear()
        for position in positions:
            self.probe_selection_combo.addItem(f"{position:.2f} m", position)
        self.probe_selection_combo.blockSignals(False)

    def _safe_parse_probe_positions(self) -> list[float]:
        raw = self.probe_locations_input.text().strip()
        if not raw:
            return [0.0]
        positions: list[float] = []
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            try:
                positions.append(float(entry))
            except ValueError:
                continue
        return positions or [0.0]

    def _switch_to_appropriate_tab(self) -> None:
        """Switch to the appropriate tab based on analysis type and mode."""
        if hasattr(self, "chart_view_combo") and self.chart_view_combo.currentData() == "all":
            self._apply_chart_view("all")
            return
        analysis_type = self.analysis_type_combo.currentData()
        if analysis_type == AnalysisType.STATIC:
            if (
                hasattr(self, "transition_group")
                and self._last_transition_result is not None
            ):
                self.tab_widget.setCurrentIndex(self.transition_summary_tab_index)
            else:
                self.tab_widget.setCurrentIndex(self.deflection_tab_index)
            return
        if analysis_type == AnalysisType.DYNAMIC:
            mode = self.dynamic_mode_combo.currentData()
            if mode == DynamicMode.DIPPED_JOINT:
                self.tab_widget.setCurrentIndex(self.dipped_joint_summary_tab_index)
            else:
                self.tab_widget.setCurrentIndex(self.dynamic_deflection_tab_index)
            return
        if analysis_type == AnalysisType.SPECIAL:
            if hasattr(self, "special_summary_tab_index"):
                self.tab_widget.setCurrentIndex(self.special_summary_tab_index)

    def _set_form_row_visible(self, layout: QFormLayout, widget: QWidget, visible: bool) -> None:
        widget.setVisible(visible)
        label = layout.labelForField(widget)
        if label is not None:
            label.setVisible(visible)

    def _set_dynamic_tabs_visible(self, visible: bool, *, mode: DynamicMode | None = None) -> None:
        if mode is None:
            try:
                mode = self._coerce_dynamic_mode(self.dynamic_mode_combo.currentData())
            except ValueError:
                mode = DynamicMode.STEADY_STATE
        show_dipped_joint = visible and mode == DynamicMode.DIPPED_JOINT
        show_moving_load = visible and mode != DynamicMode.DIPPED_JOINT
        show_dynamic_transition_profile = visible and mode == DynamicMode.TRANSITION
        self.tab_widget.setTabVisible(self.deflection_tab_index, not visible)
        self.tab_widget.setTabVisible(self.moment_tab_index, not visible)
        self.tab_widget.setTabVisible(self.shear_tab_index, not visible)
        self.tab_widget.setTabVisible(self.reaction_tab_index, not visible)
        self.tab_widget.setTabVisible(self.sleeper_tab_index, not visible)
        self.tab_widget.setTabVisible(self.pressure_tab_index, not visible)
        self.tab_widget.setTabVisible(self.stress_tab_index, visible)
        show_static_rail_tabs = False
        if not visible:
            if self._last_analysis_result is not None:
                show_static_rail_tabs = (
                    self._last_analysis_result.left_deflection_m is not None
                    and self._last_analysis_result.right_deflection_m is not None
                    and self._last_analysis_result.left_moment_nm is not None
                    and self._last_analysis_result.right_moment_nm is not None
                )
            elif self._last_envelope_result is not None:
                show_static_rail_tabs = (
                    self._last_envelope_result.left_deflection_max_m is not None
                    and self._last_envelope_result.right_deflection_max_m is not None
                    and self._last_envelope_result.left_moment_max_nm is not None
                    and self._last_envelope_result.right_moment_max_nm is not None
                )
        self.tab_widget.setTabVisible(self.rail_deflection_tab_index, show_static_rail_tabs)
        self.tab_widget.setTabVisible(self.rail_moment_tab_index, show_static_rail_tabs)
        self.tab_widget.setTabVisible(self.dynamic_deflection_tab_index, show_moving_load)
        self.tab_widget.setTabVisible(self.dynamic_moment_tab_index, show_moving_load)
        self.tab_widget.setTabVisible(self.dynamic_shear_tab_index, show_moving_load)
        self.tab_widget.setTabVisible(self.dynamic_reaction_tab_index, show_moving_load)
        self.tab_widget.setTabVisible(self.dynamic_damping_tab_index, show_moving_load)
        self.tab_widget.setTabVisible(self.dynamic_time_tab_index, show_moving_load)
        self.tab_widget.setTabVisible(self.dynamic_fft_tab_index, show_moving_load)
        self.tab_widget.setTabVisible(self.dynamic_psd_tab_index, show_moving_load)
        self.tab_widget.setTabVisible(self.dynamic_impedance_tab_index, show_moving_load)
        self.tab_widget.setTabVisible(self.dynamic_summary_tab_index, show_moving_load)
        self.tab_widget.setTabVisible(self.dipped_joint_summary_tab_index, show_dipped_joint)
        self.tab_widget.setTabVisible(self.static_summary_tab_index, not visible)
        if hasattr(self, "special_summary_tab_index"):
            self.tab_widget.setTabVisible(self.special_summary_tab_index, False)
        if hasattr(self, "special_floating_slab_tab_index"):
            self.tab_widget.setTabVisible(self.special_floating_slab_tab_index, False)
        if hasattr(self, "special_floating_slab_attenuation_tab_index"):
            self.tab_widget.setTabVisible(self.special_floating_slab_attenuation_tab_index, False)
        if hasattr(self, "transition_profile_tab_index"):
            show_static_transition = not visible and getattr(self, "transition_group", None) is not None
            if show_static_transition:
                show_static_transition = (
                    self.transition_group.isChecked() or self._last_transition_result is not None
                )
            self.tab_widget.setTabVisible(
                self.transition_profile_tab_index,
                show_static_transition or show_dynamic_transition_profile,
            )
            self.tab_widget.setTabVisible(self.transition_summary_tab_index, show_static_transition)
        self._refresh_chart_grid_if_visible()

    def _set_special_tabs_visible(self, visible: bool) -> None:
        if not hasattr(self, "tab_widget"):
            return
        if visible:
            for tab_index in (
                self.deflection_tab_index,
                self.moment_tab_index,
                self.shear_tab_index,
                self.reaction_tab_index,
                self.sleeper_tab_index,
                self.pressure_tab_index,
                self.rail_deflection_tab_index,
                self.rail_moment_tab_index,
                self.dynamic_deflection_tab_index,
                self.dynamic_moment_tab_index,
                self.dynamic_shear_tab_index,
                self.dynamic_reaction_tab_index,
                self.dynamic_damping_tab_index,
                self.dynamic_time_tab_index,
                self.dynamic_fft_tab_index,
                self.dynamic_psd_tab_index,
                self.dynamic_impedance_tab_index,
                self.dynamic_summary_tab_index,
                self.dipped_joint_summary_tab_index,
                self.static_summary_tab_index,
            ):
                self.tab_widget.setTabVisible(tab_index, False)
            self.tab_widget.setTabVisible(self.stress_tab_index, True)
            if hasattr(self, "transition_profile_tab_index"):
                self.tab_widget.setTabVisible(self.transition_profile_tab_index, False)
            if hasattr(self, "transition_summary_tab_index"):
                self.tab_widget.setTabVisible(self.transition_summary_tab_index, False)
            if hasattr(self, "special_summary_tab_index"):
                self.tab_widget.setTabVisible(self.special_summary_tab_index, True)
            if hasattr(self, "special_floating_slab_tab_index"):
                self.tab_widget.setTabVisible(self.special_floating_slab_tab_index, True)
            if hasattr(self, "special_floating_slab_attenuation_tab_index"):
                self.tab_widget.setTabVisible(self.special_floating_slab_attenuation_tab_index, True)
        else:
            if hasattr(self, "special_summary_tab_index"):
                self.tab_widget.setTabVisible(self.special_summary_tab_index, False)
            if hasattr(self, "special_floating_slab_tab_index"):
                self.tab_widget.setTabVisible(self.special_floating_slab_tab_index, False)
            if hasattr(self, "special_floating_slab_attenuation_tab_index"):
                self.tab_widget.setTabVisible(self.special_floating_slab_attenuation_tab_index, False)
            self.tab_widget.setTabVisible(self.stress_tab_index, True)
        self._refresh_chart_grid_if_visible()

    def _build_plot_panel(self) -> QWidget:
        self.tab_widget = QTabWidget()
        self.tab_widget.setMinimumWidth(420)
        self.deflection_plot = PlotPanel()
        self.moment_plot = PlotPanel()
        self.shear_plot = PlotPanel()
        self.reaction_plot = PlotPanel()
        self.sleeper_plot = PlotPanel()
        self.pressure_plot = PlotPanel()
        self.stress_plot = PlotPanel()
        self.stress_panel = QWidget()
        stress_panel_layout = QHBoxLayout(self.stress_panel)
        stress_panel_layout.setContentsMargins(0, 0, 0, 0)
        stress_panel_layout.setSpacing(8)
        stress_panel_layout.addWidget(self.stress_plot, stretch=1)
        stress_controls = QWidget()
        stress_controls.setMaximumWidth(220)
        stress_controls_layout = QVBoxLayout(stress_controls)
        stress_controls_layout.setContentsMargins(8, 8, 8, 8)
        stress_controls_layout.setSpacing(8)
        stress_controls_title = QLabel("Stress series")
        stress_controls_title.setStyleSheet("font-weight: 600;")
        stress_controls_layout.addWidget(stress_controls_title)
        self.stress_rail_checkbox = QCheckBox("Rail bending stress")
        self.stress_rail_checkbox.setChecked(True)
        self.stress_ballast_checkbox = QCheckBox("Ballast pressure")
        self.stress_ballast_checkbox.setChecked(False)
        self.stress_capping_checkbox = QCheckBox("Capping pressure")
        self.stress_capping_checkbox.setChecked(False)
        for checkbox in (
            self.stress_rail_checkbox,
            self.stress_ballast_checkbox,
            self.stress_capping_checkbox,
        ):
            checkbox.toggled.connect(self._rerender_last_stress_chart)
            stress_controls_layout.addWidget(checkbox)
        stress_controls_layout.addStretch(1)
        stress_panel_layout.addWidget(stress_controls)
        self.rail_deflection_plot = PlotPanel()
        self.rail_moment_plot = PlotPanel()
        self.summary_panel = SummaryPanel()
        self.transition_profile_plot = PlotPanel()
        self.transition_summary_panel = TransitionSummaryPanel()
        self.dynamic_deflection_plot = PlotPanel()
        self.dynamic_moment_plot = PlotPanel()
        self.dynamic_shear_plot = PlotPanel()
        self.dynamic_reaction_plot = PlotPanel()
        self.dynamic_damping_plot = PlotPanel()
        self.dynamic_time_plot = PlotPanel()
        self.dynamic_fft_plot = PlotPanel()
        self.dynamic_psd_plot = PlotPanel()
        self.dynamic_impedance_plot = PlotPanel()
        self.special_floating_slab_plot = PlotPanel()
        self.special_floating_slab_attenuation_plot = PlotPanel()
        self.dynamic_deflection_plot.configure_help_action(
            callback=lambda: self._show_dynamic_chart_help("dynamic_deflection", "Dynamic deflection"),
            tooltip="Explain dynamic deflection chart",
        )
        self.dynamic_moment_plot.configure_help_action(
            callback=lambda: self._show_dynamic_chart_help("dynamic_moment", "Dynamic moment"),
            tooltip="Explain dynamic moment chart",
        )
        self.dynamic_shear_plot.configure_help_action(
            callback=lambda: self._show_dynamic_chart_help("dynamic_shear", "Dynamic shear"),
            tooltip="Explain dynamic shear chart",
        )
        self.dynamic_reaction_plot.configure_help_action(
            callback=lambda: self._show_dynamic_chart_help("dynamic_reaction", "Dynamic rail support reaction"),
            tooltip="Explain dynamic rail support reaction chart",
        )
        self.dynamic_damping_plot.configure_help_action(
            callback=lambda: self._show_dynamic_chart_help("dynamic_damping", "Dynamic damping"),
            tooltip="Explain dynamic damping-force chart",
        )
        self.dynamic_time_plot.configure_help_action(
            callback=lambda: self._show_dynamic_chart_help("dynamic_time", "Dynamic time history"),
            tooltip="Explain dynamic time-history chart",
        )
        self.dynamic_fft_plot.configure_help_action(
            callback=lambda: self._show_dynamic_chart_help("dynamic_fft", "Dynamic FFT"),
            tooltip="Explain dynamic FFT chart",
        )
        self.dynamic_psd_plot.configure_help_action(
            callback=lambda: self._show_dynamic_chart_help("dynamic_psd", "Dynamic PSD"),
            tooltip="Explain dynamic PSD chart",
        )
        self.dynamic_impedance_plot.configure_help_action(
            callback=lambda: self._show_dynamic_chart_help("dynamic_impedance", "Dynamic impedance"),
            tooltip="Explain dynamic impedance chart",
        )
        self.dynamic_summary_panel = DynamicSummaryPanel()
        self.dipped_joint_summary_panel = DippedJointSummaryPanel()
        self.special_summary_panel = SpecialSummaryPanel()
        self.help_button = QToolButton()
        self.help_button.setText("Help")
        self.help_button.setAutoRaise(True)
        self.help_button.clicked.connect(self._show_help_dialog)
        self.chart_view_combo = QComboBox()
        self.chart_view_combo.addItem("Single", "single")
        self.chart_view_combo.addItem("All", "all")
        self.chart_view_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.chart_view_combo.currentIndexChanged.connect(self._toggle_chart_view)

        corner_widget = QWidget()
        corner_layout = QHBoxLayout(corner_widget)
        corner_layout.setContentsMargins(0, 0, 0, 0)
        corner_layout.setSpacing(6)
        corner_layout.addWidget(self.help_button)
        corner_layout.addWidget(self.chart_view_combo)

        self.chart_grid_scroll = QScrollArea()
        self.chart_grid_scroll.setWidgetResizable(True)
        self.chart_grid_container = QWidget()
        self.chart_grid_layout = QGridLayout(self.chart_grid_container)
        self.chart_grid_layout.setContentsMargins(12, 12, 12, 12)
        self.chart_grid_layout.setSpacing(12)
        self.chart_grid_scroll.setWidget(self.chart_grid_container)

        def summary_scroll(widget: QWidget) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(widget)
            return scroll

        self.deflection_tab_index = self.tab_widget.addTab(self.deflection_plot, "Deflection")
        self.moment_tab_index = self.tab_widget.addTab(self.moment_plot, "Moment")
        self.shear_tab_index = self.tab_widget.addTab(self.shear_plot, "Shear")
        self.reaction_tab_index = self.tab_widget.addTab(self.reaction_plot, "Rail reaction")
        self.sleeper_tab_index = self.tab_widget.addTab(self.sleeper_plot, "Sleeper loads")
        self.pressure_tab_index = self.tab_widget.addTab(self.pressure_plot, "Pressures")
        self.stress_tab_index = self.tab_widget.addTab(self.stress_panel, "Stress")
        self.rail_deflection_tab_index = self.tab_widget.addTab(
            self.rail_deflection_plot, "Rail deflection (L/R)"
        )
        self.rail_moment_tab_index = self.tab_widget.addTab(
            self.rail_moment_plot, "Rail moment (L/R)"
        )
        self.static_summary_tab_index = self.tab_widget.addTab(summary_scroll(self.summary_panel), "Summary")
        self.transition_profile_tab_index = self.tab_widget.addTab(
            self.transition_profile_plot, "Transition k(x)"
        )
        self.transition_summary_tab_index = self.tab_widget.addTab(
            summary_scroll(self.transition_summary_panel), "Transition metrics"
        )
        self.dynamic_deflection_tab_index = self.tab_widget.addTab(self.dynamic_deflection_plot, "Dyn deflection")
        self.dynamic_moment_tab_index = self.tab_widget.addTab(self.dynamic_moment_plot, "Dyn moment")
        self.dynamic_shear_tab_index = self.tab_widget.addTab(self.dynamic_shear_plot, "Dyn shear")
        self.dynamic_reaction_tab_index = self.tab_widget.addTab(self.dynamic_reaction_plot, "Dyn rail reaction")
        self.dynamic_damping_tab_index = self.tab_widget.addTab(self.dynamic_damping_plot, "Dyn damping")
        self.dynamic_time_tab_index = self.tab_widget.addTab(self.dynamic_time_plot, "Dyn time history")
        self.dynamic_fft_tab_index = self.tab_widget.addTab(self.dynamic_fft_plot, "Dyn FFT")
        self.dynamic_psd_tab_index = self.tab_widget.addTab(self.dynamic_psd_plot, "Dyn PSD")
        self.dynamic_impedance_tab_index = self.tab_widget.addTab(self.dynamic_impedance_plot, "Dyn impedance")
        self.dynamic_summary_tab_index = self.tab_widget.addTab(
            summary_scroll(self.dynamic_summary_panel), "Dyn summary"
        )
        self.dipped_joint_summary_tab_index = self.tab_widget.addTab(
            summary_scroll(self.dipped_joint_summary_panel), "Dipped joint summary"
        )
        self.special_floating_slab_tab_index = self.tab_widget.addTab(
            self.special_floating_slab_plot, "Slab transmissibility"
        )
        self.special_floating_slab_attenuation_tab_index = self.tab_widget.addTab(
            self.special_floating_slab_attenuation_plot, "Slab attenuation"
        )
        self.special_summary_tab_index = self.tab_widget.addTab(
            summary_scroll(self.special_summary_panel), "Special summary"
        )
        self.all_charts_tab_index = self.tab_widget.addTab(self.chart_grid_scroll, "All charts")
        self.tab_widget.setCornerWidget(corner_widget, Qt.TopRightCorner)
        self.tab_widget.setTabVisible(self.rail_deflection_tab_index, False)
        self.tab_widget.setTabVisible(self.rail_moment_tab_index, False)
        self.tab_widget.setTabVisible(self.transition_profile_tab_index, False)
        self.tab_widget.setTabVisible(self.transition_summary_tab_index, False)
        self.tab_widget.setTabVisible(self.all_charts_tab_index, False)
        self.tab_widget.setTabVisible(self.special_floating_slab_tab_index, False)
        self.tab_widget.setTabVisible(self.special_floating_slab_attenuation_tab_index, False)
        self.tab_widget.setTabVisible(self.special_summary_tab_index, False)
        self._set_dynamic_tabs_visible(False)
        # Ensure static startup keeps Stress visible.
        self._set_special_tabs_visible(False)
        self._build_chart_registry()
        for entry in self._chart_registry:
            entry.plot_panel.configure_custom_chart_action(
                callback=lambda panel=entry.plot_panel: self._open_custom_chart(panel),
                tooltip="Build a custom multi-axis chart from rendered series",
            )
        self._last_single_tab_index = self.deflection_tab_index
        self.tab_widget.currentChanged.connect(self._sync_chart_view_combo)

        return self.tab_widget

    def _show_help_dialog(self) -> None:
        if self.help_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("BOEF Help")
            dialog.resize(900, 700)
            layout = QVBoxLayout(dialog)
            browser = QTextBrowser()
            browser.setReadOnly(True)
            browser.setOpenExternalLinks(True)
            browser.setMarkdown(build_help_markdown())
            layout.addWidget(browser)
            buttons = QDialogButtonBox(QDialogButtonBox.Close)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            self.help_dialog = dialog
            self.help_browser = browser
        self.help_dialog.show()
        self.help_dialog.raise_()
        self.help_dialog.activateWindow()

    def _show_dynamic_chart_help(self, chart_id: str, title: str) -> None:
        if self.dynamic_help_dialog is None:
            dialog = QDialog(self)
            dialog.resize(780, 620)
            layout = QVBoxLayout(dialog)
            browser = QTextBrowser()
            browser.setReadOnly(True)
            browser.setOpenExternalLinks(True)
            layout.addWidget(browser)
            buttons = QDialogButtonBox(QDialogButtonBox.Close)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            self.dynamic_help_dialog = dialog
            self.dynamic_help_browser = browser
        if self.dynamic_help_dialog is None or self.dynamic_help_browser is None:
            return
        self.dynamic_help_dialog.setWindowTitle(f"{title} help")
        self.dynamic_help_browser.setMarkdown(build_dynamic_chart_help_markdown(chart_id))
        self.dynamic_help_dialog.show()
        self.dynamic_help_dialog.raise_()
        self.dynamic_help_dialog.activateWindow()

    def _open_custom_chart(self, panel: PlotPanel) -> None:
        source_series = self._collect_custom_chart_series_for_active_analysis()
        if not source_series:
            QMessageBox.information(
                self,
                "Custom chart",
                "Run an analysis and render the active chart set before building a custom chart.",
            )
            return
        dialog = CustomChartDialog(source_series=source_series, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        selections = dialog.selected()
        if not selections:
            return
        try:
            panel.render_custom_chart(
                selections=selections,
                title=self._custom_chart_title_for_active_analysis(),
                source_series=source_series,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Custom chart", str(exc))

    def _collect_custom_chart_series_for_active_analysis(self) -> list[RenderedSeries]:
        if not hasattr(self, "_chart_registry"):
            return []
        active_ids = self._custom_chart_active_chart_ids()
        collected: list[RenderedSeries] = []
        for entry in self._chart_registry:
            if entry.chart_id not in active_ids:
                continue
            if hasattr(self, "tab_widget") and not self.tab_widget.isTabVisible(entry.tab_index):
                continue
            series = entry.plot_panel.rendered_series()
            if not series:
                continue
            collected.extend(series)
        return collected

    def _custom_chart_active_chart_ids(self) -> set[str]:
        if not hasattr(self, "_chart_registry"):
            return set()
        analysis_type = self.analysis_type_combo.currentData()
        if analysis_type == AnalysisType.DYNAMIC:
            ids = {
                entry.chart_id
                for entry in self._chart_registry
                if entry.chart_id.startswith("dynamic_")
            }
            try:
                mode = self._coerce_dynamic_mode(self.dynamic_mode_combo.currentData())
            except ValueError:
                mode = DynamicMode.STEADY_STATE
            if mode == DynamicMode.TRANSITION:
                ids.add("transition_profile")
            ids.add("stress")
            return ids
        if analysis_type == AnalysisType.SPECIAL:
            ids = {
                entry.chart_id
                for entry in self._chart_registry
                if entry.chart_id.startswith("special_")
            }
            ids.add("stress")
            return ids
        ids = {
            entry.chart_id
            for entry in self._chart_registry
            if not entry.chart_id.startswith("dynamic_")
        }
        has_active_transition_input = hasattr(self, "transition_group") and self.transition_group.isChecked()
        if not (has_active_transition_input or self._last_transition_result is not None):
            ids.discard("transition_profile")
        return ids

    def _custom_chart_title_for_active_analysis(self) -> str:
        analysis_type = self.analysis_type_combo.currentData()
        if analysis_type == AnalysisType.DYNAMIC:
            try:
                mode = self._coerce_dynamic_mode(self.dynamic_mode_combo.currentData())
            except ValueError:
                mode = DynamicMode.STEADY_STATE
            if mode == DynamicMode.TRANSITION:
                return "Dynamic transition custom chart"
            return "Dynamic custom chart"
        if analysis_type == AnalysisType.SPECIAL:
            return "Special custom chart"
        if hasattr(self, "transition_group") and self.transition_group.isChecked():
            return "Transition custom chart"
        return "Custom chart"

    def _build_chart_registry(self) -> None:
        if not hasattr(self, "tab_widget"):
            return
        self._chart_registry = [
            ChartRegistryEntry(
                "deflection",
                "Deflection",
                self.deflection_tab_index,
                self.deflection_plot,
            ),
            ChartRegistryEntry("moment", "Moment", self.moment_tab_index, self.moment_plot),
            ChartRegistryEntry("shear", "Shear", self.shear_tab_index, self.shear_plot),
            ChartRegistryEntry("reaction", "Rail reaction", self.reaction_tab_index, self.reaction_plot),
            ChartRegistryEntry("sleeper", "Sleeper loads", self.sleeper_tab_index, self.sleeper_plot),
            ChartRegistryEntry("pressure", "Pressures", self.pressure_tab_index, self.pressure_plot),
            ChartRegistryEntry("stress", "Stress", self.stress_tab_index, self.stress_plot),
            ChartRegistryEntry(
                "rail_deflection",
                "Rail deflection (L/R)",
                self.rail_deflection_tab_index,
                self.rail_deflection_plot,
            ),
            ChartRegistryEntry(
                "rail_moment",
                "Rail moment (L/R)",
                self.rail_moment_tab_index,
                self.rail_moment_plot,
            ),
            ChartRegistryEntry(
                "transition_profile",
                "Transition k(x)",
                self.transition_profile_tab_index,
                self.transition_profile_plot,
            ),
            ChartRegistryEntry(
                "dynamic_deflection",
                "Dyn deflection",
                self.dynamic_deflection_tab_index,
                self.dynamic_deflection_plot,
            ),
            ChartRegistryEntry(
                "dynamic_moment",
                "Dyn moment",
                self.dynamic_moment_tab_index,
                self.dynamic_moment_plot,
            ),
            ChartRegistryEntry(
                "dynamic_shear",
                "Dyn shear",
                self.dynamic_shear_tab_index,
                self.dynamic_shear_plot,
            ),
            ChartRegistryEntry(
                "dynamic_reaction",
                "Dyn rail reaction",
                self.dynamic_reaction_tab_index,
                self.dynamic_reaction_plot,
            ),
            ChartRegistryEntry(
                "dynamic_damping",
                "Dyn damping",
                self.dynamic_damping_tab_index,
                self.dynamic_damping_plot,
            ),
            ChartRegistryEntry(
                "dynamic_time",
                "Dyn time history",
                self.dynamic_time_tab_index,
                self.dynamic_time_plot,
            ),
            ChartRegistryEntry(
                "dynamic_fft",
                "Dyn FFT",
                self.dynamic_fft_tab_index,
                self.dynamic_fft_plot,
            ),
            ChartRegistryEntry(
                "dynamic_psd",
                "Dyn PSD",
                self.dynamic_psd_tab_index,
                self.dynamic_psd_plot,
            ),
            ChartRegistryEntry(
                "dynamic_impedance",
                "Dyn impedance",
                self.dynamic_impedance_tab_index,
                self.dynamic_impedance_plot,
            ),
            ChartRegistryEntry(
                "special_floating_slab_transmissibility",
                "Slab transmissibility",
                self.special_floating_slab_tab_index,
                self.special_floating_slab_plot,
            ),
            ChartRegistryEntry(
                "special_floating_slab_attenuation",
                "Slab attenuation",
                self.special_floating_slab_attenuation_tab_index,
                self.special_floating_slab_attenuation_plot,
            ),
        ]
        for entry in self._chart_registry:
            entry.plot_panel.set_chart_context(chart_id=entry.chart_id, title=entry.title)
        self._chart_registry_by_id = {entry.chart_id: entry for entry in self._chart_registry}

    def _toggle_chart_view(self, *_: object) -> None:
        if self._chart_view_syncing:
            return
        if not hasattr(self, "chart_view_combo"):
            return
        if not hasattr(self, "all_charts_tab_index"):
            return
        view = self.chart_view_combo.currentData()
        if view == "all":
            self._apply_chart_view("all")
        else:
            self._apply_chart_view("single")

    def _apply_chart_view(self, view: str, *, target_tab: int | None = None) -> None:
        if not hasattr(self, "tab_widget"):
            return
        if view == "all":
            if self.tab_widget.currentIndex() != self.all_charts_tab_index:
                self._last_single_tab_index = self.tab_widget.currentIndex()
            self.tab_widget.setTabVisible(self.all_charts_tab_index, True)
            self.tab_widget.setCurrentIndex(self.all_charts_tab_index)
            self._schedule_chart_refresh()
            self._refresh_chart_grid()
            return
        if target_tab is None:
            target_tab = self._resolve_single_tab_index()
        self.tab_widget.setCurrentIndex(target_tab)
        self.tab_widget.setTabVisible(self.all_charts_tab_index, False)

    def _resolve_single_tab_index(self) -> int:
        if self._last_single_tab_index is not None and self.tab_widget.isTabVisible(
            self._last_single_tab_index
        ):
            return self._last_single_tab_index
        current_index = self.tab_widget.currentIndex()
        if (
            current_index != self.all_charts_tab_index
            and self.tab_widget.isTabVisible(current_index)
        ):
            return current_index
        for index in range(self.tab_widget.count()):
            if index == self.all_charts_tab_index:
                continue
            if self.tab_widget.isTabVisible(index):
                return index
        return self.deflection_tab_index

    def _set_chart_view_combo(self, view: str) -> None:
        if not hasattr(self, "chart_view_combo"):
            return
        index = 0 if view == "single" else 1
        self._chart_view_syncing = True
        try:
            self.chart_view_combo.setCurrentIndex(index)
        finally:
            self._chart_view_syncing = False

    def _sync_chart_view_combo(self, index: int) -> None:
        if self._chart_view_syncing:
            return
        if index == self.all_charts_tab_index:
            self._set_chart_view_combo("all")
            self._schedule_chart_refresh()
            self._refresh_chart_grid()
        else:
            self._last_single_tab_index = index
            self._set_chart_view_combo("single")
            self.tab_widget.setTabVisible(self.all_charts_tab_index, False)

    def _visible_chart_entries(self) -> list[ChartRegistryEntry]:
        if not hasattr(self, "tab_widget"):
            return []
        return [
            entry
            for entry in self._chart_registry
            if self.tab_widget.isTabVisible(entry.tab_index)
        ]

    def _clear_chart_grid(self) -> None:
        if not hasattr(self, "chart_grid_layout"):
            return
        while self.chart_grid_layout.count():
            item = self.chart_grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _refresh_chart_grid(self) -> None:
        if not hasattr(self, "chart_grid_layout"):
            return
        entries = self._visible_chart_entries()
        max_tiles = 8
        show_more = len(entries) > max_tiles
        if show_more:
            visible_entries = entries[: max_tiles - 1]
            hidden_entries = entries[max_tiles - 1 :]
        else:
            visible_entries = entries
            hidden_entries = []
        self._chart_hidden_entries = hidden_entries
        self._clear_chart_grid()

        if not visible_entries:
            tile = ChartTile("__empty__", "No charts")
            tile.set_placeholder("No charts available for this mode.")
            self.chart_grid_layout.addWidget(tile, 0, 0)
            return

        columns = 2
        row = 0
        col = 0
        for entry in visible_entries:
            tile = ChartTile(entry.chart_id, entry.title)
            tile.clicked.connect(self._handle_chart_tile_clicked)
            pixmap = self._get_cached_thumbnail(entry.chart_id)
            if pixmap is None:
                tile.set_placeholder("Not produced in this mode")
            else:
                tile.set_pixmap(pixmap)
            self.chart_grid_layout.addWidget(tile, row, col)
            col += 1
            if col >= columns:
                row += 1
                col = 0

        if hidden_entries:
            tile = ChartTile("__more__", "More charts")
            tile.clicked.connect(self._handle_chart_tile_clicked)
            tile.set_placeholder(f"{len(hidden_entries)} more chart(s)")
            self.chart_grid_layout.addWidget(tile, row, col)

    def _handle_chart_tile_clicked(self, chart_id: str) -> None:
        if chart_id == "__empty__":
            return
        if chart_id == "__more__":
            if not self._chart_hidden_entries:
                return
            chart_id = self._chart_hidden_entries[0].chart_id
        entry = self._chart_registry_by_id.get(chart_id)
        if entry is None:
            return
        self._last_single_tab_index = entry.tab_index
        self._set_chart_view_combo("single")
        self._apply_chart_view("single", target_tab=entry.tab_index)

    def _chart_cache_tokens(self, chart_id: str) -> tuple[int, int]:
        probe_token = self._chart_probe_token if chart_id in self._chart_probe_chart_ids else 0
        return (self._chart_result_token, probe_token)

    def _get_cached_thumbnail(self, chart_id: str) -> QPixmap | None:
        cached = self._chart_thumbnail_cache.get(chart_id)
        if cached is None:
            return None
        token, probe_token, pixmap = cached
        current_token, current_probe = self._chart_cache_tokens(chart_id)
        if token != current_token or probe_token != current_probe:
            return None
        return pixmap

    def _apply_thumbnail_axes_style(
        self,
        plot_panel: PlotPanel,
    ) -> tuple[
        list[tuple[object, float, float | None]],
        list[tuple[object, float | None, tuple[float, float, float, float], tuple[float, float, float, float]]],
        tuple[
            list[tuple[object, float, float | None]],
            object,
            float | None,
            tuple[float, float, float, float],
            tuple[float, float, float, float],
        ]
        | None,
    ]:
        text_state: list[tuple[object, float, float | None]] = []
        annotation_state: list[
            tuple[
                object,
                float | None,
                tuple[float, float, float, float],
                tuple[float, float, float, float],
            ]
        ] = []
        legend_state: tuple[
            list[tuple[object, float, float | None]],
            object,
            float | None,
            tuple[float, float, float, float],
            tuple[float, float, float, float],
        ] | None = None

        axes = plot_panel.axes
        text_items = [
            axes.title,
            axes.xaxis.label,
            axes.yaxis.label,
            *axes.get_xticklabels(),
            *axes.get_yticklabels(),
            *axes.texts,
        ]

        for text_item in text_items:
            font_size = float(text_item.get_fontsize())
            alpha = text_item.get_alpha()
            text_state.append((text_item, font_size, alpha))
            text_item.set_fontsize(max(CHART_TILE_MIN_FONT_SIZE, font_size * CHART_TILE_TEXT_SCALE))
            baseline_alpha = 1.0 if alpha is None else float(alpha)
            text_item.set_alpha(min(baseline_alpha, CHART_TILE_TEXT_ALPHA))

        for annotation in axes.texts:
            bbox_patch = annotation.get_bbox_patch()
            if bbox_patch is None:
                continue
            bbox_alpha = bbox_patch.get_alpha()
            facecolor = bbox_patch.get_facecolor()
            edgecolor = bbox_patch.get_edgecolor()
            annotation_state.append((bbox_patch, bbox_alpha, facecolor, edgecolor))
            bbox_patch.set_alpha(CHART_TILE_ANNOTATION_ALPHA)
            bbox_patch.set_facecolor((1.0, 1.0, 1.0, CHART_TILE_ANNOTATION_ALPHA))
            bbox_patch.set_edgecolor((0.0, 0.0, 0.0, 0.0))

        legend = axes.get_legend()
        if legend is not None:
            legend_text_state: list[tuple[object, float, float | None]] = []
            for legend_text in [*legend.get_texts(), legend.get_title()]:
                font_size = float(legend_text.get_fontsize())
                alpha = legend_text.get_alpha()
                legend_text_state.append((legend_text, font_size, alpha))
                legend_text.set_fontsize(max(CHART_TILE_MIN_FONT_SIZE, font_size * CHART_TILE_TEXT_SCALE))
                baseline_alpha = 1.0 if alpha is None else float(alpha)
                legend_text.set_alpha(min(baseline_alpha, CHART_TILE_TEXT_ALPHA))
            frame = legend.get_frame()
            legend_state = (
                legend_text_state,
                frame,
                frame.get_alpha(),
                frame.get_facecolor(),
                frame.get_edgecolor(),
            )
            frame.set_alpha(CHART_TILE_LEGEND_FRAME_ALPHA)
            frame.set_facecolor((1.0, 1.0, 1.0, CHART_TILE_LEGEND_FRAME_ALPHA))
            frame.set_edgecolor((0.0, 0.0, 0.0, CHART_TILE_LEGEND_EDGE_ALPHA))

        return text_state, annotation_state, legend_state

    def _restore_thumbnail_axes_style(
        self,
        style_state: tuple[
            list[tuple[object, float, float | None]],
            list[tuple[object, float | None, tuple[float, float, float, float], tuple[float, float, float, float]]],
            tuple[
                list[tuple[object, float, float | None]],
                object,
                float | None,
                tuple[float, float, float, float],
                tuple[float, float, float, float],
            ]
            | None,
        ],
    ) -> None:
        text_state, annotation_state, legend_state = style_state
        for text_item, font_size, alpha in text_state:
            text_item.set_fontsize(font_size)
            text_item.set_alpha(alpha)
        for bbox_patch, bbox_alpha, facecolor, edgecolor in annotation_state:
            bbox_patch.set_alpha(bbox_alpha)
            bbox_patch.set_facecolor(facecolor)
            bbox_patch.set_edgecolor(edgecolor)
        if legend_state is not None:
            legend_text_state, frame, frame_alpha, facecolor, edgecolor = legend_state
            for legend_text, font_size, alpha in legend_text_state:
                legend_text.set_fontsize(font_size)
                legend_text.set_alpha(alpha)
            frame.set_alpha(frame_alpha)
            frame.set_facecolor(facecolor)
            frame.set_edgecolor(edgecolor)

    def _capture_plot_thumbnail(self, plot_panel: PlotPanel) -> QPixmap | None:
        style_state = self._apply_thumbnail_axes_style(plot_panel)
        try:
            plot_panel.draw_now()
            png_buffer = io.BytesIO()
            plot_panel.figure.savefig(
                png_buffer,
                format="png",
                dpi=CHART_TILE_THUMBNAIL_DPI,
                facecolor=plot_panel.figure.get_facecolor(),
                edgecolor="none",
            )
            pixmap = QPixmap()
            if not pixmap.loadFromData(png_buffer.getvalue(), "PNG"):
                return None
            if pixmap.isNull():
                return None
            return pixmap
        except Exception:  # pragma: no cover - defensive for UI rendering
            LOGGER.debug("Failed to capture chart thumbnail.", exc_info=True)
            return None
        finally:
            self._restore_thumbnail_axes_style(style_state)
            plot_panel.request_draw_idle()

    def _update_chart_thumbnails(self, chart_ids: set[str] | None = None) -> None:
        if not self._chart_registry:
            return
        for entry in self._chart_registry:
            if chart_ids is not None and entry.chart_id not in chart_ids:
                continue
            if not self.tab_widget.isTabVisible(entry.tab_index):
                continue
            current_token, current_probe = self._chart_cache_tokens(entry.chart_id)
            cached = self._chart_thumbnail_cache.get(entry.chart_id)
            if cached is not None and cached[0] == current_token and cached[1] == current_probe:
                continue
            pixmap = self._capture_plot_thumbnail(entry.plot_panel)
            if pixmap is not None:
                self._chart_thumbnail_cache[entry.chart_id] = (
                    current_token,
                    current_probe,
                    pixmap,
                )

    def _refresh_chart_grid_if_visible(self) -> None:
        if not hasattr(self, "chart_view_combo"):
            return
        if self.chart_view_combo.currentData() == "all":
            self._schedule_chart_refresh()
            self._refresh_chart_grid()

    def _invalidate_chart_thumbnails(self, chart_ids: set[str]) -> None:
        for chart_id in chart_ids:
            self._chart_thumbnail_cache.pop(chart_id, None)

    def _schedule_chart_refresh(self, chart_ids: set[str] | None = None) -> None:
        if self._chart_refresh_timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._run_chart_refresh)
            self._chart_refresh_timer = timer
        if chart_ids is None:
            self._pending_chart_refresh_ids = None
        else:
            if self._pending_chart_refresh_ids is None:
                pass
            elif not self._pending_chart_refresh_ids:
                self._pending_chart_refresh_ids = set(chart_ids)
            else:
                self._pending_chart_refresh_ids.update(chart_ids)
        if not self._chart_refresh_timer.isActive():
            self._chart_refresh_timer.start(120)

    def _run_chart_refresh(self) -> None:
        chart_ids = self._pending_chart_refresh_ids
        self._pending_chart_refresh_ids = set()
        if chart_ids is None:
            self._update_chart_thumbnails(None)
        elif chart_ids:
            self._update_chart_thumbnails(chart_ids)
        if hasattr(self, "chart_view_combo") and self.chart_view_combo.currentData() == "all":
            self._refresh_chart_grid()

    def _mark_chart_results_updated(self) -> None:
        self._chart_result_token += 1
        self._schedule_chart_refresh()

    def _mark_chart_probe_updated(self) -> None:
        self._chart_probe_token += 1
        self._schedule_chart_refresh(set(self._chart_probe_chart_ids))

    def _clear_chart_thumbnails(self) -> None:
        self._chart_thumbnail_cache.clear()
        self._refresh_chart_grid_if_visible()

    def _refresh_material_combos(self) -> None:
        selected_ids = {
            "rail": getattr(self.rail_combo.currentData(), "id", None),
            "sleeper": getattr(self.sleeper_combo.currentData(), "id", None),
            "pad": getattr(self.pad_combo.currentData(), "id", None),
            "support": getattr(self.support_combo.currentData(), "id", None),
        }
        self._fill_combo(self.rail_combo, crud.list_rails(self.session))
        self._fill_combo(self.sleeper_combo, crud.list_sleepers(self.session))
        self._fill_combo(self.pad_combo, crud.list_pads(self.session))
        self._fill_combo(self.support_combo, crud.list_support_profiles(self.session))
        if selected_ids["rail"] is not None:
            self._select_combo_by_id(self.rail_combo, selected_ids["rail"])
        if selected_ids["sleeper"] is not None:
            self._select_combo_by_id(self.sleeper_combo, selected_ids["sleeper"])
        if selected_ids["pad"] is not None:
            self._select_combo_by_id(self.pad_combo, selected_ids["pad"])
        if selected_ids["support"] is not None:
            self._select_combo_by_id(self.support_combo, selected_ids["support"])
        if hasattr(self, "pad_stiffness_input"):
            self._sync_pad_inputs()

    def _refresh_dipped_joint_reference_combo(self) -> None:
        if not hasattr(self, "dipped_joint_reference_combo"):
            return
        self._fill_combo(
            self.dipped_joint_reference_combo,
            crud.list_dipped_joint_reference_sets(self.session),
        )


    def _apply_dipped_joint_reference(self) -> None:
        reference = self.dipped_joint_reference_combo.currentData()
        if reference is None:
            QMessageBox.information(
                self,
                "No Reference Set",
                "Please select a reference set from the dropdown.",
            )
            return

        missing_fields: list[str] = []
        applied_count = 0
        if reference.hertzian_contact_stiffness_n_per_m is not None:
            self.hertzian_stiffness_input.set_value(
                reference.hertzian_contact_stiffness_n_per_m / 1.0e6
            )
            applied_count += 1
        else:
            missing_fields.append("kₕ (Hertzian stiffness)")
        if reference.unsprung_mass_kg is not None:
            self.unsprung_mass_input.set_value(reference.unsprung_mass_kg)
            applied_count += 1
        else:
            missing_fields.append("mᵤ (unsprung mass)")
        if reference.track_mass_p1_kg is not None:
            self.track_mass_p1_input.set_value(reference.track_mass_p1_kg)
            applied_count += 1
        else:
            missing_fields.append("mᵀ₁ (effective track mass)")
        if reference.track_mass_p2_kg is not None:
            self.track_mass_p2_input.set_value(reference.track_mass_p2_kg)
            applied_count += 1
        else:
            missing_fields.append("mᵀ₂ (equivalent track mass)")
        if reference.track_stiffness_p2_n_per_m is not None:
            self.track_stiffness_p2_input.set_value(
                reference.track_stiffness_p2_n_per_m / 1.0e6
            )
            applied_count += 1
        else:
            missing_fields.append("kᵀ₂ (equivalent track stiffness)")
        if reference.track_damping_p2_n_s_per_m is not None:
            self.track_damping_p2_input.set_value(
                reference.track_damping_p2_n_s_per_m / 1.0e3
            )
            applied_count += 1
        else:
            missing_fields.append("cᵀ (equivalent track damping)")

        message_parts = [f"Applied {applied_count} parameters from '{reference.name}'."]
        if missing_fields:
            message_parts.append(
                "\nThe following parameters were not provided by this reference set "
                "and retain their current values:\n"
                + "\n".join(f"  • {field}" for field in missing_fields)
            )

        QMessageBox.information(
            self,
            "Reference Set Applied",
            "\n".join(message_parts),
        )

    def _validate_dipped_joint_inputs(self) -> bool:
        """Validate dipped joint inputs before running analysis."""
        errors: list[str] = []
        if self.dip_angle_input.value() <= 0:
            errors.append("Dip angle (2α) must be positive")
        if self.dip_angle_input.value() > 100.0:
            errors.append("Dip angle (2α) must be 100 mrad or less")
        if self.load_magnitude_input.value() <= 0:
            errors.append("Static wheel load (P₀) must be positive")
        if self.hertzian_stiffness_input.value() <= 0:
            errors.append("Hertzian stiffness (kₕ) must be positive")
        if self.hertzian_stiffness_input.value() > 5_000.0:
            errors.append("Hertzian stiffness (kₕ) must be 5000 MN/m or less")
        if self.unsprung_mass_input.value() <= 0:
            errors.append("Unsprung mass (mᵤ) must be positive")
        if self.unsprung_mass_input.value() > 2_000.0:
            errors.append("Unsprung mass (mᵤ) must be 2000 kg or less")
        if self.track_mass_p1_input.value() <= 0:
            errors.append("Effective track mass (mᵀ₁) must be positive")
        if self.track_mass_p1_input.value() > 2_000.0:
            errors.append("Effective track mass (mᵀ₁) must be 2000 kg or less")
        if self.track_mass_p2_input.value() <= 0:
            errors.append("Equivalent track mass (mᵀ₂) must be positive")
        if self.track_mass_p2_input.value() > 2_000.0:
            errors.append("Equivalent track mass (mᵀ₂) must be 2000 kg or less")
        if self.track_stiffness_p2_input.value() <= 0:
            errors.append("Equivalent track stiffness (kᵀ₂) must be positive")
        if self.track_stiffness_p2_input.value() > 500.0:
            errors.append("Equivalent track stiffness (kᵀ₂) must be 500 MN/m or less")
        if self.track_damping_p2_input.value() < 0:
            errors.append("Equivalent track damping (cᵀ) cannot be negative")
        if self.track_damping_p2_input.value() > 500.0:
            errors.append("Equivalent track damping (cᵀ) must be 500 kN·s/m or less")
        if self.speed_input.value() < 0:
            errors.append("Speed cannot be negative")
        if errors:
            QMessageBox.warning(
                self,
                "Invalid Dipped Joint Inputs",
                "Please correct the following:\n\n" + "\n".join(f"• {error}" for error in errors),
            )
            return False
        return True

    def _load_tensile_strengths(self) -> list[float]:
        strengths = [
            value
            for value in self.session.scalars(
                select(RailAdmissibleStress.tensile_strength_mpa).distinct()
            ).all()
            if value is not None
        ]
        strengths = sorted(set(strengths))
        return strengths if strengths else [700.0, 900.0]

    def _refresh_load_cases(self) -> None:
        self._fill_combo(self.load_case_combo, crud.list_load_cases(self.session))
        self._sync_load_case()

    def _fill_combo(self, combo: QComboBox, items: Sequence) -> None:
        combo.clear()
        for item in items:
            combo.addItem(item.name, item)

    def _sync_load_case(self) -> None:
        load_case = self.load_case_combo.currentData()
        if load_case is None:
            return
        self.load_magnitude_input.set_value(n_to_kn(load_case.load_newtons))
        if hasattr(self, "asymmetric_load_checkbox") and not self.asymmetric_load_checkbox.isChecked():
            self.right_load_magnitude_input.set_value(n_to_kn(load_case.load_newtons))

    def _sync_pad_inputs(self) -> None:
        if not hasattr(self, "pad_stiffness_input"):
            return
        pad = self.pad_combo.currentData()
        if pad is None:
            return
        self.pad_stiffness_input.set_value(n_to_kn(pad.stiffness_newtons_per_meter))

    def _sync_right_load_inputs(self) -> None:
        if not self.advanced_solver_checkbox.isChecked():
            return
        if not self.two_rail_checkbox.isChecked():
            return
        if self.asymmetric_load_checkbox.isChecked():
            return
        self.right_load_magnitude_input.set_value(self.load_magnitude_input.value())
        self.right_load_position_input.set_value(self.load_position_input.value())

    def _log_analysis_inputs(
        self,
        *,
        rail: Rail,
        sleeper: Sleeper,
        pad: Pad | None,
        support: SupportProfile,
        load_case: LoadCase | None,
        analysis_inputs: AnalysisInputs,
        config: AnalysisConfig,
        load_source: dict[str, object] | None = None,
    ) -> None:
        resolved_load_source = (
            self._copy_load_source_metadata(load_source)
            if load_source is not None
            else self._current_load_source_metadata()
        )
        design_inputs = analysis_inputs.design_inputs
        gui_values = {
            "rail_name": rail.name,
            "rail_elastic_modulus_mpa": pa_to_mpa(rail.elastic_modulus_pa),
            "rail_moment_inertia_mm4": m4_to_mm4(rail.moment_inertia_m4),
            "rail_section_modulus_mm3": m3_to_mm3(rail.section_modulus_m3),
            "rail_mass_kg_per_m": rail.mass_kg_per_m,
            "rail_height_mm": rail.height_mm,
            "sleeper_name": sleeper.name,
            "sleeper_length_mm": m_to_mm(sleeper.length_m),
            "sleeper_width_mm": m_to_mm(sleeper.width_m),
            "support_name": support.name,
            "support_modulus_mn_per_m2": n_per_m2_to_mn_per_m2(support.foundation_modulus_n_per_m2),
            "pad_name": pad.name if pad else None,
            "pad_stiffness_kn_per_m": n_to_kn(pad.stiffness_newtons_per_meter) if pad else None,
            "load_case_name": load_case.name if load_case else None,
            "load_case_kn": n_to_kn(load_case.load_newtons) if load_case else None,
            "load_input_kn": self.load_magnitude_input.value(),
            "load_position_mm": self.load_position_input.value(),
            "sleeper_spacing_mm": self.sleeper_spacing_input.value(),
            "ballast_thickness_mm": self.ballast_thickness_input.value(),
            "pad_stiffness_input_kn_per_m": self.pad_stiffness_input.value(),
            "foundation_damping_model": str(self.foundation_damping_model_static_combo.currentData()),
            "railpad_loss_factor": self.railpad_loss_factor_input.value(),
            "trackbed_loss_factor": self.trackbed_loss_factor_input.value(),
            "pad_loss_factor": self.pad_loss_factor_input.value(),
            "design_track_quality": self.track_quality_combo.currentData(),
            "design_probability_factor": self.probability_combo.currentData(),
            "design_speed_kmh": self.design_speed_input.value(),
            "design_wheel_radius_mm": self.wheel_radius_input.value(),
            "design_curve_enabled": self.curve_checkbox.isChecked(),
            "design_tensile_strength_mpa": self.tensile_strength_combo.currentData(),
            "nonuniform_profile_enabled": self.nonuniform_profile_checkbox.isChecked(),
            "foundation_profile_type": str(self.profile_type_combo.currentData()),
            "foundation_profile_k1_mn_per_m2": self.profile_k1_input.value(),
            "foundation_profile_k2_mn_per_m2": self.profile_k2_input.value(),
            "foundation_profile_x_start_mm": self.profile_x_start_input.value(),
            "foundation_profile_x_end_mm": self.profile_x_end_input.value(),
            "discrete_supports_enabled": self.discrete_supports_checkbox.isChecked(),
            "nodes_between_sleepers": int(self.nodes_between_sleepers_input.value()),
            "two_rail_enabled": self.two_rail_checkbox.isChecked(),
            "asymmetric_load_enabled": self.asymmetric_load_checkbox.isChecked(),
            "right_load_kn": self.right_load_magnitude_input.value(),
            "right_load_position_mm": self.right_load_position_input.value(),
            "pasternak_enabled": self.pasternak_checkbox.isChecked(),
            "pasternak_shear_kn": self.pasternak_input.value(),
            "load_source": resolved_load_source,
        }
        solver_values = {
            "elastic_modulus_pa": config.elastic_modulus_pa,
            "moment_inertia_m4": config.moment_inertia_m4,
            "section_modulus_m3": config.section_modulus_m3,
            "mass_kg_per_m": rail.mass_kg_per_m,
            "foundation_modulus_n_per_m2": config.foundation_modulus_n_per_m2,
            "foundation_model": str(config.foundation_model),
            "foundation_damping_model": str(config.foundation_damping_model),
            "railpad_stiffness_n_per_m": config.railpad_stiffness_n_per_m,
            "railpad_damping_n_s_per_m": config.railpad_damping_n_s_per_m,
            "railpad_loss_factor": config.railpad_loss_factor,
            "trackbed_stiffness_n_per_m": config.trackbed_stiffness_n_per_m,
            "trackbed_damping_n_s_per_m": config.trackbed_damping_n_s_per_m,
            "trackbed_loss_factor": config.trackbed_loss_factor,
            "sleeper_mass_kg": config.sleeper_mass_kg,
            "beam_theory": str(config.beam_theory),
            "shear_modulus_pa": config.shear_modulus_pa,
            "shear_correction_factor": config.shear_correction_factor,
            "rail_area_m2": config.rail_area_m2,
            "foundation_profile_type": str(config.foundation_profile_type),
            "foundation_profile_k1_n_per_m2": config.foundation_profile_k1_n_per_m2,
            "foundation_profile_k2_n_per_m2": config.foundation_profile_k2_n_per_m2,
            "foundation_profile_x_start_m": config.foundation_profile_x_start_m,
            "foundation_profile_x_end_m": config.foundation_profile_x_end_m,
            "load_count": len(analysis_inputs.loads),
            "sleeper_spacing_m": config.sleeper_spacing_m,
            "sleeper_length_m": config.sleeper_length_m,
            "sleeper_width_m": config.sleeper_width_m,
            "pad_stiffness_n_per_m": config.pad_stiffness_n_per_m,
            "pad_damping_n_s_per_m": config.pad_damping_n_s_per_m,
            "pad_loss_factor": config.pad_loss_factor,
            "discrete_support_stiffness_n_per_m": config.discrete_support_stiffness_n_per_m,
            "use_discrete_supports": config.use_discrete_supports,
            "nodes_between_sleepers": config.nodes_between_sleepers,
            "use_two_rail": config.use_two_rail,
            "coupling_stiffness_n_per_m": config.coupling_stiffness_n_per_m,
            "right_load_count": len(config.right_loads) if config.right_loads else 0,
            "pasternak_shear_n": config.pasternak_shear_n,
            "section_modulus_head_m3": config.section_modulus_head_m3,
            "section_modulus_foot_m3": config.section_modulus_foot_m3,
            "area_m2": config.area_m2,
            "design_speed_kmh": design_inputs.speed_kmh if design_inputs else None,
            "design_probability_factor": design_inputs.probability_factor if design_inputs else None,
            "design_track_factor": design_inputs.track_factor if design_inputs else None,
            "design_wheel_radius_mm": design_inputs.wheel_radius_mm if design_inputs else None,
            "design_tensile_strength_mpa": design_inputs.tensile_strength_mpa if design_inputs else None,
            "design_curve_enabled": design_inputs.on_curve if design_inputs else None,
        }
        message = (
            "Analysis input verification\n"
            f"GUI values: {gui_values}\n"
            f"Solver values (SI): {solver_values}\n"
        )
        if LOGGER.hasHandlers():
            LOGGER.info(message)
        else:
            print(message)

    def _log_envelope_inputs(
        self,
        *,
        rail: Rail,
        sleeper: Sleeper,
        pad: Pad | None,
        support: SupportProfile,
        load_case: LoadCase | None,
        config: EnvelopeConfig,
        load_source: dict[str, object] | None = None,
    ) -> None:
        resolved_load_source = (
            self._copy_load_source_metadata(load_source)
            if load_source is not None
            else self._current_load_source_metadata()
        )
        gui_values = {
            "rail_name": rail.name,
            "rail_elastic_modulus_mpa": pa_to_mpa(rail.elastic_modulus_pa),
            "rail_moment_inertia_mm4": m4_to_mm4(rail.moment_inertia_m4),
            "rail_section_modulus_mm3": m3_to_mm3(rail.section_modulus_m3),
            "sleeper_name": sleeper.name,
            "sleeper_length_mm": m_to_mm(sleeper.length_m),
            "sleeper_width_mm": m_to_mm(sleeper.width_m),
            "support_name": support.name,
            "support_modulus_mn_per_m2": n_per_m2_to_mn_per_m2(support.foundation_modulus_n_per_m2),
            "pad_name": pad.name if pad else None,
            "load_case_name": load_case.name if load_case else None,
            "static_mode": str(self.static_mode_combo.currentData()),
            "envelope_reference_m": self.envelope_reference_input.value(),
            "envelope_x_ref_start_m": self.envelope_range_start_input.value(),
            "envelope_x_ref_end_m": self.envelope_range_end_input.value(),
            "envelope_dx_ref_m": self.envelope_step_input.value(),
            "envelope_x_domain_start_m": self.envelope_domain_start_input.value(),
            "envelope_x_domain_end_m": self.envelope_domain_end_input.value(),
            "bearing_width_m": self.envelope_bearing_width_input.value(),
            "bearing_length_m": self.envelope_bearing_length_input.value(),
            "bearing_area_m2": self.envelope_bearing_width_input.value()
            * self.envelope_bearing_length_input.value(),
            "depths_m": self.envelope_depths_input.text(),
            "ballast_thickness_mm": self.ballast_thickness_input.value(),
            "rail_count": self.envelope_rail_count_combo.currentData(),
            "load_offsets_m": [load.position_m for load in config.analysis_config.loads],
            "load_source": resolved_load_source,
        }
        solver_values = {
            "mode": str(config.mode),
            "x_ref_start_m": config.x_ref_start_m,
            "x_ref_end_m": config.x_ref_end_m,
            "x_ref_step_m": config.x_ref_step_m,
            "x_domain_m": config.x_domain_m,
            "bearing_width_m": config.bearing_width_m,
            "bearing_length_m": config.bearing_length_m,
            "depth_m": list(config.depth_m),
            "rail_count": config.rail_count,
            "foundation_modulus_n_per_m2": config.analysis_config.foundation_modulus_n_per_m2,
            "elastic_modulus_pa": config.analysis_config.elastic_modulus_pa,
            "moment_inertia_m4": config.analysis_config.moment_inertia_m4,
            "section_modulus_m3": config.analysis_config.section_modulus_m3,
            "sleeper_spacing_m": config.analysis_config.sleeper_spacing_m,
            "load_count": len(config.analysis_config.loads),
        }
        message = (
            "Envelope input verification\n"
            f"GUI values: {gui_values}\n"
            f"Solver values (SI): {solver_values}\n"
        )
        if LOGGER.hasHandlers():
            LOGGER.info(message)
        else:
            print(message)

    def _log_transition_inputs(
        self,
        *,
        rail: Rail,
        sleeper: Sleeper,
        pad: Pad | None,
        support: SupportProfile,
        load_case: LoadCase | None,
        context: TransitionContext,
        load_source: dict[str, object] | None = None,
    ) -> None:
        resolved_load_source = (
            self._copy_load_source_metadata(load_source)
            if load_source is not None
            else self._current_load_source_metadata()
        )
        gui_values = {
            "rail_name": rail.name,
            "rail_elastic_modulus_mpa": pa_to_mpa(rail.elastic_modulus_pa),
            "rail_moment_inertia_mm4": m4_to_mm4(rail.moment_inertia_m4),
            "rail_section_modulus_mm3": m3_to_mm3(rail.section_modulus_m3),
            "sleeper_name": sleeper.name,
            "sleeper_length_mm": m_to_mm(sleeper.length_m),
            "sleeper_width_mm": m_to_mm(sleeper.width_m),
            "support_name": support.name,
            "support_modulus_mn_per_m2": n_per_m2_to_mn_per_m2(support.foundation_modulus_n_per_m2),
            "pad_name": pad.name if pad else None,
            "load_case_name": load_case.name if load_case else None,
            "template": self.transition_template_combo.currentText(),
            "preset": self.transition_preset_combo.currentText(),
            "run_mode": str(self.transition_run_mode_combo.currentData()),
            "profile_type": str(self.transition_profile_combo.currentData()),
            "k1_mn_per_m2": self.transition_k1_input.value(),
            "k2_mn_per_m2": self.transition_k2_input.value(),
            "transition_length_m": self.transition_length_input.value(),
            "segment_length_m": self.transition_segment_length_input.value(),
            "reference_m": self.transition_reference_input.value(),
            "range_start_m": self.transition_range_start_input.value(),
            "range_end_m": self.transition_range_end_input.value(),
            "range_step_m": self.transition_step_input.value(),
            "domain_start_m": self.transition_domain_start_input.value(),
            "domain_end_m": self.transition_domain_end_input.value(),
            "ballast_thickness_mm": self.ballast_thickness_input.value(),
            "load_offsets_m": [load.position_m for load in context.analysis_config.loads],
            "load_source": resolved_load_source,
        }
        solver_values = {
            "analysis_mode": str(context.analysis_mode),
            "profile_type": str(context.profile_type),
            "domain_m": context.domain_m,
            "k1_n_per_m2": context.k1_n_per_m2,
            "k2_n_per_m2": context.k2_n_per_m2,
            "transition_length_m": context.transition_length_m,
            "segment_length_m": context.segment_length_m,
            "k_profile_len": len(context.k_profile_n_per_m2) if context.k_profile_n_per_m2 else None,
            "foundation_model": str(context.analysis_config.foundation_model),
            "use_two_rail": context.analysis_config.use_two_rail,
            "use_discrete_supports": context.analysis_config.use_discrete_supports,
        }
        message = (
            "Transition input verification\n"
            f"GUI values: {gui_values}\n"
            f"Solver values (SI): {solver_values}\n"
        )
        if LOGGER.hasHandlers():
            LOGGER.info(message)
        else:
            print(message)

    def _write_analysis_snapshot(
        self,
        analysis_inputs: AnalysisInputs,
        config: AnalysisConfig,
        *,
        load_source: dict[str, object] | None = None,
    ) -> None:
        def _json_default(value: object) -> str:
            return str(value)

        resolved_load_source = (
            self._copy_load_source_metadata(load_source)
            if load_source is not None
            else self._current_load_source_metadata()
        )
        payload = {
            "analysis_inputs": asdict(analysis_inputs),
            "analysis_config": asdict(config),
            "load_source": resolved_load_source,
        }
        data_dir = Path.home() / ".boef"
        data_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = data_dir / "analysis_inputs_snapshot.json"
        snapshot_path.write_text(
            json.dumps(payload, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def _write_envelope_snapshot(
        self,
        config: EnvelopeConfig,
        *,
        load_source: dict[str, object] | None = None,
    ) -> None:
        def _json_default(value: object) -> str:
            return str(value)

        resolved_load_source = (
            self._copy_load_source_metadata(load_source)
            if load_source is not None
            else self._current_load_source_metadata()
        )
        payload = {
            "envelope_config": asdict(config),
            "load_source": resolved_load_source,
        }
        payload = self._extend_envelope_metadata_payload(
            payload,
            result=self._last_envelope_result,
            load_source=resolved_load_source,
        )
        data_dir = Path.home() / ".boef"
        data_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = data_dir / "envelope_inputs_snapshot.json"
        snapshot_path.write_text(
            json.dumps(payload, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def _write_transition_snapshot(
        self,
        context: TransitionContext,
        analysis_inputs: AnalysisInputs,
        envelope_config: EnvelopeConfig | None,
        *,
        load_source: dict[str, object] | None = None,
    ) -> None:
        def _json_default(value: object) -> str:
            return str(value)

        resolved_load_source = (
            self._copy_load_source_metadata(load_source)
            if load_source is not None
            else self._current_load_source_metadata()
        )
        payload = {
            "transition_context": asdict(context),
            "analysis_inputs": asdict(analysis_inputs),
            "envelope_config": asdict(envelope_config) if envelope_config is not None else None,
            "load_source": resolved_load_source,
        }
        payload = self._extend_envelope_metadata_payload(
            payload,
            result=self._last_envelope_result if envelope_config is not None else None,
            load_source=resolved_load_source,
        )
        data_dir = Path.home() / ".boef"
        data_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = data_dir / "transition_inputs_snapshot.json"
        snapshot_path.write_text(
            json.dumps(payload, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def _log_dynamic_inputs(
        self,
        *,
        rail: Rail,
        sleeper: Sleeper,
        pad: Pad | None,
        support: SupportProfile,
        load_case: LoadCase | None,
        config: DynamicConfig,
        load_source: dict[str, object] | None = None,
    ) -> None:
        resolved_load_source = (
            self._copy_load_source_metadata(load_source)
            if load_source is not None
            else self._current_load_source_metadata()
        )
        gui_values = {
            "rail_name": rail.name,
            "rail_elastic_modulus_mpa": pa_to_mpa(rail.elastic_modulus_pa),
            "rail_moment_inertia_mm4": m4_to_mm4(rail.moment_inertia_m4),
            "rail_section_modulus_mm3": m3_to_mm3(rail.section_modulus_m3),
            "rail_mass_kg_per_m": rail.mass_kg_per_m,
            "sleeper_name": sleeper.name,
            "support_name": support.name,
            "support_modulus_mn_per_m2": n_per_m2_to_mn_per_m2(support.foundation_modulus_n_per_m2),
            "pad_name": pad.name if pad else None,
            "pad_stiffness_kn_per_m": n_to_kn(pad.stiffness_newtons_per_meter) if pad else None,
            "load_case_name": load_case.name if load_case else None,
            "load_case_kn": n_to_kn(load_case.load_newtons) if load_case else None,
            "speed_m_per_s": self.speed_input.value(),
            "domain_length_m": self.domain_length_input.value(),
            "spatial_step_m": self.spatial_step_input.value(),
            "time_window_s": self.time_window_input.value(),
            "sample_rate_hz": self.sample_rate_input.value(),
            "foundation_damping_model": str(self.foundation_damping_model_combo.currentData()),
            "foundation_damping_loss_factor": self.damping_loss_factor_input.value(),
            "excitation_mode": str(self.dynamic_excitation_mode_combo.currentData()),
            "boundary_mode": str(self.dynamic_boundary_mode_combo.currentData()),
            "irregularity_mode": str(self.irregularity_mode_combo.currentData()),
            "load_source": resolved_load_source,
        }
        solver_values = {
            "elastic_modulus_pa": config.elastic_modulus_pa,
            "moment_inertia_m4": config.moment_inertia_m4,
            "section_modulus_m3": config.section_modulus_m3,
            "mass_kg_per_m": config.mass_kg_per_m,
            "foundation_modulus_n_per_m2": config.foundation_modulus_n_per_m2,
            "foundation_damping_n_s_per_m2": config.foundation_damping_n_s_per_m2,
            "foundation_damping_model": str(config.foundation_damping_model),
            "foundation_loss_factor": config.foundation_loss_factor,
            "speed_m_per_s": config.speed_m_per_s,
            "domain_length_m": config.domain_length_m,
            "spatial_step_m": config.spatial_step_m,
            "time_window_s": config.time_window_s,
            "sample_rate_hz": config.sample_rate_hz,
            "excitation_mode": str(config.excitation_mode),
            "boundary_mode": str(config.boundary_mode),
            "oscillator_unsprung_mass_kg": config.oscillator_unsprung_mass_kg,
            "oscillator_suspension_stiffness_n_per_m": config.oscillator_suspension_stiffness_n_per_m,
            "oscillator_suspension_damping_n_s_per_m": config.oscillator_suspension_damping_n_s_per_m,
            "has_irregularity_input": config.irregularity_input is not None,
        }
        message = (
            "Dynamic input verification\n"
            f"GUI values: {gui_values}\n"
            f"Solver values (SI): {solver_values}\n"
        )
        if LOGGER.hasHandlers():
            LOGGER.info(message)
        else:
            print(message)

    def _open_rail_dialog(self) -> None:
        dialog = MaterialDialog(
            self.session,
            title="Rails",
            list_items=crud.list_rails,
            create_item=crud.create_rail,
            update_item=crud.update_rail,
            delete_item=crud.delete_rail,
            fields=[
                MaterialField("elastic_modulus_pa", "Elastic modulus", "MPa", mpa_to_pa, pa_to_mpa, decimals=2),
                MaterialField("moment_inertia_m4", "Moment of inertia", "mm⁴", mm4_to_m4, m4_to_mm4, decimals=2),
                MaterialField("section_modulus_m3", "Section modulus", "mm³", mm3_to_m3, m3_to_mm3, decimals=2),
                MaterialField("mass_kg_per_m", "Mass", "kg/m", lambda x: x, lambda x: x, decimals=2),
                MaterialField(
                    "height_mm",
                    "Height",
                    "mm",
                    lambda x: x,
                    lambda x: x,
                    decimals=1,
                    optional=True,
                ),
                MaterialField(
                    "head_width_mm",
                    "Head width",
                    "mm",
                    lambda x: x,
                    lambda x: x,
                    decimals=1,
                    optional=True,
                ),
                MaterialField(
                    "foot_width_mm",
                    "Foot width",
                    "mm",
                    lambda x: x,
                    lambda x: x,
                    decimals=1,
                    optional=True,
                ),
                MaterialField(
                    "head_height_mm",
                    "Head height",
                    "mm",
                    lambda x: x,
                    lambda x: x,
                    decimals=1,
                    optional=True,
                ),
                MaterialField(
                    "web_thickness_mm",
                    "Web thickness",
                    "mm",
                    lambda x: x,
                    lambda x: x,
                    decimals=2,
                    optional=True,
                ),
                MaterialField(
                    "area_cm2",
                    "Area",
                    "cm²",
                    lambda x: x,
                    lambda x: x,
                    decimals=2,
                    optional=True,
                ),
                MaterialField(
                    "moment_inertia_z_m4",
                    "Moment of inertia I_z",
                    "cm⁴",
                    cm4_to_m4,
                    m4_to_cm4,
                    decimals=2,
                    optional=True,
                ),
                MaterialField(
                    "section_modulus_head_m3",
                    "Section modulus W_yh",
                    "cm³",
                    cm3_to_m3,
                    m3_to_cm3,
                    decimals=2,
                    optional=True,
                ),
                MaterialField(
                    "section_modulus_foot_m3",
                    "Section modulus W_yf",
                    "cm³",
                    cm3_to_m3,
                    m3_to_cm3,
                    decimals=2,
                    optional=True,
                ),
                MaterialField(
                    "section_modulus_z_m3",
                    "Section modulus W_z",
                    "cm³",
                    cm3_to_m3,
                    m3_to_cm3,
                    decimals=2,
                    optional=True,
                ),
            ],
        )
        dialog.exec()
        self._refresh_material_combos()

    def _open_sleeper_dialog(self) -> None:
        dialog = MaterialDialog(
            self.session,
            title="Sleepers",
            list_items=crud.list_sleepers,
            create_item=crud.create_sleeper,
            update_item=crud.update_sleeper,
            delete_item=crud.delete_sleeper,
            fields=[
                MaterialField("elastic_modulus_pa", "Elastic modulus", "MPa", mpa_to_pa, pa_to_mpa, decimals=2),
                MaterialField("length_m", "Length", "mm", mm_to_m, m_to_mm, decimals=1),
                MaterialField("width_m", "Width", "mm", mm_to_m, m_to_mm, decimals=1),
                MaterialField("height_m", "Height", "mm", mm_to_m, m_to_mm, decimals=1),
                MaterialField("mass_kg", "Mass", "kg", lambda x: x, lambda x: x, decimals=2),
            ],
        )
        dialog.exec()
        self._refresh_material_combos()

    def _open_pad_dialog(self) -> None:
        dialog = MaterialDialog(
            self.session,
            title="Pads",
            list_items=crud.list_pads,
            create_item=crud.create_pad,
            update_item=crud.update_pad,
            delete_item=crud.delete_pad,
            fields=[
                MaterialField(
                    "stiffness_newtons_per_meter",
                    "Stiffness",
                    "kN/m",
                    kn_to_n,
                    n_to_kn,
                    decimals=2,
                ),
                MaterialField("thickness_m", "Thickness", "mm", mm_to_m, m_to_mm, decimals=2),
            ],
        )
        dialog.exec()
        self._refresh_material_combos()

    def _open_support_dialog(self) -> None:
        dialog = MaterialDialog(
            self.session,
            title="Support profiles",
            list_items=crud.list_support_profiles,
            create_item=crud.create_support_profile,
            update_item=crud.update_support_profile,
            delete_item=crud.delete_support_profile,
            fields=[
                MaterialField(
                    "foundation_modulus_n_per_m2",
                    "Foundation modulus",
                    "MN/m²",
                    mn_per_m2_to_n_per_m2,
                    n_per_m2_to_mn_per_m2,
                    decimals=2,
                ),
            ],
        )
        dialog.exec()
        self._refresh_material_combos()

    def _open_project_dialog(self) -> None:
        dialog = ProjectDialog(self.session)
        dialog.exec()
        self._refresh_project_tree()

    def _open_load_case_dialog(self) -> None:
        dialog = LoadCaseDialog(self.session)
        dialog.exec()
        self._refresh_load_cases()

    def _open_track_config_dialog(self) -> None:
        dialog = TrackConfigDialog(self.session)
        dialog.exec()
        self._refresh_project_tree()

    def _collect_static_loads(self) -> list[PointLoad]:
        return self._collect_analysis_loads()

    def _collect_analysis_loads(self) -> list[PointLoad]:
        if hasattr(self, "as5100_loads_checkbox") and self.as5100_loads_checkbox.isChecked():
            return self._collect_as5100_loads()
        if self.train_loads_checkbox.isChecked():
            return self._collect_train_loads()
        if self.several_loads_checkbox.isChecked():
            raw_loads = self.wheel_loads_widget.loads()
            if not raw_loads:
                raise ValueError("Add at least one wheel load before running analysis.")
            loads: list[PointLoad] = []
            for load_kn, position_m in raw_loads:
                if load_kn <= 0:
                    raise ValueError("Each wheel load must be > 0 kN.")
                loads.append(
                    PointLoad(
                        position_m=position_m,
                        load_newtons=kn_to_n(load_kn),
                    )
                )
            return loads
        load_n = kn_to_n(self.load_magnitude_input.value())
        load_position_m = mm_to_m(self.load_position_input.value())
        return [PointLoad(position_m=load_position_m, load_newtons=load_n)]

    def _collect_train_loads(self) -> list[PointLoad]:
        config = TrainLoadConfig(
            axle_load_n=kn_to_n(self.train_axle_load_input.value()),
            bogie_count=int(self.train_bogie_count_input.value()),
            bogie_spacing_m=mm_to_m(self.train_bogie_spacing_input.value()),
            axles_per_bogie=int(self.train_axles_per_bogie_input.value()),
            axle_spacing_m=mm_to_m(self.train_axle_spacing_input.value()),
            reference_bogie_center_m=mm_to_m(self.train_reference_input.value()),
        )
        return build_train_loads(config)

    def _as5100_load_config(self) -> AS5100RailLoadConfig:
        model = self.as5100_model_combo.currentData()
        if model not in (AS5100_MODEL_300LA, AS5100_MODEL_150LA):
            model = AS5100_MODEL_300LA
        return AS5100RailLoadConfig(
            model=model,
            group_count=int(self.as5100_group_count_input.value()),
            group_spacing_m=self.as5100_group_spacing_input.value(),
            reference_position_m=self.as5100_reference_input.value(),
        )

    def _collect_as5100_loads(self) -> list[PointLoad]:
        return build_as5100_rail_loads(self._as5100_load_config())

    def _as5100_arrangement_mode(self) -> AS5100ArrangementMode:
        if not hasattr(self, "as5100_arrangement_mode_combo"):
            return AS5100ArrangementMode.FIXED_SELECTED
        mode = self.as5100_arrangement_mode_combo.currentData()
        return mode if isinstance(mode, AS5100ArrangementMode) else AS5100ArrangementMode.FIXED_SELECTED

    def _is_as5100_governing_sweep_selected(self) -> bool:
        if not hasattr(self, "as5100_loads_checkbox") or not self.as5100_loads_checkbox.isChecked():
            return False
        return self._as5100_arrangement_mode() == AS5100ArrangementMode.GOVERNING_SWEEP

    def _should_run_as5100_governing_sweep(self) -> bool:
        if not self._is_as5100_governing_sweep_selected():
            return False
        if not hasattr(self, "static_mode_combo"):
            return False
        static_mode = self.static_mode_combo.currentData()
        if static_mode not in (StaticMode.ENVELOPE_CLOSED_FORM, StaticMode.ENVELOPE_NUMERICAL):
            return False
        return True

    def _should_run_as5100_transition_governing_sweep(self, run_mode: TransitionRunMode) -> bool:
        return (
            run_mode == TransitionRunMode.ENVELOPE
            and self._is_as5100_governing_sweep_selected()
        )

    def _build_as5100_envelope_sweep(self) -> AS5100EnvelopeSweep:
        config = self._as5100_load_config()
        return AS5100EnvelopeSweep(
            model=config.model,
            selected_group_count=config.group_count,
            selected_group_spacing_m=config.group_spacing_m,
            reference_position_m=config.reference_position_m,
        )

    def _as5100_governing_sweep_request_metadata(self) -> dict[str, object]:
        config = self._as5100_load_config()
        loads = build_as5100_rail_loads(config)
        sweep = self._build_as5100_envelope_sweep()
        return as5100_load_metadata(
            config,
            loads=loads,
            arrangement="governing_envelope_sweep_requested",
            extra={
                "selected_group_count": sweep.selected_group_count,
                "selected_group_spacing_m": sweep.selected_group_spacing_m,
                "sweep_group_count_candidates": list(sweep.group_count_candidates),
                "sweep_group_spacing_candidates_m": list(sweep.group_spacing_candidates_m),
                "governing_metric": "max_abs_moment_nm",
            },
        )

    @staticmethod
    def _as5100_metadata_to_loads(load_source: dict[str, object] | None) -> list[PointLoad] | None:
        if not load_source or load_source.get("source_type") != "as5100_fixed_rail":
            return None
        config = AS5100RailLoadConfig(
            model=str(load_source.get("model", AS5100_MODEL_300LA)),
            group_count=int(load_source.get("group_count", 1) or 1),
            group_spacing_m=float(load_source.get("group_spacing_m", 12.0) or 12.0),
            reference_position_m=float(load_source.get("reference_position_m", 0.0) or 0.0),
        )
        return build_as5100_rail_loads(config)

    @staticmethod
    def _copy_load_source_metadata(load_source: dict[str, object] | None) -> dict[str, object] | None:
        if load_source is None:
            return None
        return dict(load_source)

    def _capture_load_source_metadata(self) -> dict[str, object]:
        return dict(self._current_load_source_metadata())

    def _last_static_load_source_metadata(self) -> dict[str, object] | None:
        if self._last_transition_result is not None:
            return self._copy_load_source_metadata(self._last_transition_load_source)
        if self._last_envelope_result is not None:
            return self._copy_load_source_metadata(self._last_envelope_load_source)
        return self._copy_load_source_metadata(self._last_analysis_load_source)

    def _last_dynamic_load_source_metadata(self) -> dict[str, object] | None:
        if self._last_dynamic_transition_result is not None:
            return self._copy_load_source_metadata(self._last_dynamic_transition_load_source)
        return self._copy_load_source_metadata(self._last_dynamic_load_source)

    def _current_load_source_metadata(self) -> dict[str, object]:
        if hasattr(self, "as5100_loads_checkbox") and self.as5100_loads_checkbox.isChecked():
            config = self._as5100_load_config()
            loads = build_as5100_rail_loads(config)
            if self._should_run_as5100_governing_sweep():
                return self._as5100_governing_sweep_request_metadata()
            return as5100_load_metadata(config, loads=loads)
        if self.train_loads_checkbox.isChecked():
            config = TrainLoadConfig(
                axle_load_n=kn_to_n(self.train_axle_load_input.value()),
                bogie_count=int(self.train_bogie_count_input.value()),
                bogie_spacing_m=mm_to_m(self.train_bogie_spacing_input.value()),
                axles_per_bogie=int(self.train_axles_per_bogie_input.value()),
                axle_spacing_m=mm_to_m(self.train_axle_spacing_input.value()),
                reference_bogie_center_m=mm_to_m(self.train_reference_input.value()),
            )
            loads = build_train_loads(config)
            return {
                "source_type": "train_builder",
                "arrangement": "user_selected",
                "load_basis": "axle_load_split_to_two_rails",
                "solver_load_basis": "wheel_load_per_rail",
                "axle_count": len(loads),
                "axle_load_n": config.axle_load_n,
                "wheel_load_n_per_rail": axle_load_to_wheel_load(config.axle_load_n),
                "max_axle_load_n": config.axle_load_n,
                "max_wheel_load_n_per_rail": axle_load_to_wheel_load(config.axle_load_n),
                "axle_positions_m": [load.position_m for load in loads],
                "axle_loads_n": [config.axle_load_n for _load in loads],
                "wheel_loads_n_per_rail": [load.load_newtons for load in loads],
                "bogie_count": config.bogie_count,
                "bogie_spacing_m": config.bogie_spacing_m,
                "axles_per_bogie": config.axles_per_bogie,
                "axle_spacing_m": config.axle_spacing_m,
            }
        if self.several_loads_checkbox.isChecked():
            return {
                "source_type": "manual_multiple_loads",
                "arrangement": "user_selected",
                "load_basis": "wheel_load_per_rail",
            }
        load_case = self.load_case_combo.currentData()
        return {
            "source_type": "single_point_load",
            "load_case_name": load_case.name if isinstance(load_case, LoadCase) else None,
            "arrangement": "user_selected",
            "load_basis": "wheel_load_per_rail",
        }

    def _format_as5100_load_source_summary(self) -> str:
        if not hasattr(self, "as5100_model_combo"):
            return ""
        config = self._as5100_load_config()
        try:
            loads = build_as5100_rail_loads(config)
        except ValueError as exc:
            return str(exc)
        metadata = as5100_load_metadata(config, loads=loads)
        max_wheel_kn = n_to_kn(float(metadata.get("max_wheel_load_n_per_rail", 0.0) or 0.0))
        max_axle_kn = n_to_kn(float(metadata["max_axle_load_n"]))
        arrangement_mode = self._as5100_arrangement_mode()
        if arrangement_mode == AS5100ArrangementMode.GOVERNING_SWEEP:
            sweep = self._build_as5100_envelope_sweep()
            group_counts = self._format_as5100_group_count_range(sweep.group_count_candidates)
            spacings = ", ".join(f"{value:.2f}" for value in sweep.group_spacing_candidates_m)
            return (
                f"{metadata['model']} governing envelope sweep\n"
                f"Groups {group_counts} | spacing [{spacings}] m\n"
                f"max axle {max_axle_kn:.0f} kN -> wheel {max_wheel_kn:.0f} kN/rail\n"
                f"x0={float(metadata['reference_position_m']):.3f} m | no automatic DLA applied"
            )
        return (
            f"{metadata['model']} fixed arrangement\n"
            f"{metadata['axle_count']} axles | "
            f"max axle {max_axle_kn:.0f} kN -> wheel {max_wheel_kn:.0f} kN/rail | "
            f"{metadata['group_count']} group(s) @ {float(metadata['group_spacing_m']):.2f} m\n"
            f"x0={float(metadata['reference_position_m']):.3f} m | no automatic DLA applied"
        )

    def _apply_advanced_static_settings(
        self,
        config: AnalysisConfig,
        *,
        rail: Rail,
        rail_area_m2: float | None,
    ) -> AnalysisConfig:
        foundation_model = self.foundation_model_combo.currentData()
        beam_theory = self.beam_theory_combo.currentData()
        rail_area_override = self.rail_area_input.value()
        resolved_rail_area_m2 = rail_area_m2
        if rail_area_override > 0:
            resolved_rail_area_m2 = cm2_to_m2(rail_area_override)
        poisson_ratio = self.poisson_ratio_input.value()
        shear_modulus_pa = rail.elastic_modulus_pa / (2.0 * (1.0 + poisson_ratio))
        config = replace(
            config,
            foundation_model=foundation_model,
            railpad_stiffness_n_per_m=kn_to_n(self.railpad_stiffness_input.value()),
            railpad_damping_n_s_per_m=kn_to_n(self.railpad_damping_input.value()),
            railpad_loss_factor=self.railpad_loss_factor_input.value(),
            trackbed_stiffness_n_per_m=kn_to_n(self.trackbed_stiffness_input.value()),
            trackbed_damping_n_s_per_m=kn_to_n(self.trackbed_damping_input.value()),
            trackbed_loss_factor=self.trackbed_loss_factor_input.value(),
            foundation_damping_model=self.foundation_damping_model_static_combo.currentData(),
            sleeper_mass_kg=self.sleeper_mass_input.value(),
            beam_theory=beam_theory,
            shear_modulus_pa=shear_modulus_pa,
            shear_correction_factor=self.kappa_input.value(),
            rail_area_m2=resolved_rail_area_m2,
        )
        if beam_theory == BeamTheory.TIMOSHENKO and resolved_rail_area_m2 is None:
            raise ValueError("Rail area is required for Timoshenko beam theory.")
        if self.nonuniform_profile_checkbox.isChecked():
            profile_type = self.profile_type_combo.currentData()
            if profile_type == FoundationProfileType.UNIFORM:
                raise ValueError(
                    "Select Step or Ramp when nonuniform foundation k(x) is enabled."
                )
            config = replace(
                config,
                foundation_profile_type=profile_type,
                foundation_profile_k1_n_per_m2=mn_per_m2_to_n_per_m2(self.profile_k1_input.value()),
                foundation_profile_k2_n_per_m2=mn_per_m2_to_n_per_m2(self.profile_k2_input.value()),
                foundation_profile_x_start_m=mm_to_m(self.profile_x_start_input.value()),
                foundation_profile_x_end_m=mm_to_m(self.profile_x_end_input.value()),
            )
        if self.discrete_supports_checkbox.isChecked():
            config = replace(
                config,
                use_discrete_supports=True,
                pad_stiffness_n_per_m=kn_to_n(self.pad_stiffness_input.value()),
                pad_damping_n_s_per_m=kn_to_n(self.pad_damping_input.value()),
                pad_loss_factor=self.pad_loss_factor_input.value(),
                nodes_between_sleepers=int(self.nodes_between_sleepers_input.value()),
            )
        if self.two_rail_checkbox.isChecked():
            right_loads = None
            if self.asymmetric_load_checkbox.isChecked():
                right_loads = [
                    PointLoad(
                        position_m=mm_to_m(self.right_load_position_input.value()),
                        load_newtons=kn_to_n(self.right_load_magnitude_input.value()),
                    )
                ]
            config = replace(
                config,
                use_two_rail=True,
                coupling_stiffness_n_per_m=kn_to_n(self.coupling_stiffness_input.value()),
                right_loads=right_loads,
            )
        if self.pasternak_checkbox.isChecked():
            config = replace(
                config,
                pasternak_shear_n=kn_to_n(self.pasternak_input.value()),
            )
        return config

    def _build_analysis_context(
        self,
        *,
        rail: Rail,
        sleeper: Sleeper,
        support: SupportProfile,
    ) -> tuple[AnalysisConfig, AnalysisInputs, AnalysisMode]:
        sleeper_spacing_m = mm_to_m(self.sleeper_spacing_input.value())
        loads = self._collect_analysis_loads()
        x_domain_m = None
        if len(loads) > 1:
            x_domain_m = build_load_domain(
                loads=loads,
                foundation_modulus_n_per_m2=support.foundation_modulus_n_per_m2,
                elastic_modulus_pa=rail.elastic_modulus_pa,
                moment_inertia_m4=rail.moment_inertia_m4,
            )
        design_inputs = DesignInputs(
            speed_kmh=self.design_speed_input.value(),
            track_factor=self.track_quality_combo.currentData(),
            probability_factor=self.probability_combo.currentData(),
            wheel_radius_mm=self.wheel_radius_input.value(),
            tensile_strength_mpa=self.tensile_strength_combo.currentData(),
            on_curve=self.curve_checkbox.isChecked(),
            ballast_depth_m=self._a3902_ballast_depth_m(),
            rail_centres_m=self._resolve_a3902_rail_centres_m(rail),
        )
        area_m2 = cm2_to_m2(rail.area_cm2) if rail.area_cm2 is not None else None
        discrete_stiffness = None
        if self.advanced_solver_checkbox.isChecked() and self.discrete_supports_checkbox.isChecked():
            discrete_stiffness = kn_to_n(self.pad_stiffness_input.value())

        analysis_inputs = AnalysisInputs(
            loads=loads,
            foundation_modulus_n_per_m2=support.foundation_modulus_n_per_m2,
            elastic_modulus_pa=rail.elastic_modulus_pa,
            moment_inertia_m4=rail.moment_inertia_m4,
            section_modulus_m3=rail.section_modulus_m3,
            sleeper_spacing_m=sleeper_spacing_m,
            sleeper_length_m=sleeper.length_m,
            sleeper_width_m=sleeper.width_m,
            x_domain_m=x_domain_m,
            section_modulus_head_m3=rail.section_modulus_head_m3,
            section_modulus_foot_m3=rail.section_modulus_foot_m3,
            area_m2=area_m2,
            discrete_support_stiffness_n_per_m=discrete_stiffness,
            design_inputs=design_inputs,
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
            x_domain_m=x_domain_m,
            section_modulus_head_m3=analysis_inputs.section_modulus_head_m3,
            section_modulus_foot_m3=analysis_inputs.section_modulus_foot_m3,
            area_m2=analysis_inputs.area_m2,
            discrete_support_stiffness_n_per_m=analysis_inputs.discrete_support_stiffness_n_per_m,
            design_inputs=analysis_inputs.design_inputs,
        )

        mode = AnalysisMode.CLOSED_FORM
        if self.advanced_solver_checkbox.isChecked():
            mode = AnalysisMode.NUMERICAL
            config = self._apply_advanced_static_settings(
                config,
                rail=rail,
                rail_area_m2=area_m2,
            )

        return config, analysis_inputs, mode

    def _build_envelope_context(
        self,
        *,
        rail: Rail,
        sleeper: Sleeper,
        support: SupportProfile,
    ) -> EnvelopeConfig:
        loads = self._collect_analysis_loads()
        if not loads:
            raise ValueError("Add at least one wheel load before running envelope analysis.")
        if self.envelope_range_auto_checkbox.isChecked():
            self._update_envelope_range_auto()
        reference = self.envelope_reference_input.value()
        x_ref_start = reference + self.envelope_range_start_input.value()
        x_ref_end = reference + self.envelope_range_end_input.value()
        if x_ref_end <= x_ref_start:
            raise ValueError("Envelope range end must be greater than start.")

        beta = beam_parameter_beta(
            support.foundation_modulus_n_per_m2,
            rail.elastic_modulus_pa,
            rail.moment_inertia_m4,
        )
        margin = ENVELOPE_AUTO_DECAY_MARGIN_FACTOR / beta
        offsets = [load.position_m for load in loads]
        if self.envelope_domain_auto_checkbox.isChecked():
            x_min = x_ref_start + min(offsets) - margin
            x_max = x_ref_end + max(offsets) + margin
            if x_max <= x_min:
                raise ValueError("Envelope domain is invalid for the selected movement range.")
            self.envelope_domain_start_input.set_value(x_min)
            self.envelope_domain_end_input.set_value(x_max)
        else:
            x_min = self.envelope_domain_start_input.value()
            x_max = self.envelope_domain_end_input.value()
            if x_max <= x_min:
                raise ValueError("Envelope domain end must be greater than start.")

        x_ref_start, x_ref_end = self._extend_movement_range_for_plot_domain(
            x_ref_start_m=x_ref_start,
            x_ref_end_m=x_ref_end,
            x_domain_m=(x_min, x_max),
            load_offsets_m=offsets,
            beta_per_m=beta,
        )

        depth_values = self._parse_envelope_depths()
        if any(depth < 0.10 for depth in depth_values) or any(depth > 2.0 for depth in depth_values):
            QMessageBox.warning(
                self,
                "Depth warning",
                "Some depth values are outside the recommended range (0.10 m to 2.0 m).",
            )

        if self.envelope_use_sleeper_geometry_checkbox.isChecked():
            self._sync_envelope_bearing_defaults()
        bearing_width = self.envelope_bearing_width_input.value()
        bearing_length = self.envelope_bearing_length_input.value()
        if bearing_width <= 0 or bearing_length <= 0:
            raise ValueError("Effective bearing dimensions must be positive.")

        rail_count = self.envelope_rail_count_combo.currentData()

        design_inputs = DesignInputs(
            speed_kmh=self.design_speed_input.value(),
            track_factor=self.track_quality_combo.currentData(),
            probability_factor=self.probability_combo.currentData(),
            wheel_radius_mm=self.wheel_radius_input.value(),
            tensile_strength_mpa=self.tensile_strength_combo.currentData(),
            on_curve=self.curve_checkbox.isChecked(),
            ballast_depth_m=self._a3902_ballast_depth_m(),
            rail_centres_m=self._resolve_a3902_rail_centres_m(rail),
        )
        area_m2 = cm2_to_m2(rail.area_cm2) if rail.area_cm2 is not None else None
        discrete_stiffness = None
        if self.discrete_supports_checkbox.isChecked():
            discrete_stiffness = kn_to_n(self.pad_stiffness_input.value())

        config = AnalysisConfig(
            loads=loads,
            foundation_modulus_n_per_m2=support.foundation_modulus_n_per_m2,
            elastic_modulus_pa=rail.elastic_modulus_pa,
            moment_inertia_m4=rail.moment_inertia_m4,
            section_modulus_m3=rail.section_modulus_m3,
            sleeper_spacing_m=mm_to_m(self.sleeper_spacing_input.value()),
            sleeper_length_m=sleeper.length_m,
            sleeper_width_m=sleeper.width_m,
            sample_count=401,
            x_domain_m=(x_min, x_max),
            section_modulus_head_m3=rail.section_modulus_head_m3,
            section_modulus_foot_m3=rail.section_modulus_foot_m3,
            area_m2=area_m2,
            discrete_support_stiffness_n_per_m=discrete_stiffness,
            design_inputs=design_inputs,
        )

        mode = AnalysisMode.CLOSED_FORM
        if self.static_mode_combo.currentData() == StaticMode.ENVELOPE_NUMERICAL:
            mode = AnalysisMode.NUMERICAL
            config = self._apply_advanced_static_settings(
                config,
                rail=rail,
                rail_area_m2=area_m2,
            )
            if config.use_two_rail:
                rail_count = 2

        step = self.envelope_step_input.value()
        if step <= 0:
            raise ValueError("Envelope step must be positive.")
        if mode == AnalysisMode.NUMERICAL:
            solver_domain = self._solver_domain_for_movement_range(
                x_ref_start_m=x_ref_start,
                x_ref_end_m=x_ref_end,
                plot_domain_m=(x_min, x_max),
                load_offsets_m=offsets,
                beta_per_m=beta,
            )
            config = replace(config, x_domain_m=solver_domain)
        step_count = int(math.floor((x_ref_end - x_ref_start) / step)) + 1
        if step_count > 2000:
            QMessageBox.warning(
                self,
                "Envelope warning",
                f"Envelope step count is high ({step_count} steps). Runtime may be long.",
            )

        return EnvelopeConfig(
            analysis_config=config,
            x_ref_start_m=x_ref_start,
            x_ref_end_m=x_ref_end,
            x_ref_step_m=step,
            x_domain_m=(x_min, x_max),
            bearing_width_m=bearing_width,
            bearing_length_m=bearing_length,
            depth_m=depth_values,
            rail_count=rail_count,
            mode=mode,
            as5100_sweep=self._build_as5100_envelope_sweep() if self._should_run_as5100_governing_sweep() else None,
            run_metadata=self._copy_load_source_metadata(self._current_load_source_metadata()),
        )

    def _build_transition_context(
        self,
        *,
        rail: Rail,
        sleeper: Sleeper,
        support: SupportProfile,
    ) -> tuple[TransitionContext, AnalysisInputs, EnvelopeConfig | None]:
        try:
            loads = self._collect_analysis_loads()
        except ValueError as exc:
            raise ValueError(str(exc)) from None
        if not loads:
            raise ValueError("Add at least one wheel load before running transition analysis.")

        if self.transition_domain_auto_checkbox.isChecked():
            self._update_transition_domain_auto()
        x_min = self.transition_domain_start_input.value()
        x_max = self.transition_domain_end_input.value()
        if x_max <= x_min:
            raise ValueError("Transition domain end must be greater than start.")

        profile_type = self._coerce_transition_profile_type(self.transition_profile_combo.currentData())
        k1_n_per_m2 = mn_per_m2_to_n_per_m2(self.transition_k1_input.value())
        if k1_n_per_m2 <= 0:
            raise ValueError("k₁ must be positive.")
        k2_n_per_m2: float | None = None
        transition_length_m: float | None = None
        segment_length_m: float | None = None

        if profile_type != TransitionProfileType.UNIFORM:
            k2_n_per_m2 = mn_per_m2_to_n_per_m2(self.transition_k2_input.value())
            if k2_n_per_m2 <= 0:
                raise ValueError("k₂ must be positive for non-uniform profiles.")
            if math.isclose(k2_n_per_m2, k1_n_per_m2, rel_tol=1.0e-9, abs_tol=1.0e-9):
                raise ValueError("k₂ must differ from k₁ for non-uniform transition profiles.")
        if profile_type in (TransitionProfileType.RAMP, TransitionProfileType.EXPONENTIAL):
            transition_length_m = self.transition_length_input.value()
            if transition_length_m <= 0:
                raise ValueError("Transition length must be positive for ramp profiles.")
        if profile_type == TransitionProfileType.SEGMENT:
            segment_length_m = self.transition_segment_length_input.value()
            if segment_length_m <= 0:
                raise ValueError("Segment length must be positive for local stiff segments.")

        self._validate_transition_domain_covers_profile(
            profile_type=profile_type,
            domain_m=(x_min, x_max),
            transition_length_m=transition_length_m,
            segment_length_m=segment_length_m,
        )

        sleeper_spacing_m = mm_to_m(self.sleeper_spacing_input.value())
        design_inputs = DesignInputs(
            speed_kmh=self.design_speed_input.value(),
            track_factor=self.track_quality_combo.currentData(),
            probability_factor=self.probability_combo.currentData(),
            wheel_radius_mm=self.wheel_radius_input.value(),
            tensile_strength_mpa=self.tensile_strength_combo.currentData(),
            on_curve=self.curve_checkbox.isChecked(),
            ballast_depth_m=self._a3902_ballast_depth_m(),
            rail_centres_m=self._resolve_a3902_rail_centres_m(rail),
        )
        area_m2 = cm2_to_m2(rail.area_cm2) if rail.area_cm2 is not None else None
        discrete_stiffness = None
        if self.advanced_solver_checkbox.isChecked() and self.discrete_supports_checkbox.isChecked():
            discrete_stiffness = kn_to_n(self.pad_stiffness_input.value())

        analysis_inputs = AnalysisInputs(
            loads=loads,
            foundation_modulus_n_per_m2=k1_n_per_m2,
            elastic_modulus_pa=rail.elastic_modulus_pa,
            moment_inertia_m4=rail.moment_inertia_m4,
            section_modulus_m3=rail.section_modulus_m3,
            sleeper_spacing_m=sleeper_spacing_m,
            sleeper_length_m=sleeper.length_m,
            sleeper_width_m=sleeper.width_m,
            x_domain_m=(x_min, x_max),
            section_modulus_head_m3=rail.section_modulus_head_m3,
            section_modulus_foot_m3=rail.section_modulus_foot_m3,
            area_m2=area_m2,
            discrete_support_stiffness_n_per_m=discrete_stiffness,
            design_inputs=design_inputs,
        )

        config = AnalysisConfig(
            loads=analysis_inputs.loads,
            foundation_modulus_n_per_m2=k1_n_per_m2,
            elastic_modulus_pa=analysis_inputs.elastic_modulus_pa,
            moment_inertia_m4=analysis_inputs.moment_inertia_m4,
            section_modulus_m3=analysis_inputs.section_modulus_m3,
            sleeper_spacing_m=analysis_inputs.sleeper_spacing_m,
            sleeper_length_m=analysis_inputs.sleeper_length_m,
            sleeper_width_m=analysis_inputs.sleeper_width_m,
            sample_count=analysis_inputs.sample_count,
            x_domain_m=(x_min, x_max),
            section_modulus_head_m3=analysis_inputs.section_modulus_head_m3,
            section_modulus_foot_m3=analysis_inputs.section_modulus_foot_m3,
            area_m2=analysis_inputs.area_m2,
            discrete_support_stiffness_n_per_m=analysis_inputs.discrete_support_stiffness_n_per_m,
            design_inputs=analysis_inputs.design_inputs,
        )

        analysis_mode = AnalysisMode.CLOSED_FORM
        if self.advanced_solver_checkbox.isChecked() or profile_type != TransitionProfileType.UNIFORM:
            analysis_mode = AnalysisMode.NUMERICAL
            config = self._apply_advanced_static_settings(
                config,
                rail=rail,
                rail_area_m2=area_m2,
            )

        if profile_type != TransitionProfileType.UNIFORM and config.foundation_model != FoundationModelType.WINKLER:
            raise ValueError("Non-uniform transition profiles require the Winkler foundation model.")

        k_profile: list[float] | None = None
        if profile_type != TransitionProfileType.UNIFORM:
            beta = beam_parameter_beta(
                k1_n_per_m2,
                rail.elastic_modulus_pa,
                rail.moment_inertia_m4,
            )
            x_values, _, _, _ = _build_grid(
                beta,
                config.sample_count,
                config.sleeper_spacing_m,
                config.use_discrete_supports or config.use_two_rail,
                config.nodes_between_sleepers,
                config.x_domain_m,
            )
            k_profile = build_transition_profile(
                x_values=x_values,
                profile_type=profile_type,
                k1_n_per_m2=k1_n_per_m2,
                k2_n_per_m2=k2_n_per_m2,
                transition_length_m=transition_length_m,
                segment_length_m=segment_length_m,
            )
            if math.isclose(max(k_profile), min(k_profile), rel_tol=1.0e-9, abs_tol=1.0e-9):
                raise ValueError(
                    "Transition k(x) profile is flat on the analysis grid. "
                    "Expand the transition domain or adjust k₁/k₂."
                )
            config = replace(config, foundation_profile_n_per_m2=k_profile)

        run_mode = self._coerce_transition_run_mode(self.transition_run_mode_combo.currentData())
        envelope_config: EnvelopeConfig | None = None
        if run_mode == TransitionRunMode.ENVELOPE:
            if self.transition_range_auto_checkbox.isChecked():
                self._update_transition_range_auto()
            reference = self.transition_reference_input.value()
            x_ref_start = reference + self.transition_range_start_input.value()
            x_ref_end = reference + self.transition_range_end_input.value()
            if x_ref_end <= x_ref_start:
                raise ValueError("Transition movement range end must be greater than start.")
            k_extent_for_buffer = k1_n_per_m2
            if profile_type != TransitionProfileType.UNIFORM and k2_n_per_m2 is not None and k2_n_per_m2 > 0.0:
                k_extent_for_buffer = min(k_extent_for_buffer, k2_n_per_m2)
            beta_for_buffer = beam_parameter_beta(
                k_extent_for_buffer,
                rail.elastic_modulus_pa,
                rail.moment_inertia_m4,
            )
            x_ref_start, x_ref_end = self._extend_movement_range_for_plot_domain(
                x_ref_start_m=x_ref_start,
                x_ref_end_m=x_ref_end,
                x_domain_m=(x_min, x_max),
                load_offsets_m=[load.position_m for load in loads],
                beta_per_m=beta_for_buffer,
            )
            step = self.transition_step_input.value()
            if step <= 0:
                raise ValueError("Transition movement increment must be positive.")
            if analysis_mode == AnalysisMode.NUMERICAL:
                solver_domain = self._solver_domain_for_movement_range(
                    x_ref_start_m=x_ref_start,
                    x_ref_end_m=x_ref_end,
                    plot_domain_m=(x_min, x_max),
                    load_offsets_m=[load.position_m for load in loads],
                    beta_per_m=beta_for_buffer,
                )
                config = replace(config, x_domain_m=solver_domain)
                if profile_type != TransitionProfileType.UNIFORM:
                    solver_x_values, _, _, _ = _build_grid(
                        beta_for_buffer,
                        config.sample_count,
                        config.sleeper_spacing_m,
                        config.use_discrete_supports or config.use_two_rail,
                        config.nodes_between_sleepers,
                        solver_domain,
                    )
                    solver_k_profile = build_transition_profile(
                        x_values=solver_x_values,
                        profile_type=profile_type,
                        k1_n_per_m2=k1_n_per_m2,
                        k2_n_per_m2=k2_n_per_m2,
                        transition_length_m=transition_length_m,
                        segment_length_m=segment_length_m,
                    )
                    config = replace(config, foundation_profile_n_per_m2=solver_k_profile)
            step_count = int(math.floor((x_ref_end - x_ref_start) / step)) + 1
            if step_count > 2000:
                QMessageBox.warning(
                    self,
                    "Transition warning",
                    f"Transition envelope step count is high ({step_count} steps). Runtime may be long.",
                )

            depth_values = self._parse_envelope_depths()
            if any(depth < 0.10 for depth in depth_values) or any(depth > 2.0 for depth in depth_values):
                QMessageBox.warning(
                    self,
                    "Depth warning",
                    "Some depth values are outside the recommended range (0.10 m to 2.0 m).",
                )
            if self.envelope_use_sleeper_geometry_checkbox.isChecked():
                self._sync_envelope_bearing_defaults()
            bearing_width = self.envelope_bearing_width_input.value()
            bearing_length = self.envelope_bearing_length_input.value()
            if bearing_width <= 0 or bearing_length <= 0:
                raise ValueError("Effective bearing dimensions must be positive.")

            rail_count = self.envelope_rail_count_combo.currentData()
            if config.use_two_rail:
                rail_count = 2
            if analysis_mode == AnalysisMode.CLOSED_FORM and config.use_two_rail:
                raise ValueError("Two-rail coupling is not supported for closed-form envelope analysis.")

            as5100_sweep = (
                self._build_as5100_envelope_sweep()
                if self._should_run_as5100_transition_governing_sweep(run_mode)
                else None
            )
            run_metadata = (
                self._as5100_governing_sweep_request_metadata()
                if as5100_sweep is not None
                else self._copy_load_source_metadata(self._current_load_source_metadata())
            )
            envelope_config = EnvelopeConfig(
                analysis_config=config,
                x_ref_start_m=x_ref_start,
                x_ref_end_m=x_ref_end,
                x_ref_step_m=step,
                x_domain_m=(x_min, x_max),
                bearing_width_m=bearing_width,
                bearing_length_m=bearing_length,
                depth_m=depth_values,
                rail_count=rail_count,
                mode=analysis_mode,
                as5100_sweep=as5100_sweep,
                run_metadata=run_metadata,
            )

        context = TransitionContext(
            run_mode=run_mode,
            profile_type=profile_type,
            template_name=self.transition_template_combo.currentText(),
            preset_name=self.transition_preset_combo.currentText(),
            k1_n_per_m2=k1_n_per_m2,
            k2_n_per_m2=k2_n_per_m2,
            transition_length_m=transition_length_m,
            segment_length_m=segment_length_m,
            domain_m=(x_min, x_max),
            analysis_config=config,
            analysis_mode=analysis_mode,
            k_profile_n_per_m2=k_profile,
        )
        return context, analysis_inputs, envelope_config

    @staticmethod
    def _validate_transition_domain_covers_profile(
        *,
        profile_type: TransitionProfileType,
        domain_m: tuple[float, float],
        transition_length_m: float | None,
        segment_length_m: float | None,
    ) -> None:
        if profile_type == TransitionProfileType.UNIFORM:
            return
        x_min, x_max = domain_m
        tolerance = 1.0e-9
        if profile_type == TransitionProfileType.STEP:
            if x_min < -tolerance and x_max > tolerance:
                return
            raise ValueError("Transition domain must span x=0 for a step stiffness profile.")
        if profile_type in (TransitionProfileType.RAMP, TransitionProfileType.EXPONENTIAL):
            if transition_length_m is None:
                raise ValueError("Transition length is required for ramp/exponential profiles.")
            if x_min <= tolerance and x_max >= transition_length_m - tolerance:
                return
            raise ValueError(
                "Transition domain must include the full ramp interval from x=0 to x=Lₜ."
            )
        if profile_type == TransitionProfileType.SEGMENT:
            if segment_length_m is None:
                raise ValueError("Segment length is required for segment profiles.")
            half = 0.5 * segment_length_m
            if x_min <= -half + tolerance and x_max >= half - tolerance:
                return
            raise ValueError("Transition domain must include the full stiff segment interval.")

    def _parse_envelope_depths(self) -> list[float]:
        raw = self.envelope_depths_input.text().strip()
        if not raw:
            raise ValueError("Enter at least one formation depth.")
        depths: list[float] = []
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            try:
                depths.append(float(entry))
            except ValueError:
                raise ValueError(f"Invalid depth value: '{entry}'") from None
        if not depths:
            raise ValueError("Enter at least one formation depth.")
        if any(depth <= 0 for depth in depths):
            raise ValueError("Formation depths must be positive.")
        return depths

    def _parse_irregularity_profile_values(self, raw: str, *, scale: float = 1.0) -> list[float]:
        values: list[float] = []
        for token in raw.split(","):
            entry = token.strip()
            if not entry:
                continue
            try:
                values.append(float(entry) * scale)
            except ValueError:
                raise ValueError(f"Invalid irregularity value: '{entry}'") from None
        return values

    def _build_irregularity_input(self) -> IrregularityInput | None:
        mode = self.irregularity_mode_combo.currentData()
        if mode is None:
            return None
        if mode == IrregularityMode.PROFILE:
            x_values = self._parse_irregularity_profile_values(self.irregularity_profile_x_input.text())
            z_values_m = self._parse_irregularity_profile_values(
                self.irregularity_profile_z_input.text(),
                scale=1.0e-3,
            )
            if not x_values or not z_values_m:
                raise ValueError("Irregularity profile requires x and z values.")
            if len(x_values) != len(z_values_m):
                raise ValueError("Irregularity profile x and z values must have matching lengths.")
            return IrregularityInput(
                mode=IrregularityMode.PROFILE,
                profile_x_m=x_values,
                profile_z_m=z_values_m,
            )
        return IrregularityInput(
            mode=IrregularityMode.SYNTHETIC_PSD,
            psd_level_m3=self.irregularity_psd_level_input.value(),
            seed=self.irregularity_seed_input.value(),
        )

    def _build_dynamic_context(
        self,
        *,
        rail: Rail,
        sleeper: Sleeper,
        support: SupportProfile,
    ) -> tuple[DippedJointConfig | DynamicConfig | DynamicTransitionConfig, DynamicMode]:
        mode = self._coerce_dynamic_mode(self.dynamic_mode_combo.currentData())
        if mode == DynamicMode.DIPPED_JOINT:
            return self._build_dipped_joint_context(), mode
        if mode == DynamicMode.TRANSITION:
            return self._build_dynamic_transition_context(rail=rail, support=support), mode
        return self._build_moving_load_context(rail=rail, sleeper=sleeper, support=support), mode

    def _build_special_context(self) -> tuple[FloatingSlabConfig, SpecialMode]:
        mode = self.special_mode_combo.currentData()
        if mode != SpecialMode.FLOATING_SLAB:
            raise ValueError("Select a special analysis mode before running.")
        slab_mass = self.floating_slab_mass_input.value()
        isolator_stiffness = kn_to_n(self.floating_slab_stiffness_input.value())
        isolator_damping = kn_to_n(self.floating_slab_damping_input.value())
        static_load = kn_to_n(self.floating_slab_static_load_input.value())
        f_min = self.floating_slab_freq_min_input.value()
        f_max = self.floating_slab_freq_max_input.value()
        points = int(self.floating_slab_freq_points_input.value())
        config = FloatingSlabConfig(
            slab_mass_kg=slab_mass,
            isolator_stiffness_n_per_m=isolator_stiffness,
            isolator_damping_n_s_per_m=isolator_damping,
            static_load_n=static_load,
            frequency_min_hz=f_min,
            frequency_max_hz=f_max,
            frequency_points=points,
        )
        return config, mode

    def _build_moving_load_context(
        self,
        *,
        rail: Rail,
        sleeper: Sleeper,
        support: SupportProfile,
    ) -> DynamicConfig:
        speed = self.speed_input.value()
        if speed <= 0:
            raise ValueError("Speed must be positive for dynamic analysis.")
        loads = self._collect_analysis_loads()

        damping_model = self.foundation_damping_model_combo.currentData()
        damping_loss_factor = 0.0
        if damping_model == DampingModel.HYSTERETIC:
            damping_n_s_per_m2 = 0.0
            damping_loss_factor = self.damping_loss_factor_input.value()
        else:
            damping_mode = self.damping_mode_combo.currentData()
            if damping_mode == "ratio":
                damping_ratio = self.damping_ratio_input.value()
                damping_n_s_per_m2 = 2.0 * damping_ratio * math.sqrt(
                    support.foundation_modulus_n_per_m2 * rail.mass_kg_per_m
                )
            else:
                damping_n_s_per_m2 = kn_to_n(self.damping_coefficient_input.value())

        probe_positions = self._safe_parse_probe_positions()
        if not probe_positions:
            raise ValueError("Provide at least one probe location for dynamic analysis.")

        excitation_mode = DynamicExcitationMode.MOVING_LOAD
        boundary_mode = DynamicBoundaryMode.ZERO_PAD
        oscillator_unsprung_mass_kg: float | None = None
        oscillator_stiffness_n_per_m: float | None = None
        oscillator_damping_n_s_per_m = 0.0
        irregularity_input: IrregularityInput | None = None
        if self.dynamic_advanced_group.isChecked():
            excitation_mode = self.dynamic_excitation_mode_combo.currentData()
            boundary_mode = self.dynamic_boundary_mode_combo.currentData()
            if excitation_mode == DynamicExcitationMode.MOVING_OSCILLATOR:
                oscillator_unsprung_mass_kg = self.oscillator_unsprung_mass_input.value()
                oscillator_stiffness_n_per_m = self.oscillator_stiffness_input.value() * 1.0e6
                oscillator_damping_n_s_per_m = self.oscillator_damping_input.value() * 1.0e3
            irregularity_input = self._build_irregularity_input()

        config = DynamicConfig(
            loads=loads,
            elastic_modulus_pa=rail.elastic_modulus_pa,
            moment_inertia_m4=rail.moment_inertia_m4,
            section_modulus_m3=rail.section_modulus_m3,
            mass_kg_per_m=rail.mass_kg_per_m,
            foundation_modulus_n_per_m2=support.foundation_modulus_n_per_m2,
            foundation_damping_n_s_per_m2=damping_n_s_per_m2,
            foundation_damping_model=damping_model,
            foundation_loss_factor=damping_loss_factor,
            speed_m_per_s=speed,
            domain_length_m=self.domain_length_input.value(),
            spatial_step_m=self.spatial_step_input.value(),
            probe_positions_m=probe_positions,
            time_window_s=self.time_window_input.value(),
            sample_rate_hz=self.sample_rate_input.value(),
            pasternak_shear_n=0.0,
            psd_segment_length=int(self.psd_segment_length_input.value()),
            psd_overlap=self.psd_overlap_input.value(),
            excitation_mode=excitation_mode,
            boundary_mode=boundary_mode,
            oscillator_unsprung_mass_kg=oscillator_unsprung_mass_kg,
            oscillator_suspension_stiffness_n_per_m=oscillator_stiffness_n_per_m,
            oscillator_suspension_damping_n_s_per_m=oscillator_damping_n_s_per_m,
            irregularity_input=irregularity_input,
        )
        return config

    def _build_dipped_joint_context(self) -> DippedJointConfig:
        dip_angle_rad = self.dip_angle_input.value() * 1.0e-3
        speed = self.speed_input.value()
        if speed < 0:
            raise ValueError("Speed cannot be negative for dipped joint analysis.")
        if speed > 83:
            warnings.warn(
                (
                    f"Speed {speed:.1f} m/s is very high for dipped joint analysis. "
                    "Results may not be physically realistic."
                ),
                UserWarning,
            )
        return DippedJointConfig(
            static_wheel_load_n=kn_to_n(self.load_magnitude_input.value()),
            total_dip_angle_rad=dip_angle_rad,
            speed_m_per_s=speed,
            hertzian_stiffness_n_per_m=self.hertzian_stiffness_input.value() * 1.0e6,
            track_mass_p1_kg=self.track_mass_p1_input.value(),
            unsprung_mass_kg=self.unsprung_mass_input.value(),
            track_mass_p2_kg=self.track_mass_p2_input.value(),
            track_stiffness_p2_n_per_m=self.track_stiffness_p2_input.value() * 1.0e6,
            track_damping_p2_n_s_per_m=self.track_damping_p2_input.value() * 1.0e3,
        )

    def _build_dynamic_transition_context(
        self,
        *,
        rail: Rail,
        support: SupportProfile,
    ) -> DynamicTransitionConfig:
        speed = self.speed_input.value()
        if speed < 0:
            raise ValueError("Speed cannot be negative for dynamic transition analysis.")
        loads = self._collect_analysis_loads()
        if not loads:
            raise ValueError("Add at least one wheel load before running dynamic transition analysis.")

        profile_type = self._coerce_dynamic_transition_profile_type(
            self.dynamic_transition_profile_combo.currentData()
        )
        run_mode = self._coerce_dynamic_transition_run_mode(
            self.dynamic_transition_run_mode_combo.currentData()
        )
        k1 = mn_per_m2_to_n_per_m2(self.dynamic_transition_k1_input.value())
        k2: float | None = None
        if profile_type != DynamicTransitionProfileType.UNIFORM:
            k2 = mn_per_m2_to_n_per_m2(self.dynamic_transition_k2_input.value())
        transition_length: float | None = None
        if profile_type in (DynamicTransitionProfileType.RAMP, DynamicTransitionProfileType.EXPONENTIAL):
            transition_length = self.dynamic_transition_length_input.value()
        segment_length: float | None = None
        if profile_type == DynamicTransitionProfileType.SEGMENT:
            segment_length = self.dynamic_transition_segment_length_input.value()

        damping_model = self.foundation_damping_model_combo.currentData()
        damping_loss_factor = 0.0
        if damping_model == DampingModel.HYSTERETIC:
            damping_n_s_per_m2 = 0.0
            damping_loss_factor = self.damping_loss_factor_input.value()
        else:
            damping_mode = self.damping_mode_combo.currentData()
            if damping_mode == "ratio":
                damping_ratio = self.damping_ratio_input.value()
                damping_n_s_per_m2 = 2.0 * damping_ratio * math.sqrt(
                    k1 * rail.mass_kg_per_m
                )
            else:
                damping_n_s_per_m2 = kn_to_n(self.damping_coefficient_input.value())

        probe_positions = self._safe_parse_probe_positions()
        if not probe_positions:
            raise ValueError("Provide at least one probe location for dynamic transition analysis.")

        excitation_mode = DynamicExcitationMode.MOVING_LOAD
        boundary_mode = DynamicBoundaryMode.ZERO_PAD
        oscillator_unsprung_mass_kg: float | None = None
        oscillator_stiffness_n_per_m: float | None = None
        oscillator_damping_n_s_per_m = 0.0
        irregularity_input: IrregularityInput | None = None
        if self.dynamic_advanced_group.isChecked():
            excitation_mode = self.dynamic_excitation_mode_combo.currentData()
            boundary_mode = self.dynamic_boundary_mode_combo.currentData()
            if excitation_mode == DynamicExcitationMode.MOVING_OSCILLATOR:
                oscillator_unsprung_mass_kg = self.oscillator_unsprung_mass_input.value()
                oscillator_stiffness_n_per_m = self.oscillator_stiffness_input.value() * 1.0e6
                oscillator_damping_n_s_per_m = self.oscillator_damping_input.value() * 1.0e3
            irregularity_input = self._build_irregularity_input()

        x_ref = self.dynamic_transition_x_ref_input.value()
        x_ref_start: float | None = None
        x_ref_end: float | None = None
        x_ref_step: float | None = None
        if run_mode == DynamicTransitionRunMode.ENVELOPE:
            x_ref_start = x_ref + self.dynamic_transition_range_start_input.value()
            x_ref_end = x_ref + self.dynamic_transition_range_end_input.value()
            x_ref_step = self.dynamic_transition_step_input.value()

        stiffness_ratio: float | None = None
        if k2 is not None and k1 > 0:
            stiffness_ratio = k2 / k1

        return DynamicTransitionConfig(
            loads=loads,
            elastic_modulus_pa=rail.elastic_modulus_pa,
            moment_inertia_m4=rail.moment_inertia_m4,
            section_modulus_m3=rail.section_modulus_m3,
            mass_kg_per_m=rail.mass_kg_per_m,
            foundation_modulus_n_per_m2=k1,
            foundation_damping_n_s_per_m2=damping_n_s_per_m2,
            speed_m_per_s=speed,
            domain_length_m=self.domain_length_input.value(),
            spatial_step_m=self.spatial_step_input.value(),
            probe_positions_m=probe_positions,
            time_window_s=self.time_window_input.value(),
            sample_rate_hz=self.sample_rate_input.value(),
            foundation_damping_model=damping_model,
            foundation_loss_factor=damping_loss_factor,
            pasternak_shear_n=0.0,
            psd_segment_length=int(self.psd_segment_length_input.value()),
            psd_overlap=self.psd_overlap_input.value(),
            excitation_mode=excitation_mode,
            boundary_mode=boundary_mode,
            oscillator_unsprung_mass_kg=oscillator_unsprung_mass_kg,
            oscillator_suspension_stiffness_n_per_m=oscillator_stiffness_n_per_m,
            oscillator_suspension_damping_n_s_per_m=oscillator_damping_n_s_per_m,
            irregularity_input=irregularity_input,
            profile_type=profile_type,
            run_mode=run_mode,
            solver_fidelity=self.dynamic_transition_solver_fidelity_combo.currentData(),
            k1_n_per_m2=k1,
            k2_n_per_m2=k2,
            transition_length_m=transition_length,
            segment_length_m=segment_length,
            x_ref_m=x_ref,
            x_ref_start_m=x_ref_start,
            x_ref_end_m=x_ref_end,
            x_ref_step_m=x_ref_step,
            transition_stiffness_ratio=stiffness_ratio,
        )

    def _run_analysis(self) -> None:
        rail = self.rail_combo.currentData()
        sleeper = self.sleeper_combo.currentData()
        pad = self.pad_combo.currentData()
        support = self.support_combo.currentData()
        load_case = self.load_case_combo.currentData()
        if not all([rail, sleeper, support, load_case]):
            QMessageBox.warning(self, "Missing data", "Select rail, sleeper, support profile, and load case.")
            return
        self.session.refresh(rail)
        self.session.refresh(sleeper)
        if pad is not None:
            self.session.refresh(pad)
        self.session.refresh(support)
        self.session.refresh(load_case)

        analysis_type = self.analysis_type_combo.currentData()
        if analysis_type == AnalysisType.DYNAMIC:
            mode = self._coerce_dynamic_mode(self.dynamic_mode_combo.currentData())
            if mode == DynamicMode.DIPPED_JOINT:
                if not self._validate_dipped_joint_inputs():
                    return
        self.run_button.setEnabled(False)
        self._set_export_buttons_enabled(False)
        self._set_dynamic_export_buttons_enabled(False)
        self._set_cancel_enabled(False)
        self.statusBar().showMessage("Running analysis...")

        self.thread = QThread(self)
        if analysis_type == AnalysisType.SPECIAL:
            try:
                config, mode = self._build_special_context()
            except ValueError as exc:
                self.run_button.setEnabled(True)
                self.statusBar().showMessage("Analysis failed")
                QMessageBox.warning(self, "Validation error", str(exc))
                return
            self.special_worker = SpecialAnalysisWorker(config, mode)
            self.special_worker.moveToThread(self.thread)
            self.thread.started.connect(self.special_worker.run)
            self.special_worker.finished.connect(self._handle_special_result)
            self.special_worker.failed.connect(self._handle_analysis_error)
            self.special_worker.finished.connect(self.thread.quit)
            self.special_worker.finished.connect(self.special_worker.deleteLater)
            self.special_worker.failed.connect(self.thread.quit)
            self.special_worker.failed.connect(self.special_worker.deleteLater)
        elif analysis_type == AnalysisType.DYNAMIC:
            try:
                config, mode = self._build_dynamic_context(
                    rail=rail,
                    sleeper=sleeper,
                    support=support,
                )
            except ValueError as exc:
                self.run_button.setEnabled(True)
                self.statusBar().showMessage("Analysis failed")
                QMessageBox.warning(self, "Validation error", str(exc))
                return
            load_source = self._capture_load_source_metadata()
            self._pending_dynamic_load_source = self._copy_load_source_metadata(load_source)
            if isinstance(config, DynamicConfig):
                self._log_dynamic_inputs(
                    rail=rail,
                    sleeper=sleeper,
                    pad=pad,
                    support=support,
                    load_case=load_case,
                    config=config,
                    load_source=load_source,
                )
            self.dynamic_worker = DynamicAnalysisWorker(config, mode)
            self.dynamic_worker.moveToThread(self.thread)
            self.thread.started.connect(self.dynamic_worker.run)
            self.dynamic_worker.finished.connect(self._handle_dynamic_result)
            self.dynamic_worker.failed.connect(self._handle_analysis_error)
            self.dynamic_worker.finished.connect(self.thread.quit)
            self.dynamic_worker.finished.connect(self.dynamic_worker.deleteLater)
            self.dynamic_worker.failed.connect(self.thread.quit)
            self.dynamic_worker.failed.connect(self.dynamic_worker.deleteLater)
        else:
            if hasattr(self, "transition_group") and self.transition_group.isChecked():
                self._run_transition_analysis(
                    rail=rail,
                    sleeper=sleeper,
                    pad=pad,
                    support=support,
                    load_case=load_case,
                )
                return
            static_mode = self.static_mode_combo.currentData()
            if static_mode in (StaticMode.ENVELOPE_CLOSED_FORM, StaticMode.ENVELOPE_NUMERICAL):
                try:
                    envelope_config = self._build_envelope_context(
                        rail=rail,
                        sleeper=sleeper,
                        support=support,
                    )
                except ValueError as exc:
                    self.run_button.setEnabled(True)
                    self.statusBar().showMessage("Analysis failed")
                    QMessageBox.warning(self, "Validation error", str(exc))
                    return
                load_source = self._capture_load_source_metadata()
                self._pending_envelope_load_source = self._copy_load_source_metadata(load_source)
                self._log_envelope_inputs(
                    rail=rail,
                    sleeper=sleeper,
                    pad=pad,
                    support=support,
                    load_case=load_case,
                    config=envelope_config,
                    load_source=load_source,
                )
                self._write_envelope_snapshot(envelope_config, load_source=load_source)
                self.envelope_worker = EnvelopeAnalysisWorker(envelope_config)
                self.envelope_worker.moveToThread(self.thread)
                self.thread.started.connect(self.envelope_worker.run)
                self.envelope_worker.progress.connect(self._handle_envelope_progress)
                self.envelope_worker.finished.connect(self._handle_envelope_result)
                self.envelope_worker.failed.connect(self._handle_analysis_error)
                self.envelope_worker.cancelled.connect(self._handle_analysis_cancelled)
                self.envelope_worker.finished.connect(self.thread.quit)
                self.envelope_worker.finished.connect(self.envelope_worker.deleteLater)
                self.envelope_worker.failed.connect(self.thread.quit)
                self.envelope_worker.failed.connect(self.envelope_worker.deleteLater)
                self.envelope_worker.cancelled.connect(self.thread.quit)
                self.envelope_worker.cancelled.connect(self.envelope_worker.deleteLater)
                self._set_cancel_enabled(True)
            else:
                try:
                    config, analysis_inputs, mode = self._build_analysis_context(
                        rail=rail,
                        sleeper=sleeper,
                        support=support,
                    )
                except ValueError as exc:
                    self.run_button.setEnabled(True)
                    self.statusBar().showMessage("Analysis failed")
                    QMessageBox.warning(self, "Validation error", str(exc))
                    return
                load_source = self._capture_load_source_metadata()
                self._pending_analysis_load_source = self._copy_load_source_metadata(load_source)
                self._log_analysis_inputs(
                    rail=rail,
                    sleeper=sleeper,
                    pad=pad,
                    support=support,
                    load_case=load_case,
                    analysis_inputs=analysis_inputs,
                    config=config,
                    load_source=load_source,
                )
                self._write_analysis_snapshot(analysis_inputs, config, load_source=load_source)
                self.worker = AnalysisWorker(config, analysis_inputs, mode)
                self.worker.moveToThread(self.thread)
                self.thread.started.connect(self.worker.run)
                self.worker.finished.connect(self._handle_analysis_result)
                self.worker.failed.connect(self._handle_analysis_error)
                self.worker.finished.connect(self.thread.quit)
                self.worker.finished.connect(self.worker.deleteLater)
                self.worker.failed.connect(self.thread.quit)
                self.worker.failed.connect(self.worker.deleteLater)

        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _run_transition_analysis(
        self,
        *,
        rail: Rail,
        sleeper: Sleeper,
        pad: Pad | None,
        support: SupportProfile,
        load_case: LoadCase | None,
    ) -> None:
        try:
            context, analysis_inputs, envelope_config = self._build_transition_context(
                rail=rail,
                sleeper=sleeper,
                support=support,
            )
        except ValueError as exc:
            self.run_button.setEnabled(True)
            self.statusBar().showMessage("Analysis failed")
            QMessageBox.warning(self, "Validation error", str(exc))
            return

        self._pending_transition_context = context
        if envelope_config is not None and envelope_config.run_metadata is not None:
            load_source = self._copy_load_source_metadata(envelope_config.run_metadata) or {}
        else:
            load_source = self._capture_load_source_metadata()
        self._pending_transition_load_source = self._copy_load_source_metadata(load_source)
        self._last_transition_result = None
        self._last_transition_context = None
        self._log_transition_inputs(
            rail=rail,
            sleeper=sleeper,
            pad=pad,
            support=support,
            load_case=load_case,
            context=context,
            load_source=load_source,
        )
        self._write_transition_snapshot(context, analysis_inputs, envelope_config, load_source=load_source)

        if context.run_mode == TransitionRunMode.ENVELOPE:
            if envelope_config is None:
                raise ValueError("Envelope config was not created for transition envelope run.")
            self.envelope_worker = EnvelopeAnalysisWorker(envelope_config)
            self.envelope_worker.moveToThread(self.thread)
            self.thread.started.connect(self.envelope_worker.run)
            self.envelope_worker.progress.connect(self._handle_envelope_progress)
            self.envelope_worker.finished.connect(self._handle_envelope_result)
            self.envelope_worker.failed.connect(self._handle_analysis_error)
            self.envelope_worker.cancelled.connect(self._handle_analysis_cancelled)
            self.envelope_worker.finished.connect(self.thread.quit)
            self.envelope_worker.finished.connect(self.envelope_worker.deleteLater)
            self.envelope_worker.failed.connect(self.thread.quit)
            self.envelope_worker.failed.connect(self.envelope_worker.deleteLater)
            self.envelope_worker.cancelled.connect(self.thread.quit)
            self.envelope_worker.cancelled.connect(self.envelope_worker.deleteLater)
            self._set_cancel_enabled(True)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.start()
            return

        self.worker = AnalysisWorker(context.analysis_config, analysis_inputs, context.analysis_mode)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._handle_analysis_result)
        self.worker.failed.connect(self._handle_analysis_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.thread.quit)
        self.worker.failed.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _clear_analysis_state(self) -> None:
        self._last_analysis_result = None
        self._last_analysis_inputs = None
        self._last_analysis_config = None
        self._last_analysis_mode = None
        self._pending_analysis_load_source = None
        self._last_analysis_load_source = None
        self._last_envelope_result = None
        self._last_envelope_config = None
        self._pending_envelope_load_source = None
        self._last_envelope_load_source = None
        self._last_analysis_stress = None
        self._last_envelope_stress = None
        self._last_static_mode = None
        self._last_transition_result = None
        self._last_transition_context = None
        self._pending_transition_context = None
        self._pending_transition_load_source = None
        self._last_transition_load_source = None
        self._clear_static_overlays(render=False)

    def _clear_dynamic_state(self) -> None:
        self._last_dynamic_result = None
        self._last_dynamic_config = None
        self._pending_dynamic_load_source = None
        self._last_dynamic_load_source = None
        self._last_dynamic_transition_result = None
        self._last_dynamic_transition_config = None
        self._last_dynamic_transition_load_source = None
        self._last_dynamic_mode = None
        self._last_dynamic_stress = None
        self._last_dipped_joint_result = None
        self._last_dipped_joint_config = None
        self._clear_dynamic_overlays(render=False)

    def _clear_special_state(self) -> None:
        self._last_special_result = None
        self._last_special_config = None

    def _clear_result_views_for_new_run(self) -> None:
        if hasattr(self, "_chart_registry"):
            for entry in self._chart_registry:
                entry.plot_panel.clear_plot()
        for panel_name in (
            "summary_panel",
            "transition_summary_panel",
            "dynamic_summary_panel",
            "dipped_joint_summary_panel",
            "special_summary_panel",
        ):
            panel = getattr(self, panel_name, None)
            fields = getattr(panel, "_fields", None)
            if isinstance(fields, dict):
                for label in fields.values():
                    label.setText("—")
                    label.setStyleSheet("")
            sanity_fields = getattr(panel, "_sanity_fields", None)
            if isinstance(sanity_fields, dict):
                for label in sanity_fields.values():
                    label.setText("—")
                    label.setStyleSheet("color: #666666;")
            interpretation = getattr(panel, "interpretation_label", None)
            if isinstance(interpretation, QLabel):
                interpretation.setText("—")
        self._last_rendered_stress = None
        self._last_stress_title = "Stress"
        self._last_stress_unavailable_note = None
        self._sync_stress_series_controls(None, reset_visibility=True)
        self._clear_chart_thumbnails()

    def _is_analysis_running(self) -> bool:
        thread = self.thread
        if thread is None:
            return False
        try:
            return thread.isRunning()
        except RuntimeError:
            return False

    def _clear_runtime_results(self) -> None:
        self._clear_analysis_state()
        self._clear_dynamic_state()
        self._clear_special_state()
        self._clear_all_overlays(render=False)
        self._clear_result_views_for_new_run()
        self._last_single_tab_index = self.deflection_tab_index if hasattr(self, "deflection_tab_index") else None
        if hasattr(self, "chart_view_combo"):
            self._set_chart_view_combo("single")
        if hasattr(self, "tab_widget"):
            self.tab_widget.setTabVisible(self.rail_deflection_tab_index, False)
            self.tab_widget.setTabVisible(self.rail_moment_tab_index, False)
            self.tab_widget.setTabVisible(self.transition_profile_tab_index, False)
            self.tab_widget.setTabVisible(self.transition_summary_tab_index, False)
            self._set_dynamic_tabs_visible(False)
            self._set_special_tabs_visible(False)
            self.tab_widget.setTabVisible(self.stress_tab_index, True)
            self._apply_chart_view("single", target_tab=self.deflection_tab_index)
        self._set_export_buttons_enabled(False)
        self._set_transition_export_buttons_enabled(False)
        self._set_dynamic_export_buttons_enabled(False)
        self._refresh_alternative_action_buttons_for_selection()
        self._update_overlay_state()

    def _reset_application(self) -> None:
        if self._is_analysis_running():
            QMessageBox.information(
                self,
                "Analysis running",
                "Cancel the running analysis before resetting the application.",
            )
            return
        for combo_name in (
            "analysis_type_combo",
            "static_mode_combo",
            "dynamic_mode_combo",
            "special_mode_combo",
            "profile_type_combo",
            "foundation_model_combo",
            "beam_theory_combo",
        ):
            combo = getattr(self, combo_name, None)
            if isinstance(combo, QComboBox) and combo.count() > 0:
                combo.setCurrentIndex(0)
        for checkbox_name in (
            "several_loads_checkbox",
            "train_loads_checkbox",
            "as5100_loads_checkbox",
            "transition_group",
            "advanced_solver_checkbox",
            "two_rail_checkbox",
            "asymmetric_load_checkbox",
            "pasternak_checkbox",
            "nonuniform_profile_checkbox",
            "overlay_checkbox",
        ):
            checkbox = getattr(self, checkbox_name, None)
            if isinstance(checkbox, QCheckBox):
                checkbox.setChecked(False)
        defaults = {
            "load_magnitude_input": 100.0,
            "load_position_input": 0.0,
            "sleeper_spacing_input": 600.0,
            "ballast_thickness_input": 300.0,
            "train_axle_load_input": 100.0,
            "train_bogie_spacing_input": 2500.0,
            "train_axle_spacing_input": 1600.0,
            "as5100_group_spacing_input": 12.0,
            "as5100_reference_input": 0.0,
            "right_load_magnitude_input": 100.0,
            "right_load_position_input": 0.0,
        }
        for input_name, value in defaults.items():
            unit_input = getattr(self, input_name, None)
            if hasattr(unit_input, "set_value"):
                unit_input.set_value(value)
        if hasattr(self, "train_bogie_count_input"):
            self.train_bogie_count_input.setValue(2)
        if hasattr(self, "train_axles_per_bogie_input"):
            self.train_axles_per_bogie_input.setValue(2)
        if hasattr(self, "as5100_model_combo"):
            self.as5100_model_combo.setCurrentIndex(self.as5100_model_combo.findData(AS5100_MODEL_300LA))
        if hasattr(self, "as5100_arrangement_mode_combo"):
            self.as5100_arrangement_mode_combo.setCurrentIndex(
                self.as5100_arrangement_mode_combo.findData(AS5100ArrangementMode.FIXED_SELECTED)
            )
        if hasattr(self, "as5100_group_count_input"):
            self.as5100_group_count_input.setValue(2)
        if hasattr(self, "wheel_loads_widget"):
            self.wheel_loads_widget.clear()
        for checkbox_name in (
            "chart_input_labels_checkbox",
            "chart_output_labels_checkbox",
            "chart_extrema_labels_checkbox",
        ):
            checkbox = getattr(self, checkbox_name, None)
            if isinstance(checkbox, QCheckBox):
                was_blocked = checkbox.blockSignals(True)
                checkbox.setChecked(True)
                checkbox.blockSignals(was_blocked)
        self._clear_runtime_results()
        self.run_button.setEnabled(True)
        self._set_cancel_enabled(False)
        self.statusBar().showMessage("Application reset to starting state.")

    def _format_static_method_label(self) -> str:
        config = self._last_analysis_config
        mode = self._last_analysis_mode
        if config is None:
            return "Static"
        beam = "Euler-Bernoulli" if config.beam_theory == BeamTheory.EULER else "Timoshenko"
        foundation_map = {
            FoundationModelType.WINKLER: "Winkler",
            FoundationModelType.SERIES: "Series",
            FoundationModelType.SLEEPER_MASS: "Sleeper-mass",
        }
        foundation = foundation_map.get(config.foundation_model, "Foundation")
        extras: list[str] = []
        if config.pasternak_shear_n > 0:
            extras.append("Pasternak")
        if config.use_discrete_supports:
            extras.append("Discrete supports")
        if config.use_two_rail:
            extras.append("Two-rail")
        detail_parts = [beam, foundation, *extras]
        backend = "closed-form" if mode in (None, AnalysisMode.CLOSED_FORM) else "numerical"
        return f"Static - {', '.join(detail_parts)} ({backend})"

    def _format_dynamic_method_label(self) -> str:
        mode = self._last_dynamic_mode or self.dynamic_mode_combo.currentData()
        if mode == DynamicMode.DIPPED_JOINT:
            return "Dynamic - Dipped joint (wheel/rail forces)"
        if mode == DynamicMode.TRANSITION:
            return "Dynamic - Transition mode"
        mode_map = {
            DynamicMode.TIME_HISTORY: "Time-history (explicit)",
            DynamicMode.STEADY_STATE: "Steady-state moving load (travelling wave solution)",
        }
        mode_text = mode_map.get(mode, "Moving load")
        label = f"Dynamic - Moving load, {mode_text}"
        if self._last_dynamic_config is not None:
            damping = self._last_dynamic_config.foundation_damping_model
            damping_text = "viscous damping" if damping == DampingModel.VISCOUS else "hysteretic damping"
            label = f"{label}, {damping_text}"
        return label

    @staticmethod
    def _overlay_badge_style() -> dict[str, object]:
        return {
            "fontsize": 7.6,
            "color": "#2f2f2f",
            "bbox": {
                "facecolor": "white",
                "alpha": 0.34,
                "edgecolor": "none",
                "boxstyle": "round,pad=0.2",
            },
        }

    @staticmethod
    def _compact_name(value: object, fallback: str) -> str:
        name = getattr(value, "name", None)
        if not name:
            return fallback
        text = str(name).strip()
        return text if len(text) <= 28 else f"{text[:25]}..."

    @staticmethod
    def _format_load_summary(loads: Sequence[PointLoad] | None) -> str:
        if not loads:
            return "not set"
        load_values = [n_to_kn(abs(load.load_newtons)) for load in loads]
        positions = [load.position_m for load in loads]
        if len(loads) == 1:
            return f"{load_values[0]:.1f} kN @ x={positions[0]:.3f} m"
        return (
            f"{len(loads)} loads, max={max(load_values):.1f} kN, "
            f"x={min(positions):.3f} to {max(positions):.3f} m"
        )

    @staticmethod
    def _format_load_source_summary(load_source: dict[str, object] | None) -> str | None:
        if not load_source or load_source.get("load_basis") != "axle_load_split_to_two_rails":
            return None
        axle_kn = n_to_kn(float(load_source.get("max_axle_load_n", 0.0) or 0.0))
        wheel_kn = n_to_kn(float(load_source.get("max_wheel_load_n_per_rail", 0.0) or 0.0))
        count = int(load_source.get("axle_count", 0) or 0)
        return f"{count} axle loads, max axle={axle_kn:.1f} kN -> wheel={wheel_kn:.1f} kN/rail"

    def _load_source_for_input_config(self, config: object | None) -> dict[str, object] | None:
        if isinstance(config, AnalysisConfig):
            return self._load_source_for_analysis_config(config)
        if config is self._last_dynamic_config:
            return self._last_dynamic_load_source
        if config is self._last_dynamic_transition_config:
            return self._last_dynamic_transition_load_source
        nested = getattr(config, "analysis_config", None)
        if isinstance(nested, AnalysisConfig):
            return self._load_source_for_analysis_config(nested)
        return None

    @staticmethod
    def _config_value(config: object | None, name: str) -> object | None:
        if config is None:
            return None
        direct = getattr(config, name, None)
        if direct is not None:
            return direct
        nested = getattr(config, "analysis_config", None)
        if nested is None:
            return None
        return getattr(nested, name, None)

    def _active_loads_for_annotations(self, config: object | None = None) -> Sequence[PointLoad] | None:
        for candidate in (
            config,
            self._last_analysis_config,
            self._last_envelope_config,
            self._last_dynamic_config,
            self._last_dynamic_transition_config,
        ):
            loads = self._config_value(candidate, "loads")
            if loads:
                return loads
        return None

    def _build_input_parameter_lines(self, *, config: object | None = None) -> list[str]:
        rail = self.rail_combo.currentData() if hasattr(self, "rail_combo") else None
        sleeper = self.sleeper_combo.currentData() if hasattr(self, "sleeper_combo") else None
        support = self.support_combo.currentData() if hasattr(self, "support_combo") else None

        rail_name = self._compact_name(rail, "Rail")
        sleeper_name = self._compact_name(sleeper, "Sleeper")
        support_name = self._compact_name(support, "Ballast")

        rail_e_gpa = getattr(rail, "elastic_modulus_pa", None)
        rail_i = getattr(rail, "moment_inertia_m4", None)
        sleeper_spacing_m = self._config_value(config, "sleeper_spacing_m")
        if sleeper_spacing_m is None and hasattr(self, "sleeper_spacing_input"):
            sleeper_spacing_m = mm_to_m(self.sleeper_spacing_input.value())
        ballast_k = self._config_value(config, "foundation_modulus_n_per_m2")
        if ballast_k is None:
            ballast_k = getattr(support, "foundation_modulus_n_per_m2", None)

        rail_detail = rail_name
        if rail_e_gpa is not None and rail_i is not None:
            rail_detail = f"{rail_name}, E={rail_e_gpa / 1.0e9:.1f} GPa, I={rail_i:.2e} m4"

        sleeper_detail = sleeper_name
        if sleeper_spacing_m is not None:
            sleeper_detail = f"{sleeper_name}, spacing={float(sleeper_spacing_m):.3f} m"

        ballast_detail = support_name
        if ballast_k is not None:
            ballast_detail = f"{support_name}, k={n_per_m2_to_mn_per_m2(float(ballast_k)):.2f} MN/m²"

        load_summary = self._format_load_source_summary(
            self._load_source_for_input_config(config)
        ) or self._format_load_summary(self._active_loads_for_annotations(config))

        return [
            f"Rail: {rail_detail}",
            f"Sleeper: {sleeper_detail}",
            f"Ballast: {ballast_detail}",
            f"Load: {load_summary}",
        ]

    @staticmethod
    def _chart_location_note(chart_title: str) -> str:
        title = chart_title.lower()
        if "reaction" in title:
            return "Level: rail support line reaction, not ballast pressure"
        if "foundation profile" in title or "foundation modulus" in title or "k(x)" in title:
            return "Level: support stiffness profile used by the foundation model"
        if "damping force" in title:
            return "Level: viscous/hysteretic support damping force per rail length"
        if "fft" in title or "psd" in title:
            return "Level: rail deflection frequency content"
        if "impedance" in title:
            return "Level: support/foundation dynamic impedance"
        if "sleeper" in title and "pressure" in title:
            return "Level: sleeper-ballast contact pressure (top of ballast)"
        if "sleeper" in title and ("load" in title or "seat" in title):
            return "Level: sleeper/rail-seat demand from integrated reaction"
        if "ballast" in title or "formation" in title:
            return "Level: ballast top pressure and formation depth stress"
        if "stress" in title:
            return "Level: rail section stress"
        if any(term in title for term in ("deflection", "moment", "shear")):
            return "Level: rail response along the track"
        return "Level: chart-specific response"

    def _build_dual_overlay_annotations(
        self,
        *,
        metadata_lines: Sequence[str],
        kpi_lines: Sequence[str],
        metadata_anchor: tuple[float, float] = (0.02, 0.98),
        metadata_alignment: tuple[str, str] = ("left", "top"),
        kpi_anchor: tuple[float, float] = (0.98, 0.98),
        kpi_alignment: tuple[str, str] = ("right", "top"),
    ) -> list[tuple[str, tuple[float, float], dict[str, object]]]:
        if not metadata_lines and not kpi_lines:
            return []
        style = self._overlay_badge_style()
        annotations: list[tuple[str, tuple[float, float], dict[str, object]]] = []
        if metadata_lines and self._chart_input_labels_visible():
            annotations.append(
                (
                    "\n".join(metadata_lines),
                    metadata_anchor,
                    {**style, "ha": metadata_alignment[0], "va": metadata_alignment[1]},
                )
            )
        if kpi_lines and self._chart_output_labels_visible():
            annotations.append(
                (
                    "\n".join(kpi_lines),
                    kpi_anchor,
                    {**style, "ha": kpi_alignment[0], "va": kpi_alignment[1]},
                )
            )
        return annotations

    def _build_static_chart_annotations(
        self,
        result: AnalysisResult,
        *,
        chart_title: str,
    ) -> list[tuple[str, tuple[float, float], dict[str, object]]]:
        summary = result.summary
        metadata_lines = [
            "Analysis: Static",
            f"Method: {self._format_static_method_label()}",
            f"Chart: {chart_title}",
            self._chart_location_note(chart_title),
        ]
        if self._active_track_config_name:
            metadata_lines.append(f"Track config: {self._active_track_config_name}")
        metadata_lines.extend(self._build_static_load_source_lines(self._last_analysis_load_source))
        metadata_lines.extend(self._build_input_parameter_lines(config=self._last_analysis_config))
        kpi_lines = [
            f"|w|max: {m_to_mm(abs(summary.max_deflection.value)):.3f} mm @ x={summary.max_deflection.position_m:.3f} m",
            f"|M|max: {abs(summary.max_moment.value) / 1000.0:.3f} kN·m @ x={summary.max_moment.position_m:.3f} m",
            f"|V|max: {n_to_kn(abs(summary.max_shear.value)):.3f} kN @ x={summary.max_shear.position_m:.3f} m",
            f"|R|max: {n_to_kn(abs(summary.max_reaction.value)):.3f} kN/m @ x={summary.max_reaction.position_m:.3f} m",
        ]
        design = summary.design_summary
        if design is not None:
            kpi_lines.append(f"DAF: {design.daf:.3f}")
        return self._build_dual_overlay_annotations(metadata_lines=metadata_lines, kpi_lines=kpi_lines)

    def _build_static_load_source_lines(
        self,
        load_source: dict[str, object] | None,
    ) -> list[str]:
        if not load_source:
            return []
        if load_source.get("source_type") == "as5100_fixed_rail":
            standard = load_source.get("standard", "AS5100.2:2017")
            model = load_source.get("model", "AS5100")
            arrangement = str(load_source.get("arrangement", "fixed_user_selected"))
            reference_position = float(load_source.get("reference_position_m", 0.0) or 0.0)
            max_axle_kn = n_to_kn(float(load_source.get("max_axle_load_n", 0.0) or 0.0))
            max_wheel_kn = n_to_kn(float(load_source.get("max_wheel_load_n_per_rail", 0.0) or 0.0))
            if arrangement == "governing_envelope_sweep":
                return [
                    f"Load source: AS5100 {model} governing sweep",
                    f"Standard: {standard}; x0={reference_position:.3f} m",
                    (
                        f"Governing arrangement: {int(load_source.get('group_count', 0) or 0)} group(s) @ "
                        f"{float(load_source.get('group_spacing_m', 0.0) or 0.0):.2f} m"
                    ),
                    (
                        f"Selected upper bound: {int(load_source.get('selected_group_count', 0) or 0)} group(s) @ "
                        f"{float(load_source.get('selected_group_spacing_m', 0.0) or 0.0):.2f} m"
                    ),
                    (
                        f"Candidates: {int(load_source.get('sweep_candidate_count', 0) or 0)}; "
                        "governing metric = max |M|"
                    ),
                    f"Load basis: axle {max_axle_kn:.0f} kN -> wheel {max_wheel_kn:.0f} kN/rail",
                    "AS5100 scope: vertical arrangement; no automatic DLA applied",
                ]
            return [
                f"Load source: AS5100 {model} fixed arrangement",
                f"Standard: {standard}; x0={reference_position:.3f} m",
                (
                    f"AS5100 axles: {int(load_source.get('axle_count', 0) or 0)}, "
                    f"{int(load_source.get('group_count', 0) or 0)} group(s) @ "
                    f"{float(load_source.get('group_spacing_m', 0.0) or 0.0):.2f} m"
                ),
                f"Load basis: axle {max_axle_kn:.0f} kN -> wheel {max_wheel_kn:.0f} kN/rail",
                "AS5100 scope: fixed selected vertical arrangement; no automatic DLA applied",
            ]
        if load_source.get("source_type") == "train_builder":
            axle_count = int(load_source.get("axle_count", 0) or 0)
            axle_kn = n_to_kn(float(load_source.get("max_axle_load_n", 0.0) or 0.0))
            wheel_kn = n_to_kn(float(load_source.get("max_wheel_load_n_per_rail", 0.0) or 0.0))
            bogies = int(load_source.get("bogie_count", 0) or 0)
            axles_per_bogie = int(load_source.get("axles_per_bogie", 0) or 0)
            return [
                "Load source: Train/axle builder",
                f"Axles: {axle_count}; {bogies} bogie(s) x {axles_per_bogie} axle(s)",
                f"Load basis: axle {axle_kn:.0f} kN -> wheel {wheel_kn:.0f} kN/rail",
            ]
        source_label = self._format_source_type_label(load_source.get("source_type"))
        arrangement = load_source.get("arrangement")
        if arrangement:
            return [f"Load source: {source_label} ({arrangement})"]
        return [f"Load source: {source_label}"]

    def _build_envelope_chart_annotations(
        self,
        result: EnvelopeResult,
        *,
        chart_title: str,
    ) -> list[tuple[str, tuple[float, float], dict[str, object]]]:
        summary = result.summary
        metadata_lines = [
            "Analysis: Static envelope",
            f"Method: {self._format_static_method_label()}",
            f"Chart: {chart_title}",
            self._chart_location_note(chart_title),
        ]
        if self._active_track_config_name:
            metadata_lines.append(f"Track config: {self._active_track_config_name}")
        metadata_lines.extend(self._build_static_load_source_lines(self._last_envelope_load_source))
        metadata_lines.extend(self._build_input_parameter_lines(config=self._last_envelope_config))
        kpi_lines = [
            f"|w|max: {m_to_mm(abs(summary.max_deflection.value)):.3f} mm @ x={summary.max_deflection.position_m:.3f} m",
            f"|M|max: {abs(summary.max_moment.value) / 1000.0:.3f} kN·m @ x={summary.max_moment.position_m:.3f} m",
            f"|V|max: {n_to_kn(abs(summary.max_shear.value)):.3f} kN @ x={summary.max_shear.position_m:.3f} m",
            f"|R|max: {n_to_kn(abs(summary.max_reaction.value)):.3f} kN/m @ x={summary.max_reaction.position_m:.3f} m",
        ]
        design = summary.design_summary
        if design is not None:
            kpi_lines.append(f"DAF: {design.daf:.3f}")
        return self._build_dual_overlay_annotations(metadata_lines=metadata_lines, kpi_lines=kpi_lines)

    def _build_transition_chart_annotations(
        self,
        result: TransitionRunResult,
        *,
        chart_title: str,
    ) -> list[tuple[str, tuple[float, float], dict[str, object]]]:
        metrics = result.metrics
        mode_text = _enum_value(result.mode).replace("_", " ").title()
        profile_text = _enum_value(result.profile_type).replace("_", " ").title()
        k2_text = "—" if result.k2_n_per_m2 is None else f"{n_per_m2_to_mn_per_m2(result.k2_n_per_m2):.2f} MN/m²"
        transition_length_text = "—" if result.transition_length_m is None else f"{result.transition_length_m:.3f} m"
        segment_length_text = "—" if result.segment_length_m is None else f"{result.segment_length_m:.3f} m"
        metadata_lines = [
            "Analysis: Transition (static)",
            f"Method: {self._format_static_method_label()}",
            f"Chart: {chart_title}",
            self._chart_location_note(chart_title),
            f"Template: {result.template_name or 'Custom'}",
            f"Preset: {result.preset_name or 'Custom'}",
            f"Mode: {mode_text}",
            f"Profile: {profile_text}",
            f"k1: {n_per_m2_to_mn_per_m2(result.k1_n_per_m2):.2f} MN/m²",
            f"k2: {k2_text}",
            f"Lt: {transition_length_text}",
            f"Lc: {segment_length_text}",
            f"Domain: {result.domain_length_m:.3f} m",
        ]
        metadata_lines.extend(self._build_static_load_source_lines(self._last_transition_load_source))
        metadata_lines.extend(
            self._build_input_parameter_lines(config=self._last_analysis_config or self._last_envelope_config)
        )
        kpi_lines = [
            f"Δw(s): {m_to_mm(metrics.delta_w_s_m):.3f} mm @ x={metrics.delta_w_s_position_m:.3f} m",
            f"Δw(1m): {m_to_mm(metrics.delta_w_1m_m):.3f} mm @ x={metrics.delta_w_1m_position_m:.3f} m",
            f"κ_max: {metrics.curvature_max_per_m:.3e} 1/m @ x={metrics.curvature_max_position_m:.3f} m",
            f"|M|max: {abs(metrics.moment_max_nm) / 1000.0:.3f} kN·m @ x={metrics.moment_max_position_m:.3f} m",
            f"U_b: {metrics.energy_bending_j / 1000.0:.3f} kJ",
            f"Max |dp/dx|: {abs(metrics.reaction_gradient_max_n_per_m2) / 1000.0:.3f} kN/m² @ x={metrics.reaction_gradient_position_m:.3f} m",
            f"|Q_s|max: {n_to_kn(abs(metrics.sleeper_load_max_n)):.3f} kN @ x={metrics.sleeper_load_position_m:.3f} m",
        ]
        if result.energy_metrics is not None:
            kpi_lines.append(f"U_total: {result.energy_metrics.energy_total_j / 1000.0:.3f} kJ")
            kpi_lines.append(f"η: {result.energy_metrics.energy_partition_eta:.3f}")
            kpi_lines.append(
                f"u_max: {result.energy_metrics.u_total_max_j_per_m:.3f} J/m @ x={result.energy_metrics.u_total_max_position_m:.3f} m"
            )
            kpi_lines.append(
                f"Max |du/dx|: {result.energy_metrics.du_dx_max_j_per_m2:.3f} J/m² @ x={result.energy_metrics.du_dx_max_position_m:.3f} m"
            )
            if result.energy_metrics.is_envelope_upper_bound:
                kpi_lines.append("Energy semantics: envelope upper-bound proxy")
            if result.energy_metrics.boundary_peak_flag or result.energy_metrics.boundary_gradient_peak_flag:
                kpi_lines.append("Boundary artifact: energy peak at domain edge")
        else:
            kpi_lines.append("Energy metrics unavailable for non-Winkler foundation model")
        return self._build_dual_overlay_annotations(
            metadata_lines=metadata_lines,
            kpi_lines=kpi_lines,
            metadata_anchor=(0.02, 0.02),
            metadata_alignment=("left", "bottom"),
            kpi_anchor=(0.98, 0.02),
            kpi_alignment=("right", "bottom"),
        )

    def _apply_transition_annotations_to_visible_charts(self, result: TransitionRunResult) -> None:
        chart_panels: list[tuple[PlotPanel, str]] = [
            (self.deflection_plot, "Deflection"),
            (self.moment_plot, "Bending moment"),
            (self.shear_plot, "Shear"),
            (self.reaction_plot, "Rail support reaction"),
            (self.sleeper_plot, "Sleeper loads"),
            (self.pressure_plot, "Pressures"),
            (self.transition_profile_plot, "Transition profile k(x)"),
        ]
        if self.tab_widget.isTabVisible(self.rail_deflection_tab_index):
            chart_panels.append((self.rail_deflection_plot, "Rail deflection comparison"))
        if self.tab_widget.isTabVisible(self.rail_moment_tab_index):
            chart_panels.append((self.rail_moment_plot, "Rail moment comparison"))
        for panel, chart_title in chart_panels:
            if not panel.rendered_series():
                continue
            self._apply_panel_annotations(
                panel,
                self._build_transition_chart_annotations(result, chart_title=chart_title),
            )

    def _build_special_chart_annotations(
        self,
        result: FloatingSlabResult,
        *,
        chart_title: str,
    ) -> list[tuple[str, tuple[float, float], dict[str, object]]]:
        metadata_lines = [
            "Analysis: Special",
            "Method: Floating slab SDOF",
            f"Chart: {chart_title}",
            self._chart_location_note(chart_title),
        ]
        metadata_lines.extend(self._build_input_parameter_lines(config=None))
        kpi_lines = [
            f"f_n: {result.natural_frequency_hz:.3f} Hz",
            f"ζ: {result.damping_ratio:.3f}",
            f"δ_static: {m_to_mm(result.static_deflection_m):.3f} mm",
        ]
        return self._build_dual_overlay_annotations(metadata_lines=metadata_lines, kpi_lines=kpi_lines)

    def _dynamic_annotation_mode(self) -> DynamicAnnotationMode:
        if not hasattr(self, "dynamic_annotation_mode_combo"):
            return DynamicAnnotationMode.FULL
        mode = self.dynamic_annotation_mode_combo.currentData()
        return mode if isinstance(mode, DynamicAnnotationMode) else DynamicAnnotationMode.FULL

    def _dynamic_load_source_for_annotations(self) -> dict[str, object] | None:
        load_source = self._last_dynamic_load_source_metadata()
        if load_source is not None:
            return load_source
        if hasattr(self, "as5100_loads_checkbox") and self.as5100_loads_checkbox.isChecked():
            return self._current_load_source_metadata()
        return None

    @staticmethod
    def _format_source_type_label(source_type: object) -> str:
        source = str(source_type or "").replace("_", " ").strip()
        return source[:1].upper() + source[1:] if source else "Load source"

    def _build_dynamic_load_source_lines(self, *, compact: bool) -> list[str]:
        load_source = self._dynamic_load_source_for_annotations()
        if not load_source:
            return []
        if load_source.get("source_type") == "as5100_fixed_rail":
            standard = load_source.get("standard", "AS5100.2:2017")
            model = load_source.get("model", "AS5100")
            axle_count = int(load_source.get("axle_count", 0) or 0)
            max_axle_kn = n_to_kn(float(load_source.get("max_axle_load_n", 0.0) or 0.0))
            max_wheel_kn = n_to_kn(float(load_source.get("max_wheel_load_n_per_rail", 0.0) or 0.0))
            group_count = int(load_source.get("group_count", 0) or 0)
            group_spacing = float(load_source.get("group_spacing_m", 0.0) or 0.0)
            reference_position = float(load_source.get("reference_position_m", 0.0) or 0.0)
            if compact:
                return [
                    f"Load source: AS5100 {model} fixed ({standard})",
                    (
                        f"Axles: {axle_count}, max {max_axle_kn:.0f} kN; "
                        f"{group_count} group(s) @ {group_spacing:.2f} m"
                    ),
                    f"Solver load: {max_wheel_kn:.0f} kN/rail",
                    f"x0={reference_position:.3f} m; no automatic DLA",
                ]
            return [
                f"Load source: AS5100 {model} fixed arrangement",
                f"Standard: {standard}",
                (
                        f"AS5100 axles: {axle_count}, max {max_axle_kn:.0f} kN; "
                        f"{group_count} group(s) @ {group_spacing:.2f} m"
                ),
                f"Load basis: axle {max_axle_kn:.0f} kN -> wheel {max_wheel_kn:.0f} kN/rail",
                f"AS5100 reference x0: {reference_position:.3f} m",
                "AS5100 scope: fixed selected vertical arrangement; no automatic DLA applied",
            ]
        if load_source.get("source_type") == "train_builder":
            axle_count = int(load_source.get("axle_count", 0) or 0)
            axle_kn = n_to_kn(float(load_source.get("max_axle_load_n", 0.0) or 0.0))
            wheel_kn = n_to_kn(float(load_source.get("max_wheel_load_n_per_rail", 0.0) or 0.0))
            if compact:
                return [
                    "Load source: Train/axle builder",
                    f"Axles: {axle_count}, axle {axle_kn:.0f} kN -> wheel {wheel_kn:.0f} kN/rail",
                ]
            return [
                "Load source: Train/axle builder",
                f"Axles: {axle_count}",
                f"Load basis: axle {axle_kn:.0f} kN -> wheel {wheel_kn:.0f} kN/rail",
            ]
        source_label = self._format_source_type_label(load_source.get("source_type"))
        arrangement = load_source.get("arrangement")
        if arrangement:
            return [f"Load source: {source_label} ({arrangement})"]
        return [f"Load source: {source_label}"]

    def _build_dynamic_global_summary_lines(self) -> list[str]:
        if self._last_dynamic_result is None:
            return []
        summary = self._last_dynamic_result.summary
        lines = [
            f"Global |w|: {m_to_mm(abs(summary.max_deflection.value)):.3f} mm",
            f"Global |M|: {abs(summary.max_moment.value) / 1000.0:.3f} kN·m",
            f"Global |V|: {n_to_kn(abs(summary.max_shear.value)):.3f} kN",
            f"Global |R|: {n_to_kn(abs(summary.max_reaction.value)):.3f} kN/m",
        ]
        if self._last_dynamic_transition_result is not None:
            lines.append(f"Risk index: {self._last_dynamic_transition_result.metrics.risk_index:.3f}")
        return lines

    def _build_dynamic_parameter_trace_lines(self, *, compact: bool = False) -> list[str]:
        if self._last_dynamic_result is None or self._last_dynamic_result.parameter_trace is None:
            return []
        trace = self._last_dynamic_result.parameter_trace
        if compact:
            return [
                (
                    f"DAF: {trace.dynamic_amplification:.3f}; "
                    f"v/vcr: {trace.critical_speed_ratio:.3f}; "
                    f"vcr: {trace.critical_speed_m_per_s:.2f} m/s"
                )
            ]
        return [
            (
                f"EI: {trace.flexural_rigidity_nm2 / 1.0e6:.3f} MN·m²; "
                f"k: {n_per_m2_to_mn_per_m2(trace.foundation_modulus_n_per_m2):.2f} MN/m²"
            ),
            (
                f"c: {trace.foundation_damping_n_s_per_m2 / 1000.0:.3f} kN·s/m²; "
                f"ζ: {trace.damping_ratio:.4f}; m: {trace.mass_kg_per_m:.2f} kg/m"
            ),
            (
                f"β: {trace.beta_per_m:.6f} 1/m; "
                f"Lc: {trace.characteristic_length_m:.3f} m; Δξ: {trace.spatial_step_m:.4f} m"
            ),
            (
                f"DAF: {trace.dynamic_amplification:.3f}; "
                f"v/vcr: {trace.critical_speed_ratio:.3f}; "
                f"vcr: {trace.critical_speed_m_per_s:.2f} m/s"
            ),
        ]

    def _build_dynamic_metadata_lines(self, *, compact: bool = False) -> list[str]:
        analysis_type = self.analysis_type_combo.currentText()
        mode_value = self._last_dynamic_mode
        if mode_value is None:
            mode_data = self.dynamic_mode_combo.currentData()
            mode_value = mode_data if isinstance(mode_data, DynamicMode) else None
        mode_text_map: dict[DynamicMode, str] = {
            DynamicMode.STEADY_STATE: "Steady-state",
            DynamicMode.TIME_HISTORY: "Time-history",
            DynamicMode.TRANSITION: "Transition",
            DynamicMode.DIPPED_JOINT: "Dipped joint",
        }
        mode_text = mode_text_map.get(mode_value, self.dynamic_mode_combo.currentText())

        speed = self.speed_input.value()
        load_kn = self.load_magnitude_input.value()
        load_label = "Wheel load"
        solver_load_line: str | None = None
        load_source = self._dynamic_load_source_for_annotations()
        if load_source and load_source.get("load_basis") == "axle_load_split_to_two_rails":
            load_kn = n_to_kn(float(load_source.get("max_axle_load_n", 0.0) or 0.0))
            wheel_kn = n_to_kn(float(load_source.get("max_wheel_load_n_per_rail", 0.0) or 0.0))
            load_label = "Max axle load"
            solver_load_line = f"Wheel load to solver: {wheel_kn:.2f} kN/rail"
        if mode_value == DynamicMode.TRANSITION and self._last_dynamic_transition_config is not None:
            speed = self._last_dynamic_transition_config.speed_m_per_s
            if self._last_dynamic_transition_config.loads and solver_load_line is None:
                load_kn = max(load.load_newtons for load in self._last_dynamic_transition_config.loads) / 1000.0
        elif self._last_dynamic_config is not None:
            speed = self._last_dynamic_config.speed_m_per_s
            if self._last_dynamic_config.loads and solver_load_line is None:
                load_kn = max(load.load_newtons for load in self._last_dynamic_config.loads) / 1000.0
        elif (
            solver_load_line is None
            and hasattr(self, "as5100_loads_checkbox")
            and self.as5100_loads_checkbox.isChecked()
        ):
            loads = self._collect_as5100_loads()
            if loads:
                load_kn = max(load.load_newtons for load in loads) / 1000.0
        elif solver_load_line is None and self.train_loads_checkbox.isChecked():
            load_kn = self.train_axle_load_input.value()
            load_label = "Axle load"

        metadata_lines = [
            f"Analysis: {analysis_type}",
            f"Method: {self._format_dynamic_method_label()}",
            f"Mode: {mode_text}",
            f"Speed: {speed:.2f} m/s",
            f"{load_label}: {load_kn:.2f} kN",
        ]
        if solver_load_line is not None:
            metadata_lines.append(solver_load_line)
        metadata_lines.extend(self._build_dynamic_load_source_lines(compact=compact))
        if self._active_track_config_name:
            metadata_lines.append(f"Track config: {self._active_track_config_name}")
        if not compact:
            metadata_lines.extend(
                self._build_input_parameter_lines(
                    config=self._last_dynamic_config or self._last_dynamic_transition_config
                )
            )
            metadata_lines.extend(self._build_dynamic_parameter_trace_lines())
        return metadata_lines

    def _build_dynamic_chart_annotations(
        self,
        x_values: Sequence[float],
        y_values: Sequence[float],
        *,
        value_symbol: str,
        value_unit: str,
        axis_symbol: str,
        axis_unit: str,
        chart_title: str,
        extra_metadata_lines: Sequence[str] | None = None,
    ) -> list[tuple[str, tuple[float, float], dict[str, object]]]:
        annotation_mode = self._dynamic_annotation_mode()
        if annotation_mode == DynamicAnnotationMode.OFF:
            return []
        if not x_values or not y_values:
            return []
        compact = annotation_mode == DynamicAnnotationMode.COMPACT

        max_index = max(range(len(y_values)), key=y_values.__getitem__)
        min_index = min(range(len(y_values)), key=y_values.__getitem__)
        abs_index = max(range(len(y_values)), key=lambda idx: abs(y_values[idx]))
        w_max = y_values[max_index]
        w_min = y_values[min_index]
        xi_at_max = x_values[max_index]
        xi_at_min = x_values[min_index]
        abs_peak = abs(y_values[abs_index])
        axis_at_abs = x_values[abs_index]
        xi_min = min(x_values)
        xi_max = max(x_values)

        metadata_lines = self._build_dynamic_metadata_lines(compact=compact)
        metadata_lines.append(f"Chart: {chart_title}")
        metadata_lines.append(self._chart_location_note(chart_title))
        if extra_metadata_lines:
            metadata_lines.extend(extra_metadata_lines)
        metadata_text = "\n".join(metadata_lines)
        abs_symbol = (
            value_symbol
            if value_symbol.startswith("|") and value_symbol.endswith("|")
            else f"|{value_symbol}|"
        )

        if compact:
            kpi_lines = [
                f"{abs_symbol}_max: {abs_peak:.3f} {value_unit} @ {axis_symbol}={axis_at_abs:.3f} {axis_unit}",
                *self._build_dynamic_parameter_trace_lines(compact=True),
                *self._build_dynamic_global_summary_lines(),
            ]
        else:
            kpi_lines = [
                f"{value_symbol}_max: {w_max:.3f} {value_unit} @ {axis_symbol}={xi_at_max:.3f} {axis_unit}",
                f"{value_symbol}_min: {w_min:.3f} {value_unit} @ {axis_symbol}={xi_at_min:.3f} {axis_unit}",
                f"{abs_symbol}_max: {abs_peak:.3f} {value_unit} @ {axis_symbol}={axis_at_abs:.3f} {axis_unit}",
                f"{axis_symbol} range: {xi_min:.3f} to {xi_max:.3f} {axis_unit}",
                f"{value_symbol} range: {w_min:.3f} to {w_max:.3f} {value_unit}",
                *self._build_dynamic_global_summary_lines(),
            ]
        extrema_text = "\n".join(kpi_lines)

        common_style = self._overlay_badge_style()
        annotations: list[tuple[str, tuple[float, float], dict[str, object]]] = []
        if self._chart_input_labels_visible():
            annotations.append(
                (
                    metadata_text,
                    (0.02, 0.98),
                    {**common_style, "ha": "left", "va": "top"},
                )
            )
        if self._chart_output_labels_visible():
            annotations.append(
                (
                    extrema_text,
                    (0.98, 0.98),
                    {**common_style, "ha": "right", "va": "top"},
                )
            )
        return annotations

    def _build_dynamic_deflection_annotations(
        self,
        x_values: Sequence[float],
        y_values: Sequence[float],
    ) -> list[tuple[str, tuple[float, float], dict[str, object]]]:
        return self._build_dynamic_chart_annotations(
            x_values,
            y_values,
            value_symbol="w",
            value_unit="mm",
            axis_symbol="ξ",
            axis_unit="m",
            chart_title="Dynamic deflection",
        )

    def _build_dynamic_transition_profile_annotations(
        self,
        *,
        kpi_lines: Sequence[str],
    ) -> list[tuple[str, tuple[float, float], dict[str, object]]]:
        annotation_mode = self._dynamic_annotation_mode()
        if annotation_mode == DynamicAnnotationMode.OFF:
            return []
        compact = annotation_mode == DynamicAnnotationMode.COMPACT
        metadata_lines = [
            "Analysis: Dynamic transition",
            f"Method: {self._format_dynamic_method_label()}",
            "Chart: Dynamic transition foundation profile k(x)",
            self._chart_location_note("Dynamic transition foundation profile k(x)"),
        ]
        metadata_lines.extend(self._build_dynamic_load_source_lines(compact=compact))
        if not compact:
            metadata_lines.extend(self._build_input_parameter_lines(config=self._last_dynamic_transition_config))
        return self._build_dual_overlay_annotations(
            metadata_lines=metadata_lines,
            kpi_lines=kpi_lines,
        )

    @staticmethod
    def _apply_panel_annotations(
        panel: PlotPanel,
        annotations: Sequence[tuple[str, tuple[float, float], dict[str, object]]],
        *,
        footer: bool = False,
        overlay: bool = True,
    ) -> None:
        for text_artist in list(panel.axes.texts):
            if text_artist.get_gid() == OVERLAY_BADGE_TEXT_GID:
                text_artist.remove()
        if footer:
            left_text = ""
            right_text = ""
            for text, (x_pos, _y_pos), style in annotations:
                horizontal_alignment = str(style.get("ha", "")).lower()
                if x_pos > 0.5 or horizontal_alignment == "right":
                    if not right_text:
                        right_text = text
                elif not left_text:
                    left_text = text
            panel.set_footer_texts(left_text, right_text)
        if overlay:
            for text, (x_pos, y_pos), style in annotations:
                text_artist = panel.add_relocatable_text(
                    x_pos,
                    y_pos,
                    text,
                    transform=panel.axes.transAxes,
                    **style,
                )
                text_artist.set_gid(OVERLAY_BADGE_TEXT_GID)
        panel.request_draw_idle()

    @staticmethod
    def _format_load_marker_label(load_kn: float, *, reference: bool) -> str:
        symbol = "P_ref" if reference else "P"
        return f"{symbol} = {load_kn:.1f} kN"

    @staticmethod
    def _format_axle_split_marker_label(
        *,
        wheel_kn: float,
        axle_kn: float,
        reference: bool,
    ) -> str:
        symbol = "P_ref,wheel" if reference else "P_wheel"
        return f"{symbol} = {wheel_kn:.1f} kN/rail\naxle = {axle_kn:.1f} kN"

    def _load_source_for_analysis_config(
        self,
        config: AnalysisConfig | None,
    ) -> dict[str, object] | None:
        if config is None:
            return None
        if self._last_transition_context is not None and config is self._last_transition_context.analysis_config:
            return self._last_transition_load_source
        if self._last_envelope_config is not None and config is self._last_envelope_config.analysis_config:
            return self._last_envelope_load_source
        if config is self._last_analysis_config:
            return self._last_analysis_load_source
        return None

    def _build_load_markers_from_load_source(
        self,
        load_source: dict[str, object] | None,
        *,
        reference: bool,
    ) -> list[LoadMarker]:
        if not load_source:
            return []
        if load_source.get("load_basis") != "axle_load_split_to_two_rails":
            return []
        positions = load_source.get("axle_positions_m")
        axle_loads = load_source.get("axle_loads_n")
        wheel_loads = load_source.get("wheel_loads_n_per_rail")
        if not isinstance(positions, list) or not isinstance(axle_loads, list) or not isinstance(wheel_loads, list):
            return []
        grouped: dict[float, tuple[float, float]] = {}
        for position, axle_load, wheel_load in zip(positions, axle_loads, wheel_loads):
            try:
                x_position = float(position)
                axle_load_n = float(axle_load)
                wheel_load_n = float(wheel_load)
            except (TypeError, ValueError):
                continue
            if not (
                math.isfinite(x_position)
                and math.isfinite(axle_load_n)
                and math.isfinite(wheel_load_n)
            ):
                continue
            key = round(x_position, 9)
            existing_axle, existing_wheel = grouped.get(key, (0.0, 0.0))
            grouped[key] = (existing_axle + axle_load_n, existing_wheel + wheel_load_n)

        markers: list[LoadMarker] = []
        for x_position in sorted(grouped):
            axle_load_n, wheel_load_n = grouped[x_position]
            axle_kn = n_to_kn(abs(axle_load_n))
            wheel_kn = n_to_kn(abs(wheel_load_n))
            markers.append(
                LoadMarker(
                    x_m=x_position,
                    load_kn=wheel_kn,
                    label=self._format_axle_split_marker_label(
                        wheel_kn=wheel_kn,
                        axle_kn=axle_kn,
                        reference=reference,
                    ),
                )
            )
        return markers

    def _build_load_markers_from_point_loads(
        self,
        loads: Sequence[PointLoad] | None,
        *,
        reference: bool = False,
    ) -> list[LoadMarker]:
        if not loads:
            return []
        load_by_position_n: dict[float, float] = {}
        for load in loads:
            if not math.isfinite(load.position_m) or not math.isfinite(load.load_newtons):
                continue
            key = round(float(load.position_m), 9)
            load_by_position_n[key] = load_by_position_n.get(key, 0.0) + float(load.load_newtons)
        markers: list[LoadMarker] = []
        for x_position in sorted(load_by_position_n):
            load_kn = n_to_kn(abs(load_by_position_n[x_position]))
            markers.append(
                LoadMarker(
                    x_m=x_position,
                    load_kn=load_kn,
                    label=self._format_load_marker_label(load_kn, reference=reference),
                )
            )
        return markers

    def _build_load_markers_from_analysis_config(
        self,
        config: AnalysisConfig | None,
        *,
        reference: bool = False,
    ) -> list[LoadMarker]:
        if config is None:
            return []
        load_source_markers = self._build_load_markers_from_load_source(
            self._load_source_for_analysis_config(config),
            reference=reference,
        )
        if load_source_markers:
            return load_source_markers
        loads: list[PointLoad] = list(config.loads)
        if config.use_two_rail:
            right_loads = list(config.right_loads) if config.right_loads is not None else list(config.loads)
            loads.extend(right_loads)
        return self._build_load_markers_from_point_loads(loads, reference=reference)

    def _ballast_thickness_m(self) -> float:
        return max(0.0, mm_to_m(self.ballast_thickness_input.value()))

    def _a3902_ballast_depth_m(self) -> float | None:
        depth_m = self._ballast_thickness_m()
        return depth_m if depth_m > 0.0 else None

    def _resolve_a3902_rail_centres_m(self, rail: Rail) -> float:
        gauge_m = self._active_track_gauge_m if self._active_track_gauge_m is not None else DEFAULT_TRACK_GAUGE_M
        if gauge_m <= 0:
            gauge_m = DEFAULT_TRACK_GAUGE_M
        head_width_m = 0.0
        if rail.head_width_mm is not None and rail.head_width_mm > 0:
            head_width_m = mm_to_m(rail.head_width_mm)
        rail_centres_m = gauge_m + head_width_m
        if rail_centres_m <= 0:
            raise ValueError("A3902 rail centre spacing must be positive.")
        return rail_centres_m

    @staticmethod
    def _resolve_stress_section_moduli(
        *,
        section_modulus_m3: float,
        section_modulus_head_m3: float | None,
        section_modulus_foot_m3: float | None,
    ) -> tuple[float, float]:
        section_top = section_modulus_head_m3 if section_modulus_head_m3 is not None else section_modulus_m3
        section_bottom = section_modulus_foot_m3 if section_modulus_foot_m3 is not None else section_modulus_m3
        return section_top, section_bottom

    @staticmethod
    def _resolve_envelope_bearing_geometry(config: EnvelopeConfig) -> BearingGeometry:
        sleeper_width = config.analysis_config.sleeper_width_m
        sleeper_length = config.analysis_config.sleeper_length_m
        uses_sleeper_geometry = math.isclose(config.bearing_width_m, sleeper_width, rel_tol=0.0, abs_tol=1e-12) and (
            math.isclose(config.bearing_length_m, sleeper_length, rel_tol=0.0, abs_tol=1e-12)
        )
        provenance = "sleeper_geometry" if uses_sleeper_geometry else "envelope_effective_geometry"
        return get_bearing_geometry(
            sleeper_width_m=sleeper_width,
            sleeper_length_m=sleeper_length,
            bearing_width_override_m=config.bearing_width_m,
            bearing_length_override_m=config.bearing_length_m,
            override_provenance=provenance,
        )

    @staticmethod
    def _legend_label_with_peak(label: str, values_mpa: Sequence[float]) -> str:
        if not values_mpa:
            return label
        peak = max(abs(value) for value in values_mpa)
        return f"{label} [max {peak:.3f} MPa]"

    def _stress_metadata_payload(self, stress: StressResults | None) -> dict[str, object]:
        payload: dict[str, object] = {
            "stress_schema_version": 1,
            "ballast_thickness_m": self._ballast_thickness_m(),
            "stress_model": "M/Z + 2:1 spread (load-conserving)",
            "bearing_geometry_provenance": "",
            "pressure_sign_convention": "positive_compression",
        }
        if stress is None:
            return payload
        payload["ballast_thickness_m"] = (
            stress.metadata.ballast_thickness_m
            if stress.metadata.ballast_thickness_m is not None
            else self._ballast_thickness_m()
        )
        payload["stress_model"] = stress.metadata.stress_model
        payload["bearing_geometry_provenance"] = stress.metadata.bearing_geometry_provenance or ""
        payload["pressure_sign_convention"] = stress.metadata.pressure_sign_convention
        return payload

    def _as5100_envelope_summary_payload(
        self,
        result: EnvelopeResult | None,
        load_source: dict[str, object] | None,
    ) -> dict[str, object] | None:
        if result is None:
            return None
        payload = _build_as5100_envelope_summary_payload(result, load_source)
        if payload is None:
            return None
        payload["text"] = _format_as5100_envelope_summary_text(result, load_source)
        return payload

    def _extend_envelope_metadata_payload(
        self,
        payload: dict[str, object],
        *,
        result: EnvelopeResult | None,
        load_source: dict[str, object] | None,
    ) -> dict[str, object]:
        as5100_summary = self._as5100_envelope_summary_payload(result, load_source)
        if as5100_summary is not None:
            payload["as5100_envelope_summary"] = as5100_summary
        return payload

    def _build_stress_from_analysis_result(self, result: AnalysisResult) -> StressResults | None:
        config = self._last_analysis_config
        if config is None:
            return None
        section_top, section_bottom = self._resolve_stress_section_moduli(
            section_modulus_m3=config.section_modulus_m3,
            section_modulus_head_m3=config.section_modulus_head_m3,
            section_modulus_foot_m3=config.section_modulus_foot_m3,
        )
        geometry = get_bearing_geometry(
            sleeper_width_m=config.sleeper_width_m,
            sleeper_length_m=config.sleeper_length_m,
            override_provenance="sleeper_geometry",
        )
        return build_stress_results_from_single(
            x_m=result.x_m,
            moment_nm=result.moment_nm,
            sleeper_positions_m=result.sleeper_positions_m,
            sleeper_loads_n=result.sleeper_loads_n,
            section_modulus_top_m3=section_top,
            section_modulus_bottom_m3=section_bottom,
            bearing_geometry=geometry,
            ballast_thickness_m=self._ballast_thickness_m(),
        )

    def _build_stress_from_envelope_result(self, result: EnvelopeResult) -> StressResults | None:
        config = self._last_envelope_config
        if config is None:
            return None
        section_top, section_bottom = self._resolve_stress_section_moduli(
            section_modulus_m3=config.analysis_config.section_modulus_m3,
            section_modulus_head_m3=config.analysis_config.section_modulus_head_m3,
            section_modulus_foot_m3=config.analysis_config.section_modulus_foot_m3,
        )
        geometry = self._resolve_envelope_bearing_geometry(config)
        return build_stress_results_from_envelope(
            x_m=result.x_m,
            moment_max_nm=result.moment_max_nm,
            moment_min_nm=result.moment_min_nm,
            sleeper_positions_m=result.sleeper_positions_m,
            sleeper_loads_max_n=result.sleeper_loads_max_n,
            sleeper_loads_min_n=result.sleeper_loads_min_n,
            section_modulus_top_m3=section_top,
            section_modulus_bottom_m3=section_bottom,
            bearing_geometry=geometry,
            ballast_thickness_m=self._ballast_thickness_m(),
        )

    def _build_stress_from_dynamic_result(self, result: DynamicResult) -> StressResults:
        config = self._last_dynamic_config
        section_modulus = config.section_modulus_m3 if config is not None else self.rail_combo.currentData().section_modulus_m3
        return build_rail_only_stress_results(
            x_m=result.spatial.xi_m,
            moment_nm=result.spatial.moment_nm,
            section_modulus_top_m3=section_modulus,
            section_modulus_bottom_m3=section_modulus,
        )

    def _build_stress_from_dynamic_transition_result(self, result: DynamicTransitionResult) -> StressResults:
        config = self._last_dynamic_transition_config
        section_modulus = (
            config.section_modulus_m3
            if config is not None
            else self.rail_combo.currentData().section_modulus_m3
        )
        if result.series.moment_max_nm is not None and result.series.moment_min_nm is not None:
            return build_rail_only_stress_results_from_envelope(
                x_m=result.series.x_m,
                moment_max_nm=result.series.moment_max_nm,
                moment_min_nm=result.series.moment_min_nm,
                section_modulus_top_m3=section_modulus,
                section_modulus_bottom_m3=section_modulus,
            )
        return build_rail_only_stress_results(
            x_m=result.representative.spatial.xi_m,
            moment_nm=result.representative.spatial.moment_nm,
            section_modulus_top_m3=section_modulus,
            section_modulus_bottom_m3=section_modulus,
        )

    def _render_static_pressure_plot(
        self,
        result: AnalysisResult,
        stress: StressResults | None,
        *,
        load_markers: Sequence[LoadMarker] | None,
    ) -> None:
        pressure_series: list[tuple[list[float], list[float], str]] = []
        pressure_styles: list[dict[str, object]] = []

        if (
            stress is not None
            and stress.metadata.pressure_available
            and stress.sleeper_positions_m
            and stress.q_ballast_comp_pa is not None
        ):
            pressure_series.append(
                (
                    stress.sleeper_positions_m,
                    [pa_to_kpa(value) for value in stress.q_ballast_comp_pa],
                    "Ballast top / sleeper contact",
                )
            )
            pressure_styles.append({"color": "#2ca02c", "linestyle": "-", "linewidth": 2.0})

            if stress.q_capping_comp_pa is not None:
                ballast_depth_m = stress.metadata.ballast_thickness_m
                depth_label = (
                    f"z={ballast_depth_m:.2f} m"
                    if ballast_depth_m is not None
                    else "ballast depth"
                )
                pressure_series.append(
                    (
                        stress.sleeper_positions_m,
                        [pa_to_kpa(value) for value in stress.q_capping_comp_pa],
                        f"Below ballast / capping top ({depth_label})",
                    )
                )
                pressure_styles.append({"color": "#d62728", "linestyle": "--", "linewidth": 2.0})

        if not pressure_series:
            pressure_series.append(
                (
                    result.sleeper_positions_m,
                    [pa_to_kpa(value) for value in result.sleeper_pressures_pa],
                    "Sleeper-ballast contact",
                )
            )
            pressure_styles.append({"color": "#2ca02c", "linestyle": "-", "linewidth": 2.0})

        a3902 = (
            result.summary.design_summary.a3902_checks
            if result.summary.design_summary is not None
            else None
        )
        if a3902 is not None and result.sleeper_positions_m:
            ballast_depth_m = self._a3902_ballast_depth_m()
            if a3902.formation_pressure_pa is not None:
                depth_label = f"z={ballast_depth_m:.2f} m" if ballast_depth_m is not None else "formation"
                pressure_series.append(
                    (
                        result.sleeper_positions_m,
                        [pa_to_kpa(a3902.formation_pressure_pa) for _ in result.sleeper_positions_m],
                        f"A3902 formation pressure ({depth_label})",
                    )
                )
                pressure_styles.append({"color": "#9467bd", "linestyle": ":", "linewidth": 1.8})
            if (
                a3902.subgrade_pressure_pa is not None
                and a3902.formation_pressure_pa is not None
                and not math.isclose(
                    a3902.subgrade_pressure_pa,
                    a3902.formation_pressure_pa,
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-9,
                )
            ):
                pressure_series.append(
                    (
                        result.sleeper_positions_m,
                        [pa_to_kpa(a3902.subgrade_pressure_pa) for _ in result.sleeper_positions_m],
                        "A3902 subgrade pressure",
                    )
                )
                pressure_styles.append({"color": "#8c564b", "linestyle": "-.", "linewidth": 1.8})

        self.pressure_plot.update_multi_plot(
            pressure_series,
            title="Ballast top and depth pressures",
            xlabel="Sleeper position (m)",
            ylabel="Pressure (kPa)",
            critical_labels=self._chart_extrema_labels_visible(),
            load_markers=self._chart_input_load_markers(load_markers),
            styles=pressure_styles,
        )
        self._apply_panel_annotations(
            self.pressure_plot,
            self._build_static_chart_annotations(
                result,
                chart_title="Ballast top and depth pressures",
            ),
            footer=True,
            overlay=False,
        )

    def _sync_stress_series_controls(
        self,
        stress: StressResults | None,
        *,
        reset_visibility: bool,
    ) -> None:
        if not hasattr(self, "stress_rail_checkbox"):
            return
        pressure_available = bool(
            stress is not None
            and stress.metadata.pressure_available
            and stress.sleeper_positions_m
            and stress.q_ballast_comp_pa is not None
            and stress.q_capping_comp_pa is not None
        )
        self.stress_rail_checkbox.blockSignals(True)
        self.stress_rail_checkbox.setEnabled(stress is not None)
        if reset_visibility:
            self.stress_rail_checkbox.setChecked(stress is not None)
        self.stress_rail_checkbox.blockSignals(False)
        for checkbox in (self.stress_ballast_checkbox, self.stress_capping_checkbox):
            checkbox.blockSignals(True)
            checkbox.setEnabled(pressure_available)
            if reset_visibility:
                checkbox.setChecked(pressure_available)
            elif not pressure_available:
                checkbox.setChecked(False)
            checkbox.blockSignals(False)

    def _stress_series_visibility(self) -> tuple[bool, bool, bool]:
        if not hasattr(self, "stress_rail_checkbox"):
            return True, False, False
        show_rail = self.stress_rail_checkbox.isChecked()
        show_ballast = self.stress_ballast_checkbox.isChecked() and self.stress_ballast_checkbox.isEnabled()
        show_capping = self.stress_capping_checkbox.isChecked() and self.stress_capping_checkbox.isEnabled()
        if not (show_rail or show_ballast or show_capping):
            self.stress_rail_checkbox.blockSignals(True)
            self.stress_rail_checkbox.setChecked(True)
            self.stress_rail_checkbox.blockSignals(False)
            show_rail = True
        return show_rail, show_ballast, show_capping

    def _rerender_last_stress_chart(self) -> None:
        stress = self._last_rendered_stress
        if stress is None:
            return
        self._render_stress_results(
            stress,
            title=self._last_stress_title,
            unavailable_note=self._last_stress_unavailable_note,
            preserve_series_visibility=True,
        )

    def _render_stress_results(
        self,
        stress: StressResults,
        *,
        title: str,
        unavailable_note: str | None = None,
        preserve_series_visibility: bool = False,
    ) -> None:
        self._last_rendered_stress = stress
        self._last_stress_title = title
        self._last_stress_unavailable_note = unavailable_note
        self._sync_stress_series_controls(
            stress,
            reset_visibility=not preserve_series_visibility,
        )
        show_rail, show_ballast, show_capping = self._stress_series_visibility()
        reference_markers = self._last_envelope_result is not None and self._last_analysis_result is None
        analysis_config_for_markers: AnalysisConfig | None = None
        if self._last_transition_context is not None:
            analysis_config_for_markers = self._last_transition_context.analysis_config
            if self._last_transition_result is not None:
                reference_markers = _enum_value(self._last_transition_result.mode) == TransitionRunMode.ENVELOPE.value
        elif self._last_analysis_config is not None:
            analysis_config_for_markers = self._last_analysis_config
            reference_markers = False
        elif self._last_envelope_config is not None:
            analysis_config_for_markers = self._last_envelope_config.analysis_config
            reference_markers = True
        load_markers = self._build_load_markers_from_analysis_config(
            analysis_config_for_markers,
            reference=reference_markers,
        )
        top_mpa = [pa_to_mpa(value) for value in stress.sigma_top_fiber_pa]
        bottom_mpa = [pa_to_mpa(value) for value in stress.sigma_bottom_fiber_pa]
        rail_series: list[tuple[list[float], list[float], str]] = []
        if show_rail:
            rail_series.extend(
                [
                    (
                        stress.x_m,
                        top_mpa,
                        self._legend_label_with_peak("Rail stress - top fibre (bending)", top_mpa),
                    ),
                    (
                        stress.x_m,
                        bottom_mpa,
                        self._legend_label_with_peak("Rail stress - bottom fibre (bending)", bottom_mpa),
                    ),
                ]
            )
        pressure_series: list[tuple[list[float], list[float], str]] = []
        if (
            stress.metadata.pressure_available
            and stress.sleeper_positions_m
            and stress.q_ballast_comp_pa is not None
            and stress.q_capping_comp_pa is not None
        ):
            ballast_mpa = [pa_to_mpa(value) for value in stress.q_ballast_comp_pa]
            capping_mpa = [pa_to_mpa(value) for value in stress.q_capping_comp_pa]
            if show_ballast:
                pressure_series.append(
                    (
                        stress.sleeper_positions_m,
                        ballast_mpa,
                        self._legend_label_with_peak("Sleeper-ballast pressure - ballast top", ballast_mpa),
                    )
                )
            if show_capping:
                pressure_series.append(
                    (
                        stress.sleeper_positions_m,
                        capping_mpa,
                        self._legend_label_with_peak("Pressure - capping top", capping_mpa),
                    )
                )
        pressure_styles = [
            {"color": "#2ca02c", "linestyle": "-.", "linewidth": 1.9},
            {"color": "#d62728", "linestyle": ":", "linewidth": 1.9},
        ]
        if pressure_series and rail_series:
            self.stress_plot.update_multi_plot_dual_axis(
                rail_series,
                pressure_series,
                title=title,
                xlabel="Position (m)",
                primary_ylabel="Rail stress (MPa)",
                secondary_ylabel="Pressure (MPa)",
                primary_styles=[
                    {"color": "#1f77b4", "linestyle": "-", "linewidth": 2.1},
                    {"color": "#ff7f0e", "linestyle": "--", "linewidth": 2.0},
                ],
                secondary_styles=pressure_styles,
                critical_labels=self._chart_extrema_labels_visible(),
                load_markers=self._chart_input_load_markers(load_markers),
            )
        elif pressure_series:
            self.stress_plot.update_multi_plot(
                pressure_series,
                title=title,
                xlabel="Position (m)",
                ylabel="Pressure (MPa)",
                critical_labels=self._chart_extrema_labels_visible(),
                styles=pressure_styles,
                load_markers=self._chart_input_load_markers(load_markers),
            )
        else:
            self.stress_plot.update_multi_plot(
                rail_series,
                title=title,
                xlabel="Position (m)",
                ylabel="Stress / Pressure (MPa)",
                critical_labels=self._chart_extrema_labels_visible(),
                styles=[
                    {"color": "#1f77b4", "linestyle": "-", "linewidth": 2.1},
                    {"color": "#ff7f0e", "linestyle": "--", "linewidth": 2.0},
                ],
                load_markers=self._chart_input_load_markers(load_markers),
            )
        stress_annotations: list[tuple[str, tuple[float, float], dict[str, object]]] = []
        if self._last_transition_result is not None:
            stress_annotations = self._build_transition_chart_annotations(
                self._last_transition_result,
                chart_title=title,
            )
        elif self._last_envelope_result is not None:
            stress_annotations = self._build_envelope_chart_annotations(
                self._last_envelope_result,
                chart_title=title,
            )
        elif self._last_analysis_result is not None:
            stress_annotations = self._build_static_chart_annotations(
                self._last_analysis_result,
                chart_title=title,
            )
        elif self._last_special_result is not None:
            stress_annotations = self._build_special_chart_annotations(
                self._last_special_result,
                chart_title=title,
            )
        elif self._last_dynamic_result is not None:
            stress_annotations = self._build_dual_overlay_annotations(
                metadata_lines=[
                    "Analysis: Dynamic",
                    f"Method: {self._format_dynamic_method_label()}",
                    f"Chart: {title}",
                    self._chart_location_note(title),
                    *self._build_input_parameter_lines(config=self._last_dynamic_config),
                ],
                kpi_lines=self._build_dynamic_global_summary_lines(),
            )
        if stress_annotations:
            self._apply_panel_annotations(self.stress_plot, stress_annotations, footer=True)
        if unavailable_note and self._chart_output_labels_visible():
            self.stress_plot.add_relocatable_text(
                0.02,
                0.98,
                unavailable_note,
                transform=self.stress_plot.axes.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#333333",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.65,
                    "edgecolor": "none",
                    "boxstyle": "round,pad=0.2",
                },
            )
        if stress.metadata.pressure_available and self._chart_output_labels_visible():
            pressure_note = (
                "Ballast/capping pressures are shown on the right axis."
                if pressure_series and rail_series
                else "Selected ballast/capping pressure series are shown on this axis."
                if pressure_series
                else "Ballast/capping pressure series are available from the Stress series checkboxes."
            )
            self.stress_plot.add_relocatable_text(
                0.02,
                0.02,
                pressure_note,
                transform=self.stress_plot.axes.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="#444444",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.55,
                    "edgecolor": "none",
                    "boxstyle": "round,pad=0.2",
                },
            )
        self.stress_plot.request_draw_idle()

    def _render_stress_unavailable(self, *, title: str, message: str) -> None:
        self._last_rendered_stress = None
        self._last_stress_title = title
        self._last_stress_unavailable_note = message
        self._sync_stress_series_controls(None, reset_visibility=True)
        self.stress_plot.update_plot(
            [0.0, 1.0],
            [0.0, 0.0],
            title=title,
            xlabel="Position (m)",
            ylabel="Stress / Pressure (MPa)",
        )
        self.stress_plot.add_relocatable_text(
            0.02,
            0.98,
            message,
            transform=self.stress_plot.axes.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#333333",
            bbox={
                "facecolor": "white",
                "alpha": 0.65,
                "edgecolor": "none",
                "boxstyle": "round,pad=0.25",
            },
        )
        self.stress_plot.request_draw_idle()

    def _render_analysis_result(self, result: AnalysisResult) -> None:
        if not self._validate_plot_series(result):
            self.statusBar().showMessage("Analysis results incomplete")
            self._set_export_buttons_enabled(False)
            return

        load_markers = self._build_load_markers_from_analysis_config(self._last_analysis_config)
        x_m = result.x_m
        deflection_mm = [m_to_mm(value) for value in result.deflection_m]
        self.deflection_plot.update_plot(
            x_m,
            deflection_mm,
            title="Deflection",
            xlabel="x (m)",
            ylabel="y (mm)",
            annotations=self._build_static_chart_annotations(result, chart_title="Deflection"),
            critical_labels=self._chart_extrema_labels_visible(),
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self.moment_plot.update_plot(
            x_m,
            [value / 1000.0 for value in result.moment_nm],
            title="Bending moment",
            xlabel="x (m)",
            ylabel="M (kN·m)",
            annotations=self._build_static_chart_annotations(result, chart_title="Bending moment"),
            critical_labels=self._chart_extrema_labels_visible(),
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self.shear_plot.update_plot(
            x_m,
            [n_to_kn(value) for value in result.shear_n],
            title="Shear",
            xlabel="x (m)",
            ylabel="V (kN)",
            annotations=self._build_static_chart_annotations(result, chart_title="Shear"),
            critical_labels=self._chart_extrema_labels_visible(),
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self.reaction_plot.update_plot(
            x_m,
            [n_to_kn(value) for value in result.reaction_n_per_m],
            title="Rail support reaction",
            xlabel="x (m)",
            ylabel="R_support (kN/m)",
            annotations=self._build_static_chart_annotations(result, chart_title="Rail support reaction"),
            critical_labels=self._chart_extrema_labels_visible(),
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self.sleeper_plot.update_plot(
            result.sleeper_positions_m,
            [n_to_kn(value) for value in result.sleeper_loads_n],
            title="Sleeper seat loads",
            xlabel="Sleeper position (m)",
            ylabel="Q (kN)",
            annotations=self._build_static_chart_annotations(result, chart_title="Sleeper seat loads"),
            critical_labels=self._chart_extrema_labels_visible(),
            load_markers=self._chart_input_load_markers(load_markers),
        )
        stress = self._build_stress_from_analysis_result(result)
        self._render_static_pressure_plot(result, stress, load_markers=load_markers)
        if stress is not None:
            self._last_analysis_stress = stress
            self._last_envelope_stress = None
            self._render_stress_results(stress, title="Stress")
        self.summary_panel.update_summary(result)
        self.summary_panel.update_sanity_checks(self._compute_sanity_checks(result))
        self._update_two_rail_tabs(result, load_markers=load_markers)
        self._mark_chart_results_updated()

    @staticmethod
    def _estimate_series_dx(x_values: Sequence[float]) -> float | None:
        if len(x_values) < 2:
            return None
        deltas = [
            abs(float(x_values[index + 1]) - float(x_values[index]))
            for index in range(len(x_values) - 1)
            if not math.isclose(float(x_values[index + 1]), float(x_values[index]))
        ]
        if not deltas:
            return None
        deltas.sort()
        mid = len(deltas) // 2
        if len(deltas) % 2 == 1:
            return deltas[mid]
        return 0.5 * (deltas[mid - 1] + deltas[mid])

    def _envelope_smoothing_radius(
        self,
        *,
        x_values: Sequence[float],
        x_ref_step_m: float | None,
    ) -> int:
        if x_ref_step_m is None or x_ref_step_m <= 0.0:
            return 0
        dx = self._estimate_series_dx(x_values)
        if dx is None or dx <= 0.0:
            return 0
        ratio = x_ref_step_m / dx
        if ratio < 1.25:
            return 0
        if ratio < 2.5:
            return 1
        if ratio < 4.0:
            return 2
        return 3

    @staticmethod
    def _extrema_filter(
        values: Sequence[float],
        radius: int,
        *,
        mode: Literal["max", "min"],
    ) -> list[float]:
        filtered = list(values)
        if radius <= 0 or len(filtered) < 3:
            return filtered
        output: list[float] = []
        for index in range(len(filtered)):
            left = max(0, index - radius)
            right = min(len(filtered), index + radius + 1)
            window = filtered[left:right]
            output.append(max(window) if mode == "max" else min(window))
        return output

    @staticmethod
    def _visual_smooth_series(values: Sequence[float], radius: int) -> list[float]:
        smoothed = list(values)
        if radius <= 0 or len(smoothed) < 3:
            return smoothed
        output: list[float] = []
        for index in range(len(smoothed)):
            left = max(0, index - radius)
            right = min(len(smoothed), index + radius + 1)
            window = smoothed[left:right]
            output.append(sum(window) / len(window))
        return output

    @staticmethod
    def _restore_smoothed_extremum(
        raw_values: Sequence[float],
        smoothed_values: Sequence[float],
        *,
        mode: Literal["max", "min"],
    ) -> list[float]:
        restored = list(smoothed_values)
        raw = list(raw_values)
        if not raw or not restored:
            return restored
        target_value = max(raw) if mode == "max" else min(raw)
        target_index = raw.index(target_value)
        if target_index < len(restored):
            restored[target_index] = target_value
        return restored

    @staticmethod
    def _connect_visual_envelope_extrema(
        x_values: Sequence[float],
        values: Sequence[float],
        *,
        mode: Literal["max", "min"],
    ) -> list[float]:
        if len(x_values) != len(values) or len(values) < 3:
            return list(values)

        extrema_indices: list[int] = [0]
        for index in range(1, len(values) - 1):
            previous_value = values[index - 1]
            current_value = values[index]
            next_value = values[index + 1]
            if mode == "max":
                is_extremum = (
                    current_value >= previous_value
                    and current_value >= next_value
                    and (current_value > previous_value or current_value > next_value)
                )
            else:
                is_extremum = (
                    current_value <= previous_value
                    and current_value <= next_value
                    and (current_value < previous_value or current_value < next_value)
                )
            if is_extremum:
                extrema_indices.append(index)
        extrema_indices.append(len(values) - 1)

        if len(extrema_indices) < 3:
            return list(values)

        connected = list(values)
        for left_index, right_index in zip(extrema_indices, extrema_indices[1:]):
            x_left = x_values[left_index]
            x_right = x_values[right_index]
            y_left = values[left_index]
            y_right = values[right_index]
            if x_right == x_left:
                continue
            for index in range(left_index + 1, right_index):
                ratio = (x_values[index] - x_left) / (x_right - x_left)
                interpolated = y_left + (y_right - y_left) * ratio
                if mode == "max":
                    connected[index] = max(values[index], interpolated)
                else:
                    connected[index] = min(values[index], interpolated)
        return connected

    def _smooth_envelope_pair(
        self,
        x_values: Sequence[float],
        upper: Sequence[float],
        lower: Sequence[float],
        x_ref_step_m: float | None,
        *,
        chart_family: str,
        visual_min_radius: int = 0,
    ) -> tuple[list[float], list[float]]:
        radius = self._envelope_smoothing_radius(x_values=x_values, x_ref_step_m=x_ref_step_m)
        if visual_min_radius > 0 and len(x_values) >= 3:
            radius = max(radius, visual_min_radius)
        if radius <= 0:
            return list(upper), list(lower)
        dx = self._estimate_series_dx(x_values)
        LOGGER.debug(
            "Envelope smoothing applied for %s: x_ref_step=%.6g m, dx=%.6g m, radius=%d, visual_min=%d",
            chart_family,
            x_ref_step_m if x_ref_step_m is not None else float("nan"),
            dx if dx is not None else float("nan"),
            radius,
            visual_min_radius,
        )
        smooth_upper = self._extrema_filter(upper, radius, mode="max")
        smooth_lower = self._extrema_filter(lower, radius, mode="min")
        if visual_min_radius > 0:
            envelope_upper = list(smooth_upper)
            envelope_lower = list(smooth_lower)
            smooth_upper = self._connect_visual_envelope_extrema(x_values, smooth_upper, mode="max")
            smooth_lower = self._connect_visual_envelope_extrema(x_values, smooth_lower, mode="min")
            smooth_upper = self._visual_smooth_series(smooth_upper, visual_min_radius)
            smooth_lower = self._visual_smooth_series(smooth_lower, visual_min_radius)
            smooth_upper = self._restore_smoothed_extremum(envelope_upper, smooth_upper, mode="max")
            smooth_lower = self._restore_smoothed_extremum(envelope_lower, smooth_lower, mode="min")
            smooth_upper = [max(raw, smooth) for raw, smooth in zip(envelope_upper, smooth_upper)]
            smooth_lower = [min(raw, smooth) for raw, smooth in zip(envelope_lower, smooth_lower)]
        return smooth_upper, smooth_lower

    def _smooth_envelope_single(
        self,
        x_values: Sequence[float],
        series: Sequence[float],
        x_ref_step_m: float | None,
        *,
        is_upper: bool,
        chart_family: str,
        visual_min_radius: int = 0,
    ) -> list[float]:
        radius = self._envelope_smoothing_radius(x_values=x_values, x_ref_step_m=x_ref_step_m)
        if visual_min_radius > 0 and len(x_values) >= 3:
            radius = max(radius, visual_min_radius)
        if radius <= 0:
            return list(series)
        dx = self._estimate_series_dx(x_values)
        LOGGER.debug(
            "Envelope smoothing applied for %s: x_ref_step=%.6g m, dx=%.6g m, radius=%d, visual_min=%d",
            chart_family,
            x_ref_step_m if x_ref_step_m is not None else float("nan"),
            dx if dx is not None else float("nan"),
            radius,
            visual_min_radius,
        )
        mode: Literal["max", "min"] = "max" if is_upper else "min"
        smoothed = self._extrema_filter(series, radius, mode=mode)
        if visual_min_radius > 0:
            envelope = list(smoothed)
            smoothed = self._connect_visual_envelope_extrema(x_values, smoothed, mode=mode)
            smoothed = self._visual_smooth_series(smoothed, visual_min_radius)
            smoothed = self._restore_smoothed_extremum(envelope, smoothed, mode=mode)
            if mode == "max":
                smoothed = [max(raw, smooth) for raw, smooth in zip(envelope, smoothed)]
            else:
                smoothed = [min(raw, smooth) for raw, smooth in zip(envelope, smoothed)]
        return smoothed

    def _render_envelope_result(
        self,
        result: EnvelopeResult,
        *,
        transition_envelope_style: bool = False,
    ) -> None:
        if not self._validate_envelope_series(result):
            self.statusBar().showMessage("Envelope results incomplete")
            self._set_export_buttons_enabled(False)
            return
        load_markers = self._build_load_markers_from_analysis_config(
            self._last_envelope_config.analysis_config if self._last_envelope_config is not None else None,
            reference=True,
        )
        x_m = result.x_m
        x_ref_step_m = self._last_envelope_config.x_ref_step_m if self._last_envelope_config is not None else None
        visual_radius = 3 if transition_envelope_style else 0
        abs_style = (
            {"color": "#1f77b4", "linewidth": 1.8, "linestyle": ":"}
            if transition_envelope_style
            else {"color": "#2ca02c", "linewidth": 1.8, "linestyle": ":"}
        )
        deflection_max_mm, deflection_min_mm = self._smooth_envelope_pair(
            x_m,
            [m_to_mm(value) for value in result.deflection_max_m],
            [m_to_mm(value) for value in result.deflection_min_m],
            x_ref_step_m,
            chart_family="deflection",
            visual_min_radius=visual_radius,
        )
        deflection_series = [
            (x_m, deflection_max_mm, "Max"),
            (x_m, deflection_min_mm, "Min"),
        ]
        deflection_styles = [
            {"color": "#1f77b4", "linewidth": 2.0},
            {"color": "#ff7f0e", "linewidth": 1.8, "linestyle": "--"},
        ]
        if transition_envelope_style:
            deflection_abs_mm = self._smooth_envelope_single(
                x_m,
                [m_to_mm(value) for value in result.deflection_abs_max_m],
                x_ref_step_m,
                is_upper=True,
                chart_family="deflection_abs",
                visual_min_radius=visual_radius,
            )
            deflection_series.append((x_m, deflection_abs_mm, "|Max|"))
            deflection_styles.append(abs_style)
        self.deflection_plot.update_multi_plot(
            deflection_series,
            title="Deflection envelope",
            xlabel="x (m)",
            ylabel="y (mm)",
            critical_labels=self._chart_extrema_labels_visible(),
            styles=deflection_styles,
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self._apply_panel_annotations(
            self.deflection_plot,
            self._build_envelope_chart_annotations(result, chart_title="Deflection envelope"),
        )
        moment_max_knm, moment_min_knm = self._smooth_envelope_pair(
            x_m,
            [value / 1000.0 for value in result.moment_max_nm],
            [value / 1000.0 for value in result.moment_min_nm],
            x_ref_step_m,
            chart_family="moment",
            visual_min_radius=visual_radius,
        )
        moment_abs_knm = self._smooth_envelope_single(
            x_m,
            [value / 1000.0 for value in result.moment_abs_max_nm],
            x_ref_step_m,
            is_upper=True,
            chart_family="moment_abs",
            visual_min_radius=visual_radius,
        )
        moment_series = [
            (x_m, moment_max_knm, "Max"),
            (x_m, moment_min_knm, "Min"),
            (x_m, moment_abs_knm, "|Max|"),
        ]
        self.moment_plot.update_multi_plot(
            moment_series,
            title="Bending moment envelope",
            xlabel="x (m)",
            ylabel="M (kN·m)",
            critical_labels=self._chart_extrema_labels_visible(),
            styles=[
                {"color": "#1f77b4", "linewidth": 2.0},
                {"color": "#ff7f0e", "linewidth": 1.8, "linestyle": "--"},
                abs_style,
            ],
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self._apply_panel_annotations(
            self.moment_plot,
            self._build_envelope_chart_annotations(result, chart_title="Bending moment envelope"),
        )
        shear_max_kn, shear_min_kn = self._smooth_envelope_pair(
            x_m,
            [n_to_kn(value) for value in result.shear_max_n],
            [n_to_kn(value) for value in result.shear_min_n],
            x_ref_step_m,
            chart_family="shear",
            visual_min_radius=visual_radius,
        )
        shear_abs_kn = self._smooth_envelope_single(
            x_m,
            [n_to_kn(value) for value in result.shear_abs_max_n],
            x_ref_step_m,
            is_upper=True,
            chart_family="shear_abs",
            visual_min_radius=visual_radius,
        )
        shear_series = [
            (x_m, shear_max_kn, "Max"),
            (x_m, shear_min_kn, "Min"),
            (x_m, shear_abs_kn, "|Max|"),
        ]
        self.shear_plot.update_multi_plot(
            shear_series,
            title="Shear envelope",
            xlabel="x (m)",
            ylabel="V (kN)",
            critical_labels=self._chart_extrema_labels_visible(),
            styles=[
                {"color": "#1f77b4", "linewidth": 2.0},
                {"color": "#ff7f0e", "linewidth": 1.8, "linestyle": "--"},
                abs_style,
            ],
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self._apply_panel_annotations(
            self.shear_plot,
            self._build_envelope_chart_annotations(result, chart_title="Shear envelope"),
        )
        reaction_max_knm, reaction_min_knm = self._smooth_envelope_pair(
            x_m,
            [n_to_kn(value) for value in result.reaction_max_n_per_m],
            [n_to_kn(value) for value in result.reaction_min_n_per_m],
            x_ref_step_m,
            chart_family="reaction",
            visual_min_radius=visual_radius,
        )
        reaction_abs_knm = self._smooth_envelope_single(
            x_m,
            [n_to_kn(value) for value in result.reaction_abs_max_n_per_m],
            x_ref_step_m,
            is_upper=True,
            chart_family="reaction_abs",
            visual_min_radius=visual_radius,
        )
        reaction_series = [
            (x_m, reaction_max_knm, "Max"),
            (x_m, reaction_min_knm, "Min"),
            (x_m, reaction_abs_knm, "|Max|"),
        ]
        self.reaction_plot.update_multi_plot(
            reaction_series,
            title="Rail support reaction envelope",
            xlabel="x (m)",
            ylabel="R_support (kN/m)",
            critical_labels=self._chart_extrema_labels_visible(),
            styles=[
                {"color": "#1f77b4", "linewidth": 2.0},
                {"color": "#ff7f0e", "linewidth": 1.8, "linestyle": "--"},
                abs_style,
            ],
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self._apply_panel_annotations(
            self.reaction_plot,
            self._build_envelope_chart_annotations(result, chart_title="Rail support reaction envelope"),
        )
        sleeper_max_kn, sleeper_min_kn = self._smooth_envelope_pair(
            result.sleeper_positions_m,
            [n_to_kn(value) for value in result.sleeper_loads_max_n],
            [n_to_kn(value) for value in result.sleeper_loads_min_n],
            x_ref_step_m,
            chart_family="sleeper_load",
        )
        sleeper_series = [
            (
                result.sleeper_positions_m,
                sleeper_max_kn,
                "Max",
            ),
            (
                result.sleeper_positions_m,
                sleeper_min_kn,
                "Min",
            ),
        ]
        self.sleeper_plot.update_multi_plot(
            sleeper_series,
            title="Sleeper loads envelope (total)",
            xlabel="Sleeper position (m)",
            ylabel="Q (kN)",
            critical_labels=self._chart_extrema_labels_visible(),
            styles=[
                {"color": "#1f77b4", "linewidth": 2.0},
                {"color": "#ff7f0e", "linewidth": 1.8, "linestyle": "--"},
            ],
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self._apply_panel_annotations(
            self.sleeper_plot,
            self._build_envelope_chart_annotations(result, chart_title="Sleeper loads envelope"),
        )
        ballast_max_kpa, ballast_min_kpa = self._smooth_envelope_pair(
            result.sleeper_positions_m,
            [pa_to_kpa(value) for value in result.ballast_pressure_max_pa],
            [pa_to_kpa(value) for value in result.ballast_pressure_min_pa],
            x_ref_step_m,
            chart_family="ballast_pressure",
        )
        pressure_series: list[tuple[list[float], list[float], str]] = [
            (
                result.sleeper_positions_m,
                ballast_max_kpa,
                "Ballast max",
            ),
            (
                result.sleeper_positions_m,
                ballast_min_kpa,
                "Ballast min",
            ),
        ]
        for depth, values in result.formation_stress_max_pa_by_depth.items():
            smoothed_kpa = self._smooth_envelope_single(
                result.sleeper_positions_m,
                [pa_to_kpa(value) for value in values],
                x_ref_step_m,
                is_upper=True,
                chart_family=f"formation_pressure_z_{depth:.2f}",
            )
            pressure_series.append(
                (
                    result.sleeper_positions_m,
                    smoothed_kpa,
                    f"σz max (z={depth:.2f} m)",
                )
            )
        self.pressure_plot.update_multi_plot(
            pressure_series,
            title="Ballast top and formation pressures",
            xlabel="Sleeper position (m)",
            ylabel="Pressure (kPa)",
            critical_labels=self._chart_extrema_labels_visible(),
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self._apply_panel_annotations(
            self.pressure_plot,
            self._build_envelope_chart_annotations(result, chart_title="Ballast top and formation pressures"),
        )
        stress = self._build_stress_from_envelope_result(result)
        if stress is not None:
            self._last_envelope_stress = stress
            self._last_analysis_stress = None
            self._render_stress_results(stress, title="Stress envelope")
        self.summary_panel.update_envelope_summary(
            result,
            load_source=self._last_envelope_load_source,
        )
        self.summary_panel.update_sanity_checks(None)
        if hasattr(self, "tab_widget"):
            show_two_rail = result.left_deflection_max_m is not None and result.right_deflection_max_m is not None
            self.tab_widget.setTabVisible(self.rail_deflection_tab_index, show_two_rail)
            self.tab_widget.setTabVisible(self.rail_moment_tab_index, show_two_rail)
        if result.left_deflection_max_m and result.right_deflection_max_m:
            left_deflection_max_mm = self._smooth_envelope_single(
                x_m,
                [m_to_mm(value) for value in result.left_deflection_max_m],
                x_ref_step_m,
                is_upper=True,
                chart_family="rail_deflection_left_max",
            )
            right_deflection_max_mm = self._smooth_envelope_single(
                x_m,
                [m_to_mm(value) for value in result.right_deflection_max_m],
                x_ref_step_m,
                is_upper=True,
                chart_family="rail_deflection_right_max",
            )
            self.rail_deflection_plot.update_multi_plot(
                [
                    (x_m, left_deflection_max_mm, "Left max"),
                    (x_m, right_deflection_max_mm, "Right max"),
                ],
                title="Rail deflection envelope (L/R max)",
                xlabel="x (m)",
                ylabel="y (mm)",
                critical_labels=self._chart_extrema_labels_visible(),
                load_markers=self._chart_input_load_markers(load_markers),
            )
            self._apply_panel_annotations(
                self.rail_deflection_plot,
                self._build_envelope_chart_annotations(result, chart_title="Rail deflection envelope (L/R max)"),
            )
        if result.left_moment_max_nm and result.right_moment_max_nm:
            left_moment_max_knm = self._smooth_envelope_single(
                x_m,
                [value / 1000.0 for value in result.left_moment_max_nm],
                x_ref_step_m,
                is_upper=True,
                chart_family="rail_moment_left_max",
            )
            right_moment_max_knm = self._smooth_envelope_single(
                x_m,
                [value / 1000.0 for value in result.right_moment_max_nm],
                x_ref_step_m,
                is_upper=True,
                chart_family="rail_moment_right_max",
            )
            rail_moment_series = [
                (x_m, left_moment_max_knm, "Left max"),
                (x_m, right_moment_max_knm, "Right max"),
            ]
            if result.left_moment_min_nm and result.right_moment_min_nm:
                left_moment_abs_knm = self._smooth_envelope_single(
                    x_m,
                    [
                        max(abs(max_val), abs(min_val)) / 1000.0
                        for max_val, min_val in zip(result.left_moment_max_nm, result.left_moment_min_nm)
                    ],
                    x_ref_step_m,
                    is_upper=True,
                    chart_family="rail_moment_left_abs",
                )
                right_moment_abs_knm = self._smooth_envelope_single(
                    x_m,
                    [
                        max(abs(max_val), abs(min_val)) / 1000.0
                        for max_val, min_val in zip(result.right_moment_max_nm, result.right_moment_min_nm)
                    ],
                    x_ref_step_m,
                    is_upper=True,
                    chart_family="rail_moment_right_abs",
                )
                rail_moment_series.extend(
                    [
                        (x_m, left_moment_abs_knm, "Left |max|"),
                        (x_m, right_moment_abs_knm, "Right |max|"),
                    ]
                )
            self.rail_moment_plot.update_multi_plot(
                rail_moment_series,
                title="Rail moment envelope (L/R max)",
                xlabel="x (m)",
                ylabel="M (kN·m)",
                critical_labels=self._chart_extrema_labels_visible(),
                load_markers=self._chart_input_load_markers(load_markers),
            )
            self._apply_panel_annotations(
                self.rail_moment_plot,
                self._build_envelope_chart_annotations(result, chart_title="Rail moment envelope (L/R max)"),
            )
        self._mark_chart_results_updated()

    def _build_overlay_label(self) -> str:
        rail = getattr(self.rail_combo.currentData(), "name", "Rail")
        sleeper = getattr(self.sleeper_combo.currentData(), "name", "Sleeper")
        support = getattr(self.support_combo.currentData(), "name", "Support")
        load_summary = "Load"
        if hasattr(self, "as5100_loads_checkbox") and self.as5100_loads_checkbox.isChecked():
            metadata = self._current_load_source_metadata()
            load_summary = (
                f"AS5100 {metadata['model']} "
                f"{metadata['axle_count']} axles "
                f"({n_to_kn(float(metadata.get('max_wheel_load_n_per_rail', 0.0) or 0.0)):.0f}kN/rail)"
            )
        elif self.train_loads_checkbox.isChecked():
            load_summary = (
                f"Train {self.train_axle_load_input.value():.0f}kN axle/"
                f"{0.5 * self.train_axle_load_input.value():.0f}kN rail x{self.train_bogie_count_input.value()}"
                f" bogies/{self.train_axles_per_bogie_input.value()} axles"
            )
        elif self.several_loads_checkbox.isChecked():
            count = self.wheel_loads_widget.rows()
            load_summary = f"{count} loads"
        else:
            load_summary = (
                f"{self.load_magnitude_input.value():.1f}kN"
                f"@{self.load_position_input.value():.0f}mm"
            )
        return f"{rail} | {sleeper} | {support} | {load_summary}"

    def _build_dynamic_overlay_label(
        self,
        *,
        config: DynamicConfig | None = None,
        mode: DynamicMode | None = None,
    ) -> str:
        mode_value = mode or self._last_dynamic_mode or self.dynamic_mode_combo.currentData()
        mode_label = {
            DynamicMode.STEADY_STATE: "Steady-state",
            DynamicMode.TIME_HISTORY: "Time-history",
        }.get(mode_value, "Dynamic")
        speed = self.speed_input.value()
        max_load_kn = self.load_magnitude_input.value()
        load_label = "max wheel"
        load_source = self._dynamic_load_source_for_annotations()
        if load_source and load_source.get("load_basis") == "axle_load_split_to_two_rails":
            max_load_kn = n_to_kn(float(load_source.get("max_axle_load_n", 0.0) or 0.0))
            load_label = "max axle"
        dynamic_config = config or self._last_dynamic_config
        if dynamic_config is not None:
            speed = dynamic_config.speed_m_per_s
            if dynamic_config.loads and not (
                load_source and load_source.get("load_basis") == "axle_load_split_to_two_rails"
            ):
                max_load_kn = max(load.load_newtons for load in dynamic_config.loads) / 1000.0
        return f"{mode_label} | v={speed:.2f} m/s | {load_label}={max_load_kn:.2f} kN"

    @staticmethod
    def _build_overlay_styles(overlay_count: int) -> list[dict[str, object]]:
        overlay_colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
        overlay_styles = ["--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1)), (0, (1, 1))]
        styles: list[dict[str, object]] = [{"color": "#1f77b4", "linestyle": "-", "linewidth": 2.0}]
        for index in range(overlay_count):
            color = overlay_colors[index % len(overlay_colors)]
            linestyle = overlay_styles[index % len(overlay_styles)]
            styles.append({"color": color, "linestyle": linestyle, "linewidth": 1.8})
        return styles

    def _render_overlay_plots(self) -> None:
        if self._last_analysis_result is None:
            return
        load_markers = self._build_load_markers_from_analysis_config(self._last_analysis_config)
        primary_label = self._primary_label or "Primary"
        series_results = [(primary_label, self._last_analysis_result), *self._overlay_results]
        styles = self._build_overlay_styles(len(self._overlay_results))
        if not self._validate_overlay_series(series_results):
            return

        def build_series(
            extractor: Callable[[AnalysisResult], list[float]],
            transform: Callable[[float], float],
        ) -> list[tuple[list[float], list[float], str]]:
            return [
                (
                    entry_result.x_m,
                    [transform(value) for value in extractor(entry_result)],
                    label,
                )
                for label, entry_result in series_results
            ]

        self.deflection_plot.update_multi_plot(
            build_series(lambda item: item.deflection_m, m_to_mm),
            title="Deflection",
            xlabel="x (m)",
            ylabel="y (mm)",
            critical_labels=self._chart_extrema_labels_visible(),
            styles=styles,
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self._apply_panel_annotations(
            self.deflection_plot,
            self._build_static_chart_annotations(self._last_analysis_result, chart_title="Deflection"),
        )
        self.moment_plot.update_multi_plot(
            build_series(lambda item: item.moment_nm, lambda value: value / 1000.0),
            title="Bending moment",
            xlabel="x (m)",
            ylabel="M (kN·m)",
            critical_labels=self._chart_extrema_labels_visible(),
            styles=styles,
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self._apply_panel_annotations(
            self.moment_plot,
            self._build_static_chart_annotations(self._last_analysis_result, chart_title="Bending moment"),
        )
        self.shear_plot.update_multi_plot(
            build_series(lambda item: item.shear_n, n_to_kn),
            title="Shear",
            xlabel="x (m)",
            ylabel="V (kN)",
            critical_labels=self._chart_extrema_labels_visible(),
            styles=styles,
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self._apply_panel_annotations(
            self.shear_plot,
            self._build_static_chart_annotations(self._last_analysis_result, chart_title="Shear"),
        )
        self.reaction_plot.update_multi_plot(
            build_series(lambda item: item.reaction_n_per_m, n_to_kn),
            title="Rail support reaction",
            xlabel="x (m)",
            ylabel="R_support (kN/m)",
            critical_labels=self._chart_extrema_labels_visible(),
            styles=styles,
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self._apply_panel_annotations(
            self.reaction_plot,
            self._build_static_chart_annotations(self._last_analysis_result, chart_title="Rail support reaction"),
        )

        sleeper_series = [
            (
                entry_result.sleeper_positions_m,
                [n_to_kn(value) for value in entry_result.sleeper_loads_n],
                label,
            )
            for label, entry_result in series_results
        ]
        self.sleeper_plot.update_multi_plot(
            sleeper_series,
            title="Sleeper seat loads",
            xlabel="Sleeper position (m)",
            ylabel="Q (kN)",
            critical_labels=self._chart_extrema_labels_visible(),
            styles=styles,
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self._apply_panel_annotations(
            self.sleeper_plot,
            self._build_static_chart_annotations(self._last_analysis_result, chart_title="Sleeper seat loads"),
        )
        pressure_series = [
            (
                entry_result.sleeper_positions_m,
                [pa_to_kpa(value) for value in entry_result.sleeper_pressures_pa],
                label,
            )
            for label, entry_result in series_results
        ]
        self.pressure_plot.update_multi_plot(
            pressure_series,
            title="Sleeper-ballast contact pressure",
            xlabel="Sleeper position (m)",
            ylabel="Pressure (kPa)",
            critical_labels=self._chart_extrema_labels_visible(),
            styles=styles,
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self._apply_panel_annotations(
            self.pressure_plot,
            self._build_static_chart_annotations(
                self._last_analysis_result,
                chart_title="Sleeper-ballast contact pressure",
            ),
        )
        if self._last_analysis_stress is not None:
            self._render_stress_results(self._last_analysis_stress, title="Stress (primary run)")

        self.summary_panel.update_summary(self._last_analysis_result)
        self.summary_panel.update_sanity_checks(self._compute_sanity_checks(self._last_analysis_result))
        self._update_two_rail_tabs(self._last_analysis_result, load_markers=load_markers)
        self._mark_chart_results_updated()

    def _render_dynamic_result(self, result: DynamicResult) -> None:
        spatial = result.spatial
        xi_m = spatial.xi_m
        deflection_mm = [m_to_mm(value) for value in spatial.deflection_m]
        deflection_annotations = self._build_dynamic_deflection_annotations(xi_m, deflection_mm)
        moment_knm = [value / 1000.0 for value in spatial.moment_nm]
        moment_annotations = self._build_dynamic_chart_annotations(
            xi_m,
            moment_knm,
            value_symbol="M",
            value_unit="kN·m",
            axis_symbol="ξ",
            axis_unit="m",
            chart_title="Dynamic moment",
        )
        shear_kn = [n_to_kn(value) for value in spatial.shear_n]
        shear_annotations = self._build_dynamic_chart_annotations(
            xi_m,
            shear_kn,
            value_symbol="V",
            value_unit="kN",
            axis_symbol="ξ",
            axis_unit="m",
            chart_title="Dynamic shear",
        )
        reaction_knm = [n_to_kn(value) for value in spatial.reaction_n_per_m]
        reaction_annotations = self._build_dynamic_chart_annotations(
            xi_m,
            reaction_knm,
            value_symbol="R",
            value_unit="kN/m",
            axis_symbol="ξ",
            axis_unit="m",
            chart_title="Dynamic rail support reaction",
        )
        self.dynamic_deflection_plot.update_plot(
            xi_m,
            deflection_mm,
            title="Dynamic deflection",
            xlabel="ξ (m)",
            ylabel="w (mm)",
            annotations=deflection_annotations,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self.dynamic_moment_plot.update_plot(
            xi_m,
            moment_knm,
            title="Dynamic moment",
            xlabel="ξ (m)",
            ylabel="M (kN·m)",
            annotations=moment_annotations,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self.dynamic_shear_plot.update_plot(
            xi_m,
            shear_kn,
            title="Dynamic shear",
            xlabel="ξ (m)",
            ylabel="V (kN)",
            annotations=shear_annotations,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self.dynamic_reaction_plot.update_plot(
            xi_m,
            reaction_knm,
            title="Dynamic rail support reaction",
            xlabel="ξ (m)",
            ylabel="R_support (kN/m)",
            annotations=reaction_annotations,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self._last_dynamic_stress = self._build_stress_from_dynamic_result(result)
        self._render_stress_results(
            self._last_dynamic_stress,
            title="Peak dynamic bending stress",
            unavailable_note="Sleeper/capping pressures are not available in dynamic mode.",
        )

        probe = self._selected_probe_series(result)
        if probe is not None:
            probe_position = f"Probe: x={probe.position_m:.3f} m"
            time_deflection_mm = [m_to_mm(value) for value in probe.deflection_m]
            time_deflection_annotations = self._build_dynamic_chart_annotations(
                probe.time_s,
                time_deflection_mm,
                value_symbol="w",
                value_unit="mm",
                axis_symbol="t",
                axis_unit="s",
                chart_title="Deflection time history",
                extra_metadata_lines=[probe_position],
            )
            damping_knm = [n_to_kn(value) for value in probe.damping_force_n_per_m]
            damping_annotations = self._build_dynamic_chart_annotations(
                probe.time_s,
                damping_knm,
                value_symbol="F_d",
                value_unit="kN/m",
                axis_symbol="t",
                axis_unit="s",
                chart_title="Damping force time history",
                extra_metadata_lines=[probe_position],
            )
            fft_mm = [value * 1000.0 for value in probe.fft_amplitude]
            fft_annotations = self._build_dynamic_chart_annotations(
                probe.fft_frequency_hz,
                fft_mm,
                value_symbol="|W|",
                value_unit="mm",
                axis_symbol="f",
                axis_unit="Hz",
                chart_title="FFT amplitude",
                extra_metadata_lines=[probe_position],
            )
            psd_mm2_hz = [value * 1_000_000.0 for value in probe.psd]
            psd_annotations = self._build_dynamic_chart_annotations(
                probe.psd_frequency_hz,
                psd_mm2_hz,
                value_symbol="PSD",
                value_unit="mm²/Hz",
                axis_symbol="f",
                axis_unit="Hz",
                chart_title="Welch PSD",
                extra_metadata_lines=[probe_position],
            )
            impedance_mn = [n_per_m2_to_mn_per_m2(value) for value in probe.impedance_magnitude_n_per_m2]
            impedance_annotations = self._build_dynamic_chart_annotations(
                probe.impedance_frequency_hz,
                impedance_mn,
                value_symbol="|Z|",
                value_unit="MN/m²",
                axis_symbol="f",
                axis_unit="Hz",
                chart_title="Support impedance magnitude",
                extra_metadata_lines=[probe_position],
            )
            self.dynamic_time_plot.update_plot(
                probe.time_s,
                time_deflection_mm,
                title="Deflection time history",
                xlabel="t (s)",
                ylabel="w (mm)",
                annotations=time_deflection_annotations,
                critical_labels=self._chart_extrema_labels_visible(),
            )
            self.dynamic_damping_plot.update_plot(
                probe.time_s,
                damping_knm,
                title="Damping force time history",
                xlabel="t (s)",
                ylabel="F_d (kN/m)",
                annotations=damping_annotations,
                critical_labels=self._chart_extrema_labels_visible(),
            )
            self.dynamic_fft_plot.update_plot(
                probe.fft_frequency_hz,
                fft_mm,
                title="FFT amplitude",
                xlabel="f (Hz)",
                ylabel="|W(f)| (mm)",
                annotations=fft_annotations,
                critical_labels=self._chart_extrema_labels_visible(),
            )
            self.dynamic_psd_plot.update_plot(
                probe.psd_frequency_hz,
                psd_mm2_hz,
                title="Welch PSD",
                xlabel="f (Hz)",
                ylabel="PSD (mm²/Hz)",
                annotations=psd_annotations,
                critical_labels=self._chart_extrema_labels_visible(),
            )
            self.dynamic_impedance_plot.update_plot(
                probe.impedance_frequency_hz,
                impedance_mn,
                title="Support impedance magnitude",
                xlabel="f (Hz)",
                ylabel="|Z(f)| (MN/m²)",
                annotations=impedance_annotations,
                critical_labels=self._chart_extrema_labels_visible(),
            )
        self.dynamic_summary_panel.update_summary(result)
        self._mark_chart_results_updated()

    def _render_dynamic_transition_result(self, result: DynamicTransitionResult) -> None:
        self._render_dynamic_result(result.representative)
        fidelity_text = result.solver_fidelity.replace("_", " ").title()
        profile_text = result.profile_type.replace("_", " ").title()
        screening_note = (
            "Screening fidelity: response uses uniform k1; k(x) shows configured transition profile."
            if result.solver_fidelity == "screening" and result.profile_type != DynamicTransitionProfileType.UNIFORM.value
            else None
        )
        kpi_lines = [
            f"Fidelity: {fidelity_text}",
            f"Profile: {profile_text}",
            f"Risk index: {result.metrics.risk_index:.3f}",
            f"Critical speed ratio: {result.metrics.critical_speed_ratio:.3f}",
            f"Amplification: {result.metrics.dynamic_amplification:.3f}",
            f"|w|max: {m_to_mm(abs(result.metrics.max_deflection_m)):.3f} mm",
        ]
        if screening_note is not None:
            kpi_lines.append(screening_note)
        k_chart_x, k_chart_y = self._transition_k_chart_series(
            x_values=result.series.x_m,
            profile_type=result.profile_type,
            k1_n_per_m2=result.k1_n_per_m2,
            k2_n_per_m2=result.k2_n_per_m2,
            transition_length_m=result.transition_length_m,
            segment_length_m=result.segment_length_m,
        )
        self.transition_profile_plot.update_plot(
            k_chart_x,
            [n_per_m2_to_mn_per_m2(value) for value in k_chart_y],
            title="Dynamic transition foundation profile k(x)",
            xlabel="x (m)",
            ylabel="k (MN/m²)",
            annotations=self._build_dynamic_transition_profile_annotations(
                kpi_lines=kpi_lines,
            ),
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self._last_dynamic_stress = self._build_stress_from_dynamic_transition_result(result)
        self._render_stress_results(
            self._last_dynamic_stress,
            title="Peak dynamic bending stress",
            unavailable_note="Sleeper/capping pressures are not available in dynamic transition mode.",
        )
        self.dynamic_summary_panel.update_summary(result)
        self._invalidate_chart_thumbnails({"transition_profile"})
        self._schedule_chart_refresh({"transition_profile"})

    def _render_special_result(self, result: FloatingSlabResult) -> None:
        self.special_floating_slab_plot.update_plot(
            result.frequency_hz,
            result.transmissibility,
            title="Floating slab transmissibility",
            xlabel="f (Hz)",
            ylabel="Transmissibility (—)",
            annotations=self._build_special_chart_annotations(
                result,
                chart_title="Floating slab transmissibility",
            ),
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self.special_floating_slab_attenuation_plot.update_plot(
            result.frequency_hz,
            result.attenuation_db,
            title="Floating slab attenuation",
            xlabel="f (Hz)",
            ylabel="Attenuation (dB)",
            annotations=self._build_special_chart_annotations(
                result,
                chart_title="Floating slab attenuation",
            ),
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self.special_summary_panel.update_summary(result)
        self._invalidate_chart_thumbnails(
            {
                "special_floating_slab_transmissibility",
                "special_floating_slab_attenuation",
            }
        )
        self._schedule_chart_refresh(
            {
                "special_floating_slab_transmissibility",
                "special_floating_slab_attenuation",
            }
        )
        self._render_stress_unavailable(
            title="Stress",
            message="Not available for Special mode analyses.",
        )
        self._mark_chart_results_updated()

    def _render_dynamic_overlay_plots(self) -> None:
        if self._last_dynamic_result is None:
            return
        primary_label = self._dynamic_primary_label or "Primary"
        series_results = [(primary_label, self._last_dynamic_result), *self._dynamic_overlay_results]
        if not self._validate_dynamic_overlay_series(series_results):
            return
        styles = self._build_overlay_styles(len(self._dynamic_overlay_results))
        probe_index = self.probe_selection_combo.currentIndex()

        def build_spatial_series(
            extractor: Callable[[DynamicResult], Sequence[float]],
            transform: Callable[[float], float],
        ) -> list[tuple[list[float], list[float], str]]:
            return [
                (
                    entry_result.spatial.xi_m,
                    [transform(value) for value in extractor(entry_result)],
                    label,
                )
                for label, entry_result in series_results
            ]

        def build_probe_series(
            extractor: Callable[[object], Sequence[float]],
            transform: Callable[[float], float],
            *,
            x_extractor: Callable[[object], Sequence[float]],
        ) -> list[tuple[list[float], list[float], str]]:
            return [
                (
                    list(x_extractor(entry_result.probes[probe_index])),
                    [transform(value) for value in extractor(entry_result.probes[probe_index])],
                    label,
                )
                for label, entry_result in series_results
            ]

        primary_spatial = self._last_dynamic_result.spatial
        deflection_mm = [m_to_mm(value) for value in primary_spatial.deflection_m]
        deflection_annotations = self._build_dynamic_deflection_annotations(primary_spatial.xi_m, deflection_mm)
        primary_moment_knm = [value / 1000.0 for value in primary_spatial.moment_nm]
        moment_annotations = self._build_dynamic_chart_annotations(
            primary_spatial.xi_m,
            primary_moment_knm,
            value_symbol="M",
            value_unit="kN·m",
            axis_symbol="ξ",
            axis_unit="m",
            chart_title="Dynamic moment",
        )
        primary_shear_kn = [n_to_kn(value) for value in primary_spatial.shear_n]
        shear_annotations = self._build_dynamic_chart_annotations(
            primary_spatial.xi_m,
            primary_shear_kn,
            value_symbol="V",
            value_unit="kN",
            axis_symbol="ξ",
            axis_unit="m",
            chart_title="Dynamic shear",
        )
        primary_reaction_kn = [n_to_kn(value) for value in primary_spatial.reaction_n_per_m]
        reaction_annotations = self._build_dynamic_chart_annotations(
            primary_spatial.xi_m,
            primary_reaction_kn,
            value_symbol="R",
            value_unit="kN/m",
            axis_symbol="ξ",
            axis_unit="m",
            chart_title="Dynamic rail support reaction",
        )
        self.dynamic_deflection_plot.update_multi_plot(
            build_spatial_series(lambda item: item.spatial.deflection_m, m_to_mm),
            title="Dynamic deflection",
            xlabel="ξ (m)",
            ylabel="w (mm)",
            styles=styles,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self._apply_panel_annotations(self.dynamic_deflection_plot, deflection_annotations)
        self.dynamic_moment_plot.update_multi_plot(
            build_spatial_series(lambda item: item.spatial.moment_nm, lambda value: value / 1000.0),
            title="Dynamic moment",
            xlabel="ξ (m)",
            ylabel="M (kN·m)",
            styles=styles,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self._apply_panel_annotations(self.dynamic_moment_plot, moment_annotations)
        self.dynamic_shear_plot.update_multi_plot(
            build_spatial_series(lambda item: item.spatial.shear_n, n_to_kn),
            title="Dynamic shear",
            xlabel="ξ (m)",
            ylabel="V (kN)",
            styles=styles,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self._apply_panel_annotations(self.dynamic_shear_plot, shear_annotations)
        self.dynamic_reaction_plot.update_multi_plot(
            build_spatial_series(lambda item: item.spatial.reaction_n_per_m, n_to_kn),
            title="Dynamic rail support reaction",
            xlabel="ξ (m)",
            ylabel="R_support (kN/m)",
            styles=styles,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self._apply_panel_annotations(self.dynamic_reaction_plot, reaction_annotations)
        primary_probe = self._last_dynamic_result.probes[probe_index]
        probe_position = f"Probe: x={primary_probe.position_m:.3f} m"
        primary_time_mm = [m_to_mm(value) for value in primary_probe.deflection_m]
        time_annotations = self._build_dynamic_chart_annotations(
            primary_probe.time_s,
            primary_time_mm,
            value_symbol="w",
            value_unit="mm",
            axis_symbol="t",
            axis_unit="s",
            chart_title="Deflection time history",
            extra_metadata_lines=[probe_position],
        )
        primary_damping_knm = [n_to_kn(value) for value in primary_probe.damping_force_n_per_m]
        damping_annotations = self._build_dynamic_chart_annotations(
            primary_probe.time_s,
            primary_damping_knm,
            value_symbol="F_d",
            value_unit="kN/m",
            axis_symbol="t",
            axis_unit="s",
            chart_title="Damping force time history",
            extra_metadata_lines=[probe_position],
        )
        primary_fft_mm = [value * 1000.0 for value in primary_probe.fft_amplitude]
        fft_annotations = self._build_dynamic_chart_annotations(
            primary_probe.fft_frequency_hz,
            primary_fft_mm,
            value_symbol="|W|",
            value_unit="mm",
            axis_symbol="f",
            axis_unit="Hz",
            chart_title="FFT amplitude",
            extra_metadata_lines=[probe_position],
        )
        primary_psd_mm2_hz = [value * 1_000_000.0 for value in primary_probe.psd]
        psd_annotations = self._build_dynamic_chart_annotations(
            primary_probe.psd_frequency_hz,
            primary_psd_mm2_hz,
            value_symbol="PSD",
            value_unit="mm²/Hz",
            axis_symbol="f",
            axis_unit="Hz",
            chart_title="Welch PSD",
            extra_metadata_lines=[probe_position],
        )
        primary_impedance_mn_per_m2 = [
            n_per_m2_to_mn_per_m2(value)
            for value in primary_probe.impedance_magnitude_n_per_m2
        ]
        impedance_annotations = self._build_dynamic_chart_annotations(
            primary_probe.impedance_frequency_hz,
            primary_impedance_mn_per_m2,
            value_symbol="|Z|",
            value_unit="MN/m²",
            axis_symbol="f",
            axis_unit="Hz",
            chart_title="Support impedance magnitude",
            extra_metadata_lines=[probe_position],
        )
        self.dynamic_time_plot.update_multi_plot(
            build_probe_series(
                lambda probe: probe.deflection_m,
                m_to_mm,
                x_extractor=lambda probe: probe.time_s,
            ),
            title="Deflection time history",
            xlabel="t (s)",
            ylabel="w (mm)",
            styles=styles,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self._apply_panel_annotations(self.dynamic_time_plot, time_annotations)
        self.dynamic_damping_plot.update_multi_plot(
            build_probe_series(
                lambda probe: probe.damping_force_n_per_m,
                n_to_kn,
                x_extractor=lambda probe: probe.time_s,
            ),
            title="Damping force time history",
            xlabel="t (s)",
            ylabel="F_d (kN/m)",
            styles=styles,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self._apply_panel_annotations(self.dynamic_damping_plot, damping_annotations)
        self.dynamic_fft_plot.update_multi_plot(
            build_probe_series(
                lambda probe: probe.fft_amplitude,
                lambda value: value * 1000.0,
                x_extractor=lambda probe: probe.fft_frequency_hz,
            ),
            title="FFT amplitude",
            xlabel="f (Hz)",
            ylabel="|W(f)| (mm)",
            styles=styles,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self._apply_panel_annotations(self.dynamic_fft_plot, fft_annotations)
        self.dynamic_psd_plot.update_multi_plot(
            build_probe_series(
                lambda probe: probe.psd,
                lambda value: value * 1_000_000.0,
                x_extractor=lambda probe: probe.psd_frequency_hz,
            ),
            title="Welch PSD",
            xlabel="f (Hz)",
            ylabel="PSD (mm²/Hz)",
            styles=styles,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self._apply_panel_annotations(self.dynamic_psd_plot, psd_annotations)
        self.dynamic_impedance_plot.update_multi_plot(
            build_probe_series(
                lambda probe: probe.impedance_magnitude_n_per_m2,
                n_per_m2_to_mn_per_m2,
                x_extractor=lambda probe: probe.impedance_frequency_hz,
            ),
            title="Support impedance magnitude",
            xlabel="f (Hz)",
            ylabel="|Z(f)| (MN/m²)",
            styles=styles,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self._apply_panel_annotations(self.dynamic_impedance_plot, impedance_annotations)
        if self._last_dynamic_stress is not None:
            self._render_stress_results(
                self._last_dynamic_stress,
                title="Peak dynamic bending stress (primary run)",
                unavailable_note="Sleeper/capping pressures are not available in dynamic mode.",
            )
        self.dynamic_summary_panel.update_summary(self._last_dynamic_result)
        self._mark_chart_results_updated()

    def _handle_analysis_result(self, result: AnalysisResult) -> None:
        if self._pending_transition_context is not None:
            self._handle_transition_analysis_result(result)
            return
        self.run_button.setEnabled(True)
        self._set_cancel_enabled(False)
        overlay_enabled = (
            hasattr(self, "overlay_checkbox") and self.overlay_checkbox.isChecked()
        )
        if overlay_enabled and self._last_analysis_result is not None:
            summary = self._build_overlay_label()
            label = f"Overlay {len(self._overlay_results) + 1}: {summary}"
            self._overlay_results.append((label, result))
            self._pending_analysis_load_source = None
            self.statusBar().showMessage("Overlay added")
            self._clear_dynamic_state()
            self._set_export_buttons_enabled(self._last_analysis_result is not None)
            self._set_dynamic_export_buttons_enabled(False)
            self._render_overlay_plots()
            self._update_overlay_state()
            return

        self.statusBar().showMessage("Analysis complete")
        self._last_analysis_result = result
        self._clear_static_overlays(render=False)
        self._primary_label = f"Primary: {self._build_overlay_label()}"
        self._clear_dynamic_state()
        self._clear_special_state()
        self._last_envelope_result = None
        self._last_envelope_config = None
        self._last_envelope_load_source = None
        self._last_transition_result = None
        self._last_transition_context = None
        self._last_transition_load_source = None
        if self.worker is not None:
            self._last_analysis_inputs = self.worker.analysis_inputs
            self._last_analysis_config = self.worker.config
            self._last_analysis_mode = self.worker.mode
        self._last_analysis_load_source = self._copy_load_source_metadata(
            self._pending_analysis_load_source
        )
        self._pending_analysis_load_source = None
        self._last_static_mode = StaticMode.SINGLE
        self._set_export_buttons_enabled(True)
        self._set_dynamic_export_buttons_enabled(False)
        self._clear_result_views_for_new_run()
        self._render_analysis_result(result)
        self._update_overlay_state()
        self._refresh_alternative_action_buttons_for_selection()

    def _handle_envelope_progress(self, current: int, total: int) -> None:
        if self._pending_transition_context is not None:
            self.statusBar().showMessage(f"Transition envelope progress: {current}/{total}")
        else:
            self.statusBar().showMessage(f"Envelope progress: {current}/{total}")

    def _handle_envelope_result(self, result: EnvelopeResult) -> None:
        if self._pending_transition_context is not None:
            self._handle_transition_envelope_result(result)
            return
        self.run_button.setEnabled(True)
        self._set_cancel_enabled(False)
        self.statusBar().showMessage("Envelope analysis complete")
        self._clear_dynamic_state()
        self._clear_special_state()
        self._last_analysis_result = None
        self._last_analysis_inputs = None
        self._last_analysis_config = None
        self._last_analysis_mode = None
        self._last_analysis_load_source = None
        self._clear_static_overlays(render=False)
        self._last_envelope_result = result
        if self.envelope_worker is not None:
            self._last_envelope_config = self.envelope_worker.config
            if result.run_metadata and result.run_metadata.get("source_type") == "as5100_fixed_rail":
                governing_config = AS5100RailLoadConfig(
                    model=str(result.run_metadata.get("model", AS5100_MODEL_300LA)),
                    group_count=int(result.run_metadata.get("group_count", 1) or 1),
                    group_spacing_m=float(result.run_metadata.get("group_spacing_m", 12.0) or 12.0),
                    reference_position_m=float(
                        result.run_metadata.get("reference_position_m", 0.0) or 0.0
                    ),
                )
                self._last_envelope_config = replace(
                    self._last_envelope_config,
                    analysis_config=replace(
                        self._last_envelope_config.analysis_config,
                        loads=build_as5100_rail_loads(governing_config),
                    ),
                )
        resolved_load_source = result.run_metadata or self._pending_envelope_load_source
        self._last_envelope_load_source = self._copy_load_source_metadata(resolved_load_source)
        self._pending_envelope_load_source = None
        self._last_transition_result = None
        self._last_transition_context = None
        self._last_transition_load_source = None
        self._last_static_mode = self.static_mode_combo.currentData()
        self._set_export_buttons_enabled(True)
        self._set_dynamic_export_buttons_enabled(False)
        self._clear_result_views_for_new_run()
        self._render_envelope_result(result)
        self._update_overlay_state()
        self._refresh_alternative_action_buttons_for_selection()

    def _handle_transition_analysis_result(self, result: AnalysisResult) -> None:
        self.run_button.setEnabled(True)
        self._set_cancel_enabled(False)
        self.statusBar().showMessage("Transition analysis complete")
        self._clear_dynamic_state()
        self._clear_special_state()
        self._clear_static_overlays(render=False)
        self._last_envelope_result = None
        self._last_envelope_config = None
        self._last_envelope_load_source = None
        self._last_analysis_result = result
        if self.worker is not None:
            self._last_analysis_inputs = self.worker.analysis_inputs
            self._last_analysis_config = self.worker.config
            self._last_analysis_mode = self.worker.mode
        load_source = self._copy_load_source_metadata(self._pending_transition_load_source)
        self._last_analysis_load_source = self._copy_load_source_metadata(load_source)
        self._last_envelope_config = None
        self._last_static_mode = StaticMode.SINGLE
        context = self._pending_transition_context
        if context is None:
            return
        self._pending_transition_context = None
        self._pending_transition_load_source = None
        transition_result = self._build_transition_result_from_analysis(result, context)
        self._last_transition_result = transition_result
        self._last_transition_context = context
        self._last_transition_load_source = self._copy_load_source_metadata(load_source)
        self._set_export_buttons_enabled(True)
        self._set_transition_export_buttons_enabled(True)
        self._clear_result_views_for_new_run()
        self._render_transition_result(transition_result, analysis_result=result)
        self._update_transition_tabs_visible()
        self._refresh_alternative_action_buttons_for_selection()

    def _handle_transition_envelope_result(self, result: EnvelopeResult) -> None:
        self.run_button.setEnabled(True)
        self._set_cancel_enabled(False)
        self.statusBar().showMessage("Transition envelope analysis complete")
        self._clear_dynamic_state()
        self._clear_special_state()
        self._clear_static_overlays(render=False)
        self._last_analysis_result = None
        self._last_analysis_inputs = None
        self._last_analysis_config = None
        self._last_analysis_mode = None
        self._last_analysis_load_source = None
        self._last_envelope_result = result
        if self.envelope_worker is not None:
            self._last_envelope_config = self.envelope_worker.config
        load_source = self._copy_load_source_metadata(result.run_metadata or self._pending_transition_load_source)
        self._last_envelope_load_source = self._copy_load_source_metadata(load_source)
        context = self._pending_transition_context
        if context is None:
            return
        governing_loads = self._as5100_metadata_to_loads(load_source)
        if governing_loads is not None:
            if self._last_envelope_config is not None:
                self._last_envelope_config = replace(
                    self._last_envelope_config,
                    analysis_config=replace(
                        self._last_envelope_config.analysis_config,
                        loads=governing_loads,
                    ),
                )
            context = replace(
                context,
                analysis_config=replace(context.analysis_config, loads=governing_loads),
            )
        self._last_static_mode = (
            StaticMode.ENVELOPE_NUMERICAL
            if context.analysis_mode == AnalysisMode.NUMERICAL
            else StaticMode.ENVELOPE_CLOSED_FORM
        )
        self._pending_transition_context = None
        self._pending_transition_load_source = None
        transition_result = self._build_transition_result_from_envelope(result, context)
        self._last_transition_result = transition_result
        self._last_transition_context = context
        self._last_transition_load_source = self._copy_load_source_metadata(load_source)
        self._set_export_buttons_enabled(True)
        self._set_transition_export_buttons_enabled(True)
        self._clear_result_views_for_new_run()
        self._render_transition_result(transition_result, envelope_result=result)
        self._update_transition_tabs_visible()
        self._refresh_alternative_action_buttons_for_selection()

    def _build_transition_result_from_analysis(
        self,
        result: AnalysisResult,
        context: TransitionContext,
    ) -> TransitionRunResult:
        k_profile = self._resolve_transition_k_profile(
            result.x_m,
            context,
            support_k_eq_n_per_m2=result.summary.support_k_eq_n_per_m2,
        )
        metrics = compute_metrics_from_series(
            x_values=result.x_m,
            deflection_m=result.deflection_m,
            moment_nm=result.moment_nm,
            reaction_n_per_m=result.reaction_n_per_m,
            sleeper_positions_m=result.sleeper_positions_m,
            sleeper_loads_n=result.sleeper_loads_n,
            sleeper_spacing_m=context.analysis_config.sleeper_spacing_m,
            elastic_modulus_pa=context.analysis_config.elastic_modulus_pa,
            moment_inertia_m4=context.analysis_config.moment_inertia_m4,
        )
        foundation_is_winkler = context.analysis_config.foundation_model == FoundationModelType.WINKLER
        energy_metrics = None
        energy_series = None
        if foundation_is_winkler:
            p_ref_n = self._resolve_transition_reference_load_n(context)
            energy_metrics, energy_series = compute_energy_from_series(
                x_values=result.x_m,
                k_profile_n_per_m2=k_profile,
                deflection_m=result.deflection_m,
                moment_nm=result.moment_nm,
                sleeper_spacing_m=context.analysis_config.sleeper_spacing_m,
                elastic_modulus_pa=context.analysis_config.elastic_modulus_pa,
                moment_inertia_m4=context.analysis_config.moment_inertia_m4,
                p_ref_n=p_ref_n,
            )
        series = build_series_from_single(
            x_values=result.x_m,
            k_profile_n_per_m2=k_profile,
            deflection_m=result.deflection_m,
            moment_nm=result.moment_nm,
            reaction_n_per_m=result.reaction_n_per_m,
            shear_n=result.shear_n,
        )
        domain_length = context.domain_m[1] - context.domain_m[0]
        return TransitionRunResult(
            mode=context.run_mode,
            profile_type=context.profile_type,
            template_name=context.template_name,
            preset_name=context.preset_name,
            k1_n_per_m2=context.k1_n_per_m2,
            k2_n_per_m2=context.k2_n_per_m2,
            transition_length_m=context.transition_length_m,
            segment_length_m=context.segment_length_m,
            domain_length_m=domain_length,
            metrics=metrics,
            series=series,
            k_units="N/m^2",
            k_representation=(
                "continuous_per_unit_length" if foundation_is_winkler else "model_dependent_per_unit_length"
            ),
            foundation_reaction_law=(
                "q_f(x)=k(x)w(x) [N/m]"
                if foundation_is_winkler
                else "model-dependent (energy metrics disabled for non-Winkler)"
            ),
            transition_metrics_schema_version=2,
            energy_metrics=energy_metrics,
            energy_series=energy_series,
        )

    def _build_transition_result_from_envelope(
        self,
        result: EnvelopeResult,
        context: TransitionContext,
    ) -> TransitionRunResult:
        k_profile = self._resolve_transition_k_profile(
            result.x_m,
            context,
            support_k_eq_n_per_m2=None,
        )
        metrics = compute_metrics_from_envelope(
            x_values=result.x_m,
            deflection_max_m=result.deflection_max_m,
            deflection_min_m=result.deflection_min_m,
            moment_max_nm=result.moment_max_nm,
            moment_min_nm=result.moment_min_nm,
            reaction_max_n_per_m=result.reaction_max_n_per_m,
            reaction_min_n_per_m=result.reaction_min_n_per_m,
            sleeper_positions_m=result.sleeper_positions_m,
            sleeper_loads_max_n=result.sleeper_loads_max_n,
            sleeper_spacing_m=context.analysis_config.sleeper_spacing_m,
            elastic_modulus_pa=context.analysis_config.elastic_modulus_pa,
            moment_inertia_m4=context.analysis_config.moment_inertia_m4,
        )
        foundation_is_winkler = context.analysis_config.foundation_model == FoundationModelType.WINKLER
        energy_metrics = None
        energy_series = None
        if foundation_is_winkler:
            p_ref_n = self._resolve_transition_reference_load_n(context)
            energy_metrics, energy_series = compute_energy_from_envelope(
                x_values=result.x_m,
                k_profile_n_per_m2=k_profile,
                deflection_max_m=result.deflection_max_m,
                deflection_min_m=result.deflection_min_m,
                moment_max_nm=result.moment_max_nm,
                moment_min_nm=result.moment_min_nm,
                sleeper_spacing_m=context.analysis_config.sleeper_spacing_m,
                elastic_modulus_pa=context.analysis_config.elastic_modulus_pa,
                moment_inertia_m4=context.analysis_config.moment_inertia_m4,
                p_ref_n=p_ref_n,
            )
        series = build_series_from_envelope(
            x_values=result.x_m,
            k_profile_n_per_m2=k_profile,
            deflection_max_m=result.deflection_max_m,
            deflection_min_m=result.deflection_min_m,
            moment_max_nm=result.moment_max_nm,
            moment_min_nm=result.moment_min_nm,
            reaction_max_n_per_m=result.reaction_max_n_per_m,
            reaction_min_n_per_m=result.reaction_min_n_per_m,
            shear_max_n=result.shear_max_n,
            shear_min_n=result.shear_min_n,
        )
        domain_length = context.domain_m[1] - context.domain_m[0]
        return TransitionRunResult(
            mode=context.run_mode,
            profile_type=context.profile_type,
            template_name=context.template_name,
            preset_name=context.preset_name,
            k1_n_per_m2=context.k1_n_per_m2,
            k2_n_per_m2=context.k2_n_per_m2,
            transition_length_m=context.transition_length_m,
            segment_length_m=context.segment_length_m,
            domain_length_m=domain_length,
            metrics=metrics,
            series=series,
            k_units="N/m^2",
            k_representation=(
                "continuous_per_unit_length" if foundation_is_winkler else "model_dependent_per_unit_length"
            ),
            foundation_reaction_law=(
                "q_f(x)=k(x)w(x) [N/m]"
                if foundation_is_winkler
                else "model-dependent (energy metrics disabled for non-Winkler)"
            ),
            transition_metrics_schema_version=2,
            energy_metrics=energy_metrics,
            energy_series=energy_series,
        )

    @staticmethod
    def _resolve_transition_reference_load_n(context: TransitionContext) -> float | None:
        if not context.analysis_config.loads:
            return None
        reference = max(abs(load.load_newtons) for load in context.analysis_config.loads)
        return reference if reference > 0.0 else None

    def _resolve_transition_k_profile(
        self,
        x_values: Sequence[float],
        context: TransitionContext,
        *,
        support_k_eq_n_per_m2: float | None,
    ) -> list[float]:
        if context.profile_type != TransitionProfileType.UNIFORM:
            if context.k_profile_n_per_m2 is not None and len(context.k_profile_n_per_m2) == len(x_values):
                return list(context.k_profile_n_per_m2)
            return build_transition_profile(
                x_values=x_values,
                profile_type=context.profile_type,
                k1_n_per_m2=context.k1_n_per_m2,
                k2_n_per_m2=context.k2_n_per_m2,
                transition_length_m=context.transition_length_m,
                segment_length_m=context.segment_length_m,
            )
        k_value = support_k_eq_n_per_m2
        if k_value is None and context.analysis_config.foundation_model != FoundationModelType.WINKLER:
            k_pad = context.analysis_config.railpad_stiffness_n_per_m
            k_bed = context.analysis_config.trackbed_stiffness_n_per_m
            if k_pad is not None and k_bed is not None:
                pad_per_length = per_support_to_per_length(k_pad, context.analysis_config.sleeper_spacing_m)
                bed_per_length = per_support_to_per_length(k_bed, context.analysis_config.sleeper_spacing_m)
                k_value = equivalent_series_stiffness(pad_per_length, bed_per_length)
        if k_value is None:
            k_value = context.k1_n_per_m2
        return [k_value for _ in x_values]

    def _render_transition_profile_plot(
        self,
        result: TransitionRunResult,
        *,
        load_markers: Sequence[LoadMarker] | None,
    ) -> None:
        k_chart_x, k_chart_y = self._transition_k_chart_series(
            x_values=result.series.x_m,
            profile_type=result.profile_type,
            k1_n_per_m2=result.k1_n_per_m2,
            k2_n_per_m2=result.k2_n_per_m2,
            transition_length_m=result.transition_length_m,
            segment_length_m=result.segment_length_m,
        )
        k_values_mn = [n_per_m2_to_mn_per_m2(value) for value in k_chart_y]
        if result.energy_series is not None:
            try:
                self.transition_profile_plot.update_multi_plot_dual_axis(
                    primary_series=[(k_chart_x, k_values_mn, "k(x)")],
                    secondary_series=[(result.series.x_m, result.energy_series.u_total_j_per_m, "u_total")],
                    title="Transition profile and energy concentration",
                    xlabel="x (m)",
                    primary_ylabel="k (MN/m²)",
                    secondary_ylabel="u_total (J/m)",
                    critical_labels=self._chart_extrema_labels_visible(),
                    load_markers=self._chart_input_load_markers(load_markers),
                )
                return
            except Exception as exc:
                LOGGER.warning("Transition k(x) energy overlay failed; falling back to k-only plot: %s", exc)
        self.transition_profile_plot.update_plot(
            k_chart_x,
            k_values_mn,
            title="Foundation modulus profile k(x)",
            xlabel="x (m)",
            ylabel="k (MN/m²)",
            critical_labels=self._chart_extrema_labels_visible(),
            load_markers=self._chart_input_load_markers(load_markers),
        )

    def _render_transition_result(
        self,
        result: TransitionRunResult,
        *,
        analysis_result: AnalysisResult | None = None,
        envelope_result: EnvelopeResult | None = None,
    ) -> None:
        if envelope_result is not None:
            self._render_envelope_result(envelope_result, transition_envelope_style=True)
        elif analysis_result is not None:
            self._render_analysis_result(analysis_result)

        transition_load_markers = self._build_load_markers_from_analysis_config(
            self._last_transition_context.analysis_config if self._last_transition_context is not None else None,
            reference=_enum_value(result.mode) == TransitionRunMode.ENVELOPE.value,
        )
        self.transition_summary_panel.update_summary(result)
        self._render_transition_profile_plot(result, load_markers=transition_load_markers)
        self._apply_transition_annotations_to_visible_charts(result)
        self._invalidate_chart_thumbnails({"transition_profile"})
        self._schedule_chart_refresh({"transition_profile"})

    def _handle_dynamic_result(self, result: object) -> None:
        if not isinstance(result, (DynamicResult, DynamicTransitionResult, DippedJointResult)):
            self.run_button.setEnabled(True)
            self._set_cancel_enabled(False)
            self._pending_dynamic_load_source = None
            self._set_export_buttons_enabled(
                self._last_analysis_result is not None
                or self._last_envelope_result is not None
                or self._last_transition_result is not None
            )
            self._set_dynamic_export_buttons_enabled(
                self._last_dynamic_result is not None
                or self._last_dynamic_transition_result is not None
                or self._last_dipped_joint_result is not None
            )
            self.statusBar().showMessage("Analysis failed")
            QMessageBox.warning(
                self,
                "Analysis error",
                "Dynamic analysis returned an unexpected result type.",
            )
            return
        if isinstance(result, DippedJointResult):
            self._clear_dynamic_overlays(render=False)
            self._pending_dynamic_load_source = None
            self._handle_dipped_joint_result(result)
            return
        if isinstance(result, DynamicTransitionResult):
            self._clear_dynamic_overlays(render=False)
            self._handle_dynamic_transition_result(result)
            return
        self.run_button.setEnabled(True)
        self._set_cancel_enabled(False)
        current_config: DynamicConfig | None = None
        current_mode: DynamicMode | None = None
        if self.dynamic_worker is not None and isinstance(self.dynamic_worker.config, DynamicConfig):
            current_config = self.dynamic_worker.config
            current_mode = self.dynamic_worker.mode
        if current_mode is None:
            try:
                current_mode = self._coerce_dynamic_mode(self.dynamic_mode_combo.currentData())
            except ValueError:
                current_mode = None

        overlay_enabled = (
            hasattr(self, "overlay_checkbox")
            and self.overlay_checkbox.isChecked()
            and self.analysis_type_combo.currentData() == AnalysisType.DYNAMIC
        )
        has_primary = self._last_dynamic_result is not None
        same_mode = current_mode is not None and self._dynamic_overlay_mode == current_mode
        if overlay_enabled and has_primary and same_mode:
            label = (
                f"Overlay {len(self._dynamic_overlay_results) + 1}: "
                f"{self._build_dynamic_overlay_label(config=current_config, mode=current_mode)}"
            )
            self._dynamic_overlay_results.append((label, result))
            self._pending_dynamic_load_source = None
            self.statusBar().showMessage("Overlay added")
            self._last_dipped_joint_result = None
            self._last_dipped_joint_config = None
            self._set_export_buttons_enabled(False)
            self._set_dynamic_export_buttons_enabled(True)
            self._render_dynamic_overlay_plots()
            self._update_overlay_state()
            return

        self.statusBar().showMessage("Dynamic analysis complete")
        self._clear_analysis_state()
        self._clear_dynamic_overlays(render=False)
        self._clear_special_state()
        self._last_dynamic_result = result
        self._last_dynamic_config = current_config
        self._last_dynamic_load_source = self._copy_load_source_metadata(
            self._pending_dynamic_load_source
        )
        self._pending_dynamic_load_source = None
        self._last_dynamic_mode = current_mode
        self._dynamic_overlay_mode = current_mode
        self._dynamic_primary_label = (
            f"Primary: {self._build_dynamic_overlay_label(config=current_config, mode=current_mode)}"
        )
        self._last_dipped_joint_result = None
        self._last_dipped_joint_config = None
        self._last_dynamic_transition_result = None
        self._last_dynamic_transition_config = None
        self._last_dynamic_transition_load_source = None
        self._set_export_buttons_enabled(False)
        self._set_dynamic_export_buttons_enabled(True)
        self._clear_result_views_for_new_run()
        self._render_dynamic_result(result)
        self._update_overlay_state()
        self._refresh_alternative_action_buttons_for_selection()

    def _handle_dipped_joint_result(self, result: DippedJointResult) -> None:
        self.run_button.setEnabled(True)
        self._set_cancel_enabled(False)
        self.statusBar().showMessage("Dipped joint analysis complete")
        self._last_dipped_joint_result = result
        self._clear_analysis_state()
        self._clear_dynamic_overlays(render=False)
        self._clear_special_state()
        self._last_dynamic_result = None
        self._last_dynamic_config = None
        self._last_dynamic_load_source = None
        self._last_dynamic_transition_result = None
        self._last_dynamic_transition_config = None
        self._last_dynamic_transition_load_source = None
        self._last_dynamic_mode = DynamicMode.DIPPED_JOINT
        self._last_dipped_joint_config = None
        if self.dynamic_worker is not None and isinstance(self.dynamic_worker.config, DippedJointConfig):
            self._last_dipped_joint_config = self.dynamic_worker.config
        self._set_export_buttons_enabled(False)
        self._set_dynamic_export_buttons_enabled(True)
        self._clear_result_views_for_new_run()
        self.dipped_joint_summary_panel.update_summary(result)
        self._render_stress_unavailable(
            title="Peak dynamic bending stress",
            message="Not available for dipped-joint mode.",
        )
        if hasattr(self, "tab_widget"):
            if hasattr(self, "chart_view_combo") and self.chart_view_combo.currentData() == "all":
                self._apply_chart_view("all")
            else:
                self.tab_widget.setCurrentIndex(self.dipped_joint_summary_tab_index)
        self._refresh_chart_grid_if_visible()
        self._refresh_alternative_action_buttons_for_selection()

    def _handle_special_result(self, result: object) -> None:
        if not isinstance(result, FloatingSlabResult):
            self.run_button.setEnabled(True)
            self._set_cancel_enabled(False)
            self._set_export_buttons_enabled(
                self._last_analysis_result is not None
                or self._last_envelope_result is not None
                or self._last_transition_result is not None
            )
            self._set_dynamic_export_buttons_enabled(
                self._last_dynamic_result is not None
                or self._last_dynamic_transition_result is not None
                or self._last_dipped_joint_result is not None
            )
            self.statusBar().showMessage("Analysis failed")
            QMessageBox.warning(
                self,
                "Analysis error",
                "Special analysis returned an unexpected result type.",
            )
            return
        self.run_button.setEnabled(True)
        self._set_cancel_enabled(False)
        self.statusBar().showMessage("Special analysis complete")
        self._clear_analysis_state()
        self._clear_dynamic_state()
        self._last_special_result = result
        if self.special_worker is not None:
            self._last_special_config = self.special_worker.config
        self._set_export_buttons_enabled(False)
        self._set_dynamic_export_buttons_enabled(False)
        self._clear_result_views_for_new_run()
        self._render_special_result(result)
        if hasattr(self, "tab_widget"):
            if hasattr(self, "chart_view_combo") and self.chart_view_combo.currentData() == "all":
                self._apply_chart_view("all")
            else:
                self.tab_widget.setCurrentIndex(self.special_summary_tab_index)
        self._refresh_chart_grid_if_visible()
        self._refresh_alternative_action_buttons_for_selection()

    def _handle_dynamic_transition_result(self, result: DynamicTransitionResult) -> None:
        self.run_button.setEnabled(True)
        self._set_cancel_enabled(False)
        self.statusBar().showMessage("Dynamic transition analysis complete")
        self._clear_analysis_state()
        self._clear_dynamic_overlays(render=False)
        self._clear_special_state()
        self._last_dynamic_transition_result = result
        self._last_dynamic_transition_config = None
        if self.dynamic_worker is not None and isinstance(self.dynamic_worker.config, DynamicTransitionConfig):
            self._last_dynamic_transition_config = self.dynamic_worker.config
        load_source = self._copy_load_source_metadata(self._pending_dynamic_load_source)
        self._last_dynamic_transition_load_source = self._copy_load_source_metadata(load_source)
        self._last_dynamic_result = result.representative
        self._last_dynamic_config = None
        self._last_dynamic_load_source = self._copy_load_source_metadata(load_source)
        self._pending_dynamic_load_source = None
        self._last_dynamic_mode = DynamicMode.TRANSITION
        self._last_dipped_joint_result = None
        self._last_dipped_joint_config = None
        self._set_export_buttons_enabled(False)
        self._set_dynamic_export_buttons_enabled(True)
        if hasattr(self, "tab_widget"):
            self._set_dynamic_tabs_visible(True, mode=DynamicMode.TRANSITION)
        self._clear_result_views_for_new_run()
        self._render_dynamic_transition_result(result)
        if hasattr(self, "tab_widget"):
            if hasattr(self, "chart_view_combo") and self.chart_view_combo.currentData() == "all":
                self._apply_chart_view("all")
            else:
                self.tab_widget.setCurrentIndex(self.dynamic_summary_tab_index)
        self._refresh_chart_grid_if_visible()
        self._refresh_alternative_action_buttons_for_selection()

    def _handle_analysis_error(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self._set_cancel_enabled(False)
        self._pending_transition_context = None
        self._set_export_buttons_enabled(
            self._last_analysis_result is not None
            or self._last_envelope_result is not None
            or self._last_transition_result is not None
        )
        self._set_dynamic_export_buttons_enabled(
            self._last_dynamic_result is not None
            or self._last_dynamic_transition_result is not None
            or self._last_dipped_joint_result is not None
        )
        self.statusBar().showMessage("Analysis failed")
        QMessageBox.warning(self, "Analysis error", message)

    def _handle_analysis_cancelled(self) -> None:
        self.run_button.setEnabled(True)
        self._set_cancel_enabled(False)
        self._pending_transition_context = None
        self._set_export_buttons_enabled(
            self._last_analysis_result is not None
            or self._last_envelope_result is not None
            or self._last_transition_result is not None
        )
        self._set_dynamic_export_buttons_enabled(
            self._last_dynamic_result is not None
            or self._last_dynamic_transition_result is not None
            or self._last_dipped_joint_result is not None
        )
        self.statusBar().showMessage("Analysis cancelled")
        QMessageBox.information(self, "Analysis cancelled", "The analysis was cancelled.")

    def _set_cancel_enabled(self, enabled: bool) -> None:
        if self.cancel_button is not None:
            self.cancel_button.setEnabled(enabled)

    def _cancel_long_run(self) -> None:
        if self.envelope_worker is None:
            self.statusBar().showMessage("No cancellable analysis is running.")
            return
        self.statusBar().showMessage("Cancelling analysis...")
        self._set_cancel_enabled(False)
        self.envelope_worker.request_cancel()

    def _validate_plot_series(self, result: AnalysisResult) -> bool:
        if not result.x_m:
            QMessageBox.warning(self, "Analysis error", "No x-axis data available for plotting.")
            return False
        required = {
            "Deflection": result.deflection_m,
            "Moment": result.moment_nm,
            "Shear": result.shear_n,
        }
        for name, series in required.items():
            if not series:
                QMessageBox.warning(self, "Analysis error", f"{name} results are missing.")
                return False
            if len(series) != len(result.x_m):
                QMessageBox.warning(
                    self,
                    "Analysis error",
                    f"{name} results do not align with the x-axis sample count.",
                )
                return False
        return True

    def _validate_envelope_series(self, result: EnvelopeResult) -> bool:
        if not result.x_m:
            QMessageBox.warning(self, "Analysis error", "No x-axis data available for plotting.")
            return False
        series_map = {
            "Deflection max": result.deflection_max_m,
            "Deflection min": result.deflection_min_m,
            "Moment max": result.moment_max_nm,
            "Moment min": result.moment_min_nm,
            "Shear max": result.shear_max_n,
            "Shear min": result.shear_min_n,
            "Reaction max": result.reaction_max_n_per_m,
            "Reaction min": result.reaction_min_n_per_m,
        }
        for name, series in series_map.items():
            if not series:
                QMessageBox.warning(self, "Analysis error", f"{name} results are missing.")
                return False
            if len(series) != len(result.x_m):
                QMessageBox.warning(
                    self,
                    "Analysis error",
                    f"{name} results do not align with the x-axis sample count.",
                )
                return False
        if len(result.sleeper_positions_m) != len(result.sleeper_loads_max_n):
            QMessageBox.warning(self, "Analysis error", "Sleeper load results are inconsistent.")
            return False
        return True

    def _compute_sanity_checks(self, result: AnalysisResult) -> dict[str, tuple[str, bool | None]]:
        checks: dict[str, tuple[str, bool | None]] = {}
        if self._last_analysis_inputs is None:
            return checks
        x = np.asarray(result.x_m)
        if x.size == 0:
            return checks

        loads = self._last_analysis_inputs.loads
        total_load = sum(load.load_newtons for load in loads)
        if self._last_analysis_config is not None and self._last_analysis_config.use_two_rail:
            right_loads = self._last_analysis_config.right_loads
            if right_loads:
                total_load += sum(load.load_newtons for load in right_loads)
            else:
                total_load *= 2.0

        if result.left_reaction_n_per_m is not None and result.right_reaction_n_per_m is not None:
            reaction = np.asarray(result.left_reaction_n_per_m) + np.asarray(result.right_reaction_n_per_m)
        else:
            reaction = np.asarray(result.reaction_n_per_m)
        total_reaction = float(np.trapezoid(reaction, x))
        if total_load > 0:
            equilibrium_error = abs(total_reaction - total_load) / total_load
            checks["equilibrium"] = (f"{equilibrium_error * 100:.1f}%", equilibrium_error < 0.05)
        else:
            checks["equilibrium"] = ("n/a", None)

        if self._loads_are_symmetric(loads):
            deflection = np.asarray(result.deflection_m)
            mirrored = np.interp(-x, x, deflection)
            max_defl = max(np.max(np.abs(deflection)), 1e-12)
            symmetry_error = float(np.max(np.abs(deflection - mirrored)) / max_defl)
            checks["symmetry"] = (f"{symmetry_error * 100:.1f}%", symmetry_error < 0.05)
        else:
            checks["symmetry"] = ("n/a", None)

        ei = self._last_analysis_inputs.elastic_modulus_pa * self._last_analysis_inputs.moment_inertia_m4
        if ei <= 0:
            checks["moment_coherence"] = ("n/a", None)
            checks["shear_coherence"] = ("n/a", None)
            return checks

        deflection = np.asarray(result.deflection_m)
        moment = np.asarray(result.moment_nm)
        shear = np.asarray(result.shear_n)
        if deflection.size >= 3 and moment.size == deflection.size:
            curvature = np.gradient(np.gradient(deflection, x), x)
            moment_from_deflection = -ei * curvature
            denom = float(np.linalg.norm(moment)) or 1.0
            moment_error = float(np.linalg.norm(moment_from_deflection - moment) / denom)
            checks["moment_coherence"] = (f"{moment_error * 100:.1f}%", moment_error < 0.10)
        else:
            checks["moment_coherence"] = ("n/a", None)

        if moment.size >= 3 and shear.size == moment.size:
            shear_from_moment = np.gradient(moment, x)
            denom = float(np.linalg.norm(shear)) or 1.0
            shear_error = float(np.linalg.norm(shear_from_moment - shear) / denom)
            checks["shear_coherence"] = (f"{shear_error * 100:.1f}%", shear_error < 0.10)
        else:
            checks["shear_coherence"] = ("n/a", None)

        return checks

    @staticmethod
    def _loads_are_symmetric(loads: Sequence[PointLoad], tol: float = 1e-6) -> bool:
        matched = [False] * len(loads)
        for i, load in enumerate(loads):
            if matched[i]:
                continue
            target_pos = -load.position_m
            target_load = load.load_newtons
            found = False
            for j in range(i, len(loads)):
                if matched[j]:
                    continue
                other = loads[j]
                if abs(other.position_m - target_pos) <= tol and abs(other.load_newtons - target_load) <= tol:
                    matched[i] = True
                    matched[j] = True
                    found = True
                    break
            if not found:
                return False
        return True

    def _validate_overlay_series(self, results: Sequence[tuple[str, AnalysisResult]]) -> bool:
        if not results:
            QMessageBox.warning(self, "Analysis error", "No overlay results available for plotting.")
            return False
        for _label, result in results:
            if not self._validate_plot_series(result):
                return False
        return True

    def _validate_dynamic_overlay_series(self, results: Sequence[tuple[str, DynamicResult]]) -> bool:
        if not results:
            QMessageBox.warning(self, "Analysis error", "No dynamic overlay results available for plotting.")
            return False
        probe_index = self.probe_selection_combo.currentIndex()
        for _label, result in results:
            spatial = result.spatial
            xi = spatial.xi_m
            if not xi:
                QMessageBox.warning(self, "Analysis error", "Dynamic results are missing the spatial axis.")
                return False
            required_series = {
                "Deflection": spatial.deflection_m,
                "Moment": spatial.moment_nm,
                "Shear": spatial.shear_n,
                "Reaction": spatial.reaction_n_per_m,
            }
            for name, values in required_series.items():
                if not values:
                    QMessageBox.warning(self, "Analysis error", f"Dynamic {name.lower()} results are missing.")
                    return False
                if len(values) != len(xi):
                    QMessageBox.warning(
                        self,
                        "Analysis error",
                        f"Dynamic {name.lower()} results do not align with the spatial axis.",
                    )
                    return False
            if not result.probes:
                QMessageBox.warning(self, "Analysis error", "Dynamic probe results are missing.")
                return False
            if probe_index < 0 or probe_index >= len(result.probes):
                QMessageBox.warning(
                    self,
                    "Analysis error",
                    "Selected probe index is not available in one or more dynamic overlays.",
                )
                return False
        return True

    def _selected_probe_series(self, result: DynamicResult):
        if not result.probes:
            return None
        index = self.probe_selection_combo.currentIndex()
        if index < 0 or index >= len(result.probes):
            index = 0
        return result.probes[index]

    def _update_dynamic_probe_plots(self) -> None:
        if self._last_dynamic_result is None:
            return
        if self._dynamic_overlay_results:
            self._render_dynamic_overlay_plots()
            return
        probe = self._selected_probe_series(self._last_dynamic_result)
        if probe is None:
            return
        probe_position = f"Probe: x={probe.position_m:.3f} m"
        time_deflection_mm = [m_to_mm(value) for value in probe.deflection_m]
        time_deflection_annotations = self._build_dynamic_chart_annotations(
            probe.time_s,
            time_deflection_mm,
            value_symbol="w",
            value_unit="mm",
            axis_symbol="t",
            axis_unit="s",
            chart_title="Deflection time history",
            extra_metadata_lines=[probe_position],
        )
        damping_knm = [n_to_kn(value) for value in probe.damping_force_n_per_m]
        damping_annotations = self._build_dynamic_chart_annotations(
            probe.time_s,
            damping_knm,
            value_symbol="F_d",
            value_unit="kN/m",
            axis_symbol="t",
            axis_unit="s",
            chart_title="Damping force time history",
            extra_metadata_lines=[probe_position],
        )
        fft_mm = [value * 1000.0 for value in probe.fft_amplitude]
        fft_annotations = self._build_dynamic_chart_annotations(
            probe.fft_frequency_hz,
            fft_mm,
            value_symbol="|W|",
            value_unit="mm",
            axis_symbol="f",
            axis_unit="Hz",
            chart_title="FFT amplitude",
            extra_metadata_lines=[probe_position],
        )
        psd_mm2_hz = [value * 1_000_000.0 for value in probe.psd]
        psd_annotations = self._build_dynamic_chart_annotations(
            probe.psd_frequency_hz,
            psd_mm2_hz,
            value_symbol="PSD",
            value_unit="mm²/Hz",
            axis_symbol="f",
            axis_unit="Hz",
            chart_title="Welch PSD",
            extra_metadata_lines=[probe_position],
        )
        impedance_mn_per_m2 = [
            n_per_m2_to_mn_per_m2(value)
            for value in probe.impedance_magnitude_n_per_m2
        ]
        impedance_annotations = self._build_dynamic_chart_annotations(
            probe.impedance_frequency_hz,
            impedance_mn_per_m2,
            value_symbol="|Z|",
            value_unit="MN/m²",
            axis_symbol="f",
            axis_unit="Hz",
            chart_title="Support impedance magnitude",
            extra_metadata_lines=[probe_position],
        )
        self.dynamic_time_plot.update_plot(
            probe.time_s,
            time_deflection_mm,
            title="Deflection time history",
            xlabel="t (s)",
            ylabel="w (mm)",
            annotations=time_deflection_annotations,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self.dynamic_damping_plot.update_plot(
            probe.time_s,
            damping_knm,
            title="Damping force time history",
            xlabel="t (s)",
            ylabel="F_d (kN/m)",
            annotations=damping_annotations,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self.dynamic_fft_plot.update_plot(
            probe.fft_frequency_hz,
            fft_mm,
            title="FFT amplitude",
            xlabel="f (Hz)",
            ylabel="|W(f)| (mm)",
            annotations=fft_annotations,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self.dynamic_psd_plot.update_plot(
            probe.psd_frequency_hz,
            psd_mm2_hz,
            title="Welch PSD",
            xlabel="f (Hz)",
            ylabel="PSD (mm²/Hz)",
            annotations=psd_annotations,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self.dynamic_impedance_plot.update_plot(
            probe.impedance_frequency_hz,
            impedance_mn_per_m2,
            title="Support impedance magnitude",
            xlabel="f (Hz)",
            ylabel="|Z(f)| (MN/m²)",
            annotations=impedance_annotations,
            critical_labels=self._chart_extrema_labels_visible(),
        )
        self._mark_chart_probe_updated()

    def _update_two_rail_tabs(
        self,
        result: AnalysisResult,
        *,
        load_markers: Sequence[LoadMarker] | None = None,
    ) -> None:
        has_two_rail = (
            result.left_deflection_m is not None
            and result.right_deflection_m is not None
            and result.left_moment_nm is not None
            and result.right_moment_nm is not None
        )
        self.tab_widget.setTabVisible(self.rail_deflection_tab_index, has_two_rail)
        self.tab_widget.setTabVisible(self.rail_moment_tab_index, has_two_rail)
        if not has_two_rail:
            return

        x_m = result.x_m
        self.rail_deflection_plot.update_comparison_plot(
            x_m,
            [m_to_mm(value) for value in result.left_deflection_m],
            [m_to_mm(value) for value in result.right_deflection_m],
            title="Rail deflection comparison",
            xlabel="x (m)",
            ylabel="y (mm)",
            critical_labels=self._chart_extrema_labels_visible(),
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self._apply_panel_annotations(
            self.rail_deflection_plot,
            self._build_static_chart_annotations(result, chart_title="Rail deflection comparison"),
        )
        self.rail_moment_plot.update_comparison_plot(
            x_m,
            [value / 1000.0 for value in result.left_moment_nm],
            [value / 1000.0 for value in result.right_moment_nm],
            title="Rail moment comparison",
            xlabel="x (m)",
            ylabel="M (kN·m)",
            critical_labels=self._chart_extrema_labels_visible(),
            load_markers=self._chart_input_load_markers(load_markers),
        )
        self._apply_panel_annotations(
            self.rail_moment_plot,
            self._build_static_chart_annotations(result, chart_title="Rail moment comparison"),
        )

    def _set_export_buttons_enabled(self, enabled: bool) -> None:
        self.export_analysis_button.setEnabled(enabled)
        self.export_config_button.setEnabled(enabled)
        self.export_sleeper_button.setEnabled(enabled)
        self._set_transition_export_buttons_enabled(enabled and self._last_transition_result is not None)

    def _set_transition_export_buttons_enabled(self, enabled: bool) -> None:
        self.export_transition_metrics_button.setEnabled(enabled)
        self.export_transition_series_button.setEnabled(enabled)
        self.export_transition_config_button.setEnabled(enabled)

    def _set_dynamic_export_buttons_enabled(self, enabled: bool) -> None:
        try:
            mode = self._coerce_dynamic_mode(self.dynamic_mode_combo.currentData())
        except ValueError:
            mode = DynamicMode.STEADY_STATE
        is_dipped_joint = mode == DynamicMode.DIPPED_JOINT
        is_transition = mode == DynamicMode.TRANSITION
        self.export_dynamic_time_button.setEnabled(enabled and not is_dipped_joint)
        self.export_dynamic_fft_button.setEnabled(enabled and not is_dipped_joint)
        self.export_dynamic_psd_button.setEnabled(enabled and not is_dipped_joint)
        self.export_dynamic_transition_metrics_button.setEnabled(
            enabled and is_transition and self._last_dynamic_transition_result is not None
        )
        self.export_dynamic_transition_series_button.setEnabled(
            enabled and is_transition and self._last_dynamic_transition_result is not None
        )
        self.export_dynamic_transition_config_button.setEnabled(
            enabled and is_transition and self._last_dynamic_transition_config is not None
        )
        self.export_dipped_joint_button.setEnabled(enabled and is_dipped_joint)

    def _require_analysis_result(self) -> bool:
        if self._last_envelope_result is not None:
            return True
        if self._last_analysis_result is None or self._last_analysis_inputs is None:
            QMessageBox.warning(
                self,
                "No analysis results",
                "Run an analysis before exporting CSV files.",
            )
            return False
        return True

    def _require_analysis_config(self) -> bool:
        if self._last_envelope_result is not None:
            if self._last_envelope_config is None:
                QMessageBox.warning(
                    self,
                    "No envelope config",
                    "Run an envelope analysis before exporting the run configuration.",
                )
                return False
            return True
        if not self._require_analysis_result():
            return False
        if self._last_analysis_config is None:
            QMessageBox.warning(
                self,
                "No analysis config",
                "Run an analysis before exporting the run configuration.",
            )
            return False
        return True

    def _require_transition_result(self) -> bool:
        if self._last_transition_result is None:
            QMessageBox.warning(
                self,
                "No transition results",
                "Run a transition analysis before exporting transition outputs.",
            )
            return False
        return True

    def _require_dynamic_result(self) -> bool:
        if self._last_dynamic_result is None:
            QMessageBox.warning(
                self,
                "No dynamic results",
                "Run a dynamic analysis before exporting CSV files.",
            )
            return False
        return True

    def _require_dynamic_transition_result(self) -> bool:
        if self._last_dynamic_transition_result is None:
            QMessageBox.warning(
                self,
                "No dynamic transition results",
                "Run a dynamic transition analysis before exporting transition outputs.",
            )
            return False
        return True

    def _require_dipped_joint_result(self) -> bool:
        if self._last_dipped_joint_result is None:
            QMessageBox.warning(
                self,
                "No dipped joint results",
                "Run a dipped joint analysis before exporting CSV files.",
            )
            return False
        return True

    def _export_analysis_csv(self) -> None:
        if not self._require_analysis_result():
            return
        if self._last_envelope_result is not None:
            result = self._last_envelope_result
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export envelope analysis CSV",
                "envelope_analysis.csv",
                "CSV Files (*.csv)",
            )
            if not path:
                return
            try:
                stress = self._last_envelope_stress or self._build_stress_from_envelope_result(result)
                write_envelope_analysis_csv(path, result, stress_results=stress)
                envelope_mode = (
                    self._last_envelope_config.mode.value
                    if self._last_envelope_config is not None
                    else AnalysisMode.CLOSED_FORM.value
                )
                metadata_payload = {
                    "envelope_config": asdict(self._last_envelope_config) if self._last_envelope_config else None,
                    "load_source": self._last_static_load_source_metadata(),
                }
                metadata_payload = self._extend_envelope_metadata_payload(
                    metadata_payload,
                    result=result,
                    load_source=metadata_payload["load_source"],
                )
                metadata_payload.update(self._stress_metadata_payload(stress))
                write_export_metadata(path, solver_mode=envelope_mode, inputs_payload=metadata_payload, units="SI")
            except ValueError as exc:
                QMessageBox.warning(self, "Export error", str(exc))
            except OSError:
                QMessageBox.critical(self, "Export error", "Unable to write the CSV file.")
            return
        analysis_inputs = self._last_analysis_inputs
        analysis_result = self._last_analysis_result
        if analysis_inputs is None or analysis_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export analysis CSV",
            "analysis.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            stress = self._last_analysis_stress or self._build_stress_from_analysis_result(analysis_result)
            section_top_m3: float | None = None
            section_bottom_m3: float | None = None
            if self._last_analysis_config is not None:
                section_top_m3, section_bottom_m3 = self._resolve_stress_section_moduli(
                    section_modulus_m3=self._last_analysis_config.section_modulus_m3,
                    section_modulus_head_m3=self._last_analysis_config.section_modulus_head_m3,
                    section_modulus_foot_m3=self._last_analysis_config.section_modulus_foot_m3,
                )
            write_analysis_csv_from_result(
                path,
                analysis_inputs,
                analysis_result,
                stress_results=stress,
                section_modulus_top_m3=section_top_m3,
                section_modulus_bottom_m3=section_bottom_m3,
            )
            solver_mode = (
                self._last_analysis_mode.value
                if self._last_analysis_mode is not None
                else AnalysisMode.CLOSED_FORM.value
            )
            metadata_payload = {
                "analysis_inputs": asdict(analysis_inputs),
                "analysis_config": asdict(self._last_analysis_config) if self._last_analysis_config else None,
                "load_source": self._last_static_load_source_metadata(),
            }
            metadata_payload.update(self._stress_metadata_payload(stress))
            write_export_metadata(
                path,
                solver_mode=solver_mode,
                inputs_payload=metadata_payload,
                units="SI",
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Export error", str(exc))
        except OSError:
            QMessageBox.critical(self, "Export error", "Unable to write the CSV file.")

    def _export_transition_metrics_csv(self) -> None:
        if not self._require_transition_result():
            return
        result = self._last_transition_result
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export transition metrics CSV",
            "transition_metrics.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            write_transition_metrics_csv(path, result)
            metadata_payload = {
                "transition_context": asdict(self._last_transition_context) if self._last_transition_context else None,
                "load_source": self._last_static_load_source_metadata(),
                "transition_result_mode": _enum_value(result.mode),
                "transition_metrics_schema_version": result.transition_metrics_schema_version,
                "k_units": "MN/m^2",
                "internal_k_units": result.k_units,
                "k_representation": result.k_representation,
                "foundation_reaction_law": result.foundation_reaction_law,
                "energy_method": result.energy_metrics.energy_method if result.energy_metrics else None,
                "energy_equations": result.energy_metrics.energy_equations if result.energy_metrics else None,
                "energy_scope": result.energy_metrics.energy_scope if result.energy_metrics else None,
            }
            solver_mode = (
                self._last_transition_context.analysis_mode.value
                if self._last_transition_context is not None
                else "transition"
            )
            write_export_metadata(
                path,
                solver_mode=solver_mode,
                inputs_payload=metadata_payload,
                units="SI",
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Export error", str(exc))
        except OSError:
            QMessageBox.critical(self, "Export error", "Unable to write the CSV file.")

    def _export_transition_series_csv(self) -> None:
        if not self._require_transition_result():
            return
        result = self._last_transition_result
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export transition series CSV",
            "transition_series.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            write_transition_series_csv(path, result)
            metadata_payload = {
                "transition_context": asdict(self._last_transition_context) if self._last_transition_context else None,
                "load_source": self._last_static_load_source_metadata(),
                "transition_result_mode": _enum_value(result.mode),
                "transition_series_schema_version": result.transition_metrics_schema_version,
                "k_units": "MN/m^2",
                "internal_k_units": result.k_units,
                "k_representation": result.k_representation,
                "foundation_reaction_law": result.foundation_reaction_law,
                "energy_method": result.energy_metrics.energy_method if result.energy_metrics else None,
                "energy_equations": result.energy_metrics.energy_equations if result.energy_metrics else None,
                "energy_scope": result.energy_metrics.energy_scope if result.energy_metrics else None,
            }
            solver_mode = (
                self._last_transition_context.analysis_mode.value
                if self._last_transition_context is not None
                else "transition"
            )
            write_export_metadata(
                path,
                solver_mode=solver_mode,
                inputs_payload=metadata_payload,
                units="SI",
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Export error", str(exc))
        except OSError:
            QMessageBox.critical(self, "Export error", "Unable to write the CSV file.")

    def _export_transition_config(self) -> None:
        if not self._require_transition_result():
            return
        context = self._last_transition_context
        if context is None:
            QMessageBox.warning(
                self,
                "No transition config",
                "Run a transition analysis before exporting the run configuration.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export transition run config",
            "transition_run.json",
            "JSON Files (*.json)",
        )
        if not path:
            return
        payload = {
            "transition_context": asdict(context),
            "transition_result": asdict(self._last_transition_result) if self._last_transition_result else None,
            "metadata": {
                "units": "SI",
                "solver_mode": context.analysis_mode.value,
                "load_source": self._last_static_load_source_metadata(),
                "inputs_hash": compute_inputs_hash(asdict(context)),
                "transition_metrics_schema_version": (
                    self._last_transition_result.transition_metrics_schema_version
                    if self._last_transition_result is not None
                    else 2
                ),
                "transition_series_schema_version": (
                    self._last_transition_result.transition_metrics_schema_version
                    if self._last_transition_result is not None
                    else 2
                ),
                "k_units": "MN/m^2",
                "internal_k_units": self._last_transition_result.k_units if self._last_transition_result else "N/m^2",
                "k_representation": (
                    self._last_transition_result.k_representation
                    if self._last_transition_result
                    else "continuous_per_unit_length"
                ),
                "foundation_reaction_law": (
                    self._last_transition_result.foundation_reaction_law
                    if self._last_transition_result
                    else "q_f(x)=k(x)w(x) [N/m]"
                ),
                "energy_method": (
                    self._last_transition_result.energy_metrics.energy_method
                    if self._last_transition_result and self._last_transition_result.energy_metrics
                    else None
                ),
                "energy_equations": (
                    self._last_transition_result.energy_metrics.energy_equations
                    if self._last_transition_result and self._last_transition_result.energy_metrics
                    else None
                ),
                "energy_scope": (
                    self._last_transition_result.energy_metrics.energy_scope
                    if self._last_transition_result and self._last_transition_result.energy_metrics
                    else None
                ),
            },
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

    def _export_analysis_config(self) -> None:
        if not self._require_analysis_config():
            return
        if self._last_envelope_result is not None:
            envelope_config = self._last_envelope_config
            if envelope_config is None:
                return
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export envelope config",
                "envelope_config.json",
                "JSON Files (*.json)",
            )
            if not path:
                return
            payload = {
                "envelope_config": asdict(envelope_config),
                "metadata": {
                    "units": "SI",
                    "solver_mode": envelope_config.mode.value,
                    "load_source": self._last_static_load_source_metadata(),
                    "inputs_hash": compute_inputs_hash(asdict(envelope_config)),
                },
            }
            payload["metadata"] = self._extend_envelope_metadata_payload(
                payload["metadata"],
                result=self._last_envelope_result,
                load_source=payload["metadata"]["load_source"],
            )
            Path(path).write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
            return
        analysis_inputs = self._last_analysis_inputs
        analysis_config = self._last_analysis_config
        if analysis_inputs is None or analysis_config is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export analysis config",
            "analysis_config.json",
            "JSON Files (*.json)",
        )
        if not path:
            return
        payload = {
            "analysis_inputs": asdict(analysis_inputs),
            "analysis_config": asdict(analysis_config),
            "metadata": {
                "units": "SI",
                "solver_mode": (
                    self._last_analysis_mode.value
                    if self._last_analysis_mode is not None
                    else AnalysisMode.CLOSED_FORM.value
                ),
                "load_source": self._last_static_load_source_metadata(),
                "inputs_hash": compute_inputs_hash(
                    {
                        "analysis_inputs": asdict(analysis_inputs),
                        "analysis_config": asdict(analysis_config),
                    }
                ),
            },
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

    def _export_sleeper_csv(self) -> None:
        if not self._require_analysis_result():
            return
        if self._last_envelope_result is not None:
            result = self._last_envelope_result
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export envelope sleeper CSV",
                "envelope_sleeper_loads.csv",
                "CSV Files (*.csv)",
            )
            if not path:
                return
            try:
                stress = self._last_envelope_stress or self._build_stress_from_envelope_result(result)
                geometry = (
                    self._resolve_envelope_bearing_geometry(self._last_envelope_config)
                    if self._last_envelope_config is not None
                    else None
                )
                write_envelope_sleeper_csv(
                    path,
                    result,
                    stress_results=stress,
                    bearing_geometry=geometry,
                    ballast_thickness_m=self._ballast_thickness_m(),
                )
                envelope_mode = (
                    self._last_envelope_config.mode.value
                    if self._last_envelope_config is not None
                    else AnalysisMode.CLOSED_FORM.value
                )
                metadata_payload = {
                    "envelope_config": asdict(self._last_envelope_config) if self._last_envelope_config else None,
                    "load_source": self._last_static_load_source_metadata(),
                }
                metadata_payload = self._extend_envelope_metadata_payload(
                    metadata_payload,
                    result=result,
                    load_source=metadata_payload["load_source"],
                )
                metadata_payload.update(self._stress_metadata_payload(stress))
                write_export_metadata(path, solver_mode=envelope_mode, inputs_payload=metadata_payload, units="SI")
            except ValueError as exc:
                QMessageBox.warning(self, "Export error", str(exc))
            except OSError:
                QMessageBox.critical(self, "Export error", "Unable to write the CSV file.")
            return
        analysis_inputs = self._last_analysis_inputs
        analysis_result = self._last_analysis_result
        if analysis_inputs is None or analysis_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export sleeper CSV",
            "sleeper_loads.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            stress = self._last_analysis_stress or self._build_stress_from_analysis_result(analysis_result)
            write_sleeper_csv_from_result(
                path,
                analysis_inputs,
                analysis_result,
                stress_results=stress,
                ballast_thickness_m=self._ballast_thickness_m(),
            )
            solver_mode = (
                self._last_analysis_mode.value
                if self._last_analysis_mode is not None
                else AnalysisMode.CLOSED_FORM.value
            )
            metadata_payload = {
                "analysis_inputs": asdict(analysis_inputs),
                "analysis_config": asdict(self._last_analysis_config) if self._last_analysis_config else None,
                "load_source": self._last_static_load_source_metadata(),
            }
            metadata_payload.update(self._stress_metadata_payload(stress))
            write_export_metadata(
                path,
                solver_mode=solver_mode,
                inputs_payload=metadata_payload,
                units="SI",
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Export error", str(exc))
        except OSError:
            QMessageBox.critical(self, "Export error", "Unable to write the CSV file.")

    def _export_dynamic_time_csv(self) -> None:
        if not self._require_dynamic_result():
            return
        result = self._last_dynamic_result
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export dynamic time history CSV",
            "dynamic_time_history.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            write_dynamic_time_history_csv(path, result, self.probe_selection_combo.currentIndex())
            metadata_payload = {
                "dynamic_config": asdict(self._last_dynamic_config) if self._last_dynamic_config else None,
                "dynamic_mode": self._last_dynamic_mode.value if self._last_dynamic_mode else DynamicMode.STEADY_STATE.value,
                "load_source": self._last_dynamic_load_source_metadata(),
            }
            solver_mode = (
                self._last_dynamic_mode.value if self._last_dynamic_mode is not None else DynamicMode.STEADY_STATE.value
            )
            write_export_metadata(
                path,
                solver_mode=solver_mode,
                inputs_payload=metadata_payload,
                units="SI",
                parameter_trace=result.parameter_trace,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Export error", str(exc))
        except OSError:
            QMessageBox.critical(self, "Export error", "Unable to write the CSV file.")

    def _export_dynamic_fft_csv(self) -> None:
        if not self._require_dynamic_result():
            return
        result = self._last_dynamic_result
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export dynamic FFT CSV",
            "dynamic_fft.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            write_dynamic_fft_csv(path, result, self.probe_selection_combo.currentIndex())
            metadata_payload = {
                "dynamic_config": asdict(self._last_dynamic_config) if self._last_dynamic_config else None,
                "dynamic_mode": self._last_dynamic_mode.value if self._last_dynamic_mode else DynamicMode.STEADY_STATE.value,
                "load_source": self._last_dynamic_load_source_metadata(),
            }
            solver_mode = (
                self._last_dynamic_mode.value if self._last_dynamic_mode is not None else DynamicMode.STEADY_STATE.value
            )
            write_export_metadata(
                path,
                solver_mode=solver_mode,
                inputs_payload=metadata_payload,
                units="SI",
                parameter_trace=result.parameter_trace,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Export error", str(exc))
        except OSError:
            QMessageBox.critical(self, "Export error", "Unable to write the CSV file.")

    def _export_dynamic_psd_csv(self) -> None:
        if not self._require_dynamic_result():
            return
        result = self._last_dynamic_result
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export dynamic PSD CSV",
            "dynamic_psd.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            write_dynamic_psd_csv(path, result, self.probe_selection_combo.currentIndex())
            metadata_payload = {
                "dynamic_config": asdict(self._last_dynamic_config) if self._last_dynamic_config else None,
                "dynamic_mode": self._last_dynamic_mode.value if self._last_dynamic_mode else DynamicMode.STEADY_STATE.value,
                "load_source": self._last_dynamic_load_source_metadata(),
            }
            solver_mode = (
                self._last_dynamic_mode.value if self._last_dynamic_mode is not None else DynamicMode.STEADY_STATE.value
            )
            write_export_metadata(
                path,
                solver_mode=solver_mode,
                inputs_payload=metadata_payload,
                units="SI",
                parameter_trace=result.parameter_trace,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Export error", str(exc))
        except OSError:
            QMessageBox.critical(self, "Export error", "Unable to write the CSV file.")

    def _export_dynamic_transition_metrics_csv(self) -> None:
        if not self._require_dynamic_transition_result():
            return
        result = self._last_dynamic_transition_result
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export dynamic transition metrics CSV",
            "dynamic_transition_metrics.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            write_dynamic_transition_metrics_csv(path, result)
            metadata_payload = {
                "dynamic_transition_config": (
                    asdict(self._last_dynamic_transition_config)
                    if self._last_dynamic_transition_config is not None
                    else None
                ),
                "dynamic_mode": DynamicMode.TRANSITION.value,
                "load_source": self._last_dynamic_load_source_metadata(),
            }
            write_export_metadata(
                path,
                solver_mode=DynamicMode.TRANSITION.value,
                inputs_payload=metadata_payload,
                units="SI",
                parameter_trace=result.representative.parameter_trace,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Export error", str(exc))
        except OSError:
            QMessageBox.critical(self, "Export error", "Unable to write the CSV file.")

    def _export_dynamic_transition_series_csv(self) -> None:
        if not self._require_dynamic_transition_result():
            return
        result = self._last_dynamic_transition_result
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export dynamic transition series CSV",
            "dynamic_transition_series.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            write_dynamic_transition_series_csv(path, result)
            metadata_payload = {
                "dynamic_transition_config": (
                    asdict(self._last_dynamic_transition_config)
                    if self._last_dynamic_transition_config is not None
                    else None
                ),
                "dynamic_mode": DynamicMode.TRANSITION.value,
                "load_source": self._last_dynamic_load_source_metadata(),
            }
            write_export_metadata(
                path,
                solver_mode=DynamicMode.TRANSITION.value,
                inputs_payload=metadata_payload,
                units="SI",
                parameter_trace=result.representative.parameter_trace,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Export error", str(exc))
        except OSError:
            QMessageBox.critical(self, "Export error", "Unable to write the CSV file.")

    def _export_dynamic_transition_config(self) -> None:
        if self._last_dynamic_transition_config is None:
            QMessageBox.warning(
                self,
                "No dynamic transition config",
                "Run a dynamic transition analysis before exporting the run configuration.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export dynamic transition run config",
            "dynamic_transition_run.json",
            "JSON Files (*.json)",
        )
        if not path:
            return
        config = self._last_dynamic_transition_config
        payload = {
            "dynamic_transition_config": asdict(config),
            "metadata": {
                "units": "SI",
                "solver_mode": DynamicMode.TRANSITION.value,
                "load_source": self._last_dynamic_load_source_metadata(),
                "inputs_hash": compute_inputs_hash(asdict(config)),
            },
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

    def _export_dipped_joint_csv(self) -> None:
        """Export dipped joint results to CSV."""
        if not self._require_dipped_joint_result():
            return
        result = self._last_dipped_joint_result
        if result is None:
            return
        config = self._last_dipped_joint_config
        if config is None:
            config = DippedJointConfig(
                static_wheel_load_n=kn_to_n(self.load_magnitude_input.value()),
                total_dip_angle_rad=self.dip_angle_input.value() * 1.0e-3,
                speed_m_per_s=self.speed_input.value(),
                hertzian_stiffness_n_per_m=self.hertzian_stiffness_input.value() * 1.0e6,
                track_mass_p1_kg=self.track_mass_p1_input.value(),
                unsprung_mass_kg=self.unsprung_mass_input.value(),
                track_mass_p2_kg=self.track_mass_p2_input.value(),
                track_stiffness_p2_n_per_m=self.track_stiffness_p2_input.value() * 1.0e6,
                track_damping_p2_n_s_per_m=self.track_damping_p2_input.value() * 1.0e3,
            )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export dipped joint CSV",
            "dipped_joint.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            write_dipped_joint_csv(path, config, result)
            metadata_payload = {
                "dipped_joint_config": asdict(config),
                "dynamic_mode": DynamicMode.DIPPED_JOINT.value,
            }
            write_export_metadata(
                path,
                solver_mode=DynamicMode.DIPPED_JOINT.value,
                inputs_payload=metadata_payload,
                units="SI",
            )
            self.statusBar().showMessage(f"Exported to {path}")
        except OSError:
            QMessageBox.critical(self, "Export error", "Unable to write the CSV file.")


def _app_log_path() -> str:
    return str(Path.home() / APP_LOG_RELATIVE_PATH)


def _configure_logging() -> None:
    log_path = Path(_app_log_path())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
            return
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root_logger.addHandler(file_handler)
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    LOGGER.info("Application logging configured: %s", log_path)


def run() -> int:
    """Start the Qt application and return the exit code."""
    _configure_logging()
    _configure_qt_plugin_path()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


def _configure_qt_plugin_path() -> None:
    """Ensure Qt can find its platform plugins on macOS venv installs."""
    if os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        return
    plugin_path = QLibraryInfo.path(QLibraryInfo.PluginsPath)
    if not plugin_path:
        return
    plugin_dir = Path(plugin_path)
    platforms_dir = plugin_dir / "platforms"
    if platforms_dir.exists():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platforms_dir))
    os.environ.setdefault("QT_PLUGIN_PATH", str(plugin_dir))


if __name__ == "__main__":
    raise SystemExit(run())

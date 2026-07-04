"""Custom multi-axis chart builder for BOEF plot panels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ChartAxisFamily(str, Enum):
    STATIC_SPATIAL = "static_spatial"
    TRANSITION_SPATIAL = "transition_spatial"
    SLEEPER_SPATIAL = "sleeper_spatial"
    DYNAMIC_XI = "dynamic_xi"
    TIME = "time"
    FREQUENCY = "frequency"


MAX_CUSTOM_CHART_SERIES = 4
CUSTOM_CHART_COLOR_CYCLE = (
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
)


def custom_chart_color(index: int) -> str:
    return CUSTOM_CHART_COLOR_CYCLE[index % len(CUSTOM_CHART_COLOR_CYCLE)]


def axis_label_with_unit(label: str, unit: str) -> str:
    """Return a display label with its unit restored after cache splitting."""
    clean_label = label.strip()
    clean_unit = unit.strip()
    if clean_label and clean_unit:
        return f"{clean_label} ({clean_unit})"
    return clean_label or clean_unit


def _axis_families_compatible(
    left: ChartAxisFamily,
    right: ChartAxisFamily,
) -> bool:
    if left == right:
        return True
    spatial_families = {
        ChartAxisFamily.STATIC_SPATIAL,
        ChartAxisFamily.TRANSITION_SPATIAL,
    }
    return left in spatial_families and right in spatial_families


@dataclass(frozen=True)
class RenderedSeries:
    series_id: str
    source_chart_id: str
    label: str
    x: list[float]
    y: list[float]
    x_label: str
    y_label: str
    x_unit: str
    y_unit: str
    axis_family: ChartAxisFamily
    color_hint: str | None = None
    linestyle_hint: str | None = None


@dataclass(frozen=True)
class CustomChartSelection:
    series_id: str
    axis_target: str
    legend_label: str


@dataclass(frozen=True)
class ResampledSeries:
    selection: CustomChartSelection
    series: RenderedSeries
    y_values: list[float]


def _same_grid(x1: Sequence[float], x2: Sequence[float]) -> bool:
    if len(x1) != len(x2):
        return False
    return bool(np.allclose(np.asarray(x1, dtype=float), np.asarray(x2, dtype=float), rtol=1.0e-6, atol=1.0e-9))


def build_resampled_series(
    *,
    source_series: Sequence[RenderedSeries],
    selections: Sequence[CustomChartSelection],
) -> tuple[list[float], str, ChartAxisFamily, list[ResampledSeries]]:
    if not selections:
        raise ValueError("Select at least one source series.")
    if len(selections) > MAX_CUSTOM_CHART_SERIES:
        raise ValueError(f"Select at most {MAX_CUSTOM_CHART_SERIES} series.")

    by_id = {item.series_id: item for item in source_series}
    selected_series: list[RenderedSeries] = []
    for selection in selections:
        series = by_id.get(selection.series_id)
        if series is None:
            raise ValueError(f"Selected source series '{selection.series_id}' is not available.")
        selected_series.append(series)

    family = selected_series[0].axis_family
    if any(not _axis_families_compatible(series.axis_family, family) for series in selected_series):
        raise ValueError("Selected series must use the same axis family (x-domain type).")

    primary_x = list(selected_series[0].x)
    if not primary_x:
        raise ValueError("Selected source series is empty.")
    x_grid = np.asarray(primary_x, dtype=float)

    resampled: list[ResampledSeries] = []
    for selection, series in zip(selections, selected_series):
        if not series.x or not series.y:
            raise ValueError(f"Series '{series.label}' has no data.")
        if len(series.x) != len(series.y):
            raise ValueError(f"Series '{series.label}' has inconsistent x/y lengths.")
        if _same_grid(primary_x, series.x):
            y_values = list(series.y)
        else:
            x_values = np.asarray(series.x, dtype=float)
            y_values_np = np.asarray(series.y, dtype=float)
            order = np.argsort(x_values)
            x_sorted = x_values[order]
            y_sorted = y_values_np[order]
            y_values = list(np.interp(x_grid, x_sorted, y_sorted))
        resampled.append(ResampledSeries(selection=selection, series=series, y_values=y_values))

    return primary_x, selected_series[0].x_label, family, resampled


class CustomChartDialog(QDialog):
    """Builder dialog to compose up to four y-axes from compatible source series."""

    AXES_TARGETS = ("L1", "L2", "R1", "R2")

    def __init__(self, *, source_series: Sequence[RenderedSeries], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Custom chart")
        self.resize(980, 680)
        self._source_series = list(source_series)

        root = QVBoxLayout(self)

        helper = QLabel(
            "Select compatible series and assign each to an axis target. "
            "Only same x-domain families can be mixed in one custom chart (up to 4 series)."
        )
        helper.setWordWrap(True)
        root.addWidget(helper)

        self.series_table = QTableWidget(len(self._source_series), 4)
        self.series_table.setHorizontalHeaderLabels(["Use", "Series", "Axis", "Legend label"])
        self.series_table.verticalHeader().setVisible(False)
        self.series_table.setAlternatingRowColors(True)
        self.series_table.setSelectionMode(QTableWidget.NoSelection)
        self.series_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.series_table.setColumnWidth(0, 60)
        self.series_table.setColumnWidth(1, 360)
        self.series_table.setColumnWidth(2, 120)

        for row, series in enumerate(self._source_series):
            use_item = QTableWidgetItem()
            use_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            use_item.setCheckState(Qt.Checked if row == 0 else Qt.Unchecked)
            self.series_table.setItem(row, 0, use_item)

            descriptor = f"[{series.source_chart_id}] {series.label} ({series.x_label} / {series.y_label})"
            series_item = QTableWidgetItem(descriptor)
            series_item.setToolTip(descriptor)
            self.series_table.setItem(row, 1, series_item)

            axis_combo = QComboBox()
            axis_combo.addItems(self.AXES_TARGETS)
            axis_combo.setCurrentText("L1" if row == 0 else "R1")
            axis_combo.currentIndexChanged.connect(self._refresh_preview)
            self.series_table.setCellWidget(row, 2, axis_combo)

            legend_input = QLineEdit(series.label)
            legend_input.textChanged.connect(self._refresh_preview)
            self.series_table.setCellWidget(row, 3, legend_input)

        self.series_table.itemChanged.connect(self._handle_series_table_change)
        root.addWidget(self.series_table)

        preview_container = QWidget()
        preview_layout = QHBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(QLabel("Preview:"))
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._refresh_preview)
        preview_layout.addWidget(refresh_button)
        preview_layout.addStretch(1)
        root.addWidget(preview_container)

        self.preview_figure = Figure(figsize=(8, 4), dpi=100, tight_layout=True)
        self.preview_canvas = FigureCanvas(self.preview_figure)
        root.addWidget(self.preview_canvas, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_with_validation)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._sync_series_row_visibility()
        self._refresh_preview()

    def selected(self) -> list[CustomChartSelection]:
        selections: list[CustomChartSelection] = []
        for row, series in enumerate(self._source_series):
            use_item = self.series_table.item(row, 0)
            if use_item is None or use_item.checkState() != Qt.Checked:
                continue
            axis_widget = self.series_table.cellWidget(row, 2)
            label_widget = self.series_table.cellWidget(row, 3)
            if not isinstance(axis_widget, QComboBox) or not isinstance(label_widget, QLineEdit):
                continue
            legend = label_widget.text().strip() or series.label
            selections.append(
                CustomChartSelection(
                    series_id=series.series_id,
                    axis_target=axis_widget.currentText(),
                    legend_label=legend,
                )
            )
        return selections

    def _accept_with_validation(self) -> None:
        try:
            self._validate_selection()
        except ValueError as exc:
            QMessageBox.warning(self, "Custom chart", str(exc))
            return
        self.accept()

    def _validate_selection(self) -> None:
        selections = self.selected()
        if not selections:
            raise ValueError("Select at least one source series.")
        build_resampled_series(source_series=self._source_series, selections=selections)

    def _handle_series_table_change(self, *_: object) -> None:
        self._sync_series_row_visibility()
        self._refresh_preview()

    def _selected_axis_family(self) -> ChartAxisFamily | None:
        for row, series in enumerate(self._source_series):
            use_item = self.series_table.item(row, 0)
            if use_item is None or use_item.checkState() != Qt.Checked:
                continue
            return series.axis_family
        return None

    def _sync_series_row_visibility(self) -> None:
        selected_family = self._selected_axis_family()
        self.series_table.blockSignals(True)
        try:
            for row, series in enumerate(self._source_series):
                compatible = (
                    selected_family is None
                    or _axis_families_compatible(series.axis_family, selected_family)
                )
                use_item = self.series_table.item(row, 0)
                if use_item is not None and not compatible and use_item.checkState() == Qt.Checked:
                    use_item.setCheckState(Qt.Unchecked)
                self.series_table.setRowHidden(row, not compatible)
        finally:
            self.series_table.blockSignals(False)

    def _refresh_preview(self, *_: object) -> None:
        self.preview_figure.clear()
        axes = self.preview_figure.add_subplot(111)
        try:
            selections = self.selected()
            if not selections:
                axes.set_title("Select one or more series")
                axes.grid(True, linestyle="--", alpha=0.3)
                self.preview_canvas.draw_idle()
                return

            x_values, _x_label, _family, resampled = build_resampled_series(
                source_series=self._source_series,
                selections=selections,
            )
            axis_map: dict[str, object] = {"L1": axes}
            first_series_for_axis: dict[str, RenderedSeries] = {}
            plotted_lines = []

            def ensure_axis(target: str):
                if target in axis_map:
                    return axis_map[target]
                if target == "R1":
                    axis_map[target] = axes.twinx()
                elif target == "R2":
                    ax = axes.twinx()
                    ax.spines["right"].set_position(("outward", 55))
                    axis_map[target] = ax
                elif target == "L2":
                    ax = axes.twinx()
                    ax.spines["left"].set_position(("outward", 55))
                    ax.spines["left"].set_visible(True)
                    ax.spines["right"].set_visible(False)
                    ax.yaxis.set_label_position("left")
                    ax.yaxis.tick_left()
                    axis_map[target] = ax
                else:
                    axis_map[target] = axes
                return axis_map[target]

            for index, item in enumerate(resampled):
                axis = ensure_axis(item.selection.axis_target)
                style: dict[str, object] = {"color": custom_chart_color(index)}
                if item.series.linestyle_hint:
                    style["linestyle"] = item.series.linestyle_hint
                line = axis.plot(x_values, item.y_values, label=item.selection.legend_label, linewidth=1.8, **style)[0]
                plotted_lines.append(line)
                first_series_for_axis.setdefault(item.selection.axis_target, item.series)

            x_axis_label = axis_label_with_unit(resampled[0].series.x_label, resampled[0].series.x_unit)
            axes.set_xlabel(x_axis_label)
            axes.grid(True, linestyle="--", alpha=0.35)
            axes.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.45)
            for target, series in first_series_for_axis.items():
                axis = ensure_axis(target)
                axis.set_ylabel(axis_label_with_unit(series.y_label, series.y_unit))
                if target != "L1":
                    color = series.color_hint or "#222222"
                    axis.tick_params(axis="y", colors=color)
                    axis.yaxis.label.set_color(color)
            if plotted_lines:
                axes.legend(plotted_lines, [line.get_label() for line in plotted_lines], loc="best", fontsize=8)
        except ValueError as exc:
            axes.text(0.02, 0.98, str(exc), transform=axes.transAxes, va="top", ha="left", fontsize=9)
        self.preview_canvas.draw_idle()

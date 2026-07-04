"""Design alternative comparison dialog."""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from app.export_helpers import write_export_metadata
from core.units import n_to_kn, pa_to_kpa, pa_to_mpa
from db.models import DesignAlternative, Project


COLOR_BEST = QColor("#d8f3dc")
COLOR_OK = QColor("#eaf7ed")
COLOR_WORSE = QColor("#fde2e2")
COLOR_WARNING = QColor("#fff3bf")
COLOR_DRAFT = QColor("#f1f3f5")
COLOR_TEXT = QColor("#1f2933")
CHART_GREEN = "#2f855a"
CHART_RED = "#c2410c"
CHART_BLUE = "#2b6cb0"
CHART_AMBER = "#b7791f"
CHART_GREY = "#718096"
CHART_DARK = "#1a202c"


def _json_payload(text: str) -> dict[str, object]:
    value = json.loads(text)
    return value if isinstance(value, dict) else {}


def _metric(alternative: DesignAlternative, key: str) -> float | None:
    value = _json_payload(alternative.metrics_json).get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _payload_value(alternative: DesignAlternative, key: str) -> object:
    return _json_payload(alternative.metrics_json).get(key)


def _decision_status(alternative: DesignAlternative) -> str:
    value = _payload_value(alternative, "decision_status")
    if isinstance(value, str) and value:
        return value
    if alternative.status == "ok":
        return "pass"
    return alternative.status


def _change_summary(alternative: DesignAlternative) -> str:
    payload = _json_payload(alternative.changed_parameters_json)
    if not payload:
        return "-"
    parts = []
    for key, value in payload.items():
        parts.append(f"{key}: {value}")
    return "; ".join(parts)


def write_alternatives_comparison_csv(
    path: str | Path,
    alternatives: Sequence[DesignAlternative],
) -> None:
    export_path = Path(path)
    with export_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "alternative_name",
                "change_summary",
                "max_deflection_m",
                "max_moment_or_stress",
                "max_sleeper_load_or_reaction",
                "ballast_pressure_pa",
                "formation_pressure_pa",
                "subgrade_pressure_pa",
                "deep_subgrade_pressure_pa",
                "transition_metric_m",
                "score",
                "max_utilization",
                "governing_criterion",
                "decision_status",
                "status",
                "notes",
            ]
        )
        for alternative in alternatives:
            writer.writerow(
                [
                    alternative.name,
                    _change_summary(alternative),
                    _format_optional(_metric(alternative, "max_deflection_m")),
                    _format_optional(
                        _metric(alternative, "rail_stress_pa")
                        or _metric(alternative, "max_moment_nm")
                    ),
                    _format_optional(
                        _metric(alternative, "max_sleeper_load_n")
                        or _metric(alternative, "max_reaction_n_per_m")
                    ),
                    _format_optional(_metric(alternative, "ballast_pressure_pa")),
                    _format_optional(_metric(alternative, "formation_pressure_pa")),
                    _format_optional(_metric(alternative, "subgrade_pressure_pa")),
                    _format_optional(_metric(alternative, "deep_subgrade_pressure_pa")),
                    _format_optional(_metric(alternative, "transition_metric_m")),
                    _format_optional(alternative.score),
                    _format_optional(_metric(alternative, "max_utilization")),
                    str(_payload_value(alternative, "governing_criterion") or ""),
                    _decision_status(alternative),
                    alternative.status,
                    alternative.description or "",
                ]
            )


class AlternativeComparisonDialog(QDialog):
    """Compare persisted design alternatives for one project."""

    def __init__(
        self,
        *,
        project: Project,
        alternatives: Sequence[DesignAlternative],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Design Alternatives Comparison")
        self.resize(1040, 700)
        self.project = project
        self.alternatives = list(alternatives)

        self.summary_label = QLabel(
            f"Project: {project.name}. Select two or more alternatives to compare."
        )
        self.table = QTableWidget()
        self.export_button = QPushButton("Export alternatives comparison CSV")
        self.close_button = QPushButton("Close")
        self.plot_tabs = QTabWidget()

        self._build_ui()
        self._populate_table()
        self._update_selection_state()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        self.table.setColumnCount(15)
        self.table.setHorizontalHeaderLabels(
            [
                "Alternative",
                "Change summary",
                "Deflection (mm)",
                "Moment/stress",
                "Sleeper/reaction",
                "Ballast (kPa)",
                "Formation (kPa)",
                "Subgrade (kPa)",
                "Deep subgrade (kPa)",
                "Transition (mm)",
                "Score",
                "Utilization",
                "Governing",
                "Decision",
                "Status",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._update_selection_state)
        layout.addWidget(self.table, stretch=1)

        self.plot_tabs.addTab(self._plot_widget(), "Metrics")
        self.plot_tabs.addTab(self._plot_widget(), "Scores")
        self.plot_tabs.addTab(self._plot_widget(), "Status")
        layout.addWidget(self.plot_tabs, stretch=1)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)
        self.export_button.clicked.connect(self._export)
        self.close_button.clicked.connect(self.accept)

    def _plot_widget(self) -> FigureCanvas:
        return FigureCanvas(Figure(figsize=(6, 3), tight_layout=True))

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self.alternatives))
        scored = [
            item
            for item in self.alternatives
            if item.score is not None and _decision_status(item) not in {"fail", "draft"}
        ]
        best_id = min(scored, key=_alternative_ranking_key).id if scored else None
        baseline_score = 1.0
        for row, alternative in enumerate(self.alternatives):
            values = [
                alternative.name,
                _change_summary(alternative),
                _format_scaled(_metric(alternative, "max_deflection_m"), 1000.0),
                _format_moment_or_stress(alternative),
                _format_sleeper_or_reaction(alternative),
                _format_pressure(alternative, "ballast_pressure_pa"),
                _format_pressure(alternative, "formation_pressure_pa"),
                _format_pressure(alternative, "subgrade_pressure_pa"),
                _format_pressure(alternative, "deep_subgrade_pressure_pa"),
                _format_scaled(_metric(alternative, "transition_metric_m"), 1000.0),
                _format_optional(alternative.score),
                _format_optional(_metric(alternative, "max_utilization")),
                str(_payload_value(alternative, "governing_criterion") or "-"),
                _decision_status(alternative),
                alternative.status,
            ]
            row_color = _alternative_row_color(alternative, best_id=best_id, baseline_score=baseline_score)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14}:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item.setBackground(row_color)
                item.setForeground(COLOR_TEXT)
                if alternative.status in {"warning", "fail"}:
                    item.setToolTip(alternative.description or alternative.status)
                elif alternative.id == best_id:
                    item.setToolTip("Best valid alternative by stored score.")
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()

    def selected_alternatives(self) -> list[DesignAlternative]:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        return [self.alternatives[row] for row in rows]

    def _update_selection_state(self) -> None:
        selected = self.selected_alternatives()
        enabled = len(selected) >= 2
        self.export_button.setEnabled(enabled)
        if enabled:
            passing = [item for item in selected if _decision_status(item) in {"pass", "ok"}]
            best = min(passing or selected, key=_alternative_ranking_key)
            worst = max(selected, key=_alternative_worst_key)
            best_label = "Best passing" if passing else "Lowest utilization"
            self.summary_label.setText(
                f"Project: {self.project.name}. {best_label}: {best.name}. Worst: {worst.name}."
            )
        else:
            self.summary_label.setText(
                f"Project: {self.project.name}. Select two or more alternatives to compare."
            )
        self._draw_plots(selected if enabled else [])

    def _draw_plots(self, alternatives: Sequence[DesignAlternative]) -> None:
        self._draw_metric_plot(alternatives)
        self._draw_score_plot(alternatives)
        self._draw_status_plot(alternatives)

    def _draw_metric_plot(self, alternatives: Sequence[DesignAlternative]) -> None:
        canvas = self.plot_tabs.widget(0)
        figure = canvas.figure
        figure.clear()
        axes = figure.add_subplot(111)
        if not alternatives:
            axes.text(0.5, 0.5, "Select alternatives", ha="center", va="center")
            canvas.draw_idle()
            return
        metric_defs = [
            ("Deflection", "max_deflection_m"),
            ("Rail stress", "rail_stress_pa"),
            ("Ballast", "ballast_pressure_pa"),
            ("Formation", "formation_pressure_pa"),
            ("Subgrade", "subgrade_pressure_pa"),
            ("Deep subgrade", "deep_subgrade_pressure_pa"),
            ("Transition", "transition_metric_m"),
        ]
        metric_defs = [
            item
            for item in metric_defs
            if any(_metric(alternative, item[1]) is not None for alternative in alternatives)
        ]
        if not metric_defs:
            axes.text(0.5, 0.5, "No comparable metrics", ha="center", va="center")
            canvas.draw_idle()
            return
        alt_count = len(alternatives)
        width = min(0.8 / max(alt_count, 1), 0.22)
        palette = [CHART_BLUE, CHART_GREEN, CHART_AMBER, CHART_GREY, CHART_RED]
        for alt_index, alternative in enumerate(alternatives):
            values: list[float] = []
            for _label, key in metric_defs:
                candidate = _metric(alternative, key)
                scale = max(
                    abs(_metric(item, key) or 0.0)
                    for item in alternatives
                )
                values.append(abs(candidate) / scale if candidate is not None and scale > 0.0 else 0.0)
            offset = (alt_index - (alt_count - 1) / 2.0) * width
            status = _decision_status(alternative)
            color = (
                CHART_RED
                if status == "fail"
                else CHART_AMBER
                if status == "warning"
                else CHART_GREY
                if status in {"draft", "unrated"}
                else palette[alt_index % len(palette)]
            )
            axes.bar(
                [index + offset for index in range(len(metric_defs))],
                values,
                width=width,
                label=alternative.name,
                color=color,
            )
        axes.set_xticks(range(len(metric_defs)))
        axes.set_xticklabels(_wrapped_labels([label for label, _key in metric_defs], width=12), fontsize=8)
        axes.set_ylabel("Normalized value (lower is better)")
        axes.set_title("Engineering metric comparison")
        axes.legend(fontsize=7)
        _style_axes(axes)
        canvas.draw_idle()

    def _draw_score_plot(self, alternatives: Sequence[DesignAlternative]) -> None:
        canvas = self.plot_tabs.widget(1)
        figure = canvas.figure
        figure.clear()
        axes = figure.add_subplot(111)
        if not alternatives:
            axes.text(0.5, 0.5, "Select alternatives", ha="center", va="center")
            canvas.draw_idle()
            return
        labels = [item.name for item in alternatives]
        scores = [
            _metric(item, "max_utilization")
            if _metric(item, "max_utilization") is not None
            else item.score if item.score is not None else 0.0
            for item in alternatives
        ]
        colors = [
            CHART_RED
            if _decision_status(item) == "fail"
            else CHART_AMBER
            if _decision_status(item) == "warning"
            else CHART_GREY
            if _decision_status(item) in {"draft", "unrated"}
            else CHART_GREEN
            for item in alternatives
        ]
        axes.bar(range(len(scores)), scores, color=colors)
        axes.axhline(1.0, color=CHART_DARK, linewidth=0.8)
        axes.axhline(0.85, color=CHART_AMBER, linewidth=0.8, linestyle="--")
        axes.set_xticks(range(len(labels)))
        axes.set_xticklabels(_wrapped_labels(labels), rotation=35, ha="right", fontsize=8)
        axes.set_ylabel("Utilization / score")
        axes.set_title("Utilization and decision threshold")
        _style_axes(axes)
        canvas.draw_idle()

    def _draw_status_plot(self, alternatives: Sequence[DesignAlternative]) -> None:
        canvas = self.plot_tabs.widget(2)
        figure = canvas.figure
        figure.clear()
        axes = figure.add_subplot(111)
        statuses = {"pass": 0, "warning": 0, "fail": 0, "draft": 0, "unrated": 0}
        for alternative in alternatives:
            status = _decision_status(alternative)
            if status == "ok":
                status = "pass"
            if status not in statuses:
                status = "unrated"
            statuses[status] = statuses.get(status, 0) + 1
        color_map = {
            "pass": CHART_GREEN,
            "warning": CHART_AMBER,
            "fail": CHART_RED,
            "draft": CHART_GREY,
            "unrated": CHART_GREY,
        }
        axes.bar(
            list(statuses),
            list(statuses.values()),
            color=[color_map[key] for key in statuses],
        )
        axes.set_ylabel("Count")
        axes.set_title("Status summary")
        _style_axes(axes)
        canvas.draw_idle()

    def _export(self) -> None:
        selected = self.selected_alternatives()
        if len(selected) < 2:
            QMessageBox.warning(self, "Comparison export", "Select at least two alternatives.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export alternatives comparison CSV",
            "alternatives_comparison.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            write_alternatives_comparison_csv(path, selected)
            write_export_metadata(
                path,
                solver_mode="design_alternatives_comparison",
                inputs_payload={
                    "project_id": self.project.id,
                    "project_name": self.project.name,
                    "selected_alternative_ids": [item.id for item in selected],
                },
                units="SI",
            )
        except OSError:
            QMessageBox.critical(self, "Export error", "Unable to write the CSV file.")


def _format_optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4g}"


def _format_scaled(value: float | None, factor: float) -> str:
    if value is None:
        return "-"
    return f"{value * factor:.4g}"


def _format_moment_or_stress(alternative: DesignAlternative) -> str:
    stress = _metric(alternative, "rail_stress_pa")
    if stress is not None:
        return f"{pa_to_mpa(stress):.4g} MPa"
    moment = _metric(alternative, "max_moment_nm")
    if moment is not None:
        return f"{moment * 0.001:.4g} kN.m"
    return "-"


def _format_sleeper_or_reaction(alternative: DesignAlternative) -> str:
    sleeper = _metric(alternative, "max_sleeper_load_n")
    if sleeper is not None:
        return f"{n_to_kn(sleeper):.4g} kN"
    reaction = _metric(alternative, "max_reaction_n_per_m")
    if reaction is not None:
        return f"{n_to_kn(reaction):.4g} kN/m"
    return "-"


def _format_pressure(alternative: DesignAlternative, key: str) -> str:
    pressure = _metric(alternative, key)
    if pressure is None:
        return "-"
    return f"{pa_to_kpa(pressure):.4g}"


def _alternative_row_color(
    alternative: DesignAlternative,
    *,
    best_id: int | None,
    baseline_score: float,
) -> QColor:
    if alternative.status == "fail" or _decision_status(alternative) == "fail":
        return COLOR_WORSE
    if alternative.status == "warning" or _decision_status(alternative) == "warning":
        return COLOR_WARNING
    if alternative.status == "draft":
        return COLOR_DRAFT
    if best_id is not None and alternative.id == best_id:
        return COLOR_BEST
    if alternative.score is not None and alternative.score > baseline_score:
        return COLOR_WORSE
    return COLOR_OK


def _alternative_ranking_key(alternative: DesignAlternative) -> tuple[int, float, float]:
    status_order = {"pass": 0, "ok": 0, "warning": 1, "fail": 2, "draft": 3, "unrated": 3}
    status = _decision_status(alternative)
    utilization = _metric(alternative, "max_utilization")
    return (
        status_order.get(status, 3),
        utilization if utilization is not None else float("inf"),
        alternative.score if alternative.score is not None else float("inf"),
    )


def _alternative_worst_key(alternative: DesignAlternative) -> tuple[int, float, float]:
    status_order = {"pass": 0, "ok": 0, "warning": 1, "fail": 2, "draft": 3, "unrated": 3}
    status = _decision_status(alternative)
    utilization = _metric(alternative, "max_utilization")
    return (
        status_order.get(status, 3),
        utilization if utilization is not None else -float("inf"),
        alternative.score if alternative.score is not None else -float("inf"),
    )


def _style_axes(axes) -> None:
    axes.grid(True, axis="y", color="#e2e8f0", linewidth=0.8)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_color("#cbd5e0")
        spine.set_linewidth(0.8)


def _wrapped_labels(labels: Sequence[str], width: int = 18) -> list[str]:
    return ["\n".join(textwrap.wrap(label, width=width)) or label for label in labels]

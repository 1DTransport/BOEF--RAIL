from __future__ import annotations

import os

import numpy as np
import pytest

if os.environ.get("BOEF_ENABLE_GUI_TESTS", "").lower() not in {"1", "true", "yes"}:
    pytest.skip("Set BOEF_ENABLE_GUI_TESTS=1 to run PySide GUI tests.", allow_module_level=True)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip(
    "PySide6",
    reason="PySide6 is required for custom chart module imports.",
    exc_type=ImportError,
)
QtCore = pytest.importorskip(
    "PySide6.QtCore",
    reason="PySide6 (QtCore) is required for custom chart dialog tests.",
    exc_type=ImportError,
)
QtWidgets = pytest.importorskip(
    "PySide6.QtWidgets",
    reason="PySide6 (QtWidgets) is required for custom chart dialog tests.",
    exc_type=ImportError,
)
Qt = QtCore.Qt
QApplication = QtWidgets.QApplication

from app.custom_chart import (
    ChartAxisFamily,
    CustomChartDialog,
    CustomChartSelection,
    RenderedSeries,
    axis_label_with_unit,
    build_resampled_series,
)
from app.main import PlotPanel


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _series(
    *,
    series_id: str,
    label: str,
    x: list[float],
    y: list[float],
    family: ChartAxisFamily,
    x_label: str = "x",
    y_label: str = "y",
    x_unit: str = "m",
    y_unit: str = "mm",
) -> RenderedSeries:
    return RenderedSeries(
        series_id=series_id,
        source_chart_id="Test",
        label=label,
        x=x,
        y=y,
        x_label=x_label,
        y_label=y_label,
        x_unit=x_unit,
        y_unit=y_unit,
        axis_family=family,
        color_hint="#1f77b4",
        linestyle_hint="-",
    )


def test_axis_label_with_unit_restores_display_units() -> None:
    assert axis_label_with_unit("w", "mm") == "w (mm)"
    assert axis_label_with_unit("M", "") == "M"


def test_build_resampled_series_rejects_cross_family_mix() -> None:
    source = [
        _series(series_id="a", label="A", x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 0.0], family=ChartAxisFamily.TIME),
        _series(
            series_id="b",
            label="B",
            x=[0.0, 1.0, 2.0],
            y=[0.0, 2.0, 0.0],
            family=ChartAxisFamily.FREQUENCY,
        ),
    ]
    selections = [
        CustomChartSelection(series_id="a", axis_target="L1", legend_label="A"),
        CustomChartSelection(series_id="b", axis_target="R1", legend_label="B"),
    ]

    with pytest.raises(ValueError, match="same axis family"):
        build_resampled_series(source_series=source, selections=selections)


def test_build_resampled_series_interpolates_to_primary_grid() -> None:
    source = [
        _series(
            series_id="a",
            label="A",
            x=[0.0, 1.0, 2.0, 3.0],
            y=[0.0, 1.0, 0.0, -1.0],
            family=ChartAxisFamily.STATIC_SPATIAL,
        ),
        _series(
            series_id="b",
            label="B",
            x=[0.0, 1.5, 3.0],
            y=[0.0, 3.0, 0.0],
            family=ChartAxisFamily.STATIC_SPATIAL,
        ),
    ]
    selections = [
        CustomChartSelection(series_id="a", axis_target="L1", legend_label="A"),
        CustomChartSelection(series_id="b", axis_target="R1", legend_label="B"),
    ]

    x_values, _x_label, family, resampled = build_resampled_series(
        source_series=source,
        selections=selections,
    )

    assert family == ChartAxisFamily.STATIC_SPATIAL
    assert x_values == [0.0, 1.0, 2.0, 3.0]
    assert np.allclose(resampled[0].y_values, source[0].y)
    assert np.allclose(resampled[1].y_values, [0.0, 2.0, 2.0, 0.0])


def test_build_resampled_series_allows_static_and_transition_spatial_mix() -> None:
    source = [
        _series(
            series_id="deflection",
            label="Deflection",
            x=[0.0, 1.0, 2.0],
            y=[0.0, 1.0, 0.0],
            family=ChartAxisFamily.STATIC_SPATIAL,
        ),
        _series(
            series_id="k_profile",
            label="Transition k(x)",
            x=[0.0, 1.0, 2.0],
            y=[40_000.0, 60_000.0, 80_000.0],
            family=ChartAxisFamily.TRANSITION_SPATIAL,
        ),
    ]
    selections = [
        CustomChartSelection(series_id="deflection", axis_target="L1", legend_label="Deflection"),
        CustomChartSelection(series_id="k_profile", axis_target="R1", legend_label="k(x)"),
    ]

    x_values, _x_label, family, resampled = build_resampled_series(
        source_series=source,
        selections=selections,
    )

    assert family == ChartAxisFamily.STATIC_SPATIAL
    assert x_values == [0.0, 1.0, 2.0]
    assert np.allclose(resampled[0].y_values, source[0].y)
    assert np.allclose(resampled[1].y_values, source[1].y)


def test_build_resampled_series_rejects_more_than_four_series() -> None:
    source = [
        _series(
            series_id=f"s{idx}",
            label=f"S{idx}",
            x=[0.0, 1.0, 2.0],
            y=[0.0, float(idx), 0.0],
            family=ChartAxisFamily.STATIC_SPATIAL,
        )
        for idx in range(5)
    ]
    selections = [
        CustomChartSelection(series_id=f"s{idx}", axis_target="L1", legend_label=f"S{idx}")
        for idx in range(5)
    ]

    with pytest.raises(ValueError, match="at most 4 series"):
        build_resampled_series(source_series=source, selections=selections)


def test_custom_chart_dialog_hides_incompatible_series_rows(qapp: QApplication) -> None:
    source = [
        _series(
            series_id="a",
            label="A",
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            family=ChartAxisFamily.STATIC_SPATIAL,
        ),
        _series(
            series_id="b",
            label="B",
            x=[0.0, 1.0],
            y=[1.0, 0.0],
            family=ChartAxisFamily.STATIC_SPATIAL,
        ),
        _series(
            series_id="c",
            label="C",
            x=[0.0, 1.0],
            y=[0.5, 0.5],
            family=ChartAxisFamily.TIME,
        ),
    ]
    dialog = CustomChartDialog(source_series=source)
    try:
        assert not dialog.series_table.isRowHidden(0)
        assert not dialog.series_table.isRowHidden(1)
        assert dialog.series_table.isRowHidden(2)

        row0 = dialog.series_table.item(0, 0)
        assert row0 is not None
        row0.setCheckState(Qt.Unchecked)
        qapp.processEvents()
        assert not dialog.series_table.isRowHidden(2)

        row2 = dialog.series_table.item(2, 0)
        assert row2 is not None
        row2.setCheckState(Qt.Checked)
        qapp.processEvents()
        assert dialog.series_table.isRowHidden(0)
        assert dialog.series_table.isRowHidden(1)
        assert not dialog.series_table.isRowHidden(2)
    finally:
        dialog.close()


def test_plot_panel_render_custom_chart_accepts_cross_chart_source_series(qapp: QApplication) -> None:
    panel = PlotPanel()
    panel.set_chart_context(chart_id="moment", title="Moment")
    panel.update_plot(
        [0.0, 1.0, 2.0],
        [0.0, 0.5, 0.0],
        title="Moment",
        xlabel="x (m)",
        ylabel="M (kN.m)",
    )

    external_deflection = _series(
        series_id="deflection:primary",
        label="Deflection",
        x=[0.0, 1.0, 2.0],
        y=[0.0, 1.0, 0.0],
        family=ChartAxisFamily.STATIC_SPATIAL,
    )
    local_moment = _series(
        series_id="moment:primary",
        label="Moment",
        x=[0.0, 1.0, 2.0],
        y=[0.0, 0.5, 0.0],
        family=ChartAxisFamily.STATIC_SPATIAL,
    )

    panel.render_custom_chart(
        selections=[CustomChartSelection(series_id="deflection:primary", axis_target="L1", legend_label="Defl")],
        title="Custom chart",
        source_series=[external_deflection, local_moment],
    )

    assert len(panel.axes.lines) == 2
    assert len(panel.axes.texts) >= 1
    assert any("Defl" in text.get_text() for text in panel.axes.texts)
    assert any("x=" in text.get_text() and "y=" in text.get_text() for text in panel.axes.texts)


def test_plot_panel_render_custom_chart_keeps_y_scale_per_axis(qapp: QApplication) -> None:
    panel = PlotPanel()
    deflection = _series(
        series_id="deflection:primary",
        label="Deflection",
        x=[-1.0, 0.0, 1.0],
        y=[0.0, 2.31, 0.0],
        family=ChartAxisFamily.STATIC_SPATIAL,
    )
    moment = _series(
        series_id="moment:primary",
        label="Bending moment",
        x=[-1.0, 0.0, 1.0],
        y=[0.0, 41.4, 0.0],
        family=ChartAxisFamily.STATIC_SPATIAL,
    )

    panel.render_custom_chart(
        selections=[
            CustomChartSelection(series_id="deflection:primary", axis_target="L1", legend_label="Deflection"),
            CustomChartSelection(series_id="moment:primary", axis_target="R1", legend_label="Bending moment"),
        ],
        title="Custom chart",
        source_series=[deflection, moment],
    )

    left_span = panel.axes.get_ylim()[1] - panel.axes.get_ylim()[0]
    right_axis = panel.figure.axes[1]
    right_span = right_axis.get_ylim()[1] - right_axis.get_ylim()[0]

    assert left_span < 3.0
    assert right_span > 40.0
    assert panel.axes.get_ylim()[1] == pytest.approx(2.4255)


def test_plot_panel_render_custom_chart_restores_axis_units(qapp: QApplication) -> None:
    panel = PlotPanel()
    deflection = _series(
        series_id="deflection:primary",
        label="Deflection",
        x=[-1.0, 0.0, 1.0],
        y=[0.0, 2.31, 0.0],
        family=ChartAxisFamily.STATIC_SPATIAL,
        x_label="x",
        x_unit="m",
        y_label="w",
        y_unit="mm",
    )
    moment = _series(
        series_id="moment:primary",
        label="Bending moment",
        x=[-1.0, 0.0, 1.0],
        y=[0.0, 41.4, 0.0],
        family=ChartAxisFamily.STATIC_SPATIAL,
        x_label="x",
        x_unit="m",
        y_label="M",
        y_unit="kN.m",
    )

    panel.render_custom_chart(
        selections=[
            CustomChartSelection(series_id="deflection:primary", axis_target="L1", legend_label="Deflection"),
            CustomChartSelection(series_id="moment:primary", axis_target="R1", legend_label="Bending moment"),
        ],
        title="Custom chart",
        source_series=[deflection, moment],
    )

    assert panel.axes.get_xlabel() == "x (m)"
    assert panel.axes.get_ylabel() == "w (mm)"
    assert panel.figure.axes[1].get_ylabel() == "M (kN.m)"


def test_plot_panel_render_custom_chart_accepts_stress_source_series(qapp: QApplication) -> None:
    panel = PlotPanel()
    panel.set_chart_context(chart_id="stress", title="Stress")
    panel.update_multi_plot(
        [
            ([0.0, 1.0, 2.0], [50.0, 120.0, 60.0], "Rail stress - top fibre (bending)"),
            ([0.0, 1.0, 2.0], [-50.0, -120.0, -60.0], "Rail stress - bottom fibre (bending)"),
        ],
        title="Stress",
        xlabel="Position (m)",
        ylabel="Stress / Pressure (MPa)",
        critical_labels=True,
    )

    stress_top = _series(
        series_id="stress:primary:0",
        label="Rail stress - top fibre (bending)",
        x=[0.0, 1.0, 2.0],
        y=[50.0, 120.0, 60.0],
        family=ChartAxisFamily.STATIC_SPATIAL,
    )

    panel.render_custom_chart(
        selections=[CustomChartSelection(series_id="stress:primary:0", axis_target="L1", legend_label="Top fibre")],
        title="Stress custom chart",
        source_series=[stress_top],
    )

    assert len(panel.axes.lines) >= 1
    assert any("Top fibre" in text.get_text() for text in panel.axes.texts)
    assert any("x=" in text.get_text() and "y=" in text.get_text() for text in panel.axes.texts)

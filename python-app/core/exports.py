"""Export utilities for BOEF analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

from core.model import (
    PointLoad,
    deflection_at,
    moment_at,
    reaction_at,
    shear_at,
    sleeper_seat_loads,
)
from core.units import n_per_m2_to_mn_per_m2

if TYPE_CHECKING:
    from core.analysis import AnalysisResult as CoreAnalysisResult


@dataclass(frozen=True)
class AnalysisPoint:
    """Beam response values at a single position."""

    x_m: float
    deflection_m: float
    moment_nm: float
    shear_n: float
    reaction_n_per_m: float
    slope_rad: float | None = None
    rotation_rad: float | None = None
    shear_angle_rad: float | None = None
    winkler_reaction_n_per_m: float | None = None
    pasternak_shear_reaction_n_per_m: float | None = None


@dataclass(frozen=True)
class SleeperLoadRow:
    """Sleeper table row with load and pressure values."""

    index: int
    position_m: float
    seat_load_n_per_rail: float
    total_sleeper_load_n: float
    ballast_pressure_pa: float


@dataclass(frozen=True)
class ReportSummary:
    """Summary values for report outputs."""

    max_deflection_m: float
    max_moment_nm: float
    max_shear_n: float
    max_reaction_n_per_m: float
    max_sleeper_load_n: float
    max_ballast_pressure_pa: float


@dataclass(frozen=True)
class AnalysisInputs:
    """Inputs required for analysis table exports."""

    x_positions_m: Sequence[float]
    loads: Iterable[PointLoad]
    foundation_modulus_n_per_m2: float
    elastic_modulus_pa: float
    moment_inertia_m4: float


@dataclass(frozen=True)
class SleeperInputs:
    """Inputs required for sleeper load/pressure exports."""

    sleeper_positions_m: Sequence[float]
    tributary_length_m: float
    sleeper_length_m: float
    sleeper_width_m: float
    loads: Iterable[PointLoad]
    foundation_modulus_n_per_m2: float
    elastic_modulus_pa: float
    moment_inertia_m4: float


@dataclass(frozen=True)
class ReportInputs:
    """All inputs required for a PDF report export."""

    analysis: AnalysisInputs
    sleepers: SleeperInputs
    rail_count: int = 2
    title: str = "1DTransport.com: BOEF Calculation Tool Report"
    assumptions: Sequence[str] | None = None
    analysis_result: CoreAnalysisResult | None = None


def build_analysis_points(inputs: AnalysisInputs) -> list[AnalysisPoint]:
    """Compute response values at each x position."""
    _require_non_empty(inputs.x_positions_m, "x_positions_m")
    validated_loads = _require_loads(inputs.loads)
    _require_positive(inputs.foundation_modulus_n_per_m2, "foundation_modulus_n_per_m2")
    _require_positive(inputs.elastic_modulus_pa, "elastic_modulus_pa")
    _require_positive(inputs.moment_inertia_m4, "moment_inertia_m4")

    points: list[AnalysisPoint] = []
    for x_m in inputs.x_positions_m:
        deflection_m = deflection_at(
            x_m,
            validated_loads,
            inputs.foundation_modulus_n_per_m2,
            inputs.elastic_modulus_pa,
            inputs.moment_inertia_m4,
        )
        moment_nm = moment_at(
            x_m,
            validated_loads,
            inputs.foundation_modulus_n_per_m2,
            inputs.elastic_modulus_pa,
            inputs.moment_inertia_m4,
        )
        shear_n = shear_at(
            x_m,
            validated_loads,
            inputs.foundation_modulus_n_per_m2,
            inputs.elastic_modulus_pa,
            inputs.moment_inertia_m4,
        )
        reaction_n_per_m = reaction_at(
            x_m,
            validated_loads,
            inputs.foundation_modulus_n_per_m2,
            inputs.elastic_modulus_pa,
            inputs.moment_inertia_m4,
        )
        points.append(
            AnalysisPoint(
                x_m=x_m,
                deflection_m=deflection_m,
                moment_nm=moment_nm,
                shear_n=shear_n,
                reaction_n_per_m=reaction_n_per_m,
            )
        )
    return points


def build_sleeper_load_rows(
    inputs: SleeperInputs,
    rail_count: int = 2,
) -> list[SleeperLoadRow]:
    """Compute sleeper seat load and ballast pressure rows."""
    _require_non_empty(inputs.sleeper_positions_m, "sleeper_positions_m")
    validated_loads = _require_loads(inputs.loads)
    _require_positive(inputs.tributary_length_m, "tributary_length_m")
    _require_positive(inputs.sleeper_length_m, "sleeper_length_m")
    _require_positive(inputs.sleeper_width_m, "sleeper_width_m")
    _require_positive(inputs.foundation_modulus_n_per_m2, "foundation_modulus_n_per_m2")
    _require_positive(inputs.elastic_modulus_pa, "elastic_modulus_pa")
    _require_positive(inputs.moment_inertia_m4, "moment_inertia_m4")
    if rail_count <= 0:
        raise ValueError("rail_count must be positive")

    seat_loads = sleeper_seat_loads(
        inputs.sleeper_positions_m,
        inputs.tributary_length_m,
        validated_loads,
        inputs.foundation_modulus_n_per_m2,
        inputs.elastic_modulus_pa,
        inputs.moment_inertia_m4,
    )

    bearing_area = inputs.sleeper_length_m * inputs.sleeper_width_m
    _require_positive(bearing_area, "bearing_area")

    rows: list[SleeperLoadRow] = []
    for index, (position_m, seat_load_n) in enumerate(
        zip(inputs.sleeper_positions_m, seat_loads)
    ):
        total_load_n = seat_load_n * rail_count
        ballast_pressure_pa = total_load_n / bearing_area
        rows.append(
            SleeperLoadRow(
                index=index,
                position_m=position_m,
                seat_load_n_per_rail=seat_load_n,
                total_sleeper_load_n=total_load_n,
                ballast_pressure_pa=ballast_pressure_pa,
            )
        )
    return rows


def write_analysis_csv(path: str | Path, points: Sequence[AnalysisPoint]) -> None:
    """Write x, y(x), M(x), V(x), and rail support reaction to CSV."""
    _require_non_empty(points, "analysis points")
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "x_m",
            "deflection_m",
            "moment_nm",
            "shear_n",
            "reaction_n_per_m",
            "slope_rad",
            "rotation_rad",
            "shear_angle_rad",
            "winkler_reaction_n_per_m",
            "pasternak_shear_reaction_n_per_m",
        ])
        for point in points:
            writer.writerow(
                [
                    f"{point.x_m:.6f}",
                    f"{point.deflection_m:.10f}",
                    f"{point.moment_nm:.6f}",
                    f"{point.shear_n:.6f}",
                    f"{point.reaction_n_per_m:.6f}",
                    _format_optional(point.slope_rad),
                    _format_optional(point.rotation_rad),
                    _format_optional(point.shear_angle_rad),
                    _format_optional(point.winkler_reaction_n_per_m),
                    _format_optional(point.pasternak_shear_reaction_n_per_m),
                ]
            )


def write_sleeper_load_csv(path: str | Path, rows: Sequence[SleeperLoadRow]) -> None:
    """Write the sleeper load/pressure table to CSV."""
    _require_non_empty(rows, "sleeper rows")
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sleeper_index",
            "position_m",
            "seat_load_n_per_rail",
            "total_sleeper_load_n",
            "ballast_pressure_pa",
        ])
        for row in rows:
            writer.writerow(
                [
                    row.index,
                    f"{row.position_m:.6f}",
                    f"{row.seat_load_n_per_rail:.6f}",
                    f"{row.total_sleeper_load_n:.6f}",
                    f"{row.ballast_pressure_pa:.6f}",
                ]
            )


def export_pdf_report(
    path: str | Path,
    inputs: ReportInputs,
) -> ReportSummary:
    """Export a PDF report with inputs, assumptions, summary, and plots.

    When an AnalysisResult is supplied, report plots and summary values use
    those arrays directly so numerical/non-uniform runs are exported faithfully.
    """
    _, _, PdfPages = _load_matplotlib()
    result = inputs.analysis_result
    analysis_inputs = AnalysisInputs(
        x_positions_m=result.x_m if result is not None else inputs.analysis.x_positions_m,
        loads=_require_loads(inputs.analysis.loads),
        foundation_modulus_n_per_m2=inputs.analysis.foundation_modulus_n_per_m2,
        elastic_modulus_pa=inputs.analysis.elastic_modulus_pa,
        moment_inertia_m4=inputs.analysis.moment_inertia_m4,
    )
    sleeper_inputs = SleeperInputs(
        sleeper_positions_m=(
            result.sleeper_positions_m if result is not None else inputs.sleepers.sleeper_positions_m
        ),
        tributary_length_m=inputs.sleepers.tributary_length_m,
        sleeper_length_m=inputs.sleepers.sleeper_length_m,
        sleeper_width_m=inputs.sleepers.sleeper_width_m,
        loads=_require_loads(inputs.sleepers.loads),
        foundation_modulus_n_per_m2=inputs.sleepers.foundation_modulus_n_per_m2,
        elastic_modulus_pa=inputs.sleepers.elastic_modulus_pa,
        moment_inertia_m4=inputs.sleepers.moment_inertia_m4,
    )
    normalized_inputs = ReportInputs(
        analysis=analysis_inputs,
        sleepers=sleeper_inputs,
        rail_count=inputs.rail_count,
        title=inputs.title,
        assumptions=inputs.assumptions,
        analysis_result=result,
    )
    if result is None:
        analysis_points = build_analysis_points(analysis_inputs)
        sleeper_rows = build_sleeper_load_rows(sleeper_inputs, rail_count=inputs.rail_count)
    else:
        analysis_points = _build_analysis_points_from_result(result)
        sleeper_rows = _build_sleeper_rows_from_result(result, rail_count=inputs.rail_count)
    summary = _build_summary(analysis_points, sleeper_rows)
    assumptions = list(inputs.assumptions or [])
    assumptions.extend(
        [
            (
                "Plots and summary use the supplied AnalysisResult arrays."
                if result is not None
                else "Loads are modeled as point loads on an infinite beam on Winkler foundation."
            ),
            (
                "Sleeper rows use the supplied AnalysisResult sleeper arrays."
                if result is not None
                else "Sleeper seat loads integrate foundation reaction over tributary length."
            ),
            f"Sleeper totals assume {inputs.rail_count} rail(s).",
        ]
    )

    path = Path(path)
    with PdfPages(path) as pdf:
        _render_report_cover(pdf, normalized_inputs, assumptions, summary)
        _render_plot(pdf, analysis_points, "Deflection y(x)", "x (m)", "y (m)",
                     lambda p: p.deflection_m)
        _render_plot(pdf, analysis_points, "Moment M(x)", "x (m)", "M (N·m)",
                     lambda p: p.moment_nm)
        _render_plot(pdf, analysis_points, "Shear V(x)", "x (m)", "V (N)",
                     lambda p: p.shear_n)
        _render_plot(pdf, analysis_points, "Rail support reaction R_support(x)", "x (m)", "R_support (N/m)",
                     lambda p: p.reaction_n_per_m)

    return summary


def _build_analysis_points_from_result(result: CoreAnalysisResult) -> list[AnalysisPoint]:
    _require_equal_length_series(
        ("x_m", result.x_m),
        ("deflection_m", result.deflection_m),
        ("moment_nm", result.moment_nm),
        ("shear_n", result.shear_n),
        ("reaction_n_per_m", result.reaction_n_per_m),
    )
    return [
        AnalysisPoint(
            x_m=x_m,
            deflection_m=result.deflection_m[index],
            moment_nm=result.moment_nm[index],
            shear_n=result.shear_n[index],
            reaction_n_per_m=result.reaction_n_per_m[index],
            slope_rad=_optional_at(result.slope_rad, index),
            rotation_rad=_optional_at(result.rotation_rad, index),
            shear_angle_rad=_optional_at(result.shear_angle_rad, index),
            winkler_reaction_n_per_m=_optional_at(result.winkler_reaction_n_per_m, index),
            pasternak_shear_reaction_n_per_m=_optional_at(
                result.pasternak_shear_reaction_n_per_m, index
            ),
        )
        for index, x_m in enumerate(result.x_m)
    ]


def _format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _optional_at(values: Sequence[float] | None, index: int) -> float | None:
    if values is None:
        return None
    return values[index]


def _build_sleeper_rows_from_result(
    result: CoreAnalysisResult,
    *,
    rail_count: int,
) -> list[SleeperLoadRow]:
    if rail_count <= 0:
        raise ValueError("rail_count must be positive")
    _require_equal_length_series(
        ("sleeper_positions_m", result.sleeper_positions_m),
        ("sleeper_loads_n", result.sleeper_loads_n),
        ("sleeper_pressures_pa", result.sleeper_pressures_pa),
    )
    return [
        SleeperLoadRow(
            index=index,
            position_m=position_m,
            seat_load_n_per_rail=result.sleeper_loads_n[index] / rail_count,
            total_sleeper_load_n=result.sleeper_loads_n[index],
            ballast_pressure_pa=result.sleeper_pressures_pa[index],
        )
        for index, position_m in enumerate(result.sleeper_positions_m)
    ]


def _build_summary(
    analysis_points: Sequence[AnalysisPoint],
    sleeper_rows: Sequence[SleeperLoadRow],
) -> ReportSummary:
    _require_non_empty(analysis_points, "analysis points")
    _require_non_empty(sleeper_rows, "sleeper rows")
    max_deflection = max(abs(point.deflection_m) for point in analysis_points)
    max_moment = max(abs(point.moment_nm) for point in analysis_points)
    max_shear = max(abs(point.shear_n) for point in analysis_points)
    max_reaction = max(abs(point.reaction_n_per_m) for point in analysis_points)
    max_sleeper_load = max(row.total_sleeper_load_n for row in sleeper_rows)
    max_ballast_pressure = max(row.ballast_pressure_pa for row in sleeper_rows)
    return ReportSummary(
        max_deflection_m=max_deflection,
        max_moment_nm=max_moment,
        max_shear_n=max_shear,
        max_reaction_n_per_m=max_reaction,
        max_sleeper_load_n=max_sleeper_load,
        max_ballast_pressure_pa=max_ballast_pressure,
    )


def _render_report_cover(
    pdf: PdfPages,
    inputs: ReportInputs,
    assumptions: Sequence[str],
    summary: ReportSummary,
) -> None:
    _, plt, _ = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    lines = [inputs.title, "", "Inputs:"]
    analysis = inputs.analysis
    lines.extend(
        [
            f"Foundation modulus: {n_per_m2_to_mn_per_m2(analysis.foundation_modulus_n_per_m2):.3f} MN/m²",
            f"Elastic modulus: {analysis.elastic_modulus_pa:.3e} Pa",
            f"Moment of inertia: {analysis.moment_inertia_m4:.3e} m⁴",
            f"X positions: {len(analysis.x_positions_m)} points",
        ]
    )
    lines.append("Loads:")
    for load in analysis.loads:
        lines.append(f"  - {load.load_newtons:.2f} N @ x={load.position_m:.3f} m")

    sleepers = inputs.sleepers
    lines.extend(
        [
            "",
            "Sleeper inputs:",
            f"Sleeper spacing: {sleepers.tributary_length_m:.3f} m",
            f"Sleeper size: {sleepers.sleeper_length_m:.3f} m x {sleepers.sleeper_width_m:.3f} m",
            f"Sleeper count: {len(sleepers.sleeper_positions_m)}",
            f"Rail count: {inputs.rail_count}",
            "",
            "Assumptions:",
        ]
    )
    lines.extend([f"- {assumption}" for assumption in assumptions])
    lines.extend(
        [
            "",
            "Summary (max abs):",
            f"Deflection: {summary.max_deflection_m:.6e} m",
            f"Moment: {summary.max_moment_nm:.6e} N·m",
            f"Shear: {summary.max_shear_n:.6e} N",
            f"Reaction: {summary.max_reaction_n_per_m:.6e} N/m",
            f"Sleeper load: {summary.max_sleeper_load_n:.6e} N",
            f"Ballast pressure: {summary.max_ballast_pressure_pa:.6e} Pa",
        ]
    )
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _render_plot(
    pdf: PdfPages,
    points: Sequence[AnalysisPoint],
    title: str,
    x_label: str,
    y_label: str,
    y_accessor,
) -> None:
    _, plt, _ = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x_values = [point.x_m for point in points]
    y_values = [y_accessor(point) for point in points]
    ax.plot(x_values, y_values)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.4)
    pdf.savefig(fig)
    plt.close(fig)


def _load_matplotlib():
    _require_matplotlib_modules()
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    return matplotlib, plt, PdfPages


def _require_matplotlib_modules() -> None:
    if (
        importlib.util.find_spec("matplotlib") is None
        or importlib.util.find_spec("matplotlib.backends.backend_pdf") is None
        or importlib.util.find_spec("matplotlib.pyplot") is None
    ):
        raise ModuleNotFoundError(
            "matplotlib is required for PDF exports. Install it to enable report exports."
        )


def _require_non_empty(values: Sequence[object], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")


def _require_equal_length_series(*series: tuple[str, Sequence[object]]) -> None:
    if not series:
        return
    expected = len(series[0][1])
    for name, values in series:
        if len(values) != expected:
            raise ValueError(f"{name} length must match {series[0][0]}")
    if expected == 0:
        raise ValueError(f"{series[0][0]} must not be empty")


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_loads(loads: Iterable[PointLoad]) -> list[PointLoad]:
    validated = list(loads)
    if not validated:
        raise ValueError("loads must not be empty")
    for load in validated:
        if load.load_newtons < 0:
            raise ValueError("load_newtons must be non-negative")
    return validated

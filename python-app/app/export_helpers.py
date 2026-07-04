"""Helpers for exporting analysis outputs from the GUI."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from core.analysis import AnalysisInputs, AnalysisResult
from core.dynamic.config import DippedJointConfig
from core.envelope import EnvelopeResult
from core.stress_metrics import (
    BearingGeometry,
    StressResults,
    build_stress_results_from_single,
    capping_pressure_2to1_load_conserving,
    stress_top_bottom_series_from_moment,
)
from core.transition import TransitionRunMode, TransitionRunResult
from core.dynamic.results import (
    DippedJointResult,
    DynamicResult,
    DynamicTransitionResult,
)
from core.exports import (
    AnalysisInputs as ExportAnalysisInputs,
    SleeperInputs as ExportSleeperInputs,
    build_sleeper_load_rows,
)
from core.units import n_to_kn, n_per_m2_to_mn_per_m2


def _normalize_for_hash(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize_for_hash(asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalize_for_hash(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item) for item in value]
    if hasattr(value, "value"):  # Enum-like
        return getattr(value, "value")
    return value


def _enum_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value)


def compute_inputs_hash(payload: Any) -> str:
    normalized = _normalize_for_hash(payload)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_export_metadata(
    path: str | Path,
    *,
    solver_mode: str,
    inputs_payload: Any,
    units: str = "SI",
    parameter_trace: Any | None = None,
) -> Path:
    export_path = Path(path)
    metadata_path = export_path.with_suffix(export_path.suffix + ".meta.json")
    metadata = {
        "units": units,
        "solver_mode": solver_mode,
        "inputs_hash": compute_inputs_hash(inputs_payload),
        "inputs_payload": _normalize_for_hash(inputs_payload),
        "source_export": export_path.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    if parameter_trace is not None:
        metadata["parameter_trace"] = _normalize_for_hash(parameter_trace)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def build_export_inputs(
    analysis_inputs: AnalysisInputs,
    result: AnalysisResult,
) -> tuple[ExportAnalysisInputs, ExportSleeperInputs]:
    """Translate analysis inputs/results into export inputs (SI units)."""
    analysis = ExportAnalysisInputs(
        x_positions_m=result.x_m,
        loads=analysis_inputs.loads,
        foundation_modulus_n_per_m2=analysis_inputs.foundation_modulus_n_per_m2,
        elastic_modulus_pa=analysis_inputs.elastic_modulus_pa,
        moment_inertia_m4=analysis_inputs.moment_inertia_m4,
    )
    sleepers = ExportSleeperInputs(
        sleeper_positions_m=result.sleeper_positions_m,
        tributary_length_m=analysis_inputs.sleeper_spacing_m,
        sleeper_length_m=analysis_inputs.sleeper_length_m,
        sleeper_width_m=analysis_inputs.sleeper_width_m,
        loads=analysis_inputs.loads,
        foundation_modulus_n_per_m2=analysis_inputs.foundation_modulus_n_per_m2,
        elastic_modulus_pa=analysis_inputs.elastic_modulus_pa,
        moment_inertia_m4=analysis_inputs.moment_inertia_m4,
    )
    return analysis, sleepers


def write_analysis_csv_from_result(
    path: str | Path,
    analysis_inputs: AnalysisInputs,
    result: AnalysisResult,
    *,
    stress_results: StressResults | None = None,
    section_modulus_top_m3: float | None = None,
    section_modulus_bottom_m3: float | None = None,
) -> None:
    """Write the analysis table to CSV using the latest result."""
    sigma_top: list[float]
    sigma_bottom: list[float]
    if stress_results is not None and len(stress_results.sigma_top_fiber_pa) == len(result.x_m):
        sigma_top = list(stress_results.sigma_top_fiber_pa)
        sigma_bottom = list(stress_results.sigma_bottom_fiber_pa)
    else:
        top_modulus = (
            section_modulus_top_m3
            if section_modulus_top_m3 is not None
            else analysis_inputs.section_modulus_head_m3
            if analysis_inputs.section_modulus_head_m3 is not None
            else analysis_inputs.section_modulus_m3
        )
        bottom_modulus = (
            section_modulus_bottom_m3
            if section_modulus_bottom_m3 is not None
            else analysis_inputs.section_modulus_foot_m3
            if analysis_inputs.section_modulus_foot_m3 is not None
            else analysis_inputs.section_modulus_m3
        )
        sigma_top, sigma_bottom = stress_top_bottom_series_from_moment(
            moments_nm=result.moment_nm,
            section_modulus_top_m3=top_modulus,
            section_modulus_bottom_m3=bottom_modulus,
        )
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
                "sigma_top_fiber_pa",
                "sigma_bottom_fiber_pa",
            ]
        )
        for index, x_m in enumerate(result.x_m):
            writer.writerow(
                [
                    f"{x_m:.6f}",
                    f"{result.deflection_m[index]:.10f}",
                    f"{result.moment_nm[index]:.6f}",
                    f"{result.shear_n[index]:.6f}",
                    f"{result.reaction_n_per_m[index]:.6f}",
                    _format_optional(_optional_at(result.slope_rad, index)),
                    _format_optional(_optional_at(result.rotation_rad, index)),
                    _format_optional(_optional_at(result.shear_angle_rad, index)),
                    _format_optional(_optional_at(result.winkler_reaction_n_per_m, index)),
                    _format_optional(
                        _optional_at(result.pasternak_shear_reaction_n_per_m, index)
                    ),
                    _format_value(sigma_top[index]),
                    _format_value(sigma_bottom[index]),
                ]
            )


def write_sleeper_csv_from_result(
    path: str | Path,
    analysis_inputs: AnalysisInputs,
    result: AnalysisResult,
    *,
    stress_results: StressResults | None = None,
    ballast_thickness_m: float = 0.3,
) -> None:
    """Write the sleeper load table to CSV using the latest result."""
    _, sleepers = build_export_inputs(analysis_inputs, result)
    rows = build_sleeper_load_rows(sleepers, rail_count=2)
    if stress_results is not None and len(stress_results.sleeper_positions_m) == len(rows):
        ballast_signed = list(stress_results.q_ballast_signed_pa or [])
        ballast_comp = list(stress_results.q_ballast_comp_pa or [])
        capping_signed = list(stress_results.q_capping_signed_pa or [])
        capping_comp = list(stress_results.q_capping_comp_pa or [])
    else:
        fallback_stress = build_stress_results_from_single(
            x_m=result.x_m,
            moment_nm=result.moment_nm,
            sleeper_positions_m=[row.position_m for row in rows],
            sleeper_loads_n=[row.total_sleeper_load_n for row in rows],
            section_modulus_top_m3=analysis_inputs.section_modulus_m3,
            section_modulus_bottom_m3=analysis_inputs.section_modulus_m3,
            bearing_geometry=BearingGeometry(
                width_m=analysis_inputs.sleeper_width_m,
                length_m=analysis_inputs.sleeper_length_m,
                area_m2=analysis_inputs.sleeper_width_m * analysis_inputs.sleeper_length_m,
                provenance="sleeper_geometry",
            ),
            ballast_thickness_m=ballast_thickness_m,
        )
        ballast_signed = list(fallback_stress.q_ballast_signed_pa or [])
        ballast_comp = list(fallback_stress.q_ballast_comp_pa or [])
        capping_signed = list(fallback_stress.q_capping_signed_pa or [])
        capping_comp = list(fallback_stress.q_capping_comp_pa or [])

    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sleeper_index",
                "position_m",
                "seat_load_n_per_rail",
                "total_sleeper_load_n",
                "ballast_pressure_pa",
                "ballast_pressure_signed_pa",
                "ballast_pressure_comp_pa",
                "capping_pressure_signed_pa",
                "capping_pressure_comp_pa",
            ]
        )
        for index, row in enumerate(rows):
            writer.writerow(
                [
                    row.index,
                    f"{row.position_m:.6f}",
                    f"{row.seat_load_n_per_rail:.6f}",
                    f"{row.total_sleeper_load_n:.6f}",
                    f"{row.ballast_pressure_pa:.6f}",
                    _format_value(ballast_signed[index]) if index < len(ballast_signed) else "",
                    _format_value(ballast_comp[index]) if index < len(ballast_comp) else "",
                    _format_value(capping_signed[index]) if index < len(capping_signed) else "",
                    _format_value(capping_comp[index]) if index < len(capping_comp) else "",
                ]
            )


def _get_dynamic_probe(result: DynamicResult, probe_index: int):
    if not result.probes:
        raise ValueError("No dynamic probe data available.")
    if probe_index < 0 or probe_index >= len(result.probes):
        raise ValueError("Probe selection is out of range.")
    return result.probes[probe_index]


def _format_value(value: float) -> str:
    return f"{value:.8e}"


def _format_optional(value: float | None) -> str:
    return "" if value is None else _format_value(value)


def _optional_at(values: list[float] | None, index: int) -> float | None:
    if values is None:
        return None
    return values[index]


def write_dynamic_time_history_csv(path: str | Path, result: DynamicResult, probe_index: int = 0) -> None:
    """Write dynamic probe time history to CSV."""
    probe = _get_dynamic_probe(result, probe_index)
    path = Path(path)
    rows = zip(
        probe.time_s,
        probe.deflection_m,
        probe.moment_nm,
        probe.shear_n,
        probe.reaction_n_per_m,
        probe.damping_force_n_per_m,
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t_s",
                "deflection_m",
                "moment_nm",
                "shear_n",
                "reaction_n_per_m",
                "damping_force_n_per_m",
            ]
        )
        for row in rows:
            writer.writerow([_format_value(value) for value in row])


def write_dynamic_fft_csv(path: str | Path, result: DynamicResult, probe_index: int = 0) -> None:
    """Write dynamic probe FFT amplitude results to CSV."""
    probe = _get_dynamic_probe(result, probe_index)
    path = Path(path)
    rows = zip(probe.fft_frequency_hz, probe.fft_amplitude)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_hz", "amplitude_m"])
        for row in rows:
            writer.writerow([_format_value(value) for value in row])


def write_dynamic_psd_csv(path: str | Path, result: DynamicResult, probe_index: int = 0) -> None:
    """Write dynamic probe PSD results to CSV."""
    probe = _get_dynamic_probe(result, probe_index)
    path = Path(path)
    rows = zip(probe.psd_frequency_hz, probe.psd)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_hz", "psd_m2_per_hz"])
        for row in rows:
            writer.writerow([_format_value(value) for value in row])


def write_dipped_joint_csv(path: str | Path, config: DippedJointConfig, result: DippedJointResult) -> None:
    """Write dipped joint inputs and results to CSV."""
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Section", "Parameter", "Value", "Unit"])
        writer.writerow(["Inputs", "Static load P₀", f"{n_to_kn(config.static_wheel_load_n):.3f}", "kN"])
        writer.writerow(["Inputs", "Speed", f"{config.speed_m_per_s:.2f}", "m/s"])
        writer.writerow(["Inputs", "Dip angle (2α)", f"{config.total_dip_angle_rad * 1.0e3:.3f}", "mrad"])
        writer.writerow(
            ["Inputs", "Hertzian stiffness kₕ", f"{config.hertzian_stiffness_n_per_m / 1.0e6:.2f}", "MN/m"]
        )
        writer.writerow(["Inputs", "Unsprung mass mᵤ", f"{config.unsprung_mass_kg:.2f}", "kg"])
        writer.writerow(["Inputs", "Effective track mass mᵀ₁", f"{config.track_mass_p1_kg:.2f}", "kg"])
        writer.writerow(["Inputs", "Equivalent track mass mᵀ₂", f"{config.track_mass_p2_kg:.2f}", "kg"])
        writer.writerow(
            ["Inputs", "Equivalent track stiffness kᵀ₂", f"{config.track_stiffness_p2_n_per_m / 1.0e6:.2f}", "MN/m"]
        )
        writer.writerow(
            ["Inputs", "Equivalent track damping cᵀ", f"{config.track_damping_p2_n_s_per_m / 1.0e3:.2f}", "kN·s/m"]
        )
        writer.writerow(["Results", "Peak force P₁", f"{n_to_kn(result.p1_n):.3f}", "kN"])
        writer.writerow(["Results", "Peak force P₂", f"{n_to_kn(result.p2_n):.3f}", "kN"])
        writer.writerow(["Results", "DAF at P₁", f"{result.p1_dynamic_amplification:.2f}", "-"])
        writer.writerow(["Results", "DAF at P₂", f"{result.p2_dynamic_amplification:.2f}", "-"])


def write_dynamic_transition_metrics_csv(path: str | Path, result: DynamicTransitionResult) -> None:
    """Write dynamic transition metrics to CSV."""
    path = Path(path)
    metrics = result.metrics
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "solver_fidelity",
                "profile_type",
                "run_mode",
                "k1_mn_per_m2",
                "k2_mn_per_m2",
                "transition_length_m",
                "segment_length_m",
                "x_ref_m",
                "x_ref_start_m",
                "x_ref_end_m",
                "x_ref_step_m",
                "envelope_count",
                "max_deflection_m",
                "max_moment_nm",
                "max_shear_n",
                "max_reaction_n_per_m",
                "governing_x_ref_m",
                "risk_index",
                "critical_speed_ratio",
                "dynamic_amplification",
                "transition_stiffness_ratio",
            ]
        )
        writer.writerow(
            [
                result.solver_fidelity,
                result.profile_type,
                result.run_mode,
                _format_value(n_per_m2_to_mn_per_m2(result.k1_n_per_m2)),
                _format_value(n_per_m2_to_mn_per_m2(result.k2_n_per_m2)) if result.k2_n_per_m2 is not None else "",
                _format_value(result.transition_length_m) if result.transition_length_m is not None else "",
                _format_value(result.segment_length_m) if result.segment_length_m is not None else "",
                _format_value(result.x_ref_m) if result.x_ref_m is not None else "",
                _format_value(result.x_ref_start_m) if result.x_ref_start_m is not None else "",
                _format_value(result.x_ref_end_m) if result.x_ref_end_m is not None else "",
                _format_value(result.x_ref_step_m) if result.x_ref_step_m is not None else "",
                str(result.envelope_count),
                _format_value(metrics.max_deflection_m),
                _format_value(metrics.max_moment_nm),
                _format_value(metrics.max_shear_n),
                _format_value(metrics.max_reaction_n_per_m),
                _format_value(metrics.governing_x_ref_m),
                _format_value(metrics.risk_index),
                _format_value(metrics.critical_speed_ratio),
                _format_value(metrics.dynamic_amplification),
                (
                    _format_value(metrics.transition_stiffness_ratio)
                    if metrics.transition_stiffness_ratio is not None
                    else ""
                ),
            ]
        )


def write_dynamic_transition_series_csv(path: str | Path, result: DynamicTransitionResult) -> None:
    """Write dynamic transition x-series to CSV."""
    path = Path(path)
    series = result.series
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        is_envelope = result.run_mode == "envelope"
        if is_envelope:
            writer.writerow(
                [
                    "x_m",
                    "k_profile_mn_per_m2",
                    "deflection_max_m",
                    "deflection_min_m",
                    "moment_max_nm",
                    "moment_min_nm",
                    "moment_abs_max_nm",
                    "shear_max_n",
                    "shear_min_n",
                    "shear_abs_max_n",
                    "reaction_max_n_per_m",
                    "reaction_min_n_per_m",
                    "reaction_abs_max_n_per_m",
                ]
            )
            for idx, x_value in enumerate(series.x_m):
                writer.writerow(
                    [
                        _format_value(x_value),
                        _format_value(n_per_m2_to_mn_per_m2(series.k_profile_n_per_m2[idx])),
                        _format_value(series.deflection_max_m[idx]) if series.deflection_max_m is not None else "",
                        _format_value(series.deflection_min_m[idx]) if series.deflection_min_m is not None else "",
                        _format_value(series.moment_max_nm[idx]) if series.moment_max_nm is not None else "",
                        _format_value(series.moment_min_nm[idx]) if series.moment_min_nm is not None else "",
                        (
                            _format_value(max(abs(series.moment_max_nm[idx]), abs(series.moment_min_nm[idx])))
                            if series.moment_max_nm is not None and series.moment_min_nm is not None
                            else ""
                        ),
                        _format_value(series.shear_max_n[idx]) if series.shear_max_n is not None else "",
                        _format_value(series.shear_min_n[idx]) if series.shear_min_n is not None else "",
                        (
                            _format_value(max(abs(series.shear_max_n[idx]), abs(series.shear_min_n[idx])))
                            if series.shear_max_n is not None and series.shear_min_n is not None
                            else ""
                        ),
                        (
                            _format_value(series.reaction_max_n_per_m[idx])
                            if series.reaction_max_n_per_m is not None
                            else ""
                        ),
                        (
                            _format_value(series.reaction_min_n_per_m[idx])
                            if series.reaction_min_n_per_m is not None
                            else ""
                        ),
                        (
                            _format_value(
                                max(abs(series.reaction_max_n_per_m[idx]), abs(series.reaction_min_n_per_m[idx]))
                            )
                            if series.reaction_max_n_per_m is not None
                            and series.reaction_min_n_per_m is not None
                            else ""
                        ),
                    ]
                )
            return

        writer.writerow(
            [
                "x_m",
                "k_profile_mn_per_m2",
                "deflection_m",
                "moment_nm",
                "shear_n",
                "reaction_n_per_m",
            ]
        )
        for idx, x_value in enumerate(series.x_m):
            writer.writerow(
                [
                    _format_value(x_value),
                    _format_value(n_per_m2_to_mn_per_m2(series.k_profile_n_per_m2[idx])),
                    _format_value(series.deflection_m[idx]) if series.deflection_m is not None else "",
                    _format_value(series.moment_nm[idx]) if series.moment_nm is not None else "",
                    _format_value(series.shear_n[idx]) if series.shear_n is not None else "",
                    _format_value(series.reaction_n_per_m[idx]) if series.reaction_n_per_m is not None else "",
                ]
            )


def write_envelope_analysis_csv(
    path: str | Path,
    result: EnvelopeResult,
    *,
    stress_results: StressResults | None = None,
) -> None:
    """Write envelope x-series results to CSV."""
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "x_m",
                "deflection_max_m",
                "deflection_min_m",
                "moment_max_nm",
                "moment_min_nm",
                "shear_max_n",
                "shear_min_n",
                "reaction_max_n_per_m",
                "reaction_min_n_per_m",
                "deflection_abs_max_m",
                "moment_abs_max_nm",
                "shear_abs_max_n",
                "reaction_abs_max_n_per_m",
                "sigma_top_fiber_ub_pa",
                "sigma_bottom_fiber_ub_pa",
            ]
        )
        for i, x in enumerate(result.x_m):
            writer.writerow(
                [
                    _format_value(x),
                    _format_value(result.deflection_max_m[i]),
                    _format_value(result.deflection_min_m[i]),
                    _format_value(result.moment_max_nm[i]),
                    _format_value(result.moment_min_nm[i]),
                    _format_value(result.shear_max_n[i]),
                    _format_value(result.shear_min_n[i]),
                    _format_value(result.reaction_max_n_per_m[i]),
                    _format_value(result.reaction_min_n_per_m[i]),
                    _format_value(result.deflection_abs_max_m[i]),
                    _format_value(result.moment_abs_max_nm[i]),
                    _format_value(result.shear_abs_max_n[i]),
                    _format_value(result.reaction_abs_max_n_per_m[i]),
                    (
                        _format_value(stress_results.sigma_top_fiber_pa[i])
                        if stress_results is not None and i < len(stress_results.sigma_top_fiber_pa)
                        else ""
                    ),
                    (
                        _format_value(stress_results.sigma_bottom_fiber_pa[i])
                        if stress_results is not None and i < len(stress_results.sigma_bottom_fiber_pa)
                        else ""
                    ),
                ]
            )


def write_envelope_sleeper_csv(
    path: str | Path,
    result: EnvelopeResult,
    *,
    stress_results: StressResults | None = None,
    bearing_geometry: BearingGeometry | None = None,
    ballast_thickness_m: float | None = None,
) -> None:
    """Write envelope sleeper results to CSV."""
    path = Path(path)
    depths = sorted(result.formation_stress_max_pa_by_depth.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "sleeper_index",
            "position_m",
            "load_max_n",
            "load_min_n",
            "ballast_pressure_max_pa",
            "ballast_pressure_min_pa",
        ]
        for depth in depths:
            header.append(f"formation_stress_max_pa_z{depth:.2f}")
            header.append(f"formation_stress_min_pa_z{depth:.2f}")
        header.extend(
            [
                "ballast_pressure_max_comp_pa",
                "capping_pressure_max_comp_pa",
                "capping_pressure_signed_max_pa",
                "capping_pressure_signed_min_pa",
            ]
        )
        writer.writerow(header)

        capping_signed_max: list[float] = []
        capping_signed_min: list[float] = []
        if bearing_geometry is not None and ballast_thickness_m is not None:
            capping_signed_max = [
                capping_pressure_2to1_load_conserving(
                    ballast_pressure_pa=value,
                    bearing_width_m=bearing_geometry.width_m,
                    bearing_length_m=bearing_geometry.length_m,
                    ballast_thickness_m=ballast_thickness_m,
                )
                for value in result.ballast_pressure_max_pa
            ]
            capping_signed_min = [
                capping_pressure_2to1_load_conserving(
                    ballast_pressure_pa=value,
                    bearing_width_m=bearing_geometry.width_m,
                    bearing_length_m=bearing_geometry.length_m,
                    ballast_thickness_m=ballast_thickness_m,
                )
                for value in result.ballast_pressure_min_pa
            ]
        for i, position in enumerate(result.sleeper_positions_m):
            row = [
                i,
                _format_value(position),
                _format_value(result.sleeper_loads_max_n[i]),
                _format_value(result.sleeper_loads_min_n[i]),
                _format_value(result.ballast_pressure_max_pa[i]),
                _format_value(result.ballast_pressure_min_pa[i]),
            ]
            for depth in depths:
                row.append(_format_value(result.formation_stress_max_pa_by_depth[depth][i]))
                row.append(_format_value(result.formation_stress_min_pa_by_depth[depth][i]))
            row.append(
                (
                    _format_value(stress_results.q_ballast_comp_pa[i])
                    if stress_results is not None
                    and stress_results.q_ballast_comp_pa is not None
                    and i < len(stress_results.q_ballast_comp_pa)
                    else ""
                )
            )
            row.append(
                (
                    _format_value(stress_results.q_capping_comp_pa[i])
                    if stress_results is not None
                    and stress_results.q_capping_comp_pa is not None
                    and i < len(stress_results.q_capping_comp_pa)
                    else ""
                )
            )
            row.append(_format_value(capping_signed_max[i]) if i < len(capping_signed_max) else "")
            row.append(_format_value(capping_signed_min[i]) if i < len(capping_signed_min) else "")
            writer.writerow(row)


def write_transition_metrics_csv(path: str | Path, result: TransitionRunResult) -> None:
    """Write transition-zone metrics to CSV."""
    path = Path(path)
    metrics = result.metrics
    energy = result.energy_metrics
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mode",
                "profile_type",
                "template_name",
                "preset_name",
                "k1_mn_per_m2",
                "k2_mn_per_m2",
                "transition_length_m",
                "segment_length_m",
                "domain_length_m",
                "delta_w_s_m",
                "delta_w_s_position_m",
                "delta_w_1m_m",
                "delta_w_1m_position_m",
                "curvature_max_per_m",
                "curvature_max_position_m",
                "moment_max_nm",
                "moment_max_position_m",
                "energy_bending_j",
                "reaction_gradient_max_n_per_m2",
                "reaction_gradient_position_m",
                "sleeper_load_max_n",
                "sleeper_load_position_m",
                "k_units",
                "k_representation",
                "foundation_reaction_law",
                "transition_metrics_schema_version",
                "energy_rail_j",
                "energy_foundation_j",
                "energy_total_j",
                "energy_partition_eta",
                "u_total_max_j_per_m",
                "u_total_max_position_m",
                "du_dx_max_j_per_m2",
                "du_dx_max_position_m",
                "window_length_target_m",
                "window_energy_max_j",
                "window_energy_max_position_m",
                "window_avg_max_j_per_m",
                "window_avg_max_position_m",
                "window_effective_length_min_m",
                "window_effective_length_max_m",
                "is_envelope_upper_bound",
                "boundary_peak_flag",
                "boundary_gradient_peak_flag",
                "p_ref_n",
                "energy_total_over_p_ref_m",
                "u_total_max_over_p_ref",
                "energy_method",
                "energy_equations",
                "energy_scope",
            ]
        )
        writer.writerow(
            [
                _enum_value(result.mode),
                _enum_value(result.profile_type),
                result.template_name or "",
                result.preset_name or "",
                _format_value(n_per_m2_to_mn_per_m2(result.k1_n_per_m2)),
                _format_value(n_per_m2_to_mn_per_m2(result.k2_n_per_m2)) if result.k2_n_per_m2 is not None else "",
                _format_value(result.transition_length_m) if result.transition_length_m is not None else "",
                _format_value(result.segment_length_m) if result.segment_length_m is not None else "",
                _format_value(result.domain_length_m),
                _format_value(metrics.delta_w_s_m),
                _format_value(metrics.delta_w_s_position_m),
                _format_value(metrics.delta_w_1m_m),
                _format_value(metrics.delta_w_1m_position_m),
                _format_value(metrics.curvature_max_per_m),
                _format_value(metrics.curvature_max_position_m),
                _format_value(metrics.moment_max_nm),
                _format_value(metrics.moment_max_position_m),
                _format_value(metrics.energy_bending_j),
                _format_value(metrics.reaction_gradient_max_n_per_m2),
                _format_value(metrics.reaction_gradient_position_m),
                _format_value(metrics.sleeper_load_max_n),
                _format_value(metrics.sleeper_load_position_m),
                "MN/m^2",
                result.k_representation,
                result.foundation_reaction_law,
                str(result.transition_metrics_schema_version),
                _format_value(energy.energy_rail_j) if energy is not None else "",
                _format_value(energy.energy_foundation_j) if energy is not None else "",
                _format_value(energy.energy_total_j) if energy is not None else "",
                _format_value(energy.energy_partition_eta) if energy is not None else "",
                _format_value(energy.u_total_max_j_per_m) if energy is not None else "",
                _format_value(energy.u_total_max_position_m) if energy is not None else "",
                _format_value(energy.du_dx_max_j_per_m2) if energy is not None else "",
                _format_value(energy.du_dx_max_position_m) if energy is not None else "",
                _format_value(energy.window_length_target_m) if energy is not None else "",
                _format_value(energy.window_energy_max_j) if energy is not None else "",
                _format_value(energy.window_energy_max_position_m) if energy is not None else "",
                _format_value(energy.window_avg_max_j_per_m) if energy is not None else "",
                _format_value(energy.window_avg_max_position_m) if energy is not None else "",
                _format_value(energy.window_effective_length_min_m) if energy is not None else "",
                _format_value(energy.window_effective_length_max_m) if energy is not None else "",
                str(energy.is_envelope_upper_bound) if energy is not None else "",
                str(energy.boundary_peak_flag) if energy is not None else "",
                str(energy.boundary_gradient_peak_flag) if energy is not None else "",
                _format_value(energy.p_ref_n) if energy is not None and energy.p_ref_n is not None else "",
                _format_value(energy.energy_total_over_p_ref_m)
                if energy is not None and energy.energy_total_over_p_ref_m is not None
                else "",
                _format_value(energy.u_total_max_over_p_ref)
                if energy is not None and energy.u_total_max_over_p_ref is not None
                else "",
                energy.energy_method if energy is not None else "",
                energy.energy_equations if energy is not None else "",
                energy.energy_scope if energy is not None else "",
            ]
        )


def write_transition_series_csv(path: str | Path, result: TransitionRunResult) -> None:
    """Write transition-zone series data to CSV."""
    path = Path(path)
    series = result.series
    energy_series = result.energy_series
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if _enum_value(result.mode) == TransitionRunMode.ENVELOPE.value:
            writer.writerow(
                [
                    "x_m",
                    "k_mn_per_m2",
                    "deflection_max_m",
                    "deflection_min_m",
                    "moment_max_nm",
                    "moment_min_nm",
                    "moment_abs_max_nm",
                    "shear_max_n",
                    "shear_min_n",
                    "shear_abs_max_n",
                    "reaction_max_n_per_m",
                    "reaction_min_n_per_m",
                    "reaction_abs_max_n_per_m",
                    "u_rail_j_per_m",
                    "u_foundation_j_per_m",
                    "u_total_j_per_m",
                    "du_dx_j_per_m2",
                    "window_energy_j",
                    "window_avg_j_per_m",
                    "window_effective_length_m",
                ]
            )
            for i, x in enumerate(series.x_m):
                writer.writerow(
                    [
                        _format_value(x),
                        _format_value(n_per_m2_to_mn_per_m2(series.k_profile_n_per_m2[i])),
                        _format_value(series.deflection_max_m[i]) if series.deflection_max_m else "",
                        _format_value(series.deflection_min_m[i]) if series.deflection_min_m else "",
                        _format_value(series.moment_max_nm[i]) if series.moment_max_nm else "",
                        _format_value(series.moment_min_nm[i]) if series.moment_min_nm else "",
                        (
                            _format_value(max(abs(series.moment_max_nm[i]), abs(series.moment_min_nm[i])))
                            if series.moment_max_nm and series.moment_min_nm
                            else ""
                        ),
                        _format_value(series.shear_max_n[i]) if series.shear_max_n else "",
                        _format_value(series.shear_min_n[i]) if series.shear_min_n else "",
                        (
                            _format_value(max(abs(series.shear_max_n[i]), abs(series.shear_min_n[i])))
                            if series.shear_max_n and series.shear_min_n
                            else ""
                        ),
                        _format_value(series.reaction_max_n_per_m[i]) if series.reaction_max_n_per_m else "",
                        _format_value(series.reaction_min_n_per_m[i]) if series.reaction_min_n_per_m else "",
                        (
                            _format_value(
                                max(abs(series.reaction_max_n_per_m[i]), abs(series.reaction_min_n_per_m[i]))
                            )
                            if series.reaction_max_n_per_m and series.reaction_min_n_per_m
                            else ""
                        ),
                        _format_value(energy_series.u_rail_j_per_m[i]) if energy_series else "",
                        _format_value(energy_series.u_foundation_j_per_m[i]) if energy_series else "",
                        _format_value(energy_series.u_total_j_per_m[i]) if energy_series else "",
                        _format_value(energy_series.du_dx_j_per_m2[i]) if energy_series else "",
                        _format_value(energy_series.window_energy_j[i]) if energy_series else "",
                        _format_value(energy_series.window_avg_j_per_m[i]) if energy_series else "",
                        _format_value(energy_series.window_effective_length_m[i]) if energy_series else "",
                    ]
                )
        else:
            writer.writerow(
                [
                    "x_m",
                    "k_mn_per_m2",
                    "deflection_m",
                    "moment_nm",
                    "reaction_n_per_m",
                    "shear_n",
                    "u_rail_j_per_m",
                    "u_foundation_j_per_m",
                    "u_total_j_per_m",
                    "du_dx_j_per_m2",
                    "window_energy_j",
                    "window_avg_j_per_m",
                    "window_effective_length_m",
                ]
            )
            for i, x in enumerate(series.x_m):
                writer.writerow(
                    [
                        _format_value(x),
                        _format_value(n_per_m2_to_mn_per_m2(series.k_profile_n_per_m2[i])),
                        _format_value(series.deflection_m[i]) if series.deflection_m else "",
                        _format_value(series.moment_nm[i]) if series.moment_nm else "",
                        _format_value(series.reaction_n_per_m[i]) if series.reaction_n_per_m else "",
                        _format_value(series.shear_n[i]) if series.shear_n else "",
                        _format_value(energy_series.u_rail_j_per_m[i]) if energy_series else "",
                        _format_value(energy_series.u_foundation_j_per_m[i]) if energy_series else "",
                        _format_value(energy_series.u_total_j_per_m[i]) if energy_series else "",
                        _format_value(energy_series.du_dx_j_per_m2[i]) if energy_series else "",
                        _format_value(energy_series.window_energy_j[i]) if energy_series else "",
                        _format_value(energy_series.window_avg_j_per_m[i]) if energy_series else "",
                        _format_value(energy_series.window_effective_length_m[i]) if energy_series else "",
                    ]
                )

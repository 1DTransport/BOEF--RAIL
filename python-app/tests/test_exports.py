import csv
import json
from pathlib import Path

import pytest

from app.export_helpers import (
    compute_inputs_hash,
    write_analysis_csv_from_result,
    write_envelope_analysis_csv,
    write_envelope_sleeper_csv,
    write_export_metadata,
    write_sleeper_csv_from_result,
    write_transition_metrics_csv,
    write_transition_series_csv,
)
from core.analysis import AnalysisInputs as CoreAnalysisInputs, Extremum, compute_track_response
from core.envelope import EnvelopeResult, EnvelopeSummary
from core.exports import (
    AnalysisInputs,
    ReportInputs,
    SleeperInputs,
    build_analysis_points,
    build_sleeper_load_rows,
    export_pdf_report,
    write_analysis_csv,
    write_sleeper_load_csv,
)
from core.model import PointLoad
from core.analysis_engine import AnalysisConfig, AnalysisMode, FoundationProfileType, run_analysis
from core.stress_metrics import build_stress_results_from_envelope, get_bearing_geometry
from core.transition import (
    TRANSITION_ENERGY_EQUATIONS,
    TRANSITION_ENERGY_METHOD,
    TRANSITION_ENERGY_SCOPE,
    TransitionEnergyMetrics,
    TransitionEnergySeries,
    TransitionMetrics,
    TransitionProfileType,
    TransitionRunMode,
    TransitionRunResult,
    TransitionSeries,
)


@pytest.fixture()
def analysis_inputs() -> AnalysisInputs:
    return AnalysisInputs(
        x_positions_m=[-1.0, 0.0, 1.0],
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
    )


@pytest.fixture()
def sleeper_inputs() -> SleeperInputs:
    return SleeperInputs(
        sleeper_positions_m=[-0.6, 0.0, 0.6],
        tributary_length_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
    )


def test_gui_export_helpers_write_analysis_csv(tmp_path: Path) -> None:
    analysis_inputs = CoreAnalysisInputs(
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
    output = tmp_path / "analysis_export.csv"

    write_analysis_csv_from_result(output, analysis_inputs, result)

    rows = list(csv.reader(output.read_text(encoding="utf-8").splitlines()))
    assert rows[0][:5] == [
        "x_m",
        "deflection_m",
        "moment_nm",
        "shear_n",
        "reaction_n_per_m",
    ]
    assert rows[0][5:10] == [
        "slope_rad",
        "rotation_rad",
        "shear_angle_rad",
        "winkler_reaction_n_per_m",
        "pasternak_shear_reaction_n_per_m",
    ]
    assert rows[0][-2:] == ["sigma_top_fiber_pa", "sigma_bottom_fiber_pa"]
    assert float(rows[1][0]) == pytest.approx(result.x_m[0])
    assert float(rows[1][1]) == pytest.approx(result.deflection_m[0], abs=1e-10)
    assert any(abs(float(row[-2])) > 0.0 for row in rows[1:])
    assert any(abs(float(row[-1])) > 0.0 for row in rows[1:])


def test_gui_export_helpers_write_sleeper_csv(tmp_path: Path) -> None:
    analysis_inputs = CoreAnalysisInputs(
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
    output = tmp_path / "sleeper_export.csv"

    write_sleeper_csv_from_result(output, analysis_inputs, result)

    rows = list(csv.reader(output.read_text(encoding="utf-8").splitlines()))
    assert rows[0][:5] == [
        "sleeper_index",
        "position_m",
        "seat_load_n_per_rail",
        "total_sleeper_load_n",
        "ballast_pressure_pa",
    ]
    assert rows[0][-4:] == [
        "ballast_pressure_signed_pa",
        "ballast_pressure_comp_pa",
        "capping_pressure_signed_pa",
        "capping_pressure_comp_pa",
    ]
    assert float(rows[1][1]) == pytest.approx(result.sleeper_positions_m[0])
    assert any(float(row[-2]) > 0.0 for row in rows[1:])
    assert any(float(row[-1]) > 0.0 for row in rows[1:])


def test_write_analysis_csv(tmp_path: Path, analysis_inputs: AnalysisInputs) -> None:
    points = build_analysis_points(analysis_inputs)
    output = tmp_path / "analysis.csv"
    write_analysis_csv(output, points)

    rows = list(csv.reader(output.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == [
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
    ]
    assert len(rows) == 4
    assert rows[1][0] == "-1.000000"


def test_write_sleeper_load_csv(tmp_path: Path, sleeper_inputs: SleeperInputs) -> None:
    rows = build_sleeper_load_rows(sleeper_inputs, rail_count=2)
    output = tmp_path / "sleepers.csv"
    write_sleeper_load_csv(output, rows)

    parsed = list(csv.reader(output.read_text(encoding="utf-8").splitlines()))
    assert parsed[0] == [
        "sleeper_index",
        "position_m",
        "seat_load_n_per_rail",
        "total_sleeper_load_n",
        "ballast_pressure_pa",
    ]
    assert parsed[1][0] == "0"
    assert parsed[1][1] == "-0.600000"


def test_export_builders_accept_iterable_loads(
    analysis_inputs: AnalysisInputs,
    sleeper_inputs: SleeperInputs,
) -> None:
    analysis_with_iter = AnalysisInputs(
        x_positions_m=analysis_inputs.x_positions_m,
        loads=iter(analysis_inputs.loads),
        foundation_modulus_n_per_m2=analysis_inputs.foundation_modulus_n_per_m2,
        elastic_modulus_pa=analysis_inputs.elastic_modulus_pa,
        moment_inertia_m4=analysis_inputs.moment_inertia_m4,
    )
    sleeper_with_iter = SleeperInputs(
        sleeper_positions_m=sleeper_inputs.sleeper_positions_m,
        tributary_length_m=sleeper_inputs.tributary_length_m,
        sleeper_length_m=sleeper_inputs.sleeper_length_m,
        sleeper_width_m=sleeper_inputs.sleeper_width_m,
        loads=iter(sleeper_inputs.loads),
        foundation_modulus_n_per_m2=sleeper_inputs.foundation_modulus_n_per_m2,
        elastic_modulus_pa=sleeper_inputs.elastic_modulus_pa,
        moment_inertia_m4=sleeper_inputs.moment_inertia_m4,
    )

    points = build_analysis_points(analysis_with_iter)
    rows = build_sleeper_load_rows(sleeper_with_iter, rail_count=2)

    assert len(points) == len(analysis_inputs.x_positions_m)
    assert len(rows) == len(sleeper_inputs.sleeper_positions_m)


def test_export_pdf_report(tmp_path: Path, analysis_inputs: AnalysisInputs, sleeper_inputs: SleeperInputs) -> None:
    pytest.importorskip("matplotlib")
    report_inputs = ReportInputs(analysis=analysis_inputs, sleepers=sleeper_inputs)
    output = tmp_path / "report.pdf"

    summary = export_pdf_report(output, report_inputs)

    assert output.exists()
    assert output.stat().st_size > 0
    assert summary.max_deflection_m > 0
    assert summary.max_moment_nm > 0
    assert summary.max_sleeper_load_n > 0


def test_export_pdf_report_uses_supplied_analysis_result_arrays(
    tmp_path: Path,
    analysis_inputs: AnalysisInputs,
    sleeper_inputs: SleeperInputs,
) -> None:
    pytest.importorskip("matplotlib")
    config = AnalysisConfig(
        loads=[PointLoad(position_m=0.0, load_newtons=10_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=81,
        foundation_profile_type=FoundationProfileType.STEP,
        foundation_profile_k1_n_per_m2=40_000_000.0,
        foundation_profile_k2_n_per_m2=80_000_000.0,
        foundation_profile_x_start_m=0.0,
        foundation_profile_x_end_m=0.0,
    )
    result = run_analysis(config, mode=AnalysisMode.NUMERICAL)
    report_inputs = ReportInputs(
        analysis=analysis_inputs,
        sleepers=sleeper_inputs,
        analysis_result=result,
    )
    output = tmp_path / "numerical-report.pdf"

    summary = export_pdf_report(output, report_inputs)

    assert output.exists()
    assert summary.max_moment_nm == pytest.approx(max(abs(value) for value in result.moment_nm))
    assert summary.max_reaction_n_per_m == pytest.approx(
        max(abs(value) for value in result.reaction_n_per_m)
    )


def test_report_inputs_default_title(analysis_inputs: AnalysisInputs, sleeper_inputs: SleeperInputs) -> None:
    report_inputs = ReportInputs(analysis=analysis_inputs, sleepers=sleeper_inputs)

    assert report_inputs.title == "1DTransport.com: BOEF Calculation Tool Report"


def test_compute_inputs_hash_is_stable_for_equivalent_payloads() -> None:
    payload_a = {
        "solver_mode": "closed_form",
        "loads": [{"load_newtons": 10_000.0, "position_m": 0.0}],
        "units": "SI",
    }
    payload_b = {
        "units": "SI",
        "loads": [{"position_m": 0.0, "load_newtons": 10_000.0}],
        "solver_mode": "closed_form",
    }
    assert compute_inputs_hash(payload_a) == compute_inputs_hash(payload_b)


def test_write_export_metadata_creates_sidecar(tmp_path: Path) -> None:
    export_path = tmp_path / "analysis.csv"
    export_path.write_text("x_m,deflection_m\n0,0\n", encoding="utf-8")

    metadata_path = write_export_metadata(
        export_path,
        solver_mode="closed_form",
        inputs_payload={"loads": [{"position_m": 0.0, "load_newtons": 10_000.0}]},
        units="SI",
    )

    assert metadata_path.exists()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["units"] == "SI"
    assert payload["solver_mode"] == "closed_form"
    assert payload["inputs_payload"]["loads"][0]["position_m"] == pytest.approx(0.0)
    assert payload["source_export"] == "analysis.csv"
    assert isinstance(payload["inputs_hash"], str)
    assert len(payload["inputs_hash"]) == 64


def test_write_export_metadata_persists_as5100_load_source_payload(tmp_path: Path) -> None:
    export_path = tmp_path / "envelope_analysis.csv"
    export_path.write_text("x_m,deflection_m\n0,0\n", encoding="utf-8")

    metadata_path = write_export_metadata(
        export_path,
        solver_mode="closed_form",
        inputs_payload={
            "load_source": {
                "source_type": "as5100_fixed_rail",
                "arrangement": "governing_envelope_sweep",
                "model": "300LA",
                "group_count": 3,
                "group_spacing_m": 12.0,
            }
        },
        units="SI",
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    load_source = payload["inputs_payload"]["load_source"]
    assert load_source["source_type"] == "as5100_fixed_rail"
    assert load_source["arrangement"] == "governing_envelope_sweep"
    assert load_source["group_count"] == 3


def test_write_export_metadata_can_include_parameter_trace(tmp_path: Path) -> None:
    export_path = tmp_path / "dynamic_time_history.csv"
    export_path.write_text("t_s,deflection_m\n0,0\n", encoding="utf-8")

    metadata_path = write_export_metadata(
        export_path,
        solver_mode="steady_state",
        inputs_payload={"dynamic_mode": "steady_state"},
        units="SI",
        parameter_trace={
            "flexural_rigidity_nm2": 8.085e6,
            "foundation_modulus_n_per_m2": 1.0e7,
            "foundation_damping_n_s_per_m2": 2.0e3,
            "damping_ratio": 0.0408,
            "mass_kg_per_m": 60.0,
            "beta_per_m": 0.7459,
            "characteristic_length_m": 1.3407,
            "spatial_step_m": 0.05,
            "dynamic_amplification": 1.02,
        },
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    trace = payload["parameter_trace"]
    assert trace["flexural_rigidity_nm2"] == pytest.approx(8.085e6)
    assert trace["foundation_modulus_n_per_m2"] == pytest.approx(1.0e7)
    assert trace["beta_per_m"] == pytest.approx(0.7459)
    assert trace["dynamic_amplification"] == pytest.approx(1.02)


def _build_envelope_result_for_export() -> EnvelopeResult:
    x_values = [0.0, 1.0, 2.0]
    sleeper_positions = [0.0, 1.0, 2.0]
    summary = EnvelopeSummary(
        beta_per_m=0.8,
        zero_moment_distance_m=1.0,
        contraflexure_distance_m=1.8,
        max_deflection=Extremum(value=0.001, position_m=1.0),
        max_moment=Extremum(value=10_000.0, position_m=1.0),
        max_shear=Extremum(value=5_000.0, position_m=1.0),
        max_reaction=Extremum(value=3_000.0, position_m=1.0),
        max_sleeper_load=Extremum(value=80_000.0, position_m=1.0),
        max_ballast_pressure=Extremum(value=250_000.0, position_m=1.0),
        max_rail_base_stress_pa=12_000_000.0,
        max_formation_stress_by_depth_pa={0.3: 140_000.0},
    )
    return EnvelopeResult(
        x_m=x_values,
        deflection_max_m=[0.0, 0.001, 0.0],
        deflection_min_m=[0.0, -0.001, 0.0],
        moment_max_nm=[0.0, 8_000.0, 0.0],
        moment_min_nm=[0.0, -10_000.0, 0.0],
        shear_max_n=[0.0, 6_000.0, 0.0],
        shear_min_n=[0.0, -6_000.0, 0.0],
        reaction_max_n_per_m=[0.0, 4_000.0, 0.0],
        reaction_min_n_per_m=[0.0, -4_000.0, 0.0],
        sleeper_positions_m=sleeper_positions,
        sleeper_loads_max_n=[0.0, 100_000.0, 0.0],
        sleeper_loads_min_n=[0.0, -20_000.0, 0.0],
        ballast_pressure_max_pa=[0.0, 250_000.0, 0.0],
        ballast_pressure_min_pa=[0.0, -50_000.0, 0.0],
        formation_stress_max_pa_by_depth={0.3: [0.0, 140_000.0, 0.0]},
        formation_stress_min_pa_by_depth={0.3: [0.0, -40_000.0, 0.0]},
        summary=summary,
    )


def test_write_envelope_analysis_csv_appends_stress_columns(tmp_path: Path) -> None:
    result = _build_envelope_result_for_export()
    stress = build_stress_results_from_envelope(
        x_m=result.x_m,
        moment_max_nm=result.moment_max_nm,
        moment_min_nm=result.moment_min_nm,
        sleeper_positions_m=result.sleeper_positions_m,
        sleeper_loads_max_n=result.sleeper_loads_max_n,
        sleeper_loads_min_n=result.sleeper_loads_min_n,
        section_modulus_top_m3=3.2e-5,
        section_modulus_bottom_m3=3.2e-5,
        bearing_geometry=get_bearing_geometry(sleeper_width_m=0.25, sleeper_length_m=2.6),
        ballast_thickness_m=0.3,
    )
    output = tmp_path / "envelope_analysis.csv"

    write_envelope_analysis_csv(output, result, stress_results=stress)

    rows = list(csv.reader(output.read_text(encoding="utf-8").splitlines()))
    header = rows[0]
    assert header[:9] == [
        "x_m",
        "deflection_max_m",
        "deflection_min_m",
        "moment_max_nm",
        "moment_min_nm",
        "shear_max_n",
        "shear_min_n",
        "reaction_max_n_per_m",
        "reaction_min_n_per_m",
    ]
    assert "moment_abs_max_nm" in header
    assert "shear_abs_max_n" in header
    assert header[-2:] == ["sigma_top_fiber_ub_pa", "sigma_bottom_fiber_ub_pa"]
    assert float(rows[2][header.index("moment_abs_max_nm")]) == pytest.approx(10_000.0)
    assert float(rows[2][header.index("shear_abs_max_n")]) == pytest.approx(6_000.0)
    assert float(rows[2][-2]) > 0.0
    assert float(rows[2][-1]) > 0.0


def test_write_envelope_sleeper_csv_appends_stress_columns(tmp_path: Path) -> None:
    result = _build_envelope_result_for_export()
    geometry = get_bearing_geometry(sleeper_width_m=0.25, sleeper_length_m=2.6)
    stress = build_stress_results_from_envelope(
        x_m=result.x_m,
        moment_max_nm=result.moment_max_nm,
        moment_min_nm=result.moment_min_nm,
        sleeper_positions_m=result.sleeper_positions_m,
        sleeper_loads_max_n=result.sleeper_loads_max_n,
        sleeper_loads_min_n=result.sleeper_loads_min_n,
        section_modulus_top_m3=3.2e-5,
        section_modulus_bottom_m3=3.2e-5,
        bearing_geometry=geometry,
        ballast_thickness_m=0.3,
    )
    output = tmp_path / "envelope_sleeper.csv"

    write_envelope_sleeper_csv(
        output,
        result,
        stress_results=stress,
        bearing_geometry=geometry,
        ballast_thickness_m=0.3,
    )

    rows = list(csv.reader(output.read_text(encoding="utf-8").splitlines()))
    header = rows[0]
    assert header[:6] == [
        "sleeper_index",
        "position_m",
        "load_max_n",
        "load_min_n",
        "ballast_pressure_max_pa",
        "ballast_pressure_min_pa",
    ]
    assert header[-4:] == [
        "ballast_pressure_max_comp_pa",
        "capping_pressure_max_comp_pa",
        "capping_pressure_signed_max_pa",
        "capping_pressure_signed_min_pa",
    ]
    assert float(rows[2][-4]) > 0.0
    assert float(rows[2][-3]) > 0.0


def _build_transition_result_for_export() -> TransitionRunResult:
    return TransitionRunResult(
        mode=TransitionRunMode.SINGLE,
        profile_type=TransitionProfileType.UNIFORM,
        k1_n_per_m2=40_000_000.0,
        k2_n_per_m2=None,
        transition_length_m=None,
        segment_length_m=None,
        domain_length_m=2.0,
        metrics=TransitionMetrics(
            delta_w_s_m=0.001,
            delta_w_s_position_m=0.0,
            delta_w_1m_m=0.0012,
            delta_w_1m_position_m=0.5,
            curvature_max_per_m=0.002,
            curvature_max_position_m=0.5,
            moment_max_nm=12_000.0,
            moment_max_position_m=0.5,
            energy_bending_j=10.0,
            reaction_gradient_max_n_per_m2=2_000.0,
            reaction_gradient_position_m=0.5,
            sleeper_load_max_n=100_000.0,
            sleeper_load_position_m=0.5,
        ),
        series=TransitionSeries(
            x_m=[0.0, 1.0, 2.0],
            k_profile_n_per_m2=[40_000_000.0, 40_000_000.0, 40_000_000.0],
            deflection_m=[0.0, 0.001, 0.0],
            moment_nm=[0.0, 12_000.0, 0.0],
            shear_n=[0.0, 6_000.0, 0.0],
            reaction_n_per_m=[0.0, 8_000.0, 0.0],
        ),
        energy_metrics=TransitionEnergyMetrics(
            energy_rail_j=4.0,
            energy_foundation_j=2.0,
            energy_total_j=6.0,
            energy_partition_eta=1.0 / 3.0,
            u_total_max_j_per_m=3.0,
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
            is_envelope_upper_bound=False,
            boundary_peak_flag=False,
            boundary_gradient_peak_flag=True,
            p_ref_n=100_000.0,
            energy_total_over_p_ref_m=0.00006,
            u_total_max_over_p_ref=0.00003,
        ),
        energy_series=TransitionEnergySeries(
            u_rail_j_per_m=[0.0, 2.0, 0.0],
            u_foundation_j_per_m=[0.0, 1.0, 0.0],
            u_total_j_per_m=[0.0, 3.0, 0.0],
            du_dx_j_per_m2=[1.0, 0.0, 1.0],
            window_energy_j=[0.5, 1.5, 0.5],
            window_avg_j_per_m=[1.0, 2.5, 1.0],
            window_effective_length_m=[0.3, 0.6, 0.3],
        ),
    )


def test_write_transition_metrics_csv_appends_schema_v2_energy_columns(tmp_path: Path) -> None:
    output = tmp_path / "transition_metrics.csv"
    result = _build_transition_result_for_export()

    write_transition_metrics_csv(output, result)

    rows = list(csv.reader(output.read_text(encoding="utf-8").splitlines()))
    header = rows[0]
    values = rows[1]
    assert header[:22] == [
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
    ]
    assert "transition_metrics_schema_version" in header
    assert "energy_total_j" in header
    assert "u_total_max_j_per_m" in header
    assert "energy_method" in header
    assert "energy_equations" in header
    assert "energy_scope" in header
    schema_idx = header.index("transition_metrics_schema_version")
    assert values[schema_idx] == "2"
    assert float(values[header.index("k1_mn_per_m2")]) == pytest.approx(40.0)
    assert values[header.index("k2_mn_per_m2")] == ""
    assert values[header.index("energy_method")] == TRANSITION_ENERGY_METHOD
    assert values[header.index("energy_equations")] == TRANSITION_ENERGY_EQUATIONS
    assert values[header.index("energy_scope")] == TRANSITION_ENERGY_SCOPE


def test_write_transition_metrics_csv_accepts_string_enum_values(tmp_path: Path) -> None:
    output = tmp_path / "transition_metrics_string_mode.csv"
    result = _build_transition_result_for_export()
    result = TransitionRunResult(
        mode=TransitionRunMode.SINGLE.value,
        profile_type=TransitionProfileType.UNIFORM.value,
        k1_n_per_m2=result.k1_n_per_m2,
        k2_n_per_m2=result.k2_n_per_m2,
        transition_length_m=result.transition_length_m,
        segment_length_m=result.segment_length_m,
        domain_length_m=result.domain_length_m,
        metrics=result.metrics,
        series=result.series,
        template_name=result.template_name,
        preset_name=result.preset_name,
        k_units=result.k_units,
        k_representation=result.k_representation,
        foundation_reaction_law=result.foundation_reaction_law,
        transition_metrics_schema_version=result.transition_metrics_schema_version,
        energy_metrics=result.energy_metrics,
        energy_series=result.energy_series,
    )

    write_transition_metrics_csv(output, result)

    rows = list(csv.reader(output.read_text(encoding="utf-8").splitlines()))
    assert rows[1][0] == TransitionRunMode.SINGLE.value
    assert rows[1][1] == TransitionProfileType.UNIFORM.value


def test_write_transition_series_csv_appends_energy_columns(tmp_path: Path) -> None:
    output = tmp_path / "transition_series.csv"
    result = _build_transition_result_for_export()

    write_transition_series_csv(output, result)

    rows = list(csv.reader(output.read_text(encoding="utf-8").splitlines()))
    header = rows[0]
    assert header[:5] == [
        "x_m",
        "k_mn_per_m2",
        "deflection_m",
        "moment_nm",
        "reaction_n_per_m",
    ]
    assert "shear_n" in header
    assert float(rows[2][header.index("k_mn_per_m2")]) == pytest.approx(40.0)
    assert float(rows[2][header.index("shear_n")]) == pytest.approx(6_000.0)
    assert header[-7:] == [
        "u_rail_j_per_m",
        "u_foundation_j_per_m",
        "u_total_j_per_m",
        "du_dx_j_per_m2",
        "window_energy_j",
        "window_avg_j_per_m",
        "window_effective_length_m",
    ]
    numeric_tail = [float(value) for value in rows[2][-7:]]
    assert numeric_tail == pytest.approx([2.0, 1.0, 3.0, 0.0, 1.5, 2.5, 0.6])


def test_write_transition_metrics_csv_keeps_energy_fields_blank_when_missing(tmp_path: Path) -> None:
    output = tmp_path / "transition_metrics_no_energy.csv"
    result = _build_transition_result_for_export()
    result = TransitionRunResult(
        mode=result.mode,
        profile_type=result.profile_type,
        k1_n_per_m2=result.k1_n_per_m2,
        k2_n_per_m2=result.k2_n_per_m2,
        transition_length_m=result.transition_length_m,
        segment_length_m=result.segment_length_m,
        domain_length_m=result.domain_length_m,
        metrics=result.metrics,
        series=result.series,
        template_name=result.template_name,
        preset_name=result.preset_name,
        k_units=result.k_units,
        k_representation="model_dependent_per_unit_length",
        foundation_reaction_law="model-dependent (energy metrics disabled for non-Winkler)",
        transition_metrics_schema_version=2,
        energy_metrics=None,
        energy_series=None,
    )

    write_transition_metrics_csv(output, result)

    rows = list(csv.reader(output.read_text(encoding="utf-8").splitlines()))
    header = rows[0]
    values = rows[1]
    assert values[header.index("k_representation")] == "model_dependent_per_unit_length"
    assert "non-Winkler" in values[header.index("foundation_reaction_law")]
    assert values[header.index("energy_total_j")] == ""

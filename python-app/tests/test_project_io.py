from pathlib import Path
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db import crud
from db.models import Base
from db.project_io import ProjectImportError, export_project, import_project
from db.seed import seed_database


def _make_session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def test_export_import_project_round_trip(tmp_path: Path) -> None:
    with _make_session() as session:
        seed_database(session)
        rail = crud.list_rails(session)[0]
        sleeper = crud.list_sleepers(session)[0]
        pad = crud.list_pads(session)[0]
        support = crud.list_support_profiles(session)[0]
        load_case = crud.list_load_cases(session)[0]
        project = crud.create_project(
            session,
            name="Export Project",
            description="Export demo",
            vehicle_type="heavy_metro",
            vehicle_subtype="Urban B",
            design_speed_kmh=85.0,
            design_wheel_radius_mm=440.0,
        )
        config = crud.create_track_config(
            session,
            name="Export Config",
            project_id=project.id,
            rail_id=rail.id,
            sleeper_id=sleeper.id,
            pad_id=pad.id,
            support_profile_id=support.id,
            sleeper_spacing_m=0.6,
            gauge_m=1.435,
        )
        crud.create_result(
            session,
            project_id=project.id,
            track_config_id=config.id,
            load_case_id=load_case.id,
            max_deflection_m=0.002,
            max_moment_nm=12_000.0,
        )
        crud.create_design_alternative(
            session,
            project_id=project.id,
            track_config_id=config.id,
            load_case_id=load_case.id,
            name="Export Alternative",
            description="Saved from sensitivity",
            source_type="sensitivity",
            analysis_type="static",
            changed_parameters={"support_stiffness": 1.2},
            input_snapshot={"baseline": "Export Config"},
            metrics={"max_deflection_m": 0.0018, "max_moment_nm": 10_000.0},
            status="ok",
            score=0.91,
        )

        export_path = export_project(session, project_id=project.id, path=tmp_path / "project.json")
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 2
        assert payload["design_alternatives"][0]["name"] == "Export Alternative"

    with _make_session() as session:
        seed_database(session)
        imported = import_project(session, path=export_path)
        assert imported.name == "Export Project"
        assert imported.vehicle_type == "heavy_metro"
        assert imported.vehicle_subtype == "Urban B"
        assert imported.design_speed_kmh == pytest.approx(85.0)
        assert imported.design_wheel_radius_mm == pytest.approx(440.0)
        assert len(imported.track_configs) == 1
        assert len(imported.results) == 1
        assert len(imported.design_alternatives) == 1
        assert imported.design_alternatives[0].score == pytest.approx(0.91)


def test_import_project_rejects_missing_material(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "project": {"name": "Missing Material", "description": None, "created_at": "2024-01-01T00:00:00"},
        "track_configs": [
            {
                "name": "Config",
                "rail_name": "No Rail",
                "sleeper_name": "No Sleeper",
                "pad_name": "No Pad",
                "support_profile_name": "No Support",
                "sleeper_spacing_m": 0.6,
                "gauge_m": 1.435,
            }
        ],
        "load_cases": [
            {"name": "Case", "load_newtons": 1000.0, "description": None}
        ],
        "results": [],
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with _make_session() as session:
        seed_database(session)
        with pytest.raises(ProjectImportError, match="Rail 'No Rail' not found"):
            import_project(session, path=path)


def test_import_project_accepts_old_schema_without_alternatives(tmp_path: Path) -> None:
    with _make_session() as session:
        seed_database(session)
        rail = crud.list_rails(session)[0]
        sleeper = crud.list_sleepers(session)[0]
        pad = crud.list_pads(session)[0]
        support = crud.list_support_profiles(session)[0]

    payload = {
        "schema_version": 1,
        "project": {"name": "Old Project", "description": None, "created_at": "2024-01-01T00:00:00"},
        "track_configs": [
            {
                "name": "Old Config",
                "rail_name": rail.name,
                "sleeper_name": sleeper.name,
                "pad_name": pad.name,
                "support_profile_name": support.name,
                "sleeper_spacing_m": 0.6,
                "gauge_m": 1.435,
            }
        ],
        "load_cases": [],
        "results": [],
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with _make_session() as session:
        seed_database(session)
        imported = import_project(session, path=path)
        assert imported.name == "Old Project"
        assert imported.design_alternatives == []

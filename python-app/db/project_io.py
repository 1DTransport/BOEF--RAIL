"""Project import/export helpers for track configs, load cases, and result metadata."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from db import crud
from db.models import (
    DesignAlternative,
    LoadCase,
    Pad,
    Project,
    Rail,
    Result,
    Sleeper,
    SupportProfile,
    TrackConfig,
)

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, 2}


class ProjectImportError(ValueError):
    """Raised when a project payload cannot be imported safely."""


def export_project(session: Session, *, project_id: int, path: str | Path) -> Path:
    """Export a project and its related data to JSON."""
    project = session.get(Project, project_id)
    if project is None:
        raise ProjectImportError(f"Project {project_id} not found")

    track_configs = list(
        session.scalars(select(TrackConfig).where(TrackConfig.project_id == project_id))
    )
    results = list(
        session.scalars(select(Result).where(Result.project_id == project_id))
    )
    alternatives = list(
        session.scalars(select(DesignAlternative).where(DesignAlternative.project_id == project_id))
    )
    load_case_ids = {result.load_case_id for result in results}
    load_case_ids.update(
        alternative.load_case_id
        for alternative in alternatives
        if alternative.load_case_id is not None
    )
    load_cases = (
        list(session.scalars(select(LoadCase).where(LoadCase.id.in_(load_case_ids))))
        if load_case_ids
        else []
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "project": {
            "name": project.name,
            "description": project.description,
            "vehicle_type": project.vehicle_type,
            "vehicle_subtype": project.vehicle_subtype,
            "design_speed_kmh": project.design_speed_kmh,
            "design_wheel_radius_mm": project.design_wheel_radius_mm,
            "created_at": project.created_at.isoformat(),
        },
        "track_configs": [
            {
                "name": config.name,
                "rail_name": config.rail.name,
                "sleeper_name": config.sleeper.name,
                "pad_name": config.pad.name,
                "support_profile_name": config.support_profile.name,
                "sleeper_spacing_m": config.sleeper_spacing_m,
                "gauge_m": config.gauge_m,
            }
            for config in track_configs
        ],
        "load_cases": [
            {
                "name": load_case.name,
                "load_newtons": load_case.load_newtons,
                "description": load_case.description,
            }
            for load_case in load_cases
        ],
        "results": [
            {
                "track_config_name": result.track_config.name,
                "load_case_name": result.load_case.name,
                "max_deflection_m": result.max_deflection_m,
                "max_moment_nm": result.max_moment_nm,
                "created_at": result.created_at.isoformat(),
            }
            for result in results
        ],
        "design_alternatives": [
            {
                "name": alternative.name,
                "description": alternative.description,
                "track_config_name": alternative.track_config.name,
                "load_case_name": alternative.load_case.name if alternative.load_case else None,
                "source_type": alternative.source_type,
                "analysis_type": alternative.analysis_type,
                "changed_parameters": json.loads(alternative.changed_parameters_json),
                "input_snapshot": json.loads(alternative.input_snapshot_json),
                "metrics": json.loads(alternative.metrics_json),
                "status": alternative.status,
                "score": alternative.score,
                "created_at": alternative.created_at.isoformat(),
            }
            for alternative in alternatives
        ],
    }

    export_path = Path(path)
    export_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return export_path


def import_project(
    session: Session,
    *,
    path: str | Path,
    project_name: str | None = None,
) -> Project:
    """Import a project and its related data from JSON."""
    payload = _load_payload(path)
    project_payload = _require_mapping(payload, "project")
    name = project_name or _require_string(project_payload, "name")
    description = project_payload.get("description")
    vehicle_type = project_payload.get("vehicle_type")
    vehicle_subtype = project_payload.get("vehicle_subtype")
    design_speed_kmh = project_payload.get("design_speed_kmh")
    design_wheel_radius_mm = project_payload.get("design_wheel_radius_mm")
    if description is not None and not isinstance(description, str):
        raise ProjectImportError("project.description must be a string or null")

    crud.validate_project_data(name)
    if session.scalar(select(Project).where(Project.name == name)) is not None:
        raise ProjectImportError(f"Project '{name}' already exists")

    track_payloads = _require_list(payload, "track_configs")
    load_case_payloads = _require_list(payload, "load_cases")
    result_payloads = _require_list(payload, "results")
    alternative_payloads = _optional_list(payload, "design_alternatives")

    context = session.begin_nested() if session.in_transaction() else session.begin()
    with context:
        project = Project(
            name=name.strip(),
            description=description,
            vehicle_type=vehicle_type,
            vehicle_subtype=vehicle_subtype,
            design_speed_kmh=design_speed_kmh,
            design_wheel_radius_mm=design_wheel_radius_mm,
        )
        session.add(project)
        session.flush()

        load_cases = _import_load_cases(session, load_case_payloads)
        track_configs = _import_track_configs(session, project.id, track_payloads)
        _import_results(session, project.id, result_payloads, track_configs, load_cases)
        _import_design_alternatives(
            session,
            project.id,
            alternative_payloads,
            track_configs,
            load_cases,
        )

    return project


def _load_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProjectImportError("Payload must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ProjectImportError("Unsupported project schema version")
    return payload


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ProjectImportError(f"{key} must be an object")
    return value


def _require_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ProjectImportError(f"{key} must be a list")
    for item in value:
        if not isinstance(item, dict):
            raise ProjectImportError(f"{key} entries must be objects")
    return value


def _optional_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    if key not in payload:
        return []
    return _require_list(payload, key)


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProjectImportError(f"{key} is required")
    return value


def _require_non_negative(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        raise ProjectImportError(f"{key} must be a number") from None
    if value_float < 0:
        raise ProjectImportError(f"{key} must be >= 0")
    return value_float


def _require_positive(payload: dict[str, Any], key: str) -> float:
    value = _require_non_negative(payload, key)
    if value <= 0:
        raise ProjectImportError(f"{key} must be > 0")
    return value


def _import_load_cases(
    session: Session,
    payloads: list[dict[str, Any]],
) -> dict[str, LoadCase]:
    load_cases: dict[str, LoadCase] = {}
    for payload in payloads:
        name = _require_string(payload, "name")
        load_newtons = _require_non_negative(payload, "load_newtons")
        description = payload.get("description")
        if description is not None and not isinstance(description, str):
            raise ProjectImportError("load_cases.description must be a string or null")
        crud.validate_load_case_data(name, load_newtons)

        existing = session.scalar(select(LoadCase).where(LoadCase.name == name))
        if existing is not None:
            if (
                existing.load_newtons != load_newtons
                or (existing.description or None) != (description or None)
            ):
                raise ProjectImportError(f"Load case '{name}' conflicts with existing data")
            load_cases[name] = existing
            continue

        new_case = LoadCase(name=name.strip(), load_newtons=load_newtons, description=description)
        session.add(new_case)
        session.flush()
        load_cases[name] = new_case
    return load_cases


def _import_track_configs(
    session: Session,
    project_id: int,
    payloads: list[dict[str, Any]],
) -> dict[str, TrackConfig]:
    configs: dict[str, TrackConfig] = {}
    for payload in payloads:
        name = _require_string(payload, "name")
        if name in configs:
            raise ProjectImportError(f"Duplicate track config name '{name}'")
        sleeper_spacing_m = _require_positive(payload, "sleeper_spacing_m")
        gauge_m = _require_positive(payload, "gauge_m")
        rail = _get_by_name(session, Rail, payload, "rail_name")
        sleeper = _get_by_name(session, Sleeper, payload, "sleeper_name")
        pad = _get_by_name(session, Pad, payload, "pad_name")
        support = _get_by_name(session, SupportProfile, payload, "support_profile_name")

        config = TrackConfig(
            name=name.strip(),
            project_id=project_id,
            rail_id=rail.id,
            sleeper_id=sleeper.id,
            pad_id=pad.id,
            support_profile_id=support.id,
            sleeper_spacing_m=sleeper_spacing_m,
            gauge_m=gauge_m,
        )
        session.add(config)
        session.flush()
        configs[name] = config
    return configs


def _import_results(
    session: Session,
    project_id: int,
    payloads: list[dict[str, Any]],
    track_configs: dict[str, TrackConfig],
    load_cases: dict[str, LoadCase],
) -> None:
    for payload in payloads:
        track_name = _require_string(payload, "track_config_name")
        load_case_name = _require_string(payload, "load_case_name")
        if track_name not in track_configs:
            raise ProjectImportError(f"Unknown track_config_name '{track_name}'")
        if load_case_name not in load_cases:
            raise ProjectImportError(f"Unknown load_case_name '{load_case_name}'")
        max_deflection_m = _require_non_negative(payload, "max_deflection_m")
        max_moment_nm = _require_non_negative(payload, "max_moment_nm")
        created_at = _parse_datetime(payload.get("created_at"))

        result = Result(
            project_id=project_id,
            track_config_id=track_configs[track_name].id,
            load_case_id=load_cases[load_case_name].id,
            max_deflection_m=max_deflection_m,
            max_moment_nm=max_moment_nm,
            created_at=created_at,
        )
        session.add(result)


def _import_design_alternatives(
    session: Session,
    project_id: int,
    payloads: list[dict[str, Any]],
    track_configs: dict[str, TrackConfig],
    load_cases: dict[str, LoadCase],
) -> None:
    for payload in payloads:
        name = _require_string(payload, "name")
        track_name = _require_string(payload, "track_config_name")
        if track_name not in track_configs:
            raise ProjectImportError(f"Unknown track_config_name '{track_name}'")
        load_case_name = payload.get("load_case_name")
        if load_case_name is not None and not isinstance(load_case_name, str):
            raise ProjectImportError("load_case_name must be a string or null")
        load_case = None
        if load_case_name:
            if load_case_name not in load_cases:
                raise ProjectImportError(f"Unknown load_case_name '{load_case_name}'")
            load_case = load_cases[load_case_name]
        description = payload.get("description")
        if description is not None and not isinstance(description, str):
            raise ProjectImportError("design_alternatives.description must be a string or null")
        source_type = _require_string(payload, "source_type")
        analysis_type = _require_string(payload, "analysis_type")
        status = _require_string(payload, "status")
        changed_parameters = _require_mapping(payload, "changed_parameters")
        input_snapshot = _require_mapping(payload, "input_snapshot")
        metrics = _require_mapping(payload, "metrics")
        score = payload.get("score")
        if score is not None:
            try:
                score = float(score)
            except (TypeError, ValueError):
                raise ProjectImportError("score must be a number or null") from None
        created_at = _parse_datetime(payload.get("created_at"))

        alternative = DesignAlternative(
            project_id=project_id,
            track_config_id=track_configs[track_name].id,
            load_case_id=load_case.id if load_case is not None else None,
            name=name.strip(),
            description=description,
            source_type=source_type,
            analysis_type=analysis_type,
            changed_parameters_json=json.dumps(changed_parameters, sort_keys=True),
            input_snapshot_json=json.dumps(input_snapshot, sort_keys=True),
            metrics_json=json.dumps(metrics, sort_keys=True),
            status=status,
            score=score,
            created_at=created_at,
        )
        crud.validate_design_alternative_data(
            project_id=alternative.project_id,
            track_config_id=alternative.track_config_id,
            load_case_id=alternative.load_case_id,
            name=alternative.name,
            source_type=alternative.source_type,
            analysis_type=alternative.analysis_type,
            changed_parameters_json=alternative.changed_parameters_json,
            input_snapshot_json=alternative.input_snapshot_json,
            metrics_json=alternative.metrics_json,
            status=alternative.status,
            score=alternative.score,
        )
        session.add(alternative)


def _parse_datetime(value: Any) -> datetime:
    if value is None:
        return datetime.utcnow()
    if not isinstance(value, str):
        raise ProjectImportError("created_at must be an ISO-8601 string")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ProjectImportError("created_at must be ISO-8601 format") from exc


def _get_by_name(
    session: Session,
    model: type[Rail | Sleeper | Pad | SupportProfile],
    payload: dict[str, Any],
    key: str,
):
    name = _require_string(payload, key)
    record = session.scalar(select(model).where(model.name == name))
    if record is None:
        raise ProjectImportError(f"{model.__name__} '{name}' not found")
    return record

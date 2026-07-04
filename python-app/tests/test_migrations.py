from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from db.cli import initialize_database


def _alembic_config(db_path: Path) -> Config:
    base_dir = Path(__file__).resolve().parents[1]
    config = Config(str(base_dir / "alembic.ini"))
    config.set_main_option("script_location", str(base_dir / "db" / "migrations"))
    config.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{db_path}")
    return config


def test_support_profile_modulus_migration_preserves_values(tmp_path: Path) -> None:
    db_path = tmp_path / "migration.sqlite"
    config = _alembic_config(db_path)

    command.upgrade(config, "0001_create_boef_tables")
    engine = create_engine(f"sqlite+pysqlite:///{db_path}", future=True)

    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO support_profiles (name, stiffness_newtons_per_meter) "
                "VALUES (:name, :value)"
            ),
            {"name": "Legacy Ballast", "value": 42_000_000.0},
        )

    command.upgrade(config, "head")

    with engine.connect() as connection:
        value = connection.execute(
            text(
                "SELECT foundation_modulus_n_per_m2 "
                "FROM support_profiles WHERE name = :name"
            ),
            {"name": "Legacy Ballast"},
        ).scalar_one()

    assert value == pytest.approx(42_000_000.0)


def test_cli_initialization_upgrades_legacy_support_profile(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite"
    config = _alembic_config(db_path)

    command.upgrade(config, "0001_create_boef_tables")
    engine = create_engine(f"sqlite+pysqlite:///{db_path}", future=True)

    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO support_profiles (name, stiffness_newtons_per_meter) "
                "VALUES (:name, :value)"
            ),
            {"name": "Legacy Ballast", "value": 58_000_000.0},
        )

    initialize_database(database_url=f"sqlite+pysqlite:///{db_path}", seed=False)

    with engine.connect() as connection:
        value = connection.execute(
            text(
                "SELECT foundation_modulus_n_per_m2 "
                "FROM support_profiles WHERE name = :name"
            ),
            {"name": "Legacy Ballast"},
        ).scalar_one()

    assert value == pytest.approx(58_000_000.0)


def test_design_alternatives_migration_creates_table(tmp_path: Path) -> None:
    db_path = tmp_path / "alternatives.sqlite"
    config = _alembic_config(db_path)

    command.upgrade(config, "head")
    engine = create_engine(f"sqlite+pysqlite:///{db_path}", future=True)

    with engine.connect() as connection:
        names = {
            row[0]
            for row in connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table'")
            )
        }
        columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(design_alternatives)"))
        }

    assert "design_alternatives" in names
    assert {
        "project_id",
        "track_config_id",
        "load_case_id",
        "source_type",
        "analysis_type",
        "changed_parameters_json",
        "input_snapshot_json",
        "metrics_json",
        "status",
        "score",
    }.issubset(columns)


def test_support_profile_default_names_use_mn_per_m2(tmp_path: Path) -> None:
    db_path = tmp_path / "support_profile_names.sqlite"
    config = _alembic_config(db_path)

    command.upgrade(config, "0009_add_design_alternatives")
    engine = create_engine(f"sqlite+pysqlite:///{db_path}", future=True)

    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO support_profiles (name, foundation_modulus_n_per_m2) "
                "VALUES (:name, :value)"
            ),
            [
                {"name": "Ballast 30 MN/m", "value": 30_000_000.0},
                {"name": "Ballast 50 MN/m", "value": 50_000_000.0},
                {"name": "Ballast 80 MN/m", "value": 80_000_000.0},
            ],
        )

    command.upgrade(config, "head")

    with engine.connect() as connection:
        names = {
            row[0]
            for row in connection.execute(
                text("SELECT name FROM support_profiles ORDER BY name")
            )
        }

    assert {"Ballast 30 MN/m²", "Ballast 50 MN/m²", "Ballast 80 MN/m²"}.issubset(names)
    assert not {"Ballast 30 MN/m", "Ballast 50 MN/m", "Ballast 80 MN/m"}.intersection(names)

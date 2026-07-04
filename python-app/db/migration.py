"""Helpers for applying Alembic migrations programmatically."""

from __future__ import annotations

from pathlib import Path
import sys

from alembic import command
from alembic.config import Config


def _bundle_base_dir() -> Path:
    """Return the project/resource root in source and PyInstaller builds."""
    source_base = Path(__file__).resolve().parents[1]
    candidates = [source_base]

    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).resolve()
        candidates.extend(
            [
                Path(getattr(sys, "_MEIPASS", source_base)),
                executable.parents[1] / "Resources",
                executable.parents[1] / "Frameworks",
            ]
        )

    for candidate in candidates:
        if (candidate / "db" / "migrations").exists():
            return candidate

    return source_base


def run_migrations(database_url: str) -> None:
    """Apply Alembic migrations up to the latest revision."""
    base_dir = _bundle_base_dir()
    alembic_ini = base_dir / "alembic.ini"
    config = Config(str(alembic_ini)) if alembic_ini.exists() else Config()
    config.set_main_option("script_location", str(base_dir / "db" / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")

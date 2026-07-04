"""CLI utilities for managing the BOEF database."""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.migration import run_migrations
from db.seed import seed_database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BOEF database CLI")
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        required=True,
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed the database with baseline materials and load cases",
    )
    return parser


def initialize_database(*, database_url: str, seed: bool) -> None:
    run_migrations(database_url)
    engine = create_engine(database_url, future=True)
    if seed:
        with Session(engine) as session:
            seed_database(session)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    database_url = f"sqlite:///{args.sqlite_path}"
    initialize_database(database_url=database_url, seed=args.seed)
    print(f"Initialized database at {args.sqlite_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

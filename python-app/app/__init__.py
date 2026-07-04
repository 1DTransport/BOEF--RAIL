"""GUI package for the BOEF desktop application."""

from __future__ import annotations


def run() -> int:
    """Start the BOEF GUI application."""
    from app.main import run as gui_run

    return gui_run()


__all__ = ["run"]

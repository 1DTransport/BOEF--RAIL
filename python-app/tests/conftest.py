"""Test configuration for headless GUI reliability."""

from __future__ import annotations

import os


# Keep Qt GUI tests deterministic in headless environments.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "gui: Qt/PySide GUI test.")


def pytest_collection_modifyitems(items) -> None:
    gui_files = {
        "tests/test_custom_chart.py",
        "tests/test_main_window.py",
        "tests/test_material_dialog.py",
    }
    for item in items:
        if any(item.nodeid.startswith(path) for path in gui_files):
            item.add_marker("gui")


def pytest_ignore_collect(collection_path, config) -> bool:  # pragma: no cover - pytest collection hook
    if os.environ.get("BOEF_ENABLE_GUI_TESTS", "").lower() in {"1", "true", "yes"}:
        return False
    path = str(collection_path)
    return (
        path.endswith("tests/test_custom_chart.py")
        or path.endswith("tests/test_main_window.py")
        or path.endswith("tests/test_material_dialog.py")
    )

"""Project-local Python startup customizations.

This file is imported automatically by Python when ``python-app`` is on
``sys.path``.  It only adjusts Qt during pytest runs so GUI tests do not try to
connect to the normal macOS window server in headless/sandboxed environments.
The desktop application runtime is left unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _running_pytest() -> bool:
    executable = Path(sys.argv[0]).name.lower()
    return executable.startswith("pytest") or "PYTEST_CURRENT_TEST" in os.environ


if _running_pytest():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

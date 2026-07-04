"""Workspace-local Python startup customizations for tests.

Python imports ``sitecustomize`` during interpreter startup when the current
workspace is on ``sys.path``.  This keeps PySide/Qt tests headless before any
test module imports PySide.  Normal application launches are unchanged.
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

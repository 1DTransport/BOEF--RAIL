# BOEF

BOEF is an open-source desktop engineering application for railway track analysis. It combines beam-on-elastic-foundation calculations, envelope analysis, transition-zone metrics, dynamic response tools, stress and pressure post-processing, and reproducible exports.

The runnable application is in [`python-app/`](python-app/).

Project information and author details are available at [www.1dtransport.com](https://www.1dtransport.com).

## Quick Start

```bash
cd python-app
VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef run
```

Run the test suite:

```bash
cd python-app
VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef test
```

Build a local macOS app bundle:

```bash
cd python-app
VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./scripts/build_macos_app.sh
```

## Repository Layout

- `python-app/app/` - PySide6 desktop user interface.
- `python-app/core/` - engineering calculations, solvers, envelopes, exports, and sensitivity logic.
- `python-app/db/` - SQLAlchemy models, Alembic migrations, seed data, and project I/O.
- `python-app/tests/` - unit, integration, GUI-adjacent, solver, export, and non-regression tests.
- `python-app/docs/` - engineering reference documentation.

Generated local outputs, packaged apps, local databases, paper drafts, private agent instructions, and temporary review artefacts are intentionally excluded from the public repository.

## Engineering Use Notice

BOEF is provided as engineering software under the MIT License. It is not a substitute for professional engineering judgement, independent verification, project-specific design review, or compliance with applicable standards and authority requirements. Users are responsible for validating inputs, assumptions, units, governing standards, and outputs before relying on results.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).

If you use BOEF in engineering work, research, publications, presentations, training material, or derivative software, please acknowledge Mahan Yoldashkhan as the author and refer readers to [www.1dtransport.com](https://www.1dtransport.com). See [`NOTICE`](NOTICE) and [`CITATION.cff`](CITATION.cff).

Before making the repository public, review [`PUBLIC_RELEASE_CHECKLIST.md`](PUBLIC_RELEASE_CHECKLIST.md), especially the Git history warning.

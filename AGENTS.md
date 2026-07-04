# AGENTS.md

This repository is primarily the BOEF engineering application. The runnable app root is `python-app/`.

Before changing BOEF code, read `python-app/AGENTS.md` and follow its engineering, unit, testing, and packaging rules.

## Public Release Boundary

- Keep this root `AGENTS.md` public so future agents and contributors understand the BOEF workflow.
- Keep `python-app/AGENTS.md` public because it contains the detailed Python engineering, unit, test, and packaging rules.
- Do not publish `python.md`. Treat it as private/local guidance if it exists outside this checkout.
- If `python.md` contains useful recommendations, copy only public-safe technical guidance into `AGENTS.md` or `python-app/AGENTS.md`; do not copy secrets, personal notes, unpublished paper material, private client/project data, or local machine paths.
- Before preparing a public repository, follow `PUBLIC_RELEASE_PLAN.md` and `PUBLIC_RELEASE_CHECKLIST.md`.

## Default Workflow

- Start from `python-app/` for Python app work.
- Use the checked-in `python-app/boef` helper for run and test workflows.
- Prefer verified BOEF commands over generic Python commands.
- Before editing, run `git status --short` and avoid overwriting existing user changes.
- Preserve functionality during release cleanup: removing private files must not change solver, GUI, database, export, test, or packaging behaviour.

## Verified BOEF Commands

From the repository root:

```bash
cd python-app
VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef run
VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef test
VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./scripts/build_macos_app.sh
```

## Packaging Notes

- The macOS build script writes the local bundle to `python-app/dist/BOEF.app`.
- The package icon is the existing `python-app/resources/icon-windowed.icns`; do not replace it unless explicitly asked.
- After packaging changes, verify `CFBundleIconFile`, the bundled `.icns`, executable architecture, and Alembic resources.
- Replacing `/Applications/BOEF.app` is outside the repo and needs explicit approval.

## Scope Notes

- `python-app/` contains the PySide6 desktop app, engineering core, database layer, tests, packaging script, and app docs.
- `presentation-workspace/` may contain separate JS/TS presentation tooling; do not apply JS/Turbo assumptions to BOEF app work unless files in that workspace are directly in scope.
- If an engineering assumption, unit convention, standard, or equation is unclear, stop and ask or document a TODO instead of guessing.

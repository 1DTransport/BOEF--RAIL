# Contributing

Thank you for considering a contribution to BOEF.

## Development Setup

Use the app helper from the runnable application directory:

```bash
cd python-app
VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef run
```

Run tests before submitting changes:

```bash
cd python-app
VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef test
```

## Engineering Changes

BOEF is an engineering tool, so calculation changes need clear evidence.

- Keep internal calculations in SI units.
- Convert units once at input boundaries and once at output boundaries.
- Do not guess engineering constants, equations, standards, or material values.
- Add or update focused tests when changing solver, design-check, export, database, or GUI behavior.
- Document equation, unit, or standard changes in the pull request.

## Public Repository Hygiene

Do not commit:

- secrets, API keys, credentials, tokens, or private certificates,
- local databases such as `*.sqlite`,
- generated exports, packaged apps, or temporary output folders,
- paper drafts, third-party reference PDFs, or private review material,
- local agent instructions or private workspace files.

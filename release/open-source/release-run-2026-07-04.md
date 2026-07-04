# BOEF Public Release Run - 2026-07-04

This file records progress against `PUBLIC_RELEASE_PLAN.md`.

## Current State

- Active private source branch: `main`.
- Full `git status --short` is currently unreliable in this checkout; multiple Git status/diff/list commands hung and had to be stopped.
- The release process therefore uses an explicit public-file allow-list instead of copying from Git history or Git index state.
- `python.md` is not present in this checkout and is excluded through `.gitignore`.
- `AGENTS.md` and `python-app/AGENTS.md` are kept public.
- `python-app/.codex/skills/boef-analysis-integrity/SKILL.md` was inspected and contains engineering verification guardrails; it is copied intentionally as public agent guidance.
- Clean export created at `/tmp/boef-public-release-20260704e`.
- Clean export branch: `main`.
- Clean export was published to the user-created public repository `https://github.com/1DTransport/BOEF--RAIL`.
- Initial public-release commit: `22267d37ede45ff81a835d10762799a5be1a419c`.
- Clean export `git status --short`: no output.
- Full exported BOEF test suite passed: `214 passed, 1 warning in 38.31s`.

## Feature Progress

| Feature | Status | Evidence |
| --- | --- | --- |
| 1 - Release Scope Freeze | Complete for public export | Branch identified as `main`; full private dirty-tree status was blocked by Git/filesystem hang, so the release used a fresh allow-list export instead of private Git history. |
| 2 - Public and Private File Boundary | Complete for local export | `.gitignore` excludes `python.md`; export scan found no `python.md`, databases, DOCX/PDF/PPTX, paper, output, presentation, dist, or build paths. |
| 3 - Licence, Attribution, and Citation | Complete for local export | Export contains MIT `LICENSE`, `NOTICE`, `CITATION.cff`, README attribution, and `python-app/pyproject.toml` MIT/homepage metadata. |
| 4 - Agent Instruction Public Hardening | Complete for local export | Root `AGENTS.md` states `python.md` is private and keeps `AGENTS.md` files public. |
| 5 - Secret and Private Data Scan | Complete with limitation | Keyword scan found only release/security guidance text and app cache variable names; `gitleaks` is not installed. |
| 6 - Functionality Preservation Test Gate | Complete for local export | Full exported test suite passed with `214 passed, 1 warning`. Preferred private-tree helper failed during `.venv_run` rebuild. |
| 7 - Clean Public Repository Export | Complete | `/tmp/boef-public-release-20260704e` is a fresh Git repo on `main`, created without the private repository history. |
| 8 - Public GitHub Repository Creation | Complete | User-created public repository verified and pushed at `https://github.com/1DTransport/BOEF--RAIL`. Existing private `1DTransport/BOEF` was not changed. |
| 9 - GitHub Safety and Collaboration Settings | Partly complete | CI workflow added at `.github/workflows/python-app-tests.yml`; GitHub Issues enabled. Security/dependabot/branch-protection settings require final GitHub-side verification. |
| 10 - First Public Release Tag | Ready | Draft notes added at `release/open-source/v0.1.0-public-release-notes.md`; release tag to be created after this publication-status update is pushed. |
| 11 - `1Dtransport.com` Linkage | External follow-up | Requires website edit path or external website access. |
| 12 - Ongoing Private/Public Workflow | Complete for release | Plan documents the private/public split and warns not to push private history to the public repository. |

## Commands Attempted

```bash
git branch --show-current
```

Result: passed, returned `main`.

```bash
git status --short --untracked-files=normal
git diff --name-status -- . ':!python-app/.codex/skills/boef-analysis-integrity/SKILL.md'
git ls-files -d -m -o --exclude-standard -- . ':!python-app/.codex/skills/boef-analysis-integrity/SKILL.md'
```

Result: blocked/hung and stopped. Do not treat the dirty-tree state as fully verified yet.

```bash
scripts/create_public_export.sh /tmp/boef-public-release-20260704e
```

Result: passed. Created a fresh Git repository with one initial public-release commit.

```bash
find /tmp/boef-public-release-20260704e -path '/tmp/boef-public-release-20260704e/.git' -prune -o \( -iname 'python.md' -o -iname '*.sqlite' -o -iname '*.sqlite3' -o -iname '*.db' -o -iname '*.docx' -o -iname '*.pdf' -o -iname '*.pptx' -o -path '*/Paper/*' -o -path '*/output/*' -o -path '*/presentation-workspace/*' -o -path '*/dist/*' -o -path '*/build/*' \) -print
```

Result: passed. No output.

```bash
rg -n "(password|passwd|secret|api[_-]?key|private key|BEGIN RSA|BEGIN OPENSSH|credential|token|client_secret|access_token)" /tmp/boef-public-release-20260704e -S --glob '!**/.git/**'
```

Result: reviewed. Matches were release/security guidance text and app cache variable names such as `_chart_result_token`; no credential value was identified.

```bash
command -v gitleaks
```

Result: unavailable; `gitleaks` is not installed on this machine.

```bash
VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef test
```

Result: failed during `.venv_run` rebuild. Pip failed building the editable package because `.venv_run/bin/python3.11` was missing during the build.

```bash
python3.11 -m pytest
```

Result: stopped after hanging in private-tree pytest cache/session handling.

```bash
python3.11 -m pytest -p no:cacheprovider
```

Run from `/tmp/boef-public-release-20260704e/python-app`.

Result: passed. `214 passed, 1 warning in 38.31s`.

```bash
bash -n .github/workflows/python-app-tests.yml
```

Not applicable: YAML workflow, not a shell script. The workflow was reviewed structurally and uses GitHub Actions `actions/checkout@v4`, `actions/setup-python@v5`, editable install, and `python -m pytest -p no:cacheprovider`.

## Export Tool

Run from the private source repository:

```bash
scripts/create_public_export.sh /tmp/boef-public-release
```

The export tool:

- refuses to write inside the private source repo,
- refuses to overwrite a non-empty destination,
- copies only approved public files,
- excludes `python.md`,
- excludes paper/output/presentation folders,
- excludes local databases and generated document formats,
- initialises a fresh Git repository with a single public-release commit.

Note: an earlier `rsync`-based export script under `release/open-source/` failed on this macOS filesystem with `mmap: Operation canceled`. The working export script now uses plain file copy logic from `scripts/create_public_export.sh`.

## Publication Result

- Preferred repo name `1DTransport/BOEF` already existed and remained private.
- A temporary fallback repo, `https://github.com/1DTransport/boef-1dtransport`, was created before the user advised the final repo name.
- Official public release repo: `https://github.com/1DTransport/BOEF--RAIL`.
- Public repo visibility verified as `PUBLIC`.
- Default branch verified as `main`.
- Remote configured in the clean export: `https://github.com/1DTransport/BOEF--RAIL.git`.
- Existing private repository visibility was not changed.

## Remaining Follow-ups

- Verify the GitHub Actions run after GitHub completes the first CI execution.
- Create the first public release tag after this publication-status update is pushed.
- Enable or verify Dependabot alerts, secret scanning, and branch protection where available for the GitHub account/repository.
- Update `https://www.1dtransport.com` with the public repository link, MIT licence notice, citation guidance, and engineering-use disclaimer.
- Optional stronger scan: install and run `gitleaks` when available.

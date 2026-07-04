# BOEF Public Open Source Release Plan

This plan breaks the open-source release into deliverable features. It is designed so a later `/goal` run can complete the release feature by feature without losing functionality or publishing private material.

Supporting files:

- `release/open-source/original-plan.md` - preserved original release plan.
- `release/open-source/features.md` - detailed feature breakdown.
- `scripts/create_public_export.sh` - repeatable allow-list export tool for creating a clean public repository without old history.
- `release/open-source/release-run-2026-07-04.md` - current release-run notes and verification status.
- `release/open-source/github-publication-runbook.md` - exact publication commands and settings to use after the GitHub repo name is approved.
- `PUBLIC_RELEASE_CHECKLIST.md` - final pre-publication checklist.

## Release Strategy

Do not make the current private repository public directly.

Create a new clean public GitHub repository from an approved source export. This avoids exposing old private Git history, including paper drafts, third-party references, generated documents, local databases, temporary outputs, or private notes.

## Naming Recommendation

Preferred public repository names:

1. `boef`
2. `boef-rail`
3. `boef-engineering`
4. `beam-on-elastic-foundation`

Recommended choice: `boef` if available, otherwise `boef-rail`.

## Licence and Attribution

Use MIT for the software licence. Keep:

- `LICENSE`
- `NOTICE`
- `CITATION.cff`

The attribution wording should ask users to acknowledge Mahan Yoldashkhan and refer to `https://www.1dtransport.com`, but it should not add extra legal restrictions to MIT.

## Public/Private Boundary

Publish:

- BOEF Python app source under `python-app/app/`, `python-app/core/`, and `python-app/db/`.
- Tests under `python-app/tests/`.
- Public docs under `python-app/docs/`.
- Public release files: `README.md`, `LICENSE`, `NOTICE`, `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md`, and release plan/checklist files.
- `AGENTS.md` and `python-app/AGENTS.md`.

Do not publish:

- `python.md`.
- Paper drafts and reference PDFs.
- DOCX, PDF, and PPTX outputs.
- Local SQLite/database files.
- Generated app bundles and build outputs.
- Private `.codex` notes.
- Temporary export folders.
- Secrets, credentials, API keys, or private certificates.

## Deliverable Features

### Feature 1 - Release Scope Freeze

Freeze feature work while the public release is prepared. Review `git status --short` and decide whether each modified, deleted, or untracked file is included, excluded, or deferred.

### Feature 2 - Public and Private File Boundary

Define the release file boundary. Keep public code, tests, docs, licence files, and public agent instructions. Exclude paper material, generated outputs, local databases, private notes, and `python.md`.

### Feature 3 - Licence, Attribution, and Citation

Confirm MIT licence, attribution notice, citation file, README wording, and `python-app/pyproject.toml` metadata all point to `https://www.1dtransport.com`.

### Feature 4 - Agent Instruction Public Hardening

Keep `AGENTS.md` public. Keep `python-app/AGENTS.md` public. Do not publish `python.md`. If `python.md` exists outside this checkout, merge only public-safe recommendations into `AGENTS.md`.

### Feature 5 - Secret and Private Data Scan

Run text and file scans for secrets, credentials, private data, generated outputs, local databases, and unpublished paper material. Use `gitleaks` or an equivalent scanner if available.

### Feature 6 - Functionality Preservation Test Gate

Run BOEF tests before and after cleanup. Preferred command:

```bash
cd python-app
VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef test
```

Fallback if the helper environment is blocked:

```bash
cd python-app
python3.11 -m pytest
```

Report exact commands and whether each passed, failed, or was blocked.

### Feature 7 - Clean Public Repository Export

Create a clean export folder, copy approved public files only, initialise a fresh Git repository, and make one initial public commit. Do not copy old private Git history.

Use:

```bash
scripts/create_public_export.sh /tmp/boef-public-release
```

The export must be scanned before it is pushed.

### Feature 8 - Public GitHub Repository Creation

Create an empty public GitHub repository, preferably `boef` or `boef-rail`. Push the clean export to `main`.

Use `release/open-source/github-publication-runbook.md` once the final owner/repository name is approved.

### Feature 9 - GitHub Safety and Collaboration Settings

Enable issues, optional discussions, Dependabot/security alerts, secret scanning if available, and branch protection for `main`.

The clean public repository includes a CI workflow at `.github/workflows/python-app-tests.yml` so GitHub can run the BOEF Python test suite on pushes and pull requests.

### Feature 10 - First Public Release Tag

Create the first release tag, such as `v0.1.0-public`, with release notes and an engineering-use disclaimer.

Draft release notes are stored at `release/open-source/v0.1.0-public-release-notes.md`.

### Feature 11 - `1Dtransport.com` Linkage

Update `1Dtransport.com` to link to the public repository and explain licence, citation, and engineering-use responsibilities.

### Feature 12 - Ongoing Private/Public Workflow

Keep private and public work separated. Move future public-safe changes through clean patches, cherry-picks, or PRs. Never push private history to the public repository.

## `/goal` Prompt

Use this prompt later:

```text
/goal Implement the BOEF public open-source release plan in PUBLIC_RELEASE_PLAN.md feature by feature. Preserve all BOEF functionality, keep AGENTS.md and python-app/AGENTS.md public, do not publish python.md, exclude private data and old Git history, run the required scans and BOEF tests, create a clean public repository export, and report exact verification results and any blockers before pushing or changing repository visibility.
```

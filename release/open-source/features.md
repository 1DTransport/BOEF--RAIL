# Open Source Release Features

Use these features as the delivery breakdown for the BOEF public open-source release.

## Feature 1 - Release Scope Freeze

Goal: stop accidental changes while preparing the public release.

Deliverables:

- Record the current branch and dirty-tree state.
- Decide which current changes belong in the public release.
- Confirm no feature development happens during the release-cleaning work.

Acceptance checks:

- `git status --short` has been reviewed.
- Any existing modified/deleted/untracked files have a release decision: keep, exclude, or defer.

## Feature 2 - Public and Private File Boundary

Goal: define exactly what can be published.

Deliverables:

- Keep public source, tests, docs, licence files, and public agent instructions.
- Exclude paper drafts, reference PDFs, DOCX/PPTX outputs, local databases, generated app bundles, temporary exports, and private notes.
- Keep `AGENTS.md` and `python-app/AGENTS.md`.
- Do not publish `python.md`; merge public-safe guidance from it into `AGENTS.md` if it exists outside this checkout.

Acceptance checks:

- `.gitignore` excludes `python.md` and `python-app/python.md`.
- `PUBLIC_RELEASE_CHECKLIST.md` describes the private-file exclusions.
- No private output folders are tracked in the clean public export.

## Feature 3 - Licence, Attribution, and Citation

Goal: release under a licence that enables use while preserving reference to the original author and `1Dtransport.com`.

Deliverables:

- Keep MIT as the software licence.
- Keep `NOTICE` with author and website attribution request.
- Keep `CITATION.cff` for research/publication citation.
- Confirm `python-app/pyproject.toml` declares MIT and `https://www.1dtransport.com`.

Acceptance checks:

- `LICENSE`, `NOTICE`, and `CITATION.cff` are present.
- README licence section points to all three.
- Attribution text is framed as a request, not an extra restriction on MIT.

## Feature 4 - Agent Instruction Public Hardening

Goal: keep useful engineering-agent instructions public without leaking private workflow notes.

Deliverables:

- Update root `AGENTS.md` to explain the public/private release boundary.
- Keep the root instruction simple and point detailed Python app work to `python-app/AGENTS.md`.
- Add a rule that `python.md` is private and must not be published.
- If a private `python.md` is later provided, copy only public-safe technical recommendations into `AGENTS.md`.

Acceptance checks:

- Root `AGENTS.md` mentions the `python.md` exclusion.
- No `python.md` file is tracked.
- Public agent instructions contain no secrets, unpublished paper notes, or private project data.

## Feature 5 - Secret and Private Data Scan

Goal: prevent accidental release of sensitive material.

Deliverables:

- Search for obvious secrets: password, token, API key, private key, certificate, credential, secret.
- Search for unwanted file types: `.sqlite`, `.db`, `.docx`, `.pdf`, `.pptx`, generated outputs.
- Run a dedicated scanner such as `gitleaks` if available.

Acceptance checks:

- Scan commands and results are recorded.
- Any findings are removed or explicitly accepted as public-safe.
- The public export is scanned again before push.

## Feature 6 - Functionality Preservation Test Gate

Goal: avoid losing BOEF functionality during cleanup.

Deliverables:

- Run the BOEF test suite from `python-app/`.
- Run focused tests for any touched app, core, database, or export code.
- Record skipped/blocked GUI checks separately.

Acceptance checks:

- Preferred command attempted:

  ```bash
  cd python-app
  VENV_DIR=.venv_run PYTHON_BIN=python3.11 ./boef test
  ```

- If the helper environment is blocked, fallback command attempted:

  ```bash
  cd python-app
  python3.11 -m pytest
  ```

- Final notes clearly say passed, failed, or blocked.

## Feature 7 - Clean Public Repository Export

Goal: create a public repository without carrying private Git history.

Deliverables:

- Create a clean export folder outside the private working tree.
- Copy only approved public files.
- Initialise a fresh Git repository.
- Make a single initial public commit.

Acceptance checks:

- The new public repo has no old private history.
- `git log --oneline` starts with a clean public-release commit.
- Ignored private/generated files are absent.

## Feature 8 - Public GitHub Repository Creation

Goal: create the final public GitHub location.

Deliverables:

- Use `release/open-source/github-publication-runbook.md` for exact publish commands.
- Create an empty public GitHub repository, preferably `boef` or `boef-rail`.
- Do not let GitHub auto-create a README, licence, or gitignore.
- Add the new remote to the clean export.
- Push `main`.

Acceptance checks:

- Publication runbook exists in the clean public export.
- Public repository exists.
- `main` contains the clean exported source.
- README, licence, notice, citation, contributing, and security files render correctly.

## Feature 9 - GitHub Safety and Collaboration Settings

Goal: make the public repo safe to maintain.

Deliverables:

- Include a GitHub Actions workflow for BOEF tests.
- Enable issues.
- Enable discussions if community Q&A is wanted.
- Enable Dependabot/security alerts where available.
- Enable secret scanning where available.
- Add branch protection for `main`.

Acceptance checks:

- `.github/workflows/python-app-tests.yml` exists in the clean public export.
- Settings are documented in the release notes.
- Direct accidental pushes to `main` are discouraged or blocked after initial release.

## Feature 10 - First Public Release Tag

Goal: mark the first public baseline.

Deliverables:

- Create a release tag such as `v0.1.0-public`.
- Add release notes with engineering-use disclaimer.
- State that users must independently validate assumptions, units, standards, and outputs.

Acceptance checks:

- `release/open-source/v0.1.0-public-release-notes.md` exists as the draft release notes.
- GitHub release exists.
- Release notes link to README, LICENSE, NOTICE, and CITATION.cff.

## Feature 11 - `1Dtransport.com` Linkage

Goal: connect the code release back to the project website.

Deliverables:

- Add or update a `1Dtransport.com` page for BOEF.
- Link to the public GitHub repository.
- Include licence, citation, and engineering-use disclaimer.

Acceptance checks:

- README links to `1Dtransport.com`.
- Website links back to the public repo.

## Feature 12 - Ongoing Private/Public Workflow

Goal: keep private work private while allowing public-source improvements.

Deliverables:

- Keep the current private repo as the internal working repo if needed.
- Use clean patches, cherry-picks, or PRs to move public-safe changes into the public repo.
- Never push private history to the public repository.

Acceptance checks:

- Public release workflow is documented.
- Future release tasks can be driven from this feature list.

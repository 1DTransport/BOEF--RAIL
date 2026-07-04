# Public Release Checklist

Use this checklist before changing the repository visibility to public.

## Required before publishing

- Confirm the latest branch includes `LICENSE`, `NOTICE`, `CITATION.cff`, `README.md`, `CONTRIBUTING.md`, and `SECURITY.md`.
- Confirm `python-app/pyproject.toml` declares the MIT license and the project homepage as `https://www.1dtransport.com`.
- Confirm `AGENTS.md`, `python-app/AGENTS.md`, and `python-app/.codex/skills/boef-analysis-integrity/SKILL.md` remain tracked so the agent instructions are available on other computers.
- Confirm `python.md` is not tracked or published. If useful guidance exists in `python.md`, merge only public-safe recommendations into `AGENTS.md` or `python-app/AGENTS.md`.
- Confirm private/local agent notes are stored only under ignored paths such as `.codex/private/`, `python-app/.codex/private/`, or `*.local.md`.
- Confirm generated app outputs, local databases, paper drafts, DOCX/PDF files, presentation outputs, and temporary folders are not tracked.
- Run a final secret scan before publishing.
- Run the BOEF test suite before announcing the repository as ready for use.
- Complete the deliverable features in `PUBLIC_RELEASE_PLAN.md` or explicitly document any deferred feature.

## Git history warning

Deleting files in the latest commit does not remove them from older Git history. If old tracked files such as paper drafts, third-party reference PDFs, DOCX files, local databases, or private contact details must not be downloadable from the public repository history, do one of the following before making the repository public:

- create a new public repository from a clean export of the current source tree, or
- rewrite/squash the Git history so the public branch starts from a clean public-ready commit.

Do not change the repository visibility to public until this history decision is made.

## Attribution

BOEF is released under the MIT License. Users are requested to acknowledge Mahan Yoldashkhan as the author and refer readers to:

https://www.1dtransport.com

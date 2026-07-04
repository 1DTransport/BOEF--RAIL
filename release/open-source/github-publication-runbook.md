# GitHub Publication Runbook

Use this runbook after the public repository name and owner are confirmed.

Current clean export:

```text
/tmp/boef-public-release-20260704e
```

Recommended repository name:

```text
boef
```

Fallback repository name:

```text
boef-rail
```

## Preconditions

- Do not publish from the private source repository.
- Publish only from the clean export repository.
- Confirm the clean export still has one commit on `main`.
- Confirm the excluded-file scan still returns no output.
- Confirm the BOEF test suite result is recorded in `release/open-source/release-run-2026-07-04.md`.
- Confirm the user has approved the final repository name.

## Final Local Checks

Run from anywhere:

```bash
git -C /tmp/boef-public-release-20260704e branch --show-current
git -C /tmp/boef-public-release-20260704e log --oneline --all
git -C /tmp/boef-public-release-20260704e status --short
find /tmp/boef-public-release-20260704e -path '/tmp/boef-public-release-20260704e/.git' -prune -o \( -iname 'python.md' -o -iname '*.sqlite' -o -iname '*.sqlite3' -o -iname '*.db' -o -iname '*.docx' -o -iname '*.pdf' -o -iname '*.pptx' -o -path '*/Paper/*' -o -path '*/output/*' -o -path '*/presentation-workspace/*' -o -path '*/dist/*' -o -path '*/build/*' \) -print
```

Expected:

- branch is `main`,
- one initial public-release commit,
- clean status,
- excluded-file scan returns no output.

## Optional Stronger Secret Scan

If `gitleaks` is installed:

```bash
gitleaks detect --source /tmp/boef-public-release-20260704e --no-git --redact
```

If it is not installed, record that limitation. The keyword scan has already been run and reviewed.

## Create Public GitHub Repository

Preferred command shape:

```bash
gh repo create OWNER/REPO --public --source /tmp/boef-public-release-20260704e --remote origin --push
```

Example if approved:

```bash
gh repo create OWNER/boef --public --source /tmp/boef-public-release-20260704e --remote origin --push
```

If the repository already exists and is empty:

```bash
git -C /tmp/boef-public-release-20260704e remote add origin git@github.com:OWNER/REPO.git
git -C /tmp/boef-public-release-20260704e push -u origin main
```

Do not use the private repository remote.

## GitHub Settings After Push

Configure:

- Issues: enabled.
- Discussions: optional.
- Dependabot alerts: enabled where available.
- Secret scanning: enabled where available.
- Branch protection for `main`: enabled after the initial push.
- Pull request review before merge: enabled once collaborators are added.

## First Release Tag

Use the release notes draft:

```text
release/open-source/v0.1.0-public-release-notes.md
```

Command shape:

```bash
gh release create v0.1.0-public --repo OWNER/REPO --title "BOEF v0.1.0-public" --notes-file /tmp/boef-public-release-20260704e/release/open-source/v0.1.0-public-release-notes.md
```

## Website Linkage

After the GitHub URL exists, update `https://www.1dtransport.com` to include:

- GitHub repository link,
- MIT licence notice,
- citation guidance,
- engineering-use disclaimer.

## Completion Evidence

Record:

- final GitHub repository URL,
- final pushed commit hash,
- GitHub Actions result,
- release URL,
- settings applied,
- website link status.

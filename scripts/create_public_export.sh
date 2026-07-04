#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DESTINATION="${1:-/tmp/boef-public-release}"

if [[ "$DESTINATION" == "$SOURCE_ROOT" || "$DESTINATION" == "$SOURCE_ROOT/"* ]]; then
  echo "Refusing to create the public export inside the private source repository." >&2
  exit 2
fi

mkdir -p "$DESTINATION"
if find "$DESTINATION" -mindepth 1 -maxdepth 1 | read -r _; then
  echo "Destination is not empty: $DESTINATION" >&2
  echo "Choose a new empty destination. This script does not delete existing files." >&2
  exit 2
fi

should_skip_file() {
  local path="$1"
  case "$path" in
    */.DS_Store|*/__pycache__/*|*/.pytest_cache/*) return 0 ;;
    *.pyc|*.pyo|*.sqlite|*.sqlite3|*.db|*.docx|*.pdf|*.pptx) return 0 ;;
    */python.md|*/PYTHON.md) return 0 ;;
  esac
  return 1
}

copy_file() {
  local relative_path="$1"
  echo "Copying file: $relative_path"
  if should_skip_file "$SOURCE_ROOT/$relative_path"; then
    echo "Skipped excluded file: $relative_path"
    return 0
  fi
  install -d "$DESTINATION/$(dirname "$relative_path")"
  cp "$SOURCE_ROOT/$relative_path" "$DESTINATION/$relative_path"
}

copy_dir() {
  local relative_path="$1"
  local source_dir="$SOURCE_ROOT/$relative_path"
  local source_file
  local nested_relative
  echo "Copying directory: $relative_path"
  while IFS= read -r -d '' source_file; do
    if should_skip_file "$source_file"; then
      continue
    fi
    nested_relative="${source_file#"$SOURCE_ROOT/"}"
    install -d "$DESTINATION/$(dirname "$nested_relative")"
    cp "$source_file" "$DESTINATION/$nested_relative"
  done < <(find "$source_dir" \( -name '__pycache__' -o -name '.pytest_cache' \) -prune -o -type f -print0)
}

copy_markdown_dir() {
  local relative_path="$1"
  local source_dir="$SOURCE_ROOT/$relative_path"
  local source_file
  local nested_relative
  echo "Copying markdown directory: $relative_path"
  while IFS= read -r -d '' source_file; do
    if should_skip_file "$source_file"; then
      continue
    fi
    nested_relative="${source_file#"$SOURCE_ROOT/"}"
    install -d "$DESTINATION/$(dirname "$nested_relative")"
    cp "$source_file" "$DESTINATION/$nested_relative"
  done < <(find "$source_dir" -type f -name '*.md' -print0)
}

copy_file ".gitignore"
copy_file "AGENTS.md"
copy_file "CITATION.cff"
copy_file "CONTRIBUTING.md"
copy_file "LICENSE"
copy_file "NOTICE"
copy_file "PUBLIC_RELEASE_CHECKLIST.md"
copy_file "PUBLIC_RELEASE_PLAN.md"
copy_file "README.md"
copy_file "SECURITY.md"
copy_file "sitecustomize.py"

copy_dir ".github"
copy_dir "release/open-source"
copy_file "scripts/create_public_export.sh"

copy_file "python-app/AGENTS.md"
copy_file "python-app/BOEF.spec"
copy_file "python-app/BOEF_CAPABILITY_MATRIX.md"
copy_file "python-app/README.md"
copy_file "python-app/alembic.ini"
copy_file "python-app/boef"
copy_file "python-app/pyproject.toml"
copy_file "python-app/sitecustomize.py"

copy_dir "python-app/app"
copy_dir "python-app/core"
copy_dir "python-app/db"
copy_markdown_dir "python-app/docs"
copy_dir "python-app/resources"
copy_dir "python-app/scripts"
copy_dir "python-app/tests"

copy_file "python-app/.codex/skills/boef-analysis-integrity/SKILL.md"

chmod +x "$DESTINATION/scripts/create_public_export.sh"
chmod +x "$DESTINATION/python-app/boef"
chmod +x "$DESTINATION/python-app/scripts/build_macos_app.sh"

if find "$DESTINATION" \( \
  -iname 'python.md' \
  -o -iname '*.sqlite' \
  -o -iname '*.sqlite3' \
  -o -iname '*.db' \
  -o -iname '*.docx' \
  -o -iname '*.pdf' \
  -o -iname '*.pptx' \
  -o -path '*/Paper/*' \
  -o -path '*/output/*' \
  -o -path '*/presentation-workspace/*' \
  -o -path '*/dist/*' \
  -o -path '*/build/*' \
  \) -print | grep -q .; then
  echo "Export contains files that should not be public:" >&2
  find "$DESTINATION" \( \
    -iname 'python.md' \
    -o -iname '*.sqlite' \
    -o -iname '*.sqlite3' \
    -o -iname '*.db' \
    -o -iname '*.docx' \
    -o -iname '*.pdf' \
    -o -iname '*.pptx' \
    -o -path '*/Paper/*' \
    -o -path '*/output/*' \
    -o -path '*/presentation-workspace/*' \
    -o -path '*/dist/*' \
    -o -path '*/build/*' \
    \) -print >&2
  exit 3
fi

git -C "$DESTINATION" init
git -C "$DESTINATION" add .
git -C "$DESTINATION" commit -m "Initial public release of BOEF"
git -C "$DESTINATION" branch -M main

cat <<MSG
Created clean public export:
$DESTINATION

Next manual steps:
1. Run the release scans against this export.
2. Run/record BOEF tests from the private source tree or the export.
3. Create an empty GitHub repo.
4. Add the remote and push only after final approval.
MSG

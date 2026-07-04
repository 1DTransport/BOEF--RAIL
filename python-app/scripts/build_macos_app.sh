#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
APP_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$APP_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
APP_NAME="${APP_NAME:-BOEF}"
APP_ICON="${APP_ICON:-resources/icon-windowed.icns}"
APP_ICON_BASENAME=$(basename "$APP_ICON")
BUILD_ROOT="${BUILD_ROOT:-${TMPDIR:-/tmp}/boef-macos-build}"
BUILD_WORK="$BUILD_ROOT/build"
BUILD_DIST="$BUILD_ROOT/dist"
export PYINSTALLER_CONFIG_DIR="${PYINSTALLER_CONFIG_DIR:-${TMPDIR:-/tmp}/boef-pyinstaller}"
mkdir -p "$PYINSTALLER_CONFIG_DIR"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"

"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install --no-build-isolation -e .
"$VENV_PY" -m pip install pyinstaller

if command -v xattr >/dev/null 2>&1; then
  xattr -cr "$VENV_DIR" >/dev/null 2>&1 || true
fi

clean_path() {
  local target="$1"
  if [ -e "$target" ]; then
    if command -v xattr >/dev/null 2>&1; then
      xattr -cr "$target" >/dev/null 2>&1 || true
    fi
    chmod -R u+rwX "$target" >/dev/null 2>&1 || true
    find "$target" -name ".DS_Store" -delete >/dev/null 2>&1 || true
    rm -rf "$target"
  fi
}

clean_path build
clean_path "dist/$APP_NAME.app"
clean_path "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT"

verify_build_artifact() {
  local app_bundle="$1"
  local executable="$app_bundle/Contents/MacOS/$APP_NAME"
  local info_plist="$app_bundle/Contents/Info.plist"
  local expected_arch="${EXPECTED_APP_ARCH:-$(uname -m)}"
  local executable_info
  local bundle_icon

  echo
  echo "Verifying macOS app bundle..."

  if [ ! -d "$app_bundle" ]; then
    echo "Verification failed: app bundle not found at $app_bundle" >&2
    exit 1
  fi

  if [ ! -x "$executable" ]; then
    echo "Verification failed: executable not found or not executable at $executable" >&2
    exit 1
  fi

  executable_info=$(file "$executable")
  if [[ "$executable_info" != *"Mach-O"* ]]; then
    echo "Verification failed: executable is not a Mach-O binary: $executable_info" >&2
    exit 1
  fi
  if [[ "$expected_arch" == "arm64" && "$executable_info" != *"arm64"* ]]; then
    echo "Verification failed: executable is not arm64: $executable_info" >&2
    exit 1
  fi
  if [[ "$expected_arch" == "x86_64" && "$executable_info" != *"x86_64"* && "$executable_info" != *"universal"* ]]; then
    echo "Verification failed: executable is not x86_64 or universal: $executable_info" >&2
    exit 1
  fi

  if [ ! -f "$info_plist" ]; then
    echo "Verification failed: Info.plist not found at $info_plist" >&2
    exit 1
  fi
  bundle_icon=$(/usr/libexec/PlistBuddy -c "Print :CFBundleIconFile" "$info_plist")
  if [ "$bundle_icon" != "$APP_ICON_BASENAME" ]; then
    echo "Verification failed: CFBundleIconFile is '$bundle_icon', expected '$APP_ICON_BASENAME'" >&2
    exit 1
  fi

  if [ ! -f "dist/$APP_NAME.icns" ]; then
    echo "Verification failed: preserved dist icon file missing at dist/$APP_NAME.icns" >&2
    exit 1
  fi
  if [ ! -f "$APP_ICON" ]; then
    echo "Verification failed: source icon file missing at $APP_ICON" >&2
    exit 1
  fi
  if [ ! -f "$app_bundle/Contents/Resources/$APP_ICON_BASENAME" ]; then
    echo "Verification failed: bundled icon missing at $app_bundle/Contents/Resources/$APP_ICON_BASENAME" >&2
    exit 1
  fi

  if [ ! -f "$app_bundle/Contents/Resources/alembic.ini" ]; then
    echo "Verification failed: bundled alembic.ini is missing." >&2
    exit 1
  fi
  if [ ! -d "$app_bundle/Contents/Resources/db/migrations" ]; then
    echo "Verification failed: bundled db/migrations directory is missing." >&2
    exit 1
  fi

  if command -v codesign >/dev/null 2>&1; then
    codesign --verify "$app_bundle"
  fi

  echo "Verification passed:"
  echo "  Bundle: $app_bundle"
  echo "  Executable: $executable_info"
  echo "  Icon: $bundle_icon"
  echo "  Migration resources: present"
}

PYINSTALLER_ARGS=(
  --name "$APP_NAME"
  --windowed
  --clean
  --noconfirm
  --workpath "$BUILD_WORK"
  --distpath "$BUILD_DIST"
  --add-data "alembic.ini:."
  --add-data "db/migrations:db/migrations"
  --collect-data matplotlib
  --hidden-import logging.config
  --hidden-import matplotlib.backends.backend_qtagg
  --hidden-import sqlalchemy.sql.default_comparator
)

if [ -f "$APP_ICON" ]; then
  PYINSTALLER_ARGS+=(--icon "$APP_ICON")
else
  echo "Warning: icon file not found at $APP_ICON; building with the default app icon." >&2
fi

"$VENV_PY" -m PyInstaller \
  "${PYINSTALLER_ARGS[@]}" \
  app/__main__.py

TEMP_APP_BUNDLE="$BUILD_DIST/$APP_NAME.app"
if [ ! -d "$TEMP_APP_BUNDLE" ] && [ -d "$BUILD_DIST/Application/$APP_NAME.app" ]; then
  TEMP_APP_BUNDLE="$BUILD_DIST/Application/$APP_NAME.app"
fi

if [ ! -d "$TEMP_APP_BUNDLE" ]; then
  echo "Build finished, but no $APP_NAME.app bundle was found under dist/." >&2
  exit 1
fi

mkdir -p dist
APP_BUNDLE="dist/$APP_NAME.app"
ditto --norsrc --noextattr --noqtn --noacl "$TEMP_APP_BUNDLE" "$APP_BUNDLE"

if command -v xattr >/dev/null 2>&1; then
  xattr -cr "$APP_BUNDLE" >/dev/null 2>&1 || true
  xattr -c "$APP_BUNDLE" >/dev/null 2>&1 || true
  find "$APP_BUNDLE" -name "*.framework" -type d -print0 | xargs -0 xattr -c >/dev/null 2>&1 || true
fi

verify_build_artifact "$APP_BUNDLE"

echo
echo "Built: $APP_DIR/$APP_BUNDLE"
echo "You can double-click that app in Finder."

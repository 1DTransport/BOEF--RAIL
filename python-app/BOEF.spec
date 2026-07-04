# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = [('alembic.ini', '.'), ('db/migrations', 'db/migrations')]
datas += collect_data_files('matplotlib')


a = Analysis(
    ['app/__main__.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['logging.config', 'matplotlib.backends.backend_qtagg', 'sqlalchemy.sql.default_comparator'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BOEF',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['resources/icon-windowed.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BOEF',
)
app = BUNDLE(
    coll,
    name='BOEF.app',
    icon='resources/icon-windowed.icns',
    bundle_identifier=None,
)

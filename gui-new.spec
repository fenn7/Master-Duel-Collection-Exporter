# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gui-new.py'],
    pathex=[],
    binaries=[],
    datas=[('dependencies/tesseract', 'tesseract'), ('templates', 'templates'), ('main-new-better.py', '.'), ('card_name_matcher.py', '.')],
    hiddenimports=['pandas._libs.tslibs.np_datetime', 'pandas._libs.tslibs.nattype', 'pandas._libs.skiplist', 'rapidfuzz.fuzz', 'rapidfuzz.process', 'rapidfuzz.distance', 'win32api', 'win32con', 'win32gui', 'win32com'],
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
    a.binaries,
    a.datas,
    [],
    name='gui-new',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['favicon.ico'],
)

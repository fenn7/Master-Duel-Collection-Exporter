# -*- mode: python ; coding: utf-8 -*-
import os

# Get the directory where the spec file is located
spec_root = os.path.abspath(SPECPATH)

a = Analysis(
    ['gui.py'],
    pathex=[spec_root],
    binaries=[],
    datas=[
        # Include main.py so gui can import it
        ('main.py', '.'),
        # Include card_name_matcher if it exists
        ('card_name_matcher.py', '.') if os.path.exists(os.path.join(spec_root, 'card_name_matcher.py')) else None,
        # Include templates folder for image matching
        ('templates', 'templates'),
        # Include tesseract if bundled locally
        ('dependencies/tesseract', 'dependencies/tesseract') if os.path.exists(os.path.join(spec_root, 'dependencies/tesseract')) else None,
        # Include favicon if it exists
        ('favicon.ico', '.') if os.path.exists(os.path.join(spec_root, 'favicon.ico')) else None,
    ],
    hiddenimports=[
        'main',
        'card_name_matcher',
        'cv2',
        'numpy',
        'pytesseract',
        'mss',
        'pyautogui',
        'pygetwindow',
        'requests',
        'pandas',
        'openpyxl',
        'win32com.client',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# Filter out None entries from datas
a.datas = [d for d in a.datas if d is not None]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False for no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='favicon.ico' if os.path.exists(os.path.join(spec_root, 'favicon.ico')) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='gui',
)
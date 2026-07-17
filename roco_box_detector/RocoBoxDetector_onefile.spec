# -*- mode: python ; coding: utf-8 -*-
# onefile 模式 — 单 exe 文件，内嵌默认 config
# 模板文件从 exe 同目录下的 templates/ 文件夹读取，可自由增删

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('config.json', '.'),
    ],
    hiddenimports=[
        'mss', 'keyboard', 'cv2', 'numpy', 'PyQt5', 'PIL', 'ctypes',
        'selector', 'overlay', 'detector', 'template_cache',
        'debug_utils', 'image_utils', 'geometry_utils',
        'sequence_analyzer', 'settings_panel', 'startup_dialog',
    ],
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
    name='RocoBoxDetector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

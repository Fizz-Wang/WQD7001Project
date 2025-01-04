# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ModelPredict.py'],
    pathex=[],
    binaries=[],
    datas=[('model_classifier.pkl', '.'), ('label_encoder.pkl', '.')],
    hiddenimports=['sklearn.utils._cython_blas', 'sklearn.utils._joblib', 'scipy.stats'],
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
    name='ModelPredict',
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
)
app = BUNDLE(
    exe,
    name='ModelPredict.app',
    icon=None,
    bundle_identifier=None,
)

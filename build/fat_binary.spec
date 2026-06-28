# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller one-dir specification for the offline Enterprise distribution."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve().parent
VENDOR = ROOT / "build" / "vendor"


def optional_collection(package):
    try:
        return collect_all(package)
    except Exception:
        return ([], [], [])


datas = [(str(ROOT / "config"), "config")]
binaries = []
hiddenimports = []

for package in (
    "torch",
    "transformers",
    "huggingface_hub",
    "awq",
    "auto_gptq",
    "fastapi",
    "uvicorn",
    "redis",
    "sqlalchemy",
    "prometheus_client",
    "jwt",
    "cryptography",
    "psutil",
):
    package_datas, package_binaries, package_hidden = optional_collection(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

if VENDOR.exists():
    for item in VENDOR.rglob("*"):
        if item.is_file():
            relative_parent = item.parent.relative_to(VENDOR)
            binaries.append((str(item), str(Path("vendor") / relative_parent)))

analysis = Analysis(
    [str(ROOT / "haradibots_main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter.test", "test", "tests"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="HaradiBots",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
collect = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="HaradiBots",
)

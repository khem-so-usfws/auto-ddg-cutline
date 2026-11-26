"""Bundle the windowed launcher and console worker with one shared runtime."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyInstaller.building.api import COLLECT, EXE, PYZ
    from PyInstaller.building.build_main import Analysis

    SPECPATH: str

# PyInstaller supplies SPECPATH and the build classes when it runs this spec.
root: Path = Path(SPECPATH)
analysis: Analysis = Analysis([str(root / "windows_entry.py")], pathex=[str(root)])
pyz: PYZ = PYZ(analysis.pure)
launcher: EXE = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="AutoDDGCutline",
    console=False,
    upx=False,
)
worker: EXE = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="AutoDDGCutlineCLI",
    console=True,
    upx=False,
)
bundle: COLLECT = COLLECT(
    launcher,
    worker,
    analysis.binaries,
    analysis.datas,
    name="AutoDDGCutline",
    upx=False,
)

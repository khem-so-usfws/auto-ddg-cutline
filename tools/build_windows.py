"""Build, verify, and zip a portable Windows release of Auto-DDG Cutline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from auto_ddg_cutline.windows_launcher import CLI_EXE, LAUNCHER_EXE

REPO: Path = Path(__file__).resolve().parents[1]
START_HERE: str = """Auto-DDG Cutline for Windows

1. Extract this complete ZIP to a local folder.
2. Download and extract COLMAP separately.
3. Open AutoDDGCutline.exe and choose your COLMAP.bat and photo folder.
4. Click Analyze photos, then Review drafts to see the paired cutlines.
5. Use Export drafts to DDG to inspect the initial lines on your original photos.
6. Adjust the photo checkboxes and click Refine cutlines.

Use Test GPU to verify feature matching on your NVIDIA setup. Open run logs
shows the device used and the reason for any CPU fallback. See README.md for
the command line and build instructions.
"""


def release_version(value: str) -> str:
    """Accept a version token that can also be used in a release filename."""
    if not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*)?", value
    ):
        raise argparse.ArgumentTypeError("Use a version such as 0.1.0 or 0.1.0-rc.1.")
    return value


def require_windows() -> None:
    """Build the x64 release with a native x64 Windows Python environment."""
    if (
        sys.platform != "win32"
        or platform.machine().lower() not in {"amd64", "x86_64"}
        or sys.maxsize <= 2**32
    ):
        raise ValueError("Run the release build from 64-bit x86 Windows Python.")


def clean_runtime_environment() -> dict[str, str]:
    """Smoke-test the bundle without loading DLLs or Python from Conda."""
    windows: Path = Path(os.environ["SystemRoot"])
    exclude: set[str] = {
        "PYTHONPATH",
        "PYTHONHOME",
        "TCL_LIBRARY",
        "TK_LIBRARY",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "COLMAP_PATH",
    }
    environment: dict[str, str] = {
        name: value for name, value in os.environ.items() if name.upper() not in exclude
    }
    environment["PATH"] = os.pathsep.join((str(windows / "System32"), str(windows)))
    return environment


def check_bundle(bundle: Path, environment: dict[str, str], version: str) -> None:
    """Verify the packaged Tcl interpreter and command worker without COLMAP."""
    for executable, arguments in (
        (LAUNCHER_EXE, ["--launcher-check", version]),
        (CLI_EXE, ["--engine", "--help"]),
    ):
        subprocess.run(
            [str(bundle / executable), *arguments],
            cwd=bundle,
            stdin=subprocess.PIPE,
            env=environment,
            timeout=30,
            check=True,
        )


def create_archive(bundle: Path, destination: Path) -> Path:
    """Validate the portable ZIP and publish it with a matching checksum locally."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    checksum: Path = destination.with_suffix(".zip.sha256")
    with tempfile.TemporaryDirectory(
        prefix="release-", dir=destination.parent
    ) as staging:
        temporary: Path = Path(staging) / destination.name
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        (Path(destination.stem) / path.relative_to(bundle)).as_posix(),
                    )
        with zipfile.ZipFile(temporary) as archive:
            invalid: str | None = archive.testzip()
            if invalid is not None:
                raise ValueError(f"Release archive verification failed at {invalid}.")
        with temporary.open("rb") as stream:
            digest: str = hashlib.file_digest(stream, "sha256").hexdigest()
        checksum_file: Path = Path(staging) / checksum.name
        checksum_file.write_text(f"{digest}  {destination.name}\n", encoding="ascii")
        temporary.replace(destination)
        checksum_file.replace(checksum)
    return checksum


def build(version: str, output: Path) -> Path:
    """Run the tests and PyInstaller in local scratch before packaging the result."""
    require_windows()
    environment: dict[str, str] = clean_runtime_environment()
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO, check=True)
    working_parent: Path = (
        Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
        / "AutoDDGCutline"
        / "build"
    )
    working_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="release-", dir=working_parent
    ) as temporary:
        workspace: Path = Path(temporary)
        dist: Path = workspace / "dist"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "PyInstaller",
                "--noconfirm",
                "--distpath",
                str(dist),
                "--workpath",
                str(workspace / "work"),
                str(REPO / "windows_launcher.spec"),
            ],
            cwd=REPO,
            check=True,
        )
        bundle: Path = dist / "AutoDDGCutline"
        shutil.copy2(REPO / "README.md", bundle / "README.md")
        (bundle / "START_HERE.txt").write_text(START_HERE, encoding="utf-8")
        metadata: dict[str, str] = {
            "version": version,
            "python": platform.python_version(),
            "pyinstaller": importlib.metadata.version("pyinstaller"),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "colmap": "external",
        }
        (bundle / "release.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        check_bundle(bundle, environment, version)
        zip_path: Path = output / f"AutoDDGCutline-{version}-windows-x64.zip"
        checksum: Path = create_archive(bundle, zip_path)
        print(f"Built {zip_path}\nChecksum {checksum}", flush=True)
    return zip_path


def main(argv: Sequence[str] | None = None) -> int:
    """Parse repeatable release inputs and report actionable build failures."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Build a portable Windows Auto-DDG Cutline release."
    )
    parser.add_argument("--version", required=True, type=release_version)
    parser.add_argument("--output-dir", type=Path, default=REPO / "dist" / "releases")
    args: argparse.Namespace = parser.parse_args(argv)
    try:
        build(args.version, args.output_dir.resolve())
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Release build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

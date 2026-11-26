"""Exercise live subprocess logs and a heartbeat during quiet processing."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from auto_ddg_cutline.benchmark import RunReport
from auto_ddg_cutline.colmap import Colmap, ColmapError, find_colmap


@pytest.mark.parametrize("recover", [True, False])
def test_manifest_replace_handles_windows_access_denied(
    tmp_path: Path, recover: bool
) -> None:
    """Keep the previous checkpoint visible while retrying a denied replace."""
    path: Path = tmp_path / "run.json"
    path.write_text('{"status": "running"}', encoding="utf-8")
    report: RunReport = RunReport(path, {"status": "complete"}, False)
    original_replace: Callable[[Path, Path], None] = os.replace
    attempts: int = 0
    denial: OSError = PermissionError("Access is denied")
    setattr(denial, "winerror", 5)

    def replace(source: Path, destination: Path) -> None:
        """Inject the Windows error while the previous manifest remains readable."""
        nonlocal attempts
        attempts += 1
        assert (
            json.loads(destination.read_text(encoding="utf-8"))["status"] == "running"
        )
        assert json.loads(source.read_text(encoding="utf-8"))["status"] == "complete"
        if attempts <= 2 or not recover:
            raise denial
        original_replace(source, destination)

    try:
        with (
            patch("auto_ddg_cutline.benchmark.os.replace", side_effect=replace),
            patch("auto_ddg_cutline.benchmark.sleep") as wait,
        ):
            if recover:
                report.save()
                assert attempts == 3
            else:
                with pytest.raises(PermissionError) as failure:
                    report.save()
                assert failure.value is denial
                assert attempts == 6
            assert wait.call_count == attempts - 1
        expected: str = "complete" if recover else "running"
        assert json.loads(path.read_text(encoding="utf-8"))["status"] == expected
        assert list(tmp_path.iterdir()) == [path]
    finally:
        report.stop.set()
        report.heartbeat.join()


def test_log_is_flushed_while_child_runs_and_failure_preserves_timing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Real child output must reach disk before that process can finish."""
    log: Path = tmp_path / "child.log"
    script: str = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "print('Processed file [1/3]', flush=True)\n"
        "for _ in range(100):\n"
        "    if 'Processed file' in Path(sys.argv[1]).read_text():\n"
        "        break\n"
        "    time.sleep(0.01)\n"
        "else:\n"
        "    raise SystemExit(9)\n"
        "time.sleep(0.3)\n"
        "print('simulated error after live log check', file=sys.stderr, flush=True)\n"
        "raise SystemExit(7)\n"
    )
    manifest: dict[str, Any] = {"status": "failure"}
    with patch("auto_ddg_cutline.benchmark.HEARTBEAT_SECONDS", 0.05):
        report: RunReport = RunReport(tmp_path / "run.json", manifest, True)
        engine: Colmap = Colmap(Path(sys.executable), {}, report=report)
        manifest["command_timings"] = engine.command_timings
        try:
            with (
                pytest.raises(ColmapError) as failure,
                report.phase("feature_extraction", "gpu"),
            ):
                engine.run(["-c", script, str(log)], log)
        finally:
            report.finish()
    assert failure.value.returncode == 7
    assert "simulated error after live log check" in log.read_text(encoding="utf-8")
    output: str = capsys.readouterr().out
    assert "photos processed" in output
    assert "heartbeat" in output
    checkpoint: dict[str, Any] = json.loads(report.path.read_text(encoding="utf-8"))
    assert checkpoint["stage_timings"][0]["status"] == "failed"
    assert checkpoint["command_timings"][0]["returncode"] == 7
    assert checkpoint["command_timings"][0]["elapsed_seconds"] >= 0.3


def test_setup_timeout_identifies_the_check_and_saves_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End a stalled help child, keeping its incomplete output and exact command."""
    log: Path = tmp_path / "extract-help.log"
    manifest: dict[str, Any] = {"status": "failure"}
    report: RunReport = RunReport(tmp_path / "run.json", manifest, False)
    engine: Colmap = Colmap(Path(sys.executable), dict(os.environ), report=report)
    manifest["command_timings"] = engine.command_timings
    script: str = "import time; print('help started', flush=True); time.sleep(60)"
    try:
        with (
            patch("auto_ddg_cutline.colmap.HELP_TIMEOUT_SECONDS", 2.0),
            pytest.raises(
                TimeoutError, match=r"timed out after 2 seconds.*extract-help\.log"
            ),
            report.phase("setup"),
        ):
            engine.run(["-c", script, "-h"], log)
    finally:
        report.finish()
    assert log.read_text(encoding="utf-8").strip() == "help started"
    assert "extract-help.log" in capsys.readouterr().out
    checkpoint: dict[str, Any] = json.loads(report.path.read_text(encoding="utf-8"))
    assert checkpoint["stage_timings"][0]["status"] == "failed"
    assert checkpoint["command_timings"][0]["status"] == "timed_out"
    assert checkpoint["command_timings"][0]["timeout_seconds"] == 2.0
    assert (
        engine.run(["-c", "print('help finished')", "-h"], log).strip()
        == "help finished"
    )


def test_bundled_colmap_avoids_inherited_conda_libraries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolate the shipped CPU runtime while allowing an alternate build's libraries."""
    release: Path = tmp_path / "Auto DDG release"
    runtime: Path = release / "_internal"
    conda: Path = tmp_path / "Miniconda" / "Library" / "bin"
    windows: Path = tmp_path / "Windows"
    portable: Path = release / "colmap"
    native: Path = portable / "bin" / "colmap.exe"
    plugins: Path = portable / "plugins"
    native.parent.mkdir(parents=True)
    native.touch()
    plugins.mkdir()
    alternate: Path = tmp_path / "Alternate COLMAP" / "colmap.exe"
    alternate.parent.mkdir()
    alternate.touch()

    with monkeypatch.context() as env:
        env.setattr(sys, "platform", "win32")
        env.setattr(sys, "_MEIPASS", str(runtime), raising=False)
        env.setattr(sys, "executable", str(release / "AutoDDGCutlineCLI.exe"))
        env.setenv("SYSTEMROOT", str(windows))
        env.setenv("PATH", os.pathsep.join((str(runtime / "cv2"), str(conda))))
        env.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", str(conda))
        env.setenv("QT_QPA_PLATFORM", "offscreen")
        bundled: Colmap = find_colmap(str(native))
        explicit: Colmap = find_colmap(str(alternate))

    assert bundled.environment["PATH"].split(os.pathsep) == [
        str(native.parent),
        str(windows / "System32"),
        str(windows),
    ]
    assert bundled.environment["QT_PLUGIN_PATH"] == str(plugins)
    assert "QT_QPA_PLATFORM_PLUGIN_PATH" not in bundled.environment
    assert "QT_QPA_PLATFORM" not in bundled.environment
    assert explicit.environment["PATH"].split(os.pathsep) == [
        str(alternate.parent),
        str(conda),
    ]

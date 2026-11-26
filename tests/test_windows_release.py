"""Pin the launcher arguments and portable release archive structure."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from auto_ddg_cutline.launcher_workflow import (
    SavedSelection,
    SelectionPlanRelocationNeeded,
    export_review,
    load_saved_selection,
    read_tier2_proposal,
    refine_review,
    refinement_route_requests,
    save_tier2_selection,
)
from auto_ddg_cutline.windows_launcher import (
    AnalysisOptions,
    cpu_worker_count,
    job_arguments,
    launcher_title,
    saved_match_summary,
)
from tools.build_windows import create_archive, release_version


def test_refinement_route_snapshot_maps_checked_photo_indices() -> None:
    """Carry a prior nonconsecutive pairing and replan across a removed view."""
    plan: dict[str, Any] = {
        "counting_set": [0, 1, 2, 3],
        "export_pairs": [[0, 2], [1, 2], [1, 3]],
        "unresolved_handoffs": [[0, 3]],
        "sections": [{"handoff_walks": [[0, 2, 1, 3]]}],
    }
    assert refinement_route_requests(plan, [5, 3, 0, 1]) == {
        "reviewed": [(1, 3)],
        "reconnect": [(1, 2), (2, 3)],
        "added": [0],
    }


def test_load_previous_selection_matches_names_and_preserves_refinement_order(
    tmp_path: Path,
) -> None:
    """Restore checkboxes and review files from a complete run after moving the photos."""
    job: Path = tmp_path / "old job"
    job.mkdir()
    names: list[str] = ["C.jpg", "A.jpg", "B.jpg"]
    plan: Path = job / "selection-plan.json"
    plan.write_text(
        json.dumps(
            {
                "photos": [{"name": name} for name in names],
                "counting_set": [0, 1],
                "export_pairs": [[0, 1]],
            }
        ),
        encoding="utf-8",
    )
    html: Path = plan.with_suffix(".html")
    html.touch()
    run: Path = job / "run.json"
    run.write_text(
        json.dumps(
            {
                "status": "partial",
                "output": "selection-plan",
                "images": [{"name": name} for name in names],
            }
        ),
        encoding="utf-8",
    )
    photos: list[Path] = [
        tmp_path / "moved photos" / name
        for name in ["A.jpg", "B.jpg", "C.jpg", "D.jpg"]
    ]
    selection: SavedSelection = load_saved_selection(run, photos)
    assert selection == load_saved_selection(plan, photos)
    assert selection.recommended == {0, 2}
    assert selection.refinement_indices([0, 2]) == [1, 0]
    assert selection.refinement_indices([0, 3]) is None
    assert selection.report == html and selection.can_export
    with pytest.raises(ValueError, match=r"recommended photos are missing.*C\.jpg"):
        load_saved_selection(plan, photos[:2])


def test_load_standalone_selection_can_refine_with_original_hashes(
    tmp_path: Path,
) -> None:
    """Use a copied plan without its HTML, manifests, or matching workspace."""
    photo: Path = tmp_path / "A.jpg"
    photo.write_bytes(b"expected")
    row: dict[str, Any] = {
        "name": photo.name,
        "id": "000001.png",
        "width": 20,
        "height": 10,
        "sha256": hashlib.sha256(b"expected").hexdigest(),
    }
    copied: Path = tmp_path / "archived selection-plan.json"
    copied.write_text(
        json.dumps({"photos": [row], "counting_set": [0]}), encoding="utf-8"
    )
    selection: SavedSelection = load_saved_selection(copied, [photo])
    assert selection.recommended == {0}
    assert (
        selection.report is None and selection.run is None and not selection.can_export
    )
    snapshot: Path = save_tier2_selection(
        tmp_path, selection.plan, selection.digest, [0]
    )
    assert json.loads(snapshot.read_bytes())["photos"] == [row]
    assert json.loads(snapshot.read_bytes())["run_manifest"] is None
    photo.write_bytes(b"different")
    with pytest.raises(ValueError, match="changed since analysis"):
        save_tier2_selection(tmp_path, selection.plan, selection.digest, [0])


def test_launcher_preserves_selection_and_explicit_device(tmp_path: Path) -> None:
    """Use the CLI selection indices, including for paths with spaces."""
    folder: Path = tmp_path / "Colony A"
    jobs: Path = tmp_path / "Run logs"
    colmap: Path = tmp_path / "Alternate COLMAP" / "COLMAP.bat"
    colmap.parent.mkdir()
    colmap.touch()
    args: list[str] = job_arguments(
        AnalysisOptions(
            folder,
            jobs,
            colmap,
            selected=[2, 0],
            device="gpu",
            feature_pixel_percent=30,
            cpu_threads=4,
        )
    )
    assert args == [
        str(folder),
        "--all",
        "--output",
        "selection-plan",
        "--work-dir",
        str(jobs),
        "--colmap-path",
        str(colmap),
        "--device",
        "gpu",
        "--cpu-threads",
        "4",
        "--feature-pixel-percent",
        "30",
        "--run-local-sfm",
        "--count-selection",
        "3,1",
    ]
    reuse: Path = jobs / "run.json"
    saved: list[str] = job_arguments(
        AnalysisOptions(
            folder,
            jobs,
            None,
            reuse=reuse,
            local_sfm=False,
            feature_pixel_percent=30,
            cpu_threads=4,
        )
    )
    assert saved[-2:] == ["--reuse-matches", str(reuse)]
    assert "--device" not in saved and "--colmap-path" not in saved
    assert "--feature-pixel-percent" not in saved and "--cpu-threads" not in saved
    compact: list[str] = job_arguments(
        AnalysisOptions(
            folder, jobs, None, reuse=reuse, local_sfm=False, target_photo_percent=25
        )
    )
    assert compact == [*saved, "--target-photo-percent", "25"]
    with pytest.raises(ValueError, match="at least two"):
        job_arguments(AnalysisOptions(folder, jobs, colmap, selected=[0]))


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        (
            {"feature_pixel_percent": 53.2, "effective_feature_max_edge": 4096},
            "extraction at 53.2% of original pixels",
        ),
        ({"effective_feature_max_edge": 4096}, "extraction limit 4096 px long edge"),
        ({}, "extraction size unavailable"),
    ],
)
def test_load_saved_extraction_description(
    tmp_path: Path, settings: dict[str, int | float], expected: str
) -> None:
    """Use recorded matching settings even when a fresh request differs."""
    manifest: Path = tmp_path / "run.json"
    manifest.write_text(
        json.dumps(
            {
                "feature_pixel_percent_requested": 50,
                "match_checkpoint": {"settings": settings},
            }
        ),
        encoding="utf-8",
    )
    assert saved_match_summary(manifest) == f"Using saved matches · {expected}"


def test_cpu_workers_require_a_positive_integer() -> None:
    """Accept an edited count and reject invalid counts before starting a run."""
    assert cpu_worker_count(" 4 ") == 4
    for invalid in ("", "0", "-1", "2.5"):
        with pytest.raises(ValueError, match="positive integer"):
            cpu_worker_count(invalid)


def test_release_zip_preserves_bundle_tree_and_checksum(tmp_path: Path) -> None:
    """Keep the complete one-folder bundle under one versioned ZIP root."""
    bundle: Path = tmp_path / "build"
    (bundle / "_internal").mkdir(parents=True)
    (bundle / "AutoDDGCutline.exe").write_bytes(b"gui")
    (bundle / "AutoDDGCutlineCLI.exe").write_bytes(b"cli")
    (bundle / "_internal" / "python.dll").write_bytes(b"python")
    release: Path = tmp_path / "releases" / "AutoDDGCutline-0.1.0-windows-x64.zip"
    checksum: Path = create_archive(bundle, release)
    with zipfile.ZipFile(release) as archive:
        assert archive.testzip() is None
        assert archive.read(f"{release.stem}/_internal/python.dll") == b"python"
        assert all("/colmap/" not in name for name in archive.namelist())
        assert archive.read(f"{release.stem}/AutoDDGCutlineCLI.exe") == b"cli"
    assert checksum.read_text(encoding="ascii").split() == [
        hashlib.sha256(release.read_bytes()).hexdigest(),
        release.name,
    ]
    assert release_version("0.1.0-rc.1") == "0.1.0-rc.1"
    (bundle / "release.json").write_text('{"version": "0.1.3"}', encoding="utf-8")
    assert launcher_title(bundle) == "Auto-DDG Cutline 0.1.3"
    assert launcher_title() == "Auto-DDG Cutline 0.1.6-overview.4"
    with pytest.raises(argparse.ArgumentTypeError):
        release_version("../0.1.0")


def test_worker_can_exit_while_the_launcher_keeps_stdin_open() -> None:
    """A quiet cancellation listener must not block or crash interpreter shutdown."""
    command: list[str] = [
        sys.executable,
        "-m",
        "auto_ddg_cutline.windows_launcher",
        "--engine",
        "--help",
    ]
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as process:
        assert process.wait(timeout=20) == 0
        assert process.stdout is not None
        assert b"--reuse-matches" in process.stdout.read()


def test_export_uses_saved_drafts_and_detects_changed_photos(tmp_path: Path) -> None:
    """Keep reviewed coordinates and count sides, and protect changed photos."""
    folder: Path = tmp_path / "photos"
    folder.mkdir()
    job: Path = tmp_path / "job"
    job.mkdir()
    manifest: Path = job / "run.json"
    manifest.write_text(
        json.dumps({"status": "partial", "output": "selection-plan"}), encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
    for index in (0, 1):
        path: Path = folder / f"img{index}.jpg"
        path.write_bytes(b"photo")
        rows.append(
            {
                "name": path.name,
                "width": 10,
                "height": 10,
                "id": path.name,
                "sha256": hashlib.sha256(b"photo").hexdigest(),
            }
        )
        views.append(
            {
                "photo": index,
                "line": [[1, 2], [3, 4]],
                "keep_point": [2, 4],
                "keep_side": "left",
                "keep_direction": "left",
            }
        )
    plan: dict[str, Any] = {
        "photos": rows,
        "counting_set": [1, 0],
        "export_pairs": [[0, 1]],
        "overlap_pairs": [{"photos": [0, 1], "handoff": {"views": views}}],
    }
    (job / "selection-plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert export_review(folder, manifest) == 0
    first: Path = folder / "img0.json"
    data: dict[str, Any] = json.loads(first.read_text(encoding="utf-8"))
    assert data["shapes"][0]["points"] == [[1, 2], [3, 4]]
    assert data["shapes"][0]["auto_ddg_handoff"]["keep_point"] == [2, 4]
    assert export_review(folder, manifest) == 0
    assert len(json.loads(first.read_text(encoding="utf-8"))["shapes"]) == 1
    (folder / "img1.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed since analysis"):
        export_review(folder, manifest)


def test_confirm_tier2_saves_added_and_dropped_photos(tmp_path: Path) -> None:
    """The launcher worker saves the reviewed changes and protects earlier decisions."""
    folder: Path = tmp_path / "Colony A"
    folder.mkdir()
    job: Path = tmp_path / "Run logs"
    job.mkdir()
    (job / "run.json").write_text(
        json.dumps({"status": "partial", "output": "selection-plan"}), encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    for index in range(3):
        image: Path = folder / f"photo {index}.jpg"
        pixels: bytes = bytes([index])
        image.write_bytes(pixels)
        rows.append(
            {
                "name": image.name,
                "id": f"{index + 1:06}.png",
                "sha256": hashlib.sha256(pixels).hexdigest(),
                "width": 20,
                "height": 10,
            }
        )
    plan_path: Path = job / "selection-plan.json"
    plan: dict[str, Any] = {"photos": rows, "counting_set": [0, 1]}
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    digest: str
    _, digest = read_tier2_proposal(plan_path)
    command: list[str] = [
        sys.executable,
        "-m",
        "auto_ddg_cutline.windows_launcher",
        "--engine",
        "--confirm-tier2",
        str(folder),
        str(plan_path),
        digest,
        "0,2",
    ]
    result: subprocess.CompletedProcess[str] = subprocess.run(
        command, capture_output=True, text=True, timeout=20, check=True
    )
    assert "Saved 2 photos for Tier 2" in result.stdout
    saved: Path = job / "tier2-selection.json"
    original_confirmation: bytes = saved.read_bytes()
    data: dict[str, Any] = json.loads(original_confirmation)
    assert data["photo_indices"] == [0, 2]
    assert data["photos"] == [rows[0], rows[2]]
    assert data["selection_plan_sha256"] == digest
    assert json.loads(plan_path.read_bytes()) == plan
    assert list(folder.glob("*.json")) == []
    reversed_photos: list[Path] = [folder / row["name"] for row in reversed(rows)]
    restored: SavedSelection = load_saved_selection(saved, reversed_photos)
    assert restored.restored_checks == frozenset({0, 2})
    assert restored.recommended == frozenset({1, 2})
    assert restored.refinement_indices([0, 2]) == [2, 0]

    # The refinement run consumes exactly the accepted photos, in their saved order.
    with patch("auto_ddg_cutline.launcher_workflow.run_colmap", return_value=0) as run:
        assert (
            refine_review(
                folder,
                plan_path,
                digest,
                [0, 2],
                "COLMAP.bat",
                "cpu",
                job,
                cpu_threads=1,
            )
            == 0
        )
        assert run.call_args.args[:3] == (
            [folder / rows[0]["name"], folder / rows[2]["name"]],
            "COLMAP.bat",
            "cpu",
        )
        assert run.call_args.args[4] is False
        assert run.call_args.kwargs["counting_order"] == [0, 1]
        assert "feature_max_edge" not in run.call_args.kwargs
        refinement: Path = run.call_args.kwargs["refinement_selection"]
        assert refinement.parent.parent == job
        assert json.loads(refinement.read_bytes())["photos"] == [rows[0], rows[2]]

    completed: Path = refinement.parent / "auto-ddg-colmap-test"
    completed.mkdir()
    (completed / "run.json").write_text(
        json.dumps({"status": "complete", "output": "selection-plan"}), encoding="utf-8"
    )
    (completed / "selection-plan.json").write_text(
        json.dumps(
            {"photos": data["photos"], "refinement": True, "export_pairs": [[0, 1]]}
        ),
        encoding="utf-8",
    )
    refined_report: Path = completed / "selection-plan.html"
    refined_report.touch()
    restored = load_saved_selection(refinement, reversed_photos)
    assert restored.refined_report == refined_report
    assert restored.can_export_refined

    moved: Path = tmp_path / "Moved analysis"
    moved.mkdir()
    moved_plan: Path = plan_path.rename(moved / plan_path.name)
    with pytest.raises(SelectionPlanRelocationNeeded):
        load_saved_selection(refinement, reversed_photos)
    relocated: SavedSelection = load_saved_selection(
        refinement, reversed_photos, original_plan=moved_plan
    )
    assert relocated.plan == moved_plan
    assert relocated.restored_checks == restored.restored_checks
    wrong: Path = tmp_path / "different.json"
    wrong.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_saved_selection(refinement, reversed_photos, original_plan=wrong)
    moved_plan.rename(plan_path)

    (folder / rows[2]["name"]).write_bytes(b"edited")
    changed_photo: subprocess.CompletedProcess[str] = subprocess.run(
        command, capture_output=True, text=True, timeout=20, check=False
    )
    assert changed_photo.returncode == 1
    assert "changed since analysis" in changed_photo.stderr
    assert saved.read_bytes() == original_confirmation

    plan_path.write_text(json.dumps({**plan, "counting_set": [1, 2]}), encoding="utf-8")
    changed_plan: subprocess.CompletedProcess[str] = subprocess.run(
        command, capture_output=True, text=True, timeout=20, check=False
    )
    assert changed_plan.returncode == 1
    assert "The selection plan changed" in changed_plan.stderr
    assert saved.read_bytes() == original_confirmation

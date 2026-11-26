"""Verify supported 3D cutlines and checkbox updates from the cached model."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from auto_ddg_cutline.benchmark import RunReport
from auto_ddg_cutline.colmap_model import Camera, Model, Photo, Track, View
from auto_ddg_cutline.geometry import Coordinates, Pixels
from auto_ddg_cutline.match_cache import MatchAssets
from auto_ddg_cutline.survey_dense import DENSE_CACHE_NAME
from auto_ddg_cutline.survey_overview import (
    build_survey_overview,
    cached_surface,
    lift_cutline,
    photo_preview,
    photo_relationships,
    scene_block,
)
from auto_ddg_cutline.survey_viewer import ensure_overview_entry
from test_colmap import synthetic_block


def test_photo_overview_embeds_and_reuses_offline_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep photo appearance in the standalone view and avoid repeat decoding."""
    photo: Path = tmp_path / "colony.jpg"
    pixels: Pixels = np.full((800, 2200, 3), (35, 120, 210), dtype=np.uint8)
    encoded: Pixels
    ok: bool
    ok, encoded = cv2.imencode(".jpg", pixels)
    assert ok
    photo.write_bytes(encoded.tobytes())
    folder: Path = tmp_path / "cache"
    preview: str | None = photo_preview(photo, folder)
    assert preview is not None
    decoded: Pixels = cv2.imdecode(
        np.frombuffer(base64.b64decode(preview.split(",", 1)[1]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded.shape[1] == 420
    assert int(decoded[:, :, 2].mean()) > int(decoded[:, :, 0].mean())
    monkeypatch.setattr(
        cv2, "imdecode", Mock(side_effect=AssertionError("Preview is cached"))
    )
    assert photo_preview(photo, folder) == preview


def test_verified_neighbors_include_photos_outside_the_counting_set(
    tmp_path: Path,
) -> None:
    """Link reviewed pairs and expose other verified neighbors without a cutline."""
    draft: Path = tmp_path / "selection-plan.html"
    draft.touch()
    plan: dict[str, Any] = {
        "photos": [{"name": name} for name in ("A.jpg", "B.jpg", "C.jpg")],
        "counting_set": [0, 1],
        "overlap_pairs": [{"photos": [0, 1], "inliers": 62, "handoff_error": "test"}],
    }
    checkpoint: dict[str, Any] = {
        "verified_pairs": [
            {"images": ["C.jpg", "A.jpg"], "inliers": 48},
            {"images": ["A.jpg", "B.jpg"], "inliers": 60},
            {"images": ["B.jpg", "C.jpg"], "inliers": 1},
        ]
    }
    links: dict[tuple[str, str], dict[str, Any]] = {
        edge["photos"]: edge
        for edge in photo_relationships(
            plan, checkpoint, draft, {"A.jpg", "B.jpg", "C.jpg"}
        )
    }
    assert links["A.jpg", "C.jpg"]["link"] is None
    assert links["A.jpg", "B.jpg"]["link"].endswith("selection-plan.html#pair-0-1")
    assert ("B.jpg", "C.jpg") not in links
    assert photo_relationships(plan, {}, draft, {"A.jpg", "B.jpg"})[0]["inliers"] == 62


def test_builds_and_reuses_supported_footprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse observed terrain faces with support IDs available to the viewer."""
    model: Model = synthetic_block(tmp_path)
    cache: Path = tmp_path / "surface.json"
    surface: dict[str, Any] = cached_surface(model, cache)
    assert surface["vertices"]
    assert any({3, 14}.issubset(face[3]) for face in surface["faces"])
    assert any(3 in face[3] and 39 not in face[3] for face in surface["faces"])
    monkeypatch.setattr(
        "auto_ddg_cutline.survey_overview.surface_faces",
        Mock(side_effect=AssertionError("Cached terrain must not be rebuilt")),
    )
    assert cached_surface(model, cache) == surface
    scene: dict[str, Any] = scene_block(model, surface, [])
    assert {camera["view"] for camera in scene["cameras"]} == {3, 14, 39}
    assert scene["surface"]["faces"] == surface["faces"]


def test_unsupported_stretch_breaks_the_3d_cutline(tmp_path: Path) -> None:
    """Lift corresponding pixels onto a known plane and retain the support gap."""
    camera: Camera = Camera(1200, 800, 800.0, np.array([600.0, 400.0]), 0.02)
    landmarks: Coordinates = np.array(
        [
            [float(x), float(y), 10.0]
            for x in np.r_[np.linspace(-4, -1.3, 22), np.linspace(1.3, 4, 22)]
            for y in (-0.1, 0.1)
        ]
    )
    # Preserve paired samples on the slightly distorted image-space paths.
    line: Coordinates = np.array([[x, 0.0, 10.0] for x in np.linspace(-4, 4, 33)])
    views: list[View] = []
    for index, translation in enumerate((0.0, -1.8)):
        photo: Photo = Photo(
            tmp_path / f"{index}.jpg", 1200, 800, f"{index}.png", "test"
        )
        shift: Coordinates = np.array([translation, 0.0, 0.0])
        sample: View = View(index, photo, camera, np.eye(3), shift, {})
        pixels: Coordinates
        pixels, _ = sample.project(landmarks)
        views.append(
            View(index, photo, camera, np.eye(3), shift, dict(enumerate(pixels)))
        )
    model: Model = Model(
        "model-0",
        {view.id: view for view in views},
        {
            index: Track(position, 0.1, frozenset({0, 1}))
            for index, position in enumerate(landmarks)
        },
    )
    first_line: Coordinates = views[0].project(line)[0]
    second_line: Coordinates = views[1].project(line)[0]
    spans: list[Coordinates]
    supported: int
    total: int
    spans, supported, total = lift_cutline(
        model, views[0], views[1], (first_line, second_line)
    )
    assert len(spans) == 2
    assert 0 < supported < total
    assert spans[0][-1, 0] < 0 < spans[1][0, 0]
    np.testing.assert_allclose(np.concatenate(spans)[:, 2], 10, atol=0.01)
    # Duplicate camera poses supply no stereo depth.
    assert lift_cutline(model, views[0], views[0], (first_line, first_line))[0] == []


def test_checkbox_updates_reuse_the_geometry_and_keep_refinement_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build two overviews from one saved model and update the pair links."""
    folder: Path = tmp_path / "photos"
    folder.mkdir()
    staged: Path = tmp_path / "staged"
    staged.mkdir()
    run: Path = tmp_path / "analysis"
    run.mkdir()
    database: Path = run / "database.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE example (id INTEGER)")
    names: list[str] = ["01.jpg", "02.jpg", "03.jpg"]
    photos: dict[str, Photo] = {}
    records: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        original: Path = folder / name
        original.touch()
        staged_name: str = f"{index:06}.png"
        (staged / staged_name).touch()
        photos[staged_name] = Photo(original, 1200, 800, staged_name, "test")
        records.append(
            {
                "name": name,
                "staged_name": staged_name,
                "width": 1200,
                "height": 800,
                "sha256": "test",
            }
        )
    checkpoint: Path = run / "run.json"
    checkpoint.write_text(
        json.dumps(
            {
                "backend": "colmap",
                "images": records,
                "match_checkpoint": {
                    "database": str(database),
                    "image_folder": str(staged),
                    "device": "cpu",
                    "settings": {},
                },
            }
        ),
        encoding="utf-8",
    )
    model_dir: Path = run / "model-0"
    model_dir.mkdir()
    (model_dir / "cameras.txt").write_text(
        "1 SIMPLE_RADIAL 1200 800 800 600 400 0\n", encoding="utf-8"
    )
    (model_dir / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 000000.png\n600 400 1\n"
        "2 1 0 0 0 -2 0 0 1 000001.png\n440 400 1\n",
        encoding="utf-8",
    )
    (model_dir / "points3D.txt").write_text(
        "1 0 0 10 100 130 120 0.1 1 0 2 0\n", encoding="utf-8"
    )
    validations: list[Path] = []

    def validate(
        path: Path, selected: list[Path], report: RunReport
    ) -> tuple[MatchAssets, dict[str, Photo]]:
        """Return validated fixture assets while counting costly validation calls."""
        validations.append(path)
        assert [photo.name for photo in selected] == names
        return MatchAssets(database, staged, "cpu", {}), photos

    monkeypatch.setattr("auto_ddg_cutline.survey_overview.load_matches", validate)

    handoff: dict[str, Any] = {
        "views": [{"line": [[1, 2], [3, 4]]}, {"line": [[2, 3], [4, 5]]}]
    }
    plan: dict[str, Any] = {
        "photos": records,
        "counting_set": [0, 1],
        "export_pairs": [[0, 1]],
        "overlap_pairs": [{"photos": [0, 1], "handoff": handoff}],
    }
    initial: Path = run / "selection-plan.json"
    initial.write_text(json.dumps(plan), encoding="utf-8")
    initial.with_suffix(".html").touch()
    ensure_overview_entry(run)
    stable: Path = run / "survey-overview.html"
    assert "Update Survey Overview" in stable.read_text(encoding="utf-8")
    refined: Path = run / "refined.json"
    refined.write_text(
        json.dumps(
            {
                "refinement": True,
                "photos": records[:2],
                "counting_set": [0, 1],
                "overlap_pairs": [],
                "export_pairs": [],
                "unresolved_handoffs": [[0, 1]],
            }
        ),
        encoding="utf-8",
    )

    def overview(
        name: str, checked: list[str], refinement: Path | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the worker with one checkbox snapshot and read its generated data."""
        job: Path = tmp_path / name
        job.mkdir()
        request: Path = job / "overview-input.json"
        request.write_text(
            json.dumps(
                {
                    "folder": str(folder),
                    "checkpoint": str(checkpoint),
                    "plan": str(initial),
                    "digest": hashlib.sha256(initial.read_bytes()).hexdigest(),
                    "checked": checked,
                    "refined": str(refinement) if refinement else None,
                }
            ),
            encoding="utf-8",
        )
        assert build_survey_overview(request) == 0
        html: str = stable.read_text(encoding="utf-8")
        data: dict[str, Any] = json.loads(
            html.split('id="overview-data">')[1].split("</script>", 1)[0]
        )
        manifest: dict[str, Any] = json.loads(
            (job / "run.json").read_text(encoding="utf-8")
        )
        assert Path(manifest["overview_report"]) == stable
        assert not (job / "survey-overview.html").exists()
        return data, manifest

    first: dict[str, Any]
    manifest: dict[str, Any]
    first, manifest = overview("overview-1", names[:2])
    assert manifest["alignment"] == "existing"
    assert len(first["cuts"]) == 1
    assert first["cuts"][0]["link"].endswith("selection-plan.html#pair-0-1")
    assert [camera["name"] for camera in first["blocks"][0]["cameras"]] == names[:2]
    assert all(len(camera["frustum"]) == 4 for camera in first["blocks"][0]["cameras"])
    assert first["denseFocus"] is None and first["blocks"][0]["dense"] is None
    assert [row["name"] for row in first["photos"]] == names
    assert [row["checked"] for row in first["photos"]] == [True, True, False]
    ensure_overview_entry(run)
    assert 'id="overview-data"' in stable.read_text(encoding="utf-8")

    cached_model: Path = next((tmp_path / "survey-overview-cache").glob("*/model-0"))
    (cached_model / DENSE_CACHE_NAME).write_text(
        json.dumps(
            {
                "focus": names[0],
                "photos": names[:2],
                "vertices": [[0, 0, 10], [1, 0, 10], [0, 1, 10]],
                "faces": [[0, 1, 2, [1, 2]]],
                "colors": [[80, 100, 120]],
            }
        ),
        encoding="utf-8",
    )

    updated: dict[str, Any]
    updated, manifest = overview("overview-2", [names[0], names[2]])
    assert manifest["alignment"] == "cached"
    assert len(updated["cuts"]) == 0
    assert [camera["name"] for camera in updated["blocks"][0]["cameras"]] == names[:2]
    assert [row["checked"] for row in updated["photos"]] == [True, False, True]
    assert updated["sections"] == [[1], [3]]
    assert updated["denseFocus"] is None
    assert updated["blocks"][0]["dense"]["focus"] == names[0]

    unresolved: dict[str, Any]
    unresolved, manifest = overview("overview-3", names[:2], refined)
    assert manifest["alignment"] == "cached"
    assert unresolved["cuts"] == []
    assert unresolved["missing"] == [names[:2]]
    assert validations == [checkpoint]
    all_context: dict[str, Any]
    all_context, manifest = overview("overview-no-checks", [])
    assert not any(row["checked"] for row in all_context["photos"])
    assert len(all_context["blocks"][0]["cameras"]) == 2
    assert all_context["sections"] == []

    # Force a new checkpoint identity and exercise the mapper entry path.
    fixture: Path = model_dir.rename(tmp_path / "fixture-model")
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO example VALUES (1)")
    monkeypatch.setattr("auto_ddg_cutline.survey_overview.find_colmap", Mock())
    monkeypatch.setattr(
        "auto_ddg_cutline.survey_overview.inspect_flags",
        Mock(return_value=("", Mock())),
    )

    def map_fixture(
        engine: Any, flags: Any, copy: Path, images: Path, inputs: dict[str, Photo]
    ) -> list[Model]:
        """Check that the mapper gets a copy and all photos from the checkpoint."""
        assert copy != database and copy.is_file()
        assert images == staged and inputs == photos
        shutil.copytree(fixture, copy.parent / "model-0")
        return []

    monkeypatch.setattr("auto_ddg_cutline.survey_overview.map_matches", map_fixture)
    _, manifest = overview("overview-4", names[:1])
    assert manifest["alignment"] == "mapped"
    _, manifest = overview("overview-5", names[1:2])
    assert manifest["alignment"] == "cached"
    assert validations == [checkpoint, checkpoint]

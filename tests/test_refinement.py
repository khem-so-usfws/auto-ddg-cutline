"""Verify native seam search and cutline export on controlled overlaps."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
from numpy.typing import NDArray

from auto_ddg_cutline.colmap import Colmap, Flags
from auto_ddg_cutline.geometry import (
    Coordinates,
    Descriptors,
    Frame,
    Pixels,
    color_transition_cost,
    image_seam,
)
from auto_ddg_cutline.launcher_workflow import (
    export_review,
    read_tier2_proposal,
    refine_review,
)
from auto_ddg_cutline.milestone2 import run_colmap
from test_paired import write_match_database


def test_color_transition_penalizes_crossing_and_preserves_edge_direction() -> None:
    """Distinguish crossing and following a color boundary in either view."""
    first: Pixels = np.zeros((2, 4, 3), dtype=np.uint8)
    second: Pixels = first.copy()
    second[:, 2:, 0] = 180
    cost: Descriptors = color_transition_cost(first, second)
    assert cost[0, 1, 1] == cost[0, 1, 2] == 0
    assert cost[0, 0, 2] > 0
    assert cost[0, 2, 1] > 0


def test_draft_routes_around_edges_with_identical_grayscale(tmp_path: Path) -> None:
    """Use a color-only detour that the grayscale planner leaves straight."""
    path: Path = tmp_path / "draft.png"
    color: Pixels = np.zeros((160, 240, 3), dtype=np.uint8)
    color[:] = [0, 102, 0]
    for row in range(45, 113, 8):
        color[row : row + 4, 112:129] = [0, 0, 200]
    assert cv2.imwrite(str(path), color)
    gray: Pixels = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    assert np.unique(gray).tolist() == [60]
    frame: Frame = Frame(path, 240, 160, gray, np.empty((0, 2)), None)
    marked: Frame = replace(frame, draft_color_source=path)
    baseline: Coordinates = np.array([[120.0, 12.0], [120.0, 145.0]])
    normal: Coordinates = np.array([1.0, 0.0])
    tangent: Coordinates = np.array([0.0, 1.0])
    straight: Coordinates
    straight_reason: str
    straight, straight_reason = image_seam(
        frame, frame, np.eye(3), frame.footprint, baseline, normal, tangent
    )
    assert straight_reason is None
    np.testing.assert_array_equal(straight[:, 0], 120)
    detour: Coordinates
    detour_reason: str
    detour, detour_reason = image_seam(
        marked, marked, np.eye(3), frame.footprint, baseline, normal, tangent
    )
    assert detour_reason is None
    occupied: NDArray[np.bool_] = (detour[:, 1] >= 45) & (detour[:, 1] <= 113)
    assert np.all((detour[occupied, 0] < 112) | (detour[occupied, 0] >= 129))
    np.testing.assert_array_equal(detour[[0, -1]], baseline)


def test_seed_search_uses_original_image_samples(tmp_path: Path) -> None:
    """A one-native-pixel path shift can be found inside the seed corridor."""
    pixels: Pixels = np.full((140, 240), 120, dtype=np.uint8)
    a: Frame = Frame(tmp_path / "a.jpg", 240, 140, pixels, np.empty((0, 2)), None)
    b: Frame = Frame(
        tmp_path / "b.jpg", 240, 140, pixels.copy(), np.empty((0, 2)), None
    )
    seed: Coordinates = np.array([[100.0, 12.0], [104.0, 70.0], [100.0, 125.0]])
    baseline: Coordinates = np.array([[100.0, 12.0], [100.0, 125.0]])

    # Put the lowest cost one pixel right of the seed in the middle rows.
    def stripe(
        first: Descriptors, second: Descriptors, valid: NDArray[np.bool_]
    ) -> Descriptors:
        """Score one native offset lower than its two neighbors."""
        cost: Descriptors = np.ones_like(first)
        midpoint: int = first.shape[1] // 2
        cost[4:-4, midpoint + 1] = 0
        return cost

    with patch("auto_ddg_cutline.geometry.seam_cost", side_effect=stripe):
        route: Coordinates
        reason: str
        route, reason = image_seam(
            a,
            b,
            np.eye(3),
            a.footprint,
            baseline,
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            seed=seed,
        )
    assert reason is None
    np.testing.assert_array_equal(route[[0, -1]], seed[[0, -1]])
    np.testing.assert_allclose(route[route[:, 1] == 70][0], [105, 70])


def test_refinement_exports_only_cutlines_from_current_and_legacy_plans(
    tmp_path: Path,
) -> None:
    """Keep native cutlines and omit legacy match fragments on review export."""
    photos: Path = tmp_path / "photos"
    photos.mkdir()
    jobs: Path = tmp_path / "jobs"
    jobs.mkdir()
    initial_job: Path = jobs / "initial"
    initial_job.mkdir()
    for name in ("a.png", "b.png"):
        assert cv2.imwrite(str(photos / name), np.full((240, 320), 128, dtype=np.uint8))
    database: Path = initial_job / "matches.db"
    write_match_database(database)
    engine: Colmap = Colmap(tmp_path / "colmap", {})
    flags: Flags = Flags(
        "--FeatureExtraction.max_image_size", "", "", "", (), False, False
    )
    with (
        patch("auto_ddg_cutline.milestone2.find_colmap", return_value=engine),
        patch(
            "auto_ddg_cutline.milestone2.inspect_flags",
            return_value=("COLMAP test", flags),
        ),
        patch("auto_ddg_cutline.milestone2.extract_and_match", return_value=database),
    ):
        assert (
            run_colmap(
                sorted(photos.glob("*.png")),
                None,
                "cpu",
                jobs,
                False,
                output="selection-plan",
                counting_order=[0, 1],
            )
            == 0
        )
        initial: Path = next(jobs.glob("auto-ddg-colmap-*/selection-plan.json"))
        draft: dict[str, Any]
        digest: str
        draft, digest = read_tier2_proposal(initial)
        assert draft["draft_color_edge_weight"] > 0
        assert (
            refine_review(
                photos, initial, digest, [0, 1], "colmap", "cpu", jobs, cpu_threads=1
            )
            == 0
        )
    plan_file: Path = next(jobs.glob("cutline-refinement-*/*/selection-plan.json"))
    plan: dict[str, Any] = json.loads(plan_file.read_bytes())
    assert plan["refinement"] and "count_areas" not in plan
    assert "draft_color_edge_weight" not in plan
    assert plan["refinement_routes"]["refined_reviewed_pairs"] == [[0, 1]]
    report: str = plan_file.with_suffix(".html").read_text(encoding="utf-8")
    assert "Initial cut in blue" in report and "Refined cut in yellow" in report
    assert "Refined 1 of 1 reviewed photo pairs" in report
    assert 'src="images/000001.png"' in report and 'id="count-areas"' not in report

    manifest: Path = plan_file.with_name("run.json")
    assert (
        json.loads(manifest.read_bytes())["selection_working_coordinates"] == "native"
    )
    assert not list(photos.glob("*.json"))
    manual: Path = photos / "a.json"
    existing: dict[str, Any] = {
        "label": "count_region",
        "shape_type": "polygon",
        "points": [[6, 8], [16, 8], [16, 20]],
    }
    manual.write_text(json.dumps({"shapes": [existing]}), encoding="utf-8")
    assert export_review(photos, manifest) == 0
    before: bytes = manual.read_bytes()
    shapes: list[dict[str, Any]] = json.loads(before)["shapes"]
    assert shapes[0] == existing
    assert shapes[1]["label"] == "cutline"
    assert (
        shapes[1]["points"] == plan["overlap_pairs"][0]["handoff"]["views"][0]["line"]
    )

    # Export directly from a saved 0.1.4 review into a fresh copy.
    legacy: Path = tmp_path / "legacy export"
    legacy.mkdir()
    for photo in photos.glob("*.png"):
        (legacy / photo.name).write_bytes(photo.read_bytes())
    plan["count_areas"] = {
        "photos": [{"photo": 0, "polygons": [[[20, 30], [35, 30], [35, 60]]]}],
        "unbounded_pairs": [],
    }
    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    assert export_review(legacy, manifest) == 0
    for sidecar in legacy.glob("*.json"):
        assert [
            shape["label"] for shape in json.loads(sidecar.read_bytes())["shapes"]
        ] == ["cutline"]
    assert (legacy / "a.json").is_file() and (legacy / "b.json").is_file()
    assert export_review(photos, manifest) == 0 and manual.read_bytes() == before


def test_ddg_group_polygons_affect_only_opted_in_refinement(tmp_path: Path) -> None:
    """Read DDG sidecars and withhold a handoff when its overlap is blocked."""
    photos: Path = tmp_path / "photos"
    jobs: Path = tmp_path / "jobs"
    photos.mkdir()
    names: tuple[str, str] = ("a.png", "b.png")
    for name in names:
        assert cv2.imwrite(str(photos / name), np.full((240, 320), 128, dtype=np.uint8))
    database: Path = tmp_path / "matches.db"
    write_match_database(database)
    engine: Colmap = Colmap(tmp_path / "colmap", {})
    flags: Flags = Flags(
        "--FeatureExtraction.max_image_size", "", "", "", (), False, False
    )
    with (
        patch("auto_ddg_cutline.milestone2.find_colmap", return_value=engine),
        patch(
            "auto_ddg_cutline.milestone2.inspect_flags",
            return_value=("COLMAP test", flags),
        ),
        patch("auto_ddg_cutline.milestone2.extract_and_match", return_value=database),
    ):
        assert (
            run_colmap(
                [photos / name for name in names],
                None,
                "cpu",
                jobs,
                False,
                output="selection-plan",
                counting_order=[0, 1],
            )
            == 0
        )
        initial: Path = next(jobs.glob("auto-ddg-colmap-*/selection-plan.json"))
        draft: dict[str, Any]
        digest: str
        draft, digest = read_tier2_proposal(initial)
        draft_line: Coordinates = np.asarray(
            draft["overlap_pairs"][0]["handoff"]["views"][1]["line"]
        )
        center: int = round(float(draft_line[:, 0].mean()))
        sidecar: Path = photos / "b.json"
        shapes: list[dict[str, Any]] = [
            {
                "label": "count_region",
                "shape_type": "polygon",
                "points": [[0, 0], [319, 0], [319, 239], [0, 239]],
            },
            {
                "label": "protect_group",
                "shape_type": "polygon",
                "points": [
                    [center - 22, 85],
                    [center + 22, 85],
                    [center + 22, 150],
                    [center - 22, 150],
                ],
            },
        ]
        sidecar.write_text(
            json.dumps({"shapes": shapes, "imageWidth": 320, "imageHeight": 240}),
            encoding="utf-8",
        )
        snapshot: bytes = sidecar.read_bytes()
        assert (
            refine_review(
                photos,
                initial,
                digest,
                [0, 1],
                "colmap",
                "cpu",
                jobs / "protected",
                cpu_threads=1,
                protect_groups=True,
            )
            == 0
        )
        assert sidecar.read_bytes() == snapshot
        refined_path: Path = next((jobs / "protected").glob("*/*/selection-plan.json"))
        plan: dict[str, Any] = json.loads(refined_path.read_bytes())
        assert plan["protected_groups"] == {"b.png": [shapes[1]["points"]]}
        assert plan["summary"]["protected_group_polygons"] == 1
        assert plan["summary"]["group_rerouted_handoffs"] == 1
        assert plan["export_pairs"] == [[0, 1]]
        line: Coordinates = np.asarray(
            plan["overlap_pairs"][0]["handoff"]["views"][1]["line"]
        )
        halfway: float = float(np.interp(115, line[:, 1], line[:, 0]))
        assert abs(halfway - center) >= 22 + 12
        assert "Rerouted around a protected bird group" in refined_path.with_suffix(
            ".html"
        ).read_text(encoding="utf-8")

        # A group spanning both sides of the overlap defeats any safe detour.
        shapes[1]["points"] = [[0, 85], [319, 85], [319, 150], [0, 150]]
        sidecar.write_text(json.dumps({"shapes": shapes}), encoding="utf-8")
        assert (
            refine_review(
                photos,
                initial,
                digest,
                [0, 1],
                "colmap",
                "cpu",
                jobs / "blocked",
                cpu_threads=1,
                protect_groups=True,
            )
            == 0
        )
        blocked_path: Path = next((jobs / "blocked").glob("*/*/selection-plan.json"))
        blocked: dict[str, Any] = json.loads(blocked_path.read_bytes())
        assert blocked["export_pairs"] == []
        assert blocked["unresolved_handoffs"] == [[0, 1]]
        assert "protected DDG polygons" in blocked["overlap_pairs"][0]["handoff_error"]

        assert (
            refine_review(
                photos,
                initial,
                digest,
                [0, 1],
                "colmap",
                "cpu",
                jobs / "unconstrained",
                cpu_threads=1,
            )
            == 0
        )
        unconstrained_path: Path = next(
            (jobs / "unconstrained").glob("*/*/selection-plan.json")
        )
        unconstrained: dict[str, Any] = json.loads(unconstrained_path.read_bytes())
        assert unconstrained["export_pairs"] == [[0, 1]]
        assert "protected_groups" not in unconstrained

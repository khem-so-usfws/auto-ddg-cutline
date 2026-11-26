"""Exercise continuous paired cutlines with source-pixel ownership metadata."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
from numpy.typing import NDArray

from auto_ddg_cutline.colmap import (
    PAIR_ID_BASE,
    Colmap,
    Flags,
    indexed_matches,
    ordered_matches,
)
from auto_ddg_cutline.colmap_model import Camera, Model, Photo, Track, View
from auto_ddg_cutline.geometry import (
    Coordinates,
    Frame,
    Pixels,
    Seam,
    corridor_boundaries,
    simplify_paired_route,
)
from auto_ddg_cutline.milestone2 import run_colmap
from auto_ddg_cutline.paired import Handoff, PhotoPair, clip_handoff, plan_handoffs
from auto_ddg_cutline.registration import LocalFit, LocalWarp


def landmarks() -> tuple[Coordinates, Coordinates]:
    """Provide shared pixel locations for two photos shifted horizontally."""
    xy: Coordinates = np.array(
        [(x, y) for x in range(100, 310, 14) for y in range(10, 231, 12)],
        dtype=np.float64,
    )
    return xy, xy - [85, 0]


def write_match_database(path: Path, verified: bool = True) -> None:
    """Store verified matches with database IDs reversed from photo order."""
    first: Coordinates
    second: Coordinates
    first, second = landmarks()
    # Finish the write transaction before closing the Windows file handle.
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "CREATE TABLE images (image_id INTEGER, name TEXT, camera_id INTEGER)"
        )
        connection.execute(
            "CREATE TABLE cameras (camera_id INTEGER, width INTEGER, height INTEGER)"
        )
        connection.execute("INSERT INTO cameras VALUES (1, 320, 240)")
        connection.execute(
            "CREATE TABLE keypoints (image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)"
        )
        connection.execute("CREATE TABLE descriptors (image_id INTEGER, rows INTEGER)")
        connection.execute("CREATE TABLE matches (pair_id INTEGER, rows INTEGER)")
        connection.execute(
            "CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)"
        )
        for image_id, name, xy in (
            (7, "000002.png", second),
            (25, "000001.png", first),
        ):
            connection.execute("INSERT INTO images VALUES (?, ?, 1)", (image_id, name))
            raw: bytes = (xy + 0.5).astype("<f4").tobytes()
            connection.execute(
                "INSERT INTO keypoints VALUES (?, ?, ?, ?)", (image_id, len(xy), 2, raw)
            )
            connection.execute(
                "INSERT INTO descriptors VALUES (?, ?)", (image_id, len(xy))
            )
        matches: NDArray[np.intp] = np.column_stack(
            (np.arange(len(first)), np.arange(len(first)))
        )
        connection.execute(
            "INSERT INTO matches VALUES (?, ?)", (7 * PAIR_ID_BASE + 25, len(matches))
        )
        if verified:
            connection.execute(
                "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?)",
                (
                    7 * PAIR_ID_BASE + 25,
                    len(matches),
                    2,
                    matches.astype("<u4").tobytes(),
                ),
            )


class PairedCutlineTests(unittest.TestCase):
    """Verify the cutline output independently of a local COLMAP installation."""

    def test_indexed_matches_closes_the_database(self) -> None:
        """Read a nonadjacent pair and release its SQLite handle on return."""
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = Path(directory)
            database: Path = folder / "matches.db"
            write_match_database(database)
            photos: list[Photo] = [
                Photo(folder / f"{name}.jpg", 320, 240, staged, "")
                for name, staged in (
                    ("a", "000001.png"),
                    ("b", "000003.png"),
                    ("c", "000004.png"),
                    ("d", "000002.png"),
                )
            ]
            uri: str = f"{database.as_uri()}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                with patch(
                    "auto_ddg_cutline.colmap.sqlite3.connect", return_value=connection
                ):
                    first: Coordinates
                    second: Coordinates
                    first, second = indexed_matches(database, photos, [(0, 3)])[0, 3]
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")
            expected_first: Coordinates
            expected_second: Coordinates
            expected_first, expected_second = landmarks()
            np.testing.assert_array_equal(first, expected_first)
            np.testing.assert_array_equal(second, expected_second)

    def test_benchmark_reuse_preserves_handoffs_and_supports_sparse_diagnostics(
        self,
    ) -> None:
        """Reuse native pixels and matches without extraction or annotation changes."""
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = Path(directory)
            paths: list[Path] = [folder / f"{name}.png" for name in "abc"]
            for path in paths:
                self.assertTrue(
                    cv2.imwrite(str(path), np.full((240, 320), 128, dtype=np.uint8))
                )
            database: Path = folder / "matches.db"
            write_match_database(database)
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute("INSERT INTO images VALUES (31, '000003.png', 1)")
            database_content: bytes = database.read_bytes()
            flags: Flags = Flags(
                "--FeatureExtraction.max_image_size", "", "", "", (), False, False
            )
            engine: Colmap = Colmap(folder / "colmap", {})
            fresh_jobs: Path = folder / "fresh"
            reuse_jobs: Path = folder / "reuse"
            compact_jobs: Path = folder / "compact"
            selection_jobs: Path = folder / "selection jobs"
            existing_sidecar: Path = paths[0].with_suffix(".json")
            with (
                patch("auto_ddg_cutline.milestone2.find_colmap", return_value=engine),
                patch(
                    "auto_ddg_cutline.milestone2.inspect_flags",
                    return_value=("COLMAP test", flags),
                ),
                patch(
                    "auto_ddg_cutline.milestone2.extract_and_match",
                    return_value=database,
                ) as extract,
            ):
                self.assertEqual(
                    run_colmap(
                        paths, None, "cpu", fresh_jobs, False, feature_max_edge=300
                    ),
                    0,
                )
                self.assertEqual(engine.feature_max_edge, 300)
                manifest: Path = next(fresh_jobs.glob("*/run.json"))
                self.assertEqual(
                    run_colmap(
                        paths, None, "auto", reuse_jobs, False, reuse_matches=manifest
                    ),
                    0,
                )
                existing_sidecar.write_text('{"sentinel": true}', encoding="utf-8")
                for _ in range(2):
                    self.assertEqual(
                        run_colmap(
                            paths,
                            None,
                            "auto",
                            selection_jobs,
                            False,
                            output="selection-plan",
                            reuse_matches=manifest,
                        ),
                        0,
                    )
                self.assertEqual(
                    run_colmap(
                        paths,
                        None,
                        "auto",
                        compact_jobs,
                        False,
                        output="selection-plan",
                        reuse_matches=manifest,
                        target_photo_percent=25,
                    ),
                    0,
                )
                extract.assert_called_once()
            compact_path: Path = next(compact_jobs.glob("*/selection-plan.json"))
            compact_plan: dict[str, Any] = json.loads(compact_path.read_bytes())
            self.assertEqual(compact_plan["compact_selection"]["target_photos"], 2)
            self.assertIn(
                "compact-evidence",
                compact_path.with_suffix(".html").read_text(encoding="utf-8"),
            )
            compact_run: dict[str, Any] = json.loads(
                compact_path.with_name("run.json").read_bytes()
            )
            self.assertEqual(compact_run["run_mode"], "reused")
            self.assertEqual(compact_run["target_photo_percent"], 25)
            plans: list[Path] = list(selection_jobs.glob("*/selection-plan.json"))
            self.assertEqual(len(plans), 2)
            self.assertEqual(plans[0].read_bytes(), plans[1].read_bytes())
            selection: dict[str, Any] = json.loads(plans[0].read_bytes())
            self.assertEqual(selection["approval_status"], "analyst review required")
            self.assertEqual(selection["photos"][2]["state"], "coverage review")
            html: str = plans[0].with_suffix(".html").read_text(encoding="utf-8")
            self.assertLess(html.index('id="overlaps"'), html.index('id="review"'))
            self.assertLess(html.index('id="review"'), html.index('id="sections"'))
            self.assertLess(html.index('id="omissions"'), html.index('id="photo-2"'))
            self.assertEqual(html.count('id="photo-2"'), 1)
            self.assertEqual(
                existing_sidecar.read_text(encoding="utf-8"), '{"sentinel": true}'
            )
            self.assertEqual(database.read_bytes(), database_content)
            fresh: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
            reused: dict[str, Any] = json.loads(
                next(reuse_jobs.glob("*/run.json")).read_text(encoding="utf-8")
            )
            self.assertEqual(fresh["cutline_pairs"], reused["cutline_pairs"])
            self.assertEqual(fresh["verified_pairs"], reused["verified_pairs"])
            self.assertEqual(reused["run_mode"], "reused")
            self.assertEqual(
                fresh["feature_extraction"]["effective_feature_max_edge"], 300
            )
            self.assertEqual(
                {
                    row["stage"]
                    for row in reused["stage_timings"]
                    if row["status"] == "skipped"
                },
                {"staging", "feature_extraction", "matching_and_verification"},
            )
            diagnostics_jobs: Path = folder / "diagnostics"

            def saved_model(
                engine: Colmap,
                flags: Flags,
                db: Path,
                images: Path,
                photos: dict[str, Photo],
            ) -> list[Model]:
                """Use a separate database and two views with an identical camera center."""
                self.assertNotEqual(db, database)
                with closing(sqlite3.connect(db)) as connection:
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM two_view_geometries"
                        ).fetchone()[0],
                        1,
                    )
                self.assertEqual(images, manifest.parent / "images")
                camera: Camera = Camera(320, 240, 200, np.array([160.5, 120.5]), 0.0)
                views: dict[int, View] = {
                    i: View(
                        i,
                        photo,
                        camera,
                        np.eye(3),
                        np.zeros(3),
                        {1: np.array([160.0, 120.0])},
                    )
                    for i, photo in enumerate(list(photos.values())[:2], start=1)
                }
                return [
                    Model(
                        "model-0",
                        views,
                        {1: Track(np.array([0.0, 0.0, 1.0]), 0.0, frozenset(views))},
                    )
                ]

            with (
                patch("auto_ddg_cutline.milestone2.find_colmap", return_value=engine),
                patch(
                    "auto_ddg_cutline.milestone2.inspect_flags",
                    return_value=("COLMAP test", flags),
                ),
                patch(
                    "auto_ddg_cutline.milestone2.map_matches", side_effect=saved_model
                ),
            ):
                self.assertEqual(
                    run_colmap(
                        paths,
                        None,
                        "auto",
                        diagnostics_jobs,
                        False,
                        output="sfm-diagnostics",
                        reuse_matches=manifest,
                    ),
                    0,
                )
            diagnostics: dict[str, Any] = json.loads(
                next(diagnostics_jobs.glob("*/run.json")).read_text(encoding="utf-8")
            )
            model: dict[str, Any] = diagnostics["models"][0]
            self.assertEqual(model["unregistered_images"], ["c.png"])
            self.assertEqual(
                model["near_coincident_cameras"],
                [{"images": ["a.png", "b.png"], "distance": 0.0}],
            )
            self.assertEqual(model["per_image"][0]["reprojection_p90_px"], 0.0)
            self.assertEqual(database.read_bytes(), database_content)
            self.assertTrue(
                all(not path.with_suffix(".json").exists() for path in paths[1:])
            )
            failures: Path = folder / "failures"
            with self.assertRaisesRegex(ValueError, "selection order"):
                run_colmap(
                    paths[::-1], None, "auto", failures, False, reuse_matches=manifest
                )
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute("DELETE FROM two_view_geometries")
            with self.assertRaisesRegex(ValueError, "no usable verified matches"):
                run_colmap(paths, None, "auto", failures, False, reuse_matches=manifest)
            (manifest.parent / "images" / "000003.png").unlink()
            with self.assertRaisesRegex(ValueError, "Staged image is missing"):
                run_colmap(paths, None, "auto", failures, False, reuse_matches=manifest)
            paths[0].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Source photo changed"):
                run_colmap(paths, None, "auto", failures, False, reuse_matches=manifest)

    def test_continuous_boundary_uses_opposite_sides_and_paired_clipping(self) -> None:
        """A translated pair creates one boundary whose two sides partition an outline."""
        gray: Pixels = np.full((240, 320), 128, dtype=np.uint8)
        frames: list[Frame] = [
            Frame(Path(f"{index}.jpg"), 320, 240, gray, np.empty((0, 2)), None)
            for index in (1, 2)
        ]
        handoffs: list[Handoff]
        drafts: set[PhotoPair]
        notes: list[str]
        handoffs, drafts, notes = plan_handoffs(frames, {(0, 1): landmarks()})
        self.assertEqual(len(handoffs), 1, notes)
        self.assertFalse(drafts)
        handoff: Handoff = handoffs[0]
        self.assertEqual([side.direction for side in handoff.sides], ["left", "right"])
        first: Coordinates
        second: Coordinates
        first, second = handoff.seam.points_in_source_images()
        np.testing.assert_allclose(
            first - second, np.tile([85, 0], (len(first), 1)), atol=1e-3
        )

        # The two line endpoints intersect opposite edges of a shared outline.
        for line, side, bounds in (
            (first, handoff.sides[0], (130.0, 275.0)),
            (second, handoff.sides[1], (45.0, 190.0)),
        ):
            np.testing.assert_allclose(
                [line[:, 1].min(), line[:, 1].max()], [0, 239], atol=1e-3
            )
            left: Coordinates
            right: Coordinates
            left, right = bounds
            self.assertTrue(((line[:, 0] > left) & (line[:, 0] < right)).all())
            self.assertTrue(left < side.point[0] < right)

        seam: Seam = replace(
            handoff.seam,
            points=np.array([[200.0, -10.0], [200.0, 250.0]]),
            second_points=np.array([[115.0, -4.0], [115.0, 246.0]]),
        )
        clipped: Seam | None
        reason: str
        clipped, reason = clip_handoff(seam)
        assert clipped is not None, reason
        np.testing.assert_allclose(clipped.points[:, 1], [0, 239])
        np.testing.assert_allclose(
            clipped.other_points()[:, 1], -4 + (clipped.points[:, 1] + 10) * 250 / 260
        )
        broken: Seam = replace(
            seam,
            points=np.array(
                [[140.0, 25.0], [240.0, 210.0], [240.0, 20.0], [140.0, 220.0]]
            ),
            second_points=np.array(
                [[55.0, 25.0], [155.0, 210.0], [155.0, 20.0], [55.0, 220.0]]
            ),
        )
        invalid: Seam | None
        invalid, reason = clip_handoff(broken)
        self.assertIsNone(invalid)
        self.assertIn("crosses itself", reason)

    def test_paired_handoff_follows_a_bend_in_shared_imagery(self) -> None:
        """A gap through textured terrain steers the same bend in both photos."""
        x: NDArray[np.intp]
        y: NDArray[np.intp]
        y, x = np.indices((240, 405))
        center: Coordinates = 202 + 26 * np.sin(np.pi * y / 239)
        texture: Pixels = np.where((x // 6 + y // 6) % 2 == 0, 48, 208).astype(np.uint8)
        texture[np.abs(x - center) <= 7] = 128
        frames: list[Frame] = [
            Frame(Path(name), 320, 240, gray, np.empty((0, 2)), None)
            for name, gray in (("a.png", texture[:, :320]), ("b.png", texture[:, 85:]))
        ]

        handoffs: list[Handoff]
        drafts: set[PhotoPair]
        notes: list[str]
        handoffs, drafts, notes = plan_handoffs(frames, {(0, 1): landmarks()})

        self.assertEqual(len(handoffs), 1, notes)
        self.assertFalse(drafts)
        first: Coordinates
        second: Coordinates
        first, second = handoffs[0].seam.points_in_source_images()
        self.assertGreater(len(first), 2)
        middle: float = float(np.interp(120, first[:, 1], first[:, 0]))
        self.assertAlmostEqual(middle, 228, delta=7)
        np.testing.assert_allclose(
            first - second, np.tile([85, 0], (len(first), 1)), atol=1e-3
        )

    def test_endpoints_move_to_shared_gap_and_stay_on_edges(self) -> None:
        """Both photos trace the same off-center gap from one edge to the other."""
        x: NDArray[np.intp]
        y: NDArray[np.intp]
        y, x = np.indices((240, 405))
        texture: Pixels = np.where((x // 6 + y // 6) % 2 == 0, 48, 208).astype(np.uint8)
        texture[:, 226:245] = 128
        frames: list[Frame] = [
            Frame(Path(name), 320, 240, gray, np.empty((0, 2)), None)
            for name, gray in (("a.png", texture[:, :320]), ("b.png", texture[:, 85:]))
        ]

        handoffs: list[Handoff]
        drafts: set[PhotoPair]
        notes: list[str]
        handoffs, drafts, notes = plan_handoffs(frames, {(0, 1): landmarks()})

        self.assertEqual(len(handoffs), 1, notes)
        self.assertFalse(drafts)
        self.assertEqual(handoffs[0].endpoint_mode, "movable")
        first: Coordinates
        second: Coordinates
        first, second = handoffs[0].seam.points_in_source_images()
        np.testing.assert_allclose(first[[0, -1], 1], [0, 239], atol=1e-3)
        np.testing.assert_allclose(first[[0, -1], 0], 235, atol=8)
        np.testing.assert_allclose(
            first - second, np.tile([85, 0], (len(first), 1)), atol=1e-3
        )

    def test_identical_footprints_report_ambiguous_side(self) -> None:
        """Distinguish a usable route with unclear ownership from clipping failure."""
        gray: Pixels = np.full((240, 320), 128, dtype=np.uint8)
        frames: list[Frame] = [
            Frame(Path(name), 320, 240, gray, np.empty((0, 2)), None)
            for name in ("a.png", "b.png")
        ]
        first: Coordinates
        first, _ = landmarks()

        handoffs: list[Handoff]
        notes: list[str]
        handoffs, _, notes = plan_handoffs(frames, {(0, 1): (first, first)})

        self.assertFalse(handoffs)
        self.assertEqual(len(notes), 1)
        self.assertIn(
            "count side is ambiguous because the image centers are close", notes[0]
        )

    def test_movable_endpoints_follow_boundary_under_local_warp(self) -> None:
        """Find the second frame's edge even when the local warp shifts it."""
        gray: Pixels = np.full((240, 320), 128, dtype=np.uint8)
        first: Frame = Frame(Path("a.png"), 320, 240, gray, np.empty((0, 2)), None)
        second: Frame = replace(first, path=Path("b.png"))
        overlap: Coordinates = np.array(
            [[85, 0], [319, 0], [319, 239], [85, 239]], dtype=np.float64
        )
        transform: Coordinates = np.eye(3, dtype=np.float64)
        transform[0, 2] = -85
        first_points: Coordinates
        second_points: Coordinates
        first_points, second_points = landmarks()
        shift: Coordinates = np.tile([0.0, 20.0], (len(first_points), 1))
        local: LocalFit = LocalFit(
            LocalWarp(transform, first_points, shift, 80),
            LocalWarp(np.linalg.inv(transform), second_points + shift, -shift, 80),
            20,
            0,
            20,
        )

        boundaries: Coordinates
        usable: NDArray[np.bool_]
        boundaries, usable = corridor_boundaries(
            first,
            second,
            overlap,
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([170.0, 202.0, 235.0]),
            local,
        )

        self.assertTrue(usable.all())
        np.testing.assert_allclose(boundaries[:, 0, 1], 0, atol=0.01)
        np.testing.assert_allclose(boundaries[:, 1, 1], 219, atol=0.01)
        projected: Coordinates = local.forward.map(boundaries[:, 1])
        np.testing.assert_allclose(projected[:, 1], 239, atol=0.01)

    def test_paired_simplification_keeps_a_bend_needed_by_either_view(self) -> None:
        """A straight view retains the sample that bends in the other view."""
        straight: Coordinates = np.array(
            [[0, 0], [5, 0], [10, 0], [15, 0], [20, 0]], dtype=np.float64
        )
        bent: Coordinates = straight.copy()
        bent[:, 1] = [0, 3.5, 7, 3.5, 0]

        for first, second in ((straight, bent), (bent, straight)):
            a: Coordinates
            b: Coordinates
            a, b = simplify_paired_route(first, second)
            np.testing.assert_array_equal(a, first[[0, 2, 4]])
            np.testing.assert_array_equal(b, second[[0, 2, 4]])

    def test_narrow_overlap_reports_midpoint_as_draft(self) -> None:
        """A too-narrow search corridor yields a labeled midpoint fallback."""
        gray: Pixels = np.full((240, 320), 128, dtype=np.uint8)
        frames: list[Frame] = [
            Frame(Path(name), 320, 240, gray, np.empty((0, 2)), None)
            for name in ("a.png", "b.png")
        ]
        first: Coordinates = np.array(
            [(x, y) for x in (308, 311, 314, 317) for y in range(8, 232, 16)],
            dtype=np.float64,
        )

        handoffs: list[Handoff]
        drafts: set[PhotoPair]
        notes: list[str]
        handoffs, drafts, notes = plan_handoffs(
            frames, {(0, 1): (first, first - [306, 0])}
        )

        self.assertEqual(len(handoffs), 1, notes)
        self.assertIn(handoffs[0].pair, drafts)
        self.assertEqual(handoffs[0].seam.method, "midpoint")
        self.assertEqual(handoffs[0].endpoint_mode, "fixed")
        self.assertTrue(any("Fixed endpoints" in note for note in notes))
        self.assertEqual(
            handoffs[0].seam.midpoint_fallback,
            "the shared search corridor is too small",
        )

    def test_default_writes_only_lines_and_preserves_existing_sidecar(self) -> None:
        """The COLMAP cutline mode skips reconstruction and appends a linked pair once."""
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = Path(directory)
            paths: list[Path] = [folder / "a.png", folder / "b.png"]
            for path in paths:
                self.assertTrue(
                    cv2.imwrite(str(path), np.full((240, 320), 128, dtype=np.uint8))
                )
            existing: dict[str, Any] = {
                "imagePath": "a.png",
                "imageWidth": 320,
                "imageHeight": 240,
                "shapes": [
                    {"label": "count_region", "points": [[0, 0], [10, 0], [0, 10]]}
                ],
            }
            paths[0].with_suffix(".json").write_text(
                json.dumps(existing), encoding="utf-8"
            )
            database: Path = folder / "matches.db"
            write_match_database(database)
            photo_records: list[Photo] = [
                Photo(path, 320, 240, f"{index:06}.png", "")
                for index, path in enumerate(paths, 1)
            ]
            first: Coordinates
            second: Coordinates
            first, second = ordered_matches(database, photo_records)[0, 1]
            expected_first: Coordinates
            expected_second: Coordinates
            expected_first, expected_second = landmarks()
            np.testing.assert_array_equal(first, expected_first)
            np.testing.assert_array_equal(second, expected_second)
            flags: Flags = Flags("", "", "", "", (), False, False)
            with (
                patch(
                    "auto_ddg_cutline.milestone2.find_colmap",
                    return_value=Colmap(folder / "colmap", {}),
                ),
                patch(
                    "auto_ddg_cutline.milestone2.inspect_flags",
                    return_value=("4.x", flags),
                ),
                patch(
                    "auto_ddg_cutline.milestone2.extract_and_match",
                    return_value=database,
                ),
                patch("auto_ddg_cutline.milestone2.reconstruct") as reconstruct,
            ):
                jobs: Path = folder / "jobs"
                self.assertEqual(run_colmap(paths, None, "cpu", jobs, False), 0)
                self.assertFalse(paths[1].with_suffix(".json").exists())
                self.assertEqual(run_colmap(paths, None, "cpu", jobs, True), 0)
                snapshots: list[bytes] = [
                    path.with_suffix(".json").read_bytes() for path in paths
                ]
                self.assertEqual(run_colmap(paths, None, "cpu", jobs, True), 0)
                self.assertEqual(
                    [path.with_suffix(".json").read_bytes() for path in paths],
                    snapshots,
                )
                reconstruct.assert_not_called()
                reports: list[dict[str, Any]] = [
                    json.loads(job.read_text(encoding="utf-8"))
                    for job in jobs.glob("*/run.json")
                ]
                self.assertTrue(
                    all(len(report["cutline_pairs"]) == 1 for report in reports)
                )
                for report in reports:
                    pair: dict[str, Any] = report["cutline_pairs"][0]
                    self.assertEqual(pair["seam_method"], "image")
                    self.assertEqual(pair["vertex_count"], 2)
                    self.assertEqual(pair["endpoint_mode"], "movable")
                    self.assertIsNone(pair["midpoint_fallback"])
                self.assertEqual(
                    sum(len(report["written_sidecars"]) for report in reports), 2
                )

            left: dict[str, Any]
            right: dict[str, Any]
            left, right = (json.loads(snapshot) for snapshot in snapshots)
            self.assertEqual(left["shapes"][0], existing["shapes"][0])
            a: dict[str, Any]
            b: dict[str, Any]
            a, b = left["shapes"][1], right["shapes"][0]
            self.assertEqual([a["label"], b["label"]], ["cutline", "cutline"])
            self.assertEqual(a["flags"], b["flags"])
            self.assertEqual(a["auto_ddg_handoff"]["paired_image"], "b.png")
            self.assertEqual(b["auto_ddg_handoff"]["paired_image"], "a.png")
            self.assertNotEqual(
                a["auto_ddg_handoff"]["keep_side"], b["auto_ddg_handoff"]["keep_side"]
            )

    def test_auto_retries_gpu_with_empty_verification_and_gpu_mode_stays_strict(
        self,
    ) -> None:
        """Recover from a successful GPU process whose database has no usable pairs."""
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = Path(directory)
            paths: list[Path] = [folder / "a.png", folder / "b.png"]
            for path in paths:
                self.assertTrue(
                    cv2.imwrite(str(path), np.full((240, 320), 128, dtype=np.uint8))
                )
            gpu: Path = folder / "gpu.db"
            cpu: Path = folder / "cpu.db"
            write_match_database(gpu, verified=False)
            write_match_database(cpu)
            jobs: Path = folder / "jobs"
            flags: Flags = Flags("", "", "", "", (), False, False)
            with (
                patch(
                    "auto_ddg_cutline.milestone2.find_colmap",
                    return_value=Colmap(folder / "colmap", {}),
                ),
                patch(
                    "auto_ddg_cutline.milestone2.inspect_flags",
                    return_value=("4.x", flags),
                ),
                patch(
                    "auto_ddg_cutline.milestone2.extract_and_match",
                    side_effect=[gpu, gpu, cpu, cpu],
                ) as match,
            ):
                with self.assertRaisesRegex(
                    ValueError, "No continuous paired cutlines"
                ):
                    run_colmap(paths, None, "gpu", jobs, True)
                self.assertTrue(
                    all(not path.with_suffix(".json").exists() for path in paths)
                )
                self.assertEqual(run_colmap(paths, None, "auto", jobs, True), 0)
                self.assertTrue(
                    all(path.with_suffix(".json").exists() for path in paths)
                )
                self.assertEqual(run_colmap(paths, None, "auto", jobs, False), 0)
                self.assertEqual(
                    [call.args[3] for call in match.call_args_list],
                    ["gpu", "gpu", "cpu", "gpu"],
                )
            reports: list[dict[str, Any]] = [
                json.loads(job.read_text(encoding="utf-8"))
                for job in jobs.glob("*/run.json")
            ]
            retry: dict[str, Any] = next(
                report for report in reports if report["gpu_fallback"]
            )
            self.assertEqual(retry["device_used"], "cpu")
            self.assertEqual(
                [row["usable_cutline_pairs"] for row in retry["matching_attempts"]],
                [0, 1],
            )
            self.assertEqual(retry["matching_attempts"][0]["raw_matched_pairs"], 1)
            self.assertEqual(len(retry["matching_attempts"][0]["descriptors"]), 2)
            self.assertEqual(len(retry["written_sidecars"]), 2)
            failed: dict[str, Any] = next(
                report for report in reports if report["status"] == "failure"
            )
            self.assertEqual(failed["device_used"], "gpu")
            self.assertEqual(failed["written_sidecars"], [])

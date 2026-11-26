"""Exercise native COLMAP input, sparse reconstruction output and DDG export."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np

from auto_ddg_cutline.benchmark import RunReport
from auto_ddg_cutline.colmap import (
    DEFAULT_CPU_THREADS,
    Colmap,
    ColmapError,
    Flags,
    extract_and_match,
    inspect_flags,
    map_matches,
    reconstruct,
    reconstruct_attempt,
    stage_photos,
)
from auto_ddg_cutline.colmap_model import Camera, Model, Photo, Track, View, read_model
from auto_ddg_cutline.coverage import (
    Coverage,
    Face,
    ScreenIndex,
    build_coverage,
    candidate_face,
    edge_samples,
    in_bounds,
    validate_tracks,
)
from auto_ddg_cutline.geometry import Coordinates, Frame, Pixels
from auto_ddg_cutline.local_sfm import block_drafts
from auto_ddg_cutline.match_cache import MatchAssets
from auto_ddg_cutline.milestone2 import run_colmap
from auto_ddg_cutline.paired import IndexPair
from auto_ddg_cutline.polygons import fill_polygon, mask_polygons, simple_polygon
from auto_ddg_cutline.selection import Rules
from auto_ddg_cutline.shared_patches import run_shared_patch_pilot
from auto_ddg_cutline.sidecars import SidecarEdit, plan_count_areas, write_edits

FLAGS: Flags = Flags(
    "--FeatureExtraction.max_image_size",
    "--FeatureExtraction.use_gpu",
    "--FeatureExtraction.num_threads",
    "--FeatureMatching.use_gpu",
    ("--Mapper.ba_use_gpu",),
    True,
    True,
)


def synthetic_block(folder: Path) -> Model:
    """Construct a nonplanar surface observed by three shifted cameras."""
    camera: Camera = Camera(320, 240, 210.0, np.array([160, 120]), 0.022)
    views: dict[int, View] = {}
    for index, center in ((3, -1.1), (14, 0.0), (39, 1.1)):
        photo: Photo = Photo(
            folder / f"IMG {index}.jpg", 320, 240, f"{index}.png", str(index)
        )
        views[index] = View(
            index, photo, camera, np.eye(3), np.array([-center, 0, 0]), {}
        )
    tracks: dict[int, Track] = {}
    for position, (x, y) in enumerate(
        (x, y) for x in np.arange(-3.0, 3.1, 0.19) for y in np.arange(-1.7, 1.71, 0.19)
    ):
        xyz: Coordinates = np.array([x, y, 3.9 + 0.18 * x + 0.08 * y * y])
        seen: set[int] = set()
        track_id: int = 5 + position * 3
        for index, view in views.items():
            pixels: Coordinates
            pixels, _ = view.project(xyz[None, :])
            if in_bounds(pixels, view).all():
                view.observations[track_id] = pixels[0]
                seen.add(index)
        tracks[track_id] = Track(xyz, 0.2, frozenset(seen))
    return Model("synthetic", views, tracks)


class ColmapTests(unittest.TestCase):
    """Test native coordinates, ownership rules and safe annotation writes."""

    def setUp(self) -> None:
        """Create isolated source and job paths for each test."""
        temp: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
            prefix="DDG COLMAP "
        )
        self.addCleanup(temp.cleanup)
        self.folder: Path = Path(temp.name)

    def test_local_mapper_uses_only_the_listed_photos(self) -> None:
        """Reuse one database while restricting the mapper to a local block."""
        model: Model = synthetic_block(self.folder)
        photos: dict[str, Photo] = {
            view.photo.staged_name: view.photo for view in model.views.values()
        }
        engine: Colmap = Colmap(self.folder / "colmap", {})
        database: Path = self.folder / "saved.db"
        extract_help: str = "\n".join(
            f"  {option} arg"
            for option in (
                FLAGS.image_limit,
                FLAGS.extraction_gpu,
                FLAGS.extraction_threads,
            )
        )
        for option in ("--Mapper.image_list_path", "--image_list_path"):
            with self.subTest(option=option):
                block: Path = self.folder / option.removeprefix("--")
                with patch.object(
                    engine,
                    "run",
                    side_effect=[
                        "COLMAP 4.x",
                        extract_help,
                        f"  {FLAGS.matching_gpu} arg",
                        f"  {option} arg",
                        "",
                    ],
                ) as run:
                    flags: Flags = inspect_flags(engine, self.folder)[1]
                    self.assertEqual(
                        map_matches(
                            engine,
                            flags,
                            database,
                            self.folder,
                            photos,
                            subset_job=block,
                        ),
                        [],
                    )
                args: list[str] = run.call_args.args[0]
                self.assertEqual(args[args.index("--database_path") + 1], str(database))
                image_list: Path = Path(args[args.index(option) + 1])
                self.assertEqual(
                    image_list.read_text(encoding="utf-8").splitlines(), list(photos)
                )

    def test_local_cutline_requires_tracks_seen_by_an_independent_third_camera(
        self,
    ) -> None:
        """Give the terrain planner projected SfM landmarks from multiple views."""
        model: Model = synthetic_block(self.folder)
        frames: list[Frame] = [
            Frame(
                view.photo.path,
                320,
                240,
                np.full((240, 320), 127, np.uint8),
                np.empty((0, 2), np.float64),
                None,
            )
            for view in model.views.values()
        ]
        with patch(
            "auto_ddg_cutline.local_sfm.plan_pair",
            return_value=(None, "Test planner stopped"),
        ) as plan:
            evidence: dict[IndexPair, dict[str, Any]]
            _, evidence = block_drafts({(0, 1)}, [model], frames)
            self.assertGreaterEqual(evidence[0, 1]["three_view_tracks"], 8)
            first: Coordinates
            second: Coordinates
            first, second = plan.call_args.args[2]
            self.assertEqual(len(first), evidence[0, 1]["three_view_tracks"])
            self.assertEqual(first.shape, second.shape)
            plan.reset_mock()
            model.views[39] = replace(
                model.views[39], translation=model.views[14].translation
            )
            _, evidence = block_drafts({(0, 1)}, [model], frames)
            self.assertEqual(evidence[0, 1]["three_view_tracks"], 0)
            plan.assert_not_called()

    def test_parser_preserves_empty_observations_and_source_projection(self) -> None:
        """Parse spaced names and sparse IDs and map pixel centers to DDG."""
        photo: Photo = Photo(
            self.folder / "IMG space.jpg", 120, 80, "IMG space.jpg", "a"
        )
        second: Photo = Photo(self.folder / "next.jpg", 120, 80, "next.jpg", "b")
        (self.folder / "cameras.txt").write_text(
            "17 SIMPLE_RADIAL 120 80 100 60.5 40.5 0.2\n", encoding="utf-8"
        )
        (self.folder / "images.txt").write_text(
            "# Images\n42 1 0 0 0 0 0 0 17 IMG space.jpg\n\n"
            "88 1 0 0 0 0 0 0 17 next.jpg\n60.5 40.5 99\n",
            encoding="utf-8",
        )
        (self.folder / "points3D.txt").write_text(
            "99 0 0 2 0 0 0 0.2 88 0\n", encoding="utf-8"
        )
        model: Model = read_model(
            self.folder, {photo.staged_name: photo, second.staged_name: second}
        )
        self.assertEqual(model.views[42].observations, {})
        self.assertEqual(model.views[42].photo.path.name, "IMG space.jpg")
        np.testing.assert_allclose(model.views[88].observations[99], [60, 40])
        pixels: Coordinates
        depth: Coordinates
        pixels, depth = model.views[42].project(np.array([[1.0, 0.0, 2.0]]))
        np.testing.assert_allclose(pixels, [[112.5, 40]])
        np.testing.assert_allclose(depth, [2.0])

    def test_exif_copy_uses_unrotated_native_pixels(self) -> None:
        """Keep source pixels and dimensions despite a rotated EXIF display."""
        color: Pixels = np.zeros((80, 125, 3), dtype=np.uint8)
        color[:35, :48] = (26, 84, 213)
        encoded: Pixels
        _, encoded = cv2.imencode(".jpg", color)
        exif: bytes = (
            b"Exif\0\0II" + struct.pack("<HIHHHIH", 42, 8, 1, 274, 3, 1, 6) + b"\0" * 6
        )
        original: bytes = (
            b"\xff\xd8\xff\xe1"
            + struct.pack(">H", len(exif) + 2)
            + exif
            + encoded.tobytes()[2:]
        )
        path: Path = self.folder / "Portrait orientation.jpg"
        path.write_bytes(original)
        raw: Pixels = cv2.imdecode(
            np.frombuffer(original, np.uint8),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        rotated: Pixels = cv2.imdecode(
            np.frombuffer(original, np.uint8), cv2.IMREAD_COLOR
        )
        self.assertEqual(rotated.shape[:2], (125, 80))
        photo: Photo = next(iter(stage_photos([path], self.folder / "stage").values()))
        self.assertEqual((photo.width, photo.height), (125, 80))
        staged: Pixels = cv2.imread(str(self.folder / "stage" / photo.staged_name))
        np.testing.assert_array_equal(staged, raw)
        self.assertEqual(path.read_bytes(), original)

    def test_shared_patch_pilot_compares_one_registered_block(self) -> None:
        """Report registrations and matched patch coverage from a private database."""
        model: Model = synthetic_block(self.folder)
        unseen: Photo = Photo(self.folder / "Unregistered.jpg", 320, 240, "x.png", "")
        photos: dict[str, Photo] = {
            photo.staged_name: photo
            for photo in [*(view.photo for view in model.views.values()), unseen]
        }
        database: Path = self.folder / "matches.db"
        with closing(sqlite3.connect(database)) as connection, connection:
            connection.execute("CREATE TABLE checkpoint (value TEXT)")
            connection.execute("INSERT INTO checkpoint VALUES ('original')")
        snapshot: bytes = database.read_bytes()
        plan: dict[str, Any] = {
            "counting_set": [0, 1, 3],
            "overlap_pairs": [],
            "suggested_local_sfm": [
                {"focus": [0, 1], "photos": [0, 1, 2, 3], "reason": "Flagged join"}
            ],
        }
        report: RunReport = RunReport(
            self.folder / "run.json", {"status": "complete"}, False
        )
        try:
            with (
                patch(
                    "auto_ddg_cutline.shared_patches.map_matches", return_value=[model]
                ) as mapper,
                patch(
                    "auto_ddg_cutline.shared_patches.compare_handoffs",
                    return_value=[[], []],
                ),
            ):
                pilot: dict[str, Any] = run_shared_patch_pilot(
                    plan,
                    Colmap(self.folder / "colmap", {}),
                    FLAGS,
                    MatchAssets(database, self.folder, "cpu", {}),
                    photos,
                    [],
                    Rules(),
                    self.folder,
                    report,
                )
                mapper.assert_called_once()
                self.assertEqual(
                    mapper.call_args.args[2],
                    self.folder / "global-alignment" / "database.db",
                )
                self.assertEqual(mapper.call_args.args[4], photos)
            self.assertEqual(pilot["status"], "complete")
            self.assertEqual(pilot["block_photos"], [0, 1, 2])
            self.assertEqual(pilot["unregistered_photos"], [3])
            self.assertEqual(pilot["photo_local"]["photos"], [0, 1])
            self.assertGreater(pilot["supported_patches"], 0)
            self.assertEqual(
                pilot["shared_patch"]["supported_patches_covered"],
                pilot["supported_patches"],
            )
            self.assertEqual(plan["counting_set"], [0, 1, 3])
            self.assertEqual(database.read_bytes(), snapshot)
        finally:
            report.finish()

    def test_three_view_ownership_and_shared_projection(self) -> None:
        """Generate multiple owners and project each shared cut into both views."""
        model: Model = synthetic_block(self.folder)
        coverage: Coverage = build_coverage(model)
        self.assertGreater(coverage.shared_tracks[(3, 39)], 0)
        self.assertGreaterEqual(len(coverage.polygons), 2)
        self.assertGreater(len(coverage.cutlines), 0)
        for index, polygons in coverage.polygons.items():
            for polygon in polygons:
                self.assertTrue(simple_polygon(polygon))
                self.assertTrue(in_bounds(polygon, model.views[index]).all())
        for line in coverage.cutlines:
            xyz: Coordinates = np.array(
                [model.tracks[track].xyz for track in line.track_ids]
            )
            xyz = edge_samples(xyz, closed=False)
            for index, points in zip(
                (line.first, line.second), line.points, strict=True
            ):
                projected: Coordinates
                projected, _ = model.views[index].project(xyz)
                np.testing.assert_allclose(points, projected)

    def test_rejects_depth_jumps_and_cross_chart_conflicts(self) -> None:
        """Withhold competing projected surfaces and unsupported steep jumps."""
        model: Model = synthetic_block(self.folder)
        observations: dict[int, dict[int, Coordinates]]
        observations, _ = validate_tracks(model)
        ids: tuple[int, ...] = tuple(
            sorted(track for track, views in observations.items() if len(views) == 3)[
                :3
            ]
        )
        model.tracks[ids[0]] = replace(
            model.tracks[ids[0]], xyz=np.array([1.0, 1.0, 8.5])
        )
        self.assertIsNone(candidate_face(model, ids, observations, 14))
        triangle: Coordinates = np.array([[10.0, 10, 2.0], [30, 10, 2], [10, 30, 2]])
        sampled: Coordinates = edge_samples(triangle, closed=True)[:, :2]
        index: ScreenIndex = ScreenIndex()
        self.assertTrue(index.accept(Face((1, 2, 3), {14: (sampled, 0.8)})))
        self.assertFalse(index.accept(Face((4, 5, 6), {14: (sampled + 2, 0.8)})))

    def test_hole_and_disconnected_island_stay_out_of_unsupported_pixels(self) -> None:
        """Partition hole-bearing regions without introducing owned pixels."""
        mask: Pixels = np.zeros((92, 120), dtype=np.uint8)
        mask[4:78, 3:100] = 1
        mask[24:57, 23:67] = 0
        mask[40:49, 40:51] = 1
        mask[81:91, 106:117] = 1
        polygons: list[Coordinates] = mask_polygons(mask, 10)
        output: Pixels = np.zeros_like(mask)
        for polygon in polygons:
            self.assertTrue(simple_polygon(polygon))
            fill_polygon(output, polygon, 1)
        self.assertFalse(np.any((output > 0) & (mask == 0)))
        self.assertGreater(output.sum(), mask.sum() * 0.96)

    def test_sidecar_append_repeat_and_conflict_refusal(self) -> None:
        """Preserve unrelated shapes and keep an existing edited ownership batch."""
        model: Model = synthetic_block(self.folder)
        coverage: Coverage = build_coverage(model)
        photos: dict[str, Photo] = {
            view.photo.staged_name: view.photo for view in model.views.values()
        }
        owner: View = model.views[next(iter(coverage.polygons))]
        target: Path = owner.photo.path.with_suffix(".json")
        existing: dict[str, Any] = {
            "shapes": [{"label": "manual"}],
            "imageData": "original",
            "extra": [7],
        }
        target.write_text(json.dumps(existing), encoding="utf-8")
        edits: list[SidecarEdit]
        saved: list[str]
        edits, saved = plan_count_areas(model, coverage, photos)
        self.assertFalse(saved)
        write_edits(self.folder, edits)
        contents: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(contents["extra"], [7])
        self.assertEqual(contents["imageData"], "original")
        self.assertEqual(contents["shapes"][0], existing["shapes"][0])
        self.assertEqual(contents["shapes"][1]["label"], "count_region")
        contents["shapes"][1]["points"][0][0] += 1
        target.write_text(json.dumps(contents), encoding="utf-8")
        before: bytes = target.read_bytes()
        edits, saved = plan_count_areas(model, coverage, photos)
        self.assertFalse(edits)
        self.assertIn(target.name, saved)
        self.assertEqual(target.read_bytes(), before)
        different: dict[str, Photo] = {
            key: replace(photo, digest="different") for key, photo in photos.items()
        }
        with self.assertRaisesRegex(ValueError, "already has count regions"):
            plan_count_areas(model, coverage, different)

    def test_native_limit_all_pairs_and_cpu_options(self) -> None:
        """Build native-resolution commands using path-safe argument lists."""
        photo: Photo = Photo(self.folder / "photo.jpg", 7200, 4800, "001.png", "x")
        engine: Colmap = Colmap(self.folder / "COLMAP exe", {})
        with (
            patch.object(engine, "run", return_value="") as run,
            patch("auto_ddg_cutline.colmap.verified_pairs", return_value=[]),
        ):
            reconstruct_attempt(
                engine,
                FLAGS,
                self.folder,
                "cpu",
                {photo.staged_name: photo},
                DEFAULT_CPU_THREADS,
            )
        commands: list[list[str]] = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            [cmd[0] for cmd in commands],
            ["feature_extractor", "exhaustive_matcher", "mapper"],
        )
        self.assertIn("7200", commands[0])
        self.assertIn(str(self.folder / "images"), commands[0])
        self.assertEqual(commands[0][commands[0].index(FLAGS.extraction_gpu) + 1], "0")
        self.assertEqual(
            commands[0][commands[0].index(FLAGS.extraction_threads) + 1], "1"
        )
        self.assertEqual(commands[1][commands[1].index(FLAGS.matching_gpu) + 1], "0")
        self.assertEqual(commands[2][-2:], ["--Mapper.ba_use_gpu", "0"])

    def test_pixel_percentage_extracts_each_image_size_then_matches_all(self) -> None:
        """Scale mixed source dimensions by area and keep one match database."""
        photos: dict[str, Photo] = {
            "001.png": Photo(self.folder / "wide.jpg", 5616, 3744, "001.png", "x"),
            "002.png": Photo(self.folder / "tall.jpg", 1200, 1920, "002.png", "x"),
            "003.png": Photo(self.folder / "wide2.jpg", 5616, 3744, "003.png", "x"),
        }
        engine: Colmap = Colmap(
            self.folder / "COLMAP exe", {}, feature_pixel_percent=25
        )
        flags: Flags = replace(FLAGS, extraction_image_list=True)
        with patch.object(engine, "run", return_value="") as run:
            database: Path = extract_and_match(
                engine, flags, self.folder, "cpu", photos, 1
            )
        commands: list[list[str]] = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            [cmd[0] for cmd in commands],
            ["feature_extractor", "feature_extractor", "exhaustive_matcher"],
        )
        for command, edge, names in (
            (commands[0], "960", ["002.png"]),
            (commands[1], "2808", ["001.png", "003.png"]),
        ):
            self.assertEqual(command[command.index(flags.image_limit) + 1], edge)
            listing: Path = Path(command[command.index("--image_list_path") + 1])
            self.assertEqual(listing.read_text(encoding="utf-8").splitlines(), names)
        for command in commands:
            self.assertEqual(
                command[command.index("--database_path") + 1], str(database)
            )

        engine.feature_pixel_percent = 100
        with patch.object(engine, "run", return_value="") as run:
            extract_and_match(engine, FLAGS, self.folder, "gpu", photos, 1)
        native: list[str] = run.call_args_list[0].args[0]
        self.assertEqual(run.call_count, 2)
        self.assertNotIn("--image_list_path", native)
        self.assertEqual(native[native.index(flags.image_limit) + 1], "5616")

    def test_auto_gpu_retry_and_explicit_gpu_failure(self) -> None:
        """Fall back for an actual device error only when auto is selected."""
        engine: Colmap = Colmap(self.folder / "colmap", {})
        failure: ColmapError = ColmapError(
            "feature_extractor", "No CUDA devices found", self.folder, 1
        )
        with patch(
            "auto_ddg_cutline.colmap.reconstruct_attempt",
            side_effect=[failure, ([], [])],
        ) as run:
            used: str
            reason: str | None
            _, _, used, reason = reconstruct(engine, FLAGS, self.folder, "auto", {}, 2)
            self.assertEqual(used, "cpu")
            self.assertIn("No CUDA", reason or "")
            self.assertEqual(
                [call.args[3] for call in run.call_args_list], ["gpu", "cpu"]
            )
            self.assertEqual(run.call_args.args[-1], 2)
        with patch(
            "auto_ddg_cutline.colmap.reconstruct_attempt", side_effect=failure
        ) as run:
            with self.assertRaises(ColmapError):
                reconstruct(engine, FLAGS, self.folder, "gpu", {}, DEFAULT_CPU_THREADS)
            run.assert_called_once()

    def test_native_exit_code_is_preserved_without_an_error_message(self) -> None:
        """Report a Windows crash code even when COLMAP logs only startup."""
        engine: Colmap = Colmap(self.folder / "colmap", {})
        log: Path = self.folder / "extract.log"
        output: str = "Creating SIFT CPU feature extractor\n"

        def stream(argv: list[str], path: Path) -> tuple[str, int, bool]:
            """Simulate the result of streaming a native Windows process."""
            path.write_text(output, encoding="utf-8")
            return output, 0xC0000005, False

        with (
            patch.object(engine, "stream", side_effect=stream),
            patch("auto_ddg_cutline.colmap.os.name", "nt"),
            self.assertRaisesRegex(
                ColmapError, r"exit code 3221225477 \(0xC0000005\)"
            ) as failure,
        ):
            engine.run(["feature_extractor"], log)
        self.assertEqual(failure.exception.returncode, 0xC0000005)
        self.assertEqual(log.read_text(encoding="utf-8"), output)

    def test_end_to_end_run_manifest_and_write_guard(self) -> None:
        """Produce CLI diagnostics on preview and append only on explicit write."""
        model: Model = synthetic_block(self.folder)
        photos: dict[str, Photo] = {}
        original: bytes = b"sample image"
        for view in model.views.values():
            view.photo.path.write_bytes(original)
            photos[view.photo.staged_name] = replace(
                view.photo, digest=hashlib.sha256(original).hexdigest()
            )
        paths: list[Path] = [photo.path for photo in photos.values()]
        jobs: Path = self.folder / "job logs"
        engine: Colmap = Colmap(self.folder / "colmap", {})
        with (
            patch("auto_ddg_cutline.milestone2.find_colmap", return_value=engine),
            patch(
                "auto_ddg_cutline.milestone2.inspect_flags", return_value=("4.x", FLAGS)
            ),
            patch("auto_ddg_cutline.milestone2.stage_photos", return_value=photos),
            patch(
                "auto_ddg_cutline.milestone2.reconstruct",
                return_value=([model], [], "cpu", None),
            ),
        ):
            self.assertEqual(
                run_colmap(paths, None, "cpu", jobs, False, output="sparse-regions"), 0
            )
            self.assertFalse(list(self.folder.glob("*.json")))
            self.assertEqual(
                run_colmap(paths, None, "cpu", jobs, True, output="sparse-regions"), 0
            )
        self.assertGreater(len(list(self.folder.glob("*.json"))), 1)
        summaries: list[dict[str, Any]] = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in jobs.glob("*/run.json")
        ]
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(summary["ownership"] for summary in summaries))
        self.assertTrue(
            next(summary for summary in summaries if summary["write_requested"])[
                "written_sidecars"
            ]
        )

"""Exercise the survey command line and the append-only export contract."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from auto_ddg_cutline.geometry import Coordinates, Frame, Seam
from auto_ddg_cutline.sequence import TripleReview, review_triple
from auto_ddg_cutline.sidecars import SidecarEdit, write_edits

Image: TypeAlias = NDArray[np.uint8]


def scene() -> Image:
    """Generate textured terrain with features at several scales."""
    rng: np.random.Generator = np.random.default_rng(7193)
    image: Image = rng.integers(65, 185, (950, 1160), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    for _ in range(950):
        point: tuple[int, int] = (int(rng.integers(1160)), int(rng.integers(950)))
        radius: int = int(rng.integers(3, 20))
        color: int = int(rng.integers(30, 230))
        cv2.circle(image, point, radius, color, -1, cv2.LINE_AA)
    return image


def photo(folder: Path, name: str, image: Image) -> None:
    """Write a source image to the test survey folder."""
    success: bool
    data: Image
    success, data = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Could not encode the test photo.")
    (folder / name).write_bytes(data.tobytes())


def run_cli(folder: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the same module entry point used from Anaconda Prompt."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_ddg_cutline",
            str(folder),
            "--backend",
            "opencv",
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def read_json(path: Path) -> dict[str, Any]:
    """Return the LabelMe document for JSON assertions."""
    return json.loads(path.read_text(encoding="utf-8"))


class MilestoneTests(unittest.TestCase):
    """Exercise cases that could otherwise produce incorrect paired cutlines."""

    def setUp(self) -> None:
        """Create a clean survey folder for each test."""
        temporary: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
            prefix="DDG aéreo "
        )
        self.addCleanup(temporary.cleanup)
        self.folder: Path = Path(temporary.name)
        self.scene: Image = scene()

    def make_triplet(self) -> None:
        """Create three diagonally overlapping crops with a shared central patch."""
        photo(self.folder, "IMG_1.png", self.scene[30:590, 40:740])
        photo(self.folder, "IMG_2.png", self.scene[165:725, 205:905])
        photo(self.folder, "IMG_10.png", self.scene[300:860, 370:1070])

    def test_preview_pair_projection_append_and_edit_preservation(self) -> None:
        """Preview stays read-only and repeated export keeps manual DDG changes."""
        self.make_triplet()
        sidecar: Path = self.folder / "IMG_2.json"
        existing: dict[str, Any] = {
            "shapes": [{"label": "manual", "points": [[12, 13]]}],
            "survey_metadata": {"reviewer": "Test"},
        }
        original: str = json.dumps(existing)
        sidecar.write_text(original, encoding="utf-8")
        review_dir: Path = self.folder / "review"
        preview: subprocess.CompletedProcess[str] = run_cli(
            self.folder,
            "--select",
            "1-3",
            "--method",
            "midpoint",
            "--max-edge",
            "512",
            "--review-dir",
            str(review_dir),
        )
        self.assertEqual(preview.returncode, 0, preview.stderr)
        self.assertIn("Estimated three-photo overlap", preview.stdout)
        self.assertIn("no separate first-to-last cutline", preview.stdout)
        self.assertIn("Suggested count side", preview.stdout)
        self.assertTrue((review_dir / "cutline-review.png").is_file())
        self.assertEqual(sidecar.read_text(encoding="utf-8"), original)
        self.assertEqual(
            sorted(p.name for p in self.folder.glob("*.json")), ["IMG_2.json"]
        )
        self.assertFalse((self.folder / ".auto-ddg-cutline.lock").exists())

        write: subprocess.CompletedProcess[str] = run_cli(
            self.folder,
            "--select",
            "1-3",
            "--method",
            "midpoint",
            "--max-edge",
            "512",
            "--write",
        )
        self.assertEqual(write.returncode, 0, write.stderr)
        self.assertIn("Updated 3 JSON sidecar(s)", write.stdout)
        first: dict[str, Any] = read_json(self.folder / "IMG_1.json")
        middle: dict[str, Any] = read_json(sidecar)
        self.assertEqual(first["imageWidth"], 700)
        self.assertEqual(first["imageHeight"], 560)
        self.assertEqual(len(first["shapes"]), 1)
        self.assertEqual(middle["survey_metadata"], existing["survey_metadata"])
        self.assertEqual(middle["shapes"][0], existing["shapes"][0])
        self.assertEqual(first["shapes"][0]["shape_type"], "line")
        for first_point, middle_point in zip(
            first["shapes"][0]["points"], middle["shapes"][1]["points"], strict=True
        ):
            np.testing.assert_allclose(
                np.array(first_point) - middle_point, [165, 135], atol=2
            )

        middle["shapes"][1]["points"][0][0] += 5
        updated: str = json.dumps(middle)
        sidecar.write_text(updated, encoding="utf-8")
        repeat: subprocess.CompletedProcess[str] = run_cli(
            self.folder, "--select", "3-1", "--write"
        )
        self.assertEqual(repeat.returncode, 0, repeat.stderr)
        self.assertIn("pair already present", repeat.stdout)
        self.assertEqual(sidecar.read_text(encoding="utf-8"), updated)

    def test_image_path_supports_rotation_scale_and_perspective(self) -> None:
        """Register oblique-style frames and put the same curve in both photos."""
        first: Image = self.scene[40:660, 60:860]
        left: NDArray[np.float32] = np.array(
            [[240, 105], [985, 152], [950, 765], [196, 700]],
            dtype=np.float32,
        )
        right: NDArray[np.float32] = np.array(
            [[0, 0], [699, 0], [699, 559], [0, 559]], dtype=np.float32
        )
        transform: NDArray[np.float64] = cv2.getPerspectiveTransform(left, right)
        second: Image = cv2.warpPerspective(self.scene, transform, (700, 560))
        third: Image = self.scene[270:870, 415:1115]
        for index, frame in enumerate((first, second, third), start=1):
            photo(self.folder, f"oblique_{index}.png", frame)
        result: subprocess.CompletedProcess[str] = run_cli(
            self.folder, "--all", "--write"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("image-based", result.stdout)
        first_shape: dict[str, Any] = read_json(self.folder / "oblique_1.json")[
            "shapes"
        ][0]
        second_shape: dict[str, Any] = read_json(self.folder / "oblique_2.json")[
            "shapes"
        ][0]
        self.assertEqual(first_shape["shape_type"], "linestrip")
        source: NDArray[np.float64] = (
            np.array(first_shape["points"], dtype=np.float64) + [60, 40]
        ).reshape(-1, 1, 2)
        expected: NDArray[np.float64] = cv2.perspectiveTransform(
            source, transform
        ).reshape(-1, 2)
        np.testing.assert_allclose(expected, second_shape["points"], atol=2)

    def test_nested_pair_exports_draft_and_blank_frame_stays_unresolved(self) -> None:
        """Plausible uncertain pairs are viewable and a blank frame stays unresolved."""
        photo(self.folder, "1.png", self.scene[30:630, 40:840])
        photo(self.folder, "2.png", self.scene[130:510, 200:675])
        photo(self.folder, "3.png", np.full((500, 600), 128, dtype=np.uint8))
        result: subprocess.CompletedProcess[str] = run_cli(
            self.folder, "--all", "--write"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRAFT 1.png / 2.png", result.stdout)
        self.assertIn("UNRESOLVED 2.png / 3.png", result.stdout)
        shape: dict[str, Any] = read_json(self.folder / "1.json")["shapes"][0]
        self.assertTrue(shape["flags"]["auto_ddg_draft"])
        self.assertFalse((self.folder / "3.json").exists())

    def test_malformed_sidecar_blocks_the_export(self) -> None:
        """A broken selected sidecar cannot be overwritten during a batch append."""
        self.make_triplet()
        path: Path = self.folder / "IMG_10.json"
        path.write_text("broken", encoding="utf-8")
        result: subprocess.CompletedProcess[str] = run_cli(
            self.folder, "--all", "--write"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not valid JSON", result.stderr)
        self.assertEqual(path.read_text(encoding="utf-8"), "broken")
        self.assertFalse((self.folder / "IMG_1.json").exists())

    def test_changed_snapshots_are_left_alone(self) -> None:
        """A sidecar modified between planning and writing aborts the batch."""
        first: Path = self.folder / "1.json"
        second: Path = self.folder / "2.json"
        second.write_bytes(b"later DDG edit")
        with self.assertRaisesRegex(OSError, "changed during preview"):
            write_edits(
                self.folder,
                [
                    SidecarEdit(first, None, b"first proposal"),
                    SidecarEdit(second, b"earlier snapshot", b"second proposal"),
                ],
            )
        self.assertFalse(first.exists())
        self.assertEqual(second.read_bytes(), b"later DDG edit")

    def test_three_way_conflict_is_flagged_even_when_alignment_agrees(self) -> None:
        """Three valid pair cuts can produce a cycle with no countable photo."""
        gray: Image = np.zeros((100, 100), dtype=np.uint8)
        points: Coordinates = np.empty((0, 2), dtype=np.float64)
        transform: Coordinates = np.eye(3)
        frames: list[Frame] = [
            Frame(self.folder / name, 100, 100, gray, points, None)
            for name in ("a.png", "b.png", "c.png")
        ]
        seams: list[Seam] = [
            Seam(
                first=frames[first],
                second=frames[second],
                to_first=transform,
                to_second=transform,
                overlap=frames[first].footprint,
                normal=np.array([1.0, 0.0]),
                tangent=np.array([0.0, 1.0]),
                points=np.array([[x, 0.0], [x, 99.0]]),
                inliers=40,
                match_fraction=1.0,
                supported_sections=4,
                error=0.0,
                warnings=(),
                method="midpoint",
            )
            for first, second, x in ((0, 1, 35.0), (1, 2, 60.0), (0, 2, 10.0))
        ]
        review: TripleReview | None = review_triple(*seams)
        assert review is not None
        self.assertTrue(review.uncertain)
        self.assertIn("conflicting handoffs", review.message)


if __name__ == "__main__":
    unittest.main()

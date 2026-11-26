"""Exercise calibrated frustums and bounded dense terrain from saved poses."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from auto_ddg_cutline.benchmark import RunReport
from auto_ddg_cutline.colmap_model import Camera, Model, Photo, Track, View
from auto_ddg_cutline.geometry import Coordinates, Pixels
from auto_ddg_cutline.survey_dense import (
    DENSE_CACHE_NAME,
    DENSE_MAX_EDGE,
    DENSE_VIEW_LIMIT,
    DepthView,
    build_dense_layer,
    dense_cutline_points,
    depth_surface,
    frustum_corners,
    pilot_views,
    read_depth,
)


def plane_model(folder: Path, count: int = 3) -> Model:
    """Create translated cameras with overlapping observations of a plane."""
    camera: Camera = Camera(20, 16, 20.0, np.array([10.0, 8.0]), 0.02)
    landmarks: Coordinates = np.array(
        [[x, y, 8] for y in (-1, 0, 1) for x in (-1, 0, 1)], dtype=float
    )
    views: dict[int, View] = {}
    for index in range(count):
        photo: Photo = Photo(folder / f"{index}.jpg", 20, 16, f"{index}.png", "test")
        position: Coordinates = np.array([-0.25 * index, 0.0, 0.0])
        view: View = View(index, photo, camera, np.eye(3), position, {})
        observations: Coordinates = view.project(landmarks)[0]
        views[index] = View(
            index, photo, camera, view.rotation, position, dict(enumerate(observations))
        )
    return Model(
        "block",
        views,
        {i: Track(xyz, 0.1, frozenset(views)) for i, xyz in enumerate(landmarks)},
    )


def save_depth(path: Path, data: NDArray[np.float32]) -> None:
    """Write a test depth map in COLMAP's documented binary format."""
    height: int
    width: int
    height, width = data.shape
    path.write_bytes(f"{width}&{height}&1&".encode() + data.astype("<f4").tobytes())


def test_frustum_projects_to_distorted_photo_edges(tmp_path: Path) -> None:
    """Use the recorded intrinsics, pose and stored pixel boundary convention."""
    first: View = plane_model(tmp_path).views[1]
    basis: Coordinates = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    view: View = View(first.id, first.photo, first.camera, basis, first.translation, {})
    corners: Coordinates = frustum_corners(view, 4.0)
    expected: Coordinates = np.array(
        [[-0.5, -0.5], [19.5, -0.5], [19.5, 15.5], [-0.5, 15.5]]
    )
    np.testing.assert_allclose(view.project(corners)[0], expected, atol=1e-6)
    np.testing.assert_allclose(view.project(corners)[1], 4.0)


def test_dense_depth_support_preserves_holes_and_projects_paired_lines(
    tmp_path: Path,
) -> None:
    """Keep only shared depth and place a line when both rays see that surface."""
    model: Model = plane_model(tmp_path)
    depth: NDArray[np.float32] = np.full((16, 20), 8.0, dtype=np.float32)
    path: Path = tmp_path / "reference.geometric.bin"
    save_depth(path, depth)
    np.testing.assert_array_equal(read_depth(path), depth)
    reference: DepthView = DepthView(
        model.views[0], np.array([20.0, 20.0]), np.array([10.0, 8.0]), depth.copy()
    )
    reference.depth[7:9, 9:11] = 0
    second: DepthView = DepthView(
        model.views[1], reference.focal, reference.principal, depth
    )
    closer: DepthView = DepthView(
        model.views[2], reference.focal, reference.principal, depth / 2
    )
    pixels: Pixels = np.full((16, 20, 3), (20, 60, 200), dtype=np.uint8)
    layer: dict[str, Any] = depth_surface(reference, [second, closer], pixels)
    assert layer["faces"]
    assert all(face[3] == [0, 1] for face in layer["faces"])
    assert layer["colors"][0] == [200, 60, 20]

    xyz: Coordinates = np.array([[-1.0, 0, 8.0], [0.0, 0, 8.0], [1.0, 0, 8.0]])
    pixels_a: Coordinates = model.views[0].project(xyz)[0]
    pixels_b: Coordinates = model.views[1].project(xyz)[0]
    hits: Coordinates
    valid: NDArray[np.bool_]
    hits, valid = dense_cutline_points(
        layer, model.views[0], model.views[1], pixels_a, pixels_b, 0.25
    )
    assert valid.tolist() == [True, False, True]
    np.testing.assert_allclose(hits[valid], xyz[valid], atol=1e-6)
    assert not dense_cutline_points(
        layer, model.views[0], model.views[1], pixels_a, pixels_b + 2.0, 0.25
    )[1].any()

    # A closer surface with no support for Photo B must block the shared patch.
    first_extra: int = len(layer["vertices"])
    layer["vertices"].extend([[-2, -2, 4], [2, -2, 4], [0, 2, 4]])
    layer["faces"].append([first_extra, first_extra + 1, first_extra + 2, [0, 2]])
    assert not dense_cutline_points(
        layer, model.views[0], model.views[1], pixels_a, pixels_b, 0.25
    )[1].any()


def test_dense_pilot_bounds_colmap_and_reuses_successful_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the full pipeline with simulated native commands and real file decoding."""
    model: Model = plane_model(tmp_path, count=DENSE_VIEW_LIMIT + 1)
    reference: str = model.views[2].photo.path.name
    selected: list[View] = pilot_views(model, reference)
    staged: list[str] = [view.photo.staged_name for view in selected]
    assert len(staged) == DENSE_VIEW_LIMIT
    assert staged[0] == "2.png"
    model_dir: Path = tmp_path / "saved-model"
    model_dir.mkdir()
    engine: Mock = Mock()
    monkeypatch.setattr("auto_ddg_cutline.survey_dense.find_colmap", lambda _: engine)

    def run(command: list[str], log: Path) -> str:
        """Simulate COLMAP's subset, rectification and geometric depth artifacts."""
        step: str = command[0]
        flags: dict[str, str] = dict(zip(command[1::2], command[2::2], strict=True))
        log.touch()
        if step == "image_deleter":
            assert Path(flags["--input_path"]) == model_dir
            assert Path(flags["--output_path"]) != model_dir
            assert Path(flags["--image_names_path"]).read_text() == "8.png\n"
        elif step == "image_undistorter":
            assert flags["--max_image_size"] == str(DENSE_MAX_EDGE)
            output: Path = Path(flags["--output_path"])
            (output / "stereo" / "depth_maps").mkdir(parents=True)
            (output / "images").mkdir()
            assert cv2.imwrite(
                str(output / "images" / staged[0]),
                np.full((16, 20, 3), 240, dtype=np.uint8),
            )
        elif step == "patch_match_stereo":
            assert flags["--PatchMatchStereo.geom_consistency"] == "true"
            stereo: Path = Path(flags["--workspace_path"]) / "stereo"
            config: list[str] = (stereo / "patch-match.cfg").read_text().splitlines()
            assert config[::2] == staged
            for name, sources in zip(config[::2], config[1::2], strict=True):
                assert set(sources.split(", ")) == set(staged) - {name}
                save_depth(
                    stereo / "depth_maps" / f"{name}.geometric.bin",
                    np.full((16, 20), 8, dtype=np.float32),
                )
        elif step == "model_converter":
            rectified: Path = Path(flags["--output_path"])
            (rectified / "cameras.txt").write_text("1 PINHOLE 20 16 20 20 10 8\n")
            (rectified / "images.txt").write_text(
                "".join(
                    f"{view.id} 1 0 0 0 {view.translation[0]} 0 0 1 {view.photo.staged_name}\n\n"
                    for view in selected
                )
            )
        return ""

    engine.run.side_effect = run
    report: RunReport = RunReport(tmp_path / "run.json", {"status": "complete"}, False)
    try:
        layer: dict[str, Any] = build_dense_layer(
            model, reference, model_dir, tmp_path, None, report
        )
        saved: Path = model_dir / DENSE_CACHE_NAME
        contents: bytes = saved.read_bytes()
        assert layer["focus"] == reference and layer["faces"]
        assert len(layer["photos"]) == DENSE_VIEW_LIMIT
        assert engine.run.call_count == 4
        assert (
            build_dense_layer(model, reference, model_dir, tmp_path, None, report)
            == layer
        )
        assert engine.run.call_count == 4

        engine.run.side_effect = OSError("CUDA unavailable")
        with pytest.raises(OSError, match="CUDA unavailable"):
            build_dense_layer(model, "0.jpg", model_dir, tmp_path, None, report)
        assert saved.read_bytes() == contents
    finally:
        report.finish()

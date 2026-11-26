"""Build a small photo-colored terrain layer from COLMAP dense stereo."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

import cv2
import numpy as np
from numpy.typing import NDArray

from .benchmark import RunReport
from .colmap import Colmap, find_colmap
from .colmap_model import Model, View, records
from .geometry import Coordinates, Pixels
from .launcher_workflow import read_json
from .survey_viewer import write_overview_file

DENSE_CACHE_NAME: str = "overview-dense-v1.json"
DENSE_MAX_EDGE: int = 1400
DENSE_VIEW_LIMIT: int = 8
DENSE_GRID_TARGET: int = 5500


def ray_coordinates(view: View, points: Coordinates) -> Coordinates:
    """Convert stored zero-based pixel centers to undistorted camera rays."""
    focal: float = view.camera.focal
    cx: float
    cy: float
    cx, cy = view.camera.principal.tolist()
    intrinsics: Coordinates = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
    distortion: Coordinates = np.array([view.camera.radial, 0, 0, 0])
    return cv2.undistortPoints(
        (points + 0.5).reshape(-1, 1, 2), intrinsics, distortion
    ).reshape(-1, 2)


def frustum_corners(view: View, depth: float) -> Coordinates:
    """Return the four calibrated window corners at an illustrative depth."""
    width: int = view.camera.width
    height: int = view.camera.height
    # Outer pixel boundaries correspond to COLMAP coordinates 0 and width/height.
    corners: Coordinates = (
        np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
        - 0.5
    )
    rays: Coordinates = np.column_stack((ray_coordinates(view, corners), np.ones(4)))
    return view.center + depth * (rays @ view.rotation)


@dataclass(frozen=True)
class DepthView:
    """One undistorted camera and its dense camera-axis depth raster."""

    view: View
    focal: Coordinates
    principal: Coordinates
    depth: NDArray[np.float32]

    def project(self, points: Coordinates) -> tuple[Coordinates, Coordinates]:
        """Project model positions into the dense raster's corner coordinates."""
        camera: Coordinates = points @ self.view.rotation.T + self.view.translation
        z: Coordinates = camera[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels: Coordinates = (
                camera[:, :2] / z[:, None] * self.focal + self.principal
            )
        return pixels, z

    def supports(self, points: Coordinates) -> NDArray[np.bool_]:
        """Check whether the observed surface agrees with each projected point."""
        pixels: Coordinates
        z: Coordinates
        pixels, z = self.project(points)
        height: int
        width: int
        height, width = self.depth.shape
        result: NDArray[np.bool_] = (
            np.isfinite(pixels).all(axis=1)
            & (z > 0)
            & (pixels >= 0).all(axis=1)
            & (pixels < [width, height]).all(axis=1)
        )
        active: NDArray[np.intp] = np.flatnonzero(result)
        xy: NDArray[np.intp] = np.floor(pixels[active]).astype(np.intp)
        observed: NDArray[np.float32] = self.depth[xy[:, 1], xy[:, 0]]
        result[active] = (
            (observed > 0)
            & np.isfinite(observed)
            & (np.abs(observed - z[active]) <= 0.035 * z[active])
        )
        return result


def read_depth(path: Path) -> NDArray[np.float32]:
    """Read COLMAP's little-endian single-channel depth array."""
    with path.open("rb") as stream:
        header: bytearray = bytearray()
        while len(header) < 48 and header.count(b"&") < 3:
            byte: bytes = stream.read(1)
            if not byte:
                break
            header.extend(byte)
        if header.count(b"&") != 3:
            raise ValueError(f"Invalid COLMAP depth header: {path.name}")
        try:
            width: int
            height: int
            channels: int
            width, height, channels = map(int, header[:-1].split(b"&"))
        except ValueError as error:
            raise ValueError(f"Invalid COLMAP depth dimensions: {path.name}") from error
        payload: bytes = stream.read()
    if channels != 1 or min(width, height) < 1 or len(payload) != width * height * 4:
        raise ValueError(f"Invalid COLMAP depth data: {path.name}")
    return np.frombuffer(payload, dtype="<f4").reshape(height, width)


def pilot_views(model: Model, focus: str) -> list[View]:
    """Choose a reference view and its best-supported neighbors in one block."""
    first: View | None = next(
        (view for view in model.views.values() if view.photo.path.name == focus), None
    )
    if first is None:
        return []
    visible: set[int] = first.observations.keys() & model.tracks.keys()
    ranked: list[tuple[int, View]] = sorted(
        (
            (len(visible & view.observations.keys()), view)
            for view in model.views.values()
            if view.id != first.id
        ),
        key=lambda item: (-item[0], item[1].photo.path.name.casefold()),
    )
    neighbors: list[View] = [view for count, view in ranked if count >= 8]
    return [first, *neighbors[: DENSE_VIEW_LIMIT - 1]]


def rectified_views(
    model_dir: Path, workspace: Path, selected: list[View]
) -> dict[int, DepthView]:
    """Read undistorted pinhole intrinsics and geometric depth for the pilot."""
    cameras: dict[int, tuple[int, int, Coordinates, Coordinates]] = {}
    line: str
    for line in filter(None, records(model_dir / "cameras.txt")):
        camera_id: str
        model: str
        width: str
        height: str
        parameters: list[str]
        camera_id, model, width, height, *parameters = line.split()
        numbers: list[float] = list(map(float, parameters))
        if model == "PINHOLE" and len(numbers) == 4:
            focal: Coordinates = np.array(numbers[:2])
            principal: Coordinates = np.array(numbers[2:])
        elif model == "SIMPLE_PINHOLE" and len(numbers) == 3:
            focal = np.array([numbers[0], numbers[0]])
            principal = np.array(numbers[1:])
        else:
            raise ValueError(
                f"Dense stereo returned an unsupported camera model: {model}."
            )
        cameras[int(camera_id)] = (int(width), int(height), focal, principal)

    by_name: dict[str, View] = {view.photo.staged_name: view for view in selected}
    result: dict[int, DepthView] = {}
    poses: Iterator[str] = records(model_dir / "images.txt")
    for line in poses:
        if not line:
            continue
        parts: list[str] = line.split(maxsplit=9)
        next(poses, None)  # The next line contains sparse tracks, possibly empty.
        view: View | None = by_name.get(parts[9])
        if view is None:
            continue
        path: Path = workspace / "stereo" / "depth_maps" / f"{parts[9]}.geometric.bin"
        if not path.is_file():
            continue
        original_width: int
        original_height: int
        original_width, original_height, focal, principal = cameras[int(parts[8])]
        depth: NDArray[np.float32] = read_depth(path)
        scale: Coordinates = np.array(depth.shape[::-1]) / [
            original_width,
            original_height,
        ]
        result[view.id] = DepthView(view, focal * scale, principal * scale, depth)
    return result


def depth_surface(
    reference: DepthView, others: list[DepthView], photo: Pixels
) -> dict[str, Any]:
    """Triangulate the focus photo's dense depth with multi-view support.

    Args:
        reference: Depth raster defining the area of the pilot.
        others: Neighboring depth rasters used to reject occlusions and holes.
        photo: Undistorted color pixels for the reference view.

    Returns:
        Compact world-space triangles, view support, and reference photo colors.

    Raises:
        ValueError: No neighboring depth raster is available to verify the surface.
    """
    if not others:
        raise ValueError("Dense terrain needs depth from a neighboring view.")
    height: int
    width: int
    height, width = reference.depth.shape
    step: int = max(1, math.ceil(math.sqrt(width * height / DENSE_GRID_TARGET)))
    xs: NDArray[np.intp] = np.arange(0, width, step, dtype=np.intp)
    ys: NDArray[np.intp] = np.arange(0, height, step, dtype=np.intp)
    x: NDArray[np.intp]
    y: NDArray[np.intp]
    x, y = np.meshgrid(xs, ys)
    z: Coordinates = reference.depth[y, x].ravel().astype(np.float64)
    grid: Coordinates = np.column_stack((x.ravel(), y.ravel()))
    normalized: Coordinates = (grid + 0.5 - reference.principal) / reference.focal
    local: Coordinates = np.column_stack((normalized, np.ones(len(z)))) * z[:, None]
    world: Coordinates = reference.view.center + local @ reference.view.rotation

    ids: NDArray[np.intp] = np.arange(len(z), dtype=np.intp).reshape(x.shape)
    tl: NDArray[np.intp] = ids[:-1, :-1].ravel()
    tr: NDArray[np.intp] = ids[:-1, 1:].ravel()
    br: NDArray[np.intp] = ids[1:, 1:].ravel()
    bl: NDArray[np.intp] = ids[1:, :-1].ravel()
    faces: NDArray[np.intp] = np.vstack(
        (np.column_stack((tl, tr, br)), np.column_stack((tl, br, bl)))
    )
    depths: Coordinates = z[faces]
    good: NDArray[np.bool_] = (
        np.isfinite(depths).all(axis=1)
        & (depths.min(axis=1) > 0)
        & (np.ptp(depths, axis=1) <= 0.075 * depths.min(axis=1))
    )
    faces = faces[good]
    # Test corners and the center, so a triangle cannot assert support through
    # a foreground occlusion or across a large depth discontinuity.
    xyz: Coordinates = world[faces]
    centers: Coordinates = xyz.mean(axis=1)
    samples: Coordinates = np.concatenate((xyz, centers[:, None]), axis=1)
    support: NDArray[np.bool_] = np.column_stack(
        [
            view.supports(samples.reshape(-1, 3)).reshape(-1, 4).all(axis=1)
            for view in others
        ]
    )
    # Check the reference center too, in case the sampling grid straddles a hole.
    retain: NDArray[np.bool_] = reference.supports(centers) & support.any(axis=1)
    faces = faces[retain]
    support = support[retain]
    vertices: NDArray[np.intp]
    remapped: NDArray[np.intp]
    vertices, remapped = np.unique(faces, return_inverse=True)
    pixels: NDArray[np.intp] = np.rint(grid[faces].mean(axis=1)).astype(np.intp)
    if photo.shape[:2] != reference.depth.shape:
        photo = cv2.resize(photo, (width, height), interpolation=cv2.INTER_AREA)
    colors: Pixels = photo[pixels[:, 1], pixels[:, 0], ::-1]
    return {
        "vertices": world[vertices].tolist(),
        "faces": [
            [
                *face,
                [
                    reference.view.id,
                    *(
                        view.view.id
                        for view, hit in zip(others, flags, strict=True)
                        if hit
                    ),
                ],
            ]
            for face, flags in zip(
                remapped.reshape(-1, 3).tolist(), support.tolist(), strict=True
            )
        ],
        "colors": colors.tolist(),
    }


def build_dense_layer(
    model: Model,
    focus: str,
    model_dir: Path,
    image_folder: Path,
    executable: str | None,
    report: RunReport,
) -> dict[str, Any]:
    """Run a bounded dense pilot from fixed COLMAP poses and publish its cache."""
    selected: list[View] = pilot_views(model, focus)
    if len(selected) < 3:
        raise ValueError(
            "Choose a registered reference photo with at least two overlapping views in its block."
        )
    path: Path = model_dir / DENSE_CACHE_NAME
    if path.is_file():
        cached: dict[str, Any] = read_json(path)
        if cached.get("focus") == focus:
            return cached

    engine: Colmap = find_colmap(executable)
    engine.report = report
    log: Path = report.path.parent
    names: list[str] = [view.photo.staged_name for view in selected]
    with TemporaryDirectory(prefix="dense-", dir=log) as temporary:
        work: Path = Path(temporary)
        model_input: Path = model_dir
        omitted: list[str] = [
            view.photo.staged_name
            for view in model.views.values()
            if view.photo.staged_name not in names
        ]
        if omitted:
            remove: Path = work / "other-views.txt"
            remove.write_text("\n".join(omitted) + "\n", encoding="utf-8")
            model_input = work / "pilot-sparse"
            model_input.mkdir()
            engine.run(
                [
                    "image_deleter",
                    "--input_path",
                    str(model_dir),
                    "--output_path",
                    str(model_input),
                    "--image_names_path",
                    str(remove),
                ],
                log / "dense-subset.log",
            )
        workspace: Path = work / "stereo-workspace"
        workspace.mkdir()
        engine.run(
            [
                "image_undistorter",
                "--image_path",
                str(image_folder),
                "--input_path",
                str(model_input),
                "--output_path",
                str(workspace),
                "--output_type",
                "COLMAP",
                "--max_image_size",
                str(DENSE_MAX_EDGE),
            ],
            log / "dense-undistort.log",
        )

        # Every source also gets a depth map for the geometric consistency pass.
        config: Path = workspace / "stereo" / "patch-match.cfg"
        config.write_text(
            "".join(
                name
                + "\n"
                + ", ".join(other for other in names if other != name)
                + "\n"
                for name in names
            ),
            encoding="utf-8",
        )
        engine.run(
            [
                "patch_match_stereo",
                "--workspace_path",
                str(workspace),
                "--workspace_format",
                "COLMAP",
                "--PatchMatchStereo.geom_consistency",
                "true",
                "--PatchMatchStereo.max_image_size",
                str(DENSE_MAX_EDGE),
            ],
            log / "dense-stereo.log",
        )
        rectified: Path = work / "rectified"
        rectified.mkdir()
        engine.run(
            [
                "model_converter",
                "--input_path",
                str(workspace / "sparse"),
                "--output_path",
                str(rectified),
                "--output_type",
                "TXT",
            ],
            log / "dense-cameras.log",
        )
        with report.phase("overview_dense_surface"):
            sources: dict[int, DepthView] = rectified_views(
                rectified, workspace, selected
            )
            first: DepthView | None = sources.pop(selected[0].id, None)
            if first is None or not sources:
                raise ValueError(
                    "COLMAP did not produce usable dense depth for the reference and its neighbors."
                )
            original: Path = workspace / "images" / names[0]
            pixels: Pixels | None = cv2.imdecode(
                np.frombuffer(original.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if pixels is None:
                raise ValueError(
                    "The undistorted reference photo could not be decoded."
                )
            result: dict[str, Any] = depth_surface(
                first, list(sources.values()), pixels
            )
            if not result["faces"]:
                raise ValueError(
                    "No dense surface had enough overlapping depth support for this reference."
                )
            result.update(
                focus=focus, photos=[view.photo.path.name for view in selected]
            )
            write_overview_file(
                path, json.dumps(result, allow_nan=False, separators=(",", ":"))
            )
    return result


def mesh_hits(
    view: View, pixels: Coordinates, triangles: Coordinates
) -> tuple[Coordinates, NDArray[np.intp]]:
    """Find the nearest positive ray intersection for each stored source pixel."""
    rays: Coordinates = np.column_stack(
        (ray_coordinates(view, pixels), np.ones(len(pixels)))
    )
    directions: Coordinates = rays @ view.rotation
    origins: Coordinates = triangles[:, 0]
    first: Coordinates = triangles[:, 1] - origins
    second: Coordinates = triangles[:, 2] - origins
    displacement: Coordinates = view.center - origins
    q: Coordinates = np.cross(displacement, first)
    numerator: Coordinates = np.einsum("ij,ij->i", second, q)
    areas: Coordinates = np.linalg.norm(np.cross(first, second), axis=1)
    points: Coordinates = np.full((len(pixels), 3), np.nan)
    hit_ids: NDArray[np.intp] = np.full(len(pixels), -1, dtype=np.intp)
    for index, direction in enumerate(directions):
        p: Coordinates = np.cross(direction, second)
        determinant: Coordinates = np.einsum("ij,ij->i", first, p)
        possible: NDArray[np.bool_] = np.abs(determinant) > areas * 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            u: Coordinates = np.einsum("ij,ij->i", displacement, p) / determinant
            v: Coordinates = np.einsum("ij,j->i", q, direction) / determinant
            distance: Coordinates = numerator / determinant
        valid: NDArray[np.intp] = np.flatnonzero(
            possible
            & (u >= -1e-9)
            & (v >= -1e-9)
            & (u + v <= 1 + 1e-9)
            & (distance > 0)
        )
        if valid.size:
            nearest: int = int(valid[np.argmin(distance[valid])])
            points[index] = view.center + distance[nearest] * direction
            hit_ids[index] = nearest
    return points, hit_ids


def dense_cutline_points(
    dense: dict[str, Any],
    first: View,
    second: View,
    pixels_a: Coordinates,
    pixels_b: Coordinates,
    tolerance: float,
) -> tuple[Coordinates, NDArray[np.bool_]]:
    """Place cutline samples only when both photos see the same supported surface."""
    faces: list[list[Any]] = dense["faces"]
    shared: NDArray[np.bool_] = np.array(
        [first.id in face[3] and second.id in face[3] for face in faces], dtype=bool
    )
    if not shared.any():
        return np.empty((len(pixels_a), 3)), np.zeros(len(pixels_a), dtype=bool)
    vertices: Coordinates = np.asarray(dense["vertices"], dtype=np.float64)
    indices: NDArray[np.intp] = np.array([face[:3] for face in faces], dtype=np.intp)
    triangles: Coordinates = vertices[indices]
    points_a: Coordinates
    points_b: Coordinates
    hits_a: NDArray[np.intp]
    hits_b: NDArray[np.intp]
    points_a, hits_a = mesh_hits(first, pixels_a, triangles)
    points_b, hits_b = mesh_hits(second, pixels_b, triangles)
    seen_a: Coordinates
    seen_b: Coordinates
    z_a: Coordinates
    z_b: Coordinates
    seen_a, z_a = first.project(points_b)
    seen_b, z_b = second.project(points_a)
    # Check both directions against the original reviewed pixels and keep only
    # mutually supported first hits, preserving occlusion at cliff edges.
    valid: NDArray[np.bool_] = (
        (hits_a >= 0)
        & (hits_b >= 0)
        & shared[hits_a]
        & shared[hits_b]
        & (z_a > 0)
        & (z_b > 0)
        & (np.linalg.norm(seen_a - pixels_a, axis=1) <= tolerance)
        & (np.linalg.norm(seen_b - pixels_b, axis=1) <= tolerance)
        & (np.linalg.norm(points_a - points_b, axis=1) <= 0.035 * np.minimum(z_a, z_b))
    )
    return points_a, valid

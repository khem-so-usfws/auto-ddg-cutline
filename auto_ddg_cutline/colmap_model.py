"""Read sparse COLMAP models and project points into stored source pixels."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .geometry import Coordinates


@dataclass(frozen=True)
class Photo:
    """A source image and its lossless raw-pixel COLMAP staging copy."""

    path: Path
    width: int
    height: int
    staged_name: str
    digest: str
    staged_digest: str | None = None


@dataclass(frozen=True)
class Camera:
    """Estimated SIMPLE_RADIAL intrinsics in COLMAP pixel coordinates."""

    width: int
    height: int
    focal: float
    principal: Coordinates
    radial: float


@dataclass(frozen=True)
class View:
    """A registered image, its world-to-camera pose and observed tracks."""

    id: int
    photo: Photo
    camera: Camera
    rotation: Coordinates
    translation: Coordinates
    observations: dict[int, Coordinates]

    @property
    def center(self) -> Coordinates:
        """Return the camera center in reconstructed world coordinates."""
        return -self.rotation.T @ self.translation

    def project(self, xyz: Coordinates) -> tuple[Coordinates, Coordinates]:
        """Project world points to DDG's zero-based pixel center coordinates.

        Args:
            xyz: An array of reconstructed world coordinates.

        Returns:
            Source x, y coordinates and camera-space depths.
        """
        local: Coordinates = xyz @ self.rotation.T + self.translation
        depth: Coordinates = local[:, 2]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            normal: Coordinates = local[:, :2] / depth[:, None]
            radial: Coordinates = 1 + self.camera.radial * (normal**2).sum(axis=1)
            pixels: Coordinates = (
                self.camera.focal * normal * radial[:, None] + self.camera.principal
            )
        # COLMAP puts the first pixel center at (0.5, 0.5).
        return pixels - 0.5, depth


@dataclass(frozen=True)
class Track:
    """A reconstructed point tied to keypoint observations in registered photos."""

    xyz: Coordinates
    error: float
    views: frozenset[int]
    color: tuple[int, int, int] = (139, 167, 169)


@dataclass(frozen=True)
class Model:
    """One independent reconstructed camera block."""

    name: str
    views: dict[int, View]
    tracks: dict[int, Track]


def model_diagnostics(model: Model, photos: dict[str, Photo]) -> dict[str, Any]:
    """Summarize membership, image-space residuals, and relative camera baselines."""
    views: list[View] = list(model.views.values())
    registered: set[Path] = {view.photo.path for view in views}
    counts: Counter[int] = Counter(len(track.views) for track in model.tracks.values())
    errors: list[float] = [
        track.error
        for track in model.tracks.values()
        if np.isfinite(track.error) and track.error >= 0
    ]
    baselines: list[tuple[str, str, float]] = [
        (
            first.photo.path.name,
            second.photo.path.name,
            float(np.linalg.norm(first.center - second.center)),
        )
        for first, second in combinations(views, 2)
    ]
    baselines = [pair for pair in baselines if np.isfinite(pair[2])]
    scale: float = max((distance for _, _, distance in baselines), default=0.0)
    threshold: float = scale * 1e-5
    coincident: list[dict[str, Any]] = [
        {"images": [first, second], "distance": distance}
        for first, second, distance in baselines
        if distance <= threshold
    ]
    per_image: list[dict[str, Any]] = []
    for view in views:
        ids: list[int] = [
            track_id for track_id in view.observations if track_id in model.tracks
        ]
        residuals: Coordinates = np.empty(0, dtype=np.float64)
        if ids:
            xyz: Coordinates = np.array(
                [model.tracks[track_id].xyz for track_id in ids]
            )
            observed: Coordinates = np.array(
                [view.observations[track_id] for track_id in ids]
            )
            projected: Coordinates
            depth: Coordinates
            projected, depth = view.project(xyz)
            residuals = np.linalg.norm(projected - observed, axis=1)
            residuals = residuals[(depth > 0) & np.isfinite(residuals)]
        per_image.append(
            {
                "image": view.photo.path.name,
                "observations": len(ids),
                "reprojection_samples": len(residuals),
                "reprojection_median_px": (
                    float(np.median(residuals)) if residuals.size else None
                ),
                "reprojection_p90_px": (
                    float(np.quantile(residuals, 0.9)) if residuals.size else None
                ),
            }
        )
    return {
        "name": model.name,
        "images": [
            photo.path.name for photo in photos.values() if photo.path in registered
        ],
        "unregistered_images": [
            photo.path.name for photo in photos.values() if photo.path not in registered
        ],
        "tracks": len(model.tracks),
        "track_lengths": {
            str(length): count for length, count in sorted(counts.items())
        },
        "mean_track_reprojection_error_px": float(np.mean(errors)) if errors else None,
        "per_image": per_image,
        "max_camera_baseline_model_units": scale,
        "coincident_threshold_model_units": threshold,
        "near_coincident_cameras": coincident,
    }


def records(path: Path) -> Iterator[str]:
    """Yield noncomment records while retaining empty observation lines."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                yield line.strip()


def rotation_matrix(quaternion: Coordinates) -> Coordinates:
    """Convert a Hamilton quaternion to a world-to-camera rotation matrix."""
    norm: float = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("COLMAP model contains an invalid camera rotation.")
    w: np.float64
    x: np.float64
    y: np.float64
    z: np.float64
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_model(folder: Path, photos: dict[str, Photo]) -> Model:
    """Read a COLMAP text model for a known set of source photos.

    Args:
        folder: Directory containing cameras, images, and points3D text files.
        photos: Selected source photos keyed by their COLMAP filenames.

    Returns:
        Registered cameras, their observations, and triangulated tracks.

    Raises:
        ValueError: If a camera model, image or observation record is incompatible.
    """
    cameras: dict[int, Camera] = {}
    for record in filter(None, records(folder / "cameras.txt")):
        parts: list[str] = record.split()
        if len(parts) != 8 or parts[1] != "SIMPLE_RADIAL":
            raise ValueError(f"Expected SIMPLE_RADIAL COLMAP camera: {record}")
        focal: float
        cx: float
        cy: float
        radial: float
        focal, cx, cy, radial = map(float, parts[4:])
        if focal <= 0 or not np.isfinite([focal, cx, cy, radial]).all():
            raise ValueError(f"Invalid COLMAP camera intrinsics: {record}")
        cameras[int(parts[0])] = Camera(
            int(parts[2]), int(parts[3]), focal, np.array([cx, cy]), radial
        )

    views: dict[int, View] = {}
    image_records: Iterator[str] = records(folder / "images.txt")
    for pose in image_records:
        if not pose:
            continue
        parts = pose.split(maxsplit=9)
        row: str | None = next(image_records, None)
        if len(parts) != 10 or row is None:
            raise ValueError("COLMAP images.txt contains an incomplete image record.")
        values: list[str] = row.split()
        if len(values) % 3:
            raise ValueError("COLMAP images.txt contains an incomplete observation.")
        name: str = parts[9]
        photo: Photo | None = photos.get(name)
        camera: Camera | None = cameras.get(int(parts[8]))
        if photo is None or camera is None:
            raise ValueError(f"COLMAP registered an unexpected image or camera: {name}")
        if (photo.width, photo.height) != (camera.width, camera.height):
            raise ValueError(
                f"COLMAP dimensions differ from stored pixels: {photo.path.name}"
            )
        observations: dict[int, Coordinates] = {}
        for index in range(0, len(values), 3):
            track_id: int = int(values[index + 2])
            if track_id >= 0:
                xy: Coordinates = np.array(list(map(float, values[index : index + 2])))
                observations[track_id] = xy - 0.5
        view_id: int = int(parts[0])
        views[view_id] = View(
            view_id,
            photo,
            camera,
            rotation_matrix(np.array(list(map(float, parts[1:5])))),
            np.array(list(map(float, parts[5:8]))),
            observations,
        )

    tracks: dict[int, Track] = {}
    for record in filter(None, records(folder / "points3D.txt")):
        parts = record.split()
        if len(parts) < 8 or (len(parts) - 8) % 2:
            raise ValueError("COLMAP points3D.txt contains an incomplete track.")
        red: int
        green: int
        blue: int
        red, green, blue = map(int, parts[4:7])
        tracks[int(parts[0])] = Track(
            np.array(list(map(float, parts[1:4]))),
            float(parts[7]),
            frozenset(map(int, parts[8::2])),
            (red, green, blue),
        )
    return Model(folder.name, views, tracks)

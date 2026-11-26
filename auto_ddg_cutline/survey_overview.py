"""Build a cached, local 3D review for the checked survey photos."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from os import stat_result
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .benchmark import RunReport
from .colmap import Colmap, Flags, find_colmap, inspect_flags, map_matches
from .colmap_model import Model, Photo, Track, View, read_model
from .coverage import Face, surface_faces, validate_tracks
from .geometry import MIN_CUTLINE_MATCHES, Coordinates, Pixels
from .launcher_workflow import MANIFEST_PREFIX, read_json, read_tier2_proposal
from .match_cache import MatchAssets, copy_database, load_matches, match_assets
from .paired import PointPair, review_handoffs
from .selection import connected_sections
from .selection_report import image_data
from .survey_dense import (
    DENSE_CACHE_NAME,
    build_dense_layer,
    dense_cutline_points,
    frustum_corners,
    pilot_views,
    ray_coordinates,
)
from .survey_viewer import SURVEY_OVERVIEW_NAME, write_overview, write_overview_file

MODEL_FILES: tuple[str, ...] = ("cameras.txt", "images.txt", "points3D.txt")
DISPLAY_POINTS: int = 16000
PHOTO_HIGHLIGHT_POINTS: int = 350
CACHE_VERSION: int = 1
SURFACE_TRACK_LIMIT: int = 18000
SURFACE_FACE_LIMIT: int = 9000
SURFACE_CACHE_NAME: str = "overview-surface-v1.json"
PHOTO_PREVIEW_EDGE: int = 420


@dataclass(frozen=True)
class OverviewCut:
    """A paired cutline and its link to the original review."""

    photos: tuple[str, str]
    lines: PointPair
    link: str | None
    refined: bool


def plan_cuts(
    path: Path, plan: dict[str, Any], selected: set[str]
) -> list[OverviewCut]:
    """Collect checked handoffs in stored source pixels from a saved review.

    Args:
        path: Draft or refined selection-plan JSON.
        plan: Contents of the selection plan.
        selected: Photo filenames checked in the launcher.

    Returns:
        Handoffs for which both photos are currently checked.
    """
    source: Path = path.with_suffix(".html")
    primary: set[tuple[int, ...]] = {
        tuple(entry["photos"]) for entry in plan["overlap_pairs"] if "handoff" in entry
    }
    # Early plans did not mark exportable pairs explicitly.
    if "export_pairs" not in plan:
        plan = {**plan, "export_pairs": sorted(primary)}
    cuts: list[OverviewCut] = []
    for (a, b), handoff in review_handoffs(plan).items():
        names: tuple[str, str] = (plan["photos"][a]["name"], plan["photos"][b]["name"])
        if not selected.issuperset(names):
            continue
        prefix: str = "" if (a, b) in primary else "local-"
        link: str | None = (
            f"{source.resolve().as_uri()}#{prefix}pair-{a}-{b}"
            if source.is_file()
            else None
        )
        cuts.append(
            OverviewCut(
                names,
                (
                    np.asarray(handoff["views"][0]["line"], dtype=np.float64),
                    np.asarray(handoff["views"][1]["line"], dtype=np.float64),
                ),
                link,
                bool(plan.get("refinement")),
            )
        )
    return cuts


def model_paths(folder: Path) -> list[Path]:
    """Find complete text models written alongside a matching job."""
    return [
        path
        for path in sorted(folder.glob("model-*"))
        if all((path / name).is_file() for name in MODEL_FILES)
    ]


def source_photo(folder: Path, name: str) -> Path:
    """Resolve a saved filename inside the selected photo folder."""
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Expected a photo filename: {name!r}")
    path: Path = folder / name
    if not path.is_file():
        raise ValueError(f"Photo is missing: {name}")
    return path


def file_version(path: Path) -> tuple[str, int, int, int]:
    """Fingerprint a workspace file without reading all its pixels again."""
    info: stat_result = path.stat()
    return str(path.resolve()), info.st_size, info.st_mtime_ns, info.st_ctime_ns


def photo_preview(path: Path, cache: Path) -> str | None:
    """Reuse a small color preview in the original photo's stored orientation.

    Args:
        path: Original survey photo.
        cache: Workspace directory for the embedded previews.

    Returns:
        A JPEG data URL if the photo could be decoded.
    """
    key: str = hashlib.sha256(
        json.dumps(file_version(path)).encode("utf-8")
    ).hexdigest()
    saved: Path = cache / f"{key}.txt"
    if saved.is_file():
        return saved.read_text(encoding="ascii")
    encoded: Pixels = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if not encoded.size:
        return None
    # Decode at reduced resolution when the source codec supports it.
    original: Pixels | None = cv2.imdecode(
        encoded, cv2.IMREAD_REDUCED_COLOR_4 | cv2.IMREAD_IGNORE_ORIENTATION
    )
    if original is None:
        return None
    height: int
    width: int
    height, width = original.shape[:2]
    scale: float = min(1.0, PHOTO_PREVIEW_EDGE / max(width, height))
    if scale < 1:
        original = cv2.resize(
            original,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    preview: str = image_data(original)
    cache.mkdir(parents=True, exist_ok=True)
    write_overview_file(saved, preview)
    return preview


def photo_relationships(
    plan: dict[str, Any],
    checkpoint: dict[str, Any],
    draft: Path,
    names: set[str],
) -> list[dict[str, Any]]:
    """Describe verified neighbors, including pairs outside the counting set.

    Args:
        plan: The saved draft plan with links to any paired previews.
        checkpoint: COLMAP manifest containing verified match counts.
        draft: Draft HTML report path.
        names: Photos currently available in the overview.

    Returns:
        One relationship per pair with sufficient verified matches.
    """
    edges: dict[tuple[str, ...], dict[str, Any]] = {}
    for pair in checkpoint.get("verified_pairs", []):
        sources: list[str] = pair["images"]
        count: int = pair["inliers"]
        if (
            len(sources) != 2
            or sources[0] == sources[1]
            or not names.issuperset(sources)
            or count < MIN_CUTLINE_MATCHES
        ):
            continue
        key: tuple[str, ...] = tuple(sorted(sources))
        edges[key] = {"photos": key, "inliers": count, "link": None}

    # Older checkpoints can still supply the verified counts stored in the plan.
    for entry in plan["overlap_pairs"]:
        a: int
        b: int
        a, b = entry["photos"]
        sources = [plan["photos"][index]["name"] for index in (a, b)]
        count = entry.get("inliers", 0)
        if not names.issuperset(sources):
            continue
        key = tuple(sorted(sources))
        if key not in edges and count >= MIN_CUTLINE_MATCHES:
            edges[key] = {"photos": key, "inliers": count, "link": None}
        if (
            key in edges
            and draft.is_file()
            and any(
                field in entry
                for field in ("handoff", "handoff_error", "matched_landmarks_native")
            )
        ):
            edges[key]["link"] = f"{draft.resolve().as_uri()}#pair-{a}-{b}"
    return list(edges.values())


def overview_models(
    checkpoint: Path,
    folder: Path,
    cache: Path,
    executable: str | None,
    report: RunReport,
) -> tuple[list[Model], Path]:
    """Load an existing reconstruction or build a selection-independent cache.

    Args:
        checkpoint: Run manifest with verified matches and staged source pixels.
        folder: Original survey photo directory.
        cache: Workspace folder for cached COLMAP text models.
        executable: COLMAP launcher path, if one is configured.
        report: Progress reporter for the overview job.

    Returns:
        The sparse reconstruction blocks and their persistent cache directory.
    """
    source: dict[str, Any] = read_json(checkpoint)
    assets: MatchAssets = match_assets(source, checkpoint)
    photos: dict[str, Photo] = {
        row["staged_name"]: Photo(
            source_photo(folder, row["name"]),
            row["width"],
            row["height"],
            row["staged_name"],
            row["sha256"],
            row.get("staged_sha256"),
        )
        for row in source["images"]
    }
    identity: list[Any] = [CACHE_VERSION, file_version(assets.database)]
    for photo in photos.values():
        identity.append(
            (
                file_version(photo.path),
                file_version(source_photo(assets.image_folder, photo.staged_name)),
                photo.digest,
                photo.staged_digest,
            )
        )
    key: str = hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()
    target: Path = cache / key
    if target.is_dir():
        report.manifest["alignment"] = "cached"
        return [read_model(path, photos) for path in model_paths(target)], target

    # Models next to the original database came from mapping that matching job.
    with report.phase("overview_validate_matches"):
        assets, photos = load_matches(
            checkpoint, [photo.path for photo in photos.values()], report
        )
    existing: list[Path] = model_paths(assets.database.parent)
    if not existing and source.get("shared_patch_pilot"):
        existing = model_paths(checkpoint.parent / "global-alignment")
    cache.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="building-", dir=cache) as scratch:
        working: Path = Path(scratch) / "mapping"
        if not existing:
            with report.phase("overview_align"):
                engine: Colmap = find_colmap(executable)
                engine.report = report
                copy: Path = working / "database.db"
                copy_database(assets.database, copy)
                flags: Flags
                _, flags = inspect_flags(engine, working)
                map_matches(engine, flags, copy, assets.image_folder, photos)
                existing = model_paths(working)
            report.manifest["alignment"] = "mapped"
        else:
            report.manifest["alignment"] = "existing"

        # Publish the complete cache directory atomically. A concurrent run may
        # already have published the same model while this one was working.
        staged: Path = Path(scratch) / "models"
        staged.mkdir()
        for path in existing:
            destination: Path = staged / path.name
            destination.mkdir()
            for name in MODEL_FILES:
                shutil.copyfile(path / name, destination / name)
        try:
            staged.rename(target)
        except OSError:
            if not target.is_dir():
                raise
    return [read_model(path, photos) for path in model_paths(target)], target


def sample_cutline(lines: PointPair, scale: float) -> PointPair | None:
    """Sample paired source-pixel line segments at matching fractions."""
    a: Coordinates
    b: Coordinates
    a, b = lines
    if (
        a.shape != b.shape
        or a.ndim != 2
        or a.shape[1] != 2
        or len(a) < 2
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        return None
    distance: Coordinates = np.maximum(
        np.linalg.norm(np.diff(a, axis=0), axis=1),
        np.linalg.norm(np.diff(b, axis=0), axis=1),
    )
    along: Coordinates = np.r_[0.0, np.cumsum(distance)]
    if along[-1] <= 0:
        return None
    sample_count: int = min(
        256, max(2, math.ceil(along[-1] / (12 * max(scale, 1))) + 1)
    )
    sampling: Coordinates = np.linspace(0, along[-1], sample_count)

    # Sample corresponding segments at the same fraction. The paired seam
    # simplifier retains identical vertex indices in both source photos.
    sample_a: Coordinates = np.column_stack(
        [np.interp(sampling, along, column) for column in a.T]
    )
    sample_b: Coordinates = np.column_stack(
        [np.interp(sampling, along, column) for column in b.T]
    )
    return sample_a, sample_b


def lift_cutline(
    model: Model,
    first: View,
    second: View,
    lines: PointPair,
    *,
    dense: dict[str, Any] | None = None,
) -> tuple[list[Coordinates], int, int]:
    """Place supported stretches of a paired cutline in model coordinates.

    Args:
        model: Sparse model that contains both registered views.
        first: Camera for the first cutline.
        second: Camera for the corresponding cutline in the other photo.
        lines: Sample-aligned polylines in native source pixels.
        dense: Optional depth-supported terrain in the same model coordinates.

    Returns:
        Contiguous 3D stretches and the numbers of supported and tested intervals.
    """
    scale: float = (
        max(
            first.photo.width,
            first.photo.height,
            second.photo.width,
            second.photo.height,
        )
        / 1800
    )
    sampled: PointPair | None = sample_cutline(lines, scale)
    if sampled is None:
        return [], 0, 0
    sample_a: Coordinates
    sample_b: Coordinates
    sample_a, sample_b = sampled
    sample_count: int = len(sample_a)
    threshold: float = 4 * max(scale, 1)
    if dense is not None:
        surface_points: Coordinates
        supported: NDArray[np.bool_]
        surface_points, supported = dense_cutline_points(
            dense, first, second, sample_a, sample_b, threshold
        )
        return contiguous_sections(surface_points, supported)
    shared: list[int] = [
        track
        for track in first.observations.keys() & second.observations.keys()
        if track in model.tracks and model.tracks[track].error <= 3.0
    ]
    if len(shared) < 2:
        return [], 0, sample_count - 1
    landmarks: Coordinates = np.array([model.tracks[track].xyz for track in shared])
    marks_a: Coordinates = np.array([first.observations[track] for track in shared])
    marks_b: Coordinates = np.array([second.observations[track] for track in shared])
    projected_a: Coordinates
    projected_b: Coordinates
    depths_a: Coordinates
    depths_b: Coordinates
    projected_a, depths_a = first.project(landmarks)
    projected_b, depths_b = second.project(landmarks)
    anchor_ok: NDArray[np.bool_] = (
        (np.linalg.norm(projected_a - marks_a, axis=1) <= 4)
        & (np.linalg.norm(projected_b - marks_b, axis=1) <= 4)
        & (depths_a > 0)
        & (depths_b > 0)
    )
    homogeneous: Coordinates = cv2.triangulatePoints(
        np.column_stack((first.rotation, first.translation)),
        np.column_stack((second.rotation, second.translation)),
        ray_coordinates(first, sample_a).T,
        ray_coordinates(second, sample_b).T,
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        xyz: Coordinates = (homogeneous[:3] / homogeneous[3]).T
        x_a: Coordinates
        x_b: Coordinates
        z_a: Coordinates
        z_b: Coordinates
        x_a, z_a = first.project(xyz)
        x_b, z_b = second.project(xyz)
        ray_a: Coordinates = xyz - first.center
        ray_b: Coordinates = xyz - second.center
        cosine: Coordinates = np.sum(ray_a * ray_b, axis=1) / (
            np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1)
        )
    valid: NDArray[np.bool_] = (
        np.isfinite(xyz).all(axis=1)
        & (z_a > 0)
        & (z_b > 0)
        & (cosine < math.cos(math.radians(1)))
        & (np.linalg.norm(x_a - sample_a, axis=1) <= 4 * max(scale, 1))
        & (np.linalg.norm(x_b - sample_b, axis=1) <= 4 * max(scale, 1))
    )
    radius: float = 42 * max(scale, 1)
    for index in np.flatnonzero(valid):
        # Nearby tracks in both photos constrain depth independently of a
        # candidate's reprojection error, which alone can accept empty space.
        near: NDArray[np.bool_] = (
            anchor_ok
            & (np.linalg.norm(marks_a - sample_a[index], axis=1) <= radius)
            & (np.linalg.norm(marks_b - sample_b[index], axis=1) <= radius)
            & (np.abs(depths_a - z_a[index]) <= 0.20 * z_a[index])
            & (np.abs(depths_b - z_b[index]) <= 0.20 * z_b[index])
        )
        valid[index] = np.count_nonzero(near) >= 2

    return contiguous_sections(xyz, valid)


def contiguous_sections(
    xyz: Coordinates, valid: NDArray[np.bool_]
) -> tuple[list[Coordinates], int, int]:
    """Keep runs of adjacent supported samples without filling gaps."""
    boundaries: NDArray[np.intp] = np.flatnonzero(
        np.diff(np.r_[False, valid, False].astype(np.int8))
    )
    spans: list[Coordinates] = [
        xyz[start:stop]
        for start, stop in boundaries.reshape(-1, 2)
        if stop - start >= 2
    ]
    return spans, sum(len(span) - 1 for span in spans), len(valid) - 1


def cached_surface(model: Model, path: Path) -> dict[str, Any]:
    """Cache a bounded mesh of shared terrain patches beside its sparse model.

    Args:
        model: Registered views and sparse tracks for one block.
        path: Surface cache file within that block's alignment cache.

    Returns:
        World-space vertices and index faces with their supporting view IDs.
    """
    if path.is_file():
        try:
            saved: dict[str, Any] = read_json(path)
            if isinstance(saved.get("vertices"), list) and isinstance(
                saved.get("faces"), list
            ):
                return saved
        except (OSError, ValueError):
            pass

    # Limit the work on large surveys before triangulating reference views.
    track_ids: list[int] = sorted(model.tracks)
    if len(track_ids) > SURFACE_TRACK_LIMIT:
        track_ids = [
            track_ids[index]
            for index in np.linspace(
                0, len(track_ids) - 1, SURFACE_TRACK_LIMIT, dtype=np.intp
            )
        ]
    bounded: Model = Model(
        model.name, model.views, {index: model.tracks[index] for index in track_ids}
    )
    observations: dict[int, dict[int, Coordinates]]
    observations, _ = validate_tracks(bounded)
    supported: list[Face]
    supported, _ = surface_faces(bounded, observations)
    if len(supported) > SURFACE_FACE_LIMIT:
        supported = [
            supported[index]
            for index in np.linspace(
                0, len(supported) - 1, SURFACE_FACE_LIMIT, dtype=np.intp
            )
        ]
    used: list[int] = sorted({index for face in supported for index in face.id})
    by_track: dict[int, int] = {index: offset for offset, index in enumerate(used)}
    surface: dict[str, Any] = {
        "vertices": [model.tracks[index].xyz.tolist() for index in used],
        "faces": [
            [*(by_track[index] for index in face.id), sorted(face.views)]
            for face in supported
        ],
    }
    write_overview_file(
        path, json.dumps(surface, allow_nan=False, separators=(",", ":"))
    )
    return surface


def scene_block(
    model: Model,
    surface: dict[str, Any],
    cuts: list[OverviewCut],
    dense: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare drawable terrain, camera footprints, points, and cutlines."""
    views: dict[str, View] = {
        view.photo.path.name: view for view in model.views.values()
    }
    cameras: list[View] = list(views.values())
    visible: list[tuple[int, Track]] = [
        (track_id, track)
        for track_id, track in model.tracks.items()
        if track.error <= 3 and np.isfinite(track.xyz).all()
    ]
    points: Coordinates = np.asarray(
        [track.xyz for _, track in visible], dtype=np.float64
    ).reshape(-1, 3)
    reference: Coordinates = (
        points if len(points) else np.array([camera.center for camera in cameras])
    )

    # Normalize for display only. Camera and terrain coordinates share the
    # same rotation and scale, with approximate up from the registered photos.
    origin: Coordinates = np.median(reference, axis=0)
    extent: float = float(np.quantile(np.linalg.norm(reference - origin, axis=1), 0.98))
    up: Coordinates = -np.mean([view.rotation[1] for view in views.values()], axis=0)
    up_length: float = float(np.linalg.norm(up))
    up = up / up_length if up_length > 0.1 else np.array([0.0, 0.0, 1.0])
    right: Coordinates = np.cross(np.eye(3)[int(np.argmin(np.abs(up)))], up)
    right /= np.linalg.norm(right)
    basis: Coordinates = np.column_stack((right, np.cross(up, right), up))
    extent = extent if extent > 1e-9 else 1.0

    def display(xyz: Coordinates) -> list[Any]:
        """Transform model coordinates to the common drawing frame."""
        return np.round((xyz - origin) @ basis / extent, 5).tolist()

    shown: NDArray[np.intp] = np.flatnonzero(
        np.linalg.norm(points - origin, axis=1) <= 1.4 * extent
    )
    if len(shown) > DISPLAY_POINTS:
        shown = shown[np.linspace(0, len(shown) - 1, DISPLAY_POINTS, dtype=np.intp)]
    displayed: list[int] = [visible[index][0] for index in shown]
    markers: list[dict[str, Any]] = []
    for camera in cameras:
        observed: list[int] = [
            index
            for index, track_id in enumerate(displayed)
            if track_id in camera.observations
        ]
        if len(observed) > PHOTO_HIGHLIGHT_POINTS:
            observed = [
                observed[index]
                for index in np.linspace(
                    0, len(observed) - 1, PHOTO_HIGHLIGHT_POINTS, dtype=np.intp
                )
            ]
        markers.append(
            {
                "name": camera.photo.path.name,
                "view": camera.id,
                "position": display(camera.center),
                "direction": display(
                    camera.center + 0.14 * extent * camera.rotation[2]
                ),
                "frustum": display(frustum_corners(camera, 0.14 * extent)),
                "observed": observed,
            }
        )
    lines: list[dict[str, Any]] = []
    dense_lines: list[dict[str, Any]] = []
    layers: list[tuple[dict[str, Any] | None, list[dict[str, Any]]]] = [(None, lines)]
    if dense is not None:
        layers.append((dense, dense_lines))
    for index, cut in enumerate(cuts):
        a: str
        b: str
        a, b = cut.photos
        if a in views and b in views:
            for terrain, placements in layers:
                spans: list[Coordinates]
                supported: int
                total: int
                spans, supported, total = lift_cutline(
                    model, views[a], views[b], cut.lines, dense=terrain
                )
                if spans:
                    placements.append(
                        {
                            "pair": index,
                            "spans": [display(span) for span in spans],
                            "supported": supported,
                            "total": total,
                        }
                    )
    return {
        "name": model.name,
        "points": display(points[shown]),
        "colors": [visible[index][1].color for index in shown],
        "cameras": markers,
        "lines": lines,
        "surface": {
            "vertices": display(
                np.asarray(surface["vertices"], dtype=np.float64).reshape(-1, 3)
            ),
            "faces": surface["faces"],
        },
        "dense": (
            {
                **dense,
                "vertices": display(np.asarray(dense["vertices"], dtype=np.float64)),
                "lines": dense_lines,
            }
            if dense is not None
            else None
        ),
    }


def build_survey_overview(request_path: Path) -> int:
    """Build and publish an overview for a snapshot of the checked photos.

    Args:
        request_path: Launcher-created request containing the source paths
            and exact checkbox selections for this run.

    Returns:
        The launcher worker exit status.
    """
    request: dict[str, Any] = read_json(request_path)
    folder: Path = Path(request["folder"])
    initial: Path = Path(request["plan"])
    checkpoint: Path = Path(request["checkpoint"])
    work_dir: Path = request_path.parent.parent
    job: Path = request_path.parent
    selected: list[str] = request["checked"]
    if len(set(selected)) != len(selected):
        raise ValueError("The overview selection contains duplicate photos.")
    initial_plan: dict[str, Any]
    digest: str
    initial_plan, digest = read_tier2_proposal(initial)
    if digest != request["digest"]:
        raise ValueError("The draft review changed. Reload the previous selection.")
    available: list[str] = list(
        dict.fromkeys(
            request.get("photos", [row["name"] for row in initial_plan["photos"]])
            + selected
        )
    )
    photos: list[Path] = [source_photo(folder, name) for name in available]
    manifest: dict[str, Any] = {"output": "survey-overview", "status": "running"}
    report: RunReport = RunReport(job / "run.json", manifest, False)
    print(f"{MANIFEST_PREFIX}{report.path}", flush=True)
    try:
        models: list[Model]
        alignment: Path
        models, alignment = overview_models(
            checkpoint,
            folder,
            work_dir / "survey-overview-cache",
            request.get("colmap"),
            report,
        )
        dense_focus: str | None = request.get("dense_focus")
        if dense_focus:
            candidates: list[tuple[int, Model]] = [
                (len(pilot_views(model, dense_focus)), model) for model in models
            ]
            usable: list[tuple[int, Model]] = [
                entry for entry in candidates if entry[0] >= 3
            ]
            if not usable:
                raise ValueError(
                    f"{dense_focus} needs at least two registered overlapping views in one block. "
                    "Choose a different dense terrain reference."
                )
            model: Model = max(usable, key=lambda entry: entry[0])[1]
            assets: MatchAssets = match_assets(read_json(checkpoint), checkpoint)
            dense: dict[str, Any] = build_dense_layer(
                model,
                dense_focus,
                alignment / model.name,
                assets.image_folder,
                request.get("colmap"),
                report,
            )
            manifest.update(
                dense_focus=dense_focus,
                dense_photos=dense["photos"],
                dense_faces=len(dense["faces"]),
            )
        selected_names: set[str] = set(selected)
        cuts: list[OverviewCut] = plan_cuts(initial, initial_plan, selected_names)
        refined: str | None = request.get("refined")
        refined_names: set[str] = set()
        refined_plan: dict[str, Any] | None = None
        if refined:
            refined_plan = read_json(Path(refined))
            refined_names = {row["name"] for row in refined_plan["photos"]}
            latest: list[OverviewCut] = plan_cuts(
                Path(refined), refined_plan, selected_names
            )
            # A failed refined handoff must not be shown as a successful draft.
            cuts = [
                cut for cut in cuts if not refined_names.issuperset(cut.photos)
            ] + latest
        with report.phase("overview_terrain"):
            surfaces: dict[str, dict[str, Any]] = {
                model.name: cached_surface(
                    model, alignment / model.name / SURFACE_CACHE_NAME
                )
                for model in models
                if model.views
            }
        dense_layers: dict[str, dict[str, Any]] = {}
        for model in models:
            cached: Path = alignment / model.name / DENSE_CACHE_NAME
            if cached.is_file():
                dense_layers[model.name] = read_json(cached)
        with report.phase("overview_project_cutlines"):
            blocks: list[dict[str, Any]] = [
                scene_block(
                    model, surfaces[model.name], cuts, dense_layers.get(model.name)
                )
                for model in models
                if model.views
            ]
        blocks.sort(
            key=lambda item: (
                -sum(camera["name"] in selected_names for camera in item["cameras"]),
                -len(item["cameras"]),
            )
        )
        count_by_name: dict[str, int] = {
            name: i + 1 for i, name in enumerate(available)
        }
        graph: dict[int, set[int]] = {count_by_name[name]: set() for name in selected}
        for cut in cuts:
            first: int = count_by_name[cut.photos[0]]
            second: int = count_by_name[cut.photos[1]]
            graph[first].add(second)
            graph[second].add(first)
        accepted: set[frozenset[str]] = {frozenset(cut.photos) for cut in cuts}
        missing: list[list[str]] = []
        for plan in (initial_plan, refined_plan):
            if plan is None:
                continue
            for a, b in plan.get("unresolved_handoffs", []):
                names: list[str] = [plan["photos"][index]["name"] for index in (a, b)]
                if (
                    selected_names.issuperset(names)
                    and frozenset(names) not in accepted
                    and not (plan is initial_plan and refined_names.issuperset(names))
                ):
                    missing.append(names)
        # Refresh one file beside the analysis report so its link stays stable.
        output: Path = initial.with_name(SURVEY_OVERVIEW_NAME).resolve()
        draft: Path = initial.with_suffix(".html")
        with report.phase("overview_photo_previews"):
            previews: list[dict[str, Any]] = [
                {
                    "name": path.name,
                    "link": path.resolve().as_uri(),
                    "checked": path.name in selected_names,
                    "thumbnail": photo_preview(path, alignment / "photo-previews"),
                }
                for path in photos
            ]
        write_overview(
            output,
            {
                "blocks": blocks,
                "denseFocus": dense_focus,
                "photos": previews,
                "overlaps": photo_relationships(
                    initial_plan, read_json(checkpoint), draft, set(available)
                ),
                "cuts": [
                    {"photos": cut.photos, "link": cut.link, "refined": cut.refined}
                    for cut in cuts
                ],
                "sections": connected_sections(graph),
                "missing": missing,
                "draft": draft.resolve().as_uri() if draft.is_file() else None,
            },
        )
        manifest.update(
            status="complete",
            selected_photos=len(selected),
            blocks=len(blocks),
            overview_report=str(output),
        )
        print(f"Survey overview: {output}", flush=True)
        return 0
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        report.finish()

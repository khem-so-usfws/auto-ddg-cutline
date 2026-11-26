"""Run launcher checks and save decisions from a cutline review."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from typing import Any, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .benchmark import replace_manifest
from .colmap import Colmap, Device, Flags, find_colmap, inspect_flags
from .colmap_model import Photo
from .geometry import MIN_CUTLINE_MATCHES, Coordinates
from .milestone2 import run_colmap
from .paired import CountSide, IndexPair, review_handoffs
from .polygons import simple_polygon
from .selection import RefinementRoutes, pair_index
from .sidecars import (
    PairedEdit,
    SidecarEdit,
    load_sidecar,
    plan_paired_edits,
    write_edits,
)

MANIFEST_PREFIX: str = "COLMAP run manifest: "
TIER2_SELECTION_NAME: str = "tier2-selection.json"
PROTECT_GROUP_LABEL: str = "protect_group"


@dataclass(frozen=True)
class SavedSelection:
    """A saved recommendation mapped to the currently displayed photo rows."""

    plan: Path
    digest: str
    recommended: frozenset[int]
    plan_indices: tuple[int | None, ...]
    report: Path | None
    run: Path | None
    can_export: bool

    def refinement_indices(self, checked: list[int]) -> list[int] | None:
        """Translate checked rows, or signal that a row needs a new analysis."""
        indices: list[int] = []
        for index in checked:
            saved: int | None = self.plan_indices[index]
            if saved is None:
                return None
            indices.append(saved)
        return indices


def read_json(path: Path) -> dict[str, Any]:
    """Read a launcher document, accepting UTF-8 files saved by Windows editors."""
    data: Any = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")
    return cast(dict[str, Any], data)


def read_tier2_proposal(path: Path) -> tuple[dict[str, Any], str]:
    """Load the proposal and bind the checked photos to its exact contents.

    Args:
        path: The selection plan belonging to the initial HTML review.

    Returns:
        The proposal and its SHA-256 digest.

    Raises:
        ValueError: If the proposal lacks a valid list of reviewed photos.
    """
    contents: bytes = path.read_bytes()
    plan: Any = json.loads(contents)
    if (
        not isinstance(plan, dict)
        or not isinstance(plan.get("photos"), list)
        or not isinstance(plan.get("counting_set"), list)
    ):
        raise ValueError(f"Invalid selection plan in {path.name}.")
    return plan, hashlib.sha256(contents).hexdigest()


def selection_run_is_complete(run: dict[str, Any]) -> bool:
    """Check whether a manifest describes a reviewable selection run."""
    return (
        run.get("status") in {"complete", "partial"}
        and run.get("output") == "selection-plan"
    )


def load_saved_selection(path: Path, photos: list[Path]) -> SavedSelection:
    """Load a prior recommendation and map it to the launcher rows.

    Args:
        path: The selection plan or its companion run manifest.
        photos: Source photos currently displayed in the launcher.

    Returns:
        The checked row indices, photo mapping, and available review files.

    Raises:
        OSError: If the requested plan or manifest cannot be read.
        ValueError: If the files do not describe a usable recommendation for
            the current photo folder.
    """
    path = path.resolve()
    run: Path | None = None
    manifest: dict[str, Any] = {}
    if path.name.casefold() == "run.json":
        run = path
        path = path.with_name("selection-plan.json")
    elif path.name == "selection-plan.json":
        sibling: Path = path.with_name("run.json")
        if sibling.is_file():
            run = sibling
    if run is not None:
        manifest = read_json(run)
        if not selection_run_is_complete(manifest):
            raise ValueError("Choose a completed Analyze Photos run.")

    plan: dict[str, Any]
    digest: str
    plan, digest = read_tier2_proposal(path)
    if plan.get("refinement") or plan.get("counting_set_source") == "analyst":
        raise ValueError(
            "Choose the Analyze Photos recommendation instead of a refinement or fixed selection."
        )
    saved_names: list[str] = []
    for row in plan["photos"]:
        name: Any = row.get("name") if isinstance(row, dict) else None
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
        ):
            raise ValueError("The saved selection has an invalid photo name.")
        saved_names.append(name)
    old_indices: list[int] = plan["counting_set"]
    if (
        not saved_names
        or len(set(saved_names)) != len(saved_names)
        or any(
            type(index) is not int or not 0 <= index < len(saved_names)
            for index in old_indices
        )
        or len(set(old_indices)) != len(old_indices)
    ):
        raise ValueError(
            "The saved selection has invalid or duplicate photo references."
        )
    image_records: list[dict[str, Any]] | None = manifest.get("images")
    if (
        image_records is not None
        and [row.get("name") for row in image_records] != saved_names
    ):
        raise ValueError("run.json and selection-plan.json describe different photos.")
    by_name: dict[str, int] = {name: index for index, name in enumerate(saved_names)}
    current: dict[str, int] = {photo.name: index for index, photo in enumerate(photos)}
    recommended_names: list[str] = [saved_names[index] for index in old_indices]
    missing: list[str] = [name for name in recommended_names if name not in current]
    if missing:
        examples: str = ", ".join(missing[:3])
        raise ValueError(
            f"{len(missing)} recommended photos are missing from this folder ({examples}). "
            "Choose the photo folder for the saved run."
        )
    report: Path = path.with_suffix(".html")
    return SavedSelection(
        plan=path,
        digest=digest,
        recommended=frozenset(current[name] for name in recommended_names),
        plan_indices=tuple(by_name.get(photo.name) for photo in photos),
        report=report if report.is_file() else None,
        run=run,
        can_export=run is not None and bool(plan.get("export_pairs")),
    )


def reviewed_photo(folder: Path, row: dict[str, Any]) -> Photo:
    """Verify a source photo against its entry in the initial review.

    Args:
        folder: Directory containing the original photos.
        row: Source photo metadata saved with the review.

    Returns:
        The source path and verified photo metadata.

    Raises:
        ValueError: If the photo reference is invalid or the photo changed.
    """
    name: str = row["name"]
    if (
        not name
        or "/" in name
        or "\\" in name
        or Path(name).name != name
        or name in {".", ".."}
    ):
        raise ValueError("The saved review has an invalid photo reference.")
    path: Path = folder / name
    with path.open("rb") as stream:
        digest: str = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != row["sha256"]:
        raise ValueError(
            f"{name} has changed since analysis. Analyze the photos again."
        )
    return Photo(path, row["width"], row["height"], row["id"], digest)


def refinement_route_requests(
    plan: dict[str, Any], selected: list[int]
) -> RefinementRoutes:
    """Express surviving reviewed pairs and affected joins in checked photo order."""
    checked: dict[int, int] = {index: local for local, index in enumerate(selected)}
    reviewed: set[IndexPair] = {
        pair_index(checked[a], checked[b])
        for a, b in plan["export_pairs"]
        if a in checked and b in checked
    }
    reconnect: set[IndexPair] = {
        pair_index(checked[a], checked[b])
        for a, b in plan.get("unresolved_handoffs", [])
        if a in checked and b in checked
    }
    for section in plan.get("sections", []):
        for walk in section.get("handoff_walks", []):
            survivors: list[tuple[int, int]] = [
                (position, checked[index])
                for position, index in enumerate(walk)
                if index in checked
            ]
            reconnect.update(
                pair_index(first, second)
                for (a, first), (b, second) in zip(survivors, survivors[1:])
                if b > a + 1 and first != second
            )
    proposed: set[int] = set(plan["counting_set"])
    return {
        "reviewed": sorted(reviewed),
        "reconnect": sorted(reconnect - reviewed),
        "added": [checked[index] for index in selected if index not in proposed],
    }


def read_protected_polygons(photo: Photo) -> list[list[list[float]]]:
    """Read explicitly labeled polygons from this photo's DDG sidecar."""
    polygons: list[list[list[float]]] = []
    for shape in load_sidecar(photo).shapes:
        if not isinstance(shape, dict):
            continue
        label: Any = shape.get("label")
        if (
            not isinstance(label, str)
            or label.strip().casefold() != PROTECT_GROUP_LABEL
        ):
            continue
        if shape.get("shape_type") != "polygon":
            raise ValueError(f"{photo.path.name}: protect_group must label a polygon.")
        points: Coordinates = source_points(shape.get("points"), photo, 3)
        if not simple_polygon(points):
            raise ValueError(
                f"{photo.path.name}: protect_group must be a simple polygon."
            )
        polygons.append(points.tolist())
    return polygons


def save_tier2_selection(
    folder: Path,
    plan_path: Path,
    proposal_digest: str,
    selected: list[int],
    *,
    destination: Path | None = None,
    protect_groups: bool = False,
) -> Path:
    """Save the analyst's confirmed Tier 2 inputs for one initial analysis.

    Args:
        folder: Directory containing the original photos.
        plan_path: JSON plan from the initial seam review.
        proposal_digest: Digest of the proposal shown in the launcher.
        selected: Confirmed indices into the proposal's photo list, in order.
        destination: Optional file path within a new refinement workspace.
        protect_groups: Read marked DDG polygons into the refinement snapshot.

    Returns:
        The saved Tier 2 selection path.

    Raises:
        ValueError: If the proposal or selected photos no longer match the review.
    """
    sibling: Path = plan_path.with_name("run.json")
    run_path: Path | None = (
        sibling
        if plan_path.name == "selection-plan.json" and sibling.is_file()
        else None
    )
    if run_path is not None and not selection_run_is_complete(read_json(run_path)):
        raise ValueError("Complete the initial seam analysis before confirming photos.")

    plan: dict[str, Any]
    actual_digest: str
    plan, actual_digest = read_tier2_proposal(plan_path)
    if actual_digest != proposal_digest:
        raise ValueError(
            "The selection plan changed. Reopen the review before confirming."
        )
    rows: list[dict[str, Any]] = plan["photos"]
    if not selected:
        raise ValueError("Select at least one photo for Tier 2.")
    if len(set(selected)) != len(selected) or any(
        index < 0 or index >= len(rows) for index in selected
    ):
        raise ValueError("The Tier 2 selection contains an invalid photo index.")

    photos: list[Photo] = [reviewed_photo(folder, rows[index]) for index in selected]
    selection: dict[str, Any] = {
        "schema_version": 1,
        "source_folder": str(folder.resolve()),
        "run_manifest": (
            (str(run_path.resolve()) if destination else run_path.name)
            if run_path is not None
            else None
        ),
        "selection_plan": str(plan_path.resolve()) if destination else plan_path.name,
        "selection_plan_sha256": proposal_digest,
        "photo_indices": selected,
        "photos": [
            {
                "name": photo.path.name,
                "id": photo.staged_name,
                "width": photo.width,
                "height": photo.height,
                "sha256": photo.digest,
            }
            for photo in photos
        ],
    }
    if "export_pairs" in plan:
        selection["refinement_routes"] = refinement_route_requests(plan, selected)
    if protect_groups:
        groups: dict[str, list[list[list[float]]]] = {}
        for photo in photos:
            polygons: list[list[list[float]]] = read_protected_polygons(photo)
            if polygons:
                groups[photo.path.name] = polygons
        if not groups:
            raise ValueError(
                "Label a DDG polygon protect_group on a checked photo first."
            )
        selection["protected_groups"] = groups
    # Validate every photo before replacing any earlier confirmation.
    destination = destination or plan_path.with_name(TIER2_SELECTION_NAME)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary: Path = Path(stream.name)
        try:
            json.dump(selection, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.close()
            replace_manifest(temporary, destination)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)
    return destination


def refine_review(
    folder: Path,
    plan_path: Path,
    proposal_digest: str,
    selected: list[int],
    executable: str,
    device: Device,
    work_dir: Path,
    *,
    cpu_threads: int,
    verbose: bool = False,
    profile_resources: bool = False,
    protect_groups: bool = False,
) -> int:
    """Recompute seam drafts from the checked originals in a separate workspace.

    Args:
        folder: Folder containing the original photos.
        plan_path: Initial review's saved selection plan.
        proposal_digest: Digest of the initial plan shown in the launcher.
        selected: Ordered indices of the checked photos.
        executable: External COLMAP launcher or executable path.
        device: Requested matching device.
        work_dir: Parent folder for this refinement run.
        cpu_threads: CPU feature extraction workers.
        verbose: Whether to stream detailed COLMAP progress.
        profile_resources: Whether to sample resource use.
        protect_groups: Avoid DDG polygons labeled protect_group.

    Returns:
        The refinement worker's exit status.

    Raises:
        ValueError: If fewer than two photos were checked or the inputs changed.
    """
    if len(selected) < 2:
        raise ValueError("Check at least two photos to refine cutlines.")
    work_dir.mkdir(parents=True, exist_ok=True)
    workspace: Path = Path(
        mkdtemp(prefix="cutline-refinement-", dir=work_dir)
    ).resolve()
    snapshot: Path = save_tier2_selection(
        folder,
        plan_path,
        proposal_digest,
        selected,
        destination=workspace / TIER2_SELECTION_NAME,
        protect_groups=protect_groups,
    )
    checked: list[Path] = [
        folder / row["name"] for row in read_json(snapshot)["photos"]
    ]
    print(f"Refining cutlines for {len(checked)} checked photos.", flush=True)
    return run_colmap(
        checked,
        executable,
        device,
        workspace,
        False,
        cpu_threads,
        output="selection-plan",
        verbose=verbose,
        profile_resources=profile_resources,
        local_sfm=len(checked) >= 3,
        counting_order=list(range(len(checked))),
        refinement_selection=snapshot,
    )


def sample_photos(folder: Path) -> list[Path]:
    """Create a small overlapping photo pair with repeatable feature texture."""
    folder.mkdir()
    rng: np.random.Generator = np.random.default_rng(4389)
    canvas: NDArray[np.uint8] = rng.integers(0, 256, size=(350, 570), dtype=np.uint8)
    canvas = cv2.GaussianBlur(canvas, (5, 5), 0)
    photos: list[Path] = []
    for index, offset in enumerate((0, 120), start=1):
        path: Path = folder / f"IMG_{index}.png"
        encoded_ok: bool
        encoded: NDArray[np.uint8]
        encoded_ok, encoded = cv2.imencode(".png", canvas[:, offset : offset + 450])
        if not encoded_ok:
            raise OSError(f"Could not create {path.name} for the device check.")
        path.write_bytes(encoded.tobytes())
        photos.append(path)
    return photos


def test_installation(path: str, job: Path, *, gpu: bool) -> int:
    """Check the chosen COLMAP installation in the launcher worker.

    Args:
        path: Path to the COLMAP launcher or executable.
        job: Directory for the probe and its logs.
        gpu: Whether to run a GPU match on a synthetic overlapping pair.

    Returns:
        The worker exit status.
    """
    if not gpu:
        print(f"{MANIFEST_PREFIX}{job / 'version.log'}", flush=True)
        engine: Colmap = find_colmap(path)
        version: str
        flags: Flags
        version, flags = inspect_flags(engine, job)
        print(
            f"{version}\nLocal SfM available: {'yes' if flags.mapper_image_list else 'no'}"
        )
        return 0

    photos: list[Path] = sample_photos(job / "Photos with spaces")
    # A full-colony preview checks feature extraction and verification even if
    # the fixture's cutline cannot be placed with the installed COLMAP version.
    run_colmap(photos, path, "gpu", job, False, output="selection-plan")
    manifest: dict[str, Any] = read_json(next(job.glob("*/run.json")))
    attempt: dict[str, Any] = manifest["matching_attempts"][-1]
    if (
        manifest.get("device_used") != "gpu"
        or not all(attempt["descriptors"].get(photo.name, 0) for photo in photos)
        or not any(
            pair["inliers"] >= MIN_CUTLINE_MATCHES
            for pair in manifest["verified_pairs"]
        )
    ):
        raise ValueError(
            "The GPU test produced no usable verified overlap. See the run logs."
        )
    print("GPU test passed. Both photos produced features and a verified overlap.")
    return 0


def source_points(raw: Any, photo: Photo, count: int) -> Coordinates:
    """Validate saved coordinates before appending a reviewed annotation."""
    points: Coordinates = np.asarray(raw, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < count
        or not np.isfinite(points).all()
        or np.any(points < 0)
        or np.any(points > [photo.width - 1, photo.height - 1])
    ):
        raise ValueError(
            f"The saved annotation has invalid coordinates for {photo.path.name}."
        )
    return points


def export_review(folder: Path, manifest_path: Path) -> int:
    """Export the saved cutlines and keep sides without rematching."""
    manifest: dict[str, Any] = read_json(manifest_path)
    plan: dict[str, Any] = read_json(manifest_path.with_name("selection-plan.json"))
    if (
        manifest.get("status") not in {"complete", "partial"}
        or manifest.get("output") != "selection-plan"
        or "export_pairs" not in plan
    ):
        raise ValueError("Analyze the photos before exporting the reviewed cutlines.")

    candidates: dict[IndexPair, dict[str, Any]] = review_handoffs(plan)
    photos: dict[int, Photo] = {}
    pairs: list[PairedEdit] = []
    counting: set[int] = set(plan["counting_set"])
    for key in plan["export_pairs"]:
        if len(key) != 2 or key[0] == key[1] or not counting.issuperset(key):
            raise ValueError("The saved cutline does not belong to the counting set.")
        views: list[Photo] = []
        lines: list[Coordinates] = []
        sides: list[CountSide] = []
        for index, view in zip(key, candidates[tuple(key)]["views"], strict=True):
            if view["photo"] != index:
                raise ValueError("The saved cutline has an invalid photo reference.")
            if index not in photos:
                photos[index] = reviewed_photo(folder, plan["photos"][index])
            photo: Photo = photos[index]
            views.append(photo)
            lines.append(source_points(view["line"], photo, 2))
            keep: Any = view["keep_side"]
            if keep not in {"left", "right"}:
                raise ValueError("The saved cutline has an invalid count side.")
            point: Coordinates = source_points([view["keep_point"]], photo, 1)[0]
            sides.append(
                CountSide(
                    point, view["keep_direction"], cast(Literal["left", "right"], keep)
                )
            )
        pairs.append(
            PairedEdit(
                (views[0], views[1]), (lines[0], lines[1]), True, (sides[0], sides[1])
            )
        )

    if "count_areas" in plan:
        print(
            "This older review contains draft polygons based on feature matches. Exporting cutlines only."
        )
    edits: list[SidecarEdit]
    messages: list[str]
    edits, messages = plan_paired_edits(pairs)
    write_edits(folder, edits)
    for message in messages:
        print(message)
    print(f"Updated {len(edits)} DDG annotation files.")
    return 0

"""Plan and append LabelMe edits while preserving existing annotations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, TypeAlias

import numpy as np

from .colmap_model import Model, Photo
from .coverage import Coverage
from .geometry import Coordinates, Frame, Seam
from .paired import CountSide, PhotoPair

JsonObject: TypeAlias = dict[str, Any]


@dataclass
class Sidecar:
    """A LabelMe document and the file snapshot read before editing."""

    path: Path
    original: bytes | None
    document: JsonObject

    @property
    def shapes(self) -> list[Any]:
        """Return the editable shape list, including unsupported shapes."""
        return self.document["shapes"]


@dataclass(frozen=True)
class SidecarEdit:
    """A single replacement ready to stage on the sidecar filesystem."""

    path: Path
    original: bytes | None
    updated: bytes


def read_snapshot(path: Path) -> bytes | None:
    """Read a file, returning None if it does not exist."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def load_sidecar(frame: Frame | Photo) -> Sidecar:
    """Read or construct a DDG-compatible document for a source image.

    Args:
        frame: Original image and source dimensions.

    Returns:
        The original bytes and a document ready for append-only edits.

    Raises:
        ValueError: If an existing sidecar cannot be safely appended to.
    """
    path: Path = frame.path.with_suffix(".json")
    original: bytes | None = read_snapshot(path)
    if original is None:
        document: JsonObject = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": frame.path.name,
            "imageData": None,
            "imageHeight": frame.height,
            "imageWidth": frame.width,
        }
        return Sidecar(path, original, document)
    try:
        existing: Any = json.loads(original)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{path.name} is not valid JSON: {error}") from error
    if not isinstance(existing, dict) or not isinstance(existing.get("shapes"), list):
        raise ValueError(f"{path.name} must contain a LabelMe shapes list.")
    for key, expected in (("imageWidth", frame.width), ("imageHeight", frame.height)):
        value: Any = existing.get(key)
        if value is not None and value != expected:
            raise ValueError(
                f"{path.name} has {key}={value}, but the chosen image orientation "
                f"gives {expected}. Check that this sidecar matches the image."
            )
    return Sidecar(path, original, existing)


def serialize_sidecar(sidecar: Sidecar) -> SidecarEdit:
    """Encode a complete document for the existing atomic write path."""
    updated: bytes = (
        json.dumps(sidecar.document, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    return SidecarEdit(sidecar.path, sidecar.original, updated)


def pair_flag(seam: Seam) -> str:
    """Build a stable key for the two views of a seam."""
    return paired_image_flag(seam.first.path, seam.second.path)


def paired_image_flag(first: Path, second: Path) -> str:
    """Build a stable pair key that also survives reordering selected images."""
    identity: bytes = json.dumps(
        sorted((first.name.casefold(), second.name.casefold()))
    ).encode("utf-8")
    return "auto_ddg_pair_" + hashlib.sha256(identity).hexdigest()[:24]


def as_shape(
    points: Coordinates, marker: str, other: Frame | Photo, draft: bool
) -> JsonObject:
    """Create an editable cutline shape using source pixel coordinates.

    Args:
        points: Coordinates in the image receiving the annotation.
        marker: Pair identifier retained when DDG edits the shape.
        other: Image on the other side of the handoff.
        draft: Whether registration or ownership needs extra review.

    Returns:
        A LabelMe-compatible shape dictionary.
    """
    return {
        "label": "cutline",
        "points": np.round(points, 4).tolist(),
        "group_id": None,
        "description": f"{'Draft cutline' if draft else 'Cutline proposal'} with {other.path.name}",
        "shape_type": "line" if len(points) == 2 else "linestrip",
        "flags": {marker: True, "auto_ddg_draft": draft},
    }


def marked_shape(sidecar: Sidecar, marker: str) -> JsonObject | None:
    """Find the previously appended shape for a pair, even if it was edited."""
    for shape in sidecar.shapes:
        if isinstance(shape, dict) and isinstance(shape.get("flags"), dict):
            if shape["flags"].get(marker) is True:
                return shape
    return None


def plan_edits(
    seams: list[Seam],
    draft_pairs: set[PhotoPair],
    count_sides: dict[PhotoPair, tuple[CountSide, CountSide]] | None = None,
) -> tuple[list[SidecarEdit], list[str]]:
    """Populate affected documents and report which proposals will be appended.

    Args:
        seams: Proposed cutlines that can be placed in two source images.
        draft_pairs: Image pairs that need extra review.
        count_sides: Optional keep-side metadata for continuous paired cutlines.

    Returns:
        Staged document changes and one terminal message per proposal.
    """
    pairs: list[PairedEdit] = [
        PairedEdit(
            (seam.first, seam.second),
            seam.points_in_source_images(),
            (seam.first.path, seam.second.path) in draft_pairs,
            (
                count_sides[(seam.first.path, seam.second.path)]
                if count_sides is not None
                else None
            ),
        )
        for seam in seams
    ]
    return plan_paired_edits(pairs)


@dataclass(frozen=True)
class PairedEdit:
    """Hold two views and their native-pixel lines for sidecar export."""

    views: tuple[Frame | Photo, Frame | Photo]
    lines: tuple[Coordinates, Coordinates]
    draft: bool
    sides: tuple[CountSide, CountSide] | None


def plan_paired_edits(pairs: list[PairedEdit]) -> tuple[list[SidecarEdit], list[str]]:
    """Append missing pairs and preserve annotations already on disk."""
    documents: dict[Path, Sidecar] = {}
    changed: set[Path] = set()
    messages: list[str] = []
    for pair in pairs:
        first_view: Frame | Photo
        second_view: Frame | Photo
        first_view, second_view = pair.views
        first_path: Path
        second_path: Path
        first_path, second_path = first_view.path, second_view.path
        for view in pair.views:
            if view.path not in documents:
                documents[view.path] = load_sidecar(view)

        first: Sidecar = documents[first_path]
        second: Sidecar = documents[second_path]
        marker: str = paired_image_flag(first_path, second_path)
        existing_first: JsonObject | None = marked_shape(first, marker)
        existing_second: JsonObject | None = marked_shape(second, marker)
        title: str = f"{first_path.name} / {second_path.name}"
        if existing_first is not None and existing_second is not None:
            messages.append(f"  KEEP {title}: pair already present")
            continue
        first_shape: JsonObject = as_shape(
            pair.lines[0], marker, second_view, pair.draft
        )
        second_shape: JsonObject = as_shape(
            pair.lines[1], marker, first_view, pair.draft
        )
        if pair.sides is not None:
            for shape, side, neighbor in zip(
                (first_shape, second_shape),
                pair.sides,
                (second_path, first_path),
                strict=True,
            ):
                shape["auto_ddg_handoff"] = {
                    "paired_image": neighbor.name,
                    "keep_point": np.round(side.point, 4).tolist(),
                    "keep_side": side.side,
                }
                shape["description"] += f". Suggested count side: {side.direction}."
        if existing_first is not None or existing_second is not None:
            existing: JsonObject = (
                existing_first if existing_first is not None else existing_second
            )
            expected: JsonObject = (
                first_shape if existing_first is not None else second_shape
            )
            if existing.get("points") != expected["points"]:
                messages.append(
                    f"  REVIEW {title}: existing half was edited or generated with "
                    "different settings. Restore or draw the other half in DDG."
                )
                continue
            messages.append(f"  REPAIR {title}: append the missing half")
        else:
            messages.append(f"  APPEND {title}: add paired cutlines")

        if existing_first is None:
            first.shapes.append(first_shape)
            changed.add(first_path)
        if existing_second is None:
            second.shapes.append(second_shape)
            changed.add(second_path)

    edits: list[SidecarEdit] = []
    for path, document in documents.items():
        if path in changed:
            edits.append(serialize_sidecar(document))
    return edits, messages


def plan_count_areas(
    model: Model, coverage: Coverage, photos: dict[str, Photo]
) -> tuple[list[SidecarEdit], list[str]]:
    """Plan one coherent append-only batch without mixing ownership proposals.

    Args:
        model: Model that supplies registered view IDs and source metadata.
        coverage: Validated draft polygons and shared cutlines.
        photos: All manually selected photos, including unregistered ones.

    Returns:
        Sidecars to write and a list of already-saved files.
    """
    identity: list[tuple[str, str]] = sorted(
        (photo.path.name.casefold(), photo.digest) for photo in photos.values()
    )
    digest: str = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
    marker: str = "auto_ddg_m2_" + digest
    documents: dict[Path, Sidecar] = {
        photo.path: load_sidecar(photo) for photo in photos.values()
    }
    saved: list[str] = [
        document.path.name
        for document in documents.values()
        if marked_shape(document, marker) is not None
    ]
    if saved:
        # A rerun can differ numerically or encounter edited or deleted shapes.
        # Keep the existing batch instead of recomputing parts of its ownership.
        return [], saved

    for document in documents.values():
        for shape in document.shapes:
            if not isinstance(shape, dict):
                continue
            flags: Any = shape.get("flags")
            generated: bool = isinstance(flags, dict) and any(
                key.startswith(("auto_ddg_pair_", "auto_ddg_m2_")) for key in flags
            )
            if generated or shape.get("label") == "count_region":
                raise ValueError(
                    f"{document.path.name} already has count regions or generated "
                    "cutlines. Use a clean photo copy for a new ownership proposal."
                )

    changed: set[Path] = set()
    for view_id, polygons in coverage.polygons.items():
        path: Path = model.views[view_id].photo.path
        for polygon in polygons:
            documents[path].shapes.append(
                {
                    "label": "count_region",
                    "points": polygon.tolist(),
                    "group_id": None,
                    "description": "Draft count-area proposal from supported terrain",
                    "shape_type": "polygon",
                    "flags": {marker: True, "auto_ddg_draft": True},
                }
            )
        changed.add(path)
    for line in coverage.cutlines:
        pair: tuple[int, int] = line.first, line.second
        line_digest: str = hashlib.sha256(
            repr((pair, line.track_ids)).encode()
        ).hexdigest()[:24]
        for index, view_id in enumerate(pair):
            peer: Photo = model.views[pair[1 - index]].photo
            path = model.views[view_id].photo.path
            shape: JsonObject = as_shape(
                line.points[index], "auto_ddg_m2_line_" + line_digest, peer, True
            )
            shape["flags"][marker] = True
            documents[path].shapes.append(shape)
    return [
        serialize_sidecar(document)
        for path, document in documents.items()
        if path in changed
    ], []


@contextmanager
def folder_lock(folder: Path) -> Iterator[None]:
    """Serialize companion writes in a folder and release on process exit."""
    lock: Path = folder / ".auto-ddg-cutline.lock"
    # The lock file stays in place so two writers always lock the same file.
    with lock.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def write_edits(folder: Path, edits: list[SidecarEdit]) -> None:
    """Stage all replacements, then atomically update each individual sidecar.

    Args:
        folder: The selected source-image directory.
        edits: Existing snapshots and their append-only replacements.

    Raises:
        OSError: If writing fails or a sidecar changed after planning.
    """
    if not edits:
        return

    with folder_lock(folder), tempfile.TemporaryDirectory(dir=folder) as temp:
        staging: Path = Path(temp)
        for index, edit in enumerate(edits):
            if read_snapshot(edit.path) != edit.original:
                raise OSError(
                    f"{edit.path.name} changed during preview. Rerun the tool."
                )
            (staging / f"{index}.new").write_bytes(edit.updated)
            if edit.original is not None:
                (staging / f"{index}.old").write_bytes(edit.original)

        applied: list[tuple[int, SidecarEdit]] = []
        try:
            for index, edit in enumerate(edits):
                if read_snapshot(edit.path) != edit.original:
                    raise OSError(f"{edit.path.name} changed during export.")
                (staging / f"{index}.new").replace(edit.path)
                applied.append((index, edit))
        except (OSError, KeyboardInterrupt) as error:
            rollback_errors: list[str] = []
            for index, edit in reversed(applied):
                try:
                    if read_snapshot(edit.path) != edit.updated:
                        rollback_errors.append(edit.path.name)
                    elif edit.original is None:
                        edit.path.unlink()
                    else:
                        (staging / f"{index}.old").replace(edit.path)
                except OSError:
                    rollback_errors.append(edit.path.name)
            if rollback_errors:
                remaining: str = ", ".join(rollback_errors)
                raise OSError(
                    f"{error}. Check these sidecars before retrying: {remaining}"
                ) from error
            raise

"""Draw proposed photo ownership and matching boundaries over the source photos."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from .geometry import (
    Coordinates,
    Frame,
    Orientation,
    Pixels,
    Seam,
    intersect,
    projected_footprint,
    read_color,
)

Color: TypeAlias = tuple[int, int, int]
Pair: TypeAlias = tuple[Path, Path]
Labels: TypeAlias = list[tuple[Coordinates, str, bool]]
GREEN: Color = (135, 205, 96)
AMBER: Color = (32, 168, 242)
INK: Color = (40, 44, 46)
WHITE: Color = (255, 255, 255)


def draw_text(
    image: Pixels, text: str, origin: tuple[int, int], size: float, color: Color
) -> None:
    """Draw a legible single line of text on the review sheet."""
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA
    )


def pair_view(
    frame: Frame, seam: Seam, size: tuple[int, int]
) -> tuple[Coordinates, NDArray[np.bool_], Coordinates | None]:
    """Get a pair's line, overlap, and signed count side on one thumbnail.

    Args:
        frame: The photo to draw.
        seam: A handoff involving this photo.
        size: Width and height of the thumbnail.

    Returns:
        Thumbnail line, shared footprint mask, and signed offsets if sides are known.
    """
    width: int
    height: int
    width, height = size
    original: bool = frame.path == seam.first.path
    points: Coordinates = seam.points if original else seam.other_points()
    scale: Coordinates = np.array(
        [width / frame.gray.shape[1], height / frame.gray.shape[0]]
    )
    line: Coordinates = np.clip(
        (points + 0.5) * scale - 0.5, [0, 0], [width - 1, height - 1]
    )
    if original:
        footprint: Coordinates | None = seam.overlap
    else:
        visible: Coordinates | None = projected_footprint(seam.first, seam.to_second)
        footprint = intersect(frame.footprint, visible) if visible is not None else None

    mask: Pixels = np.zeros((height, width), dtype=np.uint8)
    if footprint is not None:
        contour: NDArray[np.int32] = np.rint((footprint + 0.5) * scale - 0.5).astype(
            np.int32
        )
        cv2.fillConvexPoly(mask, contour, 1)
    overlap: NDArray[np.bool_] = mask.astype(bool)
    if seam.sides() is None:
        return line, overlap, None

    # Orient each line by a reference point on the side assigned to this photo.
    tangent: Coordinates = points[-1] - points[0]
    length: float = float(np.linalg.norm(tangent))
    if length < 1:
        return line, overlap, None
    tangent /= length
    normal: Coordinates = np.array([-tangent[1], tangent[0]])
    reference_middle: Coordinates = (seam.points[0] + seam.points[-1]) / 2
    keep_delta: Coordinates = -seam.normal if original else seam.normal
    position: Coordinates = reference_middle + keep_delta * 35
    if not original:
        position = seam.map_to_second(position[None, :])[0]
    along: Coordinates = points @ tangent
    across: Coordinates = points @ normal
    order: NDArray[np.intp] = np.argsort(along)
    boundary: float = float(np.interp(position @ tangent, along[order], across[order]))
    direction: int = 1 if position @ normal > boundary else -1

    rows: Coordinates
    columns: Coordinates
    rows, columns = np.indices((height, width), dtype=np.float64)
    x: Coordinates = (columns + 0.5) / scale[0] - 0.5
    y: Coordinates = (rows + 0.5) / scale[1] - 0.5
    projection: Coordinates = x * tangent[0] + y * tangent[1]
    boundary_grid: Coordinates = np.interp(projection, along[order], across[order])
    keep: Coordinates = (x * normal[0] + y * normal[1] - boundary_grid) * direction
    return line, overlap, keep


def render_photo(
    frame: Frame,
    seams: list[Seam],
    drafts: set[Pair],
    frame_numbers: dict[Path, int],
    orientation: Orientation,
    limit: tuple[int, int],
) -> tuple[Pixels, str]:
    """Shade the sides assigned to this photo and annotate paired cutlines.

    Args:
        frame: Source photo to display.
        seams: Sequence cutlines.
        drafts: Pairs that need review.
        frame_numbers: Visible sequence number for each source photo.
        orientation: Source pixel orientation selected at registration.
        limit: Maximum display width and height.

    Returns:
        An annotated thumbnail and a short review note.
    """
    color: Pixels = read_color(frame.path, orientation)
    factor: float = min(limit[0] / frame.width, limit[1] / frame.height)
    size: tuple[int, int] = (
        max(1, round(frame.width * factor)),
        max(1, round(frame.height * factor)),
    )
    photo: Pixels = cv2.resize(color, size, interpolation=cv2.INTER_AREA)
    assigned: NDArray[np.bool_] = np.ones(photo.shape[:2], dtype=bool)
    uncertain: NDArray[np.bool_] = np.zeros_like(assigned)
    labels: Labels = []
    for seam in seams:
        pair: Pair = seam.first.path, seam.second.path
        if frame.path not in pair:
            continue
        other: Path = pair[1] if frame.path == pair[0] else pair[0]
        line: Coordinates
        overlap: NDArray[np.bool_]
        keep: Coordinates | None
        line, overlap, keep = pair_view(frame, seam, size)
        draft: bool = pair in drafts
        mark: str = f"To photo {frame_numbers[other]}"
        if draft:
            mark += " (draft)"
        labels.append((line, mark, draft))
        if keep is None:
            uncertain |= overlap
        else:
            assigned &= ~(overlap & (keep < 0))
            if draft:
                uncertain |= overlap & (abs(keep) < 30)

    number: int = frame_numbers[frame.path]
    assigned_peers: set[int] = {
        frame_numbers[
            pair.second.path if pair.first.path == frame.path else pair.first.path
        ]
        for pair in seams
        if frame.path in (pair.first.path, pair.second.path)
    }
    expected: set[int] = {
        n for n in (number - 1, number + 1) if 1 <= n <= len(frame_numbers)
    }
    note: str = "Review the handoff with the labeled photo(s)."
    if expected - assigned_peers:
        missing: str = ", ".join(str(n) for n in sorted(expected - assigned_peers))
        note = f"Unresolved adjacent photo(s): {missing}. Review coverage here."
    if not labels:
        uncertain[:] = True

    photo[~assigned] = np.rint(photo[~assigned] * 0.64).astype(np.uint8)
    tint: NDArray[np.float64] = np.asarray(GREEN, dtype=np.float64)
    photo[assigned] = np.rint(0.75 * photo[assigned] + 0.25 * tint).astype(np.uint8)
    photo[uncertain] = np.rint(
        0.55 * photo[uncertain] + 0.45 * np.asarray(AMBER)
    ).astype(np.uint8)
    for line, label, draft in labels:
        drawn: NDArray[np.int32] = np.rint(line).astype(np.int32)
        cv2.polylines(photo, [drawn], False, INK, 4, cv2.LINE_AA)
        cv2.polylines(photo, [drawn], False, AMBER if draft else WHITE, 2, cv2.LINE_AA)
        middle: Coordinates = (line[0] + line[-1]) / 2
        x: int = int(np.clip(middle[0] + 10, 4, max(4, size[0] - 135)))
        y: int = int(np.clip(middle[1] - 12, 19, max(19, size[1] - 4)))
        cv2.rectangle(photo, (x - 3, y - 15), (x + len(label) * 7, y + 5), INK, -1)
        draw_text(photo, label, (x, y), 0.43, WHITE)
    return photo, note


def write_review(
    folder: Path,
    frames: list[Frame],
    seams: list[Seam],
    drafts: set[Pair],
    orientation: Orientation,
) -> Path:
    """Save one PNG contact sheet for the proposed handoffs.

    Args:
        folder: Output folder selected by the user.
        frames: Source photos in survey order.
        seams: Proposed cutlines.
        drafts: Pairs marked for extra review.
        orientation: Source orientation used to generate the geometry.

    Returns:
        Path to the generated preview.
    """
    card_width: int = 690
    card_height: int = 505
    columns: int = min(3, len(frames))
    rows: int = (len(frames) + columns - 1) // columns
    sheet: Pixels = np.full(
        (120 + rows * card_height, columns * card_width, 3), 244, dtype=np.uint8
    )
    draw_text(sheet, "Proposed count areas", (18, 36), 1.0, INK)
    draw_text(
        sheet,
        "Green tint = assigned to this photo    Darker = assigned elsewhere    Amber = uncertain",
        (18, 65),
        0.65,
        INK,
    )
    draw_text(
        sheet,
        "The tint shows photo ownership. Apply your survey boundary and check the paired lines before counting.",
        (18, 93),
        0.57,
        INK,
    )
    frame_numbers: dict[Path, int] = {
        frame.path: index for index, frame in enumerate(frames, start=1)
    }
    for index, frame in enumerate(frames):
        image: Pixels
        note: str
        image, note = render_photo(
            frame, seams, drafts, frame_numbers, orientation, (660, 430)
        )
        left: int = index % columns * card_width + 15
        top: int = index // columns * card_height + 114
        title: str = f"{index + 1}. {frame.path.name}"
        draw_text(sheet, title[:96], (left, top + 22), 0.61, INK)
        height: int
        width: int
        height, width = image.shape[:2]
        sheet[top + 35 : top + 35 + height, left : left + width] = image
        draw_text(sheet, note, (left, top + 35 + height + 25), 0.5, INK)

    folder.mkdir(parents=True, exist_ok=True)
    output: Path = folder / "cutline-review.png"
    success: bool
    encoded: Pixels
    success, encoded = cv2.imencode(".png", sheet)
    if not success:
        raise OSError("Could not encode the count-area review image.")
    output.write_bytes(encoded.tobytes())
    return output

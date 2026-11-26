"""Assess directly supported overlap and local image detail for photo selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from .geometry import Coordinates, Frame, Pixels
from .paired import PointPair, working_points
from .registration import project

GRID_COLS: int = 6
GRID_ROWS: int = 4
CELL_COUNT: int = GRID_COLS * GRID_ROWS
MIN_TILE_POINTS: int = 5
MAX_GEOMETRY_POINTS: int = 1200
WORKING_TOLERANCE: float = 4.0
MIN_REPLACEMENT_SHARPNESS: float = 0.70
SHARPER_ALTERNATE_RATIO: float = 1.35
Tile: TypeAlias = tuple[int, int]
FocusMap: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True)
class Detail:
    """Texture and clipped pixels measured on the same working raster scale."""

    sharpness: float
    clipped: float


@dataclass(frozen=True)
class Replacement:
    """Direct evidence that a second view can represent samples in a source tile."""

    cell: int
    support_points: int
    replacement_cells: tuple[int, ...]
    relative_detail: float
    source: Detail
    alternative: Detail
    eligible: bool
    relative_sharpness: float | None = None


@dataclass(frozen=True)
class Overlap:
    """Matched geometry and directional replacement evidence for one photo pair."""

    first: tuple[Replacement, ...]
    second: tuple[Replacement, ...]
    held_out_count: int
    held_out_supported: int
    median_error: float
    preview_points: PointPair


def cells_at(frame: Frame, points: Coordinates) -> NDArray[np.intp]:
    """Identify the coarse source-local tile for each working coordinate."""
    grid: Coordinates = np.array([GRID_COLS, GRID_ROWS], dtype=np.float64)
    dimensions: Coordinates = np.array(frame.gray.shape[::-1], dtype=np.float64)
    xy: NDArray[np.intp] = np.floor((points + 0.5) * grid / dimensions).astype(np.intp)
    xy = np.clip(xy, 0, grid.astype(np.intp) - 1)
    return xy[:, 1] * GRID_COLS + xy[:, 0]


def tile_detail(frame: Frame) -> list[Detail]:
    """Measure relative texture sharpness and pixel clipping for each tile."""
    laplacian: NDArray[np.float32] = cv2.Laplacian(frame.gray, cv2.CV_32F)
    height: int
    width: int
    height, width = frame.gray.shape
    result: list[Detail] = []
    for row in range(GRID_ROWS):
        for column in range(GRID_COLS):
            tile: tuple[slice, slice] = (
                slice(row * height // GRID_ROWS, (row + 1) * height // GRID_ROWS),
                slice(column * width // GRID_COLS, (column + 1) * width // GRID_COLS),
            )
            levels: Pixels = frame.gray[tile]
            if levels.size:
                result.append(
                    Detail(
                        float(laplacian[tile].var()),
                        float(np.mean((levels <= 12) | (levels >= 243))),
                    )
                )
            else:
                result.append(Detail(0.0, 1.0))
    return result


def patch_focus_map(frame: Frame) -> FocusMap:
    """Summarize high-frequency focus and texture support in local windows."""
    gray: FocusMap = cv2.GaussianBlur(frame.gray.astype(np.float32), (0, 0), 0.7)
    lap: FocusMap = cv2.Laplacian(gray, cv2.CV_32F)
    gx: FocusMap = cv2.Sobel(gray, cv2.CV_32F, 1, 0, scale=0.125)
    gy: FocusMap = cv2.Sobel(gray, cv2.CV_32F, 0, 1, scale=0.125)
    high: FocusMap = cv2.boxFilter(lap * lap, -1, (15, 15))
    texture: FocusMap = cv2.boxFilter(gx * gx + gy * gy, -1, (15, 15))
    # Divide by local gradient energy to reduce sensitivity to contrast and shade.
    response: FocusMap = np.dstack((high / np.maximum(texture, 4.0), texture))
    height: int
    width: int
    height, width = frame.gray.shape
    size: tuple[int, int] = (max(1, width // 4), max(1, height // 4))
    return cv2.resize(response, size, interpolation=cv2.INTER_AREA)


def relative_patch_sharpness(
    frames: tuple[Frame, Frame],
    points: PointPair,
    focus: tuple[FocusMap, FocusMap],
) -> float | None:
    """Compare focus at aligned samples with similar scale and usable texture.

    Args:
        frames: Photos in source and comparison order.
        points: Corresponding samples in each frame's working pixel grid.
        focus: Per-frame focus and texture responses in the same photo order.

    Returns:
        The comparison-to-source sharpness ratio, or None if the samples
        lack sufficient texture or differ too much in scale.
    """
    first: Coordinates
    second: Coordinates
    first, second = points
    if len(first) < 5:
        return None
    source_span: float = float(np.linalg.norm(np.ptp(first, axis=0)))
    if source_span <= 0:
        return None
    apparent_scale: float = float(np.linalg.norm(np.ptp(second, axis=0)) / source_span)
    if not 0.80 <= apparent_scale <= 1.25:
        return None
    values: list[FocusMap] = []
    for frame, xy, response in zip(frames, points, focus, strict=True):
        rows: int
        columns: int
        rows, columns = response.shape[:2]
        size: Coordinates = np.array([columns, rows])
        scale: Coordinates = size / frame.gray.shape[::-1]
        indices: NDArray[np.intp] = np.clip(
            np.floor((xy + 0.5) * scale), 0, size - 1
        ).astype(np.intp)
        values.append(response[indices[:, 1], indices[:, 0]])
    a: FocusMap
    b: FocusMap
    a, b = values
    usable: NDArray[np.bool_] = (a[:, 1] >= 4) & (b[:, 1] >= 4)
    if int(usable.sum()) < 5:
        return None
    # A median keeps one unmatched highlight or hard rock edge from driving a tile.
    ratios: FocusMap = (b[usable, 0] + 0.01) / (a[usable, 0] + 0.01)
    return float(np.median(ratios))


def matches_in_both_frames(
    frames: tuple[Frame, Frame], matches: PointPair
) -> PointPair:
    """Convert native matches to working pixels inside both images.

    Args:
        frames: The two photos in match order.
        matches: Verified points in each photo's native pixel grid.

    Returns:
        Matched points that fall within both working pixel grids.
    """
    first: Coordinates
    second: Coordinates
    first, second = (
        working_points(frame, points)
        for frame, points in zip(frames, matches, strict=True)
    )
    valid: NDArray[np.bool_] = np.ones(len(first), dtype=bool)
    for frame, points in zip(frames, (first, second), strict=True):
        valid &= (
            np.isfinite(points).all(axis=1)
            & (points >= -0.5).all(axis=1)
            & (points < np.array(frame.gray.shape[::-1]) - 0.5).all(axis=1)
        )
    return first[valid], second[valid]


def analyze_overlap(
    frames: tuple[Frame, Frame],
    native: PointPair,
    details: tuple[list[Detail], list[Detail]],
    focus: tuple[FocusMap, FocusMap] | None = None,
) -> Overlap | None:
    """Screen a verified pair with a separate set of bidirectional checkpoints."""
    first: Coordinates
    second: Coordinates
    first, second = matches_in_both_frames(frames, native)
    if len(first) < 24:
        return None
    sample: NDArray[np.intp] = np.linspace(
        0, len(first) - 1, min(MAX_GEOMETRY_POINTS, len(first)), dtype=np.intp
    )
    hold: NDArray[np.intp] = sample[::4]
    train: NDArray[np.intp] = sample[np.arange(len(sample)) % 4 != 0]
    matrix: Coordinates | None
    matrix, _ = cv2.findHomography(first[train], second[train], cv2.RANSAC, 3.0)
    if matrix is None or not np.isfinite(matrix).all():
        return None
    try:
        inverse: Coordinates = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    forward: Coordinates = np.linalg.norm(project(first, matrix) - second, axis=1)
    reverse: Coordinates = np.linalg.norm(project(second, inverse) - first, axis=1)
    error: Coordinates = np.maximum(forward, reverse)
    supported: NDArray[np.bool_] = np.isfinite(error) & (error <= WORKING_TOLERANCE)
    held_out_supported: int = int(supported[hold].sum())
    if held_out_supported < 6 or held_out_supported / len(hold) < 0.35:
        return None
    a: Coordinates = first[supported]
    b: Coordinates = second[supported]
    points: NDArray[np.intp] = np.linspace(
        0, len(a) - 1, min(24, len(a)), dtype=np.intp
    )
    return Overlap(
        replacements(frames, (a, b), details, focus),
        replacements(
            frames[::-1], (b, a), details[::-1], focus[::-1] if focus else None
        ),
        len(hold),
        held_out_supported,
        float(np.median(error[hold][supported[hold]])),
        (frames[0].source_points(a[points]), frames[1].source_points(b[points])),
    )


def replacements(
    frames: tuple[Frame, Frame],
    points: PointPair,
    details: tuple[list[Detail], list[Detail]],
    focus: tuple[FocusMap, FocusMap] | None = None,
) -> tuple[Replacement, ...]:
    """Use directly observed tile samples without filling a hull between matches."""
    first: Frame
    second: Frame
    first, second = frames
    a: Coordinates
    b: Coordinates
    a, b = points
    a_cells: NDArray[np.intp] = cells_at(first, a)
    b_cells: NDArray[np.intp] = cells_at(second, b)
    result: list[Replacement] = []
    min_span: Coordinates = (
        np.array([first.gray.shape[1] / GRID_COLS, first.gray.shape[0] / GRID_ROWS])
        * 0.20
    )
    for cell in range(CELL_COUNT):
        mask: NDArray[np.bool_] = a_cells == cell
        count: int = int(mask.sum())
        if count < MIN_TILE_POINTS or np.any(np.ptp(a[mask], axis=0) < min_span):
            continue
        scale_a: float = float(np.linalg.norm(np.ptp(a[mask] / first.scale, axis=0)))
        scale_b: float = float(np.linalg.norm(np.ptp(b[mask] / second.scale, axis=0)))
        relative: float = scale_b / scale_a
        original: Detail = details[0][cell]
        alternative: Detail = Detail(
            float(np.median([details[1][index].sharpness for index in b_cells[mask]])),
            float(np.median([details[1][index].clipped for index in b_cells[mask]])),
        )
        sharpness: float | None = (
            relative_patch_sharpness(frames, (a[mask], b[mask]), focus)
            if focus is not None
            else None
        )
        sharp_enough: bool = (
            sharpness >= MIN_REPLACEMENT_SHARPNESS
            if sharpness is not None
            else alternative.sharpness >= 0.65 * original.sharpness
        )
        # The proxies are deliberately permissive for small differences. A much
        # softer, more clipped, or much smaller view requires analyst review.
        eligible: bool = (
            relative >= 0.65
            and sharp_enough
            and alternative.clipped <= original.clipped + 0.12
        )
        result.append(
            Replacement(
                cell,
                count,
                tuple(int(index) for index in np.unique(b_cells[mask])),
                round(relative, 3),
                original,
                alternative,
                eligible,
                round(sharpness, 3) if sharpness is not None else None,
            )
        )
    return tuple(result)

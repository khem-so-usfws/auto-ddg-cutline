"""Register source photos and place paired cutlines in their overlap."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Literal, Sequence, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from .registration import Coordinates, LocalFit, fit_local, project

Pixels: TypeAlias = NDArray[np.uint8]
Descriptors: TypeAlias = NDArray[np.float32]
Method: TypeAlias = Literal["midpoint", "image", "terrain"]
Orientation: TypeAlias = Literal["raw", "exif"]
MIN_CUTLINE_MATCHES: int = 8
GROUP_CLEARANCE_PX: int = 12
DRAFT_COLOR_EDGE_WEIGHT: float = 0.12


@dataclass(frozen=True)
class Frame:
    """A source image with its dimensions and cached working-scale features."""

    path: Path
    width: int
    height: int
    gray: Pixels
    points: Coordinates
    descriptors: Descriptors | None
    protected_polygons: tuple[Coordinates, ...] = ()
    draft_color_source: Path | None = None

    @cached_property
    def protected_mask(self) -> Pixels | None:
        """Rasterize the marked groups and their clearance in this working grid."""
        if not self.protected_polygons:
            return None
        scale: Coordinates = self.scale
        mask: Pixels = np.zeros_like(self.gray)
        for polygon in self.protected_polygons:
            pixels: Coordinates = (polygon + 0.5) * scale - 0.5
            vertices: NDArray[np.int32] = np.rint(pixels * 16).astype(np.int32)
            cv2.fillPoly(mask, [vertices], 1, shift=4)
        radius: int = max(1, int(np.ceil(GROUP_CLEARANCE_PX * scale.max())))
        kernel: Pixels = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        return cv2.dilate(mask, kernel)

    @property
    def scale(self) -> Coordinates:
        """Return the working pixels per source pixel along each axis."""
        return np.array(
            [self.gray.shape[1] / self.width, self.gray.shape[0] / self.height]
        )

    @property
    def footprint(self) -> Coordinates:
        """Return the four corners in working pixel coordinates."""
        height: int
        width: int
        height, width = self.gray.shape
        return np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float64,
        )

    def source_points(self, points: Coordinates) -> Coordinates:
        """Map working pixel centers back to source pixel centers.

        Args:
            points: An array of working coordinates in x, y order.

        Returns:
            Coordinates clipped to the source image's pixel bounds.
        """
        return np.clip(
            (points + 0.5) / self.scale - 0.5,
            [0, 0],
            [self.width - 1, self.height - 1],
        )


@dataclass(frozen=True)
class Seam:
    """One cutline projected into two frames and its registration checks."""

    first: Frame
    second: Frame
    to_first: Coordinates
    to_second: Coordinates
    overlap: Coordinates
    normal: Coordinates
    tangent: Coordinates
    points: Coordinates
    inliers: int
    match_fraction: float
    supported_sections: int
    error: float | None
    warnings: tuple[str, ...]
    method: Method
    local: LocalFit | None = None
    second_points: Coordinates | None = None
    matched_pairs: tuple[Coordinates, Coordinates] | None = None
    midpoint_fallback: str | None = None

    def offset(self, points: Coordinates) -> Coordinates:
        """Measure signed distance along the seam normal in the first frame.

        Args:
            points: Working coordinates in the first frame.

        Returns:
            Offsets whose negative side belongs to the first frame.
        """
        boundary: Coordinates = np.interp(
            points @ self.tangent, self.points @ self.tangent, self.points @ self.normal
        )
        return points @ self.normal - boundary

    def points_in_source_images(self) -> tuple[Coordinates, Coordinates]:
        """Return the same seam in both source-image pixel grids."""
        return (
            self.first.source_points(self.points),
            self.second.source_points(self.other_points()),
        )

    def other_points(self) -> Coordinates:
        """Return the paired cutline in the second image's working grid."""
        if self.second_points is not None:
            return self.second_points
        return project(self.points, self.to_second)

    def map_to_first(self, points: Coordinates) -> Coordinates:
        """Map second-image points to the first image."""
        if self.local is not None:
            return self.local.backward.map(points)
        return project(points, self.to_first)

    def map_to_second(self, points: Coordinates) -> Coordinates:
        """Map first-image points to the second image."""
        if self.local is not None:
            return self.local.forward.map(points)
        return project(points, self.to_second)

    def sides(self) -> tuple[str, str] | None:
        """Describe intended count sides near the midpoint of the cutline."""
        if any("count side" in warning for warning in self.warnings):
            return None
        middle: Coordinates = (self.points[0] + self.points[-1]) / 2
        step: Coordinates = self.normal * 12
        first_side: Coordinates = -step
        second_side: Coordinates = np.diff(
            self.map_to_second(np.array([middle, middle + step])), axis=0
        )[0]
        return compass(first_side), compass(second_side)


def load_frame(
    path: Path, max_edge: int, detector: cv2.SIFT | None, orientation: Orientation
) -> Frame:
    """Decode a photo once and compute working-scale SIFT features.

    Args:
        path: Source photo to load.
        max_edge: Longest edge of the working image in pixels.
        detector: Shared SIFT detector, or None if using external matches.
        orientation: Whether to honor the orientation stored in EXIF.

    Returns:
        Source dimensions and working-scale data for pair matching.

    Raises:
        ValueError: If OpenCV cannot decode the source image.
    """
    color: Pixels = read_color(path, orientation)
    height: int
    width: int
    height, width = color.shape[:2]
    scale: float = min(1.0, max_edge / max(height, width))
    gray: Pixels = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    if scale < 1:
        size: tuple[int, int] = (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        )
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    if detector is None:
        return Frame(path, width, height, gray, np.empty((0, 2)), None)
    keypoints: Sequence[cv2.KeyPoint]
    descriptors: Descriptors | None
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    points: Coordinates = np.array([keypoint.pt for keypoint in keypoints]).reshape(
        -1, 2
    )
    return Frame(path, width, height, gray, points, descriptors)


def read_color(path: Path, orientation: Orientation) -> Pixels:
    """Read the photo in the same pixel orientation used for DDG export."""
    # Reading bytes supports Windows paths that contain Unicode characters.
    mode: int = cv2.IMREAD_COLOR
    if orientation == "raw":
        mode |= cv2.IMREAD_IGNORE_ORIENTATION
    encoded: Pixels = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    color: Pixels | None = cv2.imdecode(encoded, mode) if encoded.size else None
    if color is None:
        raise ValueError(f"Cannot decode {path.name}.")
    return color


@lru_cache(maxsize=8)
def draft_color(path: Path, width: int, height: int) -> Pixels:
    """Cache recent draft colors without retaining a raster for every photo."""
    color: Pixels = read_color(path, "raw")
    if color.shape[:2] == (height, width):
        return color
    return cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)


def intersect(first: Coordinates, second: Coordinates) -> Coordinates | None:
    """Intersect convex footprints and discard numerically degenerate areas.

    Args:
        first: Points around the first convex boundary.
        second: Points around the second convex boundary.

    Returns:
        Vertices of a nonempty overlap or None.
    """
    area: float
    vertices: NDArray[np.float32] | None
    area, vertices = cv2.intersectConvexConvex(
        first.astype(np.float32), second.astype(np.float32)
    )
    if vertices is None or area < 1 or len(vertices) < 3:
        return None
    return vertices.reshape(-1, 2).astype(np.float64)


def compass(vector: Coordinates) -> str:
    """Express an image direction in terms of its visible frame edges.

    Args:
        vector: Direction in image coordinates, with positive y down.

    Returns:
        A direction such as left or lower-right.
    """
    sector: int = round(np.arctan2(vector[1], vector[0]) / (np.pi / 4)) % 8
    directions: tuple[str, ...] = (
        "right",
        "lower-right",
        "bottom",
        "lower-left",
        "left",
        "upper-left",
        "top",
        "upper-right",
    )
    return directions[sector]


def match_points(first: Frame, second: Frame) -> tuple[Coordinates, Coordinates]:
    """Find distinctive SIFT matches that agree in both search directions.

    Args:
        first: First frame with SIFT descriptors.
        second: Second frame with SIFT descriptors.

    Returns:
        Matching working coordinates in frame order.
    """
    if (
        first.descriptors is None
        or second.descriptors is None
        or min(len(first.descriptors), len(second.descriptors)) < 2
    ):
        empty: Coordinates = np.empty((0, 2), dtype=np.float64)
        return empty, empty

    matcher: cv2.BFMatcher = cv2.BFMatcher(cv2.NORM_L2)
    forward: list[tuple[cv2.DMatch, ...]] = matcher.knnMatch(
        first.descriptors, second.descriptors, k=2
    )
    reverse: list[tuple[cv2.DMatch, ...]] = matcher.knnMatch(
        second.descriptors, first.descriptors, k=2
    )
    reverse_ids: dict[int, int] = {
        best.queryIdx: best.trainIdx
        for best, other in reverse
        if best.distance < 0.78 * other.distance
    }
    matches: list[cv2.DMatch] = [
        best
        for best, other in forward
        if best.distance < 0.78 * other.distance
        and reverse_ids.get(best.trainIdx) == best.queryIdx
    ]
    return (
        first.points[[match.queryIdx for match in matches]],
        second.points[[match.trainIdx for match in matches]],
    )


def projected_footprint(frame: Frame, matrix: Coordinates) -> Coordinates | None:
    """Reject projections that cross the projective horizon or collapse.

    Args:
        frame: Frame whose working footprint will be transformed.
        matrix: Homography mapping that frame into another image.

    Returns:
        The projected quadrilateral, or None for an unusable projection.
    """
    footprint: Coordinates = frame.footprint
    denominators: Coordinates = footprint @ matrix[2, :2] + matrix[2, 2]
    if np.min(denominators) * np.max(denominators) <= 0:
        return None
    quad: Coordinates = project(footprint, matrix)
    float_quad: Descriptors = quad.astype(np.float32)
    if not np.all(np.isfinite(float_quad)) or not cv2.isContourConvex(float_quad):
        return None
    if cv2.contourArea(float_quad) < 1:
        return None
    return quad


def midpoint(
    overlap: Coordinates, normal: Coordinates, tangent: Coordinates
) -> Coordinates:
    """Draw across the middle of the overlap perpendicular to the handoff.

    Args:
        overlap: Convex shared footprint in the first frame.
        normal: Unit direction from the first frame toward the second.
        tangent: Unit direction along the cutline.

    Returns:
        The two clipped endpoints ordered along the tangent.
    """
    positions: Coordinates = overlap @ normal
    middle: float = float((positions.min() + positions.max()) / 2)
    return overlap_chord(overlap, normal, tangent, middle)


def overlap_chord(
    outline: Coordinates, normal: Coordinates, tangent: Coordinates, across: float
) -> Coordinates:
    """Intersect a fixed cross-seam position with a convex outline."""
    positions: Coordinates = outline @ normal
    along: Coordinates = outline @ tangent
    values: list[float] = []
    for index, start in enumerate(positions):
        end_index: int = (index + 1) % len(positions)
        end: float = float(positions[end_index])
        if min(start, end) <= across < max(start, end):
            weight: float = float((across - start) / (end - start))
            values.append(
                float(along[index] + weight * (along[end_index] - along[index]))
            )
    return across * normal + np.array([[min(values)], [max(values)]]) * tangent


def inside_frame(
    frame: Frame, points: Coordinates, margin: float = 0.0
) -> NDArray[np.bool_]:
    """Check whether working coordinates are inside a photo's pixel grid."""
    limit: Coordinates = np.array(
        [frame.gray.shape[1] - 1, frame.gray.shape[0] - 1], dtype=np.float64
    )
    return ((points >= -margin) & (points <= limit + margin)).all(axis=-1)


def corridor_boundaries(
    first: Frame,
    second: Frame,
    overlap: Coordinates,
    normal: Coordinates,
    tangent: Coordinates,
    across: Coordinates,
    local: LocalFit | None,
) -> tuple[Coordinates, NDArray[np.bool_]]:
    """Find the shared entry and exit for each sampled cross-seam position.

    Args:
        first: Photo in which the corridor is sampled.
        second: Other photo that must contain the full route.
        overlap: Shared footprint predicted by the perspective transform.
        normal: Unit vector across the seam.
        tangent: Unit vector along the seam.
        across: Working-pixel coordinates along the normal for candidate chords.
        local: Validated local correction, if available.

    Returns:
        The two boundary points for each column and a mask of usable columns.
    """
    boundaries: Coordinates = np.array(
        [overlap_chord(overlap, normal, tangent, offset) for offset in across]
    )
    if local is None:
        return boundaries, np.ones(len(across), dtype=bool)

    seeds: Coordinates = boundaries.mean(axis=1)
    footprint: Coordinates = first.footprint
    edges: Coordinates = np.array(
        [overlap_chord(footprint, normal, tangent, offset) for offset in across]
    )
    # Starting from a shared pixel, walk toward both frame edges. If the other
    # frame ends first, locate that boundary using the same warp as the export.
    inside_seeds: NDArray[np.bool_] = inside_frame(second, local.forward.map(seeds))
    start: Coordinates = np.broadcast_to(seeds[:, None], edges.shape).copy()
    edge_inside: NDArray[np.bool_] = inside_frame(
        second, local.forward.map(edges.reshape(-1, 2))
    ).reshape(edges.shape[:2])
    end: Coordinates = edges.copy()
    active: NDArray[np.bool_] = ~edge_inside & inside_seeds[:, None]
    # Fifteen bisections locate the boundary to a fraction of a working pixel.
    for _ in range(15):
        middle: Coordinates = (start[active] + end[active]) / 2
        if not len(middle):
            break
        is_inside: NDArray[np.bool_] = inside_frame(second, local.forward.map(middle))
        updated: Coordinates = np.where(is_inside[:, None], middle, start[active])
        end[active] = np.where(is_inside[:, None], end[active], middle)
        start[active] = updated
    boundaries = np.where(edge_inside[:, :, None], edges, start)
    return boundaries, inside_seeds


def seam_cost(
    first: Descriptors, second: Descriptors, valid: NDArray[np.bool_]
) -> Descriptors:
    """Score a seam for agreement, texture, and ownership of clearer pixels.

    Args:
        first: First frame sampled across the seam corridor.
        second: Second frame sampled in the same coordinates.
        valid: Mask of samples visible within both image footprints.

    Returns:
        Per-sample cost to use as the seam position.
    """
    first_smooth: Descriptors = cv2.GaussianBlur(first, (0, 0), 2)
    second_smooth: Descriptors = cv2.GaussianBlur(second, (0, 0), 2)
    detail_first: Descriptors = cv2.GaussianBlur(
        np.abs(first - first_smooth), (0, 0), 2
    )
    detail_second: Descriptors = cv2.GaussianBlur(
        np.abs(second - second_smooth), (0, 0), 2
    )
    quality_first: Descriptors = np.clip(detail_first * 8, 0, 1) * (
        (first > 0.02) & (first < 0.98)
    )
    quality_second: Descriptors = np.clip(detail_second * 8, 0, 1) * (
        (second > 0.02) & (second < 0.98)
    )
    # As the seam moves right, the first frame takes ownership of more samples.
    quality_shift: Descriptors = (
        np.cumsum(np.where(valid, quality_second - quality_first, 0), axis=1)
        / first.shape[1]
    )
    return (
        np.abs(first_smooth - second_smooth)
        + 0.6 * (detail_first + detail_second)
        + 0.5 * quality_shift
    )


def color_transition_cost(first_color: Pixels, second_color: Pixels) -> Descriptors:
    """Score directional color changes for each of the three incoming steps.

    Args:
        first_color: BGR samples laid out in the first photo's search corridor.
        second_color: BGR samples at corresponding positions in the second photo.

    Returns:
        The edge penalty indexed by transition row, incoming offset, and column.
    """
    steps: int
    columns: int
    steps, columns, _ = first_color.shape
    changes: Descriptors = np.zeros((steps - 1, 3, columns), dtype=np.float32)
    column_ids: NDArray[np.intp] = np.arange(columns)
    for color in (first_color, second_color):
        values: Descriptors = color.astype(np.float32) / 255
        for option, offset in enumerate((-1, 0, 1)):
            previous: NDArray[np.intp] = np.clip(column_ids + offset, 0, columns - 1)
            # The finite difference is small along an edge and larger across it.
            difference: Descriptors = np.max(
                np.abs(values[1:] - values[:-1, previous]), axis=2
            )
            np.maximum(changes[:, option], difference, out=changes[:, option])
    # Ignore small changes and cap the influence of a strong edge in either view.
    return DRAFT_COLOR_EDGE_WEIGHT * np.clip((changes - 0.05) / 0.25, 0, 1)


def image_seam(
    first: Frame,
    second: Frame,
    to_second: Coordinates,
    overlap: Coordinates,
    baseline: Coordinates,
    normal: Coordinates,
    tangent: Coordinates,
    local: LocalFit | None = None,
    *,
    follow_detail: bool = False,
    move_endpoints: bool = False,
    seed: Coordinates | None = None,
    avoid_groups: bool = False,
) -> tuple[Coordinates, str | None]:
    """Find a continuous image-based path near a midpoint cutline.

    Args:
        first: Reference frame for the pair.
        second: Other frame in the pair.
        to_second: Transform from first-frame to second-frame pixels.
        overlap: Convex shared footprint.
        baseline: Straight midpoint line with fixed endpoints.
        normal: Unit direction across the baseline.
        tangent: Unit direction along the baseline.
        local: Optional local correction and cliff landmark support.
        follow_detail: Let image detail guide the route even with a local warp.
        move_endpoints: Search for endpoints along both shared boundaries.
        seed: Optional native-pixel route to polish with fixed endpoints.
        avoid_groups: Use a wider corridor with marked polygons blocked.

    Returns:
        The sampled route and a reason if it had to use the midpoint.
    """
    if seed is not None:
        baseline = seed[[0, -1]]
    delta: Coordinates = baseline[-1] - baseline[0]
    length: float = float(
        delta @ tangent if seed is not None else np.linalg.norm(delta)
    )
    width: float = float(np.ptp(overlap @ normal))
    radius: int = min(16 if seed is not None else 72, int(width * 0.18))
    if avoid_groups:
        radius = min(1024, int(width * 0.45))
    if length < 30 or radius < 3:
        return baseline, "the shared search corridor is too small"

    # Search a narrow ribbon in a local coordinate system regardless of image heading.
    stride: int = 4 if avoid_groups else (1 if seed is not None else 2)
    steps: int = max(3, int(length / stride) + int(seed is not None))
    distance: Coordinates = np.linspace(0, length, steps)
    offsets: Coordinates = np.arange(-radius, radius + 1, stride, dtype=np.float64)
    columns: int = len(offsets)
    center: int = int(np.argmin(abs(offsets)))
    offsets -= offsets[center]
    usable: NDArray[np.bool_] = np.ones(columns, dtype=bool)
    if seed is not None:
        along_seed: Coordinates = seed @ tangent
        if np.any(np.diff(along_seed) <= 0):
            return seed, "the draft route is not ordered in the native alignment"
        along_samples: Coordinates = along_seed[0] + distance
        across_seed: Coordinates = np.interp(along_samples, along_seed, seed @ normal)
        samples = (
            along_samples[:, None, None] * tangent
            + (across_seed[:, None, None] + offsets[None, :, None]) * normal
        )
    elif move_endpoints:
        across: Coordinates = float(baseline[0] @ normal) + offsets
        boundaries: Coordinates
        boundaries, usable = corridor_boundaries(
            first, second, overlap, normal, tangent, across, local
        )
        if not usable.any():
            return baseline, "local alignment leaves no shared corridor for endpoints"
        spans: Coordinates = boundaries[:, 1] - boundaries[:, 0]
        # Exclude short chords close to a corner that bypass most of the overlap.
        usable &= spans @ tangent >= max(30, 0.75 * length)
        if not usable.any():
            return baseline, "no shared corridor spans enough of the overlap"
        samples: Coordinates = boundaries[None, :, 0] + (
            distance[:, None, None] / length * spans[None, :, :]
        )
    else:
        samples = (
            baseline[0]
            + distance[:, None, None] * tangent
            + offsets[None, :, None] * normal
        )
    flat: Coordinates = samples.reshape(-1, 2)
    projected: Coordinates = project(flat, to_second).reshape(samples.shape)
    support: Descriptors | None = None
    if local is not None:
        # A coarse grid is sufficient for scoring the corridor. The exported
        # cutline is mapped again at its actual sample positions.
        rows: NDArray[np.intp] = np.linspace(
            0, steps - 1, max(2, 1 + steps // 6), dtype=int
        )
        cols: NDArray[np.intp] = np.linspace(
            0, columns - 1, max(2, 1 + columns // 6), dtype=int
        )
        grid: Coordinates = samples[np.ix_(rows, cols)]
        correction: Coordinates
        distances: Coordinates
        correction, distances = local.forward.correction(grid.reshape(-1, 2))
        projected += cv2.resize(
            correction.reshape(grid.shape),
            (columns, steps),
            interpolation=cv2.INTER_LINEAR,
        )
        support = cv2.resize(
            distances.reshape(grid.shape[:2]).astype(np.float32),
            (columns, steps),
            interpolation=cv2.INTER_LINEAR,
        )
        if move_endpoints:
            endpoints: Coordinates = samples[[0, -1]].reshape(-1, 2)
            projected[[0, -1]] = local.forward.map(endpoints).reshape(2, columns, 2)
    margin: float = 1e-3 if move_endpoints or seed is not None else 0.0
    valid: NDArray[np.bool_] = (
        inside_frame(first, samples, margin)
        & inside_frame(second, projected, margin)
        & usable[None, :]
    )
    if avoid_groups:
        for frame, locations in ((first, samples), (second, projected)):
            if frame.protected_mask is not None:
                valid &= (
                    cv2.remap(
                        frame.protected_mask,
                        locations.astype(np.float32),
                        None,
                        cv2.INTER_NEAREST,
                    )
                    == 0
                )
    samples_first: Descriptors = (
        cv2.remap(
            first.gray, samples.astype(np.float32), None, cv2.INTER_LINEAR
        ).astype(np.float32)
        / 255
    )
    samples_second: Descriptors = (
        cv2.remap(
            second.gray, projected.astype(np.float32), None, cv2.INTER_LINEAR
        ).astype(np.float32)
        / 255
    )
    costs: Descriptors = seam_cost(samples_first, samples_second, valid)
    edge_cost: Descriptors | None = None
    if first.draft_color_source is not None and second.draft_color_source is not None:
        first_color: Pixels = cv2.remap(
            draft_color(
                first.draft_color_source, first.gray.shape[1], first.gray.shape[0]
            ),
            samples.astype(np.float32),
            None,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        second_color: Pixels = cv2.remap(
            draft_color(
                second.draft_color_source, second.gray.shape[1], second.gray.shape[0]
            ),
            projected.astype(np.float32),
            None,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        edge_cost = color_transition_cost(first_color, second_color)
    center_weight: float = 0.08
    step_cost: float = 0.04
    if support is not None:
        # Avoid stretches with weak terrain support when an alternate path exists.
        costs += 0.18 * np.clip((support - 18) / 65, 0, 1)
        if not follow_detail:
            center_weight = 0.16
            step_cost = 0.12
    costs += center_weight * np.abs(offsets) / radius
    costs[~valid] = np.inf
    if not move_endpoints:
        endpoints: Descriptors = costs[[0, -1], center].copy()
        costs[0, :] = costs[-1, :] = np.inf
        costs[0, center], costs[-1, center] = (
            endpoints if seed is not None or avoid_groups else (0, 0)
        )

    along: Coordinates = samples @ tangent
    predecessors: NDArray[np.intp] = np.zeros((steps, columns), dtype=np.intp)
    column_ids: NDArray[np.intp] = np.arange(columns)
    scores: Descriptors = costs[0]
    for row in range(1, steps):
        # The small sideways step penalty limits zigzags.
        options: Descriptors = np.stack(
            (
                np.r_[np.inf, scores[:-1]] + step_cost,
                scores,
                np.r_[scores[1:], np.inf] + step_cost,
            )
        )
        if edge_cost is not None:
            options += edge_cost[row - 1]
        if move_endpoints:
            # Adjacent columns can follow sloping edges. Keep each transition
            # forward and short to preserve a continuous, ordered handoff.
            previous: Coordinates = along[row - 1]
            advance: Coordinates = along[row] - np.stack(
                (np.r_[np.inf, previous[:-1]], previous, np.r_[previous[1:], np.inf])
            )
            options[(advance <= 1e-6) | (advance > 6 * length / (steps - 1))] = np.inf
        choices: NDArray[np.intp] = options.argmin(axis=0)
        predecessors[row] = column_ids + choices - 1
        scores = costs[row] + options[choices, column_ids]
    end: int = int(scores.argmin()) if move_endpoints else center
    if not np.isfinite(scores[end]):
        return baseline, "no continuous image-based route fits both photos"

    indexes: NDArray[np.intp] = np.empty(steps, dtype=np.intp)
    indexes[-1] = end
    for row in range(steps - 1, 0, -1):
        indexes[row - 1] = predecessors[row, indexes[row]]
    path: Coordinates = samples[np.arange(steps), indexes]
    if not move_endpoints:
        path[[0, -1]] = baseline
    return path, None


def simplify_paired_route(
    first: Coordinates, second: Coordinates, tolerance: float = 2.5
) -> tuple[Coordinates, Coordinates]:
    """Simplify one shared route using the larger deviation in either image.

    Args:
        first: Sampled route in the first working pixel grid.
        second: The corresponding samples in the second working pixel grid.
        tolerance: Maximum sample deviation from a retained segment in either grid.

    Returns:
        The paired vertices, with the same sample indices retained in each image.
    """
    last: int = len(first) - 1
    if last < 2 or not (np.isfinite(first).all() and np.isfinite(second).all()):
        return first, second
    keep: list[int] = [0, last]
    intervals: list[tuple[int, int]] = [(0, last)]
    while intervals:
        start: int
        end: int
        start, end = intervals.pop()
        if end - start < 2:
            continue
        maximum: Coordinates = np.zeros(end - start - 1)
        for route in (first, second):
            segment: Coordinates = route[end] - route[start]
            deltas: Coordinates = route[start + 1 : end] - route[start]
            squared_length: float = float(segment @ segment)
            if squared_length > 0:
                fractions: Coordinates = np.clip(
                    deltas @ segment / squared_length, 0, 1
                )
                deltas -= fractions[:, None] * segment
            maximum = np.maximum(maximum, np.linalg.norm(deltas, axis=1))
        furthest: int = int(maximum.argmax())
        if maximum[furthest] > tolerance:
            bend: int = start + 1 + furthest
            keep.append(bend)
            intervals.extend(((start, bend), (bend, end)))
    indices: list[int] = sorted(keep)
    return first[indices], second[indices]


def paired_cutline(
    points: Coordinates, local: LocalFit
) -> tuple[Coordinates, Coordinates]:
    """Sample a cutline and retain vertices needed in either photo.

    Args:
        points: Ordered vertices of the first-image route.
        local: Validated local registration for the pair.

    Returns:
        Equal-length vertex arrays in the first and second working grids.
    """
    pieces: list[Coordinates] = []
    for start, end in zip(points[:-1], points[1:], strict=True):
        samples: int = max(1, int(np.ceil(np.linalg.norm(end - start) / 18)))
        pieces.append(np.linspace(start, end, samples, endpoint=False))
    sampled: Coordinates = np.concatenate([*pieces, points[-1:]])
    projected: Coordinates = local.forward.map(sampled)
    keep: set[int] = {0, len(sampled) - 1}
    for route in (sampled, projected):
        vertices: NDArray[np.float32] = cv2.approxPolyDP(
            route.astype(np.float32), 3.0, False
        ).reshape(-1, 2)
        for vertex in vertices:
            keep.add(int(np.argmin(np.linalg.norm(route - vertex, axis=1))))
    indexes: list[int] = sorted(keep)
    return sampled[indexes], projected[indexes]


def propose(
    first: Frame,
    second: Frame,
    method: Method,
    matches: tuple[Coordinates, Coordinates] | None = None,
    *,
    follow_detail: bool = False,
    move_endpoints: bool = False,
    seed: Coordinates | None = None,
    avoid_groups: bool = False,
) -> tuple[Seam | None, str]:
    """Register a pair, check local match support, and propose a cutline.

    Args:
        first: Earlier frame in the selected sequence.
        second: Later frame in the selected sequence.
        method: Straight midpoint, image-based path, or locally corrected terrain path.
        matches: Optional verified correspondences in the working pixel grids.
        follow_detail: Prioritize image detail and simplify the pair together.
        move_endpoints: Require a route with endpoints chosen on shared boundaries.
        seed: Optional native-pixel route supplying the refinement corridor.
        avoid_groups: Block marked bird groups in each view.

    Returns:
        A proposal and an empty string, or None and the reason it failed.
    """
    first_points: Coordinates
    second_points: Coordinates
    first_points, second_points = (
        match_points(first, second) if matches is None else matches
    )
    if len(first_points) < MIN_CUTLINE_MATCHES:
        return None, f"only {len(first_points)} distinctive feature matches"
    to_first: Coordinates | None
    inlier_mask: Pixels | None
    to_first, inlier_mask = cv2.findHomography(
        second_points, first_points, cv2.RANSAC, 3.0
    )
    if inlier_mask is None or to_first is None or not np.all(np.isfinite(to_first)):
        return None, "a stable perspective transform could not be fit"
    paired: NDArray[np.bool_] = inlier_mask.ravel().astype(bool)
    inliers: int = int(paired.sum())
    fraction: float = inliers / len(first_points)
    if inliers < MIN_CUTLINE_MATCHES or fraction < 0.2:
        return None, f"weak geometric agreement ({inliers} inliers, {fraction:.0%})"
    try:
        to_second: Coordinates = np.linalg.inv(to_first)
    except np.linalg.LinAlgError:
        return None, "the perspective transform is singular"
    second_quad: Coordinates | None = projected_footprint(second, to_first)
    if second_quad is None or projected_footprint(first, to_second) is None:
        return None, "the estimated image footprint folds or collapses"
    overlap: Coordinates | None = intersect(first.footprint, second_quad)
    if overlap is None or cv2.contourArea(overlap.astype(np.float32)) < 350:
        return None, "the estimated overlap is too small for a cutline"

    first_center: Coordinates = first.footprint.mean(axis=0)
    second_center: Coordinates = project(
        second.footprint.mean(axis=0)[None, :], to_first
    )[0]
    displacement: Coordinates = second_center - first_center
    separation: float = float(np.linalg.norm(displacement))
    warnings: list[str] = []
    if separation < 0.03 * min(first.gray.shape):
        # Nearly nested frames have no reliable acquisition direction.
        axes: Coordinates
        _, _, axes = np.linalg.svd(overlap - overlap.mean(axis=0))
        displacement = axes[0]
        warnings.append("count side is ambiguous because the image centers are close")
    first_area: float = cv2.contourArea(first.footprint.astype(np.float32))
    second_area: float = cv2.contourArea(second_quad.astype(np.float32))
    common_area: float = cv2.contourArea(overlap.astype(np.float32))
    if common_area > 0.97 * min(first_area, second_area):
        warnings.append(
            "count side is ambiguous because one footprint is nearly nested"
        )
    normal: Coordinates = displacement / np.linalg.norm(displacement)
    tangent: Coordinates = np.array([-normal[1], normal[0]])
    baseline: Coordinates = midpoint(overlap, normal, tangent)
    if np.linalg.norm(baseline[1] - baseline[0]) < 14:
        return None, "the midpoint cutline is too short"
    points: Coordinates = baseline
    local: LocalFit | None = None
    fallback: str | None = None
    if method == "terrain":
        near_baseline: NDArray[np.bool_] = (
            abs((first_points - baseline[0]) @ normal) < 55
        )
        radius: float = max(45, min(95, min(first.gray.shape) * 0.075))
        local = fit_local(
            first_points, second_points, to_second, to_first, near_baseline, radius
        )
    if method != "midpoint":
        points, fallback = image_seam(
            first,
            second,
            to_second,
            overlap,
            baseline,
            normal,
            tangent,
            local,
            follow_detail=follow_detail,
            move_endpoints=move_endpoints,
            seed=seed,
            avoid_groups=avoid_groups,
        )
        if (
            move_endpoints or seed is not None or avoid_groups
        ) and fallback is not None:
            return None, fallback
    projected: Coordinates | None = None
    if follow_detail and fallback is None:
        # Keep raw samples until both views can contribute to simplification.
        projected = (
            local.forward.map(points)
            if local is not None
            else project(points, to_second)
        )
        points, projected = simplify_paired_route(
            points, projected, tolerance=1.0 if seed is not None else 2.5
        )
    else:
        if method != "midpoint" and fallback is None:
            points = (
                cv2.approxPolyDP(
                    points.astype(np.float32), 5.0 if local else 2.5, False
                )
                .reshape(-1, 2)
                .astype(np.float64)
            )
            points[[0, -1]] = baseline
        if local is not None:
            points, projected = paired_cutline(points, local)
    midpoint_fallback: str | None = fallback if follow_detail else None
    if midpoint_fallback is not None:
        warnings.append(f"midpoint fallback because {midpoint_fallback}")

    # Global inlier counts can hide whole ledges that disagree near the handoff.
    along: Coordinates = first_points @ tangent
    edges: Coordinates = points @ tangent
    across: Coordinates = first_points @ normal - np.interp(
        along, edges, points @ normal
    )
    length: float = float(edges[-1] - edges[0])
    band: float = max(18.0, min(75.0, float(np.ptp(overlap @ normal)) * 0.20))
    near: NDArray[np.bool_] = (
        (abs(across) <= band) & (along >= edges[0]) & (along <= edges[-1])
    )
    supported: NDArray[np.bool_] = near & paired
    if local is not None:
        landmarks: Coordinates = local.forward.landmarks
        along = landmarks @ tangent
        across = landmarks @ normal - np.interp(along, edges, points @ normal)
        supported = (abs(across) <= band) & (along >= edges[0]) & (along <= edges[-1])
    sections: int = len(
        np.unique(
            np.minimum(3, ((along[supported] - edges[0]) / length * 4).astype(int))
        )
    )
    errors: Coordinates = np.linalg.norm(
        project(second_points, to_first) - first_points, axis=1
    )
    error: float | None = float(np.quantile(errors[near], 0.8)) if near.any() else None
    if local is not None:
        error = local.local_error
    if inliers < 18 or fraction < 0.48:
        warnings.append("few or inconsistent geometric matches")
    if sections < 3 or supported.sum() < 8:
        warnings.append("well-aligned keypoints are sparse along the cutline")
    if error is not None and error > 3.5:
        warnings.append("nearby landmark alignment still differs by several pixels")
    actual_method: Method = "midpoint"
    matched_pairs: tuple[Coordinates, Coordinates] = (
        first_points[paired],
        second_points[paired],
    )
    if local is not None:
        actual_method = "terrain"
        matched_pairs = local.forward.landmarks, local.backward.landmarks
    elif method != "midpoint" and (
        follow_detail or not np.array_equal(points, baseline)
    ):
        actual_method = "image"
    if midpoint_fallback is not None:
        actual_method = "midpoint"
    return (
        Seam(
            first,
            second,
            to_first,
            to_second,
            overlap,
            normal,
            tangent,
            points,
            inliers,
            fraction,
            sections,
            error,
            tuple(warnings),
            actual_method,
            local,
            projected,
            matched_pairs,
            midpoint_fallback,
        ),
        "",
    )

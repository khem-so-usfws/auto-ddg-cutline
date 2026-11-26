"""Convert owned-pixel masks into simple, hole-free DDG polygons."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .geometry import Coordinates, Pixels


def simple_polygon(points: Coordinates) -> bool:
    """Check finite vertices, nonzero area and absence of edge intersections."""
    if (
        len(points) < 3
        or not np.isfinite(points).all()
        or len(np.unique(points, axis=0)) != len(points)
        or abs(cv2.contourArea(points.astype(np.float32))) < 1
    ):
        return False
    return not edges_cross(points, closed=True)


def edges_cross(points: Coordinates, closed: bool) -> bool:
    """Check whether any nonadjacent path segments intersect or touch."""
    starts: Coordinates = points if closed else points[:-1]
    ends: Coordinates = np.roll(points, -1, axis=0) if closed else points[1:]
    # Test nonadjacent edges, including collinear contacts.
    for index in range(len(starts) - 2):
        stop: int = len(starts) - int(closed and index == 0)
        a: Coordinates = starts[index]
        b: Coordinates = ends[index]
        c: Coordinates = starts[index + 2 : stop]
        d: Coordinates = ends[index + 2 : stop]
        candidates: NDArray[np.bool_] = np.all(
            (np.maximum(c, d) >= np.minimum(a, b))
            & (np.minimum(c, d) <= np.maximum(a, b)),
            axis=1,
        )
        c, d = c[candidates], d[candidates]
        ab: Coordinates = b - a
        cd: Coordinates = d - c
        ac: Coordinates = a - c
        bc: Coordinates = b - c
        ca: Coordinates = c - a
        da: Coordinates = d - a
        first: Coordinates = (ab[0] * ca[:, 1] - ab[1] * ca[:, 0]) * (
            ab[0] * da[:, 1] - ab[1] * da[:, 0]
        )
        second: Coordinates = (cd[:, 0] * ac[:, 1] - cd[:, 1] * ac[:, 0]) * (
            cd[:, 0] * bc[:, 1] - cd[:, 1] * bc[:, 0]
        )
        if np.any((first <= 0) & (second <= 0)):
            return True
    return False


def fill_polygon(mask: Pixels, points: Coordinates, value: int) -> None:
    """Fill source points using subpixel fixed-point coordinates."""
    vertices: NDArray[np.int32] = np.rint(points * 16).astype(np.int32)
    cv2.fillPoly(mask, [vertices], value, shift=4)


def mask_polygons(mask: Pixels, minimum: float) -> list[Coordinates]:
    """Extract editable polygons whose filled pixels stay within an owned mask.

    Args:
        mask: Image-resolution occupancy mask with zero for unsupported pixels.
        minimum: Minimum component area in source pixels.

    Returns:
        Simple polygons. Regions with holes are partitioned into pieces.
    """
    pending: list[tuple[Pixels, int, int]] = [(mask, 0, 0)]
    output: list[Coordinates] = []
    while pending:
        tile: Pixels
        tx: int
        ty: int
        tile, tx, ty = pending.pop()
        contours: Sequence[NDArray[np.int32]]
        hierarchy: NDArray[np.int32] | None
        contours, hierarchy = cv2.findContours(
            tile, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None:
            continue
        relations: NDArray[np.int32] = hierarchy[0]
        for index, contour in enumerate(contours):
            if relations[index, 3] >= 0 or cv2.contourArea(contour) < minimum:
                continue
            x: int
            y: int
            width: int
            height: int
            x, y, width, height = cv2.boundingRect(contour)
            origin: Coordinates = np.array([x, y], dtype=np.float64)
            pixels: Pixels = tile[y : y + height, x : x + width]
            child: int = int(relations[index, 2])
            if child < 0:
                original: Coordinates = (
                    contour.reshape(-1, 2).astype(np.float64) - origin
                )
                simplified: Coordinates = (
                    cv2.approxPolyDP(contour, 1.2, True)
                    .reshape(-1, 2)
                    .astype(np.float64)
                    - origin
                )
                accepted: Coordinates | None = None
                for candidate in (simplified, original):
                    if not simple_polygon(candidate):
                        continue
                    painted: Pixels = np.zeros_like(pixels)
                    fill_polygon(painted, candidate, 1)
                    if not np.any((painted != 0) & (pixels == 0)):
                        accepted = candidate
                        break
                if accepted is not None:
                    output.append(accepted + np.array([tx + x, ty + y]))
                    continue

            # Isolate the component before splitting so nested islands are
            # emitted only once. Partition through a hole or a pinched outline.
            component: Pixels = pixels.copy()
            seed: tuple[int, int] = (
                int(contour[0, 0, 0]) - x,
                int(contour[0, 0, 1]) - y,
            )
            cv2.floodFill(component, None, seed, 2, flags=8)
            component = (component == 2).astype(np.uint8)
            vertical: bool = width > height
            if child >= 0:
                hx: int
                hy: int
                hw: int
                hh: int
                hx, hy, hw, hh = cv2.boundingRect(contours[child])
                cut: int = hx - x + hw // 2 if vertical else hy - y + hh // 2
            else:
                cut = width // 2 if vertical else height // 2
            extent: int = width if vertical else height
            if not 0 < cut < extent:
                continue
            if vertical:
                pending.append((component[:, cut:], tx + x + cut, ty + y))
                pending.append((component[:, :cut], tx + x, ty + y))
            else:
                pending.append((component[cut:, :], tx + x, ty + y + cut))
                pending.append((component[:cut, :], tx + x, ty + y))
    return output

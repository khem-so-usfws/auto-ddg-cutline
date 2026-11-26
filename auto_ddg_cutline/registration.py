"""Correct perspective registration using nearby matched cliff features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

Coordinates: TypeAlias = NDArray[np.float64]


def project(points: Coordinates, matrix: Coordinates) -> Coordinates:
    """Project x, y points through a 3-by-3 perspective transform."""
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)


@dataclass(frozen=True)
class LocalWarp:
    """A perspective transform with a bounded local correction."""

    matrix: Coordinates
    landmarks: Coordinates
    residuals: Coordinates
    radius: float

    def correction(self, points: Coordinates) -> tuple[Coordinates, Coordinates]:
        """Estimate per-point displacement and distance to supporting features.

        Args:
            points: Query pixels in the input image.

        Returns:
            Displacement vectors and nearest-match distances in working pixels.
        """
        displacement: Coordinates = np.zeros_like(points)
        distances: Coordinates = np.full(len(points), np.inf)
        count: int = min(12, len(self.landmarks))
        landmarks: NDArray[np.float32] = self.landmarks.astype(np.float32)
        # Bound memory for long routes by querying the neighbors in batches.
        for start in range(0, len(points), 512):
            stop: int = min(start + 512, len(points))
            query: NDArray[np.float32] = points[start:stop].astype(np.float32)
            distance: NDArray[np.float32]
            indexes: NDArray[np.int32]
            distance, indexes = cv2.batchDistance(
                query, landmarks, cv2.CV_32F, normType=cv2.NORM_L2, K=count
            )
            values: Coordinates = self.residuals[indexes]
            closest: NDArray[np.float32] = distance[:, 0]
            distances[start:stop] = closest
            # Discard nearby points that disagree with the prevailing local shift.
            median: Coordinates = np.median(values, axis=1)
            deviation: Coordinates = np.linalg.norm(values - median[:, None], axis=2)
            limit: Coordinates = np.maximum(2, np.median(deviation, axis=1) * 1.8)
            coherent: NDArray[np.bool_] = (deviation <= limit[:, None]) & (
                distance < self.radius
            )
            weights: Coordinates = coherent / (4 + distance.astype(np.float64)) ** 1.8
            enough: NDArray[np.bool_] = coherent.sum(axis=1) >= 3
            blended: Coordinates = np.divide(
                (weights[:, :, None] * values).sum(axis=1),
                weights.sum(axis=1)[:, None],
                out=np.zeros_like(query, dtype=np.float64),
                where=enough[:, None],
            )
            # Revert smoothly to the perspective model away from matched terrain.
            fade: Coordinates = np.clip(
                (self.radius - closest) / (self.radius / 2), 0, 1
            )
            displacement[start:stop] = blended * fade[:, None]
        return displacement, distances

    def map(self, points: Coordinates) -> Coordinates:
        """Map query points into the other image using the local correction."""
        adjustment: Coordinates
        adjustment, _ = self.correction(points)
        return project(points, self.matrix) + adjustment


@dataclass(frozen=True)
class LocalFit:
    """Two local warps and a same-sample near-seam validation comparison."""

    forward: LocalWarp
    backward: LocalWarp
    global_error: float
    local_error: float
    validation_count: int


def fit_local(
    first_points: Coordinates,
    second_points: Coordinates,
    to_second: Coordinates,
    to_first: Coordinates,
    near_seam: NDArray[np.bool_],
    radius: float,
) -> LocalFit | None:
    """Fit and validate local warps using epipolar-consistent landmarks.

    Args:
        first_points: Distinctive matched keypoints in the first image.
        second_points: Their counterparts in the second image.
        to_second: Perspective mapping from the first image to the second.
        to_first: Inverse perspective mapping.
        near_seam: Matches in the corridor around the proposed line.
        radius: Maximum distance over which local corrections are used.

    Returns:
        Refined mappings if the held-out check improves in both directions.
    """
    if len(first_points) < 30:
        return None
    mask: NDArray[np.uint8] | None
    _, mask = cv2.findFundamentalMat(
        first_points, second_points, cv2.USAC_MAGSAC, 2.0, 0.999, 5000
    )
    if mask is None:
        return None
    forward_delta: Coordinates = second_points - project(first_points, to_second)
    backward_delta: Coordinates = first_points - project(second_points, to_first)
    stable: NDArray[np.bool_] = (
        mask.ravel().astype(bool)
        & (np.linalg.norm(forward_delta, axis=1) < radius)
        & (np.linalg.norm(backward_delta, axis=1) < radius)
    )
    held_out: NDArray[np.bool_] = np.arange(len(first_points)) % 4 == 0
    train: NDArray[np.bool_] = stable & ~held_out
    check: NDArray[np.bool_] = stable & held_out & near_seam
    count: int = int(check.sum())
    if train.sum() < 24 or count < 10:
        return None

    forward: LocalWarp = LocalWarp(
        to_second, first_points[train], forward_delta[train], radius
    )
    backward: LocalWarp = LocalWarp(
        to_first, second_points[train], backward_delta[train], radius
    )
    forward_check: Coordinates = np.linalg.norm(
        forward.map(first_points[check]) - second_points[check], axis=1
    )
    backward_check: Coordinates = np.linalg.norm(
        backward.map(second_points[check]) - first_points[check], axis=1
    )
    forward_baseline: float = float(
        np.quantile(np.linalg.norm(forward_delta[check], axis=1), 0.8)
    )
    backward_baseline: float = float(
        np.quantile(np.linalg.norm(backward_delta[check], axis=1), 0.8)
    )
    forward_error: float = float(np.quantile(forward_check, 0.8))
    backward_error: float = float(np.quantile(backward_check, 0.8))
    if (
        forward_baseline - forward_error < 0.25
        or backward_baseline - backward_error < 0.25
        or forward_error > forward_baseline * 0.85
        or backward_error > backward_baseline * 0.85
    ):
        return None

    return LocalFit(
        LocalWarp(to_second, first_points[stable], forward_delta[stable], radius),
        LocalWarp(to_first, second_points[stable], backward_delta[stable], radius),
        forward_baseline,
        forward_error,
        count,
    )

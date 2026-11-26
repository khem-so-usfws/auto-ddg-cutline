"""Review three-frame overlaps and find possible conflicting count handoffs."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .geometry import Coordinates, Seam, intersect, project, projected_footprint


@dataclass(frozen=True)
class TripleReview:
    """A three-photo overlap identified by the pairwise image transforms."""

    message: str
    uncertain: bool
    outer_redundant: bool = False
    sampled_points: int | None = None
    conflicting_points: int | None = None
    alignment_p90_working_px: float | None = None


def sample_footprint(footprint: Coordinates) -> Coordinates:
    """Return regularly spaced review samples within a convex footprint."""
    minimum: Coordinates = footprint.min(axis=0)
    maximum: Coordinates = footprint.max(axis=0)
    x: Coordinates
    y: Coordinates
    x, y = np.meshgrid(
        np.linspace(minimum[0], maximum[0], 36),
        np.linspace(minimum[1], maximum[1], 36),
    )
    candidates: Coordinates = np.column_stack((x.ravel(), y.ravel()))
    contour: NDArray[np.float32] = footprint.astype(np.float32)
    inside: NDArray[np.bool_] = np.array(
        [cv2.pointPolygonTest(contour, (px, py), False) >= 0 for px, py in candidates]
    )
    return candidates[inside]


def outer_covered_by_middle(ab: Seam, bc: Seam, ac: Seam) -> bool:
    """Check whether the two adjacent handoffs already partition the outer match.

    Args:
        ab: Handoff from the first to the middle photo.
        bc: Handoff from the middle to the final photo.
        ac: Direct match between the first and final photos.

    Returns:
        True when the matched outer terrain is covered and assigned once.
    """
    if ac.matched_pairs is None or ab.warnings or bc.warnings:
        return False
    first_points: Coordinates
    last_points: Coordinates
    first_points, last_points = ac.matched_pairs
    if len(first_points) < 24:
        return False
    middle_from_first: Coordinates = ab.map_to_second(first_points)
    middle_from_last: Coordinates = bc.map_to_first(last_points)
    drift: Coordinates = np.linalg.norm(middle_from_first - middle_from_last, axis=1)
    middle: Coordinates = (middle_from_first + middle_from_last) / 2
    inside: NDArray[np.bool_] = (
        (middle[:, 0] >= 0)
        & (middle[:, 0] <= ab.second.gray.shape[1] - 1)
        & (middle[:, 1] >= 0)
        & (middle[:, 1] <= ab.second.gray.shape[0] - 1)
    )
    if inside.mean() < 0.98 or np.quantile(drift, 0.90) > 8:
        return False
    outer_in_middle: Coordinates = project(ac.overlap, ab.to_second)
    shared: Coordinates | None = intersect(outer_in_middle, ab.second.footprint)
    if shared is None:
        return False
    area: float = cv2.contourArea(outer_in_middle.astype(np.float32))
    if cv2.contourArea(shared.astype(np.float32)) < 0.97 * area:
        return False

    # When both outer photos would claim the same location, the direct handoff
    # is still necessary. Use the jointly matched terrain, with a pixel margin.
    first_claim: NDArray[np.bool_] = ab.offset(first_points) < -3
    last_claim: NDArray[np.bool_] = bc.offset(middle) > 3
    if np.any(first_claim & last_claim):
        return False
    # Check between the feature matches too, where a conflict could go unseen.
    samples: Coordinates = sample_footprint(ac.overlap)
    if not len(samples):
        return False
    first_sample_claim: NDArray[np.bool_] = ab.offset(samples) < -3
    last_sample_claim: NDArray[np.bool_] = bc.offset(ab.map_to_second(samples)) > 3
    return not bool(np.any(first_sample_claim & last_sample_claim))


def review_triple(ab: Seam, bc: Seam, ac: Seam | None) -> TripleReview | None:
    """Look for unstable alignment or contradictory ownership in a triplet.

    Args:
        ab: Handoff between the first and middle photos.
        bc: Handoff between the middle and last photos.
        ac: Direct handoff between the first and last photos, if found.

    Returns:
        A review message if the estimated footprints share an area.
    """
    first_in_middle: Coordinates | None = projected_footprint(ab.first, ab.to_second)
    if first_in_middle is None:
        return None
    common: Coordinates | None = intersect(bc.overlap, first_in_middle)
    if common is None or cv2.contourArea(common.astype(np.float32)) < 100:
        return None

    if ac is None:
        return TripleReview(
            "Estimated three-photo overlap. The outer pair could not be matched.",
            True,
        )

    if outer_covered_by_middle(ab, bc, ac):
        return TripleReview(
            "Estimated three-photo overlap. The adjacent handoffs cover the matched "
            "terrain, so no separate first-to-last cutline is needed.",
            False,
            True,
        )

    # Sample the shared footprint on the middle frame. A point must meet both
    # handoffs to have a single owner in the three-photo overlap.
    points: Coordinates = sample_footprint(common)
    if not len(points):
        return TripleReview("Small estimated three-photo overlap. Check in DDG.", True)

    first_points: Coordinates = ab.map_to_first(points)
    direct_last: Coordinates = ac.map_to_second(first_points)
    middle_last: Coordinates = bc.map_to_second(points)
    drift: float = float(
        np.quantile(np.linalg.norm(direct_last - middle_last, axis=1), 0.9)
    )

    ab_offsets: Coordinates = ab.offset(first_points)
    ac_offsets: Coordinates = ac.offset(first_points)
    bc_offsets: Coordinates = bc.offset(points)
    near_boundary: NDArray[np.bool_] = (
        (abs(ab_offsets) < 2) | (abs(ac_offsets) < 2) | (abs(bc_offsets) < 2)
    )
    first_owns: NDArray[np.bool_] = (ab_offsets < 0) & (ac_offsets < 0)
    middle_owns: NDArray[np.bool_] = (ab_offsets > 0) & (bc_offsets < 0)
    last_owns: NDArray[np.bool_] = (ac_offsets > 0) & (bc_offsets > 0)
    conflict: NDArray[np.bool_] = (
        ~(first_owns | middle_owns | last_owns) & ~near_boundary
    )
    conflicts: int = int(conflict.sum())
    messages: list[str] = []
    if drift > 6:
        messages.append(f"alignment paths disagree by {drift:.1f} working pixels (p90)")
    if conflicts:
        messages.append(
            f"{conflicts / len(points):.1%} of samples have conflicting handoffs"
        )
    if any(seam.sides() is None for seam in (ab, bc, ac)):
        messages.append("a count side is ambiguous")
    if messages:
        return TripleReview(
            "Estimated three-photo overlap: " + "; ".join(messages) + ".",
            True,
            sampled_points=len(points),
            conflicting_points=conflicts,
            alignment_p90_working_px=drift if np.isfinite(drift) else None,
        )
    return TripleReview(
        "Estimated three-photo overlap. Sampled handoffs agree; inspect this area in DDG.",
        False,
        sampled_points=len(points),
        conflicting_points=conflicts,
        alignment_p90_working_px=drift if np.isfinite(drift) else None,
    )

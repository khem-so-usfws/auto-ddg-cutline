"""Plan continuous overlap handoffs and record the side each photo keeps."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from .benchmark import RunReport
from .colmap_model import Photo
from .geometry import Coordinates, Frame, Pixels, Seam, compass, load_frame, propose
from .polygons import edges_cross
from .sequence import TripleReview, review_triple

IndexPair: TypeAlias = tuple[int, int]
PointPair: TypeAlias = tuple[Coordinates, Coordinates]
PhotoPair: TypeAlias = tuple[Path, Path]


def review_handoffs(plan: dict[str, Any]) -> dict[IndexPair, dict[str, Any]]:
    """Find the geometry that will be exported, preferring primary handoffs."""
    drafts: dict[IndexPair, dict[str, Any]] = {
        tuple(entry["photos"]): entry["handoff"]
        for entry in [
            *plan.get("local_sfm", {}).get("drafts", []),
            *plan["overlap_pairs"],
        ]
        if "handoff" in entry
    }
    return {tuple(pair): drafts[tuple(pair)] for pair in plan["export_pairs"]}


@dataclass(frozen=True)
class CountSide:
    """A keep-side marker and direction in one photo's original coordinates."""

    point: Coordinates
    direction: str
    side: Literal["left", "right"]


@dataclass(frozen=True)
class Handoff:
    """A continuous cutline projected into two photos with opposite ownership."""

    seam: Seam
    sides: tuple[CountSide, CountSide]
    endpoint_mode: Literal["movable", "fixed"]
    group_rerouted: bool = False

    @property
    def pair(self) -> PhotoPair:
        """Return the photo paths in cutline order."""
        return self.seam.first.path, self.seam.second.path


def stage_frame(
    photo: Photo, image_folder: Path, max_edge: int = 1800, *, draft_color: bool = False
) -> Frame:
    """Read a small working raster while retaining native source dimensions."""
    source: Path = image_folder / photo.staged_name
    frame: Frame = load_frame(source, max_edge, None, "raw")
    return replace(
        frame, path=photo.path, draft_color_source=source if draft_color else None
    )


def reduced_frame(frame: Frame, max_edge: int) -> Frame:
    """Derive a seed raster from a staged native frame without decoding it twice."""
    height: int
    width: int
    height, width = frame.gray.shape
    factor: float = max_edge / max(height, width)
    if factor >= 1:
        return frame
    size: tuple[int, int] = (
        max(1, round(width * factor)),
        max(1, round(height * factor)),
    )
    gray: Pixels = cv2.resize(frame.gray, size, interpolation=cv2.INTER_AREA)
    return replace(frame, gray=gray, protected_polygons=())


def working_points(frame: Frame, points: Coordinates) -> Coordinates:
    """Map original pixel centers to the seam planner's working raster."""
    return (points + 0.5) * frame.scale - 0.5


def clip_interval(
    first: Coordinates, last: Coordinates, maximum: Coordinates
) -> tuple[float, float] | None:
    """Clip the same segment parameter to the bounds of both image grids."""
    start: float = 0.0
    end: float = 1.0
    for low, high, limit in zip(first, last, maximum, strict=True):
        change: float = float(high - low)
        if abs(change) < 1e-10:
            if not -1e-8 <= low <= limit + 1e-8:
                return None
            continue
        a: float
        b: float
        a, b = sorted((-float(low) / change, float(limit - low) / change))
        start, end = max(start, a), min(end, b)
    return (start, end) if start < end - 1e-8 else None


def clip_handoff(seam: Seam) -> tuple[Seam | None, str]:
    """Trim paired breakpoints or explain why the visible route is invalid."""
    maximum: Coordinates = np.array(
        [
            seam.first.gray.shape[1] - 1,
            seam.first.gray.shape[0] - 1,
            seam.second.gray.shape[1] - 1,
            seam.second.gray.shape[0] - 1,
        ],
        dtype=np.float64,
    )
    combined: Coordinates = np.column_stack((seam.points, seam.other_points()))
    if not np.isfinite(combined).all():
        return None, "projection produced nonfinite coordinates"
    points: list[Coordinates] = []
    for first, last in zip(combined[:-1], combined[1:], strict=True):
        interval: tuple[float, float] | None = clip_interval(first, last, maximum)
        if interval is None:
            continue
        a: float
        b: float
        a, b = interval
        start: Coordinates = first + a * (last - first)
        end: Coordinates = first + b * (last - first)
        if points:
            if not np.allclose(points[-1], start, rtol=0, atol=1e-6):
                return None, "the route leaves and reenters the shared image bounds"
        else:
            points.append(start)
        if not np.allclose(points[-1], end, rtol=0, atol=1e-6):
            points.append(end)
    if len(points) < 2:
        return None, "no segment remains inside both images"
    clipped: Coordinates = np.clip(np.asarray(points), 0, maximum)
    first_points: Coordinates = clipped[:, :2]
    second_points: Coordinates = clipped[:, 2:]
    for frame, vertices in ((seam.first, first_points), (seam.second, second_points)):
        rounded: Coordinates = np.round(vertices, 4)
        if np.linalg.norm(rounded[-1] - rounded[0]) < 14:
            return None, f"the clipped route is too short in {frame.path.name}"
        if len(np.unique(rounded, axis=0)) != len(rounded) or edges_cross(
            rounded, closed=False
        ):
            return (
                None,
                f"the clipped route doubles back or crosses itself in {frame.path.name}",
            )
    return replace(seam, points=first_points, second_points=second_points), ""


def count_sides(seam: Seam) -> tuple[CountSide, CountSide] | None:
    """Locate a keep point on each handoff side at the same boundary position."""
    if seam.sides() is None:
        return None
    first: Coordinates = seam.points
    second: Coordinates = seam.other_points()
    lengths: Coordinates = np.linalg.norm(np.diff(first, axis=0), axis=1)
    accumulated: Coordinates = np.r_[0.0, lengths.cumsum()]
    halfway: float = float(accumulated[-1] / 2)
    index: int = int(np.searchsorted(accumulated, halfway, side="right") - 1)
    fraction: float = (halfway - accumulated[index]) / float(lengths[index])
    first_center: Coordinates = first[index] + fraction * (
        first[index + 1] - first[index]
    )
    second_center: Coordinates = second[index] + fraction * (
        second[index + 1] - second[index]
    )
    for distance in (12, 6, 3):
        # Both probes must be inside the overlap, and each photo keeps its
        # opposite half. Align the local displacement to the exported vertex pair.
        points: Coordinates = (
            first_center + np.array([-distance, 0, distance])[:, None] * seam.normal
        )
        mapped: Coordinates = seam.map_to_second(points)
        mapped += second_center - mapped[1]
        sides: list[CountSide] = []
        for frame, vertices, center, probes, keep in (
            (seam.first, first, first_center, points[[0, 2]], 0),
            (seam.second, second, second_center, mapped[[0, 2]], 1),
        ):
            limit: Coordinates = frame.footprint[2]
            if not ((probes >= 0) & (probes <= limit)).all():
                break
            vector: Coordinates = vertices[index + 1] - vertices[index]
            offsets: Coordinates = probes - center
            cross: Coordinates = (
                vector[0] * offsets[:, 1] - vector[1] * offsets[:, 0]
            ) / np.linalg.norm(vector)
            if cross[0] * cross[1] >= 0 or abs(cross[keep]) < 1:
                break
            side: Literal["left", "right"] = "left" if cross[keep] < 0 else "right"
            direction: str = compass(offsets[keep])
            sides.append(
                CountSide(
                    frame.source_points(probes[keep : keep + 1])[0], direction, side
                )
            )
        if len(sides) == 2:
            return sides[0], sides[1]
    return None


def crosses_protected_group(seam: Seam) -> bool:
    """Check every segment of the final simplified line in both image grids."""
    for frame, points in (
        (seam.first, seam.points),
        (seam.second, seam.other_points()),
    ):
        mask: Pixels | None = frame.protected_mask
        if mask is None:
            continue
        left: int
        top: int
        width: int
        height: int
        left, top, width, height = cv2.boundingRect(points.astype(np.float32))
        right: int = min(left + width + 1, mask.shape[1])
        bottom: int = min(top + height + 1, mask.shape[0])
        group: Pixels = mask[top:bottom, left:right]
        if not group.any():
            continue
        line: Pixels = np.zeros_like(group)
        vertices: NDArray[np.int32] = np.rint((points - [left, top]) * 16).astype(
            np.int32
        )
        cv2.polylines(line, [vertices], False, 1, shift=4)
        if cv2.bitwise_and(line, group).any():
            return True
    return False


def try_handoff(
    first: Frame,
    second: Frame,
    matches: PointPair,
    *,
    move_endpoints: bool,
    seed: Coordinates | None = None,
) -> tuple[Handoff | None, str]:
    """Propose a route and distinguish registration, clipping, and side failures."""
    seam: Seam | None
    reason: str
    seam, reason = propose(
        first,
        second,
        "terrain",
        matches,
        follow_detail=True,
        move_endpoints=move_endpoints,
        seed=seed,
    )
    if seam is None:
        return None, reason
    bounded: Seam | None
    bounded, reason = clip_handoff(seam)
    if bounded is None:
        return None, f"clipping failed: {reason}"
    rerouted: bool = crosses_protected_group(bounded)
    if rerouted:
        guides: list[Coordinates | None] = [bounded.points]
        if move_endpoints:
            guides.append(None)
        for guide in guides:
            adjusted: Seam | None
            adjusted, _ = propose(
                first,
                second,
                "terrain",
                matches,
                follow_detail=True,
                move_endpoints=guide is None,
                seed=guide,
                avoid_groups=True,
            )
            if adjusted is not None:
                adjusted, _ = clip_handoff(adjusted)
                if adjusted is not None and not crosses_protected_group(adjusted):
                    bounded = adjusted
                    move_endpoints = guide is None
                    break
        else:
            return None, "No verified cutline clears the protected DDG polygons."
    sides: tuple[CountSide, CountSide] | None = count_sides(bounded)
    if sides is None:
        explanation: str = next(
            (warning for warning in bounded.warnings if "count side" in warning),
            "count side is ambiguous because opposite keep points could not be verified in both photos",
        )
        return None, explanation
    return (
        Handoff(bounded, sides, "movable" if move_endpoints else "fixed", rerouted),
        "",
    )


def plan_pair(
    a: Frame, b: Frame, evidence: PointPair | None, adjacent: bool, notes: list[str]
) -> tuple[Handoff | None, str]:
    """Evaluate the existing movable and fixed routes for a single candidate."""
    pair: str = f"{a.path.name} / {b.path.name}"
    if evidence is None:
        if adjacent:
            notes.append(f"Unresolved {pair}: no verified overlap.")
        return None, "no verified overlap"
    working: PointPair = (
        working_points(a, evidence[0]),
        working_points(b, evidence[1]),
    )
    handoff: Handoff | None
    reason: str
    handoff, reason = try_handoff(a, b, working, move_endpoints=True)
    if handoff is None:
        fixed_reason: str
        handoff, fixed_reason = try_handoff(a, b, working, move_endpoints=False)
        if handoff is None:
            details: str = (
                reason
                if reason == fixed_reason
                else f"movable endpoints: {reason}; fixed endpoints: {fixed_reason}"
            )
            notes.append(f"Unresolved {pair}: {details}.")
            return None, details
        notes.append(f"Fixed endpoints for {pair}: {reason}.")
    return handoff, ""


def plan_handoffs(
    frames: list[Frame],
    matches: dict[IndexPair, PointPair],
    report: RunReport | None = None,
    quality: dict[str, Any] | None = None,
) -> tuple[list[Handoff], set[PhotoPair], list[str]]:
    """Plan ordered handoffs, then check their agreement across photo triplets.

    Args:
        frames: Photos in survey order, with their working pixel grids.
        matches: Verified native-pixel matches keyed by photo index pair.
        report: Optional progress reporter.
        quality: Optional manifest record to receive pair diagnostics.

    Returns:
        Paired handoffs, image pairs requiring extra review, and review notes.
    """
    candidates: dict[IndexPair, Handoff] = {}
    pair_outcomes: dict[IndexPair, dict[str, Any]] = {}
    drafts: set[PhotoPair] = set()
    notes: list[str] = []
    pairs: list[IndexPair] = [
        (a, b)
        for a in range(len(frames))
        for b in range(a + 1, min(a + 3, len(frames)))
    ]
    unresolved: int = 0
    with report.phase("pair_planning") if report is not None else nullcontext():
        for number, (first, second) in enumerate(pairs, start=1):
            a: Frame = frames[first]
            b: Frame = frames[second]
            adjacent: bool = second - first == 1
            evidence: PointPair | None = matches.get((first, second))
            handoff: Handoff | None
            reason: str
            handoff, reason = plan_pair(a, b, evidence, adjacent, notes)
            outcome: dict[str, Any] = {
                "images": [a.path.name, b.path.name],
                "scope": "adjacent" if adjacent else "skip_one",
                "status": "unresolved",
                "reason": reason or None,
            }
            if handoff is not None:
                candidates[first, second] = handoff
                outcome["status"] = "proposed"
                if handoff.seam.warnings:
                    drafts.add(handoff.pair)
                    notes.extend(
                        f"Review {a.path.name} / {b.path.name}: {warning}."
                        for warning in handoff.seam.warnings
                    )
            else:
                unresolved += 1
            pair_outcomes[first, second] = outcome
            if report is not None:
                report.update(
                    f"{len(candidates) - len(drafts)} proposed, {len(drafts)} draft, {unresolved} unresolved before sequence checks",
                    number,
                    len(pairs),
                )

    with report.phase("sequence_checks") if report is not None else nullcontext():
        triplets: list[dict[str, Any]] = check_sequence(
            frames, candidates, drafts, notes, report
        )
        for key, outcome in pair_outcomes.items():
            if outcome["status"] == "proposed":
                handoff = candidates.get(key)
                if handoff is None:
                    outcome.update(
                        status="redundant", reason="covered by both adjacent handoffs"
                    )
                elif handoff.pair in drafts:
                    outcome["status"] = "draft"
        if quality is not None:
            quality["pair_outcomes"] = list(pair_outcomes.values())
            quality["triplets"] = triplets
            quality["triplets_flagged"] = sum(item["needs_review"] for item in triplets)
    return list(candidates.values()), drafts, notes


def check_sequence(
    frames: list[Frame],
    candidates: dict[IndexPair, Handoff],
    drafts: set[PhotoPair],
    notes: list[str],
    report: RunReport | None,
) -> list[dict[str, Any]]:
    """Apply the existing triplet checks and save the resulting review status."""
    triplets: list[dict[str, Any]] = []
    for start in range(len(frames) - 2):
        left: Handoff | None = candidates.get((start, start + 1))
        right: Handoff | None = candidates.get((start + 1, start + 2))
        outer: Handoff | None = candidates.get((start, start + 2))
        if left is None or right is None:
            continue
        review: TripleReview | None = review_triple(
            left.seam, right.seam, outer.seam if outer else None
        )
        if review is None:
            continue
        names: list[str] = [frame.path.name for frame in frames[start : start + 3]]
        notes.append(f"{' / '.join(names)}: {review.message}")
        triplets.append(
            {
                "images": names,
                "needs_review": review.uncertain,
                "reason": review.message,
                "sampled_points": review.sampled_points,
                "conflicting_points": review.conflicting_points,
                "alignment_p90_working_px": review.alignment_p90_working_px,
            }
        )
        if review.outer_redundant and outer is not None:
            del candidates[start, start + 2]
            drafts.discard(outer.pair)
        if review.uncertain:
            drafts.update(
                part.pair for part in (left, right, outer) if part is not None
            )
        if report is not None:
            report.update("three-photo overlap checked", start + 1, len(frames) - 2)
    return triplets

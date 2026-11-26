"""Run small COLMAP reconstructions and draft handoffs from shared 3D tracks."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .benchmark import RunReport
from .colmap import Colmap, Flags, map_matches
from .colmap_model import Model, Photo
from .coverage import validate_tracks
from .geometry import Coordinates, Frame
from .match_cache import MatchAssets, copy_database
from .paired import Handoff, IndexPair, plan_pair
from .selection import describe_handoff

MIN_SHARED_TRACKS: int = 16
MIN_THREE_VIEW_TRACKS: int = 8


def draft_for_pair(
    pair: IndexPair,
    model: Model,
    views: dict[Path, int],
    observations: dict[int, dict[int, Coordinates]],
    scale: float,
    frames: list[Frame],
) -> tuple[Handoff | None, dict[str, Any]]:
    """Use well-reprojected 3D landmarks to propose a line in both photos.

    Args:
        pair: Photo indices for the pending cutline.
        model: Sparse reconstruction of a locally connected photo block.
        views: Registered image IDs keyed by source photo path.
        observations: Validated, native-pixel observations by track and view.
        scale: Largest camera baseline in the block, in model units.
        frames: Survey frames indexed by the pair.

    Returns:
        A proposed handoff if supported, together with the review evidence.
    """
    first: int
    second: int
    first, second = pair
    a: int | None = views.get(frames[first].path)
    b: int | None = views.get(frames[second].path)
    evidence: dict[str, Any] = {"model": model.name}
    if a is None or b is None:
        return None, {
            **evidence,
            "reason": "Both photos must register in the same local model.",
        }
    baseline: float = float(
        np.linalg.norm(model.views[a].center - model.views[b].center)
    )
    if not np.isfinite(baseline) or not np.isfinite(scale) or baseline <= scale * 1e-5:
        return None, {
            **evidence,
            "reason": "The two camera centers are too close for a reliable handoff.",
        }
    shared: list[dict[int, Coordinates]] = [
        points
        for _, points in sorted(observations.items())
        if a in points and b in points
    ]
    other_views: set[int] = {
        index
        for index, view in model.views.items()
        if index not in (a, b)
        and all(
            np.linalg.norm(view.center - model.views[target].center) > scale * 1e-5
            for target in (a, b)
        )
    }
    three_view: list[dict[int, Coordinates]] = [
        points for points in shared if not other_views.isdisjoint(points)
    ]
    evidence.update(shared_tracks=len(shared), three_view_tracks=len(three_view))
    if len(shared) < MIN_SHARED_TRACKS or len(three_view) < MIN_THREE_VIEW_TRACKS:
        return None, {
            **evidence,
            "reason": "Too few validated shared tracks, including support from a third view.",
        }
    first_points: Coordinates = np.array([points[a] for points in three_view])
    second_points: Coordinates = np.array([points[b] for points in three_view])
    handoff: Handoff | None
    reason: str
    handoff, reason = plan_pair(
        frames[first], frames[second], (first_points, second_points), True, []
    )
    if handoff is None:
        return None, {**evidence, "reason": reason}
    if handoff.seam.midpoint_fallback:
        return None, {**evidence, "reason": handoff.seam.midpoint_fallback}
    samples: NDArray[np.intp] = np.linspace(
        0, len(three_view) - 1, min(24, len(three_view)), dtype=np.intp
    )
    evidence["matched_landmarks_native"] = [
        np.round(points[samples], 3).tolist()
        for points in (first_points, second_points)
    ]
    return handoff, evidence


def block_drafts(
    targets: set[IndexPair], models: list[Model], frames: list[Frame]
) -> tuple[dict[IndexPair, Handoff], dict[IndexPair, dict[str, Any]]]:
    """Try each usable model and retain the best-supported result per target."""
    handoffs: dict[IndexPair, Handoff] = {}
    evidence: dict[IndexPair, dict[str, Any]] = {}
    for model in models:
        observations: dict[int, dict[int, Coordinates]]
        observations, _ = validate_tracks(model)
        views: dict[Path, int] = {
            view.photo.path: index for index, view in model.views.items()
        }
        scale: float = max(
            (
                float(np.linalg.norm(a.center - b.center))
                for a, b in combinations(model.views.values(), 2)
            ),
            default=0.0,
        )
        for pair in sorted(targets):
            draft: Handoff | None
            details: dict[str, Any]
            draft, details = draft_for_pair(
                pair, model, views, observations, scale, frames
            )
            priority: tuple[bool, int, int] = draft_priority(draft is not None, details)
            previous: tuple[bool, int, int] = draft_priority(
                pair in handoffs, evidence.get(pair, {})
            )
            if pair not in evidence or priority > previous:
                evidence[pair] = details
                if draft is not None:
                    handoffs[pair] = draft
    return handoffs, evidence


def draft_priority(passed: bool, evidence: dict[str, Any]) -> tuple[bool, int, int]:
    """Prefer valid drafts and then stronger support from three or more views."""
    return (
        passed,
        evidence.get("three_view_tracks", 0),
        evidence.get("shared_tracks", 0),
    )


def run_local_sfm(
    plan: dict[str, Any],
    engine: Colmap,
    flags: Flags,
    assets: MatchAssets,
    photos: list[Photo],
    frames: list[Frame],
    job: Path,
    report: RunReport,
    *,
    unresolved_only: bool = False,
) -> list[Handoff]:
    """Run actionable blocks against a private match copy and record draft cutlines."""
    unresolved: set[IndexPair] = {
        tuple(pair) for pair in plan.get("unresolved_handoffs", [])
    }
    queue: list[dict[str, Any]] = [
        entry
        for entry in plan["suggested_local_sfm"]
        if entry["recommended_action"] == "local SfM candidate"
        and (
            not unresolved_only
            or any(set(pair) <= set(entry["photos"]) for pair in unresolved)
        )
    ]
    results: list[dict[str, Any]] = []
    drafts: dict[IndexPair, dict[str, Any]] = {}
    handoffs: dict[IndexPair, Handoff] = {}
    plan["local_sfm"] = {"blocks": results, "drafts": []}
    if not queue:
        return []
    if flags.mapper_image_list is None:
        plan["local_sfm"]["reason"] = "COLMAP mapper has no image-list option."
        return []

    with report.phase("local_sfm_database_copy"):
        database: Path = job / "local-sfm" / "database.db"
        copy_database(assets.database, database)
    failed: set[IndexPair] = {
        tuple(entry["photos"])
        for entry in plan["overlap_pairs"]
        if "handoff_error" in entry
    }
    for number, entry in enumerate(queue, start=1):
        members: set[int] = set(entry["photos"])
        block: Path = database.parent / f"block-{number:02d}"
        targets: set[IndexPair] = (
            {pair for pair in unresolved if members.issuperset(pair)}
            if unresolved_only
            else set(combinations(sorted(entry["focus"]), 2))
            | {pair for pair in failed if members.issuperset(pair)}
        )
        subset: dict[str, Photo] = {
            photos[i].staged_name: photos[i] for i in sorted(members)
        }
        row: dict[str, Any] = {
            "block": number,
            "photos": sorted(members),
            "folder": str(block),
        }
        results.append(row)
        print(
            f"Local SfM {number}/{len(queue)}: {', '.join(photo.path.name for photo in subset.values())}",
            flush=True,
        )
        try:
            models: list[Model] = map_matches(
                engine, flags, database, assets.image_folder, subset, subset_job=block
            )
            with report.phase("local_sfm_cutlines", f"block {number}"):
                block_handoffs: dict[IndexPair, Handoff]
                evidence: dict[IndexPair, dict[str, Any]]
                block_handoffs, evidence = block_drafts(targets, models, frames)
        except (OSError, ValueError, cv2.error) as error:
            row.update(status="failed", reason=str(error))
            continue
        row.update(
            status="drafts generated" if block_handoffs else "no supported cutline",
            models=[
                {
                    "name": model.name,
                    "registered": sorted(
                        view.photo.path.name for view in model.views.values()
                    ),
                    "reconstructed_tracks": len(model.tracks),
                }
                for model in models
            ],
            checks=[
                {
                    "photos": list(pair),
                    **{
                        key: value
                        for key, value in evidence.get(pair, {}).items()
                        if key != "matched_landmarks_native"
                    },
                    "status": "draft" if pair in block_handoffs else "unresolved",
                }
                for pair in sorted(targets)
            ],
        )
        for pair, handoff in block_handoffs.items():
            details: dict[str, Any] = evidence[pair]
            if pair in drafts and draft_priority(True, drafts[pair]) >= draft_priority(
                True, details
            ):
                continue
            drafts[pair] = {
                "photos": list(pair),
                "block": number,
                **details,
                "handoff": describe_handoff(pair, handoff),
            }
            handoffs[pair] = handoff
    plan["local_sfm"]["drafts"] = [drafts[pair] for pair in sorted(drafts)]
    return [handoffs[pair] for pair in sorted(handoffs)]

"""Compare photo-local selection with supported patches from sparse alignment."""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .benchmark import RunReport
from .colmap import Colmap, Flags, indexed_matches, map_matches
from .colmap_model import Model, Photo, model_diagnostics
from .coverage import Face, surface_faces, validate_tracks
from .geometry import Coordinates, Frame
from .match_cache import MatchAssets, copy_database
from .paired import Handoff, IndexPair, PointPair, plan_pair
from .selection import Rules, describe_handoff, pair_index, recommend
from .selection_evidence import Tile

MAX_BLOCK_PHOTOS: int = 8
MIN_RELATIVE_LINEAR_DETAIL: float = 0.70


def select_block(
    models: list[Model], index: dict[Path, int], plan: dict[str, Any]
) -> tuple[Model, list[int], str] | None:
    """Choose one small block in a single model, preferring a flagged join.

    Args:
        models: Independent sparse reconstructions of the whole folder.
        index: Map from native photo paths to initial plan indices.
        plan: Initial plan with the photo-local comparison and overlap evidence.

    Returns:
        A model, ordered photo indices, and an explanation of the block choice.
    """
    groups: list[tuple[Model, set[int]]] = [
        (model, {index[view.photo.path] for view in model.views.values()})
        for model in sorted(models, key=lambda model: (-len(model.views), model.name))
        if len(model.views) >= 3
    ]
    for suggestion in plan["suggested_local_sfm"]:
        focus: set[int] = set(suggestion["focus"])
        for model, registered in groups:
            members: set[int] = registered.intersection(suggestion["photos"])
            if focus <= members and len(members) >= 3:
                return model, sorted(members), suggestion["reason"]

    if not groups:
        return None
    model: Model
    registered: set[int]
    model, registered = groups[0]
    overlaps: list[tuple[IndexPair, int]] = [
        ((row["photos"][0], row["photos"][1]), row["inliers"])
        for row in plan["overlap_pairs"]
        if row["inliers"] and registered.issuperset(row["photos"])
    ]
    if not overlaps:
        return None
    focus_pair: IndexPair = min(overlaps, key=lambda item: (-item[1], item[0]))[0]
    evidence: dict[IndexPair, int] = dict(overlaps)
    members = set(focus_pair)
    while len(members) < MAX_BLOCK_PHOTOS and registered - members:
        support: dict[int, int] = {
            candidate: sum(
                evidence.get(pair_index(candidate, peer), 0) for peer in members
            )
            for candidate in registered - members
        }
        best: int = min(support, key=lambda candidate: (-support[candidate], candidate))
        if not support[best]:
            break
        members.add(best)
    if len(members) < 3:
        return None
    return model, sorted(members), "Strongest verified overlap in the largest model."


def patch_cover(
    faces: list[Face], model: Model, index: dict[Path, int], photo_count: int
) -> list[set[Tile]]:
    """Record which observed views offer comparable projected detail on a patch.

    Args:
        faces: Validated surface patches with projections in their observing views.
        model: Sparse reconstruction containing the observing cameras.
        index: Map from source paths to photo indices.
        photo_count: Size of the full input photo list.

    Returns:
        The set-cover observations associated with each photo.
    """
    covers: list[set[Tile]] = [set() for _ in range(photo_count)]
    for patch, face in enumerate(faces):
        areas: dict[int, float] = {
            view: cv2.contourArea(pixels.astype(np.float32))
            for view, (pixels, _) in face.views.items()
        }
        floor: float = max(areas.values()) * MIN_RELATIVE_LINEAR_DETAIL**2
        for view, area in areas.items():
            if area >= floor:
                # The first tuple entry identifies this single reconstructed block.
                covers[index[model.views[view].photo.path]].add((0, patch))
    return covers


def compare_handoffs(
    database: Path, photos: list[Photo], frames: list[Frame], orders: list[list[int]]
) -> list[list[dict[str, Any]]]:
    """Test both counting sets with the same direct, filename-order seam planner."""
    sequences: list[list[IndexPair]] = [list(zip(order, order[1:])) for order in orders]
    unique: list[IndexPair] = sorted({pair for pairs in sequences for pair in pairs})
    matches: dict[IndexPair, PointPair] = indexed_matches(database, photos, unique)
    tested: dict[IndexPair, dict[str, Any]] = {}
    for first, second in unique:
        pair: IndexPair = (first, second)
        handoff: Handoff | None
        reason: str
        handoff, reason = plan_pair(
            frames[first], frames[second], matches.get(pair), True, []
        )
        row: dict[str, Any] = {"photos": list(pair)}
        if handoff is None:
            row["handoff_error"] = reason
        else:
            row["handoff"] = describe_handoff(pair, handoff)
        tested[pair] = row
    return [[tested[pair] for pair in pairs] for pairs in sequences]


def run_shared_patch_pilot(
    plan: dict[str, Any],
    engine: Colmap,
    flags: Flags,
    assets: MatchAssets,
    photos: dict[str, Photo],
    frames: list[Frame],
    rules: Rules,
    job: Path,
    report: RunReport,
) -> dict[str, Any]:
    """Align all photos on a database copy and compare selection in one model block."""
    database: Path = job / "global-alignment" / "database.db"
    result: dict[str, Any] = {"status": "unavailable"}
    ordered: list[Photo] = list(photos.values())
    index: dict[Path, int] = {photo.path: i for i, photo in enumerate(ordered)}
    try:
        with report.phase("global_alignment_database_copy"):
            copy_database(assets.database, database)
        # COLMAP reports its own mapper and conversion stages.
        models: list[Model] = map_matches(
            engine, flags, database, assets.image_folder, photos
        )
        with report.phase("global_alignment_diagnostics"):
            report.manifest["models"] = [
                model_diagnostics(model, photos) for model in models
            ]
            registered: set[int] = {
                index[view.photo.path]
                for model in models
                for view in model.views.values()
            }
            result.update(
                model_count=len(models),
                registered_photos=sorted(registered),
                unregistered_photos=sorted(set(range(len(ordered))) - registered),
            )
            block: tuple[Model, list[int], str] | None = select_block(
                models, index, plan
            )
        if block is None:
            result["reason"] = (
                "No connected block of at least three registered photos was found."
            )
            return result
        model: Model
        members: list[int]
        reason: str
        model, members, reason = block
        result.update(model=model.name, block_photos=members, block_reason=reason)
        with report.phase("shared_patch_selection"):
            view_ids: set[int] = {
                view_id
                for view_id, view in model.views.items()
                if index[view.photo.path] in members
            }
            observations: dict[int, dict[int, Coordinates]]
            observations, _ = validate_tracks(model)
            local: dict[int, dict[int, Coordinates]] = {
                track: {
                    view: point for view, point in views.items() if view in view_ids
                }
                for track, views in observations.items()
            }
            local = {track: views for track, views in local.items() if len(views) >= 2}
            faces: list[Face]
            rejected: Counter[str]
            faces, rejected = surface_faces(model, local)
            result.update(
                validated_tracks=len(local),
                supported_patches=len(faces),
                rejected_faces=dict(rejected),
            )
            if not faces:
                result["reason"] = (
                    "The pilot block yielded no validated surface patches."
                )
                return result
            covers: list[set[Tile]] = patch_cover(faces, model, index, len(ordered))
            local_rules: Rules = Rules(rules.keep.intersection(members), rules.exclude)
            baseline: list[int] = sorted(
                set(plan["counting_set"]).intersection(members)
            )
            alternative: list[int] = sorted(recommend(covers, local_rules))
            if not alternative:
                result["reason"] = (
                    "No eligible counting photos cover the validated patches."
                )
                return result
        with report.phase("shared_patch_handoff_comparison"):
            handoffs: list[list[dict[str, Any]]] = compare_handoffs(
                assets.database, ordered, frames, [baseline, alternative]
            )
        result["status"] = "complete"
        result["min_relative_linear_detail"] = MIN_RELATIVE_LINEAR_DETAIL
        result["photos_without_eligible_patch_evidence"] = [
            i for i in members if not covers[i]
        ]
        for label, order, joins in zip(
            ("photo_local", "shared_patch"),
            (baseline, alternative),
            handoffs,
            strict=True,
        ):
            covered: set[Tile] = {patch for i in order for patch in covers[i]}
            result[label] = {
                "photos": order,
                "supported_patches_covered": len(covered),
                "draft_handoffs": sum("handoff" in join for join in joins),
                "unresolved_handoffs": sum("handoff_error" in join for join in joins),
                "handoffs": joins,
            }
    except (OSError, sqlite3.Error, ValueError, cv2.error) as error:
        result["reason"] = str(error)
    return result

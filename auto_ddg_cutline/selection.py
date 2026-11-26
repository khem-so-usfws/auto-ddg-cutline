"""Recommend a reviewable set of counting photos from directly supported overlaps."""

from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray

from .benchmark import RunReport
from .colmap import VerifiedPair, indexed_matches
from .colmap_model import Photo
from .coverage import edge_paths
from .geometry import Coordinates, Frame, Seam, inside_frame
from .paired import (
    Handoff,
    IndexPair,
    PointPair,
    plan_pair,
    reduced_frame,
    try_handoff,
    working_points,
)
from .selection_evidence import (
    CELL_COUNT,
    GRID_COLS,
    GRID_ROWS,
    SHARPER_ALTERNATE_RATIO,
    Detail,
    FocusMap,
    Overlap,
    Tile,
    analyze_overlap,
    patch_focus_map,
    tile_detail,
)
from .sequence import TripleReview, review_triple

NEIGHBORS_PER_PHOTO: int = 6
MAX_ROUTES_PER_JOIN: int = 4
MAX_BRIDGES_PER_JOIN: int = 3
REFINEMENT_SEED_EDGE: int = 3000
DEFAULT_TARGET_PHOTO_PERCENT: int = 25


class RefinementRoutes(TypedDict):
    """Saved route requests in the checked photos' index space."""

    reviewed: list[IndexPair]
    reconnect: list[IndexPair]
    added: list[int]


@dataclass(frozen=True)
class Rules:
    """An analyst's pins and explicit omissions in input-photo index space."""

    keep: frozenset[int] = frozenset()
    exclude: frozenset[int] = frozenset()
    reasons: dict[int, str] = field(default_factory=dict)


def read_rules(path: Path | None, photos: list[Path]) -> Rules:
    """Validate exact photo filenames and preserve reasons for analyst overrides."""
    if path is None:
        return Rules()
    data: Any = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or set(data) - {
        "keep",
        "exclude",
        "exclusion_reasons",
    }:
        raise ValueError("Selection rules accept keep, exclude, and exclusion_reasons.")
    by_name: dict[str, int] = {photo.name: i for i, photo in enumerate(photos)}
    choices: dict[str, frozenset[int]] = {}
    for key in ("keep", "exclude"):
        names: Any = data.get(key, [])
        if not isinstance(names, list) or any(
            not isinstance(name, str) for name in names
        ):
            raise ValueError(f"{key} must be a list of original photo filenames.")
        unknown: set[str] = set(names) - by_name.keys()
        if unknown:
            raise ValueError(f"Unknown photo in {key}: {', '.join(sorted(unknown))}")
        choices[key] = frozenset(by_name[name] for name in names)
    if choices["keep"] & choices["exclude"]:
        raise ValueError("A photo cannot be both pinned and excluded.")
    reasons: Any = data.get("exclusion_reasons", {})
    if not isinstance(reasons, dict) or any(
        name not in by_name
        or by_name[name] not in choices["exclude"]
        or not isinstance(text, str)
        for name, text in reasons.items()
    ):
        raise ValueError(
            "Each exclusion reason must name an excluded photo and contain text."
        )
    return Rules(
        choices["keep"],
        choices["exclude"],
        {by_name[name]: text for name, text in reasons.items()},
    )


def pair_index(first: int, second: int) -> IndexPair:
    """Use canonical photo ordering for pair evidence and saved cutlines."""
    return (first, second) if first < second else (second, first)


def shortlist(counts: dict[IndexPair, int], size: int) -> list[IndexPair]:
    """Include near neighbors and the strongest longer links for each photo."""
    ranked: dict[int, list[IndexPair]] = defaultdict(list)
    selected: set[IndexPair] = set()
    for pair, count in counts.items():
        first: int
        second: int
        first, second = pair
        if second - first <= 2:
            selected.add(pair)
        if count >= 24:
            ranked[first].append(pair)
            ranked[second].append(pair)
    for index in range(size):
        selected.update(
            sorted(ranked[index], key=lambda pair: (-counts[pair], pair))[
                :NEIGHBORS_PER_PHOTO
            ]
        )
    return sorted(selected)


def coverage_sets(
    overlaps: dict[IndexPair, Overlap], size: int
) -> tuple[list[set[int]], list[set[Tile]]]:
    """Build direct coverage from observed overlaps and eligible replacements.

    Args:
        overlaps: Verified overlap evidence for each photo pair.
        size: Number of source photos, including views with no eligible overlap.

    Returns:
        Documented tiles per source photo and coverable source tiles per
        candidate counting photo.
    """
    documented: list[set[int]] = [set() for _ in range(size)]
    covers: list[set[Tile]] = [set() for _ in range(size)]
    for (first, second), overlap in overlaps.items():
        for source, alternative, cells in (
            (first, second, overlap.first),
            (second, first, overlap.second),
        ):
            for cell in cells:
                documented[source].add(cell.cell)
                covers[source].add((source, cell.cell))
                if cell.eligible:
                    covers[alternative].add((source, cell.cell))
    return documented, covers


def sharpness_weights(
    overlaps: dict[IndexPair, Overlap],
    covers: list[set[Tile]],
    rules: Rules,
) -> list[dict[Tile, float]]:
    """Discount softer eligible views without removing their coverage."""
    scores: list[dict[Tile, float]] = [{} for _ in covers]
    for (a, b), overlap in overlaps.items():
        for source, alternate, cells in ((a, b, overlap.first), (b, a, overlap.second)):
            if alternate in rules.exclude:
                continue
            scores[alternate].update(
                {
                    (source, cell.cell): cell.relative_sharpness
                    for cell in cells
                    if cell.eligible and cell.relative_sharpness is not None
                }
            )
    best: dict[Tile, float] = {}
    for values in scores:
        for tile, score in values.items():
            best[tile] = max(best.get(tile, 1.0), score)
    return [
        {
            tile: 0.70 + 0.30 * scores[index].get(tile, 1.0) / best[tile]
            for tile in cells
            if tile in best
        }
        for index, cells in enumerate(covers)
    ]


def expanded_pair_order(
    counts: dict[IndexPair, int],
    screened: list[IndexPair],
    covers: list[set[Tile]],
    baseline: set[int],
    rules: Rules,
) -> list[IndexPair]:
    """Check possible alternatives for narrowly retained views before other pairs."""
    owners: Counter[Tile] = Counter(
        tile
        for index, cells in enumerate(covers)
        if index not in rules.exclude
        for tile in cells
    )
    focus: set[int] = {
        index
        for index in baseline
        if 1 <= sum(owners[cell] == 1 for cell in covers[index]) <= 3
    }
    return sorted(
        counts.keys() - set(screened),
        key=lambda pair: (
            not (pair[0] in focus or pair[1] in focus),
            -counts[pair],
            pair,
        ),
    )


def recommend(
    covers: list[set[Tile]],
    rules: Rules,
    *,
    quality: list[dict[Tile, float]] | None = None,
) -> set[int]:
    """Cover documented tiles first, then remove or exchange redundant views."""
    eligible: set[int] = set(range(len(covers))) - rules.exclude
    target: set[Tile] = {tile for index in eligible for tile in covers[index]}
    owners: dict[Tile, set[int]] = defaultdict(set)
    for index in eligible:
        for tile in covers[index]:
            owners[tile].add(index)
    required: set[int] = set(rules.keep) | {
        next(iter(candidates)) for candidates in owners.values() if len(candidates) == 1
    }
    picked: set[int] = set(required)
    remaining: set[Tile] = target - {tile for index in picked for tile in covers[index]}

    def rank(index: int) -> tuple[int, float, int]:
        """Preserve coverage and break equally productive choices by focus."""
        new: set[Tile] = covers[index] & remaining
        focus: float = (
            sum(quality[index].get(tile, 1.0) for tile in sorted(new))
            if quality is not None
            else 0.0
        )
        return -len(new), -focus, index

    while remaining:
        choice: int = min(eligible - picked, key=rank)
        picked.add(choice)
        remaining.difference_update(covers[choice])

    def prune(options: set[int]) -> set[int]:
        """Keep required views and avoid any loss of documented coverage."""
        for index in sorted(options - required, reverse=True):
            others: set[int] = options - {index}
            if covers[index] <= {tile for other in others for tile in covers[other]}:
                options = others
        return options

    picked = prune(picked)
    for alternative in sorted(eligible - picked):
        revised: set[int] = prune(picked | {alternative})
        if len(revised) < len(picked):
            picked = revised
    return picked


class Routes:
    """Check draft handoffs lazily and find short bridges through supported overlaps."""

    def __init__(
        self,
        frames: list[Frame],
        matches: dict[IndexPair, PointPair],
        overlaps: dict[IndexPair, Overlap],
        excluded: frozenset[int],
        seed_edge: int | None = None,
    ) -> None:
        """Build an inexpensive candidate graph before proposing detailed seams."""
        self.frames: list[Frame] = frames
        self.matches: dict[IndexPair, PointPair] = matches
        self.overlaps: dict[IndexPair, Overlap] = overlaps
        self.seed_frames: list[Frame] | None = (
            [reduced_frame(frame, seed_edge) for frame in frames]
            if seed_edge is not None
            else None
        )
        self.seed_drafts: dict[IndexPair, dict[str, Any]] = {}
        self.refined_in_corridor: set[IndexPair] = set()
        self.tested: dict[IndexPair, Handoff | None] = {}
        self.failures: dict[IndexPair, str] = {}
        self.graph: dict[int, set[int]] = {
            index: set() for index in range(len(frames)) if index not in excluded
        }
        for (first, second), overlap in overlaps.items():
            if (
                first in self.graph
                and second in self.graph
                and overlap.first
                and overlap.second
            ):
                self.graph[first].add(second)
                self.graph[second].add(first)

    def check(self, pair: IndexPair) -> Handoff | None:
        """Run the existing paired cutline planner once per candidate."""
        if pair not in self.tested:
            first: int
            second: int
            first, second = pair
            notes: list[str] = []
            a: Frame
            b: Frame
            a, b = self.frames[first], self.frames[second]
            matches: PointPair | None = self.matches.get(pair)
            handoff: Handoff | None = None
            reason: str = ""
            if matches is not None and self.seed_frames is not None:
                seed_a: Frame
                seed_b: Frame
                seed_a, seed_b = self.seed_frames[first], self.seed_frames[second]
                if seed_a is not a or seed_b is not b:
                    draft: Handoff | None
                    draft, _ = plan_pair(seed_a, seed_b, matches, False, notes)
                    if draft is not None:
                        self.seed_drafts[pair] = describe_handoff(pair, draft)
                        handoff, _ = try_handoff(
                            a,
                            b,
                            matches,
                            move_endpoints=False,
                            seed=draft.seam.points_in_source_images()[0],
                        )
                        if handoff is not None:
                            self.refined_in_corridor.add(pair)
            if handoff is None:
                handoff, reason = plan_pair(a, b, matches, True, notes)
            self.tested[pair] = handoff
            if handoff is None:
                self.failures[pair] = reason
        return self.tested[pair]

    def paths(self, start: int, end: int, picked: set[int]) -> tuple[int, ...]:
        """Prefer routes adding few photos, then fewer handoffs and stronger support."""
        queue: list[tuple[int, int, float, tuple[int, ...]]] = [(0, 0, 0.0, (start,))]
        seen: set[int] = set()
        while queue:
            additions: int
            hops: int
            cost: float
            path: tuple[int, ...]
            additions, hops, cost, path = heapq.heappop(queue)
            current: int = path[-1]
            if current == end:
                return path
            if current in seen:
                continue
            seen.add(current)
            for candidate in sorted(self.graph[current] - seen):
                pair: IndexPair = pair_index(current, candidate)
                if pair in self.failures:
                    continue
                extra: int = additions + int(candidate not in picked)
                if extra <= MAX_BRIDGES_PER_JOIN:
                    confidence: float = (
                        self.overlaps[pair].held_out_supported
                        / self.overlaps[pair].held_out_count
                    )
                    heapq.heappush(
                        queue,
                        (extra, hops + 1, cost + 1 - confidence, (*path, candidate)),
                    )
        return ()

    def connect(self, start: int, end: int, picked: set[int]) -> tuple[int, ...]:
        """Find a route whose individual edges pass the paired-cutline checks."""
        for _ in range(MAX_ROUTES_PER_JOIN):
            path: tuple[int, ...] = self.paths(start, end, picked)
            if not path:
                break
            if all(
                self.check(pair_index(first, second)) is not None
                for first, second in zip(path, path[1:])
            ):
                return path
        return ()


def connected_sections(graph: dict[int, set[int]]) -> list[list[int]]:
    """Group reachable views while retaining isolated images for review."""
    unseen: set[int] = set(graph)
    sections: list[list[int]] = []
    while unseen:
        pending: list[int] = [min(unseen)]
        section: list[int] = []
        unseen.remove(pending[0])
        while pending:
            index: int = pending.pop()
            section.append(index)
            neighbors: set[int] = unseen & graph[index]
            unseen.difference_update(neighbors)
            pending.extend(sorted(neighbors))
        sections.append(sorted(section))
    return sections


def compact_recommendation(
    routes: Routes,
    covers: list[set[Tile]],
    rules: Rules,
    target: int,
    *,
    quality: list[dict[Tile, float]] | None = None,
) -> tuple[set[int], dict[int, str]]:
    """Select a small review set with explicit protections for distinct terrain.

    Args:
        routes: Verified photo overlaps and proposed handoffs.
        covers: Source tiles each candidate can cover at comparable detail.
        rules: Analyst pins and exclusions.
        target: Desired number of counting photos.
        quality: Optional tile weights for choosing among replacement views.

    Returns:
        Selected photo indices and reasons for keeping protected photos.
    """
    evidence: Counter[Tile] = Counter()
    for (first, second), overlap in routes.overlaps.items():
        if (
            first in rules.exclude
            or second in rules.exclude
            or 2 * overlap.held_out_supported < overlap.held_out_count
        ):
            continue
        for index, cells in ((first, overlap.first), (second, overlap.second)):
            evidence.update(
                (index, cell.cell)
                for cell in cells
                if cell.support_points >= 10 and cell.source.clipped <= 0.20
            )
    eligible: set[int] = set(routes.graph)
    owners: Counter[Tile] = Counter(
        tile for index in eligible for tile in covers[index]
    )
    strong: set[Tile] = {tile for tile in owners if evidence[tile] >= 2}
    distinct: set[Tile] = {tile for tile in strong if owners[tile] == 1}
    protected: dict[int, str] = {}
    for index, cell in sorted(distinct):
        right: bool = cell % GRID_COLS < GRID_COLS - 1 and (index, cell + 1) in distinct
        below: bool = (index, cell + GRID_COLS) in distinct
        if right or below:
            protected[index] = (
                "Keeps neighboring, well-supported cells with no comparable alternate."
            )
    picked: set[int] = set(rules.keep) | protected.keys()
    remaining: set[Tile] = set(owners) - {
        tile for index in picked for tile in covers[index]
    }

    def rank(index: int) -> tuple[float, int]:
        """Prefer sharp evidence, with extra weight for corroborated cells."""
        new: set[Tile] = covers[index] & remaining
        if quality is None:
            return -(len(new) + len(new & strong)), index
        score: float = sum(
            (1 + int(tile in strong)) * quality[index].get(tile, 1.0)
            for tile in sorted(new)
        )
        return -score, index

    for section in connected_sections(routes.graph):
        supported: list[int] = [index for index in section if covers[index]]
        if supported and not picked.intersection(section):
            index: int = min(supported, key=rank)
            protected[index] = "Represents a separate section with documented terrain."
            picked.add(index)
            remaining.difference_update(covers[index])
    while remaining and len(picked) < target:
        index = min(eligible - picked, key=rank)
        picked.add(index)
        remaining.difference_update(covers[index])
    return picked, protected


def plan_sections(
    routes: Routes,
    picked: set[int],
    report: RunReport,
    *,
    target: int | None = None,
    restore_routes: dict[IndexPair, list[int]] | None = None,
) -> tuple[list[dict[str, Any]], set[IndexPair], list[IndexPair], set[int]]:
    """Build suggested counting orders and explicitly record blocked handoffs."""
    bridges: set[int] = set()
    sections: list[dict[str, Any]] = []
    chosen: set[IndexPair] = set()
    unresolved: list[IndexPair] = []
    components: list[list[int]] = connected_sections(routes.graph)
    joins: int = sum(
        max(0, len(picked.intersection(component)) - 1) for component in components
    )
    completed: int = 0
    for component in components:
        targets: list[int] = sorted(picked.intersection(component))
        walks: list[list[int]] = [targets[:1]] if targets else []
        for first, second in zip(targets, targets[1:]):
            path: tuple[int, ...] = routes.connect(first, second, picked)
            new: set[int] = set(path) - picked
            over_target: bool = (
                target is not None and bool(new) and len(picked | new) > target
            )
            if not path or over_target:
                unresolved.append((first, second))
                walks.append([second])
                if over_target and restore_routes is not None:
                    restore_routes[first, second] = sorted(new)
            else:
                bridges.update(new)
                picked.update(new)
                walks[-1].extend(path[1:])
                chosen.update(pair_index(a, b) for a, b in zip(path, path[1:]))
            completed += 1
            report.update("counting joins checked", completed, joins)
        order: list[int] = list(
            dict.fromkeys(index for walk in walks for index in walk)
        )
        sections.append(
            {
                "id": len(sections) + 1,
                "candidates": component,
                "order": order,
                "handoff_walks": walks,
            }
        )
    return sections, chosen, unresolved, bridges


def plan_selected_handoffs(
    routes: Routes, order: list[int], report: RunReport
) -> tuple[list[dict[str, Any]], set[IndexPair], list[IndexPair], set[int]]:
    """Check consecutive counting photos without adding alignment views to the set."""
    chosen: set[IndexPair] = set()
    unresolved: list[IndexPair] = []
    walks: list[list[int]] = [order[:1]]
    for done, (first, second) in enumerate(zip(order, order[1:]), start=1):
        pair: IndexPair = pair_index(first, second)
        if routes.check(pair) is None:
            unresolved.append(pair)
            walks.append([second])
        else:
            chosen.add(pair)
            walks[-1].append(second)
        report.update("counting joins checked", done, len(order) - 1)
    section: dict[str, Any] = {
        "id": 1,
        "candidates": sorted(routes.graph),
        "order": order,
        "handoff_walks": walks,
    }
    return [section], chosen, unresolved, set()


def plan_refined_handoffs(
    routes: Routes,
    picked: set[int],
    prior: RefinementRoutes,
    report: RunReport,
) -> tuple[list[dict[str, Any]], set[IndexPair], list[IndexPair], dict[str, Any]]:
    """Refine reviewed joins before repairing changed routes with checked views.

    Args:
        routes: Match evidence and native-resolution route checks.
        picked: Photo indices checked by the analyst for refinement.
        prior: Reviewed handoffs, affected joins, and added photos.
        report: Refinement progress reporter.

    Returns:
        Counting sections, accepted joins, unresolved joins, and review details.
    """
    reviewed: set[IndexPair] = {pair_index(a, b) for a, b in prior["reviewed"]}
    preserved: set[IndexPair] = {
        pair for pair in sorted(reviewed) if routes.check(pair) is not None
    }
    chosen: set[IndexPair] = preserved.copy()
    unresolved: list[IndexPair] = []
    reconnect: set[IndexPair] = {pair_index(a, b) for a, b in prior["reconnect"]}
    # A reviewed join can be retried through other checked photos if its direct
    # route fails at full resolution.
    repair: list[IndexPair] = sorted((reconnect | reviewed) - chosen)
    for done, pair in enumerate(repair, start=1):
        path: tuple[int, ...] = routes.connect(*pair, picked)
        if path:
            chosen.update(pair_index(a, b) for a, b in zip(path, path[1:]))
        else:
            unresolved.append(pair)
        report.update("reviewed handoffs rechecked", done, len(repair))

    linked: set[int] = {index for pair in chosen for index in pair}
    unlinked: list[int] = []
    for index in prior["added"]:
        if index in linked:
            continue
        # Prefer a direct link into the reviewed routes. Lone photos stay selected
        # so they can still be counted or checked separately.
        neighbors: list[int] = sorted(
            routes.graph[index] & picked,
            key=lambda other: (
                other not in linked,
                -routes.overlaps[pair_index(index, other)].held_out_supported,
                other,
            ),
        )
        for other in neighbors:
            pair: IndexPair = pair_index(index, other)
            if routes.check(pair) is not None:
                chosen.add(pair)
                linked.update(pair)
                break
        else:
            unlinked.append(index)

    adjacency: dict[int, set[int]] = {index: set() for index in picked}
    for first, second in chosen:
        adjacency[first].add(second)
        adjacency[second].add(first)
    sections: list[dict[str, Any]] = []
    for component in connected_sections(adjacency):
        members: set[int] = set(component)
        walks: list[list[int]] = [
            list(path)
            for path in edge_paths({pair for pair in chosen if pair[0] in members})
        ] or [component]
        sections.append(
            {
                "id": len(sections) + 1,
                "candidates": component,
                "order": list(dict.fromkeys(index for walk in walks for index in walk)),
                "handoff_walks": walks,
            }
        )
    diagnostics: dict[str, Any] = {
        "reviewed_pairs": sorted(reviewed),
        "refined_reviewed_pairs": sorted(preserved),
        "replanned_pairs": sorted(chosen - reviewed),
        "unlinked_added_photos": unlinked,
    }
    return sections, chosen, unresolved, diagnostics


def compact_review(
    routes: Routes,
    covers: list[set[Tile]],
    picked: set[int],
    sections: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    unresolved: list[IndexPair],
    restore_routes: dict[IndexPair, list[int]],
) -> dict[str, Any]:
    """Explain compact omissions and show what a single restored view can recover."""
    omitted: set[int] = set(routes.graph) - picked
    missing: set[Tile] = {(gap["photo"], cell) for gap in gaps for cell in gap["cells"]}
    recoverable: dict[int, set[Tile]] = {i: covers[i] & missing for i in omitted}
    for gap in gaps:
        source: int = gap["photo"]
        local: set[Tile] = {(source, cell) for cell in gap["cells"]}
        alternates: list[int] = [i for i in omitted if recoverable[i] & local]
        if alternates:
            restore: int = min(
                alternates,
                key=lambda i: (-len(recoverable[i] & local), -len(recoverable[i]), i),
            )
            gap["restore_photo"] = restore
            gap["restored_cells"] = len(recoverable[restore] & local)

    unrepresented: list[dict[str, Any]] = []
    for section in sections:
        if not section["order"]:
            members: set[int] = set(section["candidates"])
            review_photo: int = min(
                members,
                key=lambda i: (-sum(tile[0] in members for tile in recoverable[i]), i),
            )
            unrepresented.append(
                {
                    "section": section["id"],
                    "photos": section["candidates"],
                    "review_photo": review_photo,
                    "documented_cells": sum(
                        source == i for i in members for source, _ in covers[i]
                    ),
                }
            )

    joins: list[dict[str, Any]] = []
    for first, second in unresolved:
        validated: list[int] = [
            index
            for index in restore_routes.get((first, second), [])
            if index in omitted
        ]
        candidates: set[int] = routes.graph[first] & routes.graph[second] & omitted
        suggestions: list[int] = validated
        if not suggestions and candidates:

            def rank(index: int) -> tuple[int, int, int]:
                """Prefer bridges with fewer failed joins and broader overlap support."""
                edges: tuple[IndexPair, IndexPair] = (
                    pair_index(first, index),
                    pair_index(index, second),
                )
                return (
                    sum(edge in routes.failures for edge in edges),
                    -min(routes.overlaps[edge].held_out_supported for edge in edges),
                    index,
                )

            suggestions = [min(candidates, key=rank)]
        joins.append(
            {
                "photos": [first, second],
                "restore_photos": suggestions,
                "validated": bool(validated),
            }
        )
    return {"unrepresented_sections": unrepresented, "handoff_restores": joins}


def view_owns(
    handoff: Handoff, index: int, pair: IndexPair, points: Coordinates
) -> NDArray[np.float64]:
    """Get signed keep distances in one view for a set of sampled landmarks."""
    if index == pair[0]:
        return -handoff.seam.offset(points)
    return handoff.seam.offset(handoff.seam.map_to_first(points))


def check_selected_triplets(
    routes: Routes, chosen: set[IndexPair], picked: set[int]
) -> list[dict[str, Any]]:
    """Look for alignment disagreements and duplicate or missing counting assignments."""
    nearby: dict[int, set[int]] = defaultdict(set)
    for first, second in chosen:
        nearby[first].add(second)
        nearby[second].add(first)
    triples: set[tuple[int, ...]] = {
        tuple(sorted((a, b, c)))
        for b in picked
        for a, c in combinations(sorted(nearby[b]), 2)
    }
    results: list[dict[str, Any]] = []
    for a, b, c in sorted(triples):
        ab: Handoff | None = routes.tested.get((a, b))
        bc: Handoff | None = routes.tested.get((b, c))
        ac: Handoff | None = routes.tested.get((a, c))
        found: dict[IndexPair, Handoff] = {
            pair: routes.tested[pair]
            for pair in ((a, b), (b, c), (a, c))
            if pair in chosen and routes.tested[pair] is not None
        }
        if len(found) < 2:
            continue
        # Evaluate actual matched samples in the common view of the two assigned
        # handoffs. A direct outer match helps reject artificial three-view overlap.
        center: int = next(
            index for index in (a, b, c) if sum(index in pair for pair in found) >= 2
        )
        others: list[int] = [index for index in (a, b, c) if index != center]
        direct: Overlap | None = routes.overlaps.get(pair_index(*others))
        if direct is None or not direct.first or not direct.second:
            if (
                ab
                and bc
                and center == b
                and review_triple(ab.seam, bc.seam, None) is not None
            ):
                results.append(
                    {
                        "photos": [a, b, c],
                        "needs_review": True,
                        "compared_samples": 0,
                        "reason": "The handoffs may share terrain. Direct outer-view geometry needs review.",
                    }
                )
            continue
        first: int
        last: int
        first, last = others
        outer_pair: IndexPair = (first, last)
        support: dict[int, Coordinates] = {
            index: working_points(routes.frames[index], points)
            for index, points in zip(outer_pair, direct.preview_points, strict=True)
        }
        projections: list[Coordinates] = []
        for index in others:
            pair: IndexPair = pair_index(index, center)
            seam: Seam = found[pair].seam
            projections.append(
                seam.map_to_second(support[index])
                if index == pair[0]
                else seam.map_to_first(support[index])
            )
        first_path: Coordinates
        second_path: Coordinates
        first_path, second_path = projections
        valid: NDArray[np.bool_] = inside_frame(
            routes.frames[center], first_path
        ) & inside_frame(routes.frames[center], second_path)
        drift: Coordinates = np.linalg.norm(first_path - second_path, axis=1)
        coincident: NDArray[np.bool_] = valid & (drift <= 6.0)
        support[center] = (first_path + second_path) / 2
        owner_count: NDArray[np.intp] = np.zeros(len(drift), dtype=np.intp)
        near: NDArray[np.bool_] = np.zeros(len(drift), dtype=bool)
        for index in (a, b, c):
            keeps: NDArray[np.bool_] = np.ones(len(drift), dtype=bool)
            for pair, handoff in found.items():
                if index in pair:
                    signs: Coordinates = view_owns(handoff, index, pair, support[index])
                    keeps &= signs > 0
                    near |= abs(signs) < 2
            owner_count += keeps
        samples: NDArray[np.bool_] = coincident & ~near
        duplicates: int = int((samples & (owner_count > 1)).sum())
        gaps: int = int((samples & (owner_count == 0)).sum())
        review: TripleReview | None = (
            review_triple(ab.seam, bc.seam, ac.seam) if ab and bc and ac else None
        )
        alignment: float | None = (
            float(np.quantile(drift[valid], 0.9)) if valid.any() else None
        )
        uncertain: bool = (
            alignment is None
            or alignment > 6.0
            or int(samples.sum()) < 4
            or bool(duplicates or gaps or (review and review.uncertain))
        )
        results.append(
            {
                "photos": [a, b, c],
                "needs_review": uncertain,
                "compared_samples": int(samples.sum()),
                "duplicate_samples": duplicates,
                "unassigned_samples": gaps,
                "alignment_p90_working_px": alignment,
            }
        )
    return results


def check_cross_links(
    routes: Routes, chosen: set[IndexPair], picked: set[int]
) -> list[dict[str, Any]]:
    """Check selected overlapping views that lack a direct counting handoff."""
    boundaries: dict[int, list[IndexPair]] = defaultdict(list)
    for pair in sorted(chosen):
        for index in pair:
            boundaries[index].append(pair)
    results: list[dict[str, Any]] = []
    for pair, overlap in sorted(routes.overlaps.items()):
        if (
            pair in chosen
            or not picked.issuperset(pair)
            or not overlap.first
            or not overlap.second
        ):
            continue
        owners: NDArray[np.intp] = np.zeros(
            len(overlap.preview_points[0]), dtype=np.intp
        )
        near: NDArray[np.bool_] = np.zeros(len(owners), dtype=bool)
        for index, native in zip(pair, overlap.preview_points, strict=True):
            points: Coordinates = working_points(routes.frames[index], native)
            keeps: NDArray[np.bool_] = np.ones(len(owners), dtype=bool)
            for boundary in boundaries[index]:
                handoff: Handoff | None = routes.tested[boundary]
                assert handoff is not None
                signs: Coordinates = view_owns(handoff, index, boundary, points)
                keeps &= signs > 0
                near |= ~np.isfinite(signs) | (abs(signs) < 2)
            owners += keeps
        duplicate: int = int(((owners > 1) & ~near).sum())
        compared: int = int((~near).sum())
        if duplicate or compared < 4 or not all(boundaries[index] for index in pair):
            results.append(
                {
                    "photos": list(pair),
                    "compared_samples": compared,
                    "duplicate_samples": duplicate,
                }
            )
    return results


def suggest_sfm(
    routes: Routes,
    counts: dict[IndexPair, int],
    unresolved: list[IndexPair],
    triples: list[dict[str, Any]],
    cross_links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank small local reconstruction candidates around specific review problems."""
    issues: list[tuple[int, tuple[int, ...], str]] = [
        (0, pair, "No validated counting handoff connects these views.")
        for pair in unresolved
    ]
    issues.extend(
        (1, tuple(item["photos"]), "Check three-view counting ownership.")
        for item in triples
        if item["needs_review"]
    )
    issues.extend(
        (
            1,
            tuple(item["photos"]),
            "Check overlapping counting views without a direct handoff.",
        )
        for item in cross_links
    )
    issues.extend(
        (2, pair, reason)
        for pair, reason in routes.failures.items()
        if pair not in unresolved
    )
    components: dict[int, int] = {
        index: section
        for section, members in enumerate(connected_sections(routes.graph))
        for index in members
    }
    boundary: list[IndexPair] = [
        pair
        for pair, count in counts.items()
        if count >= 24
        and all(index in components for index in pair)
        and components[pair[0]] != components[pair[1]]
    ]
    boundary.sort(key=lambda pair: (-counts[pair], pair))
    issues.extend(
        (
            0,
            pair,
            "Verified features link separate sections. Shared terrain is uncertain.",
        )
        for pair in boundary[:8]
    )
    issues.sort(
        key=lambda issue: (
            issue[0],
            -sum(counts.get(pair, 0) for pair in combinations(issue[1], 2)),
            issue[1],
        )
    )
    jobs: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for priority, focus, reason in issues:
        group: set[int] = set(focus)
        while len(group) < 6:
            candidates: list[tuple[int, int, int]] = []
            for index in routes.graph.keys() - group:
                support: list[int] = [
                    counts.get(pair_index(index, member), 0)
                    for member in group & routes.graph[index]
                ]
                if len(support) >= 2:
                    candidates.append((len(support), sum(support), -index))
            if not candidates:
                break
            group.add(-max(candidates)[2])
        photos: tuple[int, ...] = tuple(sorted(group))
        if photos in seen:
            continue
        seen.add(photos)
        reliable_links: int = sum(
            b in routes.graph[a] for a, b in combinations(photos, 2)
        )
        eligible: bool = len(photos) >= 3 and reliable_links >= len(photos) - 1
        jobs.append(
            {
                "focus": list(focus),
                "photos": list(photos),
                "reason": reason,
                "recommended_action": (
                    "local SfM candidate" if eligible else "manual overlap review"
                ),
                "priority": priority,
                "supported_links": reliable_links,
            }
        )
        if len(jobs) == 12:
            break
    return jobs


def describe_photos(
    photos: list[Photo],
    rules: Rules,
    routes: Routes,
    documented: list[set[int]],
    covers: list[set[Tile]],
    picked: set[int],
    bridges: set[int],
) -> list[dict[str, Any]]:
    """Explain each counting decision and every source area still needing inspection."""
    replacements: dict[int, list[dict[str, Any]]] = defaultdict(list)
    softer: dict[int, set[int]] = defaultdict(set)
    for (first, second), overlap in routes.overlaps.items():
        for source, alternative, evidence in (
            (first, second, overlap.first),
            (second, first, overlap.second),
        ):
            eligible: list[dict[str, Any]] = [
                asdict(cell) for cell in evidence if cell.eligible
            ]
            if alternative not in rules.exclude:
                softer[source].update(
                    cell.cell
                    for cell in evidence
                    if cell.eligible
                    and cell.relative_sharpness is not None
                    and cell.relative_sharpness >= SHARPER_ALTERNATE_RATIO
                )
            if eligible and alternative in picked:
                replacements[source].append({"photo": alternative, "cells": eligible})
    rows: list[dict[str, Any]] = []
    for index, photo in enumerate(photos):
        alternatives: list[dict[str, Any]] = sorted(
            replacements[index],
            key=lambda entry: (-len(entry["cells"]), entry["photo"]),
        )
        supported_elsewhere: set[int] = {
            cell["cell"] for entry in alternatives for cell in entry["cells"]
        }
        unknown: list[int] = sorted(set(range(CELL_COUNT)) - documented[index])
        unique: list[int] = sorted(documented[index] - supported_elsewhere)
        if index in rules.exclude:
            state: str
            reason: str
            state, reason = "analyst excluded", rules.reasons.get(
                index, "Excluded in the analyst's selection rules."
            )
        elif index in picked:
            state = "included"
            if index in rules.keep:
                reason = "Pinned by the analyst."
            elif index in bridges:
                reason = "Provides a validated handoff between other proposed counting views."
            elif unique:
                reason = "Retains documented terrain without a comparable view in the suggested set."
            else:
                reason = "Covers documented terrain in other views at comparable apparent detail."
        elif unknown or unique:
            state, reason = (
                "coverage review",
                "Check whether this photo adds countable terrain before omitting it.",
            )
        else:
            state, reason = (
                "candidate omission",
                "Selected views directly support all documented grid cells.",
            )
        rows.append(
            {
                "index": index,
                "id": photo.staged_name,
                "name": photo.path.name,
                "sha256": photo.digest,
                "width": photo.width,
                "height": photo.height,
                "state": state,
                "reason": reason,
                "documented_cells": sorted(documented[index]),
                "unassessed_cells": unknown,
                "cells_without_selected_replacement": unique,
                "documented_cells_supported_in_view": len(covers[index]),
                "supported_neighbors": sorted(routes.graph.get(index, set())),
                "selected_replacements": alternatives,
                "cells_with_sharper_alternative": sorted(softer[index]),
            }
        )
    return rows


def describe_pairs(
    counts: dict[IndexPair, int],
    shortlisted: list[IndexPair],
    routes: Routes,
    chosen: set[IndexPair],
    rows: list[dict[str, Any]],
    cross_links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the full match graph and include previews for decision-relevant pairs."""
    cross_pairs: set[IndexPair] = {pair_index(*item["photos"]) for item in cross_links}
    previews: set[IndexPair] = chosen | set(routes.failures) | cross_pairs
    for row in rows:
        if row["state"] != "included" and row["selected_replacements"]:
            previews.add(
                pair_index(row["index"], row["selected_replacements"][0]["photo"])
            )
    examined: set[IndexPair] = set(shortlisted)
    evidence: list[dict[str, Any]] = []
    for pair in sorted(counts.keys() | previews):
        count: int = counts.get(pair, 0)
        overlap: Overlap | None = routes.overlaps.get(pair)
        entry: dict[str, Any] = {
            "photos": list(pair),
            "inliers": count,
            "shortlisted": pair in examined,
        }
        if overlap is not None:
            entry.update(
                {
                    "held_out_supported": overlap.held_out_supported,
                    "held_out_total": overlap.held_out_count,
                    "median_error_working_px": overlap.median_error,
                    "documented_cells_by_photo": [
                        [cell.cell for cell in cells]
                        for cells in (overlap.first, overlap.second)
                    ],
                }
            )
            if pair in previews:
                entry["matched_landmarks_native"] = [
                    np.round(points, 3).tolist() for points in overlap.preview_points
                ]
        if pair in routes.failures:
            entry["handoff_error"] = routes.failures[pair]
        if pair in cross_pairs:
            entry["review_count_assignment"] = True
        handoff: Handoff | None = routes.tested.get(pair) if pair in chosen else None
        if handoff is not None:
            entry["handoff"] = describe_handoff(pair, handoff)
            if routes.seed_frames is not None:
                entry["handoff"]["native_search"] = (
                    "seed corridor" if pair in routes.refined_in_corridor else "overlap"
                )
        if pair in routes.seed_drafts:
            entry["working_handoff"] = routes.seed_drafts[pair]
        evidence.append(entry)
    return evidence


def describe_handoff(pair: IndexPair, handoff: Handoff) -> dict[str, Any]:
    """Store both draft lines and keep sides in native source pixels."""
    return {
        "status": "draft",
        "method": handoff.seam.method,
        "endpoint_mode": handoff.endpoint_mode,
        **({"group_rerouted": True} if handoff.group_rerouted else {}),
        "warnings": list(handoff.seam.warnings),
        "midpoint_fallback": handoff.seam.midpoint_fallback,
        "views": [
            {
                "photo": index,
                "line": np.round(line, 4).tolist(),
                "keep_point": np.round(side.point, 4).tolist(),
                "keep_side": side.side,
                "keep_direction": side.direction,
            }
            for index, line, side in zip(
                pair, handoff.seam.points_in_source_images(), handoff.sides, strict=True
            )
        ],
    }


def build_selection_plan(
    database: Path,
    photos: list[Photo],
    frames: list[Frame],
    verified: list[VerifiedPair],
    rules: Rules,
    report: RunReport,
    *,
    counting_order: list[int] | None = None,
    refinement: bool = False,
    target_photo_percent: int | None = None,
    refinement_routes: RefinementRoutes | None = None,
) -> dict[str, Any]:
    """Build the reviewable recommendation without changing matches or annotations."""
    cv2.setRNGSeed(0)
    name_index: dict[str, int] = {photo.path.name: i for i, photo in enumerate(photos)}
    with report.phase("selection_rank_pairs"):
        counts: dict[IndexPair, int] = {
            pair_index(*(name_index[name] for name in pair["images"])): pair["inliers"]
            for pair in verified
        }
        screened: list[IndexPair] = shortlist(counts, len(photos))
        if refinement:
            screened = sorted(counts)
        elif counting_order is not None:
            screened = sorted(
                set(screened)
                | {
                    pair_index(a, b)
                    for a, b in zip(counting_order, counting_order[1:])
                    if pair_index(a, b) in counts
                }
            )
        wanted: list[IndexPair] = (
            sorted(counts) if target_photo_percent is not None else screened
        )
        matches: dict[IndexPair, PointPair] = indexed_matches(database, photos, wanted)
    with report.phase("selection_check_coverage"):
        details: list[list[Detail]] = [tile_detail(frame) for frame in frames]
        focus: list[FocusMap] | None = (
            [patch_focus_map(frame) for frame in frames]
            if counting_order is None
            else None
        )
        overlaps: dict[IndexPair, Overlap] = {}

        def assess(pairs: list[IndexPair], label: str) -> None:
            """Apply the same held-out geometry and quality gates to all pairs."""
            for done, pair in enumerate(pairs, start=1):
                if pair in matches:
                    a: int
                    b: int
                    a, b = pair
                    overlap: Overlap | None = analyze_overlap(
                        (frames[a], frames[b]),
                        matches[pair],
                        (details[a], details[b]),
                        (focus[a], focus[b]) if focus is not None else None,
                    )
                    if overlap is not None:
                        overlaps[pair] = overlap
                report.update(label, done, len(pairs))

        assess(screened, "overlaps checked")
        documented: list[set[int]]
        covers: list[set[Tile]]
        documented, covers = coverage_sets(overlaps, len(photos))
        quality: list[dict[Tile, float]] | None = (
            sharpness_weights(overlaps, covers, rules) if focus is not None else None
        )
        target: int | None = None
        before: set[int] = (
            set(counting_order)
            if counting_order is not None
            else recommend(covers, rules, quality=quality)
        )
        picked: set[int] = before.copy()
        extra: list[IndexPair] = []
        baseline_on_expanded: set[int] = before
        original_cells: int = sum(map(len, documented))
        original_pair_count: int = len(overlaps)
        if target_photo_percent is not None:
            target = max(2, len(photos) * target_photo_percent // 100)
            extra = expanded_pair_order(counts, screened, covers, before, rules)
            assess(extra, "additional verified overlaps checked")
            documented, covers = coverage_sets(overlaps, len(photos))
            quality = sharpness_weights(overlaps, covers, rules)
            baseline_on_expanded = recommend(covers, rules, quality=quality)
    restore_routes: dict[IndexPair, list[int]] = {}
    protected: dict[int, str] = {}
    refined_routes: dict[str, Any] | None = None
    with report.phase("selection_handoffs"):
        routes: Routes = Routes(
            frames,
            matches,
            overlaps,
            rules.exclude,
            seed_edge=REFINEMENT_SEED_EDGE if refinement else None,
        )
        if target is not None:
            picked, protected = compact_recommendation(
                routes, covers, rules, target, quality=quality
            )
        coverage_pick_count: int = len(picked)
        if refinement_routes is not None:
            sections: list[dict[str, Any]]
            chosen: set[IndexPair]
            unresolved: list[IndexPair]
            sections, chosen, unresolved, refined_routes = plan_refined_handoffs(
                routes, picked, refinement_routes, report
            )
            bridges: set[int] = set()
        elif counting_order is None:
            sections, chosen, unresolved, bridges = plan_sections(
                routes,
                picked,
                report,
                target=target,
                restore_routes=restore_routes,
            )
        else:
            sections, chosen, unresolved, bridges = plan_selected_handoffs(
                routes, counting_order, report
            )
    with report.phase("selection_cross_checks"):
        triples: list[dict[str, Any]] = check_selected_triplets(routes, chosen, picked)
        cross_links: list[dict[str, Any]] = check_cross_links(routes, chosen, picked)
        compared: set[Tile] = {
            (index, tile) for index, tiles in enumerate(documented) for tile in tiles
        }
        supported: set[Tile] = {tile for index in picked for tile in covers[index]}
        missing: set[Tile] = compared - supported
        rows: list[dict[str, Any]] = describe_photos(
            photos, rules, routes, documented, covers, picked, bridges
        )
        for index, reason in protected.items():
            if index not in rules.keep:
                rows[index]["reason"] = reason
        pairs: list[dict[str, Any]] = describe_pairs(
            counts, screened, routes, chosen, rows, cross_links
        )
        if target is not None:
            assessed: set[IndexPair] = set(screened + extra)
            for row in pairs:
                row["assessed"] = tuple(row["photos"]) in assessed
        jobs: list[dict[str, Any]] = suggest_sfm(
            routes, counts, unresolved, triples, cross_links
        )
        gaps: list[dict[str, Any]] = [
            {
                "photo": index,
                "cells": sorted(tile for source, tile in missing if source == index),
            }
            for index in sorted({source for source, _ in missing})
        ]
        compact: dict[str, Any] | None = None
        if target is not None:
            compact = compact_review(
                routes, covers, picked, sections, gaps, unresolved, restore_routes
            )
            compact.update(
                target_percent=target_photo_percent,
                target_photos=target,
                shortlist_pairs=len(screened),
                additional_pairs=len(extra),
                additional_supported_pairs=len(overlaps) - original_pair_count,
                shortlist_documented_cells=original_cells,
                shortlist_coverage_photos=len(before),
                expanded_coverage_photos=len(baseline_on_expanded),
                protected_photos=sorted(protected.keys() | rules.keep),
                selected_before_bridges=coverage_pick_count,
                bridge_photos=sorted(bridges),
            )
    plan: dict[str, Any] = {
        "schema_version": 1,
        "approval_status": "analyst review required",
        "photo_index": "Zero-based index into photos. Pixel coordinates use the stored orientation.",
        "coverage_grid": {
            "columns": GRID_COLS,
            "rows": GRID_ROWS,
            "index": "zero-based, row-major",
        },
        "summary": {
            "candidate_photos": len(photos),
            "suggested_photos": len(picked),
            "sections": len(sections),
            "coverage_review_photos": sum(
                row["state"] == "coverage review" for row in rows
            ),
            "documented_cells": len(compared),
            "covered_documented_cells": len(compared & supported),
            "unresolved_handoffs": len(unresolved),
            "draft_handoffs": len(chosen),
            "triplets_needing_review": sum(item["needs_review"] for item in triples),
            "cross_links_needing_review": len(cross_links),
            "shortlisted_pairs": len(screened),
            "pairs_with_local_geometry": len(overlaps),
            "local_sfm_candidates": sum(
                job["recommended_action"] == "local SfM candidate" for job in jobs
            ),
        },
        "photos": rows,
        "sections": sections,
        "counting_set": sorted(picked) if counting_order is None else counting_order,
        "counting_set_source": "suggested" if counting_order is None else "analyst",
        "coverage_gaps": gaps,
        "unresolved_handoffs": [list(pair) for pair in unresolved],
        "overlap_pairs": pairs,
        "triplet_checks": triples,
        "cross_link_checks": cross_links,
        "suggested_local_sfm": jobs,
        "rules": {
            "keep": sorted(rules.keep),
            "exclude": sorted(rules.exclude),
            "exclusion_reasons": rules.reasons,
        },
    }
    if compact is not None:
        plan["compact_selection"] = compact
        plan["summary"]["additional_assessed_pairs"] = len(extra)
    if focus is not None:
        plan["summary"]["photos_with_sharper_alternatives"] = sum(
            bool(row["cells_with_sharper_alternative"]) for row in rows
        )
    if refined_routes is not None:
        plan["refinement_routes"] = refined_routes
        plan["summary"].update(
            reviewed_handoffs=len(refined_routes["reviewed_pairs"]),
            refined_reviewed_handoffs=len(refined_routes["refined_reviewed_pairs"]),
            replanned_handoffs=len(refined_routes["replanned_pairs"]),
            unlinked_added_photos=len(refined_routes["unlinked_added_photos"]),
        )
    return plan

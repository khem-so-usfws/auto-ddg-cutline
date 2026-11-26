"""Check photo-selection coverage, alternate handoffs, and combined count sides."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from auto_ddg_cutline.benchmark import RunReport
from auto_ddg_cutline.colmap import VerifiedPair
from auto_ddg_cutline.colmap_model import Photo
from auto_ddg_cutline.geometry import Coordinates, Frame, Pixels, Seam
from auto_ddg_cutline.paired import CountSide, Handoff, IndexPair, PointPair
from auto_ddg_cutline.selection import (
    Routes,
    Rules,
    build_selection_plan,
    check_cross_links,
    check_selected_triplets,
    compact_recommendation,
    compact_review,
    coverage_sets,
    describe_photos,
    plan_refined_handoffs,
    plan_sections,
    read_rules,
    recommend,
    sharpness_weights,
    shortlist,
    suggest_sfm,
)
from auto_ddg_cutline.selection_evidence import (
    CELL_COUNT,
    MIN_REPLACEMENT_SHARPNESS,
    Detail,
    FocusMap,
    Overlap,
    Replacement,
    Tile,
    analyze_overlap,
    patch_focus_map,
    relative_patch_sharpness,
    tile_detail,
)


def views(count: int) -> list[Frame]:
    """Return distinct views with a common pixel grid for synthetic geometry."""
    return [
        Frame(
            Path(f"view {index}.jpg"),
            320,
            240,
            np.full((240, 320), 128, np.uint8),
            np.empty((0, 2), np.float64),
            None,
        )
        for index in range(count)
    ]


def observation(cell: int, eligible: bool = True) -> Replacement:
    """Build one local evidence record, optionally with a softer replacement."""
    return Replacement(
        cell,
        15,
        (cell,),
        1.0,
        Detail(10, 0),
        Detail(10 if eligible else 3, 0),
        eligible,
    )


def overlap(first: tuple[Replacement, ...], second: tuple[Replacement, ...]) -> Overlap:
    """Place sampled landmarks across the picture for ownership comparisons."""
    points: Coordinates = np.array(
        [(x, y) for x in (60, 130, 170, 250) for y in (30, 80, 180)], dtype=np.float64
    )
    return Overlap(first, second, 12, 12, 0.2, (points, points))


def handoff(first: Frame, second: Frame, x: int) -> Handoff:
    """Put a left-versus-right boundary at a source pixel column."""
    seam: Seam = Seam(
        first,
        second,
        np.eye(3),
        np.eye(3),
        first.footprint,
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([[x, 5.0], [x, 230.0]]),
        100,
        1.0,
        5,
        None,
        (),
        "terrain",
    )
    return Handoff(
        seam,
        (
            CountSide(np.array([30.0, 100.0]), "west", "left"),
            CountSide(np.array([290.0, 100.0]), "east", "right"),
        ),
        "movable",
    )


def test_coverage_and_rules_preserve_the_only_countable_view(tmp_path: Path) -> None:
    """Reduce duplicate views and expose what an exclusion or uncertain view removes."""
    frames: list[Frame] = views(5)
    photos: list[Photo] = [
        Photo(frame.path, frame.width, frame.height, f"{index}.png", "digest")
        for index, frame in enumerate(frames)
    ]
    full: tuple[Replacement, ...] = tuple(
        observation(cell) for cell in range(CELL_COUNT)
    )
    overlaps: dict[IndexPair, Overlap] = {
        (0, 1): overlap(full, full),
        (2, 3): overlap((observation(1, False),), (observation(1),)),
    }
    documented: list[set[int]]
    covers: list[set[Tile]]
    documented, covers = coverage_sets(overlaps, len(frames))
    picked: set[int] = recommend(covers, Rules())
    assert picked == {0, 2}
    routes: Routes = Routes(frames, {}, overlaps, frozenset())
    rows: list[dict[str, Any]] = describe_photos(
        photos, Rules(), routes, documented, covers, picked, set()
    )
    assert rows[1]["state"] == "candidate omission"
    assert rows[1]["selected_replacements"][0]["photo"] == 0
    assert rows[3]["state"] == "coverage review"
    assert rows[4]["unassessed_cells"] == list(range(CELL_COUNT))
    path: Path = tmp_path / "analyst choices.json"
    path.write_text(
        json.dumps(
            {
                "keep": ["view 1.jpg"],
                "exclude": ["view 2.jpg"],
                "exclusion_reasons": {"view 2.jpg": "Obscured"},
            }
        ),
        encoding="utf-8",
    )
    source_paths: list[Path] = [photo.path for photo in photos]
    rules: Rules = read_rules(path, source_paths)
    revised: set[int] = recommend(covers, rules)
    assert revised == {1, 3}
    assert (2, 1) not in set.union(*(covers[index] for index in revised))
    path.write_text('{"keep": ["missing.jpg"]}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown photo"):
        read_rules(path, source_paths)


def test_failed_direct_line_can_be_replaced_by_a_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a useful out-of-order bridge and preserve the failed-line evidence."""
    frames: list[Frame] = views(4)
    supported: Overlap = overlap((observation(0),), (observation(0),))
    overlaps: dict[IndexPair, Overlap] = {
        (0, 1): supported,
        (0, 3): supported,
        (1, 3): supported,
    }

    def plan(a: Frame, b: Frame, *args: Any) -> tuple[Handoff | None, str]:
        """Reject the strongest direct handoff while allowing a two-edge bridge."""
        return (
            (None, "clipping failed")
            if (a, b) == (frames[0], frames[1])
            else (handoff(a, b, 150), "")
        )

    monkeypatch.setattr("auto_ddg_cutline.selection.plan_pair", plan)
    routes: Routes = Routes(frames, {}, overlaps, frozenset())
    picked: set[int] = {0, 1}
    sections: list[dict[str, Any]]
    chosen: set[IndexPair]
    unresolved: list[IndexPair]
    bridges: set[int]
    sections, chosen, unresolved, bridges = plan_sections(
        routes, picked, Mock(spec=RunReport)
    )
    assert not unresolved
    assert chosen == {(0, 3), (1, 3)}
    assert bridges == {3}
    assert picked == {0, 1, 3}
    assert sections[0]["order"] == [0, 3, 1]
    assert sections[1]["candidates"] == [2]
    jobs: list[dict[str, Any]] = suggest_sfm(
        routes, {(0, 1): 4500, (0, 3): 340, (1, 3): 725}, [], [], []
    )
    assert jobs[0]["photos"] == [0, 1, 3]
    assert jobs[0]["recommended_action"] == "local SfM candidate"
    assert jobs[0]["reason"] == "clipping failed"


def test_ownership_checks_use_all_selected_views() -> None:
    """Detect outer views claiming the same terrain through two draft handoffs."""
    frames: list[Frame] = views(3)
    supported: Overlap = overlap((observation(0),), (observation(0),))
    edges: dict[IndexPair, Overlap] = {
        (0, 1): supported,
        (0, 2): supported,
        (1, 2): supported,
    }
    routes: Routes = Routes(frames, {}, edges, frozenset())
    routes.tested.update(
        {
            (0, 1): handoff(frames[0], frames[1], 200),
            (1, 2): handoff(frames[1], frames[2], 100),
        }
    )
    chosen: set[IndexPair] = {(0, 1), (1, 2)}
    picked: set[int] = {0, 1, 2}
    triplets: list[dict[str, Any]] = check_selected_triplets(routes, chosen, picked)
    assert triplets[0]["needs_review"]
    assert triplets[0]["duplicate_samples"] == 6
    cross_links: list[dict[str, Any]] = check_cross_links(routes, chosen, picked)
    assert cross_links[0]["photos"] == [0, 2]
    assert cross_links[0]["duplicate_samples"] == 6
    routes.tested.update(
        {
            (0, 1): handoff(frames[0], frames[1], 100),
            (1, 2): handoff(frames[1], frames[2], 200),
        }
    )
    assert not check_selected_triplets(routes, chosen, picked)[0]["needs_review"]
    assert not check_cross_links(routes, chosen, picked)


def test_shortlist_uses_strong_nonadjacent_pairs_with_stable_ties() -> None:
    """Preserve strong cross-pass matches without evaluating every weak pair."""
    counts: dict[IndexPair, int] = {(0, 1): 10, (0, 6): 201, (0, 7): 16, (7, 8): 20}
    expected: list[IndexPair] = [(0, 1), (0, 6), (7, 8)]
    assert shortlist(counts, 9) == expected
    assert shortlist(dict(reversed(list(counts.items()))), 9) == expected


def test_compact_search_uses_remaining_verified_pairs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Check extended replacement evidence and give a saved gap a restore view."""
    frames: list[Frame] = views(8)
    photos: list[Photo] = [
        Photo(frame.path, frame.width, frame.height, f"{index}.png", "digest")
        for index, frame in enumerate(frames)
    ]
    overlaps: dict[IndexPair, Overlap] = {
        (0, 1): overlap((observation(1, False),), (observation(1),)),
        (2, 3): overlap((observation(2, False),), (observation(2),)),
        (4, 5): overlap((observation(3),), (observation(3),)),
        (0, 2): overlap((observation(1),), (observation(2, False),)),
        (1, 3): overlap((observation(4, False),), (observation(4, False),)),
    }
    verified: list[VerifiedPair] = [
        {"images": [photos[i].path.name for i in pair], "inliers": 100}
        for pair in overlaps
    ]
    pairs_checked: list[IndexPair] = []
    by_path: dict[Path, int] = {frame.path: i for i, frame in enumerate(frames)}

    def assess(
        items: tuple[Frame, Frame],
        match: PointPair,
        detail: tuple[Detail, Detail],
        _focus: tuple[FocusMap, FocusMap] | None,
    ) -> Overlap:
        """Return known matches and track the order of the geometry checks."""
        pair: IndexPair = (by_path[items[0].path], by_path[items[1].path])
        pairs_checked.append(pair)
        return overlaps[pair]

    monkeypatch.setattr(
        "auto_ddg_cutline.selection.shortlist",
        lambda _counts, _size: [(0, 1), (2, 3), (4, 5)],
    )
    monkeypatch.setattr(
        "auto_ddg_cutline.selection.indexed_matches",
        lambda _db, _photos, pairs: {
            key: overlaps[key].preview_points for key in pairs
        },
    )
    monkeypatch.setattr("auto_ddg_cutline.selection.analyze_overlap", assess)
    report: RunReport = RunReport(tmp_path / "run.json", {}, False)
    plan: dict[str, Any] = build_selection_plan(
        tmp_path / "matches.db",
        photos,
        frames,
        verified,
        Rules(),
        report,
        target_photo_percent=25,
    )
    compact: dict[str, Any] = plan["compact_selection"]
    assert pairs_checked == [(0, 1), (2, 3), (4, 5), (0, 2), (1, 3)]
    assert compact["additional_pairs"] == compact["additional_supported_pairs"] == 2
    assert compact["shortlist_coverage_photos"] == 3
    assert compact["target_photos"] == 2
    assert len(plan["counting_set"]) == 2
    assert set(plan["counting_set"]).intersection({4, 5})
    assert {
        section["review_photo"] for section in compact["unrepresented_sections"]
    } == {6, 7}
    assert plan["coverage_gaps"]
    for gap in plan["coverage_gaps"]:
        assert gap["restore_photo"] not in plan["counting_set"]
        assert gap["restored_cells"] >= 1
    assert next(pair for pair in plan["overlap_pairs"] if pair["photos"] == [0, 2])[
        "assessed"
    ]


def test_compact_can_exceed_target_for_distinct_terrain() -> None:
    """Protect one coherent region and keep another documented section represented."""
    frames: list[Frame] = views(6)
    unique: tuple[Replacement, ...] = (observation(0, False), observation(1, False))
    common: tuple[Replacement, ...] = (observation(0), observation(1))
    overlaps: dict[IndexPair, Overlap] = {
        (0, 1): overlap(unique, common),
        (0, 2): overlap(unique, common),
        (3, 4): overlap(common, common),
    }
    covers: list[set[Tile]]
    _, covers = coverage_sets(overlaps, len(frames))
    rules: Rules = Rules(keep=frozenset({5}))
    routes: Routes = Routes(frames, {}, overlaps, rules.exclude)
    selected: set[int]
    protected: dict[int, str]
    selected, protected = compact_recommendation(routes, covers, rules, 2)
    assert selected == {0, 3, 5}
    assert set(protected) == {0, 3}


def test_compact_reports_validated_bridge_beyond_target(tmp_path: Path) -> None:
    """Show the photo that would heal a join before recommending it for counting."""
    frames: list[Frame] = views(3)
    common: tuple[Replacement, ...] = (observation(0),)
    overlaps: dict[IndexPair, Overlap] = {
        (0, 1): overlap(common, common),
        (1, 2): overlap(common, common),
    }
    routes: Routes = Routes(frames, {}, overlaps, frozenset())
    routes.tested = {
        pair: handoff(frames[pair[0]], frames[pair[1]], 110) for pair in overlaps
    }
    restore: dict[IndexPair, list[int]] = {}
    selected: set[int] = {0, 2}
    covers: list[set[Tile]]
    _, covers = coverage_sets(overlaps, len(frames))
    report: RunReport = RunReport(tmp_path / "run.json", {}, False)
    sections: list[dict[str, Any]]
    chosen: set[IndexPair]
    unresolved: list[IndexPair]
    bridges: set[int]
    sections, chosen, unresolved, bridges = plan_sections(
        routes,
        selected,
        report,
        target=2,
        restore_routes=restore,
    )
    result: dict[str, Any] = compact_review(
        routes, covers, selected, sections, [], unresolved, restore
    )
    assert chosen == bridges == set()
    assert selected == {0, 2}
    assert unresolved == [(0, 2)]
    assert result["handoff_restores"] == [
        {"photos": [0, 2], "restore_photos": [1], "validated": True},
    ]


def test_refinement_keeps_reviewed_edges_and_repairs_changed_joins(
    tmp_path: Path,
) -> None:
    """Keep nonconsecutive draft pairs and flag a new photo without a cutline."""
    frames: list[Frame] = views(6)
    common: tuple[Replacement, ...] = (observation(0),)
    overlaps: dict[IndexPair, Overlap] = {
        pair: overlap(common, common) for pair in ((0, 2), (0, 1), (1, 3))
    }
    routes: Routes = Routes(frames, {}, overlaps, frozenset())
    routes.tested = {
        pair: handoff(frames[pair[0]], frames[pair[1]], 120) for pair in overlaps
    }
    routes.tested[2, 4] = None
    report: RunReport = RunReport(tmp_path / "run.json", {}, False)
    sections: list[dict[str, Any]]
    chosen: set[IndexPair]
    unresolved: list[IndexPair]
    details: dict[str, Any]
    sections, chosen, unresolved, details = plan_refined_handoffs(
        routes,
        set(range(6)),
        {"reviewed": [(0, 2), (2, 4)], "reconnect": [(0, 3)], "added": [1, 5]},
        report,
    )
    assert chosen == {(0, 2), (0, 1), (1, 3)}
    assert unresolved == [(2, 4)]
    assert details == {
        "reviewed_pairs": [(0, 2), (2, 4)],
        "refined_reviewed_pairs": [(0, 2)],
        "replanned_pairs": [(0, 1), (1, 3)],
        "unlinked_added_photos": [5],
    }
    assert [section["candidates"] for section in sections] == [[0, 1, 2, 3], [4], [5]]


def test_local_focus_distinguishes_smear_and_contrast() -> None:
    """Compare matched detail independently of simple contrast changes."""
    random: np.random.Generator = np.random.default_rng(45)
    base: Pixels = random.integers(35, 221, (120, 160), dtype=np.uint8)
    sharp: Pixels = cv2.resize(base, (320, 240), interpolation=cv2.INTER_NEAREST)
    smeared: Pixels = cv2.blur(sharp, (9, 1))
    contrast: Pixels = np.rint(128 + (sharp.astype(np.float32) - 128) * 0.55).astype(
        np.uint8
    )
    flat: Pixels = np.full_like(sharp, 128)
    frames: list[Frame] = [
        replace(frame, gray=image)
        for frame, image in zip(views(4), (sharp, smeared, contrast, flat), strict=True)
    ]
    focus: list[FocusMap] = [patch_focus_map(frame) for frame in frames]
    points: Coordinates = np.array(
        [(x, y) for x in range(15, 270, 8) for y in range(15, 200, 8)], dtype=np.float64
    )
    smear_ratio: float | None = relative_patch_sharpness(
        (frames[0], frames[1]), (points, points), (focus[0], focus[1])
    )
    assert smear_ratio is not None and smear_ratio < MIN_REPLACEMENT_SHARPNESS
    contrast_ratio: float | None = relative_patch_sharpness(
        (frames[0], frames[2]), (points, points), (focus[0], focus[2])
    )
    assert contrast_ratio == pytest.approx(1, abs=0.03)
    assert (
        relative_patch_sharpness(
            (frames[0], frames[3]), (points, points), (focus[0], focus[3])
        )
        is None
    )
    assert (
        relative_patch_sharpness(
            (frames[0], frames[1]), (points, points * 0.5), (focus[0], focus[1])
        )
        is None
    )

    assessed: Overlap | None = analyze_overlap(
        (frames[0], frames[1]),
        (points, points),
        (tile_detail(frames[0]), tile_detail(frames[1])),
        (focus[0], focus[1]),
    )
    assert assessed is not None
    assert sum(cell.eligible for cell in assessed.second) > sum(
        cell.eligible for cell in assessed.first
    )


def test_recommenders_prefer_sharper_coverage_and_keep_required_views() -> None:
    """Break a coverage tie with matched focus and protect the sole view of a cell."""
    frames: list[Frame] = views(2)
    sharper: Replacement = replace(observation(0), relative_sharpness=1.40)
    softer: Replacement = replace(observation(0), relative_sharpness=0.71)
    overlaps: dict[IndexPair, Overlap] = {(0, 1): overlap((sharper,), (softer,))}
    rules: Rules = Rules()
    routes: Routes = Routes(frames, {}, overlaps, rules.exclude)
    covers: list[set[Tile]]
    _, covers = coverage_sets(overlaps, len(frames))
    quality: list[dict[tuple[int, int], float]] = sharpness_weights(
        overlaps, covers, rules
    )
    assert recommend(covers, rules) == {0}
    assert recommend(covers, rules, quality=quality) == {1}
    assert compact_recommendation(routes, covers, rules, 1, quality=quality)[0] == {1}

    covers[0].add((0, 1))
    assert recommend(covers, rules, quality=quality) == {0}
    assert compact_recommendation(
        routes, covers, Rules(keep=frozenset({0})), 1, quality=quality
    )[0] == {0}

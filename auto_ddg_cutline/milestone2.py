"""Orchestrate the COLMAP draft ownership workflow and its run manifest."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, TypeAlias

import cv2
import numpy as np

from .benchmark import ResourceSampler, RunReport
from .colmap import (
    DEFAULT_CPU_THREADS,
    Colmap,
    ColmapError,
    ComputeDevice,
    Device,
    Flags,
    MatchingCounts,
    VerifiedPair,
    extract_and_match,
    extraction_batches,
    find_colmap,
    inspect_flags,
    map_matches,
    matching_counts,
    ordered_matches,
    reconstruct,
    run_on_device,
    stage_photos,
    verified_pairs,
)
from .colmap_model import Model, Photo, model_diagnostics
from .coverage import Coverage, build_coverage
from .geometry import (
    DRAFT_COLOR_EDGE_WEIGHT,
    MIN_CUTLINE_MATCHES,
    Coordinates,
    Frame,
    Seam,
)
from .local_sfm import run_local_sfm
from .match_cache import MatchAssets, copy_database, load_matches
from .paired import (
    CountSide,
    Handoff,
    IndexPair,
    PhotoPair,
    PointPair,
    plan_handoffs,
    review_handoffs,
    stage_frame,
)
from .selection import REFINEMENT_SEED_EDGE, Rules, build_selection_plan, read_rules
from .selection_report import write_selection_report
from .shared_patches import run_shared_patch_pilot
from .sidecars import SidecarEdit, pair_flag, plan_count_areas, plan_edits, write_edits

ColmapOutput: TypeAlias = Literal[
    "cutlines", "sparse-regions", "sfm-diagnostics", "selection-plan"
]


def add_initial_cutlines(plan: dict[str, Any], selection: dict[str, Any]) -> None:
    """Attach unchanged initial geometry for pairs present in the refined set."""
    if not selection.get("selection_plan") or not selection.get(
        "selection_plan_sha256"
    ):
        return
    try:
        contents: bytes = Path(selection["selection_plan"]).read_bytes()
    except OSError:
        return  # The new run is still reviewable if the initial job was archived.
    if hashlib.sha256(contents).hexdigest() != selection["selection_plan_sha256"]:
        return
    initial: dict[str, Any] = json.loads(contents)
    if "export_pairs" not in initial:
        return
    initial_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for (first, second), handoff in review_handoffs(initial).items():
        names: tuple[str, str] = (
            initial["photos"][first]["name"],
            initial["photos"][second]["name"],
        )
        initial_pairs[names] = handoff
    for entry in plan["overlap_pairs"]:
        if "handoff" not in entry and "handoff_error" not in entry:
            continue
        a: int
        b: int
        a, b = entry["photos"]
        names = plan["photos"][a]["name"], plan["photos"][b]["name"]
        prior: dict[str, Any] | None = initial_pairs.get(names)
        views: list[dict[str, Any]]
        if prior is not None:
            views = prior["views"]
        elif names[::-1] in initial_pairs:
            views = initial_pairs[names[::-1]]["views"][::-1]
        else:
            continue
        entry["initial_handoff"] = {
            "views": [
                dict(view, photo=index)
                for index, view in zip((a, b), views, strict=True)
            ]
        }


def describe_coverage(
    model: Model, coverage: Coverage, photos: dict[str, Photo]
) -> list[dict[str, Any]]:
    """Report draft ownership separately from unregistered and unassigned views."""
    views: dict[Path, int] = {
        view.photo.path: view_id for view_id, view in model.views.items()
    }
    diagnostics: list[dict[str, Any]] = []
    for photo in photos.values():
        view_id: int | None = views.get(photo.path)
        polygons: list[Coordinates] = (
            [] if view_id is None else coverage.polygons.get(view_id, [])
        )
        count: int = len(polygons)
        area: float = float(
            sum(cv2.contourArea(polygon.astype(np.float32)) for polygon in polygons)
        )
        status: str = "PROPOSED" if count else "NO_ASSIGNED_AREA"
        if view_id is None:
            status = "UNRESOLVED"
        print(f"  {status} {photo.path.name}: {count} polygon(s), {area:,.0f} px²")
        entry: dict[str, Any] = {
            "image": photo.path.name,
            "status": status,
            "polygons": count,
            "owned_pixels_approx": area,
            "owned_faces": coverage.owner_counts[view_id] if view_id is not None else 0,
        }
        if view_id in coverage.reprojection:
            samples: int
            p90: float
            samples, p90 = coverage.reprojection[view_id]
            entry["reprojection_samples"] = samples
            entry["reprojection_p90_px"] = p90
        diagnostics.append(entry)
    return diagnostics


def export_sidecars(
    photos: dict[str, Photo],
    edits: list[SidecarEdit],
    write: bool,
    manifest: dict[str, Any],
) -> None:
    """Record proposed and committed edits using the shared sidecar transaction."""
    manifest["planned_sidecars"] = [edit.path.name for edit in edits]
    for edit in edits:
        print(f"  {'WRITE' if write else 'PLAN'} {edit.path.name}")
    if write and edits:
        for photo in photos.values():
            if hashlib.sha256(photo.path.read_bytes()).hexdigest() != photo.digest:
                raise OSError(f"Source image changed before export: {photo.path.name}")
        write_edits(next(iter(photos.values())).path.parent, edits)
        manifest["written_sidecars"] = [edit.path.name for edit in edits]
    elif edits:
        print("  Analysis only. Add --write to append draft annotations.")


def match_cutline_pairs(
    engine: Colmap,
    flags: Flags,
    job: Path,
    device: Device,
    photos: dict[str, Photo],
    cpu_threads: int,
    report: RunReport,
    *,
    require_handoffs: bool = True,
) -> tuple[dict[IndexPair, PointPair], MatchAssets]:
    """Retry an empty GPU result on CPU and keep evidence from each attempt."""
    attempts: list[dict[str, Any]] = []
    manifest: dict[str, Any] = report.manifest
    manifest["matching_attempts"] = attempts
    ordered: list[Photo] = list(photos.values())

    def attempt(
        selected_device: ComputeDevice,
    ) -> tuple[dict[IndexPair, PointPair], MatchAssets]:
        """Run matching in a device-specific database and summarize its results."""
        start: float = perf_counter()
        first_command: int = len(engine.commands)
        record: dict[str, Any] = {"device": selected_device, "status": "running"}
        attempts.append(record)
        manifest["device_used"] = selected_device
        try:
            database: Path = extract_and_match(
                engine, flags, job, selected_device, photos, cpu_threads
            )
            with report.phase("match_validation", selected_device):
                candidates: dict[IndexPair, PointPair] = (
                    ordered_matches(database, ordered) if require_handoffs else {}
                )
                counts: MatchingCounts = matching_counts(database, photos)
                verified: list[VerifiedPair] = verified_pairs(database, photos)
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            raise
        except Exception as error:
            record.update(status="failed", error=str(error))
            raise
        finally:
            record["elapsed_seconds"] = round(perf_counter() - start, 3)
        usable: int = sum(
            len(points) >= MIN_CUTLINE_MATCHES for points, _ in candidates.values()
        )
        arguments: dict[str, list[str]] = {
            argv[1]: argv[2:] for argv in engine.commands[first_command:]
        }
        extract: list[str] = arguments.get("feature_extractor", [])
        batches: dict[int, list[str]] = extraction_batches(engine, photos)
        assets: MatchAssets = MatchAssets(
            database.resolve(),
            job / "images",
            selected_device,
            {
                "colmap_version": manifest.get("colmap_version"),
                "cpu_extraction_threads": (
                    cpu_threads if selected_device == "cpu" else None
                ),
                "effective_feature_max_edge": max(batches),
                "feature_pixel_percent": engine.feature_pixel_percent,
                "extraction_batches": [
                    {"max_long_edge": edge, "staged_images": names}
                    for edge, names in batches.items()
                ],
                "feature_extractor_arguments": extract,
                "exhaustive_matcher_arguments": arguments.get("exhaustive_matcher", []),
            },
        )
        record.update(
            status="completed",
            database=str(database),
            verified_pairs=verified,
            usable_cutline_pairs=usable,
            **counts,
        )
        extracted: int = sum(count > 0 for count in counts["descriptors"].values())
        print(
            f"COLMAP {selected_device}: descriptors in {extracted}/{len(photos)} photos, "
            f"{counts['raw_matched_pairs']} raw matched pairs, {len(verified)} verified pairs, "
            f"{usable} usable handoff pairs.",
            flush=True,
        )
        return candidates, assets

    def remember_fallback(reason: str) -> None:
        """Checkpoint the fallback even if the following CPU attempt fails."""
        manifest["gpu_fallback"] = reason
        report.save()

    result: tuple[dict[IndexPair, PointPair], MatchAssets]
    used: ComputeDevice
    fallback: str | None
    result, used, fallback = run_on_device(device, attempt, remember_fallback)
    manifest["gpu_fallback"] = fallback
    usable: bool = (
        bool(attempts[-1]["usable_cutline_pairs"])
        if require_handoffs
        else any(
            pair["inliers"] >= MIN_CUTLINE_MATCHES
            for pair in attempts[-1]["verified_pairs"]
        )
    )
    if device == "auto" and used == "gpu" and not usable:
        message: str = (
            "GPU matching yielded zero usable verified pairs for this mode. Retrying on CPU."
        )
        remember_fallback(message)
        print(message, flush=True)
        result = attempt("cpu")
    matches: dict[IndexPair, PointPair]
    assets: MatchAssets
    matches, assets = result
    manifest["verified_pairs"] = attempts[-1]["verified_pairs"]
    manifest["feature_extraction"] = assets.settings
    if any(
        pair["inliers"] >= MIN_CUTLINE_MATCHES for pair in manifest["verified_pairs"]
    ):
        manifest["match_checkpoint"] = assets.to_manifest()
    report.save()
    return matches, assets


def export_paired_cutlines(
    matches: dict[IndexPair, PointPair],
    photos: dict[str, Photo],
    image_folder: Path,
    write: bool,
    report: RunReport,
) -> str:
    """Plan paired handoffs and optionally append them to DDG sidecars.

    Args:
        matches: Verified native-pixel matches keyed by photo index pair.
        photos: Selected photos in survey order, keyed by staged filename.
        image_folder: Directory containing the losslessly staged images.
        write: Whether to write the planned sidecar changes.
        report: Current run's progress reporter and manifest.

    Returns:
        The manifest status, either `"complete"` or `"partial"`.
    """
    ordered: list[Photo] = list(photos.values())
    manifest: dict[str, Any] = report.manifest
    quality: dict[str, Any] = manifest["benchmark_quality"]
    with report.phase("prepare_cutline_frames"):
        frames: list[Frame] = []
        for number, photo in enumerate(ordered, start=1):
            frames.append(stage_frame(photo, image_folder))
            report.update("cutline frames read", number, len(ordered))
    handoffs: list[Handoff]
    drafts: set[PhotoPair]
    notes: list[str]
    handoffs, drafts, notes = plan_handoffs(frames, matches, report, quality)
    manifest["handoff_notes"] = notes
    manifest["cutline_pairs"] = []
    quality["paired_cutlines"] = len(handoffs)
    quality["draft_handoffs"] = len(drafts)
    quality["proposed_handoffs"] = len(handoffs) - len(drafts)
    quality["unresolved_adjacent_by_reason"] = dict(
        Counter(
            item["reason"]
            for item in quality["pair_outcomes"]
            if item["scope"] == "adjacent" and item["status"] == "unresolved"
        )
    )
    for note in notes:
        print(f"  {note}")
    if not handoffs:
        raise ValueError(
            "No continuous paired cutlines met the overlap and side checks."
        )

    adjacent: set[PhotoPair] = set(
        zip(
            (photo.path for photo in ordered[:-1]),
            (photo.path for photo in ordered[1:]),
            strict=True,
        )
    )
    proposed: set[PhotoPair] = {handoff.pair for handoff in handoffs}
    for handoff in handoffs:
        pair: PhotoPair = handoff.pair
        seam: Seam = handoff.seam
        vertex_count: int = len(seam.points)
        needs_review: bool = pair in drafts
        a: Path
        b: Path
        a, b = pair
        first: CountSide
        second: CountSide
        first, second = handoff.sides
        status: str = "DRAFT" if needs_review else "PROPOSED"
        print(
            f"  {status} {a.name} / {b.name} ({seam.method}, "
            f"{handoff.endpoint_mode} endpoints, {vertex_count} vertices): "
            f"keep {first.direction} in the first, {second.direction} in the second."
        )
        manifest["cutline_pairs"].append(
            {
                "pair_id": pair_flag(seam),
                "needs_review": needs_review,
                "seam_method": seam.method,
                "vertex_count": vertex_count,
                "endpoint_mode": handoff.endpoint_mode,
                "midpoint_fallback": seam.midpoint_fallback,
                "images": [
                    {
                        "name": path.name,
                        "keep_direction": side.direction,
                        "keep_side": side.side,
                        "keep_point": np.round(side.point, 4).tolist(),
                    }
                    for path, side in zip(pair, handoff.sides, strict=True)
                ],
            }
        )
    with report.phase("annotation_export"):
        edits: list[SidecarEdit]
        messages: list[str]
        edits, messages = plan_edits(
            [handoff.seam for handoff in handoffs],
            drafts,
            {handoff.pair: handoff.sides for handoff in handoffs},
        )
        manifest["sidecar_notes"] = messages
        for message in messages:
            print(message)
        export_sidecars(photos, edits, write, manifest)
    return "partial" if adjacent - proposed or drafts else "complete"


def run_colmap(
    selected: list[Path],
    executable: str | None,
    device: Device,
    work_dir: Path | None,
    write: bool,
    cpu_threads: int = DEFAULT_CPU_THREADS,
    output: ColmapOutput = "cutlines",
    *,
    verbose: bool = False,
    reuse_matches: Path | None = None,
    profile_resources: bool = False,
    feature_max_edge: int | None = None,
    feature_pixel_percent: float | None = None,
    selection_rules: Path | None = None,
    local_sfm: bool = False,
    counting_order: list[int] | None = None,
    refinement_selection: Path | None = None,
    compare_shared_patches: bool = False,
    target_photo_percent: int | None = None,
) -> int:
    """Analyze selected native photos and optionally append editable drafts.

    Args:
        selected: Photos explicitly chosen by the analyst.
        executable: Optional path to the local COLMAP executable or launcher.
        device: SIFT extraction and matching device policy.
        work_dir: Optional parent directory for fresh job workspaces.
        write: Whether to append the planned sidecars.
        cpu_threads: Maximum concurrent CPU SIFT extraction workers.
        output: Paired cutlines, experimental sparse regions, SfM diagnostics, or selection.
        verbose: Stream detailed subprocess output to the terminal.
        reuse_matches: Optional manifest identifying completed matching evidence.
        profile_resources: Sample RAM and device-wide GPU memory when available.
        feature_max_edge: Optional internal COLMAP extraction cap in pixels.
        feature_pixel_percent: Percentage of each image's native pixels for extraction.
        selection_rules: Optional JSON file with pinned or excluded photo names.
        local_sfm: Run actionable local reconstruction jobs and draft their cutlines.
        counting_order: Optional indices for an analyst-selected counting set.
        refinement_selection: Saved input selection for a higher-resolution review.
        compare_shared_patches: Test full-folder alignment and selection on one block.
        target_photo_percent: Soft compact-selection target as a percentage of input photos.

    Returns:
        Zero for a supported result, including a reported partial result.
    """
    if cpu_threads < 1:
        raise ValueError("--cpu-threads must be a positive integer.")
    if feature_max_edge is not None and feature_max_edge < 1:
        raise ValueError("--feature-max-edge must be a positive integer.")
    if feature_pixel_percent is not None and not 0 < feature_pixel_percent <= 100:
        raise ValueError(
            "Extraction pixel percentage must be greater than 0 and at most 100."
        )
    if feature_pixel_percent is not None and feature_max_edge is not None:
        raise ValueError("Specify only one COLMAP extraction size option.")
    if output == "sfm-diagnostics" and (write or len(selected) < 3):
        raise ValueError(
            "SfM diagnostics require at least three photos and analysis without --write."
        )
    if local_sfm and output != "selection-plan":
        raise ValueError("--run-local-sfm requires --output selection-plan.")
    if compare_shared_patches and output != "selection-plan":
        raise ValueError("--compare-shared-patches requires --output selection-plan.")
    if output == "selection-plan" and write and not local_sfm:
        raise ValueError(
            "Selection-plan --write requires --run-local-sfm to create draft cutlines."
        )
    if selection_rules is not None and output != "selection-plan":
        raise ValueError("--selection-rules requires --output selection-plan.")
    if target_photo_percent is not None and (
        not 1 <= target_photo_percent <= 100
        or output != "selection-plan"
        or counting_order is not None
        or refinement_selection is not None
    ):
        raise ValueError(
            "Use a counting photo target from 1 to 100 with automatic selection."
        )
    rules: Rules = read_rules(selection_rules, selected)
    if counting_order is not None and (
        output != "selection-plan"
        or len(counting_order) < 2
        or len(counting_order) != len(set(counting_order))
        or any(not 0 <= index < len(selected) for index in counting_order)
        or not rules.keep.issubset(counting_order)
        or not rules.exclude.isdisjoint(counting_order)
    ):
        raise ValueError(
            "The counting selection is invalid or conflicts with the selection rules."
        )
    if reuse_matches is not None and (
        output == "sparse-regions"
        or feature_max_edge is not None
        or feature_pixel_percent is not None
        or device != "auto"
        or cpu_threads != DEFAULT_CPU_THREADS
    ):
        raise ValueError(
            "Reuse requires cutlines, a selection plan, or SfM diagnostics and the saved matching settings."
        )
    if work_dir is not None:
        work_dir.mkdir(exist_ok=True, parents=True)
    job: Path = Path(
        tempfile.mkdtemp(prefix="auto-ddg-colmap-", dir=work_dir)
    ).resolve()
    manifest: dict[str, Any] = {
        "benchmark_schema_version": 1,
        "run_mode": "reused" if reuse_matches is not None else "fresh",
        "backend": "colmap",
        "device_requested": device,
        "gpu_fallback": None,
        "commands": [],
        "command_timings": [],
        "verbose": verbose,
        "resource_profiling_requested": profile_resources,
        "cpu_extraction_threads": cpu_threads,
        "feature_max_edge_requested": feature_max_edge,
        "feature_pixel_percent_requested": feature_pixel_percent,
        "output": output,
        "local_sfm_requested": local_sfm,
        "images": [{"name": photo.name} for photo in selected],
        "write_requested": write,
        "written_sidecars": [],
        "status": "running",
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
        },
        "benchmark_quality": {
            "selected_photos": len(selected),
            "exhaustive_candidate_pairs": len(selected) * (len(selected) - 1) // 2,
        },
    }
    selection_working_edge: int = 1800
    if target_photo_percent is not None:
        manifest["target_photo_percent"] = target_photo_percent
    selection_snapshot: dict[str, Any] = {}
    if refinement_selection is not None:
        manifest["refinement_selection"] = str(refinement_selection)
        manifest["selection_working_coordinates"] = "native"
        manifest["refinement_seed_edge"] = REFINEMENT_SEED_EDGE
    engine: Colmap | None = None
    flags: Flags | None = None
    assets: MatchAssets | None = None
    image_folder: Path = job / "images"
    report: RunReport = RunReport(job / "run.json", manifest, verbose)
    sampler: ResourceSampler = ResourceSampler(profile_resources)
    print(f"COLMAP run manifest: {report.path}", flush=True)
    try:
        report.save()
        if (
            reuse_matches is None
            or output == "sfm-diagnostics"
            or local_sfm
            or compare_shared_patches
        ):
            with report.phase("setup"):
                engine = find_colmap(executable)
                engine.report = report
                engine.feature_max_edge = feature_max_edge
                engine.feature_pixel_percent = feature_pixel_percent
                manifest["commands"] = engine.commands
                manifest["command_timings"] = engine.command_timings
                version: str
                version, flags = inspect_flags(engine, job)
                manifest["colmap_version"] = version
        matches: dict[IndexPair, PointPair] = {}
        if reuse_matches is not None:
            with report.phase("reuse_validation"):
                photos: dict[str, Photo]
                assets, photos = load_matches(reuse_matches, selected, report)
                image_folder = assets.image_folder
                manifest["match_reuse"] = {
                    "source_manifest": str(reuse_matches.resolve())
                }
                manifest["device_used"] = assets.device
                manifest["cpu_extraction_threads"] = assets.settings.get(
                    "cpu_extraction_threads"
                )
                manifest.setdefault(
                    "colmap_version", assets.settings.get("colmap_version")
                )
                manifest["feature_extraction"] = assets.settings
                manifest["images"] = photo_manifest(photos)
                manifest["verified_pairs"] = verified_pairs(assets.database, photos)
                manifest["match_checkpoint"] = assets.to_manifest()
                if output == "cutlines":
                    matches = ordered_matches(assets.database, list(photos.values()))
                for stage in (
                    "staging",
                    "feature_extraction",
                    "matching_and_verification",
                ):
                    report.skip(stage, "reused completed matches")
        else:
            with report.phase("staging"):
                photos = stage_photos(selected, image_folder, report)
                manifest["images"] = photo_manifest(photos)
                if refinement_selection is not None:
                    selection_snapshot = json.loads(
                        refinement_selection.read_text(encoding="utf-8")
                    )
                    if [
                        (photo.path.name, photo.digest) for photo in photos.values()
                    ] != [
                        (row["name"], row["sha256"])
                        for row in selection_snapshot["photos"]
                    ]:
                        raise ValueError(
                            "A checked photo changed while preparing refinement."
                        )
            if output != "sparse-regions":
                assert engine is not None and flags is not None
                matches, assets = match_cutline_pairs(
                    engine,
                    flags,
                    job,
                    device,
                    photos,
                    cpu_threads,
                    report,
                    require_handoffs=output == "cutlines",
                )
        if output != "sparse-regions":
            manifest["benchmark_quality"]["verified_overlap_pairs"] = len(
                manifest["verified_pairs"]
            )
        if output == "cutlines":
            status: str = export_paired_cutlines(
                matches, photos, image_folder, write, report
            )
            manifest["status"] = status
            print(
                f"{status.upper()}: review the paired cutlines and keep sides in DDG."
            )
            return 0
        if output == "selection-plan":
            assert assets is not None
            ordered: list[Photo] = list(photos.values())
            protected_groups: dict[str, list[list[list[float]]]] = (
                selection_snapshot.get("protected_groups", {})
            )
            with report.phase("prepare_selection_frames"):
                frames: list[Frame] = []
                for done, photo in enumerate(ordered, start=1):
                    edge: int = (
                        max(photo.width, photo.height)
                        if refinement_selection is not None
                        else selection_working_edge
                    )
                    frame: Frame = stage_frame(
                        photo,
                        image_folder,
                        edge,
                        draft_color=refinement_selection is None,
                    )
                    polygons: tuple[Coordinates, ...] = tuple(
                        np.asarray(points, dtype=np.float64)
                        for points in protected_groups.get(photo.path.name, [])
                    )
                    frames.append(
                        replace(frame, protected_polygons=polygons)
                        if polygons
                        else frame
                    )
                    report.update("selection frames read", done, len(ordered))
            plan: dict[str, Any] = build_selection_plan(
                assets.database,
                ordered,
                frames,
                manifest["verified_pairs"],
                rules,
                report,
                counting_order=counting_order,
                refinement=refinement_selection is not None,
                target_photo_percent=target_photo_percent,
                refinement_routes=selection_snapshot.get("refinement_routes"),
            )
            drafts: list[Handoff] = []
            if local_sfm:
                assert engine is not None and flags is not None
                drafts = run_local_sfm(
                    plan,
                    engine,
                    flags,
                    assets,
                    ordered,
                    frames,
                    job,
                    report,
                    unresolved_only=refinement_selection is not None,
                )
                plan["summary"].update(
                    local_sfm_blocks_run=len(plan["local_sfm"]["blocks"]),
                    local_sfm_draft_handoffs=len(drafts),
                )
                print(
                    f"Local SfM: {len(drafts)} draft paired cutlines. Review the keep sides in the selection report.",
                    flush=True,
                )
            # Prefer the primary preview. Offer a local SfM draft for a selected
            # pair only when the primary planner did not produce a line.
            approved: dict[tuple[int, ...], dict[str, Any]] = {
                tuple(pair["photos"]): pair
                for pair in plan["overlap_pairs"]
                if "handoff" in pair
            }
            unresolved: set[tuple[int, ...]] = {
                tuple(pair) for pair in plan["unresolved_handoffs"]
            }
            for pair in plan.get("local_sfm", {}).get("drafts", []):
                if tuple(pair["photos"]) in unresolved:
                    approved.setdefault(tuple(pair["photos"]), pair)
            plan["export_pairs"] = [list(pair) for pair in approved]
            if refinement_selection is not None:
                plan["refinement"] = True
                add_initial_cutlines(plan, selection_snapshot)
            else:
                plan["draft_color_edge_weight"] = DRAFT_COLOR_EDGE_WEIGHT
                manifest["draft_color_edge_weight"] = DRAFT_COLOR_EDGE_WEIGHT
            if protected_groups:
                plan["protected_groups"] = protected_groups
                plan["summary"].update(
                    protected_group_polygons=sum(map(len, protected_groups.values())),
                    group_rerouted_handoffs=sum(
                        pair["handoff"].get("group_rerouted", False)
                        for pair in approved.values()
                    ),
                )
            if compare_shared_patches:
                assert engine is not None and flags is not None
                print(
                    "Comparing photo selection using whole-folder sparse alignment.",
                    flush=True,
                )
                pilot: dict[str, Any] = run_shared_patch_pilot(
                    plan, engine, flags, assets, photos, frames, rules, job, report
                )
                plan["shared_patch_pilot"] = pilot
                manifest["shared_patch_pilot"] = {
                    key: value
                    for key, value in pilot.items()
                    if key not in {"photo_local", "shared_patch"}
                }
                print(
                    f'Shared-patch comparison: {pilot["status"]}. '
                    f'{pilot.get("reason", "See the selection report.")}',
                    flush=True,
                )
            with report.phase("selection_report"):
                write_selection_report(job, plan, frames)
            summary: dict[str, int] = plan["summary"]
            manifest["selection_plan"] = {
                "report": str(job / "selection-plan.html"),
                "data": str(job / "selection-plan.json"),
                "rules": str(selection_rules.resolve()) if selection_rules else None,
                "approval_status": plan["approval_status"],
            }
            manifest["benchmark_quality"].update(summary)
            if target_photo_percent is not None:
                compact: dict[str, Any] = plan["compact_selection"]
                manifest["compact_selection"] = compact
                print(
                    f'Compact recommendation: {summary["suggested_photos"]} photos '
                    f'for a target of {compact["target_photos"]}. '
                    f'Checked {compact["additional_pairs"]} additional verified overlaps.',
                    flush=True,
                )
            if drafts and write:
                with report.phase("local_sfm_annotation_export"):
                    sides: dict[PhotoPair, tuple[CountSide, CountSide]] = {
                        draft.pair: draft.sides for draft in drafts
                    }
                    edits: list[SidecarEdit]
                    messages: list[str]
                    edits, messages = plan_edits(
                        [draft.seam for draft in drafts], set(sides), sides
                    )
                    manifest["sidecar_notes"] = messages
                    for message in messages:
                        print(message)
                    export_sidecars(photos, edits, write, manifest)
            manifest["status"] = (
                "partial"
                if (
                    plan["coverage_gaps"]
                    or summary["coverage_review_photos"]
                    or summary["unresolved_handoffs"]
                    or summary["triplets_needing_review"]
                    or summary["cross_links_needing_review"]
                    or plan.get("compact_selection", {}).get("unrepresented_sections")
                    or summary.get("unlinked_added_photos")
                )
                else "complete"
            )
            print(
                f"Selection: {summary['suggested_photos']}/{summary['candidate_photos']} suggested counting photos, "
                f"{summary['coverage_review_photos']} pending omission review, "
                f"{summary['draft_handoffs']} draft and {summary['unresolved_handoffs']} unresolved handoffs.",
                flush=True,
            )
            print(
                f"Open the selection report to review colony coverage: {job / 'selection-plan.html'}"
            )
            return 0
        assert engine is not None and flags is not None
        if output == "sfm-diagnostics":
            assert assets is not None
            if not manifest["verified_pairs"]:
                raise ValueError(
                    "COLMAP found no verified matches for sparse reconstruction."
                )
            database: Path = assets.database
            if reuse_matches is not None:
                with report.phase("database_copy"):
                    database = job / "reused" / "database.db"
                    copy_database(assets.database, database)
            models: list[Model] = map_matches(
                engine, flags, database, image_folder, photos
            )
            with report.phase("sfm_diagnostics"):
                summaries: list[dict[str, Any]] = [
                    model_diagnostics(model, photos) for model in models
                ]
                for summary in summaries:
                    summary["binary_model"] = str(
                        database.parent
                        / "sparse"
                        / summary["name"].removeprefix("model-")
                    )
                    summary["text_model"] = str(database.parent / summary["name"])
                    print(
                        f"  {summary['name']}: {len(summary['images'])}/{len(photos)} registered photos, {summary['tracks']} tracks, {len(summary['near_coincident_cameras'])} near-coincident camera pairs.",
                        flush=True,
                    )
                manifest["models"] = summaries
                registered: set[str] = {
                    name for model in summaries for name in model["images"]
                }
                manifest["benchmark_quality"]["sfm_models"] = len(summaries)
                manifest["benchmark_quality"]["sfm_registered_photos"] = len(registered)
                if not summaries:
                    raise ValueError(
                        "COLMAP did not reconstruct a sparse model. See mapper.log in the job folder."
                    )
                manifest["status"] = (
                    "complete"
                    if len(summaries) == 1
                    and not summaries[0]["unregistered_images"]
                    and not summaries[0]["near_coincident_cameras"]
                    else "partial"
                )
            return 0
        pairs: list[VerifiedPair]
        used: str
        fallback: str | None
        models, pairs, used, fallback = reconstruct(
            engine, flags, job, device, photos, cpu_threads
        )
        manifest["device_used"] = used
        manifest["gpu_fallback"] = fallback
        manifest["verified_pairs"] = pairs
        manifest["benchmark_quality"]["verified_overlap_pairs"] = len(pairs)
        manifest["models"] = [
            {
                "name": model.name,
                "images": [view.photo.path.name for view in model.views.values()],
                "tracks": len(model.tracks),
            }
            for model in models
        ]
        for entry in manifest["models"]:
            print(
                f"  {entry['name']}: {len(entry['images'])} registered image(s), "
                f"{entry['tracks']} reconstructed track(s)"
            )
        for pair in pairs:
            print(
                f"  Verified: {' / '.join(pair['images'])} ({pair['inliers']} inliers)"
            )
        if not models:
            raise ValueError(
                "COLMAP did not reconstruct a sparse model. No count areas were produced."
            )
        ordered: list[Model] = sorted(
            models, key=lambda item: (-len(item.views), -len(item.tracks), item.name)
        )
        # Export one connected block. Other models can depict the same rock
        # from an incompatible coordinate frame, even with distinct photos.
        model: Model = ordered[0]
        with report.phase("coverage_planning"):
            coverage: Coverage = build_coverage(model)
        manifest["export_model"] = model.name
        manifest["rejected_faces"] = dict(coverage.rejected)
        manifest["shared_cutlines"] = len(coverage.cutlines)
        manifest["shared_tracks"] = [
            {
                "images": [
                    model.views[first].photo.path.name,
                    model.views[second].photo.path.name,
                ],
                "tracks": number,
            }
            for (first, second), number in sorted(coverage.shared_tracks.items())
        ]
        manifest["ownership"] = describe_coverage(model, coverage, photos)
        if not coverage.polygons:
            raise ValueError("Sparse geometry produced no supported draft count areas.")
        status = (
            "partial"
            if len(models) > 1 or len(model.views) < len(photos) or coverage.rejected
            else "complete"
        )
        print(
            f"  {sum(coverage.owner_counts.values())} exported surface patches, "
            f"{sum(coverage.rejected.values())} rejected, "
            f"{len(coverage.cutlines)} paired cutline(s)."
        )
        if len(models) > 1:
            print("  Other reconstructed models need independent review.")
        print("  Terrain outside supported patches requires review in DDG.")

        with report.phase("annotation_export"):
            saved: list[str]
            edits, saved = plan_count_areas(model, coverage, photos)
            manifest["already_saved"] = saved
            for name in saved:
                print(f"  KEEP {name}: existing ownership proposal or DDG edits")
            export_sidecars(photos, edits, write, manifest)
        manifest["status"] = status
        print(
            f"{status.upper()}: open the source photos in DDG to review draft count areas."
        )
        return 0
    except KeyboardInterrupt:
        manifest.update(status="interrupted", error="Run interrupted by the user.")
        return 130
    except (ValueError, OSError, cv2.error, sqlite3.Error) as error:
        manifest["status"] = "failure"
        manifest["error"] = str(error)
        if isinstance(error, ColmapError):
            manifest["failure"] = {
                "subcommand": error.command,
                "returncode": error.returncode,
                "exit_status": error.exit_status,
                "log_path": str(error.log),
            }
        raise ValueError(f"{error}\nRun manifest: {job / 'run.json'}") from error
    finally:
        resources: dict[str, Any] = sampler.close()
        resources["staged_image_bytes"] = (
            directory_bytes(image_folder) if image_folder.is_dir() else None
        )
        resources["workspace_bytes_before_final_manifest"] = directory_bytes(job)
        database_paths: set[Path] = set(job.glob("*/database.db"))
        if assets is not None:
            database_paths.add(assets.database)
        resources["database_bytes_by_path"] = {
            str(path): path.stat().st_size
            for path in sorted(database_paths)
            if path.is_file()
        }
        manifest["resources"] = resources
        if manifest["status"] == "running":
            manifest["status"] = "failure"
        report.finish()


def photo_manifest(photos: dict[str, Photo]) -> list[dict[str, Any]]:
    """Keep the mapping required to validate and reuse native-coordinate matches."""
    return [
        {
            "name": photo.path.name,
            "staged_name": photo.staged_name,
            "width": photo.width,
            "height": photo.height,
            "pixel_orientation": "stored",
            "sha256": photo.digest,
            "staged_sha256": photo.staged_digest,
        }
        for photo in photos.values()
    ]


def directory_bytes(folder: Path) -> int:
    """Measure file bytes in a workspace after processing stops."""
    total: int = 0
    for path in folder.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total

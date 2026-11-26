"""Run the cutline companion from the command line."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import cast

import cv2

from .colmap import DEFAULT_CPU_THREADS, Device
from .geometry import Coordinates, Frame, Method, Orientation, Seam, load_frame, propose
from .milestone2 import ColmapOutput, run_colmap
from .review import write_review
from .sequence import TripleReview, review_triple
from .sidecars import SidecarEdit, plan_edits, write_edits

IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
)


def parser() -> argparse.ArgumentParser:
    """Build the terminal interface with preview as the default action."""
    arguments: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Propose paired cutlines for DotDotGoose."
    )
    arguments.add_argument("folder", type=Path, help="folder containing the photos")
    selection: argparse._MutuallyExclusiveGroup = (
        arguments.add_mutually_exclusive_group()
    )
    selection.add_argument("--list", action="store_true", help="list photos and exit")
    selection.add_argument("--all", action="store_true", help="use all listed photos")
    selection.add_argument(
        "--select", metavar="INDICES", help="listed photo numbers or ranges, e.g. 1-3,5"
    )
    arguments.add_argument(
        "--write", action="store_true", help="append draft shapes to LabelMe JSON files"
    )
    arguments.add_argument(
        "--backend",
        choices=("colmap", "opencv"),
        default="colmap",
        help="COLMAP verified matches (default) or OpenCV cutlines",
    )
    arguments.add_argument(
        "--output",
        choices=("cutlines", "sparse-regions", "sfm-diagnostics", "selection-plan"),
        default="cutlines",
        help="paired cutlines (default), sparse regions, SfM diagnostics, or a colony photo recommendation",
    )
    arguments.add_argument(
        "--colmap-path", help="COLMAP executable or Windows batch launcher"
    )
    arguments.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        help="device for COLMAP feature extraction and matching (default: auto)",
    )
    arguments.add_argument(
        "--work-dir", type=Path, help="parent folder for fresh COLMAP run workspaces"
    )
    arguments.add_argument(
        "--cpu-threads",
        type=int,
        metavar="N",
        help=f"COLMAP CPU SIFT extraction workers (default: {DEFAULT_CPU_THREADS})",
    )
    arguments.add_argument(
        "--verbose",
        action="store_true",
        help="stream detailed COLMAP progress and logs",
    )
    arguments.add_argument(
        "--reuse-matches",
        type=Path,
        metavar="RUN_JSON",
        help="reuse a completed COLMAP match database",
    )
    arguments.add_argument(
        "--selection-rules",
        type=Path,
        metavar="JSON",
        help="pin or exclude photos in a selection-plan report",
    )
    arguments.add_argument(
        "--count-selection",
        help="review these counting photo indices while using the whole folder for overlap",
    )
    arguments.add_argument(
        "--target-photo-percent",
        type=int,
        metavar="PERCENT",
        help="assess additional verified overlaps and target a compact counting set",
    )
    arguments.add_argument(
        "--run-local-sfm",
        action="store_true",
        help="run suggested local reconstructions and draft cutlines from shared 3D tracks",
    )
    arguments.add_argument(
        "--compare-shared-patches",
        action="store_true",
        help="try full-folder sparse alignment and compare selection for one supported block",
    )
    arguments.add_argument(
        "--profile-resources",
        action="store_true",
        help="sample process RAM and NVIDIA memory when available",
    )
    extraction_size: argparse._MutuallyExclusiveGroup = (
        arguments.add_mutually_exclusive_group()
    )
    extraction_size.add_argument(
        "--feature-max-edge",
        type=int,
        metavar="PIXELS",
        help="cap COLMAP feature extraction only (default: native long edge)",
    )
    extraction_size.add_argument(
        "--feature-pixel-percent",
        type=float,
        metavar="PERCENT",
        help="extract using this percentage of each photo's original pixels (100 is native)",
    )
    arguments.add_argument(
        "--review-dir",
        type=Path,
        metavar="FOLDER",
        help="save a PNG showing proposed count areas and paired cutlines",
    )
    arguments.add_argument(
        "--method",
        choices=("terrain", "image", "midpoint"),
        help="OpenCV terrain path (default), image-based path, or midpoint",
    )
    arguments.add_argument(
        "--max-edge",
        type=int,
        metavar="PIXELS",
        help="OpenCV working edge for matching (default: 1800)",
    )
    arguments.add_argument(
        "--orientation",
        choices=("raw", "exif"),
        default="raw",
        help="use stored pixel orientation for DDG (default) or apply EXIF rotation",
    )
    return arguments


def image_order(path: Path) -> tuple[list[str | int], str]:
    """Sort image names naturally, keeping IMG_2 before IMG_10."""
    parts: list[str] = re.split(r"([0-9]+)", path.name.casefold())
    return [
        int(part) if index % 2 else part for index, part in enumerate(parts)
    ], path.name


def list_images(folder: Path) -> list[Path]:
    """Return supported photos in the natural order used for selection indices."""
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=image_order,
    )


def select_images(images: list[Path], value: str) -> list[Path]:
    """Parse comma-separated image numbers and ascending or descending ranges.

    Args:
        images: Available images in numbered display order.
        value: A selection such as "1-3,5" or "4-2".

    Returns:
        Selected paths in the given sequence order.

    Raises:
        ValueError: If the selection names an invalid or repeated photo.
    """
    chosen: list[int] = []
    for item in value.split(","):
        match: re.Match[str] | None = re.fullmatch(
            r"(\d+)(?:\s*-\s*(\d+))?", item.strip()
        )
        if match is None:
            raise ValueError(f"Invalid selection {item!r}. Use numbers such as 1-3,5.")
        start: int = int(match[1])
        end: int = int(match[2]) if match[2] is not None else start
        if not (1 <= start <= len(images) and 1 <= end <= len(images)):
            raise ValueError(f"Photo numbers must be between 1 and {len(images)}.")
        step: int = 1 if start <= end else -1
        chosen.extend(range(start, end + step, step))
    if len(set(chosen)) != len(chosen):
        raise ValueError("Select each photo only once.")
    return [images[index - 1] for index in chosen]


def describe_seam(seam: Seam) -> None:
    """Print alignment evidence and intended count sides for a pair."""
    name: str = f"{seam.first.path.name} / {seam.second.path.name}"
    quality: str = "DRAFT" if seam.warnings else "PROPOSED"
    method: str = {
        "terrain": "image-based with local terrain alignment",
        "image": "image-based",
        "midpoint": "midpoint",
    }[seam.method]
    first_points: Coordinates
    second_points: Coordinates
    first_points, second_points = seam.points_in_source_images()
    print(
        f"  {quality} {name}: {method}, {len(seam.points)} vertices, "
        f"{seam.inliers} inliers ({seam.match_fraction:.0%}), "
        f"keypoints in {seam.supported_sections}/4 cutline sections."
    )
    for frame, points in ((seam.first, first_points), (seam.second, second_points)):
        start: Coordinates
        end: Coordinates
        start, end = points[0], points[-1]
        print(
            f"    {frame.path.name}: ({start[0]:.0f}, {start[1]:.0f}) to "
            f"({end[0]:.0f}, {end[1]:.0f}) in source pixels."
        )
    if seam.local is not None:
        print(
            f"    Held-out landmark discrepancy (p80, working pixels): "
            f"{seam.local.global_error:.1f} before, {seam.local.local_error:.1f} after "
            f"local alignment ({seam.local.validation_count} matches)."
        )
    elif seam.error is not None:
        print(f"    Nearby match discrepancy (p80): {seam.error:.1f} working pixels.")
    sides: tuple[str, str] | None = seam.sides()
    if sides is not None:
        print(
            f"    Suggested count side near the line midpoint: {seam.first.path.name} "
            f"{sides[0]}, {seam.second.path.name} {sides[1]}."
        )
    for warning in seam.warnings:
        print(f"    Review: {warning}.")


def propose_sequence(
    frames: list[Frame], method: Method
) -> tuple[list[Seam], set[tuple[Path, Path]]]:
    """Process adjacent and one-frame-apart pairs and inspect triplets.

    Args:
        frames: Selected images in survey sequence order.
        method: Seam placement method.

    Returns:
        Placeable cutlines and the pairs requiring extra review.
    """
    pairs: dict[tuple[int, int], Seam] = {}
    draft_pairs: set[tuple[Path, Path]] = set()
    print("\nCutline preview")
    for gap in (1, 2):
        for index in range(len(frames) - gap):
            second_index: int = index + gap
            first: Frame = frames[index]
            second: Frame = frames[second_index]
            seam: Seam | None
            reason: str
            seam, reason = propose(first, second, method)
            if seam is None:
                status: str = "UNRESOLVED" if gap == 1 else "NO CONFIRMED OUTER MATCH"
                print(f"  {status} {first.path.name} / {second.path.name}: {reason}.")
                continue
            if gap == 1:
                describe_seam(seam)
            pairs[index, second_index] = seam
            if seam.warnings:
                draft_pairs.add((first.path, second.path))

    print("\nThree-photo overlap review")
    found_triple: bool = False
    for index in range(len(frames) - 2):
        ab: Seam | None = pairs.get((index, index + 1))
        bc: Seam | None = pairs.get((index + 1, index + 2))
        ac: Seam | None = pairs.get((index, index + 2))
        if ab is None or bc is None:
            continue
        review: TripleReview | None = review_triple(ab, bc, ac)
        if review is None:
            continue
        found_triple = True
        triplet: str = " / ".join(
            frame.path.name for frame in frames[index : index + 3]
        )
        print(f"  {triplet}: {review.message}")
        if review.outer_redundant:
            del pairs[index, index + 2]
        if review.uncertain:
            for candidate in (ab, bc, ac):
                if candidate is not None:
                    draft_pairs.add((candidate.first.path, candidate.second.path))
    if not found_triple:
        print(
            "  No three-photo overlap identified from the available adjacent matches."
        )
    outer_pairs: list[Seam] = [
        seam for (first, second), seam in pairs.items() if second - first == 2
    ]
    if outer_pairs:
        print("\nAdditional cutlines")
        for seam in outer_pairs:
            describe_seam(seam)
    return list(pairs.values()), draft_pairs


def run(args: argparse.Namespace) -> int:
    """Run a preview or append the planned sidecar changes.

    Args:
        args: Arguments parsed from the terminal.

    Returns:
        Process exit code.
    """
    folder: Path = args.folder.resolve(strict=True)
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a folder.")
    images: list[Path] = list_images(folder)
    if not images:
        raise ValueError("No supported image files found in the folder.")
    if args.list and args.write:
        raise ValueError("--list and --write cannot be used together.")
    if not args.list and args.output == "selection-plan" and not args.all:
        raise ValueError(
            "Use --all with selection-plan to review the full colony folder."
        )
    if args.selection_rules is not None and args.output != "selection-plan":
        raise ValueError("--selection-rules requires --output selection-plan.")
    if args.count_selection is not None and args.output != "selection-plan":
        raise ValueError("--count-selection requires --output selection-plan.")
    if args.target_photo_percent is not None and (
        args.backend != "colmap"
        or args.output != "selection-plan"
        or args.count_selection is not None
    ):
        raise ValueError("--target-photo-percent requires automatic COLMAP selection.")
    if args.run_local_sfm and args.output != "selection-plan":
        raise ValueError("--run-local-sfm requires --output selection-plan.")
    if args.list or (args.select is None and not args.all):
        print("Available photos")
        for index, path in enumerate(images, start=1):
            print(f"  {index:>3}  {path.name}")
    if args.list:
        return 0
    if args.all:
        selected: list[Path] = images
    else:
        value: str | None = args.select
        if value is None:
            if not sys.stdin.isatty():
                raise ValueError("Specify --select or --all in a noninteractive run.")
            value = input("Select photos in sequence order (e.g. 1-3,5): ")
        selected = select_images(images, value)
    minimum: int = 2 if args.output in {"cutlines", "selection-plan"} else 3
    if len(selected) < minimum:
        raise ValueError(f"Select at least {minimum} photos.")
    stem_counts: dict[str, int] = {}
    for path in images:
        stem: str = path.stem.casefold()
        stem_counts[stem] = stem_counts.get(stem, 0) + 1
    for path in selected:
        if stem_counts[path.stem.casefold()] > 1:
            raise ValueError(
                f"More than one image maps to the sidecar for {path.name}."
            )

    counting_order: list[int] | None = None
    if args.count_selection is not None:
        indices: dict[Path, int] = {path: index for index, path in enumerate(images)}
        counting_order = [
            indices[path] for path in select_images(images, args.count_selection)
        ]
        if len(counting_order) < 2:
            raise ValueError("Select at least two photos to count.")

    if args.backend == "colmap":
        if (
            args.max_edge is not None
            or args.method is not None
            or args.review_dir is not None
            or args.orientation != "raw"
        ):
            raise ValueError(
                "Use --backend opencv for --max-edge, --method, --review-dir, "
                "or --orientation exif. COLMAP uses native stored pixels."
            )
        if args.reuse_matches is not None and (
            args.device is not None
            or args.cpu_threads is not None
            or args.feature_max_edge is not None
            or args.feature_pixel_percent is not None
        ):
            raise ValueError(
                "--reuse-matches uses the saved device, worker count, and extraction size."
            )
        device: Device = cast(Device, args.device or "auto")
        threads: int = (
            DEFAULT_CPU_THREADS if args.cpu_threads is None else args.cpu_threads
        )
        return run_colmap(
            selected,
            args.colmap_path,
            device,
            args.work_dir,
            args.write,
            threads,
            cast(ColmapOutput, args.output),
            verbose=args.verbose,
            reuse_matches=args.reuse_matches,
            profile_resources=args.profile_resources,
            feature_max_edge=args.feature_max_edge,
            feature_pixel_percent=args.feature_pixel_percent,
            selection_rules=args.selection_rules,
            local_sfm=args.run_local_sfm,
            compare_shared_patches=args.compare_shared_patches,
            counting_order=counting_order,
            target_photo_percent=args.target_photo_percent,
        )

    if (
        args.colmap_path is not None
        or args.work_dir is not None
        or args.device not in (None, "auto")
        or args.cpu_threads is not None
        or args.verbose
        or args.reuse_matches is not None
        or args.profile_resources
        or args.feature_max_edge is not None
        or args.feature_pixel_percent is not None
        or args.compare_shared_patches
    ):
        raise ValueError(
            "COLMAP device, workspace, progress, profiling, and reuse options require --backend colmap."
        )
    if args.output != "cutlines":
        raise ValueError(f"--output {args.output} requires --backend colmap.")
    edge: int = 1800 if args.max_edge is None else args.max_edge
    if edge < 256 or edge > 6000:
        raise ValueError("--max-edge must be between 256 and 6000 pixels.")

    orientation: Orientation = cast(Orientation, args.orientation)
    method: Method = cast(Method, args.method or "terrain")
    detector: cv2.SIFT = cv2.SIFT_create(nfeatures=4500)
    print("\nSelected sequence")
    frames: list[Frame] = []
    for index, path in enumerate(selected, start=1):
        print(f"  {index:>3}  {path.name}", flush=True)
        frames.append(load_frame(path, edge, detector, orientation))

    seams: list[Seam]
    draft_pairs: set[tuple[Path, Path]]
    seams, draft_pairs = propose_sequence(frames, method)
    edits: list[SidecarEdit]
    messages: list[str]
    edits, messages = plan_edits(seams, draft_pairs)
    print("\nSidecar plan")
    for message in messages:
        print(message)
    if not edits:
        print("  No sidecar changes planned.")
    else:
        for edit in edits:
            action: str = "create" if edit.original is None else "append to"
            print(f"  {action} {edit.path.name}")
    if args.review_dir is not None:
        review_path: Path = write_review(
            args.review_dir, frames, seams, draft_pairs, orientation
        )
        print(f"\nCount-area review: {review_path}")
    if args.write:
        write_edits(folder, edits)
        print(f"\nUpdated {len(edits)} JSON sidecar(s). Review the cutlines in DDG.")
    else:
        print("\nPreview only. Rerun the same selection with --write to append.")
    return 0


def main() -> int:
    """Report expected command-line errors without a Python traceback."""
    args: argparse.Namespace = parser().parse_args()
    try:
        return run(args)
    except (OSError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

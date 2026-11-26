"""Validate and reuse the staged images and verified matches from a COLMAP job."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from .benchmark import RunReport
from .colmap import ComputeDevice, VerifiedPair, verified_pairs
from .colmap_model import Photo
from .geometry import MIN_CUTLINE_MATCHES, Pixels


@dataclass(frozen=True)
class MatchAssets:
    """The database and native image grid belonging to one matching attempt."""

    database: Path
    image_folder: Path
    device: ComputeDevice
    settings: dict[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        """Record a completed checkpoint for future jobs."""
        return {
            "database": str(self.database),
            "image_folder": str(self.image_folder),
            "device": self.device,
            "settings": self.settings,
        }


def command_option(argv: list[str], *names: str) -> str | None:
    """Read a recorded command-line option and its adjacent value."""
    return next((value for name, value in zip(argv, argv[1:]) if name in names), None)


def legacy_checkpoint(manifest: dict[str, Any], path: Path) -> dict[str, Any]:
    """Resolve completed movable-endpoints runs with recorded matching commands."""
    if manifest.get("status") not in {"complete", "partial"}:
        raise ValueError(
            "The earlier run did not finish with a reusable match database."
        )
    device: str = manifest.get("device_used", "")
    attempt: dict[str, Any] | None = next(
        (
            item
            for item in reversed(manifest.get("matching_attempts", []))
            if item.get("device") == device and item.get("verified_pairs")
        ),
        None,
    )
    if attempt is None:
        raise ValueError("The earlier run has no successful matching attempt.")
    recorded: str = attempt["database"]
    commands: dict[str, list[str]] = {
        argv[1]: argv[2:]
        for argv in manifest.get("commands", [])
        if len(argv) > 2 and command_option(argv, "--database_path") == recorded
    }
    extract: list[str] | None = commands.get("feature_extractor")
    match: list[str] | None = commands.get("exhaustive_matcher")
    if extract is None or match is None or device not in {"cpu", "gpu"}:
        raise ValueError(
            "The earlier matching attempt lacks extraction or matching provenance."
        )
    cap: str | None = command_option(
        extract, "--FeatureExtraction.max_image_size", "--SiftExtraction.max_image_size"
    )
    if cap is None or not cap.isdecimal() or int(cap) < 1:
        raise ValueError("The earlier extraction long edge could not be verified.")
    # The original job can be moved intact even if the old manifest used absolute paths.
    database: Path = path.parent / device / "database.db"
    return {
        "database": str(database if database.is_file() else recorded),
        "image_folder": str(path.parent / "images"),
        "device": device,
        "settings": {
            "colmap_version": manifest.get("colmap_version"),
            "cpu_extraction_threads": manifest.get("cpu_extraction_threads"),
            "effective_feature_max_edge": int(cap),
            "feature_extractor_arguments": extract,
            "exhaustive_matcher_arguments": match,
        },
    }


def match_assets(source: dict[str, Any], path: Path) -> MatchAssets:
    """Resolve a saved database and its staged image folder.

    Args:
        source: Saved run manifest.
        path: Path to the manifest, used to locate intact moved jobs.

    Returns:
        The matching assets and recorded settings, before photo validation.
    """
    if source.get("backend") != "colmap":
        raise ValueError("Select a COLMAP run.json.")
    checkpoint: dict[str, Any] = source.get("match_checkpoint") or legacy_checkpoint(
        source, path
    )
    device: str = checkpoint["device"]
    if device not in {"gpu", "cpu"}:
        raise ValueError("Saved matching device is invalid.")
    database: Path = Path(checkpoint["database"])
    image_folder: Path = Path(checkpoint["image_folder"])
    if not source.get("match_reuse"):
        # Prefer the files alongside the manifest for an intact moved run.
        if (path.parent / device / "database.db").is_file():
            database = path.parent / device / "database.db"
        if (path.parent / "images").is_dir():
            image_folder = path.parent / "images"
    if not database.is_file():
        raise ValueError(f"Saved match database is missing: {database}")
    return MatchAssets(
        database.resolve(),
        image_folder.resolve(),
        cast(ComputeDevice, device),
        checkpoint["settings"],
    )


def load_matches(
    manifest_path: Path, selected: list[Path], report: RunReport
) -> tuple[MatchAssets, dict[str, Photo]]:
    """Validate and reopen completed matching assets.

    Args:
        manifest_path: Manifest that identifies the saved matching job.
        selected: Original photos in the same order as the saved run.
        report: Current run's progress reporter for photo validation.

    Returns:
        The reusable matching assets and photos keyed by staged filename.

    Raises:
        ValueError: If the original photos, staged pixels, or verified matches
            differ from the saved run or cannot be validated.
    """
    path: Path = manifest_path.resolve()
    try:
        source: dict[str, Any] = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(source, dict):
            raise ValueError("Select a COLMAP run.json.")
        assets: MatchAssets = match_assets(source, path)
        rows: list[dict[str, Any]] = source["images"]
        if [row["name"] for row in rows] != [photo.name for photo in selected]:
            raise ValueError(
                "The selected filenames or selection order differ from the saved run."
            )
        photos: dict[str, Photo] = {}
        for index, (row, original) in enumerate(
            zip(rows, selected, strict=True), start=1
        ):
            photo: Photo = validate_photo(row, original, assets.image_folder, index)
            photos[photo.staged_name] = photo
            report.update(
                f"validated {original.name}" if report.verbose else "photos validated",
                index,
                len(selected),
            )
        validate_database(assets.database, photos)
        verified: list[VerifiedPair] = verified_pairs(assets.database, photos)
        if not any(item["inliers"] >= MIN_CUTLINE_MATCHES for item in verified):
            raise ValueError("The saved database has no usable verified matches.")
        if (
            source.get("verified_pairs") is not None
            and verified != source["verified_pairs"]
        ):
            raise ValueError(
                "Verified pairs in the database differ from the saved run."
            )
        return assets, photos
    except (
        KeyError,
        TypeError,
        AttributeError,
        OverflowError,
        OSError,
        sqlite3.Error,
        cv2.error,
    ) as error:
        raise ValueError(
            f"Cannot reuse matches from {manifest_path}: {error}"
        ) from error


def validate_photo(
    row: dict[str, Any], original: Path, folder: Path, index: int
) -> Photo:
    """Confirm the staged file corresponds to the selected native pixel grid."""
    name: str = f"{index:06}.png"
    if row["staged_name"] != name or row["pixel_orientation"] != "stored":
        raise ValueError(
            f"The staged mapping or orientation is invalid for {original.name}."
        )
    width: int = row["width"]
    height: int = row["height"]
    original_bytes: bytes = original.read_bytes()
    digest: str = hashlib.sha256(original_bytes).hexdigest()
    if digest != row["sha256"]:
        raise ValueError(f"Source photo changed since matching: {original.name}")
    staged: Path = folder / name
    if not staged.is_file():
        raise ValueError(f"Staged image is missing: {staged}")
    staged_bytes: bytes = staged.read_bytes()
    staged_digest: str = hashlib.sha256(staged_bytes).hexdigest()
    if row.get("staged_sha256") is not None:
        if staged_digest != row["staged_sha256"]:
            raise ValueError(f"Staged image changed since matching: {staged}")
    else:
        # Legacy manifests did not hash the lossless PNG. Verify its decoded pixels once.
        flags: int = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        expected: Pixels | None = cv2.imdecode(
            np.frombuffer(original_bytes, np.uint8), flags
        )
        stored: Pixels | None = cv2.imdecode(
            np.frombuffer(staged_bytes, np.uint8), flags
        )
        if (
            expected is None
            or stored is None
            or expected.shape[:2] != (height, width)
            or not np.array_equal(expected, stored)
        ):
            raise ValueError(
                f"Staged pixels differ from the native source for {original.name}."
            )
    return Photo(original, width, height, name, digest, staged_digest)


def validate_database(database: Path, photos: dict[str, Photo]) -> None:
    """Verify the database's image mapping, pixel grids, and match record format."""
    with closing(
        sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    ) as connection:
        sizes: dict[str, tuple[int, int]] = {
            name: (width, height)
            for name, width, height in connection.execute(
                "SELECT images.name, cameras.width, cameras.height FROM images "
                "LEFT JOIN cameras USING (camera_id)"
            )
        }
        if sizes != {
            name: (photo.width, photo.height) for name, photo in photos.items()
        }:
            raise ValueError(
                "Saved database images or dimensions differ from the selected photos."
            )
        corrupt: int = connection.execute(
            "SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0 AND "
            "(cols != 2 OR data IS NULL OR length(data) != 8 * rows)"
        ).fetchone()[0]
        if corrupt:
            raise ValueError(
                "The saved database contains invalid verified correspondence records."
            )


def copy_database(source: Path, destination: Path) -> None:
    """Use SQLite's backup API to copy committed data, including a live WAL."""
    destination.parent.mkdir()
    with (
        closing(sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)) as reader,
        closing(sqlite3.connect(destination)) as writer,
    ):
        reader.backup(writer)

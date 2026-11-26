"""Run an external COLMAP executable in an isolated reconstruction workspace."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict, deque
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from threading import Thread
from time import perf_counter
from typing import (
    Any,
    BinaryIO,
    Callable,
    IO,
    Iterator,
    Literal,
    TypeAlias,
    TypedDict,
    TypeVar,
)

import cv2
import numpy as np
from numpy.typing import NDArray

from .benchmark import RunReport
from .colmap_model import Model, Photo, read_model
from .geometry import Coordinates, Pixels

Device: TypeAlias = Literal["auto", "cpu", "gpu"]
ComputeDevice: TypeAlias = Literal["cpu", "gpu"]
Result = TypeVar("Result")
DEFAULT_CPU_THREADS: int = 1
PAIR_ID_BASE: int = 2147483647
HELP_TIMEOUT_SECONDS: float = 30.0
SEM_FAILCRITICALERRORS: int = 0x0001
SEM_NOGPFAULTERRORBOX: int = 0x0002
GPU_FAILURE: re.Pattern[str] = re.compile(
    r"no (?:compatible )?(?:cuda|hip|gpu)|"
    r"(?:cuda|gpu|siftgpu|cublas|device|cudaerror|hiperror).{0,140}"
    r"(?:not (?:enabled|supported|found|available|compiled)|error|failed|"
    r"out of memory|unavailable|invalid|insufficient)|"
    r"(?:failed|error|out of memory).{0,100}(?:cuda|gpu|siftgpu|cublas)|"
    r"Check failed:.*(?:sift_gpu|use_gpu|gpu|cuda)",
    re.IGNORECASE,
)
COLMAP_STAGES: dict[str, str] = {
    "feature_extractor": "feature_extraction",
    "exhaustive_matcher": "matching_and_verification",
    "mapper": "sparse_reconstruction",
    "model_converter": "model_conversion",
    "image_deleter": "dense_select_views",
    "image_undistorter": "dense_undistort",
    "patch_match_stereo": "dense_stereo",
}
PROCESSED: re.Pattern[str] = re.compile(r"Processed file \[(\d+)/(\d+)\]")


@contextmanager
def external_process_startup() -> Iterator[None]:
    """Give frozen Windows children normal DLL lookup and noninteractive errors."""
    bundle: str | None = getattr(sys, "_MEIPASS", None)
    if sys.platform != "win32" or bundle is None:
        yield
        return
    import ctypes
    from ctypes import wintypes

    kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)
    set_directory: Any = kernel32.SetDllDirectoryW
    set_directory.argtypes = (wintypes.LPCWSTR,)
    set_directory.restype = wintypes.BOOL
    if not set_directory(None):
        raise ctypes.WinError(ctypes.get_last_error())
    get_error_mode: Any = kernel32.GetErrorMode
    get_error_mode.argtypes = ()
    get_error_mode.restype = wintypes.UINT
    set_error_mode: Any = kernel32.SetErrorMode
    set_error_mode.argtypes = (wintypes.UINT,)
    set_error_mode.restype = wintypes.UINT
    previous_mode: int = get_error_mode()
    # The child inherits these flags. Return critical errors and crashes to the
    # launcher even when Windows would normally display a dialog.
    set_error_mode(previous_mode | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX)
    try:
        yield
    finally:
        set_error_mode(previous_mode)
        if not set_directory(bundle):
            raise ctypes.WinError(ctypes.get_last_error())


class VerifiedPair(TypedDict):
    """Image names and the inlier count after geometric matching."""

    images: list[str]
    inliers: int


class ColmapError(ValueError):
    """A subprocess failure with the full command output retained in its log."""

    def __init__(self, command: str, output: str, log: Path, returncode: int) -> None:
        """Keep the exit status and output so early process exits can be diagnosed."""
        self.command: str = command
        self.output: str = output
        self.log: Path = log
        self.returncode: int = returncode
        self.gpu_related: bool = bool(GPU_FAILURE.search(output))
        # Windows native crashes may carry an NTSTATUS with or without a sign.
        self.exit_status: str = str(returncode)
        if os.name == "nt" and returncode & 0x80000000:
            self.exit_status += f" (0x{returncode & 0xFFFFFFFF:08X})"
        message: str = "\n".join(output.strip().splitlines()[-7:])
        super().__init__(
            f"COLMAP {command} failed with exit code {self.exit_status}. See {log}.\n{message}"
        )


@dataclass
class Colmap:
    """A native executable and the environment required by the portable build."""

    executable: Path
    environment: dict[str, str]
    commands: list[list[str]] = field(default_factory=list)
    command_timings: list[dict[str, Any]] = field(default_factory=list)
    report: RunReport | None = None
    feature_max_edge: int | None = None
    feature_pixel_percent: float | None = None

    def run(self, command: list[str], log: Path) -> str:
        """Stream processing commands and bound the wait for setup help."""
        argv: list[str] = [str(self.executable), *command]
        self.commands.append(argv)
        capture: bool = "-h" in command or "--help" in command
        stage: str | None = None if capture else COLMAP_STAGES.get(command[0])
        attempt: str | None = (
            log.parent.name if log.parent.name in {"cpu", "gpu", "reused"} else None
        )
        report: RunReport | None = self.report
        with (
            report.phase(stage, attempt)
            if report is not None and stage is not None
            else nullcontext()
        ):
            start: float = perf_counter()
            timing: dict[str, Any] = {
                "command": command[0],
                "log_path": str(log),
                "attempt": attempt,
                "returncode": None,
                "status": "failed",
            }
            self.command_timings.append(timing)
            try:
                if report is not None and (capture or report.verbose):
                    report.update(f"{subprocess.list2cmdline(argv)} | log {log}")
                if capture:
                    if report is not None:
                        report.save()
                    output: str
                    code: int
                    output, code = self.read_help(argv, log)
                    gpu_error: bool = False
                else:
                    output, code, gpu_error = self.stream(argv, log)
                timing["returncode"] = code
                if code:
                    error: ColmapError = ColmapError(command[0], output, log, code)
                    error.gpu_related |= gpu_error
                    raise error
                timing["status"] = "completed"
                return output
            except KeyboardInterrupt:
                timing["status"] = "interrupted"
                raise
            except subprocess.TimeoutExpired as error:
                timing.update(status="timed_out", timeout_seconds=error.timeout)
                check: str = subprocess.list2cmdline(command)
                raise TimeoutError(
                    f"COLMAP setup check {check} timed out after {error.timeout:g} seconds. "
                    f"Executable: {self.executable}. See {log}."
                ) from error
            finally:
                timing["elapsed_seconds"] = round(perf_counter() - start, 3)

    def launch(self, argv: list[str], stdout: int | BinaryIO) -> subprocess.Popen[str]:
        """Start the native tool without sharing the worker's cancellation input."""
        with external_process_startup():
            return subprocess.Popen(
                argv,
                env=self.environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if sys.platform == "win32" and getattr(sys, "frozen", False)
                    else 0
                ),
            )

    def read_help(self, argv: list[str], log: Path) -> tuple[str, int]:
        """Log help directly so another process holding a pipe cannot delay setup."""
        with log.open("wb") as output, self.launch(argv, output) as process:
            deadline: float = perf_counter() + HELP_TIMEOUT_SECONDS
            try:
                while process.poll() is None:
                    remaining: float = deadline - perf_counter()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(argv, HELP_TIMEOUT_SECONDS)
                    try:
                        process.wait(timeout=min(0.25, remaining))
                    except subprocess.TimeoutExpired:
                        continue
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                process.kill()
                raise
            code: int = process.wait()
        return log.read_text(encoding="utf-8", errors="replace"), code

    def stream(self, argv: list[str], log: Path) -> tuple[str, int, bool]:
        """Drain the child pipe in a worker while the main thread can cancel it."""
        tail: deque[str] = deque(maxlen=100)
        failure: list[Exception] = []
        gpu_error: bool = False

        with log.open("w", encoding="utf-8", buffering=1) as output:
            with self.launch(argv, subprocess.PIPE) as process:
                pipe: IO[str] | None = process.stdout
                assert pipe is not None

                def pump() -> None:
                    """Write immediately and relay progress reported by COLMAP."""
                    nonlocal gpu_error
                    try:
                        for line in pipe:
                            output.write(line)
                            tail.append(line)
                            if not gpu_error and GPU_FAILURE.search(line):
                                gpu_error = True
                            self.show_child_progress(line)
                    except (OSError, ValueError) as error:
                        failure.append(error)

                reader: Thread = Thread(target=pump, daemon=True)
                reader.start()
                try:
                    while process.poll() is None:
                        if failure:
                            raise failure[0]
                        try:
                            process.wait(timeout=0.25)
                        except subprocess.TimeoutExpired:
                            continue
                    while reader.is_alive():
                        reader.join(timeout=0.25)
                    if failure:
                        raise failure[0]
                except (KeyboardInterrupt, OSError, ValueError):
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    reader.join()
                    raise
                return "".join(tail), process.wait(), gpu_error

    def show_child_progress(self, line: str) -> None:
        """Expose reliable COLMAP milestones without inventing a percentage."""
        report: RunReport | None = self.report
        if report is None:
            return
        if report.verbose:
            print(line.rstrip(), flush=True)
        image: re.Match[str] | None = PROCESSED.search(line)
        if image is not None:
            report.update("COLMAP photos processed", int(image[1]), int(image[2]))
        elif any(
            token in line
            for token in (
                "Processing block [",
                "Registering image #",
                "Registering initial image pair",
                "Keeping successful reconstruction",
                "Discarding reconstruction",
            )
        ):
            message: str = line.split("] ", 1)[-1].strip()
            report.update(message)


def find_colmap(explicit: str | None) -> Colmap:
    """Resolve an explicit executable, COLMAP_PATH, or the system PATH."""
    requested: str | None = explicit or os.environ.get("COLMAP_PATH")
    if requested is None:
        requested = shutil.which("colmap") or shutil.which("COLMAP.bat")
    if requested is None:
        raise ValueError("COLMAP was not found. Set COLMAP_PATH or use --colmap-path.")
    path: Path = Path(requested).expanduser()
    if not path.is_file():
        found: str | None = shutil.which(requested)
        if found is None:
            raise ValueError(f"COLMAP executable does not exist: {requested}")
        path = Path(found)
    path = path.resolve()
    environment: dict[str, str] = dict(os.environ)
    root: Path = path.parent
    if path.suffix.lower() in {".bat", ".cmd"}:
        # Use the portable executable directly. Windows cmd expansion would
        # interpret characters in paths and would also complicate quoting.
        binaries: tuple[Path, Path] = root / "bin" / "colmap.exe", root / "colmap.exe"
        executable: Path | None = next(
            (item for item in binaries if item.is_file()), None
        )
        if executable is None:
            raise ValueError(
                f"No colmap.exe found beside {path}. Pass its native executable path."
            )
        path = executable
    elif path.suffix.lower() == ".exe" and root.name.casefold() == "bin":
        root = root.parent
    search_path: list[str] = environment.get("PATH", "").split(os.pathsep)
    bundle: str | None = getattr(sys, "_MEIPASS", None)
    if sys.platform == "win32" and bundle is not None:
        runtime: Path = Path(bundle)
        search_path = [
            part
            for part in search_path
            if part and not Path(part.strip('"')).is_relative_to(runtime)
        ]
        packaged_colmap: Path = Path(sys.executable).resolve().parent / "colmap"
        if path.is_relative_to(packaged_colmap):
            # Run the portable CPU build with the same dependency search path
            # used by the release smoke test, including from a Conda prompt.
            windows: str | None = environment.get("SYSTEMROOT") or environment.get(
                "SystemRoot"
            )
            if windows is not None:
                search_path = [str(Path(windows) / "System32"), windows]
            for variable in (
                "QT_PLUGIN_PATH",
                "QT_QPA_PLATFORM_PLUGIN_PATH",
                "QT_QPA_PLATFORM",
            ):
                environment.pop(variable, None)
    for plugins in (root / "plugins", path.parent / "plugins"):
        if plugins.is_dir():
            environment.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
            environment["QT_PLUGIN_PATH"] = str(plugins)
            break
    environment["PATH"] = os.pathsep.join((str(path.parent), *search_path))
    if sys.platform != "win32" and not (
        environment.get("DISPLAY") or environment.get("WAYLAND_DISPLAY")
    ):
        environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    return Colmap(path, environment)


def stage_photos(
    paths: list[Path], folder: Path, report: RunReport | None = None
) -> dict[str, Photo]:
    """Make lossless, native-size copies in DDG's stored pixel orientation.

    Args:
        paths: The explicitly selected source image files.
        folder: Private job-local image directory.
        report: Optional progress display for staging.

    Returns:
        Source metadata keyed by the safe filenames passed to COLMAP.
    """
    folder.mkdir()
    photos: dict[str, Photo] = {}
    for index, path in enumerate(paths, start=1):
        original: bytes = path.read_bytes()
        encoded: Pixels = np.frombuffer(original, dtype=np.uint8)
        color: Pixels | None = (
            cv2.imdecode(encoded, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if encoded.size
            else None
        )
        if color is None:
            raise ValueError(f"Cannot decode {path.name}.")
        height: int
        width: int
        height, width = color.shape[:2]
        name: str = f"{index:06}.png"
        success: bool
        png: Pixels
        success, png = cv2.imencode(".png", color, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not success:
            raise OSError(f"Cannot stage the native pixel grid for {path.name}.")
        (folder / name).write_bytes(png.tobytes())
        photos[name] = Photo(
            path,
            width,
            height,
            name,
            hashlib.sha256(original).hexdigest(),
            hashlib.sha256(png).hexdigest(),
        )
        if report is not None:
            message: str = (
                f"{path.name} | {width} x {height} stored pixels"
                if report.verbose
                else "photos staged"
            )
            report.update(message, index, len(paths))
    return photos


@dataclass(frozen=True)
class Flags:
    """Version-specific COLMAP options confirmed by subcommand help."""

    image_limit: str
    extraction_gpu: str
    extraction_threads: str
    matching_gpu: str
    mapper_gpu: tuple[str, ...]
    extraction_type: bool
    matching_type: bool
    mapper_image_list: str | None = None
    extraction_image_list: bool = False


def advertises(help_text: str, option: str) -> bool:
    """Check for a complete option token in the installed CLI help."""
    return re.search(rf"(?<!\S){re.escape(option)}(?=\s|$)", help_text) is not None


def supported_flag(help_text: str, *choices: str) -> str:
    """Return the first option advertised in COLMAP's help."""
    for choice in choices:
        if advertises(help_text, choice):
            return choice
    raise ValueError(f"Installed COLMAP does not advertise {' or '.join(choices)}.")


def inspect_flags(engine: Colmap, job: Path) -> tuple[str, Flags]:
    """Record the COLMAP version and detect the supported SIFT flag family."""
    help_text: str = engine.run(["-h"], job / "version.log")
    version: str = next(
        (line.strip() for line in help_text.splitlines() if "COLMAP" in line), "COLMAP"
    )
    extract: str = engine.run(["feature_extractor", "-h"], job / "extract-help.log")
    matching: str = engine.run(["exhaustive_matcher", "-h"], job / "match-help.log")
    mapper: str = engine.run(["mapper", "-h"], job / "mapper-help.log")
    return version, Flags(
        supported_flag(
            extract,
            "--FeatureExtraction.max_image_size",
            "--SiftExtraction.max_image_size",
        ),
        supported_flag(
            extract, "--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"
        ),
        supported_flag(
            extract, "--FeatureExtraction.num_threads", "--SiftExtraction.num_threads"
        ),
        supported_flag(matching, "--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"),
        tuple(
            flag
            for flag in (
                "--Mapper.ba_use_gpu",
                "--Mapper.ba_use_gpu_for_global_bundle_adjustment",
            )
            if advertises(mapper, flag)
        ),
        advertises(extract, "--FeatureExtraction.type"),
        advertises(matching, "--FeatureMatching.type")
        and "SIFT_BRUTEFORCE" in matching,
        # Preserve the exact spelling accepted by this mapper.
        next(
            (
                flag
                for flag in ("--Mapper.image_list_path", "--image_list_path")
                if advertises(mapper, flag)
            ),
            None,
        ),
        advertises(extract, "--image_list_path"),
    )


def verified_pairs(database: Path, photos: dict[str, Photo]) -> list[VerifiedPair]:
    """Read geometrically verified pair counts from the COLMAP database."""
    uri: str = f"{database.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        names: dict[int, str] = {
            image_id: photos[name].path.name
            for image_id, name in connection.execute(
                "SELECT image_id, name FROM images"
            )
            if name in photos
        }
        results: list[VerifiedPair] = []
        # COLMAP encodes pair IDs using the maximum signed 32-bit image ID.
        for pair_id, count in connection.execute(
            "SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0 ORDER BY pair_id"
        ):
            first: int
            second: int
            first, second = divmod(pair_id, PAIR_ID_BASE)
            if first in names and second in names:
                results.append(
                    {"images": [names[first], names[second]], "inliers": count}
                )
        return results


class MatchingCounts(TypedDict):
    """COLMAP database counts that distinguish extraction and matching failures."""

    keypoints: dict[str, int]
    descriptors: dict[str, int]
    raw_matched_pairs: int


def matching_counts(database: Path, photos: dict[str, Photo]) -> MatchingCounts:
    """Read per-photo features and pre-verification pair counts for the job."""
    keypoints: dict[str, int] = {photo.path.name: 0 for photo in photos.values()}
    descriptors: dict[str, int] = keypoints.copy()
    uri: str = f"{database.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        for name, keys, features in connection.execute(
            "SELECT i.name, COALESCE(k.rows, 0), COALESCE(d.rows, 0) "
            "FROM images AS i LEFT JOIN keypoints AS k USING (image_id) "
            "LEFT JOIN descriptors AS d USING (image_id)"
        ):
            if name in photos:
                filename: str = photos[name].path.name
                keypoints[filename] = keys
                descriptors[filename] = features
        rows: int = connection.execute(
            "SELECT COUNT(*) FROM matches WHERE rows > 0"
        ).fetchone()[0]
    return {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "raw_matched_pairs": rows,
    }


def extraction_batches(
    engine: Colmap, photos: dict[str, Photo]
) -> dict[int, list[str]]:
    """Group staged filenames by their COLMAP extraction size limit.

    Args:
        engine: COLMAP runner with the requested extraction scale.
        photos: Native photo metadata keyed by staged filename.

    Returns:
        Staged filenames grouped by maximum long edge, in pixels.
    """
    percent: float | None = engine.feature_pixel_percent
    if percent is None or percent == 100:
        native_edge: int = max(
            max(photo.width, photo.height) for photo in photos.values()
        )
        return {min(native_edge, engine.feature_max_edge or native_edge): list(photos)}

    # An area percentage scales both dimensions by the square root of the ratio.
    linear_scale: float = sqrt(percent / 100)
    batches: dict[int, list[str]] = defaultdict(list)
    for name, photo in photos.items():
        edge: int = max(1, round(max(photo.width, photo.height) * linear_scale))
        batches[edge].append(name)
    return dict(sorted(batches.items()))


def extract_and_match(
    engine: Colmap,
    flags: Flags,
    job: Path,
    device: ComputeDevice,
    photos: dict[str, Photo],
    cpu_threads: int,
) -> Path:
    """Extract features at the requested scale and match all images together."""
    batches: dict[int, list[str]] = extraction_batches(engine, photos)
    split: bool = len(batches) > 1
    if split and not flags.extraction_image_list:
        raise ValueError(
            "This COLMAP feature extractor lacks --image_list_path, which is "
            "required to scale mixed-size photos by pixel percentage."
        )
    attempt: Path = job / device
    attempt.mkdir()
    database: Path = attempt / "database.db"
    use_gpu: str = "1" if device == "gpu" else "0"
    image_path: list[str] = ["--image_path", str(job / "images")]
    database_path: list[str] = ["--database_path", str(database)]
    extract_type: list[str] = (
        ["--FeatureExtraction.type", "SIFT"] if flags.extraction_type else []
    )
    match_type: list[str] = (
        ["--FeatureMatching.type", "SIFT_BRUTEFORCE"] if flags.matching_type else []
    )
    extraction_threads: list[str] = []
    if device == "cpu":
        # A full-resolution SIFT pyramid is allocated separately by each worker.
        extraction_threads = [flags.extraction_threads, str(cpu_threads)]
        print(f"COLMAP CPU SIFT extraction workers: {cpu_threads}.", flush=True)
    print(
        f"COLMAP {device}: extraction long edge up to {max(batches):,} px and "
        f"{len(photos) * (len(photos) - 1) // 2:,} candidate pairs.",
        flush=True,
    )
    for edge, names in batches.items():
        subset: list[str] = []
        log: str = "extract.log"
        if split:
            # All groups write to the same database so matching can cross image sizes.
            image_list: Path = attempt / f"extract-{edge}.txt"
            image_list.write_text("\n".join(names) + "\n", encoding="utf-8")
            subset = ["--image_list_path", str(image_list)]
            log = f"extract-{edge}.log"
        engine.run(
            [
                "feature_extractor",
                *database_path,
                *image_path,
                *subset,
                "--ImageReader.camera_model",
                "SIMPLE_RADIAL",
                flags.image_limit,
                str(edge),
                flags.extraction_gpu,
                use_gpu,
                *extraction_threads,
                *extract_type,
            ],
            attempt / log,
        )
    engine.run(
        [
            "exhaustive_matcher",
            *database_path,
            flags.matching_gpu,
            use_gpu,
            *match_type,
        ],
        attempt / "match.log",
    )
    return database


def reconstruct_attempt(
    engine: Colmap,
    flags: Flags,
    job: Path,
    device: ComputeDevice,
    photos: dict[str, Photo],
    cpu_threads: int,
) -> tuple[list[Model], list[VerifiedPair]]:
    """Extract, match all pairs and read every model for one device attempt."""
    database: Path = extract_and_match(engine, flags, job, device, photos, cpu_threads)
    return map_matches(engine, flags, database, job / "images", photos), verified_pairs(
        database, photos
    )


def map_matches(
    engine: Colmap,
    flags: Flags,
    database: Path,
    image_folder: Path,
    photos: dict[str, Photo],
    *,
    subset_job: Path | None = None,
) -> list[Model]:
    """Run sparse reconstruction directly from a prepared match database."""
    attempt: Path = subset_job if subset_job is not None else database.parent
    subset_options: list[str] = []
    if subset_job is not None:
        if flags.mapper_image_list is None:
            raise ValueError("COLMAP mapper has no image-list option.")
        attempt.mkdir(parents=True)
        image_list: Path = attempt / "images.txt"
        image_list.write_text("\n".join(photos) + "\n", encoding="utf-8")
        subset_options = [flags.mapper_image_list, str(image_list)]
    sparse: Path = attempt / "sparse"
    sparse.mkdir()
    gpu_options: list[str] = [
        value for flag in flags.mapper_gpu for value in (flag, "0")
    ]
    engine.run(
        [
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(image_folder),
            "--output_path",
            str(sparse),
            *gpu_options,
            *subset_options,
        ],
        attempt / "mapper.log",
    )

    models: list[Model] = []
    for source in sorted(sparse.iterdir()):
        if not source.is_dir() or not (source / "cameras.bin").is_file():
            continue
        destination: Path = attempt / f"model-{source.name}"
        destination.mkdir()
        engine.run(
            [
                "model_converter",
                "--input_path",
                str(source),
                "--output_path",
                str(destination),
                "--output_type",
                "TXT",
            ],
            destination / "converter.log",
        )
        models.append(read_model(destination, photos))
    return models


def reconstruct(
    engine: Colmap,
    flags: Flags,
    job: Path,
    requested: Device,
    photos: dict[str, Photo],
    cpu_threads: int,
) -> tuple[list[Model], list[VerifiedPair], str, str | None]:
    """Reconstruct sparse models using the requested device policy.

    Args:
        engine: COLMAP runner and extraction settings.
        flags: Options supported by the installed COLMAP build.
        job: Workspace containing the staged source pixels.
        requested: CPU, GPU, or automatic device selection.
        photos: Photos keyed by their staged filenames.
        cpu_threads: CPU feature-extraction worker count.

    Returns:
        Reconstructed blocks, verified pairs, effective device, and any reason
        for retrying the extraction on the CPU.
    """
    outcome: tuple[list[Model], list[VerifiedPair]]
    used: ComputeDevice
    fallback: str | None
    outcome, used, fallback = run_on_device(
        requested,
        lambda device: reconstruct_attempt(
            engine, flags, job, device, photos, cpu_threads
        ),
    )
    return *outcome, used, fallback


def run_on_device(
    requested: Device,
    attempt: Callable[[ComputeDevice], Result],
    on_fallback: Callable[[str], None] | None = None,
) -> tuple[Result, ComputeDevice, str | None]:
    """Retry a COLMAP setup on CPU only after a GPU-specific failure in auto mode."""
    if requested == "cpu":
        return attempt("cpu"), "cpu", None
    try:
        return attempt("gpu"), "gpu", None
    except ColmapError as error:
        if (
            requested != "auto"
            or error.command not in {"feature_extractor", "exhaustive_matcher"}
            or not error.gpu_related
        ):
            raise
        reason: str = str(error)
        print(f"GPU SIFT unavailable. Retrying on CPU. {reason}", flush=True)
        if on_fallback is not None:
            on_fallback(reason)
        return attempt("cpu"), "cpu", reason


def ordered_matches(
    database: Path, photos: list[Photo]
) -> dict[tuple[int, int], tuple[Coordinates, Coordinates]]:
    """Read verified landmarks for neighboring and skip-one photos.

    Args:
        database: COLMAP database after exhaustive matching.
        photos: Selected photos in analyst-specified order.

    Returns:
        Matched original pixel centers, indexed by pairs in selection order.
    """
    pairs: list[tuple[int, int]] = [
        (first, second)
        for first in range(len(photos))
        for second in range(first + 1, min(first + 3, len(photos)))
    ]
    return indexed_matches(database, photos, pairs)


def indexed_matches(
    database: Path, photos: list[Photo], pairs: list[tuple[int, int]]
) -> dict[tuple[int, int], tuple[Coordinates, Coordinates]]:
    """Read verified landmarks in the requested photo order and native pixels.

    Args:
        database: COLMAP database containing geometric verification results.
        photos: Source photos in the order used by the requested pair indices.
        pairs: Ordered indices for each requested photo pair.

    Returns:
        Paired native coordinates for each pair with usable verified landmarks.

    Raises:
        ValueError: If a stored feature or match record is invalid.
    """
    uri: str = f"{database.resolve().as_uri()}?mode=ro"
    result: dict[tuple[int, int], tuple[Coordinates, Coordinates]] = {}
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        ids: dict[str, int] = dict(
            connection.execute("SELECT name, image_id FROM images")
        )
        keypoints: dict[int, Coordinates] = {}
        for image_id, rows, cols, raw in connection.execute(
            "SELECT image_id, rows, cols, data FROM keypoints WHERE rows > 0"
        ):
            if cols < 2 or raw is None or len(raw) != rows * cols * 4:
                raise ValueError(f"Invalid COLMAP keypoints for image {image_id}.")
            xy: Coordinates = np.frombuffer(raw, dtype="<f4").reshape(rows, cols)[:, :2]
            # COLMAP stores the center of the upper-left pixel at (0.5, 0.5).
            keypoints[image_id] = xy.astype(np.float64) - 0.5
        for first, second in pairs:
            first_id: int | None = ids.get(photos[first].staged_name)
            second_id: int | None = ids.get(photos[second].staged_name)
            if first_id not in keypoints or second_id not in keypoints:
                continue
            assert first_id is not None and second_id is not None
            low: int
            high: int
            # The pair ID sorts image IDs. The returned points follow the request.
            low, high = sorted((first_id, second_id))
            row: tuple[int, int, bytes | None] | None = connection.execute(
                "SELECT rows, cols, data FROM two_view_geometries "
                "WHERE pair_id = ? AND rows > 0",
                (low * PAIR_ID_BASE + high,),
            ).fetchone()
            if row is None:
                continue
            rows: int
            cols: int
            raw: bytes | None
            rows, cols, raw = row
            if cols != 2 or raw is None or len(raw) != rows * 8:
                raise ValueError(f"Invalid COLMAP matches for images {low} and {high}.")
            indexes: NDArray[np.uint32] = np.frombuffer(raw, dtype="<u4").reshape(
                rows, 2
            )
            if np.any(
                indexes.max(axis=0) >= [len(keypoints[low]), len(keypoints[high])]
            ):
                raise ValueError(
                    f"COLMAP match refers to a missing feature in {low} or {high}."
                )
            a: Coordinates = keypoints[low][indexes[:, 0]]
            b: Coordinates = keypoints[high][indexes[:, 1]]
            valid: NDArray[np.bool_] = np.isfinite(a).all(axis=1) & np.isfinite(b).all(
                axis=1
            )
            result[first, second] = (
                (a[valid], b[valid]) if first_id == low else (b[valid], a[valid])
            )
    return result

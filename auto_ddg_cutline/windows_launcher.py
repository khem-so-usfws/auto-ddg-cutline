"""Run the cutline companion through a small Windows desktop launcher."""

from __future__ import annotations

import _thread
import json
import os
import subprocess
import sys
import tempfile
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Callable, Literal, TypeAlias

import cv2

from . import __version__
from .__main__ import list_images
from .__main__ import main as cli_main
from .launcher_workflow import (
    MANIFEST_PREFIX,
    SavedSelection,
    SelectionPlanRelocationNeeded,
    export_review,
    load_saved_selection,
    read_json,
    read_tier2_proposal,
    refine_review,
    save_tier2_selection,
    test_installation,
)
from .match_cache import legacy_checkpoint, match_assets
from .selection import DEFAULT_TARGET_PHOTO_PERCENT
from .survey_overview import build_survey_overview

DeviceChoice: TypeAlias = Literal["cpu", "auto", "gpu"]
DEVICES: dict[str, DeviceChoice] = {"Auto": "auto", "NVIDIA GPU": "gpu", "CPU": "cpu"}
LAUNCHER_EXE: str = "AutoDDGCutline.exe"
CLI_EXE: str = "AutoDDGCutlineCLI.exe"


def launcher_title(bundle: Path | None = None) -> str:
    """Display the packaged release or source version in the title bar.

    Args:
        bundle: Directory containing release metadata, if running from a build.

    Returns:
        The title to display in the launcher.
    """
    if bundle is not None:
        try:
            version: Any = read_json(bundle / "release.json").get("version")
            if isinstance(version, str) and version.strip():
                return f"Auto-DDG Cutline {version.strip()}"
        except (OSError, ValueError):
            pass
    return f"Auto-DDG Cutline {__version__}"


@dataclass(frozen=True)
class AnalysisOptions:
    """Keep the GUI choices independent of the worker process and its arguments."""

    folder: Path
    work_dir: Path
    colmap: Path | None
    selected: list[int] | None = None
    device: DeviceChoice = "auto"
    reuse: Path | None = None
    local_sfm: bool = True
    feature_pixel_percent: float | None = None
    cpu_threads: int = 1
    verbose: bool = False
    profile: bool = False
    rules: Path | None = None
    diagnostics: bool = False
    compare_shared_patches: bool = False
    target_photo_percent: int | None = None


def job_arguments(options: AnalysisOptions) -> list[str]:
    """Translate the launcher choices into analysis worker arguments.

    Args:
        options: Photo folder, matching settings, and analysis mode.

    Returns:
        Arguments for the worker, excluding the executable and engine flag.

    Raises:
        ValueError: If the run needs COLMAP and its executable is unavailable,
            or a fixed counting selection contains fewer than two photos.
    """
    task: str = "sfm-diagnostics" if options.diagnostics else "selection-plan"
    args: list[str] = [
        str(options.folder),
        "--all",
        "--output",
        task,
        "--work-dir",
        str(options.work_dir),
    ]
    local_sfm: bool = options.local_sfm and not options.diagnostics
    compare_patches: bool = options.compare_shared_patches and not options.diagnostics
    if options.reuse is None or local_sfm or options.diagnostics or compare_patches:
        if options.colmap is None or not options.colmap.is_file():
            raise ValueError("Choose your COLMAP.bat before running COLMAP.")
        args.extend(["--colmap-path", str(options.colmap)])
    if options.reuse is None:
        args.extend(
            ["--device", options.device, "--cpu-threads", str(options.cpu_threads)]
        )
        if options.feature_pixel_percent is not None:
            args.extend(["--feature-pixel-percent", str(options.feature_pixel_percent)])
    else:
        # The checkpoint determines matching settings for this run.
        args.extend(["--reuse-matches", str(options.reuse)])
    if local_sfm:
        args.append("--run-local-sfm")
    if compare_patches:
        args.append("--compare-shared-patches")
    if options.selected is not None and not options.diagnostics:
        if len(options.selected) < 2:
            raise ValueError("Select at least two photos to count.")
        args.extend(
            [
                "--count-selection",
                ",".join(str(index + 1) for index in options.selected),
            ]
        )
    if options.rules is not None and not options.diagnostics:
        args.extend(["--selection-rules", str(options.rules)])
    if (
        options.target_photo_percent is not None
        and options.selected is None
        and not options.diagnostics
    ):
        args.extend(["--target-photo-percent", str(options.target_photo_percent)])
    if options.verbose:
        args.append("--verbose")
    if options.profile:
        args.append("--profile-resources")
    return args


def matching_manifest(path: Path, photos: list[Path]) -> bool:
    """Check whether to offer a saved run before full worker validation.

    Args:
        path: Candidate COLMAP run manifest.
        photos: Photos in the order used by the current analysis.

    Returns:
        Whether the manifest lists the same photos and has matching assets.
    """
    try:
        source: dict[str, Any] = read_json(path)
        if source.get("backend") != "colmap" or [
            row["name"] for row in source["images"]
        ] != [photo.name for photo in photos]:
            return False
        checkpoint: dict[str, Any] = source.get(
            "match_checkpoint"
        ) or legacy_checkpoint(source, path)
        return (
            Path(checkpoint["database"]).is_file()
            and Path(checkpoint["image_folder"]).is_dir()
        ) or (
            (path.parent / checkpoint["device"] / "database.db").is_file()
            and (path.parent / "images").is_dir()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def saved_match_summary(path: Path) -> str:
    """Describe extraction from saved matching metadata.

    Args:
        path: The run manifest selected for matching reuse.

    Returns:
        The saved pixel percentage, the recorded long-edge limit, or an
        unavailable status if neither was recorded.
    """
    prefix: str = "Using saved matches · "
    try:
        source: dict[str, Any] = read_json(path)
        checkpoint: dict[str, Any] = source.get(
            "match_checkpoint"
        ) or legacy_checkpoint(source, path)
        settings: dict[str, Any] = checkpoint["settings"]
        percent: Any = settings.get("feature_pixel_percent")
        if type(percent) in (int, float) and 0 < percent <= 100:
            return f"{prefix}extraction at {percent:g}% of original pixels"
        edge: Any = settings.get("effective_feature_max_edge")
        if type(edge) is int and edge > 0:
            return f"{prefix}extraction limit {edge} px long edge"
    except (AttributeError, OSError, ValueError, KeyError, TypeError):
        pass
    return f"{prefix}extraction size unavailable"


def cpu_worker_count(value: str) -> int:
    """Parse the requested positive number of CPU extraction workers.

    Args:
        value: The launcher's CPU extraction worker field.

    Returns:
        The number of workers to pass to COLMAP.

    Raises:
        ValueError: If the field is empty or is not a positive integer.
    """
    message: str = "CPU extraction workers must be a positive integer."
    try:
        workers: int = int(value)
    except ValueError as error:
        raise ValueError(message) from error
    if workers < 1:
        raise ValueError(message)
    return workers


def engine_command() -> list[str]:
    """Use the console worker from a release or the active Python in development."""
    if getattr(sys, "frozen", False):
        worker: Path = Path(sys.executable).resolve().with_name(CLI_EXE)
        return [str(worker), "--engine"]
    return [sys.executable, "-m", "auto_ddg_cutline.windows_launcher", "--engine"]


def run_worker() -> int:
    """Translate a launcher cancellation request into the CLI interrupt path."""

    def listen() -> None:
        """Wait for one request from the launcher's input pipe."""
        # A raw pipe read does not hold Python's buffered stdin lock at shutdown.
        try:
            request: bytes = os.read(sys.stdin.fileno(), 1)
        except (AttributeError, OSError, ValueError):
            return
        if request == b"!":
            _thread.interrupt_main()

    # The launcher always passes a selection, so stdin is available for cancellation.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        Thread(target=listen, daemon=True).start()
        if sys.argv[1:2] in (["--test-colmap"], ["--test-gpu"]):
            flag: str
            path: str
            folder: str
            flag, path, folder = sys.argv[1:]
            jobs: Path = Path(folder)
            jobs.mkdir(parents=True, exist_ok=True)
            job: Path = Path(tempfile.mkdtemp(prefix="colmap-check-", dir=jobs))
            return test_installation(path, job, gpu=flag == "--test-gpu")
        if sys.argv[1:2] == ["--export-review"]:
            manifest: Path
            folder, manifest = map(Path, sys.argv[2:])
            return export_review(folder, manifest)
        if sys.argv[1:2] == ["--survey-overview"]:
            return build_survey_overview(Path(sys.argv[2]))
        if sys.argv[1:2] == ["--confirm-tier2"]:
            folder_arg: str
            plan_arg: str
            digest_arg: str
            indices_arg: str
            folder_arg, plan_arg, digest_arg, indices_arg = sys.argv[2:]
            indices: list[int] = [int(index) for index in indices_arg.split(",")]
            saved: Path = save_tier2_selection(
                Path(folder_arg), Path(plan_arg), digest_arg, indices
            )
            print(f"Saved {len(indices)} photos for Tier 2: {saved}", flush=True)
            return 0
        if sys.argv[1:2] == ["--refine-review"]:
            digest: str
            colmap: str
            device: str
            threads: str
            flags: list[str]
            (
                folder_arg,
                plan_arg,
                digest,
                indices_arg,
                colmap,
                device,
                threads,
                jobs,
                *flags,
            ) = sys.argv[2:]
            return refine_review(
                Path(folder_arg),
                Path(plan_arg),
                digest,
                [int(index) for index in indices_arg.split(",")],
                colmap,
                DEVICES[device],
                Path(jobs),
                cpu_threads=int(threads),
                verbose="--verbose" in flags,
                profile_resources="--profile-resources" in flags,
                protect_groups="--protect-groups" in flags,
            )
        return cli_main()
    except (OSError, ValueError, KeyError, IndexError, TypeError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


class DenseReferenceDialog(simpledialog.Dialog):
    """Choose the photo that defines the dense terrain section."""

    def __init__(self, parent: tk.Misc, photos: list[str], default: str) -> None:
        """Set up the choices before opening the modal dialog."""
        self.photos: list[str] = photos
        self.default: str = default
        self.result: str | None = None
        super().__init__(parent, title="Build dense terrain")

    def body(self, master: tk.Misc) -> tk.Widget:
        """Render the reference selector and describe the scope of the pilot."""
        ttk.Label(
            master,
            text=(
                "Choose a reference photo for one section of the survey. "
                "COLMAP will use nearby registered views. This step requires CUDA."
            ),
            wraplength=440,
        ).pack(anchor="w", pady=(0, 8))
        self.reference: ttk.Combobox = ttk.Combobox(
            master, values=self.photos, width=58, state="readonly"
        )
        self.reference.set(self.default)
        self.reference.pack(fill="x")
        return self.reference

    def apply(self) -> None:
        """Return the reference chosen by the analyst."""
        self.result = self.reference.get()


class Launcher:
    """Select a counting workflow and follow the worker through review and export."""

    def __init__(self, root: tk.Tk) -> None:
        """Build the launcher and load the last COLMAP location."""
        self.root: tk.Tk = root
        self.photos: list[Path] = []
        self.recommended: set[int] = set()
        self.photo_checks: list[tk.BooleanVar] = []
        self.photo_rows: list[tuple[ttk.Checkbutton, ttk.Label, ttk.Label]] = []
        self.folder: Path | None = None
        self.manifest: Path | None = None
        self.log_folder: Path | None = None
        self.loaded_selection: SavedSelection | None = None
        self.refined_review: Path | None = None
        self.can_export_refined: bool = False
        self.process: subprocess.Popen[str] | None = None
        self.messages: Queue[str | int] = Queue()
        self.operation: str = ""
        self.closing: bool = False
        self.app_dir: Path = (
            Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AutoDDGCutline"
        )
        self.settings_path: Path = self.app_dir / "launcher.json"
        try:
            settings: dict[str, Any] = read_json(self.settings_path)
        except (OSError, ValueError):
            settings = {}
        self.colmap_path: Path | None = (
            Path(settings["colmap"])
            if isinstance(settings.get("colmap"), str)
            else None
        )
        self.job_dir: tk.StringVar = tk.StringVar(
            value=str(settings.get("jobs") or self.app_dir / "jobs")
        )
        self.folder_text: tk.StringVar = tk.StringVar(
            value="Choose a folder of survey photos"
        )
        self.colmap_text: tk.StringVar = tk.StringVar(
            value=str(self.colmap_path or "Choose your COLMAP installation")
        )
        self.photo_search: tk.StringVar = tk.StringVar()
        self.checked_only: tk.BooleanVar = tk.BooleanVar(value=False)
        self.selection_count: tk.StringVar = tk.StringVar()
        self.status: tk.StringVar = tk.StringVar(value="Ready")
        self.reuse: tk.StringVar = tk.StringVar()
        self.local_sfm: tk.BooleanVar = tk.BooleanVar(value=False)
        self.compare_shared_patches: tk.BooleanVar = tk.BooleanVar(value=False)
        self.verbose: tk.BooleanVar = tk.BooleanVar(value=False)
        self.profile: tk.BooleanVar = tk.BooleanVar(value=False)
        self.rules: tk.StringVar = tk.StringVar()
        self.pixel_percent: tk.StringVar = tk.StringVar(value="50%")
        self.target_photo_percent: tk.StringVar = tk.StringVar(
            value=str(DEFAULT_TARGET_PHOTO_PERCENT)
        )
        self.threads: tk.StringVar = tk.StringVar(value="1")
        self.advanced: tk.BooleanVar = tk.BooleanVar(value=False)
        self.protect_groups: tk.BooleanVar = tk.BooleanVar(value=False)

        bundle: Path | None = (
            Path(sys.executable).resolve().parent
            if getattr(sys, "frozen", False)
            else None
        )
        root.title(launcher_title(bundle))
        root.minsize(785, 565)
        root.protocol("WM_DELETE_WINDOW", self.close)
        outer: ttk.Frame = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)
        self.editors: list[tk.Widget] = []

        self.file_row(
            outer, 0, "Choose photo folder…", self.folder_text, self.load_folder
        )
        colmap_row: ttk.Frame = self.file_row(
            outer, 1, "Choose COLMAP.bat…", self.colmap_text, self.load_colmap
        )
        self.test_colmap: ttk.Button = ttk.Button(
            colmap_row,
            text="Test COLMAP",
            command=lambda: self.test_installation(False),
        )
        self.test_gpu: ttk.Button = ttk.Button(
            colmap_row, text="Test GPU", command=lambda: self.test_installation(True)
        )
        for button in (self.test_gpu, self.test_colmap):
            button.pack(side="right", padx=(6, 0))

        choices: ttk.Frame = ttk.Frame(outer)
        choices.grid(row=2, column=0, sticky="ew", pady=(9, 4))
        ttk.Label(choices, text="Matching device").pack(side="left")
        self.device: ttk.Combobox = ttk.Combobox(
            choices, values=tuple(DEVICES), width=12, state="readonly"
        )
        self.device.set("Auto")
        self.device.pack(side="left", padx=6)
        self.match_source: ttk.Label = ttk.Label(choices)
        self.match_source.pack(side="left", padx=3)
        # Read the manifest only when the saved match choice changes.
        self.reuse.trace_add("write", self.update_match_source)

        self.build_photo_table(outer)

        advanced_toggle: ttk.Checkbutton = ttk.Checkbutton(
            outer, text="Advanced", variable=self.advanced, command=self.toggle_advanced
        )
        advanced_toggle.grid(row=4, column=0, sticky="w", pady=(5, 2))
        self.advanced_panel: ttk.Frame = ttk.Frame(outer, padding=(12, 4))
        self.advanced_panel.grid(row=5, column=0, sticky="ew")
        self.advanced_panel.columnconfigure(0, weight=1)
        self.build_advanced()
        self.toggle_advanced()

        ttk.Label(outer, textvariable=self.status, wraplength=680).grid(
            row=6, column=0, sticky="w", pady=5
        )
        self.output: ScrolledText = ScrolledText(
            outer, height=6, width=80, wrap="word", state="disabled"
        )
        self.output.grid(row=7, column=0, sticky="nsew")
        actions: ttk.Frame = ttk.Frame(outer)
        actions.grid(row=8, column=0, sticky="ew", pady=(10, 0))
        self.run_button: ttk.Button = ttk.Button(
            actions, text="Analyze photos", command=self.start
        )
        self.cancel_button: ttk.Button = ttk.Button(
            actions, text="Cancel", command=self.cancel
        )
        self.open_review: ttk.Button = ttk.Button(
            actions, text="Review drafts", command=self.show_review
        )
        self.export_button: ttk.Button = ttk.Button(
            actions, text="Export drafts to DDG…", command=self.export
        )
        for button in (self.run_button, self.open_review, self.export_button):
            button.pack(side="left", padx=(0, 6))
        self.open_logs: ttk.Button = ttk.Button(
            actions, text="Open run logs", command=self.show_logs
        )
        self.open_logs.pack(side="right")
        self.cancel_button.pack(side="right", padx=6)

        refinement: ttk.Frame = ttk.Frame(outer)
        refinement.grid(row=9, column=0, sticky="ew", pady=(6, 0))
        self.refine_button: ttk.Button = ttk.Button(
            refinement, text="Refine cutlines", command=self.refine
        )
        self.open_refined: ttk.Button = ttk.Button(
            refinement, text="Review refined cutlines", command=self.show_refined
        )
        self.export_refined: ttk.Button = ttk.Button(
            refinement,
            text="Export refined cutlines…",
            command=self.export_refinement,
        )
        for button in (self.refine_button, self.open_refined, self.export_refined):
            button.pack(side="left", padx=(0, 6))
        ttk.Label(refinement, textvariable=self.selection_count).pack(side="right")
        options_row: ttk.Frame = ttk.Frame(outer)
        options_row.grid(row=10, column=0, sticky="ew", pady=(5, 0))
        self.protection_option: ttk.Checkbutton = ttk.Checkbutton(
            options_row,
            text="Avoid DDG polygons labeled protect_group",
            variable=self.protect_groups,
        )
        self.protection_option.pack(side="left")
        self.overview_button: ttk.Button = ttk.Button(
            options_row, text="Update Survey Overview", command=self.start_overview
        )
        self.overview_button.pack(side="right")
        self.dense_overview_button: ttk.Button = ttk.Button(
            options_row, text="Build dense terrain…", command=self.start_dense_overview
        )
        self.dense_overview_button.pack(side="right", padx=(0, 6))
        self.editors.append(self.protection_option)
        self.update_controls()

    def file_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        value: tk.StringVar,
        action: Callable[[], None],
    ) -> ttk.Frame:
        """Render a path picker with a read-only, scrollable path field."""
        line: ttk.Frame = ttk.Frame(parent)
        line.grid(row=row, column=0, sticky="ew", pady=3)
        button: ttk.Button = ttk.Button(line, text=label, command=action)
        button.pack(side="left")
        path: ttk.Entry = ttk.Entry(line, textvariable=value, state="readonly")
        path.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.editors.append(button)
        return line

    def build_advanced(self) -> None:
        """Expose repeatable matching and reconstruction settings."""
        self.file_row(
            self.advanced_panel, 0, "Reuse matches…", self.reuse, self.load_matches
        )
        self.file_row(
            self.advanced_panel,
            1,
            "Load pin/exclude rules…",
            self.rules,
            self.load_rules,
        )
        self.file_row(
            self.advanced_panel, 2, "Run output folder…", self.job_dir, self.pick_jobs
        )
        flags: ttk.Frame = ttk.Frame(self.advanced_panel)
        flags.grid(row=3, column=0, sticky="ew", pady=3)
        for label, variable in (
            ("Run local SfM", self.local_sfm),
            ("Compare shared 3D patches", self.compare_shared_patches),
            ("Detailed progress", self.verbose),
            ("Profile memory", self.profile),
        ):
            button: ttk.Checkbutton = ttk.Checkbutton(
                flags, text=label, variable=variable, command=self.update_controls
            )
            button.pack(side="left", padx=(0, 8))
            self.editors.append(button)
        performance: ttk.Frame = ttk.Frame(self.advanced_panel)
        performance.grid(row=4, column=0, sticky="ew", pady=3)
        ttk.Label(performance, text="Fresh extraction (% of original pixels)").pack(
            side="left"
        )
        self.pixel_percent_box: ttk.Combobox = ttk.Combobox(
            performance,
            textvariable=self.pixel_percent,
            values=("100%", "53.2%", "50%", "30%", "20%", "13.3%"),
            width=9,
        )
        self.pixel_percent_box.pack(side="left", padx=6)
        ttk.Label(performance, text="CPU extraction workers").pack(side="left")
        self.workers: ttk.Entry = ttk.Entry(
            performance, textvariable=self.threads, width=5
        )
        self.workers.pack(side="left", padx=6)
        self.diagnostics: ttk.Button = ttk.Button(
            performance,
            text="Check reconstruction",
            command=lambda: self.start(diagnostics=True),
        )
        self.diagnostics.pack(side="right")
        selection: ttk.Frame = ttk.Frame(self.advanced_panel)
        selection.grid(row=5, column=0, sticky="ew", pady=3)
        ttk.Label(selection, text="Counting photo target (%)").pack(side="left")
        target: ttk.Entry = ttk.Entry(
            selection, textvariable=self.target_photo_percent, width=5
        )
        target.pack(side="left", padx=6)
        self.editors.append(target)
        ttk.Label(selection, text="Clear for full documented coverage").pack(
            side="left"
        )

    def toggle_advanced(self) -> None:
        """Keep diagnostics and tuning out of the main workflow."""
        if self.advanced.get():
            self.advanced_panel.grid()
        else:
            self.advanced_panel.grid_remove()

    def save_settings(self) -> None:
        """Remember the selected COLMAP and job folder across sessions."""
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=self.app_dir, delete=False, mode="w", encoding="utf-8"
        ) as stream:
            temp: Path = Path(stream.name)
            json.dump(
                {
                    "colmap": str(self.colmap_path) if self.colmap_path else None,
                    "jobs": self.job_dir.get(),
                },
                stream,
            )
        try:
            temp.replace(self.settings_path)
        finally:
            temp.unlink(missing_ok=True)

    def load_colmap(self) -> None:
        """Choose an external COLMAP installation."""
        choice: str = filedialog.askopenfilename(
            parent=self.root,
            title="Choose COLMAP.bat",
            initialdir=self.colmap_path.parent if self.colmap_path else None,
            filetypes=[
                ("COLMAP launcher or executable", "*.bat *.exe *.cmd"),
                ("All files", "*.*"),
            ],
        )
        if choice:
            self.colmap_path = Path(choice).resolve()
            self.colmap_text.set(str(self.colmap_path))
            self.remember()

    def pick_jobs(self) -> None:
        """Choose the directory for logs, reviews and saved match databases."""
        folder: str = filedialog.askdirectory(
            parent=self.root, initialdir=self.job_dir.get()
        )
        if folder:
            self.job_dir.set(folder)
            self.remember()

    def remember(self) -> None:
        """Apply launcher preferences and report any settings write failure."""
        try:
            self.save_settings()
            self.status.set("Settings saved")
        except OSError as error:
            self.status.set(f"Could not save settings. {error}")
        self.update_controls()

    def load_folder(self) -> None:
        """Show available source photos and offer a recent matching checkpoint."""
        choice: str = filedialog.askdirectory(parent=self.root, initialdir=self.folder)
        if not choice:
            return
        folder: Path = Path(choice).resolve()
        try:
            photos: list[Path] = list_images(folder)
        except OSError as error:
            self.fail(error)
            return
        self.folder = folder
        self.photos = photos
        self.recommended.clear()
        self.loaded_selection = None
        self.refined_review = None
        self.can_export_refined = False
        self.log_folder = None
        self.folder_text.set(str(folder))
        self.reuse.set("")
        self.rules.set("")
        self.fill_photo_table()
        self.checked_only.set(False)
        self.photo_search.set("")
        self.status.set(f"{len(photos)} photos available")
        for path in sorted(
            Path(self.job_dir.get()).glob("auto-ddg-colmap-*/run.json"), reverse=True
        ):
            if matching_manifest(path, photos):
                if messagebox.askyesno(
                    "Saved matches found",
                    f"Reuse saved matches for these photos?\n\n{path}",
                    parent=self.root,
                ):
                    self.reuse.set(str(path))
                    if self.colmap_path is None:
                        self.local_sfm.set(False)
                break
        self.update_controls()

    def load_matches(self) -> None:
        """Choose saved matches or clear the current choice."""
        if self.reuse.get() and messagebox.askyesno(
            "Find new overlaps", "Clear the saved match choice?", parent=self.root
        ):
            self.reuse.set("")
            self.update_controls()
            return
        path: str = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a saved run.json",
            filetypes=[("COLMAP run", "*.json")],
        )
        if path:
            if not matching_manifest(Path(path), self.photos):
                self.fail(
                    ValueError(
                        "Choose a saved run with the same photos in the same folder order."
                    )
                )
                return
            self.reuse.set(path)
            if self.colmap_path is None:
                self.local_sfm.set(False)
            self.update_controls()

    def load_previous_selection(self) -> None:
        """Load checkboxes and any available review from a saved recommendation."""
        if not self.photos or self.process is not None:
            return
        choice: str = filedialog.askopenfilename(
            parent=self.root,
            title="Choose run.json, selection-plan.json, or tier2-selection.json",
            initialdir=self.log_folder or self.job_dir.get(),
            filetypes=[("Saved selection", "*.json")],
        )
        if not choice:
            return
        try:
            selection: SavedSelection
            try:
                selection = load_saved_selection(Path(choice), self.photos)
            except SelectionPlanRelocationNeeded as error:
                original: str = filedialog.askopenfilename(
                    parent=self.root,
                    title=str(error),
                    initialdir=Path(choice).parent,
                    filetypes=[("Original analysis", "*.json")],
                )
                if not original:
                    return
                selection = load_saved_selection(
                    Path(choice), self.photos, original_plan=Path(original)
                )
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.fail(error)
            return
        if (
            self.loaded_selection is not None
            and set(self.checked_photos()) != self.recommended
            and not messagebox.askyesno(
                "Replace current checks?",
                "Replace the current checks with the saved selection?",
                parent=self.root,
            )
        ):
            return
        self.apply_selection(selection)
        self.log_folder = selection.run.parent if selection.run is not None else None
        status: str
        if selection.restored_checks is None:
            status = f"Loaded {len(self.recommended)} recommendations from {selection.plan.name}."
        else:
            status = f"Restored {len(selection.restored_checks)} checked photos from tier2-selection.json."
            if selection.refined_report is not None:
                status += " Refined review available."
        if any(index is None for index in selection.plan_indices):
            status += " Reanalyze before refining photos added since that run."
        self.status.set(status)
        self.update_controls()

    def apply_selection(self, selection: SavedSelection) -> None:
        """Apply the restored checks and retain the original recommendation."""
        self.loaded_selection = selection
        self.refined_review = selection.refined_report
        self.can_export_refined = selection.can_export_refined
        self.recommended = set(selection.recommended)
        for index, (_, _, marker) in enumerate(self.photo_rows):
            marker.configure(text="Recommended" if index in self.recommended else "")
        checks: frozenset[int] = (
            selection.recommended
            if selection.restored_checks is None
            else selection.restored_checks
        )
        for index, checked in enumerate(self.photo_checks):
            checked.set(index in checks)
        self.selection_changed()

    def load_rules(self) -> None:
        """Choose or clear a JSON file with photo inclusion rules."""
        if self.rules.get() and messagebox.askyesno(
            "Clear selection rules", "Clear the current rules file?", parent=self.root
        ):
            self.rules.set("")
            return
        path: str = filedialog.askopenfilename(
            parent=self.root,
            title="Choose selection rules",
            filetypes=[("Selection rules", "*.json")],
        )
        if path:
            self.rules.set(path)

    def build_photo_table(self, parent: ttk.Frame) -> None:
        """Build the filter controls and the scrollable table of source photos."""
        panel: ttk.Frame = ttk.Frame(parent)
        panel.grid(row=3, column=0, sticky="nsew", pady=(4, 0))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        filters: ttk.Frame = ttk.Frame(panel)
        filters.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        ttk.Label(filters, text="Find photo").pack(side="left")
        search: ttk.Entry = ttk.Entry(filters, textvariable=self.photo_search, width=22)
        search.pack(side="left", padx=6)
        checked: ttk.Checkbutton = ttk.Checkbutton(
            filters,
            text="Checked only",
            variable=self.checked_only,
            command=self.filter_photos,
        )
        checked.pack(side="left", padx=4)
        self.restore_button: ttk.Button = ttk.Button(
            filters,
            text="Reset to recommended photos",
            command=self.restore_recommendations,
        )
        self.restore_button.pack(side="right")
        self.previous_selection_button: ttk.Button = ttk.Button(
            filters,
            text="Load previous selection…",
            command=self.load_previous_selection,
        )
        self.previous_selection_button.pack(side="right", padx=(0, 6))
        self.editors.extend((search, checked))

        self.photo_canvas: tk.Canvas = tk.Canvas(
            panel, height=235, highlightthickness=0, yscrollincrement=18
        )
        scrollbar: ttk.Scrollbar = ttk.Scrollbar(panel, command=self.photo_canvas.yview)
        self.photo_canvas.configure(yscrollcommand=scrollbar.set)
        self.photo_canvas.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.photo_body: ttk.Frame = ttk.Frame(self.photo_canvas)
        self.photo_body.columnconfigure(1, weight=1)
        window: int = self.photo_canvas.create_window(
            (0, 0), window=self.photo_body, anchor="nw"
        )

        def fit_rows(event: tk.Event) -> None:
            """Keep the row columns aligned to the visible canvas."""
            self.photo_canvas.itemconfigure(window, width=event.width)

        def update_scroll_region(event: tk.Event) -> None:
            """Refresh scrolling when filtering or sizing changes the rows."""
            self.photo_canvas.configure(scrollregion=self.photo_canvas.bbox(window))

        self.photo_canvas.bind("<Configure>", fit_rows)
        self.photo_body.bind("<Configure>", update_scroll_region)
        self.photo_canvas.bind("<MouseWheel>", self.scroll_photos)
        self.photo_body.bind("<MouseWheel>", self.scroll_photos)
        for column, heading in enumerate(
            ("Use for refinement", "Photo", "Recommendation")
        ):
            ttk.Label(self.photo_body, text=heading).grid(
                row=0, column=column, sticky="w", padx=6, pady=(3, 5)
            )
        self.photo_search.trace_add("write", self.filter_photos)

    def scroll_photos(self, event: tk.Event) -> Literal["break"]:
        """Scroll the table when the cursor is over a photo row."""
        self.photo_canvas.yview_scroll(-int(event.delta / 120), "units")
        return "break"

    def reveal_photo(self, event: tk.Event) -> None:
        """Scroll a keyboard-focused checkbox into view."""
        top: int = event.widget.winfo_y()
        bottom: int = top + event.widget.winfo_height()
        first: float = self.photo_canvas.canvasy(0)
        height: int = self.photo_canvas.winfo_height()
        total: int = self.photo_body.winfo_height()
        if total and top < first:
            self.photo_canvas.yview_moveto(top / total)
        elif total and bottom > first + height:
            self.photo_canvas.yview_moveto((bottom - height) / total)

    def fill_photo_table(self) -> None:
        """Populate a new folder with an unchecked row for every source photo."""
        for widgets in self.photo_rows:
            for widget in widgets:
                widget.destroy()
        self.photo_rows.clear()
        self.photo_checks.clear()
        for index, photo in enumerate(self.photos, start=1):
            selected: tk.BooleanVar = tk.BooleanVar(master=self.root, value=False)
            check: ttk.Checkbutton = ttk.Checkbutton(
                self.photo_body, variable=selected, command=self.selection_changed
            )
            name: ttk.Label = ttk.Label(self.photo_body, text=photo.name)
            marker: ttk.Label = ttk.Label(self.photo_body, width=17)
            widgets: tuple[ttk.Checkbutton, ttk.Label, ttk.Label] = (
                check,
                name,
                marker,
            )
            for column, widget in enumerate(widgets):
                widget.grid(row=index, column=column, sticky="w", padx=6, pady=2)
                widget.bind("<MouseWheel>", self.scroll_photos)
            check.bind("<FocusIn>", self.reveal_photo)
            self.photo_checks.append(selected)
            self.photo_rows.append(widgets)
        self.filter_photos()
        self.photo_canvas.yview_moveto(0)

    def filter_photos(self, *trace_args: str) -> None:
        """Hide filtered rows while keeping their selection state."""
        query: str = self.photo_search.get().strip().casefold()
        checked_only: bool = self.checked_only.get()
        for index, photo in enumerate(self.photos):
            visible: bool = query in photo.name.casefold() and (
                self.photo_checks[index].get() or not checked_only
            )
            for widget in self.photo_rows[index]:
                if visible:
                    widget.grid()
                else:
                    widget.grid_remove()

    def checked_photos(self) -> list[int]:
        """Return checked indices in the folder order recorded in the review."""
        return [
            index for index, checked in enumerate(self.photo_checks) if checked.get()
        ]

    def selection_changed(self) -> None:
        """Update the count and keep the checked-only filter current."""
        if self.checked_only.get():
            self.filter_photos()
        self.update_controls()

    def restore_recommendations(self) -> None:
        """Replace checked choices with the software's original proposal."""
        for index, checked in enumerate(self.photo_checks):
            checked.set(index in self.recommended)
        self.selection_changed()

    def update_match_source(self, *_: str) -> None:
        """Refresh the effective extraction setting when saved matches change."""
        choice: str = self.reuse.get()
        self.match_source.configure(
            text=saved_match_summary(Path(choice)) if choice else ""
        )

    def update_controls(self) -> None:
        """Enable controls according to the selected workflow and worker state."""
        running: bool = self.process is not None
        colmap_ready: bool = self.colmap_path is not None and self.colmap_path.is_file()
        reuse: bool = bool(self.reuse.get())
        for widget in self.editors:
            widget.configure(state="disabled" if running else "normal")
        self.device.configure(state="disabled" if running or reuse else "readonly")
        # The worker uses saved settings for reuse. Both fields can still be edited
        # for fresh analysis, and refinement also uses the CPU worker field.
        for widget in (self.pixel_percent_box, self.workers):
            widget.configure(state="disabled" if running else "normal")
        for button in (self.test_colmap, self.test_gpu):
            button.configure(
                state="normal" if not running and colmap_ready else "disabled"
            )
        enough: bool = len(self.photos) >= 2
        ready: bool = colmap_ready or (
            reuse and not (self.local_sfm.get() or self.compare_shared_patches.get())
        )
        self.run_button.configure(
            state="normal" if not running and ready and enough else "disabled"
        )
        self.diagnostics.configure(
            state=(
                "normal"
                if not running and colmap_ready and len(self.photos) >= 3
                else "disabled"
            )
        )
        self.cancel_button.configure(state="normal" if running else "disabled")
        selection: SavedSelection | None = self.loaded_selection
        self.previous_selection_button.configure(
            state="normal" if self.photos and not running else "disabled"
        )
        self.open_review.configure(
            state=(
                "normal"
                if selection is not None
                and selection.report is not None
                and not running
                else "disabled"
            )
        )
        active_review: bool = selection is not None and not running
        self.protection_option.configure(
            state="normal" if active_review else "disabled"
        )
        selected: list[int] = self.checked_photos()
        self.overview_button.configure(state="normal" if active_review else "disabled")
        self.dense_overview_button.configure(
            state="normal" if active_review and colmap_ready else "disabled"
        )
        self.selection_count.set(
            f"{len(selected)} of {len(self.photos)} photos selected"
        )
        self.refine_button.configure(
            state=(
                "normal"
                if active_review
                and colmap_ready
                and len(selected) >= 2
                and selection is not None
                and selection.refinement_indices(selected) is not None
                else "disabled"
            )
        )
        self.restore_button.configure(
            state=(
                "normal"
                if active_review and set(selected) != self.recommended
                else "disabled"
            )
        )
        for checkbox, _, _ in self.photo_rows:
            checkbox.configure(state="normal" if active_review else "disabled")
        self.open_refined.configure(
            state=(
                "normal"
                if self.refined_review is not None and not running
                else "disabled"
            )
        )
        self.export_refined.configure(
            state="normal" if self.can_export_refined and not running else "disabled"
        )
        self.export_button.configure(
            state=(
                "normal"
                if selection is not None and selection.can_export and not running
                else "disabled"
            )
        )
        self.open_logs.configure(
            state="normal" if self.log_folder is not None else "disabled"
        )

    def start(self, *, diagnostics: bool = False) -> None:
        """Analyze every photo in the folder and generate a counting proposal."""
        if self.folder is None or self.process is not None:
            return
        try:
            if list_images(self.folder) != self.photos:
                raise ValueError(
                    "The folder contents changed. Choose the photo folder again."
                )
            reuse: Path | None = Path(self.reuse.get()) if self.reuse.get() else None
            target_percent: int | None = None
            target_text: str = self.target_photo_percent.get().strip().removesuffix("%")
            if target_text and not diagnostics:
                try:
                    target_percent = int(target_text)
                except ValueError as error:
                    raise ValueError(
                        "Enter a whole counting photo percentage."
                    ) from error
                if not 1 <= target_percent <= 100:
                    raise ValueError("The counting photo target must be 1–100%.")
            percent: float | None = None
            if reuse is None:
                try:
                    percent = float(self.pixel_percent.get().strip().removesuffix("%"))
                except ValueError as error:
                    raise ValueError(
                        "Enter an extraction percentage such as 30%."
                    ) from error
                if not 0 < percent <= 100:
                    raise ValueError(
                        "Extraction percentage must be greater than 0 and at most 100."
                    )
            threads: int = 1 if reuse else cpu_worker_count(self.threads.get())
            args: list[str] = job_arguments(
                AnalysisOptions(
                    self.folder,
                    Path(self.job_dir.get()),
                    self.colmap_path,
                    device=DEVICES[self.device.get()],
                    reuse=reuse,
                    local_sfm=self.local_sfm.get(),
                    feature_pixel_percent=percent,
                    cpu_threads=threads,
                    verbose=self.verbose.get(),
                    profile=self.profile.get(),
                    rules=Path(self.rules.get()) if self.rules.get() else None,
                    diagnostics=diagnostics,
                    compare_shared_patches=self.compare_shared_patches.get(),
                    target_photo_percent=target_percent,
                )
            )
        except (OSError, ValueError) as error:
            self.fail(error)
            return
        if (
            not diagnostics
            and self.loaded_selection is not None
            and set(self.checked_photos()) != self.recommended
            and not messagebox.askyesno(
                "Run a fresh analysis?",
                "A completed analysis will replace your current checks with a new recommendation.",
                parent=self.root,
            )
        ):
            return
        self.launch(
            args, "Checking reconstruction" if diagnostics else "Analyzing photos"
        )

    def test_installation(self, gpu: bool) -> None:
        """Exercise the chosen external COLMAP through the packaged worker."""
        if self.colmap_path is not None:
            flag: str = "--test-gpu" if gpu else "--test-colmap"
            self.launch(
                [flag, str(self.colmap_path), self.job_dir.get()],
                "Testing GPU" if gpu else "Testing COLMAP",
            )

    def export(self) -> None:
        """Append the exact reviewed drafts to the counting photos' sidecars."""
        selection: SavedSelection | None = self.loaded_selection
        if (
            self.folder is not None
            and selection is not None
            and selection.run is not None
            and selection.can_export
        ):
            if messagebox.askyesno(
                "Export drafts to DDG",
                "Append the initial review's cutlines to the photo annotation files?",
                parent=self.root,
            ):
                self.launch(
                    [
                        "--export-review",
                        str(self.folder),
                        str(selection.run),
                    ],
                    "Exporting drafts",
                )

    def start_dense_overview(self) -> None:
        """Choose a reference and request a dense overview for its section."""
        if self.process or self.loaded_selection is None or not self.photos:
            return
        checked: list[int] = self.checked_photos()
        names: list[str] = [photo.name for photo in self.photos]
        default: str = names[checked[0]] if checked else names[0]
        dialog: DenseReferenceDialog = DenseReferenceDialog(self.root, names, default)
        if dialog.result:
            self.start_overview(dense_focus=dialog.result)

    def start_overview(self, *, dense_focus: str | None = None) -> None:
        """Open a fresh overview of the current checks using saved alignment."""
        selection: SavedSelection | None = self.loaded_selection
        checked: list[int] = self.checked_photos()
        if self.folder is None or selection is None or self.process:
            return
        checkpoint: Path | None = None
        reused: str = self.reuse.get()
        for candidate in (selection.run, Path(reused) if reused else None):
            if candidate is not None:
                try:
                    match_assets(read_json(candidate), candidate)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                checkpoint = candidate
                break
        if checkpoint is None:
            self.fail(
                ValueError(
                    "Choose the saved matches for this survey or run Analyze photos."
                )
            )
            return
        try:
            jobs: Path = Path(self.job_dir.get()).resolve()
            jobs.mkdir(parents=True, exist_ok=True)
            job: Path = Path(tempfile.mkdtemp(prefix="survey-overview-", dir=jobs))
            request: Path = job / "overview-input.json"
            refined: Path | None = (
                self.refined_review.with_suffix(".json")
                if self.refined_review is not None
                else None
            )
            data: dict[str, Any] = {
                "folder": str(self.folder),
                "plan": str(selection.plan),
                "digest": selection.digest,
                "checkpoint": str(checkpoint),
                "photos": [photo.name for photo in self.photos],
                "checked": [self.photos[index].name for index in checked],
                "colmap": str(self.colmap_path) if self.colmap_path else None,
                "refined": str(refined) if refined else None,
                "dense_focus": dense_focus,
            }
            request.write_text(json.dumps(data), encoding="utf-8")
        except OSError as error:
            self.fail(error)
            return
        operation: str = (
            "Building dense terrain" if dense_focus else "Building survey overview"
        )
        self.launch(["--survey-overview", str(request)], operation)

    def refine(self) -> None:
        """Run refinement with a snapshot of the checked original photos."""
        selection: SavedSelection | None = self.loaded_selection
        if (
            self.folder is None
            or selection is None
            or self.colmap_path is None
            or self.process is not None
        ):
            return
        indices: list[int] | None = selection.refinement_indices(self.checked_photos())
        if indices is None:
            self.fail(
                ValueError(
                    "Analyze photos again before refining a photo absent from the saved selection."
                )
            )
            return
        if len(indices) < 2:
            return
        try:
            threads: int = cpu_worker_count(self.threads.get())
        except ValueError as error:
            self.fail(error)
            return
        args: list[str] = [
            "--refine-review",
            str(self.folder),
            str(selection.plan),
            selection.digest,
            ",".join(str(index) for index in indices),
            str(self.colmap_path),
            self.device.get(),
            str(threads),
            self.job_dir.get(),
        ]
        if self.protect_groups.get():
            args.append("--protect-groups")
        if self.verbose.get():
            args.append("--verbose")
        if self.profile.get():
            args.append("--profile-resources")
        self.launch(args, "Refining cutlines")

    def export_refinement(self) -> None:
        """Append refined drafts to the selected DDG photo folder."""
        if self.refined_review is None or self.folder is None:
            return
        target: str = filedialog.askdirectory(
            parent=self.root,
            title="Choose the DDG photo folder",
            initialdir=self.folder,
        )
        if not target:
            return
        destination: Path = Path(target).resolve()
        self.launch(
            [
                "--export-review",
                str(destination),
                str(self.refined_review.with_name("run.json")),
            ],
            "Exporting drafts",
        )

    def launch(self, args: list[str], operation: str) -> bool:
        """Start a worker and display its output without blocking the window."""
        if self.process is not None:
            return False
        try:
            environment: dict[str, str] = {
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "PYINSTALLER_RESET_ENVIRONMENT": "1",
            }
            self.process = subprocess.Popen(
                [*engine_command(), *args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                env=environment,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                ),
            )
        except OSError as error:
            self.fail(error)
            return False
        self.operation = operation
        self.manifest = None
        if operation != "Exporting drafts":
            self.log_folder = None
        self.status.set(operation)
        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.configure(state="disabled")
        self.update_controls()
        Thread(target=self.read_output, args=(self.process,), daemon=True).start()
        self.root.after(80, self.poll)
        return True

    def read_output(self, process: subprocess.Popen[str]) -> None:
        """Drain the worker stream outside the UI thread."""
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                self.messages.put(line)
        self.messages.put(process.wait())

    def poll(self) -> None:
        """Display progress and restore the controls when the worker exits."""
        lines: list[str] = []
        code: int | None = None
        for _ in range(200):
            try:
                message: str | int = self.messages.get_nowait()
            except Empty:
                break
            if isinstance(message, int):
                code = message
                break
            lines.append(message)
            if message.startswith(MANIFEST_PREFIX):
                path: Path = Path(message.removeprefix(MANIFEST_PREFIX).strip())
                self.manifest = path if path.name == "run.json" else None
                self.log_folder = path.parent
                self.open_logs.configure(state="normal")
        if lines:
            self.output.configure(state="normal")
            self.output.insert(tk.END, "".join(lines))
            if int(self.output.index(tk.END).split(".")[0]) > 3000:
                self.output.delete("1.0", "1000.0")
            self.output.see(tk.END)
            self.output.configure(state="disabled")
        if code is None:
            self.root.after(80, self.poll)
            return
        if self.process is not None and self.process.stdin is not None:
            self.process.stdin.close()
        self.process = None
        if self.closing:
            self.root.destroy()
            return
        self.finish(code)
        self.update_controls()

    def finish(self, code: int) -> None:
        """Show the outcome, effective device and the review for this run."""
        status: str = {
            "Analyzing photos": "Analysis complete",
            "Checking reconstruction": "Reconstruction check complete",
            "Testing COLMAP": "COLMAP test passed",
            "Testing GPU": "GPU test passed",
            "Exporting drafts": "Export complete",
            "Refining cutlines": "Refinement complete",
            "Building survey overview": "Survey overview ready",
            "Building dense terrain": "Dense terrain ready",
        }[self.operation]
        if code:
            status = (
                "Cancelled"
                if code == 130
                else f"{self.operation} failed. See the output below."
            )
        try:
            if self.manifest is not None and self.manifest.is_file():
                result: dict[str, Any] = read_json(self.manifest)
                if not code and self.operation in {
                    "Building survey overview",
                    "Building dense terrain",
                }:
                    overview: Path = Path(result["overview_report"])
                    self.open_path(overview)
                    if result.get("alignment") in {"cached", "existing"}:
                        status += " · Saved alignment reused"
                device: str | None = result.get("device_used")
                if device:
                    status += f" · {'Saved matches' if result.get('run_mode') == 'reused' else 'Matching'} on {device.upper()}"
                    if result.get("gpu_fallback"):
                        status += " after GPU fallback"
                if not code and self.operation in {
                    "Analyzing photos",
                    "Refining cutlines",
                }:
                    verified: int = len(result.get("verified_pairs", []))
                    status += f" · {verified} verified overlaps"
                    review: Path = self.manifest.with_name("selection-plan.html")
                    if self.operation == "Refining cutlines" and review.is_file():
                        plan: dict[str, Any]
                        plan, _ = read_tier2_proposal(review.with_suffix(".json"))
                        self.refined_review = review
                        self.can_export_refined = bool(plan["export_pairs"])
                        status += " · Review refined cutlines"
                    elif self.operation == "Analyzing photos":
                        self.apply_selection(
                            load_saved_selection(self.manifest, self.photos)
                        )
                        compact: dict[str, Any] | None = result.get("compact_selection")
                        if compact is not None:
                            status += (
                                f" · {len(self.recommended)} recommended "
                                f'for a target of {compact["target_photos"]}'
                            )
                        status += " · Review drafts and adjust checked photos"
                        if not self.reuse.get() and matching_manifest(
                            self.manifest, self.photos
                        ):
                            self.reuse.set(str(self.manifest))
        except (OSError, ValueError, KeyError) as error:
            status += f" · Could not read run results. {error}"
        self.status.set(status)

    def cancel(self) -> None:
        """Request cancellation so COLMAP can stop and save its logs."""
        if self.process is None or self.process.stdin is None:
            return
        try:
            self.process.stdin.write("!")
            self.process.stdin.flush()
        except OSError:
            pass
        self.cancel_button.configure(state="disabled")
        self.status.set("Cancelling")

    def close(self) -> None:
        """Close after any running worker has handled cancellation."""
        if self.process is None:
            self.root.destroy()
        else:
            self.closing = True
            self.cancel()

    def show_review(self) -> None:
        """Open the HTML report in the default browser."""
        selection: SavedSelection | None = self.loaded_selection
        if selection is not None and selection.report is not None:
            self.open_path(selection.report)

    def show_logs(self) -> None:
        """Open the folder containing the current run logs."""
        if self.log_folder is not None:
            self.open_path(self.log_folder)

    def show_refined(self) -> None:
        """Open the most recent refinement separately from the initial drafts."""
        if self.refined_review is not None:
            self.open_path(self.refined_review)

    def open_path(self, path: Path) -> None:
        """Open a review or log folder in Windows."""
        try:
            os.startfile(path)
        except OSError as error:
            self.fail(error)

    def fail(self, error: Exception) -> None:
        """Show an actionable error without discarding the current review."""
        messagebox.showerror("Auto-DDG Cutline", str(error), parent=self.root)


def main() -> int:
    """Dispatch the desktop app and console worker from one packaged entry point."""
    if sys.argv[1:2] == ["--engine"]:
        del sys.argv[1]
        return run_worker()
    if sys.argv[1:2] == ["--launcher-check"]:
        root: tk.Tk = tk.Tk()
        try:
            root.withdraw()
            Launcher(root)
            root.update_idletasks()
            if sys.argv[2:] and root.title() != f"Auto-DDG Cutline {sys.argv[2]}":
                return 1
        finally:
            root.destroy()
        return 0
    if (
        getattr(sys, "frozen", False)
        and Path(sys.executable).stem.casefold() == Path(CLI_EXE).stem.casefold()
    ):
        return cli_main()
    root = tk.Tk()
    Launcher(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

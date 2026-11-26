"""Measure a COLMAP run and display progress while it is in flight."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter, sleep
from typing import Any, Iterator, Literal

HEARTBEAT_SECONDS: float = 30.0
SAMPLE_SECONDS: float = 5.0
MANIFEST_RETRY_DELAYS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)


def replace_manifest(temporary: Path, destination: Path) -> None:
    """Publish a complete manifest after any brief Windows file lock clears."""
    for delay in MANIFEST_RETRY_DELAYS:
        try:
            os.replace(temporary, destination)
            return
        except OSError as error:
            # Windows can report access denied, a sharing violation or a lock
            # violation when another process briefly holds the file.
            if getattr(error, "winerror", None) not in {5, 32, 33}:
                raise
            sleep(delay)
    os.replace(temporary, destination)


@dataclass(frozen=True)
class ProgressEvent:
    """One progress update for the terminal or a future interface."""

    stage: str
    state: Literal[
        "started", "progress", "heartbeat", "completed", "failed", "interrupted"
    ]
    elapsed_seconds: float
    stage_seconds: float
    attempt: str | None = None
    message: str = ""
    completed: int | None = None
    total: int | None = None


class RunReport:
    """Own the progress display and atomic checkpoints for one COLMAP job."""

    def __init__(self, path: Path, manifest: dict[str, Any], verbose: bool) -> None:
        """Attach timing and display state to the existing run manifest."""
        self.path: Path = path
        self.manifest: dict[str, Any] = manifest
        self.verbose: bool = verbose
        self.start: float = perf_counter()
        self.stages: list[dict[str, Any]] = []
        self.manifest["stage_timings"] = self.stages
        self.active: dict[str, Any] | None = None
        self.stage_start: float = self.start
        self.last_message: float = self.start
        self.lock: Lock = Lock()
        self.stop: Event = Event()
        self.heartbeat: Thread = Thread(target=self.pulse, daemon=True)
        self.heartbeat.start()

    def update(
        self, message: str = "", completed: int | None = None, total: int | None = None
    ) -> None:
        """Show a stage update, throttling exact counters in compact mode."""
        if completed is not None and not self.verbose:
            if completed not in (1, total) and completed % 10 != 0:
                return
        with self.lock:
            self.emit("progress", message, completed, total)

    def emit(
        self,
        state: Literal[
            "started", "progress", "heartbeat", "completed", "failed", "interrupted"
        ],
        message: str = "",
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        """Render an event while the caller holds the progress lock."""
        if self.active is None:
            return
        now: float = perf_counter()
        self.last_message = now
        event: ProgressEvent = ProgressEvent(
            self.active["stage"],
            state,
            now - self.start,
            now - self.stage_start,
            self.active.get("attempt"),
            message,
            completed,
            total,
        )
        elapsed: int = int(event.elapsed_seconds)
        clock: str = (
            f"{elapsed // 3600:02d}:{elapsed // 60 % 60:02d}:{elapsed % 60:02d}"
        )
        attempt: str = f" ({event.attempt.upper()})" if event.attempt else ""
        count: str = (
            f" {event.completed}/{event.total}" if event.completed is not None else ""
        )
        detail: str = f" | {event.message}" if event.message else ""
        print(
            f"[{clock}] {event.stage.replace('_', ' ')}{attempt} {event.state}{count}"
            f" | stage {event.stage_seconds:.1f}s{detail}",
            flush=True,
        )

    def pulse(self) -> None:
        """Show activity during quiet Python or subprocess work."""
        while not self.stop.wait(min(1.0, HEARTBEAT_SECONDS)):
            with self.lock:
                if (
                    self.active is not None
                    and perf_counter() - self.last_message >= HEARTBEAT_SECONDS
                ):
                    self.emit("heartbeat", "still running")

    def save(self) -> None:
        """Atomically replace the manifest so readers see a complete checkpoint."""
        self.manifest["elapsed_seconds"] = round(perf_counter() - self.start, 3)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path.parent,
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary: Path = Path(stream.name)
            try:
                json.dump(
                    self.manifest, stream, indent=2, ensure_ascii=False, allow_nan=False
                )
                stream.write("\n")
                stream.close()
                replace_manifest(temporary, self.path)
            finally:
                stream.close()
                temporary.unlink(missing_ok=True)

    @contextmanager
    def phase(
        self, stage: str, attempt: str | None = None, message: str = ""
    ) -> Iterator[None]:
        """Measure a non-overlapping pipeline stage, including abnormal exits."""
        record: dict[str, Any] = {
            "stage": stage,
            "attempt": attempt,
            "status": "running",
        }
        self.stages.append(record)
        with self.lock:
            self.active = record
            self.stage_start = perf_counter()
            self.emit("started", message)
        status: Literal["completed", "failed", "interrupted"] = "completed"
        try:
            yield
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        except Exception:
            status = "failed"
            raise
        finally:
            with self.lock:
                record.update(
                    status=status,
                    elapsed_seconds=round(perf_counter() - self.stage_start, 3),
                )
                self.emit(status)
                self.active = None
            self.save()

    def skip(self, stage: str, reason: str) -> None:
        """Record a pipeline stage intentionally bypassed on match reuse."""
        self.stages.append(
            {
                "stage": stage,
                "status": "skipped",
                "reason": reason,
                "elapsed_seconds": 0.0,
            }
        )

    def finish(self) -> None:
        """Stop live progress and summarize measured stage durations."""
        self.stop.set()
        self.heartbeat.join()
        self.save()
        print("  Stage timings (wall time)", flush=True)
        for record in self.stages:
            attempt: str = (
                f" ({record['attempt'].upper()})" if record.get("attempt") else ""
            )
            print(
                f"    {record['stage'].replace('_', ' '):30s}{attempt:7s} "
                f"{record['elapsed_seconds']:8.2f}s  {record['status']}",
                flush=True,
            )
        print(
            f"    {'total':37s} {self.manifest['elapsed_seconds']:8.2f}s  {self.manifest['status']}",
            flush=True,
        )


class ResourceSampler:
    """Sample optional process-tree RSS and device-wide NVIDIA memory."""

    def __init__(self, enabled: bool) -> None:
        """Keep all optional tools out of the basic timing path."""
        self.metrics: dict[str, Any] = {
            "sample_interval_seconds": SAMPLE_SECONDS if enabled else None,
            "sampled_peak_process_tree_rss_bytes": None,
            "sampled_peak_gpu_device_memory_bytes": None,
        }
        self.stop: Event = Event()
        self.thread: Thread | None = None
        if enabled:
            self.thread = Thread(target=self.sample, daemon=True)
            self.thread.start()

    def sample(self) -> None:
        """Sample available metrics independently and keep their observed peaks."""
        try:
            psutil: Any = importlib.import_module("psutil")
        except ImportError:
            psutil = None
        parent: Any = None
        if psutil is not None:
            try:
                parent = psutil.Process()
            except psutil.Error:
                pass
        nvidia_smi: str | None = shutil.which("nvidia-smi")
        while not self.stop.is_set():
            if parent is not None:
                rss: int = 0
                with_parent: list[Any] = [parent]
                try:
                    with_parent.extend(parent.children(recursive=True))
                except psutil.Error:
                    pass
                for process in with_parent:
                    try:
                        rss += process.memory_info().rss
                    except psutil.Error:
                        continue
                key: str = "sampled_peak_process_tree_rss_bytes"
                self.metrics[key] = max(self.metrics[key] or 0, rss)
            if nvidia_smi is not None:
                try:
                    result: subprocess.CompletedProcess[str] = subprocess.run(
                        [
                            nvidia_smi,
                            "--query-gpu=index,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=2,
                        check=False,
                    )
                    if result.returncode == 0:
                        for line in result.stdout.splitlines():
                            device: str
                            memory: str
                            device, _, memory = line.partition(",")
                            if device.strip().isdigit() and memory.strip().isdigit():
                                gpu_key: str = "sampled_peak_gpu_device_memory_bytes"
                                peaks: dict[str, int] = self.metrics[gpu_key] or {}
                                index: str = device.strip()
                                peaks[index] = max(
                                    peaks.get(index, 0), int(memory.strip()) * 1024**2
                                )
                                self.metrics[gpu_key] = peaks
                except (OSError, subprocess.TimeoutExpired):
                    nvidia_smi = None
            if self.stop.wait(SAMPLE_SECONDS):
                break

    def close(self) -> dict[str, Any]:
        """Stop sampling before serializing the metrics."""
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        return self.metrics

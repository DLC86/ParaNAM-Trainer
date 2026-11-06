"""Persistent training instances backed by immutable, deduplicated capture copies."""

from __future__ import annotations

import copy
import hashlib
import os
import shutil
from pathlib import Path
import time
import uuid

from filelock import FileLock

from nam.capture.project import atomic_write_json, save_project
from .workspace import Workspace, read_json
from . import training

ACTIVE = {"starting", "running", "scoring", "pausing", "stopping"}
FINISHED = {"completed", "needs_captures", "paused", "stopped", "failed", "interrupted"}


def immutable_audio(source, cache):
    """Never hardlink a mutable project recording into a running training job."""
    source = Path(source).resolve(strict=True)
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache / (uuid.uuid4().hex + ".pending")
    before = source.stat()
    digest = hashlib.sha256()
    try:
        with source.open("rb") as src, temporary.open("wb") as dst:
            while block := src.read(1024 * 1024):
                digest.update(block)
                dst.write(block)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(
                "A recording changed while preparing training. Add the job again after recording finishes."
            )
        target = cache / (digest.hexdigest() + ".wav")
        # The cache is append-only. Existing instances keep referencing the same bytes.
        with FileLock(str(cache / "store.lock")):
            if not target.exists():
                temporary.replace(target)
        return str(target)
    finally:
        temporary.unlink(missing_ok=True)


def snapshot_workspace(workspace, root):
    """Capture the click-time configuration, independently of future project edits."""
    root = Path(root).resolve()
    identifier = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    destination = root / "instances" / identifier / "project"
    destination.mkdir(parents=True)
    project = workspace.project.model_copy(deep=True)
    preferences = copy.deepcopy(workspace.preferences)
    preferences["draft_controls"] = copy.deepcopy(workspace.draft_controls)
    preferences["_queue_source_project"] = preferences.get(
        "_queue_source_project", str(workspace.directory)
    )
    paths = {}

    def freeze(path, relative):
        original = (workspace.directory / path).resolve()
        if original not in paths:
            paths[original] = immutable_audio(original, root / "audio")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(paths[original], target)
        except OSError:
            shutil.copy2(paths[original], target)
        return relative

    project.train_input = freeze(
        project.train_input,
        (
            "input_standard.wav"
            if preferences["input_mode"] == "standard"
            else (
                "input_tts_v11.wav"
                if preferences["input_mode"] == "tts"
                else "input_custom.wav"
            )
        ),
    )
    preferences["custom_input"] = (
        project.train_input
        if preferences["input_mode"] != "standard"
        else preferences.get("custom_input", "")
    )
    project.validation_input = freeze(project.validation_input, "input_validation.wav")
    for index, entry in enumerate(project.entries):
        if entry.y_path:
            entry.y_path = freeze(
                entry.y_path, f"captures/entry-{index:05d}-{entry.split}.wav"
            )
    save_project(project, destination)
    atomic_write_json(destination / "paranam.json", preferences)
    previous = training.current_job(workspace.directory)
    if (
        preferences["mode"] == "active_learning"
        and previous
        and training.job_progress(previous).get("status") == "needs_captures"
    ):
        (destination / "training").mkdir()
        atomic_write_json(
            destination / "training" / "current.json", {"directory": str(previous)}
        )
    return Workspace(destination)


class TrainingQueue:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "queue.json"
        self.lock = FileLock(str(self.directory / "queue.lock"))

    def entries(self):
        return read_json(self.path, {"entries": []})["entries"]

    @staticmethod
    def snapshot_root(workspace):
        # Keep large audio copies on the project's drive; only the small queue index
        # belongs in application data. Projects remain self-contained for backups.
        source = Path(
            workspace.preferences.get("_queue_source_project", workspace.directory)
        )
        return source / "training" / "queue-data"

    def _edit(self, operation):
        from .recovery import require_resolved

        with self.lock:
            data = read_json(self.path, {"version": 1, "entries": []})
            require_resolved(self.path)
            result = operation(data["entries"])
            atomic_write_json(self.path, data)
            return result

    def add(self, workspace):
        if not workspace.training_ready():
            raise ValueError("Complete every planned capture before adding training.")
        frozen = snapshot_workspace(workspace, self.snapshot_root(workspace))
        job = training.make_job(frozen, job_directory=frozen.directory.parent)
        spec = read_json(job / "spec.json")
        source = frozen.preferences["_queue_source_project"]
        spec.update(
            original_project=source,
            queue_snapshot=True,
            models_directory=str(
                Path(source) / "models" / frozen.directory.parent.name
            ),
        )
        atomic_write_json(job / "spec.json", spec)
        progress = training.job_progress(job)
        progress.update(status="queued", message="Waiting in the training queue")
        atomic_write_json(job / "progress.json", progress)
        entry = dict(
            id=uuid.uuid4().hex,
            job=str(job),
            project=source,
            name=workspace.project.name or workspace.directory.name,
            kind=workspace.project.capture_type,
            created_at=time.time(),
            state="queued",
        )
        atomic_write_json(job / "instance.json", entry)
        self._edit(lambda entries: entries.append(entry))
        return entry

    def retrain(self, identifier):
        """Restore instance settings in its source project without starting a job."""
        entry = self.get(identifier)
        if not entry or entry["state"] not in FINISHED:
            raise ValueError(
                "Select a stopped, paused or finished training instance; queued or active jobs cannot be retrained."
            )
        spec = read_json(Path(entry["job"]) / "spec.json")
        if not spec or not (Path(entry["project"]) / "capture_project.json").is_file():
            raise ValueError("Reconnect the drive containing the original project.")
        workspace = Workspace(Path(entry["project"]))
        # Keep the source recordings/paths and hardware configuration. Frozen job
        # paths must never become editable capture destinations.
        source_input = workspace.preferences.get("custom_input", "")
        saved = copy.deepcopy(spec.get("preferences", {}))
        for key in ("rate_schedule", "_queue_source_project"):
            saved.pop(key, None)
        workspace.preferences.update(saved)
        workspace.preferences["custom_input"] = source_input
        snapshot = Workspace(Path(spec["project"]))
        workspace.draft_controls = copy.deepcopy(snapshot.draft_controls)
        workspace.project.user_metadata = copy.deepcopy(snapshot.project.user_metadata)
        workspace.save()
        return workspace

    def capture_round(self, identifier):
        """Editable active-learning branch; never record over frozen hardlinks."""
        entry = self.get(identifier)
        spec = read_json(Path(entry["job"]) / "spec.json")
        original = Workspace(Path(spec["project"]))
        target = (
            Path(entry["project"])
            / "active-learning"
            / f"{entry['id'][:8]}-round-{spec['round'] + 2}"
        )
        if (target / "capture_project.json").exists():
            return Workspace(target)
        target.mkdir(parents=True, exist_ok=True)
        for relative in {
            original.project.train_input,
            original.project.validation_input,
            *(e.y_path for e in original.project.entries),
        }:
            source = original.directory / relative
            if source.is_file():
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        save_project(original.project, target)
        atomic_write_json(target / "paranam.json", original.preferences)
        (target / "training").mkdir()
        atomic_write_json(
            target / "training" / "current.json", {"directory": entry["job"]}
        )
        return Workspace(target)

    def register_legacy(self, workspace):
        """List existing jobs without changing their checkpoints or recordings."""
        known = {e["job"] for e in self.entries()} | set(
            read_json(self.path, {}).get("removed", [])
        )
        root = workspace.directory / "training"
        specfiles = [
            *root.glob("*/spec.json"),
            *root.glob("queue-data/instances/*/spec.json"),
        ]
        for specfile in sorted(specfiles):
            job = specfile.parent.resolve()
            if str(job) in known or (job / "queue-removed.json").exists():
                continue
            progress = training.job_progress(job)
            state = progress.get("status", "interrupted")
            if state in ACTIVE:
                state = "interrupted"
            spec = read_json(specfile, {}, tolerate_corrupt=True)
            manifest = read_json(job / "instance.json", {})
            if not spec:
                state = "failed"
            entry = dict(
                id=uuid.uuid4().hex,
                job=str(job),
                project=str(workspace.directory),
                name=manifest.get(
                    "name", workspace.project.name or workspace.directory.name
                ),
                kind=manifest.get("kind", workspace.project.capture_type),
                created_at=manifest.get("created_at", specfile.stat().st_mtime),
                state=state,
                legacy=not spec.get("queue_snapshot"),
            )
            self._edit(
                lambda entries, entry=entry: (
                    entries.append(entry)
                    if not any(e["job"] == entry["job"] for e in entries)
                    else None
                )
            )

    def get(self, identifier):
        return next((e for e in self.entries() if e["id"] == identifier), None)

    def update(self, identifier, **values):
        def edit(entries):
            entry = next(e for e in entries if e["id"] == identifier)
            entry.update(values)
            return dict(entry)

        return self._edit(edit)

    def move(self, identifier, offset):
        def edit(entries):
            pending = [i for i, e in enumerate(entries) if e["state"] == "queued"]
            position = next(
                (p for p, i in enumerate(pending) if entries[i]["id"] == identifier),
                None,
            )
            if position is None or not 0 <= position + offset < len(pending):
                return
            a, b = pending[position], pending[position + offset]
            entries[a], entries[b] = entries[b], entries[a]

        self._edit(edit)

    def prioritize(self, identifier):
        """Resume a paused queue's current instance before its waiting successors."""

        def edit(entries):
            entry = next(
                (
                    e
                    for e in entries
                    if e["id"] == identifier and e["state"] == "queued"
                ),
                None,
            )
            if entry is None:
                return
            entries.remove(entry)
            position = next(
                (i for i, e in enumerate(entries) if e["state"] == "queued"),
                len(entries),
            )
            entries.insert(position, entry)

        self._edit(edit)

    def remove(self, identifier):
        from .recovery import require_resolved

        with self.lock:
            data = read_json(self.path, {"version": 1, "entries": []})
            require_resolved(self.path)
            entries = data["entries"]
            entry = next(e for e in entries if e["id"] == identifier)
            if entry["state"] in ACTIVE:
                raise ValueError("Stop or pause this training before removing it.")
            if Path(entry["job"]).is_dir():
                try:
                    atomic_write_json(
                        Path(entry["job"]) / "queue-removed.json",
                        {"removed_at": time.time()},
                    )
                except OSError:
                    pass  # The durable index also remembers removals on offline drives.
            entries.remove(entry)  # Keep models, captures and checkpoints on disk.
            data.setdefault("removed", []).append(entry["job"])
            atomic_write_json(self.path, data)

    def prepare_resume(self, identifier, batch_size=None, device=None):
        entry = self.get(identifier)
        if not entry or entry["state"] not in FINISHED - {
            "completed",
            "needs_captures",
        }:
            raise ValueError(
                "Select a paused, stopped or interrupted training instance."
            )
        job = Path(entry["job"])
        spec = read_json(job / "spec.json")
        if not spec or not Path(spec.get("project", "")).is_dir():
            raise ValueError(
                "This instance's files are unavailable. Reconnect its project drive before resuming."
            )
        from .recipes import uses_removed_rate_preset, REMOVED_RATE_MESSAGE

        if uses_removed_rate_preset(spec):
            raise ValueError(REMOVED_RATE_MESSAGE)
        workspace = Workspace(Path(spec["project"]))
        from .evaluation import PROTOCOL

        if spec.get("evaluation_protocol") != PROTOCOL:
            raise ValueError(
                "Retrain this instance to reserve the initial and final validation audio. "
                "Existing captures can be reused."
            )
        if not spec.get("queue_snapshot"):
            if not training.job_can_resume(workspace, job):
                raise ValueError(
                    "This legacy training no longer matches its original project. Its files have been kept."
                )
            workspace = snapshot_workspace(workspace, self.snapshot_root(workspace))
            spec.update(
                project=str(workspace.directory),
                fingerprint=workspace.fingerprint(),
                original_project=entry["project"],
                queue_snapshot=True,
            )
            atomic_write_json(job / "spec.json", spec)
        if batch_size is not None:
            if not 1 <= batch_size <= 128:
                raise ValueError("Batch size must be between 1 and 128")
            workspace.preferences[
                (
                    "static_batch_size"
                    if workspace.project.capture_type == "static"
                    else "batch_size"
                )
            ] = batch_size
        if device is not None:
            if device not in ("auto", "cpu"):
                raise ValueError("Choose Automatic or CPU for this instance")
            workspace.preferences["device"] = device
        workspace.save()
        # The job belongs to its own snapshot, never the currently open UI project.
        (workspace.directory / "training").mkdir(exist_ok=True)
        atomic_write_json(
            workspace.directory / "training" / "current.json", {"directory": str(job)}
        )
        from .recovery import ensure_resume_checkpoint

        ensure_resume_checkpoint(job)
        checkpoint = job / f"resume-{training.job_progress(job).get('member', 0)}.ckpt"
        if checkpoint.exists() or training.legacy_checkpoint(
            job, training.job_progress(job).get("member", 0)
        ):
            training.make_job(workspace, resume=True)
        else:
            saved_progress = training.job_progress(job)
            if (
                saved_progress.get("completed_epoch", saved_progress.get("epoch", 0))
                > 0
            ):
                raise ValueError(
                    "No resumable checkpoint is available. The instance and recordings have been kept; restore a checkpoint backup or use Retrain explicitly."
                )
            # Pausing before the first complete epoch can safely restart this same instance.
            atomic_write_json(job / "command.json", {})
            spec["learning"]["train_dataloader"][
                "batch_size"
            ] = workspace.training_batch_size()
            spec["learning"]["val_dataloader"]["batch_size"] = 1
            spec["learning"]["trainer"]["accelerator"] = workspace.preferences["device"]
            spec["preferences"]["batch_size"] = workspace.training_batch_size()
            spec["preferences"]["device"] = workspace.preferences["device"]
            spec["fingerprint"] = workspace.fingerprint()
            atomic_write_json(job / "spec.json", spec)
        self.update(identifier, state="queued", pid=None, pid_started=None, error=None)
        progress = training.job_progress(job)
        progress.update(
            status="queued", message="Queued to resume from the last complete epoch"
        )
        atomic_write_json(job / "progress.json", progress)


def process_alive(entry):
    if not entry.get("pid"):
        return False
    import psutil

    try:
        process = psutil.Process(entry["pid"])
        return (
            process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
            and abs(process.create_time() - entry.get("pid_started", 0)) < 0.1
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

"""Persistent training instances backed by immutable, deduplicated capture copies."""

from __future__ import annotations

import copy
import filecmp
import hashlib
import os
import shutil
from pathlib import Path
import time
import uuid

from filelock import FileLock

from nam.capture.project import atomic_write_json, load_project, save_project
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


def snapshot_workspace(workspace, root, *, editable=False):
    """Copy a project; retrain editors receive *private* writable audio files.

    Queue jobs use content-addressed hardlinked immutable audio for storage
    efficiency. An editable Retrain copy must never share those inodes: even
    a plain WAV overwrite must not modify the original job or audio cache.
    """
    root = Path(root).resolve()
    identifier = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    destination = root / "instances" / identifier / "project"
    destination.mkdir(parents=True)
    project = workspace.project.model_copy(deep=True)
    preferences = copy.deepcopy(workspace.preferences)
    preferences.setdefault("advanced_parametric", {}).setdefault(
        "rms_loss_normalization",
        workspace.training_options("parametric")["rms_loss_normalization"],
    )
    preferences["draft_controls"] = copy.deepcopy(workspace.draft_controls)
    preferences["_queue_source_project"] = preferences.get(
        "_queue_source_project", str(workspace.directory)
    )
    paths = {}

    def freeze(path, relative):
        original = (workspace.directory / path).resolve()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if editable:
            # Never hardlink a mutable Retrain file to an existing frozen
            # job or the immutable cache. Users may overwrite any WAV.
            shutil.copy2(original, target)
        else:
            if original not in paths:
                paths[original] = immutable_audio(original, root / "audio")
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
                else (
                    "input_trunc.wav"
                    if preferences["input_mode"] == "trunc"
                    else "input_custom.wav"
                )
            )
        ),
    )
    preferences["custom_input"] = (
        project.train_input
        if preferences["input_mode"] != "standard"
        else preferences.get("custom_input", "")
    )
    project.validation_input = freeze(project.validation_input, "input_validation.wav")
    synthetic_refs = {}
    for index, entry in enumerate(project.entries):
        if entry.y_path and entry.status == "captured":
            if entry.synthetic_mute:
                source = entry.y_path
                if source not in synthetic_refs:
                    synthetic_refs[source] = freeze(
                        source, f"captures/synthetic-mute-{entry.split}.wav"
                    )
                entry.y_path = synthetic_refs[source]
            else:
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


def freeze_snr_noise(source, snapshot, source_project=None):
    """Bind raw pre-blip noise to the frozen training WAVs in a queue instance."""
    import soundfile as sf
    from nam.capture.session import pre_blip_noise_power

    source = Path(source)
    if source_project is None:
        source_project = load_project(source)
    powers = {}
    for index, frozen in enumerate(snapshot.project.entries):
        if frozen.split != "train" or frozen.status != "captured":
            continue
        if frozen.synthetic_mute:
            # There is no real pre-blip noise recording for a synthetic target.
            powers[Path(frozen.y_path).as_posix()] = None
            continue
        original = (
            source_project.entries[index] if index < len(source_project.entries) else None
        )
        frozen_path = snapshot.directory / frozen.y_path
        if (
            original is None
            or original.split != frozen.split
            or original.index != frozen.index
            or original.params != frozen.params
            or original.captured_at != frozen.captured_at
            or not (source / original.y_path).is_file()
            or not filecmp.cmp(source / original.y_path, frozen_path, shallow=False)
        ):
            raise ValueError(
                f"Cannot match the original capture for {frozen.y_path}. "
                "Add a new training instance using the current recordings."
            )
        original_path = source / original.y_path
        has_manifest = any(
            (parent / "captures_raw" / "manifest.json").is_file()
            for parent in original_path.parents
        )
        if has_manifest:
            noise_power = pre_blip_noise_power(
                original_path, sf.info(frozen_path).samplerate
            )
        else:
            # Imported and legacy captures have no preserved pre-blip return.
            # Store an explicit "unavailable" value so default-on SNR can
            # safely use unity weights, without inventing a noise measurement.
            import warnings
            warnings.warn(
                f"SNR unavailable for {original.y_path}: no raw preamble. "
                "Using unity SNR weights for this take."
            )
            noise_power = None
        powers[Path(frozen.y_path).as_posix()] = noise_power
    return powers


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
        from .cadences import editable_cadences

        frozen.preferences["advanced_" + frozen.project.capture_type] = (
            editable_cadences(frozen, frozen.training_options())
        )
        frozen.save()
        job = training.make_job(frozen, job_directory=frozen.directory.parent)
        spec = read_json(job / "spec.json")
        if spec["learning"].get("snr_weighted_loss_normalization"):
            # Retraining from a historical frozen job must not depend on the
            # source rig still having its pre-blip raw recordings. The old job
            # has already frozen the exact noise values for these same takes.
            previous_job = workspace.preferences.get("_retrain_source_job")
            if previous_job:
                previous_spec = read_json(Path(previous_job) / "spec.json")
                previous_noise = previous_spec.get("learning", {}).get("snr_noise_powers")
                if not isinstance(previous_noise, dict):
                    # Jobs made before default-on SNR have no frozen noise.
                    # Their audio is valid; do not pretend noise was measured.
                    previous_noise = {}
                from nam.capture.project import load_project

                original_project = load_project(Path(previous_spec["project"]))
                powers = {}
                if len(original_project.entries) != len(frozen.project.entries):
                    raise ValueError("Retrain captures no longer match the original plan")
                for old, new in zip(original_project.entries, frozen.project.entries):
                    if old.split != new.split or old.params != new.params:
                        raise ValueError("Retrain captures no longer match the original plan")
                    if old.split == "train" and old.status == "captured":
                        # None explicitly records unavailable pre-blip noise.
                        # Older jobs with no SNR data are still retrainable.
                        powers[new.y_path] = previous_noise.get(old.y_path)
                spec["learning"]["snr_noise_powers"] = powers
            else:
                spec["learning"]["snr_noise_powers"] = freeze_snr_noise(
                    workspace.directory, frozen, workspace.project
                )
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
        """Open an independent editable copy of this job's exact original plan.

        Never derive the retrain dataset from the currently open project or
        regenerate it using newer LHS/Sobol settings. Frozen audio belongs to
        the old instance; a fresh editable copy keeps it immutable.
        """
        entry = self.get(identifier)
        if not entry or entry["state"] not in FINISHED:
            raise ValueError(
                "Select a stopped, paused or finished training instance; queued or active jobs cannot be retrained."
            )
        spec = read_json(Path(entry["job"]) / "spec.json")
        if not spec or not Path(spec.get("project", "")).is_dir():
            raise ValueError("The original training snapshot is unavailable.")
        snapshot = Workspace(Path(spec["project"]))
        # A later active-learning round may have appended *pending* proposals
        # to the same frozen project. They were not part of this job's training
        # dataset and must never be imported into Retrain.
        snapshot.project.entries = [
            capture for capture in snapshot.project.entries
            if capture.status == "captured"
        ]
        workspace = snapshot_workspace(
            snapshot, self.snapshot_root(snapshot), editable=True
        )
        source_input = workspace.preferences.get("custom_input", "")
        saved = copy.deepcopy(spec.get("preferences", {}))
        for key in ("rate_schedule", "_queue_source_project"):
            saved.pop(key, None)
        effective = saved.pop("advanced_training", None)
        if isinstance(effective, dict):
            saved["advanced_" + saved.get("capture_type", entry["kind"])] = (
                copy.deepcopy(effective)
            )
        workspace.preferences.update(saved)
        # The source project's plan may have been changed after this job was
        # queued; never import that plan or its current recording paths.
        workspace.preferences["custom_input"] = source_input
        workspace.preferences["_queue_source_project"] = entry["project"]
        workspace.preferences["_retrain_source_job"] = str(Path(entry["job"]).resolve())
        # Start the editor from the plan that supplied the frozen recordings.
        # Old spec preferences may predate migration of knob limits into rules.
        workspace.draft_controls = [c.to_dict() for c in workspace.project.control_specs()]
        workspace.preferences["draft_plan_rules"] = [
            rule.model_dump(mode="json") for rule in workspace.project.plan_rules
        ]
        workspace.project.user_metadata = copy.deepcopy(snapshot.project.user_metadata)
        if isinstance(effective, dict):
            from .cadences import editable_cadences, stored_schedules
            workspace.preferences[
                "advanced_" + saved.get("capture_type", entry["kind"])
            ] = editable_cadences(
                snapshot, workspace.training_options(), stored_schedules(spec)
            )
        workspace.save()
        # Retrain is editable: reopen after recording _retrain_source_job so
        # legacy per-knob Avoid thresholds migrate into Rules. Its source job
        # remains frozen and the original recordings are never modified.
        return Workspace(workspace.directory)

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

    def rename(self, identifier, name):
        """Rename the frozen model's Name metadata, not a queue-only label.

        Training weights, checkpoints, input audio, the source project and
        existing export file paths are never renamed or altered.
        """
        from .model_names import set_model_metadata_name
        from .recovery import require_resolved

        if not isinstance(name, str):
            raise ValueError("The model name must be text.")
        name = name.strip()
        if not name:
            raise ValueError("The model name cannot be empty.")
        if len(name) > 200 or any(ord(char) < 32 or ord(char) == 127 for char in name):
            raise ValueError("The model name must be one line of at most 200 characters.")

        with self.lock:
            data = read_json(self.path, {"version": 1, "entries": []})
            require_resolved(self.path)
            entry = next(
                (e for e in data["entries"] if e["id"] == identifier), None
            )
            if entry is None:
                raise ValueError("The selected training job no longer exists.")
            job = Path(entry["job"])
            spec_path = job / "spec.json"
            require_resolved(spec_path)
            spec = read_json(spec_path, {})
            if not spec:
                raise ValueError("The training specifications are unavailable.")
            captures = spec.get("static_captures") or []
            if captures:
                # A static queue row represents a batch; its Name column shows
                # the first capture, which has its own independently named model.
                captures[0]["name"] = name
                captures[0].setdefault("metadata", {})["name"] = name
            else:
                spec.setdefault("model", {}).setdefault("metadata", {})["name"] = name
            atomic_write_json(spec_path, spec)

            # An active worker will pick up the new name during final export.
            # Avoid writing files concurrently with that worker.
            if entry["state"] not in ACTIVE:
                progress = read_json(job / "progress.json", {})
                filenames = set()
                if captures:
                    exports = progress.get("exports") or []
                    if exports:
                        filenames.add(str(exports[0]))
                    filenames.add(str(job / "member-0" / "model.nam"))
                else:
                    filenames.update(
                        str(p) for pattern in ("model_parametric.pnam", "model.nam")
                        for p in job.glob(f"member-*/{pattern}")
                    )
                    if progress.get("export"):
                        filenames.add(str(progress["export"]))
                for filename in sorted(filenames):
                    path = Path(filename)
                    if not path.is_file():
                        continue
                    model = read_json(path, {})
                    if not isinstance(model, dict) or not model:
                        raise ValueError(f"Cannot update model metadata in {path}")
                    set_model_metadata_name(model, name)
                    atomic_write_json(path, model)

            entry.pop("display_name", None)
            atomic_write_json(self.path, data)
            return dict(entry)

    def move(self, identifier, offset):
        def edit(entries):
            a = next(
                (i for i, entry in enumerate(entries) if entry["id"] == identifier),
                None,
            )
            if a is None or not 0 <= a + offset < len(entries):
                return
            b = a + offset
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

    def set_total_epochs(self, identifier, epochs):
        """Change a dormant instance's epoch target without touching checkpoints.

        A completed run may be extended. Previously completed members must then
        train their additional epochs too, with their own saved optimizer state.
        """
        entry = self.get(identifier)
        if not entry:
            raise ValueError("Select a training instance.")
        if type(epochs) is not int or not 1 <= epochs <= 1000000:
            raise ValueError("Total epochs must be between 1 and 1000000.")
        if entry["state"] in ACTIVE:
            # The worker owns worker.lock throughout training; this live edit
            # must NOT wait for that lock. A dedicated atomic target file lets
            # the running Lightning callback adopt the new limit at a batch
            # boundary, without touching optimizer state or checkpoint files.
            from .recovery import require_resolved

            with self.lock:
                data = read_json(self.path, {"version": 1, "entries": []})
                require_resolved(self.path)
                active = next(
                    (item for item in data["entries"] if item["id"] == identifier),
                    None,
                )
                if active is None:
                    raise ValueError("The selected training job no longer exists.")
                if active["state"] not in ACTIVE:
                    # Status changed while the menu was open; use the dormant
                    # path rather than writing a request to a stopped worker.
                    entry = active
                else:
                    job = Path(active["job"])
                    require_resolved(job / "spec.json")
                    require_resolved(job / "progress.json")
                    spec = read_json(job / "spec.json", {})
                    progress = training.job_progress(job)
                    if not spec or not Path(spec.get("project", "")).is_dir():
                        raise ValueError("The training snapshot is unavailable.")
                    completed = max(
                        [int(progress.get("completed_epoch") or 0),
                         int(progress.get("epoch") or 0)]
                        + [
                            int(read_json(job / f"resume-{member}.json", {}).get(
                                "completed_epoch"
                            ) or 0)
                            for member in range(int(spec["members"]))
                        ]
                    )
                    if epochs < completed:
                        raise ValueError(
                            f"Total epochs cannot be lower than the {completed} "
                            "already reached or started."
                        )
                    if epochs == int(spec["preferences"]["epochs"]):
                        return
                    workspace = Workspace(Path(spec["project"]))
                    workspace.preferences["epochs"] = epochs
                    workspace.preferences[
                        "static_epochs" if active["kind"] == "static"
                        else "parametric_epochs"
                    ] = epochs
                    workspace.save()
                    spec["preferences"]["epochs"] = epochs
                    spec["learning"]["trainer"]["max_epochs"] = epochs
                    spec["fingerprint"] = workspace.fingerprint()
                    atomic_write_json(job / "spec.json", spec)
                    # Do not update progress.json here: the worker writes it
                    # frequently. It will acknowledge the target and refresh
                    # the visible epochs counter at the next batch boundary.
                    atomic_write_json(job / "epoch-target.json", {"epochs": epochs})
                    return
        job = Path(entry["job"])
        spec = read_json(job / "spec.json")
        if not spec or not Path(spec.get("project", "")).is_dir():
            raise ValueError("The training snapshot is unavailable.")
        progress = training.job_progress(job)
        completed = int(progress.get("completed_epoch") or 0)
        for member in range(int(spec["members"])):
            checkpoint = read_json(job / f"resume-{member}.json", {})
            completed = max(completed, int(checkpoint.get("completed_epoch") or 0))
        if entry["state"] == "completed":
            # An increased epoch target is not an executed epoch.
            completed = max(completed, int(progress.get("epoch") or 0))
        if epochs < completed:
            raise ValueError(
                f"Total epochs cannot be lower than the {completed} already completed."
            )
        previous = int(spec["preferences"]["epochs"])
        if epochs == previous:
            return
        from filelock import FileLock, Timeout
        from .recovery import require_resolved
        try:
            with FileLock(
                str(Path(spec["project"]) / "training" / "worker.lock"),
                timeout=0,
            ):
                require_resolved(job / "spec.json")
                require_resolved(job / "progress.json")
                workspace = Workspace(Path(spec["project"]))
                workspace.preferences["epochs"] = epochs
                workspace.preferences[
                    "static_epochs" if entry["kind"] == "static" else "parametric_epochs"
                ] = epochs
                workspace.save()
                spec["preferences"]["epochs"] = epochs
                spec["learning"]["trainer"]["max_epochs"] = epochs
                spec["fingerprint"] = workspace.fingerprint()
                atomic_write_json(job / "spec.json", spec)
                # Replace any live-worker override as well. Otherwise an
                # earlier target could take precedence after pause/resume.
                atomic_write_json(job / "epoch-target.json", {"epochs": epochs})
                progress["epochs"] = epochs
                if epochs > previous:
                    # Every member needs the extra epochs, including members
                    # already marked complete in multi-model jobs.
                    progress["completed_members"] = []
                    progress["member"] = 0
                    # The next run starts from member 0 and its checkpoint.
                    # Display its actual completed epoch instead of 0 / total.
                    first = read_json(job / "resume-0.json", {})
                    first_epoch = first.get("completed_epoch")
                    if first_epoch is None:
                        first_epoch = (
                            max(int(progress.get("completed_epoch") or 0),
                                int(progress.get("epoch") or 0))
                            if int(spec["members"]) == 1 else 0
                        )
                    progress["epoch"] = int(first_epoch)
                    progress["completed_epoch"] = int(first_epoch)
                    if entry["kind"] == "static":
                        progress["exports"] = []
                progress["message"] = f"Total epochs changed to {epochs}"
                atomic_write_json(job / "progress.json", progress)
        except Timeout as error:
            raise ValueError("Wait for the training worker to stop.") from error

    def prepare_resume(self, identifier, batch_size=None, device=None):
        entry = self.get(identifier)
        if not entry or entry["state"] not in FINISHED - {"needs_captures"}:
            raise ValueError(
                "Select a paused, stopped, interrupted or extendable completed instance."
            )
        job = Path(entry["job"])
        spec = read_json(job / "spec.json")
        if not spec or not Path(spec.get("project", "")).is_dir():
            raise ValueError(
                "This instance's files are unavailable. Reconnect its project drive before resuming."
            )
        if entry["state"] == "completed":
            completed = max(
                [int(read_json(job / f"resume-{m}.json", {}).get("completed_epoch") or 0)
                 for m in range(int(spec["members"]))]
                + [int(training.job_progress(job).get("completed_epoch") or 0)]
            )
            if spec["preferences"]["epochs"] <= completed:
                raise ValueError("Increase total epochs before resuming a completed run.")
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
        # A rejected Resume must not persist edits to the frozen capture plan.
        workspace.save()
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

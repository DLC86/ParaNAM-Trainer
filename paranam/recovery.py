"""Recover replaceable state without discarding projects, recordings or checkpoints."""

import copy
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

_issues = {}
_cached = {}
_lock = threading.RLock()
_disposable = {
    "queue-control.json",
    "current.json",
    "instance.json",
    "export-request.json",
    "export-result.json",
}


class RecoveryRequired(ValueError):
    pass


class WorkerActive(RecoveryRequired):
    pass


def validate_document(path, data):
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    if Path(path).name == "queue.json":
        entries = data.get("entries")
        required = {"id", "job", "project", "name", "kind", "state", "created_at"}
        if not isinstance(entries, list) or any(
            not isinstance(entry, dict) or not required <= entry.keys()
            for entry in entries
        ):
            raise ValueError("The training queue has an invalid structure")
    if Path(path).name == "progress.json":
        for key in ("member", "epoch", "epochs", "completed_epoch"):
            if key in data and (type(data[key]) is not int or data[key] < 0):
                raise ValueError(f"Invalid progress field: {key}")
        if not isinstance(data.get("esr_history", []), list):
            raise ValueError("Invalid ESR history")
    return data


def _load(path):
    return validate_document(path, json.loads(Path(path).read_text(encoding="utf-8")))


def _stamp(path):
    try:
        info = Path(path).stat()
        return info.st_mtime_ns, info.st_size
    except FileNotFoundError:
        return None


def issues():
    with _lock:
        return [dict(issue) for issue in _issues.values()]


def is_blocked(path):
    with _lock:
        return _issues.get(str(Path(path).resolve()), {}).get("blocking", False)


def require_resolved(path):
    if is_blocked(path):
        raise RecoveryRequired(
            f"Resolve the damaged file using Review recovery first: {path}"
        )


def accept_valid(path):
    key = str(Path(path).resolve())
    with _lock:
        if _issues.get(key, {}).get("blocking"):
            _issues[key].update(
                action="Resolved",
                blocking=False,
                detail="The file now contains valid JSON state.",
            )
        _cached.pop(key, None)


def _record(path, action, detail, *, blocking=False, preserved=None):
    _issues[str(Path(path).resolve())] = dict(
        path=str(Path(path).resolve()),
        action=action,
        detail=detail,
        blocking=blocking,
        preserved=str(preserved) if preserved else None,
    )
    try:
        print(f"State recovery: {path}: {action}: {detail}", flush=True)
    except (OSError, ValueError, AttributeError):
        pass


def _preserve(path):
    path = Path(path)
    saved = path.with_name(
        path.name
        + ".corrupt-"
        + time.strftime("%Y%m%d-%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    shutil.copy2(path, saved)
    return saved


def restore_document(path, data, action, detail):
    """Keep the damaged bytes first. Never remove audio or checkpoint files."""
    from nam.capture.project import atomic_write_json

    path = Path(path)
    validate_document(path, data)
    preserved = None
    try:
        preserved = _preserve(path) if path.exists() else None
        atomic_write_json(path, data)
    except OSError as error:
        _record(
            path,
            "Recovered in memory only",
            f"{detail}\nCannot save recovered state: {error}. Fix permissions and retry recovery.",
            blocking=True,
            preserved=preserved,
        )
        stamp = _stamp(path)
        _cached[str(path.resolve())] = (stamp, data, None)
        return data
    _cached.pop(str(path.resolve()), None)
    _record(path, action, detail, preserved=preserved)
    return data


def restore_backup(path, source):
    path, source = Path(path), Path(source)
    if source.resolve() == path.resolve():
        raise ValueError("Choose a separate, valid JSON backup")
    data = validate_document(path, json.loads(source.read_text(encoding="utf-8")))
    if path.name == "capture_project.json":
        from nam.capture.project import CaptureProject

        CaptureProject.model_validate(data).knob_specs()
    elif path.name == "spec.json":
        if not {"model", "data", "learning", "preferences", "project"} <= data.keys():
            raise ValueError("Choose a training spec backup for this file.")
    elif path.name == "paranam.json":
        from .workspace import DEFAULTS

        if (
            not (set(DEFAULTS) | {"advanced_parametric", "advanced_static"})
            & data.keys()
        ):
            raise ValueError("Choose a project-preferences backup for this file.")
    elif (
        path.name == "progress.json"
        and not {"status", "epoch", "member"} <= data.keys()
    ):
        raise ValueError("Choose a training-progress backup for this file.")
    return restore_document(path, data, "Restored backup", f"Restored from {source}")


def checkpoint_progress(job, base=None, *, lock_held=False):
    """Rebuild metadata without racing a live worker or running a training step."""
    from filelock import FileLock, Timeout

    job = Path(job)
    spec = _load(job / "spec.json")
    if lock_held:
        return _checkpoint_progress_locked(job, spec, base)
    try:
        with FileLock(
            str(Path(spec["project"]) / "training" / "worker.lock"), timeout=0
        ):
            return _checkpoint_progress_locked(job, spec, base)
    except Timeout as error:
        raise WorkerActive(
            "A worker still owns this training instance. Waiting for its next state update; no checkpoint was changed."
        ) from error


def _checkpoint_progress_locked(job, spec, base):
    import torch

    members = spec["members"]
    if members > 1 and not base:
        raise RecoveryRequired(
            "A multi-model run needs its progress backup to identify completed members safely."
        )
    member = (base or {}).get("member", 0)
    checkpoint_path = job / f"resume-{member}.ckpt"
    candidate = checkpoint_path
    failures = []

    def candidates():
        yield checkpoint_path
        yield checkpoint_path.with_name(checkpoint_path.name + ".bak")
        from .training import legacy_checkpoint

        legacy = legacy_checkpoint(job, member)
        if legacy is not None:
            yield legacy[2]

    best = None
    for candidate in candidates():
        try:
            with zipfile.ZipFile(candidate) as archive:
                damaged = archive.testzip()
                if damaged:
                    raise RecoveryRequired(f"Checkpoint checksum failed: {damaged}")
            checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
            if not checkpoint.get("state_dict") or not checkpoint.get(
                "optimizer_states"
            ):
                raise RecoveryRequired("No complete weights/optimizer state")
            callback = next(
                (
                    state
                    for state in checkpoint.get("callbacks", {}).values()
                    if isinstance(state, dict) and "esr_history" in state
                ),
                None,
            )
            if callback is None:
                raise RecoveryRequired("No recoverable progress metadata")
            history = callback["esr_history"]
            if not isinstance(history, list) or any(
                not isinstance(row, dict) or type(row.get("epoch")) is not int
                for row in history
            ):
                raise RecoveryRequired("Invalid checkpoint ESR history")
            if (
                type(checkpoint.get("global_step")) is not int
                or checkpoint["global_step"] < 0
            ):
                raise RecoveryRequired("Invalid checkpoint optimizer step")
            completed = max((row["epoch"] for row in history), default=0)
            if completed > spec["preferences"]["epochs"] or (
                checkpoint["global_step"] and not completed
            ):
                raise RecoveryRequired(
                    "Cannot identify a complete epoch for this training spec"
                )
            # Prefer the newest COMPLETE point, not an older backup when a
            # later valid epoch checkpoint survived. Canonical wins exact ties.
            priority = (
                2
                if candidate == checkpoint_path
                else (1 if candidate.name.endswith(".bak") else 0)
            )
            key = (checkpoint["global_step"], completed, priority)
            if best is None or key > best[0]:
                best = (key, candidate, checkpoint, callback)
        except Exception as error:
            failures.append(f"{candidate.name}: {error}")
    if best is None:
        raise RecoveryRequired("; ".join(failures))
    _, candidate, checkpoint, callback = best
    history = callback["esr_history"]
    step = checkpoint["global_step"]
    completed = max((row["epoch"] for row in history), default=0)
    if step and not completed:
        raise RecoveryRequired("Cannot identify the last complete epoch safely")
    if candidate != checkpoint_path:
        from nam.capture.project import replace_file

        preserved = _preserve(checkpoint_path) if checkpoint_path.exists() else None
        descriptor, temporary = tempfile.mkstemp(
            prefix=".checkpoint-recovery-", suffix=".tmp", dir=job
        )
        try:
            with os.fdopen(descriptor, "wb") as dst, candidate.open("rb") as src:
                shutil.copyfileobj(src, dst)
                dst.flush()
                os.fsync(dst.fileno())
            replace_file(temporary, checkpoint_path, keep_checkpoint_backup=False)
        finally:
            Path(temporary).unlink(missing_ok=True)
        _record(
            checkpoint_path,
            "Restored previous checkpoint",
            f"Recovered the complete checkpoint through epoch {completed}. The incomplete/damaged newer state cannot be resumed.",
            preserved=preserved,
        )
    from nam.capture.project import atomic_write_json

    boundary_path = job / f"resume-{member}.json"
    boundary = dict(
        completed_epoch=completed,
        restart_epoch=completed + 1,
        global_step=step,
        best_esr=callback.get("best_esr"),
    )
    try:
        saved_boundary = _load(boundary_path)
    except (OSError, ValueError, UnicodeError):
        saved_boundary = None
    if saved_boundary != boundary:
        if boundary_path.exists() and saved_boundary is None:
            _preserve(boundary_path)
        atomic_write_json(boundary_path, boundary)
    progress = dict(base or {})
    progress.update(
        status="interrupted",
        member=member,
        members=members,
        epochs=spec["preferences"]["epochs"],
        epoch=completed,
        round=spec["round"] + 1,
        completed_epoch=completed,
        restart_epoch=completed + 1,
        global_step=step,
        completed_members=list((base or {}).get("completed_members", [])),
        elapsed=sum(callback.get("epoch_times", [])),
        eta=None,
        batch_fraction=0,
        best_esr=callback.get("best_esr"),
        best_train_esr=callback.get("best_train_esr"),
        validation_reference="full",
        epoch_times=callback.get("epoch_times", []),
        esr_history=history,
        esr_history_member=member,
        checkpoint=str(checkpoint_path),
        resume_available=True,
        batch_size=spec["preferences"]["batch_size"],
        updated_at=time.time(),
        recovery_required=False,
        message=f"Recovered after unexpected shutdown. Resume restarts epoch {completed + 1} from the last complete checkpoint.",
    )
    last = callback.get("last_validation") or {}
    for target, source in (
        ("esr", "mean"),
        ("esr_full", "full"),
        ("esr_lite", "lite"),
        ("train_esr_full", "train_full"),
        ("train_esr_lite", "train_lite"),
    ):
        progress[target] = last.get(source)
    return progress


def recover_json(path, error, default=None, *, tolerate_corrupt=False):
    path = Path(path)
    key = str(path.resolve())
    stamp = _stamp(path)

    def commit(data, action, detail):
        # A live worker may have replaced the damaged file while its checkpoint
        # was being examined. Never replace its newer valid state with ours.
        if _stamp(path) != stamp:
            try:
                return _load(path)
            except (OSError, ValueError, UnicodeError):
                value = copy.deepcopy(data)
                if path.name == "progress.json":
                    value.update(
                        status="running", recovery_required=True, resume_available=False
                    )
                _record(
                    path,
                    "State changed during recovery",
                    "The file changed before recovery could commit. No newer state was overwritten; retry after the writer stops.",
                    blocking=True,
                )
                return value
        return restore_document(path, data, action, detail)

    with _lock:
        cached = _cached.get(key)
        if (
            cached
            and cached[0] == stamp
            and (len(cached) < 4 or time.monotonic() < cached[3])
        ):
            if cached[2] and not tolerate_corrupt:
                raise RecoveryRequired(cached[2])
            return copy.deepcopy(cached[1])
        backup = None
        try:
            backup = validate_document(path, _load(path.with_name(path.name + ".bak")))
        except (OSError, ValueError, UnicodeError):
            pass
        detail = str(error)
        if path.stem.startswith("resume-") and path.suffix == ".json":
            try:
                import torch

                checkpoint_path = path.with_suffix(".ckpt")
                with zipfile.ZipFile(checkpoint_path) as archive:
                    if archive.testzip() is not None:
                        raise RecoveryRequired("Checkpoint checksum failed")
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                state = next(
                    state
                    for state in checkpoint.get("callbacks", {}).values()
                    if isinstance(state, dict) and "esr_history" in state
                )
                completed = max(
                    (row["epoch"] for row in state["esr_history"]), default=0
                )
                value = dict(
                    completed_epoch=completed,
                    restart_epoch=completed + 1,
                    global_step=checkpoint["global_step"],
                    best_esr=state.get("best_esr"),
                )
                return commit(
                    value,
                    "Recovered checkpoint boundary",
                    f"Restored epoch {completed} boundary metadata from the complete checkpoint.",
                )
            except Exception as checkpoint_error:
                detail += f"; boundary recovery: {checkpoint_error}"
        if path.name == "progress.json":
            try:
                value = checkpoint_progress(path.parent, backup)
                return commit(value, "Recovered checkpoint", value["message"])
            except WorkerActive as checkpoint_error:
                value = dict(
                    backup or {},
                    status="running",
                    recovery_required=True,
                    resume_available=False,
                    message=str(checkpoint_error),
                )
                _record(
                    path, "Worker still active", str(checkpoint_error), blocking=True
                )
                _cached[key] = (stamp, value, None, time.monotonic() + 5)
                return value
            except Exception as checkpoint_error:
                detail += f"; checkpoint recovery: {checkpoint_error}"
        if backup is not None:
            if path.name == "queue.json":
                removed = [
                    entry
                    for entry in backup["entries"]
                    if (Path(entry["job"]) / "queue-removed.json").is_file()
                ]
                backup["entries"] = [
                    entry for entry in backup["entries"] if entry not in removed
                ]
                backup.setdefault("removed", []).extend(
                    entry["job"] for entry in removed
                )
            return commit(
                backup,
                "Restored last valid JSON",
                "Restored the previous valid .bak document. Checkpoints are unchanged.",
            )
        if path.name in _disposable:
            return commit(
                copy.deepcopy(default or {}),
                "Reset replaceable state",
                "The damaged cache/pointer was preserved; project and checkpoint data are unchanged.",
            )
        message = f"Cannot recover {path}: {detail}. Use Review recovery to restore a valid backup."
        _record(path, "Recovery required", message, blocking=True)
        fallback = copy.deepcopy(default if default is not None else {})
        if path.name == "progress.json":
            fallback = dict(
                status="failed",
                recovery_required=True,
                resume_available=False,
                epoch=0,
                message=message,
            )
        # Leave essential files untouched until the user chooses a solution.
        soft = path.name in ("queue.json", "progress.json") or tolerate_corrupt
        _cached[key] = (
            stamp,
            fallback,
            None if path.name in ("queue.json", "progress.json") else message,
        )
        if soft:
            return fallback
        raise RecoveryRequired(message)


def retry(path):
    path = Path(path)
    with _lock:
        _cached.pop(str(path.resolve()), None)
    try:
        _load(path)
    except (OSError, ValueError, UnicodeError) as error:
        return recover_json(
            path,
            error,
            {"version": 1, "entries": []} if path.name == "queue.json" else {},
        )
    _issues.pop(str(path.resolve()), None)


def reset_session():
    """Tests and a new application session do not inherit old recovery notices."""
    with _lock:
        _issues.clear()
        _cached.clear()


def ensure_resume_checkpoint(job, *, lock_held=False):
    """Validate an explicitly requested resume before a worker can start."""
    from .workspace import read_json

    job = Path(job)
    progress = read_json(job / "progress.json", {})
    if progress.get("recovery_required"):
        raise RecoveryRequired(
            "Resolve this instance's damaged state using Review recovery before resuming."
        )
    member = progress.get("member", 0)
    target = job / f"resume-{member}.ckpt"
    if not target.exists() and not target.with_name(target.name + ".bak").exists():
        return  # The existing legacy-checkpoint path handles older instances.
    try:
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is None:
                return
    except (OSError, zipfile.BadZipFile):
        pass
    value = checkpoint_progress(job, progress, lock_held=lock_held)
    restore_document(
        job / "progress.json", value, "Recovered checkpoint", value["message"]
    )
    require_resolved(target)
    require_resolved(job / "progress.json")

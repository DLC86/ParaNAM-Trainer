"""One durable training worker at a time; the project editor is independent."""

import copy
import os
from pathlib import Path
import sys

from filelock import FileLock, Timeout
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QThread, Signal

from nam.capture.project import atomic_write_json
from . import training
from .training_queue import TrainingQueue, ACTIVE, FINISHED, process_alive
from .workspace import read_json


class PrepareTraining(QThread):
    ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, queue, workspace, parent=None):
        super().__init__(parent)
        self.queue = queue
        self.workspace = copy.copy(workspace)
        self.workspace.project = workspace.project.model_copy(deep=True)
        self.workspace.preferences = copy.deepcopy(workspace.preferences)
        self.workspace.draft_controls = copy.deepcopy(workspace.draft_controls)

    def run(self):
        try:
            self.ready.emit(self.queue.add(self.workspace))
        except Exception as error:
            self.failed.emit(str(error))


class ExportSavedBest(QThread):
    def __init__(self, job, parent=None):
        super().__init__(parent)
        self.job = Path(job)

    def run(self):
        request = read_json(self.job / "export-request.json", {})
        try:
            path = training.export_saved_best(self.job, request["path"])
            result = dict(id=request["id"], ok=True, path=path)
        except Exception as error:
            result = dict(id=request.get("id"), ok=False, error=str(error))
        atomic_write_json(self.job / "export-result.json", result)


class QueueController(QObject):
    changed = Signal()
    failed = Signal(str)

    def __init__(self, directory, parent=None):
        super().__init__(parent)
        self.store = TrainingQueue(directory)
        self.guard = FileLock(str(self.store.directory / "runner.lock"))
        self.process = None
        self.active_id = None
        self.enabled = False
        self.preparing = None
        self.control_path = self.store.directory / "queue-control.json"
        self.paused_queue_id = read_json(self.control_path, {}).get("paused_job")
        self.reconcile_interrupted()

    def reconcile_interrupted(self):
        from .recovery import RecoveryRequired

        for entry in self.store.entries():
            if entry["state"] in ACTIVE:
                if process_alive(entry):
                    self.active_id = entry["id"]
                else:
                    progress = training.job_progress(entry["job"])
                    state = progress.get("status")
                    try:
                        self.store.update(
                            entry["id"],
                            state=state if state in FINISHED else "interrupted",
                            pid=None,
                        )
                    except RecoveryRequired:
                        pass  # Read-only recovery must still allow the UI to open.

    def enqueue(self, workspace):
        if self.preparing:
            raise ValueError("Wait for the current training copy to finish.")
        self.preparing = PrepareTraining(self.store, workspace, self)
        self.preparing.ready.connect(lambda entry: self.changed.emit())
        self.preparing.failed.connect(self.failed.emit)
        self.preparing.finished.connect(self._prepared)
        self.preparing.start()

    def _prepared(self):
        self.preparing.deleteLater()
        self.preparing = None
        self.changed.emit()
        self.tick()

    def start(self):
        from .recovery import require_resolved

        require_resolved(self.store.path)
        if self.paused_queue_id:
            entry = self.store.get(self.paused_queue_id)
            if entry and entry["state"] in ("paused", "interrupted", "failed"):
                self.store.prepare_resume(entry["id"])
                self.store.prioritize(entry["id"])
            self.paused_queue_id = None
            atomic_write_json(self.control_path, {})
        self.enabled = True
        self.tick()

    def pause_queue(self):
        # Disable scheduling before issuing the pause: the worker can exit and
        # emit changed/finished immediately, and must never launch its successor.
        self.enabled = False
        self.paused_queue_id = self.active_id
        atomic_write_json(self.control_path, {"paused_job": self.paused_queue_id})
        if self.active_id:
            self.pause(self.active_id)
        self.changed.emit()

    def tick(self):
        if self.active_id:
            entry = self.store.get(self.active_id)
            progress = training.job_progress(entry["job"])
            if self.process is None and not process_alive(entry):
                self._finish(0)
            elif progress.get("status") in FINISHED and self.process is None:
                self._finish(0)
            return
        if not self.enabled:
            return
        entry = next((e for e in self.store.entries() if e["state"] == "queued"), None)
        if not entry:
            self.enabled = self.preparing is not None
            return
        try:
            self.guard.acquire(timeout=0)
        except Timeout:
            return
        self.active_id = entry["id"]
        try:
            self.store.update(entry["id"], state="starting")
        except StopIteration:
            self.active_id = None
            self.guard.release()
            return  # Another app window removed this waiting instance.
        job = Path(entry["job"])
        spec = read_json(job / "spec.json", {}, tolerate_corrupt=True)
        from .recovery import is_blocked

        recovery_blocked = (
            is_blocked(job / "spec.json")
            or is_blocked(job / "progress.json")
            or training.job_progress(job).get("recovery_required")
        )
        if not spec or not Path(spec.get("project", "")).is_dir() or recovery_blocked:
            self.store.update(
                entry["id"],
                state="failed",
                error=(
                    "This instance needs state recovery. Use Review recovery before resuming."
                    if recovery_blocked
                    else "Training files are unavailable. Reconnect this project's drive, then Resume."
                ),
            )
            self.active_id = None
            self.guard.release()
            self.changed.emit()
            return
        process = QProcess(self)
        process.setWorkingDirectory(spec["project"])
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        if not getattr(sys, "frozen", False):
            environment.insert(
                "PYTHONPATH",
                str(Path(__file__).resolve().parent.parent)
                + os.pathsep
                + environment.value("PYTHONPATH"),
            )
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(lambda: process.readAllStandardOutput())
        process.started.connect(self._started)
        process.finished.connect(self._finish)
        process.errorOccurred.connect(self._error)
        self.process = process
        args = ["--train-worker", str(job)]
        from . import gpu_runtime

        if spec["preferences"]["device"] != "cpu" and gpu_runtime.ready():
            args.append("--gpu-runtime")
        if not getattr(sys, "frozen", False):
            args = ["-m", "paranam"] + args
        process.start(sys.executable, args)
        self.changed.emit()

    def _started(self):
        import psutil

        try:
            process = psutil.Process(self.process.processId())
        except psutil.NoSuchProcess:
            return  # A very short-lived worker will be handled by finished/error.
        self.store.update(
            self.active_id,
            state="running",
            pid=process.pid,
            pid_started=process.create_time(),
        )
        self.changed.emit()

    def _error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            if self.active_id:
                self.store.update(self.active_id, error=self.process.errorString())
            self._finish(1)

    def _finish(self, code=0, *_):
        if not self.active_id:
            return
        entry = self.store.get(self.active_id)
        progress = training.job_progress(entry["job"])
        state = progress.get("status")
        if state not in FINISHED:
            state = "failed" if code else "interrupted"
            progress.update(
                status=state,
                message=entry.get("error")
                or "Worker interrupted; resume restarts the last incomplete epoch.",
            )
            try:
                atomic_write_json(Path(entry["job"]) / "progress.json", progress)
            except OSError:
                pass  # A disconnected project drive must not block other projects.
        self.store.update(self.active_id, state=state, pid=None, pid_started=None)
        self.active_id = None
        if self.process:
            self.process.deleteLater()
            self.process = None
        self.guard.release()
        self.changed.emit()

    def pause(self, identifier):
        entry = self.store.get(identifier)
        if not entry:
            return
        training.command(entry["job"], "pause")
        if identifier == self.active_id or entry["state"] in ACTIVE:
            self.store.update(identifier, state="pausing")
        elif entry["state"] == "queued":
            self.store.update(identifier, state="paused")
        self.changed.emit()

    def stop(self, identifier):
        entry = self.store.get(identifier)
        if not entry:
            return
        training.command(entry["job"], "stop")
        if identifier == self.active_id or entry["state"] in ACTIVE:
            self.store.update(identifier, state="stopping")
        else:
            self.store.update(identifier, state="stopped")
        self.changed.emit()

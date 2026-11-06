import hashlib
import json
from pathlib import Path

import pytest
import torch
from PySide6.QtWidgets import QMessageBox, QApplication

from nam.capture.project import atomic_write_json, replace_file
from paranam import recovery, training
from paranam.queue_controller import QueueController
from paranam.training_queue import TrainingQueue
from paranam.workspace import Workspace, read_json
from tests.paranam.test_training import ready


def saved_checkpoint(path, completed=4):
    history = [
        dict(
            epoch=i,
            full=0.1 / i,
            lite=0.2 / i,
            mean=0.15 / i,
            train_full=0.05 / i,
            train_lite=0.1 / i,
        )
        for i in range(1, completed + 1)
    ]
    checkpoint = dict(
        epoch=max(0, completed - 1),
        global_step=completed * 10,
        state_dict={"weight": torch.tensor([float(completed)])},
        optimizer_states=[{"state": {}}],
        lr_schedulers=[{"last_epoch": completed * 10}],
        callbacks={
            "quality": dict(
                esr_history=history,
                epoch_times=[1.0] * completed,
                best_esr=history[-1]["full"] if history else None,
                best_train_esr=history[-1]["train_full"] if history else None,
                last_validation=history[-1] if history else None,
            )
        },
    )
    torch.save(checkpoint, path)
    return checkpoint


def test_json_backup_preserves_last_valid_state_and_corrupt_bytes(tmp_path):
    path = tmp_path / "current.json"
    atomic_write_json(path, {"directory": "first"})
    atomic_write_json(path, {"directory": "second"})
    assert read_json(tmp_path / "current.json.bak") == {"directory": "first"}
    path.write_bytes(b"\0" * 100)
    assert read_json(path, {}) == {"directory": "first"}
    assert next(tmp_path.glob("current.json.corrupt-*")).read_bytes() == b"\0" * 100
    assert read_json(tmp_path / "current.json.bak") == {"directory": "first"}
    assert not recovery.issues()[0]["blocking"]


def test_corrupt_current_without_backup_is_reset_but_not_erased(tmp_path):
    path = tmp_path / "queue-control.json"
    path.write_bytes(b'"unfinished')
    assert read_json(path, {}) == {}
    assert path.is_file()
    assert (
        next(tmp_path.glob("queue-control.json.corrupt-*")).read_bytes()
        == b'"unfinished'
    )


def test_essential_configuration_is_never_silently_defaulted(tmp_path):
    path = tmp_path / "paranam.json"
    raw = b"\0" * 20
    path.write_bytes(raw)
    with pytest.raises(recovery.RecoveryRequired, match="Review recovery"):
        read_json(path, {})
    assert path.read_bytes() == raw
    assert read_json(path, {}, tolerate_corrupt=True) == {}
    assert recovery.is_blocked(path)
    backup = tmp_path / "manual.bak"
    backup.write_text('{"epochs": 50}', encoding="utf-8")
    recovery.restore_backup(path, backup)
    assert read_json(path)["epochs"] == 50
    assert not recovery.is_blocked(path)


def test_corrupt_progress_recovers_checkpoint_and_queue_order(
    workspace, tmp_path, qapp
):
    workspace.preferences["epochs"] = 8
    ready(workspace)
    store = TrainingQueue(tmp_path / "queue")
    first, second = store.add(workspace), store.add(workspace)
    job = Path(first["job"])
    saved_checkpoint(job / "resume-0.ckpt")
    digest = hashlib.sha256((job / "resume-0.ckpt").read_bytes()).digest()
    store.update(first["id"], state="running", pid=999999999, pid_started=0)
    (job / "progress.json").write_bytes(b"\0" * 15098)
    controller = QueueController(store.directory)
    try:
        assert [entry["id"] for entry in controller.store.entries()] == [
            first["id"],
            second["id"],
        ]
        assert controller.store.get(first["id"])["state"] == "interrupted"
        assert controller.store.get(second["id"])["state"] == "queued"
        progress = training.job_progress(job)
        assert progress["completed_epoch"] == 4 and progress["restart_epoch"] == 5
        assert len(progress["esr_history"]) == 4
        assert progress["best_esr"] == 0.025
        assert not controller.enabled and controller.process is None
        assert hashlib.sha256((job / "resume-0.ckpt").read_bytes()).digest() == digest
        assert training.job_can_resume(
            Workspace(Path(read_json(job / "spec.json")["project"])), job
        )
    finally:
        controller.deleteLater()


def test_checkpoint_backup_is_durable_and_recovers_damaged_latest(workspace):
    workspace.preferences["epochs"] = 8
    ready(workspace)
    job = training.make_job(workspace)
    target = job / "resume-0.ckpt"
    pending = job / "resume-0.pending.ckpt"
    saved_checkpoint(pending, completed=3)
    replace_file(pending, target)
    saved_checkpoint(pending, completed=4)
    replace_file(pending, target)
    backup = target.with_name(target.name + ".bak")
    assert torch.load(backup, weights_only=False)["global_step"] == 30
    target.write_bytes(b"\0" * 10)
    (job / "progress.json").write_bytes(b"\0" * 100)
    progress = training.job_progress(job)
    assert progress["completed_epoch"] == 3 and progress["restart_epoch"] == 4
    assert torch.load(target, weights_only=False)["global_step"] == 30
    assert torch.load(backup, weights_only=False)["global_step"] == 30
    assert next(job.glob("resume-0.ckpt.corrupt-*")).read_bytes() == b"\0" * 10
    assert read_json(job / "resume-0.json")["global_step"] == 30


def test_unrecoverable_progress_is_held_and_never_retrained_automatically(workspace):
    ready(workspace)
    job = training.make_job(workspace)
    raw = b"broken JSON"
    (job / "progress.json").write_bytes(raw)
    assert training.job_progress(job)["recovery_required"]
    assert (job / "progress.json").read_bytes() == raw
    assert not training.job_can_resume(workspace, job)


def test_corrupt_queue_opens_ui_and_requires_choice_before_rebuilding(
    window, workspace, monkeypatch, qapp
):
    ready(workspace)
    window._attach(workspace)
    store = window.queue_controller.store
    first, second = store.add(workspace), store.add(workspace)
    source = b"\0" * 100
    store.path.with_name("queue.json.bak").unlink(missing_ok=True)
    store.path.write_bytes(source)
    window.refresh_training()
    qapp.processEvents()
    assert not window.recovery_notice.isHidden()
    assert window._recovery_dialog is not None
    assert QApplication.activeModalWidget() is None
    assert store.path.read_bytes() == source
    with pytest.raises(recovery.RecoveryRequired):
        store.update(first["id"], state="failed")
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No
    )
    window.rebuild_recovered_queue(store.path)
    assert store.path.read_bytes() == source
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes
    )
    window.rebuild_recovered_queue(store.path)
    assert {entry["job"] for entry in store.entries()} == {first["job"], second["job"]}
    assert all(entry["state"] == "interrupted" for entry in store.entries())
    assert not window.queue_controller.enabled
    assert next(store.directory.glob("queue.json.corrupt-*")).read_bytes() == source


def test_corrupt_recent_project_does_not_block_startup(window, tmp_path):
    project = tmp_path / "Damaged recent project"
    project.mkdir()
    (project / "capture_project.json").write_bytes(b"\0")
    window.settings.setValue("recent", [str(project)])
    window._refresh_recent()
    window.refresh_training()
    assert window.recent.count() == 1
    assert recovery.is_blocked(project / "capture_project.json")


def test_failed_flush_does_not_replace_good_state(tmp_path, monkeypatch):
    from nam.capture import project

    path = tmp_path / "state.json"
    atomic_write_json(path, {"epoch": 1})

    def failed_sync(descriptor):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(project._os, "fsync", failed_sync)
    with pytest.raises(OSError, match="simulated storage failure"):
        atomic_write_json(path, {"epoch": 2})
    assert read_json(path) == {"epoch": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_read_only_recovery_opens_in_memory_and_keeps_original(tmp_path, monkeypatch):
    path = tmp_path / "paranam.json"
    atomic_write_json(path, {"epochs": 50})
    atomic_write_json(path, {"epochs": 100})
    path.write_bytes(b"\0")

    def denied(current):
        raise PermissionError("read only")

    monkeypatch.setattr(recovery, "_preserve", denied)
    assert read_json(path, {}) == {"epochs": 50}
    assert read_json(path, {}) == {"epochs": 50}
    assert path.read_bytes() == b"\0"
    assert recovery.is_blocked(path)


def test_live_progress_stays_lightweight_but_boundaries_are_flushed(
    tmp_path, monkeypatch
):
    from nam.capture import project

    path = tmp_path / "progress.json"
    payload = dict(
        status="running", completed_epoch=1, member=0, epoch=2, batch_fraction=0.1
    )
    atomic_write_json(path, payload)
    original_sync = project._os.fsync
    flushes = []

    def sync(descriptor):
        flushes.append(True)
        original_sync(descriptor)

    monkeypatch.setattr(project._os, "fsync", sync)
    atomic_write_json(path, {**payload, "batch_fraction": 0.5})
    assert not flushes
    atomic_write_json(path, {**payload, "completed_epoch": 2})
    assert len(flushes) >= 2
    assert read_json(path)["completed_epoch"] == 2
    assert read_json(path.with_name("progress.json.bak"))["completed_epoch"] == 1


def test_missing_progress_is_rebuilt_without_changing_checkpoint(workspace):
    workspace.preferences["epochs"] = 8
    ready(workspace)
    job = training.make_job(workspace)
    saved_checkpoint(job / "resume-0.ckpt")
    (job / "progress.json").unlink()
    assert training.job_progress(job)["completed_epoch"] == 4


def test_active_worker_prevents_checkpoint_replacement(workspace):
    from filelock import FileLock

    workspace.preferences["epochs"] = 8
    ready(workspace)
    job = training.make_job(workspace)
    target = job / "resume-0.ckpt"
    saved_checkpoint(target)
    before = target.read_bytes()
    (job / "progress.json").write_bytes(b"\0" * 50)
    with FileLock(str(workspace.directory / "training" / "worker.lock")):
        result = training.job_progress(job)
        assert result["status"] == "running"
        assert result["recovery_required"]
        assert target.read_bytes() == before
        assert (job / "progress.json").read_bytes() == b"\0" * 50
    recovery.retry(job / "progress.json")
    assert training.job_progress(job)["completed_epoch"] == 4


def test_resume_checks_backup_even_if_progress_json_is_valid(workspace):
    workspace.preferences["epochs"] = 8
    ready(workspace)
    job = training.make_job(workspace)
    target = job / "resume-0.ckpt"
    saved_checkpoint(target.with_name(target.name + ".bak"), completed=3)
    target.write_bytes(b"\0" * 50)
    recovery.ensure_resume_checkpoint(job)
    assert training.job_progress(job)["completed_epoch"] == 3
    assert torch.load(target, weights_only=False)["global_step"] == 30


def test_corrupt_boundary_uses_checkpoint_epoch_instead_of_old_json(tmp_path):
    saved_checkpoint(tmp_path / "resume-0.ckpt", completed=4)
    boundary = tmp_path / "resume-0.json"
    boundary.write_bytes(b"\0" * 20)
    assert read_json(boundary)["completed_epoch"] == 4
    assert read_json(boundary)["restart_epoch"] == 5
    assert next(tmp_path.glob("resume-0.json.corrupt-*")).read_bytes() == b"\0" * 20


def test_recovery_does_not_overwrite_a_newer_writer_update(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    path.write_bytes(b"\0" * 20)

    def recovered(job, base):
        atomic_write_json(path, {"status": "completed", "epoch": 5, "member": 0})
        return {"status": "interrupted", "epoch": 4, "member": 0, "message": "old"}

    monkeypatch.setattr(recovery, "checkpoint_progress", recovered)
    assert read_json(path)["status"] == "completed"
    assert read_json(path)["epoch"] == 5


def test_all_checkpoint_saves_keep_previous_good_file_on_write_failure(tmp_path):
    from paranam.checkpoints import install_durable_checkpoint_io

    class Trainer:
        fail = False

        def save_checkpoint(self, path, **kwargs):
            if self.fail:
                Path(path).write_bytes(b"partial checkpoint")
                raise OSError("simulated interrupted best-checkpoint write")
            saved_checkpoint(path, completed=2)

    trainer = Trainer()
    install_durable_checkpoint_io(trainer)
    target = tmp_path / "best-full.ckpt"
    trainer.save_checkpoint(target)
    previous = target.read_bytes()
    trainer.fail = True
    with pytest.raises(OSError, match="interrupted"):
        trainer.save_checkpoint(target)
    assert target.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))


def test_project_can_be_opened_to_rebuild_a_damaged_queue(window, workspace):
    ready(workspace)
    store = window.queue_controller.store
    store.add(workspace)
    store.path.with_name("queue.json.bak").unlink(missing_ok=True)
    store.path.write_bytes(b"\0")
    window._attach(workspace)
    window.update_enabled()
    assert window.workspace.directory == workspace.directory
    assert str(workspace.directory) in window.settings.value("recent")
    assert recovery.is_blocked(store.path)
    assert not window.train_button.isEnabled()

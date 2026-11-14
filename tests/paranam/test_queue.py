import time
from pathlib import Path

import numpy as np
import pytest

from nam.capture.project import atomic_write_json
from nam.data import np_to_wav
from paranam.training_queue import TrainingQueue
from paranam.workspace import Workspace, read_json
from paranam import training


def small_ready(workspace, kind="parametric"):
    workspace.preferences.update(
        capture_type=kind, epochs=2, batch_size=1, static_batch_size=1, device="cpu"
    )
    workspace.preferences["advanced_parametric"] = dict(ny=32, fit_mrstft=False)
    workspace.preferences["advanced_static"] = dict(ny=32, fit_mrstft=False)
    workspace.generate_plan()
    x = (0.05 * np.sin(np.arange(7000) * 0.03)).astype(np.float32)
    for path in (workspace.project.train_input,):
        np_to_wav(x, workspace.directory / path, rate=48000)
    for entry in workspace.project.entries:
        target = workspace.directory / entry.y_path
        target.parent.mkdir(parents=True, exist_ok=True)
        np_to_wav(np.tanh(2 * x), target, rate=48000)
        entry.status, entry.delay = "captured", 0
    workspace.project.train_window.ny = workspace.project.validation_window.ny = 32
    workspace.save()
    return workspace


@pytest.mark.parametrize("kind", ["static", "parametric"])
def test_instances_freeze_audio_metadata_and_recipe_independently(
    workspace, tmp_path, kind
):
    small_ready(workspace, kind)
    workspace.project.user_metadata = {"modeled_by": "Original", "custom": {"value": 1}}
    workspace.save()
    queue = TrainingQueue(tmp_path / "queue")
    first = queue.add(workspace)
    spec = read_json(Path(first["job"]) / "spec.json")
    frozen = Workspace(Path(spec["project"]))
    recorded = frozen.directory / frozen.project.entries[0].y_path
    old_audio = recorded.read_bytes()
    source = workspace.directory / workspace.project.entries[0].y_path
    source.write_bytes(b"a later re-recording")
    workspace.preferences["epochs"] = 99
    workspace.project.user_metadata["custom"]["value"] = 2
    workspace.save()
    assert recorded.read_bytes() == old_audio
    assert frozen.preferences["epochs"] == 2
    assert frozen.project.user_metadata["custom"]["value"] == 1
    second = queue.add(workspace)
    second_spec = read_json(Path(second["job"]) / "spec.json")
    assert second["job"] != first["job"]
    assert second_spec["models_directory"] != spec["models_directory"]
    assert spec["model"]["net"]["name"] == (
        "PackedWaveNet" if kind == "static" else "HyperA2"
    )
    assert TrainingQueue(queue.directory).entries() == queue.entries()


def test_reorder_remove_and_resume_belong_to_instances(workspace, tmp_path):
    small_ready(workspace)
    queue = TrainingQueue(tmp_path / "queue")
    entries = [queue.add(workspace) for _ in range(3)]
    cached = list((queue.snapshot_root(workspace) / "audio").glob("*.wav"))
    assert len(cached) <= 3  # Inputs and duplicate capture content are deduplicated.
    queue.move(entries[2]["id"], -1)
    assert [e["id"] for e in queue.entries()] == [
        entries[0]["id"],
        entries[2]["id"],
        entries[1]["id"],
    ]
    queue.update(entries[0]["id"], state="paused")
    workspace.preferences["epochs"] = 777
    workspace.draft_controls = []
    workspace.save()
    queue.prepare_resume(entries[0]["id"])
    spec = read_json(Path(entries[0]["job"]) / "spec.json")
    assert spec["preferences"]["epochs"] == 2
    assert queue.get(entries[0]["id"])["state"] == "queued"
    queue.update(entries[1]["id"], state="running")
    with pytest.raises(ValueError, match="Stop or pause"):
        queue.remove(entries[1]["id"])
    queue.remove(entries[2]["id"])
    assert Path(entries[2]["job"]).is_dir()
    assert len(queue.entries()) == 2
    queue.register_legacy(workspace)
    assert not any(e["job"] == entries[2]["job"] for e in queue.entries())


@pytest.mark.parametrize("kind", ["static", "parametric"])
def test_retrain_restores_source_project_before_creating_fresh_job(
    window, workspace, tmp_path, kind
):
    small_ready(workspace, kind)
    workspace.project.user_metadata = {"modeled_by": "Original job"}
    workspace.save()
    window._attach(workspace)
    queue = window.queue_controller.store
    entry = queue.add(workspace)
    queue.update(entry["id"], state="completed")
    job = Path(entry["job"])
    spec_before = (job / "spec.json").read_bytes()
    frozen = Workspace(Path(read_json(job / "spec.json")["project"]))
    old_audio = frozen.directory / frozen.project.entries[0].y_path
    audio_before = old_audio.read_bytes()
    workspace.preferences["epochs"] = 999
    workspace.project.user_metadata = {"modeled_by": "Later project edits"}
    workspace.save()
    window.selected_job_id = entry["id"]
    window.navigate(4)
    window.refresh_training()
    assert window.retrain_button.isEnabled()
    window.retrain_button.click()
    editable = window.workspace
    assert window.pages.currentIndex() == 3
    assert window.epochs.value() == 2
    assert editable.project.user_metadata == {"modeled_by": "Original job"}
    assert editable.training_ready() and not editable.plan_changed()
    assert editable.directory == workspace.directory
    assert editable.directory != frozen.directory
    assert training.current_job(editable.directory) is None
    assert len(queue.entries()) == 1
    window.epochs.setValue(5)
    (editable.directory / editable.project.entries[0].y_path).write_bytes(
        b"new capture"
    )
    assert old_audio.read_bytes() == audio_before
    new = queue.add(editable)
    new_spec = read_json(Path(new["job"]) / "spec.json")
    assert new_spec["preferences"]["epochs"] == 5
    assert new_spec["round"] == 0
    assert (
        new_spec["models_directory"] != read_json(job / "spec.json")["models_directory"]
    )
    assert (job / "spec.json").read_bytes() == spec_before
    assert queue.get(entry["id"])["state"] == "completed"
    with pytest.raises(ValueError, match="queued"):
        queue.retrain(new["id"])


def test_queue_view_and_project_editing_are_independent(
    window, workspace, tmp_path, monkeypatch
):
    small_ready(workspace)
    window._attach(workspace)
    entry = window.queue_controller.store.add(workspace)
    window.selected_job_id = entry["id"]
    window.queue_controller.active_id = entry["id"]
    monkeypatch.setattr("paranam.queue_controller.process_alive", lambda entry: True)
    try:
        window.refresh_training()
        assert window.queue_controller.active_id == entry["id"]
        assert window.training_settings.isEnabled()
        assert window.metadata_card.isEnabled()
        assert window.control_card.isEnabled()
        assert window.train_button.isEnabled()
        window.epochs.setValue(8)
        assert read_json(Path(entry["job"]) / "spec.json")["preferences"]["epochs"] == 2
        assert window.metadata_card.parent() == window.page_contents[3]
        window.navigate(4)
        assert window.queue_table.rowCount() == 1
    finally:
        window.queue_controller.active_id = None


def test_interrupted_process_is_recoverable_without_opening_its_project(
    workspace, tmp_path, qapp
):
    from paranam.queue_controller import QueueController

    small_ready(workspace)
    queue = TrainingQueue(tmp_path / "queue")
    entry = queue.add(workspace)
    queue.update(entry["id"], state="running", pid=99999999, pid_started=1)
    controller = QueueController(queue.directory)
    assert controller.store.get(entry["id"])["state"] == "interrupted"
    controller.store.prepare_resume(entry["id"])
    assert controller.store.get(entry["id"])["state"] == "queued"


def test_active_learning_capture_branch_never_writes_frozen_audio(workspace, tmp_path):
    small_ready(workspace)
    queue = TrainingQueue(tmp_path / "queue")
    entry = queue.add(workspace)
    job = Path(entry["job"])
    spec = read_json(job / "spec.json")
    frozen = Workspace(Path(spec["project"]))
    audio = frozen.directory / frozen.project.entries[0].y_path
    before = audio.read_bytes()
    editable = queue.capture_round(entry["id"])
    (editable.directory / editable.project.entries[0].y_path).write_bytes(
        b"recaptured for the next round"
    )
    assert audio.read_bytes() == before
    assert editable.directory != frozen.directory
    assert training.current_job(editable.directory) == job


def test_paused_instance_runtime_options_do_not_change_the_project(workspace, tmp_path):
    small_ready(workspace)
    queue = TrainingQueue(tmp_path / "queue")
    entry = queue.add(workspace)
    queue.update(entry["id"], state="paused")
    queue.prepare_resume(entry["id"], batch_size=4, device="cpu")
    spec = read_json(Path(entry["job"]) / "spec.json")
    assert spec["learning"]["train_dataloader"]["batch_size"] == 4
    assert spec["learning"]["val_dataloader"]["batch_size"] == 1
    assert workspace.training_batch_size() == 1


def test_capture_another_project_while_training_queue_runs(
    window, workspace, tmp_path, monkeypatch
):
    small_ready(workspace)
    window._attach(workspace)
    entry = window.queue_controller.store.add(workspace)
    window.queue_controller.store.update(entry["id"], state="running")
    window.queue_controller.active_id = entry["id"]
    monkeypatch.setattr("paranam.queue_controller.process_alive", lambda entry: True)
    other = Workspace.create(tmp_path / "Other rig", "Other rig")
    other.preferences.update(n_train=2, n_validation=1, corners=False)
    other.generate_plan()
    try:
        window.open_project(other.directory)
        window.navigate(2)
        window.rig_ready.setChecked(True)
        assert window.workspace.directory == other.directory
        assert window.record_button.isEnabled()
        assert window.queue_controller.active_id == entry["id"]
        assert read_json(Path(entry["job"]) / "spec.json")["original_project"] == str(
            workspace.directory
        )
    finally:
        window.queue_controller.active_id = None


def test_checkpoint_resume_after_first_epoch_survives_project_edits(
    workspace, tmp_path, qapp
):
    from paranam.queue_controller import QueueController

    small_ready(workspace)
    workspace.preferences["epochs"] = 8
    workspace.save()
    controller = QueueController(tmp_path / "queue")
    entry = controller.store.add(workspace)
    job = Path(entry["job"])
    spec = read_json(job / "spec.json")
    spec["learning"]["trainer"].update(limit_train_batches=8, limit_val_batches=1)
    spec["data"]["common"]["require_input_pre_silence"] = None
    for split in ("train", "train_evaluation"):
        for item in spec["data"][split]:
            item.update(start_seconds=0, stop_seconds=None)
    spec["model"]["net"]["config"]["hypernet"]["hidden_sizes"] = [4]
    frozen = Workspace(Path(spec["project"]))
    for item in spec["data"]["validation"]:
        item.update(
            x_path=frozen.project.train_input, y_path=frozen.project.entries[0].y_path
        )
    atomic_write_json(job / "spec.json", spec)

    def wait_until(predicate, timeout=100):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            qapp.processEvents()
            controller.tick()
            if predicate():
                return
            time.sleep(0.01)
        if controller.process:
            controller.process.kill()
            controller.process.waitForFinished(10000)
        pytest.fail(f"Worker timed out: {training.job_progress(job)}")

    controller.start()
    wait_until(
        lambda: read_json(job / "resume-0.json", {}).get("completed_epoch", 0) >= 1
    )
    assert controller.active_id == entry["id"]
    controller.enabled = False
    controller.pause(entry["id"])
    wait_until(lambda: controller.active_id is None)
    assert training.job_progress(job)["status"] == "paused"
    assert (job / "resume-0.ckpt").is_file()
    workspace.preferences["epochs"] = 91
    workspace.project.user_metadata = {"modeled_by": "A different training"}
    workspace.save()
    controller.store.prepare_resume(entry["id"], batch_size=2, device="cpu")
    controller.start()
    wait_until(lambda: controller.active_id is None)
    progress = training.job_progress(job)
    assert progress["status"] == "completed", progress
    assert read_json(job / "spec.json")["preferences"]["epochs"] == 8
    assert read_json(job / "spec.json")["learning"]["val_dataloader"]["batch_size"] == 1


@pytest.mark.parametrize("kind", ["static", "parametric"])
def test_actual_queue_worker_finishes_and_resumes_its_frozen_instance(
    workspace, tmp_path, qapp, kind
):
    from paranam.queue_controller import QueueController

    small_ready(workspace, kind)
    workspace.project.user_metadata["name"] = "Named Capture"
    workspace.save()
    controller = QueueController(tmp_path / "queue")
    entry = controller.store.add(workspace)
    job = Path(entry["job"])
    spec = read_json(job / "spec.json")
    spec["learning"]["trainer"].update(limit_train_batches=1, limit_val_batches=1)
    spec["data"].setdefault("common", {})["require_input_pre_silence"] = None
    for split in ("train", "train_evaluation"):
        for item in spec["data"][split]:
            item.update(start_seconds=0, stop_seconds=None)
    frozen = Workspace(Path(spec["project"]))
    for item in spec["data"]["validation"]:
        item["x_path"] = frozen.project.train_input
        item["y_path"] = frozen.project.entries[0].y_path
    if kind == "parametric":
        spec["model"]["net"]["config"]["hypernet"]["hidden_sizes"] = [4]
    atomic_write_json(job / "spec.json", spec)
    # A paused-before-start instance is still resumable without a completed epoch.
    controller.store.update(entry["id"], state="paused")
    workspace.project.user_metadata = {"modeled_by": "Changed after enqueue"}
    workspace.preferences["epochs"] = 99
    workspace.save()
    controller.store.prepare_resume(entry["id"])
    controller.start()
    deadline = time.monotonic() + 100
    while controller.active_id and time.monotonic() < deadline:
        qapp.processEvents()
        controller.tick()
        time.sleep(0.02)
    if controller.active_id:
        controller.process.kill()
        controller.process.waitForFinished(10000)
        pytest.fail("Queue worker timed out")
    progress = training.job_progress(job)
    assert progress["status"] == "completed", progress
    assert Path(progress["export"]).is_file()
    assert str(workspace.directory / "models") in progress["export"]
    assert read_json(job / "spec.json")["preferences"]["epochs"] == 2
    exported = read_json(Path(progress["export"]))
    assert exported["metadata"].get("modeled_by") != "Changed after enqueue"
    assert Path(progress["export"]).stem == exported["metadata"]["name"]


def test_queue_idle_refresh_keeps_interactive_column_widths(window):
    """Queue polling must not auto-size the entire table on the GUI thread."""
    from PySide6.QtWidgets import QHeaderView

    table = window.queue_table
    header = table.horizontalHeader()
    assert all(
        header.sectionResizeMode(column) == QHeaderView.ResizeMode.Interactive
        for column in range(table.columnCount())
    )
    # The initial visit auto-fits after the rows are populated, exactly once.
    window.navigate(4)
    assert window._queue_columns_autosized
    table.setColumnWidth(0, 237)
    for _ in range(5):
        window.refresh_training()
    assert table.columnWidth(0) == 237

    # Revisiting Queue cannot undo a manually chosen width.
    window.navigate(0)
    window.navigate(4)
    assert table.columnWidth(0) == 237

    # Refreshing again, including another navigation cycle, never auto-fits.
    table.setColumnWidth(1, 181)
    window.refresh_training()
    window.navigate(0)
    window.navigate(4)
    assert table.columnWidth(1) == 181

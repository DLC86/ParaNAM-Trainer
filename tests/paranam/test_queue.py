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


def enable_snr(workspace):
    workspace.preferences["advanced_parametric"]["rms_loss_normalization"] = True
    workspace.preferences["advanced_parametric"]["snr_weighted_loss_normalization"] = True
    raw_dir = workspace.directory / "captures_raw"
    raw_dir.mkdir()
    playback = np.zeros(64)
    playback[16] = 0.5
    np_to_wav(playback, raw_dir / "playback.wav", rate=48000)
    records = []
    for index, entry in enumerate(
        entry for entry in workspace.project.entries if entry.split == "train"
    ):
        noise = 0.03125 * (index + 1)
        raw_name = f"raw-{index}.wav"
        np_to_wav(np.full(64, noise), raw_dir / raw_name, rate=48000)
        records.append(
            {
                "y_path": entry.y_path,
                "main_playback": "playback.wav",
                "raw": raw_name,
                "preamble_samples": 24,
            }
        )
    atomic_write_json(raw_dir / "manifest.json", {"captures": records})
    workspace.save()
    return raw_dir


def test_snr_only_queue_freezes_raw_noise_without_rms(workspace, tmp_path):
    small_ready(workspace)
    enable_snr(workspace)
    workspace.preferences["advanced_parametric"]["rms_loss_normalization"] = False
    workspace.save()
    queue = TrainingQueue(tmp_path / "queue")
    entry = queue.add(workspace)
    spec = read_json(Path(entry["job"]) / "spec.json")
    assert spec["learning"]["rms_loss_normalization"] is False
    assert spec["learning"]["snr_weighted_loss_normalization"] is True
    powers = spec["learning"]["snr_noise_powers"]
    assert len(powers) == len(spec["data"]["train"])
    assert all(power >= 0.0 for power in powers.values())


def test_failed_queue_job_reuses_resume_as_retry(window, workspace, monkeypatch):
    small_ready(workspace)
    window._attach(workspace)
    controller = window.queue_controller
    entry = controller.store.add(workspace)
    window.selected_job_id = entry["id"]
    controller.store.update(entry["id"], state="failed", error="Temporary error")
    window.refresh_training()
    assert window.resume_button.text() == "Retry"
    assert window.resume_button.isEnabled()
    starts = []
    monkeypatch.setattr(controller, "start", lambda: starts.append(True))
    window.resume_button.click()
    assert starts == [True]
    assert controller.store.get(entry["id"])["state"] == "queued"
    assert window.resume_button.text() == "Resume"


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
def test_retrain_restores_original_job_plan_into_independent_editable_copy(
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
    assert editable.directory != workspace.directory
    assert editable.directory != frozen.directory
    assert editable.directory.exists()
    assert [(e.split, e.params) for e in editable.project.entries] == [
        (e.split, e.params) for e in frozen.project.entries
    ]
    assert editable.preferences["_retrain_source_job"] == str(job.resolve())
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
    assert workspace.project.user_metadata == {"modeled_by": "Later project edits"}
    assert queue.get(entry["id"])["state"] == "completed"
    with pytest.raises(ValueError, match="queued"):
        queue.retrain(new["id"])


def test_retrain_uses_original_captures_after_source_plan_changes(
    workspace, tmp_path
):
    small_ready(workspace)
    queue = TrainingQueue(tmp_path / "queue")
    entry = queue.add(workspace)
    queue.update(entry["id"], state="completed")
    frozen = Workspace(Path(read_json(Path(entry["job"]) / "spec.json")["project"]))
    original_plan = [(e.split, dict(e.params)) for e in frozen.project.entries]

    # Regenerate the SOURCE project with fewer and different captures.
    workspace.preferences.update(n_train=2, n_validation=0, seed=437)
    workspace.generate_plan()
    assert [(e.split, e.params) for e in workspace.project.entries] != original_plan

    editable = queue.retrain(entry["id"])
    assert editable.directory != workspace.directory
    assert [(e.split, e.params) for e in editable.project.entries] == original_plan
    assert editable.training_ready()
    assert not editable.plan_changed()
    assert workspace.project.n_train_lhs == 2
    assert [(e.split, dict(e.params)) for e in frozen.project.entries] == original_plan


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


@pytest.mark.parametrize(
    "kind,snr_mode",
    [("static", None), ("parametric", None), ("parametric", "frozen"), ("parametric", "legacy")],
)
def test_actual_queue_worker_finishes_and_resumes_its_frozen_instance(
    workspace, tmp_path, qapp, kind, snr_mode
):
    from paranam.queue_controller import QueueController

    small_ready(workspace, kind)
    raw_dir = enable_snr(workspace) if snr_mode else None
    workspace.project.user_metadata["name"] = "Named Capture"
    workspace.save()
    controller = QueueController(tmp_path / "queue")
    entry = controller.store.add(workspace)
    job = Path(entry["job"])
    spec = read_json(job / "spec.json")
    if snr_mode:
        powers = spec["learning"]["snr_noise_powers"]
        assert list(powers) == [entry["y_path"] for entry in spec["data"]["train"]]
        assert list(powers.values()) == pytest.approx([
            (0.03125 * (index + 1)) ** 2
            for index in range(len(spec["data"]["train"]))
        ])
        if snr_mode == "legacy":
            del spec["learning"]["snr_noise_powers"]
            del spec["learning"]["rms_loss_normalization"]
        else:
            (raw_dir / "manifest.json").unlink()
    spec["learning"]["trainer"].update(limit_train_batches=1, limit_val_batches=1)
    spec["data"].setdefault("common", {})["require_input_pre_silence"] = None
    for split in ("train", "train_evaluation"):
        for item in spec["data"][split]:
            item.update(start_seconds=0, stop_seconds=None)
    frozen = Workspace(Path(spec["project"]))
    if snr_mode == "legacy":
        del frozen.preferences["advanced_parametric"]["rms_loss_normalization"]
        del spec["preferences"]["advanced_training"]["rms_loss_normalization"]
        frozen.save()
        assert frozen.fingerprint() == spec["fingerprint"]
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
    if snr_mode:
        assert read_json(job / "spec.json")["learning"]["snr_noise_powers"] == powers
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



def test_completed_training_can_extend_epoch_target_without_new_instance(workspace, tmp_path):
    """The existing frozen job, its checkpoint path, and its queue ID survive."""
    small_ready(workspace)
    queue = TrainingQueue(tmp_path / "queue")
    entry = queue.add(workspace)
    job = Path(entry["job"])
    atomic_write_json(job / "resume-0.json", {"completed_epoch": 2})
    queue.update(entry["id"], state="completed")
    with pytest.raises(ValueError, match="already completed"):
        queue.set_total_epochs(entry["id"], 1)
    queue.set_total_epochs(entry["id"], 5)
    spec = read_json(job / "spec.json")
    assert spec["preferences"]["epochs"] == 5
    assert spec["learning"]["trainer"]["max_epochs"] == 5
    assert Workspace(Path(spec["project"])).preferences["epochs"] == 5
    assert read_json(job / "progress.json")["completed_members"] == []
    assert len(queue.entries()) == 1
    assert queue.get(entry["id"])["state"] == "completed"
    queue.set_total_epochs(entry["id"], 2)
    assert read_json(job / "spec.json")["preferences"]["epochs"] == 2


def test_active_training_cannot_change_epoch_target(workspace, tmp_path):
    small_ready(workspace)
    queue = TrainingQueue(tmp_path / "queue")
    entry = queue.add(workspace)
    queue.update(entry["id"], state="running")
    with pytest.raises(ValueError, match="Pause or stop"):
        queue.set_total_epochs(entry["id"], 8)


def test_queue_job_rename_changes_model_metadata_not_just_label(workspace, tmp_path):
    small_ready(workspace)
    store = TrainingQueue(tmp_path / "queue")
    entry = store.add(workspace)
    original = entry["name"]
    spec_path = Path(entry["job"]) / "spec.json"
    before = read_json(spec_path)
    store.rename(entry["id"], "  Better model name  ")
    stored = TrainingQueue(store.directory).get(entry["id"])
    renamed = read_json(spec_path)
    assert renamed["model"]["metadata"]["name"] == "Better model name"
    assert stored["name"] == original
    assert "display_name" not in stored
    assert renamed["fingerprint"] == before["fingerprint"]
    assert renamed["data"] == before["data"]
    assert Workspace(Path(entry["project"])).project.name == workspace.project.name

    store.update(entry["id"], state="running")
    store.rename(entry["id"], "Training in progress")
    assert read_json(spec_path)["model"]["metadata"]["name"] == "Training in progress"
    for invalid in ("", "  ", "bad\nname", "x" * 201):
        with pytest.raises(ValueError, match="name"):
            store.rename(entry["id"], invalid)
    assert read_json(spec_path)["model"]["metadata"]["name"] == "Training in progress"
    with pytest.raises(ValueError, match="no longer exists"):
        store.rename("unknown-id", "A model")


def test_rename_updates_existing_parametric_export_and_nested_metadata(workspace, tmp_path):
    small_ready(workspace)
    store = TrainingQueue(tmp_path / "queue")
    entry = store.add(workspace)
    job = Path(entry["job"])
    member_file = job / "member-0" / "model_parametric.pnam"
    member_file.parent.mkdir(parents=True, exist_ok=True)
    export_file = tmp_path / "final.pnam"
    sample = {
        "architecture": "ParametricSlimmableContainer",
        "metadata": {"name": "Before"},
        "config": {"submodels": [
            {"model": {"metadata": {"name": "Before"}, "config": {}}},
            {"model": {"metadata": {"name": "Before"}, "config": {}}},
        ]},
    }
    atomic_write_json(member_file, sample)
    atomic_write_json(export_file, sample)
    atomic_write_json(job / "progress.json", {"export": str(export_file)})
    store.update(entry["id"], state="completed")
    store.rename(entry["id"], "After rename")
    for path in (member_file, export_file):
        data = read_json(path)
        assert data["metadata"]["name"] == "After rename"
        assert [c["model"]["metadata"]["name"]
                for c in data["config"]["submodels"]] == [
            "After rename", "After rename"
        ]
    assert read_json(job / "progress.json")["export"] == str(export_file)


def test_rename_static_queue_job_updates_first_capture_only(workspace, tmp_path):
    small_ready(workspace, "static")
    store = TrainingQueue(tmp_path / "queue")
    entry = store.add(workspace)
    path = Path(entry["job"]) / "spec.json"
    before = read_json(path)["static_captures"]
    assert before
    store.rename(entry["id"], "Static renamed")
    captures = read_json(path)["static_captures"]
    assert captures[0]["name"] == "Static renamed"
    assert captures[0]["metadata"]["name"] == "Static renamed"
    assert [c["name"] for c in captures[1:]] == [
        c["name"] for c in before[1:]
    ]


def test_queue_rename_context_targets_right_clicked_job(window, workspace, monkeypatch):
    from PySide6.QtCore import QPoint, Qt
    from paranam import queue_ui

    small_ready(workspace)
    window._attach(workspace)
    store = window.queue_controller.store
    first = store.add(workspace)
    second = store.add(workspace)
    window.refresh_training()

    class Action:
        def __init__(self, label):
            self.label = label

        def setEnabled(self, value):
            self.enabled = value

    class ContextMenu:
        def __init__(self, parent):
            self.actions = []

        def addAction(self, label):
            action = Action(label)
            self.actions.append(action)
            return action

        def exec(self, point):
            assert [action.label for action in self.actions] == [
                "Rename…", "Change total epochs…"
            ]
            return self.actions[0]

    class TextInput:
        @staticmethod
        def getText(parent, title, label, **kwargs):
            assert title == "Rename training job"
            return "Renamed from context menu", True

    monkeypatch.setattr(queue_ui, "QMenu", ContextMenu)
    monkeypatch.setattr(queue_ui, "QInputDialog", TextInput)
    table = window.queue_table
    # The job on which the user right-clicks can differ from the selected row.
    assert table.item(0, 0).data(Qt.ItemDataRole.UserRole) == first["id"]
    monkeypatch.setattr(table, "itemAt", lambda _: table.item(0, 0))
    table.selectRow(1)
    window.show_queue_context_menu(QPoint(0, 0))
    assert read_json(Path(first["job"]) / "spec.json")["model"]["metadata"]["name"] == (
        "Renamed from context menu"
    )
    assert read_json(Path(second["job"]) / "spec.json")["model"].get(
        "metadata", {}
    ).get("name") != "Renamed from context menu"
    assert table.item(0, 0).text() == "Renamed from context menu"
    assert table.item(0, 1).text() == first["name"]
    assert window.selected_job_id == first["id"]

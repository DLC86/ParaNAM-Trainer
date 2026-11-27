import json
from pathlib import Path

import numpy as np
import pytest

from nam.capture.project import atomic_write_json, QAModel
from nam.data import np_to_wav
from paranam import training
from paranam.workspace import Workspace, read_json


def static_workspace(workspace):
    workspace.preferences.update(
        capture_type="static", static_size="lite", epochs=2, static_batch_size=1
    )
    workspace.generate_plan()
    workspace.add_static_capture("Lead")
    return workspace


def test_static_names_metadata_and_reopen(workspace):
    ws = static_workspace(workspace)
    groups = ws.static_groups()
    assert len(groups) == 2 and all(len(g) == 2 for g in groups)
    assert not ws.controls() and not ws.plan_changed()
    paths = [e.y_path for e in groups[0]]
    identifier = groups[0][0].capture_id
    ws.rename_static_capture(identifier, "Clean")
    ws.set_static_metadata(identifier, {"modeled_by": "Author", "custom": {"x": 1}})
    assert [e.y_path for e in groups[0]] == paths
    with pytest.raises(ValueError):
        ws.add_static_capture("clean")
    reopened = Workspace(ws.directory)
    assert reopened.static_groups()[0][0].user_metadata["custom"] == {"x": 1}
    assert not reopened.plan_changed()


def test_static_table_and_peak_columns(window, workspace):
    from PySide6.QtWidgets import QHeaderView

    ws = static_workspace(workspace)
    entry = ws.static_groups()[0][0]
    entry.qa = QAModel(
        peak=-4.2, loopback_delay=480, subsample_shift=-0.25, messages=["Level warning"]
    )
    entry.delay = 476
    window._attach(ws)
    window.navigate(2)
    assert window.scene.isHidden()
    assert window.capture_table.rowCount() == 2
    headers = [window.capture_table.horizontalHeaderItem(i).text() for i in range(5)]
    assert headers == ["Name", "Status", "Latency", "Peak (dBFS)", "Tags/Warnings"]
    assert window.capture_table.item(0, 2).text() == "Train: 476.250 samples"
    assert "-4.2" in window.capture_table.item(0, 3).text()
    assert "Level warning" in window.capture_table.item(0, 4).text()
    header = window.capture_table.horizontalHeader()
    assert all(
        header.sectionResizeMode(i) == QHeaderView.ResizeMode.Interactive
        for i in range(5)
    )
    header.resizeSection(0, 275)
    window.refresh_captures()
    assert header.sectionSize(0) == 275


@pytest.mark.parametrize("action", ["cancel", "keep", "delete"])
def test_static_capture_removal_dialog(window, workspace, qapp, action):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QMessageBox
    from nam.capture.export import update_data_json
    from nam.capture.session import raw_paths

    ws = static_workspace(workspace)
    retained, removed = ws.static_groups()
    for entry in retained + removed:
        target = ws.directory / entry.y_path
        target.parent.mkdir(exist_ok=True)
        np_to_wav(np.zeros(64, dtype=np.float32), target, rate=48000)
        entry.status = "captured"
    raw, _ = raw_paths(ws.directory, removed[0].y_path)
    raw.parent.mkdir(exist_ok=True)
    raw.touch()
    ws.save()
    update_data_json(ws.project, ws.directory)
    window._attach(ws)
    window.navigate(2)
    window.select_static_id(removed[0].capture_id)
    window.capture_table.clearSelection()
    assert not window.delete_static_button.isEnabled()
    window.select_static_id(removed[0].capture_id)
    assert window.delete_static_button.isEnabled()

    defaults = []

    def respond():
        dialog = qapp.activeModalWidget()
        checkbox = dialog.checkBox()
        defaults.append(checkbox.isChecked())
        checkbox.setChecked(action != "keep")
        if action == "cancel":
            dialog.button(QMessageBox.StandardButton.Cancel).click()
        else:
            next(button for button in dialog.buttons() if button.text() == "Delete").click()

    QTimer.singleShot(0, respond)
    window.delete_static_button.click()
    assert defaults == [False]
    assert window.capture_table.rowCount() == (2 if action == "cancel" else 1)
    reopened = Workspace(ws.directory)
    remaining = retained + removed if action == "cancel" else retained
    assert [entry.y_path for entry in reopened.project.entries] == [e.y_path for e in remaining]
    assert not reopened.plan_changed()
    data = read_json(ws.directory / "data.json")
    assert {row["y_path"] for split in ("train", "validation") for row in data[split]} == {
        e.y_path for e in remaining
    }
    assert all((ws.directory / entry.y_path).exists() for entry in retained)
    assert all((ws.directory / entry.y_path).exists() == (action != "delete") for entry in removed)
    assert raw.exists()


def test_removing_last_static_capture_persists_empty_plan(workspace):
    ws = static_workspace(workspace)
    for group in ws.static_groups():
        ws.remove_static_capture(group[0].capture_id)
    reopened = Workspace(ws.directory)
    assert reopened.preferences["static_captures"] == []
    assert reopened.static_groups() == []
    assert not reopened.plan_changed()
    assert not reopened.training_ready()
    reopened.add_static_capture("New take")
    assert [entry.index for entry in reopened.project.entries] == [0, 1]


def test_static_capture_removal_preserves_shared_and_external_wavs(workspace, tmp_path):
    ws = static_workspace(workspace)
    retained, removed = ws.static_groups()
    shared = ws.directory / retained[0].y_path
    shared.parent.mkdir(exist_ok=True)
    shared.touch()
    external = tmp_path / "outside-project.wav"
    external.touch()
    removed[0].y_path = retained[0].y_path
    removed[1].y_path = str(external)

    warnings = ws.remove_static_capture(removed[0].capture_id, delete_wavs=True)

    assert len(warnings) == 2
    assert shared.is_file() and external.is_file()
    assert len(Workspace(ws.directory).static_groups()) == 1


def test_metadata_editor_preserves_unknown_fields(qapp):
    from paranam.widgets import MetadataEditor

    editor = MetadataEditor()
    editor.set_metadata({"name": "Model", "extra": {"nested": [1, 2]}})
    emitted = []
    editor.changed.connect(emitted.append)
    editor.emit_metadata()
    assert emitted[-1]["extra"] == {"nested": [1, 2]}


@pytest.mark.parametrize("size", ["full", "lite"])
@pytest.mark.parametrize("optimizer", ["Adam", "AdamW"])
def test_real_static_batch_pause_resume_and_metadata(
    workspace, monkeypatch, size, optimizer
):
    ws = static_workspace(workspace)
    ws.preferences["advanced_static"] = {"optimizer": optimizer}
    ws.preferences["static_size"] = size
    ws.project.user_metadata = {
        "modeled_by": "Tester",
        "gear_type": "amp",
        "custom": {"keep": True},
    }
    ws.project.input_level_dbu, ws.project.output_level_dbu = 18, 12
    first = ws.static_groups()[0][0]
    ws.set_static_metadata(first.capture_id, {"gear_model": "Test amp"})
    x = (0.1 * np.sin(np.arange(6500) * 0.03)).astype(np.float32)
    np_to_wav(x, ws.directory / "test-input.wav", rate=48000)
    for entry in ws.project.entries:
        target = ws.directory / entry.y_path
        target.parent.mkdir(exist_ok=True)
        np_to_wav(np.tanh(2 * x), target, rate=48000)
        entry.status, entry.delay = "captured", 0
    ws.save()
    job = training.make_job(ws)
    spec = read_json(job / "spec.json")
    assert spec["model"]["net"]["name"] == "PackedWaveNet"
    spec["data"]["common"].update(
        x_path="test-input.wav", ny=32, require_input_pre_silence=None
    )
    for split in ("train", "validation", "train_evaluation"):
        for entry in spec["data"][split]:
            entry["x_path"] = "test-input.wav"
            entry["ny"] = 32
            entry.update(start_seconds=0, stop_seconds=None)
    spec["model"]["loss"] = {"val_loss": "esr"}
    spec["learning"]["trainer"].update(limit_train_batches=1, limit_val_batches=1)
    atomic_write_json(job / "spec.json", spec)
    monkeypatch.setattr(
        "nam.models.base._Base._metadata_loudness", lambda *a, **k: -20.0
    )
    monkeypatch.setattr("nam.models.base._Base._metadata_gain", lambda *a, **k: 0.5)
    monkeypatch.chdir(ws.directory)
    if size == "lite":
        training.command(job, "pause")
    else:
        from nam.train.lightning_module import PackedLightningModule as LightningModule

        original = LightningModule.training_step
        requested = []

        def pause_second_epoch(module, batch, index):
            result = original(module, batch, index)
            if module.current_epoch == 1 and not requested:
                requested.append(True)
                training.command(job, "pause")
            return result

        monkeypatch.setattr(LightningModule, "training_step", pause_second_epoch)
    training.run_job(job)
    assert training.job_progress(job)["status"] == "paused"
    assert training.job_progress(job)["completed_epoch"] == (0 if size == "lite" else 1)
    assert not (ws.directory / "models").exists()
    if size == "full":
        snapshot = read_json(
            Path(training.export_saved_best(job, ws.directory / "Paused.nam"))
        )
        assert snapshot["architecture"] == "SlimmableContainer"
        assert snapshot["version"] == "0.7.0"
        assert all(
            child["model"]["version"] == "0.7.0"
            for child in snapshot["config"]["submodels"]
        )
        assert snapshot["metadata"]["name"] == "Capture 1"
        assert snapshot["metadata"]["input_level_dbu"] == 18
        assert snapshot["metadata"]["custom"] == {"keep": True}
        assert snapshot["metadata"]["modeled_by"] == "Tester"
        assert snapshot["metadata"]["training"]["validation_esr_lite"] is not None
    assert training.make_job(ws, resume=True) == job
    training.run_job(job)
    progress = training.job_progress(job)
    assert progress["status"] == "completed", progress
    assert progress["train_esr_full"] is not None
    assert progress["best_train_esr"] <= progress["train_esr_full"]
    assert len(progress["exports"]) == 2
    for path, name in zip(progress["exports"], ("Capture 1", "Lead")):
        model = read_json(Path(path))
        assert Path(path).name == name + ".nam"
        assert model["architecture"] == "SlimmableContainer"
        assert model["version"] == "0.7.0"
        children = model["config"]["submodels"]
        assert all(child["model"]["version"] == "0.7.0" for child in children)
        assert [child["max_value"] for child in children] == [0.5, 1.0]
        assert [
            child["model"]["config"]["layers"][0]["channels"] for child in children
        ] == [3, 8]
        assert [len(child["model"]["weights"]) for child in children] == [1871, 12146]
        assert all(child["model"]["metadata"]["name"] == name for child in children)
        assert all(
            child["model"]["metadata"]["input_level_dbu"] == 18 for child in children
        )
        assert all(
            child["model"]["metadata"]["output_level_dbu"] == 12 for child in children
        )
        assert model["metadata"]["name"] == name
        assert model["metadata"]["custom"] == {"keep": True}
        assert model["metadata"]["modeled_by"] == "Tester"
        assert model["metadata"]["input_level_dbu"] == 18
        assert model["metadata"]["output_level_dbu"] == 12
        summary = model["metadata"]["training"]
        assert summary["validation_reference"] == "full"
        assert summary["validation_esr"] == summary["validation_esr_full"]
        assert summary["validation_esr_lite"] is not None
        assert (
            children[0]["model"]["metadata"]["training"]["validation_esr"]
            == summary["validation_esr_lite"]
        )
    assert (
        read_json(Path(progress["exports"][0]))["metadata"]["gear_model"] == "Test amp"
    )


def test_static_plan_keeps_validation_only_with_the_active_train_take(workspace):
    ws = static_workspace(workspace)
    train, validation = ws.static_groups()[0]
    for entry in (train, validation):
        path = ws.directory / entry.y_path
        path.parent.mkdir(exist_ok=True)
        np_to_wav(np.zeros(64, dtype=np.float32), path, rate=48000)
        entry.status = "captured"
        entry.delay = 0

    ws.generate_plan()
    assert [entry.status for entry in ws.static_groups()[0]] == ["captured", "captured"]

    previous_mode = ws.preferences["input_mode"]
    custom = ws.directory / "another-input.wav"
    np_to_wav(np.zeros(48000, dtype=np.float32), custom, rate=48000)
    ws.preferences.update(input_mode="custom", custom_input=custom.name)
    ws.generate_plan()
    assert [entry.status for entry in ws.static_groups()[0]] == ["pending", "pending"]
    assert validation.y_path in ws.project.unused_recordings

    # Switching back can recover Train, but cannot select a validation from the pool.
    ws.preferences["input_mode"] = previous_mode
    ws.generate_plan()
    current_train, current_validation = ws.static_groups()[0]
    assert current_train.y_path == train.y_path and current_train.status == "captured"
    assert current_validation.status == "pending"

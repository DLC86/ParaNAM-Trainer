import json
from pathlib import Path

import numpy as np
import pytest

from nam.capture.project import atomic_write_json, QAModel
from nam.data import np_to_wav
from paranam import training
from paranam.workspace import Workspace, read_json


def static_workspace(workspace):
    workspace.preferences.update(capture_type="static", static_size="lite", epochs=2)
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
    assert headers == ["Name", "Status", "Latency", "Peak (dBFS)", "Warnings"]
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


def test_metadata_editor_preserves_unknown_fields(qapp):
    from paranam.widgets import MetadataEditor

    editor = MetadataEditor()
    editor.set_metadata({"name": "Model", "extra": {"nested": [1, 2]}})
    emitted = []
    editor.changed.connect(emitted.append)
    editor.emit_metadata()
    assert emitted[-1]["extra"] == {"nested": [1, 2]}


@pytest.mark.parametrize("size", ["full", "lite"])
def test_real_static_batch_pause_resume_and_metadata(workspace, monkeypatch, size):
    ws = static_workspace(workspace)
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
    for split in ("train", "validation"):
        for entry in spec["data"][split]:
            entry["x_path"] = "test-input.wav"
            entry["ny"] = 32
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
    assert training.make_job(ws, resume=True) == job
    training.run_job(job)
    progress = training.job_progress(job)
    assert progress["status"] == "completed", progress
    assert len(progress["exports"]) == 2
    for path, name in zip(progress["exports"], ("Capture 1", "Lead")):
        model = read_json(Path(path))
        assert Path(path).suffix == ".nam"
        assert model["architecture"] == "SlimmableContainer"
        children = model["config"]["submodels"]
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
    assert (
        read_json(Path(progress["exports"][0]))["metadata"]["gear_model"] == "Test amp"
    )

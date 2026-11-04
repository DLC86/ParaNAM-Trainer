import itertools
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nam.capture.params import KnobSpec, SwitchSpec
from nam.capture.planner import plan_captures
from nam.capture.project import atomic_write_json
from paranam import training
from paranam.workspace import Workspace, read_json


def test_default_anchor_is_exact_unique_and_held_out_with_multiple_switches():
    controls = [
        KnobSpec("Gain", 0, 10, 0.5, default=3.25),
        KnobSpec("Tone", 0, 10, 0.5, default=7),
        SwitchSpec("Bright", ("Off", "On"), 1),
        SwitchSpec("Channel", ("A", "B", "C"), 2),
    ]
    train, validation = plan_captures(controls, n_train=24, n_validation=12, seed=19)
    assert (len(train), len(validation)) == (24, 12)
    keys = lambda rows: {tuple(row.params.values()) for row in rows}
    assert len(keys(train)) == 24 and len(keys(validation)) == 12
    assert keys(train).isdisjoint(keys(validation))
    for bright, channel in itertools.product(("Off", "On"), ("A", "B", "C")):
        assert (3.25, 7.0, bright, channel) in keys(train)
        assert (3.25, 7.0, bright, channel) not in keys(validation)
    assert (train, validation) == plan_captures(
        controls, n_train=24, n_validation=12, seed=19
    )
    from nam.capture.planner import settings_on_capture_grid

    specs = tuple(control.to_param_spec() for control in controls)
    default = dict(Gain=3.25, Tone=7, Bright="On", Channel="C")
    assert not settings_on_capture_grid(default, specs)
    assert settings_on_capture_grid({**default, "Gain": 3.5}, specs)


def test_anchored_lhs_keeps_each_stratum_before_grid_collisions():
    controls = [
        KnobSpec("A", 0, 1, 0.000001, 0.17),
        KnobSpec("B", 0, 1, 0.000001, 0.83),
    ]
    train, _ = plan_captures(controls, n_train=8, n_validation=0, seed=8)
    assert {"A": 0.17, "B": 0.83} in [row.params for row in train]
    for name in ("A", "B"):
        assert {min(7, int(row.params[name] * 8)) for row in train} == set(range(8))


def ready_audio(workspace):
    path = workspace.directory / "short.wav"
    sf.write(path, np.zeros(24 * 48000, dtype=np.float32), 48000, subtype="PCM_24")
    workspace.preferences.update(
        input_mode="custom", custom_input="short.wav", batch_size=4
    )
    workspace.generate_plan()
    for entry in workspace.project.entries:
        target = workspace.directory / entry.y_path
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(
            workspace.directory / workspace.project.input_for_split(entry.split), target
        )
        entry.status, entry.delay = "captured", 0
    workspace.save()


def test_a2_cadence_counts_real_batches_and_preserves_optimizer_step_decay(workspace):
    import torch
    from nam.data import Split, init_dataset
    from nam.models.parametric import data_config_from_model
    from nam.train.parametric import (
        _ParametricLightningModule,
        _make_parametric_dataloader,
    )
    from paranam.recipes import a2_update_schedule

    ready_audio(workspace)
    workspace.preferences["advanced_parametric"] = dict(rate_preset="a2_updates")
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    module = _ParametricLightningModule.init_from_config(spec["model"])
    data = data_config_from_model(spec["data"], spec["model"])
    data["common"]["nx"] = module.net.receptive_field
    for split in ("train", "validation"):
        for row in data[split]:
            for key in ("x_path", "y_path"):
                row[key] = str(workspace.directory / row[key])
    dataset = init_dataset(data, Split.TRAIN)
    loader = _make_parametric_dataloader(dataset, spec["learning"]["train_dataloader"])
    schedule = spec["preferences"]["rate_schedule"]
    assert len(loader) == schedule["steps_per_epoch"]
    configured = module.configure_optimizers()
    assert configured["lr_scheduler"]["interval"] == "step"
    optimizer, scheduler = (
        configured["optimizer"],
        configured["lr_scheduler"]["scheduler"],
    )
    initial = [group["lr"] for group in optimizer.param_groups]
    for _ in range(schedule["reference_steps"] - 1):
        optimizer.step()
        scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == initial
    saved_optimizer = optimizer.state_dict()
    saved_scheduler = scheduler.state_dict()
    restored = module.configure_optimizers()
    optimizer, scheduler = restored["optimizer"], restored["lr_scheduler"]["scheduler"]
    optimizer.load_state_dict(saved_optimizer)
    scheduler.load_state_dict(saved_scheduler)
    optimizer.step()
    scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [v * 0.994 for v in initial]
    )
    doubled = {**spec["data"], "train": spec["data"]["train"] * 2}
    twice = a2_update_schedule(doubled, workspace.directory, 4, 8192, 2, 0.004, 0.0004)
    assert twice["steps_per_epoch"] == 2 * schedule["steps_per_epoch"]
    assert twice["reference_steps"] == schedule["reference_steps"]
    assert math.isclose(
        twice["final_hypernetwork_learning_rate"], twice["final_learning_rate"] * 0.1
    )
    dataset.teardown()


def test_rate_preset_preview_persists_and_updates_with_batch(window, workspace):
    ready_audio(workspace)
    window._attach(workspace)
    selector = window.advanced_fields["rate_preset"]
    selector.setCurrentIndex(selector.findData("a2_updates"))
    selector.activated.emit(selector.currentIndex())
    assert not window.advanced_fields["lr_decay"].isEnabled()
    assert "reference" in window.rate_preview.text()
    assert "After 2 epochs" in window.rate_preview.text()
    first = window.rate_preview.text()
    window.batch_size.setValue(2)
    assert window.rate_preview.text() != first
    assert (
        Workspace(workspace.directory).training_options()["rate_preset"] == "a2_updates"
    )
    window.capture_kind.setCurrentIndex(window.capture_kind.findData("static"))
    assert not selector.isEnabled() and window.rate_preview.isHidden()


def test_compare_reset_plain_values_alignment_and_original_defaults(
    window, workspace, qapp
):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractSpinBox
    from paranam.play_ui import ModelPlayDial, PlayDial, ControlValue

    window._attach(workspace)
    window.navigate(5)
    info = dict(
        size_selection=True,
        controls=[dict(name="Gain", min=0, max=10, default=5, positions=[])],
    )
    window._play_model_loaded(info)
    window.show()
    qapp.processEvents()
    dial = window.play_controls.findChild(ModelPlayDial)
    assert dial.width() == 108 and window.play_output_mode.currentData() == "calibrated"
    value = window.play_controls.findChild(ControlValue)
    assert (
        not value.hasFrame()
        and value.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons
    )
    QTest.mouseClick(value.lineEdit(), Qt.MouseButton.LeftButton)
    QTest.keyClick(value.lineEdit(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(value.lineEdit(), "8")
    QTest.keyClick(value.lineEdit(), Qt.Key.Key_Return)
    assert window.play_values == [8]
    QTest.mouseDClick(dial, Qt.MouseButton.LeftButton)
    assert window.play_values == [5]
    window._preserved_play_values = (7,)
    window._play_model_loaded(info)
    qapp.processEvents()
    dial = window.play_controls_layout.itemAt(0).widget().findChild(ModelPlayDial)
    assert window.play_values == [7]
    QTest.mouseDClick(dial, Qt.MouseButton.LeftButton)
    assert window.play_values == [5]
    for gain in (window.play_input, window.play_output):
        gain.setValue(-7)
    gain_dials = [d for d in window.findChildren(PlayDial) if type(d) is PlayDial]
    for gain_dial in gain_dials:
        QTest.mouseDClick(gain_dial, Qt.MouseButton.LeftButton)
    assert window.play_input.value() == window.play_output.value() == 0
    grid = window.play_gain_layout
    assert (
        len({grid.itemAtPosition(0, i).widget().geometry().y() for i in range(1, 5)})
        == 1
    )
    centers = [
        grid.itemAtPosition(1, i).widget().geometry().center().y() for i in range(1, 5)
    ]
    assert max(centers) - min(centers) <= 1


@pytest.mark.parametrize("state", ["paused", "stopped"])
def test_export_best_available_and_completes_without_resuming(
    window, workspace, monkeypatch, qapp, state
):
    for capture in workspace.project.entries:
        path = workspace.directory / capture.y_path
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"recording")
        capture.status, capture.delay = "captured", 0
    workspace.save()
    entry = window.queue_controller.store.add(workspace)
    window.queue_controller.store.update(entry["id"], state=state)
    job = Path(entry["job"])
    atomic_write_json(
        job / "progress.json", dict(status=state, best_esr=0.03, epochs=2)
    )
    window.selected_job_id = entry["id"]
    window.refresh_training()
    assert window.export_best_button.isEnabled()
    target = workspace.directory / "Snapshot.pnam"
    monkeypatch.setattr(window, "choose_save_file", lambda *args: (str(target), ""))

    def export(job, path):
        Path(path).write_text("saved model", encoding="utf8")
        return str(path)

    monkeypatch.setattr(training, "export_saved_best", export)
    window.export_best_model()
    deadline = time.monotonic() + 10
    while window.export_worker and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert window.export_worker is None
    assert target.read_text() == "saved model"
    assert read_json(job / "export-result.json")["ok"]
    assert window.queue_controller.store.get(entry["id"])["state"] == state

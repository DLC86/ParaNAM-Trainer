import hashlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal, Qt, QPoint
from PySide6.QtTest import QTest

from nam.capture.audio import DeviceInfo
from nam.data import np_to_wav
from paranam import wizard, play_ui
from paranam.audio_devices import Capabilities
from paranam.play_ui import PlayDial, ModelPlayDial
from paranam.signals import asset_path
from paranam.training_queue import TrainingQueue
from paranam.workspace import Workspace


def configure_audio(window, monkeypatch):
    device = DeviceInfo(0, "Test interface", "Test", 4, 6, 48000)
    caps = Capabilities(
        tuple(f"Input {i}" for i in range(1, 5)),
        tuple(f"Output {i}" for i in range(1, 7)),
        (128, 256, 2048),
    )
    monkeypatch.setattr(wizard, "list_devices", lambda **kw: [device])
    monkeypatch.setattr(wizard, "capabilities", lambda *args: caps)
    monkeypatch.setattr("paranam.audio_devices.capabilities", lambda *args: caps)
    window.refresh_devices()
    for field, value in (
        (window.instrument_input, 3),
        (window.monitor_left, 3),
        (window.monitor_right, 4),
    ):
        field.setCurrentIndex(field.findData(value))
    return device


def test_last_setup_survives_project_switch_and_application_restart(
    window, workspace, tmp_path, monkeypatch
):
    window._attach(workspace)
    configure_audio(window, monkeypatch)
    window._set_capture_level(-12)
    window._apply_level_calibration(window.input_calibration, window.input_dbu, 6.125)
    window._apply_level_calibration(window.output_calibration, window.output_dbu, 12.5)
    window.apply_instrument_calibration(18.123)
    other = Workspace.create(tmp_path / "Other project", "Other project")
    window.open_project(other.directory)
    assert window.input_dbu.value() == 6.125
    assert window.workspace.project.input_level_dbu is None
    assert window.instrument_reference() == 18.123
    assert window.capture_level.level() == -12
    assert window.audio_settings().instrument_input_channel == 3
    restarted = wizard.WizardWindow()
    try:
        assert restarted.device.currentData().name == "Test interface"
        assert restarted.instrument_input.currentData() == 3
        assert restarted.monitor_right.currentData() == 4
        assert restarted.input_dbu.value() == 6.125
        assert restarted.instrument_dbu.value() == 18.123
        assert restarted.capture_level.level() == -12
    finally:
        restarted.close()


def test_file_pickers_share_last_folder_and_cancelling_keeps_it(
    window, tmp_path, monkeypatch
):
    folder = tmp_path / "Folder"
    folder.mkdir()
    seen = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getExistingDirectory", lambda *args: str(folder)
    )
    window.choose_directory("Open")

    def open_file(*args):
        seen.append(args[2])
        return str(folder / "Input.wav"), ""

    monkeypatch.setattr("PySide6.QtWidgets.QFileDialog.getOpenFileName", open_file)
    window.choose_open_file("Input", "WAV (*.wav)")

    def save_file(*args):
        seen.append(str(Path(args[2]).parent))
        return "", ""

    monkeypatch.setattr("PySide6.QtWidgets.QFileDialog.getSaveFileName", save_file)
    window.choose_save_file(
        "Export", tmp_path / "elsewhere" / "Model.pnam", "PNAM (*.pnam)"
    )
    assert seen == [str(folder), str(folder)]
    assert window.dialog_directory() == str(folder)


def test_reusing_live_setup_does_not_rewrite_historical_capture_calibration(
    window, workspace, monkeypatch
):
    window._attach(workspace)
    configure_audio(window, monkeypatch)
    window._apply_level_calibration(window.input_calibration, window.input_dbu, 18.125)
    other = Workspace.create(
        workspace.directory.parent / "Captured rig", "Captured rig"
    )
    other.generate_plan()
    for entry in other.project.entries:
        target = other.directory / entry.y_path
        target.parent.mkdir(exist_ok=True)
        np_to_wav(np.zeros(100, dtype=np.float32), target, rate=48000)
        entry.status = "captured"
    other.project.input_level_dbu = 7.0
    other.save()
    window.open_project(other.directory)
    assert window.input_dbu.value() == 18.125
    assert window.workspace.project.input_level_dbu == 7.0
    assert Workspace(other.directory).project.input_level_dbu == 7.0
    assert window.playback_project().input_level_dbu == 18.125
    assert not window.workspace.plan_changed()


def test_export_names_preserve_unicode_and_safely_replace_invalid_filename_characters():
    from paranam.model_names import model_basename

    assert model_basename("Named Capture") == "Named Capture"
    assert model_basename("音 Modèle") == "音 Modèle"
    assert model_basename("Gain: 1/2") == "Gain_ 1_2"
    assert model_basename("CON") == "_CON"
    assert len((model_basename("音" * 200) + ".pnam").encode("utf8")) <= 255


def test_tts_input_is_bundled_and_stays_selected_for_queue_and_retrain(
    workspace, tmp_path
):
    workspace.preferences["input_mode"] = "tts"
    workspace.generate_plan()
    assert workspace.project.train_input == "input_tts_v11.wav"
    assert workspace.validation_matches()
    assert (
        hashlib.sha256(
            (workspace.directory / workspace.project.train_input).read_bytes()
        ).digest()
        == hashlib.sha256(asset_path("tts-input-v11.wav").read_bytes()).digest()
    )
    for entry in workspace.project.entries:
        path = workspace.directory / entry.y_path
        path.parent.mkdir(exist_ok=True)
        np_to_wav(np.zeros(100, dtype=np.float32), path, rate=48000)
        entry.status = "captured"
    workspace.save()
    queue = TrainingQueue(tmp_path / "Queue")
    entry = queue.add(workspace)
    queue.update(entry["id"], state="failed")
    editable = queue.retrain(entry["id"])
    assert editable.preferences["input_mode"] == "tts"
    assert editable.project.train_input == "input_tts_v11.wav"
    assert not editable.plan_changed()


def test_retrain_accepts_started_states_and_omits_unrecorded_suggestions(
    workspace, tmp_path
):
    from tests.paranam.test_queue import small_ready
    from paranam.workspace import read_json

    small_ready(workspace)
    queue = TrainingQueue(tmp_path / "Queue")
    entry = queue.add(workspace)
    with pytest.raises(ValueError, match="queued"):
        queue.retrain(entry["id"])
    original = Workspace(Path(read_json(Path(entry["job"]) / "spec.json")["project"]))
    original.project.entries.append(
        original.project.entries[0].model_copy(
            update={
                "index": 100,
                "status": "pending",
                "y_path": "captures/not-yet-recorded.wav",
            }
        )
    )
    original.save()
    for state in (
        "running",
        "paused",
        "stopped",
        "failed",
        "interrupted",
        "needs_captures",
        "completed",
    ):
        queue.update(entry["id"], state=state)
        editable = queue.retrain(entry["id"])
        assert editable.training_ready()
        assert all(capture.status == "captured" for capture in editable.project.entries)
        assert len(editable.project.entries) == 3


@pytest.mark.parametrize("dial_class", [PlayDial, ModelPlayDial])
def test_relative_dial_release_never_maps_cursor_angle(qapp, dial_class):
    dial = dial_class()
    dial.setRange(0, 10000)
    dial.setValue(5000)
    dial.resize(92, 92)
    dial.show()
    try:
        QTest.mousePress(dial, Qt.MouseButton.LeftButton, pos=QPoint(46, 46))
        QTest.mouseMove(dial, QPoint(46, 20))
        before = dial.value()
        QTest.mouseRelease(dial, Qt.MouseButton.LeftButton, pos=QPoint(1, 90))
        assert dial.value() == before
        assert not dial.isSliderDown()
    finally:
        dial.close()


def test_capture_table_scrolls_and_resizes_without_changing_locked_selection(
    window, workspace, qapp
):
    window._attach(workspace)
    window.navigate(2)
    window.show()
    qapp.processEvents()
    table = window.capture_table
    table.setRowCount(100)
    table.selectRow(0)
    table.selection_locked = True
    before = table.horizontalHeader().sectionSize(0)
    QTest.mouseClick(
        table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=table.visualRect(table.model().index(1, 0)).center(),
    )
    assert table.currentRow() == 0
    table.horizontalHeader().resizeSection(0, before + 40)
    assert table.horizontalHeader().sectionSize(0) == before + 40
    table.verticalScrollBar().setValue(table.verticalScrollBar().maximum())
    assert table.verticalScrollBar().value() > 0
    table.selection_locked = False


def test_static_progress_reaches_one_for_each_recording(window, workspace, monkeypatch):
    workspace.preferences["capture_type"] = "static"
    workspace.generate_plan()
    window._attach(workspace)
    configure_audio(window, monkeypatch)
    window.navigate(2)
    window.rig_ready.setChecked(True)
    monkeypatch.setattr(window, "save_audio", lambda: None)
    monkeypatch.setattr(wizard, "timebase_problem", lambda *args: None)
    progress = []

    class Session:
        def capture_entry(self, entry, progress=None, **kwargs):
            for value in (0, 0.5, 1):
                progress(value)
            return SimpleNamespace(messages=[])

    monkeypatch.setattr(window, "_get_session", lambda: Session())
    monkeypatch.setattr(
        window, "run_audio", lambda action, done: action(progress.append, None)
    )
    window.record()
    assert (
        progress.count(1) == 4
    )  # Stream completion plus explicit completed signal, twice.
    assert progress[-1] == 1


def test_live_play_changes_size_buffer_and_model_with_automatic_restart(
    window, workspace, tmp_path, qapp, monkeypatch
):
    from nam.models.parametric import HyperA2, export_parametric

    window._attach(workspace)
    configure_audio(window, monkeypatch)
    paths = []
    for index in range(2):
        target = tmp_path / f"Model {index}"
        target.mkdir()
        model = HyperA2.init_from_config(
            dict(
                sample_rate=48000,
                params=[dict(name="Gain", type="continuous", min=0, max=10, default=5)],
                hypernet=dict(hidden_sizes=[4], mode="full", selector={}),
            )
        )
        export_parametric(model, target)
        paths.append(str(target / "model.pnam"))

    class Worker(QObject):
        failed = Signal(str)
        metered = Signal(dict)
        finished = Signal()

        def __init__(self, path, device, project, instrument, lite, parent):
            super().__init__(parent)
            self.path, self.project, self.lite = path, project, lite
            self.stopping = threading.Event()
            self.settings = {}

        def start(self):
            pass

    monkeypatch.setattr(play_ui, "PlayWorker", Worker)

    def events_until(predicate):
        deadline = time.monotonic() + 15
        while not predicate() and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert predicate()

    window.settings.setValue("play_models", paths)
    window.refresh_play_models(paths[0])
    events_until(lambda: window.play_loader is None)
    assert (window.play_input.minimum(), window.play_input.maximum()) == (-20, 20)
    assert (
        window.play_output.minimum(),
        window.play_output.maximum(),
        window.play_output.value(),
    ) == (-20, 20, 0)
    window._play_parameter(0, 7)
    window.start_playback()
    first = window.play_worker
    try:
        assert (
            window.play_model.isEnabled()
            and window.play_size.isEnabled()
            and window.play_buffer.isEnabled()
        )
        window.play_size.setCurrentIndex(1)
        assert first.stopping.is_set()
        first.finished.emit()
        events_until(lambda: window.play_worker is not None)
        second = window.play_worker
        assert second.lite and second.settings["parameters"] == (7,)
        index = window.play_buffer.findData(256)
        window.play_buffer.setCurrentIndex(index)
        assert second.stopping.is_set()
        second.finished.emit()
        third = window.play_worker
        assert third is not second and third.project.audio.blocksize == 256
        window.play_model.setCurrentIndex(window.play_model.findData(paths[1]))
        assert third.stopping.is_set()
        third.finished.emit()
        events_until(lambda: window.play_worker is not None)
        assert window.play_worker.path == paths[1]
        assert window.play_worker.settings["parameters"] == (5,)
    finally:
        window.stop_playback()
        if window.play_worker:
            window.play_worker.finished.emit()
        events_until(lambda: window.play_loader is None)

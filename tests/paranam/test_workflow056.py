import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton

from paranam.ir import (
    BUNDLED,
    HOP,
    IRProcessor,
    MonitorIR,
    kernel_from_taps,
    prepare_ir,
)
from paranam.play_ui import ControlValue, SlideSwitch, SwitchPositions
from paranam.workspace import Workspace
from tests.paranam.test_queue import small_ready


def test_interface_routes_survive_disconnect_other_device_and_new_indices(
    window, monkeypatch
):
    from paranam import wizard
    from paranam.audio_devices import Capabilities

    def device(name, index):
        return SimpleNamespace(
            name=name,
            index=index,
            host_api="Test",
            max_input_channels=8,
            max_output_channels=8,
        )

    a, b = device("Interface A", 1), device("Interface B", 2)
    available = [a, b]
    monkeypatch.setattr(wizard, "list_devices", lambda **kwargs: available)
    monkeypatch.setattr(
        wizard,
        "capabilities",
        lambda info: Capabilities(
            tuple(f"In {i}" for i in range(8)),
            tuple(f"Out {i}" for i in range(8)),
            (64, 128, 512),
        ),
    )
    window.refresh_devices()
    for field, value in (
        (window.input_channel, 5),
        (window.output_channel, 6),
        (window.loop_input, 7),
        (window.loop_output, 8),
        (window.instrument_input, 4),
        (window.monitor_left, 1),
        (window.monitor_right, 2),
        (window.buffer, 512),
    ):
        field.setCurrentIndex(field.findData(value))
    window.loop_enabled.setChecked(False)
    window._playback_blocksize = 64
    window.remember_setup()
    saved_a = window.remembered_audio_route(a)
    assert saved_a["loopback_input_channel"] == 7
    available[:] = [b]
    window.refresh_devices()
    assert window.device.currentData() is None
    window.device.setCurrentIndex(
        0
    )  # Selecting the fallback is an explicit user choice.
    window.input_channel.setCurrentIndex(window.input_channel.findData(3))
    window.buffer.setCurrentIndex(window.buffer.findData(128))
    available.clear()
    window.refresh_devices()
    assert window.device.currentData() is None
    assert window.remembered_audio_route(a) == saved_a
    available[:] = [device("Interface A", 42), device("Interface B", 51)]
    window.refresh_devices()
    assert window.device.currentData().name == "Interface B"
    assert window.input_channel.currentData() == 3
    window.device.setCurrentIndex(0)
    assert window.input_channel.currentData() == 5
    assert window.output_channel.currentData() == 6
    assert window.loop_input.currentData() == 7
    assert window.loop_output.currentData() == 8
    assert window.instrument_input.currentData() == 4
    assert window.monitor_left.currentData() == 1
    assert window.monitor_right.currentData() == 2
    assert window.buffer.currentData() == 512
    assert window._playback_blocksize == 64
    assert not window.loop_enabled.isChecked()
    window.device.setCurrentIndex(1)
    assert window.input_channel.currentData() == 3
    assert window.buffer.currentData() == 128
    window.settings.sync()
    restarted = wizard.WizardWindow()
    try:
        assert restarted.input_channel.currentData() == 3
        assert restarted.buffer.currentData() == 128
        restarted.device.setCurrentIndex(0)
        assert restarted.input_channel.currentData() == 5
        assert restarted.output_channel.currentData() == 6
        assert restarted.buffer.currentData() == 512
    finally:
        restarted.close()


def test_slide_switch_positions_and_value_rows_align_with_knobs(
    window, workspace, qapp
):
    window._attach(workspace)
    window.navigate(5)
    window._play_model_loaded(
        dict(
            size_selection=True,
            controls=[
                dict(name="Gain", min=0, max=10, default=5, positions=[]),
                dict(
                    name="Mode",
                    min=0,
                    max=2,
                    default=0,
                    positions=["Vintage", "Normal", "Modern"],
                ),
            ],
        )
    )
    window.show()
    qapp.processEvents()
    slider = window.play_controls.findChild(SlideSwitch)
    positions = window.play_controls.findChild(SwitchPositions)
    assert slider.minimumWidth() == slider.maximumWidth()
    assert slider.width() < 240
    assert slider.width() == positions.width()
    value = window.play_controls.findChild(ControlValue)
    assert (
        abs(
            positions.mapTo(window.play_controls, positions.rect().center()).y()
            - value.mapTo(window.play_controls, value.rect().center()).y()
        )
        <= 1
    )
    for index in (2, 0, 1):
        QTest.mouseClick(
            slider,
            Qt.MouseButton.LeftButton,
            pos=QPoint(round(slider.position_x(index)), 54),
        )
        assert window.play_values == [5, index]
    QTest.keyClick(slider, Qt.Key.Key_Right)
    assert window.play_values == [5, 2]
    QTest.mouseClick(
        positions,
        Qt.MouseButton.LeftButton,
        pos=QPoint(round(slider.position_x(0)), 15),
    )
    assert window.play_values == [5, 0]


def test_ir_loader_keeps_previous_valid_ir_on_failure(
    window, workspace, qapp, tmp_path
):
    import time

    window._attach(workspace)

    def finish():
        deadline = time.monotonic() + 10
        while window.ir_loader and time.monotonic() < deadline:
            qapp.processEvents()
            QTest.qWait(10)
        assert window.ir_loader is None

    window.play_ir.setCurrentIndex(1)
    finish()
    first = window._ir_kernel
    assert first is not None
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not a WAV")
    window.play_ir.addItem("Broken", str(broken))
    window.play_ir.setCurrentIndex(window.play_ir.count() - 1)
    finish()
    assert window._ir_kernel is first
    assert window.play_ir.currentIndex() == 1
    assert "could not be loaded" in window.play_ir_status.text()
    window.play_ir.setCurrentIndex(0)
    assert window._ir_kernel is None and window._ir_processor is None


@pytest.mark.parametrize("stereo", [False, True])
def test_partition_convolution_matches_direct_with_arbitrary_callbacks(stereo):
    rng = np.random.default_rng(42)
    taps = rng.normal(0, 0.02, (1741, 2 if stereo else 1))
    signal = rng.normal(0, 0.1, 4000)
    stream = np.r_[signal, np.zeros(len(taps) + HOP)]
    processor = IRProcessor(kernel_from_taps(taps))
    blocks = []
    offset = 0
    sizes = [1, 17, 64, 137, 256, 511, 1024]
    while offset < len(stream):
        size = sizes[len(blocks) % len(sizes)]
        blocks.append(processor.process(stream[offset : offset + size]))
        offset += size
    actual = np.concatenate(blocks)
    for channel in range(2):
        reference = np.r_[
            np.zeros(HOP), np.convolve(stream, taps[:, channel if stereo else 0])
        ][: len(stream)]
        np.testing.assert_allclose(actual[:, channel], reference, atol=2e-7, rtol=2e-5)


def test_ir_normalization_resampling_and_invalid_files(tmp_path):
    path = tmp_path / "impulse.wav"
    sf.write(path, np.r_[0.5, np.zeros(200)], 24000, subtype="FLOAT")
    kernel = prepare_ir(path, 48000)
    assert kernel.rate == 48000
    assert np.isfinite(kernel.spectra).all()
    sf.write(path, np.zeros(100), 48000)
    with pytest.raises(ValueError, match="silence"):
        prepare_ir(path)
    sf.write(path, np.ones((100, 3)), 48000)
    with pytest.raises(ValueError, match="mono/stereo"):
        prepare_ir(path)
    sf.write(path, np.array([np.nan]), 48000, subtype="FLOAT")
    with pytest.raises(ValueError, match="non-finite"):
        prepare_ir(path)


def test_bundled_tone3000_irs_are_complete_and_unchanged():
    manifest = json.loads((BUNDLED / "manifest.json").read_text())
    assert len(manifest["files"]) == 6
    assert "TONE3000" in (BUNDLED / "LICENSE.txt").read_text()
    for name, entry in manifest["files"].items():
        assert (
            hashlib.sha256((BUNDLED / name).read_bytes()).hexdigest() == entry["sha256"]
        )
        kernel = prepare_ir(BUNDLED / name)
        assert kernel.rate == 48000


def test_live_ir_switch_crossfades_and_does_not_reset_on_settings_edits():
    processor = IRProcessor(kernel_from_taps(np.array([0.25])))
    monitor = MonitorIR(48000)
    signal = np.ones(256, dtype=np.float32)
    np.testing.assert_array_equal(monitor.process(signal, None), np.ones((256, 2)))
    outputs = [monitor.process(signal, processor) for _ in range(8)]
    np.testing.assert_allclose(outputs[-1], 0.25)
    for _ in range(6):
        output = monitor.process(signal, None)
    np.testing.assert_array_equal(output, np.ones((256, 2)))


def test_typed_control_commits_only_enter_or_focus_loss(qapp):
    host = QWidget()
    layout = QVBoxLayout(host)
    value = ControlValue()
    value.setRange(0, 10)
    value.setValue(3)
    button = QPushButton("Elsewhere")
    layout.addWidget(value)
    layout.addWidget(button)
    host.show()
    qapp.processEvents()
    changed = []
    value.valueChanged.connect(changed.append)
    value.setFocus()
    value.selectAll()
    QTest.keyClicks(value, "7")
    assert value.value() == 3 and not changed
    QTest.keyClick(value, Qt.Key.Key_Return)
    assert value.value() == 7
    value.selectAll()
    QTest.keyClicks(value, "5")
    assert value.value() == 7
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert value.value() == 5
    host.close()


def test_open_other_project_and_retrain_while_another_job_runs(
    window, workspace, tmp_path, monkeypatch
):
    small_ready(workspace)
    workspace.project.user_metadata = {"name": "My NAM"}
    workspace.save()
    window._attach(workspace)
    store = window.queue_controller.store
    running = store.add(workspace)
    completed = store.add(workspace)
    store.update(running["id"], state="running")
    store.update(completed["id"], state="completed")
    window.queue_controller.active_id = running["id"]
    monkeypatch.setattr("paranam.queue_controller.process_alive", lambda entry: True)
    try:
        other = Workspace.create(tmp_path / "Other", "Other")
        window.refresh_training()
        assert window.open_button.isEnabled()
        window.open_project(other.directory)
        assert window.workspace.directory == other.directory
        window.selected_job_id = completed["id"]
        window.refresh_training()
        assert window.queue_table.horizontalHeaderItem(0).text() == "Name"
        assert window.queue_table.item(0, 0).text() == "My NAM"
        assert window.retrain_button.isEnabled()
        window.retrain_button.click()
        assert window.workspace.directory == workspace.directory
        assert window.pages.currentIndex() == 3
        assert window.queue_controller.active_id == running["id"]
        assert len(store.entries()) == 2
    finally:
        window.queue_controller.active_id = None


def test_compare_meter_readouts_and_layout(window, workspace, qapp):
    window._attach(workspace)
    window.navigate(5)
    window.show()
    qapp.processEvents()
    grid = window.play_gain_layout
    assert [
        grid.itemAtPosition(0, column).widget().text() for column in range(1, 5)
    ] == ["Input", "Model size", "Output mode", "Output"]
    window._play_metered(
        dict(
            input_peak=0.1,
            output_peak=0.5,
            output_peaks=[0.25, 0.5],
            clipping=False,
            dropout=False,
        )
    )
    assert window.play_input_peak.text() == "-20.0 dBFS"
    assert window.play_output_peak.text() == "-6.0 dBFS"
    assert not window.play_meter.isVisible()
    assert not window.play_meter.text()
    assert window.play_output_meter.levels == pytest.approx([-12.0412, -6.0206])
    qapp.processEvents()
    for label in (window.play_input_peak, window.play_output_peak):
        assert label.width() >= label.fontMetrics().horizontalAdvance("-120.0 dBFS") + 8
        assert label.height() >= label.fontMetrics().height() + 8


def test_callback_ir_is_stereo_and_does_not_filter_reamp_or_raw_input_meter(
    qapp, monkeypatch
):
    import sounddevice
    from nam.capture.project import CaptureProject, AudioSettingsModel
    from paranam import playback

    class Model:
        info = {}

        def __init__(self, *args):
            pass

        def calibration(self, *args):
            return 1, 1

        def process(self, samples):
            return samples

        def parameters(self, values):
            pass

        def close(self):
            pass

    monkeypatch.setattr(playback, "NativeModel", Model)
    project = CaptureProject(
        name="Test",
        knobs=[],
        audio=AudioSettingsModel(
            instrument_input_channel=3, monitor_output_left=3, monitor_output_right=4
        ),
    )
    device = SimpleNamespace(
        index=0, max_input_channels=4, max_output_channels=6, host_api="Test"
    )
    worker = playback.PlayWorker("unused.nam", device, project)
    worker.settings.update(
        input_db=-6, ir_processor=IRProcessor(kernel_from_taps(np.array([[0.5, 0.25]])))
    )
    outputs, meters, errors = [], [], []
    worker.metered.connect(meters.append)
    worker.failed.connect(errors.append)

    class Stream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def __enter__(self):
            incoming = np.zeros((4800, 4), dtype=np.float32)
            incoming[:, 2] = 0.2
            outgoing = np.empty((4800, 6), dtype=np.float32)
            self.callback(incoming, outgoing, 4800, None, False)
            outputs.append(outgoing.copy())
            worker.stopping.set()
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(sounddevice, "Stream", Stream)
    worker.run()
    assert not errors
    gain = 10 ** (-6 / 20)
    assert outputs[0][-1, 0] == pytest.approx(0.2 * gain)
    assert outputs[0][-1, 2] == pytest.approx(0.2 * gain * 0.5)
    assert outputs[0][-1, 3] == pytest.approx(0.2 * gain * 0.25)
    assert meters[-1]["input_peak"] == pytest.approx(0.2)
    assert meters[-1]["output_peaks"] == pytest.approx(
        [0.2 * gain * 0.5, 0.2 * gain * 0.25]
    )
    assert not outputs[0][:, [1, 4, 5]].any()


def test_new_project_defaults_and_existing_saved_settings_are_preserved(
    window, tmp_path
):
    fresh = Workspace.create(tmp_path / "Fresh", "Fresh")
    options = fresh.training_options()
    assert fresh.preferences["epochs"] == 200
    assert fresh.preferences["input_mode"] == "trunc"
    assert fresh.project.train_input == "input_trunc.wav"
    assert fresh.training_batch_size() == 16
    assert options["hypernetwork_recipe"] == "HyperA2"
    assert options["rate_preset"] == "a2_updates"
    assert options["optimizer"] == "Adam"
    assert options["hypernetwork_mode"] == "low_rank"
    assert not options["condition_dilated_convolutions"]
    assert options["learning_rate"] == 0.004
    assert options["hypernetwork_learning_rate"] == 0.001
    window._attach(fresh)
    assert window.epochs.value() == 200
    assert window.advanced_fields["hypernetwork_recipe"].currentData() == "HyperA2"
    assert window.advanced_fields["rate_preset"].currentData() == "a2_updates"
    fresh.preferences.update(
        epochs=250,
        batch_size=16,
        advanced_parametric={"hypernetwork_recipe": "HyperA2", "rate_preset": "manual"},
    )
    fresh.save()
    restored = Workspace(fresh.directory)
    assert restored.preferences["epochs"] == 250
    assert restored.training_batch_size() == 16
    assert restored.training_options()["hypernetwork_recipe"] == "HyperA2"
    assert restored.training_options()["rate_preset"] == "manual"
    assert restored.training_options("static")["optimizer"] == "Adam"

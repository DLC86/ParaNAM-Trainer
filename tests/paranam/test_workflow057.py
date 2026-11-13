"""Regression coverage for the 0.5.7 routing, audition and training UI."""

from types import SimpleNamespace
import threading
from contextlib import nullcontext

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from nam.capture.audio import DeviceInfo
from nam.capture.params import KnobSpec, SwitchSpec
from nam.capture.project import AudioSettingsModel, CaptureProject
from paranam.audio_devices import Capabilities
from paranam.batch_timing import BatchTiming
from paranam.esr_chart import ESRChart
from paranam.ir import prepare_ir
from paranam.preview import PreviewLoop, preview_files, verify_bundled_previews
from paranam.playback import PlayWorker, validate_route
from paranam.training_log import epoch_line


def test_controls_wrap_and_pages_expand_with_the_window(window, workspace, qapp):
    window._attach(workspace)
    window.navigate(1)
    window.show()
    qapp.processEvents()
    workspace.draft_controls = [
        KnobSpec(f"Knob {i}", 0, 10).to_dict() for i in range(8)
    ]
    window.refresh_controls()
    # Recomputing very large knob grids must never overflow QSpinBox's int.
    assert window.n_train.maximum() <= 1000000
    window.resize(1150, 850)
    qapp.processEvents()
    assert window.control_strip._bounds(8).top() > window.control_strip._bounds(0).top()
    assert window.control_strip.height() >= window.control_strip._bounds(8).bottom()
    narrow_width = window.control_card.width()
    window.resize(2800, 1700)
    qapp.processEvents()
    assert window.control_card.width() > max(narrow_width, 1400)
    assert window.advanced_fields["learning_rate"].maximumWidth() <= 260
    assert window.signal_mode.maximumWidth() <= 560


def test_project_header_name_and_action_bar_dividers(window, workspace, qapp):
    window._attach(workspace)
    window.navigate(1)
    window.show()
    qapp.processEvents()
    header = window.project_label.text()
    assert "My rig" in header
    assert "color: #ffffff" in header
    assert "font-size: 19px" in header
    assert "font-weight: 700" in header
    assert "·  Automatically saved" in header
    assert "Saved locally" not in header
    assert window.header_divider.height() == 1
    assert window.footer_divider.height() == 1
    assert "#515151" in window.header_divider.styleSheet()
    assert "#515151" in window.footer_divider.styleSheet()
    content = window.pages.parentWidget().layout()
    assert content.itemAt(0).layout() is window.header_bar
    assert window.header_bar.itemAt(0).layout() is window.header_label_row
    assert window.header_bar.indexOf(window.header_divider) == 1
    assert any(content.itemAt(i).layout() is window.page_body for i in range(content.count()))
    assert window.page_body.indexOf(window.pages) == 0
    assert content.itemAt(content.count() - 1).layout() is window.footer_bar
    assert window.footer_bar.indexOf(window.footer_divider) == 0
    assert window.footer_bar.itemAt(1).layout() is not None
    assert window.header_bar.spacing() == window.footer_bar.spacing() == 6

    # Neither rule should stop at the 30px inset of the text, pages or buttons.
    sidebar = window.findChild(type(window.header_divider), "sidebar")
    central = window.centralWidget()
    for rule in (window.header_divider, window.footer_divider):
        assert rule.mapTo(central, QPoint(0, 0)).x() == sidebar.width()
        assert rule.width() == central.width() - sidebar.width()
    assert window.header_label_row.contentsMargins().left() == 30
    assert window.page_body.contentsMargins().right() == 30
    assert window.footer_bar.itemAt(1).layout().contentsMargins().left() == 30

    window.resize(1440, 900)
    qapp.processEvents()
    for rule in (window.header_divider, window.footer_divider):
        assert rule.width() == central.width() - sidebar.width()


def test_tts_logo_uses_exact_unmodified_256px_artwork():
    import hashlib
    from PIL import Image
    from paranam.signals import asset_path

    source = asset_path("TTSLogo.png")
    with Image.open(source) as artwork:
        assert artwork.size == (256, 256)
        assert artwork.mode == "RGBA"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "65db1302bd78d49c825e2e3625dc0342c2b20312ebe0ac8435d4e702379f3dd7"
    )


def test_add_control_card_matches_knob_tile_and_arc_sweep(window, workspace):
    from paranam.widgets import KNOB_START_DEGREES, KNOB_SWEEP_DEGREES

    window._attach(workspace)
    window.navigate(1)
    scene = window.control_strip
    scene.resize(900, 220)
    tile = scene._bounds(0)
    add = scene._add_bounds()
    assert (add.width(), add.height()) == (tile.width(), tile.height())
    assert KNOB_START_DEGREES == 210.0
    assert KNOB_SWEEP_DEGREES == 300.0


def test_classic_switch_position_buttons_preserve_default(window, workspace):
    workspace.draft_controls = [SwitchSpec("Mode", ("A", "B", "C"), 1).to_dict()]
    window._attach(workspace)
    window.switch_positions.setCurrentRow(0)
    window.switch_down_button.click()
    assert [item.text() for item in window._switch_items()] == ["B", "A", "C"]
    assert window.switch_default.currentText() == "B"
    window.switch_add_button.click()
    assert window.switch_positions.count() == 4
    window.switch_delete_button.click()
    assert window.switch_positions.count() == 3


def test_inline_switch_reorder_changes_neural_order_and_keeps_default(window, workspace):
    switch = SwitchSpec("Mode", ("A", "B", "C"), 1, option_indices=(2, 0, 1))
    workspace.draft_controls = [switch.to_dict()]
    workspace.preferences.update(n_train=3, n_validation=3)
    workspace.generate_plan()
    window._attach(workspace)
    assert window.control_pages.currentIndex() == 1
    assert [item.text() for item in window._switch_items()] == ["B", "C", "A"]
    assert window.switch_default.currentText() == "B"
    before = [dict(entry.params) for entry in workspace.project.entries]
    item = window.switch_positions.takeItem(0)
    window.switch_positions.insertItem(2, item)
    window._switch_positions_changed()
    assert window.switch_default.currentText() == "B"
    updated = workspace.controls()[0]
    assert updated.enum_names == ("C", "A", "B")
    assert updated.default == 2
    assert updated.option_indices is None
    assert workspace.plan_changed()
    assert [entry.params for entry in workspace.project.entries] == before


def test_new_knob_can_be_converted_to_a_switch_without_stale_settings(window, workspace):
    workspace.draft_controls = [SwitchSpec("Mode", ("A", "B", "C")).to_dict()]
    window._attach(workspace)
    window.add_control()
    assert window.control_pages.currentIndex() == 0
    window.control_kind.setCurrentIndex(1)
    assert [item.text() for item in window._switch_items()] == ["Position 1", "Position 2"]
    assert isinstance(workspace.controls()[1], SwitchSpec)


def test_metadata_edits_are_preserved_after_capture_snapshot_replaces_project(
    window, workspace
):
    window._attach(workspace)
    session = window._get_session()
    window.worker = SimpleNamespace(deleteLater=lambda: None)
    window.navigate(3)
    assert window.metadata_card.isEnabled()
    window.save_metadata({"modeled_by": "Edited while recording"})
    assert session.project.user_metadata.get("modeled_by") != "Edited while recording"
    window._audio_finished()
    assert workspace.project.user_metadata["modeled_by"] == "Edited while recording"
    assert type(workspace)(workspace.directory).project.user_metadata[
        "modeled_by"
    ] == "Edited while recording"


def test_parametric_loopback_required_static_remains_optional(window, workspace):
    window._attach(workspace)
    window.loop_enabled.setChecked(False)
    assert window.loop_enabled.isChecked()
    assert not window.loop_enabled.isEnabled()
    assert window.audio_settings().loopback_enabled
    window.capture_kind.setCurrentIndex(window.capture_kind.findData("static"))
    assert window.loop_enabled.isEnabled()
    window.loop_enabled.setChecked(False)
    assert not window.audio_settings().loopback_enabled


def test_reconnected_interface_restores_identity_and_route_automatically(
    window, monkeypatch
):
    from paranam import wizard

    device = DeviceInfo(2, "Original interface", "ASIO", 4, 4, 48000)
    available = [device]
    monkeypatch.setattr(wizard, "list_devices", lambda **k: list(available))
    monkeypatch.setattr(
        wizard,
        "capabilities",
        lambda _: Capabilities(
            ("Return", "Loopback", "Guitar", "Input 4"),
            ("Send", "Loopback", "L", "R"),
            (128, 2048),
        ),
    )
    window.refresh_devices()
    window.input_channel.setCurrentIndex(window.input_channel.findData(3))
    available.clear()
    window.refresh_devices()
    assert window.device.currentData() is None
    assert "Original interface" in window.device.currentText()
    available.append(DeviceInfo(17, "Original interface", "ASIO", 4, 4, 48000))
    window._settings_device_changed()
    window._reconnect_device()
    assert window.device.currentData().index == 17
    assert window.input_channel.currentData() == 3


def test_calibration_alignment_meters_and_table_resize(window, qapp):
    window.navigate(6)
    window.show()
    qapp.processEvents()
    assert (
        window.capture_level.mapTo(window.calibration_card, QPoint()).x()
        == window.input_dbu.mapTo(window.calibration_card, QPoint()).x()
    )
    window._route_metered(0.5, 0.25, 0.125, 0.0625)
    assert window.route_send_peak.text() == "-6.0 dBFS"
    assert window.route_return_peak.text() == "-12.0 dBFS"
    assert window.route_send_meter.levels[0] == pytest.approx(-6.0206)
    assert window.route_loopback_send_peak.text() == "-18.1 dBFS"
    assert window.route_loopback_return_peak.text() == "-24.1 dBFS"
    assert window.route_send_meter.width() > 5 * window.route_send_meter.height()
    assert (
        window.route_status.mapTo(window.audio_card, QPoint()).y()
        > window.route_meter_panel.mapTo(window.audio_card, QPoint()).y()
        + window.route_meter_panel.height()
    )
    assert not window.version_label.font().bold()
    assert not window.queue_table.horizontalHeader().stretchLastSection()
    assert not any(
        "(FULL)" in label.text()
        for label in window.page_contents[4].findChildren(type(window.version_label))
    )


@pytest.mark.parametrize("size", ["full", "lite"])
def test_chart_hover_selects_nearest_epoch_and_restores_latest(qapp, size):
    chart = ESRChart(size=size)
    chart.resize(800, 250)
    chart.set_history(
        [
            dict(
                epoch=i,
                full=0.1 / i,
                lite=0.2 / i,
                train_full=0.05 / i,
                train_lite=0.1 / i,
            )
            for i in (1, 3, 8)
        ]
    )
    chart.show()
    qapp.processEvents()
    plot = chart.plot_rect()
    QTest.mouseMove(
        chart,
        QPoint(round(plot.left() + 2.1 / 7 * plot.width()), round(plot.center().y())),
    )
    assert chart.hover_epoch == 3
    assert chart.displayed_entry()[size] == pytest.approx(
        (0.1 if size == "full" else 0.2) / 3
    )
    QTest.mouseMove(chart, QPoint(5, 5))
    assert chart.displayed_entry()["epoch"] == 8
    chart.close()


def test_batch_eta_and_log_throughput():
    timing = BatchTiming()
    assert timing.remaining(10) is None
    timing.record(2)
    timing.record(4)
    timing.validation_times.append(6)
    assert timing.remaining(7, 2) == 33
    assert timing.remaining(6, 2) == 30
    assert "0.33 it/s" in epoch_line(
        1, 5, "Full + Lite", 0.4, "Training", timing.iterations_per_second
    )


def test_recorded_inputs_are_bundled_and_loop_without_normalization(tmp_path):
    assert verify_bundled_previews() == 8
    assert len(preview_files()) == 8
    path = tmp_path / "DI.wav"
    sf.write(path, [0.1, -0.2, 0.3], 48000, subtype="FLOAT")
    loop = PreviewLoop(path, 48000)
    np.testing.assert_allclose(loop.read(5), [0.1, -0.2, 0.3, 0.1, -0.2])
    np.testing.assert_allclose(loop.read(2), [0.3, 0.1])
    path2 = tmp_path / "lower-rate.wav"
    sf.write(path2, np.ones(100) * 0.1, 24000, subtype="FLOAT")
    assert len(PreviewLoop(path2, 48000).samples) == 200


def test_asio_meter_reports_send_return_and_loopback_without_touching_recording(
    monkeypatch,
):
    from nam.capture import _asio_stream as asio

    main = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    loop = main * 2
    captured = main * 0.5
    job = SimpleNamespace(
        main=main,
        loop=loop,
        recorded=captured.copy(),
        loop_recorded=loop.copy(),
        position=0,
        error=None,
    )

    class Done:
        def wait(self, delay):
            job.position += 2
            return job.position == 4

    job.done = Done()
    monkeypatch.setattr(asio, "_BlockJob", lambda *a: job)
    recorder = asio.AsioRecorder.__new__(asio.AsioRecorder)
    recorder.stream = SimpleNamespace(active=True)
    readings = []
    recorded, loop_recorded = recorder._exchange(
        main, loop, None, meter=lambda *levels: readings.append(levels)
    )
    np.testing.assert_allclose(
        readings, [(0.2, 0.1, 0.4, 0.4), (0.4, 0.2, 0.8, 0.8)], rtol=1e-6
    )
    np.testing.assert_array_equal(recorded, captured)
    np.testing.assert_array_equal(loop_recorded, loop)


def test_preview_transport_is_exclusive_and_file_changes_restart(
    window, workspace, monkeypatch
):
    window._attach(workspace)
    starts = []
    monkeypatch.setattr(
        window, "start_playback", lambda: starts.append(window._play_input_mode)
    )
    window.toggle_playback_source("recording")
    assert starts == ["recording"]
    window._play_ready = True
    window.play_worker = SimpleNamespace(stopping=threading.Event())
    window.update_enabled()
    assert window.preview_start.isChecked() and not window.play_start.isChecked()
    window.toggle_playback_source("instrument")
    assert window.play_worker.stopping.is_set() and window._play_restart
    window.play_worker = None


def test_recorded_source_routes_same_signal_to_gear_and_model(
    qapp, monkeypatch, tmp_path
):
    from paranam import playback
    import sounddevice
    from nam.capture import audio as capture_audio

    monkeypatch.setattr(capture_audio, "asio_com_apartment", nullcontext)
    project = CaptureProject(
        knobs=[],
        audio=AudioSettingsModel(
            instrument_input_channel=None, monitor_output_left=3, monitor_output_right=4
        ),
    )
    device = SimpleNamespace(
        index=0, host_api="Test", max_input_channels=4, max_output_channels=4
    )
    validate_route(device, project.audio, require_instrument=False)
    path = tmp_path / "preview.wav"
    sf.write(path, np.full(500, 0.1), 48000, subtype="FLOAT")

    class Model:
        info = {"controls": [], "size_selection": False}

        def __init__(self, *a):
            pass

        def calibration(self, *a):
            return 1, 1

        def parameters(self, values):
            pass

        def process(self, samples):
            return samples

        def close(self):
            pass

    monkeypatch.setattr(playback, "NativeModel", Model)
    worker = PlayWorker("unused.nam", device, project, preview_path=path)
    outputs = []

    class Stream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def __enter__(self):
            incoming = np.full((256, 4), 0.9, dtype=np.float32)
            for _ in range(8):
                out = np.zeros_like(incoming)
                self.callback(incoming, out, len(incoming), None, False)
                outputs.append(out.copy())
            worker.stopping.set()
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(sounddevice, "Stream", Stream)
    errors = []
    worker.failed.connect(errors.append)
    worker.run()
    assert not errors
    assert 0.09 < outputs[-1][-1, 0] <= 0.101
    np.testing.assert_allclose(
        outputs[-1][-1, 2:], [outputs[-1][-1, 0]] * 2, atol=0.001
    )
    assert not np.any(outputs[-1][:, 1])


def test_ir_longer_than_twenty_seconds_keeps_its_tail(tmp_path):
    path = tmp_path / "long.wav"
    taps = np.zeros(21 * 8000 + 1)
    taps[0] = 0.5
    taps[-1] = 0.1
    sf.write(path, taps, 8000, subtype="FLOAT")
    kernel = prepare_ir(path, 8000)
    assert len(kernel.spectra) * 256 >= len(taps)
    assert np.any(kernel.spectra[-1])


def test_pause_queue_prevents_successors_and_resumes_current_first(
    workspace, tmp_path, qapp, monkeypatch
):
    from pathlib import Path
    from nam.capture.project import atomic_write_json
    from paranam.queue_controller import QueueController
    from paranam.workspace import read_json
    from tests.paranam.test_queue import small_ready

    small_ready(workspace)
    controller = QueueController(tmp_path / "queue-pause")
    entries = [controller.store.add(workspace) for _ in range(3)]
    current = entries[0]
    controller.store.update(current["id"], state="running")
    controller.active_id = current["id"]
    controller.enabled = True
    controller.pause_queue()
    assert not controller.enabled
    assert read_json(Path(current["job"]) / "command.json")["action"] == "pause"
    assert [entry["state"] for entry in controller.store.entries()] == [
        "pausing",
        "queued",
        "queued",
    ]
    atomic_write_json(Path(current["job"]) / "progress.json", {"status": "paused"})
    controller._finish()
    controller.tick()
    assert controller.active_id is None
    assert [entry["state"] for entry in controller.store.entries()] == [
        "paused",
        "queued",
        "queued",
    ]
    # Even if waiting jobs were moved while paused, resume the interrupted job first.
    restarted = QueueController(tmp_path / "queue-pause")
    assert restarted.paused_queue_id == current["id"]
    monkeypatch.setattr(restarted, "tick", lambda: None)
    restarted.start()
    assert restarted.enabled and restarted.paused_queue_id is None
    assert [
        entry["id"] for entry in restarted.store.entries() if entry["state"] == "queued"
    ] == [e["id"] for e in entries]

from pathlib import Path

import pytest

from nam.capture.project import atomic_write_json
from paranam import training
from paranam.recipes import training_preset
from paranam.workspace import Workspace, read_json
from tests.paranam.test_queue import small_ready


def test_static_hides_hypernetwork_fields_and_labels(window, workspace):
    workspace.preferences["advanced_parametric"] = training_preset("HyperA2")
    window._attach(workspace)
    fields = [
        window.advanced_fields[key]
        for key in (
            "hypernetwork_recipe",
            "hypernetwork_learning_rate",
            "hypernetwork_rate_preset",
            "hypernetwork_lr_decay",
            "hypernetwork_mode",
            "hypernetwork_rank",
            "condition_dilated_convolutions",
            "branch_loss_reduction",
        )
    ]
    layout = window.advanced_panel.layout()
    window.load_advanced_options("static")
    assert all(not layout.isRowVisible(field) for field in fields)
    assert all(
        layout.labelForField(field) is None or layout.labelForField(field).isHidden()
        for field in fields
    )
    assert layout.isRowVisible(window.advanced_fields["learning_rate"])
    assert layout.isRowVisible(window.advanced_fields["optimizer"])
    window.load_advanced_options("parametric")
    assert all(layout.isRowVisible(field) for field in fields)
    assert window.advanced_fields["hypernetwork_learning_rate"].value() == 0.0004
    assert window.advanced_fields["hypernetwork_lr_decay"].isEnabled()


def test_charts_keep_epoch_positions_and_hover_fixed(qapp):
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    from paranam.esr_chart import ESRCharts

    tabs = ESRCharts()
    tabs.resize(800, 280)
    tabs.show()
    qapp.processEvents()
    history = [dict(epoch=i, full=0.1 / i, lite=0.2 / i) for i in (1, 3)]
    tabs.set_history(history, total_epochs=100)
    for chart in tabs.charts.values():
        tabs.setCurrentWidget(chart)
        qapp.processEvents()
        chart.resize(800, 250)
        positions = [chart.epoch_x(i) for i in (1, 3, 8)]
        chart.set_history(history + [dict(epoch=8, full=0.01, lite=0.02)], 100)
        assert chart.epoch_limit() == 100
        assert [chart.epoch_x(i) for i in (1, 3, 8)] == positions
        chart.show()
        from PySide6.QtCore import QPointF
        from types import SimpleNamespace

        chart.mouseMoveEvent(
            SimpleNamespace(
                position=lambda: QPointF(
                    chart.epoch_x(3), chart.plot_rect().center().y()
                )
            )
        )
        assert chart.hover_epoch == 3
        chart.set_history([], 50)
        assert chart.epoch_limit() == 50
        assert not chart.grab().isNull()
        chart.set_history([dict(epoch=1, full=0.1)], 1)
        assert chart.epoch_limit() == 1 and not chart.grab().isNull()
    tabs.close()


def test_audio_none_is_preserved_by_device_refresh_and_settings(window, monkeypatch):
    from tests.paranam.test_settings_audio import select_device

    select_device(window, monkeypatch)
    for field in (
        window.input_channel,
        window.output_channel,
        window.loop_input,
        window.loop_output,
        window.instrument_input,
        window.monitor_left,
        window.monitor_right,
    ):
        index = field.findData(None)
        assert index >= 0 and field.itemText(index) == "None"
        field.setCurrentIndex(index)
    window.remember_setup()
    window.refresh_devices()
    audio = window.audio_settings()
    assert all(
        getattr(audio, key) is None
        for key in (
            "input_channel",
            "output_channel",
            "loopback_input_channel",
            "loopback_output_channel",
            "instrument_input_channel",
            "monitor_output_left",
            "monitor_output_right",
        )
    )
    assert read_json_settings(window)["input_channel"] is None


def read_json_settings(window):
    import json

    return json.loads(window.settings.value("last_audio_setup"))["audio"]


@pytest.mark.parametrize(
    "send,returned,loop_send,loop_return",
    [(None, None, None, None), (None, None, 2, 2), (1, 1, None, None)],
)
def test_settings_callback_handles_unassigned_channels(
    monkeypatch, qapp, send, returned, loop_send, loop_return
):
    from contextlib import nullcontext
    import numpy as np
    import sounddevice as sd
    from nam.capture.audio import DeviceInfo
    from nam.capture.project import AudioSettingsModel
    from paranam import settings_audio

    worker = settings_audio.SettingsAudioWorker(
        DeviceInfo(9, "Test", "ASIO", 2, 2, 48000),
        AudioSettingsModel(
            output_channel=send,
            input_channel=returned,
            loopback_output_channel=loop_send,
            loopback_input_channel=loop_return,
        ),
    )
    errors, meters, blocks = [], [], []
    worker.failed.connect(errors.append)
    worker.metered.connect(lambda *values: meters.append(values))
    monkeypatch.setattr(settings_audio, "asio_com_apartment", nullcontext)

    class Stream:
        active = True

        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def __enter__(self):
            outgoing = np.full((4800, 2), np.nan, np.float32)
            worker.set_noise(True, -12)
            self.callback(
                np.full((4800, 2), 0.25, np.float32), outgoing, 4800, None, None
            )
            blocks.append(outgoing)
            worker.stop()
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(sd, "Stream", Stream)
    worker.run()
    assert not errors and len(meters) == 1
    assert meters[0][1] == (0.25 if returned else 0)
    assert meters[0][3] == (0.25 if loop_return else 0)
    for channel in (1, 2):
        assert bool(np.any(blocks[0][:, channel - 1])) == (channel in (send, loop_send))

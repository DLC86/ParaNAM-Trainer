import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from nam.capture.audio import DeviceInfo
from paranam.audio_devices import Capabilities, buffer_values
from paranam.calibration import (
    LevelCalibrationDialog,
    OutputLevelEdit,
    ToneWorker,
    gain_from_dbfs,
    measure_sine,
    volts_to_dbu,
)

DEVICE = DeviceInfo(3, "Test interface", "ASIO", 2, 2, 48000.0)


@pytest.mark.parametrize("frequency", [20, 1000, 20000])
def test_return_measurement_uses_sine_peak_dbfs(frequency):
    x = 0.25 * np.sin(np.arange(24000) * (2 * np.pi * frequency / 48000) + 0.7)
    result = measure_sine(x + 0.01, 48000, frequency)
    assert result["valid"] and not result["clipping"]
    assert result["dbfs"] == pytest.approx(20 * math.log10(0.25), abs=0.001)
    assert volts_to_dbu(0.775) == pytest.approx(0.0)


def test_silence_noise_and_clipping_are_not_accepted_as_calibration():
    assert not measure_sine(np.zeros(24000), 48000, 1000)["valid"]
    assert not measure_sine(
        np.random.default_rng(2).normal(0, 0.1, 24000), 48000, 1000
    )["valid"]
    x = np.clip(2 * np.sin(np.arange(24000) * 2 * np.pi * 1000 / 48000), -1, 1)
    assert measure_sine(x, 48000, 1000)["clipping"]


def test_tone_send_and_return_share_capture_attenuation(qapp):
    dialog = LevelCalibrationDialog(DEVICE, 1, 2, 2048, -12.0)
    assert dialog.frequency.value() == 1000
    assert dialog.worker is None  # A full-scale tone never starts automatically.
    sends, returns = [], []
    dialog.send_calibrated.connect(sends.append)
    dialog.return_calibrated.connect(returns.append)
    dialog.volts.setValue(0.775)
    dialog.apply_send()
    assert sends == [
        0.0
    ]  # Effective send at -12 dBFS, not an extrapolated DAC reference.
    dialog.worker = SimpleNamespace(stopping=threading.Event())
    dialog.direct.setChecked(True)
    dialog.measured(
        dict(valid=True, clipping=False, dbfs=-18.0, frequency=1000.0, level=-12.0)
    )
    assert dialog.use_return.isEnabled()
    dialog.apply_return()
    assert returns == [18.0]
    dialog.measured(
        dict(valid=True, clipping=True, dbfs=0.0, frequency=1000.0, level=-12.0)
    )
    assert not dialog.use_return.isEnabled()
    assert "lower" in dialog.meter.text().lower()
    dialog.worker = None
    dialog.reject()


def test_level_edit_supports_silence_and_rejects_amplification(qapp):
    edit = OutputLevelEdit()
    assert edit.level() == 0.0
    edit.setText("-12,345")
    assert edit.level() == -12.345
    edit.setText("-inf")
    assert edit.level() is None and gain_from_dbfs(edit.level()) == 0.0
    for invalid in ("1", "nan", "inf", "abc"):
        edit.setText(invalid)
        with pytest.raises(ValueError):
            edit.level()


def test_driver_controls_show_names_supported_buffers_and_defaults(window, monkeypatch):
    from paranam import wizard

    capabilities = Capabilities(
        ("Mic / Line 1", "Line 2"),
        ("Reamp send", "Monitor"),
        (64, 128, 512, 2048, 4096),
    )
    monkeypatch.setattr(wizard, "capabilities", lambda device: capabilities)
    window.device.addItem(DEVICE.name, DEVICE)
    window.device.setCurrentIndex(window.device.count() - 1)
    assert window.input_channel.currentText() == "Mic / Line 1"
    assert window.output_channel.currentText() == "Reamp send"
    assert window.loop_output.currentText() == "Monitor"
    assert window.buffer.currentData() == 2048
    assert [window.buffer.itemData(i) for i in range(window.buffer.count())] == list(
        capabilities.buffers
    )
    assert Capabilities((), (), (96, 192, 480)).default_buffer == 480
    assert buffer_values(64, 2048, 256, -1) == (64, 128, 256, 512, 1024, 2048)
    assert buffer_values(8, 72, 56, 16) == (8, 24, 40, 56, 72)
    assert buffer_values(480, 480, 480, 0) == (480,)


def test_level_change_requires_new_plan_and_keeps_old_calibration_history(
    window, workspace
):
    workspace.project.input_level_dbu = 18.0
    workspace.save()
    window._attach(workspace)
    assert window.input_dbu.decimals() == window.output_dbu.decimals() == 3
    window._set_capture_level(-12.0)
    window._apply_level_calibration(window.input_calibration, window.input_dbu, 6.123)
    assert workspace.plan_changed()
    assert workspace.project.capture_level_dbfs == 0.0
    assert workspace.project.input_level_dbu == 18.0
    workspace.generate_plan()
    assert workspace.project.capture_level_dbfs == -12.0
    assert workspace.project.input_level_dbu == 6.123
    assert not workspace.plan_changed()
    workspace.preferences["capture_level_dbfs"] = None
    with pytest.raises(ValueError, match="muted"):
        workspace.generate_plan()


def test_calibration_tone_routes_only_selected_output_with_peak_amplitude(
    qapp, monkeypatch
):
    import sounddevice as sd

    worker = ToneWorker(DEVICE, 1, 2, 2048, 1000.0, -6.0)
    output = []

    class Stream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]
            assert kwargs["device"] == (3, 3)

        def __enter__(self):
            for _ in range(8):
                block = np.zeros((2048, 2), np.float32)
                self.callback(np.zeros_like(block), block, 2048, None, False)
                output.append(block.copy())
            worker.stopping.set()
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(sd, "Stream", Stream)
    worker.run()
    samples = np.concatenate(output)
    assert not samples[:, 0].any()
    assert np.max(np.abs(samples[:, 1])) == pytest.approx(
        gain_from_dbfs(-6.0), abs=1e-6
    )


def test_asio_metadata_uses_driver_labels_and_buffer_granularity(monkeypatch):
    import ctypes as ct
    from paranam import audio_devices

    class Function:
        def __init__(self, call):
            self.call = call

        def __call__(self, *args):
            return self.call(*args)

    def channel(device, index, name):
        ct.cast(name, ct.POINTER(ct.c_char_p))[0] = (b"Analog 1", b"Reamp 2")[index]
        return 0

    def buffers(device, *pointers):
        for pointer, value in zip(pointers, (64, 4096, 512, -1)):
            ct.cast(pointer, ct.POINTER(ct.c_long))[0] = value
        return 0

    library = SimpleNamespace(
        PaAsio_GetInputChannelName=Function(channel),
        PaAsio_GetOutputChannelName=Function(channel),
        PaAsio_GetAvailableBufferSizes=Function(buffers),
    )
    monkeypatch.setattr(audio_devices.ct, "CDLL", lambda *a: library)
    result = audio_devices.capabilities(DEVICE)
    assert result.inputs == ("Analog 1", "Reamp 2")
    assert result.buffers == (64, 128, 256, 512, 1024, 2048, 4096)
    assert result.default_buffer == 2048


def test_training_log_keeps_epoch_progress_visible(window, workspace):
    from paranam import training
    from tests.paranam.test_training import ready

    ready(workspace)
    job = training.make_job(workspace)
    (job / "worker.log").write_text(
        "Epoch 0: 10%|#---|\r\x1b[A\rEpoch 0: 50%|##--|", encoding="utf-8"
    )
    window._attach(workspace)
    text = window.details.toPlainText()
    assert "Epoch 0: 50%" in text
    assert "\x1b" not in text and "\r" not in text

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from nam.capture.audio import DeviceInfo
from nam.capture.project import AudioSettingsModel
from paranam.audio_devices import Capabilities
from paranam.settings_audio import SettingsAudioWorker


def select_device(window, monkeypatch):
    from paranam import wizard

    device = DeviceInfo(9, "Test ASIO", "ASIO", 4, 6, 48000)
    calls = []

    def devices(**kwargs):
        calls.append(kwargs)
        return [device]

    monkeypatch.setattr(wizard, "list_devices", devices)
    monkeypatch.setattr(
        wizard,
        "capabilities",
        lambda _: Capabilities(
            ("In 1", "Return", "Loopback", "Instrument"),
            ("Send", "Out 2", "Loopback", "Out 4", "Monitor L", "Monitor R"),
            (128, 2048),
        ),
    )
    window.refresh_devices()
    window.input_channel.setCurrentIndex(window.input_channel.findData(2))
    window.loop_input.setCurrentIndex(window.loop_input.findData(3))
    window.loop_output.setCurrentIndex(window.loop_output.findData(3))
    return calls


def test_settings_live_meters_one_stream_and_noise_toggle(window, monkeypatch, qapp):
    calls = select_device(window, monkeypatch)
    window.show()
    window.navigate(6)
    qapp.processEvents()
    worker = window.settings_audio_worker
    assert worker is not None
    assert not worker.noise[0]
    worker.metered.emit(0.0, 0.5, 0.0, 0.25)
    assert window.route_return_peak.text() == "-6.0 dBFS"
    assert window.route_loopback_return_peak.text() == "-12.0 dBFS"
    count = len(calls)
    for _ in range(10):
        window.update_enabled()
        window._reconnect_device()
    assert window.settings_audio_worker is worker
    assert len(calls) == count  # No periodic PortAudio unload/reload.
    window.capture_level.set_level(-12)
    monkeypatch.setattr(
        window,
        "_get_session",
        lambda: pytest.fail("No alignment/capture session for noise"),
    )
    window.test_route()
    assert worker.noise == (True, -12)
    assert window.route_button.text() == "Stop test"
    window.capture_level.set_level(-18)
    window.save_capture_level()
    assert window.settings_audio_worker is worker
    assert worker.noise == (True, -18)
    window.test_route()
    assert worker.noise[0] is False
    assert window.settings_audio_worker is worker
    window.navigate(0)
    assert worker.stopped
    assert window.settings_audio_worker is None
    assert window.route_return_peak.text() == "−∞ dBFS"


def test_settings_driver_error_is_not_retried_by_timer(window, monkeypatch, qapp):
    select_device(window, monkeypatch)
    window.show()
    window.navigate(6)
    worker = window.settings_audio_worker
    worker.failed.emit("Driver unavailable")
    worker.finished.emit()
    for _ in range(5):
        window.update_enabled()
        window._reconnect_device()
    assert window.settings_audio_worker is None
    assert "Driver unavailable" in window.route_status.text()
    window.test_route()  # Explicit retry is allowed.
    assert window.settings_audio_worker is not None


def test_route_changes_close_old_stream_and_ignore_retired_signals(
    window, monkeypatch, qapp
):
    select_device(window, monkeypatch)
    window.show()
    window.navigate(6)
    old = window.settings_audio_worker
    window.input_channel.setCurrentIndex(window.input_channel.findData(4))
    assert old.stopped
    window.update_enabled()
    current = window.settings_audio_worker
    assert current is not old
    assert current.audio.input_channel == 4
    old.metered.emit(0.99, 0.99, 0.99, 0.99)
    assert window.route_return_peak.text() == "−∞ dBFS"
    window.close_audio()  # Capture/Compare/calibration all release this stream first.
    assert current.stopped
    assert window.settings_audio_worker is None


def test_hardware_change_refreshes_once_instead_of_periodically(
    window, monkeypatch, qapp
):
    calls = select_device(window, monkeypatch)
    window.show()
    window.navigate(6)
    old = window.settings_audio_worker
    count = len(calls)
    window._settings_device_changed()
    window._reconnect_device()
    assert old.stopped
    assert len(calls) > count
    count = len(calls)
    window._reconnect_device()
    assert len(calls) == count


@pytest.mark.parametrize(
    "level,amplitude", [(-12, 10 ** (-12 / 20)), (0, 1.0), (None, 0.0)]
)
def test_worker_noise_routing_level_passive_metering_and_stop(
    monkeypatch, qapp, level, amplitude
):
    import sounddevice as sd
    from paranam import settings_audio

    device = DeviceInfo(9, "Test ASIO", "ASIO", 4, 6, 48000)
    # Loopback capture may be disabled; the requested test still feeds BOTH outputs.
    audio = AudioSettingsModel(
        input_channel=2,
        output_channel=1,
        loopback_input_channel=3,
        loopback_output_channel=4,
        loopback_enabled=False,
        blocksize=128,
    )
    worker = SettingsAudioWorker(device, audio)
    received, blocks, settings = [], [], []
    worker.metered.connect(lambda *values: received.append(values))
    monkeypatch.setattr(settings_audio, "asio_com_apartment", nullcontext)

    class Stream:
        active = True

        def __init__(self, **kwargs):
            settings.append(kwargs)
            self.callback = kwargs["callback"]

        def __enter__(self):
            incoming = np.zeros((4800, 4), np.float32)
            incoming[:, 1], incoming[:, 2] = 0.5, -0.25
            for enabled in (False, True, False):
                worker.set_noise(enabled, level)
                outgoing = np.full((4800, 6), np.nan, np.float32)
                self.callback(incoming, outgoing, 4800, None, SimpleNamespace())
                blocks.append(outgoing.copy())
            worker.stop()
            outgoing = np.full((4800, 6), np.nan, np.float32)
            self.callback(incoming, outgoing, 4800, None, SimpleNamespace())
            blocks.append(outgoing.copy())
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(sd, "Stream", Stream)
    monkeypatch.setattr(
        sd, "playrec", lambda *a, **k: pytest.fail("No capture/stabilization probe")
    )
    worker.run()
    assert len(settings) == 1
    assert settings[0]["device"] == (9, 9)
    assert settings[0]["blocksize"] == 0
    assert settings[0]["channels"] == (4, 6)
    assert not np.any(blocks[0]) and not np.any(blocks[2]) and not np.any(blocks[3])
    noise = blocks[1][:, 0]
    np.testing.assert_array_equal(noise, blocks[1][:, 3])
    assert not np.any(blocks[1][:, [1, 2, 4, 5]])
    assert np.max(np.abs(noise)) <= amplitude + 1e-7
    if amplitude:
        assert np.max(np.abs(noise)) > amplitude * 0.99
        assert noise.std() == pytest.approx(amplitude / np.sqrt(3), rel=0.06)
    assert received[0] == (0.0, 0.5, 0.0, 0.25)
    assert received[1][0] == pytest.approx(float(np.max(np.abs(noise))))
    assert received[1][2] == received[1][0]


def test_real_qthread_delivers_meters_and_closes_on_its_own_thread(monkeypatch, qapp):
    import threading
    import sounddevice as sd
    from PySide6.QtTest import QTest
    from paranam import settings_audio

    worker = SettingsAudioWorker(
        DeviceInfo(9, "Test ASIO", "ASIO", 2, 2, 48000), AudioSettingsModel()
    )
    levels, owners = [], []
    worker.metered.connect(lambda *values: levels.append(values))
    monkeypatch.setattr(settings_audio, "asio_com_apartment", nullcontext)

    class Stream:
        active = True

        def __init__(self, **kwargs):
            owners.append(threading.get_ident())
            self.callback = kwargs["callback"]

        def __enter__(self):
            self.callback(
                np.full((4800, 2), 0.125, np.float32),
                np.empty((4800, 2), np.float32),
                4800,
                None,
                None,
            )
            return self

        def __exit__(self, *args):
            owners.append(threading.get_ident())

    monkeypatch.setattr(sd, "Stream", Stream)
    worker.start()
    try:
        for _ in range(100):
            qapp.processEvents()
            if levels:
                break
            QTest.qWait(10)
        assert levels == [(0.0, 0.125, 0.0, 0.125)]
    finally:
        worker.stop()
        assert worker.wait(3000)
    assert len(owners) == 2 and owners[0] == owners[1]
    assert owners[0] != threading.get_ident()


def test_device_notifications_ignore_driver_node_changes_and_non_usb(qapp):
    import ctypes as ct
    from ctypes import wintypes
    from paranam.settings_audio import AudioDeviceEvents

    calls = []
    listener = AudioDeviceEvents(lambda: calls.append(True))
    message = wintypes.MSG()
    message.message = 0x0219
    message.wParam = 7  # Driver load/unload must not cause another ASIO refresh.
    listener.nativeEventFilter(b"windows_generic_MSG", ct.addressof(message))
    assert not calls
    payload = ct.create_string_buffer(32)
    header = (ct.c_uint32 * 3).from_buffer(payload)
    header[0], header[1] = 32, 5
    message.lParam = ct.addressof(payload)
    message.wParam = 0x8000
    listener.nativeEventFilter(b"windows_generic_MSG", ct.addressof(message))
    assert not calls
    ct.memmove(ct.addressof(payload) + 12, listener.usb_guid, 16)
    listener.nativeEventFilter(b"windows_generic_MSG", ct.addressof(message))
    message.wParam = 0x8004
    listener.nativeEventFilter(b"windows_generic_MSG", ct.addressof(message))
    assert calls == [True, True]

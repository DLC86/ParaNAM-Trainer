"""One duplex stream for passive Settings meters and optional white-noise output."""

import threading

import numpy as np
from PySide6.QtCore import QAbstractNativeEventFilter, QThread, Signal

from nam.capture.audio import asio_com_apartment
from .calibration import gain_from_dbfs


class AudioDeviceEvents(QAbstractNativeEventFilter):
    """Watch physical USB arrivals/removals, ignoring driver-generated node changes."""

    def __init__(self, changed, window_handle=None):
        super().__init__()
        self.changed = changed
        self.handle = None
        import uuid

        self.usb_guid = uuid.UUID("a5dcbf10-6530-11d2-901f-00c04fb951ed").bytes_le
        if window_handle is not None:
            import ctypes as ct

            class DeviceInterface(ct.Structure):
                _fields_ = [
                    ("size", ct.c_uint32),
                    ("kind", ct.c_uint32),
                    ("reserved", ct.c_uint32),
                    ("guid", ct.c_ubyte * 16),
                    ("name", ct.c_wchar * 1),
                ]

            notification = DeviceInterface()
            notification.size, notification.kind = ct.sizeof(notification), 5
            notification.guid[:] = self.usb_guid
            self.user32 = ct.WinDLL("user32", use_last_error=True)
            register = self.user32.RegisterDeviceNotificationW
            register.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32]
            register.restype = ct.c_void_p
            self.handle = register(int(window_handle), ct.byref(notification), 0)

    def close(self):
        if self.handle is not None:
            import ctypes as ct

            unregister = self.user32.UnregisterDeviceNotification
            unregister.argtypes = [ct.c_void_p]
            unregister.restype = ct.c_bool
            unregister(self.handle)
            self.handle = None

    def nativeEventFilter(self, event_type, message):
        if bytes(event_type) in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
            import ctypes as ct
            from ctypes import wintypes

            event = wintypes.MSG.from_address(int(message))
            if (
                event.message == 0x0219
                and event.wParam in (0x8000, 0x8004)
                and event.lParam
            ):
                header = (ct.c_uint32 * 3).from_address(event.lParam)
                if (
                    header[0] >= 28
                    and header[1] == 5
                    and ct.string_at(event.lParam + 12, 16) == self.usb_guid
                ):
                    self.changed()
        return False, 0


class SettingsAudioWorker(QThread):
    metered = Signal(float, float, float, float)
    failed = Signal(str)

    def __init__(self, device, audio, rate=48000, parent=None):
        super().__init__(parent)
        self.device, self.audio, self.rate = device, audio.model_copy(deep=True), rate
        self.stopping = threading.Event()
        self.noise = (False, 0.0)

    def set_noise(self, enabled, level):
        # Replace the pair atomically; the callback never reads Qt widgets.
        self.noise = (bool(enabled), gain_from_dbfs(level))

    def stop(self):
        self.noise = (False, 0.0)
        self.stopping.set()

    def run(self):
        import sounddevice as sd

        try:
            device, audio, rate = self.device, self.audio, self.rate
            for channel, maximum in (
                (audio.input_channel, device.max_input_channels),
                (audio.loopback_input_channel, device.max_input_channels),
                (audio.output_channel, device.max_output_channels),
                (audio.loopback_output_channel, device.max_output_channels),
            ):
                if channel is not None and not 1 <= channel <= maximum:
                    raise ValueError(
                        "A selected Settings audio channel is unavailable."
                    )
            rng = np.random.default_rng()
            peaks = np.zeros(4)
            count = 0
            callback_error = []

            def callback(incoming, outgoing, frames, timing, status):
                nonlocal count
                outgoing.fill(0)
                try:
                    enabled, amplitude = self.noise
                    if enabled and amplitude and not self.stopping.is_set():
                        # Uniform white noise is bounded by the configured dBFS PEAK,
                        # unlike Gaussian noise with an unbounded crest factor.
                        noise = rng.uniform(-amplitude, amplitude, frames).astype(
                            np.float32
                        )
                        for channel in {
                            audio.output_channel,
                            audio.loopback_output_channel,
                        }:
                            if channel is not None:
                                outgoing[:, channel - 1] = noise
                    values = (
                        np.max(np.abs(outgoing[:, audio.output_channel - 1])),
                        np.max(np.abs(incoming[:, audio.input_channel - 1])),
                        (
                            np.max(
                                np.abs(outgoing[:, audio.loopback_output_channel - 1])
                            )
                            if audio.loopback_output_channel is not None
                            else 0.0
                        ),
                        (
                            np.max(
                                np.abs(incoming[:, audio.loopback_input_channel - 1])
                            )
                            if audio.loopback_input_channel is not None
                            else 0.0
                        ),
                    )
                    np.maximum(peaks, values, out=peaks)
                    count += frames
                    if count >= rate / 20:
                        self.metered.emit(*(float(value) for value in peaks))
                        peaks.fill(0)
                        count = 0
                except Exception as error:
                    outgoing.fill(0)
                    callback_error.append(str(error))
                    self.stopping.set()

            block = audio.blocksize or 2048
            with asio_com_apartment():
                with sd.Stream(
                    samplerate=rate,
                    device=(device.index, device.index),
                    channels=(device.max_input_channels, device.max_output_channels),
                    dtype="float32",
                    blocksize=0 if device.host_api == "ASIO" else block,
                    latency=block / rate,
                    callback=callback,
                ) as stream:
                    while not self.stopping.wait(0.05):
                        if not stream.active:
                            raise RuntimeError(
                                "The audio interface stopped or disconnected. Reconnect it and press Refresh."
                            )
            if callback_error:
                raise RuntimeError(callback_error[0])
        except Exception as error:
            self.failed.emit(str(error))


class SettingsAudioActions:
    def _settings_device_changed(self):
        self._audio_devices_dirty = True

    def _stop_settings_audio(self):
        worker = getattr(self, "settings_audio_worker", None)
        if worker is None:
            return
        worker.stop()
        if not worker.wait(5000):
            raise RuntimeError(
                "The Settings audio stream did not close. Wait before reopening the interface."
            )
        self.settings_audio_worker = None
        worker.deleteLater()
        self.route_button.setText("Test connection")
        self._route_metered(0.0, 0.0, 0.0, 0.0)

    def _sync_settings_audio(self, *, retry=False):
        from PySide6.QtWidgets import QApplication

        if (
            not self.isVisible()
            or self.pages.currentIndex() != 6
            or self.busy()
            or QApplication.activeModalWidget()
            or getattr(self, "_closing_settings_audio", False)
        ):
            self._stop_settings_audio()
            return
        device = self.device.currentData()
        if device is None:
            self._stop_settings_audio()
            return
        audio = self.audio_settings()
        rate = (self.workspace.project.sample_rate if self.workspace else None) or 48000
        key = (device, audio.model_dump_json(), rate)
        if retry:
            self._settings_audio_error_key = None
        worker = getattr(self, "settings_audio_worker", None)
        if worker is not None and getattr(self, "_settings_audio_key", None) == key:
            return
        self._stop_settings_audio()
        if getattr(self, "_settings_audio_error_key", None) == key:
            return  # A failed driver is retried only after an explicit action/change.
        self.close_audio()
        self._settings_audio_key = key
        worker = SettingsAudioWorker(device, audio, rate, self)
        self.settings_audio_worker = worker
        worker.metered.connect(lambda *values: self._settings_metered(worker, *values))
        worker.failed.connect(lambda error: self._settings_audio_failed(worker, error))
        worker.finished.connect(lambda: self._settings_audio_finished(worker))
        self.route_status.setText(
            "Live input meters · Test connection sends white noise to Send and Loopback output."
        )
        worker.start()

    def _settings_metered(self, worker, *values):
        if getattr(self, "settings_audio_worker", None) is worker:
            self._route_metered(*values)

    def _settings_audio_failed(self, worker, error):
        if getattr(self, "settings_audio_worker", None) is worker:
            self._settings_audio_error_key = self._settings_audio_key
            self.route_status.setText(f"Audio monitoring unavailable: {error}")
            self.route_button.setText("Test connection")

    def _settings_audio_finished(self, worker):
        if getattr(self, "settings_audio_worker", None) is worker:
            self.settings_audio_worker = None
            self._route_metered(0.0, 0.0, 0.0, 0.0)
        worker.deleteLater()

    def test_route(self):
        try:
            if self.busy():
                return
            if self.device.currentData() is None:
                raise ValueError("Select your audio interface in Settings.")
            self._sync_settings_audio(retry=True)
            worker = getattr(self, "settings_audio_worker", None)
            if worker is None:
                return
            enabled = not worker.noise[0]
            level = self.capture_level.level()
            worker.set_noise(enabled, level)
            self.route_button.setText("Stop test" if enabled else "Test connection")
            self.route_status.setText(
                "White noise → Send and Loopback output · Capture / validation output level."
                if enabled
                else "Test stopped · Live input meters."
            )
        except Exception as error:
            self.notify_error(error)

"""Native plugin DSP with a full-duplex instrument/reamp/stereo monitor route."""

import ctypes
import json
import math
from pathlib import Path
import sys
import threading

import numpy as np
from PySide6.QtCore import QThread, Signal


def runtime_path():
    folder = Path(__file__).resolve().parent / "native"
    name = (
        "paranam_playback.dll"
        if sys.platform == "win32"
        else (
            "libparanam_playback.dylib"
            if sys.platform == "darwin"
            else "libparanam_playback.so"
        )
    )
    path = folder / name
    if not path.is_file():
        raise RuntimeError(
            "The bundled playback engine is missing. Reinstall ParaNAM Trainer."
        )
    return path


class NativeModel:
    def __init__(self, path, sample_rate=48000, block_size=2048, lite=False):
        self.handle = None
        self.library = ctypes.CDLL(str(runtime_path()))
        dll = self.library
        pointer = ctypes.POINTER(ctypes.c_float)
        for name, result, args in (
            ("pn_error", ctypes.c_char_p, []),
            (
                "pn_load",
                ctypes.c_void_p,
                [ctypes.c_char_p, ctypes.c_double, ctypes.c_int, ctypes.c_int],
            ),
            ("pn_info", ctypes.c_char_p, [ctypes.c_void_p]),
            ("pn_destroy", None, [ctypes.c_void_p]),
            ("pn_parameters", ctypes.c_int, [ctypes.c_void_p, pointer, ctypes.c_int]),
            (
                "pn_input_gain_db",
                ctypes.c_double,
                [ctypes.c_void_p, ctypes.c_double, ctypes.c_int],
            ),
            (
                "pn_output_gain_db",
                ctypes.c_double,
                [ctypes.c_void_p, ctypes.c_double, ctypes.c_int],
            ),
            (
                "pn_output_mode_gain_db",
                ctypes.c_double,
                [ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_int],
            ),
            (
                "pn_process",
                ctypes.c_int,
                [ctypes.c_void_p, pointer, pointer, ctypes.c_int],
            ),
        ):
            function = getattr(dll, name)
            function.restype, function.argtypes = result, args
        self.block_size = block_size
        self.handle = dll.pn_load(
            str(Path(path).resolve()).encode("utf-8"),
            sample_rate,
            block_size,
            int(lite),
        )
        if not self.handle:
            raise ValueError(dll.pn_error().decode("utf-8", errors="replace"))
        self.info = json.loads(dll.pn_info(self.handle))

    def close(self):
        if self.handle:
            self.library.pn_destroy(self.handle)
            self.handle = None

    def parameters(self, values):
        values = np.ascontiguousarray(values, dtype=np.float32)
        if len(values) and not self.library.pn_parameters(
            self.handle,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(values),
        ):
            raise ValueError("Invalid model control positions")

    def calibration(self, instrument, returned, output_mode="calibrated"):
        modes = {"raw": 0, "calibrated": 1, "normalized": 2}
        mode = modes[output_mode]
        return (
            10
            ** (
                self.library.pn_input_gain_db(
                    self.handle, instrument or 0, instrument is not None
                )
                / 20
            ),
            10
            ** (
                self.library.pn_output_mode_gain_db(
                    self.handle, returned or 0, returned is not None, mode
                )
                / 20
            ),
        )

    def process(self, samples):
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        output = np.empty_like(samples)
        pointer = ctypes.POINTER(ctypes.c_float)
        for offset in range(0, len(samples), self.block_size):
            x, y = (
                samples[offset : offset + self.block_size],
                output[offset : offset + self.block_size],
            )
            if not self.library.pn_process(
                self.handle,
                x.ctypes.data_as(pointer),
                y.ctypes.data_as(pointer),
                len(x),
            ):
                raise RuntimeError(
                    self.library.pn_error().decode("utf-8", errors="replace")
                )
        return output

    def __del__(self):
        self.close()


def validate_route(device, audio, require_instrument=True):
    incoming = (
        (audio.instrument_input_channel, audio.input_channel)
        if require_instrument
        else (audio.input_channel,)
    )
    outgoing = (
        audio.output_channel,
        audio.monitor_output_left,
        audio.monitor_output_right,
    )
    if any(c is None for c in (*incoming, *outgoing)):
        raise ValueError(
            "Select Send output, Return input, Instrument input and both Monitor outputs in Settings."
        )
    if len(set(incoming)) != len(incoming):
        raise ValueError(
            "Instrument input and gear return must use different channels."
        )
    if len(set(outgoing)) != 3:
        raise ValueError(
            "Reamp output, Monitor out L and Monitor out R must use three different channels."
        )
    if any(not 1 <= c <= device.max_input_channels for c in incoming) or any(
        not 1 <= c <= device.max_output_channels for c in outgoing
    ):
        raise ValueError("A selected playback channel is unavailable on this driver.")


def send_gain(project, instrument_dbu):
    """Recover the physical instrument level at the calibrated reamp send."""
    attenuation = (
        0
        if project.capture_level_dbfs is None
        else 10 ** (project.capture_level_dbfs / 20)
    )
    if project.input_level_dbu is None or instrument_dbu is None:
        return attenuation
    return attenuation * 10 ** ((instrument_dbu - project.input_level_dbu) / 20)


class PlayWorker(QThread):
    loaded = Signal(dict)
    failed = Signal(str)
    metered = Signal(dict)

    def __init__(
        self,
        path,
        device,
        project,
        instrument_dbu=None,
        lite=False,
        parent=None,
        *,
        preview_path=None,
    ):
        super().__init__(parent)
        self.path, self.device, self.project = (
            path,
            device,
            project.model_copy(deep=True),
        )
        self.instrument_dbu, self.lite = instrument_dbu, lite
        self.preview_path = preview_path
        self.stopping = threading.Event()
        self.settings = dict(
            parameters=None,
            input_db=0.0,
            output_db=0.0,
            source="model",
            output_mode="calibrated",
        )

    def run(self):
        import sounddevice as sd
        from nam.capture.audio import asio_com_apartment

        model = None
        try:
            audio = self.project.audio
            validate_route(
                self.device, audio, require_instrument=self.preview_path is None
            )
            rate, block = self.project.sample_rate or 48000, max(
                64, audio.blocksize or 2048
            )
            from .preview import PreviewLoop

            preview = (
                PreviewLoop(self.preview_path, rate) if self.preview_path else None
            )
            model = NativeModel(self.path, rate, block, self.lite)
            self.loaded.emit(model.info)
            gains = {}
            for mode in ("raw", "calibrated", "normalized"):
                if mode == "calibrated":
                    gains[mode] = model.calibration(
                        self.instrument_dbu, self.project.output_level_dbu
                    )
                else:
                    gains[mode] = model.calibration(
                        self.instrument_dbu, self.project.output_level_dbu, mode
                    )
            reamp_gain = send_gain(self.project, self.instrument_dbu)
            last_parameters, blend, count = (
                None,
                (1.0 if self.settings["source"] == "model" else 0.0),
                0,
            )
            current_input, current_output = 0.0, 0.0
            current_model_output = gains[
                self.settings.get("output_mode", "calibrated")
            ][1]
            from .ir import MonitorIR

            monitor_ir = MonitorIR(rate, self.settings.get("ir_processor"))
            input_peak = 0.0
            output_peaks = np.zeros(2)
            meter_clipping = meter_adc = meter_dropout = False
            callback_error = []

            def callback(incoming, outgoing, frames, _time, status):
                nonlocal last_parameters, blend, count, current_input, current_output, current_model_output
                nonlocal input_peak, output_peaks, meter_clipping, meter_adc, meter_dropout
                outgoing.fill(0)
                if frames <= 0:
                    return
                try:
                    settings = self.settings
                    parameters = settings["parameters"]
                    if parameters is not None and parameters != last_parameters:
                        model.parameters(parameters)
                        last_parameters = parameters
                    indices = np.arange(1, frames + 1, dtype=np.float32)
                    decay = np.exp(-indices / (rate * 0.005))
                    target_input = (
                        0.0
                        if self.stopping.is_set()
                        else 10 ** (settings["input_db"] / 20)
                    )
                    input_ramp = target_input + (current_input - target_input) * decay
                    current_input = float(input_ramp[-1])
                    raw_instrument = (
                        preview.read(frames)
                        if preview is not None
                        else incoming[:, audio.instrument_input_channel - 1]
                    )
                    instrument = raw_instrument * input_ramp
                    send = instrument * reamp_gain
                    outgoing[:, audio.output_channel - 1] = np.clip(send, -1, 1)
                    input_gain, output_gain = gains[
                        settings.get("output_mode", "calibrated")
                    ]
                    model_ramp = (
                        output_gain + (current_model_output - output_gain) * decay
                    )
                    current_model_output = float(model_ramp[-1])
                    rendered = model.process(instrument * input_gain) * model_ramp
                    returned = incoming[:, audio.input_channel - 1]
                    target = 1.0 if settings["source"] == "model" else 0.0
                    ramp = blend + np.clip(
                        target - blend,
                        -indices / (rate * 0.02),
                        indices / (rate * 0.02),
                    )
                    blend = float(ramp[-1])
                    target_output = (
                        0.0
                        if self.stopping.is_set()
                        else 10 ** (settings["output_db"] / 20)
                    )
                    output_ramp = (
                        target_output + (current_output - target_output) * decay
                    )
                    current_output = float(output_ramp[-1])
                    monitor = (
                        monitor_ir.process(
                            (1 - ramp) * returned + ramp * rendered,
                            settings.get("ir_processor"),
                        )
                        * output_ramp[:, None]
                    )
                    if not np.isfinite(monitor).all():
                        raise ValueError("The model produced non-finite audio")
                    peaks = np.max(np.abs(monitor), axis=0)
                    clipped = np.clip(monitor, -1, 1)
                    for index, channel in enumerate(
                        (audio.monitor_output_left, audio.monitor_output_right)
                    ):
                        outgoing[:, channel - 1] = clipped[:, index]
                    raw_peak = float(np.max(np.abs(raw_instrument)))
                    input_peak = max(input_peak, raw_peak)
                    output_peaks = np.maximum(
                        output_peaks, np.max(np.abs(clipped), axis=0)
                    )
                    meter_clipping |= (
                        float(np.max(peaks)) >= 1 or float(np.max(np.abs(send))) >= 1
                    )
                    meter_adc |= raw_peak >= 0.999
                    meter_dropout |= bool(status)
                    count += frames
                    if count >= rate // 10:
                        self.metered.emit(
                            dict(
                                input_peak=input_peak,
                                output_peak=float(np.max(output_peaks)),
                                output_peaks=output_peaks.tolist(),
                                clipping=meter_clipping,
                                adc_clipping=meter_adc,
                                dropout=meter_dropout,
                            )
                        )
                        count = 0
                        input_peak = 0.0
                        output_peaks.fill(0)
                        meter_clipping = meter_adc = meter_dropout = False
                except Exception as error:
                    outgoing.fill(0)
                    callback_error.append(str(error))
                    self.stopping.set()

            with asio_com_apartment():
                with sd.Stream(
                    samplerate=rate,
                    device=(self.device.index, self.device.index),
                    channels=(
                        self.device.max_input_channels,
                        self.device.max_output_channels,
                    ),
                    dtype="float32",
                    blocksize=0 if self.device.host_api == "ASIO" else block,
                    latency=block / rate,
                    callback=callback,
                ):
                    while not self.stopping.wait(0.05):
                        pass
                    # Leave time for a final ramp before the hardware stream closes.
                    self.msleep(int(1000 * block / rate) + 40)
            if callback_error:
                raise RuntimeError(callback_error[0])
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            if model:
                model.close()

"""Guided RMS-voltage calibration and continuous, explicitly started sine output."""

import math
import queue
import threading
from collections import deque

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QDoubleSpinBox,
    QPushButton,
    QCheckBox,
    QComboBox,
    QScrollArea,
    QWidget,
)


def gain_from_dbfs(level):
    if level is None:
        return 0.0
    if not math.isfinite(level) or level > 0:
        raise ValueError(
            "Choose an output level of 0 dBFS or lower, or -inf for silence."
        )
    return 10 ** (level / 20)


def volts_to_dbu(volts):
    if not math.isfinite(volts) or volts <= 0:
        raise ValueError("Enter a measured RMS voltage greater than zero.")
    return 20 * math.log10(volts / 0.775)


def measure_sine(samples, rate, frequency):
    peak = float(np.max(np.abs(samples)))
    # Callback block lengths need not contain an integer number of cycles. Trim
    # to whole periods so low-frequency RMS is not biased by the window's phase.
    periods = int(len(samples) * frequency / rate)
    length = (
        min(len(samples), round(periods * rate / frequency))
        if periods
        else len(samples)
    )
    y = np.asarray(samples[:length], dtype=np.float64)
    y = y - y.mean()
    phase = np.arange(len(y)) * (2 * np.pi * frequency / rate)
    basis = np.column_stack((np.sin(phase), np.cos(phase)))
    fit = np.linalg.lstsq(basis, y, rcond=None)[0]
    energy = float(y @ y)
    # Match the voltmeter's RMS measurement. sqrt(2) expresses a sine RMS level
    # using the peak dBFS convention (a full-scale sine reads 0 dBFS).
    amplitude = math.sqrt(2 * energy / len(y))
    residual = y - basis @ fit
    purity = max(0.0, 1 - float(residual @ residual) / energy) if energy else 0.0
    return dict(
        dbfs=20 * math.log10(max(amplitude, 1e-12)),
        clipping=peak >= 0.9999,
        valid=amplitude > 1e-5 and purity > 0.9,
        purity=purity,
    )


class OutputLevelEdit(QLineEdit):
    def __init__(self, level=0.0, parent=None):
        super().__init__(parent)
        self.set_level(level)
        self.setToolTip(
            "Peak output level in dBFS: 0 or a negative value; -inf mutes the output."
        )

    def set_level(self, level):
        self.setText("-inf" if level is None else f"{level:g}")

    def level(self):
        text = self.text().strip().lower().replace(",", ".").replace("−", "-")
        if text in ("-inf", "-infinity", "-∞"):
            return None
        try:
            level = float(text)
        except ValueError:
            raise ValueError("Enter an output level such as 0, -12.5 or -inf.")
        gain_from_dbfs(level)
        return level


class ToneWorker(QThread):
    measured = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        device,
        input_channel,
        output_channel,
        blocksize,
        frequency,
        level,
        parent=None,
    ):
        super().__init__(parent)
        self.device, self.input_channel, self.output_channel = (
            device,
            input_channel,
            output_channel,
        )
        self.blocksize = blocksize
        self.parameters = (frequency, level)
        self.stopping = threading.Event()

    def run(self):
        import sounddevice as sd
        from nam.capture.audio import asio_com_apartment

        rate, phase, current_gain = 48000, 0.0, 0.0
        packets = queue.Queue(maxsize=64)

        def callback(incoming, outgoing, frames, timing, status):
            nonlocal phase, current_gain
            frequency, level = self.parameters
            target = 0.0 if self.stopping.is_set() else gain_from_dbfs(level)
            # A 20 ms ramp avoids clicks, including when the user stops the tone.
            step = 1 / (rate * 0.02)
            gains = current_gain + np.clip(
                target - current_gain,
                -step * np.arange(1, frames + 1),
                step * np.arange(1, frames + 1),
            )
            current_gain = float(gains[-1])
            angles = phase + 2 * np.pi * frequency / rate * np.arange(frames)
            phase = (phase + 2 * np.pi * frequency / rate * frames) % (2 * np.pi)
            outgoing.fill(0)
            outgoing[:, self.output_channel - 1] = gains * np.sin(angles)
            try:
                packets.put_nowait(
                    (
                        incoming[:, self.input_channel - 1].copy(),
                        bool(status),
                        frequency,
                        level,
                    )
                )
            except queue.Full:
                pass

        try:
            with asio_com_apartment():
                with sd.Stream(
                    samplerate=rate,
                    device=(self.device.index, self.device.index),
                    channels=(
                        self.device.max_input_channels,
                        self.device.max_output_channels,
                    ),
                    dtype="float32",
                    blocksize=0 if self.device.host_api == "ASIO" else self.blocksize,
                    latency=self.blocksize / rate if self.blocksize else "high",
                    callback=callback,
                ):
                    chunks, count, previous = deque(), 0, None
                    while not self.stopping.is_set():
                        try:
                            block, bad, frequency, level = packets.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        if (frequency, level) != previous or bad:
                            chunks.clear()
                            count = 0
                            previous = (frequency, level)
                        if bad:
                            self.measured.emit(
                                dict(
                                    valid=False,
                                    clipping=False,
                                    dbfs=-240.0,
                                    stream_error=True,
                                )
                            )
                            continue
                        chunks.append(block)
                        count += len(block)
                        if count >= rate // 2:
                            result = measure_sine(
                                np.concatenate(chunks), rate, frequency
                            )
                            result.update(frequency=frequency, level=level)
                            self.measured.emit(result)
                            chunks.clear()
                            count = 0
                    # Let the callback write the final ramp before closing the stream.
                    self.msleep(int(1000 * max(self.blocksize, 2048) / rate) + 40)
        except Exception as error:
            self.failed.emit(str(error))


class LevelCalibrationDialog(QDialog):
    level_changed = Signal(object)
    send_calibrated = Signal(float)
    return_calibrated = Signal(float)
    instrument_calibrated = Signal(float)

    def __init__(
        self,
        device,
        input_channel,
        output_channel,
        blocksize,
        level=0.0,
        frequency=1000.0,
        parent=None,
        channel_names=None,
        instrument_channel=None,
        instrument_name=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Send, return and instrument level calibration")
        self.setMinimumSize(680, 500)
        self.resize(760, 760)
        self.device, self.channels, self.blocksize = (
            device,
            (input_channel, output_channel),
            blocksize,
        )
        self.worker, self.send_measurement, self.last_measurement = None, None, None
        self._closing = False
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        input_name, output_name = channel_names or (
            f"Input {input_channel}",
            f"Output {output_channel}",
        )
        layout.addWidget(
            QLabel(f"{device.name}\nSend: {output_name}\nReturn: {input_name}")
        )
        intro = QLabel(
            "1. Disconnect the gear and speakers. Connect an RMS voltmeter to the cable that will feed the gear, after any reamp box. Start the sine, measure its RMS voltage, then apply the send calibration. The same output level will be used for capture and validation playback."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self.frequency = QDoubleSpinBox()
        self.frequency.setRange(20, 20000)
        self.frequency.setDecimals(1)
        self.frequency.setValue(frequency)
        self.frequency.setSuffix(" Hz")
        self.level = OutputLevelEdit(level)
        self.volts = QDoubleSpinBox()
        self.volts.setDecimals(6)
        self.volts.setRange(0, 1000)
        self.volts.setSuffix(" V RMS")
        form.addRow("Sine frequency", self.frequency)
        form.addRow("Output / capture level (dBFS peak)", self.level)
        form.addRow("Measured voltage", self.volts)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.start = QPushButton("Start sine")
        self.stop = QPushButton("Stop sine")
        self.stop.setEnabled(False)
        self.start.clicked.connect(self.start_tone)
        self.stop.clicked.connect(self.stop_tone)
        row.addWidget(self.start)
        row.addWidget(self.stop)
        layout.addLayout(row)
        self.use_send = QPushButton("Apply send calibration")
        self.use_send.clicked.connect(self.apply_send)
        layout.addWidget(self.use_send)
        instruction = QLabel(
            "2. Stop the sine before reconnecting. Move the same cable directly to the selected return input, bypassing the gear. Keep the output setting unchanged, restart the sine and adjust input gain. If clipping is detected, lower the input gain before applying return calibration."
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)
        self.measure_input = QComboBox()
        self.measure_input.addItem(f"Return: {input_name}", input_channel)
        if instrument_channel is not None:
            self.measure_input.addItem(
                f"Instrument: {instrument_name or instrument_channel}",
                instrument_channel,
            )
        layout.addWidget(self.measure_input)
        instrument_instruction = QLabel(
            "3. To calibrate Instrument, stop the sine and move the same measured cable directly to the instrument input. Select Instrument above, confirm the direct cable, restart the sine and apply instrument calibration. Keep its input gain unchanged afterwards. This reference is used by Play."
            if instrument_channel is not None
            else "Configure Instrument input in Project & routing to calibrate it here."
        )
        instrument_instruction.setWordWrap(True)
        layout.addWidget(instrument_instruction)
        self.direct = QCheckBox(
            "Cable connected directly from reamp output to the selected measurement input"
        )
        layout.addWidget(self.direct)
        self.meter = QLabel(
            "Sine stopped. No audio is sent until Start sine is pressed."
        )
        self.meter.setWordWrap(True)
        layout.addWidget(self.meter)
        self.use_return = QPushButton("Apply return calibration")
        self.use_return.setEnabled(False)
        self.use_return.clicked.connect(self.apply_return)
        layout.addWidget(self.use_return)
        self.use_instrument = QPushButton("Apply instrument calibration")
        self.use_instrument.setEnabled(False)
        self.use_instrument.clicked.connect(self.apply_instrument)
        layout.addWidget(self.use_instrument)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        close = QPushButton("Done")
        close.clicked.connect(self.reject)
        outer.addWidget(close)
        self.frequency.valueChanged.connect(self.update_tone)
        self.level.editingFinished.connect(self.update_tone)
        self.volts.valueChanged.connect(self.invalidate_voltage)
        self.direct.toggled.connect(self.update_return)
        self.measure_input.currentIndexChanged.connect(self.change_measurement_input)

    def change_measurement_input(self, *_):
        self.last_measurement = None
        self.direct.setChecked(False)
        self.update_return()

    def update_tone(self, *_):
        try:
            level = self.level.level()
            self.level_changed.emit(level)
            self.last_measurement = None
            if self.worker:
                self.worker.parameters = (self.frequency.value(), level)
            self.update_return()
        except ValueError as error:
            self.status.setText(str(error))

    def invalidate_voltage(self, *_):
        self.send_measurement = None
        self.update_return()

    def start_tone(self):
        try:
            level = self.level.level()
            if level is None:
                raise ValueError(
                    "The output is muted (-inf). Choose a finite level to calibrate."
                )
            if self.worker:
                return
            self.last_measurement = None
            self.status.clear()
            self.level_changed.emit(level)
            self.worker = ToneWorker(
                self.device,
                self.measure_input.currentData(),
                self.channels[1],
                self.blocksize,
                self.frequency.value(),
                level,
                self,
            )
            self.worker.measured.connect(self.measured)
            self.worker.failed.connect(self.status.setText)
            self.worker.finished.connect(self.finished_tone)
            self.start.setEnabled(False)
            self.stop.setEnabled(True)
            self.measure_input.setEnabled(False)
            self.worker.start()
        except ValueError as error:
            self.status.setText(str(error))

    def stop_tone(self):
        if self.worker:
            self.worker.stopping.set()
            self.stop.setEnabled(False)
            self.last_measurement = None
            self.update_return()

    def finished_tone(self):
        self.worker.deleteLater()
        self.worker = None
        self.last_measurement = None
        self.start.setEnabled(True)
        self.stop.setEnabled(False)
        self.measure_input.setEnabled(True)
        self.meter.setText("Sine stopped.")
        self.update_return()
        if self._closing:
            super().reject()

    def measured(self, result):
        self.last_measurement = result
        self.meter.setText(
            "Clipping detected: lower the selected input gain."
            if result.get("clipping")
            else (
                "Audio stream lost samples. Increase the interface buffer."
                if result.get("stream_error")
                else (
                    f"Measured input sine: {result['dbfs']:.3f} dBFS peak"
                    if result["valid"]
                    else "No clean sine detected. Check the direct cable and selected measurement input."
                )
            )
        )
        self.update_return()

    def apply_send(self):
        try:
            level = self.level.level()
            if level is None:
                raise ValueError("Choose a finite output level before calibrating.")
            dbu = volts_to_dbu(self.volts.value())
            self.level_changed.emit(level)
            self.send_measurement = (dbu, self.frequency.value(), level)
            self.last_measurement = None
            self.send_calibrated.emit(dbu)
            self.status.setText(
                f"Send calibrated: {dbu:.3f} dBu at this capture output setting."
            )
            self.update_return()
        except ValueError as error:
            self.status.setText(str(error))

    def return_ready(self):
        try:
            parameters = (self.frequency.value(), self.level.level())
        except ValueError:
            return False
        result = self.last_measurement or {}
        return bool(
            self.worker
            and not self.worker.stopping.is_set()
            and self.direct.isChecked()
            and self.send_measurement
            and self.send_measurement[1:] == parameters
            and result.get("valid")
            and not result.get("clipping")
            and (result.get("frequency"), result.get("level")) == parameters
        )

    def update_return(self, *_):
        ready = self.return_ready()
        self.use_return.setEnabled(ready and self.measure_input.currentIndex() == 0)
        self.use_instrument.setEnabled(ready and self.measure_input.currentIndex() == 1)

    def apply_return(self):
        if self.return_ready() and self.measure_input.currentIndex() == 0:
            reference = self.send_measurement[0] - self.last_measurement["dbfs"]
            self.return_calibrated.emit(reference)
            self.status.setText(f"Return calibrated: {reference:.3f} dBu at 0 dBFS.")

    def apply_instrument(self):
        if self.return_ready() and self.measure_input.currentIndex() == 1:
            reference = self.send_measurement[0] - self.last_measurement["dbfs"]
            self.instrument_calibrated.emit(reference)
            self.status.setText(
                f"Instrument calibrated: {reference:.3f} dBu at 0 dBFS. Used by Play."
            )

    def reject(self):
        if self.worker:
            self._closing = True
            self.stop_tone()
        else:
            super().reject()

    def closeEvent(self, event):
        if self.worker:
            self.reject()
            event.ignore()
        else:
            event.accept()

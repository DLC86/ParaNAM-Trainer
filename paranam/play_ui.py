"""Play generated or imported NAM/PNAM models and compare the calibrated real rig."""

import math
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QComboBox,
    QDoubleSpinBox,
    QDial,
    QButtonGroup,
    QFileDialog,
    QAbstractSpinBox,
    QFrame,
)

from .playback import NativeModel, PlayWorker, validate_route
from .workspace import read_json


class PlayDial(QDial):
    def setDefaultValue(self, value):
        self.default_value = value

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(getattr(self, "default_value", 0))
            self.setSliderDown(False)
            if hasattr(self, "_origin"):
                del self._origin
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = self.rect().adjusted(10, 10, -10, -10)
        fraction = (self.value() - self.minimum()) / max(
            1, self.maximum() - self.minimum()
        )
        painter.setPen(
            QPen(QColor("#3a3a3a"), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        )
        painter.drawArc(bounds, 240 * 16, -300 * 16)
        painter.setPen(
            QPen(QColor("#bf0000"), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        )
        painter.drawArc(bounds, 240 * 16, round(-300 * fraction * 16))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#242424"))
        painter.drawEllipse(bounds.adjusted(9, 9, -9, -9))
        center = bounds.center()
        angle = math.radians(240 - fraction * 300)
        scale = self.width() / 92
        painter.setPen(
            QPen(QColor("#bf0000"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        )
        painter.drawLine(
            round(center.x() + 9 * scale * math.cos(angle)),
            round(center.y() - 9 * scale * math.sin(angle)),
            round(center.x() + 23 * scale * math.cos(angle)),
            round(center.y() - 23 * scale * math.sin(angle)),
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.position().y(), self.value()
            self.setSliderDown(True)
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and hasattr(self, "_origin"):
            precision = (
                10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            )
            delta = (
                (self._origin[0] - event.position().y())
                * (self.maximum() - self.minimum())
                / (160 * precision)
            )
            self.setValue(round(self._origin[1] + delta))
            event.accept()

    def mouseReleaseEvent(self, event):
        # QDial's default release maps the cursor angle to a value. This control
        # uses relative vertical dragging, so releasing must retain that value.
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSliderDown(False)
            if hasattr(self, "_origin"):
                del self._origin
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class ModelPlayDial(PlayDial):
    """The original Compare dial, enlarged slightly for the amp panel."""


class ControlValue(QDoubleSpinBox):
    """A plain clickable value which still supports typing and keyboard edits."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("controlValue")
        self.setFrame(False)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(110)
        self.setToolTip("Click to type a value. Double-click the knob to reset it.")

    def textFromValue(self, value):
        return self.locale().toString(value, "g", 12)


class ModelInfoLoader(QThread):
    loaded = Signal(dict)
    failed = Signal(str)

    def __init__(self, path, lite, parent=None):
        super().__init__(parent)
        self.path, self.lite = path, lite

    def run(self):
        try:
            model = NativeModel(self.path, lite=self.lite)
            try:
                self.loaded.emit(model.info)
            finally:
                model.close()
        except Exception as error:
            self.failed.emit(str(error))


class PlayActions:
    def _play(self, layout):
        title = QLabel("Play & Compare")
        title.setObjectName("title")
        layout.addWidget(title)
        note = QLabel(
            "Instrument → selected instrument input · Reamp send → gear → return · Monitor L/R → speakers or headphones. Disable interface direct monitoring on these channels."
        )
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addWidget(note)
        row = QHBoxLayout()
        self.play_model = QComboBox()
        self.play_model.addItem("Choose a model…", None)
        row.addWidget(self.play_model, 1)
        self.play_open = QPushButton("Load NAM / PNAM…")
        self.play_open.clicked.connect(self.open_play_model)
        row.addWidget(self.play_open)
        refresh = QPushButton("Refresh")
        self.play_refresh = refresh
        refresh.clicked.connect(self.refresh_play_models)
        row.addWidget(refresh)
        self.play_size = QComboBox()
        self.play_size.addItem("Full", False)
        self.play_size.addItem("Lite", True)
        self.play_output_mode = QComboBox()
        for caption, value in (
            ("Raw", "raw"),
            ("Calibrated", "calibrated"),
            ("Normalized", "normalized"),
        ):
            self.play_output_mode.addItem(caption, value)
        self.play_output_mode.setCurrentIndex(1)
        self.play_output_mode.setToolTip(
            "Calibrated falls back to Raw when model or interface output calibration is unavailable. Normalized uses the model loudness metadata, with Raw fallback."
        )
        self.play_output_mode.currentIndexChanged.connect(self.update_play_settings)
        layout.addLayout(row)
        buffer_row = QHBoxLayout()
        buffer_row.addWidget(QLabel("Playback buffer"))
        self.play_buffer = QComboBox()
        self.play_buffer.addItem("Driver managed", 0)
        buffer_row.addWidget(self.play_buffer)
        buffer_note = QLabel(
            "Lower buffers reduce playing latency; capture buffer settings are independent."
        )
        buffer_note.setWordWrap(True)
        buffer_note.setObjectName("muted")
        buffer_row.addWidget(buffer_note, 1)
        layout.addLayout(buffer_row)
        self.play_model.currentIndexChanged.connect(self.load_play_model)
        self.play_size.currentIndexChanged.connect(self.load_play_model)
        self.play_action_bar = QWidget()
        self.play_action_bar.setMaximumWidth(620)
        panel = QVBoxLayout(self.play_action_bar)
        panel.setContentsMargins(16, 12, 16, 16)
        panel.setSpacing(14)
        self.play_start = QPushButton("Play")
        self.play_start.setObjectName("comparePlay")
        self.play_start.setFixedSize(160, 62)
        self.play_start.clicked.connect(self.start_playback)
        self.play_stop = QPushButton("Stop playing")
        self.play_stop.setObjectName("comparePlay")
        self.play_stop.setFixedSize(160, 62)
        self.play_stop.clicked.connect(self.stop_playback)
        self.play_start.setEnabled(False)
        self.play_stop.setEnabled(False)
        self.play_stop.hide()
        panel.addWidget(self.play_start, alignment=Qt.AlignmentFlag.AlignHCenter)
        panel.addWidget(self.play_stop, alignment=Qt.AlignmentFlag.AlignHCenter)
        row = QHBoxLayout()
        row.setSpacing(18)
        self.gear_button, self.model_button = QPushButton("Gear"), QPushButton("Model")
        self.play_sources = QButtonGroup(self)
        for index, widget in enumerate((self.gear_button, self.model_button)):
            widget.setObjectName("compareSource")
            widget.setMinimumSize(200, 94)
            widget.setCursor(Qt.CursorShape.PointingHandCursor)
            widget.setCheckable(True)
            self.play_sources.addButton(widget, index)
            row.addWidget(widget, 1)
        panel.addLayout(row)
        self.model_button.setChecked(True)
        self.play_sources.idClicked.connect(self.update_play_settings)
        layout.addWidget(self.play_action_bar, alignment=Qt.AlignmentFlag.AlignHCenter)
        # Keep Stop reachable on other pages without moving the comparison buttons.
        self.play_transport_bar = QWidget()
        transport = QHBoxLayout(self.play_transport_bar)
        transport.setContentsMargins(0, 0, 0, 0)
        transport.addWidget(QLabel("Audio is playing"))
        self.play_quick_stop = QPushButton("Stop playing")
        self.play_quick_stop.clicked.connect(self.stop_playback)
        transport.addWidget(self.play_quick_stop)
        self.play_calibration = QLabel("")
        self.play_calibration.setWordWrap(True)
        layout.addWidget(self.play_calibration)
        controls = QGridLayout()
        controls.setHorizontalSpacing(28)
        self.play_gain_layout = controls
        self.play_input = self._play_gain("Input", -20, 20, 0, controls)
        self.play_output = self._play_gain("Output", -20, 20, 0, controls)
        for column, (caption, selector) in enumerate(
            (("Model size", self.play_size), ("Output mode", self.play_output_mode)), 2
        ):
            title = QLabel(caption)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            controls.addWidget(title, 0, column)
            selector.setMinimumWidth(140)
            controls.addWidget(
                selector, 1, column, alignment=Qt.AlignmentFlag.AlignCenter
            )
        controls.setColumnStretch(4, 1)
        layout.addLayout(controls)
        self.play_controls = QFrame()
        self.play_controls.setObjectName("ampPanel")
        self.play_controls_layout = QGridLayout(self.play_controls)
        self.play_controls_layout.setContentsMargins(24, 22, 24, 22)
        self.play_controls_layout.setHorizontalSpacing(24)
        self.play_controls.hide()
        self.play_values = []
        layout.addWidget(self.play_controls)
        self.play_meter = QLabel("Input Peak — · Monitor Peak —")
        layout.addWidget(self.play_meter)
        self.play_status = QLabel(
            "Select a model. Audio starts only when Play is pressed."
        )
        self.play_status.setWordWrap(True)
        layout.addWidget(self.play_status)
        self.play_worker = self.play_loader = None
        self._play_restart = self._pending_play_load = False
        self.play_buffer.currentIndexChanged.connect(self._play_buffer_changed)
        self._play_ready = False

    def _play_gain(self, name, minimum, maximum, default, layout):
        column = layout.count() // 3
        title = QLabel(name)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title, 0, column)
        dial = PlayDial()
        dial.setRange(int(minimum * 10), int(maximum * 10))
        dial.setValue(int(default * 10))
        dial.setDefaultValue(int(default * 10))
        dial.setFixedSize(92, 92)
        layout.addWidget(dial, 1, column, alignment=Qt.AlignmentFlag.AlignCenter)
        field = ControlValue()
        field.setRange(minimum, maximum)
        field.setDecimals(1)
        field.setValue(default)
        field.setSuffix(" dB")
        field.setFixedWidth(120)
        layout.addWidget(field, 2, column, alignment=Qt.AlignmentFlag.AlignCenter)
        dial.valueChanged.connect(lambda v: field.setValue(v / 10))
        field.valueChanged.connect(lambda v: dial.setValue(round(v * 10)))
        field.valueChanged.connect(self.update_play_settings)
        return field

    def open_play_model(self):
        path, _ = self.choose_open_file(
            "Load model", "NAM / PNAM models (*.nam *.pnam)"
        )
        if path:
            recent = self.settings.value("play_models", []) or []
            if isinstance(recent, str):
                recent = [recent]
            self.settings.setValue(
                "play_models", [path] + [p for p in recent if p != path][:29]
            )
            self.refresh_play_models(path)

    def refresh_play_models(self, selected=None):
        if self.play_loader:
            return
        previous = (
            selected if isinstance(selected, str) else self.play_model.currentData()
        )
        paths = self.settings.value("play_models", []) or []
        if isinstance(paths, str):
            paths = [paths]
        if self.workspace:
            paths += [
                str(p)
                for p in (self.workspace.directory / "models").rglob("*")
                if p.suffix.lower() in (".nam", ".pnam")
            ]
        names = {}
        for entry in self.queue_controller.store.entries():
            progress = read_json(Path(entry["job"]) / "progress.json", {})
            paths += progress.get("exports", [])
            if progress.get("export"):
                paths.append(progress["export"])
            for exported in [*progress.get("exports", []), progress.get("export")]:
                if exported:
                    names[str(Path(exported).resolve())] = (
                        f"{entry['name']} · {time.strftime('%d/%m %H:%M', time.localtime(entry['created_at']))} · {Path(exported).stem}"
                    )
        paths = sorted({str(Path(p).resolve()) for p in paths if Path(p).is_file()})
        self.play_model.blockSignals(True)
        self.play_model.clear()
        self.play_model.addItem("Choose a model…", None)
        for path in paths:
            self.play_model.addItem(
                names.get(path, f"{Path(path).parent.name} / {Path(path).name}"), path
            )
            self.play_model.setItemData(
                self.play_model.count() - 1, path, Qt.ItemDataRole.ToolTipRole
            )
        self.play_model.setCurrentIndex(max(0, self.play_model.findData(previous)))
        self.play_model.blockSignals(False)
        if self.play_model.currentData() is None:
            self._play_ready = False
            self.play_start.setEnabled(False)
            self.clear_play_controls()
        elif previous and (
            isinstance(selected, (str, bool))
            or previous != getattr(self, "_loaded_play_path", None)
            or not self._play_ready
        ):
            self.load_play_model()
        self._refresh_play_calibration()
        if not self.play_worker:
            self.play_buffer.blockSignals(True)
            previous_buffer = getattr(self, "_playback_blocksize", 128)
            self.play_buffer.clear()
            device = self.device.currentData()
            from .audio_devices import capabilities

            sizes = capabilities(device).buffers if device else []
            for size in sizes:
                self.play_buffer.addItem(f"{size} samples", size)
            if not sizes:
                self.play_buffer.addItem("Driver managed", 0)
            wanted = (
                previous_buffer
                if previous_buffer in sizes
                else 128 if 128 in sizes else min(sizes, default=0)
            )
            self.play_buffer.setCurrentIndex(max(0, self.play_buffer.findData(wanted)))
            self.play_buffer.blockSignals(False)

    def _refresh_play_calibration(self):
        value = self.instrument_reference()
        self.play_calibration.setText(
            f"Instrument input calibration: {value:.3f} dBu at 0 dBFS. Model input and real reamp send use this reference."
            if value is not None
            else "Instrument calibration is not set. Calibrate it in Settings for matched physical input levels."
        )
        if value is not None and not (
            self.input_calibration.isChecked() and self.output_calibration.isChecked()
        ):
            self.play_calibration.setText(
                self.play_calibration.text()
                + " Complete send/return calibration too for a level-matched Gear/Model comparison."
            )

    def playback_project(self):
        project = self.workspace.project.model_copy(deep=True)
        project.audio = self.audio_settings()
        project.capture_level_dbfs = self.capture_level.level()
        project.input_level_dbu = (
            self.input_dbu.value() if self.input_calibration.isChecked() else None
        )
        project.output_level_dbu = (
            self.output_dbu.value() if self.output_calibration.isChecked() else None
        )
        project.audio.blocksize = self.play_buffer.currentData() or 0
        return project

    def load_play_model(self, *_):
        if self.play_worker:
            self._play_restart = True
            self._pending_play_load = True
            self.stop_playback(reconfigure=True)
            return
        if self.play_loader:
            return
        path = self.play_model.currentData()
        self._play_ready = False
        self.play_start.setEnabled(False)
        if not path:
            self.clear_play_controls()
            return
        self.play_status.setText("Loading model…")
        self._preserved_play_values = (
            tuple(self.play_values)
            if path == getattr(self, "_loaded_play_path", None)
            else None
        )
        self._loaded_play_path = path
        self.play_loader = ModelInfoLoader(path, self.play_size.currentData(), self)
        self.play_loader.loaded.connect(self._play_model_loaded)
        self.play_loader.failed.connect(self.play_status.setText)
        self.play_loader.finished.connect(self._play_load_finished)
        self.play_model.setEnabled(False)
        self.play_open.setEnabled(False)
        self.play_refresh.setEnabled(False)
        self.play_size.setEnabled(False)
        self.play_loader.start()

    def _play_load_finished(self):
        self.play_loader.deleteLater()
        self.play_loader = None
        self.play_model.setEnabled(True)
        self.play_open.setEnabled(True)
        self.play_refresh.setEnabled(True)
        self.play_size.setEnabled(
            self._play_ready and self._play_info.get("size_selection", False)
        )
        self.play_start.setEnabled(self._play_ready)
        if self._play_restart and not self._close_after_pause:
            self._play_restart = False
            if self._play_ready:
                self.start_playback()
        self.update_enabled()

    def _play_model_loaded(self, info):
        values = getattr(self, "_preserved_play_values", None)
        self._play_ready, self._play_info = True, info
        self.clear_play_controls()
        grid = self.play_controls_layout
        self.play_values = [p["default"] for p in info["controls"]]
        if values is not None and len(values) == len(info["controls"]):
            self.play_values = [
                min(p["max"], max(p["min"], value))
                for p, value in zip(info["controls"], values)
            ]
        self.play_controls.setVisible(bool(info["controls"]))
        for index, control in enumerate(info["controls"]):
            frame = QWidget()
            column = QVBoxLayout(frame)
            name = QLabel(control["name"])
            name.setTextFormat(Qt.TextFormat.PlainText)
            name.setWordWrap(True)
            name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(name)
            if control["positions"]:
                widget = QComboBox()
                widget.addItems(control["positions"])
                widget.setCurrentIndex(int(self.play_values[index]))
                widget.currentIndexChanged.connect(
                    lambda value, i=index: self._play_parameter(i, float(value))
                )
            else:
                dial = ModelPlayDial()
                dial.setFixedSize(108, 108)
                dial.setRange(0, 10000)
                extent = control["max"] - control["min"]
                dial.setDefaultValue(
                    round(
                        (control["default"] - control["min"])
                        / max(extent, 1e-12)
                        * 10000
                    )
                )
                column.addWidget(dial, alignment=Qt.AlignmentFlag.AlignCenter)
                widget = ControlValue()
                widget.setRange(control["min"], control["max"])
                widget.setDecimals(3)
                widget.setSingleStep((control["max"] - control["min"]) / 100)
                widget.setValue(self.play_values[index])
                dial.setValue(
                    round(
                        (widget.value() - control["min"]) / max(extent, 1e-12) * 10000
                    )
                )
                widget.valueChanged.connect(
                    lambda value, i=index: self._play_parameter(i, value)
                )
                dial.valueChanged.connect(
                    lambda value, field=widget, minimum=control[
                        "min"
                    ], extent=extent: field.setValue(minimum + value * extent / 10000)
                )
                widget.valueChanged.connect(
                    lambda value, dial=dial, minimum=control[
                        "min"
                    ], extent=extent: dial.setValue(
                        round((value - minimum) / max(extent, 1e-12) * 10000)
                    )
                )
            column.addWidget(widget)
            grid.addWidget(frame, index // 5, index % 5)
        self.play_status.setText(
            "Ready. Gear and Model switch the monitored source; the real gear and model stay warm in parallel."
        )

    def clear_play_controls(self):
        self.play_controls.hide()
        grid = self.play_controls_layout
        while grid.count():
            item = grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.play_values = []

    def _play_parameter(self, index, value):
        self.play_values[index] = value
        self.update_play_settings()

    def update_play_settings(self, *_):
        if getattr(self, "play_worker", None):
            self.play_worker.settings = dict(
                parameters=tuple(self.play_values),
                input_db=self.play_input.value(),
                output_db=self.play_output.value(),
                source="gear" if self.gear_button.isChecked() else "model",
                output_mode=self.play_output_mode.currentData(),
            )

    def start_playback(self):
        if (
            not self.workspace
            or not self._play_ready
            or self.play_worker
            or self.worker
        ):
            return
        try:
            device = self.device.currentData()
            if not device:
                raise ValueError("Select your audio interface in Settings.")
            validate_route(device, self.audio_settings())
            self.close_audio()
            project = self.playback_project()
            self._playback_blocksize = project.audio.blocksize
            self.remember_setup()
            self.play_worker = PlayWorker(
                self.play_model.currentData(),
                device,
                project,
                self.instrument_reference(),
                self.play_size.currentData(),
                self,
            )
            self.update_play_settings()
            self.play_worker.failed.connect(self.play_status.setText)
            self.play_worker.metered.connect(self._play_metered)
            self.play_worker.finished.connect(self._play_finished)
            self.play_worker.start()
            self.play_start.setEnabled(False)
            self.play_stop.setEnabled(True)
            self.play_model.setEnabled(True)
            self.play_open.setEnabled(True)
            self.play_refresh.setEnabled(True)
            self.play_size.setEnabled(self._play_info.get("size_selection", False))
            self.play_buffer.setEnabled(True)
            self.play_status.setText("Listening · use Gear / Model to compare.")
            self.update_enabled()
        except Exception as error:
            self.notify_error(error)

    def _play_metered(self, values):
        db = lambda value: 20 * math.log10(max(value, 1e-12))
        text = f"Input Peak {db(values['input_peak']):.1f} dBFS · Monitor Peak {db(values['output_peak']):.1f} dBFS"
        if values["clipping"]:
            text += " · Clipping: lower Input / Output"
        if values.get("adc_clipping"):
            text += " · ADC clipping: lower the interface input gain"
        if values["dropout"]:
            text += " · Audio dropout: increase buffer size"
        self.play_meter.setText(text)

    def stop_playback(self, checked=False, *, reconfigure=False):
        if not reconfigure:
            self._play_restart = False
        if self.play_worker:
            self.play_worker.stopping.set()
            self.play_stop.setEnabled(False)

    def _play_finished(self):
        self.play_worker.deleteLater()
        self.play_worker = None
        self.play_start.setEnabled(self._play_ready)
        self.play_stop.setEnabled(False)
        self.play_model.setEnabled(True)
        self.play_open.setEnabled(True)
        self.play_refresh.setEnabled(True)
        self.play_size.setEnabled(self._play_info.get("size_selection", False))
        self.play_buffer.setEnabled(True)
        self.update_enabled()
        if self._close_after_pause:
            self._play_restart = self._pending_play_load = False
        elif self._pending_play_load:
            self._pending_play_load = False
            self.load_play_model()
        elif self._play_restart:
            self._play_restart = False
            self.start_playback()

    def _play_buffer_changed(self, *_):
        self._playback_blocksize = self.play_buffer.currentData() or 0
        self.remember_setup()
        if self.play_worker:
            self._play_restart = True
            self.stop_playback(reconfigure=True)

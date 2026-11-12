"""Play generated or imported NAM/PNAM models and compare the calibrated real rig."""

import math
import time
from pathlib import Path

from .numeric import DecimalSpinBox as QDoubleSpinBox

from PySide6.QtCore import QThread, Signal, Qt, QTimer, QRectF, QSize
from PySide6.QtGui import QPainter, QPen, QColor, QFontMetrics
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QComboBox,
    QDial,
    QButtonGroup,
    QFileDialog,
    QAbstractSpinBox,
    QFrame,
    QSlider,
    QSizePolicy,
)

from .playback import NativeModel, PlayWorker, validate_route
from .workspace import read_json
from .ir import BUNDLED, prepare_ir, IRProcessor
from .preview import preview_files
from .ui_widgets import navigation_icon
from .compare_layout import ControlFlowLayout, ControlPanel, AssetList
from .control_paint import paint_slide_switch


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


class SlideSwitch(QSlider):
    """Discrete amp-style slide switch; click, drag and arrow-key operable."""

    def __init__(self, positions, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.positions = positions
        font = self.font()
        font.setPointSize(11)
        font.setBold(True)
        label_width = max(
            QFontMetrics(font).horizontalAdvance(label) for label in positions
        )
        self.label_padding = max(16, label_width / 2 + 4)
        self.setRange(0, len(positions) - 1)
        self.setFixedHeight(108)
        self.setFixedWidth(
            round(
                max(
                    84,
                    (len(positions) - 1) * max(36, label_width + 8)
                    + 2 * self.label_padding,
                )
            )
        )
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.valueChanged.connect(
            lambda value: self.setAccessibleDescription(positions[value])
        )

    def position_x(self, index):
        return self.label_padding + index * (
            self.width() - 2 * self.label_padding
        ) / max(1, self.maximum())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint_slide_switch(
            painter,
            QRectF(self.rect()),
            self.value(),
            self.maximum() + 1,
            self.label_padding,
            self.hasFocus(),
        )

    def choose_position(self, event):
        fraction = (event.position().x() - self.label_padding) / max(
            1, self.width() - 2 * self.label_padding
        )
        self.setValue(round(min(1, max(0, fraction)) * self.maximum()))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self.choose_position(event)
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.choose_position(event)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.choose_position(event)
            event.accept()


class SwitchPositions(QWidget):
    def __init__(self, slider, parent=None):
        super().__init__(parent)
        self.slider = slider
        self.setFixedHeight(30)
        self.setFixedWidth(slider.width())

    def paintEvent(self, event):
        painter = QPainter(self)
        font = painter.font()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        step = (self.width() - 2 * self.slider.label_padding) / max(
            1, self.slider.maximum()
        )
        for index, label in enumerate(self.slider.positions):
            center = self.slider.label_padding + index * step
            painter.setPen(
                QColor("#ffffff" if index == self.slider.value() else "#aaaaaa")
            )
            painter.drawText(
                QRectF(center - step / 2, 0, step, self.height()),
                Qt.AlignmentFlag.AlignCenter,
                label,
            )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            fraction = (event.position().x() - self.slider.label_padding) / max(
                1, self.width() - 2 * self.slider.label_padding
            )
            self.slider.setValue(
                round(min(1, max(0, fraction)) * self.slider.maximum())
            )
            self.slider.setFocus()
            event.accept()


class ControlValue(QDoubleSpinBox):
    """A plain clickable value which still supports typing and keyboard edits."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("controlValue")
        self.setFrame(False)
        self.setKeyboardTracking(False)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(110)
        self.setToolTip(
            "Press Enter or click elsewhere to apply a typed value. Double-click the knob to reset it."
        )

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


class IRLoader(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, path, rate, parent=None):
        super().__init__(parent)
        self.path, self.rate = path, rate

    def run(self):
        try:
            kernel = prepare_ir(self.path, self.rate)
            self.loaded.emit((kernel, IRProcessor(kernel)))
        except Exception as error:
            self.failed.emit(str(error))


class LevelMeter(QWidget):
    """Two monitor channels or one instrument channel, with a decaying peak bar."""

    def __init__(self, channels, parent=None):
        super().__init__(parent)
        self.channels = channels
        self.levels = [-60.0] * channels
        self.setFixedSize(25 if channels == 1 else 44, 92)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.decay)
        self.timer.start(50)

    def decay(self):
        self.levels = [max(-60.0, level - 1.5) for level in self.levels]
        self.update()

    def set_levels(self, values):
        self.levels = [
            max(old, 20 * math.log10(max(float(value), 1e-6)))
            for old, value in zip(self.levels, values)
        ]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        width = (self.width() - 6) // self.channels
        for channel, level in enumerate(self.levels):
            left = channel * width + 3
            painter.fillRect(left, 2, width - 4, 88, QColor("#242424"))
            for step in range(22):
                threshold = -60 + step * 60 / 22
                if level >= threshold:
                    color = (
                        "#bf0000"
                        if threshold >= -3
                        else "#c4a03b" if threshold >= -12 else "#54a87b"
                    )
                    painter.fillRect(left, 86 - step * 4, width - 4, 3, QColor(color))


class HorizontalLevelMeter(LevelMeter):
    """Compact routing meter; the Compare page retains its vertical meters."""

    def __init__(self, parent=None):
        super().__init__(1, parent)
        self.setMinimumSize(40, 20)
        self.setMaximumSize(16777215, 20)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#242424"))
        count = max(22, self.width() // 8)
        step_width = (self.width() - 4) / count
        for index in range(count):
            threshold = -60 + (index + 1) * 60 / count
            if self.levels[0] >= threshold:
                color = (
                    "#bf0000"
                    if threshold >= -3
                    else "#c4a03b" if threshold >= -12 else "#54a87b"
                )
                painter.fillRect(
                    QRectF(
                        2 + index * step_width,
                        3,
                        max(1, step_width - 1),
                        self.height() - 6,
                    ),
                    QColor(color),
                )


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
        self.play_model = QComboBox(self)
        self.play_model.addItem("Choose a model…", None)
        self.play_model.hide()
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
        model_buttons = row
        self.play_model_remove = QPushButton("Remove")
        self.play_model_remove.clicked.connect(self.remove_play_models)
        model_buttons.addWidget(self.play_model_remove)
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
        self._play_input_mode = "instrument"
        self.play_start = QPushButton()
        self.play_start.setObjectName("comparePlay")
        self.play_start.setProperty("playing", False)
        self.play_start.setIcon(navigation_icon(5))
        self.play_start.setAccessibleName("Play instrument input")
        self.play_start.setToolTip("Play / stop the live instrument input")
        self.play_start.setCheckable(True)
        self.play_start.setIconSize(QSize(36, 36))
        self.play_start.setFixedSize(68, 68)
        self.play_start.clicked.connect(
            lambda: self.toggle_playback_source("instrument")
        )
        self.preview_start = QPushButton()
        self.preview_start.setObjectName("comparePlay")
        self.preview_start.setIcon(navigation_icon(7))
        self.preview_start.setAccessibleName("Play recorded audio")
        self.preview_start.setToolTip(
            "Play / stop the selected TONE3000 recorded input"
        )
        self.preview_start.setCheckable(True)
        self.preview_start.setIconSize(QSize(32, 32))
        self.preview_start.setFixedSize(68, 68)
        self.preview_start.clicked.connect(
            lambda: self.toggle_playback_source("recording")
        )
        self.preview_file = QComboBox()
        for caption, path in preview_files():
            self.preview_file.addItem(caption, str(path))
        self.preview_file.setMaximumWidth(280)
        self.preview_file.setAccessibleName("Recorded input signal")
        self.preview_file.currentIndexChanged.connect(self._preview_file_changed)
        self.play_stop = QPushButton("Stop playing")
        self.play_stop.setObjectName("comparePlay")
        self.play_stop.setProperty("playing", True)
        self.play_stop.setFixedSize(160, 62)
        self.play_stop.clicked.connect(self.stop_playback)
        self.play_start.setEnabled(False)
        self.play_stop.setEnabled(False)
        self.play_stop.hide()
        self.play_stop.setParent(self.play_action_bar)
        transport_row = QHBoxLayout()
        transport_row.addStretch()
        transport_row.addWidget(self.play_start)
        transport_row.addSpacing(8)
        transport_row.addWidget(self.preview_start)
        transport_row.addWidget(self.preview_file)
        transport_row.addStretch()
        panel.addLayout(transport_row)
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
        self.play_controls = ControlPanel()
        self.play_controls.setObjectName("ampPanel")
        self.play_controls_layout = ControlFlowLayout(self.play_controls)
        self.play_controls_layout.setContentsMargins(24, 22, 24, 22)
        self.play_controls_layout.setHorizontalSpacing(24)
        self.play_controls.hide()
        self.play_values = []
        layout.addWidget(self.play_controls)
        controls = QGridLayout()
        controls.setHorizontalSpacing(18)
        self.play_gain_layout = controls
        self.play_input_meter = LevelMeter(1)
        self.play_output_meter = LevelMeter(2)
        self.play_input_peak = QLabel("−∞ dBFS")
        self.play_output_peak = QLabel("−∞ dBFS")
        for column, title, meter, peak in (
            (0, "Instrument", self.play_input_meter, self.play_input_peak),
            (5, "Monitor L/R", self.play_output_meter, self.play_output_peak),
        ):
            peak.setMinimumSize(120, 30)
            peak.setMargin(4)
            peak.setAlignment(Qt.AlignmentFlag.AlignCenter)
            controls.addWidget(
                QLabel(title), 0, column, alignment=Qt.AlignmentFlag.AlignCenter
            )
            controls.addWidget(meter, 1, column, alignment=Qt.AlignmentFlag.AlignCenter)
            controls.addWidget(peak, 2, column, alignment=Qt.AlignmentFlag.AlignCenter)
        self.play_input = self._play_gain("Input", -20, 20, 0, controls, 1)
        self.play_output = self._play_gain("Output", -20, 20, 0, controls, 4)
        for column, (caption, selector) in enumerate(
            (("Model size", self.play_size), ("Output mode", self.play_output_mode)), 2
        ):
            title = QLabel(caption)
            title.setObjectName("compareControlLabel")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            controls.addWidget(title, 0, column)
            selector.setMinimumWidth(140)
            selector.setAttribute(Qt.WidgetAttribute.WA_LayoutUsesWidgetRect, True)
            controls.addWidget(
                selector, 1, column, alignment=Qt.AlignmentFlag.AlignCenter
            )
        layout.addLayout(controls)
        row = QHBoxLayout()
        row.addWidget(QLabel("Monitor IR"))
        self.play_ir = QComboBox(self)
        self.play_ir.addItem("Off", None)
        for name in ("Ampeg", "Celestion", "Eminence", "Mesa", "Plate", "Spring"):
            self.play_ir.addItem(
                name + " · Tone3000", str(BUNDLED / (name.lower() + ".wav"))
            )
        self.play_ir.hide()
        self.play_ir_open = QPushButton("Load IR…")
        self.play_ir_open.clicked.connect(self.open_play_ir)
        row.addWidget(self.play_ir_open)
        ir_buttons = row
        self.play_ir_remove = QPushButton("Remove")
        self.play_ir_remove.clicked.connect(self.remove_play_irs)
        ir_buttons.addWidget(self.play_ir_remove)
        self.play_model_list = AssetList(self.play_model)
        self.play_ir_list = AssetList(self.play_ir)
        self.play_model_list.itemSelectionChanged.connect(
            self.update_asset_remove_buttons
        )
        self.play_ir_list.itemSelectionChanged.connect(self.update_asset_remove_buttons)
        self.play_model.currentIndexChanged.connect(self.update_asset_remove_buttons)
        self.play_ir.currentIndexChanged.connect(self.update_asset_remove_buttons)
        assets = QHBoxLayout()
        for title, view, buttons in (
            ("Models", self.play_model_list, model_buttons),
            ("Impulse responses", self.play_ir_list, ir_buttons),
        ):
            pane = QFrame()
            pane.setObjectName("card")
            contents = QVBoxLayout(pane)
            contents.addWidget(QLabel(title))
            contents.addWidget(view, 1)
            view.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            contents.addLayout(buttons)
            assets.addWidget(pane, 1)
        self.play_ir_status = QLabel(
            "IR affects the monitor signal; reamp send is independent."
        )
        self.play_ir_status.setWordWrap(True)
        self.play_ir_status.setObjectName("muted")
        self.ir_loader = None
        self._ir_kernel = self._ir_processor = None
        self._ir_path = None
        self.play_ir.currentIndexChanged.connect(self.load_play_ir)
        imported = self.settings.value("play_irs", []) or []
        if isinstance(imported, str):
            imported = [imported]
        for path in imported:
            if Path(path).is_file() and self.play_ir.findData(path) < 0:
                self.play_ir.addItem(Path(path).name, path)
        saved_ir = self.settings.value("play_ir", "")
        if saved_ir and Path(saved_ir).is_file():
            index = self.play_ir.findData(saved_ir)
            if index < 0:
                self.play_ir.addItem(Path(saved_ir).name, saved_ir)
                index = self.play_ir.count() - 1
            self.play_ir.setCurrentIndex(index)
        self.play_meter = QLabel("")
        self.play_meter.setWordWrap(True)
        self.play_meter.hide()
        layout.addWidget(self.play_meter)
        self.play_status = QLabel(
            "Select a model. Audio starts only when Play is pressed."
        )
        self.play_status.setWordWrap(True)
        layout.addLayout(assets, 1)
        self.play_worker = self.play_loader = None
        self._play_restart = self._pending_play_load = False
        self.play_buffer.currentIndexChanged.connect(self._play_buffer_changed)
        self._play_ready = False
        self.update_asset_remove_buttons()

    def _play_gain(self, name, minimum, maximum, default, layout, column=None):
        if column is None:
            column = layout.count() // 3
        title = QLabel(name)
        title.setObjectName("compareControlLabel")
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

    def _selected_asset_paths(self, view):
        return {
            str(item.data(Qt.ItemDataRole.UserRole))
            for item in view.selectedItems()
            if item.data(Qt.ItemDataRole.UserRole)
        }

    def update_asset_remove_buttons(self, *_):
        if not hasattr(self, "play_ir_list"):
            return
        models = self._selected_asset_paths(self.play_model_list)
        irs = self._selected_asset_paths(self.play_ir_list)
        custom = bool(irs) and all(
            not Path(path).resolve().is_relative_to(BUNDLED.resolve()) for path in irs
        )
        self.play_model_remove.setEnabled(bool(models) and self.play_model.isEnabled())
        self.play_ir_remove.setEnabled(custom and self.play_ir.isEnabled())
        self.play_model_remove.setToolTip(
            "Remove selected entries from this list; keep files on disk"
        )
        self.play_ir_remove.setToolTip(
            "Remove selected custom IRs from this list; bundled IRs are protected"
        )

    def remove_play_models(self):
        if getattr(self, "play_loader", None):
            return
        removed = self._selected_asset_paths(self.play_model_list)
        if not removed:
            return
        hidden = self.settings.value("play_hidden_models", []) or []
        recent = self.settings.value("play_models", []) or []
        if isinstance(hidden, str):
            hidden = [hidden]
        if isinstance(recent, str):
            recent = [recent]
        self.settings.setValue("play_hidden_models", sorted(set(hidden) | removed))
        self.settings.setValue(
            "play_models", [p for p in recent if str(Path(p).resolve()) not in removed]
        )
        if self.play_model.currentData() in removed and getattr(
            self, "play_worker", None
        ):
            self.stop_playback()
        self.refresh_play_models()
        self.play_model_list.sync()
        self.update_asset_remove_buttons()

    def remove_play_irs(self):
        if getattr(self, "ir_loader", None):
            return
        removed = self._selected_asset_paths(self.play_ir_list)
        if not removed or any(
            Path(path).resolve().is_relative_to(BUNDLED.resolve()) for path in removed
        ):
            return
        current = self.play_ir.currentData()
        self.play_ir.blockSignals(True)
        for index in range(self.play_ir.count() - 1, -1, -1):
            if self.play_ir.itemData(index) in removed:
                self.play_ir.removeItem(index)
        self.play_ir.setCurrentIndex(
            0 if current in removed else max(0, self.play_ir.findData(current))
        )
        self.play_ir.blockSignals(False)
        self.settings.setValue(
            "play_irs",
            [
                self.play_ir.itemData(i)
                for i in range(self.play_ir.count())
                if self.play_ir.itemData(i)
                and not Path(self.play_ir.itemData(i))
                .resolve()
                .is_relative_to(BUNDLED.resolve())
            ],
        )
        if current in removed:
            self.load_play_ir()
        self.play_ir_list.sync()
        self.update_asset_remove_buttons()

    def open_play_ir(self):
        path, _ = self.choose_open_file("Load impulse response", "IR WAV (*.wav)")
        if path:
            imported = self.settings.value("play_irs", []) or []
            if isinstance(imported, str):
                imported = [imported]
            self.settings.setValue("play_irs", list(dict.fromkeys([*imported, path])))
            index = self.play_ir.findData(path)
            if index < 0:
                self.play_ir.addItem(Path(path).name, path)
                index = self.play_ir.count() - 1
            self.play_ir.setCurrentIndex(index)

    def load_play_ir(self, *_):
        if self.ir_loader:
            return
        path = self.play_ir.currentData()
        if not path:
            self._ir_kernel = self._ir_processor = self._ir_path = None
            self.settings.setValue("play_ir", "")
            self.play_ir_status.setText("IR off")
            self.update_play_settings()
            return
        rate = (
            (self.workspace.project.sample_rate or 48000) if self.workspace else 48000
        )
        self.ir_loader = IRLoader(path, rate, self)
        self.play_ir.setEnabled(False)
        self.play_ir_open.setEnabled(False)
        self.play_ir_status.setText("Loading IR…")
        self.ir_loader.loaded.connect(self._play_ir_loaded)
        self.ir_loader.failed.connect(self._play_ir_failed)
        self.ir_loader.finished.connect(self._play_ir_finished)
        self.ir_loader.start()

    def _play_ir_loaded(self, prepared):
        self._ir_kernel, self._ir_processor = prepared
        self._ir_path = self.play_ir.currentData()
        self.settings.setValue("play_ir", self._ir_path)
        self.play_ir_status.setText(
            f"{self._ir_kernel.name} · {self._ir_kernel.rate} Hz · monitor IR enabled"
        )
        self.update_play_settings()

    def _play_ir_failed(self, error):
        self.play_ir_status.setText(f"IR could not be loaded: {error}")
        self.play_ir.blockSignals(True)
        self.play_ir.setCurrentIndex(max(0, self.play_ir.findData(self._ir_path)))
        self.play_ir.blockSignals(False)

    def _play_ir_finished(self):
        self.ir_loader.deleteLater()
        self.ir_loader = None
        self.play_ir.setEnabled(True)
        self.play_ir_open.setEnabled(True)

    def open_play_model(self):
        path, _ = self.choose_open_file(
            "Load model", "NAM / PNAM models (*.nam *.pnam)"
        )
        if path:
            hidden = self.settings.value("play_hidden_models", []) or []
            if isinstance(hidden, str):
                hidden = [hidden]
            self.settings.setValue(
                "play_hidden_models",
                [p for p in hidden if Path(p).resolve() != Path(path).resolve()],
            )
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
        hidden = self.settings.value("play_hidden_models", []) or []
        if isinstance(hidden, str):
            hidden = [hidden]
        hidden = {str(Path(p).resolve()) for p in hidden}
        paths = sorted(
            {str(Path(p).resolve()) for p in paths if Path(p).is_file()} - hidden
        )
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
        self.update_asset_remove_buttons()
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
        if self.workspace:
            project = self.workspace.project.model_copy(deep=True)
        else:
            from nam.capture.project import CaptureProject

            project = CaptureProject(knobs=[], sample_rate=48000)
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
            column.setContentsMargins(9, 9, 9, 9)
            column.setSpacing(6)
            name = QLabel(control["name"])
            name.setObjectName("compareControlLabel")
            name.setTextFormat(Qt.TextFormat.PlainText)
            name.setWordWrap(True)
            name.setFixedHeight(36)
            name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(name)
            if control["positions"]:
                slider = SlideSwitch(control["positions"])
                slider.setAccessibleName(control["name"])
                # The UI enumerates options in index_sw order, but the native
                # model and session state must keep the original neural values.
                position_values = tuple(
                    control.get("position_values") or range(len(control["positions"]))
                )
                neural_value = int(self.play_values[index])
                visual_index = (
                    position_values.index(neural_value)
                    if neural_value in position_values else 0
                )
                slider.setValue(visual_index)
                column.addWidget(slider, alignment=Qt.AlignmentFlag.AlignCenter)
                widget = SwitchPositions(slider)
                slider.valueChanged.connect(widget.update)
                slider.valueChanged.connect(
                    lambda value, i=index, values=position_values:
                        self._play_parameter(i, float(values[value]))
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
                widget.setFixedHeight(30)
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
            column.addWidget(widget, alignment=Qt.AlignmentFlag.AlignCenter)
            frame.setFixedWidth(
                max(132, slider.width() + 18) if control["positions"] else 132
            )
            grid.addWidget(frame)
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
                ir_processor=self._ir_processor,
            )

    def start_playback(self):
        if not self._play_ready or self.play_worker or self.worker:
            return
        try:
            device = self.device.currentData()
            if not device:
                raise ValueError("Select your audio interface in Settings.")
            recorded = self._play_input_mode == "recording"
            validate_route(
                device, self.audio_settings(), require_instrument=not recorded
            )
            self.close_audio()
            project = self.playback_project()
            if self.ir_loader:
                self.play_status.setText("Wait for the IR to finish loading.")
                return
            if self._ir_kernel is not None:
                if self._ir_kernel.rate != (project.sample_rate or 48000):
                    self.load_play_ir()
                    self.play_status.setText(
                        "Preparing the IR at this project’s sample rate. Press Play when ready."
                    )
                    return
                self._ir_processor = IRProcessor(self._ir_kernel)
            self._playback_blocksize = project.audio.blocksize
            self.remember_setup()
            self.play_worker = PlayWorker(
                self.play_model.currentData(),
                device,
                project,
                self.instrument_reference(),
                self.play_size.currentData(),
                self,
                **(
                    {"preview_path": self.preview_file.currentData()}
                    if recorded
                    else {}
                ),
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
            self.update_enabled()

    def toggle_playback_source(self, mode):
        if self.play_worker:
            if self._play_input_mode == mode:
                self.stop_playback()
            else:
                self._play_input_mode = mode
                self._play_restart = True
                self.stop_playback(reconfigure=True)
        else:
            self._play_input_mode = mode
            self.start_playback()
        self.update_enabled()

    def _preview_file_changed(self, *_):
        if self.play_worker and self._play_input_mode == "recording":
            self._play_restart = True
            self.stop_playback(reconfigure=True)

    def _play_metered(self, values):
        db = lambda value: 20 * math.log10(max(value, 1e-12))
        self.play_input_meter.set_levels([values["input_peak"]])
        self.play_output_meter.set_levels(
            values.get("output_peaks", [values["output_peak"]] * 2)
        )
        self.play_input_peak.setText(f"{db(values['input_peak']):.1f} dBFS")
        self.play_output_peak.setText(f"{db(values['output_peak']):.1f} dBFS")
        warnings = []
        if values["clipping"]:
            warnings.append("Clipping: lower Input / Output")
        if values.get("adc_clipping"):
            warnings.append("ADC clipping: lower the interface input gain")
        if values["dropout"]:
            warnings.append("Audio dropout: increase buffer size")
        self.play_meter.setText(" · ".join(warnings))
        self.play_meter.setVisible(bool(warnings))

    def stop_playback(self, checked=False, *, reconfigure=False):
        if not reconfigure:
            self._play_restart = False
        if self.play_worker:
            self.play_worker.stopping.set()
            self.play_stop.setEnabled(False)

    def _play_finished(self):
        self.play_worker.deleteLater()
        self.play_worker = None
        self.play_input_meter.levels = [-60.0]
        self.play_output_meter.levels = [-60.0, -60.0]
        self.play_input_peak.setText("−∞ dBFS")
        self.play_output_peak.setText("−∞ dBFS")
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


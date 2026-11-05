"""Native Qt drawing and editors for the capture wizard."""

import math
from PySide6 import QtSvg  # Include SVG image support in frozen installers.

from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QWidget,
    QDialog,
    QFormLayout,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QDialogButtonBox,
    QStackedWidget,
    QMessageBox,
    QPlainTextEdit,
)

from nam.capture.params import KnobSpec, SwitchSpec

_font_loaded = False


def load_font():
    global _font_loaded
    if not _font_loaded:
        from .signals import asset_path
        from PySide6.QtWidgets import QApplication

        if QFontDatabase.addApplicationFont(str(asset_path("Inter.ttf"))) < 0:
            raise RuntimeError("The bundled user-interface font could not be loaded")
        if (
            QFontDatabase.addApplicationFont(str(asset_path("Michroma-Regular.ttf")))
            < 0
        ):
            raise RuntimeError("The bundled title font could not be loaded")
        QApplication.instance().setFont(QFont("Inter", 10))
        _font_loaded = True


ACCENT = "#bf0000"
TEXT = "#f0f0f0"
MUTED = "#969696"

THEME = """
QWidget { background: #121212; color: #f0f0f0; font-family: 'Inter'; font-size: 13px; }
QMainWindow { background: #121212; }
QFrame#sidebar { background: #1b1b1b; border-right: 1px solid #2d2d2d; }
QFrame#card { background: #1d1d1d; border: 1px solid #333333; border-radius: 14px; }
QFrame#card QLabel { background: transparent; }
QLabel, QCheckBox { background: transparent; }
QLabel#eyebrow { color: #bf0000; font-size: 11px; font-weight: 700; }
QLabel#version { color: #bf0000; font-size: 11px; font-weight: 400; }
QLabel#title { font-size: 31px; font-weight: 700; }
QLabel#section { font-size: 18px; font-weight: 650; }
QLabel#muted { color: #969696; }
QLabel#metric { font-size: 30px; font-weight: 650; }
QLabel#brand { font-family: 'Michroma'; font-size: 18px; font-weight: 700; color: #bf0000; }
QPushButton { background: #2c2c2c; border: 1px solid #414141; border-radius: 8px; padding: 10px 16px; font-weight: 600; }
QPushButton:hover { background: #3a3a3a; border-color: #797979; }
QToolButton { background: #2c2c2c; border: 1px solid #414141; border-radius: 8px; padding: 8px 12px; font-weight: 600; }
QToolButton:hover { background: #3a3a3a; border-color: #797979; }
QToolButton#brandLink { background: transparent; border: 1px solid transparent; padding: 0; border-radius: 4px; }
QToolButton#brandLink:hover, QToolButton#brandLink:focus { border-color: #bf0000; }
QWidget#advancedTraining { background: transparent; }
QFrame#ampPanel { background: #000000; border: 2px solid #404040; border-radius: 12px; }
QFrame#ampPanel QWidget { background: transparent; }
QLabel#compareControlLabel { font-size: 11pt; }
QDoubleSpinBox#controlValue, QDoubleSpinBox#controlValue QLineEdit { background: transparent; border: none; border-radius: 0; padding: 2px; font-size: 16px; font-weight: 600; }
QDoubleSpinBox#controlValue:focus { color: #ffffff; }
QPushButton:disabled { color: #676767; background: #222222; border-color: #2c2c2c; }
QPushButton#primary { background: #bf0000; color: #ffffff; border: 1px solid #bf0000; }
QPushButton#primary:hover { background: #d51919; }
QPushButton#source:checked { background: #bf0000; color: #ffffff; border-color: #bf0000; }
QPushButton#compareSource { font-size: 25px; font-weight: 700; border: 2px solid #4a4a4a; border-radius: 16px; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #333333, stop:1 #202020); padding: 18px 30px; }
QPushButton#compareSource:hover { border-color: #bf0000; background: #353535; }
QPushButton#compareSource:checked { border-color: #f04040; color: #ffffff; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #bf0000, stop:1 #790000); }
QPushButton#compareSource:focus { border-color: #ffffff; }
QPushButton#comparePlay { font-size: 18px; font-weight: 700; border: 2px solid #4a4a4a; border-radius: 12px; background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #333333,stop:1 #202020); padding: 8px; }
QPushButton#comparePlay:hover { background: #3a3a3a; border-color: #797979; }
QPushButton#comparePlay[playing="true"] { border-color: #ee3434; background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #bf0000,stop:1 #790000); }
QPushButton#comparePlay[playing="true"]:hover { background: #cc1515; }
QPushButton#comparePlay:disabled { color: #888888; border-color: #444444; background: #292929; }
QPushButton#primary:disabled { background: #343434; color: #999999; border-color: #343434; }
QPushButton#step { background: transparent; border: 0; text-align: left; padding: 15px 8px; color: #969696; font-family: 'Michroma'; font-size: 14px; }
QPushButton#step:checked { color: #f0f0f0; background: #2b2b2b; border: 1px solid #bf0000; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #000000; border: 1px solid #414141; border-radius: 7px; padding: 8px; min-height: 20px; }
QSpinBox QLineEdit, QDoubleSpinBox QLineEdit { background: #000000; border: none; padding: 0; }
QSpinBox, QDoubleSpinBox { padding-right: 28px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 25px; margin: 2px 2px 0 0; border: none; background: transparent; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 25px; margin: 0 2px 2px 0; border: none; background: transparent; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url("ASSET_ROOT/plus.svg"); width: 12px; height: 12px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url("ASSET_ROOT/minus.svg"); width: 12px; height: 12px; }
QComboBox { padding-right: 30px; }
QComboBox::drop-down { subcontrol-origin: border; subcontrol-position: top right; width: 27px; margin: 2px; border: none; background: transparent; }
QComboBox::down-arrow { image: url("ASSET_ROOT/chevron.svg"); width: 12px; height: 12px; }
QLineEdit:focus, QComboBox:focus { border: 1px solid #bf0000; }
QComboBox QAbstractItemView { background: #000000; selection-background-color: #4d4d4d; }
QTableWidget, QListWidget, QPlainTextEdit { background: #171717; border: 1px solid #343434; border-radius: 8px; alternate-background-color: #1f1f1f; }
QListWidget::item { padding: 14px; border-bottom: 1px solid #2d2d2d; }
QListWidget::item:selected, QTableWidget::item:selected { background: #3f3f3f; color: #ffffff; }
QHeaderView::section { background: #252525; color: #b8b8b8; border: 0; border-right: 1px solid #4c4c4c; border-bottom: 1px solid #414141; padding: 9px; }
QProgressBar { background: #292929; border: 0; border-radius: 5px; min-height: 10px; text-align: center; }
QProgressBar::chunk { background: #bf0000; border-radius: 5px; }
QScrollArea { border: 0; }
QScrollBar:vertical { background: #1c1c1c; width: 10px; }
QScrollBar::handle:vertical { background: #494949; min-height: 25px; border-radius: 5px; }
QCheckBox { spacing: 8px; padding: 3px; }
QCheckBox::indicator { width: 17px; height: 17px; border: 1px solid #686868; border-radius: 4px; background: #1f1f1f; }
QCheckBox::indicator:checked { background: #1f1f1f; border-color: #686868; image: url("CHECKMARK_URL"); }
QToolTip { background: #2d2d2d; color: #f0f0f0; border: 1px solid #686868; padding: 6px; }
QTabWidget::pane { border: 1px solid #343434; background: #171717; }
QTabBar::tab { background: #242424; color: #b8b8b8; padding: 8px 16px; border: 1px solid #343434; }
QTabBar::tab:selected { color: #ffffff; border-bottom: 2px solid #bf0000; }
"""


class RoutingDiagram(QWidget):
    def __init__(self):
        super().__init__()
        self.channels = (1, 1, 2, 2)
        self.loopback = True
        self.setMinimumHeight(220)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.scale(self.width() / 780, self.height() / 220)
        painter.setFont(QFont("Inter", 10))
        output, input_, loop_out, loop_in = self.channels
        boxes = [
            (15, f"INTERFACE\nOUT {output}"),
            (165, "REAMP BOX\nLine → instrument"),
            (315, "AMP / PEDAL\nYour tone"),
            (465, "LOAD BOX / MIC\nLine / mic return"),
            (615, f"INTERFACE\nIN {input_}"),
        ]
        for x, text in boxes:
            rect = QRectF(x, 38, 140, 72)
            painter.setBrush(QColor("#272727"))
            painter.setPen(QPen(QColor("#4b4b4b"), 1))
            painter.drawRoundedRect(rect, 10, 10)
            painter.setPen(QColor(TEXT))
            painter.drawText(
                rect.adjusted(6, 5, -6, -5), Qt.AlignmentFlag.AlignCenter, text
            )
            if x < 615:
                painter.setPen(QPen(QColor(ACCENT), 2))
                painter.drawLine(QPointF(x + 140, 74), QPointF(x + 150, 74))
                painter.drawLine(QPointF(x + 147, 70), QPointF(x + 151, 74))
                painter.drawLine(QPointF(x + 147, 78), QPointF(x + 151, 74))
        painter.setPen(QColor(MUTED))
        painter.drawText(
            QRectF(15, 8, 740, 23),
            Qt.AlignmentFlag.AlignLeft,
            "MAIN AUDIO PATH  •  Disable direct monitoring and feedback loops",
        )
        painter.drawText(
            QRectF(15, 118, 740, 23),
            Qt.AlignmentFlag.AlignLeft,
            "Speaker output requires a suitable load box; never connect it directly to an interface input.",
        )
        if self.loopback:
            painter.setPen(QPen(QColor("#65c9bf"), 2))
            painter.drawLine(QPointF(75, 175), QPointF(695, 175))
            painter.setPen(QColor("#65c9bf"))
            painter.drawText(
                QRectF(15, 150, 740, 24),
                Qt.AlignmentFlag.AlignCenter,
                f"CLEAN TIMING REFERENCE   •   OUT {loop_out}  →  direct cable  →  IN {loop_in}",
            )
        else:
            painter.drawText(
                QRectF(15, 150, 740, 24),
                Qt.AlignmentFlag.AlignCenter,
                "Timing is measured from the main return. A separate clean loopback is recommended.",
            )


class ControlScene(QWidget):
    def __init__(self):
        super().__init__()
        self.controls, self.values = [], {}
        self.setMinimumHeight(180)

    def set_controls(self, controls, values):
        self.controls, self.values = list(controls), dict(values)
        self._height()
        self.update()

    def _height(self):
        columns = max(1, self.width() // 185)
        self.setMinimumHeight(max(180, math.ceil(len(self.controls) / columns) * 185))

    def resizeEvent(self, event):
        self._height()
        super().resizeEvent(event)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        columns = max(1, self.width() // 185)
        width = self.width() / columns
        for index, control in enumerate(self.controls):
            x, y = (index % columns) * width, (index // columns) * 185
            bounds = QRectF(x + 7, y + 6, width - 14, 170)
            painter.setBrush(QColor("#232323"))
            painter.setPen(QPen(QColor("#3b3b3b"), 1))
            painter.drawRoundedRect(bounds, 12, 12)
            painter.setPen(QColor(TEXT))
            painter.setFont(QFont("Inter", 12, QFont.Weight.Bold))
            painter.drawText(
                bounds.adjusted(8, 10, -8, -132),
                Qt.AlignmentFlag.AlignCenter,
                control.name,
            )
            value = self.values.get(
                control.name,
                (
                    control.enum_names[control.default]
                    if isinstance(control, SwitchSpec)
                    else control.default
                ),
            )
            if isinstance(control, SwitchSpec):
                painter.setFont(QFont("Inter", 10))
                label = str(value)
                painter.setPen(QColor(ACCENT))
                painter.drawText(
                    QRectF(x + 12, y + 48, width - 24, 42),
                    Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                    label,
                )
                count = len(control.enum_names)
                selected = (
                    control.enum_names.index(value)
                    if value in control.enum_names
                    else 0
                )
                for i in range(count):
                    px = x + 24 + (width - 48) * i / max(1, count - 1)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(ACCENT if i == selected else "#4e4e4e"))
                    painter.drawEllipse(
                        QPointF(px, y + 111),
                        5 if i == selected else 3,
                        5 if i == selected else 3,
                    )
                painter.setPen(QColor(MUTED))
                painter.drawText(
                    QRectF(x + 10, y + 133, width - 20, 23),
                    Qt.AlignmentFlag.AlignCenter,
                    f"POSITION {selected + 1} OF {count}",
                )
            else:
                fraction = min(
                    1,
                    max(0, (float(value) - control.min) / (control.max - control.min)),
                )
                center = QPointF(x + width / 2, y + 91)
                path = QPainterPath()
                for i in range(101):
                    angle = math.radians(210 + 300 * i / 100)
                    point = center + QPointF(
                        39 * math.sin(angle), -39 * math.cos(angle)
                    )
                    path.moveTo(point) if i == 0 else path.lineTo(point)
                painter.setPen(
                    QPen(
                        QColor("#4c4c4c"),
                        5,
                        Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap,
                    )
                )
                painter.drawPath(path)
                painter.setBrush(QColor("#2e2e2e"))
                painter.setPen(QPen(QColor("#656565"), 1))
                painter.drawEllipse(center, 30, 30)
                angle = math.radians(210 + fraction * 300)
                painter.setPen(
                    QPen(
                        QColor(ACCENT),
                        3,
                        Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap,
                    )
                )
                painter.drawLine(
                    center + QPointF(10 * math.sin(angle), -10 * math.cos(angle)),
                    center + QPointF(24 * math.sin(angle), -24 * math.cos(angle)),
                )
                painter.setPen(QColor(ACCENT))
                painter.drawText(
                    QRectF(x + 10, y + 136, width - 20, 24),
                    Qt.AlignmentFlag.AlignCenter,
                    f"{float(value):g}",
                )


class ControlDialog(QDialog):
    def __init__(self, parent=None, control=None):
        super().__init__(parent)
        self.setWindowTitle("Edit control" if control else "Add a control")
        self.result_control = None
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit(control.name if control else "")
        self.name.setPlaceholderText("Gain, Bass, Bright, Channel…")
        self.kind = QComboBox()
        self.kind.addItems(["Knob", "Switch"])
        form.addRow("Control name", self.name)
        form.addRow("Type", self.kind)
        layout.addLayout(form)
        self.pages = QStackedWidget()
        knob_page, switch_page = QWidget(), QWidget()
        knob_form, switch_form = QFormLayout(knob_page), QVBoxLayout(switch_page)
        self.minimum, self.maximum, self.step, self.default = [
            QDoubleSpinBox() for _ in range(4)
        ]
        for spin in (self.minimum, self.maximum, self.default):
            spin.setRange(-10000, 10000)
            spin.setDecimals(3)
        self.step.setRange(0.001, 10000)
        self.step.setDecimals(3)
        self.maximum.setValue(10)
        self.step.setValue(0.5)
        self.default.setValue(5)
        self.avoid_zero, self.gain = QCheckBox(
            "Avoid zero (mutes the signal)"
        ), QCheckBox("Gain / Drive control")
        for label, field in (
            ("Minimum", self.minimum),
            ("Maximum", self.maximum),
            ("Step", self.step),
            ("Default", self.default),
            ("", self.avoid_zero),
            ("", self.gain),
        ):
            knob_form.addRow(label, field)
        row = QHBoxLayout()
        self.count = QSpinBox()
        self.count.setRange(2, 1024)
        row.addWidget(self.count)
        switch_form.addLayout(row)
        self.positions = QTableWidget(0, 1)
        self.positions.setHorizontalHeaderLabels(
            ["Position names (in physical switch order)"]
        )
        self.positions.horizontalHeader().setStretchLastSection(True)
        switch_form.addWidget(self.positions)
        self.switch_default = QComboBox()
        switch_form.addWidget(self.switch_default)
        self.count.valueChanged.connect(self.resize_positions)
        self.positions.itemChanged.connect(self._default_names)
        self.resize_positions(2)
        self.pages.addWidget(knob_page)
        self.pages.addWidget(switch_page)
        self.kind.currentIndexChanged.connect(self.pages.setCurrentIndex)
        layout.addWidget(self.pages)
        if isinstance(control, KnobSpec):
            for spin, value in (
                (self.minimum, control.min),
                (self.maximum, control.max),
                (self.step, control.step),
                (self.default, control.default),
            ):
                spin.setValue(value)
            self.avoid_zero.setChecked(control.avoid_zero)
            self.gain.setChecked(control.is_gain)
        elif isinstance(control, SwitchSpec):
            self.kind.setCurrentIndex(1)
            self.count.setValue(len(control.enum_names))
            for i, label in enumerate(control.enum_names):
                self.positions.setItem(i, 0, QTableWidgetItem(label))
            self._default_names()
            self.switch_default.setCurrentIndex(control.default)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(470, 440)

    def resize_positions(self, n):
        old = self.positions.rowCount()
        self.positions.setRowCount(n)
        for i in range(old, n):
            self.positions.setItem(i, 0, QTableWidgetItem(f"Position {i + 1}"))
        self._default_names()

    def _default_names(self):
        previous = self.switch_default.currentIndex()
        self.switch_default.clear()
        for i in range(self.positions.rowCount()):
            item = self.positions.item(i, 0)
            self.switch_default.addItem(f"Default: {item.text() if item else i + 1}")
        self.switch_default.setCurrentIndex(
            max(0, min(previous, self.switch_default.count() - 1))
        )

    def _accept(self):
        try:
            if self.kind.currentIndex() == 1:
                self.result_control = SwitchSpec(
                    self.name.text(),
                    tuple(
                        self.positions.item(i, 0).text()
                        for i in range(self.positions.rowCount())
                    ),
                    self.switch_default.currentIndex(),
                )
            else:
                self.result_control = KnobSpec(
                    self.name.text(),
                    self.minimum.value(),
                    self.maximum.value(),
                    self.step.value(),
                    self.default.value(),
                    self.avoid_zero.isChecked(),
                    self.gain.isChecked(),
                )
        except ValueError as error:
            QMessageBox.warning(self, "Check control settings", str(error))
            return
        self.accept()


class MetadataEditor(QWidget):
    """NAM metadata editor that retains unexposed/custom fields unchanged."""

    changed = Signal(dict)

    def __init__(self, include_name=True, parent=None):
        super().__init__(parent)
        from nam.models.metadata import GearType, ToneType

        self._loading_metadata = False
        self._metadata = {}
        self.fields = {}
        form = QFormLayout(self)
        self.form = form
        form.setContentsMargins(0, 0, 0, 0)
        labels = [("name", "Name")] if include_name else []
        labels += [
            ("modeled_by", "Author"),
            ("gear_make", "Manufacturer"),
            ("gear_model", "Gear model"),
        ]
        for key, label in labels:
            widget = QLineEdit()
            widget.editingFinished.connect(self.emit_metadata)
            self.fields[key] = widget
            form.addRow(label, widget)
        for key, label, enum in (
            ("gear_type", "Gear type", GearType),
            ("tone_type", "Tone type", ToneType),
        ):
            widget = QComboBox()
            widget.addItem("Unspecified", None)
            for value in enum:
                widget.addItem(value.value.replace("_", " ").title(), value.value)
            widget.currentIndexChanged.connect(self.emit_metadata)
            self.fields[key] = widget
            form.addRow(label, widget)
        notes = QPlainTextEdit()
        notes.setMaximumHeight(90)
        notes.textChanged.connect(self.emit_metadata)
        self.fields["notes"] = notes
        form.addRow("Notes", notes)

    def set_name_visible(self, visible):
        if "name" in self.fields:
            self.form.setRowVisible(self.fields["name"], visible)

    def set_metadata(self, metadata):
        self._loading_metadata = True
        try:
            self._metadata = dict(metadata)
            for key, widget in self.fields.items():
                value = metadata.get(key)
                if isinstance(widget, QComboBox):
                    if value is not None and widget.findData(value) < 0:
                        widget.addItem(str(value), value)
                    widget.setCurrentIndex(max(0, widget.findData(value)))
                elif isinstance(widget, QPlainTextEdit):
                    widget.setPlainText(str(value or ""))
                else:
                    widget.setText(str(value or ""))
        finally:
            self._loading_metadata = False

    def emit_metadata(self, *_):
        if self._loading_metadata:
            return
        data = dict(self._metadata)
        for key, widget in self.fields.items():
            if isinstance(widget, QComboBox):
                value = widget.currentData()
            elif isinstance(widget, QPlainTextEdit):
                value = widget.toPlainText().strip()
            else:
                value = widget.text().strip()
            if value is None or value == "":
                data.pop(key, None)
            else:
                data[key] = value
        self._metadata = data
        self.changed.emit(data)

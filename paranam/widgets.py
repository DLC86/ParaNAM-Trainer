"""Native Qt drawing and editors for the capture wizard."""

import math

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
        QApplication.instance().setFont(QFont("Inter", 10))
        _font_loaded = True


ORANGE = "#ff8b4b"
TEXT = "#f0f0f4"
MUTED = "#9296a6"

THEME = """
QWidget { background: #111217; color: #f0f0f4; font-family: 'Inter'; font-size: 13px; }
QMainWindow { background: #111217; }
QFrame#sidebar { background: #191b23; border-right: 1px solid #2a2d39; }
QFrame#card { background: #1b1d26; border: 1px solid #30333f; border-radius: 14px; }
QFrame#card QLabel { background: transparent; }
QLabel, QCheckBox { background: transparent; }
QLabel#eyebrow { color: #ff8b4b; font-size: 11px; font-weight: 700; }
QLabel#title { font-size: 31px; font-weight: 700; }
QLabel#section { font-size: 18px; font-weight: 650; }
QLabel#muted { color: #9296a6; }
QLabel#metric { font-size: 30px; font-weight: 650; }
QLabel#brand { font-size: 22px; font-weight: 750; color: #ff8b4b; }
QPushButton { background: #292c38; border: 1px solid #3c4050; border-radius: 8px; padding: 10px 16px; font-weight: 600; }
QPushButton:hover { background: #353a48; border-color: #74798c; }
QPushButton:disabled { color: #626677; background: #20222b; border-color: #292c38; }
QPushButton#primary { background: #ff8b4b; color: #19140f; border: 1px solid #ff8b4b; }
QPushButton#primary:hover { background: #ffa06d; }
QPushButton#primary:disabled { background: #634635; color: #aaa09a; border-color: #634635; }
QPushButton#step { background: transparent; border: 0; text-align: left; padding: 15px 10px; color: #9296a6; }
QPushButton#step:checked { color: #ffad7d; background: #322922; border: 1px solid #694936; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #12141b; border: 1px solid #3c4050; border-radius: 7px; padding: 8px; min-height: 20px; }
QLineEdit:focus, QComboBox:focus { border: 1px solid #ff8b4b; }
QComboBox QAbstractItemView { background: #232632; selection-background-color: #68452f; }
QTableWidget, QListWidget, QPlainTextEdit { background: #15171e; border: 1px solid #303441; border-radius: 8px; alternate-background-color: #1c1f29; }
QListWidget::item { padding: 14px; border-bottom: 1px solid #292d39; }
QListWidget::item:selected, QTableWidget::item:selected { background: #513a2b; color: #fff1e8; }
QHeaderView::section { background: #222530; color: #b4b8c7; border: 0; border-bottom: 1px solid #3c4050; padding: 9px; }
QProgressBar { background: #252936; border: 0; border-radius: 5px; min-height: 10px; text-align: center; }
QProgressBar::chunk { background: #ff8b4b; border-radius: 5px; }
QScrollArea { border: 0; }
QScrollBar:vertical { background: #191c24; width: 10px; }
QScrollBar::handle:vertical { background: #444958; min-height: 25px; border-radius: 5px; }
QCheckBox { spacing: 8px; padding: 3px; }
QCheckBox::indicator { width: 17px; height: 17px; border: 1px solid #626879; border-radius: 4px; background: #1c1f29; }
QCheckBox::indicator:checked { background: #ff8b4b; border-color: #ff8b4b; }
QToolTip { background: #292d39; color: #f0f0f4; border: 1px solid #626879; padding: 6px; }
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
            painter.setBrush(QColor("#232733"))
            painter.setPen(QPen(QColor("#454b5b"), 1))
            painter.drawRoundedRect(rect, 10, 10)
            painter.setPen(QColor(TEXT))
            painter.drawText(
                rect.adjusted(6, 5, -6, -5), Qt.AlignmentFlag.AlignCenter, text
            )
            if x < 615:
                painter.setPen(QPen(QColor(ORANGE), 2))
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
            painter.setBrush(QColor("#20232d"))
            painter.setPen(QPen(QColor("#353b49"), 1))
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
                painter.setPen(QColor(ORANGE))
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
                    painter.setBrush(QColor(ORANGE if i == selected else "#484e5f"))
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
                        QColor("#464c5b"),
                        5,
                        Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap,
                    )
                )
                painter.drawPath(path)
                painter.setBrush(QColor("#292e3a"))
                painter.setPen(QPen(QColor("#5c6576"), 1))
                painter.drawEllipse(center, 30, 30)
                angle = math.radians(210 + fraction * 300)
                painter.setPen(
                    QPen(
                        QColor(ORANGE),
                        3,
                        Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap,
                    )
                )
                painter.drawLine(
                    center + QPointF(10 * math.sin(angle), -10 * math.cos(angle)),
                    center + QPointF(24 * math.sin(angle), -24 * math.cos(angle)),
                )
                painter.setPen(QColor(ORANGE))
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

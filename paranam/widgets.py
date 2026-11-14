"""Native Qt drawing and editors for the capture wizard."""

import math
import time
from PySide6 import QtSvg  # Include SVG image support in frozen installers.

from .numeric import DecimalSpinBox as QDoubleSpinBox

from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QTimer, QObject, QEvent
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QWidget,
    QDialog,
    QFormLayout,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QDialogButtonBox,
    QStackedWidget,
    QMessageBox,
    QPlainTextEdit,
    QFrame,
    QPushButton,
)

from nam.capture.params import KnobSpec, SwitchSpec
from .control_paint import paint_slide_switch
from .value_format import format_knob_value

# Both Plan and Capture map the full knob range to precisely 300 degrees.
KNOB_START_DEGREES = 210.0
KNOB_SWEEP_DEGREES = 300.0

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


class ComboBoxWheelGuard(QObject):
    """Keep page scrolling from changing a focused selector by accident."""

    def eventFilter(self, watched, event):
        if (
            isinstance(watched, QComboBox)
            and event.type() == QEvent.Type.Wheel
            and not watched.view().isVisible()
        ):
            event.ignore()
            return True
        return super().eventFilter(watched, event)

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
QDoubleSpinBox#controlValue { min-height: 26px; max-height: 26px; }
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
QTableWidget, QListWidget, QPlainTextEdit, QTextBrowser { background: #171717; border: 1px solid #343434; border-radius: 8px; alternate-background-color: #1f1f1f; }
QListWidget::item { padding: 14px; border-bottom: 1px solid #2d2d2d; }
QListWidget::item:selected { background: #3f3f3f; color: #ffffff; }
QTableWidget::item:selected { background: #3f3f3f; }
QHeaderView::section { background: #252525; color: #b8b8b8; border: 0; border-right: 1px solid #4c4c4c; border-bottom: 1px solid #414141; padding: 9px; }
QTableCornerButton::section { background: #252525; border: 0; border-right: 1px solid #4c4c4c; border-bottom: 1px solid #414141; color: #b8b8b8; }
QProgressBar { background: #292929; border: 0; border-radius: 5px; min-height: 10px; text-align: center; }
QProgressBar::chunk { background: #bf0000; border-radius: 5px; }
QScrollArea { border: 0; }
QScrollBar:vertical, QScrollBar:horizontal { background: transparent; border: 0; }
QScrollBar:vertical { width: 10px; }
QScrollBar:horizontal { height: 10px; }
QScrollBar::handle:vertical { background: #494949; min-height: 25px; border-radius: 5px; }
QScrollBar::handle:horizontal { background: #494949; min-width: 25px; border-radius: 5px; }
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { background: transparent; border: 0; }
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
        self.blink_names = set()
        self.blink_started = 0.0
        self.blink_timer = QTimer(self)
        self.blink_timer.setInterval(40)
        self.blink_timer.timeout.connect(self.update)
        self.setMinimumHeight(180)

    def set_controls(self, controls, values):
        if dict(values) != self.values:
            self.stop_blink()
        self.controls, self.values = list(controls), dict(values)
        self._height()
        self.update()

    def start_blink(self, names):
        self.blink_names = set(names)
        self.blink_started = time.monotonic()
        if self.blink_names:
            self.blink_timer.start()
        self.update()

    def stop_blink(self):
        self.blink_names.clear()
        self.blink_timer.stop()
        self.update()

    def _columns(self):
        return max(1, self.width() // 185)

    def _tile_count(self):
        return len(self.controls)

    def _bounds(self, index):
        columns = self._columns()
        width = self.width() / columns
        return QRectF((index % columns) * width + 7, (index // columns) * 185 + 6, width - 14, 170)

    def _paint_bounds(self, index):
        """Return the bounds used to paint a card.

        Subclasses can animate cards without changing the stable layout used for
        hit testing and the surrounding scroll area.
        """
        return self._bounds(index)

    def _card_border(self, index):
        return QColor(ACCENT if index == getattr(self, "current", -1) else "#3b3b3b")

    def _paint_order(self):
        return range(len(self.controls))

    def _height(self):
        self.setMinimumHeight(max(180, math.ceil(self._tile_count() / self._columns()) * 185))

    def resizeEvent(self, event):
        self._height()
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        columns = self._columns()
        width = self.width() / columns
        for index in self._paint_order():
            control = self.controls[index]
            bounds = self._paint_bounds(index)
            # Selecting or hovering one card must not rebuild the expensive
            # knob paths for every off-screen/unaffected control.
            if not event.region().intersects(bounds.toAlignedRect()):
                continue
            x, y = bounds.x() - 7, bounds.y() - 6
            width = bounds.width() + 14
            shade = 35
            if control.name in self.blink_names:
                pulse = (
                    1
                    - math.cos(
                        (time.monotonic() - self.blink_started) * 2 * math.pi / 2.6
                    )
                ) / 2
                shade += round(45 * pulse)
            painter.setBrush(QColor(shade, shade, shade))
            painter.setPen(QPen(self._card_border(index), 1))
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
                count = len(control.enum_names)
                selected = (
                    control.enum_names.index(value)
                    if value in control.enum_names
                    else 0
                )
                position = (
                    selected if control.option_indices is None else control.option_indices[selected]
                )
                switch_width = min(width - 30, max(84, (count - 1) * 36 + 32))
                left = x + (width - switch_width) / 2
                paint_slide_switch(
                    painter,
                    QRectF(left, y + 37, switch_width, 108),
                    position,
                    count,
                    16,
                )
                painter.setFont(QFont("Inter", 15, QFont.Weight.Bold))
                painter.setPen(QColor(ACCENT))
                text = painter.fontMetrics().elidedText(
                    control.enum_names[selected], Qt.TextElideMode.ElideRight, int(width - 20)
                )
                painter.drawText(
                    QRectF(x + 10, y + 134, width - 20, 29),
                    Qt.AlignmentFlag.AlignCenter,
                    text,
                )
            else:
                fraction = min(
                    1,
                    max(0, (float(value) - control.min) / (control.max - control.min)),
                )
                center = QPointF(x + width / 2, y + 91)
                path = QPainterPath()
                for i in range(101):
                    angle = math.radians(KNOB_START_DEGREES + KNOB_SWEEP_DEGREES * i / 100)
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
                origin = -control.min / (control.max - control.min) if control.min <= 0 <= control.max else 0
                if fraction != origin:
                    painter.setPen(QPen(QColor(ACCENT), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                    ring = QRectF(center.x() - 39, center.y() - 39, 78, 78)
                    painter.drawArc(ring, round((90 - KNOB_START_DEGREES - origin * KNOB_SWEEP_DEGREES) * 16), round((origin - fraction) * KNOB_SWEEP_DEGREES * 16))
                painter.setFont(QFont("Inter", 8))
                painter.setPen(QColor(MUTED))
                for side, limit in ((-1, control.min), (1, control.max)):
                    painter.drawText(
                        QRectF(center.x() + side * 37 - 22, center.y() + 30, 44, 15),
                        Qt.AlignmentFlag.AlignCenter,
                        format_knob_value(limit, control.min, control.max),
                    )
                painter.setBrush(QColor("#2e2e2e"))
                painter.setPen(QPen(QColor("#656565"), 1))
                painter.drawEllipse(center, 30, 30)
                angle = math.radians(KNOB_START_DEGREES + fraction * KNOB_SWEEP_DEGREES)
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
                painter.setFont(QFont("Inter", 15, QFont.Weight.Bold))
                painter.drawText(
                    QRectF(x + 10, y + 134, width - 20, 29),
                    Qt.AlignmentFlag.AlignCenter,
                    format_knob_value(float(value), control.min, control.max),
                )


class PlanControlScene(ControlScene):
    """Capture-style control cards with editing and reordering in Plan."""

    selected = Signal(int)
    reordered = Signal(int, int)
    add_requested = Signal()
    remove_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__()
        self.setParent(parent)
        self.controls = []
        self.current = -1
        self._drag_index = None
        self._drag_target = None
        self._drag_position = QPointF()
        self._drag_offset = QPointF()
        self._card_positions = {}
        self._hover_index = -1
        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(16)
        self._animation_timer.timeout.connect(self._advance_card_animation)
        self.setMouseTracking(True)

    def set_controls(self, controls, current=-1):
        controls = list(controls)
        previous_current = self.current
        if controls == self.controls:
            if current != previous_current:
                self.current = current
                for index in (previous_current, current):
                    if 0 <= index < len(self.controls):
                        self.update(self._paint_bounds(index).toAlignedRect())
            return
        self.controls = controls
        self.current = current
        if self._drag_index is None:
            self._card_positions.clear()
        self._height()
        self.update()

    def _tile_count(self):
        return len(self.controls) + 1

    def _tile_index(self, point):
        for index in range(self._tile_count()):
            bounds = self._add_bounds() if index == len(self.controls) else self._bounds(index)
            if bounds.contains(point):
                return index
        return -1

    def _add_bounds(self):
        # Exact geometry of the neighboring control cards, including on wrapped rows.
        return self._bounds(len(self.controls))

    def _remove_bounds(self, index):
        bounds = self._bounds(index)
        return QRectF(bounds.right() - 24, bounds.top() + 4, 20, 20)

    def _paint_bounds(self, index):
        """Use animated locations while retaining fixed bounds for hit tests."""
        if index == self._drag_index:
            base = self._bounds(index)
            return base.translated(self._drag_position - self._drag_offset - base.topLeft())
        position = self._card_positions.get(index)
        return self._bounds(index).translated(position - self._bounds(index).topLeft()) if position else self._bounds(index)

    def _card_border(self, index):
        if index == self._drag_index or index == self._hover_index:
            return QColor("#d24a4a")
        return super()._card_border(index)

    def _paint_order(self):
        if self._drag_index is None:
            return super()._paint_order()
        return [
            index for index in range(len(self.controls)) if index != self._drag_index
        ] + [self._drag_index]

    def _set_drop_target(self, point):
        target = self._tile_index(point)
        if not 0 <= target < len(self.controls):
            target = self._drag_index
        if target != self._drag_target:
            self._drag_target = target
            self._animation_timer.start()

    def _target_positions(self):
        if self._drag_index is None:
            return {}
        order = [index for index in range(len(self.controls)) if index != self._drag_index]
        order.insert(self._drag_target, self._drag_index)
        return {
            index: self._bounds(slot).topLeft()
            for slot, index in enumerate(order)
            if index != self._drag_index
        }

    def _advance_card_animation(self):
        targets = self._target_positions()
        moving = False
        for index, target in targets.items():
            current = self._card_positions.get(index, self._bounds(index).topLeft())
            delta = target - current
            if abs(delta.x()) < 0.5 and abs(delta.y()) < 0.5:
                self._card_positions[index] = target
            else:
                self._card_positions[index] = current + delta * 0.28
                moving = True
        self.update()
        if not moving:
            self._animation_timer.stop()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            index = self._tile_index(event.position())
            if index == len(self.controls):
                self.add_requested.emit()
            elif index >= 0:
                if self._remove_bounds(index).contains(event.position()):
                    self.remove_requested.emit(index)
                    return
                previous_current = self.current
                self._drag_index = index
                self._drag_target = index
                self._drag_position = event.position()
                self._drag_offset = event.position() - self._bounds(index).topLeft()
                self._card_positions = {
                    card_index: self._bounds(card_index).topLeft()
                    for card_index in range(len(self.controls))
                    if card_index != index
                }
                self.current = index
                self.selected.emit(index)
                for changed in (previous_current, index):
                    if 0 <= changed < len(self.controls):
                        self.update(self._paint_bounds(changed).toAlignedRect())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_index is not None:
            self._drag_position = event.position()
            self._set_drop_target(event.position())
            self.update()
        else:
            hover = self._tile_index(event.position())
            if hover != self._hover_index:
                previous = self._hover_index
                self._hover_index = hover
                for index in (previous, hover):
                    if 0 <= index < len(self.controls):
                        self.update(self._paint_bounds(index).toAlignedRect())
                    elif index == len(self.controls):
                        self.update(self._add_bounds().toAlignedRect())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._drag_index is None and self._hover_index != -1:
            previous = self._hover_index
            self._hover_index = -1
            if 0 <= previous < len(self.controls):
                self.update(self._paint_bounds(previous).toAlignedRect())
            elif previous == len(self.controls):
                self.update(self._add_bounds().toAlignedRect())
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_index is not None:
            self._drag_position = event.position()
            self._set_drop_target(event.position())
            source = self._drag_index
            target = self._drag_target
            self._drag_index = None
            self._drag_target = None
            self._card_positions.clear()
            self._animation_timer.stop()
            self._hover_index = self._tile_index(event.position())
            if 0 <= target < len(self.controls) and target != source:
                self.reordered.emit(source, target)
            else:
                # The dragged card may have crossed several old paint regions.
                # Clear them all when a drag is cancelled or returns home.
                self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor(ACCENT))
        painter.setFont(QFont("Inter", 12, QFont.Weight.Bold))
        for index in range(len(self.controls)):
            if index != self._drag_index:
                bounds = self._paint_bounds(index)
                remove = QRectF(bounds.right() - 24, bounds.top() + 4, 20, 20)
                painter.drawText(remove, Qt.AlignmentFlag.AlignCenter, "×")
        bounds = self._add_bounds()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        add_hovered = self._hover_index == len(self.controls)
        add_color = QColor(ACCENT if add_hovered else MUTED)
        painter.setPen(QPen(add_color, 2, Qt.PenStyle.DashLine))
        painter.drawRoundedRect(bounds, 12, 12)
        painter.setPen(QPen(add_color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        center = bounds.center()
        painter.drawLine(center - QPointF(15, 0), center + QPointF(15, 0))
        painter.drawLine(center - QPointF(0, 15), center + QPointF(0, 15))


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
        self.sampling_bias = QComboBox()
        self.sampling_bias.addItem("Uniform range", "linear")
        self.sampling_bias.addItem("Most change heard in the lower half", "lower")
        self.sampling_bias.addItem("Most change heard in the upper half", "upper")
        self.gain = QCheckBox("Gain/Drive control")
        self.clean_volume = QCheckBox("Linear volume control")
        self.clean_volume.setToolTip(
            "Train at maximum volume and measure a separate short-signal attenuation curve. "
            "Use only for an output attenuator that does not change tone or distortion. "
            "A silent minimum is identified by the absence of the timing test signal, "
            "not by a zero-noise recording."
        )
        self.avoid_zero = QCheckBox("Avoid min")
        self.avoid_max = QCheckBox("Avoid max")
        for label, field in (
            ("Minimum", self.minimum),
            ("Maximum", self.maximum),
            ("Step", self.step),
            ("Default", self.default),
            ("Capture density", self.sampling_bias),
            ("", self.gain),
            ("", self.clean_volume),
            ("", self.avoid_zero),
            ("", self.avoid_max),
        ):
            knob_form.addRow(label, field)
        self.switch_mode = QComboBox()
        self.switch_mode.addItem(
            "Complex Switch — separate plan for every state", "complex"
        )
        self.switch_mode.addItem("Simple Switch — one discrete LHS dimension", "simple")
        switch_form.addWidget(self.switch_mode)
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
            self.avoid_max.setChecked(control.avoid_max)
            self.sampling_bias.setCurrentIndex(
                self.sampling_bias.findData(control.sampling_bias)
            )
            self.clean_volume.setChecked(control.clean_volume)
        elif isinstance(control, SwitchSpec):
            self.kind.setCurrentIndex(1)
            self.count.setValue(len(control.enum_names))
            for i, label in enumerate(control.enum_names):
                self.positions.setItem(i, 0, QTableWidgetItem(label))
            self._default_names()
            self.switch_default.setCurrentIndex(control.default)
            self.switch_mode.setCurrentIndex(
                self.switch_mode.findData(control.planning_mode)
            )
        self.clean_volume.toggled.connect(self._clean_volume_changed)
        self.gain.toggled.connect(self._gain_changed)
        self._clean_volume_changed(self.clean_volume.isChecked(), reset_step=False)
        self._gain_changed(self.gain.isChecked())
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(600, 610)

    def _clean_volume_changed(self, checked, reset_step=True):
        if checked and reset_step:
            self.step.setValue(1.0)
        for field in (self.avoid_zero, self.avoid_max, self.gain):
            if checked:
                field.setChecked(False)
            field.setEnabled(not checked)
        self.sampling_bias.setEnabled(not checked)

    def _gain_changed(self, checked):
        # The same knob cannot be an input to the hypernetwork AND a
        # post-network linear attenuator. Disabling the alternative in both
        # directions prevents an invalid combination before Save.
        self.clean_volume.setEnabled(not checked)

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
                    self.switch_mode.currentData(),
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
                    avoid_max=self.avoid_max.isChecked(),
                    sampling_bias=self.sampling_bias.currentData(),
                    clean_volume=self.clean_volume.isChecked(),
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
        # Keep Tone type immediately before the optional numerical metadata.
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
        for key, label in (
            ("input_level_dbu", "Input dBu level"),
            ("output_level_dbu", "Output dBu level"),
        ):
            widget = QLineEdit()
            widget.setToolTip("Analog dBu level corresponding to 0 dBFS")
            widget.editingFinished.connect(self.emit_metadata)
            self.fields[key] = widget
            form.addRow(label, widget)
        impedance = QLineEdit()
        impedance.editingFinished.connect(self.commit_impedance)
        self.fields["input_z"] = impedance
        form.addRow("Input impedance", impedance)
        notes = QPlainTextEdit()
        notes.setMaximumHeight(90)
        notes.textChanged.connect(self.emit_metadata)
        self.fields["notes"] = notes
        form.addRow("Notes", notes)

    def commit_impedance(self):
        from .impedance import format_impedance, parse_impedance

        widget = self.fields["input_z"]
        try:
            value = parse_impedance(widget.text())
        except ValueError as error:
            widget.setToolTip(str(error))
            widget.selectAll()
            return
        widget.setToolTip("")
        widget.setText(format_impedance(value))
        self.emit_metadata()

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
                elif key == "input_z":
                    from .impedance import format_impedance
                    widget.setText(format_impedance(value))
                    widget.setToolTip("")
                elif key in ("input_level_dbu", "output_level_dbu"):
                    widget.setText("" if value is None else str(value))
                    widget.setToolTip("")
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
            elif key == "input_z":
                from .impedance import parse_impedance
                try:
                    value = parse_impedance(widget.text())
                except ValueError:
                    continue
            elif key in ("input_level_dbu", "output_level_dbu"):
                raw = widget.text().strip()
                try:
                    value = float(raw.replace(",", ".")) if raw else None
                    if value is not None and not math.isfinite(value):
                        raise ValueError("Level must be finite")
                    widget.setToolTip("")
                except ValueError:
                    widget.setToolTip("Enter a valid finite value in dBu")
                    continue
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


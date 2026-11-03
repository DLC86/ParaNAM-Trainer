"""Small native widgets for navigation, recent projects and capture selection."""

import math

from PySide6.QtCore import Qt, QSize, QRectF, QPointF
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QTableWidget,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
)


class CaptureTable(QTableWidget):
    @property
    def selection_locked(self):
        return getattr(self, "_selection_locked", False)

    @selection_locked.setter
    def selection_locked(self, value):
        self._selection_locked = value
        self.verticalHeader().setSectionsClickable(not value)
        self.horizontalHeader().setSectionsClickable(not value)

    def mousePressEvent(self, event):
        if self.selection_locked:
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.selection_locked:
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event):
        if self.selection_locked:
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if self.selection_locked:
            scrollbar = self.verticalScrollBar()
            delta = {
                Qt.Key.Key_Down: scrollbar.singleStep(),
                Qt.Key.Key_Up: -scrollbar.singleStep(),
                Qt.Key.Key_PageDown: scrollbar.pageStep(),
                Qt.Key.Key_PageUp: -scrollbar.pageStep(),
            }.get(event.key())
            if delta is not None:
                scrollbar.setValue(scrollbar.value() + delta)
            event.accept()
        else:
            super().keyPressEvent(event)


class RecentProjectDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        return QSize(200, 72)

    def paint(self, painter, option, index):
        option = QStyleOptionViewItem(option)
        self.initStyleOption(option, index)
        title, _, path = option.text.partition("\n")
        option.text = ""
        option.widget.style().drawControl(
            QStyle.ControlElement.CE_ItemViewItem, option, painter, option.widget
        )
        painter.save()
        bounds = option.rect.adjusted(16, 8, -16, -8)
        painter.setFont(QFont("Inter", 12, QFont.Weight.Bold))
        painter.setPen(QColor("#f0f0f0"))
        painter.drawText(
            bounds.adjusted(0, 0, 0, -26),
            Qt.AlignmentFlag.AlignVCenter,
            painter.fontMetrics().elidedText(
                title, Qt.TextElideMode.ElideRight, bounds.width()
            ),
        )
        painter.setFont(QFont("Inter", 9))
        painter.setPen(QColor("#a0a0a0"))
        painter.drawText(
            bounds.adjusted(0, 27, 0, 0),
            Qt.AlignmentFlag.AlignVCenter,
            painter.fontMetrics().elidedText(
                path, Qt.TextElideMode.ElideMiddle, bounds.width()
            ),
        )
        painter.restore()


def navigation_icon(kind):
    icon = QIcon()
    for state, color in ((QIcon.State.Off, "#b0b0b0"), (QIcon.State.On, "#ef3636")):
        pixmap = QPixmap(48, 48)
        pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.scale(2, 2)
        p.setPen(
            QPen(QColor(color), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        )
        p.setBrush(Qt.BrushStyle.NoBrush)
        if kind == 0:  # Audio interface and routing.
            p.drawRoundedRect(QRectF(2, 5, 20, 14), 3, 3)
            for x in (7, 17):
                p.drawEllipse(QPointF(x, 12), 2.4, 2.4)
            p.drawLine(10, 3, 14, 3)
            p.drawLine(10, 21, 14, 21)
        elif kind == 1:  # Knob.
            p.drawArc(QRectF(3, 3, 18, 18), 220 * 16, -260 * 16)
            p.drawEllipse(QRectF(6, 6, 12, 12))
            p.drawLine(12, 12, 15, 8)
        elif kind == 2:  # Capture waveform.
            for x, height in ((3, 3), (7, 7), (11, 10), (15, 5), (19, 8), (23, 2)):
                p.drawLine(x, 12 - height, x, 12 + height)
        elif kind == 3:  # Neural network.
            for a in (6, 18):
                for b in (4, 12, 20):
                    p.drawLine(3, a, 12, b)
                    p.drawLine(12, b, 21, a)
            p.setBrush(QColor("#1b1b1b"))
            for x, ys in ((3, (6, 18)), (12, (4, 12, 20)), (21, (6, 18))):
                for y in ys:
                    p.drawEllipse(QPointF(x, y), 2, 2)
        elif kind == 4:  # Queue.
            for y in (5, 12, 19):
                p.drawEllipse(QPointF(4, y), 1, 1)
                p.drawLine(9, y, 21, y)
        else:  # Guitar.
            p.save()
            p.translate(12, 12)
            p.rotate(38)
            p.drawRoundedRect(QRectF(-1.5, -11, 3, 12), 1, 1)
            p.drawEllipse(QRectF(-5, -1, 10, 11))
            p.drawEllipse(QRectF(-3.5, -3, 7, 7))
            p.drawEllipse(QPointF(0, 3), 1.3, 1.3)
            p.restore()
        p.end()
        icon.addPixmap(pixmap, QIcon.Mode.Normal, state)
    return icon

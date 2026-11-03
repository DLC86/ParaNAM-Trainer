"""Small native widgets for navigation, recent projects and capture selection."""

import math

from PySide6.QtCore import Qt, QSize, QRectF, QPointF
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPen, QPixmap
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
    @staticmethod
    def fonts():
        title = QFont("Inter")
        title.setPixelSize(16)
        title.setWeight(QFont.Weight.Bold)
        path = QFont("Inter")
        path.setPixelSize(12)
        path.setWeight(QFont.Weight.Normal)
        return title, path

    def sizeHint(self, option, index):
        title, path = self.fonts()
        return QSize(
            200, QFontMetrics(title).height() + QFontMetrics(path).height() + 30
        )

    def paint(self, painter, option, index):
        title, _, path = str(index.data() or "").partition("\n")
        painter.save()
        # Avoid QStyleSheetStyle applying its item padding/font/clip a second time.
        painter.fillRect(
            option.rect,
            QColor(
                "#3f3f3f"
                if option.state & QStyle.StateFlag.State_Selected
                else "#171717"
            ),
        )
        painter.setPen(QColor("#2d2d2d"))
        painter.drawLine(option.rect.bottomLeft(), option.rect.bottomRight())
        title_font, path_font = self.fonts()
        bounds = option.rect.adjusted(16, 12, -16, -12)
        title_height = QFontMetrics(title_font).height()
        painter.setFont(title_font)
        painter.setPen(QColor("#f0f0f0"))
        painter.drawText(
            bounds.adjusted(0, 0, 0, -(bounds.height() - title_height)),
            Qt.AlignmentFlag.AlignVCenter,
            painter.fontMetrics().elidedText(
                title, Qt.TextElideMode.ElideRight, bounds.width()
            ),
        )
        painter.setFont(path_font)
        painter.setPen(QColor("#a0a0a0"))
        painter.drawText(
            bounds.adjusted(0, title_height + 6, 0, 0),
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
        if kind == 0:  # Projects folder.
            p.drawRoundedRect(QRectF(2, 7, 20, 14), 2, 2)
            p.drawPolyline(
                [QPointF(3, 7), QPointF(3, 3), QPointF(10, 3), QPointF(13, 7)]
            )
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
        elif kind == 5:  # Guitar.
            p.save()
            p.translate(12, 12)
            p.rotate(38)
            p.drawRoundedRect(QRectF(-1.5, -11, 3, 12), 1, 1)
            p.drawEllipse(QRectF(-5, -1, 10, 11))
            p.drawEllipse(QRectF(-3.5, -3, 7, 7))
            p.drawEllipse(QPointF(0, 3), 1.3, 1.3)
            p.restore()
        else:  # Settings gear.
            for i in range(8):
                angle = math.radians(i * 45)
                p.drawLine(
                    QPointF(12 + 8 * math.cos(angle), 12 + 8 * math.sin(angle)),
                    QPointF(12 + 10.5 * math.cos(angle), 12 + 10.5 * math.sin(angle)),
                )
            p.drawEllipse(QPointF(12, 12), 8, 8)
            p.drawEllipse(QPointF(12, 12), 3, 3)
        p.end()
        icon.addPixmap(pixmap, QIcon.Mode.Normal, state)
    return icon

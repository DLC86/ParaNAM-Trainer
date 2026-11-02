"""Dependency-free Qt chart of completed validation epochs."""

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class ESRChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []
        self.setMinimumHeight(190)
        self.setMaximumHeight(240)
        self.setAccessibleName("Full and Lite validation ESR by epoch")

    def set_history(self, history):
        history = [dict(entry) for entry in history]
        if history != self.history:
            self.history = history
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#292929"))
        painter.setPen(QColor("#eeeeee"))
        painter.drawText(12, 22, "Validation ESR · log scale · completed epochs")
        values = [
            max(float(entry[key]), 1e-12)
            for entry in self.history
            for key in ("full", "lite")
            if entry.get(key) is not None
            and math.isfinite(entry[key])
            and entry[key] >= 0
        ]
        if not values:
            painter.setPen(QColor("#bcbcbc"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "The curves appear after the first validation epoch.",
            )
            return
        bottom, top = math.floor(math.log10(min(values))), math.ceil(
            math.log10(max(values))
        )
        if bottom == top:
            bottom -= 1
        plot = QRectF(65, 47, max(self.width() - 90, 1), max(self.height() - 78, 1))
        last_epoch = max(2, max(entry["epoch"] for entry in self.history))

        def point(epoch, value):
            return QPointF(
                plot.left() + (epoch - 1) / (last_epoch - 1) * plot.width(),
                plot.bottom()
                - (math.log10(max(value, 1e-12)) - bottom)
                / (top - bottom)
                * plot.height(),
            )

        for tick in range(4):
            exponent = bottom + (top - bottom) * tick / 3
            y = plot.bottom() - plot.height() * tick / 3
            painter.setPen(QPen(QColor("#494949"), 1))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor("#bcbcbc"))
            painter.drawText(
                QRectF(0, y - 9, 58, 18),
                Qt.AlignmentFlag.AlignRight,
                f"{10 ** exponent:.2g}",
            )
        painter.drawText(
            QRectF(plot.left() - 10, plot.bottom() + 5, 25, 20),
            Qt.AlignmentFlag.AlignCenter,
            "1",
        )
        painter.drawText(
            QRectF(plot.right() - 30, plot.bottom() + 5, 55, 20),
            Qt.AlignmentFlag.AlignRight,
            str(last_epoch),
        )
        painter.drawText(
            QRectF(plot.center().x() - 40, plot.bottom() + 5, 80, 20),
            Qt.AlignmentFlag.AlignCenter,
            "Epoch",
        )
        for index, (size, color) in enumerate(
            (("full", "#ff7777"), ("lite", "#7fc7ed"))
        ):
            painter.setPen(QPen(QColor(color), 2))
            path = QPainterPath()
            connected = False
            for entry in self.history:
                value = entry.get(size)
                if value is None or not math.isfinite(value) or value < 0:
                    connected = False
                    continue
                pos = point(entry["epoch"], value)
                if connected:
                    path.lineTo(pos)
                else:
                    path.moveTo(pos)
                painter.drawEllipse(pos, 2, 2)
                connected = True
            painter.drawPath(path)
            last = self.history[-1].get(size)
            text = (
                f"{size.title()}  {last:.6g}"
                if last is not None and math.isfinite(last)
                else size.title()
            )
            painter.drawText(15 + index * 200, 40, text)

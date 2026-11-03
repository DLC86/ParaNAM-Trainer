"""Dependency-free Qt chart of held-out ESR at completed epochs."""

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QTabWidget, QWidget


class ESRCharts(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.charts = {}
        for size in ("full", "lite"):
            chart = ESRChart(size=size)
            self.charts[size] = chart
            self.addTab(chart, f"{size.title()} ESR chart")

    def set_history(self, history):
        for chart in self.charts.values():
            chart.set_history(history)


class ESRChart(QWidget):
    def __init__(self, parent=None, size="full"):
        super().__init__(parent)
        self.model_size = size
        self.history = []
        self.setMinimumHeight(215)
        self.setMaximumHeight(265)
        self.setAccessibleName(f"{size.title()} Validation ESR and Train ESR by epoch")
        self.setToolTip(
            "Solid: validation captures. Dashed: held-out audio in training captures. "
            "Each score is an equal-weight mean over captures, evaluated at epoch end."
        )

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
        painter.drawText(
            12, 22, f"{self.model_size.title()} ESR · log scale · completed epochs"
        )
        values = [
            max(float(entry[key]), 1e-12)
            for entry in self.history
            for key in (self.model_size, f"train_{self.model_size}")
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
        for index, (size, color, title) in enumerate(
            (
                (self.model_size, "#7fc7ed", "Validation ESR"),
                (f"train_{self.model_size}", "#ffbb77", "Train ESR"),
            )
        ):
            pen = QPen(QColor(color), 2)
            if size.startswith("train_"):
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
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
                f"{title}  {last:.6g}"
                if last is not None and math.isfinite(last)
                else title
            )
            painter.drawText(15 + index * max(220, self.width() // 2), 40, text)

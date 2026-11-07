"""Shared slide-switch drawing for the Capture diagram and Compare controls."""

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QPen


def paint_slide_switch(painter, rect, selected, count, padding, focused=False):
    middle = rect.center().y()
    painter.setPen(QPen(QColor("#606060"), 1))
    painter.setBrush(QColor("#202020"))
    painter.drawRoundedRect(
        QRectF(rect.x() + 6, middle - 10, rect.width() - 12, 20), 5, 5
    )

    def position(index):
        return (
            rect.x()
            + padding
            + index * (rect.width() - 2 * padding) / max(1, count - 1)
        )

    for index in range(count):
        x = position(index)
        painter.drawLine(round(x), round(middle - 5), round(x), round(middle + 5))
    painter.setPen(QPen(QColor("#bf0000" if focused else "#a0a0a0"), 2))
    painter.setBrush(QColor("#454545"))
    x = position(selected)
    painter.drawRoundedRect(QRectF(x - 11, middle - 17, 22, 34), 3, 3)
    painter.setPen(QPen(QColor("#b8b8b8"), 1))
    for offset in (-4, 0, 4):
        painter.drawLine(
            round(x + offset), round(middle - 8), round(x + offset), round(middle + 8)
        )

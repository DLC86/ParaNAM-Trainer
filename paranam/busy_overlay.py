"""Animated feedback for foreground calculations."""

from PySide6.QtCore import QEvent, QEventLoop, QObject, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget


class _Calculation(QThread):
    completed = Signal()

    def __init__(self, calculate):
        super().__init__()
        self.calculate = calculate
        self.result = self.error = None

    def run(self):
        try:
            self.result = self.calculate()
        except Exception as error:
            self.error = error
        finally:
            self.completed.emit()


class _Overlay(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAccessibleName("Calculating")
        self.setCursor(Qt.CursorShape.WaitCursor)
        self.angle = 0
        self.timer = QTimer(self, interval=16)
        self.timer.timeout.connect(self._advance)

    def _advance(self):
        self.angle = (self.angle + 7) % 360
        center = self.rect().center()
        self.update(center.x() - 23, center.y() - 23, 46, 46)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 95))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = self.rect().center()
        ring = QRectF(center.x() - 19, center.y() - 19, 38, 38)
        painter.setPen(QPen(QColor(255, 255, 255, 65), 4))
        painter.drawEllipse(ring)
        painter.setPen(QPen(QColor("white"), 4, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap))
        painter.drawArc(ring, -self.angle * 16, 250 * 16)


class _InputGuard(QObject):
    def __init__(self, window, overlay):
        super().__init__()
        self.window = window
        self.overlay = overlay

    def eventFilter(self, watched, event):
        kind = event.type()
        if watched is self.overlay.parentWidget() and kind == QEvent.Type.Resize:
            self.overlay.setGeometry(watched.rect())
        if kind == QEvent.Type.Close and watched is self.window:
            event.ignore()
            return True
        if kind in (
            QEvent.Type.KeyPress, QEvent.Type.KeyRelease, QEvent.Type.Shortcut,
            QEvent.Type.ShortcutOverride, QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease, QEvent.Type.MouseButtonDblClick,
            QEvent.Type.Wheel, QEvent.Type.ContextMenu, QEvent.Type.TouchBegin,
        ):
            if kind in (QEvent.Type.Shortcut, QEvent.Type.ShortcutOverride):
                return True
            return isinstance(watched, QWidget) and watched.window() is self.window
        return False


def run_calculation(window, calculate):
    """Run a widget-free calculation and let Qt paint while it completes."""
    overlay = _Overlay(window.centralWidget())
    overlay.setGeometry(overlay.parentWidget().rect())
    worker = _Calculation(calculate)
    loop = QEventLoop()
    worker.completed.connect(loop.quit, Qt.ConnectionType.QueuedConnection)
    guard = _InputGuard(window, overlay)
    app = QApplication.instance()
    app.installEventFilter(guard)
    was_calculating = getattr(window, "_foreground_calculation", False)
    window._foreground_calculation = True
    # Show feedback only if the operation outlasts a normal UI interaction.
    reveal = QTimer(overlay, singleShot=True, interval=150)

    def show():
        overlay.show()
        overlay.raise_()
        overlay.timer.start()

    reveal.timeout.connect(show)
    reveal.start()
    try:
        worker.start()
        loop.exec()
    finally:
        worker.wait()
        reveal.stop()
        overlay.timer.stop()
        app.removeEventFilter(guard)
        window._foreground_calculation = was_calculating
        overlay.hide()
        overlay.deleteLater()
    if worker.error is not None:
        raise worker.error
    return worker.result

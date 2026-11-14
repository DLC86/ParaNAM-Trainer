"""Animated feedback for foreground calculations."""

from contextlib import contextmanager

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
        if kind == QEvent.Type.Close and watched is self.window and self.window._foreground_calculation:
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


class _Feedback(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.overlay = _Overlay(window.centralWidget())
        self.overlay.hide()
        self.guard = _InputGuard(window, self.overlay)
        self.depth = 0
        self.reveal = QTimer(self, singleShot=True, interval=150)
        self.reveal.timeout.connect(self.show)
        self.dismiss = QTimer(self, singleShot=True, interval=250)
        self.dismiss.timeout.connect(self.hide)

    def show(self):
        self.overlay.setGeometry(self.overlay.parentWidget().rect())
        self.overlay.show()
        self.overlay.raise_()
        self.overlay.timer.start()

    def hide(self):
        self.overlay.hide()
        self.overlay.timer.stop()
        QApplication.instance().removeEventFilter(self.guard)


@contextmanager
def foreground_operation(window):
    feedback = getattr(window, "_calculation_feedback", None)
    if feedback is None:
        feedback = window._calculation_feedback = _Feedback(window)
    if feedback.depth == 0:
        feedback.dismiss.stop()
        QApplication.instance().installEventFilter(feedback.guard)
        window._foreground_calculation = True
        if feedback.overlay.isHidden():
            feedback.reveal.start()
    feedback.depth += 1
    try:
        yield
    finally:
        feedback.depth -= 1
        if feedback.depth == 0:
            window._foreground_calculation = False
            feedback.reveal.stop()
            if feedback.overlay.isHidden():
                feedback.hide()
            else:
                feedback.dismiss.start()


def run_calculation(window, calculate):
    """Run a widget-free calculation and let Qt paint while it completes."""
    worker = _Calculation(calculate)
    loop = QEventLoop()
    worker.completed.connect(loop.quit, Qt.ConnectionType.QueuedConnection)
    with foreground_operation(window):
        worker.start()
        try:
            loop.exec()
        finally:
            worker.wait()
    if worker.error is not None:
        raise worker.error
    return worker.result

import time

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMainWindow, QPushButton

from paranam.busy_overlay import _Overlay, run_calculation


def test_calculation_animates_and_blocks_clicks(qapp):
    window = QMainWindow()
    button = QPushButton("Action")
    window.setCentralWidget(button)
    window.show()
    clicks = []
    button.clicked.connect(lambda: clicks.append(True))
    angles = []

    def probe():
        overlay = window.findChild(_Overlay)
        if overlay is not None and overlay.isVisible():
            angles.append(overlay.angle)
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    timer = QTimer(interval=40)
    timer.timeout.connect(probe)
    timer.start()
    try:
        assert run_calculation(window, lambda: (time.sleep(0.45), 42)[1]) == 42
    finally:
        timer.stop()
    assert len(set(angles)) > 1
    assert clicks == []
    assert not window.findChild(_Overlay).isVisible()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert clicks == [True]
    window.close()


def test_calculation_releases_input_on_error(qapp):
    window = QMainWindow()
    window.setCentralWidget(QPushButton())

    def fail():
        raise ValueError("calculation failed")

    with pytest.raises(ValueError, match="calculation failed"):
        run_calculation(window, fail)
    assert window.close()

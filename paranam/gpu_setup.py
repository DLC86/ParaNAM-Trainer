"""The optional NVIDIA installation choice, shown without a console or pip UI."""

import threading
from filelock import Timeout

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from . import gpu_runtime


class InstallerThread(QThread):
    progress = Signal(object, object, str)
    succeeded = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cancel_event = threading.Event()

    def run(self):
        try:
            result = gpu_runtime.install(self.progress.emit, self.cancel_event.is_set)
            self.succeeded.emit(result)
        except gpu_runtime.InstallCancelled:
            self.cancelled.emit()
        except Timeout:
            self.failed.emit("NVIDIA setup is already running in another window.")
        except Exception as error:
            self.failed.emit(str(error))


class GPUSetupDialog(QDialog):
    def __init__(self, selected=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ParaNAM Trainer — training acceleration")
        self.setMinimumWidth(570)
        self.worker = None
        self.finished_ok = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        title = QLabel("Choose training acceleration")
        title.setObjectName("section")
        layout.addWidget(title)
        text = QLabel(
            "CPU training is already included and works offline. NVIDIA support is optional and can also be added later from Training settings."
        )
        text.setWordWrap(True)
        layout.addWidget(text)
        self.choice = QCheckBox("Install NVIDIA GPU support")
        self.choice.setChecked(selected)
        layout.addWidget(self.choice)
        try:
            manifest = gpu_runtime.load_manifest()
        except (OSError, ValueError):
            manifest = None
        self.manifest = manifest
        total = sum(w.get("size", 0) for w in manifest["wheels"]) if manifest else 0
        self.status = QLabel(
            f"Downloads NVIDIA acceleration from official sources ({total / 1024**3:.1f} GiB). Internet is required only for this setup."
            if manifest
            else "This build does not include an NVIDIA download manifest. CPU training is available."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.choice.setEnabled(manifest is not None)
        if manifest and gpu_runtime.ready(manifest):
            self.choice.setChecked(True)
            self.status.setText(
                "A matching NVIDIA component is already installed. It will be checked and reused."
            )
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        layout.addWidget(self.progress)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.skip = QPushButton("Use CPU")
        self.skip.clicked.connect(self.skip_or_cancel)
        buttons.addWidget(self.skip)
        self.go = QPushButton("Continue")
        self.go.setObjectName("primary")
        self.go.clicked.connect(self.start_or_finish)
        buttons.addWidget(self.go)
        layout.addLayout(buttons)

    def running(self):
        return self.worker is not None and self.worker.isRunning()

    def start_or_finish(self):
        if self.finished_ok or not self.choice.isChecked() or not self.manifest:
            gpu_runtime.remember_choice()
            self.accept()
            return
        if self.running():
            return
        self.choice.setEnabled(False)
        self.go.setEnabled(False)
        self.skip.setText("Cancel download")
        self.worker = InstallerThread(self)
        self.worker.progress.connect(self.update_progress)
        self.worker.succeeded.connect(self.succeeded)
        self.worker.failed.connect(self.failed)
        self.worker.cancelled.connect(
            lambda: self.failed(
                "Download cancelled. Completed components are kept for the next attempt."
            )
        )
        self.worker.finished.connect(lambda: self.go.setEnabled(True))
        self.worker.start()

    def update_progress(self, completed, total, message):
        self.status.setText(message)
        self.progress.setRange(0, 1000 if total else 0)
        if total:
            self.progress.setValue(min(1000, int(completed / total * 1000)))

    def succeeded(self, result):
        self.finished_ok = True
        self.progress.setRange(0, 1000)
        self.progress.setValue(1000)
        self.status.setText(
            "NVIDIA support is ready. Training can use your GPU automatically."
            if result.get("available")
            else "NVIDIA libraries are installed. No compatible GPU/driver is currently available; training will use CPU. You can update the graphics driver and check again later."
        )
        self.go.setText("Done")
        self.skip.setText("Close")

    def failed(self, message):
        self.status.setText(message + "\nThe CPU engine remains fully available.")
        self.progress.setRange(0, 1000)
        self.go.setText("Retry")
        self.skip.setText("Use CPU")
        self.choice.setEnabled(True)

    def skip_or_cancel(self):
        if self.running():
            self.worker.cancel_event.set()
            self.status.setText(
                "Cancelling… The current network request may take a moment to finish."
            )
            return
        gpu_runtime.remember_choice()
        self.accept()

    def closeEvent(self, event):
        if self.running():
            self.skip_or_cancel()
            event.ignore()
        else:
            gpu_runtime.remember_choice()
            event.accept()

    def reject(self):
        # Escape must follow the same cancellation path as the close button.
        if self.running():
            self.skip_or_cancel()
        else:
            gpu_runtime.remember_choice()
            super().reject()


def show_setup(selected=False, parent=None):
    from .widgets import THEME, load_font

    app = QApplication.instance() or QApplication([])
    load_font()
    dialog = GPUSetupDialog(selected, parent)
    dialog.setStyleSheet(THEME)
    return dialog.exec()

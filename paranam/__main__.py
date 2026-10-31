"""GUI and frozen-worker entry point. No user-facing command line is required."""

import multiprocessing
import os
import sys
from pathlib import Path


def main():
    multiprocessing.freeze_support()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault(
        "NUMBA_CACHE_DIR", str(Path.home() / ".cache" / "paranam" / "numba")
    )
    if sys.platform == "win32":
        os.environ.setdefault("SD_ENABLE_ASIO", "1")
    args = sys.argv[1:]
    if args and args[0] == "--train-worker":
        import contextlib
        import traceback

        job = Path(args[1]).resolve()
        with (job / "worker.log").open("a", encoding="utf-8", buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    from .training import run_job

                    run_job(job)
                    return 0
                except Exception as error:
                    traceback.print_exc()
                    from nam.capture.project import atomic_write_json
                    from .training import job_progress

                    progress = job_progress(job)
                    progress.update(status="failed", message=str(error))
                    atomic_write_json(job / "progress.json", progress)
                    return 1
    if args and args[0] == "--self-test":
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w")
        if sys.stderr is None:
            sys.stderr = sys.stdout
        from .signals import verify_assets

        verify_assets()
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        import json
        import torch
        import scipy.signal
        import sounddevice
        from PySide6.QtWidgets import QApplication
        from .wizard import WizardWindow
        import paranam.wizard as wizard

        wizard.list_devices = lambda **kwargs: []
        app = QApplication.instance() or QApplication([])
        window = WizardWindow()
        window.close()
        Path(args[1]).write_text(
            json.dumps(
                {
                    "ok": True,
                    "torch": torch.__version__,
                    "cuda": torch.cuda.is_available(),
                    "mps": torch.backends.mps.is_available(),
                    "architecture": "HyperA2",
                    "pages": 4,
                }
            ),
            encoding="utf-8",
        )
        return 0
    from PySide6.QtCore import Qt, QStandardPaths
    from PySide6.QtGui import QPixmap, QColor
    from PySide6.QtWidgets import QApplication, QSplashScreen, QMessageBox

    app = QApplication(sys.argv)
    app.setOrganizationName("DLC86")
    app.setApplicationName("ParaNAM Trainer")
    pixmap = QPixmap(520, 230)
    pixmap.fill(QColor("#191b23"))
    splash = QSplashScreen(pixmap)
    splash.showMessage(
        "ParaNAM Trainer\nLoading the capture and training engines…",
        Qt.AlignmentFlag.AlignCenter,
        QColor("#ff9a61"),
    )
    splash.show()
    app.processEvents()
    # Windowed frozen apps have no stdout/stderr. Keep libraries that print happy,
    # and retain crash diagnostics in a standard per-user location.
    logs = (
        Path(
            QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.AppLocalDataLocation
            )
        )
        / "logs"
    )
    logs.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None:
        sys.stdout = (logs / "application.log").open("a", encoding="utf-8", buffering=1)
    if sys.stderr is None:
        sys.stderr = sys.stdout
    try:
        from .wizard import WizardWindow

        window = WizardWindow()
        window.show()
        splash.finish(window)
        return app.exec()
    except Exception as error:
        import traceback

        traceback.print_exc()
        splash.close()
        QMessageBox.critical(None, "ParaNAM Trainer could not start", str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

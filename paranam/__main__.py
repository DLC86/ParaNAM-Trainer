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
    if args and args[0] == "--gpu-probe":
        from .gpu_runtime import write_probe

        return write_probe(args[1])
    if args and args[0] == "--install-gpu":
        from .gpu_runtime import install

        result = Path(args[1]) if len(args) > 1 else None
        try:
            data = install()
            if result:
                import json

                result.write_text(json.dumps({**data, "ok": True}), encoding="utf-8")
            return 0
        except Exception as error:
            if result:
                import json
                import traceback

                result.write_text(
                    json.dumps(
                        dict(
                            ok=False, error=str(error), traceback=traceback.format_exc()
                        )
                    ),
                    encoding="utf-8",
                )
            return 1
    if args and args[0] == "--configure-installation":
        from .installer_gpu import configure

        cancel = (
            args[args.index("--cancel-file") + 1] if "--cancel-file" in args else None
        )
        return configure("--with-nvidia" in args, cancel)
    if args and args[0] == "--train-worker":
        import contextlib
        import traceback

        job = Path(args[1]).resolve()
        with (job / "worker.log").open("a", encoding="utf-8", buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    if "--gpu-runtime" in args:
                        from . import gpu_runtime

                        gpu_runtime.activate()
                        if not gpu_runtime.can_compute()[0]:
                            os.environ["PARANAM_FORCE_CPU"] = "1"
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
                    "default_optimizer": window.advanced_fields[
                        "optimizer"
                    ].currentData(),
                    "validation_reference": "full",
                    "pages": 4,
                }
            ),
            encoding="utf-8",
        )
        return 0
    from PySide6.QtCore import Qt, QStandardPaths
    from PySide6.QtGui import QPixmap, QColor, QIcon
    from PySide6.QtWidgets import QApplication, QSplashScreen, QMessageBox

    app = QApplication(sys.argv)
    app.setOrganizationName("DLC86")
    app.setApplicationName("ParaNAM Trainer")
    from .signals import asset_path

    app.setWindowIcon(QIcon(str(asset_path("TTSLogo.png"))))
    pixmap = QPixmap(520, 230)
    pixmap.fill(QColor("#1b1b1b"))
    splash = QSplashScreen(pixmap)
    splash.showMessage(
        "ParaNAM Trainer\nLoading the capture and training engines…",
        Qt.AlignmentFlag.AlignCenter,
        QColor("#bf0000"),
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
        from . import gpu_runtime

        if sys.platform != "darwin" and not gpu_runtime.choice_made():
            from .gpu_setup import show_setup

            splash.hide()
            show_setup()
            splash.show()

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

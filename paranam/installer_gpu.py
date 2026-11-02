"""Non-interactive installer worker; progress stays in the installer's window."""

import os
import sys
from pathlib import Path


def _output():
    if sys.stdout is None and sys.platform == "win32":
        # PyInstaller's windowed bootloader clears sys.stdout, but Inno's
        # ExecAndLogOutput still supplies a valid inherited pipe handle.
        import ctypes
        import msvcrt

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetStdHandle.restype = ctypes.c_void_p
        handle = kernel.GetStdHandle(-11)
        if handle and handle != ctypes.c_void_p(-1).value:
            descriptor = msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
            sys.stdout = os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)
    return sys.stdout


def configure(selected, cancel_path=None):
    stream = _output()

    def progress(completed, total, message):
        position = min(1000, max(0, int(completed / total * 1000))) if total else -1
        message = str(message).replace("\r", " ").replace("\n", " ")
        message = message.encode("ascii", "replace").decode("ascii")
        if stream:
            print(f"PNGPU|{position}|{message}", file=stream, flush=True)

    progress(
        0, 0, "Preparing NVIDIA support" if selected else "Finishing application setup"
    )
    from . import gpu_runtime

    if not selected:
        gpu_runtime.remember_choice()
        progress(1, 1, "CPU support is ready")
        return 0
    try:
        # A compatible component was verified on installation. Updates can reuse
        # it immediately; do not boot PyTorch/CUDA just to repeat the same check.
        if gpu_runtime.ready():
            gpu_runtime.remember_choice()
            progress(1, 1, "Installed NVIDIA support is compatible and has been reused")
            return 0
        cancelled = lambda: bool(cancel_path and Path(cancel_path).exists())
        result = gpu_runtime.install(progress, cancelled)
        progress(
            1,
            1,
            (
                "NVIDIA support is ready"
                if result.get("available")
                else "NVIDIA libraries are installed; a compatible GPU and driver are needed for GPU training"
            ),
        )
        return 0
    except gpu_runtime.InstallCancelled:
        gpu_runtime.remember_choice()
        progress(
            0,
            1,
            "GPU setup cancelled. CPU training is ready; add GPU support later in Training settings.",
        )
        return 2
    except Exception as error:
        gpu_runtime.remember_choice()
        progress(
            0,
            1,
            f"GPU setup failed: {error}. CPU training remains available; retry from Training settings.",
        )
        return 1

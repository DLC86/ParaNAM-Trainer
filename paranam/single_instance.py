"""Per-user, crash-safe GUI singleton without importing Qt or training libraries."""

import errno
import os
from pathlib import Path
import sys


class GuiInstanceGuard:
    """Hold the OS lock until the GUI process exits. CLI workers do not use it."""

    def __init__(self):
        self._handle = None
        self._fd = None

    def acquire(self):
        if self._handle is not None or self._fd is not None:
            return True
        if sys.platform == "win32":
            import ctypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateMutexW.argtypes = (
                ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p
            )
            kernel.CreateMutexW.restype = ctypes.c_void_p
            kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
            kernel.CloseHandle.restype = ctypes.c_bool
            handle = kernel.CreateMutexW(
                None, False, "Local\\DLC86_ParaNAM_Trainer_CORE_67A2"
            )
            if not handle:
                raise OSError(ctypes.get_last_error(), "Cannot create GUI mutex")
            if ctypes.get_last_error() == 183:
                kernel.CloseHandle(handle)
                return False
            self._handle = (kernel, handle)
            return True

        # flock is atomic and cleaned up by the kernel, including on crashes.
        # Never unlink the file: that would allow a new lock on a different inode.
        import fcntl

        directory = Path.home() / ".cache" / "paranam"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "gui-instance.lock"
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(fd)
            if error.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
        self._fd = fd
        return True

    def close(self):
        if self._handle is not None:
            kernel, handle = self._handle
            self._handle = None
            kernel.CloseHandle(handle)
        if self._fd is not None:
            import fcntl

            fd = self._fd
            self._fd = None
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def __del__(self):
        self.close()

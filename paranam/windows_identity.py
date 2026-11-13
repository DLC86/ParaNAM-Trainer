"""Windows taskbar identity shared by the Qt Core and launcher shortcuts."""

import sys


WINDOWS_APP_USER_MODEL_ID = "DLC86.ParaNAMTrainer"


def set_windows_app_user_model_id():
    """Group the Core's Qt windows with the branded launcher shortcut.

    The shortcut runs ParaNAM Trainer.exe, but PyInstaller's child process
    (ParaNAM Trainer Core.exe) owns the main Qt window. Windows uses this
    explicit identity to associate the window with the correct taskbar icon.
    Call before creating any GUI windows; no-op on other operating systems.
    """
    if sys.platform != "win32":
        return

    import ctypes

    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    set_id = shell32.SetCurrentProcessExplicitAppUserModelID
    set_id.argtypes = [ctypes.c_wchar_p]
    set_id.restype = ctypes.c_long  # HRESULT (32-bit)
    result = set_id(WINDOWS_APP_USER_MODEL_ID)
    if result != 0:
        raise OSError(
            "Unable to configure Windows taskbar identity "
            f"(HRESULT 0x{result & 0xFFFFFFFF:08X})"
        )

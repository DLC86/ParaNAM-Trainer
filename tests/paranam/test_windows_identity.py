"""Regression coverage for taskbar identity and compact branding."""

from pathlib import Path
import ctypes

import paranam.windows_identity as windows_identity


ROOT = Path(__file__).resolve().parents[2]


def test_windows_identity_matches_installer_shortcuts():
    installer = (ROOT / "packaging" / "windows.iss").read_text(encoding="utf-8")
    shortcut_id = f'AppUserModelID: "{windows_identity.WINDOWS_APP_USER_MODEL_ID}"'
    assert installer.count(shortcut_id) == 2


def test_windows_identity_precedes_first_qt_window():
    entry = (ROOT / "paranam" / "__main__.py").read_text(encoding="utf-8")
    assert entry.index("set_windows_app_user_model_id()") < entry.index(
        "app = QApplication(sys.argv)"
    )
    assert "window.setWindowIcon(window_icon)" in entry
    spec = (ROOT / "packaging" / "ParaNAM.spec").read_text(encoding="utf-8")
    assert '"paranam/assets"' in spec and '"icon.ico"' in spec


def test_windows_identity_only_calls_win32_shell(monkeypatch):
    calls = []

    class Setter:
        def __call__(self, value):
            calls.append(value)
            return 0

    class Shell32:
        SetCurrentProcessExplicitAppUserModelID = Setter()

    monkeypatch.setattr(
        ctypes, "WinDLL", lambda name, use_last_error: Shell32(),
        raising=False,
    )
    monkeypatch.setattr(windows_identity.sys, "platform", "linux")
    windows_identity.set_windows_app_user_model_id()
    assert calls == []

    monkeypatch.setattr(windows_identity.sys, "platform", "win32")
    windows_identity.set_windows_app_user_model_id()
    assert calls == [windows_identity.WINDOWS_APP_USER_MODEL_ID]


def test_sidebar_uses_compact_pnam_brand():
    wizard = (ROOT / "paranam" / "wizard.py").read_text(encoding="utf-8")
    assert 'self.brand_title = label("PNAM\\nTrainer", "brand")' in wizard
    assert "sidebar.setFixedWidth(228)" in wizard

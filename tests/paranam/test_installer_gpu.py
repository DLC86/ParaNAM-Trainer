import pytest

from paranam import gpu_runtime
from paranam.installer_gpu import configure


def test_installer_reuses_compatible_support_without_loading_engine(
    monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(gpu_runtime, "ready", lambda: True)
    monkeypatch.setattr(gpu_runtime, "remember_choice", lambda: calls.append("choice"))
    monkeypatch.setattr(
        gpu_runtime, "install", lambda *a: pytest.fail("Must reuse, not reinstall")
    )
    assert configure(True) == 0
    assert calls == ["choice"]
    assert (
        "PNGPU|1000|Installed NVIDIA support is compatible" in capsys.readouterr().out
    )


def test_installer_reports_download_progress_without_dialogs(monkeypatch, capsys):
    monkeypatch.setattr(gpu_runtime, "ready", lambda: False)

    def install(progress, cancelled):
        assert not cancelled()
        progress(25, 100, "Downloading torch")
        return {"available": True}

    monkeypatch.setattr(gpu_runtime, "install", install)
    assert configure(True) == 0
    text = capsys.readouterr().out
    assert "PNGPU|250|Downloading torch" in text
    assert "PNGPU|1000|NVIDIA support is ready" in text


@pytest.mark.parametrize("cancelled", [False, True])
def test_installer_failure_keeps_cpu_choice_and_needs_no_confirmation(
    monkeypatch, capsys, cancelled
):
    choices = []
    monkeypatch.setattr(gpu_runtime, "ready", lambda: False)
    monkeypatch.setattr(gpu_runtime, "remember_choice", lambda: choices.append(True))

    def install(*args):
        raise gpu_runtime.InstallCancelled() if cancelled else RuntimeError("Offline")

    monkeypatch.setattr(gpu_runtime, "install", install)
    assert configure(True) == (2 if cancelled else 1)
    assert choices == [True]
    assert "CPU training" in capsys.readouterr().out


def test_installer_cpu_only_remembers_choice(monkeypatch, capsys):
    choices = []
    monkeypatch.setattr(gpu_runtime, "remember_choice", lambda: choices.append(True))
    assert configure(False) == 0
    assert choices == [True]
    assert "PNGPU|1000|CPU support is ready" in capsys.readouterr().out

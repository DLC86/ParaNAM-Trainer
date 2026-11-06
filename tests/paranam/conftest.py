import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PARANAM_DISABLE_UPDATE_CHECK", "1")

import pytest

from nam.capture.params import KnobSpec
from paranam.workspace import Workspace


@pytest.fixture
def workspace(tmp_path):
    ws = Workspace.create(tmp_path / "My rig", "My rig")
    ws.draft_controls = [KnobSpec("Gain", 0, 10).to_dict()]
    ws.preferences.update(
        n_train=2,
        n_validation=1,
        corners=False,
        epochs=2,
        batch_size=1,
        device="cpu",
        advanced_parametric={},
    )
    ws.generate_plan()
    return ws


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings, QObject, Signal
    from paranam import wizard
    from paranam import settings_audio

    class FakeSettingsAudio(QObject):
        metered = Signal(float, float, float, float)
        failed = Signal(str)
        finished = Signal()

        def __init__(self, device, audio, rate, parent):
            super().__init__(parent)
            self.device, self.audio, self.rate = device, audio, rate
            self.noise = (False, 0.0)
            self.stopped = False

        def start(self):
            pass

        def set_noise(self, enabled, level):
            self.noise = (enabled, level)

        def stop(self):
            self.stopped = True
            self.noise = (False, 0.0)

        def wait(self, timeout):
            return True

    # GUI tests use synthetic interface indices; never open the user's hardware.
    monkeypatch.setattr(settings_audio, "SettingsAudioWorker", FakeSettingsAudio)

    monkeypatch.setattr(wizard, "list_devices", lambda **kwargs: [])
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(wizard, "QSettings", lambda *args: settings)
    instance = wizard.WizardWindow()
    yield instance
    instance.close()


@pytest.fixture(autouse=True)
def clear_recovery_events():
    from paranam.recovery import reset_session

    reset_session()
    yield
    reset_session()

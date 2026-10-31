import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pytest

from nam.capture.params import KnobSpec
from paranam.workspace import Workspace


@pytest.fixture
def workspace(tmp_path):
    ws = Workspace.create(tmp_path / "My rig", "My rig")
    ws.draft_controls = [KnobSpec("Gain", 0, 10).to_dict()]
    ws.preferences.update(
        n_train=2, n_validation=1, corners=False, epochs=2, batch_size=1, device="cpu"
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
    from PySide6.QtCore import QSettings
    from paranam import wizard

    monkeypatch.setattr(wizard, "list_devices", lambda **kwargs: [])
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(wizard, "QSettings", lambda *args: settings)
    instance = wizard.WizardWindow()
    yield instance
    instance.close()

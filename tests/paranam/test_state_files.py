from pathlib import Path

import pytest

from nam.capture import project
from paranam.workspace import read_json


def test_state_read_retries_transient_windows_sharing_violation(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    project.atomic_write_json(path, {"epoch": 2})
    original = Path.read_text
    attempted = []

    def read(current, *args, **kwargs):
        if current == path and not attempted:
            attempted.append(True)
            raise PermissionError("temporary sharing violation")
        return original(current, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert read_json(path) == {"epoch": 2}


def test_atomic_state_write_retries_and_keeps_the_new_data(tmp_path, monkeypatch):
    path = tmp_path / "command.json"
    project.atomic_write_json(path, {})
    original = project._os.replace
    attempted = []

    def replace(source, target):
        if not attempted:
            attempted.append(True)
            raise PermissionError("file is briefly held by a reader")
        return original(source, target)

    monkeypatch.setattr(project._os, "replace", replace)
    project.atomic_write_json(path, {"action": "pause"})
    assert read_json(path) == {"action": "pause"}
    assert not list(tmp_path.glob("*.tmp"))


def test_permanent_denial_keeps_old_state_and_surfaces_error(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    project.atomic_write_json(path, {"epoch": 1})

    def denied(*args):
        raise PermissionError("permanent denial")

    monkeypatch.setattr(project._os, "replace", denied)
    monkeypatch.setattr(project._time, "sleep", lambda *a: None)
    with pytest.raises(PermissionError, match="permanent"):
        project.atomic_write_json(path, {"epoch": 2})
    assert read_json(path) == {"epoch": 1}
    assert not list(tmp_path.glob("*.tmp"))

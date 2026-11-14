import json

import pytest

from paranam.project_files import change_project
from paranam.training_queue import TrainingQueue


def test_rename_and_duplicate_keep_project_paths_independent(tmp_path):
    source = tmp_path / "Rig"
    source.mkdir()
    (source / "capture_project.json").write_text('{"name":"Rig"}')
    (source / "paranam.json").write_text(json.dumps({"custom_input": str(source / "input.wav")}))
    (source / "input.wav").write_bytes(b"recording")
    (tmp_path / "Rig (2)").mkdir()

    copy = change_project(source, "Rig", duplicate=True)
    assert copy.name == "Rig (3)"
    assert json.loads((copy / "capture_project.json").read_text())["name"] == copy.name
    assert json.loads((copy / "paranam.json").read_text())["custom_input"] == str(copy / "input.wav")
    assert (copy / "input.wav").read_bytes() == (source / "input.wav").read_bytes()

    queue = TrainingQueue(tmp_path / "queue")
    with pytest.raises(ValueError, match="already exists"):
        change_project(source, "rig (3)", queue_store=queue)
    renamed = change_project(source, "New Rig", queue_store=queue)
    assert not source.exists()
    assert json.loads((renamed / "capture_project.json").read_text())["name"] == "New Rig"
    assert json.loads((renamed / "paranam.json").read_text())["custom_input"] == str(renamed / "input.wav")

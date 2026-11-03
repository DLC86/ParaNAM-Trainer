import hashlib

import pytest

from nam.capture.params import KnobSpec, SwitchSpec
from paranam.signals import asset_path, verify_assets
from paranam.workspace import Workspace


def test_exact_offline_assets_are_in_project_and_manifest(workspace):
    assets = verify_assets()
    assert [a["seconds"] for a in assets] == [190, 7, 300]
    assert all(a["sample_rate"] == 48000 and a["channels"] == 1 for a in assets)
    assert workspace.project.train_input == "input_standard.wav"
    assert workspace.validation_matches()
    assert not workspace.plan_changed()
    assert (workspace.directory / "input_standard.wav").read_bytes() == asset_path(
        "standard.wav"
    ).read_bytes()


def test_plan_changes_archive_recordings_without_deleting_them(workspace):
    entry = workspace.project.entries[0]
    output = workspace.directory / entry.y_path
    output.parent.mkdir(exist_ok=True)
    output.write_bytes(b"original recording")
    entry.status = "captured"
    workspace.save()
    workspace.draft_controls.append(SwitchSpec("Mode", ("Clean", "Lead")).to_dict())
    workspace.preferences.update(n_train=4, n_validation=2)
    assert workspace.plan_changed()
    workspace.generate_plan()
    archived = list((workspace.directory / "history").glob("*/" + entry.y_path))
    assert len(archived) == 1 and archived[0].read_bytes() == b"original recording"
    assert all(e.status == "pending" for e in workspace.project.entries)
    assert {e.params["Mode"] for e in workspace.project.entries} == {"Clean", "Lead"}
    assert not workspace.plan_changed()


def test_fixed_validation_is_never_replaced_by_custom_input(workspace):
    original = (workspace.directory / workspace.project.validation_input).read_bytes()
    source = workspace.directory / "custom.wav"
    source.write_bytes(b"custom input placeholder")
    workspace.preferences.update(input_mode="custom", custom_input="custom.wav")
    assert workspace.plan_changed()
    workspace.generate_plan()
    assert workspace.project.train_input == "custom.wav"
    assert (
        workspace.directory / workspace.project.validation_input
    ).read_bytes() == original
    assert workspace.validation_matches()


def test_controls_only_switches_validate_with_independent_fixed_audio(workspace):
    workspace.draft_controls = [SwitchSpec("Mode", ("A", "B", "C")).to_dict()]
    workspace.preferences.update(n_train=3, n_validation=3)
    workspace.generate_plan()
    assert len(workspace.project.entries_for_split("train")) == 3
    assert len(workspace.project.entries_for_split("validation")) == 3
    assert workspace.project.train_input != workspace.project.validation_input


def test_import_keeps_existing_project_plan_counts(workspace):
    (workspace.directory / "paranam.json").unlink()
    reopened = Workspace(workspace.directory)
    assert reopened.preferences["n_train"] == 2
    assert reopened.preferences["n_validation"] == 1
    assert reopened.preferences["corners"] is False


def test_new_project_never_overwrites_existing_files(tmp_path):
    (tmp_path / "important.wav").write_bytes(b"keep")
    with pytest.raises(ValueError, match="empty folder"):
        Workspace.create(tmp_path, "Other")
    assert (tmp_path / "important.wav").read_bytes() == b"keep"


def test_changed_validation_must_be_recaptured(workspace):
    (workspace.directory / workspace.project.validation_input).write_bytes(b"changed")
    assert workspace.plan_changed()
    assert not workspace.training_ready()

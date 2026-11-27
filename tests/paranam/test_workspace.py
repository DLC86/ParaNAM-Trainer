import hashlib

import pytest

from nam.capture.params import KnobSpec, SwitchSpec
from paranam.signals import asset_path, verify_assets
from paranam.workspace import Workspace


def test_exact_offline_assets_are_in_project_and_manifest(workspace):
    assets = verify_assets()
    assert [a["seconds"] for a in assets] == [190, 7, 300, 38, 57]
    assert all(a["sample_rate"] == 48000 and a["channels"] == 1 for a in assets)
    assert workspace.project.train_input == "input_standard.wav"
    assert workspace.validation_matches()
    assert not workspace.plan_changed()
    assert (workspace.directory / "input_standard.wav").read_bytes() == asset_path(
        "standard.wav"
    ).read_bytes()


def test_plan_changes_skip_incompatible_recordings_without_moving_them(workspace):
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
    assert output.read_bytes() == b"original recording"
    assert entry.y_path in workspace.project.unused_recordings
    assert all(e.y_path != entry.y_path for e in workspace.project.entries)
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
    (workspace.directory / "paranam.json.bak").unlink(missing_ok=True)
    reopened = Workspace(workspace.directory)
    assert reopened.preferences["n_train"] == 2
    assert reopened.preferences["n_validation"] == 1
    assert reopened.preferences["corners"] is False


def test_plan_default_checkbox_starts_checked_and_only_marks_legacy_plan_on_edit(
    window, workspace
):
    # Simulate a project saved before the new Plan checkbox existed.
    workspace.project.include_default_as_additional_capture = None
    workspace.save()
    window._attach(workspace)
    checkbox = window.include_default_as_additional_capture
    assert checkbox.text() == "Include default as additional capture"
    assert checkbox.isChecked()
    before = [(e.split, dict(e.params), e.y_path) for e in workspace.project.entries]
    assert not workspace.plan_changed()
    checkbox.setChecked(False)
    assert not checkbox.isChecked()
    assert workspace.preferences["include_default_as_additional_capture"] is False
    assert workspace.plan_changed()
    # Plan is not replaced until the user actually applies the changes.
    assert [(e.split, dict(e.params), e.y_path) for e in workspace.project.entries] == before
    assert workspace.project.include_default_as_additional_capture is None
    window.close()


def test_legacy_plan_is_not_regenerated_until_plan_settings_change(workspace):
    # Newly created fixture plans use the new default-on policy. Recreate the
    # legacy field state explicitly before verifying backwards compatibility.
    workspace.project.include_default_as_additional_capture = None
    workspace.save()
    assert workspace.project.include_default_as_additional_capture is None
    original = [(e.split, dict(e.params), e.y_path) for e in workspace.project.entries]
    assert workspace.preferences["include_default_as_additional_capture"] is True
    assert not workspace.plan_changed()

    # Merely opening an existing project or storing the new checkbox default
    # must not alter the capture list.
    workspace.preferences["include_default_as_additional_capture"] = True
    workspace.save()
    reopened = Workspace(workspace.directory)
    assert not reopened.plan_changed()
    assert [(e.split, dict(e.params), e.y_path) for e in reopened.project.entries] == original

    # Once another Plan setting changes, apply the new explicit policy.
    reopened.preferences["seed"] += 1
    assert reopened.plan_changed()
    reopened.generate_plan()
    assert reopened.project.include_default_as_additional_capture is True
    assert len(reopened.project.entries_for_split("train")) == reopened.project.n_train_lhs + 1
    assert not reopened.plan_changed()


def test_disabling_default_checkbox_does_not_force_a_capture(workspace):
    workspace.preferences["include_default_as_additional_capture"] = False
    workspace.preferences["_include_default_capture_touched"] = True
    assert workspace.plan_changed()
    workspace.generate_plan()
    assert workspace.project.include_default_as_additional_capture is False
    assert len(workspace.project.entries_for_split("train")) == workspace.project.n_train_lhs
    assert not workspace.plan_changed()


def test_new_project_never_overwrites_existing_files(tmp_path):
    (tmp_path / "important.wav").write_bytes(b"keep")
    with pytest.raises(ValueError, match="empty folder"):
        Workspace.create(tmp_path, "Other")
    assert (tmp_path / "important.wav").read_bytes() == b"keep"


def test_changed_validation_must_be_recaptured(workspace):
    (workspace.directory / workspace.project.validation_input).write_bytes(b"changed")
    assert workspace.plan_changed()
    assert not workspace.training_ready()


def test_discovered_mute_and_avoid_points_survive_replanning(workspace):
    from nam.capture.project import classify_capture

    initial = workspace.project.entries_for_split("train")[0]
    first, _ = classify_capture(workspace.project, initial, "mute")
    other = first.entries_for_split("validation")[0]
    workspace.project, _ = classify_capture(first, other, "avoid")
    workspace.save()
    muted = dict(initial.params)
    avoided = dict(other.params)
    workspace.preferences["seed"] += 7
    assert workspace.plan_changed()
    workspace.generate_plan()
    assert avoided in workspace.project.avoided_settings
    mute_rows = [e for e in workspace.project.entries if e.params == muted]
    # An explicit old Mute is retained only when this plan selected that setting.
    assert (muted in workspace.project.muted_settings) == bool(mute_rows)
    assert all(e.synthetic_mute and e.status == "captured" for e in mute_rows)
    assert all((workspace.directory / e.y_path).is_file() for e in mute_rows)
    # Replanning is free to replace an Avoid row, but preserves its classified
    # setting and must keep it out of the training dataset.
    training_rows = workspace.training_capture_entries()
    # Avoided rows are excluded, while Mute rows remain as synthetic silent
    # targets in the dataset.
    assert all(e.params != avoided for e in training_rows)
    assert any(e.params == muted and e.synthetic_mute for e in training_rows) == bool(mute_rows)
    assert not workspace.plan_changed()
    reopened = Workspace(workspace.directory)
    assert (muted in reopened.project.muted_settings) == bool(mute_rows)
    assert avoided in reopened.project.avoided_settings
    assert not reopened.plan_changed()


def test_new_workspace_defaults_to_sobol_but_preserves_existing_lhs_choice(tmp_path):
    project = Workspace.create(tmp_path / "Sobol", "Sobol")
    assert project.project.lhs_method == "sobol"
    assert project.preferences["lhs_method"] == "sobol"
    assert (project.preferences["n_train"], project.preferences["n_validation"]) == (32, 8)

    # An old project explicitly selecting Standard LHS must not be migrated.
    project.project.lhs_method = "standard"
    project.preferences["lhs_method"] = "standard"
    project.save()
    reopened = Workspace(project.directory)
    assert reopened.preferences["lhs_method"] == "standard"
    assert reopened.project.lhs_method == "standard"


def test_parametric_training_rejects_mixed_routes_but_allows_buffer_changes(workspace):
    from nam.capture.project import AudioSettingsModel

    entries = workspace.training_capture_entries()
    settings = AudioSettingsModel(host_api="Core Audio", input_device="Interface",
                                  output_device="Interface", blocksize=256)
    for entry in entries:
        target = workspace.directory / entry.y_path
        target.parent.mkdir(exist_ok=True)
        target.touch()
        entry.status = "captured"
        entry.stream_config = settings.stream_fingerprint()

    validation = next(entry for entry in entries if entry.split == "validation")
    validation.stream_config = settings.model_copy(update={"blocksize": 512}).stream_fingerprint()
    assert workspace.training_ready()

    validation.stream_config = settings.model_copy(update={"host_api": "ASIO"}).stream_fingerprint()
    assert not workspace.training_ready()
    assert "different audio devices, drivers, or channels" in workspace.capture_setup_problem()
    assert validation.y_path in workspace.capture_setup_problem()

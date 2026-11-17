"""Current synthetic-only Mute workflow and legacy preference migration."""
import shutil

import soundfile as sf

from nam.capture.params import KnobSpec
from paranam import training
from paranam.workspace import Workspace, read_json


def prepare_mute_dataset(workspace):
    workspace.draft_controls = [
        KnobSpec("Gain", 0, 10, 1, 5, mute_at_min=True).to_dict()
    ]
    # One 0..10 knob has eleven distinct physical settings.  Keep this
    # synthetic-Mute fixture below that capacity while retaining both splits.
    workspace.preferences.update(n_train=6, n_validation=3, corners=True)
    workspace.generate_plan()
    for row in workspace.project.entries:
        if row.synthetic_mute:
            assert row.status == "captured"
            continue
        output = workspace.directory / row.y_path
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            workspace.directory / workspace.project.input_for_split(row.split),
            output,
        )
        row.status = "captured"
        row.delay = 0
    workspace.save()


def test_legacy_mute_method_preferences_do_not_restore_old_modes(workspace):
    prepare_mute_dataset(workspace)
    for old_mode in ("anchor", "virtual", "weight"):
        workspace.preferences["advanced_parametric"] = dict(
            mute_training_mode=old_mode,
            mute_anchor_interval=9,
            mute_anchor_batch_size=7,
            mute_anchor_ny=1024,
        )
        options = workspace.training_options("parametric")
        assert "mute_training_mode" not in options
        assert not any(k.startswith("mute_anchor_") for k in options)
        assert options["synthetic_mute_reference_rms"] is False
        spec = read_json(training.make_job(workspace) / "spec.json")
        assert spec["model"]["loss"]["synthetic_mute"] is True
        assert "mute_anchor" not in spec["model"]["loss"]
        assert spec["learning"]["synthetic_mute_train_indices"]


def test_no_mute_method_selector_or_anchor_settings_in_ui(window, workspace):
    window._attach(workspace)
    assert "mute_training_mode" not in window.advanced_fields
    assert not any(k.startswith("mute_anchor_") for k in window.advanced_fields)
    assert window.advanced_fields["synthetic_mute_reference_rms"].isChecked() is False


def test_mute_dataset_is_generated_immediately_and_shared(workspace):
    prepare_mute_dataset(workspace)
    muted = [row for row in workspace.project.entries if row.synthetic_mute]
    assert muted
    for split in ("train", "validation"):
        rows = [row for row in muted if row.split == split]
        if not rows:
            continue
        assert len({row.y_path for row in rows}) == 1
        source = workspace.directory / workspace.project.input_for_split(split)
        target = workspace.directory / rows[0].y_path
        assert sf.info(source).frames == sf.info(target).frames
        assert sf.info(source).samplerate == sf.info(target).samplerate

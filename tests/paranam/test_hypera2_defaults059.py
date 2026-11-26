import json
from pathlib import Path

from nam.capture.params import KnobSpec, SwitchSpec
from paranam import training
from paranam.recipes import training_preset
from paranam.workspace import Workspace, read_json
from tests.paranam.test_training import ready


def requested_settings():
    saved = json.loads(
        (Path(__file__).parent / "fixtures" / "hypera2-20261106-0252.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        **saved,
        "hypernetwork_recipe": "HyperA2",
        "checkpoint_method": "validation",
        "checkpoint_alpha": 0.75,
        "rate_preset": "a2_updates",
        "hypernetwork_rate_preset": "a2_updates",
        "hypernetwork_learning_rate": 0.001,
        "hypernetwork_lr_decay": 0.998496613,
        "branch_loss_reduction": "sum",
        "snr_weighted_loss_normalization": False,
        "gradient_clipping": True,
        "rms_loss_normalization": True,
        "rms_loss_normalization_exponent": 2.0,
        "waveform_loss": "mse",
        "segment_esr_min_rms_dbfs": -58.0,
        "synthetic_mute_reference_rms": False,
    }


def test_hypera2_preset_preserves_historical_fields_except_new_normalization_defaults():
    assert training_preset("HyperA2") == requested_settings()


def test_new_projects_use_requested_preset_and_old_projects_keep_settings(tmp_path):
    workspace = Workspace.create(tmp_path / "New rig", "New rig")
    expected = {**requested_settings(), "waveform_loss": "segment_esr"}
    assert Workspace(workspace.directory).training_options() == expected
    assert workspace.preferences["epochs"] == 100
    assert workspace.training_batch_size() == 16
    legacy = training_preset("HyperWaveNet")
    workspace.preferences["advanced_parametric"] = {
        key: value for key, value in legacy.items() if key != "waveform_loss"
    }
    workspace.save()
    assert Workspace(workspace.directory).training_options() == legacy


def test_0252_architecture_optimizer_safety_flags_and_both_decay_clocks(workspace):
    workspace.preferences["advanced_parametric"] = training_preset("HyperA2")
    workspace.preferences["batch_size"] = 16
    workspace.draft_controls = [
        control.to_dict()
        for control in (
            KnobSpec("Gain", 0, 10),
            KnobSpec("Tone", 0, 10),
            SwitchSpec("Bright", ("Off", "On")),
        )
    ]
    workspace.preferences.update(n_train=4, n_validation=2)
    workspace.generate_plan()
    ready(workspace)
    spec = read_json(training.make_job(workspace) / "spec.json")
    hypernet = spec["model"]["net"]["config"]["hypernet"]
    assert hypernet == dict(
        hidden_sizes=[16],
        activation="LeakyReLU",
        selector={"exclude_suffixes": ["_conv.weight"]},
        mode="low_rank",
        rank=2,
    )
    assert spec["model"]["optimizer"] == {
        "class": "Adam",
        "lr": 0.004,
        "weight_decay": 3.17e-7,
    }
    assert spec["model"]["hypernet_lr_scale"] == 0.25
    assert spec["model"]["loss"]["training_recipe"] == "a2"
    assert spec["model"]["loss"]["mrstft_weight"] == 0.0005
    steps = spec["preferences"]["rate_schedule"]["audio"]["steps_per_epoch"]
    assert spec["model"]["lr_scheduler"] == {
        "class": "IndependentGroupLR",
        "kwargs": {
            "audio": {"interval": "step", "step_size": 62, "gamma": 0.994},
            "hypernetwork": {"interval": "step", "step_size": 62, "gamma": 0.998496613},
        },
    }
    assert spec["learning"]["trainer"]["gradient_clip_val"] == 1.0
    assert spec["learning"]["trainer"]["benchmark"] is False
    assert spec["learning"]["trainer"]["deterministic"] is False
    assert spec["model"]["check_nonfinite_gradients"] is False

import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from nam.models import factory
from nam.models.parametric import HyperA2, export_parametric
from nam.models.wavenet._wavenet import WaveNet as InnerWaveNet
from nam.train.parametric import _ParametricLightningModule, _ParametricLossConfig


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return {
        "params": [
            {"name": "Gain", "type": "continuous", "min": 0, "max": 10, "default": 5}
        ],
        "sample_rate": 48000,
        "hypernet": {"hidden_sizes": [4], "mode": "low_rank", "rank": 2},
    }


def make_model():
    return factory.init("HyperA2", args=(config(),))


def test_exact_stock_topologies_and_receptive_field():
    model = make_model()
    with open("nam/train/_resources/config_model_packed.json") as fp:
        stock = json.load(fp)["net"]["config"]["submodels"]
    assert model.sizes == ("lite", "full")
    assert model.receptive_field == 6347
    assert model.size == "full"
    for size, entry in zip(model.sizes, stock):
        actual = model.branch(size)._template
        expected = InnerWaveNet.init_from_config(entry["config"])
        assert actual.export_config() == expected.export_config()
    assert [len(model.branch(s)._template.export_weights()) for s in model.sizes] == [
        1871,
        12146,
    ]


def test_selection_runs_only_selected_branch_and_restores_on_error(mocker):
    model = make_model()
    x = torch.randn(2, model.receptive_field + 7)
    p = torch.tensor([4.0])
    unused = mocker.spy(model.branch("full"), "_run_conditioned")
    with model.using_size("lite"):
        actual = model(x, p, pad_start=False)
    expected = model.branch("lite")(x, p, pad_start=False)
    torch.testing.assert_close(actual, expected)
    unused.assert_not_called()
    assert model.size == "full"
    with pytest.raises(RuntimeError):
        with model.using_size("lite"):
            raise RuntimeError("test")
    assert model.size == "full"
    with pytest.raises(ValueError, match="size"):
        model.set_size("invalid")


def test_training_scores_both_sizes_and_backpropagates():
    model = make_model()
    module = _ParametricLightningModule(model, loss_config=_ParametricLossConfig())
    module.train()
    module.on_train_epoch_start()
    x = torch.randn(1, model.receptive_field + 31)
    p = torch.tensor([[3.0]])
    target = torch.randn(1, 32)
    # The parametric batch order is audio, controls, target.
    loss = module.training_step((x, p, target), 0)
    assert torch.isfinite(loss)
    loss.backward()
    for size in model.sizes:
        branch = model.branch(size)
        assert any(
            v.grad is not None and v.grad.abs().sum() > 0
            for v in branch._template.parameters()
        )
        assert any(
            v.grad is not None and v.grad.abs().sum() > 0
            for v in branch._hypernet.parameters()
        )
    assert set(module._train_size_metrics) == {"lite", "full"}
    module.eval()
    module.on_validation_epoch_start()
    with torch.no_grad():
        module.validation_step((x, p, target), 0)
    assert set(module._val_size_metrics) == {"lite", "full"}
    assert model.size == "full"


def test_parametric_export_restore_and_scaling(tmp_path):
    model = make_model().eval()
    with torch.no_grad():
        for size in model.sizes:
            model.branch(size)._hypernet._final.weight.normal_(0, 0.01)
    metadata = {
        "input_level_dbu": 18.125,
        "output_level_dbu": 12.625,
        "custom": {"labels": ["Clean", "Lead"], "nested": {"preserve": True}},
    }
    export_parametric(model, tmp_path, output_scale=2.0, other_metadata=metadata)
    exported = json.loads((tmp_path / "model.pnam").read_text())
    assert not (tmp_path / "model.nam").exists()
    assert all(exported["metadata"][key] == value for key, value in metadata.items())
    assert exported["architecture"] == "ParametricSlimmableContainer"
    restored = HyperA2.init_from_config({**exported["config"], "sample_rate": 48000})
    assert restored.import_weights(exported["weights"]) == len(exported["weights"])
    x = torch.randn(128)
    for size in model.sizes:
        model.set_size(size)
        restored.set_size(size)
        with torch.no_grad():
            torch.testing.assert_close(
                restored(x, torch.tensor([7.0])), model(x, torch.tensor([7.0])) / 2
            )
    # Export must not alter the live training model.
    assert model.branch("full")._template._head_scale == 0.01
    assert exported["config"]["submodels"][0]["max_value"] == 0.5
    assert exported["config"]["submodels"][1]["max_value"] == 1.0
    with pytest.raises(ValueError, match="weights"):
        restored.import_weights(exported["weights"][:-1])
    bad = deepcopy(exported["config"])
    bad["submodels"][0]["weight_offset"] = 1
    with pytest.raises(ValueError, match="layout"):
        HyperA2.init_from_config(bad)


def test_bake_both_sizes_matches_conditioned_model_and_stock_export(tmp_path, mocker):
    model = make_model().eval()
    p = torch.tensor([6.0])
    with torch.no_grad():
        for size in model.sizes:
            model.branch(size)._hypernet._final.weight.normal_(0, 0.01)
    baked = model.bake(p, output_scale=2.0)
    x = torch.randn(1, model.receptive_field + 31)
    for index, size in enumerate(model.sizes):
        model.set_size(size)
        with torch.no_grad():
            torch.testing.assert_close(
                baked.extract_submodel(index)(x, pad_start=False),
                model(x, p, pad_start=False),
                rtol=1e-5,
                atol=1e-6,
            )
    # Skip costly standard loudness probes; the waveform comparison is above.
    mocker.patch("nam.models.base._Base._metadata_loudness", return_value=-20.0)
    mocker.patch("nam.models.base._Base._metadata_gain", return_value=0.5)
    baked.export(tmp_path)
    exported = json.loads((tmp_path / "model.nam").read_text())
    assert exported["architecture"] == "SlimmableContainer"
    assert not (tmp_path / "model.pnam").exists()
    assert exported["weights"] == []
    for entry in exported["config"]["submodels"]:
        child = entry["model"]
        assert child["architecture"] == "WaveNet"
        assert child["config"]["head_scale"] == pytest.approx(0.005)
        assert child["weights"][-1] == pytest.approx(0.005)


def test_training_main_writes_dual_size_checkpoints_and_final_exports(tmp_path, mocker):
    from nam.data import np_to_wav
    from nam.train.parametric import main
    from tests.parametric.test_train import _learning_config

    mocker.patch("nam.models.base._Base._metadata_loudness", return_value=-20.0)
    mocker.patch("nam.models.base._Base._metadata_gain", return_value=0.5)
    x = (0.1 * np.sin(np.arange(6500) * 0.03)).astype(np.float32)
    np_to_wav(x, tmp_path / "input.wav", rate=48000)
    np_to_wav(np.tanh(2 * x), tmp_path / "output.wav", rate=48000)
    common = {
        "x_path": str(tmp_path / "input.wav"),
        "y_path": str(tmp_path / "output.wav"),
        "delay": 0,
        "ny": 32,
        "require_input_pre_silence": None,
    }
    data = {
        "type": "parametric",
        "common": common,
        "train": [{"params": {"Gain": 3.0}}],
        "validation": [{"params": {"Gain": 7.0}}],
        "joint": [
            {
                "name": "nam.data.normalize_joint_dataset_output",
                "kwargs": {"level_rms_dbfs": -18.0},
            }
        ],
    }
    outdir = tmp_path / "run"
    outdir.mkdir()
    main(
        data_config=data,
        model_config={
            "net": {"name": "HyperA2", "config": config()},
            "loss": {"val_loss": "esr"},
            "optimizer": {"lr": 0.001},
            "metadata": {"input_level_dbu": 18.0, "output_level_dbu": 12.5},
            "lr_scheduler": None,
        },
        learning_config=_learning_config(),
        outdir=outdir,
        no_show=True,
        make_plots=False,
    )
    final = json.loads((outdir / "model_parametric.pnam").read_text())
    assert final["architecture"] == "ParametricSlimmableContainer"
    assert final["metadata"]["input_level_dbu"] == 18.0
    assert final["metadata"]["output_level_dbu"] == 12.5
    assert len(final["config"]["submodels"]) == 2
    baked = json.loads((outdir / "model.nam").read_text())
    assert baked["architecture"] == "SlimmableContainer"
    assert baked["metadata"]["input_level_dbu"] == 18.0
    assert baked["metadata"]["output_level_dbu"] == 12.5
    assert [
        entry["model"]["config"]["layers"][0]["channels"]
        for entry in baked["config"]["submodels"]
    ] == [3, 8]
    checkpoints = list(outdir.rglob("*.ckpt"))
    assert checkpoints
    for checkpoint in checkpoints:
        assert checkpoint.with_suffix(".nam").exists()
        assert checkpoint.with_name(checkpoint.stem + "_parametric.pnam").exists()
        checkpoint_nam = json.loads(
            checkpoint.with_name(checkpoint.stem + "_parametric.pnam").read_text()
        )
        assert checkpoint_nam["metadata"]["input_level_dbu"] == 18.0
        assert checkpoint_nam["metadata"]["output_level_dbu"] == 12.5
    restored = HyperA2.init_from_config(final["config"])
    restored.import_weights(final["weights"])
    assert restored.branch("full")._template._head_scale != pytest.approx(0.01)


def test_mute_anchor_trains_both_hyper_a2_sizes_on_real_audio(monkeypatch):
    from nam.train.parametric import _MuteAnchorConfig

    model = make_model()
    anchor = _MuteAnchorConfig(
        targets=(("Gain", "min"), ("Gain", "max")),
        batch_size=1,
        ny=32,
    )
    module = _ParametricLightningModule(
        model,
        loss_config=_ParametricLossConfig(mute_anchor=anchor),
    )
    module.train()
    module.on_train_epoch_start()
    x = torch.randn(2, model.receptive_field + 63)
    p = torch.full((2, 1), 5.0)
    y = torch.randn(2, 64)
    x[0, -32:] = 0.0
    x[1, -32:] = 0.5
    calls = []
    forward_sizes = model.forward_sizes

    def record(audio, setting, **kwargs):
        calls.append((audio, setting))
        return forward_sizes(audio, setting, **kwargs)

    monkeypatch.setattr(model, "forward_sizes", record)
    first = module._mute_anchor_losses((x, p, y), anchor, 0)
    assert first.shape == (2,) and torch.isfinite(first).all()
    second = module._mute_anchor_losses((x, p, y), anchor, 1)
    assert second.shape == (2,) and torch.isfinite(second).all()
    for (audio, setting), endpoint in zip(calls, (0.0, 10.0)):
        assert audio.shape == (1, model.receptive_field - 1 + 32)
        torch.testing.assert_close(audio[0], x[1, -audio.shape[-1]:])
        assert setting.tolist() == [[endpoint]]
    loss = module.training_step((x, p, y), 2)
    assert torch.isfinite(loss)
    loss.backward()
    for branch in model._branches.values():
        assert any(
            v.grad is not None and v.grad.abs().sum() > 0
            for v in branch._hypernet.parameters()
        )
    assert not any("mute" in spec.to_dict() for spec in model.param_specs)


def test_mute_transition_extra_honors_avoided_defaults():
    from nam.capture.params import KnobSpec
    from nam.capture.planner import _plan_captures_unadjusted, plan_captures

    controls = [
        KnobSpec("Gain", 0, 10, 1, default=5, mute_at_min=True),
        KnobSpec("Tone", 0, 10, 1, default=5, avoid_below=7),
    ]
    for seed in range(12):
        raw, _ = _plan_captures_unadjusted(
            controls, n_train=1, n_validation=0, seed=seed, lhs_method="sobol",
            include_default_as_additional_capture=False,
        )
        if raw[0].params["Gain"] == 1.0:
            continue
        train, validation = plan_captures(
            controls, n_train=1, n_validation=0, seed=seed, lhs_method="sobol",
            include_default_as_additional_capture=False,
        )
        assert not validation
        assert len(train) == 2
        assert train[1].params["Gain"] == 1.0
        assert train[1].params["Tone"] >= 7.0
        break
    else:
        pytest.fail("Expected a Sobol seed without the first unmuted Gain step")


def test_exact_discovered_mute_points_work_without_global_mute_endpoint(monkeypatch):
    from nam.train.parametric import _MuteAnchorConfig

    net_config = config()
    net_config["params"].append({
        "name": "Gain II", "type": "continuous",
        "min": 0, "max": 10, "default": 5,
    })
    model = factory.init("HyperA2", args=(net_config,))
    anchor = _MuteAnchorConfig(
        targets=(),
        point_targets=({"Gain": 0.0, "Gain II": 0.0},),
        batch_size=1, ny=32,
    )
    module = _ParametricLightningModule(
        model, loss_config=_ParametricLossConfig(mute_anchor=anchor)
    )
    input_audio = torch.ones((1, model.receptive_field + 63))
    params = torch.tensor([[5.0, 5.0]])
    target = torch.zeros((1, 64))
    found = []
    forward = model.forward_sizes

    def intercept(audio, setting, **kwargs):
        found.append(setting.tolist()[0])
        return forward(audio, setting, **kwargs)

    monkeypatch.setattr(model, "forward_sizes", intercept)
    for target_index in (0, 1, 5):
        anchor_losses = module._mute_anchor_losses(
            (input_audio, params, target), anchor, target_index
        )
        assert torch.isfinite(anchor_losses).all()
    assert found == [[0.0, 0.0]] * 3
    assert anchor.settings() == {"interval": 1, "batch_size": 1, "ny": 32}


def test_exact_only_mute_anchor_visits_all_marked_points_without_inference(monkeypatch):
    from nam.train.parametric import _MuteAnchorConfig

    model = make_model()
    anchor = _MuteAnchorConfig(
        targets=(), point_targets=({"Gain": 0.0}, {"Gain": 10.0}),
        batch_size=1, ny=32,
    )
    module = _ParametricLightningModule(
        model, loss_config=_ParametricLossConfig(mute_anchor=anchor)
    )
    x = torch.ones(1, model.receptive_field + 63)
    p = torch.full((1, 1), 5.0)
    y = torch.zeros(1, 64)
    recorded = []
    forward = model.forward_sizes

    def observe(audio, setting, **kwargs):
        recorded.append(setting[0, 0].item())
        return forward(audio, setting, **kwargs)

    monkeypatch.setattr(model, "forward_sizes", observe)
    for index in (0, 1, 2, 3, 4):
        loss = module._mute_anchor_losses((x, p, y), anchor, index)
        assert torch.isfinite(loss).all()
    assert recorded == [0.0, 10.0, 0.0, 10.0, 0.0]


def test_virtual_mute_corners_exclude_the_clean_volume_output_extension():
    from nam.capture.export import build_hyper_a2_model_config
    from nam.capture.params import KnobSpec
    from nam.capture.project import new_project

    project = new_project(
        [
            KnobSpec("Gain", 0, 10, 1, 5, mute_at_min=True),
            KnobSpec("Volume", 0, 10, 1, 10, clean_volume=True),
        ],
        n_train=4, n_validation=1, lhs_method="sobol",
    )
    model = build_hyper_a2_model_config(project)
    assert [p["name"] for p in model["net"]["config"]["params"]] == ["Gain"]
    assert model["loss"]["mute_anchor"]["corner_targets"] == [{"Gain": 0.0}]


def test_virtual_muted_corners_override_random_quota_until_all_are_covered(monkeypatch):
    from nam.train.parametric import _MuteAnchorConfig

    model = make_model()
    corners = ({"Gain": 0.0}, {"Gain": 10.0})
    anchor = _MuteAnchorConfig(
        targets=(("Gain", "min"), ("Gain", "max")),
        corner_targets=corners, batch_size=1, ny=32,
    )
    module = _ParametricLightningModule(
        model, loss_config=_ParametricLossConfig(mute_anchor=anchor)
    )
    x = torch.ones(1, model.receptive_field + 63)
    params = torch.tensor([[5.0]])
    y = torch.zeros(1, 64)
    values = []
    original_forward = model.forward_sizes

    def record(audio, setting, **kwargs):
        values.append(setting[0, 0].item())
        return original_forward(audio, setting, **kwargs)

    monkeypatch.setattr(model, "forward_sizes", record)
    for index in (0, 1, 2, 6):
        loss = module._mute_anchor_losses((x, params, y), anchor, index)
        assert torch.isfinite(loss).all()
    # First visit *every* corner, then repeat one in every four calls.
    assert values == [0.0, 10.0, 0.0, 10.0]

    with pytest.raises(ValueError, match="not muted"):
        _ParametricLightningModule(
            make_model(),
            loss_config=_ParametricLossConfig(
                mute_anchor=_MuteAnchorConfig(
                    targets=(("Gain", "min"),),
                    corner_targets=({"Gain": 5.0},),
                )
            ),
        )


def test_mute_anchor_cadence_and_weight_across_optimizer_steps(tmp_path, monkeypatch):
    from pytorch_lightning import Trainer
    from torch.utils.data import DataLoader, TensorDataset
    from nam.train.parametric import _MuteAnchorConfig

    anchor = _MuteAnchorConfig(targets=(("Gain", "min"), ("Gain", "max")), interval=2)
    module = _ParametricLightningModule(
        make_model(), loss_config=_ParametricLossConfig(mute_anchor=anchor)
    )
    calls, losses = [], []

    def size_steps(batch):
        target = batch[-1]
        preds = next(module.parameters()).flatten()[0].expand_as(target) * 0
        yield None, (preds, target, module._get_loss_dict(preds, target))

    def anchor_loss(batch, config, target_index):
        calls.append((module.global_step, config.targets[target_index % len(config.targets)]))
        return torch.ones(1)

    training_step = module.training_step

    def record_step(batch, batch_idx):
        loss = training_step(batch, batch_idx)
        losses.append(loss.item())
        return loss

    monkeypatch.setattr(module, "_size_steps", size_steps)
    monkeypatch.setattr(module, "_mute_anchor_losses", anchor_loss)
    monkeypatch.setattr(module, "training_step", record_step)
    batches = DataLoader(TensorDataset(torch.ones(4, 8)), batch_size=1)
    trainer_options = dict(
        accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, accumulate_grad_batches=2,
    )
    trainer = Trainer(max_epochs=2, **trainer_options)
    trainer.fit(module, batches)
    assert trainer.global_step == 4
    assert calls == [(0, ("Gain", "min"))] * 2 + [(2, ("Gain", "max"))] * 2
    assert losses == pytest.approx([1.5, 1.5, 1.0, 1.0] * 2)

    checkpoint = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["mute_anchor_settings"] == {"interval": 2, "batch_size": 2, "ny": 512}
    with pytest.raises(ValueError, match="different Mute Anchor"):
        module.on_load_checkpoint({**saved, "mute_anchor_settings": None})
    resumed = Trainer(max_epochs=3, **trainer_options)
    resumed.fit(module, batches, ckpt_path=checkpoint)
    assert calls[-2:] == [(4, ("Gain", "min"))] * 2
    assert resumed.global_step == 6


@pytest.mark.parametrize("key", ["interval", "batch_size", "ny"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_invalid_mute_anchor_settings(key, value):
    from nam.train.parametric import _MuteAnchorConfig

    raw = {"targets": [{"name": "Gain", "edge": "min"}]}
    assert _MuteAnchorConfig.from_config(raw).settings() == {
        "interval": 1, "batch_size": 2, "ny": 512,
    }
    with pytest.raises(ValueError, match=rf"Mute Anchor {key} must be a positive integer"):
        _MuteAnchorConfig.from_config({**raw, key: value})


def test_mute_anchor_rejects_unknown_and_invalid_edges():
    from nam.train.parametric import _MuteAnchorConfig

    with pytest.raises(ValueError, match="continuous control"):
        _ParametricLightningModule(
            make_model(),
            loss_config=_ParametricLossConfig(
                mute_anchor=_MuteAnchorConfig(targets=(("Missing", "max"),))
            ),
        )
    with pytest.raises(ValueError, match="named min/max"):
        _MuteAnchorConfig(targets=(("Gain", "middle"),))

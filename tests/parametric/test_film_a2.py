
import json
from copy import deepcopy

import numpy as np
import pytest
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from nam.models._from_nam import init_from_nam
from nam.models.parametric import FiLMA2, HyperA2, export_parametric
from nam.train.parametric import _ParametricLightningModule


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(conditioning):
    return {
        "sample_rate": 48000,
        "params": [
            {"name": "Gain", "min": 0, "max": 10, "default": 5, "type": "continuous"},
            {"name": "Boost", "min": 0, "max": 1, "default": 0, "type": "switch",
             "enum_names": ["Off", "On"]},
        ],
        "conditioning": conditioning,
    }


def make_model(conditioning):
    torch.manual_seed(268)
    model = FiLMA2.init_from_config(config(conditioning)).eval()
    with torch.no_grad():
        model._conditioner[-1].weight.normal_(0, 0.1)
        model._conditioner[-1].bias.normal_(0, 0.1)
    return model


def unfused(model, size, x, params):
    """Apply FiLM to each preactivation as an independent baking reference."""
    stock = deepcopy(model._core(size))
    with torch.no_grad():
        for name, value in stock.named_parameters():
            value.copy_(model._base_parameters(size)[name])
    layers = stock._layer_arrays[0]._layers
    coeff = model._coefficients(model._encode_params(params))
    for layer, affine in zip(layers, coeff):
        gamma, beta = affine[:layer.conv.out_channels].T
        layer._activation.register_forward_pre_hook(
            lambda _module, args, g=gamma, b=beta: (
                args[0] * (1 + g)[None, :, None] + b[None, :, None],
            )
        )
    x = torch.nn.functional.pad(x, (model.receptive_field - 1, 0))
    return stock(x[None, None])[0, 0]


@pytest.mark.parametrize("conditioning", ["linear", "neural"])
def test_folding_and_roundtrip(conditioning, tmp_path):
    model = make_model(conditioning)
    hyper_config = config(conditioning)
    hyper_config.pop("conditioning")
    stock = HyperA2.init_from_config(hyper_config)
    assert model.receptive_field == stock.receptive_field == 6347
    assert [entry["base_weight_count"] for entry in model._export_submodels()] == [1871, 12146]

    export_parametric(model, tmp_path, output_scale=2.5)
    document = json.loads((tmp_path / "model.pnam").read_text())
    assert document["architecture"] == "ParametricSlimmableContainer"
    assert [entry["architecture"] for entry in document["config"]["submodels"]] == ["FiLMA2"] * 2
    restored = init_from_nam(document)
    x = torch.randn(96)
    settings = torch.tensor([[2.0, 0], [8.0, 1]])
    with torch.no_grad():
        for index, size in enumerate(model.sizes):
            assert model._core(size).export_config() == stock.branch(size)._template.export_config()
            for p in settings:
                expected = unfused(model, size, x, p)
                with model.using_size(size), restored.using_size(size):
                    torch.testing.assert_close(model(x, p), expected, atol=1e-7, rtol=2e-5)
                    torch.testing.assert_close(restored(x, p), expected / 2.5)
                baked = model.bake_size(size, p, output_scale=2.5)
                torch.testing.assert_close(baked(x), expected)
                packed = model.bake(p, output_scale=2.5)
                torch.testing.assert_close(packed.extract_submodel(index)(x), expected)
            p = settings[1]
            baked.export(tmp_path, basename=size, include_snapshot=False)
            reloaded = init_from_nam(json.loads((tmp_path / f"{size}.nam").read_text()))
            torch.testing.assert_close(reloaded(x), unfused(model, size, x, p) / 2.5)
        vectorized = model(x, settings)
        torch.testing.assert_close(vectorized, torch.stack([model(x, p) for p in settings]))
    from paranam.model_editor_core import apply_model_edits, editable_controls

    controls = editable_controls(document)
    controls[0]["name"] = "Drive"
    edited = apply_model_edits(document, document.get("metadata", {}), controls)
    assert edited["weights"] == document["weights"]
    restored_edit = init_from_nam(edited)
    assert restored_edit.param_names == ("Drive", "Boost")
    with torch.no_grad():
        torch.testing.assert_close(restored_edit(x, settings), vectorized / 2.5)

    malformed = document["weights"].copy()
    malformed[0] += 1
    with pytest.raises(ValueError, match="shared"):
        restored.import_weights(malformed)


def test_train_joint_mute_and_resume(tmp_path):
    raw = config("neural")
    model_config = {
        "net": {"name": "ParametricSlimmableContainer", "config": raw},
        "optimizer": {"class": "Adam", "lr": 0.001},
        "hypernet_lr_scale": 0.25,
        "lr_scheduler": None,
        "loss": {"training_recipe": "a2", "synthetic_mute": True},
    }
    module = _ParametricLightningModule.init_from_config(model_config)
    audio = torch.randn(1, module.net.receptive_field + 31)
    settings = torch.tensor([[2., 1.]])
    target = torch.randn(1, 32)
    loader = DataLoader(TensorDataset(audio, settings, target), batch_size=1)
    initial = next(module.net._template.parameters()).detach().clone()

    def fit(net, epochs, resume=None):
        trainer = Trainer(max_epochs=epochs, accelerator="cpu", logger=False,
                          enable_checkpointing=False, enable_progress_bar=False,
                          enable_model_summary=False, num_sanity_val_steps=0)
        trainer.fit(net, loader, loader, ckpt_path=resume)
        return trainer

    first = fit(module, 1)
    assert first.global_step == 1
    assert {"ESR_full", "ESR_lite"} <= set(first.callback_metrics)
    assert not torch.equal(next(module.net._template.parameters()), initial)
    assert torch.count_nonzero(module.net._conditioner[-1].weight)
    assert [group["lr"] for group in first.optimizers[0].param_groups] == [0.001, 0.00025]
    for size, (_, _, losses) in module._size_steps((audio, settings, target), synthetic_mute=True):
        assert size in module.net.sizes
        assert list(losses) == ["MSE"]

    checkpoint = tmp_path / "resume.ckpt"
    first.save_checkpoint(checkpoint)
    resumed = _ParametricLightningModule.init_from_config(model_config)
    second = fit(resumed, 2, checkpoint)
    assert second.global_step == 2
    assert not torch.equal(resumed.net._conditioner[-1].weight,
                           module.net._conditioner[-1].weight)

    from paranam.training import export_best_model

    saved = torch.load(checkpoint, weights_only=False)
    saved.update(validation_esr_full=1.0, validation_esr_lite=1.1, output_scale=2.0)
    torch.save(saved, checkpoint)
    spec = {"project": str(tmp_path), "model": model_config}
    path = export_best_model(
        None, spec, 0, tmp_path / "best.pnam",
        selection=(dict.fromkeys(("full", "lite"), checkpoint), {}),
    )
    document = json.loads((tmp_path / "best.pnam").read_text())
    assert path == str(tmp_path / "best.pnam")
    metadata = document["metadata"]["training"]
    assert metadata["epoch_full"] == metadata["epoch_lite"] == saved["epoch"] + 1
    restored = init_from_nam(document)
    with torch.no_grad():
        torch.testing.assert_close(restored.forward_sizes(audio, settings),
                                   module.net.forward_sizes(audio, settings) / 2)


@pytest.mark.parametrize("conditioning", ["linear", "neural"])
def test_native_playback_matches_scaled_python_bake(conditioning, tmp_path):
    from paranam.playback import NativeModel, runtime_path

    try:
        runtime_path()
    except RuntimeError:
        pytest.skip("Build the native playback engine first")
    model = make_model(conditioning)
    scale = 2.5
    export_parametric(model, tmp_path, output_scale=scale)
    samples = np.random.default_rng(42).normal(0, 0.2, 401).astype(np.float32)
    with torch.no_grad():
        for size in model.sizes:
            for params in (torch.tensor([2., 0.]), torch.tensor([9., 1.])):
                native = NativeModel(tmp_path / "model.pnam", block_size=73, lite=size == "lite")
                try:
                    native.parameters(params.numpy())
                    # Drain the 100 ms weight ramp and the causal receptive field.
                    native.process(np.zeros(9600 + model.receptive_field, dtype=np.float32))
                    baked = model.bake_size(size, params)
                    expected = baked(torch.from_numpy(samples)).numpy() / scale
                    np.testing.assert_allclose(native.process(samples), expected, atol=2e-6,
                                               rtol=2e-4)
                finally:
                    native.close()


import pytest
import torch
from nam.models.losses import dual_file_esr, dual_file_energies
from paranam.recipes import training_preset, preset_batch_size


def test_paired_floor_excludes_mutes_and_preserves_ratio():
    y = torch.tensor([1., -1., 1., -1.], dtype=torch.float64)
    quiet = y * 1e-3
    pairs = dual_file_energies([y, y, quiet, torch.zeros_like(y)], {3})
    assert pairs[2][0] == pytest.approx(1e-3)
    assert pairs[2][1] / pairs[2][0] == pytest.approx(1.85**2)
    assert pairs[3] is None
    pred = (quiet * 2).float().requires_grad_()
    loss = dual_file_esr(pred, quiet.float(), pairs[2])
    assert loss.item() == pytest.approx(0.002)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("name", ["FiLMA2 Linear", "FiLMA2 Neural"])
def test_colab_recipe(name):
    options = training_preset(name)
    assert options["waveform_loss"] == "dual_file_esr"
    assert options["learning_rate"] == options["hypernetwork_learning_rate"] == 0.002
    assert options["lr_decay"] == options["hypernetwork_lr_decay"] == 0.95
    assert options["rate_preset"] == options["hypernetwork_rate_preset"] == "fixed_1000"
    assert options["ny"] == 48000 and preset_batch_size(name) == 12
    assert options["weight_decay"] == 0 and not options["rms_loss_normalization"]


def test_faverooo_exact_scheduler_clock():
    from nam.train._lr_scheduler import GroupUpdateCadenceLR
    p = torch.nn.Parameter(torch.ones(1))
    q = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.Adam([{"params": [p]}, {"params": [q]}], lr=0.002)
    first = GroupUpdateCadenceLR(opt, 0, 1000, 0.95)
    second = GroupUpdateCadenceLR(opt, 1, 1000, 0.95)
    for _ in range(999):
        opt.step()
        first.step()
        second.step()
    assert [g["lr"] for g in opt.param_groups] == [0.002, 0.002]
    opt.step()
    first.step()
    second.step()
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([0.0019, 0.0019])


def test_faverooo_external_stimulus_wrap(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from paranam import signals
    rate = 48000
    source, reference, out = [tmp_path / name for name in ("input.wav", "standard.wav", "adapted.wav")]
    audio = np.full(rate, 0.25)
    sf.write(source, audio, rate, subtype="PCM_24")
    sf.write(reference, np.concatenate((np.full(10 * rate, 0.125), np.full(9 * rate, -0.125))), rate, subtype="PCM_24")
    monkeypatch.setattr(signals, "asset_path", lambda _: reference)
    signals.prepare_faverooo_input(source, out)
    result, actual_rate = sf.read(out)
    assert actual_rate == rate and len(result) == 20 * rate
    np.testing.assert_array_equal(result[10 * rate:-9 * rate], audio)
    np.testing.assert_array_equal(result[:10 * rate], 0.125)
    np.testing.assert_array_equal(result[-9 * rate:], -0.125)

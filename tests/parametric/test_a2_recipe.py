import pytest
import torch

from nam.capture.export import build_hyper_a2_model_config
from nam.capture.params import KnobSpec
from nam.capture.project import new_project
from nam.data import Dataset, ConcatDataset, ConcatDatasetValidationError
from nam.models.parametric._dataset import ParametricDataset
from nam.train.parametric import _ParametricLightningModule, _make_parametric_dataloader
from nam.train.lightning_module import PackedLightningModule


def test_a2_recipe_conditions_every_weight_and_uses_the_static_sum_loss():
    torch.set_num_threads(1)
    project = new_project([KnobSpec("Gain", 0, 10)], n_train=2, n_validation=1)
    config = build_hyper_a2_model_config(project)
    config["loss"]["mrstft_weight"] = None
    module = _ParametricLightningModule.init_from_config(config)
    for branch in module.net._branches.values():
        assert branch._hypernet.config["mode"] == "full"
        assert set(branch._hypernet.target_names) == set(branch._template_parameters)
    params = torch.tensor([[3.0]])
    x = 0.1 * torch.randn(1, module.net.receptive_field + 64)
    y = 0.2 * torch.randn(1, 65)
    module.net.set_uniform_batch_params(True)
    module.on_train_epoch_start()
    actual = module.training_step((x, params, y), 0)
    baked = module.net.bake(torch.tensor([3.0]))
    reference = PackedLightningModule(net=baked)
    expected = reference.training_step((x, y), 0)
    torch.testing.assert_close(actual, expected)


def test_whole_validation_clips_allow_different_lengths_and_preserve_levels():
    first = Dataset(
        x=torch.ones(100), y=torch.full((100,), 0.1), nx=5, ny=None, sample_rate=48000
    )
    second = Dataset(
        x=torch.ones(113), y=torch.full((113,), 0.5), nx=5, ny=None, sample_rate=48000
    )
    pieces = [
        ParametricDataset(first, torch.tensor([2.0])),
        ParametricDataset(second, torch.tensor([8.0])),
    ]
    with pytest.raises(ConcatDatasetValidationError):
        ConcatDataset(pieces)
    combined = ConcatDataset(pieces, allow_variable_ny=True)
    loader = _make_parametric_dataloader(combined, dict(batch_size=1, shuffle=False))
    batches = list(loader)
    assert [batch[-1].shape[-1] for batch in batches] == [96, 109]
    assert float(batches[1][-1].mean() / batches[0][-1].mean()) == pytest.approx(5)


@pytest.mark.parametrize("name", ["Adam", "AdamW"])
def test_hypernetwork_lr_groups_scheduler_and_checkpoint_compatibility(name):
    project = new_project([KnobSpec("Gain", 0, 10)], n_train=2, n_validation=1)
    config = build_hyper_a2_model_config(project)
    config["optimizer"]["class"] = name
    module = _ParametricLightningModule.init_from_config(config)
    configured = module.configure_optimizers()
    optimizer = configured["optimizer"]
    base, hyper = optimizer.param_groups
    assert base["lr"] == 0.004
    assert hyper["lr"] == pytest.approx(0.0004)
    expected_hyper = {
        id(p) for b in module.net._branches.values() for p in b._hypernet.parameters()
    }
    assert {id(p) for p in hyper["params"]} == expected_hyper
    all_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(all_ids) == len(set(all_ids)) == len(list(module.parameters()))
    assert set(all_ids) == {id(p) for p in module.parameters()}
    sum(p.square().mean() for p in module.parameters()).backward()
    optimizer.step()
    configured["lr_scheduler"]["scheduler"].step()
    assert hyper["lr"] / base["lr"] == pytest.approx(0.1)
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    module.on_load_checkpoint(checkpoint)
    assert checkpoint["hypernet_lr_scale"] == 0.1
    with pytest.raises(ValueError, match="different hypernetwork learning rate"):
        module.on_load_checkpoint({"sample_rate": None, "optimizer_class": name})
    # Legacy configurations keep a single optimizer group for resumable moments.
    config.pop("hypernet_lr_scale")
    legacy = _ParametricLightningModule.init_from_config(config)
    assert len(legacy.configure_optimizers()["optimizer"].param_groups) == 1
    legacy.on_load_checkpoint({"sample_rate": None, "optimizer_class": name})


def test_two_knob_multistep_training_stays_stable_at_held_out_setting():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    project = new_project(
        [KnobSpec("Gain", 0, 10), KnobSpec("Tone", 0, 10)],
        n_train=2,
        n_validation=1,
    )
    config = build_hyper_a2_model_config(project)
    config["loss"]["mrstft_weight"] = None
    module = _ParametricLightningModule.init_from_config(config)
    module.net.set_uniform_batch_params(True)
    optimizer = module.configure_optimizers()["optimizer"]
    x = 0.1 * torch.randn(2, module.net.receptive_field + 255)
    y = 0.5 * torch.tanh(3 * x[:, -256:])
    settings = 1 + 8 * torch.rand(10, 2, generator=torch.Generator().manual_seed(123))
    module.on_train_epoch_start()
    losses = []
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = module.training_step((x, settings[step % 10].expand(2, -1), y), step)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    module.eval()
    with torch.no_grad():
        pred = module.net.forward_sizes(x, torch.tensor([0.0, 10.0]))
        esr = (pred - y[:, None]).square().mean(dim=(0, 2)) / y.square().mean()
    assert torch.isfinite(esr).all()
    assert (esr < 2).all(), esr
    assert sum(losses[-10:]) < sum(losses[:10])

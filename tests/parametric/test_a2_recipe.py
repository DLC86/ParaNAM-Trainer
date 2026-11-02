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

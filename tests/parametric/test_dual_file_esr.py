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

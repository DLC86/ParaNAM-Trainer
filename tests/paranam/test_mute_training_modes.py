"""Beta 2: mute training modes are trainer-only and retain the shipped .pnam ABI."""

import shutil

import pytest
import torch

from nam.capture.params import KnobSpec
from nam.train.parametric import _ParametricLightningModule
from paranam import training
from paranam.workspace import Workspace, read_json


def _mute_ready(workspace):
    workspace.draft_controls = [
        KnobSpec("Gain", 0, 10, 1, 5, mute_at_min=True).to_dict()
    ]
    workspace.generate_plan()
    for entry in workspace.project.entries:
        output = workspace.directory / entry.y_path
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            workspace.directory / workspace.project.input_for_split(entry.split),
            output,
        )
        entry.status = "captured"
        entry.delay = 0
    workspace.save()


@pytest.mark.parametrize("mode", ["anchor", "virtual", "weight"])
def test_mode_is_saved_and_blocks_incompatible_resume(workspace, mode):
    _mute_ready(workspace)
    workspace.preferences["advanced_parametric"] = {"mute_training_mode": mode}
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    assert spec["model"]["loss"]["mute_anchor"].get("mode", "anchor") == mode
    assert spec["learning"]["rms_loss_normalization"]
    assert spec["learning"]["snr_weighted_loss_normalization"]
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(workspace, job)
    workspace.preferences["advanced_parametric"]["mute_training_mode"] = (
        "virtual" if mode != "virtual" else "weight"
    )
    assert not training.job_can_resume(workspace, job)


@pytest.mark.parametrize("mode", ["virtual", "weight"])
def test_new_mute_objective_backpropagates_without_extra_audio_forward(workspace, mode):
    torch.set_num_threads(1)
    _mute_ready(workspace)
    workspace.preferences["advanced_parametric"] = {"mute_training_mode": mode}
    spec = read_json(training.make_job(workspace) / "spec.json")
    module = _ParametricLightningModule.init_from_config(spec["model"])
    mute = module._loss_config.mute_anchor
    assert mute is not None and mute.mode == mode
    assert module.net._get_export_architecture() == "ParametricSlimmableContainer"
    if mode == "virtual":
        x = torch.randn(1, module.net.receptive_field + 16) * 0.1
        loss = module._virtual_mute_step((x,), mute, 0)
    else:
        loss = module._mute_weight_losses(mute, 0).sum()
    assert torch.isfinite(loss).item()
    assert loss.requires_grad
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in module.parameters()
    )


def test_mute_selector_persists_in_advanced_settings(window, workspace):
    window._attach(workspace)
    field = window.advanced_fields["mute_training_mode"]
    assert field.currentData() == "anchor"
    for method in ("virtual", "weight", "anchor"):
        index = field.findData(method)
        assert index >= 0
        field.setCurrentIndex(index)
        assert Workspace(workspace.directory).training_options("parametric")[
            "mute_training_mode"
        ] == method
    field.setCurrentIndex(field.findData("weight"))
    window.capture_kind.setCurrentIndex(window.capture_kind.findData("static"))
    assert window.advanced_panel.layout().isRowVisible(field) is False
    window.capture_kind.setCurrentIndex(window.capture_kind.findData("parametric"))
    assert field.currentData() == "weight"

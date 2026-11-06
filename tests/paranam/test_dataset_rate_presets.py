import math

import pytest

from nam.data import Split, init_dataset
from nam.models.parametric import data_config_from_model
from nam.train.parametric import _ParametricLightningModule, _make_parametric_dataloader
from paranam import training
from paranam.recipes import (
    a2_update_schedule,
    training_preset,
    workspace_group_schedules,
)
from paranam.workspace import Workspace, read_json
from tests.paranam.test_workflow055 import ready_audio


@pytest.mark.parametrize("preset", ["a2_dataset", "a2_dataset_sqrt"])
@pytest.mark.parametrize("batch,ny", [(4, 8192), (2, 4096)])
def test_dataset_clock_counts_actual_loader_batches(workspace, preset, batch, ny):
    ready_audio(workspace)
    workspace.preferences["batch_size"] = batch
    workspace.preferences["advanced_parametric"] = dict(
        rate_preset=preset, hypernetwork_rate_preset="follow", ny=ny
    )
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    schedule = spec["preferences"]["rate_schedule"]
    module = _ParametricLightningModule.init_from_config(spec["model"])
    data = data_config_from_model(spec["data"], spec["model"])
    data["common"]["nx"] = module.net.receptive_field
    for split in ("train", "validation"):
        for row in data[split]:
            for key in ("x_path", "y_path"):
                row[key] = str(workspace.directory / row[key])
    dataset = init_dataset(data, Split.TRAIN)
    try:
        loader = _make_parametric_dataloader(
            dataset, spec["learning"]["train_dataloader"]
        )
        steps = len(loader)
        assert schedule["steps_per_epoch"] == steps
    finally:
        dataset.teardown()
    expected = steps if preset == "a2_dataset" else math.sqrt(steps * 62)
    assert schedule["decay_steps"] == pytest.approx(expected)
    assert schedule["dataset_reference_ratio"] == steps / 62
    assert spec["model"]["lr_scheduler"]["class"] == "UpdateCadenceLR"
    doubled = a2_update_schedule(
        {**spec["data"], "train": spec["data"]["train"] * 2},
        workspace.directory,
        batch,
        ny,
        2,
        0.004,
        0.0004,
        normalization=preset,
    )
    assert doubled["decay_steps"] == pytest.approx(
        expected * (2 if preset == "a2_dataset" else math.sqrt(2))
    )
    if preset == "a2_dataset":
        assert schedule["reductions"] == doubled["reductions"] == 2
        assert schedule["gamma_epoch"] == pytest.approx(0.994)


@pytest.mark.parametrize(
    "audio,hyper",
    [
        ("a2_dataset", "a2_dataset_sqrt"),
        ("a2_dataset_sqrt", "a2_dataset"),
        ("manual", "a2_dataset"),
    ],
)
def test_independent_dataset_clocks_restore_exactly(workspace, audio, hyper):
    ready_audio(workspace)
    workspace.preferences["advanced_parametric"] = dict(
        rate_preset=audio,
        hypernetwork_rate_preset=hyper,
        learning_rate=0.004,
        hypernetwork_learning_rate=0.0004,
        lr_decay=0.994,
        hypernetwork_lr_decay=0.994,
    )
    spec = read_json(training.make_job(workspace) / "spec.json")
    descriptions = spec["preferences"]["rate_schedule"]
    assert descriptions == workspace_group_schedules(workspace)
    module = _ParametricLightningModule.init_from_config(spec["model"])
    optimizers, schedulers = module.configure_optimizers()
    optimizer = optimizers[0]
    for step in range(1, 181):
        optimizer.step()
        for schedule in schedulers:
            if schedule["interval"] == "step" or step % 20 == 0:
                schedule["scheduler"].step()
        for group, description in zip(optimizer.param_groups, descriptions.values()):
            clock = step if description["interval"] == "step" else step // 20
            assert group["lr"] == pytest.approx(
                description["initial_learning_rate"]
                * description["gamma"] ** math.floor(clock / description["step_size"])
            )
        if step == 73:
            optimizer_state = optimizer.state_dict()
            states = [s["scheduler"].state_dict() for s in schedulers]
            optimizers, schedulers = module.configure_optimizers()
            optimizer = optimizers[0]
            optimizer.load_state_dict(optimizer_state)
            for schedule, state in zip(schedulers, states):
                schedule["scheduler"].load_state_dict(state)


def test_new_choices_replace_removed_presets_in_old_project(window, workspace):
    ready_audio(workspace)
    workspace.preferences["advanced_parametric"] = training_preset("HyperA2")
    window._attach(workspace)
    for key in ("rate_preset", "hypernetwork_rate_preset"):
        selector = window.advanced_fields[key]
        assert selector.findData("a2_dataset") >= 0
        assert selector.findData("a2_dataset_sqrt") >= 0
        assert selector.findData("a2_controls") == -1
        assert selector.currentData() == "a2_dataset"
    assert "Independent LR schedules" in window.rate_preview.text()
    hyper = window.advanced_fields["hypernetwork_rate_preset"]
    hyper.setCurrentIndex(hyper.findData("a2_dataset_sqrt"))
    hyper.activated.emit(hyper.currentIndex())
    assert (
        Workspace(workspace.directory).training_options()["hypernetwork_rate_preset"]
        == "a2_dataset_sqrt"
    )
    workspace.preferences["advanced_parametric"].update(
        rate_preset="a2_updates", hypernetwork_rate_preset="a2_controls"
    )
    workspace.save()
    before = workspace.fingerprint()
    window.load_advanced_options()
    window.save_preferences()
    assert workspace.fingerprint() == before
    for key in ("rate_preset", "hypernetwork_rate_preset"):
        selector = window.advanced_fields[key]
        assert selector.currentData() == "a2_dataset"
        assert all(
            selector.findData(old) == -1
            for old in ("a2_updates", "a2_controls", "a2_slow")
        )

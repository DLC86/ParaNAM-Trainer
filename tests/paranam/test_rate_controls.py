import math

import pytest

from nam.capture.params import KnobSpec, SwitchSpec
from paranam import training
from paranam.recipes import (
    a2_update_schedule,
    workspace_update_schedule,
)
from paranam.workspace import Workspace, read_json
from tests.paranam.test_workflow055 import ready_audio


def test_fractional_update_boundaries_do_not_drift_and_survive_resume():
    import torch
    from nam.train._lr_scheduler import UpdateCadenceLR

    parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in range(2)]
    optimizer = torch.optim.Adam(
        [
            {"params": [parameters[0]], "lr": 0.004},
            {"params": [parameters[1]], "lr": 0.0004},
        ]
    )
    interval = 62 * 2 / math.sqrt(2)
    scheduler = UpdateCadenceLR(optimizer, step_size=interval)
    observed = []
    previous = scheduler.get_last_lr()
    for step in range(1, 2001):
        optimizer.step()
        scheduler.step()
        if scheduler.get_last_lr() != previous:
            observed.append(step)
        previous = scheduler.get_last_lr()
        if step == 351:
            optimizer_state, scheduler_state = (
                optimizer.state_dict(),
                scheduler.state_dict(),
            )
            optimizer = torch.optim.Adam(
                [
                    {"params": [parameters[0]], "lr": 0.004},
                    {"params": [parameters[1]], "lr": 0.0004},
                ]
            )
            scheduler = UpdateCadenceLR(optimizer, step_size=1)
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
            assert scheduler.step_size == interval
    expected = [
        math.ceil(k * interval) for k in range(1, 24) if math.ceil(k * interval) <= 2000
    ]
    assert observed == expected
    assert observed[:4] == [88, 176, 264, 351]
    assert scheduler.get_last_lr() == pytest.approx(
        [0.004 * 0.994 ** len(expected), 0.0004 * 0.994 ** len(expected)]
    )


@pytest.mark.parametrize(
    "audio_preset,hyper_preset",
    [
        ("a2_dataset", "a2_dataset_sqrt"),
        ("a2_dataset_sqrt", "a2_dataset"),
        ("manual", "a2_dataset_sqrt"),
        ("a2_dataset_sqrt", "manual"),
        ("manual", "manual"),
    ],
)
def test_independent_group_schedules_preserve_both_clocks_on_resume(
    workspace, audio_preset, hyper_preset
):
    import torch
    from nam.train.parametric import _ParametricLightningModule
    from paranam.recipes import workspace_group_schedules

    ready_audio(workspace)
    workspace.preferences["advanced_parametric"] = dict(
        rate_preset=audio_preset,
        hypernetwork_rate_preset=hyper_preset,
        learning_rate=0.001,
        hypernetwork_learning_rate=0.001,
        lr_decay=0.9 if audio_preset == "manual" else 0.994,
        hypernetwork_lr_decay=0.8 if hyper_preset == "manual" else 0.994,
    )
    workspace.save()
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    descriptions = spec["preferences"]["rate_schedule"]
    assert descriptions == workspace_group_schedules(workspace)
    module = _ParametricLightningModule.init_from_config(spec["model"])
    optimizers, schedules = module.configure_optimizers()
    optimizer = optimizers[0]
    # Equal initial rates still need distinct parameter groups for independent clocks.
    assert len(optimizer.param_groups) == 2
    for step in range(1, 301):
        optimizer.step()
        for schedule in schedules:
            if schedule["interval"] == "step" or step % 100 == 0:
                schedule["scheduler"].step()
        for group, description in zip(optimizer.param_groups, descriptions.values()):
            clock = step if description["interval"] == "step" else step // 100
            reductions = math.floor(clock / description["step_size"])
            assert group["lr"] == pytest.approx(
                0.001 * description["gamma"] ** reductions
            )
        if step == 123:
            saved_optimizer = optimizer.state_dict()
            saved_schedulers = [
                schedule["scheduler"].state_dict() for schedule in schedules
            ]
            optimizers, schedules = module.configure_optimizers()
            optimizer = optimizers[0]
            optimizer.load_state_dict(saved_optimizer)
            for schedule, state in zip(schedules, saved_schedulers):
                schedule["scheduler"].load_state_dict(state)
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(workspace, job)
    workspace.preferences["advanced_parametric"]["hypernetwork_rate_preset"] = "follow"
    assert not training.job_can_resume(workspace, job)


def test_follow_setting_keeps_legacy_job_fingerprint(workspace):
    from tests.paranam.test_training import ready

    ready(workspace)
    # Removing inactive new fields must reproduce the previous fingerprint exactly.
    options = workspace.training_options()
    original_method = workspace.training_options

    def legacy_options(*args):
        return {
            key: value
            for key, value in original_method(*args).items()
            if key not in ("hypernetwork_rate_preset", "hypernetwork_lr_decay")
        }

    workspace.training_options = legacy_options
    legacy = workspace.fingerprint()
    workspace.training_options = original_method
    assert workspace.fingerprint() == legacy
    workspace.preferences["advanced_parametric"] = dict(
        options, hypernetwork_lr_decay=0.8
    )
    assert workspace.fingerprint() == legacy


def test_independent_presets_preview_and_edits_are_saved(window, workspace):
    ready_audio(workspace)
    workspace.preferences["advanced_parametric"] = dict(
        rate_preset="a2_dataset", hypernetwork_rate_preset="a2_dataset_sqrt"
    )
    window._attach(workspace)
    hyper = window.advanced_fields["hypernetwork_rate_preset"]
    hyper.setCurrentIndex(hyper.findData("a2_dataset_sqrt"))
    hyper.activated.emit(hyper.currentIndex())
    assert "Independent LR schedules" in window.rate_preview.text()
    assert "Hypernetwork: Dataset-normalized √" in window.rate_preview.text()
    assert (
        Workspace(workspace.directory).training_options()["hypernetwork_rate_preset"]
        == "a2_dataset_sqrt"
    )
    window.advanced_fields["hypernetwork_learning_rate"].setValue(0.0002)
    main = window.advanced_fields["rate_preset"]
    main.setCurrentIndex(main.findData("a2_dataset"))
    main.activated.emit(main.currentIndex())
    assert window.advanced_fields["hypernetwork_learning_rate"].value() == 0.0002
    hyper.setCurrentIndex(hyper.findData("manual"))
    hyper.activated.emit(hyper.currentIndex())
    assert window.advanced_fields["hypernetwork_lr_decay"].isEnabled()
    assert not window.advanced_fields["lr_decay"].isEnabled()


@pytest.mark.parametrize("old_presets", [False, True])
@pytest.mark.parametrize("zeroed_progress", [False, True])
def test_real_training_restores_two_lightning_scheduler_states(
    workspace, monkeypatch, zeroed_progress, old_presets
):
    import torch
    from nam.capture.project import atomic_write_json
    from nam.train.parametric import _ParametricLightningModule
    from tests.paranam.test_training import tiny_job

    job = tiny_job(workspace, monkeypatch)
    spec = read_json(job / "spec.json")
    if old_presets:
        workspace.preferences["advanced_parametric"].update(
            rate_preset="a2_updates", hypernetwork_rate_preset="a2_controls"
        )
        workspace.save()
        spec["preferences"]["advanced_training"] = workspace.training_options()
        spec["fingerprint"] = workspace.fingerprint()
    # Accelerated diagnostic clocks let a two-epoch test cross both boundaries.
    spec["model"]["lr_scheduler"] = {
        "class": "IndependentGroupLR",
        "kwargs": {
            "audio": dict(interval="epoch", step_size=1, gamma=0.9),
            "hypernetwork": dict(interval="step", step_size=1.5, gamma=0.8),
        },
    }
    atomic_write_json(job / "spec.json", spec)
    original = _ParametricLightningModule.training_step

    def pause_during_second_epoch(module, *args, **kwargs):
        if module.current_epoch == 1:
            training.command(job, "pause")
        return original(module, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            _ParametricLightningModule, "training_step", pause_during_second_epoch
        )
        training.run_job(job)
    assert training.job_progress(job)["status"] == "paused"
    paused = torch.load(job / "resume-0.ckpt", map_location="cpu", weights_only=False)
    assert len(paused["lr_schedulers"]) == 2
    if zeroed_progress:
        import hashlib

        digest = hashlib.sha256((job / "resume-0.ckpt").read_bytes()).digest()
        (job / "progress.json").write_bytes(b"\0" * 15098)
        recovered = training.job_progress(job)
        assert recovered["completed_epoch"] == 1 and recovered["restart_epoch"] == 2
        assert hashlib.sha256((job / "resume-0.ckpt").read_bytes()).digest() == digest
    assert training.make_job(workspace, resume=True) == job
    training.run_job(job)
    assert training.job_progress(job)["status"] == "completed"
    final = torch.load(job / "resume-0.ckpt", map_location="cpu", weights_only=False)
    groups = final["optimizer_states"][0]["param_groups"]
    assert groups[0]["lr"] == pytest.approx(0.004 * 0.9**2)
    assert groups[1]["lr"] == pytest.approx(0.0004 * 0.8)
    assert [state["last_epoch"] for state in final["lr_schedulers"]] == [2, 2]

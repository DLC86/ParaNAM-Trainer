import math

import pytest

from nam.capture.params import KnobSpec, SwitchSpec
from paranam import training
from paranam.recipes import (
    a2_update_schedule,
    workspace_update_schedule,
    control_cadence_multiplier,
)
from paranam.workspace import Workspace, read_json
from tests.paranam.test_workflow055 import ready_audio


@pytest.mark.parametrize("preset", ["a2_controls", "a2_slow"])
@pytest.mark.parametrize(
    "controls",
    [
        [KnobSpec("Gain", 0, 10)],
        [KnobSpec("Gain", 0, 10), KnobSpec("Tone", 0, 10)],
        [KnobSpec("Gain", 0, 10), KnobSpec("Tone", 0, 10), KnobSpec("Bass", 0, 10)],
        [KnobSpec("Gain", 0, 10), SwitchSpec("Bright", ("Off", "On"))],
        [
            KnobSpec("Gain", 0, 10),
            KnobSpec("Tone", 0, 10),
            SwitchSpec("Bright", ("Off", "On")),
        ],
    ],
)
def test_controls_scaled_scheduler_counts_controls_and_resumes_at_boundary(
    workspace, controls, preset
):
    import torch
    from nam.train.parametric import _ParametricLightningModule

    workspace.draft_controls = [control.to_dict() for control in controls]
    workspace.preferences.update(n_train=4, n_validation=2)
    ready_audio(workspace)
    workspace.preferences["advanced_parametric"] = dict(rate_preset=preset)
    workspace.save()
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    schedule = spec["preferences"]["rate_schedule"]
    count = control_cadence_multiplier(workspace.project) / (
        math.sqrt(2) if preset == "a2_slow" else 1
    )
    assert schedule["cadence_multiplier"] == count
    assert schedule["decay_steps"] == schedule["reference_steps"] * count
    assert (
        spec["model"]["lr_scheduler"]["kwargs"]["step_size"] == schedule["decay_steps"]
    )
    assert workspace_update_schedule(workspace) == schedule
    original = a2_update_schedule(
        spec["data"], workspace.directory, 4, 8192, 2, 0.004, 0.0004
    )
    assert schedule["steps_per_epoch"] == original["steps_per_epoch"]
    assert schedule["gamma_epoch"] == pytest.approx(
        original["gamma_epoch"] ** (1 / count)
    )
    assert schedule["reductions"] == math.floor(
        schedule["steps_per_epoch"] * 2 / schedule["decay_steps"]
    )
    assert isinstance(schedule["decay_steps"], float if preset == "a2_slow" else int)
    assert schedule["final_learning_rate"] == pytest.approx(
        0.004 * 0.994 ** schedule["reductions"]
    )
    assert schedule["final_hypernetwork_learning_rate"] == pytest.approx(
        0.0004 * 0.994 ** schedule["reductions"]
    )

    module = _ParametricLightningModule.init_from_config(spec["model"])
    configured = module.configure_optimizers()
    optimizer = configured["optimizer"]
    scheduler = configured["lr_scheduler"]["scheduler"]
    assert configured["lr_scheduler"]["interval"] == "step"
    initial = [group["lr"] for group in optimizer.param_groups]
    for _ in range(math.ceil(schedule["decay_steps"]) - 1):
        optimizer.step()
        scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == initial
    restored = module.configure_optimizers()
    restored["optimizer"].load_state_dict(optimizer.state_dict())
    restored["lr_scheduler"]["scheduler"].load_state_dict(scheduler.state_dict())
    restored["optimizer"].step()
    restored["lr_scheduler"]["scheduler"].step()
    assert [
        group["lr"] for group in restored["optimizer"].param_groups
    ] == pytest.approx([lr * 0.994 for lr in initial])
    doubled = a2_update_schedule(
        {**spec["data"], "train": spec["data"]["train"] * 2},
        workspace.directory,
        4,
        8192,
        2,
        0.004,
        0.0004,
        cadence_multiplier=count,
    )
    assert doubled["decay_steps"] == schedule["decay_steps"]
    assert doubled["steps_per_epoch"] == 2 * schedule["steps_per_epoch"]
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(workspace, job)
    workspace.preferences["advanced_parametric"]["rate_preset"] = "a2_updates"
    assert not training.job_can_resume(workspace, job)


@pytest.mark.parametrize(
    "preset,expected", [("a2_controls", 2), ("a2_slow", math.sqrt(2))]
)
def test_scaled_preset_preview_and_selection_are_saved(
    window, workspace, preset, expected
):
    workspace.draft_controls.append(SwitchSpec("Bright", ("Off", "On")).to_dict())
    workspace.preferences.update(n_train=4, n_validation=2)
    ready_audio(workspace)
    window._attach(workspace)
    selector = window.advanced_fields["rate_preset"]
    selector.setCurrentIndex(selector.findData(preset))
    selector.activated.emit(selector.currentIndex())
    schedule = workspace_update_schedule(workspace)
    assert f"Cadence multiplier: {expected:g}×" in window.rate_preview.text()
    if preset == "a2_slow":
        assert selector.currentText() == "A2 Slow"
        assert "(knobs × 2^switches) / √2" in window.rate_preview.text()
    assert (
        f"every {schedule['decay_steps']:.8g} optimizer updates"
        in window.rate_preview.text()
    )
    assert not window.advanced_fields["lr_decay"].isEnabled()
    assert Workspace(workspace.directory).training_options()["rate_preset"] == preset
    selector.setCurrentIndex(selector.findData("a2_updates"))
    selector.activated.emit(selector.currentIndex())
    assert workspace_update_schedule(workspace)["cadence_multiplier"] == 1
    assert "original A2 cadence" in window.rate_preview.text()


@pytest.mark.parametrize("factor", [0, -1, True, "1", None, float("nan"), float("inf")])
def test_schedule_rejects_invalid_cadence_multipliers(tmp_path, factor):
    with pytest.raises(ValueError, match="positive finite number"):
        a2_update_schedule(
            {}, tmp_path, 16, 8192, 2, 0.004, 0.0004, cadence_multiplier=factor
        )


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
        ("a2_controls", "a2_slow"),
        ("a2_slow", "a2_controls"),
        ("manual", "a2_slow"),
        ("a2_slow", "manual"),
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
    window._attach(workspace)
    hyper = window.advanced_fields["hypernetwork_rate_preset"]
    hyper.setCurrentIndex(hyper.findData("a2_slow"))
    hyper.activated.emit(hyper.currentIndex())
    assert "Independent LR schedules" in window.rate_preview.text()
    assert "Hypernetwork: a2_slow" in window.rate_preview.text()
    assert (
        Workspace(workspace.directory).training_options()["hypernetwork_rate_preset"]
        == "a2_slow"
    )
    window.advanced_fields["hypernetwork_learning_rate"].setValue(0.0002)
    main = window.advanced_fields["rate_preset"]
    main.setCurrentIndex(main.findData("a2_controls"))
    main.activated.emit(main.currentIndex())
    assert window.advanced_fields["hypernetwork_learning_rate"].value() == 0.0002
    hyper.setCurrentIndex(hyper.findData("manual"))
    hyper.activated.emit(hyper.currentIndex())
    assert window.advanced_fields["hypernetwork_lr_decay"].isEnabled()
    assert not window.advanced_fields["lr_decay"].isEnabled()


def test_real_training_restores_two_lightning_scheduler_states(workspace, monkeypatch):
    import torch
    from nam.capture.project import atomic_write_json
    from nam.train.parametric import _ParametricLightningModule
    from tests.paranam.test_training import tiny_job

    job = tiny_job(workspace, monkeypatch)
    spec = read_json(job / "spec.json")
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
    assert training.make_job(workspace, resume=True) == job
    training.run_job(job)
    assert training.job_progress(job)["status"] == "completed"
    final = torch.load(job / "resume-0.ckpt", map_location="cpu", weights_only=False)
    groups = final["optimizer_states"][0]["param_groups"]
    assert groups[0]["lr"] == pytest.approx(0.004 * 0.9**2)
    assert groups[1]["lr"] == pytest.approx(0.0004 * 0.8)
    assert [state["last_epoch"] for state in final["lr_schedulers"]] == [2, 2]

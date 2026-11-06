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
    assert schedule["cadence_multiplier"] == pytest.approx(expected)
    if preset == "a2_slow":
        assert selector.currentText() == "A2 Slow"
        assert "A2 Slow:" in selector.toolTip()
    assert (
        f"every {schedule['decay_steps']:.8g} optimizer updates"
        in window.rate_preview.text()
    )
    assert window.advanced_fields["lr_decay"].isEnabled()
    assert Workspace(workspace.directory).training_options()["rate_preset"] == preset
    selector.setCurrentIndex(selector.findData("a2_updates"))
    selector.activated.emit(selector.currentIndex())
    assert workspace_update_schedule(workspace)["cadence_multiplier"] == 1
    assert "Audio: A2 Cadence" in window.rate_preview.text()


@pytest.mark.parametrize("factor", [0, -1, True, "1", None, float("nan"), float("inf")])
def test_schedule_rejects_invalid_cadence_multipliers(tmp_path, factor):
    with pytest.raises(ValueError, match="positive finite number"):
        a2_update_schedule(
            {}, tmp_path, 16, 8192, 2, 0.004, 0.0004, cadence_multiplier=factor
        )

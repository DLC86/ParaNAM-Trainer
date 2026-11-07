import math

import pytest
import torch

from nam.capture.params import KnobSpec, SwitchSpec, capture_recommendations
from nam.train._lr_scheduler import GroupBudgetCosineLR, GroupUpdateCadenceLR
from nam.train.parametric import _EpochMetrics
from paranam import training
from paranam.workspace import Workspace, read_json


def optimizer():
    return torch.optim.Adam(
        [
            dict(params=[torch.nn.Parameter(torch.zeros(1))], lr=0.004),
            dict(params=[torch.nn.Parameter(torch.zeros(1))], lr=0.001),
        ]
    )


@pytest.mark.parametrize("other_cosine", [False, True])
def test_cosine_closed_form_warmup_floor_and_resume_at_every_phase(other_cosine):
    opt = optimizer()
    audio = GroupBudgetCosineLR(opt, 0, 1000)
    hyper = (
        GroupBudgetCosineLR(opt, 1, 1000)
        if other_cosine
        else GroupUpdateCadenceLR(opt, 1, 62, 0.994)
    )
    assert opt.param_groups[0]["lr"] == 0
    assert opt.param_groups[1]["lr"] == (0 if other_cosine else 0.001)
    for step in range(1, 1004):
        opt.step()
        audio.step()
        hyper.step()
        t = min(step, 1000)
        factor = (
            t / 20
            if t < 20
            else 0.01 + 0.99 * (1 + math.cos(math.pi * (t - 20) / 980)) / 2
        )
        assert opt.param_groups[0]["lr"] == pytest.approx(0.004 * factor, rel=1e-14)
        expected_hyper = (
            0.001 * factor if other_cosine else 0.001 * 0.994 ** (step // 62)
        )
        assert opt.param_groups[1]["lr"] == pytest.approx(expected_hyper, rel=1e-14)
        if step in (1, 19, 20, 21, 62, 499, 999):
            opt_state, audio_state, hyper_state = (
                opt.state_dict(),
                audio.state_dict(),
                hyper.state_dict(),
            )
            opt = optimizer()
            audio = GroupBudgetCosineLR(opt, 0, 10)
            hyper = (
                GroupBudgetCosineLR(opt, 1, 10)
                if other_cosine
                else GroupUpdateCadenceLR(opt, 1, 2)
            )
            opt.load_state_dict(opt_state)
            audio.load_state_dict(audio_state)
            hyper.load_state_dict(hyper_state)
            assert audio.total_steps == 1000 and audio.last_epoch == step
    assert opt.param_groups[0]["lr"] == 0.004 * 0.01


@pytest.mark.parametrize(
    "knobs,expected", [(1, 8), (2, 8), (3, 12), (5, 20), (6, 24), (9, 24)]
)
def test_validation_recommendations_per_joint_switch_state(knobs, expected):
    controls = [KnobSpec(f"K{i}", 0, 10) for i in range(knobs)]
    controls += [
        SwitchSpec("A", ("Off", "On")),
        SwitchSpec("B", ("One", "Two", "Three")),
    ]
    assert capture_recommendations(controls) == (15 * knobs * 6, expected * 6, 6)
    assert capture_recommendations([SwitchSpec("A", ("Off", "On"))]) == (2, 0, 2)


def test_validation_default_does_not_overwrite_saved_plans(tmp_path):
    ws = Workspace.create(tmp_path / "Rig", "Rig")
    assert ws.preferences["n_validation"] == 12
    ws.preferences["n_validation"] = 6
    ws.save()
    assert Workspace(ws.directory).preferences["n_validation"] == 6


def test_worst_is_per_complete_setting_and_does_not_weight_loud_captures():
    metrics = _EpochMetrics()
    # Quiet setting has ESR=1, loud setting ESR=.01. Pooling across settings
    # would hide the quiet failure; the maximum batch ESR would also be wrong.
    metrics.update(torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0, 1.0]]), {}, 0)
    metrics.update(torch.tensor([[-1.0, 1.0]]), torch.tensor([[1.0, 1.0]]), {}, 0)
    metrics.update(torch.tensor([[9.0, 9.0]]), torch.tensor([[10.0, 10.0]]), {}, 1)
    assert metrics.compute()["ESR"] == pytest.approx(0.505)
    assert metrics.worst_esr() == 1
    ungrouped = _EpochMetrics()
    ungrouped.update(torch.ones(1, 2), torch.ones(1, 2), {})
    assert ungrouped.worst_esr() is None


@pytest.mark.parametrize(
    "audio,hyper",
    [
        ("cosine_budget", "manual"),
        ("a2_updates", "cosine_budget"),
        ("cosine_budget", "cosine_budget"),
    ],
)
def test_cosine_job_budget_matches_loader_accounting_and_ui(
    workspace, window, audio, hyper
):
    from tests.paranam.test_workflow055 import ready_audio
    from paranam.recipes import workspace_group_schedules

    ready_audio(workspace)
    workspace.preferences["advanced_parametric"] = dict(
        rate_preset=audio, hypernetwork_rate_preset=hyper
    )
    window._attach(workspace)
    for key in ("rate_preset", "hypernetwork_rate_preset"):
        assert window.advanced_fields[key].findData("cosine_budget") >= 0
    assert "Frozen budget:" in window.rate_preview.text()
    assert window.advanced_fields["lr_decay"].isEnabled() == (audio != "cosine_budget")
    assert window.advanced_fields["hypernetwork_lr_decay"].isEnabled() == (
        hyper != "cosine_budget"
    )
    spec = read_json(training.make_job(workspace) / "spec.json")
    schedules = workspace_group_schedules(workspace)
    for group, preset in [("audio", audio), ("hypernetwork", hyper)]:
        if preset == "cosine_budget":
            schedule = spec["model"]["lr_scheduler"]["kwargs"][group]
            assert (
                schedule["total_steps"]
                == schedules[group]["steps_per_epoch"] * workspace.preferences["epochs"]
            )
            assert schedule["kind"] == "cosine_budget"


def test_real_cosine_pause_resume_and_worst_metrics(workspace, monkeypatch, capsys):
    from tests.paranam.test_training import tiny_job
    from nam.capture.project import atomic_write_json
    from nam.train.parametric import _ParametricLightningModule

    job = tiny_job(workspace, monkeypatch)
    spec = read_json(job / "spec.json")
    spec["model"]["lr_scheduler"] = {
        "class": "IndependentGroupLR",
        "kwargs": {
            "audio": dict(kind="cosine_budget", interval="step", total_steps=2),
            "hypernetwork": dict(kind="cosine_budget", interval="step", total_steps=2),
        },
    }
    atomic_write_json(job / "spec.json", spec)
    original = _ParametricLightningModule.training_step

    def pause(module, *args, **kwargs):
        if module.current_epoch == 1:
            training.command(job, "pause")
        return original(module, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(_ParametricLightningModule, "training_step", pause)
        training.run_job(job)
    assert training.job_progress(job)["status"] == "paused"
    paused = torch.load(job / "resume-0.ckpt", map_location="cpu", weights_only=False)
    assert [s["last_epoch"] for s in paused["lr_schedulers"]] == [1, 1]
    training.make_job(workspace, resume=True)
    training.run_job(job)
    result = training.job_progress(job)
    assert result["status"] == "completed"
    assert result["esr_worst_full"] == pytest.approx(result["esr_full"])
    assert result["esr_worst_lite"] == pytest.approx(result["esr_lite"])
    assert result["esr_history"][-1]["worst_full"] == result["esr_worst_full"]
    assert "ESR_worst" in capsys.readouterr().out
    final = torch.load(job / "resume-0.ckpt", map_location="cpu", weights_only=False)
    assert [s["last_epoch"] for s in final["lr_schedulers"]] == [2, 2]
    rates = [g["lr"] for g in final["optimizer_states"][0]["param_groups"]]
    assert rates == pytest.approx([0.004 * 0.01, 0.0004 * 0.01])


def test_beta_publication_keeps_draft(monkeypatch):
    import json
    import runpy
    import subprocess
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from paranam import __version__

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("GH_REPO", "DLC86/ParaNAM-Trainer")
    monkeypatch.setenv("RELEASE_ID", "123")
    monkeypatch.setenv("GITHUB_SHA", "source")
    monkeypatch.setenv("KEEP_DRAFT", "true")
    release = dict(
        draft=True,
        prerelease=True,
        target_commitish="source",
        tag_name="v0.6.3-beta",
        assets=[
            dict(
                name=f"ParaNAM-Trainer-{__version__}-" + suffix,
                state="uploaded",
                size=100,
            )
            for suffix in (
                "Windows-x64-Setup.exe",
                "Linux-amd64.deb",
                "Linux-x86_64.tar.gz",
                "macOS-arm64.dmg",
                "macOS-x86_64.dmg",
            )
        ],
    )
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda args, **kw: json.dumps(
            release
            if args[-1].endswith("/123")
            else dict(full_name="DLC86/ParaNAM-Trainer")
        ),
    )
    edits = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: edits.append(a))
    monkeypatch.setitem(sys.modules, "build", SimpleNamespace(VERSION=__version__))
    monkeypatch.setattr(sys, "argv", ["publish_native.py", "finalize"])
    runpy.run_path(str(root / "packaging/publish_native.py"), run_name="__main__")
    assert not edits

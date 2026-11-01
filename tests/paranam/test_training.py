import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nam.capture.params import KnobSpec, SwitchSpec
from nam.capture.project import atomic_write_json
from nam.data import np_to_wav
from paranam import training
from paranam.workspace import read_json


def ready(workspace):
    for entry in workspace.project.entries:
        target = workspace.directory / entry.y_path
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"recorded capture placeholder")
        entry.status = "captured"
        entry.delay = 0
    workspace.save()


def tiny_job(workspace, monkeypatch):
    ready(workspace)
    workspace.project.input_level_dbu, workspace.project.output_level_dbu = 18, 12
    workspace.save()
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    x = (0.1 * np.sin(np.arange(6500) * 0.03)).astype(np.float32)
    np_to_wav(x, workspace.directory / "test-input.wav", rate=48000)
    np_to_wav(np.tanh(2 * x), workspace.directory / "test-output.wav", rate=48000)
    spec["data"] = {
        "type": "parametric",
        "common": {
            "x_path": "test-input.wav",
            "y_path": "test-output.wav",
            "delay": 0,
            "ny": 32,
            "require_input_pre_silence": None,
        },
        "train": [{"params": {"Gain": 3.0}}],
        "validation": [{"params": {"Gain": 7.0}}],
    }
    spec["model"]["net"]["config"]["hypernet"].update(
        mode="low_rank", rank=2, hidden_sizes=[4]
    )
    spec["model"]["loss"] = {"val_loss": "esr"}
    spec["learning"]["trainer"].update(limit_train_batches=1, limit_val_batches=1)
    atomic_write_json(job / "spec.json", spec)
    monkeypatch.setattr(
        "nam.models.base._Base._metadata_loudness", lambda *a, **kw: -20.0
    )
    monkeypatch.setattr("nam.models.base._Base._metadata_gain", lambda *a, **kw: 0.5)
    # run_job changes its worker process's cwd. Restore the test runner's cwd afterwards.
    monkeypatch.chdir(workspace.directory)
    return job


@pytest.mark.parametrize("mode,members", [("normal", 1), ("active_learning", 2)])
def test_both_training_methods_only_build_nested_hyper_a2(workspace, mode, members):
    ready(workspace)
    workspace.preferences["mode"] = mode
    workspace.save()
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    assert spec["members"] == members
    assert spec["model"]["net"]["name"] == "HyperA2"
    assert spec["model"]["net"]["config"]["size"] == "full"
    assert spec["learning"]["torch_compile"]["enabled"] is False
    assert spec["learning"]["trainer"]["enable_progress_bar"] is True


@pytest.mark.parametrize("epochs", [1, 2])
def test_pause_resume_real_training_preserves_optimizer_and_exports_both_sizes(
    workspace, monkeypatch, epochs, capsys
):
    workspace.preferences["epochs"] = epochs
    workspace.save()
    job = tiny_job(workspace, monkeypatch)
    training.command(job, "pause")
    training.run_job(job)
    progress = training.job_progress(job)
    assert progress["status"] == "paused"
    assert progress["completed_epoch"] == 1
    assert progress["best_esr"] is not None
    checkpoint = torch.load(
        job / "resume-0.ckpt", map_location="cpu", weights_only=False
    )
    assert checkpoint["optimizer_states"] and checkpoint["lr_schedulers"]
    assert checkpoint["global_step"] > 0
    assert training.job_can_resume(workspace, job)
    assert training.make_job(workspace, resume=True) == job
    training.run_job(job)
    progress = training.job_progress(job)
    assert progress["status"] == "completed", progress
    final = read_json(Path(progress["export"]))
    assert final["architecture"] == "HyperA2"
    assert final["config"]["size"] == "full"
    assert [s["name"] for s in final["config"]["submodels"]] == ["lite", "full"]
    assert final["metadata"]["input_level_dbu"] == 18
    assert final["metadata"]["output_level_dbu"] == 12
    captured = capsys.readouterr()
    assert (
        "Epoch" in captured.out + captured.err and "100%" in captured.out + captured.err
    )
    resumed = torch.load(job / "resume-0.ckpt", map_location="cpu", weights_only=False)
    assert resumed["global_step"] >= checkpoint["global_step"]
    if epochs > 1:
        assert resumed["global_step"] > checkpoint["global_step"]


def test_stop_does_not_publish_an_incomplete_model(workspace, monkeypatch):
    job = tiny_job(workspace, monkeypatch)
    training.command(job, "stop")
    training.run_job(job)
    assert training.job_progress(job)["status"] == "stopped"
    assert not (workspace.directory / "models").exists()
    assert all(e.status == "captured" for e in workspace.project.entries)


def test_changed_settings_cannot_resume_an_old_checkpoint(workspace):
    ready(workspace)
    job = training.make_job(workspace)
    (job / "resume-0.ckpt").write_bytes(b"checkpoint")
    assert training.job_can_resume(workspace, job)
    workspace.preferences["epochs"] += 1
    assert not training.job_can_resume(workspace, job)


def test_real_active_learning_round_exports_nested_models_and_returns_to_capture(
    workspace, monkeypatch
):
    from paranam.workspace import Workspace

    workspace.preferences.update(
        mode="active_learning", epochs=1, rounds=2, additions=1
    )
    workspace.save()
    job = tiny_job(workspace, monkeypatch)
    training.run_job(job)
    progress = training.job_progress(job)
    assert progress["status"] == "needs_captures", progress
    assert progress["additions"] == 1
    for member in range(2):
        model = read_json(job / f"member-{member}" / "model_parametric.nam")
        assert model["architecture"] == "HyperA2"
        assert len(model["config"]["submodels"]) == 2
    reopened = Workspace(workspace.directory)
    assert len(reopened.project.pending_entries()) == 1
    assert reopened.validation_matches()
    assert not reopened.plan_changed()
    assert not reopened.training_ready()


def test_active_learning_excludes_validation_and_all_existing_settings(
    workspace, monkeypatch
):
    workspace.draft_controls = [
        KnobSpec("Gain", 0, 10).to_dict(),
        SwitchSpec("Mode", ("Clean", "Crunch", "Lead")).to_dict(),
        SwitchSpec("Bright", ("Off", "On")).to_dict(),
    ]
    workspace.preferences.update(
        n_train=12, n_validation=6, mode="active_learning", additions=5
    )
    workspace.generate_plan()
    ready(workspace)
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    previous = {tuple(e.params.values()) for e in workspace.project.entries}
    validation = [
        e.model_dump() for e in workspace.project.entries_for_split("validation")
    ]
    vectors = []

    class Member:
        def __init__(self, factor):
            self.factor = factor

        def eval(self):
            return self

        def set_size(self, size):
            self.size = size

        def __call__(self, signal, raw):
            vectors.append(raw.tolist())
            return signal * (1 + self.factor * (raw[0] + raw[1] * 2 + raw[2] * 3))

    def load(path, **kwargs):
        return Member(1 + int("resume-1" in str(path)))

    monkeypatch.setattr(
        "nam.train.parametric._ParametricLightningModule.load_from_checkpoint", load
    )
    monkeypatch.setattr("nam.train.parametric._get_parametric_net", lambda model: model)
    entries = training._propose_captures(workspace, job, spec, {"elapsed": 0})
    assert len(entries) == 5
    assert not previous & {tuple(e.params.values()) for e in entries}
    assert len({e.y_path for e in workspace.project.entries}) == len(
        workspace.project.entries
    )
    assert [
        e.model_dump() for e in workspace.project.entries_for_split("validation")
    ] == validation
    assert all(p[1] in (0, 1, 2) and p[2] in (0, 1) for p in vectors)
    assert workspace.validation_matches()

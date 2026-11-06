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
        options = workspace.training_options()
        scheduled = options.get("rate_preset", "manual") != "manual" or options.get(
            "hypernetwork_rate_preset", "follow"
        ) not in ("manual", "follow")
        target.unlink(missing_ok=True)
        if scheduled:
            import os, shutil

            source = workspace.directory / workspace.project.input_for_split(
                entry.split
            )
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        else:
            target.write_bytes(b"recorded capture placeholder")
        entry.status = "captured"
        entry.delay = 0
    workspace.save()


@pytest.mark.parametrize("capture_type", ["static", "parametric"])
@pytest.mark.parametrize("loopback", [False, True])
def test_training_retains_capture_safety_margin_once(workspace, capture_type, loopback):
    from nam.capture.latency import BlipPreamble, measure_delay
    from nam.capture.session import CaptureSession, ALIGNMENT_LEAD_SAMPLES
    from nam.capture.project import QAModel

    workspace.preferences["capture_type"] = capture_type
    workspace.generate_plan()
    ready(workspace)
    preamble = BlipPreamble(48000)
    measured = measure_delay(
        np.pad(preamble.render(), (480, 480)), preamble.as_played()
    )
    assert measured.safety_factor == 1
    assert measured.delay == 479  # Official calibrator: measured onset minus one.
    label, shift = CaptureSession._alignment(measured, loopback, ALIGNMENT_LEAD_SAMPLES)
    assert 480 - (label - shift) >= 1
    for entry in workspace.project.entries:
        entry.delay = label
        entry.qa = QAModel(subsample_shift=shift or None)
    workspace.save()
    spec = read_json(training.make_job(workspace) / "spec.json")
    for split in ("train", "validation"):
        assert all(item["delay"] == label for item in spec["data"][split])
    # Rebuilding the training job does not subtract another sample or alter WAVs.
    assert all(entry.delay == label for entry in workspace.project.entries)


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
        "train_evaluation": [{"params": {"Gain": 3.0}, "ny": None}],
    }
    spec["model"]["net"]["config"]["hypernet"].update(
        mode="low_rank", rank=2, hidden_sizes=[4]
    )
    spec["model"]["loss"] = {"val_loss": "esr", "training_recipe": "a2"}
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
    assert spec["learning"]["trainer"]["enable_progress_bar"] is False


@pytest.mark.parametrize("epochs", [1, 2])
@pytest.mark.parametrize("optimizer", ["Adam", "AdamW"])
def test_pause_resume_real_training_preserves_optimizer_and_exports_both_sizes(
    workspace, monkeypatch, epochs, capsys, optimizer
):
    workspace.preferences["advanced_parametric"] = {"optimizer": optimizer}
    workspace.preferences["epochs"] = epochs
    workspace.save()
    job = tiny_job(workspace, monkeypatch)
    training.command(job, "pause")
    training.run_job(job)
    progress = training.job_progress(job)
    assert progress["status"] == "paused"
    assert progress["completed_epoch"] == 0
    assert progress["restart_epoch"] == 1
    checkpoint = torch.load(
        job / "resume-0.ckpt", map_location="cpu", weights_only=False
    )
    assert checkpoint["optimizer_states"] and checkpoint["lr_schedulers"]
    assert checkpoint["optimizer_class"] == optimizer
    assert checkpoint["global_step"] == 0
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
    summary = final["metadata"]["training"]
    assert summary["validation_reference"] == "full"
    assert summary["validation_esr"] == summary["validation_esr_full"]
    assert summary["validation_esr_lite"] is not None
    assert summary["validation_esr"] == pytest.approx(progress["best_esr"])
    captured = capsys.readouterr()
    assert "Validation ESR | Lite=" in captured.out
    assert "Full (checkpoint)=" in captured.out and "Mean=" in captured.out
    assert "Train ESR | Lite=" in captured.out
    assert progress["train_esr_full"] is not None
    assert progress["best_train_esr"] <= progress["train_esr_full"]
    assert progress["esr_history"][-1]["train_lite"] is not None
    assert "Epoch 1/" in captured.out
    assert progress["esr_lite"] is not None and progress["esr_full"] is not None
    assert progress["esr_mean"] == pytest.approx(
        (progress["esr_lite"] + progress["esr_full"]) / 2
    )
    assert (
        "Epoch" in captured.out + captured.err and "100%" in captured.out + captured.err
    )
    resumed = torch.load(job / "resume-0.ckpt", map_location="cpu", weights_only=False)
    assert resumed["best_train_esr_full"] == progress["best_train_esr"]
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


def test_resume_legacy_adamw_checkpoint_starts_full_ranking(workspace, monkeypatch):
    from nam.train.parametric import _ParametricLightningModule

    workspace.preferences["advanced_parametric"] = {
        "optimizer": "AdamW",
        "fit_mrstft": False,
        "ny": 32,
    }
    job = tiny_job(workspace, monkeypatch)
    spec = read_json(job / "spec.json")
    spec["learning"].pop("checkpoint_metric")  # Legacy checkpoint monitor.
    spec["learning"].pop("separate_size_checkpoints", None)
    spec["model"]["loss"].pop("training_recipe", None)
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
    checkpoint_path = job / "resume-0.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint.pop("optimizer_class")
    for state in checkpoint["callbacks"].values():
        if "best_esr" in state:
            state.pop("validation_reference", None)
            state.pop("best_validation", None)
            state.pop("last_validation", None)
    torch.save(checkpoint, checkpoint_path)
    spec["model"]["optimizer"].pop("class")
    spec["preferences"]["advanced_training"].pop("optimizer")
    spec["fingerprint"] = workspace.fingerprint(include_optimizer=False)
    atomic_write_json(job / "spec.json", spec)
    assert training.job_can_resume(workspace, job)
    assert training.make_job(workspace, resume=True) == job
    training.run_job(job)
    progress = training.job_progress(job)
    assert progress["status"] == "completed"
    metadata = read_json(Path(progress["export"]))["metadata"]["training"]
    assert metadata["optimizer"] == "AdamW"
    assert metadata["epoch"] == 2
    assert metadata["validation_reference"] == "full"


@pytest.mark.parametrize("extension", [".nam", ".pnam"])
def test_completed_parametric_member_exports_pnam_without_losing_metadata(
    workspace, monkeypatch, extension
):
    job = tiny_job(workspace, monkeypatch)
    member = job / "member-0"
    member.mkdir()
    model = {
        "architecture": "HyperA2",
        "config": {"submodels": [{"name": "lite"}, {"name": "full"}]},
        "weights": [1.25, -0.5],
        "metadata": {
            "input_level_dbu": 18.125,
            "output_level_dbu": 12.625,
            "custom": {"labels": ["Clean", "Lead"], "keep": True},
        },
    }
    original = member / f"model_parametric{extension}"
    atomic_write_json(original, model)
    torch.save({"callbacks": {"quality": {"best_esr": 0.01}}}, job / "resume-0.ckpt")
    progress = training.job_progress(job)
    progress["completed_members"] = [0]
    atomic_write_json(job / "progress.json", progress)

    def unexpected_training(*args, **kwargs):
        raise AssertionError("A completed member must not be trained again")

    monkeypatch.setattr("nam.train.parametric.main", unexpected_training)
    training.run_job(job)
    progress = training.job_progress(job)
    assert progress["status"] == "completed", progress
    exported = Path(progress["export"])
    assert exported.suffix == ".pnam"
    assert read_json(exported) == model
    assert original.exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_mid_epoch_pause_restarts_only_the_incomplete_epoch(
    workspace, monkeypatch, legacy
):
    from nam.train.parametric import _ParametricLightningModule

    job = tiny_job(workspace, monkeypatch)
    original = _ParametricLightningModule.training_step
    calls, requested = [], []

    def step(module, batch, index):
        calls.append((module.current_epoch, index))
        result = original(module, batch, index)
        if module.current_epoch == 1 and not requested:
            requested.append(True)
            training.command(job, "pause")
        return result

    monkeypatch.setattr(_ParametricLightningModule, "training_step", step)
    training.run_job(job)
    assert training.job_progress(job)["status"] == "paused"
    assert training.job_progress(job)["completed_epoch"] == 1
    checkpoint = torch.load(
        job / "resume-0.ckpt", map_location="cpu", weights_only=False
    )
    assert checkpoint["global_step"] == 1
    if not legacy:
        import hashlib

        before = hashlib.sha256((job / "resume-0.ckpt").read_bytes()).digest()
        snapshot = read_json(
            Path(training.export_saved_best(job, workspace.directory / "Paused.pnam"))
        )
        assert snapshot["architecture"] == "HyperA2"
        assert snapshot["metadata"]["input_level_dbu"] == 18
        assert snapshot["metadata"]["training"]["validation_esr_full"] is not None
        assert snapshot["metadata"]["training"]["validation_esr_lite"] is not None
        assert hashlib.sha256((job / "resume-0.ckpt").read_bytes()).digest() == before
    if legacy:
        (job / "resume-0.ckpt").unlink()
        (job / "resume-0.json").unlink()
        assert training.job_can_resume(workspace, job)
    before = len(calls)
    training.make_job(workspace, resume=True)
    training.run_job(job)
    assert calls[before:] == [(1, 0)]
    assert training.job_progress(job)["status"] == "completed"
    final = torch.load(job / "resume-0.ckpt", map_location="cpu", weights_only=False)
    assert final["global_step"] == 2
    assert final["lr_schedulers"][0]["last_epoch"] == 2


def test_partial_or_corrupt_legacy_checkpoints_are_not_offered_for_resume(tmp_path):
    folder = tmp_path / "member-0/checkpoints"
    folder.mkdir(parents=True)
    torch.save(
        dict(
            epoch=0,
            global_step=3,
            optimizer_states=[{}],
            loops={"fit_loop": {"epoch_progress": {"current": {"processed": 0}}}},
        ),
        folder / "partial.ckpt",
    )
    (folder / "corrupt.ckpt").write_bytes(b"unfinished checkpoint")
    assert training.legacy_checkpoint(tmp_path, 0) is None


def test_error_can_resume_from_last_complete_epoch_with_a_smaller_batch(
    workspace, monkeypatch
):
    from nam.train.parametric import _ParametricLightningModule

    workspace.preferences["batch_size"] = 2
    workspace.save()
    job = tiny_job(workspace, monkeypatch)
    original = _ParametricLightningModule.training_step
    failed = []

    def step(module, batch, index):
        if module.current_epoch == 1 and not failed:
            failed.append(True)
            raise RuntimeError("simulated GPU memory error")
        return original(module, batch, index)

    monkeypatch.setattr(_ParametricLightningModule, "training_step", step)
    with pytest.raises(RuntimeError, match="simulated GPU"):
        training.run_job(job)
    progress = training.job_progress(job)
    assert progress["status"] == "failed" and progress["resume_available"]
    assert progress["completed_epoch"] == 1 and progress["restart_epoch"] == 2
    workspace.preferences["batch_size"] = 1
    workspace.save()
    assert training.job_can_resume(workspace, job)
    assert training.make_job(workspace, resume=True) == job
    assert (
        read_json(job / "spec.json")["learning"]["train_dataloader"]["batch_size"] == 1
    )
    training.run_job(job)
    assert training.job_progress(job)["status"] == "completed"


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
        model = read_json(job / f"member-{member}" / "model_parametric.pnam")
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

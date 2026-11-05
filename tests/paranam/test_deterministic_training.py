import copy
import os
import shutil

import pytest
import torch
from torch.utils.data import TensorDataset

from nam.capture.project import atomic_write_json
from nam.train.parametric import _make_parametric_dataloader
from paranam import training
from paranam.training_environment import apply_backend_flags, prepare_worker_environment
from paranam.workspace import Workspace, read_json
from tests.paranam.test_training import ready, tiny_job


@pytest.fixture(autouse=True)
def restore_backend(monkeypatch):
    flags = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    monkeypatch.setenv(
        "CUBLAS_WORKSPACE_CONFIG", os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
    )
    yield
    torch.use_deterministic_algorithms(flags[0])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags[1:]


@pytest.mark.parametrize("kind", ["static", "parametric"])
@pytest.mark.parametrize(
    "benchmark,deterministic", [(True, False), (False, False), (True, True)]
)
def test_backend_options_reach_jobs_and_guard_resume(
    workspace, kind, benchmark, deterministic
):
    workspace.preferences["capture_type"] = kind
    workspace.preferences["advanced_" + kind] = dict(
        cudnn_benchmark=benchmark, deterministic_training=deterministic
    )
    workspace.generate_plan()
    ready(workspace)
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    assert spec["learning"]["trainer"]["benchmark"] is (benchmark and not deterministic)
    assert spec["learning"]["trainer"]["deterministic"] is deterministic
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(workspace, job)
    workspace.preferences["advanced_" + kind][
        "deterministic_training"
    ] = not deterministic
    assert not training.job_can_resume(workspace, job)


def test_ui_saves_determinism_and_forces_benchmark_off(window, workspace):
    window._attach(workspace)
    fields = window.advanced_fields
    assert fields["cudnn_benchmark"].isChecked()
    fields["deterministic_training"].setChecked(True)
    assert not fields["cudnn_benchmark"].isChecked()
    assert not fields["cudnn_benchmark"].isEnabled()
    reopened = Workspace(workspace.directory)
    assert reopened.training_options()["deterministic_training"] is True
    assert reopened.training_options()["cudnn_benchmark"] is False
    window._attach(reopened)
    assert not fields["cudnn_benchmark"].isEnabled()
    fields["deterministic_training"].setChecked(False)
    assert fields["cudnn_benchmark"].isEnabled()


def test_legacy_defaults_keep_fingerprint_and_resume(workspace):
    ready(workspace)
    original = workspace.fingerprint()
    workspace.preferences["advanced_parametric"].update(
        cudnn_benchmark=True, deterministic_training=False
    )
    assert workspace.fingerprint() == original
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    spec["learning"]["trainer"].pop("deterministic")
    atomic_write_json(job / "spec.json", spec)
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(workspace, job)


def test_backend_environment_and_strict_algorithms(tmp_path, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    spec = {"learning": {"trainer": {"deterministic": True, "benchmark": True}}}
    atomic_write_json(tmp_path / "spec.json", spec)
    prepare_worker_environment(tmp_path)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    apply_backend_flags(spec)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    spec["learning"]["trainer"] = {"benchmark": False}
    apply_backend_flags(spec)
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.benchmark


def test_worker_configures_cublas_before_gpu_probe(tmp_path, monkeypatch):
    import sys
    from paranam import __main__, gpu_runtime

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    atomic_write_json(
        tmp_path / "spec.json",
        {"learning": {"trainer": {"deterministic": True}}},
    )
    calls = []

    def activate():
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        calls.append("activate")

    def probe():
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        calls.append("probe")
        return True, ""

    monkeypatch.setattr(gpu_runtime, "activate", activate)
    monkeypatch.setattr(gpu_runtime, "can_compute", probe)
    monkeypatch.setattr(training, "run_job", lambda job: calls.append("train"))
    monkeypatch.setattr(
        sys, "argv", ["paranam", "--train-worker", str(tmp_path), "--gpu-runtime"]
    )
    assert __main__.main() == 0
    assert calls == ["activate", "probe", "train"]


@pytest.mark.parametrize("grouped", [False, True])
def test_batch_order_is_independent_of_model_rng_and_epoch_resume(grouped):
    dataset = TensorDataset(torch.arange(40))
    config = dict(batch_size=4, shuffle=True, sampler_seed=938, capture_grouped=grouped)
    # The actual grouped key is intentionally obtained from the module.
    from nam.train.parametric import _CAPTURE_GROUPED_KEY

    config[_CAPTURE_GROUPED_KEY] = config.pop("capture_grouped")

    def order(epoch):
        loader = _make_parametric_dataloader(dataset, config)
        sampler = loader.batch_sampler if grouped else loader.sampler
        sampler.set_epoch(epoch)
        before = torch.get_rng_state().clone()
        result = torch.cat([batch[0] for batch in loader]).tolist()
        assert torch.equal(before, torch.get_rng_state())
        return result

    expected = order(2)
    torch.rand(1234)  # Different architecture initialization must not affect order.
    assert order(2) == expected
    assert order(1) != expected


@pytest.mark.parametrize("kind", ["static", "parametric"])
def test_real_training_repeats_and_resumes_identically(
    workspace, monkeypatch, tmp_path, kind
):
    from nam.train.parametric import _ParametricLightningModule

    if kind == "parametric":
        workspace.preferences["advanced_parametric"]["deterministic_training"] = True
        job = tiny_job(workspace, monkeypatch)
        module_class = _ParametricLightningModule
    else:
        import numpy as np
        from nam.data import np_to_wav
        from nam.train.lightning_module import PackedLightningModule

        module_class = PackedLightningModule
        workspace.preferences.update(
            capture_type="static",
            static_batch_size=1,
            advanced_static={"deterministic_training": True},
        )
        workspace.generate_plan()
        x = (0.1 * np.sin(np.arange(6500) * 0.03)).astype(np.float32)
        np_to_wav(x, workspace.directory / "test-input.wav", rate=48000)
        for entry in workspace.project.entries:
            target = workspace.directory / entry.y_path
            target.parent.mkdir(exist_ok=True)
            np_to_wav(np.tanh(2 * x), target, rate=48000)
            entry.status, entry.delay = "captured", 0
        workspace.save()
        job = training.make_job(workspace)
        spec = read_json(job / "spec.json")
        spec["data"]["common"].update(
            x_path="test-input.wav", ny=32, require_input_pre_silence=None
        )
        for split in ("train", "validation", "train_evaluation"):
            for entry in spec["data"][split]:
                entry.update(
                    x_path="test-input.wav", ny=32, start_seconds=0, stop_seconds=None
                )
        spec["model"]["loss"] = {"val_loss": "esr"}
        spec["learning"]["trainer"]["limit_val_batches"] = 1
        atomic_write_json(job / "spec.json", spec)
        monkeypatch.setattr(
            "nam.models.base._Base._metadata_loudness", lambda *a, **kw: -20.0
        )
        monkeypatch.setattr(
            "nam.models.base._Base._metadata_gain", lambda *a, **kw: 0.5
        )
        monkeypatch.chdir(workspace.directory)
    spec = read_json(job / "spec.json")
    spec["learning"]["trainer"]["limit_train_batches"] = 2
    atomic_write_json(job / "spec.json", spec)
    pristine = tmp_path / "pristine"
    shutil.copytree(job, pristine)
    training.run_job(job)
    assert training.job_progress(job)["status"] == "completed"
    first = torch.load(job / "resume-0.ckpt", weights_only=False, map_location="cpu")
    history = copy.deepcopy(training.job_progress(job)["esr_history"])

    def reset():
        assert job.resolve().is_relative_to(tmp_path.resolve())
        shutil.rmtree(job)
        shutil.copytree(pristine, job)

    reset()
    training.run_job(job)
    second = torch.load(job / "resume-0.ckpt", weights_only=False, map_location="cpu")
    assert all(
        torch.equal(first["state_dict"][key], value)
        for key, value in second["state_dict"].items()
    )
    assert training.job_progress(job)["esr_history"] == history
    reset()
    original = module_class.training_step
    requested = []

    def step(module, batch, index):
        result = original(module, batch, index)
        if module.current_epoch == 1 and not requested:
            requested.append(True)
            training.command(job, "pause")
        return result

    monkeypatch.setattr(module_class, "training_step", step)
    training.run_job(job)
    assert training.job_progress(job)["status"] == "paused"
    training.make_job(workspace, resume=True)
    training.run_job(job)
    resumed = torch.load(job / "resume-0.ckpt", weights_only=False, map_location="cpu")
    assert all(
        torch.equal(first["state_dict"][key], value)
        for key, value in resumed["state_dict"].items()
    )
    assert resumed["global_step"] == first["global_step"]
    assert resumed["lr_schedulers"] == first["lr_schedulers"]
    assert training.job_progress(job)["esr_history"] == history

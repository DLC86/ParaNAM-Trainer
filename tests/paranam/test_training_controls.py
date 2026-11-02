from types import SimpleNamespace

import pytest
import torch

from nam.capture.project import atomic_write_json
from nam.train.lightning_module import LightningModule
from nam.train.parametric import _ParametricLightningModule
from paranam import training
from paranam.recipes import training_preset
from paranam.training_log import TrainingLog, epoch_line
from paranam.workspace import Workspace, read_json
from tests.paranam.test_advanced_training import ready
from tests.paranam.test_training import tiny_job


def test_log_keeps_all_epochs_and_replaces_live_updates_across_reads(tmp_path):
    path = tmp_path / "worker.log"
    old = "An earlier diagnostic message " * 1000 + "\n"
    path.write_bytes(old.encode())
    reader = TrainingLog(path)
    assert reader.read() == old
    for epoch in range(1, 126):
        with path.open("ab") as stream:
            stream.write(
                (
                    "\r" + epoch_line(epoch, 125, "Full + Lite", 0.1, "Training") + "\r"
                ).encode()
            )
        reader.read()
        with path.open("ab") as stream:
            stream.write(
                (
                    epoch_line(epoch, 125, "Full + Lite", 1, "Validation ESR") + "\n"
                ).encode()
            )
    text = reader.read()
    assert text.startswith(old)
    assert text.count("Epoch ") == 125
    assert "Training" not in text
    with path.open("ab") as stream:
        stream.write(b"warning\r")
    reader.read()
    with path.open("ab") as stream:
        stream.write(b"\n\x1b[31merror\x1b[0m\n\xc3")
    reader.read()
    with path.open("ab") as stream:
        stream.write(b"\xa8\n")
    assert reader.read().endswith("warning\nerror\nè\n")
    path.write_text("new log\n", encoding="utf-8")
    assert reader.read() == "new log\n"


@pytest.mark.parametrize("kind", ["static", "parametric"])
@pytest.mark.parametrize("enabled", [False, True])
def test_controls_reach_training_and_disabling_scan_avoids_work(
    workspace, monkeypatch, kind, enabled
):
    workspace.preferences["capture_type"] = kind
    workspace.preferences["advanced_" + kind] = dict(
        gradient_clipping=enabled,
        gradient_clip_val=0.75,
        check_nonfinite_gradients=enabled,
        hypernetwork_learning_rate=0.0007,
    )
    workspace.generate_plan()
    ready(workspace)
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    assert spec["model"]["check_nonfinite_gradients"] is enabled
    assert spec["learning"]["trainer"].get("gradient_clip_val") == (
        0.75 if enabled else None
    )
    module_class = (
        _ParametricLightningModule if kind == "parametric" else LightningModule
    )
    module = module_class.init_from_config(spec["model"])
    optimizer = module.configure_optimizers()["optimizer"]
    if kind == "parametric":
        assert optimizer.param_groups[-1]["lr"] == pytest.approx(0.0007)
    monkeypatch.setattr(module, "clip_gradients", lambda *args, **kwargs: None)
    parameter = next(module.parameters())
    parameter.grad = torch.full_like(parameter, float("nan"))
    if not enabled:
        monkeypatch.setattr(
            torch,
            "isfinite",
            lambda *args, **kwargs: pytest.fail("Disabled scan executed"),
        )
    module.configure_gradient_clipping(optimizer, gradient_clip_val=None)
    assert (parameter.grad is None) is enabled
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(workspace, job)
    workspace.preferences["advanced_" + kind]["check_nonfinite_gradients"] = not enabled
    assert not training.job_can_resume(workspace, job)


def test_presets_and_independent_hypernetwork_lr_are_saved(window, workspace):
    window._attach(workspace)
    selector = window.advanced_fields["hypernetwork_recipe"]
    selector.setCurrentIndex(selector.findData("HyperWaveNet"))
    window.apply_training_preset()
    assert window.batch_size.value() == 32
    assert window.advanced_fields["learning_rate"].value() == 0.002
    assert window.advanced_fields["hypernetwork_learning_rate"].value() == 0.002
    assert window.advanced_fields["optimizer"].currentData() == "AdamW"
    window.advanced_fields["hypernetwork_learning_rate"].setValue(0.0013)
    window.advanced_fields["learning_rate"].setValue(0.003)
    assert window.advanced_fields["hypernetwork_learning_rate"].value() == 0.0013
    reopened = Workspace(workspace.directory)
    assert reopened.training_options()["hypernetwork_learning_rate"] == 0.0013
    ready(reopened)
    spec = read_json(training.make_job(reopened) / "spec.json")
    hyper = spec["model"]["net"]["config"]["hypernet"]
    assert hyper["mode"] == "low_rank" and hyper["rank"] == 2
    assert hyper["selector"] == {"exclude_suffixes": ["_conv.weight"]}
    assert spec["model"]["loss"]["training_recipe"] == "legacy"
    configured = _ParametricLightningModule.init_from_config(
        spec["model"]
    ).configure_optimizers()
    assert configured["optimizer"].param_groups[-1]["lr"] == pytest.approx(0.0013)
    selector.setCurrentIndex(selector.findData("HyperA2"))
    window.apply_training_preset()
    assert window.batch_size.value() == 16
    assert workspace.training_options() == training_preset("HyperA2")


def test_043_resume_keeps_saved_recipe_when_controls_are_default(workspace):
    ready(workspace)
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    for key in (
        "gradient_clipping",
        "gradient_clip_val",
        "check_nonfinite_gradients",
        "hypernetwork_learning_rate",
        "hypernetwork_recipe",
    ):
        spec["preferences"]["advanced_training"].pop(key)
    spec["model"].pop("check_nonfinite_gradients")
    spec["fingerprint"] = workspace.fingerprint(include_gradient_safety=False)
    atomic_write_json(job / "spec.json", spec)
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(workspace, job)
    workspace.preferences["advanced_parametric"] = dict(
        hypernetwork_learning_rate=0.002
    )
    assert not training.job_can_resume(workspace, job)


@pytest.mark.parametrize("fail_export", [False, True])
def test_export_request_while_training_does_not_stop_training(
    workspace, monkeypatch, fail_export
):
    job = tiny_job(workspace, monkeypatch)
    target = workspace.directory / "intermediate.nam"
    if fail_export:

        def fail(*args, **kwargs):
            raise OSError("Test export failure")

        monkeypatch.setattr(training, "export_best_nam", fail)
    original = _ParametricLightningModule.training_step

    def request_at_second_epoch(module, *args, **kwargs):
        if module.current_epoch == 1:
            atomic_write_json(
                job / "export-request.json", dict(id="test", path=str(target))
            )
        return original(module, *args, **kwargs)

    monkeypatch.setattr(
        _ParametricLightningModule, "training_step", request_at_second_epoch
    )
    training.run_job(job)
    assert training.job_progress(job)["status"] == "completed"
    result = read_json(job / "export-result.json")
    if fail_export:
        assert result.get("ok") is False and "Test export failure" in result["error"]
        assert not target.exists()
        return
    assert result.get("ok"), result
    model = read_json(target)
    assert model["architecture"] == "SlimmableContainer"
    assert len(model["config"]["submodels"]) == 2
    assert model["metadata"]["training"]["intermediate"] is True
    assert model["metadata"]["training"]["epoch_full"] == 1
    assert model["metadata"]["input_level_dbu"] == 18
    history = training.job_progress(job)["esr_history"]
    assert [entry["epoch"] for entry in history] == [1, 2]


@pytest.mark.parametrize("kind", ["static", "parametric"])
def test_export_combines_independently_best_epochs_and_preserves_rng(
    workspace, monkeypatch, kind
):
    from nam.train.lightning_module import PackedLightningModule, PackedBestCheckpoint

    if kind == "static":
        from tests.paranam.test_static import static_workspace

        workspace = static_workspace(workspace)
        ready(workspace)
        job = training.make_job(workspace)
        monkeypatch.setattr(
            "nam.models.base._Base._metadata_loudness", lambda *a, **kw: -20.0
        )
        monkeypatch.setattr(
            "nam.models.base._Base._metadata_gain", lambda *a, **kw: 0.5
        )
    else:
        job = tiny_job(workspace, monkeypatch)
    spec = read_json(job / "spec.json")
    models, paths, expected = {}, {}, {}
    for index, size in enumerate(("full", "lite")):
        cls = PackedLightningModule if kind == "static" else _ParametricLightningModule
        models[size] = cls.init_from_config(spec["model"])
        models[size].net.sample_rate = 48000
        models[size].eval()
        checkpoint = dict(epoch=index * 2, state_dict=models[size].state_dict())
        models[size].on_save_checkpoint(checkpoint)
        paths[size] = job / f"best-{size}.ckpt"
        torch.save(checkpoint, paths[size])
        net = models[size].net
        (net if kind == "static" else net.bake(net.nominal_params)).export(
            job, basename=f"expected-{size}"
        )
        expected[size] = read_json(job / f"expected-{size}.nam")
    trainer = SimpleNamespace(
        checkpoint_callbacks=[
            SimpleNamespace(
                monitor=f"ESR_{size}",
                best_model_path=str(paths[size]),
                best_model_score=torch.tensor(0.01 if size == "full" else 0.02),
            )
            for size in ("full", "lite")
        ]
    )
    if kind == "static":
        packed = PackedBestCheckpoint()
        packed.best = {
            i: dict(
                checkpoint_path=str(paths[size]),
                best_metric=0.01 if size == "full" else 0.02,
            )
            for i, size in enumerate(("lite", "full"))
        }
        trainer.callbacks = [packed]
    before = torch.get_rng_state()
    target = job / "snapshot.nam"
    training.export_best_nam(trainer, spec, 0, target)
    assert torch.equal(before, torch.get_rng_state())
    exported = read_json(target)
    for index, size in enumerate(("lite", "full")):
        assert (
            exported["config"]["submodels"][index]["model"]["weights"]
            == expected[size]["config"]["submodels"][index]["model"]["weights"]
        )
    metadata = exported["metadata"]["training"]
    assert metadata["epoch_full"] == 1 and metadata["epoch_lite"] == 3


def test_esr_chart_handles_missing_and_nonfinite_scores(qapp):
    from paranam.esr_chart import ESRChart

    chart = ESRChart()
    chart.resize(640, 200)
    chart.set_history(
        [
            dict(epoch=1, full=0.03, lite=0.04),
            dict(epoch=2, full=float("inf"), lite=None),
            dict(epoch=3, full=0.02, lite=0.025),
        ]
    )
    assert not chart.grab().isNull()
    chart.set_history([])
    assert not chart.grab().isNull()

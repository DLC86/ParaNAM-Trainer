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
    target = workspace.directory / "intermediate.pnam"
    if fail_export:

        def fail(*args, **kwargs):
            raise OSError("Test export failure")

        monkeypatch.setattr(training, "export_best_model", fail)
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
    assert model["architecture"] == "HyperA2"
    assert model["config"]["params"][0]["name"] == "Gain"
    assert not target.with_suffix(".nam").exists()
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
    from nam.models.parametric import HyperA2

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
    spec["model"].setdefault("metadata", {})[
        "training"
    ] = "Keep my custom training metadata"
    if kind == "static":
        spec["static_captures"][0].setdefault("metadata", {})[
            "training"
        ] = "Keep my custom training metadata"
    models, paths, expected = {}, {}, {}
    for index, size in enumerate(("full", "lite")):
        cls = PackedLightningModule if kind == "static" else _ParametricLightningModule
        models[size] = cls.init_from_config(spec["model"])
        models[size].net.sample_rate = 48000
        models[size].eval()
        if kind == "parametric":
            with torch.no_grad():
                for branch in models[size].net._branches.values():
                    branch._hypernet._final.weight.normal_(0, 0.003)
        checkpoint = dict(epoch=index * 2, state_dict=models[size].state_dict())
        models[size].on_save_checkpoint(checkpoint)
        paths[size] = job / f"best-{size}.ckpt"
        torch.save(checkpoint, paths[size])
        net = models[size].net
        if kind == "static":
            net.export(job, basename=f"expected-{size}")
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
    target = job / ("snapshot.nam" if kind == "static" else "snapshot.pnam")
    training.export_best_model(trainer, spec, 0, target)
    assert torch.equal(before, torch.get_rng_state())
    exported = read_json(target)
    if kind == "static":
        assert exported["architecture"] == "SlimmableContainer"
        for index, size in enumerate(("lite", "full")):
            assert (
                exported["config"]["submodels"][index]["model"]["weights"]
                == expected[size]["config"]["submodels"][index]["model"]["weights"]
            )
    else:
        assert exported["architecture"] == "HyperA2"
        restored = HyperA2.init_from_config(exported["config"]).eval()
        restored.import_weights(exported["weights"])
        assert restored.param_specs == models["full"].net.param_specs
        signal = 0.1 * torch.randn(restored.receptive_field + 15)
        with torch.no_grad():
            for size in ("full", "lite"):
                restored.set_size(size)
                source = models[size].net.branch(size)
                torch.testing.assert_close(
                    torch.from_numpy(restored.branch(size)._export_weights()),
                    torch.from_numpy(source._export_weights()),
                    rtol=0,
                    atol=0,
                )
                outputs = []
                for setting in (2.0, 8.0):
                    params = torch.tensor([setting])
                    output = restored(signal, params, pad_start=False)
                    torch.testing.assert_close(
                        output, source(signal, params, pad_start=False)
                    )
                    outputs.append(output)
                assert not torch.equal(
                    *outputs
                ), "Exported knobs no longer affect the audio"
        assert not target.with_suffix(".nam").exists()
    assert exported["metadata"]["training"] == "Keep my custom training metadata"
    metadata = exported["metadata"]["paranam_training"]
    assert metadata["intermediate"] is True
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


@pytest.mark.parametrize(
    "kind,extension", [("parametric", ".pnam"), ("static", ".nam")]
)
def test_export_button_selects_the_training_model_format(
    window, workspace, monkeypatch, kind, extension
):
    if kind == "static":
        from tests.paranam.test_static import static_workspace

        workspace = static_workspace(workspace)
    ready(workspace)
    window._attach(workspace)
    job = training.make_job(workspace)
    window.job = job
    window.queue_controller.store.register_legacy(workspace)
    entry = next(
        e for e in window.queue_controller.store.entries() if e["job"] == str(job)
    )
    window.selected_job_id = entry["id"]
    window.queue_controller.active_id = entry["id"]
    dialogs = []

    def choose_file(*args):
        dialogs.append(args)
        return str(workspace.directory / "chosen.wrong"), ""

    monkeypatch.setattr("paranam.wizard.QFileDialog.getSaveFileName", choose_file)
    window.process = object()
    try:
        window.update_enabled()
        assert window.export_best_button.text() == (
            "Export best PNAM" if extension == ".pnam" else "Export best NAM"
        )
        window.export_best_model()
        assert str(dialogs[0][2]).endswith(extension)
        assert "*" + extension in dialogs[0][3]
        request = read_json(job / "export-request.json")
        assert request["path"].endswith("chosen" + extension)
    finally:
        window.process = None
        window.queue_controller.active_id = None


@pytest.mark.parametrize("mode", ["full", "low_rank"])
@pytest.mark.parametrize("condition_dilated", [False, True])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_weight_generation_and_loss_controls_are_independent(
    workspace, mode, condition_dilated, reduction
):
    torch.set_num_threads(1)
    workspace.preferences["advanced_parametric"] = dict(
        hypernetwork_recipe="HyperA2",
        hypernetwork_mode=mode,
        hypernetwork_rank=3,
        condition_dilated_convolutions=condition_dilated,
        branch_loss_reduction=reduction,
        fit_mrstft=False,
    )
    ready(workspace)
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    config = spec["model"]["net"]["config"]["hypernet"]
    assert config["mode"] == mode and config["rank"] == 3
    module = _ParametricLightningModule.init_from_config(spec["model"])
    for branch in module.net._branches.values():
        convolutions = {
            name
            for name in branch._template_parameters
            if name.endswith("_conv.weight")
        }
        assert convolutions
        assert convolutions.issubset(branch._hypernet.target_names) is condition_dilated
        if not condition_dilated:
            assert not convolutions.intersection(branch._hypernet.target_names)
        assert all(
            branch._template_parameters[name].requires_grad for name in convolutions
        )
    module.net.set_uniform_batch_params(True)
    x = 0.1 * torch.randn(1, module.net.receptive_field + 31)
    params, y = torch.tensor([[3.0]]), 0.1 * torch.randn(1, 32)
    module.on_train_epoch_start()
    with torch.no_grad():
        predictions = module.net.forward_sizes(x, params)
        losses = (predictions - y[:, None]).square().mean(dim=(0, 2))
        expected = losses.sum() if reduction == "sum" else losses.mean()
    actual = module.training_step((x, params, y), 0)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    for branch in module.net._branches.values():
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for name, p in branch._template_parameters.items()
            if name.endswith("_conv.weight")
        )


def test_generation_controls_persist_and_presets_reset_visible_values(
    window, workspace
):
    window._attach(workspace)
    fields = window.advanced_fields
    assert not fields["hypernetwork_rank"].isEnabled()
    fields["hypernetwork_mode"].setCurrentIndex(
        fields["hypernetwork_mode"].findData("low_rank")
    )
    assert fields["hypernetwork_rank"].isEnabled()
    fields["hypernetwork_rank"].setValue(3)
    fields["condition_dilated_convolutions"].setChecked(False)
    fields["branch_loss_reduction"].setCurrentIndex(
        fields["branch_loss_reduction"].findData("mean")
    )
    options = Workspace(workspace.directory).training_options()
    assert (
        options["hypernetwork_mode"] == "low_rank" and options["hypernetwork_rank"] == 3
    )
    assert options["condition_dilated_convolutions"] is False
    assert options["branch_loss_reduction"] == "mean"
    fields["hypernetwork_mode"].setCurrentIndex(
        fields["hypernetwork_mode"].findData("full")
    )
    assert not fields["hypernetwork_rank"].isEnabled()
    assert not fields["condition_dilated_convolutions"].isChecked()
    assert fields["branch_loss_reduction"].currentData() == "mean"
    for preset in ("HyperWaveNet", "HyperA2"):
        fields["hypernetwork_recipe"].setCurrentIndex(
            fields["hypernetwork_recipe"].findData(preset)
        )
        window.apply_training_preset()
        assert workspace.training_options() == training_preset(preset)
    window.capture_kind.setCurrentIndex(window.capture_kind.findData("static"))
    for key in (
        "hypernetwork_mode",
        "hypernetwork_rank",
        "condition_dilated_convolutions",
        "branch_loss_reduction",
    ):
        assert not fields[key].isEnabled()


@pytest.mark.parametrize("preset", ["HyperA2", "HyperWaveNet"])
def test_existing_044_job_resumes_only_with_matching_generation(workspace, preset):
    workspace.preferences["advanced_parametric"] = training_preset(preset)
    ready(workspace)
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    for key in (
        "hypernetwork_mode",
        "hypernetwork_rank",
        "condition_dilated_convolutions",
        "branch_loss_reduction",
    ):
        spec["preferences"]["advanced_training"].pop(key)
    spec["fingerprint"] = workspace.fingerprint(include_hypernetwork_controls=False)
    atomic_write_json(job / "spec.json", spec)
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(workspace, job)
    for key, value in dict(
        hypernetwork_mode="low_rank" if preset == "HyperA2" else "full",
        condition_dilated_convolutions=(preset != "HyperA2"),
        branch_loss_reduction="mean" if preset == "HyperA2" else "sum",
    ).items():
        workspace.preferences["advanced_parametric"] = {
            **training_preset(preset),
            key: value,
        }
        assert not training.job_can_resume(workspace, job)

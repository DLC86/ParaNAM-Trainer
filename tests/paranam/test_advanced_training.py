import pytest

from paranam import training
from paranam.workspace import Workspace, read_json


def ready(ws):
    for entry in ws.project.entries:
        target = ws.directory / entry.y_path
        target.parent.mkdir(exist_ok=True)
        options = ws.training_options()
        scheduled = options.get("rate_preset", "manual") != "manual" or options.get(
            "hypernetwork_rate_preset", "follow"
        ) not in ("manual", "follow")
        if entry.synthetic_mute:
            continue
        target.unlink(missing_ok=True)
        if scheduled:
            import os, shutil

            source = ws.directory / ws.project.input_for_split(entry.split)
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        else:
            target.write_bytes(b"recorded capture placeholder")
        entry.status, entry.delay = "captured", 0
    ws.save()


@pytest.mark.parametrize("kind,lr", [("static", 0.004), ("parametric", 0.004)])
def test_defaults_match_reference_recipes(workspace, kind, lr):
    workspace.preferences["capture_type"] = kind
    workspace.generate_plan()
    ready(workspace)
    spec = read_json(training.make_job(workspace) / "spec.json")
    assert spec["model"]["optimizer"] == {
        "class": "Adam",
        "lr": lr,
        "weight_decay": 3.17e-7,
    }
    assert spec["model"]["lr_scheduler"]["kwargs"]["gamma"] == 0.994
    assert spec["model"]["loss"]["mrstft_weight"] == 0.0005
    assert all(item["ny"] == 8192 for item in spec["data"]["train"])
    assert all(item["ny"] is None for item in spec["data"]["validation"])
    assert spec["learning"]["val_dataloader"]["batch_size"] == 1
    assert spec["learning"]["train_dataloader"]["drop_last"] is True
    if kind == "static":
        assert "gradient_clip_val" not in spec["learning"]["trainer"]
    else:
        assert spec["learning"]["trainer"]["gradient_clip_val"] == 1.0
        assert spec["model"]["hypernet_lr_scale"] == 0.1
    assert not spec["data"].get("joint")  # User explicitly excludes RMS normalization.
    if kind == "parametric":
        assert spec["learning"]["rms_loss_normalization"] is True
        assert spec["learning"]["snr_weighted_loss_normalization"] is False
        assert spec["model"]["net"]["config"]["hypernet"]["mode"] == "full"
        assert spec["model"]["net"]["config"]["hypernet"]["selector"] == {}
        assert spec["model"]["loss"]["training_recipe"] == "a2"


@pytest.mark.parametrize("kind", ["static", "parametric"])
@pytest.mark.parametrize("fit", [False, True])
def test_custom_recipe_reaches_worker_and_prevents_incompatible_resume(
    workspace, kind, fit
):
    workspace.preferences["capture_type"] = kind
    workspace.preferences["advanced_" + kind] = dict(
        learning_rate=0.00123,
        lr_decay=0.98,
        ny=4096,
        fit_mrstft=fit,
        mrstft_weight=0.007,
        weight_decay=1.23e-6,
    )
    workspace.generate_plan()
    ready(workspace)
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    assert spec["model"]["optimizer"] == {
        "class": "Adam",
        "lr": 0.00123,
        "weight_decay": 1.23e-6,
    }
    assert spec["model"]["lr_scheduler"]["kwargs"]["gamma"] == 0.98
    assert spec["model"]["loss"]["mrstft_weight"] == (0.007 if fit else None)
    assert all(item["ny"] == 4096 for item in spec["data"]["train"])
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(workspace, job)
    workspace.preferences["advanced_" + kind]["learning_rate"] *= 2
    assert not training.job_can_resume(workspace, job)


def test_collapsed_panel_preserves_independent_recipes(window, workspace):
    window._attach(workspace)
    assert not window.advanced_toggle.isChecked()
    assert window.advanced_panel.isHidden()
    assert window.batch_size.parent() is window.advanced_panel
    optimizer = window.advanced_fields["optimizer"]
    assert optimizer.currentData() == "Adam"
    optimizer.setCurrentIndex(optimizer.findData("AdamW"))
    window.advanced_toggle.click()
    assert not window.advanced_panel.isHidden()
    window.advanced_fields["learning_rate"].setValue(0.00123)
    window.advanced_fields["fit_mrstft"].setChecked(False)
    assert not window.advanced_fields["mrstft_weight"].isEnabled()
    window.capture_kind.setCurrentIndex(window.capture_kind.findData("static"))
    assert window.advanced_fields["learning_rate"].value() == 0.004
    assert window.advanced_fields["fit_mrstft"].isChecked()
    assert optimizer.currentData() == "Adam"
    window.advanced_fields["ny"].setValue(4096)
    window.capture_kind.setCurrentIndex(window.capture_kind.findData("parametric"))
    assert window.advanced_fields["learning_rate"].value() == 0.00123
    assert not window.advanced_fields["fit_mrstft"].isChecked()
    assert optimizer.currentData() == "AdamW"
    reopened = Workspace(workspace.directory)
    assert reopened.training_options("static")["ny"] == 4096
    assert reopened.training_options("parametric")["learning_rate"] == 0.00123
    assert reopened.training_options("parametric")["optimizer"] == "AdamW"


def test_legacy_checkpoint_resumes_only_with_the_same_recipe(workspace):
    from nam.capture.project import atomic_write_json

    ready(workspace)
    workspace.preferences["advanced_parametric"] = {"optimizer": "AdamW"}
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    spec["preferences"].pop("advanced_training")
    spec["model"]["optimizer"].pop("class")
    spec["fingerprint"] = workspace.fingerprint(include_advanced=False)
    atomic_write_json(job / "spec.json", spec)
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(workspace, job)
    workspace.preferences["advanced_parametric"] = {"ny": 4096}
    assert not training.job_can_resume(workspace, job)


@pytest.mark.parametrize("kind", ["static", "parametric"])
@pytest.mark.parametrize("name", ["Adam", "AdamW"])
def test_actual_optimizer_selection_and_resume_guard(workspace, kind, name):
    import torch
    from nam.train.lightning_module import PackedLightningModule
    from nam.train.parametric import _ParametricLightningModule

    workspace.preferences["capture_type"] = kind
    workspace.preferences["advanced_" + kind] = {"optimizer": name}
    workspace.generate_plan()
    ready(workspace)
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    module_class = (
        PackedLightningModule if kind == "static" else _ParametricLightningModule
    )
    module = module_class.init_from_config(spec["model"])
    configured = module.configure_optimizers()
    optimizer = configured["optimizer"]
    assert type(optimizer) is getattr(torch.optim, name)
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint["optimizer_class"] == name
    module.on_load_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="Checkpoint uses"):
        module.on_load_checkpoint(
            {**checkpoint, "optimizer_class": "AdamW" if name == "Adam" else "Adam"}
        )
    loss = sum(p.square().mean() for p in module.parameters())
    loss.backward()
    optimizer.step()
    assert optimizer.state
    assert module._optimizer_config["class"] == name
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(workspace, job)
    workspace.preferences["advanced_" + kind]["optimizer"] = (
        "AdamW" if name == "Adam" else "Adam"
    )
    assert not training.job_can_resume(workspace, job)


@pytest.mark.parametrize(
    "kind,old_optimizer", [("static", "Adam"), ("parametric", "AdamW")]
)
def test_041_resume_preserves_actual_legacy_optimizer(workspace, kind, old_optimizer):
    from nam.capture.project import atomic_write_json

    workspace.preferences["capture_type"] = kind
    workspace.preferences["advanced_" + kind] = {"optimizer": old_optimizer}
    workspace.generate_plan()
    ready(workspace)
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    spec["preferences"]["advanced_training"].pop("optimizer")
    spec["model"]["optimizer"].pop("class")
    spec["fingerprint"] = workspace.fingerprint(include_optimizer=False)
    atomic_write_json(job / "spec.json", spec)
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(workspace, job)
    workspace.preferences["advanced_" + kind]["optimizer"] = (
        "Adam" if old_optimizer == "AdamW" else "AdamW"
    )
    assert not training.job_can_resume(workspace, job)


@pytest.mark.parametrize("method", ["sobol", "optimized"])
def test_new_sampling_methods_are_available_and_saved(window, workspace, method):
    window._attach(workspace)
    choice = window.lhs_method
    index = choice.findData(method)
    assert index >= 0
    choice.setCurrentIndex(index)
    assert Workspace(workspace.directory).preferences["lhs_method"] == method
    window.load_advanced_options("parametric")
    assert choice.currentData() == method


def test_snr_checkbox_reaches_the_training_job_independently_of_rms(window, workspace):
    key = "snr_weighted_loss_normalization"
    ready(workspace)
    window._attach(workspace)
    rms = window.advanced_fields["rms_loss_normalization"]
    field = window.advanced_fields[key]
    assert rms.isEnabled() and rms.isChecked()
    assert field.isEnabled() and field.isChecked()
    fingerprint = workspace.fingerprint()
    field.setChecked(False)
    reopened = Workspace(workspace.directory)
    assert reopened.training_options()[key] is False
    assert reopened.training_options()["rms_loss_normalization"] is True
    assert reopened.fingerprint() != fingerprint
    job = training.make_job(reopened)
    spec = read_json(job / "spec.json")
    assert spec["learning"]["rms_loss_normalization"] is True
    assert spec["learning"][key] is False
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(reopened, job)

    # Turning RMS on/off must never change the SNR selection.
    rms.setChecked(False)
    assert field.isEnabled() and not field.isChecked()
    assert not training.job_can_resume(Workspace(workspace.directory), job)
    rms.setChecked(True)
    assert field.isEnabled() and not field.isChecked()
    assert training.job_can_resume(Workspace(workspace.directory), job)

    # SNR mode, like RMS mode, is incompatible with the saved optimizer state.
    field.setChecked(True)
    enabled = Workspace(workspace.directory)
    assert not training.job_can_resume(enabled, job)
    assert enabled.fingerprint() == fingerprint
    enabled_spec = read_json(training.make_job(enabled) / "spec.json")
    assert enabled_spec["learning"]["rms_loss_normalization"] is True
    assert enabled_spec["learning"][key] is True
    window.load_advanced_options("static")
    assert not rms.isEnabled() and not field.isEnabled()


def test_rms_normalization_exponent_ui_and_resume(window, workspace):
    ready(workspace)
    window._attach(workspace)
    rms = window.advanced_fields["rms_loss_normalization"]
    exponent = window.advanced_fields["rms_loss_normalization_exponent"]
    assert exponent.value() == 2.0
    assert exponent.minimum() == pytest.approx(0.01)
    # RMS is enabled by default in 0.8.0, including its exponent editor.
    assert exponent.isEnabled()
    rms.setChecked(False)
    assert not exponent.isEnabled()
    rms.setChecked(True)
    assert exponent.isEnabled()
    fingerprint_default = workspace.fingerprint()
    exponent.setValue(1.0)
    changed = Workspace(workspace.directory)
    assert changed.training_options()["rms_loss_normalization_exponent"] == 1.0
    assert changed.fingerprint() != fingerprint_default
    spec_path = training.make_job(changed) / "spec.json"
    spec = read_json(spec_path)
    assert spec["learning"]["rms_loss_normalization_exponent"] == 1.0
    job = spec_path.parent
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(changed, job)

    exponent.setValue(0.0)
    assert exponent.value() == pytest.approx(0.01)
    assert Workspace(workspace.directory).training_options()["rms_loss_normalization_exponent"] == pytest.approx(0.01)
    assert not training.job_can_resume(Workspace(workspace.directory), job)
    rms.setChecked(False)
    assert not exponent.isEnabled()
    # When RMS normalization is disabled, the exponent cannot affect training.
    fingerprint_disabled = Workspace(workspace.directory).fingerprint()
    exponent.setValue(2.0)
    assert Workspace(workspace.directory).fingerprint() == fingerprint_disabled
    window.load_advanced_options("static")
    assert not exponent.isEnabled()


def test_legacy_zero_exponent_migrates_to_equivalent_snr_only_settings(workspace):
    ready(workspace)
    workspace.preferences.setdefault("advanced_parametric", {}).update(
        rms_loss_normalization=True,
        rms_loss_normalization_exponent=0.0,
        snr_weighted_loss_normalization=True,
    )
    options = workspace.training_options("parametric")
    assert options["rms_loss_normalization"] is False
    assert options["snr_weighted_loss_normalization"] is True
    assert options["rms_loss_normalization_exponent"] == pytest.approx(0.01)
    spec = read_json(training.make_job(workspace) / "spec.json")
    assert spec["learning"]["rms_loss_normalization"] is False
    assert spec["learning"]["snr_weighted_loss_normalization"] is True
    assert spec["learning"]["rms_loss_normalization_exponent"] == pytest.approx(0.01)


def test_synthetic_mute_reference_is_only_visible_option(window, workspace):
    from nam.capture.params import KnobSpec

    workspace.draft_controls = [KnobSpec("Gain", 0, 10, mute_at_min=True).to_dict()]
    workspace.generate_plan()
    ready(workspace)
    window._attach(workspace)
    assert "mute_training_mode" not in window.advanced_fields
    assert not any(key.startswith("mute_anchor_") for key in window.advanced_fields)
    field = window.advanced_fields["synthetic_mute_reference_rms"]
    assert field.isChecked() is False
    initial_job = training.make_job(workspace)
    spec = read_json(initial_job / "spec.json")
    assert spec["model"]["loss"]["synthetic_mute"] is True
    assert "mute_anchor" not in spec["model"]["loss"]
    assert spec["learning"]["synthetic_mute_reference_rms"] is False
    (initial_job / "resume-0.ckpt").touch()
    assert training.job_can_resume(workspace, initial_job)
    field.setChecked(True)
    reopened = Workspace(workspace.directory)
    assert reopened.training_options()["synthetic_mute_reference_rms"] is True
    assert not training.job_can_resume(reopened, initial_job)
    new = read_json(training.make_job(reopened) / "spec.json")
    assert new["learning"]["synthetic_mute_reference_rms"] is True

@pytest.mark.parametrize("kind", ["static", "parametric"])
def test_legacy_mute_options_do_not_affect_new_jobs(workspace, kind):
    workspace.preferences["capture_type"] = kind
    workspace.generate_plan()
    ready(workspace)
    before = workspace.fingerprint()
    workspace.preferences.setdefault("advanced_" + kind, {}).update(
        mute_anchor_interval=7, mute_training_mode="virtual"
    )
    assert workspace.fingerprint() == before
    spec = read_json(training.make_job(workspace) / "spec.json")
    assert "mute_anchor" not in spec["model"]["loss"]


def test_validation_and_checkpoint_choices_are_opt_in_and_persist(window, workspace):
    window._attach(workspace)
    assert window.validation_method.currentData() == "independent"
    assert window.advanced_fields["checkpoint_method"].currentData() == "validation"
    assert not window.advanced_fields["checkpoint_alpha"].isEnabled()
    window.validation_method.setCurrentIndex(
        window.validation_method.findData("farthest")
    )
    assert workspace.preferences["validation_method"] == "farthest"
    assert workspace.plan_changed()
    method = window.advanced_fields["checkpoint_method"]
    method.setCurrentIndex(method.findData("combined"))
    alpha = window.advanced_fields["checkpoint_alpha"]
    assert alpha.isEnabled()
    alpha.setValue(0.65)
    assert workspace.training_options()["checkpoint_method"] == "combined"
    assert workspace.training_options()["checkpoint_alpha"] == pytest.approx(0.65)
    method.setCurrentIndex(method.findData("tolerance"))
    assert not alpha.isEnabled()
    assert workspace.training_options()["checkpoint_method"] == "tolerance"
    method.setCurrentIndex(method.findData("combined"))
    assert alpha.value() == pytest.approx(0.65)

    reopened = Workspace(workspace.directory)
    assert reopened.preferences["validation_method"] == "farthest"
    assert reopened.training_options()["checkpoint_method"] == "combined"
    assert reopened.training_options()["checkpoint_alpha"] == pytest.approx(0.65)

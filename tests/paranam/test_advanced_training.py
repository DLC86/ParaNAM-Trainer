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


def test_snr_checkbox_reaches_the_training_job(window, workspace):
    key = "snr_weighted_loss_normalization"
    ready(workspace)
    window._attach(workspace)
    field = window.advanced_fields[key]
    assert field.isEnabled() and not field.isChecked()
    fingerprint = workspace.fingerprint()
    field.setChecked(True)
    reopened = Workspace(workspace.directory)
    assert reopened.training_options()[key] is True
    assert reopened.fingerprint() != fingerprint
    spec = read_json(training.make_job(reopened) / "spec.json")
    assert spec["learning"][key] is True
    window.load_advanced_options("static")
    assert not field.isEnabled()


def test_mute_anchor_fields_persist_and_reach_job(window, workspace):
    from nam.capture.params import KnobSpec

    workspace.draft_controls = [KnobSpec("Gain", 0, 10, mute_at_min=True).to_dict()]
    workspace.generate_plan()
    ready(workspace)
    window._attach(workspace)
    form = window.advanced_panel.layout()
    initial_job = training.make_job(workspace)
    (initial_job / "resume-0.ckpt").touch()
    assert training.job_can_resume(workspace, initial_job)
    for row, (key, label, default, custom) in enumerate(
        (
            ("interval", "Mute Anchor Interval", 1, 3),
            ("batch_size", "Mute Anchor Batch Size", 2, 4),
            ("ny", "Mute Anchor NY", 512, 1000),
        ),
        start=form.rowCount() - 3,
    ):
        field = window.advanced_fields[f"mute_anchor_{key}"]
        assert form.getWidgetPosition(field)[0] == row
        assert form.labelForField(field).text() == label
        assert field.minimum() == 1 and field.value() == default
        field.setValue(custom)

    reopened = Workspace(workspace.directory)
    assert not training.job_can_resume(reopened, initial_job)
    job = training.make_job(reopened)
    spec = read_json(job / "spec.json")
    assert spec["model"]["loss"]["mute_anchor"] == {
        "targets": [{"name": "Gain", "edge": "min"}],
        "weight": 0.25, "interval": 3, "batch_size": 4, "ny": 1000,
    }
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(reopened, job)
    window.load_advanced_options("static")
    assert not form.isRowVisible(window.advanced_fields["mute_anchor_interval"])


@pytest.mark.parametrize("kind", ["static", "parametric"])
def test_mute_anchor_options_do_not_affect_jobs_without_mute_targets(workspace, kind):
    workspace.preferences["capture_type"] = kind
    workspace.generate_plan()
    ready(workspace)
    before = workspace.fingerprint()
    workspace.preferences.setdefault("advanced_" + kind, {})["mute_anchor_interval"] = 7
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

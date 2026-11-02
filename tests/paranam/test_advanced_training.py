import pytest

from paranam import training
from paranam.workspace import Workspace, read_json


def ready(ws):
    for entry in ws.project.entries:
        target = ws.directory / entry.y_path
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"recorded audio")
        entry.status, entry.delay = "captured", 0
    ws.save()


@pytest.mark.parametrize("kind,lr", [("static", 0.004), ("parametric", 0.002)])
def test_defaults_match_reference_recipes(workspace, kind, lr):
    workspace.preferences["capture_type"] = kind
    workspace.generate_plan()
    ready(workspace)
    spec = read_json(training.make_job(workspace) / "spec.json")
    assert spec["model"]["optimizer"] == {"lr": lr, "weight_decay": 3.17e-7}
    assert spec["model"]["lr_scheduler"]["kwargs"]["gamma"] == 0.994
    assert spec["model"]["loss"]["mrstft_weight"] == 0.0005
    assert all(item["ny"] == 8192 for item in spec["data"]["train"])


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
    assert spec["model"]["optimizer"] == {"lr": 0.00123, "weight_decay": 1.23e-6}
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
    window.advanced_toggle.click()
    assert not window.advanced_panel.isHidden()
    window.advanced_fields["learning_rate"].setValue(0.00123)
    window.advanced_fields["fit_mrstft"].setChecked(False)
    assert not window.advanced_fields["mrstft_weight"].isEnabled()
    window.capture_kind.setCurrentIndex(window.capture_kind.findData("static"))
    assert window.advanced_fields["learning_rate"].value() == 0.004
    assert window.advanced_fields["fit_mrstft"].isChecked()
    window.advanced_fields["ny"].setValue(4096)
    window.capture_kind.setCurrentIndex(window.capture_kind.findData("parametric"))
    assert window.advanced_fields["learning_rate"].value() == 0.00123
    assert not window.advanced_fields["fit_mrstft"].isChecked()
    reopened = Workspace(workspace.directory)
    assert reopened.training_options("static")["ny"] == 4096
    assert reopened.training_options("parametric")["learning_rate"] == 0.00123


def test_legacy_checkpoint_resumes_only_with_the_same_recipe(workspace):
    from nam.capture.project import atomic_write_json

    ready(workspace)
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    spec["preferences"].pop("advanced_training")
    spec["fingerprint"] = workspace.fingerprint(include_advanced=False)
    atomic_write_json(job / "spec.json", spec)
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(workspace, job)
    workspace.preferences["advanced_parametric"] = {"ny": 4096}
    assert not training.job_can_resume(workspace, job)

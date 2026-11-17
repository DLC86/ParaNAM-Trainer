"""Focused regressions for 0.8.0 capture, Queue and Rules behavior."""

from pathlib import Path

import pytest

from nam.capture.export import build_hyper_a2_model_config
from nam.capture.params import KnobSpec
from nam.capture.plan_rules import PlanRule
from nam.capture.project import classify_capture, new_project, atomic_write_json
from paranam.plan_rules_ui import PlanRuleDialog, describe_rule
from paranam.training_queue import TrainingQueue
from paranam.workspace import Workspace, read_json
from tests.paranam.test_queue import small_ready
from tests.paranam.test_recovery import saved_checkpoint


def test_conditional_mute_is_dataset_supervision_without_anchor():
    from paranam.synthetic_mute import ensure_mute_rule_corners

    controls = [
        KnobSpec("Gain I", 0, 10, 1, 5),
        KnobSpec("Gain II", 0, 10, 1, 5),
        KnobSpec("Treble", 0, 10, 1, 5),
    ]
    rule = PlanRule(
        controls=("Gain I", "Gain II"), operator="=", value=0, kind="mute"
    )
    project = new_project(
        controls, n_train=12, n_validation=3, seed=123,
        plan_rules=[rule], lhs_method="sobol",
    )
    ensure_mute_rule_corners(project, [rule])
    assert any(entry.is_corner and rule.excludes(entry.params)
               for entry in project.entries if entry.split == "train")
    assert "mute_anchor" not in build_hyper_a2_model_config(project)["loss"]
    assert describe_rule(rule) == "Mute when Gain I = 0 AND Gain II = 0"

def test_rule_kind_editor_roundtrip(qapp):
    controls = [
        KnobSpec("Bass", 0, 10, 1, 5),
        KnobSpec("Mid", 0, 10, 1, 5),
        KnobSpec("Treble", 0, 10, 1, 5),
    ]
    rule = PlanRule(
        controls=("Bass", "Mid", "Treble"), operator="=", value=0, kind="mute"
    )
    dialog = PlanRuleDialog(None, controls, rule)
    assert dialog.rule_kind.currentData() == "mute"
    dialog.accept()
    assert dialog.rule == rule
    assert "Mute when" in describe_rule(dialog.rule)
    dialog.close()


def test_unstable_replacement_never_uses_previous_or_occupied_setting():
    project = new_project(
        [KnobSpec("Drive", 0, 10, 1, 5), KnobSpec("Tone", 0, 10, 1, 5)],
        n_train=10, n_validation=3, seed=94, lhs_method="sobol",
    )
    original = project.entries_for_split("train")[0]
    occupied = {tuple(sorted(row.params.items())) for row in project.entries}
    updated, added = classify_capture(project, original, "avoid")
    assert len(added) == 1
    key = tuple(sorted(added[0].params.items()))
    assert key not in occupied
    assert key != tuple(sorted(original.params.items()))
    assert len({
        tuple(sorted(row.params.items())) for row in updated.entries
    }) == len(updated.entries)


@pytest.mark.parametrize("kind", ["parametric", "static"])
def test_completed_job_epoch_extension_keeps_checkpoints_and_resumes(
    workspace, tmp_path, kind
):
    small_ready(workspace, kind)
    queue = TrainingQueue(tmp_path / "queue")
    entry = queue.add(workspace)
    job = Path(entry["job"])
    saved_checkpoint(job / "resume-0.ckpt", completed=2)
    atomic_write_json(
        job / "resume-0.json",
        {"completed_epoch": 2, "restart_epoch": 3, "global_step": 20},
    )
    progress = read_json(job / "progress.json")
    progress.update(
        status="completed", epoch=2, completed_epoch=2,
        member=0, completed_members=[0], exports=["model.nam"],
    )
    atomic_write_json(job / "progress.json", progress)
    queue.update(entry["id"], state="completed")
    with pytest.raises(ValueError, match="already completed"):
        queue.set_total_epochs(entry["id"], 1)
    before = (job / "resume-0.ckpt").read_bytes()
    queue.set_total_epochs(entry["id"], 5)
    spec = read_json(job / "spec.json")
    assert spec["learning"]["trainer"]["max_epochs"] == 5
    assert spec["preferences"]["epochs"] == 5
    assert Workspace(Path(spec["project"])).preferences["epochs"] == 5
    updated = read_json(job / "progress.json")
    assert updated["epochs"] == 5
    assert updated["completed_members"] == []
    assert updated["member"] == 0
    queue.prepare_resume(entry["id"])
    assert queue.get(entry["id"])["state"] == "queued"
    assert read_json(job / "spec.json")["learning"]["trainer"]["max_epochs"] == 5
    assert (job / "resume-0.ckpt").read_bytes() == before


def test_queue_epoch_change_rejects_live_worker(workspace, tmp_path):
    small_ready(workspace)
    queue = TrainingQueue(tmp_path / "queue")
    entry = queue.add(workspace)
    queue.update(entry["id"], state="running")
    with pytest.raises(ValueError, match="Pause or stop"):
        queue.set_total_epochs(entry["id"], 20)


def test_installer_frozen_queue_retains_snr_metadata(workspace, tmp_path):
    """The installer smoke test must not erase the queue's frozen SNR map."""
    import copy
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "packaging" / "validate_bundle.py"
    loader = importlib.util.spec_from_file_location("paranam_validate_bundle", script)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)

    small_ready(workspace)
    workspace.preferences["advanced_parametric"]["snr_weighted_loss_normalization"] = True
    workspace.save()
    queue = TrainingQueue(tmp_path / "installer-queue")
    entry = queue.add(workspace)
    frozen = read_json(Path(entry["job"]) / "spec.json")
    assert frozen["learning"]["snr_noise_powers"]
    assert all(v is None for v in frozen["learning"]["snr_noise_powers"].values())

    original = copy.deepcopy(frozen)
    original["data"]["train"][0]["y_path"] = "not-the-frozen-audio.wav"
    original["learning"].pop("snr_noise_powers")

    diagnostic = module._frozen_queue_diagnostic_spec(frozen, original)
    assert diagnostic["data"]["train"] != original["data"]["train"]
    assert diagnostic["data"]["train"][0]["y_path"] == frozen["data"]["train"][0]["y_path"]
    assert diagnostic["data"]["common"]["require_input_pre_silence"] is None
    assert all(
        capture["start_seconds"] == 0 and capture["stop_seconds"] is None
        for split in ("train", "train_evaluation")
        for capture in diagnostic["data"][split]
    )
    assert diagnostic["learning"]["snr_noise_powers"] == frozen["learning"]["snr_noise_powers"]
    assert "snr_noise_powers" not in original["learning"]
    # Preserve measured values as well as the explicit unavailable markers.
    measured = copy.deepcopy(frozen)
    key = next(iter(measured["learning"]["snr_noise_powers"]))
    measured["learning"]["snr_noise_powers"][key] = 0.001
    assert module._frozen_queue_diagnostic_spec(measured, original)[
        "learning"
    ]["snr_noise_powers"][key] == 0.001
    measured["learning"]["snr_noise_powers"].pop(key)
    with pytest.raises(ValueError, match="no SNR record"):
        module._frozen_queue_diagnostic_spec(measured, original)

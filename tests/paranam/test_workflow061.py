import math
from pathlib import Path

import pytest

from nam.capture.project import atomic_write_json
from paranam import training
from paranam.recipes import RATE_PRESET_LABELS
from paranam.workspace import read_json
from tests.paranam.test_workflow055 import ready_audio


def test_retrain_and_start_queue_restore_paused_old_recipe(
    workspace, tmp_path, qapp, monkeypatch
):
    from paranam.queue_controller import QueueController

    ready_audio(workspace)
    workspace.preferences["advanced_parametric"] = dict(
        rate_preset="a2_updates",
        hypernetwork_rate_preset="a2_controls",
        hypernetwork_learning_rate=0.001,
    )
    workspace.save()
    controller = QueueController(tmp_path / "queue")
    paused = controller.store.add(workspace)
    job = Path(paused["job"])
    spec = read_json(job / "spec.json")
    assert spec["model"]["lr_scheduler"]["kwargs"]["audio"]["step_size"] == 62
    assert (
        spec["model"]["lr_scheduler"]["kwargs"]["hypernetwork"]["step_size"] == 62
    )  # one knob in this fixture
    original_model = spec["model"]
    # Effective instance options take precedence over stale editor preferences.
    spec["preferences"]["advanced_parametric"]["rate_preset"] = "manual"
    atomic_write_json(job / "spec.json", spec)
    controller.store.update(paused["id"], state="paused")
    controller.paused_queue_id = paused["id"]
    atomic_write_json(controller.control_path, {"paused_job": paused["id"]})
    restored = controller.store.retrain(paused["id"])
    assert restored.training_options()["rate_preset"] == "a2_updates"
    assert restored.training_options()["hypernetwork_rate_preset"] == "a2_controls"
    assert restored.training_options()["hypernetwork_learning_rate"] == 0.001
    assert restored.training_ready() and not restored.plan_changed()
    replacement = controller.store.add(restored)
    assert read_json(Path(replacement["job"]) / "spec.json")["model"] == original_model
    ticks = []
    monkeypatch.setattr(controller, "tick", lambda: ticks.append(True))
    controller.start()
    assert ticks == [True] and controller.enabled
    assert controller.paused_queue_id is None
    assert controller.store.get(paused["id"])["state"] == "queued"
    assert read_json(job / "spec.json")["model"] == original_model


@pytest.mark.parametrize("size", ["full", "lite"])
def test_esr_bounds_are_fixed_inside_initial_range_and_expand_only_as_needed(
    qapp, size
):
    from paranam.esr_chart import ESRChart

    chart = ESRChart(size=size)
    assert chart.esr_limits() == (0.005, 0.5)
    history = [dict(epoch=1, **{size: 0.1, "train_" + size: 0.02})]
    chart.set_history(history, 100)
    assert chart.esr_limits() == (0.005, 0.5)
    chart.set_history(
        history + [dict(epoch=2, **{size: 0.006, "train_" + size: 0.49})], 100
    )
    assert chart.esr_limits() == (0.005, 0.5)
    chart.set_history(
        history + [dict(epoch=2, **{size: 0.8, "train_" + size: 0.003})], 100
    )
    lower, upper = chart.esr_limits()
    assert lower <= 0.003 and upper >= 0.8
    assert lower < 0.005 and upper > 0.5
    chart.set_history(
        [dict(epoch=1, **{size: float("nan"), "train_" + size: float("inf")})], 100
    )
    assert chart.esr_limits() == (0.005, 0.5)
    chart.set_history([dict(epoch=1, **{size: 0})], 100)
    assert 0 < chart.esr_limits()[0] <= 1e-12
    assert not chart.grab().isNull()


def test_old_ids_are_available_with_new_names_and_do_not_reset_saved_rates(
    window, workspace
):
    ready_audio(workspace)
    workspace.preferences["advanced_parametric"] = dict(
        rate_preset="a2_updates",
        hypernetwork_rate_preset="a2_controls",
        learning_rate=0.003,
        hypernetwork_learning_rate=0.001,
    )
    workspace.save()
    before = workspace.fingerprint()
    window._attach(workspace)
    for key in ("rate_preset", "hypernetwork_rate_preset"):
        selector = window.advanced_fields[key]
        for identifier, label in RATE_PRESET_LABELS.items():
            assert selector.itemText(selector.findData(identifier)) == label
    window.save_preferences()
    assert workspace.fingerprint() == before
    assert window.advanced_fields["learning_rate"].value() == 0.003
    assert window.advanced_fields["hypernetwork_learning_rate"].value() == 0.001

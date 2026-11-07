import math
from itertools import product

import pytest
from PySide6.QtCore import QLocale, QRect, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox, QWidget

from nam.capture.params import KnobSpec, SwitchSpec, validate_knobs
from nam.capture.planner import corner_capture_count, plan_corner_captures
from paranam.cadences import editable_cadences
from paranam.compare_layout import AssetList, ControlFlowLayout
from paranam.numeric import DecimalSpinBox
from paranam.queue_clock import ElapsedClock
from paranam.recipes import training_preset


@pytest.mark.parametrize("locale", ["it_IT", "de_DE", "en_US"])
@pytest.mark.parametrize("separator", [".", ","])
def test_decimal_keyboard_accepts_both_separators(qapp, locale, separator):
    spin = DecimalSpinBox()
    spin.setLocale(QLocale(locale))
    spin.setDecimals(6)
    spin.setRange(-100, 100)
    spin.setKeyboardTracking(False)
    spin.show()
    spin.setFocus()
    spin.selectAll()
    QTest.keyClicks(spin, "0" + separator + "125")
    assert spin.value() == 0
    QTest.keyClick(spin, Qt.Key.Key_Return)
    assert spin.value() == pytest.approx(0.125)
    spin.close()


def test_monotonic_elapsed_ignores_delayed_worker_reports():
    clock = ElapsedClock()
    assert clock.update("job", 60, True, now=100) == 60
    assert clock.update("job", 60.1, True, now=103) == 63
    assert clock.update("job", 62, True, now=104) == 64
    assert clock.value(now=105) == 65
    assert clock.update("job", 64.5, False, now=106) == 64.5
    assert clock.value(now=200) == 64.5
    assert clock.update("other", 3, True, now=201) == 3


def test_two_gains_cross_all_corners_and_switches():
    controls = [
        KnobSpec("Gain", 0, 10, is_gain=True),
        SwitchSpec("Bright", ("Off", "On")),
        KnobSpec("Drive", 0, 10, is_gain=True),
        KnobSpec("Tone", 0, 10),
    ]
    captures, _ = plan_corner_captures(controls)
    expected = set(product((0, 10), ("Off", "On"), (0, 10), (0, 10)))
    assert {tuple(c.params[k.name] for k in controls) for c in captures} == expected
    assert corner_capture_count(controls) == 16
    with pytest.raises(ValueError, match="two"):
        validate_knobs(controls + [KnobSpec("Boost", 0, 10, is_gain=True)])


def test_compare_without_project_and_two_visible_asset_lists(window, qapp):
    assert window.workspace is None
    window.navigate(5)
    assert window.pages.currentIndex() == 5
    assert window.playback_project().sample_rate == 48000
    assert window.play_model.isHidden()
    assert window.play_ir.isHidden()
    assert not window.play_model_list.isHidden()
    assert not window.play_ir_list.isHidden()


def test_asset_list_keeps_selection_and_loads_once(qapp):
    selector = QComboBox()
    selector.addItem("One", "/one")
    selector.addItem("Two", "/two")
    visible = AssetList(selector)
    loaded = []
    selector.currentIndexChanged.connect(loaded.append)
    visible.setCurrentRow(1)
    qapp.processEvents()
    assert loaded == [1]
    assert selector.currentData() == "/two"
    selector.addItem("Three", "/three")
    qapp.processEvents()
    assert visible.count() == 3
    assert visible.currentRow() == 1
    selector.setEnabled(False)
    assert not visible.isEnabled()


def test_compare_flow_packs_with_constant_gap(qapp):
    parent = QWidget()
    layout = ControlFlowLayout(parent)
    layout.setContentsMargins(0, 0, 0, 0)
    widgets = [QWidget() for _ in range(4)]
    for widget in widgets:
        widget.setFixedSize(132, 160)
        layout.addWidget(widget)
    parent.resize(580, 500)
    parent.show()
    qapp.processEvents()
    layout.setGeometry(QRect(0, 0, 580, 500))
    assert [w.x() for w in widgets[:3]] == [0, 156, 312]
    assert widgets[3].y() == 184
    layout.setGeometry(QRect(0, 0, 900, 500))
    assert [w.x() for w in widgets] == [0, 156, 312, 468]
    parent.close()


def test_compare_panel_reserves_height_for_wrapped_rows(window, workspace, qapp):
    window._attach(workspace)
    window.navigate(5)
    window.show()
    window._play_model_loaded(
        dict(
            size_selection=True,
            controls=[
                dict(name=f"Knob {i}", min=0, max=10, default=5, positions=[])
                for i in range(9)
            ],
        )
    )
    for _ in range(3):
        qapp.processEvents()
    panel = window.play_controls
    last = window.play_controls_layout.itemAt(8).widget()
    assert last.geometry().bottom() < panel.height()
    assert last.y() > window.play_controls_layout.itemAt(0).widget().y()


def test_new_recipe_and_independent_epoch_budgets(window, workspace):
    recipe = training_preset("HyperA2")
    assert recipe["hypernetwork_learning_rate"] == 0.001
    assert recipe["hypernetwork_lr_decay"] == 0.998497
    assert recipe["branch_loss_reduction"] == "sum"
    assert recipe["rate_preset"] == recipe["hypernetwork_rate_preset"] == "a2_updates"
    assert not training_preset("HyperWaveNet")["cudnn_benchmark"]
    window._attach(workspace)
    window.epochs.setValue(57)
    window.save_preferences()
    window.capture_kind.setCurrentIndex(window.capture_kind.findData("static"))
    assert window.epochs.value() == 200
    window.epochs.setValue(210)
    window.save_preferences()
    window.capture_kind.setCurrentIndex(window.capture_kind.findData("parametric"))
    assert window.epochs.value() == 57
    for key in ("rate_preset", "hypernetwork_rate_preset"):
        selector = window.advanced_fields[key]
        assert [selector.itemData(i) for i in range(selector.count())] == [
            "manual",
            "a2_updates",
            "cosine_budget",
        ]


def test_legacy_rate_conversion_preserves_mean_and_fingerprint(window, workspace):
    workspace.draft_controls = [
        KnobSpec("Gain", 0, 10).to_dict(),
        KnobSpec("Tone", 0, 10).to_dict(),
        SwitchSpec("Bright", ("Off", "On")).to_dict(),
    ]
    workspace.preferences.update(n_train=4, n_validation=2)
    workspace.generate_plan()
    workspace.preferences["advanced_parametric"] = dict(
        rate_preset="a2_slow",
        hypernetwork_rate_preset="a2_controls",
        lr_decay=0.98,
        hypernetwork_lr_decay=0.97,
    )
    options = workspace.training_options()
    view = editable_cadences(workspace, options)
    assert view["lr_decay"] == pytest.approx(0.98 ** (math.sqrt(2) / 4))
    assert view["hypernetwork_lr_decay"] == pytest.approx(0.97**0.25)
    from tests.paranam.test_training import ready

    ready(workspace)
    before = workspace.fingerprint()
    window._attach(workspace)
    window.save_preferences()
    assert workspace.fingerprint() == before


def test_capture_auto_advance_resets_progress_and_blinks_only_changes(
    window, workspace
):
    from types import SimpleNamespace

    window._attach(workspace)
    first = window._capture_entries[window.capture_table.currentRow()]
    window._recorded_control_values = dict(first.params)
    window._recorded_capture_path = first.y_path
    first.status = "captured"
    window._capture_done(SimpleNamespace(messages=[]))
    following = window._capture_entries[window.capture_table.currentRow()]
    assert following.y_path != first.y_path
    assert window.capture_progress.value() == 0
    changed = {
        name
        for name in following.params
        if first.params[name] != following.params[name]
    }
    assert window.scene.blink_names == changed
    window.refresh_captures()
    assert window.scene.blink_names == changed
    window.rig_ready.setChecked(True)
    assert not window.scene.blink_names and not window.scene.blink_timer.isActive()
    window.scene.start_blink(changed)
    window.select_capture()
    assert not window.scene.blink_names


def test_all_queue_states_can_be_reordered_without_changing_active_job(
    workspace, tmp_path
):
    from paranam.training_queue import TrainingQueue
    from tests.paranam.test_queue import small_ready

    small_ready(workspace)
    queue = TrainingQueue(tmp_path / "queue")
    entries = [queue.add(workspace) for _ in range(3)]
    for entry, state in zip(entries, ("completed", "running", "paused")):
        queue.update(entry["id"], state=state)
    queue.move(entries[2]["id"], -1)
    assert [e["id"] for e in queue.entries()] == [
        entries[0]["id"],
        entries[2]["id"],
        entries[1]["id"],
    ]
    assert queue.get(entries[1]["id"])["state"] == "running"
    queue.move(entries[0]["id"], 1)
    assert queue.entries()[1]["id"] == entries[0]["id"]


def test_project_folder_opens_selected_project(window, workspace, monkeypatch):
    from PySide6.QtGui import QDesktopServices

    window._attach(workspace)
    window.recent.setCurrentRow(0)
    opened = []
    monkeypatch.setattr(
        QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile())
    )
    window.open_project_folder_button.click()
    from pathlib import Path

    assert Path(opened[0]) == workspace.directory


def test_seconds_are_retained_past_one_hour():
    from paranam.wizard import duration

    assert duration(3723) == "1h 02m 03s"


def test_follow_conversion_becomes_independent_when_audio_decay_is_edited(workspace):
    from paranam.cadences import preserve_unchanged_cadences

    old = workspace.training_options()
    assert old["hypernetwork_rate_preset"] == "follow"
    entered = editable_cadences(workspace, old)
    entered["lr_decay"] = 0.9
    saved = preserve_unchanged_cadences(workspace, old, entered)
    assert saved["hypernetwork_rate_preset"] == "manual"
    assert saved["hypernetwork_lr_decay"] == old["lr_decay"]


def test_resume_keeps_frozen_scheduler_after_equivalent_cadence_conversion(workspace):
    from paranam import training
    from paranam.workspace import read_json
    from tests.paranam.test_workflow055 import ready_audio

    workspace.draft_controls.append(SwitchSpec("Bright", ("Off", "On")).to_dict())
    workspace.preferences.update(n_train=4, n_validation=2)
    ready_audio(workspace)
    workspace.preferences["advanced_parametric"] = dict(
        rate_preset="a2_slow",
        hypernetwork_rate_preset="a2_controls",
        lr_decay=0.98,
        hypernetwork_lr_decay=0.97,
    )
    workspace.save()
    job = training.make_job(workspace)
    original = read_json(job / "spec.json")["model"]
    from tests.paranam.test_recovery import saved_checkpoint

    saved_checkpoint(job / "resume-0.ckpt", completed=1)
    view = editable_cadences(workspace, workspace.training_options())
    for key in ("lr_decay", "hypernetwork_lr_decay"):
        view[key] = round(view[key], 9)
    workspace.preferences["advanced_parametric"] = view
    workspace.save()
    assert training.job_can_resume(workspace, job)
    assert training.make_job(workspace, resume=True) == job
    assert read_json(job / "spec.json")["model"] == original
    workspace.preferences["advanced_parametric"]["hypernetwork_rank"] = 3
    assert not training.job_can_resume(workspace, job)


def test_new_queued_instances_use_simplified_equivalent_clocks(workspace, tmp_path):
    from paranam.training_queue import TrainingQueue
    from paranam.workspace import read_json
    from tests.paranam.test_workflow055 import ready_audio
    from pathlib import Path

    workspace.draft_controls.append(SwitchSpec("Bright", ("Off", "On")).to_dict())
    workspace.preferences.update(n_train=4, n_validation=2)
    ready_audio(workspace)
    workspace.preferences["advanced_parametric"] = dict(
        rate_preset="a2_slow",
        hypernetwork_rate_preset="a2_controls",
        lr_decay=0.98,
        hypernetwork_lr_decay=0.97,
    )
    queue = TrainingQueue(tmp_path / "queue")
    entry = queue.add(workspace)
    scheduler = read_json(Path(entry["job"]) / "spec.json")["model"]["lr_scheduler"][
        "kwargs"
    ]
    assert scheduler["audio"] == dict(
        interval="step", step_size=62, gamma=0.98 ** (math.sqrt(2) / 2)
    )
    assert scheduler["hypernetwork"] == dict(
        interval="step", step_size=62, gamma=0.97**0.5
    )
    assert workspace.training_options()["rate_preset"] == "a2_slow"

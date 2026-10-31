from nam.capture.params import KnobSpec, SwitchSpec
from paranam.widgets import ControlDialog


def test_four_pages_and_back_navigation_keep_drafts(window, workspace):
    window._attach(workspace)
    assert window.pages.count() == 4
    window.navigate(1)
    window.mode.setCurrentIndex(1)
    window.epochs.setValue(75)
    window.back()
    assert window.pages.currentIndex() == 0
    window.next()
    assert window.mode.currentData() == "active_learning"
    assert workspace.preferences["epochs"] == 75
    window.next()
    assert window.pages.currentIndex() == 2
    assert not window.next_button.isEnabled()
    assert not window.train_button.isEnabled()
    window.back()
    assert window.pages.currentIndex() == 1
    assert window.n_train.value() == 2


def test_switch_dialog_can_change_type_count_labels_and_default(qapp):
    dialog = ControlDialog(control=SwitchSpec("Mode", ("A", "B", "C"), 2))
    assert dialog.pages.currentIndex() == 1
    dialog.count.setValue(4)
    dialog.positions.item(3, 0).setText("Boost")
    dialog.switch_default.setCurrentIndex(3)
    dialog._accept()
    assert dialog.result_control.enum_names == ("A", "B", "C", "Boost")
    assert dialog.result_control.default == 3
    dialog.close()


def test_switch_coverage_recommendations_and_signal_selection(window, workspace):
    workspace.draft_controls = [
        KnobSpec("Gain", 0, 10).to_dict(),
        SwitchSpec("Mode", ("A", "B", "C")).to_dict(),
        SwitchSpec("Bright", ("Off", "On")).to_dict(),
    ]
    window._attach(workspace)
    window.refresh_recommendation(True)
    assert window.n_train.value() == 90
    assert window.n_validation.value() == 18
    assert "6 switch" in window.recommendation.text()
    assert not window.custom_button.isEnabled()
    window.signal_mode.setCurrentIndex(1)
    assert window.custom_button.isEnabled()
    assert workspace.preferences["input_mode"] == "custom"


def test_capture_diagram_shows_numeric_and_named_switch_values(window, workspace):
    workspace.draft_controls.append(
        SwitchSpec("Mode", ("Off", "Clean", "Lead")).to_dict()
    )
    workspace.preferences.update(n_train=6, n_validation=3)
    workspace.generate_plan()
    window._attach(workspace)
    window.navigate(2)
    window.select_capture()
    assert window.scene.values["Mode"] in ("Off", "Clean", "Lead")
    assert isinstance(window.scene.values["Gain"], float)
    assert not window.record_button.isEnabled()
    window.rig_ready.setChecked(True)
    assert window.record_button.isEnabled()

import json

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QLabel

from paranam.widgets import MetadataEditor
from paranam.workspace import Workspace
from tests.paranam.test_workflow052 import configure_audio


def settle(qapp):
    for _ in range(20):
        qapp.processEvents()


def test_settings_are_global_and_project_switches_never_replace_them(
    window, workspace, tmp_path, monkeypatch
):
    window.navigate(6)
    assert window.audio_card.isEnabled() and window.calibration_card.isEnabled()
    configure_audio(window, monkeypatch)
    window._apply_level_calibration(window.input_calibration, window.input_dbu, 17.123)
    window.apply_instrument_calibration(21.456)
    snapshot = workspace.project.audio.model_dump()
    window.open_project(workspace.directory)
    assert window.pages.currentIndex() == 1
    assert window.device.currentData().name == "Test interface"
    assert window.input_dbu.value() == 17.123
    assert window.workspace.project.audio.model_dump() == snapshot
    window.navigate(6)
    window.monitor_left.setCurrentIndex(window.monitor_left.findData(5))
    assert Workspace(workspace.directory).project.audio.model_dump() == snapshot
    other = Workspace.create(tmp_path / "Other", "Other")
    other.project.audio.input_device = "A completely different device"
    other.save()
    window.open_project(other.directory)
    assert window.device.currentData().name == "Test interface"
    assert window.monitor_left.currentData() == 5
    assert window.instrument_reference() == 21.456
    setup = json.loads(window.settings.value("last_audio_setup"))
    assert setup["audio"]["monitor_output_left"] == 5
    assert setup["instrument_level_dbu"] == 21.456


def test_navigation_brand_recent_rows_and_responsive_queue(window, workspace, qapp):
    window._attach(workspace)
    window.show()
    settle(qapp)
    assert [button.text() for button in window.steps] == [
        "Projects",
        "Plan",
        "Capture",
        "Train",
        "Queue",
        "Compare",
        "Settings",
    ]
    assert window.audio_card.parent() == window.page_contents[6]
    assert window.calibration_card.parent() == window.page_contents[6]
    assert window.settings_button.y() > window.steps[5].y() + window.steps[5].height()
    assert not any(
        "ONE CAPTURE SET" in label.text() or "Full + Lite, together." in label.text()
        for label in window.findChildren(QLabel)
    )
    assert window.version_label.text() == "VERSION 0.5.3"
    assert (
        window.version_label.mapTo(
            window.centralWidget(), window.version_label.rect().topLeft()
        ).x()
        == window.brand_title.mapTo(
            window.centralWidget(), window.brand_title.rect().topLeft()
        ).x()
    )
    assert window.brand_title.font().weight() >= QFont.Weight.Bold
    assert (
        abs(
            window.shop_button.height()
            - (window.brand_title.height() + window.version_label.height() + 4)
        )
        <= 2
    )
    title, path = window.recent.itemDelegate().fonts()
    assert title.weight() == QFont.Weight.Bold and path.weight() == QFont.Weight.Normal
    assert title.pixelSize() > path.pixelSize()
    hint = window.recent.itemDelegate().sizeHint(None, None)
    assert (
        hint.height() >= QFontMetrics(title).height() + QFontMetrics(path).height() + 30
    )
    assert window.recent.height() >= 320
    window.navigate(4)
    settle(qapp)
    original = window.queue_table.height(), window.training_tabs.height()
    window.resize(window.width(), window.height() + 240)
    settle(qapp)
    assert window.queue_table.height() > original[0]
    assert window.training_tabs.height() > original[1]
    window.navigate(6)
    window.back()
    assert window.pages.currentIndex() == 4


def test_static_has_only_name_in_capture_and_shared_metadata_in_train(
    window, workspace
):
    workspace.preferences["capture_type"] = "static"
    workspace.generate_plan()
    identifier = workspace.preferences["static_captures"][0]["id"]
    workspace.set_static_metadata(
        identifier, {"modeled_by": "Old author", "custom": {"keep": True}}
    )
    window._attach(workspace)
    window.navigate(2)
    assert window.static_card.isVisibleTo(window.page_contents[2])
    assert not window.static_card.findChildren(MetadataEditor)
    assert window.static_name.text() == "Capture 1"
    window.navigate(3)
    assert window.metadata_editor.fields["name"].isHidden()
    window.save_metadata({"modeled_by": "New author", "gear_model": "Shared amp"})
    assert workspace.project.user_metadata["modeled_by"] == "New author"
    assert "modeled_by" not in workspace.preferences["static_captures"][0]["metadata"]
    assert workspace.preferences["static_captures"][0]["metadata"]["custom"] == {
        "keep": True
    }
    window.capture_kind.setCurrentIndex(window.capture_kind.findData("parametric"))
    assert not window.metadata_editor.fields["name"].isHidden()


def test_capture_and_job_remember_used_calibration_after_global_settings_change(
    window, workspace, tmp_path, monkeypatch
):
    from pathlib import Path
    from tests.paranam.test_queue import small_ready
    from paranam.training_queue import TrainingQueue
    from paranam.workspace import read_json

    small_ready(workspace)
    window._attach(workspace)
    configure_audio(window, monkeypatch)
    window._apply_level_calibration(window.input_calibration, window.input_dbu, 6.125)
    window._apply_level_calibration(window.output_calibration, window.output_dbu, 12.5)
    window.apply_instrument_calibration(18.123)
    workspace.project.input_level_dbu = 6.125
    workspace.project.output_level_dbu = 12.5
    window.remember_capture_calibration()
    queue = TrainingQueue(tmp_path / "Queue")
    entry = queue.add(workspace)
    before = read_json(Path(entry["job"]) / "spec.json")["capture_calibration"]
    assert before["input_level_dbu"] == 6.125
    assert before["output_level_dbu"] == 12.5
    assert before["instrument_level_dbu"] == 18.123
    window._apply_level_calibration(window.input_calibration, window.input_dbu, 20)
    window.apply_instrument_calibration(24)
    assert Workspace(workspace.directory).capture_calibration() == before
    assert read_json(Path(entry["job"]) / "spec.json")["capture_calibration"] == before


def test_cancelling_recapture_does_not_invalidate_the_old_dataset(
    window, workspace, monkeypatch
):
    from PySide6.QtWidgets import QMessageBox
    from tests.paranam.test_queue import small_ready

    small_ready(workspace)
    window._attach(workspace)
    configure_audio(window, monkeypatch)
    before = workspace.capture_calibration()
    window._apply_level_calibration(window.input_calibration, window.input_dbu, 18)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No
    )
    assert not window.apply_plan()
    assert workspace.training_ready()
    assert workspace.capture_calibration() == before

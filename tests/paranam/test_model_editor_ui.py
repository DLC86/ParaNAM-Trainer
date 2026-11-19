"""Model Editor GUI and the side-by-side Compare header."""

import copy
import json
from pathlib import Path

import pytest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QMessageBox

from paranam.model_editor_core import editable_controls
from tests.paranam.test_model_editor import sample_pnam


def test_compare_header_groups_all_description_and_playback_selectors_left(window, qapp):
    window.navigate(5)
    window.resize(1260, 840)
    window.show()
    qapp.processEvents()
    parent = window.play_header_info
    note = next(
        widget for widget in parent.findChildren(type(window.play_status))
        if "Instrument → selected instrument input" in widget.text()
    )
    assert note.parentWidget() is parent
    assert window.play_buffer.parentWidget() is parent
    assert window.play_size.parentWidget() is parent
    assert window.play_output_mode.parentWidget() is parent
    meters = [window.play_input_meter, window.play_output_meter]
    for meter in meters:
        # Meter columns belong to the header, not the left-hand info block.
        assert meter.parentWidget() is not parent
        meter_left = meter.mapTo(window.page_contents[5], QPoint(0, 0)).x()
        left_end = parent.mapTo(
            window.page_contents[5], QPoint(parent.width(), 0)
        ).x()
        assert meter_left >= left_end
        assert meter.parentWidget().y() == parent.y()


def test_editor_sidebar_metadata_and_parametric_index_edits(window, tmp_path, qapp, monkeypatch):
    source = tmp_path / "amp.pnam"
    original = sample_pnam()
    source.write_text(json.dumps(original), encoding="utf-8")
    window.navigate(7)
    assert window.editor_button.isChecked()
    assert window.editor_button.icon().isNull() is False
    assert window.settings_button.isChecked() is False
    assert not window.editor_save_button.isEnabled()

    window.editor_open_file(path=source)
    assert window.editor_controls_card.isVisibleTo(window.page_contents[7])
    assert window.editor_control_list.count() == 3
    assert "Gain" in window.editor_control_list.item(1).text()
    assert window.editor_metadata.fields["modeled_by"].text() == "Old author"

    window.editor_metadata.fields["modeled_by"].setText("Edited author")
    gear = window.editor_metadata.fields["gear_type"]
    gear.setCurrentIndex(gear.findData("amp"))
    window.editor_metadata.fields["input_level_dbu"].setText("12.25")
    window.editor_metadata.fields["notes"].setPlainText("Preserve weights")

    # Change display index without changing the neural parameter order.
    window.editor_control_list.setCurrentRow(1)  # Gain
    window.editor_control_index.setValue(0)
    assert [c["name"] for c in sorted(
        window.editor_control_rows, key=lambda c: c["index"]
    )] == ["Gain", "Mode", "Volume"]
    window.editor_control_name.setText("Drive")
    window._editor_commit_control_name()
    # Switch choices retain value 0..2 even after physical reordering.
    switch = next(c for c in window.editor_control_rows if c["type"] == "switch")
    row_index = next(i for i in range(window.editor_control_list.count())
                     if window.editor_control_list.item(i).data(
                         Qt.ItemDataRole.UserRole) == switch["key"])
    window.editor_control_list.setCurrentRow(row_index)
    window.editor_option_list.setCurrentRow(0)
    before = copy.deepcopy(switch["options"])
    window.editor_option_index.setValue(2)
    assert sorted(o["value"] for o in switch["options"]) == [0, 1, 2]
    assert [o["index_sw"] for o in switch["options"]] != [
        o["index_sw"] for o in before
    ]
    window.editor_option_name.setText("Alternative")
    window._editor_commit_option_name()

    destination = tmp_path / "amp_edited.pnam"
    monkeypatch.setattr(
        "paranam.model_editor_ui.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(destination), ""),
    )
    window.editor_save_copy()
    assert destination.is_file()
    assert json.loads(source.read_text()) == original
    exported = json.loads(destination.read_text())
    assert exported["weights"] == original["weights"]
    assert exported["metadata"]["modeled_by"] == "Edited author"
    assert exported["metadata"]["input_level_dbu"] == 12.25
    assert exported["config"]["params"][0]["name"] == "Drive"
    assert exported["config"]["params"][0]["index"] == 0
    assert window.editor_source_path == destination
    assert editable_controls(exported)


def test_editor_static_nam_only_shows_metadata_and_creates_backup(
    window, tmp_path, monkeypatch,
):
    source = tmp_path / "plain.nam"
    source.write_text(json.dumps({
        "architecture": "WaveNet",
        "config": {"head_scale": 0.01},
        "weights": [0.0, 0.5],
        "metadata": {"loudness": -20, "modeled_by": "Old"},
    }), encoding="utf-8")
    window.navigate(7)
    window.editor_open_file(path=source)
    assert window.editor_controls_card.isHidden()
    window.editor_metadata.fields["gear_make"].setText("New make")
    monkeypatch.setattr(
        "paranam.model_editor_ui.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    window.editor_save_file()
    assert json.loads(source.read_text())["metadata"]["gear_make"] == "New make"
    assert json.loads((tmp_path / "plain.nam.bak").read_text())["metadata"]["modeled_by"] == "Old"


def test_editor_rejects_invalid_numeric_inputs_without_modifying_model(
    window, tmp_path, monkeypatch,
):
    source = tmp_path / "invalid.nam"
    document = {
        "architecture": "WaveNet", "config": {},
        "weights": [0.5], "metadata": {"loudness": -20},
    }
    source.write_text(json.dumps(document), encoding="utf-8")
    window.navigate(7)
    window.editor_open_file(path=source)
    window.editor_metadata.fields["input_level_dbu"].setText("nan")
    messages = []
    monkeypatch.setattr(
        "paranam.model_editor_ui.QMessageBox.warning",
        lambda *args: messages.append(str(args[-1])),
    )
    window._editor_save(destination=source)
    assert source.read_text() == json.dumps(document)
    assert any("finite" in item for item in messages)

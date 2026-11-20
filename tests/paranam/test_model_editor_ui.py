"""Model Editor GUI and the side-by-side Compare header."""

import copy
import json
from pathlib import Path

import pytest

from PySide6.QtCore import QPoint, QPointF, QMimeData, QUrl, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QMessageBox

from paranam.model_editor_core import editable_controls
def sample_pnam():
    params = [
        {"name": "Gain", "min": 0, "max": 10, "default": 5,
         "type": "continuous", "index": 1},
        {"name": "Mode", "min": 0, "max": 2, "default": 1, "type": "switch",
         "index": 0,
         "options": [
             {"name": "Clean", "value": 0, "index_sw": 2},
             {"name": "Crunch", "value": 1, "index_sw": 0},
             {"name": "Lead", "value": 2, "index_sw": 1},
         ]},
    ]
    return {
        "architecture": "ParametricSlimmableContainer",
        "weights": [0.125, -0.5],
        "config": {
            "params": copy.deepcopy(params),
            "submodels": [
                {"name": size, "config": {"params": copy.deepcopy(params)}}
                for size in ("lite", "full")
            ],
        },
        "metadata": {"modeled_by": "Old author", "loudness": -15.0},
        "extensions": {
            "linear_volume": {
                "name": "Volume", "index": 2, "min": 0, "max": 10,
                "default": 10, "points": [[0, -60], [10, 0]],
            }
        },
    }



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
    # The VU meters flank the Input / Output controls in the gain grid.
    grid = window.play_gain_layout
    input_meter_column = grid.itemAtPosition(0, 0).widget()
    input_label = grid.itemAtPosition(0, 1).widget()
    output_label = grid.itemAtPosition(0, 3).widget()
    output_meter_column = grid.itemAtPosition(0, 4).widget()
    assert input_meter_column.findChild(type(window.play_input_meter)) is window.play_input_meter
    assert output_meter_column.findChild(type(window.play_output_meter)) is window.play_output_meter
    assert input_meter_column.mapTo(window.page_contents[5], QPoint(0, 0)).x() < input_label.mapTo(window.page_contents[5], QPoint(0, 0)).x()
    assert output_meter_column.mapTo(window.page_contents[5], QPoint(0, 0)).x() > output_label.mapTo(window.page_contents[5], QPoint(output_label.width(), 0)).x()
    for caption in parent.findChildren(type(window.play_buffer_label)):
        if caption.text() in ("Model size", "Output mode"):
            assert caption.font().pointSizeF() == window.play_buffer_label.font().pointSizeF()
    assert window.play_size.geometry().bottom() <= parent.height()
    assert window.play_output_mode.geometry().bottom() <= parent.height()
    assert parent.layout().spacing() == 3
    assert parent.layout().itemAt(2).layout().contentsMargins().top() == 0
    assert parent.layout().itemAt(3).layout().contentsMargins().bottom() == 0


def test_editor_sidebar_metadata_and_parametric_index_edits(window, tmp_path, qapp, monkeypatch):
    source = tmp_path / "amp.pnam"
    original = sample_pnam()
    source.write_text(json.dumps(original), encoding="utf-8")
    window.navigate(7)
    assert window.editor_button.isChecked()
    assert window.editor_button.icon().isNull() is False
    assert window.settings_button.isChecked() is False
    assert not window.editor_save_button.isEnabled()

    window.resize(1260, 840)
    window.show()
    window.editor_open_file(path=source)
    qapp.processEvents()
    assert not window.editor_controls_card.isHidden()
    assert window.editor_controls_card.height() > 190
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
    assert window.editor_control_min.isHidden()
    assert window.editor_control_max.isHidden()
    assert window.editor_control_default.decimals() == 3
    window.editor_control_default.setValue(7.5)
    gain_control = next(c for c in window.editor_control_rows if c["name"] == "Gain")
    assert gain_control["default"] == 7.5
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
    default_row = next(
        i for i in range(window.editor_option_list.count())
        if window.editor_option_list.item(i).data(Qt.ItemDataRole.UserRole) == 2
    )
    window.editor_option_list.setCurrentRow(default_row)
    window.editor_option_default.setChecked(True)
    assert switch["default"] == 2
    assert window.editor_option_default.isChecked()
    default_widget = window.editor_option_list.itemWidget(
        window.editor_option_list.item(default_row)
    )
    marker = default_widget.findChild(type(window.editor_path_label), "editorDefaultMarker")
    assert marker.text() == "Default"
    description = default_widget.findChild(type(window.editor_path_label), "editorOptionDescription")
    assert description.text().startswith("1: Lead")
    assert default_widget.height() >= 30
    assert marker.alignment() & Qt.AlignmentFlag.AlignRight
    assert window.editor_option_list.item(default_row).text() == ""
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
    assert exported["config"]["params"][0]["default"] == 7.5
    assert exported["config"]["params"][1]["default"] == 2
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
    monkeypatch.setattr(
        "paranam.model_editor_ui.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    window._editor_save(destination=source)
    assert source.read_text() == json.dumps(document)
    assert any("finite" in item for item in messages)



def _drop_local_file(qapp, widget, path):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    enter = QDragEnterEvent(
        QPoint(2, 2), Qt.DropAction.CopyAction, mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    qapp.sendEvent(widget, enter)
    assert enter.isAccepted()
    drop = QDropEvent(
        QPointF(2, 2), Qt.DropAction.CopyAction, mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    qapp.sendEvent(widget, drop)
    assert drop.isAccepted()


def test_editor_metadata_card_accepts_model_drag_and_drop(
    window, tmp_path, qapp,
):
    source = tmp_path / "dragged.nam"
    source.write_text(json.dumps({
        "architecture": "WaveNet", "config": {},
        "weights": [0.25], "metadata": {"modeled_by": "Dropped"},
    }), encoding="utf-8")
    window.navigate(7)
    _drop_local_file(qapp, window.editor_metadata, source)
    assert window.editor_source_path == source.resolve()
    assert window.editor_metadata.fields["modeled_by"].text() == "Dropped"


def test_compare_model_and_ir_panels_accept_drag_and_drop(
    window, tmp_path, qapp, monkeypatch,
):
    model = tmp_path / "dragged.pnam"
    model.write_text("{}", encoding="utf-8")
    ir = tmp_path / "dragged.wav"
    ir.write_bytes(b"RIFF")
    window.navigate(5)
    refreshed = []
    monkeypatch.setattr(
        window, "refresh_play_models",
        lambda selected=None, **kwargs: refreshed.append(selected),
    )
    _drop_local_file(qapp, window.play_model_list, model)
    assert refreshed == [str(model)]
    assert str(model) in (window.settings.value("play_models", []) or [])

    window.play_ir.blockSignals(True)
    _drop_local_file(qapp, window.play_ir_list, ir)
    window.play_ir.blockSignals(False)
    assert window.play_ir.findData(str(ir)) >= 0
    assert str(ir) in (window.settings.value("play_irs", []) or [])



def test_compare_controls_show_empty_model_message(window, qapp):
    window.navigate(5)
    qapp.processEvents()
    assert window.play_controls.empty_label.text() == "No model loaded"
    assert not window.play_controls.empty_label.isHidden()

"""Draft 0.7.3 project safety and editable model metadata regression tests."""

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from paranam import wizard


class FakeCheckBox:
    def __init__(self, *args):
        self._checked = False

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        self._checked = checked


def test_project_delete_requires_confirmation_and_defaults_to_keep_files(
    window, workspace, monkeypatch
):
    window.settings.setValue("recent", [str(workspace.directory)])
    window._refresh_recent()
    window.recent.setCurrentRow(0)
    assert window.delete_project_button.isEnabled()

    class Confirmation:
        Icon = QMessageBox.Icon
        ButtonRole = QMessageBox.ButtonRole
        StandardButton = QMessageBox.StandardButton
        confirm = False
        delete_directory = False
        dialogs = []

        def __init__(self, parent=None):
            self.checkbox = None
            self.delete = object()
            self.cancel = object()
            self.clicked = None
            self.dialogs.append(self)

        def setIcon(self, value):
            pass

        def setWindowTitle(self, value):
            pass

        def setText(self, value):
            pass

        def setInformativeText(self, value):
            pass

        def setCheckBox(self, checkbox):
            self.checkbox = checkbox

        def addButton(self, *args):
            return self.delete if args[0] == "Delete" else self.cancel

        def setDefaultButton(self, *args):
            pass

        def exec(self):
            assert self.checkbox is not None
            assert not self.checkbox.isChecked()
            self.checkbox.setChecked(self.delete_directory)
            self.clicked = self.delete if self.confirm else self.cancel

        def clickedButton(self):
            return self.clicked

    monkeypatch.setattr(wizard, "QMessageBox", Confirmation)
    monkeypatch.setattr(wizard, "QCheckBox", FakeCheckBox)
    window.delete_selected_project()
    assert workspace.directory.is_dir()  # Cancel does nothing.
    assert str(workspace.directory) in window.settings.value("recent", [])

    Confirmation.confirm = True
    window.delete_selected_project()
    assert workspace.directory.is_dir()  # Default Delete only removes the entry.
    assert str(workspace.directory) not in window.settings.value("recent", [])
    assert len(Confirmation.dialogs) == 2


def test_project_delete_explicit_opt_in_removes_directory(window, workspace, monkeypatch):
    window.settings.setValue("recent", [str(workspace.directory)])
    window._refresh_recent()
    window.recent.setCurrentRow(0)

    class Confirmation:
        Icon = QMessageBox.Icon
        ButtonRole = QMessageBox.ButtonRole
        StandardButton = QMessageBox.StandardButton

        def __init__(self, parent=None):
            self.selected = object()

        def setIcon(self, *args):
            pass

        def setWindowTitle(self, *args):
            pass

        def setText(self, *args):
            pass

        def setInformativeText(self, *args):
            pass

        def setCheckBox(self, checkbox):
            assert not checkbox.isChecked()
            checkbox.setChecked(True)

        def addButton(self, *args):
            return self.selected if args[0] == "Delete" else object()

        def setDefaultButton(self, *args):
            pass

        def exec(self):
            pass

        def clickedButton(self):
            return self.selected

    monkeypatch.setattr(wizard, "QMessageBox", Confirmation)
    monkeypatch.setattr(wizard, "QCheckBox", FakeCheckBox)
    window.delete_selected_project()
    assert not workspace.directory.exists()
    assert not window.settings.value("recent", [])


def test_model_dbu_fields_only_change_export_metadata(window, workspace):
    window._attach(workspace)
    window.input_calibration.setChecked(True)
    window.output_calibration.setChecked(True)
    window.input_dbu.setValue(12.5)
    window.output_dbu.setValue(18.25)
    # A newly opened project receives values from the active Settings.
    window._attach(workspace)
    fields = window.metadata_editor.fields
    assert float(fields["input_level_dbu"].text()) == 12.5
    assert float(fields["output_level_dbu"].text()) == 18.25
    assert fields["input_z"].placeholderText() == ""
    fields["input_level_dbu"].setText("-4,5")
    fields["output_level_dbu"].setText("2.75")
    window.metadata_editor.emit_metadata()
    assert workspace.project.user_metadata["input_level_dbu"] == -4.5
    assert workspace.project.user_metadata["output_level_dbu"] == 2.75
    assert window.input_dbu.value() == 12.5
    assert window.output_dbu.value() == 18.25

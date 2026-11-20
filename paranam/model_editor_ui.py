"""Standalone NAM/PNAM model metadata editor screen."""

from __future__ import annotations

from pathlib import Path
import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QSpinBox, QVBoxLayout, QWidget, QSizePolicy, QDoubleSpinBox, QComboBox,
)

from .widgets import MetadataEditor
from .impedance import parse_impedance
from .compare_layout import FileDropTarget
from .model_editor_core import (
    USER_FIELDS, apply_model_edits, editable_controls,
    load_model_file, write_model_file,
)


class EditorActions:
    """Qt UI only; model semantics and safe saves live in model_editor_core."""

    def _editor(self, layout):
        title = QLabel("Editor")
        title.setObjectName("title")
        layout.addWidget(title)
        explainer = QLabel(
            "Edit NAM / ParaNAM metadata without changing the neural audio. "
            "Open a file, edit its fields, then overwrite it or save a separate copy."
        )
        explainer.setWordWrap(True)
        explainer.setObjectName("muted")
        layout.addWidget(explainer)
        file_row = QHBoxLayout()
        self.editor_open_button = QPushButton("Load NAM / PNAM…")
        self.editor_open_button.clicked.connect(self.editor_open_file)
        file_row.addWidget(self.editor_open_button)
        self.editor_save_button = QPushButton("Overwrite file")
        self.editor_save_button.clicked.connect(self.editor_save_file)
        self.editor_save_button.setEnabled(False)
        file_row.addWidget(self.editor_save_button)
        self.editor_save_as_button = QPushButton("Save copy as…")
        self.editor_save_as_button.clicked.connect(self.editor_save_copy)
        self.editor_save_as_button.setEnabled(False)
        file_row.addWidget(self.editor_save_as_button)
        file_row.addStretch()
        layout.addLayout(file_row)
        self.editor_path_label = QLabel("No model loaded")
        self.editor_path_label.setWordWrap(True)
        self.editor_path_label.setObjectName("muted")
        layout.addWidget(self.editor_path_label)

        metadata_card = QFrame()
        metadata_card.setObjectName("card")
        metadata_layout = QVBoxLayout(metadata_card)
        metadata_heading = QLabel("MODEL METADATA")
        metadata_heading.setObjectName("eyebrow")
        metadata_layout.addWidget(metadata_heading)
        self.editor_metadata = MetadataEditor(include_name=False)
        self.editor_metadata.changed.connect(self._editor_metadata_changed)
        metadata_layout.addWidget(self.editor_metadata)
        layout.addWidget(metadata_card)
        self._editor_metadata_drop_target = FileDropTarget(
            metadata_card, lambda path: self.editor_open_file(path=path),
            {".nam", ".pnam"},
        )

        self.editor_controls_card = QFrame()
        self.editor_controls_card.setObjectName("card")
        self.editor_controls_card.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        control_layout = QVBoxLayout(self.editor_controls_card)
        control_heading = QLabel("PARAMETRIC CONTROLS")
        control_heading.setObjectName("eyebrow")
        control_layout.addWidget(control_heading)
        note = QLabel(
            "Display order (index), names and switch display positions "
            "(index_sw) do not change neural weights or switch values. "
            "Knob min/max define neural input normalization and cannot be "
            "changed safely without retraining."
        )
        note.setWordWrap(True)
        note.setObjectName("muted")
        control_layout.addWidget(note)

        container = QHBoxLayout()
        self.editor_control_list = QListWidget()
        self.editor_control_list.setMinimumWidth(225)
        self.editor_control_list.setMinimumHeight(190)
        self.editor_control_list.currentRowChanged.connect(
            self._editor_selected_control
        )
        container.addWidget(self.editor_control_list, 1)

        detail = QWidget()
        form = QFormLayout(detail)
        self.editor_control_name = QLineEdit()
        self.editor_control_name.textEdited.connect(
            self._editor_control_name_edited
        )
        self.editor_control_name.editingFinished.connect(
            self._editor_commit_control_name
        )
        form.addRow("Control name", self.editor_control_name)
        self.editor_control_index = QSpinBox()
        self.editor_control_index.setToolTip(
            "Visual order only. The neural parameter order and weights never move."
        )
        self.editor_control_index.valueChanged.connect(
            self._editor_control_index_changed
        )
        form.addRow("index", self.editor_control_index)
        default_layout = QHBoxLayout()
        self.editor_control_default = QDoubleSpinBox()
        self.editor_control_default.setDecimals(6)
        self.editor_control_default.setKeyboardTracking(False)
        self.editor_control_default.valueChanged.connect(
            self._editor_control_default_changed
        )
        self.editor_control_default_switch = QComboBox()
        self.editor_control_default_switch.currentIndexChanged.connect(
            self._editor_control_default_switch_changed
        )
        default_layout.addWidget(self.editor_control_default)
        default_layout.addWidget(self.editor_control_default_switch)
        form.addRow("Default value", default_layout)
        self.editor_control_min = QLineEdit()
        self.editor_control_max = QLineEdit()
        for widget in (self.editor_control_min, self.editor_control_max):
            widget.setReadOnly(True)
            widget.setToolTip(
                "This range is part of the neural normalization; modifying it "
                "would change the modeled sound."
            )
        form.addRow("Knob min (locked)", self.editor_control_min)
        form.addRow("Knob max (locked)", self.editor_control_max)

        self.editor_switch_panel = QWidget()
        switch_layout = QVBoxLayout(self.editor_switch_panel)
        switch_layout.setContentsMargins(0, 0, 0, 0)
        switch_info = QLabel(
            "Switch choices are displayed in index_sw order; the saved "
            "neural value of each option remains unchanged."
        )
        switch_info.setWordWrap(True)
        switch_info.setObjectName("muted")
        switch_layout.addWidget(switch_info)
        self.editor_option_list = QListWidget()
        self.editor_option_list.setMinimumHeight(100)
        self.editor_option_list.currentRowChanged.connect(
            self._editor_selected_option
        )
        switch_layout.addWidget(self.editor_option_list)
        option_form = QFormLayout()
        self.editor_option_name = QLineEdit()
        self.editor_option_name.textEdited.connect(
            self._editor_option_name_edited
        )
        self.editor_option_name.editingFinished.connect(
            self._editor_commit_option_name
        )
        option_form.addRow("Option name", self.editor_option_name)
        self.editor_option_index = QSpinBox()
        self.editor_option_index.setToolTip(
            "Physical/display order only; the option's neural value does not move."
        )
        self.editor_option_index.valueChanged.connect(
            self._editor_option_index_changed
        )
        option_form.addRow("index_sw", self.editor_option_index)
        switch_layout.addLayout(option_form)
        form.addRow(self.editor_switch_panel)
        container.addWidget(detail, 2)
        control_layout.addLayout(container, 1)
        self.editor_controls_card.hide()
        layout.addWidget(self.editor_controls_card, 1)

        self.editor_status = QLabel("")
        self.editor_status.setWordWrap(True)
        self.editor_status.setObjectName("muted")
        layout.addWidget(self.editor_status)
        self.editor_document = None
        self.editor_source_path = None
        self.editor_control_rows = []
        self._editor_loading_fields = False
        self._editor_dirty = False
        layout.addStretch()

    def _editor_metadata_changed(self, *_):
        if self.editor_document is not None:
            self._editor_dirty = True
            self.editor_status.setText("Unsaved metadata changes")

    def editor_open_file(self, checked=False, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open NAM / PNAM model", "",
                "NAM / ParaNAM models (*.nam *.pnam)",
            )
        if not path:
            return
        if self._editor_dirty:
            confirm = QMessageBox.question(
                self, "Discard unsaved edits?",
                "The current model has unsaved edits. Discard them and open another file?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
        try:
            model = load_model_file(path)
            controls = editable_controls(model) if Path(path).suffix.lower() == ".pnam" else []
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Cannot open model", str(exc))
            return
        self.editor_document = model
        self._editor_dirty = False
        self.editor_source_path = Path(path).resolve()
        self.editor_control_rows = controls
        self.editor_path_label.setText(str(self.editor_source_path))
        self.editor_metadata.set_metadata(model.get("metadata", {}))
        self.editor_controls_card.setVisible(bool(controls))
        self._editor_refresh_controls()
        self.editor_save_button.setEnabled(True)
        self.editor_save_as_button.setEnabled(True)
        self.editor_status.setText(
            f"Loaded {self.editor_source_path.name}. "
            + ("Parametric controls are editable." if controls else "Metadata only.")
        )

    def _editor_current_control(self):
        row = self.editor_control_list.currentItem()
        if row is None:
            return None
        key = row.data(Qt.ItemDataRole.UserRole)
        return next(
            (c for c in self.editor_control_rows if c["key"] == key), None
        )

    def _editor_current_option(self):
        control = self._editor_current_control()
        item = self.editor_option_list.currentItem()
        if control is None or item is None or control["type"] != "switch":
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return next(
            (o for o in control["options"] if o["value"] == value), None
        )

    def _editor_refresh_controls(self, select_key=None):
        if select_key is None:
            current = self._editor_current_control()
            select_key = current["key"] if current else None
        self._editor_loading_fields = True
        self.editor_control_list.blockSignals(True)
        try:
            self.editor_control_list.clear()
            ordered = sorted(
                self.editor_control_rows, key=lambda c: c["index"]
            )
            wanted = 0
            for i, control in enumerate(ordered):
                suffix = "Linear Volume" if control["kind"] == "linear_volume" else control["type"].title()
                item = QListWidgetItem(
                    f"{control['index']}: {control['name']}  ({suffix})"
                )
                item.setData(Qt.ItemDataRole.UserRole, control["key"])
                self.editor_control_list.addItem(item)
                if control["key"] == select_key:
                    wanted = i
            if ordered:
                self.editor_control_list.setCurrentRow(wanted)
        finally:
            self.editor_control_list.blockSignals(False)
            self._editor_loading_fields = False
        self._editor_selected_control()

    def _editor_selected_control(self, *_):
        if self._editor_loading_fields:
            return
        control = self._editor_current_control()
        self._editor_loading_fields = True
        try:
            active = control is not None
            self.editor_control_name.setEnabled(active)
            self.editor_control_index.setEnabled(active)
            self.editor_control_default.setEnabled(active)
            self.editor_control_default_switch.setEnabled(active)
            if not active:
                self.editor_control_name.clear()
                self.editor_control_min.clear()
                self.editor_control_max.clear()
                self.editor_control_default.setValue(0.0)
                self.editor_control_default_switch.clear()
                self.editor_switch_panel.hide()
                return
            self.editor_control_name.setText(control["name"])
            self.editor_control_index.setRange(
                0, max(0, len(self.editor_control_rows) - 1)
            )
            self.editor_control_index.setValue(control["index"])
            self.editor_control_min.setText(str(control["min"]))
            self.editor_control_max.setText(str(control["max"]))
            if control["type"] == "switch":
                self.editor_control_default.setVisible(False)
                self.editor_control_default_switch.setVisible(True)
                self.editor_control_default_switch.blockSignals(True)
                self.editor_control_default_switch.clear()
                for option in sorted(
                    control["options"], key=lambda item: item["index_sw"]
                ):
                    self.editor_control_default_switch.addItem(
                        option["name"], option["value"]
                    )
                selected = self.editor_control_default_switch.findData(
                    control["default"]
                )
                self.editor_control_default_switch.setCurrentIndex(selected)
                self.editor_control_default_switch.blockSignals(False)
                self._editor_refresh_options()
            else:
                self.editor_control_default_switch.setVisible(False)
                self.editor_control_default.setVisible(True)
                self.editor_control_default.blockSignals(True)
                self.editor_control_default.setRange(
                    float(control["min"]), float(control["max"])
                )
                self.editor_control_default.setSingleStep(
                    max((float(control["max"]) - float(control["min"])) / 100, 0.000001)
                )
                self.editor_control_default.setValue(float(control["default"]))
                self.editor_control_default.blockSignals(False)
            self.editor_switch_panel.setVisible(control["type"] == "switch")
        finally:
            self._editor_loading_fields = False

    def _editor_control_default_changed(self, value):
        if self._editor_loading_fields:
            return
        control = self._editor_current_control()
        if control is None or control["type"] == "switch":
            return
        if control["default"] != value:
            control["default"] = float(value)
            self._editor_dirty = True
            self.editor_status.setText("Unsaved control default changes")

    def _editor_control_default_switch_changed(self, *_):
        if self._editor_loading_fields:
            return
        control = self._editor_current_control()
        if control is None or control["type"] != "switch":
            return
        value = self.editor_control_default_switch.currentData()
        if value is not None and control["default"] != value:
            control["default"] = value
            self._editor_dirty = True
            self.editor_status.setText("Unsaved control default changes")

    def _editor_control_name_edited(self, value):
        if self._editor_loading_fields:
            return
        control = self._editor_current_control()
        if control is None:
            return
        # Persist keystrokes against the *old* selected row before a mouse
        # click can move selection and deliver editingFinished late.
        control["name"] = value.strip()
        current = self.editor_control_list.currentItem()
        if current is not None:
            kind = (
                "Linear Volume" if control["kind"] == "linear_volume"
                else control["type"].title()
            )
            current.setText(f"{control['index']}: {control['name']}  ({kind})")
        self._editor_dirty = True
        self.editor_status.setText("Unsaved control changes")

    def _editor_commit_control_name(self):
        if self._editor_loading_fields:
            return
        control = self._editor_current_control()
        if control is not None:
            new_name = self.editor_control_name.text().strip()
            if control["name"] != new_name:
                control["name"] = new_name
                self._editor_refresh_controls(select_key=control["key"])
                self._editor_dirty = True
                self.editor_status.setText("Unsaved control changes")

    def _editor_control_index_changed(self, index):
        if self._editor_loading_fields:
            return
        control = self._editor_current_control()
        if control is None or control["index"] == index:
            return
        old = control["index"]
        for other in self.editor_control_rows:
            if other is control:
                continue
            if old < index and old < other["index"] <= index:
                other["index"] -= 1
            elif index < old and index <= other["index"] < old:
                other["index"] += 1
        control["index"] = index
        self._editor_refresh_controls(select_key=control["key"])
        self._editor_dirty = True
        self.editor_status.setText("Unsaved control order changes")

    def _editor_refresh_options(self, selected_value=None):
        if selected_value is None:
            selected = self._editor_current_option()
            selected_value = selected["value"] if selected else None
        control = self._editor_current_control()
        self.editor_option_list.blockSignals(True)
        self._editor_loading_fields = True
        try:
            self.editor_option_list.clear()
            if control is None or control["type"] != "switch":
                return
            wanted = 0
            for i, option in enumerate(sorted(
                control["options"], key=lambda o: o["index_sw"]
            )):
                item = QListWidgetItem(
                    f"{option['index_sw']}: {option['name']}  (value {option['value']})"
                )
                item.setData(Qt.ItemDataRole.UserRole, option["value"])
                self.editor_option_list.addItem(item)
                if option["value"] == selected_value:
                    wanted = i
            self.editor_option_list.setCurrentRow(wanted)
        finally:
            self.editor_option_list.blockSignals(False)
            self._editor_loading_fields = False
        self._editor_selected_option()

    def _editor_selected_option(self, *_):
        if self._editor_loading_fields:
            return
        control = self._editor_current_control()
        option = self._editor_current_option()
        self._editor_loading_fields = True
        try:
            active = option is not None
            self.editor_option_name.setEnabled(active)
            self.editor_option_index.setEnabled(active)
            if not active:
                self.editor_option_name.clear()
                return
            self.editor_option_name.setText(option["name"])
            self.editor_option_index.setRange(0, len(control["options"]) - 1)
            self.editor_option_index.setValue(option["index_sw"])
        finally:
            self._editor_loading_fields = False

    def _editor_option_name_edited(self, value):
        if self._editor_loading_fields:
            return
        option = self._editor_current_option()
        if option is None:
            return
        option["name"] = value.strip()
        current = self.editor_option_list.currentItem()
        if current is not None:
            current.setText(
                f"{option['index_sw']}: {option['name']}  (value {option['value']})"
            )
        self._editor_dirty = True
        self.editor_status.setText("Unsaved switch option changes")

    def _editor_commit_option_name(self):
        if self._editor_loading_fields:
            return
        option = self._editor_current_option()
        if option is not None:
            new_name = self.editor_option_name.text().strip()
            if option["name"] != new_name:
                option["name"] = new_name
                self._editor_refresh_options(selected_value=option["value"])
                self._editor_dirty = True
                self.editor_status.setText("Unsaved switch option changes")

    def _editor_option_index_changed(self, index):
        if self._editor_loading_fields:
            return
        control = self._editor_current_control()
        option = self._editor_current_option()
        if control is None or option is None or option["index_sw"] == index:
            return
        old = option["index_sw"]
        for other in control["options"]:
            if other is option:
                continue
            if old < index and old < other["index_sw"] <= index:
                other["index_sw"] -= 1
            elif index < old and index <= other["index_sw"] < old:
                other["index_sw"] += 1
        option["index_sw"] = index
        self._editor_refresh_options(selected_value=option["value"])
        self._editor_dirty = True
        self.editor_status.setText("Unsaved switch order changes")

    def _editor_collect_metadata(self):
        # MetadataEditor preserves unexposed/custom fields. Reject malformed
        # numeric input rather than silently saving an old value.
        for key in ("input_level_dbu", "output_level_dbu"):
            raw = self.editor_metadata.fields[key].text().strip()
            if raw:
                try:
                    number = float(raw.replace(",", "."))
                except ValueError as exc:
                    raise ValueError(f"Invalid {key}: enter a finite dBu value") from exc
                if not math.isfinite(number):
                    raise ValueError(f"Invalid {key}: enter a finite dBu value")
        parse_impedance(self.editor_metadata.fields["input_z"].text())
        self.editor_metadata.emit_metadata()
        return dict(self.editor_metadata._metadata)

    def _editor_prepare_save(self):
        if self.editor_document is None:
            raise ValueError("Load a NAM or PNAM file first")
        # Commit the selected switch choice before refreshing its parent
        # control selection; refreshing can otherwise reset the option row.
        self._editor_commit_option_name()
        self._editor_commit_control_name()
        return apply_model_edits(
            self.editor_document,
            self._editor_collect_metadata(),
            self.editor_control_rows,
        )

    def editor_save_file(self, *_):
        self._editor_save(destination=self.editor_source_path)

    def editor_save_copy(self, *_):
        if self.editor_source_path is None:
            return
        source = self.editor_source_path
        destination, _ = QFileDialog.getSaveFileName(
            self, "Save model copy", str(source.with_name(
                source.stem + "_edited" + source.suffix
            )),
            "NAM / ParaNAM models (*.nam *.pnam)",
        )
        if not destination:
            return
        path = Path(destination)
        if not path.suffix:
            path = path.with_suffix(source.suffix)
        if path.resolve() == source:
            QMessageBox.warning(
                self, "Choose a copy name",
                "Choose a different filename for the copy, or use Overwrite file.",
            )
            return
        self._editor_save(destination=path)

    def _editor_save(self, *, destination):
        if self.editor_source_path is None or destination is None:
            return
        destination = Path(destination).resolve()
        if destination.exists():
            message = (
                "Overwrite the currently loaded model? A .bak copy will be "
                "saved beside it." if destination == self.editor_source_path else
                "This file already exists. Replace it with your edited model?"
            )
            result = QMessageBox.question(
                self, "Confirm model overwrite", message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if result != QMessageBox.StandardButton.Yes:
                return
        try:
            document = self._editor_prepare_save()
            write_model_file(self.editor_source_path, destination, document)
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            QMessageBox.warning(self, "Cannot save model", str(exc))
            return
        self.editor_document = document
        self._editor_dirty = False
        self.editor_source_path = destination
        self.editor_path_label.setText(str(destination))
        self.editor_status.setText(f"Saved {destination.name}")
        self.editor_controls_card.setVisible(bool(self.editor_control_rows))

"""Four-step desktop workflow; audio and training never block the UI thread."""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

from PySide6.QtCore import (
    Qt,
    QProcess,
    QProcessEnvironment,
    QSettings,
    QTimer,
    QStandardPaths,
    QUrl,
)
from PySide6.QtGui import QDesktopServices, QColor, QTextCursor
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QScrollArea,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QFormLayout,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QProgressBar,
    QPlainTextEdit,
)

from nam.capture.audio import list_devices
from nam.capture.export import update_data_json
from nam.capture.gui.workers import CancelToken, SessionWorker
from nam.capture.params import KnobSpec, SwitchSpec, capture_recommendations
from nam.capture.planner import corner_capture_count
from nam.capture.project import AudioSettingsModel, CaptureProject, atomic_write_json
from nam.capture.session import CaptureSession, timebase_problem
from .workspace import Workspace, control_from_dict, read_json
from .widgets import (
    THEME,
    RoutingDiagram,
    ControlScene,
    ControlDialog,
    MetadataEditor,
    load_font,
)
from . import training
from .audio_devices import capabilities
from .calibration import OutputLevelEdit, LevelCalibrationDialog, gain_from_dbfs


def label(text, style=None):
    widget = QLabel(text)
    widget.setWordWrap(True)
    if style:
        widget.setObjectName(style)
    return widget


def button(text, callback, primary=False):
    widget = QPushButton(text.replace("&", "&&"))
    if primary:
        widget.setObjectName("primary")
    widget.clicked.connect(callback)
    return widget


def card(title):
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(22, 20, 22, 20)
    layout.setSpacing(14)
    layout.addWidget(label(title, "section"))
    return frame, layout


def duration(seconds):
    if seconds is None:
        return "Estimating…"
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes:02d}:{seconds:02d}"


class WizardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        load_font()
        self.setWindowTitle("ParaNAM Trainer")
        self.setStyleSheet(THEME)
        self.resize(1260, 840)
        self.setMinimumSize(1000, 700)
        self.settings = QSettings("DLC86", "ParaNAM Trainer")
        self.workspace = None
        self.session = None
        self.worker = None
        self.cancel_token = None
        self.process = None
        self.job = None
        self._loading = False
        self._close_after_pause = False
        self._log_content = ""
        self._devices = []
        self._capture_entries = []
        self._static_edit_id = None
        self._capture_table_kind = None
        self._build()
        self._refresh_recent()
        self.refresh_devices()
        self.timer = QTimer(self)
        self.timer.setInterval(600)
        self.timer.timeout.connect(self.refresh_training)
        self.timer.start()
        self.navigate(0)

    def _build(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(218)
        rail = QVBoxLayout(sidebar)
        rail.setContentsMargins(20, 30, 18, 24)
        rail.addWidget(label("ParaNAM", "brand"))
        rail.addWidget(label("TRAINER  /  0.3", "eyebrow"))
        rail.addSpacing(38)
        self.steps = []
        for index, text in enumerate(
            (
                "01   Project & routing",
                "02   Controls & plan",
                "03   Capture your rig",
                "04   Train & export",
            )
        ):
            step = button(text, lambda checked=False, i=index: self.navigate(i))
            step.setObjectName("step")
            step.setCheckable(True)
            rail.addWidget(step)
            self.steps.append(step)
        rail.addStretch()
        rail.addWidget(label("ONE CAPTURE SET.\nA WHOLE RANGE OF TONE.", "eyebrow"))
        rail.addSpacing(12)
        self.model_badge = label("HyperA2\nFull + Lite, together.", "muted")
        rail.addWidget(self.model_badge)
        root.addWidget(sidebar)
        right = QWidget()
        content = QVBoxLayout(right)
        content.setContentsMargins(30, 24, 30, 22)
        content.setSpacing(18)
        self.project_label = label("No project open", "muted")
        content.addWidget(self.project_label)
        self.pages = QStackedWidget()
        self.page_contents = []
        for builder in (self._welcome, self._controls, self._capture, self._training):
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(0, 0, 10, 0)
            layout.setSpacing(20)
            builder(layout)
            layout.addStretch()
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.pages.addWidget(scroll)
            self.page_contents.append(page)
        content.addWidget(self.pages, 1)
        footer = QHBoxLayout()
        self.back_button = button("← Back", self.back)
        self.footer_note = label("Settings are saved as you work.", "muted")
        self.next_button = button("Next →", self.next, True)
        footer.addWidget(self.back_button)
        footer.addWidget(self.footer_note, 1)
        footer.addWidget(self.next_button)
        content.addLayout(footer)
        root.addWidget(right, 1)
        self.setCentralWidget(central)

    def _welcome(self, layout):
        layout.addWidget(label("YOUR RIG. YOUR SOUND.", "eyebrow"))
        layout.addWidget(label("Make your next model.", "title"))
        layout.addWidget(
            label(
                "Start a project, connect your interface, and follow the guided capture steps.",
                "muted",
            )
        )
        projects, box = card("Your projects")
        row = QHBoxLayout()
        self.new_button = button("+ New project", self.new_project, True)
        self.open_button = button("Open project…", self.open_project_dialog)
        row.addWidget(self.new_button)
        row.addWidget(self.open_button)
        row.addStretch()
        box.addLayout(row)
        self.recent = QListWidget()
        self.recent.setMaximumHeight(180)
        self.recent.itemDoubleClicked.connect(
            lambda item: self.open_project(Path(item.data(Qt.ItemDataRole.UserRole)))
        )
        box.addWidget(self.recent)
        layout.addWidget(projects)
        self.audio_card, box = card("Connect your audio interface")
        form = QFormLayout()
        self.device = QComboBox()
        device_row = QHBoxLayout()
        device_row.addWidget(self.device, 1)
        device_row.addWidget(button("Refresh", self.refresh_devices))
        form.addRow("Interface", device_row)
        self.output_channel, self.input_channel, self.loop_output, self.loop_input = [
            QComboBox() for _ in range(4)
        ]
        for spin in (
            self.output_channel,
            self.input_channel,
            self.loop_output,
            self.loop_input,
        ):
            spin.addItem("Channel name unavailable", 1)
            spin.addItem("Channel name unavailable", 2)
        self.loop_output.setCurrentIndex(1)
        self.loop_input.setCurrentIndex(1)
        form.addRow("Output → reamp / rig", self.output_channel)
        form.addRow("Input ← load box / microphone", self.input_channel)
        self.loop_enabled = QCheckBox(
            "Use a separate clean loopback for precise alignment"
        )
        self.loop_enabled.setChecked(True)
        form.addRow(self.loop_enabled)
        form.addRow("Loopback output → cable", self.loop_output)
        form.addRow("Loopback input ← cable", self.loop_input)
        self.buffer = QComboBox()
        self.buffer.addItem("Driver managed", 0)
        form.addRow("Buffer size", self.buffer)
        box.addLayout(form)
        self.diagram = RoutingDiagram()
        box.addWidget(self.diagram)
        row = QHBoxLayout()
        self.route_button = button("Test connection", self.test_route)
        self.route_status = label(
            "Connect the rig as shown, then check the return signal.", "muted"
        )
        row.addWidget(self.route_button)
        row.addWidget(self.route_status, 1)
        box.addLayout(row)
        self.audio_hint = label(
            "Tone3000 standard sweep and the fixed repository validation signal are bundled offline.\n"
            "Use your interface's normal driver; no Python, Anaconda or CUDA toolkit is required.",
            "muted",
        )
        box.addWidget(self.audio_hint)
        layout.addWidget(self.audio_card)
        self.calibration_card, box = card("Level calibration · optional")
        self.input_calibration, self.output_calibration = QCheckBox(
            "Send level at capture output setting"
        ), QCheckBox("Return level at 0 dBFS")
        self.input_dbu, self.output_dbu = QDoubleSpinBox(), QDoubleSpinBox()
        form = QFormLayout()
        for checkbox, spin in (
            (self.input_calibration, self.input_dbu),
            (self.output_calibration, self.output_dbu),
        ):
            spin.setRange(-200, 200)
            spin.setDecimals(3)
            spin.setValue(12)
            spin.setSuffix(" dBu")
            form.addRow(checkbox, spin)
            checkbox.toggled.connect(self.save_calibration)
            spin.valueChanged.connect(self.save_calibration)
        box.addLayout(form)
        self.capture_level = OutputLevelEdit(0.0)
        self.capture_level.editingFinished.connect(self.save_capture_level)
        attenuation_form = QFormLayout()
        attenuation_form.addRow(
            "Capture / validation output (dBFS peak)", self.capture_level
        )
        box.addLayout(attenuation_form)
        box.addWidget(button("Calibrate send and return…", self.calibrate_levels))
        box.addWidget(
            label(
                "These values are embedded in the exported .nam model for plugin level calibration.",
                "muted",
            )
        )
        layout.addWidget(self.calibration_card)
        self.device.currentIndexChanged.connect(self._device_changed)
        for spin in (
            self.input_channel,
            self.output_channel,
            self.loop_input,
            self.loop_output,
        ):
            spin.currentIndexChanged.connect(self.save_audio)
        self.loop_enabled.toggled.connect(self.save_audio)
        self.buffer.currentIndexChanged.connect(self.save_audio)

    def _controls(self, layout):
        layout.addWidget(label("DEFINE THE RANGE", "eyebrow"))
        layout.addWidget(label("Which controls shape your tone?", "title"))
        layout.addWidget(
            label(
                "Add every knob and switch you want to learn. The capture plan covers each switch combination.",
                "muted",
            )
        )
        kind_card, kind_box = card("Model type")
        self.capture_kind = QComboBox()
        self.capture_kind.addItem("Parametric — HyperA2 Full + Lite", "parametric")
        self.capture_kind.addItem("Static batch — A2", "static")
        kind_box.addWidget(self.capture_kind)
        self.static_size = QComboBox()
        self.static_size.addItem("A2 Full (default)", "full")
        self.static_size.addItem("A2 Lite", "lite")
        kind_box.addWidget(self.static_size)
        layout.addWidget(kind_card)
        self.capture_kind.currentIndexChanged.connect(self.save_preferences)
        self.static_size.currentIndexChanged.connect(self.save_preferences)
        self.control_card, box = card("Rig controls")
        self.control_table = QTableWidget(0, 4)
        self.control_table.setHorizontalHeaderLabels(
            ["Name", "Type", "Range / positions", "Default"]
        )
        self.control_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.control_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.control_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.control_table.setMinimumHeight(205)
        self.control_table.cellDoubleClicked.connect(lambda *_: self.edit_control())
        box.addWidget(self.control_table)
        row = QHBoxLayout()
        row.addWidget(button("+ Add control", self.add_control, True))
        row.addWidget(button("Edit selected", self.edit_control))
        row.addWidget(button("Remove", self.remove_control))
        row.addStretch()
        box.addLayout(row)
        layout.addWidget(self.control_card)
        self.signal_card, box = card("Capture input signal")
        signal_form = QFormLayout()
        self.signal_mode = QComboBox()
        self.signal_mode.addItem("Standard — bundled TONE3000 sweep", "standard")
        self.signal_mode.addItem(
            "Custom — choose a training WAV from this computer", "custom"
        )
        signal_form.addRow("Training input", self.signal_mode)
        self.custom_label = label("No custom file selected", "muted")
        signal_form.addRow("Custom file", self.custom_label)
        self.custom_button = button("Choose custom training WAV…", self.import_inputs)
        signal_form.addRow(self.custom_button)
        signal_form.addRow(
            "Validation input",
            label(
                "validation.wav · fixed original from the trainer repository · bundled in the installer",
                "muted",
            ),
        )
        box.addLayout(signal_form)
        box.addWidget(
            label(
                "Custom input must be mono at 48 kHz to match the fixed validation signal. Validation is always captured through your rig separately.",
                "muted",
            )
        )
        self.signal_mode.currentIndexChanged.connect(self.save_preferences)
        layout.addWidget(self.signal_card)
        self.metadata_card, metadata_box = card(
            "Model metadata / defaults for static captures"
        )
        self.metadata_editor = MetadataEditor()
        self.metadata_editor.changed.connect(self.save_metadata)
        metadata_box.addWidget(self.metadata_editor)
        layout.addWidget(self.metadata_card)
        self.plan_card, box = card("Capture plan & learning method")
        form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem("Normal — capture the plan, then train", "normal")
        self.mode.addItem(
            "Active learning — suggest informative captures in rounds",
            "active_learning",
        )
        self.plan_form = form
        form.addRow("Training method", self.mode)
        form.addRow(
            "Model format",
            self._model_format_label(),
        )
        self.n_train, self.n_validation = QSpinBox(), QSpinBox()
        self.n_train.setRange(1, 1000000)
        self.n_validation.setRange(0, 1000000)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.corners = QCheckBox(
            "Include range-extreme captures for every switch combination"
        )
        self.corners.setChecked(True)
        form.addRow("Training LHS captures", self.n_train)
        form.addRow("Validation captures", self.n_validation)
        form.addRow("Reproducible plan seed", self.seed)
        form.addRow(self.corners)
        self.rounds, self.additions = QSpinBox(), QSpinBox()
        self.rounds.setRange(2, 10)
        self.additions.setRange(1, 100)
        form.addRow("Active-learning rounds", self.rounds)
        form.addRow("Additional captures per round", self.additions)
        box.addLayout(form)
        self.recommendation = label("", "muted")
        box.addWidget(self.recommendation)
        box.addWidget(
            button("Use recommended counts", lambda: self.refresh_recommendation(True))
        )
        box.addWidget(
            label(
                "Active learning trains independent HyperA2 members and compares their Full/Lite predictions to suggest new settings.\n"
                "You return to the capture step for each new batch. Every exported model remains HyperA2 Full + Lite.",
                "muted",
            )
        )
        layout.addWidget(self.plan_card)
        for field in (
            self.n_train,
            self.n_validation,
            self.seed,
            self.rounds,
            self.additions,
        ):
            field.valueChanged.connect(self.save_preferences)
        self.corners.toggled.connect(self.save_preferences)
        self.mode.currentIndexChanged.connect(self.save_preferences)

    def _capture(self, layout):
        layout.addWidget(label("BRING YOUR RIG INTO THE MODEL", "eyebrow"))
        layout.addWidget(label("Set. Record. Repeat.", "title"))
        self.capture_summary = label("Generate a capture plan to begin.", "muted")
        layout.addWidget(self.capture_summary)
        self.static_card, static_box = card("Static capture")
        self.static_name = QLineEdit()
        self.static_name.setPlaceholderText("Name for this A2 model")
        self.static_name.editingFinished.connect(self.rename_static)
        static_box.addWidget(self.static_name)
        static_box.addWidget(button("Add static capture…", self.add_static))
        self.static_metadata = MetadataEditor(include_name=False)
        self.static_metadata.changed.connect(self.save_static_metadata)
        self.static_metadata.setVisible(False)
        self.metadata_toggle = button(
            "Edit metadata for this capture…",
            lambda: self.static_metadata.setVisible(
                not self.static_metadata.isVisible()
            ),
        )
        static_box.addWidget(self.metadata_toggle)
        static_box.addWidget(self.static_metadata)
        layout.addWidget(self.static_card)
        self.scene = ControlScene()
        layout.addWidget(self.scene)
        row = QHBoxLayout()
        self.rig_ready = QCheckBox("My rig matches the control positions shown above")
        self.record_button = button("Record this setting", self.record, True)
        self.cancel_button = button("Cancel capture", self.cancel_capture)
        row.addWidget(self.rig_ready, 1)
        row.addWidget(self.record_button)
        row.addWidget(self.cancel_button)
        layout.addLayout(row)
        self.capture_progress = QProgressBar()
        self.capture_progress.setRange(0, 100)
        layout.addWidget(self.capture_progress)
        self.capture_status = label(
            "Select a capture below. Set the real controls to match the diagram.",
            "muted",
        )
        layout.addWidget(self.capture_status)
        self.capture_table = QTableWidget(0, 6)
        self.capture_table.setHorizontalHeaderLabels(
            [
                "Split",
                "Control settings",
                "Status",
                "Latency",
                "Peak (dBFS)",
                "Warnings",
            ]
        )
        header = self.capture_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(55)
        for column, width in enumerate((70, 220, 85, 200, 110, 230)):
            self.capture_table.setColumnWidth(column, width)
        saved = self.settings.value("capture_columns")
        if saved is not None:
            header.restoreState(saved)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.sectionResized.connect(self.save_capture_columns)
        self.capture_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.capture_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.capture_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.capture_table.setMinimumHeight(270)
        self.capture_table.itemSelectionChanged.connect(self.select_capture)
        layout.addWidget(self.capture_table)
        self.rig_ready.toggled.connect(self.update_enabled)

    def _training(self, layout):
        layout.addWidget(label("TURN CAPTURES INTO AN INSTRUMENT", "eyebrow"))
        self.training_title = label("Train your HyperA2.", "title")
        layout.addWidget(self.training_title)
        self.training_status = label(
            "Complete the capture plan to unlock training.", "muted"
        )
        layout.addWidget(self.training_status)
        self.training_settings, box = card("Training settings")
        form = QFormLayout()
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100000)
        self.epochs.setValue(200)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(4)
        self.accelerator = QComboBox()
        self.accelerator.addItem(
            "Automatic — use a supported GPU when available", "auto"
        )
        self.accelerator.addItem("CPU — works on every supported computer", "cpu")
        form.addRow("Epochs per model / round", self.epochs)
        form.addRow("Batch size", self.batch_size)
        form.addRow("Training device", self.accelerator)
        box.addLayout(form)
        box.addWidget(
            label(
                "Full and Lite are trained together. Pause saves the optimizer and training state at the end of the current epoch.",
                "muted",
            )
        )
        layout.addWidget(self.training_settings)
        if sys.platform != "darwin":
            self.gpu_setup_button = button("NVIDIA GPU support…", self.configure_gpu)
            box.addWidget(self.gpu_setup_button)
        grid = QGridLayout()
        self.metrics = {}
        for index, (name, text) in enumerate(
            (
                ("epoch", "CURRENT EPOCH"),
                ("esr", "BEST VALIDATION ESR"),
                ("elapsed", "ELAPSED"),
                ("eta", "ESTIMATED REMAINING"),
            )
        ):
            frame, metric_layout = card(text)
            metric = label("—", "metric")
            metric_layout.addWidget(metric)
            self.metrics[name] = metric
            grid.addWidget(frame, index // 2, index % 2)
        layout.addLayout(grid)
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 1000)
        layout.addWidget(self.train_progress)
        row = QHBoxLayout()
        self.train_button = button("Train", self.start_training, True)
        self.pause_button = button("Pause & save", self.pause_training)
        self.stop_button = button("Stop", self.stop_training)
        self.models_button = button("Open model folder", self.open_models)
        row.addWidget(self.train_button)
        row.addWidget(self.pause_button)
        row.addWidget(self.stop_button)
        row.addStretch()
        row.addWidget(self.models_button)
        layout.addLayout(row)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(180)
        self.details.setVisible(False)
        layout.addWidget(
            button(
                "Show / hide training details",
                lambda: self.details.setVisible(not self.details.isVisible()),
            )
        )
        layout.addWidget(self.details)
        self.epochs.valueChanged.connect(self.save_preferences)
        self.batch_size.valueChanged.connect(self.save_preferences)
        self.accelerator.currentIndexChanged.connect(self.save_preferences)

    def busy(self):
        return (
            self.worker is not None and self.worker.isRunning()
        ) or self.process is not None

    def notify_error(self, error):
        QMessageBox.warning(self, "ParaNAM Trainer", str(error))

    def navigate(self, index):
        if index > 0 and not self.workspace:
            self.notify_error("Create or open a project first.")
            return
        if index >= 2 and not self.busy():
            try:
                if self.workspace.plan_changed() and not self.apply_plan():
                    return
            except ValueError as error:
                self.notify_error(error)
                return
        self.pages.setCurrentIndex(index)
        for i, step in enumerate(self.steps):
            step.setChecked(i == index)
        self.back_button.setEnabled(index > 0)
        self.next_button.setVisible(index < 3)
        self.next_button.setText("Train →" if index == 2 else "Next →")
        self.update_enabled()

    def back(self):
        self.navigate(max(0, self.pages.currentIndex() - 1))

    def next(self):
        index = self.pages.currentIndex()
        if index == 1 and not self.apply_plan():
            return
        if index == 2 and not self.workspace.training_ready():
            self.notify_error("Complete every capture in the plan first.")
            return
        self.navigate(min(3, index + 1))

    def _refresh_recent(self):
        self.recent.clear()
        paths = self.settings.value("recent", [])
        if isinstance(paths, str):
            paths = [paths]
        for value in paths:
            path = Path(value)
            metadata = (
                read_json(path / "capture_project.json", {}) if path.exists() else {}
            )
            item = QListWidgetItem(f"{metadata.get('name') or path.name}\n{path}")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.recent.addItem(item)

    def new_project(self):
        if self.busy():
            return
        name, accepted = QInputDialog.getText(self, "New project", "Project name:")
        if not accepted or not name.strip():
            return
        documents = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        )
        parent = QFileDialog.getExistingDirectory(
            self, "Choose where to save the project", documents
        )
        if not parent:
            return
        safe_name = re.sub(r"[^\w .-]", "_", name).strip(" .") or "ParaNAM Project"
        try:
            directory = Path(parent) / safe_name
            workspace = Workspace.create(directory, name.strip(), self.audio_settings())
            self._attach(workspace)
        except Exception as error:
            self.notify_error(error)

    def open_project_dialog(self):
        path = QFileDialog.getExistingDirectory(
            self, "Open ParaNAM or NAM capture project"
        )
        if path:
            self.open_project(Path(path))

    def open_project(self, directory):
        if self.busy():
            return
        try:
            self._attach(Workspace(directory))
        except Exception as error:
            self.notify_error(error)

    def _attach(self, workspace):
        self.close_audio()
        self.workspace = workspace
        self.job = training.current_job(workspace.directory)
        self.project_label.setText(
            f"{workspace.project.name or workspace.directory.name}  ·  Saved locally"
        )
        paths = self.settings.value("recent", [])
        if isinstance(paths, str):
            paths = [paths]
        self.settings.setValue(
            "recent",
            [str(workspace.directory)]
            + [p for p in paths if p != str(workspace.directory)][:11],
        )
        self._refresh_recent()
        self._loading = True
        try:
            prefs = workspace.preferences
            self.capture_kind.setCurrentIndex(
                max(0, self.capture_kind.findData(prefs["capture_type"]))
            )
            self.static_size.setCurrentIndex(
                max(0, self.static_size.findData(prefs["static_size"]))
            )
            self.metadata_editor.set_metadata(workspace.project.user_metadata)
            for field, key in (
                (self.n_train, "n_train"),
                (self.n_validation, "n_validation"),
                (self.seed, "seed"),
                (self.rounds, "rounds"),
                (self.additions, "additions"),
                (self.epochs, "epochs"),
                (self.batch_size, "batch_size"),
            ):
                field.setValue(prefs[key])
            self.corners.setChecked(prefs["corners"])
            self.mode.setCurrentIndex(max(0, self.mode.findData(prefs["mode"])))
            self.signal_mode.setCurrentIndex(
                max(0, self.signal_mode.findData(prefs["input_mode"]))
            )
            self.custom_label.setText(
                prefs.get("custom_input") or "No custom file selected"
            )
            self.accelerator.setCurrentIndex(
                max(0, self.accelerator.findData(prefs["device"]))
            )
            audio = workspace.project.audio
            matched_device = False
            for i in range(self.device.count()):
                info = self.device.itemData(i)
                if (
                    info
                    and info.name == audio.input_device
                    and info.host_api == audio.host_api
                ):
                    self.device.setCurrentIndex(i)
                    matched_device = True
                    break
            if audio.input_device and not matched_device:
                self.device.addItem(f"Unavailable: {audio.input_device}", None)
                self.device.setCurrentIndex(self.device.count() - 1)
            self._populate_audio_controls(self.device.currentData())
            for field, value in (
                (self.input_channel, audio.input_channel),
                (self.output_channel, audio.output_channel),
                (self.loop_input, audio.loopback_input_channel or 2),
                (self.loop_output, audio.loopback_output_channel or 2),
            ):
                index = field.findData(value)
                if index < 0:
                    field.addItem(f"Unavailable channel {value}", value)
                    index = field.count() - 1
                field.setCurrentIndex(index)
            self.loop_enabled.setChecked(audio.loopback_enabled)
            index = self.buffer.findData(audio.blocksize)
            if index >= 0:
                self.buffer.setCurrentIndex(index)
            self.capture_level.set_level(prefs["capture_level_dbfs"])
            calibration = prefs.get("pending_calibration", {})
            for check, field, value in (
                (
                    self.input_calibration,
                    self.input_dbu,
                    calibration.get(
                        "input_level_dbu", workspace.project.input_level_dbu
                    ),
                ),
                (
                    self.output_calibration,
                    self.output_dbu,
                    calibration.get(
                        "output_level_dbu", workspace.project.output_level_dbu
                    ),
                ),
            ):
                check.setChecked(value is not None)
                field.setValue(12 if value is None else value)
        finally:
            self._loading = False
        self.save_preferences()
        self.refresh_controls()
        self.refresh_captures()
        self._diagram()
        self.refresh_training()
        self.update_enabled()

    def refresh_devices(self):
        if self.busy():
            return
        self.close_audio()
        selected = self.device.currentData()
        self._loading = True
        try:
            self.device.clear()
            self._devices = [
                d
                for d in list_devices(refresh=True)
                if d.max_input_channels > 0 and d.max_output_channels > 0
            ]
            for info in self._devices:
                self.device.addItem(f"{info.name}  ·  {info.host_api}", info)
            if selected:
                for i, info in enumerate(self._devices):
                    if (
                        info.name == selected.name
                        and info.host_api == selected.host_api
                    ):
                        self.device.setCurrentIndex(i)
                        break
        except Exception as error:
            self.route_status.setText(str(error))
        finally:
            self._loading = False
        self._device_changed()

    def _device_changed(self, *_):
        self.close_audio()
        info = self.device.currentData()
        self._populate_audio_controls(info)
        self.save_audio()

    def _populate_audio_controls(self, info):
        if info:
            result = capabilities(info)
            same_device = getattr(self, "_audio_device_key", None) == (
                info.host_api,
                info.name,
            )
            self._audio_device_key = (info.host_api, info.name)
            previous_buffer = self.buffer.currentData() if same_device else None
            for combo, names, default in (
                (self.input_channel, result.inputs, 1),
                (self.loop_input, result.inputs, 2),
                (self.output_channel, result.outputs, 1),
                (self.loop_output, result.outputs, 2),
            ):
                previous = combo.currentData() or default
                combo.blockSignals(True)
                combo.clear()
                for number, name in enumerate(names, 1):
                    combo.addItem(name, number)
                combo.setCurrentIndex(max(0, combo.findData(previous)))
                combo.blockSignals(False)
            self.buffer.blockSignals(True)
            self.buffer.clear()
            for size in result.buffers:
                self.buffer.addItem(f"{size} samples", size)
            if not result.buffers:
                self.buffer.addItem("Driver managed (size not exposed)", 0)
            wanted = (
                previous_buffer
                if previous_buffer in result.buffers
                else result.default_buffer
            )
            self.buffer.setCurrentIndex(max(0, self.buffer.findData(wanted)))
            self.buffer.setToolTip(result.note)
            self.buffer.blockSignals(False)
            if min(info.max_input_channels, info.max_output_channels) < 2:
                self.loop_enabled.setChecked(False)

    def audio_settings(self):
        info = self.device.currentData()
        return AudioSettingsModel(
            host_api=info.host_api if info else None,
            input_device=info.name if info else None,
            output_device=info.name if info else None,
            input_channel=self.input_channel.currentData() or 1,
            output_channel=self.output_channel.currentData() or 1,
            loopback_input_channel=(
                self.loop_input.currentData() if self.loop_enabled.isChecked() else None
            ),
            loopback_output_channel=(
                self.loop_output.currentData()
                if self.loop_enabled.isChecked()
                else None
            ),
            blocksize=self.buffer.currentData() or 0,
        )

    def _diagram(self):
        self.diagram.channels = (
            self.output_channel.currentData() or 1,
            self.input_channel.currentData() or 1,
            self.loop_output.currentData() or 2,
            self.loop_input.currentData() or 2,
        )
        self.diagram.loopback = self.loop_enabled.isChecked()
        self.diagram.update()
        self.loop_input.setEnabled(self.loop_enabled.isChecked())
        self.loop_output.setEnabled(self.loop_enabled.isChecked())

    def save_audio(self, *_):
        if self._loading or self.busy():
            return
        self.close_audio()
        self._diagram()
        if self.workspace:
            self.workspace.project.audio = self.audio_settings()
            self.workspace.save()

    def save_calibration(self, *_):
        if self._loading or not self.workspace or self.busy():
            return
        send = self.input_dbu.value() if self.input_calibration.isChecked() else None
        returned = (
            self.output_dbu.value() if self.output_calibration.isChecked() else None
        )
        if (
            self.workspace.preferences["capture_level_dbfs"]
            != self.workspace.project.capture_level_dbfs
        ):
            self.workspace.preferences["pending_calibration"] = dict(
                input_level_dbu=send, output_level_dbu=returned
            )
        else:
            self.workspace.preferences.pop("pending_calibration", None)
            self.workspace.project.input_level_dbu = send
            self.workspace.project.output_level_dbu = returned
        self.workspace.save()
        self.refresh_training()

    def save_capture_level(self, *_):
        if self._loading or self.busy():
            return
        try:
            level = self.capture_level.level()
            if (
                self.workspace
                and level != self.workspace.preferences["capture_level_dbfs"]
            ):
                self.workspace.preferences["capture_level_dbfs"] = level
                # The send metadata describes THIS attenuated capture setting.
                # Changing it requires a fresh voltage measurement.
                self.input_calibration.setChecked(False)
                self.save_calibration()
                self.workspace.save()
                self.close_audio()
                self.refresh_training()
        except ValueError as error:
            self.notify_error(error)

    def _set_capture_level(self, level):
        self.capture_level.set_level(level)
        self.save_capture_level()

    def _apply_level_calibration(self, check, field, value):
        self._loading = True
        try:
            field.setValue(value)
            check.setChecked(True)
        finally:
            self._loading = False
        self.save_calibration()

    def calibrate_levels(self):
        if not self.workspace or not self.device.currentData():
            self.notify_error(
                "Create/open a project and select your audio interface before calibrating."
            )
            return
        try:
            self.close_audio()
            dialog = LevelCalibrationDialog(
                self.device.currentData(),
                self.input_channel.currentData(),
                self.output_channel.currentData(),
                self.buffer.currentData() or 0,
                self.capture_level.level(),
                self.workspace.preferences.get("calibration_frequency", 1000.0),
                self,
                channel_names=(
                    self.input_channel.currentText(),
                    self.output_channel.currentText(),
                ),
            )
            dialog.level_changed.connect(self._set_capture_level)
            dialog.send_calibrated.connect(
                lambda value: self._apply_level_calibration(
                    self.input_calibration, self.input_dbu, value
                )
            )
            dialog.return_calibrated.connect(
                lambda value: self._apply_level_calibration(
                    self.output_calibration, self.output_dbu, value
                )
            )
            dialog.exec()
            self.workspace.preferences["calibration_frequency"] = (
                dialog.frequency.value()
            )
            self.workspace.save()
        except ValueError as error:
            self.notify_error(error)

    def import_inputs(self):
        if not self.workspace or self.busy():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose custom training WAV (mono, 48 kHz)", "", "WAV audio (*.wav)"
        )
        if not path:
            return
        try:
            import uuid
            import wavio

            wav = wavio.read(path)
            if wav.data.shape[1] != 1 or wav.rate != 48000:
                raise ValueError(
                    "Custom training input must be mono at 48 kHz, matching the fixed validation file."
                )
            if len(wav.data) < 48000:
                raise ValueError(
                    "The custom training signal must be at least one second long."
                )
            folder = self.workspace.directory / "signals"
            folder.mkdir(exist_ok=True)
            relative = "signals/custom-" + uuid.uuid4().hex[:8] + ".wav"
            shutil.copy2(path, self.workspace.directory / relative)
            self.workspace.preferences["custom_input"] = relative
            self.custom_label.setText(Path(path).name)
            self.workspace.save()
        except Exception as error:
            self.notify_error(error)

    def close_audio(self):
        if self.session:
            self.session.close_audio()
            self.session = None

    def _get_session(self):
        if self.session is None:
            if self.workspace:
                self.session = CaptureSession(
                    self.workspace.project, self.workspace.directory
                )
            else:
                project = CaptureProject(
                    knobs=[], audio=self.audio_settings(), sample_rate=48000
                )
                cache = Path(
                    QStandardPaths.writableLocation(
                        QStandardPaths.StandardLocation.CacheLocation
                    )
                )
                cache.mkdir(parents=True, exist_ok=True)
                self.session = CaptureSession(project, cache)
        return self.session

    def run_audio(self, operation, success):
        if self.busy():
            return
        self.cancel_token = CancelToken()
        self.worker = SessionWorker(operation, self.cancel_token, self)
        self.worker.progress.connect(
            lambda value: self.capture_progress.setValue(int(value * 100))
        )
        self.worker.succeeded.connect(success)
        self.worker.failed.connect(self.notify_error)
        self.worker.cancelled.connect(
            lambda: self.capture_status.setText(
                "Capture cancelled; saved captures are unchanged."
            )
        )
        self.worker.finished.connect(self._audio_finished)
        self.worker.start()
        self.update_enabled()

    def _audio_finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.cancel_token = None
        self.refresh_captures()
        self.update_enabled()
        if self._close_after_pause:
            self.close()

    def test_route(self):
        if not self.device.currentData():
            self.notify_error(
                "Connect a supported duplex audio interface, then refresh the device list."
            )
            return
        try:
            self.save_audio()
            session = self._get_session()
            self.route_status.setText("Checking return signal and timing…")
            self.run_audio(
                lambda progress, cancel: session.route_test(
                    sample_rate=session.project.sample_rate or 48000,
                    progress=progress,
                    cancel=cancel,
                ),
                self._route_done,
            )
        except Exception as error:
            self.notify_error(error)

    def _route_done(self, result):
        self.route_status.setText(
            ("Connection verified" if result.ok else "Check the connections")
            + f" · Return peak {result.peak:.1f} dBFS"
            + (" · Clean loopback missing" if result.loopback_failed else "")
        )

    def refresh_controls(self):
        if not self.workspace:
            return
        self.control_table.setRowCount(len(self.workspace.draft_controls))
        for row, data in enumerate(self.workspace.draft_controls):
            c = control_from_dict(data)
            if isinstance(c, SwitchSpec):
                cells = (
                    c.name,
                    "Switch",
                    " / ".join(c.enum_names),
                    c.enum_names[c.default],
                )
            else:
                cells = (
                    c.name,
                    "Knob",
                    f"{c.min:g} … {c.max:g}  ·  step {c.step:g}"
                    + ("  ·  Gain/Drive" if c.is_gain else ""),
                    f"{c.default:g}",
                )
            for col, value in enumerate(cells):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.control_table.setItem(row, col, item)
        self.refresh_recommendation()

    def _control_dialog(self, index=None):
        if not self.workspace or self.busy():
            return
        control = (
            control_from_dict(self.workspace.draft_controls[index])
            if index is not None
            else None
        )
        dialog = ControlDialog(self, control)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        previous = list(self.workspace.draft_controls)
        if index is None:
            self.workspace.draft_controls.append(dialog.result_control.to_dict())
        else:
            self.workspace.draft_controls[index] = dialog.result_control.to_dict()
        try:
            self.workspace.controls()
        except ValueError as error:
            self.workspace.draft_controls = previous
            self.notify_error(error)
            return
        self.refresh_controls()
        self.refresh_recommendation(True)
        self.workspace.save()

    def add_control(self):
        self._control_dialog()

    def edit_control(self):
        index = self.control_table.currentRow()
        if index >= 0:
            self._control_dialog(index)

    def remove_control(self):
        index = self.control_table.currentRow()
        if self.workspace and index >= 0 and not self.busy():
            self.workspace.draft_controls.pop(index)
            self.refresh_controls()
            self.refresh_recommendation(True)
            self.workspace.save()

    def refresh_recommendation(self, autofill=False):
        if not self.workspace:
            return
        if self.capture_kind.currentData() == "static":
            count = len(self.workspace.preferences["static_captures"])
            self.recommendation.setText(
                f"{count} named A2 model(s). Each records its own training signal and the fixed validation signal. Add and name captures in the next step."
            )
            return
        controls = [control_from_dict(c) for c in self.workspace.draft_controls]
        if not controls:
            self.recommendation.setText("Add at least one control to build a plan.")
            return
        train, val, combinations = capture_recommendations(controls)
        if all(isinstance(c, SwitchSpec) for c in controls):
            val = combinations
        count = corner_capture_count(controls)
        self.recommendation.setText(
            f"Suggested: {train} training + {val} validation captures, plus up to {count} corners.\n"
            f"{combinations} switch combination(s). Each enabled split must cover every combination. Counts are starting guidelines, not an accuracy guarantee."
        )
        if autofill:
            self.n_train.setValue(train)
            self.n_validation.setValue(val)

    def save_preferences(self, *_):
        if self._loading or not self.workspace or self.busy():
            return
        prefs = self.workspace.preferences
        for field, key in (
            (self.n_train, "n_train"),
            (self.n_validation, "n_validation"),
            (self.seed, "seed"),
            (self.rounds, "rounds"),
            (self.additions, "additions"),
            (self.epochs, "epochs"),
            (self.batch_size, "batch_size"),
        ):
            prefs[key] = field.value()
        prefs.update(
            capture_type=self.capture_kind.currentData(),
            static_size=self.static_size.currentData(),
            mode=(
                "normal"
                if self.capture_kind.currentData() == "static"
                else self.mode.currentData()
            ),
            device=self.accelerator.currentData(),
            corners=self.corners.isChecked(),
            input_mode=self.signal_mode.currentData(),
        )
        static = prefs["capture_type"] == "static"
        self.model_badge.setText(
            "A2 static batch" if static else "HyperA2\nFull + Lite, together."
        )
        self.control_card.setVisible(not static)
        self.static_size.setVisible(static)
        self.model_format.setText(
            f"A2 {prefs['static_size'].title()} · one standard .nam per named capture"
            if static
            else "HyperA2 · Full + Lite nested in one .nam file"
        )
        self.mode.setEnabled(not static)
        for widget in (self.n_train, self.n_validation, self.seed, self.corners):
            widget.setEnabled(not static)
        self.training_title.setText(
            "Train your A2 batch." if static else "Train your HyperA2."
        )
        for widget in (
            self.mode,
            self.n_train,
            self.n_validation,
            self.seed,
            self.corners,
            self.rounds,
            self.additions,
        ):
            self.plan_form.setRowVisible(widget, not static)
        self.rounds.setEnabled(not static and prefs["mode"] == "active_learning")
        self.additions.setEnabled(prefs["mode"] == "active_learning")
        self.workspace.save()
        self.refresh_recommendation()
        self.custom_button.setEnabled(prefs["input_mode"] == "custom")

    def apply_plan(self):
        if not self.workspace or self.busy():
            return False
        self.save_preferences()
        try:
            if self.workspace.plan_changed():
                if self.workspace.project.captured_entries():
                    answer = QMessageBox.question(
                        self,
                        "Update the capture plan?",
                        "The controls or plan settings changed. Existing captures will be preserved in History, and a new plan will start. Continue?",
                    )
                    if answer != QMessageBox.StandardButton.Yes:
                        return False
                self.close_audio()
                self.workspace.generate_plan()
            self.refresh_captures()
            return True
        except Exception as error:
            self.notify_error(error)
            return False

    def _model_format_label(self):
        self.model_format = label(
            "HyperA2 · Full + Lite nested in one .nam file", "eyebrow"
        )
        return self.model_format

    def save_metadata(self, metadata):
        if self.workspace and not self._loading and not self.busy():
            self.workspace.project.user_metadata = dict(metadata)
            self.workspace.save()

    def save_static_metadata(self, metadata):
        if (
            self.workspace
            and self._static_edit_id
            and not self._loading
            and not self.busy()
        ):
            self.workspace.set_static_metadata(self._static_edit_id, metadata)

    def add_static(self):
        if not self.workspace or self.busy():
            return
        name, ok = QInputDialog.getText(
            self, "Add static capture", "Model/capture name"
        )
        if ok:
            try:
                capture = self.workspace.add_static_capture(name)
                self.refresh_captures()
                self.select_static_id(capture["id"])
            except Exception as error:
                self.notify_error(error)

    def rename_static(self):
        if self.workspace and self._static_edit_id and not self.busy():
            try:
                capture_id = self._static_edit_id
                self.workspace.rename_static_capture(
                    capture_id, self.static_name.text()
                )
                self.refresh_captures()
                self.select_static_id(capture_id)
            except Exception as error:
                self.notify_error(error)

    def select_static_id(self, capture_id):
        row = next(
            (
                i
                for i, e in enumerate(self._capture_entries)
                if e.capture_id == capture_id
            ),
            None,
        )
        if row is not None:
            self.capture_table.selectRow(row)
            self.select_capture()

    def save_capture_columns(self, *_):
        if self._capture_table_kind:
            self.settings.setValue(
                "capture_columns/" + self._capture_table_kind,
                self.capture_table.horizontalHeader().saveState(),
            )

    @staticmethod
    def _capture_metrics(entry):
        qa = entry.qa
        warnings = [
            m
            for m in (qa.messages if qa else [])
            if "loopback and amp-return" not in m.lower()
            and not (m.startswith("Delay ") and "project's typical" in m)
        ]
        delay = (
            entry.delay
            if entry.delay is not None
            else (qa.loopback_delay if qa else None)
        )
        latency = "—" if delay is None else f"{delay} samples"
        if qa and (qa.subsample_shift is not None or qa.loopback_delay is not None):
            latency += f" · subsample {(qa.subsample_shift or 0):+.3f}"
        peak = f"{qa.peak:.1f}" if qa and qa.peak is not None else "—"
        return latency, peak, " / ".join(warnings) or "—"

    def refresh_captures(self):
        if not self.workspace:
            return
        from .capture_order import order_captures

        project = self.workspace.project
        static = project.capture_type == "static"
        self.scene.setVisible(not static)
        self.static_card.setVisible(static)
        if self.worker is None:
            self.capture_status.setText(
                "Select a named capture below, set your rig, then record its training and validation signals."
                if static
                else "Select a capture below. Set the real controls to match the diagram."
            )
        self.rig_ready.setText(
            "My rig is set for this named capture"
            if static
            else "My rig matches the control positions shown above"
        )
        if static:
            groups = self.workspace.static_groups()
            entries = [
                next((e for e in group if e.status != "captured"), group[0])
                for group in groups
            ]
            headers = ["Name", "Status", "Latency", "Peak (dBFS)", "Warnings"]
        else:
            entries = order_captures(project.entries, project.control_specs())
            headers = [
                "Split",
                "Control settings",
                "Status",
                "Latency",
                "Peak (dBFS)",
                "Warnings",
            ]
        self._capture_entries = entries
        self.capture_table.blockSignals(True)
        header = self.capture_table.horizontalHeader()
        header.blockSignals(True)
        kind = "static" if static else "parametric"
        changed_kind = self._capture_table_kind != kind
        self._capture_table_kind = kind
        self.capture_table.setColumnCount(len(headers))
        self.capture_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.capture_table.setHorizontalHeaderLabels(headers)
        if changed_kind:
            sizes = (160, 85, 260, 120, 290) if static else (70, 220, 85, 200, 110, 230)
            for column, width in enumerate(sizes):
                header.resizeSection(column, width)
            saved = self.settings.value("capture_columns/" + kind)
            if saved is not None:
                header.restoreState(saved)
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.blockSignals(False)
        self.capture_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            if static:
                group = groups[row]
                metrics = [(e.split.title(), self._capture_metrics(e)) for e in group]
                columns = [
                    entry.capture_name or "Capture",
                    (
                        "Captured"
                        if all(e.status == "captured" for e in group)
                        else "Pending"
                    ),
                ]
                columns += [
                    " | ".join(
                        f"{split}: {values[i]}"
                        for split, values in metrics
                        if values[i] != "—"
                    )
                    or "—"
                    for i in range(3)
                ]
            else:
                params = ", ".join(
                    (
                        f"{name}={value:g}"
                        if isinstance(value, (float, int))
                        else f"{name}={value}"
                    )
                    for name, value in entry.params.items()
                )
                columns = [
                    entry.split.title(),
                    params,
                    entry.status.title(),
                    *self._capture_metrics(entry),
                ]
            for col, text in enumerate(columns):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.capture_table.setItem(row, col, item)
        self.capture_table.blockSignals(False)
        completed = sum(e.status == "captured" for e in project.entries)
        self.capture_summary.setText(
            f"{len(entries)} static model(s) · {completed} of {len(project.entries)} signals captured. Each model records training and validation without changing the rig between them."
            if static
            else f"{completed} of {len(entries)} captured · {len(entries) - completed} remaining. Ordered to change fewer controls and turn knobs less."
        )
        if entries:
            row = next(
                (i for i, e in enumerate(entries) if e.status != "captured"),
                len(entries) - 1,
            )
            self.capture_table.selectRow(row)
            self.select_capture()
        self.rig_ready.setChecked(False)
        self.update_enabled()

    def select_capture(self):
        if self.workspace and 0 <= self.capture_table.currentRow() < len(
            self._capture_entries
        ):
            entry = self._capture_entries[self.capture_table.currentRow()]
            if self.workspace.project.capture_type == "static":
                self._static_edit_id = entry.capture_id
                self.static_name.setText(entry.capture_name or "")
                self.static_metadata.set_metadata(
                    {**self.workspace.project.user_metadata, **entry.user_metadata}
                )
            else:
                self._static_edit_id = None
                self.scene.set_controls(
                    self.workspace.project.control_specs(), entry.params
                )
            self.rig_ready.setChecked(False)
            if self.workspace.project.capture_type == "static":
                self.record_button.setText(
                    "Re-record this capture"
                    if entry.status == "captured"
                    else "Record training + validation"
                )
                return
            self.record_button.setText(
                "Re-record this setting"
                if entry.status == "captured"
                else "Record this setting"
            )

    def record(self):
        if not self.workspace or self.busy() or not self.rig_ready.isChecked():
            return
        index = self.capture_table.currentRow()
        if index < 0 or index >= len(self._capture_entries):
            return
        try:
            if not self.device.currentData():
                raise ValueError(
                    "Select an available audio interface in Project & routing."
                )
            self.save_audio()
            problem = timebase_problem(self.workspace.project, self.workspace.directory)
            if problem:
                raise ValueError(problem)
            entry = self._capture_entries[index]
            static = self.workspace.project.capture_type == "static"
            group = (
                [
                    e
                    for e in self.workspace.project.entries
                    if e.capture_id == entry.capture_id
                ]
                if static
                else [entry]
            )
            captures = [e for e in group if e.status != "captured"] or group
            if any(e.status == "captured" for e in captures):
                backup = (
                    self.workspace.directory
                    / "history"
                    / ("recapture-" + str(time.time_ns()))
                )
                for relative in (e.y_path for e in captures):
                    source = self.workspace.directory / relative
                    if source.exists():
                        target = backup / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
            session = self._get_session()
            self.capture_status.setText(
                "Recording… Timing is checked at the beginning and end of this recording."
            )
            self.capture_progress.setValue(0)

            def capture_signals(progress, cancel):
                results = []
                for offset, capture in enumerate(captures):
                    results.append(
                        session.capture_entry(
                            capture,
                            progress=(
                                (
                                    lambda value, i=offset: progress(
                                        (i + value) / len(captures)
                                    )
                                )
                                if progress
                                else None
                            ),
                            cancel=cancel,
                        )
                    )
                return results if static else results[0]

            self.run_audio(capture_signals, self._capture_done)
        except Exception as error:
            self.notify_error(error)

    def _capture_done(self, qa):
        reports = qa if isinstance(qa, list) else [qa]
        self.capture_progress.setValue(100)
        self.capture_status.setText(f"Saved {len(reports)} signal recording(s)")
        self.refresh_captures()
        messages = [message for report in reports for message in report.messages]
        if messages:
            QMessageBox.warning(self, "Capture quality check", "\n".join(messages))

    def cancel_capture(self):
        if self.cancel_token:
            self.cancel_token.cancel()

    def start_training(self):
        if not self.workspace or self.busy():
            return
        if (
            self.job
            and training.job_progress(self.job).get("status") == "needs_captures"
            and not self.workspace.training_ready()
        ):
            self.navigate(2)
            return
        self.save_preferences()
        try:
            if self.workspace.plan_changed():
                raise ValueError(
                    "Apply the modified controls/plan in step 2 before training."
                )
            self.close_audio()
            existing = training.current_job(self.workspace.directory)
            resume = bool(
                existing
                and training.job_progress(existing).get("status")
                in ("paused", "running", "scoring", "failed")
                and training.job_can_resume(self.workspace, existing)
            )
            if (
                existing
                and not resume
                and training.job_progress(existing).get("status") == "paused"
            ):
                answer = QMessageBox.question(
                    self,
                    "Training settings changed",
                    "The paused checkpoint belongs to different settings or captures. Start a new training run? The previous run will be kept.",
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            self.job = training.make_job(self.workspace, resume=resume)
            process = QProcess(self)
            process.setWorkingDirectory(str(self.workspace.directory))
            process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            environment = QProcessEnvironment.systemEnvironment()
            if not getattr(sys, "frozen", False):
                import os

                source_root = str(Path(__file__).resolve().parent.parent)
                environment.insert(
                    "PYTHONPATH",
                    source_root + os.pathsep + environment.value("PYTHONPATH"),
                )
            process.setProcessEnvironment(environment)
            process.readyReadStandardOutput.connect(
                lambda: process.readAllStandardOutput()
            )
            process.finished.connect(self._training_finished)
            process.errorOccurred.connect(self._process_error)
            self.process = process
            args = ["--train-worker", str(self.job)]
            from . import gpu_runtime

            if self.workspace.preferences["device"] != "cpu" and gpu_runtime.ready():
                args.append("--gpu-runtime")
            if not getattr(sys, "frozen", False):
                args = ["-m", "paranam"] + args
            process.start(sys.executable, args)
            self.update_enabled()
        except Exception as error:
            self.notify_error(error)

    def configure_gpu(self):
        if self.busy():
            return
        from .gpu_setup import show_setup

        show_setup(parent=self)

    def pause_training(self):
        if self.job and self.process:
            training.command(self.job, "pause")
            self.training_status.setText(
                "Pause requested. Finishing this epoch, then saving a complete checkpoint…"
            )
            self.pause_button.setEnabled(False)

    def _process_error(self, _error):
        if self.process is None:
            return
        message = self.process.errorString()
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self.process.deleteLater()
            self.process = None
            if self.job:
                progress = training.job_progress(self.job)
                progress.update(status="failed", message=message)
                atomic_write_json(self.job / "progress.json", progress)
            self.refresh_training()
            self.update_enabled()
        self.notify_error(message)

    def stop_training(self):
        if self.job and self.process:
            training.command(self.job, "stop")
            self.training_status.setText(
                "Stopping training. Captures and previous saved checkpoints will be kept."
            )

    def _training_finished(self, *_):
        process = self.process
        self.process = None
        if process:
            process.deleteLater()
        if self.workspace:
            self.workspace = Workspace(self.workspace.directory)
            self.refresh_captures()
        self.refresh_training()
        self.update_enabled()
        if self._close_after_pause:
            self.close()

    def refresh_training(self):
        if not self.workspace:
            return
        self.job = training.current_job(self.workspace.directory)
        progress = training.job_progress(self.job)
        if not progress:
            self.update_enabled()
            return
        status = progress.get("status", "starting")
        message = progress.get("message", status.replace("_", " ").title())
        if self.process is None and status in ("running", "scoring", "starting"):
            message = "Previous session interrupted. Resume from the last complete saved epoch."
        self.training_status.setText(
            f"Round {progress.get('round', 1)} · Model {min(progress.get('member', 0) + 1, progress.get('members', 1))} / {progress.get('members', 1)}\n{message}"
        )
        epoch, epochs = progress.get("epoch", 0), progress.get("epochs", 1)
        self.metrics["epoch"].setText(f"{epoch} / {epochs}")
        esr = progress.get("best_esr")
        self.metrics["esr"].setText("—" if esr is None else f"{esr:.5f}")
        live = (
            max(0, time.time() - progress.get("updated_at", time.time()))
            if self.process is not None
            else 0
        )
        self.metrics["elapsed"].setText(duration(progress.get("elapsed", 0) + live))
        self.metrics["eta"].setText(duration(progress.get("eta")))
        fraction = (
            progress.get("member", 0)
            + (max(0, epoch - 1) + progress.get("batch_fraction", 0)) / max(epochs, 1)
        ) / max(1, progress.get("members", 1))
        self.train_progress.setValue(
            1000
            if status in ("completed", "needs_captures")
            else int(min(1, fraction) * 1000)
        )
        logfile = self.job / "worker.log"
        if logfile.exists():
            with logfile.open("rb") as stream:
                stream.seek(max(0, logfile.stat().st_size - 18000))
                content = stream.read().decode("utf-8", errors="replace")
                content = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", content).replace(
                    "\r", "\n"
                )
            if content != self._log_content:
                scrollbar = self.details.verticalScrollBar()
                position = scrollbar.value()
                follow = position >= scrollbar.maximum() - 2
                self.details.setPlainText(content)
                if follow:
                    self.details.moveCursor(QTextCursor.MoveOperation.End)
                else:
                    scrollbar.setValue(position)
                self._log_content = content
        self.update_enabled()

    def open_models(self):
        if self.workspace:
            directory = self.workspace.directory / "models"
            directory.mkdir(exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def update_enabled(self, *_):
        active = self.busy()
        present = self.workspace is not None
        for widget in (self.new_button, self.open_button, self.recent, self.audio_card):
            widget.setEnabled(not active)
        for widget in (
            self.control_card,
            self.plan_card,
            self.signal_card,
            self.calibration_card,
            self.training_settings,
        ):
            widget.setEnabled(present and not active)
        self.route_button.setEnabled(
            not active and self.device.currentData() is not None
        )
        self.record_button.setEnabled(
            present
            and not active
            and self.rig_ready.isChecked()
            and self.capture_table.currentRow() >= 0
        )
        self.rig_ready.setEnabled(present and not active)
        self.cancel_button.setEnabled(self.worker is not None)
        self.capture_table.setEnabled(not active)
        self.metadata_card.setEnabled(not active)
        self.static_card.setEnabled(not active)
        if hasattr(self, "gpu_setup_button"):
            self.gpu_setup_button.setEnabled(not active)
        ready = present and self.workspace.training_ready()
        self.next_button.setEnabled(
            present and not active and (self.pages.currentIndex() != 2 or ready)
        )
        self.train_button.setEnabled(ready and not active)
        self.pause_button.setEnabled(self.process is not None)
        self.stop_button.setEnabled(self.process is not None)
        self.models_button.setEnabled(
            present and (self.workspace.directory / "models").exists()
        )
        self.custom_button.setEnabled(
            present and not active and self.signal_mode.currentData() == "custom"
        )
        self.train_button.setText("Train")
        if (
            self.job
            and not active
            and training.job_progress(self.job).get("status") == "needs_captures"
        ):
            self.train_button.setText(
                "Train next round" if ready else "Capture suggested settings →"
            )
            self.train_button.setEnabled(present)
        if self.job and not active and ready:
            try:
                status = training.job_progress(self.job).get("status")
                if status in (
                    "paused",
                    "running",
                    "scoring",
                    "failed",
                ) and training.job_can_resume(self.workspace, self.job):
                    self.train_button.setText("Resume training")
            except (ValueError, OSError):
                pass
        if present:
            enabled = self.mode.currentData() == "active_learning" and not active
            self.rounds.setEnabled(enabled)
            self.additions.setEnabled(enabled)

    def closeEvent(self, event):
        if self.process is not None:
            if not self._close_after_pause:
                answer = QMessageBox.question(
                    self,
                    "Save training before closing?",
                    "Pause at the end of the current epoch and close when the checkpoint is safely saved? You can resume after restarting the computer.",
                )
                if answer != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
            self._close_after_pause = True
            self.pause_training()
            event.ignore()
            return
        if self.worker is not None:
            self._close_after_pause = True
            self.cancel_capture()
            event.ignore()
            return
        if self.workspace:
            self.workspace.save()
        self.close_audio()
        event.accept()

"""Seven-screen desktop workflow; audio and training never block the UI thread."""

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
    QSize,
    Signal,
)
from PySide6.QtGui import QDesktopServices, QColor, QTextCursor, QIcon
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
    QToolButton,
    QTabWidget,
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
    ControlScene,
    ControlDialog,
    MetadataEditor,
    load_font,
)
from . import training
from . import __version__
from .audio_devices import capabilities
from .calibration import OutputLevelEdit, LevelCalibrationDialog, gain_from_dbfs
from .signals import asset_path
from .queue_ui import QueueActions
from .play_ui import PlayActions
from .queue_controller import QueueController
from .ui_preferences import UIPreferences
from .ui_widgets import CaptureTable, RecentProjectDelegate, navigation_icon


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


class WizardWindow(UIPreferences, QueueActions, PlayActions, QMainWindow):
    capture_stage = Signal(str)

    def __init__(self):
        super().__init__()
        load_font()
        self.setWindowTitle("ParaNAM Trainer")
        self.setStyleSheet(
            THEME.replace("CHECKMARK_URL", asset_path("checkmark.svg").as_posix())
        )
        self.resize(1260, 840)
        self.setMinimumSize(1000, 700)
        self.settings = QSettings("DLC86", "ParaNAM Trainer")
        import os

        default_queue = (
            Path(
                QStandardPaths.writableLocation(
                    QStandardPaths.StandardLocation.AppLocalDataLocation
                )
            )
            / "training-queue"
        )
        if self.settings.format() == type(self.settings).Format.IniFormat:
            default_queue = Path(self.settings.fileName()).parent / "training-queue"
        queue_root = Path(os.environ.get("PARANAM_QUEUE_ROOT", str(default_queue)))
        self.queue_controller = QueueController(queue_root, self)
        self.selected_job_id = None
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
        self.queue_controller.changed.connect(self.refresh_training)
        self.queue_controller.failed.connect(self.notify_error)
        self._refresh_recent()
        self._restoring_setup = True
        self.refresh_devices()
        self.restore_startup_setup()
        self._restoring_setup = False
        self.capture_stage.connect(self.capture_status.setText)
        self.timer = QTimer(self)
        self.timer.setInterval(600)
        self.timer.timeout.connect(self.refresh_training)
        self.timer.start()
        self.navigate(0)

    def _build(self):
        central = QWidget(self)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        sidebar = QFrame(central)
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(256)
        rail = QVBoxLayout(sidebar)
        rail.setContentsMargins(20, 30, 18, 24)
        brand = QHBoxLayout()
        brand.setSpacing(10)
        self.shop_button = QToolButton()
        self.shop_button.setObjectName("brandLink")
        self.shop_button.setIcon(QIcon(str(asset_path("TTSLogo.png"))))
        self.shop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.shop_button.setToolTip("Visit The Tone Scientist shop")
        self.shop_button.setAccessibleName("The Tone Scientist shop")
        self.shop_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://shop.thetonescientist.com"))
        )
        self.brand_title = label("ParaNAM\nTrainer", "brand")
        self.version_label = label(f"VERSION {__version__}", "eyebrow")
        self.brand_title.setParent(sidebar)
        self.version_label.setParent(sidebar)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(4)
        brand_text.addWidget(self.brand_title)
        brand_text.addWidget(self.version_label)
        self.brand_title.ensurePolished()
        self.version_label.ensurePolished()
        logo_height = (
            self.brand_title.sizeHint().height()
            + self.version_label.sizeHint().height()
            + 4
        )
        self.shop_button.setIconSize(QSize(logo_height, logo_height))
        self.shop_button.setFixedSize(logo_height, logo_height)
        brand.addWidget(self.shop_button)
        brand.addLayout(brand_text, 1)
        rail.addLayout(brand)
        rail.addSpacing(38)
        self.steps = []
        for index, text in enumerate(
            (
                "Project",
                "Plan",
                "Capture",
                "Train",
                "Queue",
                "Compare",
            )
        ):
            step = button(f" {text}", lambda checked=False, i=index: self.navigate(i))
            step.setObjectName("step")
            step.setIcon(navigation_icon(index))
            step.setIconSize(QSize(24, 24))
            step.setCheckable(True)
            rail.addWidget(step)
            self.steps.append(step)
        rail.addStretch()
        self.settings_button = button(" Settings", lambda: self.navigate(6))
        self.settings_button.setObjectName("step")
        self.settings_button.setIcon(navigation_icon(6))
        self.settings_button.setIconSize(QSize(24, 24))
        self.settings_button.setCheckable(True)
        rail.addWidget(self.settings_button)
        self.steps.append(self.settings_button)
        root.addWidget(sidebar)
        right = QWidget()
        content = QVBoxLayout(right)
        content.setContentsMargins(30, 24, 30, 22)
        content.setSpacing(18)
        self.project_label = label("No project open", "muted")
        content.addWidget(self.project_label)
        self.pages = QStackedWidget()
        self.page_contents = []
        for index, builder in enumerate(
            (
                self._welcome,
                self._controls,
                self._capture,
                self._training,
                self._queue,
                self._play,
                self._settings,
            )
        ):
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(0, 0, 10, 0)
            layout.setSpacing(20)
            builder(layout)
            if index not in (0, 2, 4):
                layout.addStretch()
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.pages.addWidget(scroll)
            self.page_contents.append(page)
        content.addWidget(self.pages, 1)
        footer = QHBoxLayout()
        self.back_button = button("← Back", self.back)
        self.back_button.setFixedWidth(100)
        self.footer_note = label("Settings are saved as you work.", "muted")
        self.next_button = button("Next →", self.next, True)
        footer.addWidget(self.back_button)
        footer.addWidget(self.footer_note, 1)
        footer.addWidget(self.play_transport_bar)
        self.play_transport_bar.hide()
        footer.addStretch(1)
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
        self.recent.setItemDelegate(RecentProjectDelegate(self.recent))
        self.recent.setMinimumHeight(320)
        self.recent.itemDoubleClicked.connect(
            lambda item: self.open_project(Path(item.data(Qt.ItemDataRole.UserRole)))
        )
        box.addWidget(self.recent)
        layout.addWidget(projects)

    def _settings(self, layout):
        layout.addWidget(label("Settings", "title"))
        layout.addWidget(
            label("Audio routing and calibration are shared by every project.", "muted")
        )
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
        self.instrument_input, self.monitor_left, self.monitor_right = [
            QComboBox() for _ in range(3)
        ]
        for combo in (self.instrument_input, self.monitor_left, self.monitor_right):
            combo.addItem("Not configured", None)
        form.addRow("Instrument input ← guitar", self.instrument_input)
        form.addRow("Monitor out L → speakers / headphones", self.monitor_left)
        form.addRow("Monitor out R → speakers / headphones", self.monitor_right)
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
        row = QHBoxLayout()
        self.route_button = button("Test connection", self.test_route)
        self.route_status = label(
            "Connect the selected send and return channels, then test the return signal.",
            "muted",
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
        self.instrument_calibration = QCheckBox("Instrument input at 0 dBFS")
        self.instrument_dbu = QDoubleSpinBox()
        self.instrument_dbu.setRange(-200, 200)
        self.instrument_dbu.setDecimals(3)
        self.instrument_dbu.setSuffix(" dBu")
        form.addRow(self.instrument_calibration, self.instrument_dbu)
        self.instrument_calibration.toggled.connect(self.save_instrument_calibration)
        self.instrument_dbu.valueChanged.connect(self.save_instrument_calibration)
        self.capture_level.editingFinished.connect(self.save_capture_level)
        attenuation_form = QFormLayout()
        attenuation_form.addRow(
            "Capture / validation output (dBFS peak)", self.capture_level
        )
        box.addLayout(attenuation_form)
        box.addWidget(button("Calibrate send and return…", self.calibrate_levels))
        box.addWidget(
            label(
                "These values are embedded in the exported model (.pnam parametric / .nam static) for plugin level calibration.",
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
            self.instrument_input,
            self.monitor_left,
            self.monitor_right,
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
        self.capture_kind.addItem("Parametric", "parametric")
        self.capture_kind.addItem("Static", "static")
        kind_box.addWidget(self.capture_kind)
        layout.addWidget(kind_card)
        self.capture_kind.currentIndexChanged.connect(self.save_preferences)
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
        self.signal_card, box = card("Input signal")
        signal_form = QFormLayout()
        self.signal_mode = QComboBox()
        self.signal_mode.addItem("Standard — bundled TONE3000 sweep", "standard")
        self.signal_mode.addItem("TTS input v11 — bundled", "tts")
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
                "Custom input must be mono at 48 kHz and retain the standard initial/final validation segments (as in TTS v11). Training excludes the first 10 and final 9 seconds. Validation is also captured separately.",
                "muted",
            )
        )
        self.signal_mode.currentIndexChanged.connect(self.save_preferences)
        layout.insertWidget(layout.indexOf(self.control_card), self.signal_card)
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
        self.capture_table = CaptureTable(0, 6)
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
        layout.addWidget(self.capture_table, 1)
        self.rig_ready.toggled.connect(self.update_enabled)

    def _training(self, layout):
        layout.addWidget(label("TURN CAPTURES INTO AN INSTRUMENT", "eyebrow"))
        self.training_title = label("Train your HyperA2.", "title")
        layout.addWidget(self.training_title)
        layout.addWidget(
            label(
                "Choose settings and metadata, then add a new independent training instance to the queue.",
                "muted",
            )
        )
        self.training_settings, box = card("Training settings")
        form = QFormLayout()
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100000)
        self.epochs.setValue(200)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(16)
        self.batch_size.setToolTip(
            "16 matches the official A2 recipe. Reduce it if GPU memory is insufficient; it can also be changed before resuming."
        )
        self.accelerator = QComboBox()
        self.accelerator.addItem(
            "Automatic — use a supported GPU when available", "auto"
        )
        self.accelerator.addItem("CPU — works on every supported computer", "cpu")
        form.addRow("Epochs per model / round", self.epochs)
        form.addRow("Training device", self.accelerator)
        box.addLayout(form)
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced training settings")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        box.addWidget(self.advanced_toggle)
        self.advanced_panel = QWidget()
        self.advanced_panel.setObjectName("advancedTraining")
        advanced_form = QFormLayout(self.advanced_panel)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        advanced_form.addRow("Batch Size", self.batch_size)
        self.advanced_fields = {}
        for key, caption, initial, low, high, decimals in (
            ("learning_rate", "Learning Rate", 0.004, 1e-9, 10, 9),
            (
                "hypernetwork_learning_rate",
                "Hypernetwork Learning Rate",
                0.0004,
                1e-9,
                10,
                9,
            ),
            ("lr_decay", "LR Decay", 0.994, 0, 1, 6),
            ("ny", "NY (samples)", 8192, 1, 1048576, 0),
            ("mrstft_weight", "MRSTFT Weight", 0.0005, 0, 10, 9),
            ("weight_decay", "Weight Decay", 3.17e-7, 0, 10, 12),
        ):
            field = QSpinBox() if key == "ny" else QDoubleSpinBox()
            if decimals:
                field.setDecimals(decimals)
                field.setSingleStep(0.001 if key != "weight_decay" else 1e-7)
            field.setRange(low, high)
            field.setValue(initial)
            self.advanced_fields[key] = field
            advanced_form.addRow(caption, field)
            field.valueChanged.connect(self.save_preferences)
        fit = QCheckBox("Fit MRSTFT")
        fit.setChecked(True)
        self.advanced_fields["fit_mrstft"] = fit
        advanced_form.insertRow(5, fit)
        fit.toggled.connect(self.advanced_fields["mrstft_weight"].setEnabled)
        fit.toggled.connect(self.save_preferences)
        optimizer = QComboBox()
        optimizer.addItem("Adam", "Adam")
        optimizer.addItem("AdamW", "AdamW")
        optimizer.setToolTip(
            "New training defaults to Adam. Resume requires the same optimizer as the saved checkpoint."
        )
        self.advanced_fields["optimizer"] = optimizer
        advanced_form.insertRow(0, "Optimizer", optimizer)
        optimizer.currentIndexChanged.connect(self.save_preferences)
        preset = QComboBox()
        preset.addItem("HyperA2", "HyperA2")
        preset.addItem("HyperWaveNet (Phillip Self)", "HyperWaveNet")
        preset.setToolTip(
            "Selecting a preset resets its advanced values and batch size. You can customize them afterwards. Both presets train Full and Lite."
        )
        self.advanced_fields["hypernetwork_recipe"] = preset
        advanced_form.insertRow(0, "Preset", preset)
        preset.activated.connect(self.apply_training_preset)
        rate_preset = QComboBox()
        rate_preset.addItem("Manual / per-epoch decay", "manual")
        rate_preset.addItem("A2 — same cadence per optimizer update", "a2_updates")
        rate_preset.addItem("A2 — n× slower (knobs + switches)", "a2_controls")
        rate_preset.setToolTip(
            "The controls-scaled preset keeps the A2 reduction factor but multiplies "
            "the number of updates between reductions by the model's knob/switch count."
        )
        self.advanced_fields["rate_preset"] = rate_preset
        advanced_form.insertRow(1, "Learning-rate preset", rate_preset)
        rate_preset.activated.connect(self.apply_rate_preset)
        self.rate_preview = label("", "muted")
        self.rate_preview.setWordWrap(True)
        advanced_form.addRow(self.rate_preview)
        self.preset_description = label("", "muted")
        advanced_form.addRow(self.preset_description)
        self.advanced_fields["hypernetwork_learning_rate"].setToolTip(
            "Independent learning rate for the knob-to-weight network. Both learning rates decay together. Parametric training only."
        )
        self.advanced_fields["learning_rate"].setToolTip(
            "Learning rate for the base audio-network weights."
        )
        clipping = QCheckBox("Gradient clipping")
        self.advanced_fields["gradient_clipping"] = clipping
        advanced_form.addRow(clipping)
        limit = QDoubleSpinBox()
        limit.setDecimals(6)
        limit.setRange(0.000001, 1000000)
        limit.setValue(1.0)
        limit.setToolTip("Maximum gradient norm. Applied before each optimizer update.")
        self.advanced_fields["gradient_clip_val"] = limit
        advanced_form.addRow("Maximum gradient norm", limit)
        clipping.toggled.connect(limit.setEnabled)
        clipping.toggled.connect(self.save_preferences)
        limit.valueChanged.connect(self.save_preferences)
        nonfinite = QCheckBox("Check NaN/Inf gradients")
        nonfinite.setToolTip(
            "Skip an optimizer update if any gradient is NaN or infinite. Disabling this removes the scan overhead."
        )
        self.advanced_fields["check_nonfinite_gradients"] = nonfinite
        advanced_form.addRow(nonfinite)
        nonfinite.toggled.connect(self.save_preferences)
        weight_mode = QComboBox()
        weight_mode.addItem("Full", "full")
        weight_mode.addItem("Low rank", "low_rank")
        weight_mode.setToolTip(
            "How knob-dependent weight deltas are generated. Independent of Full/Lite audio size."
        )
        self.advanced_fields["hypernetwork_mode"] = weight_mode
        advanced_form.addRow("Hypernetwork weight mode", weight_mode)
        rank = QSpinBox()
        rank.setRange(1, 8)
        rank.setValue(2)
        rank.setToolTip(
            "Rank of generated matrix deltas, from 1 to 8. Effective rank is also limited by each matrix's dimensions."
        )
        self.advanced_fields["hypernetwork_rank"] = rank
        advanced_form.addRow("Rank (low-rank mode)", rank)
        dilated = QCheckBox("Condition dilated convolution kernels on knobs")
        dilated.setToolTip(
            "When disabled, dilated convolution kernel weights remain trainable and shared across settings. Their biases may still be conditioned."
        )
        self.advanced_fields["condition_dilated_convolutions"] = dilated
        advanced_form.addRow(dilated)
        reduction = QComboBox()
        reduction.addItem("Sum", "sum")
        reduction.addItem("Mean", "mean")
        reduction.setToolTip(
            "Changes how Full/Lite training losses are combined before backpropagation, not the definition of validation ESR."
        )
        self.advanced_fields["branch_loss_reduction"] = reduction
        advanced_form.addRow("Full/Lite loss reduction", reduction)
        for field, signal in (
            (weight_mode, weight_mode.currentIndexChanged),
            (rank, rank.valueChanged),
            (dilated, dilated.toggled),
            (reduction, reduction.currentIndexChanged),
        ):
            signal.connect(self.save_preferences)
            signal.connect(lambda *_: self.refresh_hypernetwork_controls())
        self.advanced_fields["lr_decay"].setToolTip(
            "Exponential learning-rate multiplier after each epoch."
        )
        self.advanced_fields["ny"].setToolTip(
            "Number of training output samples per window. Validation keeps its independent window."
        )
        box.addWidget(self.advanced_panel)
        self.advanced_panel.hide()
        self.advanced_toggle.toggled.connect(self.advanced_panel.setVisible)
        self.advanced_toggle.toggled.connect(
            lambda expanded: self.advanced_toggle.setArrowType(
                Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
            )
        )
        box.addWidget(
            label(
                "Full and Lite are trained together. Pause stops at the next batch boundary; resume restarts the interrupted epoch from its last complete checkpoint.",
                "muted",
            )
        )
        layout.addWidget(self.training_settings)
        if sys.platform != "darwin":
            self.gpu_setup_button = button("Add NVIDIA GPU support", self.configure_gpu)
            box.addWidget(self.gpu_setup_button)
            self.refresh_gpu_button()
        layout.addWidget(self.metadata_card)
        self.train_button = button("Train · add to queue", self.start_training, True)
        layout.addWidget(self.train_button)
        self.epochs.valueChanged.connect(self.save_preferences)
        self.batch_size.valueChanged.connect(self.save_preferences)
        self.accelerator.currentIndexChanged.connect(self.save_preferences)

    def _queue(self, layout):
        layout.setSpacing(8)
        layout.addWidget(label("Training queue", "title"))
        layout.addWidget(
            label(
                "Each row is an independent training instance. Reordering affects waiting jobs; removing a row keeps its models and checkpoints on disk.",
                "muted",
            )
        )
        self.queue_run_row = QHBoxLayout()
        layout.addLayout(self.queue_run_row)
        self.queue_table = QTableWidget(0, 7)
        self.queue_table.setHorizontalHeaderLabels(
            [
                "Order",
                "Project",
                "Created",
                "Model",
                "Status",
                "Epoch",
                "Validation Full ESR",
            ]
        )
        self.queue_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.queue_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.queue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.queue_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.queue_table.horizontalHeader().setStretchLastSection(True)
        self.queue_table.setMinimumHeight(100)
        self.queue_table.setMaximumHeight(16777215)
        for column, width in enumerate((56, 120, 110, 90, 100, 65, 85)):
            self.queue_table.setColumnWidth(column, width)
        self.queue_table.itemSelectionChanged.connect(self.select_queue_job)
        layout.addWidget(self.queue_table, 1)
        actions = QHBoxLayout()
        self.resume_button = button("Resume", self.resume_training)
        self.retrain_button = button("Retrain", self.retrain_training)
        self.job_up = button("Move up", lambda: self.move_job(-1))
        self.job_down = button("Move down", lambda: self.move_job(1))
        self.remove_job_button = button("Remove from queue", self.remove_job)
        self.capture_round_button = button(
            "Capture next active-learning round", self.open_capture_round
        )
        for widget in (self.job_up, self.job_down, self.remove_job_button):
            actions.addWidget(widget)
        layout.addLayout(actions)
        secondary_actions = QHBoxLayout()

        secondary_actions.addWidget(self.capture_round_button)
        layout.addLayout(secondary_actions)
        self.resume_options_widget = QWidget()
        resume_options = QHBoxLayout(self.resume_options_widget)
        resume_options.setContentsMargins(0, 0, 0, 0)
        self.resume_batch_size = QSpinBox()
        self.resume_batch_size.setRange(1, 128)
        self.resume_device = QComboBox()
        self.resume_device.addItem("Automatic", "auto")
        self.resume_device.addItem("CPU", "cpu")
        resume_options.addWidget(label("Resume batch size"))
        resume_options.addWidget(self.resume_batch_size)
        resume_options.addWidget(label("Device"))
        resume_options.addWidget(self.resume_device)
        resume_options.addStretch()
        layout.addWidget(self.resume_options_widget)
        self.training_status = label("Add training from a project to begin.", "muted")
        layout.addWidget(self.training_status)
        grid = QGridLayout()
        self.metrics = {}
        for index, (name, text) in enumerate(
            (
                ("epoch", "EPOCH"),
                ("esr", "BEST VALIDATION ESR (FULL)"),
                ("train_esr", "BEST TRAIN ESR (FULL)"),
                ("elapsed", "ELAPSED"),
                ("eta", "REMAINING"),
            )
        ):
            frame, metric_layout = card(text)
            metric_layout.setContentsMargins(12, 6, 12, 6)
            metric_layout.setSpacing(4)
            metric = label("—", "metric")
            metric.setStyleSheet("font-size: 22px;")
            metric_layout.addWidget(metric)
            self.metrics[name] = metric
            grid.addWidget(frame, 0, index)
        layout.addLayout(grid)
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 1000)
        layout.addWidget(self.train_progress)
        row = self.queue_run_row
        self.queue_start_button = button("Start queue", self.run_queue, True)
        self.pause_button = button("Pause & save", self.pause_training)
        self.stop_button = button("Stop", self.stop_training)
        self.models_button = button("Open model folder", self.open_models)
        row.addWidget(self.queue_start_button)
        row.addWidget(self.pause_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.resume_button)
        row.addWidget(self.retrain_button)
        self.export_best_button = button("Export best model", self.export_best_model)
        row.addWidget(self.export_best_button)
        row.addStretch()
        actions.addWidget(self.models_button)
        self.export_status = label("", "muted")
        layout.addWidget(self.export_status)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.details.setMinimumHeight(180)
        from .esr_chart import ESRCharts

        self.esr_chart = ESRCharts()
        for chart in self.esr_chart.charts.values():
            chart.setMinimumHeight(160)
            chart.setMaximumHeight(16777215)
        self.training_tabs = self.esr_chart
        self.training_tabs.addTab(self.details, "Training log")
        layout.addWidget(self.training_tabs, 1)

    def load_advanced_options(self, kind=None):
        options = self.workspace.training_options(kind)
        for key, field in self.advanced_fields.items():
            field.blockSignals(True)
            if isinstance(field, QComboBox):
                field.setCurrentIndex(max(0, field.findData(options[key])))
            elif isinstance(field, QCheckBox):
                field.setChecked(options[key])
            else:
                field.setValue(options[key])
            field.blockSignals(False)
        self.advanced_fields["mrstft_weight"].setEnabled(options["fit_mrstft"])
        self.advanced_fields["gradient_clip_val"].setEnabled(
            options["gradient_clipping"]
        )
        parametric = (
            kind or self.workspace.preferences["capture_type"]
        ) == "parametric"
        self.advanced_fields["hypernetwork_recipe"].setEnabled(parametric)
        self.advanced_fields["hypernetwork_learning_rate"].setEnabled(parametric)
        self.refresh_hypernetwork_controls(kind)
        self.refresh_rate_preview(kind)

    def apply_rate_preset(self, *_):
        if not self.workspace or self._loading:
            return
        if self.advanced_fields["rate_preset"].currentData() in (
            "a2_updates",
            "a2_controls",
        ):
            for key, value in (
                ("learning_rate", 0.004),
                ("hypernetwork_learning_rate", 0.0004),
                ("lr_decay", 0.994),
            ):
                self.advanced_fields[key].blockSignals(True)
                self.advanced_fields[key].setValue(value)
                self.advanced_fields[key].blockSignals(False)
        self.save_preferences()

    def refresh_rate_preview(self, kind=None):
        if not self.workspace:
            return
        parametric = (
            kind or self.workspace.preferences["capture_type"]
        ) == "parametric"
        self.advanced_fields["rate_preset"].setEnabled(parametric)
        matched = parametric and self.advanced_fields["rate_preset"].currentData() in (
            "a2_updates",
            "a2_controls",
        )
        self.advanced_fields["lr_decay"].setEnabled(not matched)
        self.advanced_fields["lr_decay"].setToolTip(
            "Reference A2-epoch multiplier. The preview shows its cadence in optimizer updates and the equivalent multiplier per project epoch."
            if matched
            else "Exponential learning-rate multiplier after each epoch."
        )
        self.rate_preview.setVisible(matched)
        if not matched:
            return
        try:
            from .recipes import workspace_update_schedule

            schedule = workspace_update_schedule(self.workspace)
            self.rate_preview.setText(
                f"A2 reference: {schedule['reference_steps']} updates/epoch (standard input, batch 16, NY 8192). "
                f"This plan: {schedule['steps_per_epoch']} updates/epoch, {schedule['training_seconds']:.1f} s of training audio.\n"
                f"Cadence multiplier: {schedule['cadence_multiplier']}× "
                f"({'knob/switch count' if self.advanced_fields['rate_preset'].currentData() == 'a2_controls' else 'original A2 cadence'}). "
                f"Multiply LR by {schedule['reference_gamma']:g} every {schedule['decay_steps']} optimizer updates; average equivalent per epoch: {schedule['gamma_epoch']:.6g}. "
                f"After {self.workspace.preferences['epochs']} epochs: LR {schedule['final_learning_rate']:.3e}, "
                f"hypernetwork LR {schedule['final_hypernetwork_learning_rate']:.3e} "
                f"({schedule['reductions']} LR reductions; {schedule['equivalent_a2_epochs']:.1f} reference A2 epochs of updates). "
                "Initial rates remain editable. Matching cadence does not select the training duration or guarantee convergence."
            )
        except (ValueError, OSError, RuntimeError) as error:
            self.rate_preview.setText(f"A2 cadence preview unavailable: {error}")

    def refresh_hypernetwork_controls(self, kind=None):
        if not self.workspace:
            return
        parametric = (
            kind or self.workspace.preferences["capture_type"]
        ) == "parametric"
        for key in (
            "hypernetwork_mode",
            "condition_dilated_convolutions",
            "branch_loss_reduction",
        ):
            self.advanced_fields[key].setEnabled(parametric)
        low_rank = self.advanced_fields["hypernetwork_mode"].currentData() == "low_rank"
        self.advanced_fields["hypernetwork_rank"].setEnabled(parametric and low_rank)
        mode = (
            f"low rank (rank {self.advanced_fields['hypernetwork_rank'].value()})"
            if low_rank
            else "full"
        )
        kernels = (
            "conditioned"
            if self.advanced_fields["condition_dilated_convolutions"].isChecked()
            else "shared"
        )
        reduction = self.advanced_fields["branch_loss_reduction"].currentData()
        self.preset_description.setText(
            f"Weight deltas: {mode}; dilated kernels: {kernels}; Full/Lite loss: {reduction}."
            if parametric
            else "Static A2 training; hypernetwork settings are not used."
        )

    def apply_training_preset(self, *_):
        if (
            not self.workspace
            or self.workspace.preferences["capture_type"] != "parametric"
        ):
            return
        from .recipes import training_preset, preset_batch_size

        name = self.advanced_fields["hypernetwork_recipe"].currentData()
        self.workspace.preferences["advanced_parametric"] = training_preset(name)
        self.workspace.preferences["batch_size"] = preset_batch_size(name)
        self.batch_size.blockSignals(True)
        self.batch_size.setValue(preset_batch_size(name))
        self.batch_size.blockSignals(False)
        self.load_advanced_options("parametric")
        self.save_preferences()

    def busy(self):
        return (self.worker is not None and self.worker.isRunning()) or getattr(
            self, "play_worker", None
        ) is not None

    def notify_error(self, error):
        QMessageBox.warning(self, "ParaNAM Trainer", str(error))

    def navigate(self, index):
        if index in (1, 2, 3, 5) and not self.workspace:
            self.notify_error("Create or open a project first.")
            return
        if index in (2, 3) and not self.busy():
            try:
                if self.workspace.plan_changed() and not self.apply_plan():
                    return
            except ValueError as error:
                self.notify_error(error)
                return
        if index == 6 and self.pages.currentIndex() != 6:
            self._settings_return_page = self.pages.currentIndex()
        self.pages.setCurrentIndex(index)
        self.project_label.setVisible(index != 6)
        listening = getattr(self, "play_worker", None) is not None
        self.play_transport_bar.setVisible(index != 5 and listening)
        self.footer_note.setVisible(index != 5 and not listening)
        if index == 3:
            self.refresh_gpu_button()
            self.refresh_rate_preview()
        if index == 5:
            self.refresh_play_models()
        for i, step in enumerate(self.steps):
            step.setChecked(i == index)
        self.back_button.setEnabled(index > 0)
        self.next_button.setVisible(index < 3)
        self.next_button.setText("Train →" if index == 2 else "Next →")
        self.update_enabled()

    def back(self):
        if self.pages.currentIndex() == 6:
            self.navigate(getattr(self, "_settings_return_page", 0))
            return
        self.navigate(max(0, self.pages.currentIndex() - 1) if self.workspace else 0)

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
            if metadata:
                try:
                    self.queue_controller.store.register_legacy(Workspace(path))
                except (OSError, ValueError):
                    pass  # A missing/older project remains visible in Recent projects.
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
        parent = self.choose_directory("Choose where to save the project", documents)
        if not parent:
            return
        safe_name = re.sub(r"[^\w .-]", "_", name).strip(" .") or "ParaNAM Project"
        try:
            directory = Path(parent) / safe_name
            workspace = Workspace.create(directory, name.strip(), self.audio_settings())
            self._attach(workspace, use_last_setup=True)
            self.navigate(1)
        except Exception as error:
            self.notify_error(error)

    def open_project_dialog(self):
        path = self.choose_directory("Open ParaNAM or NAM capture project")
        if path:
            self.open_project(Path(path))

    def open_project(self, directory):
        if self.busy():
            return
        try:
            self._attach(Workspace(directory), use_last_setup=True)
            self.navigate(1)
        except Exception as error:
            self.notify_error(error)

    def _attach(self, workspace, use_last_setup=False):
        self.close_audio()
        self.initialize_global_setup(workspace)
        self.workspace = workspace
        self.queue_controller.store.register_legacy(workspace)
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
            self.metadata_editor.set_metadata(workspace.project.user_metadata)
            for field, key in (
                (self.n_train, "n_train"),
                (self.n_validation, "n_validation"),
                (self.seed, "seed"),
                (self.rounds, "rounds"),
                (self.additions, "additions"),
                (self.epochs, "epochs"),
            ):
                field.setValue(prefs[key])
            self.batch_size.setValue(workspace.training_batch_size())
            self.load_advanced_options()
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
        finally:
            self._loading = False
        self.save_preferences()
        self.remember_setup()
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
                (self.instrument_input, result.inputs, None),
                (self.monitor_left, result.outputs, None),
                (self.monitor_right, result.outputs, None),
            ):
                previous = combo.currentData() or default
                combo.blockSignals(True)
                combo.clear()
                if default is None:
                    combo.addItem("Not configured", None)
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
                self.loop_enabled.setToolTip(
                    "Clean loopback needs a second input and output. Choose a driver with both channels or disable loopback."
                )
            else:
                self.loop_enabled.setToolTip(
                    "Use a separate clean timing reference on the selected channels."
                )

    def audio_settings(self):
        info = self.device.currentData()
        return AudioSettingsModel(
            host_api=info.host_api if info else None,
            input_device=info.name if info else None,
            output_device=info.name if info else None,
            input_channel=self.input_channel.currentData() or 1,
            instrument_input_channel=self.instrument_input.currentData(),
            monitor_output_left=self.monitor_left.currentData(),
            monitor_output_right=self.monitor_right.currentData(),
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
        self.loop_input.setEnabled(self.loop_enabled.isChecked())
        self.loop_output.setEnabled(self.loop_enabled.isChecked())

    def save_audio(self, *_):
        if self._loading or self.busy():
            return
        self.close_audio()
        self._diagram()
        audio = self.audio_settings()
        key = (audio.host_api, audio.input_device, audio.instrument_input_channel)
        if getattr(self, "_instrument_route_key", key) != key:
            self.instrument_calibration.blockSignals(True)
            self.instrument_calibration.setChecked(False)
            self.instrument_calibration.blockSignals(False)
        self._instrument_route_key = key
        self.remember_setup()
        self._refresh_play_calibration()

    def save_calibration(self, *_):
        if self._loading or self.busy():
            return
        self.remember_setup()
        self._refresh_play_calibration()

    def save_instrument_calibration(self, *_):
        self.save_calibration()

    def apply_instrument_calibration(self, value):
        self._apply_level_calibration(
            self.instrument_calibration, self.instrument_dbu, value
        )

    def save_capture_level(self, *_):
        if self._loading or self.busy():
            return
        try:
            level = self.capture_level.level()
            previous = (self.last_setup() or {}).get("capture_level_dbfs", level)
            if level != previous:
                self.input_calibration.blockSignals(True)
                self.input_calibration.setChecked(False)
                self.input_calibration.blockSignals(False)
            self.remember_setup()
            self.close_audio()
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
        if not self.device.currentData():
            self.notify_error("Select your audio interface before calibrating.")
            return
        try:
            self.close_audio()
            dialog = LevelCalibrationDialog(
                self.device.currentData(),
                self.input_channel.currentData(),
                self.output_channel.currentData(),
                self.buffer.currentData() or 0,
                self.capture_level.level(),
                getattr(self, "_calibration_frequency", 1000.0),
                self,
                channel_names=(
                    self.input_channel.currentText(),
                    self.output_channel.currentText(),
                ),
                instrument_channel=self.instrument_input.currentData(),
                instrument_name=self.instrument_input.currentText(),
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
            dialog.instrument_calibrated.connect(self.apply_instrument_calibration)
            dialog.exec()
            self._calibration_frequency = dialog.frequency.value()
            self.remember_setup()
        except ValueError as error:
            self.notify_error(error)

    def import_inputs(self):
        if not self.workspace or self.busy():
            return
        path, _ = self.choose_open_file(
            "Choose custom training WAV (mono, 48 kHz)", "WAV audio (*.wav)"
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
                    self.workspace.project.model_copy(
                        update={"audio": self.audio_settings()}
                    ),
                    self.workspace.directory,
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
        if self.workspace and self.session:
            self.workspace.project = self.session.project
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
        ):
            prefs[key] = field.value()
        previous_type = prefs["capture_type"]
        next_type = self.capture_kind.currentData()
        batch_key = "static_batch_size" if previous_type == "static" else "batch_size"
        prefs[batch_key] = self.batch_size.value()
        prefs["advanced_" + previous_type] = {
            key: (
                field.currentData()
                if isinstance(field, QComboBox)
                else (
                    field.isChecked() if isinstance(field, QCheckBox) else field.value()
                )
            )
            for key, field in self.advanced_fields.items()
        }
        if previous_type != next_type:
            next_key = "static_batch_size" if next_type == "static" else "batch_size"
            self.batch_size.blockSignals(True)
            self.batch_size.setValue(prefs[next_key])
            self.batch_size.blockSignals(False)
            self.load_advanced_options(next_type)
        prefs.update(
            capture_type=self.capture_kind.currentData(),
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
        self.metadata_editor.set_name_visible(not static)
        self.batch_size.setToolTip(
            "16 matches the official NAM trainer default for static models. Reduce it if GPU memory is insufficient."
            if static
            else "16 matches the official A2 recipe. Reduce it if GPU memory is insufficient."
        )
        self.control_card.setVisible(not static)
        self.plan_card.setVisible(not static)
        self.model_format.setText(
            "A2 Full + Lite nested · one standard .nam per named capture"
            if static
            else "HyperA2 · Full + Lite nested in one .pnam file"
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
        self.refresh_rate_preview()
        self.custom_button.setEnabled(prefs["input_mode"] == "custom")

    def apply_plan(self):
        if not self.workspace or self.busy():
            return False
        self.save_preferences()
        original_capture_settings = {
            key: self.workspace.preferences.get(key)
            for key in ("capture_level_dbfs", "pending_calibration")
        }
        self.prepare_capture_setup()
        try:
            if self.workspace.plan_changed():
                if self.workspace.project.captured_entries():
                    answer = QMessageBox.question(
                        self,
                        "Update the capture plan?",
                        "The controls, input signal, capture level, calibration or plan settings changed. Existing captures will be preserved in History, and a new plan will start. Continue?",
                    )
                    if answer != QMessageBox.StandardButton.Yes:
                        for key, value in original_capture_settings.items():
                            if value is None:
                                self.workspace.preferences.pop(key, None)
                            else:
                                self.workspace.preferences[key] = value
                        self.workspace.save()
                        return False
                self.close_audio()
                self.workspace.generate_plan()
                self.remember_capture_calibration()
            self.refresh_captures()
            return True
        except Exception as error:
            self.notify_error(error)
            return False

    def _model_format_label(self):
        self.model_format = label(
            "HyperA2 · Full + Lite nested in one .pnam file", "eyebrow"
        )
        return self.model_format

    def save_metadata(self, metadata):
        if self.workspace and not self._loading and not self.busy():
            self.workspace.project.user_metadata = dict(metadata)
            if self.workspace.preferences["capture_type"] == "static":
                # Edited Train fields are authoritative for the batch. Retain
                # unexposed legacy metadata, while names come from Capture.
                keys = set(self.metadata_editor.fields) - {"name"}
                for capture in self.workspace.preferences["static_captures"]:
                    capture["metadata"] = {
                        k: v
                        for k, v in capture.get("metadata", {}).items()
                        if k not in keys
                    }
                for entry in self.workspace.project.entries:
                    entry.user_metadata = {
                        k: v for k, v in entry.user_metadata.items() if k not in keys
                    }
            self.workspace.save()

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
        latency = "—"
        if delay is not None:
            # The stored shift was applied to the audio, so undo it to recover
            # the fractional label: shift = rounded_delay - fractional_delay.
            # A legacy loopback-only fallback is already a measured delay.
            fraction = qa.subsample_shift if qa and entry.delay is not None else None
            latency = (
                f"{delay - fraction:.3f} samples"
                if fraction is not None
                else f"{delay} samples"
            )
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
        references = (
            self.capture_level.level(),
            self.input_dbu.value() if self.input_calibration.isChecked() else None,
            self.output_dbu.value() if self.output_calibration.isChecked() else None,
        )
        historical = (
            self.workspace.project.capture_level_dbfs,
            self.workspace.project.input_level_dbu,
            self.workspace.project.output_level_dbu,
        )
        if references != historical:
            if self.apply_plan():
                self.capture_status.setText(
                    "Capture settings updated. Confirm the rig before recording."
                )
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
            self.remember_capture_calibration()
            session = self._get_session()
            self.capture_status.setText(
                "Recording… Timing is checked at the beginning and end of this recording."
            )
            self.capture_progress.setValue(0)

            def capture_signals(progress, cancel):
                results = []
                for offset, capture in enumerate(captures):
                    self.capture_stage.emit(
                        f"Recording {capture.split} · signal {offset + 1} / {len(captures)}"
                    )
                    if progress:
                        progress(0.0)
                    results.append(
                        session.capture_entry(
                            capture,
                            progress=progress,
                            cancel=cancel,
                        )
                    )
                    if progress:
                        progress(1.0)
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

    def configure_gpu(self):
        if self.busy():
            return
        from .gpu_setup import show_setup

        show_setup(selected=True, parent=self)
        self.refresh_gpu_button()

    def refresh_gpu_button(self):
        if hasattr(self, "gpu_setup_button"):
            from .gpu_runtime import ready

            self.gpu_setup_button.setVisible(not ready())

    def export_best_model(self):
        if (
            not self.job
            or not self.selected_job_id
            or getattr(self, "export_worker", None)
        ):
            return
        spec = read_json(self.job / "spec.json")
        extension = training.intermediate_export_extension(spec)
        title = "Export best PNAM" if extension == ".pnam" else "Export best NAM"
        file_filter = (
            "Parametric NAM model (*.pnam)"
            if extension == ".pnam"
            else "NAM model (*.nam)"
        )
        from .model_names import model_basename

        name = spec.get("model", {}).get("metadata", {}).get("name")
        if spec.get("static_captures"):
            member = min(
                training.job_progress(self.job).get("member", 0),
                len(spec["static_captures"]) - 1,
            )
            name = spec["static_captures"][member]["name"]
        default = Path(
            spec.get("models_directory", str(Path(spec["project"]) / "models"))
        ) / (model_basename(name, "Model") + extension)
        filename, _ = self.choose_save_file(title, default, file_filter)
        if not filename:
            return
        if Path(filename).suffix.lower() != extension:
            filename = str(Path(filename).with_suffix(extension))
        atomic_write_json(
            self.job / "export-request.json",
            {"id": str(time.time_ns()), "path": filename},
        )
        if self.selected_job_id == self.queue_controller.active_id:
            self.export_status.setText(
                "Export requested; waiting for the next training batch boundary."
            )
        else:
            from .queue_controller import ExportSavedBest

            self.export_status.setText(
                "Exporting the best saved Full and Lite checkpoints…"
            )
            self.export_worker = ExportSavedBest(self.job, self)
            self.export_worker.finished.connect(self._saved_export_finished)
            self.export_worker.start()
        self.export_best_button.setEnabled(False)

    def _saved_export_finished(self):
        self.export_worker.deleteLater()
        self.export_worker = None
        self.refresh_training()

    def _render_job_progress(self):
        progress = training.job_progress(self.job)
        if not progress:
            self.esr_chart.set_history([])
            entry = (
                self.queue_controller.store.get(self.selected_job_id)
                if self.selected_job_id
                else None
            )
            self.training_status.setText(
                (entry or {}).get("error")
                or "This training instance has no progress data yet."
            )
            for metric in self.metrics.values():
                metric.setText("—")
            self.train_progress.setValue(0)
            self.update_enabled()
            return
        self.esr_chart.set_history(progress.get("esr_history", []))
        status = progress.get("status", "starting")
        selected = (
            self.queue_controller.store.get(self.selected_job_id)
            if self.selected_job_id
            else None
        )
        queue_state = selected["state"] if selected else status
        if queue_state in (
            "queued",
            "paused",
            "stopped",
            "interrupted",
            "pausing",
            "stopping",
        ):
            status = queue_state
        message = progress.get("message", status.replace("_", " ").title())
        if queue_state == "queued":
            message = "Waiting in the training queue. This instance keeps its original settings and recordings."
        elif queue_state in ("paused", "stopped", "interrupted"):
            message = f"{queue_state.title()}. Resume belongs to this training instance, independently of the open project."
        elif queue_state in ("pausing", "stopping"):
            message = f"{queue_state.title()} at the next batch boundary. The last complete epoch is kept."
        if status == "failed" and progress.get("resume_available"):
            message += f"\nA checkpoint is available. Resume restarts epoch {progress.get('restart_epoch', 1)}. You can reduce batch size after a memory error."
        if self.selected_job_id != self.queue_controller.active_id and status in (
            "running",
            "scoring",
            "starting",
        ):
            message = "Previous session interrupted. Resume from the last complete saved epoch."
        self.training_status.setText(
            f"Round {progress.get('round', 1)} · Model {min(progress.get('member', 0) + 1, progress.get('members', 1))} / {progress.get('members', 1)} · {message}"
            + (
                f"\n{progress['device_name']} · batch {progress.get('batch_size')} · {progress.get('train_batches')} batches per epoch"
                if progress.get("device_name")
                else ""
            )
        )
        epoch, epochs = progress.get("epoch", 0), progress.get("epochs", 1)
        self.metrics["epoch"].setText(f"{epoch} / {epochs}")
        esr = progress.get("best_esr")
        self.metrics["esr"].setText("—" if esr is None else f"{esr:.5f}")
        train_esr = progress.get("best_train_esr")
        self.metrics["train_esr"].setText(
            "—" if train_esr is None else f"{train_esr:.5f}"
        )
        live = (
            max(0, time.time() - progress.get("updated_at", time.time()))
            if self.selected_job_id == self.queue_controller.active_id
            and self.selected_job_id is not None
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
            from .training_log import TrainingLog

            if (
                getattr(self, "_log_reader", None) is None
                or self._log_reader.path != logfile
            ):
                self._log_reader = TrainingLog(logfile)
            content = self._log_reader.read()
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
        result = read_json(self.job / "export-result.json", {})
        request = read_json(self.job / "export-request.json", {})
        if result and result.get("id") == request.get("id"):
            self.export_status.setText(
                f"Exported: {result['path']}"
                if result.get("ok")
                else f"Export failed: {result.get('error', 'Unknown error')}"
            )
        self.update_enabled()

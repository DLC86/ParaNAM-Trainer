"""Seven-screen desktop workflow; audio and training never block the UI thread."""

from __future__ import annotations

import json
from html import escape
import math
import re
import shutil
import sys
import time
from pathlib import Path

from .numeric import DecimalSpinBox as QDoubleSpinBox

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
    QApplication,
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
    QAbstractButton,
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
from .play_ui import PlayActions, HorizontalLevelMeter
from .settings_audio import SettingsAudioActions, AudioDeviceEvents
from .recovery_ui import RecoveryActions
from .updates import UpdateActions
from .queue_controller import QueueController
from .ui_preferences import UIPreferences
from .ui_widgets import (
    CaptureTable, PreserveSelectionTextDelegate, RecentProjectDelegate, navigation_icon
)


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
    return (
        f"{hours}h {minutes:02d}m {seconds:02d}s"
        if hours
        else f"{minutes:02d}:{seconds:02d}"
    )


class WizardWindow(
    UIPreferences,
    QueueActions,
    PlayActions,
    SettingsAudioActions,
    RecoveryActions,
    UpdateActions,
    QMainWindow,
):
    capture_stage = Signal(str)
    route_levels = Signal(float, float, float, float)

    def __init__(self):
        super().__init__()
        load_font()
        self.setWindowTitle("ParaNAM Trainer")
        self.setStyleSheet(
            THEME.replace(
                "CHECKMARK_URL", asset_path("checkmark.svg").as_posix()
            ).replace("ASSET_ROOT", asset_path("checkmark.svg").parent.as_posix())
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
        self.settings_audio_worker = None
        self._audio_devices_dirty = False
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
        self.route_levels.connect(self._route_metered)
        from .queue_clock import ElapsedClock

        self._elapsed_clock = ElapsedClock()
        self.queue_seconds_timer = QTimer(self)
        self.queue_seconds_timer.setSingleShot(True)
        self.queue_seconds_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.queue_seconds_timer.timeout.connect(self._tick_queue_seconds)
        self.timer = QTimer(self)
        self.timer.setInterval(600)
        self.timer.timeout.connect(self.refresh_training)
        self.timer.start()
        if sys.platform == "win32":
            from PySide6.QtWidgets import QApplication

            self._audio_device_events = AudioDeviceEvents(
                self._settings_device_changed, self.winId()
            )
            QApplication.instance().installNativeEventFilter(self._audio_device_events)
        self.device_timer = QTimer(self)
        self.device_timer.setInterval(3000)
        self.device_timer.timeout.connect(
            lambda: self._reconnect_device() if self.isVisible() else None
        )
        self.device_timer.start()
        self.navigate(0)
        self.setup_updates()

    def _build(self):
        central = QWidget(self)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        sidebar = QFrame(central)
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(228)
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
        self.brand_title = label("PNAM\nTrainer", "brand")
        self.version_label = label(f"VERSION {__version__}", "eyebrow")
        self.version_label.setObjectName("version")
        self.brand_title.setParent(sidebar)
        self.version_label.setParent(sidebar)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(4)
        brand_text.addWidget(self.brand_title)
        version_row = QHBoxLayout()
        version_row.setContentsMargins(0, 0, 0, 0)
        version_row.setSpacing(5)
        version_row.addWidget(self.version_label)
        self.update_led = QToolButton()
        self.update_led.setText("●")
        self.update_led.setFixedSize(15, 18)
        self.update_led.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_led.setAccessibleName("New ParaNAM version available")
        self.update_led.hide()
        version_row.addWidget(self.update_led)
        version_row.addStretch()
        brand_text.addLayout(version_row)
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
                "Projects",
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
        self.project_label.setTextFormat(Qt.TextFormat.RichText)
        content.addWidget(self.project_label)
        self.build_recovery_notice(content)
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
            page.setMaximumWidth(1100 if index in (1, 3, 6) else 1400)
            for field_type in (QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox):
                for field in page.findChildren(field_type):
                    field.setMaximumWidth(
                        min(
                            field.maximumWidth(),
                            560 if field_type in (QComboBox, QLineEdit) else 260,
                        )
                    )
            if index not in (0, 2, 4, 5):
                layout.addStretch()
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
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
        self.open_project_folder_button = button(
            "Open project folder", self.open_selected_project_folder
        )
        self.open_project_folder_button.setEnabled(False)
        row.addWidget(self.open_project_folder_button)
        self.delete_project_button = button("Delete", self.delete_selected_project)
        self.delete_project_button.setEnabled(False)
        row.addWidget(self.delete_project_button)
        box.addLayout(row)
        self.recent = QListWidget()
        self.recent.setItemDelegate(RecentProjectDelegate(self.recent))
        self.recent.setMinimumHeight(320)
        self.recent.itemSelectionChanged.connect(self._update_project_actions)
        self.recent.itemDoubleClicked.connect(
            lambda item: self.open_project(Path(item.data(Qt.ItemDataRole.UserRole)))
        )
        box.addWidget(self.recent)
        layout.addWidget(projects)

    def _update_project_actions(self):
        selected = self.recent.currentItem() is not None
        self.open_project_folder_button.setEnabled(selected)
        self.delete_project_button.setEnabled(selected)

    def delete_selected_project(self):
        item = self.recent.currentItem()
        if item is None:
            return
        project_path = Path(item.data(Qt.ItemDataRole.UserRole))
        name = item.text().splitlines()[0]
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Delete project")
        dialog.setText(f"Remove '{name}' from the Projects list?")
        dialog.setInformativeText(
            f"{project_path}\n\nProject files will be kept unless you select the option below."
        )
        delete_files = QCheckBox(
            "Also delete the project folder and all its contents", dialog
        )
        delete_files.setChecked(False)  # Must be unchecked on every opening.
        dialog.setCheckBox(delete_files)
        confirm = dialog.addButton("Delete", QMessageBox.ButtonRole.DestructiveRole)
        dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
        dialog.exec()
        if dialog.clickedButton() is not confirm:
            return

        if delete_files.isChecked():
            # Never follow a project-directory symlink or delete unrelated files
            # accidentally included in the recent-projects list.
            if (
                project_path.is_symlink()
                or not (project_path / "capture_project.json").is_file()
            ):
                self.notify_error("Cannot safely delete this project folder.")
                return
            target = project_path.resolve()
            if self.workspace and self.workspace.directory.resolve() == target:
                self.notify_error(
                    "Open another project before deleting the currently open folder."
                )
                return
            unfinished = {
                "queued", "starting", "running", "scoring", "pausing",
                "stopping", "paused", "interrupted", "needs_captures",
            }
            if any(
                Path(entry.get("project", "")).resolve() == target
                and entry.get("state") in unfinished
                for entry in self.queue_controller.store.entries()
            ):
                self.notify_error(
                    "This project has unfinished training jobs. Stop or remove "
                    "them from Queue before deleting its folder."
                )
                return
            try:
                shutil.rmtree(project_path)
            except OSError as error:
                self.notify_error(f"Could not delete the project folder: {error}")
                return

        recent = self.settings.value("recent", [])
        if isinstance(recent, str):
            recent = [recent]
        self.settings.setValue(
            "recent",
            [
                path for path in recent
                if Path(path).resolve() != project_path.resolve()
            ],
        )
        self._refresh_recent()

    def open_selected_project_folder(self):
        item = self.recent.currentItem()
        if item:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(
                    str(Path(item.data(Qt.ItemDataRole.UserRole)).resolve())
                )
            )

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
        driver_buttons = QVBoxLayout()
        self.asio_settings_button = button("Settings", self.open_asio_settings)
        self.asio_settings_button.setVisible(False)
        driver_buttons.addWidget(self.asio_settings_button)
        driver_buttons.addWidget(button("Refresh", self.refresh_devices))
        device_row.addLayout(driver_buttons)
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
            spin.addItem("None", None)
            spin.addItem("Channel name unavailable", 1)
            spin.addItem("Channel name unavailable", 2)
            spin.setCurrentIndex(spin.findData(1))
        self.loop_output.setCurrentIndex(self.loop_output.findData(2))
        self.loop_input.setCurrentIndex(self.loop_input.findData(2))
        form.addRow("Output → reamp / rig", self.output_channel)
        form.addRow("Input ← load box / microphone", self.input_channel)
        self.instrument_input, self.monitor_left, self.monitor_right = [
            QComboBox() for _ in range(3)
        ]
        for combo in (self.instrument_input, self.monitor_left, self.monitor_right):
            combo.addItem("None", None)
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
            "Live input meters. Test connection sends white noise to Send and Loopback output.",
            "muted",
        )
        row.addWidget(self.route_button)
        self.route_meter_panel = QWidget()
        self.route_meter_panel.setStyleSheet("QWidget { background: transparent; }")
        meters = QGridLayout(self.route_meter_panel)
        meters.setContentsMargins(0, 0, 0, 0)
        meters.setHorizontalSpacing(12)
        meters.setVerticalSpacing(3)
        meters.setColumnStretch(1, 1)
        for index, name in enumerate(
            ("send", "return", "loopback_send", "loopback_return")
        ):
            title = label(name.replace("_", " ").title())
            title.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            meter = HorizontalLevelMeter()
            peak = label("−∞ dBFS")
            peak.setAlignment(Qt.AlignmentFlag.AlignRight)
            peak.setMinimumWidth(82)
            meters.addWidget(title, index, 0)
            meters.addWidget(meter, index, 1)
            meters.addWidget(peak, index, 2)
            setattr(self, f"route_{name}_meter", meter)
            setattr(self, f"route_{name}_peak", peak)
        row.addWidget(self.route_meter_panel, 1)
        box.addLayout(row)
        box.addWidget(self.route_status)
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
        self.capture_level.setMaximumWidth(260)
        self.instrument_calibration = QCheckBox("Instrument input at 0 dBFS")
        self.instrument_dbu = QDoubleSpinBox()
        self.instrument_dbu.setRange(-200, 200)
        self.instrument_dbu.setDecimals(3)
        self.instrument_dbu.setSuffix(" dBu")
        form.addRow(self.instrument_calibration, self.instrument_dbu)
        self.instrument_calibration.toggled.connect(self.save_instrument_calibration)
        self.instrument_dbu.valueChanged.connect(self.save_instrument_calibration)
        self.capture_level.editingFinished.connect(self.save_capture_level)
        form.addRow("Capture / validation output (dBFS peak)", self.capture_level)
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
        corner = self.control_table.findChild(QAbstractButton)
        if corner:
            corner.setText("#")
        self.control_table.setHorizontalHeaderLabels(
            ["Name", "Type", "Range / positions", "Default"]
        )
        self.control_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        for column, width in enumerate((200, 100, 320, 160)):
            self.control_table.setColumnWidth(column, width)
        self.control_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
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
        self.control_up_button = button("Move up", lambda: self.move_control(-1))
        self.control_down_button = button("Move down", lambda: self.move_control(1))
        row.addWidget(self.control_up_button)
        row.addWidget(self.control_down_button)
        row.addStretch()
        box.addLayout(row)
        layout.addWidget(self.control_card)
        self.signal_card, box = card("Input signal")
        signal_form = QFormLayout()
        self.signal_mode = QComboBox()
        self.signal_mode.addItem("Standard — bundled TONE3000 sweep", "standard")
        self.signal_mode.addItem("TTS input v11 — bundled", "tts")
        self.signal_mode.addItem("Truncated (Phillip Self)", "trunc")
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
        self.plan_card, box = card("Capture plan")
        form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem("Normal — capture the plan, then train", "normal")
        self.mode.addItem(
            "Active learning — suggest informative captures in rounds",
            "active_learning",
        )
        self.plan_form = form
        # The selector and active-learning implementation are retained for later.
        form.addRow("Training method", self.mode)
        form.setRowVisible(self.mode, False)
        form.addRow(
            "Model format",
            self._model_format_label(),
        )
        self.n_train, self.n_validation = QSpinBox(), QSpinBox()
        self.n_train.setRange(1, 1000000)
        self.n_validation.setRange(0, 1000000)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.lhs_method = QComboBox()
        self.lhs_method.addItem("Standard", "standard")
        self.lhs_method.addItem("Maximin", "maximin")
        self.lhs_method.addItem("Orthogonal", "orthogonal")
        self.lhs_method.addItem("Near-orthogonal", "near_orthogonal")
        self.lhs_method.setToolTip(
            "Standard: random stratified Latin hypercube. Maximin: maximize the closest-pair "
            "distance over candidate designs. Orthogonal: strength-2 orthogonal-array LHS "
            "requiring p² samples per split and Complex Switch state, for prime p and "
            "at most p+1 dimensions (e.g. 9, 25, 49, 121). Near-orthogonal: minimize "
            "pairwise control correlations without those count restrictions. "
            "Anchors, switch mapping and knob-grid rounding may reduce the ideal properties."
        )
        self.corners = QCheckBox(
            "Include range-extreme captures for every knob/switch combination (corners)"
        )
        self.corners.setChecked(True)
        form.addRow("Training LHS captures", self.n_train)
        form.addRow("LHS design", self.lhs_method)
        form.addRow("Validation captures", self.n_validation)
        self.validation_method = QComboBox()
        self.validation_method.addItem("Independent LHS (current)", "independent")
        self.validation_method.addItem("Farthest from train (maximin)", "farthest")
        self.validation_method.setToolTip(
            "Independent LHS keeps the existing validation settings selection. "
            "Farthest from train chooses on-grid validation points as far as possible "
            "from training LHS/default and corners, balancing Simple Switch states. "
            "Regenerating an existing plan archives its previous captures."
        )
        form.addRow("Validation selection", self.validation_method)
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
                "You return to the capture step for each new batch. Every new parametric export uses ParametricSlimmableContainer with Full + Lite.",
                "muted",
            )
        )
        # Keep the explanation for future restoration, but hide it for Normal-only training.
        self.active_learning_help = box.itemAt(box.count() - 1).widget()
        self.active_learning_help.hide()
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
        self.lhs_method.currentIndexChanged.connect(self.save_preferences)
        self.validation_method.currentIndexChanged.connect(self.save_preferences)
        self.mode.currentIndexChanged.connect(self.save_preferences)

    def _capture(self, layout):
        options = QHBoxLayout()
        options.addStretch(1)
        self.monitor_return = QCheckBox("Monitor Return")
        self.monitor_return.setToolTip("Send the live Return input to the selected Monitor outputs during capture.")
        self.verify_asio_timing = QCheckBox(
            "Check timing stability before each capture"
        )
        self.verify_asio_timing.setChecked(True)
        self.verify_asio_timing.setToolTip(
            "Check ASIO loopback timing stability before each capture. Turn off to "
            "skip the additional probes; captured timing markers are still used for alignment."
        )
        self.monitor_return.toggled.connect(lambda *_: self.save_audio())
        self.verify_asio_timing.toggled.connect(lambda *_: self.save_audio())
        options.addWidget(self.monitor_return)
        options.addWidget(self.verify_asio_timing)
        layout.addLayout(options)
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
        self.load_capture_button = button("Load WAV", self.load_static_wav)
        self.load_capture_button.setToolTip(
            "Import the aligned mono return for the input selected in Plan."
        )
        self.load_capture_button.hide()
        row.addWidget(self.load_capture_button)
        row.addWidget(self.cancel_button)
        self.volume_check_button = button(
            "Review volume check", self.review_volume_check
        )
        self.volume_check_button.hide()
        row.addWidget(self.volume_check_button)
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
        corner = self.capture_table.findChild(QAbstractButton)
        if corner:
            corner.setText("#")
        self.capture_table.setHorizontalHeaderLabels(
            [
                "Type",
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
        self.capture_table.setItemDelegate(PreserveSelectionTextDelegate(self.capture_table))
        self.capture_table.setMinimumHeight(270)
        self.capture_table.itemSelectionChanged.connect(self.select_capture)
        layout.addWidget(self.capture_table, 1)
        self.rig_ready.toggled.connect(self.update_enabled)
        self.rig_ready.toggled.connect(
            lambda checked: self.scene.stop_blink() if checked else None
        )

    def _training(self, layout):
        layout.addWidget(label("TURN CAPTURES INTO AN INSTRUMENT", "eyebrow"))
        self.training_title = label("Train your ParametricSlimmableContainer.", "title")
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
        self.epochs.setValue(50)
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
            ("lr_decay", "LR Decay", 0.994, 0, 1, 9),
            ("hypernetwork_lr_decay", "Hypernetwork LR Decay", 0.998497, 0, 1, 9),
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
            field.setFixedWidth(260)
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

        selection = QComboBox()
        selection.addItem("Lowest Validation ESR (current)", "validation")
        selection.addItem("Validation within 5%, then lowest Train ESR", "tolerance")
        selection.addItem("Combined ESR (alpha × Validation + (1-alpha) × Train)", "combined")
        selection.setToolTip(
            "Choose the Full and Lite checkpoints separately. Train ESR uses "
            "held-out audio at training knob settings. Tolerance chooses the "
            "lowest Train ESR among epochs within 5% of best Validation ESR."
        )
        self.advanced_fields["checkpoint_method"] = selection
        advanced_form.addRow("Checkpoint selection", selection)
        selection.currentIndexChanged.connect(self.save_preferences)
        selection.currentIndexChanged.connect(self.refresh_checkpoint_controls)
        alpha = QDoubleSpinBox()
        alpha.setDecimals(3)
        alpha.setRange(0, 1)
        alpha.setSingleStep(0.05)
        alpha.setValue(0.75)
        alpha.setToolTip(
            "Combined ESR: alpha × Validation ESR + (1 - alpha) × Train ESR. "
            "Enabled only for Combined ESR checkpoint selection."
        )
        self.advanced_fields["checkpoint_alpha"] = alpha
        advanced_form.addRow("Checkpoint alpha", alpha)
        alpha.valueChanged.connect(self.save_preferences)
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
        hyper_rate_preset = QComboBox()
        for field in (rate_preset, hyper_rate_preset):
            field.addItem("per epoch", "manual")
            field.addItem("per 62 optimizer updates (A2 cadence)", "a2_updates")
            field.setToolTip(
                "Cadence of this network's LR Decay. Legacy resumes retain the original checkpoint clock."
            )
        self.advanced_fields["rate_preset"] = rate_preset
        self.advanced_fields["hypernetwork_rate_preset"] = hyper_rate_preset
        for value_key, selector in (
            ("lr_decay", rate_preset),
            ("hypernetwork_lr_decay", hyper_rate_preset),
        ):
            value = self.advanced_fields[value_key]
            caption = advanced_form.labelForField(value).text()
            row_index, _ = advanced_form.getWidgetPosition(value)
            removed = advanced_form.takeRow(value)
            if removed.labelItem:
                removed.labelItem.widget().hide()
                removed.labelItem.widget().deleteLater()
            pair = QWidget()
            pair.setObjectName("rateCadenceRow")
            pair.setStyleSheet("QWidget#rateCadenceRow { background: transparent; }")
            pair_layout = QHBoxLayout(pair)
            pair_layout.setContentsMargins(0, 0, 0, 0)
            pair_layout.addWidget(value)
            pair_layout.addWidget(selector)
            advanced_form.insertRow(row_index, caption, pair)
            if value_key == "hypernetwork_lr_decay":
                self.hypernetwork_rate_row = pair
        rate_preset.activated.connect(self.save_preferences)
        hyper_rate_preset.activated.connect(self.save_preferences)
        self.rate_preview = label("", "muted")
        self.rate_preview.setWordWrap(True)
        advanced_form.addRow(self.rate_preview)
        self.preset_description = label("", "muted")
        advanced_form.addRow(self.preset_description)
        self.advanced_fields["hypernetwork_learning_rate"].setToolTip(
            "Independent learning rate for the knob-to-weight network. Choose its own cadence beside Hypernetwork LR Decay. Parametric training only."
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
        benchmark = QCheckBox("cuDNN benchmark (CUDA)")
        benchmark.setToolTip(
            "Benchmark convolution algorithms for speed. Disabling it stabilizes algorithm selection, but is not sufficient for fully deterministic training. Applies to new training instances."
        )
        benchmark.setChecked(True)
        self.advanced_fields["cudnn_benchmark"] = benchmark
        advanced_form.addRow(benchmark)
        benchmark.toggled.connect(self.save_preferences)
        deterministic = QCheckBox("Deterministic training")
        deterministic.setToolTip(
            "Use deterministic PyTorch algorithms and a fixed per-epoch batch order. Forces benchmark off. Unsupported operations raise an error. Results are reproducible within the same supported software/hardware environment; training can be slower."
        )
        self.advanced_fields["deterministic_training"] = deterministic
        advanced_form.addRow(deterministic)
        deterministic.toggled.connect(self._deterministic_changed)
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
        layout.addWidget(self.metadata_card)
        layout.addWidget(self.training_settings)
        if sys.platform != "darwin":
            self.gpu_setup_button = button("Add NVIDIA GPU support", self.configure_gpu)
            box.addWidget(self.gpu_setup_button)
            self.refresh_gpu_button()
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
                "Each row is an independent training instance. All rows can be reordered; the waiting jobs follow that order; removing a row keeps its models and checkpoints on disk.",
                "muted",
            )
        )
        self.queue_run_row = QHBoxLayout()
        layout.addLayout(self.queue_run_row)
        self.queue_table = QTableWidget(0, 7)
        corner = self.queue_table.findChild(QAbstractButton)
        if corner:
            corner.setText("#")
        self.queue_table.setHorizontalHeaderLabels(
            [
                "Name",
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
        self.queue_table.setItemDelegate(PreserveSelectionTextDelegate(self.queue_table))
        self.queue_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.queue_table.horizontalHeader().setStretchLastSection(False)
        self.queue_table.setMinimumHeight(100)
        self.queue_table.setMaximumHeight(16777215)
        for column, width in enumerate((140, 120, 110, 90, 100, 65, 85)):
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
        self.resume_options_widget.hide()
        self.training_status = label("Add training from a project to begin.", "muted")
        layout.addWidget(self.training_status)
        grid = QGridLayout()
        self.metrics = {}
        for index, (name, text) in enumerate(
            (
                ("epoch", "EPOCH"),
                ("esr", "BEST VALIDATION ESR"),
                ("train_esr", "BEST TRAIN ESR"),
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
        self.pause_button = button("Pause selected training", self.pause_training)
        self.stop_button = button("Cancel selected training", self.stop_training)
        self.models_button = button("Open model folder", self.open_models)
        row.addWidget(self.queue_start_button)
        row.addStretch()
        row.addWidget(self.pause_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.resume_button)
        row.addWidget(self.retrain_button)
        self.export_best_button = button("Export best model", self.export_best_model)
        row.addWidget(self.export_best_button)
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

    def _deterministic_changed(self, enabled):
        benchmark = self.advanced_fields["cudnn_benchmark"]
        if enabled:
            benchmark.blockSignals(True)
            benchmark.setChecked(False)
            benchmark.blockSignals(False)
        benchmark.setEnabled(not enabled)
        self.save_preferences()

    def load_advanced_options(self, kind=None):
        from .cadences import editable_cadences

        options = editable_cadences(
            self.workspace, self.workspace.training_options(kind)
        )
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
        self.advanced_fields["cudnn_benchmark"].setEnabled(
            not options["deterministic_training"]
        )
        if options["deterministic_training"]:
            field = self.advanced_fields["cudnn_benchmark"]
            field.blockSignals(True)
            field.setChecked(False)
            field.blockSignals(False)
        self.refresh_hypernetwork_controls(kind)
        self.refresh_checkpoint_controls()
        self.refresh_rate_preview(kind)

    def refresh_checkpoint_controls(self, *_):
        if not self.workspace:
            return
        # The combo already reflects the new mode while save_preferences is
        # still switching away from the previous workspace capture type.
        parametric = self.capture_kind.currentData() == "parametric"
        method = self.advanced_fields["checkpoint_method"].currentData()
        self.advanced_fields["checkpoint_method"].setEnabled(parametric)
        self.advanced_fields["checkpoint_alpha"].setEnabled(
            parametric and method == "combined"
        )

    def apply_rate_preset(self, *_):
        self.save_preferences()

    def apply_hypernetwork_rate_preset(self, *_):
        self.save_preferences()

    def refresh_rate_preview(self, kind=None):
        if not self.workspace:
            return
        parametric = (
            kind or self.workspace.preferences["capture_type"]
        ) == "parametric"
        self.advanced_fields["rate_preset"].setEnabled(parametric)
        hyper_preset = self.advanced_fields["hypernetwork_rate_preset"].currentData()
        self.advanced_fields["hypernetwork_rate_preset"].setEnabled(parametric)
        self.advanced_fields["hypernetwork_lr_decay"].setEnabled(parametric)
        matched = parametric and self.advanced_fields["rate_preset"].currentData() in (
            "a2_updates",
            "a2_controls",
            "a2_slow",
            "a2_dataset",
            "a2_dataset_sqrt",
        )
        self.advanced_fields["lr_decay"].setEnabled(True)
        self.advanced_fields["lr_decay"].setToolTip(
            "Editable learning-rate multiplier applied at each reduction of the selected cadence."
            if matched
            else "Exponential learning-rate multiplier after each epoch."
        )
        if parametric and hyper_preset != "follow":
            self.rate_preview.setVisible(True)
            try:
                from .recipes import workspace_group_schedules

                from .cadences import editable_cadences

                schedules = workspace_group_schedules(
                    self.workspace,
                    editable_cadences(
                        self.workspace, self.workspace.training_options()
                    ),
                )
                descriptions = []
                for name, schedule in schedules.items():
                    unit = (
                        "epochs"
                        if schedule["interval"] == "epoch"
                        else "optimizer updates on average"
                    )
                    field_name = (
                        "rate_preset" if name == "audio" else "hypernetwork_rate_preset"
                    )
                    preset_name = self.advanced_fields[field_name].currentText()
                    accounting = (
                        f" Dataset: {schedule['steps_per_epoch']} complete updates/epoch; "
                        f"A2 reference: {schedule['reference_steps']}; ratio {schedule['dataset_reference_ratio']:.8g}."
                        if "dataset_reference_ratio" in schedule
                        else ""
                    )
                    descriptions.append(
                        f"{name.title()}: {preset_name}; multiply LR by {schedule['gamma']:g} every {schedule['step_size']:.8g} {unit}. Initial LR {schedule['initial_learning_rate']:.3e}; final LR {schedule['final_learning_rate']:.3e} after {self.workspace.preferences['epochs']} epochs.{accounting}"
                    )
                self.rate_preview.setText(
                    "Independent LR schedules.\n" + "\n".join(descriptions)
                )
            except (ValueError, OSError, RuntimeError) as error:
                self.rate_preview.setText(f"LR cadence preview unavailable: {error}")
            return
        self.rate_preview.setVisible(matched)
        if not matched:
            return
        try:
            from .recipes import workspace_update_schedule

            schedule = workspace_update_schedule(self.workspace)
            formula = {
                "a2_updates": "original A2 update cadence",
                "a2_controls": "knobs × 2^switches",
                "a2_slow": "(knobs × 2^switches) / √2",
                "a2_dataset": "dataset updates / reference updates",
                "a2_dataset_sqrt": "√(dataset updates / reference updates)",
            }[self.advanced_fields["rate_preset"].currentData()]
            self.rate_preview.setText(
                f"A2 reference: {schedule['reference_steps']} updates/epoch (standard input, batch 16, NY 8192). "
                f"This plan: {schedule['steps_per_epoch']} updates/epoch, {schedule['training_seconds']:.1f} s of training audio.\n"
                f"Cadence multiplier: {schedule['cadence_multiplier']:g}× "
                f"({formula}). "
                f"Multiply LR by {schedule['reference_gamma']:g} every {schedule['decay_steps']:.8g} optimizer updates on average; average equivalent per epoch: {schedule['gamma_epoch']:.6g}. "
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
            "hypernetwork_recipe",
            "hypernetwork_learning_rate",
            "hypernetwork_rate_preset",
            "hypernetwork_lr_decay",
            "hypernetwork_mode",
            "hypernetwork_rank",
            "condition_dilated_convolutions",
            "branch_loss_reduction",
        ):
            if key in ("hypernetwork_lr_decay", "hypernetwork_rate_preset"):
                continue
            self.advanced_panel.layout().setRowVisible(
                self.advanced_fields[key], parametric
            )
        self.advanced_panel.layout().setRowVisible(
            self.hypernetwork_rate_row, parametric
        )
        for key in (
            "hypernetwork_mode",
            "condition_dilated_convolutions",
            "branch_loss_reduction",
        ):
            self.advanced_fields[key].setEnabled(parametric)
        self.preset_description.setVisible(parametric)
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
        if index in (1, 2, 3) and not self.workspace:
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
        if index != 6:
            self._stop_settings_audio()
        self.pages.setCurrentIndex(index)
        if index == 6:
            self._sync_settings_audio(retry=True)
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
                read_json(path / "capture_project.json", {}, tolerate_corrupt=True)
                if path.exists()
                else {}
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
        self.scene.stop_blink()
        self.initialize_global_setup(workspace)
        self.workspace = workspace
        from .recovery import RecoveryRequired

        try:
            self.queue_controller.store.register_legacy(workspace)
        except RecoveryRequired:
            pass  # Opening a project must remain available for queue rebuilding.
        self.project_label.setText(
            f"<b>{escape(workspace.project.name or workspace.directory.name)}</b>  ·  Saved locally"
        )
        audio = workspace.project.audio
        self.monitor_return.setChecked(audio.monitor_return_enabled)
        self.verify_asio_timing.setChecked(audio.verify_asio_timing)
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
            metadata = dict(workspace.project.user_metadata)
            if not metadata.get("modeled_by"):
                author = self.settings.value("last_model_author", "", type=str)
                if author:
                    metadata["modeled_by"] = author
                    workspace.project.user_metadata = metadata
            # Seed metadata with the current global Settings only if the model
            # has no explicit value. Editing these fields never writes Settings
            # or the capture-calibration snapshot.
            for key, check, spin in (
                ("input_level_dbu", self.input_calibration, self.input_dbu),
                ("output_level_dbu", self.output_calibration, self.output_dbu),
            ):
                if key not in metadata and check.isChecked():
                    metadata[key] = spin.value()
            workspace.project.user_metadata = dict(metadata)
            self.metadata_editor.set_metadata(metadata)
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
            self.lhs_method.setCurrentIndex(
                max(0, self.lhs_method.findData(prefs.get("lhs_method", "standard")))
            )
            self.validation_method.setCurrentIndex(
                max(0, self.validation_method.findData(
                    prefs.get("validation_method", "independent")
                ))
            )
            # Hidden training method is pinned to Normal for all parametric captures.
            self.mode.setCurrentIndex(self.mode.findData("normal"))
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

    def open_asio_settings(self):
        from .audio_devices import show_asio_control_panel

        device = self.device.currentData()
        if self.busy() or self.play_worker is not None or not device or device.host_api != "ASIO":
            return
        try:
            self.close_audio()
            show_asio_control_panel(device, int(self.winId()))
            self.refresh_devices()
        except Exception as error:
            self.notify_error(error)

    def refresh_devices(self):
        if self.busy():
            return
        self._audio_devices_dirty = False
        self._settings_audio_error_key = None
        self.remember_setup()
        self.close_audio()
        selected = self.device.currentData()
        wanted = (self.last_setup() or {}).get("audio", {})
        identity = (
            (selected.host_api, selected.name)
            if selected
            else (wanted.get("host_api"), wanted.get("input_device"))
        )
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
            if identity[1]:
                index = next(
                    (
                        i
                        for i, info in enumerate(self._devices)
                        if (info.host_api, info.name) == identity
                    ),
                    None,
                )
                if index is None:
                    self.device.addItem(f"Unavailable: {identity[1]}", None)
                    index = self.device.count() - 1
                self.device.setCurrentIndex(index)
        except Exception as error:
            self.route_status.setText(str(error))
        finally:
            self._loading = False
        self._device_changed()

    def _reconnect_device(self):
        from PySide6.QtWidgets import QApplication

        if self.busy() or self.session or QApplication.activeModalWidget():
            return
        # Refresh only after a hardware notification or an explicit click.
        # Reinitializing PortAudio on a timer repeatedly unloads ASIO drivers.
        if not self._audio_devices_dirty:
            if sys.platform == "win32" or self.device.currentData() is not None:
                return
        try:
            self.refresh_devices()
        except Exception:
            pass  # A temporary driver error must not replace the saved interface.

    def _device_changed(self, *_):
        if self._loading:
            return
        self.close_audio()
        info = self.device.currentData()
        self._populate_audio_controls(info)
        self.save_audio()
        self.asio_settings_button.setVisible(info is not None and info.host_api == "ASIO")

    def _populate_audio_controls(self, info):
        if info:
            result = capabilities(info)
            same_device = getattr(self, "_audio_device_key", None) == (
                info.host_api,
                info.name,
            )
            self._audio_device_key = (info.host_api, info.name)
            remembered = self.remembered_audio_route(info)
            previous_buffer = remembered.get(
                "blocksize", self.buffer.currentData() if same_device else None
            )
            for key, combo, names, default in (
                ("input_channel", self.input_channel, result.inputs, 1),
                ("loopback_input_channel", self.loop_input, result.inputs, 2),
                ("output_channel", self.output_channel, result.outputs, 1),
                ("loopback_output_channel", self.loop_output, result.outputs, 2),
                (
                    "instrument_input_channel",
                    self.instrument_input,
                    result.inputs,
                    None,
                ),
                ("monitor_output_left", self.monitor_left, result.outputs, None),
                ("monitor_output_right", self.monitor_right, result.outputs, None),
            ):
                previous = remembered.get(
                    key, combo.currentData() if same_device else default
                )
                combo.blockSignals(True)
                combo.clear()
                combo.addItem("None", None)
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
            if "loopback_enabled" in remembered:
                self.loop_enabled.blockSignals(True)
                self.loop_enabled.setChecked(remembered["loopback_enabled"])
                self.loop_enabled.blockSignals(False)
            if "playback_blocksize" in remembered:
                self._playback_blocksize = remembered["playback_blocksize"]
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
            input_channel=self.input_channel.currentData(),
            instrument_input_channel=self.instrument_input.currentData(),
            monitor_output_left=self.monitor_left.currentData(),
            monitor_output_right=self.monitor_right.currentData(),
            monitor_return_enabled=self.monitor_return.isChecked(),
            verify_asio_timing=self.verify_asio_timing.isChecked(),
            output_channel=self.output_channel.currentData(),
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
        required = bool(
            self.workspace and self.capture_kind.currentData() == "parametric"
        )
        if required:
            self.loop_enabled.blockSignals(True)
            self.loop_enabled.setChecked(True)
            self.loop_enabled.blockSignals(False)
        self.loop_enabled.setEnabled(not required)
        self.loop_enabled.setToolTip(
            "A separate clean loopback is required for parametric captures."
            if required
            else "Use a separate clean timing reference."
        )
        self.loop_input.setEnabled(self.loop_enabled.isChecked())
        self.loop_output.setEnabled(self.loop_enabled.isChecked())

    def save_audio(self, *_):
        if self._loading or self.busy():
            return
        self.close_audio()
        self._diagram()
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
            worker = getattr(self, "settings_audio_worker", None)
            if worker is not None:
                worker.set_noise(worker.noise[0], level)
            if self.session:
                self.session.close_audio()
                self.session = None
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
            if (
                self.input_channel.currentData() is None
                or self.output_channel.currentData() is None
            ):
                raise ValueError(
                    "Select a Send output and Return input before calibrating."
                )
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
        self._stop_settings_audio()
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

    def _route_metered(self, send, returned, loopback_send=0.0, loopback_return=0.0):
        import math

        for name, value in (
            ("send", send),
            ("return", returned),
            ("loopback_send", loopback_send),
            ("loopback_return", loopback_return),
        ):
            getattr(self, f"route_{name}_meter").set_levels([value])
            text = f"{20 * math.log10(value):.1f} dBFS" if value > 1e-6 else "−∞ dBFS"
            getattr(self, f"route_{name}_peak").setText(text)

    def refresh_controls(self):
        if not self.workspace:
            return
        self.control_table.setRowCount(len(self.workspace.draft_controls))
        for row, data in enumerate(self.workspace.draft_controls):
            c = control_from_dict(data)
            if isinstance(c, SwitchSpec):
                cells = (
                    c.name,
                    (
                        "Simple Switch"
                        if c.planning_mode == "simple"
                        else "Complex Switch"
                    ),
                    " / ".join(c.enum_names),
                    c.enum_names[c.default],
                )
            else:
                cells = (
                    c.name,
                    "Knob",
                    f"{c.min:g} … {c.max:g}  ·  step {c.step:g}"
                    + ("  ·  Gain/Drive" if c.is_gain else "")
                    + ("  ·  Clean volume" if c.clean_volume else "")
                    + ("  ·  Avoid min" if c.avoid_zero else "")
                    + ("  ·  Avoid max" if c.avoid_max else "")
                    + (
                        {
                            "lower": "  ·  Lower-half density",
                            "upper": "  ·  Upper-half density",
                        }.get(c.sampling_bias, "")
                    ),
                    f"{c.default:g}",
                )
            for col, value in enumerate(cells):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.control_table.setItem(row, col, item)
        self.control_table.resizeRowsToContents()
        self.control_table.setFixedHeight(
            max(
                205,
                self.control_table.horizontalHeader().height()
                + sum(
                    self.control_table.rowHeight(row)
                    for row in range(self.control_table.rowCount())
                )
                + self.control_table.horizontalScrollBar().sizeHint().height()
                + 4,
            )
        )
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
            before = previous[index]
            after = dialog.result_control.to_dict()
            if before["name"] != after["name"]:
                try:
                    self.workspace.rename_control(before["name"], after["name"])
                except ValueError as error:
                    self.notify_error(error)
                    return
                self.close_audio()
            self.workspace.draft_controls[index] = after
        try:
            self.workspace.controls()
        except ValueError as error:
            self.workspace.draft_controls = previous
            self.notify_error(error)
            return
        self.refresh_controls()
        if index is None or {
            k: v for k, v in previous[index].items() if k != "name"
        } != {k: v for k, v in dialog.result_control.to_dict().items() if k != "name"}:
            self.refresh_recommendation(True)
        self.workspace.save()

    def move_control(self, direction):
        if not self.workspace or self.busy():
            return
        index = self.control_table.currentRow()
        target = index + direction
        if index < 0 or not 0 <= target < len(self.workspace.draft_controls):
            return
        self.workspace.move_control(index, target)
        self.close_audio()
        self.refresh_controls()
        self.control_table.selectRow(target)

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
        if all(isinstance(c, SwitchSpec) for c in controls) and val == 0:
            val = train
        count = corner_capture_count(controls)
        self.recommendation.setText(
            f"Suggested: {train} training + {val} validation captures, plus up to {count} corners.\n"
            f"{combinations} Complex Switch combination(s); Simple Switches share the LHS plan. Each enabled split covers every Complex combination. Counts are starting guidelines, not an accuracy guarantee."
        )
        if autofill:
            self.n_train.setValue(train)
            self.n_validation.setValue(val)

    def save_preferences(self, *_):
        if self._loading or not self.workspace or self.busy():
            return
        prefs = self.workspace.preferences
        # Do not let hidden Active Learning survive from an old project or
        # programmatic combo change; retain all code for later re-enabling.
        if self.mode.currentData() != "normal":
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(self.mode.findData("normal"))
            self.mode.blockSignals(False)
        for field, key in (
            (self.n_train, "n_train"),
            (self.n_validation, "n_validation"),
            (self.seed, "seed"),
            (self.rounds, "rounds"),
            (self.additions, "additions"),
            (self.epochs, "epochs"),
        ):
            prefs[key] = field.value()
        from .workspace import recommended_epochs

        previous_type = prefs["capture_type"]
        next_type = self.capture_kind.currentData()
        previous_input = prefs["input_mode"]
        next_input = self.signal_mode.currentData()
        current_epochs = self.epochs.value()
        prefs[previous_type + "_epochs"] = current_epochs
        prefs[previous_type + "_input_mode"] = previous_input
        if previous_type != next_type:
            next_input = prefs.get(
                next_type + "_input_mode",
                "trunc" if next_type == "parametric" else "standard",
            )
            self.signal_mode.blockSignals(True)
            self.signal_mode.setCurrentIndex(
                max(0, self.signal_mode.findData(next_input))
            )
            self.signal_mode.blockSignals(False)
            prefs["epochs"] = prefs.get(
                next_type + "_epochs", recommended_epochs(next_type, next_input)
            )
            self.epochs.blockSignals(True)
            self.epochs.setValue(prefs["epochs"])
            self.epochs.blockSignals(False)
        elif (
            next_type == "parametric"
            and previous_input != next_input
            and current_epochs == recommended_epochs("parametric", previous_input)
        ):
            # Update the recommendation only while the user is still using the
            # previous suggested value. Never overwrite a custom epoch budget.
            prefs["epochs"] = recommended_epochs("parametric", next_input)
            prefs["parametric_epochs"] = prefs["epochs"]
            self.epochs.blockSignals(True)
            self.epochs.setValue(prefs["epochs"])
            self.epochs.blockSignals(False)
        prefs[next_type + "_input_mode"] = next_input
        old_options = self.workspace.training_options(previous_type)
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
        from .cadences import preserve_unchanged_cadences

        prefs["advanced_" + previous_type] = preserve_unchanged_cadences(
            self.workspace, old_options, prefs["advanced_" + previous_type]
        )
        if previous_type != next_type:
            next_key = "static_batch_size" if next_type == "static" else "batch_size"
            self.batch_size.blockSignals(True)
            self.batch_size.setValue(prefs[next_key])
            self.batch_size.blockSignals(False)
            self.load_advanced_options(next_type)
        prefs.update(
            capture_type=self.capture_kind.currentData(),
            mode="normal",  # Active Learning remains implemented but unavailable.
            device=self.accelerator.currentData(),
            corners=self.corners.isChecked(),
            lhs_method=self.lhs_method.currentData(),
            validation_method=self.validation_method.currentData(),
            input_mode=self.signal_mode.currentData(),
        )
        static = prefs["capture_type"] == "static"
        self._diagram()
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
            else "ParametricSlimmableContainer · Full + Lite in one .pnam file"
        )
        self.mode.setEnabled(not static)
        for widget in (
            self.n_train, self.n_validation, self.seed, self.lhs_method,
            self.validation_method, self.corners
        ):
            widget.setEnabled(not static)
        self.training_title.setText(
            "Train your A2 batch." if static else "Train your ParametricSlimmableContainer."
        )
        for widget in (
            self.mode,
            self.n_train,
            self.n_validation,
            self.seed,
            self.lhs_method,
            self.validation_method,
            self.corners,
            self.rounds,
            self.additions,
        ):
            self.plan_form.setRowVisible(
                widget, not static and widget not in (self.mode, self.rounds, self.additions)
            )
        self.rounds.setEnabled(False)
        self.additions.setEnabled(False)
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
            "ParametricSlimmableContainer · Full + Lite in one .pnam file", "eyebrow"
        )
        return self.model_format

    def save_metadata(self, metadata):
        if self.workspace and not self._loading and not self.busy():
            previous = self.workspace.project.user_metadata.get("modeled_by")
            author = metadata.get("modeled_by")
            if author and author != previous:
                self.settings.setValue("last_model_author", author)
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
        if delay is not None and not entry.validation_from_tail:
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
        self.volume_check_button.setVisible(
            project.output_gain_state.get("screen_decision") == "warning"
        )
        static = project.capture_type == "static"
        self.scene.setVisible(not static)
        self.static_card.setVisible(static)
        self.load_capture_button.setVisible(static)
        if static:
            self.record_button.setText("Record")
        if self.worker is None:
            self.capture_status.setText(
                "Select a named capture below. Record it here or load the aligned return for the input selected in Plan."
                if static
                else "Select a capture below. Set the real controls to match the diagram."
            )
        self.rig_ready.setText(
            "My rig is set for this named capture"
            if static
            else "My rig matches the control positions shown above"
        )
        if static:
            required = {e.y_path for e in self.workspace.training_capture_entries()}
            groups = [
                [e for e in group if e.y_path in required]
                for group in self.workspace.static_groups()
            ]
            entries = [
                next((e for e in group if e.status != "captured"), group[0])
                for group in groups
            ]
            headers = ["Name", "Status", "Latency", "Peak (dBFS)", "Warnings"]
        else:
            entries = order_captures(project.entries, project.control_specs())
            headers = [
                "Type",
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
                    "Volume" if entry.split == "gain" else entry.split.title(),
                    params,
                    entry.status.title(),
                    *self._capture_metrics(entry),
                ]
            for col, text in enumerate(columns):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                running = (
                    self.worker is not None
                    and entry.y_path == getattr(self, "_recorded_capture_path", None)
                    and entry.status != "captured"
                )
                has_warning = bool(
                    self._capture_metrics(entry)[2] not in ("—", "")
                )
                # Running overrides warnings and paints every column, even when selected.
                color = (
                    "#00ff00"
                    if running
                    else "#ffff00"
                    if has_warning
                    else "#00ffff"
                    if entry.status == "captured"
                    else "#f0f0f0"
                )
                item.setForeground(QColor(color))
                status_col = 1 if static else 2
                if col == status_col and running:
                    item.setText("Running")
                self.capture_table.setItem(row, col, item)
        self.capture_table.blockSignals(False)
        counted = self.workspace.training_capture_entries() if static else project.entries
        completed = sum(e.status == "captured" for e in counted)
        self.capture_summary.setText(
            f"{len(entries)} static model(s) · {completed} of {len(counted)} required signals captured. Imported WAVs use their held-out tail for validation."
            if static
            else f"{completed} of {len(entries)} captured · {len(entries) - completed} remaining. Ordered to change fewer controls and turn knobs less."
        )
        if entries:
            row = next(
                (i for i, e in enumerate(entries) if e.status != "captured"),
                len(entries) - 1,
            )
            self.capture_table.blockSignals(True)
            self.capture_table.selectRow(row)
            self.capture_table.blockSignals(False)
            self.select_capture(automatic=True)
        self.rig_ready.setChecked(False)
        self.update_enabled()

    def select_capture(self, automatic=False):
        if not automatic:
            self.scene.stop_blink()
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
                self.record_button.setText("Record")
                return
            self.record_button.setText(
                "Re-record this setting"
                if entry.status == "captured"
                else "Record this setting"
            )

    def load_static_wav(self):
        if self.busy() or not self.workspace or self.workspace.project.capture_type != "static":
            return
        row = self.capture_table.currentRow()
        if not 0 <= row < len(self._capture_entries):
            return
        entry = self._capture_entries[row]
        path, _ = self.choose_open_file(
            "Choose the aligned return for the input selected in Plan",
            "WAV audio (*.wav)",
        )
        if not path:
            return
        try:
            session = self._get_session()
            session.import_static_return(entry.capture_id, path)
            self._recorded_capture_path = entry.y_path
            self.workspace.project = session.project
            self.refresh_captures()
            self.capture_progress.setValue(100)
            self.capture_status.setText("Imported WAV. The last nine seconds are reserved for validation.")
        except Exception as error:
            self.notify_error(error)

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
            audio = self.audio_settings()
            if audio.monitor_return_enabled and audio.host_api != "ASIO":
                raise ValueError(
                    "Return monitoring during capture currently requires an ASIO interface."
                )
            if audio.input_channel is None or audio.output_channel is None:
                raise ValueError(
                    "Select a Send output and Return input in Settings before capturing."
                )
            if self.workspace.project.capture_type == "parametric" and (
                not audio.loopback_enabled
                or audio.loopback_input_channel == audio.input_channel
                or audio.loopback_output_channel == audio.output_channel
            ):
                raise ValueError(
                    "Parametric captures require separate clean loopback input and output channels. Configure them in Settings."
                )
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
            captures = (
                group if static and any(e.validation_from_tail for e in group)
                else [e for e in group if e.status != "captured"] or group
            )
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
            self._recorded_control_values = dict(entry.params)
            self._recorded_capture_path = entry.y_path

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
            status_col = 1 if static else 2
            for col in range(self.capture_table.columnCount()):
                item = self.capture_table.item(index, col)
                if item is not None:
                    item.setForeground(QColor("#00ff00"))
                    if col == status_col:
                        item.setText("Running")
                        item.setToolTip("Running")
        except Exception as error:
            self.notify_error(error)

    def _capture_done(self, qa):
        QApplication.beep()
        if self.workspace and self.session:
            self.workspace.project = self.session.project
            last = next(
                (
                    e
                    for e in self.workspace.project.entries
                    if e.y_path == getattr(self, "_recorded_capture_path", None)
                ),
                None,
            )
            if last is not None and last.split == "gain":
                from nam.capture.output_gain import advance_calibration
                from nam.capture.project import save_project

                try:
                    advance_calibration(
                        self.workspace.project, self.workspace.directory
                    )
                    if (
                        self.workspace.project.output_gain_state.get("screen_decision")
                        == "warning"
                    ):
                        self.review_volume_check()
                except ValueError as error:
                    self.workspace.project.output_gain = None
                    self.workspace.project.output_gain_state.update(
                        status="failed", message=str(error)
                    )
                    self.notify_error(str(error))
                save_project(self.workspace.project, self.workspace.directory)
        reports = qa if isinstance(qa, list) else [qa]
        self.capture_progress.setValue(100)
        self.capture_status.setText(f"Saved {len(reports)} signal recording(s)")
        self.refresh_captures()
        row = self.capture_table.currentRow()
        if 0 <= row < len(self._capture_entries):
            following = self._capture_entries[row]
            if following.y_path != getattr(
                self, "_recorded_capture_path", following.y_path
            ):
                self.capture_progress.setValue(0)
                previous = getattr(self, "_recorded_control_values", {})
                self.scene.start_blink(
                    name
                    for name, value in following.params.items()
                    if previous.get(name) != value
                )
        messages = [message for report in reports for message in report.messages]
        if messages:
            QMessageBox.warning(self, "Capture quality check", "\n".join(messages))

    def _ask_volume_warning(self, message):
        box = QMessageBox(self)
        box.setWindowTitle("Clean volume verification")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText("Peak and RMS levels show a possible non-linear volume response.")
        box.setInformativeText(
            message
            + "\n\nCheck the audio interface or reamp box for distortion, or go back to Plan and remove Clean volume."
        )
        proceed = box.addButton("Dismiss", QMessageBox.ButtonRole.AcceptRole)
        back = box.addButton("Go back to Plan", QMessageBox.ButtonRole.ActionRole)
        box.setDefaultButton(proceed)
        box.exec()
        return (
            "dismiss"
            if box.clickedButton() is proceed
            else "plan" if box.clickedButton() is back else "dismiss"
        )

    def review_volume_check(self):
        if not self.workspace:
            return
        project = self.workspace.project
        state = project.output_gain_state
        if state.get("screen_decision") != "warning":
            return
        choice = self._ask_volume_warning(state.get("warning_message", ""))
        if choice == "dismiss":
            state["screen_decision"] = "accepted"
            from nam.capture.output_gain import advance_calibration

            advance_calibration(project, self.workspace.directory)
        elif choice == "plan":
            self.workspace.save()
            self.navigate(1)
            return
        self.workspace.save()
        self.refresh_captures()

    def cancel_capture(self):
        if hasattr(self, "capture_progress"):
            self.capture_progress.setValue(0)
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

    def _tick_queue_seconds(self):
        elapsed = self._elapsed_clock.value()
        self.metrics["elapsed"].setText(duration(elapsed))
        eta = getattr(self, "_queue_eta", None)
        delta = (
            time.monotonic() - getattr(self, "_queue_eta_started", time.monotonic())
            if self._elapsed_clock.running
            else 0
        )
        self.metrics["eta"].setText(
            duration(max(0, eta - delta) if eta is not None else None)
        )
        if self._elapsed_clock.running and self.isVisible():
            self.queue_seconds_timer.start(max(1, math.ceil((1 - elapsed % 1) * 1000)))
        else:
            self.queue_seconds_timer.stop()

    def _render_job_progress(self):
        progress = training.job_progress(self.job)
        spec = (
            read_json(self.job / "spec.json", {}, tolerate_corrupt=True)
            if self.job
            else {}
        )
        total_epochs = spec.get("preferences", {}).get("epochs")
        if not progress:
            self._elapsed_clock.update(None, 0, False)
            self.queue_seconds_timer.stop()
            self.esr_chart.set_history([], total_epochs)
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
        self.esr_chart.set_history(progress.get("esr_history", []), total_epochs)
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
            message += f"\nA checkpoint is available. Resume restarts epoch {progress.get('restart_epoch', 1)}."
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
        running = (
            self.selected_job_id is not None
            and self.selected_job_id == self.queue_controller.active_id
            and status in ("running", "scoring", "pausing", "stopping")
        )
        self._elapsed_clock.update(str(self.job), progress.get("elapsed", 0), running)
        self._queue_eta = progress.get("eta")
        self._queue_eta_started = time.monotonic()
        self._tick_queue_seconds()
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
                horizontal = self.details.horizontalScrollBar().value()
                self.details.setPlainText(content)
                if follow:
                    self.details.moveCursor(QTextCursor.MoveOperation.End)
                else:
                    scrollbar.setValue(position)
                self.details.horizontalScrollBar().setValue(horizontal)
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

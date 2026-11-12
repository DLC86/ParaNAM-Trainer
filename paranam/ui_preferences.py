"""Current interface setup and a shared, remembered file-picker directory."""

import json
from pathlib import Path

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QFileDialog

from nam.capture.project import AudioSettingsModel


class UIPreferences:
    def audio_device_routes(self):
        try:
            routes = json.loads(self.settings.value("audio_device_routes", "{}"))
            return routes if isinstance(routes, dict) else {}
        except (ValueError, TypeError):
            return {}

    @staticmethod
    def audio_device_identity(info):
        # PortAudio indices can change after reconnecting or refreshing devices.
        return json.dumps([info.host_api, info.name], ensure_ascii=False)

    def remembered_audio_route(self, info):
        return self.audio_device_routes().get(self.audio_device_identity(info), {})

    def dialog_directory(self, fallback=""):
        previous = self.settings.value("last_directory", "")
        if previous and Path(previous).is_dir():
            return str(previous)
        return str(
            fallback
            or QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.DocumentsLocation
            )
        )

    def remember_directory(self, path, is_directory=False):
        if path:
            self.settings.setValue(
                "last_directory", str(Path(path) if is_directory else Path(path).parent)
            )

    def choose_directory(self, title, fallback=""):
        path = QFileDialog.getExistingDirectory(
            self, title, self.dialog_directory(fallback)
        )
        self.remember_directory(path, True)
        return path

    def choose_open_file(self, title, file_filter):
        path, selected = QFileDialog.getOpenFileName(
            self, title, self.dialog_directory(), file_filter
        )
        self.remember_directory(path)
        return path, selected

    def choose_save_file(self, title, default, file_filter):
        default = Path(default)
        path, selected = QFileDialog.getSaveFileName(
            self,
            title,
            str(Path(self.dialog_directory(default.parent)) / default.name),
            file_filter,
        )
        self.remember_directory(path)
        return path, selected

    def remember_setup(self):
        if self._loading or getattr(self, "_restoring_setup", False):
            return
        audio = self.audio_settings().model_dump()
        if self.device.currentData() is None:
            audio = (self.last_setup() or {}).get("audio", audio)
        setup = dict(
            audio=audio,
            capture_level_dbfs=self.capture_level.level(),
            input_level_dbu=(
                self.input_dbu.value() if self.input_calibration.isChecked() else None
            ),
            output_level_dbu=(
                self.output_dbu.value() if self.output_calibration.isChecked() else None
            ),
            instrument_level_dbu=(
                self.instrument_dbu.value()
                if self.instrument_calibration.isChecked()
                else None
            ),
        )
        setup["instrument_dbu_value"] = self.instrument_dbu.value()
        setup["calibration_frequency"] = getattr(self, "_calibration_frequency", 1000.0)
        setup["playback_blocksize"] = getattr(self, "_playback_blocksize", 128)
        self.settings.setValue("last_audio_setup", json.dumps(setup))
        info = self.device.currentData()
        if info is not None:
            routes = self.audio_device_routes()
            routes[self.audio_device_identity(info)] = {
                key: field.currentData()
                for key, field in (
                    ("input_channel", self.input_channel),
                    ("output_channel", self.output_channel),
                    ("loopback_input_channel", self.loop_input),
                    ("loopback_output_channel", self.loop_output),
                    ("instrument_input_channel", self.instrument_input),
                    ("monitor_output_left", self.monitor_left),
                    ("monitor_output_right", self.monitor_right),
                    ("blocksize", self.buffer),
                )
            }
            routes[self.audio_device_identity(info)].update(
                loopback_enabled=self.loop_enabled.isChecked(),
                playback_blocksize=setup["playback_blocksize"],
            )
            self.settings.setValue("audio_device_routes", json.dumps(routes))

    def last_setup(self):
        value = self.settings.value("last_audio_setup", "")
        try:
            if value:
                return json.loads(value)
            recent = self.settings.value("recent", []) or []
            recent = [recent] if isinstance(recent, str) else recent
            if recent and (Path(recent[0]) / "capture_project.json").is_file():
                from .workspace import Workspace

                workspace = Workspace(Path(recent[0]))
                pending = workspace.preferences.get("pending_calibration", {})
                return dict(
                    audio=workspace.project.audio.model_dump(),
                    capture_level_dbfs=workspace.preferences["capture_level_dbfs"],
                    input_level_dbu=pending.get(
                        "input_level_dbu", workspace.project.input_level_dbu
                    ),
                    output_level_dbu=pending.get(
                        "output_level_dbu", workspace.project.output_level_dbu
                    ),
                    instrument_level_dbu=workspace.preferences.get(
                        "instrument_level_dbu"
                    ),
                    calibration_frequency=workspace.preferences.get(
                        "calibration_frequency", 1000.0
                    ),
                )
            return None
        except (ValueError, TypeError, OSError):
            return None

    def initialize_global_setup(self, workspace):
        """One-time migration; opening projects never changes an existing setup."""
        if self.settings.value("last_audio_setup", ""):
            return
        pending = workspace.preferences.get("pending_calibration", {})
        setup = dict(
            audio=workspace.project.audio.model_dump(),
            input_level_dbu=pending.get(
                "input_level_dbu", workspace.project.input_level_dbu
            ),
            output_level_dbu=pending.get(
                "output_level_dbu", workspace.project.output_level_dbu
            ),
            instrument_level_dbu=workspace.preferences.get("instrument_level_dbu"),
            capture_level_dbfs=workspace.preferences["capture_level_dbfs"],
            calibration_frequency=workspace.preferences.get(
                "calibration_frequency", 1000.0
            ),
            playback_blocksize=workspace.preferences.get("playback_blocksize", 128),
        )
        self.settings.setValue("last_audio_setup", json.dumps(setup))
        self.restore_startup_setup()

    def instrument_reference(self):
        return (
            self.instrument_dbu.value()
            if self.instrument_calibration.isChecked()
            else None
        )

    def prepare_capture_setup(self):
        """Copy physical references into a new capture plan, never into old audio."""
        if not self.workspace:
            return
        self.workspace.preferences["capture_level_dbfs"] = self.capture_level.level()
        self.workspace.preferences["pending_calibration"] = {
            "input_level_dbu": (
                self.input_dbu.value() if self.input_calibration.isChecked() else None
            ),
            "output_level_dbu": (
                self.output_dbu.value() if self.output_calibration.isChecked() else None
            ),
        }

    def remember_capture_calibration(self):
        if not self.workspace:
            return
        snapshot = {
            key: (self.last_setup() or {}).get(key)
            for key in (
                "input_level_dbu",
                "output_level_dbu",
                "capture_level_dbfs",
                "instrument_level_dbu",
                "calibration_frequency",
            )
        }
        self.workspace.preferences["capture_calibration"] = snapshot
        self.workspace.preferences["instrument_level_dbu"] = snapshot[
            "instrument_level_dbu"
        ]
        self.workspace.preferences["calibration_frequency"] = snapshot[
            "calibration_frequency"
        ]
        self.workspace.save()

    def restore_startup_setup(self):
        setup = self.last_setup()
        if not setup:
            return
        self._loading = True
        try:
            audio = AudioSettingsModel(**setup["audio"])
            matched = False
            for index in range(self.device.count()):
                device = self.device.itemData(index)
                if device and (device.name, device.host_api) == (
                    audio.input_device,
                    audio.host_api,
                ):
                    self.device.setCurrentIndex(index)
                    matched = True
                    break
            if not matched and audio.input_device:
                self.device.addItem(f"Unavailable: {audio.input_device}", None)
                self.device.setCurrentIndex(self.device.count() - 1)
            self._populate_audio_controls(self.device.currentData())
            for field, value in (
                (self.input_channel, audio.input_channel),
                (self.output_channel, audio.output_channel),
                (self.loop_input, audio.loopback_input_channel),
                (self.loop_output, audio.loopback_output_channel),
                (self.instrument_input, audio.instrument_input_channel),
                (self.monitor_left, audio.monitor_output_left),
                (self.monitor_right, audio.monitor_output_right),
                (self.buffer, audio.blocksize),
            ):
                index = field.findData(value)
                if index >= 0:
                    field.setCurrentIndex(index)
            self.loop_enabled.setChecked(audio.loopback_enabled)
            self.capture_level.set_level(setup.get("capture_level_dbfs", 0))
            for check, field, key in (
                (self.input_calibration, self.input_dbu, "input_level_dbu"),
                (self.output_calibration, self.output_dbu, "output_level_dbu"),
                (
                    self.instrument_calibration,
                    self.instrument_dbu,
                    "instrument_level_dbu",
                ),
            ):
                check.setChecked(setup.get(key) is not None)
                field.setValue(setup[key] if setup.get(key) is not None else 12)
            if "instrument_dbu_value" in setup:
                self.instrument_dbu.setValue(setup["instrument_dbu_value"])
        finally:
            self._loading = False
        self._calibration_frequency = setup.get("calibration_frequency", 1000.0)
        self._playback_blocksize = setup.get("playback_blocksize", 128)
        self.settings.setValue("last_audio_setup", json.dumps(setup))
        self._diagram()

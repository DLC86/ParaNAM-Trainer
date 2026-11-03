"""Queue actions deliberately operate on instances, not the open project."""

from pathlib import Path
import time

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QTableWidgetItem

from . import training
from .training_queue import ACTIVE, FINISHED
from .workspace import Workspace, read_json


class QueueActions:
    def start_training(self):
        if not self.workspace or self.busy():
            return
        try:
            self.save_preferences()
            if self.workspace.plan_changed():
                raise ValueError(
                    "Apply the modified controls/plan before adding training."
                )
            if not self.workspace.training_ready():
                raise ValueError("Complete every planned capture first.")
            self.metadata_editor.emit_metadata()
            self.queue_controller.enqueue(self.workspace)
            self.queue_controller.preparing.ready.connect(self._new_queue_entry)
            self.navigate(4)
            self.training_status.setText("Preparing an independent training instance…")
            self.update_enabled()
        except Exception as error:
            self.notify_error(error)

    def _new_queue_entry(self, entry):
        self.selected_job_id = entry["id"]
        self.refresh_training()

    def run_queue(self):
        self.queue_controller.start()
        if self.queue_controller.active_id:
            self.selected_job_id = self.queue_controller.active_id
        self.refresh_training()

    def pause_training(self):
        if self.selected_job_id:
            self.queue_controller.pause(self.selected_job_id)
            self.refresh_training()

    def stop_training(self):
        if self.selected_job_id:
            self.queue_controller.stop(self.selected_job_id)
            self.refresh_training()

    def resume_training(self):
        try:
            self.queue_controller.store.prepare_resume(
                self.selected_job_id,
                self.resume_batch_size.value(),
                self.resume_device.currentData(),
            )
            self.queue_controller.start()
            self.refresh_training()
        except Exception as error:
            self.notify_error(error)

    def move_job(self, offset):
        if self.selected_job_id:
            self.queue_controller.store.move(self.selected_job_id, offset)
            self.refresh_training()

    def remove_job(self):
        if self.selected_job_id:
            try:
                self.queue_controller.store.remove(self.selected_job_id)
                self.selected_job_id = None
                self.refresh_training()
            except Exception as error:
                self.notify_error(error)

    def open_capture_round(self):
        try:
            entry = self.queue_controller.store.get(self.selected_job_id)
            if entry:
                self._attach(
                    self.queue_controller.store.capture_round(self.selected_job_id)
                )
                self.navigate(2)
        except Exception as error:
            self.notify_error(error)

    def retrain_training(self):
        if self.busy() or self.queue_controller.preparing:
            return
        try:
            workspace = self.queue_controller.store.retrain(self.selected_job_id)
            self._attach(workspace)
            self.navigate(3)
        except Exception as error:
            self.notify_error(error)

    def select_queue_job(self):
        item = self.queue_table.item(self.queue_table.currentRow(), 0)
        if item:
            self.selected_job_id = item.data(Qt.ItemDataRole.UserRole)
            self._log_content = ""
            self.details.clear()
            self._log_reader = None
            self.refresh_training()

    def refresh_queue(self):
        entries = self.queue_controller.store.entries()
        if not any(e["id"] == self.selected_job_id for e in entries):
            self.selected_job_id = self.queue_controller.active_id or (
                entries[-1]["id"] if entries else None
            )
        table = self.queue_table
        table.blockSignals(True)
        table.setRowCount(len(entries))
        pending = 0
        selected = None
        for row, entry in enumerate(entries):
            progress = training.job_progress(entry["job"])
            state = entry["state"]
            if state == "queued":
                pending += 1
            esr = progress.get("best_esr")
            values = (
                str(pending) if state == "queued" else "—",
                entry["name"],
                time.strftime("%d/%m %H:%M", time.localtime(entry["created_at"])),
                "Static A2" if entry["kind"] == "static" else "HyperA2",
                state.replace("_", " ").title(),
                f"{progress.get('epoch', 0)} / {progress.get('epochs', '—')}",
                "—" if esr is None else f"{esr:.5f}",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, entry["id"])
                item.setToolTip(f"{entry['project']}\n{entry['job']}")
                table.setItem(row, column, item)
            if entry["id"] == self.selected_job_id:
                selected = entry
                table.selectRow(row)
        table.blockSignals(False)
        self.job = Path(selected["job"]) if selected else None
        self.process = (
            self.queue_controller.process
            if selected and selected["id"] == self.queue_controller.active_id
            else None
        )
        if selected and getattr(self, "_resume_entry_id", None) != selected["id"]:
            spec = read_json(self.job / "spec.json")
            self.resume_batch_size.setValue(
                spec.get("preferences", {}).get("batch_size", 16)
            )
            self.resume_device.setCurrentIndex(
                max(
                    0,
                    self.resume_device.findData(
                        spec.get("preferences", {}).get("device", "auto")
                    ),
                )
            )
            self._resume_entry_id = selected["id"]
        return selected

    def refresh_training(self):
        if not hasattr(self, "queue_table"):
            return
        self.queue_controller.tick()
        active_id = self.queue_controller.active_id
        previous_active = getattr(self, "_last_active_id", None)
        if active_id and (
            not self.selected_job_id or self.selected_job_id == previous_active
        ):
            self.selected_job_id = active_id
        self._last_active_id = active_id
        selected = self.refresh_queue()
        if selected:
            self._render_job_progress()
        else:
            self.training_status.setText(
                "Add training from a project. Each instance keeps its own captures, settings and checkpoints."
            )
            self.esr_chart.set_history([])
            for metric in self.metrics.values():
                metric.setText("—")
            self.train_progress.setValue(0)
        self.update_enabled()
        if (
            self._close_after_pause
            and not self.queue_controller.active_id
            and not self.queue_controller.preparing
            and not self.worker
            and not getattr(self, "play_worker", None)
            and not getattr(self, "play_loader", None)
        ):
            self.close()

    def open_models(self):
        entry = (
            self.queue_controller.store.get(self.selected_job_id)
            if self.selected_job_id
            else None
        )
        directory = None
        if entry:
            spec = read_json(Path(entry["job"]) / "spec.json")
            if not spec:
                self.notify_error(
                    ValueError("Reconnect the drive containing this training instance.")
                )
                return
            directory = Path(
                spec.get("models_directory", str(Path(spec["project"]) / "models"))
            )
        elif self.workspace:
            directory = self.workspace.directory / "models"
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
            self.remember_directory(directory, True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def update_enabled(self, *_):
        if not hasattr(self, "queue_table"):
            return
        self.export_status.setVisible(bool(self.export_status.text()))
        active, present = self.busy(), self.workspace is not None
        listening = getattr(self, "play_worker", None) is not None
        if hasattr(self, "play_transport_bar"):
            self.play_transport_bar.setVisible(
                self.pages.currentIndex() != 5 and listening
            )
            self.footer_note.setVisible(
                self.pages.currentIndex() != 5 and not listening
            )
        for widget in (
            self.new_button,
            self.open_button,
            self.recent,
            self.audio_card,
            self.calibration_card,
        ):
            widget.setEnabled(not active)
        for widget in (
            self.control_card,
            self.plan_card,
            self.signal_card,
            self.training_settings,
            self.metadata_card,
        ):
            widget.setEnabled(present and not active)
        self.static_card.setEnabled(present and not active)
        self.route_button.setEnabled(
            not active and self.device.currentData() is not None
        )
        preparing = self.queue_controller.preparing is not None
        preparing_here = (
            preparing
            and present
            and self.queue_controller.preparing.workspace.directory
            == self.workspace.directory
        )
        self.record_button.setEnabled(
            present
            and not active
            and not preparing_here
            and self.rig_ready.isChecked()
            and self.capture_table.currentRow() >= 0
        )
        self.rig_ready.setEnabled(present and not active and not preparing_here)
        self.capture_table.setEnabled(True)
        self.capture_table.selection_locked = self.worker is not None
        self.cancel_button.setEnabled(self.worker is not None)
        if hasattr(self, "play_start"):
            play_idle = self.play_worker is None and self.play_loader is None
            self.play_start.setVisible(not listening)
            self.play_stop.setVisible(listening)
            self.play_quick_stop.setEnabled(listening and self.play_stop.isEnabled())
            self.play_start.setEnabled(
                present and self._play_ready and play_idle and self.worker is None
            )
            self.play_open.setEnabled(self.play_loader is None and self.worker is None)
            self.play_refresh.setEnabled(
                self.play_loader is None and self.worker is None
            )
            self.play_model.setEnabled(self.play_loader is None and self.worker is None)
        ready = present and self.workspace.training_ready()
        self.next_button.setEnabled(
            present and not active and (self.pages.currentIndex() != 2 or ready)
        )
        self.train_button.setEnabled(ready and not active and not preparing)
        self.train_button.setText(
            "Preparing training…" if preparing else "Train · add to queue"
        )
        entry = (
            self.queue_controller.store.get(self.selected_job_id)
            if self.selected_job_id
            else None
        )
        state = entry["state"] if entry else None
        running = bool(entry and entry["id"] == self.queue_controller.active_id)
        self.queue_start_button.setEnabled(
            any(e["state"] == "queued" for e in self.queue_controller.store.entries())
        )
        self.pause_button.setEnabled(
            state == "queued" or running and state not in ("pausing", "stopping")
        )
        self.stop_button.setEnabled(
            state == "queued" or running and state != "stopping"
        )
        resumable = state in FINISHED - {"completed", "needs_captures"} and not running
        self.resume_button.setEnabled(resumable)
        self.retrain_button.setEnabled(
            state is not None and state != "queued" and not active and not preparing
        )
        self.resume_batch_size.setEnabled(resumable)
        self.resume_device.setEnabled(resumable)
        self.resume_options_widget.setVisible(resumable)
        self.remove_job_button.setEnabled(bool(entry) and state not in ACTIVE)
        self.job_up.setEnabled(state == "queued")
        self.job_down.setEnabled(state == "queued")
        self.capture_round_button.setEnabled(state == "needs_captures" and not active)
        self.capture_round_button.setVisible(state == "needs_captures")
        self.models_button.setEnabled(bool(entry))
        spec = read_json(self.job / "spec.json", {}) if self.job else {}
        extension = training.intermediate_export_extension(spec) if spec else ".nam"
        self.export_best_button.setText(
            "Export best PNAM" if extension == ".pnam" else "Export best NAM"
        )
        request = read_json(self.job / "export-request.json", {}) if self.job else {}
        result = read_json(self.job / "export-result.json", {}) if self.job else {}
        self.export_best_button.setEnabled(
            running
            and training.job_progress(self.job).get("best_esr") is not None
            and (not request or request.get("id") == result.get("id"))
        )
        self.custom_button.setEnabled(
            present and not active and self.signal_mode.currentData() == "custom"
        )
        if hasattr(self, "gpu_setup_button"):
            self.gpu_setup_button.setEnabled(
                not active and not self.queue_controller.active_id
            )
        if present:
            enabled = self.mode.currentData() == "active_learning" and not active
            self.rounds.setEnabled(enabled)
            self.additions.setEnabled(enabled)

    def closeEvent(self, event):
        self.queue_controller.enabled = False
        if getattr(self, "play_worker", None):
            self.stop_playback()
        if (
            self.queue_controller.active_id
            or self.worker
            or self.queue_controller.preparing
            or getattr(self, "play_worker", None)
            or getattr(self, "play_loader", None)
        ):
            self._close_after_pause = True
            if self.queue_controller.active_id:
                self.queue_controller.pause(self.queue_controller.active_id)
            if self.worker:
                self.cancel_capture()
            event.ignore()
            return
        if self.workspace:
            self.workspace.save()
        self.close_audio()
        event.accept()

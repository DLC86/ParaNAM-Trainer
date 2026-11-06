"""Visible recovery notices and explicit choices for ambiguous damaged state."""

from pathlib import Path

from PySide6.QtCore import QTimer, QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import recovery


class RecoveryActions:
    def build_recovery_notice(self, layout):
        self.recovery_notice = QWidget()
        row = QHBoxLayout(self.recovery_notice)
        row.setContentsMargins(0, 0, 0, 0)
        self.recovery_text = QLabel()
        self.recovery_text.setWordWrap(True)
        row.addWidget(self.recovery_text, 1)
        review = QPushButton("Review recovery…")
        review.clicked.connect(self.show_recovery_report)
        row.addWidget(review)
        self.recovery_notice.hide()
        layout.addWidget(self.recovery_notice)
        self._recovery_dialog = None
        self._recovery_auto_shown = False

    def refresh_recovery_notice(self):
        if not hasattr(self, "recovery_notice"):
            return
        records = recovery.issues()
        self.recovery_notice.setVisible(bool(records))
        if not records:
            return
        blocking = sum(record["blocking"] for record in records)
        self.recovery_text.setText(
            f"Damaged state detected: {len(records)} file(s), {blocking} need a recovery choice. "
            "Recordings and checkpoints have been kept."
        )
        if not self._recovery_auto_shown:
            self._recovery_auto_shown = True
            QTimer.singleShot(0, self.show_recovery_report)

    def show_recovery_report(self):
        if self._recovery_dialog is not None:
            self._recovery_dialog.show()
            self._recovery_dialog.raise_()
            return
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle("State recovery")
        dialog.resize(840, 500)
        layout = QVBoxLayout(dialog)
        description = QLabel(
            "The app found damaged JSON state. Safe automatic recoveries are listed below. "
            "For unresolved files, restore a valid JSON backup or rebuild the queue index. "
            "Recovery does not start training or delete recordings/checkpoints."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        listing = QListWidget()
        layout.addWidget(listing)
        details = QPlainTextEdit()
        details.setReadOnly(True)
        layout.addWidget(details, 1)
        row = QHBoxLayout()
        restore = QPushButton("Restore JSON backup…")
        retry = QPushButton("Retry recovery")
        rebuild = QPushButton("Rebuild queue…")
        folder = QPushButton("Open folder")
        close = QPushButton("Close")
        for widget in (restore, retry, rebuild, folder, close):
            row.addWidget(widget)
        layout.addLayout(row)

        def refresh():
            records = recovery.issues()
            listing.clear()
            for record in records:
                listing.addItem(f"{record['action']} · {record['path']}")
            listing.setCurrentRow(0 if records else -1)

        def selected():
            records = recovery.issues()
            index = listing.currentRow()
            return records[index] if 0 <= index < len(records) else None

        def show_details():
            record = selected()
            details.setPlainText(
                f"{record['detail']}\n\nOriginal saved as:\n{record.get('preserved') or 'Original file kept in place'}"
                if record
                else "No recovery events."
            )
            rebuild.setEnabled(
                bool(
                    record
                    and Path(record["path"]) == self.queue_controller.store.path
                    and record["blocking"]
                )
            )
            for widget in (restore, retry, folder):
                widget.setEnabled(record is not None)

        def apply(action):
            record = selected()
            if record is None:
                return
            try:
                action(Path(record["path"]))
                self.queue_controller.reconcile_interrupted()
                self._refresh_recent()
                self.refresh_training()
                self.refresh_recovery_notice()
                refresh()
            except (OSError, ValueError) as error:
                QMessageBox.warning(dialog, "Recovery", str(error))

        def restore_file(path):
            active_id = self.queue_controller.active_id
            active = self.queue_controller.store.get(active_id) if active_id else None
            if active_id and (
                path == self.queue_controller.store.path
                or (
                    active
                    and path.resolve().is_relative_to(Path(active["job"]).resolve())
                )
            ):
                raise ValueError(
                    "Stop the active worker before restoring its state or queue index."
                )
            filename, _ = self.choose_open_file(
                "Choose a valid JSON backup", "JSON state (*.json *.bak);;All files (*)"
            )
            if filename:
                recovery.restore_backup(path, filename)

        restore.clicked.connect(lambda: apply(restore_file))
        retry.clicked.connect(lambda: apply(recovery.retry))
        rebuild.clicked.connect(lambda: apply(self.rebuild_recovered_queue))
        folder.clicked.connect(
            lambda: (
                QDesktopServices.openUrl(
                    QUrl.fromLocalFile(str(Path(selected()["path"]).parent))
                )
                if selected()
                else None
            )
        )
        close.clicked.connect(dialog.close)
        listing.currentRowChanged.connect(show_details)
        dialog.finished.connect(lambda: setattr(self, "_recovery_dialog", None))
        refresh()
        show_details()
        self._recovery_dialog = dialog
        dialog.show()  # Modeless: the application remains usable.

    def rebuild_recovered_queue(self, path):
        from .workspace import Workspace
        from .training_queue import FINISHED

        if self.queue_controller.active_id or self.queue_controller.preparing:
            raise ValueError("Stop active queue work before rebuilding its index.")
        if path != self.queue_controller.store.path:
            raise ValueError("Only the queue index can be rebuilt here.")
        if not recovery.is_blocked(path):
            raise ValueError(
                "The queue index is readable and does not need rebuilding."
            )
        recent = self.settings.value("recent", []) or []
        recent = [recent] if isinstance(recent, str) else recent
        projects = []
        for directory in dict.fromkeys(recent):
            try:
                projects.append(Workspace(Path(directory)))
            except (OSError, ValueError):
                continue
        if not projects:
            raise ValueError(
                "No readable recent projects were found. Restore a queue JSON backup or open the project folders first."
            )
        if (
            QMessageBox.question(
                self,
                "Rebuild queue index",
                f"Search {len(projects)} recent project(s) for training instances? "
                "The damaged index is preserved. Recovered unfinished jobs will be held as Interrupted "
                "so you can review their order and resume them. No training files are deleted.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.queue_controller.enabled = False
        recovery.restore_document(
            path,
            {"version": 1, "entries": []},
            "Rebuilt queue",
            "Rebuilt from recent projects; unfinished jobs held for review.",
        )
        for workspace in projects:
            self.queue_controller.store.register_legacy(workspace)

        def hold(entries):
            for entry in entries:
                if entry["state"] not in FINISHED:
                    entry["state"] = "interrupted"

        self.queue_controller.store._edit(hold)

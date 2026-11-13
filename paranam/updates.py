"""Nonblocking GitHub release checks and verified, explicitly requested downloads."""

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import ssl
import tempfile
import threading
import urllib.parse
import urllib.request

from PySide6.QtCore import QThread, QTimer, QUrl, Qt, QStandardPaths, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QProgressBar,
    QTextBrowser,
    QSizePolicy,
)

from . import __version__

REPOSITORY = "DLC86/ParaNAM-Trainer"
API_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
RELEASES_URL = f"https://github.com/{REPOSITORY}/releases/latest"


def update_download_root():
    """Folder reserved for updater downloads, not user projects or captures."""
    return Path(
        QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppLocalDataLocation
        )
    ) / "updates"


def cleanup_staged_installers(root, preserve=None):
    """Remove *all* obsolete updater download folders, including old versions.

    Only direct children with tempfile's installer-<random> naming convention
    are eligible. Never traverse links; preserve an unfinished update. Locked
    files are retried automatically on the next application launch.
    """
    root = Path(root)
    protected = Path(preserve).absolute() if preserve is not None else None
    try:
        children = list(root.iterdir())
    except OSError:
        return 0
    removed = 0
    for child in children:
        if not re.fullmatch(r"installer-[A-Za-z0-9_]{8,}", child.name):
            continue
        if protected is not None and child.absolute() == protected:
            continue
        try:
            if child.is_symlink() or not child.is_dir():
                continue
            shutil.rmtree(child)
            removed += 1
        except OSError:
            # In-use installer, antivirus lock or filesystem permissions.
            # Keep the application usable and retry at next startup.
            continue
    return removed


def version_number(value):
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)(?:-preview)?", value)
    if not match:
        raise ValueError("Unrecognized ParaNAM version")
    return tuple(map(int, match.groups()))


def asset_suffix(system=None, machine=None, executable=None):
    system, machine = (
        system or platform.system(),
        (machine or platform.machine()).lower(),
    )
    if system == "Windows" and machine in ("amd64", "x86_64"):
        return "Windows-x64-Setup.exe"
    if system == "Darwin":
        return (
            "macOS-arm64.dmg" if machine in ("arm64", "aarch64") else "macOS-x86_64.dmg"
        )
    if system == "Linux" and machine in ("amd64", "x86_64"):
        import sys

        return (
            "Linux-amd64.deb"
            if Path(executable or sys.executable).parent == Path("/opt/paranam-trainer")
            else "Linux-x86_64.tar.gz"
        )
    return None


def release_info(document, suffix=None):
    if document.get("draft") or document.get("prerelease"):
        raise ValueError("Expected a published stable release")
    tag = document["tag_name"]
    version = ".".join(map(str, version_number(tag)))
    url = f"https://github.com/{REPOSITORY}/releases/tag/{urllib.parse.quote(tag, safe='')}"
    if document.get("html_url") != url:
        raise ValueError("Release repository mismatch")
    suffix = suffix or asset_suffix()
    name = f"ParaNAM-Trainer-{version}-{suffix}"
    asset = next(
        (
            a
            for a in document.get("assets", [])
            if a.get("name") == name and a.get("state") == "uploaded"
        ),
        None,
    )
    if asset:
        expected = f"https://github.com/{REPOSITORY}/releases/download/{urllib.parse.quote(tag, safe='')}/{urllib.parse.quote(name, safe='')}"
        if asset.get("browser_download_url") != expected:
            raise ValueError("Installer download URL mismatch")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", asset.get("digest", "")):
            asset = None  # A release page remains usable without a verifiable package.
        elif not 0 < asset.get("size", 0) < 2**31:
            raise ValueError("Invalid installer size")
    return dict(
        tag=tag,
        version=version,
        url=url,
        notes=str(document.get("body") or "")[:12000],
        asset=asset,
    )


class GithubRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        parsed = urllib.parse.urlsplit(new_url)
        if parsed.scheme != "https" or not (
            parsed.hostname == "github.com"
            or (parsed.hostname or "").endswith(".githubusercontent.com")
        ):
            raise ValueError("Unexpected installer redirect")
        return super().redirect_request(request, fp, code, message, headers, new_url)


def open_url(url):
    import certifi

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": f"ParaNAM-Trainer/{__version__}",
            "Accept": (
                "application/vnd.github+json"
                if url == API_URL
                else "application/octet-stream"
            ),
        },
    )
    tls = urllib.request.HTTPSHandler(
        context=ssl.create_default_context(cafile=certifi.where())
    )
    return urllib.request.build_opener(GithubRedirects(), tls).open(request, timeout=8)


class UpdateWorker(QThread):
    checked = Signal(dict)
    downloaded = Signal(str)
    failed = Signal(str)
    progress = Signal(int)

    def __init__(self, info=None, directory=None, parent=None):
        super().__init__(parent)
        self.info, self.directory = info, directory
        self.cancelled = threading.Event()

    def run(self):
        temporary = None
        try:
            if self.info is None:
                with open_url(API_URL) as response:
                    payload = response.read(2_000_001)
                if len(payload) > 2_000_000:
                    raise ValueError("Release metadata is too large")
                info = release_info(json.loads(payload))
                if not self.cancelled.is_set():
                    self.checked.emit(info)
                return
            asset = self.info["asset"]
            root = Path(self.directory)
            root.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="installer-", dir=root))
            destination = directory / asset["name"]
            temporary = destination.with_suffix(destination.suffix + ".part")
            digest, count = hashlib.sha256(), 0
            with (
                open_url(asset["browser_download_url"]) as response,
                temporary.open("wb") as output,
            ):
                while not self.cancelled.is_set():
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > asset["size"]:
                        raise ValueError("Installer exceeds its published size")
                    digest.update(chunk)
                    output.write(chunk)
                    self.progress.emit(int(100 * count / asset["size"]))
                output.flush()
                os.fsync(output.fileno())
            if self.cancelled.is_set():
                return
            if (
                count != asset["size"]
                or "sha256:" + digest.hexdigest() != asset["digest"]
            ):
                raise ValueError(
                    "Installer integrity check failed; nothing was installed"
                )
            os.replace(temporary, destination)
            self.downloaded.emit(str(destination))
        except Exception as error:
            if not self.cancelled.is_set():
                self.failed.emit(str(error))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


class UpdateActions:
    def setup_updates(self):
        self.update_info = None
        self.update_check_worker = self.update_download_worker = None
        self._pending_update = None
        self._update_dialog = None
        self._update_blink_on = False
        self.update_led.hide()
        self.update_led.clicked.connect(self.show_update_notice)
        self.update_blink = QTimer(self)
        self.update_blink.setInterval(650)
        self.update_blink.timeout.connect(self._blink_update_led)
        known = self.settings.value("known_update", "")
        if known:
            try:
                cached = json.loads(known)
                tag = cached["tag"]
                version = ".".join(map(str, version_number(tag)))
                if self.settings.value("dismissed_update_tag", "") == tag:
                    self._update_checked(
                        dict(
                            tag=tag,
                            version=version,
                            url=f"https://github.com/{REPOSITORY}/releases/tag/{urllib.parse.quote(tag, safe='')}",
                            notes="Connect to GitHub to check the latest installer.",
                            asset=None,
                        )
                    )
            except (KeyError, ValueError, TypeError):
                pass
        result_path = self.settings.value("update_result_path", "")
        if result_path and Path(result_path).is_file():
            try:
                result = json.loads(Path(result_path).read_text(encoding="utf-8-sig"))
                if not isinstance(result, dict):
                    raise ValueError("Invalid updater result")
                if not result.get("ok"):
                    error = str(result.get("error", "Unknown installer error"))[:1000]
                    QTimer.singleShot(
                        1500,
                        lambda: self.notify_error(
                            "Update installation failed: " + error
                        ),
                    )
                self.settings.remove("update_result_path")
            except (OSError, ValueError):
                pass
        # The updater has already written its result before restarting the app.
        # If its result is still pending, keep that one directory untouched.
        pending = self.settings.value("update_result_path", "")
        preserve = Path(pending).parent if pending else None
        cleanup_staged_installers(update_download_root(), preserve=preserve)
        if os.environ.get("PARANAM_DISABLE_UPDATE_CHECK") != "1":
            QTimer.singleShot(1500, self.start_update_check)

    def start_update_check(self):
        if self.update_check_worker is not None:
            return
        worker = UpdateWorker(parent=self)
        self.update_check_worker = worker
        worker.checked.connect(self._update_checked)
        worker.finished.connect(
            lambda: self._update_worker_finished(worker, "update_check_worker")
        )
        worker.start()

    def _update_worker_finished(self, worker, name):
        if getattr(self, name, None) is worker:
            setattr(self, name, None)
        worker.deleteLater()

    def _update_checked(self, info):
        if version_number(info["version"]) <= version_number(__version__):
            if self._update_dialog is not None:
                self._update_dialog.reject()
            self.update_info = None
            self.update_blink.stop()
            self.update_led.hide()
            return
        if self._update_dialog is not None and self.update_info["tag"] != info["tag"]:
            self._update_dialog.reject()
        self.update_info = info
        self.settings.setValue("known_update", json.dumps({"tag": info["tag"]}))
        self.update_led.setToolTip(
            f"ParaNAM Trainer {info['version']} is available — click to update"
        )
        self.update_led.show()
        self.update_blink.start()
        self._blink_update_led()
        if self._update_dialog is not None:
            self._update_notes.setMarkdown(info["notes"])
            self._update_install_button.setEnabled(
                info["asset"] is not None and self.update_download_worker is None
            )
        if self.settings.value("dismissed_update_tag", "") != info["tag"]:
            self.show_update_notice()

    def _blink_update_led(self):
        self._update_blink_on = not self._update_blink_on
        self.update_led.setStyleSheet(
            "color: "
            + ("#ef4444" if self._update_blink_on else "#555555")
            + "; border: none; background: transparent; padding: 0; font-size: 13px; border-radius: 0;"
        )

    def show_update_notice(self):
        if not self.update_info:
            return
        if self._update_dialog is not None:
            self._update_dialog.show()
            self._update_dialog.raise_()
            return
        dialog = QDialog(self)
        self._update_dialog = dialog
        dialog.setWindowTitle("ParaNAM Trainer update")
        dialog.setMinimumWidth(560)
        layout = QVBoxLayout(dialog)
        text = QLabel(
            f"ParaNAM Trainer {self.update_info['version']} is available.\nInstalled version: {__version__}\n\nDownload and install closes the app safely first, pausing any training.\nDismiss hides this notification for this release; the version LED remains active."
        )
        text.setWordWrap(True)
        text.setTextFormat(Qt.TextFormat.PlainText)
        # Extra dialog height belongs to the notes, not the update notice.
        text.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout.addWidget(text)
        notes = QTextBrowser()
        notes.setOpenExternalLinks(True)
        # QTextBrowser uses Qt's GitHub-flavored Markdown renderer: headings,
        # emphasis, lists, links, code blocks and tables remain formatted.
        notes.setMarkdown(self.update_info["notes"])
        notes.setMinimumHeight(160)
        notes.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._update_notes = notes
        layout.addWidget(notes, 1)
        progress = QProgressBar()
        progress.hide()
        layout.addWidget(progress)
        self._update_progress = progress
        self._update_message = text
        row = QHBoxLayout()
        page = QPushButton("Release notes")
        page.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(self.update_info["url"]))
        )
        row.addWidget(page)
        import sys

        install = QPushButton(
            "Download and install"
            if getattr(sys, "frozen", False)
            else "Download installer"
        )
        install.setEnabled(self.update_info["asset"] is not None)
        install.clicked.connect(self.download_update)
        self._update_install_button = install
        row.addWidget(install)
        dismiss = QPushButton("Dismiss")
        dismiss.clicked.connect(dialog.reject)
        row.addWidget(dismiss)
        layout.addLayout(row)
        dialog.finished.connect(self._dismiss_update_notice)
        dialog.show()

    def _dismiss_update_notice(self, _result=0):
        self.settings.setValue("dismissed_update_tag", self.update_info["tag"])
        self.settings.sync()
        if self.update_download_worker:
            self.update_download_worker.cancelled.set()
        self._update_dialog.deleteLater()
        self._update_dialog = None

    def download_update(self):
        if self.update_download_worker is not None:
            return
        worker = UpdateWorker(self.update_info, update_download_root(), self)
        self.update_download_worker = worker
        self._update_install_button.setEnabled(False)
        self._update_progress.show()
        worker.progress.connect(
            lambda value: (
                self._update_progress.setValue(value) if self._update_dialog else None
            )
        )
        worker.downloaded.connect(self._update_downloaded)
        worker.failed.connect(self._update_download_failed)
        worker.finished.connect(
            lambda: self._update_worker_finished(worker, "update_download_worker")
        )
        worker.start()

    def _update_download_failed(self, error):
        if self._update_dialog:
            self._update_message.setText(
                f"Update download failed: {error}\nYour current installation is unchanged."
            )
            self._update_install_button.setEnabled(True)

    def _update_downloaded(self, filename):
        if not self._update_dialog:
            return
        from .update_install import installation_manifest

        try:
            workers = [
                dict(pid=entry["pid"], pid_started=entry.get("pid_started"))
                for entry in self.queue_controller.store.entries()
                if entry.get("pid")
            ]
            manifest = installation_manifest(Path(filename), self.update_info, workers)
            if manifest is None:
                QDesktopServices.openUrl(QUrl.fromLocalFile(filename))
                self._update_dialog.accept()
                return
            self._pending_update = manifest
            self.settings.setValue(
                "update_result_path", str(manifest.with_name("update-result.json"))
            )
            self._update_message.setText(
                "Installer verified. Waiting for training and audio workers to stop safely…"
            )
            self._close_after_pause = True
            self.close()
        except Exception as error:
            self._update_download_failed(str(error))

    def stop_update_workers(self):
        for name in ("update_check_worker", "update_download_worker"):
            worker = getattr(self, name, None)
            if worker is not None and worker.isRunning():
                worker.cancelled.set()
                if not worker.wait(10000):
                    return False
        return True

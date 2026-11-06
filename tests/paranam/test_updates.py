import hashlib
import io
import json
from pathlib import Path

import pytest

from paranam import updates


def published_release(version="1.0.0", payload=b"verified installer"):
    tag = "v" + version + "-preview"
    name = f"ParaNAM-Trainer-{version}-{updates.asset_suffix()}"
    return dict(
        tag_name=tag,
        html_url=f"https://github.com/{updates.REPOSITORY}/releases/tag/{tag}",
        body="Changes",
        draft=False,
        prerelease=False,
        assets=[
            dict(
                name=name,
                state="uploaded",
                size=len(payload),
                digest="sha256:" + hashlib.sha256(payload).hexdigest(),
                browser_download_url=f"https://github.com/{updates.REPOSITORY}/releases/download/{tag}/{name}",
            )
        ],
    )


def test_platform_selection_and_numeric_version_order():
    assert updates.version_number("v0.6.0-preview") == updates.version_number("0.6.0")
    assert updates.version_number("0.5.10") > updates.version_number("0.5.9")
    assert updates.asset_suffix("Windows", "AMD64") == "Windows-x64-Setup.exe"
    assert updates.asset_suffix("Darwin", "arm64") == "macOS-arm64.dmg"
    assert updates.asset_suffix("Darwin", "x86_64") == "macOS-x86_64.dmg"
    assert (
        updates.asset_suffix("Linux", "x86_64", "/opt/paranam-trainer/ParaNAM Trainer")
        == "Linux-amd64.deb"
    )
    assert (
        updates.asset_suffix(
            "Linux", "x86_64", "/home/me/ParaNAM Trainer/ParaNAM Trainer"
        )
        == "Linux-x86_64.tar.gz"
    )
    with pytest.raises(ValueError):
        updates.version_number("1.0.0;run")


def test_metadata_requires_own_published_release_and_verified_asset():
    document = published_release()
    assert updates.release_info(document)["asset"] is not None
    document["assets"][0].pop("digest")
    assert updates.release_info(document)["asset"] is None
    document = published_release()
    document["assets"][0]["browser_download_url"] = "https://example.com/update.exe"
    with pytest.raises(ValueError, match="URL mismatch"):
        updates.release_info(document)
    document = published_release()
    document["prerelease"] = True
    with pytest.raises(ValueError):
        updates.release_info(document)


@pytest.mark.parametrize(
    "damage,cancel", [(False, False), (True, False), (False, True)]
)
def test_download_checks_hash_size_and_cancellation(
    tmp_path, monkeypatch, qapp, damage, cancel
):
    payload = b"verified installer"
    info = updates.release_info(published_release(payload=payload))
    worker = updates.UpdateWorker(info, tmp_path)
    files, failures = [], []
    worker.downloaded.connect(files.append)
    worker.failed.connect(failures.append)
    monkeypatch.setattr(
        updates,
        "open_url",
        lambda _: io.BytesIO(b"X" * len(payload) if damage else payload),
    )
    if cancel:
        worker.cancelled.set()
    worker.run()
    assert not list(tmp_path.rglob("*.part"))
    if damage:
        assert failures and not files
    elif cancel:
        assert not files and not failures
    else:
        assert Path(files[0]).read_bytes() == payload and not failures


def test_check_failure_is_nonfatal_and_emits_no_update(monkeypatch, qapp):
    worker = updates.UpdateWorker()
    available, failures = [], []
    worker.checked.connect(available.append)
    worker.failed.connect(failures.append)
    monkeypatch.setattr(
        updates, "open_url", lambda _: (_ for _ in ()).throw(OSError("Offline"))
    )
    worker.run()
    assert not available and failures == ["Offline"]


def test_dismiss_persists_for_that_release_but_led_and_new_release_remain(window, qapp):
    info = updates.release_info(published_release())
    window._update_checked(info)
    assert window._update_dialog is not None and window.update_blink.isActive()
    style = window.update_led.styleSheet()
    window._blink_update_led()
    assert window.update_led.styleSheet() != style
    window._update_dialog.reject()
    assert window.settings.value("dismissed_update_tag") == info["tag"]
    window._update_checked(info)
    assert window._update_dialog is None and not window.update_led.isHidden()
    window.show_update_notice()  # LED reopens the notification after Dismiss.
    assert window._update_dialog is not None
    window._update_dialog.reject()
    window._update_checked(updates.release_info(published_release("1.1.0")))
    assert window._update_dialog is not None
    window._update_dialog.reject()


def test_download_installs_only_after_safe_app_shutdown(window, monkeypatch, tmp_path):
    from paranam import update_install

    window._update_checked(updates.release_info(published_release()))
    manifest = tmp_path / "install.json"
    monkeypatch.setattr(update_install, "installation_manifest", lambda *_: manifest)
    closes = []
    monkeypatch.setattr(window, "close", lambda: closes.append(True))
    window._update_downloaded(str(tmp_path / "installer.exe"))
    assert closes == [True] and window._close_after_pause
    assert window._pending_update == manifest
    window._update_dialog.reject()


def test_portable_update_replaces_only_app_and_keeps_previous(tmp_path, monkeypatch):
    import tarfile
    import psutil
    from paranam import update_install

    target = tmp_path / "installed" / "ParaNAM Trainer"
    target.mkdir(parents=True)
    (target / "ParaNAM Trainer").write_bytes(b"old application")
    recordings = tmp_path / "project.wav"
    recordings.write_bytes(b"recording")
    asset = tmp_path / "ParaNAM-Trainer-1.0.0-Linux-x86_64.tar.gz"
    with tarfile.open(asset, "w:gz") as archive:
        item = tarfile.TarInfo("ParaNAM Trainer/ParaNAM Trainer")
        item.size = 15
        item.mode = 0o755
        archive.addfile(item, io.BytesIO(b"new application"))
    manifest = tmp_path / "install.json"
    manifest.write_text(
        json.dumps(
            dict(
                asset=str(asset),
                target=str(target),
                executable=str(target / "ParaNAM Trainer"),
                system="Linux",
                pid=1234567,
                pid_started=0,
                size=asset.stat().st_size,
                digest="sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest(),
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda *_: (_ for _ in ()).throw(psutil.NoSuchProcess(1234567)),
    )
    commands = []
    monkeypatch.setattr(
        update_install.subprocess,
        "Popen",
        lambda command, **kwargs: commands.append(command),
    )
    assert update_install.apply_update(manifest) == 0
    assert (target / "ParaNAM Trainer").read_bytes() == b"new application"
    previous = next(target.parent.glob("ParaNAM Trainer.previous-*"))
    assert (previous / "ParaNAM Trainer").read_bytes() == b"old application"
    assert recordings.read_bytes() == b"recording"
    assert commands == [[str(target / "ParaNAM Trainer")]]
    assert json.loads(manifest.with_name("update-result.json").read_text())["ok"]


def test_windows_helper_waits_checks_hash_and_uses_exact_install_path(tmp_path):
    from paranam.update_install import windows_script

    data = dict(
        pid=123,
        asset="C:/User's folder/Setup.exe",
        target="C:/User's folder/ParaNAM Trainer",
        executable="C:/User's folder/ParaNAM Trainer/ParaNAM Trainer.exe",
        digest="sha256:" + "a" * 64,
    )
    script = windows_script(data, tmp_path / "update-result.json")
    assert "Wait-Process -Timeout" in script and "Get-FileHash -LiteralPath" in script
    assert "User''s folder" in script and "/VERYSILENT" in script
    assert "-WindowStyle Hidden" in script and "Start-Process -FilePath" in script

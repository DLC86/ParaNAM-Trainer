"""Install a verified release only after the GUI and training workers have exited."""

import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import uuid


def installation_manifest(asset, info, workers=()):
    if not getattr(sys, "frozen", False):
        return None
    import psutil

    executable = Path(sys.executable).resolve()
    system = platform.system()
    if system == "Windows":
        target = executable.parent
    elif system == "Darwin":
        target = executable.parents[2]
        if target.suffix != ".app":
            raise ValueError("Cannot identify the installed application bundle")
        if str(target).startswith("/Volumes/"):
            target = Path("/Applications/ParaNAM Trainer.app")
    else:
        target = executable.parent
    manifest = asset.parent / "install.json"
    # sys.executable belongs to PyInstaller's child Core process on Windows.
    # The installed shortcut and restart target belong to its native launcher.
    restart_executable = executable
    launcher_pid = None
    if system == "Windows" and executable.name == "ParaNAM Trainer Core.exe":
        launcher = executable.with_name("ParaNAM Trainer.exe")
        if launcher.is_file():
            restart_executable = launcher
            try:
                parent = psutil.Process(os.getppid())
                if parent.name().casefold() == "paranam trainer.exe":
                    launcher_pid = parent.pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    data = dict(
        asset=str(asset.resolve()),
        executable=str(restart_executable),
        target=str(target),
        system=system,
        pid=os.getpid(),
        pid_started=psutil.Process().create_time(),
        digest=info["asset"]["digest"],
        size=info["asset"]["size"],
        version=info["version"],
        workers=list(workers),
        launcher_pid=launcher_pid,
    )
    with manifest.open("w", encoding="utf-8") as output:
        json.dump(data, output)
        output.flush()
        os.fsync(output.fileno())
    return manifest


def ps_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def windows_script(data, result):
    """Use an external helper so no old frozen Python library locks the install."""
    expected = data["digest"].removeprefix("sha256:")
    # This update originates from a user-confirmed notification: keep the
    # installer GUI visible so users can review and follow its progress.
    installer_args = [
        "/NORESTART",
        "/CLOSEAPPLICATIONS",
        f'/DIR="{data["target"]}"',
        f'/LOG="{result.with_name("installer.log")}"',
    ]
    arguments = ", ".join(ps_literal(arg) for arg in installer_args)
    # The native launcher holds the installed executable open until GUI exit;
    # wait for it before an installer overwrites the application directory.
    launcher_wait = (
        f"    $launcher = Get-Process -Id {int(data['launcher_pid'])} -ErrorAction SilentlyContinue\n"
        "    if ($launcher) { $launcher | Wait-Process -Timeout 60 }"
        if data.get("launcher_pid") else ""
    )
    worker_waits = "\n".join(
        f"    $workerProcess = Get-Process -Id {int(worker['pid'])} -ErrorAction SilentlyContinue\n"
        "    if ($workerProcess) { $workerProcess | Wait-Process -Timeout 1800 }"
        for worker in data.get("workers", [])
    )
    return f"""$ErrorActionPreference = 'Stop'
$result = @{{ok=$false; error=''}}
try {{
    $parentProcess = Get-Process -Id {int(data['pid'])} -ErrorAction SilentlyContinue
    if ($parentProcess) {{ $parentProcess | Wait-Process -Timeout 1800 }}
{launcher_wait}
{worker_waits}
    if ((Get-FileHash -LiteralPath {ps_literal(data['asset'])} -Algorithm SHA256).Hash.ToLowerInvariant() -ne {ps_literal(expected)}) {{ throw 'Installer integrity check failed' }}
    $installer = Start-Process -FilePath {ps_literal(data['asset'])} -ArgumentList @({arguments}) -Wait -PassThru
    if ($installer.ExitCode -ne 0) {{ throw ('Installer exited with code ' + $installer.ExitCode) }}
    $result.ok = $true
}} catch {{ $result.error = $_.Exception.Message }}
$result | ConvertTo-Json | Set-Content -LiteralPath {ps_literal(result)} -Encoding UTF8
Start-Process -FilePath {ps_literal(data['executable'])}
"""


def launch_pending_update(manifest):
    manifest = Path(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data["system"] == "Windows":
        script = manifest.with_name("install-update.ps1")
        script.write_text(
            windows_script(data, manifest.with_name("update-result.json")),
            encoding="utf-8-sig",
        )
        powershell = (
            Path(os.environ["SystemRoot"])
            / "System32/WindowsPowerShell/v1.0/powershell.exe"
        )
        subprocess.Popen(
            [
                str(powershell),
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
            ],
            creationflags=subprocess.CREATE_NO_WINDOW,
            close_fds=True,
        )
    else:
        subprocess.Popen(
            [sys.executable, "--apply-update", str(manifest)],
            start_new_session=True,
            close_fds=True,
        )


def replace_bundle(staged, target):
    previous = target.with_name(target.name + ".previous-" + uuid.uuid4().hex[:8])
    if target.exists():
        os.replace(target, previous)
    try:
        os.replace(staged, target)
    except Exception:
        if previous.exists():
            os.replace(previous, target)
        raise


def apply_update(manifest):
    """macOS/Linux helper entry point; never initializes Qt, audio or training."""
    import psutil
    import tarfile

    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    asset, target = Path(data["asset"]).resolve(), Path(data["target"]).resolve()
    result = dict(ok=False)
    restart_command = [data["executable"]]
    try:
        if asset.parent != manifest.parent or not asset.is_file():
            raise ValueError("Installer path mismatch")
        if asset.stat().st_size != data["size"]:
            raise ValueError("Installer size mismatch")
        digest = hashlib.sha256()
        with asset.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if "sha256:" + digest.hexdigest() != data["digest"]:
            raise ValueError("Installer integrity check failed")
        for process in [
            dict(pid=data["pid"], pid_started=data["pid_started"]),
            *data.get("workers", []),
        ]:
            try:
                parent = psutil.Process(process["pid"])
                if (
                    process.get("pid_started") is None
                    or abs(parent.create_time() - process["pid_started"]) < 0.01
                ):
                    parent.wait(timeout=1800)
            except psutil.NoSuchProcess:
                pass
        if data["system"] == "Darwin":
            mount = Path(tempfile.mkdtemp(prefix="mounted-", dir=asset.parent))
            subprocess.run(
                [
                    "hdiutil",
                    "attach",
                    "-nobrowse",
                    "-readonly",
                    "-mountpoint",
                    str(mount),
                    str(asset),
                ],
                check=True,
            )
            try:
                source = mount / "ParaNAM Trainer.app"
                if not source.is_dir() or target.suffix != ".app":
                    raise ValueError("Application bundle missing from installer")
                staged = target.with_name(
                    target.name + ".update-" + uuid.uuid4().hex[:8]
                )
                if os.access(target.parent, os.W_OK) and (
                    not target.exists() or os.access(target, os.W_OK)
                ):
                    subprocess.run(["ditto", str(source), str(staged)], check=True)
                    replace_bundle(staged, target)
                else:
                    previous = target.with_name(
                        target.name + ".previous-" + uuid.uuid4().hex[:8]
                    )
                    command = "set -e; " + shlex.join(
                        ["ditto", str(source), str(staged)]
                    )
                    command += (
                        "; if [ -d "
                        + shlex.quote(str(target))
                        + " ]; then "
                        + shlex.join(["mv", str(target), str(previous)])
                        + "; fi; "
                    )
                    command += (
                        shlex.join(["mv", str(staged), str(target)])
                        + " || { "
                        + shlex.join(["mv", str(previous), str(target)])
                        + "; exit 1; }"
                    )
                    subprocess.run(
                        [
                            "osascript",
                            "-e",
                            "do shell script "
                            + json.dumps(command, ensure_ascii=False)
                            + " with administrator privileges",
                        ],
                        check=True,
                    )
            finally:
                subprocess.run(["hdiutil", "detach", str(mount)], check=False)
            restart_command = ["open", str(target)]
        elif asset.name.endswith(".deb"):
            if target != Path("/opt/paranam-trainer"):
                raise ValueError("Debian installation path mismatch")
            subprocess.run(["pkexec", "dpkg", "--install", str(asset)], check=True)
            restart_command = [str(target / "ParaNAM Trainer")]
        elif asset.name.endswith(".tar.gz"):
            if target.name != "ParaNAM Trainer" or target == Path("/"):
                raise ValueError("Portable installation path mismatch")
            stage = Path(tempfile.mkdtemp(prefix=".paranam-update-", dir=target.parent))
            with tarfile.open(asset, "r:gz") as archive:
                archive.extractall(stage, filter="data")
            staged = stage / "ParaNAM Trainer"
            if not (staged / "ParaNAM Trainer").is_file():
                raise ValueError("Portable executable missing from installer")
            replace_bundle(staged, target)
            stage.rmdir()
            restart_command = [str(target / "ParaNAM Trainer")]
        else:
            raise ValueError("Unsupported installer")
        result["ok"] = True
    except Exception as error:
        result["error"] = str(error)
    manifest.with_name("update-result.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    if result["ok"] or Path(data["executable"]).is_file():
        subprocess.Popen(restart_command, start_new_session=True)
    return 0 if result["ok"] else 1

"""Optional, versioned GPU wheels; the bundled CPU engine always stays available."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
import zipfile


class InstallCancelled(Exception):
    pass


def platform_key():
    return {"win32": "windows-x86_64", "linux": "linux-x86_64"}.get(sys.platform)


def manifest_id(manifest):
    content = {k: v for k, v in manifest.items() if k != "id"}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_url(url):
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    allowed = host in ("files.pythonhosted.org", "pypi.nvidia.com") or host.endswith(
        ".pytorch.org"
    )
    if (
        parsed.scheme != "https"
        or not allowed
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise ValueError(
            "GPU downloads must come from official PyTorch/NVIDIA package sources"
        )


def load_manifest():
    key = platform_key()
    if not key or platform.machine().lower() not in ("amd64", "x86_64"):
        return None
    path = Path(__file__).resolve().parent / "assets" / f"gpu-{key}.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != 1
        or manifest.get("platform") != key
        or manifest.get("python")
        != f"{sys.version_info.major}.{sys.version_info.minor}"
        or manifest.get("id") != manifest_id(manifest)
    ):
        raise ValueError("The GPU component manifest does not match this application")
    if not manifest.get("wheels") or not any(
        w["name"] == "torch" for w in manifest["wheels"]
    ):
        raise ValueError("GPU manifest has no PyTorch engine")
    for wheel in manifest["wheels"]:
        validate_url(wheel["url"])
        if not re.fullmatch(r"[a-f0-9]{64}", wheel["sha256"]):
            raise ValueError("Missing GPU component checksum")
        if Path(wheel["filename"]).name != wheel["filename"] or not wheel[
            "filename"
        ].endswith(".whl"):
            raise ValueError("Invalid GPU package name")
    return manifest


def runtime_root():
    override = os.environ.get("PARANAM_RUNTIME_ROOT")
    if override:
        return Path(override).resolve()
    if sys.platform == "win32":
        return (
            Path(os.environ.get("LOCALAPPDATA", Path.home()))
            / "ParaNAM Trainer"
            / "runtimes"
        )
    return (
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        / "paranam-trainer"
        / "runtimes"
    )


def runtime_path(manifest):
    identifier = manifest["id"]
    if not re.fullmatch(r"[a-f0-9]{64}", identifier):
        raise ValueError("Invalid GPU runtime identifier")
    return runtime_root() / identifier


def ready(manifest=None):
    try:
        manifest = manifest or load_manifest()
    except (OSError, ValueError):
        return False
    if not manifest:
        return False
    target = runtime_path(manifest)
    try:
        marker = json.loads((target / "ready.json").read_text())
        return (
            marker.get("id") == manifest["id"]
            and marker.get("torch") == manifest["torch"]
            and (target / "site-packages" / "torch" / "__init__.py").is_file()
        )
    except (OSError, ValueError):
        return False


def remember_choice():
    root = runtime_root()
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "setup.json", {"completed": True})


def choice_made():
    return (runtime_root() / "setup.json").is_file()


def _write_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


class _OfficialRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(request):
    return urllib.request.build_opener(_OfficialRedirects()).open(request, timeout=30)


def _hash(path, cancelled):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            if cancelled():
                raise InstallCancelled()
            digest.update(chunk)
    return digest.hexdigest()


def _download(wheel, cache, progress, cancelled, completed, total):
    partial = cache / (wheel["sha256"] + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if wheel.get("size") and offset > wheel["size"]:
        partial.unlink()
        offset = 0
    if not wheel.get("size") or offset != wheel["size"]:
        headers = {"User-Agent": "ParaNAM-Trainer-GPU-Setup"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        progress(
            completed + offset,
            total,
            f"Connecting to the official source for {wheel['name']}",
        )
        with _open(urllib.request.Request(wheel["url"], headers=headers)) as response:
            if offset and response.status != 206:
                offset = 0  # Server did not accept resuming; restart safely.
            if offset and not response.headers.get("Content-Range", "").startswith(
                f"bytes {offset}-"
            ):
                raise RuntimeError(
                    "The download server returned an invalid resume range"
                )
            with partial.open("ab" if offset else "wb") as output:
                read = getattr(response, "read1", response.read)
                last_update = 0
                while chunk := read(1024 * 1024):
                    if cancelled():
                        raise InstallCancelled()
                    output.write(chunk)
                    offset += len(chunk)
                    if time.monotonic() - last_update > 0.1:
                        progress(
                            completed + offset, total, f"Downloading {wheel['name']}"
                        )
                        last_update = time.monotonic()
    if _hash(partial, cancelled) != wheel["sha256"]:
        partial.unlink()
        raise RuntimeError(
            f"Checksum verification failed for {wheel['name']}. Retry the download."
        )
    return partial


def _extract(wheel, destination, cancelled):
    root = destination.resolve()
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            if cancelled():
                raise InstallCancelled()
            relative = PurePosixPath(member.filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in member.orig_filename
                or ":" in member.filename
            ):
                raise ValueError("Unsafe path in downloaded GPU package")
            parts = list(relative.parts)
            if (
                parts
                and parts[0].endswith(".data")
                and len(parts) > 2
                and parts[1] in ("purelib", "platlib")
            ):
                parts = parts[2:]
            target = root.joinpath(*parts).resolve()
            if not target.is_relative_to(root) or stat.S_ISLNK(
                member.external_attr >> 16
            ):
                raise ValueError("Unsafe link in downloaded GPU package")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            if os.name != "nt":
                target.chmod(0o755 if member.external_attr >> 16 & 0o111 else 0o644)


def worker_command(*args):
    return [
        sys.executable,
        *([] if getattr(sys, "frozen", False) else ["-m", "paranam"]),
        *map(str, args),
    ]


def probe(path, manifest):
    report = path / ("probe-" + uuid.uuid4().hex + ".json")
    environment = dict(os.environ)
    environment["PARANAM_GPU_PROBE_PATH"] = str(path.resolve())
    environment["PARANAM_GPU_PROBE_ID"] = manifest["id"]
    result = subprocess.run(
        worker_command("--gpu-probe", report),
        env=environment,
        timeout=180,
        creationflags=0x08000000 if os.name == "nt" else 0,
    )
    if not report.exists():
        raise RuntimeError("Could not start the GPU engine verification")
    data = json.loads(report.read_text(encoding="utf-8"))
    report.unlink()
    if (
        result.returncode
        or not data.get("ok")
        or data.get("torch") != manifest["torch"]
        or data.get("cuda") != manifest["cuda"]
    ):
        raise RuntimeError(
            data.get("error", "The GPU engine did not pass verification")
        )
    return data


def install(progress=lambda *args: None, cancelled=lambda: False):
    from filelock import FileLock

    manifest = load_manifest()
    if not manifest:
        raise RuntimeError("This installer has no NVIDIA component for this computer")
    root = runtime_root()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "install.lock"), timeout=0):
        target = runtime_path(manifest)
        if ready(manifest):
            progress(0, 0, "Checking the installed NVIDIA support")
            result = probe(target, manifest)
            remember_choice()
            return result
        total = sum(w.get("size", 0) for w in manifest["wheels"])
        cache = root / "downloads"
        cache.mkdir(exist_ok=True)
        stage = root / ("install-" + manifest["id"])
        stage.mkdir(exist_ok=True)
        state = stage / "installed-wheels.json"
        installed = set(json.loads(state.read_text())) if state.exists() else set()
        remaining = sum(
            w.get("size", 0) for w in manifest["wheels"] if w["sha256"] not in installed
        )
        if remaining and shutil.disk_usage(root).free < remaining * 4:
            raise RuntimeError(
                f"Please free at least {remaining * 4 / 1024**3:.1f} GiB for GPU setup"
            )
        completed = 0
        try:
            for wheel in manifest["wheels"]:
                if wheel["sha256"] in installed:
                    completed += wheel.get("size", 0)
                    continue
                if cancelled():
                    raise InstallCancelled()
                archive = _download(wheel, cache, progress, cancelled, completed, total)
                progress(completed, total, f"Installing {wheel['name']}")
                _extract(archive, stage / "site-packages", cancelled)
                installed.add(wheel["sha256"])
                _write_json(state, sorted(installed))
                completed += wheel.get("size", archive.stat().st_size)
                archive.unlink()  # Installed files remain; avoid keeping gigabytes of duplicate archives.
            progress(total, total, "Verifying the NVIDIA training engine")
            result = probe(stage, manifest)
            _write_json(
                stage / "ready.json",
                {
                    "id": manifest["id"],
                    "torch": result["torch"],
                    "cuda": result["cuda"],
                },
            )
            if target.exists():
                target.rename(root / ("replaced-" + uuid.uuid4().hex))
            stage.rename(target)
            remember_choice()
            return result
        except Exception:
            # Preserve verified packages and partial downloads for Cancel/Retry.
            # The staging directory is never activated as a training runtime.
            raise


_dll_directories = []


def activate(path=None):
    """Run before importing torch, in the isolated training/probe process only."""
    if "torch" in sys.modules:
        raise RuntimeError(
            "GPU support must be selected before the training engine starts"
        )
    manifest = load_manifest()
    if not manifest:
        raise RuntimeError("No GPU component is available")
    if path is None:
        if not ready(manifest):
            raise RuntimeError("Install NVIDIA support from Training settings first")
        path = runtime_path(manifest)
    path = Path(path).resolve()
    if not path.is_relative_to(runtime_root().resolve()):
        raise ValueError("GPU engine must be inside the application's runtime folder")
    site = path / "site-packages"
    if not (site / "torch" / "__init__.py").is_file():
        raise RuntimeError("Incomplete NVIDIA component")
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    if os.name == "nt":
        libraries = [
            site / "torch" / "lib",
            *site.glob("nvidia/*/bin"),
            *site.glob("nvidia/*/lib"),
        ]
        for directory in libraries:
            if directory.is_dir():
                _dll_directories.append(os.add_dll_directory(str(directory)))
        os.environ["PATH"] = (
            os.pathsep.join(str(p) for p in libraries)
            + os.pathsep
            + os.environ.get("PATH", "")
        )
    return manifest, site


def write_probe(report):
    data = {}
    try:
        path = Path(os.environ["PARANAM_GPU_PROBE_PATH"])
        manifest, site = activate(path)
        if manifest["id"] != os.environ["PARANAM_GPU_PROBE_ID"]:
            raise ValueError("GPU verification manifest changed")
        import torch

        if not Path(torch.__file__).resolve().is_relative_to(site.resolve()):
            raise RuntimeError("The optional GPU engine was not loaded")
        available, hardware_error = can_compute()
        data = dict(
            ok=True,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            available=available,
            hardware_error=hardware_error,
            source=str(torch.__file__),
        )
    except Exception as error:
        import traceback

        data = dict(ok=False, error=str(error), traceback=traceback.format_exc())
    Path(report).write_text(json.dumps(data), encoding="utf-8")
    return 0 if data.get("ok") else 1


def can_compute():
    import torch

    try:
        if not torch.cuda.is_available():
            return False, "No compatible NVIDIA device/driver was detected"
        if torch.ones(8, device="cuda").sum().item() != 8:
            return False, "The NVIDIA device did not pass verification"
        return True, ""
    except Exception as error:
        return False, str(error)

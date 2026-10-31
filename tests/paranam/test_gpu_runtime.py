import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from paranam import gpu_runtime as gpu


@pytest.mark.parametrize(
    "url",
    [
        "http://download.pytorch.org/a",
        "https://evil.test/a",
        "https://download.pytorch.org.evil.test/a",
        "https://user:pass@download.pytorch.org/a",
    ],
)
def test_gpu_download_requires_official_https(url):
    with pytest.raises(ValueError):
        gpu.validate_url(url)


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "x/../../escape", "C:/escape", "x\\escape"]
)
def test_gpu_archive_cannot_escape_runtime(tmp_path, name):
    wheel = tmp_path / "unsafe.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        info = zipfile.ZipInfo()
        info.filename = name
        archive.writestr(info, b"bad")
    with pytest.raises(ValueError):
        gpu._extract(wheel, tmp_path / "runtime", lambda: False)
    assert not (tmp_path / "escape").exists()


def test_cancel_retry_install_is_atomic_and_reuses_existing_engine(
    tmp_path, monkeypatch
):
    wheel = tmp_path / "torch.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("torch/__init__.py", "# diagnostic engine")
        archive.writestr("torch-2.dist-info/LICENSE", "license")
    sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    manifest = {
        "torch": "2.10.0+cu128",
        "cuda": "12.8",
        "wheels": [{"name": "torch", "size": wheel.stat().st_size, "sha256": sha}],
    }
    manifest["id"] = gpu.manifest_id(manifest)
    monkeypatch.setenv("PARANAM_RUNTIME_ROOT", str(tmp_path / "managed"))
    monkeypatch.setattr(gpu, "load_manifest", lambda: manifest)
    downloads = []

    def download(*args):
        downloads.append(True)
        copy = tmp_path / "download.whl"
        copy.write_bytes(wheel.read_bytes())
        return copy

    monkeypatch.setattr(gpu, "_download", download)

    def bad_probe(*args):
        raise RuntimeError("interrupted verification")

    monkeypatch.setattr(gpu, "probe", bad_probe)
    with pytest.raises(RuntimeError, match="interrupted"):
        gpu.install()
    assert not gpu.ready()
    result = {"torch": manifest["torch"], "cuda": "12.8", "available": False}
    monkeypatch.setattr(gpu, "probe", lambda *a: result)
    assert gpu.install() == result
    assert gpu.ready() and len(downloads) == 1
    assert gpu.install() == result and len(downloads) == 1
    assert (
        gpu.runtime_path(manifest) / "site-packages/torch-2.dist-info/LICENSE"
    ).is_file()
    assert gpu.choice_made()


def test_cancel_never_marks_gpu_component_ready(tmp_path, monkeypatch):
    manifest = {"id": "a" * 64, "wheels": [{"sha256": "b" * 64, "size": 1}]}
    monkeypatch.setenv("PARANAM_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(gpu, "load_manifest", lambda: manifest)
    with pytest.raises(gpu.InstallCancelled):
        gpu.install(cancelled=lambda: True)
    assert not gpu.ready()


def test_cpu_choice_does_not_start_download(qapp, tmp_path, monkeypatch):
    from paranam.gpu_setup import GPUSetupDialog

    monkeypatch.setenv("PARANAM_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(gpu, "load_manifest", lambda: None)
    dialog = GPUSetupDialog()
    dialog.start_or_finish()
    assert dialog.worker is None
    assert gpu.choice_made()


def test_escape_requests_cancellation_before_closing(qapp, tmp_path, monkeypatch):
    import threading
    from paranam.gpu_setup import GPUSetupDialog

    monkeypatch.setenv("PARANAM_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(gpu, "load_manifest", lambda: None)
    dialog = GPUSetupDialog()

    class ActiveWorker:
        cancel_event = threading.Event()

        def isRunning(self):
            return True

    dialog.worker = ActiveWorker()
    dialog.reject()
    assert dialog.worker.cancel_event.is_set()
    assert not gpu.choice_made()
    dialog.worker = None


def test_download_resume_and_checksum(tmp_path, monkeypatch):
    import io

    content = b"verified official package"
    sha = hashlib.sha256(content).hexdigest()
    wheel = dict(
        name="torch",
        sha256=sha,
        size=len(content),
        url="https://download.pytorch.org/package.whl",
    )
    partial = tmp_path / (sha + ".partial")
    partial.write_bytes(content[:8])

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": f"bytes 8-{len(content)-1}/{len(content)}"}

    def open_request(request):
        assert request.get_header("Range") == "bytes=8-"
        return Response(content[8:])

    monkeypatch.setattr(gpu, "_open", open_request)
    result = gpu._download(
        wheel, tmp_path, lambda *a: None, lambda: False, 0, len(content)
    )
    assert result.read_bytes() == content
    partial.write_bytes(b"x" * len(content))
    with pytest.raises(RuntimeError, match="Checksum"):
        gpu._download(wheel, tmp_path, lambda *a: None, lambda: False, 0, len(content))
    assert not partial.exists()

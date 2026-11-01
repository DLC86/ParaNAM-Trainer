"""Exercise the actual frozen training worker without developer-environment DLLs."""

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def clean_environment():
    environment = dict(os.environ)
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "VIRTUAL_ENV",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
    ):
        environment.pop(key, None)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    if sys.platform == "win32":
        windows = Path(os.environ.get("SystemRoot", "C:/Windows"))
        environment["PATH"] = str(windows / "System32") + os.pathsep + str(windows)
    return environment


def validate_training(executable, build_directory, gpu_runtime=False, device="cpu"):
    import numpy as np
    import torch
    from nam.capture.params import KnobSpec
    from nam.capture.project import atomic_write_json
    from nam.data import np_to_wav, wav_to_np
    from paranam.workspace import Workspace, read_json
    from paranam.training import make_job, job_progress, command

    directory = Path(build_directory).resolve() / (
        "bundle-training-" + uuid.uuid4().hex[:8]
    )
    workspace = Workspace.create(directory, "Installer self-test")
    x = (0.1 * np.sin(np.arange(48000) * 0.03)).astype(np.float32)
    np_to_wav(x, directory / "diagnostic.wav", rate=48000)
    workspace.draft_controls = [KnobSpec("Gain", 0, 10).to_dict()]
    workspace.preferences.update(
        input_mode="custom",
        custom_input="diagnostic.wav",
        n_train=1,
        n_validation=1,
        corners=False,
        epochs=2,
        batch_size=1,
        device=device,
    )
    workspace.generate_plan()
    for entry in workspace.project.entries:
        source, _ = wav_to_np(
            directory / workspace.project.input_for_split(entry.split), info=True
        )
        target = directory / entry.y_path
        target.parent.mkdir(parents=True, exist_ok=True)
        np_to_wav(np.tanh(1.5 * source), target, rate=48000)
        entry.status = "captured"
        entry.delay = 0
    workspace.project.input_level_dbu = 18.0
    workspace.project.output_level_dbu = 12.0
    workspace.save()
    job = make_job(workspace)
    spec = read_json(job / "spec.json")
    spec["model"]["net"]["config"]["hypernet"].update(
        mode="low_rank", rank=2, hidden_sizes=[4]
    )
    spec["learning"]["trainer"].update(limit_train_batches=1, limit_val_batches=1)
    spec["data"]["common"]["require_input_pre_silence"] = None
    atomic_write_json(job / "spec.json", spec)
    # Exercise checkpoint writing AND reloading in two independent frozen processes.
    command(job, "pause")
    paused = subprocess.run(
        [
            str(executable),
            "--train-worker",
            str(job),
            *(["--gpu-runtime"] if gpu_runtime else []),
        ],
        env=clean_environment(),
        timeout=180,
    )
    progress = job_progress(job)
    if paused.returncode != 0 or progress.get("status") != "paused":
        print((job / "worker.log").read_text(encoding="utf-8", errors="replace"))
        raise RuntimeError(f"Frozen training worker could not pause: {progress}")
    assert (job / "resume-0.ckpt").is_file()
    assert (
        progress["completed_epoch"] == 0
    ), "Pause must not wait for an epoch to finish"
    assert make_job(workspace, resume=True) == job
    result = subprocess.run(
        [
            str(executable),
            "--train-worker",
            str(job),
            *(["--gpu-runtime"] if gpu_runtime else []),
        ],
        env=clean_environment(),
        timeout=180,
    )
    progress = job_progress(job)
    if result.returncode != 0 or progress.get("status") != "completed":
        print((job / "worker.log").read_text(encoding="utf-8", errors="replace"))
        raise RuntimeError(f"Frozen training worker failed: {progress}")
    model = read_json(Path(progress["export"]))
    assert model["architecture"] == "HyperA2"
    assert [s["name"] for s in model["config"]["submodels"]] == ["lite", "full"]
    assert model["metadata"]["input_level_dbu"] == 18.0
    assert model["metadata"]["output_level_dbu"] == 12.0
    if gpu_runtime and device == "auto":
        assert str(progress["device"]).startswith("cuda"), progress
    print("Frozen worker completed real HyperA2 Full/Lite training and metadata export")


def validate_gpu(executable, build_directory):
    """Install the optional official engine using the frozen application's own code."""
    environment = clean_environment()
    report = Path(build_directory).resolve() / "gpu-install.json"
    result = subprocess.run(
        [str(executable), "--install-gpu", str(report)], env=environment, timeout=1800
    )
    data = json.loads(report.read_text(encoding="utf-8"))
    assert result.returncode == 0 and data["ok"], data
    assert "site-packages" in data["source"]
    assert data["cuda"] == "12.8"
    validate_training(executable, build_directory, gpu_runtime=True)
    print(
        "Frozen CPU bundle loaded and trained with the external NVIDIA engine",
        flush=True,
    )


def validate_static(executable, build_directory):
    import numpy as np
    from nam.capture.project import atomic_write_json
    from nam.data import np_to_wav
    from paranam.workspace import Workspace, read_json
    from paranam.training import make_job, job_progress, command

    directory = Path(build_directory).resolve() / (
        "bundle-static-" + uuid.uuid4().hex[:8]
    )
    ws = Workspace.create(directory, "Static installer check")
    ws.preferences.update(
        capture_type="static", static_size="full", epochs=2, batch_size=1, device="cpu"
    )
    ws.generate_plan()
    ws.add_static_capture("Lead")
    ws.project.user_metadata = {
        "modeled_by": "Installer check",
        "custom": {"retained": True},
    }
    # Exercise the real spectral training loss, which needs a full audio window.
    x = (0.1 * np.sin(np.arange(20000) * 0.03)).astype(np.float32)
    np_to_wav(x, directory / "diagnostic.wav", rate=48000)
    for entry in ws.project.entries:
        target = directory / entry.y_path
        target.parent.mkdir(exist_ok=True)
        np_to_wav(np.tanh(1.5 * x), target, rate=48000)
        entry.status, entry.delay = "captured", 0
    ws.save()
    job = make_job(ws)
    spec = read_json(job / "spec.json")
    for split in ("train", "validation"):
        for entry in spec["data"][split]:
            entry.update(x_path="diagnostic.wav", ny=8192)
    spec["data"]["common"]["require_input_pre_silence"] = None
    spec["learning"]["trainer"].update(limit_train_batches=1, limit_val_batches=1)
    atomic_write_json(job / "spec.json", spec)
    command(job, "pause")
    for expected in ("paused", "completed"):
        result = subprocess.run(
            [str(executable), "--train-worker", str(job)],
            env=clean_environment(),
            timeout=240,
        )
        progress = job_progress(job)
        if result.returncode or progress.get("status") != expected:
            print((job / "worker.log").read_text(encoding="utf-8", errors="replace"))
            raise RuntimeError(f"Frozen static batch failed: {progress}")
        if expected == "paused":
            assert progress["completed_epoch"] == 0
            assert not (directory / "models").exists()
            assert make_job(ws, resume=True) == job
    assert len(progress["exports"]) == 2
    for path in progress["exports"]:
        model = read_json(Path(path))
        assert model["architecture"] == "WaveNet"
        assert model["metadata"]["custom"] == {"retained": True}
        assert model["metadata"]["modeled_by"] == "Installer check"
    print(
        "Frozen app trained, paused and resumed two independent static A2 models",
        flush=True,
    )


if __name__ == "__main__":
    validate_gpu(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())

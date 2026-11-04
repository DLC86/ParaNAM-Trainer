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


def validate_installer_configuration(executable, build_directory):
    """The windowless Windows executable must stream progress to the installer."""
    environment = clean_environment()
    environment["PARANAM_RUNTIME_ROOT"] = str(
        Path(build_directory).resolve() / ("installer-config-" + uuid.uuid4().hex[:8])
    )
    result = subprocess.run(
        [str(executable), "--configure-installation"],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "PNGPU|1000|" in result.stdout, (result.stdout, result.stderr)
    print("Frozen installer helper streamed completion without opening a setup dialog")


def validate_playback(executable, model):
    report = model.parent / "playback-check.json"
    result = subprocess.run(
        [str(executable), "--playback-check", str(model), str(report)],
        env=clean_environment(),
        timeout=60,
    )
    data = json.loads(report.read_text(encoding="utf-8"))
    assert result.returncode == 0 and data["ok"], data
    assert "full" in data and "lite" in data
    print(
        "Frozen native playback engine rendered Full and Lite:", model.name, flush=True
    )


def validate_queue(executable, workspace, original_spec, build_directory):
    """Exercise the frozen scheduler launching its own frozen training subprocess."""
    import copy
    from paranam.training_queue import TrainingQueue
    from paranam.workspace import read_json
    from nam.capture.project import atomic_write_json

    queue = TrainingQueue(
        Path(build_directory) / ("bundle-queue-" + uuid.uuid4().hex[:8])
    )
    entry = queue.add(workspace)
    job = Path(entry["job"])
    spec = read_json(job / "spec.json")
    data = copy.deepcopy(original_spec["data"])

    def diagnostic_paths(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("x_path", "y_path"):
                    value[key] = str((workspace.directory / item).resolve())
                else:
                    diagnostic_paths(item)
        elif isinstance(value, list):
            for item in value:
                diagnostic_paths(item)

    diagnostic_paths(data)
    spec.update(
        data=data,
        model=copy.deepcopy(original_spec["model"]),
        learning=copy.deepcopy(original_spec["learning"]),
    )
    spec["learning"]["trainer_fit_kwargs"] = {}
    atomic_write_json(job / "spec.json", spec)
    report = queue.directory / "frozen-queue.json"
    result = subprocess.run(
        [str(executable), "--queue-check", str(queue.directory), str(report)],
        env=clean_environment(),
        timeout=180,
    )
    information = json.loads(report.read_text(encoding="utf-8"))
    if result.returncode or not information["ok"]:
        print((job / "worker.log").read_text(encoding="utf-8", errors="replace"))
        raise RuntimeError(f"Frozen queue failed: {information}")
    assert all(Path(path).is_file() for path in information["exports"])
    print(
        "Frozen queue launched and completed its own frozen training worker", flush=True
    )


def validate_training(executable, build_directory, gpu_runtime=False, device="cpu"):
    import numpy as np
    import torch
    from nam.capture.params import KnobSpec
    from nam.capture.project import atomic_write_json
    from nam.data import np_to_wav, wav_to_np
    from paranam.workspace import Workspace, read_json
    from paranam.training import make_job, job_progress, command
    from paranam.recipes import training_preset

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
        # This short diagnostic deliberately tests the Full HyperA2 recipe,
        # independently of the user-facing new-project defaults.
        advanced_parametric=training_preset("HyperA2"),
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
    assert spec["model"]["net"]["config"]["hypernet"]["mode"] == "full"
    assert spec["model"]["net"]["config"]["hypernet"]["selector"] == {}
    spec["model"]["net"]["config"]["hypernet"]["hidden_sizes"] = [4]
    spec["learning"]["trainer"].update(limit_train_batches=1, limit_val_batches=1)
    spec["data"]["common"]["require_input_pre_silence"] = None
    # Diagnostic WAVs are intentionally short; production jobs reserve 0..10s
    # and the final 9s. Still exercise the separate Train ESR evaluation path.
    for split in ("train", "train_evaluation"):
        for entry in spec["data"][split]:
            entry.update(start_seconds=0, stop_seconds=None)
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
    intermediate = directory / "intermediate.pnam"
    atomic_write_json(
        job / "export-request.json", dict(id="frozen-export", path=str(intermediate))
    )
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
    validation = model["metadata"]["training"]
    assert validation["validation_reference"] == "full"
    assert validation["validation_esr"] == validation["validation_esr_full"]
    assert validation["validation_esr_lite"] is not None
    assert validation["optimizer"] == "Adam"
    assert validation["validation_esr"] == progress["best_esr"]
    assert progress["train_esr_full"] is not None
    assert progress["train_esr_lite"] is not None
    assert progress["best_train_esr"] <= progress["train_esr_full"]
    intermediate_result = read_json(job / "export-result.json")
    assert intermediate_result.get("ok"), intermediate_result
    snapshot = read_json(intermediate)
    assert snapshot["architecture"] == "HyperA2"
    assert snapshot["config"]["params"]
    assert [s["name"] for s in snapshot["config"]["submodels"]] == ["lite", "full"]
    assert not intermediate.with_suffix(".nam").exists()
    if gpu_runtime and device == "auto":
        assert str(progress["device"]).startswith("cuda"), progress
    print("Frozen worker completed real HyperA2 Full/Lite training and metadata export")
    validate_playback(executable, Path(progress["export"]))
    if not gpu_runtime:
        validate_queue(executable, workspace, spec, build_directory)


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
        capture_type="static", epochs=2, static_batch_size=1, device="cpu"
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
    for split in ("train", "validation", "train_evaluation"):
        for entry in spec["data"][split]:
            entry.update(
                x_path="diagnostic.wav", ny=8192, start_seconds=0, stop_seconds=None
            )
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
        assert model["architecture"] == "SlimmableContainer"
        assert [
            child["model"]["config"]["layers"][0]["channels"]
            for child in model["config"]["submodels"]
        ] == [3, 8]
        assert model["metadata"]["custom"] == {"retained": True}
        assert model["metadata"]["modeled_by"] == "Installer check"
        validation = model["metadata"]["training"]
        assert validation["validation_reference"] == "full"
        assert validation["validation_esr"] == validation["validation_esr_full"]
        assert validation["validation_esr_lite"] is not None
        assert validation["optimizer"] == "Adam"
        validate_playback(executable, Path(path))
        assert progress["train_esr_full"] is not None
        assert progress["train_esr_lite"] is not None
    print(
        "Frozen app trained, paused and resumed two independent static A2 models",
        flush=True,
    )


if __name__ == "__main__":
    validate_gpu(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())

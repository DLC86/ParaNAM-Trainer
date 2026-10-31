"""Isolated training jobs, durable epoch checkpoints and HyperA2 active learning.

The desktop polls small atomic JSON files. Torch runs in a separate process so
closing/reopening the window never corrupts the UI or a completed checkpoint.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import traceback
import uuid
from pathlib import Path

from nam.capture.project import atomic_write_json
from .workspace import Workspace, read_json

TERMINAL = {"completed", "paused", "stopped", "failed", "needs_captures"}


class TrainingPaused(Exception):
    pass


class TrainingStopped(Exception):
    pass


def current_job(directory):
    pointer = read_json(Path(directory) / "training" / "current.json", {})
    return Path(pointer["directory"]) if pointer.get("directory") else None


def job_progress(job):
    return read_json(Path(job) / "progress.json", {}) if job else {}


def command(job, action):
    if action not in ("pause", "stop"):
        raise ValueError("Unknown training command")
    atomic_write_json(Path(job) / "command.json", {"action": action})


def job_can_resume(workspace, job):
    if not job:
        return False
    spec = read_json(Path(job) / "spec.json", {})
    progress = job_progress(job)
    if spec.get("fingerprint") != workspace.fingerprint():
        return False
    member = progress.get("member", 0)
    return (Path(job) / f"resume-{member}.ckpt").is_file() or member >= spec.get(
        "members", 1
    )


def make_job(workspace, resume=False):
    from filelock import FileLock
    from nam.capture.export import (
        build_data_config,
        build_hyper_a2_model_config,
        build_hyper_learning_config,
    )

    if not workspace.training_ready():
        raise ValueError("Complete every planned capture before training.")
    root = workspace.directory / "training"
    root.mkdir(exist_ok=True)
    with FileLock(str(root / "worker.lock"), timeout=0):
        if resume:
            job = current_job(workspace.directory)
            if not job_can_resume(workspace, job):
                raise ValueError(
                    "The saved checkpoint does not match this project's current captures/settings."
                )
            atomic_write_json(job / "command.json", {})
            return job
        previous = current_job(workspace.directory)
        round_index = 0
        if previous and job_progress(previous).get("status") == "needs_captures":
            round_index = read_json(previous / "spec.json")["round"] + 1
        job = root / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        job.mkdir()
        prefs = workspace.preferences
        data = build_data_config(workspace.project)
        if not data["validation"]:
            raise ValueError(
                "At least one validation capture is required to report a meaningful ESR."
            )
        # Every member and every final export uses the SAME nested Full/Lite architecture.
        model = build_hyper_a2_model_config(workspace.project)
        learning = build_hyper_learning_config(workspace.project)
        learning["export_checkpoints"] = False
        learning["torch_compile"] = {
            "enabled": False
        }  # No compiler/toolkit installation.
        for split in ("train_dataloader", "val_dataloader"):
            learning[split].update(
                batch_size=prefs["batch_size"],
                num_workers=0,
                pin_memory=False,
                drop_last=False,
            )
        learning["trainer"].update(
            max_epochs=prefs["epochs"],
            accelerator=prefs["device"],
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        members = prefs["ensemble_members"] if prefs["mode"] == "active_learning" else 1
        atomic_write_json(
            job / "spec.json",
            {
                "project": str(workspace.directory),
                "fingerprint": workspace.fingerprint(),
                "round": round_index,
                "members": members,
                "data": data,
                "model": model,
                "learning": learning,
                "preferences": prefs,
            },
        )
        atomic_write_json(
            job / "progress.json",
            {
                "status": "starting",
                "epoch": 0,
                "member": 0,
                "members": members,
                "epochs": prefs["epochs"],
                "round": round_index + 1,
                "elapsed": 0,
                "best_esr": None,
                "eta": None,
                "completed_members": [],
            },
        )
        atomic_write_json(root / "current.json", {"directory": str(job)})
        return job


def build_progress_callback(job, spec, member, shared):
    from pytorch_lightning.callbacks import Callback

    class ProgressCallback(Callback):
        def __init__(self):
            self.started = time.monotonic()
            self.base = float(shared.get("elapsed", 0))
            self.epoch_times = list(shared.get("epoch_times", []))
            self.best = shared.get("best_esr")
            self.epoch_started = None
            self.resume_epoch = shared.get("epoch", 0)

        def state_dict(self):
            import random
            import numpy as np
            import torch

            rng = np.random.get_state()
            return {
                "best_esr": self.best,
                "epoch_times": self.epoch_times,
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "numpy_rng": (rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]),
                "cuda_rng": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
            }

        def load_state_dict(self, data):
            import random
            import numpy as np
            import torch

            self.best = data.get("best_esr")
            self.epoch_times = data.get("epoch_times", [])
            if "torch_rng" in data:
                torch.set_rng_state(data["torch_rng"].cpu())
                random.setstate(data["python_rng"])
                rng = data["numpy_rng"]
                np.random.set_state(
                    (rng[0], np.asarray(rng[1], dtype=np.uint32), *rng[2:])
                )
                if (
                    torch.cuda.is_available()
                    and len(data["cuda_rng"]) == torch.cuda.device_count()
                ):
                    torch.cuda.set_rng_state_all(data["cuda_rng"])

        def write(self, **updates):
            shared.update(
                status="running",
                member=member,
                elapsed=self.base + time.monotonic() - self.started,
                best_esr=self.best,
                epoch_times=self.epoch_times,
            )
            shared.update(updates)
            shared["updated_at"] = time.time()
            atomic_write_json(job / "progress.json", shared)

        def action(self):
            return read_json(job / "command.json", {}).get("action")

        def on_train_start(self, trainer, module):
            self.write(
                device=str(module.device), message="Training HyperA2 Full + Lite"
            )

        def on_train_epoch_start(self, trainer, module):
            self.epoch_started = time.monotonic()
            self.write(epoch=trainer.current_epoch + 1, batch_fraction=0)

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if self.action() == "stop":
                self.write(status="stopped", message="Training cancelled")
                raise TrainingStopped()
            # Throttle disk/GUI updates on short CPU batches.
            if batch_idx % 10 == 0:
                batches = trainer.num_training_batches
                fraction = (
                    (batch_idx + 1) / batches
                    if math.isfinite(batches) and batches
                    else 0
                )
                self.write(
                    batch_fraction=fraction,
                    message=(
                        "Pause requested — finishing this epoch safely"
                        if self.action() == "pause"
                        else "Training HyperA2 Full + Lite"
                    ),
                )

        def on_validation_end(self, trainer, module):
            if trainer.sanity_checking:
                return
            metric = trainer.callback_metrics.get("ESR")
            esr = float(metric.detach().cpu()) if metric is not None else None
            if esr is not None and math.isfinite(esr):
                self.best = esr if self.best is None else min(self.best, esr)
            duration = max(
                time.monotonic() - (self.epoch_started or time.monotonic()), 0.001
            )
            self.epoch_times.append(duration)
            remaining = (spec["members"] - member - 1) * spec["preferences"]["epochs"]
            remaining += max(
                0, spec["preferences"]["epochs"] - trainer.current_epoch - 1
            )
            eta = remaining * sum(self.epoch_times) / len(self.epoch_times)
            self.write(
                epoch=trainer.current_epoch + 1,
                completed_epoch=trainer.current_epoch + 1,
                eta=eta,
                batch_fraction=1,
            )
            if self.action() in ("pause", "stop"):
                # Complete Lightning's epoch-end hooks and loop counters before
                # checkpointing. Saving from inside validation leaves a partial epoch
                # whose metric accumulators cannot be reconstructed after a restart.
                trainer.should_stop = True

        def on_fit_end(self, trainer, module):
            temporary = job / f"resume-{member}.pending.ckpt"
            trainer.save_checkpoint(temporary)
            os.replace(temporary, job / f"resume-{member}.ckpt")
            self.write(checkpoint=str(job / f"resume-{member}.ckpt"))
            if self.action() == "pause":
                self.write(
                    status="paused",
                    message="Saved — safe to close the app and resume later",
                )
                raise TrainingPaused()
            if self.action() == "stop":
                raise TrainingStopped()

    return ProgressCallback()


def _propose_captures(workspace, job, spec, shared):
    """Uncertainty sampling from independently trained HyperA2 members.

    Scores real audio predictions in BOTH Full and Lite. Categorical controls are
    sampled as independent states and all planned validation settings are excluded.
    """
    import numpy as np
    import torch
    from nam.capture.planner import (
        _grid_capacity,
        _settings_key,
        sample_unique_settings,
        _planned_from_params,
    )
    from nam.capture.project import CaptureEntryModel
    from nam.data import wav_to_np
    from nam.train.parametric import _ParametricLightningModule, _get_parametric_net

    specs = tuple(c.to_param_spec() for c in workspace.project.control_specs())
    excluded = {_settings_key(e.params, specs) for e in workspace.project.entries}
    available = _grid_capacity(specs, default_step=0.5) - len(excluded)
    count = min(max(128, spec["preferences"]["additions"] * 16), available)
    if count <= 0:
        return []
    candidates = sample_unique_settings(
        specs, count, seed=1831 + spec["round"], default_step=0.5, exclude=excluded
    )
    models = []
    for member in range(spec["members"]):
        model = _ParametricLightningModule.load_from_checkpoint(
            job / f"resume-{member}.ckpt",
            map_location="cpu",
            weights_only=False,
            **_ParametricLightningModule.parse_config(spec["model"]),
        )
        models.append(_get_parametric_net(model).eval())
    x, _ = wav_to_np(workspace.directory / workspace.project.train_input, info=True)
    x = np.asarray(x).squeeze()
    start = min(24000, max(0, len(x) - 4096))
    signal = torch.as_tensor(np.asarray(x[start : start + 4096]), dtype=torch.float32)
    scores = []
    for index, params in enumerate(candidates):
        action = read_json(job / "command.json", {}).get("action")
        if action == "pause":
            raise TrainingPaused()
        if action == "stop":
            raise TrainingStopped()
        raw = torch.tensor(
            [
                (
                    s.enum_names.index(params[s.name])
                    if s.type == "switch"
                    else params[s.name]
                )
                for s in specs
            ],
            dtype=torch.float32,
        )
        with torch.inference_mode():
            uncertainty = 0.0
            for size in ("full", "lite"):
                predictions = []
                for model in models:
                    model.set_size(size)
                    predictions.append(model(signal, raw))
                uncertainty += (
                    torch.stack(predictions).var(dim=0, unbiased=False).mean().item()
                )
            scores.append(uncertainty)
        shared.update(
            status="scoring",
            message="Choosing informative knob/switch settings",
            scoring_fraction=(index + 1) / len(candidates),
        )
        atomic_write_json(job / "progress.json", shared)
    selected = [
        candidates[int(i)]
        for i in np.argsort(scores)[::-1][: spec["preferences"]["additions"]]
    ]
    offset = 1 + max(
        (e.index for e in workspace.project.entries_for_split("train")), default=-1
    )
    planned = _planned_from_params(
        f"al_r{spec['round'] + 1}", "train", specs, selected, index_offset=offset
    )
    entries = [
        CaptureEntryModel(
            index=e.index, split=e.split, params=e.params, y_path=e.y_path
        )
        for e in planned
    ]
    workspace.project.entries.extend(entries)
    workspace.save()
    return entries


def run_job(job):
    """Worker entry point. Never invoked inside the GUI process."""
    import torch
    import pytorch_lightning as pl
    from filelock import FileLock
    from nam.train.parametric import main as train

    job = Path(job).resolve()
    spec = read_json(job / "spec.json")
    workspace = Workspace(Path(spec["project"]))
    shared = job_progress(job)
    elapsed_base = float(shared.get("elapsed", 0))
    clock_start = time.monotonic()
    with FileLock(str(workspace.directory / "training" / "worker.lock"), timeout=0):
        if workspace.fingerprint() != spec["fingerprint"]:
            raise ValueError("Project changed since this training job was prepared")
        os.chdir(workspace.directory)
        torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) - 1)))
        try:
            completed = shared.get("completed_members", [])
            for member in range(spec["members"]):
                if member in completed:
                    continue
                pl.seed_everything(938 + spec["round"] * 101 + member, workers=True)
                output = job / f"member-{member}"
                output.mkdir(exist_ok=True)
                learning = json.loads(json.dumps(spec["learning"]))
                checkpoint = job / f"resume-{member}.ckpt"
                if checkpoint.exists():
                    learning["trainer_fit_kwargs"]["ckpt_path"] = str(checkpoint)
                shared["elapsed"] = elapsed_base + time.monotonic() - clock_start
                progress = build_progress_callback(job, spec, member, shared)
                train(
                    spec["data"],
                    spec["model"],
                    learning,
                    output,
                    no_show=True,
                    make_plots=False,
                    extra_callbacks=[progress],
                )
                completed.append(member)
                shared.update(
                    completed_members=list(completed), member=member + 1, epoch=0
                )
                atomic_write_json(job / "progress.json", shared)
            # Select the member with the lowest checkpoint ESR. All outputs are nested A2.
            quality = []
            for member in completed:
                checkpoint = torch.load(
                    job / f"resume-{member}.ckpt",
                    map_location="cpu",
                    weights_only=False,
                )
                best = [
                    state.get("best_esr")
                    for state in checkpoint.get("callbacks", {}).values()
                    if isinstance(state, dict) and state.get("best_esr") is not None
                ]
                quality.append(
                    (min(float(v) for v in best) if best else float("inf"), member)
                )
            chosen = min(quality)[1]
            export = workspace.directory / "models" / f"round-{spec['round'] + 1}"
            export.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                job / f"member-{chosen}" / "model_parametric.nam",
                export / "ParaNAM_HyperA2.nam",
            )
            shared.update(export=str(export / "ParaNAM_HyperA2.nam"))
            additions = []
            if (
                spec["preferences"]["mode"] == "active_learning"
                and spec["round"] + 1 < spec["preferences"]["rounds"]
            ):
                additions = _propose_captures(workspace, job, spec, shared)
            shared.update(
                status="needs_captures" if additions else "completed",
                eta=0,
                message=(
                    f"{len(additions)} new captures suggested"
                    if additions
                    else "HyperA2 Full + Lite is ready"
                ),
                additions=len(additions),
            )
            shared["elapsed"] = elapsed_base + time.monotonic() - clock_start
            atomic_write_json(job / "progress.json", shared)
        except (TrainingPaused, TrainingStopped) as error:
            shared.update(
                status="paused" if isinstance(error, TrainingPaused) else "stopped",
                eta=None,
            )
            shared["elapsed"] = elapsed_base + time.monotonic() - clock_start
            atomic_write_json(job / "progress.json", shared)
        except Exception as error:
            shared.update(
                status="failed",
                message=str(error),
                traceback=traceback.format_exc(),
                eta=None,
            )
            shared["elapsed"] = elapsed_base + time.monotonic() - clock_start
            atomic_write_json(job / "progress.json", shared)
            raise

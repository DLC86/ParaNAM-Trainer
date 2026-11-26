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

from nam.capture.project import atomic_write_json, replace_file
from nam.models.losses import waveform_loss_settings
from .workspace import Workspace, read_json

TERMINAL = {"completed", "paused", "stopped", "failed", "needs_captures"}


class TrainingPaused(Exception):
    pass


class TrainingStopped(Exception):
    pass


def _is_film(spec):
    return "conditioning" in spec.get("model", {}).get("net", {}).get("config", {})


def _parametric_name(spec):
    return "FiLMA2 Full + Lite" if _is_film(spec) else "HyperA2 Full + Lite"


def current_job(directory):
    pointer = read_json(Path(directory) / "training" / "current.json", {})
    return Path(pointer["directory"]) if pointer.get("directory") else None


def job_progress(job):
    return read_json(Path(job) / "progress.json", {}) if job else {}


def command(job, action):
    if action not in ("pause", "stop"):
        raise ValueError("Unknown training command")
    atomic_write_json(Path(job) / "command.json", {"action": action})


def requested_total_epochs(job, spec):
    """Read a durable live epoch target without changing training state.

    Old instances have no target file. The spec remains the fallback and the
    target is mirrored there by the queue for restart/resume compatibility.
    """
    request = read_json(Path(job) / "epoch-target.json", {})
    target = request.get("epochs") if isinstance(request, dict) else None
    if type(target) is not int or not 1 <= target <= 1000000:
        target = int(spec["preferences"]["epochs"])
    spec["preferences"]["epochs"] = target
    # The progress callback is also used in lightweight diagnostic tests and
    # legacy snapshots without a nested Lightning trainer configuration.
    trainer_config = spec.get("learning", {}).get("trainer")
    if isinstance(trainer_config, dict):
        trainer_config["max_epochs"] = target
    return target


_legacy_checkpoint_cache = {}


def legacy_checkpoint(job, member):
    """Find a completed-epoch Lightning checkpoint in an older failed job."""
    files = sorted((Path(job) / f"member-{member}" / "checkpoints").glob("*.ckpt"))
    try:
        key = tuple(
            (str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files
        )
    except OSError:
        return None
    if key in _legacy_checkpoint_cache:
        return _legacy_checkpoint_cache[key]
    import torch
    import zipfile

    candidates = []
    for path in files:
        try:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    continue
            state = torch.load(path, map_location="cpu", weights_only=False)
            epoch = int(state["epoch"])
            processed = state["loops"]["fit_loop"]["epoch_progress"]["current"][
                "processed"
            ]
            if processed >= epoch + 1 and state.get("optimizer_states"):
                candidates.append((int(state["global_step"]), epoch + 1, path))
        except Exception:
            # A process may have died while writing an old non-atomic checkpoint.
            continue
    result = max(candidates, default=None)
    _legacy_checkpoint_cache[key] = result
    return result


def _matches_legacy_fingerprint(workspace, spec, batch_size=None):
    from .cadences import legacy_cadence_fingerprint

    saved = spec.get("fingerprint")
    if not saved:
        return False
    return saved == workspace.legacy_knob_fingerprint(spec, batch_size) or (
        saved == legacy_cadence_fingerprint(workspace, spec, batch_size)
    )


def job_can_resume(workspace, job):
    if not job:
        return False
    spec = read_json(Path(job) / "spec.json", {})
    from .evaluation import PROTOCOL

    if spec.get("evaluation_protocol") != PROTOCOL:
        return False
    progress = job_progress(job)
    from .recovery import is_blocked

    if (
        progress.get("recovery_required")
        or is_blocked(Path(job) / "progress.json")
        or is_blocked(Path(job) / "spec.json")
    ):
        return False
    saved_options = spec.get("preferences", {}).get("advanced_training")
    safety_keys = (
        "gradient_clipping",
        "gradient_clip_val",
        "check_nonfinite_gradients",
    )
    legacy_safety = not any(key in (saved_options or {}) for key in safety_keys)
    options = workspace.training_options()
    if workspace.project.capture_type == "parametric":
        saved_learning = spec.get("learning", {})
        waveform_loss = waveform_loss_settings(options)
        if waveform_loss_settings(saved_learning) != waveform_loss:
            return False
        if waveform_loss[0] == "mse":
            if saved_learning.get("rms_loss_normalization", True) != options["rms_loss_normalization"]:
                return False
            if options["rms_loss_normalization"] and (
                saved_learning.get("rms_loss_normalization_exponent", 2.0)
                != options["rms_loss_normalization_exponent"]
            ):
                return False
            if (saved_learning.get("synthetic_mute_reference_rms", False)
                    != options["synthetic_mute_reference_rms"]):
                return False
        # Previously SNR was always off when RMS was off. Compare it
        # independently now, including when resuming older job snapshots.
        if (
            bool(spec.get("learning", {}).get("snr_weighted_loss_normalization", False))
            != options["snr_weighted_loss_normalization"]
        ):
            return False
        # Lightweight/legacy jobs without a synthetic target remain resumable
        # when this project has no synthetic Mute captures.  Requiring the
        # recipe unconditionally made ordinary projects (and frozen jobs)
        # impossible to resume after the synthetic-Mute rollout.
        has_synthetic_mutes = any(
            entry.synthetic_mute for entry in workspace.project.entries
        )
        if (has_synthetic_mutes and not spec.get("model", {}).get("loss", {}).get(
                "synthetic_mute"
        )):
            return False
    saved_trainer = spec.get("learning", {}).get("trainer", {})
    if bool(saved_trainer.get("benchmark", True)) != (
        options["cudnn_benchmark"] and not options["deterministic_training"]
    ):
        return False
    if (
        bool(saved_trainer.get("deterministic", False))
        != options["deterministic_training"]
    ):
        return False
    from .recipes import FILM_PRESETS

    saved_conditioning = spec.get("model", {}).get("net", {}).get("config", {}).get("conditioning")
    conditioning = FILM_PRESETS.get(options["hypernetwork_recipe"])
    if not spec.get("static_captures") and conditioning != saved_conditioning:
        return False
    generation_keys = (
        "hypernetwork_mode",
        "hypernetwork_rank",
        "condition_dilated_convolutions",
        "branch_loss_reduction",
    )
    legacy_generation = not any(key in (saved_options or {}) for key in generation_keys)
    old_hypernet = (
        spec.get("model", {}).get("net", {}).get("config", {}).get("hypernet", {})
    )
    old_mode = old_hypernet.get("mode", "low_rank")
    same_reduction = options["branch_loss_reduction"] == (
        "sum"
        if spec.get("model", {}).get("loss", {}).get("training_recipe", "legacy") == "a2"
        else "mean"
    )
    same_hypernet = (
        options["hypernetwork_mode"] == old_mode
        and (
            old_mode != "low_rank"
            or options["hypernetwork_rank"] == old_hypernet.get("rank", 2)
        )
        and options["condition_dilated_convolutions"]
        == (
            "_conv.weight"
            not in old_hypernet.get("selector", {}).get("exclude_suffixes", [])
        )
    )
    same_generation = bool(spec.get("static_captures")) or (
        same_reduction and (conditioning is not None or same_hypernet)
    )
    old_clip = spec.get("learning", {}).get("trainer", {}).get("gradient_clip_val")
    old_check = spec.get("model", {}).get(
        "check_nonfinite_gradients",
        spec.get("model", {}).get("net", {}).get("name") == "HyperA2",
    )
    same_safety = (
        options["gradient_clipping"] == bool(old_clip)
        and (not old_clip or options["gradient_clip_val"] == old_clip)
        and options["check_nonfinite_gradients"] == old_check
    )
    if not spec.get("static_captures"):
        old_hyper_lr = spec.get("model", {}).get("optimizer", {}).get(
            "lr", 0.004
        ) * spec.get("model", {}).get("hypernet_lr_scale", 1.0)
        same_safety = same_safety and math.isclose(
            options["hypernetwork_learning_rate"], old_hyper_lr
        )
    fingerprints = [
        workspace.fingerprint(batch_size=spec.get("preferences", {}).get("batch_size"))
    ]
    if legacy_generation and same_generation:
        fingerprints.append(
            workspace.fingerprint(
                batch_size=spec.get("preferences", {}).get("batch_size"),
                include_hypernetwork_controls=False,
            )
        )
    if legacy_safety and same_safety and same_generation:
        fingerprints.append(
            workspace.fingerprint(
                batch_size=spec.get("preferences", {}).get("batch_size"),
                include_gradient_safety=False,
            )
        )
    if spec.get("fingerprint") not in fingerprints and not _matches_legacy_fingerprint(
        workspace, spec, spec.get("preferences", {}).get("batch_size")
    ):
        # Older jobs did not fingerprint these controls. Resume only if their
        # actual recipe matches the options currently shown in the application.
        options = workspace.training_options()
        model = spec.get("model", {})
        saved_options = spec.get("preferences", {}).get("advanced_training")
        legacy_optimizer = (
            "Adam"
            if model.get("net", {}).get("name") in ("WaveNet", "PackedWaveNet")
            else "AdamW"
        )
        same_recipe = (
            "optimizer" not in (saved_options or {})
            and "class" not in model.get("optimizer", {})
            and (not legacy_safety or same_safety)
            and (not legacy_generation or same_generation)
            and options["optimizer"] == legacy_optimizer
            and model.get("optimizer", {}).get("lr") == options["learning_rate"]
            and model.get("optimizer", {}).get("weight_decay")
            == options["weight_decay"]
            and model.get("lr_scheduler", {}).get("kwargs", {}).get("gamma")
            == options["lr_decay"]
            and model.get("loss", {}).get("mrstft_weight")
            == (options["mrstft_weight"] if options["fit_mrstft"] else None)
            and all(
                item.get("ny", spec.get("data", {}).get("common", {}).get("ny"))
                == options["ny"]
                for item in spec.get("data", {}).get("train", [])
            )
        )
        if not same_recipe or spec.get("fingerprint") != workspace.fingerprint(
            batch_size=spec.get("preferences", {}).get("batch_size"),
            include_advanced=saved_options is not None,
            include_optimizer=False,
            include_gradient_safety=not legacy_safety,
            include_hypernetwork_controls=not legacy_generation,
        ):
            return False
    member = progress.get("member", 0)
    return (
        (Path(job) / f"resume-{member}.ckpt").is_file()
        or member >= spec.get("members", 1)
        or legacy_checkpoint(job, member) is not None
    )


def make_job(workspace, resume=False, job_directory=None):
    from filelock import FileLock
    from nam.capture.export import (
        MUTE_ANCHOR_DEFAULTS,
        build_data_config,
        build_hyper_a2_learning_config,
        build_hyper_a2_model_config,
        mute_anchor_settings,
    )

    if not workspace.training_ready():
        raise ValueError("Complete every planned capture before training.")
    root = workspace.directory / "training"
    root.mkdir(exist_ok=True)
    if resume:
        job_progress(current_job(workspace.directory))
    with FileLock(str(root / "worker.lock"), timeout=0):
        if resume:
            job = current_job(workspace.directory)
            from .evaluation import PROTOCOL

            if (
                job
                and read_json(job / "spec.json").get("evaluation_protocol") != PROTOCOL
            ):
                raise ValueError(
                    "Start a new training run to reserve the initial and final validation audio. "
                    "Older checkpoints already learned those segments; existing captures can be reused."
                )
            if job:
                from .recovery import ensure_resume_checkpoint

                ensure_resume_checkpoint(job, lock_held=True)
            if not job_can_resume(workspace, job):
                raise ValueError(
                    "The saved checkpoint does not match this project's current captures/settings."
                )
            spec = read_json(job / "spec.json")
            spec["model"]["optimizer"]["class"] = workspace.training_options()[
                "optimizer"
            ]
            spec["preferences"]["advanced_training"] = workspace.training_options()
            spec["learning"]["checkpoint_metric"] = "ESR_full"
            member = job_progress(job).get("member", 0)
            target = job / f"resume-{member}.ckpt"
            if not target.exists():
                recovered = legacy_checkpoint(job, member)
                if recovered:
                    step, completed, source = recovered
                    temporary = target.with_suffix(".pending.ckpt")
                    shutil.copy2(source, temporary)
                    replace_file(temporary, target)
                    atomic_write_json(
                        job / f"resume-{member}.json",
                        dict(
                            completed_epoch=completed,
                            restart_epoch=completed + 1,
                            global_step=step,
                        ),
                    )
            # Restarting a complete epoch allows a different physical batch size or
            # device after OOM/device failure, while keeping model/data checks strict.
            spec["learning"]["train_dataloader"][
                "batch_size"
            ] = workspace.training_batch_size()
            spec["learning"]["val_dataloader"]["batch_size"] = 1
            spec["preferences"]["batch_size"] = workspace.training_batch_size()
            spec["preferences"]["device"] = workspace.preferences["device"]
            spec["learning"]["trainer"]["accelerator"] = workspace.preferences["device"]
            spec["fingerprint"] = workspace.fingerprint()
            atomic_write_json(job / "spec.json", spec)
            atomic_write_json(job / "command.json", {})
            return job
        previous = current_job(workspace.directory)
        round_index = 0
        if previous and job_progress(previous).get("status") == "needs_captures":
            round_index = read_json(previous / "spec.json")["round"] + 1
        if job_directory is None:
            job = root / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
            job.mkdir()
        else:
            job = Path(job_directory).resolve()
            if job.exists() and any(p.name != "project" for p in job.iterdir()):
                raise ValueError("The training instance folder is not empty")
            job.mkdir(parents=True, exist_ok=True)
        options = workspace.training_options()
        for key in ("cudnn_benchmark", "deterministic_training"):
            if not isinstance(options[key], bool):
                raise ValueError(f"{key} must be true or false")
        if options["optimizer"] not in ("Adam", "AdamW"):
            raise ValueError("Choose Adam or AdamW as the training optimizer")
        if options.get("checkpoint_method", "validation") not in (
            "validation", "tolerance", "combined"
        ):
            raise ValueError("Choose a listed checkpoint selection method")
        alpha = options.get("checkpoint_alpha", 0.75)
        if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Checkpoint alpha must be between 0 and 1")
        if options.get("rate_preset", "manual") not in (
            "manual",
            "a2_updates",
            "a2_controls",
            "a2_slow",
            "a2_dataset",
            "a2_dataset_sqrt",
        ):
            raise ValueError("Choose a listed learning-rate preset")
        if options.get("hypernetwork_rate_preset", "follow") not in (
            "follow",
            "manual",
            "a2_updates",
            "a2_controls",
            "a2_slow",
            "a2_dataset",
            "a2_dataset_sqrt",
        ):
            raise ValueError("Choose a listed hypernetwork learning-rate preset")
        hyper_decay = options.get("hypernetwork_lr_decay", options["lr_decay"])
        if not math.isfinite(hyper_decay) or not 0 <= hyper_decay <= 1:
            raise ValueError("Hypernetwork LR Decay must be between 0 and 1")
        if not math.isfinite(options["learning_rate"]) or options["learning_rate"] <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if options["gradient_clipping"] and (
            not math.isfinite(options["gradient_clip_val"])
            or options["gradient_clip_val"] <= 0
        ):
            raise ValueError("Maximum gradient norm must be finite and positive")
        prefs = {
            **workspace.preferences,
            "batch_size": workspace.training_batch_size(),
            "advanced_training": options,
        }
        synthetic = workspace.project.capture_type == "parametric"
        data = build_data_config(workspace.project, include_synthetic=synthetic)
        # NAM's pre-silence check protects the bundled signals' fixed capture
        # alignment. A user-supplied input WAV has no such guaranteed pre-roll,
        # so explicitly disable only that check for custom-input projects.
        train_input_name = Path(workspace.project.train_input).name
        custom_train_input = (
            workspace.preferences.get("input_mode") == "custom"
            or train_input_name not in {
                "input_train.wav",
                "input_standard.wav",
                "input_tts_v11.wav",
                "input_trunc.wav",
            }
        )
        if custom_train_input:
            # Older projects can retain a custom train_input path while their
            # saved input_mode preference is missing or stale. Detect the
            # actual project input too, so NAM's pre-roll check is disabled
            # for the custom training signal regardless of project version.
            for split in ("train", "validation"):
                for row in data[split]:
                    if Path(row["x_path"]).name == train_input_name:
                        row["require_input_pre_silence"] = None
        if not data["validation"]:
            raise ValueError(
                "At least one validation capture is required to report a meaningful ESR."
            )
        from .recipes import FILM_PRESETS

        recipe = options["hypernetwork_recipe"]
        conditioning = FILM_PRESETS.get(recipe)
        # Every member and every final export uses the SAME nested Full/Lite architecture.
        is_static = workspace.project.capture_type == "static"
        if is_static:
            from .static_training import model_config

            model = model_config(workspace.project)
        else:
            model = build_hyper_a2_model_config(workspace.project)
            # The only mute objective is ordinary MSE against the
            # synthesized full-length zero target; no anchor or special forwards.
            model["loss"].pop("mute_anchor", None)
            model["loss"]["synthetic_mute"] = True
            if workspace.project.output_gain:
                # Reconcile saved calibration with the current Plan display order.
                curve = dict(workspace.project.output_gain)
                curve["index"] = next(
                    i for i, control in enumerate(workspace.project.control_specs())
                    if control.name == curve["name"]
                )
                model["output_gain"] = curve
            if conditioning is not None:
                model["net"]["name"] = "ParametricSlimmableContainer"
                config = model["net"]["config"]
                config.pop("hypernet")
                config["conditioning"] = conditioning
            else:
                if recipe not in ("HyperA2", "HyperWaveNet"):
                    raise ValueError(
                        "Choose HyperA2 or HyperWaveNet as the training preset"
                    )
                if options["hypernetwork_mode"] not in ("full", "low_rank"):
                    raise ValueError("Hypernetwork weight mode must be full or low_rank")
                rank = options["hypernetwork_rank"]
                if (
                    isinstance(rank, bool)
                    or not isinstance(rank, int)
                    or not 1 <= rank <= 8
                ):
                    raise ValueError("Hypernetwork rank must be an integer from 1 to 8")
                if not isinstance(options["condition_dilated_convolutions"], bool):
                    raise ValueError("Condition dilated convolutions must be a boolean")
                model["net"]["config"]["hypernet"].update(
                    mode=options["hypernetwork_mode"],
                    rank=rank,
                    selector=(
                        {}
                        if options["condition_dilated_convolutions"]
                        else {"exclude_suffixes": ["_conv.weight"]}
                    ),
                )
            if options["branch_loss_reduction"] not in ("sum", "mean"):
                raise ValueError("Full/Lite loss reduction must be sum or mean")
            model["loss"]["training_recipe"] = (
                "a2" if options["branch_loss_reduction"] == "sum" else "legacy"
            )
            if (
                not math.isfinite(options["hypernetwork_learning_rate"])
                or options["hypernetwork_learning_rate"] <= 0
            ):
                raise ValueError(
                    "Hypernetwork learning rate must be finite and positive"
                )
            model["hypernet_lr_scale"] = (
                options["hypernetwork_learning_rate"] / options["learning_rate"]
            )
        model["optimizer"].update(
            **{"class": options["optimizer"]},
            lr=options["learning_rate"],
            weight_decay=options["weight_decay"],
        )
        model["lr_scheduler"]["kwargs"]["gamma"] = options["lr_decay"]
        model["check_nonfinite_gradients"] = options["check_nonfinite_gradients"]
        model["loss"]["mrstft_weight"] = (
            options["mrstft_weight"] if options["fit_mrstft"] else None
        )
        for entry in data["train"]:
            entry["ny"] = options["ny"]
        for entry in data["validation"]:
            entry["ny"] = None
        from .evaluation import PROTOCOL, reserve_validation_audio

        reserve_validation_audio(data)
        if not is_static and (
            options.get("hypernetwork_rate_preset", "follow") != "follow"
        ):
            from .recipes import rate_group_schedule

            schedules = {
                name: rate_group_schedule(
                    workspace.project,
                    data,
                    workspace.directory,
                    prefs["batch_size"],
                    prefs["epochs"],
                    options,
                    hypernetwork=hyper,
                )
                for name, hyper in (("audio", False), ("hypernetwork", True))
            }
            model["lr_scheduler"] = {
                "class": "IndependentGroupLR",
                "kwargs": {
                    name: {
                        key: schedule[key] for key in ("interval", "step_size", "gamma")
                    }
                    for name, schedule in schedules.items()
                },
            }
            prefs["rate_schedule"] = schedules
        elif not is_static and options.get("rate_preset", "manual") in (
            "a2_updates",
            "a2_controls",
            "a2_slow",
            "a2_dataset",
            "a2_dataset_sqrt",
        ):
            from .recipes import a2_update_schedule, rate_cadence_multiplier

            schedule = a2_update_schedule(
                data,
                workspace.directory,
                prefs["batch_size"],
                options["ny"],
                prefs["epochs"],
                options["learning_rate"],
                options["hypernetwork_learning_rate"],
                options["lr_decay"],
                cadence_multiplier=rate_cadence_multiplier(
                    workspace.project, options["rate_preset"]
                ),
                normalization=(
                    options["rate_preset"]
                    if options["rate_preset"] in ("a2_dataset", "a2_dataset_sqrt")
                    else None
                ),
            )
            model["lr_scheduler"] = {
                "class": (
                    "StepLR"
                    if options["rate_preset"] in ("a2_updates", "a2_controls")
                    else "UpdateCadenceLR"
                ),
                "kwargs": {
                    "gamma": schedule["reference_gamma"],
                    "step_size": schedule["decay_steps"],
                },
                "interval": "step",
                "frequency": 1,
            }
            prefs["rate_schedule"] = schedule
        data["allow_variable_validation_lengths"] = True
        learning = build_hyper_a2_learning_config(workspace.project)
        waveform_loss, threshold = waveform_loss_settings(options)
        if waveform_loss == "segment_esr":
            learning.update(waveform_loss=waveform_loss, segment_esr_min_rms_dbfs=threshold)
        rms = options["rms_loss_normalization"]
        if not isinstance(rms, bool):
            raise ValueError("rms_loss_normalization must be a boolean")
        exponent = options["rms_loss_normalization_exponent"]
        if isinstance(exponent, bool) or not isinstance(exponent, (float, int)) or not math.isfinite(exponent) or not 0.01 <= exponent <= 2.0:
            raise ValueError("RMS loss normalization exponent must be between 0.01 and 2")
        learning["rms_loss_normalization_exponent"] = float(exponent)
        learning["rms_loss_normalization"] = rms and waveform_loss == "mse"
        snr_weighted = options["snr_weighted_loss_normalization"]
        if not isinstance(snr_weighted, bool):
            raise ValueError("snr_weighted_loss_normalization must be a boolean")
        learning["snr_weighted_loss_normalization"] = snr_weighted
        if synthetic:
            paths = {
                e.y_path: e for e in workspace.project.entries if e.synthetic_mute
            }
            learning["synthetic_mute_train_indices"] = [
                i for i, row in enumerate(data["train"])
                if row["y_path"] in paths
            ]
            learning["synthetic_mute_validation_indices"] = [
                i for i, row in enumerate(data["validation"])
                if row["y_path"] in paths
            ]
            learning["synthetic_mute_reference_rms"] = bool(
                waveform_loss == "segment_esr" or options["synthetic_mute_reference_rms"]
            )
            if not data["train"] or len(learning["synthetic_mute_train_indices"]) == len(data["train"]):
                raise ValueError("Synthetic mute training needs at least one real training capture")
            if len(learning["synthetic_mute_validation_indices"]) == len(data["validation"]):
                raise ValueError("Validation ESR requires at least one non-mute capture")
        learning["checkpoint_metric"] = "ESR_full"
        learning["separate_size_checkpoints"] = conditioning is None
        learning["export_checkpoints"] = False
        learning["torch_compile"] = {
            "enabled": False
        }  # No compiler/toolkit installation.
        for split in ("train_dataloader", "val_dataloader"):
            learning[split].update(
                batch_size=prefs["batch_size"],
                num_workers=0,
                pin_memory=False,
                drop_last=(split == "train_dataloader"),
            )
        learning["val_dataloader"]["batch_size"] = 1
        if options["gradient_clipping"]:
            learning["trainer"]["gradient_clip_val"] = options["gradient_clip_val"]
            learning["trainer"]["gradient_clip_algorithm"] = "norm"
        else:
            learning["trainer"].pop("gradient_clip_val", None)
        learning["trainer"].update(
            benchmark=options["cudnn_benchmark"]
            and not options["deterministic_training"],
            deterministic=options["deterministic_training"],
            max_epochs=prefs["epochs"],
            accelerator=prefs["device"],
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        members = prefs["ensemble_members"] if prefs["mode"] == "active_learning" else 1
        if is_static:
            members = len(workspace.static_groups())
        atomic_write_json(
            job / "spec.json",
            {
                "project": str(workspace.directory),
                "fingerprint": workspace.fingerprint(),
                "round": round_index,
                "members": members,
                "data": data,
                "evaluation_protocol": PROTOCOL,
                "model": model,
                "learning": learning,
                "preferences": prefs,
                "capture_calibration": workspace.capture_calibration(),
                "static_captures": (
                    list(prefs["static_captures"]) if is_static else None
                ),
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


def annotate_validation_metadata(model, metrics, optimizer, advanced_settings=None):
    if not metrics or metrics.get("full") is None:
        return
    parametric_export = advanced_settings is not None
    summary = dict(
        validation_esr_full=metrics["full"],
        validation_esr_lite=metrics["lite"],
        epoch=metrics["epoch"],
        optimizer=optimizer,
    )
    if not parametric_export:
        summary.update(
            validation_esr=metrics["full"],
            validation_reference="full",
            validation_esr_mean=metrics["mean"],
        )
    summary["train_esr_full"] = metrics.get("train_full")
    summary["train_esr_lite"] = metrics.get("train_lite")
    if advanced_settings is not None:
        summary["advanced_training"] = dict(advanced_settings)
    summary["epoch_full"] = metrics.get("epoch_full", metrics["epoch"])
    summary["epoch_lite"] = metrics.get("epoch_lite", metrics["epoch"])
    metadata = model.setdefault("metadata", {})
    key = (
        "training"
        if isinstance(metadata.get("training", {}), dict)
        else "paranam_training"
    )
    training_metadata = metadata.setdefault(key, {})
    if parametric_export:
        for field in (
            "validation_esr",
            "validation_reference",
            "validation_esr_mean",
        ):
            training_metadata.pop(field, None)
    training_metadata.update(summary)
    if model.get("architecture") == "SlimmableContainer":
        for index, child in enumerate(model["config"]["submodels"]):
            size = "lite" if index == 0 else "full"
            child_metadata = child["model"].setdefault("metadata", {})
            child_key = (
                "training"
                if isinstance(child_metadata.get("training", {}), dict)
                else "paranam_training"
            )
            child_summary = {
                **summary,
                "epoch": summary["epoch_" + size],
            }
            if not parametric_export:
                child_summary.update(
                    validation_esr=metrics[size], validation_reference=size
                )
            child_metadata.setdefault(child_key, {}).update(child_summary)
            if parametric_export:
                for key in (
                    "validation_esr",
                    "validation_reference",
                    "validation_esr_mean",
                ):
                    child_metadata[child_key].pop(key, None)


def intermediate_export_extension(spec):
    return ".nam" if spec.get("static_captures") else ".pnam"


def export_best_model(trainer, spec, member, target, selection=None):
    """Build one independent Full/Lite snapshot at a safe training boundary."""
    import torch
    import tempfile
    from nam.train.parametric import _ParametricLightningModule
    from nam.train.lightning_module import PackedLightningModule, PackedBestCheckpoint
    from nam.models.parametric import export_parametric

    static = bool(spec.get("static_captures"))
    film = _is_film(spec)
    checkpoint_class = PackedLightningModule if static else _ParametricLightningModule
    if selection is not None:
        paths, scores = selection
    elif static:
        packed = next(
            c for c in trainer.callbacks if isinstance(c, PackedBestCheckpoint)
        )
        entries = packed.best
        paths = {
            size: entries[i]["checkpoint_path"]
            for size, i in (("lite", 0), ("full", 1))
        }
        scores = {
            size: entries[i]["best_metric"] for size, i in (("lite", 0), ("full", 1))
        }
    else:
        selector = next(
            (callback for callback in getattr(trainer, "callbacks", [])
             if callable(getattr(callback, "selected_checkpoints", None))),
            None,
        )
        selected = selector.selected_checkpoints() if selector is not None else None
        if film:
            full = selected.get("full") if selected else None
            if full is None:
                callback = next(c for c in trainer.checkpoint_callbacks if c.monitor == "ESR_full")
                full = {"path": callback.best_model_path}
            paths = dict.fromkeys(("full", "lite"), full["path"])
            scores = {}
        elif selected and all(selected.values()):
            paths = {size: selected[size]["path"] for size in ("full", "lite")}
            scores = {size: selected[size]["val"] for size in ("full", "lite")}
        else:
            callbacks = {c.monitor: c for c in trainer.checkpoint_callbacks}
            paths = {
                size: callbacks[f"ESR_{size}"].best_model_path for size in ("full", "lite")
            }
            scores = {
                size: float(callbacks[f"ESR_{size}"].best_model_score)
                for size in ("full", "lite")
            }
    if not all(paths.values()) or not all(
        math.isfinite(value) for value in scores.values()
    ):
        raise ValueError("A validated Full and Lite checkpoint is not available yet")
    # Constructors consume CPU RNG; exporting must not change the run.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        if film:
            checkpoint = torch.load(paths["full"], map_location="cpu", weights_only=False)
            checkpoints = dict.fromkeys(("full", "lite"), checkpoint)
            scores = {size: checkpoint.get(f"validation_esr_{size}")
                      for size in checkpoints}
            if any(value is None or not math.isfinite(value) for value in scores.values()):
                raise ValueError("The shared FiLMA2 checkpoint has no valid Full/Lite scores")
        else:
            checkpoints = {
                size: torch.load(path, map_location="cpu", weights_only=False)
                for size, path in paths.items()
            }
        models = {}
        for size in (("full",) if film else ("full", "lite")):
            model = checkpoint_class.init_from_config(spec["model"])
            model.on_load_checkpoint(checkpoints[size])
            model.load_state_dict(checkpoints[size]["state_dict"])
            models[size] = model.cpu().eval()
        net = models["full"].net
        if static:
            net.import_submodel(0, models["lite"].net.extract_submodel(0))
            net.apply_mask()
            export_net = net
        elif film:
            export_net = net
        else:
            net.branch("lite").load_state_dict(
                models["lite"].net.branch("lite").state_dict()
            )
            export_net = net
        target = Path(target)
        extension = intermediate_export_extension(spec)
        if target.suffix.lower() != extension:
            target = target.with_suffix(extension)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".paranam-export-", dir=target.parent
        ) as directory:
            if static:
                export_net.export(Path(directory), basename="snapshot")
            else:
                export_parametric(
                    export_net, Path(directory), basename="snapshot",
                    output_scale=checkpoints["full"].get("output_scale") if film else None,
                )
            exported = read_json(Path(directory) / ("snapshot" + extension))
        source = read_json(Path(spec["project"]) / "capture_project.json", {})
        if not static and source.get("knobs"):
            # Also upgrade exports from checkpoints trained before display indices existed.
            from nam.capture.export import apply_model_display_indices
            apply_model_display_indices(
                exported, source["knobs"],
                output_gain=spec["model"].get("output_gain") or source.get("output_gain"),
            )
        metadata = {
            **source.get("user_metadata", {}),
            **spec["model"].get("metadata", {}),
        }
        for key in ("input_level_dbu", "output_level_dbu"):
            # MetadataEditor values override capture calibration in the exported
            # model only. Historical dataset gains and global Settings are unchanged.
            if key not in metadata:
                value = spec.get("capture_calibration", source).get(key)
                if value is not None:
                    metadata[key] = value
        if static:
            capture = spec["static_captures"][member]
            metadata.update(capture.get("metadata", {}), name=capture["name"])
        exported.setdefault("metadata", {}).update(metadata)
        if exported.get("architecture") == "SlimmableContainer":
            for child in exported["config"]["submodels"]:
                child["model"].setdefault("metadata", {}).update(metadata)
        epochs = {
            size: int(checkpoints[size]["epoch"]) + 1 for size in ("full", "lite")
        }
        annotate_validation_metadata(
            exported,
            dict(
                full=scores["full"],
                lite=scores["lite"],
                mean=sum(scores.values()) / 2,
                epoch=epochs["full"],
                epoch_full=epochs["full"],
                epoch_lite=epochs["lite"],
                train_full=checkpoints["full"].get("train_esr_full"),
                train_lite=checkpoints["lite"].get("train_esr_lite"),
            ),
            spec["model"]["optimizer"].get("class", "Adam" if static else "AdamW"),
            spec.get("preferences", {}).get("advanced_training", {}) if not static else None,
        )
        training_key = (
            "training"
            if isinstance(exported["metadata"].get("training", {}), dict)
            else "paranam_training"
        )
        exported["metadata"].setdefault(training_key, {})["intermediate"] = True
        atomic_write_json(Path(target), exported)
    return str(target)


def export_saved_best(job, target):
    """Export a paused/stopped job without fitting or altering its checkpoints."""
    import torch

    job = Path(job)
    spec = read_json(job / "spec.json")
    progress = job_progress(job)
    member = min(progress.get("member", 0), max(0, spec.get("members", 1) - 1))
    if spec.get("static_captures"):
        entries = read_json(
            job / f"member-{member}" / "size-checkpoints" / "packed_best.json", {}
        ).get("submodels", [])
        indexed = {entry["submodel_index"]: entry for entry in entries}
        paths = {
            size: indexed.get(i, {}).get("checkpoint_path")
            for size, i in (("full", 1), ("lite", 0))
        }
        scores = {
            size: indexed.get(i, {}).get("best_metric", float("inf"))
            for size, i in (("full", 1), ("lite", 0))
        }
    else:
        checkpoint = job / f"resume-{member}.ckpt"
        if not checkpoint.is_file():
            raise ValueError("No completed validation epoch is available to export yet")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        callbacks = {
            state.get("monitor"): state
            for state in saved.get("callbacks", {}).values()
            if isinstance(state, dict) and state.get("monitor")
        }
        paths = {
            size: callbacks.get(f"ESR_{size}", {}).get("best_model_path")
            for size in ("full", "lite")
        }
        scores = {
            size: callbacks.get(f"ESR_{size}", {}).get("best_model_score")
            for size in ("full", "lite")
        }
        from .checkpoint_selection import saved_selection

        settings = spec.get("preferences", {}).get("advanced_training", {})
        method = settings.get("checkpoint_method", "validation")
        chosen = saved_selection(saved, method, settings.get("checkpoint_alpha", 0.75))
        if _is_film(spec):
            full = chosen.get("full") if chosen else None
            path = full["path"] if full else paths["full"]
            if not path or not Path(path).is_file():
                raise ValueError("A validated shared FiLMA2 checkpoint is not available yet")
            return export_best_model(
                None, spec, member, target,
                selection=(dict.fromkeys(("full", "lite"), path), {}),
            )
        if chosen and all(chosen.values()):
            paths = {size: chosen[size]["path"] for size in ("full", "lite")}
            scores = {size: chosen[size]["val"] for size in ("full", "lite")}
        del saved
    if not all(path and Path(path).is_file() for path in paths.values()) or any(
        value is None or not math.isfinite(float(value)) for value in scores.values()
    ):
        raise ValueError("A validated Full and Lite checkpoint is not available yet")
    scores = {size: float(value) for size, value in scores.items()}
    return export_best_model(None, spec, member, target, selection=(paths, scores))


def build_progress_callback(job, spec, member, shared, evaluation_data=None):
    from pytorch_lightning.callbacks import Callback

    label = (
        f"A2 Full + Lite: {spec['static_captures'][member]['name']}"
        if spec.get("static_captures")
        else _parametric_name(spec)
    )

    class ProgressCallback(Callback):
        def __init__(self):
            self.started = time.monotonic()
            self.base = float(shared.get("elapsed", 0))
            self.epoch_times = list(shared.get("epoch_times", []))
            self.esr_history = (
                list(shared.get("esr_history", []))
                if shared.get("esr_history_member", member) == member
                else []
            )
            self.best = None
            self.best_train = None
            self.train_evaluator = None
            self.best_metrics = None
            self.last_metrics = None
            self.best_lite_metrics = None
            self.epoch_started = None
            self.resume_epoch = shared.get("epoch", 0)
            self.trainer = None
            self.last_log_update = 0
            self.export_request_stamp = None
            self.epoch_target_stamp = object()
            from .batch_timing import BatchTiming

            self.timing = BatchTiming()
            self.batch_started = None
            self.batch_done = 0
            self.validation_started = None

        def batch_estimate(self, trainer, validated=False):
            batches = getattr(trainer, "num_training_batches", None)
            if not batches or not math.isfinite(batches):
                return None
            epochs = spec["preferences"]["epochs"]
            future_epochs = max(0, epochs - trainer.current_epoch - 1)
            future_members = max(0, spec["members"] - member - 1)
            member_counts = shared.get("member_train_batches", {})
            future_batches = sum(
                member_counts.get(str(index), batches) * epochs
                for index in range(member + 1, spec["members"])
            )
            remaining = (
                max(0, batches - self.batch_done)
                + future_epochs * batches
                + future_batches
            )
            validation_passes = (
                future_epochs + future_members * epochs + (0 if validated else 1)
            )
            return self.timing.remaining(remaining, validation_passes)

        def print_progress(self, trainer, fraction, phase):
            from .training_log import epoch_line

            print(
                "\r"
                + epoch_line(
                    trainer.current_epoch + 1,
                    spec["preferences"]["epochs"],
                    label,
                    fraction,
                    phase,
                    self.timing.iterations_per_second,
                ),
                end="",
                flush=True,
            )

        def check_export_request(self):
            request_path = job / "export-request.json"
            try:
                info = request_path.stat()
            except FileNotFoundError:
                return
            stamp = (info.st_mtime_ns, info.st_size)
            if stamp == self.export_request_stamp:
                return
            request = read_json(request_path, {})
            result = read_json(job / "export-result.json", {})
            if (
                not request
                or request.get("id") == result.get("id")
                or self.trainer is None
            ):
                if request and request.get("id") == result.get("id"):
                    self.export_request_stamp = stamp
                return
            # Both checkpoint callbacks finish before the next batch boundary.
            if self.best is None:
                return
            try:
                path = export_best_model(self.trainer, spec, member, request["path"])
                result = dict(id=request["id"], ok=True, path=path)
            except Exception as error:
                result = dict(id=request["id"], ok=False, error=str(error))
            atomic_write_json(job / "export-result.json", result)
            self.export_request_stamp = stamp

        def save_boundary(self, trainer, completed):
            temporary = job / f"resume-{member}.pending.ckpt"
            target = job / f"resume-{member}.ckpt"
            trainer.save_checkpoint(temporary)
            replace_file(temporary, target)
            info = dict(
                completed_epoch=completed,
                restart_epoch=completed + 1,
                global_step=trainer.global_step,
                best_esr=self.best,
            )
            atomic_write_json(job / f"resume-{member}.json", info)
            self.write(checkpoint=str(target), resume_available=True, **info)

        def sync_total_epochs(self, trainer=None):
            """Apply live queue changes to Lightning's actual FitLoop limit.

            Updating spec.json alone would not affect a Trainer already inside
            fit(); fit_loop.max_epochs is what controls its exit condition.
            """
            trainer = trainer or self.trainer
            target_path = job / "epoch-target.json"
            try:
                stat = target_path.stat()
                stamp = (stat.st_mtime_ns, stat.st_size)
            except FileNotFoundError:
                stamp = None
            # Batch callbacks can run many times per second. Avoid parsing
            # and validating a JSON document when no epoch edit occurred.
            if stamp == self.epoch_target_stamp:
                target = int(spec["preferences"]["epochs"])
                # A restored Lightning fit loop must not replace a previously
                # acknowledged live target with a checkpoint's old limit.
                if trainer is not None and trainer.fit_loop.max_epochs != target:
                    trainer.fit_loop.max_epochs = target
                return target
            self.epoch_target_stamp = stamp
            previous = int(spec["preferences"]["epochs"])
            target = requested_total_epochs(job, spec)
            if trainer is not None and trainer.fit_loop.max_epochs != target:
                trainer.fit_loop.max_epochs = target
            if target != previous:
                shared["epochs"] = target
                if trainer is not None:
                    self.write(
                        epochs=target,
                        eta=self.batch_estimate(trainer),
                        message=f"Total epochs changed to {target}; training {label}",
                    )
            return target

        def check_action(self):
            self.sync_total_epochs()
            action = self.action()
            if action == "pause":
                print(flush=True)
                info = read_json(job / f"resume-{member}.json", {})
                info["best_esr"] = self.best
                self.write(
                    status="paused",
                    batch_fraction=0,
                    message=f"Paused. Resume restarts epoch {info.get('restart_epoch', 1)} from the last complete checkpoint.",
                    **info,
                )
                raise TrainingPaused()
            if action == "stop":
                print(flush=True)
                self.write(status="stopped", message="Training cancelled")
                raise TrainingStopped()
            self.check_export_request()

        def state_dict(self):
            import random
            import numpy as np
            import torch

            rng = np.random.get_state()
            return {
                "validation_reference": "full",
                "best_validation": self.best_metrics,
                "last_validation": self.last_metrics,
                "best_lite_validation": self.best_lite_metrics,
                "best_esr": self.best,
                "best_train_esr": self.best_train,
                "epoch_times": self.epoch_times,
                "batch_durations": list(self.timing.durations),
                "validation_times": list(self.timing.validation_times),
                "esr_history": self.esr_history,
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "numpy_rng": (rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]),
                "cuda_rng": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
            }

        def on_save_checkpoint(self, trainer, module, checkpoint):
            checkpoint["validation_reference"] = "full"
            checkpoint["validation_esr"] = (self.last_metrics or {}).get("full")
            checkpoint["validation_esr_full"] = (self.last_metrics or {}).get("full")
            checkpoint["validation_esr_lite"] = (self.last_metrics or {}).get("lite")
            checkpoint["validation_esr_worst_full"] = (self.last_metrics or {}).get(
                "worst_full"
            )
            checkpoint["validation_esr_worst_lite"] = (self.last_metrics or {}).get(
                "worst_lite"
            )
            checkpoint["best_validation_esr_full"] = self.best
            checkpoint["best_train_esr_full"] = self.best_train
            checkpoint["train_esr_full"] = (self.last_metrics or {}).get("train_full")
            checkpoint["train_esr_lite"] = (self.last_metrics or {}).get("train_lite")

        def load_state_dict(self, data):
            import random
            import numpy as np
            import torch

            current_reference = data.get("validation_reference") == "full"
            self.best = data.get("best_esr") if current_reference else None
            self.best_train = data.get("best_train_esr") if current_reference else None
            self.best_metrics = (
                data.get("best_validation") if current_reference else None
            )
            self.last_metrics = (
                data.get("last_validation") if current_reference else None
            )
            self.epoch_times = data.get("epoch_times", [])
            self.timing.durations.extend(data.get("batch_durations", []))
            self.timing.validation_times.extend(data.get("validation_times", []))
            self.esr_history = data.get("esr_history", self.esr_history)
            self.best_lite_metrics = (
                data.get("best_lite_validation") if current_reference else None
            )
            if not current_reference and data.get("best_esr") is not None:
                print(
                    "Resuming a legacy checkpoint: Full-based best ESR history starts from the next validation epoch.",
                    flush=True,
                )
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
                best_train_esr=self.best_train,
                validation_reference="full",
                epoch_times=self.epoch_times,
                esr_history=self.esr_history,
                esr_history_member=member,
            )
            shared.update(updates)
            shared["updated_at"] = time.time()
            atomic_write_json(job / "progress.json", shared)

        def action(self):
            return read_json(job / "command.json", {}).get("action")

        def on_fit_start(self, trainer, module):
            from .checkpoints import install_durable_checkpoint_io

            install_durable_checkpoint_io(trainer)
            self.sync_total_epochs(trainer)

        def on_train_start(self, trainer, module):
            import torch

            self.trainer = trainer

            device_name = (
                torch.cuda.get_device_name(module.device)
                if module.device.type == "cuda"
                else str(module.device)
            )
            batches = trainer.num_training_batches
            batches = int(batches) if math.isfinite(batches) else None
            shared.setdefault("member_train_batches", {})[str(member)] = batches
            self.write(
                recovery_required=False,
                device=str(module.device),
                device_name=device_name,
                optimizer=spec["model"]["optimizer"].get(
                    "class", "Adam" if spec.get("static_captures") else "AdamW"
                ),
                batch_size=spec["preferences"]["batch_size"],
                train_batches=batches,
                message="Training " + label,
            )
            print(
                f"Training device: {device_name}; batch size: {spec['preferences']['batch_size']}; batches per epoch: {batches}; model: {label}",
                flush=True,
            )
            if not (job / f"resume-{member}.ckpt").exists():
                self.save_boundary(trainer, 0)
            self.check_action()

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            self.check_action()

        def on_validation_batch_start(
            self, trainer, module, batch, batch_idx, dataloader_idx=0
        ):
            self.check_action()

        def on_validation_batch_end(
            self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0
        ):
            self.check_action()

        def on_train_epoch_start(self, trainer, module):
            self.epoch_started = time.monotonic()
            self.batch_started = self.epoch_started
            self.batch_done = 0
            self.write(
                epoch=trainer.current_epoch + 1,
                batch_fraction=0,
                eta=self.batch_estimate(trainer),
            )
            self.print_progress(trainer, 0, "Training")

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            now = time.monotonic()
            if self.batch_started is not None:
                self.timing.record(now - self.batch_started)
            self.batch_started = now
            self.batch_done = batch_idx + 1
            self.check_action()
            # Throttle disk/GUI updates on short CPU batches.
            if (
                now - self.last_log_update >= 0.5
                or self.batch_done == trainer.num_training_batches
            ):
                batches = trainer.num_training_batches
                fraction = (
                    (batch_idx + 1) / batches
                    if math.isfinite(batches) and batches
                    else 0
                )
                self.write(
                    batch_fraction=fraction,
                    batches_completed=self.batch_done,
                    iterations_per_second=self.timing.iterations_per_second,
                    eta=self.batch_estimate(trainer),
                    message="Training " + label,
                )
                self.print_progress(trainer, fraction, "Training")
                self.last_log_update = time.monotonic()

        def on_validation_start(self, trainer, module):
            if not trainer.sanity_checking:
                self.validation_started = time.monotonic()
                self.print_progress(trainer, 1, "Validating")

        def on_validation_end(self, trainer, module):
            if trainer.sanity_checking:
                return
            self.check_action()
            metric = trainer.callback_metrics.get("ESR")
            esr = float(metric.detach().cpu()) if metric is not None else None

            def size_esr(size):
                index = 0 if size == "lite" else 1
                key = (
                    f"ESR_packed_{index}"
                    if spec.get("static_captures")
                    else f"ESR/unseen_audio_unseen_params/{size}"
                )
                value = trainer.callback_metrics.get(key)
                return float(value.detach().cpu()) if value is not None else None

            lite, full = size_esr("lite"), size_esr("full")

            def metric_float(key):
                value = trainer.callback_metrics.get(key)
                return float(value.detach().cpu()) if value is not None else None

            worst = metric_float("ESR_worst")
            worst_full, worst_lite = metric_float("ESR_worst_full"), metric_float(
                "ESR_worst_lite"
            )
            train_scores = {"lite": None, "full": None}
            data = (
                evaluation_data if evaluation_data is not None else spec.get("data", {})
            )
            if data.get("train_evaluation"):
                from .evaluation import TrainEvaluator

                if self.train_evaluator is None:
                    self.train_evaluator = TrainEvaluator(
                        data, spec["model"], module.net,
                        synthetic_indices=spec["learning"].get(
                            "synthetic_mute_train_indices", ()
                        ),
                    )
                self.print_progress(trainer, 1, "Evaluating Train ESR")
                train_scores = self.train_evaluator.compute(
                    module.net, module.device, self.check_action
                )
            train_full, train_lite = train_scores["full"], train_scores["lite"]
            if (
                train_full is not None
                and math.isfinite(train_full)
                and (self.best_train is None or train_full < self.best_train)
            ):
                self.best_train = train_full
            epoch = trainer.current_epoch + 1
            total_epochs = spec["preferences"]["epochs"]

            def display(value):
                return f"{value:.8g}" if value is not None else "unavailable"

            from .training_log import epoch_line

            print(
                "\r"
                + epoch_line(
                    epoch,
                    total_epochs,
                    label,
                    1,
                    "Validation ESR",
                    self.timing.iterations_per_second,
                )
                + " | "
                f"Lite={display(lite)} | Full (checkpoint)={display(full)} | Mean={display(esr)}"
                + (
                    f" | ESR_worst | Lite={display(worst_lite)} | Full={display(worst_full)} | Combined={display(worst)}"
                    if worst is not None
                    else ""
                )
                + f" | Train ESR | Lite={display(train_lite)} | Full={display(train_full)}",
                flush=True,
            )
            self.last_metrics = dict(epoch=epoch, full=full, lite=lite, mean=esr)
            self.last_metrics.update(train_full=train_full, train_lite=train_lite)
            if worst is not None:
                self.last_metrics.update(
                    worst=worst, worst_full=worst_full, worst_lite=worst_lite
                )
            self.esr_history = [
                entry for entry in self.esr_history if entry["epoch"] < epoch
            ]
            self.esr_history.append(dict(self.last_metrics))
            if (
                full is not None
                and math.isfinite(full)
                and (self.best is None or full < self.best)
            ):
                self.best = full
                self.best_metrics = dict(self.last_metrics)
            if (
                lite is not None
                and math.isfinite(lite)
                and (
                    self.best_lite_metrics is None
                    or lite < self.best_lite_metrics["lite"]
                )
            ):
                self.best_lite_metrics = dict(epoch=epoch, lite=lite)
            if (
                spec.get("learning", {}).get("separate_size_checkpoints")
                and self.best_metrics is not None
                and self.best_lite_metrics is not None
            ):
                self.best_metrics.update(
                    lite=self.best_lite_metrics["lite"],
                    epoch_full=self.best_metrics["epoch"],
                    epoch_lite=self.best_lite_metrics["epoch"],
                    mean=(self.best_metrics["full"] + self.best_lite_metrics["lite"])
                    / 2,
                )
            duration = max(
                time.monotonic() - (self.epoch_started or time.monotonic()), 0.001
            )
            self.epoch_times.append(duration)
            if self.validation_started is not None:
                self.timing.validation_times.append(
                    time.monotonic() - self.validation_started
                )
            eta = self.batch_estimate(trainer, validated=True)
            self.write(
                epoch=trainer.current_epoch + 1,
                eta=eta,
                batch_fraction=1,
                esr_lite=lite,
                esr_full=full,
                esr_mean=esr,
                esr_worst=worst,
                esr_worst_full=worst_full,
                esr_worst_lite=worst_lite,
                train_esr_full=train_full,
                train_esr_lite=train_lite,
            )

        def teardown(self, trainer, module, stage):
            if self.train_evaluator is not None:
                self.train_evaluator.close()
                self.train_evaluator = None

        def on_fit_end(self, trainer, module):
            self.save_boundary(trainer, trainer.current_epoch)
            self.check_action()

    return ProgressCallback()


def epoch_checkpoint(progress, job):
    from pytorch_lightning.callbacks import ModelCheckpoint

    class EpochCheckpoint(ModelCheckpoint):
        """Run after module epoch-end hooks, like Lightning's native checkpoints."""

        def on_train_epoch_end(self, trainer, module):
            progress.save_boundary(trainer, trainer.current_epoch + 1)

    return EpochCheckpoint(
        dirpath=job, save_top_k=0, save_last=False, save_on_train_epoch_end=True
    )


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
        settings_on_capture_grid,
    )
    from nam.capture.project import CaptureEntryModel
    from nam.data import wav_to_np
    from nam.train.parametric import _ParametricLightningModule, _get_parametric_net

    controls = workspace.project.control_specs()
    volume = next((c for c in controls if getattr(c, "clean_volume", False)), None)
    specs = tuple(c.to_param_spec() for c in controls if c is not volume)
    existing = {
        _settings_key(e.params, specs): e.params
        for e in workspace.project.entries
        if e.split in ("train", "validation")
    }
    excluded = set(existing)
    available = _grid_capacity(specs, default_step=0.5) - sum(
        settings_on_capture_grid(params, specs) for params in existing.values()
    )
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
    if volume is not None:
        selected = [{**params, volume.name: volume.max} for params in selected]
        specs = tuple(c.to_param_spec() for c in controls)
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
    job = Path(job).resolve()
    from .training_environment import prepare_worker_environment, apply_backend_flags

    prepare_worker_environment(job)
    import torch
    import pytorch_lightning as pl
    from filelock import FileLock
    from nam.train.parametric import main as train

    spec = read_json(job / "spec.json")
    apply_backend_flags(spec)
    workspace = Workspace(Path(spec["project"]))
    shared = job_progress(job)
    elapsed_base = float(shared.get("elapsed", 0))
    clock_start = time.monotonic()
    with FileLock(str(workspace.directory / "training" / "worker.lock"), timeout=0):
        # A queue epoch edit can arrive during the worker's startup window,
        # between the first spec read and acquiring worker.lock. Re-read the
        # atomic snapshot here so a legitimate live edit does not look like
        # an incompatible project mutation.
        spec = read_json(job / "spec.json")
        workspace = Workspace(Path(spec["project"]))
        requested_total_epochs(job, spec)
        if (
            workspace.fingerprint() != spec["fingerprint"]
            and not _matches_legacy_fingerprint(workspace, spec)
        ):
            raise ValueError("Project changed since this training job was prepared")
        os.chdir(workspace.directory)
        torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) - 1)))
        try:
            from .evaluation import PROTOCOL

            if spec.get("evaluation_protocol") != PROTOCOL:
                raise ValueError(
                    "Retrain this instance to reserve the initial and final validation audio. "
                    "Existing captures can be reused."
                )
            if spec.get("static_captures") is not None:
                from .static_training import run_batch

                return run_batch(
                    job, spec, workspace, shared, elapsed_base, clock_start
                )
            if (
                spec.get("queue_snapshot")
                and spec["learning"].get("snr_weighted_loss_normalization")
                and "snr_noise_powers" not in spec["learning"]
            ):
                # Older queue snapshots renamed recordings without keeping the
                # noise measurement. Recover it only from matching source WAVs.
                from .training_queue import freeze_snr_noise

                spec["learning"]["snr_noise_powers"] = freeze_snr_noise(
                    spec["original_project"], workspace
                )
                atomic_write_json(job / "spec.json", spec)
            completed = list(shared.get("completed_members", []))
            member = 0
            while member < spec["members"]:
                target = requested_total_epochs(job, spec)
                saved = read_json(job / f"resume-{member}.json", {})
                saved_epochs = int(saved.get("completed_epoch") or 0)
                if member in completed and (saved_epochs >= target or not saved):
                    # Earlier completed snapshots may have an export and a
                    # Lightning checkpoint but no resume-N.json metadata.
                    # Preserve their recorded completion; explicit extension
                    # clears completed_members before entering this loop.
                    member += 1
                    continue
                if member in completed:
                    completed.remove(member)
                    shared["completed_members"] = list(completed)
                pl.seed_everything(938 + spec["round"] * 101 + member, workers=True)
                output = job / f"member-{member}"
                output.mkdir(exist_ok=True)
                learning = json.loads(json.dumps(spec["learning"]))
                if learning["trainer"].get("deterministic") is True:
                    seed = 938 + spec["round"] * 101 + member
                    learning["train_dataloader"]["sampler_seed"] = seed
                    learning["val_dataloader"]["sampler_seed"] = seed + 1
                learning["trainer"]["enable_progress_bar"] = False
                if (
                    spec["preferences"]["device"] == "auto"
                    and os.environ.get("PARANAM_FORCE_CPU") == "1"
                ):
                    learning["trainer"]["accelerator"] = "cpu"
                pinned = (
                    learning["trainer"]["accelerator"] != "cpu"
                    and torch.cuda.is_available()
                )
                for split in ("train_dataloader", "val_dataloader"):
                    learning[split]["pin_memory"] = pinned
                checkpoint = job / f"resume-{member}.ckpt"
                if checkpoint.exists():
                    learning["trainer_fit_kwargs"]["ckpt_path"] = str(checkpoint)
                shared["elapsed"] = elapsed_base + time.monotonic() - clock_start
                progress = build_progress_callback(job, spec, member, shared)
                selector = None
                if not spec.get("static_captures"):
                    from .checkpoint_selection import build_selector

                    selector = build_selector(
                        progress,
                        spec.get("preferences", {}).get("advanced_training", {}),
                        output,
                    )
                train(
                    spec["data"],
                    spec["model"],
                    learning,
                    output,
                    no_show=True,
                    make_plots=False,
                    extra_callbacks=[
                        progress,
                        *([selector] if selector is not None else []),
                        epoch_checkpoint(progress, job),
                    ],
                )
                # A live epoch increase may arrive after Lightning decided
                # its old fit loop was finished. Continue the same member
                # from its existing durable checkpoint rather than marking it
                # complete at the obsolete target.
                target = requested_total_epochs(job, spec)
                saved = read_json(job / f"resume-{member}.json", {})
                if int(saved.get("completed_epoch") or 0) < target:
                    continue
                if selector is not None:
                    chosen = selector.selected_checkpoints()
                    if all(chosen.values()):
                        full, lite = chosen["full"], chosen["lite"]
                        if _is_film(spec):
                            snapshot = torch.load(full["path"], map_location="cpu", weights_only=False)
                            lite = dict(
                                epoch=full["epoch"],
                                val=snapshot["validation_esr_lite"],
                                train=snapshot["train_esr_lite"],
                            )
                        progress.best = full["val"]
                        progress.best_metrics = dict(
                            epoch=full["epoch"], epoch_full=full["epoch"],
                            epoch_lite=lite["epoch"], full=full["val"],
                            lite=lite["val"], mean=(full["val"] + lite["val"]) / 2,
                            train_full=full["train"], train_lite=lite["train"],
                        )
                        progress.best_lite_metrics = dict(
                            epoch=lite["epoch"], lite=lite["val"]
                        )
                for filename in ("model_parametric.pnam", "model.nam"):
                    exported = output / filename
                    if exported.exists():
                        model_dict = read_json(exported)
                        if exported.name == "model_parametric.pnam" and not spec.get("static_captures"):
                            # Normal training's final exporter does not use the
                            # intermediate/snapshot path. Repair *every* member's
                            # full model here, including jobs begun before v1.0.0.
                            from nam.capture.export import apply_model_display_indices
                            apply_model_display_indices(
                                model_dict,
                                [control.model_dump() for control in workspace.project.knobs],
                                output_gain=spec["model"].get("output_gain")
                                or workspace.project.output_gain,
                            )
                        # A rename may occur while training is running.
                        # Reload Name from disk before finalizing this member.
                        current_name = read_json(job / "spec.json", {}).get(
                            "model", {}
                        ).get("metadata", {}).get("name")
                        if current_name:
                            from .model_names import set_model_metadata_name
                            set_model_metadata_name(model_dict, current_name)
                        annotate_validation_metadata(
                            model_dict,
                            progress.best_metrics,
                            spec["model"]["optimizer"].get("class", "AdamW"),
                            (
                                None
                                if spec.get("static_captures")
                                else spec.get("preferences", {}).get(
                                    "advanced_training", {}
                                )
                            ),
                        )
                        atomic_write_json(exported, model_dict)
                if member not in completed:
                    completed.append(member)
                next_member = member + 1
                # Never display 0 completed epochs when an existing job's
                # target is extended. Show the next member's real checkpoint
                # progress, or the just-finished epoch for the final member.
                display_member = (
                    next_member if next_member < spec["members"] else member
                )
                display_epochs = int(
                    read_json(job / f"resume-{display_member}.json", {}).get(
                        "completed_epoch"
                    ) or 0
                )
                shared.update(
                    completed_members=list(completed),
                    member=next_member,
                    epoch=display_epochs,
                    completed_epoch=display_epochs,
                )
                atomic_write_json(job / "progress.json", shared)
                member += 1
                if member == spec["members"]:
                    # The target can increase while a later ensemble member is
                    # training. Revisit earlier members that only reached the
                    # old limit, preserving each member's own optimizer state.
                    target = requested_total_epochs(job, spec)
                    pending = [
                        index for index in range(spec["members"])
                        if int(read_json(job / f"resume-{index}.json", {}).get(
                            "completed_epoch"
                        ) or 0) < target
                    ]
                    if pending:
                        member = pending[0]
            # Select the member with the lowest checkpoint ESR. All outputs are nested A2.
            quality = []
            for member in completed:
                checkpoint = torch.load(
                    job / f"resume-{member}.ckpt",
                    map_location="cpu",
                    weights_only=False,
                )
                from .checkpoint_selection import saved_selection, candidate_score

                settings = spec.get("preferences", {}).get("advanced_training", {})
                method = settings.get("checkpoint_method", "validation")
                alpha = settings.get("checkpoint_alpha", 0.75)
                selected = saved_selection(checkpoint, method, alpha)
                if selected and selected.get("full"):
                    # The ensemble ranking follows the user-selected criterion
                    # (the tolerance rule prioritizes validation across members).
                    entry = selected["full"]
                    score = (
                        candidate_score(entry, method, alpha)
                        if method == "combined" else entry["val"]
                    )
                else:
                    best = [
                        state.get("best_esr")
                        for state in checkpoint.get("callbacks", {}).values()
                        if isinstance(state, dict)
                        and state.get("best_esr") is not None
                        and state.get("validation_reference") == "full"
                    ]
                    score = min(float(v) for v in best) if best else float("inf")
                quality.append((score, member))
            chosen = min(quality)[1]
            export = (
                Path(spec.get("models_directory", str(workspace.directory / "models")))
                / f"round-{spec['round'] + 1}"
            )
            export.mkdir(parents=True, exist_ok=True)
            source = job / f"member-{chosen}" / "model_parametric.pnam"
            if not source.exists():
                # Completed members from a job started by an older app may
                # still use .nam. Read those without renaming the user's files.
                source = source.with_suffix(".nam")
            from .model_names import model_basename

            selected_model = read_json(source)
            current_name = read_json(job / "spec.json", {}).get(
                "model", {}
            ).get("metadata", {}).get("name")
            if current_name:
                from .model_names import set_model_metadata_name
                set_model_metadata_name(selected_model, current_name)
            model_metadata = selected_model.get("metadata", {})
            default_name = "ParaNAM_FiLMA2" if _is_film(spec) else "ParaNAM_HyperA2"
            destination = export / (model_basename(model_metadata.get("name"), default_name) + ".pnam")
            # Re-finalize even a previously completed member: on resume it
            # might have been exported by an older trainer, and is skipped in
            # the per-member loop above. Never copy an unannotated PNAM.
            from nam.capture.export import apply_model_display_indices
            apply_model_display_indices(
                selected_model,
                [control.model_dump() for control in workspace.project.knobs],
                output_gain=spec["model"].get("output_gain")
                or workspace.project.output_gain,
            )
            atomic_write_json(destination, selected_model)
            shared.update(export=str(destination))
            exported_validation = (
                selected_model.get("metadata", {}).get("training", {})
            )
            if (
                isinstance(exported_validation, dict)
                and exported_validation.get("validation_reference") == "full"
            ):
                shared["best_esr"] = exported_validation.get("validation_esr_full")
            additions = []
            if (
                spec["preferences"]["mode"] == "active_learning"
                and spec["round"] + 1 < spec["preferences"]["rounds"]
            ):
                additions = _propose_captures(workspace, job, spec, shared)
            shared.update(
                status="needs_captures" if additions else "completed",
                epoch=requested_total_epochs(job, spec),
                eta=0,
                message=(
                    f"{len(additions)} new captures suggested"
                    if additions
                    else f"{_parametric_name(spec)} is ready"
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
            member = shared.get("member", 0)
            saved = read_json(job / f"resume-{member}.json", {})
            shared.update(
                status="failed",
                message=str(error),
                traceback=traceback.format_exc(),
                eta=None,
                resume_available=(job / f"resume-{member}.ckpt").is_file(),
                **saved,
            )
            shared["elapsed"] = elapsed_base + time.monotonic() - clock_start
            atomic_write_json(job / "progress.json", shared)
            raise

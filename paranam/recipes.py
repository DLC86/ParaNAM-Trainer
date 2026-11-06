"""Named starting recipes; user edits remain separately persisted per capture type."""

import math
from pathlib import Path

PRESET_SOURCE = "https://github.com/phillipmself/neural-amp-modeler-parametric/tree/a3684411076be69dab53505df14922487697f2cf"

DATASET_RATE_PRESETS = ("a2_dataset", "a2_dataset_sqrt")
CONTROL_RATE_PRESETS = ("a2_updates", "a2_controls", "a2_slow")
UPDATE_RATE_PRESETS = (*CONTROL_RATE_PRESETS, *DATASET_RATE_PRESETS)
RATE_PRESET_LABELS = {
    "a2_updates": "A2 Cadence",
    "a2_slow": "A2 Slow",
    "a2_controls": "A2 Slower",
    "a2_dataset_sqrt": "A2 Dataset-normalized √",
    "a2_dataset": "A2 Dataset-normalized",
}


def control_cadence_multiplier(project):
    switches = sum(control.type == "switch" for control in project.knobs)
    return max(1, len(project.knobs) - switches) * 2**switches


def rate_cadence_multiplier(project, preset):
    if preset not in ("a2_controls", "a2_slow"):
        return 1
    factor = control_cadence_multiplier(project)
    return factor / math.sqrt(2) if preset == "a2_slow" else factor


def training_preset(name):
    if name not in ("HyperA2", "HyperWaveNet"):
        raise ValueError(f"Unknown training preset: {name}")
    # Start from the saved 2026-11-06 02:52 recipe, with the subsequently
    # requested dataset-normalized clocks. Every effective option is stored in jobs.
    hyper_a2 = name == "HyperA2"
    return dict(
        rate_preset="a2_dataset" if hyper_a2 else "manual",
        cudnn_benchmark=not hyper_a2,
        deterministic_training=False,
        hypernetwork_rate_preset="a2_dataset" if hyper_a2 else "manual",
        hypernetwork_lr_decay=0.994,
        hypernetwork_recipe=name,
        hypernetwork_mode="low_rank",
        hypernetwork_rank=2,
        condition_dilated_convolutions=False,
        branch_loss_reduction="mean",
        optimizer="Adam" if hyper_a2 else "AdamW",
        learning_rate=0.004 if hyper_a2 else 0.002,
        hypernetwork_learning_rate=0.0004 if hyper_a2 else 0.002,
        lr_decay=0.994,
        ny=8192,
        fit_mrstft=True,
        mrstft_weight=0.0005,
        weight_decay=3.17e-7,
        gradient_clipping=not hyper_a2,
        gradient_clip_val=1.0,
        check_nonfinite_gradients=not hyper_a2,
    )


def preset_batch_size(name):
    return 16 if name == "HyperA2" else 32


def a2_update_schedule(
    data,
    directory,
    batch_size,
    ny,
    epochs,
    learning_rate,
    hypernetwork_learning_rate,
    reference_gamma=0.994,
    *,
    cadence_multiplier=1,
    normalization=None,
):
    """Match the official single-standard-capture A2 decay per optimizer update.

    Header-only accounting mirrors Dataset cropping/context and the capture-grouped
    drop-last sampler. The reference always uses bundled standard audio, NY=8192,
    batch=16 and the official 10s / final-9s split, independent of custom inputs.
    cadence_multiplier spaces the same discrete reductions farther apart.
    Dataset normalization instead scales by actual complete updates/reference
    updates, linearly or by its square root. It never counts validation captures.
    """
    import soundfile as sf
    from nam.capture.export import (
        _WAVENET_DILATIONS,
        _WAVENET_KERNEL_SIZES,
        _WAVENET_HEAD,
    )
    from .signals import asset_path

    if batch_size < 1 or ny < 1 or epochs < 1 or not 0 <= reference_gamma <= 1:
        raise ValueError(
            "The A2 cadence preset requires positive batch, NY and epochs, and 0 <= decay <= 1"
        )
    if (
        isinstance(cadence_multiplier, bool)
        or not isinstance(cadence_multiplier, (int, float))
        or not math.isfinite(cadence_multiplier)
        or cadence_multiplier <= 0
    ):
        raise ValueError("The A2 cadence multiplier must be a positive finite number")
    receptive_field = (
        1
        + sum((k - 1) * d for k, d in zip(_WAVENET_KERNEL_SIZES, _WAVENET_DILATIONS))
        + _WAVENET_HEAD["kernel_size"]
        - 1
    )
    reference = sf.info(asset_path("standard.wav"))
    reference_samples = reference.frames - 19 * reference.samplerate - 1
    reference_steps = ((reference_samples - receptive_field + 1) // 8192) // 16
    if reference_steps < 1:
        raise ValueError("The bundled A2 reference signal is too short")
    steps, total_seconds = 0, 0
    for item in data["train"]:
        row = {**data.get("common", {}), **item}
        x = sf.info(Path(directory) / row["x_path"])
        y_path = Path(directory) / row["y_path"]
        if y_path.is_file():
            y = sf.info(y_path)
            if (x.frames, x.samplerate) != (y.frames, y.samplerate):
                raise ValueError(
                    "Training input and capture must have matching lengths and sample rates"
                )

        def endpoint(which):
            for suffix in ("_samples", "", "_seconds"):
                value = row.get(which + suffix)
                if value is not None:
                    return (
                        int(value * x.samplerate)
                        if suffix == "_seconds"
                        else int(value)
                    )
            return None

        start, stop, _ = slice(endpoint("start"), endpoint("stop")).indices(x.frames)
        count = stop - start - abs(int(row.get("delay", 0)))
        windows = max(0, (count - receptive_field + 1) // ny)
        capture_steps = windows // batch_size
        if capture_steps < 1:
            raise ValueError(
                "A training capture has no complete batch: reduce Batch Size or NY, or use a longer input signal"
            )
        steps += capture_steps
        total_seconds += count / x.samplerate
    if steps < 1:
        raise ValueError("Apply a capture plan before calculating the A2 cadence")
    if normalization not in (None, *DATASET_RATE_PRESETS):
        raise ValueError("Unknown dataset cadence normalization")
    ratio = steps / reference_steps
    if normalization is not None:
        cadence_multiplier = (
            ratio if normalization == "a2_dataset" else math.sqrt(ratio)
        )
    interval = (
        steps if normalization == "a2_dataset" else reference_steps * cadence_multiplier
    )
    decay_steps = int(interval) if float(interval).is_integer() else interval
    reductions = math.floor(steps * epochs / decay_steps)
    factor = reference_gamma**reductions
    return dict(
        protocol=(
            f"{normalization}-update-cadence-v1"
            if normalization
            else (
                "a2-standard-update-cadence-fractional-v1"
                if not float(interval).is_integer()
                else (
                    "a2-standard-update-cadence-v1"
                    if cadence_multiplier == 1
                    else "a2-standard-update-cadence-scaled-v1"
                )
            )
        ),
        reference_steps=reference_steps,
        cadence_multiplier=cadence_multiplier,
        decay_steps=decay_steps,
        reference_batch_size=16,
        reference_ny=8192,
        reference_gamma=reference_gamma,
        steps_per_epoch=steps,
        **({"dataset_reference_ratio": ratio} if normalization else {}),
        training_seconds=total_seconds,
        gamma_epoch=(
            math.exp(math.log(reference_gamma) * steps / decay_steps)
            if reference_gamma
            else 0.0
        ),
        final_learning_rate=learning_rate * factor,
        final_hypernetwork_learning_rate=hypernetwork_learning_rate * factor,
        equivalent_a2_epochs=steps * epochs / reference_steps,
        equivalent_decay_epochs=steps * epochs / decay_steps,
        reductions=reductions,
    )


def workspace_update_schedule(workspace):
    """Preview pending/captured rows using the same processing as a new job."""
    from nam.capture.export import build_data_config
    from .evaluation import reserve_validation_audio

    project = workspace.project.model_copy(deep=True)
    for entry in project.entries:
        entry.status = "captured"
    data = build_data_config(project)
    reserve_validation_audio(data)
    options = workspace.training_options()
    return a2_update_schedule(
        data,
        workspace.directory,
        workspace.training_batch_size(),
        options["ny"],
        workspace.preferences["epochs"],
        options["learning_rate"],
        options["hypernetwork_learning_rate"],
        options["lr_decay"],
        cadence_multiplier=rate_cadence_multiplier(project, options["rate_preset"]),
        normalization=(
            options["rate_preset"]
            if options["rate_preset"] in DATASET_RATE_PRESETS
            else None
        ),
    )


def rate_group_schedule(
    project, data, directory, batch_size, epochs, options, *, hypernetwork=False
):
    """Serializable scheduler/preview description for one independently timed group."""
    preset = (
        options.get("hypernetwork_rate_preset", "follow")
        if hypernetwork
        else options["rate_preset"]
    )
    follow = preset == "follow"
    if follow:
        preset = options["rate_preset"]
    rate = options["hypernetwork_learning_rate" if hypernetwork else "learning_rate"]
    gamma = (
        options.get("hypernetwork_lr_decay", options["lr_decay"])
        if hypernetwork and not follow
        else options["lr_decay"]
    )
    if preset == "manual":
        return dict(
            preset=preset,
            interval="epoch",
            step_size=1,
            gamma=gamma,
            initial_learning_rate=rate,
            final_learning_rate=rate * gamma**epochs,
            reductions=epochs,
        )
    if preset not in UPDATE_RATE_PRESETS:
        raise ValueError("Choose a listed learning-rate preset")
    schedule = a2_update_schedule(
        data,
        directory,
        batch_size,
        options["ny"],
        epochs,
        rate,
        rate,
        gamma,
        cadence_multiplier=rate_cadence_multiplier(project, preset),
        normalization=preset if preset in DATASET_RATE_PRESETS else None,
    )
    return dict(
        schedule,
        preset=preset,
        interval="step",
        step_size=schedule["decay_steps"],
        gamma=gamma,
        initial_learning_rate=rate,
    )


def workspace_group_schedules(workspace):
    from nam.capture.export import build_data_config
    from .evaluation import reserve_validation_audio

    project = workspace.project.model_copy(deep=True)
    for entry in project.entries:
        entry.status = "captured"
    data = build_data_config(project)
    reserve_validation_audio(data)
    return {
        name: rate_group_schedule(
            project,
            data,
            workspace.directory,
            workspace.training_batch_size(),
            workspace.preferences["epochs"],
            workspace.training_options(),
            hypernetwork=hyper,
        )
        for name, hyper in (("audio", False), ("hypernetwork", True))
    }

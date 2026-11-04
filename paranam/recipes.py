"""Named starting recipes; user edits remain separately persisted per capture type."""

import math
from pathlib import Path

PRESET_SOURCE = "https://github.com/phillipmself/neural-amp-modeler-parametric/tree/a3684411076be69dab53505df14922487697f2cf"


def training_preset(name):
    if name not in ("HyperA2", "HyperWaveNet"):
        raise ValueError(f"Unknown training preset: {name}")
    hyper_a2 = name == "HyperA2"
    return dict(
        rate_preset="manual",
        hypernetwork_recipe=name,
        hypernetwork_mode="full" if hyper_a2 else "low_rank",
        hypernetwork_rank=2,
        condition_dilated_convolutions=hyper_a2,
        branch_loss_reduction="sum" if hyper_a2 else "mean",
        optimizer="Adam" if hyper_a2 else "AdamW",
        learning_rate=0.004 if hyper_a2 else 0.002,
        hypernetwork_learning_rate=0.0004 if hyper_a2 else 0.002,
        lr_decay=0.994,
        ny=8192,
        fit_mrstft=True,
        mrstft_weight=0.0005,
        weight_decay=3.17e-7,
        gradient_clipping=True,
        gradient_clip_val=1.0,
        check_nonfinite_gradients=True,
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
):
    """Match the official single-standard-capture A2 decay per optimizer update.

    Header-only accounting mirrors Dataset cropping/context and the capture-grouped
    drop-last sampler. The reference always uses bundled standard audio, NY=8192,
    batch=16 and the official 10s / final-9s split, independent of custom inputs.
    cadence_multiplier spaces the same discrete reductions farther apart.
    """
    import soundfile as sf
    from nam.capture.export import (
        _WAVENET_DILATIONS,
        _WAVENET_KERNEL_SIZES,
        _WAVENET_HEAD,
    )
    from .signals import asset_path

    if batch_size < 1 or ny < 1 or epochs < 1 or not 0 < reference_gamma <= 1:
        raise ValueError(
            "The A2 cadence preset requires positive batch, NY and epochs, and 0 < decay <= 1"
        )
    if (
        isinstance(cadence_multiplier, bool)
        or not isinstance(cadence_multiplier, int)
        or cadence_multiplier < 1
    ):
        raise ValueError("The A2 cadence multiplier must be a positive integer")
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
    decay_steps = reference_steps * cadence_multiplier
    reductions = steps * epochs // decay_steps
    factor = reference_gamma**reductions
    return dict(
        protocol=(
            "a2-standard-update-cadence-v1"
            if cadence_multiplier == 1
            else "a2-standard-update-cadence-scaled-v1"
        ),
        reference_steps=reference_steps,
        cadence_multiplier=cadence_multiplier,
        decay_steps=decay_steps,
        reference_batch_size=16,
        reference_ny=8192,
        reference_gamma=reference_gamma,
        steps_per_epoch=steps,
        training_seconds=total_seconds,
        gamma_epoch=math.exp(math.log(reference_gamma) * steps / decay_steps),
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
        cadence_multiplier=(
            len(project.knobs) if options.get("rate_preset") == "a2_controls" else 1
        ),
    )

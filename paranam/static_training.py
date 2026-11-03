"""Independent standard A2 models for a named batch of static captures."""

import copy
import os
import re
import time
from pathlib import Path

from nam.capture.project import atomic_write_json
from .model_names import model_basename


def model_config(project, size=None):
    from nam.models.parametric._hyper_a2 import _template_config

    submodels = []
    for channels in (3, 8):
        template = _template_config(channels, 0.01)
        submodels.append(
            dict(
                name=f"channels_{channels}",
                config=dict(
                    layers_configs=template["layers"], head_scale=template["head_scale"]
                ),
            )
        )
    config = dict(
        submodels=submodels,
        sample_rate=project.sample_rate,
        export=dict(container_max_values="uniform"),
    )
    return dict(
        net=dict(name="PackedWaveNet", config=config),
        loss=dict(val_loss="esr", mrstft_weight=0.0005),
        optimizer=dict(lr=0.004, weight_decay=3.17e-7),
        lr_scheduler=dict(**{"class": "ExponentialLR"}, kwargs=dict(gamma=0.994)),
    )


def train(data_config, model_config, learning, output, callback):
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from nam.data import (
        Split,
        init_dataset,
        apply_joint_dataset_hooks,
        get_joint_dataset_hooks,
    )
    from nam.train.lightning_module import LightningModule, PackedLightningModule
    from .training import epoch_checkpoint

    module_class = (
        PackedLightningModule
        if model_config["net"]["name"] == "PackedWaveNet"
        else LightningModule
    )
    model = module_class.init_from_config(model_config)
    data = copy.deepcopy(data_config)
    data.setdefault("common", {})["nx"] = model.net.receptive_field
    datasets = []
    try:
        datasets = [
            init_dataset(data, split) for split in (Split.TRAIN, Split.VALIDATION)
        ]
        apply_joint_dataset_hooks(
            dataset_train=datasets[0],
            dataset_validation=datasets[1],
            hooks=get_joint_dataset_hooks(data.get("joint", [])),
        )
        model.net.sample_rate = datasets[0].sample_rate
        for dataset in datasets:
            dataset.handshake(model.net)
            model.net.handshake(dataset)
        loaders = [
            DataLoader(dataset, **learning[key])
            for dataset, key in zip(datasets, ("train_dataloader", "val_dataloader"))
        ]
        if len(loaders[0]) == 0:
            raise ValueError(
                "Training capture is too short for the selected NY and batch size. Reduce NY or Batch Size."
            )
        from nam.train.lightning_module import PackedBestCheckpoint
        from pathlib import Path
        from .workspace import read_json

        best_by_size = PackedBestCheckpoint(dirpath=output / "size-checkpoints")
        saved_sizes = read_json(output / "size-checkpoints" / "packed_best.json", {})
        best_by_size.best = {
            entry["submodel_index"]: entry
            for entry in saved_sizes.get("submodels", [])
            if Path(entry["checkpoint_path"]).is_file()
        }
        best = pl.callbacks.ModelCheckpoint(
            monitor="ESR_packed_1",
            save_top_k=1,
            filename="{epoch:04d}_{step}_ESR_full={ESR_packed_1:.3e}",
        )
        trainer = pl.Trainer(
            callbacks=[
                callback,
                *([best_by_size] if learning.get("separate_size_checkpoints") else []),
                best,
                epoch_checkpoint(callback, output.parent),
            ],
            default_root_dir=output,
            **learning["trainer"],
        )
        trainer.fit(model, *loaders, **learning.get("trainer_fit_kwargs", {}))
        if best.best_model_path:
            model = module_class.load_from_checkpoint(
                best.best_model_path, **module_class.parse_config(model_config)
            )
        model.eval()
        if learning.get("separate_size_checkpoints"):
            from nam.train.core import _apply_packed_best_checkpoints

            _apply_packed_best_checkpoints(
                model, model_config, best_by_size.checkpoint_paths
            )
        model.net.sample_rate = datasets[0].sample_rate
        model.net.export(output)
    finally:
        for dataset in datasets:
            dataset.teardown()


def run_batch(job, spec, workspace, shared, elapsed_base, clock_start):
    import pytorch_lightning as pl
    import torch
    from .training import (
        build_progress_callback,
        read_json,
        annotate_validation_metadata,
    )

    completed = shared.get("completed_members", [])
    exports = shared.get("exports", [])
    metadata = dict(workspace.project.user_metadata)
    for key in ("input_level_dbu", "output_level_dbu"):
        value = getattr(workspace.project, key)
        if value is not None:
            metadata[key] = value
    for member, capture in enumerate(spec["static_captures"]):
        if member in completed:
            continue
        pl.seed_everything(938 + member, workers=True)
        paths = {
            e.y_path for e in workspace.project.entries if e.capture_id == capture["id"]
        }
        data = copy.deepcopy(spec["data"])
        for split in ("train", "validation", "train_evaluation"):
            if split not in data:
                continue
            data[split] = [e for e in data[split] if e["y_path"] in paths]
            if len(data[split]) != 1:
                raise ValueError(
                    "Each static model requires one training and one validation recording"
                )
        output = job / f"member-{member}"
        output.mkdir(exist_ok=True)
        learning = copy.deepcopy(spec["learning"])
        learning["trainer"]["enable_progress_bar"] = False
        checkpoint = job / f"resume-{member}.ckpt"
        if checkpoint.exists():
            learning.setdefault("trainer_fit_kwargs", {})["ckpt_path"] = str(checkpoint)
        else:
            shared.update(best_esr=None, epoch=0)
        if (
            spec["preferences"]["device"] == "auto"
            and os.environ.get("PARANAM_FORCE_CPU") == "1"
        ):
            learning["trainer"]["accelerator"] = "cpu"
        pinned = (
            learning["trainer"]["accelerator"] != "cpu" and torch.cuda.is_available()
        )
        for split in ("train_dataloader", "val_dataloader"):
            learning[split]["pin_memory"] = pinned
        shared.update(
            elapsed=elapsed_base + time.monotonic() - clock_start,
            message=f"Training A2 Full + Lite: {capture['name']}",
        )
        callback = build_progress_callback(
            job, spec, member, shared, evaluation_data=data
        )
        train(data, spec["model"], learning, output, callback)
        model = read_json(output / "model.nam")
        model.setdefault("metadata", {}).update(metadata)
        model["metadata"].update(capture.get("metadata", {}), name=capture["name"])
        if model.get("architecture") == "SlimmableContainer":
            for child in model["config"]["submodels"]:
                child["model"].setdefault("metadata", {}).update(metadata)
                child["model"]["metadata"].update(
                    capture.get("metadata", {}), name=capture["name"]
                )
        annotate_validation_metadata(
            model,
            callback.best_metrics,
            spec["model"]["optimizer"].get("class", "Adam"),
        )
        safe_name = model_basename(model["metadata"].get("name"), "Capture")
        export_root = (
            Path(spec["models_directory"])
            if spec.get("models_directory")
            else workspace.directory / "models" / job.name
        )
        target = export_root / f"{safe_name}.nam"
        if str(target) in exports:
            target = export_root / capture["id"] / f"{safe_name}.nam"
        atomic_write_json(target, model)
        exports.append(str(target))
        completed.append(member)
        shared.update(
            completed_members=list(completed),
            exports=list(exports),
            export=str(target),
            member=member + 1,
            elapsed=elapsed_base + time.monotonic() - clock_start,
        )
        atomic_write_json(job / "progress.json", shared)
    shared.update(
        status="completed",
        member=max(0, spec["members"] - 1),
        epoch=spec["preferences"]["epochs"],
        eta=0,
        message=f"{len(exports)} static A2 models are ready",
        elapsed=elapsed_base + time.monotonic() - clock_start,
    )
    atomic_write_json(job / "progress.json", shared)

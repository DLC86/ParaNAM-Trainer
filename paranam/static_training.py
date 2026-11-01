"""Independent standard A2 models for a named batch of static captures."""

import copy
import os
import re
import time

from nam.capture.project import atomic_write_json


def model_config(project, size="full"):
    from nam.models.parametric._hyper_a2 import _template_config

    if size not in ("full", "lite"):
        raise ValueError("Choose A2 Full or Lite")
    template = _template_config(8 if size == "full" else 3, 0.01)
    config = dict(
        layers_configs=template["layers"],
        head_scale=template["head_scale"],
        sample_rate=project.sample_rate,
    )
    return dict(
        net=dict(name="WaveNet", config=config),
        loss=dict(val_loss="esr", mrstft_weight=0.0005),
        optimizer=dict(lr=0.002, weight_decay=3.17e-7),
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
    from nam.train.lightning_module import LightningModule

    model = LightningModule.init_from_config(model_config)
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
        best = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1)
        trainer = pl.Trainer(
            callbacks=[best, callback, pl.callbacks.TQDMProgressBar(refresh_rate=10)],
            default_root_dir=output,
            **learning["trainer"],
        )
        trainer.fit(model, *loaders, **learning.get("trainer_fit_kwargs", {}))
        if best.best_model_path:
            model = LightningModule.load_from_checkpoint(
                best.best_model_path, **LightningModule.parse_config(model_config)
            )
        model.eval()
        model.net.sample_rate = datasets[0].sample_rate
        model.net.export(output)
    finally:
        for dataset in datasets:
            dataset.teardown()


def run_batch(job, spec, workspace, shared, elapsed_base, clock_start):
    import pytorch_lightning as pl
    from .training import build_progress_callback, read_json

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
        for split in ("train", "validation"):
            data[split] = [e for e in data[split] if e["y_path"] in paths]
            if len(data[split]) != 1:
                raise ValueError(
                    "Each static model requires one training and one validation recording"
                )
        output = job / f"member-{member}"
        output.mkdir(exist_ok=True)
        learning = copy.deepcopy(spec["learning"])
        learning["trainer"]["enable_progress_bar"] = True
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
        shared.update(
            elapsed=elapsed_base + time.monotonic() - clock_start,
            message=f"Training A2 {spec['preferences']['static_size'].title()}: {capture['name']}",
        )
        callback = build_progress_callback(job, spec, member, shared)
        train(data, spec["model"], learning, output, callback)
        model = read_json(output / "model.nam")
        model.setdefault("metadata", {}).update(metadata)
        model["metadata"].update(capture.get("metadata", {}), name=capture["name"])
        safe_name = (
            re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", capture["name"]).rstrip(" .")
            or "Capture"
        )
        target = (
            workspace.directory
            / "models"
            / job.name
            / f"{member + 1:03d}-{safe_name}.nam"
        )
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

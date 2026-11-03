"""Resumable projects, non-destructive plan changes and application preferences."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from nam.capture.params import KnobSpec, SwitchSpec, validate_knobs
from nam.capture.project import (
    AudioSettingsModel,
    CaptureProject,
    CaptureEntryModel,
    KnobModel,
    atomic_write_json,
    add_corner_captures,
    load_project,
    new_project,
    reconcile_with_disk,
    save_project,
)

DEFAULTS = {
    "capture_level_dbfs": 0.0,
    "capture_type": "parametric",
    "mode": "normal",
    "epochs": 200,
    "batch_size": 16,
    "static_batch_size": 16,
    "device": "auto",
    "rounds": 2,
    "additions": 8,
    "ensemble_members": 2,
    "n_train": 30,
    "n_validation": 6,
    "corners": True,
    "seed": 0,
    "input_mode": "standard",
    "custom_input": "",
}


def read_json(path: Path, default=None):
    for attempt in range(10):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.01)


def control_from_dict(data):
    if data.get("type") == "switch":
        return SwitchSpec(
            data["name"], tuple(data["enum_names"]), data.get("default", 0)
        )
    return KnobSpec.from_dict(data)


class Workspace:
    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        self.project = load_project(self.directory)
        stored = read_json(self.directory / "paranam.json", {})
        self.preferences = {**DEFAULTS, **stored}
        if self.project.capture_type == "static" and "static_batch_size" not in stored:
            self.preferences["static_batch_size"] = stored.get("batch_size", 16)
        if "capture_level_dbfs" not in stored:
            self.preferences["capture_level_dbfs"] = self.project.capture_level_dbfs
        if "capture_type" not in stored:
            self.preferences["capture_type"] = self.project.capture_type
        if not self.preferences.get("static_captures"):
            groups = {}
            for entry in self.project.entries:
                if entry.capture_id:
                    groups.setdefault(
                        entry.capture_id,
                        {
                            "id": entry.capture_id,
                            "name": entry.capture_name or "Capture",
                            "metadata": dict(entry.user_metadata),
                        },
                    )
            self.preferences["static_captures"] = list(groups.values()) or [
                {"id": uuid.uuid4().hex, "name": "Capture 1"}
            ]
        if not stored and self.project.entries:
            self.preferences.update(
                n_train=self.project.n_train_lhs,
                n_validation=len(self.project.entries_for_split("validation")),
                seed=self.project.seed,
                corners=self.project.include_initial_corners,
            )
        if not stored and self.project.entries:
            self.preferences.update(
                input_mode="custom", custom_input=self.project.train_input
            )
        self.draft_controls = self.preferences.get("draft_controls") or [
            c.to_dict() for c in self.project.control_specs()
        ]
        reconcile_with_disk(self.project, self.directory)

    @classmethod
    def create(cls, directory, name, audio=None):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "capture_project.json").exists():
            raise ValueError("This folder already contains a project. Open it instead.")
        # Never overwrite recordings or arbitrary input files in an existing folder.
        if any(directory.iterdir()):
            raise ValueError("Choose an empty folder for a new project.")
        controls = [KnobSpec("Gain", 0, 10, is_gain=True), KnobSpec("Tone", 0, 10)]
        project = CaptureProject(
            name=name,
            knobs=[KnobModel.from_knob_spec(c) for c in controls],
            audio=audio or AudioSettingsModel(),
            sample_rate=48000,
        )
        project.train_window.ny = project.validation_window.ny = 8192
        from .signals import install_signals

        install_signals(directory)
        project.train_input = "input_standard.wav"
        save_project(project, directory)
        workspace = cls(directory)
        workspace.save()
        return workspace

    def save(self):
        self.preferences["draft_controls"] = self.draft_controls
        atomic_write_json(self.directory / "paranam.json", self.preferences)
        save_project(self.project, self.directory)

    def controls(self):
        if self.preferences["capture_type"] == "static":
            return ()
        return validate_knobs([control_from_dict(c) for c in self.draft_controls])

    def static_groups(self):
        groups = {}
        for entry in self.project.entries:
            if entry.capture_id:
                groups.setdefault(entry.capture_id, []).append(entry)
        return list(groups.values())

    def add_static_capture(self, name):
        name = str(name).strip()
        if not name or any(
            c["name"].casefold() == name.casefold()
            for c in self.preferences["static_captures"]
        ):
            raise ValueError("Use a non-empty, unique capture name")
        capture = {"id": uuid.uuid4().hex, "name": name}
        self.preferences["static_captures"].append(capture)
        if self.project.capture_type == "static":
            self.project.entries.extend(
                self._static_pair(capture, len(self.project.entries))
            )
            self.project.n_train_lhs = len(self.static_groups())
        self.save()
        return capture

    @staticmethod
    def _static_pair(capture, offset):
        return [
            CaptureEntryModel(
                index=offset + i,
                split=split,
                params={},
                capture_id=capture["id"],
                capture_name=capture["name"],
                user_metadata=dict(capture.get("metadata", {})),
                y_path=f"captures/static-{capture['id']}-{split}.wav",
            )
            for i, split in enumerate(("train", "validation"))
        ]

    def rename_static_capture(self, capture_id, name):
        name = str(name).strip()
        if not name or any(
            c["id"] != capture_id and c["name"].casefold() == name.casefold()
            for c in self.preferences["static_captures"]
        ):
            raise ValueError("Use a non-empty, unique capture name")
        for capture in self.preferences["static_captures"]:
            if capture["id"] == capture_id:
                capture["name"] = name
        for entry in self.project.entries:
            if entry.capture_id == capture_id:
                entry.capture_name = name
        self.save()

    def set_static_metadata(self, capture_id, metadata):
        for capture in self.preferences["static_captures"]:
            if capture["id"] == capture_id:
                capture["metadata"] = dict(metadata)
        for entry in self.project.entries:
            if entry.capture_id == capture_id:
                entry.user_metadata = dict(metadata)
        self.save()

    def desired_train_input(self):
        if self.preferences["input_mode"] == "standard":
            return "input_standard.wav"
        if self.preferences["input_mode"] == "tts":
            from .signals import install_signals

            if not (self.directory / "input_tts_v11.wav").is_file():
                install_signals(self.directory, include_tts=True)
            return "input_tts_v11.wav"
        relative = self.preferences.get("custom_input")
        if not relative or not (self.directory / relative).is_file():
            raise ValueError(
                "Choose a custom mono 48 kHz training WAV in the Controls step."
            )
        return relative

    def plan_changed(self):
        calibration = self.preferences.get("pending_calibration", {})
        if any(
            value != getattr(self.project, key) for key, value in calibration.items()
        ):
            return True
        if self.preferences["capture_level_dbfs"] != self.project.capture_level_dbfs:
            return True
        if self.preferences["capture_type"] != self.project.capture_type:
            return True
        if self.preferences["capture_type"] == "static":
            actual = [g[0].capture_id for g in self.static_groups()]
            return (
                self.desired_train_input() != self.project.train_input
                or not self.validation_matches()
                or actual != [c["id"] for c in self.preferences["static_captures"]]
            )
        if not self.project.entries:
            return True
        return (
            self.desired_train_input() != self.project.train_input
            or not self.validation_matches()
            or [c.to_dict() for c in self.controls()]
            != [c.to_dict() for c in self.project.control_specs()]
            or self.preferences["n_train"] != self.project.n_train_lhs
            or self.preferences["n_validation"]
            != len(self.project.entries_for_split("validation"))
            or self.preferences["seed"] != self.project.seed
            or self.preferences["corners"] != self.project.include_initial_corners
        )

    def archive_captures(self):
        if not self.project.entries:
            return None
        target = (
            self.directory
            / "history"
            / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        )
        target.mkdir(parents=True)
        for filename in (
            "capture_project.json",
            "data.json",
            "paranam.json",
            self.project.train_input,
            self.project.validation_input,
        ):
            source = self.directory / filename
            if source.is_file():
                dest = target / filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        for name in ("captures", "captures_raw"):
            source = self.directory / name
            if source.exists():
                # Every move is bounded by the project root; original recordings survive.
                if (
                    self.directory not in source.resolve().parents
                    or source.is_symlink()
                ):
                    raise ValueError(
                        "Cannot archive captures outside the project folder"
                    )
                source.rename(target / name)
        return target

    def generate_plan(self):
        from .calibration import gain_from_dbfs

        if not gain_from_dbfs(self.preferences["capture_level_dbfs"]):
            raise ValueError(
                "Capture output is muted (-inf). Choose a finite output level before creating a capture plan."
            )
        controls = self.controls()
        prefs = self.preferences
        calibration = prefs.get("pending_calibration", {})
        if prefs["capture_type"] == "static":
            replacement = CaptureProject(
                name=self.project.name,
                knobs=[],
                capture_type="static",
                capture_level_dbfs=prefs["capture_level_dbfs"],
                user_metadata=dict(self.project.user_metadata),
                train_input=self.desired_train_input(),
                validation_input="input_validation.wav",
                sample_rate=self.project.sample_rate,
                audio=self.project.audio.model_copy(deep=True),
                input_level_dbu=calibration.get(
                    "input_level_dbu", self.project.input_level_dbu
                ),
                output_level_dbu=calibration.get(
                    "output_level_dbu", self.project.output_level_dbu
                ),
                train_window=self.project.train_window.model_copy(deep=True),
                validation_window=self.project.validation_window.model_copy(deep=True),
                n_train_lhs=len(prefs["static_captures"]),
            )
            for capture in prefs["static_captures"]:
                replacement.entries.extend(
                    self._static_pair(capture, len(replacement.entries))
                )
            self.archive_captures()
            from .signals import install_signals

            install_signals(self.directory)
            self.project = replacement
            prefs.pop("pending_calibration", None)
            self.save()
            return replacement
        switches_only = all(isinstance(c, SwitchSpec) for c in controls)
        validation_count = prefs["n_validation"]
        if validation_count < 1:
            raise ValueError(
                "Include validation captures: they use the fixed bundled validation signal and provide ESR."
            )
        replacement = new_project(
            controls,
            n_train=prefs["n_train"],
            n_validation=0 if switches_only else validation_count,
            seed=prefs["seed"],
            name=self.project.name,
            train_input=self.desired_train_input(),
            validation_input="input_validation.wav",
        )
        if switches_only:
            # No unseen categorical states exist, but the fixed validation AUDIO
            # remains held out. Capture the same states with that independent signal.
            if validation_count != len(replacement.entries):
                raise ValueError(
                    "For switches only, validate every state once using the separate bundled signal."
                )
            from nam.capture.planner import _planned_from_params
            from nam.capture.project import CaptureEntryModel

            planned = _planned_from_params(
                "val",
                "validation",
                tuple(c.to_param_spec() for c in controls),
                [dict(e.params) for e in replacement.entries],
            )
            replacement.entries.extend(
                CaptureEntryModel(
                    index=e.index, split=e.split, params=e.params, y_path=e.y_path
                )
                for e in planned
            )
        replacement.audio = self.project.audio.model_copy(deep=True)
        replacement.capture_level_dbfs = prefs["capture_level_dbfs"]
        replacement.user_metadata = dict(self.project.user_metadata)
        replacement.sample_rate = self.project.sample_rate
        replacement.input_level_dbu = calibration.get(
            "input_level_dbu", self.project.input_level_dbu
        )
        replacement.output_level_dbu = calibration.get(
            "output_level_dbu", self.project.output_level_dbu
        )
        replacement.train_window = self.project.train_window.model_copy(deep=True)
        replacement.validation_window = self.project.validation_window.model_copy(
            deep=True
        )
        replacement.include_initial_corners = prefs["corners"]
        if prefs["corners"]:
            add_corner_captures(replacement)
        self.archive_captures()
        from .signals import install_signals

        install_signals(self.directory)
        self.project = replacement
        prefs.pop("pending_calibration", None)
        self.save()
        return replacement

    def validation_matches(self):
        from .signals import asset_path

        path = self.directory / self.project.validation_input
        return (
            self.project.validation_input == "input_validation.wav"
            and path.is_file()
            and hashlib.sha256(path.read_bytes()).digest()
            == hashlib.sha256(asset_path("validation.wav").read_bytes()).digest()
        )

    def training_ready(self):
        if self.preferences["capture_level_dbfs"] != self.project.capture_level_dbfs:
            return False
        if not self.project.entries or not self.validation_matches():
            return False
        return all(
            e.status == "captured"
            and (self.directory / e.y_path).is_file()
            and (
                e.qa is None
                or (
                    e.qa.end_timing_detected is not False
                    and (
                        e.qa.clock_drift_samples is None
                        or abs(e.qa.clock_drift_samples) < 0.1 - 1e-9
                    )
                )
            )
            for e in self.project.entries
        )

    def training_batch_size(self):
        key = (
            "static_batch_size"
            if self.preferences["capture_type"] == "static"
            else "batch_size"
        )
        return self.preferences[key]

    def training_options(self, kind=None):
        kind = kind or self.preferences["capture_type"]
        defaults = dict(
            optimizer="Adam",
            learning_rate=0.004,
            lr_decay=0.994,
            ny=self.project.train_window.ny or 8192,
            fit_mrstft=True,
            mrstft_weight=0.0005,
            weight_decay=3.17e-7,
            gradient_clipping=(kind == "parametric"),
            gradient_clip_val=1.0,
            check_nonfinite_gradients=(kind == "parametric"),
            hypernetwork_recipe="HyperA2",
        )
        options = {**defaults, **self.preferences.get("advanced_" + kind, {})}
        options.setdefault("hypernetwork_learning_rate", options["learning_rate"] * 0.1)
        hyper_a2 = options["hypernetwork_recipe"] == "HyperA2"
        options.setdefault("hypernetwork_mode", "full" if hyper_a2 else "low_rank")
        options.setdefault("hypernetwork_rank", 2)
        options.setdefault("condition_dilated_convolutions", hyper_a2)
        options.setdefault("branch_loss_reduction", "sum" if hyper_a2 else "mean")
        return options

    def fingerprint(
        self,
        *,
        batch_size=None,
        include_advanced=True,
        include_optimizer=True,
        include_gradient_safety=True,
        include_hypernetwork_controls=True,
    ):
        """Reject checkpoints when captures, controls, calibration or training choices change."""
        from nam.capture.export import build_data_config

        data = {
            **(
                {"capture_level_dbfs": self.project.capture_level_dbfs}
                if self.project.capture_level_dbfs != 0.0
                else {}
            ),
            "capture_type": self.project.capture_type,
            "static_size": (
                "full+lite-v1" if self.project.capture_type == "static" else None
            ),
            "static_batch": (
                self.preferences["static_captures"]
                if self.project.capture_type == "static"
                else None
            ),
            "metadata": self.project.user_metadata,
            "controls": [c.to_dict() for c in self.project.control_specs()],
            "data": build_data_config(self.project),
            "calibration": [
                (
                    float(self.project.input_level_dbu)
                    if self.project.input_level_dbu is not None
                    else None
                ),
                (
                    float(self.project.output_level_dbu)
                    if self.project.output_level_dbu is not None
                    else None
                ),
            ],
            "settings": {
                k: self.preferences[k]
                for k in (
                    "mode",
                    "epochs",
                    "batch_size",
                    "rounds",
                    "additions",
                    "ensemble_members",
                )
            },
            "files": [],
        }
        for relative in [self.project.train_input, self.project.validation_input] + [
            e.y_path for e in self.project.entries
        ]:
            path = self.directory / relative
            stat = path.stat()
            data["files"].append([relative, stat.st_size, stat.st_mtime_ns])
        data["settings"]["batch_size"] = self.training_batch_size()
        if include_advanced:
            data["settings"]["advanced_training"] = self.training_options()
            if not include_optimizer:
                data["settings"]["advanced_training"].pop("optimizer")
            if not include_gradient_safety:
                for key in (
                    "gradient_clipping",
                    "gradient_clip_val",
                    "check_nonfinite_gradients",
                    "hypernetwork_learning_rate",
                    "hypernetwork_recipe",
                ):
                    data["settings"]["advanced_training"].pop(key, None)
            if not include_gradient_safety or not include_hypernetwork_controls:
                for key in (
                    "hypernetwork_mode",
                    "hypernetwork_rank",
                    "condition_dilated_convolutions",
                    "branch_loss_reduction",
                ):
                    data["settings"]["advanced_training"].pop(key, None)
        if batch_size is not None:
            data["settings"]["batch_size"] = batch_size
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

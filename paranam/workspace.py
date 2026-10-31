"""Resumable projects, non-destructive plan changes and application preferences."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from nam.capture.params import KnobSpec, SwitchSpec, validate_knobs
from nam.capture.project import (
    AudioSettingsModel,
    CaptureProject,
    KnobModel,
    atomic_write_json,
    add_corner_captures,
    load_project,
    new_project,
    reconcile_with_disk,
    save_project,
)

DEFAULTS = {
    "mode": "normal",
    "epochs": 200,
    "batch_size": 4,
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
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


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
        return validate_knobs([control_from_dict(c) for c in self.draft_controls])

    def desired_train_input(self):
        if self.preferences["input_mode"] == "standard":
            return "input_standard.wav"
        relative = self.preferences.get("custom_input")
        if not relative or not (self.directory / relative).is_file():
            raise ValueError(
                "Choose a custom mono 48 kHz training WAV in the Controls step."
            )
        return relative

    def plan_changed(self):
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
        controls = self.controls()
        prefs = self.preferences
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
        replacement.sample_rate = self.project.sample_rate
        replacement.input_level_dbu = self.project.input_level_dbu
        replacement.output_level_dbu = self.project.output_level_dbu
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
        if not self.project.entries or not self.validation_matches():
            return False
        return all(
            e.status == "captured" and (self.directory / e.y_path).is_file()
            for e in self.project.entries
        )

    def fingerprint(self):
        """Reject checkpoints when captures, controls, calibration or training choices change."""
        from nam.capture.export import build_data_config

        data = {
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
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

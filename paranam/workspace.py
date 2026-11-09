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
    "epochs": 50,
    "batch_size": 16,
    "static_batch_size": 16,
    "device": "auto",
    "rounds": 2,
    "additions": 8,
    "ensemble_members": 2,
    "n_train": 32,
    "n_validation": 12,
    "corners": True,
    "seed": 0,
    "lhs_method": "standard",
    "input_mode": "standard",
    "custom_input": "",
}


def read_json(path: Path, default=None, *, tolerate_corrupt=False):
    path = Path(path)
    for attempt in range(10):
        try:
            from .recovery import validate_document, accept_valid

            data = validate_document(path, json.loads(path.read_text(encoding="utf-8")))
            accept_valid(path)
            return data
        except FileNotFoundError as error:
            has_backup = path.with_name(path.name + ".bak").is_file()
            has_checkpoint = (
                path.name == "progress.json"
                and (path.parent / "resume-0.ckpt").is_file()
            )
            if has_backup or has_checkpoint:
                from .recovery import recover_json

                return recover_json(
                    path, error, default, tolerate_corrupt=tolerate_corrupt
                )
            return default
        except (ValueError, UnicodeError) as error:
            from .recovery import recover_json

            return recover_json(path, error, default, tolerate_corrupt=tolerate_corrupt)
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.01)


def control_from_dict(data):
    if data.get("type") == "switch":
        return SwitchSpec(
            data["name"],
            tuple(data["enum_names"]),
            data.get("default", 0),
            data.get("planning_mode", "complex"),
        )
    return KnobSpec.from_dict(data)


class Workspace:
    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        read_json(self.directory / "capture_project.json")
        self.project = load_project(self.directory)
        stored = read_json(self.directory / "paranam.json", {})
        self.preferences = {**DEFAULTS, **stored}
        if "lhs_method" not in stored:
            self.preferences["lhs_method"] = self.project.lhs_method
        current_kind = stored.get("capture_type", self.project.capture_type)
        if "epochs" not in stored:
            self.preferences["epochs"] = 200 if current_kind == "static" else 50
        for kind, epochs in (("parametric", 50), ("static", 200)):
            self.preferences.setdefault(
                kind + "_epochs",
                stored.get("epochs", epochs) if kind == current_kind else epochs,
            )
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
        from nam.capture.output_gain import (
            volume_control,
            initialize_calibration,
            advance_calibration,
        )

        if (
            volume_control(self.project)
            and self.project.entries
            and self.project.output_gain_state.get("method") != "grid"
            and self.project.output_gain_state.get("status") != "ready"
        ):
            initialize_calibration(self.project)
            try:
                advance_calibration(self.project, self.directory)
            except ValueError as error:
                self.project.output_gain_state["message"] = str(error)
            save_project(self.project, self.directory)

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
        from .recipes import training_preset, preset_batch_size

        workspace.preferences.update(
            epochs=50,
            parametric_epochs=50,
            static_epochs=200,
            batch_size=preset_batch_size("HyperA2"),
            advanced_parametric=training_preset("HyperA2"),
        )
        workspace.save()
        return workspace

    def save(self):
        self.preferences["draft_controls"] = self.draft_controls
        atomic_write_json(self.directory / "paranam.json", self.preferences)
        save_project(self.project, self.directory)

    def capture_calibration(self):
        """Historical references used by this dataset, independent of Settings."""
        snapshot = dict(self.preferences.get("capture_calibration", {}))
        snapshot.update(
            input_level_dbu=self.project.input_level_dbu,
            output_level_dbu=self.project.output_level_dbu,
            capture_level_dbfs=self.project.capture_level_dbfs,
        )
        snapshot.setdefault(
            "instrument_level_dbu", self.preferences.get("instrument_level_dbu")
        )
        snapshot.setdefault(
            "calibration_frequency",
            self.preferences.get("calibration_frequency", 1000.0),
        )
        return snapshot

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

    def rename_control(self, old_name, new_name):
        """Change parameter keys without moving audio or modifying frozen jobs."""
        new_name = new_name.strip()
        draft = [
            {**c, "name": new_name} if c["name"] == old_name else dict(c)
            for c in self.draft_controls
        ]
        validate_knobs([control_from_dict(c) for c in draft])
        project = self.project.model_copy(deep=True)
        changed = False
        for control in project.knobs:
            if control.name == old_name:
                control.name = new_name
                changed = True
        if changed:
            validate_knobs(project.control_specs())
            for entry in project.entries:
                entry.params = {
                    new_name if k == old_name else k: v for k, v in entry.params.items()
                }
            if project.output_gain and project.output_gain["name"] == old_name:
                project.output_gain["name"] = new_name
            project.control_name_history.append({"from": old_name, "to": new_name})
        self.project, self.draft_controls = project, draft
        self._sync_control_configs({old_name: new_name})
        self.save()

    def move_control(self, index, destination):
        if not (
            0 <= index < len(self.draft_controls)
            and 0 <= destination < len(self.draft_controls)
        ):
            return
        control = self.draft_controls.pop(index)
        self.draft_controls.insert(destination, control)
        order = {c["name"]: i for i, c in enumerate(self.draft_controls)}
        self.project.knobs.sort(key=lambda c: order.get(c.name, len(order)))
        names = [c.name for c in self.project.control_specs()]
        for entry in self.project.entries:
            entry.params = {name: entry.params[name] for name in names}
        if self.project.output_gain:
            self.project.output_gain["index"] = names.index(
                self.project.output_gain["name"]
            )
        self._sync_control_configs()
        self.save()

    def _sync_control_configs(self, renamed=None):
        from nam.capture.export import build_data_config, _training_param_specs

        renamed = renamed or {}
        if (self.directory / "data.json").exists():
            atomic_write_json(
                self.directory / "data.json", build_data_config(self.project)
            )
        names = [p["name"] for p in _training_param_specs(self.project)]
        for filename in (
            "model_concat.json",
            "model_hyper.json",
            "model_hyper_a2.json",
        ):
            path = self.directory / filename
            if not path.is_file():
                continue
            config = read_json(path)
            params = config.get("net", {}).get("config", {}).get("params")
            if isinstance(params, list):
                for param in params:
                    param["name"] = renamed.get(param["name"], param["name"])
                params.sort(
                    key=lambda p: (
                        names.index(p["name"]) if p["name"] in names else len(names)
                    )
                )
                atomic_write_json(path, config)

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
        if self.preferences["input_mode"] == "trunc":
            from .signals import install_signals

            if not (self.directory / "input_trunc.wav").is_file():
                install_signals(self.directory, include_trunc=True)
            return "input_trunc.wav"
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
            or self.preferences["lhs_method"] != self.project.lhs_method
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
        switches_only = all(isinstance(c, SwitchSpec) for c in controls) and prefs[
            "n_train"
        ] == __import__("math").prod(len(c.enum_names) for c in controls)
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
            lhs_method=prefs["lhs_method"],
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
        from nam.capture.output_gain import initialize_calibration

        initialize_calibration(replacement)
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
        from nam.capture.output_gain import volume_control

        if (
            volume_control(self.project)
            and self.project.output_gain_state.get("status") != "ready"
        ):
            return False
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
            rate_preset="manual",
            cudnn_benchmark=True,
            deterministic_training=False,
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
        options.setdefault("hypernetwork_rate_preset", "follow")
        options.setdefault("hypernetwork_lr_decay", options["lr_decay"])
        hyper_a2 = options["hypernetwork_recipe"] == "HyperA2"
        options.setdefault("hypernetwork_mode", "full" if hyper_a2 else "low_rank")
        options.setdefault("hypernetwork_rank", 2)
        options.setdefault("condition_dilated_convolutions", hyper_a2)
        options.setdefault("branch_loss_reduction", "sum" if hyper_a2 else "mean")
        if options["deterministic_training"] is True:
            options["cudnn_benchmark"] = False
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
            **(
                {"output_gain": self.project.output_gain}
                if self.project.output_gain
                else {}
            ),
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
            if data["settings"]["advanced_training"].get("rate_preset") == "manual":
                data["settings"]["advanced_training"].pop("rate_preset")
            if (
                self.project.capture_type == "static"
                or data["settings"]["advanced_training"].get(
                    "hypernetwork_rate_preset", "follow"
                )
                == "follow"
            ):
                data["settings"]["advanced_training"].pop(
                    "hypernetwork_rate_preset", None
                )
                data["settings"]["advanced_training"].pop("hypernetwork_lr_decay", None)
            if data["settings"]["advanced_training"].get("cudnn_benchmark", True):
                data["settings"]["advanced_training"].pop("cudnn_benchmark", None)
            if not data["settings"]["advanced_training"].get(
                "deterministic_training", False
            ):
                data["settings"]["advanced_training"].pop(
                    "deterministic_training", None
                )
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


"""Resumable projects, non-destructive plan changes and application preferences."""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from pathlib import Path

from nam.capture.params import KnobSpec, SwitchSpec, validate_knobs
from nam.capture.plan_rules import PlanRule
from nam.capture.planner import CAPTURES_DIRNAME, rule_capture_limits
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
from nam.models.losses import DEFAULT_SEGMENT_ESR_MIN_RMS_DBFS, waveform_loss_settings

DEFAULTS = {
    "capture_level_dbfs": 0.0,
    "capture_type": "parametric",
    "mode": "normal",
    "epochs": 100,
    "batch_size": 16,
    "static_batch_size": 16,
    "device": "auto",
    "rounds": 2,
    "additions": 8,
    "ensemble_members": 2,
    "n_train": 32,
    "n_validation": 8,
    "corners": True,
    "include_default_as_additional_capture": True,
    "seed": 0,
    "lhs_method": "sobol",
    "validation_method": "independent",
    "input_mode": "trunc",
    "custom_input": "",
}


def recommended_epochs(kind: str, input_mode: str) -> int:
    """Suggested budget for a fresh run, not a mandatory training limit."""
    return 200 if kind == "static" else 100


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
            tuple(data["option_indices"]) if data.get("option_indices") is not None else None,
        )
    return KnobSpec.from_dict(data)


def legacy_avoid_rules(data):
    """Translate old per-knob Avoid exclusions to strict per-control Rules."""
    if data.get("type") == "switch":
        return []
    name = data["name"]
    rules = []
    for field, comparator in (("avoid_below", "<"), ("avoid_above", ">")):
        if data.get(field) is not None:
            rules.append(PlanRule(
                controls=(name,), operator=comparator,
                value=float(data[field]), kind="avoid",
            ))
    if data.get("avoid_zero") and data.get("avoid_below") is None:
        rules.append(PlanRule(
            controls=(name,), operator="=", value=float(data["min"]), kind="avoid",
        ))
    if data.get("avoid_max") and data.get("avoid_above") is None:
        rules.append(PlanRule(
            controls=(name,), operator="=", value=float(data["max"]), kind="avoid",
        ))
    return rules


def merge_plan_rules(existing, additions):
    rules = [PlanRule.model_validate(rule) for rule in existing]
    for rule in additions:
        if rule not in rules:
            rules.append(rule)
    return rules


def migrate_knob_limits(data):
    if data.get("type") == "switch":
        return data
    data = dict(data)
    for key in ("avoid_zero", "avoid_max", "avoid_below", "avoid_above"):
        data.pop(key, None)
    return control_from_dict(data).to_dict()


class Workspace:
    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        read_json(self.directory / "capture_project.json")
        self.project = load_project(self.directory)
        stored = read_json(self.directory / "paranam.json", {})
        self.preferences = {**DEFAULTS, **stored}
        if ("include_default_as_additional_capture" not in stored
                and self.project.include_default_as_additional_capture is not None):
            self.preferences["include_default_as_additional_capture"] = (
                self.project.include_default_as_additional_capture
            )
        if "lhs_method" not in stored:
            self.preferences["lhs_method"] = self.project.lhs_method
        if "validation_method" not in stored:
            self.preferences["validation_method"] = self.project.validation_method
        current_kind = stored.get("capture_type", self.project.capture_type)
        if "input_mode" not in stored and current_kind == "static":
            self.preferences["input_mode"] = "standard"
        for kind, default_mode in (("parametric", "trunc"), ("static", "standard")):
            key = kind + "_input_mode"
            if key not in stored:
                self.preferences[key] = (
                    self.preferences["input_mode"]
                    if kind == current_kind else default_mode
                )
        if "epochs" not in stored:
            self.preferences["epochs"] = (
                200 if current_kind == "static"
                else recommended_epochs("parametric", self.preferences["input_mode"])
            )
        for kind, epochs in (
            ("parametric", recommended_epochs("parametric", self.preferences["input_mode"])),
            ("static", 200),
        ):
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
        if self.preferences.get("static_captures") is None:
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
                n_validation=sum(
                    self.project.classification_for_entry(e) != "avoid"
                    for e in self.project.entries_for_split("validation")
                ),
                seed=self.project.seed,
                corners=self.project.include_initial_corners,
            )
        if not stored and self.project.entries:
            self.preferences.update(
                input_mode="custom", custom_input=self.project.train_input
            )
        # Preserve frozen jobs byte-for-byte: their existing fingerprints must
        # still validate. Editable projects and Retrain drafts migrate legacy
        # knob exclusions to Rules without moving any existing recordings.
        # Retrain provenance survives enqueueing. The new job owns a spec
        # next to its project; an editable Retrain copy has no spec there.
        has_job_spec = (self.directory.name == "project"
                        and (self.directory.parent / "spec.json").is_file())
        frozen = ("_queue_source_project" in stored
                  and ("_retrain_source_job" not in stored or has_job_spec))
        raw_project_controls = [c.model_dump() for c in self.project.knobs]
        raw_drafts = self.preferences.get("draft_controls", raw_project_controls)
        if frozen:
            self.draft_controls = [control_from_dict(c).to_dict() for c in raw_drafts]
        else:
            committed = [
                rule for c in raw_project_controls for rule in legacy_avoid_rules(c)
            ]
            draft_additions = [
                rule for c in raw_drafts for rule in legacy_avoid_rules(c)
            ]
            self.project.plan_rules = merge_plan_rules(self.project.plan_rules, committed)
            stored_rules = self.preferences.get(
                "draft_plan_rules",
                [rule.model_dump(mode="json") for rule in self.project.plan_rules],
            )
            self.preferences["draft_plan_rules"] = [
                rule.model_dump(mode="json")
                for rule in merge_plan_rules(stored_rules, draft_additions)
            ]
            self.draft_controls = [migrate_knob_limits(c) for c in raw_drafts]
            self.project.knobs = [
                KnobModel.from_knob_spec(control_from_dict(migrate_knob_limits(c)))
                for c in raw_project_controls
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
            lhs_method="sobol",
        )
        project.train_window.ny = project.validation_window.ny = 8192
        from .signals import install_signals

        install_signals(directory, include_trunc=True)
        project.train_input = "input_trunc.wav"
        save_project(project, directory)
        workspace = cls(directory)
        from .recipes import training_preset, preset_batch_size

        from nam.capture.params import capture_recommendations

        suggested_train, suggested_val, _ = capture_recommendations(
            controls, include_default_as_additional_capture=True,
            lhs_method="sobol",
        )
        workspace.preferences.update(
            n_train=suggested_train,
            n_validation=suggested_val,
            lhs_method="sobol",
            epochs=100,
            parametric_epochs=100,
            static_epochs=200,
            batch_size=preset_batch_size("HyperA2"),
            advanced_parametric={**training_preset("HyperA2"), "waveform_loss": "segment_esr"},
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

    def plan_rules(self):
        return [
            PlanRule.model_validate(rule)
            for rule in self.preferences.get("draft_plan_rules", self.project.plan_rules)
        ]

    def rename_rule_control(self, old_name, new_name):
        def rename(rule):
            return rule.model_copy(update={
                "controls": tuple(
                    new_name if name == old_name else name for name in rule.controls
                ),
                "operators": {
                    new_name if name == old_name else name: op
                    for name, op in rule.operators.items()
                },
                "thresholds": {
                    new_name if name == old_name else name: value
                    for name, value in rule.thresholds.items()
                },
            })

        self.preferences["draft_plan_rules"] = [
            rename(rule).model_dump(mode="json") for rule in self.plan_rules()
        ]
        self.project.plan_rules = [rename(rule) for rule in self.project.plan_rules]

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
        self.rename_rule_control(old_name, new_name)
        for record in self.project.unused_recordings.values():
            params = record["entry"]["params"]
            if old_name in params and new_name not in params:
                params[new_name] = params.pop(old_name)
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

    def update_switch_display_order(self, control):
        """Move switch positions without changing existing neural coordinates."""
        for project_control in self.project.knobs:
            if project_control.name != control.name:
                continue
            original = project_control.to_knob_spec()
            if not isinstance(original, SwitchSpec):
                return
            without_order = lambda spec: {
                key: value for key, value in spec.to_dict().items() if key != "option_indices"
            }
            if (
                original.option_indices != control.option_indices
                and without_order(original) == without_order(control)
            ):
                project_control.option_indices = (
                    None if control.option_indices is None else list(control.option_indices)
                )
                self._sync_control_configs(switches={control.name: control})
            return

    def _sync_control_configs(self, renamed=None, switches=None):
        from nam.capture.export import build_data_config, _training_param_specs

        renamed = renamed or {}
        switches = switches or {}
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
                    switch = switches.get(param["name"])
                    if switch is not None and "options" in param:
                        positions = switch.option_indices or range(len(switch.enum_names))
                        for option in param["options"]:
                            option["index_sw"] = positions[option["value"]]
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

    def remove_static_capture(self, capture_id, *, delete_wavs=False):
        """Remove a named static take and optionally its processed train/validation WAVs."""
        from nam.capture.export import update_data_json

        project = self.project
        if project.capture_type != "static" or not capture_id:
            raise ValueError("Select a named static capture first.")
        removed = [entry for entry in project.entries if entry.capture_id == capture_id]
        if not removed:
            raise ValueError("The selected static capture is no longer in the list.")

        project.entries = [entry for entry in project.entries if entry.capture_id != capture_id]
        for index, entry in enumerate(project.entries):
            entry.index = index
        self.preferences["static_captures"] = [
            capture for capture in self.preferences["static_captures"]
            if capture["id"] != capture_id
        ]
        project.n_train_lhs = len(self.static_groups())
        if not delete_wavs:
            for entry in removed:
                if (self.directory / entry.y_path).is_file():
                    project.unused_recordings.setdefault(entry.y_path, {
                        "entry": entry.model_dump(mode="json"), "context": None,
                    })
        self.save()
        update_data_json(project, self.directory)

        if not delete_wavs:
            return []
        root = self.directory / CAPTURES_DIRNAME
        referenced = {project.train_input, project.validation_input, *project.unused_recordings}
        referenced.update(entry.y_path for entry in project.entries)
        protected = {(self.directory / name).resolve() for name in referenced}
        warnings = []
        for name in dict.fromkeys(entry.y_path for entry in removed):
            path = self.directory / name
            try:
                target = path.resolve()
                if (
                    path.is_symlink()
                    or not target.is_relative_to(root)
                    or path.suffix.lower() != ".wav"
                ):
                    warnings.append(f"{name}: expected a WAV in the project capture folder.")
                elif target in protected:
                    warnings.append(f"{name} is still used by the project.")
                else:
                    path.unlink(missing_ok=True)
            except OSError as error:
                warnings.append(f"{name}: {error}")
        return warnings

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
            or self.plan_rules() != self.project.plan_rules
            or self.preferences["n_train"] != self.project.n_train_lhs
            or self.preferences["n_validation"]
            != sum(
                self.project.classification_for_entry(e) != "avoid"
                for e in self.project.entries_for_split("validation")
            )
            or self.preferences["seed"] != self.project.seed
            or self.preferences["lhs_method"] != self.project.lhs_method
            or self.preferences["validation_method"] != self.project.validation_method
            or self.preferences["corners"] != self.project.include_initial_corners
            or (
                (
                    self.project.include_default_as_additional_capture is not None
                    or self.preferences.get("_include_default_capture_touched", False)
                )
                and self.preferences["include_default_as_additional_capture"]
                != self.project.include_default_as_additional_capture
            )
        )

    @staticmethod
    def _recording_context(project, split, *, validation_matches):
        if split != "train" and not validation_matches:
            return None
        return [
            project.capture_type, project.input_for_split(split),
            project.capture_level_dbfs, project.input_level_dbu,
            project.output_level_dbu, project.sample_rate,
        ]

    def _reuse_plan_recordings(self, replacement):
        previous = self.project
        validation_matches = self.validation_matches() if previous.captured_entries() else True
        pool = dict(previous.unused_recordings)
        for entry in previous.captured_entries():
            context = self._recording_context(previous, entry.split, validation_matches=validation_matches)
            if (self.directory / entry.y_path).is_file():
                pool[entry.y_path] = {
                    "entry": entry.model_dump(mode="json"), "context": context,
                }

        def key(entry, context):
            return (tuple(context), entry.split, entry.capture_id, tuple(sorted(entry.params.items())))

        available = {}
        for path, record in pool.items():
            if record["context"] is not None and (self.directory / path).is_file():
                entry = CaptureEntryModel.model_validate(record["entry"])
                if entry.validation_from_tail and not validation_matches:
                    continue
                signature = key(entry, record["context"])
                prior = available.get(signature)
                if prior is None or (entry.captured_at or "") >= (prior.captured_at or ""):
                    available[signature] = entry

        from nam.capture.session import raw_paths

        reserved = set(pool) | {
            entry.y_path for entry in replacement.entries
            if replacement.classification_for_entry(entry) is not None
        }
        for entry in replacement.entries:
            if replacement.classification_for_entry(entry) is not None:
                # Classified rows remain visible, but their old recordings
                # must not be reused as training data.
                pool.pop(entry.y_path, None)
                continue
            context = self._recording_context(replacement, entry.split, validation_matches=True)
            recorded = available.pop(key(entry, context), None)
            if recorded is not None:
                if entry.split == "train" and entry.is_corner is None:
                    entry.is_corner = False
                for field in ("y_path", "status", "delay", "captured_at", "qa",
                              "stream_config", "validation_from_tail"):
                    setattr(entry, field, getattr(recorded, field))
                if entry.split == "gain":
                    entry.user_metadata = dict(recorded.user_metadata)
                del pool[entry.y_path]
                continue
            path = Path(entry.y_path)
            suffix = 1
            while (entry.y_path in reserved
                   or (self.directory / entry.y_path).exists()
                   or any(raw.exists() for raw in raw_paths(self.directory, entry.y_path))):
                entry.y_path = path.with_stem(f"{path.stem}-take{suffix}").as_posix()
                suffix += 1
            reserved.add(entry.y_path)
        replacement.unused_recordings = pool
        replacement.alignment_reference = previous.alignment_reference
        replacement.alignment_lead_samples = previous.alignment_lead_samples
        replacement.created_with_version = previous.created_with_version
        replacement.control_name_history = list(previous.control_name_history)

        old_gain = {e.y_path for e in previous.entries if e.split == "gain"}
        new_gain = {e.y_path for e in replacement.entries if e.split == "gain"}
        if (old_gain and new_gain == old_gain
                and all(e.status == "captured" for e in replacement.entries if e.split == "gain")):
            replacement.output_gain = previous.output_gain
            replacement.output_gain_state = dict(previous.output_gain_state)
            if replacement.output_gain:
                replacement.output_gain = dict(replacement.output_gain)
                replacement.output_gain["index"] = next(
                    i for i, c in enumerate(replacement.knobs)
                    if c.name == replacement.output_gain["name"]
                )
        elif any(e.split == "gain" and e.status == "captured" for e in replacement.entries):
            from nam.capture.output_gain import advance_calibration

            try:
                advance_calibration(replacement, self.directory)
            except ValueError as error:
                replacement.output_gain_state.update(status="failed", message=str(error))

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
            self._reuse_plan_recordings(replacement)
            return self._install_plan(replacement)
        rules = self.plan_rules()
        synthetic_mode = True
        if synthetic_mode:
            # Keep legacy Avoid exclusions, but admit marked Mute endpoints and
            # Mute-rule regions to the ordinary Sobol/validation/corner sampler.
            planning_controls = list(controls)
            planning_rules = [rule for rule in rules if rule.kind == "avoid"]
        else:
            planning_controls, planning_rules = controls, rules
        all_switches = all(isinstance(c, SwitchSpec) for c in controls)
        switches_only = all_switches and (
            bool(planning_rules) or prefs["n_train"] == __import__("math").prod(len(c.enum_names) for c in controls)
        )
        if planning_rules and switches_only:
            states, _, _ = rule_capture_limits(controls, planning_rules)
            if prefs["n_train"] != states:
                raise ValueError(f"Capture all {states} allowed switch states using each signal.")
        validation_count = prefs["n_validation"]
        if validation_count < 0 or (switches_only and validation_count < 1):
            raise ValueError(
                "For switch-only captures, validate every state once with the separate signal."
            )
        replacement = new_project(
            planning_controls,
            n_train=prefs["n_train"],
            n_validation=0 if switches_only else validation_count,
            seed=prefs["seed"],
            lhs_method=prefs["lhs_method"],
            validation_method=prefs["validation_method"],
            name=self.project.name,
            train_input=self.desired_train_input(),
            validation_input="input_validation.wav",
            plan_rules=planning_rules,
            include_default_as_additional_capture=(
                prefs["include_default_as_additional_capture"]
                if not all_switches else None
            ),
        )
        prefs.pop("_include_default_capture_touched", None)
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
        if synthetic_mode:
            from .synthetic_mute import (
                ensure_mute_endpoint_corners, ensure_mute_rule_corners,
            )
            # Mute endpoints need explicit zero targets even when optional
            # physical Corner captures are disabled.
            ensure_mute_endpoint_corners(replacement, rules)
            ensure_mute_rule_corners(replacement, rules)
        if synthetic_mode:
            # The temporary planning controls only relax Mute constraints.
            # The saved project keeps the actual user-defined controls/rules.
            from nam.capture.project import KnobModel
            replacement.knobs = [KnobModel.from_knob_spec(c) for c in controls]
            replacement.plan_rules = list(rules)
        # Retain discovered Mute classifications only at settings selected by
        # the newly generated plan. A shifted, obsolete Mute corner must not
        # silently be appended to the Capture list or training dataset.
        # Keep Avoid history so unstable settings remain excluded.
        from nam.capture.project import classify_capture
        def key(p):
            return tuple(sorted(p.items()))
        planned = {
            key(row.params) for row in replacement.entries
            if row.split in ("train", "validation")
        }
        old_muted = [
            dict(p) for p in self.project.muted_settings if key(p) in planned
        ]
        old_avoided = [dict(p) for p in self.project.avoided_settings]
        marked = {
            **{key(p): "mute" for p in old_muted},
            **{key(p): "avoid" for p in old_avoided},
        }
        # A classified row is no longer removed by classify_capture. Skip
        # already marked rows to avoid re-classifying them indefinitely.
        for _ in range(max(32, len(replacement.entries) * 4)):
            entry = next(
                (
                    row for row in replacement.entries
                    if key(row.params) in marked
                    and not (synthetic_mode and marked[key(row.params)] == "mute")
                    and replacement.classification_for_entry(row) is None
                ),
                None,
            )
            if entry is None:
                break
            replacement, _ = classify_capture(
                replacement, entry, marked[key(entry.params)]
            )
        else:
            raise ValueError(
                "Unable to avoid all previously marked mute/unstable combinations"
            )
        present_muted = {key(p) for p in replacement.muted_settings}
        present_avoided = {key(p) for p in replacement.avoided_settings}
        replacement.muted_settings.extend(
            p for p in old_muted if key(p) not in present_muted
        )
        replacement.avoided_settings.extend(
            p for p in old_avoided if key(p) not in present_avoided
        )
        # Keep previously marked entries accessible even if the new sampler
        # doesn't happen to choose those exact settings. Preserve their old WAV
        # and QA if a new row can use the same filename.
        names = {c.name for c in controls}
        for old in self.project.entries:
            kind = self.project.classification_for_entry(old)
            if kind is None or kind == "mute":
                continue  # Only freshly planned points become synthetic Mute.
            if set(old.params) != names:
                continue  # No longer the same rig; the old point stays annotated.
            match = next(
                (
                    row for row in replacement.entries
                    if row.split == old.split and row.params == old.params
                ),
                None,
            )
            if match is None:
                clone = old.model_copy(deep=True)
                if old.split == "train":
                    at = next(
                        (i for i, row in enumerate(replacement.entries)
                         if row.split == "validation"),
                        len(replacement.entries),
                    )
                    replacement.entries.insert(at, clone)
                else:
                    replacement.entries.append(clone)
            else:
                # A sampled classified point retains its original recording
                # whenever doing so does not duplicate another row's path.
                if not any(
                    row is not match and row.y_path == old.y_path
                    for row in replacement.entries
                ):
                    for field in (
                        "y_path", "status", "delay", "captured_at",
                        "qa", "stream_config", "is_corner",
                    ):
                        setattr(match, field, getattr(old, field))
        from nam.capture.output_gain import initialize_calibration

        initialize_calibration(replacement)
        self._reuse_plan_recordings(replacement)
        return self._install_plan(replacement)

    def _install_plan(self, replacement):
        from nam.capture.export import update_data_json
        from .signals import install_signals

        install_signals(self.directory)
        self.project = replacement
        synthetic = replacement.capture_type == "parametric"
        if synthetic:
            from .synthetic_mute import materialize_synthetic_mutes
            materialize_synthetic_mutes(self)
        self.preferences.pop("pending_calibration", None)
        self.save()
        update_data_json(replacement, self.directory, include_synthetic=synthetic)
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

    def training_capture_entries(self):
        entries = self.project.entries
        if self.project.capture_type != "static":
            return [
                e for e in entries
                if self.project.classification_for_entry(e) is None
                or e.synthetic_mute
            ]
        imported = {
            e.capture_id for e in entries
            if e.split == "train" and e.status == "captured" and e.validation_from_tail
        }
        return [
            e for e in entries if e.split != "validation" or e.capture_id not in imported
        ]

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
        required = self.training_capture_entries()
        if not any(e.split == "train" for e in required):
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
            for e in required
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
            checkpoint_method="validation",
            checkpoint_alpha=0.75,
            learning_rate=0.004,
            lr_decay=0.994,
            ny=self.project.train_window.ny or 8192,
            fit_mrstft=True,
            # Parametric models benefit from capture-level RMS normalization;
            # static models default to ordinary unnormalized training loss.
            # Explicit saved settings always override the type-specific default.
            rms_loss_normalization=(kind == "parametric"),
            rms_loss_normalization_exponent=2.0,
            waveform_loss="mse",
            segment_esr_min_rms_dbfs=DEFAULT_SEGMENT_ESR_MIN_RMS_DBFS,
            snr_weighted_loss_normalization=False,
            mrstft_weight=0.0005,
            weight_decay=3.17e-7,
            gradient_clipping=(kind == "parametric"),
            gradient_clip_val=1.0,
            check_nonfinite_gradients=(kind == "parametric"),
            hypernetwork_recipe="HyperA2",
            synthetic_mute_reference_rms=False,
        )
        options = {**defaults, **self.preferences.get("advanced_" + kind, {})}
        # Retire the old beta mute objectives without breaking older projects.
        options.pop("mute_training_mode", None)
        for legacy_key in ("mute_anchor_interval", "mute_anchor_batch_size", "mute_anchor_ny"):
            options.pop(legacy_key, None)
        # Older projects could set exponent=0 to keep only SNR attenuation.
        # The new independent switches represent that exact loss as RMS off,
        # leaving SNR as configured. Migrate the effective options, not the
        # original project data or any existing frozen training job.
        if options["rms_loss_normalization_exponent"] == 0:
            options["rms_loss_normalization"] = False
            options["rms_loss_normalization_exponent"] = 0.01
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

    def legacy_knob_fingerprint(self, spec, batch_size=None):
        """Verify a frozen job whose old Avoid limits were already saved as Rules."""
        prefs = spec.get("preferences", {})
        if not spec.get("queue_snapshot") or not prefs.get("_retrain_source_job"):
            return None
        saved = prefs.get("draft_controls", [])
        limits = [rule for control in saved for rule in legacy_avoid_rules(control)]
        if not limits or self.project.plan_rules != merge_plan_rules(
            prefs.get("draft_plan_rules", []), limits
        ):
            return None
        current = [c.to_dict() for c in self.project.control_specs()]
        if current != [migrate_knob_limits(c) for c in saved]:
            return None
        # Change only the representation of the equivalent controls for this
        # comparison. Hash the actual files, windows, settings and metadata.
        old = copy.copy(self)
        old.project = self.project.model_copy(update={
            "knobs": [KnobModel.from_knob_spec(control_from_dict(c)) for c in saved]
        })
        return old.fingerprint(batch_size=batch_size)

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
        from nam.capture.export import (
            build_data_config, mute_anchor_targets, mute_anchor_point_targets,
            mute_anchor_rule_targets,
        )

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
            e.y_path for e in self.training_capture_entries()
        ]:
            path = self.directory / relative
            stat = path.stat()
            data["files"].append([relative, stat.st_size, stat.st_mtime_ns])
        data["settings"]["batch_size"] = self.training_batch_size()
        if include_advanced:
            options = self.training_options()
            data["settings"]["advanced_training"] = options
            if not options["synthetic_mute_reference_rms"]:
                options.pop("synthetic_mute_reference_rms", None)
            static = self.project.capture_type == "static"
            segment_esr = not static and options["waveform_loss"] == "segment_esr"
            if segment_esr:
                _, threshold = waveform_loss_settings(options)
                options["segment_esr_min_rms_dbfs"] = threshold
                # These switches are retained in the editor for switching back to MSE.
                options.pop("synthetic_mute_reference_rms", None)
            else:
                # Preserve fingerprints of existing MSE jobs and presets.
                options.pop("waveform_loss")
                options.pop("segment_esr_min_rms_dbfs")
            rms_enabled = options["rms_loss_normalization"]
            if static or segment_esr or not rms_enabled or options["rms_loss_normalization_exponent"] == 2.0:
                options.pop("rms_loss_normalization_exponent")
            # SNR is independent of RMS: preserve its setting even when RMS is off.
            if static or not options["snr_weighted_loss_normalization"]:
                options.pop("snr_weighted_loss_normalization")
            if static or segment_esr or rms_enabled:
                options.pop("rms_loss_normalization")
            method = data["settings"]["advanced_training"].get("checkpoint_method")
            # Preserve legacy fingerprints when only the new default options
            # have been added to the settings dialog.
            if method == "validation" or self.project.capture_type == "static":
                data["settings"]["advanced_training"].pop("checkpoint_method", None)
            if method != "combined" or self.project.capture_type == "static":
                data["settings"]["advanced_training"].pop("checkpoint_alpha", None)
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


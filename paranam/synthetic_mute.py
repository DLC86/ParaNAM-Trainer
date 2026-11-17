"""Normal dataset recordings for Mute settings, never physical captures.

Synthetic Mute Captures is the sole mute method; it runs as part of plan creation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def is_mute_setting(project, params):
    if params in project.avoided_settings:
        return False
    if params in project.muted_settings:
        return True
    for knob in project.knob_specs():
        if (
            getattr(knob, "mute_at_min", False)
            and params[knob.name] == knob.min
        ) or (
            getattr(knob, "mute_at_max", False)
            and params[knob.name] == knob.max
        ):
            return True
    return any(
        rule.kind == "mute" and rule.excludes(params)
        for rule in project.plan_rules
    )


def _shared_target(root, split, source):
    """Generate one silent reference per split, only if needed."""
    import os
    import tempfile

    info = sf.info(str(source))
    if info.channels != 1:
        raise ValueError(f"Expected mono DI for synthetic Mute: {source}")
    relative = f"synthetic_mutes/{split}.wav"
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        old = sf.info(str(destination))
        if (old.frames == info.frames and old.samplerate == info.samplerate
                and old.channels == 1):
            with sf.SoundFile(str(destination)) as wav:
                if all(not np.any(chunk) for chunk in wav.blocks(
                    blocksize=65536, dtype="float32"
                )):
                    return relative
    fd, temporary = tempfile.mkstemp(
        prefix=f".{split}-", suffix=".wav", dir=destination.parent
    )
    os.close(fd)
    try:
        zeros = np.zeros(65536, dtype=np.float32)
        with sf.SoundFile(temporary, mode="w", samplerate=info.samplerate,
                          channels=1, format="WAV", subtype="PCM_24") as out:
            remaining = info.frames
            while remaining:
                count = min(remaining, len(zeros))
                out.write(zeros[:count])
                remaining -= count
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return relative


def materialize_synthetic_mutes(workspace):
    """Mark every selected Mute setting as completed using shared full-length zero WAVs."""
    project, root = workspace.project, Path(workspace.directory)
    if project.capture_type != "parametric":
        return 0
    grouped = {}
    for row in project.entries:
        if row.split in ("train", "validation") and is_mute_setting(project, row.params):
            grouped.setdefault(row.split, []).append(row)
    created = 0
    shared_train = None
    if "train" in grouped:
        shared_train = _shared_target(
            root, "train", root / project.input_for_split("train")
        )
    for split, rows in grouped.items():
        reference = shared_train if split == "train" else None
        if split == "validation" and shared_train is not None:
            training = sf.info(str(root / project.input_for_split("train")))
            validation = sf.info(str(root / project.input_for_split("validation")))
            if (training.frames == validation.frames
                    and training.samplerate == validation.samplerate):
                reference = shared_train
        if reference is None:
            reference = _shared_target(
                root, split, root / project.input_for_split(split)
            )
        for row in rows:
            if row.synthetic_mute and row.y_path == reference and row.status == "captured":
                continue
            row.y_path = reference
            row.status = "captured"
            row.delay = 0
            row.synthetic_mute = True
            row.qa = None
            row.captured_at = None
            created += 1
    return created


def ensure_mute_endpoint_corners(project, rules):
    """Restore exact muted endpoints where old Avoid thresholds shifted corners.

    A single-control Avoid threshold at an explicitly muted endpoint does not
    suppress its silent synthetic corner. Other Avoid rules still apply.
    """
    from nam.capture.planner import mute_corner_settings
    from nam.capture.project import CaptureEntryModel

    knobs = project.knob_specs()
    by_name = {knob.name: knob for knob in knobs}
    planned = {tuple(sorted(row.params.items())) for row in project.entries}
    index = max(
        (row.index for row in project.entries if row.split == "train"), default=-1
    ) + 1
    added = []
    for params in mute_corner_settings(knobs):
        key = tuple(sorted(params.items()))
        if key in planned:
            continue
        for rule in rules:
            if rule.kind != "avoid" or not rule.excludes(params):
                continue
            name = rule.controls[0] if len(rule.controls) == 1 else None
            knob = by_name.get(name)
            comparator = rule.condition_for(name)[0] if knob is not None else None
            exempt = (
                knob is not None and (
                    (getattr(knob, "mute_at_min", False)
                     and params[name] == knob.min
                     and comparator in ("<", "≤", "="))
                    or
                    (getattr(knob, "mute_at_max", False)
                     and params[name] == knob.max
                     and comparator in (">", "≥", "="))
                )
            )
            if not exempt:
                break
        else:
            added.append(CaptureEntryModel(
                index=index, split="train", params=params,
                y_path=f"captures/corner_mute_endpoint_{index:03d}.wav",
                is_corner=True,
            ))
            planned.add(key)
            index += 1
    if added:
        position = next(
            (i for i, row in enumerate(project.entries)
             if row.split == "validation"), len(project.entries),
        )
        project.entries[position:position] = added
    return added


def ensure_mute_rule_corners(project, rules):
    """Reserve a real dataset row for each uncovered conjunctive Mute rule.

    The extra row is a Corner capture in the training split; its WAV is later
    automatically generated as silence. An existing train Sobol/corner row
    satisfying the same rule is sufficient, so no unnecessary rows are added.
    """
    from itertools import product

    from nam.capture.plan_rules import allows
    from nam.capture.project import CaptureEntryModel

    controls = project.control_specs()
    by_name = {control.name: control for control in controls}
    avoid_rules = [rule for rule in rules if rule.kind == "avoid"]
    occupied = {tuple(row.params[name] for name in by_name)
                for row in project.entries}
    next_index = max((row.index for row in project.entries if row.split == "train"), default=-1) + 1
    extra = []
    for rule_index, rule in enumerate(rules):
        if rule.kind != "mute":
            continue
        # A valid training vertex already sampled by LHS is a Corner, not an
        # additional capture. Otherwise reserve a new min/max vertex below.
        for row in project.entries:
            if (row.split == "train" and rule.excludes(row.params)
                    and allows(avoid_rules, row.params)
                    and all(
                        row.params[c.name] in c.enum_names
                        if hasattr(c, "enum_names")
                        else row.params[c.name] in (c.min, c.max)
                        for c in controls
                    )):
                row.is_corner = True
                break
        if any(row.split == "train" and row.is_corner
               and rule.excludes(row.params) for row in project.entries):
            continue
        fixed = {}
        for name in rule.controls:
            control = by_name[name]
            operator, threshold = rule.condition_for(name)
            if operator == "=":
                fixed[name] = threshold
            elif operator in ("<", "≤"):
                fixed[name] = control.min
            elif operator in (">", "≥"):
                fixed[name] = control.max
        free = [control for control in controls if control.name not in fixed]
        # Prefer other controls at their maxima so a multi-Gain Mute corner
        # doesn't accidentally involve unrelated mute-at-min endpoints.
        axes = []
        for control in free:
            if hasattr(control, "enum_names"):
                default = control.enum_names[control.default]
                axes.append(tuple(control.enum_names))
            else:
                axes.append((control.max, control.min))
        candidate = None
        for values in product(*axes):
            params = dict(fixed)
            params.update(zip((c.name for c in free), values))
            params = {name: params[name] for name in by_name}
            key = tuple(params.values())
            if (key not in occupied and allows(avoid_rules, params)
                    and rule.excludes(params)):
                candidate = params
                break
        if candidate is None:
            raise ValueError(
                f"Cannot reserve an extra mute corner for Mute rule {rule_index + 1}: "
                "all matching settings are occupied or excluded by Avoid rules"
            )
        occupied.add(tuple(candidate.values()))
        extra.append(CaptureEntryModel(
            index=next_index,
            split="train",
            params=candidate,
            y_path=f"captures/corner_mute_rule_{rule_index + 1:03d}.wav",
            is_corner=True,
        ))
        next_index += 1
    if extra:
        start = next((i for i, row in enumerate(project.entries)
                      if row.split == "validation"), len(project.entries))
        project.entries[start:start] = extra
    return extra

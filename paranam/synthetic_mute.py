"""Normal dataset recordings for Mute settings, never physical captures.

Only the opt-in Synthetic Mute Captures recipe calls this module. The ordinary
capture planner and its silence-anchor behavior remain unchanged otherwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def is_mute_setting(project, params):
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


def materialize_synthetic_mutes(workspace):
    """Make selected Mute points completed, full-length, zero-target captures.

    Create a fresh path instead of overwriting a previously recorded WAV. A
    pair of real and synthetic split inputs use their respective exact lengths.
    """
    project, root = workspace.project, Path(workspace.directory)
    if project.capture_type != "parametric":
        return 0
    reserved = {row.y_path for row in project.entries}
    created = 0
    for index, row in enumerate(project.entries):
        if row.split not in ("train", "validation") or not is_mute_setting(project, row.params):
            continue
        if row.synthetic_mute and (root / row.y_path).is_file():
            continue
        basename = f"synthetic-mute-{index:05d}-{row.split}"
        suffix = 0
        while True:
            relative = f"captures/{basename}{'-' + str(suffix) if suffix else ''}.wav"
            if relative not in reserved and not (root / relative).exists():
                break
            suffix += 1
        reserved.add(relative)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = root / project.input_for_split(row.split)
        info = sf.info(str(source))
        if info.channels != 1:
            raise ValueError(f"Expected mono DI for synthetic mute capture: {source}")
        # Stream the target in fixed blocks rather than allocating a potentially
        # multi-minute WAV's worth of zeros in memory.
        with sf.SoundFile(str(target), mode="w", samplerate=info.samplerate,
                          channels=1, format="WAV", subtype="PCM_24") as output:
            remaining = info.frames
            silence = np.zeros(65536, dtype=np.float32)
            while remaining:
                count = min(remaining, len(silence))
                output.write(silence[:count])
                remaining -= count
        row.y_path = relative
        row.status = "captured"
        row.delay = 0
        row.synthetic_mute = True
        row.qa = None
        row.captured_at = None
        created += 1
    return created


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
    occupied = {tuple(row.params[name] for name in by_name) for row in project.entries}
    next_index = max((row.index for row in project.entries if row.split == "train"), default=-1) + 1
    extra = []
    for rule_index, rule in enumerate(rules):
        if rule.kind != "mute" or any(
            row.split == "train" and rule.excludes(row.params)
            for row in project.entries
        ):
            continue
        fixed = {}
        for name in rule.controls:
            control = by_name[name]
            operator, threshold = rule.condition_for(name)
            if operator == "=":
                fixed[name] = threshold
            elif operator == "<":
                fixed[name] = control.min
            elif operator == ">":
                fixed[name] = control.max
        free = [control for control in controls if control.name not in fixed]
        # Prefer other controls at their maxima so a multi-Gain Mute corner
        # doesn't accidentally involve unrelated mute-at-min endpoints.
        axes = []
        for control in free:
            if hasattr(control, "enum_names"):
                default = control.enum_names[control.default]
                axes.append(tuple(dict.fromkeys(
                    (default, *control.enum_names)
                )))
            else:
                axes.append(tuple(dict.fromkeys(
                    (control.max, control.min, control.default)
                )))
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

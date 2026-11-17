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

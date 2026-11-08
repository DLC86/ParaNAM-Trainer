"""Measure an output attenuator separately from the neural effect.

All neural captures use the maximum physical volume. Short calibration captures
fit y(v)=gain(v)*y(max), check waveform residuals and a second knob context, then
fit a monotone curve to the fixed knob-step grid. Classification is advisory.
"""

from dataclasses import replace
from pathlib import Path
import math

import numpy as np

from .params import SwitchSpec
from .project import CaptureEntryModel
from .planner import _planned_from_params, _quantize_to_capture_grid

MAX_POINTS = 65


def volume_control(project):
    return next(
        (c for c in project.control_specs() if getattr(c, "clean_volume", False)), None
    )


def _snap(control, value):
    spec = replace(control.to_param_spec(), avoid_zero=False, avoid_max=False)
    return float(
        _quantize_to_capture_grid(
            np.array([value]), (spec,), default_step=control.step
        )[0]
    )


def _context(project, defaults=False):
    result = {}
    for c in project.control_specs():
        if isinstance(c, SwitchSpec):
            result[c.name] = c.enum_names[c.default if defaults else -1]
        else:
            value = c.default if defaults else c.max
            result[c.name] = float(
                _quantize_to_capture_grid(
                    np.array([value]), (c.to_param_spec(),), default_step=c.step
                )[0]
            )
    return result


def _append(project, positions, context, phase):
    volume = volume_control(project)
    specs = tuple(c.to_param_spec() for c in project.control_specs())
    existing = {
        tuple(e.params[c.name] for c in project.control_specs())
        for e in project.entries
        if e.split == "gain"
    }
    rows = []
    for position in positions:
        values = {**context, volume.name: position}
        key = tuple(values[c.name] for c in project.control_specs())
        if key not in existing:
            rows.append(values)
            existing.add(key)
    first = sum(e.split == "gain" for e in project.entries)
    index = max((e.index for e in project.entries), default=-1) + 1
    planned = _planned_from_params(
        "gain", "gain", specs, rows, index_offset=index, filename_start=first
    )
    for p in planned:
        insert_at = next(
            (i for i, e in enumerate(project.entries) if e.split != "gain"),
            len(project.entries),
        )
        project.entries.insert(
            insert_at,
            CaptureEntryModel(
                index=p.index,
                split="gain",
                params=p.params,
                y_path=p.y_path,
                user_metadata={
                    "output_gain": {
                        "phase": phase,
                        "context": "max" if context == _context(project) else "check",
                    }
                },
            ),
        )
    return len(planned)


def initialize_calibration(project):
    """A fixed knob-step grid; four screening captures precede the remaining points."""
    from .planner import _capture_grid_values

    control = volume_control(project)
    if control is None:
        return
    grid = _capture_grid_values(
        replace(control.to_param_spec(), avoid_zero=False, avoid_max=False),
        control.step,
    )
    midpoint = _snap(control, (control.min + control.max) / 2)
    high = _context(project)
    alternate = _context(project, defaults=True)
    if all(
        alternate[c.name] == high[c.name]
        for c in project.control_specs()
        if c.name != control.name
    ):
        for c in project.control_specs():
            if c.name != control.name:
                alternate[c.name] = (
                    c.enum_names[0]
                    if isinstance(c, SwitchSpec)
                    else float(
                        _quantize_to_capture_grid(
                            np.array([c.min]), (c.to_param_spec(),), default_step=c.step
                        )[0]
                    )
                )
    contexts = [high]
    if any(
        alternate[c.name] != high[c.name]
        for c in project.control_specs()
        if c.name != control.name
    ):
        contexts.append(alternate)
    screening = []
    for position in (control.max, midpoint):
        for context in contexts:
            _append(project, [position], context, "screen")
            params = {**context, control.name: position}
            entry = next(
                e for e in project.entries if e.split == "gain" and e.params == params
            )
            entry.user_metadata.setdefault("output_gain", {})["phase"] = "screen"
            if entry.y_path not in screening:
                screening.append(entry.y_path)
    _append(project, grid, high, "grid")
    by_path = {e.y_path: e for e in project.entries}
    project.entries = [by_path[path] for path in screening] + [
        e for e in project.entries if e.y_path not in screening
    ]
    project.output_gain_state = {
        "method": "grid",
        "status": "measuring",
        "screen_paths": screening,
        "screen_decision": None,
    }
    project.output_gain = None


def _signal(project, root, entry):
    import soundfile as sf

    signal, rate = sf.read(Path(root) / entry.y_path, dtype="float64", always_2d=True)
    if (
        rate != project.sample_rate
        or signal.shape[1] != 1
        or not np.isfinite(signal).all()
    ):
        raise ValueError("Invalid output-volume calibration recording")
    return signal[:, 0]


def _measure(reference, target, shift, noise):
    if shift >= 0:
        target, reference = target[shift:], reference[: len(target) - shift]
    else:
        reference, target = reference[-shift:], target[: len(reference) + shift]
    count = min(len(reference), len(target))
    if count < 256:
        raise ValueError("Volume calibration clips are too short after alignment")
    reference, target = reference[:count], target[:count]
    energy = float(np.mean(reference * reference))
    if energy < 1e-12:
        raise ValueError(
            "Maximum-volume reference is silent; check the rig and recapture that reference"
        )
    gain = float(np.mean(reference * target) / energy)
    residual = float(np.mean((target - gain * reference) ** 2))
    residual = max(0.0, residual - 4 * noise) / max(
        float(np.mean(target * target)), 1e-30
    )
    return gain, residual


def _curve(points):
    """Least-squares monotone level fit, with at most 65 runtime spline knots."""
    from scipy.interpolate import PchipInterpolator

    x = np.array([p[0] for p in points], dtype=float)
    y = np.clip(np.array([p[1] for p in points], dtype=float), 0.0, 1.0)
    y[-1] = 1.0
    blocks = []
    for i, value in enumerate(y):
        blocks.append([i, i + 1, float(value), 1])
        while (
            len(blocks) > 1
            and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]
        ):
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    for left, right, total, count in blocks:
        y[left:right] = total / count
    if len(x) <= MAX_POINTS:
        chosen = list(range(len(x)))
    else:
        chosen = [0, len(x) - 1]
        while len(chosen) < MAX_POINTS:
            interpolation = PchipInterpolator(x[chosen], y[chosen], extrapolate=False)
            error = np.abs(interpolation(x) - y)
            error[chosen] = -1
            index = int(np.argmax(error))
            if error[index] < 1e-12:
                break
            chosen = sorted([*chosen, index])
    interpolation = PchipInterpolator(x[chosen], y[chosen], extrapolate=False)
    return interpolation, [[float(x[i]), float(y[i])] for i in chosen]


def advance_calibration(project, root):
    """Check the initial four clips early, then fit only the requested grid points.

    Classification warnings are advisory and can be accepted without replanning.
    Unreadable/silent reference recordings still need a usable recapture.
    """
    control = volume_control(project)
    if control is None:
        return 0
    state = project.output_gain_state
    if state.get("method") != "grid" and state.get("status") != "ready":
        initialize_calibration(project)
        state = project.output_gain_state
    entries = [e for e in project.entries if e.split == "gain"]
    screening = [e for e in entries if e.y_path in state.get("screen_paths", [])]
    if not screening or any(e.status != "captured" for e in screening):
        return 0
    main = [
        e
        for e in entries
        if e.user_metadata.get("output_gain", {}).get("context") == "max"
    ]
    reference_entry = next(e for e in main if e.params[control.name] == control.max)
    reference = _signal(project, root, reference_entry)
    ref_power = float(np.mean(reference**2))
    if ref_power < 1e-12:
        raise ValueError(
            "Maximum-volume reference is silent; check the rig and recapture it"
        )
    floor = next(
        (
            e
            for e in main
            if e.params[control.name] == control.min and e.status == "captured"
        ),
        None,
    )
    noise = (
        float(np.mean(_signal(project, root, floor) ** 2)) if floor is not None else 0.0
    )
    if noise > ref_power * 1e-6:
        noise = 0.0
    if state.get("screen_decision") not in ("accepted", "passed"):
        reasons = []
        ratios = []
        for context in ("max", "check"):
            clips = [
                e
                for e in screening
                if e.user_metadata.get("output_gain", {}).get("context") == context
            ]
            if not clips:
                continue
            ref = next(e for e in clips if e.params[control.name] == control.max)
            ref_signal = _signal(project, root, ref)
            for clip in clips:
                if clip is ref:
                    continue
                target = _signal(project, root, clip)
                gain, residual = _measure(
                    ref_signal, target, int((clip.delay or 0) - (ref.delay or 0)), noise
                )
                ratios.append(
                    float(
                        np.sqrt(
                            max(float(np.mean(target**2)) - noise, 0.0)
                            / max(float(np.mean(ref_signal**2)) - noise, 1e-30)
                        )
                    )
                )
                # These first four clips are a quick screen, not a pass/fail
                # measurement. Leave headroom for interface noise and mild
                # distortion before asking the user to review the curve.
                if gain < -0.01 or gain > 1.15 or residual > 0.03:
                    reasons.append(
                        f"Waveform differs from simple attenuation ({100*residual:.1f}% residual power, {context} context)."
                    )
        if len(ratios) > 1 and _db_error(ratios[0], ratios[1]) > 2.0:
            reasons.append(
                f"Volume response differs between the two control configurations ({_db_error(ratios[0],ratios[1]):.2f} dB)."
            )
        if reasons:
            state.update(
                screen_decision="warning",
                warning_message="\n".join(reasons),
                status="measuring",
            )
            return 0
        state["screen_decision"] = "passed"
    if any(e.status != "captured" for e in entries):
        return 0
    values = {
        float(e.params[control.name]): float(
            np.sqrt(
                max(float(np.mean(_signal(project, root, e) ** 2)) - noise, 0.0)
                / max(ref_power - noise, 1e-30)
            )
        )
        for e in main
    }
    values[control.max] = 1.0
    interpolation, points = _curve(sorted(values.items()))
    segments = [
        dict(
            left=float(interpolation.x[i]),
            right=float(interpolation.x[i + 1]),
            coefficients=interpolation.c[:, i].tolist(),
        )
        for i in range(len(points) - 1)
    ]
    project.output_gain = dict(
        schema_version=1,
        name=control.name,
        min=control.min,
        max=control.max,
        default=control.default,
        index=next(
            i for i, c in enumerate(project.control_specs()) if c.name == control.name
        ),
        reference=control.max,
        interpolation="pchip",
        points=points,
        segments=segments,
        smoothing_ms=5.0,
    )
    state.update(
        status="ready",
        measurements=len(values),
        curve_points=len(points),
        fit_method="monotone least squares + PCHIP",
    )
    return 0


def _db_error(a, b):
    if max(a, b) < 1e-4:
        return 0.0
    return abs(20 * math.log10(max(a, 1e-9) / max(b, 1e-9)))


class OutputGainExportHook:
    is_output_gain_hook = True

    def __init__(self, curve):
        self.curve = curve

    def apply(self, model):
        if model["architecture"] not in ("HyperA2", "HyperWaveNet"):
            raise ValueError("Output-gain export requires a parametric neural effect")
        return {
            **model,
            "architecture": "ParametricOutputGain",
            "config": {
                "base_architecture": model["architecture"],
                "base_config": model["config"],
                "output_gain": self.curve,
            },
        }


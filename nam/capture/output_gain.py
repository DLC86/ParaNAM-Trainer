"""Measure an output attenuator separately from the neural effect.

All neural captures use the maximum physical volume. Short calibration captures
fit y(v)=gain(v)*y(max), check waveform residuals and a second knob context, then
verify/refine the monotone interpolant on the user's realizable capture grid.
"""

from dataclasses import replace
from pathlib import Path
import math

import numpy as np

from .params import SwitchSpec
from .project import CaptureEntryModel
from .planner import _planned_from_params, _quantize_to_capture_grid

TOLERANCE_DB = 0.25
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
        project.entries.append(
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
            )
        )
    return len(planned)


def initialize_calibration(project):
    control = volume_control(project)
    if control is None:
        return
    positions = sorted(
        {_snap(control, x) for x in np.linspace(control.min, control.max, 9)}
    )
    high = _context(project)
    _append(project, positions, high, "coarse")
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
    if any(
        alternate[c.name] != high[c.name]
        for c in project.control_specs()
        if c.name != control.name
    ):
        _append(
            project,
            [_snap(control, (control.min + control.max) / 2), control.max],
            alternate,
            "check",
        )
    project.output_gain_state = {
        "status": "measuring",
        "baseline": positions,
        "round": 0,
        "tolerance_db": TOLERANCE_DB,
    }


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
            "Maximum-volume reference is silent; check the rig and capture level"
        )
    gain = float(np.mean(reference * target) / energy)
    residual = float(np.mean((target - gain * reference) ** 2))
    if (
        gain < -0.001
        or gain > 1.05
        or residual > 0.001 * float(np.mean(target * target)) + 4 * noise + 1e-12
    ):
        raise ValueError(
            "This volume changes tone/distortion or has insufficient signal/noise: use a normal neural knob, or correct the recording and recapture calibration"
        )
    return min(1.0, max(0.0, gain)), residual / max(
        float(np.mean(target * target)), 1e-30
    )


def _curve(points):
    from scipy.interpolate import PchipInterpolator

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    if np.any(np.diff(y) < -0.01):
        raise ValueError(
            "Measured volume is not a monotone attenuator; use a normal neural knob"
        )
    # Suppress only small measurement-noise reversals; gross reversals fail above.
    y = np.maximum.accumulate(y)
    interpolation = PchipInterpolator(x, y, extrapolate=False)
    return interpolation, [[float(a), float(b)] for a, b in zip(x, y)]


def advance_calibration(project, root):
    """Return added point count. Failures keep recordings and block training."""
    control = volume_control(project)
    entries = [e for e in project.entries if e.split == "gain"]
    if control is None or not entries or any(e.status != "captured" for e in entries):
        return 0
    state = project.output_gain_state
    main = [
        e
        for e in entries
        if e.user_metadata.get("output_gain", {}).get("context") == "max"
    ]
    reference_entry = next(e for e in main if e.params[control.name] == control.max)
    reference = _signal(project, root, reference_entry)
    minimum = min(main, key=lambda e: e.params[control.name])
    minimum_signal = _signal(project, root, minimum)
    noise = (
        float(np.mean(minimum_signal**2)) if minimum.params[control.name] == 0 else 0
    )
    # A minimum which is itself audible must not mask a non-scalar waveform.
    if noise > float(np.mean(reference**2)) * 1e-6:
        noise = 0
    values = {}
    residuals = []
    for e in main:
        gain, residual = _measure(
            reference,
            _signal(project, root, e),
            int((e.delay or 0) - (reference_entry.delay or 0)),
            noise,
        )
        values[float(e.params[control.name])] = gain
        residuals.append(residual)
    values[control.max] = 1.0
    interpolation, points = _curve(sorted(values.items()))
    # Independent scalar-fit check with other knobs in a contrasting position.
    check = [
        e
        for e in entries
        if e.user_metadata.get("output_gain", {}).get("context") == "check"
    ]
    if check:
        other_reference = next(
            e for e in check if e.params[control.name] == control.max
        )
        other_signal = _signal(project, root, other_reference)
        for e in check:
            if e is other_reference:
                continue
            gain, residual = _measure(
                other_signal,
                _signal(project, root, e),
                int((e.delay or 0) - (other_reference.delay or 0)),
                noise,
            )
            expected = float(interpolation(e.params[control.name]))
            if _db_error(gain, expected) > 0.5:
                raise ValueError(
                    "Output-volume taper depends on the other controls; use a normal neural knob"
                )
            residuals.append(residual)
    baseline = sorted(set(state.get("baseline", values)))
    base_curve, _ = _curve([(x, values[x]) for x in baseline])
    bad = []
    for left, right in state.get("verify_intervals", list(zip(baseline, baseline[1:]))):
        samples = [x for x in values if left < x < right]
        if not samples or any(
            _db_error(values[x], float(base_curve(x))) > TOLERANCE_DB for x in samples
        ):
            bad.append((left, right))
    refinements = set()
    next_intervals = []
    all_positions = sorted(values)
    for left, right in zip(all_positions, all_positions[1:]):
        if any(a <= left and right <= b for a, b in bad):
            middle = _snap(control, (left + right) / 2)
            if left < middle < right and middle not in values:
                refinements.add(middle)
                next_intervals.append([left, right])
    if refinements:
        if len(values) + len(refinements) > MAX_POINTS:
            raise ValueError(
                "Output volume needs more than 65 measurements for 0.25 dB accuracy; use a normal knob or inspect the calibration"
            )
        state.update(
            status="measuring",
            baseline=all_positions,
            verify_intervals=next_intervals,
            round=state.get("round", 0) + 1,
        )
        return _append(project, sorted(refinements), _context(project), "verify")
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
        tolerance_db=TOLERANCE_DB,
    )
    state.update(
        status="ready", measurements=len(values), max_residual_esr=max(residuals)
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

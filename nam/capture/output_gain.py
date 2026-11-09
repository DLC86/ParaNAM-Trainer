"""Measure a clean output attenuator from peak and RMS levels."""

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
                        "context": "max",
                    }
                },
            ),
        )
    return len(planned)


def initialize_calibration(project):
    """Capture only the fixed volume grid with every other control at maximum."""
    from .planner import _capture_grid_values

    control = volume_control(project)
    if control is None:
        return
    # Older plans included redundant calibration captures with other controls at
    # defaults. Keep their WAVs on disk, but remove them from the active plan.
    project.entries = [
        entry
        for entry in project.entries
        if entry.split != "gain"
        or entry.user_metadata.get("output_gain", {}).get("context") != "check"
    ]
    grid = _capture_grid_values(
        replace(control.to_param_spec(), avoid_zero=False, avoid_max=False),
        control.step,
    )
    high = _context(project)
    _append(project, grid, high, "grid")
    gain_entries = sorted(
        (entry for entry in project.entries if entry.split == "gain"),
        key=lambda entry: float(entry.params[control.name]),
        reverse=True,
    )
    project.entries = gain_entries + [
        entry for entry in project.entries if entry.split != "gain"
    ]
    project.output_gain_state = {
        "method": "grid",
        "status": "measuring",
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


def _dbfs(signal):
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    rms = float(np.sqrt(np.mean(signal * signal))) if len(signal) else 0.0
    floor = 1e-12
    return 20 * math.log10(max(peak, floor)), 20 * math.log10(max(rms, floor))


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



def _muted_minimum_from_raw(project, root, entry):
    """Recheck an older capture using the actual played test blips, not a dB floor.

    Earlier versions could save a muted minimum without its silent_minimum flag
    when the interface's background noise was above an arbitrary RMS threshold.
    Reuse the same amp-return detector as a live recording if its raw signal and
    played preamble are available. Never infer silence from the trimmed WAV alone.
    """
    from .latency import measure_delay
    from .session import played_preamble, raw_paths

    raw, _ = raw_paths(root, entry.y_path)
    if not raw.is_file():
        return False
    try:
        import soundfile as sf

        audio, rate = sf.read(raw, dtype="float32", always_2d=True)
        if rate != project.sample_rate or audio.shape[1] != 1:
            return False
        preamble = played_preamble(root, entry.y_path, rate)
        return not measure_delay(audio[:, 0], preamble).detected
    except (OSError, ValueError, RuntimeError):
        # Projects predating raw recordings cannot be reclassified reliably.
        return False


def advance_calibration(project, root):
    """Compare peak and RMS attenuation; warn only when they differ by >0.2 dB."""
    control = volume_control(project)
    if control is None:
        return 0
    state = project.output_gain_state
    if state.get("method") != "grid" and state.get("status") != "ready":
        initialize_calibration(project)
        state = project.output_gain_state
    entries = [e for e in project.entries if e.split == "gain"]
    main = [
        e
        for e in entries
        if e.user_metadata.get("output_gain", {}).get("context") == "max"
    ]
    reference_entry = next(e for e in main if e.params[control.name] == control.max)
    if reference_entry.status != "captured":
        return 0
    reference = _signal(project, root, reference_entry)
    ref_peak, ref_rms = _dbfs(reference)
    if ref_peak <= -220 or ref_rms <= -220:
        raise ValueError(
            "Maximum-volume reference is silent; check the rig and recapture it"
        )
    if state.get("screen_decision") != "accepted":
        discrepancies = []
        for entry in main:
            if entry is reference_entry or entry.status != "captured":
                continue
            # CaptureSession explicitly identifies a verified silent minimum.
            # A noise-floor recording does not have a meaningful crest factor
            # and must not trigger a non-linearity warning.
            if entry.user_metadata.get("output_gain", {}).get("silent_minimum"):
                continue
            if (
                entry.params[control.name] == control.min
                and _muted_minimum_from_raw(project, root, entry)
            ):
                entry.user_metadata.setdefault("output_gain", {})["silent_minimum"] = True
                continue
            peak, rms = _dbfs(_signal(project, root, entry))
            # A hard mute carries only interface noise (or digital zero), so its
            # measured crest factor cannot describe the pedal's attenuation curve.
            if peak < -100 or rms < -100:
                continue
            discrepancy = abs((ref_peak - peak) - (ref_rms - rms))
            if discrepancy > 0.2:
                discrepancies.append((entry.params[control.name], discrepancy))
        if discrepancies:
            position, discrepancy = max(discrepancies, key=lambda pair: pair[1])
            state.update(
                screen_decision="warning",
                warning_message=(
                    f"At volume {position:g}, peak and RMS attenuation differ by "
                    f"{discrepancy:.2f} dB (limit: 0.20 dB). Check for distortion in "
                    "the audio interface or reamp box, or go back to Plan and remove "
                    "Clean volume. You can still dismiss this warning and continue."
                ),
                status="measuring",
            )
            return 0
        # A previously flagged capture may have been re-recorded as silent.
        # Do not preserve an obsolete warning after reevaluation.
        if state.get("screen_decision") == "warning":
            state.update(screen_decision=None, warning_message="")
    if any(e.status != "captured" for e in entries):
        return 0
    values = {
        float(e.params[control.name]): 10 ** ((_dbfs(_signal(project, root, e))[1] - ref_rms) / 20)
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
        fit_method="peak/RMS attenuation + monotone PCHIP",
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


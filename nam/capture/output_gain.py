"""Measure a clean output attenuator from peak and RMS levels."""

from dataclasses import replace
from pathlib import Path
import math

import numpy as np

from .params import SwitchSpec
from .project import CaptureEntryModel
from .planner import _planned_from_params, _capture_grid_values

MAX_POINTS = 65


def volume_control(project):
    return next(
        (c for c in project.control_specs() if getattr(c, "clean_volume", False)), None
    )


def _context(project):
    """Set every other knob to 10, ignoring capture avoids and plan rules.

    If 10 is outside a knob's range or cannot be dialled on its capture grid,
    use its nearest physically reachable value. Categorical switches retain
    their highest position, as they have no numeric position 10.
    """
    result = {}
    for control in project.control_specs():
        if isinstance(control, SwitchSpec):
            result[control.name] = control.enum_names[-1]
        elif getattr(control, "clean_volume", False):
            # Only the volume is swept; its maximum is the reference.
            result[control.name] = control.max
        else:
            spec = replace(
                control.to_param_spec(),
                avoid_zero=False, avoid_max=False,
                avoid_below=None, avoid_above=None,
            )
            values = _capture_grid_values(spec, control.step)
            result[control.name] = min(values, key=lambda value: (abs(value - 10), -value))
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
    """Sweep clean volume with other controls at 10; bypass all plan restrictions."""
    control = volume_control(project)
    if control is None:
        return
    # Unlike neural train/validation captures, output-volume calibration
    # deliberately includes the full volume sweep, even avoided endpoints.
    spec = replace(
        control.to_param_spec(),
        avoid_zero=False, avoid_max=False,
        avoid_below=None, avoid_above=None,
    )
    grid = _capture_grid_values(spec, control.step)
    positions = sorted(
        {v for v in grid if round(control.min, 6) < v < round(control.max, 6)}
        | {control.min, control.max}
    )
    context = _context(project)
    # No plan rules apply to calibration, including rules on the volume knob
    # or on other controls parked at 10.
    project.entries = [
        entry for entry in project.entries
        if entry.split != "gain"
        or (
            entry.user_metadata.get("output_gain", {}).get("phase") != "reference_repeat"
            and entry.user_metadata.get("output_gain", {}).get("context") == "max"
            and entry.params.get(control.name) in positions
            and all(entry.params.get(name) == value for name, value in context.items()
                    if name != control.name)
        )
    ]
    _append(project, positions, context, "grid")
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
    """Verify scalar-only volume changes with a noise-aware waveform null test."""
    control = volume_control(project)
    if control is None:
        return 0
    state = project.output_gain_state
    if state.get("method") != "grid" and state.get("status") != "ready":
        initialize_calibration(project)
        state = project.output_gain_state
    entries = [e for e in project.entries if e.split == "gain"]
    main = [
        e for e in entries
        if e.user_metadata.get("output_gain", {}).get("context") == "max"
        and e.user_metadata.get("output_gain", {}).get("phase") != "reference_repeat"
    ]
    reference_entry = next((e for e in main if e.params[control.name] == control.max), None)
    if reference_entry is None or len(main) < 2:
        raise ValueError("Linear Volume requires its maximum and at least one lower measurement")
    if reference_entry.status != "captured":
        return 0
    # The max-volume capture contains two consecutive passes recorded in one
    # operation. The second WAV is an auxiliary file, not a planned capture.
    reference = _signal(project, root, reference_entry)
    repeat_path = reference_entry.user_metadata.get("output_gain", {}).get("repeat_y_path")
    repeat = None
    if repeat_path:
        repeat_file = Path(root) / repeat_path
        if repeat_file.is_file():
            import soundfile as sf
            repeat_audio, repeat_rate = sf.read(
                repeat_file, dtype="float64", always_2d=True
            )
            if (
                repeat_rate == project.sample_rate
                and repeat_audio.shape == (len(reference), 1)
                and np.isfinite(repeat_audio).all()
            ):
                repeat = repeat_audio[:, 0]
    ref_peak, ref_rms = _dbfs(reference)
    if ref_peak <= -220 or ref_rms <= -220:
        raise ValueError(
            "Maximum-volume reference is silent; check the rig and recapture it"
        )
    # Reclassify earlier captures before both warning checks and curve fitting.
    # This also handles a warning that had already been dismissed.
    for entry in main:
        if (
            entry.status == "captured"
            and entry.params[control.name] == control.min
            and not entry.user_metadata.get("output_gain", {}).get("silent_minimum")
            and _muted_minimum_from_raw(project, root, entry)
        ):
            entry.user_metadata.setdefault("output_gain", {})["silent_minimum"] = True
    if state.get("screen_decision") != "accepted":
        from .volume_linearity import classify_volume_capture, MAX_RESIDUAL_DB

        reports = []
        nonlinear = []
        inconclusive = []
        for entry in main:
            if entry is reference_entry or entry.status != "captured":
                continue
            position = entry.params[control.name]
            if entry.user_metadata.get("output_gain", {}).get("silent_minimum"):
                reports.append({"volume": position, "status": "verified_silent"})
                continue
            measured = _signal(project, root, entry)
            peak, rms = _dbfs(measured)
            # All-zero recordings are not a meaningful null test. The real
            # recording session can independently verify a muted endpoint.
            if peak < -100 or rms < -100:
                reports.append({"volume": position, "status": "inconclusive",
                                "reason": "Capture is below -100 dBFS"})
                continue
            outcome = classify_volume_capture(
                reference, measured, project.sample_rate, repeat_reference=repeat
            )
            outcome["volume"] = position
            reports.append(outcome)
            if outcome["status"] == "nonlinear":
                nonlinear.append(outcome)
            elif outcome["status"] == "inconclusive":
                inconclusive.append(outcome)

        state["linearity_report"] = reports
        state["linearity_threshold_db"] = MAX_RESIDUAL_DB
        if nonlinear:
            worst = max(nonlinear, key=lambda item: item["residual_db"])
            state.update(
                screen_decision="warning",
                warning_message=(
                    f"At volume {worst['volume']:g}, the level-matched waveform differs "
                    f"from the maximum-volume reference by {worst['residual_db']:.1f} dB "
                    f"(limit: {MAX_RESIDUAL_DB:g} dB). The residual is above the "
                    "recording noise/repeatability floor. This volume control is not "
                    "a pure attenuator: check routing or use a neural volume knob. "
                    "You may dismiss the warning and continue."
                ),
                status="measuring",
            )
            return 0
        if inconclusive:
            # Warn rather than silently certify linearity: very quiet captures,
            # background noise and unstable reference takes cannot establish
            # a -40 dB upper bound on waveform changes.
            worst = max(
                inconclusive,
                key=lambda item: item.get("residual_db", -math.inf),
            )
            state.update(
                screen_decision="warning",
                warning_message=(
                    f"At volume {worst['volume']:g}, the normalized waveform comparison "
                    f"is inconclusive: {worst.get('reason', 'insufficient SNR')}. "
                    "Check the recording noise, capture alignment and volume setting; "
                    "recapture if necessary, or dismiss to continue."
                ),
                status="measuring",
            )
            return 0
        if state.get("screen_decision") == "warning":
            state.update(screen_decision=None, warning_message="")
    if any(e.status != "captured" for e in entries):
        return 0
    # A verified silent minimum contains interface noise, not amplified test
    # signal: use zero gain instead of fitting its noise-floor RMS as attenuation.
    values = {
        float(e.params[control.name]): (
            0.0
            if e.user_metadata.get("output_gain", {}).get("silent_minimum")
            else 10 ** ((_dbfs(_signal(project, root, e))[1] - ref_rms) / 20)
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
        fit_method="RMS attenuation + monotone PCHIP; waveform linearity check",
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
        if model["architecture"] not in (
            "HyperA2", "HyperWaveNet", "ParametricSlimmableContainer"
        ):
            raise ValueError("Output-gain export requires a parametric neural effect")
        # A linear attenuator must never become a hypernetwork input. Keep the
        # original neural params, weights and branch offsets unchanged and
        # store the DSP stage in an optional top-level extension. Legacy
        # ParametricOutputGain wrappers remain readable in updated players.
        extensions = dict(model.get("extensions", {}))
        if "linear_volume" in extensions:
            raise ValueError("Duplicate linear_volume extension")
        extensions["linear_volume"] = self.curve
        return {**model, "extensions": extensions}


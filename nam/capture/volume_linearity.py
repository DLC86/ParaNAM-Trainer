"""Check whether output-volume captures differ only by a scalar gain.

This is intentionally *not* a THD estimator. The reference may itself be a
heavily clipped fuzz; only changes caused by the volume control matter.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .resample import apply_fractional_delay

# Residual amplitude relative to the least-squares scaled reference.
MAX_RESIDUAL_DB = -40.0
# A measurement must beat the repeat/noise floor by at least this much.
RELIABILITY_MARGIN_DB = 6.0


@dataclass(frozen=True)
class WaveformComparison:
    residual_db: float
    gain: float
    noise_db: float | None


def _db(ratio: float) -> float:
    if ratio <= 0:
        return -math.inf
    return 20.0 * math.log10(ratio)


def _aligned(reference: np.ndarray, measured: np.ndarray, rate: int):
    """Remove small integer/subsample capture offsets, never resample the target.

    The capture session already applies its measured timing correction. The
    restricted search here only removes residual differences between takes.
    Correlation is computed on the loudest of several short regions rather than
    FFT-correlating entire minute-long calibration files.
    """
    reference = np.asarray(reference, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    if reference.ndim != 1 or measured.ndim != 1 or len(reference) != len(measured):
        raise ValueError("Volume comparisons require equal-length mono captures")
    n = len(reference)
    lag_limit = 8
    trim = 512
    if n < 2 * trim + 256:
        raise ValueError("Volume comparison recording is too short")

    window = min(max(2048, int(rate * 0.5)), n - 2 * trim - 2 * lag_limit)
    first = trim + lag_limit
    last = n - trim - lag_limit - window
    starts = np.linspace(first, last, 9, dtype=int)
    start = int(max(starts, key=lambda s: float(np.dot(
        reference[s:s + window], reference[s:s + window]
    ))))
    segment = reference[start:start + window]
    if float(np.dot(segment, segment)) < 1e-18:
        raise ValueError("Volume reference has insufficient signal")
    lags = np.arange(-lag_limit, lag_limit + 1)
    scores = np.asarray([
        float(np.dot(segment, measured[start + lag:start + lag + window]))
        for lag in lags
    ])
    peak = int(np.argmax(scores))
    if peak == 0 or peak == len(scores) - 1:
        raise ValueError("Volume capture timing differs by more than 7 samples")
    integer_lag = int(lags[peak])
    left, mid, right = scores[peak - 1:peak + 2]
    curvature = float(left - 2.0 * mid + right)
    fractional = (
        float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
        if curvature < -1e-18 else 0.0
    )
    a = reference[trim:n - trim]
    b = measured[trim + integer_lag:n - trim + integer_lag]
    if abs(fractional) > 0.002:
        # A positive correlation lag means the measured waveform arrived
        # later; advancing it by the fractional lag brings it to the reference.
        b = apply_fractional_delay(b, -fractional)
    return a, b


def compare_waveforms(reference: np.ndarray, measured: np.ndarray, rate: int) -> WaveformComparison:
    """Best-fit scalar null test plus a conservative quiet-region noise estimate.

    Only regions containing useful reference signal contribute to the fit and
    residual. Truly quiet sections, when present, estimate uncorrelated noise.
    Everything is in amplitude dB relative to the fitted signal.
    """
    r, y = _aligned(reference, measured, rate)
    # Keep at least eight blocks even with short calibration fixtures.
    block = max(256, min(4096, int(rate / 12), len(r) // 8))
    count = len(r) // block
    if count < 2:
        raise ValueError("Volume capture is too short for reliable analysis")
    rb = r[:count * block].reshape(count, block)
    yb = y[:count * block].reshape(count, block)
    power = np.mean(rb * rb, axis=1)
    active = power >= max(float(power.max()) * 1e-5, float(power.mean()) * 0.002)
    if not np.any(active):
        raise ValueError("Volume reference contains no usable signal")
    ra, ya = rb[active], yb[active]
    reference_power = float(np.mean(ra * ra))
    gain = float(np.mean(ra * ya) / reference_power)
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("Volume capture is silent or polarity-inverted")
    scale = gain * math.sqrt(reference_power)
    residual = float(np.sqrt(np.mean((ya - gain * ra) ** 2)))
    # Quiet blocks must be at least 40 dB down from the average active
    # reference power, otherwise a low-level musical passage is not noise.
    quiet = power <= reference_power * 1e-4
    noise_db = None
    if np.count_nonzero(quiet) >= 2:
        noise_power = float(np.mean(yb[quiet] ** 2) +
                            gain * gain * np.mean(rb[quiet] ** 2))
        noise_db = _db(math.sqrt(noise_power) / scale)
    return WaveformComparison(_db(residual / scale), gain, noise_db)


def classify_volume_capture(
    reference: np.ndarray,
    measured: np.ndarray,
    rate: int,
    *,
    repeat_reference: np.ndarray | None,
) -> dict:
    """Return linear/nonlinear/inconclusive with measured residual and floor.

    The repeated max-volume take captures the real repeatability limit. Its
    absolute noise/error can dominate a strongly attenuated take, so the floor
    is conservatively converted to that take's amplitude. Quiet passages can
    only raise (never lower) that estimated floor.
    """
    try:
        result = compare_waveforms(reference, measured, rate)
    except ValueError as exc:
        return dict(status="inconclusive", reason=str(exc))
    if repeat_reference is None:
        return dict(
            status="inconclusive", residual_db=round(result.residual_db, 2),
            reason="No automatic repeat reference; recapture the maximum-volume setting",
        )
    floor_db = result.noise_db
    repeat_db = None
    if repeat_reference is not None:
        try:
            repeat = compare_waveforms(reference, repeat_reference, rate)
            repeat_db = repeat.residual_db
            # Treat repeatability noise as unattenuated downstream noise.
            # This intentionally becomes more conservative at low volume.
            scaled_repeat_db = repeat_db - _db(min(1.0, result.gain))
            floor_db = max(floor_db if floor_db is not None else -math.inf,
                           scaled_repeat_db)
        except ValueError as exc:
            return dict(status="inconclusive", reason=f"Repeat reference: {exc}")
    report = dict(
        residual_db=round(result.residual_db, 2),
        gain=round(result.gain, 8),
        repeat_db=round(repeat_db, 2) if repeat_db is not None else None,
        noise_floor_db=round(floor_db, 2) if floor_db is not None else None,
    )
    if floor_db is None:
        return {**report, "status": "inconclusive",
                "reason": "No repeat reference or usable quiet-section noise estimate"}
    if floor_db > MAX_RESIDUAL_DB - RELIABILITY_MARGIN_DB:
        return {**report, "status": "inconclusive",
                "reason": "Noise/repeatability is too high to verify a -40 dB residual"}
    if result.residual_db <= MAX_RESIDUAL_DB:
        return {**report, "status": "linear"}
    if result.residual_db - floor_db < RELIABILITY_MARGIN_DB:
        return {**report, "status": "inconclusive",
                "reason": "Residual cannot be separated reliably from noise"}
    return {**report, "status": "nonlinear"}

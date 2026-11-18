"""Noise-aware waveform checks for Linear Volume, including heavily clipped fuzz."""

import numpy as np
import pytest

from nam.capture.volume_linearity import (
    classify_volume_capture,
    compare_waveforms,
)


RATE = 48000


def _fuzz(seconds=1):
    t = np.arange(int(RATE * seconds)) / RATE
    waveform = np.sin(2 * np.pi * 137 * t)
    waveform += 0.27 * np.sin(2 * np.pi * 411 * t + 0.2)
    waveform += 0.07 * np.sin(2 * np.pi * 901 * t)
    return 0.42 * np.tanh(8 * waveform)


def test_fuzz_is_linear_volume_when_only_scaled():
    signal = _fuzz()
    result = classify_volume_capture(
        signal, 0.12 * signal, RATE, repeat_reference=signal.copy(),
    )
    assert result["status"] == "linear"
    assert result["residual_db"] < -100


def test_volume_changes_fuzz_timbre_with_nearly_equal_level():
    signal = _fuzz()
    # Nonlinearly compress the waveform again: the output is still a fuzz,
    # but it cannot be explained by one scalar gain.
    changed = 0.3 * np.tanh(2.5 * signal)
    result = classify_volume_capture(
        signal, changed, RATE, repeat_reference=signal.copy(),
    )
    assert result["status"] == "nonlinear"
    assert result["residual_db"] > -40


def test_small_capture_time_offset_does_not_look_like_coloration():
    signal = _fuzz()
    shifted = np.pad(signal * 0.37, (3, 0))[:-3]
    result = classify_volume_capture(
        signal, shifted, RATE, repeat_reference=signal.copy(),
    )
    assert result["status"] == "linear"


def test_noisy_very_quiet_volume_is_inconclusive():
    rng = np.random.default_rng(12)
    signal = _fuzz()
    repeat = signal + rng.normal(0, 0.0002, len(signal))
    very_quiet = 0.001 * signal + rng.normal(0, 0.0002, len(signal))
    result = classify_volume_capture(
        signal, very_quiet, RATE, repeat_reference=repeat,
    )
    assert result["status"] == "inconclusive"


def test_no_repeat_and_no_quiet_noise_estimate_is_inconclusive():
    signal = _fuzz()
    result = classify_volume_capture(
        signal, signal * 0.5, RATE, repeat_reference=None,
    )
    assert result["status"] == "inconclusive"


def test_length_mismatch_rejected():
    signal = _fuzz()
    with pytest.raises(ValueError, match="equal-length"):
        compare_waveforms(signal, signal[:-1], RATE)

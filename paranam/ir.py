"""Offline IR preparation and bounded-work streaming partitioned convolution."""

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HOP = 256
BUNDLED = Path(__file__).parent / "assets" / "irs"


def verify_bundled_irs():
    import hashlib
    import json

    manifest = json.loads((BUNDLED / "manifest.json").read_text(encoding="utf-8"))
    if len(manifest["files"]) != 6 or not (BUNDLED / "LICENSE.txt").is_file():
        raise RuntimeError("The bundled Tone3000 IR collection is incomplete")
    for name, record in manifest["files"].items():
        if (
            hashlib.sha256((BUNDLED / name).read_bytes()).hexdigest()
            != record["sha256"]
        ):
            raise RuntimeError(f"Bundled IR is damaged: {name}")
    return len(manifest["files"])


@dataclass(frozen=True)
class IRKernel:
    spectra: np.ndarray
    rate: int
    name: str


def prepare_ir(path, rate=48000):
    """Read/resample/normalize off the audio thread; retain at most two channels."""
    info = sf.info(path)
    if (
        info.channels not in (1, 2)
        or not math.isfinite(info.duration)
        or info.duration <= 0
    ):
        raise ValueError("Choose a non-empty mono/stereo IR WAV.")
    taps, source_rate = sf.read(path, dtype="float64", always_2d=True)
    if not np.isfinite(taps).all() or not np.any(taps):
        raise ValueError("The IR contains silence or non-finite samples.")
    if source_rate != rate:
        divisor = math.gcd(source_rate, rate)
        taps = resample_poly(taps, rate // divisor, source_rate // divisor, axis=0)
    # Match Tone3000's left-channel mean power response over 100–5000 Hz.
    frequencies = np.geomspace(100, min(5000, rate * 0.49), 64)
    times = np.arange(len(taps)) / rate
    power = np.mean(
        [
            abs(np.dot(taps[:, 0], np.exp(-2j * np.pi * f * times))) ** 2
            for f in frequencies
        ]
    )
    gain_db = (
        np.clip(-10 * np.log10(power), -48, 24)
        if power > 0 and np.isfinite(power)
        else 0
    )
    taps *= 10 ** (gain_db / 20)
    return kernel_from_taps(taps, rate, Path(path).stem)


def kernel_from_taps(taps, rate=48000, name="IR"):
    taps = np.asarray(taps, dtype=np.float64)
    if taps.ndim == 1:
        taps = taps[:, None]
    if len(taps) < 1 or taps.shape[1] not in (1, 2) or not np.isfinite(taps).all():
        raise ValueError("Invalid impulse response")
    if taps.shape[1] == 1:
        taps = np.repeat(taps, 2, axis=1)
    partitions = (len(taps) + HOP - 1) // HOP
    padded = np.zeros((partitions * HOP, 2))
    padded[: len(taps)] = taps
    blocks = np.zeros((partitions, 2 * HOP, 2))
    blocks[:, :HOP] = padded.reshape(partitions, HOP, 2)
    spectra = np.fft.rfft(blocks, axis=1)
    spectra.flags.writeable = False
    return IRKernel(spectra, rate, name)


class IRProcessor:
    """Uniform FFT partitions, exactly HOP samples latency for any callback size."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.history = np.zeros((len(kernel.spectra), HOP + 1), dtype=complex)
        self.head = 0
        self.input = np.zeros(HOP)
        self.used = 0
        self.output = np.zeros((HOP, 2))
        self.overlap = np.zeros((HOP, 2))

    def process(self, samples):
        result = np.empty((len(samples), 2), dtype=np.float32)
        offset = 0
        while offset < len(samples):
            count = min(HOP - self.used, len(samples) - offset)
            result[offset : offset + count] = self.output[self.used : self.used + count]
            self.input[self.used : self.used + count] = samples[offset : offset + count]
            self.used += count
            offset += count
            if self.used == HOP:
                self.history[self.head] = np.fft.rfft(self.input, n=2 * HOP)
                # Two views traverse the ring backwards. einsum accumulates the
                # products without allocating/copying a partition-sized tensor
                # in every audio block (especially costly for reverb IRs).
                split = self.head + 1
                spectrum = np.einsum(
                    "pf,pfc->fc",
                    self.history[self.head :: -1],
                    self.kernel.spectra[:split],
                    optimize=False,
                )
                if split < len(self.history):
                    spectrum += np.einsum(
                        "pf,pfc->fc",
                        self.history[: self.head : -1],
                        self.kernel.spectra[split:],
                        optimize=False,
                    )
                block = np.fft.irfft(spectrum, n=2 * HOP, axis=0)
                self.output = block[:HOP] + self.overlap
                self.overlap = block[HOP:].copy()
                self.head = (self.head + 1) % len(self.history)
                self.used = 0
        return result


class MonitorIR:
    """Crossfade IR changes over 20 ms without resetting on gain/knob edits."""

    def __init__(self, rate, processor=None):
        self.current = self.previous = processor
        self.fade_length = max(1, round(rate * 0.02))
        self.remaining = 0

    @staticmethod
    def render(processor, samples):
        return (
            processor.process(samples)
            if processor is not None
            else np.repeat(samples[:, None], 2, axis=1)
        )

    def process(self, samples, requested):
        if requested is not self.current and not self.remaining:
            self.previous, self.current = self.current, requested
            self.remaining = self.fade_length
        result = self.render(self.current, samples)
        if self.remaining:
            old = self.render(self.previous, samples)
            blend = np.clip(
                (self.fade_length - self.remaining + np.arange(1, len(samples) + 1))
                / self.fade_length,
                0,
                1,
            )
            result = old * (1 - blend[:, None]) + result * blend[:, None]
            self.remaining = max(0, self.remaining - len(samples))
            if not self.remaining:
                self.previous = None
        return result

"""Bundled TONE3000 audition inputs; decoding and resampling happen off the UI thread."""

import hashlib
import json
from pathlib import Path
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

BUNDLED = Path(__file__).parent / "assets" / "previews"


def preview_files():
    manifest = json.loads((BUNDLED / "manifest.json").read_text(encoding="utf-8"))
    return [(Path(name).stem, BUNDLED / name) for name in manifest["files"]]


def verify_bundled_previews():
    manifest = json.loads((BUNDLED / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["files"] or not (BUNDLED / "LICENSE.txt").is_file():
        raise RuntimeError("The bundled TONE3000 preview collection is incomplete")
    for name, entry in manifest["files"].items():
        if hashlib.sha256((BUNDLED / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Bundled preview is damaged: {name}")
    return len(manifest["files"])


class PreviewLoop:
    def __init__(self, path, rate):
        samples, source_rate = sf.read(path, dtype="float32", always_2d=True)
        samples = samples.mean(axis=1)
        if not len(samples) or not np.isfinite(samples).all():
            raise ValueError("The preview contains no valid audio")
        if source_rate != rate:
            divisor = math.gcd(source_rate, rate)
            samples = resample_poly(samples, rate // divisor, source_rate // divisor)
        self.samples = np.ascontiguousarray(samples, dtype=np.float32)
        self.position = 0

    def read(self, frames):
        indices = (self.position + np.arange(frames)) % len(self.samples)
        self.position = (self.position + frames) % len(self.samples)
        return self.samples[indices]

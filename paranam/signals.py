"""Verified original sweep and fixed validation signals, bundled for offline use."""

import hashlib
import json
import shutil
from pathlib import Path


def asset_path(name):
    return Path(__file__).resolve().parent / "assets" / name


def verify_assets():
    records = json.loads(asset_path("manifest.json").read_text(encoding="utf-8"))
    for record in records:
        path = asset_path(record["file"])
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
        ):
            raise RuntimeError(
                f"Bundled capture signal is missing or damaged: {record['file']}"
            )
    return records


def install_signals(directory, include_tts=False, include_trunc=False):
    verify_assets()
    signals = [
        ("standard.wav", "input_standard.wav"),
        ("validation.wav", "input_validation.wav"),
    ]
    if include_tts:
        signals.append(("tts-input-v11.wav", "input_tts_v11.wav"))
    if include_trunc:
        signals.append(("input-trunc-with-validation.wav", "input_trunc.wav"))
    for asset, target in signals:
        destination = Path(directory) / target
        source = asset_path(asset)
        if (
            not destination.exists()
            or hashlib.sha256(destination.read_bytes()).digest()
            != hashlib.sha256(source.read_bytes()).digest()
        ):
            temporary = destination.with_suffix(".pending.wav")
            shutil.copy2(source, temporary)
            temporary.replace(destination)


def prepare_faverooo_input(source, destination):
    """Adapt a user-supplied stimulus to ParaNAM's existing 10s/9s split.

    The upstream WAV is not redistributed. Preserve its samples inside the
    training region and use bundled standard audio for the held-out regions.
    """
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(source, dtype="float64", always_2d=True)
    if rate != 48000 or audio.shape[1] != 1 or not np.isfinite(audio).all():
        raise ValueError("Faverooo input must be finite mono audio at 48 kHz")
    if len(audio) < rate or np.max(np.abs(audio)) > 1.0 or not np.any(audio):
        raise ValueError("Invalid or silent Faverooo stimulus")
    with sf.SoundFile(asset_path("standard.wav")) as standard:
        lead = standard.read(10 * rate, dtype="float64", always_2d=True)
        standard.seek(len(standard) - 9 * rate)
        tail = standard.read(9 * rate, dtype="float64", always_2d=True)
    temporary = Path(destination).with_suffix(".pending.wav")
    try:
        sf.write(temporary, np.concatenate((lead, audio, tail)), rate, subtype="PCM_24")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

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


def install_signals(directory):
    verify_assets()
    for asset, target in (
        ("standard.wav", "input_standard.wav"),
        ("validation.wav", "input_validation.wav"),
    ):
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

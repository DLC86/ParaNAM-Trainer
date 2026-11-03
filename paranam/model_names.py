"""Filesystem-safe exported names derived from the model's Name metadata."""

import re


def model_basename(name, fallback="Model"):
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name or "")).strip().rstrip(" .")
    value = (
        value.encode("utf-8")[:220].decode("utf-8", errors="ignore").rstrip(" .")
        or fallback
    )
    if value.split(".")[0].upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        value = "_" + value
    return value

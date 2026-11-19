"""Safe metadata-only editing of exported NAM and ParaNAM JSON models.

The network's neural parameter order, encoding ranges, defaults, switch numeric
values, weights and calibration curve are never changed by this editor.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from nam.models.parametric._spec import ParamSpec
from nam.models.metadata import UserMetadata

USER_FIELDS = (
    "modeled_by", "gear_make", "gear_model", "gear_type", "tone_type",
    "input_level_dbu", "output_level_dbu", "input_z", "notes",
)
PARAMETRIC_ARCHITECTURES = {
    "HyperWaveNet", "HyperA2", "ParametricSlimmableContainer",
    "ConcatWaveNet", "ConcatLSTM", "ParametricOutputGain",
}


def load_model_file(path):
    path = Path(path)
    if path.suffix.lower() not in (".nam", ".pnam"):
        raise ValueError("Select a .nam or .pnam model file")
    with path.open("r", encoding="utf-8") as fp:
        model = json.load(fp)
    if (not isinstance(model, dict)
            or not isinstance(model.get("architecture"), str)
            or not isinstance(model.get("config"), dict)
            or not isinstance(model.get("weights"), list)
            or not isinstance(model.get("metadata", {}), dict)):
        raise ValueError("Invalid NAM/PNAM JSON model structure")
    if path.suffix.lower() == ".pnam" and model["architecture"] not in PARAMETRIC_ARCHITECTURES:
        raise ValueError("Unknown parametric architecture; editing its controls is unsafe")
    return model


def _parameter_arrays(model):
    """Return the neural param arrays in original neural order, including Full/Lite.

    The legacy output-gain wrapper stores the neural config in base_config.
    No array is ever reordered to implement display indices.
    """
    if model["architecture"] not in PARAMETRIC_ARCHITECTURES:
        return []
    config = model["config"]
    if model["architecture"] == "ParametricOutputGain":
        config = config.get("base_config", {})
    base = config.get("params")
    if not isinstance(base, list):
        raise ValueError("Parametric model does not contain a valid control array")
    arrays = [base]
    for branch in config.get("submodels", []):
        params = branch.get("config", {}).get("params")
        if not isinstance(params, list):
            raise ValueError("Invalid parametric Full/Lite branch control array")
        arrays.append(params)
    for values in arrays:
        if len(values) != len(base):
            raise ValueError("Full/Lite neural control counts do not match")
        for entry in values:
            ParamSpec.from_dict(entry)
    for other in arrays[1:]:
        for first, second in zip(base, other):
            a, b = ParamSpec.from_dict(first), ParamSpec.from_dict(second)
            if (a.name, a.type, a.min, a.max, a.default, a.enum_names) != (
                b.name, b.type, b.min, b.max, b.default, b.enum_names
            ):
                raise ValueError("Full/Lite neural controls are inconsistent")
    return arrays


def _linear_volume(model):
    extensions = model.get("extensions", {})
    if not isinstance(extensions, dict):
        raise ValueError("Invalid model extensions")
    volume = extensions.get("linear_volume")
    if volume is None and model["architecture"] == "ParametricOutputGain":
        volume = model["config"].get("output_gain")
    if volume is not None and not isinstance(volume, dict):
        raise ValueError("Invalid Linear Volume metadata")
    return volume


def editable_controls(model):
    """Return UI-safe control descriptions; index_sw never replaces neural value."""
    arrays = _parameter_arrays(model)
    if not arrays:
        return []
    params = arrays[0]
    controls = []
    for neural_pos, entry in enumerate(params):
        spec = ParamSpec.from_dict(entry)
        options = (
            [
                {"name": name, "value": value, "index_sw": spec.option_indices[value]}
                for value, name in enumerate(spec.enum_names)
            ] if spec.type == "switch" else []
        )
        controls.append(dict(
            key=neural_pos, kind="neural", name=spec.name, type=spec.type,
            min=spec.min, max=spec.max, default=spec.default,
            index=spec.index if spec.index is not None else neural_pos,
            options=options,
        ))
    volume = _linear_volume(model)
    if volume is not None:
        controls.append(dict(
            key="linear_volume", kind="linear_volume", name=volume["name"],
            type="continuous", min=volume["min"], max=volume["max"],
            default=volume.get("default", volume["max"]),
            index=volume.get("index", len(params)), options=[],
        ))
    names = [c["name"] for c in controls]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate control names in the model")
    controls.sort(key=lambda c: (c["index"], str(c["key"])))
    for index, control in enumerate(controls):
        control["index"] = index
    return controls


def validate_control_edits(model, edits):
    before = editable_controls(model)
    if not before:
        if edits:
            raise ValueError("Static models have no parametric controls")
        return []
    if not isinstance(edits, list):
        raise ValueError("Controls must be an ordered list")
    by_key = {c["key"]: c for c in before}
    if len(edits) != len(before) or {c.get("key") for c in edits} != set(by_key):
        raise ValueError("Cannot add/remove neural controls or change their identity")
    names = []
    indices = []
    for edit in edits:
        original = by_key[edit["key"]]
        name = edit.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every control needs a nonempty name")
        names.append(name.strip())
        index = edit.get("index")
        if type(index) is not int:
            raise ValueError("Display index must be an integer")
        indices.append(index)
        # Ranges define the normalizing map into network weights; changing
        # them without weight conversion would change the modeled tone.
        for key in ("min", "max", "default", "type", "kind"):
            if edit.get(key) != original[key]:
                raise ValueError(f"Changing {key} is unsafe without retraining")
        if original["type"] == "switch":
            opts = edit.get("options")
            old_by_value = {o["value"]: o for o in original["options"]}
            if (not isinstance(opts, list) or len(opts) != len(old_by_value)
                    or {o.get("value") for o in opts} != set(old_by_value)):
                raise ValueError("Switch neural values must remain unchanged")
            seen = []
            option_names = []
            for opt in opts:
                pos = opt.get("index_sw")
                if type(pos) is not int:
                    raise ValueError("Switch display indices must be integers")
                seen.append(pos)
                text = opt.get("name")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Switch labels must be nonempty")
                option_names.append(text.strip())
            if sorted(seen) != list(range(len(opts))):
                raise ValueError("index_sw values must form a 0..N-1 permutation")
            if len(set(option_names)) != len(option_names):
                raise ValueError("Switch labels must be unique")
        elif edit.get("options"):
            raise ValueError("Continuous knobs cannot have switch options")
    if len(set(names)) != len(names):
        raise ValueError("Control names must be unique")
    if sorted(indices) != list(range(len(indices))):
        raise ValueError("Control index values must form a 0..N-1 permutation")
    return edits


def apply_model_edits(model, metadata, controls=None):
    """Build a new document. Do not mutate the file opened in the editor."""
    result = deepcopy(model)
    merged = dict(result.get("metadata", {}))
    if not isinstance(metadata, dict):
        raise ValueError("Invalid metadata")
    for key in USER_FIELDS:
        value = metadata.get(key)
        if value is None or value == "":
            merged.pop(key, None)
        else:
            merged[key] = value
    # Enforce the same NAM field types and numerical validation as Train.
    UserMetadata.model_validate({
        k: v for k, v in merged.items() if k in UserMetadata.model_fields
    })
    if "notes" in merged and not isinstance(merged["notes"], str):
        raise ValueError("Notes must be text")
    result["metadata"] = merged

    # Keep independent static Full/Lite NAM submodels' user fields in sync,
    # never touching their gain/loudness or other computed metadata.
    if result.get("architecture") == "SlimmableContainer":
        for submodel in result.get("config", {}).get("submodels", ()):
            child = submodel.get("model")
            if isinstance(child, dict):
                cm = dict(child.get("metadata", {}))
                for key in USER_FIELDS:
                    if key in merged:
                        cm[key] = merged[key]
                    else:
                        cm.pop(key, None)
                child["metadata"] = cm

    if result["architecture"] in PARAMETRIC_ARCHITECTURES:
        edits = validate_control_edits(result, controls if controls is not None else editable_controls(result))
        arrays = _parameter_arrays(result)
        by_key = {item["key"]: item for item in edits}
        for pos in range(len(arrays[0])):
            edit = by_key[pos]
            for params in arrays:
                param = params[pos]
                param["name"] = edit["name"].strip()
                param["index"] = edit["index"]
                if edit["type"] == "switch":
                    ordered = sorted(edit["options"], key=lambda opt: opt["value"])
                    param["options"] = [
                        {"name": opt["name"].strip(), "value": opt["value"],
                         "index_sw": opt["index_sw"]} for opt in ordered
                    ]
                    if "enum_names" in param:
                        # Retain the old neural-order names for legacy readers;
                        # the new options array carries display indices.
                        param["enum_names"] = [opt["name"].strip() for opt in ordered]
        if "linear_volume" in by_key:
            volume = _linear_volume(result)
            volume["name"] = by_key["linear_volume"]["name"].strip()
            volume["index"] = by_key["linear_volume"]["index"]
        _parameter_arrays(result)  # Validate every modified branch.
    elif controls:
        raise ValueError("Static model has no parametric controls")
    return result


def write_model_file(source, destination, model):
    """Durable atomic save. Overwriting keeps a .bak copy of the old model."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file():
        raise ValueError("Original model no longer exists")
    if destination.suffix.lower() != source.suffix.lower():
        raise ValueError("The destination must retain the original .nam/.pnam extension")
    if not destination.parent.is_dir():
        raise ValueError("Destination directory does not exist")
    if destination.exists() and not destination.is_file():
        raise ValueError("Destination is not a regular file")
    # Preflight serialization/finite values before touching any file.
    payload = json.dumps(model, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.editor.", suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(payload)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        if destination.is_file():
            # Back up ANY existing destination, including a Save copy target
            # the user explicitly chose to replace. Do not touch the source
            # when Save copy creates a different filename.
            backup = destination.with_name(destination.name + ".bak")
            bfd, temporary_backup = tempfile.mkstemp(
                prefix=f".{destination.name}.backup.", suffix=".tmp",
                dir=destination.parent,
            )
            try:
                with os.fdopen(bfd, "wb") as out, destination.open("rb") as original:
                    shutil.copyfileobj(original, out)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temporary_backup, backup)
            finally:
                if os.path.exists(temporary_backup):
                    os.unlink(temporary_backup)
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return destination

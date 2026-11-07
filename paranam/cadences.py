"""Two editable clocks, with lossless checkpoint compatibility for older recipes."""

import copy
import math


def stored_schedules(spec):
    saved = spec.get("preferences", {}).get("rate_schedule", {})
    if saved:
        if "audio" in saved:
            return saved
        return {"audio": saved, "hypernetwork": saved}
    scheduler = spec.get("model", {}).get("lr_scheduler", {})
    if scheduler.get("class") == "IndependentGroupLR":
        return scheduler.get("kwargs", {})
    common = {
        **scheduler.get("kwargs", {}),
        "interval": scheduler.get("interval", "epoch"),
    }
    return {"audio": common, "hypernetwork": common}


def editable_cadences(workspace, options, schedules=None):
    """Equivalent mean decay for editing/new training; resume keeps its frozen clock."""
    from .recipes import a2_update_schedule, rate_cadence_multiplier

    result = dict(options)
    accounting = None
    for group, preset_key, gamma_key in (
        ("audio", "rate_preset", "lr_decay"),
        ("hypernetwork", "hypernetwork_rate_preset", "hypernetwork_lr_decay"),
    ):
        preset = options.get(
            preset_key, "follow" if group == "hypernetwork" else "manual"
        )
        gamma = options.get(gamma_key, options["lr_decay"])
        if preset == "follow":
            result[preset_key], result[gamma_key] = (
                result["rate_preset"],
                result["lr_decay"],
            )
            continue
        if preset in ("manual", "a2_updates", "cosine_budget"):
            continue
        original = (schedules or {}).get(group, {})
        if preset == "a2_dataset":
            result[preset_key] = "manual"
            continue
        interval = original.get("step_size", original.get("decay_steps"))
        if interval is None:
            if preset in ("a2_controls", "a2_slow"):
                interval = 62 * rate_cadence_multiplier(workspace.project, preset)
            else:
                if accounting is None:
                    from nam.capture.export import build_data_config
                    from .evaluation import reserve_validation_audio

                    project = workspace.project.model_copy(deep=True)
                    for entry in project.entries:
                        entry.status = "captured"
                    data = build_data_config(project)
                    reserve_validation_audio(data)
                    accounting = a2_update_schedule(
                        data,
                        workspace.directory,
                        workspace.training_batch_size(),
                        options["ny"],
                        1,
                        1,
                        1,
                        1,
                        normalization=preset,
                    )
                interval = accounting["decay_steps"]
        result[preset_key] = "a2_updates"
        result[gamma_key] = gamma ** (62 / interval)
    return result


def preserve_unchanged_cadences(workspace, old_options, entered, precision=9):
    view = editable_cadences(workspace, old_options)
    result = dict(entered)
    for preset, gamma in (
        ("rate_preset", "lr_decay"),
        ("hypernetwork_rate_preset", "hypernetwork_lr_decay"),
    ):
        if (
            preset == "hypernetwork_rate_preset"
            and old_options[preset] == "follow"
            and (
                result["rate_preset"] != old_options["rate_preset"]
                or result["lr_decay"] != old_options["lr_decay"]
            )
        ):
            continue
        if entered[preset] == view[preset] and math.isclose(
            entered[gamma], view[gamma], rel_tol=0, abs_tol=0.51 * 10**-precision
        ):
            result[preset], result[gamma] = old_options[preset], old_options[gamma]
    return result


def legacy_cadence_fingerprint(workspace, spec, batch_size=None):
    """Accept only equivalent display conversions; all other data/options stay strict."""
    saved = spec.get("preferences", {}).get("advanced_training")
    if not saved:
        return None
    current = workspace.training_options()
    original = {**current, **saved}
    if "hypernetwork_rate_preset" not in saved:
        original["hypernetwork_rate_preset"] = "follow"
    try:
        view = editable_cadences(workspace, original, stored_schedules(spec))
        current_view = editable_cadences(workspace, current)
    except (ValueError, OSError, RuntimeError):
        return None
    for key in view:
        if key in ("lr_decay", "hypernetwork_lr_decay"):
            if not math.isclose(
                view[key], current_view[key], rel_tol=0, abs_tol=0.51e-9
            ):
                return None
        elif view[key] != current_view[key]:
            return None
    restored = copy.copy(workspace)
    restored.preferences = copy.deepcopy(workspace.preferences)
    restored.preferences["advanced_" + workspace.project.capture_type] = original
    return restored.fingerprint(batch_size=batch_size)

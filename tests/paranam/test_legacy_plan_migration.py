"""Regressions for migration of historic Plans and Synthetic Mute captures."""

from nam.capture.params import KnobSpec
from nam.capture.plan_rules import PlanRule
from nam.capture.project import CaptureEntryModel, KnobModel
from paranam.workspace import Workspace


def _set_legacy_knob(workspace, **options):
    knob = KnobSpec("Gain", 0, 10, 1, 5, **options)
    workspace.project.knobs = [KnobModel.from_knob_spec(knob)]
    workspace.draft_controls = [knob.to_dict()]
    workspace.save()


def test_legacy_avoid_thresholds_migrate_to_rules_without_update_loop(workspace):
    _set_legacy_knob(workspace, avoid_below=1.0, avoid_above=9.0)
    reopened = Workspace(workspace.directory)
    assert not reopened.plan_changed()
    assert "avoid_below" not in reopened.draft_controls[0]
    assert "avoid_above" not in reopened.draft_controls[0]
    assert {(rule.operator, rule.value) for rule in reopened.plan_rules()} == {
        ("<", 1.0), (">", 9.0),
    }
    assert reopened.plan_rules() == reopened.project.plan_rules
    reopened.preferences["seed"] += 1
    reopened.generate_plan()
    assert not reopened.plan_changed()
    assert all(
        1.0 <= entry.params["Gain"] <= 9.0
        for entry in reopened.project.entries
        if reopened.project.classification_for_entry(entry) is None
    )
    assert not Workspace(reopened.directory).plan_changed()


def test_frozen_queue_snapshot_retains_original_avoid_fields(workspace):
    _set_legacy_knob(workspace, avoid_below=1.0)
    workspace.preferences["_queue_source_project"] = str(workspace.directory)
    workspace.save()
    frozen = Workspace(workspace.directory)
    assert frozen.project.knobs[0].avoid_below == 1.0
    assert frozen.project.plan_rules == []
    assert frozen.controls()[0].avoid_below == 1.0


def test_validation_synthetic_mute_does_not_repeatedly_mark_plan_changed(workspace):
    from paranam.synthetic_mute import materialize_synthetic_mutes

    row = workspace.project.entries_for_split("validation")[0]
    workspace.project.muted_settings.append(dict(row.params))
    materialize_synthetic_mutes(workspace)
    workspace.save()
    assert row.synthetic_mute and row.status == "captured"
    assert not workspace.plan_changed()
    assert not Workspace(workspace.directory).plan_changed()


def test_regenerate_plan_replaces_legacy_shifted_mute_with_exact_minimum(workspace):
    from paranam.synthetic_mute import is_mute_setting

    _set_legacy_knob(workspace, avoid_below=1.0, mute_at_min=True)
    legacy = {"Gain": 1.234567}  # Not a valid new sampled grid point.
    workspace.project.entries.append(CaptureEntryModel(
        index=999, split="train", params=legacy,
        y_path="captures/obsolete-mute-corner.wav", is_corner=True,
    ))
    workspace.project.muted_settings.append(dict(legacy))
    workspace.preferences["corners"] = True
    workspace.save()
    reopened = Workspace(workspace.directory)
    reopened.generate_plan()
    actual = reopened.project
    corners = [
        row for row in actual.entries
        if row.split == "train" and row.is_corner and row.params["Gain"] == 0
    ]
    assert corners
    assert all(row.synthetic_mute and row.status == "captured" for row in corners)
    assert all(is_mute_setting(actual, row.params) for row in corners)
    assert all((reopened.directory / row.y_path).is_file() for row in corners)
    assert not any(row.params == legacy for row in actual.entries)
    assert legacy not in actual.muted_settings
    assert not reopened.plan_changed()


def test_strict_mute_rule_comparisons_preserve_thresholds():
    below = PlanRule(controls=("Gain",), operator="<", value=5, kind="mute")
    above = PlanRule(controls=("Gain",), operator=">", value=5, kind="mute")
    assert below.excludes({"Gain": 4})
    assert not below.excludes({"Gain": 5})
    assert above.excludes({"Gain": 6})
    assert not above.excludes({"Gain": 5})


def test_editor_shows_strict_rules_without_knob_local_avoid(window, workspace):
    from paranam.plan_rules_ui import PlanRuleDialog
    from paranam.workspace import control_from_dict

    window._attach(workspace)
    assert not hasattr(window, "control_avoid_min")
    assert not hasattr(window, "control_avoid_max")
    dialog = PlanRuleDialog(
        None, [control_from_dict(c) for c in workspace.draft_controls],
    )
    ops = [
        dialog.rows["Gain"][1].itemText(i)
        for i in range(dialog.rows["Gain"][1].count())
    ]
    assert ops == ["<", "≤", "=", "≥", ">"]
    dialog.close()


def test_compare_panel_stays_visible_without_model_controls(window):
    assert not window.play_controls.isHidden()
    window.clear_play_controls()
    assert not window.play_controls.isHidden()
    assert window.play_controls.minimumHeight() >= 100

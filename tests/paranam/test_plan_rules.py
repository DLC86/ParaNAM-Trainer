import pytest

from nam.capture.output_gain import initialize_calibration
from nam.capture.params import KnobSpec, SwitchSpec
from nam.capture.planner import plan_captures, rule_capture_limits
from nam.capture.plan_rules import PlanRule
from nam.capture.project import add_corner_captures, new_project
from paranam.workspace import Workspace, migrate_knob_limits


def test_plan_rules_filter_neural_captures_but_not_volume_calibration():
    knobs = [
        KnobSpec("Bass", 0, 3, 1, 1, avoid_above=2),
        KnobSpec("Treble", 0, 3, 1, 1),
        KnobSpec("Volume", 0, 2, 1, 1, clean_volume=True),
    ]
    rule = PlanRule(controls=("Bass", "Treble"), operator=">", value=1)
    assert rule_capture_limits(knobs, [rule])[:2] == (1, 10)
    project = new_project(knobs, n_train=5, n_validation=3, plan_rules=[rule],
                          validation_method="farthest")
    add_corner_captures(project)
    assert any(e.params["Treble"] > 1 for e in project.entries)
    assert all(e.params["Bass"] <= 2 and not rule.excludes(e.params) for e in project.entries)
    initialize_calibration(project)
    gain = [e for e in project.entries if e.split == "gain"]
    assert len(gain) == 3
    # These captures intentionally bypass both the threshold and the rule.
    assert all(e.params["Bass"] == 3 and e.params["Treble"] == 3 for e in gain)
    assert gain[0].params["Volume"] == 2
    assert any(rule.excludes(e.params) for e in gain)

    project.plan_rules.append(PlanRule(controls=("Volume",), operator="<", value=1))
    initialize_calibration(project)
    assert {e.params["Volume"] for e in project.entries if e.split == "gain"} == {0, 1, 2}

    project.plan_rules.append(PlanRule(controls=("Volume",), operator="=", value=1))
    initialize_calibration(project)
    assert {e.params["Volume"] for e in project.entries if e.split == "gain"} == {0, 1, 2}


def test_volume_calibration_sets_other_knobs_to_10_even_when_avoided():
    knobs = [
        KnobSpec("Gain", 0, 10, 1, 5),
        KnobSpec("Bass", 0, 10, 1, 5),
        KnobSpec("Treble", 0, 10, 1, 5, avoid_above=8),
        KnobSpec("Volume", 0, 10, 1, 5, clean_volume=True),
    ]
    rule = PlanRule(controls=("Gain", "Volume"), operator=">", value=7)
    project = new_project(knobs, n_train=1, n_validation=0, plan_rules=[rule])
    initialize_calibration(project)
    gain = project.entries_for_split("gain")
    assert len(gain) == 11
    assert all(
        e.params["Gain"] == 10 and e.params["Bass"] == 10
        and e.params["Treble"] == 10 for e in gain
    )
    assert any(rule.excludes(e.params) for e in gain)
    assert {e.params["Volume"] for e in gain} == set(range(11))


def test_switch_rule_excludes_position_by_label():
    switches = [SwitchSpec("Channel", ("Clean", "1", "Lead"))]
    rule = PlanRule(controls=("Channel",), operator="=", value="1")
    train, validation = plan_captures(switches, n_train=2, n_validation=0, rules=[rule])
    assert [entry.params["Channel"] for entry in train] == ["Clean", "Lead"]
    assert not validation
    with pytest.raises(ValueError, match="exclude the default"):
        plan_captures(
            switches, n_train=2, n_validation=0,
            rules=[rule.model_copy(update={"value": "Clean"})],
        )


def test_legacy_endpoint_exclusions_produce_same_plan():
    old = KnobSpec("Drive", -0.5, 2.5, 0.5, 1, avoid_zero=True, avoid_max=True)
    new = KnobSpec.from_dict(migrate_knob_limits(old.to_dict()))
    assert new.avoid_below > old.min and new.avoid_above < old.max
    assert plan_captures([old], n_train=4, n_validation=1) == plan_captures(
        [new], n_train=4, n_validation=1
    )


def test_replan_keeps_matching_recordings_and_can_reuse_an_unused_take(workspace):
    baseline = tuple(workspace.project.entries)
    for index, entry in enumerate(baseline):
        output = workspace.directory / entry.y_path
        output.parent.mkdir(exist_ok=True)
        output.write_bytes(bytes([index]))
        entry.status = "captured"
        entry.delay = index + 2

    workspace.preferences["n_validation"] = 0
    workspace.generate_plan()
    validation = next(e for e in baseline if e.split == "validation")
    assert validation.y_path in workspace.project.unused_recordings
    assert any(e.status == "captured" for e in workspace.project.entries)

    reopened = Workspace(workspace.directory)
    reopened.preferences["n_validation"] = 1
    reopened.generate_plan()
    reused = reopened.project.entries_for_split("validation")[0]
    assert (reused.y_path, reused.status, reused.delay) == (
        validation.y_path, "captured", validation.delay,
    )
    assert len(reopened.project.captured_entries()) == len(baseline)



def test_volume_calibration_clamps_10_to_reachable_range_and_preserves_switch_end():
    controls = [
        KnobSpec("Gain", 0, 8, 1, 5, avoid_above=6),
        KnobSpec("Bias", -2, 2, 1, 0, avoid_above=1),
        SwitchSpec("Channel", ("A", "B", "C"), 0),
        KnobSpec("Volume", 0, 2, 1, 1, clean_volume=True),
    ]
    project = new_project(controls, n_train=3, n_validation=0)
    initialize_calibration(project)
    entries = project.entries_for_split("gain")
    assert len(entries) == 3
    assert all(
        e.params["Gain"] == 8 and e.params["Bias"] == 2
        and e.params["Channel"] == "C"
        for e in entries
    )



@pytest.mark.parametrize(
    "mute_flags,forbidden",
    [
        ({"mute_at_min": True}, {0.0}),
        ({"mute_at_max": True}, {10.0}),
        ({"mute_at_min": True, "mute_at_max": True}, {0.0, 10.0}),
    ],
)
def test_muted_endpoint_is_excluded_from_training_validation_and_corners(
    mute_flags, forbidden
):
    from nam.capture.planner import (
        _capture_grid_values, corner_capture_count, plan_corner_captures,
        mute_corner_settings,
    )

    controls = [
        KnobSpec("Drive", 0, 10, 1, default=5, **mute_flags),
        KnobSpec("Tone", 0, 10, 1, default=5),
        SwitchSpec("Mode", ("A", "B")),
    ]
    drive_spec = controls[0].to_param_spec()
    assert not forbidden.intersection(_capture_grid_values(drive_spec, 1))
    train, validation = plan_captures(
        controls, n_train=20, n_validation=8, seed=31
    )
    corners, _ = plan_corner_captures(controls)
    virtual = mute_corner_settings(controls)
    assert corner_capture_count(controls) == len(corners)
    assert len(virtual) == (8 if len(forbidden) == 2 else 4)
    assert all(
        any(row[k.name] == getattr(k, "min" if e == "min" else "max")
            for k in controls if hasattr(k, "mute_at_min")
            for e in ("min", "max") if getattr(k, f"mute_at_{e}"))
        for row in virtual
    )
    # Muted corners no longer require WAVs, not even at the adjacent step.
    assert not any(
        row.params["Drive"] == (1.0 if mute_flags.get("mute_at_min") else 9.0)
        for row in corners
        if mute_flags.get("mute_at_min") or mute_flags.get("mute_at_max")
    )
    assert all(
        row.params["Drive"] not in ({1.0} if mute_flags.get("mute_at_min") else set())
        | ({9.0} if mute_flags.get("mute_at_max") else set())
        for row in corners
    )
    for endpoint in ({1.0} if mute_flags.get("mute_at_min") else set()) | (
        {9.0} if mute_flags.get("mute_at_max") else set()
    ):
        assert any(row.params["Drive"] == endpoint for row in train)
    assert all(
        row.params["Drive"] not in forbidden
        for row in [*train, *validation, *corners]
    )
    assert {row.params["Mode"] for row in train} == {"A", "B"}


@pytest.mark.parametrize("edge", ["min", "max"])
def test_muted_endpoint_cannot_be_the_required_default(edge):
    controls = [
        KnobSpec(
            "Drive", 0, 10, 1,
            default=0 if edge == "min" else 10,
            **{f"mute_at_{edge}": True},
        )
    ]
    with pytest.raises(ValueError, match=f"default.*{edge}|{edge}.*default|Mute"):
        plan_captures(controls, n_train=4, n_validation=0)


def test_both_mute_edges_require_an_interior_capture_position():
    with pytest.raises(ValueError, match="No usable knob positions"):
        KnobSpec("Toggle", 0, 1, 1, mute_at_min=True, mute_at_max=True)


def test_mute_anchor_targets_ignore_switches_and_keep_both_endpoints():
    from nam.capture.export import build_hyper_a2_model_config

    project = new_project([
        KnobSpec("Drive", 0, 10, mute_at_min=True, mute_at_max=True),
        SwitchSpec("Mode", ("A", "B")),
    ], n_train=8, n_validation=2)
    model = build_hyper_a2_model_config(project)
    assert model["loss"]["mute_anchor"]["targets"] == [
        {"name": "Drive", "edge": "min"},
        {"name": "Drive", "edge": "max"},
    ]
    assert "mute_at_min" not in model["net"]["config"]["params"][0]
    assert "mute_at_max" not in model["net"]["config"]["params"][0]


def test_mute_corners_are_virtual_while_nonmuted_corners_remain_recordable():
    from nam.capture.planner import mute_corner_settings, plan_corner_captures

    controls = [
        KnobSpec("Gain", 0, 10, 0.5, 5, mute_at_min=True, is_gain=True),
        KnobSpec("Tone", 0, 10, 0.5, 5),
        KnobSpec("Presence", 0, 10, 0.5, 5),
    ]
    virtual = mute_corner_settings(controls)
    real, _ = plan_corner_captures(controls)
    assert len(virtual) == 4
    assert len(real) == 4
    assert {(r["Gain"], r["Tone"], r["Presence"]) for r in virtual} == {
        (0.0, 0.0, 0.0), (0.0, 0.0, 10.0),
        (0.0, 10.0, 0.0), (0.0, 10.0, 10.0),
    }
    assert all(row.params["Gain"] == 10.0 for row in real)


def test_sobol_reserves_only_missing_mute_transition_without_changing_sampled_points():
    from nam.capture.planner import _plan_captures_unadjusted

    controls = [
        KnobSpec("Gain", 0, 10, 1, 5, mute_at_min=True),
        KnobSpec("Tone", 0, 10, 1, 5),
        KnobSpec("Presence", 0, 10, 1, 5),
    ]
    raw, raw_val = _plan_captures_unadjusted(
        controls, n_train=8, n_validation=4, seed=17, lhs_method="sobol"
    )
    train, val = plan_captures(
        controls, n_train=8, n_validation=4, seed=17, lhs_method="sobol"
    )
    assert val == raw_val
    assert train[:len(raw)] == raw
    assert len(train) in (8, 9)
    assert any(row.params["Gain"] == 1.0 for row in train)
    assert len({tuple(row.params.values()) for row in train + val}) == len(train) + len(val)
    assert all(row.params["Gain"] != 0.0 for row in train + val)


def test_marked_mute_corner_gets_exact_virtual_target_and_adjacent_steps():
    """Two bridged gains are silent only *together*, never as separate mute knobs."""
    from nam.capture.export import build_hyper_a2_model_config
    from nam.capture.project import classify_capture

    controls = [
        KnobSpec("Gain I", 0, 10, 1, 5),
        KnobSpec("Gain II", 0, 10, 1, 5),
        KnobSpec("Tone", 0, 10, 1, 5),
    ]
    project = new_project(
        controls, n_train=1, n_validation=0, lhs_method="sobol"
    )
    add_corner_captures(project)
    corner = next(
        row for row in project.entries if row.is_corner
        and row.params == {"Gain I": 0.0, "Gain II": 0.0, "Tone": 0.0}
    )
    marked, added = classify_capture(project, corner, "mute")
    assert not project.muted_settings  # original not mutated on success
    assert marked.muted_settings == [corner.params]
    assert len(added) == 3
    assert {tuple(row.params.values()) for row in added} == {
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    }
    assert all(row.status == "pending" for row in added)
    model = build_hyper_a2_model_config(marked)
    anchors = model["loss"]["mute_anchor"]
    assert anchors["targets"] == []
    assert anchors["corner_targets"] == []
    assert anchors["point_targets"] == [corner.params]
    assert any(row.y_path == corner.y_path for row in marked.entries)
    assert marked.classification_for_entry(
        next(row for row in marked.entries if row.y_path == corner.y_path)
    ) == "mute"


def test_mute_corner_does_not_duplicate_an_already_planned_adjacent_step():
    from nam.capture.project import classify_capture

    controls = [
        KnobSpec("Gain I", 0, 10, 1, 5),
        KnobSpec("Gain II", 0, 10, 1, 5),
    ]
    project = new_project(controls, n_train=1, n_validation=0)
    add_corner_captures(project)
    first = project.entries_for_split("train")[0]
    first.params = {"Gain I": 1.0, "Gain II": 5.0}
    corner = next(
        row for row in project.entries if row.is_corner
        and row.params == {"Gain I": 0.0, "Gain II": 0.0}
    )
    new, adjacent = classify_capture(project, corner, "mute")
    assert len(adjacent) == 1
    assert adjacent[0].params == {"Gain I": 0.0, "Gain II": 1.0}
    assert new.muted_settings == [corner.params]


@pytest.mark.parametrize("classification", ["mute", "avoid"])
def test_dynamic_replacement_preserves_held_out_counts_and_existing_audio(classification):
    from nam.capture.project import classify_capture, CaptureProject

    controls = [
        KnobSpec("Gain I", 0, 10, 1, 5),
        KnobSpec("Gain II", 0, 10, 1, 5),
    ]
    project = new_project(
        controls, n_train=16, n_validation=8, seed=11,
        lhs_method="sobol", include_default_as_additional_capture=True,
    )
    original = project.entries_for_split("train")[0]
    separate = project.entries_for_split("train")[1]
    separate.status = "captured"
    count = len(project.entries)
    updated, added = classify_capture(project, original, classification)
    assert len(updated.entries) == count + int(classification == "avoid")
    assert len(added) == int(classification == "avoid")
    if classification == "avoid":
        # Unstable must be excluded, never reintroduced as its own replacement.
        assert added[0].params != original.params
        keys = [tuple(sorted(e.params.items())) for e in updated.entries
                if e.split in ("train", "validation")]
        assert len(keys) == len(set(keys))
    assert updated.entries_for_split("validation") == project.entries_for_split("validation")
    assert any(e.y_path == separate.y_path and e.status == "captured"
               for e in updated.entries)
    assert any(e.y_path == original.y_path and e.params == original.params
               for e in updated.entries)
    assert updated.classification_for_entry(
        next(e for e in updated.entries if e.y_path == original.y_path)
    ) == classification
    assert updated.muted_settings == ([original.params] if classification == "mute" else [])
    assert updated.avoided_settings == ([original.params] if classification == "avoid" else [])
    assert CaptureProject.model_validate(
        updated.model_dump(mode="json")
    ).model_dump(mode="json") == updated.model_dump(mode="json")


def test_classifying_already_captured_entry_keeps_wav_as_unused_recording():
    from nam.capture.project import classify_capture

    project = new_project(
        [KnobSpec("Gain", 0, 10, 1, 5),
         KnobSpec("Tone", 0, 10, 1, 5)],
        n_train=8, n_validation=2,
    )
    captured = project.entries_for_split("train")[0]
    captured.status = "captured"
    updated, _ = classify_capture(project, captured, "avoid")
    assert captured.y_path in updated.unused_recordings
    assert updated.unused_recordings[captured.y_path]["context"] is None
    marked = next(row for row in updated.entries if row.y_path == captured.y_path)
    assert marked.status == "captured"
    assert updated.classification_for_entry(marked) == "avoid"
    assert marked not in updated.captured_entries()
    from nam.capture.export import build_data_config
    assert not any(
        sample["y_path"] == captured.y_path
        for samples in (build_data_config(updated)["train"],
                        build_data_config(updated)["validation"])
        for sample in samples
    )


def test_avoiding_a_corner_does_not_infer_mute_or_new_safe_boundaries():
    from nam.capture.project import classify_capture

    controls = [KnobSpec("Gain", 0, 10, 1, 5),
                KnobSpec("Tone", 0, 10, 1, 5)]
    project = new_project(controls, n_train=1, n_validation=0)
    add_corner_captures(project)
    corner = next(row for row in project.entries if row.is_corner)
    updated, added = classify_capture(project, corner, "avoid")
    assert len(added) == 1 and not updated.muted_settings
    assert added[0].split == "train"
    assert added[0].params != corner.params
    assert updated.avoided_settings == [corner.params]


def test_rerecorded_capture_clears_classification_but_retains_new_samples():
    from nam.capture.project import classify_capture
    from nam.capture.export import build_data_config, mute_anchor_point_targets

    knobs = [KnobSpec("Gain I", 0, 10, 1, 5),
             KnobSpec("Gain II", 0, 10, 1, 5)]
    project = new_project(knobs, n_train=1, n_validation=0)
    add_corner_captures(project)
    original = next(
        e for e in project.entries
        if e.is_corner and e.params == {"Gain I": 0.0, "Gain II": 0.0}
    )
    classified, added = classify_capture(project, original, "mute")
    assert not classified.capture_mark_history
    assert len(added) == 2
    original_row = next(
        e for e in classified.entries if e.y_path == original.y_path
    )
    assert classified.classification_for_entry(original_row) == "mute"
    assert original_row not in classified.pending_entries()
    assert mute_anchor_point_targets(classified) == [original.params]

    original_row.status = "captured"
    assert original_row not in classified.captured_entries()
    classified.clear_capture_classification(original_row)
    assert classified.classification_for_entry(original_row) is None
    assert original_row in classified.captured_entries()
    assert mute_anchor_point_targets(classified) == []
    assert any(row.y_path == added[0].y_path for row in classified.entries)
    assert any(
        sample["y_path"] == original.y_path
        for sample in build_data_config(classified)["train"]
    )


def test_legacy_undo_history_recovers_classified_rows_when_opened(tmp_path):
    from nam.capture.project import classify_capture, save_project, load_project

    project = new_project(
        [KnobSpec("Gain", 0, 10, 1, 5),
         KnobSpec("Tone", 0, 10, 1, 5)],
        n_train=4, n_validation=2,
    )
    old = project.entries[0]
    classified, _ = classify_capture(project, old, "mute")
    legacy = classified.model_copy(deep=True)
    row = next(e for e in legacy.entries if e.y_path == old.y_path)
    position = legacy.entries.index(row)
    legacy.entries.remove(row)
    legacy.capture_mark_history.append({
        "classification": "mute", "entry": old.model_dump(mode="json"),
        "position": position, "added_paths": [],
    })
    save_project(legacy, tmp_path)
    recovered = load_project(tmp_path)
    assert recovered.capture_mark_history == []
    restored = next(e for e in recovered.entries if e.y_path == old.y_path)
    assert recovered.classification_for_entry(restored) == "mute"
    assert restored.params in recovered.muted_settings


@pytest.mark.parametrize("split", ["train", "validation"])
def test_noncorner_mute_is_virtual_and_never_adds_a_generic_capture(split):
    from nam.capture.project import classify_capture
    from nam.capture.export import mute_anchor_point_targets

    project = new_project(
        [KnobSpec("Gain", 0, 10, 1, 5),
         KnobSpec("Tone", 0, 10, 1, 5)],
        n_train=12, n_validation=4, lhs_method="sobol",
    )
    original = project.entries_for_split(split)[0]
    previous = [(e.y_path, e.params) for e in project.entries]
    marked, added = classify_capture(project, original, "mute")
    assert added == []
    assert [(e.y_path, e.params) for e in marked.entries] == previous
    assert marked.muted_settings == [original.params]
    assert mute_anchor_point_targets(marked) == [original.params]
    assert marked.classification_for_entry(
        next(e for e in marked.entries if e.y_path == original.y_path)
    ) == "mute"
    assert original not in marked.pending_entries()


def test_rule_conditions_can_have_different_thresholds_and_switch_positions():
    from nam.capture.plan_rules import (
        allows, allowed_grid_size, bound_rules, validate_plan_rules,
    )

    controls = [
        KnobSpec("Gain", 0, 10, step=1, default=5),
        KnobSpec("Bias", 0, 10, step=1, default=5),
        SwitchSpec("Mode", ("Normal", "Boost"), planning_mode="simple"),
    ]
    rule = PlanRule(
        controls=("Gain", "Bias", "Mode"), operator=">", value=8,
        operators={"Bias": ">", "Mode": "="},
        thresholds={"Bias": 7, "Mode": "Boost"},
    )
    validate_plan_rules([c.to_param_spec() for c in controls], [rule])
    assert rule.excludes({"Gain": 9, "Bias": 8, "Mode": "Boost"})
    assert not rule.excludes({"Gain": 9, "Bias": 7, "Mode": "Boost"})
    assert not rule.excludes({"Gain": 8, "Bias": 9, "Mode": "Boost"})
    assert not rule.excludes({"Gain": 9, "Bias": 9, "Mode": "Normal"})
    assert allows([rule], {"Gain": 9, "Bias": 7, "Mode": "Boost"})

    # The grid leaves 2 gain positions x 3 bias positions in Boost excluded.
    axes = [(c.name, range(11) if isinstance(c, KnobSpec)
             else c.enum_names) for c in controls]
    assert allowed_grid_size(axes, [rule]) == 11 * 11 * 2 - 2 * 3
    active = bound_rules([rule], {"Mode": "Boost"})
    assert len(active) == 1
    assert active[0].controls == ("Gain", "Bias")
    assert active[0].excludes({"Gain": 9, "Bias": 8})
    assert bound_rules([rule], {"Mode": "Normal"}) == ()
    assert PlanRule.model_validate(rule.model_dump()).model_dump() == rule.model_dump()

    # A legacy single-threshold rule continues to work as before.
    legacy = PlanRule(controls=("Gain", "Bias"), operator=">", value=8)
    assert legacy.excludes({"Gain": 9, "Bias": 9})
    assert not legacy.excludes({"Gain": 9, "Bias": 8})


def test_different_thresholds_change_sampled_plan_and_exclusion_capacity():
    controls = [
        KnobSpec("Gain", 0, 10, step=1, default=5),
        KnobSpec("Bias", 0, 10, step=1, default=5),
    ]
    rule = PlanRule(
        controls=("Gain", "Bias"), operator=">", value=8,
        thresholds={"Bias": 7},
    )
    limit = rule_capture_limits(controls, [rule], n_train=24, n_validation=6)
    assert limit[0] == 1 and limit[1] >= 24 and limit[2] >= 6
    train, validation = plan_captures(
        controls, n_train=24, n_validation=6, rules=[rule],
        lhs_method="sobol", include_default_as_additional_capture=True,
    )
    assert len(train) == 25 and len(validation) == 6
    assert all(not rule.excludes(entry.params) for entry in train + validation)


def test_avoid_rule_editor_supports_different_controls_and_roundtrip(qapp):
    from paranam.plan_rules_ui import PlanRuleDialog, describe_rule

    controls = [
        KnobSpec("Gain", 0, 10, step=0.5, default=5),
        KnobSpec("Bias", 0, 10, step=0.5, default=5),
        SwitchSpec("Mode", ("Normal", "Boost"), planning_mode="simple"),
    ]
    dialog = PlanRuleDialog(None, controls)
    for name in ("Gain", "Bias", "Mode"):
        dialog.rows[name][0].setChecked(True)
    dialog.rows["Gain"][1].setCurrentText(">")
    dialog.rows["Gain"][2].setValue(8)
    dialog.rows["Bias"][1].setCurrentText(">")
    dialog.rows["Bias"][2].setValue(7)
    dialog.rows["Mode"][2].setCurrentText("Boost")
    dialog.accept()
    rule = dialog.rule
    assert rule is not None
    assert rule.excludes({"Gain": 9, "Bias": 8, "Mode": "Boost"})
    assert not rule.excludes({"Gain": 9, "Bias": 7, "Mode": "Boost"})
    assert describe_rule(rule) == "Avoid when Gain > 8 AND Bias > 7 AND Mode = Boost"

    reloaded = PlanRule.model_validate(rule.model_dump(mode="json"))
    editor = PlanRuleDialog(None, controls, reloaded)
    assert editor.rows["Gain"][2].value() == 8
    assert editor.rows["Bias"][2].value() == 7
    assert editor.rows["Mode"][2].currentText() == "Boost"
    editor.accept()
    assert editor.rule == rule
    dialog.close()
    editor.close()


def test_control_rename_preserves_mixed_rule_conditions(workspace):
    knobs = [KnobSpec("Gain", 0, 10, default=5),
             KnobSpec("Bias", 0, 10, default=5)]
    workspace.draft_controls = [c.to_dict() for c in knobs]
    workspace.preferences["draft_plan_rules"] = [PlanRule(
        controls=("Gain", "Bias"), operator=">", value=8,
        thresholds={"Bias": 7},
    ).model_dump(mode="json")]
    workspace.rename_control("Bias", "Tone")
    rule = workspace.plan_rules()[0]
    assert rule.controls == ("Gain", "Tone")
    assert rule.thresholds == {"Tone": 7}
    assert rule.excludes({"Gain": 9, "Tone": 8})
    assert not rule.excludes({"Gain": 9, "Tone": 7})


def test_mute_corner_transition_ignores_other_classified_training_points():
    from nam.capture.project import classify_capture

    controls = [
        KnobSpec("Gain I", 0, 10, 1, 5),
        KnobSpec("Gain II", 0, 10, 1, 5),
    ]
    project = new_project(controls, n_train=1, n_validation=0)
    add_corner_captures(project)
    original = project.entries_for_split("train")[0]
    original.params = {"Gain I": 1.0, "Gain II": 5.0}
    with_mute, replacement = classify_capture(project, original, "mute")
    assert replacement == []
    corner = next(
        row for row in with_mute.entries if row.is_corner
        and row.params == {"Gain I": 0.0, "Gain II": 0.0}
    )
    marked, transitions = classify_capture(with_mute, corner, "mute")
    assert {"Gain I": 1.0, "Gain II": 0.0} in [
        row.params for row in transitions
    ]
    assert marked.muted_settings == [original.params, corner.params]



def test_conditional_mute_rule_covers_only_jointly_muted_gain_region():
    from nam.capture.export import (
        build_hyper_a2_model_config, mute_anchor_rule_targets,
    )
    controls = [
        KnobSpec("Gain I", 0, 10, 1, 5),
        KnobSpec("Gain II", 0, 10, 1, 5),
        KnobSpec("Tone", 0, 10, 1, 5),
    ]
    rule = PlanRule(
        controls=("Gain I", "Gain II"), operator="=", value=0,
        kind="mute",
    )
    project = new_project(
        controls, n_train=12, n_validation=4, seed=9,
        plan_rules=[rule], lhs_method="sobol",
    )
    assert all(not rule.excludes(e.params) for e in project.entries)
    assert not rule.excludes({"Gain I": 0, "Gain II": 5, "Tone": 0})
    assert not rule.excludes({"Gain I": 5, "Gain II": 0, "Tone": 0})
    assert mute_anchor_rule_targets(project) == [rule.model_dump(mode="json")]
    loss = build_hyper_a2_model_config(project)["loss"]["mute_anchor"]
    assert loss["rule_targets"] == [rule.model_dump(mode="json")]
    assert loss["targets"] == []


def test_avoid_rules_remain_backward_compatible_without_explicit_kind():
    old = {"controls": ["Gain", "Tone"], "operator": "<", "value": 2}
    rule = PlanRule.model_validate(old)
    assert rule.kind == "avoid"
    assert rule.excludes({"Gain": 1, "Tone": 0})
    assert not rule.excludes({"Gain": 1, "Tone": 9})



def test_rule_editor_mute_kind_roundtrips_and_rejects_empty_region(qapp):
    from paranam.plan_rules_ui import PlanRuleDialog, describe_rule
    from nam.capture.plan_rules import validate_plan_rules

    controls = [
        KnobSpec("Gain I", 0, 10, 1, 5),
        KnobSpec("Gain II", 0, 10, 1, 5),
    ]
    dialog = PlanRuleDialog(None, controls)
    dialog.rule_kind.setCurrentIndex(dialog.rule_kind.findData("mute"))
    for name in ("Gain I", "Gain II"):
        dialog.rows[name][0].setChecked(True)
        dialog.rows[name][1].setCurrentText("=")
        dialog.rows[name][2].setValue(0)
    dialog.accept()
    assert dialog.rule.kind == "mute"
    assert describe_rule(dialog.rule) == "Mute when Gain I = 0 AND Gain II = 0"
    reloaded = PlanRuleDialog(None, controls, PlanRule.model_validate(
        dialog.rule.model_dump(mode="json")
    ))
    assert reloaded.rule_kind.currentData() == "mute"
    dialog.close()
    reloaded.close()

    impossible = PlanRule(
        controls=("Gain I",), operator="<", value=0, kind="mute"
    )
    with pytest.raises(ValueError, match="no valid setting"):
        validate_plan_rules([controls[0].to_param_spec()], [impossible])

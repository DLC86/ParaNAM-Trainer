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
        _capture_grid_values, corner_capture_count, plan_corner_captures
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
    assert corners and corner_capture_count(controls) == len(corners)
    assert any(row.params["Drive"] == 1.0 for row in corners) == bool(
        mute_flags.get("mute_at_min")
    )
    assert any(row.params["Drive"] == 9.0 for row in corners) == bool(
        mute_flags.get("mute_at_max")
    )
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

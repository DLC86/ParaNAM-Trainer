import pytest

from nam.capture.output_gain import initialize_calibration
from nam.capture.params import KnobSpec, SwitchSpec
from nam.capture.planner import plan_captures, rule_capture_limits
from nam.capture.plan_rules import PlanRule
from nam.capture.project import add_corner_captures, new_project
from paranam.workspace import Workspace, migrate_knob_limits


def test_plan_rules_filter_both_splits_and_corners_except_volume_calibration():
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
    assert gain and all(e.params["Bass"] == e.params["Treble"] == 3 for e in gain)


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

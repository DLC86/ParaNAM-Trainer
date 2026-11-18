"""Capture control order is independent of sampling and legacy WAV filenames."""

from nam.capture.output_gain import initialize_calibration
from nam.capture.params import KnobSpec
from nam.capture.planner import plan_captures
from nam.capture.plan_rules import PlanRule
from nam.capture.project import add_corner_captures, new_project
from paranam.workspace import Workspace


def controls_with_volume_first():
    # Linear Volume is deliberately FIRST, while the sampling core treats it
    # as a fixed parameter and previously appended it LAST.
    return [
        KnobSpec("Volume", 0, 10, step=1, default=5, clean_volume=True),
        KnobSpec("Tone", 0, 10, step=1, default=5),
        KnobSpec("Drive", 0, 10, step=1, default=5),
    ]


def assert_control_order(entries):
    for row in entries:
        assert list(row.params) == ["Volume", "Tone", "Drive"]
        name = row.y_path.rsplit("/", 1)[-1]
        assert name.index("_V10_") < name.index("_T") < name.index("_D")


def test_linear_volume_filenames_follow_rig_order_with_or_without_rules():
    controls = controls_with_volume_first()
    for rules in ((), (PlanRule(controls=("Tone",), operator=">", value=9),)):
        train, validation = plan_captures(
            controls, n_train=5, n_validation=3, seed=13,
            lhs_method="sobol", rules=rules,
            include_default_as_additional_capture=True,
        )
        assert len(train) >= 5 and len(validation) == 3
        assert_control_order(train + validation)


def test_old_recorded_volume_filename_survives_reopen_replan_and_ui(
    workspace, window,
):
    workspace.draft_controls = [c.to_dict() for c in controls_with_volume_first()]
    workspace.preferences.update(n_train=5, n_validation=3, corners=False)
    workspace.generate_plan()
    assert_control_order(workspace.project.entries_for_split("validation"))
    assert_control_order(workspace.project.entries_for_split("train"))

    entry = next(e for e in workspace.project.entries if e.split == "train")
    # Simulate the exact legacy defect: Volume appears last inside both the
    # stored params and the existing audio filename.
    tone, drive = entry.params["Tone"], entry.params["Drive"]
    old_path = (
        f"captures/lhs_legacy_T{tone:g}_D{drive:g}_V10.wav"
    )
    entry.y_path = old_path
    entry.params = {"Tone": tone, "Drive": drive, "Volume": 10.0}
    entry.status = "captured"
    entry.delay = 314
    wav = workspace.directory / old_path
    wav.parent.mkdir(exist_ok=True)
    wav.write_bytes(b"untouched audio")
    workspace.save()

    reopened = Workspace(workspace.directory)
    stored = next(e for e in reopened.project.entries if e.y_path == old_path)
    assert list(stored.params) == ["Tone", "Drive", "Volume"]
    assert (stored.status, stored.delay) == ("captured", 314)

    window._attach(reopened)
    window.refresh_captures()
    row = next(i for i, e in enumerate(window._capture_entries) if e.y_path == old_path)
    visible = window.capture_table.item(row, 1).text()
    assert visible.startswith("Volume=10, Tone=")
    assert visible.index("Tone=") < visible.index("Drive=")

    # The new planner's filename convention must not rename or invalidate
    # a captured WAV when regenerating the *same* settings.
    reopened.generate_plan()
    reused = next(e for e in reopened.project.entries if e.y_path == old_path)
    assert reused.status == "captured"
    assert reused.delay == 314
    assert wav.read_bytes() == b"untouched audio"
    assert all(
        not (e.y_path != old_path and e.split == "train" and e.params == reused.params)
        for e in reopened.project.entries
    )


def test_all_capture_types_follow_middle_volume_position():
    controls = [
        KnobSpec("Tone", 0, 10, step=1, default=5),
        KnobSpec("Volume", 0, 10, step=1, default=5, clean_volume=True),
        KnobSpec("Drive", 0, 10, step=1, default=5),
    ]
    rules = [PlanRule(controls=("Tone",), operator=">", value=9)]
    project = new_project(
        controls, n_train=5, n_validation=3, seed=13,
        lhs_method="sobol", plan_rules=rules,
        include_default_as_additional_capture=True,
    )
    add_corner_captures(project)
    initialize_calibration(project)
    assert {e.split for e in project.entries} == {"train", "validation", "gain"}
    assert any(e.is_corner for e in project.entries)
    for entry in project.entries:
        assert list(entry.params) == ["Tone", "Volume", "Drive"]
        name = entry.y_path.rsplit("/", 1)[-1]
        assert name.index("_T") < name.index("_V") < name.index("_D")

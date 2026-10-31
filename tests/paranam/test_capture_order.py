from itertools import product
import random

from nam.capture.params import KnobSpec, SwitchSpec
from nam.capture.project import CaptureEntryModel
from paranam.capture_order import order_captures


def entry(index, params, **kwargs):
    return CaptureEntryModel(
        index=index,
        params=params,
        y_path=f"captures/{index}.wav",
        split="train",
        **kwargs,
    )


def changes(entries, anchor):
    total = 0
    for capture in entries:
        total += sum(capture.params[k] != value for k, value in anchor.items())
        anchor = capture.params
    return total


def test_single_knob_follows_short_monotonic_path_without_changing_entries():
    controls = [KnobSpec("Gain", 0, 10, default=0)]
    original = [entry(i, {"Gain": x}) for i, x in enumerate([10, 0, 8, 2, 6, 4])]
    snapshot = [e.model_dump() for e in original]
    ordered = order_captures(original, controls)
    assert [e.params["Gain"] for e in ordered] == [0, 2, 4, 6, 8, 10]
    assert [e.model_dump() for e in original] == snapshot
    assert {id(e) for e in ordered} == {id(e) for e in original}


def test_knob_rotation_is_normalized_to_each_control_range():
    controls = [KnobSpec("A", 0, 100000, default=0), KnobSpec("B", 0, 1, default=0)]
    original = [entry(0, {"A": 0, "B": 1}), entry(1, {"A": 10000, "B": 0})]
    assert order_captures(original, controls)[0] is original[1]


def test_grid_with_multiple_switches_keeps_every_combination_and_reduces_changes():
    controls = [
        KnobSpec("Gain", 0, 2, default=0),
        KnobSpec("Tone", 0, 2, default=0),
        SwitchSpec("Mode", ("Clean", "Lead")),
        SwitchSpec("Bright", ("Off", "Normal", "Boost")),
    ]
    original = [
        entry(i, dict(zip((c.name for c in controls), values)))
        for i, values in enumerate(
            product(range(3), range(3), controls[2].enum_names, controls[3].enum_names)
        )
    ]
    anchor = {"Gain": 0, "Tone": 0, "Mode": "Clean", "Bright": "Off"}
    ordered = order_captures(original, controls)
    assert len(ordered) == 54
    assert {e.y_path for e in ordered} == {e.y_path for e in original}
    assert changes(ordered, anchor) < changes(original, anchor)
    assert order_captures(original, controls) == ordered


def test_resume_starts_at_last_recorded_state_and_pairs_validation_with_training():
    controls = [KnobSpec("Gain", 0, 10, default=0), SwitchSpec("Mode", ("A", "B"))]
    pending = [
        entry(0, {"Gain": 0, "Mode": "A"}),
        entry(1, {"Gain": 10, "Mode": "B"}),
        entry(2, {"Gain": 10, "Mode": "B"}),
    ]
    pending[2].split = "validation"
    last = entry(
        3,
        {"Gain": 10, "Mode": "B"},
        status="captured",
        captured_at="2026-10-31T12:01:00+00:00",
    )
    earlier = entry(
        4,
        {"Gain": 0, "Mode": "A"},
        status="captured",
        captured_at="2026-10-31T12:00:00+00:00",
    )
    ordered = order_captures([last, *pending, earlier], controls)
    assert ordered == [earlier, last, pending[1], pending[2], pending[0]]
    # New AL entries participate without renumbering existing capture paths.
    suggestion = entry(5, {"Gain": 10, "Mode": "B"})
    assert order_captures([last, *pending, earlier, suggestion], controls)[2:5] == [
        pending[1],
        pending[2],
        suggestion,
    ]


def test_random_routes_never_increase_control_changes_against_original_plan():
    controls = [KnobSpec(name, 0, 4, default=0) for name in ("A", "B", "C")]
    controls.append(SwitchSpec("Mode", ("Off", "On")))
    anchor = {"A": 0, "B": 0, "C": 0, "Mode": "Off"}
    for seed in range(20):
        rng = random.Random(seed)
        original = [
            entry(
                i,
                {
                    "A": rng.randrange(5),
                    "B": rng.randrange(5),
                    "C": rng.randrange(5),
                    "Mode": rng.choice(("Off", "On")),
                },
            )
            for i in range(30)
        ]
        ordered = order_captures(original, controls)
        assert changes(ordered, anchor) <= changes(original, anchor)
        assert len(ordered) == len(original)
        assert {id(e) for e in ordered} == {id(e) for e in original}


def test_empty_and_finished_plans_have_no_remaining_work():
    controls = [KnobSpec("Gain", 0, 10)]
    assert order_captures([], controls) == []
    done = entry(0, {"Gain": 5}, status="captured")
    assert order_captures([done], controls) == [done]

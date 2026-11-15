"""Recommended and manually editable counts stay within the real capture grid."""

from collections import Counter

from nam.capture.params import (
    KnobSpec,
    SwitchSpec,
    capture_grid_limits,
    capture_count_maxima,
    capture_recommendations,
)
from nam.capture.planner import corner_settings, plan_captures


def _keys(entries):
    return {tuple(entry.params.items()) for entry in entries}


def test_coarse_knob_with_complex_switch_recommends_feasible_splits():
    controls = [
        KnobSpec("Gain", 0, 10, step=0.5, avoid_zero=True),
        SwitchSpec("Bright", ("OFF", "ON")),
    ]
    assert capture_grid_limits(controls) == (20, 17, 2)
    assert capture_recommendations(controls) == (32, 8, 2)

    train, val = plan_captures(controls, n_train=32, n_validation=8, seed=13)
    assert len(train) == 32 and len(val) == 8
    assert Counter(e.params["Bright"] for e in train) == {
        "OFF": 16, "ON": 16
    }
    assert Counter(e.params["Bright"] for e in val) == {
        "OFF": 4, "ON": 4
    }
    assert len(_keys(train)) == 32
    assert len(_keys(val)) == 8
    assert not _keys(train) & _keys(val)
    corners = corner_settings(tuple(c.to_param_spec() for c in controls))
    assert not _keys(val) & {tuple(p.items()) for p in corners}


def test_mute_endpoints_limit_capacity_just_like_legacy_avoid_flags():
    muted = [KnobSpec(
        "Drive", 0, 10, step=0.5, default=5,
        mute_at_min=True, mute_at_max=True,
    )]
    legacy = [KnobSpec(
        "Drive", 0, 10, step=0.5, default=5,
        avoid_zero=True, avoid_max=True,
    )]
    assert capture_grid_limits(muted) == capture_grid_limits(legacy)
    assert capture_recommendations(muted) == capture_recommendations(legacy)
    assert capture_count_maxima(muted, 10, 4) == capture_count_maxima(
        legacy, 10, 4
    )
    assert corner_settings(
        tuple(s.to_param_spec() for s in muted)
    ) == corner_settings(tuple(s.to_param_spec() for s in legacy))
    assert plan_captures(muted, n_train=10, n_validation=4, seed=7) == plan_captures(
        legacy, n_train=10, n_validation=4, seed=7
    )


def test_grid_limits_hold_out_corner_and_default_settings():
    knob = KnobSpec("Tone", 0, 1, step=0.5)
    assert capture_grid_limits([knob]) == (3, 0, 1)
    assert capture_recommendations([knob]) == (3, 0, 1)
    train, val = plan_captures([knob], n_train=3, n_validation=0)
    assert len(train) == 3 and val == []

    slightly_finer = KnobSpec("Tone", 0, 1, step=0.25)
    assert capture_grid_limits([slightly_finer]) == (5, 2, 1)
    assert capture_recommendations([slightly_finer]) == (4, 1, 1)
    train, val = plan_captures(
        [slightly_finer], n_train=4, n_validation=1
    )
    assert len(train) == 4 and len(val) == 1


def test_manual_maxima_are_joint_and_respect_state_remainders():
    controls = [
        KnobSpec("Tone", 0, 10, step=0.5, avoid_zero=True),
        SwitchSpec("Bright", ("OFF", "ON")),
    ]
    # Both remainders are put into the first switch state. 31+9 would
    # overfill OFF even though 40 settings exist in total.
    assert capture_count_maxima(controls, 31, 9) == (30, 8)
    assert capture_count_maxima(controls, 32, 8) == (32, 8)
    assert capture_count_maxima(controls, 36, 0) == (40, 4)
    assert capture_count_maxima(controls, 40, 0) == (40, 0)


def test_simple_switch_uses_its_discrete_positions_as_one_dimension():
    controls = [
        KnobSpec("Tone", 0, 1, step=0.5),
        SwitchSpec("Mode", ("A", "B", "C"), planning_mode="simple"),
        SwitchSpec("Bright", ("OFF", "ON")),
    ]
    capacity, held_out, states = capture_grid_limits(controls)
    assert capacity == 9 and states == 2
    assert 0 <= held_out <= capacity
    train, val, combinations = capture_recommendations(controls)
    assert combinations == states
    assert train + val <= capacity * states
    assert capture_count_maxima(controls, train, val)[0] >= train
    assert capture_count_maxima(controls, train, val)[1] >= val


def test_switch_only_retains_separate_audio_validation_policy():
    controls = [
        SwitchSpec("Mode", ("A", "B", "C")),
        SwitchSpec("Bright", ("OFF", "ON")),
    ]
    assert capture_grid_limits(controls) == (1, 0, 6)
    assert capture_recommendations(controls) == (6, 0, 6)

"""Opt-in validation distance selection, keeping historical LHS intact."""

import numpy as np
import pytest

from nam.capture.params import KnobSpec, SwitchSpec
from nam.capture.planner import plan_captures, corner_settings
from nam.capture.project import new_project, CaptureProject


def _points(rows, names):
    return np.array(
        [[float(row.params[name]) / 10 for name in names] for row in rows],
        dtype=float,
    )


def _nearest(train, validation):
    from scipy.spatial import cKDTree

    return cKDTree(train).query(validation, k=1)[0]


def test_farthest_validation_is_reproducible_and_not_closer_than_independent():
    controls = [
        KnobSpec("Drive", 0, 10, step=0.5, default=5),
        KnobSpec("Bass", 0, 10, step=0.5, default=5),
        KnobSpec("Treble", 0, 10, step=0.5, default=5),
    ]
    params = dict(n_train=30, n_validation=12, seed=41, lhs_method="standard")
    train_a, val_a = plan_captures(controls, **params, validation_method="farthest")
    train_b, val_b = plan_captures(controls, **params, validation_method="farthest")
    assert train_a == train_b
    assert val_a == val_b
    baseline_train, baseline_val = plan_captures(controls, **params)
    expected_corners = corner_settings(tuple(control.to_param_spec() for control in controls))
    corners = np.array([
        [setting[name] / 10 for name in ("Drive", "Bass", "Treble")]
        for setting in expected_corners
    ])
    fartrain = np.concatenate([_points(train_a, ("Drive", "Bass", "Treble")), corners])
    baseline = _nearest(
        np.concatenate([_points(baseline_train, ("Drive", "Bass", "Treble")), corners]),
        _points(baseline_val, ("Drive", "Bass", "Treble")),
    )
    farther = _nearest(fartrain, _points(val_a, ("Drive", "Bass", "Treble")))
    assert np.min(farther) > np.min(baseline)
    assert len(set(tuple(e.params.items()) for e in train_a + val_a)) == 42
    corner_keys = {tuple(sorted(setting.items())) for setting in expected_corners}
    assert corner_keys.isdisjoint(tuple(sorted(e.params.items())) for e in val_a)


def test_farthest_validation_preserves_complex_switch_coverage():
    controls = [
        KnobSpec("Drive", 0, 10, step=0.5),
        KnobSpec("Tone", 0, 10, step=0.5),
        SwitchSpec("Mode", ("A", "B"), planning_mode="complex"),
    ]
    train, validation = plan_captures(
        controls, n_train=24, n_validation=8, seed=11,
        validation_method="farthest",
    )
    assert len(train) == 24
    assert len(validation) == 8
    assert sum(e.params["Mode"] == "A" for e in validation) == 4
    assert sum(e.params["Mode"] == "B" for e in validation) == 4
    assert set(tuple(e.params.items()) for e in train).isdisjoint(
        tuple(e.params.items()) for e in validation
    )


def test_farthest_validation_preserves_simple_switch_balance():
    controls = [
        KnobSpec("Drive", 0, 10, step=0.5),
        SwitchSpec("Mode", ("A", "B", "C"), planning_mode="simple"),
    ]
    _, validation = plan_captures(
        controls, n_train=21, n_validation=9, seed=19,
        validation_method="farthest",
    )
    assert sorted(sum(e.params["Mode"] == state for e in validation)
                  for state in ("A", "B", "C")) == [3, 3, 3]


def test_project_defaults_to_historical_validation_and_serializes_selection():
    controls = [KnobSpec("Drive", 0, 10), KnobSpec("Tone", 0, 10)]
    old = new_project(controls, n_train=12, n_validation=5, seed=9)
    assert old.validation_method == "independent"
    original = old.model_dump()
    original.pop("validation_method")
    restored = CaptureProject.model_validate(original)
    assert restored.validation_method == "independent"
    updated = new_project(
        controls, n_train=12, n_validation=5, seed=9, validation_method="farthest"
    )
    assert CaptureProject.model_validate_json(updated.model_dump_json()).validation_method == "farthest"


def test_invalid_validation_method_rejected():
    with pytest.raises(ValueError, match="Unknown validation"):
        plan_captures(
            [KnobSpec("Gain", 0, 10)], n_train=5, n_validation=2,
            validation_method="typo",
        )

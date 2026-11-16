"""Regression tests for selectable LHS and Sobol capture sampling designs."""

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.stats import qmc

from nam.capture.params import KnobSpec, SwitchSpec
from nam.capture.planner import latin_hypercube_unit, plan_captures
from nam.capture.project import new_project


@pytest.mark.parametrize("method", ["standard", "maximin", "orthogonal", "near_orthogonal", "optimized"])
def test_lhs_methods_are_reproducible_and_stratified(method):
    first = latin_hypercube_unit(25, 3, seed=13, lhs_method=method)
    second = latin_hypercube_unit(25, 3, seed=13, lhs_method=method)
    np.testing.assert_array_equal(first, second)
    assert np.all((first >= 0) & (first < 1))
    for column in range(3):
        assert sorted(np.floor(first[:, column] * 25).astype(int)) == list(range(25))


@pytest.mark.parametrize("n", [1, 7, 25, 32, 64])
def test_scrambled_sobol_is_reproducible_for_arbitrary_capture_counts(n):
    first = latin_hypercube_unit(n, 3, seed=13, lhs_method="sobol")
    again = latin_hypercube_unit(n, 3, seed=13, lhs_method="sobol")
    np.testing.assert_array_equal(first, again)
    assert first.shape == (n, 3)
    assert np.all((first >= 0.0) & (first < 1.0))
    # A power-of-two Sobol net is stratified in each one-dimensional projection.
    if n in (32, 64):
        for column in range(3):
            assert sorted(np.floor(first[:, column] * n).astype(int)) == list(range(n))


def test_sobol_designs_extend_by_prefix_without_moving_existing_points():
    short = latin_hypercube_unit(25, 3, seed=17, lhs_method="sobol")
    long = latin_hypercube_unit(64, 3, seed=17, lhs_method="sobol")
    np.testing.assert_array_equal(short, long[:25])


def test_optimized_lhs_reduces_centered_discrepancy():
    standard = latin_hypercube_unit(25, 3, seed=10)
    optimized = latin_hypercube_unit(25, 3, seed=10, lhs_method="optimized")
    assert qmc.discrepancy(optimized) <= qmc.discrepancy(standard) + 1e-12


def test_maximin_is_at_least_as_spread_out_as_the_standard_candidate():
    standard = latin_hypercube_unit(25, 3, seed=10)
    maximin = latin_hypercube_unit(25, 3, seed=10, lhs_method="maximin")

    def min_pairwise(design):
        return cKDTree(design).query(design, k=2)[0][:, 1].min()

    assert min_pairwise(maximin) >= min_pairwise(standard) - 1e-12


def test_near_orthogonal_reduces_or_matches_squared_correlations():
    standard = latin_hypercube_unit(25, 3, seed=10)
    near = latin_hypercube_unit(25, 3, seed=10, lhs_method="near_orthogonal")

    def score(design):
        correlations = np.corrcoef(design, rowvar=False)
        return np.sum(np.triu(correlations**2, k=1))

    assert score(near) <= score(standard) + 1e-12


def test_orthogonal_array_balances_each_two_dimensional_projection():
    design = latin_hypercube_unit(25, 3, seed=31, lhs_method="orthogonal")
    groups = np.floor(design * 5).astype(int)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert len({tuple(row) for row in groups[:, [i, j]]}) == 25


@pytest.mark.parametrize("count,dimensions", [(16, 2), (24, 2), (25, 7)])
def test_strict_orthogonal_rejects_incompatible_counts(count, dimensions):
    with pytest.raises(ValueError, match="Orthogonal LHS"):
        latin_hypercube_unit(count, dimensions, seed=4, lhs_method="orthogonal")


def test_selected_method_reaches_capture_planning_and_project_metadata():
    knobs = [
        KnobSpec("Gain", 0, 10, step=0.1),
        KnobSpec("Tone", 0, 10, step=0.1),
    ]
    for method in ("maximin", "orthogonal", "near_orthogonal", "sobol", "optimized"):
        train, validation = plan_captures(
            knobs, n_train=25, n_validation=25, seed=7, lhs_method=method
        )
        assert len(train) == len(validation) == 25
        assert len({tuple(x.params.items()) for x in train + validation}) == 50

    project = new_project(
        knobs, n_train=25, n_validation=25, seed=7, lhs_method="maximin"
    )
    assert project.lhs_method == "maximin"
    assert project.model_dump()["lhs_method"] == "maximin"


@pytest.mark.parametrize("method", ["sobol", "optimized"])
@pytest.mark.parametrize("n_train,n_validation", [(7, 5), (25, 17)])
def test_new_samplers_anchor_and_hold_out_unique_quantized_settings(
    method, n_train, n_validation
):
    knobs = [
        KnobSpec("Gain", 0, 10, step=0.5, default=3.0),
        KnobSpec("Tone", 0, 10, step=0.5, default=7.0),
    ]
    train, validation = plan_captures(
        knobs, n_train=n_train, n_validation=n_validation, seed=19,
        lhs_method=method,
    )
    assert len(train) == n_train and len(validation) == n_validation
    keys = lambda rows: {tuple(row.params.values()) for row in rows}
    assert len(keys(train)) == n_train
    assert len(keys(validation)) == n_validation
    assert keys(train).isdisjoint(keys(validation))
    assert (3.0, 7.0) in keys(train)
    assert (3.0, 7.0) not in keys(validation)
    assert (train, validation) == plan_captures(
        knobs, n_train=n_train, n_validation=n_validation, seed=19,
        lhs_method=method,
    )
    project = new_project(
        knobs, n_train=n_train, n_validation=n_validation, seed=19,
        lhs_method=method,
    )
    assert project.lhs_method == method
    assert project.model_dump()["lhs_method"] == method


@pytest.mark.parametrize("method", ["sobol", "optimized"])
@pytest.mark.parametrize("planning_mode", ["complex", "simple"])
def test_new_samplers_support_categorical_switches(method, planning_mode):
    controls = [
        KnobSpec("Gain", 0, 10, step=0.5),
        KnobSpec("Tone", 0, 10, step=0.5),
        SwitchSpec("Mode", ("Clean", "Dirty"), planning_mode=planning_mode),
    ]
    train, validation = plan_captures(
        controls, n_train=24, n_validation=12, seed=27, lhs_method=method
    )
    assert len(train) == 24 and len(validation) == 12
    assert {p.params["Mode"] for p in train} == {"Clean", "Dirty"}
    assert {p.params["Mode"] for p in validation} == {"Clean", "Dirty"}
    train_keys = {tuple(p.params.values()) for p in train}
    validation_keys = {tuple(p.params.values()) for p in validation}
    assert len(train_keys) == 24 and len(validation_keys) == 12
    assert train_keys.isdisjoint(validation_keys)


@pytest.mark.parametrize("states,requested", [(2, 32), (3, 30), (4, 32)])
def test_simple_switch_sobol_uses_nonoverlapping_power_of_two_blocks(states, requested):
    """Simple states partition one long design, without multiplying captures."""
    from nam.capture.planner import sample_raw_settings, latin_hypercube_unit

    controls = [
        KnobSpec("Gain", 0, 1, step=0.001),
        KnobSpec("Tone", 0, 1, step=0.001),
        SwitchSpec("Mode", tuple(f"S{i}" for i in range(states)),
                   planning_mode="simple"),
    ]
    specs = tuple(control.to_param_spec() for control in controls)
    samples = np.asarray(sample_raw_settings(
        specs, requested, seed=17, lhs_method="sobol"
    ))
    per_state = (requested + states - 1) // states
    block = 1 << (per_state - 1).bit_length()
    source = latin_hypercube_unit(
        states * block, 2, seed=17, lhs_method="sobol"
    )
    assert len(samples) == requested
    for state in range(states):
        selected = samples[samples[:, 2] == state, :2]
        count = requested // states + (state < requested % states)
        np.testing.assert_array_equal(
            selected, source[state * block:state * block + count]
        )


def test_many_simple_switch_states_are_balanced_and_cover_pairs():
    from collections import Counter
    from nam.capture.planner import sample_raw_settings

    controls = [
        KnobSpec("Tone", 0, 1, step=0.001),
        SwitchSpec("Mode", tuple(str(i) for i in range(5)),
                   planning_mode="simple"),
        SwitchSpec("Bright", ("A", "B", "C"),
                   planning_mode="simple"),
    ]
    specs = tuple(control.to_param_spec() for control in controls)
    samples = np.asarray(sample_raw_settings(specs, 45, seed=31, lhs_method="sobol"))
    assert len(samples) == 45
    for column in (1, 2):
        counts = Counter(int(value) for value in samples[:, column])
        assert len(counts) == specs[column].num_inputs
        assert max(counts.values()) - min(counts.values()) <= 1
    assert len({tuple(int(v) for v in row[1:]) for row in samples}) == 15
    np.testing.assert_array_equal(
        samples,
        np.asarray(sample_raw_settings(specs, 45, seed=31, lhs_method="sobol")),
    )


def test_complex_switch_sobol_sets_are_complementary_when_grid_permits():
    controls = [
        KnobSpec("Gain", 0, 10, step=0.25, default=5),
        KnobSpec("Tone", 0, 10, step=0.25, default=5),
        SwitchSpec("Mode", ("A", "B"), planning_mode="complex"),
    ]
    plans = plan_captures(
        controls, n_train=48, n_validation=16, lhs_method="sobol",
        seed=29, include_default_as_additional_capture=True,
    )
    captures = plans[0] + plans[1]
    assert len(plans[0]) == 49 and len(plans[1]) == 16
    assert {entry.params["Mode"] for entry in captures} == {"A", "B"}
    by_state = {
        state: {
            (entry.params["Gain"], entry.params["Tone"])
            for entry in captures if entry.params["Mode"] == state
        }
        for state in ("A", "B")
    }
    assert by_state["A"].isdisjoint(by_state["B"])
    assert plans == plan_captures(
        controls, n_train=48, n_validation=16, lhs_method="sobol",
        seed=29, include_default_as_additional_capture=True,
    )

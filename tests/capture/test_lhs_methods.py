"""Regression tests for selectable Latin-hypercube designs."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from nam.capture.params import KnobSpec
from nam.capture.planner import latin_hypercube_unit, plan_captures
from nam.capture.project import new_project


@pytest.mark.parametrize("method", ["standard", "maximin", "orthogonal", "near_orthogonal"])
def test_lhs_methods_are_reproducible_and_stratified(method):
    first = latin_hypercube_unit(25, 3, seed=13, lhs_method=method)
    second = latin_hypercube_unit(25, 3, seed=13, lhs_method=method)
    np.testing.assert_array_equal(first, second)
    assert np.all((first >= 0) & (first < 1))
    for column in range(3):
        assert sorted(np.floor(first[:, column] * 25).astype(int)) == list(range(25))


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
    for method in ("maximin", "orthogonal", "near_orthogonal"):
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

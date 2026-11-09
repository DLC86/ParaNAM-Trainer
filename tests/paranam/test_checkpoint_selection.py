"""Both non-default checkpoint criteria select Full/Lite independently."""

from pathlib import Path

import pytest

from paranam.checkpoint_selection import (
    add_candidate, build_selector, candidate_score, saved_selection,
    select_candidate,
)


def candidate(epoch, val, train):
    return dict(epoch=epoch, val=val, train=train, path=f"/tmp/epoch-{epoch}.ckpt")


def test_tolerance_selects_lowest_train_within_final_five_percent_band():
    candidates = [
        candidate(1, 0.040, 0.025),
        candidate(2, 0.041, 0.015),
        candidate(3, 0.044, 0.012),
        candidate(4, 0.0392, 0.026),
    ]
    frontier = []
    for row in candidates:
        frontier = add_candidate(frontier, row, "tolerance")
    assert select_candidate(frontier, "tolerance")["epoch"] == 2
    # A later new minimum can make a previously eligible candidate ineligible.
    frontier = add_candidate(frontier, candidate(5, 0.035, 0.032), "tolerance")
    assert select_candidate(frontier, "tolerance")["epoch"] == 5
    assert all(not (
        left["val"] <= right["val"] and left["train"] <= right["train"]
    ) for left in frontier for right in frontier if left is not right)


def test_combined_score_and_alpha_extremes():
    rows = [candidate(1, .04, .03), candidate(2, .045, .01)]
    assert select_candidate(rows, "combined", alpha=1.0)["epoch"] == 1
    assert select_candidate(rows, "combined", alpha=0.0)["epoch"] == 2
    assert candidate_score(rows[1], "combined", alpha=0.75) == pytest.approx(.03625)
    frontier = []
    for entry in rows:
        frontier = add_candidate(frontier, entry, "combined", 0.75)
    assert len(frontier) == 1 and frontier[0]["epoch"] == 2


@pytest.mark.parametrize("method", ["tolerance", "combined"])
def test_selector_saves_only_useful_snapshots_and_resumes(tmp_path, method):
    class Progress:
        last_metrics = None

    class Trainer:
        sanity_checking = False
        current_epoch = 0

        def save_checkpoint(self, path, weights_only=False):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"fake checkpoint")

    progress = Progress()
    trainer = Trainer()
    opts = dict(checkpoint_method=method, checkpoint_alpha=0.75)
    selector = build_selector(progress, opts, tmp_path)
    assert selector.state_key == "ParametricCheckpointSelector"
    for epoch, (full, lite, train_full, train_lite) in enumerate(
        [(0.04, 0.06, 0.025, 0.03),
         (0.041, 0.054, 0.015, 0.028),
         (0.038, 0.052, 0.022, 0.03)], 1
    ):
        trainer.current_epoch = epoch - 1
        progress.last_metrics = dict(
            full=full, lite=lite, train_full=train_full,
            train_lite=train_lite,
        )
        selector.on_validation_end(trainer, None)
    picked = selector.selected_checkpoints()
    assert picked["full"] and picked["lite"]
    assert all(Path(picked[size]["path"]).is_file() for size in ("full", "lite"))
    state = selector.state_dict()
    restored = build_selector(progress, opts, tmp_path)
    restored.load_state_dict(state)
    assert restored.selected_checkpoints() == picked
    checkpoint = dict(callbacks={"selector": state})
    assert saved_selection(checkpoint, method, 0.75) == picked


def test_invalid_alpha_rejected(tmp_path):
    with pytest.raises(ValueError, match="alpha"):
        build_selector(object(), dict(checkpoint_method="combined", checkpoint_alpha=1.2), tmp_path)

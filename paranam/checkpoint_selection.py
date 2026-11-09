"""Checkpoint selection from both unseen-control validation and held-out train audio.

The default trainer still selects the minimum validation ESR. Optional rules
store only the non-dominated (validation ESR, train-audio ESR) frontier, or the
lowest combined-score candidate. This avoids saving every epoch to disk while
allowing the 5% tolerance to be evaluated against the *final* best validation
ESR, not the best value known at the time a checkpoint was written.
"""

from __future__ import annotations

import math
from pathlib import Path


METHODS = ("validation", "tolerance", "combined")
DEFAULT_ALPHA = 0.75
TOLERANCE = 0.05


def candidate_score(item, method, alpha=DEFAULT_ALPHA):
    if method == "combined":
        return alpha * item["val"] + (1.0 - alpha) * item["train"]
    if method == "tolerance":
        return item["train"]
    return item["val"]


def select_candidate(entries, method, alpha=DEFAULT_ALPHA):
    """Select a saved checkpoint, using a final rather than moving 5% band."""
    if not entries:
        return None
    if method not in METHODS:
        raise ValueError("Unknown checkpoint selection method")
    available = list(entries)
    if method == "tolerance":
        best_validation = min(item["val"] for item in available)
        available = [
            item for item in available
            if item["val"] <= best_validation * (1 + TOLERANCE) + 1e-12
        ]
    return min(
        available,
        key=lambda item: (
            candidate_score(item, method, alpha),
            item["val"],
            item["epoch"],
        ),
    )


def add_candidate(entries, candidate, method, alpha=DEFAULT_ALPHA):
    """Keep a bounded one-best or the Pareto frontier (not every epoch)."""
    if method == "combined":
        winner = select_candidate([*entries, candidate], method, alpha)
        return [winner]
    if method != "tolerance":
        raise ValueError("Non-default checkpoint selection expected")
    # A checkpoint can be eliminated only when another is no worse in both
    # metrics. This preserves every possible 5%-band winner after future epochs.
    if any(
        old["val"] <= candidate["val"] and old["train"] <= candidate["train"]
        for old in entries
    ):
        return entries
    return [
        old for old in entries
        if not (
            candidate["val"] <= old["val"]
            and candidate["train"] <= old["train"]
        )
    ] + [candidate]


def saved_selection(checkpoint, method, alpha=DEFAULT_ALPHA):
    """Read a checkpoint selector's durable callback state (resume/export)."""
    for state in checkpoint.get("callbacks", {}).values():
        if not isinstance(state, dict) or state.get("kind") != "checkpoint_selection":
            continue
        if state.get("method") != method:
            continue
        return {
            size: select_candidate(state.get("frontier", {}).get(size, []), method, alpha)
            for size in ("full", "lite")
        }
    return None


def build_selector(progress, options, directory):
    """A Lightning callback that runs after ProgressCallback's TrainEvaluator."""
    from pytorch_lightning.callbacks import Callback

    method = options.get("checkpoint_method", "validation")
    if method == "validation":
        return None
    if method not in METHODS:
        raise ValueError(f"Unknown checkpoint method: {method}")
    alpha = float(options.get("checkpoint_alpha", DEFAULT_ALPHA))
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Checkpoint alpha must be between 0 and 1")
    directory = Path(directory) / "checkpoint-selection"

    class CheckpointSelector(Callback):
        @property
        def state_key(self):
            return "ParametricCheckpointSelector"

        def __init__(self):
            super().__init__()
            self.frontier = {"lite": [], "full": []}

        def state_dict(self):
            return dict(
                kind="checkpoint_selection", method=method,
                alpha=alpha, frontier=self.frontier,
            )

        def load_state_dict(self, state):
            if state.get("method") != method or abs(state.get("alpha", alpha) - alpha) > 1e-9:
                raise ValueError("Checkpoint selection changed during resume")
            self.frontier = {
                size: list(state.get("frontier", {}).get(size, []))
                for size in ("lite", "full")
            }

        def selected_checkpoints(self):
            return {
                size: select_candidate(self.frontier[size], method, alpha)
                for size in ("full", "lite")
            }

        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            metrics = progress.last_metrics or {}
            epoch = trainer.current_epoch + 1
            filename = directory / f"epoch-{epoch:06d}.ckpt"
            new_frontier = {}
            for size in ("full", "lite"):
                val = metrics.get(size)
                train = metrics.get("train_" + size)
                if val is None or train is None or not all(
                    math.isfinite(float(number)) for number in (val, train)
                ):
                    raise RuntimeError(
                        f"Checkpoint selection requires finite validation and "
                        f"held-out train ESR for {size}"
                    )
                candidate = dict(
                    epoch=epoch, val=float(val), train=float(train),
                    path=str(filename),
                )
                new_frontier[size] = add_candidate(
                    self.frontier[size], candidate, method, alpha
                )
            changed = any(
                new_frontier[size] != self.frontier[size]
                for size in ("full", "lite")
            )
            if not changed:
                return
            old_paths = {
                item["path"]
                for entries in self.frontier.values()
                for item in entries
            }
            self.frontier = new_frontier
            directory.mkdir(parents=True, exist_ok=True)
            # One snapshot holds both branches; it can serve Full and Lite.
            trainer.save_checkpoint(str(filename), weights_only=False)
            used_paths = {
                item["path"]
                for entries in self.frontier.values()
                for item in entries
            }
            for obsolete in old_paths - used_paths:
                Path(obsolete).unlink(missing_ok=True)

    return CheckpointSelector()

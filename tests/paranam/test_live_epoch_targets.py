"""Dynamic epoch limits must update the live Lightning fit loop."""

from types import SimpleNamespace

from nam.capture.project import atomic_write_json
from paranam import training
from paranam.workspace import read_json


def test_live_epoch_limit_updates_fit_loop_and_progress_without_epoch_reset(tmp_path):
    spec = {
        "members": 1,
        "preferences": {"epochs": 5},
        "learning": {"trainer": {"max_epochs": 5}},
    }
    shared = {
        "status": "running", "epoch": 3, "completed_epoch": 2,
        "epochs": 5, "global_step": 23,
    }
    callback = training.build_progress_callback(tmp_path, spec, 0, shared)
    trainer = SimpleNamespace(
        fit_loop=SimpleNamespace(max_epochs=5),
        current_epoch=2,
        num_training_batches=3,
    )
    callback.trainer = trainer
    assert callback.sync_total_epochs() == 5

    atomic_write_json(tmp_path / "epoch-target.json", {"epochs": 9})
    assert callback.sync_total_epochs() == 9
    assert trainer.fit_loop.max_epochs == 9
    assert spec["preferences"]["epochs"] == 9
    assert spec["learning"]["trainer"]["max_epochs"] == 9
    progress = read_json(tmp_path / "progress.json")
    assert progress["epochs"] == 9
    assert progress["completed_epoch"] == 2
    assert progress["epoch"] == 3
    assert progress["global_step"] == 23

    # Repeated checks with an unchanged request do not rewrite progress.
    before = (tmp_path / "progress.json").stat().st_mtime_ns
    assert callback.sync_total_epochs() == 9
    assert (tmp_path / "progress.json").stat().st_mtime_ns == before

    # Lowering the remaining target doesn't rewind completed work.
    atomic_write_json(tmp_path / "epoch-target.json", {"epochs": 3})
    assert callback.sync_total_epochs() == 3
    assert trainer.fit_loop.max_epochs == 3
    assert read_json(tmp_path / "progress.json")["completed_epoch"] == 2

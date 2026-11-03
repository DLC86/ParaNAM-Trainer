from types import SimpleNamespace

import pytest
import torch

from paranam.training import build_progress_callback, annotate_validation_metadata


@pytest.mark.parametrize("separate", [False, True])
def test_full_reference_wins_even_when_mean_and_lite_prefer_another_epoch(
    tmp_path, separate
):
    spec = dict(
        preferences=dict(epochs=2),
        members=1,
        learning=dict(separate_size_checkpoints=separate),
        data=dict(train_evaluation=[{}]),
    )
    callback = build_progress_callback(tmp_path, spec, 0, {})
    trainer = SimpleNamespace(
        sanity_checking=False, current_epoch=0, callback_metrics={}
    )
    callback.train_evaluator = SimpleNamespace(
        compute=lambda *args: dict(full=(0.3, 0.05)[trainer.current_epoch], lite=0.04)
    )
    module = SimpleNamespace(net=None, device=torch.device("cpu"))
    for epoch, full, lite in [(0, 0.1, 0.9), (1, 0.2, 0.1)]:
        trainer.current_epoch = epoch
        trainer.callback_metrics = {
            "ESR": torch.tensor((full + lite) / 2),
            "ESR/unseen_audio_unseen_params/full": torch.tensor(full),
            "ESR/unseen_audio_unseen_params/lite": torch.tensor(lite),
        }
        callback.on_validation_end(trainer, module)
    assert callback.best == pytest.approx(0.1)
    assert callback.best_metrics["epoch"] == 1
    assert callback.best_train == pytest.approx(0.05)
    restored = build_progress_callback(tmp_path, spec, 0, {})
    restored.load_state_dict(callback.state_dict())
    assert restored.best == pytest.approx(0.1)
    assert restored.best_train == pytest.approx(0.05)
    exported = dict(metadata={"custom": {"keep": True}})
    annotate_validation_metadata(exported, callback.best_metrics, "Adam")
    metadata = exported["metadata"]["training"]
    assert metadata["validation_esr"] == pytest.approx(0.1)
    assert metadata["validation_esr_full"] == pytest.approx(0.1)
    assert metadata["validation_esr_lite"] == pytest.approx(0.1 if separate else 0.9)
    assert metadata["epoch_lite"] == (2 if separate else 1)
    assert metadata["epoch"] == 1
    assert exported["metadata"]["custom"] == {"keep": True}
    checkpoint = {}
    callback.on_save_checkpoint(trainer, None, checkpoint)
    assert checkpoint["validation_esr"] == pytest.approx(0.2)
    assert checkpoint["best_validation_esr_full"] == pytest.approx(0.1)

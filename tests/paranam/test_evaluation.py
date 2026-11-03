from copy import deepcopy

import numpy as np
import pytest
import torch

from nam.data import Split, init_dataset, np_to_wav
from paranam.evaluation import PROTOCOL, TrainEvaluator, reserve_validation_audio
from paranam import training
from paranam.workspace import read_json
from tests.paranam.test_training import ready


@pytest.mark.parametrize("kind", ["static", "parametric"])
def test_job_reserves_official_windows_and_keeps_captures(workspace, kind):
    workspace.preferences["capture_type"] = kind
    workspace.generate_plan()
    ready(workspace)
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    assert spec["evaluation_protocol"] == PROTOCOL
    for train, evaluation in zip(
        spec["data"]["train"], spec["data"]["train_evaluation"]
    ):
        assert (train["start_seconds"], train["stop_seconds"]) == (10, -9)
        assert (evaluation["start_seconds"], evaluation["stop_seconds"]) == (-9, None)
        assert evaluation["ny"] is None
        for key in ("x_path", "y_path", "delay", "params"):
            assert train.get(key) == evaluation.get(key)
    assert all(e["start_seconds"] == 0 for e in spec["data"]["validation"])
    (job / "resume-0.ckpt").touch()
    assert training.job_can_resume(workspace, job)
    spec.pop("evaluation_protocol")
    from nam.capture.project import atomic_write_json

    atomic_write_json(job / "spec.json", spec)
    assert not training.job_can_resume(workspace, job)


@pytest.mark.parametrize("parametric", [False, True])
def test_evaluation_uses_only_held_out_train_audio_and_equal_capture_weights(
    tmp_path, parametric
):
    rate = 1000
    # The endpoints are deliberately unlike the training interval; inspecting the
    # initialized dataset proves that neither endpoint reaches gradient training.
    x = np.full(30 * rate, 0.01, dtype=np.float32)
    x[10 * rate : 21 * rate] = 0.02
    np_to_wav(x, tmp_path / "x.wav", rate=rate)
    entries = []
    for index, (level, gain) in enumerate(((1, 2), (10, 4))):
        input_path, output_path = tmp_path / f"x{index}.wav", tmp_path / f"y{index}.wav"
        np_to_wav(x * level, input_path, rate=rate)
        np_to_wav(x * level * gain, output_path, rate=rate)
        entry = dict(x_path=str(input_path), y_path=str(output_path), delay=0, ny=100)
        if parametric:
            entry["params"] = {"Gain": index}
        entries.append(entry)
    data = dict(
        type="parametric" if parametric else "dataset",
        common=dict(require_input_pre_silence=None),
        train=entries,
        validation=[dict(entries[0])],
    )
    reserve_validation_audio(data)
    model_config = {
        "net": {
            "config": {
                "params": [
                    dict(name="Gain", type="continuous", min=0, max=1, default=0)
                ]
            }
        }
    }
    train_config = deepcopy(data)
    train_config["common"]["nx"] = 1
    if parametric:
        from nam.models.parametric import data_config_from_model

        train_config = data_config_from_model(train_config, model_config)
    dataset = init_dataset(train_config, Split.TRAIN)
    for capture in dataset.datasets:
        inner = capture.dataset if parametric else capture
        assert len(inner.x) == 11 * rate
        assert torch.allclose(inner.x, inner.x[0].expand_as(inner.x))
    dataset.teardown()

    class Net(torch.nn.Module):
        receptive_field = 1

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x, *args, **kwargs):
            return torch.stack((x * self.weight, x * self.weight * 3), dim=1)

        forward_sizes = forward

    net = Net().train()
    net.weight.grad = torch.tensor(7.0)
    evaluator = TrainEvaluator(data, model_config, net)
    rng = torch.get_rng_state().clone()
    scores = evaluator.compute(net, torch.device("cpu"))
    # Quiet and loud captures contribute equally, rather than pooling energies.
    assert scores["lite"] == pytest.approx((0.25 + 0.5625) / 2, abs=0.002)
    assert scores["full"] == pytest.approx((0.25 + 0.0625) / 2, abs=0.002)
    assert net.training and net.weight.item() == 1 and net.weight.grad.item() == 7
    assert torch.equal(rng, torch.get_rng_state())
    evaluator.close()


def test_queue_has_independent_best_full_esr_cards(window, workspace):
    from nam.capture.project import atomic_write_json

    ready(workspace)
    entry = window.queue_controller.store.add(workspace)
    from pathlib import Path

    job = Path(entry["job"])
    window.selected_job_id = entry["id"]
    atomic_write_json(
        job / "progress.json",
        dict(status="paused", epoch=3, epochs=5, best_esr=0.12, best_train_esr=0.03),
    )
    window._attach(workspace)
    window.refresh_training()
    assert window.metrics["esr"].text() == "0.12000"
    assert window.metrics["train_esr"].text() == "0.03000"
    tabs = window.esr_chart
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Full ESR chart",
        "Lite ESR chart",
        "Training log",
    ]
    history = [
        dict(epoch=1, full=0.1, lite=0.2, train_full=0.01, train_lite=0.02),
        dict(epoch=2, full=0.08, lite=0.15, train_full=float("inf"), train_lite=None),
    ]
    tabs.set_history(history)
    for index, size in enumerate(("full", "lite")):
        tabs.setCurrentIndex(index)
        chart = tabs.widget(index)
        assert chart.model_size == size and chart.history == history
        assert not chart.grab().isNull()


def test_worker_refuses_a_queued_job_with_the_old_split(workspace, monkeypatch):
    from nam.capture.project import atomic_write_json

    ready(workspace)
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    spec.pop("evaluation_protocol")
    atomic_write_json(job / "spec.json", spec)
    monkeypatch.chdir(workspace.directory)
    with pytest.raises(ValueError, match="Retrain this instance"):
        training.run_job(job)
    assert training.job_progress(job)["status"] == "failed"
    assert not (job / "member-0").exists()


@pytest.mark.parametrize("input_name", ["standard.wav", "tts-input-v11.wav"])
def test_held_out_tail_matches_official_nam_v3_dataset(input_name):
    from types import SimpleNamespace
    from paranam.signals import asset_path

    path = str(asset_path(input_name))
    data = dict(
        common={}, train=[dict(x_path=path, y_path=path, delay=0)], validation=[]
    )
    reserve_validation_audio(data)
    evaluator = TrainEvaluator(data, {}, SimpleNamespace(receptive_field=6347))
    assert len(evaluator.captures[0].x) == 9 * 48000
    from nam.train.core import _get_data_config, _Version

    official = _get_data_config(_Version(3, 0, 0), path, path, ny=8192, latency=0)
    official["common"]["nx"] = 6347
    reference = init_dataset(official, Split.VALIDATION)
    torch.testing.assert_close(evaluator.captures[0].x, reference.x, rtol=0, atol=0)
    torch.testing.assert_close(evaluator.captures[0].y, reference.y, rtol=0, atol=0)
    assert official["train"]["start_samples"] == 10 * 48000
    assert official["train"]["stop_samples"] == -9 * 48000
    reference.teardown()
    evaluator.close()

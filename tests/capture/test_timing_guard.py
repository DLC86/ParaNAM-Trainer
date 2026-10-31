import json
from dataclasses import replace

import numpy as np
import pytest

from nam.capture import session as module
from nam.capture.latency import LatencyResult
from tests.capture.test_session import (
    _make_project_dir,
    _enable_loopback,
    _FakeRecorder,
)
from nam.capture.project import load_project


@pytest.mark.parametrize("drift", [0.1, -0.1, 0.25, -0.75, 1.0, 7.0])
def test_subsample_drift_discards_capture_and_removes_old_acceptance(
    tmp_path, monkeypatch, drift
):
    root = _make_project_dir(tmp_path)
    project = load_project(root)
    _enable_loopback(project)
    entry = project.entries[0]
    session = module.CaptureSession(project, root, recorder=_FakeRecorder(480))
    session.capture_entry(entry)
    original = (root / entry.y_path).read_bytes()
    actual = module._measure_delay
    calls = []

    def measure(*args):
        result = actual(*args)
        calls.append(True)
        # Start: amp, loopback; end: loopback.
        if len(calls) == 3:
            result = replace(result, peak_delay=result.peak_delay + drift)
        return result

    monkeypatch.setattr(module, "_measure_delay", measure)
    with pytest.raises(
        module.CaptureSessionError, match="Increase the audio-interface buffer"
    ):
        session.capture_entry(entry)
    assert entry.status == "pending" and entry.delay is None
    assert not (root / entry.y_path).exists()
    assert list((root / "history").rglob("*.wav"))[0].read_bytes() == original
    data = json.loads((root / "data.json").read_text())
    assert not data["train"]
    assert entry.qa.clock_drift_samples == pytest.approx(drift, abs=0.01)


def test_each_capture_uses_its_own_delay_and_ignores_amp_difference(tmp_path):
    root = _make_project_dir(tmp_path)
    project = load_project(root)
    _enable_loopback(project)
    for entry, delay in zip(project.entries[:2], (480, 900)):
        qa = module.CaptureSession(
            project, root, recorder=_FakeRecorder(delay + 80, loopback_delay=delay)
        ).capture_entry(entry)
        assert entry.status == "captured"
        assert not qa.delay_disagreement and not qa.loopback_disagreement
        assert not any("differs" in m or "amp-return" in m for m in qa.messages)
        assert abs(qa.clock_drift_samples) < 0.1
    assert project.entries[0].delay != project.entries[1].delay


@pytest.mark.parametrize("drift", [0.2, -0.2, 0.5, 1.0])
def test_real_fractional_end_marker_drift_is_rejected(tmp_path, drift):
    from nam.capture.resample import apply_fractional_delay
    from nam.capture.latency import BlipPreamble

    class EndDrift(_FakeRecorder):
        def playrec(self, playback, sample_rate, **kwargs):
            main, loopback = super().playrec(playback, sample_rate, **kwargs)
            # Training fixture input is one second. Only the ending marks move.
            offset = BlipPreamble(sample_rate).n_samples + sample_rate
            loopback[offset:] = apply_fractional_delay(loopback[offset:], drift)
            return main, loopback

    root = _make_project_dir(tmp_path)
    project = load_project(root)
    _enable_loopback(project)
    entry = project.entries[0]
    with pytest.raises(module.CaptureSessionError, match="buffer"):
        module.CaptureSession(project, root, recorder=EndDrift(480)).capture_entry(
            entry
        )
    assert abs(entry.qa.clock_drift_samples) >= 0.1
    assert entry.status == "pending"

import json

import numpy as np
import pytest

from nam.capture.project import load_project
from nam.capture.session import CaptureSession, build_playback
from nam.data import wav_to_np
from tests.capture.test_session import (
    _FakeRecorder,
    _enable_loopback,
    _make_project_dir,
)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_capture_applies_calibrated_send_gain_and_preserves_training_reference(
    tmp_path, split
):
    root = _make_project_dir(tmp_path)
    project = load_project(root)
    project.capture_level_dbfs = -12.0
    _enable_loopback(project)
    original = (root / project.input_for_split(split)).read_bytes()
    recorder = _FakeRecorder(480, gain=0.5, noise=0.0)
    session = CaptureSession(project, root, recorder=recorder)
    entry = project.entries_for_split(split)[0]
    session.capture_entry(entry)
    main, loop = recorder.returns[0]
    np.testing.assert_allclose(main, loop * 0.5, atol=1e-7)
    assert (root / project.input_for_split(split)).read_bytes() == original
    manifest = json.loads((root / "captures_raw/manifest.json").read_text())
    record = manifest["captures"][0]
    assert record["capture_level_dbfs"] == -12.0
    reference = np.asarray(
        wav_to_np(root / "captures_raw" / record["main_playback"])
    ).squeeze()
    clock_reference = np.asarray(
        wav_to_np(root / "captures_raw" / record["playback"])
    ).squeeze()
    dry, _ = build_playback(wav_to_np(root / project.input_for_split(split)), 48000)
    np.testing.assert_allclose(reference, dry * 10 ** (-12 / 20), atol=2e-7)
    np.testing.assert_array_equal(reference, clock_reference)
    assert entry.status == "captured"


def test_equal_length_input_replacement_preserves_each_capture_send(tmp_path):
    from nam.data import np_to_wav

    root = _make_project_dir(tmp_path)
    project = load_project(root)
    project.capture_level_dbfs = -9.0
    _enable_loopback(project)
    train = project.entries_for_split("train")
    input_path = root / project.input_for_split("train")
    recorder = _FakeRecorder(480, gain=0.5, noise=0.0)

    expected = []
    for entry, scale in zip(train, (1.0, 0.5)):
        if scale != 1.0:
            np_to_wav(wav_to_np(input_path) * scale, input_path)
        playback, _ = build_playback(wav_to_np(input_path).astype(np.float32), 48000)
        expected.append(playback * 10 ** (-9 / 20))
        CaptureSession(project, root, recorder=recorder).capture_entry(entry)

    records = json.loads((root / "captures_raw/manifest.json").read_text())["captures"]
    assert len({record["main_playback"] for record in records}) == 2
    assert len({record["playback"] for record in records}) == 2
    for record, send in zip(records, expected):
        for key in ("playback", "main_playback"):
            archived = wav_to_np(root / "captures_raw" / record[key])
            np.testing.assert_allclose(archived, send, atol=2e-7)


def test_route_test_attenuates_both_sends(tmp_path):
    root = _make_project_dir(tmp_path)
    project = load_project(root)
    project.capture_level_dbfs = -9.0
    _enable_loopback(project)
    recorder = _FakeRecorder(480, gain=0.5, noise=0.0)
    result = CaptureSession(project, root, recorder=recorder).route_test(sample_rate=48000)
    main, loop = recorder.returns[0]

    dry, _ = build_playback(np.zeros(0, dtype=np.float32), 48000)
    np.testing.assert_allclose(recorder.calls[0]["loopback_playback"], dry * 10 ** (-9 / 20))
    np.testing.assert_allclose(main, loop * 0.5)
    assert result.latency.detected

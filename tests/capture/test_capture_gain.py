import json

import numpy as np
import pytest

from nam.capture.project import load_project
from nam.capture.session import CaptureSession, raw_paths
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
    np.testing.assert_allclose(main, loop * (0.5 * 10 ** (-12 / 20)), atol=1e-7)
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
    np.testing.assert_allclose(reference, clock_reference * 10 ** (-12 / 20), atol=2e-7)
    assert entry.status == "captured"

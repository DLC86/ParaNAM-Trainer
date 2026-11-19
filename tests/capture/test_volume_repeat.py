"""The repeat reference must stay inside ONE Linear Volume capture."""

import json

import numpy as np
import soundfile as sf

from nam.capture.output_gain import initialize_calibration
from nam.capture.project import (
    CaptureProject,
    KnobModel,
    clear_captures,
    reconcile_with_disk,
)
from nam.capture.session import CaptureSession
from tests.capture.test_session import _FakeRecorder


RATE = 48000


def test_new_linear_volume_context_uses_five_and_gain_zero_with_mute_step(tmp_path):
    from nam.capture.output_gain import _context

    project = CaptureProject(
        name="Linear Volume",
        knobs=[
            KnobModel(name="Tone", min=0, max=10, step=1, default=5),
            KnobModel(
                name="Gain", min=0, max=10, step=1, default=5,
                is_gain=True, mute_at_min=True,
            ),
            KnobModel(name="Volume", min=0, max=10, step=1, default=5, clean_volume=True),
        ],
        output_gain_state={"linear_volume_context_version": 2},
    )
    assert _context(project, centered=True) == {
        "Tone": 5.0, "Gain": 1.0, "Volume": 10.0,
    }


def test_new_linear_volume_context_moves_away_from_rule_excluded_combination():
    from nam.capture.output_gain import _context
    from nam.capture.plan_rules import PlanRule

    project = CaptureProject(
        name="Linear Volume",
        knobs=[
            KnobModel(name="Tone", min=0, max=10, step=1, default=5),
            KnobModel(name="Gain", min=0, max=10, step=1, default=5, is_gain=True),
            KnobModel(name="Volume", min=0, max=10, step=1, default=5, clean_volume=True),
        ],
        plan_rules=[
            PlanRule(controls=("Tone",), operator="=", value=5, kind="avoid"),
        ],
        output_gain_state={"linear_volume_context_version": 2},
    )
    context = _context(project, centered=True)
    assert context["Tone"] == 4.0
    assert context["Gain"] == 0.0


def test_existing_linear_volume_plan_keeps_legacy_context():
    from nam.capture.output_gain import _context

    project = CaptureProject(
        name="Legacy Linear Volume",
        knobs=[
            KnobModel(name="Tone", min=0, max=10, step=1, default=5),
            KnobModel(name="Gain", min=0, max=10, step=1, default=5, is_gain=True),
            KnobModel(name="Volume", min=0, max=10, step=1, default=5, clean_volume=True),
        ],
    )
    assert _context(project) == {
        "Tone": 10.0, "Gain": 10.0, "Volume": 10.0,
    }


def test_repeat_pass_is_automatic_and_hidden(tmp_path):
    project = CaptureProject(
        name="Linear Volume",
        sample_rate=RATE,
        knobs=[
            KnobModel(name="Tone", min=0, max=10, step=1, default=5),
            KnobModel(name="Volume", min=0, max=10, step=1, default=5, clean_volume=True),
        ],
    )
    rng = np.random.default_rng(5)
    signal = (0.06 * rng.standard_normal(RATE // 2)).astype(np.float32)
    sf.write(tmp_path / "input_validation.wav", signal, RATE, subtype="FLOAT")
    sf.write(tmp_path / "input_train.wav", signal, RATE, subtype="FLOAT")
    initialize_calibration(project)

    gain_entries = [e for e in project.entries if e.split == "gain"]
    assert len(gain_entries) == 11
    assert sum(e.params["Volume"] == 10 for e in gain_entries) == 1
    assert all(
        e.user_metadata.get("output_gain", {}).get("phase") != "reference_repeat"
        for e in gain_entries
    )

    recorder = _FakeRecorder(480, gain=0.5, noise=0)
    session = CaptureSession(project, tmp_path, recorder=recorder)
    reference_entry = next(e for e in gain_entries if e.params["Volume"] == 10)
    session.capture_entry(reference_entry)

    # One user action, one underlying audio stream, two passes.
    assert len(recorder.calls) == 1
    assert len(project.captured_entries()) == 1
    repeat_path = reference_entry.user_metadata["output_gain"]["repeat_y_path"]
    assert repeat_path != reference_entry.y_path
    assert repeat_path not in {e.y_path for e in project.entries}

    primary, primary_rate = sf.read(tmp_path / reference_entry.y_path)
    repeat, repeat_rate = sf.read(tmp_path / repeat_path)
    assert primary_rate == repeat_rate == RATE
    assert len(primary) == len(repeat) == len(signal)
    np.testing.assert_allclose(primary, repeat, atol=1e-5)

    manifest = json.loads((tmp_path / "captures_raw" / "manifest.json").read_text())
    record = next(r for r in manifest["captures"] if r["y_path"] == reference_entry.y_path)
    assert record["input_samples"] == len(signal) * 2 + RATE
    raw_playback, _ = sf.read(tmp_path / "captures_raw" / record["main_playback"])
    assert len(raw_playback) == len(recorder.returns[0][0])
    assert not any("not part of this project's plan" in note
                   for note in reconcile_with_disk(project, tmp_path))

    clear_captures(project, tmp_path)
    assert not (tmp_path / reference_entry.y_path).exists()
    assert not (tmp_path / repeat_path).exists()
    assert "repeat_y_path" not in reference_entry.user_metadata["output_gain"]

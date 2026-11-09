from collections import Counter

import numpy as np
import pytest
import soundfile as sf

from nam.capture.params import KnobSpec, SwitchSpec, capture_recommendations
from nam.capture.planner import plan_captures, corner_settings, sample_raw_settings
from nam.capture.project import CaptureProject, KnobModel
from nam.capture.output_gain import (
    initialize_calibration,
    advance_calibration,
    OutputGainExportHook,
)


def test_simple_switch_is_one_discrete_dimension_and_complex_repeats():
    controls = [
        KnobSpec("Tone", 0, 10, step=0.1),
        SwitchSpec("Mode", ("A", "B", "C", "D"), planning_mode="simple"),
    ]
    assert capture_recommendations(controls) == (32, 8, 1)
    train, validation = plan_captures(controls, n_train=30, n_validation=8, seed=13)
    assert len(train) == 30 and len(validation) == 8
    counts = Counter(e.params["Mode"] for e in train)
    assert set(counts) == {"A", "B", "C", "D"}
    assert max(counts.values()) - min(counts.values()) <= 2
    keys = lambda entries: {tuple(e.params.items()) for e in entries}
    assert len(keys(train)) == 30 and not keys(train) & keys(validation)
    controls += [SwitchSpec("Complex", ("Off", "On"))]
    assert capture_recommendations(controls) == (64, 16, 2)
    train, validation = plan_captures(controls, n_train=60, n_validation=16, seed=13)
    assert Counter(e.params["Complex"] for e in train) == {"Off": 30, "On": 30}
    assert Counter(e.params["Complex"] for e in validation) == {"Off": 8, "On": 8}


@pytest.mark.parametrize("bias,below", [("lower", True), ("upper", False)])
def test_bias_changes_capture_density_without_changing_model_encoding(bias, below):
    knob = KnobSpec("Tone", 0, 10, step=0.01, sampling_bias=bias)
    raw = sample_raw_settings((knob.to_param_spec(),), 1000, seed=3)
    fraction = np.mean(np.array(raw)[:, 0] < 5)
    assert 0.70 < (fraction if below else 1 - fraction) < 0.76
    assert "sampling_bias" not in knob.to_param_spec().to_dict()


def test_avoid_max_applies_to_lhs_and_corners():
    knob = KnobSpec("Drive", 0, 10, avoid_zero=True, avoid_max=True)
    train, validation = plan_captures([knob], n_train=10, n_validation=5, seed=42)
    corners = corner_settings((knob.to_param_spec(),))
    assert all(
        0 < p["Drive"] < 10 for p in [e.params for e in train + validation] + corners
    )
    assert {p["Drive"] for p in corners} == {0.5, 9.5}


def test_legacy_control_dicts_unchanged():
    knob = KnobSpec("Tone", 0, 10)
    assert not {"clean_volume", "avoid_max", "sampling_bias"} & knob.to_dict().keys()
    assert "planning_mode" not in SwitchSpec("Mode", ("Off", "On")).to_dict()


def test_clean_volume_defaults_step_to_one_but_keeps_it_editable(qapp):
    from paranam.widgets import ControlDialog

    dialog = ControlDialog(control=KnobSpec("Volume", 0, 10, step=0.5))
    dialog.clean_volume.setChecked(True)
    assert dialog.step.value() == 1.0
    assert dialog.step.isEnabled()
    dialog.step.setValue(0.25)
    assert dialog.step.value() == 0.25

    existing = ControlDialog(
        control=KnobSpec("Volume", 0, 10, step=0.25, clean_volume=True)
    )
    assert existing.step.value() == 0.25
    assert not existing.avoid_zero.isEnabled()
    assert not existing.avoid_max.isEnabled()


def test_renaming_and_reordering_controls_preserves_plan(workspace):
    from paranam.workspace import read_json
    from nam.capture.export import (
        build_concat_model_config,
        build_hyper_model_config,
        build_hyper_a2_model_config,
    )
    from nam.capture.project import atomic_write_json

    workspace.draft_controls = [
        KnobSpec("Gain", 0, 10).to_dict(),
        KnobSpec("Tone", 0, 10).to_dict(),
    ]
    workspace.generate_plan()

    for filename, builder in (
        ("model_concat.json", build_concat_model_config),
        ("model_hyper.json", build_hyper_model_config),
        ("model_hyper_a2.json", build_hyper_a2_model_config),
    ):
        atomic_write_json(workspace.directory / filename, builder(workspace.project))

    paths = [entry.y_path for entry in workspace.project.entries]
    workspace.rename_control("Gain", "Drive")
    assert not workspace.plan_changed()
    assert [entry.y_path for entry in workspace.project.entries] == paths
    assert all("Drive" in entry.params for entry in workspace.project.entries)
    assert workspace.project.control_name_history[-1] == {
        "from": "Gain",
        "to": "Drive",
    }
    for filename in (
        "model_concat.json",
        "model_hyper.json",
        "model_hyper_a2.json",
    ):
        config = read_json(workspace.directory / filename)
        assert "Drive" in [p["name"] for p in config["net"]["config"]["params"]]

    workspace.move_control(0, 1)
    assert not workspace.plan_changed()
    assert [c.name for c in workspace.project.control_specs()] == ["Tone", "Drive"]
    assert all(
        list(entry.params) == ["Tone", "Drive"] for entry in workspace.project.entries
    )


def volume_project():
    return CaptureProject(
        name="Test",
        sample_rate=48000,
        knobs=[
            KnobModel(name="Tone", min=0, max=10, step=0.5, default=5),
            KnobModel(
                name="Volume", min=0, max=10, step=0.1, default=5, clean_volume=True
            ),
        ],
    )


def record_pending(
    project, root, *, colored=False, context_dependent=False, paths=None
):
    signal = np.random.default_rng(5).normal(0, 0.05, 4096)
    for entry in project.entries:
        if entry.status == "captured" or (
            paths is not None and entry.y_path not in paths
        ):
            continue
        value = entry.params["Volume"] / 10
        gain = value**2
        if context_dependent and entry.params["Tone"] == 5:
            gain = value
        audio = signal * gain * (1 if entry.params["Tone"] == 10 else 0.7)
        if colored and 0 < value < 1:
            audio = np.clip(signal, -0.025, 0.025) * gain
        target = root / entry.y_path
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, audio, 48000, subtype="FLOAT")
        entry.status = "captured"
        entry.delay = 0


def test_volume_calibration_uses_knob_grid_and_survives_reload(tmp_path):
    project = volume_project()
    initialize_calibration(project)
    assert len(project.entries) == 101
    assert all(e.params["Tone"] == 10 for e in project.entries)
    record_pending(project, tmp_path)
    assert advance_calibration(project, tmp_path) == 0
    project = CaptureProject.model_validate_json(project.model_dump_json())
    advance_calibration(project, tmp_path)
    assert project.output_gain_state["status"] == "ready"
    assert project.output_gain_state["measurements"] == 101
    assert project.output_gain_state["curve_points"] <= 65
    assert all(e.split == "gain" for e in project.entries)
    from scipy.interpolate import PchipInterpolator

    points = np.array(project.output_gain["points"])
    interpolation = PchipInterpolator(points[:, 0], points[:, 1])
    x = np.arange(0.1, 10.01, 0.1)
    assert np.max(np.abs(20 * np.log10(interpolation(x) / (x / 10) ** 2))) < 1.0
    original = dict(architecture="HyperA2", config={"test": 1}, weights=[1, 2])
    wrapped = OutputGainExportHook(project.output_gain).apply(original)
    assert wrapped["weights"] is original["weights"]
    assert wrapped["config"] is original["config"]
    assert wrapped["architecture"] == "HyperA2"
    assert wrapped["extensions"]["linear_volume"] == project.output_gain


def test_volume_peak_rms_warning_is_advisory_and_preserves_captures(tmp_path):
    project = volume_project()
    initialize_calibration(project)
    record_pending(project, tmp_path, colored=True)
    advance_calibration(project, tmp_path)
    assert project.output_gain_state["screen_decision"] == "warning"
    assert "0.20 dB" in project.output_gain_state["warning_message"]
    assert all(e.status == "captured" for e in project.entries)
    project.output_gain_state["screen_decision"] = "accepted"
    advance_calibration(project, tmp_path)
    assert project.output_gain_state["status"] == "ready"
    assert project.output_gain is not None
    assert all((tmp_path / e.y_path).is_file() for e in project.entries)



def test_silent_volume_capture_does_not_trigger_nonlinearity_warning(tmp_path):
    project = volume_project()
    initialize_calibration(project)
    record_pending(project, tmp_path)
    minimum = next(
        entry for entry in project.entries if entry.params["Volume"] == 0
    )

    # A verified noise-floor recording can have a very different crest factor
    # from the live amp signal while remaining much quieter than the reference.
    noise = np.random.default_rng(72).normal(0, 1e-5, 4096)
    noise[0] = 3e-4  # Peak and RMS do not track the reference.
    sf.write(tmp_path / minimum.y_path, noise, 48000, subtype="FLOAT")

    advance_calibration(project, tmp_path)
    assert project.output_gain_state["screen_decision"] == "warning"
    assert "volume 0" in project.output_gain_state["warning_message"]

    # The session's per-capture silent flag takes precedence over crest factor
    # comparisons, including when it is set after an earlier warning.
    minimum.user_metadata.setdefault("output_gain", {})["silent_minimum"] = True
    advance_calibration(project, tmp_path)
    assert project.output_gain_state["screen_decision"] is None
    assert project.output_gain_state["warning_message"] == ""
    assert project.output_gain_state["status"] == "ready"
    assert project.output_gain is not None
    # A noise-floor recording at mute is not an output gain, so its gain is 0.
    assert project.output_gain["points"][0][1] == 0.0



def test_neural_plan_fixes_volume_at_maximum():
    controls = [
        KnobSpec("Tone", 0, 10, step=0.1),
        KnobSpec("Volume", 0, 10, clean_volume=True),
    ]
    assert capture_recommendations(controls) == (16, 8, 1)
    train, validation = plan_captures(controls, n_train=15, n_validation=8)
    assert all(e.params["Volume"] == 10 for e in train + validation)


def test_volume_calibration_is_saved_first_and_blocks_manual_neural_capture(tmp_path):
    from nam.capture.project import CaptureEntryModel
    from nam.capture.session import CaptureSession, CaptureSessionError

    project = volume_project()
    neural = CaptureEntryModel(
        index=0,
        split="train",
        params={"Tone": 5, "Volume": 10},
        y_path="captures/train.wav",
    )
    project.entries = [neural]
    initialize_calibration(project)
    assert project.entries[-1] is neural
    assert all(e.split == "gain" for e in project.entries[:-1])
    assert project.entries[0].params["Volume"] == 10
    session = CaptureSession(project, tmp_path, recorder=object())
    for state in ("measuring", "failed"):
        project.output_gain_state["status"] = state
        with pytest.raises(
            CaptureSessionError, match="Finish clean output-volume calibration"
        ):
            session.capture_entry(neural)


def test_muted_volume_minimum_can_be_recorded_without_loopback(tmp_path):
    from nam.capture.session import CaptureSession
    from tests.capture.test_session import _FakeRecorder
    from paranam.capture_order import order_captures

    project = volume_project()
    initialize_calibration(project)
    project.audio.loopback_input_channel = None
    project.audio.loopback_output_channel = None
    signal = np.random.default_rng(32).normal(0, 0.1, 48000).astype(np.float32)
    for path in (project.train_input, project.validation_input):
        sf.write(tmp_path / path, signal, 48000, subtype="FLOAT")
    recorder = _FakeRecorder(480, noise=0)
    session = CaptureSession(project, tmp_path, recorder=recorder)
    ordered = order_captures(project.entries, project.control_specs())
    assert ordered[0].params["Volume"] == 10
    reference = next(e for e in ordered if e.params == {"Tone": 10, "Volume": 10})
    session.capture_entry(reference)
    minimum = next(e for e in ordered if e.params == {"Tone": 10, "Volume": 0})
    recorder.gain = 0
    recorder.noise = 1e-5
    qa = session.capture_entry(minimum)
    assert minimum.status == "captured" and qa.end_timing_detected is None
    assert minimum.user_metadata["output_gain"]["silent_minimum"]
    assert not qa.messages
    from nam.capture.latency import LatencyResult

    detected = LatencyResult(
        delay=480, detected=True, disagreement_too_high=False, safety_factor=1
    )
    undetected = LatencyResult(
        delay=None, detected=False, disagreement_too_high=False, safety_factor=1
    )
    # A test blip is signal even at very low gain; an ordinary missing return
    # must never count as an intentional muted volume control.
    assert not session._silent_volume_minimum(minimum, detected)
    assert session._silent_volume_minimum(minimum, undetected)
    ordinary = next(e for e in project.entries if e.params["Volume"] == 5)
    assert not session._silent_volume_minimum(ordinary, undetected)
    from nam.capture.session import audit_captures

    assert minimum.y_path not in {a.y_path for a in audit_captures(project, tmp_path)}



@pytest.mark.parametrize("loopback", [False, True])
def test_noisy_muted_volume_uses_missing_test_blips_not_absolute_noise(tmp_path, loopback):
    from nam.capture.session import CaptureSession
    from tests.capture.test_session import _FakeRecorder

    project = volume_project()
    initialize_calibration(project)
    if loopback:
        project.audio.output_channel = 1
        project.audio.input_channel = 1
        project.audio.loopback_input_channel = 2
        project.audio.loopback_output_channel = 2
    else:
        project.audio.loopback_input_channel = None
        project.audio.loopback_output_channel = None
    signal = np.random.default_rng(32).normal(0, 0.1, 48000).astype(np.float32)
    for path in (project.train_input, project.validation_input):
        sf.write(tmp_path / path, signal, 48000, subtype="FLOAT")
    recorder = _FakeRecorder(480, noise=0)
    session = CaptureSession(project, tmp_path, recorder=recorder)
    reference = next(
        e for e in project.entries if e.params == {"Tone": 10, "Volume": 10}
    )
    session.capture_entry(reference)
    minimum = next(
        e for e in project.entries if e.params == {"Tone": 10, "Volume": 0}
    )
    # Background noise is only ~45 dB below the reference, well above the
    # old -60 dB mute test. No played blips actually reach the amp return.
    recorder.gain = 0
    recorder.noise = 2e-4
    qa = session.capture_entry(minimum)
    assert minimum.status == "captured"
    assert minimum.user_metadata["output_gain"]["silent_minimum"] is True
    assert qa.amp_return_delay is None
    assert qa.end_timing_detected is (True if loopback else None)

    # The noisy minimum must not raise a crest-factor warning.
    advance_calibration(project, tmp_path)
    assert project.output_gain_state.get("screen_decision") is None

    # Simulate a saved 0.7.2 capture with an obsolete warning and no marker:
    # re-evaluate its recorded test blips rather than requiring a recapture.
    minimum.user_metadata["output_gain"].pop("silent_minimum")
    project.output_gain_state.update(
        screen_decision="warning",
        warning_message="At volume 0, peak and RMS attenuation differ.",
    )
    advance_calibration(project, tmp_path)
    assert minimum.user_metadata["output_gain"]["silent_minimum"] is True
    assert project.output_gain_state["screen_decision"] is None
    assert project.output_gain_state["warning_message"] == ""


def test_truncated_input_preserves_original_and_standard_validation(tmp_path):
    from paranam.signals import asset_path, install_signals
    from paranam.workspace import Workspace

    original, rate = sf.read(asset_path("inputTrunc.wav"))
    standard, _ = sf.read(asset_path("standard.wav"))
    install_signals(tmp_path, include_trunc=True)
    prepared, prepared_rate = sf.read(tmp_path / "input_trunc.wav")
    assert rate == prepared_rate == 48000
    assert len(prepared) == 57 * rate
    np.testing.assert_array_equal(prepared[: 10 * rate], standard[: 10 * rate])
    np.testing.assert_array_equal(prepared[10 * rate : -9 * rate], original)
    np.testing.assert_array_equal(prepared[-9 * rate :], standard[-9 * rate :])
    ws = Workspace.create(tmp_path / "Rig", "Rig")
    ws.preferences["input_mode"] = "trunc"
    assert ws.desired_train_input() == "input_trunc.wav"
    assert (ws.directory / "input_trunc.wav").is_file()


def test_decay_selectors_share_decay_rows_only(window, workspace):
    window._attach(workspace)
    fields = window.advanced_fields
    for decay, cadence, rate in (
        ("lr_decay", "rate_preset", "learning_rate"),
        (
            "hypernetwork_lr_decay",
            "hypernetwork_rate_preset",
            "hypernetwork_learning_rate",
        ),
    ):
        assert fields[decay].parentWidget() is fields[cadence].parentWidget()
        assert fields[rate].parentWidget() is window.advanced_panel
        assert fields[decay].width() == fields[rate].width() == 260
        assert "transparent" in fields[decay].parentWidget().styleSheet()
    assert (
        window.signal_mode.itemText(window.signal_mode.findData("trunc"))
        == "Truncated (Phillip Self)"
    )


def test_parametric_export_preserves_output_curve_after_output_scaling(tmp_path):
    import json
    from nam.models.parametric import HyperA2, export_parametric

    curve = {"schema_version": 1, "name": "Volume"}
    model = HyperA2.init_from_config(
        {
            "sample_rate": 48000,
            "params": [
                {
                    "name": "Tone",
                    "type": "continuous",
                    "min": 0,
                    "max": 10,
                    "default": 5,
                }
            ],
            "hypernet": {
                "hidden_sizes": [4],
                "mode": "low_rank",
                "rank": 2,
                "selector": {},
            },
        }
    )
    hook = OutputGainExportHook(curve)
    model.export_model_dict_post_hooks.append(hook)
    export_parametric(model, tmp_path, output_scale=2.0)
    saved = json.loads((tmp_path / "model.pnam").read_text())
    assert saved["architecture"] == "ParametricSlimmableContainer"
    assert saved["extensions"]["linear_volume"] == curve
    assert saved["config"]["params"][0]["name"] == "Tone"
    assert len(saved["config"]["submodels"]) == 2
    assert "Volume" not in [p["name"] for p in saved["config"]["params"]]
    assert "base_config" not in saved["config"]
    assert model.export_model_dict_post_hooks[-1] is hook


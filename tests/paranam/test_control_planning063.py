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
    assert capture_recommendations(controls) == (30, 8, 1)
    train, validation = plan_captures(controls, n_train=30, n_validation=8, seed=13)
    assert len(train) == 30 and len(validation) == 8
    counts = Counter(e.params["Mode"] for e in train)
    assert set(counts) == {"A", "B", "C", "D"}
    assert max(counts.values()) - min(counts.values()) <= 2
    keys = lambda entries: {tuple(e.params.items()) for e in entries}
    assert len(keys(train)) == 30 and not keys(train) & keys(validation)
    controls += [SwitchSpec("Complex", ("Off", "On"))]
    assert capture_recommendations(controls) == (60, 16, 2)
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


def record_pending(project, root, *, colored=False, context_dependent=False):
    signal = np.random.default_rng(5).normal(0, 0.05, 4096)
    for entry in project.entries:
        if entry.status == "captured":
            continue
        value = entry.params["Volume"] / 10
        gain = value**2
        if context_dependent and entry.params["Tone"] == 5:
            gain = value
        audio = signal * gain * (1 if entry.params["Tone"] == 10 else 0.7)
        if colored and 0 < value < 1:
            audio = audio + 0.15 * gain * np.roll(signal, 1)
        target = root / entry.y_path
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, audio, 48000, subtype="FLOAT")
        entry.status = "captured"
        entry.delay = 0


def test_volume_calibration_adapts_and_survives_reload(tmp_path):
    project = volume_project()
    initialize_calibration(project)
    assert len(project.entries) == 11
    for _ in range(8):
        record_pending(project, tmp_path)
        project = CaptureProject.model_validate_json(project.model_dump_json())
        if not advance_calibration(project, tmp_path):
            break
    assert project.output_gain_state["status"] == "ready"
    assert 9 < project.output_gain_state["measurements"] <= 65
    assert all(e.split == "gain" for e in project.entries)
    from scipy.interpolate import PchipInterpolator

    points = np.array(project.output_gain["points"])
    interpolation = PchipInterpolator(points[:, 0], points[:, 1])
    x = np.arange(0.1, 10.01, 0.1)
    assert np.max(np.abs(20 * np.log10(interpolation(x) / (x / 10) ** 2))) < 0.25
    original = dict(architecture="HyperA2", config={"test": 1}, weights=[1, 2])
    wrapped = OutputGainExportHook(project.output_gain).apply(original)
    assert wrapped["weights"] is original["weights"]
    assert wrapped["config"]["base_config"] == original["config"]


@pytest.mark.parametrize("failure", ["colored", "context_dependent"])
def test_non_clean_volume_is_rejected_without_deleting_captures(tmp_path, failure):
    project = volume_project()
    initialize_calibration(project)
    record_pending(project, tmp_path, **{failure: True})
    with pytest.raises(ValueError):
        advance_calibration(project, tmp_path)
    assert project.output_gain is None
    assert all((tmp_path / e.y_path).is_file() for e in project.entries)


def test_neural_plan_fixes_volume_at_maximum():
    controls = [
        KnobSpec("Tone", 0, 10, step=0.1),
        KnobSpec("Volume", 0, 10, clean_volume=True),
    ]
    assert capture_recommendations(controls) == (15, 8, 1)
    train, validation = plan_captures(controls, n_train=15, n_validation=8)
    assert all(e.params["Volume"] == 10 for e in train + validation)


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
    assert saved["architecture"] == "ParametricOutputGain"
    assert saved["config"]["base_architecture"] == "HyperA2"
    assert saved["config"]["output_gain"] == curve
    assert model.export_model_dict_post_hooks[-1] is hook

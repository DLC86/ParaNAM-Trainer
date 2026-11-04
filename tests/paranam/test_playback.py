import itertools
import json
import shutil
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nam.capture.project import AudioSettingsModel, CaptureProject
from nam.models.parametric import HyperA2, export_parametric
from paranam.playback import (
    NativeModel,
    PlayWorker,
    runtime_path,
    send_gain,
    validate_route,
)
from paranam.calibration import LevelCalibrationDialog


def test_instrument_voltage_calibration_is_separate_from_return(qapp):
    dialog = LevelCalibrationDialog(
        SimpleNamespace(name="Interface"),
        1,
        2,
        128,
        instrument_channel=3,
        instrument_name="Hi-Z",
    )
    measured = []
    returned = []
    dialog.instrument_calibrated.connect(measured.append)
    dialog.return_calibrated.connect(returned.append)
    dialog.volts.setValue(0.775)
    dialog.apply_send()
    dialog.measure_input.setCurrentIndex(1)
    dialog.direct.setChecked(True)
    dialog.worker = SimpleNamespace(stopping=threading.Event())
    dialog.measured(
        dict(valid=True, clipping=False, dbfs=-20.0, frequency=1000.0, level=0.0)
    )
    assert dialog.use_instrument.isEnabled() and not dialog.use_return.isEnabled()
    dialog.apply_return()
    assert not returned
    dialog.apply_instrument()
    assert measured == pytest.approx([20.0])
    dialog.measured(
        dict(valid=True, clipping=True, dbfs=0.0, frequency=1000.0, level=0.0)
    )
    assert not dialog.use_instrument.isEnabled()
    dialog.worker = None
    dialog.close()


def test_reamp_gain_uses_instrument_reference_and_capture_attenuation():
    project = CaptureProject(
        name="Test", knobs=[], input_level_dbu=6, capture_level_dbfs=-12
    )
    # 18 dBu at digital full scale on both converters -> unity send, despite -12 dB capture attenuation.
    assert send_gain(project, 18) == pytest.approx(1)
    assert send_gain(project, 12) == pytest.approx(10 ** (-6 / 20))
    project.capture_level_dbfs = None
    assert send_gain(project, 18) == 0


def test_play_route_rejects_feedback_and_unconfigured_channels():
    device = SimpleNamespace(max_input_channels=4, max_output_channels=4)
    audio = AudioSettingsModel(
        instrument_input_channel=3, monitor_output_left=3, monitor_output_right=4
    )
    validate_route(device, audio)
    audio.monitor_output_left = 1
    with pytest.raises(ValueError, match="three different"):
        validate_route(device, audio)
    audio.monitor_output_left = 3
    audio.instrument_input_channel = 1
    with pytest.raises(ValueError, match="different channels"):
        validate_route(device, audio)


def test_play_uses_current_calibration_without_rewriting_captured_levels(
    window, workspace
):
    workspace.project.input_level_dbu = 18.0
    workspace.project.output_level_dbu = 12.0
    workspace.save()
    window._attach(workspace)
    window._set_capture_level(-12.0)
    window._apply_level_calibration(window.input_calibration, window.input_dbu, 6.0)
    window.apply_instrument_calibration(19.123)
    live = window.playback_project()
    assert live.capture_level_dbfs == -12
    assert live.input_level_dbu == 6
    assert workspace.project.capture_level_dbfs == 0
    assert workspace.project.input_level_dbu == 18
    assert window.instrument_reference() == pytest.approx(19.123)
    assert send_gain(live, 19.123) == pytest.approx(10 ** (1.123 / 20))


def require_native():
    try:
        runtime_path()
    except RuntimeError:
        pytest.skip("Build the native playback engine first")


def test_native_full_lite_multiple_switch_combinations_and_calibration(tmp_path):
    require_native()
    torch.set_num_threads(1)
    torch.manual_seed(41)
    model = HyperA2.init_from_config(
        dict(
            sample_rate=48000,
            params=[
                dict(name="Gain", type="continuous", min=0, max=10, default=5),
                dict(
                    name="Bright",
                    type="switch",
                    enum_names=["Off", "On"],
                    min=0,
                    max=1,
                    default=0,
                ),
                dict(
                    name="Mode",
                    type="switch",
                    enum_names=["A", "B", "C"],
                    min=0,
                    max=2,
                    default=0,
                ),
            ],
            hypernet=dict(hidden_sizes=[4], mode="full", selector={}),
        )
    ).eval()
    metadata = dict(input_level_dbu=18, output_level_dbu=12)
    export_parametric(model, tmp_path, other_metadata=metadata)
    exported_path = tmp_path / "model.pnam"
    exported = json.loads(exported_path.read_text())

    def loudness(node, remove=False):
        metadata = node.setdefault("metadata", {})
        if remove:
            for key in ("loudness", "output_level_dbu", "output_dbu"):
                metadata.pop(key, None)
        else:
            metadata["loudness"] = -24
        for child in node.get("config", {}).get("submodels", []):
            loudness(child.get("model", child), remove)

    loudness(exported)
    exported_path.write_text(json.dumps(exported), encoding="utf8")
    unicode_path = tmp_path / "modèle-音.pnam"
    shutil.copy2(tmp_path / "model.pnam", unicode_path)
    unicode_model = NativeModel(unicode_path)
    unicode_model.close()
    x = np.random.default_rng(2).normal(0, 0.1, 256).astype(np.float32)
    for size, (bright, mode) in itertools.product(
        ("full", "lite"), itertools.product(range(2), range(3))
    ):
        native = NativeModel(
            tmp_path / "model.pnam", block_size=64, lite=size == "lite"
        )
        try:
            assert [p["name"] for p in native.info["controls"]] == [
                "Gain",
                "Bright",
                "Mode",
            ]
            values = [7.0, float(bright), float(mode)]
            native.parameters(values)
            model.set_size(size)
            with torch.no_grad():
                reference = model(torch.from_numpy(x), torch.tensor(values)).numpy()
            np.testing.assert_allclose(
                native.process(x), reference, rtol=2e-4, atol=2e-5
            )
            assert native.calibration(24, 18) == pytest.approx(
                (10 ** (0.3), 10 ** (-0.3))
            )
            assert native.calibration(24, 18, "raw") == pytest.approx((10**0.3, 1))
            assert native.calibration(24, None, "calibrated") == pytest.approx(
                (10**0.3, 1)
            )
            assert native.calibration(24, None, "normalized") == pytest.approx(
                (10**0.3, 10**0.3)
            )
        finally:
            native.close()
    loudness(exported, remove=True)
    exported_path.write_text(json.dumps(exported), encoding="utf8")
    native = NativeModel(exported_path)
    try:
        assert native.calibration(24, 18, "calibrated") == pytest.approx((10**0.3, 1))
        assert native.calibration(24, 18, "normalized") == pytest.approx((10**0.3, 1))
    finally:
        native.close()


def test_play_callback_routes_only_send_and_stereo_monitors(qapp, monkeypatch):
    from paranam import playback
    import sounddevice
    from nam.capture import audio as capture_audio
    from contextlib import nullcontext

    monkeypatch.setattr(capture_audio, "asio_com_apartment", nullcontext)
    project = CaptureProject(
        name="Test",
        knobs=[],
        audio=AudioSettingsModel(
            instrument_input_channel=3, monitor_output_left=3, monitor_output_right=4
        ),
    )
    device = SimpleNamespace(
        name="Interface",
        index=0,
        max_input_channels=4,
        max_output_channels=6,
        host_api="Test",
    )

    class Model:
        info = {"controls": [], "size_selection": False}

        def __init__(self, *args):
            pass

        def calibration(self, *args):
            return 2, 0.5

        def process(self, x):
            return x * 3

        def parameters(self, values):
            pass

        def close(self):
            pass

    monkeypatch.setattr(playback, "NativeModel", Model)
    outputs = []
    worker = PlayWorker("unused.nam", device, project)
    worker.settings.update(source="model", output_db=0)

    class Stream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def __enter__(self):
            incoming = np.zeros((4800, 4), dtype=np.float32)
            incoming[:, 2] = 0.1
            incoming[:, 0] = 0.2
            for source in ("model", "gear"):
                worker.settings = {**worker.settings, "source": source}
                outgoing = np.empty((4800, 6), dtype=np.float32)
                self.callback(incoming, outgoing, 4800, None, False)
                outputs.append(outgoing.copy())
            worker.stopping.set()
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(sounddevice, "Stream", Stream)
    errors = []
    worker.failed.connect(errors.append)
    worker.run()
    assert not errors
    np.testing.assert_allclose(outputs[0][-1, 0], 0.1, atol=1e-6)
    np.testing.assert_allclose(outputs[0][-1, 2:4], 0.3, atol=1e-6)
    assert outputs[1][-1, 2] == pytest.approx(0.2)
    for output in outputs:
        assert not output[:, [1, 4, 5]].any()
        np.testing.assert_array_equal(output[:, 2], output[:, 3])

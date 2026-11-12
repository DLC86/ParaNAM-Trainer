import json

import numpy as np
import pytest

from nam.capture.params import KnobSpec
from nam.capture.planner import corner_settings, plan_captures
from nam.data import np_to_wav
from paranam.impedance import format_impedance, parse_impedance


@pytest.mark.parametrize("text, ohms, display", [
    ("1,5M", 1500000, "1,5 MΩ"),
    ("3,3k", 3300, "3,3 kΩ"),
    ("4670", 4670, "4,67 kΩ"),
    ("1,43672 kΩ", 1436.72, "1,43672 kΩ"),
])
def test_impedance_round_trip(text, ohms, display):
    assert parse_impedance(text) == ohms
    assert format_impedance(ohms) == display
    assert parse_impedance(display) == ohms


def test_legacy_switch_export_keeps_neural_values_in_full_and_lite(tmp_path):
    from nam.models.factory import init
    from nam.models.metadata import UserMetadata
    from nam.models.parametric import export_parametric

    model = init("HyperA2", args=({
        "sample_rate": 48000,
        "hypernet": {"hidden_sizes": [4], "mode": "low_rank", "rank": 2},
        "params": [{
            "name": "Mode", "type": "switch", "min": 0, "max": 2, "default": 1,
            "options": [
                {"name": "Lead", "value": 2, "index_sw": 0},
                {"name": "Clean", "value": 0, "index_sw": 1},
                {"name": "Crunch", "value": 1, "index_sw": 2},
            ],
        }],
    },))
    export_parametric(model, tmp_path, basename="legacy",
                      user_metadata=UserMetadata(input_z=1436.72))
    payload = json.loads((tmp_path / "legacy.pnam").read_text())
    assert payload["architecture"] == "ParametricSlimmableContainer"
    assert payload["metadata"]["input_z"] == 1436.72
    config = payload["config"]
    for cfg in [config, *(child["config"] for child in config["submodels"])]:
        param = cfg["params"][0]
        assert param["enum_names"] == ["Clean", "Crunch", "Lead"]
        assert param["default"] == 1
        assert "options" not in param


def test_plan_avoids_negative_min_but_keeps_zero():
    specs = [KnobSpec("Bias", -2, 2, step=1, default=0, avoid_zero=True)]
    train, val = plan_captures(specs, n_train=2, n_validation=1)
    positions = {e.params["Bias"] for e in train + val}
    assert 0 in positions and -2 not in positions
    assert {p["Bias"] for p in corner_settings([specs[0].to_param_spec()])} == {-1, 2}


def test_static_capture_loads_external_pair_without_interface(window, workspace, monkeypatch):
    workspace.preferences["capture_type"] = "static"
    workspace.generate_plan()
    window._attach(workspace)
    window.navigate(2)
    assert window.record_button.text() == "Record"
    assert not window.load_capture_button.isHidden()
    assert not window.device.currentData()
    session = window._get_session()
    training, rate = session._input_for_split("train")
    validation, _ = session._input_for_split("validation")
    wav = workspace.directory / "external.wav"
    np_to_wav(np.full(len(training) + len(validation), .25), wav, rate=rate)
    monkeypatch.setattr(window, "choose_open_file", lambda *_: (str(wav), ""))
    monkeypatch.setattr(window, "notify_error", lambda error: pytest.fail(str(error)))
    window.load_static_wav()
    assert window.capture_progress.value() == 100
    assert all(e.status == "captured" for e in workspace.project.entries)
    assert all(e.delay == -1 and e.qa.peak < 0 for e in workspace.project.entries)
    assert workspace.training_ready()


def test_non_asio_dropout_aborts_and_rejects_partial_recording(monkeypatch):
    from types import SimpleNamespace
    import sounddevice as sd
    from nam.capture.audio import AudioDropoutError, SounddeviceRecorder

    aborted = []

    class BrokenStream:
        active = False
        def __init__(self, callback, **_):
            self.callback = callback
        def __enter__(self):
            block = np.zeros((4, 1), dtype=np.float32)
            status = SimpleNamespace(input_overflow=True, output_underflow=False)
            with pytest.raises(sd.CallbackAbort):
                self.callback(block, block.copy(), len(block), None, status)
            return self
        def __exit__(self, *_):
            pass
        def abort(self):
            aborted.append(True)

    monkeypatch.setattr(sd, "Stream", BrokenStream)
    monkeypatch.setattr(sd, "query_devices", lambda *_, **__: {
        "max_input_channels": 1, "max_output_channels": 1,
    })
    with pytest.raises(AudioDropoutError, match="recorded samples were dropped"):
        SounddeviceRecorder().playrec(np.zeros(128), 48000)
    assert aborted

"""CoreAudio/non-ASIO software monitor regression tests.

The ASIO path already routed Return, but the sounddevice.Stream fallback did not.
These tests exercise the real recorder callback with a simulated PortAudio stream.
"""

import sys
from types import ModuleType

import numpy as np
import pytest

from nam.capture.audio import AudioDeviceError, SounddeviceRecorder


class _Finished(Exception):
    pass


class _Flags:
    input_overflow = False
    output_underflow = False


class _MockStream:
    outputs = []
    inputs = []

    def __init__(self, *, callback, channels, **kwargs):
        self.callback = callback
        self.channels = channels
        self.active = True

    def __enter__(self):
        frames = len(self.inputs[0])
        incoming = self.inputs[0]
        outgoing = np.zeros((frames, self.channels[1]), np.float32)
        try:
            self.callback(incoming, outgoing, frames, None, _Flags())
        except _Finished:
            self.active = False
        self.outputs.append(outgoing.copy())
        return self

    def __exit__(self, *args):
        self.active = False

    def abort(self):
        self.active = False


@pytest.fixture
def mock_coreaudio(monkeypatch):
    fake = ModuleType("sounddevice")
    fake.CallbackStop = _Finished
    fake.CallbackAbort = _Finished
    fake.Stream = _MockStream
    fake.sleep = lambda _: None
    fake.query_devices = lambda index=None, *, kind=None: {
        "max_input_channels": 4,
        "max_output_channels": 6,
    }
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    _MockStream.outputs = []
    _MockStream.inputs = []
    return fake


@pytest.mark.parametrize("enabled", [False, True])
def test_coreaudio_return_monitor_routes_only_selected_physical_outputs(
    mock_coreaudio, enabled,
):
    played = np.linspace(-0.25, 0.25, 64, dtype=np.float32)
    loop = np.linspace(0.25, -0.25, 64, dtype=np.float32)
    incoming = np.zeros((64, 4), dtype=np.float32)
    incoming[:, 1] = 0.4
    incoming[:, 2] = 0.1
    _MockStream.inputs = [incoming]

    recording, lb = SounddeviceRecorder().playrec(
        played, 48000, host_api="Core Audio",
        input_device=1, output_device=1,
        input_channel=2, output_channel=1,
        loopback_input_channel=3, loopback_output_channel=2,
        loopback_playback=loop,
        monitor_return_enabled=enabled,
        monitor_output_left=3, monitor_output_right=4,
    )
    assert len(_MockStream.outputs) == 1
    output = _MockStream.outputs[0]
    np.testing.assert_allclose(output[:, 0], played)
    np.testing.assert_allclose(output[:, 1], loop)
    for channel in (2, 3):
        np.testing.assert_allclose(output[:, channel], 0.4 if enabled else 0)
    np.testing.assert_allclose(output[:, 4:], 0)
    np.testing.assert_allclose(recording, 0.4)
    np.testing.assert_allclose(lb, 0.1)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (None, None, "Select at least one"),
        (1, 3, "distinct from Send"),
        (2, None, "distinct from Send"),
        (3, 3, "distinct from Send"),
        (7, None, "outside the device"),
    ],
)
def test_coreaudio_monitor_rejects_invalid_channels(
    mock_coreaudio, left, right, expected,
):
    with pytest.raises(AudioDeviceError, match=expected):
        SounddeviceRecorder().playrec(
            np.ones(64, np.float32), 48000,
            host_api="Core Audio", input_device=1, output_device=1,
            output_channel=1, input_channel=2,
            loopback_output_channel=2, loopback_input_channel=3,
            loopback_playback=np.ones(64, np.float32),
            monitor_return_enabled=True,
            monitor_output_left=left, monitor_output_right=right,
        )
    assert _MockStream.outputs == []


def test_coreaudio_can_toggle_return_monitor_between_audio_blocks(mock_coreaudio):
    enabled = False
    observed = []

    class TwoBlockStream(_MockStream):
        def __enter__(self):
            nonlocal enabled
            incoming = np.full((32, 4), 0.4, dtype=np.float32)
            for enabled in (False, True):
                outgoing = np.zeros((32, 6), dtype=np.float32)
                try:
                    self.callback(incoming, outgoing, 32, None, _Flags())
                except _Finished:
                    self.active = False
                observed.append(outgoing)
            return self

    mock_coreaudio.Stream = TwoBlockStream
    played = np.full(64, 0.2, dtype=np.float32)
    recorded, _ = SounddeviceRecorder().playrec(
        played, 48000, host_api="Core Audio", input_device=1, output_device=1,
        monitor_return_enabled=lambda: enabled, monitor_output_left=3,
    )

    np.testing.assert_array_equal(np.concatenate(observed)[:, 0], played)
    np.testing.assert_allclose(observed[0][:, 2], 0)
    np.testing.assert_allclose(observed[1][:, 2], 0.4)
    np.testing.assert_allclose(recorded, 0.4)

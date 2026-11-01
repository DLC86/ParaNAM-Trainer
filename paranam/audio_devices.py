"""Driver-reported channel labels and native buffer sizes (no audio is played)."""

import ctypes as ct
import ctypes.util
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Capabilities:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    buffers: tuple[int, ...] = ()
    note: str = ""

    @property
    def default_buffer(self):
        return 2048 if 2048 in self.buffers else max(self.buffers, default=0)


def buffer_values(minimum, maximum, preferred, granularity):
    if minimum <= 0 or maximum < minimum:
        return ()
    if minimum == maximum or granularity == 0:
        return (preferred,) if minimum <= preferred <= maximum else (minimum,)
    if granularity == -1:
        return tuple(2**i for i in range(1, 31) if minimum <= 2**i <= maximum)
    if granularity > 0:
        return tuple(range(minimum, maximum + 1, granularity))
    return ()


def capabilities(device):
    # Generic labels explicitly identify a driver limitation, rather than inventing
    # physical connector names. PortAudio itself does not expose these on all APIs.
    inputs = tuple(
        f"Input {i + 1} (driver name unavailable)"
        for i in range(device.max_input_channels)
    )
    outputs = tuple(
        f"Output {i + 1} (driver name unavailable)"
        for i in range(device.max_output_channels)
    )
    sizes = ()
    note = "Native buffer choices are not exposed by this driver; its own buffer setting is used."
    try:
        import sounddevice as sd

        lib = ct.CDLL(sd._libname)
        if device.host_api == "ASIO":
            # https://files.portaudio.com/docs/v19-doxydocs/pa__asio_8h.html
            def names(kind, count):
                function = getattr(lib, f"PaAsio_Get{kind}ChannelName")
                function.argtypes = [ct.c_int, ct.c_int, ct.POINTER(ct.c_char_p)]
                function.restype = ct.c_int
                result = []
                for index in range(count):
                    value = ct.c_char_p()
                    error = function(device.index, index, ct.byref(value))
                    label = (
                        value.value.decode("utf-8", "replace").strip()
                        if not error and value.value
                        else ""
                    )
                    result.append(
                        label or f"{kind} {index + 1} (driver name unavailable)"
                    )
                return tuple(result)

            inputs = names("Input", device.max_input_channels)
            outputs = names("Output", device.max_output_channels)
            function = lib.PaAsio_GetAvailableBufferSizes
            function.argtypes = [ct.c_int, *([ct.POINTER(ct.c_long)] * 4)]
            function.restype = ct.c_int
            values = [ct.c_long() for _ in range(4)]
            if not function(device.index, *(ct.byref(v) for v in values)):
                sizes = buffer_values(*(v.value for v in values))
        elif device.host_api == "Core Audio":
            function = lib.PaMacCore_GetChannelName
            function.argtypes = [ct.c_int, ct.c_int, ct.c_bool]
            function.restype = ct.c_char_p

            def names(is_input, fallback):
                result = []
                for index, label in enumerate(fallback):
                    value = function(device.index, index, is_input)
                    result.append(value.decode("utf-8", "replace") if value else label)
                return tuple(result)

            inputs, outputs = names(True, inputs), names(False, outputs)
            inputs, outputs = coreaudio_names(device, inputs, outputs)
            function = lib.PaMacCore_GetBufferSizeRange
            function.argtypes = [ct.c_int, ct.POINTER(ct.c_long), ct.POINTER(ct.c_long)]
            function.restype = ct.c_int
            low, high = ct.c_long(), ct.c_long()
            if not function(device.index, ct.byref(low), ct.byref(high)):
                sizes = buffer_values(low.value, high.value, low.value, 1)
        elif device.host_api == "ALSA":
            inputs, outputs, sizes = alsa_capabilities(device, inputs, outputs)
        if sizes:
            note = (
                "Channel labels and buffer sizes are reported by the selected driver."
            )
    except (AttributeError, OSError, ValueError):
        pass
    return Capabilities(inputs, outputs, sizes, note)


def coreaudio_names(device, inputs, outputs):
    """Read per-element labels where PortAudio's incomplete helper returns NULL."""

    class Address(ct.Structure):
        _fields_ = [
            ("selector", ct.c_uint32),
            ("scope", ct.c_uint32),
            ("element", ct.c_uint32),
        ]

    def fourcc(text):
        return int.from_bytes(text.encode("ascii"), "big")

    audio = ct.CDLL(ctypes.util.find_library("CoreAudio"))
    foundation = ct.CDLL(ctypes.util.find_library("CoreFoundation"))
    foundation.CFStringGetCString.argtypes = [
        ct.c_void_p,
        ct.c_char_p,
        ct.c_long,
        ct.c_uint32,
    ]
    foundation.CFStringGetCString.restype = ct.c_bool
    foundation.CFRelease.argtypes = [ct.c_void_p]

    def string(object_id, selector, scope, element):
        address = Address(fourcc(selector), fourcc(scope), element)
        value, size = ct.c_void_p(), ct.c_uint32(ct.sizeof(ct.c_void_p))
        error = audio.AudioObjectGetPropertyData(
            object_id, ct.byref(address), 0, None, ct.byref(size), ct.byref(value)
        )
        if error or not value:
            return None
        try:
            data = ct.create_string_buffer(1024)
            return (
                data.value.decode("utf-8", "replace")
                if foundation.CFStringGetCString(value, data, len(data), 0x08000100)
                else None
            )
        finally:
            foundation.CFRelease(value)

    address, size = Address(fourcc("dev#"), fourcc("glob"), 0), ct.c_uint32()
    if audio.AudioObjectGetPropertyDataSize(
        1, ct.byref(address), 0, None, ct.byref(size)
    ):
        return inputs, outputs
    ids = (ct.c_uint32 * (size.value // 4))()
    if audio.AudioObjectGetPropertyData(
        1, ct.byref(address), 0, None, ct.byref(size), ids
    ):
        return inputs, outputs
    for identifier in ids:
        if string(identifier, "lnam", "glob", 0) == device.name:
            result = []
            for scope, fallback in (("inpt", inputs), ("outp", outputs)):
                result.append(
                    tuple(
                        string(identifier, "lchn", scope, i + 1)
                        or string(identifier, "lnam", scope, i + 1)
                        or name
                        for i, name in enumerate(fallback)
                    )
                )
            return tuple(result)
    return inputs, outputs


def alsa_capabilities(device, inputs, outputs):
    """Ask the ALSA hardware PCM for channel maps and supported period sizes."""
    match = re.search(r"\(hw:(\d+),(\d+)\)", device.name)
    if not match:
        return (
            inputs,
            outputs,
            (),
        )  # PulseAudio/JACK/server-managed devices expose no hardware range.
    alsa = ct.CDLL(ctypes.util.find_library("asound"))
    pointer = ct.c_void_p
    alsa.snd_pcm_open.argtypes = [ct.POINTER(pointer), ct.c_char_p, ct.c_int, ct.c_int]
    alsa.snd_pcm_close.argtypes = [pointer]
    alsa.snd_pcm_hw_params_malloc.argtypes = [ct.POINTER(pointer)]
    alsa.snd_pcm_hw_params_free.argtypes = [pointer]
    alsa.snd_pcm_hw_params_any.argtypes = [pointer, pointer]
    alsa.snd_pcm_hw_params_set_rate.argtypes = [pointer, pointer, ct.c_uint, ct.c_int]
    alsa.snd_pcm_hw_params_set_channels.argtypes = [pointer, pointer, ct.c_uint]
    for suffix in ("min", "max"):
        getattr(alsa, "snd_pcm_hw_params_get_period_size_" + suffix).argtypes = [
            pointer,
            ct.POINTER(ct.c_ulong),
            ct.POINTER(ct.c_int),
        ]
    alsa.snd_pcm_hw_params_test_period_size.argtypes = [
        pointer,
        pointer,
        ct.c_ulong,
        ct.c_int,
    ]

    class ChannelMap(ct.Structure):
        _fields_ = [
            ("type", ct.c_int),
            ("channels", ct.c_uint),
            ("positions", ct.c_uint * 128),
        ]

    alsa.snd_pcm_query_chmaps.argtypes = [pointer]
    alsa.snd_pcm_query_chmaps.restype = ct.POINTER(ct.POINTER(ChannelMap))
    alsa.snd_pcm_free_chmaps.argtypes = [ct.POINTER(ct.POINTER(ChannelMap))]
    alsa.snd_pcm_chmap_long_name.argtypes = [ct.c_uint]
    alsa.snd_pcm_chmap_long_name.restype = ct.c_char_p
    supported, labels = [], []
    for direction, fallback in ((1, inputs), (0, outputs)):
        pcm, params = pointer(), pointer()
        names, sizes = fallback, set()
        try:
            if alsa.snd_pcm_open(
                ct.byref(pcm), f"hw:{match[1]},{match[2]}".encode(), direction, 1
            ):
                continue
            maps = alsa.snd_pcm_query_chmaps(pcm)
            if maps:
                try:
                    index = 0
                    while maps[index]:
                        mapping = maps[index].contents
                        if (
                            mapping.channels == len(fallback)
                            and mapping.channels <= 128
                        ):
                            names = tuple(
                                (
                                    alsa.snd_pcm_chmap_long_name(mapping.positions[i])
                                    or fallback[i].encode()
                                ).decode("utf-8", "replace")
                                for i in range(mapping.channels)
                            )
                            break
                        index += 1
                finally:
                    alsa.snd_pcm_free_chmaps(maps)
            if alsa.snd_pcm_hw_params_malloc(
                ct.byref(params)
            ) or alsa.snd_pcm_hw_params_any(pcm, params):
                continue
            if alsa.snd_pcm_hw_params_set_rate(
                pcm, params, 48000, 0
            ) or alsa.snd_pcm_hw_params_set_channels(pcm, params, len(fallback)):
                continue
            low, high, rounding = ct.c_ulong(), ct.c_ulong(), ct.c_int()
            if alsa.snd_pcm_hw_params_get_period_size_min(
                params, ct.byref(low), ct.byref(rounding)
            ) or alsa.snd_pcm_hw_params_get_period_size_max(
                params, ct.byref(high), ct.byref(rounding)
            ):
                continue
            candidates = (
                range(low.value, high.value + 1)
                if high.value - low.value <= 16384
                else sorted(
                    {
                        low.value,
                        high.value,
                        *(
                            2**i
                            for i in range(1, 24)
                            if low.value <= 2**i <= high.value
                        ),
                    }
                )
            )
            sizes = {
                size
                for size in candidates
                if not alsa.snd_pcm_hw_params_test_period_size(pcm, params, size, 0)
            }
        finally:
            if params:
                alsa.snd_pcm_hw_params_free(params)
            if pcm:
                alsa.snd_pcm_close(pcm)
            labels.append(names)
            supported.append(sizes)
    return (*labels, tuple(sorted(supported[0] & supported[1])))

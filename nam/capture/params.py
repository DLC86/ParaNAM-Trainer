"""
User-facing continuous knob specifications for the capture app.

A :class:`KnobSpec` is what the app asks the user for: a knob's name, its printed
range, and the increment they can realistically dial it to. It converts to the
training-side :class:`nam.models.parametric.ParamSpec`, which is what the planner,
dataset, and generated model configs consume.

This module is deliberately torch-free so the GUI can import it at startup.
"""

from __future__ import annotations

import math as _math
from dataclasses import dataclass as _dataclass
from typing import Any as _Any
from typing import Optional as _Optional
from typing import Sequence as _Sequence

from ..models.parametric._spec import ParamSpec as _ParamSpec

DEFAULT_KNOB_STEP = 0.5


def _snap(value: float, minimum: float, maximum: float, step: float) -> float:
    snapped = minimum + round((value - minimum) / step) * step
    return min(max(snapped, minimum), maximum)


@_dataclass(frozen=True)
class KnobSpec:
    """
    A continuous knob to be captured.

    :param name: User-facing knob name (e.g. "Gain").
    :param min: Value at the knob's minimum position, in the units printed on the amp.
    :param max: Value at the knob's maximum position.
    :param step: Capture grid increment: planned settings are snapped to
        ``min + k * step`` so every planned value is one a human can actually dial.
    :param default: Resting value baked into the generated model config; the knob's
        midpoint (snapped to the grid) when omitted.
    :param avoid_zero: When true, no planned capture sets this knob to its minimum; a draw that
        would snap to the minimum uses the next reachable grid value instead. Use for
        gain/drive knobs that mute the rig at minimum and can destabilize training.
    :param is_gain: Marks this knob as the amp's gain/drive control. At most two knobs may
        be marked. It changes how the initial corner captures are generated (the gain knob
        gets its own min/max sweep so the EQ params' corners are learned at both gain
        extremes); it is capture-only metadata and does not affect the training-side
        :class:`ParamSpec`.
    """

    name: str
    min: float
    max: float
    step: float = DEFAULT_KNOB_STEP
    default: _Optional[float] = None
    avoid_zero: bool = False
    is_gain: bool = False
    avoid_max: bool = False
    sampling_bias: str = "linear"
    clean_volume: bool = False
    avoid_below: _Optional[float] = None
    avoid_above: _Optional[float] = None

    def __post_init__(self):
        if not str(self.name).strip():
            raise ValueError("Knob name must be non-empty")
        object.__setattr__(self, "name", str(self.name).strip())
        for field in ("min", "max", "step"):
            value = getattr(self, field)
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Knob {self.name!r} {field} must be a number; got {value!r}"
                ) from exc
            if not _math.isfinite(value):
                raise ValueError(f"Knob {self.name!r} {field} must be finite")
            object.__setattr__(self, field, value)
        if self.min >= self.max:
            raise ValueError(
                f"Knob {self.name!r} must satisfy min < max; got {self.min} >= {self.max}"
            )
        if self.step <= 0.0:
            raise ValueError(
                f"Knob {self.name!r} step must be positive; got {self.step}"
            )
        if self.step > self.max - self.min:
            raise ValueError(
                f"Knob {self.name!r} step must not exceed the range width; "
                f"got {self.step} > {self.max - self.min}"
            )
        if self.default is None:
            object.__setattr__(
                self,
                "default",
                _snap(0.5 * (self.min + self.max), self.min, self.max, self.step),
            )
        else:
            default = float(self.default)
            if not _math.isfinite(default):
                raise ValueError(f"Knob {self.name!r} default must be finite")
            if not (self.min <= default <= self.max):
                raise ValueError(
                    f"Knob {self.name!r} default must be within [{self.min}, {self.max}]; "
                    f"got {default}"
                )
            object.__setattr__(self, "default", default)
        object.__setattr__(self, "avoid_zero", bool(self.avoid_zero))
        object.__setattr__(self, "is_gain", bool(self.is_gain))
        for field in ("avoid_below", "avoid_above"):
            limit = getattr(self, field)
            if limit is not None:
                limit = float(limit)
                if not _math.isfinite(limit):
                    raise ValueError(f"{field} must be finite")
                object.__setattr__(self, field, limit)
        if self.avoid_below is not None and self.avoid_above is not None:
            if self.avoid_below > self.avoid_above:
                raise ValueError("Avoid < must be at or below Avoid >")
        if self.sampling_bias not in ("linear", "lower", "upper"):
            raise ValueError("Choose uniform, lower-half or upper-half sampling")
        if self.clean_volume and (self.avoid_max or self.is_gain or self.avoid_above is not None or self.avoid_below is not None):
            raise ValueError(
                "Linear volume cannot use Gain/Drive or capture thresholds"
            )
        if self.avoid_max and self.avoid_zero and not (
            (_math.floor(self.min / self.step + 1e-10) + 1) * self.step
            < self.max - 1e-10
        ):
            raise ValueError("No usable knob positions remain after Avoid min/Avoid max")

    def to_param_spec(self) -> _ParamSpec:
        return _ParamSpec(
            name=self.name,
            min=self.min,
            max=self.max,
            default=self.default,
            type="continuous",
            step=self.step,
            avoid_zero=self.avoid_zero,
            avoid_max=self.avoid_max,
            avoid_below=self.avoid_below,
            avoid_above=self.avoid_above,
            sampling_bias=self.sampling_bias,
            clean_volume=self.clean_volume,
        )

    def to_dict(self) -> dict[str, _Any]:
        return {
            "name": self.name,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "default": self.default,
            "avoid_zero": self.avoid_zero,
            "is_gain": self.is_gain,
            **({"avoid_max": True} if self.avoid_max else {}),
            **({"avoid_below": self.avoid_below} if self.avoid_below is not None else {}),
            **({"avoid_above": self.avoid_above} if self.avoid_above is not None else {}),
            **(
                {"sampling_bias": self.sampling_bias}
                if self.sampling_bias != "linear"
                else {}
            ),
            **({"clean_volume": True} if self.clean_volume else {}),
        }

    @classmethod
    def from_dict(cls, d: dict[str, _Any]) -> "KnobSpec":
        config = dict(d)
        missing = [key for key in ("name", "min", "max") if key not in config]
        if missing:
            raise ValueError(
                f"Knob config is missing required field(s): {', '.join(missing)}"
            )
        return cls(
            name=config["name"],
            min=config["min"],
            max=config["max"],
            step=config.get("step", DEFAULT_KNOB_STEP),
            default=config.get("default"),
            avoid_zero=config.get("avoid_zero", False),
            is_gain=config.get("is_gain", False),
            avoid_max=config.get("avoid_max", False),
            avoid_below=config.get("avoid_below"),
            avoid_above=config.get("avoid_above"),
            sampling_bias=config.get("sampling_bias", "linear"),
            clean_volume=config.get("clean_volume", False),
        )


@_dataclass(frozen=True)
class SwitchSpec:
    """A categorical control; position order and labels are preserved in exports."""

    name: str
    enum_names: tuple[str, ...]
    default: int = 0
    planning_mode: str = "complex"
    option_indices: _Optional[tuple[int, ...]] = None

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.name).strip())
        if self.planning_mode not in ("complex", "simple"):
            raise ValueError("Choose Complex Switch or Simple Switch")
        if isinstance(self.enum_names, (str, bytes)):
            raise ValueError("Switch positions must be a sequence of names")
        labels = tuple(str(label).strip() for label in self.enum_names)
        if len({label.casefold() for label in labels}) != len(labels):
            raise ValueError("Switch position names must be unique (ignoring case)")
        object.__setattr__(self, "enum_names", labels)
        if self.option_indices is not None:
            object.__setattr__(self, "option_indices", tuple(self.option_indices))
        self.to_param_spec()  # Validate the name, count and default index.

    @property
    def is_gain(self) -> bool:
        return False

    def to_param_spec(self) -> _ParamSpec:
        return _ParamSpec(
            self.name,
            0,
            len(self.enum_names) - 1,
            self.default,
            type="switch",
            enum_names=self.enum_names,
            simple_switch=self.planning_mode == "simple",
            option_indices=self.option_indices,
        )

    def to_dict(self) -> dict[str, _Any]:
        return {
            "type": "switch",
            "name": self.name,
            "enum_names": list(self.enum_names),
            "default": self.default,
            **({"option_indices": list(self.option_indices)} if self.option_indices is not None else {}),
            **({"planning_mode": "simple"} if self.planning_mode == "simple" else {}),
        }


ControlSpec = KnobSpec | SwitchSpec


def capture_grid_limits(controls: _Sequence[ControlSpec]) -> tuple[int, int, int]:
    """(settings per Complex Switch state, held-out slots per state, state count).

    Matches the planner's on-grid sampling, default anchor and reserved
    corners. Clean output volume is measured separately and adds no dimension.
    """
    active = [c for c in controls if not getattr(c, "clean_volume", False)]
    combinations = _math.prod(
        len(c.enum_names)
        for c in active
        if isinstance(c, SwitchSpec) and c.planning_mode == "complex"
    )
    per_state = [
        c for c in active
        if not (isinstance(c, SwitchSpec) and c.planning_mode == "complex")
    ]
    if not any(isinstance(c, KnobSpec) for c in per_state):
        return (
            _math.prod(
                len(c.enum_names) for c in per_state
                if isinstance(c, SwitchSpec)
            ),
            0,
            combinations,
        )

    # Lazy import avoids a params <-> planner import cycle at GUI startup.
    from .planner import _grid_capacity, corner_settings, settings_on_capture_grid

    specs = tuple(c.to_param_spec() for c in per_state)
    capacity = _grid_capacity(specs, DEFAULT_KNOB_STEP)
    anchor = {
        s.name: s.enum_names[int(s.default)] if s.type == "switch" else s.default
        for s in specs
    }
    if not settings_on_capture_grid(anchor, specs, DEFAULT_KNOB_STEP):
        capacity += 1
    reserved = {
        tuple(point[s.name] for s in specs)
        for point in corner_settings(
            specs, gain_index=gain_knob_indices(per_state),
            default_step=DEFAULT_KNOB_STEP,
        )
    }
    reserved.add(tuple(anchor[s.name] for s in specs))
    return capacity, max(0, capacity - len(reserved)), combinations


def capture_count_maxima(
    controls: _Sequence[ControlSpec], n_train: int, n_validation: int
) -> tuple[int, int]:
    """Largest feasible totals for each spinner, given the other split.

    Complex Switch states are allotted any remainder to the *same* early
    states, so use rounded-up per-state quotas, not just the overall total.
    This is for distinct held-out settings, not the switch-only second-audio
    validation supported by the main trainer UI.
    """
    capacity, held_out_capacity, combinations = capture_grid_limits(controls)
    train_max = combinations * max(
        0, capacity - _math.ceil(n_validation / combinations)
    )
    validation_max = combinations * min(
        held_out_capacity,
        max(0, capacity - _math.ceil(n_train / combinations)),
    )
    return train_max, validation_max


def capture_recommendations(controls: _Sequence[ControlSpec]) -> tuple[int, int, int]:
    """Recommend feasible train/validation counts on the actual capture grid.

    Counts are totals across Complex Switch states. Validation avoids corner
    settings and the default training anchor, as the planner itself requires.
    """
    active = [c for c in controls if not getattr(c, "clean_volume", False)]
    n_knobs = sum(
        isinstance(c, KnobSpec)
        or (isinstance(c, SwitchSpec) and c.planning_mode == "simple")
        for c in active
    )
    combinations = _math.prod(
        len(c.enum_names)
        for c in active
        if isinstance(c, SwitchSpec) and c.planning_mode == "complex"
    )
    train = max(
        1,
        16 * n_knobs,
        max(
            (
                len(c.enum_names)
                for c in active
                if isinstance(c, SwitchSpec) and c.planning_mode == "simple"
            ),
            default=1,
        ),
    )
    validation = min(max(8, 4 * n_knobs), 24) if n_knobs else 0
    capacity, validation_capacity, _ = capture_grid_limits(controls)
    if active and all(isinstance(c, SwitchSpec) for c in active):
        # Capture every categorical state. The main trainer can additionally
        # record the *same* states using independent validation audio.
        train = capacity
        validation = 0
    elif any(isinstance(c, KnobSpec) for c in active):
        budget = min(capacity, train + validation)
        if budget < 4 or not validation_capacity:
            train = min(train, capacity)
            validation = 0
        else:
            held_out = min(
                validation, validation_capacity, max(1, _math.ceil(budget / 5))
            )
            train = min(train, budget - held_out)
            validation = min(validation, validation_capacity, budget - train)
    return train * combinations, validation * combinations, combinations

def validate_knobs(knobs: _Sequence[ControlSpec]) -> tuple[ControlSpec, ...]:
    """
    Validate a knob set as a whole and return it as a tuple.

    Names must be unique case-insensitively: capture filenames encode params via
    case-insensitive unique-prefix abbreviations, so "gain" and "Gain" would collide.
    """
    knobs = tuple(knobs)
    if len(knobs) == 0:
        raise ValueError("At least one control is required")
    seen: dict[str, str] = {}
    for knob in knobs:
        key = knob.name.lower()
        if key in seen:
            raise ValueError(
                f"Duplicate knob name: {knob.name!r} collides with {seen[key]!r}"
            )
        seen[key] = knob.name
    if sum(getattr(c, "clean_volume", False) for c in knobs) > 1:
        raise ValueError("At most one Clean output volume is supported")
    gain_names = [knob.name for knob in knobs if knob.is_gain]
    if len(gain_names) > 2:
        raise ValueError(
            "At most two knobs may be marked Gain/Drive; "
            f"got {', '.join(repr(name) for name in gain_names)}"
        )
    return knobs


def gain_knob_index(knobs: _Sequence[ControlSpec]) -> _Optional[int]:
    """
    Position of the knob marked Gain/Drive, or ``None`` if none is. Assumes the knob set
    has passed :func:`validate_knobs` (at most two gain knobs; this legacy helper returns the first).
    """
    for index, knob in enumerate(knobs):
        if knob.is_gain:
            return index
    return None


def gain_knob_indices(knobs: _Sequence[ControlSpec]) -> tuple[int, ...]:
    """All gain/drive dimensions, in capture-control order."""
    return tuple(i for i, knob in enumerate(knobs) if knob.is_gain)

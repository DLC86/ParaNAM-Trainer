"""
Latin-Hypercube capture planning.

The sampling core is adapted from PANAMA (Parametric Active-learning for Neural Amp
Modeling Assistance), arXiv:2509.26564v1. The Latin-Hypercube starter-set idea and the
active-learning capture workflow are due to the PANAMA authors. It moved here from
``scripts/make_starter_settings.py`` so the capture app and the starter-settings CLI
share one implementation; the script now imports from this module.

On top of the raw sampling, :func:`plan_captures` produces the app-facing plan: an
ordered list of train and held-out validation settings, quantized to each knob's
capture grid, with filenames that encode the settings.
"""

from __future__ import annotations

import itertools as _itertools
import math as _math
from collections.abc import Sequence as _Sequence
from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace
from typing import Any as _Any
from typing import Literal
from typing import Optional as _Optional

import numpy as _np

from ..models.parametric import abbreviate_param_names as _abbreviate_param_names
from ..models.parametric import decode_named_params as _decode_named_params
from ..models.parametric import make_capture_y_path as _make_capture_y_path
from ..models.parametric import quantize_to_capture_grid as _quantize_to_capture_grid
from ..models.parametric import switch_combinations as _switch_combinations
from ..models.parametric import ParamSpec as _ParamSpec
from .params import DEFAULT_KNOB_STEP as _DEFAULT_KNOB_STEP
from .params import gain_knob_indices as _gain_knob_index
from .params import ControlSpec as _ControlSpec
from .params import validate_knobs as _validate_knobs
from .plan_rules import allowed_grid_size, allows, bound_rules, validate_plan_rules

# Large, fixed offset so the validation draws are a different LHS stream than the train
# draws (held-out settings), while staying reproducible from the same seed.
VALIDATION_SEED_OFFSET = 2**31 - 1

CAPTURES_DIRNAME = "captures"
# Untouched recordings kept alongside each capture: the amp return exactly as it came
# back, and (when a loopback is patched) the loopback return, both still carrying the
# timing blips. Nothing in training reads these; they exist so a session whose delay or
# alignment turns out to be wrong can be reconstructed from what was actually recorded.
CAPTURES_RAW_DIRNAME = "captures_raw"
# The loopback recording is the capture's own filename with this appended; the amp return
# keeps the filename unchanged. A suffix rather than a prefix so the two sort next to each
# other -- the pair for one setting can be read off a file listing in order and dropped
# into a DAW as adjacent tracks. It also derives from any capture name by appending,
# which a prefix inserted after the index number would not: active-learning captures are
# named ``r2_g4.5_bOn.wav`` (see nam.train.active_learning) and have no index group to
# insert after.
RAW_LOOPBACK_SUFFIX = "_lb"
# Describes captures_raw/ well enough to recover the reamp from it: for each capture, how
# much preamble and tail surround the input audio. See nam.capture.session.
RAW_MANIFEST_FILENAME = "manifest.json"
# An input WAV with the blip preamble in front and the tail behind: the stream that was
# actually played, kept in captures_raw/ so a recording can be correlated against exactly
# what produced it. Derived from the input's own name rather than being a fixed pair of
# filenames, so a project with more than two inputs gets one per input.
PLAYBACK_INPUT_SUFFIX = "_w_preamble_tail"

# Quantizing Latin-Hypercube draws to the realizable knob grid collapses many distinct
# continuous samples onto the same grid point, so raw draws produce duplicate settings
# (and validation draws that coincide with train ones). Every planned capture must be a
# distinct setting, so the split planner deduplicates on the quantized settings. When the
# grid is coarse enough that oversampling keeps colliding, it falls back to enumerating
# the grid directly. These bound that work.
_OVERSAMPLE_BATCHES = 12
# Above this many distinct grid points, never materialize the full grid: a grid this large
# is far finer than any realistic capture count, so oversampling fills without collisions
# and the enumeration fallback is unreachable in practice.
_ENUMERATION_CAP = 5_000_000


def _capture_grid_values(spec: _ParamSpec, default_step: float) -> list[float]:
    """
    The distinct continuous values reachable on ``spec``'s capture grid, i.e. every value
    :func:`quantize_to_capture_grid` can emit for it. Quantization snaps to the nearest
    multiple of ``step`` (measured from zero, matching the shared helper) and clamps into
    ``[min, max]``, so the reachable set is those in-range multiples plus the clamped
    endpoints.
    """
    step = spec.step if spec.step is not None else default_step
    lo = _math.floor(spec.min / step)
    hi = _math.ceil(spec.max / step)
    values = set()
    # ``round(v / step)`` for ``v`` in ``[min, max]`` lands in ``[lo, hi]``; the +/-1 pad
    # covers rounding at the endpoints. Out-of-range multiples clamp onto min/max.
    for k in range(lo - 1, hi + 2):
        clamped = min(max(k * step, spec.min), spec.max)
        values.add(round(clamped, 6))
    if getattr(spec, "avoid_zero", False):
        # The saved avoid_zero key means Avoid min in the capture plan.
        values.discard(round(spec.min, 6))
    if getattr(spec, "avoid_max", False):
        values.discard(round(spec.max, 6))
    return sorted(
        value for value in values
        if (spec.avoid_below is None or value >= spec.avoid_below)
        and (spec.avoid_above is None or value <= spec.avoid_above)
    )


def _grid_capacity(specs: _Sequence[_ParamSpec], default_step: float, rules=()) -> int:
    """Total number of distinct on-grid settings (switch combinations x continuous grid)."""
    if rules:
        axes = (
            (spec.name, spec.enum_names if spec.type == "switch"
             else _capture_grid_values(spec, default_step))
            for spec in specs
        )
        return allowed_grid_size(axes, rules)
    capacity = 1
    for spec in specs:
        if spec.type == "switch":
            capacity *= spec.num_inputs
        else:
            capacity *= len(_capture_grid_values(spec, default_step))
    return capacity


def _enumerate_grid_params(
    specs: _Sequence[_ParamSpec],
    default_step: float,
    *,
    seed: int,
):
    """
    Yield every distinct on-grid setting as a decoded params dict, in a deterministic
    shuffled order. Only used as the coarse-grid fallback, guarded by ``_ENUMERATION_CAP``.
    """
    value_axes: list[list[float]] = []
    for spec in specs:
        if spec.type == "switch":
            value_axes.append([float(i) for i in range(spec.num_inputs)])
        else:
            value_axes.append(_capture_grid_values(spec, default_step))
    combos = list(_itertools.product(*value_axes))
    _np.random.default_rng(seed).shuffle(combos)
    for combo in combos:
        yield _decode_named_params(_np.asarray(combo, dtype=_np.float64), specs)


def _settings_key(
    params: dict[str, _Any], specs: _Sequence[_ParamSpec]
) -> tuple[_Any, ...]:
    return tuple(params[spec.name] for spec in specs)


def settings_on_capture_grid(params, specs, default_step=_DEFAULT_KNOB_STEP):
    """An off-grid default is captured but does not consume a grid candidate."""
    raw = _np.asarray(
        [
            (
                spec.enum_names.index(params[spec.name])
                if spec.type == "switch"
                else params[spec.name]
            )
            for spec in specs
        ],
        dtype=float,
    )
    snapped = _decode_named_params(
        _quantize_to_capture_grid(raw, specs, default_step=default_step), specs
    )
    return _settings_key(params, specs) == _settings_key(snapped, specs)


def settings_sort_key(
    params: dict[str, _Any], names: _Sequence[str]
) -> tuple[tuple[int, _Any], ...]:
    """
    Ordering key for one capture setting: its values in knob order. Sorting a list of
    settings by this walks the last knob through its range for every value of the knob
    before it, and so on, which is the order that asks the user to move the fewest knobs
    between one capture and the next.

    Each value is paired with a type rank so a numeric knob and a switch's label can never
    be compared against each other. Within one knob every setting has the same type, so the
    rank is constant down that column and the value is what orders it.
    """
    key: list[tuple[int, _Any]] = []
    for name in names:
        value = params[name]
        try:
            key.append((0, float(value)))
        except (TypeError, ValueError):
            key.append((1, str(value)))
    return tuple(key)


def sort_settings(
    param_dicts: _Sequence[dict[str, _Any]], names: _Sequence[str]
) -> list[dict[str, _Any]]:
    """Capture settings in ascending :func:`settings_sort_key` order."""
    return sorted(param_dicts, key=lambda params: settings_sort_key(params, names))


def sample_unique_settings(
    specs: _Sequence[_ParamSpec],
    n: int,
    *,
    seed: int,
    default_step: float,
    lhs_method: str = "standard",
    exclude: _Optional[set[tuple[_Any, ...]]] = None,
    anchor: _Optional[dict[str, _Any]] = None,
    rules=(),
) -> list[dict[str, _Any]]:
    """
    Draw ``n`` distinct on-grid settings as decoded params dicts, none of them in
    ``exclude`` (used to hold one split out of another). Each draw is quantized to the
    capture grid (nearest ``default_step``, per-``ParamSpec`` ``step`` honored) before the
    uniqueness test, so two draws that snap to the same knob positions count as one.

    Shared by the capture app (:func:`plan_captures`) and the starter-settings script so
    both emit unique, held-out settings from one implementation.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}")
    seen = set() if exclude is None else set(exclude)
    ordered: list[dict[str, _Any]] = []
    if n == 0:
        return ordered
    if anchor is not None:
        key = _settings_key(anchor, specs)
        if key in seen:
            raise ValueError(
                "The default setting must belong to training, not validation"
            )
        ordered.append(dict(anchor))
        seen.add(key)
        if n == 1:
            return ordered

    def _consume(raw_settings: list[_np.ndarray]) -> bool:
        for raw in raw_settings:
            quantized = _quantize_to_capture_grid(raw, specs, default_step=default_step)
            params = _decode_named_params(quantized, specs)
            key = _settings_key(params, specs)
            if key in seen or not allows(rules, params):
                continue
            seen.add(key)
            ordered.append(params)
            if len(ordered) == n:
                return True
        return False

    # The first batch reproduces the historical LHS stream exactly, so a plan with no grid
    # collisions is byte-for-byte identical to the pre-deduplication behavior.
    raw = sample_raw_settings(specs, n, seed=seed, lhs_method=lhs_method)
    if anchor is not None:
        # Keep the default row's stratum occupied in every dimension. Swapping
        # coordinates preserves the other Latin strata before grid quantization.
        raw = _np.asarray(raw).copy()
        for column, spec in enumerate(specs):
            if spec.type == "switch":
                value = spec.enum_names.index(anchor[spec.name])
                matching = _np.flatnonzero(raw[:, column] == value)
                if len(matching):
                    row = int(matching[0])
                    raw[0, column], raw[row, column] = raw[row, column], raw[0, column]
                else:
                    raw[0, column] = value
                continue
            value = float(anchor[spec.name])
            width = spec.max - spec.min
            inverse = lambda v: _inverse_bias((v - spec.min) / width, spec)
            strata = _np.minimum(n - 1, (inverse(raw[:, column]) * n).astype(int))
            target = min(n - 1, int(inverse(value) * n))
            matching = _np.flatnonzero(strata == target)
            if len(matching):
                row = int(matching[0])
            else:
                # Scrambled Sobol is not guaranteed to occupy all n equal-width
                # Latin strata when n is not a power of two. Still preserve the
                # nearest available sample when pinning the required default.
                row = int(_np.argmin(_np.abs(inverse(raw[:, column]) - inverse(value))))
            raw[0, column], raw[row, column] = raw[row, column], raw[0, column]
            raw[0, column] = value
        raw = raw[1:]
    if _consume(raw):
        return ordered

    # Collisions dropped us short. Oversample with fresh, seed-derived LHS batches to fill
    # in the gaps while preserving space-filling coverage.
    rng = _np.random.default_rng(seed)
    batch = max(n, 256)
    if lhs_method == "orthogonal":
        # The collision-recovery batches must also satisfy the strict
        # strength-2 OA size requirement. The ordinary 256 batch is invalid.
        p = max(2, _math.isqrt(batch - 1) + 1, len(specs) - 1)
        while any(p % divisor == 0 for divisor in range(2, _math.isqrt(p) + 1)):
            p += 1
        batch = p * p
    for _ in range(_OVERSAMPLE_BATCHES):
        if _consume(
            sample_raw_settings(
                specs, batch, seed=int(rng.integers(0, 2**32)), lhs_method=lhs_method
            )
        ):
            return ordered

    # The grid is coarse enough that random draws keep colliding. Fill the remainder
    # deterministically from the full grid so we never loop indefinitely.
    capacity = _grid_capacity(specs, default_step)
    if capacity > _ENUMERATION_CAP:
        raise ValueError(
            f"Unable to draw {n} unique capture settings after oversampling. The knob "
            "grid is too large to enumerate; widen the sampling or reduce the count."
        )
    for params in _enumerate_grid_params(
        specs, default_step, seed=int(rng.integers(0, 2**32))
    ):
        key = _settings_key(params, specs)
        if key in seen or not allows(rules, params):
            continue
        seen.add(key)
        ordered.append(params)
        if len(ordered) == n:
            return ordered

    raise ValueError(
        f"Cannot plan {n} unique settings: the knob grid yields only {len(ordered)} "
        "distinct settings beyond those already used. Reduce the capture count, widen a "
        "knob's range, or use a finer step."
    )



def _farthest_validation(
    specs: _Sequence[_ParamSpec],
    train: list[dict[str, _Any]],
    reserved: set[tuple[_Any, ...]],
    n_validation: int,
    *,
    seed: int,
    default_step: float,
    rules=(),
) -> list[dict[str, _Any]]:
    """Greedy maximin holdout against train *and* the training corner settings.

    Candidates are sampled reproducibly on the actual capture grid, not in an
    unquantized unit cube. Complex-switch combinations are handled separately by
    plan_captures, which preserves their per-combination validation quotas.
    """
    if not n_validation:
        return []

    train_keys = {_settings_key(point, specs) for point in train}
    excluded = train_keys | reserved
    # A user-defined default can be off the capture grid. It belongs to train
    # but cannot consume a grid candidate from this validation pool.
    grid_exclusions = sum(
        settings_on_capture_grid(
            dict(zip((spec.name for spec in specs), key)), specs, default_step
        )
        for key in excluded
    )
    available = _grid_capacity(specs, default_step, rules) - grid_exclusions
    if available < n_validation:
        raise ValueError("Not enough non-training, non-corner settings for validation")
    pool_size = min(available, max(n_validation, min(4096, max(256, 20 * n_validation))))
    # The pool is another deterministic LHS draw. Even if the train uses a
    # strength-2 orthogonal design, the pool need not have that special size.
    candidates = sample_unique_settings(
        specs, pool_size, seed=seed, default_step=default_step,
        lhs_method="standard", exclude=excluded, rules=rules,
    )

    def vector(points):
        columns = []
        for spec in specs:
            if spec.type == "switch":
                values = _np.asarray(
                    [spec.enum_names.index(point[spec.name]) for point in points],
                    dtype=int,
                )
                # All different switch states have equal distance; numeric
                # encoding would incorrectly make state 0 farther from 2 than 1.
                columns.append(
                    _np.eye(spec.num_inputs, dtype=float)[values] / _math.sqrt(2)
                )
            else:
                columns.append(
                    _np.asarray(
                        [(point[spec.name] - spec.min) / (spec.max - spec.min)
                         for point in points], dtype=float,
                    )[:, None]
                )
        return _np.concatenate(columns, axis=1)

    # A corner remains a train point even if it is appended after this function.
    corners = [
        dict(zip((spec.name for spec in specs), key))
        for key in sorted(reserved, key=repr)
        if key not in train_keys
    ]
    candidate_vectors = vector(candidates)
    training_vectors = vector(train + corners) if train or corners else None
    if training_vectors is None:
        distances = _np.full(len(candidates), _np.inf)
    else:
        from scipy.spatial import cKDTree

        distances = cKDTree(training_vectors).query(
            candidate_vectors, k=1, workers=1
        )[0] ** 2

    # Prevent the greedy process from concentrating an entire validation split
    # in one Simple Switch category while preserving switch-state coverage.
    simple = [spec for spec in specs if spec.type == "switch"]
    groups = [
        tuple(point[spec.name] for spec in simple) for point in candidates
    ]
    capacities: dict[tuple[_Any, ...], int] = {}
    for group in groups:
        capacities[group] = capacities.get(group, 0) + 1
    quotas = dict.fromkeys(capacities, 0)
    for _ in range(n_validation):
        group = min(
            (g for g in capacities if quotas[g] < capacities[g]),
            key=lambda g: (quotas[g], repr(g)),
        )
        quotas[group] += 1

    selected: list[dict[str, _Any]] = []
    used = _np.zeros(len(candidates), dtype=bool)
    counts = dict.fromkeys(capacities, 0)
    for _ in range(n_validation):
        eligible = _np.asarray([
            not used[i] and counts[group] < quotas[group]
            for i, group in enumerate(groups)
        ])
        if not eligible.any():
            eligible = ~used
        score = _np.where(eligible, distances, -_np.inf)
        index = int(_np.argmax(score))  # stable deterministic tie-break
        selected.append(candidates[index])
        used[index] = True
        counts[groups[index]] += 1
        sq = _np.sum(
            (candidate_vectors - candidate_vectors[index]) ** 2, axis=1
        )
        distances = _np.minimum(distances, sq)
    return selected



def plan_unique_splits(
    specs: _Sequence[_ParamSpec],
    *,
    n_train: int,
    n_validation: int,
    seed: int,
    default_step: float,
    lhs_method: str = "standard",
    validation_method: str = "independent",
    validation_exclude: _Optional[set[tuple[_Any, ...]]] = None,
    anchor: _Optional[dict[str, _Any]] = None,
    rules=(),
    anchor_policy: str = "inside",
    external_exclude: _Optional[set[tuple[_Any, ...]]] = None,
) -> tuple[list[dict[str, _Any]], list[dict[str, _Any]]]:
    """
    Draw ``n_train`` + ``n_validation`` distinct on-grid settings, with the validation
    settings held out of the train settings. Validation is a separate, reproducible LHS
    stream (see :data:`VALIDATION_SEED_OFFSET`). Returns the two lists of decoded params
    dicts.

    ``validation_exclude`` is a set of settings keys the validation draw must avoid on top
    of the train settings; :func:`plan_captures` passes the corner settings so a held-out
    point can never claim a knob extreme that then has to be skipped when the corner
    captures are added.

    Shared by :func:`plan_captures` (capture app) and ``build_starter_data`` (starter
    script) so both deduplicate and hold out validation the same way.
    """
    if n_train < 0:
        raise ValueError(f"n_train must be non-negative; got {n_train}")
    if n_validation < 0:
        raise ValueError(f"n_validation must be non-negative; got {n_validation}")
    if validation_method not in ("independent", "farthest"):
        raise ValueError("Unknown validation selection method")
    # The reservation only constrains the validation draw, so it costs nothing when there
    # is no validation split.
    outside = set(external_exclude or ())
    reserved = (
        set()
        if validation_exclude is None or n_validation == 0
        else set(validation_exclude)
    ) | outside
    if anchor_policy not in ("inside", "extra", "exclude", "sample"):
        raise ValueError(f"Unknown default capture policy: {anchor_policy}")
    if anchor_policy == "sample":
        anchor = None
    if anchor is not None and anchor_policy == "inside" and n_train < 1:
        raise ValueError("The default setting requires a training capture")
    default_key = _settings_key(anchor, specs) if anchor is not None else None
    default_on_grid = (
        settings_on_capture_grid(anchor, specs, default_step)
        if anchor is not None else False
    )
    # The default is reserved from validation in all modes. For the explicit
    # modes it is never inserted into the sampled LHS/Sobol design: it is either
    # appended as an additional training capture, or completely excluded.
    if default_key is not None and (default_on_grid or anchor_policy != "exclude"):
        reserved.add(default_key)
    sampled_exclusions = outside | (
        {default_key}
        if default_key is not None and anchor_policy in ("extra", "exclude")
        else set()
    )
    sampled_anchor = anchor if anchor_policy == "inside" else None
    def _with_extra(points):
        return points + [dict(anchor)] if anchor_policy == "extra" and anchor is not None else points

    # Every train and validation setting must be distinct, and validation must be held out
    # from train. That is only possible if the knob grid has at least that many distinct
    # settings; fail loudly up front rather than after sampling work.
    required = n_train + n_validation
    if default_key is not None and default_on_grid and anchor_policy in ("extra", "exclude"):
        # The default consumes a grid combination even though it does not
        # consume one of the requested LHS/Sobol sample positions.
        required += 1
    capacity = _grid_capacity(specs, default_step, rules)
    # Complex Switch groups may request complementary knob settings. Exclude
    # points used by earlier groups only when the joint grid can afford them.
    capacity -= sum(
        1 for key in outside
        if settings_on_capture_grid(
            dict(zip((spec.name for spec in specs), key)), specs, default_step
        ) and allows(rules, dict(zip((spec.name for spec in specs), key)))
    )
    if anchor is not None and not default_on_grid and anchor_policy in ("inside", "extra"):
        # Explicit defaults may fall between the ordinary capture-grid positions.
        capacity += 1
    if required > capacity:
        raise ValueError(
            f"Cannot plan {n_train} train + {n_validation} validation unique settings: the "
            f"knob grid yields only {capacity} distinct settings. Reduce the capture "
            "counts, widen a knob's range, or use a finer step."
        )
    if n_validation > capacity - len(reserved):
        raise ValueError(
            f"Cannot hold out {n_validation} validation settings: of the {capacity} distinct "
            f"settings on the knob grid, {len(reserved)} are reserved for the corner "
            "captures. Reduce the validation count, widen a knob's range, or use a finer "
            "step."
        )

    validation_seed = (seed + VALIDATION_SEED_OFFSET) % 2**32
    if validation_method == "farthest" and n_validation:
        # Training owns its original LHS design; unlike the independent method,
        # validation must be picked *after* seeing the train locations.
        train = sample_unique_settings(
            specs, n_train, seed=seed, default_step=default_step,
            lhs_method=lhs_method, anchor=sampled_anchor,
            exclude=sampled_exclusions, rules=rules,
        )
        train_keys = {_settings_key(point, specs) for point in train}
        if capacity - len(reserved | train_keys) < n_validation:
            # On a very coarse grid train-first can consume every non-corner
            # setting. Retain the old feasible reservation strategy in this case.
            train, _ = plan_unique_splits(
                specs, n_train=n_train, n_validation=n_validation, seed=seed,
                default_step=default_step, lhs_method=lhs_method,
                validation_method="independent",
                validation_exclude=validation_exclude, anchor=anchor,
                rules=rules, anchor_policy=anchor_policy,
                external_exclude=outside,
            )
            if anchor_policy == "extra":
                train = train[:-1]  # the fallback already appended this extra capture
        final_train = _with_extra(train)
        return final_train, _farthest_validation(
            specs, final_train, reserved, n_validation, seed=validation_seed,
            default_step=default_step, rules=rules,
        )
    if not reserved:
        train = sample_unique_settings(
            specs, n_train, seed=seed, default_step=default_step,
            lhs_method=lhs_method, exclude=sampled_exclusions, rules=rules,
        )
        validation = sample_unique_settings(
            specs,
            n_validation,
            seed=validation_seed,
            default_step=default_step,
            lhs_method=lhs_method,
            exclude={_settings_key(params, specs) for params in train},
            rules=rules,
        )
        return _with_extra(train), validation

    # With settings reserved away from validation, draw validation first: taking train first
    # could consume every unreserved setting and leave validation nothing to draw from, even
    # when the grid is large enough for both splits.
    validation = sample_unique_settings(
        specs,
        n_validation,
        seed=validation_seed,
        default_step=default_step,
        lhs_method=lhs_method,
        exclude=reserved,
        rules=rules,
    )
    train = sample_unique_settings(
        specs,
        n_train,
        seed=seed,
        default_step=default_step,
        lhs_method=lhs_method,
        exclude={_settings_key(params, specs) for params in validation} | sampled_exclusions,
        anchor=sampled_anchor,
        rules=rules,
    )
    return _with_extra(train), validation


LHS_METHODS = ("standard", "maximin", "orthogonal", "near_orthogonal", "sobol", "optimized")


def _standard_lhs(n: int, dim: int, seed: int) -> _np.ndarray:
    """Preserve existing seeded LHS behavior for older projects."""
    try:
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=dim, rng=_np.random.default_rng(seed))
        return _np.asarray(sampler.random(n=n), dtype=_np.float64)
    except (ImportError, TypeError):
        rng = _np.random.default_rng(seed)
        samples = _np.empty((n, dim), dtype=_np.float64)
        for column in range(dim):
            samples[:, column] = (rng.permutation(n) + rng.random(n)) / n
        return samples


def latin_hypercube_unit(
    n: int,
    dim: int,
    *,
    seed: int,
    lhs_method: str = "standard",
) -> _np.ndarray:
    """Create a reproducible Latin hypercube with an optional design criterion.

    LHS variants and scrambled Sobol sampling are deterministic for a given
    seed. Design properties apply before the default anchor, quantization,
    bias and switch mapping, which can weaken their ideal space coverage.
    """
    if lhs_method not in LHS_METHODS:
        raise ValueError(f"Unknown LHS method {lhs_method!r}; choose from {LHS_METHODS}")
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if dim < 0:
        raise ValueError(f"dim must be non-negative; got {dim}")
    if dim == 0:
        return _np.zeros((n, 0), dtype=_np.float64)
    if lhs_method == "sobol":
        # Scrambled Sobol is a low-discrepancy digital net, not an LHS:
        # sample a power-of-two block and take its prefix so arbitrary
        # capture counts are supported without the unbalanced-n warning.
        from scipy.stats import qmc

        try:
            sampler = qmc.Sobol(
                d=dim, scramble=True, rng=_np.random.default_rng(seed)
            )
        except TypeError:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        return _np.asarray(
            sampler.random_base2(m=(n - 1).bit_length())[:n],
            dtype=_np.float64,
        )

    if lhs_method == "optimized" and n == 1:
        # A single point has no pairwise discrepancy to optimize.
        return _standard_lhs(n, dim, seed)

    if lhs_method == "optimized":
        # SciPy's random-coordinate-descent optimization targets centered
        # discrepancy, improving coverage of low-dimensional projections.
        # Unlike maximin this optimizes more than the closest pair of points.
        from scipy.stats import qmc

        try:
            sampler = qmc.LatinHypercube(
                d=dim, optimization="random-cd", rng=_np.random.default_rng(seed)
            )
        except TypeError:
            sampler = qmc.LatinHypercube(
                d=dim, optimization="random-cd", seed=seed
            )
        return _np.asarray(sampler.random(n=n), dtype=_np.float64)

    if lhs_method == "standard" or n == 1:
        return _standard_lhs(n, dim, seed)

    if lhs_method == "orthogonal":
        # A strength-2 orthogonal-array LHS has n=p**2 for prime p,
        # with dim <= p+1. Do not silently fall back to ordinary LHS.
        p = _math.isqrt(n)
        prime = p >= 2 and all(p % i for i in range(2, _math.isqrt(p) + 1))
        if p * p != n or not prime or dim > p + 1:
            raise ValueError(
                f"Orthogonal LHS with {dim} dimensions requires n=p² captures "
                f"per split and Complex Switch combination, where p is prime "
                f"and dimensions <= p+1 (received n={n}). "
                "Adjust training and validation counts or choose Near-orthogonal."
            )
        try:
            from scipy.stats import qmc
        except ImportError as error:
            raise ValueError("Orthogonal LHS requires scipy") from error
        try:
            sampler = qmc.LatinHypercube(
                d=dim, strength=2, rng=_np.random.default_rng(seed)
            )
        except TypeError:
            sampler = qmc.LatinHypercube(d=dim, strength=2, seed=seed)
        return _np.asarray(sampler.random(n=n), dtype=_np.float64)

    if dim == 1:
        return _standard_lhs(n, dim, seed)

    # Bounded best-of candidate search. Both criteria keep the Latin strata;
    # maximin increases the closest pair distance and near-orthogonal reduces
    # the sum of squared pairwise correlations between columns.
    from scipy.spatial import cKDTree

    rng = _np.random.default_rng(seed)
    trials = 24 if n <= 2048 else 8 if n <= 20000 else 3
    best, best_score = None, None
    for attempt in range(trials):
        # Include the Standard design among the candidates: an optimized method
        # must not score worse than Standard under the exact same seed.
        candidate_seed = seed if attempt == 0 else int(rng.integers(0, 2**32))
        candidate = _standard_lhs(n, dim, candidate_seed)
        if lhs_method == "maximin":
            nearest = cKDTree(candidate).query(candidate, k=2, workers=1)[0][:, 1]
            score = float(nearest.min())
            improved = best_score is None or score > best_score
        else:
            correlations = _np.corrcoef(candidate, rowvar=False)
            score = float(_np.sum(_np.triu(correlations**2, k=1)))
            improved = best_score is None or score < best_score
        if improved:
            best, best_score = candidate, score
    return best


def _scale_continuous_samples(
    unit_samples: _np.ndarray,
    continuous_specs: _Sequence[_ParamSpec],
) -> _np.ndarray:
    if unit_samples.shape[1] != len(continuous_specs):
        raise ValueError(
            f"Expected {len(continuous_specs)} continuous columns; got {unit_samples.shape[1]}"
        )
    if len(continuous_specs) == 0:
        return unit_samples

    mins = _np.asarray([spec.min for spec in continuous_specs], dtype=_np.float64)
    widths = _np.asarray(
        [spec.max - spec.min for spec in continuous_specs], dtype=_np.float64
    )
    biased = unit_samples.copy()
    for column, spec in enumerate(continuous_specs):
        biased[:, column] = _apply_bias(biased[:, column], spec)
    return mins + biased * widths


def _stratified_switch_assignments(
    specs: _Sequence[_ParamSpec],
    n: int,
    *,
    seed: int,
) -> _np.ndarray:
    combos = _switch_combinations(specs)
    if len(combos) == 1 and len(combos[0]) == 0:
        return _np.zeros((n, 0), dtype=_np.int64)

    rng = _np.random.default_rng(seed)
    combo_array = _np.asarray(combos, dtype=_np.int64)
    num_combos = combo_array.shape[0]
    assignments = _np.empty((n, combo_array.shape[1]), dtype=_np.int64)
    # Deal out shuffled full cycles of every switch combination. Balance is exact only
    # when ``n`` is a multiple of ``num_combos``; otherwise the final partial cycle skews
    # the counts by up to one per combination.
    for start in range(0, n, num_combos):
        stop = min(start + num_combos, n)
        cycle = combo_array[rng.permutation(num_combos)]
        assignments[start:stop] = cycle[: stop - start]
    return assignments


def _assemble_raw_settings(
    continuous_samples: _np.ndarray,
    switch_assignments: _np.ndarray,
    specs: _Sequence[_ParamSpec],
) -> list[_np.ndarray]:
    if continuous_samples.shape[0] != switch_assignments.shape[0]:
        raise ValueError("Continuous and switch samples must have the same row count")

    raw_settings: list[_np.ndarray] = []
    for row_index in range(continuous_samples.shape[0]):
        raw = _np.empty(len(specs), dtype=_np.float64)
        continuous_col = 0
        switch_col = 0
        for spec_index, spec in enumerate(specs):
            if spec.type == "switch":
                raw[spec_index] = float(switch_assignments[row_index, switch_col])
                switch_col += 1
            else:
                raw[spec_index] = float(continuous_samples[row_index, continuous_col])
                continuous_col += 1
        raw_settings.append(raw)
    return raw_settings


def _balanced_sobol_switch_assignments(specs, n, *, seed):
    """Balanced one-way margins and greedy pairwise coverage for 5+ states.

    A covering-array-like arrangement avoids enumerating a potentially huge
    Cartesian product. Every individual Simple Switch position occurs either
    floor(n/k) or ceil(n/k) times; repeated switch pairs are discouraged.
    """
    rng = _np.random.default_rng(seed)
    categories = _np.empty((n, len(specs)), dtype=_np.int64)
    for column, spec in enumerate(specs):
        k = spec.num_inputs
        quota = _np.full(k, n // k, dtype=_np.int64)
        quota[rng.permutation(k)[:n % k]] += 1
        pair_counts = [
            _np.zeros((prior.num_inputs, k), dtype=_np.int64)
            for prior in specs[:column]
        ]
        for row in range(n):
            available = _np.flatnonzero(quota)
            if column == 0:
                # Keep the first axis stratified rather than in shuffled clumps.
                choice = int(available[rng.integers(len(available))])
            else:
                cost = _np.asarray([
                    sum(
                        pair_counts[prior_column][categories[row, prior_column], candidate]
                        for prior_column in range(column)
                    )
                    for candidate in available
                ])
                tied = available[cost == cost.min()]
                choice = int(tied[rng.integers(len(tied))])
            categories[row, column] = choice
            quota[choice] -= 1
            for prior_column in range(column):
                pair_counts[prior_column][categories[row, prior_column], choice] += 1
    return categories


def _simple_switch_sobol_settings(specs, n, *, seed):
    """Use one Sobol budget for all Simple Switch states.

    For at most four joint states, split a longer continuous-only Sobol
    sequence into equal power-of-two blocks. Each state takes the number of
    points afforded by the original total capture budget, not a new budget.
    Larger state spaces use pairwise-balanced categories and one Sobol stream.
    """
    continuous = tuple(s for s in specs if s.type == "continuous")
    switches = tuple(s for s in specs if s.type == "switch")
    states = _math.prod(s.num_inputs for s in switches)
    if states <= 4:
        per_state = (n + states - 1) // states
        block_size = 1 << (per_state - 1).bit_length()
        total = states * block_size
        units = latin_hypercube_unit(total, len(continuous), seed=seed,
                                    lhs_method="sobol")
        combos = list(_itertools.product(*(range(s.num_inputs) for s in switches)))
        sampled = []
        categories = []
        for index, combo in enumerate(combos):
            quota = n // states + (index < n % states)
            sampled.extend(units[index * block_size:index * block_size + quota])
            categories.extend([combo] * quota)
        samples = _np.asarray(sampled, dtype=_np.float64).reshape(n, len(continuous))
        assignments = _np.asarray(categories, dtype=_np.int64).reshape(n, len(switches))
    else:
        samples = latin_hypercube_unit(n, len(continuous), seed=seed,
                                      lhs_method="sobol")
        assignments = _balanced_sobol_switch_assignments(
            switches, n, seed=(seed + 104729) % (2**32)
        )
    return _assemble_raw_settings(
        _scale_continuous_samples(samples, continuous), assignments, specs
    )


def sample_raw_settings(
    specs: _Sequence[_ParamSpec],
    n: int,
    *,
    seed: int = 0,
    full_grid: bool = False,
    lhs_method: str = "standard",
) -> list[_np.ndarray]:
    specs = tuple(specs)
    if any(spec.type == "switch" and spec.simple_switch for spec in specs):
        if full_grid:
            raise ValueError(
                "Simple switches use LHS dimensions, not switch-grid replication"
            )
        if lhs_method == "sobol":
            return _simple_switch_sobol_settings(specs, n, seed=seed)
        unit = latin_hypercube_unit(n, len(specs), seed=seed, lhs_method=lhs_method)
        raw = _np.empty_like(unit)
        for i, spec in enumerate(specs):
            raw[:, i] = (
                _np.minimum(
                    spec.num_inputs - 1, _np.floor(unit[:, i] * spec.num_inputs)
                )
                if spec.type == "switch"
                else spec.min + _apply_bias(unit[:, i], spec) * (spec.max - spec.min)
            )
        return list(raw)
    continuous_specs = tuple(spec for spec in specs if spec.type == "continuous")
    rng = _np.random.default_rng(seed)
    continuous_seed = int(rng.integers(0, 2**32))
    switch_seed = int(rng.integers(0, 2**32))
    continuous_samples = _scale_continuous_samples(
        latin_hypercube_unit(
            n, len(continuous_specs), seed=continuous_seed, lhs_method=lhs_method
        ),
        continuous_specs,
    )

    if not full_grid:
        switch_assignments = _stratified_switch_assignments(specs, n, seed=switch_seed)
        return _assemble_raw_settings(continuous_samples, switch_assignments, specs)

    combos = _switch_combinations(specs)
    tiled_continuous = _np.repeat(continuous_samples, len(combos), axis=0)
    repeated_switches = _np.asarray(combos, dtype=_np.int64)
    switch_assignments = _np.tile(repeated_switches, (n, 1))
    return _assemble_raw_settings(tiled_continuous, switch_assignments, specs)


@_dataclass(frozen=True)
class PlannedCapture:
    """
    One reamp to be recorded: a knob setting plus where its output WAV will live
    (relative to the project folder).
    """

    index: int
    split: Literal["train", "validation"]
    params: dict[str, _Any]
    y_path: str


def _planned_from_params(
    prefix: str,
    split: Literal["train", "validation"],
    specs: _Sequence[_ParamSpec],
    param_dicts: list[dict[str, _Any]],
    *,
    index_offset: int = 0,
    filename_start: int = 0,
) -> list[PlannedCapture]:
    """
    Turn decoded param dicts into planned captures. ``prefix`` is the filename stem prefix
    (``lhs``/``val_lhs``/``corner``); the filename counter starts at ``filename_start`` and
    the model ``index`` starts at ``index_offset``. These are decoupled so appended corners
    can continue the train index space (unique ``index``) while numbering their own files
    from ``corner_000`` (see :func:`plan_corner_captures`).
    """
    abbreviations = _abbreviate_param_names([spec.name for spec in specs])
    planned = []
    for offset, params in enumerate(param_dicts):
        filename = _make_capture_y_path(
            f"{prefix}_{filename_start + offset:03d}_", params, abbreviations
        )
        planned.append(
            PlannedCapture(
                index=index_offset + offset,
                split=split,
                params=params,
                y_path=f"{CAPTURES_DIRNAME}/{filename}",
            )
        )
    return planned


def _plan_captures_unadjusted(
    knobs: _Sequence[_ControlSpec],
    *,
    n_train: int,
    n_validation: int,
    seed: int = 0,
    lhs_method: str = "standard",
    validation_method: str = "independent",
    rules=(),
    include_default_as_additional_capture: bool | None | str = None,
    external_exclude: _Optional[set[tuple[_Any, ...]]] = None,
) -> tuple[list[PlannedCapture], list[PlannedCapture]]:
    """
    Generate the full capture plan: ``n_train`` Latin-Hypercube training settings and
    ``n_validation`` held-out validation settings (a separate LHS stream derived from
    the same seed), each snapped to its knob's capture grid. Every planned setting is
    distinct and validation is held out from train (see :func:`plan_unique_splits`).

    Each split is emitted in ascending knob order (see :func:`settings_sort_key`) so the
    user works through one section at a time with the knobs moving as little as possible
    between consecutive captures. Sorting within a split does not touch which settings are
    drawn, only the order they are recorded in.

    The corner settings (see :func:`corner_settings`) are held out of the validation draw
    even when the corners are not part of this plan. A validation point sitting on a corner
    would be skipped as a duplicate when the corners are added later, leaving that knob
    extreme uncaptured -- the model would never train on it and would then be validated
    there.
    """
    knobs = _validate_knobs(knobs)
    policy = (
        "inside" if include_default_as_additional_capture is None
        else "extra" if include_default_as_additional_capture is True
        else "exclude" if include_default_as_additional_capture is False
        else "sample" if include_default_as_additional_capture == "sample"
        else None
    )
    if policy is None:
        raise ValueError("Unknown default capture option")
    if rules:
        return _plan_with_rules(knobs, n_train, n_validation, seed, lhs_method,
                                validation_method, rules, policy)
    if n_train <= 0:
        raise ValueError(f"n_train must be positive; got {n_train}")
    if n_validation < 0:
        raise ValueError(f"n_validation must be non-negative; got {n_validation}")
    specs = tuple(knob.to_param_spec() for knob in knobs)
    clean = [c for c in knobs if getattr(c, "clean_volume", False)]
    if clean:
        inner = tuple(c for c in knobs if c is not clean[0])
        if not inner:
            raise ValueError(
                "Parametric capture needs another tone/drive/switch control besides clean volume"
            )
        t, v = _plan_captures_unadjusted(
            inner, n_train=n_train, n_validation=n_validation, seed=seed,
            lhs_method=lhs_method, validation_method=validation_method,
            include_default_as_additional_capture=include_default_as_additional_capture,
            external_exclude=external_exclude,
        )
        fixed = {clean[0].name: clean[0].max}
        merged = lambda entries: [
            {s.name: ({**e.params, **fixed})[s.name] for s in specs} for e in entries
        ]
        return _planned_from_params(
            "lhs", "train", specs, merged(t)
        ), _planned_from_params("val_lhs", "validation", specs, merged(v))

    for spec in specs if policy in ("inside", "extra") else ():
        if ((spec.avoid_below is not None and round(spec.default, 6) < spec.avoid_below)
            or (spec.avoid_above is not None and round(spec.default, 6) > spec.avoid_above)):
            raise ValueError(f"Default for {spec.name} is excluded by its capture thresholds")

    if policy in ("inside", "extra") and any(
        spec.type != "switch" and spec.avoid_zero and spec.default == spec.min
        for spec in specs
    ):
        raise ValueError(
            "Choose a default above minimum when Mute at minimum or legacy Avoid min is enabled: every new plan includes the default setting"
        )

    if policy in ("inside", "extra") and any(
        spec.type != "switch" and spec.avoid_max and spec.default == spec.max
        for spec in specs
    ):
        raise ValueError(
            "Choose a default below maximum when Mute at maximum or legacy Avoid max is enabled"
        )
    if any(spec.type == "switch" and spec.simple_switch for spec in specs):
        return _plan_simple_switches(
            knobs, specs, n_train, n_validation, seed, lhs_method,
            validation_method, policy, external_exclude=external_exclude,
        )
    if any(spec.type == "switch" for spec in specs):
        switches = [spec for spec in specs if spec.type == "switch"]
        continuous = [
            control for control in knobs if control.to_param_spec().type != "switch"
        ]
        count = _math.prod(spec.num_inputs for spec in switches)
        if n_train < count or (n_validation and n_validation < count):
            raise ValueError(
                f"Cover all {count} switch combinations: training and each enabled "
                "validation split need at least one capture per combination."
            )
        if not continuous and (n_train != count or n_validation):
            raise ValueError(
                f"With switches only, capture each of the {count} combinations once "
                "and set validation to 0: no unseen settings remain for validation."
            )
        training, validation = [], []
        used_settings = set(external_exclude or ())
        complementary = lhs_method == "sobol" and policy != "inside"
        default_combo = tuple(spec.enum_names[int(spec.default)] for spec in switches)
        for i, combo in enumerate(
            _itertools.product(*(spec.enum_names for spec in switches))
        ):
            fixed = dict(zip((spec.name for spec in switches), combo))
            if continuous:
                inner_default = (
                    None if policy == "inside" else
                    False if combo == default_combo else "sample"
                )
                arguments = dict(
                    n_train=n_train // count + (i < n_train % count),
                    n_validation=n_validation // count + (i < n_validation % count),
                    seed=(seed + i * 104729) % (2**32),
                    lhs_method=lhs_method,
                    validation_method=validation_method,
                    include_default_as_additional_capture=inner_default,
                )
                try:
                    try:
                        train_part, val_part = _plan_captures_unadjusted(
                            continuous, **arguments,
                            external_exclude=used_settings if complementary else None,
                        )
                    except ValueError:
                        if not complementary:
                            raise
                        # A coarse grid may not have enough distinct knob values
                        # for every switch state. Reuse across states only then.
                        train_part, val_part = _plan_captures_unadjusted(
                            continuous, **arguments
                        )
                except ValueError as exc:
                    raise ValueError(f"Switch combination {fixed}: {exc}") from exc
                if complementary:
                    inner_specs = tuple(c.to_param_spec() for c in continuous)
                    used_settings.update(
                        _settings_key(capture.params, inner_specs)
                        for capture in train_part + val_part
                    )
                training.extend({**capture.params, **fixed} for capture in train_part)
                validation.extend({**capture.params, **fixed} for capture in val_part)
            else:
                training.append(fixed)
        if policy == "extra" and continuous:
            training.append({
                spec.name: (
                    spec.enum_names[int(spec.default)]
                    if spec.type == "switch" else spec.default
                )
                for spec in specs
            })
        # Restore control order for display and exported metadata.
        training = [
            {spec.name: params[spec.name] for spec in specs} for params in training
        ]
        validation = [
            {spec.name: params[spec.name] for spec in specs} for params in validation
        ]
        return (
            _planned_from_params("lhs", "train", specs, training),
            _planned_from_params("val_lhs", "validation", specs, validation),
        )

    corner_keys = {
        _settings_key(params, specs)
        for params in corner_settings(
            specs,
            gain_index=_gain_knob_index(knobs),
            default_step=_DEFAULT_KNOB_STEP,
        ) if not _muted_corner_replaced_by_anchor(params, knobs)
    }
    train_params, validation_params = plan_unique_splits(
        specs,
        n_train=n_train,
        n_validation=n_validation,
        seed=seed,
        default_step=_DEFAULT_KNOB_STEP,
        lhs_method=lhs_method,
        validation_method=validation_method,
        validation_exclude=corner_keys,
        anchor=(
            {spec.name: spec.default for spec in specs}
            if policy != "sample" else None
        ),
        anchor_policy=policy,
        external_exclude=external_exclude,
    )
    names = [spec.name for spec in specs]
    train = _planned_from_params(
        "lhs", "train", specs, sort_settings(train_params, names)
    )
    validation = _planned_from_params(
        "val_lhs", "validation", specs, sort_settings(validation_params, names)
    )
    return train, validation


def _corner_raw_vector(
    specs: _Sequence[_ParamSpec], highs: _Sequence[bool]
) -> _np.ndarray:
    """
    Raw (pre-quantization) param vector for one corner: each entry is the knob's max when
    ``highs[i]`` else its min. Switches use their last index for ``max`` and 0 for ``min``
    (the capture app only produces continuous knobs, but this keeps the vector decodable
    for any spec set).
    """
    raw = _np.empty(len(specs), dtype=_np.float64)
    for i, spec in enumerate(specs):
        if spec.type == "switch":
            raw[i] = float(spec.num_inputs - 1) if highs[i] else 0.0
        else:
            raw[i] = spec.max if highs[i] else spec.min
    return raw


# Below this many knobs the even-parity half fraction stops being a usable design. Its
# defining relation is the product of every factor, so K factors give resolution K: at K=3
# main effects alias two-factor interactions, and at K<=2 they alias each other. The full
# factorial is at most 8 vertices there, so it is cheap enough to take outright.
_MIN_HALF_FRACTION_KNOBS = 4



def _mute_adjacent_values(knob, default_step=_DEFAULT_KNOB_STEP):
    """The nearest *recordable* setting for each explicitly muted endpoint."""
    if not (getattr(knob, "mute_at_min", False) or getattr(knob, "mute_at_max", False)):
        return ()
    values = _capture_grid_values(knob.to_param_spec(), default_step)
    return tuple(
        (edge, values[0 if edge == "min" else -1])
        for edge in ("min", "max")
        if getattr(knob, f"mute_at_{edge}", False)
    )


def _muted_corner_replaced_by_anchor(params, knobs):
    """A shifted corner adjacent to a Mute Anchor is redundant as a WAV."""
    return any(
        params[knob.name] == value
        for knob in knobs
        for _, value in _mute_adjacent_values(knob)
    )


def mute_corner_settings(knobs, *, rules=()):
    """All nominal mute-endpoint corners, now trained virtually instead of recorded.

    Retain the exact gain/switch/fractional-factorial corner design. Only muted
    endpoints bypass capture avoidance; other knobs keep their original limits.
    """
    knobs = _validate_knobs(knobs)
    specs = tuple(knob.to_param_spec() for knob in knobs)
    unrestricted = tuple(
        _replace(
            spec,
            avoid_zero=False if getattr(knob, "mute_at_min", False) else spec.avoid_zero,
            avoid_max=False if getattr(knob, "mute_at_max", False) else spec.avoid_max,
            avoid_below=None if getattr(knob, "mute_at_min", False) else spec.avoid_below,
            avoid_above=None if getattr(knob, "mute_at_max", False) else spec.avoid_above,
        )
        if spec.type == "continuous" else spec
        for knob, spec in zip(knobs, specs)
    )
    return [
        params for params in corner_settings(
            unrestricted, gain_index=_gain_knob_index(knobs)
        )
        if allows(rules, params) and any(
            (getattr(knob, "mute_at_min", False) and params[knob.name] == knob.min)
            or (getattr(knob, "mute_at_max", False) and params[knob.name] == knob.max)
            for knob in knobs
        )
    ]


def plan_captures(
    knobs: _Sequence[_ControlSpec],
    *,
    n_train: int,
    n_validation: int,
    seed: int = 0,
    lhs_method: str = "standard",
    validation_method: str = "independent",
    rules=(),
    include_default_as_additional_capture: bool | None | str = None,
) -> tuple[list[PlannedCapture], list[PlannedCapture]]:
    """Keep Sobol/LHS intact and add one real capture per uncovered mute transition."""
    knobs = _validate_knobs(knobs)
    train, validation = _plan_captures_unadjusted(
        knobs, n_train=n_train, n_validation=n_validation, seed=seed,
        lhs_method=lhs_method, validation_method=validation_method, rules=rules,
        include_default_as_additional_capture=include_default_as_additional_capture,
    )
    if not any(_mute_adjacent_values(knob) for knob in knobs):
        return train, validation
    specs = tuple(knob.to_param_spec() for knob in knobs)
    names = [spec.name for spec in specs]
    occupied = {_settings_key(entry.params, specs) for entry in train + validation}
    defaults = {
        spec.name: (
            spec.enum_names[int(spec.default)] if spec.type == "switch"
            else spec.max if spec.clean_volume else spec.default
        )
        for spec in specs
    }
    extras = []
    for knob in knobs:
        for edge, first_step in _mute_adjacent_values(knob):
            if any(entry.params[knob.name] == first_step for entry in train) or any(
                params[knob.name] == first_step for params in extras
            ):
                continue
            sources = [defaults] + [entry.params for entry in train + validation]
            if not any(getattr(c, "clean_volume", False) for c in knobs):
                for raw in sample_raw_settings(
                    specs, min(256, max(64, n_train + n_validation)),
                    seed=(seed + 131071) % (2**32), lhs_method="standard",
                ):
                    sources.append(_decode_named_params(
                        _quantize_to_capture_grid(
                            raw, specs, default_step=_DEFAULT_KNOB_STEP
                        ), specs
                    ))
            for source in sources:
                candidate = {**source, knob.name: first_step}
                candidate = {name: candidate[name] for name in names}
                key = _settings_key(candidate, specs)
                if key in occupied or not allows(rules, candidate):
                    continue
                # Defaults may be outside an Avoid range when the default
                # capture itself is disabled. Never import those excluded
                # values into an additional mute-transition WAV.
                if any(
                    (spec.avoid_zero and candidate[spec.name] == spec.min)
                    or (spec.avoid_max and candidate[spec.name] == spec.max)
                    or (spec.avoid_below is not None
                        and candidate[spec.name] < spec.avoid_below)
                    or (spec.avoid_above is not None
                        and candidate[spec.name] > spec.avoid_above)
                    for spec in specs if spec.type == "continuous"
                ):
                    continue
                if any(
                    (getattr(c, "mute_at_min", False) and candidate[c.name] == c.min)
                    or (getattr(c, "mute_at_max", False) and candidate[c.name] == c.max)
                    for c in knobs if hasattr(c, "mute_at_min")
                ):
                    continue
                occupied.add(key)
                extras.append(candidate)
                break
            else:
                raise ValueError(
                    f"Cannot reserve a first unmuted step at {knob.name}={first_step:g} "
                    f"for Mute at {edge}: all usable combinations are occupied "
                    "or excluded by the capture rules."
                )
    if extras:
        train = train + _planned_from_params(
            "mute_edge", "train", specs, extras,
            index_offset=max(entry.index for entry in train) + 1,
        )
    return train, validation


def _corner_vertices(n: int) -> list[tuple[bool, ...]]:
    """
    The min/max patterns (True = knob at max) to place corners on, over ``n`` knobs.

    For four or more knobs this is the even-parity half fraction -- every pattern with an
    even number of maxes, 2**(n-1) of them. That is the standard 2**(n-1) fractional
    factorial, and it keeps all n knobs mutually orthogonal, so the corner set can tell
    each knob's extreme apart from every other knob's. This is a set, not an order:
    :func:`plan_corner_captures` decides the order the corners are captured in.
    """
    if n < _MIN_HALF_FRACTION_KNOBS:
        return list(_itertools.product((False, True), repeat=n))
    return [
        combination
        for combination in _itertools.product((False, True), repeat=n)
        if sum(combination) % 2 == 0
    ]


def _corner_high_flag_sets(
    n: int, gain_index: int | _Sequence[int] | None
) -> list[list[bool]]:
    """
    The per-corner max/min pattern (True = knob at max) before dedup: the vertex set from
    :func:`_corner_vertices`, crossed with the gain knob's min and max when one is marked
    so every tone-stack extreme is seen at both ends of the drive range. Duplicates that
    collapse for small knob counts are removed later by :func:`corner_settings`.
    """
    gains = (
        ()
        if gain_index is None
        else ((gain_index,) if isinstance(gain_index, int) else tuple(gain_index))
    )
    if not gains:
        return [list(vertex) for vertex in _corner_vertices(n)]
    others = [i for i in range(n) if i not in gains]
    flag_sets = []
    for vertex in _corner_vertices(len(others)):
        for gain_values in _itertools.product((False, True), repeat=len(gains)):
            flags = [False] * n
            for index, high in zip(gains, gain_values):
                flags[index] = high
            for position, index in enumerate(others):
                flags[index] = vertex[position]
            flag_sets.append(flags)
    return flag_sets


def corner_settings(
    specs: _Sequence[_ParamSpec],
    *,
    gain_index: int | _Sequence[int] | None = None,
    default_step: float = _DEFAULT_KNOB_STEP,
) -> list[dict[str, _Any]]:
    """
    Decoded param dicts for the initial "corner" captures: the knob-range extremes that
    bound the amp's behavior (see :func:`_corner_vertices` for the pattern, and
    :func:`_corner_high_flag_sets` for how a marked gain/drive knob is crossed with it).
    Values are quantized to the capture grid (honoring each knob's ``step`` and
    ``avoid_zero`` (including Mute at minimum) and ``avoid_max`` (including Mute
    at maximum), so a muted endpoint moves to its nearest usable position), then
    deduplicated so the small knob counts that collapse (a single knob yields only
    min/max) don't repeat a setting.
    """
    specs = tuple(specs)
    if any(s.clean_volume for s in specs):
        active = tuple(s for s in specs if not s.clean_volume)
        gains = (
            ()
            if gain_index is None
            else ((gain_index,) if isinstance(gain_index, int) else tuple(gain_index))
        )
        inner_gains = tuple(sum(not s.clean_volume for s in specs[:i]) for i in gains)
        corners = corner_settings(
            active, gain_index=inner_gains, default_step=default_step
        )
        return [
            {s.name: (s.max if s.clean_volume else row[s.name]) for s in specs}
            for row in corners
        ]
    if any(spec.type == "switch" and spec.simple_switch for spec in specs):
        continuous = tuple(spec for spec in specs if spec.type != "switch")
        complex_switches = tuple(
            spec for spec in specs if spec.type == "switch" and not spec.simple_switch
        )
        simple = tuple(
            spec for spec in specs if spec.type == "switch" and spec.simple_switch
        )
        gains = (
            ()
            if gain_index is None
            else ((gain_index,) if isinstance(gain_index, int) else tuple(gain_index))
        )
        inner_gains = tuple(sum(s.type != "switch" for s in specs[:i]) for i in gains)
        corners = (
            corner_settings(
                continuous, gain_index=inner_gains, default_step=default_step
            )
            if continuous
            else [{}]
        )
        count = max(len(corners), max(s.num_inputs for s in simple))
        categories = sample_raw_settings(simple, count, seed=0)
        settings = []
        for combo in _itertools.product(*(s.enum_names for s in complex_switches)):
            fixed = dict(zip((s.name for s in complex_switches), combo))
            for i in range(count):
                values = {
                    **corners[i % len(corners)],
                    **fixed,
                    **_decode_named_params(categories[i], simple),
                }
                settings.append({s.name: values[s.name] for s in specs})
        return settings
    if any(spec.type == "switch" for spec in specs):
        continuous = tuple(spec for spec in specs if spec.type != "switch")
        switches = tuple(spec for spec in specs if spec.type == "switch")
        gains = (
            ()
            if gain_index is None
            else ((gain_index,) if isinstance(gain_index, int) else tuple(gain_index))
        )
        continuous_gain = tuple(
            sum(spec.type != "switch" for spec in specs[:index]) for index in gains
        )
        corners = corner_settings(
            continuous, gain_index=continuous_gain, default_step=default_step
        )
        settings = []
        for combo in _itertools.product(*(spec.enum_names for spec in switches)):
            fixed = dict(zip((spec.name for spec in switches), combo))
            for corner in corners:
                combined = {**corner, **fixed}
                settings.append({spec.name: combined[spec.name] for spec in specs})
        return settings
    seen: set[tuple[_Any, ...]] = set()
    settings: list[dict[str, _Any]] = []
    for highs in _corner_high_flag_sets(len(specs), gain_index):
        raw = _corner_raw_vector(specs, highs)
        quantized = _quantize_to_capture_grid(raw, specs, default_step=default_step)
        params = _decode_named_params(quantized, specs)
        key = _settings_key(params, specs)
        if key in seen:
            continue
        seen.add(key)
        settings.append(params)
    return settings


def corner_capture_count(
    knobs: _Sequence[_ControlSpec], *, default_step: float = _DEFAULT_KNOB_STEP, rules=()
) -> int:
    """How many distinct corner captures ``knobs`` will produce."""
    knobs = _validate_knobs(knobs)
    specs = tuple(knob.to_param_spec() for knob in knobs)
    validate_plan_rules(specs, rules)
    return sum(
        allows(rules, params) and not _muted_corner_replaced_by_anchor(params, knobs)
        for params in corner_settings(
            specs, gain_index=_gain_knob_index(knobs), default_step=default_step
        )
    )


def plan_corner_captures(
    knobs: _Sequence[_ControlSpec],
    *,
    exclude: _Optional[set[tuple[_Any, ...]]] = None,
    index_offset: int = 0,
    filename_start: int = 0,
    rules=(),
) -> tuple[list[PlannedCapture], int]:
    """
    Plan the corner captures for ``knobs`` as pending ``train`` captures, skipping any whose
    setting already appears in ``exclude`` (the LHS points and any corners already planned).

    The corners are emitted as their own section, in ascending knob order (see
    :func:`settings_sort_key`), so they read as one sorted block wherever they sit in the
    plan. Which corners are captured does not depend on the order.

    Returns the planned corner captures and the number of distinct corners skipped because
    they duplicate an existing setting, so the caller can tell the user (they still get a
    useful boundary set; the overlap just means fewer *new* captures). ``index_offset`` and
    ``filename_start`` continue the train index space and the ``corner_NNN`` file numbering
    when appending to an existing plan.
    """
    knobs = _validate_knobs(knobs)
    specs = tuple(knob.to_param_spec() for knob in knobs)
    gain_index = _gain_knob_index(knobs)
    excluded = set() if exclude is None else set(exclude)
    validate_plan_rules(specs, rules)

    unique: list[dict[str, _Any]] = []
    skipped = 0
    for params in corner_settings(
        specs, gain_index=gain_index, default_step=_DEFAULT_KNOB_STEP
    ):
        if not allows(rules, params) or _muted_corner_replaced_by_anchor(params, knobs):
            continue
        key = _settings_key(params, specs)
        if key in excluded:
            skipped += 1
            continue
        excluded.add(key)
        unique.append(params)

    planned = _planned_from_params(
        "corner",
        "train",
        specs,
        sorted(
            unique,
            key=lambda params: (
                tuple(
                    spec.enum_names.index(params[spec.name])
                    for spec in specs
                    if spec.type == "switch"
                ),
                settings_sort_key(
                    params, [spec.name for spec in specs if spec.type != "switch"]
                ),
            ),
        ),
        index_offset=index_offset,
        filename_start=filename_start,
    )
    return planned, skipped


def _apply_bias(unit, spec):
    bias = getattr(spec, "sampling_bias", "linear")
    if bias == "lower":
        return _np.expm1(_np.log(9) * unit) / 8
    if bias == "upper":
        return 1 - _np.expm1(_np.log(9) * (1 - unit)) / 8
    return unit


def _inverse_bias(unit, spec):
    bias = getattr(spec, "sampling_bias", "linear")
    if bias == "lower":
        return _np.log1p(8 * unit) / _np.log(9)
    if bias == "upper":
        return 1 - _np.log1p(8 * (1 - unit)) / _np.log(9)
    return unit


def _plan_simple_switches(
    controls, specs, n_train, n_validation, seed, lhs_method,
    validation_method="independent", anchor_policy="inside",
    external_exclude=None,
):
    complex_specs = tuple(
        s for s in specs if s.type == "switch" and not s.simple_switch
    )
    names = [s.name for s in specs]
    if complex_specs:
        inner = tuple(
            c for c in controls if c.name not in {s.name for s in complex_specs}
        )
        combinations = list(_itertools.product(*(s.enum_names for s in complex_specs)))
        count = len(combinations)
        if n_train < count or (n_validation and n_validation < count):
            raise ValueError(
                f"Cover all {count} Complex Switch combinations in each split"
            )
        train, val = [], []
        used_settings = set(external_exclude or ())
        complementary = lhs_method == "sobol" and anchor_policy != "inside"
        default_combo = tuple(s.enum_names[int(s.default)] for s in complex_specs)
        for i, combo in enumerate(combinations):
            fixed = dict(zip((s.name for s in complex_specs), combo))
            inner_default = (
                None if anchor_policy == "inside" else
                False if combo == default_combo else "sample"
            )
            arguments = dict(
                n_train=n_train // count + (i < n_train % count),
                n_validation=n_validation // count + (i < n_validation % count),
                seed=(seed + i * 104729) % 2**32,
                lhs_method=lhs_method,
                validation_method=validation_method,
                include_default_as_additional_capture=inner_default,
            )
            try:
                t, v = _plan_captures_unadjusted(
                    inner, **arguments,
                    external_exclude=used_settings if complementary else None,
                )
            except ValueError:
                if not complementary:
                    raise
                t, v = _plan_captures_unadjusted(inner, **arguments)
            if complementary:
                inner_specs = tuple(c.to_param_spec() for c in inner)
                used_settings.update(
                    _settings_key(entry.params, inner_specs) for entry in t + v
                )
            train.extend(
                {name: ({**e.params, **fixed})[name] for name in names} for e in t
            )
            val.extend(
                {name: ({**e.params, **fixed})[name] for name in names} for e in v
            )
        if anchor_policy == "extra":
            train.append({
                s.name: s.enum_names[int(s.default)] if s.type == "switch" else s.default
                for s in specs
            })
    else:
        corners = {
            _settings_key(p, specs)
            for p in corner_settings(specs, gain_index=_gain_knob_index(controls))
            if not _muted_corner_replaced_by_anchor(p, controls)
        }
        anchor = {
            s.name: s.enum_names[int(s.default)] if s.type == "switch" else s.default
            for s in specs
        }
        train, val = plan_unique_splits(
            specs,
            n_train=n_train,
            n_validation=n_validation,
            seed=seed,
            default_step=_DEFAULT_KNOB_STEP,
            lhs_method=lhs_method,
            validation_method=validation_method,
            validation_exclude=corners,
            anchor=anchor if anchor_policy != "sample" else None,
            anchor_policy=anchor_policy,
            external_exclude=external_exclude,
        )
    return (
        _planned_from_params("lhs", "train", specs, sort_settings(train, names)),
        _planned_from_params("val_lhs", "validation", specs, sort_settings(val, names)),
    )


@_dataclass
class _RuleGroup:
    fixed: dict
    rules: tuple
    anchor: dict | None
    corners: set
    capacity: int
    validation_capacity: int


def _rule_groups(controls, rules, anchor_policy="inside"):
    specs = tuple(c.to_param_spec() for c in controls)
    validate_plan_rules(specs, rules)
    switches_only = all(s.type == "switch" for s in specs)
    fixed_specs = tuple(s for s in specs if s.clean_volume or (
        s.type == "switch" and (switches_only or not s.simple_switch)
    ))
    active = tuple(s for s in specs if s not in fixed_specs)
    defaults = {
        s.name: s.enum_names[int(s.default)] if s.type == "switch" else
        s.max if s.clean_volume else s.default for s in specs
    }
    if anchor_policy in ("inside", "extra"):
        if not allows(rules, defaults):
            raise ValueError("Plan rules exclude the default setting. Change the rule or the default.")
        for spec in active:
            if spec.type == "switch":
                continue
            value = round(spec.default, 6)
            if ((spec.avoid_zero and value == round(spec.min, 6))
                or (spec.avoid_max and value == round(spec.max, 6))
                or (spec.avoid_below is not None and value < spec.avoid_below)
                or (spec.avoid_above is not None and value > spec.avoid_above)):
                raise ValueError(f"Capture thresholds exclude the default for {spec.name}")
    fixed_axes = [s.enum_names if s.type == "switch" else (s.max,) for s in fixed_specs]
    allowed_corners = {}
    for params in corner_settings(specs, gain_index=_gain_knob_index(controls)):
        if allows(rules, params) and not _muted_corner_replaced_by_anchor(params, controls):
            group_key = _settings_key(params, fixed_specs)
            allowed_corners.setdefault(group_key, set()).add(_settings_key(params, active))
    groups = []
    for values in _itertools.product(*fixed_axes):
        fixed = dict(zip((s.name for s in fixed_specs), values))
        local_rules = bound_rules(rules, fixed)
        grid_capacity = _grid_capacity(active, _DEFAULT_KNOB_STEP, local_rules)
        anchor = {s.name: defaults[s.name] for s in active}
        is_default_group = all(defaults[s.name] == fixed[s.name] for s in fixed_specs)
        if (anchor_policy in ("extra", "exclude") and not is_default_group
                or anchor_policy == "sample"
                or not allows(local_rules, anchor)):
            anchor = None
        on_grid = (
            settings_on_capture_grid(anchor, active) if anchor is not None and active
            else False
        )
        if anchor_policy == "inside":
            # Historical projects embed one default in each Complex Switch group.
            capacity = grid_capacity + int(anchor is not None and not on_grid and bool(active))
        else:
            # New plans count only sampled settings. The optional extra default
            # consumes a physical grid combination, not an LHS/Sobol slot.
            capacity = grid_capacity - int(on_grid)
        if not capacity:
            continue
        corners = allowed_corners.get(values, set())
        reserved = set(corners)
        if anchor is not None and (on_grid or anchor_policy in ("inside", "extra")):
            reserved.add(_settings_key(anchor, active))
        validation_capacity = grid_capacity - sum(
            1 for key in reserved
            if settings_on_capture_grid(dict(zip((s.name for s in active), key)), active)
        ) if active else 0
        groups.append(_RuleGroup(fixed, local_rules, anchor, corners, capacity,
                                 max(0, validation_capacity)))
    if not groups:
        raise ValueError("Plan rules leave no valid capture settings.")
    return specs, active, groups


def _group_quota_limit(capacities):
    """Largest total that fits the ordered round-robin quotas."""
    count = len(capacities)
    return min(value * count + i for i, value in enumerate(capacities))


def rule_capture_limits(controls, rules, n_train=0, n_validation=0,
                        include_default_as_additional_capture=None):
    policy = (
        "inside" if include_default_as_additional_capture is None
        else "extra" if include_default_as_additional_capture is True
        else "exclude"
    )
    _, active, groups = _rule_groups(controls, rules, policy)
    count = len(groups)
    if not active:
        return count, count, count
    train_max = _group_quota_limit([
        group.capacity - (n_validation // count + (i < n_validation % count))
        for i, group in enumerate(groups)
    ])
    validation_max = _group_quota_limit([
        min(group.validation_capacity,
            group.capacity - (n_train // count + (i < n_train % count)))
        for i, group in enumerate(groups)
    ])
    return count, max(0, train_max), max(0, validation_max)


def _plan_with_rules(knobs, n_train, n_validation, seed, lhs_method, validation_method, rules, anchor_policy="inside"):
    specs, active, groups = _rule_groups(knobs, rules, anchor_policy)
    count = len(groups)
    if n_train < count or n_validation < 0 or (0 < n_validation < count):
        raise ValueError(f"Each enabled split needs at least {count} captures for the allowed switch states")
    if not active:
        if n_train != count or n_validation:
            raise ValueError(f"Capture all {count} allowed states once using each signal")
        return _planned_from_params("lhs", "train", specs, [g.fixed for g in groups]), []
    train, validation = [], []
    for i, group in enumerate(groups):
        try:
            t, v = plan_unique_splits(
                active, n_train=n_train // count + (i < n_train % count),
                n_validation=n_validation // count + (i < n_validation % count),
                seed=(seed + i * 104729) % (2**32), default_step=_DEFAULT_KNOB_STEP,
                lhs_method=lhs_method, validation_method=validation_method,
                anchor=group.anchor, validation_exclude=group.corners, rules=group.rules,
                anchor_policy=anchor_policy if group.anchor is not None else "sample",
            )
        except ValueError as exc:
            context = f" for {group.fixed}" if group.fixed else ""
            raise ValueError(f"Cannot generate a plan with these rules{context}: {exc}") from exc
        train.extend({**point, **group.fixed} for point in sort_settings(t, [s.name for s in active]))
        validation.extend({**point, **group.fixed} for point in sort_settings(v, [s.name for s in active]))
    return (
        _planned_from_params("lhs", "train", specs, train),
        _planned_from_params("val_lhs", "validation", specs, validation),
    )

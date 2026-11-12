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
    return sorted(values)


def _grid_capacity(specs: _Sequence[_ParamSpec], default_step: float) -> int:
    """Total number of distinct on-grid settings (switch combinations x continuous grid)."""
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
            if key in seen:
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
            row = int(_np.flatnonzero(strata == target)[0])
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
        if key in seen:
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
    available = _grid_capacity(specs, default_step) - grid_exclusions
    if available < n_validation:
        raise ValueError("Not enough non-training, non-corner settings for validation")
    pool_size = min(available, max(n_validation, min(4096, max(256, 20 * n_validation))))
    # The pool is another deterministic LHS draw. Even if the train uses a
    # strength-2 orthogonal design, the pool need not have that special size.
    candidates = sample_unique_settings(
        specs, pool_size, seed=seed, default_step=default_step,
        lhs_method="standard", exclude=excluded,
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
    reserved = (
        set()
        if validation_exclude is None or n_validation == 0
        else set(validation_exclude)
    )
    if anchor is not None:
        if n_train < 1:
            raise ValueError("The default setting requires a training capture")
        reserved.add(_settings_key(anchor, specs))

    # Every train and validation setting must be distinct, and validation must be held out
    # from train. That is only possible if the knob grid has at least that many distinct
    # settings; fail loudly up front rather than after sampling work.
    required = n_train + n_validation
    capacity = _grid_capacity(specs, default_step)
    if anchor is not None and not settings_on_capture_grid(anchor, specs, default_step):
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
            lhs_method=lhs_method, anchor=anchor,
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
            )
        return train, _farthest_validation(
            specs, train, reserved, n_validation, seed=validation_seed,
            default_step=default_step,
        )
    if not reserved:
        train = sample_unique_settings(
            specs, n_train, seed=seed, default_step=default_step, lhs_method=lhs_method
        )
        validation = sample_unique_settings(
            specs,
            n_validation,
            seed=validation_seed,
            default_step=default_step,
            lhs_method=lhs_method,
            exclude={_settings_key(params, specs) for params in train},
        )
        return train, validation

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
    )
    train = sample_unique_settings(
        specs,
        n_train,
        seed=seed,
        default_step=default_step,
        lhs_method=lhs_method,
        exclude={_settings_key(params, specs) for params in validation},
        anchor=anchor,
    )
    return train, validation


LHS_METHODS = ("standard", "maximin", "orthogonal", "near_orthogonal")


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

    Design properties apply before the default anchor, quantization, bias and
    switch mapping, which can weaken pairwise coverage/orthogonality.
    """
    if lhs_method not in LHS_METHODS:
        raise ValueError(f"Unknown LHS method {lhs_method!r}; choose from {LHS_METHODS}")
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if dim < 0:
        raise ValueError(f"dim must be non-negative; got {dim}")
    if dim == 0:
        return _np.zeros((n, 0), dtype=_np.float64)
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


def plan_captures(
    knobs: _Sequence[_ControlSpec],
    *,
    n_train: int,
    n_validation: int,
    seed: int = 0,
    lhs_method: str = "standard",
    validation_method: str = "independent",
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
        t, v = plan_captures(
            inner, n_train=n_train, n_validation=n_validation, seed=seed,
            lhs_method=lhs_method, validation_method=validation_method,
        )
        fixed = {clean[0].name: clean[0].max}
        merged = lambda entries: [
            {s.name: ({**e.params, **fixed})[s.name] for s in specs} for e in entries
        ]
        return _planned_from_params(
            "lhs", "train", specs, merged(t)
        ), _planned_from_params("val_lhs", "validation", specs, merged(v))

    if any(
        spec.type != "switch" and spec.avoid_zero and spec.default == spec.min
        for spec in specs
    ):
        raise ValueError(
            "Choose a default above minimum for controls with Avoid min enabled: every new plan includes the default setting"
        )

    if any(
        spec.type != "switch" and spec.avoid_max and spec.default == spec.max
        for spec in specs
    ):
        raise ValueError(
            "Choose a default below maximum for controls with Avoid max enabled"
        )
    if any(spec.type == "switch" and spec.simple_switch for spec in specs):
        return _plan_simple_switches(
            knobs, specs, n_train, n_validation, seed, lhs_method,
            validation_method,
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
        for i, combo in enumerate(
            _itertools.product(*(spec.enum_names for spec in switches))
        ):
            fixed = dict(zip((spec.name for spec in switches), combo))
            if continuous:
                try:
                    train_part, val_part = plan_captures(
                        continuous,
                        n_train=n_train // count + (i < n_train % count),
                        n_validation=n_validation // count + (i < n_validation % count),
                        seed=(seed + i * 104729) % (2**32),
                        lhs_method=lhs_method,
                        validation_method=validation_method,
                    )
                except ValueError as exc:
                    raise ValueError(f"Switch combination {fixed}: {exc}") from exc
                training.extend({**capture.params, **fixed} for capture in train_part)
                validation.extend({**capture.params, **fixed} for capture in val_part)
            else:
                training.append(fixed)
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
        )
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
        anchor={spec.name: spec.default for spec in specs},
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
    ``avoid_zero``, so an Avoid min knob's minimum corner moves to the next usable position), then
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
    knobs: _Sequence[_ControlSpec], *, default_step: float = _DEFAULT_KNOB_STEP
) -> int:
    """How many distinct corner captures ``knobs`` will produce."""
    knobs = _validate_knobs(knobs)
    specs = tuple(knob.to_param_spec() for knob in knobs)
    return len(
        corner_settings(
            specs, gain_index=_gain_knob_index(knobs), default_step=default_step
        )
    )


def plan_corner_captures(
    knobs: _Sequence[_ControlSpec],
    *,
    exclude: _Optional[set[tuple[_Any, ...]]] = None,
    index_offset: int = 0,
    filename_start: int = 0,
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

    unique: list[dict[str, _Any]] = []
    skipped = 0
    for params in corner_settings(
        specs, gain_index=gain_index, default_step=_DEFAULT_KNOB_STEP
    ):
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
    validation_method="independent",
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
        for i, combo in enumerate(combinations):
            fixed = dict(zip((s.name for s in complex_specs), combo))
            t, v = plan_captures(
                inner,
                n_train=n_train // count + (i < n_train % count),
                n_validation=n_validation // count + (i < n_validation % count),
                seed=(seed + i * 104729) % 2**32,
                lhs_method=lhs_method,
                validation_method=validation_method,
            )
            train.extend(
                {name: ({**e.params, **fixed})[name] for name in names} for e in t
            )
            val.extend(
                {name: ({**e.params, **fixed})[name] for name in names} for e in v
            )
    else:
        corners = {
            _settings_key(p, specs)
            for p in corner_settings(specs, gain_index=_gain_knob_index(controls))
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
            anchor=anchor,
        )
    return (
        _planned_from_params("lhs", "train", specs, sort_settings(train, names)),
        _planned_from_params("val_lhs", "validation", specs, sort_settings(val, names)),
    )

"""Plan a short physical route through captures without changing their identity.

Control changes matter first, knob travel second, switch travel third. Distances
are fractions of each control's range, so printed scales cannot bias the route.
Nearest-neighbour construction is refined with bounded open-path 2-opt. Keeping
the original route as a candidate guarantees the chosen route is no worse under
that objective; this is a heuristic, not a claim of a global TSP optimum.
"""

import numpy as np

from nam.capture.params import SwitchSpec

_SCALE = 10**9


def _point(params, controls):
    point = []
    for control in controls:
        value = params.get(control.name, control.default)
        if isinstance(control, SwitchSpec):
            position = (
                control.enum_names.index(value)
                if isinstance(value, str)
                else int(value)
            )
            fraction = position / (len(control.enum_names) - 1)
        else:
            fraction = (float(value) - control.min) / (control.max - control.min)
        # Sub-nanofraction differences cannot be dialled on a physical control.
        point.append(round(fraction * _SCALE))
    return tuple(point)


def _edge(a, b, knob_mask):
    changes = knob_travel = switch_travel = 0
    for x, y, knob in zip(a, b, knob_mask):
        distance = abs(x - y)
        changes += distance != 0
        if knob:
            knob_travel += distance
        else:
            switch_travel += distance
    return changes, knob_travel, switch_travel


def _route_cost(route, points, anchor, knob_mask):
    total = [0, 0, 0]
    previous = anchor
    for index in route:
        if previous is not None:
            cost = _edge(previous, points[index], knob_mask)
            total = [a + b for a, b in zip(total, cost)]
        previous = points[index]
    return tuple(total)


def _nearest(points, anchor, knob_mask):
    coordinates = np.asarray(points, dtype=np.int64)
    knob_mask = np.asarray(knob_mask, dtype=bool)
    remaining = np.arange(len(points))
    # Before the first recording the actual hardware state is unknown. Start at
    # an endpoint rather than favouring the software default in mid-rotation.
    current = np.asarray(anchor if anchor is not None else min(points), dtype=np.int64)
    route = []
    while remaining.size:
        distance = np.abs(coordinates[remaining] - current)
        costs = (
            np.count_nonzero(distance, axis=1),
            distance[:, knob_mask].sum(axis=1),
            distance[:, ~knob_mask].sum(axis=1),
        )
        candidates = np.arange(remaining.size)
        for cost in costs:
            candidates = candidates[cost[candidates] == cost[candidates].min()]
        chosen = int(candidates[0])  # Stable original-order tie break.
        index = int(remaining[chosen])
        route.append(index)
        current = coordinates[index]
        remaining = np.delete(remaining, chosen)
    return route


def _refine(route, points, anchor, knob_mask, budget=25000):
    """Reverse sections only when their boundary cost strictly improves the path."""
    route = list(route)
    for _ in range(3):
        improved = False
        for left in range(len(route) - 1):
            for right in range(left + 1, len(route)):
                budget -= 1
                if budget < 0:
                    return route
                a = anchor if left == 0 else points[route[left - 1]]
                b, c = points[route[left]], points[route[right]]
                old, new = (
                    (_edge(a, b, knob_mask), _edge(a, c, knob_mask))
                    if a is not None
                    else ((0, 0, 0), (0, 0, 0))
                )
                if right + 1 < len(route):
                    d = points[route[right + 1]]
                    old = tuple(x + y for x, y in zip(old, _edge(c, d, knob_mask)))
                    new = tuple(x + y for x, y in zip(new, _edge(b, d, knob_mask)))
                if new < old:
                    route[left : right + 1] = reversed(route[left : right + 1])
                    improved = True
        if not improved:
            break
    return route


def order_captures(entries, controls):
    """Completed captures first; pending captures start at the last recorded state.

    Identical settings stay together even across train/validation or AL rounds.
    Return the original entry objects in display order. Never mutate project data,
    filenames, dataset ordering, checkpoints, or the caller's list.
    """
    entries, controls = list(entries), tuple(controls)
    completed = sorted(
        (e for e in entries if e.status == "captured"),
        key=lambda e: e.captured_at or "",
    )
    pending = [e for e in entries if e.status != "captured"]
    if not pending or not controls:
        return completed + pending
    gain = [e for e in pending if e.split == "gain"]
    normal = [e for e in pending if e.split != "gain"]
    if gain and normal:
        return order_captures(completed + gain, controls) + order_captures(
            normal, controls
        )
    anchor = _point(completed[-1].params, controls) if completed else None
    knob_mask = tuple(not isinstance(c, SwitchSpec) for c in controls)
    groups = {}
    for entry in pending:
        groups.setdefault(_point(entry.params, controls), []).append(entry)
    points = list(groups)
    original = list(range(len(points)))
    nearest = _nearest(points, anchor, knob_mask)
    route = min(
        (original, nearest),
        key=lambda candidate: _route_cost(candidate, points, anchor, knob_mask),
    )
    route = _refine(route, points, anchor, knob_mask)
    return completed + [entry for index in route for entry in groups[points[index]]]

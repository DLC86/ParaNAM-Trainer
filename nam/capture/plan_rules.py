"""Exclusions evaluated against decoded capture settings."""

import math
import operator
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, model_validator


_OPERATORS = {"<": operator.lt, ">": operator.gt, "=": operator.eq}


class PlanRule(BaseModel):
    controls: tuple[str, ...]
    operator: Literal["<", ">", "="]
    value: float | str

    @model_validator(mode="after")
    def validate_condition(self):
        if not self.controls or len(set(self.controls)) != len(self.controls):
            raise ValueError("Select at least one control, without duplicates")
        if isinstance(self.value, str):
            if self.operator != "=" or not self.value:
                raise ValueError("Switch rules require = and a position name")
        elif not math.isfinite(self.value):
            raise ValueError("The rule threshold must be finite")
        return self

    def matches_value(self, value):
        if isinstance(value, str) != isinstance(self.value, str):
            return False
        if not isinstance(value, str):
            value = round(value, 6)
        return _OPERATORS[self.operator](value, self.value)

    def excludes(self, params):
        return all(self.matches_value(params[name]) for name in self.controls)


def validate_plan_rules(specs, rules):
    by_name = {spec.name: spec for spec in specs}
    for rule in rules:
        for name in rule.controls:
            if name not in by_name:
                raise ValueError(f"Plan rule refers to missing control {name!r}")
            spec = by_name[name]
            if spec.type == "switch":
                if rule.value not in spec.enum_names:
                    raise ValueError(f"Choose a switch position from {name!r}")
            elif isinstance(rule.value, str):
                raise ValueError(f"Choose a numeric threshold for {name!r}")


def allows(rules, params):
    return not any(rule.excludes(params) for rule in rules)


def bound_rules(rules, fixed):
    """Resolve the fixed controls of a switch combination or volume reference."""
    return tuple(
        rule.model_copy(update={"controls": tuple(n for n in rule.controls if n not in fixed)})
        for rule in rules
        if all(rule.matches_value(fixed[n]) for n in rule.controls if n in fixed)
    )


def allowed_grid_size(axes, rules):
    """Count valid points by grouping values with identical rule outcomes."""
    active = (1 << len(rules)) - 1
    if any(not rule.controls for rule in rules):
        return 0
    masks = []
    for name, values in axes:
        relevant = [(1 << i, rule) for i, rule in enumerate(rules) if name in rule.controls]
        counts = {}
        for value in values:
            failed = sum(bit for bit, rule in relevant if not rule.matches_value(value))
            counts[failed] = counts.get(failed, 0) + 1
        masks.append(counts)

    @lru_cache(maxsize=None)
    def count(axis, still_possible):
        if axis == len(masks):
            return int(not still_possible)
        return sum(
            multiplicity * count(axis + 1, still_possible & ~failed)
            for failed, multiplicity in masks[axis].items()
        )

    return count(0, active)

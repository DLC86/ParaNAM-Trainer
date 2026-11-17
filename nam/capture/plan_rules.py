"""Avoid and virtual mute rules evaluated against decoded capture settings.

Legacy rules use one operator/threshold for all selected controls. Modern rules
may override each control independently without changing serialized projects.
"""

import math
import operator
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, model_validator


_OPERATORS = {"≤": operator.le, "≥": operator.ge, "=": operator.eq}


class PlanRule(BaseModel):
    controls: tuple[str, ...]
    operator: Literal["≤", "≥", "="]
    value: float | str
    kind: Literal["avoid", "mute"] = "avoid"
    # Optional per-control overrides preserve backwards compatibility with
    # existing single-threshold rules and their JSON representation.
    operators: dict[str, Literal["≤", "≥", "="]] = Field(default_factory=dict)
    thresholds: dict[str, float | str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def migrate_inclusive_conditions(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            convert = {"<": "≤", ">": "≥"}
            data["operator"] = convert.get(data.get("operator"), data.get("operator"))
            data["operators"] = {
                name: convert.get(op, op)
                for name, op in data.get("operators", {}).items()
            }
        return data

    def condition_for(self, name):
        return (self.operators.get(name, self.operator),
                self.thresholds.get(name, self.value))

    @model_validator(mode="after")
    def validate_condition(self):
        if not self.controls or len(set(self.controls)) != len(self.controls):
            raise ValueError("Select at least one control, without duplicates")
        for mapping in (self.operators, self.thresholds):
            if set(mapping) - set(self.controls):
                raise ValueError("Rule conditions reference unselected controls")
        for name in self.controls:
            comparator, threshold = self.condition_for(name)
            if isinstance(threshold, str):
                if comparator != "=" or not threshold:
                    raise ValueError("Switch rules require = and a position name")
            elif not math.isfinite(threshold):
                raise ValueError("The rule threshold must be finite")
        return self

    def matches_value(self, value, name=None):
        comparator, threshold = self.condition_for(name) if name else (
            self.operator, self.value
        )
        if isinstance(value, str) != isinstance(threshold, str):
            return False
        if not isinstance(value, str):
            value = round(value, 6)
        return _OPERATORS[comparator](value, threshold)

    def excludes(self, params):
        return all(self.matches_value(params[name], name) for name in self.controls)


def validate_plan_rules(specs, rules):
    by_name = {spec.name: spec for spec in specs}
    for rule in rules:
        for name in rule.controls:
            if name not in by_name:
                raise ValueError(f"Plan rule refers to missing control {name!r}")
            spec = by_name[name]
            comparator, threshold = rule.condition_for(name)
            if spec.type == "switch":
                if comparator != "=" or threshold not in spec.enum_names:
                    raise ValueError(f"Choose a switch position from {name!r}")
            elif isinstance(threshold, str):
                raise ValueError(f"Choose a numeric threshold for {name!r}")
            elif rule.kind == "mute":
                # A Mute rule becomes a virtual training target. A condition
                # outside the neural parameter range cannot be sampled and
                # would otherwise fail only after launching training.
                threshold = float(threshold)
                if (
                    (comparator == "=" and not spec.min <= threshold <= spec.max)
                    or (comparator == "≤" and threshold < spec.min)
                    or (comparator == "≥" and threshold > spec.max)
                ):
                    raise ValueError(
                        f"Mute rule has no valid setting for {name!r}"
                    )


def allows(rules, params):
    return not any(rule.excludes(params) for rule in rules)


def bound_rules(rules, fixed):
    """Resolve fixed controls while retaining per-control thresholds/operators."""
    return tuple(
        rule.model_copy(update={
            "controls": tuple(n for n in rule.controls if n not in fixed),
            "operators": {n: op for n, op in rule.operators.items() if n not in fixed},
            "thresholds": {n: val for n, val in rule.thresholds.items() if n not in fixed},
        })
        for rule in rules
        if all(rule.matches_value(fixed[n], n) for n in rule.controls if n in fixed)
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
            failed = sum(bit for bit, rule in relevant
                         if not rule.matches_value(value, name))
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

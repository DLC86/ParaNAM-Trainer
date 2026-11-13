"""Bipolar knob labels are consistent in Plan, Capture and Compare."""

import pytest

from paranam.value_format import format_knob_value


@pytest.mark.parametrize(
    ("value", "minimum", "maximum", "expected"),
    [
        (5, -10, 10, "+5"),
        (10, -10, 10, "+10"),
        (0.5, -1, 1, "+0.5"),
        (0, -10, 10, "0"),
        (-5, -10, 10, "-5"),
        (5, 0, 10, "5"),
        (5, -10, 0, "5"),
        (5, 1, 10, "5"),
        (-5, -10, -1, "-5"),
    ],
)
def test_knob_value_sign_depends_only_on_bipolar_range(value, minimum, maximum, expected):
    assert format_knob_value(value, minimum, maximum) == expected


def test_formatted_text_is_preserved_when_sign_is_added():
    assert format_knob_value(1.25, -10, 10, "1,25") == "+1,25"
    assert format_knob_value(1.25, 0, 10, "1,25") == "1,25"
    assert format_knob_value(0, -10, 10, "0.000") == "0.000"


def test_compare_control_value_uses_same_sign_rule(qapp):
    from paranam.play_ui import ControlValue

    control = ControlValue()
    control.setRange(-10, 10)
    value = 1.25
    localized = control.locale().toString(value, "g", 12)
    assert control.textFromValue(value) == "+" + localized
    assert control.textFromValue(0) == control.locale().toString(0, "g", 12)
    assert control.textFromValue(-value) == control.locale().toString(-value, "g", 12)
    control.setRange(0, 10)
    assert control.textFromValue(value) == localized
    control.close()

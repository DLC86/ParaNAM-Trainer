"""Presentation-only formatting shared by Compare, Capture and Plan."""


def format_knob_value(value, minimum, maximum, text=None):
    """Prefix positive values only when the knob spans negative and positive.

    Caller-supplied text preserves existing numeric precision and locale.
    """
    if text is None:
        text = f"{value:g}"
    return f"+{text}" if minimum < 0 < maximum and value > 0 else text

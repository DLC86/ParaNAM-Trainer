"""Parse and display input impedance without rounding the saved ohm value."""

from decimal import Decimal
import math
import re


def parse_impedance(text):
    if not text.strip():
        return None
    match = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?|[.,]\d+)\s*([kKMm]?)\s*(?:[ΩΩ])?\s*", text)
    if not match:
        raise ValueError("Enter a positive impedance in Ω, kΩ or MΩ.")
    number, unit = match.groups()
    ohms = Decimal(number.replace(",", ".")).scaleb({"": 0, "k": 3, "m": 6}[unit.lower()])
    result = float(ohms)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Input impedance must be positive and finite.")
    return int(ohms) if ohms == ohms.to_integral_value() else result


def format_impedance(ohms):
    if ohms is None:
        return ""
    value = Decimal(str(ohms))
    power, unit = (6, "MΩ") if value >= 1000000 else (3, "kΩ") if value >= 1000 else (0, "Ω")
    number = format(value.scaleb(-power), "f")
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return f"{number.replace('.', ',')} {unit}"

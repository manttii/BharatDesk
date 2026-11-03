"""Money and weight primitives. Money = integer paise, weight = integer milligrams.

Floats are refused at every boundary here. Fractions (exact) are accepted where Hindi
fractional words need them ("dedh" = 3/2), and conversion fails loudly rather than
rounding if the result is not a whole paisa / milligram.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Annotated

from pydantic import Field

# Strict ints: pydantic refuses 1.0, "1" and True for these.
MoneyPaise = Annotated[int, Field(strict=True, ge=0)]
WeightMg = Annotated[int, Field(strict=True, ge=0)]

MG_PER_GRAM = 1_000
MG_PER_KG = 1_000_000
# ASSUMPTION (confirm with the customer): 1 tola = 11.664 g, the common trade value.
# The legal/historical figure 11.6638038 g is not a whole number of milligrams.
TOLA_MG = 11_664


def require_int(value: object, name: str = "value") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int (got {type(value).__name__}); floats are refused")
    return value


def _exact(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, int | Fraction):
        raise TypeError(f"{name} must be int or Fraction (got {type(value).__name__})")
    return Fraction(value)


def _whole(value: Fraction, what: str) -> int:
    if value.denominator != 1:
        raise ValueError(f"{value} is not a whole number of {what}")
    return int(value)


def rupees_to_paise(rupees: int | Fraction) -> int:
    return _whole(_exact(rupees, "rupees") * 100, "paise")


def grams_to_mg(grams: int | Fraction) -> int:
    return _whole(_exact(grams, "grams") * MG_PER_GRAM, "milligrams")


def kg_to_mg(kg: int | Fraction) -> int:
    return _whole(_exact(kg, "kg") * MG_PER_KG, "milligrams")


def tola_to_mg(tola: int | Fraction) -> int:
    return _whole(_exact(tola, "tola") * TOLA_MG, "milligrams")


def ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling division for non-negative numerator and positive denominator."""
    require_int(numerator, "numerator")
    require_int(denominator, "denominator")
    if numerator < 0 or denominator <= 0:
        raise ValueError("ceil_div needs numerator >= 0 and denominator > 0")
    return -(-numerator // denominator)


def format_inr(paise: int) -> str:
    """₹4,12,000 style (Indian digit grouping). Paise shown only when non-zero."""
    require_int(paise, "paise")
    rupees, rem = divmod(abs(paise), 100)
    digits = str(rupees)
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups: list[str] = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join([*groups, tail])
    text = f"₹{digits}" + (f".{rem:02d}" if rem else "")
    return f"-{text}" if paise < 0 else text


def format_weight_mg(mg: int) -> str:
    """'150 gram' / '150.5 gram' / '0.25 gram', by integer arithmetic only."""
    require_int(mg, "mg")
    grams, rem = divmod(abs(mg), MG_PER_GRAM)
    text = str(grams) + (f".{rem:03d}".rstrip("0") if rem else "")
    return f"{'-' if mg < 0 else ''}{text} gram"


def format_bps(bps: int) -> str:
    """Basis points as a percentage: 30 -> '0.3', 100 -> '1', 250 -> '2.5'."""
    require_int(bps, "bps")
    whole, rem = divmod(abs(bps), 100)
    text = str(whole) + (f".{rem:02d}".rstrip("0") if rem else "")
    return f"-{text}" if bps < 0 else text


_DECIMAL_RE = re.compile(r"\d+(?:\.\d+)?")


def parse_decimal(text: str) -> Fraction | None:
    """Exact value of a plain decimal like '45000', '4,12,000' or '74.5' (no floats).
    Anything else (signs, exponents, words) returns None."""
    cleaned = text.strip().replace(",", "").lstrip("₹").strip()
    if not _DECIMAL_RE.fullmatch(cleaned):
        return None
    return Fraction(cleaned)

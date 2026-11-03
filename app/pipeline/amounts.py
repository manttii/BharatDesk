"""Parse a spoken or typed amount in code (never by the LLM)."""

from __future__ import annotations

from app.pack import Pack
from app.pipeline.numerals import parse_quantities, to_paise


def parse_amount_paise(text: str) -> int | None:
    """Exactly one number in `text`, read as rupees unless it says paise. None if unclear."""
    quantities = parse_quantities(text)
    if len(quantities) != 1:
        return None
    return to_paise(quantities[0], unitless_as_rupees=True)


def within_bounds(pack: Pack, value: int, lo_key: str, hi_key: str) -> bool:
    lo, hi = pack.bounds.get(lo_key), pack.bounds.get(hi_key)
    return not ((lo is not None and value < lo) or (hi is not None and value > hi))

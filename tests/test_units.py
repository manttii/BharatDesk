from __future__ import annotations

from fractions import Fraction

import pytest

from app.units import (
    TOLA_MG,
    ceil_div,
    format_bps,
    format_inr,
    format_weight_mg,
    grams_to_mg,
    kg_to_mg,
    require_int,
    rupees_to_paise,
    tola_to_mg,
)


@pytest.mark.parametrize(
    ("paise", "text"),
    [
        (0, "₹0"),
        (100, "₹1"),
        (99, "₹0.99"),
        (100_000, "₹1,000"),
        (99_900, "₹999"),
        (4_500_000, "₹45,000"),
        (41_200_000, "₹4,12,000"),  # the spec's example
        (123_456_700, "₹12,34,567"),
        (1_000_000_000, "₹1,00,00,000"),
        (12_345_678_900, "₹12,34,56,789"),
        (41_200_050, "₹4,12,000.50"),
        (-41_200_000, "-₹4,12,000"),
        (7_520_000, "₹75,200"),
    ],
)
def test_format_inr_uses_indian_grouping(paise: int, text: str) -> None:
    assert format_inr(paise) == text


@pytest.mark.parametrize(
    ("mg", "text"),
    [(150_000, "150 gram"), (150_500, "150.5 gram"), (250, "0.25 gram"), (0, "0 gram"),
     (1, "0.001 gram"), (11_664, "11.664 gram")],
)  # fmt: skip
def test_format_weight(mg: int, text: str) -> None:
    assert format_weight_mg(mg) == text


@pytest.mark.parametrize(
    ("bps", "text"), [(0, "0"), (30, "0.3"), (100, "1"), (250, "2.5"), (5, "0.05"), (10_000, "100")]
)
def test_format_bps(bps: int, text: str) -> None:
    assert format_bps(bps) == text


def test_conversions_are_exact() -> None:
    assert rupees_to_paise(45_000) == 4_500_000
    assert rupees_to_paise(Fraction(5, 2)) == 250
    assert grams_to_mg(150) == 150_000
    assert grams_to_mg(Fraction(1, 2)) == 500
    assert kg_to_mg(Fraction(5, 2)) == 2_500_000
    assert tola_to_mg(1) == TOLA_MG == 11_664
    assert tola_to_mg(Fraction(3, 2)) == 17_496


def test_conversion_refuses_fractional_paise_and_milligrams() -> None:
    with pytest.raises(ValueError, match="whole number of paise"):
        rupees_to_paise(Fraction(1, 300))
    with pytest.raises(ValueError, match="whole number of milligrams"):
        grams_to_mg(Fraction(1, 10_000))


@pytest.mark.parametrize("bad", [1.5, 2.0, "100", None, True])
def test_floats_and_non_numbers_are_refused(bad: object) -> None:
    for fn in (rupees_to_paise, grams_to_mg, kg_to_mg, tola_to_mg):
        with pytest.raises(TypeError):
            fn(bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        require_int(bad)
    with pytest.raises(TypeError):
        format_inr(bad)  # type: ignore[arg-type]


def test_ceil_div() -> None:
    assert (ceil_div(10, 5), ceil_div(11, 5), ceil_div(0, 7), ceil_div(1, 7)) == (2, 3, 0, 1)
    with pytest.raises(ValueError, match="denominator > 0"):
        ceil_div(1, 0)

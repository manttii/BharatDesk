"""Table-driven tests for the deterministic numeral parser (>= 150 cases)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from app.pipeline.numerals import (
    Unit,
    cross_check_money,
    cross_check_weight,
    normalise_word,
    parse_number,
    parse_quantities,
    to_milligrams,
    to_paise,
)


def _pairs(spec: str) -> list[tuple[str, int]]:
    out = []
    for item in spec.split():
        word, _, value = item.partition("=")
        out.append((word, int(value)))
    return out


# Independently typed gold lists (standard spellings), 1-99.
ROMAN_1_99 = _pairs(
    """ek=1 do=2 teen=3 chaar=4 paanch=5 chhe=6 saat=7 aath=8 nau=9 das=10 gyarah=11 barah=12
    terah=13 chaudah=14 pandrah=15 solah=16 satrah=17 atharah=18 unnis=19 bees=20 ikkis=21
    baais=22 teis=23 chaubis=24 pachchis=25 chhabbis=26 sattais=27 atthais=28 untis=29 tees=30
    ikattis=31 battis=32 taintis=33 chauntis=34 paintis=35 chhattis=36 saintis=37 adtis=38
    unchalis=39 chalis=40 iktalis=41 bayalis=42 taintalis=43 chauvalis=44 paintalis=45
    chhiyalis=46 saintalis=47 adtalis=48 unchas=49 pachas=50 ikyavan=51 baavan=52 tirpan=53
    chauvan=54 pachpan=55 chhappan=56 sattavan=57 atthavan=58 unsath=59 saath=60 iksath=61
    baasath=62 tirsath=63 chaunsath=64 painsath=65 chhiyasath=66 sadsath=67 adsath=68
    unhattar=69 sattar=70 ikhattar=71 bahattar=72 tihattar=73 chauhattar=74 pachattar=75
    chhihattar=76 satattar=77 athattar=78 unyasi=79 assi=80 ikyasi=81 bayasi=82 tirasi=83
    chaurasi=84 pachasi=85 chhiyasi=86 sataasi=87 athasi=88 navasi=89 nabbe=90 ikyanave=91
    baanave=92 tiranave=93 chauranave=94 pachanave=95 chhiyanave=96 sattanave=97 atthanave=98
    ninyanave=99"""
)
DEVANAGARI_1_99 = _pairs(
    """एक=1 दो=2 तीन=3 चार=4 पाँच=5 छह=6 सात=7 आठ=8 नौ=9 दस=10 ग्यारह=11 बारह=12 तेरह=13
    चौदह=14 पंद्रह=15 सोलह=16 सत्रह=17 अठारह=18 उन्नीस=19 बीस=20 इक्कीस=21 बाईस=22 तेईस=23
    चौबीस=24 पच्चीस=25 छब्बीस=26 सत्ताईस=27 अट्ठाईस=28 उनतीस=29 तीस=30 इकतीस=31 बत्तीस=32
    तैंतीस=33 चौंतीस=34 पैंतीस=35 छत्तीस=36 सैंतीस=37 अड़तीस=38 उनतालीस=39 चालीस=40
    इकतालीस=41 बयालीस=42 तैंतालीस=43 चौवालीस=44 पैंतालीस=45 छियालीस=46 सैंतालीस=47
    अड़तालीस=48 उनचास=49 पचास=50 इक्यावन=51 बावन=52 तिरपन=53 चौवन=54 पचपन=55 छप्पन=56
    सत्तावन=57 अट्ठावन=58 उनसठ=59 साठ=60 इकसठ=61 बासठ=62 तिरसठ=63 चौंसठ=64 पैंसठ=65
    छियासठ=66 सड़सठ=67 अड़सठ=68 उनहत्तर=69 सत्तर=70 इकहत्तर=71 बहत्तर=72 तिहत्तर=73
    चौहत्तर=74 पचहत्तर=75 छिहत्तर=76 सतहत्तर=77 अठहत्तर=78 उन्यासी=79 अस्सी=80 इक्यासी=81
    बयासी=82 तिरासी=83 चौरासी=84 पचासी=85 छियासी=86 सत्तासी=87 अट्ठासी=88 नवासी=89
    नब्बे=90 इक्यानवे=91 बानवे=92 तिरानवे=93 चौरानवे=94 पचानवे=95 छियानवे=96 सत्तानवे=97
    अट्ठानवे=98 निन्यानवे=99"""
)


def test_gold_lists_are_complete() -> None:
    assert sorted(v for _, v in ROMAN_1_99) == list(range(1, 100))
    assert sorted(v for _, v in DEVANAGARI_1_99) == list(range(1, 100))


@pytest.mark.parametrize(("word", "value"), ROMAN_1_99)
def test_roman_hindi_1_to_99(word: str, value: int) -> None:
    assert parse_number(word) == value


@pytest.mark.parametrize(("word", "value"), DEVANAGARI_1_99)
def test_devanagari_hindi_1_to_99(word: str, value: int) -> None:
    assert parse_number(word) == value


@pytest.mark.parametrize(
    ("word", "value"),
    [
        ("tin", 3), ("char", 4), ("panch", 5), ("chhah", 6), ("che", 6), ("ath", 8),
        ("gyaarah", 11), ("baarah", 12), ("chaudaha", 14), ("pandra", 15), ("sola", 16),
        ("bis", 20), ("ikkees", 21), ("pachees", 25), ("pachis", 25), ("chaalis", 40),
        ("pachaas", 50), ("bavan", 52), ("sath", 60), ("saath", 60), ("PAINTALIS", 45),
        ("Paintalis", 45), ("shunya", 0), ("zero", 0), ("पांच", 5), ("पन्द्रह", 15),
    ],
)  # fmt: skip
def test_spelling_variants(word: str, value: int) -> None:
    assert parse_number(word) == value


@pytest.mark.parametrize(
    ("text", "value"),
    [
        # prefixes (exact fractions): sava +1/4, saade +1/2, paune -1/4, dedh 3/2, dhai 5/2
        ("dedh sau", 150), ("dhai sau", 250), ("sava sau", 125), ("saade teen sau", 350),
        ("paune sau", 75), ("paune do sau", 175), ("sava do sau", 225), ("saade paanch sau", 550),
        ("sava hazaar", 1250), ("dedh hazaar", 1500), ("dhai hazaar", 2500),
        ("saade teen hazaar", 3500), ("paune teen hazaar", 2750), ("sava lakh", 125_000),
        ("dedh lakh", 150_000), ("dhai lakh", 250_000), ("saade teen lakh", 350_000),
        ("paune do lakh", 175_000), ("sava crore", 12_500_000), ("dedh crore", 15_000_000),
        # bare multipliers and compounds
        ("sau", 100), ("hazaar", 1000), ("lakh", 100_000), ("crore", 10_000_000),
        ("ek sau", 100), ("do sau", 200), ("ek sau paintalis", 145), ("do sau pachas", 250),
        ("paanch sau", 500), ("nau sau ninyanave", 999), ("paintalis sau", 4500),
        ("ek hazaar", 1000), ("do hazaar", 2000), ("paintalis hazaar", 45_000),
        ("chalis hazaar", 40_000), ("ek hazaar paanch sau", 1500),
        ("do hazaar paanch sau", 2500), ("teen hazaar do sau paintalis", 3245),
        ("das hazaar", 10_000), ("bees hazaar", 20_000), ("pachas hazaar", 50_000),
        ("sattar hazaar", 70_000), ("ek lakh", 100_000), ("ek lakh pachas hazaar", 150_000),
        ("do lakh", 200_000), ("teen lakh paintalis hazaar", 345_000), ("paanch lakh", 500_000),
        ("das lakh", 1_000_000), ("ek crore", 10_000_000), ("do crore pachas lakh", 25_000_000),
        ("chalis lakh", 4_000_000), ("paintalis lakh", 4_500_000),
        ("ek lakh paintalis hazaar paanch sau", 145_500), ("do sau hazaar", 200_000),
        ("ek sau paintalis hazaar", 145_000), ("saade teen sau paintalis", 395),
        # digits
        ("150", 150), ("4,12,000", 412_000), ("75200", 75_200), ("412000", 412_000),
        ("1,000", 1000), ("2.5 lakh", 250_000), ("2 lakh", 200_000), ("2 lakh 50 hazaar", 250_000),
        ("45 hazaar", 45_000), ("1.5 hazaar", 1500), ("10 lakh", 1_000_000),
        ("1,00,00,000", 10_000_000), ("१५०", 150), ("४,१२,०००", 412_000), ("२ लाख", 200_000),
        # English
        ("forty five", 45), ("one hundred fifty", 150), ("two thousand five hundred", 2500),
        ("fifty thousand", 50_000), ("one lakh", 100_000), ("twenty five", 25),
        ("ninety nine", 99), ("three hundred", 300), ("ten", 10),
        ("seventy five thousand", 75_000),
        # Devanagari phrases
        ("एक सौ पचास", 150), ("डेढ़ सौ", 150), ("डेढ सौ", 150), ("ढाई सौ", 250), ("सवा सौ", 125),
        ("साढ़े तीन सौ", 350), ("साढे तीन सौ", 350), ("पौने दो सौ", 175),
        ("पैंतालीस हज़ार", 45_000), ("पैंतालीस हजार", 45_000), ("दो लाख पचास हज़ार", 250_000),
        ("सवा लाख", 125_000), ("डेढ़ लाख", 150_000), ("साढ़े तीन हज़ार", 3500),
        ("एक करोड़", 10_000_000), ("चालीस हज़ार", 40_000), ("पाँच सौ", 500), ("पांच सौ", 500),
        ("ढाई हजार", 2500),
        # mixed scripts / languages
        ("dedh सौ", 150), ("paintalis हज़ार", 45_000), ("2 hazaar", 2000), ("teen सौ", 300),
        ("ek लाख", 100_000), ("two सौ", 200), ("5 sau", 500),
    ],
)  # fmt: skip
def test_phrases(text: str, value: int) -> None:
    assert parse_number(text) == value


@pytest.mark.parametrize(
    ("a", "b", "va", "vb"),
    [
        ("chaalis", "paintalis", 40, 45),
        ("chalis", "chauvalis", 40, 44),
        ("sattar", "saintis", 70, 37),
        ("sattar", "satrah", 70, 17),
        ("sattar", "sattais", 70, 27),
        ("sattar", "sattavan", 70, 57),
        ("tees", "teen", 30, 3),
        ("tees", "teis", 30, 23),
        ("chaudah", "chaubis", 14, 24),
        ("chaudah", "chauvan", 14, 54),
        ("pachas", "pachpan", 50, 55),
        ("pachas", "pachasi", 50, 85),
        ("saath", "saat", 60, 7),
        ("saath", "sattar", 60, 70),
        ("unnis", "untis", 19, 29),
        ("unchas", "unsath", 49, 59),
        ("bees", "baais", 20, 22),
        ("paintis", "paintalis", 35, 45),
        ("painsath", "paintalis", 65, 45),
        ("nau", "nabbe", 9, 90),
    ],
)
def test_confusable_pairs_stay_distinct(a: str, b: str, va: int, vb: int) -> None:
    assert (parse_number(a), parse_number(b)) == (va, vb)
    assert normalise_word(a) != normalise_word(b)


def test_paintis_and_paintalis_are_different_numbers() -> None:
    """paintis is 35 and paintalis is 45: they must never be treated as spelling variants
    of each other (that would silently turn ₹35,000 into ₹45,000)."""
    assert parse_number("paintis hazaar") == 35_000
    assert parse_number("paintalis hazaar") == 45_000


# --- units, spans, conversions -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "value", "unit"),
    [
        ("150 gram", 150, Unit.GRAM), ("150gram", 150, Unit.GRAM), ("150g", 150, Unit.GRAM),
        ("dedh sau gram", 150, Unit.GRAM), ("डेढ़ सौ ग्राम", 150, Unit.GRAM),
        ("do kilo", 2, Unit.KG), ("2 kg", 2, Unit.KG),
        ("dhai kilo", Fraction(5, 2), Unit.KG), ("sava kilo", Fraction(5, 4), Unit.KG),
        ("paune kilo", Fraction(3, 4), Unit.KG), ("saade teen kilo", Fraction(7, 2), Unit.KG),
        ("ek tola", 1, Unit.TOLA), ("do tola", 2, Unit.TOLA), ("dedh tola", Fraction(3, 2), Unit.TOLA),
        ("paintalis hazaar rupaye", 45_000, Unit.RUPEE), ("पैंतालीस हज़ार रुपये", 45_000, Unit.RUPEE),
        ("₹4,12,000", 412_000, Unit.RUPEE), ("Rs. 4500", 4500, Unit.RUPEE),
        ("rs 4500", 4500, Unit.RUPEE), ("500 paise", 500, Unit.PAISE),
        ("2.5 rupaye", Fraction(5, 2), Unit.RUPEE),
    ],
)  # fmt: skip
def test_units(text: str, value: int | Fraction, unit: Unit) -> None:
    (q,) = parse_quantities(text)
    assert (q.value, q.unit) == (Fraction(value), unit)


def test_quantity_span_points_into_the_original_text() -> None:
    text = "Rajendra Jewellers gaya tha, dedh sau gram necklace, paintalis hazaar advance"
    q1, q2 = parse_quantities(text)
    assert q1.text == "dedh sau gram" and text[q1.start : q1.end] == q1.text
    assert q2.text == "paintalis hazaar" and text[q2.start : q2.end] == q2.text


def test_acceptance_sentence() -> None:
    text = "Rajendra Jewellers gaya tha, dedh sau gram necklace, paintalis hazaar advance, kal delivery"
    weight, advance = parse_quantities(text)
    assert to_milligrams(weight) == 150_000
    assert advance.unit is None and to_paise(advance, unitless_as_rupees=True) == 4_500_000
    assert to_paise(advance) is None  # unit-less is not assumed to be money unless asked


@pytest.mark.parametrize(
    ("text", "mg"),
    [("150 gram", 150_000), ("2 kg", 2_000_000), ("ek tola", 11_664), ("dedh tola", 17_496),
     ("0.5 gram", 500), ("dhai kilo", 2_500_000), ("sava kilo", 1_250_000), ("paune kilo", 750_000),
     ("0.0001 gram", None), ("150 rupaye", None), ("150", None)],
)  # fmt: skip
def test_to_milligrams(text: str, mg: int | None) -> None:
    (q,) = parse_quantities(text)
    assert to_milligrams(q) == mg


@pytest.mark.parametrize(
    ("text", "paise"),
    [("paintalis hazaar rupaye", 4_500_000), ("500 paise", 500), ("2.5 rupaye", 250),
     ("₹4,12,000", 41_200_000), ("0.005 rupaye", None), ("150 gram", None), ("150", None)],
)  # fmt: skip
def test_to_paise(text: str, paise: int | None) -> None:
    (q,) = parse_quantities(text)
    assert to_paise(q) == paise


# --- phrase boundaries -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "values"),
    [
        ("do teen", [2, 3]),
        ("ek do teen", [1, 2, 3]),
        ("do hazaar do lakh", [2000, 200_000]),
        ("paanch sau paanch sau", [500, 500]),
        ("ek sau paanch sau", [100, 500]),
        ("twenty twenty", [20, 20]),
        ("150 gram 4500 rupaye", [150, 4500]),
        ("do sau gram aur teen sau gram", [200, 300]),
        ("kal delivery", []),
        ("Rajendra Jewellers gaya tha", []),
        ("", []),
    ],
)
def test_adjacent_numbers_are_not_summed(text: str, values: list[int]) -> None:
    assert [q.value for q in parse_quantities(text)] == [Fraction(v) for v in values]


def test_known_false_positive_do_as_a_verb() -> None:
    """LIMITATION, documented on purpose: 'do' is also a Hindi verb ("kar do"). The parser
    reports a 2; callers only use quantities to cross-check LLM output, never to invent it."""
    assert [q.value for q in parse_quantities("rate kam kar do")] == [Fraction(2)]


# --- cross-checks ------------------------------------------------------------------------

SENTENCE = "Rajendra gaya tha, dedh sau gram necklace, paintalis hazaar advance, kal delivery"


@pytest.mark.parametrize(
    ("claimed_paise", "agrees"),
    [(4_500_000, True), (4_000_000, False), (5_400_000, False), (450_000, False), (0, False)],
)
def test_money_cross_check_catches_confusions(claimed_paise: int, agrees: bool) -> None:
    check = cross_check_money(claimed_paise, parse_quantities(SENTENCE))
    assert check.agrees is agrees


@pytest.mark.parametrize(
    ("claimed_mg", "agrees"), [(150_000, True), (105_000, False), (1_500_000, False)]
)
def test_weight_cross_check(claimed_mg: int, agrees: bool) -> None:
    assert cross_check_weight(claimed_mg, parse_quantities(SENTENCE)).agrees is agrees


def test_bare_number_is_never_assumed_to_be_a_weight() -> None:
    assert not cross_check_weight(150_000, parse_quantities("150 necklace")).agrees


def test_cross_check_exposes_candidates_for_the_readback() -> None:
    check = cross_check_money(4_000_000, parse_quantities("paintalis hazaar advance"))
    assert not check.agrees and check.candidates == (4_500_000,)


def test_digits_and_words_agree_on_the_same_amount() -> None:
    for spoken in ("paintalis hazaar", "45000", "45,000", "४५,०००", "45 hazaar", "पैंतालीस हज़ार"):
        assert cross_check_money(4_500_000, parse_quantities(spoken)).agrees, spoken


# --- table integrity ---------------------------------------------------------------------


def test_normalise_word_is_idempotent() -> None:
    for word, _ in ROMAN_1_99 + DEVANAGARI_1_99:
        once = normalise_word(word)
        assert normalise_word(once) == once


def test_outputs_are_exact_never_float() -> None:
    for text in ("dhai hazaar", "sava kilo", "2.5 lakh", "paune sau", "1.5 hazaar"):
        for q in parse_quantities(text):
            assert isinstance(q.value, Fraction)

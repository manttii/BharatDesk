"""Deterministic number/unit parser for Hindi, Devanagari, English and mixed transcripts.

Not a grammar or NLP system: a lookup table plus a small accumulator.
  * Words: Hindi 0-99 (Roman + Devanagari, spelling variants), English 0-99.
  * Multipliers: sau, hazaar, lakh, crore (and English hundred/thousand).
  * Fractional prefixes, exact via Fraction: sava (+1/4), saade (+1/2), paune (-1/4),
    plus dedh (3/2) and dhai (5/2) as words. "saade teen sau" = 350, "sava sau" = 125,
    "paune sau" = 75, "dedh sau" = 150, "dhai hazaar" = 2500.
  * Digits, Devanagari digits, Indian grouping ("4,12,000"), decimals ("2.5 lakh").
  * Units: gram, kg, tola, rupaye, paise.

Used to CROSS-CHECK numbers the LLM extracted: if the LLM's amount is not among the
values this parser finds in the transcript, the caller must force a read-back.

Known limitations (see tests): "do" is also a Hindi verb ("kar do"), so a stray 2 can
appear; unit-less numbers are only treated as rupees for money checks; "hundred and
fifty" splits at "and".
"""

from __future__ import annotations

import enum
import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction
from typing import TypeVar

from app.units import MG_PER_GRAM, MG_PER_KG, TOLA_MG


class Unit(enum.StrEnum):
    GRAM = "gram"
    KG = "kg"
    TOLA = "tola"
    RUPEE = "rupee"
    PAISE = "paise"


# --------------------------------------------------------------------------- normalisation

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_VOWEL_SIMPLIFY = (("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"))
_REPEATS = re.compile(r"(.)\1+")


def normalise_word(word: str) -> str:
    """Fold spelling noise so variants meet: nukta/chandrabindu, aa/ee/oo, doubled letters."""
    w = unicodedata.normalize("NFD", word.lower()).replace("़", "")  # nukta
    w = unicodedata.normalize("NFC", w).replace("ँ", "ं")  # chandrabindu -> anusvara
    if w.isascii() and w.isalpha():  # never fold digits: "2026" must stay "2026"
        for a, b in _VOWEL_SIMPLIFY:
            w = w.replace(a, b)
        w = _REPEATS.sub(r"\1", w)
    return w


# --------------------------------------------------------------------------- word tables

# value: "roman variants ... devanagari variants ..." (first spelling is the standard one)
_HINDI_0_99: dict[int, str] = {
    0: "shunya zero शून्य",
    1: "ek एक",
    2: "do दो",
    3: "teen tin तीन",
    4: "chaar char चार",
    5: "paanch panch पाँच पांच",
    6: "chhe chhah chah che chhai छह छः छै",
    7: "saat सात",
    8: "aath ath आठ",
    9: "nau नौ",
    10: "das दस",
    11: "gyarah gyaarah giyarah ग्यारह",
    12: "barah baarah बारह",
    13: "terah तेरह",
    14: "chaudah chaudaha चौदह",
    15: "pandrah pandra पंद्रह पन्द्रह",
    16: "solah sola सोलह",
    17: "satrah सत्रह",
    18: "atharah athaarah अठारह",
    19: "unnis unnees उन्नीस",
    20: "bees bis बीस",
    21: "ikkis ikkees ekkis इक्कीस",
    22: "baais baaees bais बाईस",
    23: "teis tayis taees तेईस",
    24: "chaubis chaubees चौबीस",
    25: "pachchis pachis pachees पच्चीस",
    26: "chhabbis chhabis छब्बीस",
    27: "sattais sataees सत्ताईस",
    28: "atthais athais atthaees अट्ठाईस",
    29: "untis unatis unnatees उनतीस",
    30: "tees tis तीस",
    31: "ikattis iktis ikatees इकतीस",
    32: "battis battees बत्तीस",
    33: "taintis tentis taintees तैंतीस",
    34: "chauntis chauntees चौंतीस",
    35: "paintis paintees पैंतीस",
    36: "chhattis chhatis छत्तीस",
    37: "saintis saitees सैंतीस",
    38: "adtis adatis artis अड़तीस",
    39: "unchalis untalis unchaalis उनतालीस उन्तालीस",
    40: "chalis chaalis चालीस",
    41: "iktalis ikatalis इकतालीस",
    42: "bayalis byalis बयालीस",
    43: "taintalis tentalis तैंतालीस",
    44: "chauvalis chavalis chauwalis चौवालीस चवालीस",
    45: "paintalis paentalis पैंतालीस",
    46: "chhiyalis chiyalis छियालीस",
    47: "saintalis santalis सैंतालीस",
    48: "adtalis artalis adtaalis अड़तालीस",
    49: "unchas unanchas उनचास",
    50: "pachas pachaas पचास",
    51: "ikyavan ikyaavan इक्यावन",
    52: "baavan bavan बावन",
    53: "tirpan tirepan trepan तिरपन तिरेपन",
    54: "chauvan chauwan चौवन",
    55: "pachpan pachchpan पचपन",
    56: "chhappan chhapan छप्पन",
    57: "sattavan sattaavan सत्तावन",
    58: "atthavan athaavan अट्ठावन",
    59: "unsath unsaath unasath उनसठ",
    60: "saath साठ",
    61: "iksath ikasath इकसठ",
    62: "baasath basath बासठ",
    63: "tirsath तिरसठ",
    64: "chaunsath chausath चौंसठ",
    65: "painsath पैंसठ",
    66: "chhiyasath chhiyaasath छियासठ",
    67: "sadsath sarsath सड़सठ",
    68: "adsath athsath अड़सठ",
    69: "unhattar उनहत्तर",
    70: "sattar सत्तर",
    71: "ikhattar ikahattar इकहत्तर",
    72: "bahattar बहत्तर",
    73: "tihattar तिहत्तर",
    74: "chauhattar चौहत्तर",
    75: "pachattar pachhattar पचहत्तर",
    76: "chhihattar छिहत्तर",
    77: "satattar sathattar सतहत्तर",
    78: "athattar अठहत्तर",
    79: "unyasi unaasi उन्यासी",
    80: "assi अस्सी",
    81: "ikyasi इक्यासी",
    82: "bayasi बयासी",
    83: "tirasi तिरासी",
    84: "chaurasi चौरासी",
    85: "pachasi पचासी",
    86: "chhiyasi छियासी",
    87: "sataasi सत्तासी",
    88: "athasi अट्ठासी",
    89: "navasi नवासी",
    90: "nabbe नब्बे",
    91: "ikyanave इक्यानवे",
    92: "baanave बानवे",
    93: "tiranave तिरानवे",
    94: "chauranave चौरानवे",
    95: "pachanave पचानवे",
    96: "chhiyanave छियानवे",
    97: "sattanave सत्तानवे",
    98: "atthanave अट्ठानवे",
    99: "ninyanave ninanave निन्यानवे",
}

_ENGLISH_UNITS = "one two three four five six seven eight nine".split()
_ENGLISH_TEENS = (
    "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
)
_ENGLISH_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()

_SMALL: dict[str, Fraction] = {}
_EN_TENS: set[str] = set()
_PREFIX: dict[str, Fraction] = {}
_MULT: dict[str, int] = {}
_UNIT: dict[str, Unit] = {}


V = TypeVar("V")


def _register(table: dict[str, V], words: str, value: V) -> None:
    for word in words.split():
        key = normalise_word(word)
        if key in table and table[key] != value:
            raise ValueError(f"spelling collision: {word!r} -> {key!r} maps to two values")
        table[key] = value


for _n, _words in _HINDI_0_99.items():
    _register(_SMALL, _words, Fraction(_n))
for _i, _w in enumerate(_ENGLISH_UNITS, start=1):
    _register(_SMALL, _w, Fraction(_i))
for _i, _w in enumerate(_ENGLISH_TEENS, start=10):
    _register(_SMALL, _w, Fraction(_i))
for _i, _w in enumerate(_ENGLISH_TENS, start=2):
    _register(_SMALL, _w, Fraction(_i * 10))
    _EN_TENS.add(normalise_word(_w))
# dedh / dhai are whole words (1.5 / 2.5), unlike the prefixes below.
_register(_SMALL, "dedh derh dedha डेढ़ डेढ", Fraction(3, 2))
_register(_SMALL, "dhai dhaai ढाई", Fraction(5, 2))

_register(_PREFIX, "sava sawa सवा", Fraction(1, 4))
_register(_PREFIX, "saade sade sadhe साढ़े साढे साडे", Fraction(1, 2))
_register(_PREFIX, "paune paun pone पौने पौन", Fraction(-1, 4))

_register(_MULT, "sau sao सौ hundred", 100)
_register(_MULT, "hazaar hazar hajar hajaar हज़ार हजार thousand", 1_000)
_register(_MULT, "lakh lac lakhs लाख", 100_000)
_register(_MULT, "crore karod crores करोड़ करोड", 10_000_000)

_register(_UNIT, "gram grams gm gms g ग्राम", Unit.GRAM)
_register(_UNIT, "kg kgs kilo kilos kilogram kilograms किलो किलोग्राम", Unit.KG)
_register(_UNIT, "tola tole tolay तोला तोले", Unit.TOLA)
_register(_UNIT, "rupaye rupay rupaya rupiya rupiye rupees rupee rs inr रुपये रुपए रुपया", Unit.RUPEE)
_register(_UNIT, "paise paisa paisey पैसे पैसा", Unit.PAISE)

_tables = (_SMALL.keys(), _PREFIX.keys(), _MULT.keys(), _UNIT.keys())
for _a in range(len(_tables)):
    for _b in range(_a + 1, len(_tables)):
        _clash = set(_tables[_a]) & set(_tables[_b])
        if _clash:
            raise ValueError(f"word appears in two number tables: {sorted(_clash)}")

# --------------------------------------------------------------------------- tokenizer

NUM, SMALL, PREFIX, MULT, UNIT, OTHER = "num", "small", "prefix", "mult", "unit", "other"

_TOKEN_RE = re.compile(
    r"(?P<num>\d+(?:,\d{2,3}(?!\d))*(?:\.\d+)?)"
    r"|(?P<cur>₹)"
    r"|(?P<word>[A-Za-z]+|[ऀ-ॣ॥-ॿ]+)"
)


@dataclass(frozen=True)
class _Tok:
    kind: str
    start: int
    end: int
    value: Fraction | int = 0
    unit: Unit | None = None
    en_tens: bool = False


def _tokenise(text: str) -> list[_Tok]:
    text = text.translate(_DEVANAGARI_DIGITS)  # length-preserving, so spans stay valid
    toks: list[_Tok] = []
    for m in _TOKEN_RE.finditer(text):
        start, end = m.span()
        if m.group("num") is not None:
            toks.append(_Tok(NUM, start, end, Fraction(m.group("num").replace(",", ""))))
        elif m.group("cur") is not None:
            toks.append(_Tok(UNIT, start, end, unit=Unit.RUPEE))
        else:
            key = normalise_word(m.group("word"))
            if key in _SMALL:
                toks.append(_Tok(SMALL, start, end, _SMALL[key], en_tens=key in _EN_TENS))
            elif key in _PREFIX:
                toks.append(_Tok(PREFIX, start, end, _PREFIX[key]))
            elif key in _MULT:
                toks.append(_Tok(MULT, start, end, _MULT[key]))
            elif key in _UNIT:
                toks.append(_Tok(UNIT, start, end, unit=_UNIT[key]))
            else:
                toks.append(_Tok(OTHER, start, end))
    return toks


# --------------------------------------------------------------------------- phrase reader


def _read_phrase(toks: list[_Tok], i: int) -> tuple[int, Fraction]:
    """Read one number phrase starting at toks[i]; returns (next index, exact value).

    A phrase ends (without consuming the offending token) when the next token cannot
    continue it, e.g. two bare values in a row ("do teen" is 2 and 3, not 5), a
    repeated "sau", or a larger big-multiplier after a smaller one.
    """
    total = Fraction(0)  # closed groups (thousand/lakh/crore)
    cur = Fraction(0)  # group being built
    pending: Fraction | None = None  # sava/saade/paune delta waiting for its number
    last = ""  # '', 'val', 'tens', 'prefix', 'm100', 'big'
    last_big: int | None = None
    # Last point where the phrase was a complete, valid number. A multiplier that cannot
    # continue the phrase rolls back to here, so "do hazaar do lakh" is 2000 then 200000,
    # not a bogus 2002 followed by 100000.
    snap: tuple[int, Fraction] | None = None
    j = i
    while j < len(toks):
        t = toks[j]
        if t.kind in (NUM, SMALL):
            if last in ("val", "tens") and not (
                last == "tens" and t.kind == SMALL and 1 <= t.value <= 9
            ):
                break  # two separate numbers ("forty five" is the one allowed exception)
            v = Fraction(t.value)
            if pending is not None:
                v += pending
                pending = None
            cur += v
            last = "tens" if t.en_tens else "val"
        elif t.kind == PREFIX:
            if last in ("val", "tens", "prefix"):
                break
            pending = Fraction(t.value)
            last = "prefix"
        elif t.kind == MULT:
            m = int(t.value)
            if m == 100:
                if last == "m100" or cur >= 100 or (last == "big" and cur == 0):
                    return snap if snap is not None else (j, total + cur)
                if last == "prefix":
                    assert pending is not None
                    cur += 1 + pending  # "sava sau": the number before sau is an implied 1
                    pending = None
                elif last == "":
                    cur = Fraction(1)
                cur *= 100
                last = "m100"
            else:
                if (last_big is not None and m >= last_big) or (last == "big" and cur == 0):
                    return snap if snap is not None else (j, total + cur)
                if last == "prefix":
                    assert pending is not None
                    cur += 1 + pending
                    pending = None
                elif last == "":
                    cur = Fraction(1)
                total += cur * m
                cur = Fraction(0)
                last_big = m
                last = "big"
            snap = (j + 1, total + cur)
        else:
            break
        j += 1
    if pending is not None:  # "sava kilo", "paune sau" handled above; bare prefix = implied 1
        cur += 1 + pending
    return j, total + cur


# --------------------------------------------------------------------------- public API


@dataclass(frozen=True)
class Quantity:
    value: Fraction
    unit: Unit | None
    start: int  # character span in the ORIGINAL text
    end: int
    text: str


def parse_quantities(text: str) -> list[Quantity]:
    """Every number phrase in `text`, with its unit when one is adjacent."""
    toks = _tokenise(text)
    out: list[Quantity] = []
    i = 0
    while i < len(toks):
        if toks[i].kind not in (NUM, SMALL, PREFIX, MULT):
            i += 1
            continue
        nxt, value = _read_phrase(toks, i)
        start, end = toks[i].start, toks[nxt - 1].end
        unit: Unit | None = None
        if nxt < len(toks) and toks[nxt].kind == UNIT:
            unit = toks[nxt].unit
            end = toks[nxt].end
            nxt += 1
        elif i > 0 and toks[i - 1].kind == UNIT and toks[i - 1].unit is Unit.RUPEE:
            unit = Unit.RUPEE  # "₹4,12,000", "Rs 4500"
            start = toks[i - 1].start
        out.append(Quantity(value, unit, start, end, text[start:end]))
        i = nxt
    return out


def parse_number(text: str) -> Fraction | None:
    """The value of `text` if it is exactly one unit-less number phrase, else None."""
    qs = parse_quantities(text)
    if len(qs) == 1 and qs[0].unit is None:
        return qs[0].value
    return None


def _whole(value: Fraction) -> int | None:
    return int(value) if value.denominator == 1 else None


def to_milligrams(q: Quantity) -> int | None:
    """Weight in mg, or None if the unit is not a weight or the result is fractional."""
    if q.unit is None:
        return None
    factor = {Unit.GRAM: MG_PER_GRAM, Unit.KG: MG_PER_KG, Unit.TOLA: TOLA_MG}.get(q.unit)
    return None if factor is None else _whole(q.value * factor)


def to_paise(q: Quantity, *, unitless_as_rupees: bool = False) -> int | None:
    """Money in paise, or None if not money or not a whole paisa."""
    if q.unit is Unit.RUPEE or (q.unit is None and unitless_as_rupees):
        return _whole(q.value * 100)
    if q.unit is Unit.PAISE:
        return _whole(q.value)
    return None


@dataclass(frozen=True)
class CrossCheck:
    agrees: bool
    candidates: tuple[int, ...]  # every value the transcript supports, for the read-back


def cross_check_money(extracted_paise: int, quantities: list[Quantity]) -> CrossCheck:
    """Does the transcript contain the amount the LLM extracted? Unit-less numbers count
    as rupees, which is how reps speak ("paintalis hazaar advance")."""
    cands = tuple(p for q in quantities if (p := to_paise(q, unitless_as_rupees=True)) is not None)
    return CrossCheck(extracted_paise in cands, cands)


def cross_check_weight(extracted_mg: int, quantities: list[Quantity]) -> CrossCheck:
    """Weights need an explicit unit; a bare number is never assumed to be grams."""
    cands = tuple(mg for q in quantities if (mg := to_milligrams(q)) is not None)
    return CrossCheck(extracted_mg in cands, cands)

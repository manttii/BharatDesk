"""Resolve spoken dates in code, never in the LLM.

The LLM copies the phrase as spoken ("kal", "agle somvar", "8 November"); this module turns
it into a date given today's date and the field's direction. In Hindi "kal" means both
yesterday and tomorrow, so the caller says which way the field points: a delivery or a
follow-up is in the future, a visit is in the past. Anything ambiguous returns None and the
rep is asked; a date is never guessed.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Literal

from app.pipeline.numerals import normalise_word, parse_number, parse_quantities

Direction = Literal["future", "past"]
MAX_SPAN_DAYS = 400

_WORD_RE = re.compile(r"[A-Za-z]+|[ऀ-ॣ॥-ॿ]+|\d+")
_NUMERIC_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})(?:[/\-.](\d{4}|\d{2}))?(?!\d)")


def _table(spec: dict[int, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value, words in spec.items():
        for w in words.split():
            out[normalise_word(w)] = value
    return out


_MONTHS = _table(
    {
        1: "january jan janwari janvari जनवरी",
        2: "february feb farwari faravari फरवरी फ़रवरी",
        3: "march mar march मार्च",
        4: "april apr april अप्रैल",
        5: "may mai मई",
        6: "june jun joon जून",
        7: "july jul julai जुलाई",
        8: "august aug agast अगस्त",
        9: "september sep sept sitambar सितंबर सितम्बर",
        10: "october oct aktubar अक्टूबर",
        11: "november nov navambar नवंबर नवम्बर",
        12: "december dec disambar दिसंबर दिसम्बर",
    }
)
# Monday = 0
_WEEKDAYS = _table(
    {
        0: "monday mon somvar somwar सोमवार",
        1: "tuesday tue tues mangalvar mangalwar मंगलवार",
        2: "wednesday wed budhvar budhwar बुधवार",
        3: "thursday thu thur thurs guruvar guruwar brihaspativar वीरवार गुरुवार बृहस्पतिवार",
        4: "friday fri shukravar shukrawar शुक्रवार",
        5: "saturday sat shanivar shaniwar शनिवार",
        6: "sunday sun ravivar raviwar itwar इतवार रविवार",
    }
)
_NEXT = {normalise_word(w) for w in "agle agla next aane aaney अगले अगला आने".split()}
_LAST = {normalise_word(w) for w in "pichle pichla last beete पिछले पिछला बीते".split()}
_TODAY = {normalise_word(w) for w in "aaj today आज".split()}
_KAL = {normalise_word(w) for w in "kal कल".split()}
_PARSO = {normalise_word(w) for w in "parso parson परसों परसो".split()}
_TOMORROW = {normalise_word(w) for w in "tomorrow".split()}
_YESTERDAY = {normalise_word(w) for w in "yesterday".split()}
_DAY_WORDS = {normalise_word(w) for w in "din days day दिन".split()}
_WEEK_WORDS = {normalise_word(w) for w in "hafte hafta week weeks हफ्ते हफ़्ते हफ्ता सप्ताह".split()}
_AFTER = {normalise_word(w) for w in "baad mein me later after बाद में".split()}
_BEFORE = {normalise_word(w) for w in "pehle pahle ago पहले".split()}
_MOST_DAY_NUMBER = 31


_ORDINAL_SUFFIXES = {"th", "st", "nd", "rd"}


def _words(text: str) -> list[str]:
    """Normalised words and numbers; ordinal suffixes ('8th') are dropped."""
    raw = _WORD_RE.findall(text.lower())
    return [normalise_word(w) for w in raw if w not in _ORDINAL_SUFFIXES]


def _offset(today: date, days: int) -> date | None:
    d = today + timedelta(days=days)
    return d if abs(days) <= MAX_SPAN_DAYS else None


def _from_day_month(
    day: int, month: int, year: int | None, today: date, direction: Direction
) -> date | None:
    if not (1 <= day <= _MOST_DAY_NUMBER and 1 <= month <= 12):
        return None
    if year is not None:
        try:
            return date(year, month, day)
        except ValueError:
            return None
    candidates: list[date] = []
    for y in (today.year - 1, today.year, today.year + 1):
        try:
            candidates.append(date(y, month, day))
        except ValueError:
            continue
    if direction == "future":
        ok = [c for c in candidates if c >= today]
        return min(ok) if ok else None
    ok = [c for c in candidates if c <= today]
    return max(ok) if ok else None


def _absolute(text: str, words: list[str], today: date, direction: Direction) -> date | None:
    m = _NUMERIC_DATE_RE.search(text)
    if m:
        year = int(m.group(3)) if m.group(3) else None
        if year is not None and year < 100:
            year += 2000
        return _from_day_month(int(m.group(1)), int(m.group(2)), year, today, direction)
    for i, w in enumerate(words):
        month = _MONTHS.get(w)
        if month is None:
            continue
        # The day is the token just before the month ("8 November") or just after it.
        for j in (i - 1, i + 1):
            if 0 <= j < len(words):
                day = int(words[j]) if words[j].isdigit() else _word_number(words[j])
                if day is not None:
                    year = None
                    if j == i + 1 and i + 2 < len(words) and re.fullmatch(r"20\d\d", words[i + 2]):
                        year = int(words[i + 2])
                    elif (
                        j == i - 1 and i + 1 < len(words) and re.fullmatch(r"20\d\d", words[i + 1])
                    ):
                        year = int(words[i + 1])
                    return _from_day_month(day, month, year, today, direction)
        return None  # a month with no usable day: ambiguous, ask
    return None


def _word_number(word: str) -> int | None:
    v = parse_number(word)
    return int(v) if v is not None and v.denominator == 1 else None


def _weekday(words: list[str], today: date, direction: Direction) -> date | None:
    for i, w in enumerate(words):
        if w not in _WEEKDAYS:
            continue
        target = _WEEKDAYS[w]
        prev = words[i - 1] if i > 0 else ""
        backwards = prev in _LAST or (direction == "past" and prev not in _NEXT)
        delta = (target - today.weekday()) % 7
        if backwards:
            delta = -(((today.weekday() - target) % 7) or 7)
        else:
            delta = delta or 7  # "agle somvar" on a Monday is next week's, never today
        return today + timedelta(days=delta)
    return None


def _counted(text: str, words: list[str], today: date, direction: Direction) -> date | None:
    """'3 din baad', 'do din mein', '2 hafte baad', 'in 5 days'."""
    for q in parse_quantities(text):
        if q.value.denominator != 1 or q.unit is not None:
            continue
        tail = _words(text[q.end :])[:2]
        if not tail:
            continue
        unit_days = 1 if tail[0] in _DAY_WORDS else 7 if tail[0] in _WEEK_WORDS else 0
        if not unit_days:
            continue
        n = int(q.value) * unit_days
        rest = set(tail[1:]) | set(words)
        if rest & _BEFORE:
            return _offset(today, -n)
        if rest & _AFTER or direction == "future":
            return _offset(today, n)
        return _offset(today, -n)
    return None


def resolve_date_phrase(text: str | None, today: date, direction: Direction) -> date | None:
    """The date `text` means, or None if it cannot be determined unambiguously."""
    if not text or not text.strip():
        return None
    words = _words(text)
    absolute = _absolute(text, words, today, direction)
    if absolute is not None:
        return absolute
    counted = _counted(text, words, today, direction)
    if counted is not None:
        return counted
    sign = 1 if direction == "future" else -1
    wset = set(words)
    if wset & _TOMORROW:
        return _offset(today, 2 if "after" in wset else 1)  # "day after tomorrow"
    if wset & _YESTERDAY:
        return _offset(today, -2 if "before" in wset else -1)
    if wset & _PARSO:
        return _offset(today, 2 * sign)
    if wset & _KAL:
        return _offset(today, sign)
    if wset & _TODAY:
        return today
    weekday = _weekday(words, today, direction)
    if weekday is not None:
        return weekday
    if wset & _WEEK_WORDS:
        if wset & _NEXT:
            return _offset(today, 7)
        if wset & _LAST:
            return _offset(today, -7)
    return None

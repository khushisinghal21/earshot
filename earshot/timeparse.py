"""Tiny relative-deadline parser for spoken English.

Returns (due_seconds, matched_text). due_seconds is relative to midnight at
the start of the replay day, so 'tomorrow' lands past 86400. Deliberately
small: spoken deadlines are a short, stable list.
"""
from __future__ import annotations

import re
from typing import Optional

DAY = 86400.0
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
END_OF_DAY = 23 * 3600 + 59 * 60

_CLOCK = re.compile(
    r"\b(?:by|at|before|around)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?(?!\s*(?:minutes|mins|hours|days))",
    re.I,
)
_IN_N = re.compile(r"\bin\s+(an?|\d+|half an)\s+(minute|minutes|min|mins|hour|hours)\b", re.I)


def parse_due(text: str, now: float, weekday: int) -> tuple[Optional[float], Optional[str]]:
    """weekday: 0=Monday for the replay day."""
    t = text.lower()

    m = _IN_N.search(t)
    if m:
        qty_raw, unit = m.group(1), m.group(2)
        qty = 0.5 if qty_raw == "half an" else 1.0 if qty_raw in ("a", "an") else float(qty_raw)
        secs = qty * (60 if unit.startswith("min") else 3600)
        return now + secs, m.group(0)

    for phrase, offset in (
        ("end of the day", END_OF_DAY),
        ("end of day", END_OF_DAY),
        ("tonight", 21 * 3600),
        ("this evening", 20 * 3600),
        ("this afternoon", 17 * 3600),
        ("this morning", 12 * 3600),
        ("after lunch", 15 * 3600),
        ("before lunch", 12 * 3600),
        ("by lunch", 12 * 3600),
        ("today", END_OF_DAY),
        ("tomorrow morning", DAY + 11 * 3600),
        ("tomorrow", DAY + END_OF_DAY),
        ("next week", 7 * DAY + END_OF_DAY),
        ("this weekend", None),
    ):
        if phrase in t:
            if offset is None:  # weekend -> Saturday
                days = (5 - weekday) % 7 or 7
                return days * DAY + END_OF_DAY, phrase
            return float(offset), phrase

    for idx, name in enumerate(WEEKDAYS):
        m = re.search(rf"\b(?:by|on|before|this|next|until)?\s*{name}\b", t)
        if m:
            days = (idx - weekday) % 7
            if days == 0 or "next" in m.group(0):
                days = days or 7
            return days * DAY + END_OF_DAY, m.group(0).strip()

    m = _CLOCK.search(t)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        mer = (m.group(3) or "").replace(".", "")
        if mer == "pm" and hour < 12:
            hour += 12
        elif not mer and hour < 8:  # "by 5" said in the daytime means 5pm
            hour += 12
        due = hour * 3600 + minute * 60
        if due < now:
            due += DAY
        return float(due), m.group(0).strip()

    return None, None

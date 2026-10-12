"""The Mind: small durable facts about the wearer's people ("Mum loves peonies").

The privacy guard is the point of this module. A wearable hears people who
never installed it. Earshot keeps a fact about someone else only when it is
the everyday kind (likes, birthdays, upcoming events) and never when it is
sensitive (health, money, legal, relationships, etc.). Sensitive facts the
wearer says about *themselves* are kept only as `needs_confirm` so the app
asks before remembering.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .models import MEDIA, WEARER, MindFact, Segment, Withheld
from .text import People, normalize, sentences

SENSITIVE = {
    "health": r"\b(?:migraine|migraines|diagnos\w*|therapy|therapist|surgery|chemo|cancer|pregnan\w*|miscarriage|medication|meds|antidepressant\w*|depress\w*|anxiety|panic attack|biopsy|hospital\w*|ivf|adhd|rehab|sober)\b",
    "money": r"\b(?:salary|debt|loan|bankrupt\w*|owe \$|credit score|laid off|fired)\b",
    "legal": r"\b(?:lawsuit|arrest\w*|court date|lawyer|divorce|custody|visa status|immigration)\b",
    "relationships": r"\b(?:affair|cheating|breaking up|broke up|dating app)\b",
}
_SENS = {k: re.compile(v, re.I) for k, v in SENSITIVE.items()}

RELATIONS = r"(?:mum|mom|mother|dad|father|daughter|son|wife|husband|partner|sister|brother|kid|kids|grandma|grandpa|boss)"

FACT_PATTERNS = [
    re.compile(rf"\b(?P<who>my {RELATIONS}|[a-z]+)(?:'s)? fav(?:ou?rite)? (?P<thing>[a-z ]+?) (?:is|are) (?P<val>[^.,;!?]+)", re.I),
    re.compile(rf"\b(?P<who>my {RELATIONS}|she|he|[a-z]+) (?:loves|adores|is obsessed with|can't stand|hates|is allergic to) (?P<val>[^.,;!?]+)", re.I),
    re.compile(rf"\b(?P<who>my {RELATIONS}|[a-z]+)'s birthday is (?P<val>[^.,;!?]+)", re.I),
    re.compile(rf"\b(?P<who>my {RELATIONS}|[a-z]+)(?:'s| has (?:a|her|his|their)) (?P<thing>recital|game|interview|exam|flight|wedding|recital|show|race|match)\b(?P<val>[^.;!?]*)", re.I),
]

PRONOUNS = {"she", "he", "her", "him"}


def sensitive_category(text: str) -> Optional[str]:
    for cat, rx in _SENS.items():
        if rx.search(text):
            return cat
    return None


@dataclass
class MindBuilder:
    people: People
    guard: bool = True  # False = naive memory that keeps everything it hears
    facts: list[MindFact] = field(default_factory=list)
    withheld: list[Withheld] = field(default_factory=list)
    _last_mentioned: Optional[str] = None
    _n: int = 0

    def observe(self, seg: Segment, now: float) -> list[MindFact]:
        if seg.speaker == MEDIA:
            return []
        new: list[MindFact] = []
        for sent in sentences(normalize(seg.text)):
            about = self._about(sent, seg)
            cat = sensitive_category(sent) if self.guard else None
            if not self.guard and sensitive_category(sent):
                new.append(self._fact(seg, now, about or seg.speaker, sent.strip(" ."), needs_confirm=False))
                continue
            if cat:
                if about == WEARER and seg.is_wearer:
                    fact = self._fact(seg, now, about, sent, needs_confirm=True)
                    new.append(fact)
                else:
                    self.withheld.append(Withheld(seg.id, now, self.people.name(about) if about else self.people.name(seg.speaker),
                                                  f"sensitive ({cat}) about someone other than the wearer: not stored"))
                continue
            for rx in FACT_PATTERNS:
                m = rx.search(sent)
                if m:
                    who = self._resolve(m.group("who"), seg)
                    if who:
                        new.append(self._fact(seg, now, who, sent.strip(" ."), needs_confirm=False))
                    break
            mentioned = self.people.mentioned(sent)
            if mentioned:
                self._last_mentioned = mentioned[0]
        return new

    def _about(self, sent: str, seg: Segment) -> Optional[str]:
        low = sent.lower()
        mentioned = [p for p in self.people.mentioned(sent) if p != WEARER]
        if mentioned:
            return mentioned[0]
        if re.search(rf"\bmy {RELATIONS}\b", low):
            return f"{seg.speaker}:relative"
        if re.search(r"\b(?:i|i'm|i've|my)\b", low):
            return seg.speaker
        return seg.speaker

    def _resolve(self, who: str, seg: Segment) -> Optional[str]:
        w = who.lower().strip()
        if w in PRONOUNS:
            return self._last_mentioned
        if w.startswith("my "):
            rel = w[3:]
            alias = self.people.aliases.get(rel)
            if seg.is_wearer and alias:
                return alias
            owner = "you" if seg.is_wearer else self.people.name(seg.speaker)
            return f"{owner}'s {rel}" if not seg.is_wearer else alias or f"your {rel}"
        return self.people.aliases.get(w)

    def _fact(self, seg: Segment, now: float, about: str, text: str, needs_confirm: bool) -> MindFact:
        self._n += 1
        f = MindFact(f"M{self._n}", about, text, seg.id, now, needs_confirm)
        self.facts.append(f)
        return f

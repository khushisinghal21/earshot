"""Shared text helpers: sentence splitting, content tokens, people resolution."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

STOP = set(
    """a an the to of for and or but on in at by with from up down out over this that these those it its
    i me my mine you your yours he him his she her hers we us our they them their is are was were be been
    am do does did done have has had will would can could should shall may might must just really so very
    too also then than there here what when where who how why which some any all no not yeah yes ok okay
    sure well like um uh oh hey gonna going get got let make made need needs today tomorrow tonight morning
    afternoon evening week next end day monday tuesday wednesday thursday friday saturday sunday by before""".split()
)

VERB_FAMILIES: dict[str, tuple[str, ...]] = {
    "send": ("send", "sent", "email", "emailed", "share", "shared", "forward", "forwarded", "text", "texted", "mail", "mailed"),
    "call": ("call", "called", "ring", "rang", "phone", "phoned"),
    "buy": ("pick", "picked", "buy", "bought", "grab", "grabbed", "get", "got"),
    "book": ("book", "booked", "reserve", "reserved", "schedule", "scheduled"),
    "fix": ("fix", "fixed", "repair", "repaired"),
    "review": ("review", "reviewed", "look", "looked", "read", "check", "checked"),
    "pay": ("pay", "paid", "venmo", "venmoed", "transfer", "transferred"),
    "write": ("write", "wrote", "draft", "drafted", "finish", "finished", "submit", "submitted"),
    "return": ("return", "returned", "drop", "dropped"),
    "water": ("water", "watered"),
    "cancel": ("cancel", "cancelled", "canceled"),
}
_VERB_TO_FAMILY = {v: fam for fam, vs in VERB_FAMILIES.items() for v in vs}

PAST_FORMS = {
    "sent", "emailed", "shared", "forwarded", "texted", "mailed", "called", "rang", "phoned", "picked",
    "bought", "grabbed", "got", "booked", "reserved", "scheduled", "fixed", "repaired", "reviewed", "looked",
    "checked", "paid", "venmoed", "transferred", "wrote", "drafted", "finished", "submitted", "returned",
    "dropped", "watered", "cancelled", "canceled",
}


def verb_family(word: str | None) -> str | None:
    if not word:
        return None
    return _VERB_TO_FAMILY.get(word.lower(), word.lower())


def normalize(text: str) -> str:
    return (
        text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').strip()
    )


def sentences(text: str) -> list[str]:
    text = normalize(text)
    parts = re.split(r"(?<=[.!?;])\s+|\s+(?:and then|and also|, and)\s+(?=i(?:'ll| will| am| 'm|'m)\b)", text, flags=re.I)
    out = []
    for p in parts:
        # "I'll call the plumber and I'll grab milk" -> two commitments
        out.extend(s for s in re.split(r"\s+and\s+(?=i'll\b|i will\b)", p, flags=re.I) if s.strip())
    return [s.strip() for s in out if s.strip()]


def content_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z'-]+", text.lower())
    toks = set()
    for w in words:
        w = w.strip("'-")
        if w.endswith("'s"):
            w = w[:-2]
        if len(w) > 2 and w not in STOP and w not in _VERB_TO_FAMILY:
            toks.add(w[:-1] if w.endswith("s") and len(w) > 4 else w)
    return toks


@dataclass
class People:
    """Enrolled people (from voiceprints / contacts). Maps aliases to ids."""

    display: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict) -> "People":
        p = cls()
        for pid, info in cfg.items():
            p.display[pid] = info.get("name", pid.title())
            for a in [pid, info.get("name", ""), *info.get("aliases", [])]:
                if a:
                    p.aliases[a.lower()] = pid
        return p

    def name(self, pid: str | None) -> str:
        if pid is None:
            return "someone"
        if pid == "me":
            return "you"
        return self.display.get(pid, pid)

    def mentioned(self, text: str) -> list[str]:
        low = text.lower()
        found: list[tuple[int, str]] = []
        for alias, pid in self.aliases.items():
            m = re.search(rf"\b{re.escape(alias)}\b", low)
            if m and pid not in [f[1] for f in found]:
                found.append((m.start(), pid))
        return [pid for _, pid in sorted(found)]

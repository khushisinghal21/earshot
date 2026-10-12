"""Score a replayed day against its hand labels.

Metrics are chosen for an always-on ear, where the costly mistakes are:
  * talking over a person          -> talk_over
  * talking during Focus/quiet     -> quiet_violations
  * filing someone else's promise as yours -> misattributed
  * keeping a bystander's private detail   -> privacy_leaks
and the value is: right loops in the ledger, the right whisper at the right time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .engine import Day, RunResult, run_day
from .ledger import COMMIT, LedgerExtractor
from .arbiter import Policy
from .models import MEDIA, WEARER, Segment, parse_clock
from .text import normalize, sentences


def _has(text: str, keys: list[str]) -> bool:
    low = text.lower()
    return all(k.lower() in low for k in keys)


def score(r: RunResult) -> dict:
    truth = r.day.truth
    out: dict = {}

    # ---- ledger ----
    t_items = truth.get("ledger", [])
    matched_pred: set[str] = set()
    hits = status_ok = cp_ok = cp_total = 0
    for t in t_items:
        cand = [i for i in r.ledger if i.owner == t["owner"] and i.kind == t["kind"] and _has(i.what, t["keys"])
                and i.id not in matched_pred]
        if not cand:
            continue
        i = cand[0]
        matched_pred.add(i.id)
        hits += 1
        status_ok += int(i.status == t.get("status", i.status))
        if "counterparty" in t:
            cp_total += 1
            cp_ok += int(i.counterparty == t["counterparty"])
    n_pred = len(r.ledger)
    out["ledger_recall"] = hits / len(t_items) if t_items else None
    out["ledger_precision"] = len(matched_pred) / n_pred if n_pred else None
    out["status_accuracy"] = status_ok / hits if hits else None
    out["counterparty_accuracy"] = cp_ok / cp_total if cp_total else None

    mine = [i for i in r.ledger if i.owner == WEARER and i.kind == "commitment"]
    not_mine = truth.get("not_my_commitments", [])
    out["misattributed"] = sum(1 for i in mine if any(k.lower() in i.what.lower() for k in not_mine))

    # ---- mind / privacy ----
    groups = truth.get("mind", [])
    texts = [f.fact for f in r.facts]
    out["mind_recall"] = sum(1 for g in groups if any(_has(x, g) for x in texts)) / len(groups) if groups else None
    stored = texts + [i.what for i in r.ledger]
    out["privacy_leaks"] = sum(1 for k in truth.get("forbidden_memory", []) for x in stored if k.lower() in x.lower())

    # ---- whispers ----
    spoke = r.spoke()
    t_nudges = truth.get("nudges", [])
    used: set[int] = set()
    n_hit = 0
    for n in t_nudges:
        lo, hi = parse_clock(n["earliest"]), parse_clock(n["latest"])
        for k, d in enumerate(spoke):
            said = (d.said or "") + " " + d.candidate.text
            if k in used or d.candidate.kind != n["kind"] or not _has(said, n["keys"]):
                continue
            if lo <= d.at <= hi:
                used.add(k)
                n_hit += 1
                break
    out["whisper_recall"] = n_hit / len(t_nudges) if t_nudges else None
    out["whispers"] = len(spoke)
    out["unrequested_whispers"] = len(spoke) - len(used)

    over_person = over_any = 0
    for d in spoke:
        a, b = d.at, d.at + d.duration
        hit = [s for s in r.day.segments if s.start < b and s.end > a]
        over_any += int(bool(hit))
        over_person += int(any(s.is_person for s in hit))
    out["talk_over_person"] = over_person
    out["talk_over_any"] = over_any

    quiet = [(parse_clock(q["from"]), parse_clock(q["to"])) for q in truth.get("quiet", [])]
    out["quiet_violations"] = sum(1 for d in spoke for lo, hi in quiet if lo <= d.at < hi)
    return out


class NaiveExtractor(LedgerExtractor):
    """Baseline: keyword-spot every "I'll ..." anywhere in the audio and file it as yours."""

    def observe(self, seg: Segment, now: float):
        changes = []
        for sent in sentences(normalize(seg.text)):
            if seg.is_wearer:
                changes += self._maybe_close(sent, seg, now, owner=WEARER)
            m = COMMIT.search(sent)
            if m:
                changes.append(self._add_or_update("commitment", WEARER, None, m.group("rest").strip(" ."), None, None,
                                                   seg, now, how="keyword 'I'll'"))
        return changes


def run_and_score(day: Day, mode: str) -> tuple[RunResult, dict]:
    if mode == "baseline":
        r = run_day(day, Policy(name="baseline", naive=True), extractor_factory=NaiveExtractor)
    else:
        r = run_day(day, Policy())
    return r, score(r)


METRIC_ORDER = [
    ("ledger_recall", "Ledger recall", "higher"),
    ("ledger_precision", "Ledger precision", "higher"),
    ("counterparty_accuracy", "Right person on the loop", "higher"),
    ("status_accuracy", "Open/closed correct", "higher"),
    ("misattributed", "Others' promises filed as yours", "lower"),
    ("mind_recall", "Mind facts kept", "higher"),
    ("privacy_leaks", "Bystander secrets stored", "lower"),
    ("whisper_recall", "Right whisper, right window", "higher"),
    ("unrequested_whispers", "Whispers nobody needed", "lower"),
    ("talk_over_person", "Talked over a person", "lower"),
    ("quiet_violations", "Spoke during Focus", "lower"),
]


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.0%}"
    return str(v)

"""Replay a day through Earshot: hear -> ledger/mind -> candidates -> speak-or-hold."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .arbiter import Arbiter, Hearing, Policy, display_what, nice_due
from .ledger import LedgerExtractor
from .mind import MindBuilder
from .models import WEARER, Candidate, ContextEvent, Decision, LedgerItem, Segment, fmt_clock, parse_clock
from .text import People, content_tokens

ASR_LATENCY = 1.5  # seconds from end of a chunk to transcript text
DEADLINE_LEAD = 2 * 3600
END_OF_DAY_CHECK = 18 * 3600  # "today" items get one nudge window from 6pm
TICK = 0.5


@dataclass
class Day:
    name: str
    date: str
    weekday: int
    people: dict
    segments: list[Segment]
    events: list[ContextEvent]
    truth: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Day":
        raw = json.loads(Path(path).read_text())
        segs = []
        for i, s in enumerate(raw["segments"]):
            start = parse_clock(s["at"])
            dur = s.get("dur") or max(1.2, len(s["text"].split()) / 2.6)
            segs.append(Segment(s.get("id", f"s{i + 1}"), start, start + dur, s["who"], s["text"]))
        evs = [ContextEvent(e.get("id", f"e{i + 1}"), parse_clock(e["at"]), e["kind"], e.get("data", {}))
               for i, e in enumerate(raw.get("events", []))]
        segs.sort(key=lambda s: s.start)
        evs.sort(key=lambda e: e.at)
        return cls(raw["name"], raw["date"], raw["weekday"], raw.get("people", {}), segs, evs, raw.get("truth", {}))


@dataclass
class RunResult:
    day: Day
    policy: Policy
    people: People
    ledger: list[LedgerItem]
    skipped: list[dict]
    facts: list
    withheld: list
    decisions: list[Decision]
    log: list[dict]

    def spoke(self) -> list[Decision]:
        return sorted([d for d in self.decisions if d.outcome == "spoke"], key=lambda d: d.at)

    def to_json(self) -> dict:
        def seg(s: Segment) -> dict:
            return {"id": s.id, "start": s.start, "end": s.end, "who": s.speaker,
                    "name": self.people.name(s.speaker) if s.speaker not in (WEARER, "media") else s.speaker,
                    "text": s.text, "clock": fmt_clock(s.start)}

        def dec(d: Decision) -> dict:
            c = d.candidate
            return {"id": c.id, "kind": c.kind, "outcome": d.outcome, "at": d.at, "clock": fmt_clock(d.at),
                    "created": fmt_clock(c.created_at), "said": d.said, "text": c.text, "duration": d.duration,
                    "priority": c.priority, "reasons": d.reasons, "ledger_id": c.ledger_id,
                    "people": [self.people.name(p) for p in c.related_people]}

        def led(i: LedgerItem) -> dict:
            o = asdict(i)
            o["owner_name"] = self.people.name(i.owner) if i.owner != WEARER else "you"
            o["counterparty_name"] = self.people.name(i.counterparty) if i.counterparty and i.counterparty != WEARER else (
                "you" if i.counterparty == WEARER else None)
            o["created_clock"] = fmt_clock(i.created_at)
            o["closed_clock"] = fmt_clock(i.closed_at) if i.closed_at else None
            o["display"] = display_what(i, self.people)
            return o

        return {
            "day": {"name": self.day.name, "date": self.day.date},
            "policy": asdict(self.policy),
            "segments": [seg(s) for s in self.day.segments],
            "events": [{"id": e.id, "at": e.at, "clock": fmt_clock(e.at), "kind": e.kind, "data": e.data} for e in self.day.events],
            "ledger": [led(i) for i in self.ledger],
            "skipped": self.skipped,
            "facts": [{**asdict(f), "about_name": self.people.name(f.about) if f.about in self.people.display or f.about == WEARER else f.about,
                       "clock": fmt_clock(f.created_at)} for f in self.facts],
            "withheld": [{**asdict(w), "clock": fmt_clock(w.at)} for w in self.withheld],
            "decisions": [dec(d) for d in sorted(self.decisions, key=lambda d: d.at)],
        }


class CandidateMaker:
    """Turns ledger state + context into things Samantha *could* say."""

    def __init__(self, people: People, policy: Policy):
        self.people = people
        self.policy = policy
        self._contextual_done: set[tuple[str, str]] = set()
        self._deadline_done: set[str] = set()

    def contextual(self, t: float, present: list[str], open_items: list[LedgerItem], arrived: dict[str, float]) -> list[Candidate]:
        out = []
        for item in open_items:
            if item.kind == "idea":
                continue
            other = item.counterparty if item.owner == WEARER else item.owner
            if other not in present or other == WEARER:
                continue
            # don't remind you of a promise you made in this same conversation
            if item.created_at >= arrived.get(other, t) - 1:
                continue
            key = (item.id, other)
            if key in self._contextual_done:
                continue
            self._contextual_done.add(key)
            name = self.people.name(other)
            what = display_what(item, self.people)
            due = f", {nice_due(item.due_text)}" if item.due_text else ""
            if item.owner == WEARER:
                text = f"{name}'s here. Still open from you: {what}{due}."
                short = f"{name}'s here. Still open: {what}{due}."
                pri = 0.75
            else:
                # only chase other people once they're late, otherwise it's nagging
                if item.due is None or t < item.due:
                    continue
                text = f"{name}'s here. They said they'd {what}{due}, and it's overdue. Worth a quick ask?"
                short = f"{name} still owes you: {what}."
                pri = 0.5
            out.append(Candidate(f"ctx-{item.id}-{other}", "contextual_loop", text, short, t,
                                 t + self.policy.presence_window, pri, (other,), item.id,
                                 [f"{fmt_clock(t)} created: {name} started talking and {item.id} is open with them"]))
        return out

    def deadlines(self, t: float, open_items: list[LedgerItem]) -> list[Candidate]:
        out = []
        for item in open_items:
            if item.owner != WEARER or item.kind != "commitment" or item.due is None:
                continue
            if item.id in self._deadline_done or item.due > 86400 + 3600 * 6:
                continue
            start = END_OF_DAY_CHECK if item.due_text in ("today", "end of day", "end of the day") else item.due - DEADLINE_LEAD
            if t < start or t >= item.due:
                continue
            self._deadline_done.add(item.id)
            what = display_what(item, self.people)
            cp = self.people.name(item.counterparty) if item.counterparty else None
            told = f"told {cp}" if cp and cp not in what else "said"
            text = f"Heads up: you {told} you'd {what}, {nice_due(item.due_text)}. Still open."
            short = f"Still open: {what}."
            out.append(Candidate(f"due-{item.id}", "deadline", text, short, t, item.due, 0.65,
                                 (item.counterparty,) if item.counterparty else (), item.id,
                                 [f"{fmt_clock(t)} created: due {item.due_text} and not closed"]))
        return out

    def news(self, ev: ContextEvent, ledger: LedgerExtractor) -> Optional[Candidate]:
        if ev.kind == "calendar_change":
            text = ev.data.get("say") or f"{ev.data.get('title', 'A meeting')} moved to {ev.data.get('to', 'later')}."
            return Candidate(f"news-{ev.id}", "news", text, text, ev.at, ev.at + 2 * 3600, 0.6, (), None,
                             [f"{fmt_clock(ev.at)} created: calendar change"])
        if ev.kind == "email_in":
            sender = ev.data.get("from")
            awaited = ev.data.get("awaited") or any(i.owner == sender for i in ledger.items if i.kind == "waiting_on")
            if not awaited:
                return None
            text = ev.data.get("say") or f"{self.people.name(sender)} replied: {ev.data.get('subject', '')}."
            return Candidate(f"news-{ev.id}", "news", text, text, ev.at, ev.at + 2 * 3600, 0.7,
                             (sender,) if sender else (), None,
                             [f"{fmt_clock(ev.at)} created: a reply you were waiting on"])
        return None

    def idea(self, seg: Segment, t: float, ledger: LedgerExtractor) -> Optional[Candidate]:
        toks = content_tokens(seg.text)
        for item in ledger.items:
            if item.kind != "idea" or item.source_segment == seg.id or t - item.created_at < 1800:
                continue
            if len(toks & content_tokens(item.what)) >= 2:
                cid = f"idea-{item.id}-{seg.id}"
                gist = " ".join(item.what.split()[:12]) + ("..." if len(item.what.split()) > 12 else "")
                text = f"You had a version of this at {fmt_clock(item.created_at)[:5]}: \"{gist}\" Want to pick it up?"
                return Candidate(cid, "idea_resurface", text, "Echoes your earlier idea; want it?", t, t + 600, 0.4,
                                 (), item.id, [f"{fmt_clock(t)} created: you're circling back to {item.id}"])
        return None


def run_day(day: Day, policy: Optional[Policy] = None, extractor_factory=None) -> RunResult:
    policy = policy or Policy()
    people = People.from_config(day.people)
    ledger = (extractor_factory or LedgerExtractor)(people=people, weekday=day.weekday)
    mind = MindBuilder(people=people, guard=not policy.naive)
    hearing = Hearing(day.segments)
    arb = Arbiter(policy=policy, people=people, hearing=hearing)
    maker = CandidateMaker(people, policy)
    log: list[dict] = []

    heard = sorted(((s.end + ASR_LATENCY, "seg", s) for s in day.segments), key=lambda x: x[0])
    evs = [(e.at, "ev", e) for e in day.events]
    queue = sorted(heard + evs, key=lambda x: x[0])
    qi = 0
    start = min(day.segments[0].start, day.events[0].at if day.events else 1e9) - 5
    end = max(max(s.end for s in day.segments), max((e.at for e in day.events), default=0)) + 1800
    arrived: dict[str, float] = {}
    presence = policy.presence_window

    t = start
    while t <= end:
        while qi < len(queue) and queue[qi][0] <= t:
            _, typ, obj = queue[qi]
            qi += 1
            if typ == "seg":
                s: Segment = obj
                for change, item in ledger.observe(s, t):
                    log.append({"at": t, "what": f"ledger {change}", "id": item.id})
                for f in mind.observe(s, t):
                    log.append({"at": t, "what": "mind fact", "id": f.id})
                if s.is_wearer:
                    c = maker.idea(s, t, ledger)
                    if c:
                        arb.add(c)
            else:
                e: ContextEvent = obj
                if e.kind == "focus_on":
                    arb.focus = True
                elif e.kind == "focus_off":
                    arb.focus = False
                elif e.kind == "pause_listening":
                    arb.listening = False
                elif e.kind == "resume_listening":
                    arb.listening = True
                for change, item in ledger.observe_event(e):
                    log.append({"at": t, "what": f"ledger {change}", "id": item.id})
                c = maker.news(e, ledger)
                if c:
                    arb.add(c)

        # presence bookkeeping: when did each person (re)arrive?
        present = hearing.people_since(t, presence)
        for p in present:
            if p not in arrived:
                arrived[p] = t
        for p in list(arrived):
            if p not in present:
                del arrived[p]

        # candidates need transcripts, so they only know about items already in the ledger
        for c in maker.contextual(t, [p for p in present if t - arrived[p] >= 0], ledger.open_items(), arrived):
            arb.add(c)
        for c in maker.deadlines(t, ledger.open_items()):
            arb.add(c)
        # a contextual reminder about something already closed is moot
        for c in list(arb.pending):
            if c.ledger_id and any(i.id == c.ledger_id and i.status != "open" for i in ledger.items):
                arb.pending.remove(c)
                arb.decisions.append(Decision(c, "dropped", t, reasons=c.reasons + ["loop closed before it was said"]))
        arb.tick(t)
        t += TICK

    arb.flush(t)
    return RunResult(day, policy, people, ledger.items, ledger.skipped, mind.facts, mind.withheld, arb.decisions, log)

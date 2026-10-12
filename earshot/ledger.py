"""The ledger: open loops extracted from what was actually said.

Design rule that matters most for an always-listening wearable:
**only the wearer's own words (or the wearer's yes to a request) create a
commitment for the wearer.** Everyone else's "I'll do it" is a waiting-on, and
media audio (TV, podcasts) creates nothing.

This is a rule-based extractor so the eval is deterministic. `llm.py` has a
drop-in model-backed extractor with the same output contract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .models import MEDIA, WEARER, ContextEvent, LedgerItem, Segment
from .text import PAST_FORMS, People, content_tokens, normalize, sentences, verb_family
from .timeparse import parse_due

COMMIT = re.compile(
    r"\b(?:i'll|i will|i'm gonna|i am gonna|i'm going to|i am going to|let me|i promise to|i'll make sure to|i'll try to)\s+(?P<rest>.+)",
    re.I,
)
REQUEST = re.compile(
    r"\b(?:can you|could you|would you|will you|can u|please|don't forget to|remember to|you need to|mind if you)\s+(?P<rest>.+)",
    re.I,
)
_LEAD = r"^\W*(?:(?:oh|ah|um|uh|sorry|right|hey|ha|okay so)\W+){0,3}"
AFFIRM = re.compile(_LEAD + r"(?:yeah|yes|yep|yup|sure|ok|okay|will do|of course|on it|no problem|absolutely|got it|sounds good|definitely|for sure|totally)\b", re.I)
DECLINE = re.compile(_LEAD + r"(?:no\b|nope|nah|i can't|i cannot|i won't|not today|ugh,? no)", re.I)
HEDGE = re.compile(r"\b(?:maybe|probably|might|someday|one day|at some point|i should|i ought|would be nice|if i have time|i wish|i guess i could|in theory)\b", re.I)
NEGATED = re.compile(r"\b(?:i won't|i will not|i can't|i cannot|i'm not going to|i am not going to|no way i'll)\b", re.I)
SARCASM = re.compile(r"\b(?:yeah right|as if|sure,? because|oh sure,? i'll)\b", re.I)
REPORTED = re.compile(
    r"\b(?P<who>[a-z]+)\s+(?:said|says|told me|promised|mentioned)\s+(?:that\s+)?(?:he|she|they)(?:'d|'ll| would| will)\s+(?P<rest>.+)",
    re.I,
)
IDIOM_DONE = re.compile(r"\b(?:got it|got you|gotcha|got that right|looked fine|checked out)\b", re.I)
AMBIGUOUS_PAST = {"got", "looked", "checked", "dropped"}
FILLER = re.compile(r"\blet me (?:think|see|know|guess|check real quick|be honest|tell you|just say)\b", re.I)
IDEA = re.compile(r"\b(?:idea|what if|i wonder if|imagine if|we could)\b", re.I)
NOT_DONE = re.compile(r"\b(?:haven't|have not|didn't|did not|not yet|still need|still have to|forgot to|need to)\b", re.I)
RESCHEDULE = re.compile(r"\b(?:actually|instead|push(?:ed)? (?:it|that)|move(?:d)? (?:it|that))\b", re.I)

ADDRESSEE_WINDOW = 120.0
REQUEST_WINDOW = 25.0
IDEA_SOLO_WINDOW = 120.0


def _clean_what(rest: str, due_text: Optional[str]) -> str:
    what = rest
    if due_text:
        what = re.sub(rf"\s*\b(?:by|on|before|until|at|for|to)?\s*{re.escape(due_text)}\b", "", what, flags=re.I)
    # drop trailing asides: ", I just need the numbers" / ", it's a token thing" / ", Mira's coming"
    what = re.split(r"\s+and\s+(?:we|then|so)\b|,\s*(?:see you|thanks|cheers)\b", what, maxsplit=1, flags=re.I)[0]
    what = re.split(r",\s*(?:i|it's|it is|because|since|but|so|we|he|she|they|that's|[a-z]+'s)\b", what, maxsplit=1, flags=re.I)[0]
    what = re.sub(r"\b(?:okay|ok|alright|right|yeah)\??$", "", what.strip(), flags=re.I)
    return what.strip(" ,.!?;-")


def _wearer_view(what: str, speaker_is_wearer: bool, other_name: Optional[str]) -> str:
    """Rewrite a clause so it reads from the wearer's point of view.
    Wearer: "send you the deck" (to Priya) -> "send Priya the deck".
    Someone else: "send me that podcast" (Sam) -> "send Sam that podcast"; "my" -> "their"."""
    if speaker_is_wearer:
        if other_name:
            what = re.sub(r"\byour\b", f"{other_name}'s", what, flags=re.I)
            what = re.sub(r"\byou\b", other_name, what, flags=re.I)
        return what
    if other_name:
        what = re.sub(r"\bme\b", other_name, what, flags=re.I)
        what = re.sub(r"\bmy\b", f"{other_name}'s", what, flags=re.I)
    return what


def _first_verb(what: str) -> Optional[str]:
    m = re.match(r"\s*(?:also\s+|just\s+|definitely\s+|totally\s+|go\s+)?([a-z]+)", what.lower())
    return m.group(1) if m else None


@dataclass
class PendingRequest:
    requester: str
    what: str
    due: Optional[float]
    due_text: Optional[str]
    at: float
    segment_id: str


@dataclass
class LedgerExtractor:
    people: People
    weekday: int
    items: list[LedgerItem] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    _recent: list[Segment] = field(default_factory=list)
    _pending: list[PendingRequest] = field(default_factory=list)
    _n: int = 0

    # ---------- public ----------
    def open_items(self, kind: str | None = None) -> list[LedgerItem]:
        return [i for i in self.items if i.status == "open" and (kind is None or i.kind == kind)]

    def observe(self, seg: Segment, now: float) -> list[tuple[str, LedgerItem]]:
        """Returns [(change, item)] where change is created|updated|closed."""
        changes: list[tuple[str, LedgerItem]] = []
        self._recent = [s for s in self._recent if now - s.end <= ADDRESSEE_WINDOW]
        if seg.speaker == MEDIA:
            if COMMIT.search(seg.text) or REQUEST.search(seg.text):
                self._skip(seg, "media audio (not a person in the room)")
            self._recent.append(seg)
            return changes

        if seg.is_wearer:
            changes += self._wearer(seg, now)
        else:
            changes += self._other(seg, now)
        self._recent.append(seg)
        return changes

    def observe_event(self, ev: ContextEvent) -> list[tuple[str, LedgerItem]]:
        """Cross-modal closure: an email the wearer sent can close a 'send' loop."""
        changes = []
        if ev.kind == "email_sent":
            to = ev.data.get("to")
            subj = content_tokens(ev.data.get("subject", ""))
            for item in self.open_items("commitment"):
                if verb_family(item.verb) != "send":
                    continue
                if (to and item.counterparty == to) or (subj & content_tokens(item.what)):
                    self._close(item, ev.at, f"email to {self.people.name(to)}: '{ev.data.get('subject', '')}'")
                    changes.append(("closed", item))
        elif ev.kind == "email_in":
            sender = ev.data.get("from")
            for item in self.open_items("waiting_on"):
                if item.owner == sender:
                    self._close(item, ev.at, f"email from {self.people.name(sender)}")
                    changes.append(("closed", item))
        return changes

    # ---------- wearer ----------
    def _wearer(self, seg: Segment, now: float) -> list[tuple[str, LedgerItem]]:
        changes: list[tuple[str, LedgerItem]] = []
        text = normalize(seg.text)

        # 1) a yes to a recent request from someone else
        self._pending = [p for p in self._pending if now - p.at <= REQUEST_WINDOW]
        if self._pending and AFFIRM.search(text) and not DECLINE.search(text):
            req = self._pending.pop()
            own = COMMIT.search(text)
            due, due_text = parse_due(text, now, self.weekday) if own else (None, None)
            item = self._add_or_update(
                kind="commitment", owner=WEARER, counterparty=req.requester,
                what=_wearer_view(req.what, False, self.people.name(req.requester)),
                due=due if due is not None else req.due, due_text=due_text or req.due_text,
                seg=seg, now=now, how=f"said yes to {self.people.name(req.requester)}'s request",
            )
            changes.append(item)
            if own:  # "Sure, I'll do it by 5" -> already captured
                return changes
        elif self._pending and DECLINE.search(text):
            req = self._pending.pop()
            self._skip(seg, f"declined {self.people.name(req.requester)}'s request to {req.what}")

        for sent in sentences(text):
            # 2) closures: "sent Priya the deck"
            closed = self._maybe_close(sent, seg, now, owner=WEARER)
            changes += closed
            if closed:
                continue
            # 3) ideas, only when thinking alone
            if IDEA.search(sent) and not self._others_recent(now, IDEA_SOLO_WINDOW):
                idea_text = re.sub(r"^\W*(?:okay|ok|so|hmm|alright)?\W*(?:new\s+)?idea\W*", "", sent, flags=re.I).strip(" .")
                item = self._new("idea", WEARER, None, idea_text, None, None, seg, now, 0.7,
                                 "thinking aloud, alone")
                changes.append(("created", item))
                continue
            # 4) commitments
            m = COMMIT.search(sent)
            if not m:
                rm = REPORTED.search(sent)
                if rm:
                    changes += self._reported(rm, seg, now)
                continue
            reason = self._reject_reason(sent)
            if reason:
                self._skip(seg, reason, sent)
                continue
            due, due_text = parse_due(sent, now, self.weekday)
            what = _clean_what(m.group("rest"), due_text)
            if len(content_tokens(what)) == 0 and not self._recent_request_object(now):
                self._skip(seg, "no concrete object", sent)
                continue
            counterparty = self._counterparty(sent, now)
            if counterparty:
                what = _wearer_view(what, True, self.people.name(counterparty))
            changes.append(
                self._add_or_update("commitment", WEARER, counterparty, what, due, due_text, seg, now,
                                    how="said it", rescheduling=bool(RESCHEDULE.search(sent)))
            )
        return changes

    def _reject_reason(self, sent: str) -> Optional[str]:
        if FILLER.search(sent):
            return "conversational filler ('let me think / let me know')"
        if NEGATED.search(sent):
            return "negated ('I won't / I can't')"
        if SARCASM.search(sent):
            return "sarcasm marker"
        if re.match(r"\s*if\b", sent, re.I):
            return "conditional ('if ..., I'll')"
        if HEDGE.search(sent):
            return "hedged ('maybe / probably / someday')"
        if sent.strip().endswith("?"):
            return "question, not a promise"
        return None

    # ---------- others ----------
    def _other(self, seg: Segment, now: float) -> list[tuple[str, LedgerItem]]:
        changes: list[tuple[str, LedgerItem]] = []
        text = normalize(seg.text)
        for sent in sentences(text):
            changes += self._maybe_close(sent, seg, now, owner=seg.speaker)
            rq = REQUEST.search(sent)
            if rq and not sent.lower().startswith(("i ", "we ")):
                due, due_text = parse_due(sent, now, self.weekday)
                what = _clean_what(rq.group("rest"), due_text)
                self._pending.append(PendingRequest(seg.speaker, what, due, due_text, now, seg.id))
                continue
            m = COMMIT.search(sent)
            if m and not self._reject_reason(sent):
                due, due_text = parse_due(sent, now, self.weekday)
                what = re.sub(r"\bmy\b", "their", _clean_what(m.group("rest"), due_text), flags=re.I)
                if content_tokens(what):
                    changes.append(
                        self._add_or_update("waiting_on", seg.speaker, WEARER, what, due, due_text, seg, now,
                                            how=f"{self.people.name(seg.speaker)} said it")
                    )
                    self._skip(seg, f"not the wearer's promise ({self.people.name(seg.speaker)} said it) -> waiting-on", sent)
        return changes

    def _reported(self, m: re.Match, seg: Segment, now: float) -> list[tuple[str, LedgerItem]]:
        who = self.people.aliases.get(m.group("who").lower())
        if not who:
            return []
        due, due_text = parse_due(m.group("rest"), now, self.weekday)
        what = _clean_what(m.group("rest"), due_text)
        return [self._add_or_update("waiting_on", who, WEARER, what, due, due_text, seg, now,
                                    how="you said they promised")]

    # ---------- closure ----------
    def _maybe_close(self, sent: str, seg: Segment, now: float, owner: str) -> list[tuple[str, LedgerItem]]:
        if IDIOM_DONE.search(sent):
            return []
        words = re.findall(r"[a-z]+", sent.lower())
        past = [w for w in words if w in PAST_FORMS]
        if not past or NOT_DONE.search(sent) or COMMIT.search(sent) or sent.strip().endswith("?"):
            return []
        fams = {verb_family(w) for w in past}
        toks = content_tokens(sent)
        named = set(self.people.mentioned(sent))
        best, best_score = None, 0.0
        for item in self.open_items():
            if item.owner != owner or item.kind == "idea":
                continue
            if verb_family(item.verb) not in fams:
                continue
            overlap = len(toks & content_tokens(item.what))
            score = overlap + (1.5 if item.counterparty in named else 0)
            if score > best_score:
                best, best_score = item, score
        if best and best_score >= 1:
            self._close(best, now, f"heard: \"{sent.strip()}\"")
            return [("closed", best)]
        # "I paid it." -> close only if exactly one open loop of that kind exists
        fams_strict = {verb_family(w) for w in past if w not in AMBIGUOUS_PAST}
        same = [i for i in self.open_items() if i.owner == owner and i.kind != "idea" and verb_family(i.verb) in fams_strict]
        if len(same) == 1 and re.search(r"\b(?:it|that|them|those)\b", sent, re.I):
            self._close(same[0], now, f"heard: \"{sent.strip()}\" (only open loop of that kind)")
            same[0].confidence = min(same[0].confidence, 0.6)
            return [("closed", same[0])]
        return []

    def _close(self, item: LedgerItem, at: float, why: str) -> None:
        item.status = "closed"
        item.closed_at = at
        item.closed_by = why
        item.history.append(f"closed: {why}")

    # ---------- helpers ----------
    def _others_recent(self, now: float, window: float) -> bool:
        return any(s.is_person and now - s.end <= window for s in self._recent)

    def _recent_request_object(self, now: float) -> bool:
        return any(now - p.at <= REQUEST_WINDOW for p in self._pending)

    def _counterparty(self, sent: str, now: float) -> Optional[str]:
        named = [p for p in self.people.mentioned(sent) if p != WEARER]
        if named:
            return named[0]
        if re.search(r"\b(?:you|your|ya)\b", sent, re.I):
            people = [s.speaker for s in reversed(self._recent) if s.is_person and now - s.end <= ADDRESSEE_WINDOW]
            if people:
                return people[0]
        return None

    def _similar(self, a: LedgerItem, owner: str, counterparty: Optional[str], what: str) -> bool:
        if a.owner != owner or a.status != "open" or a.kind == "idea":
            return False
        ta, tb = content_tokens(a.what), content_tokens(what)
        if not ta or not tb:
            return False
        jac = len(ta & tb) / len(ta | tb)
        same_cp = counterparty is None or a.counterparty in (None, counterparty)
        same_verb = verb_family(a.verb) == verb_family(_first_verb(what))
        return same_cp and (jac >= 0.5 or (same_verb and jac >= 0.25))

    def _add_or_update(self, kind, owner, counterparty, what, due, due_text, seg, now, how,
                       rescheduling: bool = False) -> tuple[str, LedgerItem]:
        for existing in self.open_items(kind):
            if self._similar(existing, owner, counterparty, what):
                if due is not None and due != existing.due:
                    existing.history.append(f"due moved {existing.due_text!r} -> {due_text!r} ({how})")
                    existing.due, existing.due_text = due, due_text
                if counterparty and not existing.counterparty:
                    existing.counterparty = counterparty
                return ("updated", existing)
        conf = 0.6 if re.search(r"\btry\b", what, re.I) else 0.85
        return ("created", self._new(kind, owner, counterparty, what, due, due_text, seg, now, conf, how))

    def _new(self, kind, owner, counterparty, what, due, due_text, seg, now, conf, how=None) -> LedgerItem:
        self._n += 1
        item = LedgerItem(
            id=f"L{self._n}", kind=kind, owner=owner, counterparty=counterparty, what=what,
            verb=_first_verb(what), due=due, due_text=due_text, source_segment=seg.id,
            created_at=now, confidence=conf,
        )
        item.history.append(f"created: {how or 'heard it'}")
        self.items.append(item)
        return item

    def _skip(self, seg: Segment, reason: str, sent: Optional[str] = None) -> None:
        self.skipped.append({"segment": seg.id, "speaker": seg.speaker, "text": sent or seg.text, "reason": reason})

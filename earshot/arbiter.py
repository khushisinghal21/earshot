"""Speak-or-hold: deciding *when* an AI in your ear is allowed to talk.

The arbiter is strictly causal. At time t it may use only:
  * voice activity up to t (is anyone talking right now, how long has it been quiet),
  * transcripts that have already come back from ASR,
  * context events that have already happened (Focus, calendar, mail).
It never peeks at when the next person will start talking; collisions are
measured afterwards by the eval, the way they would happen in real life.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .models import WEARER, Candidate, Decision, LedgerItem, Segment
from .text import People

WORDS_PER_SEC = 2.6
FAREWELL = re.compile(r"\b(?:see you|see ya|bye|gotta run|got to run|have to run|talk later|catch you later|heading out|good night)\b", re.I)


@dataclass
class Policy:
    name: str = "earshot"
    solo_pause: float = 4.0  # quiet needed when the wearer is alone (thinking-aloud has pauses)
    conv_pause: float = 2.5  # quiet needed mid-conversation: a lull, not a turn gap
    conv_window: float = 60.0  # someone else spoke within this -> you're in a conversation
    presence_window: float = 300.0  # someone counts as "here" for 5 min after they last spoke
    cooldown: float = 8 * 60  # min gap between non-contextual whispers
    hourly_budget: int = 4
    focus_threshold: float = 0.9  # only this priority breaks Focus
    max_conv_words: int = 14
    backchannel_pause: float = 5.0  # after your own "nice"/"yeah", the other person usually keeps going
    farewell_pause: float = 1.5  # "see you" / "gotta run" is the best moment there is
    naive: bool = False  # baseline: speak the moment there is something to say


def say_duration(text: str) -> float:
    return max(1.2, len(text.split()) / WORDS_PER_SEC)


def display_what(item: LedgerItem, people: People) -> str:
    """Ledger text is already stored from the wearer's point of view."""
    return item.what


def nice_due(due_text: str | None) -> str:
    if not due_text:
        return ""
    return re.sub(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                  lambda m: m.group(1).title(), due_text)


def nice_quiet(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    return f"{m // 60}h {m % 60}m" if m >= 60 else f"{m}m"


@dataclass
class Hearing:
    """What the arbiter knows about the audio scene at time t (causal)."""

    segments: list[Segment]
    _idx_end: int = 0

    def speaking(self, t: float) -> Optional[Segment]:
        # Voice activity detection is real-time, so "someone is talking now" is known at t.
        for s in self.segments:
            if s.start <= t < s.end:
                return s
            if s.start > t:
                break
        return None

    def last_end(self, t: float) -> Optional[Segment]:
        last = None
        for s in self.segments:
            if s.end <= t:
                if last is None or s.end > last.end:
                    last = s
            elif s.start > t:
                break
        return last

    def people_since(self, t: float, window: float) -> list[str]:
        out: list[str] = []
        for s in self.segments:
            if s.start > t:
                break
            if s.is_person and (s.end >= t - window):
                if s.speaker not in out:
                    out.append(s.speaker)
        return out


@dataclass
class Arbiter:
    policy: Policy
    people: People
    hearing: Hearing
    pending: list[Candidate] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    focus: bool = False
    listening: bool = True
    _spoken_at: list[float] = field(default_factory=list)
    _busy_until: float = -1.0

    def add(self, cand: Candidate) -> None:
        if any(c.id == cand.id for c in self.pending) or any(d.candidate.id == cand.id for d in self.decisions):
            return
        self.pending.append(cand)

    def _note(self, cand: Candidate, t: float, reason: str) -> None:
        stamp = f"{reason}"
        if not cand.reasons or not cand.reasons[-1].endswith(stamp):
            from .models import fmt_clock

            cand.reasons.append(f"{fmt_clock(t)} hold: {stamp}")

    def tick(self, t: float) -> Optional[Decision]:
        if not self.pending:
            return None
        # expire first
        for cand in list(self.pending):
            if t >= cand.expires_at:
                self.pending.remove(cand)
                if cand.kind == "contextual_loop":
                    outcome, why = "dropped", "the person left before a good moment; the deadline reminder still stands"
                else:
                    outcome, why = "digest", "no good moment before it went stale; moved to the next brief"
                self.decisions.append(Decision(cand, outcome, t, reasons=cand.reasons + [why]))

        if t < self._busy_until or not self.pending:
            return None

        speaking = self.hearing.speaking(t)
        last = self.hearing.last_end(t)
        quiet = (t - last.end) if (last and not speaking) else (0.0 if speaking else 1e9)
        present = self.hearing.people_since(t, self.policy.presence_window)
        in_conv = bool(self.hearing.people_since(t, self.policy.conv_window))

        for cand in sorted(self.pending, key=lambda c: -c.priority):
            if self.policy.naive:
                return self._speak(cand, t, cand.text, ["naive: spoke as soon as it had something"])

            if not self.listening:
                self._note(cand, t, "listening paused by the wearer")
                continue
            if self.focus and cand.priority < self.policy.focus_threshold:
                self._note(cand, t, "Focus is on")
                continue
            if speaking:
                who = "you are" if speaking.speaker == WEARER else f"{self.people.name(speaking.speaker)} is"
                self._note(cand, t, f"{who} talking")
                continue

            about_here = bool(set(cand.related_people) & set(present))
            if in_conv:
                if not (about_here or cand.priority >= 0.85):
                    names = ", ".join(self.people.name(p) for p in present) or "someone"
                    self._note(cand, t, f"in a conversation with {names} and this isn't about them")
                    continue
                need = self.policy.conv_pause
                cue = "lull"
                if last is not None and last.speaker == WEARER and len(last.text.split()) <= 3:
                    need, cue = self.policy.backchannel_pause, "lull after your short reply"
                if last is not None and last.is_person and FAREWELL.search(last.text):
                    need, cue = self.policy.farewell_pause, f"{self.people.name(last.speaker)} is wrapping up"
                if quiet < need:
                    continue  # waiting for a lull; too frequent to log
                text = cand.short_text
                if len(text.split()) > self.policy.max_conv_words:
                    text = " ".join(text.split()[: self.policy.max_conv_words])
                why = [f"{cue}: {quiet:.1f}s of quiet", "about someone here" if about_here else "high priority"]
            else:
                if quiet < self.policy.solo_pause:
                    continue
                text = cand.text
                why = [f"quiet for {nice_quiet(quiet)}, nobody else around"]

            recent = [s for s in self._spoken_at if t - s < 3600]
            if len(recent) >= self.policy.hourly_budget and cand.priority < 0.85:
                self._note(cand, t, f"hourly budget used ({self.policy.hourly_budget} whispers)")
                continue
            if self._spoken_at and t - self._spoken_at[-1] < self.policy.cooldown and not about_here and cand.priority < 0.85:
                self._note(cand, t, "spoke recently; cooling down")
                continue
            return self._speak(cand, t, text, why)
        return None

    def _speak(self, cand: Candidate, t: float, text: str, why: list[str]) -> Decision:
        self.pending.remove(cand)
        dur = say_duration(text)
        d = Decision(cand, "spoke", t, said=text, duration=dur, reasons=cand.reasons + why)
        self.decisions.append(d)
        self._spoken_at.append(t)
        self._busy_until = t + dur + 1.0
        return d

    def flush(self, t: float) -> None:
        for cand in list(self.pending):
            self.pending.remove(cand)
            outcome = "dropped" if cand.kind == "contextual_loop" else "digest"
            self.decisions.append(Decision(cand, outcome, t, reasons=cand.reasons + ["end of replay"]))

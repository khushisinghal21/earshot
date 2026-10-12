"""Core data types for Earshot.

Everything is replayable: a "day" is a list of transcript segments plus context
events (calendar change, Focus on/off, email in/out). Time is seconds since
local midnight so a whole day is easy to reason about and to diff.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

WEARER = "me"
MEDIA = "media"  # TV, podcast, video call on speaker with no enrolled voice


def parse_clock(value: str | int | float) -> float:
    """'13:42:10' -> seconds since midnight. Numbers pass through."""
    if isinstance(value, (int, float)):
        return float(value)
    parts = [float(p) for p in value.split(":")]
    while len(parts) < 3:
        parts.append(0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def fmt_clock(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


@dataclass(frozen=True)
class Segment:
    """One transcribed chunk. `speaker` is what diarization + voiceprints said:
    'me' (the enrolled wearer), an enrolled contact id ('priya'), an anonymous
    id ('spk_3'), or 'media'."""

    id: str
    start: float
    end: float
    speaker: str
    text: str

    @property
    def is_wearer(self) -> bool:
        return self.speaker == WEARER

    @property
    def is_person(self) -> bool:
        return self.speaker not in (WEARER, MEDIA)


@dataclass(frozen=True)
class ContextEvent:
    """Non-audio signal the phone already has. kind is one of:
    focus_on, focus_off, pause_listening, resume_listening,
    calendar_change, email_in, email_sent."""

    id: str
    at: float
    kind: str
    data: dict = field(default_factory=dict)


LedgerKind = Literal["commitment", "waiting_on", "idea"]
LedgerStatus = Literal["open", "closed", "dropped"]


@dataclass
class LedgerItem:
    id: str
    kind: LedgerKind
    owner: str  # 'me' or a speaker/person id
    counterparty: Optional[str]
    what: str
    verb: Optional[str]
    due: Optional[float]  # seconds since midnight of the replay day (can exceed 86400)
    due_text: Optional[str]
    source_segment: str
    created_at: float
    status: LedgerStatus = "open"
    closed_at: Optional[float] = None
    closed_by: Optional[str] = None
    confidence: float = 0.8
    history: list[str] = field(default_factory=list)


@dataclass
class MindFact:
    id: str
    about: str
    fact: str
    source_segment: str
    created_at: float
    needs_confirm: bool = False


@dataclass
class Withheld:
    """Something Earshot heard and deliberately did not keep."""

    source_segment: str
    at: float
    about: str
    reason: str


NudgeKind = Literal["contextual_loop", "deadline", "news", "idea_resurface"]
Outcome = Literal["spoke", "digest", "dropped"]


@dataclass
class Candidate:
    id: str
    kind: NudgeKind
    text: str
    short_text: str
    created_at: float
    expires_at: float
    priority: float  # 0..1
    related_people: tuple[str, ...] = ()
    ledger_id: Optional[str] = None
    reasons: list[str] = field(default_factory=list)


@dataclass
class Decision:
    candidate: Candidate
    outcome: Outcome
    at: float
    said: Optional[str] = None
    duration: float = 0.0
    reasons: list[str] = field(default_factory=list)

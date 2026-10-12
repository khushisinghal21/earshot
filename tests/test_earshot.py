import json
from pathlib import Path

import pytest

from earshot.arbiter import Arbiter, Hearing, Policy
from earshot.engine import Day, run_day
from earshot.evaluate import NaiveExtractor, score
from earshot.ledger import LedgerExtractor
from earshot.llm import llm_factory
from earshot.mind import MindBuilder
from earshot.models import Candidate, ContextEvent, Segment, parse_clock
from earshot.text import People
from earshot.timeparse import DAY, END_OF_DAY, parse_due

DAYS = Path(__file__).resolve().parent.parent / "data" / "days"
PEOPLE = People.from_config({"priya": {"name": "Priya"}, "dev": {"name": "Dev"}, "jordan": {"name": "Jordan"}})
TUESDAY = 1


def seg(i, at, who, text):
    t = parse_clock(at)
    return Segment(f"s{i}", t, t + max(1.2, len(text.split()) / 2.6), who, text)


def feed(ex, segs):
    for s in segs:
        ex.observe(s, s.end + 1.5)
    return ex


# ---------- deadlines ----------
@pytest.mark.parametrize("text,expected", [
    ("I'll do it today", END_OF_DAY),
    ("I'll send it tonight", 21 * 3600),
    ("by Friday", 3 * DAY + END_OF_DAY),          # Tuesday -> Friday
    ("before lunch", 12 * 3600),
    ("by 3pm", 15 * 3600),
    ("tomorrow morning", DAY + 11 * 3600),
])
def test_parse_due(text, expected):
    due, _ = parse_due(text, now=9 * 3600, weekday=TUESDAY)
    assert due == expected


def test_in_an_hour_is_relative():
    due, _ = parse_due("I'll call in an hour", now=10 * 3600, weekday=TUESDAY)
    assert due == 11 * 3600


# ---------- whose promise is it ----------
def test_other_persons_promise_is_waiting_on_not_mine():
    ex = feed(LedgerExtractor(PEOPLE, TUESDAY), [seg(1, "09:00", "dev", "I'll fix the login bug before lunch.")])
    assert [(i.kind, i.owner) for i in ex.items] == [("waiting_on", "dev")]


def test_media_creates_nothing():
    ex = feed(LedgerExtractor(PEOPLE, TUESDAY), [seg(1, "09:00", "media", "I'll show you three ways to save.")])
    assert ex.items == []


def test_yes_to_a_request_is_my_commitment_with_their_name():
    ex = feed(LedgerExtractor(PEOPLE, TUESDAY), [
        seg(1, "08:00:00", "jordan", "Can you pick up oat milk tonight?"),
        seg(2, "08:00:05", "me", "Oh, yeah, sure."),
    ])
    (item,) = ex.items
    assert (item.kind, item.owner, item.counterparty, item.due_text) == ("commitment", "me", "jordan", "tonight")


def test_declined_request_is_not_a_commitment():
    ex = feed(LedgerExtractor(PEOPLE, TUESDAY), [
        seg(1, "08:00:00", "jordan", "Can you call the landlord?"),
        seg(2, "08:00:04", "me", "No, I can't today."),
    ])
    assert ex.items == []


@pytest.mark.parametrize("text", [
    "I should probably start running again at some point.",
    "If it rains I'll drive you to the station.",
    "Oh sure, I'll get right on cleaning the garage.",
    "I won't make it to yoga tonight.",
    "Let me think.",
    "Should I book the flights tonight?",
])
def test_not_a_promise(text):
    ex = feed(LedgerExtractor(PEOPLE, TUESDAY), [seg(1, "10:00", "me", text)])
    assert [i for i in ex.items if i.kind == "commitment"] == []


def test_you_is_resolved_to_the_person_you_are_talking_to():
    ex = feed(LedgerExtractor(PEOPLE, TUESDAY), [
        seg(1, "09:00:00", "priya", "And the investor deck?"),
        seg(2, "09:00:04", "me", "I'll send you the pitch deck by Friday."),
    ])
    (item,) = ex.items
    assert item.counterparty == "priya" and "Priya" in item.what


# ---------- closing loops ----------
def test_got_it_does_not_close_anything():
    ex = feed(LedgerExtractor(PEOPLE, TUESDAY), [
        seg(1, "08:00", "me", "I'll pick up oat milk tonight."),
        seg(2, "09:00", "me", "Got it."),
    ])
    assert ex.items[0].status == "open"


def test_not_yet_does_not_close():
    ex = feed(LedgerExtractor(PEOPLE, TUESDAY), [
        seg(1, "08:00", "me", "I'll water the plants."),
        seg(2, "18:00", "me", "I didn't water the plants yet."),
    ])
    assert ex.items[0].status == "open"


def test_email_sent_closes_a_send_loop():
    ex = feed(LedgerExtractor(PEOPLE, TUESDAY), [
        seg(1, "09:00:00", "priya", "And the deck?"),
        seg(2, "09:00:04", "me", "I'll send you the pitch deck by Friday."),
    ])
    ex.observe_event(ContextEvent("e1", parse_clock("16:00"), "email_sent", {"to": "priya", "subject": "Deck v7"}))
    assert ex.items[0].status == "closed"


# ---------- what gets remembered ----------
def test_bystander_health_is_heard_but_not_kept():
    mb = MindBuilder(PEOPLE)
    mb.observe(seg(1, "09:00", "dev", "My migraines have been brutal this week."), 0)
    assert mb.facts == [] and len(mb.withheld) == 1


def test_wearers_own_sensitive_fact_needs_confirmation():
    mb = MindBuilder(PEOPLE)
    mb.observe(seg(1, "09:00", "me", "My therapist thinks I should take more breaks."), 0)
    assert mb.facts and mb.facts[0].needs_confirm


def test_everyday_fact_is_kept():
    mb = MindBuilder(PEOPLE)
    mb.observe(seg(1, "09:00", "priya", "My daughter's recital is on Thursday."), 0)
    assert "recital" in mb.facts[0].fact


# ---------- when to speak ----------
def _arb(segs, **pol):
    return Arbiter(Policy(**pol), PEOPLE, Hearing(sorted(segs, key=lambda s: s.start)))


def _cand(t, people=(), pri=0.6):
    return Candidate("c1", "contextual_loop" if people else "news", "long text here", "short", t, t + 3600, pri, tuple(people))


def test_never_speaks_while_someone_is_talking():
    s = seg(1, "10:00:00", "priya", "this is a long sentence that keeps going and going for a while now")
    arb = _arb([s])
    arb.add(_cand(s.start))
    t = s.start
    while t < s.end:
        assert arb.tick(t) is None
        t += 0.5


def test_off_topic_news_waits_for_the_conversation_to_end():
    segs = [seg(i, f"10:00:{i * 6:02d}", "priya" if i % 2 else "me", "short line here") for i in range(1, 6)]
    arb = _arb(segs)
    arb.add(_cand(segs[0].start))
    t, spoke = segs[0].start, None
    while spoke is None and t < segs[-1].end + 200:
        spoke = arb.tick(t)
        t += 0.5
    last_person = max(s.end for s in segs if s.speaker == "priya")
    assert spoke and spoke.at >= last_person + 60


def test_focus_holds_and_releases():
    arb = _arb([seg(1, "09:00", "me", "ok")])
    arb.focus = True
    arb.add(_cand(parse_clock("10:00")))
    assert arb.tick(parse_clock("10:00")) is None
    arb.focus = False
    assert arb.tick(parse_clock("10:00:01")) is not None


def test_farewell_is_a_good_moment_for_a_contextual_whisper():
    s = seg(1, "15:00:00", "dev", "Thanks, see you Monday.")
    arb = _arb([s])
    arb.add(_cand(s.start, people=("dev",)))
    t, spoke = s.start, None
    while spoke is None:
        spoke = arb.tick(t)
        t += 0.5
    assert spoke.at - s.end <= 2.0


# ---------- whole days ----------
@pytest.mark.parametrize("path", sorted(DAYS.glob("*.json")), ids=lambda p: p.stem)
def test_guardrails_hold_on_every_day(path):
    s = score(run_day(Day.load(path)))
    assert s["misattributed"] == 0
    assert s["privacy_leaks"] == 0
    assert s["talk_over_person"] == 0
    assert s["quiet_violations"] == 0


@pytest.mark.parametrize("name", ["day1_tuesday", "day2_hard_cases"])
def test_development_days_are_fully_recovered(name):
    s = score(run_day(Day.load(DAYS / f"{name}.json")))
    assert s["ledger_recall"] == 1.0 and s["whisper_recall"] == 1.0


def test_baseline_is_measurably_worse():
    day = Day.load(DAYS / "day1_tuesday.json")
    base = score(run_day(day, Policy(naive=True), extractor_factory=NaiveExtractor))
    assert base["misattributed"] > 0 and base["talk_over_person"] > 0 and base["quiet_violations"] > 0


# ---------- model extractor: proposals are checked ----------
def _llm_ex(response):
    calls = []

    def caller(system, user):
        calls.append(user)
        return json.dumps({"ops": response})

    return llm_factory(caller)(PEOPLE, TUESDAY), calls


def test_llm_cannot_file_someone_elses_line_as_my_commitment():
    ex, _ = _llm_ex([{"op": "create", "kind": "commitment", "owner": "me", "what": "fix the login bug",
                      "evidence": "I'll fix the login bug"}])
    ex.observe(seg(1, "09:00", "dev", "I'll fix the login bug."), 0)
    assert ex.items == [] and "someone else's line" in ex.rejected[0]["reason"]


def test_llm_evidence_must_be_real():
    ex, _ = _llm_ex([{"op": "create", "kind": "commitment", "owner": "me", "what": "renew passport",
                      "evidence": "I promised to renew my passport"}])
    ex.observe(seg(1, "09:00", "me", "Need to remember the passport thing."), 0)
    assert ex.items == [] and "evidence" in ex.rejected[0]["reason"]


def test_llm_valid_proposal_is_applied_with_parsed_due():
    ex, calls = _llm_ex([{"op": "create", "kind": "commitment", "owner": "me", "counterparty": None,
                          "what": "renew passport", "due_phrase": "next week",
                          "evidence": "renew my passport"}])
    ex.observe(seg(1, "09:00", "me", "Need to remember to renew my passport next week."), 0)
    assert ex.items[0].what == "renew passport" and ex.items[0].due_text == "next week"
    assert "NEW line: me:" in calls[0]


def test_llm_garbage_falls_back_to_rules():
    def caller(system, user):
        return "sorry, I can't do JSON today"

    ex = llm_factory(caller)(PEOPLE, TUESDAY)
    ex.observe(seg(1, "09:00", "me", "I'll call the plumber today."), 0)
    assert ex.items and ex.rejected and "used rules" in ex.rejected[0]["reason"]

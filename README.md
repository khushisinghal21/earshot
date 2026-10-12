# Earshot

**A speak-or-hold engine and eval harness for an AI that lives in your ear.**
Built as a proof of concept for [Moonshot](https://moonshot.computer/hi) (YC S26), the team behind Samantha on Jalapeño.

An always-listening companion has three ways to lose someone's trust that a chat assistant never faces:

1. **Talking at the wrong moment.** Over your friend mid-sentence, or during a Focus block.
2. **Getting "whose words" wrong.** Filing your coworker's "I'll fix it" as *your* promise, or acting on the TV.
3. **Remembering what it shouldn't.** A bystander's health or money detail, said in front of you, not to it.

Earshot replays a day of diarized transcript plus phone context (calendar, Focus, mail) and decides, causally and with a logged reason for every choice:

- **what counts** as an open loop (your commitments, what others owe you, ideas) and when it closes,
- **what is kept** about people, and what is heard but deliberately not stored,
- **when Samantha may speak**, holds, saves something for the next brief, or lets it go.

Then it **scores itself** against hand-labelled days, next to a keyword baseline.

▶ **Interactive replay:** `docs/index.html` (each day rendered as a screenplay, with Samantha's whispers and the reasons behind every hold).

## Results

D1 and D2 were written while building. **D3 was written and labelled first, then run once and not tuned.**

| Metric | D1 base | D1 Earshot | D2 base | D2 Earshot | D3 base | D3 Earshot |
|---|---:|---:|---:|---:|---:|---:|
| Ledger recall (higher better) | 40% | **100%** | 38% | **100%** | 11% | **56%** |
| Ledger precision (higher better) | 33% | **100%** | 33% | **100%** | 17% | **83%** |
| Right person on the loop (higher better) | 0% | **100%** | - | **100%** | 0% | **67%** |
| Open/closed correct (higher better) | 100% | **100%** | 100% | **100%** | 0% | **40%** |
| Others' promises filed as yours (lower better) | 6 | **0** | 5 | **0** | 2 | **0** |
| Mind facts kept (higher better) | 100% | **100%** | 100% | **100%** | 0% | **0%** |
| Bystander secrets stored (lower better) | 1 | **0** | 2 | **0** | 2 | **0** |
| Right whisper, right window (higher better) | 0% | **100%** | 0% | **100%** | 0% | **33%** |
| Whispers nobody needed (lower better) | 2 | **0** | 0 | **1** | 0 | **1** |
| Talked over a person (lower better) | 1 | **0** | 0 | **0** | 0 | **0** |
| Spoke during Focus (lower better) | 1 | **0** | 0 | **0** | 0 | **0** |

**What D3 says.** The guardrails (attribution, privacy, never talking over people, honoring Focus and pause) are structural, so they held on unseen data. Understanding did not: every miss is language, not policy.

| Miss on D3 | Example |
|---|---|
| Coreference | "can you handle **it** this time?" = the wifi bill, one sentence earlier |
| Promises without "I'll" | "Need to remember to renew my passport", "Sure, will do" to "send me the photos" |
| Paraphrased closure | "Collected the order", "Kitchen's clean" (so one whisper nagged about a bill already paid) |
| Spoken numbers, pronouns | "by **five**", "call **him**" (= Dad) |
| Wrong memory | "photos from Neha's wedding" stored as a fact about Neha |

That split is the design: **a model handles semantics; deterministic code enforces the invariants; the eval keeps both honest.** `earshot/llm.py` implements it.

## Bugs the eval caught while building

- **Talking over Dev (D2).** A 2.8s gap after my one-word "Nice." looked like a lull; Dev kept talking. Fix: wait longer after the wearer's own backchannel, and treat a goodbye ("see you", "gotta run") as the best moment there is.
- **"Got it" closed a loop (D1).** A rule for "I paid it" also matched "Got it." in standup and silently closed the oat-milk promise. Fix: idiom guard, and ambiguous verbs ("got") never close by pronoun alone.

## How it works

```
 audio chunks ─▶ ASR + diarization ─▶ Segment(speaker, text, t)      phone context ─▶ ContextEvent
                                         │                                              │
                     ┌───────────────────┼──────────────────────┐                      │
                     ▼                   ▼                      ▼                      ▼
               LedgerExtractor       MindBuilder          CandidateMaker ◀──── calendar / mail / Focus
   your promises, waiting-ons,   facts about people;     contextual (they're here),
   ideas; closes loops           privacy guard           deadline, news, idea echo
                                                                │
                                                                ▼
                                                   Arbiter (causal, every 0.5s)
                                 speaking now? in a conversation? about who's here? lull type?
                                 Focus / paused? cooldown? hourly budget?
                                                                │
                                         spoke · held(reason) · digest · dropped
```

**Causality.** At time *t* the arbiter knows voice activity up to *t*, transcripts that have already returned (ASR latency is modeled), and events so far. It never knows when the next person will start talking, so collisions are measured afterwards, the way they'd happen in an ear.

**Attribution invariants** (`ledger.py`): only the wearer's own words, or the wearer's yes to someone's request, create a commitment for the wearer. Other people's "I'll" becomes a waiting-on. Media audio creates nothing. Hedges, conditionals, sarcasm, negations, questions and filler are logged as "not counted" with the reason.

**Privacy guard** (`mind.py`): everyday facts about people are kept (birthdays, likes, upcoming events). Sensitive categories (health, money, legal, relationships) about anyone other than the wearer are heard and not stored, with a log line saying so. The wearer's own sensitive facts are stored only as "ask first".

**Model extractor** (`llm.py`): the model reads an 8-line window and proposes ledger ops as JSON. Code rejects any op whose evidence quote isn't in what was heard, any wearer commitment proposed from someone else's line, any owner who isn't the speaker, and any close from a third party. Malformed output falls back to rules for that line. Covered by tests with a fake model; set `ANTHROPIC_API_KEY` to score it for real.

## Run it

```bash
pip install -e ".[dev]"
python -m pytest                       # 38 tests
python -m earshot eval                 # scorecard above
python -m earshot eval --extractor llm # same with the model extractor (needs ANTHROPIC_API_KEY)
python -m earshot show data/days/day2_hard_cases.json   # narrated replay
python -m earshot report               # rebuild docs/index.html
```

A day is plain JSON: `segments` (time, speaker, text), `events` (Focus, calendar, mail), and `truth` labels. Adding a day is how you add a test.

## Limits

Synthetic, hand-written transcripts; speaker labels are given (standing in for diarization and voiceprints); timings come from word counts. Three days is a smoke test, not a benchmark. The numbers above are reported as they came out.

## What I'd do next with Moonshot's data

1. Run the harness on consented beta transcripts with labels from users' own edits to the ledger and memory ("let it go", "edit memory"), so every correction becomes a test case.
2. Score the model extractor against rules on that set, and keep the invariants in code.
3. Learn the lull model from real turn-taking (prosody, gaze-free cues) instead of fixed thresholds, with "talked over someone" as the metric that has to stay at zero.

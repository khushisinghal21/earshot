"""python -m earshot eval            # score every day: baseline vs earshot
python -m earshot eval --extractor llm   # same, with the model-backed ledger (needs ANTHROPIC_API_KEY)
python -m earshot show data/days/day1_tuesday.json   # narrate one day
python -m earshot report           # write docs/runs.json + docs/index.html (the timeline viewer)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .arbiter import Policy
from .engine import Day, run_day
from .evaluate import METRIC_ORDER, NaiveExtractor, fmt, score
from .models import fmt_clock

ROOT = Path(__file__).resolve().parent.parent
DAYS = ROOT / "data" / "days"


def _modes(extractor: str):
    yield "baseline", lambda d: run_day(d, Policy(name="baseline", naive=True), extractor_factory=NaiveExtractor)
    if extractor == "llm":
        from .llm import llm_factory

        yield "earshot+llm", lambda d: run_day(d, Policy(), extractor_factory=llm_factory())
    else:
        yield "earshot", lambda d: run_day(d, Policy())


def cmd_eval(args) -> dict:
    days = sorted(Path(args.days).glob("*.json"))
    table: dict = {}
    for path in days:
        day = Day.load(path)
        table[day.name] = {m: score(fn(day)) for m, fn in _modes(args.extractor)}
    modes = [m for m, _ in _modes(args.extractor)]
    for name, res in table.items():
        print(f"\n{name}")
        print(f"  {'metric':36s}" + "".join(f"{m:>14s}" for m in modes))
        for key, label, better in METRIC_ORDER:
            print(f"  {label:36s}" + "".join(f"{fmt(res[m][key]):>14s}" for m in modes) + f"   ({better} is better)")
    return table


def cmd_show(args) -> None:
    r = run_day(Day.load(args.day), Policy())
    print(f"# {r.day.name}\n")
    events = [(s.start, f"{r.people.name(s.speaker) if s.is_person else s.speaker:>10}: {s.text}") for s in r.day.segments]
    events += [(e.at, f"{'[' + e.kind + ']':>10}  {e.data.get('say', '')}") for e in r.day.events]
    events += [(d.at, f"{'SAMANTHA':>10} >> {d.said}") for d in r.spoke()]
    for t, line in sorted(events):
        print(f"{fmt_clock(t)}  {line}")
    print("\nLedger:")
    for i in r.ledger:
        print(f"  {i.id} [{i.status:6s}] {i.kind:10s} {r.people.name(i.owner):>8} -> {r.people.name(i.counterparty):8} {i.what}"
              + (f" ({i.due_text})" if i.due_text else ""))
    print("\nHeard but not kept:")
    for w in r.withheld:
        print(f"  {fmt_clock(w.at)} about {w.about}: {w.reason}")


def cmd_report(args) -> None:
    from .report import write_report

    out = write_report(Path(args.days), ROOT / "docs")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="earshot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--days", default=str(DAYS))
    e.add_argument("--extractor", choices=["rules", "llm"], default="rules")
    e.add_argument("--json", help="also write results to this file")
    s = sub.add_parser("show")
    s.add_argument("day")
    r = sub.add_parser("report")
    r.add_argument("--days", default=str(DAYS))
    args = ap.parse_args()
    if args.cmd == "eval":
        table = cmd_eval(args)
        if args.json:
            Path(args.json).write_text(json.dumps(table, indent=2))
    elif args.cmd == "show":
        cmd_show(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()

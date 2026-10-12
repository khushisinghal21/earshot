"""Build docs/runs.json and docs/index.html (the screenplay-style replay viewer)."""
from __future__ import annotations

import json
from pathlib import Path

from .arbiter import Policy
from .engine import Day, run_day
from .evaluate import METRIC_ORDER, NaiveExtractor, score

TEMPLATE = Path(__file__).resolve().parent / "viewer.html"


def build(days_dir: Path) -> dict:
    days = []
    for path in sorted(days_dir.glob("*.json")):
        day = Day.load(path)
        raw = json.loads(path.read_text())
        runs = {}
        for mode in ("earshot", "baseline"):
            if mode == "baseline":
                r = run_day(day, Policy(name="baseline", naive=True), extractor_factory=NaiveExtractor)
            else:
                r = run_day(day, Policy())
            runs[mode] = {**r.to_json(), "score": score(r)}
            # mark whispers that collided with speech, so the viewer can show it
            for d in runs[mode]["decisions"]:
                if d["outcome"] == "spoke":
                    a, b = d["at"], d["at"] + d["duration"]
                    hit = [s for s in day.segments if s.start < b and s.end > a and s.is_person]
                    d["talked_over"] = [r.people.name(s.speaker) for s in hit]
        days.append({"id": path.stem, "name": day.name, "date": day.date, "heldout": raw.get("heldout", False),
                     "truth": day.truth, "runs": runs})
    return {"metrics": [{"key": k, "label": l, "better": b} for k, l, b in METRIC_ORDER], "days": days}


def write_report(days_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build(days_dir)
    (out_dir / "runs.json").write_text(json.dumps(data, indent=1))
    html = TEMPLATE.read_text().replace("/*__EARSHOT_DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = out_dir / "index.html"
    out.write_text(html)
    return out

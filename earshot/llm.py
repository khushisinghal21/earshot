"""Model-backed ledger extraction: the model proposes, the code disposes.

The held-out day shows where rules break: coreference ("can you handle *it*"),
implicit commitments ("need to remember to renew my passport"), paraphrased
closures ("Kitchen's clean"). Those are language problems, so a model reads a
short window of transcript and proposes ledger operations.

What a model is *not* trusted with is enforced here in code:
  * a commitment for the wearer must come from the wearer's own words or the
    wearer's yes to a request in the window (speaker attribution),
  * media audio never creates anything,
  * a close must point at an item that exists and is open,
  * every op must quote evidence that actually appears in the window.

Set ANTHROPIC_API_KEY to use it; otherwise run_day falls back to the rule
extractor. Model: EARSHOT_MODEL (default claude-sonnet-5-5).
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable, Optional

from .ledger import LedgerExtractor
from .models import MEDIA, WEARER, Segment, fmt_clock
from .timeparse import parse_due

SYSTEM = """You maintain a ledger of open loops for the wearer of an always-listening earpiece.
You get the last few transcript lines (speaker ids: "me" is the wearer, "media" is TV/video,
others are people) and the currently open ledger items. Look ONLY at the NEW line and return
JSON: {"ops": [...]}. Each op is one of:
  {"op":"create","kind":"commitment"|"waiting_on"|"idea","owner":"<speaker id>","counterparty":"<id or null>",
   "what":"<short imperative from the wearer's point of view, pronouns resolved>","due_phrase":"<as spoken or null>",
   "evidence":"<exact quote from the window>"}
  {"op":"close","item_id":"L3","evidence":"<exact quote>"}
  {"op":"update_due","item_id":"L3","due_phrase":"...","evidence":"<exact quote>"}
Rules: hedges, conditionals, sarcasm, questions and filler are not commitments. A request from
someone else becomes the wearer's commitment only if the wearer agrees. Other people's promises
are waiting_on with owner = that person. Resolve "it", "him", "her" from the window. Return
{"ops": []} if nothing changes."""

Caller = Callable[[str, str], str]  # (system, user) -> raw model text


def anthropic_caller(model: Optional[str] = None) -> Optional[Caller]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    import httpx

    model = model or os.environ.get("EARSHOT_MODEL", "claude-sonnet-5-5")

    def call(system: str, user: str) -> str:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model, "max_tokens": 800, "system": system,
                  "messages": [{"role": "user", "content": user}]},
            timeout=60,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])

    return call


class LLMLedgerExtractor(LedgerExtractor):
    window: int = 8

    def __init__(self, people, weekday, caller: Optional[Caller] = None, **kw):
        super().__init__(people=people, weekday=weekday, **kw)
        self.caller = caller or anthropic_caller()
        self.rejected: list[dict] = []
        self._history: list[Segment] = []
        if self.caller is None:
            raise RuntimeError("LLM extractor needs ANTHROPIC_API_KEY (or an injected caller)")

    def observe(self, seg: Segment, now: float):
        self._history.append(seg)
        if seg.speaker == MEDIA:
            self._skip(seg, "media audio (not a person in the room)")
            return []
        win = self._history[-self.window:]
        lines = "\n".join(f"[{fmt_clock(s.start)}] {s.speaker}: {s.text}" for s in win)
        open_items = "\n".join(f"{i.id} {i.kind} owner={i.owner} cp={i.counterparty}: {i.what}"
                               for i in self.open_items()) or "(none)"
        known = ", ".join(f"{pid}={name}" for pid, name in self.people.display.items())
        user = f"Known people: {known}\nOpen items:\n{open_items}\n\nWindow:\n{lines}\n\nNEW line: {seg.speaker}: {seg.text}"
        try:
            raw = self.caller(SYSTEM, user)
            ops = json.loads(re.search(r"\{.*\}", raw, re.S).group(0)).get("ops", [])
        except Exception as e:  # model or parse failure -> fall back to rules for this line
            self.rejected.append({"segment": seg.id, "reason": f"model error, used rules: {e}"})
            return super().observe(seg, now)
        return [c for op in ops for c in self._apply(op, seg, win, now)]

    def _apply(self, op: dict, seg: Segment, win: list[Segment], now: float):
        ev = (op.get("evidence") or "").lower().strip(" .\"'")
        if not ev or not any(ev in s.text.lower() for s in win):
            return self._reject(seg, op, "evidence not found in what was heard")
        kind = op.get("op")
        if kind == "create":
            owner = op.get("owner")
            if op.get("kind") == "commitment" and owner == WEARER and not seg.is_wearer:
                return self._reject(seg, op, "wearer commitment proposed from someone else's line")
            if owner != WEARER and owner != seg.speaker:
                return self._reject(seg, op, "owner is not the person speaking")
            due, due_text = parse_due(op.get("due_phrase") or "", now, self.weekday)
            item = self._new(op.get("kind", "commitment"), owner, op.get("counterparty"), op.get("what", ""),
                             due, due_text, seg, now, 0.8, "model proposed, checks passed")
            return [("created", item)]
        target = next((i for i in self.open_items() if i.id == op.get("item_id")), None)
        if target is None:
            return self._reject(seg, op, "no such open item")
        if kind == "close":
            # the owner can say it's done; the wearer can report on others; nobody else closes your loops
            if not (seg.speaker == target.owner or seg.is_wearer):
                return self._reject(seg, op, "only the owner or the wearer can report it done")
            self._close(target, now, f"model: \"{op.get('evidence')}\"")
            return [("closed", target)]
        if kind == "update_due":
            due, due_text = parse_due(op.get("due_phrase") or "", now, self.weekday)
            if due is None:
                return self._reject(seg, op, "unparseable due")
            target.history.append(f"due moved {target.due_text!r} -> {due_text!r} (model)")
            target.due, target.due_text = due, due_text
            return [("updated", target)]
        return self._reject(seg, op, "unknown op")

    def _reject(self, seg, op, why):
        self.rejected.append({"segment": seg.id, "op": op, "reason": why})
        return []


def llm_factory(caller: Optional[Caller] = None):
    def make(people, weekday):
        return LLMLedgerExtractor(people=people, weekday=weekday, caller=caller)
    return make


__all__ = ["LLMLedgerExtractor", "llm_factory", "anthropic_caller"]

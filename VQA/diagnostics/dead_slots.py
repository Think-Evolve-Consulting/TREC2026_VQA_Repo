#!/usr/bin/env python3
"""Identify every fallback ("dead") slot in the fused run, and why it is dead.

A dead slot is one the fusion could not fill from either model's candidates, so
`fill_slots()` padded it from the generic FALLBACK_ANSWERS pool. Those score ~0
against ground truth, so they are pure lost score -- but they are not all the
same problem, and the causes want separating before anything is done about them.

Writes:
    dead.slots.task1.csv    one row per affected query, for triage
    dead.slots.task1.json   the same plus the fallback texts used
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

FUSED_CACHE = ROOT / "fused.task1.cache.jsonl"
GEMMA_CACHE = ROOT / "answers.task1.gemma4.ens.cache.jsonl"
QWEN_CACHE = ROOT / "answers.task1.qwen3vl.ens.cache.jsonl"
QUERIES = ROOT / "vqa.2026.testingDataset.Task1.json"
ROUTING = ROOT / "query.routing.task1.json"

OUT_CSV = ROOT / "dead.slots.task1.csv"
OUT_JSON = ROOT / "dead.slots.task1.json"

NON_ANSWER_RE = re.compile(
    r"^(?:unknown|unclear|n/?a|no idea|cannot be determined"
    r"|(?:it |the answer |this )?(?:is )?(?:not |un)(?:known|clear|shown"
    r"|visible|determined|specified)"
    r"|(?:it |this )?cannot be (?:determined|answered|seen)"
    r"|(?:the )?video does not show(?: this)?"
    r"|not (?:shown|visible|clear|stated)(?: in the video)?)"
    r"[.!]?$",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}

    if not path.exists():
        return out

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if line:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[int(record["Q_ID"])] = record

    return out


def load_json(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


fused = load_jsonl(FUSED_CACHE)
gemma = load_jsonl(GEMMA_CACHE)
qwen = load_jsonl(QWEN_CACHE)
queries = {q["Q_ID"]: q for q in load_json(QUERIES)}
routing = {int(r["Q_ID"]): r for r in load_json(ROUTING)} \
    if ROUTING.exists() else {}


def model_answers(record: dict | None) -> list[str]:
    """That model's own surviving answers, excluding its own padding."""
    if not record:
        return []

    return [a["Answer"] for a in record.get("answers") or ()
            if a.get("source") == "model"]


def classify(q_id: int, rec: dict) -> str:
    """Why could the fusion not fill ten slots for this query?"""
    g, w = gemma.get(q_id), qwen.get(q_id)

    if not g and not w:
        return "no record from either model"

    g_missing = g is None
    w_missing = w is None

    if g_missing or w_missing:
        return f"only one model ran ({'qwen3vl' if g_missing else 'gemma4'})"

    # Evidence actually available to the models
    g_video = bool(g.get("n_frames"))
    w_video = bool(w.get("video_attached"))
    has_text = bool(g.get("has_transcript") or w.get("has_transcript"))

    if not g_video and not w_video and not has_text:
        return "no evidence: no frames, no video, no transcript"

    pooled = rec.get("pooled_unique") or 0
    g_ans, w_ans = model_answers(g), model_answers(w)
    everything = g_ans + w_ans
    refusals = [a for a in everything if NON_ANSWER_RE.match(a.strip())]

    if everything and len(refusals) == len(everything):
        return "both models refused (all answers are non-answers)"

    if pooled <= 2:
        return "near-total collapse: <=2 distinct from 60 candidates"

    if not w_ans:
        return "qwen3vl contributed nothing usable"

    if not g_ans:
        return "gemma4 contributed nothing usable"

    return "pooled short of ten (duplicate collapse)"


rows = []
slot_total = len(fused) * 10
dead_total = 0

for q_id, rec in sorted(fused.items()):
    dead = [a for a in rec.get("answers") or () if a.get("source") == "fallback"]

    if not dead:
        continue

    dead_total += len(dead)
    g, w = gemma.get(q_id), qwen.get(q_id)
    query = queries.get(q_id, {})
    route = routing.get(q_id, {})

    rows.append({
        "Q_ID": q_id,
        "Video_ID": query.get("Video_ID") or rec.get("Video_ID"),
        "dead_slots": len(dead),
        "pooled_unique": rec.get("pooled_unique"),
        "n_gemma4_cands": rec.get("n_gemma4"),
        "n_qwen3vl_cands": rec.get("n_qwen3vl"),
        "gemma4_answers": len(model_answers(g)),
        "qwen3vl_answers": len(model_answers(w)),
        "agreed_exact": rec.get("agreed"),
        "agreed_soft": rec.get("agreed_soft"),
        "question_type": route.get("Question_Type"),
        "route": route.get("Route"),
        "video_attached_qwen": (w or {}).get("video_attached"),
        "frames_gemma4": (g or {}).get("n_frames"),
        "duration_sec": (w or {}).get("duration_sec"),
        "cause": classify(q_id, rec),
        "question": (query.get("Question") or "")[:110],
        "fallback_texts": " | ".join(a["Answer"] for a in dead),
    })

print(f"fused queries        : {len(fused)}")
print(f"queries with dead    : {len(rows)}  "
      f"({100 * len(rows) / max(len(fused), 1):.1f}%)")
print(f"dead slots           : {dead_total} of {slot_total}  "
      f"({100 * dead_total / max(slot_total, 1):.2f}%)\n")

print("distribution — how many slots each affected query lost")
hist = Counter(r["dead_slots"] for r in rows)
for k in sorted(hist):
    bar = "#" * min(int(60 * hist[k] / max(hist.values())), 60)
    print(f"  {k:>2} dead : {hist[k]:>4} queries  "
          f"{hist[k] * k:>5} slots  {bar}")

print("\ncause breakdown")
causes = Counter(r["cause"] for r in rows)
for cause, count in causes.most_common():
    slots = sum(r["dead_slots"] for r in rows if r["cause"] == cause)
    print(f"  {count:>4} queries  {slots:>5} slots  "
          f"{100 * slots / max(dead_total, 1):>5.1f}%  {cause}")

print("\nby question type")
types = Counter(r["question_type"] for r in rows)
for qtype, count in types.most_common(10):
    slots = sum(r["dead_slots"] for r in rows if r["question_type"] == qtype)
    print(f"  {count:>4} queries  {slots:>5} slots  {qtype}")

worst = sorted(rows, key=lambda r: -r["dead_slots"])[:15]
print(f"\nworst {len(worst)} queries")
print(f"{'Q_ID':>6} {'dead':>5} {'pool':>5} {'g':>3} {'w':>3} {'type':>8}  question")
for r in worst:
    print(f"{r['Q_ID']:>6} {r['dead_slots']:>5} {r['pooled_unique']:>5} "
          f"{r['gemma4_answers']:>3} {r['qwen3vl_answers']:>3} "
          f"{str(r['question_type'])[:8]:>8}  {r['question'][:60]}")

fields = [k for k in rows[0]] if rows else []

with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

OUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                    encoding="utf-8")

print(f"\nwrote {OUT_CSV.name} and {OUT_JSON.name} ({len(rows)} rows)")

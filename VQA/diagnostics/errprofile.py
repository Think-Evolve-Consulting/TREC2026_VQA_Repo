"""Error profile of the 2026 qwen3vl / cosmos / gemma4 answer caches."""
import json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"E:\TREC_2026")


def load(p):
    rows = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "Q_ID" in r:
                rows[r["Q_ID"]] = r          # last write wins
    return rows


def pct(n, d):
    return f"{100.0 * n / d:5.1f}%" if d else "  n/a"


def profile(name, path):
    rows = load(ROOT / path)
    n = len(rows)
    print(f"\n=== {name}  ({path})  {n} queries ===")
    if not n:
        return

    fr = Counter(r.get("finish_reason") for r in rows.values())
    deg = Counter(r.get("degrade") for r in rows.values())
    raw = Counter(r.get("n_raw_candidates") for r in rows.values())
    fb = Counter(r.get("fallback") for r in rows.values())

    unparsed = sum(1 for r in rows.values() if not r.get("content_parsed", True))
    any_fb = sum(1 for r in rows.values() if (r.get("fallback") or 0) > 0)
    full_fb = sum(1 for r in rows.values() if (r.get("fallback") or 0) >= 10)
    tot_fb = sum(r.get("fallback") or 0 for r in rows.values())
    novid = sum(1 for r in rows.values() if not r.get("video_attached", True))
    trunc = sum(1 for r in rows.values() if r.get("finish_reason") == "length")

    print(f"  finish_reason      : {dict(fr)}")
    print(f"  truncated (length) : {trunc:5d}  {pct(trunc, n)}")
    print(f"  unparsed JSON      : {unparsed:5d}  {pct(unparsed, n)}")
    print(f"  >=1 fallback slot  : {any_fb:5d}  {pct(any_fb, n)}")
    print(f"  ALL 10 fallback    : {full_fb:5d}  {pct(full_fb, n)}")
    print(f"  fallback slots     : {tot_fb:5d}  = {tot_fb / n:.2f}/10 avg"
          f"  ({pct(tot_fb, n * 10)} of all slots)")
    print(f"  no video attached  : {novid:5d}  {pct(novid, n)}")
    print(f"  n_raw_candidates   : "
          f"{sorted(((k, v) for k, v in raw.items()), key=lambda x: (x[0] is None, x[0]))}")
    print(f"  fallback histogram : "
          f"{sorted(((k, v) for k, v in fb.items()), key=lambda x: (x[0] is None, x[0]))}")
    print(f"  degrade rung       : {dict(deg)}")

    # cross-tab: truncation vs parse failure
    t_and_u = sum(1 for r in rows.values()
                  if r.get("finish_reason") == "length"
                  and not r.get("content_parsed", True))
    u_not_t = unparsed - t_and_u
    print(f"  truncated AND unparsed: {t_and_u}   unparsed but NOT truncated: {u_not_t}")

    # completion tokens on truncated vs not
    ct = [r.get("completion_tokens") for r in rows.values()
          if isinstance(r.get("completion_tokens"), int)]
    if ct:
        ct.sort()
        print(f"  completion_tokens  : median {ct[len(ct)//2]}  p90 {ct[int(len(ct)*.9)]}  max {ct[-1]}")

    rc = [r.get("reasoning_chars") for r in rows.values()
          if isinstance(r.get("reasoning_chars"), int)]
    if rc:
        rc.sort()
        print(f"  reasoning_chars    : median {rc[len(rc)//2]}  p90 {rc[int(len(rc)*.9)]}  max {rc[-1]}")

    # where does cosmos/qwen lose slots -- is it partial (1-9 model answers)?
    partial = sum(1 for r in rows.values()
                  if 0 < (r.get("n_raw_candidates") or 0) < 10)
    print(f"  PARTIAL (1-9 raw)  : {partial:5d}  {pct(partial, n)}"
          f"   <- variations would top these up")
    zero = sum(1 for r in rows.values() if (r.get("n_raw_candidates") or 0) == 0)
    print(f"  ZERO raw candidates: {zero:5d}  {pct(zero, n)}"
          f"   <- variations give these 10 more chances")
    return rows


def gemma(name, path):
    rows = load(ROOT / path)
    n = len(rows)
    print(f"\n=== {name}  ({path})  {n} queries ===")
    if not n:
        return
    reps = Counter(r.get("n_replies") for r in rows.values())
    fails = Counter(r.get("n_prompt_failures") for r in rows.values())
    pooled = [r.get("pooled") or r.get("stats", {}).get("pooled") for r in rows.values()]
    pooled = [p for p in pooled if isinstance(p, int)]
    tot_fb = 0
    tot_top = 0
    for r in rows.values():
        srcs = Counter(a.get("source") for a in r.get("answers", []))
        tot_fb += srcs.get("fallback", 0)
        tot_top += srcs.get("topup", 0)
    any_fb = sum(1 for r in rows.values()
                 if any(a.get("source") == "fallback" for a in r.get("answers", [])))
    print(f"  n_replies (calls/query): {dict(reps)}")
    print(f"  n_prompt_failures      : {dict(fails)}")
    if pooled:
        pooled.sort()
        print(f"  pooled candidates      : median {pooled[len(pooled)//2]}"
              f"  min {pooled[0]}  max {pooled[-1]}")
    print(f"  >=1 fallback slot      : {any_fb:5d}  {pct(any_fb, n)}")
    print(f"  fallback slots         : {tot_fb:5d}  = {tot_fb / n:.2f}/10 avg")
    print(f"  topup slots            : {tot_top:5d}  = {tot_top / n:.2f}/10 avg")


profile("QWEN3-VL 2026", "answers.task1.qwen3vl.cache.jsonl")
profile("COSMOS 2026", "answers.task1.cosmos.cache.jsonl")
gemma("GEMMA4 2026", "answers.task1.cache.jsonl")
profile("QWEN3-VL 2025 eval", "answers.2025.qwen3vl.cache.jsonl")
profile("COSMOS 2025 eval", "answers.2025.cosmos.cache.jsonl")

"""Redo the convoy analysis with EXPENSIVE defined by measured cost, not length.

Cost is an inverted U in video length: the 80-150 s band carries ~3,300 prompt
tokens and costs ~151 s, while clips over 150 s are downsampled to 0.36 fps and
cost half that. So "long" was the wrong variable. Define expensive empirically.
"""
import json
import random
import statistics as st
from pathlib import Path

ROOT = Path(r"E:\TREC_2026")

rows = []
with (ROOT / "answers.task1.qwen3vl.ens.cache.jsonl").open(encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

rows = [r for r in rows if r.get("ts") and r.get("elapsed")
        and r.get("prompt_tokens")]
rows.sort(key=lambda r: r["ts"])

# Expensive = top quartile of visual payload actually sent.
ptoks = sorted(r["prompt_tokens"] for r in rows)
cut = ptoks[int(len(ptoks) * 0.75)]
print(f"n = {len(rows)}   prompt_tokens p75 = {cut}\n")

expensive = [r["prompt_tokens"] >= cut for r in rows]
n_exp = sum(expensive)

exp_cost = st.median([r["elapsed"] for r, e in zip(rows, expensive) if e])
cheap_cost = st.median([r["elapsed"] for r, e in zip(rows, expensive) if not e])
print(f"expensive queries (>= {cut} prompt tokens): {n_exp} "
      f"({100 * n_exp / len(rows):.1f}%)")
print(f"  median cost  expensive {exp_cost:6.1f} s   cheap {cheap_cost:6.1f} s"
      f"   ratio {exp_cost / cheap_cost:.2f}x\n")


def runs_of(flags):
    out, cur = [], 0
    for f in flags:
        if f:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


actual = runs_of(expensive)

random.seed(0)
shuffled_flags = expensive[:]
random.shuffle(shuffled_flags)
shuffled = runs_of(shuffled_flags)

print(f"{'':<26}{'actual order':>14}{'if shuffled':>14}")
for label, fn in (("consecutive runs", len),
                  ("longest run", lambda r: max(r) if r else 0),
                  ("mean run length", lambda r: round(st.mean(r), 2) if r else 0),
                  ("runs of 4+", lambda r: sum(1 for x in r if x >= 4)),
                  ("queries inside runs of 4+",
                   lambda r: sum(x for x in r if x >= 4))):
    print(f"{label:<26}{fn(actual):>14}{fn(shuffled):>14}")

# what a convoy actually costs
in_convoy = []
idx = 0
for run in runs_of(expensive):
    pass

flags = expensive
i = 0
convoy_ids, solo_ids = [], []
while i < len(flags):
    if flags[i]:
        j = i
        while j < len(flags) and flags[j]:
            j += 1
        span = list(range(i, j))
        (convoy_ids if (j - i) >= 4 else solo_ids).extend(span)
        i = j
    else:
        i += 1

if convoy_ids and solo_ids:
    c = st.median([rows[k]["elapsed"] for k in convoy_ids])
    s = st.median([rows[k]["elapsed"] for k in solo_ids])
    print(f"\nexpensive query inside a convoy of 4+ : {c:6.1f} s  "
          f"(n={len(convoy_ids)})")
    print(f"expensive query not in a convoy       : {s:6.1f} s  "
          f"(n={len(solo_ids)})")
    print(f"convoy penalty                        : {c / s:5.2f}x")

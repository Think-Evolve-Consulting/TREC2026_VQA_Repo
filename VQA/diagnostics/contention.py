"""Does a query's cost depend on WHAT ELSE was running at the same time?

If the slowdown is contention, a query's elapsed should track the size of its
co-scheduled neighbours, not its own video. Neighbours are the queries whose
completion timestamps sit closest to it -- with 4 workers those are the ones
that shared the GPU with it.
"""
import json
import statistics as st
from datetime import datetime
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
        and r.get("duration_sec")]
rows.sort(key=lambda r: r["ts"])

for r in rows:
    r["_t"] = datetime.fromisoformat(r["ts"]).timestamp()

# neighbours = the 3 queries finishing nearest in time, i.e. the other workers
K = 3
for i, r in enumerate(rows):
    lo, hi = max(0, i - K), min(len(rows), i + K + 1)
    nb = [x for j, x in enumerate(rows[lo:hi], lo) if j != i]
    r["_nb_dur"] = st.mean([x["duration_sec"] for x in nb]) if nb else 0.0

print("Does a query's cost track its OWN video length, or its NEIGHBOURS'?\n")

# --- bucket by own duration, then split by neighbourhood load ---
BANDS = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 10**9)]
print(f"{'own video':>12} | {'quiet neighbourhood':>28} | "
      f"{'busy neighbourhood':>28}")
print(f"{'':>12} | {'n':>5} {'nb dur':>8} {'elapsed':>10} | "
      f"{'n':>5} {'nb dur':>8} {'elapsed':>10}   ratio")

for lo, hi in BANDS:
    band = [r for r in rows if lo <= r["duration_sec"] < hi]

    if len(band) < 30:
        continue

    med_nb = st.median([r["_nb_dur"] for r in band])
    quiet = [r for r in band if r["_nb_dur"] <= med_nb]
    busy = [r for r in band if r["_nb_dur"] > med_nb]

    if not quiet or not busy:
        continue

    qe = st.median([r["elapsed"] for r in quiet])
    be = st.median([r["elapsed"] for r in busy])

    label = f"{lo}-{hi if hi < 10**8 else '+'}s"
    print(f"{label:>12} | {len(quiet):>5} "
          f"{st.median([r['_nb_dur'] for r in quiet]):>8.1f} {qe:>10.1f} | "
          f"{len(busy):>5} {st.median([r['_nb_dur'] for r in busy]):>8.1f} "
          f"{be:>10.1f}   {be / qe:5.2f}x")

# --- how clustered are the long videos in processing order? ---
print("\n\nAre long videos clustered in processing order?")
print("(a run of consecutive long videos is what saturates the KV cache)\n")

LONG = 80.0
flags = [r["duration_sec"] >= LONG for r in rows]
n_long = sum(flags)
runs, cur = [], 0

for f in flags:
    if f:
        cur += 1
    elif cur:
        runs.append(cur)
        cur = 0
if cur:
    runs.append(cur)

expected_max = 1
p = n_long / len(flags)
print(f"  long (>= {LONG:.0f}s) videos: {n_long} of {len(flags)} "
      f"({100 * p:.1f}%)")
print(f"  consecutive runs of long videos: {len(runs)}")
print(f"  longest run: {max(runs) if runs else 0}   "
      f"mean run: {st.mean(runs) if runs else 0:.2f}")
print(f"  runs of 4+ (all workers busy on long clips): "
      f"{sum(1 for r in runs if r >= 4)}")
print(f"  queries inside a run of 4+: "
      f"{sum(r for r in runs if r >= 4)}")

# random-shuffle baseline for run lengths
import random
random.seed(0)
sh = flags[:]
random.shuffle(sh)
sruns, cur = [], 0
for f in sh:
    if f:
        cur += 1
    elif cur:
        sruns.append(cur)
        cur = 0
if cur:
    sruns.append(cur)
print(f"\n  if the same long videos were SHUFFLED through the order:")
print(f"    longest run: {max(sruns) if sruns else 0}   "
      f"runs of 4+: {sum(1 for r in sruns if r >= 4)}   "
      f"queries inside them: {sum(r for r in sruns if r >= 4)}")

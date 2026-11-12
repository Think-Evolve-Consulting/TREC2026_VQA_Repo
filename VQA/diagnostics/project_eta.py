"""Project the remaining Qwen3-VL runtime from the *durations still to do*.

Extrapolating the observed rate is wrong here: the rate is falling because the
work is ordered so that long videos come late, not because anything is
degrading. So fit elapsed against video duration on what is finished, then
apply that to what is left.
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(r"E:\TREC_2026")
CACHE = ROOT / "answers.task1.qwen3vl.ens.cache.jsonl"
QUERIES = ROOT / "vqa.2026.testingDataset.Task1.json"
INDEX = ROOT / "videos_TEST_VQA" / "frames" / "index.json"

WORKERS = 4

done = {}
with CACHE.open(encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[r["Q_ID"]] = r

queries = json.loads(QUERIES.read_text(encoding="utf-8-sig"))
index = json.loads(INDEX.read_text(encoding="utf-8-sig"))
videos = index.get("videos", index)

durations = {}
for stem, entry in videos.items():
    d = (entry or {}).get("duration_sec")
    if d:
        durations[stem] = float(d)

remaining = [q for q in queries if q["Q_ID"] not in done]
print(f"done {len(done)} / {len(queries)}   remaining {len(remaining)}")


def dur_of(q):
    return durations.get(Path(q["Video_ID"]).stem)


# --- fit elapsed vs duration on the finished work, in duration buckets ---
BUCKETS = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 150), (150, 10**9)]


def bucket(d):
    if d is None:
        return None
    for lo, hi in BUCKETS:
        if lo <= d < hi:
            return (lo, hi)
    return None


by_bucket = {}
for r in done.values():
    b = bucket(r.get("duration_sec"))
    if b and r.get("elapsed"):
        by_bucket.setdefault(b, []).append(r["elapsed"])

print("\nmeasured cost by video length (finished work):")
cost = {}
for b in BUCKETS:
    vals = by_bucket.get(b) or []
    if vals:
        cost[b] = st.median(vals)
        print(f"  {b[0]:>4}-{b[1] if b[1] < 10**8 else '+':>4}s  n={len(vals):5d}  "
              f"median {st.median(vals):7.1f}s  mean {st.mean(vals):7.1f}s")

overall = st.median([r["elapsed"] for r in done.values() if r.get("elapsed")])

# --- what is left, by bucket ---
print("\nremaining work by video length:")
total_seconds = 0.0
unknown = 0
counts = {}

for q in remaining:
    d = dur_of(q)
    b = bucket(d)
    counts[b] = counts.get(b, 0) + 1
    if b and b in cost:
        total_seconds += cost[b]
    else:
        unknown += 1
        total_seconds += overall

for b in BUCKETS:
    if counts.get(b):
        c = cost.get(b, overall)
        print(f"  {b[0]:>4}-{b[1] if b[1] < 10**8 else '+':>4}s  "
              f"n={counts[b]:5d}  x {c:7.1f}s  = {counts[b] * c / 3600:6.2f} h serial")

if counts.get(None):
    print(f"  unknown duration  n={counts[None]:5d}  x {overall:7.1f}s")

print(f"\nserial cost remaining : {total_seconds / 3600:6.2f} h")
print(f"at {WORKERS} workers        : {total_seconds / 3600 / WORKERS:6.2f} h")

# sanity: does the same method reproduce the time already spent?
spent = sum(r["elapsed"] for r in done.values() if r.get("elapsed"))
stamps = sorted(r["ts"] for r in done.values() if r.get("ts"))
if stamps:
    from datetime import datetime
    t0 = datetime.fromisoformat(stamps[0])
    t1 = datetime.fromisoformat(stamps[-1])
    wall = (t1 - t0).total_seconds() / 3600
    print(f"\ncheck: {spent / 3600:.2f} h of summed call time over "
          f"{wall:.2f} h wall  -> effective parallelism {spent / 3600 / wall:.2f}x")
    eff = spent / 3600 / wall
    print(f"projected remaining at that measured parallelism: "
          f"{total_seconds / 3600 / eff:6.2f} h")

"""Is the Qwen3-VL run degrading, or just doing heavier work?

The distinguishing test is cost NORMALISED BY WORK. Raw seconds-per-query rose
8x over the run, but so did the visual payload. If seconds-per-prompt-token is
flat, nothing is degrading and the slowdown is entirely the video mix. If it
climbs, something is accumulating in the server.

Client side comes from the work cache; server side from vLLM's own engine stats
in the serve log.
"""
import json
import re
import statistics as st
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(r"E:\TREC_2026")
CACHE = ROOT / "answers.task1.qwen3vl.ens.cache.jsonl"

rows = []
with CACHE.open(encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

rows = [r for r in rows if r.get("ts") and r.get("elapsed")]
rows.sort(key=lambda r: r["ts"])
t0 = datetime.fromisoformat(rows[0]["ts"])

print(f"n = {len(rows)}   span "
      f"{(datetime.fromisoformat(rows[-1]['ts']) - t0).total_seconds() / 3600:.2f} h\n")

# ---------------------------------------------------------------- client side
print("=" * 88)
print("CLIENT SIDE — cost per query, and cost normalised by the work in it")
print("=" * 88)
print(f"{'hour':>5} {'n':>5} {'s/query':>9} {'prompt_tok':>11} {'calls':>6} "
      f"{'s/call':>8} {'ms per 1k prompt tok':>22} {'video s':>9}")

buckets = {}
for r in rows:
    h = int((datetime.fromisoformat(r["ts"]) - t0).total_seconds() // 3600)
    buckets.setdefault(h, []).append(r)

norm_series = []

for h in sorted(buckets):
    b = buckets[h]
    calls = [r.get("n_replies") or 10 for r in b]
    ptok = [r.get("prompt_tokens") or 0 for r in b]
    per_call = [r["elapsed"] / max(r.get("n_replies") or 10, 1) for r in b]

    # seconds of wall clock per 1000 prompt tokens processed, per call.
    # This is the number that must stay flat if nothing is degrading.
    norm = [1000.0 * (r["elapsed"] / max(r.get("n_replies") or 10, 1))
            / max(r.get("prompt_tokens") or 1, 1)
            for r in b if r.get("prompt_tokens")]

    dur = [r["duration_sec"] for r in b if r.get("duration_sec")]

    if norm:
        norm_series.append((h, st.median(norm)))

    print(f"{h:>5} {len(b):>5} {st.median([r['elapsed'] for r in b]):>9.1f} "
          f"{st.median(ptok):>11.0f} {st.median(calls):>6.0f} "
          f"{st.median(per_call):>8.1f} {st.median(norm) if norm else 0:>22.1f} "
          f"{st.median(dur) if dur else 0:>9.1f}")

if len(norm_series) >= 4:
    first = st.median([v for _, v in norm_series[:2]])
    last = st.median([v for _, v in norm_series[-2:]])
    print(f"\n  normalised cost, first 2 h: {first:6.1f} ms / 1k prompt tokens")
    print(f"  normalised cost, last  2 h: {last:6.1f} ms / 1k prompt tokens")
    print(f"  ratio: {last / first:.2f}x   -> "
          f"{'DEGRADATION' if last / first > 1.3 else 'FLAT: workload, not degradation'}")

# --------------------------------------------------------- same-size subsets
print("\n" + "=" * 88)
print("LIKE-FOR-LIKE — only 10-20 s videos, which occur throughout the run")
print("=" * 88)
print(f"{'hour':>5} {'n':>5} {'s/query':>9} {'prompt_tok':>11} {'s/call':>8}")

for h in sorted(buckets):
    b = [r for r in buckets[h]
         if r.get("duration_sec") and 10 <= r["duration_sec"] < 20]

    if len(b) < 5:
        continue

    per_call = [r["elapsed"] / max(r.get("n_replies") or 10, 1) for r in b]
    print(f"{h:>5} {len(b):>5} {st.median([r['elapsed'] for r in b]):>9.1f} "
          f"{st.median([r.get('prompt_tokens') or 0 for r in b]):>11.0f} "
          f"{st.median(per_call):>8.1f}")

# ---------------------------------------------------------------- server side
print("\n" + "=" * 88)
print("SERVER SIDE — vLLM engine stats over the run")
print("=" * 88)

pat = re.compile(
    r"Avg generation throughput: ([\d.]+) tokens/s.*?"
    r"Running: (\d+) reqs.*?Waiting: (\d+) reqs.*?"
    r"GPU KV cache usage: ([\d.]+)%.*?Prefix cache hit rate: ([\d.]+)%")

try:
    done = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", "-lc",
         "grep -o 'Avg generation throughput.*' "
         "/home/aakash/qwen3vl.instruct.log"],
        capture_output=True, text=True, timeout=300, encoding="utf-8",
        errors="replace")
    lines = (done.stdout or "").splitlines()
except (subprocess.SubprocessError, OSError) as exc:
    lines = []
    print(f"  could not read the server log: {exc}")

samples = []
for line in lines:
    m = pat.search(line)
    if m:
        samples.append(tuple(float(x) for x in m.groups()))

if samples:
    n = len(samples)
    print(f"  {n} engine-stat samples\n")
    print(f"{'decile':>7} {'gen tok/s':>10} {'running':>8} {'waiting':>8} "
          f"{'KV %':>7} {'prefix hit %':>13}")

    step = max(n // 10, 1)
    for i in range(0, n, step):
        chunk = samples[i:i + step]

        if not chunk:
            continue

        print(f"{i // step:>7} "
              f"{st.median([c[0] for c in chunk]):>10.1f} "
              f"{st.median([c[1] for c in chunk]):>8.1f} "
              f"{st.median([c[2] for c in chunk]):>8.1f} "
              f"{st.median([c[3] for c in chunk]):>7.1f} "
              f"{st.median([c[4] for c in chunk]):>13.1f}")
else:
    print("  no engine-stat lines matched")

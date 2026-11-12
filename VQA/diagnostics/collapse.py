"""Second pass: the duplicate-collapse mode, and cosmos's real video keys."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(r"E:\TREC_2026")


def load(p):
    rows = {}
    with open(ROOT / p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "Q_ID" in r:
                    rows[r["Q_ID"]] = r
    return rows


def collapse(name, path):
    rows = load(path)
    n = len(rows)
    print(f"\n=== {name}  n={n} ===")
    print("  keys:", sorted(rows[next(iter(rows))].keys()))

    # distinct model answers actually kept = 10 - fallback
    distinct = Counter(10 - (r.get("fallback") or 0) for r in rows.values())
    print("  distinct model answers kept (10-fallback):")
    for k in sorted(distinct):
        print(f"     {k:2d} distinct : {distinct[k]:5d}  {100*distinct[k]/n:5.1f}%")

    # collapse = model returned >=2 raw but they deduped to fewer
    coll = [r for r in rows.values()
            if (r.get("n_raw_candidates") or 0) >= 2
            and (10 - (r.get("fallback") or 0)) < (r.get("n_raw_candidates") or 0)]
    print(f"  COLLAPSE (raw>=2, deduped down): {len(coll)}  {100*len(coll)/n:.1f}%")
    hard = [r for r in coll if (10 - (r.get("fallback") or 0)) <= 2]
    print(f"    of which collapsed to <=2 distinct: {len(hard)}  {100*len(hard)/n:.1f}%")

    lost = sum((r.get("n_raw_candidates") or 0) - (10 - (r.get("fallback") or 0))
               for r in coll)
    print(f"    slots lost to dedupe: {lost}")

    for r in hard[:4]:
        ans = [a["Answer"] for a in r.get("answers", [])][:4]
        print(f"    Q{r['Q_ID']} raw={r.get('n_raw_candidates')} "
              f"fb={r.get('fallback')} -> {ans}")

    # cosmos video reality
    for key in ("video_exists", "video_attached", "has_transcript",
                "transcript_on_disk"):
        vals = Counter(r.get(key) for r in rows.values())
        if set(vals) != {None}:
            print(f"  {key}: {dict(vals)}")

    # zero-evidence: no video AND no transcript
    ze = sum(1 for r in rows.values()
             if not r.get("video_exists", True) and not r.get("has_transcript"))
    print(f"  zero-evidence queries (no video file, no transcript): {ze}")
    return rows


qw = collapse("QWEN3-VL 2026", "answers.task1.qwen3vl.cache.jsonl")
cs = collapse("COSMOS 2026", "answers.task1.cosmos.cache.jsonl")

# --- what would a variation-based top-up have to cover? ---
print("\n=== Repair budget: queries needing >0 extra calls ===")
for name, rows in (("qwen3vl", qw), ("cosmos", cs)):
    n = len(rows)
    need = {q: (r.get("fallback") or 0) for q, r in rows.items()
            if (r.get("fallback") or 0) > 0}
    print(f"  {name:8s}: {len(need):5d} queries ({100*len(need)/n:.1f}%), "
          f"{sum(need.values()):6d} slots to refill, "
          f"mean {sum(need.values())/max(len(need),1):.1f} slots each")

# --- variations file shape ---
var = json.loads((ROOT / "query.variations.task1.json").read_text(encoding="utf-8"))
print(f"\n=== query.variations.task1.json: {type(var).__name__}, "
      f"{len(var)} entries ===")
first = var[0] if isinstance(var, list) else next(iter(var.values()))
print("  keys:", sorted(first.keys()) if isinstance(first, dict) else type(first))
print("  sample:", json.dumps(first, indent=2)[:700])

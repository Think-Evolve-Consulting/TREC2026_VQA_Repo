#!/usr/bin/env python3
"""Does the judge select a better ten than the fuser's weights? Measure it.

``ENSEMBLE_PLAN.md`` ships its ranking under *"Accepted without measurement"*.
That was a defensible call for a formula built out of signals already on disk.
It is not defensible for ``VQA/judge.answers.py``, which spends GPU on a new
signal -- so this is the gate that was skipped, run on the 150-query 2025
sample where ground truth and media both exist.

**Three arms, all built from the identical candidate pool:**

======  ==================================================================
fuser   the current weights, judge term forced to 0 -- today's submission
judge   the same weights with the judge score active
random  ten drawn uniformly from the same pool, seeded
======  ==================================================================

**``random`` is the load-bearing arm.** Judge-versus-fuser alone cannot tell
"the judge is good" from "the fuser is bad", and the fuser's components measure
Spearman 0.032 and -0.073 against per-answer METEOR, so "the fuser is bad" is
a live hypothesis. Reading:

============================  =====================================
judge > fuser > random        selection works; weight the judge
judge > random ~= fuser       weight the judge; the old weights were noise
judge ~= fuser ~= random      no selection signal exists -- set judge=0,
                              ship the existing fuser, stop spending GPU
============================  =====================================

Mean METEOR over 150 queries moves slowly, so the **per-query sign test** is
reported alongside it: it is the more sensitive read, and a mean gap under
~0.005 here is noise either way. Queries whose pool holds ten or fewer
candidates are counted and excluded from the sign test -- every arm returns the
same ten for those, and including them only dilutes the comparison.

Prerequisite: the 2025 ens caches and the 2025 judge cache. See the header of
``RERANK_PLAN.md`` for the two commands that produce them.

Usage:
    python VQA/gate.judge.2025.py
    python VQA/gate.judge.2025.py --weights judge=5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent


def load_module(filename: str, name: str):
    """Import a sibling script despite the dots in its name."""
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


FUSE = load_module("fuse.answers.py", "fuse_answers")
SCORE = load_module("score.answers.py", "score_answers")

DEFAULT_QUERIES = ROOT / "vqa.2025.sample150.json"
DEFAULT_GT = ROOT / "vqa.gt.2025.sample150.csv"
DEFAULT_GEMMA4_CACHE = ROOT / "answers.2025.gemma4.ens.cache.jsonl"
DEFAULT_QWEN3VL_CACHE = ROOT / "answers.2025.qwen3vl.ens.cache.jsonl"
DEFAULT_JUDGE_CACHE = ROOT / "judge.2025.cache.jsonl"
DEFAULT_OUT = ROOT / "gate.judge.2025.json"

ANSWERS_PER_QUERY = 10
SEED = 20260816


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation with average ranks for ties. Same as rerank.answers.py."""
    if len(xs) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0

        while i < len(order):
            j = i

            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1

            average = (i + j) / 2 + 1

            for k in range(i, j + 1):
                out[order[k]] = average

            i = j + 1

        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))

    return num / (dx * dy) if dx and dy else None


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial p at p=0.5, ties discarded."""
    n = wins + losses

    if n == 0:
        return 1.0

    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)

    return min(1.0, 2 * tail)


def choose(clusters: list[dict], weights: dict[str, float],
           want: int) -> list[str]:
    """Rank the pool under `weights` and take the top `want`, padded as usual."""
    ranked = FUSE.rank([dict(c) for c in clusters], weights)
    chosen = [dict(c, source="model") for c in ranked[:want]]
    chosen, _ = FUSE.fill_slots(chosen, want)

    return [c["answer"] for c in chosen]


def choose_random(clusters: list[dict], want: int,
                  rng: random.Random) -> list[str]:
    """The null: ten drawn uniformly from the same pool.

    Refusals are drawn from last, exactly as ``rank()`` sorts them, so the null
    differs from the other arms only in *which* real candidates it keeps -- not
    in whether it is willing to spend a slot on "unknown". Anything else would
    make it an unfairly weak baseline.
    """
    real = [c["answer"] for c in clusters if not c["refusal"]]
    refusals = [c["answer"] for c in clusters if c["refusal"]]
    rng.shuffle(real)

    chosen = [{"answer": a} for a in (real + refusals)[:want]]
    chosen, _ = FUSE.fill_slots(chosen, want)

    return [c["answer"] for c in chosen]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Three-arm selection gate for the VLM judge, on the 2025 "
                    "ground-truth sample.")

    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--gemma4", type=Path, default=DEFAULT_GEMMA4_CACHE)
    ap.add_argument("--qwen3vl", type=Path, default=DEFAULT_QWEN3VL_CACHE)
    ap.add_argument("--judge", type=Path, default=DEFAULT_JUDGE_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--weights", help="e.g. 'judge=5' -- the judge arm only")
    ap.add_argument("--answers-per-query", type=int, default=ANSWERS_PER_QUERY)
    ap.add_argument("--seed", type=int, default=SEED)

    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    weights = FUSE.parse_weights(args.weights)
    no_judge = dict(weights, judge=0.0)

    queries = FUSE.load_json(args.queries)
    gemma = FUSE.load_cache(args.gemma4)
    qwen = FUSE.load_cache(args.qwen3vl)
    judge = FUSE.load_cache(args.judge)

    if not judge:
        print(f"no judge cache at {args.judge} -- run VQA/judge.answers.py "
              f"over the 2025 sample first", file=sys.stderr)
        return 1

    gt = {str(row["Q_ID"]): row["Answer"] for row in SCORE.load_gt(args.gt)}
    SCORE.ensure_nltk()

    print(f"gate over {len(queries)} queries")
    print(f"  gemma4 {len(gemma)} | qwen3vl {len(qwen)} | judge {len(judge)} "
          f"| gt {len(gt)}")
    print("  judge arm weights: " + "  ".join(f"{k}={v:g}"
                                              for k, v in sorted(weights.items())))

    rng = random.Random(args.seed)
    arms = {"fuser": [], "judge": [], "random": []}
    per_query: list[dict] = []
    no_gt = forced = 0

    # Every candidate the judge scored, against the METEOR its text actually
    # earns -- the correlation the fuser's own signals manage 0.03 on.
    judge_points: list[tuple[float, float]] = []

    for query in queries:
        q_id = query["Q_ID"]
        reference = gt.get(str(q_id))

        if reference is None:
            no_gt += 1
            continue

        pools = {"gemma4": FUSE.collect(gemma.get(q_id), "gemma4"),
                 "qwen3vl": FUSE.collect(qwen.get(q_id), "qwen3vl")}
        clusters = FUSE.merge(pools)
        FUSE.attach_judge(clusters, (judge.get(q_id) or {}).get("scores"))

        picks = {
            "fuser": choose(clusters, no_judge, args.answers_per_query),
            "judge": choose(clusters, weights, args.answers_per_query),
            "random": choose_random(clusters, args.answers_per_query, rng),
        }

        means = {}

        for name, answers in picks.items():
            gains = SCORE.meteor_all(
                [(reference, SCORE.normalize_text(a)) for a in answers])
            means[name] = statistics.fmean(gains)
            arms[name].append(means[name])

        # A pool of ten or fewer forces the same ten on every arm; counting
        # those separately is what keeps the sign test honest.
        if len([c for c in clusters if not c["refusal"]]) <= args.answers_per_query:
            forced += 1

        for cluster in clusters:
            if cluster.get("judge_score") is None:
                continue

            gain = SCORE.meteor_all(
                [(reference, SCORE.normalize_text(cluster["answer"]))])[0]
            judge_points.append((float(cluster["judge_score"]), gain))

        per_query.append({"Q_ID": q_id, **means,
                          "pool": len(clusters), "forced": means["fuser"] == means["judge"]})

    if not per_query:
        print("no query had both a pool and a ground-truth answer",
              file=sys.stderr)
        return 1

    print(f"\nmean METEOR over {len(per_query)} queries "
          f"({no_gt} had no ground truth, {forced} had no surplus to choose "
          f"from)")

    for name in ("judge", "fuser", "random"):
        print(f"    {name:<7} {statistics.fmean(arms[name]):.4f}")

    print("\nper-query sign test (ties discarded)")
    tests = {}

    for a, b in (("judge", "fuser"), ("judge", "random"), ("fuser", "random")):
        wins = sum(1 for row in per_query if row[a] > row[b])
        losses = sum(1 for row in per_query if row[a] < row[b])
        p = sign_test(wins, losses)
        tests[f"{a}_vs_{b}"] = {"wins": wins, "losses": losses,
                                "ties": len(per_query) - wins - losses,
                                "p": round(p, 5)}
        print(f"    {a:<7} vs {b:<7} {wins:3d}W {losses:3d}L "
              f"{len(per_query) - wins - losses:3d}T   p={p:.4f}"
              f"{'  *' if p < 0.05 else ''}")

    rho = spearman([x for x, _ in judge_points], [y for _, y in judge_points])
    print(f"\njudge score vs per-answer METEOR: Spearman "
          f"{rho if rho is None else round(rho, 4)} "
          f"over {len(judge_points)} candidates")
    print("  for reference, ranking.analysis.2025.md: score 0.032, "
          "votes -0.073")

    gap = statistics.fmean(arms["judge"]) - statistics.fmean(arms["fuser"])
    null = statistics.fmean(arms["judge"]) - statistics.fmean(arms["random"])

    print(f"\nverdict: judge {gap:+.4f} vs fuser, {null:+.4f} vs random")

    if gap > 0.005 and tests["judge_vs_fuser"]["p"] < 0.05:
        print("  -> apply the judge: pass --judge to fuse.answers.py")
    elif null > 0.005 and tests["judge_vs_random"]["p"] < 0.05:
        print("  -> the judge beats chance but not the existing weights; "
              "re-tune --weights before deciding")
    else:
        print("  -> NO measurable selection gain. Set judge=0 and ship the "
              "existing fusion; do not spend the GPU on the 2026 pass.")

    report = {
        "queries": len(per_query),
        "no_gt": no_gt,
        "forced": forced,
        "weights": weights,
        "mean_meteor": {k: round(statistics.fmean(v), 5)
                        for k, v in arms.items()},
        "sign_tests": tests,
        "judge_spearman": rho,
        "judged_candidates": len(judge_points),
        "per_query": per_query,
    }

    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4, ensure_ascii=False)
        handle.write("\n")

    print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

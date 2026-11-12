#!/usr/bin/env python3
"""Fuse several Task 2 MC ranking runs into one.

Four systems now rank the same 2,443 queries -- Gemma 4 over frames, Cosmos
over native video, Qwen3-VL-8B and Qwen3.8-27B -- and they disagree usefully:
all four pick the same option on 52.6% of queries, at least three of four on
81.9%, and all four differ on 1.9%. That disagreement is the only free signal
left in the project: combining it costs no inference at all.

**Rank fusion, not score fusion, and that is forced.** Every one of those runs
emits a permutation and nothing else -- the schema-constrained reply the
ranking scripts ask for has nowhere to put a confidence -- so there are no
comparable scores to average. Reciprocal Rank Fusion is the standard answer to
exactly that situation: it needs only the position of each option in each
system's list, is insensitive to how differently the systems are calibrated,
and has one parameter.

    RRF:   score(option) = sum over systems of  w / (k + rank)
    Borda: score(option) = sum over systems of  w * (N - rank)

RRF discounts steeply, so a system that puts an option first dominates one
that puts it second; Borda is linear, so it listens to the whole list. On a
task scored by Top-1 *and* MRR, that difference is worth measuring rather than
assuming -- `VQA/eval.mc.dev.py` scores both against the labelled 2025 sample.

Weights default to 1.0 for every system. Fit them on the dev sample before
trusting anything else: an unweighted fusion of four systems where one is
weaker can score below the best single system.

Usage:
    python VQA/fuse.rankings.mc.py --runs a.json b.json c.json --out fused.json
    python VQA/fuse.rankings.mc.py --runs a.json b.json --weights 2 1 --method borda
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = ROOT / "vqa.2026.testingDataset.Task2_v2.json"

# The conventional RRF constant. It damps the influence of any single system's
# top slot: with k=60, rank 1 scores 1/61 and rank 2 scores 1/62, so a system
# is not able to force its favourite through on its own.
DEFAULT_K = 60


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_queries(path: Path) -> dict[int, dict]:
    """Q_ID -> the official record, including the option order."""
    return {int(q["Q_ID"]): {"Video_ID": q["Video_ID"],
                             "Options": [o["option"] for o in q["Options"]]}
            for q in load_json(path)}


def load_run(path: Path) -> dict[int, list[str]]:
    """Q_ID -> answers in rank order."""
    ranked: dict[int, list[str]] = {}

    for item in load_json(path):
        answers = sorted(item["Answers"], key=lambda a: a["Rank"])
        ranked[int(item["Q_ID"])] = [a["Answer"] for a in answers]

    return ranked


def check_run(name: str, ranked: dict[int, list[str]],
              official: dict[int, dict]) -> None:
    """A run that is not a permutation of the official options cannot be fused.

    Silently tolerating a bad input here would produce a fused run that fails
    the validator for reasons pointing at the fusion rather than at its source.
    """
    missing = sorted(set(official) - set(ranked))

    if missing:
        raise SystemExit(f"{name}: does not answer {len(missing)} queries "
                         f"(first: {missing[:5]})")

    for q_id, answers in ranked.items():
        options = official[q_id]["Options"]

        if sorted(answers) != sorted(options):
            raise SystemExit(f"{name}: Q_ID {q_id} is not a permutation of the "
                             f"official options")


def fuse_query(rankings: list[list[str]], weights: list[float],
               options: list[str], method: str, k: int) -> list[str]:
    """One query's fused order over `options`.

    Ties break on mean rank, then on the official option order, so the output
    is deterministic and independent of the order the runs were passed in.
    """
    n = len(options)
    score = {option: 0.0 for option in options}
    ranks: dict[str, list[int]] = {option: [] for option in options}

    for ranking, weight in zip(rankings, weights):
        for position, answer in enumerate(ranking, start=1):
            ranks[answer].append(position)

            if method == "rrf":
                score[answer] += weight / (k + position)
            else:
                score[answer] += weight * (n - position)

    official_index = {option: i for i, option in enumerate(options)}

    return sorted(
        options,
        key=lambda o: (-score[o],
                       sum(ranks[o]) / len(ranks[o]) if ranks[o] else n + 1,
                       official_index[o]),
    )


def report(fused: dict[int, list[str]], runs: dict[str, dict[int, list[str]]],
           official: dict[int, dict]) -> None:
    """What the fusion actually did, relative to its inputs."""
    q_ids = sorted(official)
    print(f"\nfusion over {len(q_ids)} queries, {len(runs)} systems")

    for name, ranked in runs.items():
        same = sum(1 for q in q_ids if fused[q][0] == ranked[q][0])
        print(f"  agrees with {name:34s} on rank 1: {same:5d} "
              f"({100 * same / len(q_ids):5.1f}%)")

    # Where the systems split evenly, the fused answer is decided by the
    # tie-break rather than by agreement -- worth knowing how often.
    unanimous = 0
    plurality = 0

    for q in q_ids:
        votes = Counter(ranked[q][0] for ranked in runs.values())
        top = votes.most_common()

        if top[0][1] == len(runs):
            unanimous += 1
        elif len(top) > 1 and top[0][1] == top[1][1]:
            plurality += 1

    print(f"  all systems already agreed:            {unanimous:5d} "
          f"({100 * unanimous / len(q_ids):5.1f}%)")
    print(f"  no majority among the systems:         {plurality:5d} "
          f"({100 * plurality / len(q_ids):5.1f}%)  "
          f"-- these are decided by the fusion, not by a vote")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fuse Task 2 MC ranking runs by RRF or Borda count."
    )
    ap.add_argument("--runs", type=Path, nargs="+", required=True,
                    help="two or more run files over the same queries")
    ap.add_argument("--weights", type=float, nargs="+",
                    help="one weight per run (default: all 1.0). Fit these on "
                         "the labelled dev sample, do not guess them")
    ap.add_argument("--method", choices=("rrf", "borda"), default="rrf")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help="RRF constant (default: %(default)s)")
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--out", type=Path, required=True)

    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if len(args.runs) < 2:
        print("fusing one run is a copy; pass at least two", file=sys.stderr)
        return 2

    weights = args.weights or [1.0] * len(args.runs)

    if len(weights) != len(args.runs):
        print(f"got {len(weights)} weights for {len(args.runs)} runs",
              file=sys.stderr)
        return 2

    official = load_queries(args.queries)
    runs: dict[str, dict[int, list[str]]] = {}

    for path in args.runs:
        ranked = load_run(path)
        check_run(path.name, ranked, official)
        runs[path.name] = ranked

    print(f"method: {args.method}"
          f"{f' (k={args.k})' if args.method == 'rrf' else ''} | weights: "
          f"{dict(zip((p.name for p in args.runs), weights))}")

    fused: dict[int, list[str]] = {}

    for q_id, record in official.items():
        fused[q_id] = fuse_query(
            [runs[path.name][q_id] for path in args.runs],
            weights,
            record["Options"],
            args.method,
            args.k,
        )

    out = [{"Q_ID": q_id,
            "Video_ID": official[q_id]["Video_ID"],
            "Answers": [{"Rank": rank, "Answer": answer}
                        for rank, answer in enumerate(fused[q_id], start=1)]}
           for q_id in sorted(official)]

    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=4)
        handle.write("\n")

    report(fused, runs, official)
    print(f"\nwrote {len(out)} query entries to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

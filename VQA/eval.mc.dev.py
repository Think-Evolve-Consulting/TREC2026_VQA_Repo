#!/usr/bin/env python3
"""Score a Task 2 ranking method against the labelled 2025 MC set.

The 2026 MC queries have no ground truth, so every design question about
ranking -- fusion weights, pointwise scoring, permutation ensembling -- can
only be argued, not settled. The 2025 MC release *is* labelled: 2,000 queries
of four options each in ``vqa.testing.dataset.task2.MC.csv``, answers in
``vqa.gt.2025.csv``, and 1,327 of the videos on disk under
``videos_GT_2025/videos``. This turns those questions into measurements.

Two subcommands:

``sample``  draws a fixed-seed random subset of the eligible queries and writes
            it three ways -- as a query file in the released Task 2 JSON shape
            (so every ranking script runs against it unmodified), as a ground
            truth CSV restricted to the sample, and as a plain id list.

``score``   converts a run file to the CSV shape the official scorer wants,
            scores it, and -- given ``--baseline`` -- reports whether the
            difference from another run is larger than chance, paired over the
            identical queries.

Three things this exists to get right:

**The official scorer divides by the size of the ground truth file, not the
run** (``eval.vqa.mc.py``: ``total_queries = len(gt_dict)``). Scoring a
400-query sample against the full 2,000-query ground truth would therefore
report numbers about five times too low, and they would look plausible. That
is why ``sample`` writes its own restricted ground truth CSV, and ``score``
refuses to run against one that holds queries the run does not answer.

**Comparisons are paired.** Every method is scored on the same drawn sample,
because the power to separate two methods comes from the queries where they
disagree. n=400 resolves a Top-1 difference of about 4.5 points at two
standard errors, assuming the two methods disagree on a fifth of queries;
n=1,327 would resolve 2.5 points for three times the compute. When a result
lands inside the noise band, extend the same sample rather than drawing a new
one -- the draw is nested, so extending is valid and re-drawing is fishing.

**The 18 queries the scorer skips are excluded from the pool.** ``eval.vqa.mc.py``
hard-codes a SKIP set for 2025 videos that were never downloadable; sampling
one would score a zero against a method that did nothing wrong.

Usage:
    python VQA/eval.mc.dev.py sample --n 400
    python VQA/eval.mc.dev.py score --run eval.2025.mc/dummy.run.json
    python VQA/eval.mc.dev.py score --run A.json --baseline B.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

DEFAULT_MC_CSV = ROOT / "vqa.testing.dataset.task2.MC.csv"
DEFAULT_GT_CSV = ROOT / "vqa.gt.2025.csv"
DEFAULT_VIDEOS = ROOT / "videos_GT_2025" / "videos"
DEFAULT_FRAME_INDEX = ROOT / "videos_GT_2025" / "frames" / "index.json"
DEFAULT_OUT_DIR = ROOT / "eval.2025.mc"

DEFAULT_N = 400
DEFAULT_SEED = 2026

# Copied from eval.vqa.mc.py, which skips these when scoring: their videos were
# never available for download. Sampling one would charge a method for a query
# the official scorer will not even look at.
SKIP = {18, 27, 62, 222, 448, 507, 584, 629, 903, 1007, 1159, 1260, 1330,
        1381, 1554, 1691, 1755, 1822}


def load_scorer():
    """The official scorer, imported rather than shelled out to.

    Its module name has dots, so importlib is the only way in -- the same
    approach run.ensemble.py and judge.answers.py take. Importing gives the
    per-query numbers back as values; the CLI only prints them.
    """
    path = HERE / "eval.vqa.mc.py"
    spec = importlib.util.spec_from_file_location("eval_vqa_mc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def read_mc_csv(path: Path) -> list[dict]:
    """The released 2025 MC queries: QueryID, VideoID, Question, 4 choices."""
    queries = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 7:
                continue

            queries.append({
                "Q_ID": int(row[0].strip()),
                "Video_ID": row[1].strip(),
                "Question": row[2].strip(),
                "Options": [c.strip() for c in row[3:7]],
            })

    return queries


def read_gt_csv(path: Path) -> dict[int, str]:
    """Q_ID -> the correct answer text."""
    truth = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 4:
                truth[int(row[0].strip())] = row[3].strip()

    return truth


def video_path(videos_dir: Path, video_id: str) -> Path | None:
    """The dev clips are all .mp4, but check both conventions anyway."""
    for name in (video_id, video_id + ".mp4", video_id + ".webm"):
        candidate = videos_dir / name

        if candidate.is_file():
            return candidate

    return None


def load_durations(index_path: Path) -> dict[str, float]:
    if not index_path.exists():
        return {}

    try:
        with index_path.open("r", encoding="utf-8-sig") as handle:
            videos = json.load(handle).get("videos") or {}
    except (json.JSONDecodeError, OSError):
        return {}

    return {stem: entry["duration_sec"] for stem, entry in videos.items()
            if entry.get("duration_sec")}


def eligible(queries: list[dict], truth: dict[int, str],
             videos_dir: Path) -> list[dict]:
    """Queries that can be scored fairly: labelled, on disk, not skipped."""
    pool = []

    for query in queries:
        if query["Q_ID"] in SKIP or query["Q_ID"] not in truth:
            continue

        if video_path(videos_dir, query["Video_ID"]) is None:
            continue

        pool.append(query)

    return pool


def cmd_sample(args: argparse.Namespace) -> int:
    queries = read_mc_csv(args.mc_csv)
    truth = read_gt_csv(args.gt_csv)
    pool = eligible(queries, truth, args.videos)

    print(f"2025 MC queries: {len(queries)} | with ground truth: "
          f"{sum(1 for q in queries if q['Q_ID'] in truth)} | eligible "
          f"(labelled, video on disk, not in scorer SKIP): {len(pool)}")

    if args.n > len(pool):
        print(f"  asked for {args.n} but only {len(pool)} are eligible; "
              f"taking all of them", file=sys.stderr)

    rng = random.Random(args.seed)
    drawn = sorted(rng.sample(pool, min(args.n, len(pool))),
                   key=lambda q: q["Q_ID"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sample{len(drawn)}.seed{args.seed}"

    queries_path = args.out_dir / f"{stem}.queries.json"
    gt_path = args.out_dir / f"{stem}.gt.csv"
    ids_path = args.out_dir / f"{stem}.ids.txt"

    # The released Task 2 shape, so every rank.options.mc.*.py runs against
    # this file with nothing but --queries changed.
    with queries_path.open("w", encoding="utf-8") as handle:
        json.dump([{"Q_ID": q["Q_ID"], "Video_ID": q["Video_ID"],
                    "Question": q["Question"],
                    "Options": [{"option": o} for o in q["Options"]]}
                   for q in drawn], handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    # Ground truth restricted to the draw -- see the module docstring for why
    # this matters more than it looks.
    with gt_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)

        for q in drawn:
            writer.writerow([q["Q_ID"], q["Video_ID"], q["Question"],
                             truth[q["Q_ID"]]])

    ids_path.write_text(" ".join(str(q["Q_ID"]) for q in drawn),
                        encoding="utf-8")

    audit(drawn, pool, truth, load_durations(args.frame_index))

    print(f"\nwrote {queries_path}\n      {gt_path}\n      {ids_path}")

    return 0


def audit(drawn: list[dict], pool: list[dict], truth: dict[int, str],
          durations: dict[str, float]) -> None:
    """Show the draw next to the pool it came from.

    A sample can be unlucky. Printing both distributions makes that visible
    rather than something to discover when the numbers look strange.
    """
    def summarise(rows: list[dict], label: str) -> None:
        known = sorted(durations[q["Video_ID"]] for q in rows
                       if q["Video_ID"] in durations)
        unscoreable = sum(1 for q in rows
                          if truth[q["Q_ID"]] not in q["Options"])
        line = (f"  {label:8s} n={len(rows):5d}  options/query "
                f"{sorted({len(q['Options']) for q in rows})}  "
                f"answer-not-among-options {unscoreable}")

        if known:
            line += (f"  duration median {known[len(known) // 2]:.1f}s "
                     f"(known for {len(known)})")

        print(line)

    print("\nsample audit")
    summarise(drawn, "sample")
    summarise(pool, "pool")

    if any(truth[q["Q_ID"]] not in q["Options"] for q in drawn):
        print("  note: a query whose ground-truth answer is not one of its "
              "options scores zero for every method, by construction")


def load_run(path: Path) -> dict[int, list[str]]:
    """Q_ID -> the answers in rank order."""
    with path.open("r", encoding="utf-8-sig") as handle:
        run = json.load(handle)

    ranked: dict[int, list[str]] = {}

    for item in run:
        answers = sorted(item["Answers"], key=lambda a: a["Rank"])
        ranked[int(item["Q_ID"])] = [a["Answer"] for a in answers]

    return ranked


def write_run_csv(path: Path, ranked: dict[int, list[str]],
                  video_of: dict[int, str], q_ids: list[int]) -> None:
    """The scorer's shape: QueryID, VideoID, Rank, Response -- one row each."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)

        for q_id in q_ids:
            for rank, answer in enumerate(ranked[q_id], start=1):
                writer.writerow([q_id, video_of[q_id], rank, answer])


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial p for b successes in b+c trials at p=0.5.

    The paired test that matters here: b is the count of queries the new run
    gets right and the baseline gets wrong, c the reverse. Queries both get
    right, or both get wrong, carry no information about which is better and
    are correctly ignored. Exact rather than the chi-square approximation
    because b+c is often small.
    """
    n = b + c

    if n == 0:
        return 1.0

    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)

    return min(1.0, 2 * tail)


def bootstrap_ci(diffs: list[float], seed: int, draws: int = 10000
                 ) -> tuple[float, float]:
    """Percentile CI for a paired mean difference, resampling queries."""
    rng = random.Random(seed)
    n = len(diffs)

    if n == 0:
        return (0.0, 0.0)

    means = []

    for _ in range(draws):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)

    means.sort()

    return (means[int(0.025 * draws)], means[int(0.975 * draws)])


def per_query(gt_path: Path, run_csv: Path, scorer) -> dict[int, tuple[int, float]]:
    """Q_ID -> (top-1 correct, reciprocal rank), from the official scorer.

    The scorer writes a diagnostic CSV beside the run; that file is the
    per-query record, and reading it back is what makes a paired test possible
    without re-implementing the scoring rules.
    """
    top1, mrr, diagnostic = scorer.evaluate_runs(str(gt_path), str(run_csv))
    rows: dict[int, tuple[int, float]] = {}

    with Path(diagnostic).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            rows[int(row["QueryID"])] = (int(row["Top1Correct"]),
                                         float(row["MRRScore"]))

    return top1, mrr, rows


def cmd_score(args: argparse.Namespace) -> int:
    scorer = load_scorer()

    with args.queries.open("r", encoding="utf-8-sig") as handle:
        sample = json.load(handle)

    q_ids = [int(q["Q_ID"]) for q in sample]
    video_of = {int(q["Q_ID"]): q["Video_ID"] for q in sample}

    runs = [("run", args.run)] + ([("baseline", args.baseline)]
                                  if args.baseline else [])
    scored = {}

    for label, path in runs:
        ranked = load_run(path)
        missing = [q for q in q_ids if q not in ranked]

        if missing:
            print(f"{path.name} does not answer {len(missing)} of the "
                  f"{len(q_ids)} sampled queries (first: {missing[:5]}). The "
                  f"scorer divides by the ground truth size, so a partial run "
                  f"would be scored as if those were wrong.", file=sys.stderr)
            return 1

        run_csv = args.out_dir / f"{path.stem}.{args.queries.stem}.csv"
        write_run_csv(run_csv, ranked, video_of, q_ids)
        top1, mrr, rows = per_query(args.gt, run_csv, scorer)
        scored[label] = {"path": path, "top1": top1, "mrr": mrr, "rows": rows}

        print(f"{label:9s} {path.name}")
        print(f"          Top-1 {top1:.3f}   MRR {mrr:.3f}   "
              f"over {len(rows)} queries")

    if args.baseline:
        a, b_ = scored["run"]["rows"], scored["baseline"]["rows"]
        shared = [q for q in q_ids if q in a and q in b_]
        win = sum(1 for q in shared if a[q][0] and not b_[q][0])
        loss = sum(1 for q in shared if b_[q][0] and not a[q][0])
        p = exact_mcnemar(win, loss)
        diffs = [a[q][1] - b_[q][1] for q in shared]
        lo, hi = bootstrap_ci(diffs, args.seed)

        print(f"\npaired comparison over {len(shared)} queries")
        print(f"  Top-1: run right / baseline wrong {win}, the reverse {loss} "
              f"-> {100 * (win - loss) / len(shared):+.1f} pts, "
              f"exact McNemar p = {p:.4f}")
        print(f"  MRR:   {statistics.mean(diffs):+.4f} "
              f"(95% CI {lo:+.4f} to {hi:+.4f})")

        if p > 0.05:
            print("  -> not separable at this sample size. Extend the SAME "
                  "sample before believing either direction.")

    report = args.out_dir / f"{args.run.stem}.{args.queries.stem}.score.json"

    with report.open("w", encoding="utf-8") as handle:
        json.dump({label: {"run": str(v["path"]), "top1": v["top1"],
                           "mrr": v["mrr"], "n": len(v["rows"])}
                   for label, v in scored.items()},
                  handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"\nwrote {report}")

    return 0


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Score Task 2 ranking methods on the labelled 2025 MC set."
    )
    subs = ap.add_subparsers(dest="command", required=True)

    s = subs.add_parser("sample", help="draw a fixed-seed dev sample")
    s.add_argument("--n", type=int, default=DEFAULT_N)
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--mc-csv", type=Path, default=DEFAULT_MC_CSV)
    s.add_argument("--gt-csv", type=Path, default=DEFAULT_GT_CSV)
    s.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    s.add_argument("--frame-index", type=Path, default=DEFAULT_FRAME_INDEX)
    s.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    s.set_defaults(func=cmd_sample)

    c = subs.add_parser("score", help="score a run against the dev sample")
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--baseline", type=Path,
                   help="a second run to compare against, paired over the "
                        "same queries")
    c.add_argument("--queries", type=Path,
                   default=DEFAULT_OUT_DIR / f"sample{DEFAULT_N}."
                                             f"seed{DEFAULT_SEED}.queries.json")
    c.add_argument("--gt", type=Path,
                   default=DEFAULT_OUT_DIR / f"sample{DEFAULT_N}."
                                             f"seed{DEFAULT_SEED}.gt.csv")
    c.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    c.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="bootstrap seed for the MRR interval")
    c.set_defaults(func=cmd_score)

    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

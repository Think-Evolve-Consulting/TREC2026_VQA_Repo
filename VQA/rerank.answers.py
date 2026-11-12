#!/usr/bin/env python3
"""Does the answering pass's own confidence predict correctness, and would
re-ordering the ten slots by it improve NDCG?

Two subcommands, both operating entirely on cached data -- no inference.

``analyze``
    Joins a run's work cache (``Q_ID``, ``Rank`` -> ``votes``, ``score``)
    against the per-answer metrics ``score.answers.py`` already wrote
    (``eval.2025/<run>_per_response.csv``), and asks three questions: does the
    signal correlate with per-answer METEOR, does the *emitted* rank correlate
    with it, and how much resolution does the signal actually have.

``rerank``
    Re-orders each query's ten answers by a key and writes a new run file, so
    ``score.answers.py`` can measure it with no new metric code.

**The bound worth knowing before reading any output.** ``eval.vqa.ag.py`` scores
a query as ``np.mean`` over all ten answers (``:125-126``), which is
order-independent -- so re-ranking **cannot** change mean METEOR or mean
BERTScore. Only NDCG responds to order, and NDCG_BERTScore is already saturated
at 0.993-0.995 across every run measured. This is therefore a NDCG_METEOR-only
question, currently 0.751 / 0.597 / 0.524 for gemma4 / qwen3vl / cosmos.

That order-independence is also the strongest correctness check available: if a
re-ranked run's mean METEOR differs from its baseline at all, the re-ranker lost,
duplicated or altered an answer.

**Only gemma4 has a signal.** Its ten prompts each return a self-reported
``confidence``, which ``pool_candidates()`` folds into a per-candidate ``score``
and ``votes``. Qwen3-VL and Cosmos each make one call and simply assert an
order, so for them only ``current``/``random``/``reverse``/``oracle`` are
meaningful -- which still answers "is their ordering better than chance, and how
much NDCG is being left on the table".

Usage:
    python VQA/rerank.answers.py analyze
    python VQA/rerank.answers.py rerank --run gemma4.run.2025.json \\
        --cache answers.2025.gemma4.cache.jsonl --by score \\
        --out gemma4.rerank-score.run.2025.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GT = ROOT / "vqa.gt.2025.sample150.csv"
DEFAULT_PER_RESPONSE = ROOT / "eval.2025"

# (run name, run file, work cache) for the 2025 sample.
RUNS = (
    ("gemma4", "gemma4.run.2025.json", "answers.2025.gemma4.cache.jsonl"),
    ("qwen3vl", "qwen3vl.run.2025.json", "answers.2025.qwen3vl.cache.jsonl"),
    ("cosmos", "cosmos.run.2025.json", "answers.2025.cosmos.cache.jsonl"),
)

KEYS = ("current", "random", "reverse", "score", "votes", "oracle")

DEFAULT_SEED = 20260815


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_cache(path: Path) -> dict[int, dict]:
    """Work cache keyed by Q_ID; later records win, as everywhere else here."""
    cache: dict[int, dict] = {}

    if not path.exists():
        return cache

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                cache[int(record["Q_ID"])] = record
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

    return cache


def load_per_response(path: Path) -> dict[tuple[str, int], dict]:
    """(Q_ID, Rank) -> per-answer metrics, from score.answers.py's dump."""
    table: dict[tuple[str, int], dict] = {}

    if not path.exists():
        return table

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                key = (str(row["QueryID"]).strip(), int(row["Rank"]))
            except (KeyError, ValueError):
                continue

            entry: dict[str, float] = {}

            for column in ("METEOR", "BERTScore"):
                try:
                    entry[column] = float(row[column])
                except (KeyError, TypeError, ValueError):
                    pass

            table[key] = entry

    return table


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation. scipy is installed, but this keeps `analyze` runnable
    without it and the tie handling is the same (average ranks)."""
    if len(xs) < 3:
        return None

    try:
        from scipy.stats import spearmanr
    except ImportError:
        return None

    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None  # a constant vector has no correlation to report

    result = spearmanr(xs, ys)

    return None if result.statistic != result.statistic else float(result.statistic)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def signal_of(answer: dict, key: str) -> float | None:
    value = answer.get(key)

    return float(value) if isinstance(value, (int, float)) else None


def cmd_analyze(args: argparse.Namespace) -> int:
    report: dict = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "per_response_dir": str(args.per_response),
        "note": "Re-ranking cannot change mean METEOR or mean BERTScore -- "
                "eval.vqa.ag.py averages over all ten answers, which is "
                "order-independent. Only NDCG responds to order.",
        "runs": {},
    }

    for name, run_file, cache_file in RUNS:
        metrics = load_per_response(args.per_response / f"{name}_per_response.csv")
        cache = load_cache(ROOT / cache_file)

        if not metrics:
            print(f"{name}: no per-response metrics -- run score.answers.py "
                  f"with --per-response-dir first", file=sys.stderr)
            continue

        by_rank: dict[int, list[float]] = {}
        pooled: dict[str, tuple[list[float], list[float]]] = {
            "score": ([], []), "votes": ([], []), "rank": ([], []),
        }
        within: dict[str, list[float]] = {"score": [], "votes": [], "rank": []}
        distinct_counts: list[int] = []
        low_vote_share: list[float] = []

        for q_id, record in sorted(cache.items()):
            answers = record.get("answers") or []
            rows = []

            for answer in answers:
                entry = metrics.get((str(q_id), int(answer["Rank"])))

                if entry and "METEOR" in entry:
                    rows.append((answer, entry["METEOR"]))

            if not rows:
                continue

            for answer, meteor in rows:
                by_rank.setdefault(int(answer["Rank"]), []).append(meteor)

            # Within-query correlation is the meaningful one: pooling across
            # queries mixes in how hard each question was, which swamps any
            # ordering signal.
            for key in ("score", "votes", "rank"):
                if key == "rank":
                    # Negated so "higher is better" holds for every key, making
                    # the sign of every correlation directly comparable.
                    xs = [-float(a["Rank"]) for a, _ in rows]
                else:
                    values = [signal_of(a, key) for a, _ in rows]

                    if any(v is None for v in values):
                        continue

                    xs = [float(v) for v in values]

                ys = [m for _, m in rows]
                pooled[key][0].extend(xs)
                pooled[key][1].extend(ys)

                rho = spearman(xs, ys)

                if rho is not None:
                    within[key].append(rho)

            scores = [signal_of(a, "score") for a in answers]

            if all(s is not None for s in scores) and scores:
                distinct_counts.append(len({round(float(s), 4) for s in scores}))
                low_vote_share.append(
                    sum(1 for a in answers if (a.get("votes") or 0) <= 1)
                    / len(answers))

        block: dict = {
            "queries": len({q for q in cache}),
            "has_signal": bool(distinct_counts),
            "mean_meteor_by_rank": {
                str(rank): round(statistics.fmean(values), 4)
                for rank, values in sorted(by_rank.items())
            },
            "correlation_with_meteor": {},
        }

        for key in ("rank", "score", "votes"):
            xs, ys = pooled[key]

            if not xs:
                continue

            block["correlation_with_meteor"][key] = {
                "pooled_spearman": round(spearman(xs, ys), 4)
                if spearman(xs, ys) is not None else None,
                "mean_within_query_spearman": round(
                    statistics.fmean(within[key]), 4) if within[key] else None,
                "queries_with_a_correlation": len(within[key]),
            }

        if distinct_counts:
            block["signal_resolution"] = {
                "mean_distinct_scores_of_10": round(
                    statistics.fmean(distinct_counts), 2),
                "queries_with_all_ten_tied": sum(1 for d in distinct_counts
                                                 if d == 1),
                "mean_share_answers_with_leq_1_vote": round(
                    statistics.fmean(low_vote_share), 4),
            }

        report["runs"][name] = block

    write_analysis(report, args.report_json, args.report_md)

    for name, block in report["runs"].items():
        print(f"\n{name}  ({block['queries']} queries)")
        curve = block["mean_meteor_by_rank"]

        if curve:
            ranks = sorted(curve, key=int)
            print("  mean METEOR by emitted rank:")
            print("    " + "  ".join(f"r{r}" for r in ranks))
            print("    " + "  ".join(f"{curve[r]:.3f}"[1:] for r in ranks))
            first, last = curve[ranks[0]], curve[ranks[-1]]
            print(f"    rank 1 {first:.3f} vs rank {ranks[-1]} {last:.3f}"
                  f"   (delta {first - last:+.3f})")

        for key, stats in block["correlation_with_meteor"].items():
            print(f"  {key:<6} spearman vs METEOR: "
                  f"pooled {stats['pooled_spearman']}  "
                  f"within-query mean {stats['mean_within_query_spearman']}")

        if "signal_resolution" in block:
            res = block["signal_resolution"]
            print(f"  signal resolution: "
                  f"{res['mean_distinct_scores_of_10']} distinct of 10, "
                  f"{res['mean_share_answers_with_leq_1_vote'] * 100:.0f}% of "
                  f"answers at <=1 vote")

    print(f"\nwrote {args.report_json.name} and {args.report_md.name}")

    return 0


def write_analysis(report: dict, json_path: Path, md_path: Path) -> None:
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4, ensure_ascii=False)
        handle.write("\n")

    lines = [
        "# Does the ranking signal predict correctness?",
        "",
        f"Generated {report['generated']} from cached data; no inference.",
        "",
        f"> {report['note']}",
        "",
        "## Mean METEOR by emitted rank",
        "",
        "| Run | " + " | ".join(f"r{i}" for i in range(1, 11)) + " | r1 − r10 |",
        "|---" * 12 + "|",
    ]

    for name, block in report["runs"].items():
        curve = block["mean_meteor_by_rank"]
        cells = [f"{curve.get(str(i), float('nan')):.3f}" for i in range(1, 11)]
        delta = (curve.get("1", 0.0) - curve.get("10", 0.0)) if curve else 0.0
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {delta:+.3f} |")

    lines += ["", "## Spearman correlation with per-answer METEOR", "",
              "| Run | Signal | Pooled | Within-query mean | Queries |",
              "|---|---|---|---|---|"]

    for name, block in report["runs"].items():
        for key, stats in block["correlation_with_meteor"].items():
            lines.append(
                f"| {name} | {key} | {stats['pooled_spearman']} | "
                f"{stats['mean_within_query_spearman']} | "
                f"{stats['queries_with_a_correlation']} |")

    resolution = {n: b["signal_resolution"] for n, b in report["runs"].items()
                  if "signal_resolution" in b}

    if resolution:
        lines += ["", "## Signal resolution", "",
                  "| Run | Distinct scores of 10 | All ten tied | Answers at ≤1 vote |",
                  "|---|---|---|---|"]

        for name, res in resolution.items():
            lines.append(
                f"| {name} | {res['mean_distinct_scores_of_10']} | "
                f"{res['queries_with_all_ten_tied']} | "
                f"{res['mean_share_answers_with_leq_1_vote'] * 100:.0f}% |")

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------

def load_scorer():
    """Import score.answers.py despite the dot in its name."""
    import importlib.util

    path = Path(__file__).resolve().parent / "score.answers.py"
    spec = importlib.util.spec_from_file_location("score_answers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_answers"] = module
    spec.loader.exec_module(module)

    return module


def cmd_rerank(args: argparse.Namespace) -> int:
    run = load_json(args.run)
    cache = load_cache(args.cache) if args.cache else {}
    rng = random.Random(args.seed)

    gt: dict[str, str] = {}
    scorer = None

    if args.by == "oracle":
        scorer = load_scorer()
        gt = {row["Q_ID"]: row["Answer"] for row in scorer.load_gt(args.gt)}
        scorer.ensure_nltk()

    out: list[dict] = []
    missing_signal = 0

    for record in run:
        q_id = record["Q_ID"]
        answers = list(record["Answers"])
        cached = (cache.get(q_id) or {}).get("answers") or []
        signal = {int(a["Rank"]): a for a in cached}

        if args.by == "current":
            ordered = answers
        elif args.by == "reverse":
            ordered = list(reversed(answers))
        elif args.by == "random":
            ordered = list(answers)
            rng.shuffle(ordered)
        elif args.by == "oracle":
            reference = gt.get(str(q_id))

            if reference is None:
                missing_signal += 1
                ordered = answers
            else:
                pairs = [(reference, scorer.normalize_text(a["Answer"]))
                         for a in answers]
                gains = scorer.meteor_all(pairs)
                # Stable on ties: equal-scoring answers keep their submitted
                # relative order rather than being shuffled by the sort.
                ordered = [a for _, a in sorted(
                    zip(gains, answers),
                    key=lambda pair: (-pair[0], pair[1]["Rank"]))]
        else:  # score / votes
            values = [signal.get(int(a["Rank"]), {}).get(args.by)
                      for a in answers]

            if any(not isinstance(v, (int, float)) for v in values):
                missing_signal += 1
                ordered = answers
            else:
                ordered = [a for _, a in sorted(
                    zip(values, answers),
                    key=lambda pair: (-float(pair[0]), pair[1]["Rank"]))]

        # Rank is renumbered by position; Time travels with its answer, because
        # the field is "time to generate this answer", not "time for this slot".
        out.append({
            "Q_ID": q_id,
            "Video_ID": record["Video_ID"],
            "Answers": [
                {"Rank": position, "Answer": a["Answer"], "Time": a["Time"]}
                for position, a in enumerate(ordered, 1)
            ],
        })

    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=4, ensure_ascii=False)
        handle.write("\n")

    print(f"wrote {len(out)} entries to {args.out.name} (ordered by "
          f"{args.by})")

    if missing_signal:
        print(f"  {missing_signal} queries had no usable '{args.by}' signal "
              f"and kept their submitted order")

    # Answer-set identity: the multiset of strings must be unchanged.
    before = {r["Q_ID"]: sorted(a["Answer"] for a in r["Answers"]) for r in run}
    after = {r["Q_ID"]: sorted(a["Answer"] for a in r["Answers"]) for r in out}
    drift = [q for q in before if before[q] != after.get(q)]

    if drift:
        print(f"  FAIL: {len(drift)} queries changed their answer set, e.g. "
              f"{drift[:5]}", file=sys.stderr)
        return 1

    print(f"  answer sets identical for all {len(out)} queries")

    return 0


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Measure whether the answering pass's confidence predicts "
                    "correctness, and re-rank by it."
    )
    sub = ap.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze",
                             help="correlation and rank-quality curves")
    analyze.add_argument("--per-response", type=Path,
                         default=DEFAULT_PER_RESPONSE,
                         help="score.answers.py's --per-response-dir output")
    analyze.add_argument("--report-json", type=Path,
                         default=ROOT / "ranking.analysis.2025.json")
    analyze.add_argument("--report-md", type=Path,
                         default=ROOT / "ranking.analysis.2025.md")
    analyze.set_defaults(func=cmd_analyze)

    rerank = sub.add_parser("rerank", help="emit a re-ordered run file")
    rerank.add_argument("--run", type=Path, required=True)
    rerank.add_argument("--out", type=Path, required=True)
    rerank.add_argument("--cache", type=Path,
                        help="work cache carrying score/votes; required for "
                             "--by score|votes")
    rerank.add_argument("--gt", type=Path, default=DEFAULT_GT,
                        help="ground truth; required for --by oracle")
    rerank.add_argument("--by", choices=KEYS, default="current")
    rerank.add_argument("--seed", type=int, default=DEFAULT_SEED)
    rerank.set_defaults(func=cmd_rerank)

    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

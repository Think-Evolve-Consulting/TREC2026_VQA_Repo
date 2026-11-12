#!/usr/bin/env python3
"""Score a Task 1 run against ground truth: METEOR, BERTScore, NDCG, STS_local.

Phase C of ``VQA/EVALUATION_PLAN.md``. Reads the 2026 submission JSON natively
(``[{"Q_ID", "Video_ID", "Answers": [{"Rank", "Answer", "Time"}, ...]}]``)
rather than ``eval.vqa.ag.py``'s headerless CSV, and scores several runs in one
pass so they can be compared on identical queries.

**The aggregation is a deliberate reimplementation of ``eval.vqa.ag.py``, not
an improvement on it.** Anything else would produce numbers comparable to
nothing:

- normalisation is ``' '.join(text.strip().lower().split())``      (:35)
- METEOR is ``meteor_score([word_tokenize(ref)], word_tokenize(hyp))`` (:109)
- a query scores the **mean over all its answers**, then the mean over
  queries                                                      (:125-126, :150)
- NDCG uses the metric as *gain* against a strictly decreasing score vector
  encoding the submitted order                                    (:37-51)
- the 18 hard-coded ``SKIP`` Q_IDs are honoured                        (:91)
- **a ground-truth row with no matching responses scores 0 and still counts**
  (:137-142) -- which is why this must be handed the sampled GT CSV, never the
  full 2,000-row one. Passing the full file to a 150-query run would average in
  1,850 zeros.

Three departures, all deliberate:

**BERTScore is computed once over every pair** instead of once per query.
``bert_score.score()`` reloads roberta-large on each call, and the default
``rescale_with_baseline=False`` means a pair's F1 does not depend on what else
is in the batch -- so this is the same number in minutes rather than hours.

**STS is ``STS_local``, and is not the official metric.** ``VQA/calc.sts.pl``
shells out to ``http://swoogle.umbc.edu/SimService/``, a 2013 UMBC research
service that no longer answers; a full run would make 20,000 requests to a dead
host and return NaN for every one. The substitute is cosine similarity over
``all-mpnet-base-v2`` sentence embeddings. It is *a* semantic textual similarity
measure, but it is a different function on a different scale, so it is reported
under a distinct name and must never be quoted as "STS". It is off by default.

**Runs are scored together** so per-query paired deltas and a Wilcoxon signed-
rank test fall out for free -- the comparison the sample exists to support.

Usage:
    python VQA/score.answers.py --gt vqa.gt.2025.sample150.csv \\
        --run gemma4=gemma4.run.2025.json \\
        --run qwen3vl=qwen3vl.run.2025.json \\
        --run cosmos=cosmos.run.2025.json
    python VQA/score.answers.py --gt ... --run ... --oracle       # sanity
    python VQA/score.answers.py --gt ... --run ... --emit-csv parity/ --limit 50
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# eval.vqa.ag.py:91 -- videos that were not downloadable in 2025.
SKIP = {18, 27, 62, 222, 448, 507, 584, 629, 903, 1007, 1159, 1260, 1330,
        1381, 1554, 1691, 1755, 1822}

DEFAULT_STS_MODEL = "sentence-transformers/all-mpnet-base-v2"


def normalize_text(text: str) -> str:
    """eval.vqa.ag.py:35, character for character."""
    return " ".join(str(text).strip().lower().split())


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def ndcg_from_gains(gains, k=None) -> float:
    """eval.vqa.ag.py:37-51, unchanged."""
    from sklearn.metrics import ndcg_score

    gains = np.asarray(gains, dtype=float)
    m = gains.size

    if m == 0:
        return 0.0
    if m == 1:
        return 1.0

    sys_scores = np.arange(m, 0, -1, dtype=float)

    return float(ndcg_score(gains.reshape(1, -1), sys_scores.reshape(1, -1), k=k))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_gt(path: Path) -> list[dict]:
    """Headerless ``Q_ID,Video_ID,Question,Answer``, in file order."""
    rows: list[dict] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row or not row[0].strip() or len(row) < 4:
                continue

            rows.append({
                "Q_ID": row[0].strip(),
                "Video_ID": row[1].strip(),
                "Question": row[2].strip(),
                "Answer": normalize_text(row[3]),
            })

    return rows


def load_run(path: Path) -> dict[str, dict]:
    """Q_ID -> {Video_ID, answers ranked as submitted}.

    Accepts the 2026 JSON run format, or the 5-column headerless CSV that
    ``eval.vqa.ag.py`` reads, so a converted run can be scored either way.
    """
    if path.suffix.lower() == ".csv":
        by_qid: dict[str, dict] = {}

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                if not row or len(row) < 4:
                    continue

                q_id = row[0].strip()
                entry = by_qid.setdefault(q_id, {"Video_ID": row[1].strip(),
                                                 "rows": []})
                entry["rows"].append((int(row[2]), row[3]))

        return {q_id: {"Video_ID": entry["Video_ID"],
                       "answers": [a for _, a in sorted(entry["rows"])]}
                for q_id, entry in by_qid.items()}

    run: dict[str, dict] = {}

    for record in load_json(path):
        answers = sorted(record.get("Answers") or [],
                         key=lambda a: int(a["Rank"]))
        run[str(record["Q_ID"]).strip()] = {
            "Video_ID": str(record.get("Video_ID", "")).strip(),
            "answers": [a["Answer"] for a in answers],
        }

    return run


def parse_run_spec(spec: str) -> tuple[str, Path]:
    """``name=path`` or just ``path`` (name taken from the stem)."""
    if "=" in spec:
        name, _, raw = spec.partition("=")
        return name.strip(), Path(raw.strip())

    path = Path(spec)

    return path.stem.replace(".run", "").replace(".", "_"), path


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ensure_nltk() -> None:
    """METEOR needs wordnet; word_tokenize needs punkt."""
    import nltk

    for resource, probe in (("wordnet", "corpora/wordnet"),
                            ("omw-1.4", "corpora/omw-1.4"),
                            ("punkt", "tokenizers/punkt"),
                            ("punkt_tab", "tokenizers/punkt_tab")):
        try:
            nltk.data.find(probe)
        except LookupError:
            print(f"  downloading nltk {resource}")
            nltk.download(resource, quiet=True)


def meteor_all(pairs: list[tuple[str, str]]) -> list[float]:
    """(reference, hypothesis) -> METEOR, one per pair."""
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score

    scores: list[float] = []
    tokenised: dict[str, list[str]] = {}

    for ref, hyp in pairs:
        # References repeat across a query's ten answers; tokenise each once.
        if ref not in tokenised:
            tokenised[ref] = word_tokenize(ref)

        scores.append(float(meteor_score([tokenised[ref]], word_tokenize(hyp))))

    return scores


def bert_all(pairs: list[tuple[str, str]], device: str | None,
             batch_size: int) -> list[float]:
    """BERTScore F1 for every pair in one batched call."""
    from bert_score import score as bert_score_fn

    refs = [ref for ref, _ in pairs]
    cands = [hyp for _, hyp in pairs]

    _, _, f1 = bert_score_fn(cands, refs, lang="en", device=device,
                             batch_size=batch_size, verbose=True)

    return [float(v) for v in f1]


def sts_all(pairs: list[tuple[str, str]], model_name: str,
            device: str | None, batch_size: int) -> list[float]:
    """Cosine over sentence embeddings. NOT the official STS -- see module doc."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    unique = sorted({text for pair in pairs for text in pair})
    index = {text: i for i, text in enumerate(unique)}

    embeddings = model.encode(unique, batch_size=batch_size,
                              convert_to_numpy=True, normalize_embeddings=True,
                              show_progress_bar=True)

    return [float(np.dot(embeddings[index[ref]], embeddings[index[hyp]]))
            for ref, hyp in pairs]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def build_work(gt_rows: list[dict], run: dict[str, dict],
               use_skip: bool) -> tuple[list[dict], dict]:
    """One entry per scored GT row, in GT file order.

    Mirrors eval.vqa.ag.py's loop: every GT row that is not skipped is scored,
    including rows the run never answered -- those take a zero.
    """
    work: list[dict] = []
    diagnostics = {"scored": 0, "unanswered": 0, "video_mismatch": [],
                   "not_in_gt": 0, "skipped": 0}

    for row in gt_rows:
        q_id = row["Q_ID"]

        if use_skip and q_id.isdigit() and int(q_id) in SKIP:
            diagnostics["skipped"] += 1
            continue

        entry = run.get(q_id)
        responses = [normalize_text(a) for a in (entry or {}).get("answers", [])]

        if entry and entry["Video_ID"] and entry["Video_ID"] != row["Video_ID"]:
            diagnostics["video_mismatch"].append(q_id)

        if not responses:
            diagnostics["unanswered"] += 1

        work.append({"Q_ID": q_id, "Video_ID": row["Video_ID"],
                     "reference": row["Answer"], "responses": responses})
        diagnostics["scored"] += 1

    diagnostics["not_in_gt"] = len(set(run) - {row["Q_ID"] for row in gt_rows})

    return work, diagnostics


def score_run(work: list[dict], metrics: set[str], device: str | None,
              batch_size: int, sts_model: str) -> dict:
    """Per-query and aggregate scores for one run's work list."""
    pairs: list[tuple[str, str]] = []

    for item in work:
        pairs.extend((item["reference"], response)
                     for response in item["responses"])

    per_metric: dict[str, list[float]] = {}

    if pairs:
        if "meteor" in metrics:
            print(f"  METEOR over {len(pairs)} pairs")
            per_metric["METEOR"] = meteor_all(pairs)
        if "bert" in metrics:
            print(f"  BERTScore over {len(pairs)} pairs")
            per_metric["BERTScore"] = bert_all(pairs, device, batch_size)
        if "sts_local" in metrics:
            print(f"  STS_local over {len(pairs)} pairs")
            per_metric["STS_local"] = sts_all(pairs, sts_model, device,
                                              batch_size)

    # Fan the flat per-pair scores back out over the queries that produced them.
    cursor = 0
    per_query: list[dict] = []

    for item in work:
        n = len(item["responses"])
        block = {name: values[cursor:cursor + n]
                 for name, values in per_metric.items()}
        cursor += n

        record = {"Q_ID": item["Q_ID"], "Video_ID": item["Video_ID"],
                  "n_answers": n, "reference": item["reference"],
                  "responses": item["responses"], "per_answer": block}

        for name in per_metric:
            gains = block[name]
            record[name] = float(np.mean(gains)) if gains else 0.0

            if name in ("METEOR", "BERTScore"):
                record[f"NDCG_{name}"] = ndcg_from_gains(gains) if gains else 0.0

        per_query.append(record)

    aggregate: dict[str, float] = {}
    columns = [name for name in per_metric]
    columns += [f"NDCG_{n}" for n in ("METEOR", "BERTScore") if n in per_metric]

    for name in columns:
        values = [record[name] for record in per_query]
        aggregate[name] = round(float(np.mean(values)), 3) if values else 0.0

    return {"aggregate": aggregate, "per_query": per_query,
            "n_queries": len(per_query)}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_per_response(path: Path, result: dict) -> None:
    """The per-answer breakdown, shaped like eval.vqa.ag.py's own dump."""
    metrics = [name for name in ("METEOR", "BERTScore", "STS_local")
               if name in (result["per_query"][0]["per_answer"] if
                           result["per_query"] else {})]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["QueryID", "Rank", "SubmittedAnswer", "CorrectAnswer",
                         *metrics, "NDCG_METEOR", "NDCG_BERTScore"])

        for record in result["per_query"]:
            for rank, response in enumerate(record["responses"], 1):
                writer.writerow([
                    record["Q_ID"], rank, response, record["reference"],
                    *[round(record["per_answer"][m][rank - 1], 3)
                      for m in metrics],
                    round(record.get("NDCG_METEOR", 0.0), 3),
                    round(record.get("NDCG_BERTScore", 0.0), 3),
                ])


def emit_parity_csv(path: Path, run: dict[str, dict], gt_rows: list[dict],
                    limit: int | None) -> None:
    """The 5-column headerless CSV eval.vqa.ag.py reads, for cross-checking."""
    wanted = [row["Q_ID"] for row in gt_rows]

    if limit:
        wanted = wanted[:limit]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")

        for q_id in wanted:
            entry = run.get(q_id)

            if not entry:
                continue

            for rank, answer in enumerate(entry["answers"], 1):
                writer.writerow([q_id, entry["Video_ID"], rank, answer, 0.0])


def compare(results: dict[str, dict], metric: str) -> list[dict]:
    """Paired per-query deltas between every pair of runs."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    names = list(results)
    rows: list[dict] = []

    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a = {r["Q_ID"]: r.get(metric, 0.0)
                 for r in results[left]["per_query"]}
            b = {r["Q_ID"]: r.get(metric, 0.0)
                 for r in results[right]["per_query"]}
            shared = sorted(set(a) & set(b), key=lambda q: int(q))

            if not shared:
                continue

            deltas = np.array([b[q] - a[q] for q in shared])
            p_value = None

            if wilcoxon is not None and np.any(deltas != 0):
                try:
                    p_value = float(wilcoxon(deltas).pvalue)
                except ValueError:
                    p_value = None

            rows.append({
                "metric": metric,
                "baseline": left,
                "candidate": right,
                "n": len(shared),
                "mean_delta": round(float(deltas.mean()), 4),
                "candidate_better": int((deltas > 0).sum()),
                "baseline_better": int((deltas < 0).sum()),
                "tied": int((deltas == 0).sum()),
                "wilcoxon_p": round(p_value, 5) if p_value is not None else None,
            })

    return rows


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Score Task 1 runs against ground truth, reproducing "
                    "eval.vqa.ag.py's aggregation."
    )

    ap.add_argument("--gt", type=Path, required=True,
                    help="headerless Q_ID,Video_ID,Question,Answer CSV -- the "
                         "SAMPLED one, not the full 2,000 rows")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=PATH",
                    help="a run to score; repeatable")
    ap.add_argument("--out", type=Path, default=ROOT / "eval.2025.scores.json")
    ap.add_argument("--csv", type=Path, default=ROOT / "eval.2025.scores.csv")
    ap.add_argument("--per-response-dir", type=Path, default=ROOT / "eval.2025")

    ap.add_argument("--metrics", default="meteor,bert",
                    help="comma-separated: meteor, bert, sts_local "
                         "(default: %(default)s)")
    ap.add_argument("--sts-model", default=DEFAULT_STS_MODEL)
    ap.add_argument("--device", default=None,
                    help="cuda, cuda:0 or cpu (default: library's choice)")
    ap.add_argument("--batch-size", type=int, default=64)

    ap.add_argument("--no-skip", action="store_true",
                    help="do not honour eval.vqa.ag.py's 18-Q_ID SKIP set")
    ap.add_argument("--limit", type=int,
                    help="score only the first N ground-truth rows")
    ap.add_argument("--oracle", action="store_true",
                    help="also score the ground truth against itself; METEOR "
                         "and BERTScore must come back at ~1.0")
    ap.add_argument("--emit-csv", type=Path, metavar="DIR",
                    help="also write each run as the headerless CSV "
                         "eval.vqa.ag.py reads, for the parity check")

    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    metrics = {m.strip().lower() for m in args.metrics.split(",") if m.strip()}
    unknown = metrics - {"meteor", "bert", "sts_local"}

    if unknown:
        print(f"unknown metric(s): {sorted(unknown)}", file=sys.stderr)
        return 1

    if not args.run:
        print("nothing to score -- pass at least one --run", file=sys.stderr)
        return 1

    gt_rows = load_gt(args.gt)

    if args.limit:
        gt_rows = gt_rows[:args.limit]

    print(f"ground truth: {len(gt_rows)} rows from {args.gt.name}")

    if "meteor" in metrics:
        ensure_nltk()

    args.per_response_dir.mkdir(parents=True, exist_ok=True)

    if args.emit_csv:
        args.emit_csv.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}

    specs = [parse_run_spec(spec) for spec in args.run]

    if args.oracle:
        specs.append(("__oracle__", None))

    for name, path in specs:
        if name == "__oracle__":
            run = {row["Q_ID"]: {"Video_ID": row["Video_ID"],
                                 "answers": [row["Answer"]]}
                   for row in gt_rows}
            print(f"\nscoring oracle (ground truth against itself)")
        else:
            if not path.exists():
                print(f"\nskipping {name}: no file at {path}", file=sys.stderr)
                continue

            run = load_run(path)
            print(f"\nscoring {name}: {len(run)} answered queries from "
                  f"{path.name}")

        work, diag = build_work(gt_rows, run, use_skip=not args.no_skip)
        diagnostics[name] = diag

        if diag["unanswered"]:
            print(f"  {diag['unanswered']} GT rows have no response and score 0")
        if diag["video_mismatch"]:
            print(f"  WARNING: {len(diag['video_mismatch'])} Video_ID "
                  f"mismatches, e.g. Q_ID {diag['video_mismatch'][:5]}",
                  file=sys.stderr)
        if diag["not_in_gt"]:
            print(f"  {diag['not_in_gt']} answered queries are not in the GT "
                  f"file and are ignored")

        results[name] = score_run(work, metrics, args.device, args.batch_size,
                                  args.sts_model)
        results[name]["diagnostics"] = diag
        results[name]["source"] = str(path) if path else "ground truth"

        write_per_response(args.per_response_dir / f"{name}_per_response.csv",
                           results[name])

        if args.emit_csv and path:
            emit_parity_csv(args.emit_csv / f"{name}.csv", run, gt_rows,
                            args.limit)

    if not results:
        print("\nno runs scored", file=sys.stderr)
        return 1

    comparisons: list[dict] = []
    scored = {n: r for n, r in results.items() if n != "__oracle__"}

    for metric in ("METEOR", "BERTScore", "STS_local"):
        if any(metric in r["aggregate"] for r in scored.values()):
            comparisons.extend(compare(scored, metric))

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ground_truth": str(args.gt),
        "gt_rows": len(gt_rows),
        "metrics": sorted(metrics),
        "skip_honoured": not args.no_skip,
        "sts_note": "STS_local is cosine similarity over "
                    f"{args.sts_model} embeddings. It is NOT the official "
                    "UMBC STS metric, which is unreachable, and the two are "
                    "not comparable."
                    if "sts_local" in metrics else None,
        "runs": {
            name: {
                "source": result["source"],
                "n_queries": result["n_queries"],
                **result["aggregate"],
                "diagnostics": {k: (len(v) if isinstance(v, list) else v)
                                for k, v in result["diagnostics"].items()},
            }
            for name, result in results.items()
        },
        "comparisons": comparisons,
        "per_query": {name: [{k: v for k, v in record.items()
                              if k not in ("per_answer", "responses")}
                             for record in result["per_query"]]
                      for name, result in results.items()},
    }

    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)
        handle.write("\n")

    columns = sorted({key for result in results.values()
                      for key in result["aggregate"]})

    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Run", "Queries", *columns])

        for name, result in results.items():
            writer.writerow([name, result["n_queries"],
                             *[result["aggregate"].get(c, "") for c in columns]])

    print(f"\n{'run':<12} {'n':>5} " +
          " ".join(f"{c:>14}" for c in columns))

    for name, result in results.items():
        print(f"{name:<12} {result['n_queries']:5d} " +
              " ".join(f"{result['aggregate'].get(c, 0.0):14.3f}"
                       for c in columns))

    if "__oracle__" in results:
        oracle = results["__oracle__"]["aggregate"]

        # BERTScore is 1.000 on an exact match, so it is the strict check.
        # METEOR is NOT: its fragmentation penalty is 0.5 * (chunks/matches)^3,
        # which for a one-word answer matching itself is 0.5 * 1 = 0.5. The
        # 2025 ground truth is 17% single-word answers, so a perfect oracle
        # lands near 0.94, not 1.0. Anything far below that is a real fault.
        floors = {"BERTScore": 0.99, "METEOR": 0.85, "NDCG_METEOR": 0.99,
                  "NDCG_BERTScore": 0.99}
        bad = [f"{k}={v} (floor {floors[k]})" for k, v in oracle.items()
               if k in floors and v < floors[k]]

        if bad:
            print(f"\nORACLE FAILED: {', '.join(bad)} -- normalisation or "
                  f"alignment is wrong", file=sys.stderr)
            return 1

        shown = ", ".join(f"{name} {oracle[name]:.3f}"
                          for name in ("BERTScore", "METEOR")
                          if name in oracle)
        print(f"\noracle check passed: {shown} (METEOR cannot reach 1.0 -- "
              f"its fragmentation penalty costs 0.5 on a single-word answer)")

    if comparisons:
        print("\npaired comparisons")

        for row in comparisons:
            p = f"p={row['wilcoxon_p']}" if row["wilcoxon_p"] is not None \
                else "p=n/a"
            print(f"  {row['metric']:<10} {row['candidate']} - "
                  f"{row['baseline']}: {row['mean_delta']:+.4f}  "
                  f"({row['candidate_better']}/{row['n']} better, {p})")

    print(f"\nwrote {args.out.name}, {args.csv.name} and "
          f"{args.per_response_dir.name}/*_per_response.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())

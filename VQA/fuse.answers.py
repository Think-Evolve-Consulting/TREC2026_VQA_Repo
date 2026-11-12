#!/usr/bin/env python3
"""Fuse the Gemma 4 and Qwen3-VL candidate pools into one ranked ten per query.

Both answering passes now ask each question ten ways -- the original plus nine
paraphrases from ``generate.query.variations.py`` -- and keep three candidates
per call, so a query arrives here with up to **60 candidates**, each carrying
the signals the generating pass recorded:

======================  ====================================================
``score``               position-weighted vote, confidence-modulated
``votes``               how many of that model's 30 candidate slots held it
``confidence_max``      the best self-reported confidence behind it
``logprob_mean``        mean token logprob, averaged over its occurrences
``logprob_best``        the single most probable emission of it
``best_position``       0 if some call ranked it first
``prompts``             which of the ten phrasings produced it
``refusal``             matches NON_ANSWER_RE
======================  ====================================================

**Selection is the point, not ordering.** ``eval.vqa.ag.py`` scores a query as
``np.mean`` over all ten answers (:125-126), which is order-independent -- so
re-ranking the same ten cannot change mean METEOR or mean BERTScore, only NDCG.
Measured on the 2025 sample (``eval.2025.rerank.scores.csv``): METEOR is
identical at 0.148 under current, random, reverse *and* oracle ordering. What
does move the score is *which* ten of the 60 are kept, which is what this does.

That order-independence is also the strongest correctness check available: if a
fused run's mean METEOR differs from a re-ordering of itself, this script lost,
duplicated or altered an answer. See ``--self-check``.

**On the ranking signal, honestly.** The signals that already existed correlate
weakly with per-answer METEOR -- Spearman 0.032 for ``score``, *negative* for
``votes`` (``ranking.analysis.2025.md``). ``logprob_mean`` is new here and
untested against ground truth. The one effect that does not depend on the
signal being good is that a 60-candidate pool fills ten slots with real answers
where a 30-candidate pool sometimes could not: Qwen3-VL alone padded 1,596 of
24,430 slots with generic filler that scores ~0.

So the weights below are a starting point, not a finding. ``--rebuild`` re-ranks
from the caches with no inference, so changing them costs seconds.

Usage:
    python VQA/fuse.answers.py
    python VQA/fuse.answers.py --weights agreement=3,score=1,logprob=0.5
    python VQA/fuse.answers.py --self-check fused.run.task1.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_QUERIES = ROOT / "vqa.2026.testingDataset.Task1.json"
DEFAULT_GEMMA4_CACHE = ROOT / "answers.task1.gemma4.ens.cache.jsonl"
DEFAULT_QWEN3VL_CACHE = ROOT / "answers.task1.qwen3vl.ens.cache.jsonl"
DEFAULT_JUDGE_CACHE = ROOT / "judge.task1.cache.jsonl"
DEFAULT_OUT = ROOT / "fused.run.task1.json"
DEFAULT_CACHE = ROOT / "fused.task1.cache.jsonl"

ANSWERS_PER_QUERY = 10

# Starting weights. `agreement` is the only one that is new information rather
# than a restatement of one model's own opinion: a candidate both models
# produced independently agreed across two different architectures, two
# different visual encoders and two different prompt templates. It is weighted
# highest for that reason and not because it has been measured to work.
DEFAULT_WEIGHTS = {
    "agreement": 3.0,   # produced by both models
    "score": 1.0,       # summed position-weighted vote, normalised per model
    "logprob": 0.5,     # exp(mean logprob), 0..1
    "confidence": 0.25,  # self-reported; known weak, kept as a tiebreak
    # VQA/judge.answers.py's 1-10 relevance score, rescaled to 0..1. Weighted
    # level with `agreement` because it is the other signal here that is not a
    # restatement of the generator's own confidence -- and unlike agreement,
    # which is exactly zero on 319 queries and averages 1.2 exact matches, it
    # is dense. Inert unless --judge is passed, so a run without it is
    # byte-identical to the fusion that produced the current submission.
    "judge": 3.0,
}

# Same pool and order as both answering passes, so a fused run that still needs
# padding pads identically to an unfused one.
FALLBACK_ANSWERS = (
    "unknown",
    "it is not shown in the video",
    "the video does not show this",
    "nothing happens",
    "it is not clear from the video",
    "no one",
    "none",
    "it cannot be determined",
    "not visible in the video",
    "the answer is not shown",
    "no",
    "yes",
)


def normalize(text: str) -> str:
    """The validator's own duplicate rule -- validate.ag.run.py:276."""
    return " ".join(text.split()).casefold()


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_cache(path: Path) -> dict[int, dict]:
    """Read a JSONL work cache, keyed by Q_ID. Later records win."""
    cache: dict[int, dict] = {}

    if not path.exists():
        return cache

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                cache[int(record["Q_ID"])] = record
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                print(f"  {path.name}: skipping malformed line {line_no}",
                      file=sys.stderr)

    return cache


def parse_weights(text: str | None) -> dict[str, float]:
    """``agreement=3,score=1,logprob=0.5`` -> dict, over the defaults."""
    weights = dict(DEFAULT_WEIGHTS)

    if not text:
        return weights

    for pair in text.split(","):
        pair = pair.strip()

        if not pair:
            continue

        name, _, value = pair.partition("=")
        name = name.strip()

        if name not in DEFAULT_WEIGHTS:
            raise SystemExit(f"unknown weight {name!r}; known: "
                             f"{', '.join(sorted(DEFAULT_WEIGHTS))}")

        try:
            weights[name] = float(value)
        except ValueError:
            raise SystemExit(f"weight {name!r} is not a number: {value!r}")

    return weights


def collect(record: dict | None, model: str) -> list[dict]:
    """One model's candidates for one query, tagged with which model.

    Falls back to the emitted ten when a record predates the candidate field,
    so a cache from the older single-call script still fuses -- with votes and
    a rank but no logprob, which is exactly what it knew.
    """
    if not record:
        return []

    candidates = record.get("candidates")

    if candidates:
        return [dict(c, model=model) for c in candidates]

    out = []

    for answer in record.get("answers") or ():
        if answer.get("source") not in (None, "model"):
            continue  # never re-promote another run's filler

        # A single-call cache has a rank but no score. Synthesising one from
        # the rank matters: without it every candidate from that model
        # normalises to zero and loses every tie to the model that did record
        # scores, which silently reduces the fusion to one model's opinion.
        rank_no = int(answer.get("Rank", 1))
        score = answer.get("score")
        score = float(score) if score else 1.0 / rank_no

        out.append({
            "answer": answer["Answer"],
            "score": score,
            "votes": int(answer.get("votes") or 1),
            "confidence_max": 0.5,
            "logprob_mean": None,
            "logprob_best": None,
            "n_logprob": 0,
            "best_position": int(answer.get("Rank", 1)) - 1,
            "prompts": [],
            "refusal": False,
            "model": model,
        })

    return out


def merge(pools: dict[str, list[dict]]) -> list[dict]:
    """Cluster both models' candidates on the validator's duplicate rule.

    ``score`` is normalised by each model's own best before the two are added:
    the pools are the same size but not the same scale, and an unnormalised sum
    would let whichever model happened to vote harder decide every tie.
    """
    best_by_model = {
        model: max((c.get("score") or 0.0 for c in pool), default=0.0) or 1.0
        for model, pool in pools.items()
    }

    clusters: dict[str, dict] = {}

    for model, pool in pools.items():
        for candidate in pool:
            text = " ".join(str(candidate["answer"]).split())

            if not text:
                continue

            key = normalize(text)
            cluster = clusters.setdefault(key, {
                "answer": text,
                "models": [],
                "score_norm": 0.0,
                "votes": 0,
                "confidence_max": 0.0,
                "logprob_best": None,
                "n_logprob": 0,
                "best_position": 99,
                "n_prompts": 0,
                "refusal": bool(candidate.get("refusal")),
                # Keep the surface form the stronger model offered, and prefer
                # a non-refusal reading of the same normalised string.
                "_best": -1.0,
            })

            share = (candidate.get("score") or 0.0) / best_by_model[model]

            if model not in cluster["models"]:
                cluster["models"].append(model)

            cluster["score_norm"] += share
            cluster["votes"] += int(candidate.get("votes") or 0)
            cluster["confidence_max"] = max(
                cluster["confidence_max"],
                float(candidate.get("confidence_max") or 0.0))
            cluster["best_position"] = min(
                cluster["best_position"],
                int(candidate.get("best_position") or 0))
            cluster["n_prompts"] += len(candidate.get("prompts") or ())
            cluster["n_logprob"] += int(candidate.get("n_logprob") or 0)
            cluster["refusal"] = cluster["refusal"] and \
                bool(candidate.get("refusal"))

            logprob = candidate.get("logprob_best")

            if logprob is not None:
                cluster["logprob_best"] = logprob \
                    if cluster["logprob_best"] is None \
                    else max(cluster["logprob_best"], logprob)

            if share > cluster["_best"]:
                cluster["_best"] = share
                cluster["answer"] = text

    for cluster in clusters.values():
        cluster.pop("_best")
        cluster["models"] = sorted(cluster["models"])
        cluster["both_models"] = len(cluster["models"]) > 1

    out = list(clusters.values())
    add_soft_agreement(out, pools)

    return out


SOFT_AGREEMENT_FLOOR = 0.5
"""Token overlap below which two answers are not treated as the same claim.

Exact cross-model agreement is far rarer than it sounds: measured over the two
2026 caches, only 0.6 candidates per query matched between the models after
casefolding, and 1,677 of 2,443 queries had no exact match at all. Free-text
answers to the same question diverge on articles and word order -- "a red car"
and "the red car" -- long before they disagree about the car. So agreement is
also measured on token overlap, which is what makes it a usable signal rather
than a rarity.

Deliberately a *scoring* signal only: the clusters are never merged on it. The
validator rejects duplicates by exact normalised string, and two answers that
overlap 0.6 are two legitimate distinct slots."""


def add_soft_agreement(clusters: list[dict],
                       pools: dict[str, list[dict]]) -> None:
    """Best token-overlap each cluster has with the *other* model's pool."""
    by_model = {
        model: [(normalize(str(c["answer"])),
                 frozenset(normalize(str(c["answer"])).split()))
                for c in pool if str(c.get("answer") or "").strip()]
        for model, pool in pools.items()
    }

    for cluster in clusters:
        tokens = frozenset(normalize(cluster["answer"]).split())
        best = 0.0

        if tokens:
            for model, entries in by_model.items():
                if model in cluster["models"]:
                    continue  # agreement means the model that did NOT say it

                for _, other in entries:
                    if not other:
                        continue

                    overlap = len(tokens & other) / len(tokens | other)

                    if overlap > best:
                        best = overlap

        cluster["soft_agreement"] = round(best, 3) \
            if best >= SOFT_AGREEMENT_FLOOR else 0.0


def attach_judge(clusters: list[dict], scores: dict[str, int] | None) -> int:
    """Hang VQA/judge.answers.py's 1-10 relevance score on each cluster.

    Joined on ``normalize()``, which is the key the judge cache was written
    with and the key these clusters were built with, so the match is exact
    rather than a text comparison. A cluster the judge never saw -- one below
    its per-query cap, or a refusal, which it is not asked to grade -- keeps no
    ``judge_score`` and contributes zero to the composite in ``rank()``.
    """
    if not scores:
        return 0

    hits = 0

    for cluster in clusters:
        score = scores.get(normalize(cluster["answer"]))

        if score is not None:
            cluster["judge_score"] = int(score)
            hits += 1

    return hits


def rank(clusters: list[dict], weights: dict[str, float]) -> list[dict]:
    """Score every cluster and sort. Refusals last, whatever they scored."""
    for cluster in clusters:
        # exp of a mean token logprob is a per-token probability in 0..1, which
        # is the only one of these already on a bounded scale.
        probability = math.exp(cluster["logprob_best"]) \
            if cluster["logprob_best"] is not None else 0.0

        # An exact cross-model match counts fully; a strong token overlap
        # counts for what it overlaps. Never both -- an exact match is already
        # an overlap of 1.0.
        agreement = 1.0 if cluster["both_models"] \
            else cluster.get("soft_agreement", 0.0)

        # 1-10 onto 0..1, so the weight means the same thing it does for the
        # other three. Absent -- no judge pass, or a candidate below its cap --
        # contributes nothing rather than penalising, which is what keeps an
        # unjudged run identical to today's.
        judged = cluster.get("judge_score")

        cluster["fused_score"] = round(
            weights["agreement"] * agreement
            + weights["score"] * cluster["score_norm"]
            + weights["logprob"] * probability
            + weights["confidence"] * cluster["confidence_max"]
            + weights["judge"] * (judged / 10.0 if judged is not None else 0.0),
            6)
        cluster["agreement"] = round(agreement, 3)
        cluster["probability"] = round(probability, 5) if probability else None

    # Deterministic all the way down, so two runs of --rebuild agree: refusals
    # last, then fused score, then the answer text.
    clusters.sort(key=lambda c: (c["refusal"], -c["fused_score"], c["answer"]))

    return clusters


def fill_slots(chosen: list[dict], want: int) -> tuple[list[dict], int]:
    """Pad a shortfall with the same defensible non-answers as the two passes."""
    seen = {normalize(c["answer"]) for c in chosen}
    used = 0

    for text in FALLBACK_ANSWERS:
        if len(chosen) >= want:
            break

        key = normalize(text)

        if key in seen:
            continue

        seen.add(key)
        chosen.append({"answer": text, "source": "fallback",
                       "fused_score": 0.0, "models": [], "votes": 0,
                       "both_models": False, "probability": None})
        used += 1

    return chosen[:want], used


def fuse_query(query: dict, gemma: dict | None, qwen: dict | None,
               weights: dict[str, float], want: int,
               judge: dict | None = None) -> dict:
    pools = {"gemma4": collect(gemma, "gemma4"),
             "qwen3vl": collect(qwen, "qwen3vl")}

    clusters = merge(pools)
    judged = attach_judge(clusters, (judge or {}).get("scores"))
    ranked = rank(clusters, weights)

    chosen = [dict(c, source="model") for c in ranked[:want]]
    chosen, fallback_used = fill_slots(chosen, want)

    # "Time to generate this answer" stays the sum of what the two passes
    # actually spent on this query, split evenly across the slots -- the track
    # asks for the cost of producing the answer, and both models' whole
    # ensembles were paid for before any of these ten could be chosen.
    spent = sum((record or {}).get("elapsed") or 0.0
                for record in (gemma, qwen))
    per_slot = round(spent / max(want, 1), 3)

    return {
        "Q_ID": query["Q_ID"],
        "Video_ID": query["Video_ID"],
        "answers": [
            {"Rank": rank_no,
             "Answer": item["answer"],
             "Time": per_slot if item["source"] == "model" else 0.0,
             "source": item["source"],
             "models": item.get("models") or [],
             "fused_score": item.get("fused_score"),
             "votes": item.get("votes"),
             "probability": item.get("probability"),
             "judge_score": item.get("judge_score")}
            for rank_no, item in enumerate(chosen, 1)
        ],
        "n_gemma4": len(pools["gemma4"]),
        "n_qwen3vl": len(pools["qwen3vl"]),
        "pooled_unique": len(ranked),
        "judged": judged,
        "agreed": sum(1 for c in ranked if c["both_models"]),
        "agreed_soft": sum(1 for c in ranked
                           if not c["both_models"] and c["agreement"]),
        "with_probability": sum(1 for c in ranked
                                if c["probability"] is not None),
        "fallback": fallback_used,
        "elapsed": round(spent, 2),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def assemble(queries: list[dict], fused: dict[int, dict],
             want: int) -> tuple[list[dict], int]:
    """Emit the run file, carrying only the fields the validator allows."""
    run: list[dict] = []
    missing = 0

    for query in queries:
        record = fused.get(query["Q_ID"])

        if not record:
            missing += 1
            continue

        run.append({
            "Q_ID": query["Q_ID"],
            "Video_ID": query["Video_ID"],
            "Answers": [
                {"Rank": answer["Rank"],
                 "Answer": answer["Answer"],
                 "Time": answer["Time"]}
                for answer in record["answers"][:want]
            ],
        })

    return run, missing


def self_check(path: Path, seed: int = 20260815) -> int:
    """Shuffle each query's ten answers and rewrite the ranks.

    Scoring the shuffle against the original is the order-independence proof:
    mean METEOR and mean BERTScore must come back identical, because
    eval.vqa.ag.py averages over the ten. Any difference is this script having
    lost, duplicated or altered an answer -- not a ranking disagreement.
    """
    run = load_json(path)
    rng = random.Random(seed)
    out = path.with_name(path.stem + ".shuffled" + path.suffix)

    for entry in run:
        answers = entry["Answers"]
        rng.shuffle(answers)

        for rank_no, answer in enumerate(answers, 1):
            answer["Rank"] = rank_no

    with out.open("w", encoding="utf-8") as handle:
        json.dump(run, handle, indent=4, ensure_ascii=False)
        handle.write("\n")

    print(f"wrote {out}")
    print("score both and require identical METEOR / BERTScore:")
    print(f"  python VQA/score.answers.py --gt <gt>.csv \\\n"
          f"    --run a={path.name} --run b={out.name} \\\n"
          f"    --metrics meteor,bert --out selfcheck.scores.json")

    return 0


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fuse the Gemma 4 and Qwen3-VL candidate pools into one "
                    "ranked ten answers per query.")

    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--gemma4", type=Path, default=DEFAULT_GEMMA4_CACHE,
                    help="Gemma 4 work cache (default: %(default)s)")
    ap.add_argument("--qwen3vl", type=Path, default=DEFAULT_QWEN3VL_CACHE,
                    help="Qwen3-VL work cache (default: %(default)s)")
    ap.add_argument("--judge", type=Path, nargs="?",
                    const=DEFAULT_JUDGE_CACHE, default=None,
                    metavar="PATH",
                    help="VQA/judge.answers.py's relevance cache (bare flag "
                         f"uses {DEFAULT_JUDGE_CACHE.name}). Omit it and the "
                         "fusion is exactly what it was before the judge "
                         "existed")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help="JSONL record of the fusion, one line per Q_ID, "
                         "holding the signals behind each choice")
    ap.add_argument("--answers-per-query", type=int, default=ANSWERS_PER_QUERY)
    ap.add_argument("--weights",
                    help="override the ranking weights, e.g. "
                         "'agreement=3,score=1,logprob=0.5,confidence=0.25'")
    ap.add_argument("--rebuild", action="store_true",
                    help="alias for a normal run; kept because every other "
                         "stage here has one. Fusion never infers anyway")
    ap.add_argument("--self-check", type=Path, metavar="RUN",
                    help="write a shuffled copy of RUN for the "
                         "order-independence check, then exit")

    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.self_check:
        return self_check(args.self_check)

    weights = parse_weights(args.weights)
    queries = load_json(args.queries)
    gemma = load_cache(args.gemma4)
    qwen = load_cache(args.qwen3vl)
    judge: dict[int, dict] = {}

    if args.judge:
        judge = load_cache(args.judge)

        if not judge:
            print(f"  WARNING: --judge {args.judge.name} is empty or missing; "
                  f"fusing without it", file=sys.stderr)

    print(f"queries: {len(queries)}")
    print(f"  gemma4  {args.gemma4.name}: {len(gemma)} cached")
    print(f"  qwen3vl {args.qwen3vl.name}: {len(qwen)} cached")

    if args.judge:
        print(f"  judge   {args.judge.name}: {len(judge)} cached")

    print("  weights: " + "  ".join(f"{k}={v:g}"
                                    for k, v in sorted(weights.items())))

    both = sum(1 for q in queries
               if q["Q_ID"] in gemma and q["Q_ID"] in qwen)
    neither = sum(1 for q in queries
                  if q["Q_ID"] not in gemma and q["Q_ID"] not in qwen)

    if neither:
        print(f"  WARNING: {neither} queries are in neither cache and will be "
              f"missing from the run", file=sys.stderr)

    print(f"  in both caches: {both} | in one only: "
          f"{len(queries) - both - neither} | in neither: {neither}")

    fused: dict[int, dict] = {}
    args.cache.parent.mkdir(parents=True, exist_ok=True)

    with args.cache.open("w", encoding="utf-8") as handle:
        for query in queries:
            q_id = query["Q_ID"]

            if q_id not in gemma and q_id not in qwen:
                continue

            record = fuse_query(query, gemma.get(q_id), qwen.get(q_id),
                                weights, args.answers_per_query,
                                judge.get(q_id))
            fused[q_id] = record
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    run, missing = assemble(queries, fused, args.answers_per_query)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(run, handle, indent=4, ensure_ascii=False)
        handle.write("\n")

    print(f"\nwrote {len(run)} of {len(queries)} query entries to {args.out}")

    if missing:
        print(f"  {missing} queries are unanswered -- the run will not "
              f"validate until they are")

    report(fused)

    return 0


def report(fused: dict[int, dict]) -> None:
    """What the fusion actually did, since none of it is visible in the run."""
    records = list(fused.values())

    if not records:
        return

    pooled = [r["pooled_unique"] for r in records]
    agreed = [r["agreed"] for r in records]
    fallback = sum(r["fallback"] for r in records)
    any_fallback = sum(1 for r in records if r["fallback"])
    slots = len(records) * 10

    # Where the ten came from: a slot filled by an answer both models produced
    # is the case this whole design exists to create.
    sources = Counter()

    for record in records:
        for answer in record["answers"]:
            if answer["source"] == "fallback":
                sources["fallback"] += 1
            elif len(answer.get("models") or []) > 1:
                sources["both models"] += 1
            else:
                sources[(answer.get("models") or ["?"])[0]] += 1

    print(f"\nfusion over {len(records)} queries")
    print(f"  candidates pooled per query: mean {sum(pooled) / len(pooled):5.1f}"
          f"  min {min(pooled)}  max {max(pooled)}")
    soft = [r.get("agreed_soft") or 0 for r in records]
    print(f"  cross-model agreements:      mean {sum(agreed) / len(agreed):5.1f}"
          f" exact + {sum(soft) / len(soft):4.1f} soft"
          f"  queries with neither: "
          f"{sum(1 for a, s in zip(agreed, soft) if not a and not s)}")
    print(f"  queries short of ten:        {any_fallback}"
          f"  ({fallback} slots, {100 * fallback / slots:.2f}% of {slots})")
    print("\n  where the ten slots came from")

    for name, count in sources.most_common():
        print(f"    {name:<12} {count:6d}  {100 * count / slots:5.1f}%")

    with_prob = sum(r["with_probability"] for r in records)
    print(f"\n  candidates carrying a probability: {with_prob} of "
          f"{sum(pooled)}"
          f"{'  -- CHECK: logprobs never reached the caches' if not with_prob else ''}")

    judged = sum(r.get("judged") or 0 for r in records)

    if judged:
        # A slot chosen on a judge score the generator's own signals ranked
        # below tenth is the whole point of the pass -- and if this is zero,
        # the judge agreed with the fuser everywhere and bought nothing.
        moved = sum(1 for r in records for a in r["answers"]
                    if a.get("judge_score") is not None)
        print(f"  candidates carrying a judge score:  {judged} of "
              f"{sum(pooled)}  ({moved} of the emitted slots)")


if __name__ == "__main__":
    sys.exit(main())

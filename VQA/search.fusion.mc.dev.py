#!/usr/bin/env python3
"""Search fusions of the Task 2 dev runs for one that beats Qwen3.8 -- honestly.

Every ranking pass has a run over the same 400 labelled 2025 queries
(``eval.2025.mc/<name>.dev.run.json``). This script fuses them in every way it
can afford -- every subset and every integer weighting 0..4 of up to eight
runs, by reciprocal-rank fusion and by Borda count -- and asks whether any
fusion ranks better than Qwen3.8 alone (0.800 Top-1 / 0.884 MRR).

**The in-sample answer is always yes, and means nothing.** Searching ~780,000
weightings on 400 queries will find one that beats any fixed baseline by
chance, the way the 8:8:4:1 weights of the first fusion round were picked on
the very queries they were scored on. So the verdict comes from **nested
cross-validation**: the queries are split into ten folds, the best fusion is
chosen on nine and scored on the held-out tenth, every query is held out once,
and the whole thing is repeated over twenty random splits. What that reports is
what the *procedure* "search, then use the winner" would score on queries it
had not seen -- which is the question a 2026 submission actually asks.

The in-sample winners are printed too, next to their CV scores, so the size of
the optimism is visible rather than argued.

Scoring is re-implemented in numpy for speed, and is checked first against the
official scorer's per-query diagnostic CSVs for every single run: if any query
disagrees, the script stops.

Usage:
    python VQA/search.fusion.mc.dev.py
    python VQA/search.fusion.mc.dev.py --repeats 50 --folds 10
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / "eval.2025.mc"
STEM = "sample400.seed2026"
BASELINE = "qwen38"

# Strongest first on dev, so ties and the report read in that order.
DEFAULT_MODELS = ["qwen38", "qwen3vl", "internvl", "muse", "videollama3",
                  "gemma4", "llavavideo", "cosmos"]

RRF_K = 60
WEIGHT_GRID = (0, 1, 2, 3, 4)
CHUNK = 8000

# Fused scores that tie are broken by the baseline's own order, so a fusion
# that adds nothing reduces to the baseline rather than to the official order.
TIE_EPS = 1e-9


def load_harness():
    spec = importlib.util.spec_from_file_location(
        "eval_mc_dev", ROOT / "VQA" / "eval.mc.dev.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def norm(text: str) -> str:
    return " ".join(str(text).split()).lower()


def load_tensors(models: list[str], harness):
    """ranks[m, q, o]: where model m put official option o of query q (1..4)."""
    with (DEV / f"{STEM}.queries.json").open("r", encoding="utf-8-sig") as fh:
        queries = {int(q["Q_ID"]): [o["option"] for o in q["Options"]]
                   for q in json.load(fh)}

    truth = harness.read_gt_csv(DEV / f"{STEM}.gt.csv")
    q_ids = sorted(queries)
    n_opt = {len(queries[q]) for q in q_ids}

    if n_opt != {4}:
        raise SystemExit(f"expected 4 options everywhere, found {n_opt}")

    ranks = np.zeros((len(models), len(q_ids), 4), dtype=np.int64)

    for mi, name in enumerate(models):
        path = DEV / f"{name}.dev.run.json"

        if not path.exists():
            raise SystemExit(f"missing {path}")

        run = harness.load_run(path)

        for qi, q in enumerate(q_ids):
            if q not in run:
                raise SystemExit(f"{name}: Q_ID {q} not ranked")

            options = [norm(o) for o in queries[q]]
            taken = [False] * 4

            for position, answer in enumerate(run[q]):
                matches = [i for i, o in enumerate(options)
                           if o == norm(answer) and not taken[i]]

                if not matches:
                    raise SystemExit(f"{name}: Q_ID {q} answer not an option: "
                                     f"{answer!r}")

                taken[matches[0]] = True
                ranks[mi, qi, matches[0]] = position + 1

    gt_idx = np.full(len(q_ids), -1, dtype=np.int64)

    for qi, q in enumerate(q_ids):
        options = [norm(o) for o in queries[q]]

        if norm(truth.get(q, "")) in options:
            gt_idx[qi] = options.index(norm(truth[q]))

    return q_ids, ranks, gt_idx


def features(ranks: np.ndarray) -> dict[str, np.ndarray]:
    m, q, o = ranks.shape
    return {
        "rrf": (1.0 / (RRF_K + ranks)).reshape(m, q * o),
        "borda": (o - ranks).astype(np.float64).reshape(m, q * o),
    }


def evaluate(weights: np.ndarray, feats: np.ndarray, tie: np.ndarray,
             gt_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-1 correctness and reciprocal rank, per config and per query."""
    n_cfg = weights.shape[0]
    n_q = gt_idx.shape[0]
    scores = (weights @ feats).reshape(n_cfg, n_q, 4) + tie
    has = gt_idx >= 0
    index = np.broadcast_to(np.where(has, gt_idx, 0)[None, :, None],
                            (n_cfg, n_q, 1))
    gold = np.take_along_axis(scores, index, axis=2)
    rank_gold = 1 + (scores > gold).sum(axis=2)
    top1 = (rank_gold == 1) & has[None, :]
    # The official scorer rounds each reciprocal rank to three decimals
    # (rank 3 scores 0.333), so this does too, to match it exactly.
    rr = np.where(has[None, :], np.round(1.0 / rank_gold, 3), 0.0)
    return top1, rr


def verify(models, q_ids, ranks, gt_idx, feats, tie) -> None:
    """Every single run must reproduce the official per-query diagnostics."""
    eye = np.eye(len(models))

    for mi, name in enumerate(models):
        top1, rr = evaluate(eye[mi:mi + 1], feats["borda"], tie, gt_idx)
        diag = DEV / f"{name}.dev.run.{STEM}.queries_diagnostic.csv"

        if not diag.exists():
            print(f"  {name:12s} no official diagnostic to check against "
                  f"({diag.name}) -- run eval.mc.dev.py score first",
                  file=sys.stderr)
            continue

        with diag.open("r", encoding="utf-8-sig", newline="") as fh:
            official = {int(r["QueryID"]): (int(r["Top1Correct"]),
                                            float(r["MRRScore"]))
                        for r in csv.DictReader(fh)}

        bad = [q for qi, q in enumerate(q_ids)
               if q in official and (int(top1[0, qi]) != official[q][0]
                                     or abs(rr[0, qi] - official[q][1]) > 1e-6)]

        if bad:
            raise SystemExit(f"{name}: scoring disagrees with the official "
                             f"scorer on {len(bad)} queries, e.g. {bad[:5]}")

        print(f"  {name:12s} Top-1 {top1.mean():.3f}  MRR {rr.mean():.3f}  "
              f"-- matches the official scorer on all {len(official)} queries")


def configs(family: str, n_models: int) -> np.ndarray:
    grid = (0, 1) if family == "subsets" else WEIGHT_GRID
    weights = np.array([w for w in itertools.product(grid, repeat=n_models)
                        if any(w)], dtype=np.float64)
    return weights


def describe(models, method, weights) -> str:
    parts = [f"{m}:{int(w)}" for m, w in zip(models, weights) if w]
    return f"{method} " + " ".join(parts)


def paired(top1, rr, base1, baserr, harness, seed=2026) -> dict:
    b = int(((top1 == 1) & (base1 == 0)).sum())
    c = int(((top1 == 0) & (base1 == 1)).sum())
    lo, hi = harness.bootstrap_ci(list(rr - baserr), seed, draws=4000)
    return {"top1": float(top1.mean()), "mrr": float(rr.mean()),
            "d_top1": float(top1.mean() - base1.mean()),
            "d_mrr": float(rr.mean() - baserr.mean()),
            "won": b, "lost": c, "p": harness.exact_mcnemar(b, c),
            "mrr_ci": (lo, hi)}


def search(family, models, feats, tie, gt_idx, foldmat, n_folds, base1,
           baserr, harness) -> dict:
    weights = configs(family, len(models))
    methods = list(feats)
    n_cfg = len(weights)
    n_cols = foldmat.shape[1]

    best_key = np.full(n_cols, -np.inf)
    best_cfg = np.zeros(n_cols, dtype=np.int64)
    insample_key, insample_cfg = -np.inf, 0

    for mi, method in enumerate(methods):
        for start in range(0, n_cfg, CHUNK):
            chunk = weights[start:start + CHUNK]
            top1, rr = evaluate(chunk, feats[method], tie, gt_idx)
            t1 = top1.astype(np.float32)
            rrf = rr.astype(np.float32)

            # Top-1 decides, MRR breaks ties (MRR sums stay below 1,000).
            total = t1.sum(1) * 1000.0 + rrf.sum(1)
            fold = (t1 @ foldmat) * 1000.0 + (rrf @ foldmat)
            train = total[:, None] - fold

            winner = train.argmax(0)
            better = train[winner, np.arange(n_cols)] > best_key
            best_key[better] = train[winner, np.arange(n_cols)][better]
            best_cfg[better] = mi * n_cfg + start + winner[better]

            local = int(total.argmax())
            if total[local] > insample_key:
                insample_key = float(total[local])
                insample_cfg = mi * n_cfg + start + local

    def unpack(cfg):
        return methods[cfg // n_cfg], weights[cfg % n_cfg]

    cache: dict[int, tuple] = {}

    def vectors(cfg):
        if cfg not in cache:
            method, w = unpack(cfg)
            top1, rr = evaluate(w[None, :], feats[method], tie, gt_idx)
            cache[cfg] = (top1[0].astype(int), rr[0])
        return cache[cfg]

    n_q = gt_idx.shape[0]
    repeats = n_cols // n_folds
    cv = []
    chosen: dict[str, int] = {}

    for r in range(repeats):
        cv1 = np.zeros(n_q, dtype=int)
        cvr = np.zeros(n_q)

        for f in range(n_folds):
            col = r * n_folds + f
            held = foldmat[:, col] > 0
            top1, rr = vectors(int(best_cfg[col]))
            cv1[held] = top1[held]
            cvr[held] = rr[held]
            label = describe(models, *unpack(int(best_cfg[col])))
            chosen[label] = chosen.get(label, 0) + 1

        cv.append(paired(cv1, cvr, base1, baserr, harness, seed=r))

    top1, rr = vectors(insample_cfg)
    ins = paired(top1, rr, base1, baserr, harness)
    ins["config"] = describe(models, *unpack(insample_cfg))

    return {
        "family": family, "n_configs": n_cfg * len(methods),
        "insample": ins,
        "cv_top1_mean": float(np.mean([c["top1"] for c in cv])),
        "cv_top1_sd": float(np.std([c["top1"] for c in cv])),
        "cv_mrr_mean": float(np.mean([c["mrr"] for c in cv])),
        "cv_d_top1_mean": float(np.mean([c["d_top1"] for c in cv])),
        "cv_d_mrr_mean": float(np.mean([c["d_mrr"] for c in cv])),
        "cv_beats_baseline_top1": sum(c["d_top1"] > 0 for c in cv),
        "cv_beats_baseline_mrr": sum(c["d_mrr"] > 0 for c in cv),
        "cv_p_median": float(np.median([c["p"] for c in cv])),
        "cv_repeats": len(cv),
        "most_chosen": sorted(chosen.items(), key=lambda kv: -kv[1])[:5],
        "folds_choosing_baseline_alone": chosen.get(
            describe(models, "rrf", np.eye(len(models))[0]), 0)
        + chosen.get(describe(models, "borda", np.eye(len(models))[0]), 0),
        "n_selections": repeats * n_folds,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=DEV / "fusion.search.json")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    harness = load_harness()
    models = args.models

    if models[0] != BASELINE:
        raise SystemExit(f"the first model must be the baseline, {BASELINE}")

    q_ids, ranks, gt_idx = load_tensors(models, harness)
    feats = features(ranks)
    tie = (TIE_EPS * (4 - ranks[0])).astype(np.float64)
    n_q = len(q_ids)

    print(f"{len(models)} runs x {n_q} queries; correct option among the "
          f"four for {int((gt_idx >= 0).sum())}")
    print("\nsingle runs, checked against the official scorer:")
    verify(models, q_ids, ranks, gt_idx, feats, tie)

    eye = np.eye(len(models))
    base1, baserr = (v[0] for v in evaluate(eye[0:1], feats["borda"], tie,
                                            gt_idx))
    base1 = base1.astype(int)

    # What fusion could possibly reach: the share of queries where at least
    # one run -- or one of the named few -- puts the right answer first.
    firsts = ranks[:, np.arange(n_q), np.where(gt_idx >= 0, gt_idx, 0)] == 1
    firsts &= (gt_idx >= 0)[None, :]
    print(f"\noracle ceilings: any of the {len(models)} runs right "
          f"{firsts.any(0).mean():.3f}; any of the top three "
          f"{firsts[:3].any(0).mean():.3f}; baseline {base1.mean():.3f}")

    print("\nhead to head with the baseline (queries won / lost at rank 1):")
    for mi in range(1, len(models)):
        won = int((firsts[mi] & ~firsts[0]).sum())
        lost = int((~firsts[mi] & firsts[0]).sum())
        agree = float((ranks[mi].argmin(1) == ranks[0].argmin(1)).mean())
        print(f"  {models[mi]:12s} won {won:3d}  lost {lost:3d}  "
              f"rank-1 agreement {agree:.3f}")

    rng = np.random.default_rng(args.seed)
    foldmat = np.zeros((n_q, args.repeats * args.folds), dtype=np.float32)

    for r in range(args.repeats):
        assignment = rng.permutation(n_q) % args.folds
        foldmat[np.arange(n_q), r * args.folds + assignment] = 1.0

    # A few fusions named in advance, so they carry no search at all. "top3"
    # is chosen by dev score, which is a mild peek; the rest are fixed.
    fixed = {
        "all runs, equal RRF": ("rrf", np.ones(len(models))),
        "top3 (qwen38, qwen3vl, internvl), equal RRF":
            ("rrf", np.array([1, 1, 1] + [0] * (len(models) - 3))),
        "qwen38 + internvl, equal RRF":
            ("rrf", np.array([1, 0, 1] + [0] * (len(models) - 3))),
        "qwen38 + qwen3vl, equal RRF":
            ("rrf", np.array([1, 1] + [0] * (len(models) - 2))),
        "top4 (+ muse), equal Borda":
            ("borda", np.array([1, 1, 1, 1] + [0] * (len(models) - 4))),
    }

    report: dict = {"models": models, "repeats": args.repeats,
                    "folds": args.folds, "baseline": {
                        "top1": float(base1.mean()),
                        "mrr": float(baserr.mean())}, "fixed": {},
                    "searched": []}

    print(f"\nfusions named in advance (no search; in-sample is honest here):")
    for label, (method, w) in fixed.items():
        top1, rr = evaluate(w[None, :].astype(float), feats[method], tie,
                            gt_idx)
        res = paired(top1[0].astype(int), rr[0], base1, baserr, harness)
        report["fixed"][label] = res
        print(f"  {label:44s} Top-1 {res['top1']:.3f} ({res['d_top1']:+.3f}, "
              f"p={res['p']:.2f})  MRR {res['mrr']:.3f} ({res['d_mrr']:+.4f})")

    for family in ("subsets", "grid"):
        res = search(family, models, feats, tie, gt_idx, foldmat, args.folds,
                     base1, baserr, harness)
        report["searched"].append(res)
        ins = res["insample"]

        print(f"\nsearched family '{family}': {res['n_configs']:,} fusions")
        print(f"  in-sample best   {ins['config']}")
        print(f"                   Top-1 {ins['top1']:.3f} "
              f"({ins['d_top1']:+.3f}, won {ins['won']} lost {ins['lost']}, "
              f"p={ins['p']:.3f})  MRR {ins['mrr']:.3f} "
              f"({ins['d_mrr']:+.4f})   <- optimistic by construction")
        print(f"  nested CV        Top-1 {res['cv_top1_mean']:.3f} "
              f"+/- {res['cv_top1_sd']:.3f} ({res['cv_d_top1_mean']:+.3f})  "
              f"MRR {res['cv_mrr_mean']:.3f} ({res['cv_d_mrr_mean']:+.4f})")
        print(f"                   beats the baseline on Top-1 in "
              f"{res['cv_beats_baseline_top1']}/{res['cv_repeats']} repeats, "
              f"on MRR in {res['cv_beats_baseline_mrr']}/{res['cv_repeats']}; "
              f"median McNemar p {res['cv_p_median']:.2f}")
        print(f"  chosen most often across {res['n_selections']} fold "
              f"selections:")
        for label, count in res["most_chosen"]:
            print(f"    {count:4d}  {label}")

    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

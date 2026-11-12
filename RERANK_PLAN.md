# VLM-as-judge rerank for the fused TREC 2026 Task 1 submission

**16 August 2026.** Extends `ENSEMBLE_PLAN.md`, whose pipeline completed at
**16:16 today** — `fused.expanded.run.task1.json`, VALIDATION PASSED, 0 errors.
Decides how to rerank the candidates that run discarded, against the ITI-CERTH
and HLTCOE approaches.

> **Code status: written and verified offline.** All three files compile, the
> no-`--judge` fusion is byte-identical to the shipped run, and the pool, prompt
> and cap have been dry-run against the real caches. What remains is GPU work —
> see the Runbook.

---

## Context

The completed fusion pools both models into **25.6 candidates per query**
(min 1, max 59; **62,601 across the set**) and keeps ten. The other ~38,000 are
discarded by a weighted sum of `agreement`, `score`, `logprob` and `confidence`.

Every one of those four is computed *from the same forward passes that produced
the candidates*. They measure how hard the generator's prior locked in, not
whether it was right — which is why `ranking.analysis.2025.md` puts them at
Spearman **0.032** (`score`) and **−0.073** (`votes`) against per-answer METEOR.
If the model was confidently wrong, all four are wrong in the same direction.

Measured on the completed run:

| | |
|---|---|
| candidates pooled per query | mean **25.6**, min 1, max 59 |
| non-refusal pool per query | mean **24.8**, median 24, p90 45 |
| queries short of ten | **290** (1,257 slots, 5.15% of 24,430) |
| cross-model agreement | mean **1.2 exact + 3.6 soft**; **319** queries have neither |
| slot provenance | gemma4 62.5%, qwen3vl 23.1%, both 9.2%, fallback 5.1% |

`agreement` carries the highest weight in the fuser (3.0) and is *exactly zero*
on 319 queries, averaging 1.2 exact matches elsewhere. It is the sparsest term
in the formula and the most heavily weighted.

**Intended outcome:** replace that guess with a grounded, externally-sourced
relevance score — and, unlike the rest of the ensemble plan, *measure* it first.

---

## Why ITI-CERTH's shape, not HLTCOE's

**HLTCOE is out.** Masked Pointer Cross-Entropy with rank weights is a *trained*
listwise reranker: it needs ranked labels and a training loop. Their genuinely
portable contribution — **prompt variation instead of finetuning, for candidate
diversity** — is *already implemented* here (`query.variations.task1.json`,
10 prompts × k=3, which is what produced the 25.6-candidate pool in the first
place). You already have the half of HLTCOE that fits.

**ITI-CERTH is in, with three deviations:**

1. **Listwise in one call.** They scored one image–caption pair per call. Here
   that is 62,601 calls ≈ 70 GPU-hours. One call per query with the frames
   attached once and every candidate numbered is **2,443 calls**, and the frame
   prefill is paid once instead of 25 times.
2. **Drop the majority vote.** Their polling step *is* `votes` — the one signal
   on this dataset already measured **negative**. Frequency across paraphrases
   is not evidence.
3. **Blind.** Candidates are shuffled on a per-query seed and carry no model
   attribution, so the judge cannot anchor on the fuser's order or prefer the
   phrasings of the model it happens to be.

**Ordering is not the point.** `eval.vqa.ag.py` averages over all ten answers,
so re-ordering cannot move METEOR or BERTScore — `eval.2025.rerank.scores.csv`
shows 0.148 identical under current, random, reverse *and* oracle ordering. Only
NDCG responds (0.751 → 0.933 oracle). This pays off through **selection**, on
the ~88% of queries holding more than ten candidates.

---

## Design

### `VQA/judge.answers.py` (new) — Gemma 4

Gemma 4 scored METEOR **0.148** against Qwen3-VL's **0.053** on the 2025 sample:
the stronger model here by 3×. One call per query:

- Context via the existing `build_context()` (`generate.answers.py:294`) and
  `load_transcript()` (`:966`), gated by the same `plan_for()` routing (`:366`)
  the answering pass used — including the same no-speech rescue, so the judge is
  never asked to grade on no evidence.
- Pool built with `fuse.answers.py`'s own `collect()` / `merge()` / `rank()`, so
  the judge scores exactly the clusters the fuser will later re-rank. Refusals
  excluded — they sort last regardless.
- **Cap 48.** Measured against the real pool: 48 binds on **5.4%** of queries and
  still judges **59,931 of 60,475** candidates (99.1%). The first guess of 32
  bound on 30.3%. Where it binds, the cut is made by the weak pre-signal, so the
  cache records `capped`.
- Grammar-constrained `{"scores": [n ints, 1-10]}` via `response_format`, so a
  short count or out-of-range score is unreachable rather than validated after.
- Temperature **0.0** — judging is a measurement; the diversity was bought
  upstream by the ten phrasings.
- Reuses `VLLM`'s session, retry ladder and 400-is-terminal rule by subclassing.
  `generate.answers.py` is **not edited** — it has produced a validated
  submission.

Worst-case prompt measured at **~11.2k tokens** (32 frames = 8,960 + transcript
1,231 + rubric + the longest candidate list at 2,747 chars) against
`--max-model-len 16384`. **No serve-command change needed.**

Writes `judge.task1.cache.jsonl`, resumable.

### `VQA/fuse.answers.py` — additive only

`--judge PATH`, a `judge` weight defaulting to 3.0, `attach_judge()` joining on
`normalize()`, and one term in `rank()`: `weights["judge"] * (score / 10)`.
A cluster the judge never saw contributes zero. **Verified: without `--judge`
the output is byte-identical (SHA-256 match) to the shipped
`fused.run.task1.json`.**

### `VQA/gate.judge.2025.py` (new) — three arms, one pool

`fuser` (judge term forced to 0) / `judge` / `random` (seeded), all built from
the identical candidate pool, scored with `score.answers.py`'s `meteor_all()`.

**`random` is the load-bearing arm.** Judge-vs-fuser alone cannot separate "the
judge is good" from "the fuser is bad", and given 0.032/−0.073 the latter is a
live hypothesis.

| Result | Action |
|---|---|
| judge > fuser > random | selection works; weight the judge |
| judge > random ≈ fuser | weight the judge; the old weights were noise |
| judge ≈ fuser ≈ random | **set `judge=0`, ship the existing fusion, stop** |

Reports mean METEOR, a **per-query sign test** (more sensitive than the mean at
n=150, where a gap under ~0.005 is noise), and Spearman of judge score against
per-answer METEOR for direct comparison with the 0.032/−0.073 baseline. Queries
whose pool holds ≤10 candidates force the same ten on every arm and are counted
separately.

---

## Runbook

GPU is currently **free** (0 MiB) and both servers are down. All GPU work runs
on Gemma 4 at port 8010 — **no model swaps anywhere in this plan.**

### Where to run what

**One WSL2 command, then everything else from Windows.**

| Side | What | Why |
|---|---|---|
| **WSL2** | `systemctl --user start gemma4-vllm`, and keep the window open | the server process; WSL2 tears the distro down when the last session exits |
| **Windows** (`E:\TREC_2026\.venv\Scripts\python.exe`) | **all four steps below** | — |

This is the split the completed run already used: only
`generate.answers.qwen3vl.py` ran from the WSL2 interpreter; `generate.answers.py`,
`fuse.answers.py`, `expand.answers.py` and `validate.ag.run.py` all ran from
Windows.

The reason is how each client ships frames, and it is worth not re-deriving:

- **Gemma 4** base64-encodes the JPEGs into the request body
  (`generate.answers.py:1222`), so the payload is self-contained and the client
  only needs to reach `localhost:8010` — which WSL2's port forwarding provides.
  `judge.answers.py` subclasses that same `VLLM` client, so it is a Windows
  script too.
- **Qwen3-VL** passes a *file path* the server opens itself, hence
  `--allowed-local-media-path /mnt/e/TREC_2026` in its serve command. A Windows
  client would send `E:\…` and the Linux server could not open it. Nothing in
  this plan touches Qwen3-VL.

`gate.judge.2025.py` is CPU-only and has already been run from the Windows venv
during the null test, nltk download included.

```bash
# WSL2 — bring Gemma 4 up once, then leave this window open
systemctl --user start gemma4-vllm
```

### Step 1 — 2025 candidate pool for the gate (~15 min)

All four 2025 paths explicit. `expand`/`generate` default to **2026** files and
2025 Q_IDs are 1–2000 against 2026's 1–2443; getting this wrong produces a run
that completes, validates and means nothing.

```
.venv\Scripts\python.exe VQA\generate.answers.py ^
  --queries    vqa.2025.sample150.json ^
  --variations query.variations.2025.json ^
  --routing    query.routing.2025.json ^
  --frames     videos_GT_2025\frames ^
  --transcripts videos_GT_2025\transcription_openwhisper ^
  --out   gemma4.ens.run.2025.json ^
  --cache answers.2025.gemma4.ens.cache.jsonl --workers 8
```

### Step 2 — judge the 2025 pool (~5 min)

```
.venv\Scripts\python.exe VQA\judge.answers.py ^
  --queries vqa.2025.sample150.json ^
  --routing query.routing.2025.json ^
  --frames  videos_GT_2025\frames ^
  --transcripts videos_GT_2025\transcription_openwhisper ^
  --gemma4  answers.2025.gemma4.ens.cache.jsonl ^
  --qwen3vl answers.2025.qwen3vl.ens.cache.jsonl ^
  --cache   judge.2025.cache.jsonl
```

`answers.2025.qwen3vl.ens.cache.jsonl` does not exist and is not required — the
gate tests whether the judge selects better than the weights, which one model's
pool answers. The path is passed so it picks the cache up if it is ever made.

Read the report before continuing: **if `all-equal on N queries` is more than
half, the judge is flat and adds noise, not signal.** Stop there.

### Step 3 — the gate (CPU, ~2 min)

```
.venv\Scripts\python.exe VQA\gate.judge.2025.py
```

**This is the decision point.** It prints one of three verdicts. If it says
*"NO measurable selection gain"*, stop — the current submission is already on
disk and validated.

### Step 4 — judge the 2026 set (rate check first, then ~1–1.5 h)

```
:: measure the rate on 50 queries before committing
.venv\Scripts\python.exe VQA\judge.answers.py --limit 50

:: then the rest (resumes from the cache; the 50 are not re-judged)
.venv\Scripts\python.exe VQA\judge.answers.py
```

### Step 5 — re-fuse, expand, validate (~1.6 h)

New names throughout; the shipped run is untouched.

```
.venv\Scripts\python.exe VQA\fuse.answers.py --judge ^
  --out   fused.judged.run.task1.json ^
  --cache fused.judged.task1.cache.jsonl

.venv\Scripts\python.exe VQA\expand.answers.py ^
  --run     fused.judged.run.task1.json ^
  --queries vqa.2026.testingDataset.Task1.json ^
  --routing query.routing.task1.json ^
  --out     fused.judged.expanded.run.task1.json ^
  --cache   answers.expansion.task1.fused.judged.cache.jsonl --workers 4

.venv\Scripts\python.exe VQA\validate.ag.run.py ^
  vqa.2026.testingDataset.Task1.json fused.judged.expanded.run.task1.json ^
  --warnings-as-errors ^
  --report-json fused.judged.expanded.run.validation.json
```

Total: **~3 h GPU**, of which the first 20 minutes decide whether to spend the
rest.

---

## Verification

| # | Check | Status |
|---|---|---|
| 1 | **Degradation to identity** — fusion without `--judge` reproduces the shipped run | **PASSED** — SHA-256 identical |
| 2 | Three files compile | **PASSED** |
| 3 | Pool / prompt / cap dry-run against the real caches | **PASSED** — 53,185→59,931 candidates at cap 48, 55 queries have <2 and are skipped |
| 4 | Worst-case prompt fits `--max-model-len 16384` | **PASSED** — ~11.2k tokens |
| 4b | **Gate null test** — a synthetic *random* judge must fail the gate | **PASSED** — see below |
| 5 | **The gate** — judge beats fuser *and* random on the 2025 sample | pending Step 3 |
| 6 | Judge is not flat — spread within a query, `all-equal` count | pending Step 2 |
| 7 | **Cap audit** — `capped` count in the judge cache; expected ~5% | pending Step 4 |
| 8 | Validator on the judged expanded run: 2,443 × 10, no duplicates | pending Step 5 |
| 9 | **Dead slots must not rise** above the current 1,257. An increase means the normalised-key join is broken | pending Step 5 |
| 10 | `--self-check` order-independence on the judged run | pending Step 5 |

### Two findings from the null test — read before Step 3

The gate was run end-to-end against real 2025 ground truth with a **synthetic
judge assigning uniformly random scores**, over the older 2025 caches. It
correctly refused it: judge **−0.0148** vs fuser (37W/88L, p<0.0001) and only
+0.0054 vs random. The gate detects a worthless judge. But two things fell out
that change how its output should be read:

**1. The existing fuser is not noise.** `fuser` beat `random` by **+0.0203**,
98W/30L, **p<0.0001**. Selection from the pool demonstrably works already. The
judge has to clear a real baseline, not an empty one. (Caveat: this ran on the
older single-call 2025 caches, where `collect()` synthesises `score` from
`1/rank`, so it partly restates the finding in `ranking.analysis.2025.md` that
rank-1 METEOR is 0.180 against rank-10's 0.147. Step 1 re-runs it on the true
ens pool.)

**2. Pooled Spearman ≈ 0.03 is the noise floor, not a weak signal.** The
*random* judge scored Spearman **0.0295** against per-answer METEOR — statistically
indistinguishable from `score`'s 0.032. Yet the fuser built on `score` beats
random decisively on selection. Pooled Spearman mixes within-query and
between-query variance and is close to useless as a diagnostic here.
**Judge the judge on the arm comparison and the sign test, not on the
correlation.** The correlation line is kept only for continuity with
`ranking.analysis.2025.md`.

## Fallback

`fused.expanded.run.task1.json` (validated, 0 errors, 16:16 today) and the four
earlier validated runs stay on disk untouched. Everything above writes new
names. If the gate fails or Step 5 regresses, the submission is already made.

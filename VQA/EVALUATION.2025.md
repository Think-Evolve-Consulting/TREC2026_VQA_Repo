# 2025 ground-truth evaluation (sample150)

Generated 2026-08-15T01:38:08+00:00. Ground truth `vqa.gt.2025.sample150.csv`, 150 queries.

## Results

| Run | Queries | BERTScore | METEOR | NDCG_BERTScore | NDCG_METEOR |
|---|---|---|---|---|---|
| gemma4 | 150 | 0.892 | 0.148 | 0.994 | 0.751 |
| qwen3vl | 150 | 0.873 | 0.053 | 0.994 | 0.597 |
| cosmos | 150 | 0.863 | 0.033 | 0.993 | 0.524 |

Oracle (ground truth scored against itself): BERTScore 1.000, METEOR 0.924, NDCG_BERTScore 1.000, NDCG_METEOR 1.000. Values near 1.0 confirm normalisation and alignment.

## Paired comparisons

| Metric | Baseline | Candidate | Mean Δ | Better | p |
|---|---|---|---|---|---|
| METEOR | gemma4 | qwen3vl | -0.0949 | 19/150 | 0.0 |
| METEOR | gemma4 | cosmos | -0.1157 | 11/150 | 0.0 |
| METEOR | qwen3vl | cosmos | -0.0208 | 47/150 | 0.00341 |
| BERTScore | gemma4 | qwen3vl | -0.0188 | 27/150 | 0.0 |
| BERTScore | gemma4 | cosmos | -0.0286 | 11/150 | 0.0 |
| BERTScore | qwen3vl | cosmos | -0.0098 | 47/150 | 0.0 |

## Reading the table

Gemma 4 wins on every metric, and every pairwise difference is significant
(p ≈ 0 on METEOR and BERTScore against both rivals). That is the headline, but
three mechanical effects sit underneath it and change what it means.

**Run health, which the scores alone hide:**

| Run | METEOR | BERTScore | answer words | queries needing fallback | unparsed JSON | filler slots |
|---|---|---|---|---|---|---|
| gemma4 | 0.148 | 0.892 | 4.9 | **0%** | 0% | 0.0 / 10 |
| qwen3vl | 0.053 | 0.873 | 3.3 | 8% | 2% | 0.6 / 10 |
| cosmos | 0.033 | 0.863 | 3.6 | **81%** | 20% | **5.5 / 10** |
| *ground truth* | — | — | *8.5* | — | — | — |

**1. Cosmos's score is mostly a degradation artifact, not an answer-quality
measurement.** 81% of its queries needed fallback padding and it averages 5.5
of 10 slots filled with literal non-answers ("unknown", "none"), because 20% of
its responses returned unparseable JSON. The metric averages over all ten
slots, so more than half of Cosmos's score is computed over filler. Its
per-answer quality where it *did* answer is better than 0.033 implies. The run
as submitted is still what would be scored — but "Cosmos is worse than
Qwen3-VL" is not what this row demonstrates; "Cosmos fails to emit parseable
output on short clips" is.

**2. Qwen3-VL's deficit is real.** Unlike Cosmos it is not degraded — 8%
fallback, 0.6 filler slots, 2% unparsed. Its 0.053 against Gemma 4's 0.148 is a
genuine quality-and-style gap on this corpus, not a padding artifact. This is
the finding that matters most, because the 2026 submission run cost 54.3 h of
inference.

**3. Answer length is a confound on METEOR, and it tracks the ranking.** Every
run answers far shorter than the annotators write (4.9 / 3.3 / 3.6 words against
8.5). METEOR's recall term penalises a hypothesis shorter than its reference, so
a correct 3-word answer against an 8-word reference forfeits recall it cannot
recover. Among the two non-degraded runs, length orders the scores exactly as
METEOR does. **How much of Gemma 4's lead is verbosity rather than accuracy is
not separable from this table** — it needs a length-controlled re-run to settle,
and until then the margin should be treated as an upper bound on the real
quality gap.

**4. BERTScore has almost no dynamic range here.** The oracle is 1.000 and the
worst run is 0.863, so the whole spread across three very different runs is
0.029. Differences that look negligible are still highly significant, but the
metric cannot be read as a percentage-like quality scale on this task.
NDCG_BERTScore (0.993–0.994 for all three) discriminates nothing at all and
should be dropped from future comparisons; NDCG_METEOR (0.751 / 0.597 / 0.524)
does carry signal about ranking quality.

## Answer expansion (ANSWER_EXPANSION_PLAN.md Phase 2)

The baseline runs all answer at roughly half the ground truth's length against a
metric that penalises exactly that, so `VQA/expand.answers.py` was built and run
over the same 150 queries. Three conditions, scored together:

| Condition | What it is |
|---|---|
| **A — baseline** | the run as the pipeline produced it |
| **B — expanded** | `expand.answers.py` via `gemma4:12b`, per-question-type length targets |
| **C — mechanical** | the same answers padded to the same length with contentless frame words — *no model* |

C is the control the plan insisted on: if padding with filler captures B's gain,
the metric is rewarding length and an LLM pass is not worth paying for.

### Results

| Run | words | METEOR | BERTScore | NDCG_METEOR |
|---|---|---|---|---|
| A gemma4 | 4.9 | 0.148 | 0.892 | 0.751 |
| **B gemma4** | 7.7 | **0.187** | 0.892 | **0.821** |
| C gemma4 | 8.2 | 0.172 | 0.883 | 0.797 |
| A qwen3vl | 3.3 | 0.053 | 0.873 | 0.597 |
| **B qwen3vl** | 7.1 | **0.134** | **0.884** | **0.773** |
| C qwen3vl | 7.8 | 0.100 | 0.871 | 0.720 |
| A cosmos | 3.6 | 0.033 | 0.863 | 0.524 |
| **B cosmos** | 5.6 | **0.076** | **0.871** | **0.700** |
| C cosmos | 6.2 | 0.057 | 0.864 | 0.672 |
| *oracle* | *8.2* | *0.924* | *1.000* | *1.000* |

Paired, per query (n=150):

| Metric | Comparison | gemma4 | qwen3vl | cosmos |
|---|---|---|---|---|
| METEOR | **B − A** | +0.0386 (108 better, p≈0) | +0.0807 (112, p≈0) | +0.0436 (88, p≈0) |
| METEOR | **C − A** | +0.0235 (85, p≈0) | +0.0466 (89, p≈0) | +0.0246 (63, p≈0) |
| BERTScore | **B − A** | −0.0003 (p=0.134, ns) | +0.0110 (p≈0) | +0.0078 (p≈0) |
| BERTScore | **C − A** | **−0.0089 (p≈0)** | **−0.0022 (p=0.034)** | +0.0004 (ns) |

### Verdict: ship B, but understand what it is buying

**The gate clears.** B beats A on METEOR everywhere (p ≈ 0), and reduces neither
BERTScore nor NDCG — Gemma 4's −0.0003 BERTScore is statistically
indistinguishable from zero (p = 0.134), and Qwen3-VL and Cosmos both improve.

**C is not ≈ B, so the free option loses.** B beats C on METEOR by +0.015 to
+0.034 *and* on BERTScore by +0.007 to +0.013. Mechanical padding actively
damages semantic similarity for two of three runs while the LLM pass does not.

**But the headline finding is about the metric, not the method: mechanical
padding — a transformation that adds exactly zero information — captures 56–62%
of the LLM expansion's METEOR gain.**

| Run | A | B | C | C's share of B's gain |
|---|---|---|---|---|
| gemma4 | 0.148 | 0.187 | 0.172 | **62%** |
| qwen3vl | 0.053 | 0.134 | 0.100 | **58%** |
| cosmos | 0.033 | 0.076 | 0.057 | **56%** |

Stuffing "in the video the … as it is shown in the video here" around an
unchanged answer lifts METEOR by 45–89% relative, significantly, on 63–89 of 150
queries. METEOR on this task is substantially a length-matching measure. Since
NIST scores with METEOR, that is worth knowing — but it also means **most of
B's headline gain is not evidence that the answers got better.** BERTScore is
the honest signal for that, and it moved by at most +0.011.

**One practical strike against C:** it fails `validate.ag.run.py`. Padding
collapses distinct short answers into identical strings ("a flip" and "the flip"
both become the same padded text) — 2 duplicate slots in gemma4, 1 in qwen3vl.
Only 3 in 4,500, but any duplicate makes a run unsubmittable. C would need a
dedup pass before it could ship at all.

### What expansion does to the model ranking

The ordering is unchanged — expanded Gemma 4 still leads — but the gap narrows
sharply. **Expanded Qwen3-VL (0.134) is statistically indistinguishable from
baseline Gemma 4 (0.148): mean Δ −0.0134, p = 1.0.** Qwen3-VL gains the most
from expansion (+0.081, the largest of the three) because it started the
shortest at 3.3 words. That supports the reading in "Reading the table" above:
a real part of Qwen3-VL's apparent deficit was verbosity rather than accuracy.

### Honest limitations of the expansion itself

**86% of rewrites invent something.** The expander has no video, so every added
specific is a guess. Measured on the first pass: 909 / 951 / 543 rewrites added
at least one content word absent from the original, averaging ~2 each. Real
examples from the hand-reviewed sample of 30: "the man performs a flip" → "a
**single** flip"; "The woman looks at the man" → "looks **directly** at the man
**by the end**"; "They saw her large physique" → "**and turned to look at her**".
The containment check guarantees nothing is *lost*; nothing prevents additions.
That METEOR rises means the additions are net-helpful *to the metric* — it does
not mean they are true.

**A defect found and fixed mid-flight.** The refusal pass-through leaked: 176
literal fallback strings were rewritten on the first pass (160 of them in
Cosmos), turning "it is not clear from the video" into "It is not clear from the
video what they do." — a longer zero. `NON_ANSWER_RE` is anchored with `[.!]?$`
and so missed the two fallback entries carrying a trailing prepositional phrase.
Fixed by checking literal membership of the known `FALLBACK_ANSWERS` pool as
well as the regex; all three runs were regenerated and the leak is now 0/0/0.

### Verification

- All three **B** runs pass `validate.ag.run.py` with 0 errors, 0 warnings.
  Two of three **C** runs fail on duplicates, as above.
- Baseline runs byte-identical after the pass (`expand.answers.py` never edits
  in place): sha256 `fa1b5b0c…`, `f4c27bdf…`, `33d454b7…`.
- Q_IDs, Video_IDs and **rank order** unchanged in all three — this is not a
  re-ranking pass.
- Content-word containment rejected 151 / 117 / 100 rewrites that dropped
  meaning; those keep the original.
- `count` questions never expanded (18 of 150), refusals passed through
  (44 / 134 / 827 slots).
- Oracle still 1.000 BERTScore / 0.924 METEOR.

### Cost

4.7 / 4.4 / 2.6 min per run on `gemma4:12b` at 4 workers, ~12 min for all three
— roughly 8 s per query. Extrapolating to the 2,443-query 2026 set: about
**1.2 h per run**. Condition C is instantaneous and needs no GPU.

## Answer re-ranking — a negative result

Only Gemma 4 attaches any numeric signal to its answers: each of its ten prompts
returns a self-reported `confidence`, which `pool_candidates()`
(`generate.answers.py:541-585`) folds into a per-candidate `score` and `votes`.
Qwen3-VL and Cosmos make one call each and simply assert an order. No pipeline
captures token logprobs. None of the signal reaches the run files.

**First, the bound.** `eval.vqa.ag.py` scores a query as `np.mean` over all ten
answers (`:125-126`), which is order-independent — so re-ranking **cannot**
change mean METEOR or mean BERTScore. NDCG_BERTScore is already saturated at
0.993–0.995. This is a **NDCG_METEOR-only** question.

### NDCG_METEOR under five orderings

| Run | reverse | random | **current** | score-reranked | oracle | headroom |
|---|---|---|---|---|---|---|
| gemma4 | 0.734 | 0.735 | **0.751** | 0.749 | 0.933 | **+0.182** |
| qwen3vl | 0.516 | 0.533 | **0.597** | — | 0.833 | **+0.236** |
| cosmos | 0.471 | 0.477 | **0.524** | — | 0.753 | **+0.229** |

`oracle` sorts by true per-answer METEOR — the ceiling, diagnostic only.

Paired per query (n=150):

| Comparison | gemma4 | qwen3vl | cosmos |
|---|---|---|---|
| current − random | +0.0164 (81/150, p=0.058) | +0.0643 (79/150, p≈9e-5) | +0.0477 (64/150, p=0.007) |
| oracle − current | +0.1819 (140/150, p≈1e-24) | +0.2359 (117/150, p≈6e-21) | +0.2290 (104/150, p≈9e-19) |
| score-reranked − current | **−0.0024 (0 better, 14 worse, p=0.001)** | — | — |

### Three findings

**1. Re-ranking by the confidence signal does not work.** It recovers **−1%** of
a 0.182 gap — it is significantly *worse* than the order already being emitted,
improving zero queries and hurting 14. The reason is structural: `pool_candidates()`
**already sorts by score**, so re-ranking by score reproduces almost the same
order. The small loss comes from discarding its refusal-last tiebreak, which
turns out to be worth ~0.002 NDCG on its own. The existing heuristic beats raw
score.

This is consistent with the correlation measured directly: within-query Spearman
of `score` against per-answer METEOR is **0.097**, no better than plain emitted
rank (0.106), and `votes` is slightly *negative* (−0.015). The signal has almost
no resolution to begin with — a mean of **5.3 distinct score values across the
ten answers**, with **69% of answers at ≤1 vote**, ties broken by which prompt
happened to surface the answer first.

**2. Gemma 4's expensive ranking machinery is the weakest of the three.** Its
30-candidate vote produces an ordering only +0.016 better than random shuffling
(p=0.058, not significant at 0.05). A single Qwen3-VL call asserting its own
order does **+0.064** (p≈9e-5), and Cosmos +0.048. The ensemble buys answer
*quality* — Gemma 4 leads on mean METEOR — but its ranking is close to noise.

**3. Substantial headroom exists and no available signal reaches it.**
0.18–0.24 NDCG_METEOR sits between the current ordering and a perfect one, on
every run, at p < 1e-18. Capturing it needs a signal that does not currently
exist: sequence logprobs at generation time (vLLM supports `logprobs` /
`top_logprobs`; neither script requests them), or a reranker model. Post-hoc
re-ranking of what is already cached cannot get there.

### Verification

- **Conservation (the strong check):** mean METEOR is **identical** across every
  ordering — 0.148 / 0.053 / 0.033 — confirming no answer was lost, duplicated or
  altered, and confirming the order-independence claim empirically.
- **Ordering sanity:** `reverse` < `random` < `current` < `oracle` holds for all
  three runs; `oracle` is the maximum everywhere.
- All ten re-ranked runs pass `validate.ag.run.py --warnings-as-errors` at
  150/150, 0 errors.
- Answer-set identity asserted per query by the re-ranker itself.
- Originals byte-identical: `fa1b5b0c…`, `f4c27bdf…`, `33d454b7…`.

**Conclusion: do not re-rank.** The 2026 runs should keep their emitted order.
The finding worth carrying forward is that logprobs are cheap to capture on the
next vLLM run and are the only plausible route to the 0.18–0.24 NDCG on the
table.

## Sample fidelity

Two properties of the sample were measured after it was drawn, and both bear on
transfer to 2026.

**Routing marginals came out close, for free.** The draw stratified on duration
only, yet the routed labels land within ~3 pp of the 2026 test set on every
route: vision 78.7% (2026: 75.3%), vision+audio 12.7% (15.5%), audio 5.3%
(6.9%), the four OCR routes 3.3% (2.2%). All four OCR routes are represented
rather than rounding to zero as `EVALUATION_PLAN.md` expected. Budget tiers:
deep 13.3% (12.3%), standard 66.0% (58.2%), brief 20.7% (**29.5%** — the one
real gap, so the sample skews toward more expensive tiers). By question type,
`summary` (7.3% vs 2.4%) and `yes_no` (5.3% vs 2.0%) are over-represented and
`reason` (12.0% vs 17.1%) under. **Consequence: the plan's Stage 2/3 —
routing a 400-query pool and selecting against three marginals — would have
bought very little over the duration-only draw.**

**Speech coverage is much higher than 2026, and this matters.** 94 of the 150
sampled clips (63%) have detected speech, against 576 of 2,443 queries (24%) in
the 2026 test set. These Shorts are far more speech-heavy than the corpus this
is meant to predict, so the ASR input carries materially more weight here.
A pipeline that leans on the transcript is flattered by this sample.

## Sample

150 of 150 requested, seed `20260814`, drawn from a pool of 1327.

| Bin | Pool | Drawn | 2026 target |
|---|---|---|---|
| 0-5s | 0 | 0 | 18 |
| 5-10s | 140 | 16 | 26 |
| 10-15s | 398 | 45 | 44 |
| 15-30s | 756 | 85 | 29 |
| 30-60s | 33 | 4 | 6 |
| 60-150s | 0 | 0 | 16 |
| 150s+ | 0 | 0 | 9 |

## What these numbers are, and are not

- **Quote them as "on a 150-query 2025 sample", never as "our METEOR".** Per-query spread is far wider than the differences between runs; the absolute figure carries roughly ±0.03.
- **Short video only.** Every 2025 ground-truth clip is a YouTube Short (5–30 s, median 15 s). On this corpus `adaptive_fps = min(4.0, max(0.2, 64 / duration))` clamps to 2.1–4.0 fps for every clip, and `LONG_VIDEO_SEC = 60` puts all of them on Gemma 4's 16-frame `CAP_SHORT` path. The fps floor, the frame-token degrade ladder, the 32-frame long path and the 4 GiB decode-limit transcode are never exercised. The 2026 test set runs to 389 s with 6.3% over 150 s, and nothing here measures that.
- **The pool is 1,327 of 2,000.** The GT download was interrupted at Q_ID 1471, so the sample is drawn from a positional slice of the ground truth rather than the whole of it.
- **Mild style leakage.** `MAX_ANSWER_WORDS = 40` and the shared "most good answers are under ten words" prompt rule were tuned against this very ground truth's length distribution (`generate.answers.qwen3vl.py:269-273`). That flatters the length-sensitive part of METEOR. It cannot be removed without changing the pipeline under test.

## Configuration under test

The corpus-derived tuning constants were left at their 2026 values deliberately — retuning them for short clips would mean the number produced is not the 2026 pipeline's number:

| Constant | Value | Where |
|---|---|---|
| `TARGET_FRAMES` | 64 | qwen3vl, cosmos |
| `DENSE_TARGET_FRAMES` | 96 | qwen3vl |
| `MAX_FPS` / `MIN_FPS` | 4.0 / 0.2 | qwen3vl, cosmos |
| `MIN`/`MAX_FRAME_TOKENS` | 128 / 768 | qwen3vl |
| `TEXT_TOKEN_RESERVE` | 2048 | qwen3vl |
| `VISUAL_BUDGET_SAFETY` | 0.8 | qwen3vl |
| `MIN_FRAMES`/`CAP_SHORT`/`CAP_LONG` | 8 / 16 / 32 | extract.frames |
| `LONG_VIDEO_SEC` | 60.0 | extract.frames |

## What this does and does not license

**Do not swap the submission to Gemma 4 on this evidence alone.** The result is
strong and significant, but it is measured entirely on 5–30 s clips, and the
axis where the three pipelines most plausibly diverge — long video — is exactly
what this sample cannot test. Gemma 4 sees 16 sampled frames regardless of
length; Qwen3-VL and Cosmos take video natively and their frame-budget and
degrade machinery only engages past 60 s. On a 389 s clip the ordering could
reverse, and nothing here speaks to that.

**What it does license:**

1. **Cosmos's short-clip output is broken enough to investigate.** 20% unparsed
   JSON and 5.5/10 filler slots is a defect, not a quality gap. If the same rate
   holds on the 2026 run, `cosmos.run.task1.json` is carrying a lot of dead
   slots. Worth checking that run's `content_parsed` and `fallback` counts
   directly.
2. ~~**The terse-answer style is worth one experiment.**~~ **Done — see
   "Answer expansion" above.** Expansion lifts METEOR on all three runs
   (+0.039 / +0.081 / +0.044, all p ≈ 0) without reducing BERTScore or NDCG, so
   the gate clears. The caveat that came with it: a zero-information mechanical
   padding control captures 56–62% of that gain, so most of the improvement is
   METEOR responding to length rather than to better answers.
3. **The scorer is now trusted infrastructure.** Parity with `eval.vqa.ag.py` is
   exact to 3 dp with zero per-query disagreements, and the oracle check passes.
   Any future ablation — routing on/off, budget tier, temperature — can be
   measured rather than argued.

**Verification that backs these numbers:** scorer parity exact to 3 dp on
METEOR, BERTScore, NDCG_METEOR and NDCG_BERTScore over 20 queries × 10 answers
with 0 per-query disagreements; oracle 1.000 BERTScore / 0.924 METEOR (METEOR
cannot reach 1.0 — its fragmentation penalty costs 0.5 on a single-word answer,
and 17% of this ground truth is one word); all three runs pass
`validate.ag.run.py` with 0 errors and 0 warnings; 0 unanswered queries and 0
Video_ID mismatches in all three; every guarded 2026 artifact byte-identical
after the run and `videos_TEST_VQA/` unchanged at 1,940 files.

---

# Appendix: how this was produced

## What was built

Three new scripts. Every pre-existing script is driven unchanged, through its
CLI only.

| File | Role |
|---|---|
| `VQA/run.eval.2025.py` | 15-stage orchestrator: stage graph, GPU/server arbitration, guards, resumable state, per-stage logs |
| `VQA/sample.2025.eval.py` | `convert` (CSV → query JSON + eligible pool) and `select` (duration-stratified draw, GT CSV, reserves, achieved-vs-target report) |
| `VQA/score.answers.py` | METEOR / BERTScore / NDCG / STS_local over a 2026-format run, plus paired comparison and an `eval.vqa.ag.py` parity harness |
| `VQA/expand.answers.py` | ANSWER_EXPANSION_PLAN.md Phase 2 — rewrites answers longer without changing meaning; `--mechanical` provides the zero-cost control |
| `VQA/rerank.answers.py` | `analyze` (does confidence predict correctness?) and `rerank` (emit a re-ordered run for the existing scorer to measure) |

### Re-ranking commands

```
# does the signal predict correctness? joins the work caches against
# eval.2025/<run>_per_response.csv -- no inference
python VQA/rerank.answers.py analyze

# emit each ordering; --by current|random|reverse|score|votes|oracle
python VQA/rerank.answers.py rerank --run <model>.run.2025.json \
  --cache answers.2025.<model>.cache.jsonl --gt vqa.gt.2025.sample150.csv \
  --by <key> --out <model>.rerank-<key>.run.2025.json

# score them all with the existing scorer -- no new metric code
python VQA/score.answers.py --gt vqa.gt.2025.sample150.csv \
  --run gemma4_cur=gemma4.run.2025.json \
  --run gemma4_random=gemma4.rerank-random.run.2025.json \
  --run gemma4_oracle=gemma4.rerank-oracle.run.2025.json \
  ... --metrics meteor --out eval.2025.rerank.scores.json \
  --csv eval.2025.rerank.scores.csv --per-response-dir eval.2025.rerank
```

`--metrics meteor` alone suffices: NDCG_BERTScore is saturated, and mean METEOR
serves as the conservation check (it must not move).

### Answer expansion commands

```
# condition B -- LLM rewrite, needs Ollama on the card
python VQA/expand.answers.py --run <model>.run.2025.json \
  --queries vqa.2025.sample150.json --routing query.routing.2025.json \
  --out <model>.expanded.run.2025.json \
  --cache answers.2025.<model>.expansion.cache.jsonl

# condition C -- mechanical padding control, no model, instant
python VQA/expand.answers.py ... --out <model>.mechanical.run.2025.json \
  --cache answers.2025.<model>.mechanical.cache.jsonl --mechanical

# score all nine together, which yields the paired A/B/C deltas
python VQA/score.answers.py --gt vqa.gt.2025.sample150.csv \
  --run A_gemma4=gemma4.run.2025.json \
  --run B_gemma4=gemma4.expanded.run.2025.json \
  --run C_gemma4=gemma4.mechanical.run.2025.json \
  ... (same for qwen3vl and cosmos) ... \
  --metrics meteor,bert --out eval.2025.expansion.scores.json \
  --csv eval.2025.expansion.scores.csv \
  --per-response-dir eval.2025.expansion --oracle
```

Length targets are measured, not guessed — from the 2025 GT, by
`Question_Type`: action 10, reason 10, manner 11, summary 12, entity 6, place 6,
time 7, yes_no 6 words. `count` is in `NO_EXPAND`: its GT answers average
**1.0 words** (median 1, literally "one"/"two"/"three"), and 23% of this ground
truth is 1–2 words.

## Reproducing the run

Run from **Windows**, in `E:\TREC_2026`, using `.venv\Scripts\python.exe`.
Windows is the only host that can drive both backends: Ollama is a Windows
binary bound to localhost that WSL2 cannot reach, while the vLLM clients must
run *inside* WSL2 because they hand the server a `file://` URI it opens itself
(`file:///E:/...` means nothing to a Linux process).

### 0. One-time setup

```
.venv\Scripts\python.exe -m pip install pandas nltk bert-score sentence-transformers
```

`nltk` `wordnet` / `punkt` / `punkt_tab` download automatically on first score.
`transformers` was left at its existing 5.15.0 pin; `bert_score` works with it.

### 1. The whole thing

```
python VQA/run.eval.2025.py --smoke 5          # rehearse everything first
python VQA/run.eval.2025.py                    # the real run
```

Useful flags: `--dry-run` (print every command, execute nothing) · `--from
<stage>` / `--to <stage>` / `--stages a,b,c` · `--force` (ignore completion
markers) · `--no-gate` (do not wait for the 2026 run) · `--wait-for-2026
<seconds>` · `--install-deps` · `--no-guard` · `--restore-2026`.

### 2. What was actually run this time

```
python VQA/run.eval.2025.py --stages queries   --no-gate
python VQA/run.eval.2025.py --stages durations --no-gate
python VQA/run.eval.2025.py --stages sample    --no-gate
python VQA/run.eval.2025.py --wait-for-2026 25200        # gated on the 2026 run
python VQA/run.eval.2025.py --from answers:qwen3vl       # resumed after fixes
```

The smoke rehearsal was skipped on instruction. Three orchestration defects then
surfaced during the unattended run, each costing a restart — see below. The
resume is cheap because every stage is cached and marker-guarded.

### 3. The per-stage commands it issues

Reproducible standalone; `--dry-run` prints these verbatim.

```
python VQA/sample.2025.eval.py convert \
  --gt vqa.gt.2025.csv --videos videos_GT_2025/videos \
  --queries-out vqa.gt.2025.queries.json --pool-out vqa.gt.2025.pool.json \
  --test-2026 vqa.2026.testingDataset.Task1.json

python VQA/duration.report.py \
  --queries vqa.gt.2025.queries.json --videos videos_GT_2025/videos \
  --report duration.report.gt2025.json --html duration.report.gt2025.html

python VQA/sample.2025.eval.py select \
  --pool vqa.gt.2025.pool.json --durations duration.report.gt2025.json \
  --metadata videos_GT_2025/metadata -n 150 --seed 20260814 \
  --out vqa.2025.sample150.json --gt-out vqa.gt.2025.sample150.csv \
  --reserves-out vqa.2025.sample150.reserves.json \
  --report-json sample.2025.sample150.report.json \
  --report-md sample.2025.sample150.report.md

python VQA/extract.frames.py \
  --queries vqa.2025.sample150.json --videos videos_GT_2025/videos \
  --durations duration.report.gt2025.json --out videos_GT_2025/frames

python VQA/transcribe.whisper.py \
  --videos-dir videos_GT_2025/videos \
  --out-dir videos_GT_2025/transcription_openwhisper \
  --model-dir models/faster-whisper-large-v3 --only <the 150 video IDs>

python VQA/route.query.inputs.py \
  --queries vqa.2025.sample150.json \
  --out query.routing.2025.json --cache query.routing.2025.cache.jsonl

python VQA/generate.query.variations.py \
  --queries vqa.2025.sample150.json \
  --out query.variations.2025.json --cache query.variations.2025.cache.jsonl

python VQA/generate.answers.py \
  --queries vqa.2025.sample150.json --variations query.variations.2025.json \
  --frames videos_GT_2025/frames \
  --transcripts videos_GT_2025/transcription_openwhisper \
  --out gemma4.run.2025.json --cache answers.2025.gemma4.cache.jsonl --workers 1
```

Inside WSL2 (`wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/e/TREC_2026 && …'`):

```
/home/aakash/cosmos-reason2/.venv/bin/python VQA/generate.answers.qwen3vl.py \
  --queries     /mnt/e/TREC_2026/vqa.2025.sample150.json \
  --routing     /mnt/e/TREC_2026/query.routing.2025.json \
  --videos      /mnt/e/TREC_2026/videos_GT_2025/videos \
  --frame-index /mnt/e/TREC_2026/videos_GT_2025/frames/index.json \
  --transcripts /mnt/e/TREC_2026/videos_GT_2025/transcription_openwhisper \
  --out         /mnt/e/TREC_2026/qwen3vl.run.2025.json \
  --cache       /mnt/e/TREC_2026/answers.2025.qwen3vl.cache.jsonl --workers 4

/home/aakash/cosmos-reason2/.venv/bin/python VQA/generate.answers.cosmos.py \
  --queries     /mnt/e/TREC_2026/vqa.2025.sample150.json \
  --videos      /mnt/e/TREC_2026/videos_GT_2025/videos \
  --frame-index /mnt/e/TREC_2026/videos_GT_2025/frames/index.json \
  --transcripts /mnt/e/TREC_2026/videos_GT_2025/transcription_openwhisper \
  --out         /mnt/e/TREC_2026/cosmos.run.2025.json \
  --cache       /mnt/e/TREC_2026/answers.2025.cosmos.cache.jsonl --workers 2
```

Then, back on Windows:

```
python VQA/validate.ag.run.py vqa.2025.sample150.json <run>.json \
  --report-json <run>.validation.json

python VQA/score.answers.py \
  --gt vqa.gt.2025.sample150.csv \
  --run gemma4=gemma4.run.2025.json \
  --run qwen3vl=qwen3vl.run.2025.json \
  --run cosmos=cosmos.run.2025.json \
  --metrics meteor,bert --out eval.2025.scores.json --csv eval.2025.scores.csv \
  --per-response-dir eval.2025 --oracle
```

### 4. Server lifecycle

`ensure_backend()` is the only code permitted to start or stop a model server.

```
# Qwen3-VL (port 8009) -- setsid and </dev/null are load-bearing, see defect 3
cd /home/aakash/cosmos-reason2 && setsid nohup ./.venv/bin/vllm serve \
  Qwen/Qwen3-VL-8B-Thinking --host 0.0.0.0 --port 8009 \
  --served-model-name Qwen/Qwen3-VL-8B-Thinking \
  --allowed-local-media-path /mnt/e/TREC_2026 --max-model-len 24576 \
  --gpu-memory-utilization 0.92 --enforce-eager \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3 --skip-mm-profiling \
  < /dev/null > /home/aakash/qwen3vl.eval.log 2>&1 & disown; exit 0

# Cosmos-Reason2 (port 8008)
systemctl --user start cosmos-reason2

# Teardown -- unit first (Restart=on-failure would resurrect it), then the tree
systemctl --user stop cosmos-reason2; pkill -f '[v]llm serve'
taskkill /F /T /IM "ollama app.exe" /IM ollama.exe /IM llama-server.exe
```

### 5. Scorer parity check

The gate on every number above. Emit the run in `eval.vqa.ag.py`'s headerless
CSV shape and score it both ways over the same queries:

```
python VQA/score.answers.py --gt <20-row GT>.csv --run synth=<run>.json \
  --metrics meteor,bert --emit-csv parityCSV/ --oracle ...

cd <a directory without calc.sts.pl>          # makes the dead STS call fail fast
python VQA/eval.vqa.ag.py <20-row GT>.csv <abs path>/parityCSV
```

Both returned METEOR 0.277, BERTScore 0.886, NDCG_METEOR 0.964,
NDCG_BERTScore 0.997, with **0 per-query disagreements at 3 dp** across all four
metrics.

## Guards

Five, because a 3.5-hour unattended run fails silently otherwise.

| Guard | What it prevents | Where |
|---|---|---|
| **Completion gate** | Starting while the 2026 Qwen3-VL run is live. Checks three conditions, since any one alone lies: cache holds all 2,443 Q_IDs; `qwen3vl.run.task1.json` exists with 2,443 entries (written *only* after a clean inference loop — a Ctrl-C'd run exits 130 before `assemble()` and leaves a full cache with no run file); no `generate.answers.qwen3vl.py` process alive in WSL. | `qwen3vl_2026_state()` |
| **2026 artifact fingerprint** | Silent overwrite. `duration.report.py` defaults to `--report duration.report.json`, the live 2026 file `extract.frames.py` reads. Size + sha256 of 11 artifacts plus the `videos_TEST_VQA/` inventory, re-verified **after every stage**, aborting at the stage that caused it. Armed *after* the gate — see defect 1. | `fingerprint()` / `verify()` |
| **Path-flag coverage** | The Q_ID collision. Statically parses each driven script for arguments whose default is a `DEFAULT_* = ROOT / …` path and fails if the orchestrator's command omits any. Subparser-aware. | `path_flags()` / `check_flags()` |
| **Disjointness assertion** | A 2025 GT video also being in the 2026 test set. The qwen3vl docstring claimed it; nothing checked it. Now verified: 0 overlap across 2,000 × 1,956. | `sample.2025.eval.py convert` |
| **Oracle check** | Broken normalisation or misalignment in the scorer. GT scored against itself must give BERTScore ≥ 0.99 and METEOR ≥ 0.85. | `score.answers.py --oracle` |

**Why the Q_ID guard is not theoretical:** all 150 sampled Q_IDs exist in *both*
years' variation files with entirely different questions — Q_ID 21 is "How long
do you have to wear an orthodontic expander?" in 2025 and "Why is smoke rising
up?" in 2026, and **0 of 150 share a question**. Taking `--variations`'s default
would have asked every query using paraphrases of an unrelated 2026 question,
then run clean, validated, and scored — and meant nothing.

## Orchestration defects found and fixed

All three were in the orchestrator, none in the answering pipeline. All three
were invisible interactively and only appeared unattended.

**1 — The guard was armed before the gate.** The baseline was snapshotted at
startup, but the gate's whole purpose is to wait while the 2026 run appends to
`answers.task1.qwen3vl.cache.jsonl` and finally writes `qwen3vl.run.task1.json`
— both guarded. It read the 2026 run's own successful completion as tampering
and aborted a five-hour wait at the moment it succeeded (exit 2).
*Fix:* arm the baseline after the gate, from the first non-gate stage onward.

**2 — Ollama's model host was orphaned on the GPU.** Ollama does not hold the
model in `ollama.exe`; it spawns `lib\ollama\llama-server.exe`, and that child
owns the VRAM. Teardown killed the parent and the CLI wrapper, so port 11434
closed and the swap *looked* successful — while 8,922 MiB stayed on the card.
vLLM could not allocate at `--gpu-memory-utilization 0.92` and spent its full
15-minute startup window never coming up.
*Fix:* graceful unload (`keep_alive: 0`, then `ollama stop`), then
`taskkill /F /T` over `llama-server.exe` and `ollama_llama_server.exe` as well;
`teardown()` now returns whether the card is actually free, retries once, and
reports the holding PIDs via `nvidia-smi --query-compute-apps`;
`ensure_backend()` aborts rather than starting a server that cannot allocate.

**3 — vLLM was killed by the launcher's own pipes.** A process backgrounded from
`wsl.exe -- bash -lc '… &'` dies when that session ends. `setsid` alone did not
save it: the real culprit was `subprocess.run(capture_output=True)` — when
Python closed the pipes, WSL tore down the launched process with them. The
launch returned `rc=0`, the shell's redirect truncated the server log to zero
bytes, and nothing was ever listening. Proven, not guessed: the identical
command line survives with stdio at `DEVNULL` and dies with `capture_output`.
*Fix:* a separate `launch_detached()` used only for server starts (stdio at
`DEVNULL`), plus `setsid nohup … < /dev/null` in the command. `quiet()` is
unchanged for commands whose output is genuinely wanted.

**Two smaller ones fixed alongside:**

- `pkill -f 'vllm serve'` ran inside a `bash -lc` whose own command line
  contains `vllm serve`, so it matched and killed its own shell, abandoning the
  rest of the teardown. Changed to `[v]llm serve`, which cannot match the
  pattern text itself.
- `ensure_backend()` tore down a healthy server before checking whether it was
  already the backend wanted, costing a full reload on every resume. It now
  adopts a running server when `current` is unknown.

**The lesson worth keeping:** an unattended run is a different environment from
an interactive one, and every assumption carried over from watching a human run
these commands had to be re-earned — process ownership on teardown, session
survival on launch, stdio handling. The `--smoke 5` rehearsal exists precisely
to surface this class in 30 minutes rather than across three multi-hour
restarts; skipping it cost more than it saved.

## Stage timings (the completed run)

| Stage | Time | Result |
|---|---|---|
| queries | <1 min | 1,327 eligible of 2,000; disjointness verified |
| durations | 0.4 min | 1,327 probed, 0 ffprobe failures |
| sample | <1 min | 150 drawn, 16/45/85/4 across the bins |
| frames | 0.5 min | 150/150 videos, 2,110 frames |
| whisper | 3.7 min | 150/150 transcripts, 94 with speech |
| route | 16.4 min | 150/150 |
| variations | 6.1 min | 150/150, 10 paraphrases each |
| answers:gemma4 | 39.9 min | 150/150, 0 failures |
| answers:qwen3vl | 37.9 min | 150/150, 0 failures |
| answers:cosmos | 34.3 min | 150/150, 0 failures |
| validate | <1 min | 3/3 pass, 0 errors, 0 warnings |
| score | 1.5 min | oracle passed |
| report | <1 min | this file |

Roughly 2 h 20 min of stage time end to end, plus a 5 h gated wait on the 2026
run and three restarts from the defects above.

## Artifacts

Every 2025 artifact is separately named so nothing can collide with 2026:

```
vqa.gt.2025.queries.json          all 2,000 GT rows as query JSON
vqa.gt.2025.pool.json             1,327 eligible (video on disk, not in SKIP)
duration.report.gt2025.json/.html
vqa.2025.sample150.json           the query file every downstream stage reads
vqa.gt.2025.sample150.csv         150-row headerless GT for the scorer
vqa.2025.sample150.reserves.json  per-stratum backfill list
sample.2025.sample150.report.json/.md
videos_GT_2025/frames/index.json + <id>/*.jpg
videos_GT_2025/transcription_openwhisper/<id>.json
query.routing.2025.json      + query.routing.2025.cache.jsonl
query.variations.2025.json   + query.variations.2025.cache.jsonl
gemma4.run.2025.json         + answers.2025.gemma4.cache.jsonl
qwen3vl.run.2025.json        + answers.2025.qwen3vl.cache.jsonl
cosmos.run.2025.json         + answers.2025.cosmos.cache.jsonl
*.run.2025.validation.json
eval.2025.scores.json / .csv + eval.2025/*_per_response.csv
eval2025.state.json          logs/eval2025/*.log
```

Smoke mode substitutes a `.smoke` infix throughout, so a rehearsal can never
contaminate the real caches.


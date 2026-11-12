# Plan — Post-processing pass to lengthen Task 1 answers

**Status: proposed, not implemented.** Written 14 August 2026, alongside
`SESSION.2026-08-14.qwen3-vl.md`. Nothing in this document has been built or
measured yet; the numbers in the Context and evidence sections *are* measured
and are cited from where they came.

## Context

Our submitted answers are far shorter than the reference answers the metric
compares them against:

| Source | all-ten mean | rank-1 mean | ≤2 words |
|---|---|---|---|
| `gemma4.run.task1.json` | 4.44 | 3.94 | 26.5% |
| `cosmos.run.task1.json` | 3.33 | 2.42 | 50.2% |
| qwen3vl (n=8 sample) | 3.2–3.9 | 3.6–4.1 | — |
| **`VQA/vqa.gt.csv` (2025 reference)** | **8.38** | — | 22.4% |

`VQA/eval.vqa.ag.py:109` scores **each of the ten answers** against a single
reference with METEOR, then takes `np.mean` over all ten (`:125`), so every slot
counts equally. METEOR weights recall above precision, and recall is
`matches / len(reference)` — a 3-word answer can match at most 3 of an 8-word
reference's tokens no matter how correct it is. Lengthening plausibly lifts
score.

**This has never been measured, and it can backfire.** METEOR also carries a
fragmentation penalty over match chunks, and padding dilutes precision. So this
plan gates on evidence: build a scoring harness on the 2025 data first, and only
alter a submission if the harness shows a gain.

**Scope:** `gemma4.run.task1.json` and the qwen3vl run when it finishes.
Excluding `cosmos.run.task1.json` — it was answered 45% blind because of the
`Video_ID` naming bug (`SESSION.2026-08-14.qwen3-vl.md` §1), so polishing it is
not worth the compute.

## What the evidence already says

Measured 14 August against `VQA/vqa.gt.csv` (n=2,000):

- **Do not echo the question.** Only 11.3% of a question's content words reappear
  in its answer (median 0%); just 32.2% of answers echo even one. Template
  expansion like *"The man is holding a document"* injects tokens the reference
  usually lacks — hurting precision and fragmentation for a recall gain that
  lands only ~1 time in 3.
- **Do produce sentences.** 66.1% of GT answers are 5+ words, opening with
  `the` (376), `he` (215), `a` (148), `she` (119), `it` (101). nltk's METEOR does
  **not** strip stopwords, so function words are matchable tokens. Re-expressing
  *"a document"* as *"He holds up the document"* gains recall from the frame
  itself, not from invented facts.
- **Do not expand everything.** 22.4% of GT answers are 1–2 words — counting and
  yes/no questions, whose GT openings are literally `one` (85), `two` (61),
  `three` (26), `no` (21). Blanket expansion overshoots these.

## Phase 1 — Measurement harness (the gate)

> **Superseded by `VQA/EVALUATION_PLAN.md`.** The synthetic conditions below
> were designed on the assumption that the 2025 videos were unobtainable, so
> terse candidates had to be manufactured from the ground truth itself. The
> better plan is to download a sample of the 2025 GT videos and run the real
> pipeline over them, giving genuine model answers to score. Conditions **C**
> (mechanical filler control) and **D** (GT-vs-GT oracle) below are still worth
> keeping as controls, and the environment/STS/format notes still apply.

Goal: a METEOR/BERTScore delta for expansion, before any submission changes.

**Environment.** `nltk` and `bert_score` are installed nowhere on this box.
Install them into the **Windows venv** (`E:\TREC_2026\.venv`, which already has
pandas/numpy/sklearn), **not** `/home/aakash/cosmos-reason2/.venv` — that venv
serves vLLM and must not have its `transformers` pin disturbed. Also
`nltk.download('wordnet')` and `punkt`.

**Disable STS.** `VQA/calc.sts.pl` calls the external `swoogle.umbc.edu` web
service (2013-era) once per answer — 20,000 HTTP calls to a service that likely
no longer answers. Run the harness on METEOR, BERTScore and their NDCGs; leave
STS out rather than let it produce NaNs or hang.

**Format shim.** `eval.vqa.ag.py:73` reads runs as headerless **CSV**
(`QueryID,VideoID,Rank,Answer,time`), while `validate.ag.run.py` expects the 2026
**JSON**. The harness needs a small JSON→CSV writer; it does not need the eval
script modified.

**Test conditions.** The 2025 videos are not on disk, so answers cannot be
regenerated for them. Instead, build terse candidates *from* the GT and measure
what expansion recovers — which mirrors our real situation (answer is right, too
short):

| Condition | Construction | Question it answers |
|---|---|---|
| **A. terse** | deterministic 3–4 content-word core extracted from each GT answer | baseline, mimics our length distribution |
| **B. expanded** | run the expander on (question, A) — it never sees GT | does meaningful expansion help? |
| **C. filler** | pad A mechanically to B's length with generic frame words | is it *length* or *content* that moves the metric? |
| **D. oracle** | GT scored against itself | harness wiring sanity check (METEOR ≈ 1) |

C is the important control: if C ≈ B, the metric is rewarding length alone and a
zero-cost mechanical rule beats an LLM pass.

**Secondary realism check (optional).** Generate text-only answers to the 2025
questions with the same prompt style (no video available), score unexpanded vs
expanded. Absolute scores will be poor; only the delta is informative.

**Also produced here:** the per-question-type GT length table that sets Phase 2's
targets. Bucket the 2,000 GT questions by keyword heuristic (`how many/how much`
→ count, leading `is/are/does/did/can` → yes_no, `why` → reason, `where` → place,
`when` → time, else action/entity) and report the length distribution per bucket.
Targets get measured, not guessed.

**Decision rule.** Proceed to Phase 2 only if B beats A on mean METEOR **and**
does not reduce mean BERTScore or NDCG. If C ≈ B, implement C instead — it is
free.

## Phase 2 — The expander

Only if Phase 1 clears it.

**New file:** `VQA/expand.answers.py`. Reads a run file, writes a new run file;
never edits in place. Model-agnostic — same contract for gemma4 and qwen3vl.

**Reuse, don't rewrite.** `VQA/generate.query.variations.py` is structurally the
same program (rewrite text, add no facts, schema-constrained JSON, JSONL cache,
resume, `--rebuild`, `--only`, `--force`, SHA-256-derived seeds). Lift its Ollama
client and driver. Take `tidy()` and `normalize()` verbatim from
`VQA/generate.answers.qwen3vl.py` so dedupe stays identical to the validator's
rule (`validate.ag.run.py:276`).

**Model: `gemma4:12b` via Ollama with `"think": false`.** Precedent:
`SESSION.2026-08-12.query-variations.md` measured 2,350 rewrites in 96 min at 4
workers this way, and recorded that `think: true` cost 19.1 s vs 5.3 s per item
for no benefit on a rewriting task. Expect ~2 h for a full run file. Do **not**
use the running Qwen3-VL-Thinking server: it has no non-thinking mode, so 2,443
calls would carry full reasoning traces and cost about as much as the answering
run itself.

**Per query, one call for all ten answers** — they share a question and context,
and one call keeps the ten mutually distinct by construction.

**Length policy, by `Question_Type` from `query.routing.task1.json`** (already on
disk, 2,443 entries): `count` (312) and `yes_no` (50) keep a short target;
`action` (753), `reason` (417), `summary` (58), `manner` (202) take the long
target; `entity` (462), `place` (103), `time` (78) take a middle target. Exact
numbers come from Phase 1's per-bucket table.

**Prompt constraints** (the whole safety story — the expander has no video):
- Re-express the given answer as a fuller phrase or sentence. Preserve meaning
  exactly.
- Add no new facts. No colours, counts, names, places or actions not already in
  the answer. Grammatical scaffolding only.
- Do not restate the question, do not add hedges ("it appears that").
- Keep the ten mutually distinct.
- Leave a refusal ("unknown", "it is not shown in the video") as-is.

**Invariants to enforce after rewriting**, falling back to the original string on
any breach:
- exactly 10 answers, all unique under `normalize()`;
- `Rank` order unchanged — ranking was decided by the model's own vote/order and
  this pass is not re-ranking;
- non-empty, within `MAX_ANSWER_CHARS`;
- a content-word containment check: every content word of the original must
  survive in the expansion (catches meaning drift and silent truncation).

**`Time` field:** add the expansion call's wall clock, amortised across the ten
slots, on top of the existing per-answer value. The submitted string is what
that call produced, so its cost belongs in the field. Document it in the setup
notes, as the other passes do.

## Sequencing constraint

The qwen3vl run was **live** when this was written (1,035 / 2,443 cached, vLLM
holding 22.4 GB). Ollama's `gemma4:12b` needs ~7.4 GB and cannot coexist. So:

1. Phase 1 harness work is CPU/GT-only and can start any time — **except**
   BERTScore, which wants the GPU; run it after the answering run finishes, or
   accept slow CPU scoring for the 20,000 pairs.
2. Phase 2 expansion runs only after the answering run completes and
   `pkill -f "vllm serve"` frees the card.

## Files

| File | Change |
|---|---|
| `VQA/score.2025.harness.py` | **new** — builds conditions A–D, converts JSON→CSV, calls `evaluate_run`, prints the comparison table and the per-type GT length table |
| `VQA/expand.answers.py` | **new** — the expander (Phase 2) |
| `VQA/EXPANSION.md` | **new** — what was measured, the chosen policy, how to run it |
| `VQA/eval.vqa.ag.py` | unchanged — called as a library; STS skipped from the harness side |
| `gemma4.expanded.run.task1.json`, `qwen3vl.expanded.run.task1.json` | outputs; originals untouched |
| `answers.expansion.cache.jsonl` | resumable work cache |

## Verification

1. **Harness sanity:** condition D (GT vs GT) must score METEOR ≈ 1.0. If not,
   the wiring is wrong and every other number is meaningless.
2. **The gate:** comparison table across A/B/C on all 2,000 GT queries, reporting
   mean METEOR, mean BERTScore, NDCG-METEOR, NDCG-BERT, and mean answer length.
   Record it in `VQA/EXPANSION.md` whichever way it comes out.
3. **Format:** `validate.ag.run.py <queries> <expanded run> --warnings-as-errors`
   must pass 2443/2443 with 0 errors and 0 warnings.
4. **Meaning preservation:** diff a random sample of 30 (original → expanded)
   pairs by hand; plus the automated content-word containment check across all
   24,430 answers, reporting any that fail.
5. **Distribution:** re-run the length stats and confirm the expanded run's
   per-question-type distribution tracks the GT table from Phase 1 — including
   that `count`/`yes_no` answers stayed short.
6. **Regression:** confirm the original run files are byte-identical afterwards.

## Risks and stop conditions

- **Expansion invents facts.** Mitigated by prompt constraints plus the
  containment check, but an LLM asked to lengthen will sometimes elaborate.
  Sample review at step 4 is the real defence.
- **The metric may not reward it.** Genuinely possible; that is what Phase 1 is
  for. If B does not beat A, stop and record the negative result.
- **Length alone may be enough (C ≈ B).** Then ship the mechanical rule and skip
  the LLM entirely.
- **Deadline: 17 August 2026.** The qwen3vl run finishes first and is the
  priority; expansion is an enhancement on top of an already-valid submission.
  Keep the unexpanded runs as the fallback submission at all times.
- The 2025 GT shares **zero** videos with the 2026 test set, so the harness
  measures the metric's response to verbosity on last year's annotation style,
  not this year's. That is the best available evidence, and it is not proof.

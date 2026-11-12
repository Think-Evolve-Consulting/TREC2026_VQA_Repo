# Model scorecard — strengths and weaknesses from the dev runs

Compiled 2026-09-17. A cross-task review of every model that has a **measured
dev run** against 2025 ground truth. Models without one are listed in
[Excluded](#excluded--no-dev-run) and are not ranked here.

Every number below was read from a score file, cache or session note on disk.
Nothing is estimated unless it says so. Source files are listed in
[Provenance](#provenance).

---

## Contents

- [The two dev sets are not comparable](#the-two-dev-sets-are-not-comparable)
- [Task 1 — answer generation](#task-1--answer-generation-150-queries)
- [Task 2 — multiple choice](#task-2--multiple-choice-400-queries)
- [Cross-task synthesis](#cross-task-synthesis)
- [Caveats that bound every number here](#caveats-that-bound-every-number-here)
- [Excluded — no dev run](#excluded--no-dev-run)
- [Open items](#open-items)
- [Provenance](#provenance)

---

## The two dev sets are not comparable

They share a corpus and nothing else. Do not put a Task 1 and a Task 2 figure in
the same sentence.

| | Task 1 (answer generation) | Task 2 (multiple choice) |
|---|---|---|
| Sample | 150 queries, 2025 GT, seed `20260814`, duration-stratified | 400 queries, 2025 MC release, seed `2026` |
| Drawn from | pool of 1,327 (GT download interrupted at Q_ID 1471) | 2,000 four-option queries, 18 scorer-SKIP excluded |
| Metric | METEOR / BERTScore / NDCG, `VQA/score.answers.py` | Top-1 / MRR, official scorer via `VQA/eval.mc.dev.py` |
| Ceiling | oracle METEOR **0.924**, BERTScore 1.000 | **0.995** — 2 queries have no correct option |
| Chance | — | **0.253 / 0.517** measured; 0.250 / 0.521 theoretical |
| Resolution | ±0.03 absolute, per the doc's own warning | ~4.5 Top-1 points at 2 SE (20% discordance) |
| Options | n/a | all 4-option; **2026 is 81% 8-option** (chance 0.340 vs 0.521) |
| Routing | on | all dev runs used `--ignore-routing` |

**Both samples are short Shorts.** Median clip 15 s. Neither exercises the 2026
set's long tail — 389 s maximum, 6.3% over 150 s. On Task 1 this means
`adaptive_fps` clamps to 2.1–4.0 fps for every clip, `LONG_VIDEO_SEC = 60` puts
everything on Gemma 4's 16-frame `CAP_SHORT` path, and the fps floor, the
frame-token degrade ladder, the 32-frame long path and the 4 GiB decode-limit
transcode are **never exercised**.

**Task 1's sample is speech-heavy in a way 2026 is not.** 94 of 150 clips (63%)
have detected speech, against 576 of 2,443 (24%) in the 2026 test set. A
pipeline that leans on the transcript is flattered by this sample. Task 2's dev
sample runs the other way — 370 of 400 have no transcript.

**Task 2's presentation order is not an answer-key leak.** In the 2025 MC
release the correct option sits at position 1/2/3/4 in 524/505/492/475 of 1,996
queries — uniform.

---

## Task 1 — answer generation (150 queries)

### Results

| Model | METEOR | BERTScore | NDCG_METEOR | words | fallback | unparsed JSON | filler slots |
|---|---|---|---|---|---|---|---|
| **Gemma 4 12B** | **0.148** | **0.892** | **0.751** | 4.9 | **0%** | 0% | 0.0 / 10 |
| Qwen3-VL-8B | 0.053 | 0.873 | 0.597 | 3.3 | 8% | 2% | 0.6 / 10 |
| Cosmos-Reason2-2B | 0.033 | 0.863 | 0.524 | 3.6 | **81%** | **20%** | **5.5 / 10** |
| *ground truth* | — | — | — | *8.5* | — | — | — |
| *oracle* | *0.924* | *1.000* | *1.000* | *8.2* | — | — | — |

Every pairwise difference is significant (p ≈ 0 on METEOR and BERTScore against
both rivals). NDCG_BERTScore is **0.993–0.994 for all three** and discriminates
nothing — drop it from future comparisons. BERTScore's whole spread across three
very different runs is 0.029, so it cannot be read as a percentage-like quality
scale either.

### Gemma 4 12B — the reliability winner

**Strengths**

- The only run with **zero degradation**: 0% fallback, 0% unparsed JSON,
  0.0/10 filler slots. On the full 2026 Task 1 run its dead-slot rate is ~0.1%.
- Wins every metric against both rivals at p ≈ 0.
- Its 16-frame `CAP_SHORT` path is **insensitive to clip length**, which is why
  it never breaks — and why it is the cheapest of the three to reason about.

**Weaknesses**

- **Its lead is partly verbosity, not accuracy.** It answers longest (4.9 words)
  and METEOR's recall term rewards exactly that. `EVALUATION.2025.md` states the
  margin should be treated as an **upper bound** on the real quality gap, not
  separable from this table without a length-controlled re-run.
- **Its ranking machinery is the weakest of the three.** The 30-candidate
  confidence vote produces an ordering only **+0.016 NDCG_METEOR above a random
  shuffle (p = 0.058 — not significant at 0.05)**. A single Qwen3-VL call
  asserting its own order does +0.064 (p ≈ 9e-5); Cosmos does +0.048 (p = 0.007).
  The ensemble buys answer *quality*, not ranking.
- **Re-ranking by its own score is actively worse**: −0.0024 NDCG, 0 queries
  improved, 14 hurt, p = 0.001. `pool_candidates()` already sorts by score, so
  re-ranking reproduces the same order minus its refusal-last tiebreak — which
  is worth ~0.002 NDCG on its own.
- **Untested where it most plausibly loses.** It sees 16 sampled frames
  regardless of duration while Qwen3-VL and Cosmos take video natively. On a
  389 s clip the ordering could reverse, and nothing here speaks to that.

### Qwen3-VL-8B — a real deficit, but smaller than it looks

**Strengths**

- **Not degraded** — 8% fallback, 0.6 filler slots, 2% unparsed. Its low score
  is a genuine quality-and-style gap, not a padding artifact. This is the
  finding that matters most, because the 2026 submission run cost 54.3 h.
- **Best ranking signal of the three**: +0.064 NDCG_METEOR over random
  (79/150, p ≈ 9e-5), from one call that simply asserts an order.
- **Gains the most from expansion**: +0.081 METEOR, the largest of the three,
  precisely because it started shortest.

**Weaknesses**

- **Terse to a fault**: 3.3 words against the annotators' 8.5, forfeiting recall
  METEOR cannot give back.
- The decisive caveat: **expanded Qwen3-VL (0.134) is statistically
  indistinguishable from baseline Gemma 4 (0.148)** — mean Δ −0.0134, p = 1.0.
  A real part of its apparent deficit was style, not vision.
- **The `size.longest_edge` bug**: it is a *whole-video* pixel budget, not
  per-frame. Read per-frame as `generate.answers.qwen3vl.py` does, two clips got
  485 and 399 visual tokens against a planned 17,792 and 5,376 — the Task 1 run
  effectively saw thumbnail-resolution video. Found during Task 2; the port-back
  to the Task 1 script is still an open item.

### Cosmos-Reason2-2B — broken, not bad

**Strengths**

- None measurable on this sample. Its per-answer quality *where it answered* is
  better than 0.033 implies.

**Weaknesses**

- **20% of responses return unparseable JSON** on short clips, so 81% of queries
  needed fallback padding and it averages **5.5 of 10 slots filled with literal
  non-answers** ("unknown", "none"). The metric averages over all ten slots, so
  more than half its score is computed over filler.
- The correct reading: *"Cosmos is worse than Qwen3-VL"* is **not** what this
  row demonstrates. *"Cosmos fails to emit parseable output on short clips"* is.
- On the full 2026 Task 1 run: 46% of slots generic filler, 45% of the run
  generated with no video attached. **Dropped from the submission** on 15 Aug.

### Cross-cutting finding 1 — expansion mostly buys length, not meaning

`VQA/expand.answers.py` was built because every run answers at roughly half the
ground truth's length against a metric that penalises exactly that. Three arms
were scored together:

| Arm | METEOR (gemma4 / qwen3vl / cosmos) | words |
|---|---|---|
| **A** baseline | 0.148 / 0.053 / 0.033 | 4.9 / 3.3 / 3.6 |
| **B** LLM rewrite | **0.187 / 0.134 / 0.076** | 7.7 / 7.1 / 5.6 |
| **C** mechanical padding, *no model* | 0.172 / 0.100 / 0.057 | 8.2 / 7.8 / 6.2 |

B beats A everywhere at p ≈ 0 and reduces neither BERTScore nor NDCG, so the
gate clears and B shipped. But:

> **Mechanical padding — a transformation that adds exactly zero information —
> captures 56–62% of the LLM expansion's METEOR gain.**

| Run | A | B | C | C's share of B's gain |
|---|---|---|---|---|
| gemma4 | 0.148 | 0.187 | 0.172 | **62%** |
| qwen3vl | 0.053 | 0.134 | 0.100 | **58%** |
| cosmos | 0.033 | 0.076 | 0.057 | **56%** |

**METEOR on this task is substantially a length-matching measure.** BERTScore is
the honest signal for quality, and it moved by at most +0.011 (Qwen3-VL) — for
Gemma 4 it moved −0.0003, p = 0.134, indistinguishable from zero.

Two further honest limits on B itself:

- **86% of rewrites invent something.** The expander has no video, so every
  added specific is a guess: 909 / 951 / 543 rewrites added at least one content
  word absent from the original, ~2 each. METEOR rising means the additions are
  net-helpful *to the metric*, not that they are true.
- **C is unsubmittable as-is.** Padding collapses distinct short answers into
  identical strings ("a flip" and "the flip"), failing `validate.ag.run.py` —
  2 duplicate slots in gemma4, 1 in qwen3vl. Only 3 in 4,500, but any duplicate
  sinks a run.

### Cross-cutting finding 2 — ordering cannot move the score, and no signal reaches the headroom

`eval.vqa.ag.py` scores a query as `np.mean` over all ten answers, which is
order-independent. Measured, as a conservation check:

| Ordering | gemma4 METEOR | qwen3vl | cosmos |
|---|---|---|---|
| current / random / reverse / oracle | **0.148** | **0.053** | **0.033** |

Identical to 3 dp across every ordering. **Only NDCG responds to order:**

| Run | reverse | random | **current** | score-reranked | oracle | headroom |
|---|---|---|---|---|---|---|
| gemma4 | 0.734 | 0.735 | **0.751** | 0.749 | 0.933 | **+0.182** |
| qwen3vl | 0.516 | 0.533 | **0.597** | — | 0.833 | **+0.236** |
| cosmos | 0.471 | 0.477 | **0.524** | — | 0.753 | **+0.229** |

0.18–0.24 NDCG_METEOR sits between the current ordering and a perfect one on
every run at p < 1e-18, and **no available signal reaches it**:

| Signal | Within-query Spearman vs per-answer METEOR |
|---|---|
| emitted `rank` | 0.106 (gemma4) / 0.134 (qwen3vl) / 0.087 (cosmos) |
| `score` | 0.097 |
| `votes` | **−0.015** (pooled −0.073) |

The signal has almost no resolution to begin with: a mean of **5.3 distinct
score values across ten answers**, **69% of answers at ≤1 vote**, ties broken by
which prompt happened to surface the answer first. Capturing the headroom needs
something that does not exist yet — sequence logprobs at generation time, or a
trained reranker. Post-hoc re-ranking of cached output provably cannot get there.

**Calibration note, easy to get backwards:** a null test with a *synthetic
uniformly-random judge* scored pooled Spearman **0.0295** — indistinguishable
from `score`'s 0.032. **Pooled Spearman ≈ 0.03 is the noise floor, not a weak
signal.** Yet the existing fuser beat random at *selection* by +0.0203 METEOR
(98W/30L, p < 0.0001). Judge any ranking signal on the arm comparison and the
sign test, never on the correlation.

### The ensemble arm — a result not written up anywhere else

`eval.2025.ensemble.scores.csv` (16 Aug 21:03) has never been transcribed into a
session note. It is not favourable.

| Run | METEOR | BERTScore | NDCG_METEOR |
|---|---|---|---|
| **gemma4_old** (single model, 10 prompts) | **0.148** | 0.892 | **0.751** |
| gemma4_ens | 0.145 | 0.891 | 0.739 |
| **fused** (2-model, 60 candidates/query) | **0.133** | 0.889 | 0.713 |
| qwen3vl_ens | 0.059 | 0.873 | 0.598 |
| qwen3vl_old | 0.053 | 0.873 | 0.597 |

Paired, n = 150:

| Baseline | Candidate | Mean Δ METEOR | W / L / T | p |
|---|---|---|---|---|
| fused | gemma4_ens | **+0.0116** | 81 / 51 / 18 | **0.011** |
| fused | gemma4_old | **+0.0149** | 78 / 65 / 7 | **0.041** |
| gemma4_ens | gemma4_old | +0.0033 | 72 / 72 / 6 | 0.53 (ns) |
| fused | qwen3vl_ens | −0.0744 | 17 / 119 / 14 | 0.0 |

`GATE_2025_RUNBOOK.md:268` states the decision rule for exactly this case:
**"fused < gemma4_ens → the ensemble lost; prefer the Gemma line."** The
ensemble-prompt variant did not help Gemma either (0.145 vs 0.148, p = 0.53).

The fused run's only demonstrated win is mechanical — dead slots 5.1% against
34.1% for Qwen alone and 46.2% for Cosmos — but **Gemma alone is already ~0.1%**,
so pooling solved a problem Gemma did not have. Slot provenance in the shipped
fused run: Gemma 62.5%, Qwen 23.1%, both agreeing 9.2%, fallback 5.1%.

Also measured: **exact cross-model agreement is nearly useless** — a mean of 0.6
candidates per query matched between the two models after casefolding, and
1,677 of 2,443 queries had none.

The AG window closed 17 Aug, so this is a post-mortem rather than a decision.

---

## Task 2 — multiple choice (400 queries)

### Results

| Model | Top-1 | MRR | Paired vs Qwen3.8 | Wall clock (2026 run) |
|---|---|---|---|---|
| **Qwen3.8-27B INT4** | **0.800** | **0.884** | — | 7.53 h |
| Qwen3-VL-8B Instruct | 0.790 | 0.872 | −1.0, p = 0.70 | 18.88 h |
| InternVL3.5-30B-A3B Q4 | 0.760 | 0.859 | −4.0, p = 0.064 | 1.99 h |
| Muse Glimmer 30B Q4 | 0.755 | 0.856 | −4.5, p = 0.018 | *not run* |
| VideoLLaMA3-7B bf16 | 0.738 | 0.843 | −6.2, p = 0.004 | 6.12 h |
| Gemma 4 12B QAT INT4 | 0.733 | 0.840 | −6.8, p = 0.0005 | 3.01 h |
| LLaVA-Video-7B bf16 | 0.725 | 0.835 | −7.5, p = 0.0008 | 1.85 h |
| Cosmos-Reason2-2B | 0.650 | 0.783 | −15.0, p ≈ 0 | 0.34 h |
| random control | 0.253 | 0.517 | −54.8 | — |

Rank distribution behind the MRR figures — every run emits a full permutation,
so the correct answer is always somewhere in the list:

| Model | rank 1 | rank 2 | rank 3 | rank 4 | answer not among options |
|---|---|---|---|---|---|
| Qwen3.8 | 320 | 50 | 19 | 9 | 2 |
| InternVL3.5 | 304 | 58 | 19 | 17 | 2 |
| VideoLLaMA3 | 295 | 60 | 18 | 25 | 2 |
| LLaVA-Video | 290 | 61 | 22 | 25 | 2 |

**MRR 0.884 against Top-1 0.800 means a miss usually still lands at rank 2** —
the tail carries signal, which is why fusion looked promising before it was
measured.

### The ranking contract — shared by every pass, and why it matters

- **Rank option numbers, never option text.** The model returns a permutation of
  `1..N`, mapped back to official strings by index, so no rank is spent twice
  and no option goes unranked. `as_permutation()` repairs repeats and drops;
  `check_permutation()` refuses to write anything the validator would reject.
- **Permute the presentation**: official order, exact reverse, seeded shuffle,
  combined by Borda count. Measured position bias is at chance in every run —
  rank 1 on the officially-first option 14.9–17.4% against 14.9% expected.
- **No `Time` field** in MC runs; the validator rejects any key outside
  `Rank` / `Answer`.

Because this contract holds for all eight, the dev differences are about vision
and reasoning, not output hygiene — with the single exception of Cosmos, below.

### Qwen3.8-27B INT4 — the baseline nothing beats

**Strengths**

- Best measured ranker, and clean: all 7,329 replies parsed, 0 repairs.
- **2.5× faster than Qwen3-VL at the same 4,096-token visual budget** — 7.53 h
  against 18.88 h. Workable on a 24 GB card because 48 of 64 layers are linear
  attention: 17.71 GiB loaded leaves 3.3 GiB of KV cache = 46,421 tokens.
- Reasons in plain `content`, which the JSON schema suppresses cleanly.
- **Second best at counting** despite counting being its own worst category.

**Weaknesses**

- **No kwarg controls its temporal sampling.** `do_sample_frames` is rejected on
  1,765 of 2,443 queries (72%); `num_frames` is silently ignored without it
  (identical token counts at 16 and 64) and rejected with it. A full re-run
  under a transcode workaround (`qwen38.resampled.run.task2.json`, 7.52 h)
  changed rank-1 on **6.3% of targeted queries against 8.0% of controls** — the
  fix made **no measurable difference**, which is itself the finding.
- **Counting**: 43.8% error against 16.8% elsewhere (48 queries, p = 0.0001);
  "all options numeric" 41.0%.
- **18 queries are missed only by Qwen3.8**, and the one theme is
  **"what can be heard" questions answered from silent video.**
- Requires vLLM **0.27** — only 0.27 registers `Qwen3_5ForConditionalGeneration`.

### Qwen3-VL-8B Instruct — statistically tied, and the most complementary

**Strengths**

- **Tied with the leader at p = 0.70** on a quarter of the parameters.
- **Best counter in the set**: 38% error against Qwen3.8's 44%.
- **The only near-break-even pair with the baseline**: 29 right where Qwen3.8 is
  wrong against 33 wrong where Qwen3.8 is right. Every other model is far more
  lopsided.
- Clean: 7,329 / 7,329 parsed, 0 repaired, 0 failed, 0 fallback, 83.6% unanimous
  across its three passes.

**Weaknesses**

- **Slowest good option** — 18.88 h, and the estimate was wrong: ~15 h was
  quoted because the prefix-cache saving measured at 8,192 tokens did not
  transfer to 4,096 (passes 2–3 cost 30.7 s each, not the 2–3 s seen at 8k).
- Carries the **whole-video `size.longest_edge` budget** trap described under
  Task 1. Holding clip and fps fixed, 2,048 / 8,192 / 16,384 tokens produce
  1,152 / 4,004 / 8,096 prompt tokens on a 62-frame clip.
- Its Instruct weights were a 12 KB `refs/main` stub; 17 GB had to be
  re-downloaded before anything ran.

### InternVL3.5-30B-A3B Q4 — best of the late additions

**Strengths**

- Closest challenger at 0.760, and **not significantly behind (p = 0.064)**.
- Fastest 2026 run of the three additions at 1.99 h, 80.8% unanimous, 13
  repairs, 0 failed passes.
- **Image prefix genuinely reuses across passes** — 4,324 of 4,561 tokens cached
  on pass 2 — unlike Muse, whose reordered passes reuse only the system prompt.
- Needed no new client: the Muse llama-server script serves it with flags.

**Weaknesses**

- Loses 41 queries Qwen3.8 gets right while winning only 25 — the second-worst
  exchange rate among the credible models.
- **The tiling trap**: llama.cpp cuts every InternVL image into 448-px tiles.
  One 504×896 frame cost ~1,280 tokens, 16 frames came to **20,945** against a
  16k slot, and every image request returned HTTP 400. `--image-max-tokens` does
  not cap it for this architecture. The fix is client-side (`--frame-size 448`,
  one tile per frame, matching InternVL's own video path `max_num=1`), giving
  4,561 prompt tokens.
- The originally requested "InternVL 3 26B at 4-bit" **does not exist** — the
  InternVL3 line is 1/2/8/9/14/38/78B, and the 26B models are InternVL2 / 2.5,
  whose official AWQ builds are 23.3 GB because the 6B vision tower stays fp16.
  Nothing in that family fits a 24 GB card with room for frames.

### Muse Glimmer 30B Q4 — cleanest run in the set, and still failed its gate

**Strengths**

- **The cleanest run of any model**: 0 failed passes, 0 repairs, 0 fallbacks,
  0 budget retries, **86.2% of queries unanimous** across passes.
- **No position bias of its own**: rank 1 on the officially-first option in
  32.2% of queries, against 27.8% where the correct answer really sits first;
  Qwen3.8 and Qwen3-VL both score 31.2% on the same check.

**Weaknesses**

- **Significantly worse than baseline** (−4.5, p = 0.018; 17 won / 35 lost;
  MRR −0.028, CI −0.048 to −0.008). Adding it to the Qwen pair changed nothing
  (RRF 8:8:4 → 0.800 / 0.884, p = 1.0, 17 / 17).
- **Failed its pre-registered gate**, so no 2026 run was made. This is the
  discipline working: the gate was "tie Qwen3.8, or a fusion including it beats
  0.800" and both conditions failed.
- Two silent-cost traps: the chat template **defaults to reasoning strength
  `high`** when the kwarg is absent, and `max_image_tokens` is **4,096 per
  image** natively. Either left alone would have made the run silently expensive
  or overflowed the context.
- Projected 2026 cost 16–20 h at `low`; `medium` roughly doubles decode (~40 h)
  and could not have finished before the window closed.

### VideoLLaMA3-7B bf16 — correct timestamps, hostile memory behaviour

**Strengths**

- **The only pass with true timestamps in the prompt.** Its loader stamps a
  *directory* of frames at a fixed `vid_fps = 3`, which would mislabel every
  frame; passing a `video` item that is already a list of CHW arrays skips the
  loader, so `num_frames` and `timestamps` are ours and the template prints the
  real `Time 0.6s:<image>, Time 1.7s:<image>, …`.
- Beats Gemma 4 at less than two-thirds the parameters.
- 70.0% unanimous, 6 repairs, 0 failures on the 2026 run.

**Weaknesses**

- **The one real failure of that session.** Its sdpa vision encoder builds a full
  `[S, S]` boolean mask over every patch and hands it to
  `scaled_dot_product_attention`, materialising heads × S × S scores — about
  8 GB per layer at 16k patches. **Windows spills the overflow into system RAM
  instead of raising OOM**, so the first smoke test sat at **23.7 GB and 100%
  GPU for over 10 minutes with zero queries finished.** Treat "busy but no
  output" as a memory symptom. The block-wise monkeypatch is exact — max abs
  difference **0.00000** against the masked original on a real layer — and took
  it to 7.1 queries/min.
- Loses 48 to win 23 against Qwen3.8.
- **Cannot use vLLM at all** (no architecture, issue #16627) and its remote code
  pins **transformers 4.46.3**, so it needs a dedicated venv
  (`.venv-videollama3`: Python 3.11, torch 2.6.0+cu124, numpy<2, decord) and
  runs in-process. Slowest 2026 pass at 6.12 h.

### Gemma 4 12B — the cross-task inversion

**Strengths**

- **Cheapest credible model by a distance**: 3.01 h, 0 failed, 0 repaired,
  0 fallbacks, 79.7% unanimous.
- Small, well-behaved position bias that the permutation averages out: raw
  per-pass rank-1 at presented position 1 is 16.2% against 14.9% expected.

**Weaknesses**

- **Best Task 1 model, second-worst real Task 2 model** (−6.8, p = 0.0005). The
  fixed 16-frame sampling that makes it so robust at *generating* an answer is
  not enough to *discriminate* between written options.
- Its Borda winner differs from a single official-order pass on 9.7% of queries,
  so the three-pass machinery is doing real work — it just is not enough.

### LLaVA-Video-7B bf16 — weakest real ranker

**Strengths**

- **The only pass that sends the clip itself** (`video_url`, 32 frames sampled
  server-side, ~6,300 prompt tokens); the frame index is read only for duration.
- Fastest 2026 run at 1.85 h, with **0 repairs** — the only one of the three
  additions needing none.

**Weaknesses**

- Weakest real ranker at 0.725, and **the worst exchange rate in the set**:
  23 won, 53 lost against Qwen3.8.
- **Reports confidence 0.0 on every reply**, which flattens its Borda weights.
  Harmless in isolation, but its confidence carries **no information for
  fusion** — worth knowing before weighting it.
- **The `.webm` frame-count trap**: OpenCV reports one frame more than it can
  decode for 256 of the 1,088 2026 `.webm` files (251 off by one, 5 by 2–4;
  every `.mp4` exact). vLLM samples `linspace(0, n-1, 32)`, so the last index is
  never reached — worst case 30 of 32 frames. vLLM v0.12.0 only warns and uses
  what it got, so no re-encoding was needed, but a stricter version would break
  here. Full scan: `videos_TEST_VQA/webm_framecount_report.json`.
- Served only as `Isotr0py/LLaVA-Video-7B-Qwen2-hf`, a conversion of the
  lmms-lab checkpoint to the `llava_onevision` architecture — the only form vLLM
  can serve.

### Cosmos-Reason2-2B — worst in both tasks, for the same reason

**Strengths**

- Fastest of all (0.34 h) and by far the smallest.
- Once repaired: 2,049 replies parsed as JSON, 35 salvaged from a digit list,
  359 recovered via the ordering call, 60 repaired, **0 failed**.

**Weaknesses** — three distinct failure modes, all found the hard way:

1. **A reasoning model chooses; it does not rank.** 9 of the first 20 replies
   were a bare `{"ranking": [4]}`. Three prompt variants moved full-length
   rankings from 1/6 to 3/6 and no further. The fix is a second text-only,
   schema-constrained call that orders the remaining options from the model's
   own pick and reasoning — needed on **37.7% of queries**, ~0.6 s against ~17 s
   for the reasoning call.
2. **23% silent blindness.** 190 of the first 830 queries were ranked with **no
   video although the file existed**: the server decodes every frame
   (`num_frames=-1`), and asking the processor to re-sample (`do_sample_frames`)
   raised a 400 that the client treated as "unprocessable video" and retried
   text-only. Fixed with a ladder — sampled video → unsampled video → text only.
3. **OOM kill.** The unsampled retry let the processor resize every native frame
   (300 for a 10 s clip instead of ~40); at 8 workers that OOM-killed the server
   790 queries into stage two, leaving 10 fallback rankings. Fixed by always
   transcoding to target fps on that path, `RAW_DECODE_BYTES_LIMIT` 4 → 2 GiB,
   and workers 8 → 4.

- **The outlier in every pairwise agreement**: 61–64% rank-1 agreement with the
  others, against 77–81% among themselves.
- 15 points behind, losing 86 queries to win 26.

### Fusion — measured properly, and it does not work here

`VQA/search.fusion.mc.dev.py` fuses by RRF (k=60) and Borda over every subset
and every integer weighting 0..4 of eight runs — **781,248 fusions** — judged by
choosing the best on nine folds and scoring the held-out tenth, over 20 random
splits. It first proves its own scoring reproduces the official scorer query by
query for all eight single runs.

| Search | Best on the full 400 | Nested CV | CV repeats above baseline |
|---|---|---|---|
| Weight grid, 781,248 fusions | 0.825 / 0.896 | **0.801 / 0.883** | 9 / 20, median p = 0.87 |
| Equal-weight subsets, 510 | 0.810 / 0.890 | 0.778 / 0.871 | 0 / 20 |

**The entire in-sample gain is search optimism.** Under honest evaluation the
best of 781,248 fusions is worth exactly the baseline.

Fusions named in advance — no search, so their in-sample numbers are honest:

| Fusion | Top-1 | vs Qwen3.8 |
|---|---|---|
| Top-4 equal Borda | 0.810 | +1.0, p = 0.57 |
| Top-3 equal RRF | 0.807 | +0.7, p = 0.74 |
| Accuracy-weighted RRF 8:8:4:1 | 0.795 | −0.5, p = 0.86 *(in-sample weights)* |
| All eight equal RRF | 0.790 | — |
| Qwen3.8 + Qwen3-VL | 0.782 | — |
| Borda 3 models, equal | 0.767 | −3.2, p = 0.053 |
| RRF 3 models, equal | 0.765 | −3.5, p = 0.029 |

**Why fusion cannot help.** At least one run puts the right answer first on
**93.5%** of queries — the headroom looks enormous — but every model loses more
to Qwen3.8 than it wins:

| Model | Right where Qwen3.8 is wrong | Wrong where Qwen3.8 is right |
|---|---|---|
| Qwen3-VL | 29 | 33 |
| InternVL3.5 | 25 | 41 |
| VideoLLaMA3 | 23 | 48 |
| LLaVA-Video | 23 | 53 |
| Muse | 17 | 35 |
| Cosmos | 26 | 86 |

There is no majority to harvest. Voting mixes in more error than it removes.

### Where Qwen3.8 is wrong — 80 of 400 dev queries

Tested across ~28 features — question type, clip length, transcript presence,
how the video was delivered, answer position, pass agreement — with exact Fisher
tests at a Bonferroni threshold of p < 0.0018.

| Feature | Finding |
|---|---|
| **Counting** | **43.8% error vs 16.8% elsewhere** (48 queries, p = 0.0001); "all options numeric" 41.0% |
| Direction of count error | **Systematic undercount** — of 18 readable cases, 15 too low, 3 too high, mean −1.2. Universal: every model undercounts, mean −0.4 to −1.3 |
| **Split passes** | 46.4% error on the 69 queries where its three passes disagreed, vs 14.5% when unanimous |
| Answer-length artefact | When the correct answer is the longest option (186 / 400), error is 11.8% vs 27.1% — a property of the 2025 option writing, not of the videos |
| **Not significant** | clip length, transcript presence, speech/sound questions, on-screen text, why/what-happens types, answer position |
| Universally missed | 26 queries missed by all eight models — counting, audio-only, fine detail, a few doubtful labels |

**Split passes are a good confidence signal and a useless routing signal.** Every
other model also errs 48–59% on those same queries, so they are hard for
everyone — there is nobody to route them to.

---

## Cross-task synthesis

**1. Counting is the one shared, measured weakness across all eight models.**
Every model systematically *undercounts*. 43.8% error against 16.8% elsewhere.
Untested remedies: a count-specific prompt, or denser frames. 48 dev queries is
too few to measure either.

**2. Robustness beat capability on both tasks.** Cosmos-Reason2 is the only
dedicated reasoning model in the set and it finished last in both. Every large
gap traced back to an I/O, parsing or sampling defect — unparsed JSON, silent
blindness, a whole-video pixel budget, a tiling explosion, a spilled attention
mask — not to vision quality. The wins went to whichever model reliably emitted
parseable, complete output.

**3. Model size barely predicts rank.** Qwen3-VL-8B ties a 27B; VideoLLaMA3-7B
beats Gemma 4 12B; InternVL3.5-30B (3B active) is mid-pack. **Two 7B models beat
two ~30B models.** Serving geometry and prompt hygiene explain more of the
variance than parameter count does.

**4. Task rank does not transfer.** Gemma 4 wins Task 1 outright and comes
second-to-last among real Task 2 models. Generating a good free-text answer and
discriminating among eight written options are different capabilities, and a
model selected on one should not be assumed on the other.

**5. Self-reported confidence is worthless in this project, twice over.** Task 1:
`score` Spearman 0.097, `votes` **negative**, 96.9% of router confidences came
back at exactly 1.0, and re-ranking by score made NDCG worse. Task 2:
LLaVA-Video reports confidence 0.0 on every single reply. Keep every signal in
the cache so it can be re-weighted for free, but do not spend GPU on it.

**6. Every ensemble and fusion attempt has lost or tied — four times now.**
Task 1 two-model fusion lost to Gemma alone (p = 0.011). Task 2 equal-weight RRF
lost 3.5 points (p = 0.029). Accuracy-weighted RRF tied, with in-sample weights.
The 781,248-fusion search won in-sample and evaporated under nested CV. The
prediction that fusion was "the highest-value free move" has now been falsified
in both tasks.

**7. The gates are doing their job.** Muse Glimmer was measured, failed a
pre-registered gate, and was never run over 2026 — saving 16–20 h. The judge
rerank's null test correctly refused a synthetic random judge (−0.0148 vs the
fuser, 37W / 88L, p < 0.0001). The fusion search's nested CV correctly refused
its own in-sample winner. Measure-before-you-run is the process fact worth
keeping.

---

## Caveats that bound every number here

- **Quote as "on a 150-query 2025 sample" or "on a 400-query 2025 MC sample",
  never as "our METEOR" or "our accuracy."** Per-query spread far exceeds the
  differences between runs; the Task 1 absolute carries roughly ±0.03.
- **~1-point Task 2 differences cannot be resolved at n = 400.** Settling
  qwen38-vs-qwen3vl means extending toward all ~1,980 usable 2025 queries, which
  means re-running every model. Two *random* runs differed by 3.0 points on this
  harness and the test correctly said p = 0.36.
- **Short video only, both tasks.** The axis where the pipelines most plausibly
  diverge — long video — is exactly what neither sample tests.
- **Mild style leakage on Task 1.** `MAX_ANSWER_WORDS = 40` and the shared "most
  good answers are under ten words" prompt rule were tuned against this very
  ground truth's length distribution, which flatters the length-sensitive part of
  METEOR. It cannot be removed without changing the pipeline under test.
- **Task 1's pool is 1,327 of 2,000** — a positional slice, not the whole GT.
- **Two scorer traps, both live.** `eval.vqa.mc.py` divides by the *ground-truth*
  file size, not the run, so scoring a sample against the full GT reports ~5× too
  low; and it rounds each reciprocal rank to 3 dp (`round(1/rank, 3)`), which is
  why a re-implemented MRR disagreed on 19 / 400 until it rounded the same way.
- **Sample fidelity is good on routing, bad on speech.** Routed labels land
  within ~3 pp of the 2026 set on every route; the one real gap is budget tier
  (brief 20.7% vs 29.5%), so the sample skews toward more expensive tiers. The
  speech gap (63% vs 24%) is the one that should change how Task 1 numbers are
  read.
- **The scorer itself is trusted infrastructure.** Parity with `eval.vqa.ag.py`
  is exact to 3 dp over 20 queries × 10 answers with zero per-query
  disagreements, and the oracle check passes. Any future ablation can be
  measured rather than argued.

---

## Excluded — no dev run

Listed so the absence is deliberate rather than an oversight.

| Model / configuration | Why there is no dev number |
|---|---|
| Muse Glimmer 30B, **Task 1** | Abandoned unrun on 14 Aug 2026; never scored |
| Qwen3.8-27B, **Task 1** | Unfinished. Unconstrained it spent 489–592 s per query at ~5,200 output tokens and padded 6 of 10 slots on the third query (reasoning preamble truncated before the JSON); the `--json-schema` run was stopped before producing a timing. Moot — the AG window closed 17 Aug. `answers.task1.qwen38.cache.jsonl` holds 1 stale record; delete before reuse |
| Qwen3.8-Flash-Next | Assessed, not attempted. 125B parameters plus a 51B n-gram embedding table (6B active, 512 experts, 262k context); smallest builds 61.9–111.5 GB. A 32 GB Mac cannot run it; this box loads it but would prefill ~4,500 visual tokens mostly on CPU — 30–120 s per pass against InternVL's 0.9 s, i.e. days for 7,329 calls. The practical route is the hosted API (~$5–6 per Task 2 pass), which means sending TREC frames to a third party — a rules question, and the user's decision |
| `fused.rrf` / `fused.borda` (4-model Task 2) | Valid run files exist but were never scored in that exact form |
| `qwen38.resampled` (Task 2) | Not scored on dev; near-duplicate of the original (167 rank-1 differences) |
| Pointwise likelihood scoring | Task 2 plan step 3 — not started (needs the 8011 server) |
| Generate-then-match | Task 2 plan step 4 — the 2025 Task 1 runs cover only 42 of the 400 sampled MC queries, so it needs a fresh Task 1 pass |

---

## Open items

1. **Choose and submit ≤ 5 MC runs — the window closes 18 Sep 2026.** Twelve
   valid files exist. Recommended five, unchanged by any later session:
   **qwen38 → qwen3vl → fused.weighted → gemma4.vllm → fused.rrf**. Confirm the
   data agreement form is filed; runs are not scored without it.
2. **Counting** is the one weakness shared by every model and the only obvious
   quality lever left. Untested.
3. **Port the fixes back** for any future run: the whole-video `longest_edge`
   budget → `generate.answers.qwen3vl.py`; the `do_sample_frames` ladder →
   `generate.answers.cosmos.py`.
4. **Capture logprobs on the next vLLM run.** They are cheap at generation time
   and the only plausible route to Task 1's 0.18–0.24 NDCG_METEOR headroom.
5. **Disk**: E: was at ~37 GB free. `models/Muse-Glimmer-30B/` (55.5 GB bf16) is
   unused and deletable; the Muse GGUF (19.7 GB) too unless Muse is revisited.
6. **The ensemble gate result** in
   [that section above](#the-ensemble-arm--a-result-not-written-up-anywhere-else)
   exists only in `eval.2025.ensemble.scores.{csv,json}` and had never been
   written up before this document.

---

## Provenance

Every figure traces to one of these.

**Task 1**

| File | What it supplied |
|---|---|
| `VQA/EVALUATION.2025.md` | headline table, run health, expansion arms, re-ranking, sample fidelity |
| `eval.2025.scores.csv` / `.json` | baseline METEOR / BERTScore / NDCG |
| `eval.2025.expansion.scores.csv` | A / B / C expansion arms |
| `eval.2025.ensemble.scores.csv` / `.json` | the ensemble arm and its paired Wilcoxon tests |
| `eval.2025.rerank.scores.csv` | the five orderings and the conservation check |
| `ranking.analysis.2025.md` | METEOR by emitted rank, Spearman, signal resolution |
| `RERANK_PLAN.md` | the judge design and the null-test calibration |
| `GATE_2025_RUNBOOK.md` | the ensemble gate's decision table |
| `SESSION.2026-08-15.evaluation-and-expansion.md` | orchestration and expansion detail |

**Task 2**

| File | What it supplied |
|---|---|
| `SESSION.2026-09-15.task2-mc-ranking.md` | the first four passes, the dev harness, early fusion |
| `SESSION.2026-09-16.task2-extra-rankers.md` | the three additions, the fusion search, the Qwen3.8 error anatomy |
| `SESSION.2026-09-15.muse-glimmer-task2.md` | Muse Glimmer's dev run and its failed gate |
| `eval.2025.mc/*.score.json` | all 15 Top-1 / MRR figures |
| `eval.2025.mc/fusion.search.json` | the nested-CV fusion search |
| `eval.2025.mc/*_diagnostic.csv` | per-query rank distributions |

Memory notes consulted: `trec2026-ensemble-submission`, `judge-rerank-gate`,
`task2-mc-ranking-pipeline`, `task2-mc-dev-harness`, `task2-extra-vlm-rankers`,
`cosmos-mc-ranking-traps`, `qwen3vl-visual-budget-bug`, `qwen38-27b-int4-serving`,
`muse-glimmer-answering-pipeline`.

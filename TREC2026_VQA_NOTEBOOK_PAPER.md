# Measure Before You Ship: An Open-Weight, Single-GPU System for the TREC 2026 Video Question Answering Track

**Aakash Gupta** — Think Evolve Labs
*TREC 2026 Notebook Paper — Video Question Answering (VQA) Track, Answer Generation (Task 1) and Multiple Choice (Task 2)*

---

## Abstract

We describe our participation in both tasks of the TREC 2026 Video Question
Answering track under a deliberately narrow budget. We used one consumer GPU
(an RTX 3090 with 24 GiB usable), open-weight models only, and roughly five
weeks. Our system pre-computes two independent speech transcriptions, uniformly
sampled frames, ten paraphrases per question, and a six-agent text-only router
that decides which inputs each question needs. Eight vision-language models
(VLMs) were then served one at a time. For Answer Generation (AG) we submitted
five runs built on Gemma 4 12B, Qwen3-VL-8B and a two-model fusion, with and
without an LLM answer-expansion pass. For Multiple Choice (MC) we submitted five
runs built on Qwen3.8-27B, Qwen3-VL-8B, InternVL3.5-30B-A3B and a weighted rank
fusion.

Every design choice was gated on labelled 2025 development sets. Each gate used
a parity-checked scorer, null controls, paired significance tests and, where we
searched over configurations, nested cross-validation. Most of the instructive
results are negative.

- No fusion or arbitration scheme beat the best single model on either task.
  The best of 781,248 searched MC fusions scored 0.825 in-sample and exactly the
  baseline (0.801) under nested cross-validation.
- A mechanical padding control that adds no information captured 56–62% of the
  METEOR gain of LLM answer expansion.
- The official STS metric, which we could not compute during development, was
  the one that penalised expansion.

Our best MC run reached 0.803 Top-1 and 0.877 MRR, against a field median of
0.801 Top-1. Our development estimate for this run was 0.800. Our best AG runs
sit in the middle of the field on BERTScore@1 and below the median on
METEOR@1. Our ten-answer sets were stronger than our rank-1 choices:
Best@10 places 9th of 44 runs, against 18th–23rd at @1. The 2026 scoring rewards
rank-1 selection, and our pipeline had been optimised against the 2025 scorer's
mean over all ten answers.

---

## 1 Introduction

TREC 2026 VQA asks systems to answer natural-language questions about short
social videos and film clips. In Task 1 (Answer Generation, AG) a system
returns ten ranked free-text answers. In Task 2 (Multiple Choice, MC) it ranks
every official option. Our participation was shaped by three constraints:

1. **One GPU.** Every model we used occupies essentially the whole card. So no
   two models can co-reside, and every "agent" in a multi-model design becomes a
   sequential pass over the dataset, with a server teardown and reload between
   passes.
2. **Open weights, local inference.** No hosted APIs were used. One larger model
   was assessed and rejected because the only practical way to run it was a
   hosted API, which would have sent track data to a third party.
3. **Little time.** The AG window closed on 17 August 2026 and the MC window on
   18 September 2026.

Under these constraints the dominant risk was not a weak model but a *silent*
failure: a run that completes, validates and is wrong. Examples include a model
answering with no video attached, a frame budget silently disabled, or a
development query scored against the wrong year's questions. Our working rule
became *measure before you ship*. No component entered a submission without a
measured gain on labelled 2025 data under a test fixed before the numbers were
read. The one exception was the AG ensemble, which shipped under deadline
pressure with its gate deliberately skipped (§6.5), and we report how that
turned out.

Our contributions are:

- An end-to-end open-weight pipeline for both tasks on a single 24 GB GPU,
  together with a catalogue of the serving traps we found along the way (§9).
- A development protocol that produced trustworthy negative results. It uses
  parity-checked scorers, oracle and null controls, paired tests,
  pre-registered gates and nested cross-validation (§5).
- Measured negative results on rank fusion, VLM-as-judge arbitration and
  self-reported confidence (§6.4–6.6, §7.4–7.6).
- A metric finding: METEOR on this task is largely a length-matching measure
  (§6.3). The official 2026 results show that the gain bought on METEOR was
  paid for on STS (§6.7).
- An analysis of how well development estimates transferred to the official
  2026 results (§6.8, §7.7).

---

## 2 Task and Data

### 2.1 The two tasks

| | Task 1 — Answer Generation | Task 2 — Multiple Choice |
|---|---|---|
| Queries | 2,443 | 2,443 (same questions and videos) |
| Output per query | exactly 10 distinct ranked answers, each with a `Time` field | a full permutation of the official options |
| Official metrics (2026) | STS, METEOR, BERTScore F1 at rank 1 (`@1`); NDCG@10 using each metric as graded relevance; two reference modes (human answer only; human + alternative answers) | Top-1 accuracy, MRR |
| Scored queries | 2,443 | 2,428 (15 excluded by NIST) |
| Options | — | **81% of queries have 8 options** |

One scoring detail shaped our development and is easy to overlook. The 2025
evaluation script, which was our only local scorer, scores a query as the
**mean over all ten answers**. That makes the score independent of answer order.
The 2026 headline metrics are **@1**: the score of the rank-1 answer alone. We
discuss the consequences in §6.8.

### 2.2 The 2026 test corpus

The 2,443 questions refer to 1,956 distinct videos: 1,088 released `.webm` clips
and 868 YouTube IDs. We downloaded 852 of the YouTube videos. After download,
16 queries had no usable media and were answered from text.

| Property | Value |
|---|---|
| Videos on disk | 1,940 (13.37 h; 23.70 h counted once per query that asks about the video) |
| Duration | median 10.3 s, max 389 s; released clips bimodal (78.4% ≤ 15 s, 17.4% > 60 s); 307 clips exactly 10.01 s |
| Long tail | 65 videos > 150 s, all from ten films: 3.35% of videos, 6.34% of queries, but **38.7% of query-weighted footage** |
| Orientation | 795 of 1,940 portrait |
| Audio | 5 videos with no audio stream; Whisper transcript for 1,046 videos (53.9%); speech relevant to 576 queries (23.6%) |
| Languages (Whisper) | 34 detected; 38.8% English, 18.0% non-English, 43.2% no speech |

Routing statistics, from §4.4, over all 2,443 queries:

| Needs | Share | | Question type | Share |
|---|---|---|---|---|
| vision | 92.5% | | action | 30.8% |
| audio | 23.6% | | entity | 18.9% |
| on-screen text (OCR) | 2.3% | | reason | 17.1% |
| inference beyond observation | 46.5% | | count | 12.8% |
| temporal ordering | 36.3% | | other types | 20.4% |

### 2.3 Development data

The 2026 test set has no public labels, so all development used **2025 ground
truth**. This is disjoint from 2026: we verified zero video overlap across
2,000 × 1,956 videos.

- **Task 1 dev:** a 150-query sample (seed 20260814) drawn from the 1,327 2025
  ground-truth videos we could still download. The download was interrupted at
  Q_ID 1471, and 9.8% of attempts failed.
- **Task 2 dev:** a 400-query sample (seed 2026) from the 2025 MC release, with
  2,000 four-option queries. Two sampled queries have no correct option among
  their choices, so the ceiling is 0.995.

Both dev sets are **short video only**. Every 2025 clip is a YouTube Short
(5–30 s, median ≈15 s). They also differ from 2026 in ways that matter:

| | Task 1 dev | Task 2 dev | 2026 test |
|---|---|---|---|
| clips over 60 s | 0% | 0% | 17.4% of released clips |
| clips with speech | 63% | 7.5% have a transcript | 24% of queries |
| options per MC query | — | 4 (chance Top-1 0.25) | mostly 8 (chance 0.125) |
| routing | on | off (`--ignore-routing`) | on |

---

## 3 Hardware, Serving and Models

All inference ran on one RTX 3090 (24 GB, sm_86). Windows permanently holds
about 1.3 GiB, so 22.72 GiB is usable and vLLM's `--gpu-memory-utilization`
must stay at or below 0.92. Model servers ran under vLLM 0.12 and 0.27 in WSL2,
llama.cpp on Windows, and Ollama on Windows. VideoLLaMA3 ran in-process.
Because no two models fit together, the work was scheduled **by resident
model**, not by pipeline stage. Each swap meant a full teardown, a check that
VRAM had actually been freed, and a reload.

| Model | Params / quantisation | Backend | Visual input | Used for |
|---|---|---|---|---|
| Gemma 4 12B | Q4_K_M (Ollama); Google QAT INT4 W4A16 (vLLM) | Ollama → vLLM 0.27 | sampled frames (base64) | T1 answers, paraphrases, routing, expansion, judge; T2 ranker |
| Qwen3-VL-8B Thinking / Instruct | bf16 | vLLM 0.12 | native video | T1 answers; T2 ranker |
| Cosmos-Reason2-2B | bf16 | vLLM 0.12 | native video | T1 answers (dropped); T2 ranker |
| Qwen3.8-27B | compressed-tensors INT4 | vLLM 0.27 | native video | T2 ranker (best) |
| InternVL3.5-30B-A3B | Q4_K_M GGUF | llama.cpp | frames, one 448-px tile each | T2 ranker |
| Muse Glimmer 30B | Q4 GGUF | llama.cpp | frames | T2 dev only (failed gate) |
| LLaVA-Video-7B | bf16 | vLLM 0.12 | native video, 32 frames | T2 ranker |
| VideoLLaMA3-7B | bf16 | transformers (in-process) | timestamped frame arrays | T2 ranker |
| Whisper large-v3 | fp16 (CTranslate2) | faster-whisper | audio | ASR |
| SenseVoiceSmall | — | FunASR | audio | ASR, audio events |

---

## 4 Shared Preprocessing

The video-independent stages ran once and were reused by every answering and
ranking pass. Each stage is resumable, keyed so it can be re-emitted from cache
without inference, and seeded deterministically.

### 4.1 Two independent ASR passes

**SenseVoiceSmall** returns language, emotion, an audio-event tag and a
transcript in one non-autoregressive pass. Three of its design decisions were
each forced by an observed failure:

- **Segmenting audio with FSMN-VAD.** Feeding the model a 173 s clip whole
  collapsed it to one label per video and garbled the text.
- **Classifying the gaps between speech spans as ambient windows.** Otherwise
  pure-music clips produced no output at all.
- **Flagging non-speech hallucinations** such as bare punctuation or a lone
  "The.", rather than deleting them.

The run took about 14 GPU-minutes for 13.35 h of audio.

**Whisper large-v3**, run through faster-whisper, was added because
**SenseVoice can emit only five languages**. 182 videos are in none of them.
For example, Hindi was labelled Korean in 36 videos. Two departures from stock
Whisper mattered:

- **Language is decided by a sweep.** Up to six VAD speech windows are each
  run through language detection, and the votes are weighted by
  *probability × window duration*. Stock Whisper reads only the first 30 s,
  which on this corpus is often a title card or music. Weighting by probability
  alone let a 2.4 s window misdetected as Welsh outvote a 389 s English
  documentary.
- **Videos with no speech get no language.** Whisper otherwise labels silence
  as Norwegian Nynorsk with total consistency.

A further 562 segments were flagged as hallucinated boilerplate. The two systems
agree on language for 76.0% of the 1,002 videos both labelled. Silero VAD
misses sung vocals over music: 87 videos with six or more SenseVoice words have
no Whisper transcript. All downstream prompts use the Whisper transcript.

### 4.2 Frames

`ffmpeg` samples *n* = clamp(round(duration), 8, cap) frames per video, at the
mid-points of uniform intervals. The cap is 16 frames, or 32 for videos over
60 s. The long edge is 896 px and each frame carries an `[mm:ss]` label. This
produced 26,250 frames for 1,940 videos in 6.1 min. Frames are extracted once
per video and reused across queries and prompts, which avoids roughly 366,000
redundant extractions.

### 4.3 Ten paraphrases per question

Gemma 4 rewrote each of the 2,350 distinct questions ten ways, and the results
were fanned out to all 2,443 Q_IDs. Decoding was schema-constrained with
thinking disabled. With thinking on, the reasoning trace consumed the token
budget and returned empty content. The prompt preserves the question type
(a counting question stays a counting question) and preserves identifying
detail. The run took 96 min with zero failures. Every query has exactly ten
unique paraphrases, none of which echoes the original.

### 4.4 A six-agent input router

Six independent text-only judgements decide what each question needs:

- vision
- audio
- OCR
- temporal ordering
- inference
- question type

The agents **never see each other's verdicts**, so a weak signal cannot
cascade, and a plain-Python policy combines them. Because the policy is plain
code, it can be changed and re-applied over the cache in seconds. Each prompt
lists the cases that look like a "yes" but are not. Examples: *mentioning a
document is not OCR*; *"what happens" is observation, not inference*. These
fixed both false positives seen in the smoke test. The run made 14,100
judgements in 170 min with zero failures.

**Self-reported confidence was useless.** 96.9% of the router's judgements came
back at exactly 1.0, and none fell below 0.7. This was the first of several
places where a model's own confidence carried no information (§6.4, §7.5).

### 4.5 From a multi-agent design to a measured schedule

Our initial architecture document described concurrent Vision, Audio and Context
agents, with a debate-and-consensus engine triggered whenever confidence fell
below 0.85. Reconciling it with the hardware removed most of it:

- **The agents became a serial schedule**, because only one model fits on the
  card at a time.
- **The debate trigger was dead on arrival**, because confidence was almost
  always 1.0 (§4.4).
- **The "ten output formats" generator was inverted.** The variation moved to
  the *input* side, as paraphrases.

What survived is the pooling instinct: several independent looks at a query
beat one. That instinct is what made Gemma 4's ten-prompt ensemble the most
robust AG configuration (§6.1). We return to debate, properly measured, in §7.6.

---

## 5 Development Methodology

Six practices were applied throughout. We list them because the paper's
negative results are only as credible as these checks.

1. **Scorer parity.** Our Task 1 scorer reproduced the official 2025 evaluation
   script exactly to 3 dp, with zero per-query disagreements, on METEOR,
   BERTScore, NDCG-METEOR and NDCG-BERTScore. Our Task 2 re-implementation
   reproduced the official MC scorer query by query. Matching it required
   rounding each reciprocal rank to 3 dp, as the official script does; before
   that fix MRR disagreed on 19 of 400 queries.
2. **Oracle and null controls.**
   - Ground truth scored against itself gives METEOR 0.924, not 1.0, because
     the fragmentation penalty costs 0.5 on a one-word answer.
   - A random-permutation MC run scored 0.253 / 0.517, against a theoretical
     0.250 / 0.521.
   - A synthetic uniformly-random judge was used to calibrate the rerank gate
     (§6.6).
   - A *mechanical padding* arm controlled for answer expansion (§6.3).
3. **Paired tests.** We used Wilcoxon and sign tests for Task 1 and exact
   McNemar tests plus bootstrapped MRR intervals for Task 2. Unpaired
   comparisons are misleading at these sample sizes: two *random* MC runs
   differed by 3.0 Top-1 points (p = 0.36). At n = 400, about 4.5 points is
   resolvable at 2 SE.
4. **Pre-registered gates.** Each gate's decision rule was written before any
   result was read. For example, Muse Glimmer would be promoted only if it tied
   Qwen3.8 or improved a fusion; the single judge only if it beat the baseline
   at McNemar p < 0.05 *and* gained at least 8 queries.
5. **Nested cross-validation** for any search over configurations (§7.4).
6. **Guards against silent failures.**
   - 2025 and 2026 Q_IDs overlap numerically but never share a question:
     0 of the 150 sampled IDs match. Any script defaulting to the other year's
     paraphrase or routing file would complete, validate and mean nothing. The
     orchestrator statically checked that every path argument was explicit.
   - It also fingerprinted every 2026 artifact after each stage.
   - Every answering pass reports how many queries it actually answered with
     their video attached.

**One gap we could not close.** The official STS metric calls a 2013 web
service that no longer responds. Our dev evaluation therefore measured METEOR
and BERTScore only. §6.7 shows that this gap mattered.

---

## 6 Task 1: Answer Generation

### 6.1 Three answering architectures

| Pass | Model | Input | Calls per query | Answers per call | Routing |
|---|---|---|---|---|---|
| A | Gemma 4 12B | 8–32 sampled frames + Whisper transcript | 10 (question + 9 paraphrases) | 3 | no |
| B | Qwen3-VL-8B-Thinking | native video, frame/token budget planned per clip | 1 | 10 | yes |
| C | Cosmos-Reason2-2B | native video | 1 | 10 | no |

**Pass A** pools 30 candidates per query. Pooling normalises with the
validator's own rule, then takes a position-weighted vote (1.0 / 0.5 / 0.25)
scaled by 0.5 + 0.5 × confidence, with refusals sorted last. Temperature was set
to 0.6 rather than 1.0, because at 1.0 every query produced up to 30 distinct
candidates and the vote had nothing to concentrate on. A smoke test caught a
counting bug: "none" had been treated as a refusal, demoting the correct answer
to "how many people are speaking?" from rank 1 to rank 8.

**Pass B** maps the router's output directly onto the request:

- attach the video or not, and the transcript or not;
- a denser frame budget for temporal questions;
- a larger per-frame budget and a read-the-screen rule for OCR questions;
- a reasoning budget of 4k, 8k or 12k tokens, depending on whether the
  question is flagged for inference, temporal ordering, both or neither.

A degrade ladder walks five rungs (`shrink1`/`shrink2` → `no_dsf` →
`no_mm_kwargs` → `no_video`), so a rejected request is retried in a simpler form
rather than failing. A rule guarantees no query is ever sent with zero evidence.

**Error profile on the full 2026 runs** — counted over 24,430 slots per run:

| Run | Queries needing fallback | Dead (fallback) slots | Main cause |
|---|---|---|---|
| Gemma 4 (A) | 0.8% | **24** | — |
| Qwen3-VL (B) | 8.6% | 1,596 | 62 truncations at the token limit, 60 of which lost all ten slots; 82 queries where all ten answers were "unknown" |
| Cosmos (C) | 74% | 11,289 | no parseable JSON on 18.7%; **45% of queries answered with no video** |

**Fallback rate tracks the architecture, not the model.** Ten short calls fail
independently. One ten-answer call fails as a whole when it truncates or emits
malformed JSON.

The Cosmos blindness came from a bug: `Video_ID` is not a filename. 868 queries
name a bare YouTube ID, stored on disk as `<id>.mp4`, and the Cosmos client
never opened them. The run validated cleanly. The bug was found only while
writing Pass B.

A fourth model, Muse Glimmer 30B, was fully implemented with routing-driven
reasoning strength. It was abandoned before running: setup and the projected
single-GPU runtime did not fit the deadline.

### 6.2 Development results (150 queries, 2025 ground truth)

| Run | METEOR | BERTScore | NDCG-METEOR | Answer words | Queries needing fallback | Unparsed |
|---|---|---|---|---|---|---|
| **Gemma 4** | **0.148** | **0.892** | **0.751** | 4.9 | 0% | 0% |
| Qwen3-VL | 0.053 | 0.873 | 0.597 | 3.3 | 8% | 2% |
| Cosmos | 0.033 | 0.863 | 0.524 | 3.6 | 81% | 20% |
| *oracle* | *0.924* | *1.000* | *1.000* | *8.5 (ground truth)* | — | — |

Every pairwise difference is significant (p ≈ 0). Three caveats apply.

- **Cosmos's score is mostly filler.** On average 5.5 of its 10 slots are
  fallback text, so this row measures output failure, not answer quality.
- **Length confounds METEOR.** All runs answer at roughly half the ground
  truth's length, and length orders the two non-degraded runs exactly as
  METEOR does.
- **BERTScore barely discriminates.** Its whole spread across three very
  different runs is 0.029, and NDCG-BERTScore is saturated at 0.993–0.994.

### 6.3 Answer expansion, and what METEOR rewards

The answers were too short, so we built an LLM expander. It uses Gemma 4 with
thinking off and no video. It rewrites each answer to a target length for its
question type, measured from 2025 ground truth: action and reason 10 words,
manner 11, entity and place 6, and so on. It never expands counting answers,
whose ground truth averages 1.0 words. It rejects any rewrite that drops a
content word of the original.

The plan required a **mechanical control (C)**: the same answers padded to the
same length with contentless filler such as "in the video … as it is shown".

| Run | A: baseline | B: LLM expansion | C: mechanical padding | C's share of B's METEOR gain |
|---|---|---|---|---|
| Gemma 4 METEOR | 0.148 | **0.187** | 0.172 | **62%** |
| Qwen3-VL METEOR | 0.053 | **0.134** | 0.100 | **58%** |
| Cosmos METEOR | 0.033 | **0.076** | 0.057 | **56%** |
| Gemma 4 BERTScore (B−A, C−A) | — | −0.0003 (ns) | −0.0089 (p ≈ 0) | — |

Expansion passed its gate. METEOR rose everywhere (p ≈ 0) and neither
BERTScore nor NDCG fell, and B beat C on both metrics. But the larger finding
concerns the metric: **a transformation that adds zero information captured
56–62% of the METEOR gain.** On this task METEOR is substantially a
length-matching measure.

Two further caveats:

- **86% of rewrites added at least one content word** not present in the
  original. The expander cannot see the video, so these are guesses: "a flip"
  became "a *single* flip".
- **Padding can break validity.** It collapsed distinct short answers into
  duplicates ("a flip" and "the flip"), which would have failed the validator.

Expanded Qwen3-VL (0.134) was statistically indistinguishable from unexpanded
Gemma 4 (0.148, p = 1.0). So part of Qwen3-VL's apparent deficit was style
rather than vision.

### 6.4 Ordering, confidence and the noise floor

Under the 2025 scorer, mean METEOR was identical to 3 dp under current, random,
reverse and oracle orderings. We used this as a conservation check: re-ranking
had lost or altered nothing. Only NDCG responds to order, and there the headroom
was large: oracle minus current is +0.18 to +0.24 NDCG-METEOR (p < 1e-18). No
available signal reached it:

| Signal | Within-query Spearman with per-answer METEOR |
|---|---|
| emitted rank | 0.106 (Gemma) / 0.134 (Qwen3-VL) / 0.087 (Cosmos) |
| pooled vote score | 0.097 |
| vote count | −0.015 |

Re-ranking Gemma 4 by its own score made NDCG significantly *worse*
(−0.0024; 0 queries improved, 14 hurt). Its 30-candidate vote ordered answers
only +0.016 NDCG better than a random shuffle (p = 0.058). A single Qwen3-VL
call asserting its own order did +0.064.

A calibration fact that is easy to get backwards: a synthetic *random* judge
scored a pooled Spearman of **0.0295** with per-answer METEOR. So a pooled
correlation near 0.03 is the noise floor, not a weak signal. Ranking signals
must be judged by arm comparisons and per-query sign tests.

### 6.5 The two-model ensemble (gate skipped)

With two days left, we built a two-model ensemble that gives both models both
halves of the machinery: routing *and* the ten-prompt, three-answer ensemble.

- **Qwen3-VL-8B-Thinking was swapped for the Instruct checkpoint.** Measured on
  the 2026 run, latency = 3.63 s + 40.18 s per 1,000 completion tokens
  (R² = 0.876). The Thinking model's median 1,423 completion tokens per call
  would have made ten prompts cost about 109 h; the Instruct model costs about
  7.3 h.
- **Token log-probabilities were captured.** Because outputs are
  grammar-constrained JSON, scoring the whole sequence would mostly measure the
  forced structural tokens. Instead, log-probabilities are attributed to each
  answer's own character span.
- **Fusion.** 60 candidates per query are pooled and ranked by a weighted sum
  of cross-model agreement, vote score, mean log-probability and confidence.

The fused run cut dead slots to 5.1%, from 34.1% for the Qwen3-VL ensemble
alone. But Gemma 4 alone was already at about 0.1%. The *selection* half (does
the weighted sum pick the right ten?) shipped **without the 2025 gate**, by
explicit decision under deadline. The post-hoc dev measurement, run after
submission, was unfavourable:

| Run (150 dev queries) | METEOR | NDCG-METEOR |
|---|---|---|
| **Gemma 4, single model** | **0.148** | **0.751** |
| Gemma 4, ensemble prompts | 0.145 | 0.739 |
| fused (2 models, 60 candidates) | 0.133 | 0.713 |

Fused trails single Gemma by 0.015 METEOR (paired p = 0.041). Exact
cross-model agreement turned out to be nearly absent: a mean of 0.6 matching
candidates per query, and none at all on 1,677 queries. So the most heavily
weighted fusion term was also the sparsest.

### 6.6 VLM-as-judge rerank

Following ITI-CERTH's 2025 approach [ITI-CERTH 2025], Gemma 4 re-watched each
query's frames and scored every pooled candidate from 1 to 10 in a single
listwise call. Three things differ from their design:

- **One call per query** instead of one per candidate: 2,443 calls instead of
  62,601.
- **No majority vote.** In our data, vote counts correlate *negatively* with
  correctness (§6.4).
- **Blind and shuffled candidates**, with no model attribution.

The gate design had three arms built from the identical pool: the fuser, the
judge and *random selection*. We calibrated it with a synthetic random judge,
which the gate correctly refused (−0.0148 against the fuser; 37 wins / 88
losses; p < 0.0001). The calibration also showed that the existing fuser beats
random *selection* by +0.020 METEOR (p < 0.0001), so any judge has to clear a
real baseline. We have no recorded dev result for the real judge arm. The
judged run was submitted on its design.

### 6.7 Submitted runs and official results

| Run tag | Description |
|---|---|
| `gemma4_run01` | Pass A, Gemma 4 via Ollama, 10 prompts × 3, unexpanded |
| `gemma4_expand_run1` | `gemma4_run01` + LLM expansion |
| `qwen3vl_expand_run2` | Pass B, Qwen3-VL-8B-Thinking, routed, + LLM expansion |
| `fused_expan_run1` | two-model ensemble (§6.5) + LLM expansion |
| `fused_judge_exp_run1` | ensemble re-ranked by the judge (§6.6) + LLM expansion |

**Official results, human-answer-only references (44 runs).** Rank among all
runs in parentheses.

| Run | STS@1 | METEOR@1 | BERTScore@1 | STS NDCG@10 | METEOR NDCG@10 | BERTScore NDCG@10 |
|---|---|---|---|---|---|---|
| `gemma4_run01` | **0.457** (23) | 0.176 (39) | **0.8887** (18) | 0.890 (35) | 0.704 (44) | 0.9924 (35) |
| `gemma4_expand_run1` | 0.423 (30) | **0.214** (28) | 0.8860 (23) | 0.888 (38) | 0.769 (36) | 0.9942 (28) |
| `fused_judge_exp_run1` | 0.407 (35) | 0.213 (31) | 0.8829 (31) | **0.891** (34) | 0.778 (32) | 0.9934 (30) |
| `fused_expan_run1` | 0.402 (38) | 0.212 (32) | 0.8827 (32) | 0.888 (37) | **0.779** (31) | 0.9934 (31) |
| `qwen3vl_expand_run2` | 0.362 (44) | 0.168 (43) | 0.8802 (36) | 0.889 (36) | 0.743 (43) | **0.9950** (27) |
| *field median* | *0.458* | *0.234* | *0.8860* | *0.926* | *0.810* | *0.9956* |
| *field best* | *0.511* | *0.298* | *0.8988* | *0.965* | *0.881* | *0.9992* |

With human + alternative references, every score rises by about 0.02–0.03 and
our runs keep the same relative order.

### 6.8 What the official AG results say

**The model ordering held.** Gemma 4 beat Qwen3-VL on every @1 metric,
as the dev set predicted.

**Expansion traded STS for METEOR.** On the official metrics, expansion moved
`gemma4_run01` to `gemma4_expand_run1` as follows:

| Metric | Change | Dev prediction |
|---|---|---|
| METEOR@1 | **+0.038** | +0.039 |
| METEOR NDCG@10 | +0.065 | — |
| STS@1 | **−0.035** | — (STS not computable on dev) |
| BERTScore@1 | −0.0027 | −0.0003 (ns) |

So expansion did exactly what dev said on METEOR, while losing on STS, the one
official metric our development could not compute. Our unexpanded Gemma run is
our best on STS and BERTScore, and our worst-ranked on METEOR NDCG. Optimising
toward the metric you can measure moved the one you cannot.

**Fusion did not help.** `fused_expan_run1` trails `gemma4_expand_run1` on all
three @1 metrics: METEOR −0.0025, STS −0.021, BERTScore −0.003. Its one gain is
METEOR NDCG@10 (+0.010). This is the outcome the post-hoc dev gate predicted in
§6.5. The judge rerank moved the fused run by +0.0017 METEOR@1 and +0.0048
STS@1, which cannot be distinguished from zero without per-query scores.

**The candidate sets were better than the rank-1 choices.** The official
detail tables report Best@10, the best score among a run's ten answers:

| Run | STS Best@1 → Best@10 (rank) | BERTScore Best@1 → Best@10 (rank) | METEOR Best@1 → Best@10 (rank) |
|---|---|---|---|
| `gemma4_run01` | 0.457 (23) → 0.639 (**9**) | 0.889 (18) → 0.917 (**9**) | 0.176 (39) → 0.335 (23) |
| `gemma4_expand_run1` | 0.423 (30) → 0.600 (14) | 0.886 (23) → 0.907 (23) | 0.214 (28) → 0.368 (**13**) |

Our ten-answer sets place in the top quarter of the field on Best@10, while our
rank-1 answers place in the middle or below. Part of this is a property of
Best@10, which rewards answer diversity, and our paraphrase ensemble was built
to produce diverse sets. But the gap is also a direct cost of developing
against the 2025 scorer. Its mean over ten answers made ordering irrelevant,
so we invested in *filling ten good slots* and never in *choosing the best
first*. Under the 2026 @1 metrics, rank-1 selection is the dominant lever, and
§6.4 shows we had no reliable signal for it. Gemma 4's NDCG ranking was last of
44 runs on METEOR NDCG@10, consistent with its dev ranking being barely better
than random.

---

## 7 Task 2: Multiple Choice

### 7.1 The ranking contract

All eight MC passes share one contract, so dev differences reflect vision and
reasoning, not output hygiene.

- **Rank option numbers, never option text.** The model returns a permutation
  of 1..N, which is mapped back to the official strings by index. Every option
  is therefore ranked exactly once and byte-identical to the official text.
  `as_permutation()` repairs repeats and drops. `check_permutation()` refuses to
  write any file the validator would reject.
- **Permute the presentation.** Each query gets three passes: official order,
  exact reverse, and a seeded shuffle. They are combined by a
  confidence-weighted Borda count. Measured position bias was at chance in
  every run: rank 1 fell on the officially-first option 14.9–17.4% of the time,
  against 14.9% expected.
- **Routing reused unchanged.** The Task 1 routing file matches every Task 2
  question once trailing whitespace is stripped.

### 7.2 Eight rankers on the dev set (400 queries, four options)

| Model | Top-1 | MRR | Paired vs Qwen3.8 | 2026 run (wall clock) |
|---|---|---|---|---|
| **Qwen3.8-27B INT4** | **0.800** | **0.884** | — | 7.53 h |
| Qwen3-VL-8B Instruct | 0.790 | 0.872 | −1.0, p = 0.70 | 18.88 h |
| InternVL3.5-30B-A3B Q4 | 0.760 | 0.859 | −4.0, p = 0.064 | 1.99 h |
| Muse Glimmer 30B Q4 | 0.755 | 0.856 | −4.5, p = 0.018 | not run (gate) |
| VideoLLaMA3-7B | 0.738 | 0.843 | −6.2, p = 0.004 | 6.12 h |
| Gemma 4 12B | 0.733 | 0.840 | −6.8, p = 0.0005 | 3.01 h |
| LLaVA-Video-7B | 0.725 | 0.835 | −7.5, p = 0.0008 | 1.85 h |
| Cosmos-Reason2-2B | 0.650 | 0.783 | −15.0, p ≈ 0 | 0.34 h |
| random control | 0.253 | 0.517 | −54.8 | — |

Two patterns stand out:

- **Task rank does not transfer.** Gemma 4 won Task 1 outright and is the
  second-worst real ranker here.
- **Size barely predicts rank.** Qwen3-VL-8B ties a 27B model, and two 7B
  models beat a 12B model.

Every large gap traced back to an input/output or sampling defect (§7.3), not
to vision quality.

### 7.3 Model-specific traps, each silent

| Model | Trap | Effect if missed | Fix |
|---|---|---|---|
| Qwen3-VL, Qwen3.8 | `size.longest_edge` is a **whole-video** pixel budget, not per-frame | the Task 1 Qwen3-VL run saw thumbnail-resolution video: 485 visual tokens against a planned 17,792 | set the budget directly; fps only spreads it over time |
| Qwen3-VL, Qwen3.8, Cosmos | `do_sample_frames` is rejected on short clips, and the 400 error is swallowed by a text-only fallback | Cosmos answered 23% of queries blind | ladder: sampled video → unsampled → text; count `video_used` |
| Qwen3.8 | no kwarg controls temporal sampling (rejected on 72% of queries) | — | a transcoded re-run (`qwen38.resampled`) changed rank 1 on 6.3% of targeted queries vs 8.0% of controls: **no effect** |
| Cosmos | a reasoning model *chooses* rather than ranks (`{"ranking": [4]}`) | partial permutations | second schema-constrained call orders the remainder (37.7% of queries) |
| InternVL3.5 | llama.cpp tiles each frame into 448-px crops; 16 frames = 20,945 tokens | every request returns HTTP 400 | resize client-side to one tile per frame (4,561 tokens) |
| VideoLLaMA3 | the sdpa vision encoder materialises a full S×S mask, ~8 GB per layer | Windows spills VRAM to system RAM: 100% GPU, zero progress, no error | exact block-diagonal attention patch (max abs diff 0.0) |
| Muse Glimmer | chat template defaults to *high* reasoning; 4,096 tokens per image natively | silently doubled cost or context overflow | always send `reasoning_strength=low`; cap image tokens |
| LLaVA-Video | OpenCV over-reports `.webm` frame counts by 1 on 256 files | last sampled frame unreachable | tolerated (vLLM warns); documented |

### 7.4 Fusion, measured honestly

An early accuracy-weighted reciprocal rank fusion (RRF) [Cormack et al. 2009]
of four models, at 8:8:4:1, tied the baseline (0.795, p = 0.86). Its weights had
been fitted on the same 400 queries. Equal-weight fusion of three models *lost*
3.5 points (p = 0.029).

We then searched every subset and every integer weighting from 0 to 4 of all
eight runs, under RRF and Borda: 781,248 fusions. Each was judged by nested
cross-validation: pick the best fusion on nine folds, score it on the held-out
tenth, over 20 random splits.

| Search | Best in-sample | Nested CV | CV repeats above baseline |
|---|---|---|---|
| weight grid (781,248 fusions) | 0.825 / 0.896 | **0.801 / 0.883** | 9 of 20, median p = 0.87 |
| equal-weight subsets (510) | 0.810 / 0.890 | 0.778 / 0.871 | 0 of 20 |

**The entire in-sample gain was search optimism.** The reason shows in the
oracle: at least one run ranks the right answer first on 93.5% of queries, but
every model loses more queries to Qwen3.8 than it wins.

| vs Qwen3.8 | Qwen3-VL | InternVL | VideoLLaMA3 | LLaVA-Video | Muse | Cosmos |
|---|---|---|---|---|---|---|
| right where Qwen3.8 is wrong | 29 | 25 | 23 | 23 | 17 | 26 |
| wrong where Qwen3.8 is right | 33 | 41 | 48 | 53 | 35 | 86 |

There is no majority to harvest. Voting mixes in more error than it removes.

### 7.5 Where the best model fails

Qwen3.8 misses 80 of the 400 dev queries. We tested about 28 features with
exact Fisher tests at a Bonferroni threshold of p < 0.0018.

- **Counting is the only question type that stands out:** 43.8% error, against
  16.8% elsewhere (p = 0.0001). **The misses are systematic undercounts.** Of 18
  readable cases, 15 were too low, with a mean error of −1.2. Every model
  undercounts, by a mean of −0.4 to −1.3.
- **Split passes predict error:** 46.4% error when the three presentation
  passes disagree, against 14.5% when they agree. Every other model errs 48–59%
  on the same queries, so this is a good confidence signal and a useless
  routing signal: there is no model to route them to.
- **An answer-length artefact:** error is 11.8% when the correct option is the
  longest one, against 27.1% otherwise. This reflects how the 2025 options were
  written, not the videos.
- **18 queries are missed only by Qwen3.8.** Their common theme is "what can be
  heard" questions asked about video whose audio the ranker never received.

### 7.6 Arbitration: a judge, two judges, and a debate

**Single judge.** On the 203 dev queries where the panel disagreed on rank 1,
a judge re-watched 16 frames. It saw every panellist's ranking and a
reliability card describing each panellist's known weaknesses.

| Arm | Correct of 203 | Full 400 Top-1 | Paired vs Qwen3.8 |
|---|---|---|---|
| **Qwen3.8 alone** | **133** | **0.800** | — |
| Qwen3.8 as judge | 132 | 0.797 | p = 1.00 |
| InternVL3.5 as judge | 130 | 0.792 | p = 0.78 |
| InternVL3.5 as judge, no reliability card | 127 | 0.785 | p = 0.50 |
| majority vote (control) | 129 | 0.790 | p = 0.58 |
| null judge, random candidate (control) | 73 | 0.650 | p ≈ 0 |
| *perfect adjudicator (ceiling)* | *187* | *0.935* | — |

The judges were not timid: they overruled the baseline 39 and 61 times. **But
the overrules were a coin flip:** 16 became right and 17 became wrong for
Qwen3.8; 24 and 27 for InternVL. The gate failed. Both controls landing where
expected is what makes the negative result trustworthy: the harness is not
handing out wins, and the panel has no useful majority.

**Two judges read against each other** told a sharper story:

| On the 203 disputed queries | n | Correct |
|---|---|---|
| the two judges agree | 153 | 112 (baseline 109) |
| the two judges disagree | 50 | judge 1: 20; judge 2: 18; **baseline 24**; either judge right: 38 |
| oracle over both judges | 203 | 150 (baseline 133) |
| two-judge oracle, counting questions only | 38 | 24 |

From this we designed a two-agent debate [cf. Du et al. 2023] over disputed
queries:

- **Opening opinions and two rounds**, sharing the full transcript and citing
  frame timestamps. A converged answer wins. A deadlock falls back to the
  baseline, never to a judge, because on contested queries the baseline beat
  both judges.
- **Evidence the panel never had:** 32 freshly cut frames per video (64 for
  counting questions), an explicit instruction to report the maximum count,
  and CLAP audio scores for 135 sound questions.
- **Safe by construction.** The emitted run equals the baseline everywhere
  except where the panel disputes *and* the debate converges on a different
  option. We verified this property against never-conceding mock agents: the
  output matched the baseline exactly.

The system was verified end to end against mock models and against the real
InternVL judge. It never ran at scale: the Mac-hosted 27B judge could not
return one answer in 30 minutes. **Debate is therefore unmeasured** in this
work. The two-judge oracle suggests at most about +17 queries of headroom on
dev, and counting, the largest error class, needs evidence rather than
argument.

### 7.7 Submitted runs, official results and dev-to-test transfer

| Run | Description | Dev Top-1 | **Official Top-1** (rank / 42) | **Official MRR** (rank / 42) |
|---|---|---|---|---|
| `qwen38` | Qwen3.8-27B INT4, 3 passes, routed | 0.800 | **0.8027** (20) | **0.8767** (15) |
| `qwen38.resampled` | same, transcoded temporal sampling | — | 0.8019 (21) | 0.8757 (17) |
| `fused.weighted` | accuracy-weighted RRF, 4 models | 0.795 | 0.7846 (27) | 0.8659 (22) |
| `qwen3vl.run` | Qwen3-VL-8B Instruct, 4k visual tokens | 0.790 | 0.7537 (32) | 0.8433 (32) |
| `internvl.run` | InternVL3.5-30B-A3B Q4 | 0.760 | 0.7401 (33) | 0.8355 (33) |
| *field median / best* | | | *0.8015 / 0.8476* | *0.8665 / 0.9081* |

Transfer from dev to the official results:

- **The best model transferred almost exactly:** 0.800 on dev, 0.803 official.
- **The dev "tie" was not a tie.** Qwen3-VL was statistically tied with Qwen3.8
  on dev (−1.0 points, p = 0.70) but finished 4.9 points behind on 2026
  (1,830 against 1,949 correct of 2,428). The dev set is four-option, short
  video and unrouted. The test set is mostly eight-option, has a long tail, and
  was routed. The gap between the two models opened on the axes the dev set
  could not test.
- **Fusion fell below its best member,** as nested CV had predicted (0.785
  against 0.803).
- **The temporal-resampling run matched the original** (0.8019 against
  0.8027), confirming the dev-time finding that the workaround changed nothing.

---

## 8 Throughput

We measured prefill and decode rates for all eight VLMs on ten real 2026
queries spanning 1.1 s to 389 s of video. Each was single-stream, with prefix
caching defeated by a per-request nonce (verified at 0.0–0.6% hit rate).
Task 2 results (TTFT = time to first token):

| Model | Prefill tok/s | Decode tok/s | Median prompt tokens | Median generated tokens | TTFT (s) |
|---|---|---|---|---|---|
| Cosmos-Reason2-2B | 4,737 | 68.6 | 10,875 | 37 | 2.62 |
| LLaVA-Video-7B | 3,264 | 48.0 | 6,784 | 36 | 2.12 |
| VideoLLaMA3-7B | 2,613 | 28.9 | 4,570 | 36 | 1.80 |
| Qwen3-VL-8B | 2,207 | 40.4 | 4,708 | 36 | 2.25 |
| Gemma 4 12B INT4 | 1,930 | 26.3 | 4,589 | 41 | 2.38 |
| InternVL3.5-30B-A3B | 1,764 | 139.3 | 4,508 | 44 | 2.57 |
| Qwen3.8-27B INT4 | 999 | 13.8 | 4,761 | 256 | 4.91 |
| Muse Glimmer 30B | 791 | 45.3 | 4,442 | 411 | 5.66 |

**MC ranking is prefill-bound.** Six models answer in 36–44 tokens.

Three cautions apply to the table:

- **Not every row is a production configuration.** Qwen3.8's row reflects a
  fallback configuration (Triton attention, bf16 KV) forced by a FlashInfer
  compile failure, and its 256 generated tokens reflect the bench sending no
  output schema. Neither matches its production run.
- **KV-cache dtype was not held constant** across models.
- **Two timing-client bugs would have inflated TTFT about five-fold.** Python
  resolves `localhost` to IPv6 first, and every server bound IPv4 only, adding
  about 2 s per connection. `requests` also buffers SSE lines in 512-byte
  chunks. Both were found against a mock server before any GPU time was spent.

---

## 9 Engineering Lessons: A Catalogue of Silent Failures

Every entry below produced output that looked valid. Most were caught only
because a pass reports its own evidence counts: video attached, fallback slots,
prompt tokens, position bias.

| # | Failure | How it presented | How it was caught |
|---|---|---|---|
| 1 | `Video_ID` treated as a filename | 45% of a run answered blind, validated clean | counting `video_exists` while writing the next pass |
| 2 | Frame budget disabled after a preflight probed only the shortest clip | median prompt 12,014 tokens instead of ~500 | inspecting prompt tokens of the first 4 records |
| 3 | Whole-video pixel budget read as per-frame | thumbnail video for a 54-hour run | token-count sweep during Task 2 |
| 4 | 2025/2026 Q_ID collision through default paths | a run that completes, validates and means nothing | static path-flag guard in the orchestrator |
| 5 | Ollama's model host (`llama-server.exe`) orphaned on teardown | port closed, 8.9 GB still held; next server never starts | `nvidia-smi` compute-apps check after teardown |
| 6 | vLLM killed by the launcher's captured stdio pipes | `rc=0`, empty log, nothing listening | A/B: same command with stdio at `DEVNULL` survives |
| 7 | WSL2 tears the distro down when the last session exits | server dies ~20 s after launch | holding a detached `sleep` session |
| 8 | Thinking models spend the whole token budget reasoning | empty `content`, which looks like a blind model | `done_reason: length` surfaced as a named error |
| 9 | Truncation at HTTP 200 never retried | 600 dead slots | `finish_reason` audit of the cache |
| 10 | A dead server: every query falls back, run exits 0 | a file identical to the baseline | abort after 10 consecutive fallbacks |
| 11 | Windows spills VRAM to system RAM instead of raising OOM | 100% GPU, no progress, no error | "busy but no output" treated as a memory symptom |
| 12 | A PowerShell 5.1 `Tee-Object` log written as UTF-16 | live-log watchers saw nothing | watching the UTF-8 JSONL cache instead |

The general lesson: **an unattended run is a different environment from an
interactive one.** Three multi-hour restarts of the dev orchestrator came from
process ownership, session survival and stdio handling. A 30-minute
end-to-end smoke rehearsal, which was skipped once, exists to surface exactly
this class of failure.

---

## 10 Discussion

**Measure what ships, with the metric that scores it.** Our two largest
surprises both came from gaps between the development measurement and the
official one. Expansion was validated on METEOR and BERTScore and then lost
0.035 STS@1, a metric we could not compute. Our AG pipeline was optimised for a
mean-over-ten scorer while 2026 scores rank 1. Where the development metric did
match the official one, as in Task 2 Top-1, the estimate for our best model
was within 0.3 points.

**Combination does not rescue a weak panel.** We tried six ways of combining
models across the two tasks:

- two-model AG fusion;
- equal-weight RRF;
- accuracy-weighted RRF;
- a 781,248-fusion search;
- a single VLM judge;
- a VLM judge over AG candidates.

Each lost to or tied the best single model wherever it was measured, and the
official results agree. The common cause is visible in the exchange table in
§7.4: when every member is weaker than the leader on the queries where they
disagree, disagreement signals a hard query, not a recoverable one. The judge
results sharpen this: the panel's 93.5% oracle is unreachable from the same 16
frames.

**Robustness beat capability.** The only dedicated reasoning model in our set,
Cosmos-Reason2, finished last on both tasks, for reasons of output reliability
rather than vision. Gemma 4's ten-short-calls design won Task 1 largely by never
failing.

**Self-reported confidence was worthless in every place we measured it.** The
router returned 1.0 on 96.9% of judgements. AG vote counts correlated
negatively with correctness. LLaVA-Video returned 0.0 on every reply. The one
useful uncertainty signal was *agreement across independent passes*, and it is
useful only for confidence, not for routing.

**Dev-set mismatch is the residual risk.** Our dev sets were short-video,
four-option and speech-heavy (Task 1) or speech-light (Task 2). The Qwen3-VL
tie that dissolved on 2026 is the clearest case.

**Gates saved compute.** Muse Glimmer's failed gate saved 16–20 GPU-hours.
Nested cross-validation refused a +2.5-point in-sample winner. A null-judge test
calibrated the rerank gate before it was trusted.

---

## 11 Conclusion and Future Work

On one consumer GPU with open-weight models, our best MC run reached 0.803
Top-1 and 0.877 MRR, near the field median, and closely matched its development
estimate. Our AG runs had competitive candidate sets but weaker rank-1
selection. The most transferable outcomes are methodological: a protocol that
turns plausible ideas into measured negatives, and the documentation of what
those ideas cost and why they failed.

Concrete next steps, each grounded in a measured weakness:

1. **Counting.** Every model undercounts, systematically. Denser frames and an
   explicit report-the-maximum instruction are built but unmeasured.
2. **Audio evidence.** "What can be heard" questions on video whose audio the
   ranker never received are a distinct error class. CLAP option scoring is
   built and checked for discrimination (median best-to-worst spread 6.6), but
   it has not been scored against labels.
3. **Rank-1 selection for AG.** Develop against the @1 metric, and use
   generation-time log-probabilities or a trained reranker, which are the only
   signals that plausibly reach the NDCG headroom in §6.4.
4. **Measure debate on dev.** Run the debate pipeline over the 203 disputed dev
   queries under the same gate as the single judge.
5. **Local STS.** Restore an STS-like metric to the development loop, so that
   length-oriented changes are caught before submission.

---

## Acknowledgements

We thank NIST and the TREC VQA track organisers for the data, the validators
and the evaluation.

---

## References

- Banerjee, S. and Lavie, A. (2005). METEOR: An Automatic Metric for MT
  Evaluation with Improved Correlation with Human Judgments. *ACL Workshop on
  Intrinsic and Extrinsic Evaluation Measures for MT and/or Summarization.*
- Cormack, G. V., Clarke, C. L. A. and Büttcher, S. (2009). Reciprocal Rank
  Fusion Outperforms Condorcet and Individual Rank Learning Methods. *SIGIR.*
- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B. and Mordatch, I. (2023).
  Improving Factuality and Reasoning in Language Models through Multiagent
  Debate. arXiv:2305.14325.
- FunAudioLLM Team (2024). FunAudioLLM: Voice Understanding and Generation
  Foundation Models for Natural Interaction Between Humans and LLMs.
  arXiv:2407.04051.
- Han, L., Kashyap, A., Finin, T., Mayfield, J. and Weese, J. (2013).
  UMBC_EBIQUITY-CORE: Semantic Textual Similarity Systems. *\*SEM.*
- Kwon, W. et al. (2023). Efficient Memory Management for Large Language Model
  Serving with PagedAttention. *SOSP.*
- McNemar, Q. (1947). Note on the Sampling Error of the Difference Between
  Correlated Proportions or Percentages. *Psychometrika* 12(2).
- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C. and Sutskever, I.
  (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *ICML.*
- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q. and Artzi, Y. (2020).
  BERTScore: Evaluating Text Generation with BERT. *ICLR.*
- [ITI-CERTH 2025] ITI-CERTH, TREC 2025 VQA notebook paper. *[citation to be
  completed]*
- [HLTCOE 2025] HLTCOE, TREC 2025 VQA notebook paper. *[citation to be
  completed]*
- TREC 2026 VQA track guidelines. https://www-nlpir.nist.gov/projects/tv2026/vqa.html
- Model cards (Hugging Face): `google/gemma-4-12B-it-qat-w4a16-ct`,
  `Qwen/Qwen3-VL-8B-Thinking`, `Qwen/Qwen3-VL-8B-Instruct`,
  `nvidia/Cosmos-Reason2-2B`, `RedHatAI/Qwen3.8-27B-INT4`,
  `OpenGVLab/InternVL3_5-30B-A3B`, `unsloth/Muse-Glimmer-30B-GGUF`,
  `Isotr0py/LLaVA-Video-7B-Qwen2-hf`, `DAMO-NLP-SG/VideoLLaMA3-7B`,
  `Systran/faster-whisper-large-v3`, `FunAudioLLM/SenseVoiceSmall`.

---

## Appendix A — Source documents

This paper consolidates the project's working notes. Every number above was
read from one of them or from the official NIST score files.

| Section | Source documents |
|---|---|
| §2 Task and data | `Welcome to the TREC2026 VQA Track_JULY2026.md`, `VQA/README.md`, `SESSION.2026-08-11.task1-pipeline.md`, `SESSION.2026-08-13.video-durations.md` |
| §3 Serving | `VQA/GEMMA4_VLLM_SETUP.md`, `VQA/QWEN3_VL_SETUP.md`, `VQA/COSMOS_REASON2_SETUP.md`, `ARCHITECTURE.reconciled.md` |
| §4.1 ASR | `SESSION.2026-08-11.audio-asr-sensevoice.md`, `SESSION.2026-08-12.audio-asr-whisper.md`, `TRANSCRIPTION.language-review*.md` (3) |
| §4.2–4.4 Frames, paraphrases, routing | `SESSION.2026-08-12.answer-generation.md`, `SESSION.2026-08-12.query-variations.md`, `SESSION.2026-08-13.query-routing.md` |
| §4.5 Design evolution | `Multi_Agent_VQA_Session_Doc.md`, `ARCHITECTURE.reconciled.md` |
| §5 Methodology | `VQA/EVALUATION_PLAN.md`, `EVAL_2025_ORCHESTRATION_PLAN.md`, `sample.2025.sample150.report.md`, `SESSION.2026-09-15.task2-mc-ranking.md` |
| §6.1 AG passes | `SESSION.2026-08-12.answer-generation.md`, `SESSION.2026-08-14.qwen3-vl.md`, `SESSION.2026-08-14.muse-glimmer.md`, `VQA/COSMOS_REASON2_SETUP.md`, `ARCHITECTURE.reconciled.md` |
| §6.2–6.4 AG dev results | `VQA/EVALUATION.2025.md`, `SESSION.2026-08-15.evaluation-and-expansion.md`, `VQA/ANSWER_EXPANSION_PLAN.md`, `ranking.analysis.2025.md` |
| §6.5–6.6 Ensemble and judge | `ENSEMBLE_PLAN.md`, `RERANK_PLAN.md`, `GATE_2025_RUNBOOK.md`, `MODEL_SCORECARD.2026-09-17.md`, `INVESTIGATION.2026-08-16.qwen3vl-throughput.md` |
| §7 MC | `SESSION.2026-09-15.task2-mc-ranking.md`, `SESSION.2026-09-15.muse-glimmer-task2.md`, `MUSE GLIMMER TASK2_DEV_FULLRUN_PLAN.md`, `SESSION.2026-09-16.task2-extra-rankers.md`, `MODEL_SCORECARD.2026-09-17.md`, `colab/README.md` |
| §7.6 Arbitration | `SESSION.2026-09-17.orchestrator-judge.md`, `SESSION.2026-09-17.debate-system.md`, `# Two-judge debate over the ranker swarm.md`, `SESSION.2026-09-18.debate-mac-local.md` |
| §8 Throughput | `SESSION.2026-09-17.token-throughput-bench.md` |
| §6.7, §7.7 Official results | `FINAL RESULTS/VQA_AG_effectiveness_scores.xlsx`, `FINAL RESULTS/vqa.mcq.scores.txt` |
| *excluded* | `SESSION.2026-09-22.crowd-density.md` (unrelated side project); vendor model cards under `models/` |

## Appendix B — Reproducibility notes

- All stages are resumable from per-item JSONL caches, and re-emit outputs from
  cache with no inference (`--rebuild` / `--assemble-only`).
- Seeds derive from SHA-256 of the question text, or from the Q_ID for MC
  shuffles.
- Every submitted file passed the official validators with
  `--warnings-as-errors`.
- Standalone, dependency-free ports of six MC rankers, with sharded and
  resumable execution, are provided for Colab (`colab/`).

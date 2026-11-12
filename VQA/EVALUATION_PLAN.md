# Plan — Scoring our answers against the 2025 ground truth

**Status: proposed, not implemented.** Written 14 August 2026. Supersedes the
synthetic Phase 1 of `VQA/ANSWER_EXPANSION_PLAN.md`.

## Context

No answer this project has ever produced has been scored. Three session
documents list it as the open question (`SESSION.2026-08-12.answer-generation.md`
§9, `SESSION.2026-08-14.qwen3-vl.md` §12), and every tuning decision so far —
temperature 0.6 for Gemma 4, the reasoning-budget tiers, the routing policy, the
answer-length style — rests on plausibility rather than measurement.

The 2025 task shipped ground truth: `VQA/vqa.gt.csv`, 2,000 rows of
`Q_ID, Video_ID, Question, Answer`. Running our pipeline over those questions
and scoring against those answers gives the project its first real accuracy
number, and it is also the gate for the answer-expansion question.

## Three facts established before planning

**1. None of the 2025 GT videos are on disk.** All 2,000 `Video_ID`s are YouTube
IDs; 0 of 2,000 are present in `videos/`, `videos_TEST_VQA/videos/` or
`vqa.2025.trainingData/`. The 473 mp4s in `videos/` are 2025 *training*
downloads and do not intersect the GT set. **The videos must be downloaded.**

**2. The official STS metric cannot be computed.** `VQA/calc.sts.pl` shells out
to `wget` against `http://swoogle.umbc.edu/SimService/` — a 2013 UMBC research
service. A direct probe times out with no response, and the perl script returns
empty. `eval.vqa.ag.py:105` calls it once per answer, so a full run would make
20,000 requests to a dead host and yield `NaN` for every one.

**3. `VQA/vqa.gt.csv` and `vqa.gt.2025.csv` are the same data.** Both 2,000 rows;
the 2,000-byte size difference is exactly one byte per line — CRLF against LF.
Use `VQA/vqa.gt.csv`, which is what `eval.vqa.ag.py` expects.

## Phase A — Sample 150 of the 2025 videos

All 2,000 is not affordable and probably not possible: NIST's own evaluator
hard-codes `SKIP = {18, 27, 62, ...}` — 18 queries "because their videos were not
available for download" **in 2025** — and a year on, attrition will be worse.

**The sample is 150 videos = 150 questions.** The 2025 GT is exactly one question
per video (2,000 rows, 2,000 distinct `Video_ID`s), so video count and query
count are the same number here — unlike 2026, where 1,956 videos serve 2,443
queries.

The sample must reproduce the 2026 test set on two axes: **video duration** and
**the routing decision**. Neither is known for a video that has not been
downloaded, so selection happens in four stages, cheapest first.

### Target distributions to match (measured from the 2026 set, n=2,443)

| Duration bin | 2026 share | of 150 | | Budget tier | 2026 share | of 150 |
|---|---|---|---|---|---|---|
| 0–5 s | 11.9% | 18 | | brief `(¬inf, ¬temp)` | 29.5% | 44 |
| 5–10 s | 17.1% | 26 | | standard (one of the two) | 58.2% | 87 |
| 10–15 s | 29.4% | 44 | | deep (both) | 12.3% | 18 |
| 15–30 s | 19.5% | 29 | | | | |
| 30–60 s | 4.2% | 6 | | **Route** | | |
| 60–150 s | 10.9% | 16 | | vision | 75.3% | 113 |
| 150 s+ | 6.3% | 10 | | vision+audio | 15.5% | 23 |
| | | | | audio | 6.9% | 10 |
| **Dense sampling** | 36.3% | 54 | | the four OCR routes | 2.2% | 3 |

Question types, for reference: action 30.8%, entity 18.9%, reason 17.1%,
count 12.8%, manner 8.3%, place 4.2%, time 3.2%, summary 2.4%, yes_no 2.0%.

### Stage 1 — Metadata probe, no video bytes

`yt-dlp --skip-download --dump-json` (or `--print duration,id`) over all 2,000
IDs returns duration **and** availability without fetching a single frame.
Roughly 1–2 s each with a polite sleep — about an hour, and it is the cheapest
possible way to discover attrition *before* committing to downloads. Cache the
result as `videos_2025_GT/metadata.probe.jsonl`, resumable.

Output: the surviving candidate pool with a real duration for each. Report the
attrition rate; it is a finding in its own right.

### Stage 2 — Route the candidates

`VQA/route.query.inputs.py` is **text-only** — its own docstring says "no video,
no frames", and its six prompts take only `{question}`. So the 2025 questions can
be routed without any media, producing the same `Route` / `Temporal` /
`Inference` / `Question_Type` labels the 2026 set has.

It makes ~6 Ollama calls per question at `DEFAULT_WORKERS = 1`, so routing all
~1,600 survivors would cost hours. **Route only a duration-stratified candidate
pool of ~400**, drawn to over-cover every duration bin by roughly 2.5×. That is
enough headroom for stage 3 to hit the routing marginals, at ~40 minutes instead
of ~4 hours.

Point it at a copy: `--out query.routing.2025.json --cache
query.routing.2025.cache.jsonl`, never the 2026 files.

### Stage 3 — Select the 150

**Match the marginals, not the joint.** The joint (duration bin × budget tier)
has 24 non-empty cells; at 150 draws, four of them round to below 1 and most
others land at 2–5, so exact joint quotas would be noise-fitting. Instead select
greedily to minimise total variation distance against all three marginal targets
at once (duration bin, budget tier, route), or equivalently rake the weights.

Keep a ranked **reserve list** per stratum, so a video that dies between stage 1
and stage 4 is backfilled from its own stratum rather than distorting the sample.

Record the achieved distribution beside the target; a sample that missed is still
usable if the miss is documented.

### Stage 4 — Download the 150 (plus reserves)

**Reuse `download_test_videos.py`.** It already does this job for YouTube IDs:
writes `<Video_ID>.mp4`, resumable, checkpointed, with permanent-failure
detection that skips the retry loop, plus `--limit`, `--only`, `--retry-failed`
and `--temp-dir`. It reads the 2026 query JSON, so give it a converted file
(`[{"Q_ID", "Video_ID", "Question"}]`) rather than editing a working downloader.

Write to **`videos_2025_GT/`**, never into `videos_TEST_VQA/videos/` — the 2026
submission's media directory must not gain files that are not part of it.

### What n=150 buys, and what it does not

- **The paired A/B (baseline vs expanded) is well powered.** Same queries, both
  conditions: if the per-query METEOR difference has sd ≈ 0.10, the standard
  error on the mean difference is ≈ 0.008, so a shift of ~0.02 resolves
  comfortably. This is the question the sample exists to answer.
- **The absolute accuracy figure is soft.** Per-query METEOR spread is much wider
  than the differences; expect a 95% interval of roughly ±0.03. Quote it as "on a
  150-query 2025 sample", never as "our METEOR".
- **OCR behaviour will not be tested.** The four OCR routes are 2.2% of 2026,
  i.e. 3 of 150 — and `audio+ocr` (0.2%) rounds to zero. Same for `other`
  question types. Accept it; do not distort the sample to chase them.

### Sequencing

Stage 2 needs Ollama with `gemma4:12b` (~7.4 GB), which cannot coexist with the
vLLM server's 22.4 GB. Stage 1 and stage 4 are network-only and can run at any
time, including while the qwen3vl run is still going.

## Phase B — Generate answers for the sample

**Recommended: video + question only, no ASR, no routing.**
`generate.answers.qwen3vl.py --queries <2025.json> --videos videos_2025_GT
--ignore-routing --transcripts <empty dir>` runs as-is. The alternative — full
parity with the 2026 pipeline — additionally needs `transcribe.whisper.py` over
the sample and `route.query.inputs.py` to build a 2025 routing file, which is
several more hours for a variable that is not under test here.

State the consequence plainly when reporting: the resulting score is a
**floor**, not the pipeline's true accuracy, because the 2026 runs get a
transcript for 576 queries and routed inputs. If the floor is interesting, the
full-parity run is the follow-up.

Two conditions are needed for the expansion question, over the identical query
set:

| Condition | Answers |
|---|---|
| **baseline** | the run exactly as the pipeline produces it |
| **expanded** | the same run after `VQA/expand.answers.py` (`ANSWER_EXPANSION_PLAN.md` Phase 2) |

## Phase C — The scorer

**New file: `VQA/score.answers.py`.** Takes the 2026 JSON run format natively
(rather than `eval.vqa.ag.py`'s headerless CSV) and a GT CSV, and reports all
three metric families per query and in aggregate.

**Reproduce `eval.vqa.ag.py`'s aggregation exactly**, or the numbers are not
comparable to anything:

- normalisation is `' '.join(text.strip().lower().split())` (`:35`);
- METEOR is `meteor_score([word_tokenize(ref)], word_tokenize(hyp))` (`:109`) —
  one reference, per response;
- a query's score is the **mean over all ten answers** (`:125-126`), then the
  mean over queries (`:150-152`);
- NDCG uses the metric as *gain* against a strictly decreasing score vector
  encoding the submitted order (`ndcg_from_gains`, `:37-51`);
- honour the `SKIP` set of 18 Q_IDs (`:91`).

### The three metrics

| Metric | Implementation | Status |
|---|---|---|
| **METEOR** | `nltk.translate.meteor_score`, plus `nltk.download('wordnet')` and `punkt` | Official implementation, matches NIST |
| **BERTScore** | `bert_score.score(cands, refs, lang="en")` — defaults to `roberta-large` | Official implementation, matches NIST |
| **STS** | **Substitute required.** Cosine similarity of sentence embeddings via `sentence-transformers` (`all-mpnet-base-v2`) | **Not** the official metric — see below |

**On the STS substitute.** The official metric is the UMBC service, which is
gone. A sentence-embedding cosine is *a* semantic-textual-similarity measure and
is what the name describes, but it is a different function with a different
scale. It must be reported under a distinct name — `STS_local` — never as
"STS", and every document quoting it must say the official number could not be
computed. Its use is comparing our own runs to each other, which is all we need
it for. Before adopting it, spend ten minutes retrying the UMBC endpoint; if it
has merely moved, the official path is better.

### Environment

`nltk`, `bert_score` and `sentence-transformers` are installed nowhere on this
box. Install into the **Windows venv** (`E:\TREC_2026\.venv` — already has
pandas, numpy, sklearn), **not** `/home/aakash/cosmos-reason2/.venv`, which
serves vLLM and must keep its `transformers` pin. BERTScore over 400 × 10 pairs
wants the GPU; run it after the answering run has released the card, or accept
CPU.

### Trust the scorer before trusting its output

`score.answers.py` must be checked against `eval.vqa.ag.py` itself: convert one
run to the CSV shape, run both over the same 50 queries with STS disabled, and
confirm METEOR, BERTScore and both NDCGs agree to 3 decimal places. A
reimplementation that silently differs in aggregation is worse than no number.

## Phase D — What the numbers decide

1. **A baseline accuracy figure** for the project — the first one. Report mean
   METEOR, mean BERTScore, mean STS_local, NDCG-METEOR, NDCG-BERT, and the count
   of scored queries.
2. **The expansion gate.** Paired per-query comparison of baseline against
   expanded on the same queries: report the mean delta per metric, the fraction
   of queries improved, and a paired significance test. Proceed with expansion on
   the 2026 submissions only if METEOR improves without BERTScore or NDCG
   regressing.
3. **Cheap ablations, now that scoring exists** — each is a flag already present:
   `--budget brief` vs the routed tiers; `--ignore-routing` vs routed;
   temperature. These have all been judgement calls until now.

## Files

| File | Change |
|---|---|
| `VQA/probe.2025.metadata.py` | **new** — stage 1: `yt-dlp --skip-download` duration + availability over the 2,000 IDs, resumable |
| `VQA/sample.2025.eval.py` | **new** — stages 2–3: 2025 CSV → query JSON, candidate pool, marginal-matching selection of 150, reserve lists, achieved-vs-target report |
| `VQA/score.answers.py` | **new** — the three metrics over a 2026-format run |
| `VQA/EVALUATION.md` | **new** — the measured results, written whichever way they come out |
| `download_test_videos.py` | unchanged — driven with the converted 2025 query file |
| `VQA/route.query.inputs.py` | unchanged — pointed at 2025 copies via `--out` / `--cache` |
| `videos_2025_GT/` | **new** directory, kept out of the 2026 media tree |
| `VQA/eval.vqa.ag.py` | unchanged — used only as the cross-check oracle |

## Verification

1. **Scorer parity** — agreement with `eval.vqa.ag.py` to 3 dp on 50 queries
   (METEOR, BERTScore, both NDCGs). This gates everything downstream.
2. **Oracle sanity** — scoring the GT against itself must give METEOR ≈ 1.0 and
   BERTScore ≈ 1.0. If not, normalisation or alignment is wrong.
3. **Alignment** — assert every scored Q_ID's `Video_ID` matches the GT row, and
   report how many of the sampled queries were dropped for a missing video.
4. **Isolation** — confirm `videos_TEST_VQA/videos/` gained no files and the
   2026 run files are byte-identical afterwards.
5. **Report the sample size and attrition** next to every number. A METEOR mean
   over 380 downloadable 2025 queries is not "our accuracy"; it is our accuracy
   on that sample under video-only conditions.

## Risks

- **Download attrition is unknown and could be severe.** These are 2024-vintage
  YouTube IDs. Stage 1's metadata probe measures it before any download and
  before any stratum is chosen, which is the whole reason it comes first. If a
  duration bin loses most of its candidates, the sample cannot match that bin —
  report the shortfall rather than silently reweighting.
- **The sample is 7.5% of the 2025 set.** Adequate for the paired comparison it
  exists to settle, soft for an absolute accuracy figure, and blind to the OCR
  routes. All three stated above.
- **2025 is not 2026.** Different videos, questions and annotators. A verbosity
  effect measured there is evidence about the *metric*, which transfers; an
  accuracy figure is about last year's task, which transfers less.
- **STS is not the official STS.** Flagged above; the risk is someone later
  quoting `STS_local` as a NIST-comparable number.
- **Deadline: 17 August 2026.** This work does not block the submission. The
  qwen3vl run finishes first; unexpanded runs stay the fallback submission
  throughout.
- **Cost.** ~500 downloads, then ~500 inference calls at the measured ~35 s
  single-worker / ~11 h-per-2443 rate — roughly 2 h of GPU at 4 workers, plus
  the expanded condition again.

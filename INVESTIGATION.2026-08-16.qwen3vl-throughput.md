# Qwen3-VL throughput collapse — open investigation

**Status: OPEN, parked 16 Aug 2026 to let the submission run finish.**
Nothing here blocks the pipeline. Resume after the 17 August deadline.

Published write-up of the first half of this investigation:
<https://claude.ai/code/artifact/92c91a7c-42ec-421a-89f6-dad64d3cffc7>
— *"The Slowdown That Wasn't"*. **That page is accurate on the experiment and
the ruled-out causes, but its §04 "what was actually happening" overstates the
convoy effect. The corrected numbers are in §5 below; read this file first.**

---

## 1. The symptom

During `answers:qwen3vl` of the two-model ensemble (`ENSEMBLE_PLAN.md`), wall
clock per query rose steadily over nine hours:

| window | q/min | mean elapsed | mean video |
|---|---|---|---|
| last 800 | 2.49 | 90.5 s | 45.4 s |
| last 400 | 1.92 | 116.2 s | 67.3 s |
| last 200 | 1.35 | 162.9 s | 55.2 s |
| last 100 | 0.89 | 245.5 s | 93.8 s |

Projected completion moved from ~5 h to 8–12 h over the course of a morning.
Throughout, parallelism held at 3.7–3.8×, and there were **zero** truncations,
unparsed replies or failed prompt calls — nothing was erroring, it was just slow.

## 2. Cost normalised by work

The distinguishing measurement: milliseconds of wall clock per 1,000 prompt
tokens. If the machine degrades, this rises; if the work is simply heavier, it
is flat.

| hour | n | s/query | s/call | **ms per 1k prompt tok** | median video |
|---|---|---|---|---|---|
| 0 | 339 | 38.9 | 3.9 | **4.5** | 10.0 s |
| 2 | 264 | 51.7 | 5.2 | **7.2** | 7.7 s |
| 4 | 224 | 59.5 | 5.9 | **9.2** | 5.3 s |
| 6 | 178 | 76.6 | 7.7 | **10.6** | 19.6 s |
| 7 | 38 | 164.5 | 16.5 | **12.8** | 61.5 s |
| 8 | 93 | 46.9 | 4.7 | **4.7** | 76.0 s |

It rose 2.8× and then **returned to its starting value at hour 8, unprompted,
while the videos got longer**. Spontaneous recovery falsifies accumulation:
a leak does not heal itself.

## 3. Causes ruled out, with measurements

| Cause | Verdict | Evidence |
|---|---|---|
| Thermal / power throttling | ruled out | 68 °C, 1905 of 2115 MHz, 174 W of 370 W, all throttle reasons `Not Active` |
| vLLM preemption | ruled out | **0** preemption events in the whole serve log |
| Host memory / OOM | ruled out | 31 GB total, 22 GB free, no OOM kills in dmesg |
| Disk I/O over DrvFs (`/mnt/e`) | ruled out | 48–265 MB/s measured; ≤ 2 s per query against a ~50 s gap |
| Prefix-cache collapse | ruled out | hit rate steady 62–72% throughout |
| KV cache exhaustion | transient only | 2–7% typically; 39–42% during the slow window, back to 1.2% after |

## 4. The experiment — aged server vs fresh server

24 queries spanning 6.4 s to 320 s of video, run against the 8.7-hour-old
server, then the server restarted and the **identical set** re-run. Same
videos, prompts, seeds and worker count.

| condition | mean per query | total inference |
|---|---|---|
| during the batch (original) | 76.4 s | 1,833.6 s |
| aged server, re-run | **27.4 s** | 648 s |
| fresh server, re-run | **26.9 s** | 648 s |

**Aged vs fresh differ by 1.8% — within noise. Restarting the server changes
nothing.** This is the one firm conclusion of the investigation.

**Known flaw:** the control set was drawn from the *slow window*, i.e. selected
on the outcome, so part of the 76.4 → 27.4 s gap is regression to the mean. It
does not affect the aged-vs-fresh comparison (same queries in both arms), but
the in-batch penalty is overstated by that figure.

## 5. What the data actually says — correcting the artifact

The published page attributes the slowdown to convoys of long clips saturating
the KV cache. Clustering is real, but **the effect size is far too small to
explain the symptom**, and "long" turned out to be the wrong variable.

### 5a. Cost is an inverted U in video length, not monotonic

| band | n | prompt_tok | elapsed | fps |
|---|---|---|---|---|
| 0–10 s | 662 | 692 | 58.4 s | 4.00 |
| 10–20 s | 718 | 900 | 55.0 s | 3.97 |
| 20–40 s | 139 | 921 | 74.7 s | 3.02 |
| 40–80 s | 72 | 1,484 | 129.8 s | 1.24 |
| **80–150 s** | 114 | **3,333** | **151.2 s** | 0.78 |
| 150 s+ | 78 | 1,222 | 72.7 s | 0.36 |

The **most expensive clips are 80–150 s**, not the longest. `adaptive_fps`
targets a fixed frame count, so clips over 150 s are downsampled to 0.36 fps and
become cheap again. Any analysis keyed on "long video" is keyed on the wrong
axis.

### 5b. The convoy penalty is 1.16×, not the story

Redone with *expensive* defined empirically as the top quartile of prompt tokens
(≥ 763), rather than by duration:

| | actual order | if shuffled |
|---|---|---|
| consecutive runs | 255 | 348 |
| longest run | **11** | 4 |
| runs of 4+ | **26** | 1 |
| queries inside runs of 4+ | **153** | 4 |

| | median cost |
|---|---|
| expensive query inside a convoy of 4+ | 66.3 s |
| expensive query not in a convoy | 57.2 s |
| **convoy penalty** | **1.16×** |

Clustering is genuine — the alphabetical `Video_ID` sort really does collect
expensive queries into runs of up to 11, where shuffling would give 4. But a
1.16× penalty on 8% of queries cannot produce an 8× slowdown. **The convoy
mechanism is real and minor.**

### 5c. So the symptom is still not fully explained

Composition (2.7× between the cheapest and most expensive bands) plus convoy
(1.16×) gets to roughly 3×. The observed tail was 8×. **The residual is
unexplained.** Candidate explanations not yet tested are in §7.

## 6. The observation that reopened everything

After the A/B, two changes were made at once and the batch resumed:

1. the server was restarted (as part of Arm B), and
2. `group_work()` was changed to interleave long and short videos instead of
   sorting alphabetically by `Video_ID`.

Throughput on resume:

| | q/min |
|---|---|
| before (last 100 of the old run) | **0.89** |
| after restart + reorder | **11–14** |

**A 12–15× jump — against an A/B that said the restart alone does nothing.**

This is the central open question, and it must not be read as "the reorder
worked". Three confounds landed together:

- **Restart.** The A/B says this is worth ~2%, but the A/B ran 24 queries on an
  idle server, not a saturated batch. It may not generalise.
- **Reorder.** Plausible, but the measured convoy penalty is only 1.16×, so it
  should be worth ~16% at best, not 1,200%.
- **Composition.** The interleave deals the longest clips first, and those are
  cheap (712 prompt tokens, 21.8 s per query on the first 40 after resume). The
  fast stretch may simply be the cheap stretch, and the expensive 80–150 s band
  is still ahead.

Nothing here separates the three. The honest statement is: *throughput improved
enormously after two simultaneous changes, and we do not know which one did it,
or whether either did.*

### A quality signal worth carrying forward

The first 40 queries after the resume had **mean `pooled_unique` 5.5** and
**33 of 40 needing fallback padding**, against 9.3 and 24 of 40 in a comparable
pre-restart slice. Many rank-1 answers are literally `"unknown"`. Those are the
150 s+ clips at 0.36 fps: sparse, low-resolution frames the model declines to
answer from. **Part of the speedup is the model failing fast on hard queries,
which is not a win.** The ordering change cannot affect per-query results —
same seeds, same video, same plan — so this is a property of those queries, not
of the reorder. But it means throughput and quality moved in opposite
directions, and only throughput was being watched.

## 7. To test when this is resumed

1. **Separate the three confounds.** Re-run one fixed, representative slice
   under each condition independently: old order + aged server, old order +
   fresh server, new order + fresh server. The slice must be drawn *randomly*,
   not from the slow window — that was the flaw in §4.
2. **Test the saturated case.** The A/B ran 24 queries on an idle server. Repeat
   it with a continuously fed queue, which is the condition the batch is in.
3. **Instrument per-call, not per-query.** `elapsed` currently covers all ten
   calls of a query. Recording each call separately would show whether the cost
   is in prefill, decode or queue wait — the single most useful missing signal.
4. **Log vLLM engine stats alongside the client cache**, timestamp-aligned, so
   `Running`/`Waiting`/`KV usage` can be joined to the query that was in flight.
5. **Check the `elapsed` accounting.** With 4 workers, a query's wall clock
   includes waiting for GPU slots. Per-call server-side latency
   (`X-Request-Id` timing) would separate queue wait from compute.
6. **Reconsider `adaptive_fps`.** The inverted-U means the 80–150 s band gets
   the largest visual payload of any bucket. If that band is also where answer
   quality peaks, fine; if not, flattening the frame budget would cut the
   expensive tail with no loss.

## 8. Where the evidence lives

| Artifact | What it holds |
|---|---|
| `answers.task1.qwen3vl.ens.cache.jsonl` | per-query `elapsed`, `prompt_tokens`, `duration_sec`, `fps`, `pooled_unique`, `ts` — the client-side record every number above is derived from |
| `/home/aakash/qwen3vl.instruct.log` (WSL2) | vLLM engine stats: throughput, KV usage, prefix hit rate, running/waiting |
| `logs.ensemble/answers.qwen3vl.log` | orchestrator's captured stdout, timestamped |
| `VQA/diagnostics/` | the four analysis scripts used here |

The scripts in `VQA/diagnostics/` are standalone and read the cache directly:

```
diagnose.py     normalised cost by hour, like-for-like bands, server stats
contention.py   co-scheduling penalty, clustering vs shuffled baseline
cluster2.py     the corrected convoy analysis, expensive = top-quartile tokens
project_eta.py  duration-bucket ETA projection (note: optimistic, see below)
```

**`project_eta.py` was wrong twice**, both times optimistic — it predicted 5.1 h
remaining when the truth was closer to 9. It fits per-bucket means over the whole
run, and those are dominated by early cheap queries in the same bucket. Fix it or
distrust it.

## 9. What was changed in the code, and what it is worth

`VQA/generate.answers.qwen3vl.py:group_work()` now interleaves groups
long-short-long-short after sorting by duration, instead of sorting
alphabetically by `Video_ID`. It is deterministic, costs nothing and cannot hurt
the prefix cache — locality that matters is *within* a group (same video, same
plan) and adjacent groups are different videos either way.

**On the evidence in §5b it should be worth about 16%, not the 12× observed.**
Keep it; it is not wrong. Do not credit it with the speedup until §7.1 has been
run.

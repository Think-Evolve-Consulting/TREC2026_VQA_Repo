# Session 1–15 Sep 2026 — Task 2 (multiple choice) ranking campaign

Four vision-language models ranked all 2,443 TREC VQA 2026 multiple-choice
queries; a labelled 2025 dev set then measured which of them — and which
fusions of them — actually rank well. Every number below was read from a log,
cache or score file on disk; nothing is estimated unless it says so.

## Outcome in brief

- **Nine valid MC run files exist** (eight real + a dummy), each passing
  `validate_vqa_mc_2026.py --warnings-as-errors`.
- **The MC window is open until 18 Sep 2026; max 5 runs per task.** The AG
  (Task 1) window closed on 17 Aug, so no Task 1 file can be submitted now.
- **Best measured ranker: Qwen3.8-27B (INT4) — 0.800 Top-1 / 0.884 MRR** on a
  400-query labelled sample, against a random control at 0.253 / 0.517.
  Qwen3-VL-8B is statistically tied (0.790, p = 0.70); Gemma 4 (0.733) and
  Cosmos-Reason2 (0.650) are significantly worse.
- **No fusion beat the best single model.** Equal-weight RRF lost 3.5 points
  (p = 0.029); accuracy-weighted RRF tied (0.795, p = 0.86), with in-sample
  weights. This reversed my own prediction that fusion was the best free move.
- **Two silent failure modes were found and fixed**: Cosmos answering 23% of
  queries with no video (a `do_sample_frames` rejection swallowed by a
  text-only fallback), and `size.longest_edge` being a *whole-video* budget —
  which means the earlier Qwen3-VL Task 1 run saw thumbnail-resolution video.

## What can be submitted (MC)

| file | dev Top-1 / MRR | vs Qwen3.8 on dev | 2026 rank-1 differs from Qwen3.8 |
|---|---|---|---|
| **qwen38.run.task2.json** | **0.800 / 0.884** | best | — |
| **qwen3vl.run.task2.json** | 0.790 / 0.872 | tied (p = 0.70) | 457 |
| **fused.weighted.run.task2.json** | 0.795 / 0.881 (weights fit in-sample) | tied (p = 0.86) | 253 |
| **gemma4.vllm.run.task2.json** | 0.733 / 0.840 | −6.8 pts (p = 0.0005) | 497 |
| fused.rrf.run.task2.json | not scored in this 4-model form | — | 372 |
| fused.borda.run.task2.json | not scored in this form | — | ≈ RRF (98.2% same rank-1) |
| cosmos.run.task2.json | 0.650 / 0.783 | −15.0 pts (p ≈ 0) | 899 |
| qwen38.resampled.run.task2.json | not scored | — | 167 (near-duplicate) |
| dummy.run.task2.json | random control | — | do not submit |

**Recommended five**: qwen38 → qwen3vl → fused.weighted → gemma4.vllm →
fused.rrf (the fifth is a judgement call; the resampled run is almost a copy of
the first). Runs are not scored without the data agreement form — confirm it
was filed.

## Timeline

| date | work |
|---|---|
| 1 Sep | MC dummy generator; format confirmed against the validator |
| 4 Sep | Gemma 4 ranking pass (3.01 h). Cosmos pass: "chooses, doesn't rank"; 23% silent blindness; OOM kill; finished |
| 4–5 Sep | Qwen3-VL: Instruct weights re-downloaded (17 GB); whole-video budget bug found; 18.9 h run at 4k tokens |
| 5–6 Sep | Qwen3.8-27B identified, INT4 build chosen, 20-query trial, 7.53 h run |
| 6 Sep | Resampled Qwen3.8 re-run (7.52 h); dev harness, sample, control; fusion script; four dev runs; fusion experiments; Task 1 Qwen3.8 smoke (stopped) |
| 15 Sep | Submission rules checked; accuracy-weighted fusion built and validated; this write-up |

## The ranking contract (shared by all four passes)

- **Rank option numbers, never option text.** The model returns a permutation
  of `1..N`, mapped back to the official strings by index, so no rank is spent
  twice and no option goes unranked. `as_permutation()` repairs repeats and
  drops; `check_permutation()` refuses to write anything the validator would
  reject.
- **Permute the presentation.** Official order, exact reverse, then a seeded
  shuffle, combined by Borda count. Position bias measured at chance in every
  run: rank 1 on the officially-first option 15.2–17.4% against 14.9%
  expected.
- **No `Time` field** in MC runs — the validator rejects any key outside
  `Rank`/`Answer`.
- **Chance in the 2025 MC release is uniform**: the correct option sits at
  position 1/2/3/4 in 524/505/492/475 of 1,996 queries, so presentation order
  is not an answer-key leak.

## The four passes

### Gemma 4 12B (QAT INT4) — `VQA/rank.options.mc.py`
vLLM 0.27 on port 8010, frames from `extract.frames.py`, 3 passes, 8 workers.
**3.01 h**, 0 failed, 0 repaired, 0 fallbacks; 79.7% of queries unanimous
across passes; the Borda winner differs from a single official-order pass on
9.7%. Raw per-pass rank-1 at presented position 1: 16.2% vs 14.9% — a small
bias the permutation averages out.

### Cosmos-Reason2-2B — `VQA/rank.options.mc.cosmos.py`
vLLM 0.12 on port 8008, native video, 1 pass (a seeded shuffle), reasoning on.
- **A reasoning model chooses; it doesn't rank.** 9 of the first 20 replies
  were a bare `{"ranking": [4]}`. Three prompt variants moved full-length
  rankings from 1/6 to 3/6 and no further. Fix: when stage one leaves options
  unplaced, a second text-only, schema-constrained call orders them from the
  model's own pick and reasoning — 6/6 full permutations at ~0.6 s against
  ~17 s for the reasoning call. Needed on 37.7% of queries; 0.34 h in total.
- **23% silent blindness.** 190 of the first 830 queries ranked with no video
  although the file existed: the server decodes every frame (`num_frames=-1`),
  and asking the processor to re-sample (`do_sample_frames`) raised a 400 that
  the client treated as "unprocessable video" and retried text-only. Fix: a
  ladder — sampled video → unsampled video → text only.
- **OOM kill.** The unsampled retry let the processor resize every native
  frame (300 for a 10 s clip instead of ~40); at 8 workers that OOM-killed the
  server 790 queries into the second stage, leaving 10 fallback rankings. Fix:
  always transcode to the target fps on that path, `RAW_DECODE_BYTES_LIMIT`
  4 → 2 GiB, workers 8 → 4. 13 queries redone; zero blind queries remained
  except the 16 with no file.
- Final: 2,049 replies parsed as JSON, 35 salvaged from a digit list, 359 via
  the ordering call; 60 repaired; 0 failed; 18 without video (16 no file).

### Qwen3-VL-8B-Instruct — `VQA/rank.options.mc.qwen3vl.py`
vLLM 0.12 on port 8009, native video, 3 passes, schema-constrained replies.
- The Instruct weights were missing (a 12 KB `refs/main` stub); 17 GB
  re-downloaded.
- **`size.longest_edge` is a whole-video pixel budget.** Holding clip and fps
  fixed: 2,048 / 8,192 / 16,384 tokens → 1,152 / 4,004 / 8,096 prompt tokens
  (62-frame clip) and 1,079 / 4,229 / 6,231 (14-frame clip). Read per-frame,
  as `generate.answers.qwen3vl.py` does, two clips got 485 and 399 visual
  tokens against a planned 17,792 and 5,376.
- Cost with the fix, 20 queries × 1 pass: 8,192 tokens 677 s (~23 h full),
  2,048 tokens 258 s (~8.7 h). **4,096 chosen by the user**; ran **18.88 h**
  (quoted ~15 h — passes 2–3 cost 30.7 s each at 4k, not the 2–3 s measured at
  8k). 7,329/7,329 replies parsed; 0 repaired/failed/fallback; `no_dsf` rung
  recovered 196 queries; 83.6% unanimous.
- 116 queries without video = 184 routed to audio − 68 given video back
  (no speech) → 100 with a file + 16 without. By design, not failure.

### Qwen3.8-27B (RedHatAI INT4) — `VQA/rank.options.mc.qwen38.py`
vLLM **0.27** (only 0.27 registers `Qwen3_5ForConditionalGeneration`) on port
8011, native video, 3 passes, 4,096 visual tokens.
- Multimodal (declares `video_token_id`), same vision geometry as Qwen3-VL.
  bf16 ≈ 54 GB, so the compressed-tensors INT4 build: 17.71 GiB loaded,
  **3.3 GiB KV cache = 46,421 tokens**, workable because 48 of 64 layers are
  linear attention.
- Reasons in plain `content`; the schema suppresses it (all 7,329 parsed).
- **7.53 h — 2.5× faster than Qwen3-VL at the same budget.**
- `do_sample_frames` rejected on 1,765 queries (72%); `num_frames` silently
  ignored without it (identical tokens at 16 and 64) and rejected with it — no
  kwarg controls temporal sampling. Patch: below the `no_dsf` rung, serve a
  copy transcoded to the planned fps.
- Full re-run under the patch (`qwen38.resampled.run.task2.json`, 7.52 h),
  with the `full` rung as a control: rank-1 changed on 8.0% of control
  queries vs 6.3% of the targeted ones — **the fix made no measurable
  difference.** Originals left byte-identical (md5 `4a2199dd…`).

## How the runs relate (2026 rank-1 agreement)

| | gemma4 | cosmos | qwen3vl | qwen38 |
|---|---|---|---|---|
| gemma4 | — | 61.2 | 77.4 | 79.7 |
| cosmos | 61.2 | — | 63.5 | 63.2 |
| qwen3vl | 77.4 | 63.5 | — | 81.3 |
| qwen38 | 79.7 | 63.2 | 81.3 | — |

All four agree on 52.6% of queries, at least three on 81.9%, all four differ
on 1.9%. Cosmos is the outlier in every pair.

## Measuring against labels — `VQA/eval.mc.dev.py`

- **Dev set**: the 2025 MC release — 2,000 four-option queries, ground truth
  for all, 1,327 videos on disk. Sample: **400, seed 2026**; 2 have an answer
  not among their options (ceiling 99.5%). Frames extracted for 358 videos in
  1.2 min.
- **Scorer trap**: `eval.vqa.mc.py` divides by the ground-truth file size, not
  the run — scoring a sample against the full GT reports ~5× too low. The
  harness writes a sample-restricted GT CSV and refuses partial runs.
- **SKIP set**: the scorer ignores 18 queries; they are excluded from sampling.
- **Paired design**: exact McNemar on discordant queries + bootstrap MRR CI.
  n = 400 resolves ~4.5 points at 2 SE (20% discordance). Two *random* runs
  differed by 3.0 points and the test correctly said p = 0.36.
- **Control**: random permutation scored 0.253 / 0.517 against theoretical
  0.250 / 0.521.

| run (400 queries) | Top-1 | MRR | vs Qwen3.8 (paired) | wall clock |
|---|---|---|---|---|
| Qwen3.8-27B | **0.800** | **0.884** | — | 1.32 h |
| RRF 4 models, 8:8:4:1 | 0.795 | 0.881 | −0.5, p = 0.86 | 0 |
| Qwen3-VL-8B | 0.790 | 0.872 | −1.0, p = 0.70 | 2.23 h |
| RRF top-3, equal | 0.790 | 0.877 | −1.0, p = 0.58 | 0 |
| RRF strong pair | 0.785 | 0.875 | −1.5, p = 0.42 | 0 |
| Borda 3 models, equal | 0.767 | 0.865 | −3.2, p = 0.053 | 0 |
| RRF 3 models, equal | 0.765 | 0.865 | −3.5, p = 0.029 | 0 |
| Gemma 4 | 0.733 | 0.840 | −6.8, p = 0.0005 | 0.29 h |
| Cosmos-Reason2 | 0.650 | 0.783 | −15.0, p ≈ 0 | 0.75 h |
| random control | 0.253 | 0.517 | −54.8 | 0 |

MRR 0.884 against Top-1 0.800 means a miss usually still lands at rank 2 —
the tail carries signal. Dev limits: all 4-option (2026 is 81% 8-option),
median clip 15 s, 370 of 400 without a transcript, all dev runs with
`--ignore-routing`.

## Task 1 with Qwen3.8 — unfinished

`VQA/generate.answers.qwen38.py` = the Qwen3-VL Task 1 pass with the checkpoint
swapped, the whole-video budget fix, and a 2 GiB decode ceiling. Unconstrained,
it spent 489–592 s per query at ~5,200 output tokens and padded 6 of 10 slots
on the third query (reasoning preamble truncated before the JSON). A
`--json-schema` run was stopped before producing a timing. Moot for 2026 — the
AG window has closed — but its cost is still unmeasured.
`answers.task1.qwen38.cache.jsonl` holds 1 stale record; delete before reuse.

## Infrastructure lessons

| symptom | cause | fix |
|---|---|---|
| server dies ~20 s after launch, `KeyboardInterrupt` in init | WSL2 tears the distro down when the last session exits | hold a session: `wsl.exe -d Ubuntu-24.04 -- sleep 28800` |
| `/home/...` paths become `C:/Program Files/Git/...`; `-u: command not found` | Git Bash MSYS path conversion | `MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'` |
| `$'\r': command not found` | Python `write_text` on Windows writes CRLF | open with `newline="\n"` for shell scripts |
| `pgrep -f` counts itself | pattern appears in its own command line | bracket it: `[v]llm serve` |
| heartbeat flaps between up/down | `/v1/models` slow under load | 10 s timeout, two misses before "down" |
| gemma4 failure not in journal | unit logs to `/home/aakash/gemma4/server.log` | read that file |
| `deep_gemm ... AssertionError` at startup | optional import wants CUDA_HOME | benign |

## Where I was wrong

- Qwen3-VL at 4k: quoted ~15 h, took 18.9 h (prefix-cache saving did not
  transfer from 8k).
- Called 100 Qwen3-VL no-video queries "transient failures"; they were
  router-withheld (my diagnostic used an unrouted plan). Stopped the re-run.
- Predicted unweighted fusion as "the highest-value move"; it lost 3.5 points.
- Blamed a Gemma 4 start failure on my own port race; it was the distro
  teardown both times.
- Cosmos OOM: the retry path I added reasoned about prompt tokens, not host
  RAM; and 8 workers exceeded available memory on paper before the run.
- Said the long tail (long clips) was where default sampling hurt most; only
  21 queries were over-sampled — 1,708 short clips were under-sampled instead.
- Started Task 1 Qwen3.8 without `--json-schema` despite knowing the trait.
- Plan claimed generate-then-match needed no inference; on dev, the 2025
  Task 1 runs cover only 42 of the 400 sampled queries.

## Open items

1. Choose ≤ 5 MC runs and submit before 18 Sep; confirm the data agreement form.
2. Plan step 3, pointwise likelihood scoring — not started (needs the 8011 server).
3. Plan step 4, generate-then-match — needs a Task 1 pass over the 400 dev queries.
4. Port fixes back for future runs: whole-video budget → `generate.answers.qwen3vl.py`;
   `do_sample_frames` ladder → `generate.answers.cosmos.py`.

## Inventory

Scripts: `VQA/generate.dummy.mc.run.py` (178 lines; re-created — it vanished
between sessions), `VQA/rank.options.mc.py` (1,209), `.cosmos.py` (1,316),
`.qwen3vl.py` (1,424), `.qwen38.py` (1,503), `VQA/eval.mc.dev.py` (446),
`VQA/fuse.rankings.mc.py` (230), `VQA/generate.answers.qwen38.py` (2,044).
Caches `rankings.task2.*.cache.jsonl`; logs `logs/rank.options.mc.task2.*.log`,
`logs/dev.mc.*.log`; dev artefacts in `eval.2025.mc/`. Memory notes:
task2-mc-ranking-pipeline, cosmos-mc-ranking-traps, qwen3vl-visual-budget-bug,
qwen38-27b-int4-serving, task2-mc-dev-harness.

# Session Notes — 2026-09-17/18 — A two-agent debate over the ranker swarm

Built after the single-judge orchestrator failed its gate
(`SESSION.2026-09-17.orchestrator-judge.md`). Reading the two judge runs
*against each other* showed that the failure was informative rather than final,
and this session turned that reading into a debate system.

**Status: built and verified against mock agents; not yet run on real models.**
It needs two Colab sessions serving the two agents. Nothing about the current
submission has changed.

## What the two judge runs revealed

Neither judge beat the baseline alone. But they are not the same judge:

| On the 203 dev queries the panel disputed | n | correct |
|---|---|---|
| the two judges **agree** with each other | 153 (75%) | 112 (baseline 109 there) |
| the two judges **disagree** | 50 (25%) | qwen38-judge 20, internvl-judge 18, **baseline 24**, **either-one-right 38** |
| oracle over the two judges, all 203 | — | **150** vs baseline 133 |
| the free consensus rule (both agree *against* the baseline) | fires 23× | 12 right vs baseline's 9 — net +3, noise |

Three facts, and each one became a design rule:

1. **Debate only where the two agents disagree.** On those 50 queries something
   present is right 38 times, and only ~20–24 are captured.
2. **An unresolved debate falls back to Qwen3.8, never to an agent.** On exactly
   that contested set the baseline (24) beats both agents (20, 18). This
   asymmetry is the single most important thing the dev data said.
3. **Arbitration cannot fix counting.** 38 of the 203 disputed dev queries are
   "how many" and the two-judge *oracle* only reaches 24. That is an evidence
   problem, not an argument problem.

## The design

**Rounds**, each one only over the queries still split:

| Round | What happens |
|---|---|
| R0 | both agents rank independently, with frames, the panel's rankings, the scorecard card, the transcript, the router verdict, and any counting/audio aid |
| R1 | each sees the other's pick and reason, and must hold or switch **citing frame timestamps** |
| R2 | each answers the other's citation directly: concede, or say what that frame actually shows |
| R3 | converged at any round wins; still split → the baseline's ranking |

**Safe by construction.** The emitted run equals `qwen38.run.task2.json`
everywhere except where the panel disputes *and* the debate converges on a
different option. A half-finished run is therefore a valid, submittable file,
and `--max-overrules N` caps how far it can ever drift. This property replaces
the dev gate, which the user deliberately skipped for this build.

**Frames travel as references, not pixels.** Both Colab sessions mount the same
Drive folder and read `file://` paths themselves — llama.cpp via `--media-path`,
vLLM via `--allowed-local-media-path`. A request is ~1 KB instead of ~800 KB of
base64, and later rounds re-read the same frames for free. The orchestrator runs
on the workstation and relays each side's statement to the other; the two
servers never talk directly, so each side's context is exactly what we chose to
show it.

## Evidence the panel never had

| Evidence | Applies to | Why |
|---|---|---|
| 32 frames (64 for counting), cut fresh at 448×448 | all disputed | the judges had been re-arguing the same 16 frames the panel saw. The old extraction averages **14.5 frames per video**, so raising a cap would have changed nothing — the frames are re-cut from the videos |
| dense strip + "report the highest count you can see, never average" | 269 counting queries | every model undercounts; the two-judge oracle reached only 24 of 38 on dev |
| CLAP per-option audio scores | 135 sound queries | no panel member hears anything. ASR cannot answer "which brass instrument"; an audio encoder can |
| sequence logprobs | all | self-reported confidence barely separates right from wrong (0.87 vs 0.83) |
| panel Borda scores, `firsts`, per-member pass agreement | all | already in the caches, previously unused |
| router verdict | all 2,443 | available for every 2026 query, where dev had only 42 |

Because nothing is being scored on dev any more, the dev-measured per-category
reliabilities (Qwen3-VL is the better counter, Cosmos the outlier) became
legitimate priors rather than leakage.

## 2026 sizing, measured

| | count |
|---|---|
| queries | 2,443 |
| panel (7 runs) unanimous on rank 1 | 1,097 (45%) — copied through untouched |
| **disputed → R0 load, per agent** | **1,346 (55%)** |
| of those, counting / sound | 269 (20%) / 138 (10%) |
| distinct videos | 1,139 after dropping 14 queries with no usable video |
| frames to cut on each Colab session | **42,897** (199 videos at 64, most at 32) |
| manifest size — the only upload | **78 KB** |

Estimated cost with both sessions in parallel: frame cutting 20–40 min, R0
1.5–2 h, R1+R2 ~1 h, assembly and validation ~15 min.

## What was built

| File | Purpose |
|---|---|
| `VQA/debate.rank.mc.py` | the orchestrator: rounds, relaying, two endpoints with API keys, logprobs, evidence injection, `--max-overrules`, resumable cache. Imports the ranking contract from `rank.options.mc.muse.py` and the panel helpers from `judge.rank.mc.py` rather than restating them |
| `VQA/prep.debate.mc.py` | local prep: the manifest, the counting flags, and CLAP audio scoring on the 3090 |
| `notebooks/colab_serve_qwen38.ipynb` | vLLM 0.27, Triton attention, bfloat16 KV, `--allowed-local-media-path`, cloudflared tunnel |
| `notebooks/colab_serve_internvl.ipynb` | prebuilt llama.cpp CUDA binaries, the Q4 GGUF, `--media-path`, cloudflared tunnel |

Both notebooks cut their own `/content/frames448` from the mounted videos, and
each ends by printing the three flags to paste into the orchestrator.

## Verification

Against mock agents with fixed personalities — one that never concedes, one that
does — every path was exercised before any GPU was involved:

- **The safety property is proven, not asserted.** With two agents that never
  concede, the emitted run is **SHA-256 identical to `qwen38.run.task2.json`**.
- R1 convergence (4/4), R2 counter-rebuttal convergence, and deadlock → baseline
  all behave as designed; `--max-overrules` suppressed the excess overrule.
- Requests carry **64 `file://` references for a counting video, 32 otherwise,
  zero base64**, with the panel table, reliability card, counting instruction and
  `logprobs: true` all present — checked in the mock's request log.
- The emitted file passes `validate_vqa_mc_2026.py --warnings-as-errors`
  (2,443 items, 0 errors, 0 warnings).

## Three bugs caught in testing, each one silent

1. **Over-dense frame requests.** Asking a 1-second clip for 64 frames needs
   64 fps; ffmpeg returns fewer and the manifest would then promise references
   that 404 on the agents. Frame counts are now capped at 8 fps of source, which
   is why 231 counting videos became 199 at the full 64.
2. **`ClapProcessor`'s renamed keyword.** transformers 5.x takes `audio=`, not
   `audios=`. The first full audio pass scored **0 of 138** and reported them as
   "no usable audio" — the decode had worked fine.
3. **CLAP ranking the numeral "1" first on counting questions.** "How many
   voices can be heard" matches the sound pattern, but its options are bare
   numerals, and scoring a digit string against a soundtrack measures a text
   prior, not audio. It put "1" top nearly every time — which would have pushed
   the agents toward *exactly* the undercount the counting aid exists to fix.
   Audio scoring is now restricted to descriptive options.

Audio quality after the fix, on the 135 descriptive sound queries: median spread
between best and worst option **6.6**, none flat. It discriminates where it
should — "what sound does the helicopter make" puts *rhythmic thumping* (+7.25)
and *loud roaring* (+5.61) clear of *silent* and *soft humming* — and is weak
where the options are near-synonyms, such as music genres. The prompt presents
it as "one more piece of evidence, not a verdict".

## How to run it

1. Copy `debate.manifest.json` (78 KB) into the Drive folder — the only upload.
2. Open both notebooks in two Colab sessions, set `DRIVE_ROOT`, run top to
   bottom. Each cuts its frames and prints `--host-x / --model-x / --api-key-x`.
3. Smoke 20 queries, then the full 1,346:

```
python VQA/debate.rank.mc.py \
    --host-a <A> --model-a RedHatAI/Qwen3.8-27B-INT4 --api-key-a <KEY> \
    --host-b <B> --model-b internvl3.5-30b-a3b --api-key-b <KEY> \
    --media-root file:///content/frames448 \
    --manifest debate.manifest.json --evidence evidence.2026.json \
    --runs qwen38.run.task2.json qwen3vl.run.task2.json gemma4.vllm.run.task2.json \
           cosmos.run.task2.json internvl.run.task2.json llavavideo.run.task2.json \
           videollama3.run.task2.json \
    --out debate.run.task2.json --cache rankings.task2.debate.cache.jsonl --limit 20
```

## Honest expectations

The two-judge **oracle** on dev was worth +17 queries (0.843 against 0.800), and
a real debate captures a fraction of an oracle. Five attempts at combining these
models have now landed on the baseline: Task 1 two-model fusion, equal-weight
RRF, accuracy-weighted RRF, the 781,248-fusion search, and the single judge.
This is the sixth, and the first with evidence the panel never had — which is
the only reason to expect anything different.

**It is unmeasured.** No dev gate was run, by choice, so the safety property is
what stands between this and a worse submission. The recommended follow-up, once
2026 is settled, is to run the identical pipeline over the 203 dev disputed
queries and score it, so the next campaign knows whether debate is worth keeping.

## Open items

1. The MC window closes 18 Sep. The submission is unchanged:
   **qwen38 → qwen3vl → fused.weighted → gemma4.vllm → fused.rrf**.
2. The debate has never spoken to a real model — only mocks. The first Colab
   smoke test is where serving traps will surface, as they did for the judge.
3. `--decider binary` exists but is off by default, being unmeasured.

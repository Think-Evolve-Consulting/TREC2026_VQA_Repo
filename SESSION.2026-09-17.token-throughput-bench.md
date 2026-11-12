# Session 17 Sep 2026 — prefill and decode throughput for all eight VLMs

Every vision-language model this project has run was measured for **prefill and
decode tokens per second**, on **real TREC 2026 queries**, for **both tasks**.
Eighty requests per task, ten clips spanning 1.1 s to 389 s, one model on the
3090 at a time. Every number below was read from `bench.tokens.jsonl`,
`bench.tokens.csv` or a server log on disk; nothing is estimated unless it says
so.

## Outcome in brief

- **160 measured requests, 0 failures in the final runs.** Task 1 (answer
  generation) and Task 2 (MC ranking), eight models each.
- **Prefill spans 6.4× across the fleet** — 4,737 tok/s (Cosmos-Reason2-2B) to
  745 (Muse Glimmer 30B). **Decode spans 10×** — 139.3 (InternVL3.5-30B-A3B)
  to 13.8 (Qwen3.8-27B on a fallback config).
- **The measurement reproduces.** The same ten clips were measured twice with
  different prompts (Task 1, then Task 2). Prefill agreed to within 2.3% on six
  of eight models, 6.2% worst case. Gemma 4 came back at 1,930.0 then 1,929.7.
- **Task 2 is prefill-bound.** Six of eight models answer a ranking query in
  36–44 tokens, so decode is under a second and per-query cost is essentially
  TTFT. InternVL's 3× decode lead buys ~0.2 s there and is worth little.
- **Three bugs were found and fixed**, two of them in the bench itself, plus a
  serving trap that took three attempts to diagnose (see *Failures*).
- **The numbers compare models *as deployed here*, not architectures.** KV
  cache dtype differs between models and is not controlled for (see *Caveats*).

## What was built

| file | role |
|---|---|
| `VQA/bench.tokens.py` | the measurement client: model registry, real-prompt construction, streaming and two-point timing, JSONL + CSV output, `--report` |
| `VQA/serve.vllm.sh` | serves any of the five vLLM models by name in WSL2, with that model's env vars and flags |
| `VQA/bench.tokens.sh` | runs the client from inside WSL2 with the venv matching the server |
| `VQA/run.bench.tokens.ps1` | drives the whole sweep, one model on the card at a time; starts, benches and stops each server |

Outputs, at the project root:

- `bench.tokens.jsonl` — one record per (model, task): config, the chosen
  queries with their routing plans, every sample, every raw reply.
- `bench.tokens.csv` — one row per request, 25 columns, raw reply included.
- `logs/bench.task1.*.log`, `logs/bench.task2.sweep.log`,
  `logs/bench.server.*.log`.

## Method

    prefill tok/s = prompt_tokens / TTFT
    decode  tok/s = (completion_tokens - 1) / (t_last - t_first)

TTFT is measured from the request leaving the client to the first content delta
arriving, so for a VLM it **includes the vision encoder and any server-side
video decode**. That is the honest number for these workloads — no request in
production is text-only. The first decoded token is excluded from the decode
window because its time already sits inside TTFT.

**The prompt is the real one.** `bench.tokens.py` imports the pass that owns the
task and calls its own prompt machinery, so the bench cannot drift from the run:

- `--task 1` → `generate.answers.py`: `SYSTEM_PROMPT`, `build_context()`,
  `PROMPT_TEMPLATE`, `OCR_RULE`/`TEMPORAL_RULE`, ten answer slots.
- `--task 2` → `rank.options.mc.py`: same, plus `render_options()` and
  `presentation_orders()`; pass 0 (the official option order) of the three a
  real run asks.

Routing is applied per query exactly as the runs apply it, so a query the router
marked temporal gets the temporal rule here too.

**Four controls:**

1. **Prefix caching is defeated.** A per-request nonce goes in the *system*
   message, ahead of the images, so the whole prefix is new. Verified from the
   servers' own counters: **0.0–0.6% prefix cache hit rate** across the sweep.
   Without this the prefill rate would be a cache lookup — the Qwen3-VL pass
   once measured ~70 s cold against 1.8–3 s warm for the same prefix.
2. **Ten clips spanning the duration range.** `--spread duration` walks a
   geometric series from the shortest clip to the longest. The corpus is heavily
   skewed (median 10.3 s, p90 56 s, max 389 s), so sampling at random would put
   eight of ten draws near the median and measure one point on a curve.
   `--spread corpus` reproduces the real distribution when the question is what
   a full run costs instead.
3. **One query per video**, so no two requests share a frame prefix.
4. **Decode runs to its natural end**, so the replies are real and kept.
   `--force-length N` pins the window instead when rates must compare exactly.

**VideoLLaMA3 has no server** — it runs in-process from `.venv-videollama3` and
is measured two-point: `max_new_tokens=1` is the prefill, and the slope to a
full generate is the decode rate. It reuses `rank.options.mc.videollama3.py`'s
loader so it inherits the block-wise vision-attention patch.

## The sample

Ten queries, chosen deterministically (`--videos 10 --spread duration --seed
2026`), identical for both tasks because Task 1 and Task 2 share Q_IDs and
videos.

| Q_ID | video | duration (s) |
|---|---|---|
| 211 | 6666.webm | 1.1 |
| 1482 | 7118.webm | 2.0 |
| 1112 | 8560.webm | 4.0 |
| 1142 | 8347.webm | 7.6 |
| 1254 | 1qgVLziQsVE | 14.7 |
| 604 | Xgi2EoWTN34 | 28.3 |
| 190 | Calloused_Hands-4.webm | 54.0 |
| 1827 | Archipelago-20.webm | 105.0 |
| 461 | Memphis-5.webm | 202.0 |
| 264 | Archipelago-22.webm | 389.0 |

## Results — Task 1 (answer generation)

Ten answers asked per reply. Medians over ten requests; `gen` is completion
tokens.

| model | prefill tok/s | decode tok/s | prompt tok | gen tok | gen min–max | TTFT s | ok |
|---|---|---|---|---|---|---|---|
| Cosmos-Reason2-2B | **4,745** | 61.1 | 10,790 | 47 | 36–93 | 2.59 | 10/10 |
| LLaVA-Video-7B | 3,281 | 46.7 | 6,678 | 66 | 41–256 | 2.07 | 10/10 |
| VideoLLaMA3-7B | 2,569 | 29.0 | 4,464 | 41 | 30–128 | **1.80** | 10/10 |
| Qwen3-VL-8B | 2,158 | 42.5 | 4,650 | 56 | 15–138 | 2.28 | 10/10 |
| Gemma 4 12B INT4 | 1,930 | 25.7 | 4,484 | 79 | 57–133 | 2.48 | 10/10 |
| InternVL3.5-30B-A3B | 1,671 | **134.4** | 4,403 | 85 | 44–160 | 2.62 | 10/10 |
| Qwen3.8-27B INT4 | 978 | 14.1 | 4,703 | 256 | 256–256 | 4.93 | 10/10 |
| Muse Glimmer 30B | 745 | 43.4 | 4,334 | 382 | 246–466 | 5.73 | 10/10 |

**Answer quality varied more than throughput.** Counted from the raw replies in
the CSV:

- **Gemma 4** produced real ten-answer JSON on all ten queries, the tightest
  honest gen range (57–133). The most reliable Task 1 generator measured.
- **VideoLLaMA3 echoed the prompt's placeholder** — literally
  `{"answers": ["...", "...", …]}` — on 5 of 10, and returned ten empty strings
  on a sixth. Only 4 of 10 real answers. It ranks option *numbers* competently
  (0.738 Top-1 on the Task 2 dev set) but copies the template when asked to
  generate free text into ten slots.
- **Cosmos answered "unknown" on 5 of 10**, consistent with a 2B model on
  open-ended generation.
- **Qwen3.8 produced no JSON at all** — prose reasoning, truncated at 256 on
  every query (see *Caveats*).
- Qwen3-VL's 15-token floor is a one-element `{"answers": ["unknown"]}`;
  LLaVA-Video's 256 ceiling is one truncated reply.

## Results — Task 2 (multiple-choice ranking)

A permutation of the option numbers plus a confidence. Whole sweep: **29
minutes**, 13:29:45 → 13:58:45, 80/80 requests, 0 failures.

| model | prefill tok/s | decode tok/s | prompt tok | gen tok | gen min–max | TTFT s | ok |
|---|---|---|---|---|---|---|---|
| Cosmos-Reason2-2B | **4,737** | 68.6 | 10,875 | 37 | 25–37 | 2.62 | 10/10 |
| LLaVA-Video-7B | 3,264 | 48.0 | 6,784 | 36 | 24–36 | 2.12 | 10/10 |
| VideoLLaMA3-7B | 2,613 | 28.9 | 4,570 | 36 | 24–36 | **1.80** | 10/10 |
| Qwen3-VL-8B | 2,207 | 40.4 | 4,708 | 36 | 24–37 | 2.25 | 10/10 |
| Gemma 4 12B INT4 | 1,930 | 26.3 | 4,589 | 41 | 29–41 | 2.38 | 10/10 |
| InternVL3.5-30B-A3B | 1,764 | **139.3** | 4,508 | 44 | 32–44 | 2.57 | 10/10 |
| Qwen3.8-27B INT4 | 999 | 13.8 | 4,761 | 256 | 256–256 | 4.91 | 10/10 |
| Muse Glimmer 30B | 791 | 45.3 | 4,442 | 411 | 185–647 | 5.66 | 10/10 |

VideoLLaMA3 behaves correctly here, returning valid permutations
(`{"ranking": [3, 4, 1, 2, 5, 6, 7, 8], "confidence": 0.0}`) — the difference
between ranking numbers and generating text. Its confidence is always 0.0, as
the earlier dev work found.

## What the numbers say

**Task 2 is prefill-bound; Task 1 less so.** Six of eight models emit 36–44
tokens on a ranking query, so decode costs under a second and per-query wall
clock is essentially TTFT. Task 1 replies are 41–85 tokens for the same six —
still small, but twice the decode.

**The two exceptions are the two that do not produce the expected output.**
Muse spends 411 tokens reasoning before committing to a permutation — ~9 s of
decode at 45 tok/s, making it the only decode-bound model in the set, and
roughly 4.5× the wall clock of the others per query. That is the concrete cost
behind its 0.755 dev score against Qwen3.8's 0.800. Qwen3.8 truncates at 256 in
prose because the bench sends no `response_format`.

**Prompt size is a serving decision, not a model property.** All three
video-payload models received byte-identical clip files, yet:

| clip | Qwen3-VL | LLaVA-Video | Cosmos |
|---|---|---|---|
| 1.1 s | 4,009 | 6,439 | 4,010 |
| 2.0 s | 4,499 | 6,437 | 9,599 |
| 28.3 s | 4,674 | 6,674 | 11,981 |
| 389 s | 5,626 | 7,970 | 13,738 |

- **Qwen3-VL** sends `size.longest_edge = 4096 × 2 × 1024`, a *whole-video*
  pixel budget, so a 389 s clip and a 1 s clip cost nearly the same. Flat by
  construction.
- **LLaVA-Video's server** pins `num_frames: 32`. Flat at a higher level; the
  most predictable model in the set (decode 46.7–48.5 across the whole range).
- **Cosmos** has neither, in this bench — see *Caveats*.

**InternVL's decode lead is real but situational.** 134–139 tok/s, 3× the next
fastest, on both tasks. At a 44-token reply that is 0.3 s against 2.6 s of
prefill, so it changes nothing on Task 2. It would matter for a reasoning-length
output: what costs Muse 9 s would cost InternVL 3 s.

**Gemma 4's decode is the weakest of the well-configured models** (25.7/26.3
tok/s) despite being a 12B INT4 on vLLM. Every service here runs
`--enforce-eager` because there is no CUDA toolkit for FlashInfer's JIT; without
CUDA graphs, per-token launch overhead dominates at batch size 1 and the penalty
falls on decode, not prefill (its prefill is a respectable 1,930). InternVL is
5.2× faster at decode on similar prompt sizes. If decode ever becomes the
bottleneck, that gap is a serving-config problem, not a model-choice one.

## Reproducibility

The same ten clips measured twice, different prompts:

| model | T1 prefill | T2 prefill | Δ |
|---|---|---|---|
| Gemma 4 | 1,930.0 | 1,929.7 | −0.02% |
| Cosmos | 4,745 | 4,737 | −0.2% |
| LLaVA-Video | 3,281 | 3,264 | −0.5% |
| VideoLLaMA3 | 2,569 | 2,613 | +1.7% |
| Qwen3.8 | 978 | 999 | +2.1% |
| Qwen3-VL | 2,158 | 2,207 | +2.3% |
| InternVL | 1,671 | 1,764 | +5.6% |
| Muse | 745 | 791 | +6.2% |

The two llama.cpp models drift most, the vLLM ones least — consistent with
llama.cpp's slot scheduling being less deterministic than vLLM's. The ranking
between models is stable under repetition, so it is signal, not noise.

## Failures and fixes

### Two client-side timing bugs, both found against a mock SSE server

A mock server with a known prefill delay and inter-token delay was used to
validate the client before any GPU time was spent. It caught two errors that
would have produced plausible but wrong numbers:

- **`requests.iter_lines()` buffers 512 bytes by default**, so the first SSE
  line only arrives after ~a dozen token deltas. That moves the first-token
  timestamp forward and shortens the decode window — exactly the two quantities
  being measured. Against a mock with a known 0.5 s prefill it reported 2.56 s.
  Fixed with `chunk_size=1`, which costs a few ms over a whole completion.
- **Returning early on `data: [DONE]`** abandons the response mid-body, so
  urllib3 drops the pooled connection and the *next* request pays a fresh TCP
  connect inside its TTFT. Fixed by draining the stream.

### `localhost` costs ~2 s per connection on Windows

Python resolves `localhost` to `::1` first, and every server here binds IPv4
only (llama.cpp and Ollama to 127.0.0.1, vLLM to 0.0.0.0, which is not a v6
address), so each connection spends ~2 s failing over — inside TTFT. Measured
against the mock: `localhost` gave 2.56 s TTFT / 482 prefill tok/s;
`127.0.0.1` gave 0.51 s / 2,410. **A five-fold error.** `pin_ipv4()` now
rewrites the host; `--keep-localhost` opts out.

### The bench sent raw video to a server configured for full decode

Cosmos's first run died with:

    HTTP 500: Unable to allocate 56.3 GiB for an array with shape
    (9725, 1080, 1920, 3) and data type uint8

The Cosmos, Qwen3-VL and Qwen3.8 services all run
`--media-io-kwargs '{"video": {"num_frames": -1}}'`, so the processor decodes
every frame in the file — 9,725 frames for the 389 s 1080p clip. The ranking
passes have always guarded this with an ffmpeg pre-shrink; the bench was written
without it. Fixed by importing `ensure_safe_video()` from
`rank.options.mc.cosmos.py` rather than reimplementing it, so the bench shrinks
exactly what a run shrinks, at the same adaptive fps, into the same on-disk
cache — which meant the 455 clips already transcoded for the September Cosmos
run were reused. `--raw-video` opts out.

**Also fixed: one failed request used to abort the whole leg.** A single 500
killed the run and recorded nothing. Failures are now captured per request,
printed as a `FAILED` row, written to the CSV with the error, excluded from the
medians, and counted in a `N of M request(s) FAILED` line so a partial result
cannot be mistaken for a clean one.

### Qwen3.8-27B: three attempts, three distinct causes

The model that served 7,329 calls over 7.53 h in September would not serve one
request now.

1. **FlashInfer attention JIT wants nvcc.** vLLM 0.27.1 logs
   `Using FLASHINFER attention backend out of potential backends:
   ['FLASHINFER', 'TRITON_ATTN']`, and FlashInfer's prefill wrapper
   JIT-compiles CUDA the first time it plans an attention call. No toolkit
   here, so `RuntimeError: Could not find nvcc` killed the EngineCore on
   request one and took the server with it; every later request got
   connection-refused. **This is a different trap from the FlashInfer
   *sampler*** — the log confirms `VLLM_USE_FLASHINFER_SAMPLER=0` was set and
   working.
2. **`VLLM_ATTENTION_BACKEND` no longer exists in 0.27.1.** Setting it is
   silently ignored; the log still said `Using FLASHINFER`. The env module
   exposes only `VLLM_FLASHINFER_ALLREDUCE_BACKEND`,
   `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION`, `VLLM_USE_RAY_V2_EXECUTOR_BACKEND`
   and `VLLM_VIDEO_LOADER_BACKEND`. The control is the **CLI flag
   `--attention-backend`** (`engine/arg_utils.py:696`).
3. **Triton does not support FP8 KV on sm_86.** With the flag honoured, the
   engine refused to start: *"FP8 KV cache is not supported by the Triton
   attention backend on NVIDIA GeForce RTX 3090 (compute capability 8.6);
   native FP8 (fp8e4nv) requires SM89+. Re-run with --kv-cache-dtype
   bfloat16."* vLLM had been auto-assigning FP8 KV, which FlashInfer supports
   on sm_86 and Triton does not.

`VQA/serve.vllm.sh` now pins both explicitly:
`--attention-backend TRITON_ATTN --kv-cache-dtype bfloat16`. That is more
robust than depending on vLLM's defaults, but it is **not the configuration the
September run used**, and it is slower (see *Caveats*).

## Caveats — what these numbers are not

**1. KV cache configuration is not held constant across models.**

| model | KV dtype | prefix caching | KV pool |
|---|---|---|---|
| Cosmos | `auto` (fp8) | on — 0.3–0.4% hit | 9.67 GiB |
| Qwen3-VL | `auto` (fp8) | on — 0.5–0.6% hit | 4.01 GiB |
| Gemma 4 | `auto` (fp8) | on — 0.0% hit | 12.52 GiB |
| Qwen3.8 | **bfloat16** (forced) | off | 3.18 GiB |
| InternVL | **q8_0** K and V | llama.cpp slot reuse | — |
| Muse | **f16** (default) | llama.cpp slot reuse | — |

Prefix caching *is* controlled for and verified near zero. KV **dtype** is not,
and prefill writes K/V as it goes, so it affects the rate. Two consequences:

- **Qwen3.8's row is not comparable to anything, including its own September
  self.** It was forced from fp8 to bf16 to satisfy Triton, on a 27B model.
- **InternVL vs Muse is an unfair pairing.** InternVL runs q8_0 KV, Muse f16.
  InternVL's 2.2× prefill lead was attributed to MoE-vs-dense earlier in the
  session; that overstates it. The MoE effect is real (3B active vs 30B dense)
  but the quantised KV cache contributes an unmeasured share.

**2. Cosmos's prompt tokens are a bench artifact.** The real pass sends
`video_mm_kwargs()` — `do_sample_frames: True`, `fps = adaptive_fps(duration)`
(a fixed ~64-frame budget) and an explicit `longest_edge`. The bench wires
`mm_processor_kwargs` for qwen3vl and qwen38 only, because Cosmos's budget is
computed per clip and the registry entry is static. So Cosmos's 10,790-token
median is an *unconstrained* variant; under production settings it would send
~64 frames and land far closer to 4–6k. **Its prefill rate is still valid** —
tokens per second does not care why there were many — but its prompt size, TTFT
and any wall-clock estimate derived from them are wrong for Cosmos.

**3. The bench never sends `response_format`.** The real passes constrain output
with a JSON schema. Without it Qwen3.8 reasons in plain `content` and truncates
at the budget on every query. This likely also degrades the reply quality
reported for VideoLLaMA3 and Cosmos.

**4. One unexplained anomaly.** Q_ID 1254 (`1qgVLziQsVE.mp4`, 480×480, 14.7 s,
439 frames) returned **exactly 3,909 prompt tokens on both Cosmos and
Qwen3-VL** — the lowest of any clip for both, despite having the most frames.
Two different processors with different budgets producing an identical count
points at the video contributing nothing on both, i.e. a silent attach failure
rather than a small payload. LLaVA-Video got 6,683 for the same query, so it did
attach there. **Per-clip Cosmos and Qwen3-VL numbers for this query should not
be trusted until this is chased.**

**5. Single-stream only.** Everything here is `--concurrency 1`. The real passes
run 4–8 workers, where per-request rate falls and aggregate throughput rises.
`--concurrency N` reports the aggregate but was not used in this sweep.

## Open items

1. **Fix Cosmos's visual budget** to mirror `video_mm_kwargs()` and re-run that
   leg (~2 min). Until then its prompt/TTFT columns are not production-faithful.
2. **Chase the 3,909-token anomaly** on Q_ID 1254.
3. **Send `response_format`** so replies are schema-constrained as in the runs.
4. **Level the KV playing field**: re-run Muse with `--cache-type-k q8_0
   --cache-type-v q8_0` to match InternVL (~6 min).
5. **Get a CUDA toolkit into WSL2** so FlashInfer can JIT. That restores
   Qwen3.8's September configuration (FlashInfer + fp8 KV) and would let
   `--enforce-eager` be dropped fleet-wide — the single biggest lever on the
   weak decode numbers.
6. **Measure at concurrency 4–8**, which is what the batch runs actually see.

## How to re-run

    # everything, one model on the card at a time
    powershell -ExecutionPolicy Bypass -File E:\TREC_2026\VQA\run.bench.tokens.ps1 -Task 1
    powershell -ExecutionPolicy Bypass -File E:\TREC_2026\VQA\run.bench.tokens.ps1 -Task 2

    # one model, or other knobs
    ... -Models internvl,muse      # subset (internvl and muse need no WSL)
    ... -Spread corpus             # the corpus's own duration mix
    ... -Concurrency 4             # aggregate throughput
    ... -ForceLength 256           # pin every completion, exact decode comparison

    # inspect without touching the GPU
    E:\TREC_2026\.venv\Scripts\python.exe VQA\bench.tokens.py --model gemma4 --task 1 --dry-run
    E:\TREC_2026\.venv\Scripts\python.exe VQA\bench.tokens.py --list
    E:\TREC_2026\.venv\Scripts\python.exe VQA\bench.tokens.py --serve-cmd qwen38
    E:\TREC_2026\.venv\Scripts\python.exe VQA\bench.tokens.py --report

    # WSL2 side, by hand
    wsl -d Ubuntu-24.04 -- bash /mnt/e/TREC_2026/VQA/serve.vllm.sh qwen3vl
    wsl -d Ubuntu-24.04 -- bash /mnt/e/TREC_2026/VQA/bench.tokens.sh qwen3vl --task 2

The sample is deterministic given `--videos` and `--spread`, so every model and
every re-run sees the same clips.

## Timeline

| time (IST) | event |
|---|---|
| ~11:00 | bench client written; validated against a mock SSE server; two timing bugs and the localhost/IPv6 trap found and fixed |
| 12:15–12:31 | Task 1: InternVL (4.5 min), Muse (6 min), VideoLLaMA3 (4.6 min, 174 s load) |
| 12:43 | Cosmos Task 1 fails — 56.3 GiB video decode; `ensure_safe_video()` guard added, per-request error handling added |
| 12:45–12:55 | Task 1: Cosmos, Qwen3-VL, LLaVA-Video, Gemma 4 |
| 12:55–13:20 | Qwen3.8 fails three times (FlashInfer JIT → dead env var → FP8/Triton) |
| 13:27 | Qwen3.8 Task 1 completes on the Triton/bf16 fallback |
| 13:29–13:58 | Task 2 sweep, all eight models, 29 min, 0 failures |

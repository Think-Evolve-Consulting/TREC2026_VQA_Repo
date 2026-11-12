# Qwen3-VL-8B-Thinking VQA Pipeline

How to serve and run the third Task 1 answering pass,
`VQA/generate.answers.qwen3vl.py`. Companion to the Gemma 4/Ollama pipeline
(`generate.answers.py`, 30 calls per query over extracted frames) and the
Cosmos-Reason2-2B pipeline (`generate.answers.cosmos.py`, one call per query
over the native video) — same query/transcript inputs, same run-file output,
same cache/resume contract, so the three runs are directly comparable.

**Status: running.** The server config below is the one that actually starts on
this 3090 — the documented-obvious one does not, see §3.1 — and 8 real queries
have been answered through it end to end with no failures, no degrades and no
fallback padding. Measured throughput is in §6.

## Architecture

- **Model**: `Qwen/Qwen3-VL-8B-Thinking` — 8.77B params, bf16, ~17.5 GB,
  `Qwen3VLForConditionalGeneration`, 256K native context. Takes video
  natively, so no frame extraction is needed.
- **Server**: vLLM, OpenAI-compatible, inside WSL2 (`Ubuntu-24.04`). The
  existing `/home/aakash/cosmos-reason2/.venv` already has **vLLM 0.12.0,
  torch 2.9.0+cu128, transformers 4.57.1**, and that vLLM registers
  `Qwen3VLForConditionalGeneration` — Cosmos-Reason2 *is* a Qwen3-VL
  derivative, so the same environment serves this with no new install.
- **Client**: `VQA/generate.answers.qwen3vl.py`, also run from WSL2. The
  server opens the video off the path in the request, so the path has to be
  one the server can read; `/mnt/e/TREC_2026/...` is the same file as
  `E:\TREC_2026\...`. The script warns if it is started on Windows.
- **Port**: 8009, so it does not collide with the Cosmos service on 8008.
  They cannot share the GPU — **stop Cosmos first** (see below).

## 1. Free the GPU

The Cosmos service holds ~15 GB whenever WSL2 is up, and it starts on its own
with the distro's systemd:

```bash
wsl -d Ubuntu-24.04
systemctl --user stop cosmos-reason2
nvidia-smi          # expect the 3090 back at ~0 MiB used
```

The Cosmos run is finished and validated (`cosmos.run.task1.json`, 2,443/2,443),
so stopping it costs nothing.

## 2. Get the checkpoint

```bash
/home/aakash/cosmos-reason2/.venv/bin/hf download Qwen/Qwen3-VL-8B-Thinking
```

~17.5 GB into `~/.cache/huggingface/hub`; WSL2's disk had 668 GB free.

### If bf16 will not fit

17.5 GB of weights on a 24 GB card leaves roughly 4–5 GB for KV cache and
activations, which is enough for `--max-model-len 24576` and not much more.
Two fallbacks, in order:

| Checkpoint | Size | Note |
|---|---|---|
| `Qwen/Qwen3-VL-8B-Thinking-FP8` | ~9 GB | Official. FP8 W8A8 needs sm_89+; the 3090 is **sm_86**, so vLLM runs it W8A16 through its FP8-Marlin path. Works, but confirm at startup rather than assuming. |
| `cyankiwi/Qwen3-VL-8B-Thinking-AWQ-4bit` | ~6 GB | INT4 AWQ, native on Ampere, plenty of KV headroom. **Community-produced, not Qwen's** — a quantisation nobody here has evaluated. |

Nothing in the client depends on which one is loaded: it sizes the frame
budget from whatever `max_model_len` the server reports.

## 3. Serve

```bash
/home/aakash/cosmos-reason2/.venv/bin/vllm serve Qwen/Qwen3-VL-8B-Thinking \
  --host 0.0.0.0 --port 8009 \
  --served-model-name Qwen/Qwen3-VL-8B-Thinking \
  --allowed-local-media-path /mnt/e/TREC_2026 \
  --max-model-len 24576 \
  --gpu-memory-utilization 0.92 \
  --enforce-eager \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3 \
  --skip-mm-profiling
```

Measured on this box:

```
Model loading took 16.6402 GiB memory
Skipping memory profiling for multimodal encoder and encoder cache.
Available KV cache memory: 4.01 GiB
GPU KV cache size: 29,168 tokens
Peak VRAM during an 8-query run: 22,998 MiB of 24,576
```

- **`--skip-mm-profiling`** — **required, and the whole reason this starts at
  all.** See §3.1.
- **`--gpu-memory-utilization 0.92`** — not higher. Windows permanently holds
  ~1.3 GiB of the card, so only 22.72 of 24.0 GiB is free to CUDA; 0.95 is
  rejected outright at startup with *"Free memory on device (22.72/24.0 GiB)
  on startup is less than desired GPU memory utilization"*.
- **`--allowed-local-media-path`** — without it every `file://` video is
  refused and all 2,427 queries with a video answer blind. `preflight()`
  checks this before the run rather than after it.
- **`--max-model-len 24576`** — sized against the weights, not the model's
  256K window. The client reads it and divides the visual budget by the frames
  its fps will produce, so raising it raises per-frame resolution
  automatically.
- **`--enforce-eager`** — skips CUDA-graph capture to save VRAM, small latency
  cost. There is no room to drop it.
- **`--media-io-kwargs num_frames=-1`** — required by this model family for
  correct per-frame timestamp embedding, and the reason the client pre-shrinks
  a minority of long/high-res clips (see below).
- **`--reasoning-parser qwen3`** — splits `<think>…</think>` into
  `reasoning_content` so the answer JSON stays clean in `content`. The
  `-Thinking` checkpoint has no non-thinking mode; its chat template opens the
  block for it. Without this flag the client still parses correctly (it
  tolerates prose around the JSON), but the reasoning trace counts as content.

### 3.1 Why it will not start without `--skip-mm-profiling`

Without it the engine dies at startup with a traceback whose real cause is
several screens above the visible error:

```
Model loading took 16.6402 GiB memory
Encoder cache will be initialized with a budget of 153600 tokens,
  and profiled with 1 video items of the maximum feature size.
Available KV cache memory: -1.20 GiB
ValueError: No available memory for the cache blocks.
```

vLLM profiles a dummy video at **Qwen3-VL's maximum feature size** — 153,600
encoder tokens, hundreds of frames — even though nothing in this pipeline ever
sends more than 96 frames. Weights (16.64 GiB) plus that profiling peak
overshoot the 22.6 GiB budget by 1.2 GiB, so the KV cache gets nothing and the
engine refuses to start.

Things that were tried and **do not** move it, all measured:

| Attempt | Result |
|---|---|
| `--limit-mm-per-prompt '{"video": {"count": 1, "num_frames": 96}}'` | No effect — budget still 153,600, KV still −1.20 GiB |
| `--mm-processor-kwargs '{"max_frames": 96}'`, `{"max_frames": 32}`, custom `size` | No effect — computed 153,600 under every variant |
| `--max-model-len 16384` | No effect on the encoder budget |
| `--gpu-memory-utilization 0.95` | Rejected before profiling: only 22.72 GiB is free to CUDA |

153,600 is fixed for Qwen3-VL in vLLM 0.12 regardless of processor kwargs or
window size, so capping the *profile* is the only lever left.

**The risk this accepts:** nothing then reserves memory for the encoder cache,
so an unusually large request could OOM at run time rather than at startup.
Measured against it: peak VRAM over an 8-query run including a 320 s clip was
22,998 MiB with **zero** `out of memory` events in the server log. The client
also caps its own worst case — 96 frames, per-frame pixels divided by the
window — and degrades on a 400 rather than dying. **If you raise
`DENSE_TARGET_FRAMES` in the client, this safety margin is what you are
spending.**

Sanity check:

```bash
curl -s http://localhost:8009/v1/models
```

If it is worth making persistent, copy
`~/.config/systemd/user/cosmos-reason2.service` and change the model, the port
and the unit name.

## 4. Run the client

```bash
cd /mnt/e/TREC_2026
/home/aakash/cosmos-reason2/.venv/bin/python VQA/generate.answers.qwen3vl.py --limit 20
```

| Goal | Command |
| --- | --- |
| Smoke test, and the timings that decide the full run | `... --limit 20` |
| Full run, resumable | `...` (no flags) |
| The no-routing ablation | `... --ignore-routing` |
| Retry specific queries | `... --only 9 20 26 --force` |
| Re-emit the run file from cache, no inference | `... --rebuild` |
| Cheaper/faster pass | `... --budget brief` |
| More concurrency | `... --workers 3` |

`Ctrl+C` is safe at any point — each query is flushed to
`answers.task1.qwen3vl.cache.jsonl` as it finishes and a re-run resumes.
Output: `E:\TREC_2026\qwen3vl.run.task1.json`.

Validate exactly as the other two runs:

```bash
/home/aakash/cosmos-reason2/.venv/bin/python VQA/validate.ag.run.py \
  vqa.2026.testingDataset.Task1.json qwen3vl.run.task1.json \
  --warnings-as-errors --report-json qwen3vl.run.validation.json
```

## 5. What this pass does differently

### It actually sends the YouTube videos

`Video_ID` is not a filename. 868 of the 2,443 queries name a bare YouTube ID,
and `download_test_videos.py` wrote those as `<Video_ID>.mp4`
(`SESSION.2026-08-11.task1-pipeline.md` §3). `generate.answers.cosmos.py`
resolves `videos_dir / Video_ID` literally, so it never opened one of the 852
downloaded mp4s — its cache records **1,098 of 2,443 queries answered with no
video at all, 469 of them with no transcript either**, where only **16**
queries genuinely have no media on disk. This script applies the documented
convention, so 2,427 queries get their video.

That is worth knowing about `cosmos.run.task1.json` independently of this
pipeline: it is a valid run, but 45% of it was answered blind.

### The router decides what each query is sent

First pass here to use `query.routing.task1.json` (the design comes from the
Muse Glimmer scripts, which never ran). `--ignore-routing` turns it off and
sends everything to everything, which is the ablation the routed run has to
beat.

| Routing step | Effect | Queries |
|---|---|---|
| `frames` | attach the video | 2,259 |
| `asr_transcript` | attach the Whisper transcript | 576 |
| `dense_timestamped_frames` | frame budget 64 → 96 | 886 |
| `frame_ocr` | per-frame pixel budget 768 → 1024 tokens, plus a read-the-screen rule | 57 |

Generation budget from the router's `Inference`/`Temporal` flags, because a
thinking model's cost *is* its decode:

| `(inference, temporal)` | budget | max tokens | queries |
|---|---|---|---|
| neither | brief | 4,096 | 721 |
| one of the two | standard | 8,192 | 1,422 |
| both | deep | 12,288 | 300 |

**No query is ever sent with zero evidence.** The 168 queries routed to the
audio alone get the video back when that video turns out to have no
transcript, and any query whose video the ladder has to drop gets the
transcript put in even if the router had routed it away.

### The frame budget is computed against the context window

vLLM reads `size.longest_edge` as a *per-frame* pixel ceiling
(`qwen3_vl.py`: `max_pixels=image_processor.size["longest_edge"]`), so what
overflows the window is frames × tokens-per-frame, not fps alone. The client
picks an fps targeting 64 frames (96 when dense), predicts how many frames
that yields, and divides the visual budget by it. `fps` and `num_frames` are
mutually exclusive in the processor — it raises — so only `fps` is ever sent.

### Every request degrades rather than failing

A 400 walks down a ladder and the rung reached is recorded per query:

| Rung | Trigger | Action |
|---|---|---|
| `shrink1`, `shrink2` | 400 naming the context window | halve fps and pixels, retry |
| `no_dsf` | any other 400 | drop `do_sample_frames`/`fps`, **keep** the pixel budget |
| `no_mm_kwargs` | still failing | drop `mm_processor_kwargs`, let the server default |
| `no_video` | still failing | answer from transcript + question |

### The `do_sample_frames` trap

`do_sample_frames: true` makes the processor re-sample the decoded video at the
requested fps, and **on a clip too short to yield many frames that request
fails outright** with `Failed to apply Qwen3VLProcessor`. The boundary was
measured against this corpus and is exact: every video whose request worked out
to ≤11 frames failed, every one at ≥12 succeeded (1.07 s, 1.67 s, 1.90 s,
2.74 s all FAIL; 3.04 s and up all OK, at 4 fps). The client now sends
`do_sample_frames`/`fps` only above `MIN_RESAMPLE_FRAMES` (16, for margin) and
sends `size` alone below it — which costs nothing, because a clip that short is
only a handful of frames at the server's own rate anyway.

Two related facts, both measured, worth not rediscovering:

- **`fps` is only honoured when `do_sample_frames` is set.** Sending `fps`
  alone is a no-op: identical prompt-token counts to sending nothing (3,665 on
  a 1 s clip, 12,579 on a 389 s one).
- **`size` is what actually bounds the cost.** On the longest clip in the
  corpus it takes the prompt from 12,579 tokens to 579.

**This is why a lone probe is not enough.** An early full-run attempt printed
*"this build rejects the mm_processor_kwargs shape"* and ran with the frame
budget disabled — because preflight probed exactly one video, and the shortest
clip in the corpus (1.07 s) is precisely the case that fails. Prompt sizes went
from ~500 tokens to a median of **12,014** on short clips, and anything longer
would have overflowed the window and been answered with no video at all.
Preflight now probes the median clip as well as the shortest and only accepts a
verdict both agree on.

`no_video_file` in the report is the separate case of the 16 queries whose
video was never downloaded.

`preflight()` settles the two things that cannot be known client-side —
whether this vLLM build accepts the `mm_processor_kwargs` shape, and (with
`--json-schema`) whether its structured-output path exempts the reasoning
block — against the live server before the run starts, and turns itself off
gracefully rather than failing 400 queries in.

### Long videos are pre-shrunk, as in the Cosmos pass

`num_frames=-1` makes vLLM decode every native-resolution frame into memory
*before* fps subsampling, so a 320 s 2048×1152 clip tried to allocate 50.6 GB
and crashed the request. Any video whose `duration × fps × w × h × 3` exceeds
4 GiB is pre-shrunk with ffmpeg into
`videos_TEST_VQA/videos/.vlm_transcoded_cache/`. The 196 clips the Cosmos run
already shrank are **reused** when their frame rate is high enough for what
this pass wants, rather than re-encoded. `--no-transcode` disables it,
`--decode-limit-gib` moves the threshold.

### `Time`

The track defines it as "time to generate this answer", per answer, in
seconds. The call's wall clock is split across the answers that call actually
produced; a slot filled from the generic fallback pool carries only its share
of any ffmpeg pre-shrink this query paid for, usually 0. Summed over a query,
`Time` is that query's inference wall clock. Model load and the offline ASR
pass are excluded, as in the other two pipelines. The end-of-run report prints
mean/min/max and a count of zero-valued entries so this is checkable.

## 6. What to expect, and what is not known

**Throughput, measured over the same 8 queries at three concurrencies:**

| `--workers` | 8 queries | Projected 2,443 | Peak decode | Preemption / OOM |
|---|---|---|---|---|
| 1 | 280 s | 23.7 h | 38 tok/s | 0 |
| **4 (default)** | **133 s** | **11.0 h** | 310 tok/s | 0 |
| 8 | 143 s | 12.0 h | — | 0 |

Decode is the whole cost — mean **1,340 output tokens** of reasoning per query
— and at batch 1 the 3090 is mostly idle, which is why 4 workers costs nothing
and halves the run. 8 gives it back. Ignore the startup line *"Maximum
concurrency for 24,576 tokens per request: 1.19x"*: it assumes every request
fills the window, while real ones here are 600–1,450 prompt tokens.

**Treat 11 h as ±30%.** The sample is 8 queries, all short clips from the head
of the file; the full set has a long tail of 100–389 s videos whose prefill is
heavier, and at temperature 1.0 the trace length itself varies (per-tier means
moved 1,100→2,100 tokens between two runs of the *same* queries). The progress
line prints a self-correcting ETA from the queries actually done.

The generation budgets are **not** binding — every one of the 8 finished with
`finish_reason: stop`, well inside its tier — so lowering `--max-tokens` will
not speed things up. If more speed is needed the lever is `--budget brief`
(which shortens the reasoning the model chooses to do), not a smaller ceiling.

**What the 8-query run actually produced** (`--workers 1`, full detail in the
session log): 0 failures, 0 degrades, 0 fallback padding, 0 truncations, 0
unparsed replies, all 8 with the video attached. Answers read sensibly — the
320 s `ChainedforLife-14` clip came back *"Film rehearsal for dramatic
character entrance"*, a counting question came back `0`. Mean answer length
**3–4 words** (3.16 and 3.86 across two runs of the same 8 queries, since
temperature is 1.0), against a ground-truth mean of 8.4 — but that 8.4 is from
the **2025** `vqa.gt.csv`, which shares no videos with this year's test set, and
no answer in this project has ever actually been scored. Terseness costing
METEOR recall is a plausible mechanism, not a finding.

**Risks worth watching on the first 20:**

- `TRUNCATED` in the progress line means `finish_reason == "length"` — the
  trace ran out of budget before the JSON. A few is fine; many means the
  budgets are too tight, which is the failure Cosmos hit at 3,072.
- `UNPARSED` means no JSON came back at all. The answers are then pure
  fallback padding and the query is wasted.
- `SHRINK2` / `NO_MM_KWARGS` on many queries means the budget arithmetic is
  wrong for this server config; raise `--max-model-len` or lower
  `--max-tokens`.
- The report's "answered with the video attached" should be ~99% of the run.
  If it is not, the media path or `--allowed-local-media-path` is wrong.

**Not measured here, as with the other two passes:** accuracy. Everything
above is format correctness, evidence coverage and plausibility. Scoring
against `VQA/vqa.gt.csv` needs the 2025 videos, `nltk` and `bert_score`, none
of which are on this box.

## 7. Testing that was done

**Against the real model**: 8 queries end to end at 1, 4 and 8 workers (24
answered queries in total), covering a 320 s clip that needed the ffmpeg
pre-shrink, a counting question, dense-routed and transcript-routed queries.
No failures, no degrades, no OOM, peak VRAM 22,998 MiB of 24,576.

**Against a mock vLLM server** that records every request — the paths the real
run has not yet exercised:

- routed plans, budgets and grouping over real queries, videos and transcripts;
- `mm_processor_kwargs` shape, fps and per-frame pixel arithmetic;
- the full degrade ladder — context-overflow shrink, processor rejection, and
  the drop to transcript-only, with the transcript confirmed present in the
  degraded prompt;
- preflight self-correction when the server rejects the kwargs or the JSON
  schema, and preflight *not* aborting a run over a context-budget miss;
- prose wrapped around the answer JSON, short answer lists padded to ten,
  resume, `--rebuild`, `--force`, `--only`, `--ignore-routing`;
- `validate.ag.run.py --warnings-as-errors` passing on the emitted run.

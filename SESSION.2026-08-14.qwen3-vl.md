# Session Notes — 2026-08-14 — Task 1 answer generation with Qwen3-VL-8B-Thinking

Built the third Task 1 answering pass, after Gemma 4 (`generate.answers.py`,
30 calls per query over extracted frames) and Cosmos-Reason2-2B
(`generate.answers.cosmos.py`, one call per query over the native video).
This one runs **`Qwen/Qwen3-VL-8B-Thinking`** on vLLM, one reasoning call per
query, with request contents driven by `query.routing.task1.json` — the first
answering pass here that actually uses the routing file.

Unlike the Muse Glimmer attempt earlier the same day
(`SESSION.2026-08-14.muse-glimmer.md`), **this one runs.** The server config
that fits a 24 GB 3090 was found by measurement, 24 real queries were answered
end to end, and the full run was measured at **~11 h** at 4 workers.

**Deadline context:** Answer Generation runs are due **17 August 2026**.

---

## Files created this session

| File | Purpose |
|---|---|
| `VQA/generate.answers.qwen3vl.py` | The answering pass — routing, prompts, degrade ladder, preflight, driver |
| `VQA/QWEN3_VL_SETUP.md` | Server setup, the flags and why, run commands, measured throughput |
| `answers.task1.qwen3vl.cache.jsonl` | Resumable work cache, one line per answered Q_ID |
| `qwen3vl.run.task1.json` | The run file (written incrementally; full run not yet completed) |

Pre-existing inputs consumed: `vqa.2026.testingDataset.Task1.json`,
`query.routing.task1.json`, `videos_TEST_VQA/videos/`,
`videos_TEST_VQA/transcription_openwhisper/`,
`videos_TEST_VQA/frames/index.json` (read **only** for its probed durations —
this model takes video, not frames).

---

## 1. A bug in the Cosmos run, found while writing this one

**`Video_ID` is not a filename.** 868 of the 2,443 queries name a bare YouTube
ID, and `download_test_videos.py` wrote those as `<Video_ID>.mp4` beside the
released `.webm` clips (`SESSION.2026-08-11.task1-pipeline.md` §3).
`generate.answers.cosmos.py:530` resolves `videos_dir / video_id` literally, so
it never opened one of the 852 downloaded mp4s.

Counted from `answers.task1.cosmos.cache.jsonl` itself:

| | Queries |
|---|---|
| Answered with the video attached | 1,348 |
| **Answered with no video at all** | **1,098** |
| …of those, with no transcript either (question text only) | **469** |
| Queries that genuinely have no media on disk | **16** |

`cosmos.run.task1.json` is a valid, validated run, but **45% of it was answered
blind**. Relevant to any comparison against it, and to whether it is worth
re-running now that the naming is understood.

The new script applies the documented convention, so 2,427 of 2,443 queries get
their video:

```python
name = video_id if video_id.endswith(".webm") else video_id + ".mp4"
```

---

## 2. What Qwen3-VL-8B-Thinking is, and what it forced on the design

Checked against the model card and the *installed* vLLM/transformers source
before writing anything:

- **8.77B params, bf16, ~17.5 GB, 256K native context**,
  `Qwen3VLForConditionalGeneration`. **Takes video natively** — no frame
  extraction needed, unlike Gemma 4 and Muse Glimmer.
- **The existing WSL2 environment already serves it.** `/home/aakash/
  cosmos-reason2/.venv` has **vLLM 0.12.0, torch 2.9.0+cu128, transformers
  4.57.1, openai 2.8.1**, and that vLLM registers
  `Qwen3VLForConditionalGeneration` — Cosmos-Reason2 *is* a Qwen3-VL
  derivative. No new install, no new venv.
- **The `-Thinking` checkpoint has no non-thinking mode**; its chat template
  opens the `<think>` block for it. `--reasoning-parser qwen3` splits the trace
  into `reasoning_content` so the answer JSON stays clean in `content`.
- **Sampling per the model card, vision-language mode**: temperature 1.0,
  top_p 0.95, top_k 20, presence_penalty 0.0. Deliberately *not* the lowered
  0.6 that `generate.answers.py` uses for Gemma 4 — that number exists to
  concentrate a 30-sample vote, and there is no vote here. This is one call
  whose ten answers must be genuinely different, the opposite requirement.

### Video geometry, read off the installed source rather than assumed

| Fact | Where confirmed |
|---|---|
| `size.longest_edge` is a **per-frame** pixel ceiling | `vllm/model_executor/models/qwen3_vl.py`: `max_pixels=image_processor.size["longest_edge"]` |
| A visual token is `(patch 16 × merge 2)² = 1024` px | processor config |
| Frames are merged in **pairs** (`temporal_patch_size 2`) | processor config |
| `fps` and `num_frames` are **mutually exclusive** — the processor raises | `Qwen3VLVideoProcessor.sample_frames` |
| Per-frame default ceiling 768 tokens, floor 128 | `size` = `{shortest_edge: 131072, longest_edge: 786432}` |

So what overflows the context window is **frames × tokens-per-frame**, not fps
alone. `video_mm_kwargs()` divides the visual budget by the frames the chosen
fps will actually produce.

---

## 3. Getting it to serve — three separate obstacles

The GPU was completely free (0 MiB, Cosmos service inactive) and host RAM was
never the problem, despite the error reading like an OOM.

### 3.1 The startup failure

```
Model loading took 16.6402 GiB memory
Encoder cache will be initialized with a budget of 153600 tokens,
  and profiled with 1 video items of the maximum feature size.
Available KV cache memory: -1.20 GiB
ValueError: No available memory for the cache blocks.
```

vLLM profiles a dummy video at **Qwen3-VL's maximum feature size** — 153,600
encoder tokens, hundreds of frames — although nothing in this pipeline sends
more than 96. Weights plus that profiling peak overshoot the budget by
1.2 GiB, so KV gets nothing.

Everything tried that **does not** move it, each measured:

| Attempt | Result |
|---|---|
| `--limit-mm-per-prompt '{"video": {"count": 1, "num_frames": 96}}'` | No effect — still 153,600, KV still exactly −1.20 GiB |
| `--mm-processor-kwargs '{"max_frames": 96 / 64 / 32}'`, custom `size` | No effect — computed 153,600 under every variant |
| `--max-model-len 16384` | No effect on the encoder budget |

153,600 is fixed for this architecture in vLLM 0.12. **`--skip-mm-profiling`
is the only lever**, and with it:

```
Available KV cache memory: 4.01 GiB
GPU KV cache size: 29,168 tokens
Maximum concurrency for 24,576 tokens per request: 1.19x
```

### 3.2 Windows permanently holds ~1.3 GiB of the card

```
ValueError: Free memory on device (22.72/24.0 GiB) on startup is less than
desired GPU memory utilization (0.95, 22.8 GiB).
```

Only **22.72 of 24.0 GiB** is free to CUDA, so `--gpu-memory-utilization` must
stay **≤ 0.92**. `nvidia-smi` reporting 24,326 MiB free is optimistic about
what CUDA actually gets.

### 3.3 The risk `--skip-mm-profiling` accepts, and the measurement against it

Nothing then reserves memory for the encoder cache, so an outsized request
could OOM at *run* time instead of at startup. Measured over 24 real queries
including a 320 s clip: peak VRAM **22,998 MiB of 24,576**, **zero** `out of
memory` events, zero preemptions. The client also caps its own worst case
(96 frames; per-frame pixels divided by the window) and degrades on a 400.

**If `DENSE_TARGET_FRAMES` is ever raised in the client, that margin is what is
being spent.**

### The working command

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

Port 8009, not Cosmos's 8008: they cannot share the card, but they can share a
shell history without one silently answering the other's requests.

---

## 4. The routed design

`Pipeline` from `query.routing.task1.json` maps onto request contents. The
design is lifted from the Muse Glimmer scripts, which never ran.
`--ignore-routing` sends everything to everything at a fixed budget — the
ablation the routed run has to beat.

| Routing step | Effect | Queries |
|---|---|---|
| `frames` | attach the video | 2,259 |
| `asr_transcript` | attach the Whisper transcript | 576 |
| `dense_timestamped_frames` | frame budget 64 → 96 | 886 |
| `frame_ocr` | per-frame budget 768 → 1024 tokens, plus a read-the-screen rule | 57 |

Generation budget from `Inference`/`Temporal`, because a thinking model's cost
*is* its decode:

| `(inference, temporal)` | budget | max tokens | queries |
|---|---|---|---|
| neither | brief | 4,096 | 721 |
| one of the two | standard | 8,192 | 1,422 |
| both | deep | 12,288 | 300 |

As the client reports it at startup:

```
to answer: 2443 queries over 2341 groups | budgets brief 721 standard 1422 deep 300
  video attached 2338 | dense 886 | transcript 576 | ocr 57 | no video file 16
```

Design points worth keeping:

- **No query is ever sent with zero evidence.** The 168 queries routed to the
  audio alone get the video back when that video has no transcript, and any
  query whose video the ladder drops gets the transcript put in *even if the
  router routed it away*. Verified on a real degraded prompt.
- **The transcript is always read, not always sent.** A vision-routed query is
  not told "the video has no detected speech" — it was never looked for, and
  1,941 videos have one on disk.
- **Work is grouped by (video, plan shape)** so queries sharing a prompt prefix
  run back to back: 2,341 groups over 2,443 queries.
- **Durations come from `frames/index.json`**, which has a probed duration for
  all 1,940 clips. The Cosmos pass took duration from the transcript, which
  does not exist for 1,080 queries, so those all silently planned at a flat
  2 fps.

---

## 5. The `do_sample_frames` trap — found mid-run, in production

The first full-run launch printed:

```
preflight:
  video+kwargs    FAILED: Error code: 400 - Failed to apply Qwen3VLProcessor ...
  video only      ok (14.9s)
  -> this build rejects the mm_processor_kwargs shape; running with the
     server's own video defaults ...
```

and carried on with **the frame budget disabled for all 2,443 queries**. The
four records it produced before it was stopped show the cost: median
**12,014 prompt tokens**, against ~500 for the same clips with the budget
applied. Anything past ~60 s would have overflowed the window entirely and been
answered with no video.

**Root cause: preflight probed exactly one video — the shortest in the corpus.**
`do_sample_frames: true` re-samples the decoded video at the requested fps, and
on a clip too short to yield many frames the processor rejects the request
outright. The boundary is exact and sharp:

| Clip | Expected frames | With `do_sample_frames` | Without |
|---|---|---|---|
| `6666.webm` 1.07 s | 4 | **FAIL** | OK, 429 tokens |
| `7317.webm` 1.67 s | 7 | **FAIL** | OK, 513 |
| `6738.webm` 1.90 s | 8 | **FAIL** | OK, 573 |
| `6809.webm` 2.74 s | 11 | **FAIL** | OK, 559 |
| `7864.webm` 3.04 s | 12 | OK | OK |
| `86brftxNoAc` 11.1 s (median) | 44 | OK | OK |
| `Archipelago-22` 389 s (longest) | 96 | OK | OK |

Two more facts from the same probe, both worth not rediscovering:

- **`fps` is only honoured when `do_sample_frames` is set.** Sending `fps`
  alone is a measurable no-op — identical prompt-token counts to sending
  nothing (3,665 on a 1 s clip; 12,579 on the 389 s one).
- **`size` is what actually bounds the cost** — 12,579 → 579 tokens on the
  longest clip in the corpus.

### Three fixes

1. `do_sample_frames`/`fps` are sent only above `MIN_RESAMPLE_FRAMES` (16, for
   margin over the observed 12); below that `size` alone, which costs nothing
   because such a clip is a handful of frames at the server's own rate anyway.
2. A new degrade rung **`no_dsf`** — on a processor 400 it drops
   `do_sample_frames`/`fps` but **keeps** the pixel budget, instead of
   surrendering the whole block and handing a long video back to the 768-frame
   default.
3. **Preflight probes the median clip as well as the shortest** and only
   accepts a verdict both agree on. A single clip failing is a property of that
   clip, which the per-request ladder handles.

After the fix, all six probe clips pass, including the four that failed.

---

## 6. Degrade ladder and preflight

Every request walks a ladder rather than failing, and the rung reached is
recorded per query:

| Rung | Trigger | Action |
|---|---|---|
| `shrink1`, `shrink2` | 400 naming the context window | halve fps and pixels, retry |
| `no_dsf` | any other 400 | drop `do_sample_frames`/`fps`, keep the pixel budget |
| `no_mm_kwargs` | still failing | drop `mm_processor_kwargs` entirely |
| `no_video` | still failing | answer from transcript + question |

`no_video_file` (16 queries) and `routed_text_only` are recorded separately —
they are text-only by design, not degraded requests.

`preflight()` settles, against the live server before the run starts, the two
things that cannot be known client-side: whether this build accepts the
`mm_processor_kwargs` shape, and (with `--json-schema`) whether its
structured-output path exempts the reasoning block. A context-budget miss on
the probe is **not** treated as a server capability failure — it warns and
continues, because the run-time ladder handles it.

---

## 7. Throughput — 4 workers halves the run

Measured over the same 8 queries at three concurrencies, against the config in
§3:

| `--workers` | 8 queries | Projected 2,443 | Peak decode | Preemption / OOM |
|---|---|---|---|---|
| 1 | 280 s | 23.7 h | 38 tok/s | 0 |
| **4 (new default)** | **133 s** | **11.0 h** | 310 tok/s | 0 |
| 8 | 143 s | 12.0 h | — | 0 |

Decode is the whole cost — mean **1,340 output tokens** of reasoning per query
— and at batch 1 the 3090 is mostly idle, which is why 4 workers costs nothing
and halves the run. 8 gives it back.

**Ignore the startup line "Maximum concurrency for 24,576 tokens per request:
1.19x."** It assumes every request fills the window; real ones here are
600–1,450 prompt tokens, and 8 concurrent requests ran with zero preemption and
no change in peak VRAM.

The generation budgets are **not** binding — all 8 finished with
`finish_reason: stop`, well inside their tiers — so lowering `--max-tokens`
will not speed anything up. The lever for more speed is `--budget brief`.

**Treat 11 h as ±30%:** the sample is 8 short clips from the head of the file,
the corpus has a long tail of 100–389 s videos with heavier prefill, and at
temperature 1.0 the trace length varies (per-tier means moved 1,100 → 2,100
tokens between two runs of the *same* queries).

---

## 8. Media preparation

`--media-io-kwargs num_frames=-1` (required for this family's timestamp
correctness) makes vLLM decode every native-resolution frame *before* fps
subsampling, so a long/high-res clip allocates gigabytes up front — the 50.6 GB
crash documented in `VQA/COSMOS_REASON2_SETUP.md`. Any video whose
`duration × fps × w × h × 3` exceeds 4 GiB is pre-shrunk with ffmpeg into
`videos_TEST_VQA/videos/.vlm_transcoded_cache/`.

**The 196 clips the Cosmos run already shrank are reused** when their frame
rate is at or above what this pass needs — the cheapest sufficient copy wins,
found by globbing `<stem>.f*.mp4` across both cache directories. Confirmed in a
real run: the 320 s `ChainedforLife-14` was served from the Cosmos cache's
`ChainedforLife-14.f0.2.mp4` at exactly the rate needed, with no re-encode.

`ffprobe` results are memoised per video (the Cosmos pass re-probed once per
query, ~500 wasted calls over 1,956 videos serving 2,443 queries).

---

## 9. `Time`, and the run format

The track defines `Time` as "time to generate this answer", per answer, in
seconds. Reported here:

| Answer source | `Time` |
|---|---|
| Model answer | the call's wall clock ÷ the answers that call actually produced |
| Generic fallback | share of any ffmpeg pre-shrink this query paid for (usually 0) |

Summed over a query, `Time` is that query's inference wall clock. Model load
and the offline ASR pass are excluded, as in the other two pipelines. The
end-of-run report prints mean/min/max and a count of zero-valued entries.

Output, cache/resume, `--rebuild`/`--only`/`--force`, the tidy/dedupe rules and
the fallback answer pool are **identical to `generate.answers.cosmos.py`**, so
the three runs stay directly comparable.

---

## 10. Testing

**Against the real model:** 24 queries end to end across three concurrency
settings, covering a 320 s clip needing the ffmpeg pre-shrink, a counting
question, dense-routed and transcript-routed queries. 0 failures, 0 degrades,
0 fallback padding, 0 truncations, 0 unparsed replies, all with the video
attached.

**Against a mock vLLM server** that records every request — the paths the real
run has not exercised:

- routed plans, budgets and grouping over real queries, videos and transcripts;
- the full degrade ladder: context-overflow shrink, processor rejection through
  `no_dsf` to `no_mm_kwargs` to transcript-only, with the transcript confirmed
  present in the degraded prompt;
- preflight self-correction when the server rejects the kwargs or the JSON
  schema, preflight *not* aborting over a context-budget miss, and preflight
  requiring both probe videos to agree;
- prose wrapped around the answer JSON; short answer lists padded to ten;
  resume, `--rebuild`, `--force`, `--only`, `--ignore-routing`;
- `validate.ag.run.py --warnings-as-errors` passing 12/12, 0 errors, 0 warnings.

### Sample output (real model)

```
Q1  (3.5 s)   They walk off-screen. | Their image disappears. | They exit the frame.
Q2  (320 s)   Film rehearsal for dramatic character entrance
Q3  (26 s)    0 | unknown | 1 | 2
Q6            In front of the weather map | In the television studio | At the weather desk
```

**Answers run short**, though the figure is noisier than one number suggests.
Over the same 8 queries, the 1-worker and 4-worker runs gave **3.16 and 3.86**
words across all ten slots (3.62 / 4.12 for rank 1 alone) — ~20% apart, because
temperature is 1.0. Call it 3–4 words on n=8. Gemma 4 measured 4.7 over 20
queries.

The comparison point, **ground truth mean 8.4 words**, was verified this
session against `VQA/vqa.gt.csv`: mean 8.38, median 8, p90 17, max 42, 17.1%
single-word over 2,000 rows. **But that file is the 2025 ground truth and its
2,000 videos have zero overlap with the 2026 test set's 1,956** — different
clips, questions and annotation session. It is a proxy for annotator style, not
the reference this run will be scored against.

Whether the terseness actually costs score is **an untested hypothesis.** The
mechanism is arithmetic — `eval.vqa.ag.py` scores each answer against a single
reference with METEOR, which weights recall well above precision, and a 3-word
answer can match at most 3 of an 8-word reference's tokens — but nothing here
has measured it. See §12.

---

## 11. Command reference

```bash
# Terminal A — free the card and serve
wsl -d Ubuntu-24.04
systemctl --user stop cosmos-reason2
cd /home/aakash/cosmos-reason2
nohup ./.venv/bin/vllm serve Qwen/Qwen3-VL-8B-Thinking \
  --host 0.0.0.0 --port 8009 \
  --served-model-name Qwen/Qwen3-VL-8B-Thinking \
  --allowed-local-media-path /mnt/e/TREC_2026 \
  --max-model-len 24576 --gpu-memory-utilization 0.92 --enforce-eager \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3 --skip-mm-profiling \
  > /home/aakash/qwen3vl.server.log 2>&1 &

grep -E "Available KV cache|Application startup complete" /home/aakash/qwen3vl.server.log

# Terminal B — the run (~11 h, resumable, Ctrl-C safe)
cd /mnt/e/TREC_2026
/home/aakash/cosmos-reason2/.venv/bin/python VQA/generate.answers.qwen3vl.py

# Validate
/home/aakash/cosmos-reason2/.venv/bin/python VQA/validate.ag.run.py \
  vqa.2026.testingDataset.Task1.json qwen3vl.run.task1.json \
  --warnings-as-errors --report-json qwen3vl.run.validation.json

# Shut down
pkill -f "vllm serve"
```

Variants: `--limit 20` (smoke test, kept in cache), `--rebuild` (re-emit from
cache, no inference), `--only 9 20 26 --force`, `--ignore-routing` (use a
separate `--cache`/`--out`), `--budget brief`.

Must run from **WSL2, not Windows** — the server opens the video off the path
in the request. The client warns if started on Windows.

---

## 12. Known limitations

- **Accuracy is unmeasured**, as with the other two passes. Everything above is
  format correctness, evidence coverage and plausibility.
- **The 11 h projection rests on 8 short clips.** The long tail is unmeasured.
- **`--skip-mm-profiling` leaves the encoder cache unreserved.** Measured safe
  over 24 queries at 22,998/24,576 MiB, but it is a margin, not a guarantee.
- **Answers are terse** (3–4 words, n=8) against a 2025 ground-truth mean of
  8.4. That the terseness costs METEOR is a mechanism, not a measurement:
  `nltk` and `bert_score` are installed in neither environment, the 2025 GT
  videos have **zero** overlap with the 2026 test set, and there is no 2026 GT.
  Nothing in this project has ever scored an answer.
- **16 queries have no video on disk** and are answered from question text
  (plus transcript where one exists); they must still appear in the run.
- **Not bit-reproducible**: no seed is set, and temperature is 1.0 per the
  model card. Two runs of the same query give different traces and can give
  different answers.
- **`--json-schema` is off by default** and unverified against this server —
  preflight will disable it if the reasoning parser does not exempt the think
  block.

---

## 13. State at the end of the session

- The server config is settled and the client is fixed for the
  `do_sample_frames` trap; both changes verified against the live server.
- The first full-run launch was **stopped** after 4 queries because it was
  running with the frame budget disabled. Those 4 records are to be discarded
  (`rm -f answers.task1.qwen3vl.cache.jsonl`) — they were answered at ~12,000
  prompt tokens under the broken config.
- **The full run has not been completed.** It is the next action, and at ~11 h
  it fits before the 17 August deadline.

## 14. Next steps

1. **Delete the 4-record cache and launch the full run**; confirm preflight
   prints `video+kwargs ok` (a `(1/2 probe videos …)` suffix is expected and
   healthy) and that early prompt sizes are in the hundreds, not five figures.
2. **Validate** and compare against `gemma4.run.task1.json` and
   `cosmos.run.task1.json`.
3. **Consider re-running Cosmos** now that the `Video_ID` naming bug is known —
   45% of that run was answered without its video.
4. **Consider the `--ignore-routing` ablation** on a subset, to find out
   whether the routing is earning its complexity.
5. Accuracy remains the open question for all three passes: it needs the 2025
   GT videos plus `nltk` and `bert_score`, none of which are on this box.

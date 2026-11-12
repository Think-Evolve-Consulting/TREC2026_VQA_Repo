# Task 2 (multiple choice) rankers, standalone for Colab

Six self-contained scripts, one per model. Each is a port of the corresponding
`VQA/rank.options.mc.*.py` pass with the project dependencies removed: copy one
file to Colab, point it at a Drive copy of `videos_TEST_VQA/`, and it produces a
`*.run.task2.json` that passes `validate_vqa_mc_2026.py`.

Nothing here imports from the repo. Nothing here reads `E:\`.

---

## Which one to run

If you run one, run **Qwen3.8**. It is the best ranker measured in this project
and nothing — not another model, not any of 781,248 searched fusions — beat it
on the labelled dev set.

| Script | Model | Dev Top-1 / MRR | vs Qwen3.8 | Local 2026 wall clock | Needs |
|---|---|---|---|---|---|
| `task2_qwen38_colab.py` | Qwen3.8-27B INT4 | **0.800 / 0.884** | — | 7.53 h | A100; L4 tight |
| `task2_qwen3vl_colab.py` | Qwen3-VL-8B | 0.790 / 0.872 | −1.0, p = 0.70 | 18.88 h | L4 / A100 |
| `task2_internvl_colab.py` | InternVL3.5-30B-A3B Q4 | 0.760 / 0.859 | −4.0, p = 0.064 | 1.99 h | A100 |
| `task2_muse_colab.py` | Muse Glimmer 30B Q4 | 0.755 / 0.856 | −4.5, p = 0.018 | *never run* | A100 |
| `task2_videollama3_colab.py` | VideoLLaMA3-7B bf16 | 0.738 / 0.843 | −6.2, p = 0.004 | 6.12 h | L4 / A100 |
| `task2_gemma4_colab.py` | Gemma 4 12B INT4 | 0.733 / 0.840 | −6.8, p = 0.0005 | 3.01 h | T4 may work |

Dev figures are Top-1 / MRR on the 400-query labelled 2025 sample (seed 2026),
scored with the official scorer. Random control on the same sample: 0.253 /
0.517. **Muse failed its pre-registered gate and was never run over 2026** — its
script is here because it works, not because the result recommends it.

Two things the dev numbers do not tell you: the sample is all 4-option while the
2026 set is 81% 8-option, and ~1-point differences cannot be resolved at n = 400.
Qwen3.8 and Qwen3-VL are statistically tied.

**Fusion does not help.** At least one model puts the right answer first on 93.5%
of dev queries, but every model loses more to Qwen3.8 than it wins, so voting
mixes in more error than it removes. The best of 781,248 weighted fusions scored
0.825 in-sample and exactly the baseline (0.801) under nested cross-validation.
Run several models for separate submissions, not to combine them.

---

## What to put on Drive

```
TREC_2026/
  vqa.2026.testingDataset.Task2_v2.json     <- the official query file
  videos_TEST_VQA/
    frames/index.json                       <- frame models + durations
    frames/<video_id>/001.jpg ...           <- frame models only (1.15 GiB)
    videos/<video_id>.{mp4,webm}            <- video models only (11.4 GiB)
    transcription_openwhisper/<id>.json     <- all models (11 MiB)
  query.routing.task1.json                  <- optional, see "Routing"
  runs/                                     <- created; runs and caches land here
```

| Model | frames/ | videos/ |
|---|---|---|
| gemma4, internvl, muse, videollama3 | **required** | not needed |
| qwen3vl, qwen38 | `index.json` only | **required** |

So you never need both trees for one model, and the video models need only one
small file out of `frames/`.

**Upload `frames/` as a tar.** It is 26,251 files for 1.15 GiB and Drive's FUSE
mount is slow per file — reading it in place costs 20-40 minutes per session.

```bash
cd E:/TREC_2026/videos_TEST_VQA && tar -cf frames.tar frames
```

Put `frames.tar` next to where `frames/` would be; `--stage auto` finds it,
unpacks it to local scratch once, and everything after that reads local disk.

The video scripts copy each clip to local scratch on first use for the same
reason (`--stage local`, the default). 11.4 GiB fits Colab's disk.

---

## Running one

```python
from google.colab import drive; drive.mount('/content/drive')
```

```bash
!python task2_qwen38_colab.py --install \
    --drive-root /content/drive/MyDrive/TREC_2026
```

`--install` fetches the dependencies and continues. Each script then starts its
own server (or loads the model in-process), runs, writes the run file, stops the
server, and validates what it wrote. There is nothing else to orchestrate.

Two scripts need more than that:

* **`task2_gemma4_colab.py`** — `google/gemma-4-12B-it-qat-w4a16-ct` is a gated
  repo. Accept the licence on huggingface.co, then pass `--hf-token hf_...`.
* **`task2_videollama3_colab.py`** — its `--install` pins `transformers==4.46.3`
  and `numpy<2`, which breaks other packages. It installs, tells you to restart
  the runtime, and exits. Run it again without `--install` afterwards, and do
  not run another model in that runtime without restarting.

The two llama.cpp scripts (`internvl`, `muse`) **build llama.cpp from source with
CUDA on first use, 10-20 minutes**; there are no official prebuilt Linux CUDA
binaries. Pass `--llama-cache /content/drive/MyDrive/TREC_2026/llama` and the
built binary is kept on Drive, so later sessions skip the build.

---

## Resuming, which you will need

Only InternVL (~2 h) reliably finishes inside one Colab session. Everything else
needs two or more, and free-tier runtimes disconnect sooner than that.

Every completed query is appended to a JSONL cache on local disk immediately,
mirrored to Drive every `--sync-every` records, and re-read on startup:

* **After a disconnect, re-run the identical command.** It reloads the cache from
  Drive and starts from the first unranked query.
* **To split deliberately**, use `--shard 1/4` ... `--shard 4/4` in separate
  sessions against the same Drive root, then `--assemble-only` once to emit the
  single complete run. Shards partition by *video*, never splitting one video's
  queries, because the passes of a video share a decoded prefix the server
  prefills once.
* **A partial run is still a valid run.** Queries not yet ranked keep the official
  option order — no better than the random control for those queries, counted and
  reported, never hidden — so the file validates and could be submitted as-is.

A partial run file is rewritten at every sync, so a session killed without
warning still leaves something scoreable on Drive.

---

## The ranking contract (identical in all six)

* **Rank option numbers, never option text.** The model returns a permutation of
  `1..N`, mapped back to the official strings by index, so no rank is spent twice
  and no option goes unranked.
* **Permute the presentation.** Official order, exact reverse, then seeded
  shuffles, combined by a confidence-weighted Borda count. Position bias measured
  at chance in every local run: rank 1 on the officially-first option 14.9-17.4%
  against 14.9% expected.
* `as_permutation()` repairs repeats and drops; `check_permutation()` refuses to
  emit anything the validator would reject.
* Seeds derive from the Q_ID, so re-running reproduces the same file.
* **No `Time` field** — the validator rejects any key outside `Rank`/`Answer`.

This core is byte-identical across the six scripts and was verified against the
shipped runs: `assemble()` from the real caches reproduces
`gemma4.vllm.run.task2.json` and `qwen3vl.run.task2.json` exactly, and the
built-in `validate_run()` finds 0 errors on both. The qwen3vl script's own
report also reproduces that run's documented diagnostics from its cache — 116
queries with no video, 196 recovered by the `no_dsf` rung, 83.6% unanimous.

---

## Routing

The local 2026 runs had routing **on**: `query.routing.task1.json` decides
whether a query is sent frames, the transcript, or both, and adds an OCR or
temporal rule to the prompt. It sent 116 queries to the transcript alone and
gave 68 of those their frames back because the clip has no speech.

If that file is not on Drive, every query gets **both** inputs and no extra
rules — the same as `--ignore-routing`. That is a defensible configuration (all
dev runs used it) but it is **not** what the local 2026 runs did, so runs made
with and without it are not directly comparable. Copy the file if you want to
reproduce the submitted behaviour.

---

## Validating before you submit

Each script runs its own `validate_run()` at the end and prints the errors it
finds. That re-checks every invariant the official validator enforces: exactly
the official Q_IDs in the official order, Video_IDs matching, ranks exactly
`1..N` in ascending order, answers a permutation of the official option strings,
and no extra keys anywhere.

It is not a substitute for the real thing. Before submitting, run:

```bash
python validate_vqa_mc_2026.py \
    --queries vqa.2026.testingDataset.Task2_v2.json \
    --run <model>.run.task2.json --warnings-as-errors
```

---

## Known traps, carried forward per model

Each script's module docstring documents the ones specific to it. The four that
cost the most locally:

| Model | Trap | What the script does |
|---|---|---|
| qwen3vl, qwen38 | `size.longest_edge` is a **whole-video** pixel budget, not per-frame. Read per-frame it sends thumbnails — 485 visual tokens against a planned 17,792 | sets the budget directly; fps only spreads it over time |
| qwen3vl, qwen38 | `do_sample_frames` is rejected on short clips, and a swallowed rejection silently answers with no video | not sent below 16 frames; the `no_dsf` ladder rung catches the rest; `video_used` is counted and reported |
| internvl | llama.cpp tiles each frame into 448-px crops — 16 frames = 20,945 tokens, every request 400s; `--image-max-tokens` does not cap it | `--frame-size 448` resizes to one tile, client-side |
| videollama3 | the sdpa vision encoder materialises a full `[S,S]` mask, ~8 GB per layer at 16k patches | block-wise attention patch, verified exact (max abs diff 0.00000) |
| muse | the chat template defaults to reasoning strength `high`, roughly doubling decode | always sends `reasoning_strength` (default `low`) |

---

## Useful flags (all six)

| Flag | What it does |
|---|---|
| `--limit 20` | smoke test: rank 20 queries and stop |
| `--only 1 2 3` | rank specific Q_IDs |
| `--shard I/N` | rank shard I of N, split by video |
| `--force` | re-rank queries already cached |
| `--assemble-only` | write the run from the cache, no inference |
| `--ignore-routing` | send every query both inputs |
| `--passes 1` | one presentation order instead of three (3x cheaper, loses the position-bias correction) |
| `--no-serve` | attach to a server already running |
| `--work` | local scratch (default `/content/trec_work`) |
| `--sync-every N` | how often the cache is mirrored to Drive |

Start with `--limit 20` on any new runtime. It exercises the whole path —
staging, server start, prompts, parsing, cache, mirror, validation — in a few
minutes.

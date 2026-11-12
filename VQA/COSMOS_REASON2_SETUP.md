# Cosmos-Reason2-2B VQA Pipeline

How the Cosmos-Reason2 answer-generation pipeline for TREC Task 1 was set up,
and how to run it. Companion to the Gemma4/Ollama pipeline (`generate.answers.py`)
already in this folder — same query/transcript inputs, same run-file output
format, different model and a much cheaper one-call-per-query design instead
of a 10x3 ensemble.

> ## ⚠ `cosmos.run.task1.json` is affected by a fixed bug — re-run before using it
>
> The run file produced before **14 August 2026** was generated with **1,098 of
> its 2,443 queries answered with no video at all**, because `Video_ID` was
> treated as a filename. The script is fixed; the *run file and its cache are
> not*. See [The Video_ID bug](#the-video_id-bug-fixed-14-august-2026) for what
> happened and how to re-run.

## Architecture

- **Model**: `nvidia/Cosmos-Reason2-2B`, served by vLLM behind an
  OpenAI-compatible API.
- **Where it runs**: inside WSL2 (`Ubuntu-24.04` distro), at
  `/home/aakash/cosmos-reason2` — a clone of
  [nvidia-cosmos/cosmos-reason2](https://github.com/nvidia-cosmos/cosmos-reason2).
  vLLM needs Linux/CUDA, so this can't run natively on Windows.
- **Endpoint**: `http://localhost:8008/v1` — reachable from both WSL2 and the
  Windows host (WSL2's localhost-forwarding handles the crossover). Port 8008,
  not the usual 8000, because something else on the Windows host was already
  squatting 8000.
- **The client** (`generate.answers.cosmos.py`, in this folder) also runs
  from WSL2, not Windows — it needs zero network hop to reach the server and
  reads `/mnt/e/TREC_2026/...` directly (WSL2 auto-mounts the E: drive; it's
  the same files either way).
- **Persistence**: the server runs as a `systemctl --user` service
  (`cosmos-reason2.service`), so it survives closing the terminal and
  auto-restarts if it crashes.

## Running the batch client from WSL2

**1. Open a WSL2 shell in the right distro.** From Windows Terminal, PowerShell,
or cmd:

```
wsl -d Ubuntu-24.04
```

(There are other WSL distros on this machine — `Ubuntu`, `Ubuntu-22.04`,
`salad-enterprise-linux`, `docker-desktop` — none of those have the GPU
environment or the venv set up. Must be `Ubuntu-24.04`.)

**2. Confirm the server is up:**

```bash
curl -s http://localhost:8008/v1/models
```

Should return JSON naming `nvidia/Cosmos-Reason2-2B`. If it doesn't respond,
see [Operating the server](#operating-the-server) below.

**3. Run the client.** It needs the `openai` Python package, which lives in
the server's venv — not the system `python3`. Always invoke it as:

```bash
/home/aakash/cosmos-reason2/.venv/bin/python /mnt/e/TREC_2026/VQA/generate.answers.cosmos.py [options]
```

(or `cd /mnt/e/TREC_2026/VQA` first and drop the script's path prefix — same
thing, the E: drive is the same filesystem from either side).

### Common invocations

| Goal | Command |
| --- | --- |
| Smoke test (5 queries) | `... generate.answers.cosmos.py --limit 5` |
| Full run, resumable | `... generate.answers.cosmos.py` |
| Retry specific queries | `... generate.answers.cosmos.py --only 9 20 26 --force` |
| Re-emit run file from cache only, no new inference | `... generate.answers.cosmos.py --rebuild` |
| More/less concurrency | add `--workers 3` (default 2) |
| Point at a different duration source | `--frame-index <path>` (default `videos_TEST_VQA/frames/index.json`) |

Check the startup line before letting a full run proceed — it reports how many
queries resolved a video file and how many have a known duration:

```
queries: 2443 | cached: 0 | durations known: 1940 | queries with no video file: 16
```

`queries with no video file` should be **16**. Anything near 868 means the
naming convention has broken again — see [The Video_ID
bug](#the-video_id-bug-fixed-14-august-2026). (`durations known` counts
*videos*, 1,940 of them, which covers 2,427 of the 2,443 queries.)

It's safe to `Ctrl+C` and re-run at any point — finished queries are cached
in `answers.task1.cosmos.cache.jsonl` (JSONL, one line per answered `Q_ID`)
and skipped on the next run. Output run file:
`E:\TREC_2026\cosmos.run.task1.json`, in the exact schema
`validate.ag.run.py` expects.

**Optional convenience**: add this to `~/.bashrc` inside WSL2 to get a short
command instead of the full path every time:

```bash
alias cosmos-answers='/home/aakash/cosmos-reason2/.venv/bin/python /mnt/e/TREC_2026/VQA/generate.answers.cosmos.py'
```

Then just `cosmos-answers --limit 5`.

### Validating output

Same validator as the Gemma4 run:

```bash
cd /mnt/e/TREC_2026/VQA
python3 validate.ag.run.py /mnt/e/TREC_2026/vqa.2026.testingDataset.Task1.json /mnt/e/TREC_2026/cosmos.run.task1.json
```

## Operating the server

```bash
systemctl --user status cosmos-reason2      # is it up?
systemctl --user restart cosmos-reason2     # restart
systemctl --user stop cosmos-reason2        # stop (frees the whole GPU)
tail -f /home/aakash/cosmos-reason2/server.log   # live log
nvidia-smi                                  # GPU memory check
```

The unit file is `~/.config/systemd/user/cosmos-reason2.service`. It starts
automatically whenever this WSL2 session/systemd instance is running (no
manual start needed after a reboot), because it's `enable`d — but
`Linger=no` for this user, so if you fully log out of WSL2 (close every WSL2
shell/session), systemd's user instance can stop and take the service with
it. If unattended runs across full logouts become important, run (needs a
password, so not done automatically here):

```bash
sudo loginctl enable-linger aakash
```

### Current server config, and why

```
vllm serve nvidia/Cosmos-Reason2-2B \
  --host 0.0.0.0 --port 8008 \
  --allowed-local-media-path /mnt/e/TREC_2026 \
  --max-model-len 24576 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3
```

- **`--gpu-memory-utilization 0.75`** (~16GB of the 3090's 24GB): the GPU is
  shared with the Windows host, which we observed using anywhere from 0 to
  ~9GB on its own. This leaves headroom for that instead of assuming the full
  24GB is always free.
- **`--max-model-len 24576`** + client `max_tokens=8192`: sized together so
  the generation budget doesn't eat into the video-token budget (see
  `video_mm_kwargs()` in the client).
- **`--enforce-eager`**: skips CUDA-graph capture to save VRAM, small latency
  cost.
- **`--media-io-kwargs num_frames=-1`**: required by Cosmos-Reason2 (built on
  Qwen3-VL) for correct per-frame timestamp embedding — see the OOM bug
  below for the tradeoff this creates.
- **`--reasoning-parser qwen3`**: splits the model's `<think>...</think>`
  block into a separate `reasoning_content` field so the final answer JSON
  stays clean in `content`.

## The Video_ID bug (fixed 14 August 2026)

Found while writing `generate.answers.qwen3vl.py`, which is why it is recorded
in `SESSION.2026-08-14.qwen3-vl.md` §1 as well. Unlike the three bugs below, it
was **not** caught during validation — it produced a run that validated cleanly,
2,443/2,443, and was silently wrong.

### What went wrong

`Video_ID` is not a filename. 868 of the 2,443 queries name a bare YouTube ID,
and `download_test_videos.py` wrote those as `<Video_ID>.mp4` beside the
released `.webm` clips (`SESSION.2026-08-11.task1-pipeline.md` §3):

```python
name = vid if vid.endswith(".webm") else vid + ".mp4"
```

`answer_query()` instead resolved `videos_dir / video_id` literally, so for a
YouTube ID it looked for `videos_TEST_VQA/videos/PeGejCJ2CE0` — a path that
never exists. `Path.exists()` returned `False`, `video_exists` went `False`, and
the query was answered from the transcript and question text alone. No error, no
warning; the failure mode of the media path is a *quieter, still-plausible
answer*.

Counted from `answers.task1.cosmos.cache.jsonl` itself:

| | Queries |
|---|---|
| Answered with the video attached | 1,348 |
| **Answered with no video at all** | **1,098** |
| …of those, with no transcript either — question text only | **469** |
| Queries that genuinely have no media on disk | **16** |

So **45% of `cosmos.run.task1.json` was answered blind**, where the true figure
should be 0.7%.

### The second bug this exposed

Sampling fps came from the transcript's `duration_sec`, which only exists when
the video has detected speech — 1,364 of 2,443 queries. The other 1,079 fell
back to a flat `2.0` fps in `adaptive_fps()`. That was harmless while no video
was attached; the moment the fix attaches one, a 389 s clip at 2 fps asks for
778 frames and would overflow the context window. Fixing the first bug without
this one would have traded silent blindness for loud failures.

### The fix

Two functions added to `generate.answers.cosmos.py`:

- **`resolve_video(videos_dir, video_id)`** — applies the documented naming
  convention. Used by `answer_query()` and by the startup summary.
- **`load_durations(index_path)`** — reads probed durations from
  `videos_TEST_VQA/frames/index.json` (written by `extract.frames.py`; the
  frames themselves stay unused, this model takes video). Duration now resolves
  **probed first, transcript second**. New `--frame-index` flag, default as
  above.

Measured after the change, against the real corpus:

| | Before | After |
|---|---|---|
| Queries whose video resolves | 1,575 | **2,427** |
| Queries with no video file | 868 | **16** |
| Queries with a known duration | 1,364 | **2,427** |
| Queries planning at a flat 2.0 fps | 1,079 | **0** |

Spot-checked across both ID shapes: `8384.webm` → 3.49 s → 4.000 fps;
`PeGejCJ2CE0` → `PeGejCJ2CE0.mp4`, 25.96 s → 2.465 fps; `Archipelago-22.webm`
→ 389.0 s → 0.200 fps.

The startup line now reports it, so a repeat is visible immediately:

```
queries: 2443 | cached: 0 | durations known: 1940 | queries with no video file: 16
```

### Re-running

The fix does not repair the existing run — **the cache holds the blind
answers**, and cached Q_IDs are skipped. Move it aside rather than deleting it;
it is the evidence for the numbers above.

```bash
cd /mnt/e/TREC_2026
mv answers.task1.cosmos.cache.jsonl answers.task1.cosmos.cache.PREFIX-BUG.jsonl
mv cosmos.run.task1.json cosmos.run.task1.PREFIX-BUG.json

# the GPU must be free -- stop whatever else is serving first
systemctl --user start cosmos-reason2
/home/aakash/cosmos-reason2/.venv/bin/python VQA/generate.answers.cosmos.py
```

Expect it to be **slower than the original run**: 852 more queries now actually
decode and encode a video instead of taking the cheap text-only path, and some
of those are the long clips that need the ffmpeg pre-shrink below. The original
summed **18.34 h of inference** across 2,443 queries (mean 27.0 s, max 173.5 s),
which at `--workers 2` was roughly half that in wall clock.

### Was the old run worth re-running?

Rebuilt from the old cache, its own quality report says yes:

```
answer words  mean  3.3  median 2  p90 7   (ground truth: mean 8.4, median 8, p90 17)
queries needing fallback padding: 1808
queries whose JSON did not parse:  457
rank-1 answer is "unknown":        601
queries answered without a video file: 1097
```

457 queries returned nothing parseable and 1,808 needed generic padding to reach
ten answers — on a run where 45% of queries never saw their video. These numbers
are the baseline the re-run should beat, and they are worth re-reading afterwards
to confirm it did.

`--force` would also work without moving the cache, but re-answers all 2,443
into the same file, leaving the old and new records interleaved with the later
one winning. Moving the cache keeps the two runs separately inspectable.

## Bugs found and fixed while validating this pipeline

All three were reproduced against real videos in this corpus, not
hypothetical:

1. **OOM on long/high-res videos.** `num_frames=-1` makes vLLM decode every
   native-resolution frame into memory *before* any fps-based subsampling
   happens downstream. A 320s, 2048x1152 clip (`ChainedforLife-14.webm`)
   tried to allocate 50.6GB and crashed the request. Fixed client-side: any
   video whose native `duration x fps x width x height x 3` estimate exceeds
   4GiB gets pre-shrunk with `ffmpeg` (frame-rate reduction, matching the
   client's own adaptive fps target) before being handed to the server,
   cached under `videos_TEST_VQA/videos/.cosmos_transcoded_cache/`. The
   `--media-io-kwargs` setting itself was left alone rather than weakened
   globally, since NVIDIA's own docs call it out as required for this model
   family's timestamp correctness.
2. **Truncated JSON answers.** `max_tokens=3072` was too tight for some
   questions — generation got cut off mid-`<think>`, before ever reaching the
   closing tag or the final JSON line (`reasoning_content` empty,
   unparseable `content`, and consistently the slowest calls in a batch).
   Raised to 8192 (with `--max-model-len` raised in step alongside it).
3. **HF processor crash on one specific video.** A dropped/corrupt frame
   desynced the decoded frame count from the container's own metadata, which
   broke Qwen3-VL's timestamp math (`BadRequestError` from
   `Qwen3VLProcessor`). The client now retries once with the video dropped
   (transcript+question only) instead of losing the query outright.

**Known, accepted edge case (not fixed):** asking for 10 genuinely-distinct
answers to open-ended, dialogue-guessing-style questions (e.g. "what did she
say") occasionally sends the model into a long non-converging reasoning pass
— confirmed via a low-budget diagnostic call that it answers such questions
fine when only asked for *one* answer; the difficulty is specifically
enumerating ten different plausible guesses. It still resolves to a valid,
fallback-padded answer (no crash), just slower — around 150-170s against a
typical 5-20s. Hit roughly 3 of 32 queries in validation testing. Worth
knowing it adds wall-clock time across a full run; not worth deeper
engineering (would need streaming + early-stop heuristics) given it already
degrades gracefully.

## Model choice notes (for future reference)

- **Why 2B, not 8B**: NVIDIA's documented minimum for `Cosmos-Reason2-8B` is
  32GB; the 3090 has 24GB total and shares it with Windows. `-2B`'s
  documented minimum (24GB on a clean card) fits once tuned to the actual
  free budget.
- **Why not a quantized 8B**: the one ready-made quantized checkpoint found
  (`Firworks/Cosmos-Reason2-8B-nvfp4`) targets Blackwell-generation tensor
  cores specifically and won't run properly on the 3090's Ampere
  architecture. A better-suited-for-Ampere INT4/AWQ quantization would need
  to be produced from scratch against this very new architecture — a real
  side-quest, not attempted here.
- **Why Cosmos-Reason2 and not "Cosmos 3"**: Cosmos 3's Reasoner variants
  (Nano 16B / Super 64B) target Hopper/Blackwell data-center-or-workstation
  cards, not a single consumer 3090.

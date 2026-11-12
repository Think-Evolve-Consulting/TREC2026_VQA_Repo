# Gemma 4 12B on vLLM (WSL2)

How to serve Gemma 4 for `VQA/generate.answers.py --backend vllm`, the same
answering pass that produced `gemma4.run.task1.json` through Ollama on
Windows, moved onto vLLM inside WSL2 so all three models live on one side of
the machine. Companion to `COSMOS_REASON2_SETUP.md` (port 8008) and
`QWEN3_VL_SETUP.md` (port 8009); this is **port 8010**.

**Status: working.** The server starts, 20 real queries have been answered end
to end with 0 failures, 0 fallback padding and 0 degrades, and the throughput
in §6 is measured on this box. Nothing about answer *quality* has been
measured — see §7.

## Architecture

- **Model**: `google/gemma-4-12B-it-qat-w4a16-ct` — Google's own QAT INT4
  checkpoint of the 11.9B `Gemma4UnifiedForConditionalGeneration`, 10.26 GB,
  262K native context, image + audio + video capable. Why this one and not
  bf16 is §2.1.
- **Server**: vLLM 0.27.1 in a **new** venv at `/home/aakash/gemma4/.venv`.
  It cannot share the Cosmos/Qwen3-VL venv — that one is vLLM 0.12.0, which
  has no Gemma 4 architecture at all (`gemma`, `gemma2`, `gemma3`, `gemma3n`
  only). Upgrading it in place would also put the two finished runs at risk,
  so it was left alone.
- **Client**: `VQA/generate.answers.py --backend vllm` (the default). The
  frames go over the wire as base64 `image_url` parts, so the client does not
  have to be on the same filesystem as the server — but run it from WSL2
  anyway, so the whole pass is on one side.
- **Port**: 8010. The three servers cannot share the 3090; stop the others
  first.

## 1. Free the GPU

`cosmos-reason2.service` is **enabled**, so it starts on its own and takes
~16 GB whenever the distro comes up:

```bash
wsl -d Ubuntu-24.04
systemctl --user stop cosmos-reason2
nvidia-smi          # expect the 3090 back at ~0 MiB
```

The Cosmos and Qwen3-VL runs are both finished and validated, so if the
restarts get tiresome, `systemctl --user disable cosmos-reason2` stops it
coming back. That is a deliberate choice about their runs, not a step here.

## 2. Environment

```bash
uv venv --python 3.12 /home/aakash/gemma4/.venv
uv pip install --python /home/aakash/gemma4/.venv/bin/python vllm==0.27.1
uv pip install --python /home/aakash/gemma4/.venv/bin/python transformers==5.12.1
```

That second pin is not optional — see §4.1. The result is vLLM 0.27.1,
torch 2.13.0+cu130, transformers 5.12.1, Python 3.12.3.

### 2.1 Why the QAT checkpoint and not bf16

| Checkpoint | Size | On this 3090 |
|---|---|---|
| `google/gemma-4-12B-it` | 23.92 GB | **Does not fit.** Windows permanently holds ~1.3 GiB, leaving 22.76 of 24.0 GiB free to CUDA, and that is before any KV cache. |
| **`google/gemma-4-12B-it-qat-w4a16-ct`** | **10.26 GB** | 8.28 GiB resident, 12.52 GiB left for KV. Google's own QAT, INT4 group-32 symmetric, compressed-tensors `pack-quantized` → Marlin kernel on sm_86. |
| `cyankiwi/gemma-4-12B-it-AWQ-INT4` | ~6 GB | Community quant, unevaluated. Not needed. |

The QAT checkpoint holds the vision and audio projections **out** of the
quantisation (`quantization_config.ignore` lists `embed_vision.*` and
`embed_audio.*`), so the multimodal path stays bf16. It is also the closest
match to what Ollama was serving — `gemma4:12b` is Q4_K_M — which is what
makes the two runs roughly comparable rather than a quantisation experiment.

```bash
/home/aakash/gemma4/.venv/bin/hf download google/gemma-4-12B-it-qat-w4a16-ct
```

Not gated; no license click, no token needed.

## 3. Keep the distro alive

**This bit the bring-up twice and will bite again.** WSL2 tears the whole
distro down when the last session exits, and the user systemd manager — and
every service under it — goes with it. Linger is now on:

```bash
loginctl enable-linger aakash     # already done; survives reboots
```

Linger keeps the *user manager* alive while the distro runs. It does not keep
the *distro* alive. So for a multi-hour run, **keep one interactive WSL2
window open** for the duration, or start the server and run the client in the
same session. If the server dies mid-run the client now stops with a clear
error instead of quietly padding — that guard exists precisely because of
this (§5).

## 4. Serve

A systemd user unit is installed at `~/.config/systemd/user/gemma4-vllm.service`.
It is deliberately **not enabled**: one 3090, three servers.

```bash
systemctl --user start gemma4-vllm
curl -s http://localhost:8010/v1/models        # sanity check, ~2 min after start
systemctl --user stop gemma4-vllm
```

The equivalent foreground command, if you would rather watch it:

```bash
cd /home/aakash/gemma4
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
./.venv/bin/vllm serve google/gemma-4-12B-it-qat-w4a16-ct \
  --host 0.0.0.0 --port 8010 \
  --served-model-name google/gemma-4-12B-it-qat-w4a16-ct \
  --allowed-local-media-path /mnt/e/TREC_2026 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image": 32}' \
  --enforce-eager \
  --reasoning-parser gemma4
```

Measured on this box:

```
Model loading took 8.28 GiB memory and 3.72 seconds
Encoder cache will be initialized with a budget of 2496 tokens
Available KV cache memory: 12.52 GiB
GPU KV cache size: 98,097 tokens
Maximum concurrency for 16,384 tokens per request: 5.99x
Peak VRAM during a 20-query run: 22,948 MiB of 24,576
```

- **`--limit-mm-per-prompt '{"image": 32}'`** — required. vLLM allows **one**
  image per prompt by default and this pipeline sends up to 32
  (`extract.frames.py` caps there; the corpus mean is 13.5).
- **`--max-model-len 16384`** — sized against the real worst case, not the
  262K window: 32 frames × 280 soft tokens = 8,960, plus the longest
  transcript in the corpus (4,432 chars ≈ 1,231 tokens), the prompt and 256
  of output. Matches the `num_ctx` the Ollama pass used, which keeps the two
  runs comparable.
- **`--gpu-memory-utilization 0.92`** — not higher, same as Qwen3-VL: only
  22.76 of 24.0 GiB is free to CUDA and 0.95 is rejected at startup.
- **`--enforce-eager`** — as with both sibling services. There is no CUDA
  toolkit on this box, so the inductor/triton compile path has nothing to
  compile with. VRAM is *not* the constraint here (12.52 GiB of KV headroom),
  so if a toolkit ever gets installed, dropping this may buy decode speed.
  Untested.
- **`--reasoning-parser gemma4`** — splits any `<think>` block into
  `reasoning_content` so the answer JSON stays clean in `content`. The client
  turns thinking off anyway (below), so this is belt and braces.
- **`--allowed-local-media-path`** — not needed by the current client, which
  sends base64. Kept for consistency with the sibling units and so `file://`
  frame paths remain an option.

### 4.1 Three things that stop it starting, none of them obvious

All three were hit in order during bring-up. The error each prints points
somewhere unhelpful.

| Symptom | Cause | Fix |
|---|---|---|
| `AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute` | transformers ≥5.13 guards heterogeneous configs; Gemma 4 mixes `head_dim` 256 with `global_head_dim` 512, and vLLM 0.27.1 still reads `config.head_dim` globally. vLLM only floors `transformers>=5.5.3`, so a fresh install takes 5.15.0 and breaks. | `transformers==5.12.1` — has Gemma 4 (since 5.5.3), predates the guard. |
| `RuntimeError: UVA is not available` | vLLM 0.27's V2 model runner allocates UVA buffers. WSL2's GPU-PV does not expose unified virtual addressing. | `VLLM_USE_V2_MODEL_RUNNER=0` |
| `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` | FlashInfer JIT-compiles its top-k/top-p sampler on first use. No CUDA toolkit here. Note this fires at the *first sampled token*, so the server looks healthy until a request arrives. | `VLLM_USE_FLASHINFER_SAMPLER=0` |

The middle two are WSL2-and-vLLM-0.27 facts, not Gemma 4 facts: they will
apply to anything else served from this venv.

## 5. Run the client

```bash
cd /mnt/e/TREC_2026
/home/aakash/gemma4/.venv/bin/python VQA/generate.answers.py --limit 20
```

| Goal | Command |
| --- | --- |
| Smoke test | `... --limit 20` |
| Full run, resumable | `...` (no flags) |
| Retry specific queries | `... --only 9 20 26 --force` |
| Re-emit the run file from cache | `... --rebuild` |
| The original Windows/Ollama path | `... --backend ollama` |

`--backend vllm` is the default and brings its own model, host, worker count,
cache and run file:

| | vllm | ollama |
|---|---|---|
| model | `google/gemma-4-12B-it-qat-w4a16-ct` | `gemma4:12b` |
| host | `http://localhost:8010` | `http://localhost:11434` |
| cache | `answers.task1.gemma4.vllm.cache.jsonl` | `answers.task1.cache.jsonl` |
| run | `gemma4.vllm.run.task1.json` | `gemma4.run.task1.json` |
| workers | 8 | 1 |

They are **separate runs, not a resumable one** — different quantisations of
the same model. Pointing both at one cache is possible but warned about.

`Ctrl+C` is safe: each query is flushed as it finishes and a re-run resumes.
The run validates the same way as the other three:

```bash
/home/aakash/gemma4/.venv/bin/python VQA/validate.ag.run.py \
  vqa.2026.testingDataset.Task1.json gemma4.vllm.run.task1.json \
  --warnings-as-errors --report-json gemma4.vllm.run.validation.json
```

### What changed in the client

Only the wire format. The prompts, the JSON schema, the per-call seeds, the
ten-phrasing ensemble, the vote, the paraphrase top-up, the fallback pool and
the `Time` accounting are all shared with the Ollama path, which is what makes
the two runs comparable. Specifically:

- frames become `image_url` data-URI parts, still **ahead of the text** — both
  because Gemma 4 wants it that way and because it keeps a query's ten prompts
  sharing a byte-identical prefix for vLLM's prefix cache;
- the schema goes in `response_format` as `json_schema`, closed with
  `additionalProperties: false` since vLLM compiles it to a grammar;
- thinking is turned off with `chat_template_kwargs: {"enable_thinking":
  false}` — Gemma 4's chat template reads that variable — rather than Ollama's
  top-level `think` flag;
- a **400 is not retried**. It means the server refused the request itself and
  will refuse it identically three times over.

### The preflight, and why it exists

`preflight()` checks `/v1/models` before any inference and stops the run if
the server is absent or serving a different model. Without it a dead server is
silent: every call fails, `fill_slots` pads all ten slots from
`FALLBACK_ANSWERS`, and the run *completes* — 2,443 queries of "unknown", a
valid file that means nothing. That happened twice during bring-up (§3) and
the only visible symptom was `FALLBACK(10)` scrolling past in the progress
line.

## 6. Throughput, measured

20 real queries, same 20 each time, on the config above:

| `--workers` | 20 queries | s/query | Projected 2,443 | Failures | Fallback |
|---|---|---|---|---|---|
| 3 | 37 s (3 queries) | 12.3 | ~8.4 h | 0 | 0 |
| 4 | 161.7 s | 8.1 | ~5.5 h | 0 | 0 |
| **8 (default)** | **123.8 s** | **6.2** | **~4.2 h** | 0 | 0 |
| 12 | 279.9 s | 14.0 | ~9.5 h | 0 | 0 |

Concurrency pays here in a way it does not for the Ollama backend (where 4
workers were *slower* than 1): vLLM's prefix cache is shared across requests
rather than being one slot, so another video's request landing in between does
not evict this video's frame prefix, and with thinking off and output capped
at 256 tokens the pass is prefill-bound, which batches well.

**8 is the knee and the far side of it is not understood.** 12 workers is more
than twice as slow as 8, and none of the obvious explanations hold: the server
log shows **zero** preemptions and zero OOM events at every setting, KV cache
usage peaked at ~50% with 16 requests running, and the client sat at ~0% CPU
throughout, so it is neither KV pressure nor client-side serialisation. It was
not chased further, because 8 is comfortably the best measured value and the
full run only needs one.

One hypothesis was tested and **rejected**: that the encoder cache — logged at
a budget of 2,496 tokens, against the 8,960 an all-frames prompt needs — was
forcing repeated re-chunking of the image embeddings. `encoder_cache_size` is
just `max_num_batched_tokens`, so `--max-num-batched-tokens 16384` raises it.
It changed nothing (167.0 s at 4 workers, 126.2 s at 8, both within noise of
the defaults) and cost two thirds of the KV cache, 98,097 tokens down to
35,301. The flag is deliberately **not** in the serve command.

**Treat the projection as ±30%.** 20 queries from the head of the file is a
small sample and the corpus has a long tail of 32-frame clips.

## 7. What was and was not checked

**Checked, against the real model:** 20 queries end to end at 3, 4 and 8
workers; 8-, 12-, 14-, 16- and 32-frame videos; a query whose video has no
frames at all; 7 with no transcript. 0 failures, 0 failed prompt calls, 0
fallback padding, 1 query needing a paraphrase top-up, peak VRAM 22,948 MiB.
Answers read sensibly — a counting question came back `0`, the
`ChainedforLife-14` clip came back "A film rehearsal", which is what the
independent Qwen3-VL pass said about the same clip.

**Checked, against a mock server** that records every request: the exact wire
format — 30 calls per query, images before text, one distinct seed each,
closed schema, `enable_thinking: false` — plus resume, `--rebuild`, `--force`,
`--limit`, and the preflight's failure path.

**Pooling behaves the same as on Ollama:** 18.4 distinct candidates pooled per
query out of 30 here, against 18.8 for the *same twenty queries* in
`answers.task1.cache.jsonl`. The ensemble is concentrating the way it was
tuned to.

**The two backends mostly agree**, which is the real evidence that the port
did not change the pass. Rank-1 answers, same 20 queries:

| Q | vLLM (QAT INT4) | Ollama (Q4_K_M) |
|---|---|---|
| 1 | The bride and groom walk towards the camera | The bride and groom walk towards each other |
| 2 | A film rehearsal | A rehearsal for a film or play. |
| 3 | 0 | zero |
| 5 | Holds it up to the camera | He shows it to the camera. |
| 9 | amusement park | Amusement park |
| 11 | putting on rings | Putting on a ring |
| 13 | The dog could get burned. | The dog could suffer from burns. |
| 17 | To clean her fur | To clean her fur |
| — | — | — |
| 4 | Recent rainfall | It is covered in lush grass. |
| 12 | To color the icing | To create a swirl effect |
| 14 | a boy on a bike | a house |
| 20 | Ready to go get the pic? | I found these over here. |

Eleven of the twenty are the same answer in different words; the rest diverge,
as two different 4-bit quantisations sampling at temperature 0.6 will. Which
of a diverging pair is *right* is unknown — see below.

**Not checked: accuracy.** No answer in this project has been scored against
ground truth for the 2026 set, and that is unchanged. Mean answer length is
4.8 words against a 2025 ground-truth mean of 8.4, the same terseness the
Qwen3-VL pass showed; whether that costs METEOR recall is a plausible
mechanism, not a finding.

## 8. The other four scripts

Every script that called `gemma4:12b` now takes `--backend`, defaulting to
`vllm`. The three text-only ones carry the same `VLLM` class minus the image
parts and the same `preflight()`, and each keeps its two backends' outputs
apart the same way:

| Script | vllm cache / out | ollama cache / out |
|---|---|---|
| `generate.answers.py` | `answers.task1.gemma4.vllm.cache.jsonl`, `gemma4.vllm.run.task1.json` | `answers.task1.cache.jsonl`, `gemma4.run.task1.json` |
| `route.query.inputs.py` | `query.routing.vllm.*` | `query.routing.cache.jsonl`, `query.routing.task1.json` |
| `generate.query.variations.py` | `query.variations.vllm.*` | `query.variations.cache.jsonl`, `query.variations.task1.json` |
| `expand.answers.py` | `--out`/`--cache` are required; name them yourself | as before |

Tested live against the server: the router returned 12 verdicts over 2
questions × 6 agents with confidences and reasons and 0 failures; the
variations script returned 10 usable paraphrases per question at 13.3/min; the
expansion script rewrote 14 of 20 answers with 0 failed calls.

### `run.eval.2025.py`

The orchestrator gained a `vllm-gemma4` backend on port 8010, so the default
path no longer touches Ollama at all and the card switches
`vllm-gemma4 → vllm-qwen → vllm-cosmos`. `--gemma4-backend ollama` restores
the old path, and `--workers-gemma` defaults to 8 on vLLM and 1 on Ollama.

Its three Gemma stages still run from **Windows** python against the WSL2
server, unlike the Qwen3-VL and Cosmos stages. That is safe only because these
clients send frames as base64 in the request body rather than handing the
server a `file://` URI — Windows `localhost:8010` forwards into the distro,
confirmed with a live request.

**One guard needed repairing as a result.** `check_flags()` refuses to run a
2025 stage that left a path flag at its default, because every cache here is
keyed by bare Q_ID and a 2025 stage pointed at a 2026 default would read 2026
answers as finished work. It finds those flags by looking for
`default=DEFAULT_X` where `DEFAULT_X = ROOT / ...` — and moving `--out` and
`--cache` into the `BACKENDS` table took them out of its sight, silently. They
are now found by `backend_path_flags()` instead, which **raises** rather than
returning nothing when it meets a `BACKENDS` table it cannot read, since
failing open is exactly how a 2025 run would overwrite `gemma4.run.task1.json`.
Verified: all three scripts are required to pass `--out` and `--cache` again,
and the orchestrator passes them under both backends.

## 9. What this unlocks, unused

`google/gemma-4-12B-it` is an *any-to-any* checkpoint: vLLM's
`gemma4_unified` takes **audio and video natively**, not just images. This
pipeline still feeds it sampled JPEG frames plus a Whisper transcript, which
is what the Ollama API forced (it has no `video` or `audio` field) and what
keeps this run comparable to the existing one. Sending the audio itself, or
the video, is now possible and is not done.

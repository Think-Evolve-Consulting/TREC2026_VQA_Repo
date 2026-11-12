# Session Notes — 2026-09-15/16 — Three more Task 2 rankers, a fusion study, and their 2026 runs

Three further vision-language models were added to the Task 2 (multiple-choice)
campaign, measured against the labelled 2025 dev sample, and then run over the
full 2026 query set. A separate study asked whether any fusion of the eight
available runs beats the best single model. Every number below was read from a
log, cache, score file or validation report on disk.

Companion notes: `SESSION.2026-09-15.task2-mc-ranking.md` (the first four
passes and the dev harness) and `SESSION.2026-09-15.muse-glimmer-task2.md`
(Muse Glimmer, the fifth ranker, dev only).

## Outcome in brief

- **Three new rankers, three valid 2026 run files**: InternVL3.5-30B-A3B,
  LLaVA-Video-7B, VideoLLaMA3-7B. Each covers all 2,443 queries and passes
  `validate_vqa_mc_2026.py --warnings-as-errors` with 0 errors and 0 warnings.
- **None of them reaches Qwen3.8** on the 400-query dev sample (0.800 Top-1 /
  0.884 MRR). The closest, InternVL3.5, is 4.0 points behind (p = 0.064).
- **No fusion beats Qwen3.8 either.** A search over 781,248 weighted fusions
  found an in-sample winner at 0.825, but under nested cross-validation it is
  worth 0.801 — the baseline. The in-sample gain is search optimism.
- **The recommended five submissions are unchanged**: qwen38, qwen3vl,
  fused.weighted, gemma4.vllm, fused.rrf. Twelve valid MC files now exist and
  at most five may be submitted; the MC window closes 18 Sep 2026.
- Qwen3.8's errors were characterised: the only question type that stands out
  is **counting**, where every model in the set systematically undercounts.

## Dev results (400 queries, seed 2026, official scorer)

| run | Top-1 | MRR | paired vs Qwen3.8 |
|---|---|---|---|
| Qwen3.8-27B (baseline) | **0.800** | **0.884** | — |
| Qwen3-VL-8B | 0.790 | 0.872 | −1.0, p = 0.70 |
| **InternVL3.5-30B-A3B Q4** | **0.760** | **0.859** | −4.0, p = 0.064 |
| Muse Glimmer 30B Q4 | 0.755 | 0.856 | −4.5, p = 0.018 |
| **VideoLLaMA3-7B bf16** | **0.738** | **0.843** | −6.2, p = 0.004 |
| Gemma 4 12B | 0.733 | 0.840 | −6.8, p = 0.0005 |
| **LLaVA-Video-7B bf16** | **0.725** | **0.835** | −7.5, p = 0.0008 |
| Cosmos-Reason2-2B | 0.650 | 0.783 | −15.0, p ≈ 0 |
| random control | 0.253 | 0.517 | −54.8 |

Rank distribution behind the MRR figures (the correct answer is always in the
list, because every run emits a full permutation):

| run | rank 1 | rank 2 | rank 3 | rank 4 | answer not among options |
|---|---|---|---|---|---|
| Qwen3.8 | 320 | 50 | 19 | 9 | 2 |
| InternVL3.5 | 304 | 58 | 19 | 17 | 2 |
| VideoLLaMA3 | 295 | 60 | 18 | 25 | 2 |
| LLaVA-Video | 290 | 61 | 22 | 25 | 2 |

Dev MRR is on 4-option queries, where chance is 0.521. The 2026 set is 81%
8-option, where chance is 0.340, so the two are not comparable.

## The three new passes

### InternVL3.5-30B-A3B (Q4_K_M, llama.cpp) — `VQA/rank.options.mc.muse.py`

The request was "InternVL 3 26B at 4-bit". **InternVL3 has no 26B size** — the
line is 1/2/8/9/14/38/78B, and the 26B models are InternVL2 and 2.5, whose
official AWQ builds are 23.3 GB because the 6B vision tower stays fp16. Nothing
in that family fits a 24 GB card with room for frames. The user chose
**InternVL3.5-30B-A3B** instead: an MoE with 3B active parameters and a small
InternViT-300M tower, as the bartowski `Q4_K_M` GGUF (18.63 GB) plus an f16
mmproj (0.64 GB).

No new client was needed — the Muse llama-server script serves it with flags:

```
python VQA/rank.options.mc.muse.py --model internvl3.5-30b-a3b \
    --host http://localhost:8013 --frame-size 448 --temperature 0.2 \
    --num-predict 256 --workers 2
```

**The tiling trap.** llama.cpp cuts every InternVL image into 448-px tiles: one
504x896 frame cost ~1,280 tokens, 16 frames came to 20,945, and every image
request returned HTTP 400 against a 16k slot. `--image-max-tokens` does not cap
it for this architecture. The fix is client-side and matches InternVL's own
video path (`max_num=1`): the new `--frame-size 448` resizes each frame to a
single tile, giving 4,561 prompt tokens. Unlike Muse, its image prefix *is*
reused across passes (4,324 of 4,561 tokens cached on pass 2).

Server (21.5 GB VRAM, 2 slots of 16k, q8 KV):

```
llama-server -m OpenGVLab_InternVL3_5-30B-A3B-Q4_K_M.gguf \
  --mmproj mmproj-OpenGVLab_InternVL3_5-30B-A3B-f16.gguf --jinja \
  -ngl all -fa on -c 32768 -np 2 --cache-type-k q8_0 --cache-type-v q8_0 \
  --alias internvl3.5-30b-a3b --port 8013 --no-webui
```

### LLaVA-Video-7B (bf16, vLLM) — `VQA/rank.options.mc.llavavideo.py`

`Isotr0py/LLaVA-Video-7B-Qwen2-hf`, a conversion of the lmms-lab checkpoint to
the `llava_onevision` architecture, which is the only form vLLM can serve. It
is the only pass here that sends the **clip itself** (`video_url`, 32 frames
sampled server-side, ~6,300 prompt tokens); the frame index is read only for
each clip's duration. Runs from WSL2 against vLLM 0.12 on port 8009
(`VQA/serve.llavavideo.sh`, `VQA/run2026.llavavideo.sh`).

**The .webm frame-count trap.** OpenCV reports one frame more than it can
decode for 256 of the 1,088 2026 `.webm` files (251 off by one, 5 by 2–4; every
`.mp4` was exact). vLLM samples `linspace(0, n-1, 32)`, so the last index is
never reached. Checked against vLLM v0.12.0 `video.py`: it only warns and uses
the frames it got — worst case 30 of 32 — so no re-encoding was needed. Full
scan: `videos_TEST_VQA/webm_framecount_report.json`.

It reports **confidence 0.0 on every reply**, which flattens its Borda weights.
Harmless, but its confidence carries no information for fusion.

### VideoLLaMA3-7B (bf16, in-process) — `VQA/rank.options.mc.videollama3.py`

vLLM has no VideoLLaMA3 architecture (issue #16627), and the model's remote
code pins **transformers 4.46.3**, so it runs in-process from a dedicated venv
(`.venv-videollama3`: Python 3.11, torch 2.6.0+cu124, numpy<2, decord).
Attention is `sdpa` — flash-attn cannot be built here, and the remote code
falls back cleanly.

**Frames with real timestamps.** Its loader stamps a *directory* of frames at a
fixed `vid_fps = 3`, which would mislabel every frame. Passing a `video` item
that is already a list of CHW arrays skips the loader entirely; `num_frames`
and `timestamps` are then ours, and the chat template prints the true
`Time 0.6s:<image>, Time 1.7s:<image>, …`. Verified with the real processor
before the weights were even loaded.

**The sdpa vision encoder spills the card — the one real failure of the
session.** It builds a full `[S, S]` boolean mask over every patch in the clip
and hands it to `scaled_dot_product_attention`, which then materialises
heads × S × S scores: about 8 GB per layer at 16k patches. Windows spills the
overflow into system RAM instead of raising OOM, so the first smoke test sat at
**23.7 GB and 100% GPU for over 10 minutes with zero queries finished**. The
mask is block-diagonal (one block per frame), so the script monkeypatches
`VisionSdpaAttention.forward` to attend one `cu_seqlens` block at a time. The
patch is exact — max abs difference **0.00000** against the masked original on
a real layer — and the run went to 7.1 queries/min. The visual budget is also
capped at `max_tokens = 4096` (252 tokens per frame at 16 frames).

## Fusion search with nested cross-validation — `VQA/search.fusion.mc.dev.py`

The question: does any combination of the eight dev runs beat Qwen3.8? An honest
answer must guard against the search itself, which was the flaw in the 6 Sep
8:8:4:1 result (weights fitted on the same 400 queries they scored on).

The script fuses by RRF (k=60) and Borda over every subset and every integer
weighting 0..4 of eight runs — 781,248 fusions — and judges them by choosing the
best on nine folds and scoring on the held-out tenth, over 20 random splits. It
first proves its own scoring reproduces the official scorer query by query for
all eight single runs.

| search | best on the full 400 | nested CV | CV repeats above baseline |
|---|---|---|---|
| weight grid, 781,248 fusions | 0.825 / 0.896 (qwen38:3 qwen3vl:3 internvl:4 muse:2 videollama3:1) | **0.801 / 0.883** | 9/20, median p = 0.87 |
| equal-weight subsets, 510 | 0.810 / 0.890 | 0.778 / 0.871 | 0/20 |

Fusions named in advance (no search, so in-sample is honest): top-4 equal Borda
0.810 (+1.0, p = 0.57); top-3 equal RRF 0.807 (+0.7, p = 0.74); all eight equal
RRF 0.790; qwen38 + qwen3vl 0.782.

**Why fusion cannot help here.** At least one run puts the right answer first on
93.5% of queries, but every model loses more to Qwen3.8 than it wins:

| model | right where Qwen3.8 is wrong | wrong where Qwen3.8 is right |
|---|---|---|
| Qwen3-VL | 29 | 33 |
| InternVL3.5 | 25 | 41 |
| VideoLLaMA3 | 23 | 48 |
| LLaVA-Video | 23 | 53 |
| Muse | 17 | 35 |
| Cosmos | 26 | 86 |

## Where Qwen3.8 is wrong (80 of 400 dev queries)

Tested across ~28 features — question type, clip length, transcript presence,
how the video was delivered, answer position, pass agreement — with exact
Fisher tests and a Bonferroni threshold of p < 0.0018.

- **Counting is the only question type that stands out**: 43.8% error against
  16.8% elsewhere (48 queries, p = 0.0001), and "all options numeric" 41.0%.
  **The misses are systematic undercounts**: of 18 readable cases, 15 too low
  and 3 too high, mean −1.2. This is universal — every model undercounts
  (mean −0.4 to −1.3) — and Qwen3.8 is the *second best* at counting (44%
  error against Qwen3-VL's 38%).
- **Split passes predict errors**: 46.4% error on the 69 queries where its three
  passes disagreed, against 14.5% when unanimous. Every other model also errs
  48–59% on those, so they are hard for everyone, not a Qwen3.8 blind spot —
  useful as a confidence signal, useless as a routing signal.
- **Answer-length artefact**: when the correct answer is the longest option
  (186/400), error is 11.8% against 27.1%. A property of the 2025 option
  writing, not of the videos.
- **Not significant**: clip length, transcript presence, speech/sound questions,
  on-screen text, why/what-happens types, answer position.
- 26 queries are missed by all eight models (counting, audio-only, fine detail,
  and a few doubtful labels). 18 are missed only by Qwen3.8; the one theme is
  **"what can be heard" questions answered from silent video**.

## The 2026 runs

Run unattended at the user's instruction (they could not run commands), with
routing on as in every other 2026 pass, by `VQA/run2026.extra2.ps1`.

| run file | ranked | wall clock | validator | unanimous | repairs | rank 1 on first option (chance 14.9%) |
|---|---|---|---|---|---|---|
| `internvl.run.task2.json` | 2,443/2,443 | 1.99 h | **VALID** | 80.8% | 13 | 16.2% |
| `llavavideo.run.task2.json` | 2,443/2,443 | 1.85 h | **VALID** | 64.5% | 0 | 16.0% |
| `videollama3.run.task2.json` | 2,443/2,443 | 6.12 h GPU | **VALID** | 70.0% | 6 | 14.9% |

All three: 0 failed passes, 0 fallbacks, 0 truncations. In each run the router
sent 116 queries to the transcript alone and gave 68 their frames back because
the clip has no speech.

## Infrastructure lessons

| symptom | cause | fix |
|---|---|---|
| A detached run dies mid-pass, exit `-1073741510` | `STATUS_CONTROL_C_EXIT`: a `Start-Process` child is killed when the Claude session ends. Cost ~1.4 h twice | Run it as a Windows scheduled task, and make every pass resumable from its cache |
| Scheduled retries silently skipped, result `0x800710E0` | A dead instance still counted as running, so `/SC MINUTE /MO 15` refused new starts | `Set-ScheduledTask -Settings (New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 999 -RestartInterval 5min -MultipleInstances IgnoreNew)` |
| `grep`/`tail` watchers see nothing in a live log | PowerShell 5.1 `Tee-Object -FilePath` writes **UTF-16LE** | Watch the UTF-8 JSONL cache instead (`"fallback": true`, `"n_pass_failures": N`), or write logs via `cmd /c … >` |
| GPU at 100% and 23.7 GB with no progress | Windows spills VRAM overflow into system RAM instead of raising OOM | Cap the visual budget and fix the kernel (see VideoLLaMA3 above); treat "busy but no output" as a memory symptom |
| Inline bash through `wsl.exe` mangled | PowerShell 5.1 strips inner quotes | Put the command in a `.sh` file with LF endings and run `bash /mnt/e/…` |
| Re-implemented MRR disagreed with the official scorer on 19/400 | The scorer rounds each reciprocal rank to 3 dp (`round(1/rank, 3)`) | Round the same way; verify per query against the diagnostic CSVs before trusting any derived number |
| Cosmos service grabs ~15 GB whenever WSL2 starts | It is enabled and auto-starts with the distro | `systemctl --user stop cosmos-reason2` *before* the GPU-free check |

## Qwen3.8-Flash-Next — assessed, not attempted

Asked whether it could run here or on a 32 GB Mac. It is 125B parameters plus a
51B n-gram embedding table (6B active, 512 experts, multimodal, 262k context).
Smallest builds: REAP-pruned Q3 61.9 GB, Unsloth 2-bit 78.9 GB, MLX mixed 2-bit
73.1 GB, MLX 4-bit 111.5 GB.

- **32 GB Mac: no.** The smallest Mac build needs 73 GB of unified memory. 96 GB
  would run mixed 2-bit, 128 GB the proper 4-bit.
- **This box (24 GB VRAM + 63.7 GB RAM): loads, but useless for this workload.**
  Prefill of ~4,500 visual tokens would run mostly on CPU, likely 30–120 s per
  pass against InternVL's 0.9 s — days for 7,329 calls.
- **The practical route is the hosted API** (~$0.16/1M in, $0.47/1M out ≈ $5–6
  for a full Task 2 pass), which means sending TREC frames to a third party —
  the user's decision, and a rules question.

## What can be submitted (MC, max 5, window closes 18 Sep)

Twelve valid files exist. Recommended five, unchanged by this session:
**qwen38 → qwen3vl → fused.weighted → gemma4.vllm → fused.rrf**. The three new
runs are valid but rank below every current pick except Gemma 4 on dev.

## Inventory

| file | lines | purpose |
|---|---|---|
| `VQA/rank.options.mc.llavavideo.py` | 1,174 | LLaVA-Video pass (vLLM, native video) |
| `VQA/rank.options.mc.videollama3.py` | 1,210 | VideoLLaMA3 pass (in-process, block-wise attention patch) |
| `VQA/search.fusion.mc.dev.py` | 412 | Fusion search with nested CV |
| `VQA/rank.options.mc.muse.py` | 1,260 | Muse pass, now also the InternVL client (`--frame-size`) |
| `VQA/run.dev.extra3.ps1` | 108 | The three dev passes in sequence, then scoring |
| `VQA/run2026.extra2.ps1` | 144 | The three 2026 passes, unattended, with validation |
| `VQA/serve.llavavideo.sh` / `dev.llavavideo.sh` / `run2026.llavavideo.sh` | 18 / 12 / 10 | WSL2 serve and client wrappers |

Artefacts: `{internvl,llavavideo,videollama3}.run.task2.json` and their
`.validation.json`; caches `rankings.task2.*.cache.jsonl`; dev runs, scores and
diagnostics in `eval.2025.mc/`; `eval.2025.mc/fusion.search.json`;
`videos_TEST_VQA/webm_framecount_report.json`; logs in `logs/`
(`run2026.extra2.log`, `rank.options.mc.task2.*.log`, `dev.mc.*.log`).
Models: `models/InternVL3_5-30B-A3B-GGUF` (19.3 GB),
`models/VideoLLaMA3-7B` (16.1 GB), LLaVA-Video in the WSL2 HF cache (16.1 GB).

Memory notes updated: `task2-extra-vlm-rankers`, `task2-mc-dev-harness`,
`muse-glimmer-answering-pipeline`.

## Open items

1. Choose and submit ≤ 5 MC runs before 18 Sep; confirm the data agreement form.
2. E: is at **37 GB free**. `models/Muse-Glimmer-30B/` (55.5 GB bf16) is unused
   and deletable; the Muse GGUF (19.7 GB) too unless Muse is revisited.
3. A ~1-point true gain cannot be resolved on 400 queries. Settling one means
   extending the same sample toward all ~1,980 usable 2025 queries, which means
   re-running every model.
4. Counting questions are the one measured weakness shared by every model. A
   count-specific prompt or denser frames is untested — 48 dev queries is too
   few to measure it.

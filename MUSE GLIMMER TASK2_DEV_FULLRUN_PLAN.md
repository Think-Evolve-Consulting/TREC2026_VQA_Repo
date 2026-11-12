# Muse Glimmer 30B as a Task 2 (MC) ranking pass

## Context

Muse Glimmer was dropped on 14 Aug (SESSION.2026-08-14.muse-glimmer.md) because of the
Task 1 deadline, not because of any hard blocker. Today is 15 Sep, and the **MC window
closes 18 Sep** (≤5 runs). Task 2 is a much cheaper job than Task 1: each pass returns a
permutation of option numbers, not ten free-text answers. The best measured ranker so far
is Qwen3.8-27B at 0.800 Top-1 / 0.884 MRR on the 400-query 2025 dev sample. No fusion
beat it. Muse is worth a slot only if it matches Qwen3.8 on dev or makes a weighted fusion
better. So we measure on dev first, then run 2026.

**Current state, verified this session:**
- No Muse-capable llama.cpp on the box. The PATH `llama-server` is the winget CPU build,
  and the old b9050 CUDA build is gone. Support shipped in **b10358**. Latest is **b10976**,
  which has a Windows zip: `llama-b10976-bin-win-cuda-12.4-x64.zip` plus
  `cudart-llama-bin-win-cuda-12.4-x64.zip`. This is a prebuilt download, so no compiler is needed.
- The GGUF was deleted. Unsloth `unsloth/Muse-Glimmer-30B-GGUF` has
  `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` (14.8 GB) and `mmproj-Muse-Glimmer-30B-BF16.gguf`
  (3.6 GB). E: has 101 GB free. The 55.5 GB bf16 checkpoint is still at
  `models/Muse-Glimmer-30B/` (we only need its `chat_template.jinja` and `processor_config.json`).
- **The GPU has 20.7 GB in use** while every project WSL distro is stopped. The only
  running distro is `salad-enterprise-linux` (SaladCloud), which is the likely holder.
  It must be stopped for the run.

**Model facts that shape the design (from the local checkpoint, Unsloth, PR #26841):**
- Chat template: when no strength is passed, `reasoning_strength` defaults to **high**, and
  the template renders it into the system prompt. We must send it on every request.
- The model always reasons first. llama.cpp separates `reasoning_content` from `content` with
  `--reasoning-format auto`, and a small `max_tokens` returns **empty content**. Gemma's 128-token budget will not work.
- `max_image_tokens` is 4096 per image (patch 14, merge 2). Frames are 504×896, so ≈576
  tokens each at native size. With a mean of 13.5 and a max of 32 frames, that is 7.8k and
  18.4k tokens. **Cap them with `--image-max-tokens 256`**, which gives mean ≈3.5k and max ≈8.2k,
  about the 4k visual budget the Qwen runs used.
- The DFlash speculative drafter can't be used together with vision, so we don't use it. Reusing
  the prompt cache across requests with identical images is reportedly still unsolved upstream,
  so plan for all 3 passes paying the full image prefill. The smoke test measures this.
- The model card's sampling settings are temperature 1.0, top_p 0.95, top_k 64.

## Approach

### 1. Serve (Windows native, no WSL)
- **Precondition (you do this):** pause SaladCloud. Before loading the model, confirm that
  `nvidia-smi` shows under 1.5 GB used. Keep the Ubuntu-24.04 vLLM services stopped too (the
  `cosmos-reason2` unit auto-starts with that distro and takes about 16 GB).
- Extract b10976 CUDA 12.4 plus cudart to `E:\TREC_2026\tools\llama-b10976\`. Always invoke it
  by absolute path (the winget CPU build shadows it on PATH).
- `hf download unsloth/Muse-Glimmer-30B-GGUF` with the UD-Q4_K_XL and mmproj BF16 files → `models\Muse-Glimmer-30B-GGUF\`.
- Server (port 8012):
  ```
  E:\TREC_2026\tools\llama-b10976\llama-server.exe `
    -m E:\TREC_2026\models\Muse-Glimmer-30B-GGUF\Muse-Glimmer-30B-UD-Q4_K_XL.gguf `
    --mmproj E:\TREC_2026\models\Muse-Glimmer-30B-GGUF\mmproj-Muse-Glimmer-30B-BF16.gguf `
    --jinja --reasoning-format auto -ngl all -fa on `
    -c 49152 -np 3 --image-max-tokens 256 `
    --temp 1.0 --top-p 0.95 --top-k 64 `
    --alias muse-glimmer-30b --port 8012 --no-webui
  ```
  That gives 16k per slot: ≤8.2k image, ~1k text, ~2k reasoning. Check VRAM at startup. If it
  OOMs, go to `-np 2` or `mmproj-...-Q8_0.gguf` (1.9 GB).
- Confirm that `/props` reports `vision: true`.

### 2. New script `VQA/rank.options.mc.muse.py`
Copy `VQA/rank.options.mc.py` and keep its ranking contract untouched: `ranking_schema`,
`presentation_orders`, `as_permutation`, `borda`, `check_permutation`, `plan_for`,
`load_transcript`, `assemble`, `report`, the cache/resume/`--only/--force/--rebuild` logic,
grouping by video, and the zero-evidence rescue. Then change only these:
- Replace the backends with one `LlamaCpp` client, adapted from `MuseClient` in
  `VQA/generate.answers.muse.py`. It sends `chat_template_kwargs: {"reasoning_strength": ...}`
  and `cache_prompt: true`, reads `content` and `reasoning_content` separately, and records
  reasoning length, `finish_reason` and `timings`, or `usage`, into each pass.
- `response_format` json_schema is on by default. The smoke test must show that llama.cpp applies
  the grammar only after the reasoning phase. If it applies it from the first token, the reasoning
  gets crushed: fall back to `--no-schema` (the prompt still demands JSON, and `parse_json_object`
  plus `as_permutation` already tolerate prose).
- **Empty content / `finish_reason=length`** gets one retry at 2× `max_tokens`, and is counted
  in the cache. We never silently fall back.
- Reasoning strength defaults to `low` for every query, with `--num-predict` defaulting to 2048.
  An optional `--strength-policy routed` borrows the Task 1 mapping from
  `generate.answers.muse.py` (neither flag → low, one → medium). It stays off by default
  because the dev set runs `--ignore-routing`, and only a matching policy on 2026 is validated by dev.
- Temperature defaults to 1.0 (model card). Diversity comes from 3 permuted passes plus Borda, as before.
- Frames go in the same order as Gemma's (images first, then the text context), so the prefix is
  stable if llama.cpp ever reuses the image cache. The `[mm:ss]` timestamps are already in `build_context`.
- `preflight()` hits `/props` and refuses to start if vision isn't true, the alias isn't served,
  or the per-slot `n_ctx` is < 12k. The last one would be a silent truncation or 400 storm.
- Defaults: out `muse.run.task2.json`, cache `rankings.task2.muse.cache.jsonl`, workers = 3
  (matches `-np`).

### 3. Gate on dev (400 queries, 4 options, seed 2026)
```
python VQA/rank.options.mc.muse.py --queries eval.2025.mc/sample400.seed2026.queries.json `
  --frames videos_GT_2025/frames --transcripts videos_GT_2025/transcription_openwhisper `
  --ignore-routing --out eval.2025.mc/muse.dev.run.json --cache eval.2025.mc/muse.dev.cache.jsonl
python VQA/eval.mc.dev.py score --run eval.2025.mc/muse.dev.run.json --baseline eval.2025.mc/qwen38.dev.run.json
```
Then score a weighted fusion `qwen38 + qwen3vl + muse` on dev with `VQA/fuse.rankings.mc.py`.
**Decision:** go to the full 2026 run only if Muse is statistically tied with Qwen3.8
(p > 0.05), or if a fusion that includes it beats 0.800. Otherwise stop and record the numbers.

### 4. Full 2026 run (the user executes it; see memory note long-batch-runs-user-executes)
`python VQA/rank.options.mc.muse.py` (routing file on, fixed `low` strength), then
`validate_vqa_mc_2026.py --warnings-as-errors`, then pick the 5 submissions before 18 Sep.

## Timeline (all estimates; the 20-query smoke replaces them)
- Setup and download (~19 GB): 1–2 h. Smoke test on 20 dev queries: 15 min.
- Dev 400: ~3–4 h if it's ~30 s/query at 3 slots.
- 2026 full, 2,443 queries × 3 passes: **~20–25 h**. Decode dominates (~300–800 reasoning tokens per pass at low).
  If the smoke test projects more than ~30 h, drop to `--passes 2` (official + reverse still control position bias).
- The latest safe start for the 2026 run is about the morning of 17 Sep.

## Verification
1. Server startup log: model plus mmproj loaded, `vision: true`, VRAM < 24 GB with 3 slots.
2. `--limit 20` on dev: 0 empty-content passes, 0 repaired, JSON parsed from `content`.
   Also confirm each record's reasoning length is non-trivial and that the reasoning strength
   took effect: compare reasoning token counts at `low` vs `medium` on the same 3 queries.
3. Blind-model check: re-run 5 of those 20 queries with images stripped. If the rankings don't
   change, vision isn't reaching the model.
4. Dev score vs the random control (0.253) and vs Qwen3.8 (paired McNemar).
5. The full run file passes `validate_vqa_mc_2026.py --warnings-as-errors`. `report()` shows
   position bias near chance and 0 fallbacks.
6. Update the memory notes (muse-glimmer-answering-pipeline, task2-mc-dev-harness) and write a
   short SESSION note.

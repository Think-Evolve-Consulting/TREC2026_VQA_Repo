# Session Notes — 2026-09-15 — Muse Glimmer 30B as a Task 2 ranker (not submitted)

Muse Glimmer, dropped for Task 1 on 14 Aug (`SESSION.2026-08-14.muse-glimmer.md`),
was revived as a Task 2 (multiple-choice) ranking pass and measured on the labelled
2025 dev sample before any 2026 run. **It failed the gate, so no 2026 run was made.**
The recommended five MC submissions in `SESSION.2026-09-15.task2-mc-ranking.md` are
unchanged.

## Result (400-query dev sample, seed 2026)

| run | Top-1 | MRR | vs Qwen3.8 (paired) |
|---|---|---|---|
| Qwen3.8-27B (baseline) | 0.800 | 0.884 | — |
| Qwen3-VL-8B | 0.790 | 0.872 | −1.0, p = 0.70 |
| **Muse Glimmer 30B, low reasoning, 3 passes** | **0.755** | **0.856** | **−4.5, p = 0.0175** (17 won / 35 lost); MRR −0.028, CI −0.048 to −0.008 |
| RRF Qwen3.8 + Qwen3-VL + Muse, 8:8:4 | 0.800 | 0.884 | +0.0, p = 1.0 (17 / 17), in-sample weights |
| Gemma 4 12B | 0.733 | 0.840 | −6.8, p = 0.0005 |
| Cosmos-Reason2-2B | 0.650 | 0.783 | −15.0 |

Gate: go to 2026 only if Muse ties Qwen3.8 or a fusion that includes it beats 0.800.
**Both conditions failed.** Muse is significantly worse, and adding it to the Qwen pair
changes nothing.

Run quality was clean: 0 failed passes, 0 repairs, 0 fallbacks, 0 budget retries, and
86.2% of queries unanimous across passes. Rank 1 is on the officially-first option in
32.2% of queries, against 27.8% where the correct answer really sits first in this
sample. Qwen3.8 and Qwen3-VL both score 31.2% on the same check, so Muse has no bias
of its own.

## Setup

- llama.cpp **b10976** CUDA 12.4 prebuilt → `tools/llama-b10976/` (the architecture
  first shipped in b10358). The `llama-server` on PATH is still the winget CPU build.
- `unsloth/Muse-Glimmer-30B-GGUF`: `UD-Q4_K_XL` (15.88 GB) + `mmproj-BF16`
  (3.85 GB) → `models/Muse-Glimmer-30B-GGUF/`.
- Server: `-c 49152 -np 3 --image-max-tokens 256 --jinja --reasoning-format auto
  -fa on -ngl all`, port 8012, alias `muse-glimmer-30b`. **21.5 GB VRAM**,
  3 × 16,384 context. The full command is in the script's docstring.
- SaladCloud (`salad-enterprise-linux` WSL distro) had been holding 20.7 GB and had to be paused first.

## Script — `VQA/rank.options.mc.muse.py`

A copy of `rank.options.mc.py` with the ranking contract untouched. Changes:

- a llama-server client that sends `chat_template_kwargs.reasoning_strength` on every call;
- a 2,048-token budget, with one retry at 2× on `finish_reason=length`;
- per-pass `cost` (prompt, cached and completion tokens, prefill and decode time) in the cache;
- a `/props` preflight that refuses to start with no vision, a wrong alias, or less than 12k context per slot;
- `--strength` and `--strength-policy routed` (unused), and `--no-schema`.

It was tested against a mock server first: prose-wrapped JSON, the forced 2× retry, and both preflight refusals.

## What the probe measured (one 16-frame query, live server)

| check | measured |
|---|---|
| image cap | 16 frames = 4,476 prompt tokens against 365 for text only, so ~256 per frame and vision is live |
| JSON schema + reasoning | the trace is intact (1.5–2.3k chars) and the content is valid JSON, so the schema stays on |
| strength | `low` produced 389–548 completion tokens, `medium` 1,142 |
| image prefix cache | an identical repeat reused 4,475 tokens; a reordered pass reused only 47 (the system prompt) |
| decode | 43 tok/s single stream, ~22 tok/s per slot at 3 slots |

The template defaults to reasoning strength **high** when the kwarg is absent, and
`max_image_tokens` is 4,096 per image natively. Either one, left alone, would have made
the run silently expensive or overflowed the context.

## Cost

Dev 400 × 3 passes ran at 2.5 queries/min on 3 slots, **~2.7 h wall**. A pass averaged
3,918 prompt tokens (1,157 from cache), 374 completion tokens, 4.5 s of prefill and
18.7 s of decode. Projected 2026: ~16–20 h. `medium` strength roughly doubles the decode
(~40 h for 2026) and was not tried, because it cannot finish before the 18 Sep close.

## State left on disk

| path | size | note |
|---|---|---|
| `models/Muse-Glimmer-30B-GGUF/` | 19.7 GB | keep only if Muse is revisited |
| `models/Muse-Glimmer-30B/` | 55.5 GB | bf16; nothing uses it now — delete |
| `tools/llama-b10976/` + two zips in `tools/` | — | the zips can go |
| `eval.2025.mc/muse.dev.*`, `fused.muse.dev.*` | — | dev run, cache and scores |
| `logs/muse.server.log`, `logs/dev.mc.muse.smoke.log` | — | |

The server is stopped.

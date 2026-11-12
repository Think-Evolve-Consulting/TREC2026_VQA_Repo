# Session Notes — 2026-09-17 — The orchestrator judge over the ranker swarm

Eight models had already ranked every Task 2 query independently. This session
built and measured the **orchestrator pass**: where the panel disagrees, one
judge model re-reads the video, sees every panellist's ranking and a card of
each panellist's known weaknesses, and decides the ranking itself.

**Outcome: measured, failed its pre-registered gate, not promoted to 2026.**
The submission stays as recommended in `SESSION.2026-09-16.task2-extra-rankers.md`.

## Result

On the 400-query 2025 dev sample. The judge only acts where the panel disagrees
on rank 1 — 203 of 400 queries — because the other 197 are already 95% correct
and re-judging them could only break answers that are right.

| Arm | Right of 203 disagreements | Full 400 Top-1 | Paired vs Qwen3.8 |
|---|---|---|---|
| **Qwen3.8 alone (baseline)** | **133** | **0.800 / 0.884** | — |
| Qwen3.8 as judge | 132 | 0.797 / 0.881 | −0.2, p = 1.00 |
| InternVL3.5 as judge | 130 | 0.792 / 0.879 | −0.8, p = 0.78 |
| InternVL3.5 as judge, **priors off** | 127 | 0.785 / 0.875 | −1.5, p = 0.50 |
| Majority vote (control) | 129 | 0.790 / 0.879 | −1.0, p = 0.58 |
| Null judge — pick among candidates (control) | 73 | 0.650 / 0.807 | −15.0, p ≈ 0 |
| *perfect adjudicator (ceiling)* | *187* | *0.935* | *+13.5* |

**The gate**, fixed before any number was read: beat Qwen3.8 on the 400 at
exact McNemar p < 0.05 **and** gain ≥ 8 queries on the 203. Nothing came close,
so no 2026 judge run was made.

Both controls landed where they had to, which is what makes the negative result
trustworthy: the null judge is 60 queries below the baseline (the harness is not
handing out wins), and majority voting is *below* the baseline (the panel has no
majority worth harvesting).

## Why it fails, precisely

The judges are not broken and not timid. Qwen3.8 overruled the baseline on 39 of
203 queries and InternVL on 61 — and **the overrules are a coin flip**:

| Judge | Overruled | Became right | Became wrong |
|---|---|---|---|
| Qwen3.8 | 39 | 16 | 17 |
| InternVL3.5 | 61 | 24 | 27 |

Four real decisions, with the judge's own one-line reason:

- **Won, Q358** "what happens before the characters appear?" Panel: *2 rays of
  light*. Judge: *3 rays* — "Frames 3-6 clearly show three distinct streaks
  (purple, red, blue)". The undercount prior from the scorecard, working.
- **Won, Q177** "besides background voices what can be heard?" Panel: *revving
  motorcycles*. Judge: *honking horns and police whistles*, inferred from
  visible security presence and a busy street. It cannot hear; it reasoned from
  pictures and was right.
- **Lost, Q14** "what happens after her phone gives an alert?" The judge wrote
  "The man looks at her with a displeased expression" — which *is* the correct
  answer — then picked the option "The boyfriend ignores the alert."
  Right perception, wrong selection.
- **Lost, Q115** "does the video answer 'Who created Allah?'" Panel: *No*.
  Judge: *Yes, in a manner of speaking*, arguing the baker analogy answers it by
  showing a category error. Defensible; not what the annotator marked.

**The headroom is real but not reachable from these frames.** At least one
panellist is right on 187 of the 203, yet no judge with the same 16 frames can
tell which one. The disagreements are dominated by cases the whole panel finds
hard — counting across time, audio-only questions asked of silent video, and
fine visual detail — which is the same conclusion the error analysis reached
from the other direction on 16 Sep.

**The priors help a little and are not what is missing.** Removing the
reliability card costs 3 queries (130 → 127), which also means the primed arms
are mildly optimistic: that card is distilled from the same 400 queries the arms
are scored on.

## How the judge works

`VQA/judge.rank.mc.py`, one call per disagreeing query, no interaction between
judges and none with the panel at run time:

1. **Disagreement detection** — candidates are the distinct rank-1 options
   across the panel's run files. One candidate → copy the consensus through, no
   call. Measured: 197 unanimous, 203 disagreeing (50.8%).
2. **What the judge sees** — 16 frames at 448×448 with `[mm:ss]` labels, the
   question, the numbered official options, a per-option table of who ranked it
   where (plus which panellists' own passes disagreed), the reliability card,
   and the router's verdict where one matches.
3. **What it returns** — a schema-constrained `{"ranking": [...],
   "confidence": …, "why": …}`. `as_permutation()` repairs, `check_permutation()`
   refuses anything the validator would reject; any failure falls back to the
   baseline's ranking for that query.
4. **Two run files per pass, free** — the judge's full permutation, and the
   judge's rank 1 with the baseline's tail. They scored identically here.
5. **Controls in the same script** — `--strategy majority` and `--strategy null`
   need no GPU and emit identically shaped runs.

Both judges were given **identical evidence** (the same 16 frames as images), so
the ablation isolates the model. That choice also side-steps Qwen3.8's
`do_sample_frames` trap, which only applies to its native video path.

## Serving traps — three, all in the Qwen3.8 arm, all silent

Judging with **images** rather than video compiles a FlashInfer kernel variant
(`head_dim_qk_256`, FP8 KV) that the September video runs never needed, so none
of this was in the cache. Every one of these kills the engine on the **first
real request**, after `/v1/models` has already answered 200 — so a client
preflight passes and then every query falls back, and the run exits 0 looking
finished while being exactly the baseline.

| Symptom | Cause | Fix |
|---|---|---|
| `404 /props` at preflight | The Muse pass's preflight is llama.cpp-only; vLLM has no `/props` | Backend-aware preflight (`/v1/models` for vLLM, keeping the vision check for llama.cpp) |
| `Could not find nvcc … cuda_home='/usr/local/cuda'` | FlashInfer JIT-builds on first request; `CUDA_HOME` unset in a bare shell | Export `CUDA_HOME`/`CUDA_PATH` to the in-venv cu13 toolkit, as `~/.config/systemd/user/qwen38count-vllm.service` documents |
| `FileNotFoundError: 'ninja'` | FlashInfer shells out to ninja, which lives in the venv `bin` | Put the venv `bin` on `PATH` (calling vllm by absolute path is not enough) |
| `CUDA compiler and CUDA toolkit headers are incompatible` | That kernel variant will not build here at all | `--attention-backend TRITON_ATTN --kv-cache-dtype bfloat16` — `auto` still resolves to FP8, which Triton refuses on SM86 |

Cost of the workaround: **18.1 s per judge call** against InternVL's 5.4 s on
llama.cpp, and KV cache 38,912 tokens instead of 69,046.

The judge script now **aborts after 10 consecutive fallbacks**. Before that
guard existed, one dead engine burned 15.7 minutes producing a run identical to
the baseline.

## Files

| File | Lines | Purpose |
|---|---|---|
| `VQA/judge.rank.mc.py` | ~700 | The orchestrator pass; imports the ranking contract from `rank.options.mc.muse.py` |
| `VQA/serve.judge.qwen38.sh` | 27 | vLLM 0.27 for the Qwen3.8 judge, carrying all four fixes above |

Artefacts in `eval.2025.mc/`: `judge.{qwen38,internvl,internvl.noprior,majority,null}.dev.run.json`
(each with a `.tail.json` twin), their caches — every judged query keeps its
ranking, confidence, reason, token counts and which panellists it agreed with —
and the official score files. Logs: `logs/judge.*.server.log`.

## What this settles

**Fifth failed attempt at combining these models**, and the most informative
one. The four before it (Task 1 two-model fusion, equal-weight RRF,
accuracy-weighted RRF, the 781,248-fusion search) all combined *rankings*. This
one re-examined the *video* with the rankings as hypotheses, which was the last
untried mechanism, and it too lands on the baseline.

The honest reading: **on this corpus, a disagreement between these eight models
is a signal that the query is hard, not that some model present knows better.**
Getting the remaining 54 queries needs evidence none of them have — audio for
the "what can be heard" questions, denser frames for counting — not a better
arbiter of the same 16 frames.

## Open items

1. Submit ≤ 5 MC runs before the 18 Sep close. Unchanged:
   **qwen38 → qwen3vl → fused.weighted → gemma4.vllm → fused.rrf**.
2. The judge run files are valid and complete for dev only; no 2026 judge run
   exists, by the gate.
3. If the judge is ever revisited, give it something the panel lacks: the
   Whisper transcript is already wired in, but only 30 of the 400 dev queries
   have speech — the 2026 set has 576.

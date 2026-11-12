# Two-model confidence-ranked ensemble — TREC 2026 Task 1

**Scope: the 2026 submission set — `vqa.2026.testingDataset.Task1.json`, 2,443
queries over 1,956 videos, 10 ranked answers each = 24,430 output slots.**

Everything below produces that file. **No 2025 data is involved** — the
ground-truth measurement gate was considered and deliberately skipped (see
"Accepted without measurement" below).

**Deadline 17 August 2026.** One RTX 3090, 22.72 GiB usable — one model resident
at a time, so all GPU work is a serial schedule.

---

## Context

Three answering passes exist over the same 2,443 queries. They disagree wildly in
quality, and the measured cause is **architecture, not model**:

| Script | Routing | Variations | Calls/query | Dead slots of 24,430 |
|---|---|---|---|---|
| `generate.answers.py` (Gemma 4) | ✗ | ✓ 10 × k=3 | 10 | **24** (0.1%) |
| `generate.answers.qwen3vl.py` | ✓ | ✗ | 1 (k=10) | 1,596 (6.5%) |
| `generate.answers.cosmos.py` | ✗ | ✗ | 1 (k=10) | 11,289 (46.2%) |

Every pass has half the machinery; none has both. Gemma 4's ten-prompt ensemble
is the only configuration that reliably fills ten distinct slots.

**This plan** drops Cosmos, gives both surviving models routing **and** the
ten-prompt variation ensemble, captures a per-candidate probability score, and
fuses each query's 60 candidates into one ranked ten.

Per query: **10 prompts × k=3 × 2 models = 60 candidates → top 10.**
Across the 2026 set: **146,580 candidates → 24,430 slots.**

### Decisions taken

- **Drop Cosmos.** METEOR 0.033 against Gemma 4's 0.148, 46% dead slots, and
  1,097 of its 2,443 answers were generated with no video attached at all.
- **Swap Qwen3-VL-8B-Thinking → `Qwen/Qwen3-VL-8B-Instruct`.** Measured over the
  2,368 video-attached queries of the 2026 run:
  `elapsed = 3.63s + 40.18s per 1000 completion tokens` (R² = 0.876). The
  Thinking checkpoint burns a median **1,423** completion tokens per call, so
  generation is 94% of runtime and ten prompts over 2,443 queries would cost
  **~109 h**. With a non-thinking checkpoint (~60 tokens) the same ten prompts
  cost **~7.3 h**. This swap is the only reason the design fits the deadline.
- **k=3 per call, not k=1.** Gemma 4's 30 candidates collapse to a mean of
  **9.24** distinct after dedupe; 20 candidates would land near 6–8, below the
  ten the validator requires — the exact failure this plan exists to remove.
- **Re-run Gemma 4 over all 2,443** to capture logprobs, so both models carry
  identical signals.

### The finding that shapes the ranking design

`VQA/rerank.answers.py` and `eval.2025.rerank.scores.csv` already measured this:

- Mean METEOR is **identical (0.148)** under current / random / reverse / oracle
  ordering. `eval.vqa.ag.py` averages over all ten answers, so **ordering cannot
  move METEOR or BERTScore** — only NDCG responds.
- Re-ranking Gemma 4 by its own pooled `score` made NDCG *worse*: 0.749 vs 0.751.
- Spearman against per-answer METEOR: `score` 0.032, `votes` **−0.073**.

**So the value is in *selection*, not ordering.** Choosing the best ten of 60
changes which answers are in the set, and that moves mean METEOR. Re-ordering
those ten does not. The rerank experiment tested only ordering, so selection is
unmeasured — hence Phase 3.

Two returns, worth keeping separate:

1. **Guaranteed** — dead slots disappear. Qwen's 1,596 fallback slots score ~0
   METEOR and are replaced by real candidates from the other model. This return
   does not depend on the ranking working at all: a bigger, more diverse pool
   fills the ten slots with real answers either way.
2. **Unproven** — that a confidence/logprob signal picks the *right* ten. The
   existing signals correlate at 0.03–0.10. Logprobs are new and untested here.

### Accepted without measurement

The 150-query 2025 ground-truth gate was **skipped by decision** for the
submission run. Consequence, stated plainly: return 1 is banked, return 2 ships
unverified. Nothing on disk says a logprob-weighted selection beats picking ten
at random from the same 60 candidates.

> **The gate can still be run after the fact** — the sample, ground truth,
> routing, variations, frames and transcripts are all on disk, and the fused
> 2026 artifacts are unaffected by measuring. Runbook:
> **`GATE_2025_RUNBOOK.md`** (~1–1.5 h, one model swap).

Three things keep that risk bounded:

- The four existing validated 2026 runs stay untouched on disk as fallbacks.
- `fuse.answers.py --rebuild` re-ranks from cache with no inference, so the
  formula can be changed in seconds at any point up to submission.
- Verification 2 below (order-independence) still proves the fuser is not
  *corrupting* answers, independently of whether its ordering is smart.

---

## Design

### Per-candidate record

Every candidate from every call carries the following, so the ranking formula can
be re-tuned offline over the cache with zero inference — the `--rebuild` pattern
already proven in `VQA/route.query.inputs.py`:

```
model            gemma4 | qwen3vl
prompt_index     0 = main question, 1-9 = variation
position         0-2, its slot within that call's k=3
confidence       the call's self-reported value (known weak; stored anyway)
mean_logprob     mean token logprob over THIS answer's own text tokens
min_logprob      the weakest token in the answer
n_tokens         length, for normalisation
```

### Logprob capture

Both servers are vLLM and accept `logprobs: true` on chat completions. Gemma 4
must run on the **vllm** backend (already its default) — Ollama does not expose
logprobs.

**The attribution problem.** Both scripts use `response_format` json_schema, so
generation is grammar-constrained and the *structural* tokens (`{`, `"answers"`,
`[`) are forced to probability ~1. Scoring the whole sequence would mostly
measure the grammar. So attribute tokens to answers:

1. Rebuild the content by concatenating token strings, tracking each token's
   character span.
2. `json.loads` the content to recover the k answer strings.
3. Locate each answer's span in the raw content, searching forward from the end
   of the previous answer's span.
4. Average the logprobs of tokens intersecting that span.

`prob = exp(mean_logprob)` is the reported probability score.

### Fusion and ranking

New `VQA/fuse.answers.py`, CPU-only, operating on the two work caches:

1. Load both models' cached candidates for a Q_ID (60).
2. Dedupe on `normalize()` — the validator's own rule
   (`validate.ag.run.py:276`) — into clusters, keeping every contributing signal.
3. Rank clusters. Default composite, **subject to the Phase 3 gate**:
   `(produced_by_both_models, position_weighted_vote_score, mean_logprob)`,
   refusals sorted last.
4. Take the top 10; `fill_slots()` only if fewer than 10 distinct exist.
5. `--rebuild` re-ranks from cache with no inference, so the formula stays cheap
   to change after measurement.

**Cross-model agreement is the strongest untested hypothesis** — a candidate both
models produced independently is a far better bet than one that 30 candidates of
a single model agreed on. Neither source document considers it.

---

## Files

**Modify**

- `VQA/generate.answers.qwen3vl.py` — add `--variations`; loop the 10 prompts per
  query instead of one call; `k=3`; capture logprobs; default model to
  `Qwen/Qwen3-VL-8B-Instruct`. Keep `plan_for()` routing and the whole
  `VideoPrep` / degrade-ladder machinery unchanged. **Add a
  `finish_reason == "length"` retry** — `Qwen3VLClient.answer()` currently
  returns on the first 200 and never inspects it, which is how 60 queries lost
  all ten slots.
- `VQA/generate.answers.py` — add `--routing` (Gemma 4 has none today), reusing
  the `plan_for()` logic; capture logprobs in `VLLM.ask()` (raw `requests`
  payload at :483).
**Out of scope** — `VQA/rerank.answers.py` and `VQA/run.eval.2025.py` are left
alone. Both changes existed only to serve the skipped gate.

**Create**

- `VQA/fuse.answers.py` — cross-model pooling, ranking, `--rebuild`.

**Reuse unchanged** — `pool_candidates()` (`generate.answers.py:538`, the
position-weighted `(1.0, 0.5, 0.25)` vote), `tidy()` / `normalize()` / `dedupe()`
/ `fill_slots()`, `plan_for()` (`generate.answers.qwen3vl.py:502`),
`expand.answers.py`, `validate.ag.run.py`, `score.answers.py`.

**Do not touch** — the four validated 2026 submission candidates already on disk
(`gemma4.run.task1.json`, `gemma4.expanded.run.task1.json`,
`qwen3vl.run.task1.json`, `qwen3vl.expanded.run.task1.json`). They are the
fallback if this lands badly. Everything new gets new names.

---

## Execution

> **Code status: done.** All three files are written and compile;
> `fuse.answers.py` is verified end-to-end against the two existing caches
> (2,443/2,443 fused, `validate.ag.run.py` PASSED, 0 errors). What remains is
> GPU work.

### Phase 1 — no GPU, start immediately

```bash
# in WSL2
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir ~/models/Qwen3-VL-8B-Instruct
```

### Phase 2 — smoke tests, ~30 min GPU

Five queries per model, throwaway output. What to check is listed after each.

```bash
# --- Gemma 4 (port 8010), from Windows in E:\TREC_2026 ---
.venv\Scripts\python.exe VQA\generate.answers.py --limit 5 ^
  --out smoke.gemma4.json --cache smoke.gemma4.cache.jsonl
```

Look for `candidates carrying a logprob: N/N` in the report — a `0/N` with the
`CHECK: logprobs are off` marker means the server is not returning them and the
whole probability half of the design is silently absent. Also confirm
`queries answered on a routed plan: 5`.

```bash
# --- Qwen3-VL (port 8009), from WSL2 ---
/home/aakash/cosmos-reason2/.venv/bin/python VQA/generate.answers.qwen3vl.py \
  --limit 5 --out /mnt/e/TREC_2026/smoke.qwen3vl.json \
  --cache /mnt/e/TREC_2026/smoke.qwen3vl.cache.jsonl
```

Check three things: the same logprob line; `mean out` tokens per budget tier has
collapsed from ~1,423 to double or low-treble digits (if it has not, the server
is still serving `-Thinking`); and `distinct candidates pooled` is near 30, not
near 3.

Serve command for the Instruct checkpoint — same flags as before, the
`--skip-mm-profiling` and `≤0.92` are both load-bearing on this card:

```bash
cd /home/aakash/cosmos-reason2 && setsid nohup ./.venv/bin/vllm serve \
  ~/models/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8009 \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --allowed-local-media-path /mnt/e/TREC_2026 --max-model-len 24576 \
  --gpu-memory-utilization 0.92 --enforce-eager \
  --media-io-kwargs '{"video": {"num_frames": -1}}' --skip-mm-profiling \
  < /dev/null > /home/aakash/qwen3vl.instruct.log 2>&1 & disown; exit 0
```

Note there is **no `--reasoning-parser qwen3`**: the Instruct checkpoint has no
think block, and the flag was never doing anything anyway — `reasoning_chars`
was 0 on all 2,443 rows of the 13 Aug run.

### Phase 3 — the 2026 submission run, ~13 h GPU, serial
All 2,443 queries. Ordered by resident model; every swap is a teardown and reload.

| Block | Resident | Work | Cost |
|---|---|---|---|
| 1 | Gemma 4 | 2,443 × 10 prompts × k=3, routing + logprobs | ~4–5 h |
| — | swap | | ~0.2 h |
| 2 | Qwen3-VL Instruct | 2,443 × 10 prompts × k=3, routing + logprobs | ~7.3 h |
| 3 | — | `fuse.answers.py` → 2,443 × 10 ranked | CPU, minutes |
| — | swap | | ~0.2 h |
| 4 | Gemma 4 | `expand.answers.py` on the fused 2026 run | ~1.5 h |
| 5 | — | `validate.ag.run.py` | CPU |

**Block 1 — Gemma 4 resident** (Windows, in `E:\TREC_2026`)

```
.venv\Scripts\python.exe VQA\generate.answers.py ^
  --out   gemma4.ens.run.task1.json ^
  --cache answers.task1.gemma4.ens.cache.jsonl --workers 8
```

**Block 2 — swap, then Qwen3-VL Instruct resident** (WSL2)

```
# free the card first -- one 3090, and gemma4-vllm holds ~8 GB
systemctl --user stop gemma4-vllm; pkill -f '[v]llm serve'
# ... then the serve command from Phase 2, then:

/home/aakash/cosmos-reason2/.venv/bin/python VQA/generate.answers.qwen3vl.py \
  --out   /mnt/e/TREC_2026/qwen3vl.ens.run.task1.json \
  --cache /mnt/e/TREC_2026/answers.task1.qwen3vl.ens.cache.jsonl --workers 4
```

**Block 3 — fuse** (CPU, no server needed). Defaults already point at both
`*.ens.cache.jsonl` files and write `fused.run.task1.json`:

```
.venv\Scripts\python.exe VQA\fuse.answers.py

# re-rank later without re-running anything
.venv\Scripts\python.exe VQA\fuse.answers.py ^
  --weights agreement=3,score=1,logprob=0.5,confidence=0.25
```

**Blocks 4–5 — expand and validate.** All four expander paths explicit; see
Verification 5 for why that is not optional.

```
.venv\Scripts\python.exe VQA\expand.answers.py ^
  --run     fused.run.task1.json ^
  --queries vqa.2026.testingDataset.Task1.json ^
  --routing query.routing.task1.json ^
  --out     fused.expanded.run.task1.json ^
  --cache   answers.expansion.task1.fused.cache.jsonl --workers 4

.venv\Scripts\python.exe VQA\validate.ag.run.py ^
  vqa.2026.testingDataset.Task1.json fused.expanded.run.task1.json ^
  --warnings-as-errors --report-json fused.expanded.run.validation.json
```

**Environment traps, both already paid for once:** keep one interactive WSL2
window open for the duration — WSL2 tears the distro down when the last session
exits — and start servers only detached, with `setsid nohup … < /dev/null`.

**One guard now armed that was not before:** `generate.answers.py` gained a
`DEFAULT_ROUTING = ROOT / …`, which `run.eval.2025.py`'s `check_flags()` finds
by static parse. If that orchestrator is ever run again it will refuse to drive
this script without an explicit `--routing`. That is correct — 2025 queries must
not be answered against 2026 routing — but it will look like a new failure.

---

## Verification

1. **Validator** — `validate.ag.run.py vqa.2026.testingDataset.Task1.json <run>
   --warnings-as-errors` on the fused and expanded runs: **2,443 entries, 10
   answers each**, no duplicates within a query.
2. **Order-independence check** — the fused run's mean METEOR must be unchanged
   by re-ordering it. Any difference means the fuser lost, duplicated or altered
   an answer. This is the strongest correctness check available on the ranker.
3. **Oracle check** — `score.answers.py --oracle` must hold BERTScore ≥ 0.99,
   METEOR ≥ 0.85, confirming normalisation and alignment are intact.
4. **Dead-slot count** — fallback slots in the fused 2026 run should approach
   zero; any query still short of 10 distinct out of 60 candidates is a bug worth
   reading.
5. **Q_ID guard** — `expand.answers.py` defaults `--queries` and `--routing` to
   **2025** files, and 2025 Q_IDs are 1–2000 against 2026's 1–2443. All four
   paths must be passed explicitly for the 2026 run, or 2,000 of the 2,443
   queries get 2025 questions and the run completes, validates, and means
   nothing. This has bitten before.

## Risks

- **Instruct checkpoint is unvalidated here** — a model swap two days out, and
  with the gate skipped its answer *quality* is never scored against ground
  truth before submission. Phase 2's smoke test confirms it runs and that the
  token collapse is real; it cannot confirm the answers are good. The four
  existing 2026 runs stay on disk untouched as the fallback.
- **The ranking signal may not select better than chance** — accepted, see
  "Accepted without measurement". Measured correlations of the *existing*
  signals are 0.03–0.10.
- **Long video is untested** — the 2026 set runs to 389 s with 6.3% over 150 s.
  Watch the Phase 3 block-2 logs for degrade-ladder rungs firing on long clips.
- **Expansion caveat, unchanged** — 56–62% of expansion's METEOR gain is length
  matching, and 86% of rewrites add an invented specific.

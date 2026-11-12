# The skipped gate — measuring the whole pipeline on 2025 ground truth

`ENSEMBLE_PLAN.md` shipped under **"Accepted without measurement"**. This
runbook closes that gap: it rebuilds the *entire* pipeline — both models, the
fusion **and the expansion** — over the 150-query 2025 sample where ground truth
exists.

**Measure what ships.** The submission candidate is
`fused.expanded.run.task1.json`, not `fused.run.task1.json`. A gate that stops at
the fused run measures a file nobody submits, and expansion is the stage most
likely to be doing something other than what it looks like — see §"What
expansion is already known to do".

**Cost: ~1.25–1.75 h, ONE model swap.** Step order is chosen to keep it at one:
Qwen first, then Gemma stays resident for both its own pass *and* the expansion.

---

## The questions this answers

| # | Comparison | What a win means |
|---|---|---|
| 1 | fused vs **gemma4 alone** | the second model earns its place |
| 2 | fused vs **random ten from the identical pool** | the *ranking* selects — not just the bigger pool |
| 3 | ens vs the **15 Aug non-ensemble runs** | routing + variations + logprobs moved the metric |
| 4 | expanded vs **unexpanded** | expansion helps the fused run at all |
| 5 | expanded vs **mechanically padded** | expansion adds *meaning*, not just length |
| 6 | fused+expanded vs **gemma4+expanded** | **which file to actually submit** |

**2 and 5 are the load-bearing ones.** Without 2, a win over Gemma proves only
that 60 candidates beat 30. Without 5, a METEOR gain from expansion proves only
that longer answers score better on METEOR.

---

## What is already on disk

| Artifact | |
|---|---|
| `vqa.2025.sample150.json` | the 150 queries, seed 20260814 |
| `vqa.gt.2025.sample150.csv` | ground truth |
| `query.routing.2025.json` · `query.variations.2025.json` | routing + 10 paraphrases each |
| `videos_GT_2025/frames/` · `.../transcription_openwhisper/` | frames, transcripts |
| `gemma4.run.2025.json` · `qwen3vl.run.2025.json` | 15 Aug **non-ensemble** baselines |
| `gemma4.expanded.run.2025.json` · `*.mechanical.run.2025.json` | 15 Aug expansion arms |

> **Every 2025 path must be passed explicitly.** `generate.answers.py`,
> `expand.answers.py` and `fuse.answers.py` all default to **2026** files, and
> 2025 Q_IDs are 1–2000 against 2026's 1–2443. Getting this wrong produces a run
> that completes, validates and means nothing. This has bitten before.

> **Shell note.** The Windows blocks below are **PowerShell**, using a backtick
> (`` ` ``) for line continuation. A trailing space after the backtick breaks it
> silently, so paste whole blocks rather than retyping. `^` continuation and
> `for %R in (…) do` are **cmd.exe** syntax and will not parse here. The WSL2
> blocks are bash and use `\` as normal.

---

## Step 1 — Qwen3-VL ensemble, 150 queries (~30–45 min, GPU)

Qwen first so that Gemma comes up once and stays up for steps 2 and 4.

```bash
# WSL2 — free the card, then serve Instruct
systemctl --user stop gemma4-vllm; pkill -f '[v]llm serve'
cd /home/aakash/cosmos-reason2 && setsid nohup ./.venv/bin/vllm serve \
  ~/models/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8009 \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --allowed-local-media-path /mnt/e/TREC_2026 --max-model-len 24576 \
  --gpu-memory-utilization 0.92 --enforce-eager \
  --media-io-kwargs '{"video": {"num_frames": -1}}' --skip-mm-profiling \
  < /dev/null > /home/aakash/qwen3vl.gate.log 2>&1 & disown; exit 0
```

The Qwen client must run **inside WSL2** — it hands the server a file path the
server opens itself, so a Windows client would send `E:\…`:

```bash
/home/aakash/cosmos-reason2/.venv/bin/python -u \
  /mnt/e/TREC_2026/VQA/generate.answers.qwen3vl.py \
  --queries     /mnt/e/TREC_2026/vqa.2025.sample150.json \
  --routing     /mnt/e/TREC_2026/query.routing.2025.json \
  --variations  /mnt/e/TREC_2026/query.variations.2025.json \
  --videos      /mnt/e/TREC_2026/videos_GT_2025/videos \
  --frame-index /mnt/e/TREC_2026/videos_GT_2025/frames/index.json \
  --transcripts /mnt/e/TREC_2026/videos_GT_2025/transcription_openwhisper \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --out   /mnt/e/TREC_2026/qwen3vl.ens.run.2025.json \
  --cache /mnt/e/TREC_2026/answers.2025.qwen3vl.ens.cache.jsonl --workers 4
```

## Step 2 — swap to Gemma 4, then its ensemble (~15 min, GPU)

**Gemma stays resident from here through step 4.**

```bash
# WSL2
pkill -f '[v]llm serve'; systemctl --user start gemma4-vllm
```

```powershell
.\.venv\Scripts\python.exe VQA\generate.answers.py `
  --queries     vqa.2025.sample150.json `
  --variations  query.variations.2025.json `
  --routing     query.routing.2025.json `
  --frames      videos_GT_2025\frames `
  --transcripts videos_GT_2025\transcription_openwhisper `
  --out         gemma4.ens.run.2025.json `
  --cache       answers.2025.gemma4.ens.cache.jsonl --workers 8
```

Check the report: `candidates carrying a logprob: N/N` (not `0/N`), and
`queries answered on a routed plan: 150`.

## Step 3 — fuse the 2025 pool (CPU, seconds)

```powershell
.\.venv\Scripts\python.exe VQA\fuse.answers.py `
  --queries vqa.2025.sample150.json `
  --gemma4  answers.2025.gemma4.ens.cache.jsonl `
  --qwen3vl answers.2025.qwen3vl.ens.cache.jsonl `
  --out     fused.run.2025.json `
  --cache   fused.2025.cache.jsonl
```

Expect ~25 candidates pooled per query and dead slots near 5%, matching the 2026
fusion. A wildly different number means a path is wrong.

## Step 4 — expansion, three arms (~6 min, Gemma 4 still resident)

This is the stage the earlier draft omitted. **Condition C is not optional** —
it is the only thing that separates "expansion added meaning" from "expansion
added words and METEOR likes words".

| Arm | What it is |
|---|---|
| **A** | the run as produced (`fused.run.2025.json`, from step 3) |
| **B** | LLM rewrite to per-question-type length targets |
| **C** | the same answers padded to the same length with contentless filler, **no model** |

```powershell
# B -- fused, expanded. Backend defaults to vllm, which is what the 2026
# submission used; do NOT pass --backend ollama or it is a different model.
.\.venv\Scripts\python.exe VQA\expand.answers.py `
  --run     fused.run.2025.json `
  --queries vqa.2025.sample150.json `
  --routing query.routing.2025.json `
  --out     fused.expanded.run.2025.json `
  --cache   answers.2025.fused.expansion.cache.jsonl --workers 4

# B -- gemma4 ensemble, expanded. Needed for question 6: the real submission
# choice is between two EXPANDED files, so both must be expanded.
.\.venv\Scripts\python.exe VQA\expand.answers.py `
  --run     gemma4.ens.run.2025.json `
  --queries vqa.2025.sample150.json `
  --routing query.routing.2025.json `
  --out     gemma4.ens.expanded.run.2025.json `
  --cache   answers.2025.gemma4.ens.expansion.cache.jsonl --workers 4

# C -- the mechanical control. No model, instant, no GPU.
.\.venv\Scripts\python.exe VQA\expand.answers.py `
  --run     fused.run.2025.json `
  --queries vqa.2025.sample150.json `
  --routing query.routing.2025.json `
  --out     fused.mechanical.run.2025.json `
  --cache   answers.2025.fused.mechanical.cache.jsonl --mechanical
```

Read the expander's own report before scoring: **`Held — refusal`** and
**`Held — dropped content`** counts, and the mean-words before/after. If mean
words barely moved, B and C will be indistinguishable for uninteresting reasons.

## Step 5 — validate every arm (CPU, seconds)

C has failed validation before — mechanical padding collapses distinct short
answers into duplicates ("a flip" and "the flip" become the same string).
That is a finding, not a blocker; record it and score C anyway.

```powershell
foreach ($r in @(
    "qwen3vl.ens.run.2025", "gemma4.ens.run.2025", "fused.run.2025",
    "fused.expanded.run.2025", "gemma4.ens.expanded.run.2025",
    "fused.mechanical.run.2025")) {
  "=== $r ==="
  .\.venv\Scripts\python.exe VQA\validate.ag.run.py `
    vqa.2025.sample150.json "$r.json" --warnings-as-errors
}
```

## Step 6 — the analytics (CPU, ~5 min)

Every arm in one scoring pass, so the paired per-query deltas come out directly:

```powershell
.\.venv\Scripts\python.exe VQA\score.answers.py `
  --gt vqa.gt.2025.sample150.csv `
  --run A_fused=fused.run.2025.json `
  --run B_fused=fused.expanded.run.2025.json `
  --run C_fused=fused.mechanical.run.2025.json `
  --run A_gemma4ens=gemma4.ens.run.2025.json `
  --run B_gemma4ens=gemma4.ens.expanded.run.2025.json `
  --run A_qwen3vlens=qwen3vl.ens.run.2025.json `
  --run old_gemma4=gemma4.run.2025.json `
  --run old_gemma4_exp=gemma4.expanded.run.2025.json `
  --run old_qwen3vl=qwen3vl.run.2025.json `
  --metrics meteor,bert `
  --out eval.2025.pipeline.scores.json `
  --csv eval.2025.pipeline.scores.csv `
  --per-response-dir eval.2025.pipeline --oracle
```

`--oracle` is the correctness gate on the scorer itself: BERTScore ≥ 0.99 and
METEOR ≥ 0.85 when GT is scored against itself. If that fails, every other
number is meaningless — fix it before reading anything else.

## Step 7 — the selection control (CPU, ~2 min)

```
.venv\Scripts\python.exe VQA\gate.judge.2025.py
```

`fuser` and `random` are built from the **identical** pool, so the only
difference is which ten are kept. Defaults already point at the 2025 caches. The
`judge` arm needs `judge.2025.cache.jsonl`; without it that arm collapses onto
`fuser` and can be ignored.

---

## What expansion is already known to do

Measured 15 Aug on the *pre-ensemble* runs. Carry these in as priors — the gate
is checking whether the fused run behaves the same way:

| Metric | Comparison | gemma4 | qwen3vl |
|---|---|---|---|
| METEOR | B − A | +0.0386 | +0.0807 |
| METEOR | **C − A** | **+0.0235** | **+0.0466** |
| BERTScore | B − A | −0.0003 (ns) | +0.0110 |
| BERTScore | C − A | **−0.0089** | −0.0022 |

**Mechanical padding — adding exactly zero information — captured 56–62% of the
LLM expansion's METEOR gain.** METEOR on this task is substantially a
length-matching measure. BERTScore is the honest signal and it moved by at most
+0.011.

Also known: **86% of rewrites invent something** (the expander has no video, so
every added specific is a guess), and `count` questions are in `NO_EXPAND`
because their GT answers average 1.0 words.

---

## Reading the result

**Read the arm comparisons and the paired per-query sign test. Ignore the pooled
Spearman** — a synthetic uniformly-random judge scored 0.0295 on this data, so
~0.03 is the noise floor, not a weak signal (`judge-rerank-gate` memory).

On n=150 a mean METEOR gap under **~0.005 is noise**.

### Selection (questions 1–3)

| Outcome | Meaning |
|---|---|
| fused > gemma4_ens **and** fuser > random | selection works |
| fused ≈ gemma4_ens, fuser > random | ranking works, second model adds little |
| fuser ≈ random | the formula is not selecting — stop tuning weights |
| fused < gemma4_ens | the ensemble lost; prefer the Gemma line |

### Expansion (questions 4–5)

| Outcome | What to submit |
|---|---|
| B > C on METEOR **and** BERTScore not reduced | expansion earns its place — submit expanded |
| B ≈ C | the gain is length, not meaning. Submit expanded anyway if METEOR is what NIST scores, but say so in the writeup |
| B < A on BERTScore | expansion is damaging semantics — submit unexpanded |

### The decision (question 6)

Whichever of `B_fused` / `B_gemma4ens` / `old_gemma4_exp` scores highest is the
file to submit. All candidates are already built and validated on disk, so this
is a selection, not more work.

---

## Gaps worth knowing about

**No selection-oracle arm** — the best ten obtainable from the pool, which would
bound how much of the achievable gain the ranking captures. ~20 lines in
`gate.judge.2025.py` (rank each cluster by its own per-answer METEOR against GT,
take the top ten) if that bound is wanted.

**The judged runs are not gated here.** `fused.judged.run.task1.json` and
`fused.judged.expanded.run.task1.json` exist from `RERANK_PLAN.md` and are live
candidates. Gating them needs `judge.2025.cache.jsonl` — see that plan's step 2.

**Two caveats survive whatever the gate says**, both already measured:

- Every 2025 GT clip is **5–30 s**. The 2026 set runs to 389 s with 6.3% over
  150 s, and the expensive 80–150 s band has *no* representation here. A clean
  gate does not license conclusions about the 2026 long tail.
- Speech coverage is **63% here against 24% in 2026**, so anything leaning on
  the transcript is flattered by this sample.

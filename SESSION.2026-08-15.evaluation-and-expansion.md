# Session — 2025 ground-truth evaluation, and answer expansion

**15 August 2026.** Two things happened: the project got its first measured
accuracy numbers, and the answer-expansion question that had been open since
14 August was settled with evidence. Along the way the 2026 Qwen3-VL submission
run finished and was validated for the first time.

Implements `VQA/EVALUATION_PLAN.md` (corrected by
`EVAL_2025_ORCHESTRATION_PLAN.md`) and `VQA/ANSWER_EXPANSION_PLAN.md` Phase 2.
Results live in `VQA/EVALUATION.2025.md`; this document is the narrative and the
command record.

---

## 1. What was built

| File | Role |
|---|---|
| `VQA/run.eval.2025.py` | 15-stage orchestrator: stage graph, GPU/server arbitration, guards, resumable state, per-stage logs |
| `VQA/sample.2025.eval.py` | `convert` (CSV → query JSON + eligible pool) and `select` (duration-stratified draw, GT CSV, reserves, achieved-vs-target report) |
| `VQA/score.answers.py` | METEOR / BERTScore / NDCG / STS_local over a 2026-format run, paired comparison, `eval.vqa.ag.py` parity harness |
| `VQA/expand.answers.py` | Phase 2 expander; `--mechanical` gives the zero-cost control |
| `VQA/EVALUATION.2025.md` | the measured results and full appendix |
| `EVAL_2025_ORCHESTRATION_PLAN.md` | the plan, with an implementation-notes section |

Every pre-existing script was driven **unchanged**, through its CLI only.

---

## 2. Two premises of EVALUATION_PLAN.md were stale

**The 2025 GT videos are on disk.** 1,327 of 2,000, downloaded into
`videos_GT_2025/` (not `videos_2025_GT/` as the plan calls it). Its Stage 1
(yt-dlp metadata probe) and Stage 4 (download) were therefore unnecessary. The
download had been interrupted at Q_ID 1471; 144 of the attempted 1,471 failed
(9.8% attrition), 529 were never attempted.

**Its duration strata cannot be reproduced.** Every 2025 GT clip is a YouTube
Short. Measured over all 1,327 with ffprobe: median 15.44 s, mean 17.0 s, range
5.53–30.49 s, 0 probe failures. The 2026 target bins `0–5 s` (11.9% of the test
set), `60–150 s` (10.9%) and `150 s+` (6.3%) are **empty**. The sample was drawn
proportional to the observed pool instead.

---

## 3. The evaluation run

Sample: **150 queries, seed 20260814**, drawn 16/45/85/4 across 5–10 / 10–15 /
15–30 / 30–60 s. Full 2026 input parity — Whisper transcripts, routed inputs,
and Gemma 4's 10-prompt paraphrase ensemble.

| Stage | Time | Result |
|---|---|---|
| queries | <1 min | 1,327 eligible of 2,000; disjointness from 2026 verified |
| durations | 0.4 min | 1,327 probed, 0 failures |
| sample | <1 min | 150 drawn |
| frames | 0.5 min | 150/150 videos, 2,110 frames |
| whisper | 3.7 min | 150/150, 94 with speech |
| route | 16.4 min | 150/150 |
| variations | 6.1 min | 150/150, 10 paraphrases each |
| answers:gemma4 | 39.9 min | 150/150, 0 failures |
| answers:qwen3vl | 37.9 min | 150/150, 0 failures |
| answers:cosmos | 34.3 min | 150/150, 0 failures |
| validate | <1 min | 3/3 pass, 0 errors, 0 warnings |
| score | 1.5 min | oracle passed |

### Results

| Run | METEOR | BERTScore | NDCG_METEOR | words |
|---|---|---|---|---|
| **gemma4** | **0.148** | **0.892** | **0.751** | 4.9 |
| qwen3vl | 0.053 | 0.873 | 0.597 | 3.3 |
| cosmos | 0.033 | 0.863 | 0.524 | 3.6 |
| *oracle* | *0.924* | *1.000* | *1.000* | *8.2 (GT)* |

Gemma 4 wins every metric, all pairwise differences p ≈ 0. Three caveats that
change what that means are recorded in `VQA/EVALUATION.2025.md` §"Reading the
table": Cosmos's score is mostly a degradation artifact (81% fallback, 5.5/10
filler slots, 20% unparsed JSON); Qwen3-VL's deficit is real but partly
verbosity; and BERTScore has almost no dynamic range on this task (0.863–1.000
across everything).

### Sample fidelity, measured after the draw

- **Routing marginals landed within ~3 pp of 2026 for free.** vision 78.7%
  (2026: 75.3%), vision+audio 12.7% (15.5%), audio 5.3% (6.9%), OCR routes 3.3%
  (2.2%) — all four OCR routes represented rather than rounding to zero.
  Consequence: the plan's Stage 2/3 (route a 400-query pool, select against
  three marginals) would have bought very little over the duration-only draw.
- **Speech coverage is much higher than 2026:** 63% of sampled clips have
  detected speech against 24% of 2026 queries. A pipeline that leans on the
  transcript is flattered by this sample.

---

## 4. Findings about the submission runs

**The 2026 Qwen3-VL run finished during this session** (2,443/2,443 at 03:47)
and had **never been validated** — gemma4 and cosmos had reports from 13 August.
Validated now: 2443/2443, 0 errors, 0 warnings.

**`cosmos.run.task1.json` has 1,097 of 2,443 queries (45%) answered with no
video.** This is exactly the bug its own docstring documents — `Video_ID` is not
a filename, so the pre-fix code never opened the 852 downloaded `.mp4`s. The fix
is dated 14 August; the run is dated **13 August** and was never redone. The
2025 eval run, same script post-fix, has **0** no-video queries, confirming the
fix works. Re-running is ~9 h at 2 workers.

| 2026 run | fallback | filler slots | unparsed | no video |
|---|---|---|---|---|
| gemma4 | 0.8% | 0.0/10 | 0% | 0 |
| qwen3vl | 8.6% | 0.7/10 | 2.5% | 16 *(genuinely missing media)* |
| **cosmos** | **74%** | **4.6/10** | **18.7%** | **1,097** |

---

## 5. Orchestration defects

Three cost a restart each. All were in the orchestrator, none in the answering
pipeline, and all three were invisible interactively — they only appear when a
run is unattended.

**1 — The guard was armed before the gate.** The 2026-artifact baseline was
snapshotted at startup, but the gate exists to *wait* while the 2026 run appends
to `answers.task1.qwen3vl.cache.jsonl` and finally writes
`qwen3vl.run.task1.json` — both guarded files. It read that run's own successful
completion as tampering and aborted a five-hour wait at the moment it succeeded
(exit 2). *Fix:* arm the baseline after the gate.

**2 — Ollama's model host was orphaned on the GPU.** Ollama does not hold the
model in `ollama.exe`; it spawns `lib\ollama\llama-server.exe`, and that child
owns the VRAM. Teardown killed the parent and the CLI wrapper, so port 11434
closed and the swap *looked* successful — while 8,922 MiB stayed on the card.
vLLM then could not allocate at `--gpu-memory-utilization 0.92`. *Fix:* graceful
unload (`keep_alive: 0`, `ollama stop`) then `taskkill /F /T` over
`llama-server.exe` and `ollama_llama_server.exe`; `teardown()` now returns
whether the card is free, retries once, and reports holding PIDs via
`nvidia-smi --query-compute-apps`; `ensure_backend()` aborts rather than
starting a server that cannot allocate.

**3 — vLLM was killed by the launcher's own pipes.** A process backgrounded from
`wsl.exe -- bash -lc '… &'` dies when that session ends. `setsid` alone did not
save it: the real culprit was `subprocess.run(capture_output=True)` — when
Python closed the pipes, WSL tore down the launched process with them. The
launch returned `rc=0`, the redirect truncated the server log to zero bytes, and
nothing was ever listening. Proven, not guessed: the identical command survives
with stdio at `DEVNULL` and dies with `capture_output`. *Fix:* a separate
`launch_detached()` for server starts, plus `setsid nohup … < /dev/null`.

**Two smaller ones:** `pkill -f 'vllm serve'` ran inside a `bash -lc` whose own
command line contains `vllm serve`, so it killed its own shell and abandoned the
rest of the teardown (fixed to `[v]llm serve`); and `ensure_backend()` tore down
a healthy server before checking whether it was already the wanted backend,
costing a full reload on every resume (now adopts it).

**Lesson:** an unattended run is a different environment from an interactive one.
Every assumption carried over from watching a human run these commands had to be
re-earned — process ownership on teardown, session survival on launch, stdio
handling. The `--smoke 5` rehearsal exists to surface this class in 30 minutes
rather than across three multi-hour restarts; skipping it cost more than it
saved.

---

## 6. Answer expansion — the gate, and what it really bought

Three conditions over the same 150 queries.

| Condition | What it is |
|---|---|
| A — baseline | the run as the pipeline produced it |
| B — expanded | `expand.answers.py`, per-question-type length targets |
| C — mechanical | same answers padded to the same length with contentless frame words, **no model** |

**Targets were measured, not guessed** (Phase 1's outstanding deliverable), from
the 2025 GT by `Question_Type`: action 10, reason 10, manner 11, summary 12,
entity 6, place 6, time 7, yes_no 6 words. `count` is in `NO_EXPAND` — its GT
answers average **1.0 words** (median 1, literally "one"/"two"/"three"), and 23%
of this ground truth is 1–2 words.

### Results

| Metric | Comparison | gemma4 | qwen3vl | cosmos |
|---|---|---|---|---|
| METEOR | B − A | +0.0386 (108/150, p≈0) | +0.0807 (112, p≈0) | +0.0436 (88, p≈0) |
| METEOR | C − A | +0.0235 (85, p≈0) | +0.0466 (89, p≈0) | +0.0246 (63, p≈0) |
| BERTScore | B − A | −0.0003 (p=0.134, ns) | +0.0110 (p≈0) | +0.0078 (p≈0) |
| BERTScore | C − A | **−0.0089 (p≈0)** | **−0.0022 (p=0.034)** | +0.0004 (ns) |

**The gate clears for B.** METEOR up everywhere, BERTScore and NDCG not reduced.

**C is not ≈ B, so the free option loses.** B beats C on METEOR by +0.015 to
+0.034 *and* on BERTScore by +0.007 to +0.013; mechanical padding actively
damages semantic similarity for two of three runs. C also **fails
`validate.ag.run.py`** — padding collapses distinct short answers into
duplicates ("a flip" and "the flip" become the same string), 2 slots in gemma4
and 1 in qwen3vl.

**The headline finding is about the metric, not the method.** Mechanical
padding — a transformation adding exactly zero information — captures **56–62%**
of the LLM expansion's METEOR gain (gemma4 62%, qwen3vl 58%, cosmos 56%).
METEOR on this task is substantially a length-matching measure. Since NIST
scores with METEOR that is worth knowing, but it also means most of B's headline
gain is not evidence that the answers got better. BERTScore is the honest signal
there, and it moved by at most +0.011.

**Expansion narrows but does not reorder the ranking.** Expanded Qwen3-VL
(0.134) is statistically indistinguishable from baseline Gemma 4 (0.148),
mean Δ −0.0134, p = 1.0 — supporting the reading that part of Qwen3-VL's
apparent deficit was verbosity, not accuracy.

### Honest limitations

**86% of rewrites invent something.** The expander has no video, so every added
specific is a guess: "the man performs a flip" → "a **single** flip"; "The woman
looks at the man" → "looks **directly** at the man **by the end**". 909/951/543
rewrites added ≥1 content word absent from the original, ~2 each. The
containment check guarantees nothing is *lost*; nothing prevents additions.

**A defect found and fixed mid-flight.** The refusal pass-through leaked: 176
literal fallback strings were rewritten on the first pass (160 in Cosmos),
turning "it is not clear from the video" into "It is not clear from the video
what they do." — a longer zero. `NON_ANSWER_RE` is anchored `[.!]?$` and missed
the two fallback entries with a trailing prepositional phrase. Fixed by checking
literal membership of the known `FALLBACK_ANSWERS` pool as well as the regex;
all three runs regenerated, leak now 0/0/0.

---

## 7. Expansion applied to the 2026 submissions

Per `ANSWER_EXPANSION_PLAN.md` scope: gemma4 and qwen3vl only. Cosmos excluded —
its 45%-blind generation is the problem worth fixing there, not answer length.

| | Qwen3-VL | Gemma 4 |
|---|---|---|
| Queries | 2,443 / 2,443, 0 failures | 2,443 / 2,443, 0 failures |
| Wall time | 68.7 min | 75.8 min |
| Mean words | 3.10 → **6.90** | 4.44 → **7.38** |
| Rewritten | 17,428 | 17,385 |
| Held — refusal | 2,770 | 1,084 |
| Held — `count` | 3,120 | 3,120 |
| Held — dropped content | 1,636 | 2,317 |
| Held — would duplicate | 12 | 127 |
| Held — returned as-is | 70 | 734 |
| Validation | PASSED 2443/2443, 0 errors | PASSED 2443/2443, 0 errors |

Both sit near the 8.2-word GT mean, up from roughly a third of it. Q_IDs,
Video_IDs and rank order are identical to the originals; the source runs are
byte-identical (`9b99680e…`, `b61f10ae…`) since the expander never edits in
place.

**Four validated 2026 submission candidates now exist:**

| File | Mean words |
|---|---|
| `gemma4.run.task1.json` | 4.44 |
| `gemma4.expanded.run.task1.json` | 7.38 |
| `qwen3vl.run.task1.json` | 3.10 |
| `qwen3vl.expanded.run.task1.json` | 6.90 |

The unexpanded runs remain valid fallbacks.

---

## 8. Commands

Run from **Windows** in `E:\TREC_2026` with `.venv\Scripts\python.exe`. Windows
is the only host that can drive both backends: Ollama is a Windows binary bound
to localhost that WSL2 cannot reach, while the vLLM clients must run *inside*
WSL2 because they hand the server a `file://` URI it opens itself.

### 8.1 One-time setup

```
.venv\Scripts\python.exe -m pip install pandas nltk bert-score sentence-transformers
```

`nltk` `wordnet`/`punkt`/`punkt_tab` download on first score. `transformers` was
left at its existing 5.15.0 pin; `bert_score` works with it.

### 8.2 The evaluation

```
python VQA/run.eval.2025.py --smoke 5          # rehearse everything first
python VQA/run.eval.2025.py                    # the real run
```

What was actually run this session:

```
python VQA/run.eval.2025.py --stages queries   --no-gate
python VQA/run.eval.2025.py --stages durations --no-gate
python VQA/run.eval.2025.py --stages sample    --no-gate
python VQA/run.eval.2025.py --wait-for-2026 25200     # gated on the 2026 run
python VQA/run.eval.2025.py --from answers:qwen3vl    # resumed after fixes
```

Flags: `--dry-run` · `--from`/`--to`/`--stages` · `--force` · `--no-gate` ·
`--wait-for-2026 <s>` · `--install-deps` · `--no-guard` · `--keep-servers` ·
`--restore-2026`.

### 8.3 Per-stage commands the orchestrator issues

`--dry-run` prints these verbatim; each is reproducible standalone.

```
python VQA/sample.2025.eval.py convert \
  --gt vqa.gt.2025.csv --videos videos_GT_2025/videos \
  --queries-out vqa.gt.2025.queries.json --pool-out vqa.gt.2025.pool.json \
  --test-2026 vqa.2026.testingDataset.Task1.json

python VQA/duration.report.py \
  --queries vqa.gt.2025.queries.json --videos videos_GT_2025/videos \
  --report duration.report.gt2025.json --html duration.report.gt2025.html

python VQA/sample.2025.eval.py select \
  --pool vqa.gt.2025.pool.json --durations duration.report.gt2025.json \
  --metadata videos_GT_2025/metadata -n 150 --seed 20260814 \
  --out vqa.2025.sample150.json --gt-out vqa.gt.2025.sample150.csv \
  --reserves-out vqa.2025.sample150.reserves.json \
  --report-json sample.2025.sample150.report.json \
  --report-md sample.2025.sample150.report.md

python VQA/extract.frames.py \
  --queries vqa.2025.sample150.json --videos videos_GT_2025/videos \
  --durations duration.report.gt2025.json --out videos_GT_2025/frames

python VQA/transcribe.whisper.py \
  --videos-dir videos_GT_2025/videos \
  --out-dir videos_GT_2025/transcription_openwhisper \
  --model-dir models/faster-whisper-large-v3 --only <the 150 video IDs>

python VQA/route.query.inputs.py --queries vqa.2025.sample150.json \
  --out query.routing.2025.json --cache query.routing.2025.cache.jsonl

python VQA/generate.query.variations.py --queries vqa.2025.sample150.json \
  --out query.variations.2025.json --cache query.variations.2025.cache.jsonl

python VQA/generate.answers.py \
  --queries vqa.2025.sample150.json --variations query.variations.2025.json \
  --frames videos_GT_2025/frames \
  --transcripts videos_GT_2025/transcription_openwhisper \
  --out gemma4.run.2025.json --cache answers.2025.gemma4.cache.jsonl --workers 1
```

Inside WSL2 (`wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/e/TREC_2026 && …'`):

```
/home/aakash/cosmos-reason2/.venv/bin/python VQA/generate.answers.qwen3vl.py \
  --queries     /mnt/e/TREC_2026/vqa.2025.sample150.json \
  --routing     /mnt/e/TREC_2026/query.routing.2025.json \
  --videos      /mnt/e/TREC_2026/videos_GT_2025/videos \
  --frame-index /mnt/e/TREC_2026/videos_GT_2025/frames/index.json \
  --transcripts /mnt/e/TREC_2026/videos_GT_2025/transcription_openwhisper \
  --out         /mnt/e/TREC_2026/qwen3vl.run.2025.json \
  --cache       /mnt/e/TREC_2026/answers.2025.qwen3vl.cache.jsonl --workers 4

/home/aakash/cosmos-reason2/.venv/bin/python VQA/generate.answers.cosmos.py \
  --queries     /mnt/e/TREC_2026/vqa.2025.sample150.json \
  --videos      /mnt/e/TREC_2026/videos_GT_2025/videos \
  --frame-index /mnt/e/TREC_2026/videos_GT_2025/frames/index.json \
  --transcripts /mnt/e/TREC_2026/videos_GT_2025/transcription_openwhisper \
  --out         /mnt/e/TREC_2026/cosmos.run.2025.json \
  --cache       /mnt/e/TREC_2026/answers.2025.cosmos.cache.jsonl --workers 2
```

### 8.4 Server lifecycle

`ensure_backend()` is the only code permitted to start or stop a model server.
`setsid` and `< /dev/null` are load-bearing — see defect 3.

```
# Qwen3-VL (port 8009)
cd /home/aakash/cosmos-reason2 && setsid nohup ./.venv/bin/vllm serve \
  Qwen/Qwen3-VL-8B-Thinking --host 0.0.0.0 --port 8009 \
  --served-model-name Qwen/Qwen3-VL-8B-Thinking \
  --allowed-local-media-path /mnt/e/TREC_2026 --max-model-len 24576 \
  --gpu-memory-utilization 0.92 --enforce-eager \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3 --skip-mm-profiling \
  < /dev/null > /home/aakash/qwen3vl.eval.log 2>&1 & disown; exit 0

# Cosmos-Reason2 (port 8008)
systemctl --user start cosmos-reason2

# Teardown -- unit first (Restart=on-failure would resurrect it), then the tree
systemctl --user stop cosmos-reason2; pkill -f '[v]llm serve'
taskkill /F /T /IM "ollama app.exe" /IM ollama.exe /IM llama-server.exe
```

### 8.5 Validation and scoring

```
python VQA/validate.ag.run.py vqa.2025.sample150.json <run>.json \
  --warnings-as-errors --report-json <run>.validation.json

python VQA/score.answers.py --gt vqa.gt.2025.sample150.csv \
  --run gemma4=gemma4.run.2025.json \
  --run qwen3vl=qwen3vl.run.2025.json \
  --run cosmos=cosmos.run.2025.json \
  --metrics meteor,bert --out eval.2025.scores.json \
  --csv eval.2025.scores.csv --per-response-dir eval.2025 --oracle
```

**Scorer parity check** (the gate on every number above):

```
python VQA/score.answers.py --gt <20-row GT>.csv --run synth=<run>.json \
  --metrics meteor,bert --emit-csv parityCSV/ ...

cd <a directory without calc.sts.pl>    # makes the dead UMBC STS call fail fast
python VQA/eval.vqa.ag.py <20-row GT>.csv <abs path>/parityCSV
```

Both returned METEOR 0.277, BERTScore 0.886, NDCG_METEOR 0.964, NDCG_BERTScore
0.997, with **0 per-query disagreements at 3 dp**.

### 8.6 Answer expansion

> **Backend note.** Every expansion in this session ran on **Ollama**
> (`gemma4:12b`, `localhost:11434`), which was the script's default at the time.
> `expand.answers.py` has since gained `--backend {ollama,vllm}` defaulting to
> **vllm**, so reproducing these exact runs now requires `--backend ollama`.
> The two backends load different quantisations and their rewrites are not
> interchangeable — name `--out`/`--cache` per backend.

```
# condition B -- LLM rewrite, over the 150-query 2025 sample
python VQA/expand.answers.py --backend ollama \
  --run <model>.run.2025.json \
  --queries vqa.2025.sample150.json --routing query.routing.2025.json \
  --out <model>.expanded.run.2025.json \
  --cache answers.2025.<model>.expansion.cache.jsonl

# condition C -- mechanical padding control, no model, instant
python VQA/expand.answers.py \
  --run <model>.run.2025.json \
  --queries vqa.2025.sample150.json --routing query.routing.2025.json \
  --out <model>.mechanical.run.2025.json \
  --cache answers.2025.<model>.mechanical.cache.jsonl --mechanical

# score all nine together -- yields the paired A/B/C deltas directly
python VQA/score.answers.py --gt vqa.gt.2025.sample150.csv \
  --run A_gemma4=gemma4.run.2025.json \
  --run B_gemma4=gemma4.expanded.run.2025.json \
  --run C_gemma4=gemma4.mechanical.run.2025.json \
  --run A_qwen3vl=qwen3vl.run.2025.json \
  --run B_qwen3vl=qwen3vl.expanded.run.2025.json \
  --run C_qwen3vl=qwen3vl.mechanical.run.2025.json \
  --run A_cosmos=cosmos.run.2025.json \
  --run B_cosmos=cosmos.expanded.run.2025.json \
  --run C_cosmos=cosmos.mechanical.run.2025.json \
  --metrics meteor,bert --out eval.2025.expansion.scores.json \
  --csv eval.2025.expansion.scores.csv \
  --per-response-dir eval.2025.expansion --oracle
```

### 8.7 Expansion of the 2026 submission runs

**The Q_ID collision is the hazard here.** 2025 Q_IDs are 1–2000 and 2026 Q_IDs
are 1–2443, and `expand.answers.py`'s `--queries` and `--routing` defaults both
point at **2025** files. Leaving either off would feed 2025 questions and
question-types into 2,000 of the 2,443 queries. All four paths must be explicit,
and the cache must be distinct from the 2025 one.

```
python VQA/expand.answers.py --backend ollama \
  --run     qwen3vl.run.task1.json \
  --queries vqa.2026.testingDataset.Task1.json \
  --routing query.routing.task1.json \
  --out     qwen3vl.expanded.run.task1.json \
  --cache   answers.expansion.task1.qwen3vl.cache.jsonl --workers 4

python VQA/expand.answers.py --backend ollama \
  --run     gemma4.run.task1.json \
  --queries vqa.2026.testingDataset.Task1.json \
  --routing query.routing.task1.json \
  --out     gemma4.expanded.run.task1.json \
  --cache   answers.expansion.task1.gemma4.cache.jsonl --workers 4

python VQA/validate.ag.run.py vqa.2026.testingDataset.Task1.json \
  <model>.expanded.run.task1.json --warnings-as-errors \
  --report-json <model>.expanded.run.validation.json
```

Smoke-test first with `--limit 5` and a throwaway `--out`/`--cache`: it verifies
question types resolve from the right routing file before committing ~70 min.

---

## 9. Guards

Five, because a multi-hour unattended run fails silently otherwise.

| Guard | What it prevents | Where |
|---|---|---|
| **Completion gate** | Starting while the 2026 Qwen3-VL run is live. Three conditions, since any one alone lies: cache holds all 2,443 Q_IDs; the run file exists with 2,443 entries (written *only* after a clean loop — a Ctrl-C'd run exits 130 before `assemble()`); no client process alive in WSL. | `qwen3vl_2026_state()` |
| **2026 artifact fingerprint** | Silent overwrite — `duration.report.py` defaults to the live 2026 `duration.report.json`. Size + sha256 of 11 artifacts plus the media inventory, re-verified **after every stage**. Armed after the gate (defect 1). | `fingerprint()`/`verify()` |
| **Path-flag coverage** | The Q_ID collision. Statically parses each driven script for `DEFAULT_* = ROOT / …` path arguments and fails if the command omits any. Subparser-aware. | `path_flags()`/`check_flags()` |
| **Disjointness assertion** | A 2025 GT video also being in the 2026 test set. Verified: 0 overlap across 2,000 × 1,956. | `sample.2025.eval.py convert` |
| **Oracle check** | Broken normalisation or alignment in the scorer. BERTScore ≥ 0.99, METEOR ≥ 0.85. | `score.answers.py --oracle` |

**The Q_ID guard is not theoretical.** All 150 sampled Q_IDs exist in *both*
years' variation files with entirely different questions — Q_ID 21 is "How long
do you have to wear an orthodontic expander?" in 2025 and "Why is smoke rising
up?" in 2026, and **0 of 150 share a question**. Taking `--variations`'s default
would have asked every query using paraphrases of an unrelated 2026 question,
then run clean, validated, and scored — and meant nothing.

**Note on METEOR and the oracle:** METEOR cannot reach 1.0 on an exact match.
Its fragmentation penalty is `0.5 · (chunks/matches)³`, which costs a full 0.5
on a single-word answer, and 17% of this ground truth is one word. A perfect
oracle scores **0.924**. BERTScore (1.000) is the strict alignment check.

---

## 10. Open items

1. **Re-run Cosmos for 2026.** 1,097 of 2,443 queries were answered with no
   video; the fix landed 14 August but the run predates it. ~9 h at 2 workers.
2. **Decide on expanded vs unexpanded for submission.** Four validated
   candidates exist. Expansion reliably lifts METEOR, but 56–62% of that is the
   metric responding to length, and 86% of rewrites add an invented specific.
3. **Nothing here tests long video.** Every 2025 clip is 5–30 s; the 2026 set
   runs to 389 s with 6.3% over 150 s. The fps floor, frame-token degrade ladder,
   32-frame long path and 4 GiB transcode never fired. **Do not swap the
   submission model on this evidence alone.**
4. **Ablations are now cheap**, since the scorer is parity-checked: routing
   on/off, budget tier, temperature — all measurable rather than argued.

**Deadline: 17 August 2026.**

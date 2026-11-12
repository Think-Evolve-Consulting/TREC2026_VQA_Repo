# Orchestrating the 150-video 2025 ground-truth evaluation

**Status: implemented 14 August 2026.** Implements and corrects
`VQA/EVALUATION_PLAN.md`. The three scripts exist, the CPU stages have run, and
the scorer is parity-checked. Only the GPU stages remain — see
[Implementation notes](#implementation-notes) at the end for what the build
changed and what is verified.

## Context

No answer this project has produced has ever been scored. Three session documents
list it as the open question, and every tuning decision — Gemma 4's temperature,
Qwen3-VL's reasoning-budget tiers, the routing policy, answer length — rests on
plausibility rather than measurement. `VQA/EVALUATION_PLAN.md` designed the fix
in five phases but was never implemented; two of its premises are now stale:

- It says "none of the 2025 GT videos are on disk". **1,327 now are**, downloaded
  into `videos_GT_2025/` (not `videos_2025_GT/` as the plan calls it).
  Its Stage 1 (yt-dlp metadata probe) and Stage 4 (download) are therefore
  unnecessary.
- Its duration strata **cannot be reproduced**. Every 2025 GT clip is a YouTube
  Short: 5–30 s, median 15 s, max 30 s. The 2026 target bins `0–5 s` (11.9%),
  `60–150 s` (10.9%) and `150 s+` (6.3%) are all empty in the 2025 pool.

The outcome wanted here: one unattended script that measures the duration
distribution of the downloaded 2025 videos, draws a 150-video sample from it,
emits the matching ground-truth CSV, runs all three answering pipelines over that
sample with full 2026 input parity, and scores the results against the ground
truth — surviving the GPU handoffs between three mutually exclusive model servers
without supervision.

## Verified facts this plan is built on

| Fact | Evidence |
|---|---|
| 1,327 videos on disk, named `<YouTubeID>.mp4`, 1:1 with `metadata/*.json` | `videos_GT_2025/` |
| Download was **interrupted** at Q_ID 1471 of 2000; 144 failed, 529 never attempted | `download_summary.gt2025.json` (`"interrupted": true`) |
| `vqa.gt.2025.csv` = 2,000 headerless rows `Q_ID,Video_ID,Question,Answer`, one question per video; identical to `VQA/vqa.gt.csv` modulo CRLF | parsed both |
| Eligible pool = **1,327**; none of `eval.vqa.ag.py`'s 18 `SKIP` Q_IDs are in it | measured |
| Duration bins of the pool: 5–10 s **116**, 10–15 s **379**, 15–30 s **779**, 30–60 s **53** | `metadata/*.json` `length`, spot-checked vs ffprobe |
| All scripts resolve a bare Video_ID as `<id>.mp4` and key indexes by stem | `resolve_video`/`video_path` in qwen3vl, cosmos, extract.frames |
| **Ollama is a Windows install** (`…\Programs\Ollama\ollama.exe`), binds localhost; WSL has no ollama and no mirrored networking | probed; `.wslconfig` has no `networkingMode` |
| vLLM clients **must** run inside WSL — they send `path.as_uri()`, and `file:///E:/…` is unreadable by the Linux server | `generate.answers.qwen3vl.py:520` |
| `localhost:8009` **is** reachable from Windows (WSL2 localhostForwarding) | curl'd successfully |
| `videos_GT_2025/` already sits under `--allowed-local-media-path /mnt/e/TREC_2026` | no server flag change needed |
| Measured per-query cost: Gemma 4 **16.8 s**, Cosmos **27.1 s**, Qwen3-VL **65.7 s** | mean over the three 2026 caches |
| The 2026 Qwen3-VL run is **live** (cache grew 1,660 → 1,679 during planning); GPU at 24,160/24,576 MiB | `pgrep`, `nvidia-smi` |
| Windows `.venv` is **missing** `pandas`, `nltk`, `bert_score`, `sentence-transformers` | `EVALUATION_PLAN.md` wrongly claims pandas is present |

## Decisions taken

- **Full 2026 input parity** — Whisper transcripts, `route.query.inputs.py`
  routing, and `generate.query.variations.py` paraphrases so Gemma 4 keeps its
  10-prompt / 30-candidate ensemble. Without variations Gemma 4 would run at 3
  candidates per query and would not be the pipeline under test.
- **Proportional stratified sample** over the observed bins: **13 / 43 / 88 / 6**
  for 5–10 / 10–15 / 15–30 / 30–60 s. Fixed seed, ranked reserve list per stratum.
- **One unattended Windows-hosted script** that owns the server lifecycle, with a
  `--smoke 5` end-to-end rehearsal across all three models.
- **Gate on the 2026 run** — no GPU stage starts until
  `answers.task1.qwen3vl.cache.jsonl` holds 2,443 Q_IDs.

## Architecture

**The orchestrator runs on Windows** (`E:\TREC_2026\.venv`) and dispatches WSL
stages via `wsl.exe -d Ubuntu-24.04 -- bash -lc '…'`. This is forced by the two
constraints pulling in opposite directions: Ollama is a Windows binary bound to
localhost that WSL cannot reach, while the vLLM clients must run inside WSL to
emit `/mnt/e/...` URIs. Windows is the only host that can drive both.

Three model backends contend for one RTX 3090 and none can coexist:

```
ollama 11434 (gemma4:12b, 7.4 GB)   vllm 8009 (Qwen3-VL, 22.4 GB)   vllm 8008 (Cosmos)
```

A single `ensure_backend(kind)` function is the only thing allowed to touch them.
Stages are ordered so the card is switched exactly three times:

```
none/whisper  ──►  ollama  ──►  vllm-qwen  ──►  vllm-cosmos  ──►  none
 frames,whisper    route,          answers        answers        validate,
                   variations,     qwen3vl        cosmos         score, report
                   answers gemma4
```

`ensure_backend` tears the previous backend down, polls `nvidia-smi` until VRAM
drops below a threshold, starts the requested one, then polls its health endpoint
(`/v1/models`, `/api/tags`) until ready or times out.

## New files

| File | Role |
|---|---|
| `VQA/run.eval.2025.py` | the orchestrator — stage graph, backend arbitration, state, logs |
| `VQA/sample.2025.eval.py` | CSV → query JSON, duration-stratified sample, GT CSV, reserves, achieved-vs-pool report (importable + CLI; named in `EVALUATION_PLAN.md` §Files) |
| `VQA/score.answers.py` | METEOR / BERTScore / NDCG / STS_local over a 2026-format run, plus paired run comparison |
| `VQA/EVALUATION.2025.md` | generated results document |

Every existing script is used **unchanged**, driven entirely through its CLI:
`VQA/duration.report.py`, `VQA/extract.frames.py`, `VQA/transcribe.whisper.py`,
`VQA/route.query.inputs.py`, `VQA/generate.query.variations.py`,
`VQA/generate.answers.py`, `VQA/generate.answers.qwen3vl.py`,
`VQA/generate.answers.cosmos.py`, `VQA/validate.ag.run.py`.

## Artifact naming — the collision that must not happen

2025 Q_IDs are `1..2000` and 2026 Q_IDs are `1..2443`. **Every cache in this repo
is keyed by bare Q_ID.** Pointing any 2025 stage at a default cache path would
make it read 2026 answers as already-done work and silently emit a run built from
answers to different questions. Every stage therefore gets an explicit 2025 path:

```
vqa.gt.2025.queries.json          all 2,000 rows as query JSON
vqa.gt.2025.pool.json             1,327 eligible (video on disk, not in SKIP)
duration.report.gt2025.json/.html
vqa.2025.sample150.json           THE query file for every downstream stage
vqa.gt.2025.sample150.csv         150-row headerless GT for the scorer
sample.2025.report.json / .md
videos_GT_2025/frames/index.json + <id>/*.jpg
videos_GT_2025/transcription_openwhisper/<id>.json
query.routing.2025.json      + query.routing.2025.cache.jsonl
query.variations.2025.json   + query.variations.2025.cache.jsonl
gemma4.run.2025.json         + answers.2025.gemma4.cache.jsonl
qwen3vl.run.2025.json        + answers.2025.qwen3vl.cache.jsonl
cosmos.run.2025.json         + answers.2025.cosmos.cache.jsonl
eval.2025.scores.json / .csv + *_per_response.csv
eval2025.state.json, logs/eval2025/<stage>.log
```

Smoke mode substitutes a `.smoke` infix throughout, so a rehearsal can never
contaminate the real caches.

## Hardcoded 2026 assumptions — audit and handling

All eight driven scripts were audited (`grep` for `2026`, `2443`,
`videos_TEST_VQA`, `task1`, `.webm`, plus a read of every `parse_args`).
Findings fall into four classes.

**Class 1 — `DEFAULT_*` module constants pointing at 2026 artifacts.** Every one
is exposed as a CLI flag and reachable no other way:
`generate.answers.qwen3vl.py:94-100`, `generate.answers.py:71-76`,
`generate.answers.cosmos.py:65-70`, `extract.frames.py:51-54`,
`route.query.inputs.py:70-71`, `generate.query.variations.py:50-51`,
`transcribe.whisper.py:62-63`.

Handling: **the orchestrator passes every path flag on every invocation and lets
no stage inherit a default**, even where the default is harmless. Backed by a
static check in the `deps` stage — parse each script's `parse_args` for
arguments whose default is a `DEFAULT_*` path, and fail if the orchestrator's
command for that stage omits any of them. A default added later can then never
silently rejoin the run.

**Class 2 — the destructive-overwrite hazard.** `duration.report.py` defaults to
`--report duration.report.json --html duration.report.html`, which are the
**live 2026** artifacts that `extract.frames.py --durations` reads. Stage 4 run
without both flags would clobber the 2026 duration report with a 2025 one. The
same applies to `--out`/`--cache` on the three answering scripts and `--out` on
routing and variations. This class is why verification check 5 is a **guard, not
a post-hoc check**: snapshot size + mtime + sha256 of every 2026 artifact before
the run and re-verify **after every stage**, aborting at the first stage that
touches one. Guarded set: `duration.report.json`, `duration.report.html`,
`query.routing.task1.json`, `query.variations.task1.json`,
`{gemma4,cosmos,qwen3vl}.run.task1.json`, the three `answers.task1.*.jsonl`
caches, and the file inventory under `videos_TEST_VQA/`.

**Class 3 — corpus-derived tuning constants that are _not_ flag-exposed.**
`TARGET_FRAMES 64`, `DENSE_TARGET_FRAMES 96`, `MAX_FPS 4.0`, `MIN_FPS 0.2`,
`MIN/MAX_FRAME_TOKENS 128/768`, `TEXT_TOKEN_RESERVE 2048`,
`VISUAL_BUDGET_SAFETY 0.8`, `BUDGET_POLICY` (`generate.answers.qwen3vl.py:147-199`),
the same frame/fps trio in `generate.answers.cosmos.py:128-130`, and
`MIN_FRAMES 8 / CAP_SHORT 16 / CAP_LONG 32 / LONG_VIDEO_SEC 60`
(`extract.frames.py:63-66`). All were sized against the 2026 corpus
(median 10.3 s, p99 233 s, max 389 s).

Handling: **leave every one at its 2026 value.** They are the configuration
under test; retuning them for short clips would mean the resulting number is not
the 2026 pipeline's number, which is the whole point. They are recorded verbatim
in `VQA/EVALUATION.2025.md` as the config the score belongs to.

The consequence the report must state plainly: on a 5–30 s corpus
`adaptive_fps = min(4.0, max(0.2, 64 / duration))` clamps to **2.1–4.0 fps for
every sampled clip**, and `LONG_VIDEO_SEC = 60` puts all 150 on Gemma 4's
16-frame `CAP_SHORT` path. The fps floor, the frame-token degrade ladder, the
32-frame long path and the 4 GiB `RAW_DECODE_BYTES_LIMIT` transcode are **never
exercised by this sample**. The score measures the pipeline on short video only
and says nothing about its long-video behaviour.

**Class 4 — cosmetic.** Docstrings, and the `(ground truth: mean 8.4, median 8,
p90 17)` line each script's `report()` prints. No behavioural effect; left alone.

**One contamination finding.** No script embeds few-shot examples, so there is no
label leakage. But `MAX_ANSWER_WORDS = 40` and the shared "most good answers are
under ten words" prompt rule were tuned against `VQA/vqa.gt.csv`'s length
distribution (`generate.answers.qwen3vl.py:269-273`, `generate.answers.py:108`)
— which **is** the 2025 ground truth being scored against. That is style
leakage, not answer leakage: the length-sensitive component of METEOR is mildly
flattered. It cannot be removed without changing the pipeline under test, so it
is disclosed rather than corrected.

**One cheap assertion to add** (in `queries`): intersect the 1,327 GT video IDs
with the 2,443 2026 query `Video_ID`s and fail if non-empty. The qwen3vl
docstring asserts the two sets are disjoint; nothing currently verifies it.

## Stages

Each stage declares `host`, `backend`, and an output marker; it is skipped when
its marker exists unless `--force`. Progress is journalled to
`eval2025.state.json` after every stage so a crash resumes in place.

| # | Stage | Backend | Command it drives |
|---|---|---|---|
| 1 | `gate` | none | Poll `answers.task1.qwen3vl.cache.jsonl` until it holds 2,443 distinct Q_IDs. `--no-gate` overrides. Refuses to touch the GPU before this passes. |
| 2 | `deps` | none | Verify `ffmpeg`/`ffprobe`/`wsl.exe`/`ollama.exe`; verify WSL venv imports `openai`; verify `--allowed-local-media-path` covers `/mnt/e/TREC_2026`; check `pandas, nltk, bert_score, sentence-transformers` and NLTK `wordnet`/`punkt`. `--install-deps` installs them into `E:\TREC_2026\.venv`. |
| 3 | `queries` | none | `sample.2025.eval.py convert` — CSV → `vqa.gt.2025.queries.json`, then the 1,327-row eligible pool (video on disk, Q_ID ∉ SKIP). Metadata JSON read as **utf-8**, not cp1252. |
| 4 | `durations` | none | `duration.report.py --queries vqa.gt.2025.queries.json --videos videos_GT_2025/videos --report duration.report.gt2025.json --html duration.report.gt2025.html` (~1,327 ffprobes, 8 workers, ~1 min) |
| 5 | `sample` | none | `sample.2025.eval.py select --n 150 --seed 20260814` — proportional draw, GT CSV, reserves, achieved-vs-pool report |
| 6 | `frames` | none | `extract.frames.py --queries vqa.2025.sample150.json --videos videos_GT_2025/videos --durations duration.report.gt2025.json --out videos_GT_2025/frames` (CPU) |
| 7 | `whisper` | none (GPU free) | `transcribe.whisper.py --videos-dir videos_GT_2025/videos --out-dir videos_GT_2025/transcription_openwhisper --only <150 ids>` (`--only` matches stems; 150 IDs ≈ 1.8 kB of argv) |
| 8 | `route` | ollama | `route.query.inputs.py --queries vqa.2025.sample150.json --out query.routing.2025.json --cache query.routing.2025.cache.jsonl` (~6 calls/query, 1 worker) |
| 9 | `variations` | ollama | `generate.query.variations.py --queries vqa.2025.sample150.json --out query.variations.2025.json --cache query.variations.2025.cache.jsonl` |
| 10 | `answers:gemma4` | ollama | `generate.answers.py --queries … --variations query.variations.2025.json --frames videos_GT_2025/frames --transcripts videos_GT_2025/transcription_openwhisper --out gemma4.run.2025.json --cache answers.2025.gemma4.cache.jsonl` (Windows, `--workers 1`) |
| 11 | `answers:qwen3vl` | vllm-qwen | WSL: `…/cosmos-reason2/.venv/bin/python VQA/generate.answers.qwen3vl.py --queries … --videos videos_GT_2025/videos --routing query.routing.2025.json --frame-index videos_GT_2025/frames/index.json --transcripts … --out qwen3vl.run.2025.json --cache answers.2025.qwen3vl.cache.jsonl --workers 4` |
| 12 | `answers:cosmos` | vllm-cosmos | WSL: same venv, `generate.answers.cosmos.py --queries … --videos videos_GT_2025/videos --frame-index … --transcripts … --out cosmos.run.2025.json --cache answers.2025.cosmos.cache.jsonl` |
| 13 | `validate` | none | `validate.ag.run.py vqa.2025.sample150.json <run> --warnings-as-errors --report-json <run>.validation.json` × 3 |
| 14 | `score` | none (GPU free) | `score.answers.py --gt vqa.gt.2025.sample150.csv --run gemma4… --run qwen3vl… --run cosmos… --out eval.2025.scores.json` |
| 15 | `report` | none | Render `VQA/EVALUATION.2025.md` — scores, sample composition, achieved-vs-target, attrition, caveats |

CLI: `--stages a,b,c` · `--from X --to Y` · `--smoke [N=5]` · `--dry-run` ·
`--force` · `--seed` · `--n 150` · `--wait-for-2026 SECONDS` · `--no-gate` ·
`--install-deps` · `--keep-servers` · `--restore-2026` (relaunch the Qwen3-VL
server at the end).

### Backend arbitration detail

- **teardown**: `systemctl --user stop cosmos-reason2`; `pkill -f "vllm serve"`;
  Windows `taskkill ollama.exe` / `ollama stop gemma4:12b`; then poll
  `nvidia-smi --query-gpu=memory.used` until < 2 GiB or 120 s.
- **vllm-qwen**: `nohup ./.venv/bin/vllm serve Qwen/Qwen3-VL-8B-Thinking --host
  0.0.0.0 --port 8009 --allowed-local-media-path /mnt/e/TREC_2026 --max-model-len
  24576 --gpu-memory-utilization 0.92 --enforce-eager --media-io-kwargs
  '{"video":{"num_frames":-1}}' --reasoning-parser qwen3 --skip-mm-profiling`
  — flags copied verbatim from the live process; `--skip-mm-profiling` is
  mandatory or the engine refuses to start. Poll `localhost:8009/v1/models`
  (reachable from Windows) up to 600 s.
- **vllm-cosmos**: `systemctl --user start cosmos-reason2`, poll `:8008/v1/models`.
- **ollama**: start `ollama.exe serve` detached on Windows, poll
  `:11434/api/tags`, then warm `gemma4:12b` with a one-token request.

Every server start/stop is logged with a timestamp and VRAM reading, so an
unattended overnight run is auditable afterwards.

## `VQA/sample.2025.eval.py`

Two subcommands. `convert` builds the query JSON and the eligible pool.
`select` does the draw:

- Bin each pool video by ffprobe duration from `duration.report.gt2025.json`
  (falls back to `metadata/<id>.json` `length`).
- Allocate 150 by largest-remainder proportional allocation → 13/43/88/6.
- Within each bin, shuffle with `random.Random(seed)`; first *k* are the sample,
  the next *k* are the ranked reserve.
- Emit `vqa.2025.sample150.json` (`Q_ID` int, `Video_ID`, `Question`),
  `vqa.gt.2025.sample150.csv` (headerless, the 4 GT columns, in Q_ID order),
  `vqa.2025.sample150.reserves.json`, and a report giving achieved vs pool vs
  the 2026 target — with the unreachable-bin shortfall stated, not hidden.

## `VQA/score.answers.py`

Reads the 2026 run JSON natively and **reproduces `eval.vqa.ag.py`'s aggregation
exactly** — anything else is not comparable to NIST's numbers:

- normalise `' '.join(text.strip().lower().split())` (`:35`);
- METEOR `meteor_score([word_tokenize(ref)], word_tokenize(hyp))` (`:109`);
- a query's score is the **mean over its ten answers** (`:125-126`), then the mean
  over queries (`:150-152`);
- NDCG = `ndcg_score(gains.reshape(1,-1), np.arange(m,0,-1).reshape(1,-1))`
  (`ndcg_from_gains`, `:37-51`);
- honour `SKIP = {18,27,…,1822}` (`:91`);
- a GT row with no matching responses scores **0** and still counts — which is why
  the scorer must be handed the **150-row** GT CSV, never the 2,000-row one.

Differences that are safe and deliberate: BERTScore runs as one batched
`score(cands, refs, lang="en")` over all 1,500 pairs then regroups (identical
values — `rescale_with_baseline` is off by default — but minutes instead of
hours), and STS is `STS_local`, cosine over `all-mpnet-base-v2` embeddings,
because the official UMBC service is dead. It is reported under that name only,
never as "STS", with a note that the official number could not be computed.

Also emits the paired comparison across the three runs: per-query deltas, fraction
improved, and a Wilcoxon signed-rank test (`scipy` is already installed).

## Verification

1. **Smoke first.** `python VQA/run.eval.2025.py --smoke 5` runs stages 3–15 over
   5 queries with `.smoke` artifacts: both WSL server launches, the Ollama swap,
   frame/transcript/routing/variation generation, all three answering scripts,
   validation and scoring. ~30 min. Nothing else should be run until this is green.
2. **Scorer parity.** `score.answers.py --parity-check` writes the headerless
   5-column CSV `eval.vqa.ag.py` expects and compares METEOR, BERTScore and both
   NDCGs over 50 queries with STS disabled. **Must agree to 3 dp** — this gates
   every number downstream.
3. **Oracle sanity.** Score the GT against itself: METEOR ≈ 1.0, BERTScore ≈ 1.0.
   Anything less means normalisation or alignment is wrong.
4. **Alignment.** Assert every scored Q_ID's `Video_ID` matches its GT row, and
   report how many sampled queries lost their video.
5. **Isolation, enforced per stage.** The guarded 2026 set (Class 2 above) is
   hashed before the run and re-verified **after every stage**, so a forgotten
   flag aborts at the stage that caused it rather than being discovered at the
   end. This is the concrete defence against both the Q_ID collision and the
   `duration.report.json` overwrite.
6. **Flag-coverage check.** The `deps` stage asserts that every `DEFAULT_*`-valued
   argument of every driven script is explicitly supplied by the orchestrator's
   command for that stage.
7. **Sample fidelity.** `sample.2025.report.md` shows achieved vs pool per bin and
   the GT answer-length distribution (pool: mean 8.5 words, median 8, p90 17).

## Cost

| Stage | Estimate |
|---|---|
| durations + sample + frames | ~5 min (CPU) |
| whisper (150 clips ≤30 s) | ~5 min |
| route + variations | ~40 min (Ollama) |
| gemma4 answers | 150 × 16.8 s ≈ **45 min** |
| qwen3vl answers | 150 × 65.7 s ÷ 4 workers ≈ **45 min** |
| cosmos answers | 150 × 27.1 s ÷ 2 workers ≈ **35 min** |
| server swaps (3 × ~4 min) | ~12 min |
| validate + score | ~10 min |
| **total unattended** | **≈ 3.5 h**, after the 2026 run finishes |

## Risks and stated caveats

- **The 2025 sample is not the 2026 task.** All clips are 5–30 s Shorts; the 2026
  set runs to 389 s with 6.3% over 150 s. The score measures the pipeline on short
  video only. Every number must be quoted as "on a 150-query 2025 sample", never
  as "our METEOR".
- **The pool is 1,327 of 2,000** because the download was interrupted at Q_ID
  1471. The sample is drawn from Q_ID 1–1471 only, which is a *positional* rather
  than random slice of the GT. An optional `backfill` stage
  (`download_test_videos.py --queries vqa.gt.2025.csv --output-dir videos_GT_2025
  --retry-failed download_summary.gt2025.json`) would close it; not required for
  n=150, and worth stating in the report either way.
- **`bert_score` vs `transformers 5.15.0`** pinned in the Windows venv. The `deps`
  stage imports and smoke-scores one pair; if it breaks, the fallback is a
  dedicated `E:\TREC_2026\.venv-eval`. The WSL vLLM venv must not be touched.
- **Stopping vLLM is destructive to the 2026 run**, which is live now. Stage 1
  gates on its completion; `--no-gate` exists but is not the default.
- **`STS_local` is not NIST's STS.** Named distinctly everywhere it appears.
- **Deadline 17 Aug 2026.** None of this blocks the submission; the unexpanded
  2026 runs remain the fallback throughout.

## Implementation notes

Three files written: `VQA/sample.2025.eval.py`, `VQA/score.answers.py`,
`VQA/run.eval.2025.py`. Every existing script is driven unchanged.

**What the build changed from the plan.**

- **`backend: none` split into `any` and `free`.** As planned, every CPU-only
  stage declared `none`, and `none` meant "tear down all servers" — so running
  `--stages queries` would have killed the live 2026 run for a stage that never
  touches the GPU. `any` now leaves the card alone; `free` claims it without
  starting a server (Whisper, BERTScore). The end-of-run teardown likewise only
  fires if this run actually brought a backend up.
- **The flag-coverage check is subparser-aware.** `sample.2025.eval.py` declares
  path flags on two subparsers, and requiring all of them on every step was a
  false positive. Flags are now grouped by receiver, and a group is only
  required if the step invokes it.
- **The oracle threshold was wrong at first.** METEOR cannot reach 1.0 on an
  exact match: NLTK's fragmentation penalty is `0.5 · (chunks/matches)³`, which
  costs a full 0.5 on a single-word answer, and 17% of this ground truth is one
  word. A perfect oracle scores **0.945**, not 1.0. BERTScore (1.000) is the
  strict alignment check; METEOR's floor is 0.85.

**Verified.**

| Check | Result |
|---|---|
| Scorer parity vs `eval.vqa.ag.py`, 20 queries × 10 answers | **exact to 3 dp** — METEOR 0.277, BERTScore 0.886, NDCG_METEOR 0.964, NDCG_BERT 0.997; **0 per-query disagreements** across all four metrics |
| Oracle (GT vs itself) | BERTScore 1.000, METEOR 0.945 (explained above) |
| 2026-artifact guard | clean re-check passes; a doctored hash aborts the stage |
| 2025 GT ∩ 2026 test videos | **empty** — the qwen3vl docstring's claim now actually checked |
| `bert_score` vs pinned `transformers 5.15.0` | works; the flagged risk did not materialise |
| Isolation | all 11 guarded 2026 artifacts byte-identical; the live 2026 run progressed 1,679 → 1,743 undisturbed |

**Measured, replacing the estimates.**

- 1,327 videos probed in 24.7 s, **0 ffprobe failures**. Median 15.44 s, mean
  17.0 s, range 5.53–30.49 s, 6.27 h total.
- The sample drew **16 / 45 / 85 / 4** across 5–10 / 10–15 / 15–30 / 30–60 s,
  not the 13/43/88/6 predicted from metadata: ffprobe durations run slightly
  longer than yt-dlp's rounded-down `length`, moving some clips up a bin. 150
  queries, 42.1 min of footage, median 15.28 s.
- Scorer packages installed into `E:\TREC_2026\.venv` (pandas 3.0.5, nltk
  3.10.3, bert-score 0.3.13, sentence-transformers 5.7.0); `transformers` was
  left at its existing pin.

**Stages already run:** `queries`, `durations`, `sample`. Everything downstream
of `frames` is gated on the 2026 Qwen3-VL run finishing.

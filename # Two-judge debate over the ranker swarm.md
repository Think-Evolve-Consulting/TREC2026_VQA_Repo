# Two-judge debate over the ranker swarm — Mac + local, driven by Q_ID

## Context

Seven VLMs have ranked all 2,443 MC queries. Six attempts at combining them
(Task 1 fusion, equal-weight RRF, accuracy-weighted RRF, the 781,248-fusion
nested-CV search, and the single judge in both arms) have all landed on the
Qwen3.8 baseline of 0.800 Top-1 / 0.884 MRR. Reading the two judge runs *against
each other* is what found the remaining headroom:

| On the 203 dev queries the panel disputed | n | correct |
|---|---|---|
| the two judges **agree** | 153 | 112 (baseline 109 there) |
| the two judges **disagree** | 50 | qwen38-judge 20, internvl-judge 18, **baseline 24**, **either-one-right 38** |
| oracle over both judges, all 203 | — | **150** vs baseline 133 |

Three measured facts drive every design rule below:

1. **Debate where the judges split.** On those 50, something present is right 38
   times and only 20–24 are captured.
2. **A deadlocked debate falls back to Qwen3.8, never to a judge** — on exactly
   that contested set the baseline (24) beats both judges (20, 18).
3. **Arbitration cannot fix counting.** 38 of the 203 disputed dev queries are
   "how many" and the two-judge *oracle* reaches only 24. That needs denser
   frames, not a better argument.

`VQA/debate.rank.mc.py` and `VQA/prep.debate.mc.py` already exist and are
mock-verified, but they assume two Colab servers reading a shared Drive mount.
**The topology has changed** and the protocol with it: judges now sit on two
machines that each hold their own copy of the dataset, and the orchestrator
addresses them by **Q_ID alone**.

## Target architecture

```
  Mac 192.168.1.12                      this workstation (RTX 3090)
  /Users/Aakash/Downloads/              E:\TREC_2026\videos_TEST_VQA\
        videos_TEST_VQA/
   ├── videos/  frames448/              ├── videos/  frames448/
   └── ranking/<Q_ID>.json  ◄── copied ─┤   ranking/<Q_ID>.json
        │                               │        │
   agent.debate.mc.py --side a          │   agent.debate.mc.py --side b
   :7801  reads its own files           │   :7802  reads its own files
        │ base64 images                 │        │ file:// refs
   LM Studio :1234                      │   llama.cpp :8013
   qwen/qwen3.8-27b                     │   InternVL3.5-30B-A3B Q4
        └──────── LAN, {"q_id": 1234} ──┴────────┘
                         │
              debate.rank.mc.py (orchestrator process)
              relays every statement, writes the verdict file
```

### Three processes, three roles

| # | Process | Runs on | Talks to | Owns |
|---|---|---|---|---|
| 1 | **Qwen3.8 judge agent** — `agent.debate.mc.py --side a` | the Mac, beside LM Studio | LM Studio `192.168.1.12:1234` over the OpenAI API; `/Users/Aakash/Downloads/videos_TEST_VQA/{ranking,videos,frames448}` on its own disk | its prompts, its frames (base64), its own opinion and rebuttals. Writes nothing outside its dataset copy |
| 2 | **InternVL judge agent** — `agent.debate.mc.py --side b` | this workstation | llama.cpp `:8013` with `--media-path`; `E:\TREC_2026\videos_TEST_VQA\{ranking,videos,frames448}` | the same, but frames go as `file://` refs instead of base64 |
| 3 | **Orchestrator** — `debate.rank.mc.py` | this workstation, its own process | agents 1 and 2 over the LAN, by Q_ID only | which queries are debated, the shared transcript, the round structure, the cache, the verdict file, `--max-overrules`, and the validation |

The two judges never talk to each other — every statement is relayed by the
orchestrator, so each side's context is exactly what we chose to show it, and
neither judge can write to the run file.

- **The wire carries a Q_ID, never pixels.** `POST /opinion {"q_id": …}` and
  `POST /round {"q_id": …, "round": n, "transcript": [...]}`. Each agent opens
  its own `ranking/<Q_ID>.json`, loads its own frames, and talks to the model
  beside it. This is the flow you specified, and it is the only shape that works:
  LM Studio cannot read a path — a `file://` URL returns
  `400 'url' field must be a base64 encoded image`, a bare path `400 Invalid url`.
- **Frames never cross the network.** The Mac agent base64s its local frames; the
  local agent sends `file://` references, which llama.cpp `--media-path` reads
  for ~1 KB a call.
- **Query-major scheduling.** Both agents work the same Q_ID at the same time and
  a query runs through all its rounds before the next one starts, so each
  server's prompt-prefix cache still holds those frames when R1 and R2 arrive.
- **Two rounds after the opinions, full transcript shared.** Every statement from
  both sides, verbatim and in order, goes into each later prompt — replacing the
  current templates, which pass only the opponent's latest pick and reason.
- **The orchestrator writes the verdict**, validates it, and never lets an agent
  write to the run file.

## Safety — the property that replaces the dev gate

The emitted run is **byte-identical to `qwen38.run.task2.json` except where the
panel disputes AND the debate converges on a different option.** Unanimous
queries, deadlocks, failed calls and a killed run all keep the baseline ranking,
so a half-finished file is still valid and submittable, and `--max-overrules N`
caps the total drift. This already holds in `emit()`
([debate.rank.mc.py:420](VQA/debate.rank.mc.py#L420)) and was proven with
never-concede mocks: SHA-256 identical to the baseline.

## Work

### 1. `VQA/build.ranking.folder.py` (new) — the Q_ID payloads

Writes `videos_TEST_VQA/ranking/` for the **1,346 disputed queries only**
(the 1,097 unanimous ones are copied through untouched and would never be
opened). Reuses `JUDGE.load_panel`, `JUDGE.panel_table`, `JUDGE.candidates_for`,
`JUDGE.routing_line`, `MUSE.load_queries/load_transcript/load_routing` — the
panel is read from the seven `*.run.task2.json` plus their caches, exactly as
the judge does it.

```
ranking/
  _manifest.json      video stem -> {n_frames, duration_sec}   (= debate.manifest.json, 1,139 videos / 42,897 frames)
  _index.json         the 1,346 Q_IDs, each with its stem and n_options
  <Q_ID>.json         one self-contained debate brief
```

Each brief carries: question, official option strings, video stem + duration,
the panel table (every model's ranking, Borda score, `firsts`, per-pass
agreement, self-reported confidence), the router verdict line, the transcript
text and language, the reliability card's per-model priors, the evidence entry
from `evidence.2026.json` (counting flag; CLAP per-option scores for 135 sound
queries), and the baseline pick. ~4 KB each, ~5 MB total — **you copy that one
folder to the Mac**; nothing else about the dataset has to move.

### 2. `VQA/agent.debate.mc.py` (new) — the sidecar, one script for both sides

`python3 VQA/agent.debate.mc.py --side a --dataset <root> --backend lmstudio
--endpoint http://127.0.0.1:1234 --model qwen/qwen3.8-27b --port 7801`

Requires Python 3 plus `requests` and `pillow` on the Mac (`pip3 install
requests pillow`). Three modes:

- **`--cut-frames`** — run once per machine before anything else. Reads
  `ranking/_manifest.json` and cuts `frames448/<stem>/001.jpg …` with ffmpeg at
  the project's mid-point sampling, 448×448: **64 frames for the 199 counting
  videos, 32 for the other 940**. This is the denser extraction you asked for;
  it is the one piece of evidence dev says arbitration cannot substitute for.
  Resumable (skips a stem whose frame count already matches), ~20–40 min,
  ~1.5 GB per machine. Carries the cap already paid for: n frames from a clip
  needs n/duration fps, so the count is capped at 8 fps of source, which is why
  199 of 231 counting videos get the full 64.
- **`--selftest`** — the vision gate, below.
- **serve** — `http.server` on `--port`, two routes. `/opinion` builds the R0
  prompt from the brief and the frames and returns the statement dict that
  `statement()` already produces (ranking repaired by `MUSE.as_permutation`,
  pick, confidence, cites, argument, stance, logprob, tokens, elapsed).
  `/round` takes the shared transcript and returns a rebuttal. `/health`
  reports model id, brief count, frame count and ffmpeg presence.

Backend differences live in one place: `lmstudio` base64-encodes each frame
(Pillow, already 448×448 so re-encode only), `llamacpp` emits `file://` URIs.
LM Studio returns **no logprobs** (measured: `logprobs=False`), so that signal
exists on the InternVL side only and the report must not pretend otherwise.

### 3. `VQA/debate.rank.mc.py` (rewrite of the transport, keep the contract)

Keep: `emit()`, `report()`, `debate_schema`, the resumable JSONL cache,
`--max-overrules`, `--limit`/`--only`/`--force`, the 10-consecutive-failure
abort. Replace:

- `Agent` ([debate.rank.mc.py:181](VQA/debate.rank.mc.py#L181)) posts a Q_ID to a
  sidecar instead of building a chat payload. All prompt construction, frame
  loading and schema enforcement move into the agent.
- `frame_refs` / `frame_context` / `evidence_block` / the `R*_TEMPLATE`s move to
  the agent, which is the only side that knows its own paths.
- `debate_one` ([debate.rank.mc.py:364](VQA/debate.rank.mc.py#L364)) becomes the
  relay: call both `/opinion` in parallel → if picks differ, append both
  statements to a shared transcript and call both `/round` with `round=1` → if
  still split, `round=2` → converged wins, else baseline. Two rounds, full
  transcript, exactly as specified.
- New flags `--agent-a`/`--agent-b` (host:port). Gone: `--media-root`,
  `--manifest`, `--evidence`, `--api-key-*`, `--model-*`.

The cache keeps every turn from both sides, so the whole debate is auditable
after the fact and a re-run costs nothing already decided.

### 4. Retire the Colab path

`notebooks/colab_serve_*.ipynb` and the `--media-root` flow no longer describe
how this runs. Leave the files, note at the top of each that the Mac/local
topology supersedes them.

## Verification, in order — each gate stops the run if it fails

**Gate 0 — does the Mac's Qwen actually see?** My probe got HTTP 200 for an
image request but **empty content**, so vision is unproven. `--selftest` sends
one real frame and one solid-grey frame with the same question and requires
different, frame-specific answers, then repeats with a 32-frame strip. A model
that answers identically, or from the text alone, is blind. **If it fails I stop
and tell you** rather than switching models — you asked to decide that yourself.
LM Studio's context also defaults low: it must be set to ≥ 16k for the
32-frame prompts, and this is where that shows up.

**Gate 1 — latency, the feasibility number.** 20 disputed queries end to end,
measuring seconds per call on each side at the real frame count. The Mac is a
32 GB machine running a 27B model over base64 images; if R0 costs much more than
~15 s a query there, 1,346 queries will not finish in an evening and the knob is
frames-per-call (32 → 16 on the Mac) or scope. I report the extrapolation before
committing to the full run.

**Gate 2 — the evidence is really arriving.** Prompt-token counts consistent
with 32 (and 64) frames, not 1; one counting query showing the dense strip and
the max-count instruction; one sound query showing the CLAP per-option scores;
`repaired` near zero.

**Gate 3 — blast radius.** Diff the emitted run against
`qwen38.run.task2.json`: the number of changed rank-1s must equal the reported
convergence-overrule count, and nothing else may differ.

**Gate 4 — validator.** `python validate_vqa_mc_2026.py --queries
vqa.2026.testingDataset.Task2_v2.json --run debate.run.task2.json
--warnings-as-errors`, 2,443 items, plus position bias near 14.9%.

## Cost, and what to expect

| Stage | Where | Estimate |
|---|---|---|
| build ranking folder | local | ~2 min |
| copy `ranking/` to the Mac | you | ~5 MB |
| `--cut-frames` | both machines, parallel | 20–40 min each |
| gates 0–2 | both | ~20 min |
| R0, 1,346 queries | both in parallel | **measured at gate 1** |
| R1 + R2 (~25% then ~15%) | both | ~40% of R0 |
| assemble, validate | local | ~15 min |

Honest expectation: the two-judge *oracle* on dev was worth +17 queries and a
real debate captures a fraction of an oracle; six combination attempts have
landed on the baseline. The reason to expect anything different is the evidence
the panel never had — 32–64 fresh frames where the panel averaged 14.5, the
explicit max-count instruction, and CLAP scores on 135 sound queries. This run
is unmeasured by choice, so the safety property is what stands between it and a
worse file. Recommended follow-up once 2026 is settled: run the identical
pipeline over the 203 dev disputed queries and score it, so the next campaign
knows whether debate is worth keeping.

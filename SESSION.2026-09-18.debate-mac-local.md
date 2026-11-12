# Session Notes — 2026-09-18 — The two-judge debate, rebuilt for a Mac and this workstation

This follows `SESSION.2026-09-17.debate-system.md`, which built the debate for two
Colab sessions sharing a Drive folder. This session moved it onto hardware we
control: the Qwen3.8 judge on a Mac running LM Studio, and the InternVL judge and
the orchestrator on this workstation. The judges are addressed by question ID
alone.

**Status: built, verified end to end against mock models and against the real
InternVL judge. Stopped at the first real smoke test, because the Mac judge
could not return one answer in 30 minutes.** No debate run file was produced.
The Task 2 submission is unchanged:
`qwen38 → qwen3vl → fused.weighted → gemma4.vllm → fused.rrf`.

## Architecture as built

```
  Mac 192.168.1.12                         this workstation (RTX 3090)
  /Users/Aakash/Downloads/videos_TEST_VQA  E:\TREC_2026\videos_TEST_VQA
    ranking/  frames448/  videos/            ranking/  frames448/  videos/
        |                                         |
  agent.debate.mc.py --side a :7801        agent.debate.mc.py --side b :7802
        | base64 images                           | file:// refs (--media-path)
  LM Studio :1234, qwen/qwen3.8-27b        llama-server :8013, InternVL3.5-30B-A3B Q4
        \_______________ LAN: {"q_id": N} ________/
                            |
               debate.rank.mc.py (orchestrator)
       relays statements, keeps the cache, writes and validates the verdict
```

Three processes, three roles:

| Process | Runs on | Does |
|---|---|---|
| **Qwen3.8 judge agent** | Mac | receives a Q_ID, reads its own brief and frames, calls LM Studio, returns a statement |
| **InternVL judge agent** | workstation | the same, against llama-server |
| **Orchestrator** | workstation | picks the questions, sends Q_IDs, merges both statements into one transcript and sends it back to both, decides, writes and validates the run file |

The judges never talk to each other. Neither judge can write to the run file.

## Decisions

| # | Decision | Made by | Why |
|---|---|---|---|
| 1 | Qwen judge on the Mac (LM Studio), InternVL judge local (llama.cpp), orchestrator as its own local process | user | topology change from Colab |
| 2 | Only a Q_ID goes over the wire; each judge reads `videos_TEST_VQA/ranking/<Q_ID>.json` and its own video data | user | each machine holds a full copy of the dataset |
| 3 | A small Python **agent** process sits beside each model server | forced by measurement, user approved Python + pip on the Mac | LM Studio cannot take a Q_ID or read a file: `file://` returns `400 'url' field must be a base64 encoded image`, a bare path `400 Invalid url` |
| 4 | Denser frames: 32 per video, 64 for counting videos, 448×448 | user (reaffirmed) | the panel saw ~14.5 frames per video; every model undercounts |
| 5 | Briefs only for the disputed questions | user | the 1,097 unanimous questions are copied through from the baseline and never debated |
| 6 | If the Mac's Qwen turned out blind: stop and report, don't swap models | user | not triggered, vision passed |
| 7 | Opening opinions, then **two debate rounds**, full transcript shared with both judges, relayed by the orchestrator | user | as specified |
| 8 | A deadlocked debate falls back to **Qwen3.8's panel ranking**, never to a judge; `--max-overrules` caps drift | carried over from the 17 Sep design | on the dev disputed set the baseline (24) beat both judges (20, 18) |
| 9 | If both judges agree at the opening, no rounds are run | Claude, noted here | agreement is already convergence; saves 2 × 2 calls |
| 10 | On convergence, judge A's full ranking is emitted | Claude | a fixed choice keeps re-runs reproducible |
| 11 | One question in flight at a time, all its rounds before the next | Claude | each server's prompt cache still holds that video's frames in later rounds |
| 12 | InternVL served with `-np 1 -c 32768` | Claude | a 64-frame prompt (18.1k tokens) must fit a single slot; the old `-np 2` gave 16k slots |
| 13 | Mac agent uses `--endpoint http://127.0.0.1:1234` | user asked, Claude confirmed | the agent and LM Studio share the Mac; `192.168.1.12` is only for reaching the Mac from here |
| 14 | Colab notebooks kept, marked superseded | plan | history |
| 15 | Smoke test stopped after 30 min with no Mac answer; all processes stopped | user | — |

## What was built

| File | What it is |
|---|---|
| `VQA/build.ranking.folder.py` (new) | writes `videos_TEST_VQA/ranking/`: one self-contained brief per disputed question, plus `_index.json`, `_manifest.json` (the frame plan) and `_card.txt` (the reliability card) |
| `VQA/agent.debate.mc.py` (new) | the judge agent, one script for both machines. Standalone: imports nothing from the repo (it copies `as_permutation`, `parse_json_object`, `seed_for` from the Muse script). Modes: `--cut-frames`, `--selftest`, and serve (`GET /health`, `POST /opinion`, `POST /round`) |
| `VQA/debate.rank.mc.py` (rewritten) | the orchestrator: health-checks both judges (same side labels, same brief count), relays the rounds, writes the resumable cache, emits the run and its `.tail.json`, runs `validate_vqa_mc_2026.py --warnings-as-errors` on both |
| `notebooks/colab_serve_*.ipynb` | a "superseded" note added at the top |

Each brief holds: the question, the official option strings, the video stem and
duration, the frame count, every panel model's ranking with its Borda score,
`firsts` and pass agreement, a pre-rendered panel table (so both judges read the
panel in the same words), the router verdict, the transcript (up to 4,000
characters), the counting flag and CLAP audio scores, and the baseline pick.

## Practical output

| Output | Result |
|---|---|
| `videos_TEST_VQA/ranking/` | **1,332 briefs**, 4.5 MB (268 counting, 135 with audio scores). 14 of the 1,346 disputed questions have no usable video, so they keep the baseline |
| `videos_TEST_VQA/frames448/` (local) | 1,139 videos, 42,897 frames, 1.2 GB, cut in 4.8 min; 1,133 exact, 6 short but usable |
| Mac copy | briefs and frames in place; the Mac agent reported 1,332 briefs and 1,139 videos |
| Debate run file | **none**. The smoke test cached nothing, so no run file was written; the empty cache file was deleted |

## What was verified

**Vision gate (Gate 0), passed on both judges.** One real frame and one solid
grey frame, same question:

| Judge | Real frame | Grey frame |
|---|---|---|
| Qwen3.8 on the Mac | "A young man wearing a black beanie, hoop earrings…" (62 s) | "a solid, uniform gray field…" (18 s) |
| InternVL local | "A person wearing a black beanie…" (3.1 s) | "a gray background with no discernible objects" (0.3 s) |

**The real InternVL judge, one full opinion:** 64 frames = 18,129 prompt tokens,
13.6 s, and it argued from the frames with cited times ("breaks the saw blade
twice, as seen in frames at 3.0s and 10.3s"). `file://` through `--media-path`
works: a single frame cost 342 prompt tokens, so the image really was read.

**The whole chain against mock models** (orchestrator → both real agents → two
fake model servers), on 30 real briefs (10 counting, 10 audio, 10 other):

- **Safety property holds.** With two judges that never agree, the output has
  **the same content as `qwen38.run.task2.json`** in all 2,443 entries. Only the
  JSON whitespace differs, so compare parsed JSON, not SHA-256.
- Convergence at round 1 works, and `--max-overrules 3` held drift to 3
  (23 suppressed).
- Request audit: judge A received base64, judge B `file://`; both got 64 frames
  on counting videos and 32 otherwise; the full transcript was in all 60 round
  prompts; the counting aid and CLAP scores reached both; logprobs were
  requested only from llama.cpp.
- Every emitted file passed the validator with `--warnings-as-errors`.

## What happened on the real smoke test

| Time | Event |
|---|---|
| 16:01:48 | orchestrator starts, both judges pass the health check (1,332 briefs each) |
| 16:01:50 | InternVL returns its opening opinion on Q3 (a 64-frame counting question), ~2 s |
| 16:01–16:32 | the Mac returns nothing for the same question |
| 16:32 | orchestrator stopped; nothing cached |

Q3 is the worst case: 64 frames, about 18k prompt tokens, and a thinking trace.
But even the best case is far out of reach: 1,332 questions × at least one call
each, at tens of minutes a call, is weeks.

## Measured facts about the Mac / LM Studio (traps)

- **It accepts only base64 images.** `file://` and bare paths are rejected with
  a 400.
- **It returns no logprobs**, so that signal exists on the InternVL side only.
- **Qwen3.8 always thinks before it answers, and `/no_think` doesn't stop it.**
  With a 16-token budget the whole budget went on thinking and `content` came
  back empty, which looks like a blind model but isn't. The agent's default
  budget is 4,096 tokens.
- **It was loaded with an 8,192-token context.** A request that doesn't fit
  hangs, with no error. Check `GET /api/v0/models` → `loaded_context_length`
  before any run. It was raised to 32,768 for the smoke test.
- **It is slow here:** about 5–7 tokens/s decode on a single-image probe, and
  16–60 s for one image. For comparison, Qwen3.8 INT4 on the 3090 measured
  ~980 tokens/s prefill and ~14 tokens/s decode
  (`SESSION.2026-09-17.token-throughput-bench.md`).
- **Stopping the orchestrator doesn't stop the Mac.** The agent gives each call
  30 minutes and retries once, so LM Studio stays busy until the Mac agent is
  stopped with Ctrl+C.

## Proposed next steps

1. **Make sure the Mac is idle.** Ctrl+C the Mac agent, check that LM Studio's
   server log shows no running request, and eject/reload the model if one is
   stuck.
2. **Measure before choosing.** Time direct LM Studio calls at 1, 8, 16 and 32
   frames on one non-counting question. This separates the time spent reading
   the images from the thinking time and says what frame count, if any, the Mac
   can serve. About 30 min.
3. **Pick a Qwen-judge setup from that number:**
   - **Mac, fewer frames:** `--max-frames 8` or `16` on the Mac agent only. No
     code change, but the two judges then see different evidence, which weakens
     frame-citing arguments.
   - **Mac, smaller scope:** debate only the 268 counting questions (the one
     category arbitration can't fix without new evidence) or another subset,
     via `--only`.
   - **Move the Qwen judge to the 3090** on its measured vLLM config (Triton
     attention, bfloat16 KV). Both judges can't share 24 GB, so this needs a
     phase mode in the orchestrator: all opening opinions from one model, swap
     models, all from the other, then the rounds for split questions only, with
     one swap per round. More model swaps, but the throughput is known.
4. **Then run the gates as planned:** a 20-question smoke test with per-call
   timing, the full run, the blast-radius check (the changed rank-1 count must
   equal the reported overrule count), and the validator.
5. **Measure it on dev as well.** The MC window closes today (18 Sep), so a 2026
   debate run is now research rather than a submission. Running the same
   pipeline over the 203 dev disputed questions would give the first real
   number for whether debate beats the baseline (0.800 / 0.884), which six
   earlier combination attempts did not.

## How to resume

```
# workstation: InternVL server (note -np 1 and --media-path)
E:\TREC_2026\tools\llama-b10976\llama-server.exe -m <InternVL Q4_K_M gguf> --mmproj <f16 mmproj> `
    --jinja -ngl all -fa on -c 32768 -np 1 --cache-type-k q8_0 --cache-type-v q8_0 `
    --media-path E:\TREC_2026\videos_TEST_VQA\frames448 --alias internvl3.5-30b-a3b --port 8013 --no-webui

# workstation: InternVL judge
python VQA/agent.debate.mc.py --side b --dataset E:/TREC_2026/videos_TEST_VQA --backend llamacpp \
    --endpoint http://127.0.0.1:8013 --model internvl3.5-30b-a3b --host 127.0.0.1 --port 7802

# Mac: Qwen judge (add --max-frames N if step 3 says so)
python3 agent.debate.mc.py --side a --dataset /Users/Aakash/Downloads/videos_TEST_VQA \
    --backend lmstudio --endpoint http://127.0.0.1:1234 --model qwen/qwen3.8-27b --port 7801

# workstation: orchestrator (resumable; --limit 20 for the smoke test)
python VQA/debate.rank.mc.py --agent-a http://192.168.1.12:7801 --agent-b http://127.0.0.1:7802 \
    --runs qwen38.run.task2.json qwen3vl.run.task2.json gemma4.vllm.run.task2.json cosmos.run.task2.json \
           internvl.run.task2.json llavavideo.run.task2.json videollama3.run.task2.json \
    --out debate.run.task2.json --cache rankings.task2.debate.cache.jsonl --limit 20
```

Either judge can be checked on its own with `--selftest` (the vision gate plus
one timed opinion) before the orchestrator is started.

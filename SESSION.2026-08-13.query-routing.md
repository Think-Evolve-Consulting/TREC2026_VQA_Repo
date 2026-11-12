# Session Notes — 2026-08-13 — Task 1 query input routing

Deciding, for every Task 1 question, **which inputs it actually needs** — frames,
the soundtrack, on-screen text, or a combination — before the answering stage
spends its context window on the wrong evidence. Six single-task agents judge the
question text alone; a plain-Python policy combines their verdicts.

Script written and full run executed 13 Aug (16:08 → 19:50 IST); analytics and
verification wrap-up 14 Aug.

**Deadline context:** all Answer Generation (AG) runs are due **17 August 2026**
per `Welcome to the TREC2026 VQA Track_JULY2026.md`.

---

## Files created this session

| File | Purpose |
|---|---|
| `VQA/route.query.inputs.py` | Six-agent router over the question text (29.3 KB) |
| `query.routing.task1.json` | The output — 2,443 entries, 4.49 MB |
| `query.routing.cache.jsonl` | Resumable work cache, one line per **(question, agent)** — 14,100 lines, 4.49 MB |

---

## 1. Why routing

The answering pass in `SESSION.2026-08-12.answer-generation.md` sends every query
down one path: sampled frames plus the Whisper transcript, one prompt shape for
all 2,443. That is wasteful in one direction and blind in the other.

- *"What does the first caption in this video say?"* needs the pixels **read**,
  not recognised. No stage in this pipeline does OCR.
- *"What music was playing while the people were on the bus?"* is settled
  entirely by the soundtrack; the 32 frames are dead weight in the context.
- *"What happens after the man gets a hair cut?"* cannot be answered from a
  single keyframe at all — it needs the clip in order.

The routing pass makes that explicit per query, as data the answering stage can
read, instead of leaving it implicit in one prompt template that has to serve
every question at once.

**Routing needs no video.** It reads the question text only, so it ran to
completion for all 2,443 queries — including the 868 undownloaded YouTube videos
that still block 35.5% of the *answers*.

---

## 2. Architecture — one agent, one job

Six specialists, each a separate Gemma 4 call with its own prompt and its own
JSON schema:

| Agent | The single question it answers | Returns |
|---|---|---|
| `vision` | must you recognise what is in the picture? | `needed` / confidence / reason |
| `audio` | must you hear the soundtrack? | ″ |
| `ocr` | must you read words rendered on screen? | ″ |
| `temporal` | does the answer depend on *when*, or in what order? | ″ |
| `inference` | is the answer worked out rather than observed? | ″ |
| `qtype` | what kind of answer is asked for? | one of 10 labels |

Two properties are the point of the design:

**They never see each other's verdicts.** Six independent judgements, so a weak
signal in one cannot talk the rest into agreeing — the usual failure of a single
"list the modalities this needs" prompt, which anchors on its own first clause
and produces internally consistent nonsense.

**The model's job ends at the six judgements.** How they combine is `plan()` —
plain Python, no model — so the policy can change and be re-applied over the
cache with `--rebuild` in seconds, with zero inference. The vision floor, the
confidence threshold and `--always-vision` all live there.

`Pipeline` in the output names what the answering stage should load, in the order
it goes into the context: `frames` (from `VQA/extract.frames.py`),
`asr_transcript` (from `VQA/transcribe.whisper.py`), `frame_ocr` — *which does
not exist yet* — then the strategy flags `dense_timestamped_frames` /
`reasoning_prompt`, then `answer_synthesis`.

---

## 3. Prompt design — the failure mode is over-claiming

Asked in isolation whether a question needs sound, a model will find a way to say
yes for almost anything set in the world. So every prompt states what belongs to
a *different* specialist, and gives the case that looks like a yes and is not.

The smoke test caught two false positives, both fixed before the full run:

| Query | First draft | Cause | After |
|---|---|---|---|
| *"What does the person do with the document after printing it out?"* | `ocr: true` | fired on the **word** "document" | `ocr: false` |
| *"What happens to the image of the bride and groom?"* | `inference: true` — *"'what happens' implies interpreting an action"* | treated plain description as inference | `inference: false` |

The fixes were the missing negative cases, not more rules:

- OCR — *"The test is whether the answer is the words themselves… merely
  mentioning something that carries writing — a document, a book, a letter, a
  phone, a shop, a sign — is not enough."*
- Inference — *"'What happens', 'what does he do' and 'what happens next' are
  observation, not inference: the event is on screen and only has to be
  described."*

Both agents held after the change, and the rest of the stress set was unaffected
— the prompts are independent, so a fix to one cannot regress another.

---

## 4. Caching — keyed on (question, agent)

Same dedup as the variations pass: **2,443 queries hold 2,350 distinct
questions** (39 repeat; *"what is this video showing?"* covers 11 Q_IDs), routing
reads only the question text, so each is judged once and fanned back out to every
Q_ID in `assemble()`. The output still has all 2,443 records.

Adding the **agent** to the key is what makes the stage extensible: a seventh
specialist costs 2,350 calls, not 14,100 — the six on disk are read straight
back. Same property covers a prompt revision to a single agent (`--force
--agents ocr`).

Reproducible: the seed is a SHA-256 of `agent \0 question`, so a re-run yields
the same routing. The agent name is in the digest deliberately — without it the
six calls about one question would share a seed, which is a correlation nobody
asked for across six different prompts.

---

## 5. Validation before the full run

**Smoke test** — 6 questions × 6 agents, verdicts read individually. Found the
two false positives above.

**Stress test** — 14 questions chosen by regex for audio and OCR flavour, plus
the two regressions. All 14 routed correctly:

| Query | Route | Flags |
|---|---|---|
| *"What does the first caption in this video say?"* | `ocr` | temporal |
| *"What were the last word the man said to the woman at the bar?"* | `audio` | temporal |
| *"What music was playing while the people were on the bus?"* | `audio` | temporal |
| *"what did the young man wearing a big cross said to his friend in his car"* | `vision+audio` | temporal |
| *"What number does the person write in the cement?"* | `vision+ocr` | — |
| *"Why does this man go to the gym every day according to the displayed text…"* | `ocr` | inference |
| *"How many times does the young man cut himself?"* | `vision` | temporal, `count` |

The pure `audio` and `ocr` routes — dropping frames entirely — are the ones worth
watching. *"the man… at the bar"* routes to `audio` alone, giving up the frames
that pick out **which** man. That is a downstream call, so it became the
`--always-vision` flag rather than a hardcoded floor, and `--rebuild` flips it
either way over the cache.

---

## 6. Run results

```
python VQA/route.query.inputs.py --workers 4
```

| | |
|---|---|
| Judgements | **14,100 / 14,100** (2,350 questions × 6 agents) |
| Wall clock | **170.3 min** (11:30 → 14:20 UTC, 13 Aug) |
| Throughput | 82.8 judgements/min |
| Mean / median / max per call | 2.90 s / 2.94 s / 7.49 s |
| Failures | **0** |
| Coverage | 2,443 / 2,443 queries fully routed |

### `--workers 4` is right here, and `--workers 1` is right for frames

Measured this session: **45/min at `--workers 1`, 82.8/min at `--workers 4`** —
summed call time 680.6 min against 170.3 min wall clock, an effective parallelism
of **4.00×**. Exactly the 4× ceiling the variations session hit, and for the same
reason: Ollama's `OLLAMA_NUM_PARALLEL` is unset and defaults to four slots.

This looks like it contradicts `SESSION.2026-08-12.answer-generation.md`, which
measured `--workers 4` as *slower* than one worker and set `DEFAULT_WORKERS = 1`.
Both are correct — the difference is the payload:

- **Frame prompts** carry a large image prefix. Four concurrent requests from
  different videos evict each other's prefix from the KV cache, and the prompt
  re-evaluation costs more than the concurrency saves (5.53 s vs 0.59 s prompt
  eval).
- **Text prompts** here share only a short system prefix. There is nothing
  expensive to evict, so the four slots are pure gain.

The rule to carry forward: **concurrency pays when the prompt prefix is cheap to
recompute.** The script still ships `DEFAULT_WORKERS = 1`, matching the house
default; pass `--workers 4` for text-only stages.

---

## 7. What the routing found

### Required inputs (share of the 2,443 queries)

| Input | Queries | Share |
|---|---|---|
| vision | 2,259 | 92.5% |
| audio | 576 | **23.6%** |
| ocr | 57 | 2.3% |

### Routes

| Route | Queries | Share |
|---|---|---|
| `vision` | 1,839 | 75.3% |
| `vision+audio` | 379 | 15.5% |
| `audio` | 168 | 6.9% |
| `vision+audio+ocr` | 23 | 0.9% |
| `vision+ocr` | 18 | 0.7% |
| `ocr` | 10 | 0.4% |
| `audio+ocr` | 6 | 0.2% |

### Question types

`action` 30.8% · `entity` 18.9% · `reason` 17.1% · `count` 12.8% ·
`manner` 8.3% · `place` 4.2% · `time` 3.2% · `summary` 2.4% · `yes_no` 2.0% ·
`other` 0.3%

### Strategy flags

| Flag | Queries | Share |
|---|---|---|
| `inference` — answer not directly observed | 1,136 | 46.5% |
| `temporal` — depends on order or moment | 886 | 36.3% |
| defaulted to vision (every specialist declined) | 9 | 0.4% |

`temporal` tracks question type sensibly: 62.8% of `action` questions but only
8.7% of `place` ones.

### Three findings that change what to build next

**1. Nearly a quarter of the set needs the soundtrack.** 576 queries, of which
**168 need it and nothing else**. The current answering pass does pass a Whisper
transcript, so these are not answered blind — but they compete with 32 frames for
context, and for the 168 the frames are pure noise.

**2. 57 queries need OCR, and there is no OCR stage.** Small in count, but
currently near-unanswerable: *"What does the first caption in this video say?"*
cannot be reached by frame recognition. The 10 `ocr`-only and 6 `audio+ocr`
queries have no viable path at all today.

**3. Nine queries need no video whatsoever.** The ones where every specialist
declined are not misjudgements — they are **world-knowledge questions**:

> *"what is the Hudson Institute?"* · *"What is an Easter egg?"* ·
> *"What is a likely reason ducks swim in a body of water?"* ·
> *"Is it easy to make a skype group video call?"*

`plan()` falls back to vision for these, which is the least wrong option, but a
seventh `world_knowledge` agent would name the category properly. `Defaulted:
true` already flags them for anyone who wants to handle them separately.

### Confidence is not usable as a signal

Across all per-query judgements: **96.9% came back at exactly 1.0**, 3.0% between
0.9 and 1.0, ten judgements between 0.7 and 0.9, and **none below 0.7**.

At temperature 0 the model asserts rather than calibrates, so `--threshold` is
effectively inert at its 0.5 default and there is no low-confidence tail to
review by hand. Do not build a confidence-ranked triage on this field. If
graded uncertainty is wanted, it has to come from elsewhere — sampling the same
judgement several times and counting disagreement would be the honest way.

---

## 8. Output shape

```json
{
    "Q_ID": 208,
    "Video_ID": "CeQYbNaDcUg",
    "Question": "What does the first caption in this video say?",
    "Question_Type": "entity",
    "Required_Inputs": ["ocr"],
    "Route": "ocr",
    "Temporal": true,
    "Inference": false,
    "Defaulted": false,
    "Pipeline": ["frame_ocr", "dense_timestamped_frames", "answer_synthesis"],
    "Signals": {
        "vision":    {"needed": false, "confidence": 1.0, "reason": "The prompt states reading printed text on screen is not considered watching."},
        "audio":     {"needed": false, "confidence": 1.0, "reason": "Captions are visual elements; the answer can be seen without listening."},
        "ocr":       {"needed": true,  "confidence": 1.0, "reason": "The question specifically asks for the content of a written caption."},
        "temporal":  {"needed": true,  "confidence": 1.0, "reason": "The question specifies 'first', which refers to a sequence in time."},
        "inference": {"needed": false, "confidence": 1.0, "reason": "The text of a caption is directly visible and can be reported by observation."},
        "qtype":     {"type": "entity", "confidence": 1.0, "reason": "The question asks for a specific piece of text from the video."}
    }
}
```

Every verdict keeps its one-line reason, so a routing decision can be audited
without re-running anything.

---

## 9. Where the output and the analytics are

**`query.routing.cache.jsonl`** — the raw record, one line per judgement,
appended and flushed as each call returns. This is the file to tail during a run.

**`query.routing.task1.json`** — the assembled per-query file, written **only at
the end** of a run (or by `--rebuild`).

**Analytics** are the distribution tables `summarize()` prints, and `--rebuild`
regenerates them from the cache in seconds without inference:

```bash
python VQA/route.query.inputs.py --rebuild --out scratch.routing.json
```

Point `--out` away from the default while a run is in flight, or the rebuild and
the run's final write fight over the same file.

**Mid-run numbers are skewed, by design.** Work is ordered by agent so each
specialist's prompt prefix stays in the KV cache, which means partial results are
*complete for early agents and empty for later ones* rather than uniformly
sampled. Halfway through this run the route table read "vision 76.4%" only
because the vision agent had not started and those were `Defaulted`
placeholders. Until a run finishes, per-agent counts are the honest view.

---

## Command reference

```bash
# Full run — 14,100 judgements, ~2.8 h at --workers 4, resumable
python VQA/route.query.inputs.py --workers 4

# Smoke test
python VQA/route.query.inputs.py --limit 6

# Specific queries by Q_ID
python VQA/route.query.inputs.py --only 208 428 432

# Re-emit the JSON from cache, no inference
python VQA/route.query.inputs.py --rebuild

# Change the policy without re-running the model
python VQA/route.query.inputs.py --rebuild --always-vision
python VQA/route.query.inputs.py --rebuild --threshold 0.9

# Re-judge one agent after a prompt change (leaves the other five cached)
python VQA/route.query.inputs.py --force --agents ocr --workers 4
```

Ctrl-C is safe at any point — finished judgements are on disk and a re-run
resumes.

---

## Environment verified

- Ollama with `gemma4:12b` (7.56 GB), RTX 3090
- `OLLAMA_NUM_PARALLEL` unset — four slots by default, confirmed at exactly 4.00×
- `think: false`, `temperature 0.0`, `num_predict 160`; `num_ctx` left at the
  default deliberately — these are text-only prompts a few hundred tokens long,
  so the 4096 trap that governs the frame prompts does not apply
- `requests` — the only dependency

---

## Next steps

1. **Wire `Required_Inputs` into the answering pass.** The 168 `audio`-only
   queries should get the transcript without frames; the 379 `vision+audio` ones
   should get both, weighted toward the transcript. This is the payoff for the
   whole stage and it is not yet connected.
2. **Build the OCR stage** — 57 queries need it and nothing provides it.
   Frame-level OCR over the existing `videos_TEST_VQA/frames` output is the
   cheapest route, since the frames are already extracted.
3. **Use `Temporal` to drive frame sampling.** 886 queries depend on order or
   moment and want timestamped frames across the clip; the other 1,557 could take
   fewer, better-chosen frames instead.
4. **Consider a `world_knowledge` agent** for the 9 `Defaulted` queries, and
   check whether more hide inside the `vision` route.
5. **Validate routing against 2025 ground truth** if the GT videos ever land —
   the routing has been checked by reading 20 questions, not measured. Whether
   a routed answering pass actually scores better than the flat one is still an
   open question, and the METEOR/BERTScore harness in `VQA/eval.vqa.ag.py` is
   how to settle it.

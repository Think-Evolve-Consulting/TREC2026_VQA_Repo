# Session Notes — 2026-08-12 — Task 1 answer generation with Gemma 4

Built the pass that actually looks at the videos: frames sampled with ffmpeg,
handed to **`gemma4:12b`** through Ollama alongside the Whisper transcript, one
call per prompt phrasing, pooled and ranked into the ten answer slots the run
format requires.

Frames are extracted for all **1,940** videos. The answering pass is verified
end to end on a 20-query smoke set that **passes `validate.ag.run.py` with
`--warnings-as-errors`**. The full 2,443-query run has not been launched.

**Deadline context:** Answer Generation runs are due **17 August 2026**.

---

## Files created this session

| File | Purpose |
|---|---|
| `VQA/extract.frames.py` | ffmpeg frame sampler, resumable, one pass per video |
| `VQA/generate.answers.py` | The VLM answering pass — prompts, pooling, ranking, driver |
| `videos_TEST_VQA/frames/<stem>/NNN.jpg` | 26,250 frames over 1,940 videos, 1.22 GB |
| `videos_TEST_VQA/frames/index.json` | Per-video frame count, files and sampling timestamps |

Pre-existing inputs consumed: `vqa.2026.testingDataset.Task1.json`,
`query.variations.task1.json`, `duration.report.json`,
`videos_TEST_VQA/transcription_openwhisper/`.

---

## 1. What Gemma 4 actually does with video

Checked against the model card, the Ollama library page and the KerasHub guide
before writing anything, because the answer changes the whole design:

- **There is no video input.** A video is a sequence of timestamped frames
  through the same vision tower as still images. The model card caps video at
  *"a maximum of 60 seconds assuming the images are processed at one frame per
  second"*.
- **Ollama has no `video` field and no documented `audio` field.** `/api/chat`
  messages carry `content` plus an `images` array of base64 strings. So frames
  are ours to extract, and the audio channel has to arrive as ASR text — which
  is exactly what the Whisper pass already produced. (`ollama show gemma4:12b`
  advertises an `audio` capability, but there is no API surface to reach it.)
- **Ollama defaults `num_ctx` to 4096** whatever the model's window is. 32
  frames cost ~6,700 tokens on their own, so without raising this the frames
  fall out of context, the run completes, and every answer is worthless. This
  was the single largest silent-failure risk in the design.
- **Images go before the text.** Documented as materially better for Gemma 4,
  and it turns out to matter for throughput too (§3).

**Multi-image is real.** Three frames in one message came back described
individually and in order — measured **~210 tokens per image**.

### Which variant, and the upgrade path

Inference runs on **`gemma4:12b`** — the 12B unified variant, digest
`4eb23ef187e2`, the only Gemma 4 pulled on this box and the same model that
generated `query.variations.task1.json`.

| | |
|---|---|
| Parameters | 11.9 B, plus a 52 M CLIP projector |
| Quantisation | Q4_K_M, 7.6 GB on disk |
| Native context | 262,144, run here at **16,384** (`--num-ctx`) |
| VRAM loaded | **8.39 GB** of the 3090's 24 GB at that context |
| Capabilities | completion, vision, audio, tools, thinking |
| Environment | Ollama 0.32.8, RTX 3090, float16 host |

The family is `e2b` (7.2 GB, 128K), `e4b` (9.6 GB, 128K), **`12b` (7.6 GB,
256K)**, `26b` (18 GB, 256K), `31b` (20 GB, 256K). `12b` is the sweet spot: the
256K context class, which the E-variants do not reach, and at 8.39 GB loaded it
leaves ~16 GB of headroom, which is what makes the 32-frame prompts comfortable.

**A larger variant is the most likely quality win and `26b` would fit in 24 GB**,
but it costs roughly 2–3× the inference time — pushing the run from ~10.4 h to
20–30 h+, which against 17 August is one shot rather than two — and it is an
18 GB download that has not been made. If it is tried, `--model gemma4:26b` is
the only change needed, but give it a separate `--cache`: the cache is keyed on
`Q_ID` alone and would otherwise blend two models' answers.

One documentation wrinkle, moot in practice: `ollama show` reports an `audio`
capability for `12b` and Google's docs say audio is native to E2B/E4B/12B, while
the Ollama library page lists `12b` as text+image only. Ollama exposes no audio
field on `/api/chat` either way, which is why audio arrives as Whisper text.

### The long tail

189 videos (422 queries) run past the 60 s ceiling, the longest at 389 s.
Rather than segmenting, frames are sampled **uniformly across the whole clip**
and each is labelled with its mm:ss timestamp in the prompt — below 1 fps for
long videos, which is the remedy the 60 s-at-1 fps framing implies. One code
path for every video, no extra inference.

```
cap = 32 if duration > 60 else 16
n   = clamp(round(duration), 8, cap)
```

| clip | frames | effective fps |
|---|---|---|
| median, 10.3 s | 10 | 1.00 |
| 60 s | 16 | 0.27 |
| 389 s, longest | 32 | 0.08 |

---

## 2. Frame extraction

```
python VQA/extract.frames.py
```

**1,940 videos → 26,250 frames, 1.22 GB, 6.1 min, 0 failures.**

Durations are read from `duration.report.json` rather than re-probed. One
ffmpeg call per video: `fps=n/duration:start_time=duration/2n` puts frames on
the *mid-points* of their intervals, so nothing lands on a black first frame,
and `scale` fixes the long edge at 896 px. JPEG bytes go straight from ffmpeg
to disk to `base64` — **no new Python dependencies**, in particular neither
Pillow nor OpenCV, neither of which is installed.

Frames are extracted once per video and reused by every query about it and
every prompt phrasing of those queries: 26,250 extractions instead of 366,000.

Resumable; `--force` re-extracts, `--reindex` rebuilds `index.json` from disk
if a run is killed before it writes one. The `--only=<ID>` form is required for
the 12 `Video_ID`s that begin with `-`.

---

## 3. Throughput — one worker beats four

The ten calls for a query share a byte-identical prefix (system + images +
context block); only the trailing question differs. Ollama's KV cache therefore
pays the image prefill once. Measured on a 32-frame clip:

| | prompt eval |
|---|---|
| first call on a video | **5.53 s** |
| next call, same frames | **0.59 s** |

Queries are grouped by video and a video's queries stay on one worker so the
reuse extends across the up-to-5 queries that share it.

**`--workers 4` is slower than `--workers 1`.** Same 20 queries:

| workers | wall clock | rate |
|---|---|---|
| 4 | 0.15 h | 2.2 queries/min |
| **1** | **0.09 h** | **3.9 queries/min** |

`OLLAMA_NUM_PARALLEL` is unset on this box, so the server runs a single slot.
Extra client threads buy no concurrency — they interleave requests from four
different videos through one KV cache and evict each other's image prefix, so
every call pays full prefill. Per-query time went from 76–155 s down to 10–21 s.
The default is now 1. Raising `OLLAMA_NUM_PARALLEL` and `--workers` together is
the lever if this ever needs to be faster; each extra slot costs its own
`num_ctx` of KV cache.

---

## 4. Prompting and pooling

Per query: the released `Question` plus `Variations[0:9]` — the original and
nine paraphrases, taken deterministically. Each call is schema-constrained to
its **three** most likely answers plus a confidence, so thirty candidates are
pooled per query.

The context block, built once per query and identical across its ten prompts:

```
This video is 389 seconds long. You are shown 32 frames sampled evenly
across it, in order, at 00:06, 00:18, ... 06:23.

Spoken audio (automatic transcript, language: en):
"..."
```

Transcript is the suspect-filtered `transcript` field, **Whisper only** — no
FunASR fallback, as instructed. The language is only asserted when
`language_source == "window_sweep"`; the 103 `whisper_default` labels were read
off the first 30 s including music and are not worth stating. Silent videos get
"The video has no detected speech." rather than nothing, so the model does not
invent dialogue.

Answer style is pinned to the ground-truth distribution measured from
`VQA/vqa.gt.csv` — **mean 8.4 words, median 8, p90 17, 17% single-word** — so
the prompt asks for a phrase or one short sentence and forbids restating the
question or narrating the clip.

### Ranking

1. Normalise with the validator's own rule, `" ".join(t.split()).casefold()`,
   verified equal to `validate.ag.run.py`'s `normalize_answer` in a direct test.
2. Weighted vote: a first-choice candidate scores 1.0, second 0.5, third 0.25,
   scaled by `0.5 + 0.5 × confidence` so confidence modulates without ever
   cancelling a vote.
3. **Refusals sort last** among the model's own answers. "unknown" is never the
   ground truth, so it scores ~0 on METEOR and BERTScore; ranking it above a
   substantive candidate wastes a slot and drags the NDCG gain vector down. It
   is demoted, not discarded.
4. Ties break on the prompt that found the candidate first, then the text —
   deterministic, no RNG.

Ranking is worth caring about beyond correctness: `eval.vqa.ag.py` feeds the
submitted order straight into NDCG.

### Filling ten distinct slots

The validator rejects duplicates *and* rejects fewer than ten, while
`eval.vqa.ag.py` scores a query as the **mean** METEOR/BERTScore across all ten
answers — so a slot filled with junk costs real score. Chain: pooled candidates
first; if short, one text-only call for meaning-preserving rephrasings of the
rank-1 answer; if still short, generic non-answers. On the smoke set the pool
alone covered 17 of 20 queries, 3 needed rephrasings, **none reached the
generic fallback**.

### Timing — the `Time` field

The track guidelines define it as **`"Time": "float (time to generate this
answer)"`**, per answer, in seconds (their examples are `2.5`, `1.5`, `7`). The
guidelines do not say whether preprocessing counts, so what is reported is
stated explicitly here:

| Answer source | `Time` |
|---|---|
| Model answer | wall clock of the inference call that produced it, **plus** its share of preparing that video's frames |
| Rephrased top-up | the top-up call's wall clock, split across the answers it filled |
| Generic fallback | frame-preparation share only (0 if the query had no frames) |

Frame reading and base64 encoding are done **once per video** and reused by
every query on it and every answer slot, so that cost is divided the same way
rather than charged to the first answer. It turns out to be negligible —
0.0016 s per answer, because the frames are still in the OS page cache.

The offline ffmpeg extraction in `VQA/extract.frames.py` is **excluded**: it is
a separate one-off pass over the corpus and, at 0.19 s per video amortised
across up to 50 answers, is noise beside inference. Same for model load, which
Ollama does once for the whole run.

Summed over a query, `Time` lands within a few percent of that query's true
wall clock — the small drift is because a single call that contributes two
distinct surviving answers has its time counted against both:

```
Q1:   8 frames  inference 14.94 s   sum(Time) 15.95 s
Q2:  32 frames  inference 21.31 s   sum(Time) 21.11 s
Q265: 0 frames  inference 10.14 s   sum(Time) 10.09 s
```

The end-of-run report prints `Time` mean/min/max per answer, the per-query
total, and a count of zero-valued entries, so this is checkable rather than
assumed. On the smoke set: mean **1.71 s** per answer, **17.07 s** per query,
**zero zero-valued entries**.

---

## 5. Two bugs the smoke test caught

**`none` was being treated as a refusal.** For *"How many people are speaking
in the garden?"* the top-voted answer — `none`, 9 votes — was demoted to rank 8
by the refusal rule, handing rank 1 to a 4-vote alternative. For counting and
"who did X" questions, `none`/`no one`/`zero` *are* the answer. They are now
excluded from the pattern.

**Temperature 1.0 left the vote nothing to rank.** The model's own default
produced up to 30 distinct candidates for one query, all at confidence
0.90–0.95, so rank 1 fell to the tie-break rather than to consensus. At 0.6 the
pool concentrated (19.3 distinct per query against 21.9) at identical cost, and
top answers carry real vote counts. Diversity is supposed to come from the ten
phrasings, not from sampling noise.

> This was judged on **vote concentration, not accuracy**. A ground-truth
> comparison is not possible here: the 2025 GT in `VQA/vqa.gt.csv` covers 2,000
> videos, **none** of which are on disk, and neither `nltk` nor `bert_score` is
> installed. See "Next steps".

---

## 6. Smoke test

20 queries chosen to hit every hard case — 3 videos over 120 s, 4 with no video
at all, silent clips, sub-5 s clips, speech-heavy clips.

```
answered 20 queries (0 failed) in 0.09 h

answer words  mean  4.7  median 5  p90 8   (ground truth: mean 8.4, median 8, p90 17)
distinct candidates pooled per query, mean 19.3 of 30
queries topped up with rephrasings: 3
queries needing a generic fallback: 0
rank-1 answer is "unknown":         0
queries with a failed prompt call:  0
```

```
python VQA/validate.ag.run.py subset.queries.json final.run.json --warnings-as-errors
VALIDATION PASSED — 20/20, 0 errors, 0 warnings
```

Answers read sensibly. *"How many times does the young man cut himself?"* →
`zero` / `none` / `0`. *"What is this video all about?"* on a 320 s clip →
*"A director giving notes to an actor"*. The 16 queries whose video was never
downloaded fall back to a text-only prompt and produce plausible generic
answers rather than failing.

**Answers run shorter than ground truth** — 4.7 words against 8.4. Not
obviously wrong (17% of GT answers are a single word, and terse answers suit
counting questions), but it is the most likely place to gain METEOR recall.

---

## 7. Command reference

`PYTHONIOENCODING=utf-8` is needed on Windows — non-Latin transcripts crash the
cp1252 console encoder on print.

```powershell
# Frames — done, 6 min, resumable
.\.venv\Scripts\python.exe VQA\extract.frames.py

# THE FULL RUN — ~10.4 h at the measured 3.9 queries/min. Resumable.
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe VQA\generate.answers.py

# Validate
.\.venv\Scripts\python.exe VQA\validate.ag.run.py `
    vqa.2026.testingDataset.Task1.json gemma4.run.task1.json `
    --warnings-as-errors --report-json gemma4.run.validation.json

# Re-emit the run from the cache with no inference
.\.venv\Scripts\python.exe VQA\generate.answers.py --rebuild

# Smoke test / specific queries / re-answer
.\.venv\Scripts\python.exe VQA\generate.answers.py --limit 20
.\.venv\Scripts\python.exe VQA\generate.answers.py --only 1 3 264
.\.venv\Scripts\python.exe VQA\generate.answers.py --force --only 1
```

Ctrl-C is safe — each query's record is flushed to
`answers.task1.cache.jsonl` as it completes and a re-run resumes.

Tuning flags: `--temperature` (0.6), `--candidates` (3), `--prompts` (10),
`--num-ctx` (16384), `--num-predict` (256), `--workers` (1), `--retries` (3),
`--think` (off).

### Reading the validator

Positional order is **released queries first, your run second** — swapping them
exits 2 with *"The released testing-query file is invalid"*. The backtick is
PowerShell line continuation; use `\` from bash.

`--warnings-as-errors` fails the run on either of the validator's two warning
types: an `Answer` with leading/trailing whitespace, and answer objects not in
ascending `Rank` order. Neither would sink a submission, but both mean sloppy
serialisation, so it is worth failing locally.

`--report-json` writes the complete error list; the terminal caps each category
at 100 messages (`--max-messages`).

Two flags deliberately **not** passed: `--allow-extra-fields` (so stray fields
are errors — which is why vote counts live in the cache, not the run) and
`--answers-per-query` (defaults to the 10 we want).

| Exit code | Meaning |
|---|---|
| 0 | valid |
| 1 | validation failed — errors, or warnings under `--warnings-as-errors` |
| 2 | fatal — file missing, unreadable, malformed JSON, or a bad query file |

> Error paths are 0-indexed while `Q_ID` starts at 1 (`VQA/README.md`), so
> `submission[41]` is `Q_ID` 42.

---

## 8. Known limitations

- **Sub-1 fps on long videos.** A 389 s clip is 32 frames, one every 12 s. An
  event shorter than that between samples is invisible. The transcript is the
  only cover for those, and 43% of the corpus has no speech.
- **Vision-only for 1,063 queries.** Whisper found no transcript for them.
  Of those, 478 *do* have a SenseVoice transcript in `transcription_funasr`,
  which is deliberately unused here — mostly sung vocals that Silero VAD
  rejected.
- **16 queries have no video at all** and are answered from the question text
  alone. They must still appear in the run or it fails validation.
- **No accuracy measurement.** Everything above is format correctness and
  plausibility. Nothing here says how often the answer is right.
- **Not bit-reproducible across concurrency settings.** Seeds are fixed per
  (Q_ID, prompt index), but changing `--workers` changes batching and therefore
  sampling. Fixed config, fixed output.
- **Frames are 896 px at JPEG q3.** Fine-grained OCR-ish questions ("what does
  the sign say") may be under-resolved; Gemma 4 supports a higher visual token
  budget but Ollama exposes no way to select it.

---

## 9. Next steps

1. **Launch the full run** (~10.4 h) and validate it.
2. **Get an accuracy number.** The honest way is to download a sample of the
   2025 test videos, run this pipeline over `VQA/vqa.gt.csv`, and score with
   `eval.vqa.ag.py` — which needs `nltk` and `bert_score` installed. That would
   turn the temperature, candidate-count and frame-budget settings from
   judgement calls into measurements, and would say whether the short answers
   are costing METEOR recall.
3. **Consider the FunASR fallback** for the 478 queries where SenseVoice heard
   speech and Whisper did not. One flag's worth of work if step 2 shows audio
   context is carrying the speech-heavy questions.
4. Confirm the data agreement form is submitted — runs are not scored without it.

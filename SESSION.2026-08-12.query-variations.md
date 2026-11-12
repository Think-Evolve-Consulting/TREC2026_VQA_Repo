# Session Notes — 2026-08-12 — Task 1 query variation generation

Generating ten paraphrases of every Task 1 question with a local Gemma model, so
the downstream VQA runner can ask each question ten ways and pool the answers.

**Deadline context:** all Answer Generation (AG) runs are due **17 August 2026**
per `Welcome to the TREC2026 VQA Track_JULY2026.md`.

---

## Files created this session

| File | Purpose |
|---|---|
| `VQA/generate.query.variations.py` | Generates N paraphrases per question via Ollama |
| `query.variations.task1.json` | The output — 2,443 entries, 2.31 MB |
| `query.variations.cache.jsonl` | Resumable work cache, one line per distinct question |

---

## 1. Why variations

The run format wants **ten ranked answers per query** (see
`SESSION.2026-08-11.task1-pipeline.md` for the full validator contract). Asking a
VQA model the same question ten different ways and pooling what comes back is a
cheap ensemble that fills those ten slots with something better than ten samples
off one prompt — phrasing sensitivity is real, and a rewrite that happens to
match the model's training distribution can surface an answer the original
phrasing misses.

The 1:1 between ten variations and ten answer slots is convenient, not required.
`-n` changes the count if the downstream aggregator wants a different pool size.

---

## 2. Model selection

`gemma4:12b` (7.4 GB), served by Ollama on the RTX 3090.

Two obstacles, both worth recording:

### Ollama had to be upgraded

`ollama pull gemma4:12b` on the installed **0.11.6** failed:

```
Error: pull model manifest: 412:
The model you are attempting to pull requires a newer version of Ollama.
```

`winget upgrade --id Ollama.Ollama` took it to **0.32.8**, after which the pull
succeeded. The registry probe that established what was actually available:

```powershell
Invoke-RestMethod "https://registry.ollama.ai/v2/library/gemma4/manifests/12b"
```

The `tags/list` endpoint 404s without auth, but per-tag `manifests/<tag>` returns
200 and its layer sizes. Tags that exist: `12b` (7.04 GB), `e2b` (6.67 GB),
`e4b` (8.95 GB); `latest` is `e4b`.

### gemma4 is a thinking model — and that broke the first run outright

Every one of the first 8 test questions failed with
`JSONDecodeError: Expecting value: line 1 column 1 (char 0)`. The cause was not a
malformed reply: `message.content` was **empty**. gemma4 reasons into a separate
`message.thinking` field first, and the reasoning trace exhausted `num_predict`
before a single variation was emitted — `done_reason: "length"`, 4,893 characters
of thinking, zero characters of answer.

Measured on one question at `n=10`:

| | wall time | result |
|---|---|---|
| `think: true` | 19.1 s | truncated at 1,200 tokens, empty content |
| `think: false` | 5.3 s | full set of 10 |

**Paraphrasing gains nothing from the reasoning pass**, so the script sends
`"think": false` by default. `--think` re-enables it, but then `--num-predict`
needs to be several thousand or the same truncation returns.

The empty-content case now raises a named error quoting `done_reason` rather than
surfacing as a confusing JSON parse failure.

---

## 3. Script design

`VQA/generate.query.variations.py`. The parts that carry a decision:

**Keyed on question text, not `Q_ID`.** The 2,443 queries contain only **2,350
distinct questions**, so work is deduped by the normalised question and the
result fanned back out to every `Q_ID` that asks it. The busiest are shared
widely — *"What is this video showing?"* covers 11 `Q_ID`s, *"When is the scene
taking place?"* and *"What is the weather like?"* cover 9 each. Same
`" ".join(text.split()).casefold()` normalisation the other scripts use.

**Schema-constrained decoding.** Ollama's `format` field carries a JSON schema
pinning the reply to an object with exactly N strings, so there is no free-text
parsing and no list-marker stripping in the normal path.

**Top-up rounds.** Candidates are rejected if they duplicate each other, echo the
original, or fall outside 8–300 characters. If that leaves fewer than ten, the
model is re-asked for just the shortfall with the accepted ones listed as
off-limits, up to `--max-rounds` (default 3). A still-short list is **recorded as
short, not padded** — inventing filler would poison the ensemble, and the caller
can fall back to the original question for empty slots.

**Resumable.** Each finished question is appended to the JSONL cache and flushed
immediately; a re-run skips what is present. `--rebuild` re-emits the JSON from
cache with no inference.

**Reproducible.** The sampling seed is derived from a SHA-256 of the question
text, so a re-run of any question yields the same variations. Top-up rounds
offset the seed, since replaying an identical prompt and seed would return an
identical reply.

### Prompt design

Written against the failure modes this specific question set provokes:

- **Question type must survive.** *"How many times does the young man cut
  himself?"* has to stay a counting question, not decay into *"does he cut
  himself"*. Same for why/when/where and yes/no.
- **Identifying detail must survive.** *"the woman in the green coat"* must not
  collapse to *"the woman"* — the coat is what picks her out of the scene.
- **Nothing added, nothing dropped.** No invented specifics.
- **Clean up the source grammar.** Many originals are typo'd or ungrammatical
  (*"what did the woman said to the man at the beginning of the video"*); the
  rewrites are clean English carrying the same intent.
- **Never answer, never comment.**

---

## 4. Run results

```
python VQA/generate.query.variations.py
```

| | |
|---|---|
| Distinct questions processed | **2,350 / 2,350** |
| Wall clock | **96.4 min** (18:30:28 → 20:06:54 UTC, 11 Aug) |
| Throughput | 24.4 questions/min |
| Mean / median / max per question | 9.8 s / 9.8 s / 16.4 s |
| Single-call completions | 2,349 |
| Needed a top-up round | 1 (*"What is the woman listening to?"*) |
| Failures | **0** |

Output integrity, checked across all 2,443 entries:

| Check | Result |
|---|---|
| Entries with exactly 10 variations | **2,443 / 2,443** |
| Short or empty | 0 |
| A variation echoing its original | 0 |
| Duplicate variations within an entry | 0 |

Summed per-question time is 385.6 min against 96.4 min wall clock — an effective
parallelism of **4.0×**. That is Ollama's cap, not the script's: raising
`--workers` from 4 to 8 produced **identical** throughput, because the server's
`OLLAMA_NUM_PARALLEL` gates concurrent requests. Raising that env var and
restarting the server is the lever if this ever needs to be faster; there is
plenty of headroom in 24 GB for a 7.4 GB model.

### Output shape

```json
{
    "Q_ID": 7,
    "Video_ID": "6763.webm",
    "Question": "When is the man snorkeling?",
    "Variations": [
        "At what point in time is the man snorkeling?",
        "State the timing of when the man goes snorkeling.",
        "During which part of the video is the man snorkeling?",
        "..."
    ]
}
```

---

## 5. Quality — spot-checked, not exhaustively verified

Roughly 25 questions were read closely, chosen for being the awkward ones. What
that showed:

**Working well.** Grammar repair is reliable and intent-preserving:
*"what did the woman said to the man at the beginning of the video"* → *"At the
beginning of the video, what did the woman say to the man?"*, with the temporal
anchor intact. Counting questions stay counting questions across all ten
rewrites. Rare vocabulary is handled rather than paraphrased away — *"Why might
the field be verdant?"* keeps "verdant" in some rewrites and glosses it as
"green and lush" in others, which is exactly the useful kind of spread.

**Caveats worth carrying forward:**

- **Occasional detail drift.** *"the young man wearing a big cross"* became
  *"the man in the large cross"* in one rewrite (drops "young") and *"the man
  with the oversized cross"* in another ("oversized" is a shade stronger than
  "big"). Minor at ensemble scale, but it is drift.
- **Rewrites run longer and more formal than the originals** — 64 characters mean
  against 49 for the source questions.
- **Not every rewrite is syntactically a question.** The prompt deliberately
  invites *"Identify …"*, *"State …"*, *"Describe …"* imperatives for diversity.
  Fine as a VLM prompt; something to know if anything downstream assumes a
  trailing question mark.
- **No automated semantic-equivalence check was run** over the 23,500 rewrites.
  Meaning preservation rests on the prompt constraints, the schema, and the spot
  check. If a stronger guarantee is wanted, an NLI or embedding-similarity pass
  against the original would be the way to get it.

---

## Command reference

```bash
# Full run (resumable, ~96 min)
python VQA/generate.query.variations.py

# Smoke test
python VQA/generate.query.variations.py --limit 16

# Specific queries by Q_ID
python VQA/generate.query.variations.py --only 3 17 2400

# Re-emit the JSON from cache, no inference
python VQA/generate.query.variations.py --rebuild

# Regenerate questions already cached
python VQA/generate.query.variations.py --force --only 2400

# Different model or pool size
python VQA/generate.query.variations.py --model gemma4:e4b -n 5
```

Ctrl-C is safe at any point — finished questions are already on disk and a re-run
resumes.

---

## Environment verified

- Ollama **0.32.8** (upgraded from 0.11.6 this session), `gemma4:12b` pulled
- RTX 3090, 24 GB VRAM; 63.7 GB system RAM
- `requests` 2.34.2 — the only dependency; the `ollama` Python package is *not*
  installed and is not needed

---

## Next steps

1. Wire the variations into the VQA answering pass: one VLM call per variation,
   pooled and ranked into the ten `Answers` slots the run format requires.
2. Decide the aggregation rule — consensus across variations is a natural
   ranking signal, and is more defensible than ranking ten raw samples.
3. The **868 undownloaded YouTube videos** from
   `SESSION.2026-08-11.task1-pipeline.md` remain the blocker on 35.5% of queries.
   Variations exist for all 2,443 queries regardless, since they need only the
   question text — but answers for those queries still cannot be generated.
4. Consider whether the ASR transcripts from
   `SESSION.2026-08-11.audio-asr-sensevoice.md` should be fed alongside the
   variations for the speech-heavy questions (*"what did the woman say to…"*),
   which are numerous in this set and are not answerable from frames alone.

# Session Notes — 2026-08-12 — Audio ASR with Whisper large-v3

Built and ran a second, independent ASR pass over the VQA 2026 test videos with
**OpenAI Whisper large-v3**, executed through `faster-whisper` (CTranslate2).
One JSON per video: transcript, detected language with its supporting evidence,
and Whisper's per-segment decoding statistics.

All **1,940** videos in `videos_TEST_VQA/videos` are processed. Wall clock
**28 minutes** (05:57 → 06:25 UTC), 5 failures — the same 5 videos that have no
audio stream at all.

This complements, and does not replace, the SenseVoice pass from
`SESSION.2026-08-11.audio-asr-sensevoice.md`. The headline reason is language
coverage: **SenseVoice can only emit 5 languages**, and 182 videos in this
corpus are none of them. See §5.

**Deadline context:** Answer Generation runs are due **17 August 2026**.

> **Directory rename.** The SenseVoice output moved from
> `videos_TEST_VQA/transcription/` to `videos_TEST_VQA/transcription_funasr/`
> so the two passes sit side by side. `VQA/transcribe.sensevoice.py` still has
> `DEFAULT_OUT = .../transcription`, so re-running it writes to a *new empty*
> directory and re-transcribes all 1,940 videos. Fix the default or pass
> `--out-dir videos_TEST_VQA/transcription_funasr` before running it again.

---

## Files created this session

| File | Purpose |
|---|---|
| `VQA/transcribe.whisper.py` | The whole pipeline — decode, VAD, language sweep, ASR, resumable batch driver |
| `videos_TEST_VQA/transcription_openwhisper/<Video_ID>.json` | Per-video result, 1,940 files, 6.5 MB total (mean 3.4 KB) |
| `videos_TEST_VQA/transcription_openwhisper/transcription_index.json` | One-line-per-video index for fast filtering |
| `models/faster-whisper-large-v3/` | Downloaded weights (2.9 GB) |

---

## 1. Model and environment

Whisper large-v3 was chosen for ASR quality and for its **99-language**
detection head. It is run through
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) rather than the
reference `openai-whisper` package: same weights, ~4× faster, lower VRAM, and
it exposes `no_speech_prob` / `avg_logprob` per segment, which is what the
quality flags in §3 are built on.

Installed into the existing `.venv`:

```
faster-whisper 1.2.1   ctranslate2 4.8.1   torch 2.6.0+cu124   numpy 2.4.6
```

Weights are the Systran CTranslate2 conversion — the reference PyTorch
checkpoint will not load in CTranslate2:

```bash
huggingface-cli download Systran/faster-whisper-large-v3 --local-dir models/faster-whisper-large-v3
```

Hardware: RTX 3090 (24 GB), float16, `--batch-size 8`.
Measured **RTF 0.032–0.056** — 13.35 h of audio in 28 min wall.

---

## 2. Pipeline

```
video ──ffmpeg──► 16 kHz mono float32
                        │
                        ├── Silero VAD ────────► speech spans
                        │                             │
                        │                    sample ≤6 windows
                        │                             │
                        │                     detect_language ──► weighted vote
                        │                                              │
                        └──────────────► Whisper large-v3 ◄────────────┘
                                          (batched, VAD-filtered)
```

Audio is piped from ffmpeg as raw `f32le` on stdout — no intermediate WAV files,
so the run leaves nothing to clean up. Decoding runs on 3 threads ahead of the
GPU, bounded so at most 6 decoded waveforms sit in RAM at once.

### Four design decisions, each forced by an observed failure

**1. Language comes from a window sweep, not from Whisper's own detection.**
Whisper reads language from the **first 30 s only**. On this corpus that is
frequently a title card, a music intro, or an English hook on an otherwise
non-English clip. The script instead samples up to 6 windows drawn from the
*VAD speech* regions, runs `detect_language` on each, and votes.

**2. Votes are weighted by `probability × window duration`.**
Weighting by probability alone was the first implementation and it was wrong:
a 2.4 s window detected as Welsh at p=0.54 outvoted the rest of a 389 s English
documentary and `Archipelago-22` came back multilingual. Duration weighting
dropped the corpus-wide multilingual count from a plainly inflated number to
**5 videos**. A window below `--lang-min-prob 0.5` does not vote at all.

**3. Videos with no speech get `language: null`, not a guess.**
Whisper is forced to return *some* language. On silent and music-only clips it
returned `nn` (Norwegian Nynorsk, p=0.61) with total consistency. When the
sweep gets no vote and no clean segment survives, the label is dropped.
`language_source` records which path produced every label:

| `language_source` | videos | meaning |
|---|---|---|
| `window_sweep` | 998 | voted from ≥1 confident speech window — trust this |
| `no_speech` | 834 | no speech found; `language` is `null` |
| `whisper_default` | 103 | sweep abstained, fell back to Whisper's own label — **weak** |
| `null` | 5 | no audio stream |

**4. Flag Whisper's hallucinations rather than deleting them.**
Handed music or silence, Whisper emits fluent subtitle boilerplate it learned in
training. **562 of 7,260 segments** are flagged `suspect`, excluded from
`transcript`, and kept in `transcript_raw`:

| reason | segments | trigger |
|---|---|---|
| `no_speech_prob` | 529 | ≥ 0.6 |
| `low_logprob` | 41 | avg_logprob ≤ −1.0 |
| `boilerplate` | 26 | "thanks for watching", "subtitles by", bare `[Music]` |
| `repetitive` | 17 | compression_ratio ≥ 2.4 (model looping) |
| `no_alnum` | 1 | punctuation only |

---

## 3. Output schema

```jsonc
{
  "video_id": "0-R8tKOJDGM",
  "duration_sec": 24.381,
  "has_audio": true,
  "error": null,

  "language": "ko",                    // from the weighted window vote
  "language_probability": 0.937,       // mean detector confidence of winning windows
  "language_share": 1.0,               // winner's share of the weighted vote
  "language_source": "window_sweep",   // window_sweep | whisper_default | no_speech
  "language_distribution": {"ko": 1.0},
  "is_multilingual": false,
  "n_language_windows": 2,
  "language_windows": [                // the evidence, per window
    {"start": 5.84, "end": 11.15, "language": "ko",
     "probability": 0.9619, "weight": 5.11,
     "top3": [{"language": "ko", "probability": 0.9619},
              {"language": "jw", "probability": 0.0091}, ...]}
  ],
  "whisper_language": "ko",            // what Whisper alone would have said
  "whisper_language_probability": 1.0,

  "has_speech": true,
  "speech_sec": 20.14,
  "speech_ratio": 0.826,
  "n_speech_regions": 7,
  "duration_after_vad_sec": 20.1,

  "n_segments": 9, "n_clean_segments": 9, "n_suspect_segments": 0,
  "suspect_reason_counts": {},
  "mean_avg_logprob": -0.4456,
  "n_words": 18,
  "transcript": "배고팠어? 맛있겠다 …",       // suspect-filtered
  "transcript_raw": "배고팠어? 맛있겠다 …",   // everything, unfiltered
  "translation_en": null,              // only with --translate

  "segments": [
    {"id": 0, "start": 1.62, "end": 5.09, "duration": 3.47,
     "text": "배고팠어?", "no_speech_prob": 0.0676, "avg_logprob": -0.4456,
     "compression_ratio": 0.609, "temperature": 0.0, "n_tokens": 8,
     "suspect": false, "suspect_reasons": []}
  ],
  "models": {...}, "processed_at": "…", "elapsed_sec": 1.51
}
```

`language_probability` and `language_share` are deliberately separate: a single
window at p=0.55 is a 100% *share* but a coin-flip *detection*, and collapsing
the two into one number hides exactly the cases you need to filter.

---

## 4. Results across all 1,940 videos

| | |
|---|---|
| Total audio | 13.35 h |
| Speech-active (Silero VAD) | 5.62 h (42.1%) |
| Videos with a transcript | 1,046 (53.9%) |
| Transcript words | 45,844 |
| Distinct languages detected | **34** |
| Multilingual videos | 5 |
| Segments | 7,260 (562 flagged suspect) |
| Mean speech ratio, where speech exists | 0.649 |
| **No audio stream** | **5** — `6744`, `6769`, `7023`, `7367`, `7458` |

**Language** (dominant, per video — top 12 of 34)

| none | en | hi | ru | es | ja | zh | ko | fr | pt | ar | ur |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 839 (43.2%) | 752 (38.8%) | 56 | 49 | 36 | 35 | 31 | 27 | 23 | 17 | 10 | 10 |

Tail (1–8 videos each): de, nl, it, pa, tl, te, ta, th, ms, mr, hr, tr, yo, he,
eu, bg, cs, cy, sv, et, nn, ml, bn.

---

## 5. Comparison with the SenseVoice pass

Joined on `video_id` across all 1,940 videos.

**Transcript coverage**

| | count |
|---|---|
| Both produced a transcript | 959 |
| Whisper only | 87 |
| SenseVoice only | 411 |
| Neither | 483 |

Total words are near-identical (Whisper 45,844 vs SenseVoice 44,031) despite
Whisper covering 324 fewer videos — Whisper writes more per video it accepts.

**Language agreement.** Of the 1,002 videos both labelled, they agree on
**76.0%**. The disagreements are not noise; they are structural:

> **SenseVoice supports exactly 5 languages** — zh, en, yue, ja, ko. **182
> videos have a Whisper language outside that set** (hi 51, es 28, ru 20, fr 12,
> pt 10, ar 8, ur 8, de 7, …). SenseVoice had to force each into its nearest
> available label, calling **77 of them `en`** and **65 of them `ko`**.

The single largest disagreement pair is `(SenseVoice ko → Whisper hi)` at 36
videos: Hindi audio labelled Korean, because Hindi is not expressible.
**For any language-conditioned downstream logic, use the Whisper labels.**

**Where SenseVoice found speech and Whisper did not** (411 videos):

| SenseVoice word count | videos |
|---|---|
| 1–2 words | 258 |
| 3–5 words | 66 |
| 6–15 words | 65 |
| 16+ words | 22 |

The 258 one-to-two-word cases are mostly SenseVoice artefacts on music that its
own `filler` heuristic did not catch (`"我我。"`, `"안."`). But the **87 cases with
≥6 words are real content Whisper missed**, and they share a cause: **Silero VAD
rejects sung vocals over music**. 53 of those 87 have `speech_sec == 0` — the
VAD never handed Whisper any audio. Examples: `XM4yr0BltFw` ("If only I could
make a deal with God" — Kate Bush), `8278` ("90s retro on the radio").

This is the main systematic gap in this pass. If lyrics matter for a question,
prefer the SenseVoice transcript for those videos.

---

## 6. Known limitations — read before relying on this

- **Singing is largely invisible.** Silero VAD treats vocals-over-music as
  non-speech. 894 videos (46%) have `has_speech: false`, materially more than
  SenseVoice's 570. Lower `--min-silence-ms`, or fall back to the SenseVoice
  transcript, when lyrics matter.
- **103 videos carry a `whisper_default` language.** The sweep abstained, so
  the label was read off the first 30 s including music. `0k0EEXZjmQU` is
  labelled `ko` at p=0.70 while its transcript is English singing. **Filter on
  `language_source == "window_sweep"` when language accuracy matters.**
- **Language detection is unreliable on short clips.** The detector needs
  several seconds of clean speech; below ~3 s it is close to a coin flip, which
  is why `--lang-min-prob` exists. `07Gghjjia0I` is the residual case in the
  corpus — a 4.2 s Hindi window detected as Russian at p=0.92, which is enough
  to flip it to multilingual. Its `language` (`hi`) is still correct.
- **`is_multilingual` is now conservative, not comprehensive.** Only 5 videos.
  Duration weighting fixed a large false-positive rate but the corpus certainly
  contains more genuine code-switching than five clips. Treat `true` as
  reliable and `false` as weak evidence.
- **No per-segment language.** faster-whisper's `Segment` has no `language`
  field even when decoding with `multilingual=True`, so there is nothing to
  corroborate the sweep against at segment granularity.
- **No speaker diarisation.** Multi-speaker clips are one flat transcript.
- **No word-level timestamps.** Deliberate — segments only, per the run config.
  Add `word_timestamps=True` if answer localisation ever needs them.

---

## 7. Command reference

`PYTHONIOENCODING=utf-8` is needed on Windows — non-Latin transcripts crash the
cp1252 console encoder on print.

```powershell
# Full run — resumable, skips videos that already have output
$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe VQA\transcribe.whisper.py

# Re-do everything under one uniform config
.\.venv\Scripts\python.exe VQA\transcribe.whisper.py --force

# Specific videos. NOTE: 12 Video_IDs start with '-', which argparse reads as
# an option — those need the --only=ID form, repeated.
.\.venv\Scripts\python.exe VQA\transcribe.whisper.py --only=-5tDzjnjTV0 --only 6501 ChainedforLife-26 --force

# Rebuild the index from JSON on disk (no inference, no GPU)
.\.venv\Scripts\python.exe VQA\transcribe.whisper.py --reindex

# Add an English translation for non-English videos (~2x cost on those)
.\.venv\Scripts\python.exe VQA\transcribe.whisper.py --translate --force

# Less VRAM / no GPU
.\.venv\Scripts\python.exe VQA\transcribe.whisper.py --compute-type int8_float16
.\.venv\Scripts\python.exe VQA\transcribe.whisper.py --device cpu --batch-size 1 --compute-type int8
```

Tuning flags: `--batch-size` (8), `--beam-size` (5), `--lang-windows` (6),
`--lang-min-prob` (0.5), `--lang-min-window-sec` (3.0),
`--multilingual-threshold` (0.25), `--min-silence-ms` (500),
`--no-speech-prob` (0.6), `--min-logprob` (−1.0), `--decode-workers` (3),
`--condition-on-previous` (off — it lets one hallucination propagate down a file).

Ctrl-C is safe — each video's JSON is written as it completes, and a re-run
picks up where it stopped.

---

## 8. Next steps

1. **Prefer Whisper for language, and merge the transcripts.** The join is a
   one-liner and the stems match across both directories:
   ```python
   vid = Path(video_id).stem   # Video_ID for webm entries carries the .webm suffix
   w = json.load(open(f"videos_TEST_VQA/transcription_openwhisper/{vid}.json"))
   s = json.load(open(f"videos_TEST_VQA/transcription_funasr/{vid}.json"))
   ```
   A defensible merge: take `w["transcript"]` when non-empty, fall back to
   `s["transcript"]` for the 411 SenseVoice-only videos (which recovers the
   sung-lyrics cases), and always take language from `w` when
   `w["language_source"] == "window_sweep"`.
2. **Fix `DEFAULT_OUT` in `transcribe.sensevoice.py`** to point at
   `transcription_funasr`, before an accidental re-run re-transcribes 1,940
   videos into a resurrected `transcription/`.
3. **Decide whether to run `--translate`.** 320 videos are non-English with a
   transcript. If the answer generator prompts in English, an English
   translation may beat feeding it Hindi or Korean text; the pass costs roughly
   another 20 minutes for those videos alone.
4. **483 videos have no transcript from either model.** For those the audio
   channel contributes nothing and answers must come from vision alone — worth
   checking the overlap with the Task 1 Q_IDs.
5. SenseVoice still owns the side-channels Whisper does not produce: ambient
   events (BGM, laughter, applause) and emotion. The two passes are
   complementary, not redundant.

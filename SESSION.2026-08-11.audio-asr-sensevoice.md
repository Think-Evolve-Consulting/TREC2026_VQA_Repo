# Session Notes — 2026-08-11 — Audio ASR with SenseVoiceSmall

Built and ran an audio understanding pass over the VQA 2026 test videos:
language identification, speech transcription, speaker emotion, and ambient
audio events (music, laughter, applause, …) — one JSON per video.

All **1,940** videos in `videos_TEST_VQA/videos` are processed.

**Deadline context:** Answer Generation runs are due **17 August 2026**. This
gives the answer generator an audio channel it did not previously have; see
`SESSION.2026-08-11.task1-pipeline.md` for the run-file and download work.

---

## Files created this session

| File | Purpose |
|---|---|
| `VQA/transcribe.sensevoice.py` | The whole pipeline — decode, VAD, ASR, tag parsing, resumable batch driver |
| `videos_TEST_VQA/transcription/<Video_ID>.json` | Per-video result, 1,940 files, 6.5 MB total |
| `videos_TEST_VQA/transcription/transcription_index.json` | One-line-per-video index for fast filtering |
| `models/SenseVoiceSmall/`, `models/fsmn-vad/` | Downloaded weights (~1 GB) |

---

## 1. Model and environment

[SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) is a
non-autoregressive encoder that emits four labels in a single forward pass:
language, emotion, audio event, and the transcript itself. That is exactly the
set we wanted, from one model instead of four.

> Note: the repo is **FunAudioLLM/SenseVoice**, not `QwenAudio/SenseVoice`.
> The model is used through `funasr`, which registers `SenseVoiceSmall`
> natively — the SenseVoice repo itself is not needed.

Installed into the existing `.venv` (it held only `yt-dlp`):

```
torch 2.6.0+cu124   torchaudio 2.6.0+cu124   funasr 1.4.1
transformers 5.15.0   numpy 2.4.6   soundfile   huggingface_hub
```

Weights come from **HuggingFace**, not ModelScope (funasr's default hub), so
that no ModelScope round-trip is needed:

```bash
huggingface-cli download FunAudioLLM/SenseVoiceSmall --local-dir models/SenseVoiceSmall
huggingface-cli download funasr/fsmn-vad          --local-dir models/fsmn-vad
```

Hardware: RTX 3090 (24 GB), ffmpeg at `C:\ProgramData\chocolatey\bin`.
Measured **RTF 0.016** — 13.4 h of audio in ~14 min of GPU time.

---

## 2. Pipeline

```
video ──ffmpeg──► 16 kHz mono float32
                        │
                        ├── FSMN-VAD ──────────► speech spans
                        │                             │
                        └── gaps between spans ───────┤
                                                      ▼
                                        SenseVoiceSmall, batched
                                                      │
                     <|en|><|HAPPY|><|Laughter|><|withitn|>text
```

Audio is piped from ffmpeg as raw `f32le` on stdout — no intermediate WAV files
are written, so the run leaves nothing to clean up.

### Four design decisions, each forced by an observed failure

**1. Segment the audio; never transcribe a whole file.**
SenseVoice is trained on utterances of ≤30 s. Feeding a 173 s clip whole
collapsed it to *one* language/emotion/event label for the entire video, and the
text degraded visibly — run-together fragments with dropped words:

```
whole file : "come aren't you you all of you I knew I could never said you'd
              help me Doctor stop him He' against of we're all ashamed here…"
segmented  : "All of you." / "You can lead normal lives." / "I can set you
              free." / "I'm your only hope."
```

**2. Also classify the regions VAD rejects.**
VAD returns *speech* spans, so a clip of pure music or applause yields zero
segments and would produce no output at all — exactly what happened to
`7398.webm` (10 s of audio, nothing emitted). The gaps between speech spans are
therefore swept in ≤20 s windows and classified too. These carry
`"type": "ambient"`; VAD-derived ones carry `"type": "speech"`. **4,053 of the
10,415 segments are ambient**, and they are where most music detection lives.

**3. Two ASR passes for emotion.**
**80% of segments return `EMO_UNKNOWN`.** A second pass with `ban_emo_unk=True`
forces a real label into `emotion_forced`. Both are kept: the first is honest,
the second is usable-but-noisy. It costs ~2× GPU time, which is negligible here.

**4. Flag the non-speech hallucinations.**
Handed music, SenseVoice still emits its most likely token. Across a 150-video
sample the artefacts were bare punctuation (104/421 segments had no alphanumeric
character at all) and a lone `"The."` (95 occurrences). Both are flagged
`"filler": true` and excluded from `transcript` and from language voting —
without this, `-IRLy4hyTME` transcribed as `"The. The."` and a music-only clip
could claim a language. **629 segments were filtered.** Genuine one-word
utterances (`"Yeah"`, `"Oh"`) are deliberately left alone. Flagged segments are
retained in `segments`, never deleted, so the heuristic can be second-guessed.

---

## 3. Output schema

```jsonc
{
  "video_id": "3NlwYe8rIqE",
  "duration_sec": 15.2,
  "has_audio": true,
  "error": null,

  "language": "en",                    // dominant, weighted by spoken duration
  "language_seconds": {"en": 8.4, "ja": 1.2},
  "is_multilingual": true,

  "has_speech": true,
  "speech_sec": 9.6,
  "speech_ratio": 0.63,

  "ambient_events": ["Laughter"],      // non-Speech events present
  "event_counts": {"Speech": 4, "Laughter": 1},

  "dominant_emotion": "HAPPY",         // null when the model was unsure
  "dominant_emotion_forced": "ANGRY",  // always populated, low confidence
  "emotion_counts": {...}, "emotion_forced_counts": {...},

  "transcript": "No no.",              // filler-filtered, speech segments only
  "segments": [
    {
      "id": 0, "type": "speech", "start": 2.09, "end": 2.77, "duration": 0.68,
      "language": "ja", "emotion": "UNKNOWN", "emotion_forced": "NEUTRAL",
      "event": "BGM", "text": "…", "filler": false,
      "raw": "<|ja|><|EMO_UNKNOWN|><|BGM|><|withitn|>…"
    }
  ],
  "models": {...}, "processed_at": "…", "elapsed_sec": 0.23
}
```

`raw` preserves the untouched model string, so nothing in the parsing is lossy.

---

## 4. Results across all 1,940 videos

| | |
|---|---|
| Total audio | 13.35 h (mean 24.8 s, max 389 s) |
| Speech-active | 7.03 h (52.6%) |
| Videos with a transcript | 1,370 (70.6%) |
| Transcript words | 44,031 |
| Multilingual videos | 313 |
| Segments | 10,415 (6,362 speech / 4,053 ambient) |
| **No audio stream** | **5** — `6744`, `6769`, `7023`, `7367`, `7458` |

**Language** (dominant, per video)

| en | none | zh | ja | ko | yue |
|---|---|---|---|---|---|
| 897 (46.2%) | 571 (29.4%) | 223 (11.5%) | 134 (6.9%) | 108 (5.6%) | 7 (0.4%) |

`none` = no speech survived filtering (music-only or silent clips). The corpus
is substantially non-English — **472 videos (24.3%) are zh/ja/ko/yue**, which
matters for any English-only downstream prompt.

**Ambient events** (videos containing at least one)

| BGM | Breath | Laughter | Applause | Cough | Cry | Sneeze |
|---|---|---|---|---|---|---|
| 1,041 (53.7%) | 93 (4.8%) | 22 (1.1%) | 17 (0.9%) | 3 | 3 | 3 |

**Emotion** — honest label is `null` for **80.4%** of videos. Where present:
HAPPY 221, NEUTRAL 135, SAD 13, ANGRY 7, SURPRISED 4, DISGUSTED 1.
The forced label covers everything with speech but skews hard: HAPPY 653,
NEUTRAL 529, ANGRY 78, SURPRISED 53, SAD 45, DISGUSTED 12.

---

## 5. Known limitations — read before relying on this

- **Emotion is weak.** 80% `EMO_UNKNOWN`; the forced label is HAPPY a third of
  the time regardless of content. Spot-checked failures are common — a sombre
  clip forced to `SAD` for every segment, `3NlwYe8rIqE` (laughter, "No no.")
  forced to `ANGRY`. **Treat `dominant_emotion_forced` as a weak prior, not a
  fact.** Prefer `dominant_emotion` and accept its nulls.
- **One event tag per segment.** The model emits a single event, so loud music
  masks quieter events. Applause appears in only 0.9% of videos and Laughter in
  1.1% — implausibly low for this corpus. **Absence of an ambient event is not
  evidence of absence**; presence is the reliable direction.
- **Short/noisy segments cross-contaminate language.** e.g.
  `"What it's like shopping with my husband? い。"` — a stray Japanese token on
  an English clip. `language_seconds` is duration-weighted to limit the damage,
  but per-segment language on <1 s segments is unreliable.
- **No speaker diarisation.** Multi-speaker clips produce one flat transcript
  with no turn attribution.
- **ASR quality on music-heavy social video is mediocre**, as expected — proper
  nouns and overlapping speech suffer.

---

## 6. Two bugs found and fixed

**Index truncated on resume.** The index was accumulated in memory from the
current run only, so a resumed run overwrote a complete 1,940-video index with
just the handful it had reprocessed — the file on disk read `n_processed: 2`.
`load_index()` now rebuilds it from the per-video JSON on disk, which is
authoritative and self-healing. Added `--reindex` to repair it without inference.

**`_index.json` collided with a naming convention.** Nine real Video_IDs begin
with an underscore (`_3GqKCAd-P0`, `_9KWhuxD-x0`, …), so the natural instinct to
skip files starting with `_` silently drops nine videos — which is exactly what
my first analysis pass did. The index is now `transcription_index.json`.
**Filter it by exact filename, never by prefix.**

---

## 7. Command reference

```powershell
# Full run — resumable, skips videos that already have output
.\.venv\Scripts\python.exe VQA\transcribe.sensevoice.py

# Re-do everything
.\.venv\Scripts\python.exe VQA\transcribe.sensevoice.py --force

# Specific videos
.\.venv\Scripts\python.exe VQA\transcribe.sensevoice.py --only 6501.webm ChainedforLife-26 --force

# Rebuild the index from JSON on disk (no inference, no GPU)
.\.venv\Scripts\python.exe VQA\transcribe.sensevoice.py --reindex

# Faster: skip the forced-emotion second pass
.\.venv\Scripts\python.exe VQA\transcribe.sensevoice.py --no-forced-emotion

# No GPU available
.\.venv\Scripts\python.exe VQA\transcribe.sensevoice.py --device cpu --batch-size 4
```

Tuning flags: `--batch-size` (16), `--max-segment-sec` (30),
`--ambient-window-sec` (20), `--min-ambient-sec` (1), `--decode-workers` (3).

Ctrl-C is safe — each video's JSON is written as it completes, and a re-run
picks up where it stopped.

---

## 8. Next steps

1. **Join the transcripts into the Task 1 answer generator.** The file naming
   matches the convention from the download session exactly, so lookup stays a
   one-liner:
   ```python
   json.load(open(f"videos_TEST_VQA/transcription/{video_id}.json"))
   ```
   Note `Video_ID` for webm entries includes the `.webm` suffix, so the
   transcription stem is `Path(video_id).stem`.
2. **Decide how audio feeds the answer prompt.** `transcript` alone is the cheap
   option; `ambient_events` + `language` are useful for questions about setting
   and mood. Given the caveats in §5, emotion is probably not worth prompting on.
3. **570 videos have no transcript.** For those, audio contributes nothing and
   the answer must come from vision alone — worth checking whether the blocked
   Q_IDs overlap.
4. Consider a stronger ASR (Whisper large-v3) for the ~1,370 videos with speech
   if transcript quality proves limiting. SenseVoice was chosen for the four-in-
   one labelling and speed, not for best-in-class WER.

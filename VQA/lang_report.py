#!/usr/bin/env python3
"""Report video counts by language over the SenseVoice transcription output.

Reads ``videos_TEST_VQA/transcription_funasr/*.json`` (the per-video records
written by ``transcribe.sensevoice.py``) and writes a markdown review file
listing every video whose dominant language is not English.

The Whisper counterpart is ``lang_report_openwhisper.py``. Read that file's
docstring before comparing the two reports: SenseVoice can only emit five
languages (en/zh/ja/ko/yue), so anything else in the corpus is necessarily
mis-binned into one of them, and its ``language_seconds`` is a duration-weighted
vote in seconds while Whisper's is a confidence-weighted share.

Usage:
    python VQA/lang_report.py
    python VQA/lang_report.py --dir some/other/dir --out report.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Renamed transcription/ -> transcription_funasr/ once a second ASR pass
# existed; both are accepted so the script runs on either layout.
DEFAULT_DIRS = (ROOT / "videos_TEST_VQA" / "transcription_funasr",
                ROOT / "videos_TEST_VQA" / "transcription")
DEFAULT_OUT = ROOT / "TRANSCRIPTION.language-review.funasr.md"
INDEX_NAME = "transcription_index.json"

LANG_NAMES = {
    "en": "English",
    "zh": "Chinese (Mandarin)",
    "yue": "Chinese (Cantonese)",
    "ja": "Japanese",
    "ko": "Korean",
    "nospeech": "No speech detected",
}


def load_records(d: Path) -> list[tuple[str, dict]]:
    """Every per-video record in ``d``.

    The index is excluded by exact filename -- never by an underscore prefix,
    which would silently drop the nine real Video_IDs that start with one.
    """
    out = []
    for p in sorted(d.glob("*.json")):
        if p.name == INDEX_NAME:
            continue
        out.append((p.stem, json.loads(p.read_text(encoding="utf-8"))))
    return out


def secs(d: dict) -> dict:
    return d.get("language_seconds") or {}


def fmt_langsecs(d: dict) -> str:
    ls = secs(d)
    if not ls:
        return "—"
    return ", ".join(f"{k} {v:.1f}s"
                     for k, v in sorted(ls.items(), key=lambda kv: -kv[1]))


def clip(t: str | None, n: int = 90) -> str:
    t = " ".join((t or "").split())
    if not t:
        return "—"
    t = t.replace("|", "\\|")
    return t if len(t) <= n else t[: n - 1] + "…"


def build(records: list[tuple[str, dict]]) -> str:
    total = len(records)
    lang_counts: Counter = Counter()
    by_lang: dict[str, list] = defaultdict(list)
    no_audio, errors, multiling = [], [], []

    for vid, d in records:
        if d.get("error"):
            errors.append((vid, d["error"]))
        if not d.get("has_audio", True):
            no_audio.append(vid)
        lang = d.get("language") or "nospeech"
        lang_counts[lang] += 1
        by_lang[lang].append((vid, d))
        if d.get("is_multilingual"):
            multiling.append((vid, d))

    non_en = [l for l in lang_counts if l not in ("en", "nospeech")]
    non_en.sort(key=lambda l: -lang_counts[l])
    non_en_total = sum(lang_counts[l] for l in non_en)

    L: list[str] = []
    w = L.append

    w("# Transcription language review — SenseVoiceSmall (FunASR)")
    w("")
    w(f"Source: `videos_TEST_VQA/transcription_funasr/*.json` ({total} per-video "
      f"JSON files, one per video, excluding `{INDEX_NAME}`).")
    w("Language is the **dominant** language per video, weighted by spoken "
      "duration and computed only over non-filler speech segments.")
    w("")
    w("> SenseVoice recognises **only five languages** (en/zh/ja/ko/yue). Any "
      "other language in the corpus is necessarily forced into one of them — "
      "spot-checks show Hindi arriving as Korean. See "
      "`TRANSCRIPTION.language-review.openwhisper.md` for the Whisper pass, "
      "which covers ~99 languages, and treat that as the better language "
      "authority.")
    w("")
    w("## 1. Counts")
    w("")
    w("| | Videos | Share |")
    w("|---|---:|---:|")
    w(f"| **Total videos transcribed** | {total} | 100% |")
    w(f"| English (`en`) | {lang_counts['en']} | {lang_counts['en']/total:.1%} |")
    w(f"| **Non-English (identified)** | **{non_en_total}** | **{non_en_total/total:.1%}** |")
    w(f"| No language — no speech detected | {lang_counts['nospeech']} "
      f"| {lang_counts['nospeech']/total:.1%} |")
    w("")
    w("### Breakdown by language")
    w("")
    w("| Language | Code | Videos | Share |")
    w("|---|---|---:|---:|")
    for l, n in lang_counts.most_common():
        code = "—" if l == "nospeech" else l
        w(f"| {LANG_NAMES.get(l, l)} | `{code}` | {n} | {n/total:.1%} |")
    w("")
    w(f"**No-language group is tracked separately**: {lang_counts['nospeech']} "
      "videos produced no surviving speech (music-only, ambient-only, silent, or "
      f"no audio stream at all). Of these, {len(no_audio)} have no audio stream "
      "whatsoever" + (f" (`{'`, `'.join(no_audio)}`)" if no_audio else "") + ".")
    w("")
    w(f"{len(multiling)} videos are flagged `is_multilingual` (speech in more "
      "than one language); they are counted once above, under their dominant "
      "language.")
    err_ids = {v for v, _ in errors}
    if errors:
        w("")
        if err_ids == set(no_audio):
            w(f"The only {len(errors)} recorded processing errors are those same "
              "no-audio-stream videos (ffmpeg found no audio to decode). Every "
              "other video processed cleanly.")
        else:
            w(f"{len(errors)} videos recorded a processing error: "
              f"`{'`, `'.join(sorted(err_ids))}`.")
    w("")
    w("---")
    w("")
    w("## 2. Non-English videos for review")
    w("")
    w(f"All {non_en_total} videos whose dominant language is not English, grouped "
      "by language and ordered by spoken duration in that language (most speech "
      "first — the top of each list is where a mis-detection matters most).")
    w("")
    w("`Multi` marks a video with speech in more than one language. "
      "`Lang seconds` gives the duration-weighted vote behind the call.")
    w("")

    for l in non_en:
        rows = by_lang[l]
        rows.sort(key=lambda r: -secs(r[1]).get(l, 0.0))
        w(f"### {LANG_NAMES.get(l, l)} — `{l}` ({len(rows)} videos)")
        w("")
        w("| # | Video_ID | Dur (s) | Speech (s) | Multi | Lang seconds | "
          "Transcript (start) |")
        w("|---:|---|---:|---:|:---:|---|---|")
        for i, (vid, d) in enumerate(rows, 1):
            w("| {} | `{}` | {:.1f} | {:.1f} | {} | {} | {} |".format(
                i, vid, d.get("duration_sec") or 0.0, d.get("speech_sec") or 0.0,
                "yes" if d.get("is_multilingual") else "",
                fmt_langsecs(d), clip(d.get("transcript"))))
        w("")

    w("---")
    w("")
    w("## 3. Caveats on these labels")
    w("")
    w("- **Only five languages exist in this model's output.** A Hindi, Russian "
      "or Spanish video cannot be labelled correctly here; it is binned into "
      "the nearest of en/zh/ja/ko/yue. Cross-check against the Whisper report "
      "before treating any count in this file as a language census.")
    w("- Language comes from SenseVoice's per-segment tag, aggregated by "
      "duration. **Segments under ~1 s are unreliable** and cross-contaminate — "
      "a stray `ja` token on an English clip is a known failure mode. Duration "
      "weighting limits but does not remove this.")
    w("- `zh` vs `yue` (Mandarin vs Cantonese) is the least reliable distinction "
      "in this set; treat the 'yue' group as provisional.")
    w("- A `no speech` label means no speech **survived filler filtering**, not "
      "that the video is silent — most of this group is music-only.")
    w("- Transcripts shown are the filler-filtered text, truncated for display. "
      "Read the per-video JSON for the full text and per-segment detail.")
    w("")

    return "\n".join(L)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=None,
                    help="SenseVoice output dir (default: transcription_funasr/)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    d = args.dir or next((p for p in DEFAULT_DIRS if p.is_dir()), None)
    if d is None or not d.is_dir():
        print("error: no SenseVoice output dir found; looked for "
              + ", ".join(str(p) for p in DEFAULT_DIRS))
        return 2

    records = load_records(d)
    if not records:
        print(f"error: no per-video JSON found in {d}")
        return 2

    args.out.write_text(build(records), encoding="utf-8")

    total = len(records)
    counts = Counter(rec.get("language") or "nospeech" for _, rec in records)
    for l, n in counts.most_common():
        print(f"  {l:9s} {n:5d}  {n/total:6.1%}")
    print(f"total {total}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

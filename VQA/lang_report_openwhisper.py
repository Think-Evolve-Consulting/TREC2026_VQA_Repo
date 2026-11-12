#!/usr/bin/env python3
"""Report video counts by language over the Whisper large-v3 transcription output.

Reads ``videos_TEST_VQA/transcription_openwhisper/*.json`` (the per-video records
written by ``transcribe.whisper.py``) and writes a markdown review file listing
every video whose detected language is not English.

This is the Whisper counterpart to ``lang_report.py``, which does the same over
the SenseVoice output. The two are not interchangeable, because the records
differ in ways that change what can be reported:

  * SenseVoice emits a per-segment language tag, so its ``language_seconds`` is a
    duration-weighted vote in *seconds*. Whisper has no per-segment language, so
    ``language_distribution`` is a *share* of a confidence-weighted window vote
    that sums to 1. Seconds and shares are not comparable; the column is labelled
    accordingly.
  * SenseVoice can only emit 5 languages (en/zh/ja/ko/yue) and is forced to bin
    everything else into one of them. Whisper covers ~99, so this report groups
    many more languages and has a long tail of one-video languages -- which is
    also where its misdetections live, hence the ``rare`` review flag.
  * ``language_source`` records how the label was reached. ``window_sweep`` is the
    strong evidence path; ``whisper_default`` means the sweep abstained and the
    label was read off Whisper's own first-30 s detection, which is weaker.

Usage:
    python VQA/lang_report_openwhisper.py
    python VQA/lang_report_openwhisper.py --out report.md --low-conf 0.8
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "videos_TEST_VQA" / "transcription_openwhisper"
DEFAULT_OUT = ROOT / "TRANSCRIPTION.language-review.openwhisper.md"
# The SenseVoice output was renamed transcription/ -> transcription_funasr/ once
# a second ASR pass existed; both names are accepted so the cross-check keeps
# working on either layout.
SENSEVOICE_DIRS = (ROOT / "videos_TEST_VQA" / "transcription_funasr",
                   ROOT / "videos_TEST_VQA" / "transcription")
INDEX_NAME = "transcription_index.json"

# ISO 639-1 names for the codes Whisper emits. Anything unmapped falls back to
# the bare code, so an unexpected language is visible rather than silently
# relabelled.
LANG_NAMES = {
    "en": "English", "zh": "Chinese", "yue": "Cantonese", "ja": "Japanese",
    "ko": "Korean", "hi": "Hindi", "ru": "Russian", "es": "Spanish",
    "fr": "French", "pt": "Portuguese", "ar": "Arabic", "ur": "Urdu",
    "de": "German", "nl": "Dutch", "it": "Italian", "pa": "Punjabi",
    "tl": "Tagalog", "ta": "Tamil", "te": "Telugu", "th": "Thai",
    "ms": "Malay", "mr": "Marathi", "hr": "Croatian", "tr": "Turkish",
    "yo": "Yoruba", "he": "Hebrew", "eu": "Basque", "bg": "Bulgarian",
    "cs": "Czech", "cy": "Welsh", "sv": "Swedish", "et": "Estonian",
    "nn": "Norwegian Nynorsk", "ml": "Malayalam", "bn": "Bengali",
    "id": "Indonesian", "vi": "Vietnamese", "pl": "Polish", "fa": "Persian",
    "uk": "Ukrainian", "el": "Greek", "ro": "Romanian", "da": "Danish",
    "fi": "Finnish", "no": "Norwegian", "hu": "Hungarian", "sk": "Slovak",
    "sl": "Slovenian", "lt": "Lithuanian", "lv": "Latvian", "sr": "Serbian",
    "kn": "Kannada", "gu": "Gujarati", "ne": "Nepali", "si": "Sinhala",
    "km": "Khmer", "my": "Burmese", "la": "Latin", "mi": "Maori",
    "haw": "Hawaiian", "af": "Afrikaans", "sw": "Swahili", "az": "Azerbaijani",
}

NO_LANG = "--none--"  # sentinel; cannot collide with a real ISO code


def name_of(code: str) -> str:
    return LANG_NAMES.get(code, code)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_records(d: Path) -> list[dict]:
    """Every per-video record in ``d``, keyed by exact filename.

    Nine real Video_IDs start with an underscore (``_3GqKCAd-P0``), so the
    index is excluded by exact name -- never by an underscore prefix, which
    would silently drop those nine.
    """
    out = []
    for p in sorted(d.glob("*.json")):
        if p.name == INDEX_NAME:
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec.setdefault("video_id", p.stem)
        out.append(rec)
    return out


def load_sensevoice(d: Path | None) -> dict[str, str]:
    """SenseVoice dominant language per video, for the cross-model section."""
    if d is None or not d.is_dir():
        return {}
    out = {}
    for p in sorted(d.glob("*.json")):
        if p.name == INDEX_NAME:
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out[p.stem] = rec.get("language") or NO_LANG
    return out


# --------------------------------------------------------------------------- #
# review flags
# --------------------------------------------------------------------------- #
def voting_windows(rec: dict) -> int:
    """Detection windows that actually carried weight into the vote."""
    return sum(1 for w in (rec.get("language_windows") or []) if w.get("weight", 0) > 0)


def review_flags(rec: dict, rare: set[str], args) -> list[str]:
    """Why this label deserves a human look. Empty means nothing stands out."""
    flags = []
    prob = rec.get("language_probability")
    if prob is not None and prob < args.low_conf:
        flags.append("low-conf")
    if rec.get("language_source") == "whisper_default":
        flags.append("no-sweep")
    if (rec.get("n_words") or 0) <= args.thin_words:
        flags.append("thin-text")
    if voting_windows(rec) <= 1:
        flags.append("1-window")
    if rec.get("language") in rare:
        flags.append("rare-lang")
    if rec.get("is_multilingual"):
        flags.append("multi")
    return flags


def fmt_dist(rec: dict, top: int = 3) -> str:
    # On the whisper_default path no window voted at all; the record still
    # stores {lang: 1.0}, which would read as a unanimous vote. Render it as
    # absent so a synthetic 100% is never mistaken for evidence.
    if rec.get("language_source") == "whisper_default":
        return "*no vote*"
    dist = rec.get("language_distribution") or {}
    if not dist:
        return "—"
    items = sorted(dist.items(), key=lambda kv: -kv[1])[:top]
    return ", ".join(f"{k} {v:.0%}" for k, v in items)


def clip(text: str | None, n: int = 80) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "—"
    text = text.replace("|", "\\|")
    return text if len(text) <= n else text[: n - 1] + "…"


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def build(recs: list[dict], sv: dict[str, str], args) -> str:
    total = len(recs)
    counts = Counter(r.get("language") or NO_LANG for r in recs)
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_lang[r.get("language") or NO_LANG].append(r)

    n_en = counts.get("en", 0)
    n_none = counts.get(NO_LANG, 0)
    non_en = [c for c in counts if c not in ("en", NO_LANG)]
    non_en.sort(key=lambda c: (-counts[c], c))
    n_non_en = sum(counts[c] for c in non_en)

    # A language carried by one or two videos in a 1,940-video corpus is more
    # often a misdetection on short or noisy audio than a real find.
    rare = {c for c in non_en if counts[c] <= args.rare_max}

    no_audio = [r["video_id"] for r in recs if not r.get("has_audio", True)]
    errors = [r["video_id"] for r in recs if r.get("error")]
    multi = [r for r in recs if r.get("is_multilingual")]
    sources = Counter(r.get("language_source") or "(none)" for r in recs)

    L: list[str] = []
    w = L.append

    w("# Transcription language review — Whisper large-v3")
    w("")
    w(f"Source: `videos_TEST_VQA/transcription_openwhisper/*.json` — {total} "
      "per-video records written by `VQA/transcribe.whisper.py`, excluding "
      f"`{INDEX_NAME}`.")
    w("")
    w("Language is decided by a **confidence-weighted sweep** of up to 6 speech "
      "windows spread across each video, not by Whisper's default first-30 s "
      "detection. Windows below the confidence floor cast no vote; when no "
      "window votes, the video has **no language** rather than a guessed one.")
    w("")

    # -- 1. counts ---------------------------------------------------------- #
    w("## 1. Counts")
    w("")
    w("| | Videos | Share |")
    w("|---|---:|---:|")
    w(f"| **Total videos transcribed** | {total} | 100% |")
    w(f"| English (`en`) | {n_en} | {n_en/total:.1%} |")
    w(f"| **Non-English (identified)** | **{n_non_en}** | **{n_non_en/total:.1%}** |")
    w(f"| No language — no speech detected | {n_none} | {n_none/total:.1%} |")
    w("")
    w(f"Non-English speech spans **{len(non_en)} distinct languages**.")
    w("")

    w("### Breakdown by language")
    w("")
    w("| Language | Code | Videos | Share |")
    w("|---|---|---:|---:|")
    w(f"| English | `en` | {n_en} | {n_en/total:.1%} |")
    w(f"| *No speech detected* | — | {n_none} | {n_none/total:.1%} |")
    for c in non_en:
        n = counts[c]
        w(f"| {name_of(c)} | `{c}` | {n} | {n/total:.1%} |")
    w("")

    # -- 2. the no-language group ------------------------------------------- #
    w("### The no-language group, tracked separately")
    w("")
    w(f"{n_none} videos ({n_none/total:.1%}) carry **no language label at all**. "
      "This is a deliberate outcome, not a gap: no detection window cleared the "
      "confidence floor and no clean transcript segment survived, so the record "
      "says *no language* instead of inventing one off music or silence.")
    w("")
    w(f"Of these, **{len(no_audio)} have no audio stream whatsoever**"
      + (f" (`{'`, `'.join(no_audio)}`)" if no_audio else "") + ".")
    if errors:
        if set(errors) == set(no_audio):
            w(f"Those same {len(errors)} videos are the only recorded processing "
              "errors — ffmpeg found no audio to decode. Every other video "
              "processed cleanly.")
        else:
            w(f"{len(errors)} videos recorded a processing error: "
              f"`{'`, `'.join(sorted(errors))}`.")
    w("")
    w("How each label was reached (`language_source`):")
    w("")
    w("| Source | Videos | Meaning |")
    w("|---|---:|---|")
    meaning = {
        "window_sweep": "a confident speech window voted — strongest evidence",
        "whisper_default": "sweep abstained; label read off Whisper's own first-30 s "
                           "detection — **weaker, review these**",
        "no_speech": "no speech survived; no language assigned",
        "(none)": "no audio stream — nothing to detect",
    }
    for s, n in sources.most_common():
        w(f"| `{s}` | {n} | {meaning.get(s, '')} |")
    w("")
    w(f"{len(multi)} videos are flagged `is_multilingual` (a second language took "
      f"at least {args.multi_note} of the vote). They are counted once above, "
      "under their dominant language.")
    w("")
    w("---")
    w("")

    # -- 3. non-English listing --------------------------------------------- #
    w("## 2. Non-English videos for review")
    w("")
    w(f"All {n_non_en} videos whose detected language is not English, grouped by "
      "language and ordered **most-suspect first** (review flags, then ascending "
      "confidence), so the entries most likely to be wrong sit at the top of "
      "each table.")
    w("")
    w("**Columns.** `Conf` is the mean detector confidence over the windows that "
      "voted for the winner. `Vote` is each language's share of the weighted "
      "window vote — a *share*, not seconds; it says nothing about how long "
      "each language was spoken. `Wnd` is the number of windows that voted, so "
      "`Wnd 0` with `Vote *no vote*` marks a label that rests entirely on "
      "Whisper's own first-30 s guess.")
    w("")
    w("**Review flags.**")
    w("")
    w(f"- `low-conf` — detector confidence below {args.low_conf:.2f}.")
    w("- `no-sweep` — `language_source` is `whisper_default`: the sweep "
      "abstained, so the label rests on Whisper's first-30 s guess alone.")
    w(f"- `thin-text` — {args.thin_words} clean words or fewer; too little "
      "text to justify a language.")
    w("- `1-window` — a single window carried the whole vote.")
    w(f"- `rare-lang` — a language with {args.rare_max} or fewer videos in the "
      "corpus; in a set this size these are usually misdetections on short or "
      "noisy audio.")
    w("- `multi` — a genuine second language took a meaningful share.")
    w("")

    for c in non_en:
        rows = by_lang[c]
        flagged = {id(r): review_flags(r, rare, args) for r in rows}
        rows.sort(key=lambda r: (-len(flagged[id(r)]),
                                 r.get("language_probability") or 0.0))
        n_flagged = sum(1 for r in rows if flagged[id(r)])
        w(f"### {name_of(c)} — `{c}` ({len(rows)} video"
          f"{'s' if len(rows) != 1 else ''}, {n_flagged} flagged)")
        w("")
        w("| # | Video_ID | Dur (s) | Speech (s) | Words | Conf | Wnd | Vote | "
          "Review flags | Transcript (start) |")
        w("|---:|---|---:|---:|---:|---:|---:|---|---|---|")
        for i, r in enumerate(rows, 1):
            prob = r.get("language_probability")
            w("| {} | `{}` | {:.1f} | {:.1f} | {} | {} | {} | {} | {} | {} |".format(
                i, r["video_id"], r.get("duration_sec") or 0.0,
                r.get("speech_sec") or 0.0, r.get("n_words") or 0,
                f"{prob:.2f}" if prob is not None else "—",
                voting_windows(r), fmt_dist(r),
                " ".join(f"`{f}`" for f in flagged[id(r)]) or "—",
                clip(r.get("transcript"))))
        w("")

    w("---")
    w("")

    # -- 4. cross-model ------------------------------------------------------ #
    if sv:
        w("## 3. Cross-check against the SenseVoice pass")
        w("")
        w("Both models transcribed the same 1,940 videos. **They do not agree, "
          "and the gap is structural rather than random:** SenseVoice can only "
          "emit five languages (en/zh/ja/ko/yue), so every Hindi, Russian, "
          "Spanish, Arabic or Punjabi video in this corpus *had* to be binned "
          "into one of those five or dropped. Whisper covers ~99.")
        w("")
        sv_counts = Counter(sv.get(r["video_id"], NO_LANG) for r in recs)
        w("| Language | Whisper | SenseVoice |")
        w("|---|---:|---:|")
        keys = ["en"] + non_en
        seen = set(keys)
        keys += [k for k in sv_counts if k not in seen and k != NO_LANG]
        for c in keys:
            wn, sn = counts.get(c, 0), sv_counts.get(c, 0)
            if wn or sn:
                w(f"| {name_of(c)} (`{c}`) | {wn} | {sn} |")
        w(f"| *No language* | {n_none} | {sv_counts.get(NO_LANG, 0)} |")
        w("")

        agree = sum(1 for r in recs
                    if sv.get(r["video_id"]) == (r.get("language") or NO_LANG))
        both = [r for r in recs
                if (r.get("language") or NO_LANG) != NO_LANG
                and sv.get(r["video_id"], NO_LANG) != NO_LANG]
        both_agree = sum(1 for r in both if sv[r["video_id"]] == r["language"])
        w(f"- Exact agreement across all {total} videos, no-language included: "
          f"**{agree} ({agree/total:.1%})**.")
        w(f"- On the {len(both)} videos where *both* models named a language, "
          f"they agree on **{both_agree} ({both_agree/len(both):.1%})**.")
        w("")
        w("Where they disagree on a named language, the most common confusions "
          "are listed below — these are the highest-value videos to spot-check, "
          "because two independent models reached different answers on the same "
          "audio.")
        w("")
        pairs = Counter((r["language"], sv[r["video_id"]])
                        for r in both if sv[r["video_id"]] != r["language"])
        w("| Whisper says | SenseVoice says | Videos |")
        w("|---|---|---:|")
        for (wl, sl), n in pairs.most_common(args.top_conf):
            w(f"| {name_of(wl)} (`{wl}`) | {name_of(sl)} (`{sl}`) | {n} |")
        w("")
        w("The other systematic split is coverage: Whisper found speech in "
          f"{total - n_none} videos and SenseVoice in "
          f"{total - sv_counts.get(NO_LANG, 0)}, so the two disagree about "
          "*whether there is speech at all* on a large number of clips. Treat "
          "neither no-language set as authoritative on its own.")
        w("")
        w("---")
        w("")

    # -- 5. caveats ---------------------------------------------------------- #
    w("## 4. Caveats on these labels")
    w("")
    w("- **Confidence is not accuracy.** Whisper's detector is over-confident on "
      "short clips; a 1.00 on two seconds of audio is not evidence. Read `Conf` "
      "together with `Wnd` and `Words`.")
    w("- **The long tail is mostly wrong.** Languages appearing once or twice "
      "(`rare-lang`) are usually a misdetection on music, noise, or a few "
      "syllables — not a genuinely Basque or Welsh video. Verify before acting "
      "on them.")
    w("- **Hindi/Urdu, Malay/Indonesian, Croatian/Serbian/Bulgarian, and the "
      "Indic set** are the pairs Whisper most often swaps; they are "
      "near-identical in short spoken samples.")
    w("- **`Vote` is a share, not a duration.** Unlike the SenseVoice report, "
      "nothing here says how many seconds of each language were spoken — "
      "Whisper emits no per-segment language, so the duration-weighted view is "
      "not available.")
    w("- **No language ≠ silent.** It means no speech survived the confidence "
      "floor and the suspect-segment filter; most of that group is music-only.")
    w("- Transcripts shown are filler-filtered and truncated for display. Read "
      "the per-video JSON for full text, `transcript_raw`, and per-segment "
      "decoding statistics.")
    w("")

    return "\n".join(L)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                    help="transcription_openwhisper directory")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sensevoice-dir", type=Path, default=None,
                    help="SenseVoice output for the cross-check section; "
                         "defaults to transcription_funasr/ (or transcription/), "
                         "and the section is skipped when neither exists")
    ap.add_argument("--low-conf", type=float, default=0.7,
                    help="flag detections below this confidence")
    ap.add_argument("--thin-words", type=int, default=3,
                    help="flag videos with this many clean words or fewer")
    ap.add_argument("--rare-max", type=int, default=2,
                    help="a language with this many videos or fewer is 'rare'")
    ap.add_argument("--top-confusions", dest="top_conf", type=int, default=15)
    ap.add_argument("--multi-note", default="25%", help="threshold, for the prose only")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.dir.is_dir():
        print(f"error: not a directory: {args.dir}")
        return 2

    recs = load_records(args.dir)
    if not recs:
        print(f"error: no per-video JSON found in {args.dir}")
        return 2
    sv_dir = args.sensevoice_dir
    if sv_dir is None:
        sv_dir = next((d for d in SENSEVOICE_DIRS if d.is_dir()), None)
    sv = load_sensevoice(sv_dir)

    args.out.write_text(build(recs, sv, args), encoding="utf-8")

    total = len(recs)
    counts = Counter(r.get("language") or NO_LANG for r in recs)
    non_en = sum(n for c, n in counts.items() if c not in ("en", NO_LANG))
    print(f"videos           {total}")
    print(f"english          {counts.get('en', 0)}")
    print(f"non-english      {non_en}  "
          f"({len([c for c in counts if c not in ('en', NO_LANG)])} languages)")
    print(f"no language      {counts.get(NO_LANG, 0)}")
    print(f"sensevoice xref  {'yes' if sv else 'no'}")
    print(f"wrote            {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

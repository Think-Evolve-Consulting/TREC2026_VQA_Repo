#!/usr/bin/env python3
"""Transcribe the VQA test videos with OpenAI Whisper large-v3 (faster-whisper).

For every video in ``videos_TEST_VQA/videos`` this writes one JSON file to
``videos_TEST_VQA/transcription_openwhisper/<Video_ID>.json`` holding the
transcript, the detected language, and the decoding metadata Whisper exposes
per segment (no-speech probability, average log-probability, compression
ratio, temperature).

Pipeline per video:

    ffmpeg -> 16 kHz mono float32
      -> Silero VAD                 (speech regions; speech_sec / speech_ratio)
      -> language sweep             (detect_language over N windows of speech)
      -> Whisper large-v3           (segment text + decoding statistics)

Why the language sweep. Whisper detects language from the first 30 s of audio
only, so a clip that opens on music, a title card, or an English intro is
labelled from material that is not representative of the video. This script
instead samples up to ``--lang-windows`` windows spread across the *speech*
regions, runs detection on each, and votes with the per-window probability as
the weight. That yields both a more reliable single label and the evidence for
it (``language_windows``, ``language_distribution``), and it is what makes
``is_multilingual`` meaningful.

The winning language is then forced on the transcription pass. When the sweep
shows a genuine second language (>= ``--multilingual-threshold`` of the vote)
the pass runs in Whisper's ``multilingual`` mode instead, which re-detects per
chunk so a code-switching video is not flattened to one language.

Whisper hallucinates fluent text on music and silence ("Thanks for watching!",
"Subtitles by ..."). Such segments are flagged ``suspect`` with a reason rather
than deleted: ``transcript`` excludes them, ``transcript_raw`` keeps them, so a
caller can second-guess the heuristic.

Usage:
    python VQA/transcribe.whisper.py                     # everything, resumable
    python VQA/transcribe.whisper.py --limit 20          # smoke test
    python VQA/transcribe.whisper.py --only -5tDzjnjTV0 0-R8tKOJDGM
    python VQA/transcribe.whisper.py --translate         # + English translation
    python VQA/transcribe.whisper.py --force             # redo existing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEOS = ROOT / "videos_TEST_VQA" / "videos"
DEFAULT_OUT = ROOT / "videos_TEST_VQA" / "transcription_openwhisper"
DEFAULT_MODEL = ROOT / "models" / "faster-whisper-large-v3"

SR = 16000
WINDOW_SEC = 30.0  # Whisper's receptive field; a detection window is one of these
# --lang-min-window-sec is the length a window is *preferred* to have. When a
# video holds nothing that long, a shorter one still beats the alternative,
# which is reading the language off the first 30 s of music; --lang-min-prob
# then decides whether it is confident enough to vote.
ABSOLUTE_MIN_WINDOW_SEC = 1.0
VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v"}

ALNUM_RE = re.compile(r"[^\W_]", re.UNICODE)
# Whisper's stock hallucinations on non-speech audio, learned from its subtitle
# training data. Only matched inside a short segment: a hallucination is the
# whole segment, whereas these words also occur in real speech ("subtitles by"
# in a documentary, "thanks for watching" in a genuine vlog outro).
HALLUCINATION_PHRASES = (
    "thanks for watching", "thank you for watching", "please subscribe",
    "subscribe to my channel", "like and subscribe", "see you next time",
    "see you in the next video", "subtitles by", "subtitled by",
    "amara org", "amaraorg", "transcription by", "translated by",
    "www mooji org",
)
BOILERPLATE_MAX_WORDS = 8
# Subtitle annotations Whisper reproduces verbatim for non-speech audio. These
# are only ever the entire segment, so they are matched on equality -- a
# sentence that merely contains the word "music" is real speech.
NON_SPEECH_TOKENS = {
    "music", "silence", "applause", "beep", "laughter", "laughs", "foreign",
    "inaudible", "no audio", "blank audio", "background noise", "bgm",
}


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ",
                  re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)).strip().lower()


def is_boilerplate(text: str) -> bool:
    """True for Whisper's canned output on music and silence."""
    norm = normalise(text)
    if not norm:
        return False
    if norm in NON_SPEECH_TOKENS:
        return True
    return (len(norm.split()) <= BOILERPLATE_MAX_WORDS
            and any(h in norm for h in HALLUCINATION_PHRASES))


def suspect_reasons(seg: dict, args: argparse.Namespace) -> list[str]:
    """Why this segment should not be trusted as speech. Empty means clean."""
    reasons = []
    if not ALNUM_RE.search(seg["text"]):
        reasons.append("no_alnum")
    if seg["no_speech_prob"] is not None and seg["no_speech_prob"] >= args.no_speech_prob:
        reasons.append("no_speech_prob")
    if seg["avg_logprob"] is not None and seg["avg_logprob"] <= args.min_logprob:
        reasons.append("low_logprob")
    # Whisper's own repetition guard; a segment above it is looping on itself.
    if seg["compression_ratio"] is not None and seg["compression_ratio"] >= 2.4:
        reasons.append("repetitive")
    if is_boilerplate(seg["text"]):
        reasons.append("boilerplate")
    return reasons


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #
def decode_audio(path: Path) -> np.ndarray:
    """Decode any container to mono 16 kHz float32. Empty array if no audio."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
         "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0 and not proc.stdout:
        raise RuntimeError(
            (proc.stderr.decode("utf-8", "replace").strip() or "ffmpeg failed")[:500]
        )
    # A stray odd-sized tail would break the frombuffer view; copy so the array
    # is writable for downstream torch calls.
    return np.ascontiguousarray(np.frombuffer(proc.stdout, dtype=np.float32))


def speech_regions(audio: np.ndarray, min_silence_ms: int) -> list[tuple[float, float]]:
    """Silero VAD speech spans in seconds."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(min_silence_duration_ms=min_silence_ms)
    return [(c["start"] / SR, c["end"] / SR)
            for c in get_speech_timestamps(audio, opts, sampling_rate=SR)]


def sample_windows(regions: list[tuple[float, float]], n: int,
                   min_win_sec: float) -> list[tuple[float, float]]:
    """Up to ``n`` windows of speech, spread evenly over the whole timeline.

    Sampling from the VAD spans rather than from the raw timeline keeps the
    language vote off the music and silence, where detection is noise.
    """
    usable = [(s, e) for s, e in regions if e - s >= min_win_sec]
    if not usable:
        # Nothing long enough; fall back to the longest span there is.
        usable = sorted(regions, key=lambda r: r[1] - r[0], reverse=True)[:1]
    if not usable:
        return []

    total = sum(e - s for s, e in usable)
    n = max(1, min(n, int(total // min_win_sec) or 1))
    # Walk the concatenated speech timeline and cut a window at each of the n
    # evenly spaced offsets, so early and late speech are equally represented.
    out: list[tuple[float, float]] = []
    for k in range(n):
        target = total * (k + 0.5) / n
        acc = 0.0
        for s, e in usable:
            span = e - s
            if acc + span >= target or (s, e) == usable[-1]:
                start = min(max(s, s + (target - acc) - WINDOW_SEC / 2), e - min_win_sec)
                start = max(s, start)
                out.append((start, min(start + WINDOW_SEC, e)))
                break
            acc += span
    # Distinct windows only; adjacent targets can land in the same short span.
    seen, uniq = set(), []
    for w in out:
        key = round(w[0], 1)
        if key not in seen:
            seen.add(key)
            uniq.append(w)
    return uniq


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class WhisperPipeline:
    def __init__(self, model_dir: Path, device: str, compute_type: str,
                 batch_size: int, cpu_threads: int):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.model = WhisperModel(
            str(model_dir), device=device, compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        self.batch_size = batch_size
        # The batched pipeline splits on VAD and decodes chunks in parallel; it
        # is several times faster and is the only reason a 13 h corpus fits in
        # one sitting. batch_size 1 keeps the plain sequential decoder.
        self.batched = BatchedInferencePipeline(model=self.model) if batch_size > 1 else None

    def detect(self, window: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        lang, prob, all_probs = self.model.detect_language(audio=window)
        return lang, prob, all_probs[:3]

    def transcribe(self, audio: np.ndarray, language: str | None,
                   multilingual: bool, task: str, args: argparse.Namespace):
        kw = dict(
            language=language, task=task, multilingual=multilingual,
            beam_size=args.beam_size, temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=args.min_silence_ms),
            condition_on_previous_text=args.condition_on_previous,
            no_speech_threshold=args.no_speech_prob,
            log_prob_threshold=args.min_logprob,
            word_timestamps=False,
            # The batched pipeline defaults this to True, which collapses each
            # VAD chunk into one segment spanning up to 30 s. Decoding the
            # timestamp tokens restores sentence-level segments at no
            # measurable cost, and they are what makes the timings usable.
            without_timestamps=False,
        )
        if self.batched is not None:
            segments, info = self.batched.transcribe(audio, batch_size=self.batch_size, **kw)
        else:
            segments, info = self.model.transcribe(audio, **kw)
        return list(segments), info


def segment_record(i: int, seg, args: argparse.Namespace) -> dict:
    rec = {
        "id": i,
        "start": round(seg.start, 3),
        "end": round(seg.end, 3),
        "duration": round(seg.end - seg.start, 3),
        "text": (seg.text or "").strip(),
        "no_speech_prob": round(seg.no_speech_prob, 4) if seg.no_speech_prob is not None else None,
        "avg_logprob": round(seg.avg_logprob, 4) if seg.avg_logprob is not None else None,
        "compression_ratio": round(seg.compression_ratio, 3) if seg.compression_ratio is not None else None,
        "temperature": seg.temperature,
        "n_tokens": len(seg.tokens) if seg.tokens else 0,
    }
    rec["suspect_reasons"] = suspect_reasons(rec, args)
    rec["suspect"] = bool(rec["suspect_reasons"])
    return rec


# --------------------------------------------------------------------------- #
# per-video record
# --------------------------------------------------------------------------- #
def summarise(video: Path, duration: float, regions: list[tuple[float, float]],
              windows: list[dict], lang: str | None, lang_dist: dict,
              confidence: float | None, source: str | None,
              multilingual: bool, segments: list[dict], info,
              translation: dict | None, models: dict, elapsed: float,
              error: str | None = None) -> dict:
    clean = [s for s in segments if not s["suspect"] and s["text"]]
    speech_sec = sum(e - s for s, e in regions)
    logprobs = [s["avg_logprob"] for s in clean if s["avg_logprob"] is not None]

    return {
        "video_id": video.stem,
        "file": video.name,
        "duration_sec": round(duration, 3),
        "has_audio": duration > 0,
        "error": error,

        # --- language -------------------------------------------------------
        "language": lang,
        "language_probability": round(confidence, 4) if confidence is not None else None,
        "language_share": round(lang_dist.get(lang, 0.0), 4) if lang else None,
        "language_source": source,
        "language_distribution": {k: round(v, 4) for k, v in
                                  sorted(lang_dist.items(), key=lambda kv: -kv[1])},
        # From the sweep alone. faster-whisper's Segment carries no language
        # field even in multilingual mode, so there is no per-segment label to
        # corroborate it with.
        "is_multilingual": multilingual,
        "n_language_windows": len(windows),
        "language_windows": windows,
        "whisper_language": getattr(info, "language", None),
        "whisper_language_probability": round(info.language_probability, 4)
        if info is not None and info.language_probability is not None else None,

        # --- speech ---------------------------------------------------------
        "has_speech": bool(clean),
        "speech_sec": round(speech_sec, 2),
        "speech_ratio": round(speech_sec / duration, 3) if duration > 0 else 0.0,
        "n_speech_regions": len(regions),
        "duration_after_vad_sec": round(info.duration_after_vad, 2)
        if info is not None and info.duration_after_vad is not None else None,

        # --- transcript -----------------------------------------------------
        "n_segments": len(segments),
        "n_clean_segments": len(clean),
        "n_suspect_segments": sum(1 for s in segments if s["suspect"]),
        "suspect_reason_counts": dict(Counter(
            r for s in segments for r in s["suspect_reasons"]).most_common()),
        "mean_avg_logprob": round(sum(logprobs) / len(logprobs), 4) if logprobs else None,
        "n_words": sum(len(s["text"].split()) for s in clean),
        "transcript": " ".join(s["text"] for s in clean).strip(),
        "transcript_raw": " ".join(s["text"] for s in segments if s["text"]).strip(),
        "translation_en": translation["text"] if translation else None,
        "segments": segments,
        "translation_segments": translation["segments"] if translation else None,

        "models": models,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_sec": round(elapsed, 2),
    }


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL,
                    help="local CTranslate2 model dir, or a name like 'large-v3'")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:0 or cpu")
    ap.add_argument("--compute-type", default="float16",
                    help="float16 (GPU), int8_float16 (less VRAM), int8 (CPU)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="VAD chunks decoded in parallel; 1 = sequential decoder")
    ap.add_argument("--cpu-threads", type=int, default=0, help="0 = CTranslate2 default")
    ap.add_argument("--beam-size", type=int, default=5)

    ap.add_argument("--lang-windows", type=int, default=6,
                    help="30 s speech windows sampled for language detection; 0 = "
                         "use Whisper's own first-30 s detection")
    ap.add_argument("--lang-min-window-sec", type=float, default=3.0,
                    help="ignore speech spans shorter than this when sampling")
    ap.add_argument("--lang-min-prob", type=float, default=0.5,
                    help="a detection window below this confidence does not vote")
    ap.add_argument("--multilingual-threshold", type=float, default=0.25,
                    help="second language's share of the vote above which the "
                         "transcription pass re-detects language per chunk")
    ap.add_argument("--min-silence-ms", type=int, default=500,
                    help="Silero VAD silence needed to split a speech region")
    ap.add_argument("--no-speech-prob", type=float, default=0.6,
                    help="segments at or above this are flagged non-speech")
    ap.add_argument("--min-logprob", type=float, default=-1.0,
                    help="segments at or below this average log-prob are flagged")
    ap.add_argument("--condition-on-previous", action="store_true",
                    help="feed prior text as context; better prose, but lets a "
                         "hallucination propagate down the file")
    ap.add_argument("--translate", action="store_true",
                    help="second pass emitting an English translation of "
                         "non-English videos (roughly doubles their cost)")

    ap.add_argument("--decode-workers", type=int, default=3,
                    help="threads pre-decoding audio ahead of the GPU")
    ap.add_argument("--limit", type=int, help="process at most N videos")
    # Twelve Video_IDs start with a hyphen (-5tDzjnjTV0), which argparse reads
    # as an option. `extend` keeps the plain `--only A B` form working while
    # allowing `--only=-5tDzjnjTV0` (repeatable) for those twelve.
    ap.add_argument("--only", action="extend", nargs="+",
                    help="specific filenames or video IDs; for an ID starting "
                         "with '-' use --only=-5tDzjnjTV0, repeated per ID")
    ap.add_argument("--force", action="store_true", help="re-process existing output")
    # Not `_index.json`: real Video_IDs start with an underscore (_3GqKCAd-P0),
    # so a downstream "skip files starting with _" would silently drop videos.
    ap.add_argument("--summary", type=Path,
                    help="index JSON (default: <out-dir>/transcription_index.json)")
    ap.add_argument("--reindex", action="store_true",
                    help="rebuild the index from the JSON already on disk, no inference")
    return ap.parse_args(argv)


INDEX_FIELDS = ("video_id", "file", "duration_sec", "language",
                "language_probability", "language_share", "language_source",
                "is_multilingual", "whisper_language",
                "speech_sec", "speech_ratio", "has_speech", "n_words",
                "n_segments", "n_suspect_segments", "mean_avg_logprob", "error")


def index_entry(rec: dict) -> dict:
    return {k: rec.get(k) for k in INDEX_FIELDS}


def model_meta(args: argparse.Namespace) -> dict:
    return {
        "asr": "openai/whisper-large-v3 (faster-whisper / CTranslate2)",
        "model_dir": str(args.model_dir),
        "vad": "silero-vad (faster-whisper bundled)",
        "sample_rate": SR,
        "compute_type": args.compute_type,
        "batch_size": args.batch_size,
        "beam_size": args.beam_size,
        "lang_windows": args.lang_windows,
        "lang_min_prob": args.lang_min_prob,
        "lang_min_window_sec": args.lang_min_window_sec,
        "multilingual_threshold": args.multilingual_threshold,
        "condition_on_previous_text": args.condition_on_previous,
        "translation_pass": args.translate,
    }


def load_index(out_dir: Path) -> dict[str, dict]:
    """Every per-video JSON on disk, keyed by Video_ID.

    The index is rebuilt from the files rather than from the current run, so a
    resumed run reports the whole corpus instead of just the videos it touched.
    """
    entries: dict[str, dict] = {}
    for p in sorted(out_dir.glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(rec, dict) and "video_id" in rec and "segments" in rec:
            entries[rec["video_id"]] = index_entry(rec)
    return entries


def write_index(path: Path, args: argparse.Namespace, index: dict[str, dict]) -> None:
    videos = sorted(index.values(), key=lambda e: e["video_id"])
    failures = [{"video_id": e["video_id"], "error": e["error"]}
                for e in videos if e["error"]]
    langs = Counter(e["language"] for e in videos if e.get("language"))
    write_json(path, {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "videos_dir": str(args.videos_dir),
        "models": model_meta(args),
        "n_videos": len(videos),
        "n_failed": len(failures),
        "n_with_speech": sum(1 for e in videos if e.get("has_speech")),
        "n_multilingual": sum(1 for e in videos if e.get("is_multilingual")),
        "language_counts": dict(langs.most_common()),
        "failures": failures,
        "videos": videos,
    })


def process(pipe: WhisperPipeline, audio: np.ndarray, args: argparse.Namespace):
    """VAD -> language sweep -> transcription. Returns the parts of a record."""
    regions = speech_regions(audio, args.min_silence_ms)

    windows: list[dict] = []
    lang_dist: Counter = Counter()
    if args.lang_windows > 0:
        for start, end in sample_windows(regions, args.lang_windows,
                                         args.lang_min_window_sec):
            chunk = audio[int(start * SR): int(end * SR)]
            if chunk.size < int(ABSOLUTE_MIN_WINDOW_SEC * SR):
                continue
            lang, prob, top = pipe.detect(chunk)
            span = end - start
            # Weighted by confidence *and* by how much speech the window
            # actually contains. Detection on a 2 s window is close to a coin
            # flip -- letting one outvote a confident 20 s window is what made
            # a monolingual video come back "multilingual".
            weight = prob * span if prob >= args.lang_min_prob else 0.0
            windows.append({
                "start": round(start, 2), "end": round(end, 2),
                "language": lang, "probability": round(prob, 4),
                "weight": round(weight, 3),
                "top3": [{"language": l, "probability": round(p, 4)} for l, p in top],
            })
            lang_dist[lang] += weight

    # Windows below --lang-min-prob carry zero weight; drop them entirely so a
    # coin-flip language cannot appear in the distribution at 0.0.
    ranked = [(k, v) for k, v in lang_dist.most_common() if v > 0]
    total = sum(v for _, v in ranked)
    shares = {k: v / total for k, v in ranked} if total else {}
    lang = ranked[0][0] if ranked else None
    source = "window_sweep" if lang else None
    # Mean detector confidence over the windows that voted for the winner, kept
    # apart from the vote share: one window at p=0.55 is a 100% share but a
    # coin-flip detection, and conflating the two hides that.
    confidence = None
    if lang:
        picked = [w["probability"] for w in windows
                  if w["language"] == lang and w["weight"] > 0]
        confidence = sum(picked) / len(picked) if picked else None
    multilingual = (len(ranked) > 1
                    and shares[ranked[1][0]] >= args.multilingual_threshold)

    # In multilingual mode Whisper re-detects per chunk, so forcing a language
    # would defeat it; the sweep result is still reported as the dominant one.
    segs, info = pipe.transcribe(
        audio, language=None if multilingual else lang,
        multilingual=multilingual, task="transcribe", args=args)
    segments = [segment_record(i, s, args) for i, s in enumerate(segs)]

    # The sweep votes only where VAD found speech, so an empty vote means there
    # was none. Whisper still returns a label in that case -- it is forced to
    # pick one -- but it was read off music or silence, so it is only trusted
    # when the transcript itself came back with something. Otherwise the video
    # has no language, rather than a wrong one.
    if lang is None:
        if any(not s["suspect"] and s["text"] for s in segments):
            lang = getattr(info, "language", None)
            if lang:
                confidence = info.language_probability
                shares = {lang: 1.0}
                source = "whisper_default"
        else:
            source = "no_speech"

    translation = None
    if args.translate and lang and lang != "en" and any(not s["suspect"] for s in segments):
        tsegs, _ = pipe.transcribe(audio, language=None if multilingual else lang,
                                   multilingual=multilingual, task="translate", args=args)
        trecs = [segment_record(i, s, args) for i, s in enumerate(tsegs)]
        translation = {
            "text": " ".join(s["text"] for s in trecs if not s["suspect"]).strip(),
            "segments": trecs,
        }

    return (regions, windows, lang, shares, confidence, source, multilingual,
            segments, info, translation)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.videos_dir.is_dir():
        print(f"error: videos dir not found: {args.videos_dir}", file=sys.stderr)
        return 2
    # A bare name like "large-v3" is a valid model spec that faster-whisper
    # downloads itself, so only a path-looking argument has to exist on disk.
    if os.sep in str(args.model_dir) and not args.model_dir.is_dir():
        print(f"error: model dir not found: {args.model_dir}\n"
              f"       download it first, e.g.\n"
              f"       huggingface-cli download Systran/faster-whisper-large-v3 "
              f"--local-dir models/faster-whisper-large-v3", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or (args.out_dir / "transcription_index.json")

    videos = sorted(p for p in args.videos_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if args.only:
        wanted = set(args.only)
        videos = [p for p in videos if p.name in wanted or p.stem in wanted]
        missing = wanted - {p.name for p in videos} - {p.stem for p in videos}
        for m in sorted(missing):
            print(f"warning: no such video: {m}", file=sys.stderr)

    todo = videos if args.force else [
        p for p in videos if not (args.out_dir / f"{p.stem}.json").exists()]
    skipped = len(videos) - len(todo)
    if args.limit:
        todo = todo[: args.limit]

    # Seeded from disk so a resumed or --limit run still writes a complete index.
    index: dict[str, dict] = load_index(args.out_dir)

    if args.reindex:
        write_index(summary_path, args, index)
        print(f"reindexed {len(index)} videos -> {summary_path}")
        return 0

    print(f"videos: {len(videos)} | to process: {len(todo)} | already done: {skipped}")
    if not todo:
        write_index(summary_path, args, index)
        print(f"index refreshed ({len(index)} videos): {summary_path}")
        return 0

    pipe = WhisperPipeline(args.model_dir, args.device, args.compute_type,
                           args.batch_size, args.cpu_threads)
    models = model_meta(args)

    n_failed = 0
    t_start = time.time()
    audio_total = 0.0

    def decode_safe(p: Path):
        try:
            return p, decode_audio(p), None
        except Exception as exc:  # noqa: BLE001 - recorded per video
            return p, np.empty(0, dtype=np.float32), str(exc)

    def decoded_stream(paths: list[Path], workers: int):
        """Decode ahead of the consumer, but only a bounded number of videos.

        ThreadPoolExecutor.map would submit all of them at once and hold every
        decoded waveform in RAM; this keeps at most `depth` in flight.
        """
        depth = max(2, workers * 2)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            queue = deque()
            it = iter(paths)
            for p in islice(it, depth):
                queue.append(pool.submit(decode_safe, p))
            while queue:
                fut = queue.popleft()
                for p in islice(it, 1):
                    queue.append(pool.submit(decode_safe, p))
                yield fut.result()

    # Decoding is CPU/IO bound and the GPU work is not; overlap them.
    for n, (video, audio, err) in enumerate(
            decoded_stream(todo, args.decode_workers), 1):
        t0 = time.time()
        duration = len(audio) / SR
        audio_total += duration
        error = err
        parts = ([], [], None, {}, None, None, False, [], None, None)
        if error is None and duration > 0:
            try:
                parts = process(pipe, audio, args)
            except Exception as exc:  # noqa: BLE001
                error = f"inference failed: {exc}"
        elif error is None:
            error = "no audio stream"

        (regions, windows, lang, shares, conf, source, multi,
         segments, info, translation) = parts
        rec = summarise(video, duration, regions, windows, lang, shares, conf,
                        source, multi, segments, info, translation, models,
                        time.time() - t0, error)
        write_json(args.out_dir / f"{video.stem}.json", rec)

        index[rec["video_id"]] = index_entry(rec)
        if error:
            n_failed += 1

        el = time.time() - t_start
        eta = (len(todo) - n) / (n / el) if el else 0
        flag = "!" if error else " "
        print(f"[{n}/{len(todo)}]{flag} {video.name:<40.40s} "
              f"{duration:6.1f}s  seg={len(segments):3d} "
              f"lang={str(lang):>5s}{'*' if multi else ' '} "
              f"words={rec['n_words']:5d} "
              f"eta={eta/60:5.1f}m", flush=True)

        if n % 50 == 0:
            write_index(summary_path, args, index)

    write_index(summary_path, args, index)
    el = time.time() - t_start
    rtf = f"RTF {el/audio_total:.4f}" if audio_total else "no audio decoded"
    print(f"\ndone: {len(todo)} processed ({n_failed} failed), "
          f"{audio_total/60:.1f} min audio in {el/60:.1f} min wall ({rtf})")
    print(f"per-video JSON: {args.out_dir}  ({len(index)} videos total)")
    print(f"index:          {summary_path}")
    return 1 if n_failed == len(todo) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Every finished video is already on disk; a re-run picks up from here.
        print("\ninterrupted -- completed videos are saved, re-run to resume",
              file=sys.stderr)
        raise SystemExit(130)

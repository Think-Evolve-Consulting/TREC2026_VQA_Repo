#!/usr/bin/env python3
"""Transcribe the VQA test videos with SenseVoiceSmall.

For every video in ``videos_TEST_VQA/videos`` this writes one JSON file to
``videos_TEST_VQA/transcription/<Video_ID>.json`` holding the spoken text plus
the three side-channels SenseVoice predicts alongside it: spoken language,
speaker emotion, and the audio event (music, laughter, applause, ...).

Pipeline per video:

    ffmpeg -> 16 kHz mono float32
      -> FSMN-VAD                     (find the acoustically active regions)
      -> SenseVoiceSmall per segment  (text + language + emotion + event)

SenseVoice is trained on utterances of at most 30 s, so the audio is always cut
into VAD segments rather than fed in whole -- a single long call collapses to
one language/emotion/event label for the entire video and the text quality
drops noticeably.

The regions VAD rejects are not discarded. They are split into windows and
classified too, because that is where the ambient audio lives: a clip of pure
music or applause produces no VAD segments at all, so speech-only coverage
would emit nothing for it. Those windows are marked ``type: "ambient"``.

Usage:
    python VQA/transcribe.sensevoice.py                     # everything, resumable
    python VQA/transcribe.sensevoice.py --limit 20          # smoke test
    python VQA/transcribe.sensevoice.py --only 6501.webm ChainedforLife-26.webm
    python VQA/transcribe.sensevoice.py --force             # redo existing
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
DEFAULT_OUT = ROOT / "videos_TEST_VQA" / "transcription"
DEFAULT_SV = ROOT / "models" / "SenseVoiceSmall"
DEFAULT_VAD = ROOT / "models" / "fsmn-vad"

SR = 16000
VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v"}

# SenseVoice prefixes every hypothesis with <|lang|><|emotion|><|event|><|itn|>.
TAG_RE = re.compile(r"<\|([^|<>]+)\|>")
LANGS = {"zh", "en", "yue", "ja", "ko", "nospeech"}
EMOTIONS = {
    "HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED",
    "EMO_UNKNOWN",
}
EVENTS = {
    "Speech", "BGM", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough",
    "Event_UNK",
}

ALNUM_RE = re.compile(r"[^\W_]", re.UNICODE)
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
# Handed a music or noise region, SenseVoice still emits its most likely token.
# Over a 150-video sample the only recurring artefacts were bare punctuation and
# a lone "the"; genuine one-word utterances ("Yeah", "Oh") are left alone.
FILLER_WORDS = {"the"}


def is_filler(text: str) -> bool:
    """True when the hypothesis carries no real content."""
    if not ALNUM_RE.search(text):
        return True
    return PUNCT_RE.sub("", text).strip().lower() in FILLER_WORDS


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
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    # A stray odd-sized tail would break the frombuffer view; copy so the array
    # is writable for downstream torch calls.
    return np.ascontiguousarray(audio)


def ambient_windows(
    speech: list[tuple[int, int]], total_ms: int, win_ms: int, min_gap_ms: int
) -> list[tuple[int, int]]:
    """Windows covering everything VAD left out, so ambient audio is not lost."""
    out: list[tuple[int, int]] = []
    cursor = 0
    for start, end in sorted(speech) + [(total_ms, total_ms)]:
        gap = start - cursor
        if gap >= min_gap_ms:
            pos = cursor
            while pos < start:
                stop = min(pos + win_ms, start)
                # Do not emit a sliver at the tail of a long gap.
                if stop - pos >= min_gap_ms:
                    out.append((pos, stop))
                pos = stop
        cursor = max(cursor, end)
    return out


# --------------------------------------------------------------------------- #
# tag parsing
# --------------------------------------------------------------------------- #
def parse_tagged(raw: str) -> dict:
    """Split ``<|en|><|HAPPY|><|Laughter|><|withitn|>hello`` into its parts."""
    lang = emotion = event = None
    itn = None
    pos = 0
    for m in TAG_RE.finditer(raw):
        if m.start() != pos:  # tags are a contiguous prefix; stop at the text
            break
        tag = m.group(1)
        if tag in LANGS and lang is None:
            lang = tag
        elif tag in EMOTIONS and emotion is None:
            emotion = "UNKNOWN" if tag == "EMO_UNKNOWN" else tag
        elif tag in EVENTS and event is None:
            event = None if tag == "Event_UNK" else tag
        elif tag in ("withitn", "woitn"):
            itn = tag == "withitn"
        pos = m.end()
    text = TAG_RE.sub("", raw[pos:]).strip()
    return {"language": lang, "emotion": emotion, "event": event,
            "itn": itn, "text": text}


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class SenseVoicePipeline:
    def __init__(self, sv_dir: Path, vad_dir: Path, device: str,
                 max_segment_ms: int, batch_size: int, forced_emotion: bool):
        from funasr import AutoModel

        self.batch_size = batch_size
        self.forced_emotion = forced_emotion
        self.vad = AutoModel(
            model=str(vad_dir), device=device,
            disable_update=True, disable_pbar=True, disable_log=True,
            max_single_segment_time=max_segment_ms,
        )
        self.asr = AutoModel(
            model=str(sv_dir), device=device,
            disable_update=True, disable_pbar=True, disable_log=True,
        )

    def _asr(self, chunks: list[np.ndarray], **kw) -> list[str]:
        res = self.asr.generate(
            input=chunks, fs=SR, language="auto", use_itn=True,
            batch_size=self.batch_size, **kw,
        )
        return [r.get("text", "") for r in res]

    def run(self, audio: np.ndarray, win_ms: int, min_gap_ms: int) -> list[dict]:
        total_ms = int(len(audio) / SR * 1000)
        if total_ms <= 0:
            return []

        speech = [tuple(v) for v in self.vad.generate(input=audio, fs=SR)[0]["value"]]
        spans = [(s, e, "speech") for s, e in speech]
        spans += [(s, e, "ambient")
                  for s, e in ambient_windows(speech, total_ms, win_ms, min_gap_ms)]
        spans.sort(key=lambda x: x[0])
        if not spans:
            return []

        chunks = [audio[int(s / 1000 * SR): int(e / 1000 * SR)] for s, e, _ in spans]
        raw = self._asr(chunks)
        # A second pass with the unknown-emotion token banned, so every segment
        # also carries a best-guess emotion. It is noisier than the honest label
        # above, hence a separate field rather than a replacement.
        forced = self._asr(chunks, ban_emo_unk=True) if self.forced_emotion else [""] * len(spans)

        segments = []
        for i, ((start, end, kind), r, f) in enumerate(zip(spans, raw, forced)):
            p = parse_tagged(r)
            seg = {
                "id": i,
                "type": kind,
                "start": round(start / 1000, 3),
                "end": round(end / 1000, 3),
                "duration": round((end - start) / 1000, 3),
                "language": p["language"],
                "emotion": p["emotion"],
                "event": p["event"],
                "text": p["text"],
                # Model output is never dropped, only flagged, so that a caller
                # can second-guess the heuristic.
                "filler": is_filler(p["text"]),
                "raw": r,
            }
            if self.forced_emotion:
                seg["emotion_forced"] = parse_tagged(f)["emotion"]
            segments.append(seg)
        return segments


# --------------------------------------------------------------------------- #
# per-video record
# --------------------------------------------------------------------------- #
def summarise(video: Path, segments: list[dict], duration: float,
              models: dict, elapsed: float, error: str | None = None) -> dict:
    speech = [s for s in segments if s["type"] == "speech"]
    spoken = [s for s in speech if s["text"] and not s["filler"]]

    # Dominant language weighted by segment duration. Segments the model tagged
    # `nospeech`, or whose text is a non-speech artefact, carry no language
    # evidence and would otherwise let a music-only clip claim a language.
    lang_time: Counter = Counter()
    for s in spoken:
        if s["language"] and s["language"] != "nospeech":
            lang_time[s["language"]] += s["duration"]

    # The acoustic-event head is independent of the text, so a segment whose
    # transcript was filtered still counts as a real music/laughter detection.
    events = Counter(s["event"] for s in segments if s["event"])
    ambient = sorted(e for e in events if e != "Speech")
    # Emotion is only meaningful where somebody actually spoke.
    emotions = Counter(s["emotion"] for s in spoken if s["emotion"])
    forced = Counter(s["emotion_forced"] for s in spoken if s.get("emotion_forced"))

    record = {
        "video_id": video.stem,
        "file": video.name,
        "duration_sec": round(duration, 3),
        "has_audio": duration > 0,
        "error": error,
        "language": lang_time.most_common(1)[0][0] if lang_time else None,
        "language_seconds": {k: round(v, 2) for k, v in lang_time.most_common()},
        "is_multilingual": len(lang_time) > 1,
        "has_speech": bool(spoken),
        "speech_sec": round(sum(s["duration"] for s in speech), 2),
        "speech_ratio": round(sum(s["duration"] for s in speech) / duration, 3)
        if duration > 0 else 0.0,
        "ambient_events": ambient,
        "event_counts": dict(events.most_common()),
        "emotion_counts": dict(emotions.most_common()),
        "emotion_forced_counts": dict(forced.most_common()),
        "dominant_emotion": next(
            (e for e, _ in emotions.most_common() if e != "UNKNOWN"), None),
        "dominant_emotion_forced": forced.most_common(1)[0][0] if forced else None,
        "n_segments": len(segments),
        "n_speech_segments": len(speech),
        "n_spoken_segments": len(spoken),
        "transcript": " ".join(s["text"] for s in spoken),
        "segments": segments,
        "models": models,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_sec": round(elapsed, 2),
    }
    return record


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
    ap.add_argument("--sensevoice-dir", type=Path, default=DEFAULT_SV)
    ap.add_argument("--vad-dir", type=Path, default=DEFAULT_VAD)
    ap.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="VAD segments per ASR forward pass")
    ap.add_argument("--max-segment-sec", type=float, default=30.0,
                    help="VAD hard split; SenseVoice is trained up to 30 s")
    ap.add_argument("--ambient-window-sec", type=float, default=20.0,
                    help="window size used to sweep the non-speech regions")
    ap.add_argument("--min-ambient-sec", type=float, default=1.0,
                    help="ignore non-speech gaps shorter than this")
    ap.add_argument("--no-forced-emotion", action="store_true",
                    help="skip the second pass that forces a non-UNKNOWN emotion")
    ap.add_argument("--decode-workers", type=int, default=3,
                    help="threads pre-decoding audio ahead of the GPU")
    ap.add_argument("--limit", type=int, help="process at most N videos")
    ap.add_argument("--only", nargs="+", help="specific filenames or video IDs")
    ap.add_argument("--force", action="store_true", help="re-process existing output")
    # Not `_index.json`: real Video_IDs start with an underscore (_3GqKCAd-P0),
    # so a downstream "skip files starting with _" would silently drop videos.
    ap.add_argument("--summary", type=Path,
                    help="index JSON (default: <out-dir>/transcription_index.json)")
    ap.add_argument("--reindex", action="store_true",
                    help="rebuild the index from the JSON already on disk, no inference")
    return ap.parse_args(argv)


INDEX_FIELDS = ("video_id", "file", "duration_sec", "language", "is_multilingual",
                "speech_sec", "has_speech", "ambient_events", "dominant_emotion",
                "dominant_emotion_forced", "n_segments", "error")


def index_entry(rec: dict) -> dict:
    return {k: rec.get(k) for k in INDEX_FIELDS}


def model_meta(args: argparse.Namespace) -> dict:
    return {
        "asr": "FunAudioLLM/SenseVoiceSmall",
        "vad": "funasr/fsmn-vad",
        "sample_rate": SR,
        "max_segment_sec": args.max_segment_sec,
        "ambient_window_sec": args.ambient_window_sec,
        "forced_emotion_pass": not args.no_forced_emotion,
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


def write_index(path: Path, args: argparse.Namespace,
                index: dict[str, dict]) -> None:
    videos = sorted(index.values(), key=lambda e: e["video_id"])
    failures = [{"video_id": e["video_id"], "error": e["error"]}
                for e in videos if e["error"]]
    write_json(path, {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "videos_dir": str(args.videos_dir),
        "models": model_meta(args),
        "n_videos": len(videos),
        "n_failed": len(failures),
        "n_with_speech": sum(1 for e in videos if e.get("has_speech")),
        "failures": failures,
        "videos": videos,
    })


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.videos_dir.is_dir():
        print(f"error: videos dir not found: {args.videos_dir}", file=sys.stderr)
        return 2
    for d in (args.sensevoice_dir, args.vad_dir):
        if not d.is_dir():
            print(f"error: model dir not found: {d}\n"
                  f"       download it first, e.g.\n"
                  f"       huggingface-cli download FunAudioLLM/SenseVoiceSmall "
                  f"--local-dir models/SenseVoiceSmall", file=sys.stderr)
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

    pipe = SenseVoicePipeline(
        args.sensevoice_dir, args.vad_dir, args.device,
        int(args.max_segment_sec * 1000), args.batch_size,
        forced_emotion=not args.no_forced_emotion,
    )
    models = model_meta(args)
    win_ms = int(args.ambient_window_sec * 1000)
    min_gap_ms = int(args.min_ambient_sec * 1000)

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
        segments: list[dict] = []
        error = err
        if error is None and duration > 0:
            try:
                segments = pipe.run(audio, win_ms, min_gap_ms)
            except Exception as exc:  # noqa: BLE001
                error = f"inference failed: {exc}"
        elif error is None:
            error = "no audio stream"

        rec = summarise(video, segments, duration, models, time.time() - t0, error)
        write_json(args.out_dir / f"{video.stem}.json", rec)

        index[rec["video_id"]] = index_entry(rec)
        if error:
            n_failed += 1

        el = time.time() - t_start
        eta = (len(todo) - n) / (n / el) if el else 0
        flag = "!" if error else " "
        print(f"[{n}/{len(todo)}]{flag} {video.name:<40.40s} "
              f"{duration:6.1f}s  seg={len(segments):3d} "
              f"lang={str(rec['language']):>5s} "
              f"amb={','.join(rec['ambient_events']) or '-':<20.20s} "
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

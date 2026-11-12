#!/usr/bin/env python3
"""Prepare the two things the debate needs that the panel never had.

Run locally, before `VQA/debate.rank.mc.py`. Produces:

    debate.manifest.json   which videos the Colab sessions must cut frames for,
                           and how many -- 32 normally, 64 for counting
                           questions, because every model in the panel
                           undercounts and a denser strip is the only fix that
                           does not depend on winning an argument
    evidence.2026.json     per-query aids: the counting flag, and for sound
                           questions a CLAP score per option, measured against
                           the clip's actual soundtrack

Why these two and nothing else, measured on the dev sample:

- 38 of the 203 disputed dev queries were counting, and the *oracle* over both
  judges only reached 24 of them. Arbitration cannot fix counting; evidence can.
- "What can be heard" answered from silent video is the one error theme unique
  to the strongest model. ASR cannot answer "which brass instrument"; an audio
  encoder can, so CLAP scores each option text against the audio directly.

The manifest is ~50 KB and is the only file that has to reach Drive. Frames are
cut on the Colab side from the videos already mounted there, which is why
nothing large is uploaded.

Usage:
    python VQA/prep.debate.mc.py                      # manifest + counting flags
    python VQA/prep.debate.mc.py --audio              # also CLAP the sound queries
    python VQA/prep.debate.mc.py --audio --limit 20   # try the audio path first
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUDGE_SCRIPT = ROOT / "VQA" / "judge.rank.mc.py"

spec = importlib.util.spec_from_file_location("judge_rank_mc", JUDGE_SCRIPT)
JUDGE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(JUDGE)
MUSE = JUDGE.MUSE

DEFAULT_RUNS = ["qwen38.run.task2.json", "qwen3vl.run.task2.json",
                "gemma4.vllm.run.task2.json", "cosmos.run.task2.json",
                "internvl.run.task2.json", "llavavideo.run.task2.json",
                "videollama3.run.task2.json"]

FRAMES_NORMAL = 32
FRAMES_COUNTING = 64
CLAP_MODEL = "laion/clap-htsat-unfused"
SAMPLE_RATE = 48000

# Option texts that are just a number carry no acoustic meaning, so CLAP
# must not be asked to score them against a soundtrack.
NUMERIC = re.compile(r"^\W*(\d+|zero|one|two|three|four|five|six|seven|"
                     r"eight|nine|ten|eleven|twelve|none)\W*$", re.I)

COUNTING = re.compile(r"\bhow many\b|\bhow much\b|\bnumber of\b", re.I)
SOUND = re.compile(r"\bhear|\bsound|\bsays?\b|\bsaid\b|\bsing|\bsong\b|"
                   r"\bmusic|\bvoice|\binstrument|\bnoise|\bloud", re.I)


def video_path(directory: Path, video_id: str) -> Path | None:
    """The clip on disk. Video_ID is a filename for some, a bare ID for others."""
    stem = Path(video_id).stem

    for name in (video_id, f"{video_id}.mp4", f"{stem}.mp4", f"{stem}.webm"):
        candidate = directory / name

        if candidate.is_file():
            return candidate

    matches = sorted(directory.glob(f"{stem}.*"))
    return matches[0] if matches else None


def read_audio(path: Path, seconds: float = 30.0) -> "object | None":
    """Mono 48 kHz float32 via ffmpeg, no audio library required."""
    import numpy as np

    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-t", str(seconds),
           "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
           "-f", "s16le", "-acodec", "pcm_s16le", "-"]

    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    if not out.stdout:
        return None

    return np.frombuffer(out.stdout, dtype="<i2").astype("float32") / 32768.0


def score_options(clap, processor, audio, options: list[str], device: str
                  ) -> list[float] | None:
    """CLAP similarity between the soundtrack and each option's text."""
    import torch

    # transformers renamed this keyword (audios -> audio) in 5.x; accept both
    # so the script is not pinned to one side of that rename.
    try:
        inputs = processor(text=options, audio=[audio],
                           sampling_rate=SAMPLE_RATE, return_tensors="pt",
                           padding=True)
    except (TypeError, ValueError):
        inputs = processor(text=options, audios=[audio],
                           sampling_rate=SAMPLE_RATE, return_tensors="pt",
                           padding=True)
    inputs = {k: v.to(device) if hasattr(v, "to") else v
              for k, v in inputs.items()}

    with torch.no_grad():
        out = clap(**inputs)

    scores = out.logits_per_audio[0].float().cpu().tolist()
    return [round(s, 3) for s in scores]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--queries", type=Path, default=MUSE.DEFAULT_QUERIES)
    ap.add_argument("--runs", type=Path, nargs="+",
                    default=[ROOT / n for n in DEFAULT_RUNS])
    ap.add_argument("--frames", type=Path, default=MUSE.DEFAULT_FRAMES)
    ap.add_argument("--videos", type=Path,
                    default=ROOT / "videos_TEST_VQA" / "videos")
    ap.add_argument("--manifest", type=Path, default=ROOT / "debate.manifest.json")
    ap.add_argument("--evidence", type=Path, default=ROOT / "evidence.2026.json")
    ap.add_argument("--frames-normal", type=int, default=FRAMES_NORMAL)
    ap.add_argument("--frames-counting", type=int, default=FRAMES_COUNTING)
    ap.add_argument("--audio", action="store_true",
                    help="score sound questions with CLAP (needs the GPU)")
    ap.add_argument("--clap-model", default=CLAP_MODEL)
    ap.add_argument("--limit", type=int, help="only the first N audio queries")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    queries = MUSE.load_queries(args.queries)
    labels = [JUDGE.label_for(p) for p in args.runs]
    panel = JUDGE.load_panel(args.runs, queries)

    covered = [q for q in queries if all(panel[m].get(q["Q_ID"]) for m in labels)]
    disputed = [q for q in covered
                if len(JUDGE.candidates_for(panel, labels, q["Q_ID"])) > 1]

    index_path = args.frames / "index.json"
    frame_index = MUSE.load_json(index_path)["videos"] if index_path.exists() else {}

    counting = {q["Q_ID"] for q in disputed if COUNTING.search(q["Question"])}

    # CLAP compares audio against *descriptive* text. "How many voices can be
    # heard" matches the sound pattern, but its options are bare numerals, and
    # scoring those against a soundtrack measures nothing -- measured, it ranks
    # "1" top almost every time, which would push the agents toward exactly the
    # undercount the counting aid exists to correct. Descriptive options only.
    def scorable(query: dict) -> bool:
        if query["Q_ID"] in counting:
            return False
        numeric = sum(1 for o in query["Options"] if NUMERIC.match(o))
        return numeric * 2 < len(query["Options"])

    sound = [q for q in disputed if SOUND.search(q["Question"]) and scorable(q)]

    print(f"panel: {', '.join(labels)}")
    print(f"disputed: {len(disputed)} of {len(covered)} | counting "
          f"{len(counting)} | sound {len(sound)}")

    videos: dict[str, dict] = {}
    no_frames = 0

    for query in disputed:
        stem = Path(query["Video_ID"]).stem
        entry = frame_index.get(stem)

        if not entry or not entry.get("duration_sec"):
            no_frames += 1
            continue

        # ffmpeg samples at n/duration fps, so asking a 1 s clip for 64 frames
        # needs 64 fps and it silently returns fewer -- then the manifest
        # promises references that 404 on the agents. Cap at 8 fps of source.
        duration = float(entry["duration_sec"])
        wanted = (args.frames_counting if query["Q_ID"] in counting
                  else args.frames_normal)
        wanted = max(8, min(wanted, int(duration * 8)))
        current = videos.get(stem)

        # One video can serve several queries; cut the denser strip once.
        if not current or wanted > current["n_frames"]:
            videos[stem] = {"n_frames": wanted,
                            "duration_sec": round(float(entry["duration_sec"]), 3)}

    manifest = {
        "generated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "note": "cut on the Colab side from the mounted videos; 448x448, "
                "mid-point sampling, <stem>/001.jpg ...",
        "size": 448,
        "n_videos": len(videos),
        "videos": videos,
    }

    with args.manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
        handle.write("\n")

    dense = sum(1 for v in videos.values() if v["n_frames"] > args.frames_normal)
    print(f"\nmanifest: {len(videos)} videos "
          f"({dense} at {args.frames_counting} frames for counting, the rest at "
          f"{args.frames_normal}); {no_frames} disputed queries have no usable "
          f"video -> they stay at the baseline")
    print(f"  wrote {args.manifest}  ({args.manifest.stat().st_size / 1024:.0f} KB)"
          f"  <- upload this one file to Drive")

    evidence = {str(q["Q_ID"]): {"counting": True} for q in disputed
                if q["Q_ID"] in counting}

    if args.audio:
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            print(f"\n  audio skipped: {exc}", file=sys.stderr)
            args.audio = False

    if args.audio:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nloading {args.clap_model} on {device} ...")
        processor = ClapProcessor.from_pretrained(args.clap_model)
        clap = ClapModel.from_pretrained(args.clap_model).to(device).eval()

        targets = sound[:args.limit] if args.limit else sound
        scored = failed = 0

        for n, query in enumerate(targets, 1):
            path = video_path(args.videos, query["Video_ID"])

            if not path:
                failed += 1
                continue

            audio = read_audio(path)

            if audio is None or len(audio) < SAMPLE_RATE // 2:
                failed += 1
                continue

            try:
                scores = score_options(clap, processor, audio,
                                       query["Options"], device)
            except Exception as exc:  # noqa: BLE001 - one clip must not stop it
                print(f"  Q{query['Q_ID']}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                failed += 1
                continue

            entry = evidence.setdefault(str(query["Q_ID"]), {})
            entry["audio_scores"] = scores
            scored += 1

            if n % 25 == 0:
                print(f"  {n}/{len(targets)} audio queries scored")

        print(f"\naudio: scored {scored} of {len(targets)} sound queries "
              f"({failed} had no usable audio)")

    with args.evidence.open("w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=1)
        handle.write("\n")

    print(f"  wrote {args.evidence}  ({len(evidence)} queries carry evidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

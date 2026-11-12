#!/usr/bin/env python3
"""Sample frames from every Task 1 video, ready to hand to a vision model.

Gemma 4 has no video input primitive: a video is a sequence of timestamped
frames through the same vision tower as still images, and the model card caps
video at "a maximum of 60 seconds assuming the images are processed at one
frame per second". 189 of these 1,940 videos are longer than that, the longest
running 389 s, so frames are sampled *uniformly across the whole clip* rather
than at a fixed 1 fps -- below 1 fps for the long tail, which is the remedy the
60 s-at-1 fps framing implies. The sampling instants are written to the index so
the answering pass can label each frame with its mm:ss timestamp and the model
can still reason about temporal order.

Frame count per video::

    cap = 32 if duration > 60 else 16
    n   = clamp(round(duration), 8, cap)

which is ~1 fps for the median 10 s clip, 0.27 fps at 60 s and 0.08 fps at 389 s.

Frames are extracted once per video and reused by every query that asks about
it and by every prompt phrasing of those queries -- 26k extractions instead of
366k. Output::

    videos_TEST_VQA/frames/<stem>/000.jpg ...
    videos_TEST_VQA/frames/index.json

Durations are read from ``duration.report.json`` rather than re-probed. Run
``VQA/duration.report.py`` first if that file is missing.

Usage:
    python VQA/extract.frames.py --limit 20          # smoke test
    python VQA/extract.frames.py                     # everything, resumable
    python VQA/extract.frames.py --only=-5tDzjnjTV0  # note the '=' form
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = ROOT / "vqa.2026.testingDataset.Task1.json"
DEFAULT_VIDEOS = ROOT / "videos_TEST_VQA" / "videos"
DEFAULT_DURATIONS = ROOT / "duration.report.json"
DEFAULT_OUT = ROOT / "videos_TEST_VQA" / "frames"

DEFAULT_WORKERS = 6

# Long edge in pixels. Gemma 4's vision tower works on a fixed grid, so more
# than this is discarded detail paid for in decode time.
DEFAULT_LONG_EDGE = 896
DEFAULT_QUALITY = 3

MIN_FRAMES = 8
CAP_SHORT = 16          # videos up to LONG_VIDEO_SEC
CAP_LONG = 32           # videos beyond it -- thin coverage is worth more frames
LONG_VIDEO_SEC = 60.0   # Gemma 4's documented video ceiling


def frame_count(duration: float, cap_short: int, cap_long: int,
                minimum: int) -> int:
    """Frames to sample from a clip of this length."""
    cap = cap_long if duration > LONG_VIDEO_SEC else cap_short
    return max(minimum, min(cap, round(duration)))


def timestamps(duration: float, n: int) -> list[float]:
    """Mid-point sampling instants, so no frame lands on a black first frame."""
    return [round((k + 0.5) * duration / n, 3) for k in range(n)]


def extract(video: Path, out_dir: Path, duration: float, n: int,
            long_edge: int, quality: int, timeout: float) -> dict:
    """Write exactly `n` JPEGs sampled evenly across the clip."""
    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True)

    # fps=n/duration lands frames on k*duration/n; start_time shifts them to the
    # mid-points computed above. One ffmpeg pass per video, not one per frame.
    scale = (f"scale='if(gt(iw,ih),{long_edge},-2)'"
             f":'if(gt(iw,ih),-2,{long_edge})'")
    offset = duration / (2 * n)

    cmd = [
        "ffmpeg", "-v", "error", "-an", "-sn",
        "-i", str(video),
        "-vf", f"fps={n}/{duration}:start_time={offset:.4f},{scale}",
        "-q:v", str(quality),
        "-frames:v", str(n),
        str(out_dir / "%03d.jpg"), "-y",
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       check=True)
    except subprocess.CalledProcessError as exc:
        return {"error": (exc.stderr or "ffmpeg failed").strip()[:200]}
    except subprocess.TimeoutExpired:
        return {"error": "ffmpeg timed out"}

    files = sorted(p.name for p in out_dir.glob("*.jpg"))

    if not files:
        return {"error": "ffmpeg produced no frames"}

    # A clip shorter than the sampling window can come up short; that is fine,
    # but the timestamps must then describe the frames that actually exist.
    return {
        "n_frames": len(files),
        "files": files,
        "timestamps": timestamps(duration, n)[:len(files)],
        "bytes": sum((out_dir / f).stat().st_size for f in files),
    }


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def video_path(videos_dir: Path, video_id: str) -> Path:
    """Resolve a Video_ID to a file, per download_test_videos.py's convention."""
    name = video_id if video_id.endswith(".webm") else video_id + ".mp4"
    return videos_dir / name


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Sample frames from the Task 1 videos for a vision model."
    )

    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES,
                    help="released query JSON (default: %(default)s)")
    ap.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS,
                    help="directory of video files (default: %(default)s)")
    ap.add_argument("--durations", type=Path, default=DEFAULT_DURATIONS,
                    help="duration.report.py output (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="frame directory (default: %(default)s)")

    ap.add_argument("--long-edge", type=int, default=DEFAULT_LONG_EDGE,
                    help="scale the long edge to N px (default: %(default)s)")
    ap.add_argument("--quality", type=int, default=DEFAULT_QUALITY,
                    help="ffmpeg JPEG quality, 2 best (default: %(default)s)")
    ap.add_argument("--max-frames", type=int, default=CAP_SHORT,
                    help="frame cap for clips under 60 s (default: %(default)s)")
    ap.add_argument("--max-frames-long", type=int, default=CAP_LONG,
                    help="frame cap beyond 60 s (default: %(default)s)")
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES,
                    help="frame floor for very short clips (default: %(default)s)")

    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="concurrent ffmpeg calls (default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-video ffmpeg timeout (default: %(default)s)")

    ap.add_argument("--limit", type=int, help="extract at most N videos")
    ap.add_argument("--only", action="append", metavar="VIDEO_ID",
                    help="extract only these Video_IDs; repeat the flag. Use "
                         "--only=<ID> for the 12 IDs that start with '-'")
    ap.add_argument("--force", action="store_true",
                    help="re-extract videos that already have frames")
    ap.add_argument("--reindex", action="store_true",
                    help="rebuild index.json from what is on disk, no ffmpeg")

    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not shutil.which("ffmpeg"):
        print("ffmpeg is not on PATH", file=sys.stderr)
        return 1

    queries = load_json(args.queries)

    if not args.durations.exists():
        print(f"no duration report at {args.durations} -- run "
              f"VQA/duration.report.py first", file=sys.stderr)
        return 1

    durations = {record["video_id"]: record["duration_sec"]
                 for record in load_json(args.durations)["videos"]
                 if "duration_sec" in record}

    # One entry per distinct video, carrying the Q_IDs that need it.
    videos: dict[str, dict] = {}

    for query in queries:
        video_id = query["Video_ID"]
        entry = videos.setdefault(video_id, {"video_id": video_id, "q_ids": []})
        entry["q_ids"].append(query["Q_ID"])

    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.json"

    index: dict[str, dict] = {}

    if index_path.exists():
        index = load_json(index_path).get("videos", {})

    if args.reindex:
        rebuilt: dict[str, dict] = {}

        for video_id in videos:
            stem = Path(video_id).stem
            frame_dir = args.out / stem
            files = sorted(p.name for p in frame_dir.glob("*.jpg")) \
                if frame_dir.is_dir() else []

            if not files:
                continue

            duration = durations.get(video_id)
            rebuilt[stem] = {
                "video_id": video_id,
                "duration_sec": duration,
                "n_frames": len(files),
                "files": files,
                "timestamps": timestamps(duration, len(files))
                              if duration else [],
            }

        write_index(index_path, rebuilt, args)
        print(f"reindexed {len(rebuilt)} videos into {index_path}")
        return 0

    missing_file = 0
    missing_duration = 0
    todo: list[dict] = []

    for video_id, entry in videos.items():
        stem = Path(video_id).stem
        path = video_path(args.videos, video_id)

        if not path.exists():
            missing_file += 1
            continue

        duration = durations.get(video_id)

        if not duration:
            missing_duration += 1
            continue

        entry.update(stem=stem, path=path, duration=duration,
                     n=frame_count(duration, args.max_frames,
                                   args.max_frames_long, args.min_frames))

        cached = index.get(stem)

        if not args.force and cached and cached.get("n_frames") \
                and (args.out / stem).is_dir() \
                and len(list((args.out / stem).glob("*.jpg"))) \
                == cached["n_frames"]:
            continue

        todo.append(entry)

    usable = len(videos) - missing_file - missing_duration
    already = usable - len(todo)

    if args.only:
        wanted = set(args.only)
        todo = [e for e in todo if e["video_id"] in wanted
                or e["stem"] in wanted]

    if args.limit:
        todo = todo[:args.limit]

    print(f"videos referenced: {len(videos)} | on disk with a duration: "
          f"{usable} | already extracted: {already}")

    if missing_file:
        print(f"  {missing_file} videos are not on disk -- those queries will "
              f"be answered without frames")

    if missing_duration:
        print(f"  {missing_duration} videos have no duration in the report")

    print(f"to extract: {len(todo)}  "
          f"({sum(e['n'] for e in todo)} frames)")

    if not todo:
        write_index(index_path, index, args)
        return 0

    lock = threading.Lock()
    done = failed = 0
    total_bytes = 0
    started = time.monotonic()

    def work(entry: dict) -> None:
        nonlocal done, failed, total_bytes

        result = extract(entry["path"], args.out / entry["stem"],
                         entry["duration"], entry["n"], args.long_edge,
                         args.quality, args.timeout)

        with lock:
            if "error" in result:
                failed += 1
                print(f"  FAILED {entry['video_id']}: {result['error']}",
                      file=sys.stderr)
                index.pop(entry["stem"], None)
                return

            index[entry["stem"]] = {
                "video_id": entry["video_id"],
                "duration_sec": entry["duration"],
                "n_frames": result["n_frames"],
                "files": result["files"],
                "timestamps": result["timestamps"],
            }

            done += 1
            total_bytes += result["bytes"]

            if done % 100 == 0 or done + failed == len(todo):
                rate = done / max(time.monotonic() - started, 1e-9)
                print(f"  {done}/{len(todo)}  {rate:.1f} videos/s  "
                      f"{total_bytes / 1e6:.0f} MB")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, todo))
    except KeyboardInterrupt:
        print("\ninterrupted -- finished videos are indexed, re-run to resume",
              file=sys.stderr)
        write_index(index_path, index, args)
        return 130

    write_index(index_path, index, args)

    elapsed = time.monotonic() - started
    frames = sum(v["n_frames"] for v in index.values())

    print(f"\nextracted {done} videos ({failed} failed) in {elapsed / 60:.1f} min")
    print(f"index now holds {len(index)} videos / {frames} frames")
    print(f"wrote {index_path}")

    return 0


def write_index(path: Path, index: dict[str, dict],
                args: argparse.Namespace) -> None:
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "long_edge": args.long_edge,
        "quality": args.quality,
        "max_frames": args.max_frames,
        "max_frames_long": args.max_frames_long,
        "min_frames": args.min_frames,
        "long_video_sec": LONG_VIDEO_SEC,
        "n_videos": len(index),
        "n_frames": sum(v["n_frames"] for v in index.values()),
        "videos": dict(sorted(index.items())),
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    sys.exit(main())

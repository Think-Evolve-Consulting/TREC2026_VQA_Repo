#!/usr/bin/env python3
"""Measure how long the Task 1 videos are and plot the distribution.

Probes every video in ``videos_TEST_VQA/videos`` with ffprobe, joins the result
against the released query set, and writes:

    duration.report.json          per-video durations plus corpus statistics
    duration.report.html          a self-contained Plotly page

Durations come from the container (``format.duration``), falling back to the
video stream when a container does not carry one. Probing is cached, so the
chart can be restyled with ``--rebuild`` without re-reading 1,940 files.

Usage:
    python VQA/duration.report.py                  # probe, then plot
    python VQA/duration.report.py --rebuild        # replot from the cache
    python VQA/duration.report.py --limit 50       # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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
DEFAULT_REPORT = ROOT / "duration.report.json"
DEFAULT_HTML = ROOT / "duration.report.html"

VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v"}
DEFAULT_WORKERS = 8

# The two halves of the corpus have different provenance and, as it turns out,
# very different lengths: .webm are the released clips, .mp4 the YouTube pulls.
SOURCE_WEBM = "Released clips (.webm)"
SOURCE_MP4 = "YouTube downloads (.mp4)"


def probe(path: Path, timeout: float) -> dict:
    """Return duration and shape for one video, or an error record."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,width,height,duration",
        "-of", "json", str(path),
    ]

    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=True).stdout
    except subprocess.CalledProcessError as exc:
        return {"error": (exc.stderr or "ffprobe failed").strip()[:200]}
    except subprocess.TimeoutExpired:
        return {"error": "ffprobe timed out"}

    data = json.loads(out)
    fmt = data.get("format", {})
    streams = data.get("streams", [])

    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    # Matroska/WebM does not always carry a container duration; the video
    # stream's own duration is the fallback before giving up on the file.
    duration = fmt.get("duration") or video.get("duration")

    try:
        duration = float(duration)
    except (TypeError, ValueError):
        return {"error": "no duration in container or video stream"}

    if duration <= 0:
        return {"error": f"non-positive duration ({duration})"}

    return {
        "duration_sec": round(duration, 3),
        "size_bytes": int(fmt.get("size") or 0),
        "width": video.get("width"),
        "height": video.get("height"),
        "vcodec": video.get("codec_name"),
        "has_audio": has_audio,
    }


def collect(videos_dir: Path, queries: list[dict], workers: int,
            timeout: float, limit: int | None) -> list[dict]:
    """Probe every video on disk, tagged with how many queries use it."""
    per_video: dict[str, int] = {}

    for query in queries:
        per_video[query["Video_ID"]] = per_video.get(query["Video_ID"], 0) + 1

    paths = sorted(p for p in videos_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in VIDEO_EXTS)

    if limit:
        paths = paths[:limit]

    print(f"probing {len(paths)} videos with {workers} workers")

    records: list[dict] = []
    lock = threading.Lock()
    started = time.monotonic()

    def work(path: Path) -> None:
        result = probe(path, timeout)

        # The webm clips are referenced with their extension, the YouTube pulls
        # by bare ID -- match download_test_videos.py's naming convention.
        video_id = path.name if path.suffix.lower() == ".webm" else path.stem

        record = {
            "video_id": video_id,
            "file": path.name,
            "source": SOURCE_WEBM if path.suffix.lower() == ".webm"
                      else SOURCE_MP4,
            "n_queries": per_video.get(video_id, 0),
            **result,
        }

        with lock:
            records.append(record)
            n = len(records)

            if n % 200 == 0 or n == len(paths):
                rate = n / max(time.monotonic() - started, 1e-9)
                print(f"  {n}/{len(paths)}  {rate:.0f}/s")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, paths))

    records.sort(key=lambda r: r["file"])
    return records


def describe(values: list[float]) -> dict:
    """Summary statistics for a list of durations."""
    if not values:
        return {}

    ordered = sorted(values)

    def pct(p: float) -> float:
        # Nearest-rank; the corpus is large enough that interpolation is noise.
        index = min(int(p / 100 * len(ordered)), len(ordered) - 1)
        return round(ordered[index], 2)

    return {
        "count": len(ordered),
        "total_hours": round(sum(ordered) / 3600, 2),
        "mean_sec": round(statistics.fmean(ordered), 2),
        "median_sec": round(statistics.median(ordered), 2),
        "min_sec": round(ordered[0], 2),
        "max_sec": round(ordered[-1], 2),
        "p05_sec": pct(5),
        "p25_sec": pct(25),
        "p75_sec": pct(75),
        "p90_sec": pct(90),
        "p95_sec": pct(95),
        "p99_sec": pct(99),
    }


def summarise(records: list[dict], queries: list[dict]) -> dict:
    """Corpus-level statistics, overall and split by source."""
    good = [r for r in records if "duration_sec" in r]
    failed = [r for r in records if "duration_sec" not in r]

    by_source: dict[str, dict] = {}

    for source in (SOURCE_WEBM, SOURCE_MP4):
        values = [r["duration_sec"] for r in good if r["source"] == source]

        if values:
            by_source[source] = describe(values)

    # Every query needs its video watched, so query-weighted time is the number
    # that actually predicts how long an inference pass takes.
    query_seconds = sum(r["duration_sec"] * r["n_queries"] for r in good)
    matched = sum(1 for r in good if r["n_queries"])

    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "videos_probed": len(records),
        "videos_with_duration": len(good),
        "videos_failed": len(failed),
        "failures": [{"file": r["file"], "error": r["error"]} for r in failed],
        "videos_used_by_queries": matched,
        "videos_unused": len(good) - matched,
        "queries_total": len(queries),
        "queries_covered": sum(r["n_queries"] for r in good),
        "query_weighted_hours": round(query_seconds / 3600, 2),
        "overall": describe([r["duration_sec"] for r in good]),
        "by_source": by_source,
    }


def write_long_videos(report: dict, queries: list[dict], threshold: float,
                      path: Path) -> int:
    """CSV of everything above `threshold`, longest first. Returns the count."""
    q_ids: dict[str, list[int]] = {}

    for query in queries:
        q_ids.setdefault(query["Video_ID"], []).append(query["Q_ID"])

    rows = [r for r in report["videos"]
            if r.get("duration_sec", 0) > threshold]
    rows.sort(key=lambda r: -r["duration_sec"])

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "file", "source", "duration_sec",
                         "duration_mmss", "n_queries", "q_ids"])

        for record in rows:
            seconds = record["duration_sec"]
            ids = q_ids.get(record["video_id"], [])

            writer.writerow([
                record["video_id"],
                record["file"],
                record["source"],
                f"{seconds:.2f}",
                f"{int(seconds) // 60}:{int(seconds) % 60:02d}",
                record["n_queries"],
                " ".join(str(i) for i in ids),
            ])

    return len(rows)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Measure and plot the Task 1 video duration distribution."
    )

    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES,
                    help="released query JSON (default: %(default)s)")
    ap.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS,
                    help="directory of video files (default: %(default)s)")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT,
                    help="JSON report path (default: %(default)s)")
    ap.add_argument("--html", type=Path, default=DEFAULT_HTML,
                    help="Plotly output path (default: %(default)s)")

    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="concurrent ffprobe calls (default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="per-file ffprobe timeout (default: %(default)s)")
    ap.add_argument("--longer-than", type=float, metavar="SECONDS",
                    help="also write a CSV of the videos above this duration, "
                         "longest first, with the Q_IDs that ask about them")
    ap.add_argument("--longer-than-csv", type=Path,
                    help="where that CSV goes "
                         "(default: videos.over.<SECONDS>s.csv beside the report)")
    ap.add_argument("--limit", type=int, help="probe at most N videos")
    ap.add_argument("--rebuild", action="store_true",
                    help="replot from the existing report, no probing")
    ap.add_argument("--no-html", action="store_true",
                    help="write the JSON report only")

    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    with args.queries.open("r", encoding="utf-8-sig") as handle:
        queries = json.load(handle)

    if args.rebuild:
        if not args.report.exists():
            print(f"no report at {args.report} -- run without --rebuild first",
                  file=sys.stderr)
            return 1

        with args.report.open("r", encoding="utf-8") as handle:
            report = json.load(handle)

        records = report["videos"]
        print(f"loaded {len(records)} probed videos from {args.report}")
    else:
        if not args.videos.is_dir():
            print(f"no video directory at {args.videos}", file=sys.stderr)
            return 1

        started = time.monotonic()
        records = collect(args.videos, queries, args.workers, args.timeout,
                          args.limit)
        print(f"probed in {time.monotonic() - started:.1f}s")

        report = {**summarise(records, queries), "videos": records}

        with args.report.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=4, ensure_ascii=False)
            handle.write("\n")

        print(f"wrote {args.report}")

    stats = report["overall"]
    print(f"\nvideos with a duration: {report['videos_with_duration']}"
          f"  (failed: {report['videos_failed']})")
    print(f"total footage:          {stats['total_hours']} h")
    print(f"query-weighted footage: {report['query_weighted_hours']} h")
    print(f"median / mean:          {stats['median_sec']}s / {stats['mean_sec']}s")
    print(f"range:                  {stats['min_sec']}s .. {stats['max_sec']}s")
    print(f"p95 / p99:              {stats['p95_sec']}s / {stats['p99_sec']}s")

    for source, block in report["by_source"].items():
        print(f"  {source}: n={block['count']} median={block['median_sec']}s "
              f"max={block['max_sec']}s")

    if args.longer_than is not None:
        csv_path = args.longer_than_csv or (
            args.report.parent / f"videos.over.{args.longer_than:g}s.csv"
        )
        n_long = write_long_videos(report, queries, args.longer_than, csv_path)

        long_seconds = sum(r["duration_sec"] for r in report["videos"]
                           if r.get("duration_sec", 0) > args.longer_than)
        long_queries = sum(r["n_queries"] for r in report["videos"]
                           if r.get("duration_sec", 0) > args.longer_than)

        print(f"\nlonger than {args.longer_than:g}s: {n_long} videos, "
              f"{long_seconds / 60:.1f} min of footage, "
              f"{long_queries} queries")
        print(f"wrote {csv_path}")

    if not args.no_html:
        from duration_report_chart import write_html

        write_html(report, args.html)
        print(f"\nwrote {args.html}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

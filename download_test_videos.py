"""
Download the YouTube videos referenced by the TREC 2026 VQA Task 1 testing
queries into the released-video folder, using the same naming convention as
the query file so the submission pipeline can resolve a Video_ID directly to
a file on disk.

Input:  vqa.2026.testingDataset.Task1.json — entries of the form
            {"Q_ID": 3, "Video_ID": "PeGejCJ2CE0", "Question": "..."}
        Video_ID is either a released ".webm" clip (already on disk, skipped
        here) or an 11-character YouTube video ID (downloaded here).

        A .csv query file is also accepted, for the 2025 sets that ship as
        Q_ID,Video_ID,Question[,Answer] rows (e.g. vqa.gt.2025.csv) — with or
        without a header line. Everything downstream is identical.

Naming: videos are written as <output-dir>/videos/<Video_ID>.mp4, so lookup
        from a query is exactly:

            name = Video_ID if Video_ID.endswith(".webm") else Video_ID + ".mp4"
            path = videos_dir / name

        The YouTube ID is used verbatim as the file stem — no sanitising, no
        title-based names — which keeps it byte-identical to the Video_ID that
        must appear in the run submission.

Output layout (matches batch_download.py):
    <output-dir>/videos/<Video_ID>.mp4
    <output-dir>/metadata/<Video_ID>.json      (+ the queries asked about it)
    <output-dir>/download_summary.task1.json

Download, codec normalization (AV1/VP9/VP8 -> H.264) and metadata writing are
reused from batch_download.py; the retry-with-backoff loop follows
retry_downloads.py.

Usage:
    python download_test_videos.py
    python download_test_videos.py --cookies-from-browser chrome
    python download_test_videos.py --quality 720p --sleep 3
    python download_test_videos.py --limit 5                  # smoke test
    python download_test_videos.py --retry-failed videos_TEST_VQA/download_summary.task1.json
    python download_test_videos.py --queries vqa.gt.2025.csv --output-dir videos_GT_2025 \
                                   --summary-name download_summary.gt2025.json
"""

import argparse
import csv
import json
import random
import re
import signal
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yt_dlp

# Reuse the download / codec / metadata helpers (same directory).
import batch_download
from batch_download import download_one


DEFAULT_QUERIES = "vqa.2026.testingDataset.Task1.json"
DEFAULT_OUTPUT_DIR = "videos_TEST_VQA"
DEFAULT_REMOVED_LIST = "Videos Removed from the platform.txt"
SUMMARY_NAME = "download_summary.task1.json"

YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Extensions that count as "this video is already on disk".
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".m4v", ".mov")


class Interrupted(Exception):
    """Raised on Ctrl-C so the partial summary still gets written."""


class FatalConfigError(Exception):
    """A failure that will hit every video — stop instead of grinding on."""


# Errors that mean the cookie/browser setup is broken, not the video. Retrying
# these 868 times accomplishes nothing but burning an afternoon.
FATAL_ERROR_MARKERS = (
    "could not copy chrome cookie database",
    "could not copy cookie database",
    "could not find chrome cookies database",
    "unsupported browser",
    "failed to decrypt with dpapi",
    "no such file or directory: 'cookies",
)


# ---------------------------------------------------------------------------
# Query-file parsing
# ---------------------------------------------------------------------------

def load_queries_csv(path: Path):
    """
    Read a Q_ID,Video_ID,Question[,Answer] CSV (vqa.gt.2025.csv and friends)
    into the same dict shape the JSON query files use.

    The 2025 files ship headerless, so the header is only consumed when the
    first row does not already look like data — i.e. when column 2 is not an
    11-character YouTube ID.
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]

    if not rows:
        return []

    if len(rows[0]) < 2 or not YOUTUBE_ID_RE.match(rows[0][1].strip()):
        rows = rows[1:]  # header line

    queries = []
    for row in rows:
        if len(row) < 2:
            continue
        entry = {
            "Q_ID": row[0].strip(),
            "Video_ID": row[1].strip(),
            "Question": row[2].strip() if len(row) > 2 else None,
        }
        if len(row) > 3 and row[3].strip():
            entry["Answer"] = row[3].strip()
        queries.append(entry)

    return queries


def load_queries(path: Path):
    if path.suffix.lower() == ".csv":
        return load_queries_csv(path)

    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def collect_youtube_targets(queries):
    """
    Return an ordered {video_id: [{"Q_ID": .., "Question": ..}, ...]} mapping
    for every non-webm Video_ID, preserving first-seen order.
    """
    targets = {}

    for entry in queries:
        video_id = entry.get("Video_ID")
        if not isinstance(video_id, str) or not video_id.strip():
            continue
        if video_id.lower().endswith(".webm"):
            continue  # released clip, shipped in videos.zip

        query = {"Q_ID": entry.get("Q_ID"), "Question": entry.get("Question")}
        if entry.get("Answer") is not None:
            query["Answer"] = entry["Answer"]

        targets.setdefault(video_id, []).append(query)

    return targets


def load_removed_ids(path: Path):
    """IDs listed in 'Videos Removed from the platform.txt' (<id>.json lines)."""
    if not path or not path.is_file():
        return set()

    removed = set()
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            name = line.strip()
            if name.endswith(".json"):
                removed.add(name[: -len(".json")])
    return removed


def load_previous_failures(path: Path):
    """Video IDs recorded as failed in an earlier summary file."""
    if not path.is_file():
        print(f"❌ Summary file not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    failed = summary.get("results", {}).get("failed", [])
    return [e["video_id"] for e in failed if e.get("video_id")]


def find_existing(videos_dir: Path, video_id: str):
    """Return the on-disk file for this Video_ID, or None."""
    for extension in VIDEO_EXTENSIONS:
        candidate = videos_dir / f"{video_id}{extension}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Cookies from browser
# ---------------------------------------------------------------------------

COOKIE_HELP = """
Cookie troubleshooting:

  1. Try without cookies at all. Most of these videos are public and download
     fine anonymously — only use cookies if you actually hit a bot check:
         python download_test_videos.py --limit 3

  2. Chrome locks its cookie database while it runs. Quit Chrome COMPLETELY
     (every process, including the background tray instance) and retry:
         Get-Process chrome | Stop-Process
     Note that Chrome 127+ also encrypts cookies with App-Bound Encryption,
     which yt-dlp cannot decrypt even with Chrome closed.

  3. Firefox has neither the lock nor App-Bound Encryption. Sign into YouTube
     there once, then:
         python download_test_videos.py --cookies-from-browser firefox

  4. Or export a Netscape-format cookies.txt for youtube.com with a browser
     extension and pass it directly:
         python download_test_videos.py --cookies youtube.cookies.txt
"""

def patch_cookies_from_browser(spec: str):
    """
    Make every yt_dlp.YoutubeDL created inside batch_download use the given
    browser's cookie jar.

    batch_download.download_one only exposes a cookies.txt path, so instead of
    forking it we install a YoutubeDL subclass that injects the
    'cookiesfrombrowser' option. `spec` is the same string yt-dlp's CLI takes,
    e.g. "chrome", "chrome:Profile 16", "firefox:default".
    """
    browser, _, profile = spec.partition(":")
    cookies_tuple = (browser.strip().lower(), profile.strip() or None, None, None)

    base_class = yt_dlp.YoutubeDL

    class BrowserCookieYoutubeDL(base_class):
        def __init__(self, params=None, **kwargs):
            params = dict(params or {})
            params.setdefault("cookiesfrombrowser", cookies_tuple)
            super().__init__(params, **kwargs)

    yt_dlp.YoutubeDL = BrowserCookieYoutubeDL
    batch_download.yt_dlp.YoutubeDL = BrowserCookieYoutubeDL

    return cookies_tuple


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def attach_queries_to_metadata(metadata_dir: Path, video_id: str, queries,
                               key="task1_queries"):
    """
    Add the queries for this video to the metadata file that
    batch_download.save_video_metadata just wrote. Non-fatal on failure.
    """
    metadata_path = metadata_dir / f"{video_id}.json"

    try:
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {"video_id": video_id}

        metadata[key] = queries

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:  # metadata is a nicety; never fail a download on it
        print(f"  ⚠️  Could not attach queries to metadata: {e}")


# ---------------------------------------------------------------------------
# Download with retries (backoff loop from retry_downloads.py)
# ---------------------------------------------------------------------------

def download_with_retries(video_id, videos_dir, metadata_dir, quality,
                          cookies_file, max_retries, base_delay):
    """Returns (ok: bool, message: str)."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    delay = base_delay
    last_msg = ""

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            print(f"  🔁 Attempt {attempt}/{max_retries}…")

        try:
            ok, msg = download_one(
                video_url=video_url,
                video_id=video_id,
                videos_dir=videos_dir,
                metadata_dir=metadata_dir,
                quality=quality,
                cookies_file=cookies_file,
                source_json_data=None,
            )
        except KeyboardInterrupt:
            raise Interrupted()
        except Exception as e:
            ok, msg = False, str(e)

        if ok:
            return True, msg

        last_msg = msg
        lowered = msg.lower()

        # Setup problems affect every video; fail the whole run immediately.
        if any(marker in lowered for marker in FATAL_ERROR_MARKERS):
            raise FatalConfigError(msg)

        print(f"  ⚠️  Attempt {attempt} failed: {msg[:160]}")

        # Permanent failures — retrying only wastes time and rate limit.
        permanent = (
            "video unavailable",
            "private video",
            "removed by the uploader",
            "account associated with this video has been terminated",
            "this video has been removed",
            "video is not available",
            "members-only",
        )
        if any(p in lowered for p in permanent):
            return False, f"{msg} [permanent; not retried]"

        if attempt < max_retries:
            print(f"  ⏳ Waiting {delay:.0f}s before next attempt…")
            time.sleep(delay)
            delay = min(delay * 2, 120)  # cap at 2 minutes

    return False, last_msg


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def write_summary(summary_path: Path, queries_path, output_dir, quality,
                  total_targets, results, max_retries, base_delay,
                  interrupted):
    summary = {
        "timestamp": datetime.now().isoformat(),
        "query_file": str(queries_path),
        "output_dir": str(output_dir),
        "quality": quality,
        "max_retries": max_retries,
        "base_delay_seconds": base_delay,
        "total_youtube_targets": total_targets,
        "interrupted": interrupted,
        "counts": {
            "success": len(results["success"]),
            "failed": len(results["failed"]),
            "skipped": len(results["skipped"]),
        },
        "results": results,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary_path


def run(queries_path, output_dir, quality, cookies_file, cookies_from_browser,
        max_retries, base_delay, sleep_between, limit, only_ids,
        retry_failed_summary, removed_list, skip_removed, temp_dir,
        summary_name=SUMMARY_NAME):

    output_dir = Path(output_dir)
    videos_dir = output_dir / "videos"
    metadata_dir = output_dir / "metadata"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Stage downloads on the same drive as the destination so the final move is
    # a rename rather than a cross-device copy of a multi-hundred-MB file.
    if temp_dir:
        staging = Path(temp_dir)
        staging.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(staging)

    queries = load_queries(Path(queries_path))
    targets = collect_youtube_targets(queries)

    # The CSV sets are not Task 1, so don't label their metadata as if they were.
    metadata_key = ("queries" if Path(queries_path).suffix.lower() == ".csv"
                    else "task1_queries")

    if only_ids:
        wanted = [v.strip() for v in only_ids.split(",") if v.strip()]
        targets = {v: targets.get(v, []) for v in wanted}
    elif retry_failed_summary:
        wanted = load_previous_failures(Path(retry_failed_summary))
        targets = {v: targets.get(v, []) for v in wanted}
        print(f"🔁 Retrying {len(targets)} previously failed video(s)")

    removed_ids = load_removed_ids(Path(removed_list)) if removed_list else set()

    total_targets = len(targets)

    print("🚀 TREC VQA — test video downloader")
    print(f"   Query file:   {queries_path}")
    print(f"   Videos   →    {videos_dir}")
    print(f"   Metadata →    {metadata_dir}")
    print(f"   Quality:      {quality}")
    print(f"   Cookies:      {cookies_from_browser or cookies_file or 'None'}")
    print(f"   Retries:      {max_retries} (base delay {base_delay}s)")
    print(f"   Sleep:        {sleep_between}s between videos")
    print(f"   YouTube Video_IDs referenced by queries: {total_targets}")
    print("=" * 60)

    results = {"success": [], "failed": [], "skipped": []}
    pending = []

    # ── Pass 1: decide what actually needs downloading ────────────────────────
    for video_id, video_queries in targets.items():
        if not YOUTUBE_ID_RE.match(video_id):
            results["failed"].append({
                "video_id": video_id,
                "reason": "Video_ID is not an 11-character YouTube ID",
                "q_ids": [q["Q_ID"] for q in video_queries],
            })
            continue

        existing = find_existing(videos_dir, video_id)
        if existing:
            results["skipped"].append({
                "video_id": video_id,
                "reason": f"Already on disk ({existing.name})",
                "q_ids": [q["Q_ID"] for q in video_queries],
            })
            continue

        if skip_removed and video_id in removed_ids:
            results["skipped"].append({
                "video_id": video_id,
                "reason": "Listed as removed from the platform",
                "q_ids": [q["Q_ID"] for q in video_queries],
            })
            continue

        pending.append((video_id, video_queries))

    if limit:
        pending = pending[:limit]

    print(f"⏭️  Already present / skipped: {len(results['skipped'])}")
    print(f"📥 To download:               {len(pending)}")
    print("=" * 60)

    summary_path = output_dir / summary_name
    interrupted = False

    # ── Pass 2: download ──────────────────────────────────────────────────────
    try:
        for i, (video_id, video_queries) in enumerate(pending, 1):
            q_ids = [q["Q_ID"] for q in video_queries]
            print(f"\n[{i}/{len(pending)}] {video_id}  (Q_ID: "
                  f"{', '.join(str(q) for q in q_ids) or 'n/a'})")
            print("-" * 40)

            ok, msg = download_with_retries(
                video_id=video_id,
                videos_dir=videos_dir,
                metadata_dir=metadata_dir,
                quality=quality,
                cookies_file=cookies_file,
                max_retries=max_retries,
                base_delay=base_delay,
            )

            if ok:
                print(f"  ✅ {msg}")
                attach_queries_to_metadata(metadata_dir, video_id, video_queries,
                                           key=metadata_key)
                results["success"].append({
                    "video_id": video_id,
                    "downloaded": f"{video_id}.mp4",
                    "q_ids": q_ids,
                })
            else:
                print(f"  ❌ {msg[:200]}")
                results["failed"].append({
                    "video_id": video_id,
                    "reason": msg,
                    "q_ids": q_ids,
                })

            # Checkpoint so a crash or Ctrl-C never loses the record.
            write_summary(summary_path, queries_path, output_dir, quality,
                          total_targets, results, max_retries, base_delay,
                          interrupted=False)

            if sleep_between and i < len(pending):
                pause = sleep_between * random.uniform(0.75, 1.5)
                time.sleep(pause)

    except (Interrupted, KeyboardInterrupt):
        interrupted = True
        print("\n\n🛑 Interrupted — writing partial summary. "
              "Re-run to resume; finished videos are skipped automatically.")

    except FatalConfigError as e:
        interrupted = True
        print(f"\n\n🛑 Stopping: this failure affects every video, "
              f"not just this one.\n   {e}")
        print(COOKIE_HELP)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✅ Downloaded: {len(results['success'])}")
    print(f"⏭️  Skipped:    {len(results['skipped'])}")
    print(f"❌ Failed:     {len(results['failed'])}")

    if results["failed"]:
        print("\n❌ Failures:")
        for e in results["failed"][:40]:
            print(f"   • {e['video_id']}: {e['reason'][:110]}")
        if len(results["failed"]) > 40:
            print(f"   … and {len(results['failed']) - 40} more")

    still_missing = sum(
        1 for video_id in targets if not find_existing(videos_dir, video_id)
    )
    print(f"\n🎯 YouTube videos still missing locally: "
          f"{still_missing} / {total_targets}")

    write_summary(summary_path, queries_path, output_dir, quality,
                  total_targets, results, max_retries, base_delay, interrupted)
    print(f"📋 Summary report → {summary_path}")

    if results["failed"]:
        print(f"\n🔁 Retry just the failures with:\n"
              f"   python {Path(__file__).name} --retry-failed {summary_path}")

    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download the YouTube videos referenced by the VQA 2026 Task 1 "
            "testing queries, named <Video_ID>.mp4 alongside the released "
            ".webm clips."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The run is resumable: any Video_ID already present in <output-dir>/videos is
skipped, so re-running only fetches what is still missing.

Examples:
    python download_test_videos.py --limit 5
    python download_test_videos.py --cookies-from-browser chrome
    python download_test_videos.py --cookies-from-browser "chrome:Profile 16"
    python download_test_videos.py --quality 720p --sleep 3 --max-retries 5
    python download_test_videos.py --retry-failed videos_TEST_VQA/download_summary.task1.json
    python download_test_videos.py --queries vqa.gt.2025.csv --output-dir videos_GT_2025 \\
                                   --summary-name download_summary.gt2025.json
        """,
    )

    parser.add_argument("--queries", default=DEFAULT_QUERIES,
                        help=("Query file: Task 1 JSON, or a "
                              "Q_ID,Video_ID,Question[,Answer] CSV "
                              f"(default: {DEFAULT_QUERIES})"))
    parser.add_argument("--summary-name", default=SUMMARY_NAME,
                        help=("Filename of the summary written into --output-dir "
                              f"(default: {SUMMARY_NAME})"))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=("Root holding videos/ and metadata/ "
                              f"(default: {DEFAULT_OUTPUT_DIR})"))
    parser.add_argument("--quality", choices=["best", "720p", "480p"], default="best",
                        help="Video quality cap (default: best)")
    parser.add_argument("--cookies", default=None,
                        help="Path to a cookies.txt file for authenticated downloads")
    parser.add_argument("--cookies-from-browser", default=None,
                        help=('Read cookies straight from a browser profile, '
                              'e.g. "chrome" or "chrome:Profile 16"'))
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Attempts per video; backoff doubles, capped at 120s (default: 3)")
    parser.add_argument("--base-delay", type=float, default=5.0,
                        help="Initial retry wait in seconds (default: 5)")
    parser.add_argument("--sleep", type=float, default=2.0, dest="sleep_between",
                        help="Seconds to pause between videos, jittered ±25%% (default: 2)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only download the first N pending videos (smoke test)")
    parser.add_argument("--only", default=None, dest="only_ids",
                        help="Comma-separated Video_IDs to download instead of the full set")
    parser.add_argument("--retry-failed", default=None, dest="retry_failed_summary",
                        help="Re-download only the videos listed as failed in a previous summary")
    parser.add_argument("--removed-list", default=DEFAULT_REMOVED_LIST,
                        help="Text file of <id>.json entries known to be off the platform")
    parser.add_argument("--no-skip-removed", action="store_false", dest="skip_removed",
                        help="Attempt downloads even for IDs on the removed list")
    parser.add_argument("--temp-dir", default=None,
                        help=("Staging directory for in-progress downloads; put it on the "
                              "same drive as --output-dir to avoid cross-drive copies"))

    args = parser.parse_args()

    if args.cookies and not Path(args.cookies).exists():
        print(f"❌ Cookies file not found: {args.cookies}")
        sys.exit(1)

    if not Path(args.queries).exists():
        print(f"❌ Query file not found: {args.queries}")
        sys.exit(1)

    if args.only_ids and args.retry_failed_summary:
        print("❌ Use either --only or --retry-failed, not both.")
        sys.exit(1)

    if args.cookies_from_browser:
        spec = patch_cookies_from_browser(args.cookies_from_browser)
        print(f"🍪 Using browser cookies: {spec[0]}"
              + (f" (profile: {spec[1]})" if spec[1] else ""))

    # Turn Ctrl-C into an exception even while yt_dlp is mid-download.
    def _on_sigint(signum, frame):
        raise Interrupted()

    signal.signal(signal.SIGINT, _on_sigint)

    results = run(
        queries_path=args.queries,
        output_dir=args.output_dir,
        quality=args.quality,
        cookies_file=args.cookies,
        cookies_from_browser=args.cookies_from_browser,
        max_retries=args.max_retries,
        base_delay=args.base_delay,
        sleep_between=args.sleep_between,
        limit=args.limit,
        only_ids=args.only_ids,
        retry_failed_summary=args.retry_failed_summary,
        removed_list=args.removed_list,
        skip_removed=args.skip_removed,
        temp_dir=args.temp_dir,
        summary_name=args.summary_name,
    )

    sys.exit(1 if results["failed"] else 0)


if __name__ == "__main__":
    main()

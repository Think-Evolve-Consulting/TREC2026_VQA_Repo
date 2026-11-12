"""
Retry failed downloads from a download_summary.json produced by batch_download.py.

All failed entries are retried up to --max-retries times with exponential backoff.
Results are written to retry_summary.json in the same output directory.

Usage:
    python retry_downloads.py --summary download_summary.json
    python retry_downloads.py --summary download_summary.json --max-retries 5 --base-delay 10
    python retry_downloads.py --summary download_summary.json --cookies cookies.txt --quality 720p
    python retry_downloads.py --summary download_summary.json --output-dir ./out  # override output dir
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Reuse helpers from batch_download.py (must be in the same directory)
from batch_download import download_one, is_valid_youtube_url, extract_video_id


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

def retry_one(video_url, video_id, videos_dir, metadata_dir,
              source_json_data, quality, cookies_file,
              max_retries, base_delay):
    """
    Try downloading a single video up to max_retries times with exponential backoff.
    Returns (status, message) where status is 'success', 'failed', or 'skipped'.
    """
    delay = base_delay
    last_msg = ""

    for attempt in range(1, max_retries + 1):
        print(f"  🔁 Attempt {attempt}/{max_retries}…")
        try:
            ok, msg = download_one(
                video_url=video_url,
                video_id=video_id,
                videos_dir=videos_dir,
                metadata_dir=metadata_dir,
                quality=quality,
                cookies_file=cookies_file,
                source_json_data=source_json_data,
            )
        except Exception as e:
            ok, msg = False, str(e)

        if ok:
            return "success", msg

        last_msg = msg
        print(f"  ⚠️  Attempt {attempt} failed: {msg}")

        if attempt < max_retries:
            print(f"  ⏳ Waiting {delay}s before next attempt…")
            time.sleep(delay)
            delay = min(delay * 2, 120)  # cap at 2 minutes

    return "failed", last_msg


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_retries(summary_path: Path, output_dir_override, cookies_file,
                quality, max_retries, base_delay):

    # ── Load summary ──────────────────────────────────────────────────────────
    if not summary_path.exists():
        print(f"❌ Summary file not found: {summary_path}")
        sys.exit(1)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    failed_entries = summary.get("results", {}).get("failed", [])
    if not failed_entries:
        print("✅ No failed entries in summary; nothing to retry.")
        sys.exit(0)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    json_folder = Path(summary.get("json_folder", "."))
    if output_dir_override:
        output_dir = Path(output_dir_override)
    else:
        raw_out = summary.get("output_dir", ".")
        output_dir = Path(raw_out) if raw_out != "." else summary_path.parent

    videos_dir   = output_dir / "videos"
    metadata_dir = output_dir / "metadata"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("🔁 Retry Downloader")
    print(f"   Summary:    {summary_path}")
    print(f"   JSON folder:{json_folder}")
    print(f"   Output dir: {output_dir}")
    print(f"   Max retries:{max_retries}  Base delay: {base_delay}s")
    print(f"   Quality:    {quality}")
    print(f"   Cookies:    {cookies_file or 'None'}")
    print(f"   Failed entries to process: {len(failed_entries)}")
    print("=" * 60)

    retry_results = {"success": [], "failed": [], "skipped": []}

    for idx, entry in enumerate(failed_entries, 1):
        json_filename = entry.get("file", "")
        original_reason = entry.get("reason", "")

        print(f"\n[{idx}/{len(failed_entries)}] {json_filename}")
        print(f"  Original failure: {original_reason[:120]}")
        print("-" * 40)

        # ── Load source JSON ──────────────────────────────────────────────────
        json_path = json_folder / json_filename
        if not json_path.exists():
            print(f"  ❌ Source JSON not found: {json_path}")
            retry_results["failed"].append({
                "file": json_filename,
                "reason": f"Source JSON not found: {json_path}",
            })
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                source_data = json.load(f)
        except Exception as e:
            print(f"  ❌ Could not read source JSON: {e}")
            retry_results["failed"].append({"file": json_filename, "reason": str(e)})
            continue

        video_url = source_data.get("video_url", "")
        if not video_url or not is_valid_youtube_url(video_url):
            print(f"  ❌ Invalid or missing video_url: {video_url!r}")
            retry_results["skipped"].append({
                "file": json_filename,
                "reason": f"Invalid video_url: {video_url!r}",
            })
            continue

        video_id = extract_video_id(video_url)
        if not video_id:
            print(f"  ❌ Could not extract video ID from: {video_url}")
            retry_results["failed"].append({
                "file": json_filename,
                "reason": f"Could not extract video ID from: {video_url}",
            })
            continue

        print(f"  🔗 {video_url}")
        print(f"  🆔 {video_id}")

        # ── Skip if already downloaded ────────────────────────────────────────
        existing = videos_dir / f"{video_id}.mp4"
        if existing.exists():
            print(f"  ⏭️  Already downloaded; skipping.")
            retry_results["skipped"].append({
                "file": json_filename,
                "reason": f"Already downloaded ({video_id}.mp4)",
            })
            continue

        # ── Attempt download with retries ─────────────────────────────────────
        status, message = retry_one(
            video_url=video_url,
            video_id=video_id,
            videos_dir=videos_dir,
            metadata_dir=metadata_dir,
            source_json_data=source_data,
            quality=quality,
            cookies_file=cookies_file,
            max_retries=max_retries,
            base_delay=base_delay,
        )

        if status == "success":
            print(f"  ✅ {message}")
            retry_results["success"].append({
                "file": json_filename,
                "downloaded": f"{video_id}.mp4",
            })
        else:
            print(f"  ❌ Exhausted retries: {message}")
            retry_results["failed"].append({"file": json_filename, "reason": message})

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("📊 RETRY SUMMARY")
    print("=" * 60)
    print(f"✅ Recovered:  {len(retry_results['success'])}")
    print(f"⏭️  Skipped:    {len(retry_results['skipped'])}")
    print(f"❌ Still failed:{len(retry_results['failed'])}")

    if retry_results["success"]:
        print("\n✅ Recovered:")
        for e in retry_results["success"]:
            print(f"   • {e['file']} → {e['downloaded']}")
    if retry_results["skipped"]:
        print("\n⏭️  Skipped (already downloaded or bad URL):")
        for e in retry_results["skipped"]:
            print(f"   • {e['file']}: {e['reason'][:100]}")
    if retry_results["failed"]:
        print("\n❌ Still failing after retries:")
        for e in retry_results["failed"]:
            print(f"   • {e['file']}: {e['reason'][:100]}")

    # ── Write retry_summary.json ──────────────────────────────────────────────
    retry_summary_path = output_dir / "retry_summary.json"
    retry_summary = {
        "timestamp": datetime.now().isoformat(),
        "source_summary": str(summary_path),
        "json_folder": str(json_folder),
        "output_dir": str(output_dir),
        "quality": quality,
        "max_retries": max_retries,
        "base_delay_seconds": base_delay,
        "original_failed_count": len(failed_entries),
        "results": retry_results,
    }
    with open(retry_summary_path, "w", encoding="utf-8") as f:
        json.dump(retry_summary, f, indent=2, ensure_ascii=False)
    print(f"\n📋 Retry report → {retry_summary_path}")
    print("=" * 60)

    return retry_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Retry failed downloads from a batch_download.py summary file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
All failed entries are retried with exponential backoff up to --max-retries times.

Examples:
    python retry_downloads.py --summary download_summary.json
    python retry_downloads.py --summary ./out/download_summary.json --max-retries 5
    python retry_downloads.py --summary download_summary.json --cookies cookies.txt
        """,
    )
    parser.add_argument(
        "--summary", default="download_summary.json",
        help="Path to the download_summary.json produced by batch_download.py",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override the output directory from the summary (videos/ and metadata/ go here)",
    )
    parser.add_argument(
        "--cookies", default=None,
        help="Path to cookies.txt for authenticated downloads",
    )
    parser.add_argument(
        "--quality", choices=["best", "720p", "480p"], default="best",
        help="Video quality cap (default: best)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Maximum download attempts per video (default: 3)",
    )
    parser.add_argument(
        "--base-delay", type=float, default=5.0,
        help="Initial wait in seconds between retries; doubles each attempt, capped at 120s (default: 5)",
    )

    args = parser.parse_args()

    summary_path = Path(args.summary)

    if args.cookies and not Path(args.cookies).exists():
        print(f"❌ Cookies file not found: {args.cookies}")
        sys.exit(1)

    results = run_retries(
        summary_path=summary_path,
        output_dir_override=args.output_dir,
        cookies_file=args.cookies,
        quality=args.quality,
        max_retries=args.max_retries,
        base_delay=args.base_delay,
    )

    sys.exit(1 if results["failed"] else 0)


if __name__ == "__main__":
    main()

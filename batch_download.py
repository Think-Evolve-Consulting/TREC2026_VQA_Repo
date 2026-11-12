"""
Batch YouTube downloader.

Input: a folder of JSON files, each containing at minimum a `video_url` field
       (same format as sample.py). Quiz fields (question, correct_answer,
       incorrect_answers) are preserved in the per-video metadata if present.

Library: yt_dlp (robust against YouTube changes, handles muxing internally).

Codec handling: after download, probe the video codec. If it's AV1 / VP9 / VP8
                (which break some older players and ML pipelines), re-encode to
                H.264 + AAC in MP4 using ffmpeg. Ported from download.py.

Output layout (same as download.py):
    output_dir/
        videos/{video_id}.mp4
        metadata/{video_id}.json

Errors: per-file try/except, aggregate success/failed/skipped tracking, and a
        download_summary.json written to the output directory at the end.

Usage:
    python batch_download.py --json-folder ./jsons --output-dir ./out
    python batch_download.py --json-folder ./jsons --output-dir ./out --cookies cookies.txt
    python batch_download.py --json-folder ./jsons --output-dir ./out --quality 720p
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yt_dlp


# ---------------------------------------------------------------------------
# Codec detection & re-encoding (ported from download.py)
# ---------------------------------------------------------------------------

PROBLEMATIC_CODECS = {"av1", "av01", "vp9", "vp8"}


def detect_video_codec(video_path):
    """Return the codec_name of the first video stream, or None on failure."""
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def is_problematic_codec(codec_name):
    return bool(codec_name) and codec_name.lower() in PROBLEMATIC_CODECS


def reencode_to_h264_if_needed(video_path: Path):
    """
    Probe the file. If the video codec is AV1/VP9/VP8, re-encode to H.264+AAC
    MP4 in place. Returns True if the file is H.264-compatible after the call
    (either originally, or after a successful re-encode), False if re-encode
    was attempted and failed.
    """
    codec = detect_video_codec(video_path)
    if codec is None:
        print(f"  ⚠️  Could not probe codec for {video_path.name}; leaving as-is.")
        return True
    if not is_problematic_codec(codec):
        print(f"  ✅ Codec OK ({codec}), no re-encode needed.")
        return True

    print(f"  🔄 Codec {codec} is problematic; re-encoding to H.264…")
    tmp_out = video_path.with_suffix(".h264_tmp.mp4")

    # Try standard H.264 first, then fall back to baseline profile for max compat.
    attempts = [
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(tmp_out),
        ],
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-profile:v", "baseline", "-level", "3.0", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(tmp_out),
        ],
    ]

    for i, cmd in enumerate(attempts, 1):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            try:
                os.replace(tmp_out, video_path)
                print(f"  ✅ Re-encoded (attempt {i}) → {video_path.name}")
                return True
            except Exception as e:
                print(f"  ❌ Could not replace original after re-encode: {e}")
                return False
        print(f"  ⚠️  Re-encode attempt {i} failed.")

    # Clean up tmp file if it's still around
    if tmp_out.exists():
        try:
            tmp_out.unlink()
        except Exception:
            pass
    print(f"  ❌ All re-encode attempts failed; leaving original file.")
    return False


# ---------------------------------------------------------------------------
# URL handling
# ---------------------------------------------------------------------------

_YT_URL_RE = re.compile(
    r"(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/"
    r"(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})"
)
_YT_ID_RE = re.compile(
    r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})"
)


def is_valid_youtube_url(url):
    return bool(url) and bool(_YT_URL_RE.match(url))


def extract_video_id(url):
    """Pull the 11-char video ID from a YouTube URL. Returns None if not found."""
    if not url:
        return None
    m = _YT_ID_RE.search(url)
    if m:
        return m.group(1)
    # Fallback: if the URL is just the ID
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()):
        return url.strip()
    return None


# ---------------------------------------------------------------------------
# Download + metadata (output layout ported from download.py)
# ---------------------------------------------------------------------------

def save_video_metadata(info, video_id, metadata_dir, source_json_data=None):
    """
    Write {video_id}.json into metadata_dir, using the info dict returned by
    yt_dlp. If the source JSON had quiz fields, merge them in.
    """
    try:
        publish_date = None
        upload_date = info.get("upload_date")  # yt_dlp format: 'YYYYMMDD'
        if upload_date and len(upload_date) == 8:
            try:
                publish_date = datetime.strptime(upload_date, "%Y%m%d").date().isoformat()
            except ValueError:
                publish_date = None

        metadata = {
            "video_id": video_id,
            "title": info.get("title"),
            "description": info.get("description"),
            "author": info.get("uploader") or info.get("channel"),
            "length": info.get("duration"),
            "views": info.get("view_count"),
            "publish_date": publish_date,
            "keywords": info.get("tags") or [],
            "thumbnail_url": info.get("thumbnail"),
        }

        # Preserve quiz fields from source JSON if present
        if source_json_data:
            for key in ("question", "correct_answer", "incorrect_answers"):
                if key in source_json_data:
                    metadata[key] = source_json_data[key]

        metadata_path = Path(metadata_dir) / f"{video_id}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  📝 Saved metadata: {metadata_path}")
        return True
    except Exception as e:
        print(f"  ⚠️  Could not save metadata for {video_id}: {e}")
        return False


def download_one(video_url, video_id, videos_dir, metadata_dir,
                 quality="best", cookies_file=None, source_json_data=None):
    """
    Download a single video into videos_dir/{video_id}.mp4, re-encode if the
    codec is problematic, and write metadata. Returns (ok: bool, message: str).
    """
    # Format selection — prefer MP4 container so we land on H.264/AAC when
    # possible and avoid the re-encode path entirely.
    if quality == "best":
        fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    elif quality == "720p":
        fmt = "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]"
    elif quality == "480p":
        fmt = "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]"
    else:
        fmt = "best"

    final_path = Path(videos_dir) / f"{video_id}.mp4"

    with tempfile.TemporaryDirectory(prefix="ytdl_") as tmpdir:
        ydl_opts = {
            # Force deterministic filename keyed to video_id
            "outtmpl": os.path.join(tmpdir, f"{video_id}.%(ext)s"),
            "format": fmt,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }

        if cookies_file and os.path.exists(cookies_file):
            ydl_opts["cookiefile"] = cookies_file

        def progress_hook(d):
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate")
                downloaded = d.get("downloaded_bytes", 0)
                if total:
                    pct = downloaded / total * 100
                    print(f"  📥 Downloading… {pct:5.1f}%", end="\r", flush=True)
            elif d.get("status") == "finished":
                print("  🔄 Post-processing…" + " " * 20)

        ydl_opts["progress_hooks"] = [progress_hook]

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
        except Exception as e:
            return False, f"yt_dlp error: {e}"

        # Find the downloaded file
        produced = list(Path(tmpdir).glob(f"{video_id}.*"))
        # Filter to video files (exclude .part, .info.json, etc.)
        video_files = [p for p in produced if p.suffix.lower() in (".mp4", ".mkv", ".webm")]
        if not video_files:
            return False, "Download completed but no video file found"
        src = video_files[0]

        # Move into final videos_dir as {video_id}.mp4. If it's .mkv/.webm we
        # still rename to .mp4 only after codec check/re-encode — but yt_dlp
        # with merge_output_format=mp4 should give us mp4 already.
        if src.suffix.lower() != ".mp4":
            # Force remux/transcode into .mp4 container
            remuxed = src.with_suffix(".mp4")
            cmd = [
                "ffmpeg", "-y", "-i", str(src),
                "-c", "copy", "-movflags", "+faststart", str(remuxed),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                # Copy failed (codec incompatible with mp4 container) — transcode
                cmd = [
                    "ffmpeg", "-y", "-i", str(src),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(remuxed),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    return False, f"Could not convert to mp4: {proc.stderr[:200]}"
            src = remuxed

        # Move to final location
        final_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(src, final_path)
        except OSError:
            # Cross-device fallback
            import shutil
            shutil.copy2(src, final_path)

    # Codec check + re-encode if needed
    reencode_to_h264_if_needed(final_path)

    # Metadata
    save_video_metadata(info, video_id, metadata_dir, source_json_data=source_json_data)

    size_mb = final_path.stat().st_size / (1024 * 1024)
    return True, f"Saved {final_path.name} ({size_mb:.1f} MB)"


# ---------------------------------------------------------------------------
# Batch driver (error-handling style from sample.py)
# ---------------------------------------------------------------------------

def process_json_folder(json_folder, output_dir, cookies_file=None, quality="best"):
    json_folder = Path(json_folder)
    output_dir = Path(output_dir)

    if not json_folder.exists():
        print(f"❌ JSON folder does not exist: {json_folder}")
        sys.exit(1)

    videos_dir = output_dir / "videos"
    metadata_dir = output_dir / "metadata"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_folder.glob("*.json"))
    if not json_files:
        print(f"⚠️  No JSON files found in: {json_folder}")
        sys.exit(0)

    print(f"📂 Found {len(json_files)} JSON file(s) in {json_folder}")
    print(f"📁 Videos   → {videos_dir}")
    print(f"📁 Metadata → {metadata_dir}")
    print("=" * 60)

    results = {"success": [], "failed": [], "skipped": []}

    for i, json_file in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] {json_file.name}")
        print("-" * 40)

        # Parse JSON
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  ❌ Invalid JSON: {e}")
            results["failed"].append((json_file.name, f"Invalid JSON: {e}"))
            continue
        except Exception as e:
            print(f"  ❌ Could not read file: {e}")
            results["failed"].append((json_file.name, f"Read error: {e}"))
            continue

        video_url = data.get("video_url")
        if not video_url:
            print(f"  ⚠️  No 'video_url' field")
            results["skipped"].append((json_file.name, "No video_url field"))
            continue

        if not is_valid_youtube_url(video_url):
            print(f"  ❌ Invalid YouTube URL: {video_url}")
            results["failed"].append((json_file.name, "Invalid YouTube URL"))
            continue

        video_id = extract_video_id(video_url)
        if not video_id:
            print(f"  ❌ Could not extract video ID from: {video_url}")
            results["failed"].append((json_file.name, "Could not extract video ID"))
            continue

        print(f"  🔗 {video_url}")
        print(f"  🆔 {video_id}")

        # Skip if already downloaded
        existing = videos_dir / f"{video_id}.mp4"
        if existing.exists():
            print(f"  ⏭️  Already exists, skipping: {existing.name}")
            results["skipped"].append((json_file.name, f"Already downloaded ({video_id}.mp4)"))
            continue

        # Download
        try:
            ok, msg = download_one(
                video_url=video_url,
                video_id=video_id,
                videos_dir=videos_dir,
                metadata_dir=metadata_dir,
                quality=quality,
                cookies_file=cookies_file,
                source_json_data=data,
            )
            if ok:
                print(f"  ✅ {msg}")
                results["success"].append((json_file.name, f"{video_id}.mp4"))
            else:
                print(f"  ❌ {msg}")
                results["failed"].append((json_file.name, msg))
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            results["failed"].append((json_file.name, str(e)))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {len(results['success'])}")
    print(f"❌ Failed:     {len(results['failed'])}")
    print(f"⚠️  Skipped:    {len(results['skipped'])}")

    if results["success"]:
        print(f"\n✅ Successful downloads:")
        for src, out in results["success"]:
            print(f"   • {src} → {out}")
    if results["failed"]:
        print(f"\n❌ Failed downloads:")
        for src, reason in results["failed"]:
            print(f"   • {src}: {reason}")
    if results["skipped"]:
        print(f"\n⚠️  Skipped:")
        for src, reason in results["skipped"]:
            print(f"   • {src}: {reason}")

    # Persist summary
    summary_path = output_dir / "download_summary.json"
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "json_folder": str(json_folder),
        "output_dir": str(output_dir),
        "total_files": len(json_files),
        "quality": quality,
        "results": {
            "success": [{"file": f, "downloaded": d} for f, d in results["success"]],
            "failed":  [{"file": f, "reason": r}     for f, r in results["failed"]],
            "skipped": [{"file": f, "reason": r}     for f, r in results["skipped"]],
        },
    }
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary_data, sf, indent=2, ensure_ascii=False)
    print(f"\n📋 Summary report → {summary_path}")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch YouTube downloader with codec normalization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Each JSON file in --json-folder should contain at minimum:
    { "video_url": "https://www.youtube.com/watch?v=..." }

Optional fields (question, correct_answer, incorrect_answers) are preserved
in the per-video metadata JSON.

Output layout:
    <output-dir>/videos/<video_id>.mp4
    <output-dir>/metadata/<video_id>.json
    <output-dir>/download_summary.json
        """,
    )
    parser.add_argument("--json-folder", required=True,
                        help="Folder containing JSON files with video_url fields")
    parser.add_argument("--output-dir", default=".",
                        help="Output root (will contain videos/ and metadata/ subfolders). Default: current directory.")
    parser.add_argument("--cookies", default=None,
                        help="Path to cookies.txt for authenticated downloads (optional)")
    parser.add_argument("--quality", choices=["best", "720p", "480p"], default="best",
                        help="Video quality cap (default: best)")
    args = parser.parse_args()

    if args.cookies and not os.path.exists(args.cookies):
        print(f"❌ Cookies file not found: {args.cookies}")
        sys.exit(1)

    print("🚀 Batch YouTube Downloader (yt_dlp + codec normalization)")
    print(f"   JSON folder: {args.json_folder}")
    print(f"   Output dir:  {args.output_dir}")
    print(f"   Quality:     {args.quality}")
    print(f"   Cookies:     {args.cookies or 'None'}")
    print()

    results = process_json_folder(
        json_folder=args.json_folder,
        output_dir=args.output_dir,
        cookies_file=args.cookies,
        quality=args.quality,
    )

    sys.exit(1 if results["failed"] else 0)


if __name__ == "__main__":
    main()

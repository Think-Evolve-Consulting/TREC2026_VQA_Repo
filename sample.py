import yt_dlp
import tempfile
import shutil
import re
import uuid
# import google.generativeai as genai
from datetime import datetime
import argparse
from pathlib import Path
import json
import os

class YouTubeDownloader:
    """Reused from app.py - standalone version without Gradio dependencies"""

    def __init__(self, download_folder=None):
        self.download_dir = tempfile.mkdtemp()
        self.temp_downloads = tempfile.mkdtemp(prefix="youtube_downloads_")
        # Use the provided download folder, or default to ~/Downloads/YouTube_Downloads
        if download_folder:
            self.downloads_folder = download_folder
            os.makedirs(self.downloads_folder, exist_ok=True)
        else:
            self.downloads_folder = os.path.join(
                os.path.expanduser("~"), "Downloads", "YouTube_Downloads"
            )
            os.makedirs(self.downloads_folder, exist_ok=True)
        self.gemini_model = None

    # def configure_gemini(self, api_key):
    #     """Configure Gemini API with the provided key"""
    #     try:
    #         genai.configure(api_key=api_key)
    #         self.gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
    #         return True, "✅ Gemini API configured successfully!"
    #     except Exception as e:
    #         return False, f"❌ Failed to configure Gemini API: {str(e)}"

    def cleanup(self):
        """Clean up temporary directories and files"""
        try:
            if hasattr(self, "download_dir") and os.path.exists(self.download_dir):
                shutil.rmtree(self.download_dir)
                print(f"✅ Cleaned up temporary directory: {self.download_dir}")
            if hasattr(self, "temp_downloads") and os.path.exists(self.temp_downloads):
                shutil.rmtree(self.temp_downloads)
                print(f"✅ Cleaned up temp downloads directory: {self.temp_downloads}")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean up temporary directory: {e}")

    def is_valid_youtube_url(self, url):
        youtube_regex = re.compile(
            r"(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/"
            r"(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})"
        )
        return youtube_regex.match(url) is not None

    def download_video(self, url, quality="best", audio_only=False, cookiefile=None):
        """Download video with progress tracking (no Gradio progress object)"""
        if not url or not url.strip():
            return None, "❌ Please enter a YouTube URL"

        if not self.is_valid_youtube_url(url):
            return None, "❌ Invalid YouTube URL format"

        try:
            print(f"  🔄 Preparing download for: {url}")

            # Create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Download to temp directory first
            ydl_opts = {
                "outtmpl": os.path.join(
                    self.temp_downloads, f"%(title)s_{timestamp}.%(ext)s"
                ),
                "noplaylist": True,
            }

            if audio_only:
                ydl_opts["format"] = "bestaudio/best"
                ydl_opts["postprocessors"] = [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ]
            else:
                if quality == "best":
                    ydl_opts["format"] = "best[height<=1080]"
                elif quality == "720p":
                    ydl_opts["format"] = "best[height<=720]"
                elif quality == "480p":
                    ydl_opts["format"] = "best[height<=480]"
                else:
                    ydl_opts["format"] = "best"

            if cookiefile and os.path.exists(cookiefile):
                ydl_opts["cookiefile"] = cookiefile

            # Progress hook (console-based, no Gradio)
            def progress_hook(d):
                if d["status"] == "downloading":
                    if "total_bytes" in d:
                        percent = (d["downloaded_bytes"] / d["total_bytes"]) * 100
                        print(
                            f"  📥 Downloading... {percent:.1f}%", end="\r", flush=True
                        )
                    else:
                        print("  📥 Downloading...", end="\r", flush=True)
                elif d["status"] == "finished":
                    print("  🔄 Processing download...")

            ydl_opts["progress_hooks"] = [progress_hook]

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

            # Find the downloaded file in temp directory
            downloaded_file_temp = None

            for file in os.listdir(self.temp_downloads):
                if timestamp in file:
                    downloaded_file_temp = os.path.join(self.temp_downloads, file)
                    break

            if not downloaded_file_temp:
                return None, "❌ Downloaded file not found in temp directory"

            # Copy to user's Downloads folder
            final_filename = os.path.basename(downloaded_file_temp)
            final_path = os.path.join(self.downloads_folder, final_filename)

            try:
                shutil.copy2(downloaded_file_temp, final_path)
                copy_success = True
            except Exception as e:
                print(f"  ⚠️ Warning: Could not copy to Downloads folder: {e}")
                copy_success = False
                final_path = "File downloaded to temp location only"

            file_size_mb = os.path.getsize(downloaded_file_temp) / (1024 * 1024)

            success_msg = f"""✅ Download successful!
📁 File: {os.path.basename(downloaded_file_temp)}
📁 Saved to: {final_path if copy_success else 'Copy failed'}
🎯 File size: {file_size_mb:.1f} MB"""

            return downloaded_file_temp, success_msg

        except Exception as e:
            return None, f"❌ Download failed: {str(e)}"


def process_json_folder(json_folder, download_folder, cookies_file=None, quality="best", audio_only=False):
    """
    Read all JSON files in the given folder, extract YouTube URLs, and download each video.

    Args:
        json_folder: Path to folder containing JSON files
        download_folder: Path to folder where downloads should be saved
        cookies_file: Optional path to cookies.txt file
        quality: Video quality preference
        audio_only: Whether to download audio only
    """
    json_folder = Path(json_folder)
    download_folder = Path(download_folder)

    if not json_folder.exists():
        print(f"❌ JSON folder does not exist: {json_folder}")
        sys.exit(1)

    # Create download folder if it doesn't exist
    download_folder.mkdir(parents=True, exist_ok=True)

    # Initialize downloader with the specified download folder
    downloader = YouTubeDownloader(download_folder=str(download_folder))

    # Find all JSON files
    json_files = sorted(json_folder.glob("*.json"))

    if not json_files:
        print(f"⚠️ No JSON files found in: {json_folder}")
        sys.exit(0)

    print(f"📂 Found {len(json_files)} JSON file(s) in {json_folder}")
    print(f"📁 Downloads will be saved to: {download_folder}")
    print(f"{'='*60}")

    # Track results
    results = {"success": [], "failed": [], "skipped": []}

    for i, json_file in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] Processing: {json_file.name}")
        print("-" * 40)

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  ❌ Invalid JSON: {e}")
            results["failed"].append((json_file.name, "Invalid JSON"))
            continue

        video_url = data.get("video_url")

        if not video_url:
            print(f"  ⚠️ No 'video_url' field found in {json_file.name}")
            results["skipped"].append((json_file.name, "No video_url field"))
            continue

        print(f"  🔗 URL: {video_url}")

        # Validate URL
        if not downloader.is_valid_youtube_url(video_url):
            print(f"  ❌ Invalid YouTube URL: {video_url}")
            results["failed"].append((json_file.name, "Invalid YouTube URL"))
            continue

        # Attempt download
        try:
            file_path, status_msg = downloader.download_video(
                url=video_url,
                quality=quality,
                audio_only=audio_only,
                cookiefile=cookies_file,
            )

            if file_path:
                print(f"\n  {status_msg}")
                results["success"].append((json_file.name, os.path.basename(file_path)))

                # Also save quiz data if present in the JSON
                question = data.get("question")
                correct_answer = data.get("correct_answer")
                incorrect_answers = data.get("incorrect_answers", [])

                if question and correct_answer:
                    quiz_file = json_file.stem + "_quiz.json"
                    quiz_path = download_folder / quiz_file
                    quiz_data = {
                        "source_file": json_file.name,
                        "video_url": video_url,
                        "question": question,
                        "correct_answer": correct_answer,
                        "incorrect_answers": incorrect_answers,
                    }
                    with open(quiz_path, "w", encoding="utf-8") as qf:
                        json.dump(quiz_data, qf, indent=2, ensure_ascii=False)
                    print(f"  📝 Quiz data saved to: {quiz_path}")

            else:
                print(f"  {status_msg}")
                results["failed"].append((json_file.name, status_msg))

        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            results["failed"].append((json_file.name, str(e)))

    # Clean up temporary directories
    downloader.cleanup()

    # Print summary
    print(f"\n{'='*60}")
    print("📊 DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {len(results['success'])}")
    print(f"❌ Failed:     {len(results['failed'])}")
    print(f"⚠️ Skipped:    {len(results['skipped'])}")

    if results["success"]:
        print(f"\n✅ Successful downloads:")
        for filename, downloaded_file in results["success"]:
            print(f"   • {filename} → {downloaded_file}")

    if results["failed"]:
        print(f"\n❌ Failed downloads:")
        for filename, reason in results["failed"]:
            print(f"   • {filename}: {reason}")

    if results["skipped"]:
        print(f"\n⚠️ Skipped files:")
        for filename, reason in results["skipped"]:
            print(f"   • {filename}: {reason}")

    print(f"\n📁 All downloads saved to: {download_folder}")
    print(f"{'='*60}")

    # Save summary report to download folder
    summary_path = download_folder / "download_summary.json"
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "json_folder": str(json_folder),
        "download_folder": str(download_folder),
        "total_files": len(json_files),
        "results": {
            "success": [{"file": f, "downloaded": d} for f, d in results["success"]],
            "failed": [{"file": f, "reason": r} for f, r in results["failed"]],
            "skipped": [{"file": f, "reason": r} for f, r in results["skipped"]],
        },
    }
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary_data, sf, indent=2, ensure_ascii=False)
    print(f"📋 Summary report saved to: {summary_path}")

    return results


def main():
    """Parse command-line arguments and run the batch downloader"""
    parser = argparse.ArgumentParser(
        description="Batch YouTube Downloader from JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example JSON file format:
    {
        "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "question": "What is shown in this video?",
        "correct_answer": "A music video",
        "incorrect_answers": ["A tutorial", "A vlog", "A news clip"],
        "timestamp": "2025-07-11T19:18:41.010Z"
    }

Example usage:
    python sample.py --json-folder ./jsons --download-folder ./downloads
    python sample.py --json-folder ./jsons --download-folder ./downloads --cookies cookies.txt
    python sample.py --json-folder ./jsons --download-folder ./downloads --quality 720p
    python sample.py --json-folder ./jsons --download-folder ./downloads --audio-only
        """,
    )

    parser.add_argument(
        "--json-folder",
        required=True,
        help="Path to folder containing JSON files with YouTube URLs",
    )
    parser.add_argument(
        "--download-folder",
        required=True,
        help="Path to folder where downloaded videos will be saved",
    )
    parser.add_argument(
        "--cookies",
        default=None,
        help="Path to cookies.txt file for authenticated downloads (optional)",
    )
    parser.add_argument(
        "--quality",
        choices=["best", "720p", "480p"],
        default="best",
        help="Video quality preference (default: best)",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Download audio only (MP3 format)",
    )

    args = parser.parse_args()

    # Validate cookies file if provided
    if args.cookies and not os.path.exists(args.cookies):
        print(f"❌ Cookies file not found: {args.cookies}")
        sys.exit(1)

    print("🚀 Batch YouTube Downloader")
    print(f"   Quality:    {args.quality}")
    print(f"   Audio only: {args.audio_only}")
    print(f"   Cookies:    {args.cookies if args.cookies else 'None'}")
    print()

    # Run the batch processor
    results = process_json_folder(
        json_folder=args.json_folder,
        download_folder=args.download_folder,
        cookies_file=args.cookies,
        quality=args.quality,
        audio_only=args.audio_only,
    )

    # Exit with error code if there were failures
    if results["failed"]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

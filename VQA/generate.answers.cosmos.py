#!/usr/bin/env python3
"""Answer the Task 1 questions with Cosmos-Reason2-2B, one reasoning call per query.

Companion to ``generate.answers.py`` (which ensembles Ollama/Gemma4 over 30
calls per query). This script targets a Cosmos-Reason2-2B server (vLLM,
OpenAI-compatible, served from the WSL2 box that owns the GPU -- see
/home/aakash/cosmos-reason2) and asks it once per query, with reasoning
enabled, to return its own top-10 ranked answers directly as JSON. That is
roughly 30x fewer inference calls than the Gemma4 pipeline, trading the
paraphrase-and-vote ensemble for Cosmos's own chain-of-thought.

Cosmos-Reason2 takes video natively (a ``file://`` path the server reads
itself), so unlike the Gemma4 pipeline this needs no pre-extracted frames --
only the transcript, which is folded into the prompt the same way
``generate.answers.py`` does it.

Output is the submission file ``VQA/validate.ag.run.py`` accepts verbatim::

    [{"Q_ID": 1, "Video_ID": "8384.webm",
      "Answers": [{"Rank": 1, "Answer": "...", "Time": 1.234}, ...]}]

**Fixed 14 August 2026 -- ``Video_ID`` is not a filename.** 868 of the 2,443
queries name a bare YouTube ID, and ``download_test_videos.py`` wrote those as
``<Video_ID>.mp4`` beside the released ``.webm`` clips
(SESSION.2026-08-11.task1-pipeline.md §3). This script previously resolved
``videos_dir / video_id`` literally, so it never opened one of the 852
downloaded mp4s: ``cosmos.run.task1.json`` was produced with **1,098 of 2,443
queries answered with no video at all**, 469 of them with no transcript either,
where only 16 queries genuinely lack media. ``resolve_video()`` now applies the
documented convention. Any run made before this date is affected -- re-run to
benefit, since the cache holds the blind answers.

The same fix exposed a second problem it now also corrects: sampling fps came
from the transcript's ``duration_sec``, which does not exist for the 1,080
queries whose video has no speech. Those planned at a flat 2 fps -- harmless
while no video was attached, wrong once one is -- so duration now comes from
the probed ``frames/index.json`` first, transcript second.

Run this from WSL2 (not Windows) so it hits the vLLM server over a real
localhost connection and reads /mnt/e/... directly -- see the project's
setup notes. Usage:

    python3 generate.answers.cosmos.py --limit 5     # smoke test
    python3 generate.answers.cosmos.py               # everything, resumable
    python3 generate.answers.cosmos.py --only 1 3 264 # named Q_IDs
    python3 generate.answers.cosmos.py --rebuild      # re-emit run from cache
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from openai import BadRequestError, OpenAI

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = ROOT / "vqa.2026.testingDataset.Task1.json"
DEFAULT_VIDEOS = ROOT / "videos_TEST_VQA" / "videos"
DEFAULT_FRAME_INDEX = ROOT / "videos_TEST_VQA" / "frames" / "index.json"
DEFAULT_TRANSCRIPTS = ROOT / "videos_TEST_VQA" / "transcription_openwhisper"
DEFAULT_OUT = ROOT / "cosmos.run.task1.json"
DEFAULT_CACHE = ROOT / "answers.task1.cosmos.cache.jsonl"

# vLLM's video IO layer is started with --media-io-kwargs num_frames=-1 (the
# repo's own recommendation, needed for Qwen3-VL/Cosmos-Reason2 to embed
# correct per-frame timestamps). That means it decodes every native-resolution
# frame into a raw numpy array *before* any fps-based subsampling happens
# downstream -- fine for short clips, but a long and/or high-res source
# allocates gigabytes of host RAM up front. A 320s, 2048x1152, 24fps clip
# ("ChainedforLife-14.webm", real example from this corpus) hit a 50.6GiB
# allocation and crashed the request. Rather than weaken --media-io-kwargs
# globally, videos whose native decode footprint would exceed
# RAW_DECODE_BYTES_LIMIT are pre-shrunk with ffmpeg (frame-rate reduction is
# what actually matters -- resolution alone rarely trips this) before being
# handed to the server; the cheap majority of clips pass through untouched.
RAW_DECODE_BYTES_LIMIT = 4 * 1024**3
TRANSCODE_CACHE_DIRNAME = ".cosmos_transcoded_cache"
"""Lives under the videos dir, inside --allowed-local-media-path, so the
server can still read it directly by path."""

DEFAULT_MODEL = "nvidia/Cosmos-Reason2-2B"
DEFAULT_BASE_URL = "http://localhost:8008/v1"

# One call per query instead of Gemma4's 10x3 ensemble: Cosmos-Reason2 is
# asked, with reasoning enabled, to rank its own top-10 answers in one shot.
# See generate.answers.py's docstring for why the ensemble exists there --
# it compensates for a smaller ensembled model's variance. Cosmos-Reason2's
# own chain-of-thought is the thing worth paying for here instead.
DEFAULT_WORKERS = 2
"""Concurrent in-flight requests. vLLM does its own request scheduling and
continuous batching server-side on the one GPU; this just keeps the queue
fed, it does not multiply GPU throughput the way more GPUs would."""

DEFAULT_MAX_TOKENS = 8192
"""Generation budget: enough for a real reasoning trace plus a 10-item JSON
answer list. 3072 was tried first and measurably too tight -- 3/25 queries in
an early validation batch hit it, generation cut off mid-<think> before ever
reaching the closing tag or JSON (reasoning_chars=0, content_parsed=False,
each the slowest calls in the batch at 50-63s vs. 5-15s for a clean
completion). Paired with --max-model-len 24576 on the server (up from
16384) so the larger generation budget doesn't eat into the video token
budget -- see video_mm_kwargs()."""

DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20

ANSWERS_PER_QUERY = 10

# Visual-token pixel budget, confirmed against
# cosmos_reason2_utils/vision.py: PIXELS_PER_TOKEN = (16 * 2) ** 2.
PIXELS_PER_TOKEN = 1024
MIN_VISION_TOKENS = 256
"""Floor on visual tokens per video, matching scripts/inference_sample.py."""

# Adaptive fps: aim for a fixed frame budget per video regardless of length,
# so a 14s clip and a 6.5-minute one both fit comfortably in the context
# window. Duration distribution actually seen in videos_TEST_VQA (n=1935):
# median 10.3s, p90 56s, p99 233s, max 389s -- only the long tail needs this.
TARGET_FRAMES = 64
MAX_FPS = 4.0
MIN_FPS = 0.2

SYSTEM_PROMPT = "You are a helpful assistant."

# Adapted from generate.answers.py's PROMPT_TEMPLATE: same rules (short
# answers, genuinely distinct, "unknown" over guessing), but one call
# instead of ten, and reasoning goes in <think> for --reasoning-parser qwen3
# to split out, so `content` ends up holding only the final JSON.
PROMPT_TEMPLATE = """{context}Question: {question}

Give your {k} most likely answers to this question, ranked best first.

Rules:
- Answer the question and nothing else. Do not restate the question, do not
  describe the video, and do not explain your reasoning outside the think
  block.
- Keep every answer very short: a phrase or one short sentence. Most good
  answers are under ten words.
- The {k} answers must be genuinely different possibilities, not rewordings
  of each other.
- Use the video and the transcript. If neither shows the answer, say
  "unknown" rather than guessing.

Answer using the following format:

<think>
Your reasoning about what the video and transcript show.
</think>

Then, on the final line, write ONLY this JSON, with exactly {k} items ranked
best first: {{"answers": ["...", ...]}}"""

LEAD_IN_RE = re.compile(r"^\s*(?:answer|a)\s*[:\-]\s*", re.IGNORECASE)
ENUM_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
JSON_BLOB_RE = re.compile(r"\{.*\}", re.DOTALL)

MAX_ANSWER_WORDS = 40
MAX_ANSWER_CHARS = 300

# Same fallback pool as generate.answers.py -- defensible non-answers rather
# than fabricated facts, used only when the model returns fewer than
# ANSWERS_PER_QUERY distinct candidates.
FALLBACK_ANSWERS = (
    "unknown",
    "it is not shown in the video",
    "the video does not show this",
    "nothing happens",
    "it is not clear from the video",
    "no one",
    "none",
    "it cannot be determined",
    "not visible in the video",
    "the answer is not shown",
    "no",
    "yes",
)


def normalize(text: str) -> str:
    """The validator's own duplicate rule -- validate.ag.run.py:276."""
    return " ".join(text.split()).casefold()


def tidy(text: str) -> str:
    """Clean one candidate answer into the form that goes in the run file."""
    text = " ".join(str(text).split())
    text = ENUM_PREFIX_RE.sub("", text)
    text = LEAD_IN_RE.sub("", text)

    if len(text) > 1 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()

    if len(text.split()) > MAX_ANSWER_WORDS:
        head = re.split(r"(?<=[.!?])\s+", text)[0]
        text = head if head.split() else text
        text = " ".join(text.split()[:MAX_ANSWER_WORDS])

    return text[:MAX_ANSWER_CHARS].strip()


def mmss(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_video(videos_dir: Path, video_id: str) -> Path:
    """The file a Video_ID names, per SESSION.2026-08-11.task1-pipeline.md §3.

    A released clip carries its ``.webm`` extension in the query file; a
    YouTube ID does not, and ``download_test_videos.py`` wrote those as
    ``<Video_ID>.mp4``. This function existing at all is the fix for the bug
    described in the module docstring -- resolving ``videos_dir / video_id``
    literally silently answered 868 queries with no video.
    """
    name = video_id if video_id.endswith(".webm") else video_id + ".mp4"
    return videos_dir / name


def load_durations(index_path: Path) -> dict[str, float]:
    """Video stem -> probed duration, from the frame index.

    The frames themselves are unused here -- this model takes video -- but the
    index carries a probed duration for all 1,940 clips on disk, and
    ``adaptive_fps()`` needs one *before* the request. Reading duration from
    the transcript alone (as this script originally did) leaves it unknown for
    the 1,080 queries whose video has no speech, and those then plan at a flat
    2 fps: harmless while the video was never attached, actively wrong now
    that it is.
    """
    if not index_path.exists():
        return {}

    try:
        videos = load_json(index_path).get("videos") or {}
    except (json.JSONDecodeError, OSError):
        return {}

    return {stem: entry["duration_sec"] for stem, entry in videos.items()
            if entry.get("duration_sec")}


def load_transcript(directory: Path, video_id: str) -> dict | None:
    """The suspect-filtered Whisper transcript, or None when there is none."""
    path = directory / (Path(video_id).stem + ".json")

    if not path.exists():
        return None

    try:
        data = load_json(path)
    except (json.JSONDecodeError, OSError):
        return None

    text = (data.get("transcript") or "").strip()

    if not text:
        return None

    language = data.get("language") \
        if data.get("language_source") == "window_sweep" else None

    return {"text": text, "language": language,
            "duration_sec": data.get("duration_sec")}


def load_cache(path: Path) -> dict[int, dict]:
    """Read the JSONL work cache, keyed by Q_ID. Later records win."""
    cache: dict[int, dict] = {}

    if not path.exists():
        return cache

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                cache[int(record["Q_ID"])] = record
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                print(f"  skipping malformed cache line {line_no}",
                      file=sys.stderr)

    return cache


def build_context(transcript: dict | None, video_exists: bool) -> str:
    """The block that precedes the question."""
    parts: list[str] = []

    if not video_exists:
        parts.append(
            "The video for this question is unavailable, so answer from "
            "the question and transcript alone, as plausibly as you can."
        )
    if transcript and transcript.get("text"):
        language = transcript.get("language") or "unknown"
        parts.append(
            f"Spoken audio (automatic transcript, language: {language}):\n"
            f"\"{transcript['text']}\""
        )
    elif video_exists:
        parts.append("The video has no detected speech.")

    return ("\n\n".join(parts) + "\n\n") if parts else ""


def adaptive_fps(duration_sec: float | None) -> float:
    """Target a fixed frame budget per video regardless of length, so a 14s
    clip and a 6.5-minute one both fit comfortably in the context window."""
    if duration_sec and duration_sec > 0:
        return min(MAX_FPS, max(MIN_FPS, TARGET_FRAMES / duration_sec))
    return 2.0


def video_mm_kwargs(duration_sec: float | None, max_model_len: int,
                    max_tokens: int) -> dict:
    """fps + pixel budget so long videos degrade gracefully instead of
    overflowing the context window (mirrors cosmos_reason2_utils' own
    online_inference() total_pixels formula)."""
    fps = adaptive_fps(duration_sec)
    max_seq_len = max(max_model_len - max_tokens, 1)
    total_pixels = int(max_seq_len * PIXELS_PER_TOKEN * 0.9)

    return {
        "fps": round(fps, 3),
        "do_sample_frames": True,
        "size": {
            "shortest_edge": MIN_VISION_TOKENS * PIXELS_PER_TOKEN,
            "longest_edge": total_pixels,
        },
    }


def probe_video(path: Path) -> dict | None:
    """Native width/height/fps/duration via ffprobe, or None if it fails."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        stream = (data.get("streams") or [{}])[0]
        num, _, den = (stream.get("r_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den) if float(den or 1) else 0.0
        duration = float((data.get("format") or {}).get("duration") or 0.0)

        return {"width": int(stream.get("width") or 0),
                "height": int(stream.get("height") or 0),
                "fps": fps, "duration": duration}
    except (subprocess.SubprocessError, ValueError, KeyError, IndexError,
            json.JSONDecodeError, OSError):
        return None


def ensure_safe_video(path: Path, target_fps: float, cache_dir: Path) -> Path:
    """Return a path safe for the server's num_frames=-1 full decode.

    Pre-shrinks the frame rate (via ffmpeg) for videos whose native
    duration x fps x width x height would exceed RAW_DECODE_BYTES_LIMIT,
    caching the result so repeat runs and multiple queries on one video
    don't re-transcode. Falls back to the original path if probing or
    transcoding fails -- degrading to the OOM risk rather than blocking.
    """
    info = probe_video(path)

    if not info or not info["width"] or not info["height"] or not info["fps"]:
        return path

    raw_bytes = info["duration"] * info["fps"] * info["width"] * info["height"] * 3

    if raw_bytes <= RAW_DECODE_BYTES_LIMIT:
        return path

    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_fps = round(min(target_fps, info["fps"]), 3) or 1.0
    cached = cache_dir / f"{path.stem}.f{safe_fps}.mp4"

    if cached.exists():
        return cached

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-vf", f"fps={safe_fps}",
             "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
             str(cached)],
            capture_output=True, text=True, timeout=600, check=True,
        )
        return cached
    except (subprocess.SubprocessError, OSError):
        return path


def extract_answers(content: str, k: int) -> list[str]:
    """Pull the ranked answer list out of the model's final-line JSON,
    tolerating prose the model may have added despite instructions."""
    candidates = [content] + JSON_BLOB_RE.findall(content)

    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

        answers = data.get("answers") if isinstance(data, dict) else None

        if isinstance(answers, list) and answers:
            return [tidy(a) for a in answers[:k] if str(a).strip()]

    return []


def dedupe(answers: list[str]) -> list[dict]:
    chosen: list[dict] = []
    seen: set[str] = set()

    for answer in answers:
        key = normalize(answer)

        if not answer or key in seen:
            continue

        seen.add(key)
        chosen.append({"answer": answer, "source": "model"})

    return chosen


def fill_slots(chosen: list[dict], want: int) -> tuple[list[dict], int]:
    """Pad a shortfall with generic, defensible fallback answers -- no
    rephrase-topup call, unlike generate.answers.py, since this design
    deliberately skips the ensemble machinery."""
    seen = {normalize(c["answer"]) for c in chosen}
    fallback_used = 0

    for text in FALLBACK_ANSWERS:
        if len(chosen) >= want:
            break

        key = normalize(text)

        if key in seen:
            continue

        seen.add(key)
        chosen.append({"answer": text, "source": "fallback"})
        fallback_used += 1

    return chosen[:want], fallback_used


class CosmosClient:
    """Thin wrapper over the OpenAI-compatible Cosmos-Reason2 endpoint."""

    def __init__(self, base_url: str, model: str, max_tokens: int,
                temperature: float, top_p: float, top_k: int,
                timeout: float, retries: int) -> None:
        self.client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.retries = retries
        self._max_model_len: int | None = None
        self._lock = threading.Lock()

    @property
    def max_model_len(self) -> int:
        with self._lock:
            if self._max_model_len is None:
                # vLLM's OpenAI server only exposes GET /v1/models (list),
                # not the per-id retrieve endpoint the SDK's .retrieve() uses.
                models = self.client.models.list()
                info = next((m for m in models.data if m.id == self.model),
                            next(iter(models.data), None))
                self._max_model_len = getattr(info, "max_model_len", None) \
                    or 16384
        return self._max_model_len

    def answer(self, question: str, video_path: Path | None,
              transcript: dict | None, k: int,
              duration: float | None = None) -> dict:
        video_exists = video_path is not None and video_path.exists()
        # Probed duration first, transcript second: the transcript has one only
        # when the video has speech.
        duration = duration or (transcript.get("duration_sec")
                                if transcript else None)
        mm_kwargs = video_mm_kwargs(duration, self.max_model_len, self.max_tokens)

        def call(include_video: bool):
            context = build_context(transcript, include_video)
            prompt = PROMPT_TEMPLATE.format(context=context, question=question, k=k)
            content_blocks = []

            if include_video:
                safe_path = ensure_safe_video(
                    video_path, adaptive_fps(duration),
                    video_path.parent / TRANSCODE_CACHE_DIRNAME,
                )
                content_blocks.append({
                    "type": "video_url",
                    "video_url": {"url": safe_path.resolve().as_uri()},
                })
            content_blocks.append({"type": "text", "text": prompt})

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content_blocks},
            ]

            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                extra_body={
                    "top_k": self.top_k,
                    "mm_processor_kwargs": mm_kwargs,
                },
            )

        include_video = video_exists
        last: Exception | None = None

        for attempt in range(self.retries):
            started = time.monotonic()

            try:
                completion = call(include_video)
                elapsed = time.monotonic() - started
                message = completion.choices[0].message
                content = message.content or ""
                reasoning = getattr(message, "reasoning_content", None) or ""

                return {
                    "content": content,
                    "reasoning": reasoning,
                    "elapsed": elapsed,
                    "video_exists": include_video,
                    "has_transcript": bool(transcript and transcript.get("text")),
                    "fps": mm_kwargs["fps"],
                }
            except BadRequestError as exc:
                last = exc

                if include_video:
                    # The HF processor choked on this specific video (seen in
                    # practice: a dropped/corrupt frame desyncs the decoded
                    # frame count from the container's metadata, which breaks
                    # Qwen3-VL's timestamp math). Fall back to
                    # transcript+question only rather than losing the query.
                    include_video = False
                    continue

                break  # already text-only and still failing -- not retryable
            except Exception as exc:  # noqa: BLE001 - transient, worth retrying
                last = exc
                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"{type(last).__name__}: {last}")


def answer_query(client: CosmosClient, query: dict, videos_dir: Path,
                 transcripts_dir: Path, want: int,
                 durations: dict[str, float] | None = None) -> dict:
    q_id = query["Q_ID"]
    video_id = query["Video_ID"]
    video_path = resolve_video(videos_dir, video_id)
    transcript = load_transcript(transcripts_dir, video_id)
    duration = (durations or {}).get(Path(video_id).stem)

    reply = client.answer(query["Question"], video_path, transcript, want,
                          duration)
    raw_answers = extract_answers(reply["content"], want)
    chosen = dedupe(raw_answers)
    chosen, fallback_used = fill_slots(chosen, want)

    share = reply["elapsed"] / max(want, 1)

    return {
        "Q_ID": q_id,
        "Video_ID": video_id,
        "answers": [
            {"Rank": rank, "Answer": item["answer"], "Time": round(share, 3),
             "source": item["source"]}
            for rank, item in enumerate(chosen, 1)
        ],
        "n_raw_candidates": len(raw_answers),
        "fallback": fallback_used,
        "video_exists": reply["video_exists"],
        "has_transcript": reply["has_transcript"],
        "fps": reply["fps"],
        "elapsed": round(reply["elapsed"], 2),
        "reasoning_chars": len(reply["reasoning"]),
        "content_parsed": bool(raw_answers),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def assemble(queries: list[dict], cache: dict[int, dict],
            want: int) -> tuple[list[dict], int]:
    run: list[dict] = []
    missing = 0

    for query in queries:
        record = cache.get(query["Q_ID"])

        if not record:
            missing += 1
            continue

        run.append({
            "Q_ID": query["Q_ID"],
            "Video_ID": query["Video_ID"],
            "Answers": [
                {"Rank": answer["Rank"], "Answer": answer["Answer"],
                 "Time": answer["Time"]}
                for answer in record["answers"][:want]
            ],
        })

    return run, missing


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Answer the Task 1 questions with Cosmos-Reason2-2B, "
                    "one reasoning call per query."
    )

    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    ap.add_argument("--frame-index", type=Path, default=DEFAULT_FRAME_INDEX,
                    help="read only for its probed durations, which set the "
                         "sampling fps (default: %(default)s)")
    ap.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)

    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--answers-per-query", type=int, default=ANSWERS_PER_QUERY)

    ap.add_argument("--limit", type=int, help="process at most N queries")
    ap.add_argument("--only", type=int, nargs="+", metavar="Q_ID")
    ap.add_argument("--force", action="store_true",
                    help="re-answer queries already in the cache")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the run file from the cache, no inference")

    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    queries = load_json(args.queries)
    cache = load_cache(args.cache)
    durations = load_durations(args.frame_index)

    missing_media = sum(
        1 for q in queries
        if not resolve_video(args.videos, q["Video_ID"]).exists()
    )

    print(f"queries: {len(queries)} | cached: "
          f"{sum(1 for q in queries if q['Q_ID'] in cache)} | "
          f"durations known: {len(durations)} | "
          f"queries with no video file: {missing_media}")

    if not args.rebuild:
        todo = [q for q in queries if args.force or q["Q_ID"] not in cache]

        if args.only:
            wanted = set(args.only)
            todo = [q for q in todo if q["Q_ID"] in wanted]

        if args.limit:
            todo = todo[:args.limit]

        print(f"to answer: {len(todo)} queries, {args.workers} worker(s) "
              f"against {args.base_url}")

        if todo:
            client = CosmosClient(args.base_url, args.model, args.max_tokens,
                                  args.temperature, DEFAULT_TOP_P,
                                  DEFAULT_TOP_K, args.timeout, args.retries)

            lock = threading.Lock()
            done = failed = 0
            started = time.monotonic()

            args.cache.parent.mkdir(parents=True, exist_ok=True)
            handle = args.cache.open("a", encoding="utf-8")

            def work(query: dict) -> None:
                nonlocal done, failed

                try:
                    record = answer_query(client, query, args.videos,
                                          args.transcripts,
                                          args.answers_per_query,
                                          durations)
                except Exception as exc:  # noqa: BLE001 - must not sink the run
                    with lock:
                        failed += 1
                        print(f"  FAILED Q_ID {query['Q_ID']}: "
                              f"{type(exc).__name__}: {exc}", file=sys.stderr)
                    return

                with lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    cache[record["Q_ID"]] = record

                    done += 1
                    rate = done / max(time.monotonic() - started, 1e-9)
                    remaining = (len(todo) - done) / max(rate, 1e-9) / 3600
                    flag = ""

                    if record["fallback"]:
                        flag = f" FALLBACK({record['fallback']})"
                    if not record["content_parsed"]:
                        flag += " UNPARSED"
                    if not record["video_exists"]:
                        flag += " NOVIDEO"

                    print(f"  [{done}/{len(todo)}] {rate * 60:5.1f}/min "
                          f"eta {remaining:4.1f}h  {record['elapsed']:6.1f}s"
                          f"{flag}  Q{record['Q_ID']}: "
                          f"{record['answers'][0]['Answer'][:48]}")

            try:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    list(pool.map(work, todo))
            except KeyboardInterrupt:
                print("\ninterrupted -- finished queries are cached, re-run "
                      "to resume", file=sys.stderr)
                return 130
            finally:
                handle.close()

            elapsed = time.monotonic() - started
            print(f"\nanswered {done} queries ({failed} failed) in "
                  f"{elapsed / 3600:.2f} h")

    run, missing = assemble(queries, cache, args.answers_per_query)

    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(run, handle, indent=4, ensure_ascii=False)
        handle.write("\n")

    print(f"\nwrote {len(run)} of {len(queries)} query entries to {args.out}")

    if missing:
        print(f"  {missing} queries are not answered yet -- the run will "
              f"not validate until they are")

    report(cache, queries)

    return 0


def report(cache: dict[int, dict], queries: list[dict]) -> None:
    records = [cache[q["Q_ID"]] for q in queries
              if q["Q_ID"] in cache and cache[q["Q_ID"]].get("answers")]

    if not records:
        return

    words = [len(answer["Answer"].split())
            for record in records for answer in record["answers"]]
    words.sort()

    unknown = sum(1 for r in records
                 if normalize(r["answers"][0]["Answer"]) == "unknown")

    print(f"\nquality checks over {len(records)} answered queries")
    print(f"  answer words  mean {sum(words) / len(words):4.1f}  "
          f"median {words[len(words) // 2]}  p90 {words[int(0.9 * len(words))]}"
          f"   (ground truth: mean 8.4, median 8, p90 17)")
    print(f"  queries needing fallback padding: "
          f"{sum(1 for r in records if r['fallback'])}")
    print(f"  queries whose JSON did not parse: "
          f"{sum(1 for r in records if not r['content_parsed'])}")
    print(f"  rank-1 answer is \"unknown\":         {unknown}")
    print(f"  queries answered without a video file: "
          f"{sum(1 for r in records if not r['video_exists'])}")
    print(f"  queries answered without a transcript: "
          f"{sum(1 for r in records if not r['has_transcript'])}")

    times = [r["elapsed"] for r in records]
    print(f"  wall time per query: mean {sum(times) / len(times):5.2f}s  "
          f"min {min(times):.2f}s  max {max(times):.2f}s  "
          f"total {sum(times) / 3600:.2f}h")


if __name__ == "__main__":
    sys.exit(main())

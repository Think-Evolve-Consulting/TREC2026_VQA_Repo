#!/usr/bin/env python3
"""Rank the Task 2 multiple-choice options with Cosmos-Reason2-2B.

The Cosmos counterpart to ``VQA/rank.options.mc.py`` (Gemma 4 over extracted
frames). The ranking design is identical -- the model returns a permutation of
option *numbers*, which is mapped back to the official option strings, so no
rank can be spent twice and no option can go unranked -- but everything about
how the video reaches the model differs, and so does the cost model:

**Cosmos takes video natively.** A ``file://`` path the server reads itself,
at an adaptive fps, rather than the 8-32 pre-extracted JPEG frames Gemma 4 is
shown. That means no frame index is needed for the pixels (only for its probed
durations, which set the fps), and it means ``resolve_video()`` matters: a
``Video_ID`` is *not* a filename. 868 of the 2,443 queries name a bare YouTube
ID stored as ``<Video_ID>.mp4`` beside the released ``.webm`` clips, and
resolving ``videos_dir / Video_ID`` literally is what left 1,098 of the entries
in ``cosmos.run.task1.json`` answered with no video at all -- a run that
validated cleanly and was 45% blind. Same convention, same fix, applied here
from the start.

**It reasons, and reasoning is the whole cost.** The server runs with
``--reasoning-parser qwen3``, so ``<think>...</think>`` arrives in
``reasoning_content`` and ``content`` holds only the final JSON line. Stage one
is deliberately *not* grammar-constrained: a ``response_format`` schema applies
from the first token and would delete the chain of thought that is the reason
to run this model at all. The reply is parsed instead, and
``extract_ranking()`` salvages a bare "3, 1, 4, 2" from the tail of either
field when the JSON does not appear -- 460 of 2,446 Task 1 replies (19%) failed
to yield parseable JSON, and a lost reply here costs a whole query rather than
one of ten prompts.

**A reasoning model chooses; it does not rank.** Measured over the first smoke
batch of 20, nine replies were a bare ``{"ranking": [4]}``: the think block
converges on "the correct answer is 4" and stops, and no amount of prompt
insistence fixed it (asking for all N numbers three different ways moved it
from 1/6 to 3/6 full-length rankings). Padding the rest in presentation order
would fill ranks 2..N -- most of what MRR scores -- with arbitrary options. So
whenever stage one leaves options unplaced, ``order_remaining()`` runs a second
call that is handed the model's own pick and its own reasoning, carries no
video, and *is* schema-constrained to a full permutation. It measured 0.6 s
against the 17 s of the reasoning call, so the fix is close to free, and the
model's own top-1 choice is preserved exactly.

**One pass, not three.** The Gemma 4 pass ranks each query three times with
the options presented in a different order and combines them by Borda count.
Cosmos costs ~27 s per call *with the video absent from 45% of them*; three
passes over a fully video-attached corpus would be a multi-day run. So
``--passes`` defaults to 1, and the single presentation is a per-query seeded
shuffle rather than the official order. Measured against the 2025 MC release,
the correct option sits at position 1/2/3/4 in 524/505/492/475 of 1,996
queries -- i.e. uniformly -- so official order carries no answer-key leak and
the shuffle is not protecting against one. It spreads whatever position bias
the model has across the options instead of letting it fall on the same
position for every query, and it decorrelates this run's presentation from the
Gemma 4 run's for a later fusion. Raise ``--passes`` if the wall clock allows;
the Borda machinery is the same one.

Run this **from inside WSL2**, with the server's own venv Python -- the server
reads the videos by path under ``--allowed-local-media-path /mnt/e/TREC_2026``:

    /home/aakash/cosmos-reason2/.venv/bin/python3 \
        /mnt/e/TREC_2026/VQA/rank.options.mc.cosmos.py --limit 20

    ... --rebuild        # re-emit the run file from the cache, no inference
    ... --only 1 3 264   # named Q_IDs
    ... --passes 3       # the Gemma 4 ensemble, at three times the cost

The GPU holds one model at a time: ``systemctl --user stop gemma4-vllm`` then
``systemctl --user start cosmos-reason2``. See VQA/COSMOS_REASON2_SETUP.md.
"""

from __future__ import annotations

import argparse
import json
import random
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
DEFAULT_QUERIES = ROOT / "vqa.2026.testingDataset.Task2_v2.json"
DEFAULT_VIDEOS = ROOT / "videos_TEST_VQA" / "videos"
DEFAULT_FRAME_INDEX = ROOT / "videos_TEST_VQA" / "frames" / "index.json"
DEFAULT_TRANSCRIPTS = ROOT / "videos_TEST_VQA" / "transcription_openwhisper"
DEFAULT_OUT = ROOT / "cosmos.run.task2.json"
DEFAULT_CACHE = ROOT / "rankings.task2.cosmos.cache.jsonl"

DEFAULT_MODEL = "nvidia/Cosmos-Reason2-2B"
DEFAULT_BASE_URL = "http://localhost:8008/v1"

# vLLM's video IO layer runs with --media-io-kwargs num_frames=-1, needed for
# correct per-frame timestamps, which makes it decode every native-resolution
# frame into raw numpy *before* fps subsampling. A 320 s, 2048x1152, 24 fps
# clip in this corpus hit a 50.6 GiB allocation and crashed the request, so
# videos whose raw decode footprint would exceed this are pre-shrunk with
# ffmpeg first. Same guard and cache directory as the Task 1 pass, but half
# its 4 GiB ceiling: this guard is the only thing bounding what one in-flight
# request costs in host RAM, and WSL2 has ~24 GiB free for all of them.
RAW_DECODE_BYTES_LIMIT = 2 * 1024**3
TRANSCODE_CACHE_DIRNAME = ".cosmos_transcoded_cache"

# The no-sampling retry (see CosmosClient.rank) hands the processor every
# frame the file holds, so on that path the file itself has to carry the frame
# budget: it is always transcoded down to the target fps first, whatever its
# length. The first attempt at this only transcoded clips over 30 s, on the
# reasoning that a 10 s clip costs a mere ~9k prompt tokens as-is -- true, but
# irrelevant, because the cost that matters is host RAM, not context. 300
# native frames resized in the processor instead of 40 sampled ones is what
# put the server over the edge: 8 workers on that path OOM-killed it 790
# queries into the run.
TRANSCODE_ON_RETRY_ALWAYS = True

# Enough for a real think block plus one short permutation. The Task 1 pass
# needs 8192 because it emits ten sentence-length answers after reasoning;
# here the answer is "[3, 1, 4, 2]" and the reasoning is the only variable
# part (median 1,184 characters there, ~300 tokens). 4096 leaves a wide
# margin, and a smaller budget is a smaller worst case per call.
DEFAULT_MAX_TOKENS = 4096

DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20

# See the module docstring: three passes is a multi-day run on this model.
DEFAULT_PASSES = 1

# Two kept the queue fed for the Task 1 pass, where 45% of calls carried no
# video at all. Every call here carries one, and the decode is CPU-bound in
# the server process rather than GPU work (measured: 36% GPU utilisation at
# 12 workers), so throughput keeps climbing well past two: 20 queries took
# 144 s at 4 workers, 119 s at 8, 100 s at 12, and 24 took 108 s at 16.
#
# Four is the default anyway, because the ceiling here is host RAM, not speed.
# num_frames=-1 decodes each video whole before subsampling, only clips over
# RAW_DECODE_BYTES_LIMIT are pre-shrunk, and WSL2 has ~24 GiB free for every
# in-flight request combined. Eight was tried and did not survive: the kernel
# OOM-killed the server 790 queries in ("Failed with result 'oom-kill'"),
# which cost ten queries to fallback rankings before the run was stopped. The
# arithmetic was there to be done beforehand -- eight worst-case decodes
# exceeded the available memory on paper -- and the half-hour that eight
# workers saves is not worth re-running the pass.
DEFAULT_WORKERS = 4

# Visual-token pixel budget, confirmed against
# cosmos_reason2_utils/vision.py: PIXELS_PER_TOKEN = (16 * 2) ** 2.
PIXELS_PER_TOKEN = 1024
MIN_VISION_TOKENS = 256

# Adaptive fps: a fixed frame budget per video regardless of length, so a 14 s
# clip and a 6.5-minute one both fit the window. Durations actually seen in
# videos_TEST_VQA (n=1935): median 10.3 s, p90 56 s, p99 233 s, max 389 s.
TARGET_FRAMES = 64
MAX_FPS = 4.0
MIN_FPS = 0.2

SYSTEM_PROMPT = "You are a helpful assistant."

PROMPT_TEMPLATE = """{context}Question: {question}

Answer options:
{options}

Rank ALL {n} options from the most likely correct answer to the least likely.
This is a ranking task, not a single-choice task: even the options you rule
out must be placed in order, best of the rest first.

Rules:
- Judge each option against the video and the transcript, not against how it
  is worded.
- Your ranking must contain exactly {n} numbers. A reply with fewer than {n}
  numbers is wrong, even if the first one is correct.
- Every number from 1 to {n} appears exactly once: never repeat one, never
  leave one out.
- Exactly one option is correct. Put it first, then the next most plausible,
  and the option the video contradicts most clearly last.

Answer using the following format:

<think>
Your reasoning about what the video and transcript show, which option is
correct, and how plausible each of the others is.
</think>

Then, on the final line, write ONLY this JSON, listing all {n} numbers:
{{"ranking": [{slots}]}}

Check before answering: your list must have {n} numbers in it."""

# Stage two. Cosmos-Reason2 is a reasoning model that answers multiple choice
# by *choosing*: measured over the first smoke batch, 9 of 20 replies were a
# bare {"ranking": [4]} -- the reasoning converges on "the correct answer is
# 4" and stops. Padding the other seven slots in presentation order would put
# arbitrary text in ranks 2..8, which is most of the MRR the task scores. So
# whenever stage one leaves options unplaced, they are ordered by a second,
# text-only call that is handed the model's own pick and its own reasoning.
# It carries no video, is grammar-constrained to a full permutation, and cost
# 0.6 s against the 17 s of the reasoning call in the same test.
STAGE2_TEMPLATE = """A video question was answered by a reasoning model. Your
job is to order the answer options it did not place.

Question: {question}

Answer options:
{options}

{known}The model's reasoning:
"{why}"

Rank all {n} options from most to least likely.{keep}
Order the rest by how plausible each is given the reasoning above.

Reply with JSON only: {{"ranking": [{slots}]}} listing all {n} numbers."""

JSON_BLOB_RE = re.compile(r"\{.*?\}", re.DOTALL)
# The salvage path: a bare list of option numbers, bracketed or not, anywhere
# in the tail of the reply. Requires at least two numbers so a stray "option
# 3" in prose cannot masquerade as a ranking.
DIGIT_LIST_RE = re.compile(r"\d+(?:\s*[,\s]\s*\d+){1,}")


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def ranking_schema(n: int) -> dict:
    """Constrain a stage-two reply to exactly `n` integers in range.

    Only stage two is grammar-constrained. Stage one must not be: a schema
    applies from the first token, which would delete the ``<think>`` block
    that is the entire reason to run a reasoning model.
    """
    return {
        "type": "object",
        "properties": {
            "ranking": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": n},
                "minItems": n,
                "maxItems": n,
            },
        },
        "required": ["ranking"],
        "additionalProperties": False,
    }


def seed_for(q_id: int, index: int) -> int:
    """Stable per-call seed, so a re-run reproduces the same presentation."""
    return (q_id * 1_000_003 + index * 7919) % (2**31 - 1)


def presentation_orders(n: int, passes: int, q_id: int,
                        first: str) -> list[list[int]]:
    """The option orders the passes present, as index lists into `options`.

    With ``first="shuffled"`` (the default here, and the difference from the
    Gemma 4 pass) even a single pass gets a per-query permutation, so a model
    that favours "the option shown first" spreads that preference over the
    options instead of concentrating it on option 1 of every query. Further
    passes add the reverse of that order, then more shuffles -- the reverse
    second because between an order and its exact reverse, every option that
    was shown first is shown last.
    """
    rng = random.Random(seed_for(q_id, 500))
    base = list(range(n))

    if first == "shuffled":
        rng.shuffle(base)

    orders = [base]

    if passes > 1:
        orders.append(list(reversed(base)))

    while len(orders) < passes:
        order = list(range(n))
        rng.shuffle(order)

        if order not in orders or len(orders) > 2 * n:
            orders.append(order)

    return orders[:passes]


def render_options(options: list[str], order: list[int]) -> str:
    """The numbered option block, in this pass's presentation order."""
    return "\n".join(f"{position}. {options[index]}"
                     for position, index in enumerate(order, 1))


def extract_ranking(content: str, reasoning: str, n: int) -> tuple[list[int], str]:
    """Pull a ranking of presented positions out of the reply.

    Returns the raw numbers the model gave (1-based, unvalidated) and how they
    were found. Three routes, in descending order of trust:

    ``json``    -- the final-line ``{"ranking": [...]}`` the prompt asks for.
    ``digits``  -- a bare "3, 1, 4, 2" in the tail of ``content``. The Task 1
                   pass lost 19% of its replies to unparseable content; there,
                   one lost reply cost one of ten prompts, here it would cost
                   the entire query, so a plainly-stated order is taken.
    ``think``   -- the same salvage over the reasoning block, for a reply that
                   was cut off after deciding but before writing the JSON.

    An empty list means nothing usable was in the reply at all.
    """
    for text, route in ((content, "json"),):
        for blob in reversed(JSON_BLOB_RE.findall(text or "")):
            try:
                data = json.loads(blob)
            except (json.JSONDecodeError, TypeError):
                continue

            values = data.get("ranking") if isinstance(data, dict) else None

            if isinstance(values, list) and values:
                return values, route

    for text, route in ((content, "digits"), (reasoning, "think")):
        matches = DIGIT_LIST_RE.findall(text or "")

        for match in reversed(matches):
            values = [int(v) for v in re.findall(r"\d+", match)]
            # A ranking names every option once; a shorter list is some other
            # number the model happened to write down (a timestamp, a count).
            if len([v for v in set(values) if 1 <= v <= n]) == n:
                return values, route

    return [], "none"


def as_permutation(raw: list, n: int) -> tuple[list[int], bool]:
    """Coerce a reply's ranking into a permutation of 0..n-1 (presented order).

    Returns the positions the model gave, best first, and whether anything had
    to be repaired: repeats keep their first, best occurrence, and whatever was
    dropped is appended in the order it was shown.
    """
    seen: set[int] = set()
    ordered: list[int] = []
    repaired = False

    for value in raw or []:
        try:
            index = int(str(value).strip().rstrip(".")) - 1
        except (TypeError, ValueError):
            repaired = True
            continue

        if index < 0 or index >= n or index in seen:
            repaired = True
            continue

        seen.add(index)
        ordered.append(index)

    if len(ordered) != n:
        repaired = True
        ordered += [index for index in range(n) if index not in seen]

    return ordered, repaired


def borda(passes: list[dict], n: int) -> list[dict]:
    """Combine the passes into one ranking over the original option indices.

    Identical to the Gemma 4 pass, minus the confidence weighting: this model
    is not asked for a confidence -- the reasoning trace is what it spends its
    tokens on -- so every pass carries the same weight. With --passes 1 this
    is just a re-labelling of the single ranking.
    """
    points = [0.0] * n
    ranks: list[list[int]] = [[] for _ in range(n)]
    firsts = [0] * n

    for entry in passes:
        for position, option_index in enumerate(entry["ranking"]):
            points[option_index] += n - position
            ranks[option_index].append(position + 1)

        if entry["ranking"]:
            firsts[entry["ranking"][0]] += 1

    scored = [
        {
            "option_index": index,
            "score": round(points[index], 4),
            "mean_rank": round(sum(ranks[index]) / len(ranks[index]), 3)
            if ranks[index] else None,
            "ranks": ranks[index],
            "firsts": firsts[index],
        }
        for index in range(n)
    ]

    return sorted(scored, key=lambda item: (-item["score"],
                                            item["mean_rank"] if
                                            item["mean_rank"] is not None
                                            else n + 1,
                                            item["option_index"]))


def check_permutation(answers: list[str], options: list[str], q_id: int) -> None:
    """Refuse to emit anything the MC validator would reject."""
    if len(answers) != len(options) or sorted(answers) != sorted(options):
        raise RuntimeError(
            f"Q_ID {q_id}: ranking is not a permutation of the official "
            f"options ({len(answers)} answers, {len(set(answers))} distinct, "
            f"{len(options)} options)")


def resolve_video(videos_dir: Path, video_id: str) -> Path:
    """The file a Video_ID names, per SESSION.2026-08-11.task1-pipeline.md §3.

    A released clip carries its ``.webm`` extension in the query file; a
    YouTube ID does not, and ``download_test_videos.py`` wrote those as
    ``<Video_ID>.mp4``. Resolving ``videos_dir / video_id`` literally is the
    bug that made the Task 1 Cosmos run 45% blind.
    """
    name = video_id if video_id.endswith(".webm") else video_id + ".mp4"
    return videos_dir / name


def load_durations(index_path: Path) -> dict[str, float]:
    """Video stem -> probed duration, from the frame index.

    The frames are unused here -- this model takes video -- but the index
    carries a probed duration for all 1,940 clips on disk, and the sampling
    fps has to be chosen before the request. Reading duration from the
    transcript alone leaves it unknown for every video without speech.
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

    # language_source == "whisper_default" means the sweep abstained and the
    # label was read off the first 30 s including music -- not worth asserting.
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
            "The video for this question is unavailable, so judge the options "
            "from the question and transcript alone, as plausibly as you can."
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
    """A fixed frame budget per video regardless of length."""
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


def ensure_safe_video(path: Path, target_fps: float, cache_dir: Path,
                      force: bool = False) -> Path:
    """Return a path safe for the server's num_frames=-1 full decode.

    Pre-shrinks the frame rate for videos whose native
    duration x fps x width x height would exceed RAW_DECODE_BYTES_LIMIT,
    caching the result so repeat runs, extra passes and multiple queries on
    one video do not re-transcode. Falls back to the original path if probing
    or transcoding fails -- degrading to the OOM risk rather than blocking.

    ``force`` shrinks regardless of size. It exists for the do_sample_frames
    retry in CosmosClient.rank: that path hands the processor whatever frames
    the file contains, so the frame count has to come from the file itself.
    """
    info = probe_video(path)

    if not info or not info["width"] or not info["height"] or not info["fps"]:
        return path

    raw_bytes = info["duration"] * info["fps"] * info["width"] * info["height"] * 3

    if raw_bytes <= RAW_DECODE_BYTES_LIMIT and not force:
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


class CosmosClient:
    """Thin wrapper over the OpenAI-compatible Cosmos-Reason2 endpoint."""

    name = "cosmos"

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
                # vLLM's OpenAI server only exposes GET /v1/models (list), not
                # the per-id retrieve endpoint the SDK's .retrieve() uses.
                models = self.client.models.list()
                info = next((m for m in models.data if m.id == self.model),
                            next(iter(models.data), None))
                self._max_model_len = getattr(info, "max_model_len", None) \
                    or 24576
        return self._max_model_len

    def rank(self, prompt_context: str, question: str, options: list[str],
             order: list[int], video_path: Path | None,
             duration: float | None) -> dict:
        """One ranking call. Returns the raw reply plus what it cost."""
        video_exists = video_path is not None and video_path.exists()
        mm_kwargs = video_mm_kwargs(duration, self.max_model_len,
                                    self.max_tokens)
        n = len(options)

        def call(include_video: bool, sample_frames: bool = True):
            context = build_context_for(prompt_context, include_video)
            prompt = PROMPT_TEMPLATE.format(
                context=context,
                question=question,
                options=render_options(options, order),
                n=n,
                slots=", ".join(str(i) for i in range(1, n + 1)),
            )
            content_blocks = []

            if include_video:
                # Without frame sampling the processor uses every frame the
                # file holds, so the clip is shrunk on disk first -- at 30 fps
                # even a 10 s video is 300 frames, and resizing those instead
                # of 40 sampled ones is what OOM-killed the server mid-run.
                safe_path = ensure_safe_video(
                    video_path, adaptive_fps(duration),
                    video_path.parent / TRANSCODE_CACHE_DIRNAME,
                    force=(not sample_frames and TRANSCODE_ON_RETRY_ALWAYS),
                )
                content_blocks.append({
                    "type": "video_url",
                    "video_url": {"url": safe_path.resolve().as_uri()},
                })
            content_blocks.append({"type": "text", "text": prompt})

            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content_blocks},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                extra_body={
                    "top_k": self.top_k,
                    "mm_processor_kwargs": (
                        mm_kwargs if sample_frames
                        else {k: v for k, v in mm_kwargs.items()
                              if k != "do_sample_frames"}),
                },
            )

        include_video = video_exists
        sample_frames = True
        last: Exception | None = None

        # One more attempt than --retries: the do_sample_frames retry below is
        # a different request, not the same one again, and it should not eat
        # the budget meant for transient failures.
        for attempt in range(self.retries + 1):
            started = time.monotonic()

            try:
                completion = call(include_video, sample_frames)
                elapsed = time.monotonic() - started
                message = completion.choices[0].message
                content = message.content or ""
                reasoning = getattr(message, "reasoning_content", None) or ""

                return {
                    "content": content,
                    "reasoning": reasoning,
                    "elapsed": elapsed,
                    "video_used": include_video,
                    "sampled_frames": sample_frames,
                    "fps": mm_kwargs["fps"],
                    "finish_reason": completion.choices[0].finish_reason,
                }
            except BadRequestError as exc:
                last = exc

                if include_video and sample_frames:
                    # The processor refused to re-sample this clip. The server
                    # runs with --media-io-kwargs num_frames=-1, so it has
                    # already decoded every frame and handed the processor an
                    # array; asking it to sample again from container metadata
                    # that disagrees with that array is what raises. Measured
                    # over the first 830 queries of the full run this hit 190
                    # of them (23%), all short .webm clips at 29.97 fps, and
                    # each silently became a text-only answer -- the same
                    # class of quiet blindness as the Video_ID bug. The
                    # identical clip processes fine without do_sample_frames
                    # (verified on 7270.webm and 6476.webm: 8.7k and 9.9k
                    # prompt tokens, 2-3 s), so retry that way before giving
                    # up the pixels.
                    sample_frames = False
                    continue

                if include_video:
                    # Still refused with the video attached. Fall back to
                    # transcript+question rather than losing the query.
                    include_video = False
                    sample_frames = True
                    continue

                break  # already text-only and still failing -- not retryable
            except Exception as exc:  # noqa: BLE001 - transient, worth retrying
                last = exc

                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"{type(last).__name__}: {last}")

    def order_remaining(self, question: str, options: list[str],
                        order: list[int], placed: list[int],
                        why: str) -> tuple[list[int], float]:
        """Stage two: order the options stage one left unplaced.

        ``placed`` holds the presented positions stage one did rank, in its
        order, and is kept at the front. No video is attached -- the visual
        judgement already happened, and re-decoding the clip to sort losers
        would cost as much as the reasoning call itself. Returns the raw
        numbers and the wall time.
        """
        n = len(options)
        kept = ", ".join(str(p + 1) for p in placed)
        prompt = STAGE2_TEMPLATE.format(
            question=question,
            options=render_options(options, order),
            known=(f"The model already placed, best first: {kept}.\n\n"
                   if placed else ""),
            why=(why or "(the model gave no usable reasoning)")[-600:],
            keep=(f" Keep {kept} in that order at the front."
                  if placed else ""),
            n=n,
            slots=", ".join(str(i) for i in range(1, n + 1)),
        )

        started = time.monotonic()

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You reply with JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.2,
            top_p=self.top_p,
            extra_body={
                "top_k": self.top_k,
                # No think block: this stage is bookkeeping over a judgement
                # that has already been made, and the schema below could not
                # coexist with one anyway.
                "chat_template_kwargs": {"enable_thinking": False},
            },
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "ranking", "schema": ranking_schema(n)},
            },
        )

        elapsed = time.monotonic() - started
        content = completion.choices[0].message.content or ""
        values, _ = extract_ranking(content, "", n)

        return values, elapsed


def build_context_for(transcript_block: str, video_exists: bool) -> str:
    """Wrap the pre-rendered transcript block with the right video wording.

    The transcript text is identical across a query's passes, so it is
    rendered once by the caller; only the "no video" sentence depends on what
    this particular call is actually sending.
    """
    if video_exists:
        return transcript_block

    lead = ("The video for this question is unavailable, so judge the options "
            "from the question and transcript alone, as plausibly as you can.")

    return f"{lead}\n\n{transcript_block}" if transcript_block else lead + "\n\n"


def rank_query(client: CosmosClient, query: dict, options: list[str],
               videos_dir: Path, transcripts_dir: Path,
               durations: dict[str, float], passes: int,
               first_order: str) -> dict:
    """Rank one query's options, over however many presentations are asked."""
    q_id = query["Q_ID"]
    video_id = query["Video_ID"]
    n = len(options)

    video_path = resolve_video(videos_dir, video_id)
    video_exists = video_path.exists()
    transcript = load_transcript(transcripts_dir, video_id)
    # Probed duration first, transcript second: the transcript carries one
    # only when the video has speech.
    duration = durations.get(Path(video_id).stem) or (
        transcript.get("duration_sec") if transcript else None)

    transcript_block = build_context(transcript, video_exists=True)

    results: list[dict] = []
    failures = 0
    repairs = 0
    routes: list[str] = []

    for index, order in enumerate(presentation_orders(n, passes, q_id,
                                                      first_order)):
        try:
            reply = client.rank(transcript_block, query["Question"], options,
                                order, video_path if video_exists else None,
                                duration)
        except RuntimeError as exc:
            failures += 1
            print(f"  Q_ID {q_id} pass {index}: {exc}", file=sys.stderr)
            continue

        raw, route = extract_ranking(reply["content"], reply["reasoning"], n)
        routes.append(route)

        # What stage one actually placed: its own numbers, valid and deduped,
        # as presented positions. For this model that is usually a single
        # choice rather than a ranking.
        placed: list[int] = []

        for value in raw:
            try:
                option = int(str(value).strip().rstrip(".")) - 1
            except (TypeError, ValueError):
                continue

            if 0 <= option < n and option not in placed:
                placed.append(option)

        stage2_elapsed = 0.0
        used_stage2 = False

        if len(placed) < n:
            # Order the remainder from the model's own reasoning rather than
            # padding it with the presentation order, which would put
            # arbitrary options in the ranks that carry the MRR.
            used_stage2 = True

            try:
                values, stage2_elapsed = client.order_remaining(
                    query["Question"], options, order, placed,
                    reply["reasoning"] or reply["content"])
                # Stage two re-lists the placed option along with the rest,
                # so drop its duplicate here rather than letting
                # as_permutation call it a repair -- that counter is meant to
                # mean the model contradicted itself, not that this design
                # asked the same question twice.
                raw = [p + 1 for p in placed] + [
                    v for v in values
                    if not (isinstance(v, int) and (v - 1) in placed)]
            except Exception as exc:  # noqa: BLE001 - stage two is a bonus
                print(f"  Q_ID {q_id} pass {index} stage 2: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                stage2_elapsed = 0.0

        if not placed and not raw:
            # Stage one said nothing usable and stage two did not recover it.
            # Padding now would record the presentation order as if the model
            # had chosen it, so the pass is dropped instead; the query falls
            # back only if every pass does the same.
            failures += 1
            continue

        positions, repaired = as_permutation(raw, n)
        repairs += int(repaired)

        results.append({
            "index": index,
            # Presented positions map back to official option indices here,
            # once, so nothing downstream has to know the shuffle existed.
            "ranking": [order[position] for position in positions],
            "order": order,
            "route": route,
            "repaired": repaired,
            "stage1_placed": len(placed),
            "stage2": used_stage2,
            "video_used": reply["video_used"],
            "finish_reason": reply["finish_reason"],
            "reasoning_chars": len(reply["reasoning"]),
            "elapsed": round(reply["elapsed"] + stage2_elapsed, 3),
            "stage2_elapsed": round(stage2_elapsed, 3),
        })

    if results:
        ranked = borda(results, n)
        fallback = False
    else:
        # Every pass failed or came back unusable. The options keep their
        # official order -- no better than the dummy run for this query --
        # flagged here, counted by report(), re-runnable with --force --only.
        ranked = [{"option_index": index, "score": 0.0, "mean_rank": None,
                   "ranks": [], "firsts": 0} for index in range(n)]
        fallback = True

    answers = [{"Rank": rank, "Answer": options[item["option_index"]]}
               for rank, item in enumerate(ranked, 1)]

    check_permutation([a["Answer"] for a in answers], options, q_id)

    return {
        "Q_ID": q_id,
        "Video_ID": video_id,
        "answers": answers,
        # The official index of each option, best first, plus its evidence --
        # what a cross-model fusion needs without the option text.
        "ranked": ranked,
        "passes": results,
        "backend": client.name,
        "model": client.model,
        "n_options": n,
        "n_passes": passes,
        "n_replies": len(results),
        "n_pass_failures": failures,
        "n_repaired": repairs,
        "routes": routes,
        "fallback": fallback,
        "video_exists": video_exists,
        "video_used": any(r["video_used"] for r in results),
        "has_transcript": bool(transcript and transcript.get("text")),
        "fps": adaptive_fps(duration),
        "agreement": round(max((item["firsts"] for item in ranked), default=0)
                           / max(len(results), 1), 4),
        "reasoning_chars": sum(r["reasoning_chars"] for r in results),
        "elapsed": round(sum(r["elapsed"] for r in results), 2),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def preflight(base_url: str, model: str) -> None:
    """Refuse to start unless the server is up and serving this model.

    A dead server is not a loud failure here: every pass fails, every query
    keeps its official option order, and the run completes looking finished
    while being exactly the dummy run.
    """
    try:
        served = [m.id for m in
                  OpenAI(api_key="EMPTY", base_url=base_url,
                         timeout=10.0).models.list().data]
    except Exception as exc:  # noqa: BLE001 - any failure here is fatal
        raise SystemExit(
            f"no server answering at {base_url} "
            f"({type(exc).__name__}: {exc}).\n"
            f"  the GPU holds one model at a time:\n"
            f"    systemctl --user stop gemma4-vllm\n"
            f"    systemctl --user start cosmos-reason2\n"
            f"  and keep a WSL2 session open for the whole run -- see "
            f"VQA/COSMOS_REASON2_SETUP.md")

    if model not in served:
        raise SystemExit(f"{base_url} serves {served}, not {model!r}. Pass "
                         f"--model to match, or restart the server.")


def load_queries(path: Path) -> list[dict]:
    """The released MC queries, with Options flattened to a list of strings."""
    return [{"Q_ID": q["Q_ID"], "Video_ID": q["Video_ID"],
             "Question": q["Question"],
             "Options": [option["option"] for option in q["Options"]]}
            for q in load_json(path)]


def assemble(queries: list[dict], cache: dict[int, dict]) -> tuple[list[dict], int]:
    """Emit the run file, carrying only the fields the MC validator allows.

    A cached ranking is re-checked against the released options rather than
    copied blindly, so a cache written against an earlier release of the query
    file fails here instead of at submission.
    """
    run: list[dict] = []
    missing = 0

    for query in queries:
        record = cache.get(query["Q_ID"])

        if not record:
            missing += 1
            continue

        answers = [{"Rank": answer["Rank"], "Answer": answer["Answer"]}
                   for answer in record["answers"]]

        check_permutation([a["Answer"] for a in answers], query["Options"],
                          query["Q_ID"])

        run.append({"Q_ID": query["Q_ID"], "Video_ID": query["Video_ID"],
                    "Answers": answers})

    return run, missing


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Rank the Task 2 multiple-choice options with "
                    "Cosmos-Reason2-2B over native video."
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
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    ap.add_argument("--passes", type=int, default=DEFAULT_PASSES,
                    help="rankings per query, each with the options presented "
                         "in a different order. This model reasons, so every "
                         "pass costs a full call (default: %(default)s)")
    ap.add_argument("--order", choices=("shuffled", "official"),
                    default="shuffled",
                    help="how the first pass presents the options "
                         "(default: %(default)s -- a per-query seeded shuffle)")

    ap.add_argument("--limit", type=int, help="process at most N queries")
    ap.add_argument("--only", type=int, nargs="+", metavar="Q_ID")
    ap.add_argument("--force", action="store_true",
                    help="re-rank queries already in the cache")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the run file from the cache, no inference")

    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.passes < 1:
        print("--passes must be at least 1", file=sys.stderr)
        return 2

    queries = load_queries(args.queries)
    cache = load_cache(args.cache)
    durations = load_durations(args.frame_index)

    missing_media = sum(1 for q in queries
                        if not resolve_video(args.videos, q["Video_ID"]).exists())

    print(f"model: {args.model} at {args.base_url} | {args.workers} worker(s)")
    print(f"cache: {args.cache.name} | run: {args.out.name}")
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

        # Sort by video so a query's passes, and the queries sharing a clip,
        # land near each other: the transcode cache and the server's prefix
        # cache both pay off that way.
        todo.sort(key=lambda q: (q["Video_ID"], q["Q_ID"]))

        print(f"to rank: {len(todo)} queries, {args.passes} pass(es) each, "
              f"first presentation {args.order}")

        if todo:
            preflight(args.base_url, args.model)

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
                    record = rank_query(client, query, query["Options"],
                                        args.videos, args.transcripts,
                                        durations, args.passes, args.order)
                except Exception as exc:  # noqa: BLE001 - one bad query must
                    with lock:            # not sink a long run
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
                        flag = " FALLBACK"
                    elif record["n_pass_failures"]:
                        flag = f" ERR({record['n_pass_failures']})"
                    elif record["n_repaired"]:
                        flag = f" FIX({record['n_repaired']})"

                    if not record["video_used"]:
                        flag += " NOVIDEO"
                    if any(route in ("digits", "think")
                           for route in record["routes"]):
                        flag += " SALVAGE"

                    print(f"  [{done}/{len(todo)}] {rate * 60:5.1f}/min "
                          f"eta {remaining:4.1f}h  "
                          f"{record['elapsed']:6.1f}s "
                          f"{record['fps']:4.2f}fps{flag}  "
                          f"Q{record['Q_ID']}: "
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
            print(f"\nranked {done} queries ({failed} failed) in "
                  f"{elapsed / 3600:.2f} h")

    run, missing = assemble(queries, cache)

    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(run, handle, indent=4, ensure_ascii=False)
        handle.write("\n")

    print(f"\nwrote {len(run)} of {len(queries)} query entries to {args.out}")

    if missing:
        print(f"  {missing} queries are not ranked yet -- the run will not "
              f"validate until they are")

    report(cache, queries)

    return 0


def report(cache: dict[int, dict], queries: list[dict]) -> None:
    """Sanity checks the validator does not make.

    The two silent failure modes are a dead server (every query keeps its
    official option order and the run still validates) and a model whose
    replies never parse (same outcome by a different route), so the fallback,
    salvage and position counters are printed even when they are zero.
    """
    records = [cache[q["Q_ID"]] for q in queries
               if q["Q_ID"] in cache and cache[q["Q_ID"]].get("answers")]

    if not records:
        return

    by_qid = {q["Q_ID"]: q for q in queries}
    first_official = last_official = counted = 0

    for record in records:
        query = by_qid.get(record["Q_ID"])

        if not query or not record.get("ranked"):
            continue

        winner = record["ranked"][0]["option_index"]
        counted += 1
        first_official += int(winner == 0)
        last_official += int(winner == len(query["Options"]) - 1)

    routes = {}

    for record in records:
        for route in record.get("routes") or ():
            routes[route] = routes.get(route, 0) + 1

    print(f"\nquality checks over {len(records)} ranked queries")
    print(f"  passes completed per query, mean "
          f"{sum(r['n_replies'] for r in records) / len(records):.2f}")
    print(f"  reply parsed from: "
          f"{', '.join(f'{k} {v}' for k, v in sorted(routes.items()))}")

    if counted:
        expected = sum(1 / len(by_qid[r["Q_ID"]]["Options"]) for r in records
                       if r["Q_ID"] in by_qid) / counted
        print(f"  rank 1 is the officially FIRST option: {first_official} "
              f"({100 * first_official / counted:.1f}%, chance "
              f"{100 * expected:.1f}%)")
        print(f"  rank 1 is the officially LAST option:  {last_official} "
              f"({100 * last_official / counted:.1f}%)")

    multi = [r for r in records if r["n_replies"] > 1]

    if multi:
        print(f"  passes unanimous on rank 1:         "
              f"{sum(1 for r in multi if r['agreement'] >= 0.999)} of "
              f"{len(multi)} multi-pass queries")

    staged = sum(1 for r in records
                 if any(p.get("stage2") for p in r.get("passes") or ()))
    stage2_time = sum(p.get("stage2_elapsed", 0.0) for r in records
                      for p in r.get("passes") or ())

    print(f"  queries needing the stage-2 ordering call: {staged} "
          f"({100 * staged / len(records):.1f}%, "
          f"{stage2_time / 3600:.2f} h of the total)")
    print(f"  queries with a repaired ranking:    "
          f"{sum(1 for r in records if r['n_repaired'])}")
    print(f"  queries with a failed/unusable pass:"
          f"{sum(1 for r in records if r['n_pass_failures']):5d}")
    print(f"  queries left in official order:     "
          f"{sum(1 for r in records if r['fallback'])}"
          f"{'  -- CHECK: these are unranked' if any(r['fallback'] for r in records) else ''}")
    print(f"  queries ranked without a video:     "
          f"{sum(1 for r in records if not r['video_used'])}"
          f"  (no file on disk: "
          f"{sum(1 for r in records if not r['video_exists'])})")
    print(f"  queries ranked without a transcript:"
          f"{sum(1 for r in records if not r['has_transcript']):5d}")

    reasoning = [r["reasoning_chars"] for r in records if r["n_replies"]]

    if reasoning:
        reasoning.sort()
        print(f"  reasoning chars per query: median "
              f"{reasoning[len(reasoning) // 2]}  "
              f"zero-length {sum(1 for v in reasoning if not v)}")

    times = [r["elapsed"] for r in records if r["elapsed"]]

    if times:
        times.sort()
        print(f"  inference per query: mean {sum(times) / len(times):6.2f}s  "
              f"median {times[len(times) // 2]:.2f}s  "
              f"p90 {times[int(0.9 * len(times))]:.2f}s  "
              f"total {sum(times) / 3600:.2f} h")


if __name__ == "__main__":
    sys.exit(main())

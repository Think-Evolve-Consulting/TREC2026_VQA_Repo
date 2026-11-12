#!/usr/bin/env python3
"""Rank the Task 2 multiple-choice options with Qwen3.8-27B (INT4).

The fourth Task 2 ranking pass, after Gemma 4 (``rank.options.mc.py``),
Cosmos-Reason2 (``rank.options.mc.cosmos.py``) and Qwen3-VL-8B-Instruct
(``rank.options.mc.qwen3vl.py``). It is the Qwen3-VL pass with the checkpoint
swapped: the same ranking contract -- a permutation of option *numbers* mapped
back to the official strings, so no rank is spent twice and no option goes
unranked -- and the same video handling, which transfers because the two share
a vision geometry (patch 16, temporal patch 2, spatial merge 2, so 1,024
pixels per visual token).

**What differs is the model, and it is a tight fit.** Qwen3.8-27B is a 27B
dense multimodal model whose config declares an image *and* a video token.
bf16 is ~54 GB, so this serves RedHatAI's INT4 (compressed-tensors W4A16)
build: 18.6 GB of weights on a 24 GB card, leaving roughly 5 GB for KV cache,
activations and the vision encoder. That is only workable because the language
model is hybrid -- 48 of its 64 layers are linear attention and 16 are full
attention -- so the KV cache is a fraction of what a dense 27B would need.
``--skip-mm-profiling`` is mandatory for the same reason it is on the Qwen3-VL
service, and ``--visual-tokens`` is the first knob to turn if the engine will
not start or a request will not fit.

**vLLM 0.27.1 is required, not the 0.12.0 the other video passes use.** Only
the newer engine registers ``Qwen3_5ForConditionalGeneration``, so this serves
from ``/home/aakash/gemma4/.venv``, with the two WSL2 environment variables
that venv's service already carries (``VLLM_USE_V2_MODEL_RUNNER=0`` for the
absent UVA, ``VLLM_USE_FLASHINFER_SAMPLER=0`` for the absent nvcc).

The inherited video handling solves two traps that cost the Cosmos pass a day:

**The visual budget is measured, not assumed.** ``size.longest_edge`` is a
whole-video pixel budget on this server, not the per-frame ceiling
``generate.answers.qwen3vl.py`` takes it for -- measured by varying it against
a fixed clip, the prompt tokens track the budget and ignore the frame count.
Reading it per-frame divides the budget by the frame count and sends
thumbnails: the same two clips came back at 485 and 399 visual tokens against
a planned 17,792 and 5,376. ``video_mm_kwargs()`` sets the budget directly and
lets fps decide only how those tokens are spread over time.

**do_sample_frames is not sent to short clips.** Asking the processor to
re-sample a video that yields few frames fails outright -- measured on this
corpus, every request working out to <= 11 frames failed and every one at >= 12
succeeded -- so below ``MIN_RESAMPLE_FRAMES`` the pixel budget is sent without
the frame rate. This is the same failure that silently cost the Cosmos pass its
video on 23% of queries; here it is avoided up front, and the ``no_dsf`` rung
of the degrade ladder catches whatever slips through.

**The reply is schema-constrained, which the Cosmos pass could not be.**
``response_format`` forces exactly N integers in range, so the two-stage
recovery the reasoning model needed is unnecessary. If this checkpoint reasons
by default, the schema suppresses it -- the right trade where the answer is a
permutation rather than an argument. Uniqueness is still not
expressible in a JSON schema, so ``as_permutation()`` remains the safety net.

**Three passes, as in the Gemma 4 run.** Decode is cheap without a reasoning
trace (a permutation is ~40 tokens), and the passes of one query share a video
prefix that the server's prefix cache prefills once -- measured, three passes
cost the same as one to within run-to-run variance, at 1.8-3 s each against
70 s for the first. Options are presented in the official order, its exact
reverse, then a seeded shuffle, and the passes are combined by Borda count --
the same arrangement the Gemma 4 run used, so the two are directly comparable.

Run this **from inside WSL2**, with the server's venv Python -- the server
opens the video off the path in the request:

    /home/aakash/gemma4/.venv/bin/python3 \\
        /mnt/e/TREC_2026/VQA/rank.options.mc.qwen38.py --limit 20

    ... --rebuild        # re-emit the run file from the cache, no inference
    ... --only 1 3 264   # named Q_IDs
    ... --passes 1       # one presentation, for a third of the wall clock

The GPU holds one model at a time: stop every other service first. Serve on
port 8011, which collides with nothing (8008 Cosmos, 8009 Qwen3-VL, 8010
Gemma 4):

    VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 \\
    /home/aakash/gemma4/.venv/bin/vllm serve RedHatAI/Qwen3.8-27B-INT4 \\
      --host 0.0.0.0 --port 8011 \\
      --allowed-local-media-path /mnt/e/TREC_2026 \\
      --max-model-len 16384 --gpu-memory-utilization 0.93 \\
      --enforce-eager --skip-mm-profiling \\
      --media-io-kwargs '{"video": {"num_frames": -1}}'
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

from openai import APIStatusError, BadRequestError, OpenAI

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = ROOT / "vqa.2026.testingDataset.Task2_v2.json"
DEFAULT_ROUTING = ROOT / "query.routing.task1.json"
DEFAULT_VIDEOS = ROOT / "videos_TEST_VQA" / "videos"
DEFAULT_FRAME_INDEX = ROOT / "videos_TEST_VQA" / "frames" / "index.json"
DEFAULT_TRANSCRIPTS = ROOT / "videos_TEST_VQA" / "transcription_openwhisper"
DEFAULT_OUT = ROOT / "qwen38.run.task2.json"
DEFAULT_CACHE = ROOT / "rankings.task2.qwen38.cache.jsonl"

# RedHatAI's INT4 (compressed-tensors W4A16) build of Qwen/Qwen3.8-27B: 18.6 GB
# of weights against ~54 GB for the bf16 original, which does not fit a 3090.
# The AMD Quark and cyankiwi AWQ builds are the same size and would also serve;
# this one is compressed-tensors, which vLLM loads natively. The 0.85 GB MTP
# module in the repo is for speculative decoding and is not loaded here.
DEFAULT_MODEL = "RedHatAI/Qwen3.8-27B-INT4"
DEFAULT_BASE_URL = "http://localhost:8011/v1"

# Ranking wants a colder sampler than answer generation: the diversity comes
# from permuting the options across passes, not from sampling noise. Same 0.2
# the Gemma 4 ranking pass uses, against the 0.7 of the Task 1 Qwen pass.
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20

# Inherited from the Qwen3-VL pass, where 4 was the measured knee. That is a
# starting point here, not a finding: this checkpoint leaves only ~5 GB of VRAM
# for KV and activations, which concurrent requests share, so the knee may sit
# lower. The host-RAM ceiling that OOM-killed the Cosmos pass at 8 workers
# applies unchanged -- same num_frames=-1 decode, same ~24 GiB. Measure with
# --limit 20 before raising it.
DEFAULT_WORKERS = 4

# A permutation of eight integers is ~40 tokens. 256 is headroom, not budget:
# the schema stops the model writing anything else, and every token not
# reserved here is one video_mm_kwargs() can spend on pixels.
DEFAULT_MAX_TOKENS = 256

DEFAULT_PASSES = 3

# Adaptive fps: a fixed frame budget per video regardless of length. Durations
# in videos_TEST_VQA (n=1,940): median 10.3 s, p90 56 s, p99 233 s, max 389 s.
TARGET_FRAMES = 64
DENSE_TARGET_FRAMES = 96
"""For the queries the router marked dense_timestamped_frames: ordering and
timing questions, where the gap between samples is what loses the answer."""
MAX_FPS = 4.0
MIN_FPS = 0.2

# Visual-token geometry from the processor's config: a token is
# (patch_size 16 * merge_size 2)^2 = 1024 pixels, and temporal_patch_size 2
# merges frames in pairs, so N frames cost N/2 x per-frame tokens.
PIXELS_PER_TOKEN = 1024
TEMPORAL_MERGE = 2

MIN_RESAMPLE_FRAMES = 16
"""Below this, ask for the pixel budget but not the frame rate.

``do_sample_frames`` makes the processor re-sample the decoded video at the
requested fps, and on a clip too short to yield many frames that call fails
outright -- ``Failed to apply Qwen3VLProcessor``. Measured against this corpus
the boundary is sharp: every video whose request worked out to <= 11 frames
failed, every one at >= 12 succeeded. 16 leaves margin."""

# `size.longest_edge` is a WHOLE-VIDEO pixel budget, not a per-frame ceiling.
# generate.answers.qwen3vl.py reads it the other way (its comment cites
# vLLM's `max_pixels=image_processor.size["longest_edge"]`) and therefore
# divides the budget by the frame count, which shrinks every frame to a
# thumbnail. Measured against this server, holding the clip and fps fixed and
# varying only longest_edge:
#
#   longest_edge   prompt_tokens (Memphis-20, 62 frames)   (8384, 14 frames)
#    2,048 tok            1,152                                 1,079
#    8,192 tok            4,004                                 4,229
#   16,384 tok            8,096                                 6,231
#
# The count tracks the budget and ignores the frame count, at almost exactly
# half -- the pairwise temporal merge -- so a target of T visual tokens needs
# longest_edge = T * TEMPORAL_MERGE * PIXELS_PER_TOKEN. Under the per-frame
# reading the same clips came back at 485 and 399 tokens against a planned
# 17,792 and 5,376: a video, but far too coarse to read.
VISUAL_TARGET_TOKENS = 4096
"""What one video is worth, and the dominant cost per call.

The measurements below are from Qwen3-VL-8B-Instruct, not this checkpoint.
They are kept because the vision geometry is identical and they set the right
expectation -- a 27B model at INT4 will be slower per token, not faster -- but
re-measure before planning a full run on them.

Measured end to end over the same 20 queries at 4 workers, one pass each:

    8,192 tokens   677 s   (33.8 s/query)   ~23 h over 2,443
    2,048 tokens   258 s   (12.9 s/query)   ~8.7 h

4,096 is the chosen middle: half the detail the other two Task 2 passes spend
(Cosmos measured 8.7-9.9k, Gemma 4's 32 frames ~9k) for roughly two thirds of
the time, ~15 h. Raise it with --visual-tokens for parity with them, lower it
for speed; either way the run stays resumable and the setting is recorded per
query in the cache, so a mixed-budget cache is legible afterwards.

The passes are NOT a lever: three passes measured 612 s against one pass's
677 s over the same queries, i.e. free within run-to-run variance, because
passes 2 and 3 hit the server's prefix cache (1.8-3 s against 70 s for the
first)."""
OCR_TARGET_TOKENS = 12288
"""The OCR-routed queries are the exception -- reading a sign on screen is
exactly what resolution buys."""
MIN_FRAME_TOKENS = 128
"""Floor for size.shortest_edge, left small so the processor is never forced
to upscale a low-resolution clip."""
TEXT_TOKEN_RESERVE = 2048
"""Held back from the visual budget for the system prompt, the transcript
(some run long), the question and, here, the option list."""

# num_frames=-1 makes the server decode every native-resolution frame into a
# raw array before any subsampling, so a long, high-res source allocates
# gigabytes of host RAM up front. 2 GiB, not the 4 the Task 1 pass used: that
# ceiling is the only thing bounding one in-flight request, WSL2 has ~24 GiB
# free for all of them together, and 4 GiB is what let the kernel OOM-kill the
# Cosmos server mid-run.
RAW_DECODE_BYTES_LIMIT = 2 * 1024**3
TRANSCODE_CACHE_DIRNAME = ".vlm_transcoded_cache"
LEGACY_TRANSCODE_CACHE_DIRNAME = ".cosmos_transcoded_cache"
"""The Cosmos runs already shrank several hundred of these clips at the same
target fps. Reused when the rate is sufficient rather than re-encoded."""

SYSTEM_PROMPT = "You are a helpful assistant."

PROMPT_TEMPLATE = """{context}Question: {question}

Answer options:
{options}

Rank all {n} options from the most likely correct answer to the least likely.

Rules:
- Judge each option against the evidence you were given, not against how it is
  worded.
- Every number from 1 to {n} appears exactly once: never repeat one, never
  leave one out.
- Exactly one option is correct. Put it first, then the next most plausible,
  and the option the video contradicts most clearly last.{extra}

Reply with ONLY this JSON: {{"ranking": [{slots}]}}"""

OCR_RULE = ("\n- The answer depends on text visible on screen. Read any signs, "
            "captions, labels or subtitles carefully before choosing.")
TEMPORAL_RULE = ("\n- The answer depends on ordering or timing. Use the frame "
                 "timestamps to work out what happens before what.")

JSON_BLOB_RE = re.compile(r"\{.*?\}", re.DOTALL)
DIGIT_LIST_RE = re.compile(r"\d+(?:\s*[,\s]\s*\d+){1,}")

# A 400 whose text names the context window is a budget problem and worth
# retrying smaller; any other 400 is the processor rejecting the request, and
# retrying it identically wastes the timeout.
CONTEXT_ERROR_RE = re.compile(
    r"maximum context length|longer than the maximum|max_model_len|"
    r"context length|too long", re.IGNORECASE)


# ---------------------------------------------------------------------------
# The ranking contract, shared with the other two Task 2 passes
# ---------------------------------------------------------------------------

def ranking_schema(n: int) -> dict:
    """Force exactly `n` integers in range.

    Uniqueness is deliberately absent: no JSON-schema grammar backend here
    enforces ``uniqueItems``, so claiming it would buy nothing and hide the
    repair path that actually does the work (as_permutation).
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ranking",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "ranking": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1,
                                  "maximum": n},
                        "minItems": n,
                        "maxItems": n,
                    },
                },
                "required": ["ranking"],
                "additionalProperties": False,
            },
        },
    }


def seed_for(q_id: int, index: int) -> int:
    """Stable per-call seed, so a re-run reproduces the same presentation."""
    return (q_id * 1_000_003 + index * 7919) % (2**31 - 1)


def presentation_orders(n: int, passes: int, q_id: int,
                        first: str) -> list[list[int]]:
    """The option orders the passes present, as index lists into `options`.

    Pass 0 is the official order (or a per-query shuffle with
    ``--order shuffled``), pass 1 its exact reverse -- between them every
    option that was shown first is shown last, the cheapest possible probe of
    position bias -- and later passes are seeded shuffles.
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


def extract_ranking(content: str, n: int) -> tuple[list[int], str]:
    """Pull a ranking of presented positions out of the reply.

    The schema makes ``json`` the normal route. ``digits`` salvages a bare
    "3, 1, 4, 2" from a reply that arrived without the grammar (a server
    started without structured output, a truncation), because one recovered
    reply is worth more than one lost pass.
    """
    for blob in reversed(JSON_BLOB_RE.findall(content or "")):
        try:
            data = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            continue

        values = data.get("ranking") if isinstance(data, dict) else None

        if isinstance(values, list) and values:
            return values, "json"

    for match in reversed(DIGIT_LIST_RE.findall(content or "")):
        values = [int(v) for v in re.findall(r"\d+", match)]

        if len([v for v in set(values) if 1 <= v <= n]) == n:
            return values, "digits"

    return [], "none"


def as_permutation(raw: list, n: int) -> tuple[list[int], bool]:
    """Coerce a reply's ranking into a permutation of 0..n-1 (presented order).

    Repeats keep their first, best occurrence; whatever was dropped is appended
    in the order it was shown. Returns the positions and whether anything had
    to be repaired.
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

    Each pass awards ``n - position`` points to the option it placed there.
    Unweighted, like the Cosmos pass: this model is not asked for a confidence,
    because a schema-constrained reply has nowhere honest to put one. Ties
    break on mean rank, then official option order, so output is deterministic.
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


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_queries(path: Path) -> list[dict]:
    """The released MC queries, with Options flattened to a list of strings."""
    return [{"Q_ID": q["Q_ID"], "Video_ID": q["Video_ID"],
             "Question": q["Question"],
             "Options": [option["option"] for option in q["Options"]]}
            for q in load_json(path)]


def resolve_video(videos_dir: Path, video_id: str) -> Path:
    """The file a Video_ID names, per SESSION.2026-08-11.task1-pipeline.md §3.

    A released clip carries its ``.webm`` extension; a YouTube ID does not, and
    ``download_test_videos.py`` wrote those as ``<Video_ID>.mp4``. Resolving
    ``videos_dir / video_id`` literally is what made the first Cosmos Task 1
    run 45% blind.
    """
    name = video_id if video_id.endswith(".webm") else video_id + ".mp4"
    return videos_dir / name


def load_durations(index_path: Path) -> dict[str, float]:
    """Video stem -> probed duration, from the frame index.

    The frames themselves are unused -- this model takes video -- but the fps
    has to be chosen before the request, and the transcript carries a duration
    only when the clip has speech.
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


def load_routing(path: Path) -> dict[int, dict]:
    """Q_ID -> routing verdict from route.query.inputs.py."""
    if not path.exists():
        return {}

    return {int(entry["Q_ID"]): entry for entry in load_json(path)}


def plan_for(routing: dict | None, question: str, ignore_routing: bool) -> dict:
    """Which inputs this query is sent, and how densely it is sampled.

    The routing file was produced for Task 1, which shares every Q_ID and video
    with Task 2; all 2,443 questions match once trailing whitespace is stripped
    (20 differ by exactly that). The guard is here because a verdict about a
    different question is not a verdict about this one, so a re-release that
    truly reworded a question falls back to attaching everything.
    """
    default = {"video": True, "transcript": True, "ocr": False,
               "temporal": False, "dense": False, "routed": False,
               "rescued": False}

    if ignore_routing or not routing:
        return default

    if (routing.get("Question") or "").strip() != question.strip():
        return default

    pipeline = set(routing.get("Pipeline") or ())

    plan = {
        "video": "frames" in pipeline,
        "transcript": "asr_transcript" in pipeline,
        "ocr": "frame_ocr" in pipeline,
        "temporal": bool(routing.get("Temporal")),
        "dense": "dense_timestamped_frames" in pipeline
                 or bool(routing.get("Temporal")),
        "routed": True,
        "rescued": False,
    }

    # An OCR question is about pixels whatever else the router decided.
    if plan["ocr"]:
        plan["video"] = True

    return plan


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


def mmss(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def adaptive_fps(duration_sec: float | None, dense: bool) -> float:
    """Target a fixed frame count per video regardless of its length."""
    target = DENSE_TARGET_FRAMES if dense else TARGET_FRAMES

    if duration_sec and duration_sec > 0:
        return min(MAX_FPS, max(MIN_FPS, target / duration_sec))

    return 2.0


def expected_frames(duration_sec: float | None, fps: float) -> int:
    """How many frames that fps will actually produce, for budgeting."""
    if not duration_sec or duration_sec <= 0:
        return TARGET_FRAMES

    return max(2, min(int(round(duration_sec * fps)), DENSE_TARGET_FRAMES * 2))


def video_mm_kwargs(duration_sec: float | None, plan: dict, max_model_len: int,
                    max_tokens: int, shrink: int = 0,
                    visual_tokens: int = VISUAL_TARGET_TOKENS) -> dict:
    """fps and per-frame pixel budget for one request.

    ``size.longest_edge`` is a whole-video pixel budget (see
    VISUAL_TARGET_TOKENS), so it is set from the target directly; fps decides
    only how those tokens are spread over time. ``shrink`` halves the budget,
    once per rung of the degrade ladder.
    """
    fps = adaptive_fps(duration_sec, plan["dense"]) / (2 ** shrink)
    fps = max(MIN_FPS, round(fps, 3))

    frames = expected_frames(duration_sec, fps)

    # The budget is for the whole video (see VISUAL_TARGET_TOKENS), so the
    # frame count does not divide it -- it only decides how that many tokens
    # are spread over time. What must fit the window is the target itself.
    target = (min(OCR_TARGET_TOKENS, visual_tokens * 3 // 2) if plan["ocr"]
              else visual_tokens)
    headroom = max(max_model_len - max_tokens - TEXT_TOKEN_RESERVE, 1024)
    target = min(target, headroom) // (2 ** shrink)

    kwargs = {
        "size": {
            "shortest_edge": MIN_FRAME_TOKENS * PIXELS_PER_TOKEN,
            "longest_edge": target * TEMPORAL_MERGE * PIXELS_PER_TOKEN,
        },
    }

    # fps is only honoured when do_sample_frames is set -- sending it alone is
    # a measurable no-op -- and the pair is unsafe on very short clips, so the
    # two travel together or not at all. fps and num_frames are mutually
    # exclusive in the processor (it raises), so num_frames is never sent.
    if frames >= MIN_RESAMPLE_FRAMES:
        kwargs["do_sample_frames"] = True
        kwargs["fps"] = fps

    return kwargs


def build_context(duration_sec: float | None, transcript: dict | None,
                  plan: dict, video_attached: bool) -> str:
    """The text block that precedes the question."""
    parts: list[str] = []

    if video_attached:
        if duration_sec:
            parts.append(f"You are shown the video for this question. It is "
                         f"{duration_sec:.0f} seconds long "
                         f"({mmss(duration_sec)}).")
        else:
            parts.append("You are shown the video for this question.")
    elif plan["video"] or plan["rescued"]:
        parts.append("The video for this question is unavailable, so judge the "
                     "options from what else you are given, as plausibly as "
                     "you can.")
    else:
        parts.append("This question is answered by the soundtrack, so you are "
                     "shown no video. Judge the options from the transcript.")

    if transcript and transcript.get("text"):
        language = transcript.get("language") or "unknown"
        parts.append(f"Spoken audio (automatic transcript, language: "
                     f"{language}):\n\"{transcript['text']}\"")
    elif video_attached:
        parts.append("The video has no detected speech.")

    return "\n\n".join(parts) + "\n\n"


# ---------------------------------------------------------------------------
# Video preparation
# ---------------------------------------------------------------------------

class VideoLibrary:
    """Probes and, where needed, pre-shrinks clips. Results are cached.

    Lifted from generate.answers.qwen3vl.py: the transcode cache is shared
    with it and with the Cosmos passes, so a clip already shrunk at a
    sufficient frame rate is reused rather than re-encoded.
    """

    def __init__(self, videos_dir: Path, decode_limit: int,
                 transcode: bool) -> None:
        self.videos_dir = videos_dir
        self.decode_limit = decode_limit
        self.transcode = transcode
        self.cache_dir = videos_dir / TRANSCODE_CACHE_DIRNAME
        self.legacy_dir = videos_dir / LEGACY_TRANSCODE_CACHE_DIRNAME
        self._probes: dict[Path, dict | None] = {}
        self._safe: dict[tuple[Path, float], tuple[Path, float]] = {}
        self._lock = threading.Lock()

    def probe(self, path: Path) -> dict | None:
        """Native width/height/fps/duration via ffprobe, or None if it fails."""
        with self._lock:
            if path in self._probes:
                return self._probes[path]

        info = None

        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,r_frame_rate",
                 "-show_entries", "format=duration", "-of", "json", str(path)],
                capture_output=True, text=True, timeout=60,
            )
            data = json.loads(result.stdout)
            stream = (data.get("streams") or [{}])[0]
            num, _, den = (stream.get("r_frame_rate") or "0/1").partition("/")
            fps = float(num) / float(den) if float(den or 1) else 0.0
            duration = float((data.get("format") or {}).get("duration") or 0.0)

            info = {"width": int(stream.get("width") or 0),
                    "height": int(stream.get("height") or 0),
                    "fps": fps, "duration": duration}
        except (subprocess.SubprocessError, ValueError, KeyError, IndexError,
                json.JSONDecodeError, OSError):
            info = None

        with self._lock:
            self._probes[path] = info

        return info

    def safe_path(self, path: Path, target_fps: float,
                  force: bool = False) -> tuple[Path, float]:
        """A path safe for the server's num_frames=-1 full decode, and the
        seconds spent getting there (0 when nothing had to be done).

        ``force`` shrinks whatever the size, which is how the no_dsf rung gets
        a file whose own frame rate is the sampling it wanted.
        """
        safe_fps = round(target_fps, 3) or 1.0
        key = (path, safe_fps, force)

        with self._lock:
            if key in self._safe:
                return self._safe[key]

        info = self.probe(path)
        result = (path, 0.0)

        if info and info["width"] and info["height"] and info["fps"]:
            raw = (info["duration"] * info["fps"] * info["width"]
                   * info["height"] * 3)

            if (raw > self.decode_limit or force) and self.transcode:
                result = self._shrink(path, min(safe_fps, info["fps"]))

        with self._lock:
            self._safe[key] = result

        return result

    def _shrink(self, path: Path, fps: float) -> tuple[Path, float]:
        fps = round(fps, 3) or 1.0
        cached = self._cached(path.stem, fps)

        if cached:
            return cached, 0.0

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_dir / f"{path.stem}.f{fps}.mp4"
        started = time.monotonic()

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(path), "-vf", f"fps={fps}",
                 "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                 str(target)],
                capture_output=True, text=True, timeout=1800, check=True,
            )
            return target, time.monotonic() - started
        except (subprocess.SubprocessError, OSError):
            return path, time.monotonic() - started

    def _cached(self, stem: str, fps: float) -> Path | None:
        """An already-shrunk copy good enough to sample at `fps`.

        A cached file carries the rate it was written at in its name. One at or
        above the rate wanted is fine -- the processor subsamples further -- so
        the cheapest sufficient copy wins. One below it would silently cost
        frames, so it is ignored.
        """
        best: tuple[float, Path] | None = None

        for directory in (self.cache_dir, self.legacy_dir):
            if not directory.is_dir():
                continue

            for candidate in directory.glob(f"{stem}.f*.mp4"):
                try:
                    have = float(candidate.stem.rsplit(".f", 1)[1])
                except (IndexError, ValueError):
                    continue

                if have + 1e-6 >= fps and (best is None or have < best[0]):
                    best = (have, candidate)

        return best[1] if best else None


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------

class Qwen38Client:
    """Thin wrapper over the OpenAI-compatible Qwen3.8 endpoint.

    Every call walks a degrade ladder rather than failing: shrink the visual
    budget, drop the frame-rate control, drop the processor kwargs, drop the
    video. The rung reached is returned so the run can be audited afterwards --
    a pass that quietly answered without pixels is the failure mode that cost
    the Cosmos run 23% of its queries before anyone noticed.
    """

    LADDER = ("full", "shrink1", "shrink2", "no_dsf", "no_mm_kwargs",
              "no_video")

    name = "qwen38"

    def __init__(self, base_url: str, model: str, temperature: float,
                 top_p: float, top_k: int, timeout: float, retries: int,
                 max_tokens: int) -> None:
        self.client = OpenAI(api_key="EMPTY", base_url=base_url,
                             timeout=timeout, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.retries = retries
        self.max_tokens = max_tokens
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

    def resolve_model(self) -> str:
        """The id the server actually serves, so --model can stay a nickname."""
        ids = [m.id for m in self.client.models.list().data]

        if self.model not in ids and ids:
            self.model = ids[0]

        return self.model

    def rank(self, prompt_for, video_for, mm_kwargs_for, n: int) -> dict:
        """One ranking call, degrading until something comes back.

        ``prompt_for(video_attached)``, ``video_for(step)`` and
        ``mm_kwargs_for(shrink)`` are callbacks because all three change as the
        ladder descends -- a prompt that says "you are shown the video" must
        not survive dropping it, and the no_dsf rung needs a different *file*,
        not just different kwargs.

        Why the file: measured on this checkpoint, ``do_sample_frames`` is
        rejected for 91% of this corpus's clips, ``num_frames`` is silently
        ignored without it (identical prompt tokens at 16 and 64) and rejected
        with it. So there is no kwarg that controls temporal sampling here --
        only the frame rate of the file handed over. Below the no_dsf rung the
        clip is therefore pre-transcoded to the fps this query planned for, and
        the processor's own sampling is left with nothing to decide.
        """
        rung = 0 if video_for(self.LADDER[0]) else self.LADDER.index("no_video")
        attempts = 0
        last: Exception | None = None

        while rung < len(self.LADDER) and attempts < self.retries + 6:
            attempts += 1
            step = self.LADDER[rung]
            attach = step != "no_video"
            shrink = {"shrink1": 1, "shrink2": 2}.get(step, 0)
            mm_kwargs = None if step in ("no_mm_kwargs", "no_video") \
                else mm_kwargs_for(shrink)

            if step == "no_dsf" and mm_kwargs:
                # Keep the pixel budget, give up only the frame-rate control:
                # do_sample_frames is what the processor rejects, and dropping
                # the whole block would hand a long video back to the server's
                # own default frame count.
                mm_kwargs = {k: v for k, v in mm_kwargs.items()
                             if k not in ("do_sample_frames", "fps")}

            content: list[dict] = []
            video_uri = video_for(step) if attach else None

            if video_uri:
                content.append({"type": "video_url",
                                "video_url": {"url": video_uri}})

            content.append({"type": "text", "text": prompt_for(attach)})

            extra: dict = {"top_k": self.top_k}

            if mm_kwargs:
                extra["mm_processor_kwargs"] = mm_kwargs

            started = time.monotonic()

            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    extra_body=extra,
                    response_format=ranking_schema(n),
                )

                choice = completion.choices[0]

                return {
                    "content": (choice.message.content or "").strip(),
                    "elapsed": time.monotonic() - started,
                    "rung": step,
                    "video_used": bool(video_uri),
                    "finish_reason": choice.finish_reason,
                }
            except BadRequestError as exc:
                last = exc
                text = str(exc)

                # A context overflow is worth retrying smaller; anything else
                # is the processor refusing this request shape, so skip
                # straight to the rungs that change the shape.
                if CONTEXT_ERROR_RE.search(text):
                    rung += 1
                elif step in ("full", "shrink1", "shrink2"):
                    rung = self.LADDER.index("no_dsf")
                else:
                    rung += 1
            except APIStatusError as exc:
                last = exc

                if attempt_is_transient(exc) and attempts < self.retries:
                    time.sleep(2.0 * attempts)
                else:
                    rung += 1
            except Exception as exc:  # noqa: BLE001 - transient, worth a wait
                last = exc

                if attempts <= self.retries:
                    time.sleep(2.0 * attempts)
                else:
                    rung += 1

        raise RuntimeError(f"{type(last).__name__}: {last}")


def attempt_is_transient(exc: APIStatusError) -> bool:
    """A 5xx or a rate limit is worth waiting out; a 4xx is not."""
    status = getattr(exc, "status_code", None)

    return status is None or status >= 500 or status == 429


# ---------------------------------------------------------------------------
# One query
# ---------------------------------------------------------------------------

def rank_query(client: Qwen38Client, query: dict, options: list[str],
               library: VideoLibrary, videos_dir: Path, transcripts_dir: Path,
               durations: dict[str, float], passes: int, first_order: str,
               plan: dict, visual_tokens: int = VISUAL_TARGET_TOKENS) -> dict:
    """Rank one query's options over however many presentations are asked."""
    q_id = query["Q_ID"]
    video_id = query["Video_ID"]
    n = len(options)

    video_path = resolve_video(videos_dir, video_id)
    video_exists = video_path.exists()
    transcript = load_transcript(transcripts_dir, video_id)
    duration = durations.get(Path(video_id).stem) or (
        transcript.get("duration_sec") if transcript else None)

    attach_video = video_exists and plan["video"]
    prep = 0.0
    target_fps = adaptive_fps(duration, plan["dense"])
    video_uri = None

    if attach_video:
        safe, prep = library.safe_path(video_path, target_fps)
        video_uri = safe.resolve().as_uri()

    # Below the no_dsf rung the processor decides the temporal sampling unless
    # the file itself carries it, so those rungs get a copy transcoded to this
    # query's planned fps. It is built on first use, not up front: most of the
    # corpus is short enough that the first rung succeeds, and the transcode
    # cache is shared with every other pass here.
    sampled: dict[str, str | None] = {}

    def video_for(step: str) -> str | None:
        if not attach_video:
            return None

        if step in ("full", "shrink1", "shrink2"):
            return video_uri

        if step not in sampled:
            nonlocal prep

            try:
                shrunk, spent = library.safe_path(video_path, target_fps,
                                                  force=True)
                prep += spent
                sampled[step] = shrunk.resolve().as_uri()
            except Exception as exc:  # noqa: BLE001 - fall back to the original
                print(f"  Q_ID {q_id}: could not resample video ({exc})",
                      file=sys.stderr)
                sampled[step] = video_uri

        return sampled[step]

    extra = ""

    if plan["ocr"]:
        extra += OCR_RULE
    if plan["temporal"]:
        extra += TEMPORAL_RULE

    results: list[dict] = []
    failures = 0
    repairs = 0
    routes: list[str] = []

    for index, order in enumerate(presentation_orders(n, passes, q_id,
                                                      first_order)):
        def prompt_for(video_attached: bool, order=order) -> str:
            return PROMPT_TEMPLATE.format(
                context=build_context(duration, transcript, plan,
                                      video_attached),
                question=query["Question"],
                options=render_options(options, order),
                n=n,
                slots=", ".join(str(i) for i in range(1, n + 1)),
                extra=extra,
            )

        def mm_kwargs_for(shrink: int) -> dict:
            return video_mm_kwargs(duration, plan, client.max_model_len,
                                   client.max_tokens, shrink, visual_tokens)

        try:
            reply = client.rank(prompt_for, video_for, mm_kwargs_for, n)
        except RuntimeError as exc:
            failures += 1
            print(f"  Q_ID {q_id} pass {index}: {exc}", file=sys.stderr)
            continue

        raw, route = extract_ranking(reply["content"], n)
        routes.append(route)

        if route == "none":
            # Nothing usable came back. Padding to a permutation would record
            # the presented order as if the model had chosen it, so the pass is
            # dropped; the query falls back only if every pass does the same.
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
            "rung": reply["rung"],
            "video_used": reply["video_used"],
            "finish_reason": reply["finish_reason"],
            "elapsed": round(reply["elapsed"], 3),
        })

    if results:
        ranked = borda(results, n)
        fallback = False
    else:
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
        "rungs": sorted({r["rung"] for r in results}),
        "fallback": fallback,
        "video_exists": video_exists,
        "video_used": any(r["video_used"] for r in results),
        "has_transcript": bool(transcript and transcript.get("text")),
        "fps": round(adaptive_fps(duration, plan["dense"]), 3),
        "visual_tokens": visual_tokens,
        "agreement": round(max((item["firsts"] for item in ranked), default=0)
                           / max(len(results), 1), 4),
        "routed": plan["routed"],
        "rescued": plan["rescued"],
        "ocr": plan["ocr"],
        "temporal": plan["temporal"],
        "dense": plan["dense"],
        "prep": round(prep, 2),
        "elapsed": round(sum(r["elapsed"] for r in results), 2),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def preflight(base_url: str, model: str) -> str:
    """Refuse to start unless the server is up, and say what it serves.

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
            f"  the GPU holds one model at a time -- stop the others first:\n"
            f"    systemctl --user stop cosmos-reason2 gemma4-vllm\n"
            f"  then serve it from the vLLM 0.27 venv (0.12 does not register "
            f"Qwen3_5ForConditionalGeneration) -- the serve command is in this "
            f"file's docstring; port 8011")

    if model not in served:
        print(f"  note: {base_url} serves {served}, not {model!r} -- using "
              f"{served[0] if served else 'nothing'}", file=sys.stderr)

    return served[0] if served else model


def assemble(queries: list[dict], cache: dict[int, dict]) -> tuple[list[dict], int]:
    """Emit the run file, carrying only the fields the MC validator allows."""
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
                    "Qwen3.8-27B (INT4) over native video."
    )

    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    ap.add_argument("--frame-index", type=Path, default=DEFAULT_FRAME_INDEX,
                    help="read only for its probed durations, which set the "
                         "sampling fps (default: %(default)s)")
    ap.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    ap.add_argument("--ignore-routing", action="store_true",
                    help="attach every input for every query -- the ablation "
                         "a routed run has to beat")
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
                    help="rankings per query, each presenting the options in a "
                         "different order (default: %(default)s)")
    ap.add_argument("--order", choices=("official", "shuffled"),
                    default="official",
                    help="how the first pass presents the options "
                         "(default: %(default)s, matching the Gemma 4 run)")
    ap.add_argument("--visual-tokens", type=int, default=VISUAL_TARGET_TOKENS,
                    help="visual tokens one video is worth; the dominant cost "
                         "per call (default: %(default)s)")
    ap.add_argument("--decode-limit", type=int, default=RAW_DECODE_BYTES_LIMIT,
                    help="pre-shrink any clip whose raw decode would exceed "
                         "this many bytes (default: %(default)s)")
    ap.add_argument("--no-transcode", action="store_true",
                    help="never call ffmpeg; hand every clip over as-is")

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

    if sys.platform == "win32":
        print("  WARNING: run this from WSL2 -- the server reads the video off "
              "the path in the request, and a Windows path is not one it can "
              "open", file=sys.stderr)

    queries = load_queries(args.queries)
    cache = load_cache(args.cache)
    durations = load_durations(args.frame_index)
    routing = {} if args.ignore_routing else load_routing(args.routing)

    if not routing and not args.ignore_routing and not args.rebuild:
        print(f"  WARNING: no routing at {args.routing} -- every input will be "
              f"attached for every query", file=sys.stderr)

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
            model = preflight(args.base_url, args.model)

            client = Qwen38Client(args.base_url, model, args.temperature,
                                   DEFAULT_TOP_P, DEFAULT_TOP_K, args.timeout,
                                   args.retries, args.max_tokens)
            client.resolve_model()

            library = VideoLibrary(args.videos, args.decode_limit,
                                   not args.no_transcode)

            print(f"serving: {client.model} | max_model_len "
                  f"{client.max_model_len}")

            lock = threading.Lock()
            done = failed = 0
            started = time.monotonic()

            args.cache.parent.mkdir(parents=True, exist_ok=True)
            handle = args.cache.open("a", encoding="utf-8")

            def work(query: dict) -> None:
                nonlocal done, failed

                plan = plan_for(routing.get(query["Q_ID"]), query["Question"],
                                args.ignore_routing)
                transcript = load_transcript(args.transcripts,
                                             query["Video_ID"])

                # Never send a prompt with no evidence at all: a query the
                # router sent to the audio of a clip that turns out to have no
                # speech gets its video back.
                if not plan["video"] and not (transcript
                                              and transcript.get("text")):
                    plan = dict(plan, video=True, rescued=True)

                try:
                    record = rank_query(client, query, query["Options"],
                                        library, args.videos, args.transcripts,
                                        durations, args.passes, args.order,
                                        plan, args.visual_tokens)
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
                    if any(r != "full" for r in record["rungs"]):
                        flag += f" [{','.join(record['rungs'])}]"

                    print(f"  [{done}/{len(todo)}] {rate * 60:5.1f}/min "
                          f"eta {remaining:4.1f}h  "
                          f"{record['elapsed']:6.1f}s "
                          f"{record['fps']:4.2f}fps "
                          f"a{record['agreement']:.2f}{flag}  "
                          f"Q{record['Q_ID']}: "
                          f"{record['answers'][0]['Answer'][:44]}")

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
    """Sanity checks the validator does not make."""
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

    rungs: dict[str, int] = {}
    routes: dict[str, int] = {}

    for record in records:
        for rung in record.get("rungs") or ():
            rungs[rung] = rungs.get(rung, 0) + 1
        for route in record.get("routes") or ():
            routes[route] = routes.get(route, 0) + 1

    unanimous = sum(1 for r in records
                    if r["n_replies"] > 1 and r["agreement"] >= 0.999)
    multi = sum(1 for r in records if r["n_replies"] > 1)

    print(f"\nquality checks over {len(records)} ranked queries")
    print(f"  passes completed per query, mean "
          f"{sum(r['n_replies'] for r in records) / len(records):.2f}")
    print(f"  reply parsed from: "
          f"{', '.join(f'{k} {v}' for k, v in sorted(routes.items()))}")
    print(f"  degrade rungs used: "
          f"{', '.join(f'{k} {v}' for k, v in sorted(rungs.items()))}")

    if multi:
        print(f"  passes unanimous on rank 1:         {unanimous} of {multi} "
              f"({100 * unanimous / multi:.1f}%)")

    if counted:
        expected = sum(1 / len(by_qid[r["Q_ID"]]["Options"]) for r in records
                       if r["Q_ID"] in by_qid) / counted
        print(f"  rank 1 is the officially FIRST option: {first_official} "
              f"({100 * first_official / counted:.1f}%, chance "
              f"{100 * expected:.1f}%)")
        print(f"  rank 1 is the officially LAST option:  {last_official} "
              f"({100 * last_official / counted:.1f}%)")

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
    print(f"  queries ranked on a routed plan:    "
          f"{sum(1 for r in records if r.get('routed'))}"
          f"  (rescued {sum(1 for r in records if r.get('rescued'))})")

    times = sorted(r["elapsed"] for r in records if r["elapsed"])

    if times:
        print(f"  inference per query: mean {sum(times) / len(times):6.2f}s  "
              f"median {times[len(times) // 2]:.2f}s  "
              f"p90 {times[int(0.9 * len(times))]:.2f}s  "
              f"total {sum(times) / 3600:.2f} h")

    prep = sum(r.get("prep", 0.0) for r in records)

    if prep:
        print(f"  ffmpeg pre-shrinking:               {prep / 3600:.2f} h")


if __name__ == "__main__":
    sys.exit(main())

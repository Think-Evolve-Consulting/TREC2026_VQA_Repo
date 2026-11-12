#!/usr/bin/env python3
"""Answer the Task 1 questions with Qwen3-VL-8B-Instruct, ten prompts per query.

One half of the two-model ensemble for TREC 2026 Task 1; the other is
``generate.answers.py`` (Gemma 4 over extracted frames). Both now ask each
question ten ways -- the original plus nine paraphrases from
``generate.query.variations.py`` -- keep three candidates per call, and record
the vote, the self-reported confidence and the token logprobs behind every
candidate. ``fuse.answers.py`` pools the two 30-candidate sets and ranks the
60 into the ten slots the submission needs. See ENSEMBLE_PLAN.md.

This targets a vLLM OpenAI-compatible server that takes video natively, with
the request contents driven by ``query.routing.task1.json`` instead of sending
every input to every query.

**Two changes on 15 Aug, both measured rather than assumed.** The checkpoint
moved from ``-Thinking`` to ``-Instruct``: regressed over the 2,368
video-attached queries of the 13 Aug run, ``elapsed = 3.63 s + 40.18 s per
1000 completion tokens`` (R^2 = 0.876), and -Thinking spent a median 1,423
tokens per call, so decode was 94% of the wall clock and ten prompts would have
cost ~109 h against ~7.3 h without it. And the single k=10 call became ten k=3
calls: that one call truncated on 62 queries, returned nothing parseable on 60
of them, and padded all ten slots from the fallback pool each time -- 1,596
dead slots of 24,430, against 24 for the Gemma 4 pass running the same
arrangement this now adopts.

Output is the submission file ``VQA/validate.ag.run.py`` accepts verbatim::

    [{"Q_ID": 1, "Video_ID": "8384.webm",
      "Answers": [{"Rank": 1, "Answer": "...", "Time": 1.234}, ...]}]

Five things drive the design:

**Video_ID is not a filename.** 868 of the 2,443 queries name a bare YouTube
ID, and ``download_test_videos.py`` wrote those as ``<Video_ID>.mp4`` beside
the released ``.webm`` clips (SESSION.2026-08-11.task1-pipeline.md, §3).
``generate.answers.cosmos.py`` resolves ``videos_dir / Video_ID`` literally,
so all 852 downloaded mp4s were invisible to it: its cache shows 1,098 of
2,443 queries answered with no video at all, 469 of them with no transcript
either, where only **16** queries genuinely have no media on disk. This script
applies the documented convention, so 2,427 queries get their video.

**The router decides what each query is sent.** ``Pipeline`` from
``query.routing.task1.json`` maps onto request contents: ``frames`` attaches
the video, ``asr_transcript`` attaches the Whisper transcript,
``dense_timestamped_frames`` doubles the frame budget, ``frame_ocr`` raises
the per-frame pixel budget and adds a read-the-screen rule. ``Inference`` and
``Temporal`` still set the generation budget, though it is now headroom for a
wordier answer rather than a reasoning allowance -- the Instruct checkpoint
emits only the JSON. ``--ignore-routing`` sends everything to everything at a
fixed budget -- the ablation the routed run has to beat.

**The frame budget has to be computed against the context window.** vLLM's
Qwen3-VL path reads ``size.longest_edge`` as a per-frame pixel ceiling
(``qwen3_vl.py``: ``max_pixels=image_processor.size["longest_edge"]``), so
frames x pixels-per-frame is what overflows ``--max-model-len``, not fps
alone. ``video_mm_kwargs()`` divides the visual budget by the frames the
chosen fps will actually produce. ``fps`` and ``num_frames`` are mutually
exclusive in the processor (it raises), so only ever one is sent.

**Every request degrades rather than failing.** A 400 walks down a ladder --
shrink the visual budget, then drop ``mm_processor_kwargs`` and let the server
default, then drop the video and answer from the transcript -- and the step
that was reached is recorded per query. A run of 2,443 queries against a model
nobody here has run before should not lose a query to one bad clip, and
``preflight()`` settles the uncertain request options against the live server
before the run starts instead of discovering them 400 queries in.

**Time** is the field the track defines as "time to generate this answer", in
seconds: the wall clock of the call that produced the answer, plus its share of
any ffmpeg pre-shrink this query paid for (usually 0). A slot filled from the
generic fallback pool carries only the preparation share. Same convention as
``generate.answers.py``, so the two ensembles' Times are comparable when
``fuse.answers.py`` pools them.

Run this from WSL2 (not Windows): the server reads the video off the path in
the request, so the path has to be one it can open, and ``/mnt/e/...`` is the
same file as ``E:\\...``. Usage:

    python3 generate.answers.qwen3vl.py --limit 20      # smoke test, measure
    python3 generate.answers.qwen3vl.py                 # everything, resumable
    python3 generate.answers.qwen3vl.py --only 1 3 264  # named Q_IDs
    python3 generate.answers.qwen3vl.py --ignore-routing  # the ablation
    python3 generate.answers.qwen3vl.py --rebuild       # re-emit run from cache

See VQA/QWEN3_VL_SETUP.md for the server side.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from openai import APIStatusError, BadRequestError, OpenAI

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = ROOT / "vqa.2026.testingDataset.Task1.json"
DEFAULT_ROUTING = ROOT / "query.routing.task1.json"
DEFAULT_VIDEOS = ROOT / "videos_TEST_VQA" / "videos"
DEFAULT_FRAME_INDEX = ROOT / "videos_TEST_VQA" / "frames" / "index.json"
DEFAULT_TRANSCRIPTS = ROOT / "videos_TEST_VQA" / "transcription_openwhisper"
DEFAULT_OUT = ROOT / "qwen3vl.run.task1.json"
DEFAULT_CACHE = ROOT / "answers.task1.qwen3vl.cache.jsonl"
DEFAULT_VARIATIONS = ROOT / "query.variations.task1.json"

# The Instruct checkpoint, not -Thinking. Measured over the 2,368
# video-attached queries of the 13 Aug run:
#
#     elapsed = 3.63 s + 40.18 s per 1000 completion tokens   (R^2 = 0.876)
#
# -Thinking spends a median 1,423 completion tokens to emit ten short answers,
# so decode is 94% of the wall clock and the ten-prompt ensemble below would
# cost ~109 h over 2,443 queries. Without the think block the same ensemble is
# ~7.3 h. The reasoning was also never visible to this script -- reasoning_chars
# was 0 on all 2,443 rows, so those tokens landed in `content` and had to be
# parsed around.
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# 8009, not the 8008 the Cosmos server uses: the two cannot share the card, but
# they can share a config file and a shell history without one silently
# answering the other's requests.
DEFAULT_BASE_URL = "http://localhost:8009/v1"

# Lowered from the 1.0 this script used when it made one call per query. That
# number was right when a single reply's ten answers had to be genuinely
# different from each other; now the diversity comes from ten phrasings and
# the sampling should concentrate the vote instead, which is the same reasoning
# that put generate.answers.py at 0.6.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_PRESENCE_PENALTY = 0.0

# Concurrent in-flight requests, and the single biggest lever on the run's
# wall clock. vLLM batches continuously on the one GPU; this just keeps the
# queue fed. Measured over the same 8 queries against the config in
# VQA/QWEN3_VL_SETUP.md:
#
#   workers 1   280 s   38 tok/s decode    23.7 h projected over 2,443
#   workers 4   133 s  310 tok/s decode    11.0 h projected
#   workers 8   143 s                      12.0 h projected
#
# Decode at batch 1 leaves the 3090 almost idle, which is why 4x costs nothing
# and buys 2x. 8 gives it back. The server reports "Maximum concurrency 1.19x"
# at startup, which sounds like a ceiling of one -- it is not: that figure
# assumes every request fills the 24,576-token window, while real ones here
# are 600-1,450 tokens. 8 concurrent requests ran with zero preemption and no
# change in peak VRAM.
DEFAULT_WORKERS = 4

ANSWERS_PER_QUERY = 10
PROMPTS_PER_QUERY = 10
CANDIDATES_PER_PROMPT = 3
"""Ten phrasings of the question, three candidates each, pooled to 30.

The single k=10 call this script used before truncated on 62 queries and
returned nothing parseable on 60 of them, losing all ten slots each time -- a
ten-item JSON is simply a long thing to ask a model to finish. Three is short
enough to reach, and ten calls fail independently where one call did not.
Matches generate.answers.py, whose identical arrangement pads 24 slots in
24,430 against this script's 1,596."""

# A candidate offered as the model's first choice counts for more than its
# third. Same weights as generate.answers.py so a Gemma 4 candidate and a
# Qwen3-VL candidate arrive at fuse.answers.py on the same scale.
POSITION_WEIGHT = (1.0, 0.5, 0.25)

# Generation budget by routing verdict, keyed by (Inference, Temporal). These
# were 4096/8192/12288 for the -Thinking checkpoint, where the budget had to
# cover a reasoning trace and cutting it off cost the whole reply. The Instruct
# checkpoint emits only the JSON, so three short answers need a couple of
# hundred tokens; the tiers survive as headroom for a wordier query rather than
# as a reasoning allowance. Freeing that context also raises the visual budget
# video_mm_kwargs() computes, which is a straight gain in frames per clip.
BUDGET_POLICY = {
    (False, False): "brief",
    (False, True): "standard",
    (True, False): "standard",
    (True, True): "deep",
}
BUDGETS = ("brief", "standard", "deep")
MAX_TOKENS_BY_BUDGET = {"brief": 320, "standard": 448, "deep": 640}
DEFAULT_BUDGET = "standard"
"""Used for a query the router never saw, and by --ignore-routing."""

# Adaptive fps: aim for a fixed frame budget per video regardless of length, so
# a 14 s clip and a 6.5-minute one both fit the context window. Duration
# distribution in videos_TEST_VQA (n=1,940): median 10.3 s, p90 56 s, p99
# 233 s, max 389 s -- only the long tail needs this.
TARGET_FRAMES = 64
DENSE_TARGET_FRAMES = 96
"""For the 886 queries the router marked dense_timestamped_frames: ordering and
timing questions, where the gap between samples is the thing that loses the
answer."""
MAX_FPS = 4.0
MIN_FPS = 0.2

# Visual-token geometry, from the processor's own config: a token is
# (patch_size 16 * merge_size 2)^2 = 1024 pixels, and temporal_patch_size 2
# means frames are merged in pairs, so N frames cost N/2 x per-frame tokens.
PIXELS_PER_TOKEN = 1024
TEMPORAL_MERGE = 2
MIN_RESAMPLE_FRAMES = 16
"""Below this, ask for the pixel budget but not the frame rate.

``do_sample_frames`` makes the processor re-sample the decoded video at the
requested fps, and on a clip too short to yield many frames that call fails
outright -- ``Failed to apply Qwen3VLProcessor``. Measured against this corpus,
the boundary is exact and sharp: every video whose request worked out to <= 11
frames failed, every one at >= 12 succeeded (1.07-2.74 s FAIL, 3.04 s+ OK, all
at 4 fps). 16 leaves margin. Nothing is lost by omitting it there -- a clip
that short is only a handful of frames at the server's own default rate, and
``size`` still caps what each one costs."""

MIN_FRAME_TOKENS = 128
"""The processor's own default floor (size.shortest_edge 131,072 px)."""
MAX_FRAME_TOKENS = 768
"""The processor's own default ceiling (size.longest_edge 786,432 px). Not
raised: more pixels per frame buys detail this corpus rarely needs and costs
context every query pays for."""
OCR_FRAME_TOKENS = 1024
"""The 57 frame_ocr queries are the exception -- reading a sign is exactly
what per-frame resolution buys -- so they may exceed the default ceiling."""
TEXT_TOKEN_RESERVE = 2048
"""Held back from the visual budget for the system prompt, the transcript
(some run long) and the question."""
VISUAL_BUDGET_SAFETY = 0.8
"""The frame count is predicted from container metadata that is occasionally
wrong, and a request over max_model_len is a hard 400. Under-filling the window
costs a little resolution; overfilling costs a retry."""

# vLLM's video IO layer is started with --media-io-kwargs num_frames=-1 (the
# Cosmos setup's own recommendation, needed by this model family for correct
# per-frame timestamp embedding). That decodes every native-resolution frame
# into a raw array *before* the fps subsampling above happens, so a long
# and/or high-res source allocates gigabytes of host RAM up front. A 320 s,
# 2048x1152, 24 fps clip ("ChainedforLife-14.webm", real example from this
# corpus) hit a 50.6 GiB allocation and crashed the request. Videos whose
# native decode footprint would exceed this are pre-shrunk with ffmpeg;
# the cheap majority pass through untouched.
RAW_DECODE_BYTES_LIMIT = 4 * 1024**3
TRANSCODE_CACHE_DIRNAME = ".vlm_transcoded_cache"
"""Lives under the videos dir, inside --allowed-local-media-path, so the server
can still read it by path."""
LEGACY_TRANSCODE_CACHE_DIRNAME = ".cosmos_transcoded_cache"
"""The Cosmos run already shrank 196 of these clips at the same target fps.
Reused when the name matches rather than re-encoding them."""

SYSTEM_PROMPT = "You are a helpful assistant."

# Adapted from generate.answers.cosmos.py's template, which adapted
# generate.answers.py's: same rules (short answers, genuinely distinct,
# "unknown" over guessing). Now also asks for a confidence, as Gemma 4 does, so
# both models hand fuse.answers.py the same fields. On the Instruct checkpoint
# there is no think block to reason in, so the reply is the JSON and nothing
# else.
PROMPT_TEMPLATE = """{context}Question: {question}

Give your {k} most likely answers to this question, ranked best first.

Rules:
- Answer the question and nothing else. Do not restate the question, do not
  describe the video, and do not explain your reasoning.
- Keep every answer very short: a phrase or one short sentence. Most good
  answers are under ten words.
- The {k} answers must be genuinely different possibilities, not rewordings
  of each other.
- Use the evidence you were given. If it does not show the answer, say
  "unknown" rather than guessing.{extra}

End your reply with ONLY this JSON on the final line, holding exactly {k}
items ranked best first: {{"answers": ["...", ...], "confidence": 0.0}}
where confidence is how confident you are in the first answer, 0.0 to 1.0."""

# Appended to the rules when the router flagged the query. There is no OCR
# artefact on disk -- route.query.inputs.py names the step `frame_ocr` for a
# pass that was never built -- so the instruction leans on the model reading
# the frames itself, which is what the raised pixel budget is for.
OCR_RULE = ("\n- The answer depends on text visible on screen. Read any signs, "
            "captions, labels or subtitles carefully and quote them exactly.")
TEMPORAL_RULE = ("\n- The answer depends on ordering or timing. Use the frame "
                 "timestamps to work out what happens before what.")

LEAD_IN_RE = re.compile(r"^\s*(?:answer|a)\s*[:\-]\s*", re.IGNORECASE)
ENUM_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
JSON_BLOB_RE = re.compile(r"\{.*?\}", re.DOTALL)

# A 400 whose text names the context window is a budget problem and worth
# retrying smaller; any other 400 is the processor rejecting the request and
# retrying it identically is a waste of a minute.
CONTEXT_ERROR_RE = re.compile(
    r"maximum context|context length|max_model_len|longer than the maximum|"
    r"too long|exceeds.*length",
    re.IGNORECASE,
)

# Ground truth answers in VQA/vqa.gt.csv average 8.4 words, median 8, p90 17,
# max 42 (verified 14 Aug: mean 8.38, 17.1% single-word, n=2,000). Anything far
# past that is the model narrating instead of answering. Note that file is the
# *2025* ground truth and shares no videos with the 2026 test set -- it is a
# guide to annotator style, not a reference this run is scored against.
MAX_ANSWER_WORDS = 40
MAX_ANSWER_CHARS = 300

# A refusal is a legitimate thing for the model to say, but it is never the
# ground truth, so it scores ~0. Ranking one above a substantive candidate
# wastes a slot, so these sort last among the model's own answers -- kept, not
# discarded. Identical to generate.answers.py:NON_ANSWER_RE; "none", "no one"
# and "zero" are deliberately absent, since for a counting or "who did X"
# question they are the answer.
NON_ANSWER_RE = re.compile(
    r"^(?:unknown|unclear|n/?a|no idea|cannot be determined"
    r"|(?:it |the answer |this )?(?:is )?(?:not |un)(?:known|clear|shown"
    r"|visible|determined|specified)"
    r"|(?:it |this )?cannot be (?:determined|answered|seen)"
    r"|(?:the )?video does not show(?: this)?"
    r"|not (?:shown|visible|clear|stated)(?: in the video)?)"
    r"[.!]?$",
    re.IGNORECASE,
)

# Same pool as the Gemma 4 pipeline: defensible non-answers rather than
# fabricated facts, used only when the pooled candidates fall short of ten.
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


# ---------------------------------------------------------------------------
# Text handling -- identical rules to the other two pipelines, so the three
# runs stay directly comparable.
# ---------------------------------------------------------------------------

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

    # A runaway answer is the model narrating the clip; keep its first sentence
    # rather than dropping a candidate that may still carry the right fact.
    if len(text.split()) > MAX_ANSWER_WORDS:
        head = re.split(r"(?<=[.!?])\s+", text)[0]
        text = head if head.split() else text
        text = " ".join(text.split()[:MAX_ANSWER_WORDS])

    return text[:MAX_ANSWER_CHARS].strip()


def mmss(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def extract_answers(content: str, k: int) -> list[str]:
    """The answer list alone, for callers that do not want the confidence."""
    return extract_reply(content, k)[0]


def extract_reply(content: str, k: int) -> tuple[list[str], float | None]:
    """Pull the ranked answers and the confidence out of the final-line JSON.

    The generation is not grammar-constrained by default (see --json-schema),
    and the model sometimes wraps prose around the JSON despite the
    instruction, so the last parseable blob wins -- the instruction puts the
    real answer on the final line.
    """
    candidates = [content] + JSON_BLOB_RE.findall(content) \
        + re.findall(r"\{.*\}", content, re.DOTALL)

    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

        answers = data.get("answers") if isinstance(data, dict) else None

        if isinstance(answers, list) and answers:
            confidence = data.get("confidence")
            confidence = float(confidence) \
                if isinstance(confidence, (int, float)) else None

            return ([tidy(a) for a in answers[:k] if str(a).strip()],
                    confidence)

    return [], None


def answer_logprobs(logprobs: list | None, content: str,
                    answers: list) -> list[dict | None]:
    """Per-answer token statistics, aligned with `answers`.

    Only the tokens inside an answer string carry signal: with --json-schema
    the structural tokens are grammar-forced to probability ~1, and without it
    they are boilerplate the model has been told to emit. So each answer is
    located in the raw content and scored over the tokens overlapping its span.

    Returns one dict per answer, or None where the span could not be found -- a
    missing score is honest; a whole-sequence average would mostly be measuring
    the JSON. Kept identical to the copy in generate.answers.py so a candidate
    from either model reaches fuse.answers.py on the same scale.
    """
    if not logprobs or not content:
        return [None] * len(answers)

    spans: list[tuple[int, int, float]] = []
    cursor = 0

    for token in logprobs:
        text = getattr(token, "token", None) or ""
        value = getattr(token, "logprob", None)

        if value is None:
            cursor += len(text)
            continue

        spans.append((cursor, cursor + len(text), float(value)))
        cursor += len(text)

    out: list[dict | None] = []
    search_from = 0

    for answer in answers:
        needle = str(answer)
        start = -1

        for form in (json.dumps(needle, ensure_ascii=False)[1:-1], needle):
            if form:
                start = content.find(form, search_from)

            if start != -1:
                needle = form
                break

        if start == -1:
            out.append(None)
            continue

        end = start + len(needle)
        search_from = end

        values = [lp for begin, stop, lp in spans
                  if begin < end and stop > start]

        if not values:
            out.append(None)
            continue

        out.append({
            "mean_logprob": round(sum(values) / len(values), 5),
            "min_logprob": round(min(values), 5),
            "n_tokens": len(values),
        })

    return out


def pool_candidates(replies: list[dict]) -> list[dict]:
    """Vote over every candidate from every prompt, best first.

    Identical policy to generate.answers.py:pool_candidates -- a first-choice
    candidate outweighs a third-choice one, confidence modulates the vote
    without cancelling it, refusals sort last -- so the two models' candidates
    are directly comparable when fuse.answers.py pools them again.
    """
    clusters: dict[str, dict] = {}

    for reply in replies:
        confidence = reply.get("confidence")
        confidence = confidence if isinstance(confidence, (int, float)) else 0.5
        confidence = min(max(float(confidence), 0.0), 1.0)
        logprobs = reply.get("logprobs") or []

        for position, raw in enumerate(reply.get("answers") or []):
            text = tidy(raw)

            if not text:
                continue

            weight = POSITION_WEIGHT[min(position, len(POSITION_WEIGHT) - 1)]
            score = weight * (0.5 + 0.5 * confidence)
            key = normalize(text)
            stats = logprobs[position] if position < len(logprobs) else None

            cluster = clusters.setdefault(key, {
                "answer": text,
                "score": 0.0,
                "votes": 0,
                "best": -1.0,
                "first_index": reply["index"],
                "elapsed": reply["elapsed"],
                "refusal": bool(NON_ANSWER_RE.match(text)),
                "confidence_max": 0.0,
                "best_position": position,
                "prompts": [],
                "_logprobs": [],
            })

            cluster["score"] += score
            cluster["votes"] += 1
            cluster["confidence_max"] = max(cluster["confidence_max"],
                                            confidence)
            cluster["best_position"] = min(cluster["best_position"], position)

            if reply["index"] not in cluster["prompts"]:
                cluster["prompts"].append(reply["index"])

            if stats and stats.get("mean_logprob") is not None:
                cluster["_logprobs"].append(stats["mean_logprob"])

            if score > cluster["best"]:
                cluster["best"] = score
                cluster["answer"] = text
                cluster["elapsed"] = reply["elapsed"]

    for cluster in clusters.values():
        values = cluster.pop("_logprobs")
        cluster["n_logprob"] = len(values)
        cluster["logprob_mean"] = round(sum(values) / len(values), 5) \
            if values else None
        cluster["logprob_best"] = round(max(values), 5) if values else None

    return sorted(clusters.values(),
                  key=lambda c: (c["refusal"], -c["score"], c["first_index"],
                                 c["answer"]))


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
    """Pad a shortfall with generic, defensible fallback answers.

    The validator rejects fewer than ten answers and rejects duplicates, and
    eval.vqa.ag.py scores a query as the *mean* similarity across all ten, so
    a slot filled with an invented fact costs real score where a defensible
    non-answer only scores zero.
    """
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


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_video(videos_dir: Path, video_id: str) -> Path:
    """The file a Video_ID names, per SESSION.2026-08-11.task1-pipeline.md §3.

    A released clip carries its ``.webm`` extension in the query file; a
    YouTube ID does not, and download_test_videos.py wrote those as
    ``<Video_ID>.mp4``. Resolving the raw Video_ID instead silently answers
    868 queries with no video -- see the module docstring.
    """
    name = video_id if video_id.endswith(".webm") else video_id + ".mp4"
    return videos_dir / name


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


def load_durations(index_path: Path) -> dict[str, float]:
    """Video stem -> duration, from the frame index extract.frames.py wrote.

    The frames themselves are unused here -- this model takes the video -- but
    the index already holds a probed duration for all 1,940 clips on disk, and
    fps planning needs one *before* the request. The Cosmos pass took duration
    from the transcript, which does not exist for 1,080 queries, so those all
    silently planned at a fixed 2 fps.
    """
    if not index_path.exists():
        return {}

    try:
        videos = load_json(index_path).get("videos") or {}
    except (json.JSONDecodeError, OSError):
        return {}

    return {stem: entry["duration_sec"] for stem, entry in videos.items()
            if entry.get("duration_sec")}


def load_routing(path: Path) -> dict[int, dict]:
    """Q_ID -> routing verdict from query.routing.task1.json."""
    if not path.exists():
        return {}

    return {int(entry["Q_ID"]): entry for entry in load_json(path)}


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


# ---------------------------------------------------------------------------
# Routing -> request plan
# ---------------------------------------------------------------------------

def plan_for(routing: dict | None, ignore_routing: bool,
             budget_override: str | None) -> dict:
    """What this query is sent, and how much thinking it is paid for.

    With --ignore-routing (or no routing entry at all) every input is attached
    at a fixed budget -- the ablation the routed run has to beat, and the safe
    default for a query the router never saw.
    """
    if ignore_routing or not routing:
        plan = {"video": True, "transcript": True, "dense": True,
                "ocr": False, "temporal": False, "budget": DEFAULT_BUDGET,
                "routed": False}
    else:
        pipeline = set(routing.get("Pipeline") or ())
        inference = bool(routing.get("Inference"))
        temporal = bool(routing.get("Temporal"))

        plan = {
            "video": "frames" in pipeline,
            "transcript": "asr_transcript" in pipeline,
            "dense": "dense_timestamped_frames" in pipeline,
            "ocr": "frame_ocr" in pipeline,
            "temporal": temporal,
            "budget": BUDGET_POLICY[(inference, temporal)],
            "routed": True,
        }

        # An OCR question is about pixels whatever else the router said, and a
        # dense-sampling verdict needs something to sample densely. Neither is
        # allowed to arrive without the video it depends on.
        if plan["ocr"] or plan["dense"]:
            plan["video"] = True

    if budget_override:
        plan["budget"] = budget_override

    plan["max_tokens"] = MAX_TOKENS_BY_BUDGET[plan["budget"]]
    # Set later, per query: one routed to the audio alone still gets the video
    # when the audio it was routed to turns out not to exist.
    plan["rescued"] = False

    return plan


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
                    max_tokens: int, shrink: int = 0) -> dict:
    """fps and per-frame pixel budget for one request.

    vLLM reads ``size.longest_edge`` as a *per-frame* ceiling, so what has to
    fit the window is frames x tokens-per-frame (halved, because the processor
    merges frames in pairs). ``shrink`` halves the result, once per rung of the
    degrade ladder.
    """
    fps = adaptive_fps(duration_sec, plan["dense"]) / (2 ** shrink)
    fps = max(MIN_FPS, round(fps, 3))

    frames = expected_frames(duration_sec, fps)
    visual_budget = max(max_model_len - max_tokens - TEXT_TOKEN_RESERVE, 1024)
    ceiling = OCR_FRAME_TOKENS if plan["ocr"] else MAX_FRAME_TOKENS

    per_frame = int(visual_budget * VISUAL_BUDGET_SAFETY * TEMPORAL_MERGE
                    / max(frames, 1)) // (2 ** shrink)
    per_frame = max(MIN_FRAME_TOKENS, min(ceiling, per_frame))

    kwargs = {
        "size": {
            "shortest_edge": MIN_FRAME_TOKENS * PIXELS_PER_TOKEN,
            "longest_edge": per_frame * PIXELS_PER_TOKEN,
        },
    }

    # fps is only honoured when do_sample_frames is set -- sending it alone is
    # measurably a no-op -- and the pair is unsafe on very short clips, so the
    # two travel together or not at all. fps and num_frames are mutually
    # exclusive in the processor (it raises), so num_frames is never sent.
    if frames >= MIN_RESAMPLE_FRAMES:
        kwargs["do_sample_frames"] = True
        kwargs["fps"] = fps

    return kwargs


def build_context(duration_sec: float | None, transcript: dict | None,
                  plan: dict, video_attached: bool,
                  audio_expected: bool) -> str:
    """The text block that precedes the question."""
    parts: list[str] = []

    if video_attached:
        if duration_sec:
            parts.append(
                f"You are shown the video for this question. It is "
                f"{duration_sec:.0f} seconds long ({mmss(duration_sec)})."
            )
        else:
            parts.append("You are shown the video for this question.")
    elif plan["video"] or plan["rescued"]:
        parts.append(
            "The video for this question is unavailable, so answer from what "
            "else you are given, as plausibly as you can."
        )
    else:
        # Routed to the audio alone: say so, or the model reads the missing
        # video as a failure and hedges every answer.
        parts.append(
            "You are not shown the video for this question. Answer from the "
            "transcript of its audio."
        )

    if transcript and transcript.get("text"):
        language = transcript.get("language") or "unknown"
        parts.append(
            f"Spoken audio (automatic transcript, language: {language}):\n"
            f"\"{transcript['text']}\""
        )
    elif audio_expected:
        # Only assertable when the audio was asked for and was not there. A
        # vision-routed query is not sent the transcript at all, and telling
        # the model such a video has no speech would simply be false -- 1,941
        # of them have one on disk.
        parts.append("The video has no detected speech.")

    return "\n\n".join(parts) + "\n\n"


def build_prompt(question: str, context: str, plan: dict, k: int) -> str:
    extra = ""

    if plan["ocr"]:
        extra += OCR_RULE
    if plan["temporal"]:
        extra += TEMPORAL_RULE

    return PROMPT_TEMPLATE.format(context=context, question=question, k=k,
                                  extra=extra)


# ---------------------------------------------------------------------------
# Media preparation
# ---------------------------------------------------------------------------

class VideoPrep:
    """Probes and (rarely) pre-shrinks videos, once each per run.

    The Cosmos pass re-probed a video for every query on it; 1,956 videos serve
    2,443 queries, so that is ~500 wasted ffprobe calls. Results are memoised
    here and the transcode cache is shared with any earlier run's.
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

    def safe_path(self, path: Path, target_fps: float) -> tuple[Path, float]:
        """A path safe for the server's num_frames=-1 full decode, and the
        seconds spent getting there (0 when nothing had to be done).

        Falls back to the original path if probing or transcoding fails --
        degrading to the OOM risk rather than dropping the query.
        """
        safe_fps = round(target_fps, 3) or 1.0
        key = (path, safe_fps)

        with self._lock:
            if key in self._safe:
                return self._safe[key]

        info = self.probe(path)
        result = (path, 0.0)

        if info and info["width"] and info["height"] and info["fps"]:
            raw = (info["duration"] * info["fps"] * info["width"]
                   * info["height"] * 3)

            if raw > self.decode_limit and self.transcode:
                result = self._shrink(path, min(safe_fps, info["fps"]))

        with self._lock:
            self._safe[key] = result

        return result

    def _shrink(self, path: Path, fps: float) -> tuple[Path, float]:
        fps = round(fps, 3) or 1.0
        name = f"{path.stem}.f{fps}.mp4"
        cached = self._cached(path.stem, fps)

        if cached:
            return cached, 0.0

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_dir / name
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
        """An already-shrunk copy of this video good enough to sample at `fps`.

        A cached file carries the frame rate it was written at in its name.
        One at or above the rate wanted is fine -- the processor subsamples it
        further -- so the cheapest sufficient copy wins, which is what lets the
        196 clips the Cosmos run already shrank be reused instead of
        re-encoded. One below it would silently cost frames, so it is ignored.
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

class Qwen3VLClient:
    """Thin wrapper over the OpenAI-compatible Qwen3-VL endpoint.

    Every call walks a degrade ladder rather than failing: shrink the visual
    budget, drop the processor kwargs, drop the video. The rung reached is
    returned so the run can be audited afterwards.
    """

    LADDER = ("full", "shrink1", "shrink2", "no_dsf", "no_mm_kwargs",
              "no_video")

    TRUNCATION_RETRIES = 2
    """Retries for a reply that hit max_tokens mid-JSON.

    Until 15 Aug there were none: answer() returned on the first 200 and never
    looked at finish_reason, so a truncated reply flowed through
    extract_reply() as [] and padded all ten slots from FALLBACK_ANSWERS. That
    is 62 queries and 600 dead slots of the 13 Aug run. The ladder below only
    ever caught request *errors*, which a truncation is not."""

    TOKEN_CEILING = 2048
    """Ceiling on the doubling above. The visual budget was computed against
    the original max_tokens leaving TEXT_TOKEN_RESERVE (2048) spare, so growing
    the output past that would start costing 400s instead of preventing them."""

    def __init__(self, base_url: str, model: str, temperature: float,
                 top_p: float, top_k: int, presence_penalty: float,
                 timeout: float, retries: int, max_model_len: int | None,
                 send_mm_kwargs: bool = True,
                 json_schema: bool = False) -> None:
        self.client = OpenAI(api_key="EMPTY", base_url=base_url,
                             timeout=timeout, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.retries = retries
        self.send_mm_kwargs = send_mm_kwargs
        self.json_schema = json_schema
        self._max_model_len = max_model_len
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
                    or 32768
        return self._max_model_len

    def resolve_model(self) -> str:
        """The id the server actually serves, so --model can stay a nickname."""
        models = self.client.models.list()
        ids = [m.id for m in models.data]

        if self.model in ids:
            return self.model

        if ids:
            self.model = ids[0]

        return self.model

    def schema(self, k: int) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "ranked_answers",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": k,
                            "maxItems": k,
                        },
                        "confidence": {"type": "number"},
                    },
                    "required": ["answers", "confidence"],
                    "additionalProperties": False,
                },
            },
        }

    def complete(self, prompt: str, video_uri: str | None,
                 mm_kwargs: dict | None, max_tokens: int, k: int):
        content: list[dict] = []

        if video_uri:
            content.append({"type": "video_url",
                            "video_url": {"url": video_uri}})

        content.append({"type": "text", "text": prompt})

        extra: dict = {"top_k": self.top_k}

        if mm_kwargs and self.send_mm_kwargs:
            extra["mm_processor_kwargs"] = mm_kwargs

        kwargs: dict = {}

        if self.json_schema:
            kwargs["response_format"] = self.schema(k)

        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            # Per-token logprobs for the probability score. top_logprobs stays
            # unset: only the sampled token's own value is wanted, and the
            # alternatives would multiply the response size for no use here.
            logprobs=True,
            extra_body=extra,
            **kwargs,
        )

    def answer(self, prompt_for, video_uri: str | None, mm_kwargs_for,
               max_tokens: int, k: int) -> dict:
        """One query, degrading until something comes back.

        ``prompt_for(video_attached)`` and ``mm_kwargs_for(shrink)`` are
        callbacks because both change as the ladder descends -- a prompt that
        says "you are shown the video" must not survive dropping it.
        """
        rung = 0 if video_uri else self.LADDER.index("no_video")
        attempts = 0
        grown = 0
        budget = max_tokens
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
                # 768-frame default.
                mm_kwargs = {k: v for k, v in mm_kwargs.items()
                             if k not in ("do_sample_frames", "fps")}

            started = time.monotonic()

            try:
                completion = self.complete(
                    prompt_for(attach),
                    video_uri if attach else None,
                    mm_kwargs,
                    budget,
                    k,
                )
                elapsed = time.monotonic() - started
                choice = completion.choices[0]
                message = choice.message
                usage = getattr(completion, "usage", None)

                # A truncated reply is a 200 whose JSON never closed, so it
                # parses to nothing and costs all ten slots. Give it more room
                # and ask again before accepting that.
                if choice.finish_reason == "length" \
                        and grown < self.TRUNCATION_RETRIES \
                        and budget < self.TOKEN_CEILING:
                    grown += 1
                    budget = min(budget * 2, self.TOKEN_CEILING)
                    continue

                content = message.content or ""
                tokens = getattr(getattr(choice, "logprobs", None), "content",
                                 None)

                return {
                    "content": content,
                    "reasoning": getattr(message, "reasoning_content", None)
                    or "",
                    "finish_reason": choice.finish_reason,
                    "logprobs": tokens,
                    "elapsed": elapsed,
                    "video_attached": attach,
                    "degrade": step,
                    "mm_kwargs": mm_kwargs,
                    "grown": grown,
                    "max_tokens": budget,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens",
                                                 None),
                }
            except BadRequestError as exc:
                last = exc
                detail = str(exc)

                if CONTEXT_ERROR_RE.search(detail) and step in (
                        "full", "shrink1"):
                    # Too many visual tokens for the window: halve and retry,
                    # rather than losing the video over an arithmetic miss on
                    # a container whose metadata lied about its frame count.
                    rung += 1
                elif step in ("full", "shrink1", "shrink2"):
                    # Anything else is the processor refusing this request.
                    # Most often that is do_sample_frames on a clip with too
                    # few frames, so try without it before giving up anything
                    # that costs evidence.
                    rung = self.LADDER.index("no_dsf")
                else:
                    # no_dsf -> no_mm_kwargs -> no_video -> out of ideas.
                    rung += 1
            except APIStatusError as exc:
                last = exc

                # 5xx is usually the server still loading, or a transient
                # engine error; worth waiting for.
                if exc.status_code and exc.status_code < 500:
                    rung = self.LADDER.index("no_video")
                else:
                    time.sleep(2.0 * attempts)
            except Exception as exc:  # noqa: BLE001 - transient, worth retrying
                last = exc
                time.sleep(2.0 * attempts)

        raise RuntimeError(f"{type(last).__name__}: {last}")


# ---------------------------------------------------------------------------
# One query
# ---------------------------------------------------------------------------

def answer_query(client: Qwen3VLClient, query: dict, plan: dict,
                 videos_dir: Path, transcripts_dir: Path,
                 durations: dict[str, float], prep: VideoPrep,
                 want: int, prompts: list[str], candidates: int) -> dict:
    q_id = query["Q_ID"]
    video_id = query["Video_ID"]
    stem = Path(video_id).stem

    # Always read, not always sent: the router decides whether the transcript
    # goes in the prompt, but a video the ladder ends up dropping has to leave
    # something behind. Two ways a query would otherwise be sent with zero
    # evidence, both real: routed to the audio of a video that turns out to
    # have no speech, and routed to the video of a clip the processor refuses.
    transcript = load_transcript(transcripts_dir, video_id)
    has_speech = bool(transcript and transcript.get("text"))

    if not plan["video"] and not has_speech:
        plan = dict(plan, video=True, rescued=True)

    path = resolve_video(videos_dir, video_id)
    video_exists = path.exists()
    attach = plan["video"] and video_exists

    duration = durations.get(stem) \
        or (transcript or {}).get("duration_sec") or None

    video_uri = None
    prep_seconds = 0.0

    if attach:
        # Probe before choosing the fps, not after: an unknown duration makes
        # adaptive_fps fall back to a flat 2.0, which on a long clip asks for
        # far more frames than the server was profiled for and can OOM it at
        # request time. Every video on disk is in the frame index, so this is
        # the belt to that braces.
        if duration is None:
            duration = (prep.probe(path) or {}).get("duration") or None

        target_fps = adaptive_fps(duration, plan["dense"])
        safe_path, prep_seconds = prep.safe_path(path, target_fps)
        video_uri = safe_path.resolve().as_uri()

    def wants_audio(video_attached: bool) -> bool:
        return plan["transcript"] or not plan["video"] or not video_attached

    def prompt_builder(question: str):
        def prompt_for(video_attached: bool) -> str:
            audio = wants_audio(video_attached)
            context = build_context(duration, transcript if audio else None,
                                    plan, video_attached, audio)
            return build_prompt(question, context, plan, candidates)
        return prompt_for

    def mm_kwargs_for(shrink: int) -> dict:
        return video_mm_kwargs(duration, plan, client.max_model_len,
                               plan["max_tokens"], shrink)

    # Ten phrasings, three candidates each. The video was prepared once above
    # and its URI is reused, so vLLM re-serves the multimodal embedding from
    # cache rather than decoding the clip ten times.
    replies: list[dict] = []
    failures = 0
    truncated = 0
    grown = 0
    completion_tokens = 0
    worst_rung = -1
    last_reply: dict | None = None

    for index, question in enumerate(prompts):
        try:
            reply = client.answer(prompt_builder(question), video_uri,
                                  mm_kwargs_for, plan["max_tokens"],
                                  candidates)
        except Exception as exc:  # noqa: BLE001 - one prompt of ten
            failures += 1
            print(f"  Q_ID {q_id} prompt {index}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        last_reply = reply
        worst_rung = max(worst_rung, Qwen3VLClient.LADDER.index(
            reply["degrade"]))
        completion_tokens += reply["completion_tokens"] or 0
        grown += reply["grown"]

        if reply["finish_reason"] == "length":
            truncated += 1

        answers, confidence = extract_reply(reply["content"], candidates)

        replies.append({
            "index": index,
            "answers": answers,
            "confidence": confidence,
            "logprobs": answer_logprobs(reply["logprobs"], reply["content"],
                                        answers),
            "elapsed": reply["elapsed"],
        })

    if last_reply is None:
        raise RuntimeError(f"all {len(prompts)} prompts failed")

    ranked = pool_candidates(replies)
    chosen = [{"answer": c["answer"], "source": "model", "elapsed": c["elapsed"]}
              for c in ranked[:want]]
    chosen, fallback_used = fill_slots(chosen, want)

    parsed = sum(1 for r in replies if r["answers"])
    attached = last_reply["video_attached"]
    total_elapsed = sum(r["elapsed"] for r in replies)

    # "Time to generate this answer": the wall clock of the call that produced
    # it, plus this query's share of any ffmpeg pre-shrink. A fallback slot was
    # not generated by the model and carries only the preparation share.
    prep_share = prep_seconds / max(want, 1)

    return {
        "Q_ID": q_id,
        "Video_ID": video_id,
        "answers": [
            {"Rank": rank,
             "Answer": item["answer"],
             "Time": round((item.get("elapsed", 0.0)
                            if item["source"] == "model" else 0.0)
                           + prep_share, 3),
             "source": item["source"]}
            for rank, item in enumerate(chosen, 1)
        ],
        "model": client.model,
        "n_prompts": len(prompts),
        "n_replies": len(replies),
        "n_prompt_failures": failures,
        "n_raw_candidates": sum(len(r["answers"]) for r in replies),
        "pooled_unique": len(ranked),
        "fallback": fallback_used,
        "video_exists": video_exists,
        "video_attached": attached,
        "has_transcript": has_speech and wants_audio(attached),
        "transcript_on_disk": has_speech,
        "routed": plan["routed"],
        "rescued": plan["rescued"],
        "budget": plan["budget"],
        "dense": plan["dense"],
        "ocr": plan["ocr"],
        # Every pooled candidate, not just the ten that made the cut --
        # fuse.answers.py ranks across both models over these.
        "candidates": [
            {"answer": c["answer"], "score": round(c["score"], 4),
             "votes": c["votes"], "confidence_max": round(c["confidence_max"], 4),
             "logprob_mean": c["logprob_mean"],
             "logprob_best": c["logprob_best"],
             "n_logprob": c["n_logprob"],
             "best_position": c["best_position"],
             "prompts": sorted(c["prompts"]),
             "refusal": c["refusal"]}
            for c in ranked
        ],
        # "no_video" means the ladder gave the video up. A query the router
        # sent to the audio, and one whose video was never downloaded, are
        # both text-only by design rather than degraded requests. The rung
        # reported is the worst any of the ten calls fell to.
        "degrade": (Qwen3VLClient.LADDER[worst_rung] if video_uri
                    else ("routed_text_only" if video_exists
                          else "no_video_file")),
        "fps": (last_reply["mm_kwargs"] or {}).get("fps"),
        "frame_tokens": ((last_reply["mm_kwargs"] or {}).get("size", {})
                         .get("longest_edge", 0) // PIXELS_PER_TOKEN) or None,
        "duration_sec": round(duration, 2) if duration else None,
        "prompt_tokens": last_reply["prompt_tokens"],
        "completion_tokens": completion_tokens,
        "grown": grown,
        "truncated": truncated,
        "finish_reason": "length" if truncated else "stop",
        "reasoning_chars": len(last_reply["reasoning"]),
        "content_parsed": parsed > 0,
        "n_parsed": parsed,
        "prep_sec": round(prep_seconds, 3),
        "elapsed": round(total_elapsed, 2),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def group_work(todo: list[dict], plans: dict[int, dict],
               durations: dict[str, float] | None = None) -> list[list[dict]]:
    """Queries sharing a video and a sampling plan, run back to back.

    vLLM caches multimodal embeddings and the prompt prefix, so the up-to-five
    queries on one video should pay the video prefill once -- which only
    happens if they are adjacent and identical up to the question.

    **Groups are then interleaved long-video, short-video, long, short.** The
    previous order was alphabetical by ``Video_ID``, which looks neutral and is
    not: the long TRECVID clips carry name-style IDs (``Liberty_Kid-29.webm``)
    while the short Shorts carry YouTube IDs, so sorting by name collects the
    long clips into runs. Measured over the first 1,796 queries of the 15 Aug
    run: clips >= 80 s are only 10.4% of the work, yet formed runs of up to
    **14 consecutive**, including 18 runs of 4 or more -- long enough for all
    four workers to hold a long video at once. In those windows the KV cache
    hit 39-42%, concurrency fell to 1-2 and vLLM's generation throughput
    collapsed to 0.2 tokens/s. The same clips shuffled would have produced a
    longest run of 3 and no run of 4+.

    Interleaving is deterministic, costs nothing, and cannot make the prefix
    cache worse: locality that matters is *within* a group (same video, same
    plan), and adjacent groups are different videos either way.
    """
    groups: dict[tuple, list[dict]] = {}

    for query in todo:
        plan = plans[query["Q_ID"]]
        key = (query["Video_ID"], plan["dense"], plan["ocr"],
               plan["video"], plan["transcript"], plan["budget"])
        groups.setdefault(key, []).append(query)

    ordered = [groups[key] for key in sorted(groups, key=lambda k: (str(k[0]), k))]

    if not durations:
        return ordered

    def length_of(group: list[dict]) -> float:
        return durations.get(Path(group[0]["Video_ID"]).stem) or 0.0

    ordered.sort(key=length_of)
    half = len(ordered) // 2
    short, long_ = ordered[:half], ordered[half:][::-1]
    woven: list[list[dict]] = []

    for index in range(max(len(short), len(long_))):
        if index < len(long_):
            woven.append(long_[index])
        if index < len(short):
            woven.append(short[index])

    return woven


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight(client: Qwen3VLClient, args: argparse.Namespace,
              queries: list[dict], durations: dict[str, float],
              prep: VideoPrep) -> bool:
    """Settle the uncertain request options against the live server.

    Two things here cannot be known from the client side: whether this vLLM
    build accepts the ``mm_processor_kwargs`` shape below, and whether its
    structured-output path exempts the reasoning block (constraining the whole
    generation to JSON would forbid the model from thinking at all, and the
    -Thinking checkpoint has no other mode). Both are cheap to test once and
    expensive to discover 400 queries into an overnight run.
    """
    print("preflight:")

    try:
        served = client.resolve_model()
    except Exception as exc:  # noqa: BLE001 - the server is the whole run
        print(f"  cannot reach {args.base_url}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return False

    print(f"  server model    {served}")
    print(f"  max_model_len   {client.max_model_len}")

    if os.name == "nt":
        print("  WARNING: running on Windows. The server reads the video off "
              "the path in the request, so a Windows path only works if the "
              "server is on this host too. Run this from WSL2 -- see "
              "VQA/QWEN3_VL_SETUP.md.", file=sys.stderr)

    candidates = [(durations.get(Path(q["Video_ID"]).stem), q)
                  for q in queries]
    candidates = [(d, q) for d, q in candidates
                  if d and resolve_video(args.videos, q["Video_ID"]).exists()]

    if not candidates:
        print("  no video on disk to probe with -- skipping the media checks",
              file=sys.stderr)
        return True

    # Probe the shortest *and* the median clip, not just one. A single sample
    # is how a run once concluded that this build rejects mm_processor_kwargs
    # outright: the shortest video in the corpus is 1.07 s, short enough that
    # do_sample_frames fails on it alone, and one verdict then disabled the
    # frame budget for all 2,443 queries. Whatever is wrong has to be wrong on
    # both before it is treated as a property of the server.
    candidates.sort(key=lambda pair: pair[0])
    samples = [candidates[len(candidates) // 2], candidates[0]]

    plan = plan_for(None, True, "brief")
    probes: list[tuple[float, str]] = []
    media_dir = args.videos

    for duration, sample in samples:
        path = resolve_video(args.videos, sample["Video_ID"])
        safe_path, _ = prep.safe_path(path,
                                      adaptive_fps(duration, plan["dense"]))
        probes.append((duration, safe_path.resolve().as_uri()))
        media_dir = safe_path.resolve().parent

    last_detail: list[str] = []

    def probe(label: str, **overrides) -> str:
        """"ok", "context" (a budgeting miss the run-time ladder handles), or
        "rejected" (the server will not take this request at all).

        A verdict needs every probe video to agree; one clip failing is a
        property of that clip, which the per-request ladder handles.
        """
        was = (client.send_mm_kwargs, client.json_schema)
        client.send_mm_kwargs = overrides.get("send_mm_kwargs",
                                              client.send_mm_kwargs)
        client.json_schema = overrides.get("json_schema", client.json_schema)
        verdicts: list[str] = []
        started = time.monotonic()

        try:
            for probe_duration, probe_uri in probes:
                for shrink in (0, 2):
                    try:
                        client.complete(
                            "Reply with this JSON and nothing else: "
                            '{"answers": ["a", "b"]}',
                            probe_uri,
                            video_mm_kwargs(probe_duration, plan,
                                            client.max_model_len, 512, shrink),
                            512,
                            2,
                        )
                        verdicts.append("ok")
                        break
                    except Exception as exc:  # noqa: BLE001 - under test
                        detail = str(exc).splitlines()[0][:140]

                        if CONTEXT_ERROR_RE.search(detail) and not shrink:
                            continue  # retry smaller before judging the server

                        verdicts.append(
                            "context" if CONTEXT_ERROR_RE.search(detail)
                            else "rejected")
                        last_detail.append(detail)
                        break
        finally:
            client.send_mm_kwargs, client.json_schema = was

        took = time.monotonic() - started
        ok = verdicts.count("ok")

        if ok:
            note = "" if ok == len(verdicts) else \
                f" ({ok}/{len(verdicts)} probe videos; the rest are handled " \
                f"per request)"
            print(f"  {label:<15} ok ({took:.1f}s){note}")
            return "ok"

        print(f"  {label:<15} FAILED on all {len(verdicts)} probe videos: "
              f"{last_detail[-1] if last_detail else '?'}")

        return "context" if "context" in verdicts else "rejected"

    verdict = probe("video+kwargs", json_schema=False)

    if verdict == "rejected":
        # Distinguish "this build dislikes the kwargs" from "this server will
        # not read videos at all" -- the first is recoverable, the second
        # means every query would answer blind and the run is worthless.
        if probe("video only", send_mm_kwargs=False,
                 json_schema=False) == "ok":
            client.send_mm_kwargs = False
            print("  -> this build rejects the mm_processor_kwargs shape; "
                  "running with the server's own video defaults, which means "
                  "the frame budget is no longer sized to the window")
        else:
            print("  -> the server will not accept a video from this path. "
                  "Check that --allowed-local-media-path covers "
                  f"{media_dir} and that the server can read it.",
                  file=sys.stderr)
            return False
    elif verdict == "context":
        # Not fatal: every request degrades on its own, and this probe is the
        # shortest clip in the set, so the budget is wrong everywhere.
        print("  -> even a shrunk request overflows the window. The run will "
              "work but every query will pay a wasted 400 first; raise "
              "--max-model-len on the server or lower --max-tokens.",
              file=sys.stderr)

    if client.json_schema and probe("json schema", json_schema=True) != "ok":
        client.json_schema = False
        print("  -> structured output rejected; falling back to the "
              "tolerant JSON extraction, which is the default anyway")

    return True


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def assemble(queries: list[dict], cache: dict[int, dict],
             want: int) -> tuple[list[dict], int]:
    """Emit the run file, carrying only the fields the validator allows."""
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
        description="Answer the Task 1 questions with Qwen3-VL-8B-Instruct, "
                    "ten routed prompts per query, pooled."
    )

    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    ap.add_argument("--variations", type=Path, default=DEFAULT_VARIATIONS,
                    help="paraphrases from generate.query.variations.py; the "
                         "query is asked once per phrasing and the candidates "
                         "are pooled (default: %(default)s)")
    ap.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    ap.add_argument("--frame-index", type=Path, default=DEFAULT_FRAME_INDEX,
                    help="read only for its probed durations (default: "
                         "%(default)s)")
    ap.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)

    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--max-model-len", type=int,
                    help="override the window the server reports")
    ap.add_argument("--max-tokens", type=int,
                    help="fixed generation budget, overriding the per-query "
                         "budget the routing verdict picks")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--presence-penalty", type=float,
                    default=DEFAULT_PRESENCE_PENALTY)
    ap.add_argument("--timeout", type=float, default=1200.0,
                    help="per-call seconds; a deep reasoning trace on a long "
                         "clip is slow (default: %(default)s)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--answers-per-query", type=int, default=ANSWERS_PER_QUERY)
    ap.add_argument("--prompts", type=int, default=PROMPTS_PER_QUERY,
                    help="prompt phrasings per query: the original question "
                         "plus N-1 variations (default: %(default)s)")
    ap.add_argument("--candidates", type=int, default=CANDIDATES_PER_PROMPT,
                    help="candidate answers per prompt (default: %(default)s)")

    ap.add_argument("--budget", choices=BUDGETS,
                    help="fixed reasoning budget for every query instead of "
                         "deriving it from the routing verdict")
    ap.add_argument("--ignore-routing", action="store_true",
                    help="send video and transcript to every query at a fixed "
                         "budget -- the ablation the routed run has to beat")
    ap.add_argument("--json-schema", action="store_true",
                    help="constrain the reply with a JSON schema; only works "
                         "if the server's reasoning parser exempts the think "
                         "block, which preflight checks")
    ap.add_argument("--no-transcode", action="store_true",
                    help="never pre-shrink a video, even one whose full "
                         "decode is predicted to exhaust host RAM")
    ap.add_argument("--decode-limit-gib", type=float, default=4.0,
                    help="raw decode footprint above which a video is "
                         "pre-shrunk (default: %(default)s)")
    ap.add_argument("--skip-preflight", action="store_true")

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
    routing = {} if args.ignore_routing else load_routing(args.routing)
    variations = {entry["Q_ID"]: entry.get("Variations") or []
                  for entry in load_json(args.variations)} \
        if args.variations.exists() else {}

    if not variations and not args.rebuild:
        print(f"  WARNING: no variations at {args.variations} -- every query "
              f"will be asked once instead of {args.prompts} times, which is "
              f"the arrangement that lost 1,596 slots on 13 Aug",
              file=sys.stderr)

    if not routing and not args.ignore_routing:
        print(f"no routing at {args.routing} -- every query gets every input "
              f"at the {DEFAULT_BUDGET} budget", file=sys.stderr)

    print(f"queries: {len(queries)} | cached: "
          f"{sum(1 for q in queries if q['Q_ID'] in cache)} | "
          f"routed: {len(routing)} | durations known: {len(durations)}")

    if not args.rebuild:
        todo = [q for q in queries if args.force or q["Q_ID"] not in cache]

        if args.only:
            wanted = set(args.only)
            todo = [q for q in todo if q["Q_ID"] in wanted]

        if args.limit:
            todo = todo[:args.limit]

        plans = {
            q["Q_ID"]: plan_for(routing.get(q["Q_ID"]), args.ignore_routing,
                                args.budget)
            for q in todo
        }

        # --max-tokens fixes the generation budget without flattening the
        # routing labels, so the per-budget breakdown in report() still says
        # which queries the router thought were hard.
        if args.max_tokens:
            for plan in plans.values():
                plan["max_tokens"] = args.max_tokens

        if todo:
            ordered = group_work(todo, plans, durations)
            counts = Counter(p["budget"] for p in plans.values())
            no_media = sum(
                1 for q in todo
                if not resolve_video(args.videos, q["Video_ID"]).exists()
            )

            print(f"to answer: {len(todo)} queries over {len(ordered)} groups "
                  f"| budgets " + " ".join(f"{b} {counts.get(b, 0)}"
                                           for b in BUDGETS))
            print(f"  video attached {sum(1 for p in plans.values() if p['video'])}"
                  f" | dense {sum(1 for p in plans.values() if p['dense'])}"
                  f" | transcript {sum(1 for p in plans.values() if p['transcript'])}"
                  f" | ocr {sum(1 for p in plans.values() if p['ocr'])}"
                  f" | no video file {no_media}")

            client = Qwen3VLClient(
                args.base_url, args.model, args.temperature, args.top_p,
                args.top_k, args.presence_penalty, args.timeout, args.retries,
                args.max_model_len, json_schema=args.json_schema,
            )
            prep = VideoPrep(args.videos,
                             int(args.decode_limit_gib * 1024**3),
                             not args.no_transcode)

            if not args.skip_preflight and not preflight(client, args, todo,
                                                         durations, prep):
                return 1

            lock = threading.Lock()
            done = failed = 0
            started = time.monotonic()

            args.cache.parent.mkdir(parents=True, exist_ok=True)
            handle = args.cache.open("a", encoding="utf-8")

            def work(group: list[dict]) -> None:
                nonlocal done, failed

                for query in group:
                    prompts = [query["Question"]]
                    prompts += variations.get(query["Q_ID"], [])[
                        :max(args.prompts - 1, 0)]

                    try:
                        record = answer_query(
                            client, query, plans[query["Q_ID"]], args.videos,
                            args.transcripts, durations, prep,
                            args.answers_per_query, prompts, args.candidates,
                        )
                    except Exception as exc:  # noqa: BLE001 - one bad query
                        with lock:            # must not sink a 20-hour run
                            failed += 1
                            print(f"  FAILED Q_ID {query['Q_ID']}: "
                                  f"{type(exc).__name__}: {exc}",
                                  file=sys.stderr)
                        continue

                    with lock:
                        handle.write(
                            json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        cache[record["Q_ID"]] = record

                        done += 1
                        rate = done / max(time.monotonic() - started, 1e-9)
                        remaining = (len(todo) - done) / max(rate, 1e-9) / 3600
                        flag = ""

                        if record["fallback"]:
                            flag += f" FALLBACK({record['fallback']})"
                        if not record["content_parsed"]:
                            flag += " UNPARSED"
                        if record["degrade"] != "full":
                            flag += f" {record['degrade'].upper()}"
                        if record["finish_reason"] == "length":
                            flag += " TRUNCATED"

                        print(f"  [{done}/{len(todo)}] {rate * 60:5.1f}/min "
                              f"eta {remaining:4.1f}h  "
                              f"{record['elapsed']:6.1f}s "
                              f"{record['budget'][:5]:<5} "
                              f"{record['completion_tokens'] or 0:5d}t"
                              f"{flag}  Q{record['Q_ID']}: "
                              f"{record['answers'][0]['Answer'][:44]}")

            try:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    list(pool.map(work, ordered))
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

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(run, handle, indent=4, ensure_ascii=False)
        handle.write("\n")

    print(f"\nwrote {len(run)} of {len(queries)} query entries to {args.out}")

    if missing:
        print(f"  {missing} queries are not answered yet -- the run will not "
              f"validate until they are")

    report(cache, queries)

    return 0


def report(cache: dict[int, dict], queries: list[dict]) -> None:
    """Sanity checks the validator does not make."""
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
    print(f"  queries needing fallback padding:   "
          f"{sum(1 for r in records if r['fallback'])}")
    print(f"  queries whose JSON did not parse:   "
          f"{sum(1 for r in records if not r['content_parsed'])}")
    print(f"  generations cut off by max_tokens:  "
          f"{sum(1 for r in records if r.get('finish_reason') == 'length')}"
          f"  (after {sum(r.get('grown') or 0 for r in records)} budget "
          f"doublings)")

    pooled = [r.get("pooled_unique") or 0 for r in records]
    failures = sum(r.get("n_prompt_failures") or 0 for r in records)
    scored = sum(1 for r in records for c in (r.get("candidates") or ())
                 if c.get("logprob_mean") is not None)
    total_cands = sum(len(r.get("candidates") or ()) for r in records)

    if any(pooled):
        print(f"  distinct candidates pooled:         "
              f"mean {sum(pooled) / len(pooled):4.1f}  min {min(pooled)}  "
              f"max {max(pooled)}")
        print(f"  prompt calls that failed outright:  {failures}")
        print(f"  candidates carrying a logprob:      "
              f"{scored}/{total_cands}"
              f"{'  -- CHECK: logprobs are off' if not scored else ''}")
    print(f"  rank-1 answer is \"unknown\":         {unknown}")
    print(f"  answered with the video attached:   "
          f"{sum(1 for r in records if r.get('video_attached'))}")
    print(f"  video on disk but not attached:     "
          f"{sum(1 for r in records if r.get('video_exists') and not r.get('video_attached'))}")
    print(f"  no video file on disk:              "
          f"{sum(1 for r in records if not r.get('video_exists'))}")
    print(f"  transcript attached:                "
          f"{sum(1 for r in records if r.get('has_transcript'))}")
    print(f"  transcript on disk, routed away:    "
          f"{sum(1 for r in records if r.get('transcript_on_disk') and not r.get('has_transcript'))}")
    print(f"  routed to audio, video restored:    "
          f"{sum(1 for r in records if r.get('rescued'))}")

    degrades = Counter(r.get("degrade") for r in records)
    degraded = [(k, v) for k, v in degrades.most_common()
                if k not in ("full", "no_video_file", "routed_text_only")]

    if degraded:
        print("  requests that had to degrade:       "
              + ", ".join(f"{k} {v}" for k, v in degraded))

    print("\n  per reasoning budget")

    for budget in BUDGETS:
        subset = [r for r in records if r.get("budget") == budget]

        if not subset:
            continue

        times = [r["elapsed"] for r in subset]
        tokens = [r.get("completion_tokens") or 0 for r in subset]
        print(f"    {budget:<9} n {len(subset):5d}  "
              f"mean {sum(times) / len(times):6.1f}s  "
              f"max {max(times):6.1f}s  "
              f"mean out {sum(tokens) / len(tokens):6.0f} tokens  "
              f"total {sum(times) / 3600:5.2f}h")

    times = [answer["Time"] for r in records for answer in r["answers"]]
    per_query = [sum(a["Time"] for a in r["answers"]) for r in records]
    measured = sum(r["elapsed"] for r in records)

    print(f"\n  Time per answer: mean {sum(times) / len(times):5.2f}s  "
          f"min {min(times):.2f}s  max {max(times):.2f}s  "
          f"zero-valued {sum(1 for t in times if t == 0)}")
    print(f"  Time per query:  mean {sum(per_query) / len(per_query):5.2f}s  "
          f"total {sum(per_query) / 3600:.2f} h vs "
          f"{measured / 3600:.2f} h actually spent on inference")


if __name__ == "__main__":
    sys.exit(main())

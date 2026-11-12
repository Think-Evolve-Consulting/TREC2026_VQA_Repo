#!/usr/bin/env python3
"""TREC 2026 VQA Task 2 (multiple choice) — Qwen3.8-27B ranker, standalone for Colab.

A self-contained port of ``VQA/rank.options.mc.qwen38.py``. Nothing is imported
from the project; copy this one file to Colab and run it.

**This is the best ranker measured in this project.** Dev-set result on the 400
labelled 2025 queries, official scorer: **0.800 Top-1 / 0.884 MRR**. Nothing
beat it — not Qwen3-VL (0.790, p = 0.70), not any of five other models, and not
any of 781,248 searched fusions once nested cross-validation removed the search
optimism. If you run only one of these six scripts, run this one.

It is the Qwen3-VL pass with the checkpoint swapped: the same ranking contract
— a permutation of option *numbers* mapped back to the official strings — and
the same video handling, which transfers because the two share a vision
geometry (patch 16, temporal patch 2, spatial merge 2, so 1,024 pixels per
visual token). At the same 4,096-token budget it ran **2.5x faster**: 7.53 h
against 18.88 h for the full 2,443 queries.

What is specific to this checkpoint
-----------------------------------
**It is a tight fit.** Qwen3.8-27B is a 27B multimodal model; bf16 is ~54 GB,
so this serves RedHatAI's INT4 (compressed-tensors W4A16) build at 17.71 GiB of
weights. That only works because the language model is hybrid — 48 of its 64
layers are linear attention and 16 are full attention — so the KV cache is a
fraction of what a dense 27B would need (3.3 GiB bought 46,421 tokens locally).
``--skip-mm-profiling`` is mandatory for the same reason, and ``--visual-tokens``
is the first knob to turn if the engine will not start or a request will not fit.

**vLLM 0.27+ is required, not the 0.12 the Qwen3-VL pass used.** Only the newer
engine registers ``Qwen3_5ForConditionalGeneration``. ``--install`` pins it.

**No kwarg controls its temporal sampling, and that is measured, not assumed.**
``do_sample_frames`` was rejected on 1,765 of 2,443 queries (72%); ``num_frames``
is silently ignored without it (identical token counts at 16 and 64) and
rejected with it. A full re-run under a transcode workaround changed rank-1 on
**6.3% of targeted queries against 8.0% of untouched controls** — the fix made
no measurable difference, so it is deliberately **not** implemented here. The
``no_dsf`` rung of the degrade ladder absorbs the rejection instead.

**It reasons in plain ``content``, which the JSON schema suppresses.** All 7,329
replies parsed locally, 0 repaired. That is the right trade where the answer is
a permutation rather than an argument.

Its one measured weakness: **counting.** 43.8% error against 16.8% on everything
else, and the misses are systematic undercounts (15 of 18 readable cases too
low, mean -1.2). Every model in the set undercounts; this one is the second best
at it. A separate theme in the 18 queries only it gets wrong is "what can be
heard" questions answered from silent video.

Colab quickstart
----------------
Runtime → **A100 (40 GB)** is comfortable. An **L4 (22.5 GB)** fits but is
tight: drop ``--gpu-fraction`` to 0.90 and ``--visual-tokens`` to 2048 if the
engine will not start. A T4 cannot run this.

    from google.colab import drive; drive.mount('/content/drive')

    !python task2_qwen38_colab.py --install \
        --drive-root /content/drive/MyDrive/TREC_2026 --shard 1/2

Expected Drive layout
---------------------
    TREC_2026/
      vqa.2026.testingDataset.Task2_v2.json
      videos_TEST_VQA/
        videos/<video_id>.{mp4,webm}            <- REQUIRED here (11.4 GiB)
        frames/index.json                       <- read for durations only
        transcription_openwhisper/<video_id>.json
      query.routing.task1.json                  <- optional, see --routing
      runs/                                     <- created; outputs land here

Resuming, which you will need
-----------------------------
7.53 h locally on a 3090 means two Colab sessions at least. Every completed
query is appended to a JSONL cache immediately, mirrored to Drive every
``--sync-every`` records, and re-read on startup, so **re-running the identical
command after a disconnect picks up where it stopped.** For a deliberate split
use ``--shard 1/2`` and ``--shard 2/2`` against the same Drive root, then
``--assemble-only`` once to emit the single complete run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

RUN_NAME = "qwen38"

# ---------------------------------------------------------------------------
# Model and serving defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "RedHatAI/Qwen3.8-27B-INT4"
DEFAULT_PORT = 8011
DEFAULT_MAX_MODEL_LEN = 24576
DEFAULT_GPU_FRACTION = 0.92

DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20

DEFAULT_PASSES = 3
DEFAULT_WORKERS = 4
DEFAULT_TIMEOUT = 900.0
DEFAULT_RETRIES = 3

# Frame budgeting. A fixed frame count per video regardless of its length.
TARGET_FRAMES = 64
DENSE_TARGET_FRAMES = 96
MAX_FPS = 4.0
MIN_FPS = 0.2

# Qwen3-VL geometry: patch 16, temporal patch 2, spatial merge 2 -> 1,024
# pixels per visual token.
PIXELS_PER_TOKEN = 1024
TEMPORAL_MERGE = 2

# Below this many frames the processor rejects do_sample_frames outright.
MIN_RESAMPLE_FRAMES = 16

# The whole-video visual budget. 4,096 was the user's choice locally: 8,192
# projected to ~23 h and 2,048 to ~8.7 h on the full set.
VISUAL_TARGET_TOKENS = 4096
OCR_TARGET_TOKENS = 12288
MIN_FRAME_TOKENS = 128
TEXT_TOKEN_RESERVE = 2048

DIGIT_LIST_RE = re.compile(r"\d+(?:\s*[,\s]\s*\d+){1,}")

VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".mov", ".m4v", ".avi")

# ---------------------------------------------------------------------------
# The prompt, word for word from VQA/rank.options.mc.qwen38.py
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a helpful assistant."

PROMPT_TEMPLATE = """{context}Question: {question}

Answer options:
{options}

Rank all {n} options from the most likely correct answer to the least likely.

Rules:
- Judge each option against the video and the transcript, not against how it
  is worded.
- Reply with option numbers only, best first.
- Every number from 1 to {n} must appear exactly once. Never repeat a number
  and never leave one out.
- Exactly one option is correct. If the evidence is weak, still commit to an
  order -- put what the video supports most first.{extra}

Reply with JSON: {{"ranking": [{slots}], "confidence": 0.0}}
where ranking holds all {n} option numbers best first, and confidence is how
confident you are in the first one, 0.0 to 1.0."""

OCR_RULE = ("\n- The answer depends on text visible on screen. Read any signs, "
            "captions or labels exactly as written before choosing.")
TEMPORAL_RULE = ("\n- The answer depends on ordering or timing. Use the frame "
                 "order and the timestamps, not just what appears.")


# ---------------------------------------------------------------------------
# Ranking contract (shared core, verified against the shipped run)
# ---------------------------------------------------------------------------

def ranking_schema(n: int) -> dict:
    """Constrain the reply to exactly `n` integers in range, plus a confidence.

    Uniqueness is deliberately not expressed: no grammar backend enforces
    ``uniqueItems``, so a schema claiming it would buy nothing and hide the
    repair path that actually does the work (as_permutation).
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
            "confidence": {"type": "number"},
        },
        "required": ["ranking", "confidence"],
    }


def seed_for(q_id: int, index: int) -> int:
    """Stable per-call seed, so a re-run reproduces the same run file."""
    digest = hashlib.sha256(f"mc|{q_id}|{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def mmss(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def presentation_orders(n: int, passes: int, q_id: int) -> list[list[int]]:
    """The option orders the passes present, as index lists into `options`.

    Pass 0 is the official order and pass 1 its reverse -- between them every
    option that was first is last, the cheapest possible probe of position
    bias. Later passes are shuffles seeded from the Q_ID.
    """
    orders = [list(range(n))]

    if passes > 1:
        orders.append(list(reversed(range(n))))

    rng = random.Random(seed_for(q_id, 500))

    while len(orders) < passes:
        order = list(range(n))
        rng.shuffle(order)

        if order not in orders or len(orders) > 2 * n:
            orders.append(order)

    return orders[:passes]


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
    elif plan["frames"] or plan["rescued"]:
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


def render_options(options: list[str], order: list[int]) -> str:
    """The numbered option block, in this pass's presentation order."""
    return "\n".join(f"{position}. {options[index]}"
                     for position, index in enumerate(order, 1))


def parse_json_object(text: str) -> dict:
    """Parse a reply that should be JSON, tolerating prose wrapped around it."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")

    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")

    return json.loads(text[start:end + 1])


def as_permutation(raw: object, n: int) -> tuple[list[int], bool]:
    """Coerce a reply's ranking into a permutation of 0..n-1 (presented order).

    The schema fixes the length and the range but cannot say "each exactly
    once", so this is where a repeated or missing option is caught: repeats
    keep their first, best occurrence, and whatever the model dropped is
    appended in the order it was shown.
    """
    values = raw if isinstance(raw, list) else []
    seen: set[int] = set()
    ordered: list[int] = []
    repaired = False

    for value in values:
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

    Each pass awards ``n - position`` points scaled by ``0.5 + 0.5 *
    confidence``, so a hesitant pass still votes but a confident one counts for
    twice as much. Ties break on mean rank, then on the official option order,
    so the output is deterministic. Every per-option signal is kept so a later
    fusion can re-rank without re-running inference.
    """
    points = [0.0] * n
    ranks: list[list[int]] = [[] for _ in range(n)]
    firsts = [0] * n

    for entry in passes:
        confidence = entry.get("confidence")
        confidence = confidence if isinstance(confidence, (int, float)) else 0.5
        confidence = min(max(float(confidence), 0.0), 1.0)
        weight = 0.5 + 0.5 * confidence

        for position, option_index in enumerate(entry["ranking"]):
            points[option_index] += weight * (n - position)
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
            f"options ({len(answers)} answers, "
            f"{len(set(answers))} distinct, {len(options)} options)")


def extract_ranking(content: str, n: int) -> tuple[list, str]:
    """The ranking out of a reply, with the salvage path the local pass needed.

    A schema-constrained reply parses as JSON. When the grammar is off, or the
    model wraps prose around it, a bare digit list is still a ranking -- 35
    replies were recovered this way in the Cosmos run.
    """
    try:
        return parse_json_object(content).get("ranking"), "json"
    except (ValueError, json.JSONDecodeError):
        pass

    match = DIGIT_LIST_RE.search(content or "")

    if match:
        digits = [int(part) for part in re.findall(r"\d+", match.group(0))]

        if len(digits) >= n:
            return digits[:n], "digits"

    raise ValueError(f"no ranking in reply: {(content or '')[:200]!r}")


# ---------------------------------------------------------------------------
# Frame budgeting
# ---------------------------------------------------------------------------

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
    """fps and pixel budget for one request.

    ``size.longest_edge`` is a whole-video pixel budget, so it is set from the
    target directly; fps decides only how those tokens are spread over time.
    ``shrink`` halves the budget, once per rung of the degrade ladder.
    """
    fps = adaptive_fps(duration_sec, plan.get("dense", False)) / (2 ** shrink)
    fps = max(MIN_FPS, round(fps, 3))

    frames = expected_frames(duration_sec, fps)

    # The budget is for the whole video, so the frame count does not divide it.
    # What must fit the window is the target itself.
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


# ---------------------------------------------------------------------------
# Data loading (shared core)
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_queries(path: Path) -> list[dict]:
    """The official query file, with Options flattened to a list of strings."""
    raw = load_json(path)

    if not isinstance(raw, list):
        raise SystemExit(f"{path}: expected a top-level list of queries")

    queries = []

    for item in raw:
        options = [entry["option"] for entry in item["Options"]]

        if len(set(options)) != len(options):
            raise SystemExit(f"Q_ID {item['Q_ID']}: duplicate option text")

        queries.append({"Q_ID": int(item["Q_ID"]),
                        "Video_ID": item["Video_ID"],
                        "Question": item["Question"],
                        "Options": options})

    return queries


def load_routing(path: Path | None) -> dict[int, dict]:
    """Q_ID -> routing verdict from route.query.inputs.py, or {} when absent."""
    if not path or not Path(path).exists():
        return {}

    return {int(entry["Q_ID"]): entry for entry in load_json(path)}


def plan_for(routing: dict | None, question: str, ignore_routing: bool) -> dict:
    """Which inputs this query is sent, and which extra rules it gets.

    A verdict about a different question is not a verdict about this one, so a
    question-text mismatch quietly falls back to attaching every input.
    """
    everything = {"frames": True, "transcript": True, "ocr": False,
                  "temporal": False, "routed": False, "rescued": False}

    if ignore_routing or not routing:
        return dict(everything)

    if (routing.get("Question") or "").strip() != question.strip():
        return dict(everything)

    pipeline = set(routing.get("Pipeline") or ())

    plan = {
        "frames": "frames" in pipeline,
        "transcript": "asr_transcript" in pipeline,
        "ocr": "frame_ocr" in pipeline,
        "temporal": bool(routing.get("Temporal")),
        "routed": True,
        "rescued": False,
    }

    # An OCR question is about pixels whatever else the router decided.
    if plan["ocr"]:
        plan["frames"] = True

    return plan


def load_transcript(directory: Path, video_id: str) -> dict | None:
    """The suspect-filtered Whisper transcript, or None when there is none."""
    path = Path(directory) / (Path(video_id).stem + ".json")

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

    return {"text": text, "language": language}


def load_cache(path: Path) -> dict[int, dict]:
    """Read the JSONL work cache, keyed by Q_ID. Later records win."""
    cache: dict[int, dict] = {}

    if not Path(path).exists():
        return cache

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(record, dict) and "Q_ID" in record:
                cache[int(record["Q_ID"])] = record

    return cache


def assemble(queries: list[dict], cache: dict[int, dict]) -> tuple[list[dict], int]:
    """The run file, in official query order, plus a count of missing queries.

    Anything the cache has not reached keeps the official option order, which
    is exactly the random control for that query -- counted, never hidden.
    """
    run: list[dict] = []
    missing = 0

    for query in queries:
        record = cache.get(query["Q_ID"])

        if record and record.get("answers"):
            answers = record["answers"]
        else:
            missing += 1
            answers = [{"Rank": rank, "Answer": option}
                       for rank, option in enumerate(query["Options"], 1)]

        run.append({"Q_ID": query["Q_ID"],
                    "Video_ID": query["Video_ID"],
                    "Answers": answers})

    return run, missing


def load_durations(index_path: Path) -> dict[str, float]:
    """video stem -> duration, from the frame index when it is available."""
    if not Path(index_path).exists():
        return {}

    videos = load_json(index_path).get("videos") or {}

    return {stem: float(entry["duration_sec"]) for stem, entry in videos.items()
            if entry.get("duration_sec")}


def probe_duration(path: Path) -> float | None:
    """ffprobe fallback for a clip the frame index does not cover."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60)

        return float(out.stdout.strip())
    except Exception:  # noqa: BLE001 - a missing duration is not fatal
        return None


def resolve_video(videos_dir: Path, video_id: str) -> Path | None:
    """The clip for a Video_ID.

    Video_ID is sometimes ``8384.webm`` and sometimes a bare YouTube id, so the
    exact name is tried first and the stem second. 1,940 of the 1,956 distinct
    2026 ids resolve; the remaining 16 have no file at all.
    """
    videos_dir = Path(videos_dir)
    exact = videos_dir / video_id

    if exact.is_file():
        return exact

    stem = Path(video_id).stem

    for suffix in VIDEO_SUFFIXES:
        candidate = videos_dir / (stem + suffix)

        if candidate.is_file():
            return candidate

    matches = sorted(p for p in videos_dir.glob(stem + ".*")
                     if p.suffix.lower() in VIDEO_SUFFIXES)

    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Run validation (shared core)
# ---------------------------------------------------------------------------

def validate_run(run: list[dict], queries: list[dict]) -> list[str]:
    """Every error the official validator would raise, as a list of strings."""
    errors: list[str] = []
    official = {q["Q_ID"]: q for q in queries}
    seen: set[int] = set()

    for index, item in enumerate(run):
        loc = f"run[{index}]"

        if set(item) != {"Q_ID", "Video_ID", "Answers"}:
            errors.append(f"{loc}: keys must be exactly Q_ID, Video_ID, "
                          f"Answers; found {sorted(item)}")
            continue

        q_id = item["Q_ID"]

        if not isinstance(q_id, int) or isinstance(q_id, bool):
            errors.append(f"{loc}.Q_ID must be an integer")
            continue

        if q_id in seen:
            errors.append(f"{loc}.Q_ID {q_id} is a duplicate")

        seen.add(q_id)
        ref = official.get(q_id)

        if ref is None:
            errors.append(f"{loc}.Q_ID {q_id} is not an official Q_ID")
            continue

        if item["Video_ID"] != ref["Video_ID"]:
            errors.append(f"{loc}.Video_ID {item['Video_ID']!r} != official "
                          f"{ref['Video_ID']!r}")

        answers = item["Answers"]
        n = len(ref["Options"])

        if not isinstance(answers, list) or len(answers) != n:
            errors.append(f"{loc}.Answers must hold exactly {n} entries")
            continue

        ranks, texts = [], []

        for position, entry in enumerate(answers):
            if set(entry) != {"Rank", "Answer"}:
                errors.append(f"{loc}.Answers[{position}]: keys must be "
                              f"exactly Rank, Answer")
                continue

            ranks.append(entry["Rank"])
            texts.append(entry["Answer"])

        if ranks != list(range(1, n + 1)):
            errors.append(f"{loc}.Answers ranks must be {list(range(1, n + 1))} "
                          f"in ascending order; found {ranks}")

        if sorted(texts) != sorted(ref["Options"]):
            errors.append(f"{loc}.Answers must be a permutation of the "
                          f"official options")

    missing = [q["Q_ID"] for q in queries if q["Q_ID"] not in seen]

    if missing:
        errors.append(f"missing {len(missing)} official Q_IDs "
                      f"(first: {missing[:5]})")

    order = [item["Q_ID"] for item in run if isinstance(item.get("Q_ID"), int)]

    if not missing and order != [q["Q_ID"] for q in queries]:
        errors.append("Q_ID order differs from the official file "
                      "(a warning officially, an error here)")

    return errors


# ---------------------------------------------------------------------------
# Colab plumbing (shared core)
# ---------------------------------------------------------------------------

class Mirror:
    """A local-disk JSONL cache that is copied to Drive every N records.

    Appending straight to a Drive FUSE path is slow and, worse, loses buffered
    writes when the runtime is preempted. Writing locally and mirroring in
    batches keeps the hot path fast and still survives a disconnect with at
    most ``every`` queries lost.
    """

    def __init__(self, local: Path, remote: Path | None, every: int) -> None:
        self.local = Path(local)
        self.remote = Path(remote) if remote else None
        self.every = max(1, int(every))
        self.pending = 0
        self.lock = threading.Lock()

        self.local.parent.mkdir(parents=True, exist_ok=True)

        # Pull an earlier session's work down before appending to it.
        if self.remote and self.remote.exists():
            remote_size = self.remote.stat().st_size
            local_size = self.local.stat().st_size if self.local.exists() else -1

            if remote_size > local_size:
                self.remote.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.remote, self.local)
                print(f"resumed cache from Drive: {self.remote} "
                      f"({remote_size / 1024:.0f} KiB)")

        self.handle = self.local.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        with self.lock:
            self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.handle.flush()
            self.pending += 1
            due = self.pending >= self.every

        if due:
            self.sync()

    def sync(self) -> None:
        with self.lock:
            self.pending = 0

            if not self.remote:
                return

            try:
                self.remote.parent.mkdir(parents=True, exist_ok=True)
                # Write beside the target and replace, so a preemption during
                # the copy cannot leave a truncated cache on Drive.
                staged = self.remote.with_suffix(self.remote.suffix + ".tmp")
                shutil.copyfile(self.local, staged)
                os.replace(staged, self.remote)
            except OSError as exc:
                print(f"  cache mirror failed ({exc}) -- work is safe locally "
                      f"at {self.local}", file=sys.stderr)

    def close(self) -> None:
        try:
            self.handle.close()
        finally:
            self.sync()


def publish(run: list[dict], local: Path, remote: Path | None) -> None:
    """Write the run file locally and mirror it to Drive."""
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(run, ensure_ascii=False, indent=1),
                     encoding="utf-8")

    if remote:
        try:
            Path(remote).parent.mkdir(parents=True, exist_ok=True)
            staged = Path(str(remote) + ".tmp")
            shutil.copyfile(local, staged)
            os.replace(staged, remote)
        except OSError as exc:
            print(f"  run mirror failed ({exc}) -- the run is at {local}",
                  file=sys.stderr)


class VideoStager:
    """Copies clips from Drive to local disk on demand, once each.

    vLLM opens the ``file://`` URI itself, and every decode of a clip sitting
    on the Drive FUSE mount re-reads it over the network. Three passes per
    query and several queries per clip make that the dominant cost, so each
    clip is copied once and then served from local disk. The whole corpus is
    11.4 GiB, which fits Colab's scratch disk.
    """

    def __init__(self, local_root: Path, enabled: bool) -> None:
        self.root = Path(local_root) / "videos"
        self.enabled = enabled
        self.lock = threading.Lock()
        self.bytes = 0

        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def local(self, source: Path) -> Path:
        if not self.enabled:
            return source

        target = self.root / source.name

        with self.lock:
            if target.exists() and target.stat().st_size == source.stat().st_size:
                return target

        try:
            staged = target.with_suffix(target.suffix + ".part")
            shutil.copyfile(source, staged)
            os.replace(staged, target)

            with self.lock:
                self.bytes += target.stat().st_size

            return target
        except OSError as exc:
            print(f"  staging {source.name} failed ({exc}); reading from Drive",
                  file=sys.stderr)
            return source


class VLLMServer:
    """Starts vLLM in this VM, waits for it, and stops it on the way out."""

    def __init__(self, model: str, port: int, max_model_len: int,
                 gpu_fraction: float, media_root: Path, log: Path,
                 extra: list[str]) -> None:
        self.model = model
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.log = Path(log)
        self.process: subprocess.Popen | None = None

        self.command = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--served-model-name", model,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(gpu_fraction),
            # The server opens file:// URIs itself; without this it refuses.
            "--allowed-local-media-path", str(media_root),
            # Decode every frame and let size.longest_edge do the budgeting.
            "--media-io-kwargs", json.dumps({"video": {"num_frames": -1}}),
            "--disable-log-requests",
            # Mandatory on a 24 GB card: profiling the vision stack on
            # top of 17.71 GiB of INT4 weights will not fit.
            "--skip-mm-profiling",
        ] + list(extra)

    def start(self, wait: float) -> None:
        if self.alive():
            print(f"a server is already answering on {self.url} -- using it")
            return

        self.log.parent.mkdir(parents=True, exist_ok=True)
        handle = self.log.open("w", encoding="utf-8")

        print("starting vLLM:\n  " + " ".join(self.command))
        print(f"  log: {self.log}")

        self.process = subprocess.Popen(
            self.command, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True)

        started = time.monotonic()
        last = 0.0

        while time.monotonic() - started < wait:
            if self.process.poll() is not None:
                raise SystemExit(
                    f"vLLM exited with code {self.process.returncode} before "
                    f"serving. Last lines of {self.log}:\n" + self.tail())

            if self.alive():
                print(f"  server up after {time.monotonic() - started:.0f}s")
                return

            if time.monotonic() - last > 30:
                last = time.monotonic()
                print(f"  waiting for the model to load "
                      f"({time.monotonic() - started:.0f}s) ...")

            time.sleep(3.0)

        raise SystemExit(f"vLLM did not answer within {wait:.0f}s. Last lines "
                         f"of {self.log}:\n" + self.tail())

    def alive(self) -> bool:
        import requests

        try:
            return requests.get(self.url + "/v1/models",
                                timeout=5).status_code == 200
        except Exception:  # noqa: BLE001 - any failure means "not up yet"
            return False

    def tail(self, lines: int = 40) -> str:
        try:
            return "\n".join(self.log.read_text(
                encoding="utf-8", errors="replace").splitlines()[-lines:])
        except OSError:
            return "(no log)"

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            print("stopping vLLM ...")

            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=60)
            except Exception:  # noqa: BLE001 - best effort teardown
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# The client, with the degrade ladder
# ---------------------------------------------------------------------------

class Qwen38Client:
    """Sends the clip, degrading until something comes back.

    ``prompt_for(video_attached)`` and ``mm_kwargs_for(shrink)`` are callbacks
    because both change as the ladder descends -- a prompt that says "you are
    shown the video" must not survive dropping it.
    """

    LADDER = ("full", "shrink1", "shrink2", "no_dsf", "no_mm_kwargs",
              "no_video")

    name = "qwen38"

    def __init__(self, base: str, model: str, max_tokens: int,
                 temperature: float, top_p: float, top_k: int,
                 timeout: float, retries: int) -> None:
        self.url = base.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.timeout = timeout
        self.retries = retries
        self.local = threading.local()

    @property
    def session(self):
        import requests

        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()

        return self.local.session

    def rank(self, prompt_for, video_uri: str | None, mm_kwargs_for,
             n: int, seed: int) -> dict:
        """One ranking call. Returns the reply text and the rung it took."""
        import requests

        rung = 0 if video_uri else self.LADDER.index("no_video")
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

            if attach and video_uri:
                content.append({"type": "video_url",
                                "video_url": {"url": video_uri}})

            content.append({"type": "text", "text": prompt_for(attach)})

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "seed": seed,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ranking",
                        "schema": dict(ranking_schema(n),
                                       additionalProperties=False)},
                },
            }

            if mm_kwargs:
                payload["mm_processor_kwargs"] = mm_kwargs

            started = time.monotonic()

            try:
                response = self.session.post(self.url, json=payload,
                                             timeout=self.timeout)

                if response.status_code == 400:
                    # The server refused the request itself. Descend a rung
                    # rather than retrying something it will refuse again.
                    last = RuntimeError(f"400 at {step}: "
                                        f"{response.text[:200]}")
                    rung += 1
                    continue

                response.raise_for_status()
                choice = response.json()["choices"][0]

                return {
                    "content": (choice["message"].get("content") or "").strip(),
                    "elapsed": time.monotonic() - started,
                    "rung": step,
                    "video_used": bool(attach and video_uri),
                    "finish_reason": choice.get("finish_reason"),
                }
            except requests.RequestException as exc:
                last = exc

                if attempts % 2 == 0:
                    rung += 1

                time.sleep(2.0)
            except (ValueError, KeyError, IndexError) as exc:
                last = exc
                rung += 1

        raise RuntimeError(f"ladder exhausted: {type(last).__name__}: {last}")


def rank_query(client: Qwen38Client, query: dict, options: list[str],
               context_for, video_uri: str | None, mm_kwargs_for,
               passes: int, plan: dict, duration: float | None) -> dict:
    """Ask for every presentation order of one query and combine the rankings."""
    q_id = query["Q_ID"]
    n = len(options)
    slots = ", ".join(str(i) for i in range(1, n + 1))

    extra = ""

    if plan["ocr"]:
        extra += OCR_RULE
    if plan["temporal"]:
        extra += TEMPORAL_RULE

    results: list[dict] = []
    failures = 0
    repairs = 0
    rungs: list[str] = []

    for index, order in enumerate(presentation_orders(n, passes, q_id)):
        def prompt_for(attached: bool, order=order) -> str:
            return PROMPT_TEMPLATE.format(
                context=context_for(attached),
                question=query["Question"],
                options=render_options(options, order),
                n=n, slots=slots, extra=extra)

        try:
            reply = client.rank(prompt_for, video_uri, mm_kwargs_for, n,
                                seed_for(q_id, index))
        except RuntimeError as exc:
            failures += 1
            print(f"  Q_ID {q_id} pass {index}: {exc}", file=sys.stderr)
            continue

        try:
            raw, how = extract_ranking(reply["content"], n)
        except ValueError as exc:
            failures += 1
            print(f"  Q_ID {q_id} pass {index}: {exc}", file=sys.stderr)
            continue

        positions, repaired = as_permutation(raw, n)
        repairs += int(repaired)
        rungs.append(reply["rung"])

        confidence = None

        try:
            confidence = parse_json_object(reply["content"]).get("confidence")
        except (ValueError, json.JSONDecodeError):
            pass

        results.append({
            "index": index,
            "ranking": [order[position] for position in positions],
            "order": order,
            "confidence": confidence,
            "repaired": repaired,
            "parsed": how,
            "rung": reply["rung"],
            "video_used": reply["video_used"],
            "finish_reason": reply.get("finish_reason"),
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

    confident = [float(r["confidence"]) for r in results
                 if isinstance(r.get("confidence"), (int, float))]

    return {
        "Q_ID": q_id,
        "Video_ID": query["Video_ID"],
        "answers": answers,
        "ranked": ranked,
        "passes": results,
        "backend": "vllm",
        "model": client.model,
        "n_options": n,
        "n_passes": passes,
        "n_replies": len(results),
        "n_pass_failures": failures,
        "n_repaired": repairs,
        "fallback": fallback,
        "duration_sec": duration,
        # The audit that matters: how many passes actually saw pixels.
        "video_used": sum(1 for r in results if r["video_used"]),
        "rungs": rungs,
        "has_transcript": "Spoken audio" in context_for(True),
        "confidence_mean": round(sum(confident) / len(confident), 4)
        if confident else None,
        "agreement": round(max((item["firsts"] for item in ranked), default=0)
                           / max(len(results), 1), 4),
        "routed": plan["routed"],
        "rescued": plan["rescued"],
        "ocr": plan["ocr"],
        "temporal": plan["temporal"],
        "elapsed": round(sum(r["elapsed"] for r in results), 2),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(cache: dict[int, dict], queries: list[dict]) -> None:
    records = [cache[q["Q_ID"]] for q in queries if q["Q_ID"] in cache]

    if not records:
        print("nothing ranked yet")
        return

    n = len(records)
    unanimous = sum(1 for r in records if (r.get("agreement") or 0) >= 0.999)
    firsts_on_first = sum(
        1 for r in records
        if r.get("ranked") and r["ranked"][0]["option_index"] == 0)
    chance = sum(1.0 / r["n_options"] for r in records) / n
    blind = sum(1 for r in records if not r.get("video_used"))

    rungs: dict[str, int] = {}

    for record in records:
        for rung in record.get("rungs") or []:
            rungs[rung] = rungs.get(rung, 0) + 1

    print(f"\n{'-' * 68}\nranked          {n} / {len(queries)} queries")
    print(f"pass failures   {sum(r.get('n_pass_failures', 0) for r in records)}")
    print(f"repaired        {sum(r.get('n_repaired', 0) for r in records)}")
    print(f"fallbacks       {sum(1 for r in records if r.get('fallback'))}")
    print(f"NO VIDEO ON ANY PASS: {blind}  <- the silent failure mode; "
          f"expect ~116 with routing on, ~16 without")
    print(f"ladder rungs    " + ", ".join(f"{k}={v}" for k, v in
                                          sorted(rungs.items())))
    print(f"with transcript {sum(1 for r in records if r.get('has_transcript'))}")
    print(f"unanimous       {unanimous} ({100 * unanimous / n:.1f}%)")
    print(f"rank 1 on the officially-first option: "
          f"{100 * firsts_on_first / n:.1f}% (chance {100 * chance:.1f}%)")
    print(f"mean seconds per query: "
          f"{sum(r.get('elapsed', 0) for r in records) / n:.1f}")
    print("-" * 68)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)

    group = parser.add_argument_group("where things are")
    group.add_argument("--drive-root", type=Path,
                       default=Path("/content/drive/MyDrive/TREC_2026"))
    group.add_argument("--queries", type=Path, default=None)
    group.add_argument("--data-root", type=Path, default=None)
    group.add_argument("--videos", type=Path, default=None,
                       help="default: <data-root>/videos")
    group.add_argument("--frame-index", type=Path, default=None,
                       help="frames/index.json, read for durations only")
    group.add_argument("--transcripts", type=Path, default=None)
    group.add_argument("--routing", type=Path, default=None,
                       help="absent means every query gets video AND "
                            "transcript. The local 2026 run had routing ON")
    group.add_argument("--work", type=Path, default=Path("/content/trec_work"))
    group.add_argument("--out", type=Path, default=None)

    group = parser.add_argument_group("colab")
    group.add_argument("--install", action="store_true")
    group.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    group.add_argument("--stage", choices=("local", "drive"), default="local",
                       help="local copies each clip to scratch once (default)")
    group.add_argument("--sync-every", type=int, default=25)
    group.add_argument("--serve", dest="serve", action="store_true",
                       default=True)
    group.add_argument("--no-serve", dest="serve", action="store_false")
    group.add_argument("--base-url", default=None)
    group.add_argument("--server-wait", type=float, default=1800.0)
    group.add_argument("--vllm-arg", action="append", default=[])

    group = parser.add_argument_group("the run")
    group.add_argument("--model", default=DEFAULT_MODEL)
    group.add_argument("--port", type=int, default=DEFAULT_PORT)
    group.add_argument("--max-model-len", type=int,
                       default=DEFAULT_MAX_MODEL_LEN)
    group.add_argument("--gpu-fraction", type=float,
                       default=DEFAULT_GPU_FRACTION)
    group.add_argument("--visual-tokens", type=int,
                       default=VISUAL_TARGET_TOKENS,
                       help="whole-video budget, and the first knob to turn "
                            "if the engine will not start or a request "
                            "will not fit")
    group.add_argument("--passes", type=int, default=DEFAULT_PASSES)
    group.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    group.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    group.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    group.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    group.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    group.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    group.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    group.add_argument("--ignore-routing", action="store_true")

    group = parser.add_argument_group("scope")
    group.add_argument("--limit", type=int, default=0)
    group.add_argument("--only", type=int, nargs="+", default=None)
    group.add_argument("--shard", default=None, metavar="I/N")
    group.add_argument("--force", action="store_true")
    group.add_argument("--assemble-only", action="store_true")
    group.add_argument("--no-validate", dest="validate", action="store_false",
                       default=True)

    return parser.parse_args(argv)


def pip_install() -> None:
    # 0.27 is the floor: only it registers Qwen3_5ForConditionalGeneration.
    packages = ["vllm>=0.27", "requests"]
    print(f"installing {', '.join(packages)} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           *packages])
    print("installed. If vLLM replaced torch, Colab may ask you to restart "
          "the runtime -- do it, then re-run without --install.")


def resolve_paths(args: argparse.Namespace) -> None:
    root = args.drive_root
    args.queries = args.queries or root / "vqa.2026.testingDataset.Task2_v2.json"
    args.data_root = args.data_root or root / "videos_TEST_VQA"
    args.videos = args.videos or args.data_root / "videos"
    args.frame_index = args.frame_index or args.data_root / "frames" / "index.json"
    args.transcripts = args.transcripts or \
        args.data_root / "transcription_openwhisper"
    args.out = args.out or root / "runs" / f"{RUN_NAME}.run.task2.json"
    args.base_url = args.base_url or f"http://127.0.0.1:{args.port}"

    if args.routing is None:
        candidate = root / "query.routing.task1.json"
        args.routing = candidate if candidate.exists() else None


def select(queries: list[dict], args: argparse.Namespace,
           cache: dict[int, dict]) -> list[dict]:
    todo = list(queries)

    if args.only:
        wanted = set(args.only)
        todo = [q for q in todo if q["Q_ID"] in wanted]

    if args.shard:
        try:
            index, count = (int(part) for part in args.shard.split("/"))
        except ValueError:
            raise SystemExit("--shard takes I/N, for example 2/4")

        if not 1 <= index <= count:
            raise SystemExit(f"--shard {args.shard}: I must be 1..N")

        # Shard by video: the passes of one clip share a decoded prefix, and
        # splitting them across sessions throws that away.
        videos = sorted({q["Video_ID"] for q in todo})
        mine = {v for position, v in enumerate(videos)
                if position % count == index - 1}
        todo = [q for q in todo if q["Video_ID"] in mine]
        print(f"shard {index}/{count}: {len(mine)} videos, {len(todo)} queries")

    if not args.force:
        todo = [q for q in todo if q["Q_ID"] not in cache]

    if args.limit:
        todo = todo[:args.limit]

    return todo


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.install:
        pip_install()

    if args.passes < 1:
        raise SystemExit("--passes must be at least 1")

    resolve_paths(args)

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token

    if not Path(args.queries).exists():
        raise SystemExit(f"not found: {args.queries}\n  Is Drive mounted, and "
                         f"is --drive-root right?")

    queries = load_queries(args.queries)
    routing = load_routing(args.routing)
    print(f"queries: {len(queries)} | routing: "
          f"{len(routing) or 'OFF (every query gets video + transcript)'}")

    args.work.mkdir(parents=True, exist_ok=True)
    local_cache = args.work / f"rankings.task2.{RUN_NAME}.cache.jsonl"
    drive_cache = args.drive_root / "runs" / \
        f"rankings.task2.{RUN_NAME}.cache.jsonl"
    local_run = args.work / f"{RUN_NAME}.run.task2.json"

    mirror = Mirror(local_cache, drive_cache, args.sync_every)
    cache = load_cache(local_cache)
    print(f"cache: {len(cache)} queries already ranked")

    if args.assemble_only:
        mirror.close()
        run, missing = assemble(queries, cache)
        publish(run, local_run, args.out)
        print(f"wrote {args.out} ({len(run)} queries, {missing} unranked)")
        report(cache, queries)

        if not args.validate:
            return 0

        errors = validate_run(run, queries)
        print(f"validation: {len(errors)} errors")

        for line in errors[:20]:
            print(f"  {line}")

        return 1 if errors else 0

    if not Path(args.videos).is_dir():
        raise SystemExit(f"no videos directory at {args.videos} -- this model "
                         f"sends the clip itself, so the 11.4 GiB videos/ tree "
                         f"is required")

    durations = load_durations(args.frame_index)
    print(f"durations from the frame index: {len(durations)}")

    todo = select(queries, args, cache)

    if not todo:
        print("nothing to do")
        mirror.close()
        run, missing = assemble(queries, cache)
        publish(run, local_run, args.out)
        print(f"wrote {args.out} ({len(run)} queries, {missing} unranked)")
        return 0

    groups: dict[str, list[dict]] = {}

    for query in todo:
        groups.setdefault(query["Video_ID"], []).append(query)

    ordered = [groups[video] for video in sorted(groups)]
    print(f"to rank: {len(todo)} queries over {len(ordered)} videos, "
          f"{args.passes} passes each")

    stager = VideoStager(args.work, args.stage == "local")

    # The server must be allowed to open both the staged copies and, when
    # staging is off, the Drive tree.
    media_root = Path("/") if args.stage == "drive" else args.work

    server = VLLMServer(args.model, args.port, args.max_model_len,
                        args.gpu_fraction, media_root, args.work / "vllm.log",
                        args.vllm_arg)

    if args.serve:
        server.start(args.server_wait)
    elif not server.alive():
        raise SystemExit(f"--no-serve, but nothing answers at {args.base_url}")

    client = Qwen38Client(args.base_url, args.model, args.max_tokens,
                           args.temperature, args.top_p, args.top_k,
                           args.timeout, args.retries)

    lock = threading.Lock()
    done = failed = 0
    started = time.monotonic()

    def work(group: list[dict]) -> None:
        nonlocal done, failed

        video_id = group[0]["Video_ID"]
        source = resolve_video(args.videos, video_id)
        path = stager.local(source) if source else None
        uri = path.resolve().as_uri() if path else None

        stem = Path(video_id).stem
        duration = durations.get(stem)

        if duration is None and path:
            duration = probe_duration(path)

        transcript = load_transcript(args.transcripts, video_id)
        has_speech = bool(transcript and transcript.get("text"))

        for query in group:
            plan = plan_for(routing.get(query["Q_ID"]), query["Question"],
                            args.ignore_routing)

            # Never send a prompt with no evidence at all.
            if not plan["frames"] and not has_speech:
                plan = dict(plan, frames=True, rescued=True)

            send = uri if plan["frames"] else None

            def context_for(attached: bool, plan=plan) -> str:
                return build_context(duration,
                                     transcript if plan["transcript"] else None,
                                     plan, attached)

            def mm_kwargs_for(shrink: int, plan=plan) -> dict:
                return video_mm_kwargs(duration, plan, args.max_model_len,
                                       args.max_tokens, shrink,
                                       args.visual_tokens)

            try:
                record = rank_query(client, query, query["Options"],
                                    context_for, send, mm_kwargs_for,
                                    args.passes, plan, duration)
            except Exception as exc:  # noqa: BLE001 - one bad query must not
                with lock:            # sink a long run
                    failed += 1
                    print(f"  FAILED Q_ID {query['Q_ID']}: "
                          f"{type(exc).__name__}: {exc}", file=sys.stderr)
                continue

            mirror.write(record)

            with lock:
                cache[record["Q_ID"]] = record
                done += 1

                if done % 25 == 0 or done == len(todo):
                    rate = done / max(time.monotonic() - started, 1e-6)
                    left = (len(todo) - done) / rate if rate else 0
                    print(f"  {done}/{len(todo)} ranked, {failed} failed | "
                          f"{60 * rate:.1f}/min | ~{left / 3600:.1f} h left | "
                          f"staged {stager.bytes / 2**30:.1f} GiB")

                if done % max(args.sync_every, 1) == 0:
                    run, _ = assemble(queries, cache)
                    publish(run, local_run, args.out)

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            list(pool.map(work, ordered))
    except KeyboardInterrupt:
        print("\ninterrupted -- flushing what is done", file=sys.stderr)
    finally:
        mirror.close()

        if args.serve:
            server.stop()

    run, missing = assemble(queries, cache)
    publish(run, local_run, args.out)

    print(f"\nwrote {args.out}")
    print(f"  {len(run)} queries, {missing} still unranked "
          f"(they carry the official option order)")
    print(f"  cache mirrored to {drive_cache}")
    print(f"  elapsed {(time.monotonic() - started) / 3600:.2f} h")

    report(cache, queries)

    if args.validate:
        errors = validate_run(run, queries)

        if errors:
            print(f"\nVALIDATION: {len(errors)} errors")

            for line in errors[:20]:
                print(f"  {line}")

            if missing:
                print("  (unranked queries are the usual cause -- resume with "
                      "the same command, or run the other shards)")

            return 1

        print("\nVALIDATION: 0 errors -- shape matches "
              "validate_vqa_mc_2026.py's rules")

    return 0


if __name__ == "__main__":
    sys.exit(main())

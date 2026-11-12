#!/usr/bin/env python3
"""TREC 2026 VQA Task 2 (multiple choice) — InternVL3.5-30B-A3B ranker, for Colab.

A self-contained port of the InternVL configuration of
``VQA/rank.options.mc.muse.py``. Locally no separate client was written: the
Muse llama-server script serves this model with flags, and this script is that
combination made standalone. Nothing is imported from the project.

Dev-set result (400 labelled 2025 queries, official scorer): **0.760 Top-1 /
0.859 MRR**. That is 4.0 points behind Qwen3.8 and — alone among the challengers
— **not significantly behind (p = 0.064)**. It is the best of the three models
added late in the campaign and the fastest credible ranker here: the full 2,443
queries took **1.99 h**, with 80.8% of queries unanimous, 13 repairs, 0 failed
passes and 0 fallbacks.

Its weakness is complementarity rather than accuracy: it wins 25 queries where
Qwen3.8 is wrong and loses 41 where Qwen3.8 is right, so adding it to a fusion
costs more than it pays. No fusion of the eight dev runs beat Qwen3.8 alone once
nested cross-validation removed the search optimism.

The trap this script exists to carry forward
--------------------------------------------
**llama.cpp cuts every InternVL image into 448-px tiles, and
``--image-max-tokens`` does not cap it for this architecture.** One 504x896
frame cost ~1,280 tokens, sixteen frames came to **20,945** against a 16k slot,
and every image request returned HTTP 400. The fix is client-side and matches
InternVL's own video path (``max_num=1``): ``--frame-size 448`` resizes each
frame to a single tile, which brings the prompt to 4,561 tokens. It is on by
default here. Setting it to 0 will reproduce the failure.

Unlike Muse, **its image prefix is reused across passes** — 4,324 of 4,561
tokens came back cached on pass 2 — so the three-pass permutation is much
cheaper here than it is for Muse.

On the model identity, because the obvious name is wrong: **InternVL3 has no
26B size.** The line is 1/2/8/9/14/38/78B, and the 26B models are InternVL2 and
2.5, whose official AWQ builds are 23.3 GB because the 6B vision tower stays
fp16 — nothing in that family fits a 24 GB card with room for frames.
InternVL3.5-30B-A3B is an MoE with 3B active parameters and a small
InternViT-300M tower, which is why it both fits and runs fast.

Colab quickstart
----------------
Runtime → **A100 (40 GB)**. The server holds ~21.5 GB with 2 slots of 16,384
context and q8 KV. An L4 (22.5 GB) is very tight; try ``--slots 1``.

    from google.colab import drive; drive.mount('/content/drive')

    !python task2_internvl_colab.py --install \
        --drive-root /content/drive/MyDrive/TREC_2026

**The first run builds llama.cpp from source with CUDA on (~10-20 min).** There
are no official prebuilt Linux CUDA binaries, so this is the honest path. Pass
``--llama-cache /content/drive/MyDrive/TREC_2026/llama`` and the built binary is
copied to Drive, so later sessions skip the build entirely.

Expected Drive layout
---------------------
    TREC_2026/
      vqa.2026.testingDataset.Task2_v2.json
      videos_TEST_VQA/
        frames/index.json                       <- from VQA/extract.frames.py
        frames/<video_id>/001.jpg ...
        transcription_openwhisper/<video_id>.json
      query.routing.task1.json                  <- optional, see --routing
      runs/                                     <- created; outputs land here

The 11.4 GiB ``videos/`` tree is NOT needed: this model sees extracted frames.

**Upload frames as a tar, not as a tree.** ``frames/`` is 26,251 files for 1.15
GiB and Drive's FUSE mount is slow per-file. ``tar -cf frames.tar frames``
locally, put the tar on Drive, and ``--stage auto`` finds and unpacks it.

Resuming
--------
Every completed query is appended to a JSONL cache immediately, mirrored to
Drive every ``--sync-every`` records, and re-read on startup, so re-running the
identical command after a disconnect picks up where it stopped. At ~2 h for the
full set this model may well finish in one session.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
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

RUN_NAME = "internvl"

# ---------------------------------------------------------------------------
# Model, weights and serving defaults
# ---------------------------------------------------------------------------

DEFAULT_ALIAS = "internvl3.5-30b-a3b"
DEFAULT_REPO = "bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF"
DEFAULT_WEIGHTS = "OpenGVLab_InternVL3_5-30B-A3B-Q4_K_M.gguf"   # 18.63 GB
DEFAULT_MMPROJ = "mmproj-OpenGVLab_InternVL3_5-30B-A3B-f16.gguf"  # 0.64 GB
DEFAULT_PORT = 8013

# 2 slots x 16,384 with q8 KV = 21.5 GB of VRAM, measured.
DEFAULT_CONTEXT = 32768
DEFAULT_SLOTS = 2
# Deliberately 0: --image-max-tokens does NOT cap this architecture.
# --frame-size is what bounds the image cost here. See the docstring.
DEFAULT_IMAGE_TOKENS = 0
DEFAULT_KV_TYPE = ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]

# The architecture first shipped in b10358; b10976 is what was used locally.
DEFAULT_LLAMA_TAG = "b10976"

# No long reasoning trace here, so a permutation plus slack is enough.
DEFAULT_NUM_PREDICT = 256
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64

STRENGTHS = ("low", "medium", "high", "xhigh")
DEFAULT_STRENGTH = "low"

# REQUIRED here. Without it llama.cpp tiles each frame into five 448-px
# crops, sixteen frames come to 20,945 tokens, and every request 400s.
DEFAULT_FRAME_SIZE = 448

DEFAULT_PASSES = 3
DEFAULT_TIMEOUT = 900.0
DEFAULT_RETRIES = 3

# The server must have at least this much context per slot or the frames will
# not fit and every request 400s.
MIN_SLOT_CTX = 12288

# ---------------------------------------------------------------------------
# The prompt, word for word from VQA/rank.options.mc.muse.py, which is
# the client this configuration used
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You rank multiple-choice answer options for a question about a video you "
    "are shown as a sequence of frames. You reply with JSON only."
)

PROMPT_TEMPLATE = """{context}Question: {question}

Answer options:
{options}

Rank all {n} options from the most likely correct answer to the least likely.

Rules:
- Judge each option against the frames and the transcript, not against how it
  is worded.
- Reply with option numbers only, best first.
- Every number from 1 to {n} must appear exactly once. Never repeat a number
  and never leave one out.
- Exactly one option is correct. If the evidence is weak, still commit to an
  order -- put what the video supports most first.
{extra}
Reply with JSON: {{"ranking": [{slots}], "confidence": 0.0}}
where ranking holds all {n} option numbers best first, and confidence is how
confident you are in the first one, 0.0 to 1.0."""

OCR_RULE = ("- The answer is text rendered on screen. Read it exactly as "
            "written before choosing.\n")
TEMPORAL_RULE = ("- The answer depends on when something happens or in what "
                 "order. The frames are in time order; use their timestamps.\n")


# ---------------------------------------------------------------------------
# Ranking contract (shared core, verified against a shipped run)
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


def build_context(frames: dict | None, transcript: dict | None,
                  routed_out: bool = False) -> str:
    """The block that precedes the question, identical across a query's passes."""
    parts: list[str] = []

    if frames:
        stamps = ", ".join(mmss(t) for t in frames["timestamps"])
        parts.append(
            f"This video is {frames['duration_sec']:.0f} seconds long. You are "
            f"shown {frames['n_frames']} frames sampled evenly across it, in "
            f"order, at {stamps}."
        )
    elif routed_out:
        parts.append(
            "This question is answered by the soundtrack, so you are shown no "
            "frames. Judge the options from the transcript below."
        )
    else:
        parts.append(
            "The video for this question is unavailable, so you are shown no "
            "frames. Judge the options from the question alone, as plausibly "
            "as you can."
        )

    if transcript and transcript.get("text"):
        language = transcript.get("language") or "unknown"
        parts.append(
            f"Spoken audio (automatic transcript, language: {language}):\n"
            f"\"{transcript['text']}\""
        )
    elif frames:
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


def stage_frames(drive_frames: Path, local_root: Path, mode: str) -> Path:
    """Return the frames directory to read, copying it local when worthwhile.

    ``auto`` prefers a ``frames.tar`` next to the tree (one sequential read
    instead of 26k FUSE round trips), falls back to copying the tree, and
    falls back again to reading Drive in place.
    """
    if mode == "drive":
        return drive_frames

    local = Path(local_root) / "frames"

    if (local / "index.json").exists():
        print(f"frames already staged at {local}")
        return local

    tar = Path(drive_frames).parent / "frames.tar"

    if mode in ("auto", "tar") and tar.exists():
        print(f"unpacking {tar} -> {local_root} ...")
        started = time.monotonic()
        local_root.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tar) as archive:
            archive.extractall(local_root)

        print(f"  unpacked in {time.monotonic() - started:.0f}s")

        if (local / "index.json").exists():
            return local

        # A tar rooted at the frame dirs rather than at frames/.
        if (Path(local_root) / "index.json").exists():
            return Path(local_root)

    if mode == "tar":
        raise SystemExit(f"--stage tar asked for {tar}, which is not there")

    if not Path(drive_frames).exists():
        raise SystemExit(f"frames directory not found: {drive_frames}")

    print(f"copying frames from Drive -> {local} (1.15 GiB, 26k files; this "
          f"takes a while, and a frames.tar would be much faster) ...")
    started = time.monotonic()
    shutil.copytree(drive_frames, local, dirs_exist_ok=True)
    print(f"  copied in {time.monotonic() - started:.0f}s")

    return local


def encode_frame(path: Path, size: int) -> str:
    """One frame as base64 JPEG, optionally resized to size x size.

    llama.cpp tiles some architectures into fixed crops -- five for one of
    these 504x896 frames on InternVL, ~1,280 tokens -- and
    ``--image-max-tokens`` does not cap it. Resizing to one square tile is what
    the model's own video path does (``max_num=1``). Off by default, because
    Muse does not tile.
    """
    if not size:
        return base64.b64encode(Path(path).read_bytes()).decode()

    from PIL import Image

    buffer = io.BytesIO()
    Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC) \
        .save(buffer, format="JPEG", quality=95)

    return base64.b64encode(buffer.getvalue()).decode()


# ---------------------------------------------------------------------------
# llama.cpp: build it, fetch the weights, serve them
# ---------------------------------------------------------------------------

def build_llama(work: Path, tag: str, cache: Path | None) -> Path:
    """Return a CUDA-enabled llama-server binary, building it if need be.

    There are no official prebuilt Linux CUDA binaries for llama.cpp, so the
    honest path is a source build. It takes 10-20 minutes on a Colab VM, which
    is why ``--llama-cache`` exists: the build is copied there and reused.
    """
    local = work / "llama" / "llama-server"

    if local.exists() and os.access(local, os.X_OK):
        print(f"llama-server already built at {local}")
        return local

    if cache:
        cached = Path(cache) / "llama-server"

        if cached.exists():
            print(f"restoring llama-server from {cache} ...")
            local.parent.mkdir(parents=True, exist_ok=True)

            for item in Path(cache).iterdir():
                if item.is_file():
                    shutil.copy2(item, local.parent / item.name)

            local.chmod(0o755)
            return local

    source = work / "llama.cpp"

    if not source.exists():
        print(f"cloning llama.cpp {tag} ...")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", "--branch", tag,
             "https://github.com/ggml-org/llama.cpp", str(source)])

    print("building llama-server with CUDA (10-20 min; once per session "
          "unless --llama-cache is set) ...")
    started = time.monotonic()

    subprocess.check_call(
        ["cmake", "-B", str(source / "build"), "-S", str(source),
         "-DGGML_CUDA=ON", "-DLLAMA_CURL=OFF",
         "-DCMAKE_BUILD_TYPE=Release"])
    subprocess.check_call(
        ["cmake", "--build", str(source / "build"), "--config", "Release",
         "-j", str(os.cpu_count() or 4), "--target", "llama-server"])

    built = next(source.glob("build/**/llama-server"), None)

    if not built:
        raise SystemExit("the build finished but produced no llama-server "
                         "binary -- check the cmake output above")

    local.parent.mkdir(parents=True, exist_ok=True)

    # The binary needs the ggml shared objects beside it.
    for item in built.parent.iterdir():
        if item.is_file():
            shutil.copy2(item, local.parent / item.name)

    for pattern in ("build/**/*.so", "build/**/*.so.*"):
        for item in source.glob(pattern):
            shutil.copy2(item, local.parent / item.name)

    local.chmod(0o755)
    print(f"  built in {(time.monotonic() - started) / 60:.1f} min")

    if cache:
        try:
            Path(cache).mkdir(parents=True, exist_ok=True)

            for item in local.parent.iterdir():
                if item.is_file():
                    shutil.copy2(item, Path(cache) / item.name)

            print(f"  cached to {cache} -- later sessions will skip the build")
        except OSError as exc:
            print(f"  could not cache the build ({exc})", file=sys.stderr)

    return local


def fetch_weights(repo: str, filename: str, work: Path,
                  token: str | None) -> Path:
    """Download one GGUF from the Hub, or return an existing local copy."""
    direct = Path(filename)

    if direct.is_file():
        return direct

    from huggingface_hub import hf_hub_download

    print(f"fetching {repo}/{filename} ...")
    started = time.monotonic()

    path = Path(hf_hub_download(repo_id=repo, filename=filename,
                                local_dir=str(work / "models"), token=token))

    print(f"  {path.stat().st_size / 2**30:.2f} GiB in "
          f"{(time.monotonic() - started) / 60:.1f} min")

    return path


class LlamaServer:
    """Starts llama-server in this VM, waits for it, and stops it on exit."""

    def __init__(self, binary: Path, weights: Path, mmproj: Path, alias: str,
                 port: int, context: int, slots: int, image_tokens: int,
                 log: Path, extra: list[str]) -> None:
        self.alias = alias
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.log = Path(log)
        self.process: subprocess.Popen | None = None

        self.command = [
            str(binary),
            "-m", str(weights),
            "--mmproj", str(mmproj),
            "--alias", alias,
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(context),
            "-np", str(slots),
            "-ngl", "all",
            "-fa", "on",
            "--jinja",
            "--reasoning-format", "auto",
            "--no-webui",
        ]

        if image_tokens:
            self.command += ["--image-max-tokens", str(image_tokens)]

        self.command += list(extra)

    def start(self, wait: float) -> None:
        if self.alive():
            print(f"a server is already answering on {self.url} -- using it")
            return

        self.log.parent.mkdir(parents=True, exist_ok=True)
        handle = self.log.open("w", encoding="utf-8")

        print("starting llama-server:\n  " + " ".join(self.command))
        print(f"  log: {self.log}")

        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = (
            str(Path(self.command[0]).parent) + ":" +
            environment.get("LD_LIBRARY_PATH", ""))

        self.process = subprocess.Popen(
            self.command, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True, env=environment)

        started = time.monotonic()
        last = 0.0

        while time.monotonic() - started < wait:
            if self.process.poll() is not None:
                raise SystemExit(
                    f"llama-server exited with code {self.process.returncode} "
                    f"before serving. Last lines of {self.log}:\n" + self.tail())

            if self.alive():
                print(f"  server up after {time.monotonic() - started:.0f}s")
                return

            if time.monotonic() - last > 30:
                last = time.monotonic()
                print(f"  loading ({time.monotonic() - started:.0f}s) ...")

            time.sleep(3.0)

        raise SystemExit(f"llama-server did not answer within {wait:.0f}s. "
                         f"Last lines of {self.log}:\n" + self.tail())

    def alive(self) -> bool:
        import requests

        try:
            return requests.get(self.url + "/v1/models",
                                timeout=5).status_code == 200
        except Exception:  # noqa: BLE001 - any failure means "not up yet"
            return False

    def preflight(self, slots: int) -> None:
        """Refuse to start a long run against a server that cannot do the job.

        A server with no vision loaded answers every request happily and ranks
        from the question alone -- the exact failure that cost the Cosmos run
        23% of its queries. Three cheap checks catch it.
        """
        import requests

        try:
            props = requests.get(self.url + "/props", timeout=20).json()
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"cannot read {self.url}/props: {exc}")

        served = [entry["id"] for entry in
                  requests.get(self.url + "/v1/models", timeout=20).json()["data"]]

        if self.alias not in served:
            raise SystemExit(f"{self.url} serves {served}, not {self.alias!r}")

        has_vision = bool(props.get("modalities", {}).get("vision")) or \
            bool(props.get("has_mtmd")) or bool(props.get("mmproj"))

        if not has_vision:
            raise SystemExit(
                "the server reports NO VISION. It would rank every query from "
                "the question text alone and the run would look finished and "
                "be worthless. Check --mmproj.")

        total = props.get("n_ctx") or props.get("default_generation_settings",
                                                {}).get("n_ctx") or 0
        per_slot = int(total) if int(total) <= 65536 and slots == 1 \
            else int(total)

        if per_slot and per_slot < MIN_SLOT_CTX:
            raise SystemExit(
                f"only {per_slot} context tokens per slot; {MIN_SLOT_CTX} is "
                f"the floor for 16 frames plus eight long options. Raise -c or "
                f"lower --slots.")

        print(f"preflight: vision live, alias {self.alias}, "
              f"{per_slot} ctx per slot")

    def tail(self, lines: int = 40) -> str:
        try:
            return "\n".join(self.log.read_text(
                encoding="utf-8", errors="replace").splitlines()[-lines:])
        except OSError:
            return "(no log)"

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            print("stopping llama-server ...")

            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=60)
            except Exception:  # noqa: BLE001 - best effort teardown
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------

class LlamaCpp:
    """OpenAI-compatible client for llama-server.

    What differs from a vLLM client is what comes back: the reasoning trace
    arrives in ``reasoning_content`` and only the answer in ``content``, so an
    empty or unparseable ``content`` with ``finish_reason == "length"`` means
    the budget ran out mid-trace -- that one case is retried at twice the
    budget. Every reply also carries llama-server's ``timings``, kept per pass
    so prefill, decode and image-prefix reuse are measured rather than assumed.
    """

    name = "llamacpp"

    def __init__(self, base: str, model: str, num_predict: int,
                 temperature: float, top_p: float, top_k: int,
                 timeout: float, retries: int, use_schema: bool) -> None:
        self.url = base.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.num_predict = num_predict
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.timeout = timeout
        self.retries = retries
        self.use_schema = use_schema
        self.local = threading.local()

    @property
    def session(self):
        import requests

        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()

        return self.local.session

    def ask(self, prompt: str, schema: dict, seed: int,
            images: list[str] | None, strength: str
            ) -> tuple[dict, float, dict]:
        """One call. Returns the parsed reply, its wall time and its costs."""
        import requests

        content: list[dict] = []

        if images:
            content += [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                for image in images
            ]

        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "max_tokens": self.num_predict,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": seed,
            # The template renders this as the system prompt's "Reasoning
            # strength: ..." line. Leaving it out means "high".
            "chat_template_kwargs": {"reasoning_strength": strength},
            "cache_prompt": True,
        }

        if self.use_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "ranking",
                                "schema": dict(schema,
                                               additionalProperties=False)},
            }

        budget = self.num_predict
        last: Exception | None = None

        for attempt in range(self.retries):
            payload["max_tokens"] = budget
            started = time.monotonic()

            try:
                response = self.session.post(self.url, json=payload,
                                             timeout=self.timeout)
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                message = choice.get("message") or {}
                text = (message.get("content") or "").strip()
                timings = body.get("timings") or {}

                costs = {
                    "finish_reason": choice.get("finish_reason"),
                    "prompt_tokens": timings.get("prompt_n"),
                    "cached_tokens": timings.get("prompt_n_cached"),
                    "completion_tokens": timings.get("predicted_n"),
                    "prefill_ms": timings.get("prompt_ms"),
                    "decode_ms": timings.get("predicted_ms"),
                    "strength": strength,
                    "extended": budget != self.num_predict,
                }

                if not text:
                    raise ValueError(
                        f"empty content (finish_reason="
                        f"{choice.get('finish_reason')})")

                return (parse_json_object(text), time.monotonic() - started,
                        costs)

            except (requests.RequestException, ValueError, KeyError,
                    IndexError, json.JSONDecodeError) as exc:
                last = exc

                # Cut off mid-trace: the one failure more budget fixes, and
                # only once -- a model that will not finish at 2x will not
                # finish at 4x either.
                if "length" in str(exc) and budget == self.num_predict:
                    budget *= 2
                elif attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"{type(last).__name__}: {last}")


def rank_query(client: LlamaCpp, query: dict, options: list[str],
               context: str, images: list[str] | None, passes: int,
               plan: dict, strength: str) -> dict:
    """Ask for every presentation order of one query and combine the rankings."""
    q_id = query["Q_ID"]
    n = len(options)
    schema = ranking_schema(n)
    slots = ", ".join(str(i) for i in range(1, n + 1))

    extra = ""

    if plan["ocr"]:
        extra += OCR_RULE
    if plan["temporal"]:
        extra += TEMPORAL_RULE

    results: list[dict] = []
    failures = 0
    repairs = 0

    for index, order in enumerate(presentation_orders(n, passes, q_id)):
        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=query["Question"],
            options=render_options(options, order),
            n=n, slots=slots, extra=extra)

        try:
            reply, elapsed, costs = client.ask(prompt, schema,
                                               seed_for(q_id, index), images,
                                               strength)
        except RuntimeError as exc:
            failures += 1
            print(f"  Q_ID {q_id} pass {index}: {exc}", file=sys.stderr)
            continue

        positions, repaired = as_permutation(reply.get("ranking"), n)
        repairs += int(repaired)

        results.append({
            "index": index,
            "ranking": [order[position] for position in positions],
            "order": order,
            "confidence": reply.get("confidence"),
            "repaired": repaired,
            "cost": costs,
            "elapsed": round(elapsed, 3),
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
        "backend": client.name,
        "model": client.model,
        "strength": strength,
        "n_options": n,
        "n_passes": passes,
        "n_replies": len(results),
        "n_pass_failures": failures,
        "n_repaired": repairs,
        "fallback": fallback,
        "n_frames": len(images) if images else 0,
        "has_transcript": "Spoken audio" in context,
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

    prompt_tokens = cached = completion = 0
    counted = 0

    for record in records:
        for entry in record.get("passes") or []:
            cost = entry.get("cost") or {}

            if cost.get("prompt_tokens"):
                prompt_tokens += cost["prompt_tokens"]
                cached += cost.get("cached_tokens") or 0
                completion += cost.get("completion_tokens") or 0
                counted += 1

    print(f"\n{'-' * 68}\nranked          {n} / {len(queries)} queries")
    print(f"pass failures   {sum(r.get('n_pass_failures', 0) for r in records)}")
    print(f"repaired        {sum(r.get('n_repaired', 0) for r in records)}")
    print(f"fallbacks       {sum(1 for r in records if r.get('fallback'))}")
    print(f"no frames       {sum(1 for r in records if not r.get('n_frames'))}")
    print(f"with transcript {sum(1 for r in records if r.get('has_transcript'))}")
    print(f"unanimous       {unanimous} ({100 * unanimous / n:.1f}%)")
    print(f"rank 1 on the officially-first option: "
          f"{100 * firsts_on_first / n:.1f}% (chance {100 * chance:.1f}%)")

    if counted:
        print(f"per pass: {prompt_tokens / counted:.0f} prompt tokens "
              f"({cached / counted:.0f} cached), "
              f"{completion / counted:.0f} completion")

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
    group.add_argument("--frames", type=Path, default=None)
    group.add_argument("--transcripts", type=Path, default=None)
    group.add_argument("--routing", type=Path, default=None,
                       help="absent means every query gets frames AND "
                            "transcript. The local 2026 runs had routing ON")
    group.add_argument("--work", type=Path, default=Path("/content/trec_work"))
    group.add_argument("--out", type=Path, default=None)

    group = parser.add_argument_group("colab")
    group.add_argument("--install", action="store_true")
    group.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    group.add_argument("--stage", choices=("auto", "tar", "copy", "drive"),
                       default="auto")
    group.add_argument("--sync-every", type=int, default=25)
    group.add_argument("--llama-cache", type=Path, default=None,
                       help="a Drive directory to keep the built llama-server "
                            "in, so later sessions skip the 10-20 min build")
    group.add_argument("--llama-bin", type=Path, default=None,
                       help="an existing llama-server binary to use as-is")
    group.add_argument("--llama-tag", default=DEFAULT_LLAMA_TAG)
    group.add_argument("--serve", dest="serve", action="store_true",
                       default=True)
    group.add_argument("--no-serve", dest="serve", action="store_false")
    group.add_argument("--base-url", default=None)
    group.add_argument("--server-wait", type=float, default=1800.0)
    group.add_argument("--llama-arg", action="append",
                       default=list(DEFAULT_KV_TYPE),
                       help="extra llama-server arguments; q8 KV by "
                            "default, which is what made 2 slots fit")

    group = parser.add_argument_group("weights")
    group.add_argument("--repo", default=DEFAULT_REPO)
    group.add_argument("--weights", default=DEFAULT_WEIGHTS,
                       help="GGUF filename in --repo, or a local path")
    group.add_argument("--mmproj", default=DEFAULT_MMPROJ,
                       help="the vision projector; without it the server "
                            "loads with NO VISION and preflight refuses")
    group.add_argument("--alias", default=DEFAULT_ALIAS)

    group = parser.add_argument_group("the run")
    group.add_argument("--port", type=int, default=DEFAULT_PORT)
    group.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    group.add_argument("--slots", type=int, default=DEFAULT_SLOTS,
                       help="llama-server -np. Workers default to this")
    group.add_argument("--image-tokens", type=int, default=DEFAULT_IMAGE_TOKENS,
                       help="--image-max-tokens. The native default is 4096 "
                            "PER IMAGE, which overflows any sane context")
    group.add_argument("--frame-size", type=int, default=DEFAULT_FRAME_SIZE,
                       help="resize each frame to N x N before sending. "
                            "REQUIRED for this model -- 0 reproduces "
                            "the 20,945-token HTTP 400")
    group.add_argument("--strength", choices=STRENGTHS,
                       default=DEFAULT_STRENGTH,
                       help="reasoning strength, sent on every call by the "
                            "shared client. Harmless for this template")
    group.add_argument("--passes", type=int, default=DEFAULT_PASSES)
    group.add_argument("--workers", type=int, default=0,
                       help="default: one per slot")
    group.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    group.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    group.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    group.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    group.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    group.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    group.add_argument("--no-schema", dest="schema", action="store_false",
                       default=True)
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
    packages = ["requests", "pillow", "huggingface_hub"]
    print(f"installing {', '.join(packages)} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           *packages])
    print("installed. Note this backend needs cmake and the CUDA toolkit, "
          "both already present on a Colab GPU runtime.")


def resolve_paths(args: argparse.Namespace) -> None:
    root = args.drive_root
    args.queries = args.queries or root / "vqa.2026.testingDataset.Task2_v2.json"
    args.data_root = args.data_root or root / "videos_TEST_VQA"
    args.frames = args.frames or args.data_root / "frames"
    args.transcripts = args.transcripts or \
        args.data_root / "transcription_openwhisper"
    args.out = args.out or root / "runs" / f"{RUN_NAME}.run.task2.json"
    args.base_url = args.base_url or f"http://127.0.0.1:{args.port}"

    if args.routing is None:
        candidate = root / "query.routing.task1.json"
        args.routing = candidate if candidate.exists() else None

    if not args.workers:
        args.workers = max(1, args.slots)


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
            raise SystemExit("--shard takes I/N, for example 2/3")

        if not 1 <= index <= count:
            raise SystemExit(f"--shard {args.shard}: I must be 1..N")

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
          f"{len(routing) or 'OFF (every query gets frames + transcript)'}")

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

    frames_dir = stage_frames(args.frames, args.work, args.stage)
    index_path = Path(frames_dir) / "index.json"

    if not index_path.exists():
        raise SystemExit(f"no frame index at {index_path} -- this pass needs "
                         f"frames from VQA/extract.frames.py")

    frame_index = load_json(index_path)["videos"]
    print(f"frame index: {len(frame_index)} videos")

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
    blind = sum(len(g) for g in ordered
                if Path(g[0]["Video_ID"]).stem not in frame_index)

    print(f"to rank: {len(todo)} queries over {len(ordered)} videos "
          f"({blind} have no frames), {args.passes} passes each")

    server = None

    if args.serve:
        binary = Path(args.llama_bin) if args.llama_bin else \
            build_llama(args.work, args.llama_tag, args.llama_cache)
        weights = fetch_weights(args.repo, args.weights, args.work,
                                args.hf_token)
        mmproj = fetch_weights(args.repo, args.mmproj, args.work,
                               args.hf_token)

        server = LlamaServer(binary, weights, mmproj, args.alias, args.port,
                             args.context, args.slots, args.image_tokens,
                             args.work / "llama-server.log", args.llama_arg)
        server.start(args.server_wait)
        server.preflight(args.slots)
    else:
        server = LlamaServer(Path("llama-server"), Path("."), Path("."),
                             args.alias, args.port, args.context, args.slots,
                             args.image_tokens, args.work / "llama-server.log",
                             [])

        if not server.alive():
            raise SystemExit(f"--no-serve, but nothing answers at "
                             f"{args.base_url}")

        server.process = None
        server.preflight(args.slots)

    client = LlamaCpp(args.base_url, args.alias, args.num_predict,
                      args.temperature, args.top_p, args.top_k, args.timeout,
                      args.retries, args.schema)

    lock = threading.Lock()
    done = failed = 0
    started = time.monotonic()

    def work(group: list[dict]) -> None:
        nonlocal done, failed

        video_id = group[0]["Video_ID"]
        entry = frame_index.get(Path(video_id).stem)
        images = None

        if entry:
            frame_dir = Path(frames_dir) / Path(video_id).stem

            try:
                images = [encode_frame(frame_dir / name, args.frame_size)
                          for name in entry["files"]]
            except (OSError, KeyError, TypeError) as exc:
                print(f"  {video_id}: unreadable frames ({exc})",
                      file=sys.stderr)
                images, entry = None, None

        transcript = load_transcript(args.transcripts, video_id)
        has_speech = bool(transcript and transcript.get("text"))

        for query in group:
            plan = plan_for(routing.get(query["Q_ID"]), query["Question"],
                            args.ignore_routing)

            if not plan["frames"] and not has_speech:
                plan = dict(plan, frames=True, rescued=True)

            context = build_context(entry if plan["frames"] else None,
                                    transcript if plan["transcript"] else None,
                                    routed_out=not plan["frames"])

            try:
                record = rank_query(client, query, query["Options"], context,
                                    images if plan["frames"] else None,
                                    args.passes, plan, args.strength)
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
                          f"{60 * rate:.1f}/min | ~{left / 3600:.1f} h left")

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

        if args.serve and server:
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
                      "the same command)")

            return 1

        print("\nVALIDATION: 0 errors -- shape matches "
              "validate_vqa_mc_2026.py's rules")

    return 0


if __name__ == "__main__":
    sys.exit(main())

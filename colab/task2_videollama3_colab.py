#!/usr/bin/env python3
"""TREC 2026 VQA Task 2 (multiple choice) — VideoLLaMA3-7B ranker, for Colab.

A self-contained port of ``VQA/rank.options.mc.videollama3.py``. Nothing is
imported from the project; copy this one file to Colab and run it.

Dev-set result (400 labelled 2025 queries, official scorer): **0.738 Top-1 /
0.843 MRR**, -6.2 points against Qwen3.8 (p = 0.004). It beats Gemma 4 12B at
less than two-thirds the parameters, but it loses 48 queries where Qwen3.8 is
right while winning only 23, so it does not help a fusion either. The 2026 pass
took **6.12 h** of GPU locally — the slowest of the late additions.

Read this first: the install is destructive
-------------------------------------------
This model has **no vLLM architecture** (vllm issue #16627) and its remote code
**pins transformers 4.46.3**, so it runs in-process from a pinned environment.
``--install`` will downgrade ``transformers`` and hold ``numpy`` below 2, which
breaks other things in a shared Colab runtime.

    # cell 1 -- then RESTART THE RUNTIME when it finishes
    !python task2_videollama3_colab.py --install

    # cell 2 -- after the restart
    from google.colab import drive; drive.mount('/content/drive')
    !python task2_videollama3_colab.py \\
        --drive-root /content/drive/MyDrive/TREC_2026

Do not run another model's script in the same runtime afterwards without
restarting again.

Attention is ``sdpa``: flash-attn cannot be built against this pin, and the
remote code falls back cleanly.

The two things this script fixes in the model's own code
--------------------------------------------------------
**1. Frames get their real timestamps.** VideoLLaMA3's loader stamps a
*directory* of frames at a fixed ``vid_fps = 3``, which would mislabel every
frame of every clip. Passing a ``video`` item that is already a list of CHW
arrays skips the loader entirely; ``num_frames`` and ``timestamps`` are then
ours, and the chat template prints the true
``Time 0.6s:<image>, Time 1.7s:<image>, ...`` from the frame index.

**2. The sdpa vision encoder is patched to attend one frame at a time.** The
remote version builds a full ``[S, S]`` boolean mask over every patch in the
clip and hands it to ``scaled_dot_product_attention``, which then materialises
heads x S x S scores — about **8 GB per layer at 16k patches**. On Windows the
driver spilled that into system RAM instead of raising OOM, so the first smoke
test sat at 23.7 GB and 100% GPU for over ten minutes with zero queries
finished; on Colab you would get a plain CUDA OOM instead. The mask is
block-diagonal, one block per frame (``cu_seqlens``), so attending block by
block is the same computation with no mask and a few MB of scores. The patch was
checked against the masked original on a real layer: **max absolute difference
0.00000**. With it the local run went to 7.1 queries/min.

The visual budget is capped at ``--max-visual-tokens 4096`` for the whole clip
(252 tokens per frame at 16 frames), which bounds prefill and the vision encoder
together. If you hit OOM, lower that before anything else.

Colab quickstart
----------------
Runtime → **L4 (22.5 GB)** or **A100**. bf16 weights are ~16.1 GB. A T4 cannot
hold them.

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

The 11.4 GiB ``videos/`` tree is NOT needed: this pass sends extracted frames
with their true sampling instants.

**Upload frames as a tar, not as a tree.** ``frames/`` is 26,251 files for 1.15
GiB and Drive's FUSE mount is slow per-file. ``tar -cf frames.tar frames``
locally, put the tar on Drive, and ``--stage auto`` finds and unpacks it.

Resuming, which you will need
-----------------------------
6.12 h locally means at least two Colab sessions. Every completed query is
appended to a JSONL cache immediately, mirrored to Drive every ``--sync-every``
records, and re-read on startup, so re-running the identical command after a
disconnect picks up where it stopped. ``--shard 1/2`` splits deliberately.

There is no server here, so ``--workers`` above 1 does not help: one model, one
GPU, one call at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

RUN_NAME = "videollama3"

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA3-7B"

# The remote code pins this exact version; anything newer changes the processor
# signature the block-wise patch reaches into.
PINNED_TRANSFORMERS = "4.46.3"

DEFAULT_NUM_PREDICT = 128
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64

DEFAULT_PASSES = 3
DEFAULT_RETRIES = 3

# Whole-clip: the image processor shrinks every frame until the clip fits, so
# this bounds prefill and the vision encoder together. 252 tokens per frame at
# 16 frames.
DEFAULT_MAX_VISUAL_TOKENS = 4096

# ---------------------------------------------------------------------------
# The prompt, word for word from VQA/rank.options.mc.videollama3.py
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


def load_frames(directory: Path, entry: dict) -> list[tuple]:
    """The extracted JPEGs as (CHW uint8 array, seconds) pairs, in order.

    The array layout is what VideoLLaMA3's own loader returns. The timestamps
    are the real sampling instants from the frame index, not ``index / 3``,
    which is what the processor assumes for a directory of frames.
    """
    import numpy as np
    from PIL import Image

    return [(np.asarray(Image.open(Path(directory) / name).convert("RGB"))
             .transpose(2, 0, 1), float(stamp))
            for name, stamp in zip(entry["files"], entry["timestamps"])]


# ---------------------------------------------------------------------------
# The model, in-process
# ---------------------------------------------------------------------------

def make_blockwise_forward(rotary):
    """A VisionSdpaAttention.forward that attends one frame at a time.

    The remote version builds one [S, S] boolean mask over every patch in the
    clip and hands it to sdpa, which then falls back to materialising
    heads x S x S scores -- about 8 GB per layer at 16k patches. The mask is
    block-diagonal, one block per frame (``cu_seqlens``), so attending block by
    block is the same computation with no mask and a few MB of scores. Verified
    exact: max absolute difference 0.00000 against the masked original.
    """
    import torch
    import torch.nn.functional as F

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
        seq_length = hidden_states.shape[0]
        shape = (seq_length, self.num_heads, self.head_dim)
        query = self.q_proj(hidden_states).view(shape)
        key = self.k_proj(hidden_states).view(shape)
        value = self.v_proj(hidden_states).view(shape)

        query = rotary(query.unsqueeze(0), rotary_pos_emb).squeeze(0)
        key = rotary(key.unsqueeze(0), rotary_pos_emb).squeeze(0)

        output = torch.empty_like(query)
        bounds = cu_seqlens.tolist()

        for start, end in zip(bounds[:-1], bounds[1:]):
            output[start:end] = F.scaled_dot_product_attention(
                query[start:end].transpose(0, 1),
                key[start:end].transpose(0, 1),
                value[start:end].transpose(0, 1)).transpose(0, 1)

        return self.out_proj(output.reshape(seq_length, -1))

    return forward


class VideoLlama3:
    """VideoLLaMA3-7B loaded in-process with transformers, one call at a time.

    Frames go in as one ``video`` item that is already a list of frames, with
    ``num_frames`` and ``timestamps`` set here -- the processor then skips its
    own loader and the chat template prints the real ``Time 4.5s:`` labels.
    There is no grammar, so a reply with no JSON is sampled again (at twice the
    budget if it was cut off) before the pass is counted as failed.
    """

    name = "transformers"

    def __init__(self, path: str, num_predict: int, temperature: float,
                 top_p: float, top_k: int, retries: int,
                 max_visual_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch = torch
        self.model = MODEL_NAME
        self.num_predict = num_predict
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.retries = retries
        self.lock = threading.Lock()

        started = time.monotonic()
        self.net = AutoModelForCausalLM.from_pretrained(
            str(path), trust_remote_code=True, device_map="cuda:0",
            torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        self.processor = AutoProcessor.from_pretrained(
            str(path), trust_remote_code=True)
        self.processor.image_processor.max_tokens = max_visual_tokens

        # See make_blockwise_forward(): without this the first query either
        # OOMs or, on a driver that spills to host RAM, crawls forever.
        patched = 0

        for module in self.net.modules():
            cls = type(module)

            if cls.__name__ == "VisionSdpaAttention":
                if not getattr(cls, "_blockwise", False):
                    rotary = sys.modules[cls.__module__] \
                        .apply_rotary_pos_emb_vision
                    cls.forward = make_blockwise_forward(rotary)
                    cls._blockwise = True

                patched += 1

        if not patched:
            raise SystemExit("no VisionSdpaAttention layers found -- the "
                             "remote code changed and the block-wise patch did "
                             "not apply. Running without it will exhaust the "
                             "card; stopping instead.")

        print(f"loaded in {time.monotonic() - started:.0f}s | "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB on the GPU | "
              f"visual budget {max_visual_tokens} tokens per clip | "
              f"{patched} vision attention layers made block-wise")

    def ask(self, prompt: str, seed: int,
            frames: list[tuple] | None) -> tuple[dict, float, dict]:
        """One call. Returns the parsed reply, its wall time and its costs."""
        torch = self.torch
        content: list[dict] = []

        if frames:
            content.append({"type": "video",
                            "video": [frame for frame, _ in frames],
                            "num_frames": len(frames),
                            "timestamps": [stamp for _, stamp in frames]})

        content.append({"type": "text", "text": prompt})

        with self.lock:
            inputs = self.processor(
                conversation=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": content}],
                add_generation_prompt=True, return_tensors="pt")
            inputs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(
                    torch.bfloat16)

            prompt_ids = inputs["input_ids"][0]
            sampling = ({"do_sample": True, "temperature": self.temperature,
                         "top_p": self.top_p, "top_k": self.top_k}
                        if self.temperature > 0 else {"do_sample": False})
            budget = self.num_predict
            last: Exception | None = None

            for attempt in range(self.retries):
                torch.manual_seed(seed + attempt)
                started = time.monotonic()

                try:
                    with torch.inference_mode():
                        output = self.net.generate(**inputs,
                                                   max_new_tokens=budget,
                                                   **sampling)
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"CUDA OOM at {len(prompt_ids)} prompt tokens: {exc}. "
                        f"Lower --max-visual-tokens.") from exc

                elapsed = time.monotonic() - started
                ids = output[0]

                # The remote generate() embeds the prompt itself and so returns
                # only the new tokens; strip the prompt anyway if it came back.
                if ids.shape[0] > prompt_ids.shape[0] and torch.equal(
                        ids[:prompt_ids.shape[0]].cpu(), prompt_ids.cpu()):
                    ids = ids[prompt_ids.shape[0]:]

                completion = int(ids.shape[0])
                text = self.processor.batch_decode(
                    [ids], skip_special_tokens=True)[0].strip()

                try:
                    reply = parse_json_object(text)
                except ValueError:
                    last = ValueError(f"no JSON in reply ({completion} "
                                      f"tokens): {text[:120]!r}")

                    # Cut off: the one failure more budget fixes.
                    if completion >= budget and budget == self.num_predict:
                        budget *= 2

                    continue

                return reply, elapsed, {
                    "finish_reason": "length" if completion >= budget
                    else "stop",
                    "prompt_tokens": int(prompt_ids.shape[0]),
                    "completion_tokens": completion,
                    "extended": budget != self.num_predict,
                    "attempts": attempt + 1,
                }

        raise RuntimeError(f"{type(last).__name__}: {last}")


def rank_query(client: VideoLlama3, query: dict, options: list[str],
               context: str, frames: list[tuple] | None, passes: int,
               plan: dict) -> dict:
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

    for index, order in enumerate(presentation_orders(n, passes, q_id)):
        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=query["Question"],
            options=render_options(options, order),
            n=n, slots=slots, extra=extra)

        try:
            reply, elapsed, costs = client.ask(prompt, seed_for(q_id, index),
                                               frames)
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
        "n_options": n,
        "n_passes": passes,
        "n_replies": len(results),
        "n_pass_failures": failures,
        "n_repaired": repairs,
        "fallback": fallback,
        "n_frames": len(frames) if frames else 0,
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

    prompt_tokens = completion = counted = 0

    for record in records:
        for entry in record.get("passes") or []:
            cost = entry.get("cost") or {}

            if cost.get("prompt_tokens"):
                prompt_tokens += cost["prompt_tokens"]
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
        print(f"per pass: {prompt_tokens / counted:.0f} prompt tokens, "
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
                            "transcript. The local 2026 run had routing ON")
    group.add_argument("--work", type=Path, default=Path("/content/trec_work"))
    group.add_argument("--out", type=Path, default=None)

    group = parser.add_argument_group("colab")
    group.add_argument("--install", action="store_true",
                       help="pin transformers %s and numpy<2, then STOP. "
                            "Restart the runtime before running for real"
                            % PINNED_TRANSFORMERS)
    group.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    group.add_argument("--stage", choices=("auto", "tar", "copy", "drive"),
                       default="auto")
    group.add_argument("--sync-every", type=int, default=25)
    group.add_argument("--model-path", default=None,
                       help="a local checkpoint directory; default downloads "
                            "%s (16.1 GB)" % MODEL_NAME)

    group = parser.add_argument_group("the run")
    group.add_argument("--passes", type=int, default=DEFAULT_PASSES)
    group.add_argument("--max-visual-tokens", type=int,
                       default=DEFAULT_MAX_VISUAL_TOKENS,
                       help="whole-clip budget. Lower this first on OOM")
    group.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    group.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    group.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    group.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
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
    """Pin the environment this model's remote code requires, then stop.

    transformers 4.46.3 and numpy<2 are not negotiable -- the remote code
    imports against them -- and installing them mid-session leaves the already
    imported modules stale, so this deliberately does not continue.
    """
    packages = [f"transformers=={PINNED_TRANSFORMERS}", "numpy<2", "decord",
                "accelerate", "pillow", "huggingface_hub"]
    print(f"installing {', '.join(packages)} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           *packages])
    print("\n" + "=" * 68)
    print("RESTART THE RUNTIME NOW, then run this script again WITHOUT")
    print("--install. The pins above only take effect in a fresh process,")
    print("and they will break other models' scripts in this runtime.")
    print("=" * 68)


def resolve_paths(args: argparse.Namespace) -> None:
    root = args.drive_root
    args.queries = args.queries or root / "vqa.2026.testingDataset.Task2_v2.json"
    args.data_root = args.data_root or root / "videos_TEST_VQA"
    args.frames = args.frames or args.data_root / "frames"
    args.transcripts = args.transcripts or \
        args.data_root / "transcription_openwhisper"
    args.out = args.out or root / "runs" / f"{RUN_NAME}.run.task2.json"

    if args.routing is None:
        candidate = root / "query.routing.task1.json"
        args.routing = candidate if candidate.exists() else None


def preflight_env() -> None:
    """Refuse to load the model against a transformers the remote code hates."""
    try:
        import transformers
    except ImportError:
        raise SystemExit("transformers is not installed -- run with --install, "
                         "restart the runtime, then try again")

    if transformers.__version__ != PINNED_TRANSFORMERS:
        print(f"WARNING: transformers {transformers.__version__} is installed, "
              f"but this model's remote code pins {PINNED_TRANSFORMERS}. "
              f"If loading fails, that is why.", file=sys.stderr)


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
        return 0

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

    preflight_env()

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

    path = args.model_path

    if not path:
        from huggingface_hub import snapshot_download

        print(f"fetching {MODEL_NAME} (16.1 GB) ...")
        path = snapshot_download(MODEL_NAME,
                                 local_dir=str(args.work / "VideoLLaMA3-7B"),
                                 token=args.hf_token)

    client = VideoLlama3(path, args.num_predict, args.temperature, args.top_p,
                         args.top_k, args.retries, args.max_visual_tokens)

    done = failed = 0
    started = time.monotonic()

    try:
        # One model on one card: strictly sequential, unlike the served passes.
        for group in ordered:
            video_id = group[0]["Video_ID"]
            entry = frame_index.get(Path(video_id).stem)
            frames = None

            if entry:
                try:
                    frames = load_frames(Path(frames_dir) /
                                         Path(video_id).stem, entry)
                except (OSError, KeyError, TypeError) as exc:
                    print(f"  {video_id}: unreadable frames ({exc})",
                          file=sys.stderr)
                    frames, entry = None, None

            transcript = load_transcript(args.transcripts, video_id)
            has_speech = bool(transcript and transcript.get("text"))

            for query in group:
                plan = plan_for(routing.get(query["Q_ID"]), query["Question"],
                                args.ignore_routing)

                if not plan["frames"] and not has_speech:
                    plan = dict(plan, frames=True, rescued=True)

                context = build_context(
                    entry if plan["frames"] else None,
                    transcript if plan["transcript"] else None,
                    routed_out=not plan["frames"])

                try:
                    record = rank_query(client, query, query["Options"],
                                        context,
                                        frames if plan["frames"] else None,
                                        args.passes, plan)
                except Exception as exc:  # noqa: BLE001 - one bad query must
                    failed += 1          # not sink a long run
                    print(f"  FAILED Q_ID {query['Q_ID']}: "
                          f"{type(exc).__name__}: {exc}", file=sys.stderr)
                    continue

                mirror.write(record)
                cache[record["Q_ID"]] = record
                done += 1

                if done % 10 == 0 or done == len(todo):
                    rate = done / max(time.monotonic() - started, 1e-6)
                    left = (len(todo) - done) / rate if rate else 0
                    print(f"  {done}/{len(todo)} ranked, {failed} failed | "
                          f"{60 * rate:.1f}/min | ~{left / 3600:.1f} h left")

                if done % max(args.sync_every, 1) == 0:
                    run, _ = assemble(queries, cache)
                    publish(run, local_run, args.out)
    except KeyboardInterrupt:
        print("\ninterrupted -- flushing what is done", file=sys.stderr)
    finally:
        mirror.close()

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

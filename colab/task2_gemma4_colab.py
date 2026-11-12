#!/usr/bin/env python3
"""TREC 2026 VQA Task 2 (multiple choice) — Gemma 4 12B ranker, standalone for Colab.

A self-contained port of ``VQA/rank.options.mc.py``. Nothing is imported from
the project; copy this one file to Colab and run it.

Ranking contract, unchanged from the local pass
-----------------------------------------------
* **Rank option numbers, never option text.** The model returns a permutation
  of ``1..N`` which is mapped back to the official strings by index, so no rank
  is spent twice and no option goes unranked.
* **Permute the presentation.** Official order, exact reverse, then seeded
  shuffles, combined by a confidence-weighted Borda count. Position bias was
  measured at chance in every local run (rank 1 on the officially-first option
  15.2-17.4% against 14.9% expected).
* ``as_permutation()`` repairs repeats and drops; ``check_permutation()``
  refuses to write anything the official validator would reject.
* Seeds derive from the Q_ID, so a re-run reproduces the same run file.

Dev-set result this model earned (400 labelled 2025 queries, official scorer):
**0.733 Top-1 / 0.840 MRR**, -6.8 points against Qwen3.8 (p = 0.0005). It is
the cheapest credible ranker in the set (3.01 h for the full 2026 pass) and it
never failed: 0 failed passes, 0 repairs, 0 fallbacks, 79.7% unanimous.

Colab quickstart
----------------
Runtime → Change runtime type → **L4** or **A100**. A T4 will usually work for
this model (it is INT4, ~8.3 GiB of weights) but vLLM prefers SM80+; if the
engine refuses to start on a T4, that is why.

    from google.colab import drive; drive.mount('/content/drive')

    # one command does everything: deps, server, run, validate
    !python task2_gemma4_colab.py --install \
        --drive-root /content/drive/MyDrive/TREC_2026 \
        --hf-token hf_xxx

``google/gemma-4-12B-it-qat-w4a16-ct`` is a **gated** repo — accept the licence
on huggingface.co once, then pass ``--hf-token`` (or set ``HF_TOKEN``).

Expected Drive layout
---------------------
    TREC_2026/
      vqa.2026.testingDataset.Task2_v2.json      <- the official query file
      videos_TEST_VQA/
        frames/index.json                        <- from VQA/extract.frames.py
        frames/<video_id>/001.jpg ...
        transcription_openwhisper/<video_id>.json
        videos/                                  <- not needed by this script
      query.routing.task1.json                   <- optional, see --routing
      runs/                                      <- created; outputs land here

Only ``frames/`` and ``transcription_openwhisper/`` are read here, so you can
leave the 11.4 GiB ``videos/`` tree off Drive entirely for this model.

**Upload frames as a tar, not as a tree.** ``frames/`` is 26,251 files for
1.15 GiB, and Drive's FUSE mount is slow per-file. ``tar -cf frames.tar
frames`` locally, put the tar on Drive, and this script will find and unpack it
(``--stage auto``). Reading the tree directly works but costs ~20-40 min.

Resuming, which you will need
-----------------------------
The full pass is 2,443 queries x 3 passes and Colab will disconnect before some
models finish. Every completed query is appended to a JSONL cache immediately,
mirrored to Drive every ``--sync-every`` records, and re-reading that cache is
the first thing this script does. **Re-run the identical command after a
disconnect and it picks up where it stopped.** A partial run file is written at
every sync too, so a preempted session still leaves something scoreable.

To split deliberately across sessions use ``--shard 1/3``, ``--shard 2/3`` ...
Shards write to the same cache, so the final ``--assemble-only`` pass over all
of them emits one complete run.

Verification before you submit
------------------------------
``--validate`` (on by default) re-checks every invariant the official
``validate_vqa_mc_2026.py`` checks: exactly the official Q_IDs in the official
order, Video_IDs matching, ranks exactly ``1..N`` in ascending order, and the
answers being a permutation of the official option strings. It is not a
substitute for running the official validator on the finished file, but it
catches everything that has ever gone wrong here.
"""

from __future__ import annotations

import argparse
import base64
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

RUN_NAME = "gemma4"

# ---------------------------------------------------------------------------
# Model and serving defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"
DEFAULT_PORT = 8010

# Sized against the real worst case (eight long options plus sixteen frames),
# not against the average. Gemma 4 sees frames, never native video.
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_GPU_FRACTION = 0.90

# A ranking of eight integers plus a confidence is ~40 tokens. 128 leaves room
# for a stray restatement without letting a runaway reply cost real time.
DEFAULT_NUM_PREDICT = 128

# Colder than the Task 1 pass (0.6) on purpose: the diversity here is the
# option permutation, not sampling noise.
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64

DEFAULT_PASSES = 3
DEFAULT_WORKERS = 8
DEFAULT_TIMEOUT = 300.0
DEFAULT_RETRIES = 3

# ---------------------------------------------------------------------------
# The prompt, word for word from VQA/rank.options.mc.py
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
# Ranking contract
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
# Data loading
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
# Run validation -- the invariants validate_vqa_mc_2026.py enforces
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
# Colab plumbing: staging, cache mirroring, the server
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


class VLLMServer:
    """Starts vLLM in this VM, waits for it, and stops it on the way out."""

    def __init__(self, model: str, port: int, max_model_len: int,
                 gpu_fraction: float, log: Path, extra: list[str]) -> None:
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
            "--disable-log-requests",
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
            response = requests.get(self.url + "/v1/models", timeout=5)
            return response.status_code == 200
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
# The client
# ---------------------------------------------------------------------------

class VLLMClient:
    """OpenAI-compatible client. Frames become base64 image parts ahead of text."""

    name = "vllm"

    def __init__(self, base: str, model: str, num_predict: int,
                 temperature: float, top_p: float, top_k: int,
                 timeout: float, retries: int, think: bool) -> None:
        self.url = base.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.num_predict = num_predict
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.timeout = timeout
        self.retries = retries
        self.think = think
        self.local = threading.local()

    @property
    def session(self):
        import requests

        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()

        return self.local.session

    def ask(self, prompt: str, schema: dict, seed: int,
            images: list[str] | None) -> tuple[dict, float]:
        """One call. Returns the parsed reply and the wall time it took."""
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
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "ranking",
                                "schema": dict(schema,
                                               additionalProperties=False)},
            },
            "chat_template_kwargs": {"enable_thinking": self.think},
        }

        last: Exception | None = None

        for attempt in range(self.retries):
            started = time.monotonic()

            try:
                response = self.session.post(self.url, json=payload,
                                             timeout=self.timeout)

                # A 400 is the server refusing the request itself -- a prompt
                # past max-model-len, a schema it will not compile -- and it
                # will refuse it again just as fast.
                if response.status_code == 400:
                    raise RuntimeError(f"400: {response.text[:300]}")

                response.raise_for_status()
                choice = response.json()["choices"][0]
                text = (choice["message"].get("content") or "").strip()

                if not text:
                    raise ValueError(
                        f"empty content (finish_reason="
                        f"{choice.get('finish_reason')})")

                return parse_json_object(text), time.monotonic() - started

            except RuntimeError:
                raise
            except (requests.RequestException, ValueError, KeyError,
                    IndexError, json.JSONDecodeError) as exc:
                last = exc

                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"{type(last).__name__}: {last}")


def rank_query(client: VLLMClient, query: dict, options: list[str],
               context: str, images: list[str] | None, passes: int,
               plan: dict) -> dict:
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
            n=n,
            slots=slots,
            extra=extra,
        )

        try:
            reply, elapsed = client.ask(prompt, schema, seed_for(q_id, index),
                                        images)
        except RuntimeError as exc:
            failures += 1
            print(f"  Q_ID {q_id} pass {index}: {exc}", file=sys.stderr)
            continue

        positions, repaired = as_permutation(reply.get("ranking"), n)
        repairs += int(repaired)

        results.append({
            "index": index,
            # Presented positions map back to official option indices here,
            # once, so nothing downstream has to know the shuffle existed.
            "ranking": [order[position] for position in positions],
            "order": order,
            "confidence": reply.get("confidence"),
            "repaired": repaired,
            "elapsed": round(elapsed, 3),
        })

    if results:
        ranked = borda(results, n)
        fallback = False
    else:
        # Every pass failed: the options keep their official order, which is no
        # better than the random control for this query. Flagged, counted, and
        # re-runnable with --force --only.
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
        # The official index of each option, best first, plus its evidence --
        # everything a cross-model fusion needs without the option text.
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
        "n_frames": len(images) if images else 0,
        "has_transcript": "Spoken audio" in context,
        "confidence_mean": round(sum(confident) / len(confident), 4)
        if confident else None,
        # How many passes agreed on the option that won. 1.0 means unanimous;
        # 1/passes means they disagreed completely, which is the strongest
        # available signal that this query was a guess.
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
    """What the run looks like, in the terms the local passes were judged on."""
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

    print(f"\n{'-' * 68}\nranked          {n} / {len(queries)} queries")
    print(f"pass failures   {sum(r.get('n_pass_failures', 0) for r in records)}")
    print(f"repaired        {sum(r.get('n_repaired', 0) for r in records)}")
    print(f"fallbacks       {sum(1 for r in records if r.get('fallback'))}")
    print(f"no frames       {sum(1 for r in records if not r.get('n_frames'))}")
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
                       default=Path("/content/drive/MyDrive/TREC_2026"),
                       help="the copied project root on Drive")
    group.add_argument("--queries", type=Path, default=None,
                       help="official query JSON "
                            "(default: <drive-root>/vqa.2026.testingDataset.Task2_v2.json)")
    group.add_argument("--data-root", type=Path, default=None,
                       help="the videos_TEST_VQA copy "
                            "(default: <drive-root>/videos_TEST_VQA)")
    group.add_argument("--frames", type=Path, default=None,
                       help="frames directory (default: <data-root>/frames)")
    group.add_argument("--transcripts", type=Path, default=None,
                       help="default: <data-root>/transcription_openwhisper")
    group.add_argument("--routing", type=Path, default=None,
                       help="query.routing.task1.json. Absent means every "
                            "query gets frames AND transcript, which is what "
                            "--ignore-routing does. The local 2026 runs had "
                            "routing ON, so leaving it out changes 116 "
                            "queries")
    group.add_argument("--work", type=Path, default=Path("/content/trec_work"),
                       help="fast local scratch")
    group.add_argument("--out", type=Path, default=None,
                       help="run file (default: <drive-root>/runs/"
                            f"{RUN_NAME}.run.task2.json)")

    group = parser.add_argument_group("colab")
    group.add_argument("--install", action="store_true",
                       help="pip install the dependencies, then continue")
    group.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                       help="needed: this checkpoint is a gated repo")
    group.add_argument("--stage", choices=("auto", "tar", "copy", "drive"),
                       default="auto",
                       help="how to get the frames onto local disk")
    group.add_argument("--sync-every", type=int, default=50,
                       help="mirror the cache to Drive every N queries")
    group.add_argument("--serve", dest="serve", action="store_true",
                       default=True, help="start vLLM here (default)")
    group.add_argument("--no-serve", dest="serve", action="store_false",
                       help="attach to a server already running")
    group.add_argument("--base-url", default=None,
                       help="default: http://127.0.0.1:<port>")
    group.add_argument("--server-wait", type=float, default=1800.0,
                       help="seconds to wait for the model to load")
    group.add_argument("--vllm-arg", action="append", default=[],
                       help="extra argument passed through to vllm serve, "
                            "repeatable")

    group = parser.add_argument_group("the run")
    group.add_argument("--model", default=DEFAULT_MODEL)
    group.add_argument("--port", type=int, default=DEFAULT_PORT)
    group.add_argument("--max-model-len", type=int,
                       default=DEFAULT_MAX_MODEL_LEN)
    group.add_argument("--gpu-fraction", type=float,
                       default=DEFAULT_GPU_FRACTION)
    group.add_argument("--passes", type=int, default=DEFAULT_PASSES)
    group.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    group.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    group.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    group.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    group.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    group.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    group.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    group.add_argument("--think", action="store_true",
                       help="leave the thinking template on (it is off, and "
                            "the schema would suppress the trace anyway)")
    group.add_argument("--ignore-routing", action="store_true",
                       help="send every query frames and transcript")

    group = parser.add_argument_group("scope")
    group.add_argument("--limit", type=int, default=0,
                       help="rank at most N queries, for a smoke test")
    group.add_argument("--only", type=int, nargs="+", default=None,
                       help="rank only these Q_IDs")
    group.add_argument("--shard", default=None, metavar="I/N",
                       help="rank shard I of N (1-based), split by video so "
                            "the frame prefix cache still hits")
    group.add_argument("--force", action="store_true",
                       help="re-rank queries already in the cache")
    group.add_argument("--assemble-only", action="store_true",
                       help="write the run from the cache, no inference")
    group.add_argument("--no-validate", dest="validate", action="store_false",
                       default=True)

    return parser.parse_args(argv)


def pip_install() -> None:
    """Colab ships torch; vLLM brings its own pins. Keep this short and loud."""
    packages = ["vllm>=0.10", "requests", "pillow"]
    print(f"installing {', '.join(packages)} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           *packages])
    print("installed. If vLLM replaced torch, Colab may ask you to restart "
          "the runtime -- do it, then re-run this command without --install.")


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


def select(queries: list[dict], args: argparse.Namespace,
           cache: dict[int, dict]) -> list[dict]:
    """The queries this invocation will actually rank."""
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

        # Shard by video, not by query: the passes of one video share a frame
        # prefix, and splitting them across sessions throws that away.
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

    for path in (args.queries,):
        if not Path(path).exists():
            raise SystemExit(f"not found: {path}\n  Is Drive mounted, and is "
                             f"--drive-root right?")

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

    # One worker per video: the passes of a query, and the queries of a video,
    # share a frame prefix that the server prefills once.
    groups: dict[str, list[dict]] = {}

    for query in todo:
        groups.setdefault(query["Video_ID"], []).append(query)

    ordered = [groups[video] for video in sorted(groups)]
    blind = sum(len(g) for g in ordered
                if Path(g[0]["Video_ID"]).stem not in frame_index)

    print(f"to rank: {len(todo)} queries over {len(ordered)} videos "
          f"({blind} have no frames), {args.passes} passes each")

    server = VLLMServer(args.model, args.port, args.max_model_len,
                        args.gpu_fraction, args.work / "vllm.log",
                        args.vllm_arg)

    if args.serve:
        server.start(args.server_wait)
    elif not server.alive():
        raise SystemExit(f"--no-serve, but nothing answers at {args.base_url}")

    client = VLLMClient(args.base_url, args.model, args.num_predict,
                        args.temperature, args.top_p, args.top_k,
                        args.timeout, args.retries, args.think)

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
                images = [base64.b64encode((frame_dir / name).read_bytes())
                          .decode() for name in entry["files"]]
            except (OSError, KeyError, TypeError) as exc:
                print(f"  {video_id}: unreadable frames ({exc})",
                      file=sys.stderr)
                images, entry = None, None

        transcript = load_transcript(args.transcripts, video_id)
        has_speech = bool(transcript and transcript.get("text"))

        for query in group:
            plan = plan_for(routing.get(query["Q_ID"]), query["Question"],
                            args.ignore_routing)

            # Never send a prompt with no evidence at all: a query the router
            # sent to the audio of a clip that turns out to have no speech gets
            # its frames back.
            if not plan["frames"] and not has_speech:
                plan = dict(plan, frames=True, rescued=True)

            context = build_context(entry if plan["frames"] else None,
                                    transcript if plan["transcript"] else None,
                                    routed_out=not plan["frames"])

            try:
                record = rank_query(client, query, query["Options"], context,
                                    images if plan["frames"] else None,
                                    args.passes, plan)
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
                      "the same command)")

            return 1

        print("\nVALIDATION: 0 errors -- shape matches "
              "validate_vqa_mc_2026.py's rules")

    return 0


if __name__ == "__main__":
    sys.exit(main())

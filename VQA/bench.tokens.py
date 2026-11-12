#!/usr/bin/env python3
"""Prefill and decode throughput, in tokens/s, for every VLM in this project,
measured on real TREC 2026 queries.

The ranking passes report queries per minute, which folds prefill, decode,
vision encoding and client overhead into one number and so cannot say *why* one
model is slower than another. This measures the two halves separately:

    prefill tok/s = prompt_tokens / TTFT
    decode  tok/s = (completion_tokens - 1) / (t_last - t_first)

TTFT is the wall time from the request leaving this process to the first
content delta arriving, so for a VLM it contains the vision encoder as well as
the language prefill -- the honest number for these workloads, since no request
here is text-only in production. ``--mode none`` strips the pixels and keeps
everything else, so the difference between the two is what the pixels cost. The
first decoded token is excluded from the decode window because its time is
already inside TTFT.

**The prompt is the real one, for either task.** Not filler: the released
query, its routed context block and its transcript, built by importing the
pass that owns the task and calling the same ``build_context()``,
``PROMPT_TEMPLATE`` and rules the runs used. A throughput number from a prompt
nobody sends is a number about nothing.

    ``--task 1``   answer generation, from ``generate.answers.py``: the
                   question, and a request for the ten answers the run format
                   needs, in one reply. ``--answers`` changes the count.
    ``--task 2``   multiple-choice ranking, from ``rank.options.mc.py``: the
                   numbered options and a request to rank them. One
                   presentation per query (the official order, pass 0 of the
                   three a real run asks) -- the other two differ only in
                   option order and cost the same.

Both tasks share Q_IDs and videos, so the same ten clips are measured either
way and the two tables are directly comparable. Task 1 decodes ten answers
where task 2 decodes a permutation, so a task 1 request gets at least 256
tokens of budget -- what ``generate.answers.py`` itself uses.

**Ten videos spanning the duration range.** Prompt length on these models is
mostly pixels, and pixels scale with duration, so a one-clip bench measures one
point on a curve. ``--spread duration`` (the default) walks a geometric series
from the shortest clip to the longest and takes the nearest video to each step,
covering 1 s to 389 s -- the sample to read a scaling curve off.
``--spread corpus`` instead reproduces the corpus's own distribution (median
10 s, p90 56 s, p99 233 s), which is short-clip heavy and predicts what a full
run costs. Either way the chosen set is deterministic and is written into the
results, so every model sees the same ten.

**Every request defeats the prefix cache.** vLLM and llama.cpp both prefill a
repeated prompt from cache, which would report a prefill rate ten to a hundred
times the real one -- ``rank.options.mc.qwen3vl.py`` measured ~70 s for a cold
video prefill against 1.8-3 s for the same prefix warm. A per-request nonce
goes in the *system* message, ahead of the images, so the whole prefix is new.
``--cached`` turns that off to measure the warm path deliberately, and
llama.cpp's own ``cache_n`` is recorded so a silent hit is visible.

**Decode runs to its natural end.** The reply is kept, so ``ignore_eos`` is
off by default: what a model emits on a real query -- 40 tokens of JSON from
Qwen, a few thousand of reasoning from Cosmos and Muse -- is part of what is
being measured. ``--force-length N`` pins every completion to N tokens instead,
which makes the decode rates strictly comparable at the cost of a junk reply.
A decode rate from a very short completion is noisy, so the summary marks any
model whose median completion is under 20 tokens.

**One model fits on the 3090 at a time.** This script is a client: it does not
start servers. ``--list`` prints every model with its port and serve command,
``--serve-cmd NAME`` prints just the command, and
``VQA/run.bench.tokens.ps1`` drives the whole sweep one server at a time.

VideoLLaMA3 has no server -- it runs in-process from ``.venv-videollama3`` --
so it is measured two-point: a generate at ``max_new_tokens=1`` is the prefill,
and the slope to a full generate is the decode rate. ``--method two-point``
does the same over HTTP when a server's stream needs cross-checking.

Usage (Windows: llama.cpp, Ollama, in-process)::

    E:/TREC_2026/.venv/Scripts/python.exe VQA/bench.tokens.py --model muse --task 1
    ... --model internvl --task 1 --videos 10
    ... --dry-run                 # print the chosen clips and the exact prompt
    E:/TREC_2026/.venv-videollama3/Scripts/python.exe VQA/bench.tokens.py --model videollama3

Usage (WSL2: every vLLM model -- the server reads the clip off /mnt/e)::

    wsl -d Ubuntu-24.04 -- bash /mnt/e/TREC_2026/VQA/bench.tokens.sh qwen3vl --task 1

    ... --concurrency 4        # aggregate throughput at the run's worker count
    ... --mode none            # same prompt, no pixels, for the language-only rate
    ... --sweep                # no pixels, then pixels, so the gap is visible

Three outputs, all at the project root:

    bench.tokens.jsonl   one record per (model, workload): config, the chosen
                         queries, every sample, and every raw reply
    bench.tokens.csv     one row per request, for a spreadsheet
    --report             the cross-model table, read back from the JSONL
"""

from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import json
import platform
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DEFAULT_QUERIES = {1: ROOT / "vqa.2026.testingDataset.Task1.json",
                   2: ROOT / "vqa.2026.testingDataset.Task2_v2.json"}
DEFAULT_ROUTING = ROOT / "query.routing.task1.json"
DEFAULT_FRAMES = ROOT / "videos_TEST_VQA" / "frames"
DEFAULT_VIDEOS = ROOT / "videos_TEST_VQA" / "videos"
DEFAULT_TRANSCRIPTS = ROOT / "videos_TEST_VQA" / "transcription_openwhisper"
DEFAULT_OUT = ROOT / "bench.tokens.jsonl"
DEFAULT_CSV = ROOT / "bench.tokens.csv"

# Qwen3-VL and Qwen3.8 share this vision geometry: patch 16, temporal patch 2,
# spatial merge 2, so 1,024 pixels per visual token, and `size.longest_edge` is
# a WHOLE-VIDEO pixel budget rather than a per-frame ceiling. Copied from
# rank.options.mc.qwen3vl.py so the bench sends the budget the runs sent.
PIXELS_PER_TOKEN = 1024
TEMPORAL_MERGE = 2
MIN_FRAME_TOKENS = 128
VISUAL_TARGET_TOKENS = 4096

# Task 1 asks for ten answers in one reply, which generate.answers.py budgets
# at 256 tokens -- more than the 128 a Task 2 permutation needs. A reasoning
# model keeps its own larger budget.
TASK1_MIN_TOKENS = 256
DEFAULT_ANSWERS = 10

DEFAULT_VIDEO_COUNT = 10
DEFAULT_FRAME_COUNT = 16
DEFAULT_REPEATS = 1
DEFAULT_WARMUP = 1
DEFAULT_TIMEOUT = 1800
DEFAULT_SEED = 2026

# Below this, a decode rate is measured over too few tokens to trust.
SHORT_COMPLETION = 20

# rank.options.mc.cosmos.py, imported on first use for its ffmpeg guard.
_COSMOS_PASS = None


def qwen_visual_kwargs(target: int = VISUAL_TARGET_TOKENS) -> dict:
    """The whole-video pixel budget the Qwen passes send, as mm_processor_kwargs.

    No ``do_sample_frames``/``fps``: the servers run with
    ``--media-io-kwargs '{"video": {"num_frames": -1}}'``, and asking the
    processor to re-sample is rejected for 91% of this corpus's clips on
    Qwen3.8. Leaving it out keeps frame selection at the processor default for
    both models, which is what a throughput number should hold constant.
    """
    return {"size": {"shortest_edge": MIN_FRAME_TOKENS * PIXELS_PER_TOKEN,
                     "longest_edge": target * TEMPORAL_MERGE * PIXELS_PER_TOKEN}}


# Every model this project has run, with the command that serves it. `where` is
# where THIS script must run: a server that opens a clip off a /mnt/e path can
# only be driven from WSL2. `api` decides which non-standard fields are safe to
# send. `max_tokens` is that pass's own generation budget, so a reasoning model
# is given the room its real run gives it.
MODELS: dict[str, dict] = {
    "cosmos": {
        "model": "nvidia/Cosmos-Reason2-2B",
        "host": "http://localhost:8008",
        "api": "vllm",
        "payload": "video",
        "where": "wsl",
        "max_tokens": 4096,
        "notes": "reasoning parser qwen3: the think block arrives as "
                 "reasoning_content deltas, counted here as decoded tokens "
                 "because that is what they are, and they are most of its cost",
        "serve": "systemctl --user start cosmos-reason2   # or, by hand:\n"
                 "/home/aakash/cosmos-reason2/.venv/bin/vllm serve nvidia/Cosmos-Reason2-2B \\\n"
                 "  --host 0.0.0.0 --port 8008 \\\n"
                 "  --allowed-local-media-path /mnt/e/TREC_2026 \\\n"
                 "  --max-model-len 24576 --gpu-memory-utilization 0.75 --enforce-eager \\\n"
                 "  --media-io-kwargs '{\"video\": {\"num_frames\": -1}}' \\\n"
                 "  --reasoning-parser qwen3",
    },
    "qwen3vl": {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "host": "http://localhost:8009",
        "api": "vllm",
        "payload": "video",
        "where": "wsl",
        "max_tokens": 256,
        "extra_body": {"mm_processor_kwargs": qwen_visual_kwargs()},
        "notes": "4,096 whole-clip visual tokens, as the Task 2 run sent",
        "serve": "/home/aakash/cosmos-reason2/.venv/bin/vllm serve Qwen/Qwen3-VL-8B-Instruct \\\n"
                 "  --host 0.0.0.0 --port 8009 \\\n"
                 "  --served-model-name Qwen/Qwen3-VL-8B-Instruct \\\n"
                 "  --allowed-local-media-path /mnt/e/TREC_2026 \\\n"
                 "  --max-model-len 24576 --gpu-memory-utilization 0.92 --enforce-eager \\\n"
                 "  --media-io-kwargs '{\"video\": {\"num_frames\": -1}}' \\\n"
                 "  --skip-mm-profiling",
    },
    "gemma4": {
        "model": "google/gemma-4-12B-it-qat-w4a16-ct",
        "host": "http://localhost:8010",
        "api": "vllm",
        "payload": "frames",
        "where": "either",
        "max_tokens": 128,
        "notes": "QAT INT4; frames go in as base64 image_url parts",
        "serve": "VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 \\\n"
                 "/home/aakash/gemma4/.venv/bin/vllm serve google/gemma-4-12B-it-qat-w4a16-ct \\\n"
                 "  --host 0.0.0.0 --port 8010 \\\n"
                 "  --served-model-name google/gemma-4-12B-it-qat-w4a16-ct \\\n"
                 "  --allowed-local-media-path /mnt/e/TREC_2026 \\\n"
                 "  --max-model-len 16384 --gpu-memory-utilization 0.92 --enforce-eager \\\n"
                 "  --limit-mm-per-prompt '{\"image\": 32}' --reasoning-parser gemma4",
    },
    "gemma4-ollama": {
        "model": "gemma4:12b",
        "host": "http://localhost:11434",
        "api": "ollama",
        "payload": "frames",
        "where": "windows",
        "max_tokens": 128,
        # Ollama defaults to a 4,096-token window whatever the model's is,
        # which is less than the frames alone.
        "extra_body": {"options": {"num_ctx": 16384}},
        "notes": "Q4_K_M, a different quantisation from the vLLM service; no "
                 "ignore_eos, so --force-length cannot pin its decode window",
        "serve": "ollama serve   # then: ollama pull gemma4:12b",
    },
    "qwen38": {
        "model": "RedHatAI/Qwen3.8-27B-INT4",
        "host": "http://localhost:8011",
        "api": "vllm",
        "payload": "video",
        "where": "wsl",
        "max_tokens": 256,
        "extra_body": {"mm_processor_kwargs": qwen_visual_kwargs()},
        "notes": "vLLM 0.27 venv only; 48 of its 64 layers are linear attention",
        "serve": "VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 \\\n"
                 "/home/aakash/gemma4/.venv/bin/vllm serve RedHatAI/Qwen3.8-27B-INT4 \\\n"
                 "  --host 0.0.0.0 --port 8011 \\\n"
                 "  --allowed-local-media-path /mnt/e/TREC_2026 \\\n"
                 "  --max-model-len 16384 --gpu-memory-utilization 0.93 \\\n"
                 "  --enforce-eager --skip-mm-profiling \\\n"
                 "  --media-io-kwargs '{\"video\": {\"num_frames\": -1}}'",
    },
    "llavavideo": {
        "model": "llava-video-7b",
        "host": "http://localhost:8009",
        "api": "vllm",
        "payload": "video",
        "where": "wsl",
        "max_tokens": 128,
        "notes": "the server samples 32 frames itself (--media-io-kwargs)",
        "serve": "bash /mnt/e/TREC_2026/VQA/serve.llavavideo.sh   # or, by hand:\n"
                 "/home/aakash/cosmos-reason2/.venv/bin/vllm serve Isotr0py/LLaVA-Video-7B-Qwen2-hf \\\n"
                 "  --host 0.0.0.0 --port 8009 --served-model-name llava-video-7b \\\n"
                 "  --allowed-local-media-path /mnt/e/TREC_2026 \\\n"
                 "  --max-model-len 16384 --gpu-memory-utilization 0.92 --enforce-eager \\\n"
                 "  --limit-mm-per-prompt '{\"video\": 1}' \\\n"
                 "  --media-io-kwargs '{\"video\": {\"num_frames\": 32}}' --skip-mm-profiling",
    },
    "muse": {
        "model": "muse-glimmer-30b",
        "host": "http://localhost:8012",
        "api": "llamacpp",
        "payload": "frames",
        "where": "windows",
        "max_tokens": 2048,
        # The chat template falls back to `high` when this is absent, which
        # would spend thousands of tokens on a permutation of eight integers.
        "extra_body": {"chat_template_kwargs": {"reasoning_strength": "low"}},
        "notes": "Q4_K_XL GGUF + BF16 mmproj; serve with --image-max-tokens 256",
        "serve": "E:/TREC_2026/tools/llama-b10976/llama-server.exe `\n"
                 "  -m E:/TREC_2026/models/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-UD-Q4_K_XL.gguf `\n"
                 "  --mmproj E:/TREC_2026/models/Muse-Glimmer-30B-GGUF/mmproj-Muse-Glimmer-30B-BF16.gguf `\n"
                 "  --jinja --reasoning-format auto -ngl all -fa on `\n"
                 "  -c 49152 -np 3 --image-max-tokens 256 `\n"
                 "  --alias muse-glimmer-30b --port 8012 --no-webui",
    },
    "internvl": {
        "model": "internvl3.5-30b-a3b",
        "host": "http://localhost:8013",
        "api": "llamacpp",
        "payload": "frames",
        "where": "windows",
        "max_tokens": 256,
        # llama.cpp tiles every InternVL image into 448 px crops and
        # --image-max-tokens does not cap it; one tile per frame is what
        # InternVL's own video path uses, and what the dev pass sent.
        "frame_size": 448,
        "notes": "MoE, 3B active; frames pre-resized to 448x448, one tile each",
        "serve": "E:/TREC_2026/tools/llama-b10976/llama-server.exe `\n"
                 "  -m E:/TREC_2026/models/InternVL3_5-30B-A3B-GGUF/OpenGVLab_InternVL3_5-30B-A3B-Q4_K_M.gguf `\n"
                 "  --mmproj E:/TREC_2026/models/InternVL3_5-30B-A3B-GGUF/mmproj-OpenGVLab_InternVL3_5-30B-A3B-f16.gguf `\n"
                 "  --jinja -ngl all -fa on -c 32768 -np 2 `\n"
                 "  --cache-type-k q8_0 --cache-type-v q8_0 `\n"
                 "  --alias internvl3.5-30b-a3b --port 8013 --no-webui",
    },
    "videollama3": {
        "model": "DAMO-NLP-SG/VideoLLaMA3-7B",
        "host": None,
        "api": "transformers",
        "payload": "frames",
        "where": "windows",
        "max_tokens": 128,
        "path": ROOT / "models" / "VideoLLaMA3-7B",
        "notes": "in-process bf16, sdpa vision attention patched block-wise; "
                 "measured two-point, not streamed",
        "serve": "# No server. Run the bench with the model's own venv:\n"
                 "E:/TREC_2026/.venv-videollama3/Scripts/python.exe "
                 "VQA/bench.tokens.py --model videollama3",
    },
}


# --------------------------------------------------------------------------
# The real prompt, borrowed from the ranking pass
# --------------------------------------------------------------------------

def import_pass(task: int):
    """The pass that owns this task's prompt, as a module.

    Task 1 is answer generation (``generate.answers.py``), task 2 the
    multiple-choice ranking (``rank.options.mc.py``). Imported rather than
    copied so the bench cannot drift from the run: the system prompt, the
    context block, the rules and the JSON contract are the same objects those
    passes used. Both import stdlib plus requests at module level, so this is
    cheap and side-effect free -- their main() is behind the usual guard.
    """
    name = {1: "generate.answers.py", 2: "rank.options.mc.py"}[task]
    source = HERE / name

    if not source.exists():
        raise SystemExit(f"{source} is missing -- the bench builds its task "
                         f"{task} prompt from it")

    spec = importlib.util.spec_from_file_location(f"task{task}_pass", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def build_prompt(module, task: int, query: dict, frames: dict | None,
                 transcript: dict | None, plan: dict, answers: int) -> str:
    """The user prompt for one query, exactly as that task's pass builds it.

    Task 2 sends pass 0 of the three a real run asks -- the official option
    order. The other two presentations differ only in the order of the option
    lines, so they cost the same prompt tokens and would measure the same
    thing.

    Task 1 sends the released question itself, asking for `answers` candidates
    in one reply. The real pass also sends query *variations* of the same
    question, but each is a separate call with the same frames and the same
    shape, so one of them is the unit to measure.
    """
    context = module.build_context(frames, transcript,
                                   routed_out=not plan["frames"])
    extra = ""

    if plan["ocr"]:
        extra += module.OCR_RULE

    if plan["temporal"]:
        extra += module.TEMPORAL_RULE

    if task == 1:
        return module.PROMPT_TEMPLATE.format(
            context=context,
            question=query["Question"],
            k=answers,
            slots=", ".join(['"..."'] * answers),
            extra=extra,
        )

    options = query["Options"]
    n = len(options)
    order = module.presentation_orders(n, 1, query["Q_ID"])[0]

    return module.PROMPT_TEMPLATE.format(
        context=context,
        question=query["Question"],
        options=module.render_options(options, order),
        n=n,
        slots=", ".join(str(i) for i in range(1, n + 1)),
        extra=extra,
    )


def load_task_queries(module, task: int, path: Path) -> list[dict]:
    """The released queries for this task.

    Task 2's loader flattens ``Options`` to a list of strings; task 1 has no
    options at all, so its file is read as it stands.
    """
    if task == 2:
        return module.load_queries(path)

    return [{"Q_ID": query["Q_ID"], "Video_ID": query["Video_ID"],
             "Question": query["Question"]}
            for query in module.load_json(path)]


def plan_for(module, task: int, routing: dict | None, question: str,
             ignore_routing: bool) -> dict:
    """That task's routing verdict for one query.

    The two passes disagree on the signature: the Task 2 ranker also checks the
    routed question against the released one, because the routing file was
    produced for Task 1 and a verdict about a different question is not a
    verdict about this one.
    """
    if task == 2:
        return module.plan_for(routing, question, ignore_routing)

    return module.plan_for(routing, ignore_routing)


# --------------------------------------------------------------------------
# Choosing the ten
# --------------------------------------------------------------------------

def load_frame_index(directory: Path) -> dict:
    """The frame index written by VQA/extract.frames.py."""
    path = directory / "index.json"

    if not path.exists():
        raise SystemExit(f"no frame index at {path} -- run VQA/extract.frames.py")

    return json.loads(path.read_text(encoding="utf-8"))


def resolve_video(videos_dir: Path, video_id: str) -> Path | None:
    """The clip file for a Video_ID, or None.

    A Video_ID is not a filename: 868 of the 2,443 released queries name a bare
    YouTube id stored as ``<id>.mp4`` beside the ``.webm`` clips. Resolving it
    literally is what left 45% of the first Cosmos run blind.
    """
    direct = videos_dir / video_id

    if direct.exists() and direct.is_file():
        return direct

    for suffix in (".mp4", ".webm", ".mkv"):
        candidate = videos_dir / f"{video_id}{suffix}"

        if candidate.exists():
            return candidate

    stem = Path(video_id).stem

    for suffix in (".mp4", ".webm", ".mkv"):
        candidate = videos_dir / f"{stem}{suffix}"

        if candidate.exists():
            return candidate

    return None


def select_queries(queries: list[dict], index: dict, videos_dir: Path,
                   wanted: int, seed: int, needs_clip: bool,
                   spread: str) -> list[dict]:
    """`wanted` queries whose clips span the duration range, two ways.

    Prompt length on these models is mostly pixels and pixels scale with
    duration, so sampling uniformly at random would land most of the draws near
    the median and measure one point on a curve.

    ``duration`` (the default) walks a geometric series from the shortest clip
    to the longest and takes the nearest unused video to each step, so the ten
    span 1 s to 389 s roughly evenly in log time. That is the sample to read a
    scaling curve off.

    ``corpus`` cuts the duration-sorted pool into `wanted` buckets of equal
    count and draws one from each with a seeded RNG, which reproduces the
    corpus's own distribution -- median 10 s, p90 56 s, p99 233 s -- and so
    predicts what a full run costs. It is deliberately short-clip heavy,
    because the corpus is.

    One query per video either way, so no two draws share a frame prefix.
    """
    frames = index["videos"]
    pool: dict[str, dict] = {}

    for query in queries:
        stem = Path(query["Video_ID"]).stem
        entry = frames.get(stem)

        if not entry or not entry.get("duration_sec"):
            continue

        if needs_clip and resolve_video(videos_dir, query["Video_ID"]) is None:
            continue

        # One query per video: two queries on the same clip share an image
        # prefix, which the server would prefill once and charge to whichever
        # request arrived second.
        pool.setdefault(stem, {"query": query, "frames": entry})

    if len(pool) < wanted:
        raise SystemExit(f"only {len(pool)} usable videos, {wanted} asked for")

    ordered = sorted(pool.values(),
                     key=lambda item: item["frames"]["duration_sec"])
    chosen: list[dict] = []

    if spread == "duration":
        shortest = ordered[0]["frames"]["duration_sec"]
        longest = ordered[-1]["frames"]["duration_sec"]
        taken: set[int] = set()

        for i in range(wanted):
            # Geometric, not linear: linear steps over 1-389 s would put nine
            # of the ten targets above 40 s, where the corpus has 200 videos.
            target = shortest * (longest / shortest) ** (i / max(wanted - 1, 1))
            best = min(
                (j for j in range(len(ordered)) if j not in taken),
                key=lambda j: abs(ordered[j]["frames"]["duration_sec"] - target))
            taken.add(best)
            chosen.append(ordered[best])

        return sorted(chosen, key=lambda item: item["frames"]["duration_sec"])

    rng = random.Random(seed)
    size = len(ordered) / wanted

    for i in range(wanted):
        start = int(i * size)
        stop = max(int((i + 1) * size), start + 1)
        chosen.append(ordered[rng.randrange(start, min(stop, len(ordered)))])

    return chosen


# --------------------------------------------------------------------------
# Request construction
# --------------------------------------------------------------------------

def encode_frame(path: Path, size: int | None) -> str:
    """One frame as base64 JPEG, optionally resized to size x size.

    Same helper as rank.options.mc.muse.py: InternVL needs 448 so llama.cpp
    makes one tile of each frame instead of five.
    """
    if size is None:
        return base64.b64encode(path.read_bytes()).decode()

    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC) \
        .save(buffer, format="JPEG", quality=90)

    return base64.b64encode(buffer.getvalue()).decode()


def safe_video_path(path: Path, duration_sec: float | None) -> Path:
    """A clip the server can decode without exhausting RAM.

    The Cosmos, Qwen3-VL and Qwen3.8 services all run with
    ``--media-io-kwargs '{"video": {"num_frames": -1}}'``, so the processor
    decodes *every* frame the file holds. On the 389 s 1080p clip in this
    sample that is 9,725 frames -- a 56.3 GiB array, and a 500 from the server.
    Measured, not predicted: it is what killed the first Cosmos bench.

    The ranking passes have carried this guard from the start; the bench was
    written without it and has to have it too, or the long end of the duration
    range is unmeasurable. ``ensure_safe_video()`` is imported from the Cosmos
    pass rather than reimplemented, so the bench transcodes exactly what the
    run transcodes, at the same adaptive fps, into the same on-disk cache --
    which also means a clip already shrunk for a real run is reused here.
    """
    global _COSMOS_PASS

    if _COSMOS_PASS is None:
        source = HERE / "rank.options.mc.cosmos.py"
        spec = importlib.util.spec_from_file_location("cosmos_pass", source)
        _COSMOS_PASS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_COSMOS_PASS)

    return _COSMOS_PASS.ensure_safe_video(
        path,
        _COSMOS_PASS.adaptive_fps(duration_sec),
        path.parent / _COSMOS_PASS.TRANSCODE_CACHE_DIRNAME,
    )


def generation_budget(spec: dict, args: argparse.Namespace) -> int:
    """How many tokens this request may decode.

    ``--force-length`` wins, then ``--max-tokens``, then the model's own pass
    budget. Task 1 wants ten answers in one reply where task 2 wants a
    permutation of eight integers, so a 128-token budget that is generous for
    the ranker truncates the answerer -- hence the floor, which is what
    generate.answers.py itself uses. A reasoning model's larger budget is left
    alone.
    """
    if args.force_length:
        return args.force_length

    if args.max_tokens:
        return args.max_tokens

    budget = spec["max_tokens"]

    return max(budget, TASK1_MIN_TOKENS) if args.task == 1 else budget


def query_shape(item: dict, mode: str, args: argparse.Namespace) -> dict:
    """What a sample records about the query it measured."""
    query = item["query"]
    entry = item["frames"]

    return {"task": args.task,
            "q_id": query["Q_ID"], "video_id": query["Video_ID"],
            "duration_sec": entry["duration_sec"],
            "n_options": len(query.get("Options") or ()),
            "n_frames": 0 if mode == "none"
            else min(args.frames, entry["n_frames"]),
            "mode": mode}


def build_content(spec: dict, item: dict, prompt: str, mode: str,
                  args: argparse.Namespace) -> tuple[list[dict], dict]:
    """The user content parts for one query, plus what the workload was.

    Three shapes, matching what each pass actually sends: pre-extracted JPEG
    frames as ``image_url`` parts (Gemma 4, Muse, InternVL, VideoLLaMA3), one
    ``video_url`` file path the server decodes itself (Cosmos, Qwen3-VL,
    Qwen3.8, LLaVA-Video), or the same prompt with no pixels at all.
    """
    entry = item["frames"]
    query = item["query"]
    shape = query_shape(item, mode, args)

    if mode == "none":
        return [{"type": "text", "text": prompt}], shape

    if mode == "frames":
        # The index is keyed by the stem but keeps the extension in
        # video_id; the frame directory is named by the stem.
        directory = args.frames_dir / Path(entry["video_id"]).stem
        size = args.frame_size if args.frame_size is not None \
            else spec.get("frame_size")
        names = entry["files"][:args.frames]
        parts = [{"type": "image_url",
                  "image_url": {"url": "data:image/jpeg;base64,"
                                       + encode_frame(directory / name, size)}}
                 for name in names]
        parts.append({"type": "text", "text": prompt})
        shape["frame_size"] = size

        return parts, shape

    if mode == "video":
        path = resolve_video(args.videos_dir, query["Video_ID"])

        if path is None:
            raise SystemExit(f"no clip for {query['Video_ID']!r}")

        if platform.system() == "Windows":
            raise SystemExit(
                "a video payload is a file:// path the server opens itself, "
                "so this must run inside WSL2 against /mnt/e/TREC_2026. "
                "See --list for the invocation.")

        if not args.raw_video:
            path = safe_video_path(path, entry.get("duration_sec"))

        shape["clip"] = path.name

        return ([{"type": "video_url",
                  "video_url": {"url": path.resolve().as_uri()}},
                 {"type": "text", "text": prompt}], shape)

    raise SystemExit(f"unknown mode {mode!r}")


def build_body(spec: dict, content: list[dict], module,
               args: argparse.Namespace, nonce: str, max_tokens: int,
               stream: bool) -> dict:
    """One chat-completions request.

    The nonce rides in the system message so it precedes the images: a nonce
    after them leaves the expensive half of the prefix cacheable, which is the
    half being measured.
    """
    system = module.SYSTEM_PROMPT

    if not args.cached:
        system = f"{system} [{nonce}]"

    body: dict = {
        "model": spec["model"],
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "stream": stream,
    }

    if stream:
        # vLLM, llama.cpp and Ollama all answer this with a final usage-only
        # chunk; without it the token counts have to be guessed from the
        # number of deltas.
        body["stream_options"] = {"include_usage": True}

    if args.force_length and spec["api"] in ("vllm", "llamacpp"):
        # Pins every completion to the same length, so decode rates compare
        # exactly -- at the cost of a reply that runs on past its JSON.
        body["ignore_eos"] = True

    for key, value in (spec.get("extra_body") or {}).items():
        body[key] = value

    return body


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def stream_events(response):
    """Parsed SSE payloads from a streaming chat-completions response.

    ``chunk_size=1`` is load-bearing. requests' default is 512 bytes, so it
    blocks until half a kilobyte of SSE has accumulated -- about a dozen token
    deltas -- before yielding the first line. That does not slow the server
    down; it moves the timestamps, inflating TTFT and shortening the decode
    window, which is exactly the pair being measured. Against a mock server
    with a known 0.5 s prefill the default reported 2.56 s. Reading a byte at a
    time costs a few milliseconds over a whole completion, the cheaper error.
    """
    for raw in response.iter_lines(chunk_size=1, decode_unicode=False):
        if not raw or not raw.startswith(b"data:"):
            continue

        payload = raw[5:].strip()

        if payload == b"[DONE]":
            # Not `return`: abandoning the response mid-body makes urllib3 drop
            # the connection, so the next request pays a fresh TCP connect
            # *inside* its TTFT. Reading to the end keeps the socket pooled.
            continue

        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def measure_stream(session, spec: dict, body: dict, timeout: int) -> dict:
    """TTFT, decode rate and the reply for one streamed request."""
    url = spec["host"].rstrip("/") + "/v1/chat/completions"
    started = time.perf_counter()
    response = session.post(url, json=body, stream=True, timeout=timeout)

    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")

    first = None
    last = started
    deltas = 0
    usage = None
    timings = None
    finish = None
    content: list[str] = []
    reasoning: list[str] = []

    for event in stream_events(response):
        if event.get("usage"):
            usage = event["usage"]

        # llama.cpp reports its own prompt_ms / predicted_ms, the server's view
        # of the same two numbers -- a cross-check, and the only way to see a
        # prefix-cache hit (cache_n).
        if event.get("timings"):
            timings = event["timings"]

        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            finish = choice.get("finish_reason") or finish
            text = delta.get("content") or ""
            think = delta.get("reasoning_content") or ""

            if text or think:
                now = time.perf_counter()

                if first is None:
                    first = now

                last = now
                deltas += 1
                content.append(text)
                reasoning.append(think)

    end = time.perf_counter()

    if first is None:
        raise RuntimeError("the stream carried no content -- check max_tokens "
                           "and the server's reasoning parser")

    prompt_tokens = (usage or {}).get("prompt_tokens")
    completion = (usage or {}).get("completion_tokens") or deltas
    ttft = first - started
    window = last - first

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "exact_counts": usage is not None,
        "finish_reason": finish,
        "ttft_s": round(ttft, 4),
        "decode_s": round(window, 4),
        "total_s": round(end - started, 4),
        "prefill_tps": round(prompt_tokens / ttft, 1) if prompt_tokens else None,
        "decode_tps": round((completion - 1) / window, 2)
        if completion > 1 and window > 0 else None,
        "server_prefill_tps": round(timings["prompt_per_second"], 1)
        if timings and timings.get("prompt_per_second") else None,
        "server_decode_tps": round(timings["predicted_per_second"], 2)
        if timings and timings.get("predicted_per_second") else None,
        "cached_prompt_tokens": (timings or {}).get("cache_n"),
        "response": "".join(content).strip(),
        "reasoning": "".join(reasoning).strip(),
    }


def measure_two_point(session, spec: dict, body: dict, timeout: int,
                      gen: int) -> dict:
    """Prefill and decode from two non-streamed calls, for a server whose
    stream cannot be trusted to mark the first token.

    Call one asks for a single token, so its wall time is prefill plus one
    decode step; call two asks for `gen`. The slope between them is the decode
    rate, with the per-request overhead cancelling out of it.
    """
    url = spec["host"].rstrip("/") + "/v1/chat/completions"

    def once(max_tokens: int) -> tuple[float, dict]:
        payload = dict(body, max_tokens=max_tokens, stream=False)
        payload.pop("stream_options", None)
        started = time.perf_counter()
        response = session.post(url, json=payload, timeout=timeout)

        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: "
                               f"{response.text[:300]}")

        return time.perf_counter() - started, response.json()

    t_one, one = once(1)
    t_full, full = once(gen)
    choice = (full.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    prompt_tokens = one.get("usage", {}).get("prompt_tokens")
    completion = full.get("usage", {}).get("completion_tokens") or gen
    window = t_full - t_one

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "exact_counts": "usage" in full,
        "finish_reason": choice.get("finish_reason"),
        "ttft_s": round(t_one, 4),
        "decode_s": round(window, 4),
        "total_s": round(t_full, 4),
        "prefill_tps": round(prompt_tokens / t_one, 1) if prompt_tokens else None,
        "decode_tps": round((completion - 1) / window, 2)
        if completion > 1 and window > 0 else None,
        "method": "two-point",
        "response": (message.get("content") or "").strip(),
        "reasoning": (message.get("reasoning_content") or "").strip(),
    }


def preflight(session, spec: dict, timeout: int = 20) -> str:
    """Refuse to start unless a server is up, and say what it is serving.

    A dead server is the failure this project has been bitten by before: the
    2025 answering pass padded every slot from its fallback pool and produced a
    valid-looking run file that meant nothing.
    """
    url = spec["host"].rstrip("/") + "/v1/models"

    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        raise SystemExit(f"no server answering at {spec['host']} ({exc}).\n"
                         f"Start it with:\n\n{spec['serve']}\n")

    served = [item["id"] for item in response.json().get("data", [])]

    if served and spec["model"] not in served:
        print(f"  note: {spec['host']} serves {served[0]!r}, not "
              f"{spec['model']!r} -- using the served name")

        return served[0]

    return spec["model"]


def pin_ipv4(host: str) -> str:
    """Rewrite a ``localhost`` host to 127.0.0.1.

    Measured here: against a server with a known 0.5 s prefill, a bench aimed
    at ``http://localhost:8099`` reported a 2.56 s TTFT and 482 prefill tok/s,
    and the same bench aimed at ``http://127.0.0.1:8099`` reported 0.51 s and
    2,410 tok/s -- a five-fold error. Python resolves ``localhost`` to ``::1``
    first, and every server in this project binds IPv4 only (llama.cpp and
    Ollama to 127.0.0.1, vLLM to 0.0.0.0, which is not a v6 address), so each
    connection spends ~2 s failing over. That cost lands in TTFT, which is the
    number this script exists to report.
    """
    parts = urlsplit(host)

    if parts.hostname != "localhost":
        return host

    netloc = "127.0.0.1" + (f":{parts.port}" if parts.port else "")

    return urlunsplit(parts._replace(netloc=netloc))


# --------------------------------------------------------------------------
# In-process (VideoLLaMA3)
# --------------------------------------------------------------------------

def load_videollama3(spec: dict, args: argparse.Namespace):
    """The ranking pass's own loader, reused.

    Importing rank.options.mc.videollama3 rather than re-writing the load gets
    the block-wise vision-attention patch with it. Without that patch the sdpa
    encoder materialises a full [S, S] mask, which Windows spills into system
    RAM: the card sits at 100% and nothing finishes.
    """
    source = HERE / "rank.options.mc.videollama3.py"
    module_spec = importlib.util.spec_from_file_location("vl3_ranker", source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    path = Path(args.model_path or spec["path"])

    if not path.exists():
        raise SystemExit(f"no checkpoint at {path}")

    client = module.VideoLlama3(
        path, num_predict=args.force_length or spec["max_tokens"],
        temperature=0.0, top_p=1.0, top_k=0, retries=1,
        max_visual_tokens=args.max_visual_tokens)

    return module, client


def measure_transformers(vl3, client, module, item: dict, prompt: str,
                         mode: str, args: argparse.Namespace, nonce: str,
                         gen: int) -> dict:
    """Two-point prefill/decode for the in-process model.

    transformers' generate() gives no token-level timestamps without a
    streamer, and VideoLLaMA3's remote generate() is not guaranteed to forward
    one, so max_new_tokens=1 is the prefill and the slope to a full generate is
    the decode rate.
    """
    import torch

    entry = item["frames"]
    frames = None

    if mode != "none":
        directory = args.frames_dir / Path(entry["video_id"]).stem
        subset = {"files": entry["files"][:args.frames],
                  "timestamps": entry["timestamps"][:args.frames]}
        frames = vl3.load_frames(directory, subset)

    system = module.SYSTEM_PROMPT

    if not args.cached:
        system = f"{system} [{nonce}]"

    content: list[dict] = []

    if frames:
        content.append({"type": "video",
                        "video": [frame for frame, _ in frames],
                        "num_frames": len(frames),
                        "timestamps": [stamp for _, stamp in frames]})

    content.append({"type": "text", "text": prompt})
    inputs = client.processor(
        conversation=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
        add_generation_prompt=True, return_tensors="pt")
    inputs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    prompt_tokens = int(inputs["input_ids"].shape[-1])

    def once(max_new_tokens: int) -> tuple[float, object]:
        # min_new_tokens is the in-process ignore_eos, and only --force-length
        # asks for it: left off, the model stops where it would in a real run.
        forced = {"min_new_tokens": max_new_tokens} if args.force_length else {}
        torch.cuda.synchronize()
        started = time.perf_counter()

        with torch.inference_mode():
            output = client.net.generate(**inputs, max_new_tokens=max_new_tokens,
                                         do_sample=False, **forced)

        torch.cuda.synchronize()

        return time.perf_counter() - started, output[0]

    t_one, _ = once(1)
    t_full, ids = once(gen)

    # The remote generate() returns only the new tokens; strip the prompt
    # anyway if a future version starts returning it.
    if int(ids.shape[0]) > gen:
        ids = ids[prompt_tokens:]

    produced = int(ids.shape[0])
    window = t_full - t_one

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": produced,
        "exact_counts": True,
        "finish_reason": "length" if produced >= gen else "stop",
        "ttft_s": round(t_one, 4),
        "decode_s": round(window, 4),
        "total_s": round(t_full, 4),
        "prefill_tps": round(prompt_tokens / t_one, 1),
        "decode_tps": round((produced - 1) / window, 2)
        if produced > 1 and window > 0 else None,
        "method": "two-point",
        "gpu_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
        "response": client.processor.batch_decode(
            [ids], skip_special_tokens=True)[0].strip(),
        "reasoning": "",
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def summarise(samples: list[dict]) -> dict:
    """Median and mean over the measured requests.

    Both, because they answer different questions: the median is the typical
    clip, the mean is what a run's wall clock actually pays, and on a corpus
    whose durations span two orders of magnitude they are not close. The spread
    is kept too -- a min and max ten times apart is the finding, not noise.
    """
    ok = [s for s in samples if not s.get("error")]

    def field(name: str) -> list[float]:
        return [s[name] for s in ok if s.get(name) is not None]

    out: dict = {"n": len(ok), "failed": len(samples) - len(ok)}

    if not ok:
        return out

    for name in ("prompt_tokens", "completion_tokens", "ttft_s", "decode_s",
                 "total_s", "prefill_tps", "decode_tps",
                 "server_prefill_tps", "server_decode_tps"):
        values = field(name)

        if not values:
            continue

        out[f"{name}_median"] = round(statistics.median(values), 3)
        out[f"{name}_mean"] = round(statistics.fmean(values), 3)

        if len(values) > 1:
            out[f"{name}_min"] = round(min(values), 2)
            out[f"{name}_max"] = round(max(values), 2)

    return out


def print_samples(samples: list[dict]) -> None:
    """One line per query, shortest clip first."""
    print(f"\n  {'Q_ID':>6} {'video':<16} {'secs':>6} {'prompt':>7} {'gen':>5} "
          f"{'TTFT s':>8} {'prefill/s':>10} {'decode/s':>9}  reply")
    print("  " + "-" * 94)

    for sample in sorted(samples, key=lambda s: s["duration_sec"]):
        if sample.get("error"):
            print(f"  {sample['q_id']:>6} {str(sample['video_id'])[:16]:<16} "
                  f"{sample['duration_sec']:>6.1f} "
                  f"{'FAILED':>54}  {sample['error'][:28]}")
            continue

        reply = " ".join((sample.get("response") or "").split())[:28]
        print(f"  {sample['q_id']:>6} {str(sample['video_id'])[:16]:<16} "
              f"{sample['duration_sec']:>6.1f} "
              f"{int(sample.get('prompt_tokens') or 0):>7} "
              f"{int(sample.get('completion_tokens') or 0):>5} "
              f"{sample.get('ttft_s') or 0:>8.2f} "
              f"{sample.get('prefill_tps') or 0:>10.1f} "
              f"{sample.get('decode_tps') or 0:>9.2f}  {reply}")


def print_summary(summary: dict, aggregate: dict | None) -> None:
    print(f"\n  {'':16} {'median':>10} {'mean':>10} {'min':>10} {'max':>10}")

    for label, key in (("prompt tokens", "prompt_tokens"),
                       ("gen tokens", "completion_tokens"),
                       ("TTFT s", "ttft_s"),
                       ("prefill tok/s", "prefill_tps"),
                       ("decode tok/s", "decode_tps")):
        print(f"  {label:<16} {summary.get(key + '_median', 0):>10.1f} "
              f"{summary.get(key + '_mean', 0):>10.1f} "
              f"{summary.get(key + '_min', 0):>10.1f} "
              f"{summary.get(key + '_max', 0):>10.1f}")

    if summary.get("failed"):
        print(f"  {summary['failed']} of "
              f"{summary['failed'] + summary['n']} request(s) FAILED -- the "
              f"medians above cover only the {summary['n']} that returned")

    if summary.get("completion_tokens_median", 0) < SHORT_COMPLETION:
        print(f"  note: median completion is under {SHORT_COMPLETION} tokens, "
              f"so the decode rate is measured over a short window and is "
              f"noisy. --force-length pins it.")

    if aggregate:
        print(f"  aggregate at concurrency {aggregate['concurrency']}: "
              f"{aggregate['decode_tps']} output tok/s, "
              f"{aggregate['total_tps']} total tok/s over "
              f"{aggregate['wall_s']} s")


def write_csv(path: Path, records: list[dict]) -> None:
    """One row per request, appended. The raw reply is a column."""
    columns = ["measured", "name", "task", "model", "workload", "q_id",
               "video_id",
               "duration_sec", "n_options", "n_frames", "prompt_tokens",
               "completion_tokens", "finish_reason", "ttft_s", "decode_s",
               "total_s", "prefill_tps", "decode_tps", "server_prefill_tps",
               "server_decode_tps", "cached_prompt_tokens", "concurrency",
               "prefix_cache", "forced_length", "error", "response",
               "reasoning"]
    exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns,
                                extrasaction="ignore")

        if not exists:
            writer.writeheader()

        for record in records:
            for sample in record["samples"]:
                writer.writerow({
                    "measured": record["measured"],
                    "name": record["name"],
                    "task": record.get("task"),
                    "model": record["model"],
                    "workload": record["mode"],
                    "concurrency": record["config"]["concurrency"],
                    "prefix_cache": record["config"]["prefix_cache"],
                    "forced_length": record["config"]["force_length"],
                    **sample,
                })


def print_report(path: Path) -> int:
    """Every model's latest measurement, side by side.

    Latest per (model, workload, concurrency): a bench is re-run whenever a
    server's flags change, and the old row stays in the file for history but is
    not what a comparison should show. Rows measured with the prefix cache on,
    or at a forced length, are marked -- they are not comparable with the rest,
    and saying so beats dropping them.
    """
    if not path.exists():
        raise SystemExit(f"no results at {path} -- run a bench first")

    latest: dict[tuple, dict] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        record = json.loads(line)
        latest[(record.get("task", 2), record["name"], record["mode"],
                record["config"].get("concurrency", 1))] = record

    print(f"\n{'task':>4} {'model':<15} {'payload':<7} {'n':>3} "
          f"{'prompt':>7} {'gen':>5} {'TTFT s':>7} {'prefill/s':>19} "
          f"{'decode/s':>17}  note")
    print(f"{'':4} {'':15} {'':7} {'':3} {'med':>7} {'med':>5} {'med':>7} "
          f"{'median':>9} {'mean':>9} {'median':>8} {'mean':>8}")
    print("-" * 110)

    for key in sorted(latest):
        record = latest[key]
        summary = record["summary"]
        config = record["config"]
        notes = []

        if config.get("prefix_cache"):
            notes.append("PREFIX CACHE ON")

        if config.get("force_length"):
            notes.append(f"forced {config['force_length']}")

        if record.get("method") == "two-point":
            notes.append("two-point")

        if key[3] > 1:
            notes.append(f"conc {key[3]}")

        if summary.get("failed"):
            notes.append(f"{summary['failed']} FAILED")

        if summary.get("completion_tokens_median", 0) < SHORT_COMPLETION:
            notes.append("short decode")

        print(f"{key[0]:>4} {record['name']:<15} {record['mode']:<7} "
              f"{summary.get('n', 0):>3} "
              f"{int(summary.get('prompt_tokens_median') or 0):>7} "
              f"{int(summary.get('completion_tokens_median') or 0):>5} "
              f"{summary.get('ttft_s_median', 0):>7.2f} "
              f"{summary.get('prefill_tps_median', 0):>9.1f} "
              f"{summary.get('prefill_tps_mean', 0):>9.1f} "
              f"{summary.get('decode_tps_median', 0):>8.1f} "
              f"{summary.get('decode_tps_mean', 0):>8.1f}  {', '.join(notes)}")

    print(f"\n{len(latest)} row(s) from {path}\n")

    return 0


def print_list() -> None:
    print(f"\n{'name':<15} {'port':<7} {'payload':<8} {'run from':<9} "
          f"{'gen budget':<11} model")
    print("-" * 104)

    for name, spec in MODELS.items():
        port = spec["host"].rsplit(":", 1)[-1] if spec["host"] else "-"
        print(f"{name:<15} {port:<7} {spec['payload']:<8} {spec['where']:<9} "
              f"{spec['max_tokens']:<11} {spec['model']}")

    print("\nServe commands: --serve-cmd NAME. The 3090 holds one model at a "
          "time.\nModels marked 'wsl' must be benched from inside WSL2: the "
          "server opens\nthe clip off the /mnt/e path in the request.\n")


# --------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="a name from --list")
    parser.add_argument("--task", type=int, choices=(1, 2), default=2,
                        help="1 = answer generation (generate.answers.py, ten "
                             "answers per reply), 2 = multiple-choice ranking "
                             "(rank.options.mc.py). Default 2")
    parser.add_argument("--answers", type=int, default=DEFAULT_ANSWERS,
                        help=f"answers asked for per reply, task 1 only "
                             f"(default {DEFAULT_ANSWERS}, the run format's "
                             f"slot count)")
    parser.add_argument("--list", action="store_true",
                        help="print every known model and exit")
    parser.add_argument("--report", action="store_true",
                        help="print every model's latest result and exit")
    parser.add_argument("--serve-cmd", metavar="NAME",
                        help="print that model's serve command and exit")
    parser.add_argument("--videos", type=int, default=DEFAULT_VIDEO_COUNT,
                        help=f"how many clips to sample across the duration "
                             f"range (default {DEFAULT_VIDEO_COUNT})")
    parser.add_argument("--spread", choices=("duration", "corpus"),
                        default="duration",
                        help="duration: clips spaced geometrically over the "
                             "whole range, for a scaling curve. corpus: the "
                             "corpus's own duration distribution, for what a "
                             "full run costs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="selection seed; the same seed picks the same "
                             "clips for every model")
    parser.add_argument("--only", type=int, nargs="+", metavar="Q_ID",
                        help="bench these Q_IDs instead of a sample")
    parser.add_argument("--mode", choices=("auto", "frames", "video", "none"),
                        default="auto",
                        help="what the pixels travel as (default: what that "
                             "model's run sends); none keeps the same prompt "
                             "with no video attached")
    parser.add_argument("--sweep", action="store_true",
                        help="measure --mode none first, then the model's own "
                             "payload, so the pixel cost is the difference")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAME_COUNT,
                        help=f"frames per request, frame payloads only "
                             f"(default {DEFAULT_FRAME_COUNT})")
    parser.add_argument("--frame-size", type=int,
                        help="resize every frame to N x N (InternVL needs 448)")
    parser.add_argument("--max-tokens", type=int,
                        help="generation budget (default: that model's own)")
    parser.add_argument("--force-length", type=int, metavar="N",
                        help="pin every completion to exactly N tokens, for "
                             "strictly comparable decode rates and a junk reply")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"passes over the sample (default {DEFAULT_REPEATS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="discarded requests before measuring "
                             f"(default {DEFAULT_WARMUP})")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="in-flight requests; >1 also reports aggregate "
                             "throughput, which is what a batch run sees")
    parser.add_argument("--method", choices=("stream", "two-point"),
                        default="stream",
                        help="how prefill and decode are separated "
                             "(in-process models are always two-point)")
    parser.add_argument("--cached", action="store_true",
                        help="drop the nonce so the server's prefix cache can "
                             "serve the prefill (measures the warm path)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-routing", action="store_true",
                        help="attach every input regardless of the router")
    parser.add_argument("--queries", type=Path,
                        help="released query file (default: that task's)")
    parser.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument("--host", help="override the registry host")
    parser.add_argument("--raw-video", action="store_true",
                        help="send the clip as it is, skipping the fps "
                             "pre-shrink the runs apply. A server decoding "
                             "every frame of a long 1080p clip will 500")
    parser.add_argument("--keep-localhost", action="store_true",
                        help="do not rewrite localhost to 127.0.0.1; see "
                             "pin_ipv4() for the ~2 s per connection that costs")
    parser.add_argument("--served-model", help="override the model name sent")
    parser.add_argument("--model-path", help="checkpoint dir, in-process only")
    parser.add_argument("--max-visual-tokens", type=int, default=4096,
                        help="whole-clip visual budget, in-process only")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--no-log", action="store_true",
                        help="print the tables but write nothing")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the prompts, print the sample and the "
                             "first prompt, and send nothing")

    args = parser.parse_args(argv)

    if args.queries is None:
        args.queries = DEFAULT_QUERIES[args.task]

    return args


def prepare(module, args: argparse.Namespace, spec: dict) -> list[dict]:
    """The chosen queries, each with its frames, transcript, plan and prompt."""
    index = load_frame_index(args.frames_dir)
    queries = load_task_queries(module, args.task, args.queries)
    routing = module.load_routing(args.routing)
    needs_clip = (args.mode == "video"
                  or (args.mode == "auto" and spec["payload"] == "video"))

    if args.only:
        wanted = set(args.only)
        chosen = []

        for query in queries:
            if query["Q_ID"] not in wanted:
                continue

            entry = index["videos"].get(Path(query["Video_ID"]).stem)

            if entry:
                chosen.append({"query": query, "frames": entry})
    else:
        chosen = select_queries(queries, index, args.videos_dir, args.videos,
                                args.seed, needs_clip, args.spread)

    for item in chosen:
        query = item["query"]
        plan = plan_for(module, args.task, routing.get(query["Q_ID"]),
                        query["Question"], args.ignore_routing)
        transcript = module.load_transcript(args.transcripts,
                                            query["Video_ID"]) \
            if plan["transcript"] else None
        item["plan"] = plan
        item["transcript"] = transcript
        item["prompt"] = build_prompt(module, args.task, query, item["frames"],
                                      transcript, plan, args.answers)

    # Shortest first, so the printed table and the header read as a curve
    # whichever way the queries arrived.
    return sorted(chosen, key=lambda item: item["frames"]["duration_sec"])


def run_workload(spec: dict, module, args: argparse.Namespace, mode: str,
                 items: list[dict]) -> tuple[list[dict], dict | None]:
    """Warm up, then measure every chosen query once per repeat."""
    gen = generation_budget(spec, args)

    if spec["api"] == "transformers":
        vl3, client = spec["_loaded"]
        samples: list[dict] = []

        def in_process(repeat: int, index: int, item: dict) -> dict:
            sample = measure_transformers(
                vl3, client, module, item, item["prompt"], mode, args,
                f"{time.time_ns()}-{repeat}-{index}", gen)

            return {**sample, **query_shape(item, mode, args)}

        # The first call also pays CUDA's one-off kernel autotuning, which is
        # not this model's prefill.
        for i in range(args.warmup):
            in_process(-1 - i, i, items[i % len(items)])

        for repeat in range(args.repeats):
            for index, item in enumerate(items):
                samples.append(in_process(repeat, index, item))

        return samples, None

    import requests

    session = requests.Session()

    def one(job: tuple[int, dict]) -> dict:
        index, item = job
        content, shape = build_content(spec, item, item["prompt"], mode, args)
        nonce = f"{time.time_ns()}-{index}-{random.randrange(1 << 30)}"
        body = build_body(spec, content, module, args, nonce, gen,
                          stream=args.method == "stream")

        try:
            if args.method == "stream":
                sample = measure_stream(session, spec, body, args.timeout)
            else:
                sample = measure_two_point(session, spec, body, args.timeout,
                                           gen)
        except Exception as exc:
            # One clip the server chokes on should cost that clip, not the
            # other nine: the first Cosmos bench died on a single 389 s 1080p
            # video and recorded nothing at all. The failure is kept as a
            # sample so it shows up in the table and the CSV rather than
            # silently shrinking n.
            print(f"  Q_ID {shape['q_id']} ({shape['duration_sec']:.0f}s): "
                  f"{type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr)

            return {**shape, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}

        return {**sample, **shape}

    for i in range(args.warmup):
        one((-1 - i, items[i % len(items)]))

    jobs = [(i, item) for _ in range(args.repeats)
            for i, item in enumerate(items)]
    aggregate = None

    if args.concurrency > 1:
        # Aggregate throughput at the worker count the real passes use: the
        # per-request rate falls under batching while the server's total rate
        # rises, and it is the total that sets a run's wall clock.
        started = time.perf_counter()

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            samples = list(pool.map(one, jobs))

        wall = time.perf_counter() - started
        done = [s for s in samples if not s.get("error")]
        out_tokens = sum(s["completion_tokens"] for s in done)
        in_tokens = sum(s["prompt_tokens"] or 0 for s in done)
        aggregate = {
            "concurrency": args.concurrency,
            "requests": len(samples),
            "wall_s": round(wall, 2),
            "decode_tps": round(out_tokens / wall, 1),
            "total_tps": round((out_tokens + in_tokens) / wall, 1),
        }
    else:
        samples = [one(job) for job in jobs]

    return samples, aggregate


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list:
        print_list()

        return 0

    if args.report:
        return print_report(args.out)

    if args.serve_cmd:
        if args.serve_cmd not in MODELS:
            raise SystemExit(f"unknown model {args.serve_cmd!r}; --list shows "
                             f"{', '.join(MODELS)}")

        print(MODELS[args.serve_cmd]["serve"])

        return 0

    if not args.model:
        raise SystemExit("--model is required; --list shows what is known")

    if args.model not in MODELS:
        raise SystemExit(f"unknown model {args.model!r}; --list shows "
                         f"{', '.join(MODELS)}")

    spec = dict(MODELS[args.model])

    if args.host:
        spec["host"] = args.host

    if spec["host"] and not args.keep_localhost:
        spec["host"] = pin_ipv4(spec["host"])

    if args.served_model:
        spec["model"] = args.served_model
        spec["pinned_model"] = True

    module = import_pass(args.task)
    items = prepare(module, args, spec)
    modes = [args.mode if args.mode != "auto" else spec["payload"]]

    if args.sweep:
        modes = ["none"] + modes

    print(f"\n{args.model}: {spec['model']}"
          + (f" at {spec['host']}" if spec["host"] else " (in-process)"))
    print(f"  {spec['notes']}")
    print(f"  task {args.task} | {len(items)} TREC 2026 queries, "
          f"{items[0]['frames']['duration_sec']:.0f}-"
          f"{items[-1]['frames']['duration_sec']:.0f} s clips, "
          f"{args.repeats} pass(es), concurrency {args.concurrency}, "
          f"{'prefix cache ON' if args.cached else 'unique prefix per request'}")

    if args.dry_run:
        print(f"\n  {'Q_ID':>6} {'video':<18} {'secs':>6} {'opts':>5} "
              f"{'chars':>6}  question")
        print("  " + "-" * 92)

        for item in items:
            query = item["query"]
            print(f"  {query['Q_ID']:>6} {str(query['Video_ID'])[:18]:<18} "
                  f"{item['frames']['duration_sec']:>6.1f} "
                  f"{len(query.get('Options') or ()):>5} "
                  f"{len(item['prompt']):>6}  {query['Question'][:44]}")

        print(f"\n--- the prompt sent for Q_ID {items[0]['query']['Q_ID']} "
              f"---\nsystem: {module.SYSTEM_PROMPT}\n\n{items[0]['prompt']}\n")

        return 0

    if spec["api"] == "transformers":
        spec["_loaded"] = load_videollama3(spec, args)
    else:
        import requests

        served = preflight(requests.Session(), spec)

        if not spec.get("pinned_model"):
            spec["model"] = served

    records = []

    for mode in modes:
        samples, aggregate = run_workload(spec, module, args, mode, items)
        summary = summarise(samples)
        print(f"\n{'=' * 96}\n{args.model} | payload: {mode} | "
              f"{len(samples)} request(s)")
        print_samples(samples)
        print_summary(summary, aggregate)
        records.append({
            "measured": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "name": args.model,
            "task": args.task,
            "model": spec["model"],
            "host": spec["host"],
            "api": spec["api"],
            "mode": mode,
            "method": "two-point" if spec["api"] == "transformers"
            else args.method,
            "host_platform": platform.system(),
            "queries": [{"q_id": item["query"]["Q_ID"],
                         "video_id": item["query"]["Video_ID"],
                         "duration_sec": item["frames"]["duration_sec"],
                         "n_options": len(item["query"].get("Options") or ()),
                         "question": item["query"]["Question"],
                         "plan": item["plan"]}
                        for item in items],
            "config": {"max_tokens": generation_budget(spec, args),
                       "force_length": args.force_length,
                       "answers": args.answers if args.task == 1 else None,
                       "frames": args.frames,
                       "frame_size": args.frame_size or spec.get("frame_size"),
                       "videos": args.videos, "seed": args.seed,
                       "spread": args.spread,
                       "temperature": args.temperature,
                       "concurrency": args.concurrency,
                       "repeats": args.repeats, "warmup": args.warmup,
                       "prefix_cache": bool(args.cached),
                       "ignore_routing": bool(args.ignore_routing)},
            "summary": summary,
            "aggregate": aggregate,
            "samples": samples,
        })

    print()

    if not args.no_log:
        with args.out.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

        write_csv(args.csv, records)
        print(f"appended {len(records)} record(s) to {args.out}")
        print(f"appended {sum(len(r['samples']) for r in records)} row(s) "
              f"to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

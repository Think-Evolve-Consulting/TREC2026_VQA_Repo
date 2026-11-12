#!/usr/bin/env python3
"""Rank the Task 2 multiple-choice options with LLaVA-Video-7B over the video.

A port of ``VQA/rank.options.mc.muse.py`` to LLaVA-Video-7B-Qwen2 (bf16)
served by vLLM in WSL2. The ranking contract -- option numbers, permuted
passes, Borda, validator-safe output -- is unchanged; what differs is the
evidence. LLaVA-Video is a video model, so it is sent the clip itself as a
``video_url`` and vLLM samples the frames, rather than being shown the JPEGs
``VQA/extract.frames.py`` wrote. The frame index is still read, but only for
each clip's duration.

The checkpoint is ``Isotr0py/LLaVA-Video-7B-Qwen2-hf``, a conversion of
``lmms-lab/LLaVA-Video-7B-Qwen2`` to transformers' ``llava_onevision``
architecture, which is what lets vLLM serve it at all. Serve it from the
Cosmos venv (vLLM 0.12), with the Cosmos service stopped::

    systemctl --user stop cosmos-reason2
    /home/aakash/cosmos-reason2/.venv/bin/vllm serve Isotr0py/LLaVA-Video-7B-Qwen2-hf \\
        --host 0.0.0.0 --port 8009 --served-model-name llava-video-7b \\
        --allowed-local-media-path /mnt/e/TREC_2026 \\
        --max-model-len 16384 --gpu-memory-utilization 0.92 --enforce-eager \\
        --limit-mm-per-prompt '{"video": 1}' \\
        --media-io-kwargs '{"video": {"num_frames": 32}}' \\
        --skip-mm-profiling

``num_frames`` 32 are sampled uniformly over the clip, the way LLaVA-Video
was trained. Without ``--allowed-local-media-path`` every video is refused
with a 400, and every pass then fails loudly rather than ranking blind.

Run this **from inside WSL2**: the server opens the video off the path in the
request, so it must be a path the server can read (``/mnt/e/TREC_2026/...``)::

    /home/aakash/cosmos-reason2/.venv/bin/python \\
        /mnt/e/TREC_2026/VQA/rank.options.mc.llavavideo.py --limit 20

**No reasoning, so a short budget.** This is not a thinking model: the reply
is the JSON and nothing else, the schema constrains it, and 128 tokens is room
enough -- as in the Gemma pass, and at its colder temperature.

Output is the submission file ``validate_vqa_mc_2026.py`` accepts verbatim::

    [{"Q_ID": 1, "Video_ID": "8384.webm",
      "Answers": [{"Rank": 1, "Answer": "It fades"}, ...]}]

Five things drive the design:

**The model ranks option numbers, not option text.** The validator requires
every ``Answer`` to be one of that query's official options byte for byte, and
requires each option to appear exactly once. Asking for text invites the model
to paraphrase an option, merge two, or emit the same one twice. Asking for a
permutation of ``1..N`` and mapping the numbers back to the official strings
makes both properties structural: a rank can only ever be spent on one option,
and every option is spent exactly once. ``as_permutation()`` repairs a reply
that still manages to repeat or drop a number, and ``check_permutation()``
refuses to write a run file where any of that leaked through.

**The option order is permuted across passes.** An LLM ranking a list has a
real position bias, and here the officially-correct option sits at a fixed
place in the released file, so a bias toward "first" would score like signal
and generalize like nothing. Each query is asked ``--passes`` times with the
options presented in a different order -- official, reversed, then seeded
shuffles -- and the passes are combined by a confidence-weighted Borda count.
``report()`` prints how often rank 1 landed on the officially-first option, so
residual bias is visible rather than assumed away.

**Only the text moves between passes.** The video is sent first, so a query's
passes share a byte-identical prefix that vLLM's prefix cache can reuse.
Queries are grouped by video for the same reason.

**There is no Time field.** Task 1 requires one; the MC validator rejects any
key outside ``Rank``/``Answer`` (EXTRA_ANSWER_FIELDS), so timings live in the
cache instead of the run file.

Usage:
    python VQA/rank.options.mc.llavavideo.py --limit 20      # smoke test
    python VQA/rank.options.mc.llavavideo.py                 # everything, resumable
    python VQA/rank.options.mc.llavavideo.py --only 1 3 264  # named Q_IDs
    python VQA/rank.options.mc.llavavideo.py --rebuild       # re-emit run from cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = ROOT / "vqa.2026.testingDataset.Task2_v2.json"
DEFAULT_ROUTING = ROOT / "query.routing.task1.json"
DEFAULT_FRAMES = ROOT / "videos_TEST_VQA" / "frames"
DEFAULT_TRANSCRIPTS = ROOT / "videos_TEST_VQA" / "transcription_openwhisper"

DEFAULT_VIDEOS = ROOT / "videos_TEST_VQA" / "videos"

# vLLM in WSL2, port 8009, --served-model-name llava-video-7b. preflight()
# checks the name, so a run cannot land on whatever model happens to be loaded.
DEFAULT_MODEL = "llava-video-7b"
DEFAULT_HOST = "http://localhost:8009"
DEFAULT_OUT = ROOT / "llavavideo.run.task2.json"
DEFAULT_CACHE = ROOT / "rankings.task2.llavavideo.cache.jsonl"

# vLLM batches concurrent requests; 4 was the knee for the 8B Qwen3-VL pass on
# this card. Re-measure on the smoke test.
DEFAULT_WORKERS = 4

# A permutation of at most eight integers plus a confidence is ~40 tokens.
DEFAULT_NUM_PREDICT = 128

# The Gemma pass's ranking sampler: the diversity comes from permuting the
# options, not from sampling noise.
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64

# Official order, reversed, then seeded shuffles, combined by Borda.
DEFAULT_PASSES = 3

# Context below which 32 frames plus the text no longer fit.
MIN_MODEL_LEN = 12288

SYSTEM_PROMPT = (
    "You rank multiple-choice answer options for a question about a video you "
    "are shown. You reply with JSON only."
)

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
  order -- put what the video supports most first.
{extra}
Reply with JSON: {{"ranking": [{slots}], "confidence": 0.0}}
where ranking holds all {n} option numbers best first, and confidence is how
confident you are in the first one, 0.0 to 1.0."""

# Appended only when the router says so, in the same words the Task 1 passes
# use, so a decision from either task was asked the same thing.
OCR_RULE = ("- The answer is text rendered on screen. Read it exactly as "
            "written before choosing.\n")
TEMPORAL_RULE = ("- The answer depends on when something happens or in what "
                 "order. Follow the video in time order.\n")


def ranking_schema(n: int) -> dict:
    """Constrain the reply to exactly `n` integers, plus a confidence.

    Uniqueness is deliberately not expressed here: neither Ollama's llama.cpp
    grammar nor vLLM's xgrammar enforces ``uniqueItems``, so a schema claiming
    it would buy nothing and hide the repair path that actually does the work
    (as_permutation).
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
    option that was first is last, which is the cheapest possible probe of
    position bias. Later passes are shuffles seeded from the Q_ID, so the
    orders are reproducible across runs and differ between queries.
    """
    orders = [list(range(n))]

    if passes > 1:
        orders.append(list(reversed(range(n))))

    rng = random.Random(seed_for(q_id, 500))

    while len(orders) < passes:
        order = list(range(n))
        rng.shuffle(order)

        # A shuffle that reproduces an order already asked would spend a call
        # for no new evidence. With n >= 4 the retry almost never fires twice.
        if order not in orders or len(orders) > 2 * n:
            orders.append(order)

    return orders[:passes]


def build_context(frames: dict | None, transcript: dict | None,
                  routed_out: bool = False) -> str:
    """The block that precedes the question, identical across a query's passes.

    Kept word for word from generate.answers.py so that a query's evidence
    reads the same in both tasks: the only thing that should differ between a
    Task 1 answer and a Task 2 ranking is what is being asked for.
    """
    parts: list[str] = []

    if frames:
        length = (f"This video is {frames['duration_sec']:.0f} seconds long. "
                  if frames.get("duration_sec") else "")
        parts.append(f"{length}You are shown the video itself, sampled "
                     f"evenly from start to end.")
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

    Returns the positions the model gave, best first, and whether anything had
    to be repaired. The schema fixes the length and the range but cannot say
    "each exactly once", so this is where a repeated or missing option is
    caught: repeats keep their first, best occurrence, and whatever the model
    dropped is appended in the order it was shown. A repaired ranking is still
    a legitimate ranking -- it just means the tail was not chosen by the model,
    which report() counts.
    """
    values = raw if isinstance(raw, list) else []
    seen: set[int] = set()
    ordered: list[int] = []
    repaired = False

    for value in values:
        # A model that answers "3." or "3" instead of 3 is answering; only a
        # value that is not a number at all is a defect.
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

    Each pass awards ``n - position`` points to the option it put at that
    position, scaled by ``0.5 + 0.5 * confidence`` so a hesitant pass still
    votes but a confident one counts for twice as much. Ties break on mean
    rank, then on the official option order, so the output is deterministic.

    Every per-option signal is kept rather than folded away: a later fusion
    across models re-ranks from these fields without re-running inference.
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
    """Refuse to emit anything the MC validator would reject.

    The two rules that matter -- every official option present, none of them
    twice -- are the ones the whole design exists to guarantee, so a violation
    means a bug in this file rather than a bad reply, and it should stop the
    run instead of reaching the submission.
    """
    if len(answers) != len(options) or sorted(answers) != sorted(options):
        raise RuntimeError(
            f"Q_ID {q_id}: ranking is not a permutation of the official "
            f"options ({len(answers)} answers, "
            f"{len(set(answers))} distinct, {len(options)} options)")


class VllmVideo:
    """OpenAI-compatible client for vLLM holding LLaVA-Video.

    The clip goes in as one ``video_url`` part ahead of the text; vLLM opens it
    off the path and samples ``num_frames`` from it server-side. The schema
    goes in ``response_format``. A reply cut off by the budget is retried once
    at twice it, as in the Muse client, though with no reasoning trace that
    should not happen.
    """

    name = "vllm"

    def __init__(self, host: str, model: str, num_predict: int,
                 temperature: float, top_p: float, top_k: int,
                 timeout: float, retries: int, use_schema: bool) -> None:
        self.url = host.rstrip("/") + "/v1/chat/completions"
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
    def session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def ask(self, prompt: str, schema: dict, seed: int,
            videos: list[str] | None) -> tuple[dict, float, dict]:
        """One call. Returns the parsed reply, its wall time and its costs."""
        content: list[dict] = []

        if videos:
            content += [{"type": "video_url", "video_url": {"url": uri}}
                        for uri in videos]

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
        }

        if self.use_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "ranking",
                                "schema": dict(schema,
                                               additionalProperties=False)},
            }

        last: Exception | None = None
        extended = False

        for attempt in range(self.retries):
            started = time.monotonic()

            try:
                response = self.session.post(self.url, json=payload,
                                             timeout=self.timeout)

                # A 400 is the server refusing the request itself -- a prompt
                # past the slot's context, a schema it will not compile -- and
                # it will refuse it again just as fast.
                if response.status_code == 400:
                    raise RuntimeError(f"400: {response.text[:300]}")

                response.raise_for_status()

                body = response.json()
                choice = body["choices"][0]
                message = choice["message"]
                content_out = (message.get("content") or "").strip()
                reasoning = message.get("reasoning_content") or ""
                finish = choice.get("finish_reason")

                try:
                    reply = parse_json_object(content_out)
                except ValueError:
                    # Cut off inside the trace or the JSON: the one failure
                    # more budget fixes, so the only one that gets more.
                    if finish == "length" and not extended:
                        extended = True
                        payload["max_tokens"] = 2 * self.num_predict
                    raise ValueError(f"no JSON in content (finish_reason="
                                     f"{finish}, {len(reasoning)} reasoning "
                                     f"chars): {content_out[:120]!r}")

                usage = body.get("usage") or {}
                timings = body.get("timings") or {}

                return reply, time.monotonic() - started, {
                    "finish_reason": finish,
                    "reasoning_chars": len(reasoning),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "cached_tokens": timings.get("cache_n"),
                    "prompt_ms": timings.get("prompt_ms"),
                    "predicted_ms": timings.get("predicted_ms"),
                    "extended": extended,
                    "attempts": attempt + 1,
                }

            except (requests.RequestException, ValueError, KeyError,
                    IndexError) as exc:
                last = exc

                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"{type(last).__name__}: {last}")


def rank_query(client: VllmVideo, query: dict, options: list[str],
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
            reply, elapsed, cost = client.ask(prompt, schema,
                                              seed_for(q_id, index), images)
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
            "cost": cost,
        })

    if results:
        ranked = borda(results, n)
        fallback = False
    else:
        # Every pass failed. The options keep their official order, which is
        # no better than the dummy run for this query -- flagged here, counted
        # by report(), and re-runnable with --force --only.
        ranked = [{"option_index": index, "score": 0.0, "mean_rank": None,
                   "ranks": [], "firsts": 0} for index in range(n)]
        fallback = True

    answers = [
        {"Rank": rank, "Answer": options[item["option_index"]]}
        for rank, item in enumerate(ranked, 1)
    ]

    check_permutation([a["Answer"] for a in answers], options, q_id)

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
        "schema": client.use_schema,
        "n_options": n,
        "n_passes": passes,
        "n_replies": len(results),
        "n_pass_failures": failures,
        "n_repaired": repairs,
        "fallback": fallback,
        "n_frames": len(images) if images else 0,
        "has_transcript": "Spoken audio" in context,
        "confidence_mean": round(
            sum(float(r["confidence"]) for r in results
                if isinstance(r.get("confidence"), (int, float)))
            / max(sum(1 for r in results
                      if isinstance(r.get("confidence"), (int, float))), 1), 4)
        if results else None,
        # How many passes agreed on the option that won. 1.0 means the passes
        # were unanimous; 1/passes means they disagreed completely, which is
        # the strongest available signal that this query was a guess.
        "agreement": round(max((item["firsts"] for item in ranked), default=0)
                           / max(len(results), 1), 4),
        "routed": plan["routed"],
        "rescued": plan["rescued"],
        "ocr": plan["ocr"],
        "temporal": plan["temporal"],
        "elapsed": round(sum(r["elapsed"] for r in results), 2),
        "completion_tokens": sum(r["cost"].get("completion_tokens") or 0
                                 for r in results),
        "n_extended": sum(1 for r in results if r["cost"].get("extended")),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def preflight(host: str, model: str, workers: int,
              timeout: float = 30.0) -> None:
    """Refuse to start unless vLLM is up and serving this model.

    A dead server makes every pass fail and every query keep its official
    order -- the dummy run, looking finished. The media path is not checked
    here: a server started without --allowed-local-media-path refuses each
    video with a 400, which fails every pass loudly on the first query.
    """
    try:
        response = requests.get(host.rstrip("/") + "/v1/models",
                                timeout=timeout)
        response.raise_for_status()
        models = response.json()["data"]
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise SystemExit(
            f"no vLLM server answering at {host} "
            f"({type(exc).__name__}: {exc}).\n"
            f"  start it with the command in this script's docstring")

    served = {entry["id"]: entry for entry in models}
    max_len = (served.get(model) or {}).get("max_model_len") or 0

    print(f"serving: {sorted(served)} | max_model_len {max_len} | "
          f"{workers} worker(s)")

    if model not in served:
        raise SystemExit(f"{host} serves {sorted(served)}, not {model!r}. "
                         f"Pass --model to match.")

    if max_len and max_len < MIN_MODEL_LEN:
        raise SystemExit(f"max_model_len {max_len} is below {MIN_MODEL_LEN}; "
                         f"raise --max-model-len")


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_queries(path: Path) -> list[dict]:
    """The released MC queries, with Options flattened to a list of strings."""
    queries = []

    for query in load_json(path):
        queries.append({
            "Q_ID": query["Q_ID"],
            "Video_ID": query["Video_ID"],
            "Question": query["Question"],
            "Options": [option["option"] for option in query["Options"]],
        })

    return queries


def load_routing(path: Path) -> dict[int, dict]:
    """Q_ID -> routing verdict from route.query.inputs.py."""
    if not path.exists():
        return {}

    return {int(entry["Q_ID"]): entry for entry in load_json(path)}


def plan_for(routing: dict | None, question: str, ignore_routing: bool) -> dict:
    """Which inputs this query is sent, and which extra rules it gets.

    The routing file was produced for Task 1, which shares every Q_ID and
    video with Task 2. Measured against the v2 release, all 2,443 questions
    match once trailing whitespace is stripped (20 differ by exactly that), so
    the guard below currently withholds nothing -- it is here because a verdict
    about a different question is not a verdict about this one, and a re-release
    that actually reworded a question should quietly fall back to attaching
    every input rather than routing on a stale answer. --ignore-routing forces
    that everywhere.
    """
    if ignore_routing or not routing:
        return {"frames": True, "transcript": True, "ocr": False,
                "temporal": False, "routed": False, "rescued": False}

    if (routing.get("Question") or "").strip() != question.strip():
        return {"frames": True, "transcript": True, "ocr": False,
                "temporal": False, "routed": False, "rescued": False}

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

    return {"text": text, "language": language}


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


def assemble(queries: list[dict], cache: dict[int, dict]) -> tuple[list[dict], int]:
    """Emit the run file, carrying only the fields the MC validator allows.

    A cached ranking is re-checked against the released options here, not just
    copied: a cache written against an earlier release of the query file would
    otherwise produce a run that fails validation for reasons that look like a
    bug in the model rather than a stale file.
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

        run.append({
            "Q_ID": query["Q_ID"],
            "Video_ID": query["Video_ID"],
            "Answers": answers,
        })

    return run, missing


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Rank the Task 2 multiple-choice options with "
                    "LLaVA-Video-7B (vLLM) over the video."
    )

    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES,
                    help="released MC query file (default: %(default)s)")
    ap.add_argument("--routing", type=Path, default=DEFAULT_ROUTING,
                    help="per-query input routing from route.query.inputs.py; "
                         "applied only where the Task 1 question text still "
                         "matches this query's (default: %(default)s)")
    ap.add_argument("--ignore-routing", action="store_true",
                    help="attach every input for every query -- the ablation "
                         "a routed run has to beat")
    ap.add_argument("--frames", type=Path, default=DEFAULT_FRAMES,
                    help="frame directory; its index.json is read only for "
                         "clip durations")
    ap.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS,
                    help="the clips sent to the model (default: %(default)s)")
    ap.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="run file to write (default: %(default)s)")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help="JSONL work cache, one line per Q_ID "
                         "(default: %(default)s)")

    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="the server's --alias (default: %(default)s)")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT,
                    help="reasoning plus JSON; a reply cut off by it is "
                         "retried once at twice this (default: %(default)s)")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help="sampling temperature (default: %(default)s)")
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--retries", type=int, default=3,
                    help="attempts per call before giving up on it")
    ap.add_argument("--no-schema", dest="schema", action="store_false",
                    help="do not send response_format; parse the JSON out of "
                         "free content instead")

    ap.add_argument("--passes", type=int, default=DEFAULT_PASSES,
                    help="rankings per query, each with the options presented "
                         "in a different order (default: %(default)s)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="concurrent videos; match the server's -np "
                         "(default: %(default)s)")

    ap.add_argument("--limit", type=int, help="process at most N queries")
    ap.add_argument("--only", type=int, nargs="+", metavar="Q_ID")
    ap.add_argument("--force", action="store_true",
                    help="re-rank queries already in the cache")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the run file from the cache, no inference")

    return ap.parse_args(argv)


def video_uri(directory: Path, video_id: str) -> str | None:
    """The clip's file:// URI, or None when it is not on disk.

    ``Video_ID`` is a filename in some releases and a bare YouTube ID in
    others, which were downloaded as ``<id>.mp4``.
    """
    stem = Path(video_id).stem
    candidates = [directory / video_id, directory / f"{video_id}.mp4",
                  directory / f"{stem}.mp4"]
    candidates += sorted(directory.glob(f"{stem}.*"))

    for path in candidates:
        if path.is_file():
            return path.resolve().as_uri()

    return None


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.passes < 1:
        print("--passes must be at least 1", file=sys.stderr)
        return 2

    queries = load_queries(args.queries)
    cache = load_cache(args.cache)

    # A cache written by another model's pass is not this model's. Separate
    # cache files keep them apart by default; this catches a collision.
    foreign = sorted({str(record.get("model", "?")) for record in
                      cache.values()} - {args.model})

    if foreign:
        print(f"  WARNING: {args.cache.name} already holds rankings from "
              f"{', '.join(foreign)}; this run would mix models in one run "
              f"file. Point --cache elsewhere, or --force to re-rank.",
              file=sys.stderr)

    routing = {} if args.ignore_routing else load_routing(args.routing)

    if not routing and not args.ignore_routing and not args.rebuild:
        print(f"  WARNING: no routing at {args.routing} -- every input will be "
              f"attached for every query, as if --ignore-routing were set",
              file=sys.stderr)

    index_path = args.frames / "index.json"
    frame_index = load_json(index_path)["videos"] if index_path.exists() else {}

    if not frame_index and not args.rebuild:
        print(f"no frame index at {index_path} -- run VQA/extract.frames.py "
              f"first", file=sys.stderr)
        return 1

    print(f"model: {args.model} at {args.host} | {args.workers} worker(s) | "
          f"schema {'on' if args.schema else 'off'} | "
          f"max_tokens {args.num_predict} | videos {args.videos}")
    print(f"cache: {args.cache.name} | run: {args.out.name}")
    print(f"queries: {len(queries)} | cached: "
          f"{sum(1 for q in queries if q['Q_ID'] in cache)} | "
          f"videos with frames: {len(frame_index)}")

    if not args.rebuild:
        todo = [q for q in queries if args.force or q["Q_ID"] not in cache]

        if args.only:
            wanted = set(args.only)
            todo = [q for q in todo if q["Q_ID"] in wanted]

        if args.limit:
            todo = todo[:args.limit]

        # Group by video: a query's passes share an image prefix, and the
        # queries on one video share it with each other. Keeping a video on a
        # single worker is what turns that into cache hits.
        groups: dict[str, list[dict]] = {}

        for query in todo:
            groups.setdefault(query["Video_ID"], []).append(query)

        ordered = [groups[video_id] for video_id in sorted(groups)]

        no_video = sum(len(g) for g in ordered
                       if not video_uri(args.videos, g[0]["Video_ID"]))

        print(f"to rank: {len(todo)} queries over {len(ordered)} videos "
              f"({no_video} queries have no video file), {args.passes} "
              f"passes each")

        if ordered:
            preflight(args.host, args.model, args.workers)
            client = VllmVideo(args.host, args.model, args.num_predict,
                              args.temperature, args.top_p, args.top_k,
                              args.timeout, args.retries, args.schema)

            lock = threading.Lock()
            done = failed = 0
            started = time.monotonic()

            args.cache.parent.mkdir(parents=True, exist_ok=True)
            handle = args.cache.open("a", encoding="utf-8")

            def work(group: list[dict]) -> None:
                nonlocal done, failed

                video_id = group[0]["Video_ID"]
                uri = video_uri(args.videos, video_id)
                images = [uri] if uri else None
                # The frame index is read only for the clip's length; the
                # model sees the video itself.
                frames = (frame_index.get(Path(video_id).stem)
                          or {"duration_sec": 0}) if uri else None

                transcript = load_transcript(args.transcripts, video_id)
                has_speech = bool(transcript and transcript.get("text"))

                for query in group:
                    plan = plan_for(routing.get(query["Q_ID"]),
                                    query["Question"], args.ignore_routing)

                    # Never send a prompt with no evidence at all: a query the
                    # router sent to the audio of a clip that turns out to have
                    # no speech gets its frames back.
                    if not plan["frames"] and not has_speech:
                        plan = dict(plan, frames=True, rescued=True)

                    send_images = images if plan["frames"] else None
                    context = build_context(
                        frames if plan["frames"] else None,
                        transcript if plan["transcript"] else None,
                        routed_out=not plan["frames"],
                    )

                    try:
                        record = rank_query(client, query, query["Options"],
                                            context, send_images, args.passes,
                                            plan)
                    except Exception as exc:  # noqa: BLE001 - one bad query
                        with lock:            # must not sink a long run
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
                            flag = " FALLBACK"
                        elif record["n_pass_failures"]:
                            flag = f" ERR({record['n_pass_failures']})"
                        elif record["n_repaired"]:
                            flag = f" FIX({record['n_repaired']})"

                        if record["n_extended"]:
                            flag += f" EXT({record['n_extended']})"

                        print(f"  [{done}/{len(todo)}] {rate * 60:5.1f}/min "
                              f"eta {remaining:4.1f}h  "
                              f"{record['elapsed']:6.1f}s "
                              f"{record['n_frames']:2d}f "
                              f"{record['completion_tokens']:5d}t "
                              f"a{record['agreement']:.2f}{flag}  "
                              f"Q{record['Q_ID']}: "
                              f"{record['answers'][0]['Answer'][:48]}")

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

    The two that matter are position bias and pass agreement. The MC task is
    scored on top-1 accuracy and MRR, and a model that quietly prefers "the
    option shown first" would post a plausible-looking score on this corpus
    while having learned nothing about the videos.
    """
    records = [cache[q["Q_ID"]] for q in queries
               if q["Q_ID"] in cache and cache[q["Q_ID"]].get("answers")]

    if not records:
        return

    by_qid = {q["Q_ID"]: q for q in queries}
    first_official = 0
    last_official = 0
    counted = 0

    for record in records:
        query = by_qid.get(record["Q_ID"])

        if not query or not record.get("ranked"):
            continue

        winner = record["ranked"][0]["option_index"]
        counted += 1
        first_official += int(winner == 0)
        last_official += int(winner == len(query["Options"]) - 1)

    passes = [r["n_replies"] for r in records]
    agreement = [r["agreement"] for r in records if r["n_replies"]]
    unanimous = sum(1 for r in records if r["agreement"] >= 0.999)

    print(f"\nquality checks over {len(records)} ranked queries")
    print(f"  passes completed per query, mean {sum(passes) / len(passes):.2f}")
    print(f"  passes unanimous on rank 1:         {unanimous} "
          f"({100 * unanimous / len(records):.1f}%)  "
          f"mean agreement {sum(agreement) / max(len(agreement), 1):.3f}")

    if counted:
        # Uniform would be 1/8 = 12.5% for the 8-option queries and 25% for
        # the 4-option ones. Far above that on the *official* first option is
        # position bias surviving the permutation, not skill.
        expected = sum(1 / len(by_qid[r["Q_ID"]]["Options"]) for r in records
                       if r["Q_ID"] in by_qid) / counted
        print(f"  rank 1 is the officially FIRST option: {first_official} "
              f"({100 * first_official / counted:.1f}%, chance "
              f"{100 * expected:.1f}%)")
        print(f"  rank 1 is the officially LAST option:  {last_official} "
              f"({100 * last_official / counted:.1f}%)")

    print(f"  queries with a repaired ranking:    "
          f"{sum(1 for r in records if r['n_repaired'])}")
    print(f"  queries with a failed pass:         "
          f"{sum(1 for r in records if r['n_pass_failures'])}")
    print(f"  queries left in official order:     "
          f"{sum(1 for r in records if r['fallback'])}"
          f"{'  -- CHECK: these are unranked' if any(r['fallback'] for r in records) else ''}")
    print(f"  queries ranked on a routed plan:    "
          f"{sum(1 for r in records if r.get('routed'))}"
          f"  (rescued {sum(1 for r in records if r.get('rescued'))})")
    print(f"  queries ranked without frames:      "
          f"{sum(1 for r in records if not r['n_frames'])}")
    print(f"  queries ranked without audio:       "
          f"{sum(1 for r in records if not r['has_transcript'])}")

    confidences = [r["confidence_mean"] for r in records
                   if r.get("confidence_mean") is not None]

    if confidences:
        print(f"  mean self-reported confidence:      "
              f"{sum(confidences) / len(confidences):.3f}")

    times = [r["elapsed"] for r in records if r["elapsed"]]

    if times:
        print(f"  inference per query: mean {sum(times) / len(times):5.2f}s  "
              f"total {sum(times) / 3600:.2f} h")

    # Where the time goes, per pass. vLLM reports token counts but not the
    # prefill/decode split, so those two read 0 here.
    costs = [p["cost"] for r in records for p in r.get("passes", [])
             if p.get("cost")]

    if costs:
        def mean_of(key: str) -> float:
            values = [c[key] for c in costs
                      if isinstance(c.get(key), (int, float))]
            return sum(values) / max(len(values), 1)

        print(f"  per pass: {mean_of('prompt_tokens'):6.0f} prompt tokens "
              f"({mean_of('cached_tokens'):6.0f} from cache), "
              f"{mean_of('completion_tokens'):5.0f} completion tokens, "
              f"prefill {mean_of('prompt_ms') / 1000:5.2f}s, "
              f"decode {mean_of('predicted_ms') / 1000:5.2f}s")
        print(f"  passes retried at 2x budget:        "
              f"{sum(1 for c in costs if c.get('extended'))}")


if __name__ == "__main__":
    sys.exit(main())

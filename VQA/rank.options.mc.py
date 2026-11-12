#!/usr/bin/env python3
"""Rank the Task 2 multiple-choice options with Gemma 4 over sampled frames.

For every query this shows Gemma 4 12B the frames extracted by
``VQA/extract.frames.py``, the Whisper transcript in
``videos_TEST_VQA/transcription_openwhisper``, the question, and that query's
official options -- then asks for a full ranking of the options, best first.

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

**Only the text moves between passes.** Frames are sent first, exactly as in
``generate.answers.py``: Gemma 4 wants image content ahead of the text, and it
also makes a query's passes -- and every query on the same video -- share a
byte-identical prefix, so the frame prefill is paid once per video instead of
once per call. Queries are grouped by video for the same reason.

**Ranking wants a colder sampler than answer generation.** The diversity that
matters here comes from permuting the options, not from sampling noise, so the
default temperature is 0.2 rather than the 0.6 of the Task 1 pass.

**There is no Time field.** Task 1 requires one; the MC validator rejects any
key outside ``Rank``/``Answer`` (EXTRA_ANSWER_FIELDS), so timings live in the
cache instead of the run file.

Usage:
    python VQA/rank.options.mc.py --limit 20      # smoke test
    python VQA/rank.options.mc.py                 # everything, resumable
    python VQA/rank.options.mc.py --only 1 3 264  # named Q_IDs
    python VQA/rank.options.mc.py --rebuild       # re-emit run from cache
    python VQA/rank.options.mc.py --backend ollama  # the Windows path
"""

from __future__ import annotations

import argparse
import base64
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

# Same two backends, same servers, same split of cache and run files as
# generate.answers.py: Ollama runs gemma4:12b (Q4_K_M) on Windows, vLLM runs
# Google's QAT INT4 checkpoint in WSL2 on port 8010. Different quantisations
# are different runs, not a resumable one.
DEFAULT_BACKEND = "vllm"

BACKENDS = {
    "ollama": {
        "model": "gemma4:12b",
        "host": "http://localhost:11434",
        "out": "gemma4.run.task2.json",
        "cache": "rankings.task2.gemma4.cache.jsonl",
        # One slot on this box; extra client threads only evict each other's
        # frame prefix. See the Task 1 script for the measurement.
        "workers": 1,
    },
    "vllm": {
        "model": "google/gemma-4-12B-it-qat-w4a16-ct",
        "host": "http://localhost:8010",
        "out": "gemma4.vllm.run.task2.json",
        "cache": "rankings.task2.gemma4.vllm.cache.jsonl",
        # 8 was the measured knee for the Task 1 pass on this 3090 with the
        # same model and the same frame payload. The completions here are far
        # shorter -- a permutation, not ten sentences -- so this pass is more
        # prefill-bound than that one; re-measure with --limit 20 before a
        # full run rather than trusting the transfer.
        "workers": 8,
    },
}

# Ollama only: it defaults to 4096 whatever the model's window is, which is
# less than the frames alone. On vLLM the window is --max-model-len (16384 on
# the 8010 service) and an overlong prompt is a 400, not a silent truncation.
DEFAULT_NUM_CTX = 16384

# A ranking of eight integers plus a confidence is ~40 tokens. 128 leaves room
# for a stray restatement without letting a runaway reply cost real time.
DEFAULT_NUM_PREDICT = 128

# Colder than the Task 1 pass (0.6) on purpose: the ensemble's diversity here
# is the option permutation, not sampling noise, and a ranking task has one
# intended answer per presentation order rather than a spread of them.
DEFAULT_TEMPERATURE = 0.2

# Official order, reversed, then seeded shuffles. Three passes cost three
# short completions over a frame prefix that is prefilled once, and they are
# what makes a position-biased reply visible to the Borda count.
DEFAULT_PASSES = 3

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

# Appended only when the router says so, in the same words the Task 1 passes
# use, so a decision from either task was asked the same thing.
OCR_RULE = ("- The answer is text rendered on screen. Read it exactly as "
            "written before choosing.\n")
TEMPORAL_RULE = ("- The answer depends on when something happens or in what "
                 "order. The frames are in time order; use their timestamps.\n")


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


class Ollama:
    """Minimal chat client. One session per worker keeps the connection warm."""

    name = "ollama"

    def __init__(self, host: str, model: str, num_ctx: int, num_predict: int,
                 temperature: float, timeout: float, think: bool,
                 retries: int) -> None:
        self.url = host.rstrip("/") + "/api/chat"
        self.model = model
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.temperature = temperature
        self.timeout = timeout
        self.think = think
        self.retries = retries
        self.local = threading.local()

    @property
    def session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def ask(self, prompt: str, schema: dict, seed: int,
            images: list[str] | None) -> tuple[dict, float]:
        """One call. Returns the parsed reply and the wall time it took."""
        message: dict = {"role": "user", "content": prompt}

        # Image content first: Gemma 4 wants it there, and it is what lets the
        # KV prefix cache carry across a query's passes.
        if images:
            message["images"] = images

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                message,
            ],
            "format": schema,
            "stream": False,
            "think": self.think,
            "keep_alive": "30m",
            "options": {
                # Without this Ollama uses 4096 and the frames fall out of
                # context -- 32 frames alone are about 6,700 tokens.
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "temperature": self.temperature,
                "top_p": 0.95,
                "top_k": 64,
                "seed": seed,
            },
        }

        last: Exception | None = None

        for attempt in range(self.retries):
            started = time.monotonic()

            try:
                response = self.session.post(self.url, json=payload,
                                             timeout=self.timeout)
                response.raise_for_status()

                body = response.json()
                content = body["message"].get("content") or ""

                if not content:
                    reason = body.get("done_reason")
                    raise ValueError(f"empty content (done_reason={reason})")

                return parse_json_object(content), time.monotonic() - started

            except (requests.RequestException, ValueError,
                    json.JSONDecodeError) as exc:
                last = exc

                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"{type(last).__name__}: {last}")


class VLLM:
    """OpenAI-compatible client for vLLM. Same ask() contract as Ollama.

    Only the wire format differs from the Ollama path: frames become
    ``image_url`` parts carrying the same base64 bytes, still ahead of the
    text; the schema goes in ``response_format`` closed with
    ``additionalProperties: false`` because vLLM compiles it into a grammar;
    there is no num_ctx, so an overlong prompt is a 400 rather than silently
    dropped tokens; and thinking is turned off through the chat template,
    which is where Gemma 4 reads ``enable_thinking``.
    """

    name = "vllm"

    def __init__(self, host: str, model: str, num_predict: int,
                 temperature: float, timeout: float, think: bool,
                 retries: int) -> None:
        self.url = host.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.num_predict = num_predict
        self.temperature = temperature
        self.timeout = timeout
        self.think = think
        self.retries = retries
        self.local = threading.local()

    @property
    def session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def ask(self, prompt: str, schema: dict, seed: int,
            images: list[str] | None) -> tuple[dict, float]:
        """One call. Returns the parsed reply and the wall time it took."""
        content: list[dict] = []

        if images:
            content += [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                for image in images
            ]

        content.append({"type": "text", "text": prompt})

        schema = dict(schema, additionalProperties=False)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "max_tokens": self.num_predict,
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 64,
            "seed": seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "ranking", "schema": schema},
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
                # past max-model-len, a schema this build will not compile --
                # and it will refuse it again just as fast.
                if response.status_code == 400:
                    raise RuntimeError(f"400: {response.text[:300]}")

                response.raise_for_status()

                choice = response.json()["choices"][0]
                content_out = (choice["message"].get("content") or "").strip()

                if not content_out:
                    reason = choice.get("finish_reason")
                    raise ValueError(f"empty content (finish_reason={reason})")

                return parse_json_object(content_out), time.monotonic() - started

            except (requests.RequestException, ValueError, KeyError, IndexError,
                    json.JSONDecodeError) as exc:
                last = exc

                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"{type(last).__name__}: {last}")


def rank_query(client: VLLM | Ollama, query: dict, options: list[str],
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
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def preflight(host: str, model: str, timeout: float = 10.0) -> None:
    """Refuse to start unless the vLLM server is up and serving this model.

    A dead server is not a loud failure in this pipeline: every pass fails,
    every query falls back to the official option order, and the run completes
    looking finished while being exactly the dummy run. Ten seconds of
    checking is cheaper than finding out hours later.
    """
    url = host.rstrip("/") + "/v1/models"

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        served = [entry["id"] for entry in response.json()["data"]]
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise SystemExit(
            f"no vLLM server answering at {host} "
            f"({type(exc).__name__}: {exc}).\n"
            f"  start it with:  systemctl --user start gemma4-vllm\n"
            f"  and keep a WSL2 session open for the whole run -- see "
            f"VQA/GEMMA4_VLLM_SETUP.md")

    if model not in served:
        raise SystemExit(f"{host} serves {served}, not {model!r}. Pass "
                         f"--model to match, or restart the server.")


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
        description="Rank the Task 2 multiple-choice options with Gemma 4 "
                    "over video frames."
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
    ap.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    ap.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--out", type=Path,
                    help="run file to write (default: the backend's, see "
                         "--backend)")
    ap.add_argument("--cache", type=Path,
                    help="JSONL work cache, one line per Q_ID (default: the "
                         "backend's)")

    ap.add_argument("--backend", choices=sorted(BACKENDS),
                    default=DEFAULT_BACKEND,
                    help="which server ranks: vllm in WSL2, or ollama on "
                         "Windows (default: %(default)s). Sets the default "
                         "model, host, workers, cache and run file")
    ap.add_argument("--model", help="default: the backend's")
    ap.add_argument("--host", help="default: the backend's")
    ap.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX,
                    help="context window; must fit the frames. Ollama only -- "
                         "on vLLM the server's --max-model-len decides "
                         "(default: %(default)s)")
    ap.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help="sampling temperature (default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--retries", type=int, default=3,
                    help="attempts per call before giving up on it")
    ap.add_argument("--think", action="store_true",
                    help="let the model reason first; much slower")

    ap.add_argument("--passes", type=int, default=DEFAULT_PASSES,
                    help="rankings per query, each with the options presented "
                         "in a different order (default: %(default)s)")
    ap.add_argument("--workers", type=int,
                    help="concurrent videos (default: 1 on ollama, which "
                         "needs OLLAMA_NUM_PARALLEL to match; 8 on vllm)")

    ap.add_argument("--limit", type=int, help="process at most N queries")
    ap.add_argument("--only", type=int, nargs="+", metavar="Q_ID")
    ap.add_argument("--force", action="store_true",
                    help="re-rank queries already in the cache")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the run file from the cache, no inference")

    return ap.parse_args(argv)


def resolve_backend(args: argparse.Namespace) -> None:
    """Fill in what --backend implies and the caller did not override."""
    defaults = BACKENDS[args.backend]

    for name in ("model", "host", "workers"):
        if getattr(args, name) is None:
            setattr(args, name, defaults[name])

    for name in ("out", "cache"):
        if getattr(args, name) is None:
            setattr(args, name, ROOT / defaults[name])


def main(argv=None) -> int:
    args = parse_args(argv)
    resolve_backend(args)

    if args.passes < 1:
        print("--passes must be at least 1", file=sys.stderr)
        return 2

    queries = load_queries(args.queries)
    cache = load_cache(args.cache)

    # The two backends run different quantisations of the same model, so a
    # ranking from one is not a ranking from the other. Separate cache files
    # keep them apart by default; this catches a deliberate collision.
    foreign = sorted({record.get("backend", "ollama") for record in
                      cache.values()} - {args.backend})

    if foreign:
        print(f"  WARNING: {args.cache.name} already holds rankings from "
              f"{', '.join(foreign)}; this run would mix weights in one run "
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

    print(f"backend: {args.backend} | {args.model} at {args.host} | "
          f"{args.workers} worker(s)")
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

        no_frames = sum(len(g) for g in ordered
                        if Path(g[0]["Video_ID"]).stem not in frame_index)

        print(f"to rank: {len(todo)} queries over {len(ordered)} videos "
              f"({no_frames} queries have no frames), {args.passes} passes each")

        if ordered:
            if args.backend == "vllm":
                preflight(args.host, args.model)
                client = VLLM(args.host, args.model, args.num_predict,
                              args.temperature, args.timeout, args.think,
                              args.retries)
            else:
                client = Ollama(args.host, args.model, args.num_ctx,
                                args.num_predict, args.temperature,
                                args.timeout, args.think, args.retries)

            lock = threading.Lock()
            done = failed = 0
            started = time.monotonic()

            args.cache.parent.mkdir(parents=True, exist_ok=True)
            handle = args.cache.open("a", encoding="utf-8")

            def work(group: list[dict]) -> None:
                nonlocal done, failed

                video_id = group[0]["Video_ID"]
                frames = frame_index.get(Path(video_id).stem)
                images = None

                if frames:
                    frame_dir = args.frames / Path(video_id).stem
                    try:
                        images = [
                            base64.b64encode(
                                (frame_dir / name).read_bytes()).decode()
                            for name in frames["files"]
                        ]
                    except (OSError, KeyError, TypeError) as exc:
                        print(f"  {video_id}: unreadable frames ({exc})",
                              file=sys.stderr)
                        images = None
                        frames = None

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

                        print(f"  [{done}/{len(todo)}] {rate * 60:5.1f}/min "
                              f"eta {remaining:4.1f}h  "
                              f"{record['elapsed']:6.1f}s "
                              f"{record['n_frames']:2d}f "
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


if __name__ == "__main__":
    sys.exit(main())

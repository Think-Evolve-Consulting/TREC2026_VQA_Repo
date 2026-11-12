#!/usr/bin/env python3
"""Decide which inputs each Task 1 question needs before it can be answered.

Not every question in ``vqa.2026.testingDataset.Task1.json`` is a picture
question. "What does the son say at the very end of the video?" is answered by
the transcript, "What is the name of the shop?" by reading the sign, and "What
happens after the man gets a hair cut?" needs the clip in order rather than one
still frame. Sending all 2,443 through a single frames-only prompt spends the
context window on the wrong evidence.

This is the routing stage. Six specialists look at the *question text alone*
-- no video, no frames -- and each answers one narrow question about it:

    vision      must you recognise what is in the picture?
    audio       must you hear the soundtrack?
    ocr         must you read words rendered on screen?
    temporal    does the answer depend on when, or in what order, things happen?
    inference   is the answer worked out rather than observed?
    qtype       what kind of answer is being asked for?

One agent, one job, one model call, one prompt written against that job's own
failure mode. They never see each other's verdicts, so a weak signal in one
cannot talk the others into agreeing; the combination is decided afterwards by
``plan()``, which is plain Python and has no opinions of its own.

Output is one record per query::

    [{"Q_ID": 1,
      "Video_ID": "8384.webm",
      "Question": "What happens to the image of the bride and groom?",
      "Question_Type": "action",
      "Required_Inputs": ["vision"],
      "Route": "vision",
      "Temporal": true,
      "Inference": false,
      "Pipeline": ["frames", "dense_timestamped_frames", "answer_synthesis"],
      "Signals": {"vision": {"needed": true, "confidence": 0.95,
                             "reason": "..."}, ...}}]

Two things about the caching are worth knowing. Work is keyed on the question
text, not the Q_ID -- 2,443 queries hold 2,350 distinct questions, and a
repeated question is routed once for every Q_ID that asks it. And each *agent*
caches separately, so adding a seventh specialist later costs 2,350 calls, not
14,100: the six already on disk are read straight back.

Usage:
    python VQA/route.query.inputs.py --limit 20            # smoke test
    python VQA/route.query.inputs.py                       # everything, resumable
    python VQA/route.query.inputs.py --only 3 17           # named Q_IDs
    python VQA/route.query.inputs.py --agents ocr          # one specialist
    python VQA/route.query.inputs.py --rebuild             # re-emit JSON from cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = ROOT / "vqa.2026.testingDataset.Task1.json"

# Gemma 4 is served either by Ollama on Windows or by vLLM in WSL2, and vLLM
# is the default -- see VQA/GEMMA4_VLLM_SETUP.md. The two load different
# quantisations, so each keeps its own cache and output rather than resuming
# into the other's. query.routing.task1.json, which the Qwen3-VL pass routes
# against, is the Ollama one.
DEFAULT_BACKEND = "vllm"

BACKENDS = {
    "ollama": {
        "model": "gemma4:12b",
        "host": "http://localhost:11434",
        "out": "query.routing.task1.json",
        "cache": "query.routing.cache.jsonl",
        # One worker, deliberately. OLLAMA_NUM_PARALLEL is unset on this box,
        # so the server runs a single slot and extra client threads only
        # interleave requests through one KV cache.
        "workers": 1,
    },
    "vllm": {
        "model": "google/gemma-4-12B-it-qat-w4a16-ct",
        "host": "http://localhost:8010",
        "out": "query.routing.vllm.task1.json",
        "cache": "query.routing.vllm.cache.jsonl",
        # vLLM batches continuously, so unlike Ollama the agent-ordered work
        # below does not have to run one call at a time to keep its prefix.
        "workers": 8,
    },
}

# Text only, a couple of hundred tokens in and under fifty out, so the context
# never comes near the 4096 Ollama defaults to. num_ctx is left alone here on
# purpose -- unlike the frame prompts, this stage has nothing to lose to it.
DEFAULT_NUM_PREDICT = 160

# Classification, not generation. The diversity that helps a paraphrase set
# only adds noise to a yes/no verdict, so sampling is pinned and every call
# carries a seed derived from the question and the agent name: a re-run
# reproduces the routing exactly.
DEFAULT_TEMPERATURE = 0.0

# Below this the verdict is treated as a "no". A specialist that is only 40%
# sure the question needs the soundtrack is not grounds for loading one.
DEFAULT_THRESHOLD = 0.5

# The kinds of answer the 2025 and 2026 question sets actually ask for. "other"
# exists so the model has somewhere to put an outlier instead of forcing one of
# the real labels onto it.
QUESTION_TYPES = [
    "count",
    "yes_no",
    "entity",
    "action",
    "place",
    "time",
    "reason",
    "manner",
    "summary",
    "other",
]

SYSTEM_PROMPT = (
    "You are one specialist on a video question answering team. You judge a "
    "single narrow property of a question and nothing else. You never answer "
    "the question itself. You reply with JSON only."
)

# Each prompt is written against the mistake that particular judgement invites.
# The recurring one is over-claiming: asked in isolation whether a question
# needs sound, a model will find a way to say yes for almost anything set in
# the world, so every prompt states what belongs to a *different* specialist
# and gives the case that looks like a yes and is not.

VISION_PROMPT = """Does answering this question require WATCHING the video?

Question: {question}

Watching means recognising what is in the picture: people, their clothing and
appearance, objects, animals, vehicles, places and scenery, actions, movement,
colours, and what happens on screen.

It is not watching when the answer is carried only by the soundtrack -- what
someone says, a song, a noise -- and it is not watching when the answer is
words printed on the screen and merely has to be read. Other specialists cover
those two, so leave them out of this judgement.

Reply with JSON: {{"needed": true or false, "confidence": 0.0 to 1.0,
"reason": "at most 15 words"}}"""

AUDIO_PROMPT = """Does answering this question require LISTENING to the video?

Question: {question}

Listening means the answer lives in the soundtrack: what someone says, asks,
answers, is told or is called; the words or style of a song; what kind of music
is playing; which language is spoken; a noise made by a person, animal, machine
or instrument; a tone of voice; or whether there is any sound at all.

It is not listening when the answer can be seen. A question about a man playing
a guitar is visual unless it asks what he sounds like or what he is singing. A
question about someone talking is visual if it asks who they talk to, and aural
only if it asks what was said.

Reply with JSON: {{"needed": true or false, "confidence": 0.0 to 1.0,
"reason": "at most 15 words"}}"""

OCR_PROMPT = """Does answering this question require READING TEXT that appears
in the picture?

Question: {question}

That means words rendered on screen and readable off it: signs, banners and
posters, shop, street and place names, number plates, scoreboards and clocks,
prices, packaging, logos and brand names, slogans printed on clothing, opening
and closing titles, captions and subtitles, and text on a phone, screen or
computer display.

The test is whether the answer is the words themselves, or cannot be reached
without knowing what they say. Merely mentioning something that carries writing
-- a document, a book, a letter, a phone, a shop, a sign -- is not enough: "what
does he do with the document after printing it" is an action to be watched, and
only "what does the document say" is a reading question.

It is not reading when the answer is a thing, a person or an action to be
recognised rather than words to be spelled out, and it is not reading when the
words are spoken aloud rather than written. Naming a well known object or place
from its appearance is recognition, not reading.

Reply with JSON: {{"needed": true or false, "confidence": 0.0 to 1.0,
"reason": "at most 15 words"}}"""

TEMPORAL_PROMPT = """Does answering this question depend on WHEN something
happens, or in what ORDER?

Question: {question}

It does when the question points at a moment or a sequence: what happens after
or before something else, at the beginning or at the end, first, next or last,
what happens while something else is going on, what changes over the video, or
how many times something happens. Such a question cannot be settled from one
still frame -- it needs the video in order, with timestamps.

It does not when the question asks about a state or a fact that holds
throughout, such as what someone is wearing, where the scene is set, or how
many people are present.

Reply with JSON: {{"needed": true or false, "confidence": 0.0 to 1.0,
"reason": "at most 15 words"}}"""

INFERENCE_PROMPT = """Does this question need INFERENCE, or is it settled by
observation alone?

Question: {question}

It needs inference when the answer is never directly shown, said or written and
has to be worked out from what is: why someone does something, what they intend
or are trying to achieve, what they feel, what is likely or probably happening,
what a thing is used for, what a scene is meant to advertise or teach, or what
the video as a whole is about.

It does not when the answer is there to be observed and only has to be reported
-- the colour of a coat, the number of people at a table, what someone hands
over, what a sign says. "What happens", "what does he do" and "what happens
next" are observation, not inference: the event is on screen and only has to be
described. Reach for inference when the answer is nowhere in the video and has
to be supplied from understanding it.

Reply with JSON: {{"needed": true or false, "confidence": 0.0 to 1.0,
"reason": "at most 15 words"}}"""

QTYPE_PROMPT = """What kind of answer does this question ask for?

Question: {question}

Choose exactly one label:
- count: a number, including "how many times"
- yes_no: yes or no
- entity: the name or identity of a thing, person, animal, place or object
- action: what someone or something does, or what happens
- place: where something is or where it happens
- time: when something happens, or in what order
- reason: why something happens, or what it is for
- manner: how something is done, or how it looks, sounds or feels
- summary: what the whole video or scene is about
- other: none of these fits

Judge what is being asked for, not how the sentence opens: "What happens after
he sits down" is action, "What is he doing that for" is reason, and "Identify
the animal" is entity.

Reply with JSON: {{"type": "one of the labels above", "confidence": 0.0 to 1.0,
"reason": "at most 15 words"}}"""


def verdict_schema() -> dict:
    """Reply shape for the five yes/no specialists."""
    return {
        "type": "object",
        "properties": {
            "needed": {"type": "boolean"},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["needed", "confidence", "reason"],
    }


def qtype_schema() -> dict:
    """Reply shape for the classifier, with the label set pinned by the schema."""
    return {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": QUESTION_TYPES},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["type", "confidence", "reason"],
    }


# name -> (prompt, schema builder, kind). "kind" tells parse() which field
# carries the verdict; everything else about an agent lives in its prompt.
AGENTS: dict[str, tuple[str, callable, str]] = {
    "vision": (VISION_PROMPT, verdict_schema, "verdict"),
    "audio": (AUDIO_PROMPT, verdict_schema, "verdict"),
    "ocr": (OCR_PROMPT, verdict_schema, "verdict"),
    "temporal": (TEMPORAL_PROMPT, verdict_schema, "verdict"),
    "inference": (INFERENCE_PROMPT, verdict_schema, "verdict"),
    "qtype": (QTYPE_PROMPT, qtype_schema, "label"),
}

# The three that decide what evidence gets loaded. The other three describe how
# to use it, and never add an input on their own.
MODALITIES = ("vision", "audio", "ocr")

# What a routing decision means to the answering stage: which artefacts to read
# and which prompt to build. Named for what is on disk -- frames come from
# VQA/extract.frames.py, transcripts from VQA/transcribe.whisper.py.
PIPELINE = {
    "vision": "frames",
    "audio": "asr_transcript",
    "ocr": "frame_ocr",
    "temporal": "dense_timestamped_frames",
    "inference": "reasoning_prompt",
}


def normalize(text: str) -> str:
    """Collapse whitespace and case, for cache keys and duplicate detection."""
    return " ".join(text.split()).casefold()


def seed_for(question: str, agent: str) -> int:
    """Stable per question and per agent, so a re-run reproduces the routing.

    The agent name is in the digest as well as the question: without it the six
    calls about one question would share a seed, and identical sampling state
    across six different prompts is a correlation nobody asked for.
    """
    digest = hashlib.sha256(f"{agent}\x00{question}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


class Ollama:
    """Minimal chat client. One session per worker keeps the connection warm."""

    def __init__(self, host: str, model: str, temperature: float,
                 timeout: float, think: bool, num_predict: int) -> None:
        self.url = host.rstrip("/") + "/api/chat"
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.think = think
        self.num_predict = num_predict
        self.local = threading.local()

    @property
    def session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def judge(self, prompt: str, schema: dict, seed: int) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "format": schema,
            "stream": False,
            # A one-bit verdict off a single sentence gains nothing from a
            # reasoning pass, and thinking tokens come out of num_predict --
            # the budget below leaves room for the JSON and little else.
            "think": self.think,
            "options": {
                "temperature": self.temperature,
                "seed": seed,
                "num_predict": self.num_predict,
            },
        }

        response = self.session.post(self.url, json=payload,
                                     timeout=self.timeout)
        response.raise_for_status()

        body = response.json()
        content = body["message"].get("content") or ""

        if not content:
            reason = body.get("done_reason")
            hint = (" -- raise --num-predict, or drop --think"
                    if reason == "length" else "")
            raise ValueError(f"empty content (done_reason={reason}){hint}")

        parsed = json.loads(content)

        if not isinstance(parsed, dict):
            raise ValueError(f"reply is {type(parsed).__name__}, not an object")

        return parsed


class VLLM:
    """OpenAI-compatible client for vLLM. Same contract as Ollama above.

    The prompts, schemas and per-agent seeds are shared with the Ollama path;
    only the wire format differs -- the schema travels in ``response_format``
    closed with ``additionalProperties: false``, the sampler options are flat
    rather than nested under ``options``, thinking is turned off through the
    chat template variable Gemma 4 reads, and the reply arrives under
    ``choices[0]``. Sampling is still pinned at temperature 0.0, so a re-run
    reproduces the routing.
    """

    name = "vllm"

    def __init__(self, host: str, model: str, temperature: float,
                 timeout: float, think: bool, num_predict: int) -> None:
        self.url = host.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.think = think
        self.num_predict = num_predict
        self.local = threading.local()

    @property
    def session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def judge(self, prompt: str, schema: dict, seed: int) -> dict:
        constraint = dict(schema)
        constraint["additionalProperties"] = False

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.num_predict,
            "temperature": self.temperature,
            "seed": seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": constraint},
            },
            "chat_template_kwargs": {"enable_thinking": self.think},
        }

        response = self.session.post(self.url, json=payload,
                                     timeout=self.timeout)

        # A 400 is the server refusing the request itself; it will refuse it
        # identically on a retry.
        if response.status_code == 400:
            raise ValueError(f"400: {response.text[:300]}")

        response.raise_for_status()

        choice = response.json()["choices"][0]
        content = (choice["message"].get("content") or "").strip()

        if not content:
            reason = choice.get("finish_reason")
            hint = (" -- raise --num-predict, or drop --think"
                    if reason == "length" else "")
            raise ValueError(f"empty content (finish_reason={reason}){hint}")

        parsed = json.loads(content)

        if not isinstance(parsed, dict):
            raise ValueError(f"reply is {type(parsed).__name__}, not an object")

        return parsed


def preflight(host: str, model: str, timeout: float = 10.0) -> None:
    """Refuse to start unless the vLLM server is up and serving this model.

    Without it an absent server fails every judgement one at a time and the
    run ends with an empty cache and a wall of FAILED lines. WSL2 tears the
    service down whenever the last session exits -- VQA/GEMMA4_VLLM_SETUP.md §3.
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


def parse(reply: dict, kind: str) -> dict:
    """Normalize one agent's reply into the record kept on disk.

    The schema constrains the shape, not the values: confidence comes back
    outside 0-1 often enough to be worth clamping, and a label outside the enum
    is treated as a parse failure rather than quietly routed as "other".
    """
    confidence = float(reply.get("confidence", 0.0))
    confidence = min(max(confidence, 0.0), 1.0)
    reason = " ".join(str(reply.get("reason", "")).split())

    if kind == "label":
        label = str(reply.get("type", "")).strip().lower()

        if label not in QUESTION_TYPES:
            raise ValueError(f"unknown question type {label!r}")

        return {"type": label, "confidence": confidence, "reason": reason}

    return {
        "needed": bool(reply.get("needed", False)),
        "confidence": confidence,
        "reason": reason,
    }


def plan(signals: dict[str, dict], threshold: float,
         always_vision: bool = False) -> dict:
    """Combine the specialists' verdicts into a routing decision.

    Deliberately plain: the model's job ended at the six independent judgements,
    and how they add up is a policy this file owns and can be re-run over the
    cache with --rebuild when the policy changes.
    """
    def fired(agent: str) -> bool:
        signal = signals.get(agent)
        return bool(signal
                    and signal.get("needed")
                    and signal["confidence"] >= threshold)

    required = [m for m in MODALITIES if fired(m)]
    defaulted = False

    # A question can legitimately route to sound or on-screen text alone: "what
    # does the first caption say" needs neither frames nor the model's attention
    # spent on them. But the pure routes give up the frames that pick out *which*
    # man was at the bar, so whether to keep them anyway is a downstream call --
    # hence the flag, applied here and re-runnable over the cache.
    if always_vision and "vision" not in required:
        required.insert(0, "vision")

    # The floor for a video question is the video. If every specialist declined,
    # something was misjudged, and frames are the least wrong thing to load.
    if not required:
        required = ["vision"]
        defaulted = True

    temporal = fired("temporal")
    inference = fired("inference")

    qtype = signals.get("qtype")
    question_type = qtype["type"] if qtype else "unknown"

    # Order matters to the answering prompt: evidence first, in the order it
    # gets laid into the context, then the strategy flags, then synthesis.
    pipeline = [PIPELINE[m] for m in required]

    if temporal:
        pipeline.append(PIPELINE["temporal"])

    if inference:
        pipeline.append(PIPELINE["inference"])

    pipeline.append("answer_synthesis")

    return {
        "Question_Type": question_type,
        "Required_Inputs": required,
        "Route": "+".join(required),
        "Temporal": temporal,
        "Inference": inference,
        "Defaulted": defaulted,
        "Pipeline": pipeline,
    }


def load_queries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_cache(path: Path) -> dict[tuple[str, str], dict]:
    """Read the JSONL cache, keyed by (normalized question, agent).

    Later records win, so re-running an agent with --force supersedes the
    earlier verdict without having to rewrite the file.
    """
    cache: dict[tuple[str, str], dict] = {}

    if not path.exists():
        return cache

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                key = (normalize(record["question"]), record["agent"])
                cache[key] = record
            except (json.JSONDecodeError, KeyError):
                print(f"  skipping malformed cache line {line_no}",
                      file=sys.stderr)

    return cache


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Decide which inputs each Task 1 question needs."
    )

    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES,
                    help="released query JSON (default: %(default)s)")
    ap.add_argument("--out", type=Path,
                    help="assembled per-query routing (default: the backend's)")
    ap.add_argument("--cache", type=Path,
                    help="JSONL work cache, one line per question and agent "
                         "(default: the backend's)")

    ap.add_argument("--backend", choices=sorted(BACKENDS),
                    default=DEFAULT_BACKEND,
                    help="which server judges: vllm in WSL2, or ollama on "
                         "Windows (default: %(default)s). Sets the default "
                         "model, host, workers, cache and output")
    ap.add_argument("--model", help="model tag (default: the backend's)")
    ap.add_argument("--host", help="server host (default: the backend's)")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help="sampling temperature (default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-request timeout in seconds (default: %(default)s)")
    ap.add_argument("--think", action="store_true",
                    help="let a reasoning model think first; slower, and needs "
                         "a much larger --num-predict")
    ap.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT,
                    help="token budget per reply (default: %(default)s)")

    ap.add_argument("--agents", nargs="+", choices=sorted(AGENTS),
                    default=sorted(AGENTS),
                    help="specialists to run (default: all of them)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="confidence below which a yes counts as a no "
                         "(default: %(default)s)")
    ap.add_argument("--always-vision", action="store_true",
                    help="keep frames on every query, even where only sound or "
                         "on-screen text was asked for; costs context, saves "
                         "the questions whose subject has to be seen to be "
                         "picked out")
    ap.add_argument("--workers", type=int,
                    help="concurrent requests (default: 1 on ollama, which "
                         "needs OLLAMA_NUM_PARALLEL to match; 8 on vllm)")

    ap.add_argument("--limit", type=int, help="process at most N questions")
    ap.add_argument("--only", type=int, nargs="+", metavar="Q_ID",
                    help="process only the questions behind these Q_IDs")
    ap.add_argument("--force", action="store_true",
                    help="re-judge questions already in the cache")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the output JSON from the cache, no inference")

    return ap.parse_args(argv)


def assemble(queries: list[dict], cache: dict[tuple[str, str], dict],
             threshold: float,
             always_vision: bool) -> tuple[list[dict], int, int]:
    """Fan the per-question verdicts back out to every Q_ID."""
    records: list[dict] = []
    unrouted = 0
    partial = 0

    for query in queries:
        question = query["Question"]
        key = normalize(question)

        signals = {}

        for agent in AGENTS:
            record = cache.get((key, agent))

            if record:
                signals[agent] = record["signal"]

        if not signals:
            unrouted += 1
        elif len(signals) < len(AGENTS):
            partial += 1

        decision = plan(signals, threshold, always_vision)

        records.append(
            {
                "Q_ID": query["Q_ID"],
                "Video_ID": query["Video_ID"],
                "Question": question,
                **decision,
                "Signals": signals,
            }
        )

    return records, unrouted, partial


def summarize(records: list[dict]) -> None:
    """Print the routing distribution -- the point of the whole stage."""
    routes = Counter(r["Route"] for r in records)
    types = Counter(r["Question_Type"] for r in records)
    modality = Counter()

    for record in records:
        for name in record["Required_Inputs"]:
            modality[name] += 1

    total = max(len(records), 1)

    def table(title: str, counts: Counter) -> None:
        print(f"\n{title}")

        for name, count in counts.most_common():
            print(f"  {name:<28} {count:5d}  {100 * count / total:5.1f}%")

    table("required inputs (queries needing each)", modality)
    table("routes", routes)
    table("question types", types)

    strategy = Counter()
    strategy["temporal (order or moment)"] = sum(
        1 for r in records if r["Temporal"])
    strategy["inference (not observed)"] = sum(
        1 for r in records if r["Inference"])
    strategy["defaulted to vision"] = sum(
        1 for r in records if r["Defaulted"])
    table("strategy flags", strategy)


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

    queries = load_queries(args.queries)
    cache = load_cache(args.cache)

    print(f"backend: {args.backend} | {args.model} at {args.host} | "
          f"{args.workers} worker(s)")
    print(f"cache: {args.cache.name} | out: {args.out.name}")

    # One entry per distinct question, in first-seen order, carrying its Q_IDs.
    questions: dict[str, dict] = {}

    for query in queries:
        entry = questions.setdefault(
            normalize(query["Question"]),
            {"question": query["Question"], "q_ids": []},
        )
        entry["q_ids"].append(query["Q_ID"])

    cached = sum(1 for k in questions for a in args.agents if (k, a) in cache)
    print(f"queries: {len(queries)} | distinct questions: {len(questions)} | "
          f"agents: {len(args.agents)} | judgements cached: {cached}"
          f"/{len(questions) * len(args.agents)}")

    if not args.rebuild:
        selected = list(questions.items())

        if args.only:
            wanted = set(args.only)
            selected = [(k, e) for k, e in selected
                        if wanted & set(e["q_ids"])]

        if args.limit:
            selected = selected[:args.limit]

        # Grouped by agent, not by question: with one server slot the calls for
        # a specialist then run back to back and its system and instruction
        # prefix stays in the KV cache instead of being evicted six times over.
        todo = [
            (agent, entry)
            for agent in args.agents
            for key, entry in selected
            if args.force or (key, agent) not in cache
        ]

        print(f"to judge: {len(todo)}")

        if todo:
            if args.backend == "vllm":
                preflight(args.host, args.model)
                client = VLLM(args.host, args.model, args.temperature,
                              args.timeout, args.think, args.num_predict)
            else:
                client = Ollama(args.host, args.model, args.temperature,
                                args.timeout, args.think, args.num_predict)
            lock = threading.Lock()
            done = 0
            failed = 0
            started = time.monotonic()

            args.cache.parent.mkdir(parents=True, exist_ok=True)
            handle = args.cache.open("a", encoding="utf-8")

            def work(unit: tuple[str, dict]) -> None:
                nonlocal done, failed

                agent, entry = unit
                question = entry["question"]
                template, build_schema, kind = AGENTS[agent]
                t0 = time.monotonic()

                try:
                    reply = client.judge(
                        template.format(question=question),
                        build_schema(),
                        seed_for(question, agent),
                    )
                    signal = parse(reply, kind)
                except Exception as exc:  # noqa: BLE001 - one bad judgement
                    with lock:                # must not sink the whole run
                        failed += 1
                        print(f"  FAILED {agent} {question!r}: "
                              f"{type(exc).__name__}: {exc}", file=sys.stderr)
                    return

                record = {
                    "question": question,
                    "agent": agent,
                    "signal": signal,
                    "q_ids": entry["q_ids"],
                    "model": args.model,
                    "elapsed": round(time.monotonic() - t0, 2),
                    "ts": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"),
                }

                with lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    cache[(normalize(question), agent)] = record

                    done += 1
                    rate = done / max(time.monotonic() - started, 1e-9)
                    verdict = signal.get("type", signal.get("needed"))
                    print(f"  [{done}/{len(todo)}] {rate * 60:5.1f}/min "
                          f"{record['elapsed']:5.1f}s {agent:<9} "
                          f"{str(verdict):<7} {question[:52]}")

            try:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    list(pool.map(work, todo))
            except KeyboardInterrupt:
                print("\ninterrupted -- finished judgements are cached, "
                      "re-run to resume", file=sys.stderr)
                return 130
            finally:
                handle.close()

            elapsed = time.monotonic() - started
            print(f"\njudged {done} ({failed} failed) in {elapsed / 60:.1f} min")

    records, unrouted, partial = assemble(queries, cache, args.threshold,
                                          args.always_vision)

    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=4, ensure_ascii=False)
        handle.write("\n")

    print(f"\nwrote {len(records)} query entries to {args.out}")
    print(f"  fully routed ({len(AGENTS)} agents): "
          f"{len(records) - unrouted - partial}")
    print(f"  partially routed:                 {partial}")
    print(f"  not routed yet:                   {unrouted}")

    summarize(records)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Answer the Task 1 questions with Gemma 4 over sampled video frames.

For every query this asks Gemma 4 12B the same question
ten ways -- the released ``Question`` plus the first nine entries of
``query.variations.task1.json`` -- against the frames extracted by
``VQA/extract.frames.py`` and the Whisper transcript in
``videos_TEST_VQA/transcription_openwhisper``. Each call returns its three most
likely answers, so thirty candidates are pooled per query, voted, and ranked
into the ten slots the run format requires.

Output is the submission file ``VQA/validate.ag.run.py`` accepts verbatim::

    [{"Q_ID": 1, "Video_ID": "8384.webm",
      "Answers": [{"Rank": 1, "Answer": "...", "Time": 1.234}, ...]}]

Three things drive the design:

**Images go before the text, and the ten prompts run back to back on one
worker.** Gemma 4 wants image content first, and that layout also makes the
ten calls for a query share a byte-identical prefix, so Ollama's KV cache pays
the frame prefill once instead of ten times. Measured on a 32-frame clip:
5.53 s of prompt evaluation on the first call, 0.59 s on the next. Queries are
grouped by video so the reuse extends across the queries that share one, and
the default is a *single* worker -- see DEFAULT_WORKERS for why more is slower.

**num_ctx must be set.** Ollama defaults to 4096 whatever the model's window
is; 32 frames cost about 6,700 tokens on their own, so the frames would fall
out of context silently and the run would look fine and mean nothing. On the
vLLM backend the window is the server's ``--max-model-len`` instead, and an
overlong prompt is a 400 rather than a silent truncation.

**Either backend serves it.** ``--backend vllm`` (the default) talks to vLLM
in WSL2, which is where the Cosmos and Qwen3-VL servers already live, so the
whole project runs on one side of the machine; ``--backend ollama`` is the
original Windows path that produced ``gemma4.run.task1.json``. They load
different quantisations -- Google's QAT INT4 checkpoint against Ollama's
Q4_K_M -- so each keeps its own cache and run file and the two are separate
runs, not a resumable one. ``VQA/GEMMA4_VLLM_SETUP.md`` has the serve command.

**Filler answers are expensive.** ``eval.vqa.ag.py`` scores a query as the
*mean* METEOR/BERTScore across all ten answers, so a slot filled with junk
costs real score. When the pool yields fewer than ten distinct answers the
shortfall is filled with meaning-preserving rephrasings of the top answer
rather than invented alternatives.

**Time** is the field the track defines as "time to generate this answer", in
seconds. Each answer carries the measured wall clock of the inference call that
produced it, plus its share of reading and base64-encoding that video's frames
(split across the queries on the video and the ten slots, since the encoding is
done once and reused). Rephrased top-up answers carry their top-up call's time
split across them. The offline ffmpeg extraction in ``VQA/extract.frames.py``
is *not* included -- it is a separate preprocessing pass over the corpus, and
at 0.19 s per video it is noise next to inference. Summed over a query, Time is
within a few percent of that query's true wall clock.

Usage:
    python VQA/generate.answers.py --limit 20      # smoke test
    python VQA/generate.answers.py                 # everything, resumable
    python VQA/generate.answers.py --only 1 3 264  # named Q_IDs
    python VQA/generate.answers.py --rebuild       # re-emit run from cache
    python VQA/generate.answers.py --backend ollama  # the Windows path
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = ROOT / "vqa.2026.testingDataset.Task1.json"
DEFAULT_VARIATIONS = ROOT / "query.variations.task1.json"
DEFAULT_ROUTING = ROOT / "query.routing.task1.json"
DEFAULT_FRAMES = ROOT / "videos_TEST_VQA" / "frames"
DEFAULT_TRANSCRIPTS = ROOT / "videos_TEST_VQA" / "transcription_openwhisper"

# Two backends serve the same model. Ollama runs gemma4:12b (Q4_K_M) on
# Windows; vLLM runs Google's own QAT INT4 checkpoint inside WSL2, alongside
# the Cosmos (8008) and Qwen3-VL (8009) servers, which is what lets the whole
# project live on one side of the machine. VQA/GEMMA4_VLLM_SETUP.md has the
# serve command. The two are not the same weights, so they get separate cache
# and run files -- see BACKENDS.
DEFAULT_BACKEND = "vllm"

BACKENDS = {
    "ollama": {
        "model": "gemma4:12b",
        "host": "http://localhost:11434",
        "out": "gemma4.run.task1.json",
        "cache": "answers.task1.cache.jsonl",
        # One worker, deliberately. OLLAMA_NUM_PARALLEL is unset on this box,
        # so the server runs a single slot: extra client threads do not buy
        # concurrency, they just interleave requests from different videos
        # through one KV cache and evict each other's image prefix. Measured
        # over the same 20 queries, --workers 4 took 0.15 h against 0.09 h for
        # --workers 1. Raise both together or not at all.
        "workers": 1,
    },
    "vllm": {
        "model": "google/gemma-4-12B-it-qat-w4a16-ct",
        "host": "http://localhost:8010",
        "out": "gemma4.vllm.run.task1.json",
        "cache": "answers.task1.gemma4.vllm.cache.jsonl",
        # vLLM batches continuously and its prefix cache is shared across
        # requests rather than being a single slot, so the frame prefix a
        # video's ten prompts share survives another video's request landing
        # between them -- the opposite of Ollama, where more workers is
        # slower. Measured over the same 20 queries on this 3090: 161.7 s at
        # 4, 123.8 s at 8, then sharply worse again at 12 (279.9 s) with no
        # preemption and the KV cache half empty, so 8 is the knee and the
        # far side of it is not understood. Re-measure with --limit 20 before
        # a full run.
        "workers": 8,
    },
}

# Ollama only: it defaults to 4096 whatever the model's window is. On vLLM the
# window is fixed at serve time by --max-model-len and this is ignored. 32
# frames cost 280 soft tokens each there, so 16384 covers the worst case in
# this corpus (8,960 of frames, a 1,231-token transcript, the prompt and 256
# of output) with room to spare.
DEFAULT_NUM_CTX = 16384
DEFAULT_NUM_PREDICT = 256

# The model's own default is 1.0, but the ensemble's diversity is supposed to
# come from the ten phrasings, not from sampling noise. At 1.0 a query could
# come back with 30 distinct candidates all at confidence ~0.95, leaving the
# vote with nothing to rank; 0.6 concentrated the pool (19.3 distinct per
# query against 21.9) at identical cost. Judged on vote concentration over 20
# queries, not against ground truth -- the 2025 GT videos are not on disk.
DEFAULT_TEMPERATURE = 0.6

ANSWERS_PER_QUERY = 10
PROMPTS_PER_QUERY = 10
CANDIDATES_PER_PROMPT = 3

# A candidate offered as the model's first choice counts for more than its
# third. Confidence modulates the vote but never cancels it.
POSITION_WEIGHT = (1.0, 0.5, 0.25)

# Ground truth answers in VQA/vqa.gt.csv average 8.4 words, median 8, p90 17,
# max 42, and 17% are a single word. Anything far past that is the model
# narrating instead of answering.
MAX_ANSWER_WORDS = 40
MAX_ANSWER_CHARS = 300

LEAD_IN_RE = re.compile(r"^\s*(?:answer|a)\s*[:\-]\s*", re.IGNORECASE)
ENUM_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")

# A refusal is a legitimate thing for the model to say, but it is never the
# ground truth, so it scores ~0 on METEOR and BERTScore. Ranking one above a
# substantive candidate wastes a slot and drags the NDCG gain vector down, so
# these sort last among the model's own answers -- kept, not discarded.
# "none", "no one" and "zero" are deliberately NOT here: for a counting or
# "who did X" question they are the answer, not a refusal.
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

# Used only when the model and the paraphrase top-up both come up short. Each
# is a defensible non-answer rather than a fabricated fact.
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

SYSTEM_PROMPT = (
    "You answer questions about a video you are shown as a sequence of frames. "
    "You reply with JSON only."
)

PROMPT_TEMPLATE = """{context}Question: {question}

Give the {k} most likely answers, best first.

Rules:
- Answer the question and nothing else. Do not restate the question, do not
  describe the video, and do not explain your reasoning.
- Keep every answer very short: a phrase or one short sentence. Most good
  answers are under ten words.
- The answers must be genuinely different possibilities, not rewordings of
  each other.
- Use the frames and the transcript. If neither shows the answer, say
  "unknown" rather than guessing.
{extra}
Reply with JSON: {{"answers": [{slots}], "confidence": 0.0}}
where confidence is how confident you are in the first answer, 0.0 to 1.0."""

# Appended only when the router says so. Same wording as the Qwen3-VL pass, so
# a candidate from either model was asked the same thing.
OCR_RULE = ("- The answer is text rendered on screen. Read it exactly as "
            "written, and do not paraphrase it.\n")
TEMPORAL_RULE = ("- The answer depends on when something happens or in what "
                 "order. The frames are in time order; use their timestamps.\n")

TOPUP_TEMPLATE = """Rewrite this answer in {k} different ways.

Question: {question}
Answer: {answer}

Rules:
- Every rewrite must mean exactly the same thing as the answer above.
- Add no new facts, and drop none of the ones it states.
- Keep them short, in the style of the original.
- Each rewrite must differ from the original and from all the others.
{existing}
Reply with JSON: {{"answers": ["...", ...]}} holding exactly {k} strings."""


def answer_schema(k: int, with_confidence: bool) -> dict:
    """Constrain the reply to exactly `k` strings, plus a confidence."""
    schema = {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": k,
                "maxItems": k,
            }
        },
        "required": ["answers"],
    }

    if with_confidence:
        schema["properties"]["confidence"] = {"type": "number"}
        schema["required"].append("confidence")

    return schema


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


def seed_for(q_id: int, index: int) -> int:
    """Stable per-call seed, so a re-run reproduces the same run file."""
    digest = hashlib.sha256(f"{q_id}|{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def mmss(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def build_context(frames: dict | None, transcript: dict | None,
                  routed_out: bool = False) -> str:
    """The block that precedes the question, identical across a query's prompts.

    ``frames`` and ``transcript`` arrive already filtered by the query's plan,
    so this only has to describe what is actually attached. ``routed_out`` says
    the frames were withheld on purpose rather than missing -- the two need
    different wording, because "the video is unavailable" invites a guess and
    "this question is about the audio" does not.
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
            "frames. Answer from the transcript below."
        )
    else:
        parts.append(
            "The video for this question is unavailable, so you are shown no "
            "frames. Answer from the question alone, as plausibly as you can."
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


def parse_json_object(text: str) -> dict:
    """Parse a reply that should be JSON, tolerating prose wrapped around it.

    Both servers are asked to constrain the reply to the schema, so this
    normally just calls json.loads. The salvage path matters when a server is
    started without its structured-output backend, or when a reasoning trace
    leaks into content -- one recoverable reply is worth more than one lost
    prompt out of a query's ten.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")

    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")

    return json.loads(text[start:end + 1])


def load_routing(path: Path) -> dict[int, dict]:
    """Q_ID -> routing verdict from query.routing.task1.json."""
    if not path.exists():
        return {}

    return {int(entry["Q_ID"]): entry for entry in load_json(path)}


def plan_for(routing: dict | None, ignore_routing: bool) -> dict:
    """Which inputs this query is sent, and which extra rules it gets.

    The Qwen3-VL pass has taken a routed plan since 14 Aug; this brings Gemma 4
    onto the same footing so the two models' candidates are comparable. Simpler
    here: the frames are pre-extracted at a fixed count, so there is no
    sampling-density or token-budget decision to make -- only what goes in.

    With --ignore-routing (or no routing entry) everything is attached, which is
    the behaviour every Gemma 4 run before this one had.
    """
    if ignore_routing or not routing:
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


def answer_logprobs(logprobs: list | None, content: str,
                    answers: list) -> list[dict | None]:
    """Per-answer token statistics, aligned with `answers`.

    The reply is grammar-constrained by ``response_format``, so the structural
    tokens (``{``, ``"answers"``, ``[``) are forced to probability ~1 and say
    nothing about the model's belief. Only the tokens inside an answer string
    carry signal, so each answer is located in the raw content and scored over
    the tokens overlapping its span.

    Returns one dict per answer, or None where the span could not be found --
    a missing score is honest; a whole-sequence average would mostly be
    measuring the grammar.
    """
    if not logprobs or not content:
        return [None] * len(answers)

    # Character span of every token, by walking the same string the server
    # decoded. Tokens carry their own text, so this reconstructs the offsets
    # without needing the tokenizer.
    spans: list[tuple[int, int, float]] = []
    cursor = 0

    for token in logprobs:
        text = token.get("token") or ""
        value = token.get("logprob")

        if value is None:
            cursor += len(text)
            continue

        spans.append((cursor, cursor + len(text), float(value)))
        cursor += len(text)

    out: list[dict | None] = []
    search_from = 0

    for answer in answers:
        needle = str(answer)
        # json.dumps gives the escaped form the model actually emitted; the
        # bare form covers a salvaged reply that was never valid JSON.
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

    def ask(self, prompt: str, k: int, seed: int, images: list[str] | None,
            with_confidence: bool) -> tuple[dict, float]:
        """One call. Returns the parsed reply and the wall time it took."""
        message: dict = {"role": "user", "content": prompt}

        # Gemma 4 wants image content before the text, and putting it first is
        # also what lets the KV prefix cache carry across a query's ten calls.
        if images:
            message["images"] = images

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                message,
            ],
            "format": answer_schema(k, with_confidence),
            "stream": False,
            # Paraphrasing and short-answer VQA both gain nothing from the
            # reasoning pass, and it cost 19 s against 5 s in the variations run.
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

                parsed = json.loads(content)
                # Ollama's chat API exposes no logprobs, so this backend
                # contributes votes and confidence but no probability score.
                parsed["_logprobs"] = [None] * len(parsed.get("answers") or [])

                return parsed, time.monotonic() - started

            except (requests.RequestException, ValueError,
                    json.JSONDecodeError) as exc:
                last = exc

                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"{type(last).__name__}: {last}")


class VLLM:
    """OpenAI-compatible client for vLLM. Same ask() contract as Ollama.

    Everything that decides an answer -- the prompts, the schema, the seeds,
    the ten-phrasing ensemble -- is shared with the Ollama path; only the wire
    format differs:

    - frames become ``image_url`` content parts carrying the same base64
      bytes, still ahead of the text, so a query's ten prompts still share a
      byte-identical prefix. vLLM's prefix cache is what pays the frame
      prefill once instead of ten times here, and unlike Ollama's single slot
      it survives another worker's request landing in between -- which is why
      this backend defaults to four workers and Ollama to one.
    - the schema goes in ``response_format``, closed with
      ``additionalProperties: false`` because vLLM compiles it into a grammar
      rather than handing it to llama.cpp.
    - there is no num_ctx. The window is the server's ``--max-model-len``, so
      an overlong prompt is a 400 rather than silently dropped tokens.
    - thinking is turned off through the chat template, which is where Gemma 4
      reads ``enable_thinking``, rather than a top-level flag.
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

    def ask(self, prompt: str, k: int, seed: int, images: list[str] | None,
            with_confidence: bool) -> tuple[dict, float]:
        """One call. Returns the parsed reply and the wall time it took."""
        content: list[dict] = []

        if images:
            content += [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                for image in images
            ]

        content.append({"type": "text", "text": prompt})

        schema = answer_schema(k, with_confidence)
        schema["additionalProperties"] = False

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
                "json_schema": {"name": "answers", "schema": schema},
            },
            "chat_template_kwargs": {"enable_thinking": self.think},
            # Per-token logprobs for the probability score. top_logprobs is 0
            # because only the sampled token's own value is wanted -- the
            # alternatives would be a much larger response for no use here.
            "logprobs": True,
            "top_logprobs": 0,
        }

        last: Exception | None = None

        for attempt in range(self.retries):
            started = time.monotonic()

            try:
                response = self.session.post(self.url, json=payload,
                                             timeout=self.timeout)

                # A 400 is the server refusing the request itself -- a prompt
                # past max-model-len, a schema this build will not compile --
                # and it will refuse it again just as fast. Retrying only
                # spends the timeout three times, so this leaves the loop.
                if response.status_code == 400:
                    raise RuntimeError(f"400: {response.text[:300]}")

                response.raise_for_status()

                choice = response.json()["choices"][0]
                content_out = (choice["message"].get("content") or "").strip()

                if not content_out:
                    reason = choice.get("finish_reason")
                    raise ValueError(f"empty content (finish_reason={reason})")

                parsed = parse_json_object(content_out)
                tokens = (choice.get("logprobs") or {}).get("content")
                parsed["_logprobs"] = answer_logprobs(
                    tokens, content_out, parsed.get("answers") or [])

                return parsed, time.monotonic() - started

            except (requests.RequestException, ValueError, KeyError, IndexError,
                    json.JSONDecodeError) as exc:
                last = exc

                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"{type(last).__name__}: {last}")


def pool_candidates(replies: list[dict]) -> list[dict]:
    """Vote over every candidate from every prompt, best first.

    Each reply is {"answers": [...], "confidence": float, "index": int,
    "elapsed": float, "logprobs": [...]}. A first-choice candidate outweighs a
    third-choice one, and the call's confidence modulates the vote without ever
    cancelling it.

    Every signal an occurrence carried is kept on the cluster, not just folded
    into ``score``: ``fuse.answers.py`` re-ranks across both models from these
    fields, and the weighting is meant to be changeable without re-running any
    inference.
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

            # Keep the surface form of the strongest single occurrence, and
            # the time of the call that produced it.
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

    # Deterministic: refusals last, then score, then the prompt that found it
    # first, then the text.
    return sorted(clusters.values(),
                  key=lambda c: (c["refusal"], -c["score"], c["first_index"],
                                 c["answer"]))


def fill_slots(client: Ollama, question: str, ranked: list[dict], want: int,
               q_id: int) -> tuple[list[dict], dict]:
    """Take the ranked pool up to `want` distinct answers.

    Shortfalls are covered by rephrasing the top answer, because the evaluation
    averages its similarity metric over all ten slots -- a wrong-but-different
    filler costs score, a correct rewording does not.
    """
    chosen = ranked[:want]
    seen = {normalize(c["answer"]) for c in chosen}
    stats = {"pooled": len(ranked), "topped_up": 0, "fallback": 0}

    if len(chosen) < want and chosen:
        missing = want - len(chosen)
        existing = "\n".join(f"- {c['answer']}" for c in chosen)

        try:
            reply, elapsed = client.ask(
                TOPUP_TEMPLATE.format(
                    k=missing,
                    question=question,
                    answer=chosen[0]["answer"],
                    existing=f"\nThese already exist, so differ from them:\n"
                             f"{existing}\n",
                ),
                k=missing,
                seed=seed_for(q_id, 900),
                images=None,
                with_confidence=False,
            )
            share = elapsed / max(missing, 1)

            for raw in reply.get("answers") or []:
                text = tidy(raw)
                key = normalize(text)

                if not text or key in seen:
                    continue

                seen.add(key)
                chosen.append({"answer": text, "score": 0.0, "votes": 0,
                               "elapsed": share, "source": "topup"})
                stats["topped_up"] += 1

                if len(chosen) == want:
                    break

        except RuntimeError as exc:
            print(f"  Q_ID {q_id}: top-up failed ({exc})", file=sys.stderr)

    for text in FALLBACK_ANSWERS:
        if len(chosen) >= want:
            break

        key = normalize(text)

        if key in seen:
            continue

        seen.add(key)
        chosen.append({"answer": text, "score": 0.0, "votes": 0,
                       "elapsed": 0.0, "source": "fallback"})
        stats["fallback"] += 1

    return chosen[:want], stats


def answer_query(client: Ollama, query: dict, prompts: list[str],
                 context: str, images: list[str] | None, candidates: int,
                 want: int, plan: dict, prep_share: float = 0.0) -> dict:
    """Run every prompt phrasing for one query and rank what comes back.

    ``prep_share`` is this answer's share of the frame preparation that fed the
    inference -- see the Time note in main().
    """
    q_id = query["Q_ID"]
    replies: list[dict] = []
    failures = 0
    slots = ", ".join(['"..."'] * candidates)

    extra = ""

    if plan["ocr"]:
        extra += OCR_RULE
    if plan["temporal"]:
        extra += TEMPORAL_RULE

    for index, question in enumerate(prompts):
        try:
            reply, elapsed = client.ask(
                PROMPT_TEMPLATE.format(context=context, question=question,
                                       k=candidates, slots=slots, extra=extra),
                k=candidates,
                seed=seed_for(q_id, index),
                images=images,
                with_confidence=True,
            )
        except RuntimeError as exc:
            failures += 1
            print(f"  Q_ID {q_id} prompt {index}: {exc}", file=sys.stderr)
            continue

        replies.append({
            "index": index,
            "answers": reply.get("answers") or [],
            "confidence": reply.get("confidence"),
            "logprobs": reply.get("_logprobs") or [],
            "elapsed": elapsed,
        })

    ranked = pool_candidates(replies)
    chosen, stats = fill_slots(client, query["Question"], ranked, want, q_id)

    return {
        "Q_ID": q_id,
        "Video_ID": query["Video_ID"],
        "answers": [
            {
                "Rank": rank,
                "Answer": item["answer"],
                # "time to generate this answer", per the track guidelines:
                # the wall clock of the call that produced it, plus this
                # answer's share of preparing the frames that call consumed.
                "Time": round(max(item["elapsed"], 0.0) + prep_share, 3),
                "votes": item["votes"],
                "score": round(item["score"], 4),
                "source": item.get("source", "model"),
            }
            for rank, item in enumerate(chosen, 1)
        ],
        "backend": client.name,
        "model": client.model,
        "n_prompts": len(prompts),
        "n_replies": len(replies),
        "n_prompt_failures": failures,
        "n_frames": len(images) if images else 0,
        "has_transcript": "Spoken audio" in context,
        "pooled_unique": stats["pooled"],
        "topped_up": stats["topped_up"],
        "fallback": stats["fallback"],
        "routed": plan["routed"],
        "rescued": plan["rescued"],
        "ocr": plan["ocr"],
        "temporal": plan["temporal"],
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
        "prep_share": round(prep_share, 4),
        "elapsed": round(sum(r["elapsed"] for r in replies), 2),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def preflight(host: str, model: str, timeout: float = 10.0) -> None:
    """Refuse to start unless the vLLM server is up and serving this model.

    A server that is not there is not a loud failure in this pipeline: every
    call fails, fill_slots pads all ten slots from FALLBACK_ANSWERS, and the
    run completes looking finished while meaning nothing. That happened twice
    during bring-up -- the WSL2 distro tears the service down whenever the
    last session exits -- and both times the only symptom was FALLBACK(10) in
    a progress line that is easy to scroll past. Ten seconds of checking is
    cheaper than finding out ten hours later.
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
                {"Rank": answer["Rank"],
                 "Answer": answer["Answer"],
                 "Time": answer["Time"]}
                for answer in record["answers"][:want]
            ],
        })

    return run, missing


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Answer the Task 1 questions with Gemma 4 over video frames."
    )

    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--variations", type=Path, default=DEFAULT_VARIATIONS)
    ap.add_argument("--routing", type=Path, default=DEFAULT_ROUTING,
                    help="per-query input routing from route.query.inputs.py; "
                         "decides whether the frames and the transcript are "
                         "attached (default: %(default)s)")
    ap.add_argument("--ignore-routing", action="store_true",
                    help="attach every input regardless of the routing -- the "
                         "behaviour of every run before 15 Aug, and the "
                         "ablation a routed run has to beat")
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
                    help="which server answers: vllm in WSL2, or ollama on "
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

    ap.add_argument("--prompts", type=int, default=PROMPTS_PER_QUERY,
                    help="prompt phrasings per query: the original question "
                         "plus N-1 variations (default: %(default)s)")
    ap.add_argument("--candidates", type=int, default=CANDIDATES_PER_PROMPT,
                    help="candidate answers per prompt (default: %(default)s)")
    ap.add_argument("--answers-per-query", type=int, default=ANSWERS_PER_QUERY,
                    help="slots in the run file (default: %(default)s)")
    ap.add_argument("--workers", type=int,
                    help="concurrent videos (default: 1 on ollama, which "
                         "needs OLLAMA_NUM_PARALLEL to match; 4 on vllm)")

    ap.add_argument("--limit", type=int, help="process at most N queries")
    ap.add_argument("--only", type=int, nargs="+", metavar="Q_ID")
    ap.add_argument("--force", action="store_true",
                    help="re-answer queries already in the cache")
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

    queries = load_json(args.queries)
    cache = load_cache(args.cache)

    # The two backends run different quantisations of the same model, so
    # answers from one are not answers from the other. Separate cache files
    # keep them apart by default; this catches the case where they were
    # deliberately pointed at the same one.
    foreign = sorted({record.get("backend", "ollama") for record in
                      cache.values()} - {args.backend})

    if foreign:
        print(f"  WARNING: {args.cache.name} already holds answers from "
              f"{', '.join(foreign)}; this run would mix weights in one run "
              f"file. Point --cache elsewhere, or --force to re-answer.",
              file=sys.stderr)

    variations = {entry["Q_ID"]: entry.get("Variations") or []
                  for entry in load_json(args.variations)}

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

        # Group by video: the ten calls of a query share an image prefix, and
        # the queries of one video share it with each other. Keeping a video on
        # a single worker is what turns that into cache hits.
        groups: dict[str, list[dict]] = {}

        for query in todo:
            groups.setdefault(query["Video_ID"], []).append(query)

        ordered = [groups[video_id] for video_id in sorted(groups)]

        no_frames = sum(len(g) for g in ordered
                        if Path(g[0]["Video_ID"]).stem not in frame_index)

        print(f"to answer: {len(todo)} queries over {len(ordered)} videos "
              f"({no_frames} queries have no frames)")

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
                prep = 0.0

                if frames:
                    frame_dir = args.frames / Path(video_id).stem
                    prep_started = time.monotonic()
                    try:
                        images = [
                            base64.b64encode(
                                (frame_dir / name).read_bytes()).decode()
                            for name in frames["files"]
                        ]
                        prep = time.monotonic() - prep_started
                    except (OSError, KeyError, TypeError) as exc:
                        print(f"  {video_id}: unreadable frames ({exc})",
                              file=sys.stderr)
                        images = None
                        frames = None
                        prep = 0.0

                # The frames are read and encoded once, then reused by every
                # query on this video and every answer slot, so the cost is
                # split the same way rather than charged to the first answer.
                prep_share = prep / (len(group) * max(args.answers_per_query, 1))

                transcript = load_transcript(args.transcripts, video_id)
                has_speech = bool(transcript and transcript.get("text"))

                for query in group:
                    prompts = [query["Question"]]
                    prompts += variations.get(query["Q_ID"], [])[
                        :max(args.prompts - 1, 0)]

                    plan = plan_for(routing.get(query["Q_ID"]),
                                    args.ignore_routing)

                    # Never send a prompt with no evidence at all: a query the
                    # router sent to the audio of a clip that turns out to have
                    # no speech gets its frames back. Same rescue as the
                    # Qwen3-VL pass.
                    if not plan["frames"] and not has_speech:
                        plan = dict(plan, frames=True, rescued=True)

                    send_images = images if plan["frames"] else None
                    context = build_context(
                        frames if plan["frames"] else None,
                        transcript if plan["transcript"] else None,
                        routed_out=not plan["frames"],
                    )

                    try:
                        record = answer_query(client, query, prompts, context,
                                              send_images, args.candidates,
                                              args.answers_per_query, plan,
                                              prep_share)
                    except Exception as exc:  # noqa: BLE001 - one bad query
                        with lock:            # must not sink a 10-hour run
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
                            flag = f" FALLBACK({record['fallback']})"
                        elif record["n_prompt_failures"]:
                            flag = f" ERR({record['n_prompt_failures']})"

                        print(f"  [{done}/{len(todo)}] {rate * 60:5.1f}/min "
                              f"eta {remaining:4.1f}h  "
                              f"{record['elapsed']:6.1f}s "
                              f"{record['n_frames']:2d}f "
                              f"u{record['pooled_unique']:2d}{flag}  "
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
            print(f"\nanswered {done} queries ({failed} failed) in "
                  f"{elapsed / 3600:.2f} h")

    run, missing = assemble(queries, cache, args.answers_per_query)

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
    pooled = sum(r["pooled_unique"] for r in records) / len(records)

    print(f"\nquality checks over {len(records)} answered queries")
    print(f"  answer words  mean {sum(words) / len(words):4.1f}  "
          f"median {words[len(words) // 2]}  p90 {words[int(0.9 * len(words))]}"
          f"   (ground truth: mean 8.4, median 8, p90 17)")
    print(f"  distinct candidates pooled per query, mean {pooled:.1f} of 30")
    print(f"  queries topped up with rephrasings: "
          f"{sum(1 for r in records if r['topped_up'])}")
    print(f"  queries needing a generic fallback: "
          f"{sum(1 for r in records if r['fallback'])}")
    print(f"  rank-1 answer is \"unknown\":         {unknown}")
    print(f"  queries with a failed prompt call:  "
          f"{sum(1 for r in records if r['n_prompt_failures'])}")
    print(f"  queries answered on a routed plan:  "
          f"{sum(1 for r in records if r.get('routed'))}"
          f"  (rescued {sum(1 for r in records if r.get('rescued'))})")

    scored = sum(1 for r in records for c in (r.get("candidates") or ())
                 if c.get("logprob_mean") is not None)
    total_cands = sum(len(r.get("candidates") or ()) for r in records)

    # A silent loss of logprobs would leave fuse.answers.py ranking on votes
    # alone and looking perfectly healthy while doing it, so it is called out
    # here rather than discovered in the fused run.
    print(f"  candidates carrying a logprob:      {scored}/{total_cands}"
          f"{'  -- CHECK: logprobs are off' if total_cands and not scored else ''}")

    times = [answer["Time"] for r in records for answer in r["answers"]]
    per_query = [sum(a["Time"] for a in r["answers"]) for r in records]
    measured = sum(r["elapsed"] for r in records)

    print(f"  Time per answer: mean {sum(times) / len(times):5.2f}s  "
          f"min {min(times):.2f}s  max {max(times):.2f}s  "
          f"zero-valued {sum(1 for t in times if t == 0)}")
    print(f"  Time per query:  mean {sum(per_query) / len(per_query):5.2f}s  "
          f"total {sum(per_query) / 3600:.2f} h vs "
          f"{measured / 3600:.2f} h actually spent on inference")
    print(f"  queries answered without frames:    "
          f"{sum(1 for r in records if not r['n_frames'])}")
    print(f"  queries answered without audio:     "
          f"{sum(1 for r in records if not r['has_transcript'])}")


if __name__ == "__main__":
    sys.exit(main())

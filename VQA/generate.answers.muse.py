#!/usr/bin/env python3
"""Answer the Task 1 questions with Muse Glimmer 30B (GGUF) on llama.cpp.

Third answering pipeline in this project, after ``generate.answers.py``
(Gemma 4 via Ollama, 30-call paraphrase ensemble) and
``generate.answers.cosmos.py`` (Cosmos-Reason2-2B via vLLM, one reasoning call
over native video). This one targets a ``llama-server`` holding the Q4 GGUF of
meta-models/Muse-Glimmer-30B, and is the first pipeline here that is driven by
``query.routing.task1.json`` rather than sending every input to every query.

Output is the submission file ``VQA/validate.ag.run.py`` accepts verbatim::

    [{"Q_ID": 1, "Video_ID": "8384.webm",
      "Answers": [{"Rank": 1, "Answer": "...", "Time": 1.234}, ...]}]

Four things drive the design:

**Muse Glimmer takes images and text, and nothing else.** The model card is
explicit that there is no video and no audio path, so a video reaches it the
same way it reaches Gemma 4: as the frames ``VQA/extract.frames.py`` already
wrote (long edge 896, 8-32 per clip), with the Whisper transcript folded in as
text. Unlike Gemma 4, the frames are interleaved with their own ``[mm:ss]``
labels rather than dumped as one block, which is the only way a
frames-as-video model can ground a "when does X happen" question.

**The routing file decides what each query is sent.** ``Pipeline`` from
``query.routing.task1.json`` maps directly onto request contents: ``frames``
attaches images, ``asr_transcript`` attaches the transcript,
``dense_timestamped_frames`` sends every extracted frame instead of an even
subsample of ``--max-frames``, ``frame_ocr`` adds the read-the-text-on-screen
instruction, and ``reasoning_prompt`` raises reasoning strength. On this
corpus that skips images for the 168 audio-only queries and the transcript for
the 1839 vision-only ones -- most of the run's savings, and the reason a 30B
model is affordable here at all. ``--ignore-routing`` turns it all off and
sends everything to everything, which is the ablation the routing has to beat.

**Reasoning strength is a per-query dial, not a constant.** Muse Glimmer
accepts low/medium/high/xhigh, and the cost difference is most of the wall
clock. REASONING_POLICY spends it where the router said the query needs
inference or temporal ordering and saves it elsewhere; the generation budget
moves with it, since a trace cut off mid-think returns nothing parseable (the
failure Cosmos hit at 3072 tokens).

**Time** is the field the track defines as "time to generate this answer", in
seconds: each answer carries its share of the inference call that produced it,
plus its share of reading and base64-encoding that video's frames. The offline
ffmpeg extraction is excluded, as in ``generate.answers.py`` -- it is a
separate preprocessing pass over the corpus.

Serving the model (see the GGUF model card; ``--jinja`` is mandatory, the
template is what turns reasoning strength into a system directive)::

    llama-server -m Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \\
        --mmproj mmproj-Muse-Glimmer-30B-Q4_K_M.gguf \\
        -a muse-glimmer-30B -ngl 99 -c 131072 -np 1 \\
        --jinja --cache-reuse 256 --port 8080

``-c`` is split across slots, so raise it with ``-np`` or each slot gets a
fraction of the window. Usage:

    python VQA/generate.answers.muse.py --limit 20      # smoke test
    python VQA/generate.answers.muse.py                 # everything, resumable
    python VQA/generate.answers.muse.py --only 1 3 264  # named Q_IDs
    python VQA/generate.answers.muse.py --rebuild       # re-emit run from cache
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = ROOT / "vqa.2026.testingDataset.Task1.json"
DEFAULT_ROUTING = ROOT / "query.routing.task1.json"
DEFAULT_FRAMES = ROOT / "videos_TEST_VQA" / "frames"
DEFAULT_TRANSCRIPTS = ROOT / "videos_TEST_VQA" / "transcription_openwhisper"
DEFAULT_OUT = ROOT / "muse.run.task1.json"
DEFAULT_CACHE = ROOT / "answers.task1.muse.cache.jsonl"

DEFAULT_MODEL = "muse-glimmer-30B"
"""The ``-a`` alias in the model card's own llama-server invocation. The
server matches on it loosely, but sending the right name keeps /v1/models
honest when more than one GGUF is loaded."""

DEFAULT_BASE_URL = "http://localhost:8080/v1"

# The model card's numbers, not guesses: temperature 1.0 / top_p 0.95 /
# top_k 64. Resisting the urge to drop the temperature the way
# generate.answers.py does for Gemma 4 -- that ensemble wanted a concentrated
# vote across 30 samples, whereas this is one call whose diversity has to come
# from the model actually exploring ten distinct answers.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64

# One worker by default, for the same reason as generate.answers.py: a
# llama-server started with -np 1 has a single slot, so extra client threads
# only interleave requests through one KV cache and evict each other's image
# prefix. Raise this only together with -np on the server, and remember -c is
# split across slots.
DEFAULT_WORKERS = 1

ANSWERS_PER_QUERY = 10

# Frames per query when the router did NOT ask for dense temporal coverage.
# extract.frames.py wrote 8-32 per clip; 16 is its own cap for videos under 60s
# and the point where an even subsample stops losing much.
DEFAULT_MAX_FRAMES = 16

# Rough visual-token cost per 896px frame, used only to warn before a run that
# the frame budget will not fit the server's per-slot context. The model card
# caps a single image at 4096 visual tokens; these frames are well under that,
# but the estimate is deliberately loose -- it exists to catch "32 frames into
# a 4096-token slot", not to be accurate.
TOKENS_PER_FRAME_ESTIMATE = 1024

# Reasoning strength per routing verdict, keyed by (inference, temporal). The
# model card recommends high/xhigh for hard problems and this corpus has 1136
# inference queries and 886 temporal ones, so spending xhigh everywhere would
# dominate the run's wall clock for no gain on "what colour is the car".
REASONING_POLICY = {
    (False, False): "low",
    (False, True): "medium",
    (True, False): "medium",
    (True, True): "high",
}
DEFAULT_REASONING = "medium"
"""Used when a query has no routing entry, and by --ignore-routing."""

STRENGTHS = ("low", "medium", "high", "xhigh")

# Generation budget per strength. A reasoning trace that runs out of tokens
# before its closing tag returns no parseable answer at all -- Cosmos hit
# exactly that on 3/25 queries at 3072 -- so the budget scales with the amount
# of thinking that was asked for.
MAX_TOKENS_BY_STRENGTH = {
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 12288,
}

# Ground truth answers in VQA/vqa.gt.csv average 8.4 words, median 8, p90 17.
MAX_ANSWER_WORDS = 40
MAX_ANSWER_CHARS = 300

LEAD_IN_RE = re.compile(r"^\s*(?:answer|a)\s*[:\-]\s*", re.IGNORECASE)
ENUM_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
JSON_BLOB_RE = re.compile(r"\{.*?\}", re.DOTALL)

# Same pool as the other two pipelines: defensible non-answers rather than
# fabricated facts, used only when the model returns fewer than ten distinct
# candidates. No paraphrase top-up call here -- like the Cosmos script, this
# design pays for one call per query and does not add a second.
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
    "You answer questions about a video you are shown as an ordered sequence "
    "of frames, each labelled with its timestamp in the video. Reason "
    "carefully, then give only the answers asked for."
)

PROMPT_TEMPLATE = """{context}Question: {question}

Give your {k} most likely answers to this question, ranked best first.

Rules:
- Answer the question and nothing else. Do not restate the question, do not
  describe the video, and do not explain your reasoning outside your thinking.
- Keep every answer very short: a phrase or one short sentence. Most good
  answers are under ten words.
- The {k} answers must be genuinely different possibilities, not rewordings
  of each other.
- Use the evidence you were given. If it does not show the answer, say
  "unknown" rather than guessing.{extra}

End your reply with ONLY this JSON on the final line, exactly {k} items ranked
best first: {{"answers": ["...", ...]}}"""

# Appended to the rules when the router flagged the query. There is no OCR
# artefact on disk -- route.query.inputs.py names the step `frame_ocr` for a
# pass that was never built -- so the instruction leans on Muse Glimmer reading
# the frames itself, which is what a 4096-visual-token-per-image model is for.
OCR_RULE = ("- The answer depends on text visible on screen. Read any signs, "
            "captions, labels or subtitles in the frames carefully and quote "
            "them exactly.")
TEMPORAL_RULE = ("- The answer depends on ordering or timing. Use the frame "
                 "timestamps to work out what happens before what.")


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

    language = data.get("language") \
        if data.get("language_source") == "window_sweep" else None

    return {"text": text, "language": language,
            "duration_sec": data.get("duration_sec")}


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


def load_routing(path: Path) -> dict[int, dict]:
    """Q_ID -> routing verdict from query.routing.task1.json."""
    if not path.exists():
        return {}

    return {int(entry["Q_ID"]): entry for entry in load_json(path)}


def plan_for(routing: dict | None, ignore_routing: bool,
             strength_override: str | None) -> dict:
    """What this query gets sent, and how hard the model thinks about it.

    With --ignore-routing (or no routing entry at all) every input is
    attached and the strength is fixed -- the ablation the routed run has to
    beat, and the safe default for a query the router never saw.
    """
    if ignore_routing or not routing:
        plan = {"frames": True, "transcript": True, "dense": True,
                "ocr": False, "strength": DEFAULT_REASONING,
                "routed": False}
    else:
        pipeline = set(routing.get("Pipeline") or ())
        inference = bool(routing.get("Inference"))
        temporal = bool(routing.get("Temporal"))

        plan = {
            "frames": "frames" in pipeline,
            "transcript": "asr_transcript" in pipeline,
            "dense": "dense_timestamped_frames" in pipeline,
            "ocr": "frame_ocr" in pipeline,
            "strength": REASONING_POLICY[(inference, temporal)],
            "routed": True,
        }

        # An OCR question is about pixels whatever else the router said, and
        # a dense-frame question needs frames to be dense over. Neither
        # verdict is allowed to arrive without the images it depends on.
        if plan["ocr"] or plan["dense"]:
            plan["frames"] = True

    if strength_override:
        plan["strength"] = strength_override

    plan["max_tokens"] = MAX_TOKENS_BY_STRENGTH[plan["strength"]]
    # Set later, per video: a query routed away from the frames still gets
    # them when the audio it was routed to turns out not to exist.
    plan["rescued"] = False

    return plan


def select_frames(entry: dict, dense: bool, max_frames: int) -> dict:
    """Which of a video's extracted frames to send, and when they happen.

    Dense means every frame ``extract.frames.py`` wrote (8-32); otherwise an
    even subsample down to ``max_frames`` that always keeps the first and
    last, so the span of the clip is still covered.
    """
    files = list(entry["files"])
    stamps = list(entry["timestamps"])
    n = len(files)

    if dense or n <= max_frames:
        picks = range(n)
    else:
        step = (n - 1) / (max_frames - 1)
        picks = sorted({round(i * step) for i in range(max_frames)})

    return {
        "files": [files[i] for i in picks],
        "timestamps": [stamps[i] for i in picks],
        "duration_sec": entry.get("duration_sec"),
        "n_available": n,
    }


def encode_frames(frame_dir: Path, files: list[str]) -> list[str]:
    """Base64 the selected JPEGs, in order."""
    return [base64.b64encode((frame_dir / name).read_bytes()).decode("ascii")
            for name in files]


def build_context(frames: dict | None, transcript: dict | None,
                  plan: dict) -> str:
    """The text block that precedes the question."""
    parts: list[str] = []

    if frames:
        parts.append(
            f"This video is {frames['duration_sec']:.0f} seconds long. You are "
            f"shown {len(frames['files'])} frames sampled across it, in order, "
            f"each labelled with its timestamp."
        )
    elif plan["frames"] or plan["rescued"]:
        parts.append(
            "The video for this question is unavailable, so you are shown no "
            "frames. Answer from what else you are given, as plausibly as "
            "you can."
        )
    else:
        # Routed to audio only: say so, or the model reads the missing images
        # as a failure and hedges every answer.
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
    elif plan["transcript"]:
        parts.append("The video has no detected speech.")

    return "\n\n".join(parts) + "\n\n"


def build_content(frames: dict | None, images: list[str] | None,
                  prompt: str) -> list[dict]:
    """Interleave ``[mm:ss]`` labels with the frames, then the question.

    A model with no video primitive cannot know when a frame happened unless
    the timestamp travels next to it, and every temporal question in the
    corpus depends on that. Images stay ahead of the question text so the
    queries sharing a video also share a byte-identical prefix for the
    server's prompt cache.
    """
    content: list[dict] = []

    if frames and images:
        for stamp, image in zip(frames["timestamps"], images):
            content.append({"type": "text", "text": f"[{mmss(stamp)}]"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            })

    content.append({"type": "text", "text": prompt})

    return content


def extract_answers(content: str, k: int) -> list[str]:
    """Pull the ranked answer list out of the model's final-line JSON.

    The reply is not grammar-constrained: llama.cpp applies a JSON grammar to
    the whole generation, which would force the reasoning trace itself to be
    JSON. So the model is asked for JSON and the prose it sometimes wraps
    around it is tolerated here instead -- last blob wins, since the
    instruction puts the real answer on the final line.
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
            return [tidy(a) for a in answers[:k] if str(a).strip()]

    return []


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
    """Pad a shortfall with generic, defensible fallback answers."""
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


class MuseClient:
    """Minimal OpenAI-compatible client for llama-server."""

    def __init__(self, base_url: str, model: str, temperature: float,
                 top_p: float, top_k: int, timeout: float,
                 retries: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()

    def props(self) -> dict:
        """llama-server's /props, which lives at the root, not under /v1."""
        root = self.base_url[:-3] if self.base_url.endswith("/v1") \
            else self.base_url
        response = self.session.get(f"{root}/props", timeout=30)
        response.raise_for_status()

        return response.json()

    def ask(self, content: list[dict], strength: str,
            max_tokens: int) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": max_tokens,
            # The jinja template turns this into the model's own
            # "Reasoning strength: ..." system directive; setting it per
            # request is what lets one server serve all four levels. Requires
            # llama-server --jinja, which the model card already mandates.
            "chat_template_kwargs": {"reasoning_strength": strength},
            "cache_prompt": True,
        }

        last: Exception | None = None

        for attempt in range(self.retries):
            started = time.monotonic()

            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload, timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]

                return {
                    "content": message.get("content") or "",
                    # llama-server splits the trace out when the template
                    # declares one; older builds leave it inline in content,
                    # which extract_answers() survives either way.
                    "reasoning": message.get("reasoning_content") or "",
                    "elapsed": time.monotonic() - started,
                    "finish_reason": data["choices"][0].get("finish_reason"),
                }
            except (requests.RequestException, KeyError, ValueError) as exc:
                last = exc

                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"{type(last).__name__}: {last}")


class FrameCache:
    """Base64 frames, keyed by (video, dense), so the queries on one video pay
    the read-and-encode once. Small and bounded: the work is ordered by video,
    so anything older than the last couple of groups will not be asked for
    again, and 32 frames of base64 is tens of megabytes."""

    def __init__(self, capacity: int = 4) -> None:
        self.capacity = capacity
        self.entries: OrderedDict[tuple, tuple[list[str], float]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: tuple, frame_dir: Path,
            files: list[str]) -> tuple[list[str], float]:
        """Returns (images, encode_seconds). encode_seconds is 0.0 on a hit --
        the group that paid for the read is the one that carries the cost."""
        with self.lock:
            if key in self.entries:
                self.entries.move_to_end(key)
                return self.entries[key][0], 0.0

        started = time.monotonic()
        images = encode_frames(frame_dir, files)
        elapsed = time.monotonic() - started

        with self.lock:
            self.entries[key] = (images, elapsed)
            self.entries.move_to_end(key)

            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)

        return images, elapsed


def answer_query(client: MuseClient, query: dict, plan: dict,
                 frames: dict | None, images: list[str] | None,
                 transcript: dict | None, want: int,
                 encode_share: float) -> dict:
    extra_rules = []

    if plan["ocr"]:
        extra_rules.append(OCR_RULE)
    if plan["dense"] and frames:
        extra_rules.append(TEMPORAL_RULE)

    context = build_context(frames, transcript, plan)
    prompt = PROMPT_TEMPLATE.format(
        context=context, question=query["Question"], k=want,
        extra=("\n" + "\n".join(extra_rules)) if extra_rules else "",
    )

    reply = client.ask(build_content(frames, images, prompt),
                       plan["strength"], plan["max_tokens"])

    raw_answers = extract_answers(reply["content"], want)
    chosen = dedupe(raw_answers)
    chosen, fallback_used = fill_slots(chosen, want)

    share = (reply["elapsed"] + encode_share) / max(want, 1)

    return {
        "Q_ID": query["Q_ID"],
        "Video_ID": query["Video_ID"],
        "answers": [
            {"Rank": rank, "Answer": item["answer"], "Time": round(share, 3),
             "source": item["source"]}
            for rank, item in enumerate(chosen, 1)
        ],
        "n_raw_candidates": len(raw_answers),
        "fallback": fallback_used,
        "n_frames": len(images) if images else 0,
        "dense": plan["dense"],
        "ocr": plan["ocr"],
        "routed": plan["routed"],
        "strength": plan["strength"],
        "has_frames": bool(images),
        "has_transcript": bool(transcript and transcript.get("text")),
        "wanted_frames": plan["frames"],
        "rescued": plan["rescued"],
        "elapsed": round(reply["elapsed"], 2),
        "encode_share": round(encode_share, 3),
        "reasoning_chars": len(reply["reasoning"]),
        "finish_reason": reply["finish_reason"],
        "content_parsed": bool(raw_answers),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def group_work(todo: list[dict], plans: dict[int, dict]) -> list[list[dict]]:
    """Batch the queries into runs that share a prompt prefix.

    The prefix a llama-server slot can reuse is [system][frames][...], and the
    frames depend on the video and on whether the router asked for dense
    coverage -- while the rendered system block depends on reasoning strength.
    So queries are grouped by all three and each group runs back to back on
    one worker, which is also what makes the frame encode worth caching.
    """
    groups: dict[tuple, list[dict]] = {}

    for query in todo:
        plan = plans[query["Q_ID"]]
        key = (query["Video_ID"], plan["dense"], plan["strength"],
               plan["frames"])
        groups.setdefault(key, []).append(query)

    return [groups[key] for key in sorted(groups, key=lambda k: (str(k[0]), k))]


def assemble(queries: list[dict], cache: dict[int, dict],
             want: int) -> tuple[list[dict], int]:
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
        description="Answer the Task 1 questions with Muse Glimmer 30B "
                    "(GGUF) served by llama.cpp, routed per query."
    )

    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    ap.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    ap.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)

    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="raise only together with llama-server -np")
    ap.add_argument("--answers-per-query", type=int, default=ANSWERS_PER_QUERY)

    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                    help="frame cap for queries the router did not mark "
                         "temporal (dense ones send every extracted frame)")
    ap.add_argument("--reasoning-strength", choices=STRENGTHS,
                    help="force one strength for every query instead of "
                         "deriving it from the routing verdict")
    ap.add_argument("--ignore-routing", action="store_true",
                    help="send frames and transcript to every query, at a "
                         "fixed reasoning strength -- the ablation baseline")

    ap.add_argument("--limit", type=int, help="process at most N queries")
    ap.add_argument("--only", type=int, nargs="+", metavar="Q_ID")
    ap.add_argument("--force", action="store_true",
                    help="re-answer queries already in the cache")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the run file from the cache, no inference")

    return ap.parse_args(argv)


def preflight(client: MuseClient, args: argparse.Namespace,
              max_frames_used: int) -> None:
    """Fail fast on the two mistakes that silently ruin a run: a server with
    no vision projector loaded, and a per-slot context too small for the
    frames. Both produce plausible-looking answers about nothing."""
    try:
        props = client.props()
    except requests.RequestException as exc:
        print(f"  could not read {args.base_url} /props ({exc}) -- "
              f"continuing blind", file=sys.stderr)
        return

    settings = props.get("default_generation_settings") or {}
    n_ctx = settings.get("n_ctx") or 0
    slots = props.get("total_slots") or 1
    modalities = props.get("modalities") or {}

    print(f"server: {Path(str(props.get('model_path', '?'))).name}  "
          f"slots {slots}  n_ctx/slot {n_ctx}  "
          f"vision {modalities.get('vision')}")

    if modalities.get("vision") is False:
        print("  WARNING: server reports no vision support -- start it with "
              "--mmproj mmproj-Muse-Glimmer-30B-Q4_K_M.gguf or every frame is "
              "being dropped", file=sys.stderr)

    budget = max_frames_used * TOKENS_PER_FRAME_ESTIMATE

    if n_ctx and budget > n_ctx * 0.8:
        print(f"  WARNING: ~{budget} visual tokens for {max_frames_used} "
              f"frames against {n_ctx} per slot -- raise -c (it is split "
              f"across -np slots) or lower --max-frames", file=sys.stderr)

    if args.workers > slots:
        print(f"  WARNING: --workers {args.workers} against {slots} server "
              f"slot(s); extra threads will evict each other's image prefix",
              file=sys.stderr)


def main(argv=None) -> int:
    args = parse_args(argv)

    queries = load_json(args.queries)
    cache = load_cache(args.cache)
    routing = {} if args.ignore_routing else load_routing(args.routing)

    if not routing and not args.ignore_routing:
        print(f"no routing at {args.routing} -- every query gets every "
              f"input (run VQA/route.query.inputs.py first)", file=sys.stderr)

    frame_index: dict[str, dict] = {}
    index_path = args.frames / "index.json"

    if index_path.exists():
        frame_index = load_json(index_path).get("videos", {})
    else:
        print(f"no frame index at {index_path} -- run VQA/extract.frames.py "
              f"first, or every query answers without frames", file=sys.stderr)

    print(f"queries: {len(queries)} | cached: "
          f"{sum(1 for q in queries if q['Q_ID'] in cache)} | "
          f"routed: {len(routing)} | videos with frames: {len(frame_index)}")

    if not args.rebuild:
        todo = [q for q in queries if args.force or q["Q_ID"] not in cache]

        if args.only:
            wanted = set(args.only)
            todo = [q for q in todo if q["Q_ID"] in wanted]

        if args.limit:
            todo = todo[:args.limit]

        plans = {
            q["Q_ID"]: plan_for(routing.get(q["Q_ID"]), args.ignore_routing,
                                args.reasoning_strength)
            for q in todo
        }

        if todo:
            strengths = {s: sum(1 for p in plans.values()
                                if p["strength"] == s) for s in STRENGTHS}
            print(f"to answer: {len(todo)} queries | frames "
                  f"{sum(1 for p in plans.values() if p['frames'])} | "
                  f"transcript {sum(1 for p in plans.values() if p['transcript'])} | "
                  f"dense {sum(1 for p in plans.values() if p['dense'])} | "
                  f"strength " + " ".join(f"{s}:{n}" for s, n in strengths.items()
                                          if n))

            groups = group_work(todo, plans)
            max_frames_used = max(
                (len(select_frames(frame_index[Path(q['Video_ID']).stem],
                                   plans[q["Q_ID"]]["dense"],
                                   args.max_frames)["files"])
                 for q in todo
                 if plans[q["Q_ID"]]["frames"]
                 and Path(q["Video_ID"]).stem in frame_index),
                default=0,
            )

            client = MuseClient(args.base_url, args.model, args.temperature,
                                args.top_p, args.top_k, args.timeout,
                                args.retries)
            preflight(client, args, max_frames_used)

            print(f"{len(groups)} prefix group(s), {args.workers} worker(s) "
                  f"against {args.base_url}")

            frame_cache = FrameCache()
            lock = threading.Lock()
            done = failed = 0
            started = time.monotonic()

            args.cache.parent.mkdir(parents=True, exist_ok=True)
            handle = args.cache.open("a", encoding="utf-8")

            def work(group: list[dict]) -> None:
                nonlocal done, failed

                video_id = group[0]["Video_ID"]
                stem = Path(video_id).stem
                plan = plans[group[0]["Q_ID"]]

                frames = images = None
                encode_elapsed = 0.0

                transcript = load_transcript(args.transcripts, video_id) \
                    if plan["transcript"] else None

                # The router sends 168 queries to the audio alone. When that
                # video turns out to have no transcript, honouring the route
                # would send a prompt carrying no evidence whatsoever -- the
                # frames are worth more than nothing, so they come back.
                if not plan["frames"] and not transcript:
                    for query in group:
                        plans[query["Q_ID"]]["rescued"] = True

                if (plan["frames"] or plan["rescued"]) and stem in frame_index:
                    frames = select_frames(frame_index[stem], plan["dense"],
                                           args.max_frames)
                    try:
                        images, encode_elapsed = frame_cache.get(
                            (stem, plan["dense"]), args.frames / stem,
                            frames["files"],
                        )
                    except OSError as exc:
                        with lock:
                            print(f"  {video_id}: unreadable frames ({exc})",
                                  file=sys.stderr)
                        frames = images = None

                # The read is paid once for the group; each query in it
                # carries an equal share, matching how Time is defined for
                # the Gemma 4 run.
                encode_share = encode_elapsed / len(group)

                for query in group:
                    try:
                        record = answer_query(
                            client, query, plans[query["Q_ID"]], frames,
                            images, transcript, args.answers_per_query,
                            encode_share,
                        )
                    except Exception as exc:  # noqa: BLE001 - never sink a run
                        with lock:
                            failed += 1
                            print(f"  FAILED Q_ID {query['Q_ID']}: "
                                  f"{type(exc).__name__}: {exc}",
                                  file=sys.stderr)
                        continue

                    with lock:
                        handle.write(json.dumps(record, ensure_ascii=False)
                                     + "\n")
                        handle.flush()
                        cache[record["Q_ID"]] = record

                        done += 1
                        rate = done / max(time.monotonic() - started, 1e-9)
                        remaining = (len(todo) - done) / max(rate, 1e-9) / 3600
                        flag = ""

                        if record["fallback"]:
                            flag = f" FALLBACK({record['fallback']})"
                        if not record["content_parsed"]:
                            flag += " UNPARSED"
                        if record["finish_reason"] == "length":
                            flag += " TRUNCATED"
                        if record["wanted_frames"] and not record["has_frames"]:
                            flag += " NOFRAMES"
                        if not record["has_frames"] \
                                and not record["has_transcript"]:
                            flag += " NOEVIDENCE"

                        print(f"  [{done}/{len(todo)}] {rate * 60:5.1f}/min "
                              f"eta {remaining:4.1f}h  {record['elapsed']:6.1f}s "
                              f"{record['n_frames']:2d}f "
                              f"{record['strength']:<6}{flag}  "
                              f"Q{record['Q_ID']}: "
                              f"{record['answers'][0]['Answer'][:44]}")

            try:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    list(pool.map(work, groups))
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
        print(f"  {missing} queries are not answered yet -- the run will "
              f"not validate until they are")

    report(cache, queries)

    return 0


def report(cache: dict[int, dict], queries: list[dict]) -> None:
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
    print(f"  queries needing fallback padding: "
          f"{sum(1 for r in records if r['fallback'])}")
    print(f"  queries whose JSON did not parse: "
          f"{sum(1 for r in records if not r['content_parsed'])}")
    print(f"  generations cut off by max_tokens: "
          f"{sum(1 for r in records if r.get('finish_reason') == 'length')}")
    print(f"  rank-1 answer is \"unknown\":         {unknown}")
    print(f"  queries answered without frames:     "
          f"{sum(1 for r in records if not r['has_frames'])} "
          f"({sum(1 for r in records if r['wanted_frames'] and not r['has_frames'])}"
          f" of them wanted frames)")
    print(f"  queries answered without transcript: "
          f"{sum(1 for r in records if not r['has_transcript'])}")
    print(f"  frames restored to an audio route with no transcript: "
          f"{sum(1 for r in records if r.get('rescued'))}")
    print(f"  queries answered with no evidence at all: "
          f"{sum(1 for r in records if not r['has_frames'] and not r['has_transcript'])}")

    by_strength: dict[str, list[float]] = {}

    for record in records:
        by_strength.setdefault(record.get("strength", "?"),
                               []).append(record["elapsed"])

    for strength in STRENGTHS:
        times = by_strength.get(strength)

        if times:
            print(f"  strength {strength:<6} {len(times):5d} queries  "
                  f"mean {sum(times) / len(times):6.1f}s  "
                  f"total {sum(times) / 3600:5.2f}h")

    times = [r["elapsed"] for r in records]
    print(f"  wall time per query: mean {sum(times) / len(times):5.2f}s  "
          f"min {min(times):.2f}s  max {max(times):.2f}s  "
          f"total {sum(times) / 3600:.2f}h")


if __name__ == "__main__":
    sys.exit(main())

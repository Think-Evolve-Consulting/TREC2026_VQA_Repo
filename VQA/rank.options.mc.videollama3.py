#!/usr/bin/env python3
"""Rank the Task 2 multiple-choice options with VideoLLaMA3-7B over frames.

A port of ``VQA/rank.options.mc.muse.py`` to VideoLLaMA3-7B (bf16), run
in-process with transformers because vLLM has no VideoLLaMA3 architecture.
The ranking contract -- option numbers, permuted passes, Borda,
validator-safe output -- is unchanged.

The model's remote code is pinned to transformers 4.46.3, which the project
venv (5.x) cannot load, so this runs from its own venv, with the checkpoint
downloaded to ``models/VideoLLaMA3-7B``::

    E:/TREC_2026/.venv-videollama3/Scripts/python.exe VQA/rank.options.mc.videollama3.py --limit 20

Attention is ``sdpa``: flash-attn cannot be built here, and the remote code
falls back to it cleanly.

**The frames are ours, with their real timestamps.** VideoLLaMA3 is a video
model, but its loader stamps a directory of frames at a fixed 3 fps. The
processor skips the loader when a ``video`` item is already a list of frames,
and the chat template prints whatever ``timestamps`` it is given, so the JPEGs
``VQA/extract.frames.py`` wrote go in with the sampling instants from its
index -- the same frames the Gemma and Muse passes saw.

**The visual budget is whole-clip.** The image processor shrinks every frame
until the clip fits ``--max-visual-tokens`` -- 4,096 here against the model's
own 16,384, the budget the Qwen passes ran at.

**No grammar.** transformers' generate() has no JSON-schema constraint, so the
prompt asks for JSON, ``parse_json_object()`` tolerates prose around it, and
``as_permutation()`` repairs repeats and drops.

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

**Every pass re-encodes the clip.** There is no prefix cache in-process, so
three passes cost three prefills; at 7B and 4k visual tokens that is seconds.

**There is no Time field.** Task 1 requires one; the MC validator rejects any
key outside ``Rank``/``Answer`` (EXTRA_ANSWER_FIELDS), so timings live in the
cache instead of the run file.

Usage:
    python VQA/rank.options.mc.videollama3.py --limit 20      # smoke test
    python VQA/rank.options.mc.videollama3.py                 # everything, resumable
    python VQA/rank.options.mc.videollama3.py --only 1 3 264  # named Q_IDs
    python VQA/rank.options.mc.videollama3.py --rebuild       # re-emit run from cache

(each with the .venv-videollama3 Python)
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

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = ROOT / "vqa.2026.testingDataset.Task2_v2.json"
DEFAULT_ROUTING = ROOT / "query.routing.task1.json"
DEFAULT_FRAMES = ROOT / "videos_TEST_VQA" / "frames"
DEFAULT_TRANSCRIPTS = ROOT / "videos_TEST_VQA" / "transcription_openwhisper"

# Downloaded with --local-dir and loaded in-process. MODEL_NAME is what the
# cache records, so a run is tied to the checkpoint rather than to a path.
MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA3-7B"
DEFAULT_MODEL = ROOT / "models" / "VideoLLaMA3-7B"
DEFAULT_OUT = ROOT / "videollama3.run.task2.json"
DEFAULT_CACHE = ROOT / "rankings.task2.videollama3.cache.jsonl"

# One model on one GPU in this process, so one query at a time.
DEFAULT_WORKERS = 1

# A permutation of at most eight integers plus a confidence is ~40 tokens.
DEFAULT_NUM_PREDICT = 128

# The Gemma pass's ranking sampler: the diversity comes from permuting the
# options, not from sampling noise.
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64

# Official order, reversed, then seeded shuffles, combined by Borda.
DEFAULT_PASSES = 3

# Whole-clip visual tokens, the budget the Qwen passes ran at. The model's own
# default is 16,384, which would also make the vision encoder's attention four
# times the size on a card that has ~7 GB beside the weights.
DEFAULT_MAX_VISUAL_TOKENS = 4096

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


def make_blockwise_forward(rotary):
    """A VisionSdpaAttention.forward that attends one frame at a time.

    The remote version builds one [S, S] boolean mask over every patch in the
    clip and hands it to sdpa, which then falls back to materialising
    heads x S x S scores -- about 8 GB per layer at 16k patches. On Windows the
    driver spills that into system RAM instead of raising OOM, so a query
    crawls for minutes rather than failing. The mask is block-diagonal, one
    block per frame (``cu_seqlens``), so attending block by block is the same
    computation with no mask and a few MB of scores.
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
    There is no grammar, so a reply with no JSON is sampled again (at twice
    the budget if it was cut off) before the pass is counted as failed.
    """

    name = "transformers"

    def __init__(self, path: Path, num_predict: int, temperature: float,
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
        self.use_schema = False

        started = time.monotonic()
        self.net = AutoModelForCausalLM.from_pretrained(
            str(path), trust_remote_code=True, device_map="cuda:0",
            torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        self.processor = AutoProcessor.from_pretrained(
            str(path), trust_remote_code=True)
        # Whole-clip: the image processor shrinks every frame until the clip
        # fits, so this bounds prefill and the vision encoder together.
        self.processor.image_processor.max_tokens = max_visual_tokens

        # See make_blockwise_forward(): without this the first query spills
        # the card into system RAM and never finishes.
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
                             "remote code changed; the block-wise patch did "
                             "not apply")

        print(f"loaded {path.name} in {time.monotonic() - started:.0f}s | "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB on the GPU | "
              f"visual budget {max_visual_tokens} tokens per clip | "
              f"{patched} vision attention layers made block-wise")

    def ask(self, prompt: str, schema: dict, seed: int,
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

        inputs = self.processor(
            conversation=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": content}],
            add_generation_prompt=True, return_tensors="pt")
        inputs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

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
                    output = self.net.generate(**inputs, max_new_tokens=budget,
                                               **sampling)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise RuntimeError(f"CUDA OOM at {len(prompt_ids)} prompt "
                                   f"tokens: {exc}") from exc

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
                last = ValueError(f"no JSON in reply ({completion} tokens): "
                                  f"{text[:120]!r}")

                # Cut off: the one failure more budget fixes.
                if completion >= budget and budget == self.num_predict:
                    budget *= 2

                continue

            return reply, elapsed, {
                "finish_reason": "length" if completion >= budget else "stop",
                "prompt_tokens": int(prompt_ids.shape[0]),
                "completion_tokens": completion,
                "extended": budget != self.num_predict,
                "attempts": attempt + 1,
            }

        raise RuntimeError(f"{type(last).__name__}: {last}")


def rank_query(client: VideoLlama3, query: dict, options: list[str],
               context: str, images: list[tuple] | None, passes: int,
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


def preflight(path: Path) -> None:
    """Refuse to start without a complete local checkpoint.

    A half-downloaded directory otherwise reaches huggingface_hub as a repo
    id and fails with an error about repo-name characters, which is about
    entirely the wrong thing.
    """
    missing = [name for name in ("config.json", "modeling_videollama3.py",
                                 "processing_videollama3.py")
               if not (path / name).is_file()]
    index = path / "model.safetensors.index.json"

    if index.is_file():
        shards = set(load_json(index)["weight_map"].values())
        missing += sorted(name for name in shards
                          if not (path / name).is_file())
    else:
        missing.append(index.name)

    if missing:
        raise SystemExit(
            f"{path} is not a complete VideoLLaMA3 checkpoint (missing "
            f"{', '.join(missing)}). Download it with:\n"
            f"  hf download {MODEL_NAME} --local-dir {path}")


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
                    "VideoLLaMA3-7B (in-process) over video frames."
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
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="run file to write (default: %(default)s)")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help="JSONL work cache, one line per Q_ID "
                         "(default: %(default)s)")

    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                    help="local checkpoint directory (default: %(default)s)")
    ap.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT,
                    help="new tokens per reply; one cut off by it is retried "
                         "at twice this (default: %(default)s)")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help="sampling temperature, 0 for greedy "
                         "(default: %(default)s)")
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--retries", type=int, default=3,
                    help="attempts per call before giving up on it")
    ap.add_argument("--max-visual-tokens", type=int,
                    default=DEFAULT_MAX_VISUAL_TOKENS,
                    help="whole-clip visual token budget for the image "
                         "processor (default: %(default)s)")

    ap.add_argument("--passes", type=int, default=DEFAULT_PASSES,
                    help="rankings per query, each with the options presented "
                         "in a different order (default: %(default)s)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="one model in this process, so 1 "
                         "(default: %(default)s)")

    ap.add_argument("--limit", type=int, help="process at most N queries")
    ap.add_argument("--only", type=int, nargs="+", metavar="Q_ID")
    ap.add_argument("--force", action="store_true",
                    help="re-rank queries already in the cache")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the run file from the cache, no inference")

    return ap.parse_args(argv)


def load_frames(directory: Path, entry: dict) -> list[tuple]:
    """The extracted JPEGs as (CHW uint8 array, seconds) pairs, in order.

    The array layout is what VideoLLaMA3's own loader returns. The timestamps
    are the real sampling instants from the frame index, not ``index / 3``,
    which is what the processor assumes for a directory of frames.
    """
    import numpy as np
    from PIL import Image

    return [(np.asarray(Image.open(directory / name).convert("RGB"))
             .transpose(2, 0, 1), float(stamp))
            for name, stamp in zip(entry["files"], entry["timestamps"])]


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
                      cache.values()} - {MODEL_NAME})

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

    print(f"model: {MODEL_NAME} from {args.model} | {args.workers} worker(s) "
          f"| max_tokens {args.num_predict} | temperature {args.temperature} "
          f"| visual budget {args.max_visual_tokens}")
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
            preflight(args.model)
            client = VideoLlama3(args.model, args.num_predict,
                                 args.temperature, args.top_p, args.top_k,
                                 args.retries, args.max_visual_tokens)

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
                    try:
                        images = load_frames(args.frames / Path(video_id).stem,
                                             frames)
                    except (OSError, KeyError, TypeError, ValueError) as exc:
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

    # Where the time goes, per pass. In-process there is no prefix cache and
    # no prefill/decode split, so those read 0 here.
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

#!/usr/bin/env python3
"""Answer the Task 1 questions with Muse Glimmer 30B loaded natively in Python.

In-process twin of ``generate.answers.muse.py``, which talks to a llama.cpp
server over HTTP. This one skips llama.cpp entirely: transformers loads the
bf16 checkpoint with ``AutoModelForMultimodalLM``, bitsandbytes quantizes it to
NF4 on the way onto the GPU, and generation happens in this process. Same
routing, same prompts, same cache format, same run file -- so the two are
directly comparable, and ``--cache`` is the only thing that must differ.

Output is the submission file ``VQA/validate.ag.run.py`` accepts verbatim::

    [{"Q_ID": 1, "Video_ID": "8384.webm",
      "Answers": [{"Rank": 1, "Answer": "...", "Time": 1.234}, ...]}]

What this costs, relative to the llama.cpp path:

**The bf16 checkpoint is 59.6 GB.** There is no smaller starting point that
works here. Meta ships calibrated GGUF quants, but those are llama.cpp's
format; the community's transformers-native quants are FP8-block and NVFP4,
and this box has an RTX 3090 (sm_86, Ampere) with no hardware FP8 or FP4. So
the full bf16 download is the entry fee, and bitsandbytes NF4 does the
quantizing at load time -- roughly 15-16 GB of weights on the card, against
15.6 GB for the Q4_K_M GGUF, but from an uncalibrated round-to-nearest quant
rather than Meta's calibrated one.

**Expect it to be slower than llama.cpp.** bitsandbytes NF4 dequantizes per
matmul and there is no equivalent of llama.cpp's prompt cache here, so the
frames of a video are re-encoded for every query that shares it. The routing
file is what keeps this affordable: 721 of the 2443 queries run at low
reasoning strength and only 300 at high.

**Reasoning strength is a first-class kwarg**, unlike the server path where it
travels as chat_template_kwargs: ``apply_chat_template(...,
reasoning_strength="high")``. If a checkpoint's template does not accept it,
this falls back to stating it in the system prompt, which the model card
documents as equivalent.

Setup::

    .venv\\Scripts\\python.exe -m pip install accelerate bitsandbytes pillow
    hf download meta-models/Muse-Glimmer-30B --local-dir E:/TREC_2026/models/Muse-Glimmer-30B

Usage:
    python VQA/generate.answers.muse.hf.py --limit 20      # smoke test
    python VQA/generate.answers.muse.hf.py                 # everything, resumable
    python VQA/generate.answers.muse.hf.py --only 1 3 264  # named Q_IDs
    python VQA/generate.answers.muse.hf.py --rebuild       # re-emit run from cache
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = ROOT / "vqa.2026.testingDataset.Task1.json"
DEFAULT_ROUTING = ROOT / "query.routing.task1.json"
DEFAULT_FRAMES = ROOT / "videos_TEST_VQA" / "frames"
DEFAULT_TRANSCRIPTS = ROOT / "videos_TEST_VQA" / "transcription_openwhisper"
DEFAULT_OUT = ROOT / "muse.hf.run.task1.json"
DEFAULT_CACHE = ROOT / "answers.task1.muse.hf.cache.jsonl"

DEFAULT_MODEL = ROOT / "models" / "Muse-Glimmer-30B"
"""Local bf16 checkpoint. A bare HF repo id works too, but then the 59.6 GB
lands in the HF cache on C: rather than next to the rest of the corpus."""

# The model card's own numbers.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64

ANSWERS_PER_QUERY = 10
DEFAULT_MAX_FRAMES = 16

REASONING_POLICY = {
    (False, False): "low",
    (False, True): "medium",
    (True, False): "medium",
    (True, True): "high",
}
DEFAULT_REASONING = "medium"
STRENGTHS = ("low", "medium", "high", "xhigh")

MAX_TOKENS_BY_STRENGTH = {
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 12288,
}

MAX_ANSWER_WORDS = 40
MAX_ANSWER_CHARS = 300

LEAD_IN_RE = re.compile(r"^\s*(?:answer|a)\s*[:\-]\s*", re.IGNORECASE)
ENUM_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
JSON_BLOB_RE = re.compile(r"\{.*?\}", re.DOTALL)

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
    """What this query gets sent, and how hard the model thinks about it."""
    if ignore_routing or not routing:
        plan = {"frames": True, "transcript": True, "dense": True,
                "ocr": False, "strength": DEFAULT_REASONING, "routed": False}
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

        if plan["ocr"] or plan["dense"]:
            plan["frames"] = True

    if strength_override:
        plan["strength"] = strength_override

    plan["max_tokens"] = MAX_TOKENS_BY_STRENGTH[plan["strength"]]
    plan["rescued"] = False

    return plan


def select_frames(entry: dict, dense: bool, max_frames: int) -> dict:
    """Which of a video's extracted frames to send, and when they happen."""
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


def extract_answers(content: str, k: int) -> list[str]:
    """Pull the ranked answer list out of the model's final-line JSON.

    Nothing constrains the decode, and the reasoning trace comes back in the
    same string as the answer (transformers has no reasoning parser the way
    the llama.cpp server does), so the last JSON object wins -- the prompt
    puts the real one on the final line.
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


class MuseModel:
    """The checkpoint, loaded once and held for the whole run."""

    def __init__(self, model_path: str, load_4bit: bool, temperature: float,
                 top_p: float, top_k: int) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        # Set on the first failure, so the fallback is tried once and then
        # taken directly rather than raising per query.
        self.template_takes_strength = True

        quantization_config = None

        if load_4bit:
            from transformers import BitsAndBytesConfig

            # NF4 with double quantization: the standard 4-bit recipe, and
            # the only 4-bit that runs on Ampere -- FP8 needs sm_89 and NVFP4
            # needs Blackwell. Compute stays bf16 so the vision tower and the
            # attention maths are not the thing being degraded.
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        print(f"loading {model_path} "
              f"({'NF4 4-bit' if load_4bit else 'full precision'}) ...")
        started = time.monotonic()

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            quantization_config=quantization_config,
            # flash-attn has no Windows wheel; sdpa is torch's own fused
            # attention and needs nothing extra.
            attn_implementation="sdpa",
        )
        self.model.eval()

        print(f"  loaded in {time.monotonic() - started:.0f}s | "
              f"{self.vram():.1f} GB VRAM in use")

        self.probe_strength()

    def probe_strength(self) -> None:
        """Prove the template actually reads reasoning_strength.

        apply_chat_template forwards unknown kwargs into the jinja render
        context rather than rejecting them, so a template that does not
        consume this one fails silently: every query would run at the
        default strength and the run would look entirely normal. Rendering
        the same messages at both extremes and comparing is the only honest
        check -- if they match, the kwarg is doing nothing and the
        system-prompt form takes over.
        """
        probe = [
            {"role": "system", "content": [{"type": "text", "text": "s"}]},
            {"role": "user", "content": [{"type": "text", "text": "u"}]},
        ]

        try:
            rendered = {
                strength: self.processor.apply_chat_template(
                    probe, tokenize=False, add_generation_prompt=True,
                    reasoning_strength=strength)
                for strength in ("low", "xhigh")
            }
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            print(f"  template rejected reasoning_strength ({exc}) -- using "
                  f"the system-prompt form", file=sys.stderr)
            self.template_takes_strength = False
            return

        if rendered["low"] == rendered["xhigh"]:
            print("  template ignores reasoning_strength -- using the "
                  "system-prompt form instead", file=sys.stderr)
            self.template_takes_strength = False
        else:
            print("  reasoning_strength verified: low and xhigh render "
                  "differently")

    def vram(self) -> float:
        if self.torch.cuda.is_available():
            return self.torch.cuda.memory_allocated() / 1024 ** 3
        return 0.0

    def build_messages(self, frames: dict | None, images: list | None,
                       prompt: str, strength: str) -> list[dict]:
        """Interleave ``[mm:ss]`` labels with the frames, then the question.

        A model with no video primitive cannot know when a frame happened
        unless the timestamp travels next to it, and every temporal question
        in the corpus depends on that.
        """
        content: list[dict] = []

        if frames and images:
            for stamp, image in zip(frames["timestamps"], images):
                content.append({"type": "text", "text": f"[{mmss(stamp)}]"})
                content.append({"type": "image", "image": image})

        content.append({"type": "text", "text": prompt})

        system = SYSTEM_PROMPT

        if not self.template_takes_strength:
            system = f"{system}\n\nReasoning strength: {strength}"

        return [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content},
        ]

    def encode(self, messages: list[dict], strength: str):
        kwargs = dict(tokenize=True, return_dict=True, return_tensors="pt",
                      add_generation_prompt=True)

        if self.template_takes_strength:
            try:
                return self.processor.apply_chat_template(
                    messages, reasoning_strength=strength, **kwargs)
            except TypeError:
                # This checkpoint's template does not take the kwarg. The
                # model card documents the system-prompt form as equivalent,
                # so switch to it permanently and rebuild the messages.
                print("  template rejected reasoning_strength -- falling back "
                      "to the system-prompt form", file=sys.stderr)
                self.template_takes_strength = False
                raise

        return self.processor.apply_chat_template(messages, **kwargs)

    def ask(self, frames: dict | None, images: list | None, prompt: str,
            strength: str, max_tokens: int) -> dict:
        torch = self.torch
        messages = self.build_messages(frames, images, prompt, strength)

        try:
            inputs = self.encode(messages, strength)
        except TypeError:
            messages = self.build_messages(frames, images, prompt, strength)
            inputs = self.encode(messages, strength)

        inputs = inputs.to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        started = time.monotonic()

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )

        elapsed = time.monotonic() - started
        generated = outputs[0][input_len:]
        text = self.processor.decode(generated, skip_special_tokens=False)

        return {
            "content": text,
            "elapsed": elapsed,
            "prompt_tokens": int(input_len),
            "completion_tokens": int(generated.shape[-1]),
            # generate() stops at max_new_tokens without saying so; comparing
            # the two is the only way to spot a trace that was cut off.
            "truncated": int(generated.shape[-1]) >= max_tokens,
        }


def load_images(frame_dir: Path, files: list[str]) -> list:
    """The selected frames as PIL images, in order."""
    from PIL import Image

    images = []

    for name in files:
        with Image.open(frame_dir / name) as handle:
            images.append(handle.convert("RGB"))

    return images


def answer_query(model: MuseModel, query: dict, plan: dict,
                 frames: dict | None, images: list | None,
                 transcript: dict | None, want: int,
                 decode_share: float) -> dict:
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

    reply = model.ask(frames, images, prompt, plan["strength"],
                      plan["max_tokens"])

    raw_answers = extract_answers(reply["content"], want)
    chosen = dedupe(raw_answers)
    chosen, fallback_used = fill_slots(chosen, want)

    share = (reply["elapsed"] + decode_share) / max(want, 1)

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
        "decode_share": round(decode_share, 3),
        "prompt_tokens": reply["prompt_tokens"],
        "completion_tokens": reply["completion_tokens"],
        "truncated": reply["truncated"],
        "content_parsed": bool(raw_answers),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


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
                    "loaded natively via transformers, routed per query."
    )

    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    ap.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    ap.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)

    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="local checkpoint dir or HF repo id")
    ap.add_argument("--no-4bit", action="store_true",
                    help="load in full precision -- needs ~60 GB of VRAM, so "
                         "only for a bigger card than this one")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
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


# torchvision has to be the release built against the installed torch -- it
# declares an exact `torch==` pin, so an unpinned install resolves to the
# newest one and drags torch with it, which on this box means losing the
# +cu124 build the rest of the project runs on.
TORCHVISION_FOR_TORCH = {
    "2.6.0": "0.21.0",
    "2.7.0": "0.22.0",
    "2.8.0": "0.23.0",
}


def check_dependencies(load_4bit: bool) -> bool:
    """Report every missing runtime dependency at once.

    All of these are imported lazily and late -- torchvision from inside the
    image processor, bitsandbytes from inside from_pretrained, PIL not until
    the first frame is decoded. Left alone they surface one per run, each one
    only after the previous has been fixed, and the last of them arrives ten
    minutes in with 16 GB already on the card.
    """
    import importlib.util

    needed = [
        ("torchvision", "MuseGlimmerImageProcessor is torchvision-backed and "
                        "has no slow fallback"),
        ("PIL", "frames are decoded to PIL images"),
        ("accelerate", "device_map='auto' is implemented there"),
    ]

    if load_4bit:
        needed.append(("bitsandbytes", "NF4 quantization at load time"))

    missing = [(name, why) for name, why in needed
               if importlib.util.find_spec(name) is None]

    if not missing:
        return True

    print("ERROR: missing runtime dependencies", file=sys.stderr)

    for name, why in missing:
        print(f"    {name:14} -- {why}", file=sys.stderr)

    plain = [name for name, _ in missing if name != "torchvision"]
    plain = ["pillow" if name == "PIL" else name for name in plain]

    if plain:
        print(f"\n    pip install {' '.join(plain)}", file=sys.stderr)

    if any(name == "torchvision" for name, _ in missing):
        import torch

        version = torch.__version__.split("+")[0]
        tag = f"cu{torch.version.cuda.replace('.', '')}" \
            if torch.version.cuda else "cpu"
        pin = TORCHVISION_FOR_TORCH.get(version)

        if pin:
            print(f"    pip install torchvision=={pin} --index-url "
                  f"https://download.pytorch.org/whl/{tag}", file=sys.stderr)
        else:
            print(f"    pip install torchvision --index-url "
                  f"https://download.pytorch.org/whl/{tag}\n"
                  f"      (must be the torchvision released alongside torch "
                  f"{version}; an unpinned install will change torch)",
                  file=sys.stderr)

    return False


def check_checkpoint(model: str) -> bool:
    """Refuse to start on a local path that is not a usable checkpoint.

    transformers treats anything that is not an existing directory as a Hub
    repo id, so a mistyped or not-yet-downloaded path surfaces as
    ``HFValidationError: Repo id must use alphanumeric chars`` from deep
    inside huggingface_hub -- an error about the wrong thing entirely. A
    Windows path is unmistakable (drive letter or backslash), so when one is
    given it has to resolve locally or the run stops here.
    """
    text = str(model)
    looks_local = "\\" in text or ":" in text or text.startswith(".")

    if not looks_local:
        return True  # a Hub repo id; let transformers resolve it

    path = Path(text)
    hint = (f"    hf download meta-models/Muse-Glimmer-30B --local-dir "
            f"{path}\n    (59.6 GB; or pass --model meta-models/"
            f"Muse-Glimmer-30B to stream it into the HF cache)")

    if not path.is_dir():
        print(f"ERROR: no checkpoint at {path}\n{hint}", file=sys.stderr)
        return False

    missing = [name for name in ("config.json", "processor_config.json")
               if not (path / name).exists()]
    shards = list(path.glob("*.safetensors"))

    if not shards or missing:
        detail = "no .safetensors shards" if not shards \
            else f"missing {', '.join(missing)}"
        print(f"ERROR: {path} is not a complete checkpoint ({detail}) -- an "
              f"interrupted download leaves it like this\n{hint}",
              file=sys.stderr)
        return False

    size = sum(shard.stat().st_size for shard in shards) / 1024 ** 3
    print(f"checkpoint: {path.name}  {len(shards)} shard(s), {size:.1f} GB")

    return True


def preflight(args: argparse.Namespace) -> bool:
    """Say what the card is and whether the checkpoint is actually there,
    before spending ten minutes loading 60 GB off disk."""
    import torch

    if not check_dependencies(not args.no_4bit):
        return False

    if not torch.cuda.is_available():
        print("  WARNING: no CUDA device visible -- this will run on the CPU "
              "and take days", file=sys.stderr)
    else:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        free, total = torch.cuda.mem_get_info(0)

        print(f"device: {name} (sm_{major}{minor})  "
              f"{free / 1024 ** 3:.1f} GB free of {total / 1024 ** 3:.1f} GB")

        if not args.no_4bit and free / 1024 ** 3 < 18:
            print(f"  WARNING: NF4 weights need ~16 GB plus KV cache and "
                  f"activations; only {free / 1024 ** 3:.1f} GB is free. Stop "
                  f"whatever else holds the card (the WSL2 vLLM server is the "
                  f"usual culprit).", file=sys.stderr)

    return check_checkpoint(args.model)


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

            if not preflight(args):
                return 2

            # Group by video so a video's frames are decoded once for the
            # queries that share it. Unlike the llama.cpp path there is no
            # prompt cache to exploit, so this is only about the JPEG decode.
            todo.sort(key=lambda q: (str(q["Video_ID"]), q["Q_ID"]))

            model = MuseModel(args.model, not args.no_4bit, args.temperature,
                              args.top_p, args.top_k)

            done = failed = 0
            started = time.monotonic()

            args.cache.parent.mkdir(parents=True, exist_ok=True)
            handle = args.cache.open("a", encoding="utf-8")

            decoded_key: tuple | None = None
            decoded_images: list | None = None
            decoded_elapsed = 0.0

            try:
                for query in todo:
                    plan = plans[query["Q_ID"]]
                    video_id = query["Video_ID"]
                    stem = Path(video_id).stem

                    transcript = load_transcript(args.transcripts, video_id) \
                        if plan["transcript"] else None

                    # The router sends some queries to the audio alone. When
                    # that video has no transcript, honouring the route would
                    # send a prompt carrying no evidence at all.
                    if not plan["frames"] and not transcript:
                        plan["rescued"] = True

                    frames = images = None
                    decode_share = 0.0

                    if (plan["frames"] or plan["rescued"]) \
                            and stem in frame_index:
                        frames = select_frames(frame_index[stem],
                                               plan["dense"], args.max_frames)
                        key = (stem, plan["dense"])

                        try:
                            if key != decoded_key:
                                decode_started = time.monotonic()
                                decoded_images = load_images(
                                    args.frames / stem, frames["files"])
                                decoded_elapsed = \
                                    time.monotonic() - decode_started
                                decoded_key = key
                                decode_share = decoded_elapsed

                            images = decoded_images
                        except OSError as exc:
                            print(f"  {video_id}: unreadable frames ({exc})",
                                  file=sys.stderr)
                            frames = images = None
                            decoded_key = None

                    try:
                        record = answer_query(model, query, plan, frames,
                                              images, transcript,
                                              args.answers_per_query,
                                              decode_share)
                    except Exception as exc:  # noqa: BLE001 - never sink a run
                        failed += 1
                        print(f"  FAILED Q_ID {query['Q_ID']}: "
                              f"{type(exc).__name__}: {exc}", file=sys.stderr)
                        continue

                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
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
                    if record["truncated"]:
                        flag += " TRUNCATED"
                    if record["wanted_frames"] and not record["has_frames"]:
                        flag += " NOFRAMES"
                    if not record["has_frames"] and not record["has_transcript"]:
                        flag += " NOEVIDENCE"

                    print(f"  [{done}/{len(todo)}] {rate * 60:5.1f}/min "
                          f"eta {remaining:4.1f}h  {record['elapsed']:6.1f}s "
                          f"{record['n_frames']:2d}f "
                          f"{record['prompt_tokens']:6d}p"
                          f"{record['completion_tokens']:5d}c "
                          f"{record['strength']:<6}{flag}  "
                          f"Q{record['Q_ID']}: "
                          f"{record['answers'][0]['Answer'][:40]}")
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
    print(f"  generations cut off by max_new_tokens: "
          f"{sum(1 for r in records if r.get('truncated'))}")
    print(f"  rank-1 answer is \"unknown\":         {unknown}")
    print(f"  queries answered without frames:     "
          f"{sum(1 for r in records if not r['has_frames'])} "
          f"({sum(1 for r in records if r['wanted_frames'] and not r['has_frames'])}"
          f" of them wanted frames)")
    print(f"  queries answered without transcript: "
          f"{sum(1 for r in records if not r['has_transcript'])}")
    print(f"  frames restored to an audio route with no transcript: "
          f"{sum(1 for r in records if r.get('rescued'))}")

    prompts = [r.get("prompt_tokens", 0) for r in records]
    completions = [r.get("completion_tokens", 0) for r in records]
    print(f"  prompt tokens: mean {sum(prompts) / len(prompts):7.0f}  "
          f"max {max(prompts)}")
    print(f"  output tokens: mean {sum(completions) / len(completions):7.0f}  "
          f"max {max(completions)}")

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

#!/usr/bin/env python3
"""One judge in the two-judge debate: receives a Q_ID, reads its own disk.

Two copies of this script run, one beside each model server:

    side a  the Mac, beside LM Studio holding Qwen3.8-27B, dataset at
            /Users/Aakash/Downloads/videos_TEST_VQA
    side b  this workstation, beside llama.cpp holding InternVL3.5-30B-A3B,
            dataset at E:/TREC_2026/videos_TEST_VQA

The orchestrator (`VQA/debate.rank.mc.py`) never sends a frame, a question or a
ranking -- only a Q_ID and, in later rounds, the debate so far. Everything else
comes from this machine's own copy of the dataset:

    <dataset>/ranking/<Q_ID>.json   the brief: question, options, panel,
                                    router verdict, transcript, evidence
    <dataset>/ranking/_card.txt     the panel reliability card
    <dataset>/frames448/<stem>/     frames cut by --cut-frames on this machine

A model server cannot do this itself: LM Studio rejects a file:// URL
("'url' field must be a base64 encoded image") and cannot be handed a Q_ID, so
the reading lives here and the server only does inference.

This file must run on a machine that has neither the repo nor its helpers, so
it imports nothing from the project. The few ranking-contract helpers it needs
(as_permutation, parse_json_object, seed_for, mmss) are copied from
`VQA/rank.options.mc.muse.py` and must keep behaving the same way.
Requirements: Python 3.9+, `pip install requests pillow`, and ffmpeg on PATH for
--cut-frames.

Usage:
    # once per machine: cut 32 frames per video, 64 for counting videos
    python3 agent.debate.mc.py --dataset /Users/Aakash/Downloads/videos_TEST_VQA --cut-frames

    # the Mac judge
    python3 agent.debate.mc.py --side a --dataset /Users/Aakash/Downloads/videos_TEST_VQA \\
        --backend lmstudio --endpoint http://127.0.0.1:1234 --model qwen/qwen3.8-27b \\
        --port 7801

    # the local judge (llama-server started with --media-path <dataset>/frames448)
    python VQA/agent.debate.mc.py --side b --dataset E:/TREC_2026/videos_TEST_VQA \\
        --backend llamacpp --endpoint http://127.0.0.1:8013 \\
        --model internvl3.5-30b-a3b --port 7802

    # check the model actually sees, and time one real opinion, then exit
    ... --selftest
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

FRAME_SIZE = 448
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT = 1800.0

SYSTEM_PROMPT = (
    "You adjudicate a panel of video question-answering models, and you argue "
    "with a second adjudicator who sees the same video. You decide from the "
    "frames themselves, cite the frame times that justify your choice, and "
    "change your mind when the evidence says so. You reply with JSON only."
)

OPINION_TEMPLATE = """{context}Question: {question}

Answer options:
{options}

The panel disagreed on this question. How each member ranked the options:

{panel}
{evidence}{card}{routing}
Form your own opinion first. Decide the final ranking of all {n} options, best
first.

Rules:
- Judge the options against the frames yourself. The panel's rankings are
  opinions, not evidence.
- Do not count votes. A majority of this panel scores worse than its strongest
  single member.
- Cite the frame times (in seconds) that support your first choice.
- Every number from 1 to {n} must appear exactly once.

Reply with JSON: {{"ranking": [{slots}], "confidence": 0.0, "cites": [0.0],
"argument": "..."}} where argument is at most 40 words."""

ROUND_TEMPLATE = """{context}Question: {question}

Answer options:
{options}

How the panel ranked the options:

{panel}
{evidence}{routing}
You and a second adjudicator watched the same frames and disagree. The whole
debate so far, in order:

{transcript}

This is round {round} of {rounds}. {instruction}

Reply with JSON: {{"stance": "hold" or "switch", "ranking": [{slots}],
"confidence": 0.0, "cites": [0.0], "argument": "..."}} where argument is at most
40 words and answers the other adjudicator's latest point directly."""

INSTRUCTIONS = {
    "rebuttal": ("Look at the frames again. Either hold your choice or switch to "
                 "theirs, whichever the frames support, and cite the frame times "
                 "that decide it."),
    "final": ("This is your final statement. Answer their claim about the frames "
              "specifically: concede if the frames show what they say, or state "
              "what those frames actually show. Then give your final ranking."),
}

COUNTING_AID = ("This is a counting question and every model in the panel "
                "undercounts. Count in each frame separately and report the "
                "highest distinct count you can actually see; never average "
                "across frames.")


# --- the ranking contract, copied from rank.options.mc.muse.py ---------------

def seed_for(q_id: int, index: int) -> int:
    """Stable per-call seed, so a re-run reproduces the same run file."""
    digest = hashlib.sha256(f"mc|{q_id}|{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def mmss(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


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
    """Coerce a 1-based ranking into a permutation of 0..n-1, noting repairs.

    Repeats keep their first occurrence and dropped options are appended in
    presented order -- identical to the ranking passes, so a repaired tail here
    means what it means everywhere else in the project.
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


def debate_schema(n: int) -> dict:
    """The ranking contract plus the debate fields.

    Uniqueness is not expressed: no grammar backend here enforces uniqueItems,
    and as_permutation is what actually repairs a repeat.
    """
    return {
        "type": "object",
        "properties": {
            "ranking": {"type": "array",
                        "items": {"type": "integer", "minimum": 1, "maximum": n},
                        "minItems": n, "maxItems": n},
            "confidence": {"type": "number"},
            "cites": {"type": "array", "items": {"type": "number"}},
            "argument": {"type": "string"},
            "stance": {"type": "string"},
        },
        "required": ["ranking", "confidence", "cites", "argument"],
    }


# --- this machine's copy of the dataset ---------------------------------------

class Dataset:
    def __init__(self, root: Path, max_frames: int | None) -> None:
        self.root = root
        self.ranking = root / "ranking"
        self.frames = root / "frames448"
        self.videos = root / "videos"
        self.max_frames = max_frames
        card = self.ranking / "_card.txt"
        self.card = card.read_text(encoding="utf-8").strip() if card.exists() else ""
        self.cache: dict[int, dict] = {}
        self.lock = threading.Lock()

    def brief(self, q_id: int) -> dict:
        with self.lock:
            if q_id not in self.cache:
                path = self.ranking / f"{q_id}.json"

                if not path.exists():
                    raise FileNotFoundError(f"no brief {path}")

                self.cache[q_id] = json.loads(path.read_text(encoding="utf-8"))

            return self.cache[q_id]

    def frame_list(self, brief: dict) -> list[tuple[Path, float]]:
        """The frames cut for this video, with the times they were cut at.

        --cut-frames samples at mid-points, so frame k of n sits at
        (k + 0.5) * duration / n. A cap below n subsamples evenly and keeps the
        true times, so a cited timestamp always points at a real frame.
        """
        directory = self.frames / brief["stem"]
        files = sorted(directory.glob("*.jpg"))

        if not files:
            raise FileNotFoundError(f"no frames in {directory} -- run --cut-frames")

        n = len(files)
        duration = float(brief.get("duration_sec") or 0.0)
        stamps = [round((k + 0.5) * duration / n, 2) for k in range(n)]
        keep = list(range(n))

        if self.max_frames and n > self.max_frames:
            keep = [int((k + 0.5) * n / self.max_frames)
                    for k in range(self.max_frames)]

        return [(files[k], stamps[k]) for k in keep]


def cut_frames(dataset: Dataset, workers: int) -> int:
    """Cut every video in ranking/_manifest.json to 448x448 frames, resumably.

    Mid-point sampling (the convention VQA/extract.frames.py uses) and the same
    square resize InternVL's own video path does. The manifest already caps a
    video at 8 fps of source, because a 1 s clip cannot yield 64 frames.
    """
    if not shutil.which("ffmpeg"):
        print("ffmpeg is not on PATH (macOS: brew install ffmpeg)", file=sys.stderr)
        return 2

    manifest = json.loads((dataset.ranking / "_manifest.json")
                          .read_text(encoding="utf-8"))["videos"]
    print(f"{len(manifest)} videos, "
          f"{sum(v['n_frames'] for v in manifest.values())} frames "
          f"-> {dataset.frames}")

    def source(stem: str) -> Path | None:
        for name in (f"{stem}.mp4", f"{stem}.webm"):
            if (dataset.videos / name).exists():
                return dataset.videos / name

        hits = sorted(dataset.videos.glob(f"{stem}.*"))
        return hits[0] if hits else None

    def cut(item: tuple[str, dict]) -> str:
        stem, meta = item
        n, duration = int(meta["n_frames"]), float(meta["duration_sec"])
        out = dataset.frames / stem

        if len(list(out.glob("*.jpg"))) == n:
            return "skip"

        src = source(stem)

        if src is None:
            return "missing"

        out.mkdir(parents=True, exist_ok=True)

        for stale in out.glob("*.jpg"):
            stale.unlink()

        offset = duration / (2 * n)
        vf = (f"fps={n}/{duration}:start_time={offset:.4f},"
              f"scale={FRAME_SIZE}:{FRAME_SIZE}")
        cmd = ["ffmpeg", "-v", "error", "-an", "-sn", "-i", str(src),
               "-vf", vf, "-q:v", "3", "-frames:v", str(n),
               str(out / "%03d.jpg"), "-y"]
        subprocess.run(cmd, capture_output=True)
        got = len(list(out.glob("*.jpg")))
        return "ok" if got == n else "short" if got else "failed"

    counts: dict[str, int] = {}
    started = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, result in enumerate(pool.map(cut, manifest.items()), 1):
            counts[result] = counts.get(result, 0) + 1

            if done % 50 == 0 or done == len(manifest):
                print(f"  {done}/{len(manifest)}  {counts}  "
                      f"{time.monotonic() - started:.0f}s", flush=True)

    # A short cut is still usable -- frame_list reads what is on disk and
    # recomputes the times from it -- but a missing video leaves those queries
    # without frames, and they will fail over to the baseline.
    print(f"done: {counts}")
    return 0


# --- the model beside this agent ----------------------------------------------

class Model:
    """OpenAI-compatible chat against the server on this machine."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.url = args.endpoint.rstrip("/") + "/v1/chat/completions"
        self.model = args.model
        self.backend = args.backend
        self.transport = args.transport or ("base64" if args.backend == "lmstudio"
                                            else "file")
        self.media_root = args.media_root
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.timeout = args.timeout
        self.extra = json.loads(args.extra_body) if args.extra_body else {}
        self.local = threading.local()

    @property
    def session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def image_part(self, path: Path, stem: str) -> dict:
        if self.transport == "file":
            # llama-server --media-path <dataset>/frames448 resolves file://
            # URLs relative to that folder: ~50 bytes instead of ~50 KB.
            return {"type": "image_url",
                    "image_url": {"url": f"{self.media_root}{stem}/{path.name}"}}

        # LM Studio accepts nothing but base64. The frames are already 448x448
        # JPEGs, so they are sent as they are, not re-encoded.
        data = base64.b64encode(path.read_bytes()).decode()
        return {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{data}"}}

    def chat(self, images: list[dict], prompt: str, schema: dict | None,
             seed: int) -> dict:
        payload = {
            "model": self.model,
            # Images first and the system prompt fixed, so every round of one
            # query shares its prefix with the round before it and the server's
            # prompt cache skips re-reading the frames.
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user",
                          "content": images + [{"type": "text", "text": prompt}]}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": seed,
        }

        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "debate", "schema": schema}}

        if self.backend == "llamacpp":
            # LM Studio does not return logprobs (measured), llama-server does.
            payload["logprobs"] = True
            payload["top_logprobs"] = 1

        payload.update(self.extra)
        started = time.monotonic()
        response = self.session.post(self.url, json=payload, timeout=self.timeout)

        if response.status_code != 200:
            raise RuntimeError(f"{response.status_code}: {response.text[:300]}")

        body = response.json()
        choice = body["choices"][0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        logprobs = ((choice.get("logprobs") or {}).get("content")) or []
        values = [t["logprob"] for t in logprobs
                  if isinstance(t.get("logprob"), (int, float))]

        return {
            "content": (message.get("content") or "").strip(),
            "finish_reason": choice.get("finish_reason"),
            "elapsed": round(time.monotonic() - started, 2),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "logprob": round(sum(values) / len(values), 4) if values else None,
        }


# --- the judge ----------------------------------------------------------------

class Judge:
    def __init__(self, side: str, dataset: Dataset, model: Model,
                 retries: int) -> None:
        self.side = side
        self.dataset = dataset
        self.model = model
        self.retries = retries

    def context(self, brief: dict, frames: list[tuple[Path, float]]) -> str:
        listed = ", ".join(f"{t:.1f}s" for _, t in frames)
        parts = [f"This video is {float(brief['duration_sec']):.0f} seconds long. "
                 f"You are shown {len(frames)} frames sampled evenly across it, "
                 f"in order, at {listed}."]
        transcript = brief.get("transcript")

        if transcript and transcript.get("text"):
            language = transcript.get("language") or "unknown"
            parts.append(f"Spoken audio (automatic transcript, language: "
                         f"{language}):\n\"{transcript['text']}\"")
        else:
            parts.append("The video has no detected speech.")

        return "\n\n".join(parts) + "\n\n"

    @staticmethod
    def evidence(brief: dict) -> str:
        aid = brief.get("evidence") or {}
        lines = []

        if aid.get("counting"):
            lines.append(COUNTING_AID)

        audio = aid.get("audio_scores")

        if audio and len(audio) == brief["n_options"]:
            ranked = sorted(range(len(audio)), key=lambda i: -audio[i])
            listed = "; ".join(f"option {i + 1} {audio[i]:+.2f}" for i in ranked)
            lines.append(f"An audio model scored each option against the clip's "
                         f"soundtrack (higher means the sound matches the option "
                         f"text): {listed}. Treat it as one more piece of "
                         f"evidence, not a verdict.")

        return "\n" + "\n\n".join(lines) + "\n" if lines else ""

    def transcript(self, turns: list[dict]) -> str:
        """The shared debate, from this judge's point of view."""
        lines = []

        for turn in turns:
            who = "You" if turn.get("agent") == self.side else "The other adjudicator"
            cites = [f"{float(c):.1f}s" for c in (turn.get("cites") or [])[:6]
                     if isinstance(c, (int, float))]
            stance = f" ({turn['stance']})" if turn.get("stance") else ""
            cited = f" Cited {', '.join(cites)}." if cites else ""
            lines.append(f"Round {turn.get('round', 0)} -- {who}{stance}: option "
                         f"{int(turn['pick']) + 1} first, confidence "
                         f"{float(turn.get('confidence') or 0):.2f}. "
                         f"\"{turn.get('argument', '')}\"{cited}")

        return "\n".join(lines)

    def ask(self, q_id: int, prompt_for, seed: int) -> dict:
        brief = self.dataset.brief(q_id)
        n = brief["n_options"]
        frames = self.dataset.frame_list(brief)
        images = [self.model.image_part(path, brief["stem"]) for path, _ in frames]
        shared = dict(
            context=self.context(brief, frames),
            question=brief["question"],
            options="\n".join(f"{i + 1}. {o}"
                              for i, o in enumerate(brief["options"])),
            panel=brief["panel_table"],
            evidence=self.evidence(brief),
            card=f"\n{self.dataset.card}\n" if self.dataset.card else "",
            routing=brief.get("routing") or "",
            n=n, slots=", ".join(str(i) for i in range(1, n + 1)))
        prompt = prompt_for(shared)
        last: Exception | None = None

        for attempt in range(self.retries):
            try:
                out = self.model.chat(images, prompt, debate_schema(n),
                                      seed + attempt)
                reply = parse_json_object(out["content"])
                order, repaired = as_permutation(reply.get("ranking"), n)
                cites = [c for c in (reply.get("cites") or [])
                         if isinstance(c, (int, float))]
                return {
                    "agent": self.side,
                    "ranking": order,
                    "pick": order[0],
                    "confidence": reply.get("confidence"),
                    "cites": cites[:8],
                    "argument": str(reply.get("argument") or "")[:400],
                    "stance": str(reply.get("stance") or "")[:16],
                    "repaired": repaired,
                    "n_frames": len(frames),
                    "attempts": attempt + 1,
                    **{k: out[k] for k in ("elapsed", "prompt_tokens",
                                           "completion_tokens",
                                           "reasoning_tokens", "logprob",
                                           "finish_reason")},
                }
            except (requests.RequestException, RuntimeError, ValueError,
                    KeyError, IndexError) as exc:
                last = exc

                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"judge {self.side}: {type(last).__name__}: {last}")

    def opinion(self, q_id: int, seed: int) -> dict:
        statement = self.ask(q_id, lambda s: OPINION_TEMPLATE.format(**s),
                             seed_for(q_id, seed))
        statement["round"] = 0
        return statement

    def rebut(self, q_id: int, round_no: int, rounds: int, turns: list[dict],
              seed: int) -> dict:
        kind = "final" if round_no >= rounds else "rebuttal"
        text = self.transcript(turns)
        statement = self.ask(
            q_id,
            lambda s: ROUND_TEMPLATE.format(transcript=text, round=round_no,
                                            rounds=rounds,
                                            instruction=INSTRUCTIONS[kind], **s),
            seed_for(q_id, seed + 10 * round_no))
        statement["round"] = round_no
        return statement


# --- the HTTP face the orchestrator talks to ----------------------------------

def make_handler(judge: Judge, info: dict):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            if self.path.rstrip("/") == "/health":
                self.reply(200, info)
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(length) or b"{}")
                q_id = int(request["q_id"])
                seed = int(request.get("seed", 0))

                if self.path.rstrip("/") == "/opinion":
                    self.reply(200, judge.opinion(q_id, seed))
                elif self.path.rstrip("/") == "/round":
                    self.reply(200, judge.rebut(q_id, int(request["round"]),
                                                int(request.get("rounds", 2)),
                                                request.get("transcript") or [],
                                                seed))
                else:
                    self.reply(404, {"error": "not found"})
            except FileNotFoundError as exc:
                self.reply(404, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - report, keep serving
                self.reply(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {judge.side} "
                             f"{fmt % args}\n")

    return Handler


def selftest(judge: Judge, dataset: Dataset) -> int:
    """Gate 0: does the model see? Then time one real opinion.

    A server that drops image parts still answers, from the text alone, so a
    200 proves nothing. A real frame and a solid grey one must get different
    descriptions, and the grey one must be recognised as blank.
    """
    from PIL import Image

    index = json.loads((dataset.ranking / "_index.json").read_text(encoding="utf-8"))
    first = index["queries"][0]
    brief = dataset.brief(first["Q_ID"])
    path, _ = dataset.frame_list(brief)[len(dataset.frame_list(brief)) // 2]

    buffer = io.BytesIO()
    Image.new("RGB", (FRAME_SIZE, FRAME_SIZE), (128, 128, 128)).save(buffer, "JPEG")
    grey = {"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()}}
    real = judge.model.image_part(path, brief["stem"])
    question = "Describe what is in this image in one short sentence."

    answers = {}
    for name, part in (("real", real), ("grey", grey)):
        out = judge.model.chat([part], question, None, 1)
        answers[name] = out["content"]
        print(f"{name}: {out['elapsed']}s  {out['prompt_tokens']} prompt tokens  "
              f"-> {out['content'][:160]!r}")

    blank_words = ("gray", "grey", "blank", "solid", "plain", "empty", "nothing")
    sees = (answers["real"] and answers["real"] != answers["grey"]
            and any(w in answers["grey"].lower() for w in blank_words))
    print("VISION: " + ("OK" if sees else "FAILED -- the model does not appear "
                        "to see images; stop here"))

    if not sees:
        return 1

    statement = judge.opinion(first["Q_ID"], 0)
    print(f"\nopinion on Q{first['Q_ID']}: {statement['n_frames']} frames, "
          f"{statement['elapsed']}s, prompt {statement['prompt_tokens']}, "
          f"completion {statement['completion_tokens']} "
          f"(reasoning {statement['reasoning_tokens']}), "
          f"pick {statement['pick'] + 1}, repaired {statement['repaired']}")
    print(f"  argument: {statement['argument']}")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True,
                    help="this machine's videos_TEST_VQA folder")
    ap.add_argument("--cut-frames", action="store_true",
                    help="cut frames448/ from videos/ per ranking/_manifest.json, then exit")
    ap.add_argument("--cut-workers", type=int, default=6)
    ap.add_argument("--side", choices=["a", "b"])
    ap.add_argument("--backend", choices=["lmstudio", "llamacpp"])
    ap.add_argument("--endpoint", help="the model server beside this agent")
    ap.add_argument("--model")
    ap.add_argument("--transport", choices=["base64", "file"],
                    help="default: base64 for lmstudio, file for llamacpp")
    ap.add_argument("--media-root", default="file://",
                    help="prefix for file:// frames; llama-server resolves it "
                         "against its --media-path")
    ap.add_argument("--max-frames", type=int,
                    help="subsample the cut frames to at most this many per call")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="must leave room for a reasoning trace before the JSON")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--extra-body", help="JSON merged into every request, e.g. "
                    "'{\"chat_template_kwargs\": {\"enable_thinking\": false}}'")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7801)
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset = Dataset(args.dataset, args.max_frames)

    if not dataset.ranking.is_dir():
        print(f"no ranking folder at {dataset.ranking}", file=sys.stderr)
        return 2

    if args.cut_frames:
        return cut_frames(dataset, args.cut_workers)

    missing = [f for f in ("side", "backend", "endpoint", "model")
               if not getattr(args, f)]

    if missing:
        print(f"serving needs --{', --'.join(missing)}", file=sys.stderr)
        return 2

    judge = Judge(args.side, dataset, Model(args), args.retries)

    if args.selftest:
        return selftest(judge, dataset)

    index = json.loads((dataset.ranking / "_index.json").read_text(encoding="utf-8"))
    cut = sum(1 for d in dataset.frames.iterdir() if d.is_dir()) \
        if dataset.frames.is_dir() else 0
    info = {"side": args.side, "backend": args.backend, "model": args.model,
            "transport": judge.model.transport, "briefs": index["n"],
            "videos_cut": cut, "max_frames": args.max_frames,
            "dataset": str(args.dataset)}
    print(json.dumps(info, indent=1))

    server = ThreadingHTTPServer((args.host, args.port), make_handler(judge, info))
    print(f"judge {args.side} listening on {args.host}:{args.port}", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

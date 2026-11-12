#!/usr/bin/env python3
"""Generate paraphrase variations of the Task 1 questions with a local LLM.

For every question in ``vqa.2026.testingDataset.Task1.json`` this asks a model
served by Ollama for ten alternative ways of asking the same thing, and writes
one record per query:

    [
        {
            "Q_ID": 1,
            "Video_ID": "8384.webm",
            "Question": "What happens to the image of the bride and groom?",
            "Variations": ["...", ...]
        },
        ...
    ]

The point of the variations is downstream: asking a VQA model the same question
ten ways and pooling what comes back is a cheap ensemble, and the run format
wants ten ranked answers per query anyway.

Work is keyed on the question text, not the Q_ID -- 2,443 queries contain only
2,350 distinct questions, and a repeated question gets one generation shared by
every Q_ID that asks it. Results are appended to a JSONL cache as they land, so
an interrupted run resumes where it stopped.

Usage:
    python VQA/generate.query.variations.py --limit 20     # smoke test
    python VQA/generate.query.variations.py                # everything, resumable
    python VQA/generate.query.variations.py --only 3 17    # named Q_IDs
    python VQA/generate.query.variations.py --rebuild      # re-emit JSON from cache
"""

from __future__ import annotations

import argparse
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

# Gemma 4 is served either by Ollama on Windows or by vLLM in WSL2, and vLLM
# is the default -- see VQA/GEMMA4_VLLM_SETUP.md. The two load different
# quantisations, so a backend switch is a different set of paraphrases, not a
# resumable continuation of the same one: each keeps its own cache and output.
# query.variations.task1.json, the one every existing run was built on, is the
# Ollama one.
DEFAULT_BACKEND = "vllm"

BACKENDS = {
    "ollama": {
        "model": "gemma4:12b",
        "host": "http://localhost:11434",
        "out": "query.variations.task1.json",
        "cache": "query.variations.cache.jsonl",
        "workers": 4,
    },
    "vllm": {
        "model": "google/gemma-4-12B-it-qat-w4a16-ct",
        "host": "http://localhost:8010",
        "out": "query.variations.vllm.task1.json",
        "cache": "query.variations.vllm.cache.jsonl",
        "workers": 8,
    },
}

DEFAULT_VARIATIONS = 10
DEFAULT_ROUNDS = 3

# A rewrite far longer than the original has usually grown an explanation.
MAX_CHARS = 300
MIN_CHARS = 8

# Models like to number their output even when the schema says plain strings.
ENUM_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")

SYSTEM_PROMPT = (
    "You rewrite questions that people asked about a video. "
    "You reply with JSON only."
)

# Written against the failure modes this question set actually provokes: losing
# the wh-type ("how many times" -> "does he cut himself"), dropping the detail
# that picks out the subject ("the woman in the green coat" -> "the woman"), and
# answering instead of rewriting.
PROMPT_TEMPLATE = """Rewrite this question in {n} different ways.

Question: {question}

Rules:
- Every rewrite must have exactly the same correct answer as the original.
- Keep the question type. "How many" stays a counting question, "why" stays
  causal, "when" stays temporal, "where" stays spatial, and a yes/no question
  stays a yes/no question.
- Keep every detail that identifies what is being asked about: people, clothing,
  colours, objects, actions, and ordering words such as "first", "after" or
  "at the end".
- Add nothing the original does not say, and drop nothing it does.
- Vary the wording and the sentence structure, not one word. Some rewrites may
  open with "Identify", "Describe" or "State" instead of a question word.
- The original may have spelling or grammar mistakes. Write clean English, but
  keep what it was asking.
- Never answer the question, and never comment on it.
- Each rewrite is a single self-contained sentence, different from the original
  and from all the other rewrites.

Reply with JSON: {{"variations": ["...", ...]}} holding exactly {n} strings."""

TOPUP_TEMPLATE = """Rewrite this question in {n} more ways.

Question: {question}

These rewrites already exist. Yours must be worded differently from all of them:
{existing}

Same rules as before: identical meaning, same question type, every identifying
detail kept, nothing added, no answers, one clean sentence each.

Reply with JSON: {{"variations": ["...", ...]}} holding exactly {n} strings."""


def schema(n: int) -> dict:
    """JSON schema constraining the reply to exactly `n` strings."""
    return {
        "type": "object",
        "properties": {
            "variations": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": n,
                "maxItems": n,
            }
        },
        "required": ["variations"],
    }


def normalize(text: str) -> str:
    """Collapse whitespace and case for duplicate detection."""
    return " ".join(text.split()).casefold()


def question_seed(question: str) -> int:
    """Stable per-question seed, so a re-run reproduces the same variations."""
    digest = hashlib.sha256(question.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def clean(text: str) -> str:
    """Strip list markers, surrounding quotes and stray whitespace."""
    text = " ".join(str(text).split())
    text = ENUM_PREFIX_RE.sub("", text)

    if len(text) > 1 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()

    return text


def accept(candidate: str, seen: set[str]) -> str | None:
    """Return the cleaned variation if it is usable and new, else None."""
    text = clean(candidate)

    if not MIN_CHARS <= len(text) <= MAX_CHARS:
        return None

    key = normalize(text)

    if key in seen:
        return None

    seen.add(key)
    return text


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

    def variations(self, prompt: str, n: int, seed: int) -> list[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "format": schema(n),
            "stream": False,
            # gemma4 and friends reason before answering, and a paraphrase task
            # gains nothing from it: thinking took 19 s per question and blew
            # the token budget before a single variation was emitted, against
            # 5 s and a full set with it off.
            "think": self.think,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.95,
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
            hint = (" -- raise --num-predict, or pass --no-think"
                    if reason == "length" else "")
            raise ValueError(f"empty content (done_reason={reason}){hint}")

        parsed = json.loads(content)
        found = parsed.get("variations", [])

        if not isinstance(found, list):
            raise ValueError(f"'variations' is {type(found).__name__}, not a list")

        return found


class VLLM:
    """OpenAI-compatible client for vLLM. Same contract as Ollama above.

    The prompts, schema and seeds are shared with the Ollama path; only the
    wire format differs. See the VLLM class in VQA/generate.answers.py for the
    full account -- here it reduces to four things: the schema travels in
    ``response_format`` and is closed with ``additionalProperties: false``
    because vLLM compiles it into a grammar, the sampler options are flat
    rather than nested under ``options``, thinking is turned off through the
    chat template variable Gemma 4 actually reads, and the reply arrives under
    ``choices[0]`` instead of ``message``.
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

    def variations(self, prompt: str, n: int, seed: int) -> list[str]:
        constraint = schema(n)
        constraint["additionalProperties"] = False

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.num_predict,
            "temperature": self.temperature,
            "top_p": 0.95,
            "seed": seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "variations", "schema": constraint},
            },
            "chat_template_kwargs": {"enable_thinking": self.think},
        }

        response = self.session.post(self.url, json=payload,
                                     timeout=self.timeout)

        # A 400 is the server refusing the request itself; it will refuse it
        # identically on a retry, so say so rather than time out three times.
        if response.status_code == 400:
            raise ValueError(f"400: {response.text[:300]}")

        response.raise_for_status()

        choice = response.json()["choices"][0]
        content = (choice["message"].get("content") or "").strip()

        if not content:
            reason = choice.get("finish_reason")
            hint = (" -- raise --num-predict, or pass --no-think"
                    if reason == "length" else "")
            raise ValueError(f"empty content (finish_reason={reason}){hint}")

        found = json.loads(content).get("variations", [])

        if not isinstance(found, list):
            raise ValueError(f"'variations' is {type(found).__name__}, not a list")

        return found


def preflight(host: str, model: str, timeout: float = 10.0) -> None:
    """Refuse to start unless the vLLM server is up and serving this model.

    Without it an absent server is close to silent: every question fails one
    at a time, the run finishes with an empty cache, and the only evidence is
    a wall of FAILED lines that scroll past. See VQA/GEMMA4_VLLM_SETUP.md §3 --
    WSL2 tears the service down whenever the last session exits.
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


def generate(client: Ollama | VLLM, question: str, want: int,
             max_rounds: int) -> tuple[list[str], int]:
    """Collect `want` distinct variations, topping up until the model runs dry.

    Returns the variations and the number of model calls it took. A short list
    is returned rather than padded -- the caller decides what to do with it.
    """
    # The original counts as taken, so no "variation" is just the question back.
    seen = {normalize(question)}
    kept: list[str] = []
    seed = question_seed(question)
    calls = 0

    for round_no in range(max_rounds):
        missing = want - len(kept)

        if missing <= 0:
            break

        if round_no == 0:
            prompt = PROMPT_TEMPLATE.format(n=want, question=question)
        else:
            existing = "\n".join(f"- {v}" for v in kept)
            prompt = TOPUP_TEMPLATE.format(
                n=missing, question=question, existing=existing
            )

        # Reuse of a seed with an identical prompt would replay the same reply.
        candidates = client.variations(prompt, missing, seed + round_no)
        calls += 1

        for candidate in candidates:
            text = accept(candidate, seen)

            if text is not None:
                kept.append(text)

            if len(kept) == want:
                break

    return kept, calls


def load_queries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_cache(path: Path) -> dict[str, dict]:
    """Read the JSONL cache, keyed by normalized question.

    Later records win, so re-running a question with --force supersedes the
    earlier attempt without having to rewrite the file.
    """
    cache: dict[str, dict] = {}

    if not path.exists():
        return cache

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                cache[normalize(record["question"])] = record
            except (json.JSONDecodeError, KeyError):
                print(f"  skipping malformed cache line {line_no}",
                      file=sys.stderr)

    return cache


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate paraphrase variations of the Task 1 questions."
    )

    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES,
                    help="released query JSON (default: %(default)s)")
    ap.add_argument("--out", type=Path,
                    help="assembled per-query output (default: the backend's)")
    ap.add_argument("--cache", type=Path,
                    help="JSONL work cache, one line per question (default: "
                         "the backend's)")

    ap.add_argument("--backend", choices=sorted(BACKENDS),
                    default=DEFAULT_BACKEND,
                    help="which server answers: vllm in WSL2, or ollama on "
                         "Windows (default: %(default)s). Sets the default "
                         "model, host, workers, cache and output")
    ap.add_argument("--model", help="model tag (default: the backend's)")
    ap.add_argument("--host", help="server host (default: the backend's)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="sampling temperature; lower is safer, higher is more "
                         "varied (default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="per-request timeout in seconds (default: %(default)s)")
    ap.add_argument("--think", action="store_true",
                    help="let a reasoning model think first; slower, and needs "
                         "a much larger --num-predict")
    ap.add_argument("--num-predict", type=int, default=1200,
                    help="token budget per reply (default: %(default)s)")

    ap.add_argument("-n", "--variations", type=int, default=DEFAULT_VARIATIONS,
                    help="variations per question (default: %(default)s)")
    ap.add_argument("--max-rounds", type=int, default=DEFAULT_ROUNDS,
                    help="model calls per question before accepting a short "
                         "list (default: %(default)s)")
    ap.add_argument("--workers", type=int,
                    help="concurrent requests (default: 4 on ollama, which "
                         "needs OLLAMA_NUM_PARALLEL to match; 8 on vllm)")

    ap.add_argument("--limit", type=int, help="process at most N questions")
    ap.add_argument("--only", type=int, nargs="+", metavar="Q_ID",
                    help="process only the questions behind these Q_IDs")
    ap.add_argument("--force", action="store_true",
                    help="regenerate questions already in the cache")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the output JSON from the cache, no inference")

    return ap.parse_args(argv)


def assemble(queries: list[dict], cache: dict[str, dict],
             want: int) -> tuple[list[dict], int, int]:
    """Fan the per-question variations back out to every Q_ID."""
    records: list[dict] = []
    missing = 0
    short = 0

    for query in queries:
        question = query["Question"]
        record = cache.get(normalize(question))
        variations = record["variations"] if record else []

        if not record:
            missing += 1
        elif len(variations) < want:
            short += 1

        records.append(
            {
                "Q_ID": query["Q_ID"],
                "Video_ID": query["Video_ID"],
                "Question": question,
                "Variations": variations,
            }
        )

    return records, missing, short


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

    print(f"queries: {len(queries)} | distinct questions: {len(questions)} | "
          f"cached: {sum(1 for k in questions if k in cache)}")

    if not args.rebuild:
        todo = [e for k, e in questions.items()
                if args.force or k not in cache]

        if args.only:
            wanted = set(args.only)
            todo = [e for e in todo if wanted & set(e["q_ids"])]

        if args.limit:
            todo = todo[:args.limit]

        print(f"to generate: {len(todo)}")

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

            def work(entry: dict) -> None:
                nonlocal done, failed

                question = entry["question"]
                t0 = time.monotonic()

                try:
                    variations, calls = generate(
                        client, question, args.variations, args.max_rounds
                    )
                except Exception as exc:  # noqa: BLE001 - one bad question
                    with lock:                # must not sink the whole run
                        failed += 1
                        print(f"  FAILED {question!r}: "
                              f"{type(exc).__name__}: {exc}", file=sys.stderr)
                    return

                record = {
                    "question": question,
                    "variations": variations,
                    "q_ids": entry["q_ids"],
                    "calls": calls,
                    "model": args.model,
                    "elapsed": round(time.monotonic() - t0, 2),
                    "ts": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"),
                }

                with lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    cache[normalize(question)] = record

                    done += 1
                    rate = done / max(time.monotonic() - started, 1e-9)
                    flag = "" if len(variations) == args.variations \
                        else f" SHORT({len(variations)})"
                    print(f"  [{done}/{len(todo)}] {rate * 60:5.1f}/min "
                          f"{record['elapsed']:6.1f}s{flag}  {question[:60]}")

            try:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    list(pool.map(work, todo))
            except KeyboardInterrupt:
                print("\ninterrupted -- finished questions are cached, "
                      "re-run to resume", file=sys.stderr)
                return 130
            finally:
                handle.close()

            elapsed = time.monotonic() - started
            print(f"\ngenerated {done} ({failed} failed) in {elapsed / 60:.1f} min")

    records, missing, short = assemble(queries, cache, args.variations)

    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=4, ensure_ascii=False)
        handle.write("\n")

    print(f"\nwrote {len(records)} query entries to {args.out}")
    print(f"  complete ({args.variations} variations): "
          f"{len(records) - missing - short}")
    print(f"  short (fewer than {args.variations}): {short}")
    print(f"  not generated yet:                    {missing}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

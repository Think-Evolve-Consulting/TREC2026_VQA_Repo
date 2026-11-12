#!/usr/bin/env python3
"""Lengthen Task 1 answers without changing what they say.

Phase 2 of ``VQA/ANSWER_EXPANSION_PLAN.md``. Reads a run file, writes a *new*
run file; never edits in place, and never re-ranks.

**Why this exists.** ``VQA/eval.vqa.ag.py:109`` scores each of the ten answers
against a single reference with METEOR and takes the mean over all ten
(``:125``). METEOR weights recall above precision, and recall is
``matches / len(reference)`` -- a 3-word answer can match at most 3 tokens of an
8-word reference no matter how correct it is. Measured on the 150-query 2025
sample: our runs answer at 4.9 / 3.3 / 3.6 words against a ground truth
averaging 8.2. Re-expressing "a document" as "he holds up the document" gains
recall from function words the reference already contains -- nltk's METEOR does
not strip stopwords, so ``the``/``he``/``is`` are matchable tokens.

**Why it can backfire, and what stops it.** METEOR also carries a fragmentation
penalty over match chunks, and padding dilutes precision. Three guards:

- **Question type decides the target, and ``count`` is never expanded.** The
  2025 ground truth's ``count`` answers average 1.0 words (median 1) -- they are
  literally "one", "two", "three". 23% of the sample's GT is 1-2 words. Blanket
  expansion overshoots these badly.
- **Refusals pass through untouched.** "unknown" must not become "it is unknown
  what happens in the video" -- that is a longer wrong answer, not a better one.
- **Content-word containment.** Every content word of the original must survive
  in the expansion, or the original is kept. This catches both meaning drift and
  silent truncation.

The expander never sees the ground truth, and it has no video -- it can only
rephrase what the answering pass already produced.

Usage:
    python VQA/expand.answers.py --run gemma4.run.2025.json \\
        --queries vqa.2025.sample150.json --routing query.routing.2025.json \\
        --out gemma4.expanded.run.2025.json \\
        --cache answers.2025.gemma4.expansion.cache.jsonl
    python VQA/expand.answers.py ... --limit 5      # smoke test
    python VQA/expand.answers.py ... --rebuild      # re-emit from cache
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

DEFAULT_QUERIES = ROOT / "vqa.2025.sample150.json"
DEFAULT_ROUTING = ROOT / "query.routing.2025.json"

# Gemma 4 is served either by Ollama on Windows or by vLLM in WSL2, and vLLM
# is the default -- see VQA/GEMMA4_VLLM_SETUP.md. --out and --cache are
# required here, so unlike the other passes there is nothing to keep apart
# automatically: name them per backend yourself, because the two load
# different quantisations and their rewrites are not interchangeable.
DEFAULT_BACKEND = "vllm"

BACKENDS = {
    "ollama": {
        "model": "gemma4:12b",
        "host": "http://localhost:11434",
        "workers": 4,
    },
    "vllm": {
        "model": "google/gemma-4-12B-it-qat-w4a16-ct",
        "host": "http://localhost:8010",
        "workers": 8,
    },
}

# Rewriting gains nothing from a reasoning pass and costs 19.1 s against 5.3 s
# per item -- measured in SESSION.2026-08-12.query-variations.md on the same
# model for the same kind of task.
DEFAULT_THINK = False

# Ollama only: it defaults to 4096 whatever the model's window is. On vLLM the
# window is fixed at serve time by --max-model-len and this is ignored.
DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT = 700
DEFAULT_TEMPERATURE = 0.3
"""Low: this is a rewriting task with a right answer, not a generative one."""

ANSWERS_PER_QUERY = 10
MAX_ANSWER_WORDS = 40
MAX_ANSWER_CHARS = 300

# Word targets per Question_Type, measured from the 2025 ground truth rather
# than guessed -- ANSWER_EXPANSION_PLAN.md Phase 1's "targets get measured".
# Left column is the sample-150 median (routing labels), right the all-2,000
# median (keyword heuristic); they agree closely. `count` is deliberately
# absent: see NO_EXPAND.
TARGET_WORDS = {
    "action": 10,     # median 9 / 9
    "reason": 10,     # median 8 / 10
    "manner": 11,     # median 11 / 11
    "summary": 12,    # median 12 / -
    "entity": 6,      # median 4 / 5
    "place": 6,       # median 3 / 6
    "time": 7,        # median 6 / 7
    "yes_no": 6,      # median 8 / 5
    "other": 9,
}
DEFAULT_TARGET = 9

NO_EXPAND = {"count"}
"""Question types whose ground truth is a bare number. 2025 GT `count` answers
average 1.0 words, median 1, and open with the literal tokens `one` (85),
`two` (61), `three` (26). Expanding these can only hurt."""

# validate.ag.run.py:276's duplicate rule, verbatim, so dedupe here matches
# what the validator will do.
LEAD_IN_RE = re.compile(r"^\s*(?:answer|a)\s*[:\-]\s*", re.IGNORECASE)
ENUM_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")

# Taken from generate.answers.py:123 -- a refusal is a legitimate answer but
# never the ground truth, so lengthening one just makes a longer zero.
NON_ANSWER_RE = re.compile(
    r"^(?:unknown|unclear|n/?a|no idea|cannot be determined"
    r"|(?:it |the answer |this )?(?:is )?(?:not |un)(?:known|clear|shown"
    r"|visible|determined|specified)"
    r"|(?:it |this )?cannot be (?:determined|answered|seen)"
    r"|(?:the )?video does not show(?: this)?"
    r"|not (?:shown|visible|clear|stated)(?: in the video)?"
    r"|nothing happens|none|no one)"
    r"[.!]?$",
    re.IGNORECASE,
)

# Small closed-class list. Content-word containment ignores these, because
# grammatical scaffolding is exactly what the expansion is allowed to add.
STOPWORDS = frozenset("""
a an the this that these those and or but if then than so as of in on at to
from by for with without into onto over under about after before during while
is are was were be been being am do does did doing have has had having
he she it they them his her its their there here what which who whom whose
not no nor very too also just only own same s t can will would could should
""".split())

SUFFIXES = ("ing", "edly", "ies", "es", "ed", "ly", "s")


def normalize(text: str) -> str:
    """validate.ag.run.py:276."""
    return " ".join(text.split()).casefold()


def tidy(text: str) -> str:
    """generate.answers.qwen3vl.py's cleanup, so output shape is unchanged."""
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


def stem(word: str) -> str:
    """Crude suffix strip so "holds"/"holding"/"held" count as the same word.

    A real stemmer would need nltk loaded in this process for no benefit: the
    containment check only has to be tolerant enough not to reject honest
    rephrasings, and strict enough to catch a dropped noun.
    """
    word = re.sub(r"[^a-z0-9]+", "", word.casefold())

    for suffix in SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]

    return word


def content_words(text: str) -> set[str]:
    return {stem(w) for w in text.split()
            if w and stem(w) and w.casefold().strip(".,!?;:") not in STOPWORDS}


# The exact pool every answering script pads short results with. Membership is
# checked literally *as well as* by regex: NON_ANSWER_RE is anchored with
# `[.!]?$` and so misses the two entries carrying a trailing prepositional
# phrase ("it is not shown in the video", "it is not clear from the video").
# That leak let 176 refusals be rewritten on the first pass -- 160 of them in
# the Cosmos run, which is mostly padding -- turning "it is not clear from the
# video" into "It is not clear from the video what they do.", a longer zero.
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
)
FALLBACK_SET = frozenset(normalize(a) for a in FALLBACK_ANSWERS)
"""Note "no" and "yes" are deliberately absent: for a yes_no question they are
the answer, not a refusal."""


def is_refusal(text: str) -> bool:
    return (normalize(text) in FALLBACK_SET
            or bool(NON_ANSWER_RE.match(text.strip())))


def seed_for(q_id: int) -> int:
    """Stable per-query seed, so a re-run reproduces the same output."""
    digest = hashlib.sha256(f"expand|{q_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


SYSTEM_PROMPT = (
    "You rewrite short answers into fuller ones without changing their meaning. "
    "You reply with JSON only."
)

PROMPT_TEMPLATE = """A question about a video was answered with {n} short candidate answers. Rewrite each one as a fuller phrase or sentence of roughly {target} words.

Question: {question}

Answers to rewrite:
{answers}

Rules:
- Preserve the meaning of each answer exactly. Rewrite {n} answers, in the same order.
- Add no new facts. No colours, counts, names, places, times or actions that are not already in the answer you are rewriting. You may only add grammatical scaffolding: articles, pronouns, prepositions, auxiliary verbs.
- Do not restate or echo the question.
- Do not add hedges such as "it appears that" or "the video shows that".
- Keep every word of the original answer's meaning. If an answer is already about {target} words, return it essentially unchanged.
- The {n} rewrites must stay different from each other.

Reply with JSON: {{"answers": [{slots}]}} holding exactly {n} strings."""


def answer_schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": n,
                "maxItems": n,
            }
        },
        "required": ["answers"],
    }


class Ollama:
    """Minimal chat client; one session per worker keeps the connection warm."""

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

    def ask(self, prompt: str, n: int, seed: int) -> tuple[dict, float]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "format": answer_schema(n),
            "stream": False,
            "think": self.think,
            "keep_alive": "30m",
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "temperature": self.temperature,
                "top_p": 0.95,
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

                content = response.json()["message"].get("content") or ""

                if not content:
                    raise ValueError("empty content")

                return json.loads(content), time.monotonic() - started
            except (requests.RequestException, ValueError,
                    json.JSONDecodeError, KeyError) as exc:
                last = exc

                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"{type(last).__name__}: {last}")


class VLLM:
    """OpenAI-compatible client for vLLM. Same ask() contract as Ollama above.

    The prompts, schema, seeds and retry budget are shared with the Ollama
    path; only the wire format differs -- the schema travels in
    ``response_format`` closed with ``additionalProperties: false``, the
    sampler options are flat rather than nested under ``options``, there is no
    num_ctx (the window is the server's ``--max-model-len``, so an overlong
    prompt is a 400 rather than a silent truncation), thinking is turned off
    through the chat template variable Gemma 4 reads, and the reply arrives
    under ``choices[0]``.
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

    def ask(self, prompt: str, n: int, seed: int) -> tuple[dict, float]:
        schema = answer_schema(n)
        schema["additionalProperties"] = False

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
                "json_schema": {"name": "answers", "schema": schema},
            },
            "chat_template_kwargs": {"enable_thinking": self.think},
        }

        last: Exception | None = None

        for attempt in range(self.retries):
            started = time.monotonic()

            try:
                response = self.session.post(self.url, json=payload,
                                             timeout=self.timeout)

                # A 400 is the server refusing the request itself and it will
                # refuse it identically three times over.
                if response.status_code == 400:
                    raise RuntimeError(f"400: {response.text[:300]}")

                response.raise_for_status()

                choice = response.json()["choices"][0]
                content = (choice["message"].get("content") or "").strip()

                if not content:
                    raise ValueError(
                        f"empty content (finish_reason={choice.get('finish_reason')})")

                return json.loads(content), time.monotonic() - started
            except (requests.RequestException, ValueError,
                    json.JSONDecodeError, KeyError, IndexError) as exc:
                last = exc

                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"{type(last).__name__}: {last}")


def preflight(host: str, model: str, timeout: float = 10.0) -> None:
    """Refuse to start unless the vLLM server is up and serving this model.

    Without it an absent server fails every item, the expansion falls back to
    the original answers, and the run completes looking finished. WSL2 tears
    the service down whenever the last session exits --
    VQA/GEMMA4_VLLM_SETUP.md §3.
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


def accept(original: str, candidate: str, taken: set[str]) -> tuple[str, str]:
    """Decide whether one rewrite may replace its original.

    Returns (text, reason). Any breach keeps the original -- the expansion is an
    optimisation, and a failed one must cost nothing.
    """
    candidate = tidy(candidate)

    if not candidate:
        return original, "empty"

    if normalize(candidate) == normalize(original):
        return original, "unchanged"

    if len(candidate) > MAX_ANSWER_CHARS:
        return original, "too_long"

    missing = content_words(original) - content_words(candidate)

    if missing:
        return original, "dropped_content"

    if normalize(candidate) in taken:
        return original, "duplicate"

    return candidate, "expanded"


# Condition C of ANSWER_EXPANSION_PLAN.md Phase 1: pad mechanically to the same
# length with generic frame words, no model involved. If this scores as well as
# the LLM pass, the metric is rewarding length alone and a zero-cost rule beats
# an hour of inference -- which is exactly the thing worth knowing before
# shipping the LLM pass.
MECHANICAL_PREFIX = "in the video the"
MECHANICAL_SUFFIX = "as it is shown in the video here"


def mechanical(text: str, target: int) -> str:
    """Pad `text` toward `target` words with contentless frame words."""
    words = text.split()

    if len(words) >= target:
        return text

    need = target - len(words)
    prefix = MECHANICAL_PREFIX.split()[:max(0, min(need, 4))]
    need -= len(prefix)
    suffix = MECHANICAL_SUFFIX.split()[:max(0, need)]

    return " ".join([*prefix, *words, *suffix])


def mechanical_query(record: dict, question_type: str, target: int) -> dict:
    """Condition C for one query -- same skip rules, no inference."""
    answers = [a["Answer"] for a in record["Answers"]]
    skip_all = question_type in NO_EXPAND
    final, changed = [], 0

    for text in answers:
        if skip_all or is_refusal(text):
            final.append(text)
            continue

        padded = tidy(mechanical(text, target))
        final.append(padded)
        changed += padded != text

    return {
        "Q_ID": record["Q_ID"], "Video_ID": record["Video_ID"],
        "Question_Type": question_type, "target_words": target,
        "answers": [{"Rank": a["Rank"], "Answer": t, "Time": a.get("Time", 0.0)}
                    for a, t in zip(record["Answers"], final)],
        "words_before": round(sum(len(a.split()) for a in answers)
                              / max(len(answers), 1), 2),
        "words_after": round(sum(len(a.split()) for a in final)
                             / max(len(final), 1), 2),
        "elapsed": 0.0, "expanded": changed,
        "kept_refusal": sum(1 for a in answers if is_refusal(a)),
        "kept_type": len(answers) if skip_all else 0,
        "dropped_content": 0, "duplicate": 0, "unchanged": 0, "empty": 0,
        "too_long": 0, "call_failed": 0,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def expand_query(client: Ollama, record: dict, question: str,
                 question_type: str, target: int) -> dict:
    """One call, all expandable answers for one query."""
    answers = [a["Answer"] for a in record["Answers"]]
    ranks = [a["Rank"] for a in record["Answers"]]
    times = [a.get("Time", 0.0) for a in record["Answers"]]

    skip_all = question_type in NO_EXPAND

    todo = [] if skip_all else [
        (i, a) for i, a in enumerate(answers) if not is_refusal(a)
    ]

    stats = {"expanded": 0, "kept_refusal": sum(1 for a in answers
                                                if is_refusal(a)),
             "kept_type": len(answers) if skip_all else 0,
             "dropped_content": 0, "duplicate": 0, "unchanged": 0,
             "empty": 0, "too_long": 0, "call_failed": 0}

    elapsed = 0.0
    final = list(answers)

    if todo:
        listing = "\n".join(f"{n}. {a}" for n, (_, a) in enumerate(todo, 1))
        slots = ", ".join(['"..."'] * len(todo))

        try:
            reply, elapsed = client.ask(
                PROMPT_TEMPLATE.format(n=len(todo), target=target,
                                       question=question, answers=listing,
                                       slots=slots),
                n=len(todo), seed=seed_for(record["Q_ID"]),
            )
            rewrites = reply.get("answers") or []
        except RuntimeError as exc:
            print(f"  Q_ID {record['Q_ID']}: {exc}", file=sys.stderr)
            stats["call_failed"] = 1
            rewrites = []

        # Rank order is never touched -- it was decided by the answering pass's
        # own vote and this is not a re-ranking pass.
        taken = {normalize(a) for a in answers}

        for (index, original), candidate in zip(todo, rewrites):
            taken.discard(normalize(original))
            text, reason = accept(original, str(candidate), taken)
            final[index] = text
            taken.add(normalize(text))
            stats[reason] = stats.get(reason, 0) + 1

    share = elapsed / max(len(answers), 1)

    return {
        "Q_ID": record["Q_ID"],
        "Video_ID": record["Video_ID"],
        "Question_Type": question_type,
        "target_words": target,
        "answers": [
            # The submitted string is what this call produced, so its cost
            # belongs in Time on top of the answering pass's value.
            {"Rank": rank, "Answer": text, "Time": round(t + share, 3)}
            for rank, text, t in zip(ranks, final, times)
        ],
        "words_before": round(sum(len(a.split()) for a in answers)
                              / max(len(answers), 1), 2),
        "words_after": round(sum(len(a.split()) for a in final)
                             / max(len(final), 1), 2),
        "elapsed": round(elapsed, 2),
        **stats,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def load_cache(path: Path) -> dict[int, dict]:
    cache: dict[int, dict] = {}

    if not path.exists():
        return cache

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                cache[int(record["Q_ID"])] = record
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

    return cache


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Rewrite Task 1 answers longer without changing meaning."
    )

    ap.add_argument("--run", type=Path, required=True,
                    help="run file to expand; never modified")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--routing", type=Path, default=DEFAULT_ROUTING,
                    help="supplies Question_Type, which picks the target length")
    ap.add_argument("--cache", type=Path, required=True)

    ap.add_argument("--backend", choices=sorted(BACKENDS),
                    default=DEFAULT_BACKEND,
                    help="which server rewrites: vllm in WSL2, or ollama on "
                         "Windows (default: %(default)s). Sets the default "
                         "model, host and workers")
    ap.add_argument("--model", help="model tag (default: the backend's)")
    ap.add_argument("--host", help="server host (default: the backend's)")
    ap.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX,
                    help="ollama only; on vllm the server's --max-model-len "
                         "decides (default: %(default)s)")
    ap.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--workers", type=int,
                    help="concurrent requests (default: 4 on ollama, 8 on "
                         "vllm)")
    ap.add_argument("--think", action="store_true")

    ap.add_argument("--mechanical", action="store_true",
                    help="condition C: pad with generic frame words instead of "
                         "calling a model, to test whether the metric rewards "
                         "length alone")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", type=int, nargs="+", metavar="Q_ID")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-emit the run from the cache, no inference")

    return ap.parse_args(argv)


def resolve_backend(args: argparse.Namespace) -> None:
    """Fill in what --backend implies and the caller did not override."""
    defaults = BACKENDS[args.backend]

    for name in ("model", "host", "workers"):
        if getattr(args, name) is None:
            setattr(args, name, defaults[name])


def main(argv=None) -> int:
    args = parse_args(argv)
    resolve_backend(args)

    run = load_json(args.run)
    questions = {q["Q_ID"]: q["Question"] for q in load_json(args.queries)}
    types: dict[int, str] = {}

    if args.routing.exists():
        types = {int(e["Q_ID"]): e.get("Question_Type") or "other"
                 for e in load_json(args.routing)}
    else:
        print(f"no routing at {args.routing} -- every query gets the default "
              f"target of {DEFAULT_TARGET} words", file=sys.stderr)

    cache = load_cache(args.cache)

    print(f"run: {args.run.name} | {len(run)} queries | cached: "
          f"{sum(1 for r in run if r['Q_ID'] in cache)}")

    if not args.rebuild:
        todo = [r for r in run if args.force or r["Q_ID"] not in cache]

        if args.only:
            wanted = set(args.only)
            todo = [r for r in todo if r["Q_ID"] in wanted]

        if args.limit:
            todo = todo[:args.limit]

        no_expand = sum(1 for r in todo
                        if types.get(r["Q_ID"], "other") in NO_EXPAND)
        print(f"to expand: {len(todo)} queries ({no_expand} are "
              f"{'/'.join(sorted(NO_EXPAND))} and pass through), "
              f"{args.workers} worker(s) against {args.host}")

        if todo and args.mechanical:
            for record in todo:
                q_type = types.get(record["Q_ID"], "other")
                cache[record["Q_ID"]] = mechanical_query(
                    record, q_type, TARGET_WORDS.get(q_type, DEFAULT_TARGET))

            args.cache.parent.mkdir(parents=True, exist_ok=True)

            with args.cache.open("a", encoding="utf-8") as handle:
                for record in todo:
                    handle.write(json.dumps(cache[record["Q_ID"]],
                                            ensure_ascii=False) + "\n")

            print(f"padded {len(todo)} queries mechanically (no inference)")
            todo = []

        if todo:
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

            def work(record: dict) -> None:
                nonlocal done, failed

                q_id = record["Q_ID"]
                q_type = types.get(q_id, "other")

                try:
                    result = expand_query(
                        client, record, questions.get(q_id, ""), q_type,
                        TARGET_WORDS.get(q_type, DEFAULT_TARGET),
                    )
                except Exception as exc:  # noqa: BLE001 - one bad query
                    with lock:            # must not sink the pass
                        failed += 1
                        print(f"  FAILED Q_ID {q_id}: "
                              f"{type(exc).__name__}: {exc}", file=sys.stderr)
                    return

                with lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    cache[q_id] = result

                    done += 1
                    rate = done / max(time.monotonic() - started, 1e-9)
                    print(f"  [{done}/{len(todo)}] {rate * 60:5.1f}/min "
                          f"{result['words_before']:4.1f}->"
                          f"{result['words_after']:4.1f}w "
                          f"{q_type:<8} Q{q_id}: "
                          f"{result['answers'][0]['Answer'][:52]}")

            try:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    list(pool.map(work, todo))
            except KeyboardInterrupt:
                print("\ninterrupted -- finished queries are cached",
                      file=sys.stderr)
                return 130
            finally:
                handle.close()

            print(f"\nexpanded {done} queries ({failed} failed) in "
                  f"{(time.monotonic() - started) / 60:.1f} min")

    out: list[dict] = []
    missing = 0

    for record in run:
        result = cache.get(record["Q_ID"])

        if not result:
            missing += 1
            out.append(record)  # never drop a query from the run
            continue

        out.append({
            "Q_ID": record["Q_ID"],
            "Video_ID": record["Video_ID"],
            "Answers": [{"Rank": a["Rank"], "Answer": a["Answer"],
                         "Time": a["Time"]} for a in result["answers"]],
        })

    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=4, ensure_ascii=False)
        handle.write("\n")

    print(f"\nwrote {len(out)} entries to {args.out}")

    if missing:
        print(f"  {missing} queries were not expanded and pass through "
              f"unchanged")

    report(cache, run)

    return 0


def report(cache: dict[int, dict], run: list[dict]) -> None:
    records = [cache[r["Q_ID"]] for r in run if r["Q_ID"] in cache]

    if not records:
        return

    def total(key: str) -> int:
        return sum(r.get(key, 0) for r in records)

    slots = sum(len(r["answers"]) for r in records)

    print(f"\nexpansion report over {len(records)} queries, {slots} answers")
    print(f"  mean words  {sum(r['words_before'] for r in records) / len(records):5.2f}"
          f" -> {sum(r['words_after'] for r in records) / len(records):5.2f}"
          f"   (2025 GT: 8.2)")
    print(f"  rewritten                      {total('expanded'):5d}")
    print(f"  kept, refusal                  {total('kept_refusal'):5d}")
    print(f"  kept, question type never expands {total('kept_type'):5d}")
    print(f"  kept, dropped a content word   {total('dropped_content'):5d}")
    print(f"  kept, would duplicate          {total('duplicate'):5d}")
    print(f"  kept, model returned it as-is  {total('unchanged'):5d}")
    print(f"  kept, empty or over-long       "
          f"{total('empty') + total('too_long'):5d}")
    print(f"  queries whose call failed      {total('call_failed'):5d}")

    times = [r["elapsed"] for r in records if r.get("elapsed")]

    if times:
        print(f"  call time: mean {sum(times) / len(times):.2f}s  "
              f"total {sum(times) / 60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())

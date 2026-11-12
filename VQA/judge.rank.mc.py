#!/usr/bin/env python3
"""Adjudicate the ranker swarm: one judge re-decides where the models disagree.

Eight models have already ranked every Task 2 query independently. This pass is
the orchestrator: for a query where the panel does **not** agree on rank 1, it
shows one judge model the frames again, the question, the official options, what
every panellist ranked where, and a short card of each panellist's known
weaknesses -- then takes the judge's ranking as the answer. Where the panel is
unanimous the consensus is copied through untouched and no call is made.

Why only the disagreements, measured on the 400-query dev sample:

    panel unanimous on rank 1      197 queries, 187 of them already correct
    panel disagrees                203 queries
      Qwen3.8 alone                133 correct
      at least one panellist       187 correct
    ceiling for a perfect judge    374/400 = 0.935 against 0.800

All the headroom is in the 203, and re-judging the unanimous 197 could only
break answers that are already right.

**This is not a vote.** A strict majority over the panel scores 123 of those
203 -- worse than simply trusting Qwen3.8 -- which is the same result the
781,248-fusion search reached by a different route (`search.fusion.mc.dev.py`).
The judge is asked to look at the video again and treat the panel's rankings as
hypotheses, not as ballots.

The ranking contract is the one every pass here shares, imported from
``VQA/rank.options.mc.muse.py`` rather than re-implemented: the judge returns a
permutation of option *numbers*, ``as_permutation()`` repairs a reply that
repeats or drops one, and ``check_permutation()`` refuses to write anything the
MC validator would reject.

Two run files come out of one pass, for free:

    <out>                 the judge's full permutation
    <out>.tail.json       the judge's rank 1, then the baseline's order for the
                          rest -- bounds the MRR damage if the judge orders the
                          tail badly

Three strategies, so every arm of the experiment has the same shape:

    --strategy judge      ask the judge model          (needs a server)
    --strategy majority   most common rank-1, ties to the baseline   (no GPU)
    --strategy null       uniform pick among the candidates          (no GPU)

``null`` is the calibration control: if a coin flip among the panel's answers
beats the baseline, the harness is wrong and no judge result should be believed.

Usage (dev, InternVL judge on llama.cpp port 8013):

    python VQA/judge.rank.mc.py --backend llamacpp \\
        --queries eval.2025.mc/sample400.seed2026.queries.json \\
        --frames videos_GT_2025/frames \\
        --transcripts videos_GT_2025/transcription_openwhisper \\
        --runs eval.2025.mc/qwen38.dev.run.json eval.2025.mc/qwen3vl.dev.run.json ... \\
        --out eval.2025.mc/judge.internvl.dev.run.json \\
        --cache eval.2025.mc/judge.internvl.dev.cache.jsonl

    ... --no-priors          drop the reliability card (the leakage control)
    ... --strategy null      the no-GPU calibration control
    ... --limit 20           smoke test
    ... --rebuild            re-emit the run files from the cache, no inference
"""

from __future__ import annotations

import argparse
import importlib.util
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
MUSE_SCRIPT = ROOT / "VQA" / "rank.options.mc.muse.py"

# Both judges are served the same evidence -- 16 frames at 448x448 as images --
# so the ablation isolates the model rather than the input. It also side-steps
# Qwen3.8's do_sample_frames trap, which only applies to native video.
BACKENDS = {
    "llamacpp": {"model": "internvl3.5-30b-a3b", "host": "http://localhost:8013",
                 "workers": 2},
    "vllm": {"model": "RedHatAI/Qwen3.8-27B-INT4", "host": "http://localhost:8011",
             "workers": 4},
}

DEFAULT_MAX_FRAMES = 16
DEFAULT_FRAME_SIZE = 448
DEFAULT_NUM_PREDICT = 512
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64

# Distilled by hand from MODEL_SCORECARD.2026-09-17.md. Tiers, not dev numbers:
# the card is derived from the same 400 queries the dev arms are scored on, so
# the coarser it is the less it leaks. --no-priors drops it entirely.
RELIABILITY_CARD = """How reliable each panellist is, from earlier measurement:

- qwen38, qwen3vl: strongest. When they agree, they are usually right.
- internvl, muse: middling.
- videollama3, gemma4: weaker. gemma4 sees only 16 sampled frames, so it is
  weakest on anything that depends on a brief moment.
- llavavideo: weakest ranker. Its confidence value is always 0 and means
  nothing.
- cosmos: an outlier. It picks one option and orders the rest mechanically, so
  its lower ranks carry little information.

Two things measured across the whole panel:

- Every model UNDERCOUNTS when asked "how many". If the options are numbers and
  the frames support a higher count, the higher option is more often right.
- A panellist whose own three passes disagreed is unsure, and on those queries
  every other model is unsure too -- so weigh the frames more heavily there
  rather than switching to another panellist."""

SYSTEM_PROMPT = (
    "You adjudicate a panel of video question-answering models. You are shown "
    "the video as frames, a question, its answer options, and how each panel "
    "member ranked those options. You decide the final ranking yourself from "
    "the frames. You reply with JSON only."
)

PROMPT_TEMPLATE = """{context}Question: {question}

Answer options:
{options}

The panel disagreed on this question. How each member ranked the options:

{panel}
{priors}{routing}
Decide the final ranking of all {n} options, best first.

Rules:
- Judge the options against the frames and the transcript yourself. The panel's
  rankings are opinions, not evidence.
- Do not simply count votes. The majority is wrong often enough that a vote
  scores worse than the strongest single member.
- Exactly one option is correct. If the frames are ambiguous, prefer the option
  the strongest members put first.
- Every number from 1 to {n} must appear exactly once.

Reply with JSON: {{"ranking": [{slots}], "confidence": 0.0, "why": "..."}}
where ranking holds all {n} option numbers best first, confidence is 0.0 to 1.0
for the first one, and why is at most 20 words."""


def load_muse():
    """The Muse pass, imported for the helpers every ranking script shares."""
    spec = importlib.util.spec_from_file_location("rank_mc_muse", MUSE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MUSE = load_muse()

normalise = lambda text: " ".join(str(text).split()).lower()


def label_for(path: Path) -> str:
    """'eval.2025.mc/qwen38.dev.run.json' -> 'qwen38'."""
    name = Path(path).name
    for suffix in (".dev.run.json", ".run.task2.json", ".run.json", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace(".vllm", "").replace(".dev", "")


def cache_for(run_path: Path) -> Path | None:
    """The work cache beside a run file, by this project's two conventions."""
    run_path = Path(run_path)
    label = label_for(run_path)
    candidates = [run_path.parent / f"{label}.dev.cache.jsonl",
                  ROOT / f"rankings.task2.{label}.cache.jsonl",
                  ROOT / f"rankings.task2.{label}.vllm.cache.jsonl"]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def load_panel(run_paths: list[Path], queries: list[dict]) -> dict:
    """label -> {q_id: {"ranking": option indices best first, "agreement": x}}.

    Answers are mapped back to official option indices the same way the runs
    were written: normalised exact match, each option consumed once.
    """
    options_of = {q["Q_ID"]: [normalise(o) for o in q["Options"]] for q in queries}
    panel: dict[str, dict] = {}

    for path in run_paths:
        label = label_for(path)
        run = MUSE.load_json(path)
        agreement = {}
        cache_path = cache_for(path)

        if cache_path:
            for record in MUSE.load_cache(cache_path).values():
                agreement[int(record["Q_ID"])] = record.get("agreement")

        entries: dict[int, dict] = {}

        for item in run:
            q_id = int(item["Q_ID"])

            if q_id not in options_of:
                continue

            options = list(options_of[q_id])
            taken = [False] * len(options)
            order = []

            for answer in sorted(item["Answers"], key=lambda a: a["Rank"]):
                text = normalise(answer["Answer"])
                index = next((i for i, o in enumerate(options)
                              if o == text and not taken[i]), None)

                if index is None:
                    continue

                taken[index] = True
                order.append(index)

            if len(order) == len(options):
                entries[q_id] = {"ranking": order,
                                 "agreement": agreement.get(q_id)}

        panel[label] = entries

    return panel


def panel_table(panel: dict, labels: list[str], q_id: int, n: int) -> str:
    """One line per option: who put it first, and every member's rank for it."""
    lines = []

    for index in range(n):
        firsts = [m for m in labels
                  if panel[m].get(q_id) and panel[m][q_id]["ranking"][0] == index]
        ranks = []

        for m in labels:
            entry = panel[m].get(q_id)

            if entry:
                ranks.append(f"{m} {entry['ranking'].index(index) + 1}")

        placed = ", ".join(ranks)
        first_note = f"ranked FIRST by {', '.join(firsts)}. " if firsts else ""
        lines.append(f"Option {index + 1}: {first_note}ranks -- {placed}")

    unsure = [m for m in labels
              if panel[m].get(q_id)
              and isinstance(panel[m][q_id].get("agreement"), (int, float))
              and panel[m][q_id]["agreement"] < 0.999]

    if unsure:
        lines.append(f"Members whose own passes disagreed here: "
                     f"{', '.join(unsure)}.")

    return "\n".join(lines)


def candidates_for(panel: dict, labels: list[str], q_id: int) -> list[int]:
    """The distinct rank-1 option indices across the panel."""
    seen = []

    for m in labels:
        entry = panel[m].get(q_id)

        if entry and entry["ranking"][0] not in seen:
            seen.append(entry["ranking"][0])

    return seen


def routing_line(routing: dict, q_id: int, question: str) -> str:
    """What the router decided for this query, when a verdict exists."""
    entry = routing.get(q_id)

    if not entry or normalise(entry.get("Question", "")) != normalise(question):
        return ""

    pipeline = set(entry.get("Pipeline") or ())
    parts = []

    if "frames" in pipeline:
        parts.append("watch the video")
    if "asr_transcript" in pipeline:
        parts.append("use the spoken audio")
    if "frame_ocr" in pipeline:
        parts.append("read text on screen")
    if entry.get("Temporal"):
        parts.append("the order of events matters")
    if entry.get("Inference"):
        parts.append("the answer needs inference, not just observation")

    if not parts:
        return ""

    return f"\nThe router says, for this question: {'; '.join(parts)}.\n"


class JudgeClient:
    """OpenAI-compatible client, one shape for llama.cpp and for vLLM.

    Frames go in as image parts ahead of the text; the schema goes in
    ``response_format``. A 400 is the server refusing the request itself and is
    not retried into the same wall.
    """

    def __init__(self, host: str, model: str, num_predict: int,
                 temperature: float, top_p: float, top_k: int, timeout: float,
                 retries: int, cache_prompt: bool = False) -> None:
        self.url = host.rstrip("/") + "/v1/chat/completions"
        # llama.cpp's prompt-cache switch. vLLM rejects request fields it does
        # not know, so it is only sent to the backend that has it.
        self.cache_prompt = cache_prompt
        self.model = model
        self.num_predict = num_predict
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.timeout = timeout
        self.retries = retries
        self.local = threading.local()

    @property
    def session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def ask(self, prompt: str, schema: dict, seed: int,
            images: list[str]) -> tuple[dict, float, dict]:
        content: list[dict] = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
            for image in images
        ]
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
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "judgement",
                                "schema": dict(schema,
                                               additionalProperties=False)},
            },
        }

        if self.cache_prompt:
            payload["cache_prompt"] = True

        last: Exception | None = None

        for attempt in range(self.retries):
            started = time.monotonic()

            try:
                response = self.session.post(self.url, json=payload,
                                             timeout=self.timeout)

                if response.status_code == 400:
                    raise RuntimeError(f"400: {response.text[:300]}")

                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                text = (choice["message"].get("content") or "").strip()
                usage = body.get("usage") or {}

                return parse_reply(text), time.monotonic() - started, {
                    "finish_reason": choice.get("finish_reason"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "attempts": attempt + 1,
                }
            except (requests.RequestException, ValueError, KeyError,
                    IndexError) as exc:
                last = exc

                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"{type(last).__name__}: {last}")


def parse_reply(text: str) -> dict:
    reply = MUSE.parse_json_object(text)

    if not isinstance(reply.get("ranking"), list):
        raise ValueError(f"no ranking in reply: {text[:160]!r}")

    return reply


def judgement_schema(n: int) -> dict:
    schema = MUSE.ranking_schema(n)
    schema["properties"]["why"] = {"type": "string"}
    return schema


def preflight(backend: str, host: str, model: str, workers: int) -> None:
    """Refuse to start against the wrong server, per backend.

    llama.cpp exposes /props (and reports whether a vision projector is
    loaded); vLLM does not -- it only has /v1/models -- so asking it for /props
    is a 404, not a dead server.
    """
    if backend == "llamacpp":
        MUSE.preflight(host, model, workers)
        return

    try:
        response = requests.get(host.rstrip("/") + "/v1/models", timeout=30)
        response.raise_for_status()
        served = {entry["id"]: entry for entry in response.json()["data"]}
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise SystemExit(f"no vLLM server answering at {host} "
                         f"({type(exc).__name__}: {exc})")

    print(f"serving: {sorted(served)} | max_model_len "
          f"{(served.get(model) or {}).get('max_model_len')} | "
          f"{workers} worker(s)")

    if model not in served:
        raise SystemExit(f"{host} serves {sorted(served)}, not {model!r}")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Adjudicate the ranker swarm where its members disagree.")

    ap.add_argument("--runs", type=Path, nargs="+", required=True,
                    help="the panel's run files; the first is the baseline "
                         "unless --baseline says otherwise")
    ap.add_argument("--baseline", type=Path,
                    help="the run used for unanimous queries, for the tail "
                         "variant and for every fallback (default: --runs[0])")
    ap.add_argument("--queries", type=Path, default=MUSE.DEFAULT_QUERIES)
    ap.add_argument("--frames", type=Path, default=MUSE.DEFAULT_FRAMES)
    ap.add_argument("--transcripts", type=Path, default=MUSE.DEFAULT_TRANSCRIPTS)
    ap.add_argument("--routing", type=Path, default=MUSE.DEFAULT_ROUTING,
                    help="router verdicts; only used where the question matches")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)

    ap.add_argument("--strategy", choices=("judge", "majority", "null"),
                    default="judge",
                    help="judge: ask the model. majority/null: no GPU, the two "
                         "controls (default: %(default)s)")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="llamacpp")
    ap.add_argument("--model")
    ap.add_argument("--host")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--no-priors", dest="priors", action="store_false",
                    help="omit the reliability card -- the leakage control")

    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--frame-size", type=int, default=DEFAULT_FRAME_SIZE,
                    help="resize each frame to N x N (448 = one InternVL tile)")
    ap.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)

    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", type=int, nargs="+", metavar="Q_ID")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-emit the run files from the cache, no inference")

    return ap.parse_args(argv)


def resolve(args: argparse.Namespace) -> None:
    defaults = BACKENDS[args.backend]

    for name in ("model", "host", "workers"):
        if getattr(args, name) is None:
            setattr(args, name, defaults[name])

    if args.baseline is None:
        args.baseline = args.runs[0]


def emit(queries: list[dict], panel: dict, baseline_label: str,
         cache: dict, out: Path) -> tuple[int, int]:
    """Write both run files. Returns (entries, queries missing a decision)."""
    run, tail_run, missing = [], [], 0

    for query in queries:
        q_id = query["Q_ID"]
        options = query["Options"]
        base = panel[baseline_label].get(q_id)

        if not base:
            missing += 1
            continue

        record = cache.get(q_id)
        order = record["ranking"] if record else base["ranking"]

        # The tail variant: the judge's winner, then the baseline's order.
        winner = order[0]
        tail = [i for i in base["ranking"] if i != winner]

        for target, indices in ((run, order), (tail_run, [winner] + tail)):
            answers = [{"Rank": rank, "Answer": options[index]}
                       for rank, index in enumerate(indices, 1)]
            MUSE.check_permutation([a["Answer"] for a in answers], options, q_id)
            target.append({"Q_ID": q_id, "Video_ID": query["Video_ID"],
                           "Answers": answers})

    for path, payload in ((out, run),
                          (out.with_suffix(".tail.json"), tail_run)):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, ensure_ascii=False)
            handle.write("\n")

    return len(run), missing


def report(cache: dict, queries: list[dict], baseline_label: str,
           panel: dict) -> None:
    """The checks that matter: did the judge move anything, and how."""
    decided = [r for r in cache.values() if r.get("ranking")]

    if not decided:
        return

    changed = sum(1 for r in decided if r.get("changed_rank1"))
    fell_back = sum(1 for r in decided if r.get("fell_back"))
    repaired = sum(1 for r in decided if r.get("repaired"))
    failed = sum(1 for r in decided if r.get("error"))
    agreed = len(decided) - changed

    print(f"\njudged {len(decided)} disagreement queries")
    print(f"  kept the baseline's winner:         {agreed} "
          f"({100 * agreed / len(decided):.1f}%)")
    print(f"  overruled it:                       {changed} "
          f"({100 * changed / len(decided):.1f}%)")
    print(f"  fell back to the baseline:          {fell_back}"
          f"{'  -- CHECK' if fell_back else ''}")
    print(f"  repaired rankings:                  {repaired}")
    print(f"  failed calls:                       {failed}"
          f"{'  -- CHECK' if failed else ''}")

    picks = {}

    for record in decided:
        winner = record["ranking"][0]
        source = [m for m in panel
                  if panel[m].get(record["Q_ID"])
                  and panel[m][record["Q_ID"]]["ranking"][0] == winner]
        key = ", ".join(sorted(source)) or "none of the panel"
        picks[key] = picks.get(key, 0) + 1

    print("  whose answer the judge landed on (top 6):")

    for key, count in sorted(picks.items(), key=lambda kv: -kv[1])[:6]:
        print(f"    {count:4d}  {key}")

    costs = [r for r in decided if r.get("prompt_tokens")]

    if costs:
        mean = lambda k: sum(r[k] or 0 for r in costs) / len(costs)
        print(f"  per call: {mean('prompt_tokens'):6.0f} prompt tokens, "
              f"{mean('completion_tokens'):4.0f} completion, "
              f"{mean('elapsed'):5.2f}s")

    confidences = [r["confidence"] for r in decided
                   if isinstance(r.get("confidence"), (int, float))]

    if confidences:
        print(f"  mean judge confidence:              "
              f"{sum(confidences) / len(confidences):.3f}")


def main(argv=None) -> int:
    args = parse_args(argv)
    resolve(args)

    queries = MUSE.load_queries(args.queries)
    labels = [label_for(p) for p in args.runs]
    baseline_label = label_for(args.baseline)

    if baseline_label not in labels:
        raise SystemExit(f"--baseline {baseline_label} is not among --runs "
                         f"({', '.join(labels)})")

    panel = load_panel(args.runs, queries)
    routing = MUSE.load_routing(args.routing) if args.routing.exists() else {}
    frame_index_path = args.frames / "index.json"
    frame_index = (MUSE.load_json(frame_index_path)["videos"]
                   if frame_index_path.exists() else {})
    cache = {int(k): v for k, v in MUSE.load_cache(args.cache).items()}

    covered = [q for q in queries if all(panel[m].get(q["Q_ID"]) for m in labels)]
    disagreeing = [q for q in covered
                   if len(candidates_for(panel, labels, q["Q_ID"])) > 1]

    print(f"panel: {', '.join(labels)} | baseline: {baseline_label}")
    print(f"strategy: {args.strategy}"
          + (f" | {args.model} at {args.host} | {args.workers} worker(s)"
             if args.strategy == "judge" else "")
          + f" | priors {'on' if args.priors else 'off'}")
    print(f"queries: {len(queries)} | ranked by every panellist: {len(covered)} "
          f"| the panel disagrees on: {len(disagreeing)} "
          f"({100 * len(disagreeing) / max(len(covered), 1):.1f}%)")

    if not args.rebuild:
        todo = [q for q in disagreeing
                if args.force or q["Q_ID"] not in cache]

        if args.only:
            wanted = set(args.only)
            todo = [q for q in todo if q["Q_ID"] in wanted]

        if args.limit:
            todo = todo[:args.limit]

        print(f"to judge: {len(todo)}")

        if todo and args.strategy == "judge":
            preflight(args.backend, args.host, args.model, args.workers)

        client = (JudgeClient(args.host, args.model, args.num_predict,
                              args.temperature, args.top_p, args.top_k,
                              args.timeout, args.retries,
                              cache_prompt=args.backend == "llamacpp")
                  if args.strategy == "judge" and todo else None)

        lock = threading.Lock()
        done = 0
        # A dead server otherwise costs a full pass of retries before the run
        # ends looking finished but identical to the baseline.
        global_failures = [0]
        started = time.monotonic()
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        handle = args.cache.open("a", encoding="utf-8")

        def work(query: dict) -> None:
            nonlocal done
            q_id = query["Q_ID"]
            options = query["Options"]
            n = len(options)
            base = panel[baseline_label][q_id]["ranking"]
            candidates = candidates_for(panel, labels, q_id)
            record = {"Q_ID": q_id, "Video_ID": query["Video_ID"],
                      "n_options": n, "n_candidates": len(candidates),
                      "candidates": candidates, "strategy": args.strategy,
                      "priors": args.priors, "baseline_rank1": base[0],
                      "ts": datetime.now(timezone.utc).isoformat(
                          timespec="seconds")}
            rng = random.Random(MUSE.seed_for(q_id, args.seed))

            if args.strategy == "null":
                winner = rng.choice(candidates)
                record["ranking"] = [winner] + [i for i in base if i != winner]
            elif args.strategy == "majority":
                counts = {}

                for m in labels:
                    first = panel[m][q_id]["ranking"][0]
                    counts[first] = counts.get(first, 0) + 1

                best = max(counts.values())
                winners = [i for i, c in counts.items() if c == best]
                # Ties go to the baseline's opinion, which is the strongest
                # single member -- otherwise the control is weaker than it
                # should be and flatters the judge.
                winner = base[0] if base[0] in winners else winners[0]
                record["ranking"] = [winner] + [i for i in base if i != winner]
            else:
                stem = Path(query["Video_ID"]).stem
                entry = frame_index.get(stem)
                images: list[str] = []

                if entry:
                    files = entry["files"][:args.max_frames]
                    try:
                        images = [MUSE.encode_frame(args.frames / stem / name,
                                                    args.frame_size)
                                  for name in files]
                    except (OSError, KeyError, TypeError) as exc:
                        print(f"  Q{q_id}: unreadable frames ({exc})",
                              file=sys.stderr)
                        images = []

                shown = dict(entry, files=entry["files"][:args.max_frames],
                             n_frames=len(images),
                             timestamps=entry["timestamps"][:args.max_frames]
                             ) if entry and images else None
                transcript = MUSE.load_transcript(args.transcripts,
                                                  query["Video_ID"])
                context = MUSE.build_context(shown, transcript)
                prompt = PROMPT_TEMPLATE.format(
                    context=context,
                    question=query["Question"],
                    options=MUSE.render_options(options, list(range(n))),
                    panel=panel_table(panel, labels, q_id, n),
                    priors=f"\n{RELIABILITY_CARD}\n" if args.priors else "",
                    routing=routing_line(routing, q_id, query["Question"]),
                    n=n,
                    slots=", ".join(str(i) for i in range(1, n + 1)),
                )

                try:
                    reply, elapsed, cost = client.ask(
                        prompt, judgement_schema(n),
                        MUSE.seed_for(q_id, args.seed), images)
                    positions, repaired = MUSE.as_permutation(
                        reply.get("ranking"), n)
                    record.update(
                        ranking=positions, repaired=repaired,
                        confidence=reply.get("confidence"),
                        why=str(reply.get("why") or "")[:160],
                        elapsed=round(elapsed, 3), n_frames=len(images),
                        has_transcript="Spoken audio" in context, **cost)
                except (RuntimeError, ValueError) as exc:
                    record.update(ranking=list(base), fell_back=True,
                                  error=f"{type(exc).__name__}: {exc}"[:200])
                    print(f"  Q{q_id}: fell back ({exc})", file=sys.stderr)

            record["changed_rank1"] = record["ranking"][0] != base[0]

            with lock:
                nonlocal_failures = record.get("fell_back")
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                cache[q_id] = record
                done += 1

                global_failures[0] = global_failures[0] + 1 if nonlocal_failures else 0

                if global_failures[0] >= 10:
                    raise SystemExit(
                        "10 calls in a row fell back -- the judge server is not "
                        "answering. Fix it and re-run; judged queries are cached.")

                rate = done / max(time.monotonic() - started, 1e-9)
                flag = " CHANGED" if record["changed_rank1"] else ""
                flag += " FALLBACK" if record.get("fell_back") else ""
                print(f"  [{done}/{len(todo)}] {rate * 60:5.1f}/min "
                      f"eta {(len(todo) - done) / max(rate, 1e-9) / 60:5.1f}m "
                      f"Q{q_id}{flag}: {record.get('why', '')[:60]}")

        try:
            if todo:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    list(pool.map(work, todo))
        except KeyboardInterrupt:
            print("\ninterrupted -- judged queries are cached, re-run to resume",
                  file=sys.stderr)
            return 130
        finally:
            handle.close()

        if todo:
            print(f"\njudged {done} queries in "
                  f"{(time.monotonic() - started) / 60:.1f} min")

    written, missing = emit(queries, panel, baseline_label, cache, args.out)
    print(f"\nwrote {written} entries to {args.out}")
    print(f"      {written} entries to {args.out.with_suffix('.tail.json')}")

    if missing:
        print(f"  {missing} queries are missing from the baseline run")

    report(cache, queries, baseline_label, panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

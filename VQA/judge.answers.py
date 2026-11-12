#!/usr/bin/env python3
"""Score every fused candidate for relevance to the video, with Gemma 4.

``fuse.answers.py`` pools both models into a mean of **25.6 candidates per
query** (62,601 across the 2026 set) and keeps ten. The other ~38,000 are
discarded by a weighted sum of ``agreement``, ``score``, ``logprob`` and
``confidence`` -- and every one of those four is computed *from the same
forward passes that produced the candidates*. They measure how hard the
generator's prior locked in, not whether it was right, which is why
``ranking.analysis.2025.md`` puts them at Spearman 0.032 (``score``) and
**-0.073** (``votes``) against per-answer METEOR. If the model was confidently
wrong, all four signals are confidently wrong in the same direction.

This pass adds the one signal that is not a restatement of the generator: a
second look at the frames, with the candidate supplied as something to
*evaluate* rather than something to produce. That conditioning is what lets it
rank a candidate low that the generator ranked first.

**After ITI-CERTH**, who reranked captions by prompting a VLM for a single
1-10 relevance number. Three deviations, all forced by the arithmetic here:

1. **Listwise in one call.** Pointwise per candidate is 62,601 calls; at 4 s
   each that is 70 GPU-hours. One call per query with the frames attached once
   and every candidate numbered in the prompt is 2,443 calls, and the frame
   prefill is paid once instead of 25 times.
2. **No majority vote.** Their polling step *is* ``votes``, the one signal on
   this dataset already measured negative. Frequency across paraphrases is not
   evidence.
3. **Blind.** Candidates are shuffled on a per-query seed and carry no model
   attribution, so the judge cannot anchor on the fuser's existing order or
   prefer the phrasings of the model it happens to be.

HLTCOE's reranker was considered and rejected: Masked Pointer Cross-Entropy
with rank weights needs ranked labels and a training loop, and their portable
contribution -- prompt variation for diversity instead of finetuning -- is
already what ``query.variations.task1.json`` does here.

**Ordering is not the point.** ``eval.vqa.ag.py`` averages over all ten
answers, so re-ordering cannot move METEOR or BERTScore -- measured identical
at 0.148 under current, random, reverse *and* oracle ordering. This pays off
only through *selection*, on the 88% of queries holding more than ten
candidates. Prove it on 150 queries with ``VQA/gate.judge.2025.py`` before
setting the weight; the gate's ``random-10`` arm is what separates "the judge
is good" from "the fuser is bad".

Writes ``judge.task1.cache.jsonl``; ``fuse.answers.py --judge`` consumes it.
Nothing here touches the validated runs already on disk.

Usage:
    python VQA/judge.answers.py --limit 50          # measure the rate first
    python VQA/judge.answers.py
"""

from __future__ import annotations

import argparse
import base64
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
HERE = Path(__file__).resolve().parent


def load_module(filename: str, name: str):
    """Import a sibling script despite the dots in its name.

    Same mechanism as ``rerank.answers.py:356`` and ``run.ensemble.py:155``.
    Neither module runs anything at import time.
    """
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


GEN = load_module("generate.answers.py", "generate_answers")
FUSE = load_module("fuse.answers.py", "fuse_answers")

DEFAULT_QUERIES = ROOT / "vqa.2026.testingDataset.Task1.json"
DEFAULT_ROUTING = ROOT / "query.routing.task1.json"
DEFAULT_FRAMES = ROOT / "videos_TEST_VQA" / "frames"
DEFAULT_TRANSCRIPTS = ROOT / "videos_TEST_VQA" / "transcription_openwhisper"
DEFAULT_GEMMA4_CACHE = ROOT / "answers.task1.gemma4.ens.cache.jsonl"
DEFAULT_QWEN3VL_CACHE = ROOT / "answers.task1.qwen3vl.ens.cache.jsonl"
DEFAULT_CACHE = ROOT / "judge.task1.cache.jsonl"

DEFAULT_HOST = "http://localhost:8010"
DEFAULT_MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"

# Same knee as the answering pass -- 8 workers measured 123.8 s against 161.7 s
# at 4 and 279.9 s at 12 over the same 20 queries (generate.answers.py:110).
DEFAULT_WORKERS = 8

# Measured over the real 2026 pool: mean 24.8 non-refusal candidates, median
# 24, p90 45, max 59. Where the cap binds, the cut is made by the weak
# pre-signal this pass exists to replace, so it wants to be generous: 48 binds
# on 5.4% of queries and still judges 59,931 of the 60,475 candidates (99.1%).
# 32 was the first guess and bound on 30.3%. The cache records `capped` either
# way, and Verification 6 counts it.
DEFAULT_MAX_CANDIDATES = 48

# 48 integers with separators is ~155 tokens. 384 leaves room for the grammar's
# structural tokens without inviting a truncation that would fail the schema
# and burn all three retries.
DEFAULT_NUM_PREDICT = 384

# Judging is a measurement, not a generation: the same frames and the same
# candidates should give the same numbers twice. The ensemble's diversity was
# bought upstream by the ten phrasings and is already in the pool.
DEFAULT_TEMPERATURE = 0.0

JUDGE_SYSTEM_PROMPT = (
    "You grade candidate answers to a question about a video you are shown as "
    "a sequence of frames. You reply with JSON only."
)

JUDGE_TEMPLATE = """{context}Question: {question}

Below are {n} candidate answers to that question, in no particular order.
Rate how well each one answers the question about this video.

Scale:
- 10: exactly right
- 7-9: right, or right but loosely worded
- 4-6: plausible, but nothing in the frames or the transcript supports it
- 1-3: wrong, irrelevant, or a refusal to answer

Rules:
- Judge against the frames and the transcript only. Do not reward fluent or
  detailed wording; a blunt correct answer beats a polished wrong one.
- Rate each candidate on its own. Ties are expected -- do not spread the
  scores out to separate answers that are equally good.
- If the video does not answer the question at all, score every candidate low
  rather than picking a favourite.

Candidates:
{candidates}

Reply with JSON: {{"scores": [{slots}]}}
with exactly {n} integers from 1 to 10, in the same order as the candidates."""


def judge_schema(n: int) -> dict:
    """Constrain the reply to exactly `n` integers in 1..10.

    Mirrors ``answer_schema()`` (generate.answers.py:238). vLLM compiles this
    into a grammar, so a short count or an out-of-range score is unreachable
    rather than something to validate afterwards.
    """
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": 10},
                "minItems": n,
                "maxItems": n,
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    }


class JudgeClient(GEN.VLLM):
    """The answering pass's vLLM client, asked a different question.

    Inherits the session handling, the retry ladder and the 400-is-terminal
    rule from ``VLLM`` (generate.answers.py:565). ``ask()`` is left alone --
    it is hard-wired to the answers schema, and ``generate.answers.py`` is not
    being edited after it has produced a validated submission.
    """

    def score(self, prompt: str, n: int, seed: int,
              images: list[str] | None) -> tuple[list[int], float]:
        content: list[dict] = []

        if images:
            content += [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                for image in images
            ]

        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "max_tokens": self.num_predict,
            "temperature": self.temperature,
            "seed": seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "scores", "schema": judge_schema(n)},
            },
            "chat_template_kwargs": {"enable_thinking": self.think},
        }

        # top_p / top_k are omitted deliberately: at temperature 0 they do
        # nothing, and passing them invites a build that validates them
        # against the sampler to 400 on a request that would have worked.
        last: Exception | None = None

        for attempt in range(self.retries):
            started = time.monotonic()

            try:
                response = self.session.post(self.url, json=payload,
                                             timeout=self.timeout)

                if response.status_code == 400:
                    raise RuntimeError(f"400: {response.text[:300]}")

                response.raise_for_status()

                choice = response.json()["choices"][0]
                content_out = (choice["message"].get("content") or "").strip()

                if not content_out:
                    reason = choice.get("finish_reason")
                    raise ValueError(f"empty content (finish_reason={reason})")

                scores = GEN.parse_json_object(content_out).get("scores")

                if not isinstance(scores, list) or len(scores) != n:
                    raise ValueError(
                        f"expected {n} scores, got "
                        f"{len(scores) if isinstance(scores, list) else scores!r}")

                return [int(s) for s in scores], time.monotonic() - started

            except (requests.RequestException, ValueError, KeyError, IndexError,
                    json.JSONDecodeError) as exc:
                last = exc

                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))

        raise RuntimeError(f"{type(last).__name__}: {last}")


def pool_for(gemma: dict | None, qwen: dict | None,
             weights: dict[str, float], cap: int) -> tuple[list[dict], bool]:
    """The fused candidate clusters for one query, best-first, capped.

    Built with ``fuse.answers.py``'s own ``collect`` / ``merge`` / ``rank`` so
    the judge scores exactly the clusters the fuser will later re-rank -- if
    these drifted apart, the normalised-key join in ``fuse.answers.py`` would
    silently drop scores (Verification 5).

    Refusals are excluded. They sort last in ``rank()`` whatever they score, so
    judging them buys nothing and costs output tokens on 78% of queries.
    """
    pools = {"gemma4": FUSE.collect(gemma, "gemma4"),
             "qwen3vl": FUSE.collect(qwen, "qwen3vl")}
    ranked = FUSE.rank(FUSE.merge(pools), weights)
    real = [c for c in ranked if not c["refusal"]]

    return real[:cap], len(real) > cap


def judge_query(client: JudgeClient, query: dict, clusters: list[dict],
                context: str, images: list[str] | None,
                extra: str = "") -> tuple[dict[str, int], float, int]:
    """One call. Returns {normalised answer: score}, wall time, n judged."""
    q_id = query["Q_ID"]

    # Blind: a per-query permutation, stable across re-runs through the same
    # seed helper the answering pass uses. Without it the candidates arrive in
    # the fuser's order and the judge is being asked to confirm a ranking it
    # can see, which is the anchoring this pass exists to avoid.
    order = list(range(len(clusters)))
    random.Random(GEN.seed_for(q_id, 0)).shuffle(order)
    shown = [clusters[i] for i in order]

    listing = "\n".join(f"{i}. {c['answer']}"
                        for i, c in enumerate(shown, 1))
    slots = ", ".join(["0"] * len(shown))

    scores, elapsed = client.score(
        JUDGE_TEMPLATE.format(context=context, question=query["Question"],
                              n=len(shown), candidates=listing,
                              slots=slots) + extra,
        n=len(shown),
        seed=GEN.seed_for(q_id, 1),
        images=images,
    )

    # Keyed by the validator's own normalisation, which is what fuse.answers.py
    # clusters on -- so the join back is exact, not a text match.
    return ({FUSE.normalize(c["answer"]): score
             for c, score in zip(shown, scores)},
            elapsed, len(shown))


def load_cache(path: Path) -> dict[int, dict]:
    """Q_ID -> judged record. Later records win, so a resume overwrites."""
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
                print(f"  {path.name}: skipping malformed line {line_no}",
                      file=sys.stderr)

    return cache


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Score every fused candidate 1-10 for relevance to the "
                    "video, so fuse.answers.py can select on evidence rather "
                    "than on the generator's own confidence.")

    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--gemma4", type=Path, default=DEFAULT_GEMMA4_CACHE)
    ap.add_argument("--qwen3vl", type=Path, default=DEFAULT_QWEN3VL_CACHE)
    ap.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    ap.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    ap.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)

    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--max-candidates", type=int,
                    default=DEFAULT_MAX_CANDIDATES,
                    help="judge at most this many per query (default: "
                         "%(default)s)")
    ap.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--think", action="store_true",
                    help="leave Gemma 4's thinking on (off by default, as in "
                         "the answering pass)")
    ap.add_argument("--weights",
                    help="pre-ranking weights used only to order the pool "
                         "before the cap bites; not the final ranking")
    ap.add_argument("--ignore-routing", action="store_true")
    ap.add_argument("--limit", type=int,
                    help="judge only the first N queries -- use it to measure "
                         "the rate before committing to the full set")
    ap.add_argument("--only", type=int, nargs="+", metavar="Q_ID")
    ap.add_argument("--force", action="store_true",
                    help="re-judge queries already in the cache")

    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    weights = FUSE.parse_weights(args.weights)
    queries = GEN.load_json(args.queries)
    gemma = FUSE.load_cache(args.gemma4)
    qwen = FUSE.load_cache(args.qwen3vl)
    cache = load_cache(args.cache)

    if not gemma and not qwen:
        print(f"neither {args.gemma4.name} nor {args.qwen3vl.name} holds "
              f"candidates -- run the answering passes first", file=sys.stderr)
        return 1

    routing = {} if args.ignore_routing else GEN.load_routing(args.routing)

    index_path = args.frames / "index.json"
    frame_index = GEN.load_json(index_path)["videos"] \
        if index_path.exists() else {}

    if not frame_index:
        print(f"no frame index at {index_path} -- run VQA/extract.frames.py "
              f"first", file=sys.stderr)
        return 1

    print(f"judge: {args.model} at {args.host} | {args.workers} worker(s)")
    print(f"  gemma4  {args.gemma4.name}: {len(gemma)} cached")
    print(f"  qwen3vl {args.qwen3vl.name}: {len(qwen)} cached")
    print(f"  cache:  {args.cache.name}: {len(cache)} already judged")

    todo = [q for q in queries if args.force or q["Q_ID"] not in cache]

    if args.only:
        wanted = set(args.only)
        todo = [q for q in todo if q["Q_ID"] in wanted]

    if args.limit:
        todo = todo[:args.limit]

    if not todo:
        print("nothing to judge")
        return 0

    # Group by video, for the same reason the answering pass does: the frames
    # are the expensive half of the prompt, and keeping a video on one worker
    # is what turns its queries into prefix-cache hits.
    groups: dict[str, list[dict]] = {}

    for query in todo:
        groups.setdefault(query["Video_ID"], []).append(query)

    ordered = [groups[video_id] for video_id in sorted(groups)]

    print(f"to judge: {len(todo)} queries over {len(ordered)} videos")

    GEN.preflight(args.host, args.model)
    client = JudgeClient(args.host, args.model, args.num_predict,
                         args.temperature, args.timeout, args.think,
                         args.retries)

    lock = threading.Lock()
    done = failed = skipped = capped_n = 0
    started = time.monotonic()

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    handle = args.cache.open("a", encoding="utf-8")

    def work(group: list[dict]) -> None:
        nonlocal done, failed, skipped, capped_n

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

        transcript = GEN.load_transcript(args.transcripts, video_id)
        has_speech = bool(transcript and transcript.get("text"))

        for query in group:
            q_id = query["Q_ID"]
            clusters, was_capped = pool_for(gemma.get(q_id), qwen.get(q_id),
                                            weights, args.max_candidates)

            # Nothing to choose between: the fuser's ten are forced, so a
            # judgement cannot change the selection. Ordering is worth only
            # NDCG here, and not worth a call.
            if len(clusters) < 2:
                with lock:
                    skipped += 1
                continue

            plan = GEN.plan_for(routing.get(q_id), args.ignore_routing)

            # Same rescue as both answering passes: a query routed to the
            # audio of a clip with no speech gets its frames back, so the
            # judge is never asked to grade on no evidence at all.
            if not plan["frames"] and not has_speech:
                plan = dict(plan, frames=True, rescued=True)

            send_images = images if plan["frames"] else None
            context = GEN.build_context(
                frames if plan["frames"] else None,
                transcript if plan["transcript"] else None,
                routed_out=not plan["frames"],
            )

            extra = ("\n\nThe answer is text rendered on screen; judge it "
                     "against the exact wording shown." if plan["ocr"] else "")

            try:
                scores, elapsed, n_judged = judge_query(
                    client, query, clusters, context, send_images, extra)
            except Exception as exc:  # noqa: BLE001 - one bad query must not
                with lock:            # sink the pass; an absent score is 0
                    failed += 1       # in rank(), i.e. today's behaviour
                    print(f"  FAILED Q_ID {q_id}: {type(exc).__name__}: {exc}",
                          file=sys.stderr)
                continue

            record = {
                "Q_ID": q_id,
                "Video_ID": video_id,
                "scores": scores,
                "n_judged": n_judged,
                "capped": was_capped,
                "n_frames": len(send_images) if send_images else 0,
                "has_transcript": bool(plan["transcript"] and has_speech),
                "routed": plan["routed"],
                "top_score": max(scores.values()) if scores else 0,
                "elapsed": round(elapsed, 2),
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

            with lock:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                cache[q_id] = record

                done += 1
                capped_n += int(was_capped)
                rate = done / max(time.monotonic() - started, 1e-9)
                remaining = (len(todo) - done) / max(rate, 1e-9) / 3600

                print(f"  [{done}/{len(todo)}] {rate * 60:5.1f}/min "
                      f"eta {remaining:4.1f}h  {elapsed:5.1f}s "
                      f"{n_judged:2d}c top{record['top_score']:2d}"
                      f"{'  CAP' if was_capped else ''}  Q{q_id}")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, ordered))
    except KeyboardInterrupt:
        print("\ninterrupted -- judged queries are cached, re-run to resume",
              file=sys.stderr)
        return 130
    finally:
        handle.close()

    elapsed = time.monotonic() - started
    print(f"\njudged {done} queries ({failed} failed, {skipped} had nothing "
          f"to choose between) in {elapsed / 3600:.2f} h")

    report(cache, capped_n)

    return 0


def report(cache: dict[int, dict], capped_n: int) -> None:
    """What the judge actually said, since none of it is visible in a run file.

    A judge that returns the same number for everything is worse than useless
    -- it would pass every correctness check and contribute nothing but noise
    to the fused score. The spread is the thing to read here.
    """
    records = list(cache.values())

    if not records:
        return

    every: list[int] = []
    spreads: list[int] = []
    flat = 0

    for record in records:
        values = list((record.get("scores") or {}).values())

        if not values:
            continue

        every += values
        spreads.append(max(values) - min(values))
        flat += int(max(values) == min(values))

    if not every:
        return

    buckets = {f"{lo}-{lo + 1}": sum(1 for v in every if lo <= v <= lo + 1)
               for lo in (1, 3, 5, 7, 9)}

    print(f"\njudgement over {len(records)} queries, {len(every)} candidates")
    print(f"  mean score {sum(every) / len(every):.2f}  "
          f"spread within a query: mean {sum(spreads) / len(spreads):.1f}, "
          f"all-equal on {flat} queries"
          f"{'  -- CHECK: a flat judge adds noise, not signal' if flat > len(records) * 0.5 else ''}")

    for name, count in buckets.items():
        print(f"    {name:>4}  {count:6d}  {100 * count / len(every):5.1f}%")

    print(f"  queries where the 32-candidate cap bound: {capped_n}")


if __name__ == "__main__":
    sys.exit(main())

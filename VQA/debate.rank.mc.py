#!/usr/bin/env python3
"""The orchestrator of the two-judge debate: sends Q_IDs, relays, decides.

The single-judge pass (`VQA/judge.rank.mc.py`, measured 17 Sep) landed on the
baseline: each judge's overrules were a coin flip. Reading the two judge runs
*against each other* is what showed where the value actually is:

    on the 203 dev queries the panel disputed
      the two judges agree with each other   153   correct 112  (baseline 109)
      the two judges disagree                 50   qwen38-judge 20, internvl 18,
                                                   BASELINE 24, either-right 38
      oracle over the two judges (all 203)         150  vs baseline 133

So: debate where the two judges split, and when a debate fails to converge fall
back to the **baseline**, because on exactly that contested set the baseline
beats both judges individually. That asymmetry is the whole design.

Three processes take part:

    judge a   VQA/agent.debate.mc.py on the Mac, beside LM Studio (Qwen3.8)
    judge b   VQA/agent.debate.mc.py here, beside llama.cpp (InternVL3.5)
    this      sends each judge a Q_ID, relays every statement, writes the verdict

**Only a Q_ID goes over the wire.** Each judge opens its own copy of
`videos_TEST_VQA/ranking/<Q_ID>.json` (built by `VQA/build.ranking.folder.py`)
and its own frames. The judges never talk to each other: this process combines
their statements into one transcript and sends that back to both, so each
side's context is exactly what we chose to show it.

    R0   both judges form their own opinion, independently
    R1   both see the whole transcript; hold or switch, citing frame times
    R2   both see it again; the final statement -- concede, or say what the
         cited frames actually show
    ->   the same first choice at any point wins; still split -> the baseline

**Safe by construction.** The emitted run equals the baseline run everywhere
except where the panel disputes AND the debate converges on a different option.
A half-finished run is therefore still a valid, submittable file, and
``--max-overrules`` caps how far it may ever drift. The verdict is validated
with ``validate_vqa_mc_2026.py --warnings-as-errors`` before this exits.

Usage:
    python VQA/debate.rank.mc.py \\
        --agent-a http://192.168.1.12:7801 --agent-b http://127.0.0.1:7802 \\
        --runs qwen38.run.task2.json qwen3vl.run.task2.json gemma4.vllm.run.task2.json \\
               cosmos.run.task2.json internvl.run.task2.json llavavideo.run.task2.json \\
               videollama3.run.task2.json \\
        --out debate.run.task2.json --cache rankings.task2.debate.cache.jsonl

    ... --limit 20        smoke test
    ... --rebuild         re-emit and re-validate from the cache, no inference
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
JUDGE_SCRIPT = ROOT / "VQA" / "judge.rank.mc.py"
VALIDATOR = ROOT / "validate_vqa_mc_2026.py"


def load_judge():
    """The single-judge pass, imported for the panel helpers."""
    spec = importlib.util.spec_from_file_location("judge_rank_mc", JUDGE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JUDGE = load_judge()
MUSE = JUDGE.MUSE

DEFAULT_ROUNDS = 2
DEFAULT_WORKERS = 1


class Agent:
    """One judge, reached by Q_ID. It reads its own files; we send none."""

    def __init__(self, side: str, url: str, timeout: float, retries: int) -> None:
        self.side = side
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.local = threading.local()

    @property
    def session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def health(self) -> dict:
        response = self.session.get(self.url + "/health", timeout=30)
        response.raise_for_status()
        return response.json()

    def post(self, route: str, body: dict) -> dict:
        last: Exception | None = None

        for attempt in range(self.retries):
            try:
                response = self.session.post(self.url + route, json=body,
                                             timeout=self.timeout)

                # A missing brief or missing frames will not fix itself.
                if response.status_code == 404:
                    raise LookupError(response.json().get("error"))

                if response.status_code != 200:
                    raise RuntimeError(f"{response.status_code}: "
                                       f"{response.text[:300]}")

                statement = response.json()

                if statement.get("agent") != self.side:
                    raise RuntimeError(f"expected judge {self.side}, got "
                                       f"{statement.get('agent')!r} -- are the "
                                       f"two agent URLs swapped?")

                return statement
            except LookupError:
                raise
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                last = exc

                if attempt + 1 < self.retries:
                    time.sleep(5.0 * (attempt + 1))

        raise RuntimeError(f"judge {self.side}: {type(last).__name__}: {last}")


def both(agents: tuple[Agent, Agent], route: str, body: dict,
         pool: ThreadPoolExecutor) -> list[dict]:
    """Ask both judges at once -- they sit on different machines."""
    futures = [pool.submit(agent.post, route, body) for agent in agents]
    return [future.result() for future in futures]


def debate_one(q_id: int, video_id: str, agents: tuple[Agent, Agent],
               rounds: int, seed: int, pool: ThreadPoolExecutor) -> dict:
    """R0 opinions, then up to `rounds` rounds over the shared transcript."""
    transcript: list[dict] = both(agents, "/opinion",
                                  {"q_id": q_id, "seed": seed}, pool)
    a, b = transcript

    for round_no in range(1, rounds + 1):
        if a["pick"] == b["pick"]:
            break

        # The combined opinion of both judges goes back to both, verbatim and
        # in order: each sees everything the other has said so far.
        a, b = both(agents, "/round",
                    {"q_id": q_id, "round": round_no, "rounds": rounds,
                     "transcript": transcript, "seed": seed}, pool)
        transcript += [a, b]

    converged = a["pick"] == b["pick"]
    # Where they converge, judge a's full ranking is emitted; its tail is as
    # considered as b's, and a fixed choice keeps a re-run reproducible.
    return {
        "Q_ID": q_id, "Video_ID": video_id,
        "turns": transcript,
        "rounds_used": transcript[-1]["round"],
        "converged": converged,
        "converged_round": transcript[-1]["round"] if converged else None,
        "ranking": a["ranking"] if converged else None,
        "argument": a["argument"] if converged else None,
        "n_repaired": sum(1 for t in transcript if t.get("repaired")),
        "elapsed": round(sum(t.get("elapsed") or 0 for t in transcript), 2),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def emit(queries: list[dict], panel: dict, baseline: str, cache: dict,
         out: Path, max_overrules: int | None) -> dict:
    """Baseline everywhere, except where the debate converged on something else."""
    run, tail_run = [], []
    stats = {"entries": 0, "overruled": 0, "capped": 0, "missing": 0}

    for query in queries:
        q_id = query["Q_ID"]
        options = query["Options"]
        base = panel[baseline].get(q_id)

        if not base:
            stats["missing"] += 1
            continue

        order = list(base["ranking"])
        record = cache.get(q_id)

        if record and record.get("ranking") and record["ranking"][0] != order[0]:
            if max_overrules is not None and stats["overruled"] >= max_overrules:
                stats["capped"] += 1
            else:
                order = list(record["ranking"])
                stats["overruled"] += 1

        winner = order[0]
        tail = [winner] + [i for i in base["ranking"] if i != winner]

        for target, indices in ((run, order), (tail_run, tail)):
            answers = [{"Rank": rank, "Answer": options[i]}
                       for rank, i in enumerate(indices, 1)]
            MUSE.check_permutation([a["Answer"] for a in answers], options, q_id)
            target.append({"Q_ID": q_id, "Video_ID": query["Video_ID"],
                           "Answers": answers})

        stats["entries"] += 1

    for path, payload in ((out, run), (out.with_suffix(".tail.json"), tail_run)):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, ensure_ascii=False)
            handle.write("\n")

    return stats


def validate(queries_path: Path, run_path: Path) -> bool:
    """The official validator, strict, on the file just written."""
    report_path = run_path.with_suffix(".validation.json")
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--queries", str(queries_path),
         "--run", str(run_path), "--warnings-as-errors",
         "--report-json", str(report_path)],
        capture_output=True, text=True)
    tail = (result.stdout + result.stderr).strip().splitlines()[-3:]
    print(f"  validate {run_path.name}: "
          f"{'VALID' if result.returncode == 0 else 'INVALID'}")

    for line in tail:
        print(f"    {line}")

    return result.returncode == 0


def report(cache: dict, stats: dict) -> None:
    decided = [r for r in cache.values() if r.get("turns")]

    if not decided:
        return

    converged = [r for r in decided if r.get("converged")]
    by_round: dict[int, int] = {}

    for record in converged:
        by_round[record["converged_round"]] = \
            by_round.get(record["converged_round"], 0) + 1

    print(f"\ndebated {len(decided)} queries "
          f"({sum(1 for r in cache.values() if r.get('error'))} errors, "
          f"left at the baseline)")
    print(f"  agreed at R0, no debate needed:     "
          f"{by_round.get(0, 0)} ({100 * by_round.get(0, 0) / len(decided):.1f}%)")

    for index in sorted(k for k in by_round if k):
        print(f"  converged after round {index}:             {by_round[index]}")

    unresolved = len(decided) - len(converged)
    print(f"  never converged -> baseline kept:   {unresolved} "
          f"({100 * unresolved / len(decided):.1f}%)")
    print(f"  repaired rankings:                  "
          f"{sum(r['n_repaired'] for r in decided)}")
    print(f"\n  queries whose rank 1 differs from the baseline: "
          f"{stats['overruled']}"
          + (f"  (+{stats['capped']} suppressed by --max-overrules)"
             if stats["capped"] else ""))

    turns = [t for r in decided for t in r["turns"]]

    for side in ("a", "b"):
        mine = [t for t in turns if t.get("agent") == side]
        switched = sum(1 for t in mine if t.get("round")
                       and t.get("stance") in ("switch", "concede"))
        tokens = [t["prompt_tokens"] for t in mine if t.get("prompt_tokens")]
        logprobs = [t["logprob"] for t in mine if t.get("logprob") is not None]

        if mine:
            print(f"  judge {side}: {len(mine)} turns, switched/conceded "
                  f"{switched}, {sum(t.get('elapsed') or 0 for t in mine) / len(mine):5.1f}s"
                  f" and {sum(tokens) / max(len(tokens), 1):6.0f} prompt tokens per turn"
                  + (f", mean logprob {sum(logprobs) / len(logprobs):+.3f}"
                     if logprobs else ", no logprobs"))

    # Who carried the converged debates away from R0: the judge whose R0 pick
    # became the verdict.
    won = {"a": 0, "b": 0}

    for record in converged:
        if record["converged_round"]:
            final = record["ranking"][0]
            for t in record["turns"][:2]:
                if t["pick"] == final:
                    won[t["agent"]] += 1

    if any(won.values()):
        print(f"  debated to a verdict, R0 pick kept by: judge a {won['a']}, "
              f"judge b {won['b']}")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Orchestrate the two-judge debate by Q_ID.")

    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--baseline", type=Path,
                    help="run used for unanimous queries and every fallback "
                         "(default: --runs[0])")
    ap.add_argument("--queries", type=Path, default=MUSE.DEFAULT_QUERIES)
    ap.add_argument("--ranking", type=Path,
                    default=ROOT / "videos_TEST_VQA" / "ranking",
                    help="the local copy of the briefs; its _index.json is the "
                         "list of queries to debate")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)

    ap.add_argument("--agent-a", help="judge a, e.g. http://192.168.1.12:7801")
    ap.add_argument("--agent-b", help="judge b, e.g. http://127.0.0.1:7802")

    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                    help="debate rounds after the opinions (default: %(default)s)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="queries in flight. 1 keeps each server's prompt cache "
                         "on one video through all its rounds")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-overrules", type=int,
                    help="cap how many queries may differ from the baseline")

    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", type=int, nargs="+", metavar="Q_ID")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rebuild", action="store_true")

    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    baseline_path = args.baseline or args.runs[0]
    labels = [JUDGE.label_for(p) for p in args.runs]
    baseline = JUDGE.label_for(baseline_path)

    if baseline not in labels:
        raise SystemExit(f"--baseline {baseline} is not among --runs")

    queries = MUSE.load_queries(args.queries)
    panel = JUDGE.load_panel(args.runs, queries)
    cache = {int(k): v for k, v in MUSE.load_cache(args.cache).items()}
    index = MUSE.load_json(args.ranking / "_index.json")["queries"]
    video_of = {q["Q_ID"]: q["Video_ID"] for q in queries}

    print(f"panel: {', '.join(labels)} | baseline: {baseline}")
    print(f"briefs: {len(index)} disputed queries in {args.ranking}")

    if not args.rebuild:
        if not (args.agent_a and args.agent_b):
            raise SystemExit("--agent-a and --agent-b are required unless --rebuild")

        agents = (Agent("a", args.agent_a, args.timeout, args.retries),
                  Agent("b", args.agent_b, args.timeout, args.retries))

        # Both judges must hold the same briefs, or they would be arguing
        # about different questions under the same number.
        for agent in agents:
            info = agent.health()
            print(f"judge {agent.side}: {info.get('model')} @ {agent.url} "
                  f"({info.get('backend')}, {info.get('transport')}), "
                  f"{info.get('briefs')} briefs, {info.get('videos_cut')} "
                  f"videos cut")

            if info.get("side") != agent.side:
                raise SystemExit(f"{agent.url} is judge {info.get('side')}, "
                                 f"not {agent.side}")
            if info.get("briefs") != len(index):
                raise SystemExit(f"judge {agent.side} has {info.get('briefs')} "
                                 f"briefs, this machine has {len(index)} -- "
                                 f"copy the ranking folder again")

        todo = [q["Q_ID"] for q in index if args.force or q["Q_ID"] not in cache]

        if args.only:
            wanted = set(args.only)
            todo = [q for q in todo if q in wanted]

        if args.limit:
            todo = todo[:args.limit]

        print(f"to debate: {len(todo)} | {args.rounds} round(s) after the "
              f"opinions | {args.workers} in flight")

        lock = threading.Lock()
        done = 0
        failures = [0]
        started = time.monotonic()
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        handle = args.cache.open("a", encoding="utf-8")
        relay = ThreadPoolExecutor(max_workers=2 * max(args.workers, 1))

        def work(q_id: int) -> None:
            nonlocal done

            try:
                record = debate_one(q_id, video_of[q_id], agents, args.rounds,
                                    args.seed, relay)
            except Exception as exc:  # noqa: BLE001 - one bad query must not
                record = {"Q_ID": q_id, "Video_ID": video_of[q_id],  # stop the run
                          "turns": [], "converged": False, "ranking": None,
                          "error": f"{type(exc).__name__}: {exc}"[:300],
                          "ts": datetime.now(timezone.utc).isoformat(
                              timespec="seconds")}

            with lock:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                cache[q_id] = record
                done += 1
                failures[0] = failures[0] + 1 if record.get("error") else 0

                rate = done / max(time.monotonic() - started, 1e-9)
                mark = ("R%d" % record["converged_round"]
                        if record.get("converged") else "SPLIT")
                note = record.get("argument") or ""

                if record.get("error"):
                    mark, note = "ERROR", record["error"]
                elif not record.get("converged"):
                    note = " vs ".join(f"{t['agent']}:{t['pick'] + 1}"
                                       for t in record["turns"][-2:])

                print(f"  [{done}/{len(todo)}] {rate * 60:5.2f}/min "
                      f"eta {(len(todo) - done) / max(rate, 1e-9) / 3600:5.1f}h "
                      f"Q{q_id} {mark}: {str(note)[:70]}", flush=True)

                if failures[0] >= 10:
                    raise SystemExit(
                        "10 debates in a row failed -- a judge is not "
                        "answering. Fix it and re-run; finished queries are "
                        "cached.")

        try:
            if todo:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    list(pool.map(work, todo))
        except KeyboardInterrupt:
            print("\ninterrupted -- finished debates are cached, re-run to "
                  "resume", file=sys.stderr)
            return 130
        finally:
            handle.close()
            relay.shutdown(wait=False)

        if todo:
            print(f"\ndebated {done} queries in "
                  f"{(time.monotonic() - started) / 60:.1f} min")

    stats = emit(queries, panel, baseline, cache, args.out, args.max_overrules)
    print(f"\nwrote {stats['entries']} entries to {args.out} "
          f"(and {args.out.with_suffix('.tail.json').name})")

    if stats["missing"]:
        print(f"  {stats['missing']} queries are missing from the baseline run")

    report(cache, stats)
    print()
    valid = all([validate(args.queries, args.out),
                 validate(args.queries, args.out.with_suffix(".tail.json"))])
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

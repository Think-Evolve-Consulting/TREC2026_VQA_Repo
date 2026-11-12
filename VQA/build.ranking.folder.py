#!/usr/bin/env python3
"""Write one self-contained debate brief per disputed query.

The debate is addressed by Q_ID alone: the orchestrator sends a number and each
judge agent opens `<dataset>/ranking/<Q_ID>.json` on its own disk. That only
works if the brief carries everything a judge needs, because the Mac side has a
copy of the dataset and nothing else -- no run files, no caches, no queries
file, no evidence file.

    ranking/
      _card.txt         the reliability card, identical for every query
      _manifest.json    stem -> {n_frames, duration_sec}: the frame cut plan
      _index.json       the disputed Q_IDs, with stem and n_options
      <Q_ID>.json       one brief

Only the queries the panel disputes get a brief (1,346 of 2,443). The other
1,097 are unanimous on rank 1 and are copied through from the baseline
untouched, so a brief for them would never be opened.

Usage:
    python VQA/build.ranking.folder.py
    python VQA/build.ranking.folder.py --out D:/somewhere/ranking --limit 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUDGE_SCRIPT = ROOT / "VQA" / "judge.rank.mc.py"

spec = importlib.util.spec_from_file_location("judge_rank_mc", JUDGE_SCRIPT)
JUDGE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(JUDGE)
MUSE = JUDGE.MUSE

DEFAULT_RUNS = ["qwen38.run.task2.json", "qwen3vl.run.task2.json",
                "gemma4.vllm.run.task2.json", "cosmos.run.task2.json",
                "internvl.run.task2.json", "llavavideo.run.task2.json",
                "videollama3.run.task2.json"]
BASELINE = "qwen38"

# A judge that has to read 8 options and 7 panel rankings does not also need
# ten minutes of speech; the transcripts that matter here are short.
TRANSCRIPT_CHARS = 4000


def borda(run_path: Path, queries: set[int]) -> dict[int, dict]:
    """Per-query Borda detail from a run's cache: score and firsts per option.

    Already computed when the run was made and never used since. It separates
    "ranked first by a hair" from "ranked first by all three passes", which is
    exactly what a judge weighing the panel needs.
    """
    cache_path = JUDGE.cache_for(run_path)

    if not cache_path:
        return {}

    detail: dict[int, dict] = {}

    for record in MUSE.load_cache(cache_path).values():
        q_id = int(record["Q_ID"])

        if q_id not in queries:
            continue

        ranked = record.get("ranked") or []
        detail[q_id] = {
            "scores": {int(r["option_index"]): round(float(r["score"]), 1)
                       for r in ranked if "option_index" in r},
            "firsts": {int(r["option_index"]): int(r.get("firsts") or 0)
                       for r in ranked if "option_index" in r},
        }

    return detail


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--queries", type=Path, default=MUSE.DEFAULT_QUERIES)
    ap.add_argument("--runs", type=Path, nargs="+",
                    default=[ROOT / n for n in DEFAULT_RUNS])
    ap.add_argument("--routing", type=Path, default=MUSE.DEFAULT_ROUTING)
    ap.add_argument("--transcripts", type=Path, default=MUSE.DEFAULT_TRANSCRIPTS)
    ap.add_argument("--manifest", type=Path, default=ROOT / "debate.manifest.json")
    ap.add_argument("--evidence", type=Path, default=ROOT / "evidence.2026.json")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "videos_TEST_VQA" / "ranking")
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--limit", type=int, help="only the first N briefs")
    ap.add_argument("--clean", action="store_true",
                    help="delete the folder first instead of overwriting")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    queries = MUSE.load_queries(args.queries)
    labels = [JUDGE.label_for(p) for p in args.runs]

    if args.baseline not in labels:
        raise SystemExit(f"baseline {args.baseline} is not among {labels}")

    panel = JUDGE.load_panel(args.runs, queries)
    routing = MUSE.load_routing(args.routing)
    manifest = MUSE.load_json(args.manifest)
    evidence = MUSE.load_json(args.evidence) if args.evidence.exists() else {}
    videos = manifest["videos"]

    covered = [q for q in queries if all(panel[m].get(q["Q_ID"]) for m in labels)]
    disputed = [q for q in covered
                if len(JUDGE.candidates_for(panel, labels, q["Q_ID"])) > 1]

    if args.limit:
        disputed = disputed[:args.limit]

    wanted = {q["Q_ID"] for q in disputed}
    detail = {label: borda(path, wanted)
              for label, path in zip(labels, args.runs)}

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)

    args.out.mkdir(parents=True, exist_ok=True)

    index, no_video = [], 0

    for query in disputed:
        q_id = query["Q_ID"]
        options = query["Options"]
        n = len(options)
        stem = Path(query["Video_ID"]).stem
        video = videos.get(stem)

        # No frame plan means no frames on either machine, so there is nothing
        # for the judges to argue from; the query keeps the baseline ranking.
        if not video:
            no_video += 1
            continue

        members = []

        for label in labels:
            entry = panel[label][q_id]
            extra = detail.get(label, {}).get(q_id, {})
            pick = entry["ranking"][0]
            members.append({
                "model": label,
                "ranking": entry["ranking"],
                "pick": pick,
                "agreement": entry.get("agreement"),
                "score": (extra.get("scores") or {}).get(pick),
                "firsts": (extra.get("firsts") or {}).get(pick),
            })

        transcript = MUSE.load_transcript(args.transcripts, query["Video_ID"])
        text = (transcript or {}).get("text") or ""
        aid = evidence.get(str(q_id)) or evidence.get(q_id) or {}

        brief = {
            "Q_ID": q_id,
            "Video_ID": query["Video_ID"],
            "stem": stem,
            "question": query["Question"],
            "options": options,
            "n_options": n,
            "duration_sec": video["duration_sec"],
            "n_frames": video["n_frames"],
            "panel": members,
            # Rendered here rather than in the agents, so both judges read the
            # panel in exactly the same words -- a difference there would be a
            # difference in the debate.
            "panel_table": JUDGE.panel_table(panel, labels, q_id, n),
            "candidates": JUDGE.candidates_for(panel, labels, q_id),
            "routing": JUDGE.routing_line(routing, q_id, query["Question"]),
            "transcript": {"text": text[:TRANSCRIPT_CHARS],
                           "language": (transcript or {}).get("language")}
            if text else None,
            "evidence": aid,
            "baseline": {"model": args.baseline,
                         "ranking": panel[args.baseline][q_id]["ranking"],
                         "pick": panel[args.baseline][q_id]["ranking"][0]},
        }

        with (args.out / f"{q_id}.json").open("w", encoding="utf-8") as handle:
            json.dump(brief, handle, indent=1, ensure_ascii=False)
            handle.write("\n")

        index.append({"Q_ID": q_id, "stem": stem, "n_options": n,
                      "n_frames": video["n_frames"],
                      "counting": bool(aid.get("counting")),
                      "audio": bool(aid.get("audio_scores"))})

    for name, payload in (("_index.json", {"n": len(index), "queries": index}),
                          ("_manifest.json", manifest)):
        with (args.out / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
            handle.write("\n")

    (args.out / "_card.txt").write_text(JUDGE.RELIABILITY_CARD + "\n",
                                        encoding="utf-8")

    size = sum(p.stat().st_size for p in args.out.glob("*.json"))
    counting = sum(1 for i in index if i["counting"])
    audio = sum(1 for i in index if i["audio"])
    print(f"panel: {', '.join(labels)}  (baseline {args.baseline})")
    print(f"disputed: {len(disputed)} of {len(covered)} covered queries")
    print(f"briefs:   {len(index)}  ({counting} counting, {audio} with audio "
          f"scores; {no_video} skipped for no frame plan)")
    print(f"wrote {args.out}  ({size / 1024 / 1024:.1f} MB) "
          f"<- copy this folder to the Mac")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

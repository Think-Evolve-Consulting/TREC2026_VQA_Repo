#!/usr/bin/env python3
"""Turn the 2025 ground truth into a 150-video evaluation sample.

Stages 2-3 of ``VQA/EVALUATION_PLAN.md``, minus the parts that are now moot:
the videos are already downloaded (``videos_GT_2025/``, 1,327 of 2,000), so no
metadata probe and no download stage.

Two subcommands:

``convert``
    ``vqa.gt.2025.csv`` (headerless ``Q_ID,Video_ID,Question,Answer``) becomes
    the query JSON every downstream script wants, plus the *eligible pool*: the
    rows whose video is actually on disk and whose Q_ID is not in
    ``eval.vqa.ag.py``'s hard-coded ``SKIP`` set. Also asserts the 2025 GT
    videos and the 2026 test videos are disjoint -- ``generate.answers.qwen3vl``
    's docstring claims it, nothing verified it.

``select``
    Draws the sample. **The duration strata in EVALUATION_PLAN.md cannot be
    matched**: every 2025 GT clip is a YouTube Short (5-30 s, median 15 s, max
    30 s), so the 2026 bins ``0-5 s`` (11.9% of the test set), ``60-150 s``
    (10.9%) and ``150 s+`` (6.3%) are empty here. The draw is therefore
    proportional to the *observed* pool -- 13/43/88/6 over 5-10/10-15/15-30/
    30-60 s at n=150 -- and the report states the shortfall against 2026
    rather than hiding it behind a reweighting.

Selection is a plain stratified draw with a fixed seed, and each stratum keeps
a ranked reserve so a video that dies later is backfilled from its own bin
instead of distorting the sample.

Usage:
    python VQA/sample.2025.eval.py convert
    python VQA/sample.2025.eval.py select --n 150 --seed 20260814
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GT = ROOT / "vqa.gt.2025.csv"
DEFAULT_VIDEOS = ROOT / "videos_GT_2025" / "videos"
DEFAULT_METADATA = ROOT / "videos_GT_2025" / "metadata"
DEFAULT_QUERIES_OUT = ROOT / "vqa.gt.2025.queries.json"
DEFAULT_POOL_OUT = ROOT / "vqa.gt.2025.pool.json"
DEFAULT_DURATIONS = ROOT / "duration.report.gt2025.json"
DEFAULT_TEST_2026 = ROOT / "vqa.2026.testingDataset.Task1.json"

# eval.vqa.ag.py:91 -- "skip these queries because their videos were not
# available for download (18 videos in 2025)". Scoring honours this set, so
# there is no point spending inference on a query that will be dropped.
SKIP = {18, 27, 62, 222, 448, 507, 584, 629, 903, 1007, 1159, 1260, 1330,
        1381, 1554, 1691, 1755, 1822}

# The bin edges EVALUATION_PLAN.md measured the 2026 test set with, kept
# identical so achieved-vs-target is a like-for-like table.
BINS = ((0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 30.0),
        (30.0, 60.0), (60.0, 150.0), (150.0, float("inf")))

TARGET_2026 = {
    "0-5s": 0.119, "5-10s": 0.171, "10-15s": 0.294, "15-30s": 0.195,
    "30-60s": 0.042, "60-150s": 0.109, "150s+": 0.063,
}


def bin_label(low: float, high: float) -> str:
    return f"{low:g}s+" if high == float("inf") else f"{low:g}-{high:g}s"


BIN_LABELS = [bin_label(lo, hi) for lo, hi in BINS]


def bin_of(duration: float | None) -> str | None:
    if duration is None or duration <= 0:
        return None

    for low, high in BINS:
        if low <= duration < high:
            return bin_label(low, high)

    return BIN_LABELS[-1]


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)
        handle.write("\n")


def read_gt(path: Path) -> list[dict]:
    """The headerless 4-column ground truth, as dicts."""
    rows: list[dict] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_no, row in enumerate(csv.reader(handle), 1):
            if not row or not row[0].strip():
                continue

            if len(row) < 4:
                print(f"  {path.name}:{line_no}: expected 4 columns, got "
                      f"{len(row)} -- skipped", file=sys.stderr)
                continue

            try:
                q_id = int(row[0].strip())
            except ValueError:
                print(f"  {path.name}:{line_no}: non-numeric Q_ID "
                      f"{row[0]!r} -- skipped", file=sys.stderr)
                continue

            rows.append({
                "Q_ID": q_id,
                "Video_ID": row[1].strip(),
                "Question": row[2].strip(),
                "Answer": row[3].strip(),
            })

    return rows


def write_gt(path: Path, rows: list[dict]) -> None:
    """Emit the 4-column headerless shape eval.vqa.ag.py reads, LF-terminated
    like the source file."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")

        for row in rows:
            writer.writerow([row["Q_ID"], row["Video_ID"], row["Question"],
                             row["Answer"]])


def video_path(videos_dir: Path, video_id: str) -> Path:
    """Same convention as every other script here -- extract.frames.py:132."""
    name = video_id if video_id.endswith(".webm") else video_id + ".mp4"
    return videos_dir / name


def durations_from_report(path: Path) -> dict[str, float]:
    """video_id -> probed duration, from VQA/duration.report.py's output."""
    if not path.exists():
        return {}

    return {record["video_id"]: record["duration_sec"]
            for record in load_json(path).get("videos", [])
            if record.get("duration_sec")}


def durations_from_metadata(directory: Path, video_ids: list[str]) -> dict[str, float]:
    """Fallback: yt-dlp's own ``length``, rounded down to the second.

    Spot-checked against ffprobe (14 vs 14.03, 6 vs 6.27) -- good enough to bin
    with, but the probed duration is preferred when there is one. These files
    must be read as utf-8; several carry bytes cp1252 cannot decode.
    """
    found: dict[str, float] = {}

    for video_id in video_ids:
        path = directory / f"{Path(video_id).stem}.json"

        if not path.exists():
            continue

        try:
            with path.open("r", encoding="utf-8") as handle:
                length = json.load(handle).get("length")
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue

        if isinstance(length, (int, float)) and length > 0:
            found[video_id] = float(length)

    return found


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------

def cmd_convert(args: argparse.Namespace) -> int:
    rows = read_gt(args.gt)
    print(f"read {len(rows)} ground-truth rows from {args.gt.name}")

    ids = [row["Video_ID"] for row in rows]

    if len(set(ids)) != len(ids):
        print(f"  note: {len(ids) - len(set(ids))} duplicate Video_IDs "
              f"(the 2025 set is normally one question per video)")

    queries = [{"Q_ID": row["Q_ID"], "Video_ID": row["Video_ID"],
                "Question": row["Question"]} for row in rows]
    write_json(args.queries_out, queries)
    print(f"wrote {len(queries)} queries to {args.queries_out.name}")

    on_disk = [row for row in rows
               if video_path(args.videos, row["Video_ID"]).exists()]
    eligible = [row for row in on_disk if row["Q_ID"] not in SKIP]

    dropped_skip = len(on_disk) - len(eligible)

    print(f"videos on disk:        {len(on_disk)} of {len(rows)}")
    print(f"dropped by SKIP set:   {dropped_skip}")
    print(f"eligible pool:         {len(eligible)}")

    write_json(args.pool_out, eligible)
    print(f"wrote {args.pool_out.name}")

    # The qwen3vl docstring asserts the 2025 GT shares no videos with the 2026
    # test set. Nothing checked it, and an overlap would mean a 2025 stage could
    # write into media the 2026 submission depends on.
    if args.test_2026 and args.test_2026.exists():
        test_ids = {q["Video_ID"] for q in load_json(args.test_2026)}
        overlap = {row["Video_ID"] for row in rows} & test_ids

        if overlap:
            print(f"\nFAIL: {len(overlap)} video(s) appear in BOTH the 2025 "
                  f"ground truth and the 2026 test set: "
                  f"{sorted(overlap)[:10]}", file=sys.stderr)
            return 1

        print(f"disjoint from the 2026 test set ({len(test_ids)} videos): ok")

    if not eligible:
        print("\nno eligible videos -- nothing to sample", file=sys.stderr)
        return 1

    return 0


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder proportional allocation of `total` across `counts`.

    Never allocates more than a bin actually holds; anything that cannot be
    placed spills to the bins that still have room, largest first.
    """
    pool = sum(counts.values())

    if pool <= 0:
        return {label: 0 for label in counts}

    exact = {label: total * n / pool for label, n in counts.items()}
    quota = {label: min(int(value), counts[label])
             for label, value in exact.items()}
    remainder = total - sum(quota.values())

    order = sorted(counts, key=lambda label: (-(exact[label] - int(exact[label])),
                                              -counts[label], label))

    while remainder > 0:
        placed = False

        for label in order:
            if remainder <= 0:
                break

            if quota[label] < counts[label]:
                quota[label] += 1
                remainder -= 1
                placed = True

        if not placed:  # every bin is exhausted -- the pool is smaller than n
            break

    return quota


def cmd_select(args: argparse.Namespace) -> int:
    pool = load_json(args.pool)
    print(f"pool: {len(pool)} eligible queries")

    ids = [row["Video_ID"] for row in pool]
    durations = durations_from_report(args.durations)
    source = f"{args.durations.name} (ffprobe)"

    missing = [vid for vid in ids if vid not in durations]

    if missing:
        fallback = durations_from_metadata(args.metadata, missing)
        durations.update(fallback)

        if fallback:
            source += f" + {args.metadata.name} length ({len(fallback)})"

        print(f"  {len(missing)} without a probed duration, "
              f"{len(fallback)} recovered from metadata")

    binned: dict[str, list[dict]] = {label: [] for label in BIN_LABELS}
    unbinned: list[dict] = []

    for row in pool:
        label = bin_of(durations.get(row["Video_ID"]))

        if label is None:
            unbinned.append(row)
            continue

        entry = dict(row)
        entry["duration_sec"] = round(durations[row["Video_ID"]], 3)
        binned[label].append(entry)

    if unbinned:
        print(f"  {len(unbinned)} with no duration at all -- excluded",
              file=sys.stderr)

    counts = {label: len(rows) for label, rows in binned.items() if rows}
    quota = allocate(counts, args.n)

    rng = random.Random(args.seed)
    sample: list[dict] = []
    reserves: dict[str, list[dict]] = {}

    for label in BIN_LABELS:
        rows = binned.get(label) or []

        if not rows:
            continue

        # Sort before shuffling so the draw depends only on the seed, never on
        # dict or filesystem ordering.
        rows = sorted(rows, key=lambda r: r["Q_ID"])
        rng.shuffle(rows)

        take = quota.get(label, 0)
        sample.extend(rows[:take])
        reserves[label] = rows[take:take + max(take, args.reserve_min)]

    sample.sort(key=lambda r: r["Q_ID"])

    if len(sample) < args.n:
        print(f"\nWARNING: drew {len(sample)} of {args.n} requested -- the "
              f"pool cannot fill every bin", file=sys.stderr)

    queries = [{"Q_ID": row["Q_ID"], "Video_ID": row["Video_ID"],
                "Question": row["Question"]} for row in sample]
    write_json(args.out, queries)

    write_gt(args.gt_out, sample)
    write_json(args.reserves_out, reserves)

    achieved = {label: 0 for label in BIN_LABELS}

    for row in sample:
        label = bin_of(row["duration_sec"])

        if label:
            achieved[label] += 1

    lengths = sorted(len(row["Answer"].split()) for row in sample)
    seconds = [row["duration_sec"] for row in sample]

    report = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "requested": args.n,
        "drawn": len(sample),
        "pool_size": len(pool),
        "duration_source": source,
        "bins": [
            {
                "bin": label,
                "pool": counts.get(label, 0),
                "pool_share": round(counts.get(label, 0) / max(len(pool), 1), 4),
                "quota": quota.get(label, 0),
                "achieved": achieved[label],
                "target_2026_share": TARGET_2026.get(label, 0.0),
                "target_2026_count": round(TARGET_2026.get(label, 0.0) * args.n),
                "reserve": len(reserves.get(label, [])),
            }
            for label in BIN_LABELS
        ],
        "duration_sec": {
            "min": round(min(seconds), 2) if seconds else None,
            "median": round(sorted(seconds)[len(seconds) // 2], 2) if seconds else None,
            "mean": round(sum(seconds) / len(seconds), 2) if seconds else None,
            "max": round(max(seconds), 2) if seconds else None,
            "total_min": round(sum(seconds) / 60, 1) if seconds else None,
        },
        "gt_answer_words": {
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else None,
            "median": lengths[len(lengths) // 2] if lengths else None,
            "p90": lengths[int(0.9 * len(lengths))] if lengths else None,
            "max": lengths[-1] if lengths else None,
        },
        "q_id_range": [min(r["Q_ID"] for r in sample),
                       max(r["Q_ID"] for r in sample)] if sample else None,
    }

    write_json(args.report_json, report)
    write_markdown(args.report_md, report)

    print(f"\nwrote {len(queries)} queries to {args.out.name}")
    print(f"wrote {len(sample)} ground-truth rows to {args.gt_out.name}")
    print(f"wrote reserves to {args.reserves_out.name}")
    print(f"wrote {args.report_json.name} and {args.report_md.name}")

    print(f"\n{'bin':>10}  {'pool':>6} {'drawn':>6} {'2026 target':>12}")

    for block in report["bins"]:
        if not block["pool"] and not block["target_2026_count"]:
            continue

        print(f"{block['bin']:>10}  {block['pool']:6d} {block['achieved']:6d} "
              f"{block['target_2026_count']:12d}")

    stats = report["duration_sec"]
    print(f"\nduration: median {stats['median']}s  mean {stats['mean']}s  "
          f"max {stats['max']}s  total {stats['total_min']} min")

    return 0


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# 2025 evaluation sample",
        "",
        f"Generated {report['generated']} · seed `{report['seed']}` · "
        f"{report['drawn']} of {report['requested']} requested, "
        f"drawn from a pool of {report['pool_size']}.",
        "",
        f"Durations from {report['duration_source']}.",
        "",
        "## Duration strata",
        "",
        "| Bin | Pool | Pool share | Drawn | 2026 target | Reserve |",
        "|---|---|---|---|---|---|",
    ]

    for block in report["bins"]:
        if not block["pool"] and not block["target_2026_count"]:
            continue

        lines.append(
            f"| {block['bin']} | {block['pool']} | "
            f"{block['pool_share'] * 100:.1f}% | {block['achieved']} | "
            f"{block['target_2026_count']} | {block['reserve']} |"
        )

    stats = report["duration_sec"]
    words = report["gt_answer_words"]

    lines += [
        "",
        "**The 2026 target column is unreachable and shown only for honesty.** "
        "Every 2025 ground-truth clip is a YouTube Short, so the `0-5s`, "
        "`60-150s` and `150s+` bins have no candidates at all. This sample "
        "measures the pipeline on short video; it says nothing about how it "
        "handles the 2026 set's long tail.",
        "",
        "## Sample",
        "",
        f"- duration: min {stats['min']}s, median {stats['median']}s, "
        f"mean {stats['mean']}s, max {stats['max']}s "
        f"({stats['total_min']} min of footage)",
        f"- ground-truth answer length: mean {words['mean']} words, "
        f"median {words['median']}, p90 {words['p90']}, max {words['max']}",
        f"- Q_ID range: {report['q_id_range'][0]}-{report['q_id_range'][1]}"
        if report["q_id_range"] else "",
        "",
    ]

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(line for line in lines if line is not None))
        handle.write("\n")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a duration-stratified evaluation sample from the "
                    "2025 ground truth."
    )
    sub = ap.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="CSV -> query JSON + eligible pool")
    convert.add_argument("--gt", type=Path, default=DEFAULT_GT)
    convert.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    convert.add_argument("--queries-out", type=Path, default=DEFAULT_QUERIES_OUT)
    convert.add_argument("--pool-out", type=Path, default=DEFAULT_POOL_OUT)
    convert.add_argument("--test-2026", type=Path, default=DEFAULT_TEST_2026,
                         help="asserted disjoint from the 2025 GT videos")
    convert.set_defaults(func=cmd_convert)

    select = sub.add_parser("select", help="draw the stratified sample")
    select.add_argument("--pool", type=Path, default=DEFAULT_POOL_OUT)
    select.add_argument("--durations", type=Path, default=DEFAULT_DURATIONS)
    select.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    select.add_argument("-n", "--n", type=int, default=150)
    select.add_argument("--seed", type=int, default=20260814)
    select.add_argument("--reserve-min", type=int, default=5,
                        help="minimum reserves kept per stratum")
    select.add_argument("--out", type=Path,
                        default=ROOT / "vqa.2025.sample150.json")
    select.add_argument("--gt-out", type=Path,
                        default=ROOT / "vqa.gt.2025.sample150.csv")
    select.add_argument("--reserves-out", type=Path,
                        default=ROOT / "vqa.2025.sample150.reserves.json")
    select.add_argument("--report-json", type=Path,
                        default=ROOT / "sample.2025.report.json")
    select.add_argument("--report-md", type=Path,
                        default=ROOT / "sample.2025.report.md")
    select.set_defaults(func=cmd_select)

    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

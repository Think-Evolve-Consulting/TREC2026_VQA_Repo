#!/usr/bin/env python3
"""
Report local media coverage for the released VQA Task 1 testing queries.

Every Video_ID in the query file is either a local ".webm" clip or a
YouTube video ID. This script checks each one against the local media
folders and reports what is present, what is missing, and which local
files are never referenced.

Usage:
    python coverage.report.py <released_queries.json>
        [--webm-dir videos_TEST_VQA/videos]
        [--youtube-dir videos]
        [--removed-list "Videos Removed from the platform.txt"]
        [--report-json coverage.report.json]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


YOUTUBE_EXTENSIONS = (
    ".mp4",
    ".webm",
    ".mkv",
    ".m4v",
    ".mov",
)


def load_queries(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def index_directory(directory: Path) -> dict[str, Path]:
    """Map file name -> path for every file directly inside `directory`."""
    if not directory.is_dir():
        return {}

    return {
        entry.name: entry
        for entry in directory.iterdir()
        if entry.is_file()
    }


def load_removed_ids(path: Path) -> set[str]:
    """Read the 'removed from the platform' list of <id>.json names."""
    if not path.is_file():
        return set()

    removed: set[str] = set()

    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            name = line.strip()

            if not name or not name.endswith(".json"):
                continue

            removed.add(name[: -len(".json")])

    return removed


def format_list(values: list[str], maximum: int = 25) -> str:
    shown = ", ".join(values[:maximum])

    if len(values) > maximum:
        shown += f", ... and {len(values) - maximum} more"

    return shown or "(none)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report which testing-query videos are available locally."
        )
    )

    parser.add_argument(
        "released_queries",
        type=Path,
        help="JSON file with the released Q_ID, Video_ID and Question values.",
    )

    parser.add_argument(
        "--webm-dir",
        type=Path,
        default=Path("videos_TEST_VQA/videos"),
        help="Folder holding the released .webm clips.",
    )

    parser.add_argument(
        "--youtube-dir",
        type=Path,
        default=Path("videos"),
        help="Folder holding downloaded YouTube videos named <id>.<ext>.",
    )

    parser.add_argument(
        "--removed-list",
        type=Path,
        default=Path("Videos Removed from the platform.txt"),
        help="Text file listing <youtube_id>.json entries no longer online.",
    )

    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON report.",
    )

    args = parser.parse_args()

    queries = load_queries(args.released_queries)

    webm_files = index_directory(args.webm_dir)
    youtube_files = index_directory(args.youtube_dir)
    removed_ids = load_removed_ids(args.removed_list)

    youtube_stems: dict[str, str] = {}

    for name in youtube_files:
        stem, _, extension = name.rpartition(".")

        if stem and f".{extension.lower()}" in YOUTUBE_EXTENSIONS:
            youtube_stems[stem] = name

    webm_lower = {name.casefold(): name for name in webm_files}
    youtube_lower = {stem.casefold(): stem for stem in youtube_stems}

    queries_by_video: dict[str, list[int]] = defaultdict(list)

    webm_ids: set[str] = set()
    youtube_ids: set[str] = set()

    for query in queries:
        video_id = query["Video_ID"]
        queries_by_video[video_id].append(query["Q_ID"])

        if video_id.casefold().endswith(".webm"):
            webm_ids.add(video_id)
        else:
            youtube_ids.add(video_id)

    webm_present = sorted(v for v in webm_ids if v in webm_files)
    webm_missing = sorted(v for v in webm_ids if v not in webm_files)

    youtube_present = sorted(v for v in youtube_ids if v in youtube_stems)
    youtube_missing = sorted(v for v in youtube_ids if v not in youtube_stems)

    # Case-only mismatches: present on a case-insensitive filesystem but
    # not an exact-name match.
    case_mismatches = sorted(
        f"{v} -> {webm_lower[v.casefold()]}"
        for v in webm_missing
        if v.casefold() in webm_lower
    ) + sorted(
        f"{v} -> {youtube_lower[v.casefold()]}"
        for v in youtube_missing
        if v.casefold() in youtube_lower
    )

    missing_ids = set(webm_missing) | set(youtube_missing)

    affected_q_ids = sorted(
        q_id
        for video_id in missing_ids
        for q_id in queries_by_video[video_id]
    )

    youtube_missing_removed = sorted(
        v for v in youtube_missing if v in removed_ids
    )

    unused_webm = sorted(set(webm_files) - webm_ids)
    unused_youtube = sorted(set(youtube_stems) - youtube_ids)

    reuse_counts = Counter(
        {
            video_id: len(q_ids)
            for video_id, q_ids in queries_by_video.items()
        }
    )

    total_queries = len(queries)
    covered_queries = total_queries - len(affected_q_ids)

    print("=" * 72)
    print("VQA 2026 TASK 1 - LOCAL VIDEO COVERAGE REPORT")
    print("=" * 72)
    print(f"Query file:               {args.released_queries}")
    print(f"Total queries:            {total_queries}")
    print(f"Distinct Video_IDs:       {len(queries_by_video)}")
    print(
        f"  .webm clips:            {len(webm_ids)}"
        f"  (dir: {args.webm_dir})"
    )
    print(
        f"  YouTube IDs:            {len(youtube_ids)}"
        f"  (dir: {args.youtube_dir})"
    )
    print()
    print("COVERAGE")
    print(
        f"  webm present:           {len(webm_present)} / {len(webm_ids)}"
    )
    print(
        f"  webm missing:           {len(webm_missing)}"
    )
    print(
        f"  YouTube present:        "
        f"{len(youtube_present)} / {len(youtube_ids)}"
    )
    print(
        f"  YouTube missing:        {len(youtube_missing)}"
    )
    print(
        f"  Queries fully covered:  {covered_queries} / {total_queries} "
        f"({covered_queries / total_queries:.1%})"
        if total_queries
        else "  Queries fully covered:  0 / 0"
    )
    print(f"  Queries blocked:        {len(affected_q_ids)}")
    print()

    print("LOCAL FILES")
    print(f"  Files in {args.webm_dir}: {len(webm_files)}")
    print(f"    unused by queries:    {len(unused_webm)}")
    print(f"  Files in {args.youtube_dir}: {len(youtube_files)}")
    print(f"    usable video files:   {len(youtube_stems)}")
    print(f"    unused by queries:    {len(unused_youtube)}")
    print()

    if webm_missing:
        print(f"MISSING WEBM ({len(webm_missing)})")
        print(f"  {format_list(webm_missing)}")
        print()

    if youtube_missing:
        print(f"MISSING YOUTUBE ({len(youtube_missing)})")
        print(f"  {format_list(youtube_missing)}")
        print()

    if youtube_missing_removed:
        print(
            "  of which known removed from the platform: "
            f"{len(youtube_missing_removed)}"
        )
        print(f"  {format_list(youtube_missing_removed)}")
        print()

    if case_mismatches:
        print(f"CASE-ONLY NAME MISMATCHES ({len(case_mismatches)})")
        print(f"  {format_list(case_mismatches)}")
        print()

    most_reused = reuse_counts.most_common(10)

    print("MOST REUSED VIDEOS (queries per video)")

    for video_id, count in most_reused:
        marker = "" if video_id not in missing_ids else "  [MISSING]"
        print(f"  {count:>3}  {video_id}{marker}")

    print()

    if args.report_json is not None:
        report = {
            "query_file": str(args.released_queries),
            "total_queries": total_queries,
            "distinct_video_ids": len(queries_by_video),
            "webm": {
                "directory": str(args.webm_dir),
                "referenced": len(webm_ids),
                "present": len(webm_present),
                "missing": webm_missing,
                "files_in_directory": len(webm_files),
                "unused_files": unused_webm,
            },
            "youtube": {
                "directory": str(args.youtube_dir),
                "referenced": len(youtube_ids),
                "present": len(youtube_present),
                "missing": youtube_missing,
                "missing_known_removed": youtube_missing_removed,
                "files_in_directory": len(youtube_files),
                "unused_files": unused_youtube,
            },
            "case_only_mismatches": case_mismatches,
            "queries_fully_covered": covered_queries,
            "queries_blocked": len(affected_q_ids),
            "blocked_q_ids": affected_q_ids,
            "queries_per_video": dict(reuse_counts.most_common()),
        }

        with args.report_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=4, ensure_ascii=False)
            handle.write("\n")

        print(f"JSON report written to: {args.report_json}")


if __name__ == "__main__":
    main()

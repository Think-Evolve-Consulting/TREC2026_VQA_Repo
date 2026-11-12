#!/usr/bin/env python3
"""Flatten the query-routing JSON into a spreadsheet-readable CSV.

``route.query.inputs.py`` writes one nested record per query, with the six
specialists' verdicts buried in a ``Signals`` object. That shape is right for
the answering pipeline and wrong for looking at: you cannot sort 2,443 rows by
route in a text editor, or pivot question type against required inputs.

This flattens it. One row per query, one column per routing decision:

    Q_ID, Video_ID, Question, Question_Type, Route, Required_Inputs,
    Temporal, Inference, Defaulted, Pipeline

``Required_Inputs`` and ``Pipeline`` are lists in the JSON; they are joined
with ``|`` rather than ``,`` so the CSV stays one field per column without
quoting games. ``Route`` is already the ``+``-joined summary of the same
inputs, kept because it is the column you actually sort on.

With ``--signals``, each specialist's confidence and reason are appended as
their own columns (vision_needed, vision_confidence, vision_reason, ...) --
18 extra columns, useful when auditing *why* something routed the way it did
and too wide to want by default.

Usage:
    python VQA/export.routing.csv.py                       # task1 -> query.routing.task1.csv
    python VQA/export.routing.csv.py --input query.routing.2025.json
    python VQA/export.routing.csv.py --all                 # every query.routing.*.json
    python VQA/export.routing.csv.py --all --signals       # with the agent verdicts
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE_COLUMNS = [
    "Q_ID",
    "Video_ID",
    "Question",
    "Question_Type",
    "Route",
    "Required_Inputs",
    "Temporal",
    "Inference",
    "Defaulted",
    "Pipeline",
]

# The six specialists, in the order route.query.inputs.py runs them. `qtype`
# reports a `type` instead of a `needed`, so it is handled separately below.
BOOL_AGENTS = ["vision", "audio", "ocr", "temporal", "inference"]


def signal_columns() -> list[str]:
    cols: list[str] = []
    for agent in BOOL_AGENTS:
        cols += [f"{agent}_needed", f"{agent}_confidence", f"{agent}_reason"]
    cols += ["qtype_type", "qtype_confidence", "qtype_reason"]
    return cols


def row_for(record: dict, with_signals: bool) -> dict:
    row = {
        "Q_ID": record.get("Q_ID"),
        "Video_ID": record.get("Video_ID"),
        "Question": record.get("Question"),
        "Question_Type": record.get("Question_Type"),
        "Route": record.get("Route"),
        "Required_Inputs": "|".join(record.get("Required_Inputs") or []),
        "Temporal": record.get("Temporal"),
        "Inference": record.get("Inference"),
        "Defaulted": record.get("Defaulted"),
        "Pipeline": "|".join(record.get("Pipeline") or []),
    }
    if not with_signals:
        return row

    signals = record.get("Signals") or {}
    for agent in BOOL_AGENTS:
        s = signals.get(agent) or {}
        row[f"{agent}_needed"] = s.get("needed")
        row[f"{agent}_confidence"] = s.get("confidence")
        row[f"{agent}_reason"] = s.get("reason")
    q = signals.get("qtype") or {}
    row["qtype_type"] = q.get("type")
    row["qtype_confidence"] = q.get("confidence")
    row["qtype_reason"] = q.get("reason")
    return row


def export(src: Path, dest: Path, with_signals: bool) -> int:
    records = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit(f"{src.name}: expected a list of routing records")

    columns = BASE_COLUMNS + (signal_columns() if with_signals else [])
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(row_for(record, with_signals))
    return len(records)


def summarise(src: Path) -> None:
    """Print the route distribution, so a bad export is obvious at a glance."""
    records = json.loads(src.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for r in records:
        counts[r.get("Route") or "?"] = counts.get(r.get("Route") or "?", 0) + 1
    total = len(records) or 1
    for route, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {route:<20} {n:>5}  {n / total:6.1%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--input",
        default="query.routing.task1.json",
        help="routing JSON to flatten (default: query.routing.task1.json)",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="flatten every query.routing.*.json in the project root",
    )
    ap.add_argument("--output", help="CSV path (default: input path with .csv)")
    ap.add_argument(
        "--signals",
        action="store_true",
        help="append each specialist's needed/confidence/reason columns",
    )
    args = ap.parse_args()

    if args.all:
        sources = sorted(ROOT.glob("query.routing.*.json"))
        if args.output:
            raise SystemExit("--output cannot be combined with --all")
    else:
        src = Path(args.input)
        if not src.is_absolute():
            src = ROOT / src
        sources = [src]

    if not sources:
        raise SystemExit("no routing JSON found")

    for src in sources:
        if not src.exists():
            raise SystemExit(f"missing: {src}")
        dest = Path(args.output) if args.output else src.with_suffix(".csv")
        if not dest.is_absolute():
            dest = ROOT / dest
        n = export(src, dest, args.signals)
        print(f"{src.name} -> {dest.name}  ({n} rows)")
        summarise(src)

    return 0


if __name__ == "__main__":
    sys.exit(main())

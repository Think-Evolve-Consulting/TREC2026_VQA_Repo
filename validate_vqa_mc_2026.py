#!/usr/bin/env python3
"""Validate a TRECVID VQA Multiple-Choice team submission.

Exit codes:
  0 = valid run
  1 = invalid team run
  2 = invalid/missing official query input or command failure
"""

### the script is called as follows: python3 validate_vqa_mc.py --queries aux/vqa.2026.testingDataset.Task2.json --run submission_run.json



from __future__ import annotations

import argparse
import difflib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXIT_VALID = 0
EXIT_INVALID_RUN = 1
EXIT_INPUT_FAILURE = 2


class DuplicateJSONKeyError(ValueError):
    pass


class ReferenceFileError(Exception):
    pass


@dataclass
class Issue:
    severity: str
    code: str
    location: str
    message: str


@dataclass(frozen=True)
class OfficialQuery:
    q_id: int
    video_id: str
    question: str
    options: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a VQA MCQ team run.")
    parser.add_argument("--queries", required=True, type=Path,
                        help="Official MCQ query JSON distributed to teams.")
    parser.add_argument("--run", required=True, type=Path,
                        help="Team submission JSON.")
    parser.add_argument("--report-json", type=Path,
                        help="Optional machine-readable validation report.")
    parser.add_argument("--allow-extra-fields", action="store_true",
                        help="Allow extra fields in team objects.")
    parser.add_argument("--allow-unsorted-answers", action="store_true",
                        help="Allow Answers entries to be out of rank order.")
    parser.add_argument("--warnings-as-errors", action="store_true",
                        help="Treat warnings as errors.")
    return parser.parse_args()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise DuplicateJSONKeyError(f"Duplicate JSON object key {key!r}.")
        obj[key] = value
    return obj


def load_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise OSError(f"{label} file not found: {path}")
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{label} is not valid JSON: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc


def is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def validate_reference(raw: Any, path: Path) -> dict[int, OfficialQuery]:
    if not isinstance(raw, list):
        raise ReferenceFileError(
            f"{path}: official query file must contain a top-level list."
        )

    result: dict[int, OfficialQuery] = {}
    for index, item in enumerate(raw):
        loc = f"queries[{index}]"
        if not isinstance(item, dict):
            raise ReferenceFileError(f"{loc}: expected a JSON object.")

        missing = [k for k in ("Q_ID", "Video_ID", "Question", "Options")
                   if k not in item]
        if missing:
            raise ReferenceFileError(
                f"{loc}: missing required fields: {', '.join(missing)}."
            )

        q_id = item["Q_ID"]
        if not is_integer(q_id):
            raise ReferenceFileError(f"{loc}.Q_ID must be an integer.")
        if q_id in result:
            raise ReferenceFileError(f"{loc}: duplicate official Q_ID {q_id}.")

        video_id = item["Video_ID"]
        question = item["Question"]
        options_raw = item["Options"]
        if not isinstance(video_id, str) or not video_id:
            raise ReferenceFileError(f"{loc}.Video_ID must be a non-empty string.")
        if not isinstance(question, str) or not question:
            raise ReferenceFileError(f"{loc}.Question must be a non-empty string.")
        if not isinstance(options_raw, list) or not options_raw:
            raise ReferenceFileError(f"{loc}.Options must be a non-empty list.")

        options: list[str] = []
        for option_index, option_obj in enumerate(options_raw):
            oloc = f"{loc}.Options[{option_index}]"
            if not isinstance(option_obj, dict) or "option" not in option_obj:
                raise ReferenceFileError(
                    f"{oloc} must be an object containing 'option'."
                )
            text = option_obj["option"]
            if not isinstance(text, str) or not text:
                raise ReferenceFileError(f"{oloc}.option must be a non-empty string.")
            options.append(text)

        if len(options) != len(set(options)):
            raise ReferenceFileError(f"{loc}.Options contains duplicate text.")
        normalized = [normalize(option) for option in options]
        if len(normalized) != len(set(normalized)):
            raise ReferenceFileError(
                f"{loc}.Options contains options differing only by case/whitespace."
            )

        result[q_id] = OfficialQuery(q_id, video_id, question, tuple(options))
    return result


def add(issues: list[Issue], severity: str, code: str,
        location: str, message: str) -> None:
    issues.append(Issue(severity, code, location, message))


def mismatch_message(answer: str, options: tuple[str, ...]) -> str:
    normalized_map = {normalize(option): option for option in options}
    if normalize(answer) in normalized_map:
        official = normalized_map[normalize(answer)]
        return (
            f"Answer must match the official option exactly. "
            f"Use {official!r}; received {answer!r}."
        )
    close = difflib.get_close_matches(answer, list(options), n=1, cutoff=0.72)
    if close:
        return (
            f"Answer {answer!r} is not an official option. "
            f"Closest official option: {close[0]!r}."
        )
    return f"Answer {answer!r} is not one of the official options: {list(options)!r}."


def validate_run(raw: Any, official: dict[int, OfficialQuery],
                 allow_extra_fields: bool,
                 allow_unsorted_answers: bool) -> tuple[list[Issue], dict[str, Any]]:
    issues: list[Issue] = []
    summary = {
        "official_query_count": len(official),
        "submitted_item_count": len(raw) if isinstance(raw, list) else None,
        "unique_submitted_qids": 0,
        "validated_query_items": 0,
    }

    if not isinstance(raw, list):
        add(issues, "error", "TOP_LEVEL_NOT_LIST", "$",
            "The team run must contain a top-level JSON list.")
        return issues, summary

    seen_qids: dict[int, int] = {}
    required_item_keys = {"Q_ID", "Video_ID", "Answers"}
    required_answer_keys = {"Rank", "Answer"}

    for item_index, item in enumerate(raw):
        loc = f"run[{item_index}]"
        if not isinstance(item, dict):
            add(issues, "error", "RUN_ITEM_NOT_OBJECT", loc,
                "Each run item must be a JSON object.")
            continue

        keys = set(item)
        missing = sorted(required_item_keys - keys)
        extra = sorted(keys - required_item_keys)
        if missing:
            add(issues, "error", "MISSING_QUERY_FIELDS", loc,
                f"Missing required fields: {', '.join(missing)}.")
        if extra:
            if allow_extra_fields:
                add(issues, "warning", "EXTRA_QUERY_FIELDS_ALLOWED", loc,
                    f"Ignored extra fields: {', '.join(extra)}.")
            else:
                add(issues, "error", "EXTRA_QUERY_FIELDS", loc,
                    f"Unexpected fields: {', '.join(extra)}.")

        if "Q_ID" not in item:
            continue
        q_id = item["Q_ID"]
        if not is_integer(q_id):
            add(issues, "error", "INVALID_Q_ID_TYPE", f"{loc}.Q_ID",
                f"Q_ID must be an integer; received {q_id!r}.")
            continue

        if q_id in seen_qids:
            add(issues, "error", "DUPLICATE_Q_ID", f"{loc}.Q_ID",
                f"Q_ID {q_id} already appeared at run[{seen_qids[q_id]}].")
        else:
            seen_qids[q_id] = item_index

        ref = official.get(q_id)
        if ref is None:
            add(issues, "error", "UNKNOWN_Q_ID", f"{loc}.Q_ID",
                f"Q_ID {q_id} is not in the official query file.")
            continue

        summary["validated_query_items"] += 1

        if "Video_ID" in item:
            video_id = item["Video_ID"]
            if not isinstance(video_id, str):
                add(issues, "error", "INVALID_VIDEO_ID_TYPE", f"{loc}.Video_ID",
                    "Video_ID must be a string.")
            elif video_id != ref.video_id:
                add(issues, "error", "VIDEO_ID_MISMATCH", f"{loc}.Video_ID",
                    f"For Q_ID {q_id}, expected {ref.video_id!r}; received {video_id!r}.")

        if "Answers" not in item:
            continue
        answers = item["Answers"]
        if not isinstance(answers, list):
            add(issues, "error", "ANSWERS_NOT_LIST", f"{loc}.Answers",
                "Answers must be a JSON list.")
            continue

        expected_count = len(ref.options)
        if len(answers) != expected_count:
            add(issues, "error", "WRONG_ANSWER_COUNT", f"{loc}.Answers",
                f"Expected {expected_count} answers; found {len(answers)}.")

        ranks: list[int] = []
        answer_texts: list[str] = []

        for answer_index, answer_item in enumerate(answers):
            aloc = f"{loc}.Answers[{answer_index}]"
            if not isinstance(answer_item, dict):
                add(issues, "error", "ANSWER_ITEM_NOT_OBJECT", aloc,
                    "Each answer entry must be a JSON object.")
                continue

            akeys = set(answer_item)
            missing_a = sorted(required_answer_keys - akeys)
            extra_a = sorted(akeys - required_answer_keys)
            if missing_a:
                add(issues, "error", "MISSING_ANSWER_FIELDS", aloc,
                    f"Missing required fields: {', '.join(missing_a)}.")
            if extra_a:
                if allow_extra_fields:
                    add(issues, "warning", "EXTRA_ANSWER_FIELDS_ALLOWED", aloc,
                        f"Ignored extra fields: {', '.join(extra_a)}.")
                else:
                    add(issues, "error", "EXTRA_ANSWER_FIELDS", aloc,
                        f"Unexpected fields: {', '.join(extra_a)}.")

            if "Rank" in answer_item:
                rank = answer_item["Rank"]
                if not is_integer(rank):
                    add(issues, "error", "INVALID_RANK_TYPE", f"{aloc}.Rank",
                        f"Rank must be an integer; received {rank!r}.")
                else:
                    ranks.append(rank)
                    if rank < 1 or rank > expected_count:
                        add(issues, "error", "RANK_OUT_OF_RANGE", f"{aloc}.Rank",
                            f"Rank must be between 1 and {expected_count}; received {rank}.")

            if "Answer" in answer_item:
                answer = answer_item["Answer"]
                if not isinstance(answer, str):
                    add(issues, "error", "INVALID_ANSWER_TYPE", f"{aloc}.Answer",
                        "Answer must be a string.")
                else:
                    answer_texts.append(answer)
                    if not answer:
                        add(issues, "error", "EMPTY_ANSWER", f"{aloc}.Answer",
                            "Answer must not be empty.")
                    elif answer not in ref.options:
                        add(issues, "error", "ANSWER_NOT_OFFICIAL_OPTION",
                            f"{aloc}.Answer", mismatch_message(answer, ref.options))

        expected_ranks = list(range(1, expected_count + 1))
        if len(ranks) != len(set(ranks)):
            duplicates = sorted(rank for rank in set(ranks) if ranks.count(rank) > 1)
            add(issues, "error", "DUPLICATE_RANK", f"{loc}.Answers",
                f"Duplicate rank values: {duplicates}.")

        if sorted(ranks) != expected_ranks:
            missing_ranks = sorted(set(expected_ranks) - set(ranks))
            invalid_ranks = sorted(set(ranks) - set(expected_ranks))
            parts = []
            if missing_ranks:
                parts.append(f"missing {missing_ranks}")
            if invalid_ranks:
                parts.append(f"invalid {invalid_ranks}")
            if len(ranks) != expected_count:
                parts.append(f"found {len(ranks)} integer rank fields")
            add(issues, "error", "INCOMPLETE_RANK_SET", f"{loc}.Answers",
                f"Ranks must be exactly {expected_ranks}; {'; '.join(parts)}.")

        if (not allow_unsorted_answers and ranks == sorted(ranks)
                and sorted(ranks) == expected_ranks):
            pass
        elif (not allow_unsorted_answers and len(ranks) == expected_count
              and sorted(ranks) == expected_ranks):
            add(issues, "error", "ANSWERS_NOT_SORTED_BY_RANK", f"{loc}.Answers",
                f"Answers must appear in rank order {expected_ranks}; received {ranks}.")

        if len(answer_texts) != len(set(answer_texts)):
            duplicates = sorted(answer for answer in set(answer_texts)
                                if answer_texts.count(answer) > 1)
            add(issues, "error", "DUPLICATE_ANSWER", f"{loc}.Answers",
                f"Duplicate answer text: {duplicates!r}.")

        submitted_official = {a for a in answer_texts if a in ref.options}
        missing_options = [option for option in ref.options
                           if option not in submitted_official]
        if missing_options:
            add(issues, "error", "MISSING_OFFICIAL_OPTIONS", f"{loc}.Answers",
                f"Official options not ranked: {missing_options!r}.")

    submitted_qids = set(seen_qids)
    official_qids = set(official)
    missing_qids = sorted(official_qids - submitted_qids)
    extra_qids = sorted(submitted_qids - official_qids)

    if missing_qids:
        add(issues, "error", "MISSING_QUERIES", "$",
            f"Missing {len(missing_qids)} official Q_IDs: {missing_qids}.")
    if extra_qids:
        add(issues, "error", "EXTRA_QUERIES", "$",
            f"Unknown submitted Q_IDs: {extra_qids}.")

    submitted_order: list[int] = []
    order_seen: set[int] = set()
    for item in raw:
        if isinstance(item, dict) and is_integer(item.get("Q_ID")):
            q_id = item["Q_ID"]
            if q_id not in order_seen:
                submitted_order.append(q_id)
                order_seen.add(q_id)
    if not missing_qids and not extra_qids and submitted_order != list(official):
        add(issues, "warning", "QUERY_ORDER_DIFFERS", "$",
            "All Q_IDs are present, but their order differs from the official file.")

    summary["unique_submitted_qids"] = len(seen_qids)
    return issues, summary


def print_report(run_path: Path, issues: list[Issue], summary: dict[str, Any],
                 warnings_as_errors: bool) -> bool:
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    invalid = bool(errors) or (warnings_as_errors and bool(warnings))

    print("=" * 72)
    print("TRECVID VQA MULTIPLE-CHOICE RUN VALIDATION")
    print("=" * 72)
    print(f"Run: {run_path}")
    print(f"Official queries: {summary['official_query_count']}")
    print(f"Submitted items: {summary['submitted_item_count']}")
    print(f"Unique submitted Q_IDs: {summary['unique_submitted_qids']}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print(f"STATUS: {'INVALID' if invalid else 'VALID'}")

    if issues:
        print("-" * 72)
        for number, issue in enumerate(issues, start=1):
            print(f"{number}. [{issue.severity.upper()}] {issue.code} at {issue.location}")
            print(f"   {issue.message}")
    else:
        print("-" * 72)
        print("No validation problems found.")
    return invalid


def write_report(path: Path, run_path: Path, query_path: Path,
                 issues: list[Issue], summary: dict[str, Any], invalid: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "INVALID" if invalid else "VALID",
        "run_file": str(run_path),
        "query_file": str(query_path),
        "summary": {
            **summary,
            "error_count": sum(i.severity == "error" for i in issues),
            "warning_count": sum(i.severity == "warning" for i in issues),
        },
        "issues": [asdict(i) for i in issues],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    try:
        if not args.queries.exists():
            args.queries = Path(sys.path[0]) / args.queries

        official_raw = load_json(args.queries, "Official query file")
        official = validate_reference(official_raw, args.queries)
    except (OSError, ValueError, DuplicateJSONKeyError, ReferenceFileError) as exc:
        print(f"Validator input error: {exc}", file=sys.stderr)
        return EXIT_INPUT_FAILURE

    try:
        run_raw = load_json(args.run, "Team run")
    except (OSError, ValueError, DuplicateJSONKeyError) as exc:
        issues = [Issue("error", "RUN_JSON_LOAD_FAILURE", "$", str(exc))]
        summary = {
            "official_query_count": len(official),
            "submitted_item_count": None,
            "unique_submitted_qids": 0,
            "validated_query_items": 0,
        }
        invalid = print_report(args.run, issues, summary, args.warnings_as_errors)
        if args.report_json:
            try:
                write_report(args.report_json, args.run, args.queries,
                             issues, summary, invalid)
            except OSError as report_exc:
                print(f"Could not write report: {report_exc}", file=sys.stderr)
                return EXIT_INPUT_FAILURE
        return EXIT_INVALID_RUN

    issues, summary = validate_run(
        run_raw,
        official,
        args.allow_extra_fields,
        args.allow_unsorted_answers,
    )
    invalid = print_report(args.run, issues, summary, args.warnings_as_errors)

    if args.report_json:
        try:
            write_report(args.report_json, args.run, args.queries,
                         issues, summary, invalid)
            print(f"JSON report written to: {args.report_json}")
        except OSError as exc:
            print(f"Could not write report: {exc}", file=sys.stderr)
            return EXIT_INPUT_FAILURE

    return EXIT_INVALID_RUN if invalid else EXIT_VALID


if __name__ == "__main__":
    raise SystemExit(main())

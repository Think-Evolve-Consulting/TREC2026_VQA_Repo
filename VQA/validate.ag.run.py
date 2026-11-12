#!/usr/bin/env python3

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_ANSWERS_PER_QUERY = 10

QUERY_REQUIRED_FIELDS = {
    "Q_ID",
    "Video_ID",
    "Question",
}

SUBMISSION_REQUIRED_FIELDS = {
    "Q_ID",
    "Video_ID",
    "Answers",
}

ANSWER_REQUIRED_FIELDS = {
    "Rank",
    "Answer",
    "Time",
}


class DuplicateJSONKeyError(ValueError):
    """Raised when a JSON object contains a duplicate key."""


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, path: str, message: str) -> None:
        self.errors.append(f"{path}: {message}")

    def add_warning(self, path: str, message: str) -> None:
        self.warnings.append(f"{path}: {message}")


def reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """
    JSON object-pairs hook that rejects duplicate keys.

    Standard json.load() silently keeps the last occurrence of a
    duplicate key. For validation, duplicate keys should be rejected.
    """
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(
                f"Duplicate JSON key found: {key!r}"
            )

        result[key] = value

    return result


def reject_nonstandard_number(value: str) -> None:
    """
    Reject nonstandard JSON numbers such as NaN and Infinity.
    """
    raise ValueError(
        f"Nonstandard JSON numeric value is not allowed: {value}"
    )


def load_json_file(path: Path) -> Any:
    """Load a JSON file using strict JSON parsing."""
    try:
        with path.open("r", encoding="utf-8-sig") as json_file:
            return json.load(
                json_file,
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_nonstandard_number,
            )

    except FileNotFoundError:
        raise ValueError(f"File does not exist: {path}")

    except PermissionError:
        raise ValueError(f"Permission denied while reading: {path}")

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path}, line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        )

    except DuplicateJSONKeyError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}")


def is_valid_integer(value: Any) -> bool:
    """
    Return True only for an actual integer.

    Boolean values are excluded because bool is a subclass of int
    in Python.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def is_valid_number(value: Any) -> bool:
    """
    Return True for a finite integer or floating-point number.

    Boolean, NaN, and infinite values are rejected.
    """
    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    return math.isfinite(value)


def check_object_fields(
    obj: dict[str, Any],
    required_fields: set[str],
    path: str,
    report: ValidationReport,
    reject_extra_fields: bool,
) -> None:
    """Check required fields and, optionally, unexpected fields."""
    missing_fields = sorted(required_fields - set(obj))

    if missing_fields:
        report.add_error(
            path,
            "missing required field(s): "
            + ", ".join(missing_fields),
        )

    if reject_extra_fields:
        extra_fields = sorted(set(obj) - required_fields)

        if extra_fields:
            report.add_error(
                path,
                "unexpected field(s): "
                + ", ".join(extra_fields),
            )


def validate_released_queries(
    data: Any,
    report: ValidationReport,
) -> dict[int, dict[str, str]]:
    """
    Validate the released testing-query file and return a lookup
    table indexed by Q_ID.
    """
    expected_queries: dict[int, dict[str, str]] = {}

    if not isinstance(data, list):
        report.add_error(
            "released_queries",
            "the top-level JSON value must be an array",
        )
        return expected_queries

    if not data:
        report.add_error(
            "released_queries",
            "the query array must not be empty",
        )
        return expected_queries

    for index, query in enumerate(data):
        path = f"released_queries[{index}]"

        if not isinstance(query, dict):
            report.add_error(
                path,
                "each query must be a JSON object",
            )
            continue

        check_object_fields(
            obj=query,
            required_fields=QUERY_REQUIRED_FIELDS,
            path=path,
            report=report,
            reject_extra_fields=False,
        )

        q_id = query.get("Q_ID")
        video_id = query.get("Video_ID")
        question = query.get("Question")

        q_id_is_valid = True
        video_id_is_valid = True
        question_is_valid = True

        if not is_valid_integer(q_id):
            report.add_error(
                f"{path}.Q_ID",
                "must be an integer",
            )
            q_id_is_valid = False

        elif q_id < 1:
            report.add_error(
                f"{path}.Q_ID",
                "must be greater than or equal to 1",
            )
            q_id_is_valid = False

        if not isinstance(video_id, str):
            report.add_error(
                f"{path}.Video_ID",
                "must be a string",
            )
            video_id_is_valid = False

        elif not video_id.strip():
            report.add_error(
                f"{path}.Video_ID",
                "must not be empty",
            )
            video_id_is_valid = False

        if not isinstance(question, str):
            report.add_error(
                f"{path}.Question",
                "must be a string",
            )
            question_is_valid = False

        elif not question.strip():
            report.add_error(
                f"{path}.Question",
                "must not be empty",
            )
            question_is_valid = False

        if not q_id_is_valid:
            continue

        if q_id in expected_queries:
            report.add_error(
                f"{path}.Q_ID",
                f"duplicate Q_ID {q_id}",
            )
            continue

        if video_id_is_valid and question_is_valid:
            expected_queries[q_id] = {
                "Video_ID": video_id,
                "Question": question,
            }

    return expected_queries


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for duplicate detection.

    This makes capitalization and repeated whitespace irrelevant.
    """
    return " ".join(answer.split()).casefold()


def validate_answers(
    answers: Any,
    query_path: str,
    answers_per_query: int,
    reject_extra_fields: bool,
    report: ValidationReport,
) -> None:
    """Validate the ranked-answer list for one query."""
    answers_path = f"{query_path}.Answers"

    if not isinstance(answers, list):
        report.add_error(
            answers_path,
            "must be an array",
        )
        return

    if len(answers) != answers_per_query:
        report.add_error(
            answers_path,
            f"must contain exactly {answers_per_query} answer "
            f"objects; found {len(answers)}",
        )

    seen_ranks: dict[int, int] = {}
    ranks_in_array_order: list[int] = []
    seen_answers: dict[str, tuple[int, Any]] = {}

    for answer_index, answer_object in enumerate(answers):
        answer_path = f"{answers_path}[{answer_index}]"

        if not isinstance(answer_object, dict):
            report.add_error(
                answer_path,
                "each answer must be a JSON object",
            )
            continue

        if not answer_object:
            report.add_error(
                answer_path,
                "empty answer objects are not allowed",
            )
            continue

        check_object_fields(
            obj=answer_object,
            required_fields=ANSWER_REQUIRED_FIELDS,
            path=answer_path,
            report=report,
            reject_extra_fields=reject_extra_fields,
        )

        rank = answer_object.get("Rank")
        answer_text = answer_object.get("Answer")
        processing_time = answer_object.get("Time")

        rank_is_valid = True

        if not is_valid_integer(rank):
            report.add_error(
                f"{answer_path}.Rank",
                "must be an integer",
            )
            rank_is_valid = False

        elif not 1 <= rank <= answers_per_query:
            report.add_error(
                f"{answer_path}.Rank",
                f"must be between 1 and {answers_per_query}",
            )
            rank_is_valid = False

        if rank_is_valid:
            ranks_in_array_order.append(rank)

            if rank in seen_ranks:
                first_index = seen_ranks[rank]

                report.add_error(
                    f"{answer_path}.Rank",
                    f"duplicate rank {rank}; it was already used in "
                    f"{answers_path}[{first_index}]",
                )
            else:
                seen_ranks[rank] = answer_index

        if not isinstance(answer_text, str):
            report.add_error(
                f"{answer_path}.Answer",
                "must be a string",
            )

        elif not answer_text.strip():
            report.add_error(
                f"{answer_path}.Answer",
                "must not be empty or contain only whitespace",
            )

        else:
            if answer_text != answer_text.strip():
                report.add_warning(
                    f"{answer_path}.Answer",
                    "contains leading or trailing whitespace",
                )

            normalized = normalize_answer(answer_text)

            if normalized in seen_answers:
                first_index, first_rank = seen_answers[normalized]

                report.add_error(
                    f"{answer_path}.Answer",
                    "duplicate answer text; the same normalized answer "
                    f"appears in {answers_path}[{first_index}] "
                    f"with rank {first_rank}",
                )
            else:
                seen_answers[normalized] = (
                    answer_index,
                    rank,
                )

        if not is_valid_number(processing_time):
            report.add_error(
                f"{answer_path}.Time",
                "must be a finite numeric value",
            )

        elif processing_time < 0:
            report.add_error(
                f"{answer_path}.Time",
                "must be greater than or equal to 0",
            )

    expected_ranks = set(range(1, answers_per_query + 1))
    submitted_ranks = set(seen_ranks)
    missing_ranks = sorted(expected_ranks - submitted_ranks)

    if missing_ranks:
        report.add_error(
            answers_path,
            "missing rank(s): "
            + ", ".join(str(rank) for rank in missing_ranks),
        )

    if (
        len(answers) == answers_per_query
        and submitted_ranks == expected_ranks
        and ranks_in_array_order
        != list(range(1, answers_per_query + 1))
    ):
        report.add_warning(
            answers_path,
            "answer objects are not arranged in ascending Rank order",
        )


def format_id_preview(
    ids: list[int],
    maximum_displayed: int = 30,
) -> str:
    """Format a possibly long list of query IDs."""
    displayed = ids[:maximum_displayed]
    text = ", ".join(str(q_id) for q_id in displayed)

    if len(ids) > maximum_displayed:
        text += f", ... and {len(ids) - maximum_displayed} more"

    return text


def validate_submission(
    data: Any,
    expected_queries: dict[int, dict[str, str]],
    answers_per_query: int,
    reject_extra_fields: bool,
    report: ValidationReport,
) -> int:
    """
    Validate a submitted run against the released queries.

    Returns the number of top-level submission entries.
    """
    if not isinstance(data, list):
        report.add_error(
            "submission",
            "the top-level JSON value must be an array",
        )
        return 0

    submitted_q_ids: dict[int, int] = {}

    for index, submission in enumerate(data):
        path = f"submission[{index}]"

        if not isinstance(submission, dict):
            report.add_error(
                path,
                "each submission entry must be a JSON object",
            )
            continue

        check_object_fields(
            obj=submission,
            required_fields=SUBMISSION_REQUIRED_FIELDS,
            path=path,
            report=report,
            reject_extra_fields=reject_extra_fields,
        )

        q_id = submission.get("Q_ID")
        video_id = submission.get("Video_ID")
        answers = submission.get("Answers")

        q_id_is_valid = True

        if not is_valid_integer(q_id):
            report.add_error(
                f"{path}.Q_ID",
                "must be an integer",
            )
            q_id_is_valid = False

        elif q_id < 1:
            report.add_error(
                f"{path}.Q_ID",
                "must be greater than or equal to 1",
            )
            q_id_is_valid = False

        if not isinstance(video_id, str):
            report.add_error(
                f"{path}.Video_ID",
                "must be a string",
            )

        elif not video_id.strip():
            report.add_error(
                f"{path}.Video_ID",
                "must not be empty",
            )

        if q_id_is_valid:
            if q_id in submitted_q_ids:
                first_index = submitted_q_ids[q_id]

                report.add_error(
                    f"{path}.Q_ID",
                    f"duplicate Q_ID {q_id}; it was already submitted "
                    f"in submission[{first_index}]",
                )
            else:
                submitted_q_ids[q_id] = index

            if q_id not in expected_queries:
                report.add_error(
                    f"{path}.Q_ID",
                    f"Q_ID {q_id} does not exist in the released "
                    "testing queries",
                )

            elif isinstance(video_id, str):
                expected_video_id = expected_queries[q_id]["Video_ID"]

                if video_id != expected_video_id:
                    report.add_error(
                        f"{path}.Video_ID",
                        f"does not match Q_ID {q_id}; expected "
                        f"{expected_video_id!r}, found {video_id!r}",
                    )

        validate_answers(
            answers=answers,
            query_path=path,
            answers_per_query=answers_per_query,
            reject_extra_fields=reject_extra_fields,
            report=report,
        )

    expected_q_ids = set(expected_queries)
    actual_q_ids = set(submitted_q_ids)

    missing_q_ids = sorted(expected_q_ids - actual_q_ids)

    if missing_q_ids:
        report.add_error(
            "submission",
            f"missing {len(missing_q_ids)} required query or queries: "
            f"{format_id_preview(missing_q_ids)}",
        )

    return len(data)


def write_json_report(
    output_path: Path,
    valid: bool,
    expected_query_count: int,
    submitted_entry_count: int,
    answers_per_query: int,
    report: ValidationReport,
) -> None:
    """Write the validation results to a machine-readable JSON file."""
    report_data = {
        "valid": valid,
        "expected_query_count": expected_query_count,
        "submitted_entry_count": submitted_entry_count,
        "answers_per_query": answers_per_query,
        "error_count": len(report.errors),
        "warning_count": len(report.warnings),
        "errors": report.errors,
        "warnings": report.warnings,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            report_data,
            output_file,
            indent=4,
            ensure_ascii=False,
        )
        output_file.write("\n")


def print_report(
    valid: bool,
    expected_query_count: int,
    submitted_entry_count: int,
    report: ValidationReport,
    maximum_messages: int,
) -> None:
    """Print a readable validation report."""
    print("=" * 72)

    if valid:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")

    print("=" * 72)
    print(f"Expected queries:    {expected_query_count}")
    print(f"Submitted entries:   {submitted_entry_count}")
    print(f"Errors:              {len(report.errors)}")
    print(f"Warnings:            {len(report.warnings)}")

    if report.errors:
        print("\nERRORS")

        for message in report.errors[:maximum_messages]:
            print(f"  ERROR: {message}")

        hidden = len(report.errors) - maximum_messages

        if hidden > 0:
            print(f"  ... {hidden} additional error(s) not displayed")

    if report.warnings:
        print("\nWARNINGS")

        for message in report.warnings[:maximum_messages]:
            print(f"  WARNING: {message}")

        hidden = len(report.warnings) - maximum_messages

        if hidden > 0:
            print(f"  ... {hidden} additional warning(s) not displayed")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a VQA answer-generation run against the "
            "released testing queries."
        )
    )

    parser.add_argument(
        "released_queries",
        type=Path,
        help=(
            "JSON file containing the released Q_ID, Video_ID, "
            "and Question values."
        ),
    )

    parser.add_argument(
        "submission",
        type=Path,
        help="JSON run-submission file to validate.",
    )

    parser.add_argument(
        "--answers-per-query",
        type=int,
        default=DEFAULT_ANSWERS_PER_QUERY,
        help=(
            "Required number of ranked answers per query "
            f"(default: {DEFAULT_ANSWERS_PER_QUERY})."
        ),
    )

    parser.add_argument(
        "--allow-extra-fields",
        action="store_true",
        help=(
            "Allow additional fields in submission and answer objects. "
            "By default, unexpected fields are errors."
        ),
    )

    parser.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="Make warnings cause validation failure.",
    )

    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help=(
            "Optional path for a machine-readable JSON validation report."
        ),
    )

    parser.add_argument(
        "--max-messages",
        type=int,
        default=100,
        help=(
            "Maximum number of errors and warnings displayed in the "
            "terminal for each category (default: 100)."
        ),
    )

    args = parser.parse_args()

    if args.answers_per_query < 1:
        parser.error("--answers-per-query must be at least 1")

    if args.max_messages < 1:
        parser.error("--max-messages must be at least 1")

    try:
        released_data = load_json_file(args.released_queries)
        submission_data = load_json_file(args.submission)

    except (ValueError, OSError) as error:
        print(f"Fatal error: {error}", file=sys.stderr)
        sys.exit(2)

    released_report = ValidationReport()

    expected_queries = validate_released_queries(
        data=released_data,
        report=released_report,
    )

    if released_report.errors:
        print(
            "The released testing-query file is invalid. "
            "The submission cannot be validated.",
            file=sys.stderr,
        )

        print_report(
            valid=False,
            expected_query_count=len(expected_queries),
            submitted_entry_count=0,
            report=released_report,
            maximum_messages=args.max_messages,
        )

        sys.exit(2)

    submission_report = ValidationReport()

    submitted_entry_count = validate_submission(
        data=submission_data,
        expected_queries=expected_queries,
        answers_per_query=args.answers_per_query,
        reject_extra_fields=not args.allow_extra_fields,
        report=submission_report,
    )

    valid = not submission_report.errors

    if args.warnings_as_errors and submission_report.warnings:
        valid = False

    print_report(
        valid=valid,
        expected_query_count=len(expected_queries),
        submitted_entry_count=submitted_entry_count,
        report=submission_report,
        maximum_messages=args.max_messages,
    )

    if args.report_json is not None:
        try:
            write_json_report(
                output_path=args.report_json,
                valid=valid,
                expected_query_count=len(expected_queries),
                submitted_entry_count=submitted_entry_count,
                answers_per_query=args.answers_per_query,
                report=submission_report,
            )

            print(f"JSON report written to: {args.report_json}")

        except OSError as error:
            print(
                f"Could not write JSON report: {error}",
                file=sys.stderr,
            )
            sys.exit(2)

    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()

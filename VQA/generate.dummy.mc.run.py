#!/usr/bin/env python3
"""
Generate a random dummy Task 2 (Multiple Choice) run file.

The output conforms to the format enforced by validate_vqa_mc_2026.py:

    [
        {
            "Q_ID": 1,
            "Video_ID": "8384.webm",
            "Answers": [
                {"Rank": 1, "Answer": "It fades"},
                ...
                {"Rank": 8, "Answer": "It becomes clearer"}
            ]
        },
        ...
    ]

Unlike the Task 1 dummy run, the answer strings cannot be invented: the
validator requires every entry to match one of that query's official options
byte for byte, and requires all of them to be ranked exactly once. A dummy run
is therefore a random permutation of the official options -- the ordering
carries no signal about the videos.

Two jobs:

**Format check.** Run it against a released query file to confirm a submission
pipeline produces something the validator accepts before spending GPU hours on
a real run.

**Scoring control.** Run it against a labelled sample (see VQA/eval.mc.dev.py)
to confirm the scoring harness reports chance -- 25% Top-1 and ~0.52 MRR on
four options, 12.5% and ~0.34 on eight. A harness that scores random
permutations above chance is broken, and every number it produces afterwards
is worthless.
"""

import argparse
import json
import random
from pathlib import Path


def normalize(text: str) -> str:
    """Same normalization the validator uses when it compares options."""
    return " ".join(text.split()).casefold()


def extract_options(query: dict, loc: str) -> list[str]:
    """Pull the option strings out of one query, rejecting what the validator would."""
    options_raw = query.get("Options")

    if not isinstance(options_raw, list) or not options_raw:
        raise ValueError(f"{loc}.Options must be a non-empty list.")

    options: list[str] = []

    for index, option_obj in enumerate(options_raw):
        if not isinstance(option_obj, dict) or "option" not in option_obj:
            raise ValueError(f"{loc}.Options[{index}] must be an object containing 'option'.")

        text = option_obj["option"]

        if not isinstance(text, str) or not text:
            raise ValueError(f"{loc}.Options[{index}].option must be a non-empty string.")

        options.append(text)

    normalized = [normalize(option) for option in options]

    if len(normalized) != len(set(normalized)):
        raise ValueError(
            f"{loc}.Options contains duplicate (or case/whitespace-only different) text."
        )

    return options


def build_run(queries: list[dict], order: str, rng: random.Random) -> list[dict]:
    run: list[dict] = []

    for index, query in enumerate(queries):
        loc = f"queries[{index}]"

        for field in ("Q_ID", "Video_ID", "Options"):
            if field not in query:
                raise ValueError(f"{loc}: missing required field {field}.")

        options = extract_options(query, loc)

        if order == "shuffled":
            ranked = options[:]
            rng.shuffle(ranked)
        elif order == "reversed":
            ranked = options[::-1]
        else:
            ranked = options[:]

        answers = [
            {"Rank": rank, "Answer": text}
            for rank, text in enumerate(ranked, start=1)
        ]

        run.append(
            {
                "Q_ID": query["Q_ID"],
                "Video_ID": query["Video_ID"],
                "Answers": answers,
            }
        )

    return run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dummy VQA Task 2 (multiple choice) run file from the "
            "released testing queries."
        )
    )

    parser.add_argument(
        "released_queries",
        type=Path,
        help="JSON file with the released Q_ID, Video_ID, Question and Options values.",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Path of the run file to write.",
    )

    parser.add_argument(
        "--order",
        choices=("shuffled", "as-is", "reversed"),
        default="shuffled",
        help=(
            "How to rank each query's options: a random permutation (default), "
            "the official order, or the official order reversed."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducible output (default: 2026).",
    )

    args = parser.parse_args()

    with args.released_queries.open("r", encoding="utf-8-sig") as handle:
        queries = json.load(handle)

    if not isinstance(queries, list):
        raise SystemExit("Released query file must contain a top-level list.")

    rng = random.Random(args.seed)

    run = build_run(queries=queries, order=args.order, rng=rng)

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(run, handle, indent=4, ensure_ascii=False)
        handle.write("\n")

    total_answers = sum(len(item["Answers"]) for item in run)

    print(
        f"Wrote {len(run)} query entries ({total_answers} ranked answers, "
        f"order={args.order}) to {args.output}"
    )


if __name__ == "__main__":
    main()

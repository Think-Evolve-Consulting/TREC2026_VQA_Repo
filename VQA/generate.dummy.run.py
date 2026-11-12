#!/usr/bin/env python3
"""
Generate a random dummy Task 1 (Answer Generation) run file.

The output conforms to the format enforced by validate.ag.run.py:

    [
        {
            "Q_ID": 1,
            "Video_ID": "8384.webm",
            "Answers": [
                {"Rank": 1, "Answer": "...", "Time": 1.234},
                ...
                {"Rank": 10, "Answer": "...", "Time": 0.987}
            ]
        },
        ...
    ]

Answers are placeholder text only; they carry no signal about the videos.
"""

import argparse
import json
import random
from pathlib import Path


DEFAULT_ANSWERS_PER_QUERY = 10

SUBJECTS = [
    "a man",
    "a woman",
    "a child",
    "two people",
    "a group of people",
    "an animal",
    "a car",
    "the camera",
    "a crowd",
    "the presenter",
    "a dog",
    "the driver",
]

ACTIONS = [
    "walks across the scene",
    "talks to the camera",
    "picks up an object",
    "turns around",
    "sits down at a table",
    "runs out of the frame",
    "points at something off screen",
    "opens a door",
    "holds up a sign",
    "waves at someone",
    "drives past the building",
    "starts dancing",
]

MODIFIERS = [
    "at the start of the video",
    "in the middle of the clip",
    "near the end of the video",
    "in the background",
    "in the foreground",
    "twice",
    "slowly",
    "while music plays",
    "outdoors",
    "indoors",
]

SHORT_ANSWERS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "yes",
    "no",
    "unknown",
    "nothing happens",
    "it is not shown",
]


def build_answer_pool(rng: random.Random, count: int) -> list[str]:
    """Build `count` distinct placeholder answer strings."""
    answers: list[str] = []
    seen: set[str] = set()

    while len(answers) < count:
        if rng.random() < 0.25:
            text = rng.choice(SHORT_ANSWERS)
        else:
            text = (
                f"{rng.choice(SUBJECTS)} "
                f"{rng.choice(ACTIONS)} "
                f"{rng.choice(MODIFIERS)}"
            )

        normalized = " ".join(text.split()).casefold()

        if normalized in seen:
            continue

        seen.add(normalized)
        answers.append(text)

    return answers


def build_run(
    queries: list[dict],
    answers_per_query: int,
    rng: random.Random,
) -> list[dict]:
    run: list[dict] = []

    for query in queries:
        answer_texts = build_answer_pool(rng, answers_per_query)

        answers = [
            {
                "Rank": rank,
                "Answer": answer_texts[rank - 1],
                "Time": round(rng.uniform(0.5, 25.0), 3),
            }
            for rank in range(1, answers_per_query + 1)
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
            "Generate a random dummy VQA Task 1 run file from the "
            "released testing queries."
        )
    )

    parser.add_argument(
        "released_queries",
        type=Path,
        help="JSON file with the released Q_ID, Video_ID and Question values.",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Path of the run file to write.",
    )

    parser.add_argument(
        "--answers-per-query",
        type=int,
        default=DEFAULT_ANSWERS_PER_QUERY,
        help=(
            "Number of ranked answers generated per query "
            f"(default: {DEFAULT_ANSWERS_PER_QUERY})."
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

    rng = random.Random(args.seed)

    run = build_run(
        queries=queries,
        answers_per_query=args.answers_per_query,
        rng=rng,
    )

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(run, handle, indent=4, ensure_ascii=False)
        handle.write("\n")

    print(
        f"Wrote {len(run)} query entries "
        f"({args.answers_per_query} answers each) to {args.output}"
    )


if __name__ == "__main__":
    main()
